//! Integration tests for the cost-reporting ledger aggregate SQL
//! (spec: billing-grade-cost-reporting, Task 2).
//!
//! Seeds isolated logs.db ledgers through the real `RequestLogger` write
//! path and asserts the query layer's row shapes for each grouping,
//! empty-day zero rows, unattributed attribution, unpriced counting,
//! conjunctive filters, `#attempt` exclusion, and the sum-equality
//! invariant across groupings (Requirement 2.3).

use crate::common;

use ai_gateway::cost_reporting::{
    query::fetch_report, GroupBy, ReportRow, ValidatedCostReportParams, ValidatedRange,
};
use ai_gateway::logger::RequestLogger;
use common::{
    cost_reports_logging_config, sample_log_entry, LogEntryOverrides, DAY_ZERO,
    LEGACY_REQUESTS_DDL, LEGACY_REQUESTS_MINIMAL_DDL,
};

struct LedgerFixture {
    _logger: RequestLogger,
    _dir: tempfile::TempDir,
    db_path: String,
}

fn create_ledger() -> LedgerFixture {
    let dir = tempfile::tempdir().expect("temp dir for isolated logs.db");
    let db_path = dir.path().join("logs.db").to_string_lossy().into_owned();
    let logger = RequestLogger::new(cost_reports_logging_config(db_path.clone()))
        .expect("create isolated request logger");
    // Field order matters: `_logger` (declared first) drops before `_dir`,
    // so the writer thread's SQLite handle is closed before the TempDir
    // cleanup deletes the database file (Windows delete-failure guard).
    LedgerFixture {
        _logger: logger,
        _dir: dir,
        db_path,
    }
}

/// Seed one row through the real logger write path with the fixture's
/// baseline (all-None token/attribution columns) plus explicit overrides.
#[allow(clippy::too_many_arguments)]
fn seed_row(
    logger: &RequestLogger,
    timestamp: i64,
    provider: &str,
    model: &str,
    cost: f64,
    cache_read: Option<i64>,
    cache_creation: Option<i64>,
    reasoning: Option<u32>,
    path: &str,
) {
    let entry = sample_log_entry(
        &format!("trace-{timestamp}-{provider}-{model}-{cost}"),
        timestamp,
        provider,
        model,
        cost,
        LogEntryOverrides {
            path: Some(path.to_owned()),
            cache_read_tokens: cache_read,
            cache_creation_tokens: cache_creation,
            reasoning_tokens: reasoning,
            ..Default::default()
        },
    );
    logger
        .log(entry)
        .expect("seed log entry");
    // `log` is asynchronous (dedicated writer thread); the FIFO
    // checkpoint round-trip guarantees the row is durable before the
    // test's read connection queries it — without it readers race the
    // writer and intermittently see zero rows.
    logger.flush();
}

fn open_conn(fixture: &LedgerFixture) -> rusqlite::Connection {
    rusqlite::Connection::open(&fixture.db_path).expect("open seeded logs.db read-only")
}

fn validated(
    start_secs: i64,
    end_secs: i64,
    group_by: GroupBy,
    provider: Option<&str>,
    key_id: Option<&str>,
) -> ValidatedCostReportParams {
    ValidatedCostReportParams {
        range: ValidatedRange {
            start_secs,
            end_secs,
        },
        group_by,
        provider: provider.map(str::to_string),
        key_id: key_id.map(str::to_string),
        format: ai_gateway::cost_reporting::ReportFormat::Json,
    }
}

fn total_cost_micro(rows: &[ReportRow]) -> i64 {
    rows.iter().map(|row| row.cost_micro_usd).sum()
}

/// Three days of mixed-provider/model traffic with unpriced rows and an
/// `#attempt` marker row; day 2 is empty.
fn seed_standard(fixture: &LedgerFixture) {
    let logger = &fixture._logger;
    // Day 1: openai/gpt-4 priced, openai/gpt-4 unpriced, anthropic/claude priced.
    seed_row(logger, DAY_ZERO + 3_600, "openai", "gpt-4", 0.001_234, Some(100), None, None, "/v1/chat/completions");
    seed_row(logger, DAY_ZERO + 7_200, "openai", "gpt-4", 0.0, Some(500), None, None, "/v1/chat/completions");
    seed_row(logger, DAY_ZERO + 10_800, "anthropic", "claude-3", 0.002_468, None, Some(200), Some(64), "/v1/chat/completions");
    // Day 2: intentionally empty.
    // Day 3: openai/gpt-4 priced + an #attempt marker row that must be excluded.
    seed_row(logger, DAY_ZERO + 2 * 86_400 + 3_600, "openai", "gpt-4", 0.000_500, None, None, None, "/v1/chat/completions");
    seed_row(logger, DAY_ZERO + 2 * 86_400 + 4_000, "openai", "gpt-4", 0.0, None, None, None, "/v1/chat/completions#attempt");
}

#[test]
fn day_grouping_returns_spine_with_empty_day_zero_rows() {
    let fixture = create_ledger();
    seed_standard(&fixture);
    let conn = open_conn(&fixture);

    let rows = fetch_report(
        &conn,
        &validated(DAY_ZERO, DAY_ZERO + 3 * 86_400, GroupBy::Day, None, None),
    )
    .expect("day report");

    assert_eq!(
        rows.iter().map(|row| row.dimension.as_str()).collect::<Vec<_>>(),
        vec!["2026-01-01", "2026-01-02", "2026-01-03"],
        "one row per UTC day in range, calendar order, empty day included"
    );

    let empty = &rows[1];
    assert_eq!((empty.requests, empty.cost_micro_usd, empty.input_tokens, empty.cost_unpriced_requests), (0, 0, 0, 0));
    assert_eq!(
        (empty.window_start, empty.window_end),
        (DAY_ZERO + 86_400, DAY_ZERO + 2 * 86_400),
        "empty day carries its own UTC day window bounds"
    );

    let day1 = &rows[0];
    assert_eq!(day1.requests, 3, "#attempt row excluded, unpriced row counted");
    assert_eq!(day1.cache_read_tokens, 600);
    assert_eq!(day1.cache_creation_tokens, 200);
    assert_eq!(day1.reasoning_tokens, 64);
    assert_eq!(day1.cost_unpriced_requests, 1, "cost=0 with cache tokens counts as unpriced");
    assert_eq!(day1.cost_micro_usd, 1_234 + 2_468);

    let day3 = &rows[2];
    assert_eq!(day3.requests, 1, "the #attempt marker row is excluded from day counts");
    assert_eq!(day3.cost_unpriced_requests, 0, "attempt rows never leak into unpriced counting");
}

#[test]
fn day_grouping_end_midnight_excludes_trailing_day() {
    let fixture = create_ledger();
    seed_row(&fixture._logger, DAY_ZERO + 3_600, "openai", "gpt-4", 0.001, None, None, None, "/v1/chat/completions");
    let conn = open_conn(&fixture);

    let rows = fetch_report(
        &conn,
        &validated(DAY_ZERO, DAY_ZERO + 86_400, GroupBy::Day, None, None),
    )
    .expect("day report with midnight end");

    assert_eq!(rows.len(), 1);
    assert_eq!(rows[0].dimension, "2026-01-01");
    assert_eq!((rows[0].window_start, rows[0].window_end), (DAY_ZERO, DAY_ZERO + 86_400));
}

#[test]
fn provider_grouping_orders_cost_desc_with_dimension_asc_ties() {
    let fixture = create_ledger();
    seed_standard(&fixture);
    let conn = open_conn(&fixture);

    let mut rows = fetch_report(
        &conn,
        &validated(DAY_ZERO, DAY_ZERO + 3 * 86_400, GroupBy::Provider, None, None),
    )
    .expect("provider report");

    ai_gateway::cost_reporting::order_rows(GroupBy::Provider, &mut rows);
    let dimensions: Vec<&str> = rows.iter().map(|row| row.dimension.as_str()).collect();
    assert_eq!(dimensions, vec!["anthropic", "openai"], "cost desc ordering");

    let anthropic = &rows[0];
    assert_eq!(anthropic.cost_micro_usd, 2_468);
    assert_eq!(anthropic.requests, 1);
    let openai = &rows[1];
    assert_eq!(openai.requests, 3, "#attempt row excluded (2 on day 1 + 1 on day 3)");
    assert_eq!(openai.cost_unpriced_requests, 1);
}

#[test]
fn model_grouping_returns_one_row_per_distinct_model() {
    let fixture = create_ledger();
    seed_standard(&fixture);
    let conn = open_conn(&fixture);

    let rows = fetch_report(
        &conn,
        &validated(DAY_ZERO, DAY_ZERO + 3 * 86_400, GroupBy::Model, None, None),
    )
    .expect("model report");

    let mut dimensions: Vec<&str> = rows.iter().map(|row| row.dimension.as_str()).collect();
    dimensions.sort_unstable();
    assert_eq!(dimensions, vec!["claude-3", "gpt-4"]);
}

#[test]
fn virtual_key_grouping_buckets_all_traffic_as_unattributed_without_key_column() {
    let fixture = create_ledger();
    seed_standard(&fixture);
    let conn = open_conn(&fixture);

    let rows = fetch_report(
        &conn,
        &validated(DAY_ZERO, DAY_ZERO + 3 * 86_400, GroupBy::VirtualKey, None, None),
    )
    .expect("virtual key report");

    assert_eq!(rows.len(), 1, "ledger without key attribution collapses to one bucket");
    assert_eq!(rows[0].dimension, ai_gateway::cost_reporting::query::UNATTRIBUTED);
    assert_eq!(rows[0].requests, 4);
}

#[test]
fn key_id_unattributed_filter_matches_null_and_blank_rows() {
    let fixture = create_ledger();
    seed_standard(&fixture);
    let conn = open_conn(&fixture);

    // The key_id column exists (logger migration) but every row is NULL —
    // the grouping labels them `unattributed`, so the same filter value
    // must match them via the COALESCE expression, not raw column equality.
    let rows = fetch_report(
        &conn,
        &validated(
            DAY_ZERO,
            DAY_ZERO + 3 * 86_400,
            GroupBy::VirtualKey,
            None,
            Some(ai_gateway::cost_reporting::query::UNATTRIBUTED),
        ),
    )
    .expect("unattributed-filtered report");

    assert_eq!(rows.len(), 1);
    assert_eq!(rows[0].dimension, ai_gateway::cost_reporting::query::UNATTRIBUTED);
    assert_eq!(rows[0].requests, 4, "NULL-key rows match the unattributed filter");
}

#[test]
fn key_id_filter_still_matches_real_key_ids() {
    let fixture = create_ledger();
    let logger = &fixture._logger;
    // One attributed row (vk-42) plus one NULL-key row: the filter must
    // narrow to the attributed bucket only.
    let keyed_entry = sample_log_entry(
        &format!("trace-{}-openai-gpt-4-0.001", DAY_ZERO + 3_600),
        DAY_ZERO + 3_600,
        "openai",
        "gpt-4",
        0.001,
        LogEntryOverrides {
            virtual_key_id: Some("vk-42".to_string()),
            ..Default::default()
        },
    );
    logger.log(keyed_entry).expect("seed keyed log entry");
    logger.flush();
    seed_row(logger, DAY_ZERO + 7_200, "openai", "gpt-4", 0.001, None, None, None, "/v1/chat/completions");
    let conn = open_conn(&fixture);

    let rows = fetch_report(
        &conn,
        &validated(
            DAY_ZERO,
            DAY_ZERO + 86_400,
            GroupBy::VirtualKey,
            None,
            Some("vk-42"),
        ),
    )
    .expect("key-filtered report");

    assert_eq!(rows.len(), 1, "filter matches only the requested key bucket");
    assert_eq!(rows[0].dimension, "vk-42");
    assert_eq!(rows[0].requests, 1);
}

#[test]
fn provider_filter_narrows_report_and_counts_unpriced() {
    let fixture = create_ledger();
    seed_standard(&fixture);
    let conn = open_conn(&fixture);

    let rows = fetch_report(
        &conn,
        &validated(DAY_ZERO, DAY_ZERO + 3 * 86_400, GroupBy::Day, Some("openai"), None),
    )
    .expect("provider-filtered report");

    assert_eq!(rows.len(), 3);
    assert_eq!(rows[0].requests, 2);
    assert_eq!(rows[0].cost_unpriced_requests, 1);
    assert_eq!(rows[2].requests, 1);
    assert_eq!(rows[1].requests, 0, "empty day survives the filter");
}

#[test]
fn conjunctive_filters_intersect() {
    let fixture = create_ledger();
    seed_standard(&fixture);
    let conn = open_conn(&fixture);

    let rows = fetch_report(
        &conn,
        &validated(
            DAY_ZERO,
            DAY_ZERO + 3 * 86_400,
            GroupBy::Provider,
            Some("anthropic"),
            Some("no-such-key"),
        ),
    )
    .expect("conjunctive report");

    assert!(
        rows.is_empty(),
        "a key filter that matches nothing yields no rows even when the provider filter matches"
    );
}

#[test]
fn sum_equality_across_groupings_at_integer_micro_precision() {
    let fixture = create_ledger();
    seed_standard(&fixture);
    let conn = open_conn(&fixture);
    let range = (DAY_ZERO, DAY_ZERO + 3 * 86_400);

    let by_day = fetch_report(&conn, &validated(range.0, range.1, GroupBy::Day, None, None)).unwrap();
    let by_provider = fetch_report(&conn, &validated(range.0, range.1, GroupBy::Provider, None, None)).unwrap();
    let by_model = fetch_report(&conn, &validated(range.0, range.1, GroupBy::Model, None, None)).unwrap();
    let by_key = fetch_report(&conn, &validated(range.0, range.1, GroupBy::VirtualKey, None, None)).unwrap();

    let day_total = total_cost_micro(&by_day);
    assert_eq!(day_total, 4_202, "0.001234 + 0.002468 + 0.0005 in integer micro-USD");
    // Requirement 2.3: sum equality at 2-decimal tolerance; integer micro-USD
    // sums are exact, so equality is asserted at full precision.
    assert_eq!(total_cost_micro(&by_provider), day_total);
    assert_eq!(total_cost_micro(&by_model), day_total);
    assert_eq!(total_cost_micro(&by_key), day_total);
}

#[test]
fn legacy_ledger_missing_optional_columns_reports_zeros_not_errors() {
    // Build a pre-cache-era ledger (Requirement 8.3): only the original
    // columns, seeded directly — RequestLogger would migrate it. The
    // TempDir is held for the test body (drops after the read
    // connections close) instead of being leaked.
    let dir = tempfile::tempdir().expect("temp dir");
    let db_path = dir.path().join("logs.db").to_string_lossy().into_owned();
    {
        let conn = rusqlite::Connection::open(&db_path).unwrap();
        conn.execute_batch(&format!(
            "{LEGACY_REQUESTS_DDL}\nINSERT INTO requests (trace_id, timestamp, method, path, model, provider, status_code, duration_ms, cost)\nVALUES ('legacy', 1767229200, 'POST', '/v1/chat/completions', 'gpt-4', 'openai', 200, 5, 0.001);",
        ))
        .unwrap();
    }

    let conn = rusqlite::Connection::open(&db_path).unwrap();
    let rows = fetch_report(
        &conn,
        &validated(DAY_ZERO, DAY_ZERO + 86_400, GroupBy::Day, None, None),
    )
    .expect("legacy ledger report");

    assert_eq!(rows.len(), 1);
    let row = &rows[0];
    assert_eq!(row.requests, 1);
    assert_eq!(row.cost_micro_usd, 1_000);
    assert_eq!(
        (row.input_tokens, row.output_tokens, row.cache_read_tokens, row.cache_creation_tokens, row.reasoning_tokens),
        (0, 0, 0, 0, 0),
        "missing token columns read as zero-default"
    );
    assert_eq!(row.cost_unpriced_requests, 0, "no token evidence means no unpriced classification");
}

#[test]
fn legacy_ledger_without_key_column_key_filter_yields_no_rows() {
    // Pre-attribution-era ledger (no key_id/virtual_key column): a key_id
    // filter cannot match anything — every row groups as `unattributed`,
    // and the `AND 1 = 0` branch must yield zero rows without erroring.
    // Filtering FOR `unattributed` still matches everything. TempDir held
    // for the test body instead of leaked.
    let dir = tempfile::tempdir().expect("temp dir");
    let db_path = dir.path().join("logs.db").to_string_lossy().into_owned();
    {
        let conn = rusqlite::Connection::open(&db_path).unwrap();
        conn.execute_batch(&format!(
            "{LEGACY_REQUESTS_MINIMAL_DDL}\nINSERT INTO requests (trace_id, timestamp, method, path, model, provider, status_code, duration_ms, cost)\nVALUES ('legacy', 1767229200, 'POST', '/v1/chat/completions', 'gpt-4', 'openai', 200, 5, 0.001);",
        ))
        .unwrap();
    }

    let conn = rusqlite::Connection::open(&db_path).unwrap();
    let specific = fetch_report(
        &conn,
        &validated(DAY_ZERO, DAY_ZERO + 86_400, GroupBy::VirtualKey, None, Some("vk-42")),
    )
    .expect("legacy key filter must not error");
    assert!(specific.is_empty(), "no key column: a specific key filter matches nothing (1 = 0 branch)");

    let unattributed = fetch_report(
        &conn,
        &validated(
            DAY_ZERO,
            DAY_ZERO + 86_400,
            GroupBy::VirtualKey,
            None,
            Some(ai_gateway::cost_reporting::query::UNATTRIBUTED),
        ),
    )
    .expect("legacy unattributed filter must not error");
    assert_eq!(unattributed.len(), 1, "filtering for unattributed matches the single bucket");
    assert_eq!(unattributed[0].requests, 1);
}

#[test]
fn repeat_queries_are_deterministic() {
    let fixture = create_ledger();
    seed_standard(&fixture);
    let conn = open_conn(&fixture);

    let first = fetch_report(&conn, &validated(DAY_ZERO, DAY_ZERO + 3 * 86_400, GroupBy::Day, None, None)).unwrap();
    let second = fetch_report(&conn, &validated(DAY_ZERO, DAY_ZERO + 3 * 86_400, GroupBy::Day, None, None)).unwrap();

    assert_eq!(
        first.iter().map(|row| (row.dimension.clone(), row.cost_micro_usd, row.requests)).collect::<Vec<_>>(),
        second.iter().map(|row| (row.dimension.clone(), row.cost_micro_usd, row.requests)).collect::<Vec<_>>(),
        "identical queries return identical rows (Requirement 6.1)"
    );
}

// --- Reconciliation (spec Task 4, Requirement 3) -------------------------------

use ai_gateway::cost_reporting::query::fetch_reconciliation;
use ai_gateway::cost_reporting::{
    KeyUsageTotals, LedgerTotals, ProviderSpendTotals, ReconcileParams,
    ReconciliationReport, ValidatedRange as ReconRange,
};

/// Isolated tri-store fixture: real `RequestLogger` ledger plus empty
/// sibling budgets.db / keys.db files in the same temp dir (the placement
/// rule the server uses), so each test seeds only the store it asserts on.
/// The TempDir is held (not leaked): `_logger` drops first, then `_dir`
/// cleans up the databases.
struct ReconFixture {
    _logger: RequestLogger,
    _dir: tempfile::TempDir,
    logs_db: String,
    budgets_db: String,
    keys_db: String,
}

fn create_recon_fixture() -> ReconFixture {
    let dir = tempfile::tempdir().expect("temp dir for isolated stores");
    let logs_db = dir.path().join("logs.db").to_string_lossy().into_owned();
    let logger = RequestLogger::new(cost_reports_logging_config(logs_db.clone()))
        .expect("create isolated request logger");
    let budgets_db = dir.path().join("budgets.db").to_string_lossy().into_owned();
    let keys_db = dir.path().join("keys.db").to_string_lossy().into_owned();
    ReconFixture {
        _logger: logger,
        _dir: dir,
        logs_db,
        budgets_db,
        keys_db,
    }
}

fn recon_range(start: i64, end: i64) -> ReconRange {
    ValidatedRange { start_secs: start, end_secs: end }
}

fn reconcile(fixture: &ReconFixture, start: i64, end: i64) -> ReconciliationReport {
    fetch_reconciliation(
        &fixture.logs_db,
        &fixture.budgets_db,
        &fixture.keys_db,
        recon_range(start, end),
    )
    .expect("reconciliation never errors on zero-activity stores (Req 3.2)")
}

/// Seed the provider budget store leg with the real store schema and
/// accrual path: `add_spend` upserts under a window anchor, so the row's
/// `window_start`/`updated_at` carry the alignment semantics the caveat
/// logic reads.
fn seed_budget_row(fixture: &ReconFixture, provider: &str, usd: f64, anchor: i64) {
    let conn = rusqlite::Connection::open(&fixture.budgets_db).unwrap();
    conn.execute_batch(
        "CREATE TABLE IF NOT EXISTS provider_budgets (
provider TEXT PRIMARY KEY,
limit_usd REAL,
window TEXT CHECK(window IN ('daily','weekly','monthly','manual')),
window_start INTEGER,
current_spend_usd REAL NOT NULL DEFAULT 0.0,
updated_at INTEGER
);",
    )
    .unwrap();
    // Mirror ProviderBudgetStore::add_spend's upsert shape with a fixed
    // clock: window anchored at `anchor`, updated_at at anchor + 60.
    conn.execute(
        "INSERT INTO provider_budgets
(provider, limit_usd, window, window_start, current_spend_usd, updated_at)
VALUES (?1, NULL, 'manual', ?2, ?3, ?2 + 60)
ON CONFLICT(provider) DO UPDATE SET
current_spend_usd = current_spend_usd + ?3,
updated_at = ?2 + 60
WHERE window_start = ?2",
        rusqlite::params![provider, anchor, usd],
    )
    .unwrap();
}

/// Seed the key-usage leg with the real keys.db schema (virtual_keys +
/// virtual_key_usage with the per-row timestamp the windowed sum reads).
fn seed_usage_row(fixture: &ReconFixture, key_id: &str, cost_usd: f64, timestamp: i64) {
    let conn = rusqlite::Connection::open(&fixture.keys_db).unwrap();
    conn.execute_batch(
        "CREATE TABLE IF NOT EXISTS virtual_keys (
id TEXT PRIMARY KEY,
key_hash TEXT NOT NULL UNIQUE,
key_prefix TEXT NOT NULL,
encrypted_key TEXT NOT NULL,
name TEXT,
status TEXT NOT NULL DEFAULT 'active',
budget_limit_usd REAL,
token_budget INTEGER,
budget_window TEXT,
current_spend_usd REAL NOT NULL DEFAULT 0.0,
current_tokens_used INTEGER NOT NULL DEFAULT 0,
window_start_timestamp INTEGER,
requests_per_minute INTEGER,
tokens_per_minute INTEGER,
model_access_list TEXT,
loop_detection_json TEXT,
expires_at INTEGER,
created_at INTEGER NOT NULL,
last_used_at INTEGER,
request_count INTEGER NOT NULL DEFAULT 0
);
CREATE TABLE IF NOT EXISTS virtual_key_usage (
id INTEGER PRIMARY KEY AUTOINCREMENT,
key_id TEXT NOT NULL REFERENCES virtual_keys(id) ON DELETE CASCADE,
model_group TEXT NOT NULL,
model TEXT NOT NULL,
input_tokens INTEGER NOT NULL,
output_tokens INTEGER NOT NULL,
cost_usd REAL NOT NULL,
timestamp INTEGER NOT NULL
);",
    )
    .unwrap();
    conn.execute(
        "INSERT OR IGNORE INTO virtual_keys (id, key_hash, key_prefix, encrypted_key, status, created_at)
VALUES (?1, 'hash-' || ?1, 'vk_prefix_', 'enc', 'active', 1)",
        rusqlite::params![key_id],
    )
    .unwrap();
    conn.execute(
        "INSERT INTO virtual_key_usage (key_id, model_group, model, input_tokens, output_tokens, cost_usd, timestamp)
VALUES (?1, 'openai', 'gpt-4', 10, 20, ?2, ?3)",
        rusqlite::params![key_id, cost_usd, timestamp],
    )
    .unwrap();
}

#[test]
fn reconcile_zero_activity_window_returns_all_zeros_and_no_error() {
    let fixture = create_recon_fixture();
    // Ledger seeded with rows but querying a window with no activity;
    // budgets/keys stores never created (files absent).
    seed_row(
        &fixture._logger,
        DAY_ZERO + 3_600,
        "openai",
        "gpt-4",
        0.001,
        None,
        None,
        None,
        "/v1/chat/completions",
    );

    let report = reconcile(&fixture, DAY_ZERO + 10 * 86_400, DAY_ZERO + 11 * 86_400);

    assert_eq!(report.ledger.requests, 0);
    assert_eq!(report.ledger.cost_micro_usd, 0);
    assert_eq!(report.unpriced_requests, 0);
    assert_eq!(report.provider_spend.cost_micro_usd, 0, "Req 3.2: missing store is zero, not error");
    assert!(!report.provider_spend.caveat, "nothing to misalign against");
    assert_eq!(report.key_usage.cost_micro_usd, 0);
    assert_eq!(report.difference_micro_usd, 0);
}

#[test]
fn reconcile_difference_math_uses_ledger_minus_provider_spend() {
    let fixture = create_recon_fixture();
    // Ledger: 2 priced requests (1.5¢ total) + 1 unpriced + 1 #attempt
    // that must be excluded from every count.
    seed_row(&fixture._logger, DAY_ZERO + 3_600, "openai", "gpt-4", 0.001_000, None, None, None, "/v1/chat/completions");
    seed_row(&fixture._logger, DAY_ZERO + 7_200, "openai", "gpt-4", 0.000_500, None, None, None, "/v1/chat/completions");
    seed_row(&fixture._logger, DAY_ZERO + 9_000, "openai", "gpt-4", 0.0, None, None, Some(32), "/v1/chat/completions");
    seed_row(&fixture._logger, DAY_ZERO + 9_600, "openai", "gpt-4", 0.010, None, None, None, "/v1/chat/completions#attempt");
    // Provider store accrued 0.002 in the same window (aligned: window
    // opens exactly at DAY_ZERO, last accrual at DAY_ZERO + 60).
    seed_budget_row(&fixture, "openai", 0.002, DAY_ZERO);
    // Key usage: 0.001 in-window + 0.5 out-of-window (must be excluded).
    seed_usage_row(&fixture, "key-1", 0.001, DAY_ZERO + 3_600);
    seed_usage_row(&fixture, "key-1", 0.5, DAY_ZERO + 40 * 86_400);

    let report = reconcile(&fixture, DAY_ZERO, DAY_ZERO + 86_400);

    assert_eq!(report.ledger.requests, 3, "#attempt row excluded from count");
    assert_eq!(report.ledger.cost_micro_usd, 1_500);
    assert_eq!(report.unpriced_requests, 1, "zero-cost row with tokens counts as unpriced");
    assert_eq!(report.provider_spend.cost_micro_usd, 2_000);
    // Requirement 3.3: difference = ledger − provider spend, sign intact.
    assert_eq!(report.difference_micro_usd, 1_500 - 2_000);
    assert_eq!(report.key_usage.cost_micro_usd, 1_000, "windowed sum excludes out-of-window rows");
    assert!(!report.provider_spend.caveat, "window aligned to stored budget window");
}

#[test]
fn reconcile_provider_spend_greater_than_ledger_keeps_negative_difference() {
    let fixture = create_recon_fixture();
    seed_row(&fixture._logger, DAY_ZERO + 3_600, "openai", "gpt-4", 0.001, None, None, None, "/v1/chat/completions");
    seed_budget_row(&fixture, "openai", 0.003, DAY_ZERO);

    let report = reconcile(&fixture, DAY_ZERO, DAY_ZERO + 86_400);

    assert_eq!(report.difference_micro_usd, 1_000 - 3_000, "negative difference is reported, never masked");
}

#[test]
fn reconcile_misaligned_window_sets_caveat_and_aligned_does_not() {
    let fixture = create_recon_fixture();
    seed_row(&fixture._logger, DAY_ZERO + 3_600, "openai", "gpt-4", 0.001, None, None, None, "/v1/chat/completions");
    // Stored window opens at DAY_ZERO, last accrual DAY_ZERO + 60.
    seed_budget_row(&fixture, "openai", 0.001, DAY_ZERO);

    // Aligned: requested window starts exactly at the stored anchor and
    // ends at/after the last accrual instant.
    let aligned = reconcile(&fixture, DAY_ZERO, DAY_ZERO + 86_400);
    assert!(!aligned.provider_spend.caveat, "requested window covers the stored window exactly");
    assert_eq!((aligned.provider_spend.window_start, aligned.provider_spend.window_end), (DAY_ZERO, DAY_ZERO + 60));

    // Misaligned (start): requested window begins before the stored
    // window opened — the ledger leg includes spend the store predates.
    let start_misaligned = reconcile(&fixture, DAY_ZERO - 86_400, DAY_ZERO + 86_400);
    assert!(start_misaligned.provider_spend.caveat, "window starting before the anchor is misaligned");

    // Misaligned (start-after): requested window begins inside the
    // stored window — the store total still includes spend from before
    // the requested window opened.
    let start_inside = reconcile(&fixture, DAY_ZERO + 3_600, DAY_ZERO + 86_400);
    assert!(start_inside.provider_spend.caveat, "window starting inside the stored window is misaligned");

    // Misaligned (end): requested window ends before the last accrual —
    // the store total includes spend accrued after the requested window.
    let end_misaligned = reconcile(&fixture, DAY_ZERO, DAY_ZERO + 30);
    assert!(end_misaligned.provider_spend.caveat, "window ending before the last accrual is misaligned");

    // Misaligned (end == horizon): the ledger end bound is EXCLUSIVE, so a
    // window ending exactly at the last accrual instant still misses that
    // instant's ledger rows while the store total includes them — equality
    // is NOT alignment.
    let end_equal = reconcile(&fixture, DAY_ZERO, DAY_ZERO + 60);
    assert!(end_equal.provider_spend.caveat, "window ending exactly at the horizon is misaligned (end is exclusive)");

    // Superset window ending after the accrual horizon stays aligned.
    let superset = reconcile(&fixture, DAY_ZERO, DAY_ZERO + 5 * 86_400);
    assert!(!superset.provider_spend.caveat);
}

#[test]
fn reconcile_mixed_window_anchors_set_caveat_even_for_matching_bounds() {
    let fixture = create_recon_fixture();
    seed_row(&fixture._logger, DAY_ZERO + 3_600, "openai", "gpt-4", 0.001, None, None, None, "/v1/chat/completions");
    // Two providers whose budget windows anchored at DIFFERENT instants:
    // summing them under MIN(window_start)/MAX(updated_at) bounds would
    // pretend they share one window — the caveat must flag it honestly.
    seed_budget_row(&fixture, "openai", 0.001, DAY_ZERO);
    seed_budget_row(&fixture, "anthropic", 0.002, DAY_ZERO + 7_200);

    let report = reconcile(&fixture, DAY_ZERO, DAY_ZERO + 86_400);

    assert!(
        report.provider_spend.caveat,
        "rows from different current windows are summed: the caveat must be set even when bounds look aligned"
    );
    // Bounds still represent the observed extent (earliest anchor, latest horizon).
    assert_eq!(report.provider_spend.window_start, DAY_ZERO);
    assert_eq!(report.provider_spend.window_end, DAY_ZERO + 7_200 + 60);
}

#[test]
fn reconcile_empty_budget_table_is_aligned_zero_not_error() {
    let fixture = create_recon_fixture();
    seed_row(&fixture._logger, DAY_ZERO + 3_600, "openai", "gpt-4", 0.001, None, None, None, "/v1/chat/completions");
    // budgets.db file exists with the table but zero rows; keys.db absent.
    let conn = rusqlite::Connection::open(&fixture.budgets_db).unwrap();
    conn.execute_batch(
        "CREATE TABLE provider_budgets (
provider TEXT PRIMARY KEY,
limit_usd REAL,
window TEXT,
window_start INTEGER,
current_spend_usd REAL NOT NULL DEFAULT 0.0,
updated_at INTEGER
);",
    )
    .unwrap();
    drop(conn);

    let report = reconcile(&fixture, DAY_ZERO, DAY_ZERO + 86_400);

    assert_eq!(report.provider_spend.cost_micro_usd, 0);
    assert!(!report.provider_spend.caveat, "empty store has nothing to misalign");
    assert_eq!(report.difference_micro_usd, 1_000);
}

#[test]
fn reconcile_entry_point_surfaces_param_validation_errors() {
    // Requirement 3.4: bad start/end are rejected by the same
    // ReconcileParams validation the route uses (mapped to HTTP 400).
    let missing = ReconcileParams {
        start: None,
        end: Some("2026-01-02T00:00:00Z".to_string()),
    }
    .validate();
    assert_eq!(missing, Err(ai_gateway::cost_reporting::CostReportValidationError::MissingStart));

    let unparseable = ReconcileParams {
        start: Some("not-a-timestamp".to_string()),
        end: Some("2026-01-02T00:00:00Z".to_string()),
    }
    .validate();
    assert!(unparseable.is_err());

    let end_before_start = ReconcileParams {
        start: Some("2026-01-02T00:00:00Z".to_string()),
        end: Some("2026-01-01T00:00:00Z".to_string()),
    }
    .validate();
    assert_eq!(
        end_before_start,
        Err(ai_gateway::cost_reporting::CostReportValidationError::EndNotAfterStart)
    );

    let ok = ReconcileParams {
        start: Some("2026-01-01T00:00:00Z".to_string()),
        end: Some("2026-01-02T00:00:00Z".to_string()),
    }
    .validate();
    assert!(ok.is_ok());
}

#[test]
fn reconcile_report_type_difference_serialization_contract() {
    // The mod.rs types compute difference_usd = ledger − provider spend
    // once, in integer micro-USD (Requirement 3.3); reconcile returns
    // raw numbers and the type does the math.
    let report = ReconciliationReport::new(
        LedgerTotals { requests: 2, cost_micro_usd: 1_234_567 },
        ProviderSpendTotals {
            cost_micro_usd: 1_000_000,
            window_start: 0,
            window_end: 0,
            caveat: false,
        },
        KeyUsageTotals { cost_micro_usd: 500_000 },
        0,
    );
    assert_eq!(report.difference_micro_usd, 234_567);
    let json = serde_json::to_value(&report).unwrap();
    assert_eq!(json["difference_usd"], serde_json::json!(0.234567));
}

#[test]
fn reconcile_never_creates_missing_store_files() {
    let fixture = create_recon_fixture();
    seed_row(&fixture._logger, DAY_ZERO + 3_600, "openai", "gpt-4", 0.001, None, None, None, "/v1/chat/completions");

    // budgets.db and keys.db do not exist yet; reconcile must read them
    // as zero-activity WITHOUT materializing empty database files (the
    // readers open READ_ONLY — SQLite cannot create the file).
    let report = reconcile(&fixture, DAY_ZERO, DAY_ZERO + 86_400);

    assert_eq!(report.provider_spend.cost_micro_usd, 0);
    assert_eq!(report.key_usage.cost_micro_usd, 0);
    assert!(
        !std::path::Path::new(&fixture.budgets_db).exists(),
        "reconcile must not create budgets.db"
    );
    assert!(
        !std::path::Path::new(&fixture.keys_db).exists(),
        "reconcile must not create keys.db"
    );
}
