//! Integration tests for the trailing-spend ledger aggregate (spec:
//! spend-pace-forecasting, Task 4 / design D4-D6) and the guarded
//! `requests(key_id, timestamp)` index migration.
//!
//! Seeds isolated logs.db ledgers through the real `RequestLogger` write
//! path and asserts multi-day/multi-provider/multi-key sums, `#attempt`
//! exclusion, horizon bounds, provider vs key scoping, zero-history
//! shapes, and index idempotence across re-opens.

use crate::common;

use ai_gateway::cost_reporting::query::{fetch_trailing_spend, TrailingScope};
use ai_gateway::logger::RequestLogger;
use common::{cost_reports_logging_config, sample_log_entry, LogEntryOverrides, DAY_ZERO, LEGACY_REQUESTS_DDL};

const DAY: i64 = 86_400;

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

/// Seed one attributed row through the real logger write path.
fn seed_row(
    logger: &RequestLogger,
    timestamp: i64,
    provider: &str,
    cost: f64,
    key_id: Option<&str>,
    path: &str,
) {
    let entry = sample_log_entry(
        &format!("trace-{timestamp}-{provider}-{cost}-{}", key_id.unwrap_or("none")),
        timestamp,
        provider,
        "gpt-4",
        cost,
        LogEntryOverrides {
            path: Some(path.to_owned()),
            virtual_key_id: key_id.map(str::to_string),
            ..Default::default()
        },
    );
    logger.log(entry).expect("seed log entry");
    logger.flush();
}

fn trailing(
    conn: &rusqlite::Connection,
    scope: TrailingScope<'_>,
    now_epoch: i64,
) -> ai_gateway::cost_reporting::TrailingSpendTotals {
    fetch_trailing_spend(conn, scope, now_epoch).expect("trailing aggregate")
}

#[test]
fn sums_multi_day_multi_provider_multi_key_rows_per_scope() {
    let fixture = create_ledger();
    let logger = &fixture._logger;
    // Ten distinct UTC days of traffic across two providers and two keys.
    for day in 0..10 {
        seed_row(logger, DAY_ZERO + day * DAY, "openai", 0.001_000, Some("vk-1"), "/v1/chat/completions");
        seed_row(logger, DAY_ZERO + day * DAY + 3_600, "openai", 0.000_500, Some("vk-2"), "/v1/chat/completions");
        seed_row(logger, DAY_ZERO + day * DAY + 7_200, "anthropic", 0.002_000, None, "/v1/chat/completions");
    }
    let conn = rusqlite::Connection::open(&fixture.db_path).unwrap();
    let now = DAY_ZERO + 10 * DAY;

    let openai = trailing(&conn, TrailingScope::Provider("openai"), now);
    assert_eq!(openai.cost_micro_usd, 10 * (1_000 + 500));
    assert_eq!(openai.history_days, 10, "one distinct UTC day per seeded day");

    let vk1 = trailing(&conn, TrailingScope::Key("vk-1"), now);
    assert_eq!(vk1.cost_micro_usd, 10 * 1_000, "key scope excludes other keys' rows");
    assert_eq!(vk1.history_days, 10);

    let unattributed = trailing(
        &conn,
        TrailingScope::Key(ai_gateway::cost_reporting::query::UNATTRIBUTED),
        now,
    );
    assert_eq!(
        unattributed.cost_micro_usd,
        10 * 2_000,
        "NULL keys normalize into the unattributed bucket, matching report grouping parity"
    );
    assert_eq!(unattributed.history_days, 10);
}

#[test]
fn attempt_rows_are_excluded_from_sum_and_history() {
    let fixture = create_ledger();
    let logger = &fixture._logger;
    seed_row(logger, DAY_ZERO, "openai", 0.001_000, Some("vk-1"), "/v1/chat/completions");
    seed_row(logger, DAY_ZERO + 3_600, "openai", 0.010_000, Some("vk-1"), "/v1/chat/completions#attempt");
    seed_row(logger, DAY_ZERO + 7_200, "openai", 0.002_000, Some("vk-1"), "/v1/chat/completions#attempt");
    let conn = rusqlite::Connection::open(&fixture.db_path).unwrap();

    let totals = trailing(&conn, TrailingScope::Key("vk-1"), DAY_ZERO + DAY);
    assert_eq!(totals.cost_micro_usd, 1_000, "both #attempt marker rows excluded from the sum");
    assert_eq!(totals.history_days, 1, "attempt-only days never count as history");
}

#[test]
fn horizon_bounds_exclude_rows_at_or_beyond_now_and_before_start() {
    let fixture = create_ledger();
    let logger = &fixture._logger;
    let now = DAY_ZERO + 40 * DAY;
    // In-horizon: exactly at start (inclusive) and one second before now.
    seed_row(logger, now - 30 * DAY, "openai", 0.001_000, Some("vk-1"), "/v1/chat/completions");
    seed_row(logger, now - 1, "openai", 0.002_000, Some("vk-1"), "/v1/chat/completions");
    // Out: exactly at now (exclusive end) and one second before the start.
    seed_row(logger, now, "openai", 0.004_000, Some("vk-1"), "/v1/chat/completions");
    seed_row(logger, now - 30 * DAY - 1, "openai", 0.008_000, Some("vk-1"), "/v1/chat/completions");
    let conn = rusqlite::Connection::open(&fixture.db_path).unwrap();

    let totals = trailing(&conn, TrailingScope::Provider("openai"), now);
    assert_eq!(totals.cost_micro_usd, 3_000, "start inclusive, end exclusive");
    assert_eq!(totals.history_days, 2, "boundary rows land on two distinct UTC days");
}

#[test]
fn empty_scope_returns_zero_spend_zero_history() {
    let fixture = create_ledger();
    let logger = &fixture._logger;
    seed_row(logger, DAY_ZERO, "openai", 0.001_000, Some("vk-1"), "/v1/chat/completions");
    let conn = rusqlite::Connection::open(&fixture.db_path).unwrap();

    let missing_provider = trailing(&conn, TrailingScope::Provider("anthropic"), DAY_ZERO + DAY);
    assert_eq!((missing_provider.cost_micro_usd, missing_provider.history_days), (0, 0));

    let missing_key = trailing(&conn, TrailingScope::Key("vk-nope"), DAY_ZERO + DAY);
    assert_eq!((missing_key.cost_micro_usd, missing_key.history_days), (0, 0));
}

#[test]
fn legacy_ledger_without_key_column_matches_only_unattributed() {
    let dir = tempfile::tempdir().expect("temp dir");
    let db_path = dir.path().join("logs.db").to_string_lossy().into_owned();
    {
        let conn = rusqlite::Connection::open(&db_path).unwrap();
        conn.execute_batch(&format!(
            "{LEGACY_REQUESTS_DDL}\nINSERT INTO requests (trace_id, timestamp, method, path, model, provider, status_code, duration_ms, cost)\nVALUES ('legacy', {DAY_ZERO}, 'POST', '/v1/chat/completions', 'gpt-4', 'openai', 200, 5, 0.001);",
        ))
        .unwrap();
    }

    let conn = rusqlite::Connection::open(&db_path).unwrap();
    let specific = trailing(&conn, TrailingScope::Key("vk-1"), DAY_ZERO + DAY);
    assert_eq!((specific.cost_micro_usd, specific.history_days), (0, 0), "no key column: specific key matches nothing");

    let provider = trailing(&conn, TrailingScope::Provider("openai"), DAY_ZERO + DAY);
    assert_eq!(provider.cost_micro_usd, 1_000, "provider scope works on legacy ledgers");
}

#[test]
fn feeds_forecast_manual_run_rate_inputs() {
    use ai_gateway::metrics::forecast::{forecast_manual, ForecastStatus};

    let fixture = create_ledger();
    let logger = &fixture._logger;
    // 25 days of $1.00/day for vk-1, then silence.
    for day in 0..25 {
        seed_row(logger, DAY_ZERO + day * DAY, "openai", 1.0, Some("vk-1"), "/v1/chat/completions");
    }
    let conn = rusqlite::Connection::open(&fixture.db_path).unwrap();
    let now = DAY_ZERO + 30 * DAY;

    let totals = trailing(&conn, TrailingScope::Key("vk-1"), now);
    assert_eq!(totals.cost_micro_usd, 25_000_000);
    assert_eq!(totals.history_days, 25);

    let now_dt = chrono::DateTime::from_timestamp(now, 0).unwrap();
    let forecast = forecast_manual(120.0, 90.0, totals.cost_micro_usd as f64 / 1_000_000.0, totals.history_days, now_dt);
    // pace = 25/30 per day; projected = 90 + 25 = 115 < 120 -> within window.
    assert_eq!(forecast.status, ForecastStatus::WithinWindow);
    assert_eq!(forecast.projected_usd, 115.0);

    // History below the 7-day gate must surface insufficient data.
    let young = forecast_manual(120.0, 90.0, 1.0, 3, now_dt);
    assert_eq!(young.status, ForecastStatus::InsufficientData);
}

// --- Guarded index migration (design D6) ---------------------------------------

#[test]
fn key_id_timestamp_index_exists_and_reopen_is_idempotent() {
    let dir = tempfile::tempdir().expect("temp dir for isolated logs.db");
    let db_path = dir.path().join("logs.db").to_string_lossy().into_owned();

    let first = RequestLogger::new(cost_reports_logging_config(db_path.clone()))
        .expect("first open creates the schema and indexes");
    first.log(sample_log_entry(
        "trace-index-idempotence",
        DAY_ZERO,
        "openai",
        "gpt-4",
        0.001,
        LogEntryOverrides::default(),
    ))
    .expect("seed one row");
    first.flush();

    let index_count = || {
        let conn = rusqlite::Connection::open(&db_path).unwrap();
        conn.query_row(
            "SELECT COUNT(*) FROM sqlite_master WHERE type = 'index' AND name = 'idx_requests_key_id_timestamp'",
            [],
            |row| row.get::<_, i64>(0),
        )
        .unwrap()
    };

    assert_eq!(index_count(), 1, "first open created the guarded index exactly once");

    // Re-open the existing database through the real logger (writer thread
    // closed first so the second open runs the migration path alone): the
    // guarded CREATE INDEX IF NOT EXISTS must be a no-op — no error, no
    // duplicate.
    drop(first);
    let second = RequestLogger::new(cost_reports_logging_config(db_path.clone()))
        .expect("re-open existing logs.db through the migration path");
    second.flush();
    drop(second);
    assert_eq!(index_count(), 1, "re-open did not duplicate or drop the index");

    // A third open keeps it stable.
    let third = RequestLogger::new(cost_reports_logging_config(db_path.clone()))
        .expect("third open");
    third.flush();
    drop(third);
    assert_eq!(index_count(), 1);
}

#[test]
fn legacy_ledger_migration_adds_the_index() {
    let dir = tempfile::tempdir().expect("temp dir");
    let db_path = dir.path().join("logs.db").to_string_lossy().into_owned();
    {
        let conn = rusqlite::Connection::open(&db_path).unwrap();
        conn.execute_batch(LEGACY_REQUESTS_DDL).unwrap();
        // No logger has ever opened this database: no indexes exist yet.
        let pre: i64 = conn
            .query_row(
                "SELECT COUNT(*) FROM sqlite_master WHERE type = 'index' AND name = 'idx_requests_key_id_timestamp'",
                [],
                |row| row.get(0),
            )
            .unwrap();        assert_eq!(pre, 0);
    }

    let logger = RequestLogger::new(cost_reports_logging_config(db_path.clone()))
        .expect("open legacy ledger through the migration path");
    logger.flush();

    let conn = rusqlite::Connection::open(&db_path).unwrap();
    let post: i64 = conn
        .query_row(
            "SELECT COUNT(*) FROM sqlite_master WHERE type = 'index' AND name = 'idx_requests_key_id_timestamp'",
            [],
            |row| row.get(0),
        )
        .unwrap();
    assert_eq!(post, 1, "migration added the guarded index to a legacy database");
}
