//! Ledger aggregate SQL + query layer (spec: billing-grade-cost-reporting,
//! Task 2). All report SQL lives here; no HTTP types.
//!
//! Read-only, parameterized SELECTs over the logs.db `requests` ledger:
//! - groupings: UTC day / provider / model / virtual_key (Requirement 1);
//! - `#attempt` marker rows excluded via `path NOT LIKE '%#attempt'`
//! (design.md Component 2 — attempt rows are failure telemetry, not billable
//! outcomes);
//! - token columns read with zero-default semantics when the ledger predates
//! a column (Requirement 8.3): `cache_read_tokens`, `cache_creation_tokens`,
//! `reasoning_tokens` are nullable in the schema and COALESCE'd to 0;
//! `input_tokens`/`output_tokens`/`key_id`/`virtual_key` columns do not
//! exist in the current schema, so presence is detected per-connection via
//! `PRAGMA table_info` (mirroring the logger's `ensure_column` probe) and
//! missing columns read as 0 / `unattributed`. If additive ledger columns
//! land later, this layer picks them up with no code change.
//! - `cost_micro_usd = SUM(CAST(ROUND(cost*1000000) AS INTEGER))` computed
//! in SQL (integer micro-USD, Requirement 6.2);
//! - `cost_unpriced_requests = COUNT(cost = 0 AND tokens > 0)`
//! (Requirement 1.8) — token presence uses input+output when available and
//! falls back to cache/reasoning tokens so legacy rows with only cache
//! columns still count as usage;
//! - day grouping generates the UTC calendar-day spine in Rust so empty
//! days return zero rows (Requirement 1.3);
//! - NULL/empty key attribution collapses to the literal `unattributed`
//! dimension (Requirement 2.1).
//!
//! Call discipline: run [`fetch_report`] via `tokio::task::spawn_blocking`,
//! matching the logger query call sites (dashboard rate-history handler).
//! The connection handle is whatever the caller owns; tests open
//! `rusqlite::Connection` directly against the isolated logs.db.

use rusqlite::OpenFlags;
use rusqlite::Connection;

use crate::cost_reporting::{
    GroupBy, KeyUsageTotals, LedgerTotals, ProviderSpendTotals, ReconciliationReport,
    ReportRow, ValidatedCostReportParams, ValidatedRange,
};

/// Error returned by [`fetch_report_async`]: the SQL failure or the
/// blocking-task join failure. Callers map this to the admin 500 JSON
/// envelope; SQL errors never panic.
#[derive(Debug, thiserror::Error)]
pub enum AsyncQueryError {
    #[error("cost report query failed: {0}")]
    Query(#[from] QueryError),
    #[error("cost report blocking task failed: {0}")]
    Join(#[from] tokio::task::JoinError),
}

/// Async entry point for admin route handlers: opens a read connection
/// against the logger's database path and runs [`fetch_report`] on the
/// blocking thread pool, matching the logger query call-site discipline
/// (dashboard rate-history handler) — SQLite I/O never parks an async
/// worker. WAL mode allows this reader to run concurrently with the
/// logger's writer thread.
pub async fn fetch_report_async(
    database_path: String,
    params: ValidatedCostReportParams,
) -> Result<Vec<ReportRow>, AsyncQueryError> {
    tokio::task::spawn_blocking(move || {
        let conn = open_read_only(&database_path)?;
        fetch_report(&conn, &params).map_err(|error| {
            tracing::warn!(%error, database_path = %database_path, "cost report query failed");
            error
        })
    })
    .await?
    .map_err(AsyncQueryError::Query)
}

/// Literal dimension value for requests that carried no virtual key
/// (Requirement 2.1). A billing report must sum to total spend, so
/// unauthenticated traffic needs a stable bucket, never a dropped row.
pub const UNATTRIBUTED: &str = "unattributed";

/// Async entry point for the reconcile route (Requirement 3): runs the
/// three read-only store queries ([`fetch_reconciliation`]) on the
/// blocking thread pool. Paths are resolved by the caller (server
/// layer) with the same placement rules the stores use:
/// `budgets.db` is a sibling of the logs database, `keys.db` comes
/// from `config.virtual_keys.database_path`.
///
/// Design (Error Handling #5): the provider budget store keeps only
/// *current-window* spend, not history, so reconcile is exact only for
/// windows aligned to that current window; the `caveat` flag reports
/// misalignment instead of pretending history exists.
pub async fn fetch_reconciliation_async(
    logs_db_path: String,
    budgets_db_path: String,
    keys_db_path: String,
    range: ValidatedRange,
) -> Result<ReconciliationReport, AsyncQueryError> {
    tokio::task::spawn_blocking(move || {
        let result = fetch_reconciliation(&logs_db_path, &budgets_db_path, &keys_db_path, range);
        if let Err(error) = &result {
            tracing::warn!(%error, "cost report reconciliation failed");
        }
        result
    })
    .await?
    .map_err(AsyncQueryError::Query)
}

/// Synchronous reconciliation over the three stores (Requirement 3.1):
///
/// - **ledger** (`logs.db` `requests`): total cost + request count,
/// excluding `#attempt` marker rows, plus unpriced count (same
/// expressions as [`fetch_report`]);
/// - **provider store** (`budgets.db` `provider_budgets`): summed
/// `current_spend_usd` with the observed window bounds and a `caveat`
/// flag set when the requested window does not align to the stored
/// budget windows (design Error Handling #5 — the store keeps only
/// current-window spend, not history, so a mismatch means the number
/// is an approximation, never a silent error);
/// - **key usage** (`keys.db` `virtual_key_usage`): summed `cost_usd`
/// over the requested window (the table carries a per-row `timestamp`,
/// so the sum is a true windowed aggregate).
///
/// `difference_usd` is computed by [`ReconciliationReport::new`] in
/// integer micro-USD as ledger minus provider spend (Requirement 3.3);
/// this layer returns raw numbers only. Empty or missing stores yield
/// zero totals, never errors (Requirement 3.2).
pub fn fetch_reconciliation(
    logs_db_path: &str,
    budgets_db_path: &str,
    keys_db_path: &str,
    range: ValidatedRange,
) -> Result<ReconciliationReport, QueryError> {
    let (ledger, unpriced_requests) = ledger_totals(logs_db_path, range)?;
    let provider_spend = provider_spend_totals(budgets_db_path, range)?;
    let key_usage = key_usage_totals(keys_db_path, range)?;
    Ok(ReconciliationReport::new(
        ledger,
        provider_spend,
        key_usage,
        unpriced_requests,
    ))
}

/// Ledger leg of the reconcile (Requirement 3.1): unfiltered totals over
/// `requests` in the window. `#attempt` rows are excluded (failure
/// telemetry, not billable outcomes) and unpriced rows are counted
/// separately via the same expressions the report aggregate uses.
/// Returns zeros when the ledger has no matching rows (Requirement 3.2).
fn ledger_totals(
    logs_db_path: &str,
    range: ValidatedRange,
) -> Result<(LedgerTotals, i64), QueryError> {
    let zero = || (LedgerTotals {
        requests: 0,
        cost_micro_usd: 0,
    }, 0);
    let Some(conn) = open_existing(logs_db_path)? else {
        return Ok(zero());
    };
    if !table_exists(&conn, "requests")? {
        return Ok(zero());
    }
    let columns = LedgerColumns::probe(&conn)?;
    let sql = format!(
        "SELECT \
COUNT(*) AS requests, \
{cost_micro_usd} AS cost_micro_usd, \
{unpriced_requests} AS cost_unpriced_requests \
FROM requests \
WHERE timestamp >= ?1 AND timestamp < ?2 AND path NOT LIKE '%#attempt'",
        cost_micro_usd = columns.cost_micro_usd_sum(),
        unpriced_requests = columns.unpriced_count_expr(),
    );
    let (requests, cost_micro_usd, unpriced_requests): (i64, i64, i64) = conn.query_row(
        &sql,
        rusqlite::params![range.start_secs, range.end_secs],
        |row| Ok((row.get(0)?, row.get(1)?, row.get(2)?)),
    )?;
    Ok((
        LedgerTotals {
            requests,
            cost_micro_usd,
        },
        unpriced_requests,
    ))
}

/// Provider-store leg of the reconcile (Requirement 3.1): the summed
/// current-window `current_spend_usd` across all provider rows, with the
/// observed window bounds (earliest `window_start`, latest `updated_at`).
///
/// Caveat semantics (design Error Handling #5): the store persists only
/// the CURRENT window's spend — there is no spend history — so the sum
/// is comparable to the ledger only when the requested window is
/// aligned to the stored window. Alignment is decided from what the
/// store actually persists (`window_start` anchor + `updated_at`
/// accrual horizon; the `window` policy text is insert-time metadata
/// that always reads `manual` and is NOT authoritative): the window is
/// aligned only when every summed row shares ONE `window_start` anchor,
/// the requested range starts exactly at that anchor, and the requested
/// range's exclusive end extends strictly past the last accrual
/// instant (`end > horizon`, because the ledger end is exclusive), i.e.
/// the store's numbers were produced entirely inside the requested
/// window and the ledger window excludes nothing the store predates.
/// Any other shape — mixed anchors across rows (a MIN/MAX pair would
/// silently fuse different windows), a start before the anchor (mixes
/// in pre-window ledger spend the store rolled over), a start after it
/// (misses store spend the ledger window excludes), or an end that
/// fails to strictly cover the horizon — sets the caveat. Zero rows
/// (Requirement 3.2) report zero totals; an empty store is treated as
/// aligned (nothing to misalign).
fn provider_spend_totals(
    budgets_db_path: &str,
    range: ValidatedRange,
) -> Result<ProviderSpendTotals, QueryError> {
    let Some(conn) = open_existing(budgets_db_path)? else {
        return Ok(zero_provider_spend(range.start_secs, range.end_secs));
    };
    if !table_exists(&conn, "provider_budgets")? {
        return Ok(zero_provider_spend(range.start_secs, range.end_secs));
    }
    let (spend_usd, anchor, horizon, distinct_anchors): (
        f64,
        Option<i64>,
        Option<i64>,
        i64,
    ) = conn.query_row(
        "SELECT COALESCE(SUM(current_spend_usd), 0.0), \
MIN(window_start), MAX(updated_at), COUNT(DISTINCT window_start) \
FROM provider_budgets",
        [],
        |row| Ok((row.get(0)?, row.get(1)?, row.get(2)?, row.get(3)?)),
    )?;
    let Some(anchor) = anchor else {
        return Ok(zero_provider_spend(range.start_secs, range.end_secs));
    };
    let horizon = horizon.unwrap_or(anchor);
    // caveat=false ONLY when every summed row shares one window_start
    // anchor AND the requested window aligns to it with a strictly
    // covering exclusive end (finding: end == horizon is NOT aligned).
    let single_anchor = distinct_anchors <= 1;
    let caveat = !(single_anchor && range.start_secs == anchor && range.end_secs > horizon);
    Ok(ProviderSpendTotals {
        cost_micro_usd: usd_to_micro(spend_usd),
        window_start: anchor,
        window_end: horizon,
        caveat,
    })
}

/// Key-usage leg of the reconcile (Requirement 3.1): summed accrued
/// `cost_usd` from `virtual_key_usage` windowed by its per-row
/// `timestamp` (inclusive start, exclusive end — the same bounds
/// convention the ledger uses). No rows in the window yields zero
/// totals, never an error (Requirement 3.2).
fn key_usage_totals(
    keys_db_path: &str,
    range: ValidatedRange,
) -> Result<KeyUsageTotals, QueryError> {
    let Some(conn) = open_existing(keys_db_path)? else {
        return Ok(KeyUsageTotals { cost_micro_usd: 0 });
    };
    if !table_exists(&conn, "virtual_key_usage")? {
        return Ok(KeyUsageTotals { cost_micro_usd: 0 });
    }
    let cost_usd: f64 = conn.query_row(
        "SELECT COALESCE(SUM(cost_usd), 0.0) FROM virtual_key_usage \
WHERE timestamp >= ?1 AND timestamp < ?2",
        rusqlite::params![range.start_secs, range.end_secs],
        |row| row.get(0),
    )?;
    Ok(KeyUsageTotals {
        cost_micro_usd: usd_to_micro(cost_usd),
    })
}

/// Zero provider-spend totals for an empty store (Requirement 3.2):
/// bounds fall back to the requested window so the response still
/// names what was asked about; an empty store has nothing to
/// misalign, so the caveat stays false.
fn zero_provider_spend(window_start: i64, window_end: i64) -> ProviderSpendTotals {
    ProviderSpendTotals {
        cost_micro_usd: 0,
        window_start,
        window_end,
        caveat: false,
    }
}

/// f64 USD to integer micro-USD with the same rounding the ledger sum
/// uses (`CAST(ROUND(x * 1e6) AS INTEGER)`); non-finite values read as
/// zero so a corrupt row can never poison the total.
fn usd_to_micro(usd: f64) -> i64 {
    if usd.is_finite() {
        (usd * 1_000_000.0).round() as i64
    } else {
        0
    }
}

/// Open a read connection only when the database file exists. `Ok(None)`
/// means the store has never been created: reconcile treats that as
/// "no activity" (zero totals, Requirement 3.2) rather than an error.
/// TOCTOU-safe: the file is opened `READ_ONLY | NO_MUTEX` directly — if a
/// deletion race removes it between the exists check and the open, SQLite
/// fails to open instead of creating an empty database, and that failure
/// maps to the same `None` (no activity) path instead of surfacing an
/// error. Every connection gets `PRAGMA busy_timeout=5000` so a concurrent
/// writer lock degrades to a bounded wait, matching the logger's own
/// connection setup.
fn open_existing(path: &str) -> Result<Option<Connection>, QueryError> {
    if !std::path::Path::new(path).exists() {
        return Ok(None);
    }
    match open_read_only(path) {
        Ok(conn) => Ok(Some(conn)),
        Err(error) => match &error.0 {
            rusqlite::Error::SqliteFailure(ffi, _)
                if ffi.code == rusqlite::ErrorCode::CannotOpen =>
            {
                Ok(None)
            }
            _ => Err(error),
        },
    }
}

/// Open a strictly read-only connection with a bounded busy wait. Used by
/// both the report reader (`fetch_report_async`) and every reconcile
/// reader — none of them may create or mutate a database.
fn open_read_only(path: &str) -> Result<Connection, QueryError> {
    let conn = Connection::open_with_flags(
        path,
        OpenFlags::SQLITE_OPEN_READ_ONLY | OpenFlags::SQLITE_OPEN_NO_MUTEX,
    )?;
    conn.busy_timeout(std::time::Duration::from_millis(5_000))?;
    Ok(conn)
}

/// Does `table` exist on this connection? Missing sidecar tables (a
/// store created before a feature, or an empty file) read as no-activity
/// rather than erroring (Requirement 3.2).
fn table_exists(conn: &Connection, table: &str) -> Result<bool, QueryError> {
    let exists: bool = conn.query_row(
        "SELECT EXISTS(SELECT 1 FROM sqlite_master WHERE type = 'table' AND name = ?1)",
        rusqlite::params![table],
        |row| row.get(0),
    )?;
    Ok(exists)
}

/// Query error: rusqlite failures only (caller maps to HTTP 500 JSON).
#[derive(Debug, thiserror::Error)]
#[error("cost report query failed: {0}")]
pub struct QueryError(#[from] rusqlite::Error);

/// Which optional ledger columns exist on this connection's `requests`
/// table. Columns added by future additive migrations are detected once per
/// query and read with their real values; columns the database predates
/// read as zero-default / `unattributed` (Requirement 8.3).
#[derive(Debug, Clone, Copy)]
struct LedgerColumns {
    input_tokens: bool,
    output_tokens: bool,
    cache_read_tokens: bool,
    cache_creation_tokens: bool,
    reasoning_tokens: bool,
    key_id: bool,
    virtual_key: bool,
}

impl LedgerColumns {
    fn probe(conn: &Connection) -> Result<Self, rusqlite::Error> {
        let mut stmt = conn.prepare("PRAGMA table_info(requests)")?;
        let names = stmt
            .query_map([], |row| row.get::<_, String>(1))?
            .collect::<Result<Vec<String>, _>>()?;
        let has = |name: &str| names.iter().any(|column| column == name);
        Ok(Self {
            input_tokens: has("input_tokens"),
            output_tokens: has("output_tokens"),
            cache_read_tokens: has("cache_read_tokens"),
            cache_creation_tokens: has("cache_creation_tokens"),
            reasoning_tokens: has("reasoning_tokens"),
            key_id: has("key_id"),
            virtual_key: has("virtual_key"),
        })
    }

    /// Column supplying the virtual-key dimension: `key_id` preferred (the
    /// key's stable id, Requirement 2.2), `virtual_key` as fallback spelling.
    fn key_column(&self) -> Option<&'static str> {
        if self.key_id {
            Some("key_id")
        } else if self.virtual_key {
            Some("virtual_key")
        } else {
            None
        }
    }

    /// SQL expression for the virtual-key dimension value with NULL/empty
    /// collapsing to `unattributed` (Requirement 2.1).
    fn key_dimension_expr(&self) -> String {
        match self.key_column() {
            Some(column) => format!(
                "COALESCE(NULLIF(TRIM(CAST({column} AS TEXT)), ''), '{UNATTRIBUTED}')"
            ),
            None => format!("'{UNATTRIBUTED}'"),
        }
    }

    /// Zero-default SUM input for one optional token column: the real
    /// `COALESCE(col, 0)` when present, literal `0` when the database
    /// predates the column (Requirement 8.3).
    fn token_sum(&self, present: bool, column: &str) -> String {
        if present {
            format!("SUM(COALESCE({column}, 0))")
        } else {
            "0".to_string()
        }
    }

    /// Integer micro-USD cost sum for the ledger (`cost` is REAL): the same
    /// `CAST(ROUND(cost * 1000000) AS INTEGER)` aggregate everywhere, with
    /// zero-default so empty windows read 0 (Requirement 6.2). Shared by
    /// the report aggregate and the reconcile ledger leg — one expression,
    /// one place.
    fn cost_micro_usd_sum(&self) -> &'static str {
        "COALESCE(SUM(CAST(ROUND(cost * 1000000) AS INTEGER)), 0)"
    }

    /// Unpriced-request counter (Requirement 1.8): `cost = 0 AND tokens > 0`,
    /// where token presence uses the column-adaptive [`Self::usage_tokens_expr`].
    /// Zero-default so empty windows read 0.
    fn unpriced_count_expr(&self) -> String {
        format!(
            "COALESCE(SUM(CASE WHEN cost = 0 AND ({}) > 0 THEN 1 ELSE 0 END), 0)",
            self.usage_tokens_expr()
        )
    }

    /// Token-presence predicate for unpriced counting (Requirement 1.8):
    /// `cost = 0 AND tokens > 0`. Uses input+output when present, falling
    /// back to whichever token columns exist so legacy rows with usage in
    /// only some columns are still counted as unpriced usage.
    fn usage_tokens_expr(&self) -> String {
        let mut parts: Vec<String> = Vec::new();
        if self.input_tokens {
            parts.push("COALESCE(input_tokens, 0)".to_string());
        }
        if self.output_tokens {
            parts.push("COALESCE(output_tokens, 0)".to_string());
        }
        if self.cache_read_tokens {
            parts.push("COALESCE(cache_read_tokens, 0)".to_string());
        }
        if self.cache_creation_tokens {
            parts.push("COALESCE(cache_creation_tokens, 0)".to_string());
        }
        if self.reasoning_tokens {
            parts.push("COALESCE(reasoning_tokens, 0)".to_string());
        }
        if parts.is_empty() {
            "0".to_string()
        } else {
            parts.join(" + ")
        }
    }
}

/// Build the `WHERE` clause shared by every grouping: range bounds on
/// `timestamp` (inclusive start, exclusive end), `#attempt` exclusion, and
/// the conjunctive optional provider/key filters (Requirement 1.5). Returns
/// the clause text and the parameter values in bind order. Parameterized
/// throughout — user input never concatenates into SQL.
fn where_clause(
    columns: &LedgerColumns,
    range: ValidatedRange,
    provider: Option<&str>,
    key_id: Option<&str>,
) -> (String, Vec<Box<dyn rusqlite::ToSql>>) {
    let mut clause = String::from(
        " WHERE timestamp >= ?1 AND timestamp < ?2 AND path NOT LIKE '%#attempt'",
    );
    let mut params: Vec<Box<dyn rusqlite::ToSql>> =
        vec![Box::new(range.start_secs), Box::new(range.end_secs)];

    if let Some(provider) = provider {
        clause.push_str(" AND provider = ?");
        params.push(Box::new(provider.to_string()));
    }

// Key filter binds against whichever key column exists, applying the
// SAME NULL/empty → `unattributed` normalization the grouping uses, so
// filtering for `unattributed` also matches NULL/blank rows; with no
// key column every row groups as `unattributed` by definition, so the
// filter matches only when it asks for that literal bucket.
if let Some(key_id) = key_id {
    match columns.key_column() {
        Some(column) => {
            clause.push_str(&format!(
                " AND COALESCE(NULLIF(TRIM(CAST({column} AS TEXT)), ''), '{UNATTRIBUTED}') = ?"
            ));
            params.push(Box::new(key_id.to_string()));
        }
        None => {
            if key_id == UNATTRIBUTED {
                // No key column: all rows are unattributed; the filter
                // keeps them (narrows nothing).
            } else {
                clause.push_str(" AND 1 = 0");
            }
        }
    }
}

    (clause, params)
}

/// Aggregate one grouping of the cost report from the `requests` ledger.
///
/// SQL responsibilities: GROUP BY the dimension, sum token columns with
/// zero-defaults, sum integer micro-USD (`CAST(ROUND(cost*1000000) AS
/// INTEGER)`), and count unpriced requests in-window. Rust responsibilities:
/// the day-spine with zero rows for empty days (Requirement 1.3) and row
/// construction.
///
/// Ordering is applied by the caller through [`crate::cost_reporting::order_rows`]
/// (cost desc + dimension asc; day rows keep calendar order).
pub fn fetch_report(
    conn: &Connection,
    params: &ValidatedCostReportParams,
) -> Result<Vec<ReportRow>, QueryError> {
    let columns = LedgerColumns::probe(conn)?;
    let (where_sql, bind_values) = where_clause(
        &columns,
        params.range,
        params.provider.as_deref(),
        params.key_id.as_deref(),
    );
    let param_refs: Vec<&dyn rusqlite::ToSql> =
        bind_values.iter().map(|value| value.as_ref()).collect();

    // Day-grouping helpers: compute the spine and its window bounds ONCE
    // (the bounds derive from the spine, so merge_day_spine receives both
    // instead of recomputing them per call).
    let day_spine = day_spine(params.range);
    let day_bounds = day_window_bounds(&day_spine, params.range);

    let dimension_expr = match params.group_by {
        GroupBy::Day => "strftime('%Y-%m-%d', timestamp, 'unixepoch')".to_string(),
        GroupBy::Provider => "provider".to_string(),
        GroupBy::Model => "model".to_string(),
        GroupBy::VirtualKey => columns.key_dimension_expr(),
    };

    // Cast the day dimension to TEXT so the row mapper's String read works
    // uniformly across groupings.
    let dimension_select = if params.group_by == GroupBy::Day {
        format!("CAST({dimension_expr} AS TEXT)")
    } else {
        dimension_expr
    };

    let sql = format!(
        "SELECT {dimension_select} AS dimension, \
COUNT(*) AS requests, \
{input_sum} AS input_tokens, \
{output_sum} AS output_tokens, \
{cache_read_sum} AS cache_read_tokens, \
{cache_creation_sum} AS cache_creation_tokens, \
{reasoning_sum} AS reasoning_tokens, \
SUM(CAST(ROUND(cost * 1000000) AS INTEGER)) AS cost_micro_usd, \
{unpriced_requests} AS cost_unpriced_requests \
FROM requests{where_sql} \
GROUP BY dimension",
        input_sum = columns.token_sum(columns.input_tokens, "input_tokens"),
        output_sum = columns.token_sum(columns.output_tokens, "output_tokens"),
        cache_read_sum = columns.token_sum(columns.cache_read_tokens, "cache_read_tokens"),
        cache_creation_sum = columns.token_sum(columns.cache_creation_tokens, "cache_creation_tokens"),
        reasoning_sum = columns.token_sum(columns.reasoning_tokens, "reasoning_tokens"),
        unpriced_requests = columns.unpriced_count_expr(),
    );

    let mut stmt = conn.prepare(&sql)?;
    let aggregated: Vec<(String, i64, i64, i64, i64, i64, i64, i64, i64)> = stmt
        .query_map(param_refs.as_slice(), |row| {
            Ok((
                row.get(0)?,
                row.get(1)?,
                row.get(2)?,
                row.get(3)?,
                row.get(4)?,
                row.get(5)?,
                row.get(6)?,
                row.get(7)?,
                row.get(8)?,
            ))
        })?
        .collect::<Result<Vec<_>, _>>()?;

    let mut rows: Vec<ReportRow> = aggregated
        .into_iter()
        .map(|(dimension, requests, input_tokens, output_tokens, cache_read, cache_creation, reasoning, cost_micro_usd, unpriced)| {
            let (window_start, window_end) = match params.group_by {
                GroupBy::Day => day_bounds
                    .get(&dimension)
                    .copied()
                    .unwrap_or((params.range.start_secs, params.range.end_secs)),
                _ => (params.range.start_secs, params.range.end_secs),
            };
            ReportRow {
                dimension,
                window_start,
                window_end,
                requests,
                input_tokens,
                output_tokens,
                cache_read_tokens: cache_read,
                cache_creation_tokens: cache_creation,
                reasoning_tokens: reasoning,
            cost_micro_usd,
            cost_unpriced_requests: unpriced,
        }
    })
    .collect();

    if params.group_by == GroupBy::Day {
        merge_day_spine(&mut rows, day_spine, &day_bounds, params.range);
    }

    Ok(rows)
}

/// UTC calendar days covering the range: floor(start) to the day before the
/// exclusive end bound when end is exactly midnight. The spine is the full
/// calendar day containing each boundary (a range ending mid-day still
/// includes that day's row, matching "one row per UTC day in the range").
fn day_spine(range: ValidatedRange) -> Vec<String> {
    let start_day = floor_to_utc_day(range.start_secs);
    // A range ending exactly on midnight must not produce a trailing empty
    // day: `end < end_day + day` when end is midnight of end_day.
    let end_day = if is_utc_midnight(range.end_secs) {
        floor_to_utc_day(range.end_secs) - 86_400
    } else {
        floor_to_utc_day(range.end_secs)
    };

    let mut days = Vec::with_capacity(((end_day - start_day) / 86_400 + 1) as usize);
    let mut day = start_day;
    while day <= end_day {
        days.push(day_string(day));
        day += 86_400;
    }
    days
}

/// [day "YYYY-MM-DD" -> (window_start, window_end)] bounds: midnight of the
/// day to midnight of the next, clamped to the report range. Derived from
/// the already-computed spine so a day report builds each day's bounds
/// exactly once.
fn day_window_bounds(
    spine: &[String],
    range: ValidatedRange,
) -> std::collections::BTreeMap<String, (i64, i64)> {
    let mut bounds = std::collections::BTreeMap::new();
    for day in spine {
        let start = utc_day_epoch(day).unwrap_or(range.start_secs);
        let end = start + 86_400;
        bounds.insert(
            day.clone(),
            (
                start.max(range.start_secs),
                end.min(range_end_clamped(range)),
            ),
        );
    }
    bounds
}

/// Exclusive end for day windows: a report `end` beyond the day's own end is
/// irrelevant (per-day windows never exceed the day), but an end exactly at
/// midnight rolls into the previous day's window per the spine rule.
fn range_end_clamped(range: ValidatedRange) -> i64 {
    if is_utc_midnight(range.end_secs) {
        range.end_secs
    } else {
        floor_to_utc_day(range.end_secs) + 86_400
    }
}

fn floor_to_utc_day(secs: i64) -> i64 {
    secs.div_euclid(86_400) * 86_400
}

fn is_utc_midnight(secs: i64) -> bool {
    secs.rem_euclid(86_400) == 0
}

fn day_string(day_epoch: i64) -> String {
    chrono::DateTime::from_timestamp(day_epoch, 0)
        .map(|dt| dt.format("%Y-%m-%d").to_string())
        .unwrap_or_else(|| day_epoch.to_string())
}

fn utc_day_epoch(day: &str) -> Option<i64> {
    chrono::NaiveDate::parse_from_str(day, "%Y-%m-%d")
        .ok()
        .and_then(|date| date.and_hms_opt(0, 0, 0))
        .map(|time| time.and_utc().timestamp())
}

/// Fill zero rows for empty days and keep calendar order (Requirement 1.3):
/// every spine day present, aggregated values where they exist, zeros
/// otherwise. The result is sorted by dimension ascending (day strings sort
/// chronologically). Receives the caller's precomputed window bounds — the
/// same map `fetch_report` used for its rows — so a day report computes the
/// bounds once instead of once per aggregation pass.
fn merge_day_spine(
    rows: &mut Vec<ReportRow>,
    spine: Vec<String>,
    bounds: &std::collections::BTreeMap<String, (i64, i64)>,
    range: ValidatedRange,
) {
    let mut by_day: std::collections::BTreeMap<String, ReportRow> =
        rows.drain(..).map(|row| (row.dimension.clone(), row)).collect();

    for day in spine {
        let (window_start, window_end) = bounds
            .get(&day)
            .copied()
            .unwrap_or((range.start_secs, range.end_secs));
        by_day
            .entry(day.clone())
            .or_insert_with(|| ReportRow {
                dimension: day,
                window_start,
                window_end,
                requests: 0,
                input_tokens: 0,
                output_tokens: 0,
                cache_read_tokens: 0,
                cache_creation_tokens: 0,
                reasoning_tokens: 0,
                cost_micro_usd: 0,
                cost_unpriced_requests: 0,
            });
    }

    *rows = by_day.into_values().collect();
}
