//! Ledger daily aggregates for performance alerts (spec:
//! performance-alerts, Component 2).
//!
//! One grouped read-only SELECT per scope dimension, returning
//! per-day rows for the 8-day evaluation window (evaluated day plus
//! the 7-day baseline): gateway (a single `"gateway"` group),
//! provider (`GROUP BY provider`), and model group (`GROUP BY
//! requested_model`, NULL/blank names excluded). The SELECT reuses
//! the `fetch_trailing_spend` skeleton — timestamp range bounds on
//! `idx_requests_timestamp`, `path NOT LIKE '%#attempt'` marker-row
//! exclusion, `strftime('%Y-%m-%d', timestamp, 'unixepoch')` UTC-day
//! bucketing — and the cost-reporting `LedgerColumns` probe for the
//! micro-USD spend expression, so spend math stays byte-identical
//! with `cost_reporting::query`.
//!
//! Index decision (design D6, decided once by the implementation-time
//! benchmark, scratch test since deleted): over a representative 8-day
//! ledger (50_000 rows, 50 providers, 200 models) the model-group
//! aggregate completes in ~95ms via the plain `idx_requests_timestamp`
//! range scan (Provider ~94ms, Gateway ~89ms) — far under the 500ms
//! bar — so NO `requested_model` index ships.
//!
//! Call discipline: callers run this via `tokio::task::spawn_blocking`
//! exactly like `cost_reporting::query`; these functions take a
//! `&Connection` and never open databases themselves. Parameterized
//! throughout — values never concatenate into SQL.
//!
//! _Requirements: 2.1, 2.2_

use chrono::{Days, NaiveDate};
use rusqlite::{params, Connection};

use crate::cost_reporting::query::LedgerColumns;
use crate::performance_alerts::state::{DailyStats, Scope};

/// Query failure for the ledger aggregates. Mirrors the
/// cost-reporting error shape: SQL failures surface as errors and
/// never panic.
#[derive(Debug, thiserror::Error)]
#[error("performance alerts query failed: {0}")]
pub struct QueryError(#[from] rusqlite::Error);

/// Scope dimension one [`fetch_daily_stats`] call aggregates over.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Dimension {
    /// One group for the whole gateway (constant name `"gateway"`).
    Gateway,
    /// One group per `provider` name.
    Provider,
    /// One group per `requested_model` name (NULL/blank excluded).
    ModelGroup,
}

impl Dimension {
    /// The evaluation-core scope the dimension's rows belong to.
    fn scope(&self) -> Scope {
        match self {
            Dimension::Gateway => Scope::Gateway,
            Dimension::Provider => Scope::Provider,
            Dimension::ModelGroup => Scope::ModelGroup,
        }
    }
}

/// Identity of one aggregated scope: the dimension plus the scope's
/// NAME (provider or model-group name; the literal `"gateway"` for
/// the gateway scope) — names only, never ids.
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub struct ScopeKey {
    pub scope: Scope,
    pub name: String,
}

/// Aggregate per-day stats for one dimension over
/// `[window_start, window_end]` (both UTC days inclusive; the SQL
/// bound is the midnight after `window_end`, exclusive). Returns one
/// `(ScopeKey, DailyStats)` row per (scope name, UTC day) with data;
/// empty days are absent (the evaluation core treats them as zero
/// traffic). Read-only, parameterized; run via `spawn_blocking` at
/// the call site (Component 3), matching `cost_reporting::query`.
pub fn fetch_daily_stats(
    conn: &Connection,
    dimension: Dimension,
    window_start: NaiveDate,
    window_end: NaiveDate,
) -> Result<Vec<(ScopeKey, DailyStats)>, QueryError> {
    let columns = LedgerColumns::probe(conn)?;
    let start_secs = day_start_epoch(window_start);
    let end_secs = day_start_epoch(
        window_end
            .checked_add_days(Days::new(1))
            .expect("day after window_end always constructs"),
    );

    let (scope_expr, extra_where) = match dimension {
        Dimension::Gateway => ("'gateway'", ""),
        Dimension::Provider => ("provider", ""),
        Dimension::ModelGroup => (
            "requested_model",
            " AND requested_model IS NOT NULL AND TRIM(requested_model) <> ''",
        ),
    };

    let sql = format!(
        "SELECT {scope_expr} AS scope_name, \
         strftime('%Y-%m-%d', timestamp, 'unixepoch') AS day, \
         {cost_micro_usd} AS cost_micro_usd, \
         COUNT(*) AS requests, \
         COALESCE(SUM(CASE WHEN status_code >= 400 THEN 1 ELSE 0 END), 0) AS errors, \
         COALESCE(SUM(duration_ms), 0) AS duration_ms_sum \
         FROM requests \
         WHERE timestamp >= ?1 AND timestamp < ?2 AND path NOT LIKE '%#attempt'{extra_where} \
         GROUP BY scope_name, day \
         ORDER BY scope_name, day",
        cost_micro_usd = columns.cost_micro_usd_sum(),
    );

    let scope = dimension.scope();
    let mut stmt = conn.prepare(&sql)?;
    let aggregated = stmt
        .query_map(params![start_secs, end_secs], |row| {
            let day = parse_day(&row.get::<_, String>(1)?)?;
            Ok((
                row.get::<_, String>(0)?,
                day,
                row.get::<_, i64>(2)?,
                row.get::<_, i64>(3)?,
                row.get::<_, i64>(4)?,
                row.get::<_, i64>(5)?,
            ))
        })?
        .collect::<Result<Vec<_>, _>>()?;

    Ok(aggregated
        .into_iter()
        .map(|(name, day, cost_micro_usd, requests, errors, duration_ms_sum)| {
            (
                ScopeKey { scope, name },
                DailyStats {
                    day,
                    cost_micro_usd,
                    requests,
                    errors,
                    duration_ms_sum,
                },
            )
        })
        .collect())
}

/// UTC midnight of `day` as epoch seconds.
fn day_start_epoch(day: NaiveDate) -> i64 {
    day.and_hms_opt(0, 0, 0)
        .expect("UTC midnight always constructs")
        .and_utc()
        .timestamp()
}

/// Parse a `strftime('%Y-%m-%d', ...)` bucket into a [`NaiveDate`];
/// the format is fixed, so a malformed value surfaces as a SQL
/// conversion error rather than a panic.
fn parse_day(value: &str) -> Result<NaiveDate, rusqlite::Error> {
    NaiveDate::parse_from_str(value, "%Y-%m-%d").map_err(|error| {
        rusqlite::Error::FromSqlConversionFailure(
            1,
            rusqlite::types::Type::Text,
            Box::new(error),
        )
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    use crate::config::LoggingConfig;
    use crate::logger::RequestLogger;

    const DAY: i64 = 86_400;
    const DAY_ZERO: i64 = 1_767_225_600;
    const PATH_OK: &str = "/v1/chat/completions";
    const PATH_ATTEMPT: &str = "/v1/chat/completions#attempt";

    fn jan(day: u32) -> NaiveDate {
        NaiveDate::from_ymd_opt(2026, 1, day).expect("fixed January 2026 date")
    }

    struct LedgerFixture {
        _logger: RequestLogger,
        _dir: tempfile::TempDir,
        db_path: String,
    }

    fn create_ledger() -> LedgerFixture {
        let dir = tempfile::tempdir().expect("temp dir for isolated logs.db");
        let db_path = dir.path().join("logs.db").to_string_lossy().into_owned();
        let logger = RequestLogger::new(LoggingConfig {
            database_path: db_path.clone(),
            retention_days: 0,
            ..Default::default()
        })
        .expect("create isolated request logger");
        // Field order matters: `_logger` (declared first) drops before
        // `_dir`, so the writer thread's SQLite handle is closed before
        // the TempDir cleanup deletes the database file.
        LedgerFixture {
            _logger: logger,
            _dir: dir,
            db_path,
        }
    }

    struct SeedRow {
        timestamp: i64,
        provider: &'static str,
        model: &'static str,
        requested_model: Option<&'static str>,
        status_code: i64,
        duration_ms: i64,
        cost: f64,
        path: &'static str,
    }

    fn seed_rows(conn: &Connection, rows: &[SeedRow]) {
        conn.execute_batch("BEGIN").expect("begin seed transaction");
        for (index, row) in rows.iter().enumerate() {
            conn.execute(
                "INSERT INTO requests \
                 (trace_id, timestamp, method, path, model, provider, \
                 status_code, duration_ms, cost, requested_model) \
                 VALUES (?1, ?2, 'POST', ?3, ?4, ?5, ?6, ?7, ?8, ?9)",
                params![
                    format!("trace-{index}"),
                    row.timestamp,
                    row.path,
                    row.model,
                    row.provider,
                    row.status_code,
                    row.duration_ms,
                    row.cost,
                    row.requested_model,
                ],
            )
            .expect("seed ledger row");
        }
        conn.execute_batch("COMMIT").expect("commit seed transaction");
    }

    fn stat_for(
        rows: &[(ScopeKey, DailyStats)],
        scope: Scope,
        name: &str,
        day: NaiveDate,
    ) -> DailyStats {
        rows.iter()
            .find(|(key, stats)| key.scope == scope && key.name == name && stats.day == day)
            .map(|(_, stats)| *stats)
            .unwrap_or_else(|| panic!("seeded ({name}, {day}) row present"))
    }

    fn scope_names(rows: &[(ScopeKey, DailyStats)]) -> Vec<String> {
        let mut names: Vec<String> = rows.iter().map(|(key, _)| key.name.clone()).collect();
        names.sort();
        names.dedup();
        names
    }

    #[test]
    fn per_dimension_aggregates_match_hand_computed_totals() {
        let fixture = create_ledger();
        let conn = Connection::open(&fixture.db_path).unwrap();
        seed_rows(
            &conn,
            &[
                // Jan 1: two openai gpt-4 rows (one 500) + one anthropic claude-3 row.
                SeedRow { timestamp: DAY_ZERO, provider: "openai", model: "gpt-4", requested_model: Some("gpt-4"), status_code: 200, duration_ms: 10, cost: 0.000_002, path: PATH_OK },
                SeedRow { timestamp: DAY_ZERO + 3_600, provider: "openai", model: "gpt-4", requested_model: Some("gpt-4"), status_code: 500, duration_ms: 100, cost: 0.000_003, path: PATH_OK },
                SeedRow { timestamp: DAY_ZERO + 7_200, provider: "anthropic", model: "claude", requested_model: Some("claude-3"), status_code: 200, duration_ms: 20, cost: 0.000_001, path: PATH_OK },
                // Jan 2: one openai 400 + one NULL-name anthropic row + one blank-name anthropic row.
                SeedRow { timestamp: DAY_ZERO + DAY, provider: "openai", model: "gpt-4", requested_model: Some("gpt-4"), status_code: 400, duration_ms: 50, cost: 0.000_001, path: PATH_OK },
                SeedRow { timestamp: DAY_ZERO + DAY + 3_600, provider: "anthropic", model: "claude", requested_model: None, status_code: 200, duration_ms: 30, cost: 0.000_001, path: PATH_OK },
                SeedRow { timestamp: DAY_ZERO + DAY + 7_200, provider: "anthropic", model: "claude", requested_model: Some("   "), status_code: 200, duration_ms: 40, cost: 0.000_002, path: PATH_OK },
                // Out-of-window boundaries: the day before the window and the midnight right after Jan 8.
                SeedRow { timestamp: DAY_ZERO - DAY, provider: "openai", model: "gpt-4", requested_model: Some("gpt-4"), status_code: 200, duration_ms: 10, cost: 0.000_005, path: PATH_OK },
                SeedRow { timestamp: DAY_ZERO + 8 * DAY, provider: "openai", model: "gpt-4", requested_model: Some("gpt-4"), status_code: 200, duration_ms: 10, cost: 0.000_005, path: PATH_OK },
            ],
        );

        let gateway = fetch_daily_stats(&conn, Dimension::Gateway, jan(1), jan(8)).unwrap();
        assert_eq!(gateway.len(), 2);
        assert_eq!(scope_names(&gateway), vec!["gateway"]);
        let jan1 = stat_for(&gateway, Scope::Gateway, "gateway", jan(1));
        assert_eq!(
            (jan1.requests, jan1.errors, jan1.cost_micro_usd, jan1.duration_ms_sum),
            (3, 1, 6, 130)
        );
        let jan2 = stat_for(&gateway, Scope::Gateway, "gateway", jan(2));
        assert_eq!(
            (jan2.requests, jan2.errors, jan2.cost_micro_usd, jan2.duration_ms_sum),
            (3, 1, 4, 120)
        );

        let providers = fetch_daily_stats(&conn, Dimension::Provider, jan(1), jan(8)).unwrap();
        assert_eq!(providers.len(), 4);
        assert_eq!(scope_names(&providers), vec!["anthropic", "openai"]);
        let openai_jan1 = stat_for(&providers, Scope::Provider, "openai", jan(1));
        assert_eq!(
            (openai_jan1.requests, openai_jan1.errors, openai_jan1.cost_micro_usd, openai_jan1.duration_ms_sum),
            (2, 1, 5, 110)
        );
        let openai_jan2 = stat_for(&providers, Scope::Provider, "openai", jan(2));
        assert_eq!(
            (openai_jan2.requests, openai_jan2.errors, openai_jan2.cost_micro_usd, openai_jan2.duration_ms_sum),
            (1, 1, 1, 50)
        );
        let anthropic_jan1 = stat_for(&providers, Scope::Provider, "anthropic", jan(1));
        assert_eq!(
            (anthropic_jan1.requests, anthropic_jan1.errors, anthropic_jan1.cost_micro_usd, anthropic_jan1.duration_ms_sum),
            (1, 0, 1, 20)
        );
        let anthropic_jan2 = stat_for(&providers, Scope::Provider, "anthropic", jan(2));
        assert_eq!(
            (anthropic_jan2.requests, anthropic_jan2.errors, anthropic_jan2.cost_micro_usd, anthropic_jan2.duration_ms_sum),
            (2, 0, 3, 70)
        );

        let groups = fetch_daily_stats(&conn, Dimension::ModelGroup, jan(1), jan(8)).unwrap();
        assert_eq!(groups.len(), 3);
        assert_eq!(scope_names(&groups), vec!["claude-3", "gpt-4"]);
        let gpt4_jan1 = stat_for(&groups, Scope::ModelGroup, "gpt-4", jan(1));
        assert_eq!(
            (gpt4_jan1.requests, gpt4_jan1.errors, gpt4_jan1.cost_micro_usd, gpt4_jan1.duration_ms_sum),
            (2, 1, 5, 110)
        );
        let gpt4_jan2 = stat_for(&groups, Scope::ModelGroup, "gpt-4", jan(2));
        assert_eq!(
            (gpt4_jan2.requests, gpt4_jan2.errors, gpt4_jan2.cost_micro_usd, gpt4_jan2.duration_ms_sum),
            (1, 1, 1, 50)
        );
        let claude_jan1 = stat_for(&groups, Scope::ModelGroup, "claude-3", jan(1));
        assert_eq!(
            (claude_jan1.requests, claude_jan1.errors, claude_jan1.cost_micro_usd, claude_jan1.duration_ms_sum),
            (1, 0, 1, 20)
        );
        assert!(
            !groups
                .iter()
                .any(|(key, stats)| key.name == "claude-3" && stats.day == jan(2)),
            "NULL/blank-name rows never produce model-group days"
        );
    }

    #[test]
    fn attempt_rows_are_excluded_from_every_dimension() {
        let fixture = create_ledger();
        let conn = Connection::open(&fixture.db_path).unwrap();
        seed_rows(
            &conn,
            &[
                SeedRow { timestamp: DAY_ZERO, provider: "openai", model: "gpt-4", requested_model: Some("gpt-4"), status_code: 200, duration_ms: 10, cost: 0.000_002, path: PATH_OK },
                SeedRow { timestamp: DAY_ZERO + 3_600, provider: "openai", model: "gpt-4", requested_model: Some("gpt-4"), status_code: 500, duration_ms: 70, cost: 0.000_004, path: PATH_ATTEMPT },
                SeedRow { timestamp: DAY_ZERO + 7_200, provider: "anthropic", model: "claude", requested_model: Some("claude-3"), status_code: 502, duration_ms: 80, cost: 0.000_005, path: PATH_ATTEMPT },
            ],
        );

        for dimension in [Dimension::Gateway, Dimension::Provider, Dimension::ModelGroup] {
            let rows = fetch_daily_stats(&conn, dimension, jan(1), jan(8)).unwrap();
            assert_eq!(
                rows.iter().map(|(_, stats)| stats.requests).sum::<i64>(),
                1,
                "{dimension:?}: only the non-attempt row is aggregated"
            );
            assert_eq!(
                rows.iter().map(|(_, stats)| stats.errors).sum::<i64>(),
                0,
                "{dimension:?}: the 500-status attempt row never counts as an error"
            );
            assert_eq!(
                rows.iter().map(|(_, stats)| stats.cost_micro_usd).sum::<i64>(),
                2,
                "{dimension:?}: attempt-row cost is excluded"
            );
            assert_eq!(
                rows.iter().map(|(_, stats)| stats.duration_ms_sum).sum::<i64>(),
                10,
                "{dimension:?}: attempt-row duration is excluded"
            );
        }
    }

    #[test]
    fn null_and_blank_requested_model_rows_are_excluded_from_model_group_only() {
        let fixture = create_ledger();
        let conn = Connection::open(&fixture.db_path).unwrap();
        seed_rows(
            &conn,
            &[
                SeedRow { timestamp: DAY_ZERO, provider: "openai", model: "resolved-a", requested_model: None, status_code: 200, duration_ms: 10, cost: 0.000_001, path: PATH_OK },
                SeedRow { timestamp: DAY_ZERO + 3_600, provider: "openai", model: "resolved-b", requested_model: Some(""), status_code: 200, duration_ms: 10, cost: 0.000_001, path: PATH_OK },
                SeedRow { timestamp: DAY_ZERO + 7_200, provider: "openai", model: "resolved-c", requested_model: Some("   "), status_code: 200, duration_ms: 10, cost: 0.000_001, path: PATH_OK },
                SeedRow { timestamp: DAY_ZERO + 10_800, provider: "openai", model: "resolved-d", requested_model: Some("real-model"), status_code: 200, duration_ms: 10, cost: 0.000_001, path: PATH_OK },
            ],
        );

        let groups = fetch_daily_stats(&conn, Dimension::ModelGroup, jan(1), jan(8)).unwrap();
        assert_eq!(scope_names(&groups), vec!["real-model"]);
        let stats = stat_for(&groups, Scope::ModelGroup, "real-model", jan(1));
        assert_eq!(
            (stats.requests, stats.errors, stats.cost_micro_usd, stats.duration_ms_sum),
            (1, 0, 1, 10)
        );

        let gateway = fetch_daily_stats(&conn, Dimension::Gateway, jan(1), jan(8)).unwrap();
        let gateway_stats = stat_for(&gateway, Scope::Gateway, "gateway", jan(1));
        assert_eq!(
            (gateway_stats.requests, gateway_stats.cost_micro_usd, gateway_stats.duration_ms_sum),
            (4, 4, 40),
            "gateway scope still counts NULL/blank-name rows"
        );

        let providers = fetch_daily_stats(&conn, Dimension::Provider, jan(1), jan(8)).unwrap();
        let openai = stat_for(&providers, Scope::Provider, "openai", jan(1));
        assert_eq!(
            (openai.requests, openai.cost_micro_usd, openai.duration_ms_sum),
            (4, 4, 40),
            "provider scope still counts NULL/blank-name rows"
        );
    }

    #[test]
    fn gateway_dimension_returns_exactly_one_scope_key() {
        let fixture = create_ledger();
        let conn = Connection::open(&fixture.db_path).unwrap();
        seed_rows(
            &conn,
            &[
                SeedRow { timestamp: DAY_ZERO, provider: "openai", model: "gpt-4", requested_model: Some("gpt-4"), status_code: 200, duration_ms: 10, cost: 0.000_001, path: PATH_OK },
                SeedRow { timestamp: DAY_ZERO + DAY, provider: "anthropic", model: "claude", requested_model: Some("claude-3"), status_code: 200, duration_ms: 10, cost: 0.000_001, path: PATH_OK },
                SeedRow { timestamp: DAY_ZERO + 2 * DAY, provider: "gemini", model: "gemini", requested_model: Some("gemini-2"), status_code: 200, duration_ms: 10, cost: 0.000_001, path: PATH_OK },
            ],
        );

        let rows = fetch_daily_stats(&conn, Dimension::Gateway, jan(1), jan(8)).unwrap();
        assert_eq!(scope_names(&rows), vec!["gateway"]);
        assert_eq!(rows.len(), 3, "one row per day with traffic");
        assert!(rows
            .iter()
            .all(|(key, _)| key.scope == Scope::Gateway && key.name == "gateway"));
    }

    #[test]
    fn reopening_an_existing_database_is_a_noop() {
        let fixture = create_ledger();
        {
            let conn = Connection::open(&fixture.db_path).unwrap();
            seed_rows(
                &conn,
                &[SeedRow { timestamp: DAY_ZERO, provider: "openai", model: "gpt-4", requested_model: Some("gpt-4"), status_code: 200, duration_ms: 10, cost: 0.000_002, path: PATH_OK }],
            );
        }

        let reopened = RequestLogger::new(LoggingConfig {
            database_path: fixture.db_path.clone(),
            retention_days: 0,
            ..Default::default()
        });
        assert!(reopened.is_ok(), "guarded schema re-open must not fail or mutate");

        let conn = Connection::open(&fixture.db_path).unwrap();
        let rows = fetch_daily_stats(&conn, Dimension::Gateway, jan(1), jan(8)).unwrap();
        let stats = stat_for(&rows, Scope::Gateway, "gateway", jan(1));
        assert_eq!(
            (stats.requests, stats.errors, stats.cost_micro_usd, stats.duration_ms_sum),
            (1, 0, 2, 10),
            "seeded rows survive the idempotent re-open unchanged"
        );
    }
}
