//! Per-key ledger aggregates for usage-anomaly alerts (spec:
//! usage-anomaly-alerts, Task 3).
//!
//! Three grouped read-only SELECTs over the requests ledger, one
//! per behavioral input of the evaluation core: per-key daily
//! rows ([`fetch_key_daily_stats`]), per-key six-bucket 4-hour
//! UTC histograms ([`fetch_key_hour_histogram`]), and per-key
//! requested-model counts ([`fetch_key_model_usage`]). Each
//! SELECT reuses the `fetch_daily_stats` skeleton — timestamp
//! range bounds on `idx_requests_timestamp`, `path NOT LIKE
//! '%#attempt'` marker-row exclusion, `strftime('%Y-%m-%d',
//! timestamp, 'unixepoch')` UTC-day bucketing — plus `key_id IS
//! NOT NULL` so anonymous traffic never forms a per-key group.
//! Spend math reuses the cost-reporting `LedgerColumns` probe so
//! it stays byte-identical with `cost_reporting::query`.
//!
//! `validation_failures` counts rows with `status_code IN
//! (400,401,403)` and no output tokens — 429 is deliberately
//! excluded (it is rate limiting, not validation), and a NULL
//! `output_tokens` counts as zero (no usage reported means the
//! request produced nothing).
//!
//! Index decision (design D6, spend-pace rule: unconditional
//! index only if a benchmark shows need): NO extra index ships.
//! These aggregates share the perf-alerts shape whose D6
//! benchmark ran a representative 8-day 50_000-row ledger through
//! the plain `idx_requests_timestamp` range scan in ~95ms — far
//! under the 500ms bar — and per-key grouping is additionally
//! served by the existing `idx_requests_key_id_timestamp`, so an
//! additional index was deemed unnecessary.
//!
//! Call discipline: callers run these via
//! `tokio::task::spawn_blocking` exactly like
//! `cost_reporting::query`; the functions take a `&Connection`
//! and never open databases themselves. Parameterized
//! throughout — values never concatenate into SQL.
//!
//! _Requirements: 1.10, 2.3, 4.2, 4.7_

use chrono::{Days, NaiveDate};
use rusqlite::{params, Connection};

use crate::cost_reporting::query::LedgerColumns;
use crate::usage_anomaly_alerts::state::{KeyDailyStats, KeyHourHistogram, KeyModelUsage};

/// Query failure for the ledger aggregates. Mirrors the
/// cost-reporting error shape: SQL failures surface as errors and
/// never panic.
#[derive(Debug, thiserror::Error)]
#[error("usage anomaly alerts query failed: {0}")]
pub struct QueryError(#[from] rusqlite::Error);

/// Exclusive upper-bound epoch for `[window_start, window_end]`
/// (both UTC days inclusive): the midnight after `window_end`.
fn window_end_epoch(window_end: NaiveDate) -> i64 {
    window_end
        .checked_add_days(Days::new(1))
        .expect("day after window_end always constructs")
        .and_hms_opt(0, 0, 0)
        .expect("UTC midnight always constructs")
        .and_utc()
        .timestamp()
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

/// Aggregate per-key daily stats over `[window_start, window_end]`
/// (both UTC days inclusive; the SQL bound is the midnight after
/// `window_end`, exclusive). Returns one `(key_id, KeyDailyStats)`
/// row per (key, UTC day) with data; empty days are absent (the
/// evaluation core treats them as zero traffic). `cost_micro_usd`
/// comes from the `LedgerColumns` probe expression so spend math
/// matches `cost_reporting::query` exactly.
///
/// Read-only, parameterized; run via `spawn_blocking` at the call
/// site, matching `cost_reporting::query`.
pub fn fetch_key_daily_stats(
    conn: &Connection,
    window_start: NaiveDate,
    window_end: NaiveDate,
) -> Result<Vec<(String, KeyDailyStats)>, QueryError> {
    let columns = LedgerColumns::probe(conn)?;
    let start_secs = day_start_epoch(window_start);
    let end_secs = window_end_epoch(window_end);

    let sql = format!(
        "SELECT key_id AS key_id, \
        strftime('%Y-%m-%d', timestamp, 'unixepoch') AS day, \
        {cost_micro_usd} AS cost_micro_usd, \
        COUNT(*) AS requests, \
        COALESCE(SUM(CASE WHEN status_code IN (400, 401, 403) \
        AND COALESCE(output_tokens, 0) = 0 THEN 1 ELSE 0 END), 0) AS validation_failures, \
        COALESCE(SUM(COALESCE(input_tokens, 0)), 0) AS input_tokens, \
        COALESCE(SUM(COALESCE(output_tokens, 0)), 0) AS output_tokens \
        FROM requests \
        WHERE timestamp >= ?1 AND timestamp < ?2 AND path NOT LIKE '%#attempt' \
        AND key_id IS NOT NULL \
        GROUP BY key_id, day \
        ORDER BY key_id, day",
        cost_micro_usd = columns.cost_micro_usd_sum(),
    );

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
                row.get::<_, i64>(6)?,
            ))
        })?
        .collect::<Result<Vec<_>, _>>()?;

    Ok(aggregated
        .into_iter()
        .map(
            |(key_id, day, cost_micro_usd, requests, validation_failures, input_tokens, output_tokens)| {
                (
                    key_id,
                    KeyDailyStats {
                        day,
                        cost_micro_usd,
                        requests,
                        validation_failures,
                        input_tokens,
                        output_tokens,
                    },
                )
            },
        )
        .collect())
}

/// Aggregate per-key request counts across the six 4-hour UTC
/// buckets (index 0 = 00:00-03:59, ..., index 5 = 20:00-23:59),
/// day-resolved over `[window_start, window_end]`. Returns one
/// `(key_id, day, KeyHourHistogram)` row per (key, UTC day) with
/// in-window traffic; buckets without requests read 0. The day
/// resolution lets the emission cycle cover the whole 8-day
/// window in ONE query (Req 4.7) and split observed vs baseline
/// in memory. Read-only, parameterized; run via `spawn_blocking`
/// at the call site.
pub fn fetch_key_hour_histogram(
    conn: &Connection,
    window_start: NaiveDate,
    window_end: NaiveDate,
) -> Result<Vec<(String, NaiveDate, KeyHourHistogram)>, QueryError> {
    let start_secs = day_start_epoch(window_start);
    let end_secs = window_end_epoch(window_end);

    let sql = "\
        SELECT key_id AS key_id, \
        strftime('%Y-%m-%d', timestamp, 'unixepoch') AS day, \
        CAST(strftime('%H', timestamp, 'unixepoch') AS INTEGER) / 4 AS bucket, \
        COUNT(*) AS requests \
        FROM requests \
        WHERE timestamp >= ?1 AND timestamp < ?2 AND path NOT LIKE '%#attempt' \
        AND key_id IS NOT NULL \
        GROUP BY key_id, day, bucket \
        ORDER BY key_id, day, bucket";

    let mut stmt = conn.prepare(sql)?;
    let buckets = stmt
        .query_map(params![start_secs, end_secs], |row| {
            Ok((
                row.get::<_, String>(0)?,
                row.get::<_, String>(1)?,
                row.get::<_, i64>(2)?,
                row.get::<_, i64>(3)?,
            ))
        })?
        .collect::<Result<Vec<_>, _>>()?;

    let mut rows: Vec<(String, NaiveDate, KeyHourHistogram)> = Vec::new();
    for (key_id, day, bucket, count) in buckets {
        let day = parse_day(&day)?;
        // `strftime('%H')` is 0-23, so integer division by 4 is
        // always one of the six bucket indices.
        let index = usize::try_from(bucket).expect("hour bucket index is non-negative") % 6;
        match rows.last_mut() {
            Some((key, last_day, histogram)) if *key == key_id && *last_day == day => {
                histogram.buckets[index] += count;
            }
            _ => {
                let mut histogram = KeyHourHistogram { buckets: [0; 6] };
                histogram.buckets[index] += count;
                rows.push((key_id, day, histogram));
            }
        }
    }
    Ok(rows)
}

/// Aggregate per-key `requested_model` request counts,
/// day-resolved over `[window_start, window_end]`. Returns one
/// `(key_id, day, KeyModelUsage)` row per (key, model, UTC day)
/// with data; NULL/blank model names are excluded (they carry no
/// new-model signal). The day resolution lets the emission cycle
/// cover the whole 8-day window in ONE query (Req 4.7) and split
/// observed vs baseline in memory. Read-only, parameterized; run
/// via `spawn_blocking` at the call site.
pub fn fetch_key_model_usage(
    conn: &Connection,
    window_start: NaiveDate,
    window_end: NaiveDate,
) -> Result<Vec<(String, NaiveDate, KeyModelUsage)>, QueryError> {
    let start_secs = day_start_epoch(window_start);
    let end_secs = window_end_epoch(window_end);

    let sql = "\
        SELECT key_id AS key_id, \
        strftime('%Y-%m-%d', timestamp, 'unixepoch') AS day, \
        requested_model AS model, \
        COUNT(*) AS requests \
        FROM requests \
        WHERE timestamp >= ?1 AND timestamp < ?2 AND path NOT LIKE '%#attempt' \
        AND key_id IS NOT NULL \
        AND requested_model IS NOT NULL AND TRIM(requested_model) <> '' \
        GROUP BY key_id, day, requested_model \
        ORDER BY key_id, day, requested_model";

    let mut stmt = conn.prepare(sql)?;
    let aggregated = stmt
        .query_map(params![start_secs, end_secs], |row| {
            Ok((
                row.get::<_, String>(0)?,
                row.get::<_, String>(1)?,
                row.get::<_, String>(2)?,
                row.get::<_, i64>(3)?,
            ))
        })?
        .collect::<Result<Vec<_>, _>>()?;

    let mut rows: Vec<(String, NaiveDate, KeyModelUsage)> = Vec::new();
    for (key_id, day, model, requests) in aggregated {
        let day = parse_day(&day)?;
        rows.push((key_id, day, KeyModelUsage { model, requests }));
    }
    Ok(rows)
}

#[cfg(test)]
mod tests {
    use super::*;

    use std::collections::HashMap;

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
        key_id: Option<&'static str>,
        requested_model: Option<&'static str>,
        status_code: i64,
        cost: f64,
        path: &'static str,
        input_tokens: Option<i64>,
        output_tokens: Option<i64>,
    }

    fn seed_rows(conn: &Connection, rows: &[SeedRow]) {
        conn.execute_batch("BEGIN").expect("begin seed transaction");
        for (index, row) in rows.iter().enumerate() {
            conn.execute(
                "INSERT INTO requests \
                (trace_id, timestamp, method, path, model, provider, \
                status_code, duration_ms, cost, requested_model, key_id, \
                input_tokens, output_tokens) \
                VALUES (?1, ?2, 'POST', ?3, 'gpt-4', 'openai', ?4, 10, ?5, ?6, ?7, ?8, ?9)",
                params![
                    format!("trace-{index}"),
                    row.timestamp,
                    row.path,
                    row.status_code,
                    row.cost,
                    row.requested_model,
                    row.key_id,
                    row.input_tokens,
                    row.output_tokens,
                ],
            )
            .expect("seed ledger row");
        }
        conn.execute_batch("COMMIT").expect("commit seed transaction");
    }

    fn daily_for(rows: &[(String, KeyDailyStats)], key: &str, day: NaiveDate) -> KeyDailyStats {
        rows.iter()
            .find(|(key_id, stats)| key_id == key && stats.day == day)
            .map(|(_, stats)| *stats)
            .unwrap_or_else(|| panic!("seeded ({key}, {day}) row present"))
    }

    fn key_ids(rows: &[(String, KeyDailyStats)]) -> Vec<String> {
        let mut names: Vec<String> = rows.iter().map(|(key, _)| key.clone()).collect();
        names.sort();
        names.dedup();
        names
    }

fn hist_for(
    rows: &[(String, NaiveDate, KeyHourHistogram)],
    key: &str,
    day: NaiveDate,
) -> KeyHourHistogram {
    let mut histogram = KeyHourHistogram { buckets: [0; 6] };
    let mut seen = false;
    for (key_id, row_day, hist) in rows {
        if *key_id == key {
            seen = true;
            if *row_day == day {
                for (bucket, count) in histogram.buckets.iter_mut().zip(hist.buckets.iter()) {
                    *bucket += count;
                }
            }
        }
    }
    if !seen {
        panic!("seeded ({key}) histogram present");
    }
    histogram
}

fn models_for(
    rows: &[(String, NaiveDate, KeyModelUsage)],
    key: &str,
) -> Vec<KeyModelUsage> {
    let mut per_model: HashMap<String, i64> = HashMap::new();
    for (key_id, _, usage) in rows {
        if key_id == key {
            *per_model.entry(usage.model.clone()).or_insert(0) += usage.requests;
        }
    }
    let mut models: Vec<KeyModelUsage> = per_model
        .into_iter()
        .map(|(model, requests)| KeyModelUsage { model, requests })
        .collect();
    models.sort_by(|a, b| a.model.cmp(&b.model));
    models
}

    #[test]
    fn daily_stats_match_hand_computed_totals() {
        let fixture = create_ledger();
        let conn = Connection::open(&fixture.db_path).unwrap();
        seed_rows(
            &conn,
            &[
                // Jan 1, key-a: 200 with tokens, a 400 with no output
                // (validation failure), a 429 with no output (excluded
                // from validation), and a 400 that DID emit tokens
                // (excluded — output_tokens = 0 is required).
                SeedRow { timestamp: DAY_ZERO, key_id: Some("key-a"), requested_model: Some("gpt-4"), status_code: 200, cost: 0.000_002, path: PATH_OK, input_tokens: Some(10), output_tokens: Some(20) },
                SeedRow { timestamp: DAY_ZERO + 3_600, key_id: Some("key-a"), requested_model: Some("gpt-4"), status_code: 400, cost: 0.0, path: PATH_OK, input_tokens: None, output_tokens: None },
                SeedRow { timestamp: DAY_ZERO + 7_200, key_id: Some("key-a"), requested_model: Some("gpt-4"), status_code: 429, cost: 0.0, path: PATH_OK, input_tokens: None, output_tokens: Some(0) },
                SeedRow { timestamp: DAY_ZERO + 10_800, key_id: Some("key-a"), requested_model: Some("gpt-4"), status_code: 400, cost: 0.0, path: PATH_OK, input_tokens: None, output_tokens: Some(5) },
                // Jan 1, key-b: one 403 with no usage reported.
                SeedRow { timestamp: DAY_ZERO + 3_600, key_id: Some("key-b"), requested_model: Some("gpt-4"), status_code: 403, cost: 0.000_001, path: PATH_OK, input_tokens: Some(1), output_tokens: None },
                // Jan 1, anonymous: never forms a per-key group.
                SeedRow { timestamp: DAY_ZERO, key_id: None, requested_model: Some("gpt-4"), status_code: 200, cost: 0.000_010, path: PATH_OK, input_tokens: Some(100), output_tokens: Some(100) },
                // Jan 2, key-a: one 401 (validation failure) + one 200.
                SeedRow { timestamp: DAY_ZERO + DAY, key_id: Some("key-a"), requested_model: Some("gpt-4"), status_code: 401, cost: 0.0, path: PATH_OK, input_tokens: None, output_tokens: None },
                SeedRow { timestamp: DAY_ZERO + DAY + 3_600, key_id: Some("key-a"), requested_model: Some("gpt-4"), status_code: 200, cost: 0.000_001, path: PATH_OK, input_tokens: Some(7), output_tokens: Some(9) },
                // Out-of-window boundaries: the day before the window
                // and the midnight right after Jan 8.
                SeedRow { timestamp: DAY_ZERO - DAY, key_id: Some("key-a"), requested_model: Some("gpt-4"), status_code: 200, cost: 0.000_005, path: PATH_OK, input_tokens: Some(50), output_tokens: Some(50) },
                SeedRow { timestamp: DAY_ZERO + 8 * DAY, key_id: Some("key-a"), requested_model: Some("gpt-4"), status_code: 200, cost: 0.000_005, path: PATH_OK, input_tokens: Some(50), output_tokens: Some(50) },
            ],
        );

        let rows = fetch_key_daily_stats(&conn, jan(1), jan(8)).unwrap();
        assert_eq!(rows.len(), 3);
        assert_eq!(key_ids(&rows), vec!["key-a", "key-b"]);

        let a_jan1 = daily_for(&rows, "key-a", jan(1));
        assert_eq!(
            (a_jan1.cost_micro_usd, a_jan1.requests, a_jan1.validation_failures, a_jan1.input_tokens, a_jan1.output_tokens),
            (2, 4, 1, 10, 25)
        );
        let a_jan2 = daily_for(&rows, "key-a", jan(2));
        assert_eq!(
            (a_jan2.cost_micro_usd, a_jan2.requests, a_jan2.validation_failures, a_jan2.input_tokens, a_jan2.output_tokens),
            (1, 2, 1, 7, 9)
        );
        let b_jan1 = daily_for(&rows, "key-b", jan(1));
        assert_eq!(
            (b_jan1.cost_micro_usd, b_jan1.requests, b_jan1.validation_failures, b_jan1.input_tokens, b_jan1.output_tokens),
            (1, 1, 1, 1, 0)
        );
    }

    #[test]
    fn attempt_rows_are_excluded_from_every_aggregate() {
        let fixture = create_ledger();
        let conn = Connection::open(&fixture.db_path).unwrap();
        seed_rows(
            &conn,
            &[
                SeedRow { timestamp: DAY_ZERO, key_id: Some("key-a"), requested_model: Some("gpt-4"), status_code: 200, cost: 0.000_002, path: PATH_OK, input_tokens: Some(10), output_tokens: Some(20) },
                // A 400 zero-output attempt row would count as a
                // validation failure if the marker exclusion failed.
                SeedRow { timestamp: DAY_ZERO + 3_600, key_id: Some("key-a"), requested_model: Some("gpt-4"), status_code: 400, cost: 0.0, path: PATH_ATTEMPT, input_tokens: None, output_tokens: None },
                SeedRow { timestamp: DAY_ZERO + 7_200, key_id: Some("key-a"), requested_model: Some("claude-3"), status_code: 502, cost: 0.0, path: PATH_ATTEMPT, input_tokens: None, output_tokens: None },
            ],
        );

        let daily = fetch_key_daily_stats(&conn, jan(1), jan(8)).unwrap();
        assert_eq!(daily.len(), 1);
        let stats = daily_for(&daily, "key-a", jan(1));
        assert_eq!(
            (stats.requests, stats.validation_failures, stats.cost_micro_usd, stats.input_tokens, stats.output_tokens),
            (1, 0, 2, 10, 20)
        );

        let histogram = fetch_key_hour_histogram(&conn, jan(1), jan(8)).unwrap();
        assert_eq!(histogram.iter().map(|(_, _, h)| h.buckets.iter().sum::<i64>()).sum::<i64>(), 1);

        let models = fetch_key_model_usage(&conn, jan(1), jan(8)).unwrap();
        assert_eq!(models.len(), 1);
        assert_eq!(
            models_for(&models, "key-a"),
            vec![KeyModelUsage { model: "gpt-4".to_string(), requests: 1 }]
        );
    }

    #[test]
    fn hour_histogram_buckets_four_hour_utc_edges() {
        let fixture = create_ledger();
        let conn = Connection::open(&fixture.db_path).unwrap();
        seed_rows(
            &conn,
            &[
                // key-a across bucket edges: 00:30 and 03:59:59 land in
                // bucket 0; 04:00 (twice) in bucket 1; 11:59 in bucket 2;
                // 12:00 and 15:59 in bucket 3; 16:00 in bucket 4; 23:59
                // in bucket 5.
                SeedRow { timestamp: DAY_ZERO + 1_800, key_id: Some("key-a"), requested_model: Some("gpt-4"), status_code: 200, cost: 0.0, path: PATH_OK, input_tokens: None, output_tokens: None },
                SeedRow { timestamp: DAY_ZERO + 14_399, key_id: Some("key-a"), requested_model: Some("gpt-4"), status_code: 200, cost: 0.0, path: PATH_OK, input_tokens: None, output_tokens: None },
                SeedRow { timestamp: DAY_ZERO + 14_400, key_id: Some("key-a"), requested_model: Some("gpt-4"), status_code: 200, cost: 0.0, path: PATH_OK, input_tokens: None, output_tokens: None },
                SeedRow { timestamp: DAY_ZERO + 14_400 + 60, key_id: Some("key-a"), requested_model: Some("gpt-4"), status_code: 200, cost: 0.0, path: PATH_OK, input_tokens: None, output_tokens: None },
                SeedRow { timestamp: DAY_ZERO + 43_140, key_id: Some("key-a"), requested_model: Some("gpt-4"), status_code: 200, cost: 0.0, path: PATH_OK, input_tokens: None, output_tokens: None },
                SeedRow { timestamp: DAY_ZERO + 43_200, key_id: Some("key-a"), requested_model: Some("gpt-4"), status_code: 200, cost: 0.0, path: PATH_OK, input_tokens: None, output_tokens: None },
                SeedRow { timestamp: DAY_ZERO + 57_540, key_id: Some("key-a"), requested_model: Some("gpt-4"), status_code: 200, cost: 0.0, path: PATH_OK, input_tokens: None, output_tokens: None },
                SeedRow { timestamp: DAY_ZERO + 57_600, key_id: Some("key-a"), requested_model: Some("gpt-4"), status_code: 200, cost: 0.0, path: PATH_OK, input_tokens: None, output_tokens: None },
                SeedRow { timestamp: DAY_ZERO + 82_740, key_id: Some("key-a"), requested_model: Some("gpt-4"), status_code: 200, cost: 0.0, path: PATH_OK, input_tokens: None, output_tokens: None },
                // key-b: single request at 20:00 -> bucket 5.
                SeedRow { timestamp: DAY_ZERO + 72_000, key_id: Some("key-b"), requested_model: Some("gpt-4"), status_code: 200, cost: 0.0, path: PATH_OK, input_tokens: None, output_tokens: None },
                // Anonymous row never forms a histogram group.
                SeedRow { timestamp: DAY_ZERO + 1_800, key_id: None, requested_model: Some("gpt-4"), status_code: 200, cost: 0.0, path: PATH_OK, input_tokens: None, output_tokens: None },
            ],
        );

        let rows = fetch_key_hour_histogram(&conn, jan(1), jan(1)).unwrap();
        assert_eq!(rows.len(), 2);
        assert_eq!(hist_for(&rows, "key-a", jan(1)).buckets, [2, 2, 1, 2, 1, 1]);
        assert_eq!(hist_for(&rows, "key-b", jan(1)).buckets, [0, 0, 0, 0, 0, 1]);
    }

    #[test]
    fn model_usage_excludes_null_and_blank_models() {
        let fixture = create_ledger();
        let conn = Connection::open(&fixture.db_path).unwrap();
        seed_rows(
            &conn,
            &[
                SeedRow { timestamp: DAY_ZERO, key_id: Some("key-a"), requested_model: Some("gpt-4"), status_code: 200, cost: 0.0, path: PATH_OK, input_tokens: None, output_tokens: None },
                SeedRow { timestamp: DAY_ZERO + 3_600, key_id: Some("key-a"), requested_model: Some("gpt-4"), status_code: 200, cost: 0.0, path: PATH_OK, input_tokens: None, output_tokens: None },
                SeedRow { timestamp: DAY_ZERO + 7_200, key_id: Some("key-a"), requested_model: Some("claude-3"), status_code: 200, cost: 0.0, path: PATH_OK, input_tokens: None, output_tokens: None },
                SeedRow { timestamp: DAY_ZERO + 10_800, key_id: Some("key-a"), requested_model: None, status_code: 200, cost: 0.0, path: PATH_OK, input_tokens: None, output_tokens: None },
                SeedRow { timestamp: DAY_ZERO + 14_400, key_id: Some("key-a"), requested_model: Some(""), status_code: 200, cost: 0.0, path: PATH_OK, input_tokens: None, output_tokens: None },
                SeedRow { timestamp: DAY_ZERO + 18_000, key_id: Some("key-a"), requested_model: Some(" "), status_code: 200, cost: 0.0, path: PATH_OK, input_tokens: None, output_tokens: None },
                SeedRow { timestamp: DAY_ZERO + 21_600, key_id: Some("key-b"), requested_model: Some("gpt-4"), status_code: 200, cost: 0.0, path: PATH_OK, input_tokens: None, output_tokens: None },
                // Out-of-window model usage never counts.
                SeedRow { timestamp: DAY_ZERO + 8 * DAY, key_id: Some("key-b"), requested_model: Some("claude-3"), status_code: 200, cost: 0.0, path: PATH_OK, input_tokens: None, output_tokens: None },
            ],
        );

        let rows = fetch_key_model_usage(&conn, jan(1), jan(8)).unwrap();
        assert_eq!(models_for(&rows, "key-a").len(), 2);
        assert_eq!(models_for(&rows, "key-b").len(), 1);
        assert_eq!(
            models_for(&rows, "key-a"),
            vec![
                KeyModelUsage { model: "claude-3".to_string(), requests: 1 },
                KeyModelUsage { model: "gpt-4".to_string(), requests: 2 },
            ]
        );
        assert_eq!(
            models_for(&rows, "key-b"),
            vec![KeyModelUsage { model: "gpt-4".to_string(), requests: 1 }]
        );
    }

    #[test]
    fn reopening_an_existing_database_is_a_noop() {
        let fixture = create_ledger();
        let conn = Connection::open(&fixture.db_path).unwrap();
        seed_rows(
            &conn,
            &[SeedRow { timestamp: DAY_ZERO, key_id: Some("key-a"), requested_model: Some("gpt-4"), status_code: 200, cost: 0.000_001, path: PATH_OK, input_tokens: Some(1), output_tokens: Some(1) }],
        );
        // A second logger over the same path must not disturb the
        // schema or the aggregates.
        RequestLogger::new(LoggingConfig {
            database_path: fixture.db_path.clone(),
            retention_days: 0,
            ..Default::default()
        })
        .expect("reopen existing ledger");

        let rows = fetch_key_daily_stats(&conn, jan(1), jan(8)).unwrap();
        let stats = daily_for(&rows, "key-a", jan(1));
        assert_eq!(
            (stats.requests, stats.cost_micro_usd, stats.input_tokens, stats.output_tokens),
            (1, 1, 1, 1)
        );
    }
}
