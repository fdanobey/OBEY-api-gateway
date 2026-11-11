use crate::common;

use ai_gateway::config::Config;
use ai_gateway::cost_reporting::{
    query::fetch_report, CostReport, CostReportParams, GroupBy, ReportRow,
};
use ai_gateway::gateway::GatewayServer;
use ai_gateway::logger::RequestLogger;
use axum::body::Body;
use axum::http::{Request, StatusCode};
use rusqlite::{Connection, OpenFlags};
use serde_json::{json, Value};
use common::{sample_log_entry, LogEntryOverrides, DAY_ZERO, LEGACY_REQUESTS_DDL};
use tower::ServiceExt;

const DAY: i64 = 86_400;
const GROUPINGS: [GroupBy; 4] = [
    GroupBy::Day,
    GroupBy::Provider,
    GroupBy::Model,
    GroupBy::VirtualKey,
];
const CSV_HEADER: &str = "dimension,requests,input_tokens,output_tokens,cache_read_tokens,cache_creation_tokens,reasoning_tokens,cost_usd,cost_unpriced_requests\r\n";
const LEGACY_COLUMNS: &str = "id, trace_id, timestamp, method, path, model, provider, status_code, duration_ms, cost, request_body, response_body, requested_model, responded_model";

fn isolated_config() -> Config {
    let mut config = common::budget_test_config(None, None);
    common::isolate_databases(&mut config);
    config.logging.retention_days = 0;
    config
}

fn read_connection(path: &str) -> Connection {
    let conn = Connection::open_with_flags(path, OpenFlags::SQLITE_OPEN_READ_ONLY).unwrap();
    conn.execute_batch("PRAGMA query_only=ON;").unwrap();
    conn
}

fn timestamp(seconds: i64) -> String {
    common::cost_reports_timestamp(seconds)
}

fn report(conn: &Connection, group_by: GroupBy) -> CostReport {
    let params = CostReportParams {
        start: Some(timestamp(DAY_ZERO)),
        end: Some(timestamp(DAY_ZERO + 3 * DAY)),
        group_by: Some(group_by.as_str().to_owned()),
        ..Default::default()
    }
    .validate()
    .unwrap();
    CostReport::new(
        group_by,
        fetch_report(conn, &params).expect("complete report from durable ledger"),
        params.range.start_secs,
        params.range.end_secs,
    )
}

fn serialized_reports(conn: &Connection) -> Vec<String> {
    GROUPINGS
        .iter()
        .map(|&group_by| serde_json::to_string(&report(conn, group_by)).unwrap())
        .collect()
}

fn entry(
    trace_id: &str,
    seconds: i64,
    provider: &str,
    model: &str,
    key_id: Option<&str>,
    cost: f64,
) -> ai_gateway::logger::LogEntry {
    sample_log_entry(
        trace_id,
        seconds,
        provider,
        model,
        cost,
        LogEntryOverrides {
            cache_read_tokens: Some(3),
            cache_creation_tokens: Some(4),
            reasoning_tokens: Some(5),
            input_tokens: Some(10),
            output_tokens: Some(20),
            virtual_key_id: key_id.map(str::to_owned),
            ..Default::default()
        },
    )
}

fn seed_mixed(logger: &RequestLogger) {
    let mut input_only = entry("unpriced-input", DAY_ZERO + 2 * DAY + 200, "alpha", "alpha-model", Some("key-a"), 0.0);
    input_only.input_tokens = Some(17);
    input_only.output_tokens = None;
    input_only.cache_read_tokens = None;
    input_only.cache_creation_tokens = None;
    input_only.reasoning_tokens = None;
    let mut output_only = input_only.clone();
    output_only.trace_id = "unpriced-output".to_owned();
    output_only.timestamp =
        chrono::DateTime::from_timestamp(DAY_ZERO + 7_200, 0).unwrap();
    output_only.provider = "beta".to_owned();
    output_only.model = "beta-model".to_owned();
    output_only.input_tokens = None;
    output_only.output_tokens = Some(19);
    let mut no_usage = input_only.clone();
    no_usage.trace_id = "zero-cost-without-usage".to_owned();
    no_usage.provider = "free".to_owned();
    no_usage.model = "unknown".to_owned();
    no_usage.input_tokens = Some(0);
    no_usage.output_tokens = Some(0);
    let mut priced_attempt = entry("priced-attempt", DAY_ZERO + DAY + 100, "attempt-only", "attempt-only", Some("attempt-only"), 99.0);
    priced_attempt.path.push_str("#attempt");
    let mut unpriced_attempt = priced_attempt.clone();
    unpriced_attempt.trace_id = "unpriced-attempt".to_owned();
    unpriced_attempt.cost = 0.0;
    for row in [
        entry("highest", DAY_ZERO, "zeta", "zeta-model", Some("key-z"), 0.004_321),
        entry("beta", DAY_ZERO + 3_600, "beta", "beta-model", Some("key-b"), 0.001_234),
        entry("alpha", DAY_ZERO + 2 * DAY + 100, "alpha", "alpha-model", Some("key-a"), 0.001_234),
        input_only,
        output_only,
        entry("anonymous-null", DAY_ZERO + 10_800, "gamma", "gamma-model", None, 0.000_111),
        entry("anonymous-empty", DAY_ZERO + 2 * DAY + 300, "delta", "delta-model", Some(""), 0.000_111),
        no_usage,
        priced_attempt,
        unpriced_attempt,
        entry("before-start", DAY_ZERO - 1, "outside", "outside", Some("outside"), 66.0),
        entry("at-exclusive-end", DAY_ZERO + 3 * DAY, "outside", "outside", Some("outside"), 66.0),
    ] {
        logger.log(row).unwrap();
    }
    logger.flush();
}

fn create_legacy_database(path: &str) {
    let conn = Connection::open(path).unwrap();
    conn.execute_batch(LEGACY_REQUESTS_DDL).unwrap();
    for (id, seconds, cost) in [
        (7, DAY_ZERO, 0.001_234),
        (19, DAY_ZERO + 2 * DAY + 10, 0.004_321),
        (42, DAY_ZERO + 2 * DAY + 20, 0.0),
    ] {
        conn.execute(
            "INSERT INTO requests VALUES (?1, ?2, ?3, 'POST', '/v1/chat/completions',
             'legacy-model', 'legacy-provider', 200, 10, ?4, ?5, ?6, 'requested', 'responded')",
            rusqlite::params![id, format!("legacy-{id}"), seconds, cost, format!("request-{id}"), format!("response-{id}")],
        )
        .unwrap();
    }
}

fn legacy_row_snapshot(conn: &Connection) -> Vec<Vec<rusqlite::types::Value>> {
    let mut stmt = conn
        .prepare(&format!("SELECT {LEGACY_COLUMNS} FROM requests ORDER BY id"))
        .unwrap();
    stmt.query_map([], |row| {
        (0..14).map(|column| row.get(column)).collect()
    })
    .unwrap()
    .collect::<Result<Vec<_>, _>>()
    .unwrap()
}

fn schema_snapshot(conn: &Connection) -> Vec<(String, String, Option<String>)> {
    let mut stmt = conn
        .prepare("SELECT type, name, sql FROM sqlite_master ORDER BY type, name")
        .unwrap();
    stmt.query_map([], |row| Ok((row.get(0)?, row.get(1)?, row.get(2)?)))
        .unwrap()
        .collect::<Result<Vec<_>, _>>()
        .unwrap()
}

#[test]
fn closed_range_reports_survive_logger_restart_byte_for_byte() {
    let config = isolated_config().logging;
    let logger = RequestLogger::new(config.clone()).unwrap();
    seed_mixed(&logger);
    assert_eq!(logger.database_path(), config.database_path);
    let before = {
        let conn = read_connection(logger.database_path());
        assert_eq!(report(&conn, GroupBy::Day).totals.cost_micro_usd, 7_011);
        serialized_reports(&conn)
    };
    drop(logger);

    let reopened = RequestLogger::new(config).unwrap();
    reopened.flush();
    let conn = read_connection(reopened.database_path());
    assert_eq!(serialized_reports(&conn), before, "Req 8.1: full rows and totals survive a fresh logger");
}

#[test]
fn additive_index_migration_is_idempotent_and_preserves_legacy_rows() {
    let config = isolated_config().logging;
    create_legacy_database(&config.database_path);
    let (before_reports, before_rows) = {
        let conn = read_connection(&config.database_path);
        assert_eq!(schema_snapshot(&conn).len(), 1, "pre-index database");
        (serialized_reports(&conn), legacy_row_snapshot(&conn))
    };
    assert_eq!(before_rows.len(), 3);
    let mut first_schema = None;
    for _ in 0..2 {
        let logger = RequestLogger::new(config.clone()).unwrap();
        logger.flush();
        {
            let conn = read_connection(logger.database_path());
            assert_eq!(serialized_reports(&conn), before_reports);
            assert_eq!(legacy_row_snapshot(&conn), before_rows, "original row ids, bodies, costs and metadata are unchanged");
            let schema = schema_snapshot(&conn);
            for (index, dimension) in [
                ("idx_requests_timestamp_provider", "provider"),
                ("idx_requests_timestamp_model", "model"),
            ] {
                let mut stmt = conn.prepare(&format!("PRAGMA index_info({index})")).unwrap();
                let columns = stmt
                    .query_map([], |row| row.get::<_, String>(2))
                    .unwrap()
                    .collect::<Result<Vec<_>, _>>()
                    .unwrap();
                assert_eq!(columns, vec!["timestamp", dimension]);
            }
            let backfilled: i64 = conn.query_row(
                "SELECT COUNT(*) FROM requests WHERE input_tokens IS NOT NULL
                 OR output_tokens IS NOT NULL OR key_id IS NOT NULL
                 OR cache_read_tokens IS NOT NULL OR cache_creation_tokens IS NOT NULL
                 OR reasoning_tokens IS NOT NULL",
                [],
                |row| row.get(0),
            ).unwrap();
            assert_eq!(backfilled, 0, "migration does not backfill ledger usage or attribution");
            if let Some(expected) = &first_schema {
                assert_eq!(&schema, expected, "second open leaves schema and indexes unchanged");
            } else {
                first_schema = Some(schema);
            }
        }
        drop(logger);
    }
}

#[test]
fn unmigrated_legacy_reports_zero_tokens_and_unattributed_usage() {
    let config = isolated_config().logging;
    create_legacy_database(&config.database_path);
    let conn = read_connection(&config.database_path);
    let before_schema = schema_snapshot(&conn);
    let before_rows = legacy_row_snapshot(&conn);
    for group_by in GROUPINGS {
        let result = report(&conn, group_by);
        assert_eq!(result.totals.requests, 3);
        assert_eq!(result.totals.cost_micro_usd, 5_555);
        for row in result.rows.iter().chain(std::iter::once(&result.totals)) {
            assert_eq!(token_counts(row), [0; 5]);
            assert_eq!(row.cost_unpriced_requests, 0);
        }
        if group_by == GroupBy::VirtualKey {
            assert_eq!(result.rows.len(), 1);
            assert_eq!(result.rows[0].dimension, "unattributed");
            assert_eq!(result.rows[0].requests, 3);
        }
    }
    assert_eq!(schema_snapshot(&conn), before_schema, "reports do not migrate or create aggregate tables");
    assert_eq!(legacy_row_snapshot(&conn), before_rows, "read-only reports leave historical rows intact");
}

fn token_counts(row: &ReportRow) -> [i64; 5] {
    [row.input_tokens, row.output_tokens, row.cache_read_tokens, row.cache_creation_tokens, row.reasoning_tokens]
}

#[test]
fn repeated_mixed_reports_are_byte_identical_for_every_grouping() {
    let logger = RequestLogger::new(isolated_config().logging).unwrap();
    seed_mixed(&logger);
    let conn = read_connection(logger.database_path());
    let before_schema = schema_snapshot(&conn);
    let first = serialized_reports(&conn);
    let second = serialized_reports(&conn);
    assert_eq!(first, second, "Req 6.1: full reports, including ties, tokens, bounds and totals");
    assert_eq!(schema_snapshot(&conn), before_schema);
}

fn query(group_by: &str, suffix: &str) -> String {
    format!(
        "/admin/cost-reports?start={}&end={}&group_by={group_by}{suffix}",
        timestamp(DAY_ZERO), timestamp(DAY_ZERO + 3 * DAY),
    )
}

async fn response(app: axum::Router, uri: &str) -> (StatusCode, String, Vec<u8>) {
    let response = app
        .oneshot(Request::get(uri).body(Body::empty()).unwrap())
        .await
        .unwrap();
    let status = response.status();
    let content_type = response.headers()[axum::http::header::CONTENT_TYPE]
        .to_str().unwrap().to_owned();
    let bytes = axum::body::to_bytes(response.into_body(), 1024 * 1024)
        .await.unwrap().to_vec();
    (status, content_type, bytes)
}

async fn json_report(app: axum::Router, uri: &str) -> Value {
    let (status, content_type, bytes) = response(app, uri).await;
    assert_eq!(status, StatusCode::OK, "{}", String::from_utf8_lossy(&bytes));
    assert!(content_type.starts_with("application/json"));
    serde_json::from_slice(&bytes).unwrap()
}

#[tokio::test]
async fn mixed_activity_http_report_counts_unpriced_usage_and_zero_fills_attempt_only_day() {
    let server = GatewayServer::new(isolated_config(), None).await.unwrap();
    seed_mixed(&server.state.logger);
    let result = json_report(server.build_router(), &query("day", "")).await;
    let rows = result["rows"].as_array().unwrap();
    assert_eq!(rows.len(), 3);
    assert_eq!(rows.iter().map(|row| row["dimension"].as_str().unwrap()).collect::<Vec<_>>(),
        vec!["2026-01-01", "2026-01-02", "2026-01-03"]);
    for (index, row) in rows.iter().enumerate() {
        assert_eq!(row["window_start"], DAY_ZERO + index as i64 * DAY);
        assert_eq!(row["window_end"], DAY_ZERO + (index as i64 + 1) * DAY);
    }
    for field in ["requests", "input_tokens", "output_tokens", "cache_read_tokens", "cache_creation_tokens", "reasoning_tokens", "cost_usd", "cost_unpriced_requests"] {
        assert_eq!(rows[1][field].as_f64().unwrap(), 0.0, "empty day's {field}");
    }
    assert_eq!(rows[0]["requests"], 4);
    assert_eq!(rows[2]["requests"], 4);
    assert_eq!(rows[0]["cost_usd"], json!(0.005_666));
    assert_eq!(rows[2]["cost_usd"], json!(0.001_345));
    assert_eq!(rows[0]["cost_unpriced_requests"], 1);
    assert_eq!(rows[2]["cost_unpriced_requests"], 1);
    let totals = &result["totals"];
    for (field, expected) in [
        ("requests", 8), ("input_tokens", 67), ("output_tokens", 119),
        ("cache_read_tokens", 15), ("cache_creation_tokens", 20),
        ("reasoning_tokens", 25), ("cost_unpriced_requests", 2),
    ] {
        assert_eq!(totals[field], expected, "{field}: {result}");
    }
    assert_eq!(totals["cost_usd"], json!(0.007_011));
    assert_eq!(totals["window_start"], DAY_ZERO);
    assert_eq!(totals["window_end"], DAY_ZERO + 3 * DAY);
}

#[tokio::test]
async fn mixed_dimension_http_reports_sort_real_ties_and_reconcile_key_attribution() {
    let server = GatewayServer::new(isolated_config(), None).await.unwrap();
    seed_mixed(&server.state.logger);
    let app = server.build_router();
    let day = json_report(app.clone(), &query("day", "")).await;
    for (group_by, dimensions, costs) in [
        ("provider", vec!["zeta", "alpha", "beta", "delta", "gamma", "free"], vec![0.004_321, 0.001_234, 0.001_234, 0.000_111, 0.000_111, 0.0]),
        ("model", vec!["zeta-model", "alpha-model", "beta-model", "delta-model", "gamma-model", "unknown"], vec![0.004_321, 0.001_234, 0.001_234, 0.000_111, 0.000_111, 0.0]),
        ("virtual_key", vec!["key-z", "key-a", "key-b", "unattributed"], vec![0.004_321, 0.001_234, 0.001_234, 0.000_222]),
    ] {
        let result = json_report(app.clone(), &query(group_by, "")).await;
        let rows = result["rows"].as_array().unwrap();
        assert_eq!(rows.iter().map(|row| row["dimension"].as_str().unwrap()).collect::<Vec<_>>(), dimensions, "{group_by}");
        assert_eq!(rows.iter().map(|row| row["cost_usd"].as_f64().unwrap()).collect::<Vec<_>>(), costs, "{group_by}");
        for row in rows {
            assert_eq!(row["window_start"], DAY_ZERO);
            assert_eq!(row["window_end"], DAY_ZERO + 3 * DAY);
        }
        assert_eq!(result["totals"], day["totals"], "all aggregates agree at full micro-USD precision");
        if group_by == "virtual_key" {
            assert_eq!(rows[3]["requests"], 2, "NULL and empty keys share the unattributed bucket");
            assert_eq!(rows[1]["requests"], 4);
            let key_sum: f64 = rows.iter().map(|row| row["cost_usd"].as_f64().unwrap()).sum();
            let day_sum: f64 = day["rows"].as_array().unwrap().iter().map(|row| row["cost_usd"].as_f64().unwrap()).sum();
            assert_eq!((key_sum * 100.0).round(), (day_sum * 100.0).round(), "Req 2.3: serialized USD reconciles at two decimal places");
        }
    }
}

#[tokio::test]
async fn provider_and_existing_key_http_filters_are_conjunctive() {
    let server = GatewayServer::new(isolated_config(), None).await.unwrap();
    seed_mixed(&server.state.logger);
    let app = server.build_router();
    for (suffix, requests, unpriced) in [
        ("&provider=alpha", 2, 1),
        ("&key_id=key-a", 4, 2),
        ("&provider=alpha&key_id=key-a", 2, 1),
        ("&provider=beta&key_id=key-a", 1, 1),
    ] {
        let result = json_report(app.clone(), &query("day", suffix)).await;
        assert_eq!(result["totals"]["requests"], requests, "{suffix}: {result}");
        assert_eq!(result["totals"]["cost_unpriced_requests"], unpriced, "{suffix}");
        let expected_cost = if suffix.contains("provider=beta") { 0.0 } else { 0.001_234 };
        assert_eq!(result["totals"]["cost_usd"], json!(expected_cost));
        assert_eq!(result["rows"][1]["requests"], 0);
    }
    let disjoint = json_report(app, &query("virtual_key", "&provider=alpha&key_id=key-b")).await;
    assert_eq!(disjoint["rows"], json!([]), "each filter matches separately, but their intersection is empty");
    assert_eq!(disjoint["totals"]["requests"], 0);
    assert_eq!(disjoint["totals"]["cost_usd"], json!(0.0));
}

#[tokio::test]
async fn http_range_accepts_exactly_366_days_and_rejects_367_days() {
    let server = GatewayServer::new(isolated_config(), None).await.unwrap();
    seed_mixed(&server.state.logger);
    let app = server.build_router();
    let uri = |days| format!("/admin/cost-reports?start={}&end={}", timestamp(DAY_ZERO), timestamp(DAY_ZERO + days * DAY));
    let accepted = json_report(app.clone(), &uri(366)).await;
    assert_eq!(accepted["rows"].as_array().unwrap().len(), 366);
    assert_eq!(accepted["rows"][365]["window_end"], DAY_ZERO + 366 * DAY);
    assert_eq!(accepted["rows"][365]["requests"], 0);
    let (status, content_type, bytes) = response(app, &uri(367)).await;
    assert_eq!(status, StatusCode::BAD_REQUEST);
    assert!(content_type.starts_with("application/json"));
    let error: Value = serde_json::from_slice(&bytes).unwrap();
    assert_eq!(error["error"]["parameter"], "range");
    assert!(error["error"]["message"].as_str().unwrap().contains("366"));
}

#[tokio::test]
async fn empty_dimension_csv_is_header_only_and_explicit_json_matches_default() {
    let server = GatewayServer::new(isolated_config(), None).await.unwrap();
    seed_mixed(&server.state.logger);
    let app = server.build_router();
    for group_by in ["provider", "model", "virtual_key"] {
        let (status, content_type, bytes) = response(app.clone(), &query(group_by, "&provider=missing&format=csv")).await;
        assert_eq!(status, StatusCode::OK);
        assert!(content_type.starts_with("text/csv"));
        assert_eq!(String::from_utf8(bytes).unwrap(), CSV_HEADER, "Req 4.3: {group_by}");
    }
    let (default_status, _, default_body) = response(app.clone(), &query("virtual_key", "")).await;
    let (explicit_status, _, explicit_body) = response(app, &query("virtual_key", "&format=json")).await;
    assert_eq!(default_status, StatusCode::OK);
    assert_eq!(explicit_status, StatusCode::OK);
    assert_eq!(explicit_body, default_body, "Req 4.2: explicit JSON and default export match byte for byte");
}
