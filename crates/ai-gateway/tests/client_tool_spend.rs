//! Integration tests for coding-tool spend views (spec:
//! per-seat-coding-tool-spend-views, Task 6).
//!
//! End-to-end attribution through the chat and Anthropic front doors via
//! `tower::ServiceExt::oneshot()` (no port binding), the cost-report route
//! for the tool groupings (rows, filter, active_keys, CSV gating,
//! reconciliation against `group_by=day`), legacy pre-column degradation
//! at the query layer, and the dashboard log-filter round trip. Databases
//! are isolated per test via `isolate_databases`.

use ai_gateway::config::Config;
use ai_gateway::cost_reporting::{
    query::fetch_report, GroupBy, ReportFormat, ValidatedCostReportParams, ValidatedRange,
};
use ai_gateway::gateway::GatewayServer;
use ai_gateway::logger::LogEntry;
use axum::body::Body;
use axum::http::{Request, StatusCode};
use serde_json::json;
use tower::ServiceExt;
use wiremock::matchers::{method, path};
use wiremock::{Mock, MockServer, ResponseTemplate};

use crate::common::{
    create_ledger, isolate_databases, poll_logs, sample_log_entry, total_cost_micro,
    LogEntryOverrides, DAY_ZERO, LEGACY_REQUESTS_DDL,
};

// ---------------------------------------------------------------------------
// Harness
// ---------------------------------------------------------------------------

const CSV_HEADER: &str = "dimension,requests,input_tokens,output_tokens,cache_read_tokens,cache_creation_tokens,reasoning_tokens,cost_usd,cost_unpriced_requests";
const CSV_TOOL_HEADER: &str = "dimension,requests,input_tokens,output_tokens,cache_read_tokens,cache_creation_tokens,reasoning_tokens,cost_usd,cost_unpriced_requests,active_keys";

fn tool_spend_config() -> Config {
    let mut config: Config = serde_json::from_value(json!({
        "server": { "host": "127.0.0.1", "port": 0 },
        "providers": [{ "name": "tool-provider", "type": "openai", "base_url": "http://localhost:1234" }],
        "model_groups": [{ "name": "tool-group", "models": [{ "provider": "tool-provider", "model": "gpt-4", "priority": 1 }] }]
    }))
    .expect("test configuration should deserialize");
    isolate_databases(&mut config);
    // Fixed historical seed dates: a future retention scheduler must never
    // be able to delete them mid-test.
    config.logging.retention_days = 0;
    config
}

/// Seed one attributed row through the real logger write path.
fn seed_tool_row(
    logger: &ai_gateway::logger::RequestLogger,
    trace: &str,
    timestamp: i64,
    provider: &str,
    model: &str,
    cost: f64,
    key_id: Option<&str>,
    client_tool: Option<&str>,
) {
    let mut entry = sample_log_entry(
        trace,
        timestamp,
        provider,
        model,
        cost,
        LogEntryOverrides {
            virtual_key_id: key_id.map(str::to_string),
            input_tokens: Some(100),
            output_tokens: Some(50),
            ..Default::default()
        },
    );
    entry.client_tool = client_tool.map(str::to_string);
    logger.log(entry).expect("seed log entry");
    logger.flush();
}

/// Seed one day of tool-attributed traffic: 3 claude-code rows (keys k1,
/// k1, k2), 1 codex row (k1), 2 unattributed rows (k1, NULL key).
fn seed_attribution_traffic(server: &GatewayServer) {
    let cases: &[(i64, &str, &str, f64, Option<&str>, Option<&str>)] = &[
        (DAY_ZERO + 3_600, "openai", "gpt-4", 0.001, Some("k1"), Some("claude-code")),
        (DAY_ZERO + 7_200, "openai", "gpt-4", 0.002, Some("k1"), Some("claude-code")),
        (DAY_ZERO + 10_800, "anthropic", "claude-3", 0.004, Some("k2"), Some("claude-code")),
        (DAY_ZERO + 14_400, "openai", "gpt-4", 0.008, Some("k1"), Some("codex")),
        (DAY_ZERO + 18_000, "openai", "gpt-4", 0.016, Some("k1"), None),
        (DAY_ZERO + 21_600, "anthropic", "claude-3", 0.032, None, None),
    ];
    for (index, case) in cases.iter().enumerate() {
        let (timestamp, provider, model, cost, key_id, client_tool) = *case;
        seed_tool_row(
            &server.state.logger,
            &format!("seed-{index}"),
            timestamp,
            provider,
            model,
            cost,
            key_id,
            client_tool,
        );
    }
}

fn report_query(suffix: &str) -> String {
    format!(
        "/admin/cost-reports?start=2026-01-01T00:00:00Z&end=2026-01-02T00:00:00Z{suffix}"
    )
}

async fn get_response(app: axum::Router, uri: &str) -> (StatusCode, Vec<u8>) {
    let response = app
        .oneshot(Request::get(uri).body(Body::empty()).unwrap())
        .await
        .unwrap();
    let status = response.status();
    let bytes = axum::body::to_bytes(response.into_body(), 4 * 1024 * 1024)
        .await
        .unwrap();
    (status, bytes.to_vec())
}

async fn get_json(app: axum::Router, uri: &str) -> (StatusCode, serde_json::Value) {
    let (status, bytes) = get_response(app, uri).await;
    let json = serde_json::from_slice(&bytes).expect("JSON body");
    (status, json)
}

async fn get_text(app: axum::Router, uri: &str) -> (StatusCode, String) {
    let (status, bytes) = get_response(app, uri).await;
    let text = String::from_utf8(bytes).expect("UTF-8 body");
    (status, text)
}

// ---------------------------------------------------------------------------
// 1. End-to-end attribution through the front doors
// ---------------------------------------------------------------------------

fn completion_body() -> serde_json::Value {
    json!({
        "id": "chatcmpl-tool-spend",
        "object": "chat.completion",
        "created": 1_700_000_000_i64,
        "model": "gpt-4",
        "choices": [{
            "index": 0,
            "message": { "role": "assistant", "content": "mock answer" },
            "finish_reason": "stop"
        }],
        "usage": { "prompt_tokens": 4, "completion_tokens": 3, "total_tokens": 7 }
    })
}

async fn start_mock() -> MockServer {
    let mock = MockServer::start().await;
    Mock::given(method("POST"))
        .and(path("/v1/chat/completions"))
        .respond_with(ResponseTemplate::new(200).set_body_json(completion_body()))
        .mount(&mock)
        .await;
    mock
}

fn e2e_config(upstream_uri: &str) -> Config {
    let mut config: Config = serde_json::from_value(json!({
        "server": { "host": "127.0.0.1", "port": 0 },
        "providers": [{ "name": "tool-provider", "type": "openai", "base_url": upstream_uri }],
        "model_groups": [{ "name": "tool-group", "models": [{ "provider": "tool-provider", "model": "gpt-4", "priority": 1 }] }]
    }))
    .expect("test configuration should deserialize");
    isolate_databases(&mut config);
    config
}

fn chat_request(user_agent: Option<&str>, turn: &str) -> Request<Body> {
    let mut builder = Request::post("/v1/chat/completions").header("content-type", "application/json");
    if let Some(ua) = user_agent {
        builder = builder.header("user-agent", ua);
    }
    builder
        .body(Body::from(
            json!({
                "model": "tool-group",
                "messages": [{ "role": "user", "content": turn }]
            })
            .to_string(),
        ))
        .unwrap()
}

#[tokio::test]
async fn chat_request_with_claude_cli_user_agent_persists_claude_code() {
    let upstream = start_mock().await;
    let server = GatewayServer::new(e2e_config(&upstream.uri()), None)
        .await
        .unwrap();

    let response = server
        .build_router()
        .oneshot(chat_request(Some("claude-cli/2.0.1"), "hello"))
        .await
        .unwrap();
    assert_eq!(response.status(), StatusCode::OK);

    let rows = poll_logs(&server, |rows| {
        rows.iter()
            .any(|row| row.path == "/v1/chat/completions" && row.status_code == 200)
    })
    .await;
    let row = rows
        .iter()
        .find(|row| row.path == "/v1/chat/completions" && row.status_code == 200)
        .expect("chat outcome row must exist");
    assert_eq!(row.client_tool.as_deref(), Some("claude-code"));
}

#[tokio::test]
async fn chat_request_without_matching_user_agent_stays_unattributed() {
    let upstream = start_mock().await;
    let server = GatewayServer::new(e2e_config(&upstream.uri()), None)
        .await
        .unwrap();
    let app = server.build_router();

    // Distinct bodies so the exact cache cannot merge the two outcomes.
    let response = app
        .clone()
        .oneshot(chat_request(Some("my-cursor-tool/1.0"), "boundary guard"))
        .await
        .unwrap();
    assert_eq!(response.status(), StatusCode::OK);
    let response = app.oneshot(chat_request(None, "no user agent")).await.unwrap();
    assert_eq!(response.status(), StatusCode::OK);

    let rows = poll_logs(&server, |rows| {
        rows.iter()
            .filter(|row| row.path == "/v1/chat/completions" && row.status_code == 200)
            .count()
            == 2
    })
    .await;
    let outcomes: Vec<&LogEntry> = rows
        .iter()
        .filter(|row| row.path == "/v1/chat/completions" && row.status_code == 200)
        .collect();
    assert_eq!(outcomes.len(), 2, "rows: {rows:?}");
    for row in outcomes {
        assert_eq!(
            row.client_tool.as_deref(),
            Some("unattributed"),
            "boundary-guard and missing-UA rows persist unattributed"
        );
    }
}

#[tokio::test]
async fn anthropic_front_door_with_codex_user_agent_persists_codex() {
    let upstream = start_mock().await;
    let server = GatewayServer::new(e2e_config(&upstream.uri()), None)
        .await
        .unwrap();

    let request = Request::post("/v1/messages")
        .header("content-type", "application/json")
        .header("anthropic-version", "2023-06-01")
        .header("user-agent", "codex/0.42.0")
        .body(Body::from(
            json!({
                "model": "tool-group",
                "max_tokens": 128,
                "messages": [{ "role": "user", "content": "hello" }]
            })
            .to_string(),
        ))
        .unwrap();
    let response = server.build_router().oneshot(request).await.unwrap();
    assert_eq!(response.status(), StatusCode::OK);

    let rows = poll_logs(&server, |rows| {
        rows.iter()
            .any(|row| row.path == "/v1/messages" && row.status_code == 200)
    })
    .await;
    let row = rows
        .iter()
        .find(|row| row.path == "/v1/messages" && row.status_code == 200)
        .expect("anthropic outcome row must exist");
    assert_eq!(row.client_tool.as_deref(), Some("codex"));
    assert_eq!(row.client_protocol.as_deref(), Some("anthropic"));
}

// ---------------------------------------------------------------------------
// 2. Cost-report route: tool groupings, active_keys, reconciliation
// ---------------------------------------------------------------------------

#[tokio::test]
async fn tool_grouping_active_keys_agree_with_composite_pairs() {
    let server = GatewayServer::new(tool_spend_config(), None).await.unwrap();
    seed_attribution_traffic(&server);
    let app = server.build_router();

    let (status, tool) = get_json(app.clone(), &report_query("&group_by=client_tool")).await;
    assert_eq!(status, StatusCode::OK, "{tool}");
    let rows = tool["rows"].as_array().expect("rows array");

    let (status, pair) = get_json(app, &report_query("&group_by=client_tool_key")).await;
    assert_eq!(status, StatusCode::OK, "{pair}");
    let pair_rows = pair["rows"].as_array().expect("rows array");
    let pair_dims: Vec<&str> = pair_rows
        .iter()
        .map(|r| r["dimension"].as_str().unwrap())
        .collect();
    // active_keys agreement (Req 6.3): for attributed tools the number of
    // distinct composite pairs equals the client_tool row's active_keys
    // (every paired seat is real); the unattributed tool pairs with k1 and
    // with the unattributed seat, but only k1 counts as an active key.
    let claude_pairs: Vec<&str> = pair_dims
        .iter()
        .copied()
        .filter(|d| d.starts_with("claude-code:"))
        .collect();
    let codex_pairs: Vec<&str> = pair_dims
        .iter()
        .copied()
        .filter(|d| d.starts_with("codex:"))
        .collect();
    let claude_row = rows
        .iter()
        .find(|r| r["dimension"].as_str() == Some("claude-code"))
        .unwrap();
    let codex_row = rows
        .iter()
        .find(|r| r["dimension"].as_str() == Some("codex"))
        .unwrap();
    assert_eq!(
        claude_pairs.len() as i64,
        claude_row["active_keys"].as_i64().unwrap(),
        "{pair}"
    );
    assert_eq!(
        codex_pairs.len() as i64,
        codex_row["active_keys"].as_i64().unwrap(),
        "{pair}"
    );
    assert_eq!(claude_row["active_keys"], 2);
    assert_eq!(codex_row["active_keys"], 1);
    let unattributed_row = rows
        .iter()
        .find(|r| r["dimension"].as_str() == Some("unattributed"))
        .unwrap();
    assert_eq!(unattributed_row["active_keys"], 1, "k1 only: {tool}");
}

#[tokio::test]
async fn tool_groupings_report_rows_active_keys_and_reconciliation() {
    let server = GatewayServer::new(tool_spend_config(), None).await.unwrap();
    seed_attribution_traffic(&server);
    let app = server.build_router();

    let (status, tool) = get_json(app.clone(), &report_query("&group_by=client_tool")).await;
    assert_eq!(status, StatusCode::OK, "{tool}");
    let rows = tool["rows"].as_array().expect("rows array");
    assert_eq!(rows.len(), 3, "{tool}");
    // Cost desc, dimension asc ties.
    let dims: Vec<&str> = rows.iter().map(|r| r["dimension"].as_str().unwrap()).collect();
    assert_eq!(dims, vec!["unattributed", "codex", "claude-code"], "{tool}");
    assert_eq!(rows[0]["requests"], 2);
    assert_eq!(rows[0]["cost_usd"], json!(0.048));
    assert_eq!(rows[0]["active_keys"], 1, "k1 only: {tool}");
    assert_eq!(rows[1]["requests"], 1);
    assert_eq!(rows[1]["cost_usd"], json!(0.008));
    assert_eq!(rows[1]["active_keys"], 1, "k1 only: {tool}");
    assert_eq!(rows[2]["requests"], 3);
    assert_eq!(rows[2]["cost_usd"], json!(0.007));
    assert_eq!(rows[2]["active_keys"], 2, "k1 + k2: {tool}");

    let (status, pair) = get_json(app.clone(), &report_query("&group_by=client_tool_key")).await;
    assert_eq!(status, StatusCode::OK, "{pair}");
    let pair_rows = pair["rows"].as_array().expect("rows array");
    let pair_dims: Vec<&str> = pair_rows
        .iter()
        .map(|r| r["dimension"].as_str().unwrap())
        .collect();
    assert_eq!(
        pair_dims,
        vec![
            "unattributed:unattributed",
            "unattributed:k1",
            "codex:k1",
            "claude-code:k2",
            "claude-code:k1",
        ],
        "{pair}"
    );
    assert_eq!(pair["totals"]["requests"], 6, "{pair}");

    let (status, day) = get_json(app.clone(), &report_query("")).await;
    assert_eq!(status, StatusCode::OK, "{day}");
    assert_eq!(day["rows"].as_array().expect("rows array").len(), 1);
    assert_eq!(day["rows"][0]["requests"], 6);
    assert!(day["rows"][0].get("active_keys").is_none(), "{day}");

    let (status, key) = get_json(app.clone(), &report_query("&group_by=virtual_key")).await;
    assert_eq!(status, StatusCode::OK, "{key}");
    let key_rows = key["rows"].as_array().expect("rows array");
    assert_eq!(key_rows.len(), 3, "unattributed + k1 + k2: {key}");
    assert!(key_rows.iter().all(|row| row.get("active_keys").is_none()), "{key}");

    // Reconciliation: totals equal group_by=day across every grouping.
    for report in [&day, &key, &tool, &pair] {
        assert_eq!(report["totals"]["requests"], 6, "{report}");
        assert_eq!(report["totals"]["cost_usd"], json!(0.063), "{report}");
        assert_eq!(report["totals"]["input_tokens"], 600, "{report}");
        assert_eq!(report["totals"]["output_tokens"], 300, "{report}");
    }
}

#[tokio::test]
async fn client_tool_filter_narrows_report_to_matching_rows() {
    let server = GatewayServer::new(tool_spend_config(), None).await.unwrap();
    seed_attribution_traffic(&server);
    let app = server.build_router();

    for (filter, dimension, requests, cost_usd) in [
        ("claude-code", "claude-code", 3, json!(0.007)),
        ("codex", "codex", 1, json!(0.008)),
        ("unattributed", "unattributed", 2, json!(0.048)),
    ] {
        let (status, body) = get_json(
            app.clone(),
            &report_query(&format!("&group_by=client_tool&client_tool={filter}")),
        )
        .await;
        assert_eq!(status, StatusCode::OK, "{body}");
        let rows = body["rows"].as_array().expect("rows array");
        assert_eq!(rows.len(), 1, "filter {filter}: {body}");
        assert_eq!(rows[0]["dimension"], dimension, "filter {filter}: {body}");
        assert_eq!(rows[0]["requests"], requests, "filter {filter}: {body}");
        assert_eq!(body["totals"]["cost_usd"], cost_usd, "filter {filter}: {body}");
    }

    // A canonical token with no traffic: empty report, zero totals.
    let (status, body) = get_json(
        app,
        &report_query("&group_by=client_tool&client_tool=qwen-code"),
    )
    .await;
    assert_eq!(status, StatusCode::OK, "{body}");
    assert_eq!(body["rows"].as_array().expect("rows array").len(), 0, "{body}");
    assert_eq!(body["totals"]["requests"], 0, "{body}");
}

#[tokio::test]
async fn unknown_group_by_returns_400_listing_tool_values() {
    let server = GatewayServer::new(tool_spend_config(), None).await.unwrap();
    let app = server.build_router();

    let (status, body) = get_json(app, &report_query("&group_by=seat")).await;
    assert_eq!(status, StatusCode::BAD_REQUEST, "{body}");
    assert_eq!(body["error"]["parameter"], "group_by", "{body}");
    let message = body["error"]["message"].as_str().unwrap_or_default();
    assert!(message.contains("client_tool"), "{message}");
    assert!(message.contains("client_tool_key"), "{message}");
}

#[tokio::test]
async fn csv_tool_grouping_carries_active_keys_and_day_csv_is_unchanged() {
    let server = GatewayServer::new(tool_spend_config(), None).await.unwrap();
    seed_attribution_traffic(&server);
    let app = server.build_router();

    let (status, text) = get_text(
        app.clone(),
        &report_query("&group_by=client_tool&format=csv"),
    )
    .await;
    assert_eq!(status, StatusCode::OK);
    let mut lines = text.split("\r\n");
    let header = lines.next().unwrap();
    assert_eq!(header, CSV_TOOL_HEADER);
    let first_data_line = lines.next().unwrap();
    assert!(first_data_line.starts_with("unattributed,"), "{first_data_line}");
    assert_eq!(first_data_line.split(',').count(), 10);

    let (status, day_text) = get_text(app, &report_query("&format=csv")).await;
    assert_eq!(status, StatusCode::OK);
    let mut day_lines = day_text.split("\r\n");
    let day_header = day_lines.next().unwrap();
    assert_eq!(day_header, CSV_HEADER);
    let day_first = day_lines.next().unwrap();
    assert!(day_first.starts_with("2026-01-01,"), "{day_first}");
    assert_eq!(day_first.split(',').count(), 9);
}

// ---------------------------------------------------------------------------
// 3. Query-layer sum equality + legacy degradation
// ---------------------------------------------------------------------------

fn validated_params(group_by: GroupBy, client_tool: Option<&str>) -> ValidatedCostReportParams {
    ValidatedCostReportParams {
        range: ValidatedRange {
            start_secs: DAY_ZERO,
            end_secs: DAY_ZERO + 86_400,
        },
        group_by,
        provider: None,
        key_id: None,
        client_tool: client_tool.map(str::to_string),
        format: ReportFormat::Json,
    }
}

#[test]
fn sum_equality_across_groupings_at_integer_micro_precision() {
    let fixture = create_ledger();
    seed_tool_row(&fixture._logger, "q1", DAY_ZERO + 3_600, "openai", "gpt-4", 0.001, Some("k1"), Some("claude-code"));
    seed_tool_row(&fixture._logger, "q2", DAY_ZERO + 7_200, "openai", "gpt-4", 0.002, Some("k1"), Some("claude-code"));
    seed_tool_row(&fixture._logger, "q3", DAY_ZERO + 10_800, "anthropic", "claude-3", 0.004, Some("k2"), Some("claude-code"));
    seed_tool_row(&fixture._logger, "q4", DAY_ZERO + 14_400, "openai", "gpt-4", 0.008, Some("k1"), Some("codex"));
    seed_tool_row(&fixture._logger, "q5", DAY_ZERO + 18_000, "openai", "gpt-4", 0.016, Some("k1"), None);
    seed_tool_row(&fixture._logger, "q6", DAY_ZERO + 21_600, "anthropic", "claude-3", 0.032, None, None);
    let conn = rusqlite::Connection::open(&fixture.db_path).unwrap();

    let by_day = fetch_report(&conn, &validated_params(GroupBy::Day, None)).unwrap();
    let by_tool = fetch_report(&conn, &validated_params(GroupBy::ClientTool, None)).unwrap();
    let by_tool_key = fetch_report(&conn, &validated_params(GroupBy::ClientToolKey, None)).unwrap();
    let by_key = fetch_report(&conn, &validated_params(GroupBy::VirtualKey, None)).unwrap();

    let day_total = total_cost_micro(&by_day);
    assert_eq!(day_total, 63_000);
    for rows in [&by_tool, &by_tool_key, &by_key] {
        assert_eq!(total_cost_micro(rows), day_total);
        assert_eq!(rows.iter().map(|row| row.requests).sum::<i64>(), 6);
        assert_eq!(rows.iter().map(|row| row.input_tokens).sum::<i64>(), 600);
        assert_eq!(rows.iter().map(|row| row.output_tokens).sum::<i64>(), 300);
    }
}

#[test]
fn legacy_ledger_without_tool_column_degrades_to_unattributed() {
    let dir = tempfile::tempdir().expect("temp dir");
    let db_path = dir.path().join("logs.db").to_string_lossy().into_owned();
    {
        let conn = rusqlite::Connection::open(&db_path).unwrap();
        conn.execute_batch(&format!(
            "{LEGACY_REQUESTS_DDL}\nINSERT INTO requests (trace_id, timestamp, method, path, model, provider, status_code, duration_ms, cost)\nVALUES ('legacy-tool', 1767229200, 'POST', '/v1/chat/completions', 'gpt-4', 'openai', 200, 5, 0.001);",
        ))
        .unwrap();
    }

    let conn = rusqlite::Connection::open(&db_path).unwrap();
    let tool = fetch_report(&conn, &validated_params(GroupBy::ClientTool, None)).unwrap();
    assert_eq!(tool.len(), 1);
    assert_eq!(tool[0].dimension, "unattributed");
    assert_eq!(tool[0].requests, 1);
    assert_eq!(tool[0].cost_micro_usd, 1_000);
    assert_eq!(tool[0].active_keys, Some(0), "no key column: zero seats");

    let pair = fetch_report(&conn, &validated_params(GroupBy::ClientToolKey, None)).unwrap();
    assert_eq!(pair.len(), 1);
    assert_eq!(pair[0].dimension, "unattributed:unattributed");

    let specific =
        fetch_report(&conn, &validated_params(GroupBy::ClientTool, Some("claude-code"))).unwrap();
    assert!(specific.is_empty(), "no tool column: only unattributed can match");

    let unattributed =
        fetch_report(&conn, &validated_params(GroupBy::ClientTool, Some("unattributed"))).unwrap();
    assert_eq!(unattributed.len(), 1);
}

// ---------------------------------------------------------------------------
// 4. Dashboard log-filter round trip
// ---------------------------------------------------------------------------

#[tokio::test]
async fn dashboard_log_client_tool_filter_round_trip() {
    let server = GatewayServer::new(tool_spend_config(), None).await.unwrap();
    seed_attribution_traffic(&server);
    let app = server.build_router();

    let (status, body) = get_response(app.clone(), "/dashboard/logs").await;
    assert_eq!(status, StatusCode::OK);
    let all: Vec<LogEntry> = serde_json::from_slice(&body).expect("LogEntry JSON");
    assert_eq!(all.len(), 6);

    let (status, body) = get_response(app.clone(), "/dashboard/logs?client_tool=claude-code").await;
    assert_eq!(status, StatusCode::OK);
    let filtered: Vec<LogEntry> = serde_json::from_slice(&body).expect("LogEntry JSON");
    assert_eq!(filtered.len(), 3, "{filtered:?}");
    assert!(filtered
        .iter()
        .all(|row| row.client_tool.as_deref() == Some("claude-code")));

    let (status, body) = get_response(app.clone(), "/dashboard/logs?client_tool=unattributed").await;
    assert_eq!(status, StatusCode::OK);
    let unattributed: Vec<LogEntry> = serde_json::from_slice(&body).expect("LogEntry JSON");
    assert_eq!(unattributed.len(), 2, "{unattributed:?}");
    assert!(unattributed.iter().all(|row| row.client_tool.is_none()));

    let (status, body) = get_response(app, "/dashboard/logs?client_tool=codex").await;
    assert_eq!(status, StatusCode::OK);
    let codex: Vec<LogEntry> = serde_json::from_slice(&body).expect("LogEntry JSON");
    assert_eq!(codex.len(), 1, "{codex:?}");
    assert!(codex.iter().all(|row| row.client_tool.as_deref() == Some("codex")));
}
