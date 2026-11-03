//! Admin cost-report route integration tests (spec:
//! billing-grade-cost-reporting, Task 6).
//!
//! Exercises `GET /admin/cost-reports` and `GET /admin/cost-reports/reconcile`
//! through the full admin router via `tower::ServiceExt::oneshot()` — no live
//! server. Databases are isolated per test via `common::isolate_databases`
//! (budgets.db/keys.db are siblings of the redirected logs.db).
//!
//! Covers:
//! - 401 without credentials on both routes (existing admin auth layer)
//! - every 400 branch naming the invalid parameter (Req 1.6, 1.7, 4.4)
//! - JSON happy path with seeded ledger rows (Req 1.1)
//! - CSV happy path: text/csv, header first line, CRLF (Req 4.1)
//! - byte-identical totals on repeated identical query (Req 6.1)
//! - reconcile: 400 on bad params (Req 3.4); zero-activity zeros (Req 3.2)
//! - 500 on query failure: static envelope message, no internals leaked

mod common;

use ai_gateway::config::{AdminAuthConfig, Config};
use ai_gateway::gateway::GatewayServer;
use axum::body::Body;
use axum::http::{Request, StatusCode};
use tower::ServiceExt;

use common::{sample_log_entry, LogEntryOverrides, DAY_ZERO};

fn cost_reports_config() -> Config {
    let mut config: Config = serde_json::from_value(serde_json::json!({
        "server": { "host": "127.0.0.1", "port": 0 },
        "providers": [{ "name": "test-provider", "type": "openai", "base_url": "http://localhost:1234" }],
        "model_groups": [{ "name": "test-group", "models": [{ "provider": "test-provider", "model": "gpt-4", "priority": 1 }] }]
    }))
    .expect("test configuration should deserialize");
    common::isolate_databases(&mut config);
    // Retention off: fixtures seed fixed historical dates, so a future
    // retention scheduler must never be able to delete them.
    config.logging.retention_days = 0;
    config
}

/// Seed three days of traffic (day 2 empty) mirroring the Task 2 fixture:
/// day 1 openai/gpt-4 priced + unpriced, anthropic/claude-3 priced; day 3
/// one priced openai row plus an `#attempt` marker that must be excluded.
async fn seed_standard(server: &GatewayServer) {
    let logger = &server.state.logger;
    let billable = "/v1/chat/completions";
    let attempt = "/v1/chat/completions#attempt";
    let cases: &[(&str, i64, &str, &str, f64, &str)] = &[
        ("t1", DAY_ZERO + 3_600, "openai", "gpt-4", 0.001_234, billable),
        ("t2", DAY_ZERO + 7_200, "openai", "gpt-4", 0.0, billable),
        ("t3", DAY_ZERO + 10_800, "anthropic", "claude-3", 0.002_468, billable),
        ("t4", DAY_ZERO + 2 * 86_400 + 3_600, "openai", "gpt-4", 0.000_500, billable),
        ("t5", DAY_ZERO + 2 * 86_400 + 4_000, "openai", "gpt-4", 0.0, attempt),
    ];
    for (trace_id, timestamp, provider, model, cost, path) in cases {
        logger
            .log(sample_log_entry(
                trace_id,
                *timestamp,
                provider,
                model,
                *cost,
                LogEntryOverrides {
                    path: Some((*path).to_owned()),
                    ..Default::default()
                },
            ))
            .unwrap();
    }
    logger.flush();
}

fn query(suffix: &str) -> String {
    format!(
        "/admin/cost-reports?start=2026-01-01T00:00:00Z&end=2026-01-04T00:00:00Z{suffix}"
    )
}

async fn send(
    app: axum::Router,
    request: Request<Body>,
) -> (StatusCode, Option<String>, serde_json::Value) {
    let response = app.oneshot(request).await.unwrap();
    let status = response.status();
    let content_type = response
        .headers()
        .get(axum::http::header::CONTENT_TYPE)
        .and_then(|value| value.to_str().ok())
        .map(str::to_string);
    let bytes = axum::body::to_bytes(response.into_body(), 1024 * 1024)
        .await
        .unwrap();
    let json = if content_type.as_deref().is_some_and(|ct| ct.starts_with("text/csv")) {
        serde_json::Value::Null
    } else {
        serde_json::from_slice(&bytes).unwrap_or(serde_json::Value::Null)
    };
    (status, content_type, json)
}

// ---------------------------------------------------------------------------
// Auth: both routes behind the existing admin auth layer
// ---------------------------------------------------------------------------

#[tokio::test]
async fn cost_report_routes_require_credentials_when_auth_enabled() {
    let user_env = "COST_REPORTS_ADMIN_TEST_USER";
    let pass_env = "COST_REPORTS_ADMIN_TEST_PASS";
    // Unique env-var names per test binary: env is process-global, so
    // parallel tests never collide on these keys (webhooks_admin.rs /
    // integration.rs admin-auth precedent).
    std::env::set_var(user_env, "admin-user");
    std::env::set_var(pass_env, "admin-pass");

    let mut config = cost_reports_config();
    config.admin.auth = AdminAuthConfig {
        enabled: true,
        username_env: Some(user_env.to_string()),
        password_env: Some(pass_env.to_string()),
        allow_unauthenticated_non_loopback: false,
    };
    let server = GatewayServer::new(config, None).await.unwrap();
    let app = server.build_router();

    for uri in [
        query(""),
        "/admin/cost-reports/reconcile?start=2026-01-01T00:00:00Z&end=2026-01-02T00:00:00Z".to_string(),
    ] {
        let (status, _, _) = send(app.clone(), Request::get(uri).body(Body::empty()).unwrap()).await;
        assert_eq!(status, StatusCode::UNAUTHORIZED);
    }

    std::env::remove_var(user_env);
    std::env::remove_var(pass_env);
}

// ---------------------------------------------------------------------------
// 400 branches: each names the invalid parameter (Req 1.6, 1.7, 4.4)
// ---------------------------------------------------------------------------

#[tokio::test]
async fn cost_report_400s_name_the_offending_parameter() {
    let server = GatewayServer::new(cost_reports_config(), None).await.unwrap();
    let app = server.build_router();

    let cases: Vec<(&str, &str, &str)> = vec![
        // (path, query-suffix, expected parameter name)
        (
            "/admin/cost-reports?group_by=day",
            "",
            "start", // missing start + end
        ),
        (
            "/admin/cost-reports",
            "?start=not-a-timestamp&end=2026-01-02T00:00:00Z",
            "start",
        ),
        (
            "/admin/cost-reports",
            "?start=2026-01-01T00:00:00Z",
            "end", // missing end
        ),
        (
            "/admin/cost-reports",
            "?start=2026-01-01T00:00:00Z&end=also-not-a-timestamp",
            "end",
        ),
        (
            "/admin/cost-reports",
            "?start=2026-01-02T00:00:00Z&end=2026-01-01T00:00:00Z",
            "end", // end < start
        ),
        (
            "/admin/cost-reports",
            "?start=2026-01-01T00:00:00Z&end=2026-01-01T00:00:00Z",
            "end", // end == start
        ),
        (
            "/admin/cost-reports",
            "?start=2024-01-01T00:00:00Z&end=2026-01-01T00:00:00Z",
            "range", // > 366 days
        ),
        (
            "/admin/cost-reports",
            "?start=2026-01-01T00:00:00Z&end=2026-01-02T00:00:00Z&group_by=century",
            "group_by",
        ),
        (
            "/admin/cost-reports",
            "?start=2026-01-01T00:00:00Z&end=2026-01-02T00:00:00Z&format=xml",
            "format",
        ),
    ];

    for (path, query_suffix, expected_parameter) in cases {
        let uri = format!("{path}{query_suffix}");
        let (status, _, json) = send(app.clone(), Request::get(uri).body(Body::empty()).unwrap()).await;
        assert_eq!(
            status,
            StatusCode::BAD_REQUEST,
            "case {path}{query_suffix}: {json}"
        );
        assert_eq!(
            json["error"]["parameter"], expected_parameter,
            "case {path}{query_suffix} must name parameter {expected_parameter}: {json}"
        );
        let message = json["error"]["message"].as_str().unwrap_or_default();
        assert!(
            message.contains(expected_parameter)
                || (expected_parameter == "range" && message.contains("366")),
            "message must name the parameter or the limit: {message}"
        );
    }
}

// ---------------------------------------------------------------------------
// JSON happy path: seeded rows come back (Req 1.1, 1.2, 1.3, 1.8)
// ---------------------------------------------------------------------------

#[tokio::test]
async fn cost_report_json_returns_seeded_rows() {
    let server = GatewayServer::new(cost_reports_config(), None).await.unwrap();
    seed_standard(&server).await;
    let app = server.build_router();

    let (status, content_type, json) = send(
        app,
        Request::get(query("")).body(Body::empty()).unwrap(),
    )
    .await;
    assert_eq!(status, StatusCode::OK, "{json}");
    assert!(content_type.unwrap().starts_with("application/json"));

    let rows = json["rows"].as_array().expect("rows array");
    assert_eq!(
        rows.iter()
            .map(|row| row["dimension"].as_str().unwrap())
            .collect::<Vec<_>>(),
        vec!["2026-01-01", "2026-01-02", "2026-01-03"],
        "default group_by=day, calendar order, empty day zero-filled: {json}"
    );
    assert_eq!(rows[1]["requests"], 0, "empty day is zero-valued: {json}");
    assert_eq!(rows[0]["requests"], 3);
    // Unpriced row counted in requests but not cost (Req 1.8): day 1 has two
    // priced rows (0.001234 + 0.002468) and one cost=0 row. Compare via the
    // serialized f64 JSON number, not a float-literal sum (0.003702).
    assert_eq!(rows[0]["cost_usd"], serde_json::json!(0.003_702));
    assert_eq!(json["totals"]["requests"], 4, "#attempt row excluded: {json}");
    assert_eq!(json["totals"]["cost_usd"], serde_json::json!(0.004_202));
    // Report-row shape contract (Req 1.1).
    for key in [
        "dimension",
        "window_start",
        "window_end",
        "requests",
        "input_tokens",
        "output_tokens",
        "cache_read_tokens",
        "cache_creation_tokens",
        "reasoning_tokens",
        "cost_usd",
        "cost_unpriced_requests",
    ] {
        assert!(
            json["rows"][0].get(key).is_some(),
            "row missing field {key}: {json}"
        );
    }
}

// ---------------------------------------------------------------------------
// CSV happy path: text/csv, header first line, CRLF (Req 4.1, 4.3)
// ---------------------------------------------------------------------------

#[tokio::test]
async fn cost_report_csv_shape() {
    let server = GatewayServer::new(cost_reports_config(), None).await.unwrap();
    seed_standard(&server).await;
    let app = server.build_router();

    let request = Request::get(query("&format=csv")).body(Body::empty()).unwrap();
    let response = app.oneshot(request).await.unwrap();
    assert_eq!(response.status(), StatusCode::OK);
    let content_type = response
        .headers()
        .get(axum::http::header::CONTENT_TYPE)
        .and_then(|value| value.to_str().ok())
        .expect("content-type header");
    assert!(
        content_type.starts_with("text/csv"),
        "expected text/csv, got {content_type}"
    );
    let body = axum::body::to_bytes(response.into_body(), 1024 * 1024)
        .await
        .unwrap();
    let text = String::from_utf8(body.to_vec()).expect("UTF-8 CSV body");
    assert!(text.ends_with("\r\n"), "CRLF endings");
    let mut lines = text.split("\r\n");
    let header = lines.next().unwrap();
    assert_eq!(
        header,
        "dimension,requests,input_tokens,output_tokens,cache_read_tokens,cache_creation_tokens,reasoning_tokens,cost_usd,cost_unpriced_requests"
    );
    let first_data_line = lines.next().unwrap();
    assert!(first_data_line.starts_with("2026-01-01,"), "day row first");
    assert!(
        first_data_line.contains("0.003702"),
        "cost_usd rendered with exactly 6 decimals: {first_data_line}"
    );
}

// ---------------------------------------------------------------------------
// Byte-identical totals on repeated query (Req 6.1)
// ---------------------------------------------------------------------------

#[tokio::test]
async fn cost_report_repeated_query_is_byte_identical() {
    let server = GatewayServer::new(cost_reports_config(), None).await.unwrap();
    seed_standard(&server).await;
    let app = server.build_router();

    let first = app
        .clone()
        .oneshot(Request::get(query("")).body(Body::empty()).unwrap())
        .await
        .unwrap();
    let second = app
        .oneshot(Request::get(query("")).body(Body::empty()).unwrap())
        .await
        .unwrap();
    assert_eq!(first.status(), StatusCode::OK);
    assert_eq!(second.status(), StatusCode::OK);
    let first_body = axum::body::to_bytes(first.into_body(), 1024 * 1024)
        .await
        .unwrap();
    let second_body = axum::body::to_bytes(second.into_body(), 1024 * 1024)
        .await
        .unwrap();
    assert_eq!(
        first_body, second_body,
        "identical query must return byte-identical report (Req 6.1)"
    );
    let json: serde_json::Value = serde_json::from_slice(&first_body).unwrap();
    assert_eq!(json["totals"]["cost_usd"], serde_json::json!(0.004_202));
}

// ---------------------------------------------------------------------------
// Reconcile (Req 3)
// ---------------------------------------------------------------------------

#[tokio::test]
async fn reconcile_400_on_bad_params() {
    let server = GatewayServer::new(cost_reports_config(), None).await.unwrap();
    let app = server.build_router();

    let (status, _, json) = send(
        app.clone(),
        Request::get("/admin/cost-reports/reconcile?start=nope&end=2026-01-02T00:00:00Z")
            .body(Body::empty())
            .unwrap(),
    )
    .await;
    assert_eq!(status, StatusCode::BAD_REQUEST, "{json}");
    assert_eq!(json["error"]["parameter"], "start");

    let (status, _, json) = send(
        app,
        Request::get("/admin/cost-reports/reconcile")
            .body(Body::empty())
            .unwrap(),
    )
    .await;
    assert_eq!(status, StatusCode::BAD_REQUEST, "{json}");
    assert_eq!(json["error"]["parameter"], "start");
}

#[tokio::test]
async fn reconcile_zero_activity_returns_zeros() {
    let server = GatewayServer::new(cost_reports_config(), None).await.unwrap();
    let app = server.build_router();

    let (status, _, json) = send(
        app,
        Request::get("/admin/cost-reports/reconcile?start=2026-01-01T00:00:00Z&end=2026-01-02T00:00:00Z")
            .body(Body::empty())
            .unwrap(),
    )
    .await;
    assert_eq!(status, StatusCode::OK, "{json}");
    assert_eq!(json["ledger"]["requests"], 0);
    assert_eq!(json["ledger"]["cost_usd"], 0.0);
    assert_eq!(json["provider_spend"]["cost_usd"], 0.0);
    assert_eq!(json["key_usage"]["cost_usd"], 0.0);
    assert_eq!(json["unpriced_requests"], 0);
    assert_eq!(json["difference_usd"], 0.0);
}

// ---------------------------------------------------------------------------
// 500 branch: query failure returns a static envelope, no internals (OWASP
// A09 — rusqlite messages and DB paths must never reach the client)
// ---------------------------------------------------------------------------

#[tokio::test]
async fn cost_report_query_failure_returns_static_500_envelope() {
    // Start with valid isolated databases, then corrupt the logs database
    // file in place: startup must succeed (every store opens cleanly), and
    // the failure surfaces only when the read side opens the garbage file
    // at query time. SQLite's Windows VFS shares write access, so the
    // logger writer thread's open handle does not block the overwrite.
    let config = cost_reports_config();
    let corrupt_path = config.logging.database_path.clone();
    let server = GatewayServer::new(config, None).await.unwrap();
    std::fs::write(&corrupt_path, b"this is not a sqlite database at all").unwrap();
    let app = server.build_router();

    let (status, _, json) = send(
        app.clone(),
        Request::get(query("")).body(Body::empty()).unwrap(),
    )
    .await;
    assert_eq!(status, StatusCode::INTERNAL_SERVER_ERROR, "{json}");
    assert_eq!(json["error"]["type"], "cost_report_query_error");
    assert_eq!(json["error"]["message"], "cost report query failed");
    // No SQL text, no file-system paths, no rusqlite internals leaked.
    let body_text = json.to_string();
    assert!(
        !body_text.to_lowercase().contains("sqlite")
            && !body_text.to_lowercase().contains(".db")
            && !body_text.to_lowercase().contains("select"),
        "500 body must not leak query internals: {body_text}"
    );

    // Reconcile shares the same static 500 envelope via the helper, with
    // its own static message.
    let (status, _, json) = send(
        app,
        Request::get("/admin/cost-reports/reconcile?start=2026-01-01T00:00:00Z&end=2026-01-02T00:00:00Z")
            .body(Body::empty())
            .unwrap(),
    )
    .await;
    assert_eq!(status, StatusCode::INTERNAL_SERVER_ERROR, "{json}");
    assert_eq!(json["error"]["message"], "reconciliation query failed");
}
