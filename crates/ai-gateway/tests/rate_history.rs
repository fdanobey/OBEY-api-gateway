//! Rate-history seeding endpoint: `GET /dashboard/rate-history` returns the
//! trailing-60s per-second request buckets from the logs database so the
//! dashboard request-rate chart is populated on load and WS reconnect.

mod common;

use ai_gateway::config::Config;
use ai_gateway::gateway::GatewayServer;
use ai_gateway::logger::LogEntry;
use axum::body::Body;
use axum::http::{Request, StatusCode};
use chrono::{DateTime, Utc};
use serde_json::Value;
use tower::ServiceExt;

fn rate_history_config() -> Config {
    let mut config: Config = serde_json::from_value(serde_json::json!({
        "server": { "host": "127.0.0.1", "port": 0 },
        "providers": [{ "name": "test-provider", "type": "openai", "base_url": "http://localhost:1234" }],
        "model_groups": [{ "name": "test-group", "models": [{ "provider": "test-provider", "model": "gpt-4", "priority": 1 }] }]
    }))
    .expect("test configuration should deserialize");
    common::isolate_databases(&mut config);
    config
}

fn log_entry(trace_id: &str, timestamp: DateTime<Utc>) -> LogEntry {
    LogEntry {
        trace_id: trace_id.to_owned(),
        timestamp,
        method: "POST".to_owned(),
        path: "/v1/chat/completions".to_owned(),
        model: "gpt-4".to_owned(),
        provider: "test-provider".to_owned(),
        status_code: 200,
        duration_ms: 0,
        cost: 0.0,
        request_body: None,
        response_body: None,
        requested_model: None,
        responded_model: None,
        compression: None,
        memories_injected: 0,
        memories_stored: 0,
        injection_tokens: 0,
        detected_project: None,
        cache_read_tokens: None,
        cache_creation_tokens: None,
        cache_savings_cents: None,
        prefix_hash: None,
        reasoning_tokens: None,
        reasoning_compat_actions: None,
        error_class: None,
        context_reduction: None,
        client_protocol: None,
        claude_session_id: None,
    }
}

async fn fetch_rate_history(server: &GatewayServer) -> Value {
    let response = server
        .build_router()
        .oneshot(
            Request::get("/dashboard/rate-history")
                .body(Body::empty())
                .unwrap(),
        )
        .await
        .unwrap();
    assert_eq!(response.status(), StatusCode::OK);
    let body = axum::body::to_bytes(response.into_body(), 64 * 1024)
        .await
        .unwrap();
    serde_json::from_slice(&body).expect("rate history should be JSON")
}

#[tokio::test]
async fn rate_history_empty_when_no_logs() {
    let server = GatewayServer::new(rate_history_config(), None)
        .await
        .unwrap();

    let json = fetch_rate_history(&server).await;
    assert_eq!(json["window_seconds"], 60);
    assert_eq!(
        json["points"].as_array().map(Vec::len),
        Some(0),
        "empty logs database must yield empty points, got: {json}"
    );
}

/// Guard test (Task 10): the rate-history endpoint returns the exact shape
/// the dashboard seeding path (`seedRequestRateChart()` in
/// `dashboard/static/index.html`, verified statically: it fetches
/// `/rate-history` and zero-fills `points[].count` into epoch-aligned
/// per-second slots before converting to the rolling series on load and
/// WS reconnect) consumes — an object with `window_seconds: 60` and a
/// `points` array of sparse `{epoch_second, count}` buckets. Also carries
/// the bucketing assertions of the former
/// `rate_history_buckets_recent_logs_per_second`: one bucket per second
/// with traffic, and nothing outside the trailing-60s window.
#[tokio::test]
async fn rate_history_seeds_chart() {
    let server = GatewayServer::new(rate_history_config(), None)
        .await
        .unwrap();
    let now = Utc::now();
    let recent_second = now - chrono::Duration::seconds(5);
    // Two requests in one recent second, two in the current second, one
    // too old for the window.
    server
        .state
        .logger
        .log(log_entry("trace-a1", recent_second))
        .unwrap();
    server
        .state
        .logger
        .log(log_entry("trace-a2", recent_second))
        .unwrap();
    server
        .state
        .logger
        .log(log_entry("trace-seed-1", now))
        .unwrap();
    server
        .state
        .logger
        .log(log_entry("trace-seed-2", now))
        .unwrap();
    server
        .state
        .logger
        .log(log_entry("trace-old", now - chrono::Duration::seconds(300)))
        .unwrap();

    let json = fetch_rate_history(&server).await;
    // Shape contract for the dashboard JS seeding path.
    assert_eq!(json["window_seconds"], 60, "chart window is 60s");
    let points = json["points"]
        .as_array()
        .expect("points array for chart seeding");
    assert!(!points.is_empty(), "recent logs must produce seed points");
    for point in points {
        assert!(
            point["epoch_second"].is_i64(),
            "each point carries an integer epoch_second: {point}"
        );
        assert!(
            point["count"].is_u64(),
            "each point carries an unsigned count: {point}"
        );
    }
    assert!(
        points
            .iter()
            .any(|p| p["epoch_second"].as_i64() == Some(now.timestamp())
                && p["count"].as_u64() == Some(2)),
        "current-second bucket with both requests missing: {points:?}"
    );
    // Bucketing: one bucket per second with traffic, only inside the window.
    assert_eq!(
        points.len(),
        2,
        "one bucket per second with traffic: {points:?}"
    );
    let a = points
        .iter()
        .find(|p| {
            p["count"].as_u64() == Some(2)
                && p["epoch_second"].as_i64() == Some(recent_second.timestamp())
        })
        .unwrap_or_else(|| panic!("recent second with two requests missing: {points:?}"));
    assert_eq!(a["epoch_second"].as_i64(), Some(recent_second.timestamp()));
    assert!(
        points.iter().all(|p| p["epoch_second"].as_i64().unwrap()
            >= (now - chrono::Duration::seconds(60)).timestamp()),
        "points outside the 60s window: {points:?}"
    );
}

/// Degradation guard (finding D3): when the logs database query fails
/// (writer thread gone, so the count round-trip errors), the endpoint must
/// stay a 200 with empty points instead of a 5xx — logs are
/// observability-only and must never break the dashboard.
///
/// The writer thread cannot be stopped through the public logger API from
/// an integration test, so the degradation branch itself is unit-tested in
/// `dashboard::rate_history_degrades_to_empty_points_on_query_error`
/// (next-best per the finding). This test pins the happy-path contract the
/// branch shares: `window_seconds: 60` with a `points` array, using a
/// logger whose database path still works but holds no recent rows.
#[tokio::test]
async fn rate_history_degrades_to_empty_points_when_logger_fails() {
    let server = GatewayServer::new(rate_history_config(), None)
        .await
        .unwrap();

    let response = server
        .build_router()
        .oneshot(
            Request::get("/dashboard/rate-history")
                .body(Body::empty())
                .unwrap(),
        )
        .await
        .unwrap();
    assert_eq!(response.status(), StatusCode::OK);
    let body = axum::body::to_bytes(response.into_body(), 64 * 1024)
        .await
        .unwrap();
    let json: Value = serde_json::from_slice(&body).expect("degraded body is still JSON");
    assert_eq!(json["window_seconds"], 60);
    assert_eq!(
        json["points"].as_array().map(Vec::len),
        Some(0),
        "failed logger must yield empty points, got: {json}"
    );
}
