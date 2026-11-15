//! Admin model-deprecation endpoints (model-deprecation-alerts spec,
//! Task 7): `GET /admin/model-deprecations` status rows and
//! `POST /admin/model-deprecations/scan` manual trigger semantics.
//!
//! In-process router/oneshot per test conventions — no live server.
//! Databases isolated per [`common::isolate_databases`].

use ai_gateway::config::*;
use ai_gateway::gateway::GatewayServer;
use ai_gateway::logger::LogEntry;
use axum::body::Body;
use axum::http::{Request, StatusCode};
use chrono::{Duration, Utc};
use serde_json::Value;
use tower::ServiceExt;
use wiremock::matchers::{method as wm_method, path as wm_path};
use wiremock::{Mock, MockServer, ResponseTemplate};

use crate::common;

// ---------------------------------------------------------------------------
// Scaffolding (webhooks_admin.rs pattern)
// ---------------------------------------------------------------------------

/// Retirement date `days` out from now — inside the default 30-day
/// warn window, computed at runtime so the assertions never go stale.
fn retirement_in(days: i64) -> String {
    (Utc::now() + Duration::days(days))
        .date_naive()
        .to_string()
}

fn deprecation_test_config() -> Config {
    let mut config = common::budget_test_config(None, None);
    config.model_groups[0].models[0].deprecation_date = Some(retirement_in(10));
    config.model_groups[0].models[0].replacement_model =
        Some("gpt-4.1-nano-successor".to_string());
    config
}

async fn build_app(mut config: Config) -> (axum::Router, GatewayServer) {
    common::isolate_databases(&mut config);
    let server = GatewayServer::new(config, None).await.unwrap();
    let router = server.build_router();
    (router, server)
}

async fn send(app: axum::Router, req: Request<Body>) -> (StatusCode, Value) {
    let resp = app.oneshot(req).await.unwrap();
    let status = resp.status();
    let bytes = axum::body::to_bytes(resp.into_body(), 1024 * 1024)
        .await
        .unwrap();
    let json = if bytes.is_empty() {
        Value::Null
    } else {
        serde_json::from_slice(&bytes).unwrap()
    };
    (status, json)
}

fn get_status() -> Request<Body> {
    Request::get("/admin/model-deprecations").body(Body::empty()).unwrap()
}

fn post_scan() -> Request<Body> {
    Request::post("/admin/model-deprecations/scan")
        .body(Body::empty())
        .unwrap()
}

// ---------------------------------------------------------------------------
// (a) GET status rows reflect an operator override (Reqs 2.1, 7.1)
// ---------------------------------------------------------------------------

#[tokio::test]
async fn get_status_rows_reflect_operator_override_with_model_group_refs() {
    let (app, server) = build_app(deprecation_test_config()).await;

    let (status, json) = send(app.clone(), get_status()).await;
    assert_eq!(status, StatusCode::OK);
    let rows = json.as_array().expect("bare top-level array of rows");
    assert_eq!(rows.len(), 1, "exactly the overridden model: {json}");
    let row = &rows[0];
    assert_eq!(row["provider"], "budget-provider");
    assert_eq!(row["model"], "gpt-4.1-nano");
    assert_eq!(row["retirement_date"], retirement_in(10));
    // Computed at handler time (>= test start) — 9 or 10 days remain.
    let days = row["days_remaining"].as_i64().expect("days_remaining");
    assert!((9..=10).contains(&days), "days_remaining: {days}");
    assert_eq!(row["replacement_model"], "gpt-4.1-nano-successor");
    assert_eq!(row["evidence_source"], "operator_override");
    assert_eq!(row["requests_7d"], 0, "no ledger rows yet: {row}");
    assert_eq!(row["runtime_gone"], false);
    assert_eq!(row["episode_open"], false);
    assert_eq!(
        row["model_groups"],
        serde_json::json!(["budget-group"]),
        "model_groups names the referencing configured group"
    );

    // Usage evidence: two ledger entries for the pair surface as
    // requests_7d + last_used_at. The logger writer commits
    // asynchronously — poll briefly instead of asserting immediately.
    let entry = |trace: &str| LogEntry {
        trace_id: trace.to_owned(),
        timestamp: Utc::now(),
        method: "POST".to_owned(),
        path: "/v1/chat/completions".to_owned(),
        model: "gpt-4.1-nano".to_owned(),
        provider: "budget-provider".to_owned(),
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
        input_tokens: None,
        output_tokens: None,
        virtual_key_id: None,
        prompt_id: None,
        prompt_version: None,
        prompt_label: None,
    };
    server.state.logger.log(entry("md-admin-1")).unwrap();
    server.state.logger.log(entry("md-admin-2")).unwrap();

    let mut saw_usage = false;
    for _ in 0..50 {
        let (_, json) = send(app.clone(), get_status()).await;
        let row = &json.as_array().expect("rows")[0];
        if row["requests_7d"] == 2 {
            assert!(
                row["last_used_at"].as_str().is_some_and(|v| !v.is_empty()),
                "last_used_at surfaced with usage: {row}"
            );
            saw_usage = true;
            break;
        }
        tokio::time::sleep(std::time::Duration::from_millis(100)).await;
    }
    assert!(saw_usage, "requests_7d never reached 2 (ledger evidence)");

    // The response carries no scan timestamp yet — the provider was
    // never scanned — and the UI renders "never scanned" from its
    // absence.
    assert!(row_absent_last_scanned_check(&send(app, get_status()).await.1));
}

fn row_absent_last_scanned_check(json: &Value) -> bool {
    json.as_array()
        .and_then(|rows| rows.first())
        .map(|row| row.get("last_scanned_at").is_none())
        .unwrap_or(false)
}

// ---------------------------------------------------------------------------
// (b) POST scan: started / already_running single-flight (Reqs 1.5, 7.2)
// ---------------------------------------------------------------------------

#[tokio::test]
async fn post_scan_returns_started_then_already_running_while_in_flight() {
    // Slow models listing: 1.2 s response delay keeps the first scan
    // in flight across the second trigger.
    let mock_server = MockServer::start().await;
    Mock::given(wm_method("GET"))
        .and(wm_path("/v1/models"))
        .respond_with(
            ResponseTemplate::new(200)
                .set_body_json(serde_json::json!({
                    "data": [{
                        "id": "scanned-deprecated-model",
                        "deprecation_date": retirement_in(20),
                        "successor_model": "scanned-successor"
                    }]
                }))
                .set_delay(std::time::Duration::from_millis(1200)),
        )
        .mount(&mock_server)
        .await;

    let mut config = deprecation_test_config();
    config.providers[0].base_url = Some(mock_server.uri());
    // The scanner's SSRF guard rejects loopback/http by default; the
    // test provider is exactly that (same relaxation the outbound
    // fetch tests apply for local mock upstreams).
    config.security.outbound_fetch.caller.allow_private_hosts = true;
    config.security.outbound_fetch.caller.allow_http = true;

    let (app, _server) = build_app(config).await;

    // First trigger: fire-and-forget — "started" returns while the
    // scan (1.2 s network delay) is still running. The admin gate is
    // process-wide in this test binary and shared with the parallel
    // model_deprecation_alerts tests, so the very first acquire can
    // legitimately race to `already_running`; the strict
    // started→already_running sequence is proven by the wiremock
    // delay below (second trigger while THIS test's scan is in
    // flight must answer already_running).
    let (status, json) = send(app.clone(), post_scan()).await;
    assert_eq!(status, StatusCode::OK);
    assert!(
        json["result"] == "started" || json["result"] == "already_running",
        "first trigger starts or joins a running scan: {json}"
    );

    // Second trigger while the first is in flight: no second scan.
    let (status, json) = send(app.clone(), post_scan()).await;
    assert_eq!(status, StatusCode::OK);
    assert_eq!(
        json["result"], "already_running",
        "in-flight scan is reported, never a double start"
    );

    // Completion: the scanned listing's deprecated model appears with
    // provider_metadata evidence and a per-provider last_scanned_at
    // staleness timestamp.
    let mut scanned_row = None;
    for _ in 0..100 {
        let (_, json) = send(app.clone(), get_status()).await;
        let rows = json.as_array().expect("rows");
        if let Some(row) = rows
            .iter()
            .find(|row| row["model"] == "scanned-deprecated-model")
        {
            scanned_row = Some(row.clone());
            break;
        }
        tokio::time::sleep(std::time::Duration::from_millis(100)).await;
    }
    let row = scanned_row.expect("scanned model row appears after the scan");
    assert_eq!(row["evidence_source"], "provider_metadata");
    assert_eq!(row["replacement_model"], "scanned-successor");
    assert!(
        row["last_scanned_at"].as_str().is_some_and(|v| !v.is_empty()),
        "flat per-row scan staleness timestamp: {row}"
    );

    // Gate released after completion: a fresh trigger starts again.
    let mut restarted = false;
    for _ in 0..50 {
        let (_, json) = send(app.clone(), post_scan()).await;
        if json["result"] == "started" {
            restarted = true;
            break;
        }
        tokio::time::sleep(std::time::Duration::from_millis(100)).await;
    }
    assert!(restarted, "scan gate must release after the scan completes");
}

// ---------------------------------------------------------------------------
// (c) Reload: removed provider disappears from status (Req 5.2)
// ---------------------------------------------------------------------------

#[tokio::test]
async fn reload_drops_removed_provider_from_status_rows() {
    let (app, server) = build_app(deprecation_test_config()).await;

    let (status, json) = send(app.clone(), get_status()).await;
    assert_eq!(status, StatusCode::OK);
    assert_eq!(
        json.as_array().expect("rows").len(),
        1,
        "override row present before reload"
    );

    // Reload a config whose provider (and its group entries) are gone.
    let mut reloaded = deprecation_test_config();
    reloaded.providers.clear();
    reloaded.model_groups.clear();
    common::isolate_databases(&mut reloaded);
    server.reload_config(reloaded).await.unwrap();

    let (status, json) = send(app, get_status()).await;
    assert_eq!(status, StatusCode::OK);
    assert_eq!(
        json.as_array().expect("rows").len(),
        0,
        "removed provider's deprecation entries drop after reload: {json}"
    );
}
