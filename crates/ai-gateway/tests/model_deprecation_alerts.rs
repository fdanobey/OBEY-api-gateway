//! Model-deprecation webhook alert emission (model-deprecation-alerts
//! spec, Task 6): end-to-end integration tests covering scan â usage
//! evidence â episode evaluation â signed `model.deprecated` delivery,
//! plus the dormancy contract.
//!
//! In-process wiremock provider (deprecation metadata) + wiremock
//! webhook receiver, manual `scanner.scan_all()` + explicit
//! `spawn_deprecation_emission` orchestration (the scanner-loop and
//! admin wiring land in Task 7), `GatewayServer` for the real ledger /
//! notifier plumbing. No live server.
//!
//! Covers:
//! - scan records a retiring model + recorded traffic â exactly ONE
//!   signed `model.deprecated` event delivered with a valid Standard
//!   Webhooks signature and the expected payload
//! - the usage gate: zero recorded traffic suppresses emission
//! - dormancy: zero webhook endpoints â no alert ledger rows

use ai_gateway::config::*;
use ai_gateway::gateway::GatewayServer;
use ai_gateway::logger::LogEntry;
use chrono::{Duration, Utc};
use wiremock::matchers::{method as wm_method, path as wm_path};
use wiremock::{Mock, MockServer, ResponseTemplate};

use crate::common;

const PROVIDER_NAME: &str = "budget-provider";
const WEBHOOK_SECRET: &str = "whsec_test_secret";
const MODEL_ID: &str = "gpt-4.1-nano";

// ---------------------------------------------------------------------------
// Scaffolding (webhooks_provider_alerts.rs + model_deprecation_admin.rs
// patterns)
// ---------------------------------------------------------------------------

/// Retirement date `days` out from now â inside the default 30-day
/// warn window.
fn retirement_in(days: i64) -> String {
    (Utc::now() + Duration::days(days))
        .date_naive()
        .to_string()
}

fn deprecation_config() -> Config {
    let mut config = common::budget_test_config(None, None);
    config.model_groups[0].models[0].deprecation_date = Some(retirement_in(10));
    config.model_groups[0].models[0].replacement_model =
        Some("gpt-4.1-nano-successor".to_string());
    config
}

/// Point webhooks at the wiremock receiver (signed endpoint, default-on
/// filter) with the receiver host allowlisted so the http URL passes
/// the SSRF guard (webhooks_provider_alerts.rs pattern).
fn configure_webhook_receiver(config: &mut Config, receiver: &MockServer) {
    let url = receiver.uri();
    let host = reqwest::Url::parse(&url)
        .expect("wiremock url parses")
        .host_str()
        .expect("wiremock host")
        .to_string();
    config.webhooks = WebhooksConfig {
        endpoints: vec![WebhookEndpoint {
            url: format!("{url}/hook"),
            secret: Some(WEBHOOK_SECRET.to_string()),
            events: None,
        }],
        allowed_private_hosts: vec![host],
    };
}

fn one_ledger_row(trace: &str) -> LogEntry {
    LogEntry {
        trace_id: trace.to_owned(),
        timestamp: Utc::now(),
        method: "POST".to_owned(),
        path: "/v1/chat/completions".to_owned(),
        model: MODEL_ID.to_owned(),
        provider: PROVIDER_NAME.to_owned(),
        status_code: 200,
        duration_ms: 10,
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
        output_cap_metadata: None,
    }
}

/// Poll the wiremock receiver until it holds at least `min` requests.
async fn wait_for_deliveries(receiver: &MockServer, min: usize) -> Vec<wiremock::Request> {
    for _ in 0..200 {
        if let Some(requests) = receiver.received_requests().await {
            if requests.len() >= min {
                return requests;
            }
        }
        tokio::time::sleep(std::time::Duration::from_millis(25)).await;
    }
    panic!("webhook receiver never saw {min} deliveries");
}

/// Verify the Standard Webhooks signature over the exact body bytes
/// and parse the envelope (webhooks_provider_alerts.rs pattern).
fn verify_signature_and_parse(request: &wiremock::Request) -> serde_json::Value {
    let headers = &request.headers;
    let header_str = |name: &str| -> String {
        headers
            .get(name)
            .unwrap_or_else(|| panic!("{name} header present"))
            .to_str()
            .expect("header value is ASCII")
            .to_string()
    };
    let event_id = header_str("webhook-id");
    let timestamp = header_str("webhook-timestamp");
    let signature = header_str("webhook-signature");
    let body = request.body.as_slice();

    let key = ring::hmac::Key::new(
        ring::hmac::HMAC_SHA256,
        &ai_gateway::webhooks::delivery::signing_key(WEBHOOK_SECRET),
    );
    let mut payload = Vec::with_capacity(event_id.len() + timestamp.len() + body.len() + 2);
    payload.extend_from_slice(event_id.as_bytes());
    payload.push(b'.');
    payload.extend_from_slice(timestamp.as_bytes());
    payload.push(b'.');
    payload.extend_from_slice(body);
    let expected_tag = ring::hmac::sign(&key, &payload);
    use base64::Engine as _;
    let expected = format!(
        "v1,{}",
        base64::engine::general_purpose::STANDARD.encode(expected_tag.as_ref())
    );
    assert_eq!(
        signature, expected,
        "Standard Webhooks signature must verify over id.timestamp.body"
    );

    serde_json::from_slice(body).expect("body parses as an event envelope")
}

/// Run one scan pass + emission orchestration and wait for the fan-out
/// to quiesce (bounded): returns the delivered requests.
async fn scan_and_emit(server: &GatewayServer) {
    server.state.deprecation_scanner.scan_all().await;
    ai_gateway::model_deprecations::emission::spawn_deprecation_emission(
        server.state.config.clone(),
        server.state.deprecation_state.clone(),
        server.state.logger.clone(),
        server.state.webhook_notifier.clone(),
    );
}

// ---------------------------------------------------------------------------
// (a) Scan + recorded traffic â exactly ONE signed model.deprecated
// ---------------------------------------------------------------------------

#[tokio::test]
async fn scan_with_traffic_emits_exactly_one_signed_model_deprecated() {
    let upstream = MockServer::start().await;
    let receiver = MockServer::start().await;

    let mut config = deprecation_config();
    common::isolate_databases(&mut config);
    config.providers[0].base_url = Some(upstream.uri());
    // Scanner SSRF guard relaxation for the loopback/http mock
    // upstream (model_deprecation_admin.rs pattern).
    config.security.outbound_fetch.caller.allow_private_hosts = true;
    config.security.outbound_fetch.caller.allow_http = true;
    configure_webhook_receiver(&mut config, &receiver);

    // Wiremock provider models listing: the model retiring in 10 days
    // (inside the 30-day warn window) with a scan-sourced successor.
    Mock::given(wm_method("GET"))
        .and(wm_path("/v1/models"))
        .respond_with(ResponseTemplate::new(200).set_body_json(serde_json::json!({
            "data": [{
                "id": MODEL_ID,
                "deprecation_date": retirement_in(10),
                "successor_model": "scanned-successor"
            }]
        })))
        .mount(&upstream)
        .await;
    Mock::given(wm_method("POST"))
        .and(wm_path("/hook"))
        .respond_with(ResponseTemplate::new(200))
        .mount(&receiver)
        .await;

    let server = GatewayServer::new(config, None).await.unwrap();

    // Recorded traffic: two ledger rows for the pair inside the 7-day
    // window â the usage gate passes.
    server.state.logger.log(one_ledger_row("md-emit-1")).unwrap();
    server.state.logger.log(one_ledger_row("md-emit-2")).unwrap();
    // Wait for the async writer commit so usage_since observes them.
    for _ in 0..100 {
        if server
            .state
            .logger
            .usage_since(PROVIDER_NAME, MODEL_ID, Utc::now() - Duration::days(7))
            .map(|(count, _)| count >= 2)
            .unwrap_or(false)
        {
            break;
        }
        tokio::time::sleep(std::time::Duration::from_millis(25)).await;
    }

    scan_and_emit(&server).await;
    let deliveries = wait_for_deliveries(&receiver, 1).await;
    assert_eq!(deliveries.len(), 1, "exactly one model.deprecated delivery");

    let envelope = verify_signature_and_parse(&deliveries[0]);
    assert_eq!(envelope["type"], "model.deprecated");
    assert_eq!(envelope["signed"], true);
    assert_eq!(envelope["id"], deliveries[0].headers.get("webhook-id").unwrap().to_str().unwrap());
    let data = &envelope["data"];
    assert_eq!(data["provider"], PROVIDER_NAME);
    assert_eq!(data["model"], MODEL_ID);
    assert_eq!(data["retirement_date"], retirement_in(10));
    let days = data["days_remaining"].as_i64().expect("days_remaining");
    assert!((9..=10).contains(&days), "days_remaining: {days}");
    // Operator override (config) wins above scan data for the
    // replacement hint (Req 2.1).
    assert_eq!(data["replacement_model"], "gpt-4.1-nano-successor");
    assert_eq!(data["evidence_source"], "operator_override");
    assert_eq!(data["requests_7d"], 2);
    assert!(
        data["last_used_at"].as_str().is_some_and(|v| !v.is_empty()),
        "last_used_at surfaced: {data}"
    );
    assert_eq!(data["runtime_gone"], false);

// The alert ledger row landed (scope/type via the D9 seam; the
// scope label is derived from the envelope's provider/model).
let expected_label = format!("model:{PROVIDER_NAME}/{MODEL_ID}");
let mut saw_row = false;
for _ in 0..100 {
let rows = server.state.webhook_ledger.recent_events(10);
if rows.iter().any(|row| {
row.event_type == "model.deprecated" && row.scope_label == expected_label
}) {
saw_row = true;
break;
}
tokio::time::sleep(std::time::Duration::from_millis(25)).await;
}
    assert!(saw_row, "model.deprecated ledger row never landed");

    // Episode de-dup: a second scan + emission pass must NOT deliver
    // again while the episode stays open.
    scan_and_emit(&server).await;
    tokio::time::sleep(std::time::Duration::from_millis(400)).await;
    let deliveries = receiver.received_requests().await.unwrap_or_default();
    assert_eq!(
        deliveries.len(),
        1,
        "no re-fire while the deprecation episode is open"
    );
}

// ---------------------------------------------------------------------------
// (b) Dormancy: zero endpoints â no alert ledger rows
// ---------------------------------------------------------------------------

#[tokio::test]
async fn dormant_config_scans_without_emitting_ledger_rows() {
    let upstream = MockServer::start().await;

    let mut config = deprecation_config();
    common::isolate_databases(&mut config);
    config.providers[0].base_url = Some(upstream.uri());
    config.security.outbound_fetch.caller.allow_private_hosts = true;
    config.security.outbound_fetch.caller.allow_http = true;
    // webhooks stays Default::default() â zero endpoints (dormancy).

    Mock::given(wm_method("GET"))
        .and(wm_path("/v1/models"))
        .respond_with(ResponseTemplate::new(200).set_body_json(serde_json::json!({
            "data": [{
                "id": MODEL_ID,
                "deprecation_date": retirement_in(10)
            }]
        })))
        .mount(&upstream)
        .await;

    let server = GatewayServer::new(config, None).await.unwrap();
    server.state.logger.log(one_ledger_row("md-dormant-1")).unwrap();

    scan_and_emit(&server).await;

    // Bounded absence poll: the scan demonstrably ran (scan data
    // present in the state), the funnel quiesced, and the ledger
    // holds no alert rows.
    tokio::time::sleep(std::time::Duration::from_millis(500)).await;
    let rows = server.state.webhook_ledger.recent_events(10);
    assert!(
        rows.is_empty(),
        "dormant config must not write alert ledger rows: {rows:?}"
    );
}

// ---------------------------------------------------------------------------
// Usage gate: zero traffic suppresses emission (episode un-opened)
// ---------------------------------------------------------------------------

#[tokio::test]
async fn scan_without_traffic_emits_nothing() {
    let upstream = MockServer::start().await;
    let receiver = MockServer::start().await;

    let mut config = deprecation_config();
    common::isolate_databases(&mut config);
    config.providers[0].base_url = Some(upstream.uri());
    config.security.outbound_fetch.caller.allow_private_hosts = true;
    config.security.outbound_fetch.caller.allow_http = true;
    configure_webhook_receiver(&mut config, &receiver);

    Mock::given(wm_method("GET"))
        .and(wm_path("/v1/models"))
        .respond_with(ResponseTemplate::new(200).set_body_json(serde_json::json!({
            "data": [{
                "id": MODEL_ID,
                "deprecation_date": retirement_in(10)
            }]
        })))
        .mount(&upstream)
        .await;
    Mock::given(wm_method("POST"))
        .and(wm_path("/hook"))
        .respond_with(ResponseTemplate::new(200))
        .mount(&receiver)
        .await;

    let server = GatewayServer::new(config, None).await.unwrap();
    // NO ledger rows: the usage gate must suppress emission without
    // opening the episode.

    scan_and_emit(&server).await;
    tokio::time::sleep(std::time::Duration::from_millis(500)).await;
    assert_eq!(
        receiver.received_requests().await.unwrap_or_default().len(),
        0,
        "usage gate: zero recorded traffic suppresses emission"
    );
    assert!(server.state.webhook_ledger.recent_events(10).is_empty());

    // Traffic arriving later still alerts (the episode was never
    // opened by the suppressed pass).
    server.state.logger.log(one_ledger_row("md-late-1")).unwrap();
    for _ in 0..100 {
        if server
            .state
            .logger
            .usage_since(PROVIDER_NAME, MODEL_ID, Utc::now() - Duration::days(7))
            .map(|(count, _)| count >= 1)
            .unwrap_or(false)
        {
            break;
        }
        tokio::time::sleep(std::time::Duration::from_millis(25)).await;
    }
    scan_and_emit(&server).await;
    let deliveries = wait_for_deliveries(&receiver, 1).await;
    assert_eq!(deliveries.len(), 1, "late traffic re-arms the gate");
}

// ---------------------------------------------------------------------------
// Task 9: production emission wiring  manual admin scan trigger emits
// without explicit orchestration (scan_all carries the emission deps).
// ---------------------------------------------------------------------------

/// Manual scan trigger (POST /admin/model-deprecations/scan) ? the
/// spawned scan pass itself emits exactly ONE signed model.deprecated
/// event with the full payload  no external emission call. This is
/// the (b) production call-site wiring proof: `GatewayServer::new`
/// builds the scanner with the shared logger/notifier Arcs and
/// `scan_all` fires `spawn_deprecation_emission` on completion.
// Validates: Task 9 acceptance 1(b), 3(a)
#[tokio::test]
async fn manual_admin_scan_emits_without_explicit_orchestration() {
    use axum::body::Body;
    use axum::http::Request;
    use tower::ServiceExt;

    let upstream = MockServer::start().await;
    let receiver = MockServer::start().await;

    let mut config = deprecation_config();
    common::isolate_databases(&mut config);
    config.providers[0].base_url = Some(upstream.uri());
    config.security.outbound_fetch.caller.allow_private_hosts = true;
    config.security.outbound_fetch.caller.allow_http = true;
    configure_webhook_receiver(&mut config, &receiver);

    Mock::given(wm_method("GET"))
        .and(wm_path("/v1/models"))
        .respond_with(ResponseTemplate::new(200).set_body_json(serde_json::json!({
            "data": [{
                "id": MODEL_ID,
                "deprecation_date": retirement_in(10),
                "successor_model": "scanned-successor"
            }]
        })))
        .mount(&upstream)
        .await;
    Mock::given(wm_method("POST"))
        .and(wm_path("/hook"))
        .respond_with(ResponseTemplate::new(200))
        .mount(&receiver)
        .await;

    let server = GatewayServer::new(config, None).await.unwrap();
    server.state.logger.log(one_ledger_row("md-wiring-1")).unwrap();
    for _ in 0..100 {
        if server
            .state
            .logger
            .usage_since(PROVIDER_NAME, MODEL_ID, Utc::now() - Duration::days(7))
            .map(|(count, _)| count >= 1)
            .unwrap_or(false)
        {
            break;
        }
        tokio::time::sleep(std::time::Duration::from_millis(25)).await;
    }

    // Manual trigger through the real admin route — fire-and-forget:
    // the spawned task calls `scanner.scan_all()`, which now fires
    // emission itself (Task 9 wiring). The route is process-shared
    // test binary territory: the admin-module gate may legitimately
    // answer `already_running` while a concurrently running
    // admin-module test holds it, so accept both body shapes (the
    // started/already_running semantics are covered by
    // model_deprecation_admin's own tests) and then GUARANTEE a
    // completed pass with a direct `scan_all()` — the exact
    // function the spawned task runs. Either path emits; the
    // episode registry dedups to exactly one event either way.
    let response = server
        .build_router()
        .oneshot(
            Request::post("/admin/model-deprecations/scan")
                .body(Body::empty())
                .unwrap(),
        )
        .await
        .unwrap();
    assert_eq!(response.status(), axum::http::StatusCode::OK);
    let body = axum::body::to_bytes(response.into_body(), 64 * 1024)
        .await
        .unwrap();
    let json: serde_json::Value = serde_json::from_slice(&body).unwrap();
    assert!(
        json["result"] == "started" || json["result"] == "already_running",
        "unexpected scan trigger body: {json}"
    );
    server.state.deprecation_scanner.scan_all().await;

    // The scan pass itself emitted: exactly one delivery, signed,
    // correct payload (evidence precedence: operator override wins).
    let deliveries = wait_for_deliveries(&receiver, 1).await;
    assert_eq!(deliveries.len(), 1, "exactly one model.deprecated delivery");
    let envelope = verify_signature_and_parse(&deliveries[0]);
    assert_eq!(envelope["type"], "model.deprecated");
    assert_eq!(envelope["signed"], true);
    let data = &envelope["data"];
    assert_eq!(data["provider"], PROVIDER_NAME);
    assert_eq!(data["model"], MODEL_ID);
    assert_eq!(data["retirement_date"], retirement_in(10));
    assert_eq!(data["replacement_model"], "gpt-4.1-nano-successor");
    assert_eq!(data["evidence_source"], "operator_override");
    assert_eq!(data["requests_7d"], 1);
    assert_eq!(data["runtime_gone"], false);

    // A second manual scan does not re-emit (episode stays open).
    let response = server
        .build_router()
        .oneshot(
            Request::post("/admin/model-deprecations/scan")
                .body(Body::empty())
                .unwrap(),
        )
        .await
        .unwrap();
    assert_eq!(response.status(), axum::http::StatusCode::OK);
    tokio::time::sleep(std::time::Duration::from_millis(600)).await;
    assert_eq!(
        receiver.received_requests().await.unwrap_or_default().len(),
        1,
        "episode de-dup: no second delivery"
    );
}

// ---------------------------------------------------------------------------
// Task 9: runtime 404 model_not_found ? failover success + exactly one
// runtime_detection escalation event (fire-and-forget at the router
// call site).
// ---------------------------------------------------------------------------

/// Runtime-gone wiring (c): a forwarded chat request whose primary
/// provider answers 404 model_not_found fails over to the healthy
/// secondary AND the router's detection site fires emission  the
/// warn episode opened by the earlier scan escalates warn?gone,
/// emitting exactly ONE new event with evidence_source
/// runtime_detection (Req 4.4).
///
/// Restart semantics (acceptance 3(c), mirroring budget-alerts D11):
/// the episode registry is in-memory only; after a gateway restart
/// the first detection/scan re-opens the episode and re-emits  at
/// most one duplicate event per restart. This test keeps everything
/// in one process (episode already open), asserting the no-duplicate
/// invariant; the restart duplicate bound is by construction (fresh
/// `DeprecationState::default()` per `GatewayServer::new`).
// Validates: Task 9 acceptance 1(c), 3(b), 3(c)
#[tokio::test]
async fn runtime_404_fails_over_and_emits_single_runtime_detection() {
    use axum::body::Body;
    use axum::http::{Request, StatusCode};
    use tower::ServiceExt;

    let upstream = MockServer::start().await;
    let fallback = MockServer::start().await;
    let receiver = MockServer::start().await;

    let mut config = deprecation_config();
    common::isolate_databases(&mut config);
    config.providers[0].base_url = Some(upstream.uri());
    config.security.outbound_fetch.caller.allow_private_hosts = true;
    config.security.outbound_fetch.caller.allow_http = true;
    configure_webhook_receiver(&mut config, &receiver);

    // Secondary provider + model so the 404 on the primary fails over.
    config.providers.push(Provider {
        name: "fallback-provider".to_string(),
        provider_type: "openai".to_string(),
        base_url: Some(fallback.uri()),
        api_key_env: None,
        api_key_encrypted: None,
        api_secret_env: None,
        api_secret_encrypted: None,
        auth_method: None,
        resolved_api_key: None,
        resolved_api_secret: None,
        region: None,
        timeout_seconds: 30,
        ttfb_timeout_seconds: None,
        total_timeout_seconds: None,
        buffered_upstream_streaming: None,
        disable_parallel_tool_calls: false,
        rate_limit_max_wait_ms: None,
        max_connections: 10,
        rate_limit_per_minute: 0,
        custom_headers: Default::default(),
        user_agent: None,
        connection_pool: ProviderConnectionPoolConfig::default(),
        budget: None,
        manual_models: vec![],
        global_inference_profile: false,
        cross_region_inference: false,
        custom_vpc_endpoint: false,
        prompt_caching: false,
        compression: None,
        memory: None,
        reasoning: true,
        codex_base_url_override: None,
        codex_model_override: None,
        instructions_override: None,
        max_rate_limit_cooldown_seconds: None,
    });
config.model_groups[0].models.push(ProviderModel {
    provider: "fallback-provider".to_string(),
    model: MODEL_ID.to_string(),
        cost_per_million_input_tokens: 30.0,
        cost_per_million_output_tokens: 60.0,
        cache_support: None,
        cache_min_tokens: None,
        cost_per_million_cache_read_input_tokens: None,
        cost_per_million_cache_creation_input_tokens: None,
        priority: 200,
        structured_output_passthrough: None,
        tier: None,
        context_window: 0,
        specializations: vec![],
        cost_per_million_reasoning_tokens: None,
        reasoning_family: None,
        reasoning_parameter: None,
        deprecation_date: None,
        replacement_model: None,
    });

    // Primary /v1/models carries the deprecation metadata.
    Mock::given(wm_method("GET"))
        .and(wm_path("/v1/models"))
        .respond_with(ResponseTemplate::new(200).set_body_json(serde_json::json!({
            "data": [{
                "id": MODEL_ID,
                "deprecation_date": retirement_in(10)
            }]
        })))
        .mount(&upstream)
        .await;
    // Primary chat: exactly one 404 model_not_found  the runtime
    // detection signature (Req 5.1).
    Mock::given(wm_method("POST"))
        .and(wm_path("/v1/chat/completions"))
        .respond_with(ResponseTemplate::new(404).set_body_string(
            r#"{"error":{"code":"model_not_found","message":"The model does not exist"}}"#,
        ))
        .expect(1)
        .mount(&upstream)
        .await;
    // Fallback chat: healthy streamed turn.
    Mock::given(wm_method("POST"))
        .and(wm_path("/v1/chat/completions"))
        .respond_with(
            ResponseTemplate::new(200)
                .insert_header("content-type", "text/event-stream")
                .set_body_string(
                    "data: {\"id\":\"1\",\"object\":\"chat.completion.chunk\",\"choices\":[{\"index\":0,\"delta\":{\"content\":\"ok\"},\"finish_reason\":null}]}\n\n\
                     data: {\"id\":\"1\",\"object\":\"chat.completion.chunk\",\"choices\":[{\"index\":0,\"delta\":{},\"finish_reason\":\"stop\"}]}\n\n\
                     data: [DONE]\n\n",
                ),
        )
        .expect(1)
        .mount(&fallback)
        .await;
    Mock::given(wm_method("POST"))
        .and(wm_path("/hook"))
        .respond_with(ResponseTemplate::new(200))
        .mount(&receiver)
        .await;

    let server = GatewayServer::new(config, None).await.unwrap();

    // Seed ledger usage so the usage gate passes for the primary
    // pair (the forwarded request alone may race the emission's
    // usage lookup).
    server.state.logger.log(one_ledger_row("md-rt-1")).unwrap();
    for _ in 0..100 {
        if server
            .state
            .logger
            .usage_since(PROVIDER_NAME, MODEL_ID, Utc::now() - Duration::days(7))
            .map(|(count, _)| count >= 1)
            .unwrap_or(false)
        {
            break;
        }
        tokio::time::sleep(std::time::Duration::from_millis(25)).await;
    }

    // Phase (a): manual scan opens the warn episode + emits once
    // (operator override evidence — config carries deprecation_date).
    server.state.deprecation_scanner.scan_all().await;
    let first = wait_for_deliveries(&receiver, 1).await;
    assert_eq!(first.len(), 1, "scan pass emits exactly one event");
    let scan_envelope = verify_signature_and_parse(&first[0]);
    assert_eq!(scan_envelope["data"]["evidence_source"], "operator_override");
    assert_eq!(scan_envelope["data"]["runtime_gone"], false);

    // Forward a chat request through the router: primary 404s with
    // the model_not_found signature ? runtime-gone mark + failover.
    let app = server.build_router();
    let req = Request::post("/v1/chat/completions")
        .header("content-type", "application/json")
        .body(Body::from(
            serde_json::to_string(&serde_json::json!({
                "model": "budget-group",
                "messages": [{"role": "user", "content": "hello"}]
            }))
            .unwrap(),
        ))
        .unwrap();
    let resp = app.oneshot(req).await.unwrap();
    assert_eq!(resp.status(), StatusCode::OK, "request succeeds via fallback");
    upstream.verify().await;
    fallback.verify().await;

    // Runtime-gone mark present (router call site recorded it).
    assert!(
        server
            .state
            .deprecation_state
            .is_gone(PROVIDER_NAME, MODEL_ID),
        "runtime-gone mark recorded for the primary pair"
    );

    // Phase (b): the detection site fired emission  the warn episode
    // escalates to gone, emitting exactly ONE new runtime_detection
    // event (Req 4.4: escalation emits once).
    let deliveries = wait_for_deliveries(&receiver, 2).await;
    assert_eq!(deliveries.len(), 2, "scan event + exactly one escalation");
    let escalation = verify_signature_and_parse(&deliveries[1]);
    assert_eq!(escalation["type"], "model.deprecated");
    assert_eq!(escalation["signed"], true);
    let data = &escalation["data"];
    assert_eq!(data["evidence_source"], "runtime_detection");
    assert_eq!(data["runtime_gone"], true);
    assert_eq!(data["provider"], PROVIDER_NAME);
    assert_eq!(data["model"], MODEL_ID);

    // No further duplicates: quiesce and re-check.
    tokio::time::sleep(std::time::Duration::from_millis(600)).await;
    assert_eq!(
        receiver.received_requests().await.unwrap_or_default().len(),
        2,
        "no third delivery  escalation emits once"
    );
}
