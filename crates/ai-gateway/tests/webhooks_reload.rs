//! Config-reload hardening for the budget-alert webhooks subsystem
//! (budget-alerts-unified-webhooks, Task 10; design "Integration Points >
//! Config reload" + Error Handling 6).
//!
//! Full-router integration tests via `tower::ServiceExt::oneshot()` with
//! wiremock webhook receivers, following the `webhooks_provider_alerts.rs`
//! conventions (isolated temp-dir databases — `webhooks.db` is a sibling
//! of the redirected logs.db, so `common::isolate_databases` covers it).
//!
//! Covers:
//! - valid reload swapping endpoints: events after the reload deliver to
//! the NEW endpoint only (the notifier reads the endpoint set under the
//! config read lock per emit — no explicit re-seed needed)
//! - dormant→active reload: the provider evaluation funnel (a
//! startup-derived, non-live-read piece) is installed by
//! `apply_runtime_config_update`, so provider threshold crossings fire on
//! a gateway that started dormant
//! - invalid reload payload (non-https, non-allowlisted webhook URL):
//! the reload handler rejects it BEFORE `apply_runtime_config_update`,
//! keeping the working endpoint set (fail-closed reload semantics)
//! - fired-state registry KEPT across reload: no duplicate warning for
//! the same window after a reload (windows never reset on reload)

use axum::body::Body;
use axum::http::{Request, StatusCode};
use tower::ServiceExt;
use wiremock::matchers::{method as wm_method, path as wm_path};
use wiremock::{Mock, MockServer, ResponseTemplate};

use ai_gateway::config::*;
use ai_gateway::gateway::GatewayServer;
use ai_gateway::webhooks::events::BudgetAlertEvent;

use crate::common;

const PROVIDER_NAME: &str = "budget-provider";
const WEBHOOK_SECRET: &str = "whsec_test_secret";

// ---------------------------------------------------------------------------
// Config / plumbing helpers (webhooks_provider_alerts.rs conventions)
// ---------------------------------------------------------------------------

fn base_config() -> Config {
    let mut config = common::budget_test_config(None, None);
    // Full-config validation (run by PUT before persist) rejects port 0.
    config.server.port = 8080;
    config
}

fn provider_with_budget(uri: &str, limit_usd: f64) -> Provider {
    Provider {
        name: PROVIDER_NAME.to_string(),
        provider_type: "openai".to_string(),
        base_url: Some(uri.to_string()),
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
        rate_limit_max_wait_ms: None,
        max_connections: 10,
        rate_limit_per_minute: 0,
        custom_headers: Default::default(),
        user_agent: None,
        connection_pool: ProviderConnectionPoolConfig::default(),
        budget: Some(ProviderBudgetConfig {
            limit_usd,
            reset_policy: Some(BudgetResetPolicy::Manual),
            alert_threshold_percent: None,
        }),
        manual_models: vec![],
        global_inference_profile: false,
        cross_region_inference: false,
        custom_vpc_endpoint: false,
        prompt_caching: false,
        compression: None,
        reasoning: true,
        codex_base_url_override: None,
        codex_model_override: None,
        instructions_override: None,
        max_rate_limit_cooldown_seconds: None,
        memory: None,
    }
}

/// $1/M input, $2/M output → 1M prompt + 1M completion = $3.00 per call.
fn expensive_model() -> ProviderModel {
    ProviderModel {
        provider: PROVIDER_NAME.to_string(),
        model: "gpt-test".to_string(),
        cost_per_million_input_tokens: 1.0,
        cost_per_million_output_tokens: 2.0,
        cost_per_million_cache_read_input_tokens: None,
        cost_per_million_cache_creation_input_tokens: None,
        cache_min_tokens: None,
        cache_support: None,
        structured_output_passthrough: None,
        priority: 100,
        tier: None,
        context_window: 0,
        specializations: vec![],
        cost_per_million_reasoning_tokens: None,
        reasoning_family: None,
        reasoning_parameter: None,
    }
}

/// Point webhooks at a wiremock receiver: signed endpoint for every event
/// type, with the receiver host allowlisted (wiremock binds 127.0.0.1, an
/// http URL — the operator LAN escape hatch, Req 3.1/4.5).
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

async fn mount_ok_mock(upstream: &MockServer, expect: u64) {
    let body = serde_json::json!({
        "id": "chatcmpl-webhook-reload-test",
        "object": "chat.completion",
        "created": 1700000000_i64,
        "model": "gpt-test",
        "choices": [{
            "index": 0,
            "message": {"role": "assistant", "content": "ok"},
            "finish_reason": "stop"
        }],
        "usage": {
            "prompt_tokens": 1_000_000,
            "completion_tokens": 1_000_000,
            "total_tokens": 2_000_000
        }
    });
    Mock::given(wm_method("POST"))
        .and(wm_path("/v1/chat/completions"))
        .respond_with(ResponseTemplate::new(200).set_body_json(body))
        .expect(expect)
        .mount(upstream)
        .await;
}

async fn mount_webhook_ok(receiver: &MockServer) {
    Mock::given(wm_method("POST"))
        .respond_with(ResponseTemplate::new(200))
        .mount(receiver)
        .await;
}

async fn post_chat(app: axum::Router, turn: &str) -> StatusCode {
    let req = Request::post("/v1/chat/completions")
        .header("content-type", "application/json")
        .body(Body::from(
            serde_json::to_string(&serde_json::json!({
                "model": "budget-group",
                "messages": [{"role": "user", "content": format!("hello {turn}")}]
            }))
            .unwrap(),
        ))
        .unwrap();
    let resp = app.oneshot(req).await.unwrap();
    resp.status()
}

/// Poll until the receiver has at least `min` requests (off-path delivery
/// is spawned; give the chain time to land).
async fn wait_for_deliveries(receiver: &MockServer, min: usize) -> Vec<wiremock::Request> {
    for _ in 0..200 {
        let requests = receiver.received_requests().await.unwrap_or_default();
        if requests.len() >= min {
            return requests;
        }
        tokio::time::sleep(std::time::Duration::from_millis(25)).await;
    }
    panic!("webhook receiver never received {min} deliveries");
}

/// Poll the durable budget store until `provider`'s spend reaches at
/// least `min_usd` (the accrual write is spawned off the response path).
async fn wait_for_store_spend(server: &GatewayServer, provider: &str, min_usd: f64) {
    let store = server
        .state
        .metrics
        .provider_budget_store()
        .expect("store attached");
    for _ in 0..100 {
        if let Ok(rows) = store.get_all() {
            if let Some(row) = rows.iter().find(|row| row.provider == provider) {
                if row.current_spend_usd >= min_usd {
                    return;
                }
            }
        }
        tokio::time::sleep(std::time::Duration::from_millis(50)).await;
    }
    panic!("durable spend for {provider} never reached {min_usd} USD");
}

/// Parse a delivered request into an event envelope + assert the Standard
/// Webhooks signature over the exact body bytes (Req 3.4):
/// `v1,base64(...)` with the base64-decoded secret as HMAC key.
fn verify_signature_and_parse(request: &wiremock::Request) -> BudgetAlertEvent {
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

    serde_json::from_slice(body).expect("body parses as a BudgetAlertEvent")
}

// ---------------------------------------------------------------------------
// Valid reload swapping endpoints: new events → new endpoint only
// ---------------------------------------------------------------------------

// Validates: design "Config reload" (endpoint set follows the live
// config; in-flight retries unaffected) + Error Handling 6.
#[tokio::test]
async fn valid_reload_swaps_endpoint_set_for_new_events() {
    let upstream = MockServer::start().await;
    let old_receiver = MockServer::start().await;
    let new_receiver = MockServer::start().await;

    let mut config = base_config();
    common::isolate_databases(&mut config);
    // Manual window so spend never rolls over mid-test; $5 limit.
    config.providers = vec![provider_with_budget(&upstream.uri(), 5.0)];
    config.model_groups = vec![ModelGroup {
        name: "budget-group".to_string(),
        version_fallback_enabled: false,
        compression: None,
        structured_output: None,
        memory: None,
        models: vec![expensive_model()],
    }];
    configure_webhook_receiver(&mut config, &old_receiver);

    let server = GatewayServer::new(config.clone(), None).await.unwrap();
    let app = server.build_router();
    mount_ok_mock(&upstream, 2).await;
    mount_webhook_ok(&old_receiver).await;
    mount_webhook_ok(&new_receiver).await;

    // Request #1: $3.00 of $5.00 — below the 80% warn line ($4); nothing
    // fires yet. (The old receiver is wired but receives nothing.)
    let status = post_chat(app.clone(), "one").await;
    assert_eq!(status, StatusCode::OK);
    wait_for_store_spend(&server, PROVIDER_NAME, 3.0).await;

    // Valid reload: swap the endpoint set from the old receiver to the
    // new one. The notifier reads endpoints under the config read lock
    // per emit, so the swap needs no explicit re-seed.
    let mut reloaded = config.clone();
    configure_webhook_receiver(&mut reloaded, &new_receiver);
    ai_gateway::gateway::apply_runtime_config_update(&server.state, reloaded)
        .await;

    // Request #2: +$3.00 → $6.00 ≥ $5.00 — warning ($4) and exhausted
    // ($5) cross on this accrual, AFTER the reload → new receiver only.
    // (Arrival order across the two spawned delivery tasks is not
    // guaranteed — assert the multiset of event types.)
    let status = post_chat(app, "two").await;
    assert_eq!(status, StatusCode::OK);
    wait_for_store_spend(&server, PROVIDER_NAME, 6.0).await;

    let deliveries = wait_for_deliveries(&new_receiver, 2).await;
    assert_eq!(deliveries.len(), 2);
    let mut event_types: Vec<BudgetEventType> = deliveries
        .iter()
        .map(verify_signature_and_parse)
        .map(|event| event.event_type)
        .collect();
    event_types.sort_by_key(|kind| format!("{kind:?}"));
    assert_eq!(
        event_types,
        vec![
            BudgetEventType::BudgetExhausted,
            BudgetEventType::BudgetWarning
        ]
    );

    // The old receiver must have received nothing after the reload.
    assert!(
        old_receiver.received_requests().await.unwrap().is_empty(),
        "events emitted after the reload must deliver to the new endpoint set only"
    );
}

// ---------------------------------------------------------------------------
// Fired-state registry KEPT across reload (no window reset, no duplicate)
// ---------------------------------------------------------------------------

// Validates: design "Config reload" (fired-state registry kept — windows
// don't reset on reload; Reqs 1.5/1.6 unaffected by reload).
#[tokio::test]
async fn reload_keeps_fired_state_no_duplicate_within_window() {
    let upstream = MockServer::start().await;
    let receiver = MockServer::start().await;

    let mut config = base_config();
    common::isolate_databases(&mut config);
    // Manual window; $30 limit (warn at $24 = request #8; exhausted at
    // $30 = request #10). Every request stays at-or-under the limit so
    // the router never budget-skips — #10 lands exactly at $30.
    config.providers = vec![provider_with_budget(&upstream.uri(), 30.0)];
    config.model_groups = vec![ModelGroup {
        name: "budget-group".to_string(),
        version_fallback_enabled: false,
        compression: None,
        structured_output: None,
        memory: None,
        models: vec![expensive_model()],
    }];
    configure_webhook_receiver(&mut config, &receiver);

    let server = GatewayServer::new(config.clone(), None).await.unwrap();
    let app = server.build_router();
    mount_ok_mock(&upstream, 10).await;
    mount_webhook_ok(&receiver).await;

    // Requests #1-#10: $3.00 each → $30.00 total; warn fires at #8 ($24),
    // exhausted at #10 ($30) — arrived-order-agnostic multiset check.
    for turn in [
        "one", "two", "three", "four", "five", "six", "seven", "eight", "nine", "ten",
    ] {
        let status = post_chat(app.clone(), turn).await;
        assert_eq!(status, StatusCode::OK);
    }
    wait_for_store_spend(&server, PROVIDER_NAME, 30.0).await;
    let deliveries = wait_for_deliveries(&receiver, 2).await;
    assert_eq!(deliveries.len(), 2);
    let mut event_types: Vec<BudgetEventType> = deliveries
        .iter()
        .map(verify_signature_and_parse)
        .map(|event| event.event_type)
        .collect();
    event_types.sort_by_key(|kind| format!("{kind:?}"));
    assert_eq!(
        event_types,
        vec![
            BudgetEventType::BudgetExhausted,
            BudgetEventType::BudgetWarning
        ]
    );

    // Reload with the SAME endpoint set (a valid no-op-ish reload): the
    // fired-state registry must survive the config swap.
    ai_gateway::gateway::apply_runtime_config_update(&server.state, config.clone())
        .await;

    // REAL post-reload evaluation through the funnel: the provider is
    // AT the limit, so an eleventh chat request would budget-skip;
    // instead persist one more over-limit accrual the way the
    // admin/rollover paths do — durable add_spend in the same window
    // (anchor unchanged) plus the matching in-memory add_cost the
    // funnel read-back compares against, then drive the evaluation
    // hook with the durable read-back exactly like
    // persist_provider_spend does after its write.
    let store = server
        .state
        .metrics
        .provider_budget_store()
        .expect("store attached");
    let anchor = store
        .get_window_start(PROVIDER_NAME)
        .expect("window start readable")
        .expect("row exists after first ten requests");
    store
        .add_spend(PROVIDER_NAME, 3.0, anchor)
        .expect("durable $33.00 same window");
    server.state.metrics.add_cost(PROVIDER_NAME, 3.0);
    let (total_usd, anchor) = store
        .get_spend_and_window(PROVIDER_NAME)
        .expect("read-back")
        .expect("row present");
    server
        .state
        .provider_budget_alerts
        .evaluate_provider(&server.state.config, PROVIDER_NAME, total_usd, anchor)
        .await;

    // The same-window re-cross must NOT re-announce warn or exhausted:
    // the count stays at 2 across a bounded stability window AFTER the
    // funnel demonstrably ran (durable read-back + evaluation above),
    // and the ledger still holds the original pair.
    for _ in 0..6 {
        tokio::time::sleep(std::time::Duration::from_millis(100)).await;
        assert_eq!(
            receiver.received_requests().await.unwrap().len(),
            2,
            "fired state must be kept across reload — no duplicate events within the window"
        );
    }
    let events = server.state.webhook_ledger.recent_events(10);
    assert_eq!(events.len(), 2, "ledger holds exactly the original pair");
    assert!(
        events.iter().any(|row| row.event_type == "budget.warning")
            && events
                .iter()
                .any(|row| row.event_type == "budget.exhausted"),
        "original warn+exhausted pair intact: {events:?}"
    );
}

// ---------------------------------------------------------------------------
// Dormant → active reload: provider evaluation funnel activates
// ---------------------------------------------------------------------------

// Validates: Task 10 reload hardening (the startup-derived provider
// evaluation funnel is installed by apply_runtime_config_update; Reqs
// 1.1/1.3/1.4 fire on a gateway that started dormant).
#[tokio::test]
async fn dormant_to_active_reload_activates_provider_evaluation() {
    let upstream = MockServer::start().await;
    let receiver = MockServer::start().await;

    let mut config = base_config();
    common::isolate_databases(&mut config);
    // Manual window; $30 limit (warn at $24 = request #8; exhausted at
    // $30 = request #10) — every request stays at-or-under the limit so
    // the router never budget-skips.
    config.providers = vec![provider_with_budget(&upstream.uri(), 30.0)];

    config.model_groups = vec![ModelGroup {
        name: "budget-group".to_string(),
        version_fallback_enabled: false,
        compression: None,
        structured_output: None,
        memory: None,
        models: vec![expensive_model()],
    }];
    // Start DORMANT: no endpoints configured (Req 3.3) — startup skips
    // installing the provider evaluation funnel entirely.

    let server = GatewayServer::new(config.clone(), None).await.unwrap();
    let app = server.build_router();
    mount_ok_mock(&upstream, 10).await;
    mount_webhook_ok(&receiver).await;

    // Requests #1-#7 while dormant: $21.00 of $30.00 — below the warn
    // line, and dormancy means no evaluation, no ledger rows.
    for turn in ["one", "two", "three", "four", "five", "six", "seven"] {
        let status = post_chat(app.clone(), turn).await;
        assert_eq!(status, StatusCode::OK);
    }
    wait_for_store_spend(&server, PROVIDER_NAME, 21.0).await;
    tokio::time::sleep(std::time::Duration::from_millis(300)).await;
    assert!(
        server.state.webhook_ledger.recent_events(10).is_empty(),
        "dormant gateway must not evaluate or ledger anything"
    );

    // Activate via reload: endpoints appear; the reload path installs
    // the provider evaluation funnel. The fired-state registry starts
    // EMPTY (restart re-derivation seeding is startup-only) — the next
    // accrual re-derives from durable spend per the D11 tolerance.
    let mut reloaded = config.clone();
    configure_webhook_receiver(&mut reloaded, &receiver);
    ai_gateway::gateway::apply_runtime_config_update(&server.state, reloaded)
        .await;

    // Requests #8-#10: $24.00 (warn) then $27.00 then $30.00
    // (exhausted) — with the funnel now live the accruals evaluate and
    // fire; arrival-order-agnostic multiset check.
    for turn in ["eight", "nine", "ten"] {
        let status = post_chat(app.clone(), turn).await;
        assert_eq!(status, StatusCode::OK);
    }
    wait_for_store_spend(&server, PROVIDER_NAME, 30.0).await;

    let deliveries = wait_for_deliveries(&receiver, 2).await;
    let mut event_types: Vec<BudgetEventType> = deliveries
        .iter()
        .map(verify_signature_and_parse)
        .map(|event| event.event_type)
        .collect();
    event_types.sort_by_key(|kind| format!("{kind:?}"));
    assert_eq!(
        event_types,
        vec![
            BudgetEventType::BudgetExhausted,
            BudgetEventType::BudgetWarning
        ]
    );
}

// ---------------------------------------------------------------------------
// Invalid reload payload keeps the working endpoint set (fail-closed)
// ---------------------------------------------------------------------------

// Validates: Task 10 fail-closed reload semantics — an invalid webhooks
// URL in a reload payload is rejected by validation BEFORE any config
// swap, so the previously working endpoint set keeps delivering.
#[tokio::test]
async fn invalid_reload_payload_keeps_old_endpoint_set() {
    let upstream = MockServer::start().await;
    let receiver = MockServer::start().await;

    let mut config = base_config();
    common::isolate_databases(&mut config);
    // Manual window; $30 limit (warn at $24 = request #8; exhausted at
    // $30 = request #10) — every request stays at-or-under the limit so
    // the router never budget-skips.
    config.providers = vec![provider_with_budget(&upstream.uri(), 30.0)];

    config.model_groups = vec![ModelGroup {
        name: "budget-group".to_string(),
        version_fallback_enabled: false,
        compression: None,
        structured_output: None,
        memory: None,
        models: vec![expensive_model()],
    }];
    configure_webhook_receiver(&mut config, &receiver);

    // PUT /admin/webhooks persists YAML through save_config — keep the
    // config dir alive like the webhooks_admin tests do.
    let config_dir = tempfile::tempdir().expect("temp config dir").keep();
    let config_path = config_dir.join("config.yaml");

    let server = GatewayServer::new(config.clone(), Some(config_path))
        .await
        .unwrap();
    let app = server.build_router();
    mount_ok_mock(&upstream, 10).await;
    mount_webhook_ok(&receiver).await;

    // Attempt an INVALID webhooks payload through the admin webhooks PUT
    // (the same validate → persist → apply pipeline as config reload):
    // an http URL for a host NOT in allowed_private_hosts fails
    // validation (Req 4.5) → 400, nothing applied.
    let invalid = WebhooksConfig {
        endpoints: vec![WebhookEndpoint {
            url: "http://evil-inject.example.net/hook".to_string(),
            secret: Some(WEBHOOK_SECRET.to_string()),
            events: None,
        }],
        allowed_private_hosts: vec![],
    };
    let req = Request::put("/admin/webhooks")
        .header("content-type", "application/json")
        .body(Body::from(serde_json::to_string(&invalid).unwrap()))
        .unwrap();
    let resp = app.clone().oneshot(req).await.unwrap();
    assert_eq!(resp.status(), StatusCode::BAD_REQUEST);

    // The old endpoint set is intact: threshold crossings still deliver
    // to the ORIGINAL receiver ($24 warn + $30 exhausted across ten
    // $3.00 requests; arrival-order-agnostic multiset check).
    for turn in [
        "one", "two", "three", "four", "five", "six", "seven", "eight", "nine", "ten",
    ] {
        let status = post_chat(app.clone(), turn).await;
        assert_eq!(status, StatusCode::OK);
    }
    wait_for_store_spend(&server, PROVIDER_NAME, 30.0).await;
    let deliveries = wait_for_deliveries(&receiver, 2).await;
    assert_eq!(deliveries.len(), 2);
    let mut event_types: Vec<BudgetEventType> = deliveries
        .iter()
        .map(verify_signature_and_parse)
        .map(|event| event.event_type)
        .collect();
    event_types.sort_by_key(|kind| format!("{kind:?}"));
    assert_eq!(
        event_types,
        vec![
            BudgetEventType::BudgetExhausted,
            BudgetEventType::BudgetWarning
        ]
    );
}
