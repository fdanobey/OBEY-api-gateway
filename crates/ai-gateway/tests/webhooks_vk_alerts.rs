//! Virtual-key budget alert webhooks (budget-alerts-unified-webhooks,
//! Task 6) — full-router integration tests.
//!
//! Exercises the VK evaluation funnel via `tower::ServiceExt::oneshot()`
//! with a wiremock provider and a wiremock webhook receiver: request →
//! off-path `record_usage` spawn → durable spend accrual → enriched
//! evaluation under `ScopeRef::VirtualKey` → signed delivery. Also
//! covers admin VK reset events and the no-limit dormancy path.
//!
//! Covers (Reqs 1.2-1.4, 2.2, 2.3, 4.6):
//! - key with `budget_limit_usd` crossing 80% → `budget.warning`, 100%
//! → `budget.exhausted`, both with valid Standard Webhooks signatures
//! (Req 3.4)
//! - payload carries key id/name ONLY — the serialized body never
//! contains the `vk_` secret
//! - admin `POST /keys/{id}/reset` → `budget.reset` with `admin`
//! source and the NEW window anchor; the window re-arms (next
//! crossing fires again)
//! - key without a limit → no events (Req 1.9)

use axum::body::Body;
use axum::http::{Request, StatusCode};
use tower::ServiceExt;
use wiremock::matchers::{method as wm_method, path as wm_path};
use wiremock::{Mock, MockServer, ResponseTemplate};

use ai_gateway::config::*;
use ai_gateway::gateway::GatewayServer;
use ai_gateway::virtual_keys::models::CreateKeyParams;
use ai_gateway::webhooks::events::{BudgetAlertEvent, Scope};

const PROVIDER_NAME: &str = "vk-budget-provider";
const WEBHOOK_SECRET: &str = "whsec_test_secret";
const TEST_MODEL: &str = "gpt-test";

// ---------------------------------------------------------------------------
// Config / plumbing helpers
// ---------------------------------------------------------------------------

fn base_config() -> Config {
    Config {
        security: Default::default(),
        cache_aware_routing: Default::default(),
        server: ServerConfig {
            host: "127.0.0.1".to_string(),
            port: 0,
            request_timeout_seconds: 30,
            max_request_size_mb: 10,
        },
        tls: None,
        admin: AdminConfig::default(),
        dashboard: DashboardConfig::default(),
        cors: CorsConfig::default(),
        providers: vec![],
        model_groups: vec![],
        circuit_breaker: CircuitBreakerConfig::default(),
        retry: RetryConfig::default(),
        logging: LoggingConfig::default(),
        semantic_cache: None,
        exact_cache: ExactCacheConfig::default(),
        prometheus: None,
        context: ContextConfig::default(),
        compression: Default::default(),
        memory: None,
        first_launch_completed: false,
        tray: TrayConfig::default(),
        codex_instructions_url: None,
        streaming: None,
        virtual_keys: Default::default(),
        loop_detection: Default::default(),
        guardrails: None,
        tool_compression: Default::default(),
        smart_routing: Default::default(),
        xhigh_models_allowlist: Default::default(),
        reasoning_models_allowlist: Default::default(),
        codex_search: None,
        structured_output: None,
        reasoning_compat: Default::default(),
        webhooks: Default::default(),
    }
}

/// Isolate databases into `db_dir` (mirrors `common::isolate_databases`
/// but keeps the directory handle so tests can address `keys.db` /
/// `webhooks.db` directly).
fn isolated_config(db_dir: &std::path::Path) -> Config {
    let mut config = base_config();
    config.logging.database_path = db_dir.join("logs.db").to_string_lossy().into_owned();
    config.virtual_keys.database_path = db_dir.join("keys.db").to_string_lossy().into_owned();
    config
}

fn provider(uri: &str) -> Provider {
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
        budget: None,
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
        model: TEST_MODEL.to_string(),
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

/// Point webhooks at the wiremock receiver: signed endpoint for every
/// event type, with the receiver host allowlisted (Req 3.1, 4.5).
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

/// Enable required VK enforcement so requests carry a key.
fn require_virtual_keys(config: &mut Config) {
    config.virtual_keys.enforcement = EnforcementMode::Required;
}

/// Mount a mock chat-completions endpoint returning $3.00 of usage.
async fn mount_ok_mock(upstream: &MockServer, expect: u64) {
    let body = serde_json::json!({
        "id": "chatcmpl-vk-webhook-test",
        "object": "chat.completion",
        "created": 1700000000_i64,
        "model": TEST_MODEL,
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

/// Mount the webhook receiver: 200 for every POST.
async fn mount_webhook_ok(receiver: &MockServer) {
    Mock::given(wm_method("POST"))
        .respond_with(ResponseTemplate::new(200))
        .mount(receiver)
        .await;
}

/// Send a chat request authenticated with the created key. The request
/// `model` is the model GROUP name (same convention as the provider
/// alert tests): the middleware resolves rates for the responded model
/// within that group, attributing $3.00 per request.
async fn post_chat(app: axum::Router, bearer: &str, turn: &str) -> StatusCode {
    let req = Request::post("/v1/chat/completions")
        .header("content-type", "application/json")
        .header("authorization", format!("Bearer {bearer}"))
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

/// Poll until the receiver has at least `min` requests (off-path
/// delivery is spawned; give the chain time to land).
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

/// Poll keys.db until the key's durable spend reaches `min_usd` (the
/// record_usage write + counter update are spawned off the response
/// path).
async fn wait_for_key_spend(server: &GatewayServer, key_id: &str, min_usd: f64) {
    for _ in 0..100 {
        if let Ok(Some(state)) = server
            .state
            .virtual_key_manager
            .key_budget_state(key_id)
            .await
        {
            if state.current_spend_usd >= min_usd {
                return;
            }
        }
        tokio::time::sleep(std::time::Duration::from_millis(50)).await;
    }
    panic!("durable spend for key {key_id} never reached {min_usd} USD");
}

/// Assert the vk_ key secret is absent from every surface of one
/// captured delivery: webhook-* header values, the request URL, and
/// the request body (the body check mirrors the one inside
/// erify_signature_and_parse).
fn assert_vk_secret_absent(delivery: &wiremock::Request) {
    for (name, value) in &delivery.headers {
        let name = name.as_str();
        if name.starts_with("webhook-") {
            let value = String::from_utf8_lossy(value.as_bytes());
            assert!(
                !value.contains("vk_"),
                "webhook header {name} must never carry the vk_ secret: {value}"
            );
        }
    }
    assert!(
        !delivery.url.as_str().contains("vk_"),
        "delivery URL must never carry the vk_ secret: {}",
        delivery.url
    );
    let body = String::from_utf8_lossy(&delivery.body);
    assert!(
        !body.contains("vk_"),
        "delivery body must never carry the vk_ secret: {body}"
    );
}
/// Bounded absence poll (test-quality remediation): the durable spend
/// poll already confirmed the funnel ran; assert the ledger stays
/// empty for a short bounded window (stable across several reads)
/// before trusting the negative.
async fn assert_no_ledger_events(server: &GatewayServer, poll_ms: u64) {
    let ledger = &server.state.webhook_ledger;
    let mut last = ledger.recent_events(10);
    let mut stable_rounds = 0;
    let deadline = std::time::Instant::now() + std::time::Duration::from_millis(poll_ms);
    while std::time::Instant::now() < deadline {
        tokio::time::sleep(std::time::Duration::from_millis(50)).await;
        let current = ledger.recent_events(10);
        if current.is_empty() && last.is_empty() && stable_rounds >= 2 {
            return;
        }
        if current.len() != last.len() {
            stable_rounds = 0;
        } else {
            stable_rounds += 1;
        }
        last = current;
    }
    assert!(
        last.is_empty(),
        "ledger unexpectedly holds {} event rows: {last:?}",
        last.len()
    );
}
/// Parse a delivered request into an event envelope + assert the
/// Standard Webhooks signature over the exact body bytes (Req 3.4):
/// `v1,base64(HMAC-SHA256(key, "{id}.{timestamp}.{body}"))` where the
/// key is the base64-decoded secret (after the optional `whsec_`
/// prefix strip). Also asserts the no-secret-leak invariant (Req 1.2):
/// the body never contains the `vk_` prefix anywhere.
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

    let body_str = String::from_utf8_lossy(body);
    assert!(
        !body_str.contains("vk_"),
        "event payload must never contain the vk_ key secret: {body_str}"
    );
    serde_json::from_slice(body).expect("body parses as a BudgetAlertEvent")
}

fn create_defaults() -> CreateKeyParams {
    CreateKeyParams {
        name: None,
        budget_limit_usd: None,
        token_budget: None,
        budget_window: None,
        requests_per_minute: None,
        tokens_per_minute: None,
        model_access: None,
        expires_in: None,
        loop_detection: None,
    }
}

/// Standard test rig: upstream mock, webhook receiver, isolated DBs,
/// required VK enforcement, budget-group model group.
async fn rig<'a>(
    upstream_expect: u64,
) -> (GatewayServer, axum::Router, MockServer, MockServer, tempfile::TempDir) {
    let upstream = MockServer::start().await;
    let receiver = MockServer::start().await;
    let db_dir = tempfile::tempdir().expect("temp dir");

    let mut config = isolated_config(db_dir.path());
    config.providers = vec![provider(&upstream.uri())];
    config.model_groups = vec![ModelGroup {
        name: "budget-group".to_string(),
        version_fallback_enabled: false,
        compression: None,
        structured_output: None,
        memory: None,
        models: vec![expensive_model()],
    }];
    require_virtual_keys(&mut config);
    configure_webhook_receiver(&mut config, &receiver);

    let server = GatewayServer::new(config, None).await.unwrap();
    let app = server.build_router();
    mount_ok_mock(&upstream, upstream_expect).await;
    mount_webhook_ok(&receiver).await;
    (server, app, upstream, receiver, db_dir)
}

// ---------------------------------------------------------------------------
// Threshold crossing: warning → exhausted (Reqs 1.2-1.4)
// ---------------------------------------------------------------------------

/// Validates: Requirements 1.2, 1.3, 1.4, 3.4
#[tokio::test]
async fn key_spend_crossing_emits_signed_warning_then_exhausted() {
    let (server, app, _upstream, receiver, _db) = rig(2).await;

    // $5.00 limit (manual window semantics: lifetime until reset); $3.00
    // accrual #1 (60%) then $6.00 total (120% — warning + exhausted on
    // the same second accrual). Warn fixed at 80% ($4) — design D9.
    let created = server
        .state
        .virtual_key_manager
        .create_key(CreateKeyParams {
            name: Some("crossing-key".to_string()),
            budget_limit_usd: Some(5.0),
            ..create_defaults()
        })
        .await
        .unwrap();

    // Accrual #1: $3.00 = 60% — below the warn line, no events.
    let status = post_chat(app.clone(), &created.key, "one").await;
    assert_eq!(status, StatusCode::OK);
    wait_for_key_spend(&server, &created.id, 3.0).await;
    // Durable accrual confirmed; bounded absence poll trusts the
    // negative only after the funnel demonstrably quiesced.
    assert_no_ledger_events(&server, 600).await;

    // Accrual #2: $6.00 total = 120% — warn + exhausted in order.
    let status = post_chat(app, &created.key, "two").await;
    assert_eq!(status, StatusCode::OK);
    wait_for_key_spend(&server, &created.id, 6.0).await;

    let deliveries = wait_for_deliveries(&receiver, 2).await;
    assert_eq!(deliveries.len(), 2, "warning + exhausted, exactly");
    // Each event's delivery is a separately spawned task, so the two
    // arrivals may race: parse both and assert the pair, not the order.
    let first = verify_signature_and_parse(&deliveries[0]);
    let second = verify_signature_and_parse(&deliveries[1]);
    let types = [first.event_type, second.event_type];
    assert!(
        types.contains(&BudgetEventType::BudgetWarning)
            && types.contains(&BudgetEventType::BudgetExhausted),
        "expected warning + exhausted, got {types:?}"
    );
    let warn = if first.event_type == BudgetEventType::BudgetWarning {
        &first
    } else {
        &second
    };
    for event in [&first, &second] {
        assert_eq!(event.data.scope, Scope::VirtualKey);
        assert_eq!(event.data.name.as_deref(), Some("crossing-key"));
        assert_eq!(event.data.key_id.as_deref(), Some(created.id.as_str()));
        assert_eq!(event.data.limit_usd, Some(5.0));
    }
    assert_eq!(warn.data.percent, Some(80));
    assert_eq!(first.data.spent_usd, Some(6.0));
    assert_eq!(second.data.spent_usd, Some(6.0));

    // The vk_ secret must never leak anywhere on the wire or in the
    // ledger: scan the webhook-* header values, the request URL, the
    // request body, and the ledger's payload_json rows.
    for delivery in &deliveries {
        assert_vk_secret_absent(delivery);
    }
    for row in server.state.webhook_ledger.recent_events(10) {
        assert!(
            !row.payload_json.contains("vk_"),
            "ledger payload_json must never contain the vk_ secret: {}",
            row.payload_json
        );
    }
}

// ---------------------------------------------------------------------------
// Window reset re-arms (Req 1.6 via admin reset; Reqs 2.2, 2.3)
// ---------------------------------------------------------------------------

/// Validates: Requirements 2.2, 2.3, 4.6
#[tokio::test]
async fn admin_reset_emits_reset_and_rearms_crossing() {
    let (server, app, _upstream, receiver, _db) = rig(4).await;

    let created = server
        .state
        .virtual_key_manager
        .create_key(CreateKeyParams {
            name: Some("reset-key".to_string()),
            budget_limit_usd: Some(5.0),
            ..create_defaults()
        })
        .await
        .unwrap();

    // Exhaust the key: $6.00 total (120%) → warn + exhausted.
    let status = post_chat(app.clone(), &created.key, "one").await;
    assert_eq!(status, StatusCode::OK);
    wait_for_key_spend(&server, &created.id, 3.0).await;
    let status = post_chat(app.clone(), &created.key, "two").await;
    assert_eq!(status, StatusCode::OK);
    wait_for_key_spend(&server, &created.id, 6.0).await;
    let _ = wait_for_deliveries(&receiver, 2).await;

    // Admin VK reset → budget.reset(admin) with the NEW anchor.
    let reset_req = Request::post(format!("/admin/keys/{}/reset", created.id))
        .body(Body::empty())
        .unwrap();
    let resp = app.clone().oneshot(reset_req).await.unwrap();
    assert_eq!(resp.status(), StatusCode::OK);

    let deliveries = wait_for_deliveries(&receiver, 3).await;
    let reset = verify_signature_and_parse(&deliveries[2]);
    assert_eq!(reset.event_type, BudgetEventType::BudgetReset);
    assert_eq!(
        reset.data.reset_source,
        Some(ai_gateway::webhooks::events::ResetSource::Admin)
    );
    assert_eq!(reset.data.scope, Scope::VirtualKey);
    assert_eq!(reset.data.key_id.as_deref(), Some(created.id.as_str()));
    let new_anchor = reset
        .data
        .window_start
        .expect("reset carries the new window_start");

    // Re-arm (Req 1.6): the next crossing fires again. Post-reset the
    // spend is $0; one more $3.00 accrual is 60% — need 80%+ so we
    // cross via a second post-reset request ($6.00 → warn + exhausted).
    let status = post_chat(app.clone(), &created.key, "three").await;
    assert_eq!(status, StatusCode::OK);
    wait_for_key_spend(&server, &created.id, 3.0).await;
    let status = post_chat(app, &created.key, "four").await;
    assert_eq!(status, StatusCode::OK);
    wait_for_key_spend(&server, &created.id, 6.0).await;

    // 2 (initial warn+exhausted) + 1 (reset) + 2 (re-armed warn+exhausted).
    let deliveries = wait_for_deliveries(&receiver, 5).await;
    assert_eq!(deliveries.len(), 5, "warn + exhausted twice plus reset, exactly");
    // Deliveries 3 and 4 (post-reset re-arm) may race like the initial
    // pair; assert the pair, not the order.
    let rewarn = verify_signature_and_parse(&deliveries[3]);
    let reexhausted = verify_signature_and_parse(&deliveries[4]);
    let types = [rewarn.event_type, reexhausted.event_type];
    assert!(
        types.contains(&BudgetEventType::BudgetWarning)
            && types.contains(&BudgetEventType::BudgetExhausted),
        "expected re-armed warning + exhausted, got {types:?}"
    );
    assert!(rewarn.data.window_start.unwrap_or(new_anchor) >= new_anchor);
}

// ---------------------------------------------------------------------------
// No-limit key → no events (Req 1.9)
// ---------------------------------------------------------------------------

/// Validates: Requirement 1.9 (VK analogue)
#[tokio::test]
async fn key_without_limit_emits_no_events() {
    let (server, app, _upstream, receiver, _db) = rig(1).await;

    let created = server
        .state
        .virtual_key_manager
        .create_key(create_defaults())
        .await
        .unwrap();

    let status = post_chat(app, &created.key, "one").await;
    assert_eq!(status, StatusCode::OK);
    wait_for_key_spend(&server, &created.id, 3.0).await;

    // Durable accrual confirmed above; bounded absence poll.
    assert_no_ledger_events(&server, 600).await;
    assert!(
        receiver.received_requests().await.unwrap().is_empty(),
        "no deliveries for a no-limit key"
    );
}
