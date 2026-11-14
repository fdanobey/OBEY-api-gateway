//! Dispatch + failover integration tests for the github-copilot
//! provider (spec `github-copilot-oauth-provider`, task 13).
//!
//! Drives the REAL gateway router (`Router::route_request` /
//! `route_request_streaming`) against wiremock upstreams — the repo's
//! no-port-binding pattern — with authenticated Copilot sessions seeded
//! into an isolated `CopilotTokenManager` (same helpers as the
//! colocated `copilot_wiring` tests in `src/router/router.rs`).
//!
//! Coverage (requirements in brackets):
//! - non-streaming dispatch carries the full mandatory header set and
//!   the Copilot-token bearer [4.1, 4.2]
//! - streaming client requests ride the buffered-dispatch path and
//!   return a reassembled response — no Copilot-specific streaming
//!   behavior [4.5]
//! - upstream 401 → exactly one refresh exchange + one re-dispatch,
//!   then success; both-401 → failover to the second provider [4.3]
//! - upstream 429 with `x-ratelimit-*` headers → failover + usage
//!   tracker snapshot populated [4.4]
//! - unauthenticated Copilot candidate is skipped
//!   (`copilot_auth_skip` ledger row) and the fallback serves [9.2]
//! - successful dispatch records usage with zero cost and full
//!   model/provider attribution [9.3]
//! - `auth_method: api-key` uses the PAT bearer; a missing PAT is an
//!   actionable error with no fallback to OAuth tokens [8.1]
//!
//! No real network: every upstream (dispatch, token exchange) is a
//! wiremock server. No token literals appear in assertion output.

mod common;

use std::sync::Arc;

use ai_gateway::active_requests::{
    ActivePhase, ActiveRequestInfo, ActiveRequestRegistry, RequestKind,
};
use ai_gateway::config::{Config, ModelGroup, Provider, ProviderModel};
use ai_gateway::metrics::Metrics;
use ai_gateway::models::openai::{Message, OpenAIRequest};
use ai_gateway::oauth::github_copilot::device_flow::DeviceFlowEndpoints;
use ai_gateway::oauth::github_copilot::manager::{CopilotEndpoints, CopilotTokenManager};
use ai_gateway::oauth::github_copilot::store::{CopilotStoredCredentials, CopilotTokenStore};
use ai_gateway::oauth::UsageTracker;
use ai_gateway::router::router::{Router, StreamingResponse};
use tokio::sync::RwLock;
use wiremock::matchers::{method, path};
use wiremock::{Mock, MockServer, ResponseTemplate};

// ─── Shared helpers (mirror src/router/router.rs copilot_wiring) ──────────

/// Process-wide secure dir backing the encrypted Copilot token store.
/// Sets `AI_GATEWAY_DATA_DIR` once and loads the master key; per-test
/// stores get unique file names inside it (wiring-test pattern).
fn secure_dir() -> &'static std::path::Path {
    use std::sync::OnceLock;
    static DIR: OnceLock<std::path::PathBuf> = OnceLock::new();
    DIR.get_or_init(|| {
        let dir = tempfile::tempdir().expect("temp secure dir");
        let leaked = Box::leak(Box::new(dir.keep()));
        std::env::set_var("AI_GATEWAY_DATA_DIR", &*leaked);
        let _ = ai_gateway::secrets::load_or_create_master_key();
        leaked.to_path_buf()
    })
}

fn unique_store(tag: &str) -> CopilotTokenStore {
    static COUNTER: std::sync::atomic::AtomicU64 = std::sync::atomic::AtomicU64::new(0);
    let n = COUNTER.fetch_add(1, std::sync::atomic::Ordering::SeqCst);
    CopilotTokenStore::new(secure_dir().join(format!("dispatch_{tag}_{n}.enc")))
}

fn now_plus(secs: u64) -> u64 {
    std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .unwrap()
        .as_secs()
        .saturating_add(secs)
}

fn test_provider(name: &str, base_url: String) -> Provider {
    Provider {
        name: name.to_string(),
        provider_type: "openai".to_string(),
        base_url: Some(base_url),
        api_key_env: None,
        api_key_encrypted: None,
        api_secret_env: None,
        api_secret_encrypted: None,
        auth_method: None,
        resolved_api_key: None,
        resolved_api_secret: None,
        region: None,
        timeout_seconds: 30,
        ttfb_timeout_seconds: Some(5),
        total_timeout_seconds: Some(5),
        buffered_upstream_streaming: None,
        rate_limit_max_wait_ms: None,
        max_connections: 10,
        rate_limit_per_minute: 0,
        custom_headers: Default::default(),
        user_agent: None,
        connection_pool: Default::default(),
        budget: None,
        manual_models: vec![],
        global_inference_profile: false,
        cross_region_inference: false,
        prompt_caching: false,
        compression: None,
        memory: None,
        custom_vpc_endpoint: false,
        reasoning: true,
        codex_base_url_override: None,
        codex_model_override: None,
        instructions_override: None,
        max_rate_limit_cooldown_seconds: None,
    }
}

fn copilot_provider(name: &str, base_url: String) -> Provider {
    let mut p = test_provider(name, base_url);
    p.provider_type = "github-copilot".to_string();
    p.auth_method = None; // default device-oauth
    p
}

fn test_model_named(provider: &str, model: &str, priority: u32) -> ProviderModel {
    ProviderModel {
        provider: provider.to_string(),
        model: model.to_string(),
        cost_per_million_input_tokens: 0.0,
        cost_per_million_output_tokens: 0.0,
        priority,
        structured_output_passthrough: None,
        tier: None,
        context_window: 0,
        specializations: vec![],
        cache_min_tokens: None,
        cache_support: None,
        cost_per_million_cache_read_input_tokens: None,
        cost_per_million_cache_creation_input_tokens: None,
        cost_per_million_reasoning_tokens: None,
        reasoning_family: None,
        reasoning_parameter: None,
    }
}

fn test_group(models: Vec<ProviderModel>) -> ModelGroup {
    ModelGroup {
        name: "test-group".to_string(),
        version_fallback_enabled: false,
        compression: None,
        structured_output: None,
        memory: None,
        models,
    }
}

fn base_config() -> Config {
    Config {
        security: Default::default(),
        server: ai_gateway::config::ServerConfig {
            host: "0.0.0.0".to_string(),
            port: 8080,
            request_timeout_seconds: 30,
            max_request_size_mb: 10,
        },
        tls: None,
        admin: Default::default(),
        dashboard: Default::default(),
        cors: Default::default(),
        providers: vec![],
        model_groups: vec![],
        circuit_breaker: Default::default(),
        retry: Default::default(),
        logging: Default::default(),
        semantic_cache: None,
        exact_cache: Default::default(),
        prometheus: None,
        context: Default::default(),
        compression: Default::default(),
        structured_output: None,
        first_launch_completed: false,
        tray: Default::default(),
        codex_instructions_url: None,
        streaming: None,
        virtual_keys: Default::default(),
        loop_detection: Default::default(),
        guardrails: None,
        tool_compression: Default::default(),
        smart_routing: Default::default(),
        memory: None,
        xhigh_models_allowlist: Default::default(),
        reasoning_models_allowlist: Default::default(),
        codex_search: None,
        cache_aware_routing: Default::default(),
        reasoning_compat: Default::default(),
        webhooks: Default::default(),
        opentelemetry: Default::default(),
    }
}

fn copilot_config(copilot_base: &str, fallback_base: &str) -> Config {
    let mut config = base_config();
    config.retry.max_retries_per_provider = 0;
    // The router itself opens no SQLite databases, but isolate the
    // logging/virtual-key paths anyway (repo test pattern) so nothing
    // ever touches ./logs.db / ./keys.db in the workspace.
    common::isolate_databases(&mut config);
    config.providers = vec![
        copilot_provider("copilot", copilot_base.to_string()),
        test_provider("fallback", fallback_base.to_string()),
    ];
    config.model_groups = vec![test_group(vec![
        test_model_named("copilot", "gpt-5-copilot", 1),
        test_model_named("fallback", "upstream-model", 2),
    ])];
    config
}

fn chat_request(stream: bool) -> OpenAIRequest {
    OpenAIRequest {
        model: "test-group".to_string(),
        messages: vec![Message {
            role: "user".to_string(),
            content: serde_json::json!("hello"),
            extra: Default::default(),
        }],
        stream,
        temperature: None,
        max_tokens: None,
        extra: Default::default(),
    }
}

fn completion_response() -> serde_json::Value {
    serde_json::json!({
        "id": "chatcmpl-test",
        "object": "chat.completion",
        "created": 1,
        "model": "upstream-model",
        "choices": [{
            "index": 0,
            "message": {"role": "assistant", "content": "ok"},
            "finish_reason": "stop"
        }],
        "usage": {"prompt_tokens": 1, "completion_tokens": 1, "total_tokens": 2}
    })
}

fn copilot_completion_response() -> serde_json::Value {
    serde_json::json!({
        "id": "chatcmpl-copilot",
        "object": "chat.completion",
        "created": 1,
        "model": "gpt-5-copilot",
        "choices": [{
            "index": 0,
            "message": {"role": "assistant", "content": "ok"},
            "finish_reason": "stop"
        }],
        "usage": {"prompt_tokens": 10, "completion_tokens": 5, "total_tokens": 15}
    })
}

fn test_metrics() -> Arc<Metrics> {
    Arc::new(Metrics::new())
}

fn ledger_handle() -> ai_gateway::active_requests::ActiveRequestHandle {
    ActiveRequestRegistry::new().register(ActiveRequestInfo {
        trace_id: "dispatch-test-trace".to_string(),
        requested_model: "test-group".to_string(),
        model_group: None,
        provider: None,
        model: None,
        attempt: 0,
        phase: ActivePhase::Pending,
        stall_reason: None,
        last_error: None,
        virtual_key_id: None,
        started_at_ms: 0,
        kind: RequestKind::Chat,
        smart_routing_logged: false,
        failed_attempts: Vec::new(),
        attempt_rows_total: 0,
        rate_limit_wait_done: false,
        outcome_logged: false,
        breaker_skips_recorded: false,
        reasoning_relay: None,
    })
}

/// Seed an Authenticated Copilot session whose dispatch `api_base`
/// points at the given mock server.
async fn seeded_manager(tag: &str, api_base: String) -> Arc<CopilotTokenManager> {
    let store = unique_store(tag);
    store
        .save(&CopilotStoredCredentials {
            github_access_token: "gho-test".into(),
            copilot_token: Some("tid=live".into()),
            copilot_expires_at: Some(now_plus(3600)),
            api_base,
            machine_id: "6f9619ff".into(),
            sku: Some("copilot_pro".into()),
            scopes: "read:user".into(),
            updated_at: now_plus(0),
        })
        .unwrap();
    let manager = CopilotTokenManager::with_endpoints(
        store,
        reqwest::Client::new(),
        CopilotEndpoints::production(),
    );
    manager.load_existing_session().await.unwrap();
    Arc::new(manager)
}

/// Copilot success upstream, wiremock'd at `/chat/completions`
/// (no `/v1` — the adapter appends the path directly to the base).
async fn mount_copilot_success(copilot: &MockServer, expect: u64) {
    Mock::given(method("POST"))
        .and(path("/chat/completions"))
        .respond_with(ResponseTemplate::new(200).set_body_json(copilot_completion_response()))
        .expect(expect)
        .mount(copilot)
        .await;
}

fn header_value<'r>(request: &'r wiremock::Request, name: &str) -> Option<&'r str> {
    request.headers.get(name).and_then(|v| v.to_str().ok())
}

/// Synthetic seeded fixtures must never surface verbatim in failure
/// output (no token literals in assertion output).
fn assert_no_token_literals(text: &str) {
    let leaked = text.contains("tid=live")
        || text.contains("tid=stale")
        || text.contains("tid=refreshed")
        || text.contains("gho-test");
    // Deliberately do NOT echo `text` — that would itself leak.
    assert!(!leaked, "token literal leaked in surfaced text");
}

// ─── Req 4.1 / 4.2: mandatory headers + Copilot-token bearer ──────────────

/// Non-streaming dispatch: the upstream receives every mandatory
/// Copilot header (constant `X-Initiator: agent`) and `Authorization:
/// Bearer` carrying the Copilot token from the seeded session.
#[tokio::test]
async fn non_streaming_dispatch_carries_mandatory_headers_and_copilot_bearer() {
    let copilot = MockServer::start().await;
    let fallback = MockServer::start().await;
    mount_copilot_success(&copilot, 1).await;
    // The fallback must NOT serve this request.
    Mock::given(method("POST"))
        .and(path("/v1/chat/completions"))
        .respond_with(ResponseTemplate::new(200).set_body_json(completion_response()))
        .expect(0)
        .mount(&fallback)
        .await;

    let manager = seeded_manager("headers", copilot.uri()).await;
    let mut router = Router::new(
        Arc::new(RwLock::new(copilot_config(&copilot.uri(), &fallback.uri()))),
        test_metrics(),
    );
    router.set_copilot_manager(manager);

    let response = router
        .route_request(&chat_request(false), Some(ledger_handle()), ai_gateway::router::router::ToolCompressionRestore::default())
        .await
        .expect("copilot dispatch succeeds");
    assert_eq!(
        response.extra.get("gateway_provider"),
        Some(&serde_json::json!("copilot"))
    );

    let requests = copilot.received_requests().await.unwrap();
    assert_eq!(requests.len(), 1);
    let upstream = &requests[0];
    // Mandatory header set (Req 4.1) + constant X-Initiator (Req 4.2).
    for (name, expected) in [
        ("copilot-integration-id", "vscode-chat"),
        ("editor-version", "vscode/1.95.0"),
        ("editor-plugin-version", "copilot-chat/0.23.5"),
        ("user-agent", "GitHubCopilotChat/0.23.5"),
        ("openai-intent", "conversation-edits"),
        ("x-github-api-version", "2024-08-01"),
        ("x-initiator", "agent"),
    ] {
        let value = header_value(upstream, name)
            .unwrap_or_else(|| panic!("missing mandatory header: {name}"));
        assert_eq!(value, expected, "header {name}");
    }
    // Bearer = the seeded Copilot token (assert shape; the no-leak
    // guard separately verifies no literal surfaces).
    let is_copilot_bearer = header_value(upstream, "authorization")
        .is_some_and(|v| v.starts_with("Bearer tid="));
    assert!(is_copilot_bearer, "not a copilot-token bearer");
    // Dispatched at the no-/v1 Copilot path.
    let upstream_url = upstream.url.as_str();
    assert!(upstream_url.contains("/chat/completions"));
    assert!(!upstream_url.contains("/v1/chat"));
    copilot.verify().await;
}

// ─── Req 4.5: streaming rides the buffered path, stream:false upstream ────

/// A streaming client request to a copilot-routed group flows through
/// the buffered-dispatch path (copilot is a must-buffer provider) and
/// returns a buffered `OpenAIResponse`. The upstream request is sent
/// with `stream: false` (Bedrock-arm precedent: the copilot arm has no
/// upstream-SSE reassembly, so it never asks the provider to stream) —
/// the client's stream:true never reaches the Copilot API.
#[tokio::test]
async fn streaming_request_rides_buffered_dispatch_and_reassembles() {
    let copilot = MockServer::start().await;
    mount_copilot_success(&copilot, 1).await;

    let manager = seeded_manager("stream", copilot.uri()).await;
    let mut router = Router::new(
        Arc::new(RwLock::new(copilot_config(&copilot.uri(), "http://fallback.invalid"))),
        test_metrics(),
    );
    router.set_copilot_manager(manager);

    let streamed = router
        .route_request_streaming(&chat_request(true), Some(ledger_handle()), ai_gateway::router::router::ToolCompressionRestore::default())
        .await
        .expect("streaming copilot request succeeds");

    match streamed {
        StreamingResponse::Buffered(response) => {
            // Buffered exactly like a non-streaming dispatch: full
            // content + usage, correct provider/model attribution.
            assert_eq!(
                response.choices[0].message.content_as_text(),
                "ok",
                "buffered content mismatch"
            );
            assert_eq!(response.usage.total_tokens, 15);
            assert_eq!(
                response.extra.get("gateway_provider"),
                Some(&serde_json::json!("copilot"))
            );
            assert_eq!(
                response.extra.get("gateway_responded_model"),
                Some(&serde_json::json!("gpt-5-copilot"))
            );
        }
        StreamingResponse::PassThrough { .. } => {
            panic!("copilot must never take the pass-through streaming path");
        }
    }
    // Same upstream surface as the non-streaming dispatch: one POST to
    // the copilot /chat/completions endpoint with the mandatory headers,
    // and the upstream body carries stream:false — never the client's
    // stream:true (the adapter would choke on the SSE body).
    let requests = copilot.received_requests().await.unwrap();
    assert_eq!(requests.len(), 1);
    assert_eq!(header_value(&requests[0], "x-initiator"), Some("agent"));
    let upstream_body: serde_json::Value =
        serde_json::from_slice(&requests[0].body).expect("upstream body is JSON");
    assert_eq!(
        upstream_body.get("stream"),
        Some(&serde_json::json!(false)),
        "copilot upstream must be dispatched stream:false"
    );
    copilot.verify().await;
}

// ─── Req 4.3: 401 → one refresh + one re-dispatch; exhausted → failover ──

/// First upstream response 401, second 200: the cached-but-revoked
/// bearer is still unexpired, so dispatch-time resolution serves it
/// from cache (no exchange); the 401 rescue calls force_refresh() —
/// exactly ONE wiremock exchange minting a fresh bearer — and the
/// single re-dispatch carries the refreshed bearer and succeeds
/// (dispatch count = 2, exchange count = 1).
#[tokio::test]
async fn upstream_401_refreshes_once_and_succeeds_on_second_dispatch() {
    let copilot = MockServer::start().await;
    let exchange = MockServer::start().await;

    // Dispatch: first request (cached-but-revoked bearer) 401 — capped
    // at one hit so the re-dispatch falls through to the success mock.
    Mock::given(method("POST"))
        .and(path("/chat/completions"))
        .respond_with(ResponseTemplate::new(401).set_body_string("token revoked"))
        .up_to_n_times(1)
        .expect(1)
        .mount(&copilot)
        .await;
    Mock::given(method("POST"))
        .and(path("/chat/completions"))
        .respond_with(ResponseTemplate::new(200).set_body_json(copilot_completion_response()))
        .expect(1)
        .mount(&copilot)
        .await;
    // Forced-refresh exchange: GET with the GitHub token → fresh
    // copilot token whose endpoints.api points back at the dispatch
    // mock. expect(1): the seeded cache is unexpired, so ONLY the
    // rescue's force_refresh() exchanges.
    Mock::given(method("GET"))
        .and(path("/copilot_internal/v2/token"))
        .respond_with(ResponseTemplate::new(200).set_body_json(serde_json::json!({
            "token": "tid=refreshed",
            "expires_at": now_plus(3600),
            "refresh_in": 1500u64,
            "endpoints": { "api": copilot.uri() },
            "sku": "copilot_pro",
        })))
        .expect(1)
        .mount(&exchange)
        .await;

    // Seed Authenticated with a live-but-revoked Copilot token: the
    // cache serves it at dispatch time (no exchange), the upstream
    // rejects it 401, and only force_refresh() can replace it.
    let store = unique_store("401ok");
    store
        .save(&CopilotStoredCredentials {
            github_access_token: "gho-test".into(),
            copilot_token: Some("tid=revoked-unexpired".into()),
            copilot_expires_at: Some(now_plus(3600)),
            api_base: copilot.uri(),
            machine_id: "6f9619ff".into(),
            sku: Some("copilot_pro".into()),
            scopes: "read:user".into(),
            updated_at: now_plus(0),
        })
        .unwrap();
    let manager = CopilotTokenManager::with_endpoints(
        store,
        reqwest::Client::new(),
        CopilotEndpoints {
            device_flow: DeviceFlowEndpoints::production(),
            copilot_token_url: format!("{}/copilot_internal/v2/token", exchange.uri()),
        },
    );
    manager.load_existing_session().await.unwrap();
    assert!(matches!(
        manager.session_state().await,
        ai_gateway::oauth::github_copilot::manager::CopilotSessionState::Authenticated { .. }
    ));

    let mut router = Router::new(
        Arc::new(RwLock::new(copilot_config(&copilot.uri(), "http://fallback.invalid"))),
        test_metrics(),
    );
    router.set_copilot_manager(Arc::new(manager));

    let response = router
        .route_request(&chat_request(false), Some(ledger_handle()), ai_gateway::router::router::ToolCompressionRestore::default())
        .await
        .expect("auth rescue succeeds on the re-dispatch");
    assert_eq!(
        response.extra.get("gateway_provider"),
        Some(&serde_json::json!("copilot"))
    );
    // Exactly 2 dispatches (first 401 + one re-dispatch) and exactly
    // one forced-refresh exchange.
    copilot.verify().await;
    exchange.verify().await;
    assert_eq!(copilot.received_requests().await.unwrap().len(), 2);
    assert_eq!(exchange.received_requests().await.unwrap().len(), 1);
    // The re-dispatch carried the refreshed bearer, not the cached one.
    let requests = copilot.received_requests().await.unwrap();
    assert!(
        header_value(&requests[1], "authorization")
            .is_some_and(|v| v.starts_with("Bearer tid=refreshed")),
        "re-dispatch did not use the refreshed token"
    );
}

/// Both the copilot dispatch and the rescue's forced refresh are
/// rejected 401 → the session flips Expired, the single re-dispatch
/// fails in bearer resolution (no second upstream POST), the provider
/// is abandoned, and the fallback in the same model group serves the
/// request.
#[tokio::test]
async fn upstream_401_exhausted_fails_over_to_fallback() {
    let copilot = MockServer::start().await;
    let exchange = MockServer::start().await;
    let fallback = MockServer::start().await;

    Mock::given(method("POST"))
        .and(path("/chat/completions"))
        .respond_with(ResponseTemplate::new(401).set_body_string("token revoked"))
        // First dispatch only: the rescue's forced exchange rejects
        // 401 → session Expired → the single re-dispatch short-circuits
        // in bearer resolution (no second upstream POST).
        .expect(1)
        .mount(&copilot)
        .await;
    // Forced-refresh exchange rejects with 401 (no Copilot seat) —
    // the session flips to Expired and the re-dispatch fails without
    // another exchange.
    Mock::given(method("GET"))
        .and(path("/copilot_internal/v2/token"))
        .respond_with(ResponseTemplate::new(401).set_body_string("no copilot seat"))
        .expect(1)
        .mount(&exchange)
        .await;
    Mock::given(method("POST"))
        .and(path("/v1/chat/completions"))
        .respond_with(ResponseTemplate::new(200).set_body_json(completion_response()))
        .expect(1)
        .mount(&fallback)
        .await;

    let store = unique_store("401ex");
    store
        .save(&CopilotStoredCredentials {
            github_access_token: "gho-test".into(),
            copilot_token: Some("tid=revoked".into()),
            copilot_expires_at: Some(now_plus(3600)),
            api_base: copilot.uri(),
            machine_id: "6f9619ff".into(),
            sku: Some("copilot_pro".into()),
            scopes: "read:user".into(),
            updated_at: now_plus(0),
        })
        .unwrap();
    let manager = CopilotTokenManager::with_endpoints(
        store,
        reqwest::Client::new(),
        CopilotEndpoints {
            device_flow: DeviceFlowEndpoints::production(),
            copilot_token_url: format!("{}/copilot_internal/v2/token", exchange.uri()),
        },
    );
    manager.load_existing_session().await.unwrap();

    let mut router = Router::new(
        Arc::new(RwLock::new(copilot_config(&copilot.uri(), &fallback.uri()))),
        test_metrics(),
    );
    router.set_copilot_manager(Arc::new(manager));

    let response = router
        .route_request(&chat_request(false), Some(ledger_handle()), ai_gateway::router::router::ToolCompressionRestore::default())
        .await
        .expect("fails over after the single auth rescue");
    assert_eq!(
        response.extra.get("gateway_provider"),
        Some(&serde_json::json!("fallback"))
    );
    copilot.verify().await;
    exchange.verify().await;
    fallback.verify().await;
}

// ─── Req 4.4: 429 failover + x-ratelimit usage tracking ───────────────────

/// A 429 carrying `x-ratelimit-*` headers fails over to the fallback
/// (existing failover path) and the headers are parsed into the copilot
/// usage tracker (snapshot asserts the parsed windows).
#[tokio::test]
async fn rate_limited_429_fails_over_and_feeds_usage_tracker() {
    let copilot = MockServer::start().await;
    let fallback = MockServer::start().await;

    Mock::given(method("POST"))
        .and(path("/chat/completions"))
        .respond_with(
            ResponseTemplate::new(429)
                .insert_header("x-ratelimit-limit-requests", "60")
                .insert_header("x-ratelimit-remaining-requests", "0")
                .insert_header("x-ratelimit-reset-requests", "1m")
                .insert_header("x-ratelimit-limit-tokens", "150000")
                .insert_header("x-ratelimit-remaining-tokens", "0")
                .insert_header("x-ratelimit-reset-tokens", "1m")
                .insert_header("x-copilot-premium-requests-used", "42")
                .insert_header("x-copilot-premium-requests-limit", "100")
                .insert_header("x-copilot-premium-requests-reset", "1h")
                .set_body_string("rate limited"),
        )
        .expect(1)
        .mount(&copilot)
        .await;
    Mock::given(method("POST"))
        .and(path("/v1/chat/completions"))
        .respond_with(ResponseTemplate::new(200).set_body_json(completion_response()))
        .expect(1)
        .mount(&fallback)
        .await;

    let manager = seeded_manager("429", copilot.uri()).await;
    let tracker = Arc::new(UsageTracker::new());
    let mut router = Router::new(
        Arc::new(RwLock::new(copilot_config(&copilot.uri(), &fallback.uri()))),
        test_metrics(),
    );
    router.set_copilot_manager(manager);
    router.set_copilot_usage_tracker(tracker.clone());

    let response = router
        .route_request(&chat_request(false), Some(ledger_handle()), ai_gateway::router::router::ToolCompressionRestore::default())
        .await
        .expect("429 fails over to the fallback provider");
    assert_eq!(
        response.extra.get("gateway_provider"),
        Some(&serde_json::json!("fallback"))
    );
    copilot.verify().await;
    fallback.verify().await;

    // The copilot usage tracker parsed the 429's x-ratelimit headers.
    let snapshot = tracker.snapshot().await;
    let short = &snapshot.short;
    assert_eq!(short.requests.limit, Some(60));
    assert_eq!(short.requests.remaining, Some(0));
    assert_eq!(short.tokens.limit, Some(150000));
    assert_eq!(short.tokens.remaining, Some(0));
    assert!(short.requests.resets_at.is_some());
    assert!(short.tokens.resets_at.is_some());
    // The premium percentage window (x-copilot-premium-requests-*) is
    // populated alongside the count windows: used=42 of limit=100 →
    // remaining 58, percentage semantics.
    let premium = snapshot.premium.expect("premium window recorded");
    assert!(premium.is_percentage);
    assert_eq!(premium.limit, Some(100));
    assert_eq!(premium.remaining, Some(58));
    assert!(premium.resets_at.is_some());
}

// ─── Req 9.2: unauthenticated candidate skipped, fallback serves ──────────

/// An unauthenticated copilot candidate in the model group is skipped
/// BEFORE dispatch: no copilot upstream error attempt is recorded, the
/// router writes a `copilot_auth_skip` ledger row, and the request
/// still succeeds via the other provider.
#[tokio::test]
async fn unauthenticated_session_skips_with_ledger_row_and_fallback_serves() {
    let copilot = MockServer::start().await;
    let fallback = MockServer::start().await;
    Mock::given(method("POST"))
        .and(path("/v1/chat/completions"))
        .respond_with(ResponseTemplate::new(200).set_body_json(completion_response()))
        .expect(1)
        .mount(&fallback)
        .await;

    // Empty store → Unauthenticated session.
    let manager = CopilotTokenManager::with_endpoints(
        unique_store("unauth"),
        reqwest::Client::new(),
        CopilotEndpoints::production(),
    );

    let mut router = Router::new(
        Arc::new(RwLock::new(copilot_config(&copilot.uri(), &fallback.uri()))),
        test_metrics(),
    );
    router.set_copilot_manager(Arc::new(manager));

    let handle = ledger_handle();
    let response = router
        .route_request(&chat_request(false), Some(handle.clone()), ai_gateway::router::router::ToolCompressionRestore::default())
        .await
        .expect("fails over to the fallback provider");
    assert_eq!(
        response.extra.get("gateway_provider"),
        Some(&serde_json::json!("fallback"))
    );

    // The skip is observable: copilot_auth_skip ledger row, and NO
    // copilot upstream error attempt.
    let ledger = handle.take_failed_attempts();
    let skip = ledger
        .iter()
        .find(|a| a.error_class.as_deref() == Some("copilot_auth_skip"))
        .expect("copilot_auth_skip ledger row");
    assert_eq!(skip.provider, "copilot");
    assert_eq!(skip.status_code, Some(401));
    assert!(
        !ledger.iter().any(|a| a.provider == "copilot"
            && a.error_class.as_deref() != Some("copilot_auth_skip")),
        "ledger: {ledger:?}"
    );
    // Nothing was sent to the copilot upstream.
    assert_eq!(copilot.received_requests().await.unwrap().len(), 0);
    fallback.verify().await;
}

// ─── Req 9.3: zero-cost usage rows with attribution ────────────────────────

/// After a successful copilot dispatch the recorded usage carries the
/// full token counts and model/provider attribution, while the cost is
/// zero even with non-zero rates configured on the model (billing-grade
/// surface: `gateway_cost` extra + usage fields on the response).
#[tokio::test]
async fn successful_dispatch_records_zero_cost_usage_with_attribution() {
    let copilot = MockServer::start().await;
    mount_copilot_success(&copilot, 1).await;

    let manager = seeded_manager("cost", copilot.uri()).await;
    let mut config = copilot_config(&copilot.uri(), "http://fallback.invalid");
    // Rates configured on the copilot model — must still bill 0.
    config.model_groups[0].models[0].cost_per_million_input_tokens = 10.0;
    config.model_groups[0].models[0].cost_per_million_output_tokens = 30.0;

    let mut router = Router::new(Arc::new(RwLock::new(config)), test_metrics());
    router.set_copilot_manager(manager);

let handle = ledger_handle();
let response = router
.route_request(&chat_request(false), Some(handle.clone()), ai_gateway::router::router::ToolCompressionRestore::default())
.await
.expect("copilot dispatch succeeds");
    // Usage preserved…
    assert_eq!(response.usage.prompt_tokens, 10);
    assert_eq!(response.usage.completion_tokens, 5);
    assert_eq!(response.usage.total_tokens, 15);
    // …attribution preserved…
    assert_eq!(
        response.extra.get("gateway_provider"),
        Some(&serde_json::json!("copilot"))
    );
    assert_eq!(
        response.extra.get("gateway_responded_model"),
        Some(&serde_json::json!("gpt-5-copilot"))
    );
    // …with zero cost reported (Req 9.3).
    assert_eq!(
        response.extra.get("gateway_cost"),
        Some(&serde_json::json!(0.0))
    );
    // Clean dispatch: no failed attempts in the ledger.
    assert!(handle.take_failed_attempts().is_empty());
    copilot.verify().await;
}

// ─── Req 8.1: PAT bearer + missing-PAT actionable error ───────────────────

/// `auth_method: api-key` with a resolved PAT: dispatch carries the
/// PAT-derived bearer and the full mandatory header set.
#[tokio::test]
async fn pat_auth_dispatch_carries_pat_bearer() {
    let copilot = MockServer::start().await;
    mount_copilot_success(&copilot, 1).await;

    let mut config = copilot_config(&copilot.uri(), "http://fallback.invalid");
    let copilot_cfg = config
        .providers
        .iter_mut()
        .find(|p| p.name == "copilot")
        .unwrap();
    copilot_cfg.auth_method = Some("api-key".to_string());
    // Literal PAT resolved through the existing api_key mechanism
    // (resolve_api_key(): resolved_api_key wins; a plaintext literal in
    // api_key_env lands there at config load).
    copilot_cfg.api_key_env = Some("ghp_test_pat".to_string());
    copilot_cfg.resolved_api_key = Some("ghp_test_pat".to_string());

let mut router = Router::new(Arc::new(RwLock::new(config)), test_metrics());
router.set_copilot_manager(seeded_manager("patnoop", "http://oauth.invalid".into()).await);
// The authenticated oauth session exists but PAT mode must never use
// it for the bearer (Req 8.3) — asserted via the ghp_ bearer below.

    let response = router
        .route_request(&chat_request(false), Some(ledger_handle()), ai_gateway::router::router::ToolCompressionRestore::default())
        .await
        .expect("PAT dispatch succeeds");
    assert_eq!(
        response.extra.get("gateway_provider"),
        Some(&serde_json::json!("copilot"))
    );

    let requests = copilot.received_requests().await.unwrap();
    assert_eq!(requests.len(), 1);
    let bearer = header_value(&requests[0], "authorization").expect("authorization header");
    // PAT-derived bearer — assert shape, never the literal.
    assert!(bearer.starts_with("Bearer ghp_"), "not the PAT bearer");
    // Mandatory set rides along identically.
    assert_eq!(header_value(&requests[0], "x-initiator"), Some("agent"));
    assert_eq!(
        header_value(&requests[0], "copilot-integration-id"),
        Some("vscode-chat")
    );
    assert_no_token_literals(bearer);
    copilot.verify().await;
}

/// `auth_method: api-key` with NO resolvable PAT: the copilot provider
/// is the ONLY candidate in the group, so the request fails with an
/// actionable error naming the remedy — and never falls back to OAuth
/// copilot tokens (an authenticated session is attached, but the
/// nothing was dispatched with it).
#[tokio::test]
async fn pat_auth_missing_pat_is_actionable_error_without_oauth_fallback() {
    let copilot = MockServer::start().await;

    // Single-candidate group: the copilot failure IS the request
    // failure (no second provider to mask the actionable error).
    let mut config = copilot_config(&copilot.uri(), "http://fallback.invalid");
    config.providers.truncate(1); // copilot only
    config.model_groups[0].models.truncate(1);
    let copilot_cfg = config
        .providers
        .iter_mut()
        .find(|p| p.name == "copilot")
        .unwrap();
    copilot_cfg.auth_method = Some("api-key".to_string());
    // No api_key / api_key_env at all.
    copilot_cfg.api_key_env = None;
    copilot_cfg.resolved_api_key = None;

    // An AUTHENTICATED oauth session is attached — PAT mode must
    // still refuse to use it (Req 8.3 no-fallback).
    let manager = seeded_manager("patmissing", "http://oauth.invalid".into()).await;

    let mut router = Router::new(Arc::new(RwLock::new(config)), test_metrics());
    router.set_copilot_manager(manager);

    let handle = ledger_handle();
    let error = router
        .route_request(&chat_request(false), Some(handle), ai_gateway::router::router::ToolCompressionRestore::default())
        .await
        .expect_err("missing PAT must fail");

    // The exhaustion error carries the actionable provider message.
    let attempts = match &error {
        ai_gateway::error::GatewayError::AllProvidersFailed(agg) => agg.attempts.clone(),
        other => panic!("expected AllProvidersFailed, got {other:?}"),
    };
    let attempt = attempts
        .iter()
        .find(|a| a.provider == "copilot")
        .expect("copilot attempt in aggregated error");
    assert!(
        attempt.error.contains("api-key")
            && attempt.error.contains("no api_key/api_key_env"),
        "not actionable for api-key remedy: {}",
        attempt.error
    );
    assert_eq!(attempt.status_code, Some(401));
    // No token material surfaced.
    assert_no_token_literals(&attempt.error);
    // Nothing dispatched anywhere: no OAuth-token dispatch either.
    assert_eq!(copilot.received_requests().await.unwrap().len(), 0);
}

/// `auth_method: api-key` with ONLY `api_key_env` naming a real env var
/// (no pre-resolved literal): the dispatch-time resolution reads the env
/// var (env-name-first, `resolve_api_key`), proving the env path
/// end-to-end through a real dispatch.
#[tokio::test]
async fn pat_auth_env_var_name_resolves_through_real_dispatch() {
    let copilot = MockServer::start().await;
    mount_copilot_success(&copilot, 1).await;

    // Unique env var name so parallel tests can never collide; removed
    // before returning so the process env stays clean (an assert failure
    // aborts only this test's task, not the process).
    let env_var = "OBEY_TEST_COPILOT_PAT_DISPATCH_7F3A";
    std::env::set_var(env_var, "ghp_env_resolved_pat");

    let mut config = copilot_config(&copilot.uri(), "http://fallback.invalid");
    let copilot_cfg = config
        .providers
        .iter_mut()
        .find(|p| p.name == "copilot")
        .unwrap();
    copilot_cfg.auth_method = Some("api-key".to_string());
    // ONLY the env var name — no resolved_api_key literal.
    copilot_cfg.api_key_env = Some(env_var.to_string());
    copilot_cfg.resolved_api_key = None;

    let mut router = Router::new(Arc::new(RwLock::new(config)), test_metrics());
    router.set_copilot_manager(seeded_manager("patenv", "http://oauth.invalid".into()).await);

    let response = router
        .route_request(&chat_request(false), Some(ledger_handle()), ai_gateway::router::router::ToolCompressionRestore::default())
        .await
        .expect("PAT dispatch succeeds via env var");
    assert_eq!(
        response.extra.get("gateway_provider"),
        Some(&serde_json::json!("copilot"))
    );

    let requests = copilot.received_requests().await.unwrap();
    assert_eq!(requests.len(), 1);
    let bearer = header_value(&requests[0], "authorization").expect("authorization header");
    assert!(
        bearer.starts_with("Bearer ghp_"),
        "bearer must come from the env-var-resolved PAT: {bearer:?}"
    );
    assert_no_token_literals(bearer);
    copilot.verify().await;
    std::env::remove_var(env_var);
}

