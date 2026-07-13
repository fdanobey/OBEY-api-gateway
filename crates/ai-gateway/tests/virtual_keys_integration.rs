//! Integration tests for the virtual key management full request flow (task 13.2).
//!
//! These tests drive real requests through the gateway's Axum router via
//! `tower::ServiceExt::oneshot()` (no port binding), exercising the
//! `virtual_key_auth_middleware` enforcement pipeline end to end: authentication,
//! model access, budget, rate limiting, usage recording, cache invalidation, and
//! the three enforcement modes (disabled / optional / required).
//!
//! Each test uses its own temp SQLite key store (a `tempdir`) and a wiremock
//! upstream provider that returns an OpenAI-shaped body including a `usage`
//! object so post-response usage recording fires.
//!
//! Requirements traceability: 2.1, 2.4, 3.2, 5.2, 6.2, 7.5, 11.1, 11.2, 11.4, 11.5.

use axum::body::Body;
use axum::http::{HeaderMap, Request, StatusCode};
use chrono::{Duration as ChronoDuration, Utc};
use std::time::Duration;
use tower::ServiceExt;
use wiremock::matchers::{method, path};
use wiremock::{Mock, MockServer, ResponseTemplate};

use ai_gateway::config::*;
use ai_gateway::gateway::GatewayServer;
use ai_gateway::virtual_keys::models::{CreateKeyParams, UsageQueryParams, UsageRecord};
use ai_gateway::virtual_keys::VirtualKeyManager;

// ---------------------------------------------------------------------------
// Harness helpers
// ---------------------------------------------------------------------------

/// The model id configured in the test model group (matches the request body
/// `model` field). The mock provider echoes this id back.
const TEST_MODEL: &str = "gpt-4";

/// Start a mock OpenAI-compatible provider that returns a static chat
/// completion including a `usage` object (so usage recording fires).
async fn start_mock_provider() -> MockServer {
    let server = MockServer::start().await;
    let body = serde_json::json!({
        "id": "chatcmpl-vk-mock",
        "object": "chat.completion",
        "created": 1_700_000_000_i64,
        "model": TEST_MODEL,
        "choices": [{
            "index": 0,
            "message": { "role": "assistant", "content": "Hello from mock provider!" },
            "finish_reason": "stop"
        }],
        "usage": { "prompt_tokens": 10, "completion_tokens": 8, "total_tokens": 18 }
    });
    Mock::given(method("POST"))
        .and(path("/v1/chat/completions"))
        .respond_with(ResponseTemplate::new(200).set_body_json(body))
        .mount(&server)
        .await;
    server
}

/// Build a Config with a single provider pointed at `mock_uri`, one model group
/// (`test-group` exposing model `gpt-4`), the given enforcement mode, and a
/// virtual-key store at `db_path`.
fn test_config(mock_uri: &str, enforcement: EnforcementMode, db_path: String) -> Config {
    Config {
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
        providers: vec![Provider {
            name: "test-provider".to_string(),
            provider_type: "openai".to_string(),
            base_url: Some(mock_uri.to_string()),
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
            max_connections: 10,
            rate_limit_per_minute: 0,
            custom_headers: Default::default(),
            connection_pool: ProviderConnectionPoolConfig::default(),
            budget: None,
            manual_models: vec![],
            global_inference_profile: false,
            cross_region_inference: false,
            custom_vpc_endpoint: false,
            prompt_caching: false,
            reasoning: true,
            codex_base_url_override: None,
            codex_model_override: None,
            instructions_override: None,
            max_rate_limit_cooldown_seconds: None,
        }],
        model_groups: vec![ModelGroup {
            name: "test-group".to_string(),
            version_fallback_enabled: false,
            models: vec![ProviderModel {
                provider: "test-provider".to_string(),
                model: TEST_MODEL.to_string(),
                cost_per_million_input_tokens: 30.0,
                cost_per_million_output_tokens: 60.0,
                priority: 100,
            }],
        }],
        circuit_breaker: CircuitBreakerConfig::default(),
        retry: RetryConfig::default(),
        logging: LoggingConfig::default(),
        semantic_cache: None,
        exact_cache: ExactCacheConfig::default(),
        prometheus: None,
        context: ai_gateway::config::ContextConfig::default(),
        first_launch_completed: false,
        tray: ai_gateway::config::TrayConfig::default(),
        codex_instructions_url: None,
        streaming: None,
        virtual_keys: VirtualKeysConfig {
            enforcement,
            database_path: db_path,
        },
        guardrails: None,
    }
}

/// A test server plus the temp resources that must outlive it.
struct TestServer {
    server: GatewayServer,
    _mock: MockServer,
    _tmp: tempfile::TempDir,
}

impl TestServer {
    async fn new(enforcement: EnforcementMode) -> Self {
        let mock = start_mock_provider().await;
        let tmp = tempfile::tempdir().unwrap();
        let db_path = tmp.path().join("keys.db").to_string_lossy().into_owned();
        let config = test_config(&mock.uri(), enforcement, db_path);
        let server = GatewayServer::new(config, None).await.unwrap();
        Self {
            server,
            _mock: mock,
            _tmp: tmp,
        }
    }

    fn manager(&self) -> &VirtualKeyManager {
        &self.server.state.virtual_key_manager
    }

    fn router(&self) -> axum::Router {
        self.server.build_router()
    }
}

/// Default (all-unlimited) create params.
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
    }
}

/// Send a `POST /v1/chat/completions` through the router, optionally with a
/// Bearer token and a chosen model. Returns (status, headers, body bytes).
async fn chat_request(
    app: axum::Router,
    bearer: Option<&str>,
    model: &str,
) -> (StatusCode, HeaderMap, Vec<u8>) {
    let mut builder = Request::post("/v1/chat/completions").header("content-type", "application/json");
    if let Some(token) = bearer {
        builder = builder.header("authorization", format!("Bearer {token}"));
    }
    let body = serde_json::json!({
        "model": model,
        "messages": [{"role": "user", "content": "hello"}],
        "stream": false
    });
    let req = builder
        .body(Body::from(serde_json::to_string(&body).unwrap()))
        .unwrap();
    let resp = app.oneshot(req).await.unwrap();
    let status = resp.status();
    let headers = resp.headers().clone();
    let bytes = axum::body::to_bytes(resp.into_body(), 1024 * 1024).await.unwrap();
    (status, headers, bytes.to_vec())
}

/// A wide-open usage query window covering all recorded timestamps.
fn full_window() -> UsageQueryParams {
    UsageQueryParams {
        start: chrono::DateTime::<Utc>::from_timestamp(0, 0).unwrap(),
        end: Utc::now() + ChronoDuration::days(1),
    }
}

/// Poll `query_usage` until at least `min_requests` are recorded (usage
/// recording runs off the response path via a spawned task) or time out.
async fn wait_for_request_count(mgr: &VirtualKeyManager, key_id: &str, min_requests: u64) -> u64 {
    for _ in 0..100 {
        let agg = mgr.query_usage(key_id, full_window()).await.unwrap();
        if agg.total_requests >= min_requests {
            return agg.total_requests;
        }
        tokio::time::sleep(Duration::from_millis(20)).await;
    }
    let agg = mgr.query_usage(key_id, full_window()).await.unwrap();
    panic!(
        "timed out waiting for >= {min_requests} usage records, saw {}",
        agg.total_requests
    );
}

// ---------------------------------------------------------------------------
// enforcement = required
// ---------------------------------------------------------------------------

/// Req 2.1, 11.2: under `required`, a request bearing a valid virtual key
/// authenticates, routes to the provider (2xx), and its usage is recorded.
#[tokio::test]
async fn required_valid_key_authenticates_routes_and_records_usage() {
    let ts = TestServer::new(EnforcementMode::Required).await;
    let created = ts.manager().create_key(create_defaults()).await.unwrap();

    let (status, _headers, body) =
        chat_request(ts.router(), Some(&created.key), TEST_MODEL).await;

    assert!(
        status.is_success(),
        "expected 2xx for a valid key, got {status}: {}",
        String::from_utf8_lossy(&body)
    );

    // Usage is recorded off the response path; poll until it lands.
    let count = wait_for_request_count(ts.manager(), &created.id, 1).await;
    assert!(count >= 1, "usage request count should advance");
}

/// Req 11.2: under `required`, a request without any virtual key is rejected
/// with 401.
#[tokio::test]
async fn required_without_key_is_rejected_401() {
    let ts = TestServer::new(EnforcementMode::Required).await;

    let (status, _headers, _body) = chat_request(ts.router(), None, TEST_MODEL).await;

    assert_eq!(status, StatusCode::UNAUTHORIZED);
}

// ---------------------------------------------------------------------------
// enforcement = disabled  (Req 11.1, 11.5, 2.4)
// ---------------------------------------------------------------------------

/// Req 11.1, 2.4: under `disabled`, a request with no virtual key passes
/// through using provider keys directly (2xx), no auth required.
#[tokio::test]
async fn disabled_without_key_passes_through() {
    let ts = TestServer::new(EnforcementMode::Disabled).await;

    let (status, _headers, body) = chat_request(ts.router(), None, TEST_MODEL).await;

    assert!(
        status.is_success(),
        "expected pass-through 2xx under disabled, got {status}: {}",
        String::from_utf8_lossy(&body)
    );
}

/// Req 11.5: under `disabled`, a presented `vk_` key is ignored — the request
/// is routed without validation and no usage is tracked for that key.
#[tokio::test]
async fn disabled_with_key_is_ignored_and_not_tracked() {
    let ts = TestServer::new(EnforcementMode::Disabled).await;
    let created = ts.manager().create_key(create_defaults()).await.unwrap();

    let (status, _headers, body) =
        chat_request(ts.router(), Some(&created.key), TEST_MODEL).await;
    assert!(
        status.is_success(),
        "expected 2xx under disabled with key ignored, got {status}: {}",
        String::from_utf8_lossy(&body)
    );

    // Give any (erroneous) spawned recording a chance to run, then assert none.
    tokio::time::sleep(Duration::from_millis(200)).await;
    let agg = ts.manager().query_usage(&created.id, full_window()).await.unwrap();
    assert_eq!(agg.total_requests, 0, "disabled mode must not track vk_ usage");
    assert_eq!(agg.total_input_tokens, 0);
    assert_eq!(agg.total_output_tokens, 0);
}

// ---------------------------------------------------------------------------
// enforcement = optional  (Req 11.4, 2.4)
// ---------------------------------------------------------------------------

/// Req 11.4: under `optional`, a valid `vk_` key is validated and its usage is
/// tracked.
#[tokio::test]
async fn optional_with_valid_key_is_validated_and_tracked() {
    let ts = TestServer::new(EnforcementMode::Optional).await;
    let created = ts.manager().create_key(create_defaults()).await.unwrap();

    let (status, _headers, body) =
        chat_request(ts.router(), Some(&created.key), TEST_MODEL).await;
    assert!(
        status.is_success(),
        "expected 2xx for a valid key under optional, got {status}: {}",
        String::from_utf8_lossy(&body)
    );

    let count = wait_for_request_count(ts.manager(), &created.id, 1).await;
    assert!(count >= 1, "optional mode must track usage for presented keys");
}

/// Req 11.4, 2.4: under `optional`, a request without a key passes through
/// using provider keys directly (2xx).
#[tokio::test]
async fn optional_without_key_passes_through() {
    let ts = TestServer::new(EnforcementMode::Optional).await;

    let (status, _headers, body) = chat_request(ts.router(), None, TEST_MODEL).await;

    assert!(
        status.is_success(),
        "expected pass-through 2xx under optional without key, got {status}: {}",
        String::from_utf8_lossy(&body)
    );
}

/// Req 2.2: under `optional`, presenting an unrecognized `vk_` key is rejected
/// with 401 (it is validated, not passed through).
#[tokio::test]
async fn optional_with_invalid_key_is_rejected_401() {
    let ts = TestServer::new(EnforcementMode::Optional).await;

    let (status, _headers, _body) =
        chat_request(ts.router(), Some("vk_this_key_does_not_exist"), TEST_MODEL).await;

    assert_eq!(status, StatusCode::UNAUTHORIZED);
}

// ---------------------------------------------------------------------------
// Cache invalidation on mutation  (Req 7.5, 8.3)
// ---------------------------------------------------------------------------

/// Req 7.5 / 8.3: authenticating populates the cache; revoking invalidates it,
/// so a subsequent request with the same key is rejected with 403 (revoked)
/// within the same test — no stale cache hit.
#[tokio::test]
async fn revoke_invalidates_cache_and_rejects_next_request() {
    let ts = TestServer::new(EnforcementMode::Required).await;
    let created = ts.manager().create_key(create_defaults()).await.unwrap();

    // First request succeeds and populates the auth cache.
    let (status, _h, body) = chat_request(ts.router(), Some(&created.key), TEST_MODEL).await;
    assert!(
        status.is_success(),
        "first request should succeed, got {status}: {}",
        String::from_utf8_lossy(&body)
    );

    // Revoke the key (invalidates the cache entry + rate limiter).
    ts.manager().revoke_key(&created.id).await.unwrap();

    // Next request with the same key must be rejected (403 revoked).
    let (status, _h, _body) = chat_request(ts.router(), Some(&created.key), TEST_MODEL).await;
    assert_eq!(
        status,
        StatusCode::FORBIDDEN,
        "revoked key must be rejected with 403 after cache invalidation"
    );
}

// ---------------------------------------------------------------------------
// Budget exhaustion  (Req 3.2)
// ---------------------------------------------------------------------------

/// Req 3.2: once cumulative spend meets/exceeds the budget, subsequent requests
/// are rejected with 429 before forwarding.
///
/// Spend is driven deterministically via `record_usage` (the shared oneshot
/// harness makes real cost accrual across requests timing-dependent, since the
/// authentication cache snapshots counters at lookup time). After recording
/// past-budget spend we invalidate the cache so the next authentication loads
/// the exhausted counters from the store.
#[tokio::test]
async fn budget_exhaustion_blocks_subsequent_requests_429() {
    let ts = TestServer::new(EnforcementMode::Required).await;
    let created = ts
        .manager()
        .create_key(CreateKeyParams {
            budget_limit_usd: Some(0.01),
            ..create_defaults()
        })
        .await
        .unwrap();

    // Drive spend well past the 0.01 USD budget.
    ts.manager()
        .record_usage(UsageRecord {
            key_id: created.id.clone(),
            model_group: "test-group".to_string(),
            model: TEST_MODEL.to_string(),
            input_tokens: 1_000,
            output_tokens: 1_000,
            cost_usd: 1.0,
            timestamp: Utc::now(),
        })
        .await
        .unwrap();
    // Ensure the next authentication reloads the exhausted counters.
    ts.manager()
        .invalidate_cache(&VirtualKeyManager::hash_key(&created.key));

    let (status, _headers, _body) =
        chat_request(ts.router(), Some(&created.key), TEST_MODEL).await;

    assert_eq!(
        status,
        StatusCode::TOO_MANY_REQUESTS,
        "exhausted budget must reject with 429"
    );
}

// ---------------------------------------------------------------------------
// Rate limiting  (Req 5.2)
// ---------------------------------------------------------------------------

/// Req 5.2: with `requests_per_minute = 1`, the first request consumes the only
/// token and the second is rejected with 429 including a `Retry-After` header.
#[tokio::test]
async fn rate_limit_returns_429_with_retry_after() {
    let ts = TestServer::new(EnforcementMode::Required).await;
    let created = ts
        .manager()
        .create_key(CreateKeyParams {
            requests_per_minute: Some(1),
            ..create_defaults()
        })
        .await
        .unwrap();

    // First request consumes the single RPM token → success.
    let (status1, _h1, body1) = chat_request(ts.router(), Some(&created.key), TEST_MODEL).await;
    assert!(
        status1.is_success(),
        "first request within RPM should succeed, got {status1}: {}",
        String::from_utf8_lossy(&body1)
    );

    // Second rapid request exceeds RPM=1 → 429 + Retry-After.
    let (status2, headers2, _body2) =
        chat_request(ts.router(), Some(&created.key), TEST_MODEL).await;
    assert_eq!(status2, StatusCode::TOO_MANY_REQUESTS);
    let retry_after = headers2
        .get(axum::http::header::RETRY_AFTER)
        .expect("Retry-After header must be present on rate-limit rejection");
    // Value is an integer number of seconds.
    let secs: u64 = retry_after.to_str().unwrap().parse().expect("Retry-After is integer seconds");
    assert!(secs >= 1, "Retry-After should be at least 1 second, got {secs}");
}

// ---------------------------------------------------------------------------
// Model access denial  (Req 6.2)
// ---------------------------------------------------------------------------

/// Req 6.2: a key whose model access list does not include the requested model
/// is rejected with 403 before budget/rate-limit checks.
#[tokio::test]
async fn model_access_denial_returns_403() {
    let ts = TestServer::new(EnforcementMode::Required).await;
    let created = ts
        .manager()
        .create_key(CreateKeyParams {
            model_access: Some(vec!["allowed-group".to_string()]),
            ..create_defaults()
        })
        .await
        .unwrap();

    // Request a model that is NOT in the key's access list.
    let (status, _headers, _body) =
        chat_request(ts.router(), Some(&created.key), "other-group").await;

    assert_eq!(
        status,
        StatusCode::FORBIDDEN,
        "requesting a model outside the access list must be denied with 403"
    );
}
