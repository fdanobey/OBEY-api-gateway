//! Integration tests for the OBEY-API gateway HTTP layer.
//!
//! These tests exercise the full Axum router via `tower::ServiceExt::oneshot()`
//! without binding to a real port, validating end-to-end request flows through
//! the gateway's HTTP surface.

use axum::body::Body;
use axum::http::{Request, StatusCode};
use tower::ServiceExt;

use ai_gateway::config::*;
use ai_gateway::gateway::GatewayServer;

mod common;

/// Build a minimal valid Config for integration tests.
fn test_config() -> Config {
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
        providers: vec![Provider {
            name: "test-provider".to_string(),
            provider_type: "openai".to_string(),
            base_url: Some("http://localhost:11434".to_string()),
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
            memory: None,
            reasoning: true,
            codex_base_url_override: None,
            codex_model_override: None,
            instructions_override: None,
            max_rate_limit_cooldown_seconds: None,
        }],
        model_groups: vec![ModelGroup {
            name: "test-group".to_string(),
            version_fallback_enabled: false,
            compression: None,
            memory: None,
            structured_output: None,
            models: vec![ProviderModel {
                cache_support: None,
                cache_min_tokens: None,
                cost_per_million_cache_read_input_tokens: None,
                cost_per_million_cache_creation_input_tokens: None,
                provider: "test-provider".to_string(),
                model: "gpt-4".to_string(),
                cost_per_million_input_tokens: 30.0,
                cost_per_million_output_tokens: 60.0,
                priority: 100,
                structured_output_passthrough: None,
                tier: None,
                context_window: 0,
                specializations: vec![],
                cost_per_million_reasoning_tokens: None,
                reasoning_family: None,
                reasoning_parameter: None,
            }],
        }],
        circuit_breaker: CircuitBreakerConfig::default(),
        retry: RetryConfig::default(),
        logging: LoggingConfig::default(),
        semantic_cache: None,
        exact_cache: ExactCacheConfig::default(),
        prometheus: None,
        context: ai_gateway::config::ContextConfig::default(),
        compression: Default::default(),
        memory: None,
        first_launch_completed: false,
        tray: ai_gateway::config::TrayConfig::default(),
        codex_instructions_url: None,
        streaming: None,
        virtual_keys: Default::default(),
        loop_detection: Default::default(),
        guardrails: None,
        tool_compression: Default::default(),
        smart_routing: Default::default(),
        structured_output: None,
        xhigh_models_allowlist: Default::default(),
        reasoning_models_allowlist: Default::default(),
        codex_search: None,
        reasoning_compat: Default::default(),
    }
}

/// Helper: build a router from a config without binding to a port.
async fn build_app(mut config: Config) -> axum::Router {
    common::isolate_databases(&mut config);
    let server = GatewayServer::new(config, None).await.unwrap();
    server.build_router()
}

/// Helper: send a request and return (status, body bytes).
async fn send(app: axum::Router, req: Request<Body>) -> (StatusCode, Vec<u8>) {
    let resp = app.oneshot(req).await.unwrap();
    let status = resp.status();
    let body = axum::body::to_bytes(resp.into_body(), 1024 * 1024)
        .await
        .unwrap();
    (status, body.to_vec())
}

// ---------------------------------------------------------------------------
// 1. Health check integration
// ---------------------------------------------------------------------------

#[tokio::test]
async fn test_health_check_integration() {
    let app = build_app(test_config()).await;
    let req = Request::get("/health").body(Body::empty()).unwrap();
    let (status, body) = send(app, req).await;

    assert_eq!(status, StatusCode::OK);
    let json: serde_json::Value = serde_json::from_slice(&body).unwrap();
    assert_eq!(json["status"], "ok");
}

// ---------------------------------------------------------------------------
// 2. Admin config GET
// ---------------------------------------------------------------------------

#[tokio::test]
async fn test_admin_config_get() {
    let app = build_app(test_config()).await;
    let req = Request::get("/admin/config").body(Body::empty()).unwrap();
    let (status, body) = send(app, req).await;

    assert_eq!(status, StatusCode::OK);
    let json: serde_json::Value = serde_json::from_slice(&body).unwrap();
    // Config should contain our test provider
    let providers = json["providers"].as_array().unwrap();
    assert!(!providers.is_empty());
    assert_eq!(providers[0]["name"], "test-provider");
}

#[tokio::test]
async fn test_admin_config_get_does_not_panic_when_provider_env_var_is_missing() {
    let mut cfg = test_config();
    cfg.providers[0].api_key_env = Some("MISSING_PROVIDER_API_KEY".to_string());

    let app = build_app(cfg).await;
    let req = Request::get("/admin/config").body(Body::empty()).unwrap();
    let (status, body) = send(app, req).await;

    assert_eq!(status, StatusCode::OK);
    let json: serde_json::Value = serde_json::from_slice(&body).unwrap();
    assert_eq!(json["providers"][0]["api_key_configured"], true);
    assert_eq!(json["providers"][0]["api_key_status"], "environment");
}

// ---------------------------------------------------------------------------
// 3. Admin config validate
// ---------------------------------------------------------------------------

#[tokio::test]
async fn test_admin_config_validate() {
    let app = build_app(test_config()).await;
    // Use a config with a valid (non-zero) port for the validation endpoint
    let mut validatable = test_config();
    validatable.server.port = 8080;
    let valid_config = serde_json::to_string(&validatable).unwrap();

    let req = Request::post("/admin/config/validate")
        .header("content-type", "application/json")
        .body(Body::from(valid_config))
        .unwrap();
    let (status, body) = send(app, req).await;

    assert_eq!(status, StatusCode::OK);
    let json: serde_json::Value = serde_json::from_slice(&body).unwrap();
    assert_eq!(json["valid"], true);
}

// ---------------------------------------------------------------------------
// 4. Admin config export (YAML)
// ---------------------------------------------------------------------------

#[tokio::test]
async fn test_admin_config_export() {
    let app = build_app(test_config()).await;
    let req = Request::get("/admin/config/export")
        .body(Body::empty())
        .unwrap();
    let (status, body) = send(app, req).await;

    assert_eq!(status, StatusCode::OK);
    let text = std::str::from_utf8(&body).unwrap();
    // YAML should contain our provider name
    assert!(text.contains("test-provider"));
    // Should be valid YAML that deserializes back
    let _parsed: Config = serde_yaml::from_str(text).unwrap();
}

#[tokio::test]
async fn test_admin_config_import_yaml() {
    let app = build_app(test_config()).await;
    let yaml = r#"
server:
  host: "127.0.0.1"
  port: 8080
  request_timeout_seconds: 30
  max_request_size_mb: 10
providers:
  - name: "imported-provider"
    type: "openai"
    base_url: "https://api.openai.com/v1"
    timeout_seconds: 30
model_groups:
  - name: "imported-group"
    version_fallback_enabled: false
    models:
      - provider: "imported-provider"
        model: "gpt-4"
        priority: 100
retry:
  max_retries_per_provider: 2
  backoff_sequence_seconds: [1, 2, 4]
"#;

    let req = Request::post("/admin/config/import")
        .header("content-type", "text/plain; charset=utf-8")
        .body(Body::from(yaml))
        .unwrap();
    let (status, body) = send(app, req).await;

    assert_eq!(status, StatusCode::OK);
    let json: serde_json::Value = serde_json::from_slice(&body).unwrap();
    assert_eq!(json["valid"], true);
    assert_eq!(json["config"]["providers"][0]["name"], "imported-provider");
    assert_eq!(json["config"]["retry"]["max_retries_per_provider"], 2);
}

// ---------------------------------------------------------------------------
// 5. Chat completions with no reachable provider
// ---------------------------------------------------------------------------

#[tokio::test]
async fn test_chat_completions_no_provider() {
    let app = build_app(test_config()).await;
    let body_json = serde_json::json!({
        "model": "gpt-4",
        "messages": [{"role": "user", "content": "hello"}],
        "stream": false
    });

    let req = Request::post("/v1/chat/completions")
        .header("content-type", "application/json")
        .body(Body::from(serde_json::to_string(&body_json).unwrap()))
        .unwrap();
    let (status, body) = send(app, req).await;

    // Provider is unreachable so we expect an error (502 or 500-range)
    assert!(
        status.is_server_error() || status == StatusCode::BAD_GATEWAY,
        "Expected server error when provider is unreachable, got {}",
        status
    );
    let json: serde_json::Value = serde_json::from_slice(&body).unwrap();
    assert!(
        json["error"].is_object(),
        "Response should contain error object"
    );
}

#[tokio::test]
async fn test_chat_completions_preserves_trace_id_header() {
    let app = build_app(test_config()).await;
    let body_json = serde_json::json!({
        "model": "gpt-4",
        "messages": [{"role": "user", "content": "hello"}],
        "stream": false
    });

    let req = Request::post("/v1/chat/completions")
        .header("content-type", "application/json")
        .header("x-request-id", "trace-abc-123")
        .body(Body::from(serde_json::to_string(&body_json).unwrap()))
        .unwrap();

    let resp = app.oneshot(req).await.unwrap();
    let header = resp.headers().get("x-trace-id").unwrap().to_str().unwrap();
    assert_eq!(header, "trace-abc-123");
}

// ---------------------------------------------------------------------------
// 6. Models endpoint
// ---------------------------------------------------------------------------

#[tokio::test]
async fn test_models_endpoint() {
    let app = build_app(test_config()).await;
    let req = Request::get("/v1/models").body(Body::empty()).unwrap();
    let (status, body) = send(app, req).await;

    assert_eq!(status, StatusCode::OK);
    let json: serde_json::Value = serde_json::from_slice(&body).unwrap();
    assert_eq!(json["object"], "list");
    let data = json["data"].as_array().unwrap();
    // Should contain our configured model
    let model_ids: Vec<&str> = data.iter().map(|m| m["id"].as_str().unwrap()).collect();
    assert!(
        model_ids.contains(&"gpt-4"),
        "Expected gpt-4 in models list"
    );
}

// ---------------------------------------------------------------------------
// 7. Dashboard serves HTML
// ---------------------------------------------------------------------------

#[tokio::test]
async fn test_dashboard_serves_html() {
    let app = build_app(test_config()).await;
    let req = Request::get("/dashboard").body(Body::empty()).unwrap();
    let (status, body) = send(app, req).await;

    assert_eq!(status, StatusCode::OK);
    let html = std::str::from_utf8(&body).unwrap();
    assert!(
        html.contains("<!DOCTYPE html>") || html.contains("<html"),
        "Expected HTML content"
    );
}

#[tokio::test]
async fn test_dashboard_trailing_slash_redirects_to_canonical_path() {
    let app = build_app(test_config()).await;
    let req = Request::get("/dashboard/").body(Body::empty()).unwrap();
    let resp = app.oneshot(req).await.unwrap();

    assert_eq!(resp.status(), StatusCode::PERMANENT_REDIRECT);
    assert_eq!(resp.headers().get("location").unwrap(), "/dashboard");
}

#[tokio::test]
async fn test_custom_admin_and_dashboard_paths_are_honored() {
    let mut cfg = test_config();
    cfg.admin.path = "/control-panel".to_string();
    cfg.dashboard.path = "/ops".to_string();

    let app = build_app(cfg).await;

    let req = Request::get("/control-panel").body(Body::empty()).unwrap();
    let (admin_status, admin_body) = send(app.clone(), req).await;
    assert_eq!(admin_status, StatusCode::OK);
    let admin_html = std::str::from_utf8(&admin_body).unwrap();
    assert!(
        admin_html.contains("/ops"),
        "Admin UI should link to configured dashboard path"
    );

    let req = Request::get("/ops/").body(Body::empty()).unwrap();
    let dash_redirect = app.clone().oneshot(req).await.unwrap();
    assert_eq!(dash_redirect.status(), StatusCode::PERMANENT_REDIRECT);
    assert_eq!(dash_redirect.headers().get("location").unwrap(), "/ops");

    let req = Request::get("/ops").body(Body::empty()).unwrap();
    let (dash_status, dash_body) = send(app.clone(), req).await;
    assert_eq!(dash_status, StatusCode::OK);
    let dash_html = std::str::from_utf8(&dash_body).unwrap();
    assert!(dash_html.contains("window.__dashboardBasePath=\"/ops\""));
    assert!(dash_html.contains("window.__adminBasePath=\"/control-panel\""));

    let req = Request::get("/ops/metrics").body(Body::empty()).unwrap();
    let (metrics_status, _) = send(app, req).await;
    assert_eq!(metrics_status, StatusCode::OK);
}

#[tokio::test]
async fn test_disabled_admin_and_dashboard_routes_are_not_mounted() {
    let mut cfg = test_config();
    cfg.admin.enabled = false;
    cfg.dashboard.enabled = false;

    let app = build_app(cfg).await;

    let (admin_status, _) = send(
        app.clone(),
        Request::get("/admin/").body(Body::empty()).unwrap(),
    )
    .await;
    assert_eq!(admin_status, StatusCode::NOT_FOUND);

    let (dash_status, _) = send(
        app.clone(),
        Request::get("/dashboard/").body(Body::empty()).unwrap(),
    )
    .await;
    assert_eq!(dash_status, StatusCode::NOT_FOUND);

    let (health_status, _) = send(app, Request::get("/health").body(Body::empty()).unwrap()).await;
    assert_eq!(health_status, StatusCode::OK);
}

// ---------------------------------------------------------------------------
// 8. Prometheus metrics when enabled
// ---------------------------------------------------------------------------

#[tokio::test]
async fn test_prometheus_metrics_when_enabled() {
    let mut cfg = test_config();
    cfg.prometheus = Some(PrometheusConfig {
        enabled: true,
        path: "/metrics".to_string(),
    });

    let app = build_app(cfg).await;
    let req = Request::get("/metrics").body(Body::empty()).unwrap();
    let (status, body) = send(app, req).await;

    assert_eq!(status, StatusCode::OK);
    let text = std::str::from_utf8(&body).unwrap();
    // Prometheus text format markers
    assert!(text.contains("# HELP"), "Expected Prometheus HELP lines");
    assert!(text.contains("# TYPE"), "Expected Prometheus TYPE lines");
    assert!(
        text.contains("obey_api_requests_total"),
        "Expected request counter metric"
    );
}

/// Task 12.4 (Req 11.5): guardrail counter and histogram metrics are exposed on
/// the Prometheus endpoint with the `obey_api_guardrail_` prefix. We record a
/// guardrail stage on the server's shared `Metrics` (as the engine does at
/// runtime), then GET the endpoint through the real router via `oneshot()` and
/// assert the prefixed metric families render.
#[tokio::test]
async fn test_prometheus_exposes_guardrail_metrics_with_prefix() {
    let mut cfg = test_config();
    cfg.prometheus = Some(PrometheusConfig {
        enabled: true,
        path: "/metrics".to_string(),
    });

    // Build the server directly (not via `build_app`) so we can record a
    // guardrail stage on the same `Metrics` instance the endpoint reads.
    common::isolate_databases(&mut cfg);
    let server = GatewayServer::new(cfg, None).await.unwrap();
    server.state.metrics.record_guardrail_stage(
        "pii_pipeline",
        "pii_scan",
        "regex",
        "pre_call",
        "redact",
        12.5,
    );

    let app = server.build_router();
    let req = Request::get("/metrics").body(Body::empty()).unwrap();
    let (status, body) = send(app, req).await;

    assert_eq!(status, StatusCode::OK);
    let text = std::str::from_utf8(&body).unwrap();

    // Counter family with the guardrail prefix (Req 11.1, 11.5).
    assert!(
        text.contains("# TYPE obey_api_guardrail_stage_executions_total counter"),
        "expected guardrail counter TYPE line with obey_api_guardrail_ prefix"
    );
    assert!(
        text.contains(
            "obey_api_guardrail_stage_executions_total{pipeline=\"pii_pipeline\",stage=\"pii_scan\",provider=\"regex\",phase=\"pre_call\",action=\"redact\"} 1"
        ),
        "expected the recorded guardrail counter sample with the obey_api_guardrail_ prefix"
    );

    // Histogram family with the guardrail prefix (Req 11.2, 11.5).
    assert!(
        text.contains("# TYPE obey_api_guardrail_stage_latency_ms histogram"),
        "expected guardrail latency histogram TYPE line with obey_api_guardrail_ prefix"
    );
    assert!(
        text.contains(
            "obey_api_guardrail_stage_latency_ms_count{pipeline=\"pii_pipeline\",stage=\"pii_scan\",provider=\"regex\",phase=\"pre_call\"} 1"
        ),
        "expected the recorded guardrail latency count with the obey_api_guardrail_ prefix"
    );

    // Every guardrail metric line must carry the required prefix (Req 11.5).
    for line in text.lines().filter(|l| l.contains("guardrail")) {
        assert!(
            line.contains("obey_api_guardrail_"),
            "guardrail metric line missing obey_api_guardrail_ prefix: {line}"
        );
    }
}

#[tokio::test]
async fn test_admin_config_validate_accepts_reliability_fields() {
    let app = build_app(test_config()).await;
    let mut validatable = test_config();
    validatable.server.port = 8080;
    validatable.retry.jitter_enabled = true;
    validatable.retry.jitter_ratio = 0.25;
    validatable.providers[0].connection_pool.max_idle_per_host = 12;
    validatable.providers[0]
        .connection_pool
        .idle_timeout_seconds = 120;
    validatable.providers[0].budget = Some(ProviderBudgetConfig {
        limit_usd: 25.0,
        reset_policy: Some(BudgetResetPolicy::Manual),
    });

    let req = Request::post("/admin/config/validate")
        .header("content-type", "application/json")
        .body(Body::from(serde_json::to_string(&validatable).unwrap()))
        .unwrap();
    let (status, body) = send(app, req).await;

    assert_eq!(status, StatusCode::OK);
    let json: serde_json::Value = serde_json::from_slice(&body).unwrap();
    assert_eq!(json["valid"], true);
}

// ---------------------------------------------------------------------------
// 9. Admin auth integration — 401 when auth enabled without credentials
// ---------------------------------------------------------------------------

#[tokio::test]
async fn test_admin_auth_integration() {
    let user_env = "INTEG_TEST_ADMIN_USER";
    let pass_env = "INTEG_TEST_ADMIN_PASS";
    std::env::set_var(user_env, "admin");
    std::env::set_var(pass_env, "secret");

    let mut cfg = test_config();
    cfg.admin.auth = AdminAuthConfig {
        enabled: true,
        username_env: Some(user_env.to_string()),
        password_env: Some(pass_env.to_string()),
        allow_unauthenticated_non_loopback: false,
    };

    let app = build_app(cfg).await;

    // Unauthenticated request → 401
    let req = Request::get("/admin/config").body(Body::empty()).unwrap();
    let (status, _) = send(app, req).await;
    assert_eq!(status, StatusCode::UNAUTHORIZED);

    std::env::remove_var(user_env);
    std::env::remove_var(pass_env);
}

// ---------------------------------------------------------------------------
// 10. Streaming early synthetic SSE event (streaming-reliability task 2.3)
// ---------------------------------------------------------------------------

use wiremock::matchers::{method as wm_method, path as wm_path};
use wiremock::{Mock, MockServer, ResponseTemplate};

/// The fixed id the mock provider returns; used to distinguish the synthetic
/// early-event id (a fresh uuid) from the provider's own envelope id.
const MOCK_PROVIDER_ID: &str = "chatcmpl-mock-streaming";

/// Start a mock OpenAI-compatible provider that returns a static, cacheable
/// chat completion (finish_reason: stop, no tool calls).
async fn start_streaming_mock() -> MockServer {
    let server = MockServer::start().await;
    let body = serde_json::json!({
        "id": MOCK_PROVIDER_ID,
        "object": "chat.completion",
        "created": 1700000000_i64,
        "model": "gpt-4",
        "choices": [{
            "index": 0,
            "message": { "role": "assistant", "content": "Hello from mock provider!" },
            "finish_reason": "stop"
        }],
        "usage": { "prompt_tokens": 10, "completion_tokens": 8, "total_tokens": 18 }
    });
    Mock::given(wm_method("POST"))
        .and(wm_path("/v1/chat/completions"))
        .respond_with(ResponseTemplate::new(200).set_body_json(body))
        .mount(&server)
        .await;
    server
}

/// Build a config whose single provider points at `mock_uri`, with the given
/// `emit_early_event` streaming setting.
fn streaming_test_config(mock_uri: &str, emit_early_event: bool) -> Config {
    let mut cfg = test_config();
    cfg.providers[0].base_url = Some(mock_uri.to_string());
    cfg.streaming = Some(StreamingConfig {
        emit_early_event,
        // Task 5.5: the mock provider returns a complete chat.completion JSON
        // body (not real SSE), so force the deterministic buffer-and-replay
        // path. Pass-through would read that JSON as SSE, find no `data:`
        // lines, and emit only `[DONE]`. True-streaming integration coverage
        // (a mock that emits real SSE) is task 5.6's responsibility.
        passthrough_enabled: false,
        ..StreamingConfig::default()
    });
    cfg
}

fn streaming_request_body() -> Body {
    Body::from(
        serde_json::to_string(&serde_json::json!({
            "model": "gpt-4",
            "messages": [{"role": "user", "content": "hello"}],
            "stream": true
        }))
        .unwrap(),
    )
}

/// Parse the JSON `data:` chunks out of an SSE body, skipping the `[DONE]`
/// sentinel and any keep-alive comment lines.
fn parse_sse_chunks(body: &[u8]) -> Vec<serde_json::Value> {
    let text = std::str::from_utf8(body).unwrap();
    text.lines()
        .filter_map(|line| line.strip_prefix("data: "))
        .filter(|payload| *payload != "[DONE]")
        .map(|payload| serde_json::from_str::<serde_json::Value>(payload).unwrap())
        .collect()
}

async fn post_stream(app: axum::Router) -> Vec<serde_json::Value> {
    let req = Request::post("/v1/chat/completions")
        .header("content-type", "application/json")
        .body(streaming_request_body())
        .unwrap();
    let (status, body) = send(app, req).await;
    assert_eq!(
        status,
        StatusCode::OK,
        "streaming request should return 200"
    );
    parse_sse_chunks(&body)
}

/// Req 1.1, 1.3, 1.5: with `emit_early_event` enabled (the default), the first
/// SSE chunk is the synthetic role-only early event, and every subsequent chunk
/// reuses that early event's fresh id rather than the provider's own id.
#[tokio::test]
async fn test_streaming_emits_early_event_by_default() {
    let mock = start_streaming_mock().await;
    let app = build_app(streaming_test_config(&mock.uri(), true)).await;

    let chunks = post_stream(app).await;
    assert!(
        !chunks.is_empty(),
        "expected at least the early event chunk"
    );

    // First chunk is the synthetic early event: role delta, no content, null finish.
    let early = &chunks[0];
    assert_eq!(early["object"], "chat.completion.chunk");
    assert_eq!(early["model"], "gpt-4");
    assert_eq!(early["choices"][0]["delta"]["role"], "assistant");
    assert!(early["choices"][0]["delta"].get("content").is_none());
    assert!(early["choices"][0]["finish_reason"].is_null());

    let early_id = early["id"].as_str().unwrap();
    assert!(
        early_id.starts_with("chatcmpl-"),
        "early id should be a chatcmpl id"
    );
    assert_ne!(
        early_id, MOCK_PROVIDER_ID,
        "early event must use a fresh id, not the provider's"
    );

    // Subsequent chunks share the early event id (Req 1.5 consistent stream id).
    for chunk in &chunks {
        assert_eq!(
            chunk["id"].as_str().unwrap(),
            early_id,
            "all chunks share the early event id"
        );
    }
}

/// Req 1.6: when `emit_early_event` is false, no synthetic early event is
/// prepended — the stream begins with the normal role+content chunk and uses
/// the provider's own response id.
#[tokio::test]
async fn test_streaming_emit_early_event_false_disables_early_event() {
    let mock = start_streaming_mock().await;
    let app = build_app(streaming_test_config(&mock.uri(), false)).await;

    let chunks = post_stream(app).await;
    assert!(!chunks.is_empty());

    // First chunk is the standard role chunk (carries an empty content field),
    // not the content-free synthetic early event.
    let first = &chunks[0];
    assert_eq!(first["choices"][0]["delta"]["role"], "assistant");
    assert!(first["choices"][0]["delta"].get("content").is_some());

    // The provider's own id is used throughout (no fresh early-event id).
    for chunk in &chunks {
        assert_eq!(chunk["id"], MOCK_PROVIDER_ID);
    }
}

/// Req 1.4: a cache hit serves immediately and skips the early event. The first
/// request populates the exact cache; the second (identical) request replays
/// from cache, producing chunks that carry the provider's id with no synthetic
/// early event prepended.
#[tokio::test]
async fn test_streaming_cache_hit_skips_early_event() {
    let mock = start_streaming_mock().await;
    // Same app instance so the in-memory exact cache persists across requests.
    let app = build_app(streaming_test_config(&mock.uri(), true)).await;

    // First request: cache miss → early event present with a fresh id.
    let first_chunks = post_stream(app.clone()).await;
    assert_ne!(
        first_chunks[0]["id"], MOCK_PROVIDER_ID,
        "first (miss) emits a fresh early-event id"
    );

    // Second identical request: cache hit → no early event, provider id reused.
    let cached_chunks = post_stream(app).await;
    assert!(!cached_chunks.is_empty());
    let first = &cached_chunks[0];
    // Cache replay emits the standard role chunk (with content field), not the
    // content-free early event.
    assert_eq!(first["choices"][0]["delta"]["role"], "assistant");
    assert!(first["choices"][0]["delta"].get("content").is_some());
    for chunk in &cached_chunks {
        assert_eq!(
            chunk["id"], MOCK_PROVIDER_ID,
            "cache replay keeps the provider id (no early event)"
        );
    }
}

/// Build a streaming config pointing at `mock_uri` with a custom keep-alive
/// interval. `emit_early_event` keeps its default (true).
fn streaming_keepalive_config(mock_uri: &str, keepalive_interval_seconds: u64) -> Config {
    let mut cfg = test_config();
    cfg.providers[0].base_url = Some(mock_uri.to_string());
    cfg.streaming = Some(StreamingConfig {
        keepalive_interval_seconds,
        // Task 5.5: mock returns full JSON (not SSE) — keep the buffered path
        // so these keep-alive assertions stay deterministic (see
        // streaming_test_config for the full rationale).
        passthrough_enabled: false,
        ..StreamingConfig::default()
    });
    cfg
}

/// Assert the SSE stream completed successfully by checking the assembled
/// content matches what the mock provider returned.
fn assert_stream_has_provider_content(chunks: &[serde_json::Value]) {
    assert!(!chunks.is_empty(), "expected SSE chunks from the stream");
    let content: String = chunks
        .iter()
        .filter_map(|c| c["choices"][0]["delta"]["content"].as_str())
        .collect();
    assert_eq!(
        content, "Hello from mock provider!",
        "the relayed stream should reconstruct the provider's content"
    );
}

/// Req 2.4: a custom keep-alive interval is applied to the SSE response and the
/// stream still completes normally — the client receives a 200 and the full set
/// of content chunks. (axum's `KeepAlive` interval is not introspectable, so the
/// behavioral guarantee is that the configured stream works end-to-end.)
#[tokio::test]
async fn test_streaming_custom_keepalive_interval_produces_working_stream() {
    let mock = start_streaming_mock().await;
    let app = build_app(streaming_keepalive_config(&mock.uri(), 5)).await;

    let chunks = post_stream(app).await;
    assert_stream_has_provider_content(&chunks);
}

/// Req 2.5: a keep-alive interval of 0 disables the custom interval and falls
/// back to axum's default keep-alive behavior; the stream still returns 200 and
/// streams the expected chunks.
#[tokio::test]
async fn test_streaming_keepalive_interval_zero_falls_back_to_default() {
    let mock = start_streaming_mock().await;
    let app = build_app(streaming_keepalive_config(&mock.uri(), 0)).await;

    let chunks = post_stream(app).await;
    assert_stream_has_provider_content(&chunks);
}

// ---------------------------------------------------------------------------
// 11. Graceful timeout error events for streaming (streaming-reliability task 4.3)
// ---------------------------------------------------------------------------

/// Start a mock provider that accepts the request but delays its response body
/// well beyond the configured TTFB timeout, forcing a time-to-first-byte
/// timeout in the gateway.
async fn start_slow_streaming_mock(delay_secs: u64) -> MockServer {
    let server = MockServer::start().await;
    let body = serde_json::json!({
        "id": MOCK_PROVIDER_ID,
        "object": "chat.completion",
        "created": 1700000000_i64,
        "model": "gpt-4",
        "choices": [{
            "index": 0,
            "message": { "role": "assistant", "content": "too late" },
            "finish_reason": "stop"
        }],
        "usage": { "prompt_tokens": 1, "completion_tokens": 1, "total_tokens": 2 }
    });
    Mock::given(wm_method("POST"))
        .and(wm_path("/v1/chat/completions"))
        .respond_with(
            ResponseTemplate::new(200)
                .set_body_json(body)
                .set_delay(std::time::Duration::from_secs(delay_secs)),
        )
        .mount(&server)
        .await;
    server
}

/// Build a streaming config that points at `mock_uri`, keeps the early event
/// enabled, and forces a 1s TTFB timeout so the slow mock reliably times out.
fn streaming_ttfb_timeout_config(mock_uri: &str) -> Config {
    let mut cfg = streaming_test_config(mock_uri, true);
    cfg.providers[0].ttfb_timeout_seconds = Some(1);
    cfg
}

/// Return the ordered list of raw `data:` payload strings from an SSE body,
/// including the `[DONE]` sentinel (unlike `parse_sse_chunks`, which drops it).
/// Keep-alive comment lines (`:` prefixed) are skipped.
fn raw_sse_data_lines(body: &[u8]) -> Vec<String> {
    let text = std::str::from_utf8(body).unwrap();
    text.lines()
        .filter_map(|line| line.strip_prefix("data: "))
        .map(|s| s.to_string())
        .collect()
}

/// Req 5.1, 5.4, 5.5: when a TTFB timeout occurs after the early event has put
/// the client in SSE parsing mode, the gateway emits a graceful error frame
/// (an `{"error":{...}}` data event carrying a message, type, and trace_id)
/// followed immediately by the terminating `data: [DONE]` sentinel — never a
/// silent TCP close.
#[tokio::test]
async fn test_streaming_ttfb_timeout_emits_error_event_then_done() {
    // Mock delays 3s; gateway TTFB timeout is 1s → timeout fires.
    let mock = start_slow_streaming_mock(3).await;
    let app = build_app(streaming_ttfb_timeout_config(&mock.uri())).await;

    let req = Request::post("/v1/chat/completions")
        .header("content-type", "application/json")
        .body(streaming_request_body())
        .unwrap();
    let (status, body) = send(app, req).await;

    // The early event was already emitted, so the response is a 200 SSE stream
    // rather than an HTTP error status.
    assert_eq!(
        status,
        StatusCode::OK,
        "early event forces a 200 SSE stream"
    );

    let data_lines = raw_sse_data_lines(&body);
    assert!(
        data_lines.len() >= 2,
        "expected at least an error frame and [DONE]"
    );

    // The stream terminates with the [DONE] sentinel (Req 5.4).
    assert_eq!(
        data_lines.last().map(String::as_str),
        Some("[DONE]"),
        "stream must terminate with [DONE]"
    );

    // The data line immediately before [DONE] is the graceful error frame.
    let error_line = &data_lines[data_lines.len() - 2];
    let error_json: serde_json::Value =
        serde_json::from_str(error_line).expect("error frame before [DONE] must be valid JSON");
    let error_obj = &error_json["error"];
    assert!(
        error_obj.is_object(),
        "error frame must carry an `error` object"
    );

    // Error frame shape (Req 5.1) and trace_id correlation (Req 5.5).
    assert!(
        error_obj["message"].as_str().is_some_and(|m| !m.is_empty()),
        "error frame must include a non-empty message"
    );
    assert!(
        error_obj["type"].as_str().is_some_and(|t| !t.is_empty()),
        "error frame must include an error type"
    );
    // Req 5.1: the TTFB timeout must surface as the precise `ttfb_timeout_error`
    // type end-to-end. The router wraps single-provider timeouts in
    // AllProvidersFailed, so this guards that the handler recovers the timeout
    // kind from the aggregated attempts rather than emitting a generic
    // `stream_error`.
    assert_eq!(
        error_obj["type"].as_str(),
        Some("ttfb_timeout_error"),
        "TTFB timeout must map to the exact ttfb_timeout_error type (Req 5.1)"
    );
    assert!(
        error_obj["trace_id"]
            .as_str()
            .is_some_and(|t| !t.is_empty()),
        "error frame must include a trace_id for correlation (Req 5.5)"
    );

    // The first data line is the synthetic early event (role delta, no content),
    // proving the client was already in SSE mode before the error frame.
    let first_json: serde_json::Value = serde_json::from_str(&data_lines[0]).unwrap();
    assert_eq!(first_json["choices"][0]["delta"]["role"], "assistant");
}

// ---------------------------------------------------------------------------
// 12. True streaming pass-through (streaming-reliability task 5.6)
//
// These tests exercise the real pass-through relay path: the mock provider
// emits an actual `text/event-stream` SSE body, the gateway's
// `route_request_streaming` returns `PassThrough`, and `relay_passthrough_stream`
// forwards the provider chunks verbatim. Unlike the buffered helpers above,
// these configs keep `passthrough_enabled: true`.
//
// Inter-chunk and total timeout coverage (Req 3.11, 3.12) lives as deterministic
// unit tests against `relay_passthrough_stream` in the handlers module
// (`relay_emits_chunk_timeout_error_when_provider_stalls` and
// `relay_emits_total_timeout_error_when_stream_exceeds_budget`) because wiremock
// cannot stall mid-body and virtual-time control gives reproducible firing.
// ---------------------------------------------------------------------------

/// Build a single `data: {chunk}` SSE line carrying a content delta with the
/// mock provider's own id, so tests can assert the provider's chunks were
/// forwarded verbatim (distinct from the synthetic early-event id).
fn sse_content_chunk(content: &str, finish_reason: Option<&str>) -> String {
    let finish = match finish_reason {
        Some(fr) => format!("\"{}\"", fr),
        None => "null".to_string(),
    };
    let value = serde_json::json!({
        "id": MOCK_PROVIDER_ID,
        "object": "chat.completion.chunk",
        "created": 1700000000_i64,
        "model": "gpt-4",
        "choices": [{
            "index": 0,
            "delta": { "content": content },
            "finish_reason": serde_json::from_str::<serde_json::Value>(&finish).unwrap()
        }]
    });
    format!("data: {}\n\n", value)
}

/// Start a mock provider that responds with a real `text/event-stream` body.
async fn start_sse_streaming_mock(sse_body: String) -> MockServer {
    let server = MockServer::start().await;
    Mock::given(wm_method("POST"))
        .and(wm_path("/v1/chat/completions"))
        .respond_with(
            ResponseTemplate::new(200)
                .insert_header("content-type", "text/event-stream")
                .set_body_string(sse_body),
        )
        .mount(&server)
        .await;
    server
}

/// Build a config whose single provider points at `mock_uri` with true
/// streaming pass-through ENABLED (the default). Used for the SSE relay tests.
fn streaming_passthrough_config(mock_uri: &str) -> Config {
    let mut cfg = test_config();
    cfg.providers[0].base_url = Some(mock_uri.to_string());
    cfg.streaming = Some(StreamingConfig {
        passthrough_enabled: true,
        ..StreamingConfig::default()
    });
    cfg
}

/// Req 3.2: when the provider streams real SSE, the gateway forwards each
/// content chunk to the client and terminates with a single `[DONE]`. The
/// synthetic early event (default on) is the first data line; the provider's
/// own content chunks follow and reconstruct the upstream text.
#[tokio::test]
async fn test_streaming_passthrough_forwards_provider_chunks_and_terminates() {
    let body = format!(
        "{}{}{}",
        sse_content_chunk("Hello from ", None),
        sse_content_chunk("mock provider!", Some("stop")),
        "data: [DONE]\n\n",
    );
    let mock = start_sse_streaming_mock(body).await;
    let app = build_app(streaming_passthrough_config(&mock.uri())).await;

    let req = Request::post("/v1/chat/completions")
        .header("content-type", "application/json")
        .body(streaming_request_body())
        .unwrap();
    let (status, raw) = send(app, req).await;
    assert_eq!(status, StatusCode::OK, "pass-through stream returns 200");

    // The forwarded provider chunks reconstruct the upstream content (Req 3.2).
    let chunks = parse_sse_chunks(&raw);
    assert_stream_has_provider_content(&chunks);

    // The first data line is the synthetic early event (role delta, fresh id),
    // proving the client was in SSE mode before any provider chunk arrived.
    assert_eq!(chunks[0]["choices"][0]["delta"]["role"], "assistant");
    assert_ne!(
        chunks[0]["id"], MOCK_PROVIDER_ID,
        "early event uses a fresh id"
    );

    // The forwarded content chunks carry the provider's own id (verbatim relay).
    assert!(
        chunks.iter().any(|c| c["id"] == MOCK_PROVIDER_ID),
        "provider chunks should be forwarded with the provider id"
    );

    // The stream terminates with exactly one trailing [DONE] (Req 3.6).
    let data_lines = raw_sse_data_lines(&raw);
    assert_eq!(data_lines.last().map(String::as_str), Some("[DONE]"));
}

/// Req 3.6: the upstream `data: [DONE]` is consumed and the gateway emits
/// exactly one terminal `[DONE]` with no error frame on a clean stream.
#[tokio::test]
async fn test_streaming_passthrough_done_terminates_cleanly() {
    let body = format!(
        "{}{}",
        sse_content_chunk("clean finish", Some("stop")),
        "data: [DONE]\n\n",
    );
    let mock = start_sse_streaming_mock(body).await;
    let app = build_app(streaming_passthrough_config(&mock.uri())).await;

    let req = Request::post("/v1/chat/completions")
        .header("content-type", "application/json")
        .body(streaming_request_body())
        .unwrap();
    let (status, raw) = send(app, req).await;
    assert_eq!(status, StatusCode::OK);

    let data_lines = raw_sse_data_lines(&raw);
    // Exactly one [DONE], and it is the final line.
    let done_count = data_lines.iter().filter(|l| l.as_str() == "[DONE]").count();
    assert_eq!(done_count, 1, "exactly one terminal [DONE]");
    assert_eq!(data_lines.last().map(String::as_str), Some("[DONE]"));

    // No error frame anywhere in the stream (clean completion).
    for line in &data_lines {
        if line.as_str() == "[DONE]" {
            continue;
        }
        let json: serde_json::Value = serde_json::from_str(line).unwrap();
        assert!(
            json.get("error").is_none(),
            "clean stream must contain no error frame"
        );
    }
}

/// Req 3.4 / 3.5 / 3.6: a mid-stream error frame from the provider is detected
/// and forwarded to the client as a graceful `{"error":...}` SSE event, followed
/// by the terminating `[DONE]`. Forwarded content before the error is preserved.
#[tokio::test]
async fn test_streaming_passthrough_midstream_error_forwarded_then_done() {
    let body = format!(
        "{}{}",
        sse_content_chunk("partial answer", None),
        "data: {\"error\":{\"message\":\"boom\",\"type\":\"server_error\"}}\n\n",
    );
    let mock = start_sse_streaming_mock(body).await;
    let app = build_app(streaming_passthrough_config(&mock.uri())).await;

    let req = Request::post("/v1/chat/completions")
        .header("content-type", "application/json")
        .body(streaming_request_body())
        .unwrap();
    let (status, raw) = send(app, req).await;
    assert_eq!(
        status,
        StatusCode::OK,
        "early event forces a 200 SSE stream"
    );

    let data_lines = raw_sse_data_lines(&raw);
    assert_eq!(
        data_lines.last().map(String::as_str),
        Some("[DONE]"),
        "stream terminates with [DONE] after the error frame (Req 3.6)"
    );

    // The frame immediately before [DONE] is the graceful error event (Req 3.5).
    let error_line = &data_lines[data_lines.len() - 2];
    let error_json: serde_json::Value =
        serde_json::from_str(error_line).expect("error frame must be valid JSON");
    let error_obj = &error_json["error"];
    assert!(
        error_obj.is_object(),
        "error frame carries an `error` object"
    );
    assert_eq!(
        error_obj["message"].as_str(),
        Some("boom"),
        "the upstream error message is surfaced to the client"
    );
    assert_eq!(
        error_obj["type"].as_str(),
        Some("stream_error"),
        "mid-stream error frames map to the stream_error type"
    );

    // The partial content forwarded before the error is still delivered.
    let content: String = parse_sse_chunks(&raw)
        .iter()
        .filter_map(|c| {
            c["choices"][0]["delta"]["content"]
                .as_str()
                .map(String::from)
        })
        .collect();
    assert_eq!(content, "partial answer", "pre-error content is forwarded");
}

/// Req 3.8: Bedrock providers require response transformation, so the gateway
/// MUST NOT use true streaming pass-through — it falls back to buffer-and-replay.
/// The mock returns a full (non-SSE) `chat.completion` JSON body; if pass-through
/// were used, the relay would find no `data:` lines and emit only `[DONE]`.
/// Reconstructing the provider content from re-chunked SSE proves the buffered
/// path was taken for the Bedrock provider.
#[tokio::test]
async fn test_streaming_bedrock_uses_buffered_mode() {
    // start_streaming_mock returns a complete chat.completion JSON body.
    let mock = start_streaming_mock().await;
    let mut cfg = streaming_passthrough_config(&mock.uri());
    // Bedrock provider type triggers provider_needs_transformation == true.
    cfg.providers[0].provider_type = "bedrock".to_string();
    // No api_key is set, so the configured base_url (the mock) is used directly
    // rather than the Bedrock Mantle endpoint.
    let app = build_app(cfg).await;

    let chunks = post_stream(app).await;
    // Buffered re-chunk path reconstructs the provider's content; a pass-through
    // attempt on a JSON body would have yielded no content chunks at all.
    assert_stream_has_provider_content(&chunks);
}

// ---------------------------------------------------------------------------
// 13. Streaming failover on mid-stream failure (streaming-reliability task 6.4)
//
// These tests exercise the end-to-end pre/post-content failover loop in
// `chat_completions_stream`. Triggering a *deterministic* upstream failure with
// wiremock is the key challenge: a clean EOF on an empty body is classified as
// `Completed` (no failover), so instead the first provider emits a
// `data: {"error":...}` frame as its FIRST data line. The relay classifies that
// as `RelayLineAction::Error` with `content_forwarded == false`, yielding
// `RelayOutcome::FailedBeforeContent` — the exact pre-content failure the
// handler fails over on (Req 4.1). Emitting a content chunk BEFORE the error
// frame flips the relay to `FailedAfterContent` (Req 4.2). Both are
// wiremock-friendly and reproducible.
// ---------------------------------------------------------------------------

/// A `data: {error}` SSE frame used to provoke a deterministic relay failure.
/// As the FIRST data line it is a pre-content failure; after a content chunk it
/// is a post-content failure.
fn sse_error_frame() -> String {
    "data: {\"error\":{\"message\":\"boom\",\"type\":\"server_error\"}}\n\n".to_string()
}

/// Build a two-provider failover config: `primary` (priority 100, tried first)
/// and `backup` (priority 200) in a single `gpt-4` model group, each pointing at
/// its own mock. Both are plain `openai` providers (names contain no
/// glm/kimi/nano-gpt), so true streaming pass-through applies and the handler's
/// pre-content failover loop advances primary -> backup.
fn streaming_failover_config(primary_uri: &str, backup_uri: &str) -> Config {
    let mut cfg = test_config();
    let template = cfg.providers[0].clone();
    let primary = Provider {
        name: "primary".to_string(),
        base_url: Some(primary_uri.to_string()),
        memory: None,
        ..template.clone()
    };
    let backup = Provider {
        name: "backup".to_string(),
        base_url: Some(backup_uri.to_string()),
        memory: None,
        ..template
    };
    cfg.providers = vec![primary, backup];
    cfg.model_groups = vec![ModelGroup {
        name: "gpt-4-group".to_string(),
        version_fallback_enabled: false,
        compression: None,
        memory: None,
        structured_output: None,
        models: vec![
            ProviderModel {
                cache_support: None,
                cache_min_tokens: None,
                cost_per_million_cache_read_input_tokens: None,
                cost_per_million_cache_creation_input_tokens: None,
                provider: "primary".to_string(),
                model: "gpt-4".to_string(),
                cost_per_million_input_tokens: 30.0,
                cost_per_million_output_tokens: 60.0,
                priority: 100,
                structured_output_passthrough: None,
                tier: None,
                context_window: 0,
                specializations: vec![],
                cost_per_million_reasoning_tokens: None,
                reasoning_family: None,
                reasoning_parameter: None,
            },
            ProviderModel {
                cache_support: None,
                cache_min_tokens: None,
                cost_per_million_cache_read_input_tokens: None,
                cost_per_million_cache_creation_input_tokens: None,
                provider: "backup".to_string(),
                model: "gpt-4".to_string(),
                cost_per_million_input_tokens: 30.0,
                cost_per_million_output_tokens: 60.0,
                priority: 200,
                structured_output_passthrough: None,
                tier: None,
                context_window: 0,
                specializations: vec![],
                cost_per_million_reasoning_tokens: None,
                reasoning_family: None,
                reasoning_parameter: None,
            },
        ],
    }];
    cfg.streaming = Some(StreamingConfig {
        passthrough_enabled: true,
        ..StreamingConfig::default()
    });
    cfg
}

/// Req 4.1, 4.4: the primary provider emits an error frame BEFORE any content,
/// so the gateway transparently fails over to the backup provider. The client
/// sees an uninterrupted stream: exactly one synthetic `role: assistant` early
/// event, the backup's content chunks, and a terminating `[DONE]` — with NO
/// error frame leaked (pre-content failover is invisible to the client).
#[tokio::test]
async fn test_streaming_failover_before_content_is_transparent() {
    // Primary fails pre-content; backup streams a normal completion.
    let primary = start_sse_streaming_mock(sse_error_frame()).await;
    let backup_body = format!(
        "{}{}{}",
        sse_content_chunk("Hello from ", None),
        sse_content_chunk("mock provider!", Some("stop")),
        "data: [DONE]\n\n",
    );
    let backup = start_sse_streaming_mock(backup_body).await;

    let app = build_app(streaming_failover_config(&primary.uri(), &backup.uri())).await;

    let req = Request::post("/v1/chat/completions")
        .header("content-type", "application/json")
        .body(streaming_request_body())
        .unwrap();
    let (status, raw) = send(app, req).await;
    assert_eq!(status, StatusCode::OK, "failover stream returns 200");

    let chunks = parse_sse_chunks(&raw);

    // The backup provider's content is delivered intact (Req 4.1).
    assert_stream_has_provider_content(&chunks);

    // Exactly ONE role event — the synthetic early event. The pre-content
    // failover must NOT emit a second `role: assistant` event (Req 4.4).
    let role_events = chunks
        .iter()
        .filter(|c| c["choices"][0]["delta"]["role"] == "assistant")
        .count();
    assert_eq!(
        role_events, 1,
        "exactly one role event across a transparent failover (Req 4.4)"
    );

    // No error frame leaked to the client — the primary's failure was invisible.
    for chunk in &chunks {
        assert!(
            chunk.get("error").is_none(),
            "pre-content failover must not leak an error frame to the client"
        );
    }

    // The stream terminates cleanly with [DONE].
    let data_lines = raw_sse_data_lines(&raw);
    assert_eq!(data_lines.last().map(String::as_str), Some("[DONE]"));
}

/// Req 4.2: when the primary provider fails AFTER forwarding content (a content
/// chunk then an error frame), the gateway CANNOT transparently fail over. It
/// emits a graceful error event + `[DONE]` and does NOT switch to the backup —
/// the client receives the partial content and the error, with no duplicate
/// content from the backup provider.
#[tokio::test]
async fn test_streaming_failover_after_content_emits_error_no_failover() {
    // Primary forwards content, THEN errors → post-content failure (no failover).
    let primary_body = format!(
        "{}{}",
        sse_content_chunk("partial answer", None),
        sse_error_frame(),
    );
    let primary = start_sse_streaming_mock(primary_body).await;
    // Backup would stream different content; it must NOT be used.
    let backup_body = format!(
        "{}{}",
        sse_content_chunk("SHOULD NOT APPEAR", Some("stop")),
        "data: [DONE]\n\n",
    );
    let backup = start_sse_streaming_mock(backup_body).await;

    let app = build_app(streaming_failover_config(&primary.uri(), &backup.uri())).await;

    let req = Request::post("/v1/chat/completions")
        .header("content-type", "application/json")
        .body(streaming_request_body())
        .unwrap();
    let (status, raw) = send(app, req).await;
    assert_eq!(
        status,
        StatusCode::OK,
        "early event forces a 200 SSE stream"
    );

    // Only the primary's partial content reaches the client — the backup is
    // never used (no transparent mid-content failover).
    let content: String = parse_sse_chunks(&raw)
        .iter()
        .filter_map(|c| {
            c["choices"][0]["delta"]["content"]
                .as_str()
                .map(String::from)
        })
        .collect();
    assert_eq!(
        content, "partial answer",
        "only primary's pre-error content; backup not used (Req 4.2)"
    );

    let data_lines = raw_sse_data_lines(&raw);
    // The stream terminates with [DONE] preceded by a graceful error frame.
    assert_eq!(data_lines.last().map(String::as_str), Some("[DONE]"));
    let error_line = &data_lines[data_lines.len() - 2];
    let error_json: serde_json::Value =
        serde_json::from_str(error_line).expect("error frame before [DONE] must be valid JSON");
    assert!(
        error_json["error"].is_object(),
        "post-content failure emits an error frame (Req 4.2)"
    );
}

/// Req 4.3: when EVERY provider fails before content, the gateway exhausts the
/// failover order and emits a single aggregated error frame followed by
/// `[DONE]`. No content is delivered and the aggregated failure maps to the
/// generic `stream_error` type (none of the attempts are timeouts).
#[tokio::test]
async fn test_streaming_failover_all_providers_fail_aggregated_error() {
    // Both providers fail pre-content.
    let primary = start_sse_streaming_mock(sse_error_frame()).await;
    let backup = start_sse_streaming_mock(sse_error_frame()).await;

    let app = build_app(streaming_failover_config(&primary.uri(), &backup.uri())).await;

    let req = Request::post("/v1/chat/completions")
        .header("content-type", "application/json")
        .body(streaming_request_body())
        .unwrap();
    let (status, raw) = send(app, req).await;
    assert_eq!(
        status,
        StatusCode::OK,
        "early event forces a 200 SSE stream"
    );

    // No provider content was forwarded (only the early event).
    let content: String = parse_sse_chunks(&raw)
        .iter()
        .filter_map(|c| {
            c["choices"][0]["delta"]["content"]
                .as_str()
                .map(String::from)
        })
        .collect();
    assert!(
        content.is_empty(),
        "no content when all providers fail pre-content"
    );

    let data_lines = raw_sse_data_lines(&raw);
    assert_eq!(
        data_lines.last().map(String::as_str),
        Some("[DONE]"),
        "aggregated failure still terminates with [DONE]"
    );

    // The frame before [DONE] is the aggregated error (Req 4.3).
    let error_line = &data_lines[data_lines.len() - 2];
    let error_json: serde_json::Value =
        serde_json::from_str(error_line).expect("aggregated error frame must be valid JSON");
    assert!(
        error_json["error"].is_object(),
        "aggregated error frame carries an `error` object"
    );
    assert_eq!(
        error_json["error"]["type"].as_str(),
        Some("stream_error"),
        "non-timeout aggregated failure maps to the generic stream_error type"
    );
}

/// Extract a single `obey_api_provider_failures_total{provider="..."}` counter
/// value from a Prometheus exposition body. Returns 0 if the series is absent.
fn provider_failures_metric(metrics_body: &str, provider: &str) -> u64 {
    let needle = format!(
        "obey_api_provider_failures_total{{provider=\"{}\"}} ",
        provider
    );
    metrics_body
        .lines()
        .find_map(|line| line.strip_prefix(needle.as_str()))
        .and_then(|v| v.trim().parse::<u64>().ok())
        .unwrap_or(0)
}

/// Req 4.5: a provider that fails (disconnects/errors before content) during a
/// streaming pass-through has its failure recorded against the circuit breaker
/// and metrics. After a pre-content failover off `primary`, the Prometheus
/// `obey_api_provider_failures_total{provider="primary"}` counter is incremented.
#[tokio::test]
async fn test_streaming_failover_records_circuit_breaker_failure() {
    let primary = start_sse_streaming_mock(sse_error_frame()).await;
    let backup_body = format!(
        "{}{}",
        sse_content_chunk("recovered", Some("stop")),
        "data: [DONE]\n\n",
    );
    let backup = start_sse_streaming_mock(backup_body).await;

    let mut cfg = streaming_failover_config(&primary.uri(), &backup.uri());
    cfg.prometheus = Some(PrometheusConfig {
        enabled: true,
        path: "/metrics".to_string(),
    });
    // Same app instance so the streaming request and the /metrics scrape share
    // the same metrics registry.
    let app = build_app(cfg).await;

    // Drive the pre-content failover off `primary`.
    let req = Request::post("/v1/chat/completions")
        .header("content-type", "application/json")
        .body(streaming_request_body())
        .unwrap();
    let (status, _raw) = send(app.clone(), req).await;
    assert_eq!(status, StatusCode::OK);

    // Scrape metrics and assert the primary's failure was recorded (Req 4.5).
    let metrics_req = Request::get("/metrics").body(Body::empty()).unwrap();
    let (metrics_status, metrics_body) = send(app, metrics_req).await;
    assert_eq!(metrics_status, StatusCode::OK);
    let text = std::str::from_utf8(&metrics_body).unwrap();

    assert!(
        provider_failures_metric(text, "primary") >= 1,
        "circuit-breaker/metrics must record a failure for the disconnected primary provider (Req 4.5)\nmetrics:\n{}",
        text
    );
}

// ---------------------------------------------------------------------------
// 13b. Buffered reasoning progress relay (buffered-reasoning-progress-relay
// task 9, spec scenarios a-d).
//
// These tests force the buffered path (`passthrough_enabled: false`) and give
// the mock provider a real SSE body whose reasoning deltas precede content
// deltas. The router tap (`read_upstream_body`) relays reasoning-only frames
// while the body streams; the handler's select loop forwards them to the
// client before the final re-chunk (which suppresses reasoning when frames
// were relayed). Frame ordering is asserted via string indices only — the
// relay happens during the buffered read while content only ever appears in
// the final replay, so ordering is deterministic without inter-event delays.
//
// Scenario (e) — re-running `abandoned_stream_writes_client_disconnect_outcome_row`
// and the rate-token guards — lives in the guard binaries:
// `tests/request_outcome_logging.rs` and `src/router/router.rs`
// (`single_request_consumes_exactly_one_provider_rate_token`); they run
// unchanged in the full-binary sweep at the end of this task.
// ---------------------------------------------------------------------------

/// Build a single SSE data line carrying a `reasoning`-only delta with the
/// mock provider's envelope, mirroring how `sse_content_chunk` builds content
/// deltas. `data: [DONE]` and content/tool/usage frames are never relayed.
fn sse_reasoning_chunk(reasoning: &str) -> String {
    let value = serde_json::json!({
        "id": MOCK_PROVIDER_ID,
        "object": "chat.completion.chunk",
        "created": 1700000000_i64,
        "model": "gpt-4",
        "choices": [{
            "index": 0,
            "delta": { "reasoning": reasoning },
            "finish_reason": null
        }]
    });
    format!("data: {}\n\n", value)
}

/// Index of the first chunk carrying a non-null `delta.reasoning` (relayed or
/// replayed) inside the parsed chunk list, or None.
fn first_reasoning_index_in(chunks: &[serde_json::Value]) -> Option<usize> {
    chunks.iter().position(|c| {
        c.pointer("/choices/0/delta/reasoning")
            .and_then(serde_json::Value::as_str)
            .is_some()
    })
}

/// Index of the first chunk carrying a non-null `delta.content`, or None.
fn first_content_index_in(chunks: &[serde_json::Value]) -> Option<usize> {
    chunks.iter().position(|c| {
        c.pointer("/choices/0/delta/content")
            .and_then(serde_json::Value::as_str)
            .is_some()
    })
}

/// Build a buffered-relay streaming config: early event on, buffered path
/// forced (`passthrough_enabled: false`), relay optionally enabled.
fn relay_test_config(mock_uri: &str, relay_reasoning_progress: bool) -> Config {
    let mut cfg = test_config();
    cfg.providers[0].base_url = Some(mock_uri.to_string());
    cfg.streaming = Some(StreamingConfig {
        emit_early_event: true,
        passthrough_enabled: false,
        relay_reasoning_progress,
        ..StreamingConfig::default()
    });
    cfg
}

/// Variant of [`relay_test_config`] with the early event DISABLED (site 2:
/// the buffered fallback path where `route_request` runs first and the FIRST
/// relayed reasoning frame carries the `role` marker, Req 2.5).
fn relay_no_early_test_config(mock_uri: &str, relay_reasoning_progress: bool) -> Config {
    let mut cfg = test_config();
    cfg.providers[0].base_url = Some(mock_uri.to_string());
    cfg.streaming = Some(StreamingConfig {
        emit_early_event: false,
        passthrough_enabled: false,
        relay_reasoning_progress,
        ..StreamingConfig::default()
    });
    cfg
}

/// Post a streaming request and return (status, raw body bytes).
async fn send_streaming(app: axum::Router) -> (StatusCode, Vec<u8>) {
    let req = Request::post("/v1/chat/completions")
        .header("content-type", "application/json")
        .body(streaming_request_body())
        .unwrap();
    send(app, req).await
}

/// The reasoning-then-content SSE body shared by scenarios (a)/(b): two
/// reasoning deltas, two content deltas, then `[DONE]`.
fn reasoning_then_content_sse_body() -> String {
    format!(
        "{}{}{}{}{}",
        sse_reasoning_chunk("thinking about the question"),
        sse_reasoning_chunk("weighing approaches"),
        sse_content_chunk("Hello from ", None),
        sse_content_chunk("mock provider!", Some("stop")),
        "data: [DONE]\n\n",
    )
}

/// A minimal PACED SSE upstream: binds an ephemeral TCP port, accepts one
/// connection, and writes the given SSE frames with a small delay between
/// them, then closes.
///
/// wiremock's `ResponseTemplate` writes the whole body in one flush, so it
/// cannot reproduce the spec's "reasoning deltas (spread with delays), then
/// content" shape: with an instant body the buffered read finishes before
/// the handler's select loop is ever polled, and `select!` may complete the
/// routing arm first, dropping queued relay frames (a race — not a
/// wall-clock assertion; ordering is still asserted purely via indices).
/// This is a mock *upstream* only, in the same role as wiremock; the
/// gateway under test is still driven exclusively via tower oneshot.
async fn start_paced_sse_mock(frames: Vec<String>, delay_ms: u64) -> String {
    start_paced_sse_mock_raw(
        frames.into_iter().map(String::into_bytes).collect(),
        delay_ms,
    )
    .await
}

/// Raw-bytes variant of [`start_paced_sse_mock`]: each element of `writes` is
/// emitted verbatim with no UTF-8 boundary rounding, so tests can split a
/// multi-byte character across two writes and assert the relay tap (which
/// scans raw bytes and decodes only completed lines) never bakes U+FFFD into
/// relayed text.
async fn start_paced_sse_mock_raw(writes: Vec<Vec<u8>>, delay_ms: u64) -> String {
    use tokio::io::{AsyncReadExt, AsyncWriteExt};
    let listener = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
    let addr = listener.local_addr().unwrap();
    tokio::spawn(async move {
        // Bounded accept: without it a gateway that never dials the mock would
        // hang the test silently until the harness timeout.
        let (mut socket, _) =
            match tokio::time::timeout(std::time::Duration::from_secs(10), listener.accept()).await
            {
                Ok(accepted) => accepted.expect("paced SSE mock: accept() failed"),
                Err(_) => panic!("paced SSE mock: the gateway never connected within 10s"),
            };
        // Consume the request head (best-effort; the response is written
        // regardless — the gateway's client does not need it read first).
        let mut scratch = [0u8; 8192];
        let _ = tokio::time::timeout(std::time::Duration::from_secs(5), socket.read(&mut scratch))
            .await;
        let head =
            "HTTP/1.1 200 OK\r\ncontent-type: text/event-stream\r\nconnection: close\r\n\r\n";
        socket.write_all(head.as_bytes()).await.unwrap();
        for chunk in writes {
            tokio::time::sleep(std::time::Duration::from_millis(delay_ms)).await;
            socket.write_all(&chunk).await.unwrap();
        }
        socket.shutdown().await.unwrap();
    });
    format!("http://{addr}")
}

/// (a) Req 1.1/1.2/1.3/1.5 + 4.4: reasoning frames arrive before any content
/// frame; relayed frames share the early event's response id and carry no
/// `role`; the final replay is complete (full content), reasoning-free, and
/// `[DONE]` is the last line. The upstream emits reasoning deltas with small
/// delays, then content, then [DONE] (spec scenario a); ordering is asserted
/// via string/vec indices only — no wall-clock assertions.
#[tokio::test]
async fn relay_reasoning_frames_arrive_before_final_content() {
    let uri = start_paced_sse_mock(
        vec![
            sse_reasoning_chunk("thinking about the question"),
            sse_reasoning_chunk("weighing approaches"),
            sse_content_chunk("Hello from ", None),
            sse_content_chunk("mock provider!", Some("stop")),
            "data: [DONE]\n\n".to_string(),
        ],
        15,
    )
    .await;
    let app = build_app(relay_test_config(&uri, true)).await;

    let (status, raw) = send_streaming(app).await;
    assert_eq!(status, StatusCode::OK);

    let data_lines = raw_sse_data_lines(&raw);
    assert!(
        data_lines.last().map(String::as_str) == Some("[DONE]"),
        "[DONE] must be the last data line (got {:?})",
        data_lines.last()
    );

    let chunks = parse_sse_chunks(&raw);

    // Relayed reasoning frames arrive BEFORE any content frame (Req 1.1).
    let first_reasoning = first_reasoning_index_in(&chunks)
        .expect("at least one reasoning frame must be relayed mid-stream");
    let first_content =
        first_content_index_in(&chunks).expect("final content must be present in the replay");
    assert!(
        first_reasoning < first_content,
        "relayed reasoning (index {first_reasoning}) must precede the first content frame (index {first_content})"
    );

    // All chunks (early event, relayed frames, replay) share one response id
    // (Req 1.5) — and it is the fresh early-event id, not the provider's.
    let early_id = chunks[0]["id"].as_str().unwrap();
    assert_ne!(early_id, MOCK_PROVIDER_ID);
    for (index, chunk) in chunks.iter().enumerate() {
        assert_eq!(
            chunk["id"].as_str(),
            Some(early_id),
            "chunk at index {index} must share the early event id"
        );
    }

    // Relayed frames carry no `role`; exactly ONE role marker overall (the
    // early event) — the replay's role chunk is suppressed by the early event.
    let role_count = chunks
        .iter()
        .filter(|c| c.pointer("/choices/0/delta/role").is_some())
        .count();
    assert_eq!(
        role_count, 1,
        "single role marker overall (the early event)"
    );
    for (index, chunk) in chunks.iter().enumerate().skip(1) {
        assert!(
            chunk.pointer("/choices/0/delta/role").is_none(),
            "relayed/replayed chunk at index {index} must not carry a role"
        );
    }

    // No reasoning frame after the first content frame: everything at or
    // after `first_content` is the final replay, and the replay suppresses
    // reasoning when frames were relayed (Req 4.4 no double-emit).
    for (index, chunk) in chunks.iter().enumerate() {
        if index >= first_content {
            assert!(
                first_reasoning_index_in(std::slice::from_ref(chunk)).is_none(),
                "no reasoning frame allowed after the first content frame (index {index})"
            );
        }
    }

    // The final content is complete: both content deltas reassembled by the
    // buffered pipeline into the replay chunk.
    let replay_content = chunks
        .iter()
        .filter_map(|c| {
            c.pointer("/choices/0/delta/content")
                .and_then(|v| v.as_str())
        })
        .collect::<String>();
    assert_eq!(
        replay_content, "Hello from mock provider!",
        "final content must be the complete reassembled response"
    );
}

/// (b) Req 2.1: `relay_reasoning_progress: false` restores the legacy stream —
/// no mid-stream reasoning frames; the final response still carries the full
/// reassembled reasoning (the whole reasoning stays in the replay chunk).
#[tokio::test]
async fn relay_disabled_keeps_reasoning_only_in_final_response() {
    let mock = start_sse_streaming_mock(reasoning_then_content_sse_body()).await;
    let app = build_app(relay_test_config(&mock.uri(), false)).await;

    let (status, raw) = send_streaming(app).await;
    assert_eq!(status, StatusCode::OK);

    let data_lines = raw_sse_data_lines(&raw);
    assert!(data_lines.last().map(String::as_str) == Some("[DONE]"));

    let chunks = parse_sse_chunks(&raw);

    // No reasoning frame BEFORE the first content frame: with the relay off,
    // reasoning only ever appears inside the final replay (legacy behavior).
    let first_content =
        first_content_index_in(&chunks).expect("final content must be present in the replay");
    for (index, chunk) in chunks.iter().enumerate() {
        if index < first_content {
            assert!(
                first_reasoning_index_in(std::slice::from_ref(chunk)).is_none(),
                "no mid-stream reasoning frame allowed with the relay disabled (index {index})"
            );
        }
    }

    // The final replay still carries the full reassembled reasoning
    // (both deltas concatenated), exactly once. The reassembled response
    // stores reasoning under the `reasoning_content` carrier and the replay
    // preserves the field name, so accept either spelling.
    let reasoning_frames: Vec<&str> = chunks
        .iter()
        .filter_map(|c| {
            ["reasoning", "reasoning_content"]
                .iter()
                .find_map(|field| c.pointer(&format!("/choices/0/delta/{field}")))
                .and_then(serde_json::Value::as_str)
        })
        .collect();
    assert_eq!(
        reasoning_frames,
        vec!["thinking about the questionweighing approaches"],
        "final response carries the full reasoning (legacy behavior)"
    );

    // Content is complete and [DONE] terminated.
    let replay_content: String = chunks
        .iter()
        .filter_map(|c| {
            c.pointer("/choices/0/delta/content")
                .and_then(|v| v.as_str())
        })
        .collect();
    assert_eq!(replay_content, "Hello from mock provider!");
}

/// Shared config for the two-provider relay scenarios: `primary`
/// (priority 100, tried first) and `backup` (priority 200), single `gpt-4`
/// model group, buffered path forced via `passthrough_enabled: false` and
/// the reasoning relay enabled. Distinct reasoning/content strings per
/// provider identify whose bytes reached the client.
fn relay_failover_config(primary_uri: &str, backup_uri: &str) -> Config {
    let mut cfg = test_config();
    let template = cfg.providers[0].clone();
    let primary = Provider {
        name: "primary".to_string(),
        base_url: Some(primary_uri.to_string()),
        memory: None,
        ..template.clone()
    };
    let backup = Provider {
        name: "backup".to_string(),
        base_url: Some(backup_uri.to_string()),
        memory: None,
        ..template
    };
    cfg.providers = vec![primary, backup];
    cfg.model_groups = vec![ModelGroup {
        name: "gpt-4-group".to_string(),
        version_fallback_enabled: false,
        compression: None,
        memory: None,
        structured_output: None,
        models: vec![
            ProviderModel {
                cache_support: None,
                cache_min_tokens: None,
                cost_per_million_cache_read_input_tokens: None,
                cost_per_million_cache_creation_input_tokens: None,
                provider: "primary".to_string(),
                model: "gpt-4".to_string(),
                cost_per_million_input_tokens: 30.0,
                cost_per_million_output_tokens: 60.0,
                priority: 100,
                structured_output_passthrough: None,
                tier: None,
                context_window: 0,
                specializations: vec![],
                cost_per_million_reasoning_tokens: None,
                reasoning_family: None,
                reasoning_parameter: None,
            },
            ProviderModel {
                cache_support: None,
                cache_min_tokens: None,
                cost_per_million_cache_read_input_tokens: None,
                cost_per_million_cache_creation_input_tokens: None,
                provider: "backup".to_string(),
                model: "gpt-4".to_string(),
                cost_per_million_input_tokens: 30.0,
                cost_per_million_output_tokens: 60.0,
                priority: 200,
                structured_output_passthrough: None,
                tier: None,
                context_window: 0,
                specializations: vec![],
                cost_per_million_reasoning_tokens: None,
                reasoning_family: None,
                reasoning_parameter: None,
            },
        ],
    }];
    cfg.streaming = Some(StreamingConfig {
        emit_early_event: true,
        passthrough_enabled: false,
        relay_reasoning_progress: true,
        ..StreamingConfig::default()
    });
    cfg
}

/// (c) Req 4.1/4.3: the primary emits reasoning then truncates pre-content
/// (no `[DONE]`, no finish_reason) — the buffered dispatch treats that as a
/// retryable failure and fails over. The client still saw the primary's
/// reasoning relayed live, then receives exactly ONE final response with the
/// backup's content and a terminating `[DONE]`.
#[tokio::test]
async fn relay_failover_after_truncated_primary_relays_reasoning_from_first() {
    // Primary: reasoning deltas (paced), then the stream is cut short (no
    // [DONE], no finish_reason) — the SSE_TRUNCATED_ERROR failover trigger.
    // Pacing makes the reasoning relay deterministic: frames are drained by
    // the select loop while the read is still open.
    let primary_uri = start_paced_sse_mock(
        vec![
            sse_reasoning_chunk("primary pondering"),
            sse_reasoning_chunk("primary about to answer"),
            // truncation: connection closes without [DONE]/finish_reason
        ],
        15,
    )
    .await;

    // Backup: full reasoning + content + [DONE].
    let backup_body = format!(
        "{}{}{}",
        sse_reasoning_chunk("backup recovering"),
        sse_content_chunk("BACKUP_ANSWER", Some("stop")),
        "data: [DONE]\n\n",
    );
    let backup = start_sse_streaming_mock(backup_body).await;

    let app = build_app(relay_failover_config(&primary_uri, &backup.uri())).await;

    let (status, raw) = send_streaming(app).await;
    assert_eq!(status, StatusCode::OK);

    let data_lines = raw_sse_data_lines(&raw);
    assert!(
        data_lines.last().map(String::as_str) == Some("[DONE]"),
        "[DONE] must be last (got {:?})",
        data_lines.last()
    );
    // One final response: exactly one [DONE], one finish_reason-bearing chunk.
    let done_count = data_lines.iter().filter(|l| l.as_str() == "[DONE]").count();
    assert_eq!(done_count, 1, "exactly one [DONE] sentinel");
    let finish_count = data_lines
        .iter()
        .filter(|l| l.contains("\"finish_reason\":\"stop\""))
        .count();
    assert_eq!(finish_count, 1, "exactly one final finish_reason chunk");

    let chunks = parse_sse_chunks(&raw);

    // Reasoning from the FIRST provider was relayed before any content: the
    // primary's reasoning frames must appear mid-stream.
    let reasoning_texts: Vec<&str> = chunks
        .iter()
        .filter_map(|c| {
            c.pointer("/choices/0/delta/reasoning")
                .and_then(serde_json::Value::as_str)
        })
        .collect();
    assert!(
        reasoning_texts.iter().any(|r| *r == "primary pondering"),
        "primary's reasoning must have been relayed before failover (got {reasoning_texts:?})"
    );

    // Content came from the SECOND provider.
    let content: String = chunks
        .iter()
        .filter_map(|c| {
            c.pointer("/choices/0/delta/content")
                .and_then(|v| v.as_str())
        })
        .collect();
    assert_eq!(
        content, "BACKUP_ANSWER",
        "client must receive the backup provider's content after failover"
    );

    // Ordering guard: relayed reasoning before the final content.
    let first_reasoning =
        first_reasoning_index_in(&chunks).expect("relayed reasoning frames present");
    let first_content = first_content_index_in(&chunks).expect("final content present");
    assert!(first_reasoning < first_content);
}

/// Find the (single) error outcome row in the isolated request log: the row
/// with path `/v1/chat/completions` (not an `#attempt` row) that carries a
/// non-null error_class. Returns (error_class, reasoning_relayed_frames).
/// `/dashboard/logs` does not expose `reasoning_relayed_frames`, so this
/// reads the SQLite column directly, mirroring the handlers.rs unit tests.
fn find_error_outcome_row(cfg: &Config) -> Option<(Option<String>, Option<i64>)> {
    let conn = rusqlite::Connection::open(&cfg.logging.database_path).ok()?;
    let mut stmt = conn
        .prepare(
            "SELECT error_class, reasoning_relayed_frames FROM requests
             WHERE path = '/v1/chat/completions' AND error_class IS NOT NULL",
        )
        .ok()?;
    let rows = stmt
        .query_map([], |row| {
            let error_class: Option<String> = row.get(0)?;
            let relayed: Option<i64> = row.get(1)?;
            Ok((error_class, relayed))
        })
        .ok()?
        .filter_map(std::result::Result::ok)
        .collect::<Vec<_>>();
    rows.into_iter().next()
}

/// Poll the isolated request log (bounded) until an error outcome row with a
/// non-null error_class appears; the writer thread persists asynchronously.
async fn wait_for_error_outcome_row(cfg: &Config) -> Option<(Option<String>, Option<i64>)> {
    let deadline = std::time::Instant::now() + std::time::Duration::from_secs(15);
    loop {
        if let Some(row) = find_error_outcome_row(cfg) {
            return Some(row);
        }
        if std::time::Instant::now() >= deadline {
            return find_error_outcome_row(cfg);
        }
        tokio::time::sleep(std::time::Duration::from_millis(50)).await;
    }
}

/// (d) Req 4.2/5.1: a mid-stream failure AFTER reasoning frames were relayed
/// (both providers emit a mid-stream `data: {"error":...}` frame) yields the
/// graceful SSE error event + `[DONE]`; the request log row carries the
/// `error_class` AND `reasoning_relayed_frames > 0`.
#[tokio::test]
async fn relay_midstream_failure_emits_error_and_logs_relayed_frames() {
    // Both providers: reasoning first (paced so the relay deterministically
    // drains), then an in-band error frame — the buffered dispatch
    // retries/fails over and eventually errors out, AFTER the client
    // already saw relayed reasoning.
    let mk_body = || {
        vec![
            sse_reasoning_chunk("doomed deliberation"),
            sse_error_frame(),
        ]
    };
    let primary_uri = start_paced_sse_mock(mk_body(), 15).await;
    let backup = start_sse_streaming_mock(mk_body().join("")).await;

    let mut cfg = relay_failover_config(&primary_uri, &backup.uri());
    // build_app isolates its own copy; isolate here first so this test can
    // query the very same SQLite file for the outcome row afterwards.
    common::isolate_databases(&mut cfg);
    let server = ai_gateway::gateway::GatewayServer::new(cfg.clone(), None)
        .await
        .unwrap();
    let app = server.build_router();

    let (status, raw) = send_streaming(app).await;
    assert_eq!(
        status,
        StatusCode::OK,
        "early event forces a 200 SSE stream"
    );

    let data_lines = raw_sse_data_lines(&raw);
    assert!(
        data_lines.last().map(String::as_str) == Some("[DONE]"),
        "graceful failure must still terminate with [DONE] (got {:?})",
        data_lines.last()
    );

    let chunks = parse_sse_chunks(&raw);
    // Reasoning WAS relayed before the failure surfaced.
    assert!(
        first_reasoning_index_in(&chunks).is_some(),
        "reasoning frames must be relayed before the mid-stream failure"
    );

    // The frame before [DONE] is the graceful aggregated error event.
    let error_line = &data_lines[data_lines.len() - 2];
    let error_json: serde_json::Value =
        serde_json::from_str(error_line).expect("error frame before [DONE] must be valid JSON");
    assert!(error_json["error"].is_object());
    assert!(
        error_json["error"]["message"]
            .as_str()
            .is_some_and(|m| !m.is_empty()),
        "error frame must carry a message"
    );

    // The isolated log row carries error_class AND relayed-frame count > 0.
    let (error_class, relayed_frames) = wait_for_error_outcome_row(&cfg)
        .await
        .expect("an error outcome row must be logged");
    assert!(
        error_class.as_deref().is_some_and(|c| !c.is_empty()),
        "log row must carry an error_class (got {error_class:?})"
    );
    assert!(
        relayed_frames.unwrap_or(0) > 0,
        "log row must carry reasoning_relayed_frames > 0 (got {relayed_frames:?})"
    );
}

// ---------------------------------------------------------------------------
// 13c. Buffered reasoning progress relay — remediation wave B coverage.
//
// Wave-A fixes under test: select loops drain queued frames; the site-2
// (early-event off) path returns a proper HTTP error status when routing
// fails before any relayed frame; site 3 (`stream_buffered_with_post_call`)
// spawns routing + relays on the Buffered arm too; the relay tap scans raw
// bytes so a multi-byte character split across upstream writes survives.
// ---------------------------------------------------------------------------

use ai_gateway::guardrail::{
    FailurePolicy, GuardrailConfig, GuardrailProviderConfig, GuardrailProviderType,
    InstructionInsertionMode, PipelineConfig, PolicyAction, ProviderSettings, RegexPatternConfig,
    RegexRuleMode, StageConfig, StagePhase,
};

/// A benign post-call guardrail: a regex provider whose pattern never matches
/// the mock's response and whose action is `allow`, so `run_post_call`
/// proceeds untouched. The stage's mere existence forces the buffered
/// `stream_buffered_with_post_call` path (relay site 3).
fn benign_post_call_guardrails() -> GuardrailConfig {
    GuardrailConfig {
        providers: vec![GuardrailProviderConfig {
            name: "regex_benign".to_string(),
            provider_type: GuardrailProviderType::Regex,
            failure_policy: FailurePolicy::FailOpen,
            timeout_seconds: 5,
            settings: ProviderSettings {
                patterns: vec![RegexPatternConfig {
                    name: "never_matches".to_string(),
                    regex: r"NEVERMATCHES\d+".to_string(),
                    entity: "BENIGN".to_string(),
                    mode: RegexRuleMode::Deny,
                }],
                ..Default::default()
            },
        }],
        pipelines: vec![PipelineConfig {
            name: "post_benign".to_string(),
            stages: vec![StageConfig {
                name: "benign_scan".to_string(),
                provider: "regex_benign".to_string(),
                phase: StagePhase::PostCall,
                action: PolicyAction::Allow,
            }],
            redaction_notice_instruction: None,
            instruction_insertion_mode: InstructionInsertionMode::default(),
            failover_on_refusal: false,
            refusal_phrase_list: None,
            tool_result: ai_gateway::guardrail::config::ToolResultPhaseConfig::default(),
        }],
        global_default_pipeline: Some("post_benign".to_string()),
        bindings: Default::default(),
        ..Default::default()
    }
}

/// Site 2 (early event off), success shape: the FIRST relayed reasoning frame
/// carries the single `role: "assistant"` marker (Req 2.5); relayed frames and
/// final replay share one fresh response id; reasoning precedes content; the
/// final replay suppresses relayed reasoning (no double-emit); `[DONE]` last.
#[tokio::test]
async fn relay_first_frame_carries_role_when_early_event_disabled() {
    let uri = start_paced_sse_mock(
        vec![
            sse_reasoning_chunk("thinking about the question"),
            sse_reasoning_chunk("weighing approaches"),
            sse_content_chunk("Hello from ", None),
            sse_content_chunk("mock provider!", Some("stop")),
            "data: [DONE]\n\n".to_string(),
        ],
        15,
    )
    .await;
    let app = build_app(relay_no_early_test_config(&uri, true)).await;

    let (status, raw) = send_streaming(app).await;
    assert_eq!(status, StatusCode::OK);

    let data_lines = raw_sse_data_lines(&raw);
    assert!(
        data_lines.last().map(String::as_str) == Some("[DONE]"),
        "[DONE] must be the last data line (got {:?})",
        data_lines.last()
    );

    let chunks = parse_sse_chunks(&raw);
    assert!(
        !chunks.is_empty(),
        "relayed reasoning frames must be present"
    );

    // Exactly one role marker across the whole stream — and it sits on the
    // FIRST relayed reasoning frame (Req 2.5).
    let role_count = chunks
        .iter()
        .filter(|c| c.pointer("/choices/0/delta/role").is_some())
        .count();
    assert_eq!(
        role_count, 1,
        "single role marker overall (the first relayed frame)"
    );
    assert_eq!(
        chunks[0]
            .pointer("/choices/0/delta/role")
            .and_then(|v| v.as_str()),
        Some("assistant"),
        "the first chunk must be the role-bearing relayed reasoning frame"
    );
    assert!(
        first_reasoning_index_in(std::slice::from_ref(&chunks[0])).is_some(),
        "the role marker must ride a reasoning frame, not a content frame"
    );

    // Shared id across relayed frames AND final chunks — the fresh relay id,
    // not the provider's own envelope id.
    let shared_id = chunks[0]["id"].as_str().unwrap();
    assert_ne!(
        shared_id, MOCK_PROVIDER_ID,
        "relay frames use a fresh id, not the provider's"
    );
    for (index, chunk) in chunks.iter().enumerate() {
        assert_eq!(
            chunk["id"].as_str(),
            Some(shared_id),
            "chunk at index {index} must share the relay stream id"
        );
    }

    // Reasoning relayed before content.
    let first_reasoning =
        first_reasoning_index_in(&chunks).expect("at least one reasoning frame must be relayed");
    let first_content =
        first_content_index_in(&chunks).expect("final content must be present in the replay");
    assert!(
        first_reasoning < first_content,
        "relayed reasoning (index {first_reasoning}) must precede the first content frame (index {first_content})"
    );

    // Both reasoning deltas were relayed live.
    let reasoning_texts: Vec<&str> = chunks
        .iter()
        .filter_map(|c| {
            c.pointer("/choices/0/delta/reasoning")
                .and_then(serde_json::Value::as_str)
        })
        .collect();
    assert!(
        reasoning_texts.contains(&"thinking about the question")
            && reasoning_texts.contains(&"weighing approaches"),
        "both reasoning deltas must be relayed (got {reasoning_texts:?})"
    );

    // The final replay suppresses relayed reasoning (Req 4.4 no double-emit).
    for (index, chunk) in chunks.iter().enumerate() {
        if index >= first_content {
            assert!(
                first_reasoning_index_in(std::slice::from_ref(chunk)).is_none(),
                "no reasoning frame allowed in the final replay (index {index})"
            );
        }
    }

    // Final content complete.
    let replay_content: String = chunks
        .iter()
        .filter_map(|c| {
            c.pointer("/choices/0/delta/content")
                .and_then(|v| v.as_str())
        })
        .collect();
    assert_eq!(
        replay_content, "Hello from mock provider!",
        "final content must be the complete reassembled response"
    );
}

/// Site 2, pre-stream error shape: with the early event off and the relay on,
/// routing that fails BEFORE any frame was relayed must return the proper
/// HTTP error status (same mapping as the relay-disabled path: the aggregated
/// AllProvidersFailed error → 502 JSON), never a 200 SSE stream.
#[tokio::test]
async fn relay_routing_error_before_frames_returns_http_error_status() {
    // Upstream answers 500 immediately — no reasoning emitted first.
    let server = MockServer::start().await;
    Mock::given(wm_method("POST"))
        .and(wm_path("/v1/chat/completions"))
        .respond_with(
            ResponseTemplate::new(500)
                .set_body_json(serde_json::json!({"error": "upstream exploded"})),
        )
        .mount(&server)
        .await;

    let mut cfg = relay_no_early_test_config(&server.uri(), true);
    // No in-provider retries/backoff sleeps: surface the aggregated error
    // immediately.
    let mut retry = RetryConfig::default();
    retry.max_retries_per_provider = 0;
    retry.backoff_sequence_seconds = vec![];
    cfg.retry = retry;

    let app = build_app(cfg).await;
    let (status, raw) = send_streaming(app).await;

    assert_eq!(
        status,
        StatusCode::BAD_GATEWAY,
        "pre-stream routing failure must map to the relay-disabled error status (502 AllProvidersFailed)"
    );
    let text = std::str::from_utf8(&raw).unwrap();
    assert!(
        !text.contains("data: "),
        "the error must be a plain JSON response, not a 200 SSE stream (got: {text:?})"
    );
    let json: serde_json::Value = serde_json::from_slice(&raw).unwrap();
    assert!(
        json["error"].is_object(),
        "error response must carry an error object"
    );
    assert_eq!(
        json["error"]["type"], "all_providers_failed",
        "same aggregated-error mapping as the relay-disabled path"
    );
}

/// Site 3 (`stream_buffered_with_post_call`, Buffered arm): a bound post-call
/// guardrail stage plus `passthrough_enabled: false` makes
/// `route_request_streaming` return Buffered; reasoning frames must still be
/// relayed live BEFORE the final assembled content, the final response must
/// be intact, and the stream ends with exactly one `[DONE]`.
#[tokio::test]
async fn relay_site3_buffered_arm_relays_reasoning() {
    let uri = start_paced_sse_mock(
        vec![
            sse_reasoning_chunk("site three deliberation"),
            sse_content_chunk("Hello from ", None),
            sse_content_chunk("mock provider!", Some("stop")),
            "data: [DONE]\n\n".to_string(),
        ],
        15,
    )
    .await;
    let mut cfg = relay_test_config(&uri, true);
    cfg.guardrails = Some(benign_post_call_guardrails());
    let app = build_app(cfg).await;

    let (status, raw) = send_streaming(app).await;
    assert_eq!(status, StatusCode::OK);

    let data_lines = raw_sse_data_lines(&raw);
    let done_count = data_lines.iter().filter(|l| l.as_str() == "[DONE]").count();
    assert_eq!(done_count, 1, "exactly one [DONE] sentinel");
    assert!(
        data_lines.last().map(String::as_str) == Some("[DONE]"),
        "[DONE] must be last (got {:?})",
        data_lines.last()
    );

    let chunks = parse_sse_chunks(&raw);

    // Reasoning relayed live BEFORE the final assembled content.
    let first_reasoning = first_reasoning_index_in(&chunks)
        .expect("site 3 must relay reasoning live while buffering for post-call");
    let first_content =
        first_content_index_in(&chunks).expect("final assembled content must be present");
    assert!(
        first_reasoning < first_content,
        "relayed reasoning (index {first_reasoning}) must precede the assembled content (index {first_content})"
    );
    assert_eq!(
        chunks[first_reasoning]
            .pointer("/choices/0/delta/reasoning")
            .and_then(serde_json::Value::as_str),
        Some("site three deliberation"),
        "the live-relayed reasoning text must be intact"
    );

    // Final assembled response intact after the benign post-call pass.
    let replay_content: String = chunks
        .iter()
        .filter_map(|c| {
            c.pointer("/choices/0/delta/content")
                .and_then(|v| v.as_str())
        })
        .collect();
    assert_eq!(
        replay_content, "Hello from mock provider!",
        "final assembled content must be complete"
    );
}

/// The relay tap must scan RAW BYTES: a reasoning delta containing multi-byte
/// UTF-8 ("思考") whose SSE line is split MID-CHARACTER across upstream TCP
/// writes must be relayed with the character intact (no U+FFFD) and the final
/// assembled content must be complete.
#[tokio::test]
async fn relay_multibyte_reasoning_split_across_chunks_survives() {
    let frame = sse_reasoning_chunk("思考");
    let bytes = frame.as_bytes();
    let char_start = frame.find("思考").expect("multibyte text present");
    // 思 = E6 80 9D, 考 = E8 80 83 (3 bytes each). Split inside both chars:
    // the bytes at split1/split2 must be UTF-8 continuation bytes (10xxxxxx).
    let split1 = char_start + 1;
    let split2 = char_start + 4;
    assert_eq!(bytes[split1] & 0xC0, 0x80, "split1 must land mid-character");
    assert_eq!(bytes[split2] & 0xC0, 0x80, "split2 must land mid-character");

    let uri = start_paced_sse_mock_raw(
        vec![
            bytes[..split1].to_vec(),
            bytes[split1..split2].to_vec(),
            bytes[split2..].to_vec(),
            sse_content_chunk("final ", None).into_bytes(),
            sse_content_chunk("answer", Some("stop")).into_bytes(),
            b"data: [DONE]\n\n".to_vec(),
        ],
        15,
    )
    .await;
    let app = build_app(relay_test_config(&uri, true)).await;

    let (status, raw) = send_streaming(app).await;
    assert_eq!(status, StatusCode::OK);

    let text = std::str::from_utf8(&raw).expect("client stream must be valid UTF-8");
    assert!(
        !text.contains('\u{FFFD}'),
        "no U+FFFD replacement character may leak into the client stream"
    );

    let chunks = parse_sse_chunks(&raw);
    let reasoning_texts: Vec<&str> = chunks
        .iter()
        .filter_map(|c| {
            c.pointer("/choices/0/delta/reasoning")
                .and_then(serde_json::Value::as_str)
        })
        .collect();
    assert!(
        reasoning_texts.contains(&"思考"),
        "the relayed reasoning frame must carry the intact multi-byte character (got {reasoning_texts:?})"
    );

    let replay_content: String = chunks
        .iter()
        .filter_map(|c| {
            c.pointer("/choices/0/delta/content")
                .and_then(|v| v.as_str())
        })
        .collect();
    assert_eq!(
        replay_content, "final answer",
        "final assembled content must be complete"
    );

    let data_lines = raw_sse_data_lines(&raw);
    assert!(data_lines.last().map(String::as_str) == Some("[DONE]"));
}

// ---------------------------------------------------------------------------
// 14. Truncation detection and retry (streaming-reliability task 7.3)
//
// These tests exercise the non-streaming buffered path
// (route_request -> route_with_failover) where the gateway inspects
// finish_reason / usage.completion_tokens against the request's max_tokens to
// decide whether a `finish_reason: "length"` response was a real truncation
// (failover) or a legitimate max_tokens limit (no failover). The mock returns a
// full chat.completion JSON body and the assertions read the final JSON
// response (status + choices[0].message.content + choices[0].finish_reason +
// usage.completion_tokens), distinguishing providers by their content text.
// ---------------------------------------------------------------------------

/// Start a mock OpenAI-compatible provider returning a single chat.completion
/// with the given content, finish_reason and completion_tokens. Distinct
/// `content` per provider lets tests identify which response the client got.
async fn start_truncation_mock(
    content: &str,
    finish_reason: &str,
    completion_tokens: u64,
) -> MockServer {
    let server = MockServer::start().await;
    let body = serde_json::json!({
        "id": MOCK_PROVIDER_ID,
        "object": "chat.completion",
        "created": 1700000000_i64,
        "model": "gpt-4",
        "choices": [{
            "index": 0,
            "message": { "role": "assistant", "content": content },
            "finish_reason": finish_reason
        }],
        "usage": {
            "prompt_tokens": 10,
            "completion_tokens": completion_tokens,
            "total_tokens": 10 + completion_tokens
        }
    });
    Mock::given(wm_method("POST"))
        .and(wm_path("/v1/chat/completions"))
        .respond_with(ResponseTemplate::new(200).set_body_json(body))
        .mount(&server)
        .await;
    server
}

/// Build a two-provider failover config (`primary` priority 100 tried first,
/// `backup` priority 200) in one `gpt-4` model group, both plain `openai`
/// providers, with the given `retry_on_truncation` streaming setting.
fn truncation_config(primary_uri: &str, backup_uri: &str, retry_on_truncation: bool) -> Config {
    let mut cfg = test_config();
    let template = cfg.providers[0].clone();
    let primary = Provider {
        name: "primary".to_string(),
        base_url: Some(primary_uri.to_string()),
        memory: None,
        ..template.clone()
    };
    let backup = Provider {
        name: "backup".to_string(),
        base_url: Some(backup_uri.to_string()),
        memory: None,
        ..template
    };
    cfg.providers = vec![primary, backup];
    cfg.model_groups = vec![ModelGroup {
        name: "gpt-4-group".to_string(),
        version_fallback_enabled: false,
        compression: None,
        memory: None,
        structured_output: None,
        models: vec![
            ProviderModel {
                cache_support: None,
                cache_min_tokens: None,
                cost_per_million_cache_read_input_tokens: None,
                cost_per_million_cache_creation_input_tokens: None,
                provider: "primary".to_string(),
                model: "gpt-4".to_string(),
                cost_per_million_input_tokens: 30.0,
                cost_per_million_output_tokens: 60.0,
                priority: 100,
                structured_output_passthrough: None,
                tier: None,
                context_window: 0,
                specializations: vec![],
                cost_per_million_reasoning_tokens: None,
                reasoning_family: None,
                reasoning_parameter: None,
            },
            ProviderModel {
                cache_support: None,
                cache_min_tokens: None,
                cost_per_million_cache_read_input_tokens: None,
                cost_per_million_cache_creation_input_tokens: None,
                provider: "backup".to_string(),
                model: "gpt-4".to_string(),
                cost_per_million_input_tokens: 30.0,
                cost_per_million_output_tokens: 60.0,
                priority: 200,
                structured_output_passthrough: None,
                tier: None,
                context_window: 0,
                specializations: vec![],
                cost_per_million_reasoning_tokens: None,
                reasoning_family: None,
                reasoning_parameter: None,
            },
        ],
    }];
    cfg.streaming = Some(StreamingConfig {
        retry_on_truncation,
        ..StreamingConfig::default()
    });
    cfg
}

/// Send a non-streaming chat completion with an explicit `max_tokens` and
/// return the parsed JSON response body.
async fn post_chat_nonstream(
    app: axum::Router,
    max_tokens: u64,
) -> (StatusCode, serde_json::Value) {
    let body_json = serde_json::json!({
        "model": "gpt-4",
        "messages": [{"role": "user", "content": "hello"}],
        "stream": false,
        "max_tokens": max_tokens
    });
    let req = Request::post("/v1/chat/completions")
        .header("content-type", "application/json")
        .body(Body::from(serde_json::to_string(&body_json).unwrap()))
        .unwrap();
    let (status, body) = send(app, req).await;
    let json: serde_json::Value = serde_json::from_slice(&body).unwrap();
    (status, json)
}

/// Req 6.1: a `finish_reason: "length"` response whose completion_tokens fall
/// well short of the requested max_tokens is treated as a truncation. The
/// gateway fails over to the backup provider, so the client receives the
/// backup's complete (`finish_reason: "stop"`) response, not the primary's
/// truncated one.
#[tokio::test]
async fn test_truncation_low_token_count_triggers_failover() {
    // Primary truncates at 5 tokens (max_tokens 1000) → suspicious truncation.
    let primary = start_truncation_mock("PRIMARY_TRUNCATED", "length", 5).await;
    // Backup returns a complete answer.
    let backup = start_truncation_mock("BACKUP_COMPLETE", "stop", 42).await;

    let app = build_app(truncation_config(&primary.uri(), &backup.uri(), true)).await;
    let (status, json) = post_chat_nonstream(app, 1000).await;

    assert_eq!(
        status,
        StatusCode::OK,
        "failover should yield a successful response"
    );
    assert_eq!(
        json["choices"][0]["message"]["content"], "BACKUP_COMPLETE",
        "client must receive the backup's complete response after failover (Req 6.1)"
    );
    assert_eq!(
        json["choices"][0]["finish_reason"], "stop",
        "the returned response is the backup's untruncated completion"
    );
}

/// Req 6.3: a `finish_reason: "length"` response whose completion_tokens reach
/// the requested max_tokens is a legitimate limit hit, NOT a truncation. The
/// gateway must NOT fail over — the client receives the primary's response
/// verbatim (content + finish_reason "length").
#[tokio::test]
async fn test_truncation_matching_max_tokens_does_not_failover() {
    // Primary legitimately hits the requested limit: completion_tokens == max_tokens.
    let primary = start_truncation_mock("PRIMARY_AT_LIMIT", "length", 1000).await;
    // Backup would return different content; it must NOT be consulted.
    let backup = start_truncation_mock("BACKUP_SHOULD_NOT_APPEAR", "stop", 10).await;

    let app = build_app(truncation_config(&primary.uri(), &backup.uri(), true)).await;
    let (status, json) = post_chat_nonstream(app, 1000).await;

    assert_eq!(status, StatusCode::OK);
    assert_eq!(
        json["choices"][0]["message"]["content"], "PRIMARY_AT_LIMIT",
        "a legitimate max_tokens hit must be returned as-is, no failover (Req 6.3)"
    );
    assert_eq!(
        json["choices"][0]["finish_reason"], "length",
        "the legitimate length finish_reason is preserved"
    );
    assert_eq!(
        json["usage"]["completion_tokens"], 1000,
        "the primary's full-limit usage is returned"
    );
}

/// Req 6.2: when every provider truncates with `finish_reason: "length"`, the
/// gateway returns the LONGEST partial (highest completion_tokens) instead of an
/// error. Primary truncates at 10 tokens, backup at 40 → the backup's longer
/// partial is returned, with its truncation finish_reason preserved.
#[tokio::test]
async fn test_truncation_all_providers_truncate_returns_longest() {
    let primary = start_truncation_mock("PRIMARY_SHORT", "length", 10).await;
    let backup = start_truncation_mock("BACKUP_LONGER", "length", 40).await;

    let app = build_app(truncation_config(&primary.uri(), &backup.uri(), true)).await;
    let (status, json) = post_chat_nonstream(app, 1000).await;

    assert_eq!(
        status,
        StatusCode::OK,
        "all-truncated must return the longest partial, not an error (Req 6.2)"
    );
    assert_eq!(
        json["choices"][0]["message"]["content"], "BACKUP_LONGER",
        "the longest partial (highest completion_tokens) is returned (Req 6.2)"
    );
    assert_eq!(
        json["usage"]["completion_tokens"], 40,
        "the returned partial is the one with the most completion tokens"
    );
    assert_eq!(
        json["choices"][0]["finish_reason"], "length",
        "the truncation finish_reason is preserved on the returned partial"
    );
}

/// Req 6.4: with `retry_on_truncation: false`, truncation detection is disabled
/// entirely. A low-token `finish_reason: "length"` response from the primary is
/// returned as-is with no failover — proving the flag turns the behavior off.
#[tokio::test]
async fn test_truncation_retry_disabled_returns_truncated_response() {
    // Primary truncates badly; with detection off it must still be returned.
    let primary = start_truncation_mock("PRIMARY_TRUNCATED", "length", 5).await;
    let backup = start_truncation_mock("BACKUP_SHOULD_NOT_APPEAR", "stop", 42).await;

    let app = build_app(truncation_config(&primary.uri(), &backup.uri(), false)).await;
    let (status, json) = post_chat_nonstream(app, 1000).await;

    assert_eq!(status, StatusCode::OK);
    assert_eq!(
        json["choices"][0]["message"]["content"], "PRIMARY_TRUNCATED",
        "retry_on_truncation=false must return the primary's truncated response, no failover (Req 6.4)"
    );
    assert_eq!(
        json["choices"][0]["finish_reason"], "length",
        "the truncated finish_reason is preserved when detection is disabled"
    );
    assert_eq!(
        json["usage"]["completion_tokens"], 5,
        "the primary's low completion_tokens are returned unchanged"
    );
}
