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

/// Build a minimal valid Config for integration tests.
fn test_config() -> Config {
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
                provider: "test-provider".to_string(),
                model: "gpt-4".to_string(),
                cost_per_million_input_tokens: 30.0,
                cost_per_million_output_tokens: 60.0,
                priority: 100,
                structured_output_passthrough: None,
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
        structured_output: None,
    }
}

/// Helper: build a router from a config without binding to a port.
async fn build_app(config: Config) -> axum::Router {
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
    let server = GatewayServer::new(cfg, None).await.unwrap();
    server.state.metrics.record_guardrail_stage(
        "pii_pipeline",
        "pii_scan",
        "regex",
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
            "obey_api_guardrail_stage_executions_total{pipeline=\"pii_pipeline\",stage=\"pii_scan\",provider=\"regex\",action=\"redact\"} 1"
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
            "obey_api_guardrail_stage_latency_ms_count{pipeline=\"pii_pipeline\",stage=\"pii_scan\",provider=\"regex\"} 1"
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
        reset_policy: BudgetResetPolicy::Manual,
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
                provider: "primary".to_string(),
                model: "gpt-4".to_string(),
                cost_per_million_input_tokens: 30.0,
                cost_per_million_output_tokens: 60.0,
                priority: 100,
                structured_output_passthrough: None,
            },
            ProviderModel {
                provider: "backup".to_string(),
                model: "gpt-4".to_string(),
                cost_per_million_input_tokens: 30.0,
                cost_per_million_output_tokens: 60.0,
                priority: 200,
                structured_output_passthrough: None,
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
                provider: "primary".to_string(),
                model: "gpt-4".to_string(),
                cost_per_million_input_tokens: 30.0,
                cost_per_million_output_tokens: 60.0,
                priority: 100,
                structured_output_passthrough: None,
            },
            ProviderModel {
                provider: "backup".to_string(),
                model: "gpt-4".to_string(),
                cost_per_million_input_tokens: 30.0,
                cost_per_million_output_tokens: 60.0,
                priority: 200,
                structured_output_passthrough: None,
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
