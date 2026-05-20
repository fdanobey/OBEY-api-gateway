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
                model: "gpt-4".to_string(),
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
    let body = axum::body::to_bytes(resp.into_body(), 1024 * 1024).await.unwrap();
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
    let req = Request::get("/admin/config/export").body(Body::empty()).unwrap();
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
    assert!(json["error"].is_object(), "Response should contain error object");
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

    let resp = build_app(test_config()).await.oneshot(req).await.unwrap();
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
    assert!(model_ids.contains(&"gpt-4"), "Expected gpt-4 in models list");
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

    let req = Request::get("/control-panel/").body(Body::empty()).unwrap();
    let (admin_status, admin_body) = send(app.clone(), req).await;
    assert_eq!(admin_status, StatusCode::OK);
    let admin_html = std::str::from_utf8(&admin_body).unwrap();
    assert!(admin_html.contains("/ops"), "Admin UI should link to configured dashboard path");

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

    let (admin_status, _) = send(app.clone(), Request::get("/admin/").body(Body::empty()).unwrap()).await;
    assert_eq!(admin_status, StatusCode::NOT_FOUND);

    let (dash_status, _) = send(app.clone(), Request::get("/dashboard/").body(Body::empty()).unwrap()).await;
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
    assert!(text.contains("obey_api_requests_total"), "Expected request counter metric");
}

#[tokio::test]
async fn test_admin_config_validate_accepts_reliability_fields() {
    let app = build_app(test_config()).await;
    let mut validatable = test_config();
    validatable.server.port = 8080;
    validatable.retry.jitter_enabled = true;
    validatable.retry.jitter_ratio = 0.25;
    validatable.providers[0].connection_pool.max_idle_per_host = 12;
    validatable.providers[0].connection_pool.idle_timeout_seconds = 120;
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
