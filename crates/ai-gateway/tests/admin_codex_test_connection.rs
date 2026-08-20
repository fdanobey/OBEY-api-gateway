//! Integration tests for `POST /admin/test-connection` (Codex provider).
//!
//! These tests exercise the admin endpoint's error paths that don't require
//! a live upstream connection. Full end-to-end tests with wiremock are
//! documented in the spec but require the gateway's full startup sequence;
//! they are covered by the manual smoke test (task 23).

use axum::body::Body;
use axum::http::{Request, StatusCode};
use tower::ServiceExt;

use ai_gateway::config::*;
use ai_gateway::gateway::GatewayServer;

mod common;

/// Build a minimal valid Config with a Codex-style provider (oauth + openai).
fn codex_test_config() -> Config {
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
            name: "codex-provider".to_string(),
            provider_type: "openai".to_string(),
            base_url: Some("https://api.openai.com/v1".to_string()),
            api_key_env: None,
            api_key_encrypted: None,
            api_secret_env: None,
            api_secret_encrypted: None,
            auth_method: Some("oauth".to_string()),
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
            reasoning: true,
            codex_base_url_override: None,
            codex_model_override: None,
            instructions_override: None,
            max_rate_limit_cooldown_seconds: None,
            memory: None,
        }],
        model_groups: vec![ModelGroup {
            name: "codex-group".to_string(),
            version_fallback_enabled: false,
            compression: None,
            structured_output: None,
            memory: None,
            models: vec![ProviderModel {
                provider: "codex-provider".to_string(),
                model: "gpt-4.1-nano".to_string(),
                cost_per_million_input_tokens: 30.0,
                cost_per_million_output_tokens: 60.0,
                priority: 100,
                structured_output_passthrough: None,
                tier: None,
                context_window: 0,
                specializations: vec![],
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
        first_launch_completed: false,
        tray: ai_gateway::config::TrayConfig::default(),
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
        memory: None,
    }
}

async fn build_app(mut config: Config) -> axum::Router {
    common::isolate_databases(&mut config);
    let server = GatewayServer::new(config, None).await.unwrap();
    server.build_router()
}

async fn send(app: axum::Router, req: Request<Body>) -> (StatusCode, Vec<u8>) {
    let resp = app.oneshot(req).await.unwrap();
    let status = resp.status();
    let body = axum::body::to_bytes(resp.into_body(), 1024 * 1024)
        .await
        .unwrap();
    (status, body.to_vec())
}

// ---------------------------------------------------------------------------
// Scenario: missing provider_name → 400 with status "error"
// ---------------------------------------------------------------------------

#[tokio::test]
async fn test_connection_missing_provider_name() {
    let app = build_app(codex_test_config()).await;
    let req = Request::post("/admin/test-connection")
        .header("content-type", "application/json")
        .body(Body::from(r#"{}"#))
        .unwrap();
    let (status, body) = send(app, req).await;

    assert_eq!(status, StatusCode::BAD_REQUEST);
    let json: serde_json::Value = serde_json::from_slice(&body).unwrap();
    assert_eq!(json["status"], "error");
    assert!(json["message"].as_str().unwrap().contains("missing"));
}

// ---------------------------------------------------------------------------
// Scenario: unknown provider → 400 with status "not_found"
// ---------------------------------------------------------------------------

#[tokio::test]
async fn test_connection_unknown_provider() {
    let app = build_app(codex_test_config()).await;
    let req = Request::post("/admin/test-connection")
        .header("content-type", "application/json")
        .body(Body::from(r#"{"provider_name":"nonexistent"}"#))
        .unwrap();
    let (status, body) = send(app, req).await;

    assert_eq!(status, StatusCode::BAD_REQUEST);
    let json: serde_json::Value = serde_json::from_slice(&body).unwrap();
    assert_eq!(json["status"], "not_found");
}

// ---------------------------------------------------------------------------
// Scenario: non-Codex provider → 400 with "not a Codex provider"
// ---------------------------------------------------------------------------

#[tokio::test]
async fn test_connection_non_codex_provider() {
    let mut cfg = codex_test_config();
    // Override auth_method to make it a regular (non-Codex) provider
    cfg.providers[0].auth_method = None;

    let app = build_app(cfg).await;
    let req = Request::post("/admin/test-connection")
        .header("content-type", "application/json")
        .body(Body::from(r#"{"provider_name":"codex-provider"}"#))
        .unwrap();
    let (status, body) = send(app, req).await;

    assert_eq!(status, StatusCode::BAD_REQUEST);
    let json: serde_json::Value = serde_json::from_slice(&body).unwrap();
    assert_eq!(json["status"], "error");
    assert!(json["message"]
        .as_str()
        .unwrap()
        .contains("not a Codex provider"));
}

// ---------------------------------------------------------------------------
// Scenario B: OAuth session unauthenticated → 400 with status "unauthenticated"
// (oauth_manager is None because no OAuth config is provided in test)
// ---------------------------------------------------------------------------

#[tokio::test]
async fn test_connection_unauthenticated_when_no_oauth_session() {
    let app = build_app(codex_test_config()).await;
    let req = Request::post("/admin/test-connection")
        .header("content-type", "application/json")
        .body(Body::from(r#"{"provider_name":"codex-provider"}"#))
        .unwrap();
    let (status, body) = send(app, req).await;

    assert_eq!(status, StatusCode::BAD_REQUEST);
    let json: serde_json::Value = serde_json::from_slice(&body).unwrap();
    assert_eq!(json["status"], "unauthenticated");
}
