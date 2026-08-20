//! Dynamic request body size limiting (Req 45.1-45.5).
//!
//! Replaces Axum's static [`DefaultBodyLimit`](axum::extract::DefaultBodyLimit)
//! layer, which fixes the limit at router construction time and therefore
//! requires a process restart to change. This middleware instead reads
//! `server.max_request_size_mb` from the live configuration on every request
//! and installs a per-request [`DefaultBodyLimit`] extension, so the limit can
//! be adjusted via the admin panel or config hot-reload without restarting
//! the gateway process.
//!
//! Enforcement happens in two stages:
//! 1. Requests whose declared `Content-Length` already exceeds the limit are
//!    rejected with 413 before the body is downloaded.
//! 2. Otherwise, the per-request extension makes body-consuming extractors
//!    (`Json`, `Bytes`, `Multipart`) enforce the dynamic limit natively —
//!    including for chunked bodies that have no `Content-Length`.

use std::sync::Arc;

use axum::{
    extract::{DefaultBodyLimit, Request, State},
    http::{header, Method, StatusCode},
    middleware::Next,
    response::{IntoResponse, Response},
    Json,
};
use serde_json::json;
use tokio::sync::RwLock;

use crate::config::Config;
use crate::gateway::AppState;

/// Read the current body limit (bytes) from the live config (Req 45.1).
///
/// Saturating multiplication guards against overflow when an operator sets a
/// very large `max_request_size_mb` value.
pub async fn current_limit_bytes(config: &Arc<RwLock<Config>>) -> usize {
    let config = config.read().await;
    (config.server.max_request_size_mb as usize).saturating_mul(1024 * 1024)
}

/// Axum middleware enforcing the request body size limit dynamically (Req 45.2).
///
/// Applied at the same position in the layer stack as the former static
/// `DefaultBodyLimit` layer (innermost global layer, after CORS/tracing).
pub async fn request_body_limit_middleware(
    State(state): State<AppState>,
    mut request: Request,
    next: Next,
) -> Response {
    let max_body_bytes = current_limit_bytes(&state.config).await;

    // Fast path: reject bodyful requests whose declared size already exceeds
    // the limit without downloading the body.
    if !bodyless_method(request.method()) {
        if let Some(content_length) = content_length(&request) {
            if content_length > max_body_bytes {
                return payload_too_large(max_body_bytes);
            }
        }
    }

    // Body-consuming extractors enforce this per-request limit in place of
    // axum's built-in 2 MB default. `apply` inserts the internal
    // DefaultBodyLimitKind extension that extractors check.
    DefaultBodyLimit::max(max_body_bytes).apply(&mut request);

    next.run(request).await
}

fn bodyless_method(method: &Method) -> bool {
    matches!(*method, Method::GET | Method::HEAD | Method::OPTIONS)
}

fn content_length(request: &Request) -> Option<usize> {
    request
        .headers()
        .get(header::CONTENT_LENGTH)
        .and_then(|value| value.to_str().ok())
        .and_then(|value| value.parse::<usize>().ok())
}

fn payload_too_large(max_body_bytes: usize) -> Response {
    let max_mb = max_body_bytes / (1024 * 1024);
    (
        StatusCode::PAYLOAD_TOO_LARGE,
        Json(json!({
            "error": {
                "message": format!(
                    "Request body exceeds the configured limit of {} MB",
                    max_mb
                ),
                "type": "request_too_large",
                "code": "payload_too_large"
            }
        })),
    )
        .into_response()
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::compression::config::CompressionConfig;
    use crate::config::{
        AdminConfig, CircuitBreakerConfig, Config, CorsConfig, DashboardConfig, ExactCacheConfig,
        LoggingConfig, ModelGroup, Provider, ProviderConnectionPoolConfig, ProviderModel,
        RetryConfig, ServerConfig, TrayConfig,
    };
    use crate::gateway::GatewayServer;
    use axum::body::Body;
    use axum::http::Request;
    use axum::http::StatusCode;
    use tower::ServiceExt;

    fn minimal_config() -> Config {
        // Mirrors the minimal_config in gateway::mod::tests but kept local to
        // avoid depending on private test helpers.
        Config {
            server: ServerConfig {
                host: "127.0.0.1".to_string(),
                port: 3000,
                request_timeout_seconds: 30,
                max_request_size_mb: 10,
            },
            tls: None,
            admin: AdminConfig::default(),
            dashboard: DashboardConfig::default(),
            cors: CorsConfig::default(),
            memory: None,
            providers: vec![Provider {
                name: "test".to_string(),
                provider_type: "openai".to_string(),
                base_url: Some("http://localhost:1234".to_string()),
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
                memory: None,
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
            }],
            model_groups: vec![ModelGroup {
                name: "test-group".to_string(),
                version_fallback_enabled: false,
                memory: None,
                compression: None,
                structured_output: None,
                models: vec![ProviderModel {
                    provider: "test".to_string(),
                    model: "gpt-4".to_string(),
                    cost_per_million_input_tokens: 0.0,
                    cost_per_million_output_tokens: 0.0,
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
            context: crate::config::ContextConfig::default(),
            compression: CompressionConfig::default(),
            structured_output: None,
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
        }
    }

    fn config_with_size_limit(max_mb: u64) -> Config {
        let mut cfg = minimal_config();
        cfg.server.max_request_size_mb = max_mb;
        cfg
    }

    /// Minimal router with a body-consuming endpoint so the limit middleware
    /// is exercised, mirroring the production layer stack.
    fn build_test_router(server: &GatewayServer) -> axum::Router {
        async fn echo_handler(body: axum::body::Bytes) -> String {
            format!("{}", body.len())
        }

        let api_routes = axum::Router::new()
            .route("/v1/chat/completions", axum::routing::post(echo_handler))
            .layer(axum::middleware::from_fn_with_state(
                server.state.clone(),
                request_body_limit_middleware,
            ));

        axum::Router::new()
            .merge(api_routes)
            .with_state(server.state.clone())
    }

    fn post_body(body: Vec<u8>) -> Request<Body> {
        Request::builder()
            .method("POST")
            .uri("/v1/chat/completions")
            .header("content-type", "application/json")
            .body(Body::from(body))
            .unwrap()
    }

    #[tokio::test]
    async fn oversized_body_returns_413() {
        let cfg = config_with_size_limit(1);
        let server = GatewayServer::new(cfg, None).await.unwrap();
        let app = build_test_router(&server);

        let resp = app
            .oneshot(post_body(vec![b'x'; 1024 * 1024 + 1]))
            .await
            .unwrap();
        assert_eq!(resp.status(), StatusCode::PAYLOAD_TOO_LARGE);
    }

    #[tokio::test]
    async fn body_at_limit_passes() {
        let cfg = config_with_size_limit(1);
        let server = GatewayServer::new(cfg, None).await.unwrap();
        let app = build_test_router(&server);

        let resp = app
            .oneshot(post_body(vec![b'x'; 1024 * 1024]))
            .await
            .unwrap();
        assert_eq!(resp.status(), StatusCode::OK);
    }

    #[tokio::test]
    async fn bodies_beyond_axum_default_pass() {
        // Regression guard: without the per-request DefaultBodyLimit extension,
        // extractors fall back to axum's built-in 2 MB default and reject
        // bodies between 2 MB and the configured limit.
        let cfg = config_with_size_limit(3);
        let server = GatewayServer::new(cfg, None).await.unwrap();
        let app = build_test_router(&server);

        let resp = app
            .oneshot(post_body(vec![b'x'; 3 * 1024 * 1024]))
            .await
            .unwrap();
        assert_eq!(
            resp.status(),
            StatusCode::OK,
            "3 MB body must pass when limit is 3 MB, not trip axum's 2 MB default"
        );
    }

    #[tokio::test]
    async fn limit_changes_apply_without_restart() {
        // Regression test: the old DefaultBodyLimit read the limit once at
        // router construction; the dynamic middleware must pick up config
        // changes (admin UI / hot-reload) for subsequent requests.
        let cfg = config_with_size_limit(1);
        let server = GatewayServer::new(cfg, None).await.unwrap();
        let app = build_test_router(&server);

        let oversized = vec![b'x'; 1024 * 1024 + 1];
        let resp = app
            .clone()
            .oneshot(post_body(oversized.clone()))
            .await
            .unwrap();
        assert_eq!(
            resp.status(),
            StatusCode::PAYLOAD_TOO_LARGE,
            "1 MB + 1 body must be rejected at 1 MB limit"
        );

        // Raise the limit via live config (same path apply_runtime_config_update takes)
        {
            let mut config = server.state.config.write().await;
            config.server.max_request_size_mb = 2;
        }

        let resp = app.oneshot(post_body(oversized)).await.unwrap();
        assert_eq!(
            resp.status(),
            StatusCode::OK,
            "Same body must be accepted after limit is raised via hot-reload"
        );
    }

    #[tokio::test]
    async fn lowering_limit_rejects_without_restart() {
        let cfg = config_with_size_limit(2);
        let server = GatewayServer::new(cfg, None).await.unwrap();
        let app = build_test_router(&server);

        {
            let mut config = server.state.config.write().await;
            config.server.max_request_size_mb = 1;
        }

        let resp = app
            .oneshot(post_body(vec![b'x'; 1024 * 1024 + 1]))
            .await
            .unwrap();
        assert_eq!(
            resp.status(),
            StatusCode::PAYLOAD_TOO_LARGE,
            "Body must be rejected after limit is lowered via hot-reload"
        );
    }

    #[tokio::test]
    async fn content_length_header_enforced_without_buffering() {
        let cfg = config_with_size_limit(1);
        let server = GatewayServer::new(cfg, None).await.unwrap();
        let app = build_test_router(&server);

        let req = Request::builder()
            .method("POST")
            .uri("/v1/chat/completions")
            .header("content-type", "application/json")
            .header("content-length", (1024 * 1024 + 1).to_string())
            .body(Body::from(vec![b'x'; 1024]))
            .unwrap();

        let resp = app.oneshot(req).await.unwrap();
        assert_eq!(
            resp.status(),
            StatusCode::PAYLOAD_TOO_LARGE,
            "Declared Content-Length above the limit must be rejected"
        );
    }

    #[tokio::test]
    async fn bodyless_methods_skip_limit() {
        let cfg = config_with_size_limit(1);
        let server = GatewayServer::new(cfg, None).await.unwrap();
        let app = build_test_router(&server);

        let req = Request::builder()
            .method("GET")
            .uri("/v1/chat/completions")
            .body(Body::empty())
            .unwrap();

        // Route is POST-only; a 405 proves the request reached routing and
        // was not short-circuited by the body limit middleware.
        let resp = app.oneshot(req).await.unwrap();
        assert_eq!(resp.status(), StatusCode::METHOD_NOT_ALLOWED);
    }
}
