pub mod handlers;

use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::Arc;

use axum::{
    extract::DefaultBodyLimit,
    http::{HeaderValue, Method},
    response::Redirect,
    routing::{get, post},
    Router,
};
use tokio::sync::RwLock;
use tower_http::cors::CorsLayer;
use tower_http::trace::TraceLayer;

use crate::admin;
use crate::cache::{ExactCache, SemanticCache};
use crate::config::Config;
use crate::error::GatewayError;
use crate::logger::RequestLogger;
use crate::metrics::Metrics;
use crate::oauth::{OAuthManager, OAuthTokenStore};
use crate::router::router::Router as RequestRouter;

/// Shared application state accessible by all route handlers.
#[derive(Clone)]
pub struct AppState {
    pub config: Arc<RwLock<Config>>,
    pub config_path: Arc<std::path::PathBuf>,
    pub router: Arc<RequestRouter>,
    pub logger: Arc<RequestLogger>,
    pub cache: Option<Arc<SemanticCache>>,
    /// Tier-1 in-memory exact-match response cache (always present;
    /// disabled internally when `config.exact_cache.enabled = false`).
    pub exact_cache: Arc<ExactCache>,
    pub metrics: Arc<Metrics>,
    pub shutting_down: Arc<AtomicBool>,
    pub oauth_manager: Option<Arc<OAuthManager>>,
    pub codex_models_discovery: Arc<crate::codex::models_discovery::ModelsDiscovery>,
}

/// Core HTTP server wrapping Axum with middleware and integrated components.
pub struct GatewayServer {
    pub state: AppState,
}

impl GatewayServer {
    /// Build a new GatewayServer from a validated Config.
    pub async fn new(config: Config, config_path: Option<std::path::PathBuf>) -> Result<Self, GatewayError> {
        let logger = RequestLogger::new(config.logging.clone())
            .map_err(|e| GatewayError::Database(e.to_string()))?;

        let config_arc = Arc::new(RwLock::new(config.clone()));
        let metrics = Arc::new(Metrics::new());
        let mut router = RequestRouter::new(config_arc.clone(), metrics.clone());

        let cache = match &config.semantic_cache {
            Some(sc) if sc.enabled => {
                let embedding_provider = config
                    .providers
                    .iter()
                    .find(|p| p.name == sc.embedding_provider);

                let (base_url, api_key) = match embedding_provider {
                    Some(p) => (
                        p.base_url.clone().unwrap_or_default(),
                        p.resolve_api_key().unwrap_or_default(),
                    ),
                    None => (String::new(), String::new()),
                };

                match SemanticCache::new(sc, base_url, api_key).await {
                    Ok(semantic_cache) => Some(Arc::new(semantic_cache)),
                    Err(e) => {
                        tracing::warn!("Semantic cache unavailable, starting without cache: {}", e);
                        None
                    }
                }
            }
            _ => None,
        };

        // --- OAuth manager construction (Req 4.3, 5.1) ---
        let oauth_manager = match crate::secrets::master_key_path() {
            Ok(key_path) => {
                let storage_dir = key_path.parent().unwrap_or_else(|| std::path::Path::new("."));
                let token_path = OAuthTokenStore::default_path(storage_dir);
                let store = OAuthTokenStore::new(token_path);
                let http_client = reqwest::Client::new();
                let manager = Arc::new(OAuthManager::new(store, http_client));

                // Load any previously persisted session from disk (Req 4.3).
                if let Err(e) = manager.load_existing_session().await {
                    tracing::warn!(error = %e, "Failed to load existing OAuth session; starting unauthenticated");
                }

                // Spawn background refresh task (Req 5.1).
                manager.clone().start_refresh_loop();

                // Wire into the request router so OAuth providers can resolve tokens.
                router.set_oauth_manager(manager.clone());

                Some(manager)
            }
            Err(e) => {
                tracing::warn!(error = %e, "Could not determine OAuth storage path; OAuth disabled");
                None
            }
        };


        // --- Codex InstructionsStore construction (Req 7.1, 7.2, 7.8) ---
        // Only constructed when at least one Codex-capable provider exists.
        let has_codex_provider = config.providers.iter().any(|p| {
            p.auth_method.as_deref() == Some("oauth") && p.provider_type == "openai"
        });
        if has_codex_provider {
            match crate::codex::InstructionsStore::new(&config).await {
                Ok(store) => {
                    let store = Arc::new(store);
                    router.set_instructions_store(store.clone());
                    // Spawn a background refresh (non-blocking) so the first
                    // fetch doesn't delay /healthz readiness (Req 7.8).
                    let refresh_store = store.clone();
                    tokio::spawn(async move {
                        refresh_store.maybe_refresh().await;
                    });
                }
                Err(e) => {
                    tracing::warn!(error = %e, "Failed to initialize Codex InstructionsStore; Codex providers will use bundled fallback");
                }
            }
        }

        let state = AppState {
            config: config_arc,
            config_path: Arc::new(config_path.unwrap_or_else(|| std::path::PathBuf::from("./config.yaml"))),
            router: Arc::new(router),
            logger: Arc::new(logger),
            cache,
            exact_cache: Arc::new(ExactCache::new(&config.exact_cache)),
            metrics,
            shutting_down: Arc::new(AtomicBool::new(false)),
            oauth_manager,
            codex_models_discovery: Arc::new(crate::codex::models_discovery::ModelsDiscovery::new()),
        };

        Ok(Self { state })
    }

    /// Build the Axum router with all middleware layers.
    pub fn build_router(&self) -> Router {
        let config = self.state.config.try_read().expect("config lock poisoned");

        // --- Request size limit (Req 45.1-45.5) ---
        let max_body_bytes = config.server.max_request_size_mb as usize * 1024 * 1024;

        // --- CORS layer (Req 43.1-43.7) ---
        let cors = self.build_cors_layer(&config);

        // --- Tracing layer (Req 17.5, 20.4) ---
        let trace_layer = TraceLayer::new_for_http();

        drop(config); // release read lock before moving state

        // --- OpenAI API routes (Req 2.1-2.12) ---
        use handlers::*;

        let api_routes = Router::new()
            // Chat completions — streaming & non-streaming (Req 2.1)
            .route("/v1/chat/completions", post(chat_completions))
            // Legacy completions (Req 2.2)
            .route("/v1/completions", post(completions))
            // Embeddings (Req 2.3)
            .route("/v1/embeddings", post(embeddings))
            // Images (Req 2.4)
            .route("/v1/images/generations", post(image_generations))
            // Audio (Req 2.5)
            .route("/v1/audio/transcriptions", post(audio_transcriptions))
            .route("/v1/audio/translations", post(audio_translations))
            // Models (Req 2.6, 2.12)
            .route("/v1/models", get(list_models))
            // Assistants (Req 2.7)
            .route("/v1/assistants", post(create_assistant).get(list_assistants))
            .route(
                "/v1/assistants/{assistant_id}",
                get(get_assistant)
                    .post(modify_assistant)
                    .delete(delete_assistant),
            )
            // Threads (Req 2.8)
            .route("/v1/threads", post(create_thread))
            .route(
                "/v1/threads/{thread_id}",
                get(get_thread)
                    .post(modify_thread)
                    .delete(delete_thread),
            )
            // Messages on threads
            .route(
                "/v1/threads/{thread_id}/messages",
                post(create_message).get(list_messages),
            )
            // Runs (Req 2.9)
            .route(
                "/v1/threads/{thread_id}/runs",
                post(create_run).get(list_runs),
            )
            .route(
                "/v1/threads/{thread_id}/runs/{run_id}",
                get(get_run),
            )
            .route(
                "/v1/threads/{thread_id}/runs/{run_id}/cancel",
                post(cancel_run),
            )
            // Files (Req 2.10)
            .route("/v1/files", post(upload_file).get(list_files))
            .route(
                "/v1/files/{file_id}",
                get(get_file).delete(delete_file),
            )
            .route("/v1/files/{file_id}/content", get(get_file_content))
            // Fine-tuning (Req 2.11)
            .route(
                "/v1/fine_tuning/jobs",
                post(create_fine_tuning_job).get(list_fine_tuning_jobs),
            )
            .route(
                "/v1/fine_tuning/jobs/{fine_tuning_id}",
                get(get_fine_tuning_job),
            )
            .route(
                "/v1/fine_tuning/jobs/{fine_tuning_id}/cancel",
                post(cancel_fine_tuning_job),
            )
            .route(
                "/v1/fine_tuning/jobs/{fine_tuning_id}/events",
                get(list_fine_tuning_events),
            );

        // --- Admin panel (Req 13.1-13.18) ---
        let (admin_enabled, admin_path, dashboard_enabled, dashboard_path, prometheus_cfg) = {
            let cfg = self.state.config.try_read().expect("config lock");
            (
                cfg.admin.enabled,
                cfg.admin.path.clone(),
                cfg.dashboard.enabled,
                cfg.dashboard.path.clone(),
                cfg.prometheus.clone(),
            )
        };

        tracing::info!(
            admin_enabled,
            admin_path = %admin_path,
            dashboard_enabled,
            dashboard_path = %dashboard_path,
            prometheus_enabled = prometheus_cfg.as_ref().is_some_and(|cfg| cfg.enabled),
            "Gateway route mount configuration resolved"
        );

        let mut router = Router::new()
            .merge(api_routes)
            // Health check (Req 20.1-20.3)
            .route("/health", get(handlers::health_check));

        if admin_enabled {
            tracing::info!(path = %admin_path, "Mounting admin routes");
            router = router.nest(&admin_path, admin::admin_routes(self.state.clone()));
        } else {
            tracing::warn!("Admin routes are disabled by configuration");
        }

        if dashboard_enabled {
            let canonical_dashboard_path = dashboard_path.trim_end_matches('/').to_string();
            if !canonical_dashboard_path.is_empty() && canonical_dashboard_path != "/" {
                let slash_dashboard_path = format!("{}/", canonical_dashboard_path);
                let redirect_target = canonical_dashboard_path.clone();
                tracing::info!(from = %slash_dashboard_path, to = %redirect_target, "Registering dashboard trailing-slash redirect");
                router = router.route(
                    &slash_dashboard_path,
                    get(move || {
                        let redirect_target = redirect_target.clone();
                        async move { Redirect::permanent(&redirect_target) }
                    }),
                );
            }
            tracing::info!(path = %dashboard_path, "Mounting dashboard routes");
            router = router.nest(&dashboard_path, crate::dashboard::dashboard_routes(self.state.clone()));
        } else {
            tracing::warn!("Dashboard routes are disabled by configuration");
        }

        // Prometheus metrics endpoint (Req 20.7-20.11) — conditional on config
        if let Some(prom) = prometheus_cfg {
            if prom.enabled {
                router = router.route(&prom.path, get(handlers::prometheus_metrics));
            }
        }

        router
            .fallback(|req: axum::extract::Request| async move {
                tracing::warn!(method = %req.method(), uri = %req.uri(), "No route matched");
                axum::response::Response::builder()
                    .status(404)
                    .header("Content-Type", "application/json")
                    .body(axum::body::Body::from(
                        serde_json::json!({"error":{"message":"Not found","type":"invalid_request_error"}}).to_string()
                    ))
                    .unwrap()
            })
            .layer(DefaultBodyLimit::max(max_body_bytes))
            .layer(cors)
            .layer(trace_layer)
            .with_state(self.state.clone())
    }

    /// Validate TLS configuration on startup (Req 36.2-36.5).
    /// Returns the RustlsConfig if TLS is enabled, or None if disabled.
    async fn validate_tls(
        tls: &Option<crate::config::TlsConfig>,
    ) -> Result<Option<axum_server::tls_rustls::RustlsConfig>, GatewayError> {
        let tls_cfg = match tls {
            Some(t) if t.enabled => t,
            _ => return Ok(None), // Req 36.7: TLS disabled by default
        };

        // Req 36.4: cert/key files must exist
        let cert = std::path::Path::new(&tls_cfg.cert_path);
        let key = std::path::Path::new(&tls_cfg.key_path);

        if !cert.exists() {
            return Err(GatewayError::Configuration(format!(
                "TLS certificate file not found: {}",
                tls_cfg.cert_path
            )));
        }
        if !key.exists() {
            return Err(GatewayError::Configuration(format!(
                "TLS private key file not found: {}",
                tls_cfg.key_path
            )));
        }

        // Req 36.5: validate cert/key are parseable
        let rustls_config =
            axum_server::tls_rustls::RustlsConfig::from_pem_file(&tls_cfg.cert_path, &tls_cfg.key_path)
                .await
                .map_err(|e| {
                    GatewayError::Configuration(format!(
                        "Invalid TLS certificate or key: {}",
                        e
                    ))
                })?;

        Ok(Some(rustls_config))
    }

    /// Start listening on the configured address.
    /// Returns a future that resolves when the server shuts down.
    pub async fn start(self) -> Result<(), GatewayError> {
        self.start_with_shutdown(shutdown_signal()).await
    }

    pub async fn start_with_shutdown<F>(self, shutdown: F) -> Result<(), GatewayError>
    where
        F: std::future::Future<Output = ()> + Send + 'static,
    {
        let router = self.build_router();

        let (host, port, tls) = {
            let cfg = self.state.config.read().await;
            (cfg.server.host.clone(), cfg.server.port, cfg.tls.clone())
        };

        // Validate TLS before binding (Req 36.2-36.5)
        let rustls_config = Self::validate_tls(&tls).await?;

        let addr = format!("{}:{}", host, port);

        let shutting_down = self.state.shutting_down.clone();
        let logger = self.state.logger.clone();
        let metrics = self.state.metrics.clone();

        // Spawn a background task to reset the per-minute request counter every 60s
        let metrics_reset = self.state.metrics.clone();
        tokio::spawn(async move {
            let mut interval = tokio::time::interval(std::time::Duration::from_secs(60));
            loop {
                interval.tick().await;
                metrics_reset.reset_minute_counter();
            }
        });

        if let Some(rustls_config) = rustls_config {
            // Req 36.1: HTTPS only when TLS enabled
            let addr_parsed: std::net::SocketAddr = addr
                .parse()
                .map_err(|e| GatewayError::Configuration(format!("Invalid address {}: {}", addr, e)))?;

            tracing::info!("OBEY-API listening on {} (HTTPS)", addr);

            let handle = axum_server::Handle::new();
            let handle_clone = handle.clone();

            tokio::spawn(async move {
                shutdown.await;
                shutting_down.store(true, Ordering::Relaxed);
                handle_clone.graceful_shutdown(None);
            });

            axum_server::bind_rustls(addr_parsed, rustls_config)
                .handle(handle)
                .serve(router.into_make_service())
                .await
                .map_err(|e| GatewayError::Http(format!("Server error: {}", e)))?;
        } else {
            // Req 36.6: plain HTTP when TLS disabled
            let listener = tokio::net::TcpListener::bind(&addr)
                .await
                .map_err(|e| {
                    GatewayError::Configuration(format!("Failed to bind {}: {}", addr, e))
                })?;

            tracing::info!("OBEY-API listening on {} (HTTP)", addr);

            axum::serve(listener, router)
                .with_graceful_shutdown(async move {
                    shutdown.await;
                    shutting_down.store(true, Ordering::Relaxed);
                })
                .await
                .map_err(|e| GatewayError::Http(format!("Server error: {}", e)))?;
        }

        // Server has stopped — all in-flight requests are complete.
        // Flush metrics and close database connections (Req 18.3, 18.4).
        tracing::info!("In-flight requests drained, cleaning up resources…");
        metrics.flush();
        logger.flush();
        tracing::info!("Graceful shutdown complete");

        Ok(())
    }

    /// Reload configuration from disk (Req 26.1-26.7).
    pub async fn reload_config(&self, new_config: Config) -> Result<(), GatewayError> {
        apply_runtime_config_update(&self.state, new_config).await;
        Ok(())
    }

    // -- private helpers --

    fn build_cors_layer(&self, config: &Config) -> CorsLayer {
        if !config.cors.enabled {
            return CorsLayer::new();
        }

        let origins: Vec<HeaderValue> = config
            .cors
            .allowed_origins
            .iter()
            .filter_map(|o| {
                if o == "*" {
                    tracing::warn!("CORS wildcard origin configured — review security implications");
                }
                o.parse().ok()
            })
            .collect();

        let methods: Vec<Method> = config
            .cors
            .allowed_methods
            .iter()
            .filter_map(|m| m.parse().ok())
            .collect();

        let headers: Vec<axum::http::HeaderName> = config
            .cors
            .allowed_headers
            .iter()
            .filter_map(|h| h.parse().ok())
            .collect();

        CorsLayer::new()
            .allow_origin(origins)
            .allow_methods(methods)
            .allow_headers(headers)
    }
}

pub async fn apply_runtime_config_update(state: &AppState, new_config: Config) {
    {
        let mut cfg = state.config.write().await;
        *cfg = new_config;
    }
    state.router.clear_circuit_breakers();
    state.router.clear_rate_limiters();
    state.router.clear_http_clients();
    state.router.clear_model_capabilities();
}

/// Wait for SIGTERM / SIGINT (Req 18.1-18.5).
async fn shutdown_signal() {
    let ctrl_c = async {
        tokio::signal::ctrl_c()
            .await
            .expect("failed to install Ctrl+C handler");
    };

    #[cfg(unix)]
    let terminate = async {
        tokio::signal::unix::signal(tokio::signal::unix::SignalKind::terminate())
            .expect("failed to install SIGTERM handler")
            .recv()
            .await;
    };

    #[cfg(not(unix))]
    let terminate = std::future::pending::<()>();

    tokio::select! {
        _ = ctrl_c => tracing::info!("Received SIGINT, shutting down…"),
        _ = terminate => tracing::info!("Received SIGTERM, shutting down…"),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::config::*;
    use axum::body::Body;
    use axum::http::{Request, StatusCode};
    use proptest::prelude::*;
    use tower::ServiceExt;

    fn minimal_config() -> Config {
        Config {
            server: ServerConfig {
                host: "127.0.0.1".to_string(),
                port: 0, // OS-assigned
                request_timeout_seconds: 30,
                max_request_size_mb: 10,
            },
            tls: None,
            admin: AdminConfig::default(),
            dashboard: DashboardConfig::default(),
            cors: CorsConfig::default(),
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
                    provider: "test".to_string(),
                    model: "gpt-4".to_string(),
                    cost_per_million_input_tokens: 0.0,
                    cost_per_million_output_tokens: 0.0,
                    priority: 100,
                }],
            }],
            circuit_breaker: CircuitBreakerConfig::default(),
            retry: RetryConfig::default(),
            logging: LoggingConfig::default(),
            semantic_cache: None,
            exact_cache: ExactCacheConfig::default(),
            prometheus: None,
            context: crate::config::ContextConfig::default(),
            first_launch_completed: false,
            tray: TrayConfig::default(),
            codex_instructions_url: None,
        }
    }

    #[tokio::test]
    async fn test_gateway_server_new() {
        let server = GatewayServer::new(minimal_config(), None).await;
        assert!(server.is_ok());
        let server = server.unwrap();
        assert!(server.state.cache.is_none());
    }

    #[tokio::test]
    async fn test_build_router_returns_router() {
        let server = GatewayServer::new(minimal_config(), None).await.unwrap();
        let _router = server.build_router(); // should not panic
    }

    #[tokio::test]
    async fn test_reload_config() {
        let server = GatewayServer::new(minimal_config(), None).await.unwrap();
        let mut new_cfg = minimal_config();
        new_cfg.server.port = 9999;
        server.reload_config(new_cfg).await.unwrap();
        let cfg = server.state.config.read().await;
        assert_eq!(cfg.server.port, 9999);
    }

    #[test]
    fn test_cors_disabled_by_default() {
        // CorsConfig::default() has enabled: false
        let cfg = CorsConfig::default();
        assert!(!cfg.enabled);
    }

    #[tokio::test]
    async fn test_default_body_limit() {
        // Validates: Req 45.3 — default 10 MB
        let cfg = minimal_config();
        assert_eq!(cfg.server.max_request_size_mb, 10);
        let server = GatewayServer::new(cfg, None).await.unwrap();
        let _router = server.build_router();
    }

    // --- Health check tests (Req 20.1-20.3) ---

    #[tokio::test]
    async fn test_health_check_returns_200_when_operational() {
        let server = GatewayServer::new(minimal_config(), None).await.unwrap();
        let app = server.build_router();

        let req = Request::builder()
            .method("GET")
            .uri("/health")
            .body(Body::empty())
            .unwrap();

        let resp = app.oneshot(req).await.unwrap();
        assert_eq!(resp.status(), StatusCode::OK);

        let body = axum::body::to_bytes(resp.into_body(), 1024).await.unwrap();
        let json: serde_json::Value = serde_json::from_slice(&body).unwrap();
        assert_eq!(json["status"], "ok");
    }

    #[tokio::test]
    async fn test_health_check_returns_503_when_shutting_down() {
        let server = GatewayServer::new(minimal_config(), None).await.unwrap();
        server.state.shutting_down.store(true, Ordering::Relaxed);
        let app = server.build_router();

        let req = Request::builder()
            .method("GET")
            .uri("/health")
            .body(Body::empty())
            .unwrap();

        let resp = app.oneshot(req).await.unwrap();
        assert_eq!(resp.status(), StatusCode::SERVICE_UNAVAILABLE);

        let body = axum::body::to_bytes(resp.into_body(), 1024).await.unwrap();
        let json: serde_json::Value = serde_json::from_slice(&body).unwrap();
        assert_eq!(json["status"], "shutting_down");
    }

    fn config_with_size_limit(max_mb: u64) -> Config {
        let mut cfg = minimal_config();
        cfg.server.max_request_size_mb = max_mb;
        cfg
    }

    /// Build a minimal router with a body-consuming endpoint so DefaultBodyLimit is exercised.
    fn build_test_router(server: &GatewayServer) -> Router {
        use axum::body::Bytes;

        let config = server.state.config.try_read().expect("config lock");
        let max_body_bytes = config.server.max_request_size_mb as usize * 1024 * 1024;
        let cors = server.build_cors_layer(&config);
        let trace_layer = TraceLayer::new_for_http();
        drop(config);

        // A trivial handler that consumes the full body — triggers DefaultBodyLimit check
        async fn echo_handler(body: Bytes) -> String {
            format!("{}", body.len())
        }

        let api_routes = axum::Router::new()
            .route("/v1/chat/completions", axum::routing::post(echo_handler));

        Router::new()
            .merge(api_routes)
            .layer(DefaultBodyLimit::max(max_body_bytes))
            .layer(cors)
            .layer(trace_layer)
            .with_state(server.state.clone())
    }

    // Feature: ai-gateway, Property 28: Models Endpoint Aggregation
    // For any request to /v1/models, the response shall contain the union of all
    // models from all configured providers, with duplicates removed and provider
    // information included in metadata.
    // **Validates: Requirements 2.12, 24.2, 24.3, 24.4, 24.5**
    proptest! {
        #![proptest_config(ProptestConfig {
            cases: 100,
            .. ProptestConfig::default()
        })]

        #[test]
        fn prop_models_endpoint_aggregation(
            // Generate 1-4 model groups, each with 1-5 models
            num_groups in 1usize..=4,
            // Use a seed to deterministically build model/provider combos
            seed in 0u64..10000,
        ) {
            let rt = tokio::runtime::Runtime::new().unwrap();
            rt.block_on(async {
                // Build providers and model groups with potential duplicates
                let provider_names: Vec<String> = (0..3)
                    .map(|i| format!("provider-{}", i))
                    .collect();

                let model_names: Vec<String> = vec![
                    "gpt-4".to_string(),
                    "gpt-3.5-turbo".to_string(),
                    "claude-3".to_string(),
                    "llama-3".to_string(),
                    "mistral-7b".to_string(),
                ];

                let providers: Vec<Provider> = provider_names.iter().map(|name| {
                    Provider {
                        name: name.clone(),
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
                    }
                }).collect();

                // Build model groups using seed for deterministic selection
                let mut rng_state = seed;
                let mut model_groups = Vec::new();
                let mut expected_unique_models = std::collections::HashMap::<String, String>::new();

                // list_models also returns group names as model entries (owned_by "gateway")
                for g in 0..num_groups {
                    expected_unique_models
                        .entry(format!("group-{}", g))
                        .or_insert_with(|| "gateway".to_string());

                    let num_models = (rng_state % 5) as usize + 1;
                    rng_state = rng_state.wrapping_mul(6364136223846793005).wrapping_add(1);

                    let mut models = Vec::new();
                    for _ in 0..num_models {
                        let model_idx = (rng_state % model_names.len() as u64) as usize;
                        rng_state = rng_state.wrapping_mul(6364136223846793005).wrapping_add(1);
                        let prov_idx = (rng_state % provider_names.len() as u64) as usize;
                        rng_state = rng_state.wrapping_mul(6364136223846793005).wrapping_add(1);

                        let model_name = model_names[model_idx].clone();
                        let prov_name = provider_names[prov_idx].clone();

                        // Track first occurrence for expected owned_by
                        expected_unique_models
                            .entry(model_name.clone())
                            .or_insert_with(|| prov_name.clone());

                        models.push(ProviderModel {
                            provider: prov_name,
                            model: model_name,
                            cost_per_million_input_tokens: 0.0,
                            cost_per_million_output_tokens: 0.0,
                            priority: 100,
                        });
                    }

                    model_groups.push(ModelGroup {
                        name: format!("group-{}", g),
                        version_fallback_enabled: false,
                        models,
                    });
                }

                let cfg = Config {
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
                    providers,
                    model_groups,
                    circuit_breaker: CircuitBreakerConfig::default(),
                    retry: RetryConfig::default(),
                    logging: LoggingConfig::default(),
                    semantic_cache: None,
                    exact_cache: ExactCacheConfig::default(),
                    prometheus: None,
                    context: ContextConfig::default(),
                    first_launch_completed: false,
                    tray: TrayConfig::default(),
                    codex_instructions_url: None,
                };

                let server = GatewayServer::new(cfg, None).await.unwrap();
                let app = server.build_router();

                // Hit GET /v1/models
                let req = Request::builder()
                    .method("GET")
                    .uri("/v1/models")
                    .body(Body::empty())
                    .unwrap();

                let resp = app.oneshot(req).await.unwrap();
                prop_assert_eq!(resp.status(), StatusCode::OK);

                let body_bytes = axum::body::to_bytes(resp.into_body(), 1024 * 1024)
                    .await
                    .unwrap();
                let parsed: serde_json::Value = serde_json::from_slice(&body_bytes).unwrap();

                // Req 24.5: response is in OpenAI models format
                prop_assert_eq!(parsed["object"].as_str(), Some("list"));
                let data = parsed["data"].as_array().unwrap();

                // Req 24.3: no duplicate model IDs
                let mut seen_ids = std::collections::HashSet::new();
                for model in data {
                    let id = model["id"].as_str().unwrap().to_string();
                    prop_assert!(
                        seen_ids.insert(id.clone()),
                        "Duplicate model ID found: {}",
                        id
                    );
                }

                // Req 24.2: union — every unique model from config must appear
                for model_id in expected_unique_models.keys() {
                    prop_assert!(
                        seen_ids.contains(model_id.as_str()),
                        "Expected model '{}' missing from response",
                        model_id
                    );
                }

                // Req 24.4: provider information included in metadata (owned_by)
                for model in data {
                    let owned_by = model["owned_by"].as_str().unwrap();
                    prop_assert!(
                        !owned_by.is_empty(),
                        "Model '{}' has empty owned_by",
                        model["id"].as_str().unwrap()
                    );
                }

                // Count matches: response should have exactly the unique model count
                prop_assert_eq!(
                    data.len(),
                    expected_unique_models.len(),
                    "Response model count should equal unique model count from config"
                );

                Ok(())
            })?;
        }
    }

    // Feature: ai-gateway, Property 45: Health Check Status
    // For any operational gateway, the /health endpoint shall return HTTP 200;
    // for any gateway in shutdown state, the /health endpoint shall return HTTP 503.
    // **Validates: Requirements 20.2, 20.3**
    proptest! {
        #![proptest_config(ProptestConfig {
            cases: 100,
            .. ProptestConfig::default()
        })]

        #[test]
        fn prop_health_check_status(shutting_down in proptest::bool::ANY) {
            let rt = tokio::runtime::Runtime::new().unwrap();
            rt.block_on(async {
                let server = GatewayServer::new(minimal_config(), None).await.unwrap();
                server.state.shutting_down.store(shutting_down, Ordering::Relaxed);
                let app = server.build_router();

                let req = Request::builder()
                    .method("GET")
                    .uri("/health")
                    .body(Body::empty())
                    .unwrap();

                let resp = app.oneshot(req).await.unwrap();
                let status = resp.status();
                let body = axum::body::to_bytes(resp.into_body(), 1024).await.unwrap();
                let json: serde_json::Value = serde_json::from_slice(&body).unwrap();

                if shutting_down {
                    prop_assert_eq!(status, StatusCode::SERVICE_UNAVAILABLE,
                        "Expected 503 when shutting_down=true");
                    prop_assert_eq!(json["status"].as_str(), Some("shutting_down"));
                } else {
                    prop_assert_eq!(status, StatusCode::OK,
                        "Expected 200 when shutting_down=false");
                    prop_assert_eq!(json["status"].as_str(), Some("ok"));
                }

                Ok(())
            })?;
        }
    }

    // Feature: ai-gateway, Property 43: CORS Header Inclusion
    // For any request from an allowed origin when CORS is enabled, the response
    // shall include Access-Control-Allow-Origin header matching the request origin.
    // **Validates: Requirements 43.5**
    proptest! {
        #![proptest_config(ProptestConfig {
            cases: 100,
            .. ProptestConfig::default()
        })]

        #[test]
        fn prop_cors_header_inclusion(
            // Generate 1-5 allowed origins
            num_origins in 1usize..=5,
            seed in 0u64..10000,
        ) {
            let rt = tokio::runtime::Runtime::new().unwrap();
            rt.block_on(async {
                // Build distinct allowed origins from seed
                let allowed_origins: Vec<String> = (0..num_origins)
                    .map(|i| {
                        let id = seed.wrapping_mul(31).wrapping_add(i as u64) % 10000;
                        format!("http://example-{}.com", id)
                    })
                    .collect();

                // Pick one allowed origin to use in the request
                let chosen_idx = (seed as usize) % allowed_origins.len();
                let chosen_origin = allowed_origins[chosen_idx].clone();

                // Build a non-allowed origin that is guaranteed different
                let non_allowed_origin = format!("http://not-allowed-{}.com", seed);

                // Build config with CORS enabled and specific origins
                let mut cfg = minimal_config();
                cfg.cors = CorsConfig {
                    enabled: true,
                    allowed_origins: allowed_origins.clone(),
                    allowed_methods: vec!["GET".to_string(), "POST".to_string()],
                    allowed_headers: vec!["Content-Type".to_string()],
                };

                let server = GatewayServer::new(cfg, None).await.unwrap();
                let app = server.build_router();

                // --- Request from an allowed origin: must get ACAO header ---
                let req = Request::builder()
                    .method("GET")
                    .uri("/health")
                    .header("Origin", chosen_origin.as_str())
                    .body(Body::empty())
                    .unwrap();

                let resp = app.clone().oneshot(req).await.unwrap();
                prop_assert_eq!(resp.status(), StatusCode::OK);

                let acao = resp
                    .headers()
                    .get("access-control-allow-origin")
                    .map(|v| v.to_str().unwrap().to_string());

                prop_assert_eq!(
                    acao.as_deref(),
                    Some(chosen_origin.as_str()),
                    "Expected ACAO header '{}' for allowed origin, got {:?}",
                    chosen_origin,
                    acao
                );

                // --- Request from a non-allowed origin: must NOT get ACAO header ---
                let req2 = Request::builder()
                    .method("GET")
                    .uri("/health")
                    .header("Origin", non_allowed_origin.as_str())
                    .body(Body::empty())
                    .unwrap();

                let resp2 = app.oneshot(req2).await.unwrap();
                let acao2 = resp2
                    .headers()
                    .get("access-control-allow-origin");

                prop_assert!(
                    acao2.is_none(),
                    "Non-allowed origin '{}' should NOT receive ACAO header, but got {:?}",
                    non_allowed_origin,
                    acao2.map(|v| v.to_str().unwrap().to_string())
                );

                Ok(())
            })?;
        }
    }

    // Feature: ai-gateway, Property 37: TLS Connection Type
    // For any gateway with TLS enabled, all accepted connections shall use HTTPS protocol only.
    // **Validates: Requirements 36.1**
    proptest! {
        #![proptest_config(ProptestConfig {
            cases: 100,
            .. ProptestConfig::default()
        })]

        #[test]
        fn prop_tls_connection_type(
            tls_enabled in proptest::bool::ANY,
            has_valid_files in proptest::bool::ANY,
        ) {
            let rt = tokio::runtime::Runtime::new().unwrap();
            rt.block_on(async {
                // Install rustls crypto provider for test environment
                let _ = rustls::crypto::aws_lc_rs::default_provider().install_default();

                if !tls_enabled {
                    // Case: TLS disabled explicitly → validate_tls returns Ok(None) → HTTP mode
                    let tls_cfg = Some(TlsConfig {
                        enabled: false,
                        cert_path: "/nonexistent/cert.pem".to_string(),
                        key_path: "/nonexistent/key.pem".to_string(),
                    });
                    let result = GatewayServer::validate_tls(&tls_cfg).await;
                    prop_assert!(result.is_ok(), "Disabled TLS should return Ok");
                    prop_assert!(result.unwrap().is_none(), "Disabled TLS should return None (HTTP mode)");

                    // Case: TLS config is None → validate_tls returns Ok(None) → HTTP mode
                    let result_none = GatewayServer::validate_tls(&None).await;
                    prop_assert!(result_none.is_ok(), "None TLS should return Ok");
                    prop_assert!(result_none.unwrap().is_none(), "None TLS should return None (HTTP mode)");
                } else if !has_valid_files {
                    // Case: TLS enabled but cert/key files don't exist → must error
                    let tls_cfg = Some(TlsConfig {
                        enabled: true,
                        cert_path: "/nonexistent/cert.pem".to_string(),
                        key_path: "/nonexistent/key.pem".to_string(),
                    });
                    let result = GatewayServer::validate_tls(&tls_cfg).await;
                    prop_assert!(result.is_err(), "TLS enabled with missing files must return Err");
                } else {
                    // Case: TLS enabled with valid cert/key files → must return Ok(Some(_)) → HTTPS mode
                    let key_pair = rcgen::KeyPair::generate().unwrap();
                    let cert_signed = rcgen::CertificateParams::new(vec!["localhost".to_string()])
                        .unwrap()
                        .self_signed(&key_pair)
                        .unwrap();

                    let tmp_dir = tempfile::tempdir().unwrap();
                    let cert_path = tmp_dir.path().join("cert.pem");
                    let key_path = tmp_dir.path().join("key.pem");
                    std::fs::write(&cert_path, cert_signed.pem()).unwrap();
                    std::fs::write(&key_path, key_pair.serialize_pem()).unwrap();

                    let tls_cfg = Some(TlsConfig {
                        enabled: true,
                        cert_path: cert_path.to_str().unwrap().to_string(),
                        key_path: key_path.to_str().unwrap().to_string(),
                    });
                    let result = GatewayServer::validate_tls(&tls_cfg).await;
                    prop_assert!(
                        result.is_ok(),
                        "TLS enabled with valid cert/key should return Ok, got: {:?}",
                        result.err()
                    );
                    prop_assert!(
                        result.unwrap().is_some(),
                        "TLS enabled with valid cert/key should return Some (HTTPS mode)"
                    );
                }

                Ok(())
            })?;
        }
    }

    // Feature: ai-gateway, Property 29: Configuration Hot Reload Validation
    // For any configuration reload attempt, if validation fails, the existing
    // configuration shall remain active and an error shall be returned.
    // **Validates: Requirements 26.2, 26.3**
    proptest! {
        #![proptest_config(ProptestConfig {
            cases: 100,
            .. ProptestConfig::default()
        })]

        #[test]
        fn prop_config_hot_reload_validation(
            // Choose which kind of invalid config to write
            invalid_kind in 0u8..5,
        ) {
            let rt = tokio::runtime::Runtime::new().unwrap();
            rt.block_on(async {
                // 1. Build a gateway with a valid config
                let original_cfg = minimal_config();
                let original_port = original_cfg.server.port;
                let original_host = original_cfg.server.host.clone();
                let original_provider_count = original_cfg.providers.len();

                let tmp_dir = tempfile::tempdir().unwrap();
                let config_path = tmp_dir.path().join("config.yaml");

                // Write a valid config first (so GatewayServer can be created)
                let valid_yaml = serde_yaml::to_string(&original_cfg).unwrap();
                std::fs::write(&config_path, &valid_yaml).unwrap();

                let server = GatewayServer::new(original_cfg.clone(), Some(config_path.clone()))
                    .await
                    .unwrap();
                let app = server.build_router();

                // 2. Write an INVALID config to the same path
                let invalid_yaml = match invalid_kind {
                    0 => {
                        // Empty providers list
                        let mut bad = minimal_config();
                        bad.providers.clear();
                        serde_yaml::to_string(&bad).unwrap()
                    }
                    1 => {
                        // Port = 0 (invalid)
                        let mut bad = minimal_config();
                        bad.server.port = 0;
                        serde_yaml::to_string(&bad).unwrap()
                    }
                    2 => {
                        // Empty model group (no models)
                        let mut bad = minimal_config();
                        bad.model_groups[0].models.clear();
                        serde_yaml::to_string(&bad).unwrap()
                    }
                    3 => {
                        // Completely invalid YAML
                        "{{{{not: valid: yaml: [[[".to_string()
                    }
                    _ => {
                        // Zero timeout (invalid)
                        let mut bad = minimal_config();
                        bad.server.request_timeout_seconds = 0;
                        serde_yaml::to_string(&bad).unwrap()
                    }
                };
                std::fs::write(&config_path, &invalid_yaml).unwrap();

                // 3. Hit POST /admin/config/reload
                let req = Request::builder()
                    .method("POST")
                    .uri("/admin/config/reload")
                    .body(Body::empty())
                    .unwrap();

                let resp = app.oneshot(req).await.unwrap();
                let status = resp.status();

                // 4. Verify the response is an error
                prop_assert!(
                    status == StatusCode::BAD_REQUEST || status == StatusCode::INTERNAL_SERVER_ERROR,
                    "Expected error status for invalid config reload, got {}",
                    status
                );

                let body = axum::body::to_bytes(resp.into_body(), 1024 * 64)
                    .await
                    .unwrap();
                let json: serde_json::Value = serde_json::from_slice(&body).unwrap();
                prop_assert!(
                    json.get("error").is_some(),
                    "Response should contain an error field, got: {:?}",
                    json
                );

                // 5. Verify the original config is still active (unchanged)
                let active_cfg = server.state.config.read().await;
                prop_assert_eq!(
                    active_cfg.server.port, original_port,
                    "Original port should be preserved after failed reload"
                );
                prop_assert_eq!(
                    &active_cfg.server.host, &original_host,
                    "Original host should be preserved after failed reload"
                );
                prop_assert_eq!(
                    active_cfg.providers.len(), original_provider_count,
                    "Original provider count should be preserved after failed reload"
                );

                Ok(())
            })?;
        }
    }

    // Feature: ai-gateway, Property 41: Request Size Limit Enforcement
    // For any request with body size exceeding configured max_request_size_mb,
    // the gateway shall return HTTP 413 status without forwarding to any provider.
    // **Validates: Requirements 45.2**
    proptest! {
        #![proptest_config(ProptestConfig {
            cases: 20,
            .. ProptestConfig::default()
        })]

        #[test]
        fn prop_request_size_limit_enforcement(max_mb in 1u64..=4) {
            let rt = tokio::runtime::Runtime::new().unwrap();
            rt.block_on(async {
                let cfg = config_with_size_limit(max_mb);
                let expected_limit_bytes = max_mb as usize * 1024 * 1024;
                let server = GatewayServer::new(cfg, None).await.unwrap();
                let app = build_test_router(&server);

                // --- Oversized request: body exceeds limit by 1 byte → must get 413 ---
                let oversized_body = vec![b'x'; expected_limit_bytes + 1];
                let req = Request::builder()
                    .method("POST")
                    .uri("/v1/chat/completions")
                    .header("content-type", "application/json")
                    .body(Body::from(oversized_body))
                    .unwrap();

                let resp = app.clone().oneshot(req).await.unwrap();
                prop_assert_eq!(
                    resp.status(),
                    StatusCode::PAYLOAD_TOO_LARGE,
                    "Body of {} + 1 bytes should be rejected with 413 when limit is {} MB",
                    expected_limit_bytes,
                    max_mb
                );

                // --- Within-limit request: body exactly at limit → must NOT get 413 ---
                let ok_body = vec![b'x'; expected_limit_bytes];
                let req = Request::builder()
                    .method("POST")
                    .uri("/v1/chat/completions")
                    .header("content-type", "application/json")
                    .body(Body::from(ok_body))
                    .unwrap();

                let resp = app.oneshot(req).await.unwrap();
                prop_assert_ne!(
                    resp.status(),
                    StatusCode::PAYLOAD_TOO_LARGE,
                    "Body of exactly {} bytes should NOT be rejected when limit is {} MB",
                    expected_limit_bytes,
                    max_mb
                );

                Ok(())
            })?;
        }
    }

    // Feature: ai-gateway, Property 30: Configuration Hot Reload Application
    // For any successful configuration reload, new provider settings shall apply
    // to future requests and circuit breaker states shall be reset.
    // **Validates: Requirements 26.4, 26.5, 26.6**
    proptest! {
        #![proptest_config(ProptestConfig {
            cases: 100,
            .. ProptestConfig::default()
        })]

        #[test]
        fn prop_config_hot_reload_application(
            new_port in 1024u16..65535u16,
            provider_suffix in "[a-z]{3,8}",
            new_timeout in 1u64..=300u64,
        ) {
            let rt = tokio::runtime::Runtime::new().unwrap();
            rt.block_on(async {
                // 1. Create initial config and write to temp file
                let initial_cfg = minimal_config();
                let initial_port = initial_cfg.server.port;
                let initial_provider_name = initial_cfg.providers[0].name.clone();

                let tmp_dir = tempfile::tempdir().unwrap();
                let config_path = tmp_dir.path().join("config.yaml");
                let initial_yaml = serde_yaml::to_string(&initial_cfg).unwrap();
                std::fs::write(&config_path, &initial_yaml).unwrap();

                let server = GatewayServer::new(initial_cfg, Some(config_path.clone()))
                    .await
                    .unwrap();

                // 2. Populate circuit breakers by recording failures on the initial provider
                let cb = server.state.router.get_circuit_breaker(&initial_provider_name).await;
                cb.record_failure().await;
                cb.record_failure().await;
                cb.record_failure().await;
                // After 3 failures (default threshold), circuit should be open
                let was_available_before = cb.is_available().await;
                prop_assert!(
                    !was_available_before,
                    "Circuit breaker should be open after 3 failures"
                );

                let app = server.build_router();

                // 3. Build a NEW valid config with different settings
                let new_provider_name = format!("prov-{}", provider_suffix);
                let new_cfg = Config {
                    server: ServerConfig {
                        host: "127.0.0.1".to_string(),
                        port: new_port,
                        request_timeout_seconds: new_timeout,
                        max_request_size_mb: 10,
                    },
                    tls: None,
                    admin: AdminConfig::default(),
                    dashboard: DashboardConfig::default(),
                    cors: CorsConfig::default(),
                    providers: vec![Provider {
                        name: new_provider_name.clone(),
                        provider_type: "openai".to_string(),
                        base_url: Some("http://localhost:5678".to_string()),
                        api_key_env: None,
                        api_key_encrypted: None,
                        api_secret_env: None,
                        api_secret_encrypted: None,
                        auth_method: None,
                        resolved_api_key: None,
                        resolved_api_secret: None,
                        region: None,
                        timeout_seconds: new_timeout,
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
                        name: format!("group-{}", provider_suffix),
                        version_fallback_enabled: false,
                        models: vec![ProviderModel {
                            provider: new_provider_name.clone(),
                            model: "gpt-4".to_string(),
                            cost_per_million_input_tokens: 0.0,
                            cost_per_million_output_tokens: 0.0,
                            priority: 100,
                        }],
                    }],
                    circuit_breaker: CircuitBreakerConfig::default(),
                    retry: RetryConfig::default(),
                    logging: LoggingConfig::default(),
                    semantic_cache: None,
                    exact_cache: ExactCacheConfig::default(),
                    prometheus: None,
                    context: ContextConfig::default(),
                    first_launch_completed: false,
                    tray: TrayConfig::default(),
                    codex_instructions_url: None,
                };

                // Write new config to disk
                let new_yaml = serde_yaml::to_string(&new_cfg).unwrap();
                std::fs::write(&config_path, &new_yaml).unwrap();

                // 4. Hit POST /admin/config/reload
                let req = Request::builder()
                    .method("POST")
                    .uri("/admin/config/reload")
                    .body(Body::empty())
                    .unwrap();

                let resp = app.oneshot(req).await.unwrap();

                // 5. Verify 200 success
                let status = resp.status();
                let body = axum::body::to_bytes(resp.into_body(), 1024 * 64)
                    .await
                    .unwrap();
                let json: serde_json::Value = serde_json::from_slice(&body).unwrap();

                prop_assert_eq!(
                    status,
                    StatusCode::OK,
                    "Reload should succeed, got {} with body: {:?}",
                    status,
                    json
                );
                prop_assert_eq!(
                    json["status"].as_str(),
                    Some("ok"),
                    "Response should indicate success"
                );

                // 6. Verify new config is now active (Req 26.4)
                let active_cfg = server.state.config.read().await;
                prop_assert_eq!(
                    active_cfg.server.port, new_port,
                    "New port should be active after reload"
                );
                prop_assert_eq!(
                    active_cfg.server.request_timeout_seconds, new_timeout,
                    "New timeout should be active after reload"
                );
                prop_assert_eq!(
                    active_cfg.providers.len(), 1,
                    "Should have exactly 1 provider after reload"
                );
                prop_assert_eq!(
                    &active_cfg.providers[0].name, &new_provider_name,
                    "New provider name should be active after reload"
                );
                prop_assert_eq!(
                    active_cfg.model_groups.len(), 1,
                    "Should have exactly 1 model group after reload"
                );
                drop(active_cfg);

                // 7. Verify circuit breaker states were reset (Req 26.5)
                // After reload, getting a circuit breaker for the OLD provider should
                // return a fresh one (the DashMap was cleared), so it must be available.
                let cb_after = server.state.router.get_circuit_breaker(&initial_provider_name).await;
                let is_available_after = cb_after.is_available().await;
                prop_assert!(
                    is_available_after,
                    "Circuit breaker for '{}' should be available (reset) after config reload",
                    initial_provider_name
                );

                // Also verify a CB for the NEW provider is fresh/available
                let cb_new = server.state.router.get_circuit_breaker(&new_provider_name).await;
                let new_available = cb_new.is_available().await;
                prop_assert!(
                    new_available,
                    "Circuit breaker for new provider '{}' should be available",
                    new_provider_name
                );

                Ok(())
            })?;
        }
    }

    // --- Prometheus metrics endpoint test (Req 20.7-20.11) ---

    #[tokio::test]
    async fn test_prometheus_metrics_endpoint() {
        // Enable prometheus in config
        let mut cfg = minimal_config();
        cfg.prometheus = Some(crate::config::PrometheusConfig {
            enabled: true,
            path: "/metrics".to_string(),
        });

        let server = GatewayServer::new(cfg, None).await.unwrap();

        // Record some metrics so the output is non-trivial
        server.state.metrics.start_request();
        server.state.metrics.complete_request(150);
        server.state.metrics.record_provider_success("openai", 100);
        server.state.metrics.record_provider_failure("bedrock");
        server.state.metrics.record_provider_retry("openai", 1200);
        server.state.metrics.set_provider_budget_limit("openai", 5.0);
        server.state.metrics.record_provider_budget_exhausted("openai");
        server.state.metrics.record_provider_unknown_cost("openai");
        server.state.metrics.record_provider_rate_limit_exhausted("bedrock");
        server.state.metrics.record_cache_hit();
        server.state.metrics.record_cache_miss();
        server.state.metrics.add_cost("openai", 0.05);

        let app = server.build_router();

        let req = Request::builder()
            .method("GET")
            .uri("/metrics")
            .body(Body::empty())
            .unwrap();

        let resp = app.oneshot(req).await.unwrap();
        assert_eq!(resp.status(), StatusCode::OK);

        let ct = resp.headers().get("content-type").unwrap().to_str().unwrap();
        assert!(ct.contains("text/plain"), "Content-Type should be text/plain");

        let body = axum::body::to_bytes(resp.into_body(), 64 * 1024).await.unwrap();
        let text = String::from_utf8(body.to_vec()).unwrap();

        // Verify key metric lines are present
        assert!(text.contains("# TYPE obey_api_requests_total counter"));
        assert!(text.contains("obey_api_requests_total 1"));
        assert!(text.contains("# TYPE obey_api_active_requests gauge"));
        assert!(text.contains("obey_api_active_requests 0"));
        assert!(text.contains("obey_api_provider_requests_total{provider=\"openai\"} 1"));
        assert!(text.contains("obey_api_provider_retries_total{provider=\"openai\"} 1"));
        assert!(text.contains("obey_api_provider_retry_delay_ms_total{provider=\"openai\"} 1200"));
        assert!(text.contains("obey_api_provider_budget_limit_dollars{provider=\"openai\"} 5"));
        assert!(text.contains("obey_api_provider_budget_exhaustions_total{provider=\"openai\"} 1"));
        assert!(text.contains("obey_api_provider_unknown_cost_total{provider=\"openai\"} 1"));
        assert!(text.contains("obey_api_provider_rate_limit_exhaustions_total{provider=\"bedrock\"} 1"));
        assert!(text.contains("obey_api_cache_hit_rate"));
        assert!(text.contains("obey_api_cost_by_provider_dollars{provider=\"openai\"} 0.05"));
    }

    #[tokio::test]
    async fn test_prometheus_disabled_returns_404() {
        // Prometheus disabled (default)
        let server = GatewayServer::new(minimal_config(), None).await.unwrap();
        let app = server.build_router();

        let req = Request::builder()
            .method("GET")
            .uri("/metrics")
            .body(Body::empty())
            .unwrap();

        let resp = app.oneshot(req).await.unwrap();
        assert_eq!(resp.status(), StatusCode::NOT_FOUND);
    }
}
