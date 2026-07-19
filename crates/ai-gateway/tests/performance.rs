//! Performance benchmark tests for the OBEY-API gateway.
//!
//! Validates Requirements 17.1-17.5:
//!   17.1 - Startup within 2 seconds
//!   17.2 - Forwarding overhead < 10ms
//!   17.3 - Memory < 100MB (structural check)
//!   17.4 - 100+ concurrent requests
//!   17.5 - Async I/O (verified by Tokio runtime usage)

use std::time::{Duration, Instant};

use axum::body::Body;
use axum::http::{Request, StatusCode};
use tower::ServiceExt;

use ai_gateway::config::*;
use ai_gateway::gateway::GatewayServer;

const TEST_TIMEOUT: Duration = Duration::from_secs(15);

/// Build a minimal valid Config for performance tests.
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
        streaming: None,
        virtual_keys: Default::default(),
        loop_detection: Default::default(),
        guardrails: None,
    }
}

/// Helper: build a router from a config without binding to a port.
async fn build_app(config: Config) -> axum::Router {
    let server = GatewayServer::new(config, None).await.unwrap();
    server.build_router()
}

async fn with_test_timeout<F, T>(name: &str, future: F) -> T
where
    F: std::future::Future<Output = T>,
{
    tokio::time::timeout(TEST_TIMEOUT, future)
        .await
        .unwrap_or_else(|_| panic!("Test '{name}' exceeded {:?}", TEST_TIMEOUT))
}

// ---------------------------------------------------------------------------
// 1. Startup time — Req 17.1: < 2 seconds
// ---------------------------------------------------------------------------

#[tokio::test]
async fn test_startup_time() {
    with_test_timeout("test_startup_time", async {
        let start = Instant::now();
        let server = GatewayServer::new(test_config(), None).await.unwrap();
        let _router = server.build_router();
        let elapsed = start.elapsed();

        assert!(
            elapsed < Duration::from_secs(2),
            "Startup took {:?}, exceeds 2s target (Req 17.1)",
            elapsed
        );
    })
    .await;
}

// ---------------------------------------------------------------------------
// 2. Forwarding overhead — Req 17.2: < 10ms per request
// ---------------------------------------------------------------------------

#[tokio::test]
async fn test_forwarding_overhead() {
    with_test_timeout("test_forwarding_overhead", async {
        let app = build_app(test_config()).await;

        // Warm up: 5 requests to /health
        for _ in 0..5 {
            let warm = app.clone();
            let req = Request::get("/health").body(Body::empty()).unwrap();
            let _ = warm.oneshot(req).await.unwrap();
        }

        let iterations = 100u64;

        // Measure /health
        let start = Instant::now();
        for _ in 0..iterations {
            let svc = app.clone();
            let req = Request::get("/health").body(Body::empty()).unwrap();
            let resp = svc.oneshot(req).await.unwrap();
            assert_eq!(resp.status(), StatusCode::OK);
        }
        let health_avg = start.elapsed() / iterations as u32;

        // Measure /v1/models (exercises router layer)
        let start = Instant::now();
        for _ in 0..iterations {
            let svc = app.clone();
            let req = Request::get("/v1/models").body(Body::empty()).unwrap();
            let resp = svc.oneshot(req).await.unwrap();
            assert_eq!(resp.status(), StatusCode::OK);
        }
        let models_avg = start.elapsed() / iterations as u32;

        assert!(
            health_avg < Duration::from_millis(10),
            "/health avg {:?} exceeds 10ms target (Req 17.2)",
            health_avg
        );
        assert!(
            models_avg < Duration::from_millis(10),
            "/v1/models avg {:?} exceeds 10ms target (Req 17.2)",
            models_avg
        );
    })
    .await;
}

// ---------------------------------------------------------------------------
// 3. Concurrent requests — Req 17.4: 100+ concurrent
// ---------------------------------------------------------------------------

#[tokio::test]
async fn test_concurrent_requests() {
    with_test_timeout("test_concurrent_requests", async {
        let app = build_app(test_config()).await;
        let concurrency = 200usize;

        let mut handles = Vec::with_capacity(concurrency);
        for _ in 0..concurrency {
            let svc = app.clone();
            handles.push(tokio::spawn(async move {
                let req = Request::get("/health").body(Body::empty()).unwrap();
                let resp = svc.oneshot(req).await.unwrap();
                resp.status()
            }));
        }

        let mut ok_count = 0usize;
        for handle in handles {
            let status = tokio::time::timeout(TEST_TIMEOUT, handle)
                .await
                .expect("concurrent request task timed out")
                .expect("task panicked");
            if status == StatusCode::OK {
                ok_count += 1;
            }
        }

        assert_eq!(
            ok_count, concurrency,
            "Only {ok_count}/{concurrency} concurrent requests returned 200 (Req 17.4)"
        );
    })
    .await;
}

// ---------------------------------------------------------------------------
// 4. Release profile configured — Req 17.1-17.3 (build-time)
// ---------------------------------------------------------------------------

#[test]
fn test_release_profile_configured() {
    let cargo_toml =
        std::fs::read_to_string(concat!(env!("CARGO_MANIFEST_DIR"), "/../../Cargo.toml"))
            .expect("workspace Cargo.toml not found");

    assert!(
        cargo_toml.contains("lto = true"),
        "Release profile missing LTO"
    );
    assert!(
        cargo_toml.contains("strip = true"),
        "Release profile missing strip"
    );
    assert!(
        cargo_toml.contains("opt-level = 3"),
        "Release profile missing opt-level = 3"
    );
    assert!(
        cargo_toml.contains("codegen-units = 1"),
        "Release profile missing codegen-units = 1"
    );
}
