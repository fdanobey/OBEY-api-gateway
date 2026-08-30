//! Integration tests for the Prompt-Cache-Aware Routing feature
//! (spec: prompt-cache-routing, Requirements 1-4).
//!
//! Exercises the full gateway HTTP surface via `tower::ServiceExt::oneshot()`
//! with wiremock providers — no real ports bound, isolated SQLite databases.

use axum::body::Body;
use axum::http::{Request, StatusCode};
use tower::ServiceExt;
use wiremock::matchers::{method as wm_method, path as wm_path};
use wiremock::{Mock, MockServer, ResponseTemplate};

use ai_gateway::config::*;
use ai_gateway::gateway::GatewayServer;

mod common;

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

/// Build a minimal valid Config with the given cache-aware routing settings.
fn cache_config(cache: CacheAwareRouting) -> Config {
    Config {
        cache_aware_routing: cache,
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
        retry: RetryConfig {
            max_retries_per_provider: 0,
            ..RetryConfig::default()
        },
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
        structured_output: None,
        xhigh_models_allowlist: Default::default(),
        reasoning_models_allowlist: Default::default(),
        codex_search: None,
        reasoning_compat: Default::default(),
    }
}

fn cache_aware(enabled: bool) -> CacheAwareRouting {
    CacheAwareRouting {
        enabled,
        stickiness_ttl_seconds: 300,
        // Low floor so the short test prompts qualify for breakpoints.
        default_cache_min_tokens: 1,
        cost_sort_hit_rate: 0.8,
    }
}

fn provider(name: &str, uri: &str) -> Provider {
    Provider {
        name: name.to_string(),
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
    }
}

#[allow(clippy::too_many_arguments)]
fn model(
    provider_name: &str,
    model_name: &str,
    priority: u32,
    input_price: f64,
    output_price: f64,
    cache_read_price: Option<f64>,
    cache_creation_price: Option<f64>,
    cache_support: Option<PromptCacheSupport>,
) -> ProviderModel {
    ProviderModel {
        provider: provider_name.to_string(),
        model: model_name.to_string(),
        priority,
        cost_per_million_input_tokens: input_price,
        cost_per_million_output_tokens: output_price,
        cost_per_million_cache_read_input_tokens: cache_read_price,
        cost_per_million_cache_creation_input_tokens: cache_creation_price,
        cache_min_tokens: None,
        cache_support,
        structured_output_passthrough: None,
        tier: None,
        context_window: 0,
specializations: vec![],
cost_per_million_reasoning_tokens: None,
        reasoning_family: None,
        reasoning_parameter: None,
    }
}

/// A multi-turn chat request whose prefix (system + first exchange) stays
/// stable across calls — the newest user turn is the conversation tail.
/// `turn` distinguishes the tail so the exact-match response cache cannot
/// short-circuit successive turns (each request body hashes differently
/// while the prefix hash stays identical).
fn chat_request(group: &str, turn: &str) -> Body {
    Body::from(
        serde_json::to_string(&serde_json::json!({
            "model": group,
            "messages": [
                {"role": "system", "content": "You are terse but thorough."},
                {"role": "user", "content": "first question"},
                {"role": "assistant", "content": "first answer"},
                {"role": "user", "content": turn}
            ]
        }))
        .unwrap(),
    )
}

async fn post_chat(app: axum::Router, group: &str, turn: &str) -> (StatusCode, serde_json::Value) {
    let req = Request::post("/v1/chat/completions")
        .header("content-type", "application/json")
        .body(chat_request(group, turn))
        .unwrap();
    let resp = app.oneshot(req).await.unwrap();
    let status = resp.status();
    let body = axum::body::to_bytes(resp.into_body(), 1024 * 1024)
        .await
        .unwrap();
    (status, serde_json::from_slice(&body).unwrap_or(serde_json::Value::Null))
}

/// Mount a mock chat-completions endpoint returning `usage` JSON.
async fn mount_completion_mock(server: &MockServer, usage: serde_json::Value, expect: u64) {
    let body = serde_json::json!({
        "id": "chatcmpl-cache-test",
        "object": "chat.completion",
        "created": 1700000000_i64,
        "model": "gpt-4",
        "choices": [{
            "index": 0,
            "message": {"role": "assistant", "content": "ok"},
            "finish_reason": "stop"
        }],
        "usage": usage
    });
    Mock::given(wm_method("POST"))
        .and(wm_path("/v1/chat/completions"))
        .respond_with(ResponseTemplate::new(200).set_body_json(body))
        .expect(expect)
        .mount(server)
        .await;
}

fn plain_usage() -> serde_json::Value {
    serde_json::json!({"prompt_tokens": 10, "completion_tokens": 5, "total_tokens": 15})
}

// ---------------------------------------------------------------------------
// Req 1: prefix-hash sticky provider selection
// ---------------------------------------------------------------------------

/// Two requests with the same conversation prefix stay on the provider that
/// first served the prefix, even after a transient failure rebinds affinity
/// to the failover target and the original provider is healthy again.
/// Covers Req 1.1 (upsert on success) + 1.2 (promotion beats priority sort).
#[tokio::test]
async fn same_prefix_requests_stick_to_serving_provider() {
    let primary = MockServer::start().await; // priority 1
    let backup = MockServer::start().await; // priority 2

    let mut config = cache_config(cache_aware(true));
    config.providers = vec![provider("primary", &primary.uri()), provider("backup", &backup.uri())];
    config.model_groups = vec![ModelGroup {
        name: "cache-group".to_string(),
        version_fallback_enabled: false,
        compression: None,
        memory: None,
        structured_output: None,
        models: vec![
            model("primary", "m-primary", 1, 1.0, 1.0, None, None, None),
            model("backup", "m-backup", 2, 1.0, 1.0, None, None, None),
        ],
    }];
    common::isolate_databases(&mut config);
    let server = GatewayServer::new(config, None).await.unwrap();
    let app = server.build_router();

    // Turn 1: priority routing serves via `primary`. The success mock is
    // capped at one hit; afterwards matching falls through to the 500 mock.
    let ok_body = serde_json::json!({
        "id": "chatcmpl-cache-test",
        "object": "chat.completion",
        "created": 1700000000_i64,
        "model": "gpt-4",
        "choices": [{
            "index": 0,
            "message": {"role": "assistant", "content": "ok"},
            "finish_reason": "stop"
        }],
        "usage": {"prompt_tokens": 10, "completion_tokens": 5, "total_tokens": 15}
    });
    Mock::given(wm_method("POST"))
        .and(wm_path("/v1/chat/completions"))
        .respond_with(ResponseTemplate::new(200).set_body_json(ok_body))
        .up_to_n_times(1)
        .named("primary ok once")
        .mount(&primary)
        .await;
    // Turn 2: `primary` fails; the gateway fails over to `backup` and the
    // sticky entry rebinds to it.
    Mock::given(wm_method("POST"))
        .and(wm_path("/v1/chat/completions"))
        .respond_with(ResponseTemplate::new(500).set_body_string("upstream boom"))
        .expect(1)
        .named("primary 500 afterwards")
        .mount(&primary)
        .await;
    mount_completion_mock(&backup, plain_usage(), 2).await;
    let (status, _) = post_chat(app.clone(), "cache-group", "turn one question").await;
    assert_eq!(status, StatusCode::OK);

    let (status, _) = post_chat(app.clone(), "cache-group", "turn two question").await;
    assert_eq!(status, StatusCode::OK, "failover must succeed");

    // Turn 3: `primary` is eligible again (single failure never opened the
    // breaker) and priority alone would pick it — but sticky promotion keeps
    // the conversation on `backup` (Req 1.2). If promotion failed, the
    // request would hit `primary`'s exhausted 500 mock and error out.
    let (status, _) = post_chat(app, "cache-group", "turn three question").await;
    assert_eq!(status, StatusCode::OK);

    primary.verify().await;
    backup.verify().await;
}

// ---------------------------------------------------------------------------
// Req 3.1-3.4: cache-aware cost sort
// ---------------------------------------------------------------------------

/// With cache-aware routing enabled and hit_rate 0.8, the provider with a
/// 0.1x cache-read price ($0.30/M effective read) sorts ahead of the flat
/// $1.50/M provider despite equal priorities.
#[tokio::test]
async fn cache_aware_cost_sort_prefers_read_discounted_provider() {
    let discounted = MockServer::start().await;
    let flat = MockServer::start().await;

    let mut config = cache_config(cache_aware(true));
    config.providers = vec![
        provider("discounted", &discounted.uri()),
        provider("flat", &flat.uri()),
    ];
    config.model_groups = vec![ModelGroup {
        name: "cache-group".to_string(),
        version_fallback_enabled: false,
        compression: None,
        memory: None,
        structured_output: None,
        models: vec![
            model("discounted", "m-discounted", 1, 3.0, 0.0, Some(0.30), None, None),
            model("flat", "m-flat", 1, 1.50, 0.0, None, None, None),
        ],
    }];
    common::isolate_databases(&mut config);
    let server = GatewayServer::new(config, None).await.unwrap();
    let app = server.build_router();

    mount_completion_mock(&discounted, plain_usage(), 1).await;
    mount_completion_mock(&flat, plain_usage(), 0).await;
    let (status, _) = post_chat(app, "cache-group", "final turn question").await;
    assert_eq!(status, StatusCode::OK);

    discounted.verify().await;
    flat.verify().await;
}

/// Feature disabled: the flat $1.50/M provider wins on base price (old
/// behavior, Req 3.2 ordering unchanged).
#[tokio::test]
async fn disabled_cache_sort_keeps_base_price_order() {
    let discounted = MockServer::start().await;
    let flat = MockServer::start().await;

    let mut config = cache_config(cache_aware(false));
    config.providers = vec![
        provider("discounted", &discounted.uri()),
        provider("flat", &flat.uri()),
    ];
    config.model_groups = vec![ModelGroup {
        name: "cache-group".to_string(),
        version_fallback_enabled: false,
        compression: None,
        memory: None,
        structured_output: None,
        models: vec![
            model("discounted", "m-discounted", 1, 3.0, 0.0, Some(0.30), None, None),
            model("flat", "m-flat", 1, 1.50, 0.0, None, None, None),
        ],
    }];
    common::isolate_databases(&mut config);
    let server = GatewayServer::new(config, None).await.unwrap();
    let app = server.build_router();

    mount_completion_mock(&discounted, plain_usage(), 0).await;
    mount_completion_mock(&flat, plain_usage(), 1).await;
    let (status, _) = post_chat(app, "cache-group", "final turn question").await;
    assert_eq!(status, StatusCode::OK);

    discounted.verify().await;
    flat.verify().await;
}

// ---------------------------------------------------------------------------
// Req 2: cache-control breakpoint injection
// ---------------------------------------------------------------------------

/// An explicit-cache provider receives gateway-computed `cache_control`
/// ephemeral markers on the wire (Req 2.1/2.3).
#[tokio::test]
async fn explicit_cache_provider_receives_breakpoint_markers() {
    let upstream = MockServer::start().await;

    let mut config = cache_config(cache_aware(true));
    config.providers = vec![provider("anthropic-ish", &upstream.uri())];
    config.model_groups = vec![ModelGroup {
        name: "cache-group".to_string(),
        version_fallback_enabled: false,
        compression: None,
        memory: None,
        structured_output: None,
        models: vec![model(
            "anthropic-ish",
            "m-explicit",
            1,
            3.0,
            15.0,
            Some(0.30),
            Some(3.75),
            Some(PromptCacheSupport::Explicit {
                max_breakpoints: 4,
            }),
        )],
    }];
    common::isolate_databases(&mut config);
    let server = GatewayServer::new(config, None).await.unwrap();
    let app = server.build_router();

    mount_completion_mock(&upstream, plain_usage(), 1).await;
    let (status, _) = post_chat(app, "cache-group", "final turn question").await;
    assert_eq!(status, StatusCode::OK);

    let received = upstream.received_requests().await.unwrap();
    let body = String::from_utf8(received[0].body.clone()).unwrap();
    assert!(
        body.contains("cache_control"),
        "explicit provider must receive cache_control markers: {body}"
    );
    assert!(
        body.contains("ephemeral"),
        "markers must be ephemeral-typed: {body}"
    );
}

/// An automatic-cache provider never receives injected markers, and its
/// recorded cost reflects the cached-token discount (Req 3.6 + Req 2 no-op
/// for automatic support).
#[tokio::test]
async fn automatic_provider_gets_no_markers_and_discounted_cost() {
    let upstream = MockServer::start().await;

    let mut config = cache_config(cache_aware(true));
    config.providers = vec![provider("openai-ish", &upstream.uri())];
    config.model_groups = vec![ModelGroup {
        name: "cache-group".to_string(),
        version_fallback_enabled: false,
        compression: None,
        memory: None,
        structured_output: None,
        models: vec![model(
            "openai-ish",
            "m-auto",
            1,
            2.0,
            0.0,
            Some(1.0),
            None,
            Some(PromptCacheSupport::Automatic),
        )],
    }];
    common::isolate_databases(&mut config);
    let server = GatewayServer::new(config, None).await.unwrap();
    let app = server.build_router();

    // OpenAI shape: 1M prompt tokens, half served from cache at the 0.5x
    // read price → $1.50 recorded cost vs the $2.00 uncached baseline.
    let usage = serde_json::json!({
        "prompt_tokens": 1_000_000,
        "completion_tokens": 1_000,
        "total_tokens": 1_001_000,
        "prompt_tokens_details": {"cached_tokens": 500_000}
    });
    mount_completion_mock(&upstream, usage, 1).await;
    let (status, _) = post_chat(app, "cache-group", "final turn question").await;
    assert_eq!(status, StatusCode::OK);

    let received = upstream.received_requests().await.unwrap();
    let body = String::from_utf8(received[0].body.clone()).unwrap();
    assert!(
        !body.contains("cache_control"),
        "automatic provider must not receive markers: {body}"
    );

    let snapshot = server.state.metrics.snapshot();
    let cost = snapshot
        .cost_by_provider
        .iter()
        .find(|(p, _)| p == "openai-ish")
        .map(|(_, c)| *c)
        .expect("provider cost must be recorded");
    assert!(
        (cost - 1.50).abs() < 1e-9,
        "cached-token cost must be discounted to $1.50 (base $2.0, read $1.0, 50% cached), got {cost}"
    );
}

// ---------------------------------------------------------------------------
// Req 4: cache telemetry (metrics + SQLite log)
// ---------------------------------------------------------------------------

/// Anthropic-shaped usage: metrics snapshot carries the cache token split,
/// hit rate, and savings after the response (Req 4.2).
#[tokio::test]
async fn metrics_snapshot_contains_cache_telemetry() {
    let upstream = MockServer::start().await;

    let mut config = cache_config(cache_aware(true));
    config.providers = vec![provider("anthropic-ish", &upstream.uri())];
    config.model_groups = vec![ModelGroup {
        name: "cache-group".to_string(),
        version_fallback_enabled: false,
        compression: None,
        memory: None,
        structured_output: None,
        models: vec![model(
            "anthropic-ish",
            "m-auto",
            1,
            3.0,
            15.0,
            Some(0.30),
            Some(3.75),
            Some(PromptCacheSupport::Automatic),
        )],
    }];
    common::isolate_databases(&mut config);
    let server = GatewayServer::new(config, None).await.unwrap();
    let app = server.build_router();

    // 1M prompt tokens: 800k read at $0.30/M + 200k creation at $3.75/M
    // + 100k completion at $15/M → actual $2.49 vs $4.50 baseline → 201c
    // saved; hit rate 800k/1M = 0.8.
    let usage = serde_json::json!({
        "prompt_tokens": 1_000_000,
        "completion_tokens": 100_000,
        "total_tokens": 1_100_000,
        "cache_read_input_tokens": 800_000,
        "cache_creation_input_tokens": 200_000
    });
    mount_completion_mock(&upstream, usage, 1).await;
    let (status, _) = post_chat(app, "cache-group", "final turn question").await;
    assert_eq!(status, StatusCode::OK);

    let snapshot = server.state.metrics.snapshot();
    let health = snapshot
        .provider_health
        .iter()
        .find(|ph| ph.provider == "anthropic-ish")
        .expect("provider health must exist");
    assert_eq!(health.cache_read_tokens, 800_000);
    assert_eq!(health.cache_creation_tokens, 200_000);
    let hit_rate = health
        .cache_prompt_hit_rate
        .expect("hit rate must be computed once prompt tokens are recorded");
    assert!((hit_rate - 0.8).abs() < 1e-9, "hit rate {hit_rate}");
    assert_eq!(health.cache_savings_cents, 201);
}

/// The SQLite request log persists the cache token split, savings, and
/// prefix hash for responses that used the cache (Req 4.1/4.3).
#[tokio::test]
async fn sqlite_request_log_persists_cache_fields() {
    let upstream = MockServer::start().await;

    let mut config = cache_config(cache_aware(true));
    config.providers = vec![provider("anthropic-ish", &upstream.uri())];
    config.model_groups = vec![ModelGroup {
        name: "cache-group".to_string(),
        version_fallback_enabled: false,
        compression: None,
        memory: None,
        structured_output: None,
        models: vec![model(
            "anthropic-ish",
            "m-auto",
            1,
            3.0,
            15.0,
            Some(0.30),
            Some(3.75),
            Some(PromptCacheSupport::Automatic),
        )],
    }];
    common::isolate_databases(&mut config);
    let server = GatewayServer::new(config, None).await.unwrap();
    let app = server.build_router();

    let usage = serde_json::json!({
        "prompt_tokens": 1_000_000,
        "completion_tokens": 100_000,
        "total_tokens": 1_100_000,
        "cache_read_input_tokens": 800_000,
        "cache_creation_input_tokens": 200_000
    });
    mount_completion_mock(&upstream, usage, 1).await;
    let (status, _) = post_chat(app, "cache-group", "final turn question").await;
    assert_eq!(status, StatusCode::OK);

    // The logger drains an async writer queue — poll until the entry lands.
    let mut entry = None;
    for _ in 0..50 {
        let entries = server
            .state
            .logger
            .query(ai_gateway::logger::LogFilter {
                limit: Some(10),
                ..Default::default()
            })
            .unwrap();
        if let Some(found) = entries
            .iter()
            .find(|e| e.provider == "anthropic-ish" && e.cache_read_tokens.is_some())
        {
            entry = Some(found.clone());
            break;
        }
        tokio::time::sleep(std::time::Duration::from_millis(100)).await;
    }
    let entry = entry.expect("log entry with cache fields must be written");
    assert_eq!(entry.cache_read_tokens, Some(800_000));
    assert_eq!(entry.cache_creation_tokens, Some(200_000));
    assert_eq!(entry.cache_savings_cents, Some(201));
    let prefix_hash = entry.prefix_hash.expect("prefix hash must be logged");
    assert_eq!(prefix_hash.len(), 16);
    assert!(
        prefix_hash.chars().all(|c| c.is_ascii_hexdigit()),
        "prefix hash must be hex: {prefix_hash}"
    );
}
