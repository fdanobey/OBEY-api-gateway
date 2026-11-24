//! Integration tests for the response-output-caps feature (spec
//! `response-output-caps`, Task 6).
//!
//! Drives the full Axum router via `tower::ServiceExt::oneshot()` (no port
//! binding, per repo conventions) against a wiremock upstream standing in
//! for the LLM provider, with per-test isolated SQLite databases via
//! `common::isolate_databases` and virtual keys created through the
//! manager.
//!
//! Coverage:
//! - Clamp matrix: client value above the cap is clamped on all four
//!   front doors (chat, legacy completions, Anthropic Messages,
//!   Responses)
//! - Injection: absent field becomes typed `max_tokens` = cap
//! - Pass-through: below-cap and equal-to-cap values untouched
//! - Resolution: min(key cap, group cap) wins
//! - Optional enforcement without a key ignores key caps
//! - Zero caps: no field injected, forwarded value unchanged
//! - Group-cap hot-reload and admin PATCH apply without restart
//! - Observability: `gateway_output_caps_applied_total` counter and
//!   `output_cap_metadata` log column
//! - Truncation failover suppressed at the clamped value, kept below it

use ai_gateway::config::*;
use ai_gateway::gateway::{apply_runtime_config_update, GatewayServer};
use ai_gateway::logger::{LogEntry, LogFilter};
use ai_gateway::virtual_keys::models::{CreateKeyParams, UpdateKeyParams};
use axum::body::Body;
use axum::http::{Request, StatusCode};
use serde_json::{json, Value};
use std::time::Duration;
use tower::ServiceExt;
use wiremock::matchers::{method, path};
use wiremock::{Mock, MockServer, ResponseTemplate};

use crate::common;

const ANTHROPIC_VERSION: &str = "2023-06-01";
const TEST_MODEL: &str = "gpt-4o";

// ---------------------------------------------------------------------------
// Harness
// ---------------------------------------------------------------------------

struct TestServer {
    server: GatewayServer,
    upstream: MockServer,
}

fn single_provider_config() -> Config {
    test_config(None, "test-provider", 100)
}

fn two_provider_config() -> Config {
    let mut config = test_config(Some("test-provider-2"), "test-provider", 100);
    let template = config.providers[0].clone();
    let second = Provider {
        name: "test-provider-2".to_string(),
        base_url: None,
        ..template
    };
    config.providers.push(second);
    let mut backup_model = config.model_groups[0].models[0].clone();
    backup_model.provider = "test-provider-2".to_string();
    backup_model.priority = 200;
    config.model_groups[0].models.push(backup_model);
    config
}

fn test_config(
    second: Option<&str>,
    provider_name: &str,
    group_max_output_tokens_priority: u32,
) -> Config {
    let mut provider = Provider {
        name: provider_name.to_string(),
        provider_type: "openai".to_string(),
        base_url: None,
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
        disable_parallel_tool_calls: false,
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
        reasoning: true,
        codex_base_url_override: None,
        codex_model_override: None,
        instructions_override: None,
        max_rate_limit_cooldown_seconds: None,
        memory: None,
    };
    if second.is_some() {
        provider.name = provider_name.to_string();
    }
    Config {
        prompt_library: Default::default(),
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
        providers: vec![provider],
        model_groups: vec![ModelGroup {
            name: "test-group".to_string(),
            version_fallback_enabled: false,
            compression: None,
            structured_output: None,
            memory: None,
            max_output_tokens: None,
            models: vec![ProviderModel {
                cache_support: None,
                cache_min_tokens: None,
                cost_per_million_cache_read_input_tokens: None,
                cost_per_million_cache_creation_input_tokens: None,
                provider: provider_name.to_string(),
                model: TEST_MODEL.to_string(),
                cost_per_million_input_tokens: 30.0,
                cost_per_million_output_tokens: 60.0,
                priority: group_max_output_tokens_priority,
                structured_output_passthrough: None,
                tier: None,
                context_window: 0,
                specializations: vec![],
                cost_per_million_reasoning_tokens: None,
                reasoning_family: None,
                reasoning_parameter: None,
                deprecation_date: None,
                replacement_model: None,
            }],
        }],
        circuit_breaker: CircuitBreakerConfig::default(),
        retry: RetryConfig::default(),
        logging: LoggingConfig::default(),
        semantic_cache: None,
        exact_cache: ExactCacheConfig::default(),
        prometheus: None,
        context: ContextConfig::default(),
        compression: Default::default(),
        first_launch_completed: false,
        tray: TrayConfig::default(),
        codex_instructions_url: None,
        streaming: None,
        virtual_keys: VirtualKeysConfig {
            enforcement: EnforcementMode::Disabled,
            database_path: String::new(),
        },
        loop_detection: Default::default(),
        guardrails: None,
        tool_compression: Default::default(),
        smart_routing: Default::default(),
        xhigh_models_allowlist: Default::default(),
        reasoning_models_allowlist: Default::default(),
        codex_search: None,
        web_search: None,
        structured_output: None,
        memory: None,
        reasoning_compat: Default::default(),
        webhooks: Default::default(),
        opentelemetry: Default::default(),
        model_deprecations: Default::default(),
        performance_alerts: Default::default(),
    }
}

/// Mount a static chat completion (given content, finish reason and
/// completion tokens) on the upstream mock.
async fn mount_completion(
    upstream: MockServer,
    content: &str,
    finish_reason: &str,
    completion_tokens: u64,
) -> MockServer {
    let body = json!({
        "id": "chatcmpl-output-caps",
        "object": "chat.completion",
        "created": 1_700_000_000_i64,
        "model": "gpt-4o",
        "choices": [{
            "index": 0,
            "message": {"role": "assistant", "content": content},
            "finish_reason": finish_reason
        }],
        "usage": {
            "prompt_tokens": 4,
            "completion_tokens": completion_tokens,
            "total_tokens": (4 + completion_tokens)
        }
    });
    Mock::given(method("POST"))
        .and(path("/v1/chat/completions"))
        .respond_with(ResponseTemplate::new(200).set_body_json(body))
        .mount(&upstream)
        .await;
    upstream
}

impl TestServer {
    async fn build(
        mut config: Config,
        upstream: MockServer,
        enforcement: EnforcementMode,
        metrics: bool,
    ) -> Self {
        config.providers[0].base_url = Some(upstream.uri());
        config.virtual_keys.enforcement = enforcement;
        if metrics {
            config.prometheus = Some(PrometheusConfig {
                enabled: true,
                path: "/metrics".to_string(),
            });
        }
        common::isolate_databases(&mut config);
        // The exact-response cache would serve the second identical request
        // from memory, hiding the cap change on the wire.
        config.exact_cache.enabled = false;
        let server = GatewayServer::new(config, None).await.unwrap();
        Self { server, upstream }
    }

    async fn new(enforcement: EnforcementMode) -> Self {
        Self::build(
            single_provider_config(),
            mount_completion(MockServer::start().await, "mock answer", "stop", 3).await,
            enforcement,
            false,
        )
        .await
    }

    async fn new_with_metrics(enforcement: EnforcementMode) -> Self {
        Self::build(
            single_provider_config(),
            mount_completion(MockServer::start().await, "mock answer", "stop", 3).await,
            enforcement,
            true,
        )
        .await
    }

    fn router(&self) -> axum::Router {
        self.server.build_router()
    }
}

/// Two-provider failover server (p1 priority 100 tried first, p2 priority
/// 200) with an optional group cap and enforcement disabled.
async fn two_provider_server(
    group_cap: Option<u32>,
    primary: &MockServer,
    backup: &MockServer,
) -> GatewayServer {
    let mut config = two_provider_config();
    config.providers[0].base_url = Some(primary.uri());
    config.providers[1].base_url = Some(backup.uri());
    config.model_groups[0].max_output_tokens = group_cap;
    config.virtual_keys.enforcement = EnforcementMode::Disabled;
    common::isolate_databases(&mut config);
    GatewayServer::new(config, None).await.unwrap()
}

fn key_params(cap: Option<u32>) -> CreateKeyParams {
    CreateKeyParams {
        name: Some("output-caps-test".to_string()),
        budget_limit_usd: None,
        token_budget: None,
        budget_window: None,
        requests_per_minute: None,
        tokens_per_minute: None,
        max_output_tokens: cap,
        model_access: None,
        expires_in: None,
        loop_detection: None,
        priority: None,
    }
}

fn chat_body(max_tokens: u64) -> Value {
    json!({
        "model": TEST_MODEL,
        "messages": [{"role": "user", "content": "hello"}],
        "max_tokens": max_tokens
    })
}

async fn send(
    app: axum::Router,
    method: &str,
    uri: &str,
    bearer: Option<&str>,
    headers: &[(&str, &str)],
    body: Value,
) -> (StatusCode, Value) {
    let mut builder = Request::builder()
        .method(method)
        .uri(uri)
        .header("content-type", "application/json");
    if let Some(token) = bearer {
        builder = builder.header("authorization", format!("Bearer {token}"));
    }
    for (name, value) in headers {
        builder = builder.header(*name, *value);
    }
    let response = app
        .oneshot(builder.body(Body::from(body.to_string())).unwrap())
        .await
        .unwrap();
    let status = response.status();
    let bytes = axum::body::to_bytes(response.into_body(), 4 * 1024 * 1024)
        .await
        .unwrap();
    let json = serde_json::from_slice(&bytes).unwrap_or(Value::Null);
    (status, json)
}

/// Bodies of the chat-completion requests captured by the mock upstream.
async fn upstream_bodies(upstream: &MockServer) -> Vec<Value> {
    upstream
        .received_requests()
        .await
        .unwrap_or_default()
        .iter()
        .map(|request| serde_json::from_slice(&request.body).unwrap())
        .collect()
}

/// Poll the request log until `done` holds or 5 s pass.
async fn poll_logs(
    server: &GatewayServer,
    done: impl Fn(&[LogEntry]) -> bool,
) -> Vec<LogEntry> {
    let deadline = tokio::time::Instant::now() + Duration::from_secs(5);
    loop {
        let rows = server
            .state
            .logger
            .query(LogFilter::default())
            .unwrap_or_default();
        if done(&rows) || tokio::time::Instant::now() >= deadline {
            return rows;
        }
        tokio::time::sleep(Duration::from_millis(50)).await;
    }
}

// ---------------------------------------------------------------------------
// 1. Chat door: client value above the key cap is clamped
// ---------------------------------------------------------------------------

#[tokio::test]
async fn chat_clamps_client_max_tokens_to_key_cap() {
    let test = TestServer::new(EnforcementMode::Required).await;
    let created = test
        .server
        .state
        .virtual_key_manager
        .create_key(key_params(Some(200)))
        .await
        .unwrap();

    let (status, body) = send(
        test.router(),
        "POST",
        "/v1/chat/completions",
        Some(&created.key),
        &[],
        chat_body(5000),
    )
    .await;

    assert_eq!(status, StatusCode::OK, "body: {body}");
    let upstream = upstream_bodies(&test.upstream).await;
    assert_eq!(upstream.len(), 1);
    assert_eq!(
        upstream[0]["max_tokens"],
        json!(200),
        "upstream body must receive the clamped cap"
    );
}

// ---------------------------------------------------------------------------
// 2. Legacy completions door: clamped via delegation
// ---------------------------------------------------------------------------

#[tokio::test]
async fn legacy_completions_clamps_client_max_tokens_to_key_cap() {
    let test = TestServer::new(EnforcementMode::Required).await;
    let created = test
        .server
        .state
        .virtual_key_manager
        .create_key(key_params(Some(150)))
        .await
        .unwrap();

    let body = json!({
        "model": TEST_MODEL,
        "messages": [{"role": "user", "content": "hello"}],
        "max_tokens": 3000
    });
    let (status, body) = send(
        test.router(),
        "POST",
        "/v1/completions",
        Some(&created.key),
        &[],
        body,
    )
    .await;

    assert_eq!(status, StatusCode::OK, "body: {body}");
    let upstream = upstream_bodies(&test.upstream).await;
    assert_eq!(upstream.len(), 1);
    assert_eq!(
        upstream[0]["max_tokens"],
        json!(150),
        "legacy completions body must receive the clamped cap"
    );
}

// ---------------------------------------------------------------------------
// 3. Anthropic door: translated request clamped after translation
// ---------------------------------------------------------------------------

#[tokio::test]
async fn anthropic_clamps_translated_max_tokens_to_key_cap() {
    let test = TestServer::new(EnforcementMode::Required).await;
    let created = test
        .server
        .state
        .virtual_key_manager
        .create_key(key_params(Some(256)))
        .await
        .unwrap();

    let body = json!({
        "model": TEST_MODEL,
        "max_tokens": 4096,
        "messages": [{"role": "user", "content": "Hello there"}]
    });
    let (status, body) = send(
        test.router(),
        "POST",
        "/v1/messages",
        Some(&created.key),
        &[("anthropic-version", ANTHROPIC_VERSION)],
        body,
    )
    .await;

    assert_eq!(status, StatusCode::OK, "body: {body}");
    let upstream = upstream_bodies(&test.upstream).await;
    assert_eq!(upstream.len(), 1);
    assert_eq!(
        upstream[0]["max_tokens"],
        json!(256),
        "Anthropic translation must forward the clamped cap"
    );
}

// ---------------------------------------------------------------------------
// 4. Responses door (non-Codex): clamped after translation
// ---------------------------------------------------------------------------

#[tokio::test]
async fn responses_clamps_max_output_tokens_after_translation() {
    let test = TestServer::new(EnforcementMode::Required).await;
    let created = test
        .server
        .state
        .virtual_key_manager
        .create_key(key_params(Some(300)))
        .await
        .unwrap();

    let body = json!({
        "model": TEST_MODEL,
        "input": "hello",
        "max_output_tokens": 9000
    });
    let (status, body) = send(
        test.router(),
        "POST",
        "/v1/responses",
        Some(&created.key),
        &[],
        body,
    )
    .await;

    assert_eq!(status, StatusCode::OK, "body: {body}");
    let upstream = upstream_bodies(&test.upstream).await;
    assert_eq!(upstream.len(), 1);
    assert_eq!(
        upstream[0]["max_tokens"],
        json!(300),
        "Responses translation must forward the clamped cap"
    );
    assert_eq!(
        upstream[0]["max_output_tokens"],
        Value::Null,
        "the duplicated max_tokens extra carrier must be removed"
    );
}

// ---------------------------------------------------------------------------
// 5. Absent field is injected (typed max_tokens = cap)
// ---------------------------------------------------------------------------

#[tokio::test]
async fn absent_max_tokens_is_injected_as_cap() {
    let test = TestServer::new(EnforcementMode::Required).await;
    let created = test
        .server
        .state
        .virtual_key_manager
        .create_key(key_params(Some(512)))
        .await
        .unwrap();

    let body = json!({
        "model": TEST_MODEL,
        "messages": [{"role": "user", "content": "hello"}]
    });
    let (status, body) = send(
        test.router(),
        "POST",
        "/v1/chat/completions",
        Some(&created.key),
        &[],
        body,
    )
    .await;

    assert_eq!(status, StatusCode::OK, "body: {body}");
    let upstream = upstream_bodies(&test.upstream).await;
    assert_eq!(upstream.len(), 1);
    assert_eq!(
        upstream[0]["max_tokens"],
        json!(512),
        "an absent carrier must be injected with the cap"
    );
}

// ---------------------------------------------------------------------------
// 6. Below-cap and equal-to-cap values are untouched
// ---------------------------------------------------------------------------

#[tokio::test]
async fn below_cap_value_is_untouched() {
    let test = TestServer::new(EnforcementMode::Required).await;
    let created = test
        .server
        .state
        .virtual_key_manager
        .create_key(key_params(Some(1000)))
        .await
        .unwrap();

    let (status, body) = send(
        test.router(),
        "POST",
        "/v1/chat/completions",
        Some(&created.key),
        &[],
        chat_body(64),
    )
    .await;

    assert_eq!(status, StatusCode::OK, "body: {body}");
    let upstream = upstream_bodies(&test.upstream).await;
    assert_eq!(
        upstream[0]["max_tokens"],
        json!(64),
        "a below-cap value must never be raised"
    );
}

#[tokio::test]
async fn equal_to_cap_value_is_untouched() {
    let test = TestServer::new(EnforcementMode::Required).await;
    let created = test
        .server
        .state
        .virtual_key_manager
        .create_key(key_params(Some(777)))
        .await
        .unwrap();

    let (status, body) = send(
        test.router(),
        "POST",
        "/v1/chat/completions",
        Some(&created.key),
        &[],
        chat_body(777),
    )
    .await;

    assert_eq!(status, StatusCode::OK, "body: {body}");
    let upstream = upstream_bodies(&test.upstream).await;
    assert_eq!(
        upstream[0]["max_tokens"],
        json!(777),
        "a value equal to the cap must be forwarded unchanged"
    );
}

// ---------------------------------------------------------------------------
// 7. Key cap + group cap: the minimum wins
// ---------------------------------------------------------------------------

#[tokio::test]
async fn key_and_group_caps_use_the_minimum() {
    // Key cap 900 < group cap 500? No: min(900, 500) = 500.
    let mut config = single_provider_config();
    config.model_groups[0].max_output_tokens = Some(500);
    let test = TestServer::build(
        config,
        mount_completion(MockServer::start().await, "mock answer", "stop", 3).await,
        EnforcementMode::Required,
        false,
    )
    .await;
    let created = test
        .server
        .state
        .virtual_key_manager
        .create_key(key_params(Some(900)))
        .await
        .unwrap();

    let (status, body) = send(
        test.router(),
        "POST",
        "/v1/chat/completions",
        Some(&created.key),
        &[],
        chat_body(5000),
    )
    .await;

    assert_eq!(status, StatusCode::OK, "body: {body}");
    let upstream = upstream_bodies(&test.upstream).await;
    assert_eq!(
        upstream[0]["max_tokens"],
        json!(500),
        "the group cap must win when it is lower than the key cap"
    );
}

// ---------------------------------------------------------------------------
// 8. Optional enforcement without a key: key cap ignored
// ---------------------------------------------------------------------------

#[tokio::test]
async fn optional_enforcement_without_key_ignores_key_cap() {
    let test = TestServer::new(EnforcementMode::Optional).await;
    test.server
        .state
        .virtual_key_manager
        .create_key(key_params(Some(128)))
        .await
        .unwrap();

    let (status, body) = send(
        test.router(),
        "POST",
        "/v1/chat/completions",
        None,
        &[],
        chat_body(9000),
    )
    .await;

    assert_eq!(status, StatusCode::OK, "body: {body}");
    let upstream = upstream_bodies(&test.upstream).await;
    assert_eq!(
        upstream[0]["max_tokens"],
        json!(9000),
        "a bearer-free request must forward the client value untouched"
    );
}

// ---------------------------------------------------------------------------
// 9. Zero caps configured: forwarded body byte-identical
// ---------------------------------------------------------------------------

#[tokio::test]
async fn no_caps_configured_forwards_body_untouched() {
    let test = TestServer::new(EnforcementMode::Optional).await;

    let (status, body) = send(
        test.router(),
        "POST",
        "/v1/chat/completions",
        None,
        &[],
        chat_body(9000),
    )
    .await;

    assert_eq!(status, StatusCode::OK, "body: {body}");
    let upstream = upstream_bodies(&test.upstream).await;
    assert_eq!(
        upstream[0]["max_tokens"],
        json!(9000),
        "with no caps configured the client value must pass through unchanged"
    );
    assert_eq!(
        upstream[0].get("max_output_tokens"),
        None,
        "no carrier may be injected when the feature is off"
    );
}

// ---------------------------------------------------------------------------
// 10. Group-cap hot-reload applies without restart
// ---------------------------------------------------------------------------

#[tokio::test]
async fn group_cap_hot_reload_applies_without_restart() {
    let upstream =
        mount_completion(MockServer::start().await, "mock answer", "stop", 3).await;
    let mut config = single_provider_config();
    config.providers[0].base_url = Some(upstream.uri());
    config.virtual_keys.enforcement = EnforcementMode::Disabled;
    common::isolate_databases(&mut config);
    let server = GatewayServer::new(config, None).await.unwrap();

    // First request: no cap configured, client value passes through.
    let (status, body) = send(
        server.build_router(),
        "POST",
        "/v1/chat/completions",
        None,
        &[],
        chat_body(9000),
    )
    .await;
    assert_eq!(status, StatusCode::OK, "body: {body}");

    // Reload with a group cap of 111.
    let mut reloaded = server.state.config.read().await.clone();
    reloaded.model_groups[0].max_output_tokens = Some(111);
    apply_runtime_config_update(&server.state, reloaded).await;

    // Next request: clamped to the reloaded cap, same server, no restart.
    let (status, body) = send(
        server.build_router(),
        "POST",
        "/v1/chat/completions",
        None,
        &[],
        chat_body(9000),
    )
    .await;
    assert_eq!(status, StatusCode::OK, "body: {body}");

    let upstream = upstream_bodies(&upstream).await;
    assert_eq!(upstream.len(), 2);
    assert_eq!(
        upstream[0]["max_tokens"],
        json!(9000),
        "before the reload the client value must pass through"
    );
    assert_eq!(
        upstream[1]["max_tokens"],
        json!(111),
        "after the reload the new group cap must apply without restart"
    );
}

// ---------------------------------------------------------------------------
// 11. Admin PATCH of the key cap applies on the next request
// ---------------------------------------------------------------------------

#[tokio::test]
async fn admin_patch_updates_key_cap_and_clears_it() {
    let test = TestServer::new(EnforcementMode::Required).await;
    let created = test
        .server
        .state
        .virtual_key_manager
        .create_key(key_params(Some(400)))
        .await
        .unwrap();
    let url = format!("/admin/keys/{}", created.id);

    // PATCH to 222; the next request must clamp to the new cap.
    let patch = serde_json::to_value(UpdateKeyParams {
        max_output_tokens: Some(Some(222)),
        ..Default::default()
    })
    .unwrap();
    let (status, patched) = send(test.router(), "PATCH", &url, None, &[], patch).await;
    assert_eq!(status, StatusCode::OK, "body: {patched}");

    let (status, body) = send(
        test.router(),
        "POST",
        "/v1/chat/completions",
        Some(&created.key),
        &[],
        chat_body(8000),
    )
    .await;
    assert_eq!(status, StatusCode::OK, "body: {body}");
    assert_eq!(
        upstream_bodies(&test.upstream).await[0]["max_tokens"],
        json!(222),
        "PATCHed cap must apply on the next request without restart"
    );

    // Clearing the cap (Some(None) serializes as null) removes the ceiling.
    let clear = serde_json::to_value(UpdateKeyParams {
        max_output_tokens: Some(None),
        ..Default::default()
    })
    .unwrap();
    let (status, cleared) = send(test.router(), "PATCH", &url, None, &[], clear).await;
    assert_eq!(status, StatusCode::OK, "body: {cleared}");

    let (status, body) = send(
        test.router(),
        "POST",
        "/v1/chat/completions",
        Some(&created.key),
        &[],
        chat_body(8000),
    )
    .await;
    assert_eq!(status, StatusCode::OK, "body: {body}");
    assert_eq!(
        upstream_bodies(&test.upstream).await[1]["max_tokens"],
        json!(8000),
        "a cleared cap must forward the client value untouched"
    );
}

// ---------------------------------------------------------------------------
// 12. Observability: counter increments and log metadata lands
// ---------------------------------------------------------------------------

#[tokio::test]
async fn capped_request_increments_counter_and_logs_metadata() {
    let trace_id = "output-caps-trace-1";
    let test = TestServer::new_with_metrics(EnforcementMode::Required).await;
    let created = test
        .server
        .state
        .virtual_key_manager
        .create_key(key_params(Some(200)))
        .await
        .unwrap();

    let builder = Request::builder()
        .method("POST")
        .uri("/v1/chat/completions")
        .header("content-type", "application/json")
        .header("authorization", format!("Bearer {}", created.key))
        .header("x-request-id", trace_id);
    let response = test
        .router()
        .oneshot(
            builder
                .body(Body::from(
                    serde_json::to_string(&chat_body(5000)).unwrap(),
                ))
                .unwrap(),
        )
        .await
        .unwrap();
    assert_eq!(response.status(), StatusCode::OK);

    // The log row carries the content-free application metadata.
    let rows = poll_logs(&test.server, |rows| {
        rows.iter().any(|row| {
            row.trace_id == trace_id && row.output_cap_metadata.is_some()
        })
    })
    .await;
    let row = rows
        .iter()
        .find(|row| row.trace_id == trace_id)
        .expect("log row for the capped request");
    let metadata = row.output_cap_metadata.as_ref().unwrap();
    assert_eq!(metadata["applied_value"], json!(200));
    let originals = metadata["originals"].as_array().unwrap();
    assert_eq!(originals.len(), 1);
    assert_eq!(originals[0][0], json!("max_tokens"));
    assert_eq!(originals[0][1], json!(5000));

    // The Prometheus counter increments for this trace id.
    let req = Request::builder()
        .method("GET")
        .uri("/metrics")
        .body(Body::empty())
        .unwrap();
    let response = test.router().oneshot(req).await.unwrap();
    assert_eq!(response.status(), StatusCode::OK);
    let bytes = axum::body::to_bytes(response.into_body(), 4 * 1024 * 1024)
        .await
        .unwrap();
    let text = String::from_utf8(bytes.to_vec()).unwrap();
    assert!(
        text.contains(&format!(
            "gateway_output_caps_applied_total{{trace_id=\"{trace_id}\"}} 1"
        )),
        "expected the output-caps counter for {trace_id} in the metrics body"
    );
}

// ---------------------------------------------------------------------------
// 13. Failover suppression: length finish at the clamped cap
// ---------------------------------------------------------------------------

/// A `finish_reason: "length"` completion at the clamped cap must NOT
/// trigger suspicious-truncation failover: the client gets the primary's
/// response, and the backup is never consulted.
#[tokio::test]
async fn length_finish_at_clamped_cap_does_not_fail_over() {
    let primary = mount_completion(MockServer::start().await, "PRIMARY_CAPPED", "length", 128)
        .await;
    let backup = mount_completion(
        MockServer::start().await,
        "BACKUP_SHOULD_NOT_APPEAR",
        "stop",
        42,
    )
    .await;
    let server = two_provider_server(Some(128), &primary, &backup).await;

    let (status, body) = send(
        server.build_router(),
        "POST",
        "/v1/chat/completions",
        None,
        &[],
        chat_body(4096),
    )
    .await;

    assert_eq!(status, StatusCode::OK, "body: {body}");
    assert_eq!(
        body["choices"][0]["message"]["content"],
        json!("PRIMARY_CAPPED"),
        "a length finish at the clamped cap must not fail over"
    );
    assert_eq!(
        body["choices"][0]["finish_reason"],
        json!("length"),
        "the dialect-native finish_reason must reach the client"
    );
    assert_eq!(body["usage"]["completion_tokens"], json!(128));
    assert!(
        backup.received_requests().await.unwrap().is_empty(),
        "the backup provider must not be consulted"
    );
}

/// A `finish_reason: "length"` completion well below the clamped cap keeps
/// the existing suspicious-truncation behavior: failover to the backup.
#[tokio::test]
async fn length_finish_below_clamped_cap_still_fails_over() {
    let primary = mount_completion(MockServer::start().await, "PRIMARY_SHORT", "length", 5).await;
    let backup = mount_completion(
        MockServer::start().await,
        "BACKUP_COMPLETE",
        "stop",
        42,
    )
    .await;
    let server = two_provider_server(Some(128), &primary, &backup).await;

    let (status, body) = send(
        server.build_router(),
        "POST",
        "/v1/chat/completions",
        None,
        &[],
        chat_body(4096),
    )
    .await;

    assert_eq!(status, StatusCode::OK, "body: {body}");
    assert_eq!(
        body["choices"][0]["message"]["content"],
        json!("BACKUP_COMPLETE"),
        "a length finish far below the clamped cap must still fail over"
    );
    assert_eq!(
        body["choices"][0]["finish_reason"],
        json!("stop"),
        "the client must receive the backup's complete response"
    );
}
