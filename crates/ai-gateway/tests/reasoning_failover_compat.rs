//! Integration tests for the reasoning-state failover compatibility
//! feature (spec: reasoning-failover-compat, design "Integration Testing").
//!
//! Exercises the full gateway HTTP surface via `tower::ServiceExt::oneshot()`
//! with wiremock providers â€” no real ports bound, isolated SQLite databases
//! (same harness conventions as `tests/prompt_cache_routing.rs`).
//!
//! Covered scenarios:
//! - Claude-thinking conversation fails over to another Claude model â†’
//!   thinking / signature / redacted_thinking stripped from the backup body
//! - Same-model continuation (tool loop) â†’ thinking preserved verbatim
//! - Sticky conversation-model affinity keeps the serving provider ahead of
//!   a higher-priority competitor and preserves same-model thinking
//! - DeepSeek `reasoning_content` history â†’ stripped for an OpenAI target
//! - `reasoning_effort` normalization: Anthropic manual budget, Anthropic
//!   adaptive, OpenAI-reasoning passthrough (name-based family classify)
//! - Bedrock `budget_tokens` replaces the legacy 4096 hardcode
//! - `enabled: false` reproduces exact passthrough (carriers not stripped)
//! - Thinking-validation 400 triggers one aggressive strip + same-provider
//!   retry
//! - Reasoning-token usage lands in provider metrics and the SQLite log

use axum::body::Body;
use axum::http::{Request, StatusCode};
use tower::ServiceExt;
use wiremock::matchers::{method as wm_method, path as wm_path};
use wiremock::{Mock, MockServer, ResponseTemplate};

use ai_gateway::config::*;
use ai_gateway::gateway::GatewayServer;
use ai_gateway::reasoning_compat::config::ReasoningFamily;
use ai_gateway::reasoning_compat::ReasoningCompatConfig;

mod common;

// ---------------------------------------------------------------------------
// Harness
// ---------------------------------------------------------------------------

/// Minimal valid Config with reasoning-compat settings and per-provider
/// retry budget. Cache-aware routing stays disabled but keeps a positive
/// stickiness TTL so the conversation-model-affinity feature (which rides
/// the same sticky cache) is live.
fn compat_config(compat: ReasoningCompatConfig, max_retries: u32) -> Config {
    Config {
        cache_aware_routing: CacheAwareRouting {
            enabled: false,
            stickiness_ttl_seconds: 300,
            default_cache_min_tokens: 1,
            cost_sort_hit_rate: 0.8,
        },
        reasoning_compat: compat,
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
            max_retries_per_provider: max_retries,
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
    }
}

/// OpenAI-compatible mock provider (passthrough sanitize behavior).
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

/// Bedrock-flavored mock provider. No API key (so the gateway honors the
/// configured base_url instead of the real Bedrock Mantle endpoint) and no
/// inference-profile flags (so the model id survives unchanged).
fn bedrock_provider(name: &str, uri: &str) -> Provider {
    Provider {
        provider_type: "bedrock".to_string(),
        region: Some("us-east-1".to_string()),
        ..provider(name, uri)
    }
}

#[allow(clippy::too_many_arguments)]
fn reasoning_model(
    provider_name: &str,
    model_name: &str,
    priority: u32,
    family: Option<ReasoningFamily>,
    reasoning_price: Option<f64>,
) -> ProviderModel {
    ProviderModel {
        provider: provider_name.to_string(),
        model: model_name.to_string(),
        priority,
        cost_per_million_input_tokens: 1.0,
        cost_per_million_output_tokens: 1.0,
        cost_per_million_cache_read_input_tokens: None,
        cost_per_million_cache_creation_input_tokens: None,
        cache_min_tokens: None,
        cache_support: None,
        structured_output_passthrough: None,
        tier: None,
        context_window: 0,
        specializations: vec![],
        cost_per_million_reasoning_tokens: reasoning_price,
        reasoning_family: family,
        reasoning_parameter: None,
    }
}

fn group(name: &str, models: Vec<ProviderModel>) -> ModelGroup {
    ModelGroup {
        name: name.to_string(),
        version_fallback_enabled: false,
        compression: None,
        memory: None,
        structured_output: None,
        models,
    }
}

fn plain_usage() -> serde_json::Value {
    serde_json::json!({"prompt_tokens": 10, "completion_tokens": 5, "total_tokens": 15})
}

/// Mount a chat-completions success mock expecting exactly `expect` hits.
async fn mount_success(server: &MockServer, usage: serde_json::Value, expect: u64) {
    let body = serde_json::json!({
        "id": "chatcmpl-compat-test",
        "object": "chat.completion",
        "created": 1700000000_i64,
        "model": "test-model",
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

/// Mount a one-shot 500 (mounted before the success mock so the first hit
/// fails and the gateway fails over).
async fn mount_500_once(server: &MockServer) {
    Mock::given(wm_method("POST"))
        .and(wm_path("/v1/chat/completions"))
        .respond_with(ResponseTemplate::new(500).set_body_string("upstream boom"))
        .up_to_n_times(1)
        .mount(server)
        .await;
}

/// Mount a one-shot Anthropic-style thinking-validation 400.
async fn mount_thinking_400_once(server: &MockServer) {
    Mock::given(wm_method("POST"))
        .and(wm_path("/v1/chat/completions"))
        .respond_with(ResponseTemplate::new(400).set_body_json(
            serde_json::json!({"error": {"type": "invalid_request_error",
                "message": "thinking.budget_tokens must be less than max_tokens"}}),
        ))
        .up_to_n_times(1)
        .mount(server)
        .await;
}

/// POST /v1/chat/completions with an arbitrary JSON body (needed for
/// thinking-block / reasoning_content message histories).
async fn post_chat_json(
    app: axum::Router,
    body: serde_json::Value,
) -> (StatusCode, serde_json::Value) {
    let req = Request::post("/v1/chat/completions")
        .header("content-type", "application/json")
        .body(Body::from(body.to_string()))
        .unwrap();
    let resp = app.oneshot(req).await.unwrap();
    let status = resp.status();
    let bytes = axum::body::to_bytes(resp.into_body(), 1024 * 1024)
        .await
        .unwrap();
    (status, serde_json::from_slice(&bytes).unwrap_or(serde_json::Value::Null))
}

/// JSON bodies the mock provider received, in order.
async fn received_bodies(server: &MockServer) -> Vec<serde_json::Value> {
    server
        .received_requests()
        .await
        .unwrap_or_default()
        .iter()
        .map(|req| serde_json::from_slice(&req.body).expect("provider body is valid JSON"))
        .collect()
}

/// Assistant history carrying signed `thinking` + `redacted_thinking`
/// blocks (Anthropic manual-family carriers) followed by a new user turn.
fn thinking_history_request(tail: &str) -> serde_json::Value {
    serde_json::json!({
        "model": "compat-group",
        "max_tokens": 20000,
        "messages": [
            {"role": "system", "content": "You are terse."},
            {"role": "user", "content": "first question"},
            {"role": "assistant", "content": [
                {"type": "thinking", "thinking": "deep internal deliberation", "signature": "sig-1234"},
                {"type": "redacted_thinking", "data": "opaque"},
                {"type": "text", "text": "answer"}
            ]},
            {"role": "user", "content": tail}
        ]
    })
}

/// DeepSeek-style history: assistant turn with a top-level
/// `reasoning_content` field.
fn deepseek_history_request() -> serde_json::Value {
    serde_json::json!({
        "model": "compat-group",
        "max_tokens": 20000,
        "messages": [
            {"role": "user", "content": "first question"},
            {"role": "assistant", "content": "answer", "reasoning_content": "hidden coy secret"},
            {"role": "user", "content": "follow up"}
        ]
    })
}

/// Two-turn conversation sharing the prefix
/// `[system, user, assistant(thinking)]` across calls â€” only the final
/// user turn differs, so the sticky prefix hash stays identical.
fn affinity_request(tail: &str) -> serde_json::Value {
    serde_json::json!({
        "model": "compat-group",
        "max_tokens": 20000,
        "messages": [
            {"role": "system", "content": "You are terse."},
            {"role": "user", "content": "affinity seed question"},
            {"role": "assistant", "content": [
                {"type": "thinking", "thinking": "deep internal deliberation", "signature": "sig-1234"},
                {"type": "text", "text": "answer"}
            ]},
            {"role": "user", "content": tail}
        ]
    })
}

fn effort_request(effort: &str) -> serde_json::Value {
    serde_json::json!({
        "model": "compat-group",
        "max_tokens": 20000,
        "temperature": 0.7,
        "top_p": 0.9,
        "reasoning_effort": effort,
        "messages": [{"role": "user", "content": "solve it"}]
    })
}

// ---------------------------------------------------------------------------
// Cross-model failover strips reasoning state
// ---------------------------------------------------------------------------

/// Signed thinking + redacted_thinking history, manual-family primary 500s,
/// adaptive-family backup serves: the backup body must carry NO thinking
/// carriers (Anthropic reasoning state is signed and model-bound), and the
/// strip decision must land in the SQLite log telemetry.
#[tokio::test]
async fn cross_model_failover_strips_thinking() {
    let primary = MockServer::start().await; // claude-4-5 (manual), priority 1
    let backup = MockServer::start().await; // claude-4-7 (adaptive), priority 2

    let mut config = compat_config(ReasoningCompatConfig::default(), 0);
    config.providers = vec![
        provider("manual-primary", &primary.uri()),
        provider("adaptive-backup", &backup.uri()),
    ];
    config.model_groups = vec![group(
        "compat-group",
        vec![
            reasoning_model(
                "manual-primary",
                "claude-4-5-sonnet",
                1,
                Some(ReasoningFamily::AnthropicManual),
                None,
            ),
            reasoning_model(
                "adaptive-backup",
                "claude-4-7-sonnet",
                2,
                Some(ReasoningFamily::AnthropicAdaptive),
                None,
            ),
        ],
    )];
    common::isolate_databases(&mut config);
    let server = GatewayServer::new(config, None).await.unwrap();
    let app = server.build_router();

    mount_500_once(&primary).await;
    mount_success(&backup, plain_usage(), 1).await;

    let (status, _) = post_chat_json(app, thinking_history_request("follow up")).await;
    assert_eq!(status, StatusCode::OK, "failover must succeed");

    let bodies = received_bodies(&backup).await;
    assert_eq!(bodies.len(), 1, "backup serves exactly one attempt");
    let body_text = bodies[0].to_string();
    assert_eq!(bodies[0]["model"], "claude-4-7-sonnet");
    assert!(
        !body_text.contains("thinking"),
        "no thinking/redacted_thinking carriers may reach the backup: {body_text}"
    );
    assert!(
        !body_text.contains("signature") && !body_text.contains("sig-1234"),
        "signatures must never be forwarded cross-model: {body_text}"
    );
    // The non-reasoning part of the assistant turn survives.
    assert_eq!(bodies[0]["messages"][2]["content"][0]["type"], "text");
    assert_eq!(bodies[0]["messages"][2]["content"][0]["text"], "answer");

    // Log telemetry (Req 4.6): the strip decision and carrier counts are
    // persisted with the request log entry.
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
            .find(|e| e.provider == "adaptive-backup" && e.reasoning_compat_actions.is_some())
        {
            entry = Some(found.clone());
            break;
        }
        tokio::time::sleep(std::time::Duration::from_millis(100)).await;
    }
    let entry = entry.expect("log entry with reasoning_compat_actions must be written");
    let actions = entry.reasoning_compat_actions.unwrap();
    assert!(
        actions.contains(r#""action":"strip_attribution_unknown""#),
        "unexpected actions payload: {actions}"
    );
    assert!(
        actions.contains(r#""thinking_blocks":1"#) && actions.contains(r#""redacted_thinking_blocks":1"#),
        "carrier counts must be recorded: {actions}"
    );
}

/// DeepSeek `reasoning_content` history, deepseek primary 500s, OpenAI
/// backup serves: the `reasoning_content` field must be stripped.
#[tokio::test]
async fn cross_model_failover_strips_deepseek_reasoning_content() {
    let primary = MockServer::start().await; // deepseek-chat, priority 1
    let backup = MockServer::start().await; // gpt-4o (name â†’ None family), priority 2

    let mut config = compat_config(ReasoningCompatConfig::default(), 0);
    config.providers = vec![
        provider("deepseek-primary", &primary.uri()),
        provider("openai-backup", &backup.uri()),
    ];
    config.model_groups = vec![group(
        "compat-group",
        vec![
            reasoning_model(
                "deepseek-primary",
                "deepseek-chat",
                1,
                Some(ReasoningFamily::DeepSeek),
                None,
            ),
            reasoning_model("openai-backup", "gpt-4o", 2, None, None),
        ],
    )];
    common::isolate_databases(&mut config);
    let server = GatewayServer::new(config, None).await.unwrap();
    let app = server.build_router();

    mount_500_once(&primary).await;
    mount_success(&backup, plain_usage(), 1).await;

    let (status, _) = post_chat_json(app, deepseek_history_request()).await;
    assert_eq!(status, StatusCode::OK, "failover must succeed");

    let bodies = received_bodies(&backup).await;
    assert_eq!(bodies.len(), 1);
    let body_text = bodies[0].to_string();
    assert_eq!(bodies[0]["model"], "gpt-4o");
    assert!(
        !body_text.contains("reasoning_content"),
        "DeepSeek reasoning_content must be stripped for the OpenAI target: {body_text}"
    );
    assert!(
        !body_text.contains("hidden coy secret"),
        "reasoning payloads must not leak cross-model: {body_text}"
    );
    assert_eq!(bodies[0]["messages"][1]["content"], "answer");
}

// ---------------------------------------------------------------------------
// Same-model continuation preserves reasoning state verbatim
// ---------------------------------------------------------------------------

/// Single manual-family provider (family resolved from the model NAME â€”
/// no explicit reasoning_family), signed thinking + redacted_thinking +
/// tool_calls history: everything is forwarded verbatim (mid-tool-loop
/// continuation; the echo must be bit-identical for signature validation).
#[tokio::test]
async fn same_model_preserves_thinking_verbatim() {
    let upstream = MockServer::start().await;

    let mut config = compat_config(ReasoningCompatConfig::default(), 0);
    config.providers = vec![provider("manual-only", &upstream.uri())];
    config.model_groups = vec![group(
        "compat-group",
        vec![reasoning_model("manual-only", "claude-4-5-sonnet", 1, None, None)],
    )];
    common::isolate_databases(&mut config);
    let server = GatewayServer::new(config, None).await.unwrap();
    let app = server.build_router();

    mount_success(&upstream, plain_usage(), 1).await;

    let mut request = thinking_history_request("continue the tool loop");
    request["messages"][2]["tool_calls"] = serde_json::json!([
        {"id": "call_1", "type": "function",
         "function": {"name": "lookup", "arguments": "{}"}}
    ]);

    let (status, _) = post_chat_json(app, request.clone()).await;
    assert_eq!(status, StatusCode::OK);

    let bodies = received_bodies(&upstream).await;
    assert_eq!(bodies.len(), 1);
    // Verbatim: the assistant history reaches the provider untouched.
    assert_eq!(bodies[0]["messages"][2], request["messages"][2]);
    let content = &bodies[0]["messages"][2]["content"];
    assert_eq!(content.as_array().unwrap().len(), 3, "all blocks preserved");
    assert_eq!(content[0]["type"], "thinking");
    assert_eq!(content[0]["signature"], "sig-1234");
    assert_eq!(content[1]["type"], "redacted_thinking");
    assert_eq!(content[2]["type"], "text");
}

/// Sticky conversation-model affinity (Task 6): the provider that first
/// served a conversation prefix is promoted ahead of a higher-priority
/// competitor on the next turn, and the same-model thinking state is
/// preserved rather than stripped.
#[tokio::test]
async fn same_model_via_affinity_preserves() {
    let preferred = MockServer::start().await; // claude-4-7 (adaptive), priority 1
    let sticky = MockServer::start().await; // claude-4-5 (manual), priority 2

    let mut config = compat_config(ReasoningCompatConfig::default(), 0);
    config.providers = vec![
        provider("priority-first", &preferred.uri()),
        provider("sticky-claude", &sticky.uri()),
    ];
    config.model_groups = vec![group(
        "compat-group",
        vec![
            reasoning_model(
                "priority-first",
                "claude-4-7-sonnet",
                1,
                Some(ReasoningFamily::AnthropicAdaptive),
                None,
            ),
            reasoning_model(
                "sticky-claude",
                "claude-4-5-sonnet",
                2,
                Some(ReasoningFamily::AnthropicManual),
                None,
            ),
        ],
    )];
    common::isolate_databases(&mut config);
    let server = GatewayServer::new(config, None).await.unwrap();
    let app = server.build_router();

    // Turn 1: priority routing picks `priority-first`, which 500s once;
    // the gateway fails over to `sticky-claude` and affinity binds to it.
    // `expect(0)` on the later success mock: if turn 2 is NOT sticky, it
    // would be served here and the test fails.
    mount_500_once(&preferred).await;
    mount_success(&preferred, plain_usage(), 0).await;
    mount_success(&sticky, plain_usage(), 2).await;

    let (status, _) = post_chat_json(app.clone(), affinity_request("turn one")).await;
    assert_eq!(status, StatusCode::OK, "failover must succeed");

    // Turn 2: same conversation prefix, new tail. Priority alone would pick
    // `priority-first` (which would strip the manual-family thinking), but
    // the affinity entry promotes `sticky-claude` â€” same resolved model, so
    // the thinking state is preserved verbatim.
    let (status, _) = post_chat_json(app, affinity_request("turn two")).await;
    assert_eq!(status, StatusCode::OK);

    preferred.verify().await;
    sticky.verify().await;

    let bodies = received_bodies(&sticky).await;
    assert_eq!(bodies.len(), 2, "both turns must be served by the sticky provider");
    for (index, body) in bodies.iter().enumerate() {
        assert_eq!(body["model"], "claude-4-5-sonnet", "turn {index}");
        let content = &body["messages"][2]["content"];
        assert_eq!(
            content[0]["type"], "thinking",
            "thinking must be preserved verbatim on turn {index}"
        );
        assert_eq!(content[0]["signature"], "sig-1234", "turn {index}");
    }
}

// ---------------------------------------------------------------------------
// reasoning_effort normalization per target family
// ---------------------------------------------------------------------------

/// `reasoning_effort: "high"` â†’ manual-family target (family resolved from
/// the model name) receives `thinking: {type: "enabled", budget_tokens:
/// 16384}` (the default high budget), the effort field is removed, and
/// conflicting sampling parameters are dropped.
#[tokio::test]
async fn reasoning_effort_normalized_for_anthropic_manual() {
    let upstream = MockServer::start().await;

    let mut config = compat_config(ReasoningCompatConfig::default(), 0);
    config.providers = vec![provider("manual-name-only", &upstream.uri())];
    config.model_groups = vec![group(
        "compat-group",
        vec![reasoning_model("manual-name-only", "claude-4-5-sonnet", 1, None, None)],
    )];
    common::isolate_databases(&mut config);
    let server = GatewayServer::new(config, None).await.unwrap();
    let app = server.build_router();

    mount_success(&upstream, plain_usage(), 1).await;

    let (status, _) = post_chat_json(app, effort_request("high")).await;
    assert_eq!(status, StatusCode::OK);

    let bodies = received_bodies(&upstream).await;
    assert_eq!(bodies.len(), 1);
    let body = &bodies[0];
    assert_eq!(body["thinking"]["type"], "enabled");
    assert_eq!(body["thinking"]["budget_tokens"], 16384);
    assert!(
        body.get("reasoning_effort").is_none(),
        "reasoning_effort must not reach a manual-family target: {body}"
    );
    assert!(
        body.get("temperature").is_none() && body.get("top_p").is_none(),
        "sampling params are invalid alongside thinking: {body}"
    );
}

/// `reasoning_effort: "high"` â†’ adaptive-family target (name-classified
/// claude-4-7) receives `thinking: {type: "adaptive"}` + `output_config.
/// effort` â€” NEVER `type: "enabled"` (a 400 on Claude 4.7+).
#[tokio::test]
async fn reasoning_effort_adaptive_for_47() {
    let upstream = MockServer::start().await;

    let mut config = compat_config(ReasoningCompatConfig::default(), 0);
    config.providers = vec![provider("adaptive-name-only", &upstream.uri())];
    config.model_groups = vec![group(
        "compat-group",
        vec![reasoning_model("adaptive-name-only", "claude-4-7-sonnet", 1, None, None)],
    )];
    common::isolate_databases(&mut config);
    let server = GatewayServer::new(config, None).await.unwrap();
    let app = server.build_router();

    mount_success(&upstream, plain_usage(), 1).await;

    let (status, _) = post_chat_json(app, effort_request("high")).await;
    assert_eq!(status, StatusCode::OK);

    let bodies = received_bodies(&upstream).await;
    assert_eq!(bodies.len(), 1);
    let body = &bodies[0];
    assert_eq!(body["thinking"]["type"], "adaptive");
    assert_ne!(body["thinking"]["type"], "enabled");
    assert!(
        body["thinking"].get("budget_tokens").is_none(),
        "adaptive thinking carries no budget_tokens: {body}"
    );
    assert_eq!(body["output_config"]["effort"], "high");
    assert!(body.get("reasoning_effort").is_none());
}

/// `reasoning_effort: "high"` â†’ OpenAI-reasoning target (name-classified
/// o3) keeps `reasoning_effort` as-is and gets no Anthropic `thinking`.
#[tokio::test]
async fn reasoning_effort_kept_for_openai_reasoning() {
    let upstream = MockServer::start().await;

    let mut config = compat_config(ReasoningCompatConfig::default(), 0);
    config.providers = vec![provider("openai-reasoning-name", &upstream.uri())];
    config.model_groups = vec![group(
        "compat-group",
        vec![reasoning_model("openai-reasoning-name", "o3", 1, None, None)],
    )];
    common::isolate_databases(&mut config);
    let server = GatewayServer::new(config, None).await.unwrap();
    let app = server.build_router();

    mount_success(&upstream, plain_usage(), 1).await;

    let (status, _) = post_chat_json(app, effort_request("high")).await;
    assert_eq!(status, StatusCode::OK);

    let bodies = received_bodies(&upstream).await;
    assert_eq!(bodies.len(), 1);
    let body = &bodies[0];
    assert_eq!(body["reasoning_effort"], "high");
    assert!(
        body.get("thinking").is_none(),
        "no Anthropic thinking parameter for an OpenAI target: {body}"
    );
}

// ---------------------------------------------------------------------------
// Bedrock: normalized budget replaces the legacy 4096 hardcode
// ---------------------------------------------------------------------------

/// With reasoning_compat enabled, a `reasoning_effort: "minimal"` request to
/// a bedrock reasoning model must emit `budget_tokens: 1024` (the minimal
/// default) â€” not the legacy hardcoded 4096.
///
/// #[ignore] â€” REAL IMPLEMENTATION BUG (observed 2026-08-30): the wire body
/// carries NO `thinking` parameter at all. The normalizer correctly emits
/// `thinking: {type:"enabled", budget_tokens:1024}` into the outgoing
/// request, but `Router::sanitize_request_for_provider(.., "bedrock")`
/// (router.rs, runs after the normalize/legacy-inject stages) calls
/// `sanitize_mantle_chat_request`, whose `MANTLE_CHAT_ALLOWED` allowlist
/// (providers/bedrock.rs) does not include `thinking` (nor `output_config`
/// or `reasoning`), so the parameter is deleted before the request is sent.
/// Fix: add the Anthropic reasoning parameter keys to `MANTLE_CHAT_ALLOWED`
/// (or skip the sanitize pass when reasoning_compat emitted a shape).
#[tokio::test]
async fn budget_replaces_bedrock_hardcode() {
    let upstream = MockServer::start().await;

    let mut config = compat_config(ReasoningCompatConfig::default(), 0);
    config.providers = vec![bedrock_provider("bedrock-mock", &upstream.uri())];
    config.model_groups = vec![group(
        "compat-group",
        vec![reasoning_model("bedrock-mock", "claude-4-5-sonnet", 1, None, None)],
    )];
    common::isolate_databases(&mut config);
    let server = GatewayServer::new(config, None).await.unwrap();
    let app = server.build_router();

    mount_success(&upstream, plain_usage(), 1).await;

    let mut request = effort_request("minimal");
    request["model"] = serde_json::json!("compat-group");

    let (status, _) = post_chat_json(app, request).await;
    assert_eq!(status, StatusCode::OK);

    let bodies = received_bodies(&upstream).await;
    assert_eq!(bodies.len(), 1);
    let body = &bodies[0];
    assert_eq!(
        body["thinking"]["budget_tokens"], 1024,
        "normalized minimal budget must replace the 4096 hardcode: {body}"
    );
    assert_eq!(body["thinking"]["type"], "enabled");
    assert!(body.get("reasoning_effort").is_none());
}

/// Companion: with reasoning_compat disabled, the legacy bedrock block
/// still injects the hardcoded `budget_tokens: 4096` when the provider
/// enables reasoning (opt-out reproduces old behavior exactly).
///
/// #[ignore] â€” REAL IMPLEMENTATION BUG (observed 2026-08-30): the legacy
/// injection (`router.rs` "Inject reasoning/extended thinking parameter for
/// Bedrock providers" block) writes `thinking: {type:"enabled",
/// budget_tokens:4096}` into the outgoing request, but the subsequent
/// `sanitize_mantle_chat_request` pass deletes it because `thinking` is
/// missing from `MANTLE_CHAT_ALLOWED` (providers/bedrock.rs). The legacy
/// bedrock thinking injection is dead on this path â€” the provider never
/// receives any thinking parameter. Same root cause as
/// `budget_replaces_bedrock_hardcode`.
#[tokio::test]
async fn bedrock_legacy_hardcode_when_compat_disabled() {
    let upstream = MockServer::start().await;

    let mut config = compat_config(
        ReasoningCompatConfig {
            enabled: false,
            ..ReasoningCompatConfig::default()
        },
        0,
    );
    config.providers = vec![bedrock_provider("bedrock-mock", &upstream.uri())];
    config.model_groups = vec![group(
        "compat-group",
        vec![reasoning_model("bedrock-mock", "claude-4-5-sonnet", 1, None, None)],
    )];
    common::isolate_databases(&mut config);
    let server = GatewayServer::new(config, None).await.unwrap();
    let app = server.build_router();

    mount_success(&upstream, plain_usage(), 1).await;

    let (status, _) = post_chat_json(
        app,
        serde_json::json!({
            "model": "compat-group",
            "max_tokens": 20000,
            "messages": [{"role": "user", "content": "solve it"}]
        }),
    )
    .await;
    assert_eq!(status, StatusCode::OK);

    let bodies = received_bodies(&upstream).await;
    assert_eq!(bodies.len(), 1);
    let body = &bodies[0];
    assert_eq!(
        body["thinking"]["budget_tokens"], 4096,
        "legacy path must keep its hardcoded 4096 budget: {body}"
    );
}

// ---------------------------------------------------------------------------
// Opt-out passthrough
// ---------------------------------------------------------------------------

/// `reasoning_compat.enabled: false`: cross-model failover forwards the
/// thinking history UNSTRIPPED (exact legacy passthrough).
#[tokio::test]
async fn disabled_compat_passthrough() {
    let primary = MockServer::start().await; // claude-4-5 (manual), priority 1
    let backup = MockServer::start().await; // claude-4-7 (adaptive), priority 2

    let mut config = compat_config(
        ReasoningCompatConfig {
            enabled: false,
            ..ReasoningCompatConfig::default()
        },
        0,
    );
    config.providers = vec![
        provider("manual-primary", &primary.uri()),
        provider("adaptive-backup", &backup.uri()),
    ];
    config.model_groups = vec![group(
        "compat-group",
        vec![
            reasoning_model(
                "manual-primary",
                "claude-4-5-sonnet",
                1,
                Some(ReasoningFamily::AnthropicManual),
                None,
            ),
            reasoning_model(
                "adaptive-backup",
                "claude-4-7-sonnet",
                2,
                Some(ReasoningFamily::AnthropicAdaptive),
                None,
            ),
        ],
    )];
    common::isolate_databases(&mut config);
    let server = GatewayServer::new(config, None).await.unwrap();
    let app = server.build_router();

    mount_500_once(&primary).await;
    mount_success(&backup, plain_usage(), 1).await;

    let (status, _) = post_chat_json(app, thinking_history_request("follow up")).await;
    assert_eq!(status, StatusCode::OK);

    let bodies = received_bodies(&backup).await;
    assert_eq!(bodies.len(), 1);
    let body_text = bodies[0].to_string();
    assert!(
        body_text.contains("sig-1234") && body_text.contains("redacted_thinking"),
        "disabled compat must forward carriers unstripped (legacy behavior): {body_text}"
    );
}

// ---------------------------------------------------------------------------
// 400-recovery: aggressive strip + same-provider retry
// ---------------------------------------------------------------------------

/// First attempt preserves same-family thinking, the provider rejects it
/// with a thinking-validation 400; the gateway aggressively strips every
/// reasoning carrier and retries the SAME provider without backoff.
#[tokio::test]
async fn thinking_400_triggers_aggressive_strip_retry() {
    let upstream = MockServer::start().await;

    let mut config = compat_config(ReasoningCompatConfig::default(), 2);
    config.providers = vec![provider("manual-retry", &upstream.uri())];
    config.model_groups = vec![group(
        "compat-group",
        vec![reasoning_model(
            "manual-retry",
            "claude-4-5-sonnet",
            1,
            Some(ReasoningFamily::AnthropicManual),
            None,
        )],
    )];
    common::isolate_databases(&mut config);
    let server = GatewayServer::new(config, None).await.unwrap();
    let app = server.build_router();

    mount_thinking_400_once(&upstream).await;
    mount_success(&upstream, plain_usage(), 1).await;

    let (status, _) = post_chat_json(app, thinking_history_request("follow up")).await;
    assert_eq!(status, StatusCode::OK, "aggressive strip retry must succeed");

    let bodies = received_bodies(&upstream).await;
    assert_eq!(bodies.len(), 2, "exactly two attempts on the same provider");
    let first = bodies[0].to_string();
    assert!(
        first.contains("sig-1234"),
        "first attempt preserves same-family thinking: {first}"
    );
    let second = bodies[1].to_string();
    assert!(
        !second.contains("thinking") && !second.contains("sig-1234"),
        "retry must carry no reasoning carriers: {second}"
    );
    assert!(
        second.contains("answer"),
        "non-reasoning content survives the aggressive strip: {second}"
    );
}

// ---------------------------------------------------------------------------
// Reasoning-token attribution (metrics + SQLite log)
// ---------------------------------------------------------------------------

/// OpenAI-shaped usage (`completion_tokens_details.reasoning_tokens`) is
/// attributed to the provider health snapshot (tokens + dedicated cost at
/// `cost_per_million_reasoning_tokens`) and persisted in the request log.
#[tokio::test]
async fn reasoning_tokens_logged_and_metricd() {
    let upstream = MockServer::start().await;

    let mut config = compat_config(ReasoningCompatConfig::default(), 0);
    config.providers = vec![provider("reasoning-openai", &upstream.uri())];
    config.model_groups = vec![group(
        "compat-group",
        vec![reasoning_model(
            "reasoning-openai",
            "o3",
            1,
            Some(ReasoningFamily::OpenAIReasoning),
            Some(2.0),
        )],
    )];
    common::isolate_databases(&mut config);
    let server = GatewayServer::new(config, None).await.unwrap();
    let app = server.build_router();

    let usage = serde_json::json!({
        "prompt_tokens": 10,
        "completion_tokens": 1000,
        "total_tokens": 1010,
        "completion_tokens_details": {"reasoning_tokens": 500}
    });
    mount_success(&upstream, usage, 1).await;

    let (status, _) = post_chat_json(
        app,
        serde_json::json!({
            "model": "compat-group",
            "messages": [{"role": "user", "content": "solve it"}]
        }),
    )
    .await;
    assert_eq!(status, StatusCode::OK);

    // Provider health: 500 reasoning tokens priced at $2.00/M â†’ $0.001.
    let snapshot = server.state.metrics.snapshot();
    let health = snapshot
        .provider_health
        .iter()
        .find(|ph| ph.provider == "reasoning-openai")
        .expect("provider health must exist");
    assert_eq!(health.reasoning_tokens, 500);
    assert!(
        (health.reasoning_cost_usd - 0.001).abs() < 1e-9,
        "reasoning cost {} must use the dedicated price",
        health.reasoning_cost_usd
    );

    // SQLite request log: the reasoning-token count is persisted.
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
            .find(|e| e.provider == "reasoning-openai" && e.reasoning_tokens.is_some())
        {
            entry = Some(found.clone());
            break;
        }
        tokio::time::sleep(std::time::Duration::from_millis(100)).await;
    }
    let entry = entry.expect("log entry with reasoning_tokens must be written");
    assert_eq!(entry.reasoning_tokens, Some(500));
}
