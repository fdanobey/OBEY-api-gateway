//! Integration tests for streaming (SSE) refusal detection and refusal-triggered
//! failover (task 17.9, Req 12.9, 12.14).
//!
//! These tests drive the full Axum router via `tower::ServiceExt::oneshot()`
//! (no port binding) with a guardrails section whose pipeline enables
//! `failover_on_refusal`. A `wiremock` upstream stands in for the LLM providers
//! so the streaming handler buffers the SSE, runs refusal detection on the
//! assembled response, and triggers failover when a refusal is detected.
//!
//! Tested scenarios:
//!   * Buffered-SSE refusal detection triggers failover to a second provider
//!     whose non-refusal response is forwarded to the caller (Req 12.9).
//!   * Premature-disconnect (no `finish_reason`) with refusal content in the
//!     partial response triggers the same failover behavior (Req 12.14).
//!   * A non-refusal buffered SSE response passes through normally without
//!     triggering failover.

use axum::body::Body;
use axum::http::{Request, StatusCode};
use tower::ServiceExt;

use ai_gateway::config::*;
use ai_gateway::gateway::GatewayServer;
use ai_gateway::guardrail::{
    FailurePolicy, GuardrailConfig, GuardrailProviderConfig, GuardrailProviderType,
    InstructionInsertionMode, PipelineConfig, PolicyAction, ProviderSettings, RegexPatternConfig,
    RegexRuleMode, StageConfig, StagePhase,
};

use wiremock::matchers::{method, path};
use wiremock::{Mock, MockServer, ResponseTemplate};

// ---------------------------------------------------------------------------
// Config builders
// ---------------------------------------------------------------------------

/// A single OpenAI-typed provider pointed at `base_url` with the given `name`.
fn provider_named(name: &str, base_url: &str) -> Provider {
    Provider {
        name: name.to_string(),
        provider_type: "openai".to_string(),
        base_url: Some(base_url.to_string()),
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

/// Two-provider config: `primary` (priority 100, tried first) and `backup`
/// (priority 200) in a single `gpt-4` model group. The guardrails config
/// enables refusal-failover so that detected refusals trigger re-dispatch to
/// the backup.
fn two_provider_config(primary_uri: &str, backup_uri: &str, guardrails: GuardrailConfig) -> Config {
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
        providers: vec![
            provider_named("primary", primary_uri),
            provider_named("backup", backup_uri),
        ],
        model_groups: vec![ModelGroup {
            name: "test-group".to_string(),
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
        streaming: Some(StreamingConfig {
            emit_early_event: true,
            passthrough_enabled: false,
            ..StreamingConfig::default()
        }),
        virtual_keys: Default::default(),
        loop_detection: Default::default(),
        guardrails: Some(guardrails),
        tool_compression: Default::default(),
        structured_output: None,
    }
}

/// A guardrail config with a single post-call regex stage (to ensure post-call
/// pipeline is bound, forcing the buffered path) plus `failover_on_refusal: true`
/// and a custom refusal phrase list for deterministic detection.
fn refusal_failover_guardrails() -> GuardrailConfig {
    GuardrailConfig {
        providers: vec![GuardrailProviderConfig {
            name: "noop_regex".to_string(),
            provider_type: GuardrailProviderType::Regex,
            failure_policy: FailurePolicy::FailOpen,
            timeout_seconds: 5,
            settings: ProviderSettings {
                patterns: vec![RegexPatternConfig {
                    name: "unlikely_pattern".to_string(),
                    // A pattern that won't match normal test content — ensures the
                    // post-call stage produces `allow` so refusal detection is the
                    // deciding factor.
                    regex: r"XYZZY_NEVER_MATCH_42".to_string(),
                    entity: "NOOP".to_string(),
                    mode: RegexRuleMode::Deny,
                }],
                ..Default::default()
            },
        }],
        pipelines: vec![PipelineConfig {
            name: "refusal_pipeline".to_string(),
            stages: vec![StageConfig {
                name: "noop_scan".to_string(),
                provider: "noop_regex".to_string(),
                phase: StagePhase::PostCall,
                action: PolicyAction::Allow,
            }],
            redaction_notice_instruction: None,
            instruction_insertion_mode: InstructionInsertionMode::default(),
            failover_on_refusal: true,
            // Use a simple, deterministic phrase list for testing.
            refusal_phrase_list: Some(vec!["i cannot help with".to_string()]),
        }],
        global_default_pipeline: Some("refusal_pipeline".to_string()),
        bindings: Default::default(),
    }
}

/// Same as above but with failover disabled — used to test that non-refusal
/// content passes through normally.
fn refusal_disabled_guardrails() -> GuardrailConfig {
    let mut g = refusal_failover_guardrails();
    g.pipelines[0].failover_on_refusal = false;
    g
}

/// Same as refusal_failover_guardrails but with `fail_open` failure policy
/// for the streaming premature disconnect test (Req 12.14).
fn refusal_failover_guardrails_fail_open() -> GuardrailConfig {
    let mut g = refusal_failover_guardrails();
    g.providers[0].failure_policy = FailurePolicy::FailOpen;
    g
}

/// Build a router from `config` without binding to a port.
async fn build_app(config: Config) -> axum::Router {
    let server = GatewayServer::new(config, None).await.unwrap();
    server.build_router()
}

/// Send a streaming chat-completion request and return `(status, body bytes)`.
async fn post_stream(app: axum::Router) -> (StatusCode, Vec<u8>) {
    let body = serde_json::json!({
        "model": "gpt-4",
        "messages": [{"role": "user", "content": "hello"}],
        "stream": true
    });
    let req = Request::post("/v1/chat/completions")
        .header("content-type", "application/json")
        .body(Body::from(serde_json::to_string(&body).unwrap()))
        .unwrap();
    let resp = app.oneshot(req).await.unwrap();
    let status = resp.status();
    let bytes = axum::body::to_bytes(resp.into_body(), 16 * 1024 * 1024)
        .await
        .unwrap();
    (status, bytes.to_vec())
}

/// Collect the JSON `data:` chunks from an SSE body, skipping `[DONE]` and
/// keep-alive comment lines.
fn sse_data_chunks(body: &[u8]) -> Vec<serde_json::Value> {
    let text = std::str::from_utf8(body).unwrap();
    text.lines()
        .filter_map(|l| l.strip_prefix("data: "))
        .filter(|p| *p != "[DONE]")
        .filter_map(|p| serde_json::from_str::<serde_json::Value>(p).ok())
        .collect()
}

/// The assembled assistant content across all re-chunked content deltas.
fn assembled_content(chunks: &[serde_json::Value]) -> String {
    chunks
        .iter()
        .filter_map(|c| c["choices"][0]["delta"]["content"].as_str())
        .collect()
}

// ---------------------------------------------------------------------------
// Upstream mocks
// ---------------------------------------------------------------------------

/// Mock provider returning a non-streaming chat completion JSON body (the
/// buffered path assembles it). The `content` becomes the assistant message.
async fn start_json_mock(content: &str) -> MockServer {
    let server = MockServer::start().await;
    let body = serde_json::json!({
        "id": "chatcmpl-mock",
        "object": "chat.completion",
        "created": 1_700_000_000i64,
        "model": "gpt-4",
        "choices": [{
            "index": 0,
            "message": { "role": "assistant", "content": content },
            "finish_reason": "stop"
        }],
        "usage": { "prompt_tokens": 10, "completion_tokens": 8, "total_tokens": 18 }
    });
    Mock::given(method("POST"))
        .and(path("/v1/chat/completions"))
        .respond_with(ResponseTemplate::new(200).set_body_json(body))
        .mount(&server)
        .await;
    server
}

/// Mock provider returning a raw SSE body with content deltas but NO
/// `finish_reason` and no `[DONE]` sentinel — i.e. a premature disconnect
/// (Req 10.5, 12.14).
async fn start_incomplete_sse_mock(content: &str) -> MockServer {
    let server = MockServer::start().await;
    let sse = format!(
        "data: {}\n\n",
        serde_json::json!({
            "id": "chatcmpl-mock",
            "object": "chat.completion.chunk",
            "created": 1_700_000_000i64,
            "model": "gpt-4",
            "choices": [{
                "index": 0,
                "delta": { "role": "assistant", "content": content }
            }]
        })
    );
    Mock::given(method("POST"))
        .and(path("/v1/chat/completions"))
        .respond_with(
            ResponseTemplate::new(200)
                .insert_header("content-type", "text/event-stream")
                .set_body_string(sse),
        )
        .mount(&server)
        .await;
    server
}

// ---------------------------------------------------------------------------
// Req 12.9 — Buffered-SSE refusal detection triggers failover
// ---------------------------------------------------------------------------

/// Req 12.9: When a streaming response is buffered and the assembled content
/// matches the configured refusal phrase, the gateway fails over to the next
/// provider in the fallback ordering. The backup's non-refusal response is
/// forwarded to the caller.
#[tokio::test]
async fn streaming_refusal_detection_triggers_failover_to_backup() {
    // Primary returns a refusal phrase.
    let primary = start_json_mock("I cannot help with that request.").await;
    // Backup returns a helpful response.
    let backup = start_json_mock("Here is the code you requested.").await;

    let app = build_app(two_provider_config(
        &primary.uri(),
        &backup.uri(),
        refusal_failover_guardrails(),
    ))
    .await;

    let (status, body) = post_stream(app).await;
    assert_eq!(status, StatusCode::OK);

    let text = std::str::from_utf8(&body).unwrap();
    assert!(text.contains("[DONE]"), "stream must terminate with [DONE]");

    let content = assembled_content(&sse_data_chunks(&body));
    assert!(
        content.contains("Here is the code you requested"),
        "the backup's non-refusal content must be forwarded after failover, got: {content:?}"
    );
    assert!(
        !content.contains("I cannot help with"),
        "the primary's refusal must NOT reach the caller, got: {content:?}"
    );

    // Both providers should have been called (primary first, then backup via failover).
    let primary_calls = primary.received_requests().await.unwrap();
    let backup_calls = backup.received_requests().await.unwrap();
    assert_eq!(primary_calls.len(), 1, "primary called once");
    assert_eq!(backup_calls.len(), 1, "backup called once via failover");
}

// ---------------------------------------------------------------------------
// Req 12.14 — Premature disconnect with refusal content triggers failover
// ---------------------------------------------------------------------------

/// Req 12.14: When a streaming response terminates early (no finish_reason) and
/// the partially assembled content contains a refusal phrase, the same failover
/// behavior applies — detection runs on partial content and the backup is tried.
#[tokio::test]
async fn premature_disconnect_with_refusal_triggers_failover() {
    // Primary returns partial content with a refusal phrase but NO finish_reason
    // (premature disconnect). The pipeline uses fail_open so the partial content
    // is eligible for refusal detection rather than being discarded (Req 12.14).
    let primary =
        start_incomplete_sse_mock("I cannot help with this due to policy restrictions.").await;
    // Backup returns a complete, non-refusal response.
    let backup = start_json_mock("Here is your answer, no problem.").await;

    let mut config = two_provider_config(
        &primary.uri(),
        &backup.uri(),
        refusal_failover_guardrails_fail_open(),
    );
    // Enable passthrough so the incomplete SSE mock's raw body is consumed
    // as a stream (triggering the premature-disconnect path).
    config.streaming = Some(StreamingConfig {
        emit_early_event: true,
        passthrough_enabled: true,
        ..StreamingConfig::default()
    });

    let app = build_app(config).await;

    let (status, body) = post_stream(app).await;
    assert_eq!(status, StatusCode::OK);

    let text = std::str::from_utf8(&body).unwrap();
    assert!(text.contains("[DONE]"), "stream must terminate with [DONE]");

    let content = assembled_content(&sse_data_chunks(&body));
    assert!(
        content.contains("Here is your answer"),
        "backup's non-refusal content must be forwarded after premature-disconnect failover, got: {content:?}"
    );
    assert!(
        !content.contains("I cannot help with"),
        "primary's refusal content must NOT reach the caller, got: {content:?}"
    );
}

// ---------------------------------------------------------------------------
// Non-refusal buffered SSE passes through normally
// ---------------------------------------------------------------------------

/// A buffered SSE response that does NOT contain a refusal phrase passes
/// through normally without triggering failover — proving that refusal detection
/// does not interfere with normal responses.
#[tokio::test]
async fn non_refusal_streaming_response_passes_through_normally() {
    // Primary returns helpful content (no refusal phrase).
    let primary = start_json_mock("Here is the detailed answer you asked for.").await;
    // Backup should NEVER be called.
    let backup = start_json_mock("BACKUP_SHOULD_NOT_BE_SEEN").await;

    let app = build_app(two_provider_config(
        &primary.uri(),
        &backup.uri(),
        refusal_failover_guardrails(),
    ))
    .await;

    let (status, body) = post_stream(app).await;
    assert_eq!(status, StatusCode::OK);

    let text = std::str::from_utf8(&body).unwrap();
    assert!(text.contains("[DONE]"), "stream must terminate with [DONE]");

    let content = assembled_content(&sse_data_chunks(&body));
    assert!(
        content.contains("Here is the detailed answer"),
        "non-refusal content from primary passes through, got: {content:?}"
    );
    assert!(
        !content.contains("BACKUP_SHOULD_NOT_BE_SEEN"),
        "backup must NOT be invoked for non-refusal responses"
    );

    // Only the primary should be called.
    let primary_calls = primary.received_requests().await.unwrap();
    let backup_calls = backup.received_requests().await.unwrap();
    assert_eq!(primary_calls.len(), 1, "primary called once");
    assert_eq!(backup_calls.len(), 0, "backup NOT called for non-refusal");
}

// ---------------------------------------------------------------------------
// Refusal detected but failover disabled — response passes through
// ---------------------------------------------------------------------------

/// When `failover_on_refusal` is disabled, a refusal is detected but the
/// response is returned as-is without triggering failover (Req 12.6).
#[tokio::test]
async fn refusal_detected_but_failover_disabled_returns_refusal_to_caller() {
    // Primary returns a refusal.
    let primary = start_json_mock("I cannot help with that request.").await;
    // Backup should NOT be called since failover is disabled.
    let backup = start_json_mock("BACKUP_SHOULD_NOT_BE_SEEN").await;

    let app = build_app(two_provider_config(
        &primary.uri(),
        &backup.uri(),
        refusal_disabled_guardrails(),
    ))
    .await;

    let (status, body) = post_stream(app).await;
    assert_eq!(status, StatusCode::OK);

    let content = assembled_content(&sse_data_chunks(&body));
    assert!(
        content.contains("I cannot help with"),
        "with failover disabled, the refusal response passes through to the caller, got: {content:?}"
    );
    assert!(
        !content.contains("BACKUP_SHOULD_NOT_BE_SEEN"),
        "backup must NOT be invoked when failover is disabled"
    );

    let backup_calls = backup.received_requests().await.unwrap();
    assert_eq!(
        backup_calls.len(),
        0,
        "backup NOT called when failover disabled"
    );
}
