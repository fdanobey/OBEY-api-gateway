//! Integration tests for streaming (SSE) post-call guardrails (task 13.4, Req 10).
//!
//! These tests drive the full Axum router via `tower::ServiceExt::oneshot()`
//! (no port binding, per repo conventions) with a `guardrails` section whose
//! `global_default_pipeline` binds a post-call regex stage. A `wiremock`
//! upstream stands in for the LLM provider so `route_request_streaming` reaches
//! a real HTTP endpoint.
//!
//! Endpoint-tested here:
//!   - post-call `redact`  â†’ assembled/re-chunked SSE contains `[REDACTED]`
//!     (never the secret) and ends with `data: [DONE]` (Req 10.2/10.4, and the
//!     re-chunk itself proves the buffered path was taken).
//!   - post-call `block`   â†’ a terminal block frame carrying
//!     `guardrail_policy_violation` + the triggering category, followed by
//!     `data: [DONE]` (Req 10.3).
//!   - premature disconnect (upstream SSE ends with no `finish_reason` while
//!     buffering an eligible pass-through provider): `fail_close` discards with a
//!     `guardrail_unavailable` error frame; `fail_open` forwards the partial
//!     content, both terminating with `[DONE]` (Req 10.5).
//!
//! Covered by the `guardrail::stream` module's own unit tests (documented rather
//! than re-asserted at the endpoint level, where they are impractical):
//!   - the 10 MB buffer cap abort (Req 10.1) â€” see
//!     `stream::tests::sse_buffer_enforces_default_10mb_cap` /
//!     `append_within_cap_rejects_overflow_and_leaves_buffer_intact`; reaching
//!     10 MB through a mock is prohibitively slow and axum's SSE body is not
//!     introspectable byte-by-byte.
//!   - keep-alive comment cadence (Req 10.2) â€” axum's `KeepAlive` interval and
//!     the buffering-loop keepalive comment are timing-driven and not
//!     deterministically observable via a fast `oneshot()`; buffering behavior
//!     is instead proven by the re-chunk assertions below.

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

mod common;

// ---------------------------------------------------------------------------
// Config builders (local â€” this file must not edit shared test helpers).
// ---------------------------------------------------------------------------

/// A single OpenAI-typed provider pointed at `base_url`.
fn provider(base_url: &str) -> Provider {
    Provider {
        name: "test-provider".to_string(),
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
        reasoning: true,
        codex_base_url_override: None,
        codex_model_override: None,
        instructions_override: None,
        max_rate_limit_cooldown_seconds: None,
        memory: None,
    }
}

/// Base config with one provider/model group and no guardrails yet.
fn base_config(base_url: &str) -> Config {
    Config {
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
        providers: vec![provider(base_url)],
        model_groups: vec![ModelGroup {
            name: "test-group".to_string(),
            version_fallback_enabled: false,
            compression: None,
            structured_output: None,
            memory: None,
            models: vec![ProviderModel {
                cache_support: None,
                cache_min_tokens: None,
                cost_per_million_cache_read_input_tokens: None,
                cost_per_million_cache_creation_input_tokens: None,
                provider: "test-provider".to_string(),
                model: "gpt-4".to_string(),
                cost_per_million_input_tokens: 30.0,
                cost_per_million_output_tokens: 60.0,
                priority: 100,
                structured_output_passthrough: None,
                tier: None,
                context_window: 0,
                specializations: vec![],
                cost_per_million_reasoning_tokens: None,
                reasoning_family: None,
                reasoning_parameter: None,
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
        reasoning_compat: Default::default(),
    }
}

/// A guardrail config whose global-default pipeline runs a single post-call
/// regex stage (`SECRET` entity, deny mode) with the given `action` and
/// provider `failure_policy`.
fn post_call_guardrails(action: PolicyAction, failure_policy: FailurePolicy) -> GuardrailConfig {
    GuardrailConfig {
        providers: vec![GuardrailProviderConfig {
            name: "regex_secret".to_string(),
            provider_type: GuardrailProviderType::Regex,
            failure_policy,
            timeout_seconds: 5,
            settings: ProviderSettings {
                patterns: vec![RegexPatternConfig {
                    name: "topsecret".to_string(),
                    regex: r"TOPSECRET\d+".to_string(),
                    entity: "SECRET".to_string(),
                    mode: RegexRuleMode::Deny,
                }],
                ..Default::default()
            },
        }],
        pipelines: vec![PipelineConfig {
            name: "post_pipeline".to_string(),
            stages: vec![StageConfig {
                name: "secret_scan".to_string(),
                provider: "regex_secret".to_string(),
                phase: StagePhase::PostCall,
                action,
            }],
            redaction_notice_instruction: None,
            instruction_insertion_mode: InstructionInsertionMode::default(),
            failover_on_refusal: false,
            refusal_phrase_list: None,
            tool_result: ai_gateway::guardrail::config::ToolResultPhaseConfig::default(),
        }],
        global_default_pipeline: Some("post_pipeline".to_string()),
        bindings: Default::default(),
        ..Default::default()
    }
}

/// Build a router from `config` without binding to a port.
async fn build_app(mut config: Config) -> axum::Router {
    common::isolate_databases(&mut config);
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

/// Mock provider returning a complete non-streaming chat.completion JSON body
/// (the buffered path reassembles it). `content` becomes the assistant message.
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

/// Mock provider that returns a raw SSE body with content deltas but NO
/// `finish_reason` and no `[DONE]` sentinel â€” i.e. a premature disconnect once
/// the body ends (Req 10.5).
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

/// Buffered streaming config (pass-through disabled): the JSON mock is
/// reassembled into a complete response for post-call analysis.
fn buffered_config(base_url: &str, guardrails: GuardrailConfig) -> Config {
    let mut cfg = base_config(base_url);
    cfg.streaming = Some(StreamingConfig {
        emit_early_event: true,
        passthrough_enabled: false,
        ..StreamingConfig::default()
    });
    cfg.guardrails = Some(guardrails);
    cfg
}

/// Pass-through streaming config: an eligible provider streams live, so the
/// buffered post-call path accumulates the raw SSE body (used to exercise the
/// premature-disconnect failure policy).
fn passthrough_config(base_url: &str, guardrails: GuardrailConfig) -> Config {
    let mut cfg = base_config(base_url);
    cfg.streaming = Some(StreamingConfig {
        emit_early_event: true,
        passthrough_enabled: true,
        ..StreamingConfig::default()
    });
    cfg.guardrails = Some(guardrails);
    cfg
}

// ---------------------------------------------------------------------------
// Req 10.2 / 10.4 â€” post-call redact re-chunks the assembled response
// ---------------------------------------------------------------------------

/// A bound post-call `redact` stage forces the buffered path: the gateway
/// assembles the upstream response, redacts the matched secret to `[REDACTED]`,
/// and re-chunks the result into SSE ending with `[DONE]`. The presence of the
/// re-chunked, redacted content is itself proof the buffered path was taken
/// (Req 10.4) rather than a verbatim relay.
#[tokio::test]
async fn post_call_redact_rechunks_redacted_content_and_terminates_with_done() {
    let mock = start_json_mock("Your code is TOPSECRET42 keep it safe").await;
    let app = build_app(buffered_config(
        &mock.uri(),
        post_call_guardrails(PolicyAction::Redact, FailurePolicy::FailClose),
    ))
    .await;

    let (status, body) = post_stream(app).await;
    assert_eq!(status, StatusCode::OK);

    let text = std::str::from_utf8(&body).unwrap();
    assert!(text.contains("[DONE]"), "stream must terminate with [DONE]");

    let chunks = sse_data_chunks(&body);
    assert!(!chunks.is_empty(), "expected re-chunked SSE data events");
    let content = assembled_content(&chunks);
    assert!(
        content.contains("[REDACTED]"),
        "redacted output must contain [REDACTED], got: {content:?}"
    );
    assert!(
        !content.contains("TOPSECRET42"),
        "the secret must never reach the caller, got: {content:?}"
    );
    assert!(
        content.contains("Your code is") && content.contains("keep it safe"),
        "non-matched content is preserved around the redaction, got: {content:?}"
    );
}

// ---------------------------------------------------------------------------
// Req 10.3 â€” post-call block emits a terminal block frame then [DONE]
// ---------------------------------------------------------------------------

/// A bound post-call `block` stage terminates the SSE stream with a single
/// policy-violation event carrying `guardrail_policy_violation` and the
/// triggering category, followed by `data: [DONE]` â€” and never forwards the
/// upstream content (Req 10.3).
#[tokio::test]
async fn post_call_block_emits_terminal_frame_then_done() {
    let mock = start_json_mock("Your code is TOPSECRET42 keep it safe").await;
    let app = build_app(buffered_config(
        &mock.uri(),
        post_call_guardrails(PolicyAction::Block, FailurePolicy::FailClose),
    ))
    .await;

    let (status, body) = post_stream(app).await;
    // The SSE stream itself is a 200; the block is expressed in-band (Req 10.3).
    assert_eq!(status, StatusCode::OK);

    let text = std::str::from_utf8(&body).unwrap();
    assert!(
        text.contains("[DONE]"),
        "block stream must still close with [DONE]"
    );
    assert!(
        !text.contains("TOPSECRET42"),
        "blocked content must never reach the caller"
    );

    // The terminal block frame carries the guardrail contract fields (Req 10.3).
    let block_frame = sse_data_chunks(&body)
        .into_iter()
        .find(|c| c["error"]["type"] == "guardrail_policy_violation")
        .expect("a terminal guardrail_policy_violation frame is present");
    assert_eq!(block_frame["error"]["category"], "SECRET");
    assert_eq!(
        block_frame["error"]["message"],
        "Request blocked by guardrail policy"
    );

    // No assistant content chunks should have been forwarded.
    let content = assembled_content(&sse_data_chunks(&body));
    assert!(
        content.is_empty(),
        "block must not forward any assistant content, got: {content:?}"
    );
}

// ---------------------------------------------------------------------------
// Req 10.5 â€” premature disconnect failure policy
// ---------------------------------------------------------------------------

/// With a `fail_close` post-call stage, an upstream SSE stream that ends before
/// a `finish_reason` (premature disconnect) causes the gateway to discard the
/// partial content and emit a `guardrail_unavailable` error frame + `[DONE]`
/// (Req 10.5).
#[tokio::test]
async fn premature_disconnect_fail_close_discards_partial_content() {
    let mock = start_incomplete_sse_mock("Partial answer with no finish reason").await;
    let app = build_app(passthrough_config(
        &mock.uri(),
        // No `TOPSECRET` in the partial content, so only the disconnect policy
        // determines the outcome (not a redact/block on content).
        post_call_guardrails(PolicyAction::Redact, FailurePolicy::FailClose),
    ))
    .await;

    let (status, body) = post_stream(app).await;
    assert_eq!(status, StatusCode::OK);

    let text = std::str::from_utf8(&body).unwrap();
    assert!(
        text.contains("[DONE]"),
        "stream must still close with [DONE]"
    );

    let err_frame = sse_data_chunks(&body)
        .into_iter()
        .find(|c| c["error"]["type"] == "guardrail_unavailable")
        .expect("fail_close disconnect emits a guardrail_unavailable error frame");
    assert!(err_frame["error"]["message"].is_string());

    let content = assembled_content(&sse_data_chunks(&body));
    assert!(
        !content.contains("Partial answer"),
        "fail_close must discard the partial content, got: {content:?}"
    );
}

/// With a `fail_open` post-call stage, the same premature disconnect forwards
/// the partial content through post-call and re-chunks it to the caller,
/// terminating with `[DONE]` (Req 10.5).
#[tokio::test]
async fn premature_disconnect_fail_open_forwards_partial_content() {
    let mock = start_incomplete_sse_mock("Partial answer with no finish reason").await;
    let app = build_app(passthrough_config(
        &mock.uri(),
        post_call_guardrails(PolicyAction::Redact, FailurePolicy::FailOpen),
    ))
    .await;

    let (status, body) = post_stream(app).await;
    assert_eq!(status, StatusCode::OK);

    let text = std::str::from_utf8(&body).unwrap();
    assert!(text.contains("[DONE]"), "stream must close with [DONE]");
    assert!(
        !text.contains("guardrail_unavailable"),
        "fail_open must not emit a guardrail_unavailable error frame"
    );

    let content = assembled_content(&sse_data_chunks(&body));
    assert!(
        content.contains("Partial answer"),
        "fail_open forwards the partial content, got: {content:?}"
    );
}

// ---------------------------------------------------------------------------
// Req 9.5 (streaming) â€” pre-call redaction forces buffering for re-injection
// even when NO post-call stage is bound.
// ---------------------------------------------------------------------------

/// A guardrails section with a single PRE-CALL `redact` stage (no post-call
/// stage) bound as the global default. Redacts the `SECRET` token in the
/// request; the Re_Injection_Map must still be applied to the streamed response
/// so placeholders are restored (Req 9.5).
fn pre_call_redact_guardrails() -> GuardrailConfig {
    GuardrailConfig {
        providers: vec![GuardrailProviderConfig {
            name: "regex_secret".to_string(),
            provider_type: GuardrailProviderType::Regex,
            failure_policy: FailurePolicy::FailClose,
            timeout_seconds: 5,
            settings: ProviderSettings {
                patterns: vec![RegexPatternConfig {
                    name: "topsecret".to_string(),
                    regex: r"TOPSECRET\d+".to_string(),
                    entity: "SECRET".to_string(),
                    mode: RegexRuleMode::Deny,
                }],
                ..Default::default()
            },
        }],
        pipelines: vec![PipelineConfig {
            name: "pre_only".to_string(),
            stages: vec![StageConfig {
                name: "pii_redact".to_string(),
                provider: "regex_secret".to_string(),
                phase: StagePhase::PreCall,
                action: PolicyAction::Redact,
            }],
            redaction_notice_instruction: None,
            instruction_insertion_mode: InstructionInsertionMode::default(),
            failover_on_refusal: false,
            refusal_phrase_list: None,
            tool_result: ai_gateway::guardrail::config::ToolResultPhaseConfig::default(),
        }],
        global_default_pipeline: Some("pre_only".to_string()),
        bindings: Default::default(),
        ..Default::default()
    }
}

/// Send a streaming request with explicit user `content`.
async fn post_stream_content(app: axum::Router, content: &str) -> (StatusCode, Vec<u8>) {
    let body = serde_json::json!({
        "model": "gpt-4",
        "messages": [{"role": "user", "content": content}],
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

/// Req 9.5 (streaming): a pre-call `redact` stage with NO bound post-call stage
/// still forces the buffered path so PII placeholders are re-injected into the
/// streamed response. The upstream echoes the placeholder `<<PII_SECRET_1>>`;
/// the caller must receive the restored original (`TOPSECRET42`), never the raw
/// placeholder.
#[tokio::test]
async fn precall_redact_reinjects_placeholder_on_stream_without_postcall_stage() {
    // The mock echoes the placeholder the pre-call stage produced.
    let mock = start_json_mock("Here is <<PII_SECRET_1>> as requested").await;
    let app = build_app(buffered_config(&mock.uri(), pre_call_redact_guardrails())).await;

    // The request contains the secret; pre-call redaction maps
    // <<PII_SECRET_1>> -> TOPSECRET42 and forwards the redacted request.
    let (status, body) = post_stream_content(app, "my code is TOPSECRET42 keep it").await;
    assert_eq!(status, StatusCode::OK);

    let text = std::str::from_utf8(&body).unwrap();
    assert!(text.contains("[DONE]"), "stream must terminate with [DONE]");

    let content = assembled_content(&sse_data_chunks(&body));
    assert!(
        content.contains("TOPSECRET42"),
        "the placeholder must be re-injected to the original on stream, got: {content:?}"
    );
    assert!(
        !content.contains("<<PII_SECRET_1>>"),
        "the raw placeholder must NOT reach the caller, got: {content:?}"
    );
}
