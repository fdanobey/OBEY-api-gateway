//! Representative smoke timing checks for guardrail latency budgets (task 13.6).
//!
//! These are *smoke* checks, not micro-benchmarks: they assert that the pure,
//! in-process guardrail hot paths complete comfortably under their specified
//! budgets using a realistic regex provider, with generous margins so CI timing
//! jitter does not cause flakes.
//!
//! Covered budgets:
//! - Pre-call latency budget (Req 2.8): the pre-call stage completes scanning +
//!   action enforcement within 100 ms for bodies up to 50 KB, and within 500 ms
//!   for bodies exceeding 50 KB. We drive `GuardrailEngine::run_pre_call` with a
//!   `RegexProvider` (a few deny patterns + a redact action) over a ~50 KB body
//!   and a >50 KB body.
//! - Streaming forward-within-500 ms (Req 10.6): after post-call analysis, the
//!   assembled response is re-chunked into SSE and forwarded within 500 ms. We
//!   measure the deterministic in-process path — `stream::assemble` + a regex
//!   `run_post_call` + `stream::rechunk_full` — rather than a flaky endpoint.

use std::time::Instant;

use ai_gateway::guardrail::config::{
    GuardrailConfig, GuardrailProviderConfig, GuardrailProviderType, InstructionInsertionMode,
    PipelineConfig, PolicyAction, ProviderSettings, RegexPatternConfig, RegexRuleMode, StageConfig,
    StagePhase,
};
use ai_gateway::guardrail::factory::build_engine;
use ai_gateway::guardrail::pii::GuardrailContext;
use ai_gateway::guardrail::pipeline::BindingSelector;
use ai_gateway::guardrail::{stream, FailurePolicy};
use ai_gateway::models::openai::{OpenAIRequest, OpenAIResponse};

use reqwest::Client;
use serde_json::json;

// ---------------------------------------------------------------------------
// Config helpers
// ---------------------------------------------------------------------------

/// A realistic set of deny patterns a pre-call PII/secret stage might carry.
fn deny_patterns() -> Vec<RegexPatternConfig> {
    vec![
        RegexPatternConfig {
            name: "email".to_string(),
            regex: r"[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}".to_string(),
            entity: "EMAIL".to_string(),
            mode: RegexRuleMode::Deny,
        },
        RegexPatternConfig {
            name: "us_ssn".to_string(),
            regex: r"\b\d{3}-\d{2}-\d{4}\b".to_string(),
            entity: "US_SSN".to_string(),
            mode: RegexRuleMode::Deny,
        },
        RegexPatternConfig {
            name: "api_key".to_string(),
            regex: r"sk-[A-Za-z0-9]{20,}".to_string(),
            entity: "API_KEY".to_string(),
            mode: RegexRuleMode::Deny,
        },
    ]
}

/// Build an engine whose global-default pipeline has one pre-call `redact`
/// stage and one post-call `redact` stage, both backed by a regex provider.
fn build_regex_engine() -> ai_gateway::guardrail::GuardrailEngine {
    let config = GuardrailConfig {
        providers: vec![GuardrailProviderConfig {
            name: "scanner".to_string(),
            provider_type: GuardrailProviderType::Regex,
            failure_policy: FailurePolicy::FailOpen,
            timeout_seconds: 5,
            settings: ProviderSettings {
                patterns: deny_patterns(),
                ..Default::default()
            },
        }],
        pipelines: vec![PipelineConfig {
            name: "std".to_string(),
            stages: vec![
                StageConfig {
                    name: "pre-redact".to_string(),
                    provider: "scanner".to_string(),
                    phase: StagePhase::PreCall,
                    action: PolicyAction::Redact,
                },
                StageConfig {
                    name: "post-redact".to_string(),
                    provider: "scanner".to_string(),
                    phase: StagePhase::PostCall,
                    action: PolicyAction::Redact,
                },
            ],
            redaction_notice_instruction: None,
            instruction_insertion_mode: InstructionInsertionMode::default(),
            failover_on_refusal: false,
            refusal_phrase_list: None,
        }],
        global_default_pipeline: Some("std".to_string()),
        ..Default::default()
    };

    build_engine(&config, &Client::new(), None, None).expect("engine builds")
}

/// Build a chat request whose user message content is at least `target_bytes`
/// of realistic prose with a few embedded PII-like values sprinkled in.
fn request_with_body(target_bytes: usize) -> OpenAIRequest {
    // A repeated sentence with occasional detectable values, so the regex
    // provider does real matching work rather than scanning inert filler.
    let unit = "The quick brown fox contacts user@example.com about invoice 123-45-6789. ";
    let mut content = String::with_capacity(target_bytes + unit.len());
    while content.len() < target_bytes {
        content.push_str(unit);
    }

    serde_json::from_value(json!({
        "model": "gpt-4o",
        "messages": [
            { "role": "system", "content": "You are a helpful assistant." },
            { "role": "user", "content": content }
        ]
    }))
    .expect("valid request fixture")
}

/// Build a moderate multi-choice-free SSE body assembled from several deltas,
/// mimicking a streamed assistant response of a few KB.
fn moderate_sse_body() -> String {
    let mut body = String::new();
    // Leading role delta.
    body.push_str(
        "data: {\"id\":\"chatcmpl-x\",\"model\":\"gpt-4o\",\"choices\":[{\"index\":0,\"delta\":{\"role\":\"assistant\"}}]}\n\n",
    );
    // ~200 content deltas of a short phrase → a few KB assembled.
    for i in 0..200 {
        let frag = format!("chunk {i} of the streamed answer; ");
        let escaped = frag.replace('"', "\\\"");
        body.push_str(&format!(
            "data: {{\"id\":\"chatcmpl-x\",\"model\":\"gpt-4o\",\"choices\":[{{\"index\":0,\"delta\":{{\"content\":\"{escaped}\"}}}}]}}\n\n"
        ));
    }
    // Terminal chunk with finish_reason + DONE sentinel.
    body.push_str(
        "data: {\"id\":\"chatcmpl-x\",\"model\":\"gpt-4o\",\"choices\":[{\"index\":0,\"delta\":{},\"finish_reason\":\"stop\"}]}\n\n",
    );
    body.push_str("data: [DONE]\n\n");
    body
}

// ---------------------------------------------------------------------------
// Req 2.8 — pre-call latency budget
// ---------------------------------------------------------------------------

#[tokio::test]
async fn pre_call_scan_of_50kb_body_completes_well_under_100ms() {
    let engine = build_regex_engine();
    let selector = BindingSelector::default();

    // ~50 KB user message (at or just above 50 * 1024 bytes).
    let mut request = request_with_body(50 * 1024);
    assert!(request.messages[1].content_as_text().len() >= 50 * 1024);

    let mut ctx = GuardrailContext::default();

    let start = Instant::now();
    let outcome = engine
        .run_pre_call(&mut request, &selector, &mut ctx, "trace-50kb")
        .await;
    let elapsed = start.elapsed();

    // Budget: 100 ms for bodies up to 50 KB (Req 2.8). Smoke assertion keeps the
    // full budget as the ceiling; the in-process regex scan is expected to land
    // far below it, so no tighter bound is enforced to avoid CI flakiness.
    assert!(
        matches!(outcome, ai_gateway::guardrail::PreCallOutcome::Proceed),
        "pre-call should proceed (redact), got {outcome:?}"
    );
    assert!(
        elapsed.as_millis() < 100,
        "pre-call 50KB scan took {elapsed:?}, budget is 100ms (Req 2.8)"
    );
}

#[tokio::test]
async fn pre_call_scan_of_large_body_completes_well_under_500ms() {
    let engine = build_regex_engine();
    let selector = BindingSelector::default();

    // >50 KB user message (~200 KB) exercises the larger-body budget.
    let mut request = request_with_body(200 * 1024);
    assert!(request.messages[1].content_as_text().len() > 50 * 1024);

    let mut ctx = GuardrailContext::default();

    let start = Instant::now();
    let outcome = engine
        .run_pre_call(&mut request, &selector, &mut ctx, "trace-large")
        .await;
    let elapsed = start.elapsed();

    // Budget: 500 ms for bodies exceeding 50 KB (Req 2.8).
    assert!(
        matches!(outcome, ai_gateway::guardrail::PreCallOutcome::Proceed),
        "pre-call should proceed (redact), got {outcome:?}"
    );
    assert!(
        elapsed.as_millis() < 500,
        "pre-call >50KB scan took {elapsed:?}, budget is 500ms (Req 2.8)"
    );
}

// ---------------------------------------------------------------------------
// Req 10.6 — streaming forward-within-500ms (analysis + re-chunk)
// ---------------------------------------------------------------------------

#[tokio::test]
async fn stream_assemble_analyze_and_rechunk_completes_well_under_500ms() {
    let engine = build_regex_engine();
    let selector = BindingSelector::default();

    let body = moderate_sse_body();

    // Assemble the buffered SSE body into a response (upstream buffering already
    // completed; we measure the analysis + forward path only, per Req 10.6).
    let assembled = stream::assemble(&body).expect("assembles");
    assert!(assembled.complete, "fixture ends with finish_reason");
    let mut response: OpenAIResponse = assembled.response;

    let mut ctx = GuardrailContext::default();

    let start = Instant::now();
    // Post-call analysis (regex redact stage) ...
    let tool_ctx = ai_gateway::guardrail::ToolContext {
        tool_use_allowed: false,
        tools_provided: false,
        finish_reason_is_tool_call: false,
        has_tool_calls: false,
    };
    let (outcome, _refusal) = engine
        .run_post_call(&mut response, &selector, &mut ctx, "trace-stream", &tool_ctx)
        .await;
    // ... immediately followed by re-chunking the assembled response into SSE.
    let chunks = stream::rechunk_full(&response);
    let elapsed = start.elapsed();

    assert!(
        matches!(outcome, ai_gateway::guardrail::PostCallOutcome::Proceed),
        "post-call should proceed, got {outcome:?}"
    );
    assert!(!chunks.is_empty(), "re-chunking yields SSE events");
    // Budget: forward the re-chunked SSE within 500 ms of analysis completion
    // (Req 10.6). Measured across analysis + re-chunk as a generous ceiling.
    assert!(
        elapsed.as_millis() < 500,
        "analysis + re-chunk took {elapsed:?}, budget is 500ms (Req 10.6)"
    );
}
