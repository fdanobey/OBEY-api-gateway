//! Streaming (SSE) buffering support for post-call guardrails (Req 10).
//!
//! When a post-call guardrail pipeline is bound to a streaming request, the
//! gateway cannot relay upstream SSE chunks verbatim: policy evaluation needs
//! the *assembled* response. This module provides the pure, unit-testable
//! pieces the streaming handler ([`crate::gateway::handlers`]) wires into its
//! SSE generator:
//!
//! - [`SseBuffer`] — accumulates the upstream SSE body under a hard 10 MB cap
//!   (Req 10.1) and assembles it into an [`OpenAIResponse`], reporting whether a
//!   `finish_reason` was seen (completeness, Req 10.5).
//! - [`block_frame_payload`] — builds the terminal policy-violation SSE event
//!   emitted before `data: [DONE]` on a post-call `block` (Req 10.3).
//! - [`rechunk_after_early_event`] / [`rechunk_full`] — re-chunk the assembled
//!   (re-injected/redacted/replaced) response into SSE events matching the
//!   upstream chunk boundaries, reusing the handler's existing chunk synthesizer
//!   (Req 10.4).
//! - [`disconnect_discards_partial`] — decide whether a premature disconnect
//!   (no `finish_reason`) should discard partial content, from the bound
//!   post-call stages' failure policies (Req 10.5).
//!
//! The handler owns the async SSE generator (so it can emit keep-alive comments
//! at `keepalive_interval_seconds` while buffering, Req 10.2, and forward the
//! re-chunked events within 500 ms of analysis completion, Req 10.6); this
//! module owns the logic that is testable without a live provider connection.

use serde_json::Value;

use crate::models::openai::OpenAIResponse;
use crate::router::router::Router;

use super::{FailurePolicy, ResolvedStage, StagePhase};

/// Maximum size, in bytes, of the assembled SSE buffer before the gateway
/// aborts with an error (Req 10.1).
pub const MAX_STREAM_BUFFER_BYTES: usize = 10 * 1024 * 1024;

/// Error assembling or accumulating a buffered SSE response.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum BufferError {
    /// The accumulated buffer would exceed [`MAX_STREAM_BUFFER_BYTES`]
    /// (Req 10.1). The handler aborts the stream with a gateway error.
    TooLarge,
}

/// Append `chunk` to `buffer` iff doing so keeps the total length within `cap`
/// (Req 10.1). On overflow the buffer is left unmodified and [`BufferError::TooLarge`]
/// is returned so the caller can abort deterministically.
pub fn append_within_cap(buffer: &mut String, chunk: &str, cap: usize) -> Result<(), BufferError> {
    if buffer.len().saturating_add(chunk.len()) > cap {
        return Err(BufferError::TooLarge);
    }
    buffer.push_str(chunk);
    Ok(())
}

/// An assembled buffered response plus whether it terminated cleanly.
#[derive(Debug, Clone)]
pub struct Assembled {
    /// The response reassembled from the buffered SSE chunks.
    pub response: OpenAIResponse,
    /// `true` when a `finish_reason` was observed on the first choice — i.e. the
    /// upstream produced a complete response. `false` signals a premature
    /// disconnect for failure-policy handling (Req 10.5).
    pub complete: bool,
}

/// Accumulates the upstream SSE body under the 10 MB cap and assembles it into
/// a complete [`OpenAIResponse`] for post-call guardrail evaluation (Req 10.1).
#[derive(Debug)]
pub struct SseBuffer {
    body: String,
    cap: usize,
}

impl SseBuffer {
    /// Create a buffer with an explicit byte cap.
    pub fn new(cap: usize) -> Self {
        Self {
            body: String::new(),
            cap,
        }
    }

    /// Create a buffer with the default 10 MB cap ([`MAX_STREAM_BUFFER_BYTES`]).
    pub fn with_default_cap() -> Self {
        Self::new(MAX_STREAM_BUFFER_BYTES)
    }

    /// Append a raw upstream byte chunk, decoding as UTF-8 (lossy — SSE bodies
    /// are UTF-8; any invalid bytes are replaced rather than aborting the
    /// stream). Enforces the cap (Req 10.1).
    pub fn push_bytes(&mut self, bytes: &[u8]) -> Result<(), BufferError> {
        let text = String::from_utf8_lossy(bytes);
        append_within_cap(&mut self.body, &text, self.cap)
    }

    /// Append a decoded string chunk, enforcing the cap (Req 10.1).
    #[allow(dead_code)] // used by tests; the handler pushes raw bytes in the binary build
    pub fn push_str(&mut self, text: &str) -> Result<(), BufferError> {
        append_within_cap(&mut self.body, text, self.cap)
    }

    /// The number of bytes accumulated so far.
    #[allow(dead_code)] // used by tests; unused in the binary build
    pub fn len(&self) -> usize {
        self.body.len()
    }

    /// Whether nothing has been accumulated yet.
    #[allow(dead_code)] // public API / test-only; unused in the binary build
    pub fn is_empty(&self) -> bool {
        self.body.is_empty()
    }

    /// Borrow the accumulated raw SSE body.
    #[allow(dead_code)] // public accessor / test-only; unused in the binary build
    pub fn body(&self) -> &str {
        &self.body
    }

    /// Assemble the accumulated SSE body into a response, reporting whether it
    /// terminated with a `finish_reason` (Req 10.5). Returns `Err` when the body
    /// contained no usable chunks or a mid-stream error frame.
    pub fn assemble(&self) -> Result<Assembled, String> {
        assemble(&self.body)
    }
}

/// Assemble a buffered SSE `body` into a response and report completeness.
///
/// Reuses [`Router::reassemble_sse_response`] so streaming assembly is identical
/// to the buffered/pass-through cache path. `complete` is `true` when the first
/// choice carries a `finish_reason` (Req 10.5).
pub fn assemble(body: &str) -> Result<Assembled, String> {
    let response = Router::reassemble_sse_response(body)?;
    let complete = response
        .choices
        .first()
        .and_then(|c| c.finish_reason.as_ref())
        .is_some();
    Ok(Assembled { response, complete })
}

/// Build the terminal policy-violation SSE event payload for a post-call
/// `block` (Req 10.3). The handler emits this as a `data:` event and then the
/// `data: [DONE]` sentinel to close the stream.
///
/// The shape matches the non-streaming guardrail 403 body
/// (`{"error":{"message","type","category"}}`) so streaming and non-streaming
/// callers observe an identical error contract.
pub fn block_frame_payload(category: &str) -> Value {
    serde_json::json!({
        "error": {
            "message": "Request blocked by guardrail policy",
            "type": "guardrail_policy_violation",
            "category": category,
        }
    })
}

/// Re-chunk an assembled response into SSE chunk payloads *after* an early
/// `role: assistant` event has already been emitted (Req 10.4). Reuses the
/// handler's existing chunk synthesizer so streaming output is byte-compatible
/// with the buffer-and-replay path, sharing the early event's `id`/`created`.
pub fn rechunk_after_early_event(response: &OpenAIResponse, id: &str, created: i64) -> Vec<Value> {
    crate::gateway::handlers::streaming_chunks_after_early_event(response, id, created)
}

/// Re-chunk an assembled response into SSE chunk payloads including the leading
/// `role: assistant` delta (used when no early event was emitted) (Req 10.4).
pub fn rechunk_full(response: &OpenAIResponse) -> Vec<Value> {
    crate::gateway::handlers::streaming_chunks_from_response(response)
}

/// Decide whether a premature upstream disconnect (no `finish_reason` observed
/// while buffering) should discard the partial content (Req 10.5).
///
/// Applies the bound post-call stages' failure policy: `fail_close` discards
/// (the partial content cannot be certified safe), `fail_open` forwards. If any
/// bound post-call stage is `fail_close`, the strictest policy wins and the
/// partial content is discarded.
pub fn disconnect_discards_partial(stages: &[ResolvedStage]) -> bool {
    stages
        .iter()
        .filter(|s| s.phase == StagePhase::PostCall)
        .any(|s| matches!(s.failure_policy, FailurePolicy::FailClose))
}

/// Indicates whether the assembled response (full or partial due to premature
/// termination) should proceed through refusal detection and the failover loop
/// (Req 12.9, 12.14).
///
/// The handler calls this decision point after assembly:
/// - When `complete == true`, detection runs on the fully assembled response
///   before the failover decision (Req 12.9).
/// - When `complete == false` (premature termination) and the failure policy
///   permits forwarding partial content (fail_open), detection runs on the
///   partially assembled content and the same failover decision applies (Req
///   12.14).
///
/// Returns `true` when refusal detection should proceed on the assembled
/// content (both full and partial-forwarded cases).
#[allow(dead_code)] // used by tests and serves as documented decision logic
pub fn should_run_refusal_detection(assembled: &Assembled, discard_on_disconnect: bool) -> bool {
    // Full response: always run detection (Req 12.9).
    if assembled.complete {
        return true;
    }
    // Partial (premature termination): run detection only when failure policy
    // forwards the partial content (fail_open, Req 12.14).
    !discard_on_disconnect
}

#[cfg(test)]
mod tests {
    use super::*;

    fn sample_response() -> OpenAIResponse {
        serde_json::from_value(serde_json::json!({
            "id": "chatcmpl-test",
            "object": "chat.completion",
            "created": 1_700_000_000i64,
            "model": "gpt-4o",
            "choices": [{
                "index": 0,
                "message": { "role": "assistant", "content": "Hello world" },
                "finish_reason": "stop"
            }],
            "usage": { "prompt_tokens": 1, "completion_tokens": 2, "total_tokens": 3 }
        }))
        .expect("valid response fixture")
    }

    // -------------------------------------------------------------------
    // 10 MB cap enforcement (Req 10.1)
    // -------------------------------------------------------------------

    #[test]
    fn append_within_cap_accepts_up_to_limit() {
        let mut buf = String::new();
        assert_eq!(append_within_cap(&mut buf, "abc", 5), Ok(()));
        assert_eq!(append_within_cap(&mut buf, "de", 5), Ok(()));
        assert_eq!(buf, "abcde");
    }

    #[test]
    fn append_within_cap_rejects_overflow_and_leaves_buffer_intact() {
        let mut buf = String::from("abcd");
        assert_eq!(
            append_within_cap(&mut buf, "ef", 5),
            Err(BufferError::TooLarge)
        );
        // Buffer unmodified on overflow so the caller can abort deterministically.
        assert_eq!(buf, "abcd");
    }

    #[test]
    fn sse_buffer_enforces_default_10mb_cap() {
        let mut buf = SseBuffer::new(16);
        assert!(buf.push_str("0123456789").is_ok());
        // Next push would take it to 22 bytes > 16.
        assert_eq!(buf.push_str("abcdef012"), Err(BufferError::TooLarge));
        assert_eq!(buf.len(), 10);
        assert_eq!(MAX_STREAM_BUFFER_BYTES, 10 * 1024 * 1024);
    }

    #[test]
    fn sse_buffer_push_bytes_counts_toward_cap() {
        let mut buf = SseBuffer::new(4);
        assert!(buf.push_bytes(b"ab").is_ok());
        assert_eq!(buf.push_bytes(b"cde"), Err(BufferError::TooLarge));
    }

    // -------------------------------------------------------------------
    // Assembly + completeness (Req 10.5)
    // -------------------------------------------------------------------

    #[test]
    fn assemble_reports_complete_when_finish_reason_present() {
        let body = "data: {\"id\":\"x\",\"model\":\"m\",\"choices\":[{\"index\":0,\"delta\":{\"content\":\"Hi\"},\"finish_reason\":\"stop\"}]}\n\n\
                    data: [DONE]\n\n";
        let assembled = assemble(body).expect("assembles");
        assert!(assembled.complete);
        assert_eq!(
            assembled.response.choices[0].message.content_as_text(),
            "Hi"
        );
    }

    #[test]
    fn assemble_reports_incomplete_without_finish_reason() {
        // No finish_reason, no [DONE] — a premature disconnect (Req 10.5).
        let body =
            "data: {\"id\":\"x\",\"model\":\"m\",\"choices\":[{\"index\":0,\"delta\":{\"content\":\"partial\"}}]}\n\n";
        let assembled = assemble(body).expect("assembles");
        assert!(!assembled.complete);
        assert_eq!(
            assembled.response.choices[0].message.content_as_text(),
            "partial"
        );
    }

    #[test]
    fn assemble_errors_on_empty_body() {
        assert!(assemble("").is_err());
    }

    // -------------------------------------------------------------------
    // Terminal block frame formatting (Req 10.3)
    // -------------------------------------------------------------------

    #[test]
    fn block_frame_payload_carries_category_and_type() {
        let payload = block_frame_payload("US_SSN");
        assert_eq!(payload["error"]["type"], "guardrail_policy_violation");
        assert_eq!(payload["error"]["category"], "US_SSN");
        assert_eq!(
            payload["error"]["message"],
            "Request blocked by guardrail policy"
        );
        // Serializes as a single-line JSON object suitable for an SSE `data:` frame.
        let s = payload.to_string();
        assert!(!s.contains('\n'));
    }

    // -------------------------------------------------------------------
    // Re-chunking (Req 10.4)
    // -------------------------------------------------------------------

    #[test]
    fn rechunk_full_includes_role_delta_and_content() {
        let response = sample_response();
        let chunks = rechunk_full(&response);
        assert!(!chunks.is_empty());
        // The first chunk carries the role marker when no early event preceded it.
        let first = &chunks[0];
        assert_eq!(first["choices"][0]["delta"]["role"], "assistant");
        // Some chunk carries the content.
        let has_content = chunks.iter().any(|c| {
            c["choices"][0]["delta"]["content"]
                .as_str()
                .map(|s| s.contains("Hello world"))
                .unwrap_or(false)
        });
        assert!(has_content, "content chunk present: {chunks:?}");
    }

    #[test]
    fn rechunk_after_early_event_shares_id_and_skips_role() {
        let response = sample_response();
        let chunks = rechunk_after_early_event(&response, "chatcmpl-shared", 4242);
        assert!(!chunks.is_empty());
        // Every chunk reuses the shared id/created from the early event (Req 10.4).
        for chunk in &chunks {
            assert_eq!(chunk["id"], "chatcmpl-shared");
            assert_eq!(chunk["created"], 4242);
            // Role delta suppressed since the early event already emitted it.
            assert!(chunk["choices"][0]["delta"].get("role").is_none());
        }
    }

    // -------------------------------------------------------------------
    // Disconnect failure policy (Req 10.5)
    // -------------------------------------------------------------------

    #[test]
    fn disconnect_discards_when_any_post_stage_fail_close() {
        let stages = vec![
            make_stage(StagePhase::PostCall, FailurePolicy::FailOpen),
            make_stage(StagePhase::PostCall, FailurePolicy::FailClose),
        ];
        assert!(disconnect_discards_partial(&stages));
    }

    #[test]
    fn disconnect_forwards_when_all_post_stages_fail_open() {
        let stages = vec![
            make_stage(StagePhase::PreCall, FailurePolicy::FailClose),
            make_stage(StagePhase::PostCall, FailurePolicy::FailOpen),
        ];
        assert!(!disconnect_discards_partial(&stages));
    }

    // -------------------------------------------------------------------
    // should_run_refusal_detection (Req 12.9, 12.14)
    // -------------------------------------------------------------------

    #[test]
    fn refusal_detection_runs_on_complete_response() {
        // Req 12.9: detection runs on fully assembled response regardless of
        // failure policy.
        let assembled = Assembled {
            response: sample_response(),
            complete: true,
        };
        assert!(should_run_refusal_detection(&assembled, true));
        assert!(should_run_refusal_detection(&assembled, false));
    }

    #[test]
    fn refusal_detection_runs_on_partial_when_fail_open() {
        // Req 12.14: premature termination + fail_open → detection on partial.
        let assembled = Assembled {
            response: sample_response(),
            complete: false,
        };
        assert!(should_run_refusal_detection(&assembled, false));
    }

    #[test]
    fn refusal_detection_skipped_on_partial_when_fail_close() {
        // Req 12.14 inverse: premature termination + fail_close → discard, no
        // detection (the handler aborts before reaching the detection point).
        let assembled = Assembled {
            response: sample_response(),
            complete: false,
        };
        assert!(!should_run_refusal_detection(&assembled, true));
    }

    // Build a minimal ResolvedStage for failure-policy tests. Uses a trivial
    // no-op provider so no network/config is required.
    fn make_stage(phase: StagePhase, failure_policy: FailurePolicy) -> ResolvedStage {
        use crate::guardrail::{Finding, GuardrailProvider, GuardrailProviderError, PolicyAction};
        use std::sync::Arc;
        use std::time::Duration;

        struct NoopProvider;
        #[async_trait::async_trait]
        impl GuardrailProvider for NoopProvider {
            async fn analyze(
                &self,
                _content: &str,
            ) -> Result<Vec<Finding>, GuardrailProviderError> {
                Ok(Vec::new())
            }
            fn provider_type(&self) -> &'static str {
                "noop"
            }
        }

        ResolvedStage {
            pipeline_name: "p".to_string(),
            stage_name: "s".to_string(),
            provider: Arc::new(NoopProvider),
            provider_type: "noop",
            failure_policy,
            action: PolicyAction::Block,
            timeout: Duration::from_secs(5),
            phase,
        }
    }
}
