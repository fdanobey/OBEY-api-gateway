//! Strip/preserve policy for reasoning state on model transitions
//! (design Component 2).
//!
//! [`decide`] turns a detected [`ReasoningFootprint`], the source-model
//! attribution (when known), the failover target, and the
//! [`ReasoningCompatConfig`] knobs into a [`StripDecision`]. [`apply`]
//! mutates the cloned outgoing request accordingly and returns a
//! [`StripReport`] of counts only — payload text, signatures, and redacted
//! data are never retained, returned, or logged.
//!
//! Core correctness rule (the reported bug class): Anthropic
//! `thinking`/`redacted_thinking` blocks are signed, encrypted, model-bound
//! state. They survive verbatim only for same resolved provider-model
//! continuations (mid-tool-loop echoes); every other transition strips ALL
//! reasoning carriers. Filtering only `type == "thinking"` is
//! non-compliant — `redacted_thinking` must be removed too.

use crate::models::openai::{Message, OpenAIRequest};
use crate::reasoning_compat::config::{ReasoningCompatConfig, ReasoningFamily};
use crate::reasoning_compat::detect::ReasoningFootprint;

/// Outcome of the strip/preserve policy for one failover attempt.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum StripDecision {
    /// Keep the request verbatim (same-model continuation, empty footprint,
    /// or the feature knob is off).
    Preserve,
    /// Remove every reasoning carrier (any cross-model transition).
    StripAll,
    /// Same removal as [`StripDecision::StripAll`], but the source model is
    /// unknown (no affinity record). Logged distinctly so operators can see
    /// how often attribution is missing.
    StripAttributionUnknown,
}

impl StripDecision {
    /// Stable log-field representation.
    pub fn as_str(self) -> &'static str {
        match self {
            StripDecision::Preserve => "preserve",
            StripDecision::StripAll => "strip_all",
            StripDecision::StripAttributionUnknown => "strip_attribution_unknown",
        }
    }

    /// True when [`apply`] must strip reasoning carriers from the request.
    pub fn strips(self) -> bool {
        matches!(
            self,
            StripDecision::StripAll | StripDecision::StripAttributionUnknown
        )
    }
}

/// Lightweight source/target model reference for policy decisions.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ModelRef {
    pub provider: String,
    pub model: String,
    pub family: ReasoningFamily,
}

/// Counts-only record of what [`apply`] removed. Never contains payloads
/// (no thinking text, no signatures, no redacted data).
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub struct StripReport {
    /// Messages a carrier was removed from (dropped messages included).
    pub messages_touched: usize,
    /// `thinking` content blocks removed.
    pub thinking_blocks: usize,
    /// `redacted_thinking` content blocks removed.
    pub redacted_thinking_blocks: usize,
    /// Extra fields removed (`reasoning_content`, `reasoning`), carrier
    /// metadata fields dropped from surviving sibling blocks
    /// (`signature`/`data`/`thinking`), and removed Responses-style
    /// `reasoning` content items (no dedicated counter).
    pub fields_removed: usize,
}

/// Decide whether prior-turn reasoning state survives this transition.
///
/// Rules (requirements 2.1–2.8):
/// - Empty footprint → [`StripDecision::Preserve`] (nothing to do).
/// - `strip_on_model_change: false` → [`StripDecision::Preserve`] (knob).
/// - Known source: preserve only for the exact same resolved
///   provider + model + family (same-model tool-loop continuation — the
///   echo must be verbatim, `redacted_thinking` included). Any different
///   model (even within the same family: signatures are model-bound) or
///   different family → [`StripDecision::StripAll`].
/// - Unknown source (no affinity record): preserve only when the target
///   family matches the footprint's inferred source family and that family
///   is classified; otherwise → [`StripDecision::StripAttributionUnknown`]
///   (conservative strip, logged distinctly).
pub fn decide(
    footprint: &ReasoningFootprint,
    source: Option<&ModelRef>,
    target: &ModelRef,
    cfg: &ReasoningCompatConfig,
) -> StripDecision {
    if footprint.is_empty() {
        return StripDecision::Preserve;
    }
    if !cfg.strip_on_model_change {
        return StripDecision::Preserve;
    }

    match source {
        Some(source) => {
            let same_resolved_model = source.provider == target.provider
                && source.model == target.model;
            if same_resolved_model && source.family == target.family {
                StripDecision::Preserve
            } else {
                StripDecision::StripAll
            }
        }
        None => {
            if footprint.source_family != ReasoningFamily::None
                && footprint.source_family == target.family
            {
                StripDecision::Preserve
            } else {
                StripDecision::StripAttributionUnknown
            }
        }
    }
}

/// Apply a [`StripDecision`] to the outgoing request.
///
/// [`StripDecision::Preserve`] returns an empty report without touching the
/// request (bit-identical passthrough). The strip variants walk
/// `outgoing.messages` once and, per message:
///
/// - drop content blocks with `type` `thinking`, `redacted_thinking`, or
///   `reasoning` (the regression fix: `redacted_thinking` is stripped just
///   like `thinking`)
/// - remove carrier metadata (`signature`/`data`/`thinking`) from surviving
///   sibling blocks when a reasoning block was removed next to them
/// - remove `reasoning_content` and `reasoning` extra fields
/// - drop the message entirely only when nothing remains: no text content,
///   no `tool_calls` extra, no `tool_use` block
///
/// The returned [`StripReport`] carries counts only.
pub fn apply(outgoing: &mut OpenAIRequest, decision: StripDecision) -> StripReport {
    let mut report = StripReport::default();
    if !decision.strips() {
        return report;
    }

    let mut kept: Vec<Message> = Vec::with_capacity(outgoing.messages.len());
    for message in std::mem::take(&mut outgoing.messages) {
        let mut msg = message;
        let mut touched = false;

        if let Some(blocks) = msg.content.as_array_mut() {
            let mut retained: Vec<serde_json::Value> = Vec::with_capacity(blocks.len());
            let mut removed_reasoning_block = false;
            for block in blocks.drain(..) {
                match block.get("type").and_then(serde_json::Value::as_str) {
                    Some("thinking") => {
                        report.thinking_blocks += 1;
                        removed_reasoning_block = true;
                        touched = true;
                    }
                    Some("redacted_thinking") => {
                        report.redacted_thinking_blocks += 1;
                        removed_reasoning_block = true;
                        touched = true;
                    }
                    Some("reasoning") => {
                        report.fields_removed += 1;
                        touched = true;
                    }
                    _ => retained.push(block),
                }
            }
            if removed_reasoning_block {
                for block in &mut retained {
                    strip_carrier_metadata(block, &mut report, &mut touched);
                }
            }
            *blocks = retained;
        }

        for field in ["reasoning_content", "reasoning"] {
            if msg.extra.remove(field).is_some() {
                report.fields_removed += 1;
                touched = true;
            }
        }

        if touched {
            report.messages_touched += 1;
            if message_retains_content(&msg) {
                kept.push(msg);
            }
        } else {
            kept.push(msg);
        }
    }

    outgoing.messages = kept;
    report
}

/// Emit the structured trace-id log for an applied decision. Counts only —
/// never payloads.
pub fn log_strip_action(report: &StripReport, decision: StripDecision, trace_id: &str) {
    tracing::info!(
        trace_id = %trace_id,
        action = %decision.as_str(),
        messages_touched = report.messages_touched,
        thinking_blocks = report.thinking_blocks,
        redacted_thinking_blocks = report.redacted_thinking_blocks,
        fields_removed = report.fields_removed,
        "reasoning_compat strip decision applied"
    );
}

/// Remove `signature`/`data`/`thinking` carrier-metadata fields from a
/// surviving sibling block (requirement 1.3: never forwarded independently
/// of their reasoning block).
fn strip_carrier_metadata(
    block: &mut serde_json::Value,
    report: &mut StripReport,
    touched: &mut bool,
) {
    let Some(object) = block.as_object_mut() else {
        return;
    };
    for field in ["signature", "data", "thinking"] {
        if object.remove(field).is_some() {
            report.fields_removed += 1;
            *touched = true;
        }
    }
}

/// True when the message still carries forwardable content after stripping:
/// non-empty string content, a `text` or `tool_use` block, or `tool_calls`
/// in `extra`.
fn message_retains_content(msg: &Message) -> bool {
    if has_tool_calls(msg) {
        return true;
    }
    match &msg.content {
        serde_json::Value::String(text) => !text.is_empty(),
        serde_json::Value::Array(blocks) => blocks.iter().any(|block| {
            matches!(
                block.get("type").and_then(serde_json::Value::as_str),
                Some("text") | Some("tool_use")
            )
        }),
        _ => false,
    }
}

fn has_tool_calls(msg: &Message) -> bool {
    match msg.extra.get("tool_calls") {
        Some(serde_json::Value::Array(calls)) => !calls.is_empty(),
        Some(serde_json::Value::Null) | None => false,
        Some(_) => true,
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::reasoning_compat::detect::detect;
    use serde_json::{json, Map, Value};

    fn assistant(content: Value, extra: Map<String, Value>) -> Message {
        Message {
            role: "assistant".to_string(),
            content,
            extra,
        }
    }

    fn user(content: Value) -> Message {
        Message {
            role: "user".to_string(),
            content,
            extra: Map::new(),
        }
    }

    fn request(messages: Vec<Message>) -> OpenAIRequest {
        OpenAIRequest {
            model: "target-model".to_string(),
            messages,
            stream: false,
            temperature: None,
            max_tokens: None,
            extra: Map::new(),
        }
    }

    fn model_ref(provider: &str, model: &str, family: ReasoningFamily) -> ModelRef {
        ModelRef {
            provider: provider.to_string(),
            model: model.to_string(),
            family,
        }
    }

    fn thinking_block() -> Value {
        json!({"type": "thinking", "thinking": "deep", "signature": "sig"})
    }

    fn redacted_block() -> Value {
        json!({"type": "redacted_thinking", "data": "opaque"})
    }

    fn text_block() -> Value {
        json!({"type": "text", "text": "answer"})
    }

    fn anthropic_conversation() -> Vec<Message> {
        vec![
            user(json!("hi")),
            assistant(
                json!([thinking_block(), text_block()]),
                Map::new(),
            ),
        ]
    }

    #[test]
    fn same_model_and_family_preserves_request_verbatim() {
        let messages = anthropic_conversation();
        let footprint = detect(&messages);
        assert!(!footprint.is_empty());

        let cfg = ReasoningCompatConfig::default();
        let source = model_ref(
            "prov-a",
            "claude-4-5-sonnet",
            ReasoningFamily::AnthropicManual,
        );
        let target = model_ref(
            "prov-a",
            "claude-4-5-sonnet",
            ReasoningFamily::AnthropicManual,
        );

        let decision = decide(&footprint, Some(&source), &target, &cfg);
        assert_eq!(decision, StripDecision::Preserve);

        let mut outgoing = request(messages);
        let before = serde_json::to_value(&outgoing.messages).expect("serializable");
        let report = apply(&mut outgoing, decision);
        assert_eq!(report, StripReport::default());
        let after = serde_json::to_value(&outgoing.messages).expect("serializable");
        assert_eq!(before, after);
    }

    #[test]
    fn cross_model_strip_removes_thinking_and_redacted_thinking() {
        let messages = vec![assistant(
            json!([thinking_block(), redacted_block(), text_block()]),
            Map::new(),
        )];
        let footprint = detect(&messages);
        let cfg = ReasoningCompatConfig::default();
        let source = model_ref(
            "prov-a",
            "claude-4-5-sonnet",
            ReasoningFamily::AnthropicManual,
        );
        let target = model_ref(
            "prov-a",
            "claude-4-7-opus",
            ReasoningFamily::AnthropicAdaptive,
        );

        let decision = decide(&footprint, Some(&source), &target, &cfg);
        assert_eq!(decision, StripDecision::StripAll);

        let mut outgoing = request(messages);
        let report = apply(&mut outgoing, decision);
        assert_eq!(report.thinking_blocks, 1);
        assert_eq!(report.redacted_thinking_blocks, 1);
        assert_eq!(report.messages_touched, 1);
        assert_eq!(outgoing.messages.len(), 1);
        assert_eq!(outgoing.messages[0].content, json!([text_block()]));
    }

    #[test]
    fn cross_family_strip_removes_reasoning_content_extra() {
        let messages = vec![assistant(
            json!("answer"),
            [(
                "reasoning_content".to_string(),
                json!("chain of thought"),
            )]
            .into_iter()
            .collect(),
        )];
        let footprint = detect(&messages);
        let cfg = ReasoningCompatConfig::default();
        let source = model_ref("prov-a", "deepseek-chat", ReasoningFamily::DeepSeek);
        let target = model_ref(
            "prov-b",
            "claude-4-5-sonnet",
            ReasoningFamily::AnthropicManual,
        );

        let decision = decide(&footprint, Some(&source), &target, &cfg);
        assert_eq!(decision, StripDecision::StripAll);

        let mut outgoing = request(messages);
        let report = apply(&mut outgoing, decision);
        assert_eq!(report.fields_removed, 1);
        assert!(!outgoing.messages[0].extra.contains_key("reasoning_content"));
        assert_eq!(outgoing.messages[0].content, json!("answer"));
        assert_eq!(outgoing.messages.len(), 1);
    }

    #[test]
    fn cross_family_strip_removes_reasoning_extra() {
        let messages = vec![assistant(
            json!("answer"),
            [("reasoning".to_string(), json!("chain of thought"))]
                .into_iter()
                .collect(),
        )];
        let footprint = detect(&messages);
        let cfg = ReasoningCompatConfig::default();
        let source = model_ref("prov-a", "grok-4", ReasoningFamily::OpenRouter);
        let target = model_ref(
            "prov-b",
            "claude-4-5-sonnet",
            ReasoningFamily::AnthropicManual,
        );

        let decision = decide(&footprint, Some(&source), &target, &cfg);
        assert_eq!(decision, StripDecision::StripAll);

        let mut outgoing = request(messages);
        let report = apply(&mut outgoing, decision);
        assert_eq!(report.fields_removed, 1);
        assert!(!outgoing.messages[0].extra.contains_key("reasoning"));
        assert_eq!(outgoing.messages.len(), 1);
    }

    #[test]
    fn empty_after_strip_message_dropped_entirely() {
        let messages = vec![
            user(json!("hi")),
            assistant(json!([thinking_block()]), Map::new()),
        ];
        let mut outgoing = request(messages);
        let report = apply(&mut outgoing, StripDecision::StripAll);
        assert_eq!(report.messages_touched, 1);
        assert_eq!(report.thinking_blocks, 1);
        assert_eq!(outgoing.messages.len(), 1);
        assert_eq!(outgoing.messages[0].role, "user");
    }

    #[test]
    fn thinking_plus_text_keeps_text_and_message() {
        let messages = vec![assistant(
            json!([thinking_block(), text_block()]),
            Map::new(),
        )];
        let mut outgoing = request(messages);
        let report = apply(&mut outgoing, StripDecision::StripAll);
        assert_eq!(report.thinking_blocks, 1);
        assert_eq!(outgoing.messages.len(), 1);
        assert_eq!(outgoing.messages[0].content, json!([text_block()]));
    }

    #[test]
    fn thinking_plus_tool_calls_keeps_tool_calls() {
        let tool_calls = json!([{
            "id": "call_1",
            "type": "function",
            "function": {"name": "lookup", "arguments": "{}"}
        }]);
        let messages = vec![assistant(
            json!([thinking_block()]),
            [("tool_calls".to_string(), tool_calls.clone())]
                .into_iter()
                .collect(),
        )];
        let mut outgoing = request(messages);
        let report = apply(&mut outgoing, StripDecision::StripAll);
        assert_eq!(report.thinking_blocks, 1);
        assert_eq!(outgoing.messages.len(), 1);
        assert_eq!(outgoing.messages[0].extra.get("tool_calls"), Some(&tool_calls));
    }

    #[test]
    fn thinking_plus_tool_use_block_keeps_message() {
        let tool_use = json!({"type": "tool_use", "id": "tu_1", "name": "lookup", "input": {}});
        let messages = vec![assistant(json!([thinking_block(), tool_use]), Map::new())];
        let mut outgoing = request(messages);
        let report = apply(&mut outgoing, StripDecision::StripAll);
        assert_eq!(report.thinking_blocks, 1);
        assert_eq!(outgoing.messages.len(), 1);
        assert_eq!(outgoing.messages[0].content, json!([tool_use]));
    }

    #[test]
    fn attribution_unknown_cross_family_is_strip_attribution_unknown() {
        let messages = anthropic_conversation();
        let footprint = detect(&messages);
        let cfg = ReasoningCompatConfig::default();
        let target = model_ref("prov-b", "deepseek-chat", ReasoningFamily::DeepSeek);

        let decision = decide(&footprint, None, &target, &cfg);
        assert_eq!(decision, StripDecision::StripAttributionUnknown);
        assert!(decision.strips());
    }

    #[test]
    fn attribution_unknown_same_family_preserves() {
        let messages = anthropic_conversation();
        let footprint = detect(&messages);
        assert_eq!(footprint.source_family, ReasoningFamily::AnthropicManual);
        let cfg = ReasoningCompatConfig::default();
        let target = model_ref(
            "prov-b",
            "claude-4-5-sonnet",
            ReasoningFamily::AnthropicManual,
        );

        let decision = decide(&footprint, None, &target, &cfg);
        assert_eq!(decision, StripDecision::Preserve);
    }

    #[test]
    fn attribution_unknown_unclassified_family_strips() {
        // Mixed non-Anthropic field carriers stay unclassified (family None)
        // — must never preserve on an unclassified match.
        let messages = vec![
            assistant(json!("a"), [("reasoning_content".to_string(), json!("r"))].into_iter().collect()),
            assistant(json!("b"), [("reasoning".to_string(), json!("r"))].into_iter().collect()),
        ];
        let footprint = detect(&messages);
        assert_eq!(footprint.source_family, ReasoningFamily::None);
        let cfg = ReasoningCompatConfig::default();
        let target = model_ref("prov-b", "mystery-model", ReasoningFamily::None);

        let decision = decide(&footprint, None, &target, &cfg);
        assert_eq!(decision, StripDecision::StripAttributionUnknown);
    }

    #[test]
    fn empty_footprint_preserves() {
        let messages = vec![user(json!("hi")), assistant(json!("plain"), Map::new())];
        let footprint = detect(&messages);
        assert!(footprint.is_empty());
        let cfg = ReasoningCompatConfig::default();
        let target = model_ref("prov-b", "other-model", ReasoningFamily::DeepSeek);

        let decision = decide(&footprint, None, &target, &cfg);
        assert_eq!(decision, StripDecision::Preserve);
    }

    #[test]
    fn strip_on_model_change_false_preserves_even_cross_family() {
        let messages = anthropic_conversation();
        let footprint = detect(&messages);
        let cfg = ReasoningCompatConfig {
            strip_on_model_change: false,
            ..ReasoningCompatConfig::default()
        };
        let source = model_ref(
            "prov-a",
            "claude-4-5-sonnet",
            ReasoningFamily::AnthropicManual,
        );
        let target = model_ref("prov-b", "deepseek-chat", ReasoningFamily::DeepSeek);

        let decision = decide(&footprint, Some(&source), &target, &cfg);
        assert_eq!(decision, StripDecision::Preserve);
    }

    #[test]
    fn same_model_different_provider_strips() {
        let messages = anthropic_conversation();
        let footprint = detect(&messages);
        let cfg = ReasoningCompatConfig::default();
        let source = model_ref(
            "prov-a",
            "claude-4-5-sonnet",
            ReasoningFamily::AnthropicManual,
        );
        let target = model_ref(
            "prov-b",
            "claude-4-5-sonnet",
            ReasoningFamily::AnthropicManual,
        );

        let decision = decide(&footprint, Some(&source), &target, &cfg);
        assert_eq!(decision, StripDecision::StripAll);
    }

    #[test]
    fn same_model_different_family_strips() {
        let messages = anthropic_conversation();
        let footprint = detect(&messages);
        let cfg = ReasoningCompatConfig::default();
        let source = model_ref(
            "prov-a",
            "claude-4-5-sonnet",
            ReasoningFamily::AnthropicManual,
        );
        let target = model_ref(
            "prov-a",
            "claude-4-5-sonnet",
            ReasoningFamily::AnthropicAdaptive,
        );

        let decision = decide(&footprint, Some(&source), &target, &cfg);
        assert_eq!(decision, StripDecision::StripAll);
    }

    #[test]
    fn carrier_metadata_stripped_from_remaining_sibling_blocks() {
        let messages = vec![assistant(
            json!([
                thinking_block(),
                {"type": "text", "text": "answer", "signature": "stray", "data": "stray"}
            ]),
            Map::new(),
        )];
        let mut outgoing = request(messages);
        let report = apply(&mut outgoing, StripDecision::StripAll);
        assert_eq!(report.thinking_blocks, 1);
        assert_eq!(report.fields_removed, 2);
        assert_eq!(outgoing.messages[0].content, json!([text_block()]));
    }

    #[test]
    fn responses_reasoning_block_removed_and_counted() {
        let messages = vec![assistant(
            json!([
                {"type": "reasoning", "text": "step by step"},
                text_block()
            ]),
            Map::new(),
        )];
        let mut outgoing = request(messages);
        let report = apply(&mut outgoing, StripDecision::StripAll);
        assert_eq!(report.fields_removed, 1);
        assert_eq!(outgoing.messages[0].content, json!([text_block()]));
    }

    #[test]
    fn strip_attribution_unknown_removes_same_as_strip_all() {
        let messages = vec![assistant(
            json!([thinking_block(), redacted_block(), text_block()]),
            [("reasoning_content".to_string(), json!("r"))]
                .into_iter()
                .collect(),
        )];
        let mut outgoing = request(messages);
        let report = apply(&mut outgoing, StripDecision::StripAttributionUnknown);
        assert_eq!(report.thinking_blocks, 1);
        assert_eq!(report.redacted_thinking_blocks, 1);
        assert_eq!(report.fields_removed, 1);
        assert_eq!(outgoing.messages[0].content, json!([text_block()]));
        assert!(!outgoing.messages[0].extra.contains_key("reasoning_content"));
    }

    #[test]
    fn counts_aggregate_across_multiple_messages() {
        let messages = vec![
            user(json!("hi")),
            assistant(json!([thinking_block(), text_block()]), Map::new()),
            assistant(json!([redacted_block(), text_block()]), Map::new()),
            assistant(json!("plain"), [("reasoning".to_string(), json!("r"))]
                .into_iter()
                .collect()),
        ];
        let mut outgoing = request(messages);
        let report = apply(&mut outgoing, StripDecision::StripAll);
        assert_eq!(report.messages_touched, 3);
        assert_eq!(report.thinking_blocks, 1);
        assert_eq!(report.redacted_thinking_blocks, 1);
        assert_eq!(report.fields_removed, 1);
        assert_eq!(outgoing.messages.len(), 4);
    }

    #[test]
    fn untouched_messages_are_never_dropped() {
        let messages = vec![
            user(json!("")),
            assistant(json!([]), Map::new()),
        ];
        let mut outgoing = request(messages);
        let report = apply(&mut outgoing, StripDecision::StripAll);
        assert_eq!(report, StripReport::default());
        assert_eq!(outgoing.messages.len(), 2);
    }

    #[test]
    fn decision_as_str_is_stable() {
        assert_eq!(StripDecision::Preserve.as_str(), "preserve");
        assert_eq!(StripDecision::StripAll.as_str(), "strip_all");
        assert_eq!(
            StripDecision::StripAttributionUnknown.as_str(),
            "strip_attribution_unknown"
        );
    }
}
