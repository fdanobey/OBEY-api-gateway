//! Reasoning-carrier detection (design Component 1).
//!
//! Single-pass scan over request messages that identifies every form of
//! reasoning state clients replay in prior assistant turns: Anthropic
//! `thinking` / `redacted_thinking` content blocks, DeepSeek-style
//! `reasoning_content` message fields, OpenRouter-style `reasoning` message
//! fields, and Responses-style `reasoning` content items. The source
//! reasoning family is inferred from the observed carrier shapes.
//!
//! The footprint never retains payload text, signatures, or redacted data —
//! only flags, counts, message indexes, and an approximate token estimate —
//! so downstream strip/preserve policy (Component 2) can act without the
//! gateway ever copying foreign reasoning state into new allocations.
//!
//! The empty path allocates nothing beyond the footprint struct itself
//! (all `Vec` fields start with zero capacity and are only grown when a
//! carrier is actually found).

use crate::models::openai::Message;
use crate::reasoning_compat::config::ReasoningFamily;

/// Chars-per-token approximation used for `approx_reasoning_tokens`.
const CHARS_PER_TOKEN: u64 = 4;

/// Reasoning-state footprint of a request's message list.
///
/// Produced by [`detect`]; consumed by the strip/preserve policy to decide
/// whether prior-turn reasoning state survives a model transition.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ReasoningFootprint {
    /// At least one `thinking` content block was found (Anthropic family).
    pub has_thinking_blocks: bool,
    /// At least one `redacted_thinking` content block was found (Anthropic
    /// family, encrypted carrier). Tracked separately from
    /// `has_thinking_blocks` because redacted blocks must never be treated
    /// as plain thinking state.
    pub has_redacted_thinking: bool,
    /// Indexes of assistant messages carrying an OpenRouter-style
    /// `reasoning` extra field.
    pub reasoning_field_msgs: Vec<usize>,
    /// Indexes of assistant messages carrying a DeepSeek-style
    /// `reasoning_content` extra field.
    pub reasoning_content_msgs: Vec<usize>,
    /// Indexes of assistant messages carrying Responses-style reasoning
    /// items (`type: "reasoning"` content blocks, including gateway-
    /// synthesized summary items replayed by clients).
    pub responses_items: Vec<usize>,
    /// Source reasoning family inferred from the carrier shapes found.
    pub source_family: ReasoningFamily,
    /// Total number of reasoning content blocks found across messages.
    pub block_counts: usize,
    /// Rough token estimate (`chars / 4`) summed over every reasoning
    /// carrier whose text length was available; `None` when no carrier
    /// exposed a text length (e.g. redacted-only).
    pub approx_reasoning_tokens: Option<u32>,
}

impl Default for ReasoningFootprint {
    fn default() -> Self {
        Self {
            has_thinking_blocks: false,
            has_redacted_thinking: false,
            reasoning_field_msgs: Vec::new(),
            reasoning_content_msgs: Vec::new(),
            responses_items: Vec::new(),
            source_family: ReasoningFamily::None,
            block_counts: 0,
            approx_reasoning_tokens: None,
        }
    }
}

impl ReasoningFootprint {
    /// True when no reasoning carriers of any kind were found.
    pub fn is_empty(&self) -> bool {
        !self.has_thinking_blocks
            && !self.has_redacted_thinking
            && self.reasoning_field_msgs.is_empty()
            && self.reasoning_content_msgs.is_empty()
            && self.responses_items.is_empty()
    }
}

/// Detect all reasoning-state carriers in a message list in a single pass.
///
/// Only assistant messages are scanned: reasoning state is produced by the
/// assistant and replayed by clients in prior assistant turns. Detection
/// covers:
///
/// - content-array blocks with `type` equal to `thinking`,
///   `redacted_thinking`, or `reasoning`
/// - top-level assistant extra fields `reasoning_content` (DeepSeek) and
///   `reasoning` (OpenRouter)
///
/// Family inference: Anthropic thinking/redacted blocks dominate (manual
/// mode is preferred when `thinking` blocks carry a `signature` field;
/// unsigned thinking blocks indicate the adaptive era; redacted-only
/// footprints default to manual). Otherwise Responses-style `reasoning`
/// blocks or a `reasoning` field imply OpenRouter, `reasoning_content`
/// alone implies DeepSeek, and mixed non-Anthropic carriers stay
/// unclassified (`ReasoningFamily::None`).
pub fn detect(messages: &[Message]) -> ReasoningFootprint {
    let mut footprint = ReasoningFootprint::default();
    let mut signed_thinking = false;
    let mut reasoning_chars: u64 = 0;
    let mut has_text_length = false;

    let mut accumulate = |text: Option<&str>| {
        if let Some(text) = text {
            reasoning_chars += text.chars().count() as u64;
            has_text_length = true;
        }
    };

    for (index, message) in messages.iter().enumerate() {
        if !message.role.eq_ignore_ascii_case("assistant") {
            continue;
        }

        let mut has_reasoning_block = false;

        if let serde_json::Value::Array(blocks) = &message.content {
            for block in blocks {
                let block_type = match block.get("type").and_then(serde_json::Value::as_str) {
                    Some(block_type) => block_type,
                    None => continue,
                };
                match block_type {
                    "thinking" => {
                        footprint.has_thinking_blocks = true;
                        footprint.block_counts += 1;
                        signed_thinking |= block.get("signature").is_some();
                        accumulate(block.get("thinking").and_then(serde_json::Value::as_str));
                    }
                    "redacted_thinking" => {
                        footprint.has_redacted_thinking = true;
                        footprint.block_counts += 1;
                    }
                    "reasoning" => {
                        has_reasoning_block = true;
                        footprint.block_counts += 1;
                        for key in ["text", "reasoning"] {
                            let text = block.get(key).and_then(serde_json::Value::as_str);
                            if text.is_some() {
                                accumulate(text);
                                break;
                            }
                        }
                    }
                    _ => {}
                }
            }
        }

        if has_reasoning_block {
            footprint.responses_items.push(index);
        }

        if let Some(value) = message.extra.get("reasoning_content") {
            footprint.reasoning_content_msgs.push(index);
            accumulate(value.as_str());
        }
        if let Some(value) = message.extra.get("reasoning") {
            footprint.reasoning_field_msgs.push(index);
            accumulate(value.as_str());
        }
    }

    footprint.source_family = infer_source_family(&footprint, signed_thinking);

    footprint.approx_reasoning_tokens = if has_text_length {
        Some((reasoning_chars / CHARS_PER_TOKEN).min(u32::MAX as u64) as u32)
    } else {
        None
    };

    footprint
}

/// Infer the reasoning family from the observed carrier shapes.
fn infer_source_family(footprint: &ReasoningFootprint, signed_thinking: bool) -> ReasoningFamily {
    if footprint.has_thinking_blocks || footprint.has_redacted_thinking {
        if footprint.has_thinking_blocks && !signed_thinking {
            ReasoningFamily::AnthropicAdaptive
        } else {
            ReasoningFamily::AnthropicManual
        }
    } else if !footprint.responses_items.is_empty() {
        ReasoningFamily::OpenRouter
    } else if !footprint.reasoning_content_msgs.is_empty()
        && !footprint.reasoning_field_msgs.is_empty()
    {
        ReasoningFamily::None
    } else if !footprint.reasoning_content_msgs.is_empty() {
        ReasoningFamily::DeepSeek
    } else if !footprint.reasoning_field_msgs.is_empty() {
        ReasoningFamily::OpenRouter
    } else {
        ReasoningFamily::None
    }
}

/// Classify a model id into its reasoning family by name heuristics
/// (case-insensitive).
///
/// - `openrouter/…` namespace → [`ReasoningFamily::OpenRouter`]
/// - Claude 4.7+ (and any 5.x or newer) → [`ReasoningFamily::AnthropicAdaptive`]
/// - Other Claude models → [`ReasoningFamily::AnthropicManual`]
/// - OpenAI o-series (`o1`, `o3`, `o4` boundaries mirror
///   `crate::config::is_thinking_model`) and `gpt-5` →
///   [`ReasoningFamily::OpenAIReasoning`]
/// - `deepseek` prefix → [`ReasoningFamily::DeepSeek`]
/// - `gemini` → [`ReasoningFamily::Gemini`]
/// - `grok` → [`ReasoningFamily::XAI`]
/// - Everything else → [`ReasoningFamily::None`]
pub fn classify_family(model_id: &str) -> ReasoningFamily {
    let m = model_id.to_lowercase();

    if m.starts_with("openrouter/") {
        return ReasoningFamily::OpenRouter;
    }

    if m.contains("claude") {
        return classify_claude_family(&m);
    }

    if (m.starts_with("o1") || m.starts_with("o3") || m.starts_with("o4"))
        && m.chars().nth(2).map_or(true, |c| c == '-' || c == ' ')
    {
        return ReasoningFamily::OpenAIReasoning;
    }
    if m.contains("gpt-5") {
        return ReasoningFamily::OpenAIReasoning;
    }

    if m.starts_with("deepseek") {
        return ReasoningFamily::DeepSeek;
    }

    if m.contains("gemini") {
        return ReasoningFamily::Gemini;
    }

    if m.contains("grok") {
        return ReasoningFamily::XAI;
    }

    ReasoningFamily::None
}

/// Classify a Claude model id as manual-thinking vs adaptive-thinking.
///
/// Extracts the first adjacent major.minor version pair found after the
/// `claude` token (skipping tier names like `opus`/`sonnet`/`haiku`):
/// major >= 5, or major == 4 with minor >= 7, is adaptive; everything else
/// is manual.
fn classify_claude_family(lowercase_id: &str) -> ReasoningFamily {
    let tokens: Vec<&str> = lowercase_id
        .split(|c: char| !c.is_ascii_alphanumeric())
        .filter(|token| !token.is_empty())
        .collect();

    let claude_position = match tokens.iter().position(|token| *token == "claude") {
        Some(position) => position,
        None => return ReasoningFamily::AnthropicManual,
    };

    let mut major: Option<u32> = None;
    let mut minor: Option<u32> = None;
    let mut in_numbers = false;
    for token in &tokens[claude_position + 1..] {
        match token.parse::<u32>() {
            Ok(number) => {
                if !in_numbers {
                    in_numbers = true;
                    major = Some(number);
                } else if minor.is_none() {
                    minor = Some(number);
                    break;
                }
            }
            Err(_) => {
                if in_numbers {
                    break;
                }
            }
        }
    }

    match (major, minor) {
        (Some(major), _) if major >= 5 => ReasoningFamily::AnthropicAdaptive,
        (Some(4), Some(minor)) if minor >= 7 => ReasoningFamily::AnthropicAdaptive,
        _ => ReasoningFamily::AnthropicManual,
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;
    use serde_json::Map;

    fn assistant(content: serde_json::Value, extra: &[(&str, serde_json::Value)]) -> Message {
        let mut map = Map::new();
        for (key, value) in extra {
            map.insert((*key).to_string(), value.clone());
        }
        Message {
            role: "assistant".to_string(),
            content,
            extra: map,
        }
    }

    fn user(content: serde_json::Value) -> Message {
        Message {
            role: "user".to_string(),
            content,
            extra: Map::new(),
        }
    }

    #[test]
    fn thinking_block_in_content_array_is_detected() {
        let messages = vec![assistant(
            json!([
                {"type": "thinking", "thinking": "a".repeat(40), "signature": "sig"},
                {"type": "text", "text": "answer"}
            ]),
            &[],
        )];

        let footprint = detect(&messages);

        assert!(footprint.has_thinking_blocks);
        assert!(!footprint.has_redacted_thinking);
        assert_eq!(footprint.block_counts, 1);
        assert!(footprint.responses_items.is_empty());
        assert!(footprint.reasoning_field_msgs.is_empty());
        assert!(footprint.reasoning_content_msgs.is_empty());
        assert_eq!(footprint.source_family, ReasoningFamily::AnthropicManual);
        assert_eq!(footprint.approx_reasoning_tokens, Some(10));
        assert!(!footprint.is_empty());
    }

    #[test]
    fn unsigned_thinking_block_infers_adaptive_family() {
        let messages = vec![assistant(
            json!([{"type": "thinking", "thinking": "hmm"}]),
            &[],
        )];

        let footprint = detect(&messages);

        assert!(footprint.has_thinking_blocks);
        assert_eq!(footprint.source_family, ReasoningFamily::AnthropicAdaptive);
        assert_eq!(footprint.approx_reasoning_tokens, Some(0));
    }

    #[test]
    fn redacted_thinking_block_sets_dedicated_flag() {
        let messages = vec![assistant(
            json!([{"type": "redacted_thinking", "data": "opaque"}]),
            &[],
        )];

        let footprint = detect(&messages);

        assert!(footprint.has_redacted_thinking);
        assert!(!footprint.has_thinking_blocks);
        assert_eq!(footprint.block_counts, 1);
        assert_eq!(footprint.source_family, ReasoningFamily::AnthropicManual);
        assert_eq!(footprint.approx_reasoning_tokens, None);
        assert!(!footprint.is_empty());
    }

    #[test]
    fn reasoning_content_extra_field_is_deepseek_style() {
        let messages = vec![
            user(json!("question")),
            assistant(json!("answer"), &[("reasoning_content", json!("thoughts"))]),
        ];

        let footprint = detect(&messages);

        assert_eq!(footprint.reasoning_content_msgs, vec![1]);
        assert!(footprint.reasoning_field_msgs.is_empty());
        assert!(!footprint.has_thinking_blocks);
        assert_eq!(footprint.source_family, ReasoningFamily::DeepSeek);
        assert_eq!(footprint.approx_reasoning_tokens, Some(2));
    }

    #[test]
    fn reasoning_extra_field_is_openrouter_style() {
        let messages = vec![assistant(
            json!("answer"),
            &[("reasoning", json!("because"))],
        )];

        let footprint = detect(&messages);

        assert_eq!(footprint.reasoning_field_msgs, vec![0]);
        assert!(footprint.reasoning_content_msgs.is_empty());
        assert_eq!(footprint.source_family, ReasoningFamily::OpenRouter);
        assert_eq!(footprint.approx_reasoning_tokens, Some(1));
    }

    #[test]
    fn non_string_reasoning_field_is_still_a_carrier() {
        let messages = vec![assistant(json!("answer"), &[("reasoning", json!(null))])];

        let footprint = detect(&messages);

        assert_eq!(footprint.reasoning_field_msgs, vec![0]);
        assert_eq!(footprint.source_family, ReasoningFamily::OpenRouter);
        assert_eq!(footprint.approx_reasoning_tokens, None);
    }

    #[test]
    fn reasoning_content_block_is_responses_style() {
        let messages = vec![assistant(
            json!([
                {"type": "reasoning", "text": "step by step"},
                {"type": "text", "text": "answer"}
            ]),
            &[],
        )];

        let footprint = detect(&messages);

        assert_eq!(footprint.responses_items, vec![0]);
        assert_eq!(footprint.block_counts, 1);
        assert!(!footprint.has_thinking_blocks);
        assert_eq!(footprint.source_family, ReasoningFamily::OpenRouter);
        assert_eq!(footprint.approx_reasoning_tokens, Some(3));
    }

    #[test]
    fn mixed_carriers_record_every_index_and_prefer_anthropic() {
        let messages = vec![
            user(json!("question")),
            assistant(
                json!([{"type": "thinking", "thinking": "deep", "signature": "s"}]),
                &[],
            ),
            assistant(json!("partial"), &[("reasoning_content", json!("r"))]),
            assistant(json!("final"), &[("reasoning", json!("r"))]),
        ];

        let footprint = detect(&messages);

        assert!(footprint.has_thinking_blocks);
        assert_eq!(footprint.block_counts, 1);
        assert_eq!(footprint.reasoning_content_msgs, vec![2]);
        assert_eq!(footprint.reasoning_field_msgs, vec![3]);
        assert_eq!(footprint.source_family, ReasoningFamily::AnthropicManual);
    }

    #[test]
    fn mixed_non_anthropic_field_carriers_stay_unclassified() {
        let messages = vec![
            assistant(json!("a"), &[("reasoning_content", json!("r"))]),
            assistant(json!("b"), &[("reasoning", json!("r"))]),
        ];

        let footprint = detect(&messages);

        assert_eq!(footprint.reasoning_content_msgs, vec![0]);
        assert_eq!(footprint.reasoning_field_msgs, vec![1]);
        assert_eq!(footprint.source_family, ReasoningFamily::None);
        assert!(!footprint.is_empty());
    }

    #[test]
    fn multiple_blocks_and_carriers_accumulate_counts_and_tokens() {
        let messages = vec![assistant(
            json!([
                {"type": "thinking", "thinking": "x".repeat(40), "signature": "s"},
                {"type": "redacted_thinking", "data": "opaque"},
                {"type": "reasoning", "text": "y".repeat(20)}
            ]),
            &[("reasoning_content", json!("z".repeat(20)))],
        )];

        let footprint = detect(&messages);

        assert!(footprint.has_thinking_blocks);
        assert!(footprint.has_redacted_thinking);
        assert_eq!(footprint.block_counts, 3);
        assert_eq!(footprint.responses_items, vec![0]);
        assert_eq!(footprint.reasoning_content_msgs, vec![0]);
        assert_eq!(footprint.source_family, ReasoningFamily::AnthropicManual);
        assert_eq!(footprint.approx_reasoning_tokens, Some(20));
    }

    #[test]
    fn non_assistant_messages_are_ignored() {
        let messages = vec![
            user(json!([{"type": "thinking", "thinking": "not mine"}])),
            Message {
                role: "tool".to_string(),
                content: json!("result"),
                extra: {
                    let mut map = Map::new();
                    map.insert("reasoning_content".to_string(), json!("ignored"));
                    map
                },
            },
        ];

        let footprint = detect(&messages);

        assert!(footprint.is_empty());
        assert_eq!(footprint.block_counts, 0);
        assert_eq!(footprint.source_family, ReasoningFamily::None);
    }

    #[test]
    fn empty_message_list_yields_empty_footprint_without_allocation() {
        let footprint = detect(&[]);

        assert!(footprint.is_empty());
        assert_eq!(footprint.block_counts, 0);
        assert_eq!(footprint.source_family, ReasoningFamily::None);
        assert_eq!(footprint.approx_reasoning_tokens, None);
        assert_eq!(footprint.reasoning_field_msgs.capacity(), 0);
        assert_eq!(footprint.reasoning_content_msgs.capacity(), 0);
        assert_eq!(footprint.responses_items.capacity(), 0);
    }

    #[test]
    fn messages_without_carriers_yield_empty_footprint_without_allocation() {
        let messages = vec![
            user(json!("hello")),
            assistant(json!("hi"), &[]),
            assistant(json!([{"type": "text", "text": "plain"}]), &[]),
        ];

        let footprint = detect(&messages);

        assert!(footprint.is_empty());
        assert_eq!(footprint.block_counts, 0);
        assert_eq!(footprint.source_family, ReasoningFamily::None);
        assert_eq!(footprint.approx_reasoning_tokens, None);
        assert_eq!(footprint.reasoning_field_msgs.capacity(), 0);
        assert_eq!(footprint.reasoning_content_msgs.capacity(), 0);
        assert_eq!(footprint.responses_items.capacity(), 0);
    }

    #[test]
    fn classify_family_matrix() {
        assert_eq!(
            classify_family("claude-4-5-sonnet"),
            ReasoningFamily::AnthropicManual
        );
        assert_eq!(
            classify_family("claude-4-7"),
            ReasoningFamily::AnthropicAdaptive
        );
        assert_eq!(
            classify_family("claude-opus-4-1"),
            ReasoningFamily::AnthropicManual
        );
        assert_eq!(
            classify_family("o3-mini"),
            ReasoningFamily::OpenAIReasoning
        );
        assert_eq!(classify_family("deepseek-chat"), ReasoningFamily::DeepSeek);
        assert_eq!(classify_family("gemini-2.0"), ReasoningFamily::Gemini);
        assert_eq!(classify_family("grok-3"), ReasoningFamily::XAI);
        assert_eq!(classify_family("gpt-4o"), ReasoningFamily::None);
    }

    #[test]
    fn classify_family_extended_matrix() {
        assert_eq!(classify_family("claude-3-5-sonnet-20241022-v2:0"), ReasoningFamily::AnthropicManual);
        assert_eq!(classify_family("claude-sonnet-4-5"), ReasoningFamily::AnthropicManual);
        assert_eq!(classify_family("claude-sonnet-4-7"), ReasoningFamily::AnthropicAdaptive);
        assert_eq!(classify_family("claude-opus-4-7"), ReasoningFamily::AnthropicAdaptive);
        assert_eq!(classify_family("claude-4.7-sonnet"), ReasoningFamily::AnthropicAdaptive);
        assert_eq!(classify_family("claude-5-sonnet"), ReasoningFamily::AnthropicAdaptive);
        assert_eq!(classify_family("claude-opus-5"), ReasoningFamily::AnthropicAdaptive);
        assert_eq!(classify_family("anthropic.claude-4-7-sonnet-v1:0"), ReasoningFamily::AnthropicAdaptive);
        assert_eq!(classify_family("openrouter/anthropic/claude-3.5-sonnet"), ReasoningFamily::OpenRouter);
        assert_eq!(classify_family("o1-preview"), ReasoningFamily::OpenAIReasoning);
        assert_eq!(classify_family("o4-mini"), ReasoningFamily::OpenAIReasoning);
        assert_eq!(classify_family("gpt-5-mini"), ReasoningFamily::OpenAIReasoning);
        assert_eq!(classify_family("deepseek-r1"), ReasoningFamily::DeepSeek);
        assert_eq!(classify_family("gemini-2.0-flash-thinking"), ReasoningFamily::Gemini);
        assert_eq!(classify_family("grok-4"), ReasoningFamily::XAI);
        assert_eq!(classify_family("gpt-4-turbo"), ReasoningFamily::None);
        assert_eq!(classify_family("llama-3-1-70b"), ReasoningFamily::None);
        assert_eq!(classify_family(""), ReasoningFamily::None);
    }

    #[test]
    fn classify_family_is_case_insensitive() {
        assert_eq!(
            classify_family("CLAUDE-Opus-4-7"),
            ReasoningFamily::AnthropicAdaptive
        );
        assert_eq!(classify_family("O3-Mini"), ReasoningFamily::OpenAIReasoning);
        assert_eq!(classify_family("Grok-3"), ReasoningFamily::XAI);
        assert_eq!(classify_family("DeepSeek-Chat"), ReasoningFamily::DeepSeek);
    }
}
