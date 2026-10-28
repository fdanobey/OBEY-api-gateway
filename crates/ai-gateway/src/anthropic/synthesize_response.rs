//! Buffered Chat Completions → Anthropic Messages synthesizer.
//!
//! Converts a fully-buffered [`crate::models::openai::OpenAIResponse`] into an
//! [`AnthropicResponse`] for the `/v1/messages` front door (non-streaming path
//! only; SSE synthesis is handled separately).
//!
//! Content block order is fixed: `thinking` (if reasoning content is present)
//! → `text` (if non-empty) → one `tool_use` block per chat tool call.

use crate::anthropic::models::{
    new_message_id, AnthropicResponse, ContentBlock, StopReason, Usage,
};
use crate::models::openai::{Message as ChatMessage, OpenAIResponse, Usage as ChatUsage};
use crate::router::cache_cost::extract_cache_usage;

/// Request-side fields needed to synthesize the response.
pub struct AnthropicSynthesisContext<'a> {
    /// Model name from the original Anthropic request, used as a fallback when
    /// the provider response omits `model`.
    pub request_model: &'a str,
    /// `stop_sequences` from the original Anthropic request, used to detect a
    /// stop-sequence termination when the chat layer reports `finish_reason:
    /// "stop"`.
    pub stop_sequences: &'a [String],
}

/// Synthesize an Anthropic Messages response from a buffered chat completion.
///
/// Uses the first choice; an empty `choices` array yields an empty-content
/// message with no `stop_reason`.
pub fn synthesize_anthropic_response(
    chat: &OpenAIResponse,
    ctx: &AnthropicSynthesisContext<'_>,
) -> AnthropicResponse {
    let choice = chat.choices.first();
    let finish_reason = choice.and_then(|c| c.finish_reason.as_deref());

    let mut content: Vec<ContentBlock> = Vec::new();
    let (stop_reason, stop_sequence) = match choice {
        None => (None, None),
        Some(choice) => {
            let message = &choice.message;

            if let Some(text) = reasoning_text(message) {
                content.push(ContentBlock::Thinking {
                    thinking: text,
                    signature: String::new(),
                    extra: Default::default(),
                });
            }

            let text = message.content_as_text();
            if !text.is_empty() {
                content.push(ContentBlock::Text {
                    text,
                    extra: Default::default(),
                });
            }

            for tool_use in tool_use_blocks(message) {
                content.push(tool_use);
            }

            map_stop(finish_reason, &choice.message, ctx.stop_sequences)
        }
    };

    AnthropicResponse {
        kind: "message".to_string(),
        id: new_message_id(),
        role: "assistant".to_string(),
        content,
        model: if chat.model.is_empty() {
            ctx.request_model.to_string()
        } else {
            chat.model.clone()
        },
        stop_reason,
        stop_sequence,
        usage: map_usage(&chat.usage),
        extra: Default::default(),
    }
}

/// Maps a chat `finish_reason` to an Anthropic `stop_reason` (plus the matched
/// `stop_sequence` when one was hit).
///
/// - `stop` → `end_turn`, or `stop_sequence` when the choice text ends with one
///   of the request's stop sequences
/// - `length` → `max_tokens`
/// - `tool_calls` → `tool_use`
/// - `content_filter` → `refusal`
/// - missing/unknown → `end_turn`
fn map_stop(
    finish_reason: Option<&str>,
    message: &ChatMessage,
    request_stop_sequences: &[String],
) -> (Option<StopReason>, Option<String>) {
    match finish_reason {
        Some("length") => (Some(StopReason::MaxTokens), None),
        Some("tool_calls") => (Some(StopReason::ToolUse), None),
        Some("content_filter") => (Some(StopReason::Refusal), None),
        Some("stop") => {
            let text = message.content_as_text();
            match request_stop_sequences
                .iter()
                .find(|seq| !seq.is_empty() && text.ends_with(seq.as_str()))
            {
                Some(seq) => (Some(StopReason::StopSequence), Some(seq.clone())),
                None => (Some(StopReason::EndTurn), None),
            }
        }
        _ => (Some(StopReason::EndTurn), None),
    }
}

/// Reasoning text from a chat message, if any.
///
/// DeepSeek-style providers use `reasoning_content`; others use `reasoning`.
/// Mirrors the convention of [`crate::responses::synthesize_response`].
fn reasoning_text(message: &ChatMessage) -> Option<String> {
    for key in ["reasoning_content", "reasoning"] {
        if let Some(text) = message.extra.get(key).and_then(|v| v.as_str()) {
            if !text.is_empty() {
                return Some(text.to_string());
            }
        }
    }
    None
}

/// Converts chat `tool_calls` into Anthropic `tool_use` content blocks.
///
/// The string `arguments` are parsed into a JSON object; malformed or empty
/// arguments become an empty object, matching Anthropic's `input` shape.
fn tool_use_blocks(message: &ChatMessage) -> Vec<ContentBlock> {
    let tool_calls = message
        .extra
        .get("tool_calls")
        .and_then(|v| v.as_array())
        .map(Vec::as_slice)
        .unwrap_or_default();

    tool_calls
        .iter()
        .map(|tc| {
            let function = tc.get("function").cloned().unwrap_or_default();
            ContentBlock::ToolUse {
                id: tc
                    .get("id")
                    .and_then(|v| v.as_str())
                    .unwrap_or_default()
                    .to_string(),
                name: function
                    .get("name")
                    .and_then(|v| v.as_str())
                    .unwrap_or_default()
                    .to_string(),
                input: parse_tool_arguments(
                    function.get("arguments").and_then(|v| v.as_str()),
                ),
                extra: Default::default(),
            }
        })
        .collect()
}

/// Parses tool-call `arguments` (a JSON-encoded string) into a JSON value.
///
/// Empty or malformed arguments become an empty JSON object so `input` is
/// always a usable object for Anthropic clients.
fn parse_tool_arguments(arguments: Option<&str>) -> serde_json::Value {
    arguments
        .and_then(|s| serde_json::from_str(s).ok())
        .filter(serde_json::Value::is_object)
        .unwrap_or_else(|| serde_json::Value::Object(Default::default()))
}

/// Maps chat usage onto Anthropic usage, reusing [`extract_cache_usage`] for
/// the cache-token breakdown (Anthropic and OpenAI provider shapes alike).
fn map_usage(usage: &ChatUsage) -> Usage {
    let cache = extract_cache_usage(usage);
    Usage {
        input_tokens: u64::from(usage.prompt_tokens),
        output_tokens: u64::from(usage.completion_tokens),
        cache_read_input_tokens: (cache.cache_read_input_tokens > 0)
            .then_some(cache.cache_read_input_tokens),
        cache_creation_input_tokens: (cache.cache_creation_input_tokens > 0)
            .then_some(cache.cache_creation_input_tokens),
        extra: Default::default(),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    fn chat_from(value: serde_json::Value) -> OpenAIResponse {
        serde_json::from_value(value).expect("test chat response must parse")
    }

    fn ctx(stop_sequences: &[String]) -> AnthropicSynthesisContext<'_> {
        AnthropicSynthesisContext {
            request_model: "claude-test",
            stop_sequences,
        }
    }

    fn text_response(finish_reason: Option<&str>, content: &str) -> OpenAIResponse {
        chat_from(json!({
            "choices": [{
                "index": 0,
                "message": {"role": "assistant", "content": content},
                "finish_reason": finish_reason,
            }],
            "usage": {"prompt_tokens": 10, "completion_tokens": 5, "total_tokens": 15},
        }))
    }

    #[test]
    fn finish_reason_mapping_matrix() {
        let cases = [
            ("stop", StopReason::EndTurn),
            ("length", StopReason::MaxTokens),
            ("tool_calls", StopReason::ToolUse),
            ("content_filter", StopReason::Refusal),
        ];
        for (reason, expected) in cases {
            let chat = text_response(Some(reason), "x");
            let resp = synthesize_anthropic_response(&chat, &ctx(&[]));
            assert_eq!(resp.stop_reason, Some(expected), "finish_reason {reason}");
            assert!(resp.stop_sequence.is_none(), "finish_reason {reason}");
        }
    }

    #[test]
    fn unknown_or_missing_finish_reason_defaults_to_end_turn() {
        for reason in [Some("weird_provider_reason"), None] {
            let chat = text_response(reason, "x");
            let resp = synthesize_anthropic_response(&chat, &ctx(&[]));
            assert_eq!(resp.stop_reason, Some(StopReason::EndTurn));
        }
    }

    #[test]
    fn stop_sequence_hit_maps_to_stop_sequence_with_field() {
        let stops = vec!["END".to_string(), "STOP".to_string()];
        let chat = text_response(Some("stop"), "partial text END");
        let resp = synthesize_anthropic_response(&chat, &ctx(&stops));
        assert_eq!(resp.stop_reason, Some(StopReason::StopSequence));
        assert_eq!(resp.stop_sequence.as_deref(), Some("END"));
    }

    #[test]
    fn stop_sequence_not_hit_keeps_end_turn() {
        let stops = vec!["END".to_string()];
        let chat = text_response(Some("stop"), "no match here");
        let resp = synthesize_anthropic_response(&chat, &ctx(&stops));
        assert_eq!(resp.stop_reason, Some(StopReason::EndTurn));
        assert!(resp.stop_sequence.is_none());
    }

    #[test]
    fn empty_choices_yield_empty_content_and_no_stop_reason() {
        let chat = chat_from(json!({"choices": [], "usage": {}}));
        let resp = synthesize_anthropic_response(&chat, &ctx(&[]));
        assert!(resp.content.is_empty());
        assert_eq!(resp.stop_reason, None);
        assert_eq!(resp.role, "assistant");
        assert_eq!(resp.kind, "message");
    }

    #[test]
    fn single_tool_call_becomes_tool_use_block_with_parsed_input() {
        let chat = chat_from(json!({
            "choices": [{
                "index": 0,
                "message": {
                    "role": "assistant",
                    "content": null,
                    "tool_calls": [
                        {"id": "call_1", "type": "function",
                         "function": {"name": "get_weather", "arguments": "{\"city\":\"Berlin\"}"}}
                    ]
                },
                "finish_reason": "tool_calls",
            }],
        }));
        let resp = synthesize_anthropic_response(&chat, &ctx(&[]));
        assert_eq!(resp.content.len(), 1);
        match &resp.content[0] {
            ContentBlock::ToolUse { id, name, input, .. } => {
                assert_eq!(id, "call_1");
                assert_eq!(name, "get_weather");
                assert_eq!(input, &json!({"city": "Berlin"}));
            }
            other => panic!("expected tool_use block, got {other:?}"),
        }
        assert_eq!(resp.stop_reason, Some(StopReason::ToolUse));
    }

    #[test]
    fn multiple_tool_calls_preserve_order() {
        let chat = chat_from(json!({
            "choices": [{
                "index": 0,
                "message": {
                    "role": "assistant",
                    "content": "Checking.",
                    "tool_calls": [
                        {"id": "a", "function": {"name": "f1", "arguments": "{}"}},
                        {"id": "b", "function": {"name": "f2", "arguments": "{\"x\":1}"}},
                        {"id": "c", "function": {"name": "f3", "arguments": "not json"}}
                    ]
                },
                "finish_reason": "tool_calls",
            }],
        }));
        let resp = synthesize_anthropic_response(&chat, &ctx(&[]));
        assert_eq!(resp.content.len(), 4);
        for (i, expected_id) in ["a", "b", "c"].iter().enumerate() {
            match &resp.content[i + 1] {
                ContentBlock::ToolUse { id, .. } => assert_eq!(id, expected_id),
                other => panic!("expected tool_use block, got {other:?}"),
            }
        }
    }

    #[test]
    fn malformed_tool_arguments_become_empty_object() {
        for args in ["not json", "", "[1,2]"] {
            let chat = chat_from(json!({
                "choices": [{
                    "index": 0,
                    "message": {
                        "role": "assistant",
                        "content": null,
                        "tool_calls": [
                            {"id": "call_x",
                             "function": {"name": "f", "arguments": args}}
                        ]
                    },
                    "finish_reason": "tool_calls",
                }],
            }));
            let resp = synthesize_anthropic_response(&chat, &ctx(&[]));
            match &resp.content[0] {
                ContentBlock::ToolUse { input, .. } => {
                    assert_eq!(input, &json!({}), "arguments {args:?}");
                }
                other => panic!("expected tool_use block, got {other:?}"),
            }
        }
    }

    #[test]
    fn reasoning_content_becomes_leading_thinking_block() {
        let chat = chat_from(json!({
            "choices": [{
                "index": 0,
                "message": {
                    "role": "assistant",
                    "content": "Answer.",
                    "reasoning_content": "let me think"
                },
                "finish_reason": "stop",
            }],
        }));
        let resp = synthesize_anthropic_response(&chat, &ctx(&[]));
        assert_eq!(resp.content.len(), 2);
        match &resp.content[0] {
            ContentBlock::Thinking { thinking, .. } => {
                assert_eq!(thinking, "let me think");
            }
            other => panic!("expected thinking block, got {other:?}"),
        }
        assert!(matches!(&resp.content[1], ContentBlock::Text { text, .. } if text == "Answer."));
    }

    #[test]
    fn array_content_shape_becomes_text_block() {
        let chat = chat_from(json!({
            "choices": [{
                "index": 0,
                "message": {
                    "role": "assistant",
                    "content": [
                        {"type": "text", "text": "part one "},
                        {"type": "text", "text": "part two"}
                    ]
                },
                "finish_reason": "stop",
            }],
        }));
        let resp = synthesize_anthropic_response(&chat, &ctx(&[]));
        match &resp.content[0] {
            ContentBlock::Text { text, .. } => assert_eq!(text, "part one part two"),
            other => panic!("expected text block, got {other:?}"),
        }
    }

    #[test]
    fn usage_maps_prompt_completion_tokens() {
        let chat = text_response(Some("stop"), "x");
        let resp = synthesize_anthropic_response(&chat, &ctx(&[]));
        assert_eq!(resp.usage.input_tokens, 10);
        assert_eq!(resp.usage.output_tokens, 5);
        assert!(resp.usage.cache_read_input_tokens.is_none());
        assert!(resp.usage.cache_creation_input_tokens.is_none());
    }

    #[test]
    fn usage_maps_anthropic_shaped_cache_tokens() {
        let chat = chat_from(json!({
            "choices": [{
                "index": 0,
                "message": {"role": "assistant", "content": "x"},
                "finish_reason": "stop",
            }],
            "usage": {
                "prompt_tokens": 100, "completion_tokens": 5, "total_tokens": 105,
                "cache_read_input_tokens": 60, "cache_creation_input_tokens": 20
            },
        }));
        let resp = synthesize_anthropic_response(&chat, &ctx(&[]));
        assert_eq!(resp.usage.input_tokens, 100);
        assert_eq!(resp.usage.output_tokens, 5);
        assert_eq!(resp.usage.cache_read_input_tokens, Some(60));
        assert_eq!(resp.usage.cache_creation_input_tokens, Some(20));
    }

    #[test]
    fn usage_maps_openai_shaped_cached_tokens_to_cache_read() {
        let chat = chat_from(json!({
            "choices": [{
                "index": 0,
                "message": {"role": "assistant", "content": "x"},
                "finish_reason": "stop",
            }],
            "usage": {
                "prompt_tokens": 100, "completion_tokens": 5, "total_tokens": 105,
                "prompt_tokens_details": {"cached_tokens": 40}
            },
        }));
        let resp = synthesize_anthropic_response(&chat, &ctx(&[]));
        assert_eq!(resp.usage.cache_read_input_tokens, Some(40));
        assert!(resp.usage.cache_creation_input_tokens.is_none());
    }

    #[test]
    fn id_has_msg_prefix_and_model_falls_back_to_request() {
        let chat = text_response(Some("stop"), "x");
        let resp = synthesize_anthropic_response(&chat, &ctx(&[]));
        assert!(resp.id.starts_with("msg_"), "id was {}", resp.id);
        assert!(resp.id.len() > "msg_".len());

        let bare = chat_from(json!({
            "model": "",
            "choices": [{
                "index": 0,
                "message": {"role": "assistant", "content": "x"},
                "finish_reason": "stop",
            }],
        }));
        let resp = synthesize_anthropic_response(&bare, &ctx(&[]));
        assert_eq!(resp.model, "claude-test");
    }
}
