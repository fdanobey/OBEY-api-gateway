//! Caveman output-mode prompt handling.

use super::{
    engines::{CompressibleMessage, CompressiblePayload, MessageContent, ToolRelationshipMetadata},
    token_counter::TokenCounter,
};
use crate::models::openai::OpenAIRequest;
use serde_json::{Map, Value};

/// Exact instruction added when caveman output mode is enabled.
pub const CAVEMAN_OUTPUT_SUFFIX: &str = "Respond concisely. No filler, no pleasantries, no hedging. Use abbreviations. Omit obvious context. Code-only responses when answering code questions.";

const OUTPUT_INSTRUCTION_KEYWORDS: [&str; 4] = ["format", "style", "tone", "respond"];

/// A request representation that can receive the resolved caveman setting.
pub trait CavemanOutputTarget {
    /// Applies caveman output instructions when `enabled` is the effective
    /// global/provider/model-group setting. Returns whether the request changed.
    fn apply_caveman_output(&mut self, enabled: bool) -> bool;
}

/// Applies the already-resolved caveman setting to either supported request type.
pub fn apply_caveman_output<T>(target: &mut T, enabled: bool) -> bool
where
    T: CavemanOutputTarget + ?Sized,
{
    target.apply_caveman_output(enabled)
}

impl CavemanOutputTarget for CompressiblePayload {
    fn apply_caveman_output(&mut self, enabled: bool) -> bool {
        apply_to_payload(self, enabled)
    }
}

impl CavemanOutputTarget for OpenAIRequest {
    fn apply_caveman_output(&mut self, enabled: bool) -> bool {
        if !enabled {
            return false;
        }

        let mut payload = CompressiblePayload::from_openai_request(self.clone());
        if !apply_to_payload(&mut payload, true) {
            return false;
        }

        *self = payload.into_openai_request();
        true
    }
}

fn apply_to_payload(payload: &mut CompressiblePayload, enabled: bool) -> bool {
    if !enabled {
        return false;
    }

    payload.refresh_metadata();
    let system_indices = payload
        .messages
        .iter()
        .enumerate()
        .filter_map(|(index, message)| message.is_system().then_some(index))
        .collect::<Vec<_>>();

    if system_indices
        .iter()
        .any(|index| content_has_output_instruction(payload.messages[*index].content.as_value()))
    {
        return false;
    }

    let first_mutable_system = system_indices
        .iter()
        .copied()
        .find(|index| !payload.messages[*index].cache_protected);
    let cache_boundary = payload
        .messages
        .iter()
        .rposition(|message| message.cache_protected);

    if let Some(system_index) = first_mutable_system {
        append_suffix(&mut payload.messages[system_index].content);
        payload.messages[system_index].token_count =
            count_message_tokens(&payload.model, &payload.messages[system_index]);
        payload.refresh_metadata();
        return true;
    }

    if system_indices.is_empty() && cache_boundary.is_none() {
        payload
            .messages
            .insert(0, new_system_message(&payload.model));
        payload.refresh_metadata();
        return true;
    }

    let Some(boundary) = cache_boundary else {
        return false;
    };

    // A system message inserted after a cache marker is safe only while that
    // marker still belongs to the leading system prefix. If caching extends
    // into conversation history, inserting a system role there could invalidate
    // provider role ordering, so caveman mode deliberately becomes a no-op.
    if !payload.messages[..=boundary]
        .iter()
        .all(CompressibleMessage::is_system)
    {
        return false;
    }

    payload
        .messages
        .insert(boundary + 1, new_system_message(&payload.model));
    payload.refresh_metadata();
    true
}

fn content_has_output_instruction(content: &Value) -> bool {
    match content {
        Value::String(text) => text_has_output_instruction(text),
        Value::Array(blocks) => blocks.iter().any(content_block_has_output_instruction),
        Value::Object(_) => content_block_has_output_instruction(content),
        _ => false,
    }
}

fn content_block_has_output_instruction(block: &Value) -> bool {
    match block {
        Value::String(text) => text_has_output_instruction(text),
        Value::Object(object)
            if matches!(
                object.get("type").and_then(Value::as_str),
                Some("text" | "input_text" | "output_text")
            ) =>
        {
            object
                .get("text")
                .and_then(Value::as_str)
                .is_some_and(text_has_output_instruction)
        }
        _ => false,
    }
}

fn text_has_output_instruction(text: &str) -> bool {
    text.split(|character: char| !character.is_alphanumeric())
        .filter(|word| !word.is_empty())
        .any(|word| {
            OUTPUT_INSTRUCTION_KEYWORDS
                .iter()
                .any(|keyword| word.eq_ignore_ascii_case(keyword))
        })
}

fn append_suffix(content: &mut MessageContent) {
    match content.as_value_mut() {
        Value::String(text) => {
            if !text.is_empty() {
                text.push_str("\n\n");
            }
            text.push_str(CAVEMAN_OUTPUT_SUFFIX);
        }
        Value::Array(blocks) => blocks.push(text_block()),
        value @ Value::Object(_) => {
            let original = std::mem::replace(value, Value::Null);
            *value = Value::Array(vec![original, text_block()]);
        }
        value => {
            *value = Value::String(CAVEMAN_OUTPUT_SUFFIX.to_owned());
        }
    }
}

fn text_block() -> Value {
    let mut block = Map::new();
    block.insert("type".to_owned(), Value::String("text".to_owned()));
    block.insert(
        "text".to_owned(),
        Value::String(CAVEMAN_OUTPUT_SUFFIX.to_owned()),
    );
    Value::Object(block)
}

fn new_system_message(model: &str) -> CompressibleMessage {
    let mut message = CompressibleMessage {
        role: "system".to_owned(),
        content: MessageContent::new(Value::String(CAVEMAN_OUTPUT_SUFFIX.to_owned())),
        extra: Map::new(),
        age: 0,
        token_count: 0,
        cache_protected: false,
        original_index: usize::MAX,
        critical: true,
        relationships: ToolRelationshipMetadata::default(),
    };
    message.token_count = count_message_tokens(model, &message);
    message
}

fn count_message_tokens(model: &str, message: &CompressibleMessage) -> u32 {
    let counter = TokenCounter::new();
    let content_tokens = match message.content.as_value() {
        Value::Null => 0,
        Value::String(text) => counter.count_text(model, text),
        structured => counter.count_text(model, &structured.to_string()),
    };
    let extra_tokens = if message.extra.is_empty() {
        0
    } else {
        counter.count_text(model, &Value::Object(message.extra.clone()).to_string())
    };

    4u32.saturating_add(counter.count_text(model, &message.role))
        .saturating_add(content_tokens)
        .saturating_add(extra_tokens)
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    fn payload(value: Value) -> CompressiblePayload {
        let request: OpenAIRequest = serde_json::from_value(value).unwrap();
        request.into()
    }

    #[test]
    fn suffix_is_exact() {
        assert_eq!(
            CAVEMAN_OUTPUT_SUFFIX,
            "Respond concisely. No filler, no pleasantries, no hedging. Use abbreviations. Omit obvious context. Code-only responses when answering code questions."
        );
    }

    #[test]
    fn appends_to_existing_system_or_inserts_new_system_first() {
        let mut existing = payload(json!({
            "model": "gpt-4o",
            "messages": [
                {"role": "system", "content": "Keep facts accurate.", "name": "policy"},
                {"role": "user", "content": "Question"}
            ]
        }));
        assert!(apply_caveman_output(&mut existing, true));
        let expected = format!("Keep facts accurate.\n\n{CAVEMAN_OUTPUT_SUFFIX}");
        assert_eq!(
            existing.messages[0].content.as_text(),
            Some(expected.as_str())
        );
        assert_eq!(
            existing.messages[0].extra.get("name"),
            Some(&json!("policy"))
        );

        let mut missing = payload(json!({
            "model": "gpt-4o",
            "messages": [{"role": "user", "content": "Question", "name": "customer"}]
        }));
        assert!(apply_caveman_output(&mut missing, true));
        assert_eq!(missing.messages[0].role, "system");
        assert_eq!(
            missing.messages[0].content.as_text(),
            Some(CAVEMAN_OUTPUT_SUFFIX)
        );
        assert_eq!(missing.messages[1].role, "user");
        assert_eq!(
            missing.messages[1].extra.get("name"),
            Some(&json!("customer"))
        );
    }

    #[test]
    fn skips_case_insensitive_whole_word_output_keywords() {
        for instruction in [
            "Use this FoRmAt.",
            "Follow the requested STYLE.",
            "Maintain a terse ToNe!",
            "RESPOND with JSON.",
        ] {
            let mut input = payload(json!({
                "model": "gpt-4o",
                "messages": [{"role": "system", "content": instruction}]
            }));
            let before = input.clone();
            assert!(!apply_caveman_output(&mut input, true));
            assert_eq!(input, before);
        }

        let mut substring_only = payload(json!({
            "model": "gpt-4o",
            "messages": [{"role": "system", "content": "Reformatted historical data."}]
        }));
        assert!(apply_caveman_output(&mut substring_only, true));
    }

    #[test]
    fn disabled_mode_is_unchanged() {
        let mut input = payload(json!({
            "model": "gpt-4o",
            "messages": [{"role": "user", "content": "Question"}],
            "tools": [{"type": "function", "function": {"name": "lookup"}}]
        }));
        let before = input.clone();
        assert!(!apply_caveman_output(&mut input, false));
        assert_eq!(input, before);
    }

    #[test]
    fn application_is_idempotent() {
        let mut input = payload(json!({
            "model": "gpt-4o",
            "messages": [{"role": "system", "content": "Keep facts accurate."}]
        }));
        assert!(apply_caveman_output(&mut input, true));
        let once = input.clone();
        assert!(!apply_caveman_output(&mut input, true));
        assert_eq!(input, once);
    }

    #[test]
    fn structured_system_content_is_preserved_and_scanned() {
        let original_blocks = json!([
            {"type": "text", "text": "Keep facts accurate.", "provider_data": {"x": 1}},
            {"type": "image_url", "image_url": {"url": "https://example.test/a.png"}},
            {"type": "custom", "payload": [1, 2, 3]}
        ]);
        let mut input = payload(json!({
            "model": "gpt-4o",
            "messages": [{"role": "system", "content": original_blocks, "provider_extra": true}]
        }));
        assert!(apply_caveman_output(&mut input, true));
        let blocks = input.messages[0].content.as_value().as_array().unwrap();
        assert_eq!(&blocks[..3], original_blocks.as_array().unwrap());
        assert_eq!(blocks[3], text_block());
        assert_eq!(
            input.messages[0].extra.get("provider_extra"),
            Some(&json!(true))
        );

        let mut instructed = payload(json!({
            "model": "gpt-4o",
            "messages": [{
                "role": "system",
                "content": [
                    {"type": "image_url", "image_url": {"url": "https://format.example"}},
                    {"type": "text", "text": "Choose the TONE carefully."}
                ]
            }]
        }));
        assert!(!apply_caveman_output(&mut instructed, true));
    }

    #[test]
    fn cached_system_prefix_remains_byte_stable_and_unsafe_boundary_skips() {
        let mut input = payload(json!({
            "model": "claude-test",
            "messages": [
                {
                    "role": "system",
                    "content": [{
                        "type": "text",
                        "text": "Stable cached policy.",
                        "cache_control": {"type": "ephemeral"}
                    }],
                    "provider_extra": {"stable": true}
                },
                {"role": "user", "content": "Question"}
            ]
        }));
        let cached_system = input.messages[0].clone();
        assert!(apply_caveman_output(&mut input, true));
        assert_eq!(input.messages[0], cached_system);
        assert!(input.messages[0].cache_protected);
        assert_eq!(input.messages[1].role, "system");
        assert_eq!(
            input.messages[1].content.as_text(),
            Some(CAVEMAN_OUTPUT_SUFFIX)
        );
        assert!(!input.messages[1].cache_protected);
        assert_eq!(input.messages[2].role, "user");

        let mut cached_and_uncached_systems = payload(json!({
            "model": "claude-test",
            "messages": [
                {
                    "role": "system",
                    "content": [{
                        "type": "text",
                        "text": "Stable cached policy.",
                        "cache_control": {"type": "ephemeral"}
                    }]
                },
                {"role": "system", "content": "Mutable policy.", "name": "secondary"},
                {"role": "user", "content": "Question"}
            ]
        }));
        let cached = cached_and_uncached_systems.messages[0].clone();
        assert!(apply_caveman_output(&mut cached_and_uncached_systems, true));
        assert_eq!(cached_and_uncached_systems.messages[0], cached);
        let expected = format!("Mutable policy.\n\n{CAVEMAN_OUTPUT_SUFFIX}");
        assert_eq!(
            cached_and_uncached_systems.messages[1].content.as_text(),
            Some(expected.as_str())
        );
        assert_eq!(
            cached_and_uncached_systems.messages[1].extra.get("name"),
            Some(&json!("secondary"))
        );

        let mut unsafe_history_boundary = payload(json!({
            "model": "claude-test",
            "messages": [
                {"role": "system", "content": "Stable policy."},
                {
                    "role": "user",
                    "content": [{
                        "type": "text",
                        "text": "Cached question",
                        "cache_control": {"type": "ephemeral"}
                    }]
                }
            ]
        }));
        let before = unsafe_history_boundary.clone();
        assert!(!apply_caveman_output(&mut unsafe_history_boundary, true));
        assert_eq!(unsafe_history_boundary, before);
    }

    #[test]
    fn openai_api_preserves_tools_extras_and_existing_message_order() {
        let mut disabled_request: OpenAIRequest = serde_json::from_value(json!({
            "model": "gpt-4o",
            "messages": [{"role": "user", "content": "Question"}],
            "tools": [{"type": "function", "function": {"name": "lookup"}}]
        }))
        .unwrap();
        let disabled_before = serde_json::to_value(&disabled_request).unwrap();
        assert!(!apply_caveman_output(&mut disabled_request, false));
        assert_eq!(
            serde_json::to_value(&disabled_request).unwrap(),
            disabled_before
        );

        let mut request: OpenAIRequest = serde_json::from_value(json!({
            "model": "gpt-4o",
            "messages": [
                {
                    "role": "assistant",
                    "content": "Calling tool",
                    "tool_calls": [{
                        "id": "call_1",
                        "type": "function",
                        "function": {"name": "lookup", "arguments": "{\"id\":1}"}
                    }]
                },
                {"role": "tool", "tool_call_id": "call_1", "content": "result"},
                {"role": "user", "content": "Continue", "provider_message_extra": 7}
            ],
            "tools": [{
                "type": "function",
                "function": {
                    "name": "lookup",
                    "description": "Lookup",
                    "parameters": {"type": "object"}
                }
            }],
            "tool_choice": "auto",
            "provider_request_extra": {"kept": true}
        }))
        .unwrap();
        let original_messages = serde_json::to_value(&request.messages).unwrap();
        let original_tools = request.extra.get("tools").cloned();
        let original_extra = request.extra.get("provider_request_extra").cloned();

        assert!(apply_caveman_output(&mut request, true));
        assert_eq!(request.messages[0].role, "system");
        assert_eq!(request.messages[0].content, json!(CAVEMAN_OUTPUT_SUFFIX));
        assert_eq!(
            serde_json::to_value(&request.messages[1..]).unwrap(),
            original_messages
        );
        assert_eq!(request.extra.get("tools").cloned(), original_tools);
        assert_eq!(request.extra.get("tool_choice"), Some(&json!("auto")));
        assert_eq!(
            request.extra.get("provider_request_extra").cloned(),
            original_extra
        );
    }
}
