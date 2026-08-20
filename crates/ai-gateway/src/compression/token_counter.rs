//! Deterministic token counting for OpenAI-compatible requests.

use crate::models::openai::{Message, OpenAIRequest};
use serde_json::Value;
use tiktoken_rs::{cl100k_base_singleton, o200k_base_singleton, CoreBPE};

const TOKENS_PER_MESSAGE: u32 = 4;
const REPLY_PRIMING_TOKENS: u32 = 2;

/// Cached tokenizers used to estimate complete request sizes.
#[derive(Clone, Copy)]
pub struct TokenCounter {
    cl100k: &'static CoreBPE,
    o200k: &'static CoreBPE,
}

impl TokenCounter {
    /// Creates a counter backed by process-wide tokenizer singletons.
    pub fn new() -> Self {
        Self {
            cl100k: cl100k_base_singleton(),
            o200k: o200k_base_singleton(),
        }
    }

    /// Selects `o200k_base` for newer OpenAI model families and `cl100k_base`
    /// for all other models.
    pub fn for_model(&self, model: &str) -> &'static CoreBPE {
        if Self::uses_o200k(model) {
            self.o200k
        } else {
            self.cl100k
        }
    }

    /// Counts a text value with the tokenizer selected for `model`.
    ///
    /// This narrow API can be benchmarked without constructing a request.
    pub fn count_text(&self, model: &str, text: &str) -> u32 {
        Self::count_with(self.for_model(model), text)
    }

    /// Counts every model-visible part of an OpenAI-compatible request.
    ///
    /// The estimate includes the model name, each message role and complete
    /// content value, message extension fields such as tool calls, per-message
    /// framing, top-level tools, tool choice and all other flattened extension
    /// fields. `tools` and `tool_choice` are handled separately so neither can
    /// be counted twice.
    pub fn count_request(&self, request: &OpenAIRequest) -> u32 {
        let bpe = self.for_model(&request.model);
        let mut total = Self::count_with(bpe, &request.model);

        for message in &request.messages {
            total = total.saturating_add(Self::count_message_with(bpe, message));
        }

        if let Some(tools) = request.extra.get("tools") {
            total = total.saturating_add(Self::count_json_with(bpe, tools));
        }
        if let Some(tool_choice) = request.extra.get("tool_choice") {
            total = total.saturating_add(Self::count_json_with(bpe, tool_choice));
        }

        for (key, value) in &request.extra {
            if key == "tools" || key == "tool_choice" {
                continue;
            }
            total = total.saturating_add(Self::count_with(bpe, key));
            total = total.saturating_add(Self::count_json_with(bpe, value));
        }

        total.saturating_add(REPLY_PRIMING_TOKENS)
    }

    /// Character-based fallback for callers that cannot use a tokenizer.
    ///
    /// Text containing more than 30% CJK characters uses one token per two
    /// characters; all other text uses one token per four characters. Division
    /// rounds up so every non-empty input has a useful non-zero estimate.
    pub fn estimate_heuristic(text: &str) -> u32 {
        if text.is_empty() {
            return 0;
        }

        let mut total_chars = 0usize;
        let mut cjk_chars = 0usize;
        for character in text.chars() {
            total_chars = total_chars.saturating_add(1);
            if Self::is_cjk(character) {
                cjk_chars = cjk_chars.saturating_add(1);
            }
        }

        let chars_per_token = if cjk_chars.saturating_mul(100) > total_chars.saturating_mul(30) {
            2
        } else {
            4
        };
        let estimate = total_chars.div_ceil(chars_per_token);
        estimate.min(u32::MAX as usize) as u32
    }

    fn uses_o200k(model: &str) -> bool {
        let model = model.to_ascii_lowercase();
        model.contains("4o")
            || Self::contains_model_family(&model, "o1")
            || Self::contains_model_family(&model, "o3")
            || Self::contains_model_family(&model, "o4")
            || model.contains("chatgpt")
    }

    fn contains_model_family(model: &str, family: &str) -> bool {
        model.match_indices(family).any(|(index, _)| {
            let before = model[..index].chars().next_back();
            let after = model[index + family.len()..].chars().next();
            before.is_none_or(|character| !character.is_ascii_alphanumeric())
                && after.is_none_or(|character| !character.is_ascii_alphanumeric())
        })
    }

    fn count_message_with(bpe: &CoreBPE, message: &Message) -> u32 {
        let mut total = TOKENS_PER_MESSAGE;
        total = total.saturating_add(Self::count_with(bpe, &message.role));
        total = total.saturating_add(Self::count_content_with(bpe, &message.content));
        if !message.extra.is_empty() {
            total = total.saturating_add(Self::count_json_with(
                bpe,
                &Value::Object(message.extra.clone()),
            ));
        }
        total
    }

    fn count_content_with(bpe: &CoreBPE, content: &Value) -> u32 {
        match content {
            Value::Null => 0,
            Value::String(text) => Self::count_with(bpe, text),
            structured => Self::count_json_with(bpe, structured),
        }
    }

    fn count_json_with(bpe: &CoreBPE, value: &Value) -> u32 {
        Self::count_with(bpe, &value.to_string())
    }

    fn count_with(bpe: &CoreBPE, text: &str) -> u32 {
        bpe.encode_with_special_tokens(text)
            .len()
            .min(u32::MAX as usize) as u32
    }

    fn is_cjk(character: char) -> bool {
        matches!(
            character,
            '\u{2E80}'..='\u{2EFF}'
                | '\u{2F00}'..='\u{2FDF}'
                | '\u{3040}'..='\u{30FF}'
                | '\u{3100}'..='\u{312F}'
                | '\u{31A0}'..='\u{31BF}'
                | '\u{31C0}'..='\u{31EF}'
                | '\u{3400}'..='\u{4DBF}'
                | '\u{4E00}'..='\u{9FFF}'
                | '\u{AC00}'..='\u{D7AF}'
                | '\u{F900}'..='\u{FAFF}'
                | '\u{20000}'..='\u{2FA1F}'
        )
    }
}

impl Default for TokenCounter {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use proptest::prelude::*;
    use serde_json::{json, Map};
    use std::ptr;

    fn message(role: &str, content: Value) -> Message {
        Message {
            role: role.to_owned(),
            content,
            extra: Map::new(),
        }
    }

    fn request(model: &str, messages: Vec<Message>, extra: Map<String, Value>) -> OpenAIRequest {
        OpenAIRequest {
            model: model.to_owned(),
            messages,
            stream: false,
            temperature: None,
            max_tokens: None,
            extra,
        }
    }

    #[test]
    fn maps_modern_models_to_o200k_and_others_to_cl100k() {
        let counter = TokenCounter::new();

        for model in [
            "gpt-4o",
            "GPT-4O-mini",
            "o1-preview",
            "o3-mini",
            "o4-mini",
            "chatgpt-4o-latest",
        ] {
            assert!(ptr::eq(counter.for_model(model), o200k_base_singleton()));
        }
        for model in [
            "gpt-4",
            "gpt-3.5-turbo",
            "claude-3-5-sonnet",
            "command-r-08-2024",
            "unknown",
        ] {
            assert!(ptr::eq(counter.for_model(model), cl100k_base_singleton()));
        }
    }

    #[test]
    fn known_text_has_nonzero_counts() {
        let counter = TokenCounter::new();

        assert_eq!(counter.count_text("gpt-4", "hello world"), 2);
        assert_eq!(counter.count_text("gpt-4o", "hello world"), 2);
        assert_eq!(counter.count_text("gpt-4", ""), 0);
    }

    #[test]
    fn heuristic_uses_cjk_ratio_and_never_loses_nonempty_text() {
        assert_eq!(TokenCounter::estimate_heuristic("abcdefgh"), 2);
        assert_eq!(TokenCounter::estimate_heuristic("你好世界"), 2);
        assert_eq!(TokenCounter::estimate_heuristic("a"), 1);
        assert_eq!(TokenCounter::estimate_heuristic("界"), 1);
        assert_eq!(TokenCounter::estimate_heuristic(""), 0);
    }

    #[test]
    fn counts_complete_request_with_system_messages_and_tools() {
        let counter = TokenCounter::new();
        let mut assistant = message(
            "assistant",
            json!([{"type": "text", "text": "Checking."}, {"type": "image_url", "image_url": {"url": "https://example.test/a.png"}}]),
        );
        assistant.extra.insert(
            "tool_calls".to_owned(),
            json!([{"id": "call_1", "type": "function", "function": {"name": "weather", "arguments": "{\"city\":\"Paris\"}"}}]),
        );
        let tools = json!([{"type": "function", "function": {"name": "weather", "parameters": {"type": "object", "properties": {"city": {"type": "string"}}}}}]);
        let mut extra = Map::new();
        extra.insert("tools".to_owned(), tools.clone());
        extra.insert(
            "tool_choice".to_owned(),
            json!({"type": "function", "function": {"name": "weather"}}),
        );
        let request = request(
            "gpt-4o",
            vec![
                message("system", json!("You are concise.")),
                message("user", json!("What is the weather?")),
                assistant,
            ],
            extra,
        );

        let count = counter.count_request(&request);
        let component_floor = counter.count_text(&request.model, &request.model)
            + request
                .messages
                .iter()
                .map(|message| {
                    counter.count_text(&request.model, &message.role)
                        + TokenCounter::count_content_with(
                            counter.for_model(&request.model),
                            &message.content,
                        )
                })
                .sum::<u32>()
            + TokenCounter::count_json_with(counter.for_model(&request.model), &tools);

        assert!(count > component_floor);
        assert_eq!(count, counter.count_request(&request));
    }

    #[test]
    fn counts_one_hundred_thousand_characters_without_timing_assumptions() {
        let counter = TokenCounter::new();
        let text = "a".repeat(100_000);

        assert!(counter.count_text("gpt-4o", &text) > 0);
    }

    proptest! {
        #![proptest_config(ProptestConfig::with_cases(64))]

        #[test]
        fn property_14_request_count_covers_component_counts(
            model in "[a-z0-9.-]{1,24}",
            system in ".{0,128}",
            contents in prop::collection::vec(".{0,128}", 0..8),
            tool_name in "[a-z_]{1,24}",
        ) {
            let counter = TokenCounter::new();
            let mut messages = vec![message("system", Value::String(system))];
            messages.extend(
                contents
                    .into_iter()
                    .map(|content| message("user", Value::String(content))),
            );
            let tools = json!([{"type": "function", "function": {"name": tool_name, "parameters": {"type": "object"}}}]);
            let mut extra = Map::new();
            extra.insert("tools".to_owned(), tools.clone());
            let request = request(&model, messages, extra);
            let bpe = counter.for_model(&request.model);

            let component_sum = TokenCounter::count_with(bpe, &request.model)
                .saturating_add(request.messages.iter().fold(0u32, |sum, message| {
                    sum.saturating_add(TokenCounter::count_with(bpe, &message.role))
                        .saturating_add(TokenCounter::count_content_with(bpe, &message.content))
                }))
                .saturating_add(TokenCounter::count_json_with(bpe, &tools));

            prop_assert!(counter.count_request(&request) >= component_sum);
        }
    }
}
