//! `POST /v1/messages/count_tokens` logic (spec Requirement 5.3).
//!
//! Estimates the input token count of an Anthropic Messages request by
//! translating it exactly like `POST /v1/messages` (via
//! [`crate::anthropic::translate_request::translate`]) and counting the
//! resulting Chat Completions body with the gateway's existing tokenizer
//! helper ([`crate::compression::token_counter::TokenCounter`], the same
//! counter every `src/context/` module uses).
//!
//! Pure function module: no HTTP, no state, no clock, no model-group lookup —
//! handler wiring supplies the counter and target model later.

use crate::anthropic::models::AnthropicRequest;
use crate::compression::token_counter::TokenCounter;

/// Estimate the input token count of an Anthropic Messages request for the
/// group's primary model.
///
/// The request is translated with [`crate::anthropic::translate_request`]
/// (identical translation to Requirement 1) and counted with
/// [`TokenCounter::count_request`] using the tokenizer the counter selects
/// for `model`. `stream` is forced off so the flag never perturbs the count
/// (the field would otherwise round-trip into the translated request).
///
/// # Errors
///
/// Returns the translator's rejection message verbatim for unsupported
/// capabilities (hosted tools, `document` blocks, `context_management`);
/// the handler maps it to 400 `invalid_request_error`.
pub fn count_tokens(request: &AnthropicRequest, counter: &TokenCounter, model: &str) -> Result<u64, String> {
    let mut openai = crate::anthropic::translate_request::translate(request)?;
    openai.model = model.to_string();
    openai.stream = false;
    Ok(u64::from(counter.count_request(&openai)))
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    /// Pinned fixture: a representative request exercising system, tools,
    /// multi-message history, and sampling fields.
    fn fixture() -> AnthropicRequest {
        serde_json::from_value(json!({
            "model": "claude-group",
            "max_tokens": 1024,
            "system": "You are a concise assistant.",
            "messages": [
                { "role": "user", "content": "What is the weather in Paris?" },
                {
                    "role": "assistant",
                    "content": [
                        { "type": "text", "text": "Checking." },
                        { "type": "tool_use", "id": "toolu_1", "name": "get_weather", "input": { "city": "Paris" } }
                    ]
                },
                {
                    "role": "user",
                    "content": [
                        { "type": "tool_result", "tool_use_id": "toolu_1", "content": "18C sunny" }
                    ]
                }
            ],
            "temperature": 0.2,
            "top_p": 0.9,
            "stop_sequences": ["END"],
            "tools": [
                {
                    "name": "get_weather",
                    "description": "Get the current weather for a city.",
                    "input_schema": { "type": "object", "properties": { "city": { "type": "string" } } }
                }
            ]
        }))
        .expect("fixture must deserialize")
    }

    #[test]
    fn count_matches_helper_output_for_translated_fixture() {
        let request = fixture();
        let counter = TokenCounter::new();

        let result = count_tokens(&request, &counter, "gpt-4o");

        // Expected value computed by running the same helper here, not a
        // hardcoded magic number: the function output must equal the helper
        // output for the translated body.
        let mut translated =
            crate::anthropic::translate_request::translate(&request).expect("translates");
        translated.model = "gpt-4o".to_string();
        translated.stream = false;
        let expected = counter.count_request(&translated);
        let count = result.expect("counted");
        assert_eq!(count, u64::from(expected));

        // Stable sanity bound: fixture is small but non-trivial.
        assert!((20..200).contains(&count));
    }

    #[test]
    fn count_uses_model_selected_tokenizer() {
        let request = fixture();
        let counter = TokenCounter::new();

        let o200k = count_tokens(&request, &counter, "gpt-4o").expect("counted");
        let cl100k = count_tokens(&request, &counter, "gpt-4").expect("counted");
        // Both are valid estimates of the same body; they need not be equal,
        // but both must be positive and within a stable band.
        assert!(o200k > 0 && cl100k > 0);
        assert!((o200k as i64 - cl100k as i64).abs() < 50);
    }

    #[test]
    fn empty_request_counts_model_and_priming_only() {
        let request: AnthropicRequest = serde_json::from_value(json!({
            "model": "claude-group",
            "max_tokens": 16,
            "messages": []
        }))
        .expect("minimal request parses");
        let counter = TokenCounter::new();

        let count = count_tokens(&request, &counter, "gpt-4o").expect("counted");
        let mut translated =
            crate::anthropic::translate_request::translate(&request).expect("translates");
        translated.model = "gpt-4o".to_string();
        translated.stream = false;
        assert_eq!(count, u64::from(counter.count_request(&translated)));
        assert!(count > 0, "model name plus reply priming must count");
    }

    #[test]
    fn translator_rejections_propagate_verbatim() {
        let request: AnthropicRequest = serde_json::from_value(json!({
            "model": "claude-group",
            "max_tokens": 16,
            "messages": [{ "role": "user", "content": "hi" }],
            "context_management": { "edits": [{ "type": "clear_tool_uses_20250919" }] }
        }))
        .expect("request with rejection parses");
        let counter = TokenCounter::new();

        let err = count_tokens(&request, &counter, "gpt-4o").expect_err("must reject");
        assert_eq!(err, "context_management is not supported");
    }
}
