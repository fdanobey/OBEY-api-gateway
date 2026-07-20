//! Lite compression engine.

use super::{
    CompressibleMessage, CompressiblePayload, CompressionContext, CompressionEngine, EngineResult,
    MessageContent,
};
use crate::compression::token_counter::TokenCounter;
use async_trait::async_trait;
use serde_json::Value;
use std::time::Instant;

const TOOL_RESULT_TRIGGER_TOKENS: u32 = 2_000;
const TOOL_RESULT_PREFIX_TOKENS: u32 = 1_500;
const DATA_URI_MIN_BYTES: usize = 100;
const TRUNCATED_MARKER: &str = "[truncated]";

/// Safe, low-risk compression for request message text.
#[derive(Debug, Clone, Copy, Default)]
pub struct LiteEngine;

impl LiteEngine {
    pub fn new() -> Self {
        Self
    }
}

#[async_trait]
impl CompressionEngine for LiteEngine {
    fn name(&self) -> &str {
        "lite"
    }

    async fn compress(
        &self,
        payload: &mut CompressiblePayload,
        context: &CompressionContext,
    ) -> EngineResult {
        let started = Instant::now();
        let original = payload.clone();
        let tokens_before = count_payload_tokens(&original, context);
        let mut changed = deduplicate_system_prompts(payload);

        for message in &mut payload.messages {
            if message.cache_protected {
                continue;
            }

            let tool_tokens = if is_tool_result(message) {
                count_text_leaves(&message.content, context)
            } else {
                0
            };

            message.content.for_each_text_leaf_mut(|text| {
                let transformed = context
                    .protection_scanner
                    .transform_unprotected(text, |segment| {
                        normalize_whitespace(&replace_long_base64_data_uris(segment))
                    });
                if transformed != *text {
                    *text = transformed;
                    changed = true;
                }
            });

            if tool_tokens > TOOL_RESULT_TRIGGER_TOKENS
                && truncate_tool_result_text(&mut message.content, context)
            {
                changed = true;
            }
        }

        if changed {
            payload.refresh_metadata();
            refresh_message_token_counts(payload, context);
        }

        let (tokens_after, applied) =
            keep_only_non_increasing(payload, &original, context, tokens_before, changed);

        EngineResult {
            engine_name: self.name().to_owned(),
            tokens_before,
            tokens_after,
            duration_ms: started.elapsed().as_millis().min(u64::MAX as u128) as u64,
            applied,
        }
    }
}

fn count_payload_tokens(payload: &CompressiblePayload, context: &CompressionContext) -> u32 {
    context
        .token_counter
        .count_request(&payload.clone().into_openai_request())
}

fn keep_only_non_increasing(
    payload: &mut CompressiblePayload,
    original: &CompressiblePayload,
    context: &CompressionContext,
    tokens_before: u32,
    changed: bool,
) -> (u32, bool) {
    let tokens_after = count_payload_tokens(payload, context);
    if changed && tokens_after <= tokens_before {
        (tokens_after, true)
    } else {
        if tokens_after > tokens_before {
            *payload = original.clone();
        }
        (tokens_before.min(tokens_after), false)
    }
}

fn deduplicate_system_prompts(payload: &mut CompressiblePayload) -> bool {
    let mut seen = Vec::<MessageContent>::new();
    let mut changed = false;

    for message in &mut payload.messages {
        if !message.is_system() {
            continue;
        }

        if seen.iter().any(|content| content == &message.content) {
            if !message.cache_protected {
                message.content.for_each_text_leaf_mut(|text| {
                    if !text.is_empty() {
                        text.clear();
                        changed = true;
                    }
                });
            }
        } else {
            seen.push(message.content.clone());
        }
    }

    changed
}

fn is_tool_result(message: &CompressibleMessage) -> bool {
    matches!(message.role.as_str(), "tool" | "function")
        || !message.relationships.tool_result_for_ids.is_empty()
}

fn count_text_leaves(content: &MessageContent, context: &CompressionContext) -> u32 {
    let mut content = content.clone();
    let mut tokens = 0u32;
    content.for_each_text_leaf_mut(|text| {
        tokens = tokens.saturating_add(context.token_counter.count_text(&context.model, text));
    });
    tokens
}

fn truncate_tool_result_text(content: &mut MessageContent, context: &CompressionContext) -> bool {
    let mut remaining = TOOL_RESULT_PREFIX_TOKENS;
    let mut removed = false;

    content.for_each_text_leaf_mut(|text| {
        let transformed = context
            .protection_scanner
            .transform_unprotected(text, |segment| {
                if segment.is_empty() {
                    return String::new();
                }

                let segment_tokens = context.token_counter.count_text(&context.model, segment);
                if segment_tokens <= remaining {
                    remaining -= segment_tokens;
                    segment.to_owned()
                } else {
                    let prefix = decoded_token_prefix(
                        context.token_counter.as_ref(),
                        &context.model,
                        segment,
                        remaining,
                    );
                    remaining = 0;
                    if prefix.len() < segment.len() {
                        removed = true;
                    }
                    prefix
                }
            });
        *text = transformed;
    });

    if !removed {
        return false;
    }

    let mut leaf_count = 0usize;
    let mut snapshot = content.clone();
    snapshot.for_each_text_leaf_mut(|_| leaf_count += 1);
    let mut leaf_index = 0usize;
    content.for_each_text_leaf_mut(|text| {
        leaf_index += 1;
        if leaf_index == leaf_count {
            text.push_str(TRUNCATED_MARKER);
        }
    });

    true
}

fn decoded_token_prefix(
    counter: &TokenCounter,
    model: &str,
    text: &str,
    maximum_tokens: u32,
) -> String {
    if maximum_tokens == 0 || text.is_empty() {
        return String::new();
    }

    let tokenizer = counter.for_model(model);
    let tokens = tokenizer.encode_with_special_tokens(text);
    if tokens.len() <= maximum_tokens as usize {
        return text.to_owned();
    }

    let mut token_end = maximum_tokens as usize;
    while token_end > 0 {
        if let Ok(prefix) = tokenizer.decode(&tokens[..token_end]) {
            if text.starts_with(&prefix) {
                return prefix;
            }
        }
        token_end -= 1;
    }

    conservative_character_prefix(counter, model, text, maximum_tokens)
}

fn conservative_character_prefix(
    counter: &TokenCounter,
    model: &str,
    text: &str,
    maximum_tokens: u32,
) -> String {
    let boundaries: Vec<usize> = text
        .char_indices()
        .map(|(index, _)| index)
        .chain(std::iter::once(text.len()))
        .collect();
    let mut low = 0usize;
    let mut high = boundaries.len();

    while low + 1 < high {
        let middle = low + (high - low) / 2;
        if counter.count_text(model, &text[..boundaries[middle]]) <= maximum_tokens {
            low = middle;
        } else {
            high = middle;
        }
    }

    text[..boundaries[low]].to_owned()
}

fn normalize_whitespace(text: &str) -> String {
    let mut output = String::with_capacity(text.len());
    let mut characters = text.chars().peekable();
    let mut newline_run = 0usize;

    while let Some(character) = characters.next() {
        match character {
            '\r' => {
                if characters.peek() == Some(&'\n') {
                    characters.next();
                }
                if newline_run < 2 {
                    output.push('\n');
                }
                newline_run += 1;
            }
            '\n' => {
                if newline_run < 2 {
                    output.push('\n');
                }
                newline_run += 1;
            }
            ' ' | '\t' => {
                newline_run = 0;
                output.push(' ');
                while matches!(characters.peek(), Some(' ' | '\t')) {
                    characters.next();
                }
            }
            _ => {
                newline_run = 0;
                output.push(character);
            }
        }
    }

    output
}

fn replace_long_base64_data_uris(text: &str) -> String {
    let mut output = String::with_capacity(text.len());
    let mut cursor = 0usize;

    while let Some(relative_start) = text[cursor..].find("data:") {
        let start = cursor + relative_start;
        output.push_str(&text[cursor..start]);

        let Some((end, mime_type)) = parse_base64_data_uri(text, start) else {
            output.push_str("data:");
            cursor = start + "data:".len();
            continue;
        };
        let original_length = end - start;
        if original_length > DATA_URI_MIN_BYTES {
            output.push_str(&format!(
                "[base64 image: {mime_type}, {original_length} bytes]"
            ));
        } else {
            output.push_str(&text[start..end]);
        }
        cursor = end;
    }

    output.push_str(&text[cursor..]);
    output
}

fn parse_base64_data_uri(text: &str, start: usize) -> Option<(usize, &str)> {
    let remainder = text.get(start..)?;
    let comma_offset = remainder.find(',')?;
    let header = &remainder["data:".len()..comma_offset];
    let mut header_parts = header.split(';');
    let mime_type = header_parts.next()?;
    if mime_type.is_empty()
        || !mime_type
            .bytes()
            .all(|byte| byte.is_ascii_alphanumeric() || matches!(byte, b'/' | b'+' | b'.' | b'-'))
        || !header_parts.any(|part| part.eq_ignore_ascii_case("base64"))
    {
        return None;
    }

    let data_start = start + comma_offset + 1;
    let data_length = text[data_start..]
        .bytes()
        .take_while(|byte| {
            byte.is_ascii_alphanumeric() || matches!(byte, b'+' | b'/' | b'=' | b'_' | b'-')
        })
        .count();
    (data_length > 0).then_some((data_start + data_length, mime_type))
}

fn refresh_message_token_counts(payload: &mut CompressiblePayload, context: &CompressionContext) {
    for message in &mut payload.messages {
        let content_tokens = match message.content.as_value() {
            Value::Null => 0,
            Value::String(text) => context.token_counter.count_text(&context.model, text),
            structured => context
                .token_counter
                .count_text(&context.model, &structured.to_string()),
        };
        let extra_tokens = if message.extra.is_empty() {
            0
        } else {
            context.token_counter.count_text(
                &context.model,
                &Value::Object(message.extra.clone()).to_string(),
            )
        };
        message.token_count = 4u32
            .saturating_add(
                context
                    .token_counter
                    .count_text(&context.model, &message.role),
            )
            .saturating_add(content_tokens)
            .saturating_add(extra_tokens);
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::models::openai::OpenAIRequest;
    use serde_json::{json, Value};

    fn payload(value: Value) -> CompressiblePayload {
        let request: OpenAIRequest = serde_json::from_value(value).unwrap();
        CompressiblePayload::from(request)
    }

    fn context() -> CompressionContext {
        CompressionContext::new("gpt-4o", "test")
    }

    async fn compress(payload: &mut CompressiblePayload) -> EngineResult {
        LiteEngine::new().compress(payload, &context()).await
    }

    #[tokio::test]
    async fn normalizes_only_unprotected_whitespace_and_line_endings() {
        let protected = "```rust\r\nlet  value =  1;\r\n```";
        let input = format!(
            "Intro  \t prose\r\n\r\n\r\nTail\n{protected}\nURL https://example.test/a?q=1  done\nPath C:\\Users\\alice\\main.rs  done\nJSON {{\"key\":  \"value\"}}  done\nIdentifier camelCase  done\nMath $x  + y$  done\nfn callMe(value:  i32)  done"
        );
        let mut payload = payload(json!({
            "model": "gpt-4o",
            "messages": [{"role": "user", "content": input}]
        }));

        let result = compress(&mut payload).await;
        let output = payload.messages[0].content.as_text().unwrap();

        assert!(result.applied);
        assert!(output.starts_with("Intro prose\n\nTail"));
        assert!(output.contains(protected));
        assert!(output.contains("https://example.test/a?q=1"));
        assert!(output.contains(r"C:\Users\alice\main.rs"));
        assert!(output.contains(r#"{"key":  "value"}"#));
        assert!(output.contains("camelCase"));
        assert!(output.contains("$x  + y$"));
        assert!(output.contains("fn callMe(value:  i32)"));
        assert!(!output.starts_with("Intro  "));
    }

    #[tokio::test]
    async fn deduplicates_exact_system_content_without_removing_messages_or_fields() {
        let mut payload = payload(json!({
            "model": "gpt-4o",
            "messages": [
                {"role": "system", "content": "Keep this policy.", "name": "first"},
                {"role": "user", "content": "Question"},
                {"role": "system", "content": "Keep this policy.", "name": "duplicate", "vendor": {"kept": true}}
            ]
        }));

        let result = compress(&mut payload).await;

        assert!(result.applied);
        assert_eq!(payload.messages.len(), 3);
        assert_eq!(
            payload.messages[0].content.as_text(),
            Some("Keep this policy.")
        );
        assert_eq!(payload.messages[2].content.as_text(), Some(""));
        assert_eq!(payload.messages[0].extra["name"], "first");
        assert_eq!(payload.messages[2].extra["name"], "duplicate");
        assert_eq!(payload.messages[2].extra["vendor"], json!({"kept": true}));
    }

    #[tokio::test]
    async fn replaces_long_textual_base64_uri_and_preserves_multimodal_uri() {
        let encoded = "0123456789+/".repeat(12);
        let uri = format!("data:image/png;base64,{encoded}");
        let original_length = uri.len();
        let mut payload = payload(json!({
            "model": "gpt-4o",
            "messages": [{
                "role": "user",
                "content": [
                    {"type": "text", "text": format!("Image: {uri}")},
                    {"type": "image_url", "image_url": {"url": uri}}
                ]
            }]
        }));

        let result = compress(&mut payload).await;
        let content = payload.messages[0].content.as_value();

        assert!(result.applied);
        assert_eq!(
            content[0]["text"],
            format!("Image: [base64 image: image/png, {original_length} bytes]")
        );
        assert_eq!(content[1]["image_url"]["url"], uri);
    }

    #[tokio::test]
    async fn truncates_large_unicode_tool_output_at_a_valid_token_prefix() {
        let large_output = "🙂 漢字 résumé ".repeat(900);
        let context = context();
        assert!(
            context
                .token_counter
                .count_text(&context.model, &large_output)
                > TOOL_RESULT_TRIGGER_TOKENS
        );
        let exact_prefix = decoded_token_prefix(
            context.token_counter.as_ref(),
            &context.model,
            &large_output,
            TOOL_RESULT_PREFIX_TOKENS,
        );
        let mut payload = payload(json!({
            "model": "gpt-4o",
            "messages": [{"role": "tool", "tool_call_id": "call-1", "name": "lookup", "content": large_output}]
        }));

        let result = LiteEngine::new().compress(&mut payload, &context).await;
        let output = payload.messages[0].content.as_text().unwrap();
        let prefix = output.strip_suffix(TRUNCATED_MARKER).unwrap();

        assert!(result.applied);
        assert_eq!(prefix, exact_prefix);
        assert!(large_output.starts_with(prefix));
        assert!(context.token_counter.count_text(&context.model, prefix) <= 1_500);
        assert!(context.token_counter.count_text(&context.model, prefix) >= 1_490);
        assert!(output.is_char_boundary(output.len()));
    }

    #[tokio::test]
    async fn tool_truncation_keeps_protected_regions_byte_identical() {
        let code = "```text\r\nPROTECTED  CODE  BLOCK\r\n```";
        let large_output = format!("{}\n{code}\n{}", "word ".repeat(2_100), "tail ".repeat(100));
        let mut payload = payload(json!({
            "model": "gpt-4o",
            "messages": [{"role": "tool", "tool_call_id": "call-1", "content": large_output}]
        }));

        let result = compress(&mut payload).await;
        let output = payload.messages[0].content.as_text().unwrap();

        assert!(result.applied);
        assert!(output.contains(code));
        assert!(output.ends_with(TRUNCATED_MARKER));
    }

    #[tokio::test]
    async fn preserves_structured_and_textual_json_tool_results() {
        let structured = json!({"rows": [{"path": "C:\\work\\file.rs", "value": "a  b"}]});
        let textual = r#"{"rows": [{"value": "a  b"}]}"#;
        let mut payload = payload(json!({
            "model": "gpt-4o",
            "messages": [
                {"role": "tool", "tool_call_id": "one", "content": structured},
                {"role": "tool", "tool_call_id": "two", "content": textual}
            ]
        }));

        let result = compress(&mut payload).await;

        assert!(!result.applied);
        assert_eq!(payload.messages[0].content.as_value(), &structured);
        assert_eq!(payload.messages[1].content.as_text(), Some(textual));
    }

    #[tokio::test]
    async fn cache_protected_messages_are_never_modified() {
        let mut payload = payload(json!({
            "model": "gpt-4o",
            "messages": [
                {"role": "system", "content": "Policy  with  spaces"},
                {"role": "user", "content": [{"type": "text", "text": "Cached  text", "cache_control": {"type": "ephemeral"}}]},
                {"role": "assistant", "content": "After  boundary"}
            ]
        }));
        let cached_system = payload.messages[0].content.clone();
        let cached_user = payload.messages[1].content.clone();

        let result = compress(&mut payload).await;

        assert!(result.applied);
        assert_eq!(payload.messages[0].content, cached_system);
        assert_eq!(payload.messages[1].content, cached_user);
        assert_eq!(
            payload.messages[2].content.as_text(),
            Some("After boundary")
        );
    }

    #[tokio::test]
    async fn preserves_tool_relationships_ids_names_fields_and_order() {
        let large_output = "result value ".repeat(2_100);
        let mut payload = payload(json!({
            "model": "gpt-4o",
            "messages": [
                {
                    "role": "assistant",
                    "content": "Calling  tool",
                    "tool_calls": [{"id": "call-7", "type": "function", "function": {"name": "lookup", "arguments": "{\"id\":7}"}}]
                },
                {"role": "tool", "tool_call_id": "call-7", "name": "lookup", "vendor": {"status": "ok"}, "content": large_output},
                {"role": "assistant", "content": "Done  now"}
            ]
        }));
        let original_extras: Vec<_> = payload
            .messages
            .iter()
            .map(|message| message.extra.clone())
            .collect();
        let original_indices: Vec<_> = payload
            .messages
            .iter()
            .map(|message| message.original_index)
            .collect();

        let result = compress(&mut payload).await;

        assert!(result.applied);
        assert_eq!(payload.messages.len(), 3);
        assert_eq!(
            payload
                .messages
                .iter()
                .map(|message| message.original_index)
                .collect::<Vec<_>>(),
            original_indices
        );
        assert_eq!(
            payload
                .messages
                .iter()
                .map(|message| message.extra.clone())
                .collect::<Vec<_>>(),
            original_extras
        );
        assert_eq!(payload.messages[0].relationships.tool_call_ids, ["call-7"]);
        assert_eq!(
            payload.messages[1].relationships.tool_result_for_ids,
            ["call-7"]
        );
        assert_eq!(
            payload.messages[0].relationships.related_message_indices,
            [1]
        );
        assert_eq!(
            payload.messages[1].relationships.related_message_indices,
            [0]
        );
    }

    #[tokio::test]
    async fn reports_full_request_counts_and_never_increases_tokens() {
        let cases = [
            json!({"model": "gpt-4o", "messages": [{"role": "user", "content": "Already concise."}]}),
            json!({"model": "gpt-4o", "messages": [{"role": "user", "content": "Many     spaces\r\n\r\n\r\nremain"}]}),
            json!({"model": "gpt-4o", "messages": [
                {"role": "system", "content": "same"},
                {"role": "system", "content": "same"}
            ]}),
        ];
        let context = context();

        for case in cases {
            let mut payload = payload(case);
            let original = payload.clone();
            let expected_before = count_payload_tokens(&original, &context);
            let result = LiteEngine::new().compress(&mut payload, &context).await;
            let actual_after = count_payload_tokens(&payload, &context);

            assert_eq!(result.engine_name, "lite");
            assert_eq!(result.tokens_before, expected_before);
            assert_eq!(result.tokens_after, actual_after);
            assert!(result.tokens_after <= result.tokens_before);
            if result.tokens_after > result.tokens_before {
                assert_eq!(payload, original);
                assert!(!result.applied);
            }
        }
    }

    #[test]
    fn no_token_increase_guard_restores_original_payload() {
        let original = payload(json!({
            "model": "gpt-4o",
            "messages": [{"role": "user", "content": "short"}]
        }));
        let mut transformed = original.clone();
        *transformed.messages[0].content.as_value_mut() =
            Value::String("expanded output".repeat(100));
        let context = context();
        let tokens_before = count_payload_tokens(&original, &context);

        let (tokens_after, applied) =
            keep_only_non_increasing(&mut transformed, &original, &context, tokens_before, true);

        assert!(!applied);
        assert_eq!(tokens_after, tokens_before);
        assert_eq!(transformed, original);
    }

    #[tokio::test]
    async fn unchanged_payload_reports_unapplied() {
        let mut payload = payload(json!({
            "model": "gpt-4o",
            "messages": [{"role": "user", "content": "Already concise."}]
        }));
        let original = payload.clone();

        let result = compress(&mut payload).await;

        assert!(!result.applied);
        assert_eq!(result.tokens_before, result.tokens_after);
        assert_eq!(payload, original);
    }
}
