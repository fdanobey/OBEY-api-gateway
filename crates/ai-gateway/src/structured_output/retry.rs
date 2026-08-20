use crate::models::openai::{Message, OpenAIRequest};
use crate::structured_output::validator::SchemaViolation;
use serde_json::Value;

const MAX_ASSISTANT_CHARS: usize = 2_000;
const OVERFLOW_ASSISTANT_CHARS: usize = 500;
const MAX_USER_CHARS: usize = 4_000;
const MAX_SCHEMA_CHARS: usize = 3_500;
const MAX_ERRORS: usize = 5;
const MAX_ERROR_DESCRIPTION_CHARS: usize = 200;
const SCHEMA_TRUNCATION_NOTICE: &str =
    "\n[SCHEMA TRUNCATED AT A COMPLETE JSON BOUNDARY TO FIT THE CORRECTIVE PROMPT]";
const JSON_ONLY_INSTRUCTION: &str = "Output ONLY valid JSON matching the required schema. Do not include any other text, markdown, code fences, or explanation.";
const JSON_ONLY_INSTRUCTION_WITHOUT_SCHEMA: &str = "Output ONLY valid JSON matching the originally required schema. Do not include any other text, markdown, code fences, or explanation.";

/// Clones an OpenAI request and appends the corrective assistant/user pair.
///
/// The requested model and all pass-through fields stay unchanged. Temperature
/// is always replaced with the effective retry value. Set
/// `original_was_streaming` when retrying a response gathered from a streaming
/// request so the corrective retry is forced to use a buffered response.
#[allow(clippy::too_many_arguments)]
pub fn build_retry_request(
    original: &OpenAIRequest,
    schema: &Value,
    schema_char_len: usize,
    errors: &[SchemaViolation],
    previous_output: &str,
    effective_retry_temperature: f32,
    original_was_streaming: bool,
    context_window_token_limit: usize,
    current_original_token_estimate: usize,
) -> OpenAIRequest {
    let messages = build_corrective_messages_with_context(
        schema,
        schema_char_len,
        errors,
        previous_output,
        context_window_token_limit,
        current_original_token_estimate,
    );

    build_retry_request_with_messages(
        original,
        messages,
        effective_retry_temperature,
        original_was_streaming,
    )
}

/// Builds a retry request using an already-computed context-overflow decision.
#[allow(clippy::too_many_arguments)]
pub fn build_retry_request_for_overflow(
    original: &OpenAIRequest,
    schema: &Value,
    schema_char_len: usize,
    errors: &[SchemaViolation],
    previous_output: &str,
    effective_retry_temperature: f32,
    original_was_streaming: bool,
    context_overflow: bool,
) -> OpenAIRequest {
    let messages = build_corrective_messages_for_overflow(
        schema,
        schema_char_len,
        errors,
        previous_output,
        context_overflow,
    );

    build_retry_request_with_messages(
        original,
        messages,
        effective_retry_temperature,
        original_was_streaming,
    )
}

/// Builds the assistant/user message pair appended for a corrective retry.
///
/// `schema_char_len` is the serialized schema character count captured during
/// schema extraction. The serialized value is checked again so a stale hint
/// cannot bypass prompt limits.
pub fn build_corrective_messages(
    schema: &Value,
    schema_char_len: usize,
    errors: &[SchemaViolation],
    previous_output: &str,
) -> (Message, Message) {
    build_messages(schema, schema_char_len, errors, previous_output, false)
}

/// Builds corrective messages and applies the context-overflow fallback when
/// the current request plus the normal corrective pair would exceed the model
/// context window.
pub fn build_corrective_messages_with_context(
    schema: &Value,
    schema_char_len: usize,
    errors: &[SchemaViolation],
    previous_output: &str,
    context_window_token_limit: usize,
    current_original_token_estimate: usize,
) -> (Message, Message) {
    let normal = build_corrective_messages(schema, schema_char_len, errors, previous_output);
    let corrective_token_estimate = message_text(&normal.0)
        .map(estimate_tokens)
        .unwrap_or_default()
        .saturating_add(
            message_text(&normal.1)
                .map(estimate_tokens)
                .unwrap_or_default(),
        );
    let overflows = current_original_token_estimate
        .saturating_add(corrective_token_estimate)
        .saturating_add(1)
        > context_window_token_limit;

    build_corrective_messages_for_overflow(
        schema,
        schema_char_len,
        errors,
        previous_output,
        overflows,
    )
}

/// Builds corrective messages using an already-computed context-overflow decision.
pub fn build_corrective_messages_for_overflow(
    schema: &Value,
    schema_char_len: usize,
    errors: &[SchemaViolation],
    previous_output: &str,
    context_overflow: bool,
) -> (Message, Message) {
    if context_overflow {
        build_messages(schema, schema_char_len, errors, previous_output, true)
    } else {
        build_corrective_messages(schema, schema_char_len, errors, previous_output)
    }
}

/// Estimates tokens using the documented four-characters-per-token heuristic.
pub fn estimate_tokens(text: &str) -> usize {
    text.chars().count().saturating_add(3) / 4
}

fn build_retry_request_with_messages(
    original: &OpenAIRequest,
    messages: (Message, Message),
    effective_retry_temperature: f32,
    original_was_streaming: bool,
) -> OpenAIRequest {
    let mut retry = original.clone();
    retry.messages.extend([messages.0, messages.1]);
    retry.temperature = Some(effective_retry_temperature);
    if original_was_streaming {
        retry.stream = false;
    }
    retry
}

fn build_messages(
    schema: &Value,
    schema_char_len: usize,
    errors: &[SchemaViolation],
    previous_output: &str,
    context_overflow: bool,
) -> (Message, Message) {
    let assistant_limit = if context_overflow {
        OVERFLOW_ASSISTANT_CHARS
    } else {
        MAX_ASSISTANT_CHARS
    };
    let assistant = message(
        "assistant",
        truncate_chars(previous_output, assistant_limit),
    );
    let correction = if context_overflow {
        build_overflow_correction(errors)
    } else {
        build_standard_correction(schema, schema_char_len, errors)
    };
    let user = message("user", correction);

    (assistant, user)
}

fn build_standard_correction(
    schema: &Value,
    schema_char_len: usize,
    errors: &[SchemaViolation],
) -> String {
    let serialized_schema = serde_json::to_string(schema).unwrap_or_else(|_| "{}".to_owned());
    let schema_is_oversized =
        schema_char_len > MAX_SCHEMA_CHARS || serialized_schema.chars().count() > MAX_SCHEMA_CHARS;
    let schema_text = if schema_is_oversized {
        truncate_schema_at_json_boundary(schema)
    } else {
        serialized_schema
    };

    let mut correction = format_standard_correction(
        &schema_text,
        errors,
        !schema_is_oversized,
        JSON_ONLY_INSTRUCTION,
    );

    if correction.chars().count() > MAX_USER_CHARS {
        correction = format_standard_correction(&schema_text, errors, false, JSON_ONLY_INSTRUCTION);
    }

    debug_assert!(correction.chars().count() <= MAX_USER_CHARS);
    correction
}

fn format_standard_correction(
    schema_text: &str,
    errors: &[SchemaViolation],
    include_error_details: bool,
    instruction: &str,
) -> String {
    let mut correction = String::from(
        "Your previous output was not valid JSON conforming to the required schema.\n\nSchema:\n```json\n",
    );
    correction.push_str(schema_text);
    correction.push_str("\n```\n\n");
    append_errors(&mut correction, errors, include_error_details);
    correction.push_str("\n\n");
    correction.push_str(instruction);
    correction
}

fn build_overflow_correction(errors: &[SchemaViolation]) -> String {
    let mut correction = String::from(
        "Your previous output was not valid JSON conforming to the required schema.\n\nThe schema definition was omitted because the retry would exceed the model context window.\n\n",
    );
    append_errors(&mut correction, errors, true);
    correction.push_str("\n\n");
    correction.push_str(JSON_ONLY_INSTRUCTION_WITHOUT_SCHEMA);

    debug_assert!(correction.chars().count() <= MAX_USER_CHARS);
    correction
}

fn append_errors(correction: &mut String, errors: &[SchemaViolation], include_details: bool) {
    let shown = errors.len().min(MAX_ERRORS);
    if include_details {
        correction.push_str(&format!(
            "Validation errors (showing {shown} of {}):",
            errors.len()
        ));
        for (index, error) in errors.iter().take(MAX_ERRORS).enumerate() {
            correction.push_str(&format!(
                "\n{}. {}",
                index + 1,
                render_error_description(error)
            ));
        }
    } else {
        correction.push_str(&format!(
            "Validation errors: {} reported; individual details omitted to fit the corrective prompt.",
            errors.len()
        ));
    }
}

fn render_error_description(error: &SchemaViolation) -> String {
    let path = if error.path.is_empty() {
        "$"
    } else {
        error.path.as_str()
    };
    let description = format!(
        "{}: expected {}, got {}",
        sanitize(path),
        sanitize(&error.expected),
        sanitize(&error.actual)
    );
    truncate_with_ellipsis(&description, MAX_ERROR_DESCRIPTION_CHARS)
}

fn truncate_schema_at_json_boundary(schema: &Value) -> String {
    let notice_chars = SCHEMA_TRUNCATION_NOTICE.chars().count();
    let json_budget = MAX_SCHEMA_CHARS.saturating_sub(notice_chars);
    let mut bounded = schema.clone();
    let mut serialized = serde_json::to_string(&bounded).unwrap_or_else(|_| "{}".to_owned());

    while serialized.chars().count() > json_budget {
        if !prune_last_complete_value(&mut bounded) {
            bounded = Value::Object(serde_json::Map::new());
        }
        serialized = serde_json::to_string(&bounded).unwrap_or_else(|_| "{}".to_owned());
        if bounded.as_object().is_some_and(serde_json::Map::is_empty) {
            break;
        }
    }

    if serialized.chars().count() > json_budget {
        serialized = "{}".to_owned();
    }
    serialized.push_str(SCHEMA_TRUNCATION_NOTICE);
    serialized
}

fn prune_last_complete_value(value: &mut Value) -> bool {
    match value {
        Value::Object(object) => {
            let Some(key) = object.keys().next_back().cloned() else {
                return false;
            };
            let pruned_child = object.get_mut(&key).is_some_and(prune_last_complete_value);
            if !pruned_child {
                object.remove(&key);
            }
            true
        }
        Value::Array(array) => {
            let Some(last) = array.last_mut() else {
                return false;
            };
            if !prune_last_complete_value(last) {
                array.pop();
            }
            true
        }
        _ => false,
    }
}

fn message(role: &str, content: String) -> Message {
    Message {
        role: role.to_owned(),
        content: Value::String(content),
        extra: serde_json::Map::new(),
    }
}

fn message_text(message: &Message) -> Option<&str> {
    message.content.as_str()
}

fn sanitize(value: &str) -> String {
    value
        .chars()
        .map(|character| {
            if character.is_control() {
                ' '
            } else {
                character
            }
        })
        .collect()
}

fn truncate_chars(value: &str, max_chars: usize) -> String {
    value.chars().take(max_chars).collect()
}

fn truncate_with_ellipsis(value: &str, max_chars: usize) -> String {
    if value.chars().count() <= max_chars {
        return value.to_owned();
    }
    if max_chars == 0 {
        return String::new();
    }

    let mut truncated = truncate_chars(value, max_chars - 1);
    truncated.push('…');
    truncated
}

#[cfg(test)]
mod tests {
    use super::*;
    use proptest::prelude::*;
    use serde_json::json;

    fn violation(index: usize, description_chars: usize) -> SchemaViolation {
        SchemaViolation {
            path: format!("/items/{index}"),
            expected: "required string ".repeat(description_chars),
            actual: "wrong value\n".repeat(description_chars),
        }
    }

    fn content(message: &Message) -> &str {
        message.content.as_str().expect("message content is text")
    }

    fn original_request(stream: bool, temperature: Option<f32>) -> OpenAIRequest {
        OpenAIRequest {
            model: "requested-model-group".to_owned(),
            messages: vec![Message {
                role: "system".to_owned(),
                content: json!("Keep this original message"),
                extra: serde_json::Map::from_iter([("name".to_owned(), json!("policy"))]),
            }],
            stream,
            temperature,
            max_tokens: Some(321),
            extra: serde_json::Map::from_iter([
                ("provider".to_owned(), json!("selected-provider")),
                ("top_p".to_owned(), json!(0.75)),
                ("tools".to_owned(), json!([{"type": "function"}])),
                (
                    "response_format".to_owned(),
                    json!({"type": "json_schema", "json_schema": {"strict": true}}),
                ),
            ]),
        }
    }

    fn unicode_scalar_strategy() -> impl Strategy<Value = char> {
        prop_oneof![
        4 => Just('a'),
        2 => Just('é'),
        2 => Just('界'),
        2 => Just('🙂'),
        1 => Just('🦀'),
        ]
    }

    fn unicode_pattern_strategy(max_chars: usize) -> impl Strategy<Value = String> {
        (
            unicode_scalar_strategy(),
            unicode_scalar_strategy(),
            0_usize..=max_chars,
        )
            .prop_map(|(first, second, character_count)| {
                [first, second]
                    .into_iter()
                    .cycle()
                    .take(character_count)
                    .collect()
            })
    }

    fn overflow_previous_output_strategy() -> impl Strategy<Value = String> {
        (
            unicode_scalar_strategy(),
            unicode_scalar_strategy(),
            501_usize..=2_600,
        )
            .prop_map(|(first, second, character_count)| {
                [first, second]
                    .into_iter()
                    .cycle()
                    .take(character_count)
                    .collect()
            })
    }

    fn schema_violation_strategy(
        max_component_chars: usize,
    ) -> impl Strategy<Value = SchemaViolation> {
        (
            "[a-z][a-z0-9_]{0,10}",
            unicode_pattern_strategy(max_component_chars),
            unicode_pattern_strategy(max_component_chars),
        )
            .prop_map(|(path, expected, actual)| SchemaViolation {
                path: format!("/{path}"),
                expected,
                actual,
            })
    }

    fn bounded_schema_strategy() -> impl Strategy<Value = Value> {
        (
            "[a-z][a-z0-9_]{0,8}",
            prop::collection::vec(unicode_pattern_strategy(72), 0..=8),
            any::<bool>(),
        )
            .prop_map(|(prefix, descriptions, additional_properties)| {
                let required = (0..descriptions.len())
                    .map(|index| format!("{prefix}_{index}"))
                    .collect::<Vec<_>>();
                let properties = descriptions
                    .into_iter()
                    .enumerate()
                    .map(|(index, description)| {
                        (
                            format!("{prefix}_{index}"),
                            json!({"type": "string", "description": description}),
                        )
                    })
                    .collect::<serde_json::Map<_, _>>();
                json!({
                "type": "object",
                "properties": properties,
                "required": required,
                "additionalProperties": additional_properties,
                })
            })
    }

    fn oversized_schema_strategy() -> impl Strategy<Value = Value> {
        (
            "[a-z][a-z0-9_]{0,8}",
            80_usize..=180,
            unicode_scalar_strategy(),
            24_usize..=80,
        )
            .prop_map(
                |(prefix, property_count, description_character, description_chars)| {
                    let required = (0..property_count)
                        .map(|index| format!("{prefix}_{index:03}"))
                        .collect::<Vec<_>>();
                    let properties = required
.iter()
.cloned()
.map(|name| {
(
name,
json!({
"type": "string",
"description": description_character.to_string().repeat(description_chars),
}),
)
})
.collect::<serde_json::Map<_, _>>();
                    json!({
                    "type": "object",
                    "properties": properties,
                    "required": required,
                    "additionalProperties": false,
                    })
                },
            )
    }

    fn number_violation_paths(errors: &mut [SchemaViolation]) {
        for (index, error) in errors.iter_mut().enumerate() {
            error.path = format!("/generated_error_{index}{}", error.path);
        }
    }

    fn displayed_error_descriptions(user_text: &str) -> Vec<&str> {
        let mut lines = user_text
            .lines()
            .skip_while(|line| !line.starts_with("Validation errors (showing "));
        if lines.next().is_none() {
            return Vec::new();
        }

        lines
            .take_while(|line| !line.is_empty())
            .filter_map(|line| {
                let (number, description) = line.split_once(". ")?;
                number.parse::<usize>().ok()?;
                Some(description)
            })
            .collect()
    }

    fn corrective_schema_block(user_text: &str) -> Option<&str> {
        user_text
            .split_once("Schema:\n```json\n")?
            .1
            .split_once("\n```")
            .map(|(schema, _)| schema)
    }

    // Feature: structured-output-validation, Property 6: Corrective Prompt Structure Invariants
    proptest! {
        #![proptest_config(ProptestConfig::with_cases(64))]

        #[test]
        fn prop_corrective_prompt_structure_invariants(
        schema in bounded_schema_strategy(),
        mut errors in prop::collection::vec(schema_violation_strategy(260), 0..=10),
        previous_output in unicode_pattern_strategy(2_600),
        ) {
        number_violation_paths(&mut errors);
        let original = original_request(false, None);
        let original_message_count = original.messages.len();
        let schema_chars = serde_json::to_string(&schema).unwrap().chars().count();
        let retry = build_retry_request_for_overflow(
        &original,
        &schema,
        schema_chars,
        &errors,
        &previous_output,
        0.0,
        false,
        false,
        );

        prop_assert_eq!(retry.messages.len(), original_message_count + 2);
        let assistant = &retry.messages[original_message_count];
        let user = &retry.messages[original_message_count + 1];
        let assistant_text = content(assistant);
        let user_text = content(user);
        let expected_assistant = previous_output
        .chars()
        .take(MAX_ASSISTANT_CHARS)
        .collect::<String>();
        let displayed_errors = displayed_error_descriptions(user_text);

        prop_assert_eq!(assistant.role.as_str(), "assistant");
        prop_assert_eq!(user.role.as_str(), "user");
        prop_assert_eq!(assistant_text, expected_assistant.as_str());
        prop_assert!(assistant_text.chars().count() <= MAX_ASSISTANT_CHARS);
        prop_assert!(user_text.chars().count() <= MAX_USER_CHARS);
        prop_assert_eq!(displayed_errors.len(), errors.len().min(MAX_ERRORS));
        prop_assert!(displayed_errors.len() <= MAX_ERRORS);
    prop_assert!(displayed_errors
    .iter()
    .all(|description| description.chars().count() <= MAX_ERROR_DESCRIPTION_CHARS));
        prop_assert!(std::str::from_utf8(assistant_text.as_bytes()).is_ok());
        prop_assert!(std::str::from_utf8(user_text.as_bytes()).is_ok());
        }
        }

    // Feature: structured-output-validation, Property 7: Schema Truncation Under Size Limit
    proptest! {
        #![proptest_config(ProptestConfig::with_cases(64))]

        #[test]
        fn prop_schema_truncation_under_size_limit(
        schema in oversized_schema_strategy(),
        mut errors in prop::collection::vec(schema_violation_strategy(80), 1..=10),
        previous_output in unicode_pattern_strategy(2_200),
        ) {
        number_violation_paths(&mut errors);
        let schema_chars = serde_json::to_string(&schema).unwrap().chars().count();
        prop_assert!(schema_chars > MAX_SCHEMA_CHARS);

        let (_, user) = build_corrective_messages(&schema, schema_chars, &errors, &previous_output);
        let user_text = content(&user);
        let schema_block = corrective_schema_block(user_text)
        .ok_or_else(|| TestCaseError::fail("corrective prompt omitted the schema block"))?;
        let (bounded_json, notice_suffix) = schema_block
        .split_once(SCHEMA_TRUNCATION_NOTICE)
        .ok_or_else(|| TestCaseError::fail("corrective prompt omitted the truncation notice"))?;

        prop_assert!(notice_suffix.is_empty());
        prop_assert_eq!(schema_block.matches(SCHEMA_TRUNCATION_NOTICE).count(), 1);
        prop_assert!(schema_block.ends_with(SCHEMA_TRUNCATION_NOTICE));
        prop_assert!(schema_block.chars().count() <= MAX_SCHEMA_CHARS);
        prop_assert!(serde_json::from_str::<Value>(bounded_json).is_ok());
        prop_assert!(user_text.chars().count() <= MAX_USER_CHARS);
        prop_assert!(displayed_error_descriptions(user_text).is_empty());
    let omitted_details_summary = format!(
    "{} reported; individual details omitted",
    errors.len()
    );
    prop_assert!(user_text.contains(&omitted_details_summary));
        for error in &errors {
        prop_assert!(!user_text.contains(&error.path));
        }
        prop_assert!(std::str::from_utf8(schema_block.as_bytes()).is_ok());
        prop_assert!(std::str::from_utf8(user_text.as_bytes()).is_ok());
        }
        }

    // Feature: structured-output-validation, Property 8: Retry Temperature Override
    proptest! {
        #![proptest_config(ProptestConfig::with_cases(64))]

        #[test]
        fn retry_temperature_override_is_unconditional(
        original_temperature in proptest::option::of(0.0f32..=2.0),
        configured_temperature in 0.0f32..=2.0,
        original_stream in any::<bool>(),
        original_was_streaming in any::<bool>(),
        ) {
        let original = original_request(original_stream, original_temperature);
        let original_snapshot = serde_json::to_value(&original).unwrap();
        let original_messages = original.messages.clone();
        let schema = json!({"type": "string"});

        let retry = build_retry_request_for_overflow(
        &original,
        &schema,
        serde_json::to_string(&schema).unwrap().chars().count(),
        &[],
        "invalid output",
        configured_temperature,
        original_was_streaming,
        false,
        );

        prop_assert_eq!(retry.temperature, Some(configured_temperature));
        prop_assert_eq!(serde_json::to_value(&original).unwrap(), original_snapshot);
        prop_assert_eq!(
    serde_json::to_value(&retry.messages[..original_messages.len()]).unwrap(),
    serde_json::to_value(&original_messages).unwrap()
    );
        prop_assert_eq!(retry.model, original.model);
        prop_assert_eq!(retry.max_tokens, original.max_tokens);
        prop_assert_eq!(retry.extra, original.extra);
        prop_assert_eq!(retry.stream, if original_was_streaming { false } else { original_stream });
        }
        }

    #[test]
    fn retry_request_clones_original_and_appends_exactly_assistant_then_user() {
        let original = original_request(false, Some(1.25));
        let original_snapshot = serde_json::to_value(&original).unwrap();
        let schema = json!({"type": "integer"});
        let errors = vec![violation(0, 1)];

        let retry = build_retry_request(
            &original,
            &schema,
            serde_json::to_string(&schema).unwrap().chars().count(),
            &errors,
            "not an integer",
            0.2,
            false,
            10_000,
            10,
        );

        assert_eq!(serde_json::to_value(&original).unwrap(), original_snapshot);
        assert_eq!(retry.temperature, Some(0.2));
        assert_eq!(retry.messages.len(), original.messages.len() + 2);
        assert_eq!(
            serde_json::to_value(&retry.messages[..original.messages.len()]).unwrap(),
            serde_json::to_value(&original.messages).unwrap()
        );
        assert_eq!(retry.messages[0].role, "system");
        assert_eq!(retry.messages[1].role, "assistant");
        assert_eq!(content(&retry.messages[1]), "not an integer");
        assert_eq!(retry.messages[2].role, "user");
        assert!(content(&retry.messages[2]).contains(JSON_ONLY_INSTRUCTION));
    }

    #[test]
    fn retry_request_replaces_temperature_and_preserves_request_fields() {
        let original = original_request(false, None);
        let schema = json!({"type": "string"});

        let retry = build_retry_request_for_overflow(
            &original,
            &schema,
            serde_json::to_string(&schema).unwrap().chars().count(),
            &[violation(1, 1)],
            "42",
            0.65,
            false,
            false,
        );

        assert_eq!(retry.temperature, Some(0.65));
        assert_eq!(retry.model, original.model);
        assert_eq!(retry.stream, original.stream);
        assert_eq!(retry.max_tokens, original.max_tokens);
        assert_eq!(retry.extra, original.extra);
        assert_eq!(retry.messages[0].extra, original.messages[0].extra);
    }

    #[test]
    fn streaming_retry_is_forced_to_non_streaming() {
        let original = original_request(true, Some(1.0));
        let schema = json!({"type": "object"});

        let retry = build_retry_request_for_overflow(
            &original,
            &schema,
            serde_json::to_string(&schema).unwrap().chars().count(),
            &[],
            "{}",
            0.0,
            true,
            true,
        );

        assert!(original.stream);
        assert!(!retry.stream);
        assert_eq!(retry.temperature, Some(0.0));
        assert_eq!(retry.extra, original.extra);
    }

    #[test]
    fn builds_exactly_assistant_then_user_with_bounded_details() {
        let schema = json!({
            "type": "object",
            "properties": {"name": {"type": "string"}},
            "required": ["name"]
        });
        let previous = format!("{}tail", "🙂".repeat(MAX_ASSISTANT_CHARS));
        let errors = (0..8).map(|index| violation(index, 40)).collect::<Vec<_>>();

        let (assistant, user) = build_corrective_messages(
            &schema,
            serde_json::to_string(&schema).unwrap().chars().count(),
            &errors,
            &previous,
        );

        assert_eq!(assistant.role, "assistant");
        assert_eq!(user.role, "user");
        assert!(assistant.extra.is_empty());
        assert!(user.extra.is_empty());
        assert_eq!(content(&assistant), "🙂".repeat(MAX_ASSISTANT_CHARS));
        assert!(content(&user).contains(&serde_json::to_string(&schema).unwrap()));
        assert!(content(&user).contains("showing 5 of 8"));
        assert!(content(&user).contains(JSON_ONLY_INSTRUCTION));
        assert!(content(&user).chars().count() <= MAX_USER_CHARS);
        assert_eq!(content(&user).matches("\n1. ").count(), 1);
        assert!(!content(&user).contains("/items/5"));

        for error in &errors[..MAX_ERRORS] {
            assert!(render_error_description(error).chars().count() <= 200);
            assert!(!render_error_description(error).contains('\n'));
        }
    }

    #[test]
    fn oversized_schema_is_validly_truncated_and_omits_error_details() {
        let properties = (0..400)
            .map(|index| {
                (
                    format!("property_{index:03}"),
                    json!({"type": "string", "description": "界".repeat(30)}),
                )
            })
            .collect::<serde_json::Map<_, _>>();
        let schema = json!({"type": "object", "properties": properties});
        let serialized_chars = serde_json::to_string(&schema).unwrap().chars().count();
        assert!(serialized_chars > MAX_SCHEMA_CHARS);

        let errors = vec![violation(7, 1)];
        let (_, user) =
            build_corrective_messages(&schema, serialized_chars, &errors, "invalid output");
        let user_text = content(&user);
        let schema_block = user_text
            .split("```json\n")
            .nth(1)
            .unwrap()
            .split("\n```")
            .next()
            .unwrap();
        let (bounded_json, notice) = schema_block
            .split_once(SCHEMA_TRUNCATION_NOTICE)
            .expect("explicit schema truncation notice");

        assert!(notice.is_empty());
        assert!(serde_json::from_str::<Value>(bounded_json).is_ok());
        assert!(schema_block.chars().count() <= MAX_SCHEMA_CHARS);
        assert!(user_text.chars().count() <= MAX_USER_CHARS);
        assert!(user_text.contains("1 reported; individual details omitted"));
        assert!(!user_text.contains("/items/7"));
        assert!(user_text.contains(JSON_ONLY_INSTRUCTION));
    }

    #[test]
    fn user_limit_fallback_omits_details_before_truncating_instruction() {
        let schema = json!({"description": "x".repeat(3_300)});
        let schema_chars = serde_json::to_string(&schema).unwrap().chars().count();
        assert!(schema_chars <= MAX_SCHEMA_CHARS);
        let errors = (0..MAX_ERRORS)
            .map(|index| violation(index, 100))
            .collect::<Vec<_>>();

        let (_, user) = build_corrective_messages(&schema, schema_chars, &errors, "bad");
        let user_text = content(&user);

        assert!(user_text.chars().count() <= MAX_USER_CHARS);
        assert!(user_text.contains("individual details omitted"));
        assert!(!user_text.contains("/items/0"));
        assert!(user_text.ends_with(JSON_ONLY_INSTRUCTION));
    }

    // Feature: structured-output-validation, Property 15: Context Window Truncation
    proptest! {
        #![proptest_config(ProptestConfig::with_cases(64))]

        #[test]
        fn prop_context_window_truncation(
        base_schema in bounded_schema_strategy(),
        mut errors in prop::collection::vec(schema_violation_strategy(260), 1..=10),
        previous_output in overflow_previous_output_strategy(),
        current_original_token_estimate in 0_usize..=16_000,
        ) {
        number_violation_paths(&mut errors);
        let schema_marker = "property_15_schema_marker_must_be_omitted";
        let mut schema = base_schema;
        schema[schema_marker] = json!("context-overflow-only");
        let schema_chars = serde_json::to_string(&schema).unwrap().chars().count();
        let normal = build_corrective_messages(&schema, schema_chars, &errors, &previous_output);
        let normal_corrective_tokens = estimate_tokens(content(&normal.0))
        .saturating_add(estimate_tokens(content(&normal.1)));
        let context_window_token_limit = current_original_token_estimate
        .saturating_add(normal_corrective_tokens);
        let original = original_request(false, Some(1.0));
        let original_message_count = original.messages.len();

        let retry = build_retry_request(
        &original,
        &schema,
        schema_chars,
        &errors,
        &previous_output,
        0.0,
        false,
        context_window_token_limit,
        current_original_token_estimate,
        );
        let assistant = &retry.messages[original_message_count];
        let user = &retry.messages[original_message_count + 1];
        let assistant_text = content(assistant);
        let user_text = content(user);
        let expected_assistant = previous_output
        .chars()
        .take(OVERFLOW_ASSISTANT_CHARS)
        .collect::<String>();
        let displayed_errors = displayed_error_descriptions(user_text);

        prop_assert_eq!(retry.messages.len(), original_message_count + 2);
        prop_assert_eq!(assistant.role.as_str(), "assistant");
        prop_assert_eq!(user.role.as_str(), "user");
        prop_assert_eq!(assistant_text, expected_assistant.as_str());
        prop_assert_eq!(assistant_text.chars().count(), OVERFLOW_ASSISTANT_CHARS);
        prop_assert!(assistant_text.chars().count() <= OVERFLOW_ASSISTANT_CHARS);
        prop_assert!(user_text.chars().count() <= MAX_USER_CHARS);
        prop_assert!(!user_text.contains("Schema:\n```json"));
        prop_assert!(!user_text.contains(schema_marker));
        prop_assert!(user_text.contains("schema definition was omitted"));
        prop_assert!(user_text.ends_with(JSON_ONLY_INSTRUCTION_WITHOUT_SCHEMA));
        prop_assert_eq!(displayed_errors.len(), errors.len().min(MAX_ERRORS));
        prop_assert!(displayed_errors.len() <= MAX_ERRORS);
    prop_assert!(displayed_errors
    .iter()
    .all(|description| description.chars().count() <= MAX_ERROR_DESCRIPTION_CHARS));
        for error in errors.iter().take(MAX_ERRORS) {
        prop_assert!(user_text.contains(&error.path));
        }
        prop_assert!(std::str::from_utf8(assistant_text.as_bytes()).is_ok());
        prop_assert!(std::str::from_utf8(user_text.as_bytes()).is_ok());
        }
        }

    #[test]
    fn context_overflow_uses_500_chars_omits_schema_and_keeps_errors() {
        let schema = json!({"type": "object", "secret_marker": "must-not-appear"});
        let errors = vec![SchemaViolation {
            path: "/answer".to_owned(),
            expected: "integer".to_owned(),
            actual: "text".to_owned(),
        }];
        let previous = "🦀".repeat(900);

        let (assistant, user) = build_corrective_messages_with_context(
            &schema,
            serde_json::to_string(&schema).unwrap().chars().count(),
            &errors,
            &previous,
            100,
            100,
        );
        let user_text = content(&user);

        assert_eq!(content(&assistant), "🦀".repeat(OVERFLOW_ASSISTANT_CHARS));
        assert!(!user_text.contains("secret_marker"));
        assert!(!user_text.contains("Schema:\n```json"));
        assert!(user_text.contains("/answer: expected integer, got text"));
        assert!(user_text.contains("schema definition was omitted"));
        assert!(user_text.ends_with(JSON_ONLY_INSTRUCTION_WITHOUT_SCHEMA));
        assert!(user_text.chars().count() <= MAX_USER_CHARS);
    }

    #[test]
    fn context_that_fits_keeps_normal_prompt() {
        let schema = json!({"type": "integer"});
        let previous = "a".repeat(700);
        let errors = vec![violation(0, 1)];

        let (assistant, user) = build_corrective_messages_with_context(
            &schema,
            serde_json::to_string(&schema).unwrap().chars().count(),
            &errors,
            &previous,
            10_000,
            5,
        );

        assert_eq!(content(&assistant), previous);
        assert!(content(&user).contains("Schema:\n```json"));
    }

    #[test]
    fn context_boundary_accounts_for_appended_message_overhead() {
        let schema = json!({"type": "integer"});
        let previous = "bad";
        let errors = vec![violation(0, 1)];
        let normal = build_corrective_messages(
            &schema,
            serde_json::to_string(&schema).unwrap().chars().count(),
            &errors,
            previous,
        );
        let corrective_tokens =
            estimate_tokens(content(&normal.0)).saturating_add(estimate_tokens(content(&normal.1)));

        let (_, user) = build_corrective_messages_with_context(
            &schema,
            serde_json::to_string(&schema).unwrap().chars().count(),
            &errors,
            previous,
            corrective_tokens,
            0,
        );

        assert!(!content(&user).contains("Schema:\n```json"));
    }

    #[test]
    fn explicit_overflow_decision_uses_same_fallback() {
        let schema = json!({"marker": "omitted"});
        let errors = vec![violation(2, 1)];
        let previous = "é".repeat(600);

        let (assistant, user) = build_corrective_messages_for_overflow(
            &schema,
            serde_json::to_string(&schema).unwrap().chars().count(),
            &errors,
            &previous,
            true,
        );

        assert_eq!(
            content(&assistant).chars().count(),
            OVERFLOW_ASSISTANT_CHARS
        );
        assert!(!content(&user).contains("marker"));
        assert!(content(&user).contains("/items/2"));
        assert!(content(&user).ends_with(JSON_ONLY_INSTRUCTION_WITHOUT_SCHEMA));
    }

    #[test]
    fn token_estimate_rounds_up_by_unicode_characters() {
        assert_eq!(estimate_tokens(""), 0);
        assert_eq!(estimate_tokens("a"), 1);
        assert_eq!(estimate_tokens("abcd"), 1);
        assert_eq!(estimate_tokens("abcde"), 2);
        assert_eq!(estimate_tokens("🙂🙂🙂🙂🙂"), 2);
    }

    #[test]
    fn all_truncation_is_utf8_safe() {
        let schema = json!({
            "type": "object",
            "properties": (0..500)
                .map(|index| (format!("鍵{index}"), json!({"description": "🙂".repeat(20)})))
                .collect::<serde_json::Map<_, _>>()
        });
        let errors = vec![SchemaViolation {
            path: format!("/{}", "界".repeat(300)),
            expected: "🙂".repeat(300),
            actual: "é".repeat(300),
        }];

        let (assistant, user) = build_corrective_messages(
            &schema,
            serde_json::to_string(&schema).unwrap().chars().count(),
            &errors,
            &"🦀".repeat(3_000),
        );

        assert_eq!(content(&assistant).chars().count(), MAX_ASSISTANT_CHARS);
        assert!(content(&user).chars().count() <= MAX_USER_CHARS);
        assert!(render_error_description(&errors[0]).chars().count() <= 200);
        assert!(std::str::from_utf8(content(&assistant).as_bytes()).is_ok());
        assert!(std::str::from_utf8(content(&user).as_bytes()).is_ok());
    }
}
