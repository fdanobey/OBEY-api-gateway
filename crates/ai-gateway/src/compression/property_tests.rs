use super::engines::lite::LiteEngine;
use super::{CompressiblePayload, CompressionContext, CompressionEngine};
use crate::models::openai::{Message, OpenAIRequest};
use proptest::prelude::*;
use proptest::test_runner::TestCaseResult;
use serde_json::{Map, Value};

fn safe_prose_fragment() -> impl Strategy<Value = String> {
    prop::collection::vec(
        any::<char>().prop_filter(
            "non-whitespace prose without protection syntax",
            |character| {
                !character.is_control()
                    && !character.is_whitespace()
                    && (!character.is_ascii()
                        || character.is_ascii_lowercase()
                        || character.is_ascii_digit()
                        || matches!(character, ',' | '.' | '!' | '?' | ';' | '-'))
            },
        ),
        1..32,
    )
    .prop_map(|characters| characters.into_iter().collect())
}

fn whitespace_heavy_prose() -> impl Strategy<Value = String> {
    (
        safe_prose_fragment(),
        safe_prose_fragment(),
        safe_prose_fragment(),
        safe_prose_fragment(),
        safe_prose_fragment(),
        safe_prose_fragment(),
        safe_prose_fragment(),
    )
        .prop_map(|(first, second, third, fourth, fifth, sixth, seventh)| {
            format!(
                "{first}  \t {second}\r\n{third}\r\n\r\n\r\n{fourth}\t\t{fifth}\r{sixth}\n\n\n\n{seventh}"
            )
        })
}

fn request_with_content(content: String) -> OpenAIRequest {
    OpenAIRequest {
        model: "gpt-4o".to_owned(),
        messages: vec![Message {
            role: "user".to_owned(),
            content: Value::String(content),
            extra: Map::new(),
        }],
        stream: false,
        temperature: None,
        max_tokens: None,
        extra: Map::new(),
    }
}

fn assert_normalized_unprotected_output(
    context: &CompressionContext,
    output: &str,
) -> TestCaseResult {
    let unprotected_ranges = context.protection_scanner.unprotected_ranges(output);
    prop_assert_eq!(unprotected_ranges.as_slice(), &[0..output.len()]);

    for range in unprotected_ranges {
        let segment = &output[range];
        let has_adjacent_horizontal_whitespace = segment
            .as_bytes()
            .windows(2)
            .any(|pair| matches!(pair[0], b' ' | b'\t') && matches!(pair[1], b' ' | b'\t'));
        prop_assert!(!has_adjacent_horizontal_whitespace);
        prop_assert!(!segment.contains('\r'));
        prop_assert!(!segment.contains("\n\n\n"));
    }

    Ok(())
}

proptest! {
    #![proptest_config(ProptestConfig::with_cases(64))]

    #[test]
    fn property_10_whitespace_normalization(input in whitespace_heavy_prose()) {
        let context = CompressionContext::new("gpt-4o", "property-test");
        prop_assert!(context.protection_scanner.scan(&input).is_empty());

        let request = request_with_content(input.clone());
        let input_tokens = context.token_counter.count_request(&request);
        let mut payload = CompressiblePayload::from_openai_request(request);
        let runtime = tokio::runtime::Builder::new_current_thread()
            .build()
            .expect("property-test runtime must build");
        let result = runtime.block_on(LiteEngine::new().compress(&mut payload, &context));
        let output = payload.messages[0]
            .content
            .as_text()
            .expect("string request content must remain string")
            .to_owned();
        let output_tokens = context
            .token_counter
            .count_request(&payload.clone().into_openai_request());

        prop_assert!(result.applied);
        prop_assert_ne!(output.as_str(), input.as_str());
        prop_assert_eq!(result.tokens_before, input_tokens);
        prop_assert_eq!(result.tokens_after, output_tokens);
        prop_assert!(output_tokens <= input_tokens);
        assert_normalized_unprotected_output(&context, &output)?;
    }
}
