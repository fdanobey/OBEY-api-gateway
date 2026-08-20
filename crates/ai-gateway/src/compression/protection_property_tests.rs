use super::config::{CompressionConfig, EffectiveCompressionConfig, TimeBudgetConfig};
use super::pipeline::{CompressionPipeline, CompressionRequestMetadata};
use super::{CompressiblePayload, CompressionContext, CompressionLevel};
use crate::models::openai::{Message, OpenAIRequest};
use proptest::prelude::*;
use proptest::test_runner::TestCaseResult;
use serde_json::{json, Map, Value};

const NAMED_LEVELS: [CompressionLevel; 6] = [
    CompressionLevel::Lite,
    CompressionLevel::Standard,
    CompressionLevel::Aggressive,
    CompressionLevel::Ultra,
    CompressionLevel::Rtk,
    CompressionLevel::Stacked,
];

#[derive(Debug)]
struct GeneratedCase {
    case_id: u64,
    words: Vec<String>,
    fence_lines: Vec<String>,
    tilde_lines: Vec<String>,
    path_segments: Vec<String>,
    identifiers: Vec<String>,
}

fn lower_word() -> impl Strategy<Value = String> {
    proptest::string::string_regex("[a-z]{3,12}").expect("word regex must compile")
}

fn code_line() -> impl Strategy<Value = String> {
    proptest::string::string_regex("[a-z][a-z0-9_]{2,10}")
        .expect("code-line regex must compile")
        .prop_map(|name| format!("let {name} = {name}.len();"))
}

fn generated_case() -> impl Strategy<Value = GeneratedCase> {
    (
        any::<u64>(),
        prop::collection::vec(lower_word(), 24..40),
        prop::collection::vec(code_line(), 2..12),
        prop::collection::vec(code_line(), 2..12),
        prop::collection::vec(lower_word(), 3..7),
        prop::collection::vec(lower_word(), 4..8),
    )
        .prop_map(
            |(case_id, words, fence_lines, tilde_lines, path_segments, identifiers)| {
                GeneratedCase {
                    case_id,
                    words,
                    fence_lines,
                    tilde_lines,
                    path_segments,
                    identifiers,
                }
            },
        )
}

fn prose(case: &GeneratedCase, offset: usize) -> String {
    let selected = (0..10)
        .map(|index| case.words[(offset + index) % case.words.len()].as_str())
        .collect::<Vec<_>>()
        .join(" ");
    format!(
        "{selected}. This generated explanatory wrapper contains ordinary prose that compression may safely shorten while preserving every embedded technical region exactly. The surrounding discussion repeats context for property case {} and remains intentionally verbose.",
        case.case_id
    )
}

fn pascal(words: &[String]) -> String {
    words
        .iter()
        .map(|word| {
            let mut characters = word.chars();
            match characters.next() {
                Some(first) => first.to_ascii_uppercase().to_string() + characters.as_str(),
                None => String::new(),
            }
        })
        .collect()
}

fn protected_regions(case: &GeneratedCase) -> Vec<String> {
    let suffix = case.case_id;
    let backtick_code = format!(
        "```rust\n// property case {suffix}\n{}\n```",
        case.fence_lines.join("\n")
    );
    let tilde_code = format!(
        "~~~~python\n# property case {suffix}\n{}\n~~~~",
        case.tilde_lines.join("\n")
    );
    let url = format!(
        "https://example.test/property/{suffix}/artifact?mode=exact&attempt={}",
        suffix % 17
    );
    let unix_path = format!(
        "/var/{}/property-{suffix}/{}/result.json",
        case.path_segments[0], case.path_segments[1]
    );
    let windows_path = format!(
        r"C:\{}\property-{suffix}\{}\result.log",
        case.path_segments[1], case.path_segments[2]
    );
    let nested_json = json!({
        "case": suffix.to_string(),
        "metadata": {
            "enabled": true,
            "labels": [case.words[0].clone(), case.words[1].clone()],
            "limits": {"minimum": suffix % 11, "maximum": suffix % 11 + 100}
        },
        "items": [
            {"name": case.words[2].clone(), "values": [1, 2, 3]},
            {"name": case.words[3].clone(), "values": [4, 5, 6]}
        ]
    })
    .to_string();
    let camel = format!(
        "{}{}{}",
        case.identifiers[0],
        pascal(&case.identifiers[1..2]),
        suffix % 10_000
    );
    let snake = format!(
        "{}_{}_{}",
        case.identifiers[1],
        case.identifiers[2],
        suffix % 10_000
    );
    let pascal_identifier = format!(
        "{}{}{}",
        pascal(&case.identifiers[0..1]),
        pascal(&case.identifiers[2..3]),
        suffix % 10_000
    );
    let screaming = format!(
        "{}_{}_{}",
        case.identifiers[2].to_ascii_uppercase(),
        case.identifiers[3].to_ascii_uppercase(),
        suffix % 10_000
    );
    let math = format!(
        r"$$\sum_{{i=1}}^{{{}}} x_i^2 = \frac{{{suffix}}}{{n+1}}$$",
        suffix % 19 + 2
    );
    let function_definition = format!(
        "pub async fn {}_{}(request_id: &str, retry_count: usize) -> Result<(), GatewayError>",
        case.identifiers[0], suffix
    );
    let tool_definition = format!(
        "call_tool {}_{}(path: String, recursive: bool)",
        case.identifiers[1], suffix
    );
    let structured_tool_json = format!(
        "\"tool_call\":{}",
        json!({
            "id": format!("call-{suffix}"),
            "type": "function",
            "function": {
                "name": format!("inspect_{}_{}", case.identifiers[2], suffix),
                "arguments": {
                    "path": unix_path.clone(),
                    "recursive": false,
                    "filters": [case.words[4].clone(), case.words[5].clone()]
                }
            }
        })
    );

    vec![
        backtick_code,
        tilde_code,
        url,
        unix_path,
        windows_path,
        nested_json,
        camel,
        snake,
        pascal_identifier,
        screaming,
        math,
        function_definition,
        tool_definition,
        structured_tool_json,
    ]
}

fn tool_definitions(case: &GeneratedCase) -> Value {
    json!([{
        "type": "function",
        "function": {
            "name": format!("inspect_{}_{}", case.identifiers[2], case.case_id),
            "description": "Inspect an exact generated path and return structured metadata for the caller.",
            "parameters": {
                "type": "object",
                "properties": {
                    "path": {"type": "string"},
                    "recursive": {"type": "boolean"},
                    "filters": {"type": "array", "items": {"type": "string"}}
                },
                "required": ["path"],
                "additionalProperties": false
            }
        }
    }])
}

fn request(case: &GeneratedCase, regions: &[String]) -> OpenAIRequest {
    let body = regions
        .iter()
        .enumerate()
        .map(|(index, region)| {
            format!(
                "{}\nProtected region {index} begins on the next line:\n{region}\n{}",
                prose(case, index),
                prose(case, index + 7)
            )
        })
        .collect::<Vec<_>>()
        .join("\n\n");
    let tools = tool_definitions(case);
    let call_id = format!("structured-call-{}", case.case_id);
    let mut message_extra = Map::new();
    message_extra.insert(
        "tool_calls".to_owned(),
        json!([{
            "id": call_id,
            "type": "function",
            "function": {
                "name": format!("inspect_{}_{}", case.identifiers[2], case.case_id),
                "arguments": json!({
                    "path": format!("/var/property/{}/input.json", case.case_id),
                    "recursive": false,
                    "filters": [case.words[0].clone(), case.words[1].clone()]
                }).to_string()
            }
        }]),
    );

    let mut extra = Map::new();
    extra.insert("tools".to_owned(), tools);
    extra.insert("tool_choice".to_owned(), Value::String("auto".to_owned()));

    OpenAIRequest {
        model: "gpt-4o".to_owned(),
        messages: vec![
            Message {
                role: "assistant".to_owned(),
                content: Value::String(
                    "I will preserve the generated technical regions while processing the request."
                        .to_owned(),
                ),
                extra: message_extra,
            },
            Message {
                role: "user".to_owned(),
                content: Value::String(body),
                extra: Map::new(),
            },
        ],
        stream: false,
        temperature: None,
        max_tokens: Some(1024),
        extra,
    }
}

fn pipeline() -> CompressionPipeline {
    CompressionPipeline::from_config(CompressionConfig {
        enabled: true,
        compress_tool_definitions: false,
        time_budget_ms: TimeBudgetConfig {
            lite: 60_000,
            standard: 60_000,
            aggressive: 60_000,
            ultra: 60_000,
            rtk: 60_000,
            stacked: 60_000,
        },
        ..CompressionConfig::default()
    })
}

fn effective(level: CompressionLevel) -> EffectiveCompressionConfig {
    EffectiveCompressionConfig {
        enabled: true,
        level,
        auto_threshold_tokens: u32::MAX,
        caveman_output: false,
    }
}

fn assert_protection(case: GeneratedCase) -> TestCaseResult {
    let regions = protected_regions(&case);
    let request = request(&case, &regions);
    let original_tools = request.extra.get("tools").cloned();
    let original_tool_calls = request.messages[0].extra.get("tool_calls").cloned();
    let original_payload = CompressiblePayload::from_openai_request(request.clone());
    let counter = CompressionContext::new(&request.model, "property-test").token_counter;
    let original_tokens = counter.count_request(&request);
    let pipeline = pipeline();
    let runtime = tokio::runtime::Builder::new_current_thread()
        .enable_time()
        .build()
        .expect("property-test runtime must build");

    for level in NAMED_LEVELS {
        let context = CompressionContext {
            context_window: u32::MAX,
            target_token_budget: None,
            ..CompressionContext::new(&request.model, "property-test")
        };
        let result = runtime.block_on(pipeline.compress_explicit(
            original_payload.clone(),
            context,
            effective(level),
            CompressionRequestMetadata {
                request_id: format!("protection-{}-{level:?}", case.case_id),
                ..CompressionRequestMetadata::default()
            },
        ));
        let output_request = result.payload.clone().into_openai_request();
        let output_text = output_request
            .messages
            .iter()
            .filter_map(|message| message.content.as_str())
            .collect::<Vec<_>>()
            .join("\n");
        let output_tokens = counter.count_request(&output_request);

        prop_assert!(
            !result.timed_out,
            "{level:?} exceeded the generous test budget"
        );
        prop_assert!(
            !result.error,
            "{level:?} returned errors: {:?}",
            result.errors
        );
        prop_assert_eq!(result.original_tokens, original_tokens);
        prop_assert_eq!(result.final_tokens, output_tokens);
        prop_assert!(
            output_tokens <= original_tokens,
            "{level:?} increased tokens from {original_tokens} to {output_tokens}"
        );
        for region in &regions {
            prop_assert!(
                output_text
                    .as_bytes()
                    .windows(region.len())
                    .any(|window| window == region.as_bytes()),
                "{level:?} changed or removed protected bytes: {region:?}"
            );
        }
        prop_assert_eq!(output_request.extra.get("tools"), original_tools.as_ref());
        prop_assert_eq!(
            output_request.messages[0].extra.get("tool_calls"),
            original_tool_calls.as_ref()
        );
    }

    Ok(())
}

proptest! {
    #![proptest_config(ProptestConfig::with_cases(64))]

    #[test]
    fn property_1_protected_regions_remain_byte_identical(case in generated_case()) {
        assert_protection(case)?;
    }
}
