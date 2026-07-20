use super::config::{
    CompressionConfig, CustomPipelineConfig, EffectiveCompressionConfig, TimeBudgetConfig,
};
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
const CUSTOM_PIPELINE: &str = "mixed_monotonic";

#[derive(Debug)]
struct GeneratedCase {
    case_id: u64,
    words: Vec<String>,
    history_turns: usize,
    command_lines: usize,
    protected_kind: u8,
    structured_user: bool,
    structured_assistant: bool,
    include_tools: bool,
    cache_marker: bool,
    prompt_caching: bool,
    temperature: Option<f32>,
}

fn word() -> impl Strategy<Value = String> {
    proptest::string::string_regex("[a-z]{2,12}").expect("word regex must compile")
}

fn generated_case() -> impl Strategy<Value = GeneratedCase> {
    (
        any::<u64>(),
        prop::collection::vec(word(), 32),
        1usize..4,
        12usize..28,
        any::<u8>(),
        any::<bool>(),
        any::<bool>(),
        any::<bool>(),
        any::<bool>(),
        any::<bool>(),
        prop_oneof![
            Just(None),
            (0u8..=20).prop_map(|value| Some(f32::from(value) / 10.0)),
        ],
    )
        .prop_map(
            |(
                case_id,
                words,
                history_turns,
                command_lines,
                protected_kind,
                structured_user,
                structured_assistant,
                include_tools,
                cache_marker,
                prompt_caching,
                temperature,
            )| {
                let (structured_user, structured_assistant) =
                    if structured_user || structured_assistant {
                        (structured_user, structured_assistant)
                    } else {
                        (true, false)
                    };
                GeneratedCase {
                    case_id,
                    words,
                    history_turns,
                    command_lines,
                    protected_kind,
                    structured_user,
                    structured_assistant,
                    include_tools,
                    cache_marker,
                    prompt_caching,
                    temperature,
                }
            },
        )
}

fn message(role: &str, content: Value, extra: Map<String, Value>) -> Message {
    Message {
        role: role.to_owned(),
        content,
        extra,
    }
}

fn plain_message(role: &str, content: String) -> Message {
    message(role, Value::String(content), Map::new())
}

fn protected_region(kind: u8, case_id: u64) -> String {
    match kind % 6 {
        0 => format!("```rust\nlet case_id = {case_id};\nprintln!(\"{{case_id}}\");\n```"),
        1 => format!("https://example.test/cases/{case_id}?mode=property"),
        2 => json!({"case": case_id.to_string(), "enabled": true, "items": [1, 2, 3]}).to_string(),
        3 => format!("/var/tmp/property/{case_id}/output.json"),
        4 => format!(r"C:\property\case-{case_id}\output.log"),
        _ => format!("$x_{{{}}} = y^2 + 3$", case_id % 97),
    }
}

fn prose(words: &[String], offset: usize, label: &str, case_id: u64) -> String {
    format!(
        "{} {} {label} for generated case {case_id}. This is ordinary repeated explanatory prose that can be safely shortened, normalized, summarized, or retained without changing the request structure. {} {} provide additional context for the expected answer.",
        words[offset % words.len()],
        words[(offset + 1) % words.len()],
        words[(offset + 2) % words.len()],
        words[(offset + 3) % words.len()],
    )
}

fn content(text: String, structured: bool, case_id: u64) -> Value {
    if structured {
        json!([
            {"type": "text", "text": text},
            {
                "type": "image_url",
                "image_url": {"url": format!("https://example.test/images/{case_id}.png")}
            }
        ])
    } else {
        Value::String(text)
    }
}

fn command_output(case: &GeneratedCase) -> String {
    let mut lines = vec![
        format!("Compiling property_case_{} v0.1.0", case.case_id % 10_000),
        "Finished test profile target(s) in 0.42s".to_owned(),
        "Running unittests src/lib.rs".to_owned(),
    ];
    for index in 0..case.command_lines {
        let word = &case.words[index % case.words.len()];
        lines.push(format!(
            "test generated_case_{index}_{word} ... ok repeated diagnostic detail {word}"
        ));
    }
    lines.push(format!(
        "test result: ok. {} passed; 0 failed; 0 ignored",
        case.command_lines
    ));
    lines.join("\n")
}

fn tool_definitions(case: &GeneratedCase) -> Value {
    json!([{
        "type": "function",
        "function": {
            "name": "run_property_command",
            "description": format!(
                "Run a generated command for case {} and return its detailed output. This intentionally verbose description may be compressed safely.",
                case.case_id
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "command": {
                        "type": "string",
                        "description": "The exact command to execute for this generated property case."
                    },
                    "attempt": {
                        "type": "integer",
                        "description": "The generated attempt number used for deterministic testing."
                    }
                },
                "required": ["command"],
                "additionalProperties": false
            }
        }
    }])
}

fn request(case: &GeneratedCase) -> OpenAIRequest {
    let protected = protected_region(case.protected_kind, case.case_id);
    let structured_protected = protected_region(
        case.protected_kind.wrapping_add(1),
        case.case_id.wrapping_add(1),
    );
    let mut system_extra = Map::new();
    if case.cache_marker {
        system_extra.insert("cache_control".to_owned(), json!({"type": "ephemeral"}));
    }
    let mut messages = vec![message(
        "system",
        Value::String(format!(
            "{} Preserve this protected region exactly: {protected}",
            prose(&case.words, 0, "system guidance", case.case_id)
        )),
        system_extra,
    )];

    for turn in 0..case.history_turns {
        let offset = 4 + turn * 5;
        let user_prose = prose(&case.words, offset, "historical user request", case.case_id);
        let user_text = if case.structured_user && turn == 0 {
            format!("{user_prose} Preserve this protected region exactly: {structured_protected}")
        } else {
            user_prose
        };
        messages.push(message(
            "user",
            content(
                user_text,
                case.structured_user && turn % 2 == 0,
                case.case_id.wrapping_add(turn as u64),
            ),
            Map::new(),
        ));
        let assistant_prose = prose(
            &case.words,
            offset + 1,
            "historical assistant response",
            case.case_id,
        );
        let assistant_text = if case.structured_assistant && turn == 0 {
            format!(
                "{assistant_prose} Preserve this protected region exactly: {structured_protected}"
            )
        } else {
            assistant_prose
        };
        messages.push(message(
            "assistant",
            content(
                assistant_text,
                case.structured_assistant,
                case.case_id.wrapping_add(turn as u64).wrapping_add(1),
            ),
            Map::new(),
        ));
    }

    let call_id = format!("property-call-{}", case.case_id);
    let mut call_extra = Map::new();
    call_extra.insert(
        "tool_calls".to_owned(),
        json!([{
            "id": call_id,
            "type": "function",
            "function": {
                "name": "run_property_command",
                "arguments": json!({"command": "cargo test", "attempt": case.case_id % 5}).to_string()
            }
        }]),
    );
    messages.push(message(
        "assistant",
        Value::String("I will run the requested command and inspect its output.".to_owned()),
        call_extra,
    ));

    let mut tool_extra = Map::new();
    tool_extra.insert("tool_call_id".to_owned(), Value::String(call_id));
    tool_extra.insert(
        "name".to_owned(),
        Value::String("run_property_command".to_owned()),
    );
    tool_extra.insert("command".to_owned(), Value::String("cargo test".to_owned()));
    messages.push(message(
        "tool",
        Value::String(command_output(case)),
        tool_extra,
    ));
    messages.push(plain_message(
        "user",
        format!(
            "{} The previous command output and protected value {protected} should inform the final concise answer.",
            prose(&case.words, 24, "latest user request", case.case_id)
        ),
    ));

    let mut extra = Map::new();
    extra.insert("response_format".to_owned(), json!({"type": "json_object"}));
    extra.insert("seed".to_owned(), json!(case.case_id));
    if case.include_tools {
        extra.insert("tools".to_owned(), tool_definitions(case));
        extra.insert("tool_choice".to_owned(), Value::String("auto".to_owned()));
    }

    OpenAIRequest {
        model: if case.case_id % 2 == 0 {
            "gpt-4o".to_owned()
        } else {
            "gpt-4".to_owned()
        },
        messages,
        stream: case.case_id % 3 == 0,
        temperature: case.temperature,
        max_tokens: Some(256 + (case.case_id % 768) as u32),
        extra,
    }
}

fn pipeline() -> CompressionPipeline {
    let mut config = CompressionConfig {
        enabled: true,
        compress_tool_definitions: true,
        time_budget_ms: TimeBudgetConfig {
            lite: 30_000,
            standard: 30_000,
            aggressive: 30_000,
            ultra: 30_000,
            rtk: 30_000,
            stacked: 30_000,
        },
        ..CompressionConfig::default()
    };
    config.custom_pipelines.insert(
        CUSTOM_PIPELINE.to_owned(),
        CustomPipelineConfig {
            engines: ["lite", "perplexity", "standard", "rtk"]
                .into_iter()
                .map(str::to_owned)
                .collect(),
        },
    );
    CompressionPipeline::from_config(config)
}

fn effective(level: CompressionLevel) -> EffectiveCompressionConfig {
    EffectiveCompressionConfig {
        enabled: true,
        level,
        auto_threshold_tokens: u32::MAX,
        caveman_output: false,
    }
}

fn assert_nonincreasing(case: GeneratedCase) -> TestCaseResult {
    let request = request(&case);
    let original = CompressiblePayload::from_openai_request(request.clone());
    let counter = CompressionContext::new(&request.model, "property-test").token_counter;
    let input_tokens = counter.count_request(&request);
    let pipeline = pipeline();
    let runtime = tokio::runtime::Builder::new_current_thread()
        .enable_time()
        .build()
        .expect("property-test runtime must build");

    for level in NAMED_LEVELS {
        let result = runtime.block_on(pipeline.compress_explicit(
            original.clone(),
            CompressionContext {
                prompt_caching_enabled: case.prompt_caching,
                ..CompressionContext::new(&request.model, "property-test")
            },
            effective(level),
            CompressionRequestMetadata {
                request_id: format!("property-{}-{level:?}", case.case_id),
                ..CompressionRequestMetadata::default()
            },
        ));
        assert_result_nonincreasing(&original, input_tokens, &counter, result)?;
    }

    let custom_result = runtime.block_on(pipeline.compress_explicit(
        original.clone(),
        CompressionContext {
            prompt_caching_enabled: case.prompt_caching,
            ..CompressionContext::new(&request.model, "property-test")
        },
        effective(CompressionLevel::Stacked),
        CompressionRequestMetadata {
            request_id: format!("property-{}-custom", case.case_id),
            custom_pipeline: Some(CUSTOM_PIPELINE.to_owned()),
            ..CompressionRequestMetadata::default()
        },
    ));
    assert_result_nonincreasing(&original, input_tokens, &counter, custom_result)
}

fn assert_result_nonincreasing(
    original: &CompressiblePayload,
    input_tokens: u32,
    counter: &super::token_counter::TokenCounter,
    result: super::pipeline::CompressionPipelineResult,
) -> TestCaseResult {
    let output_tokens = counter.count_request(&result.payload.clone().into_openai_request());

    prop_assert_eq!(result.original_tokens, input_tokens);
    prop_assert!(result.final_tokens <= result.original_tokens);
    prop_assert_eq!(result.final_tokens, output_tokens);
    prop_assert!(output_tokens <= input_tokens);

    if result.timed_out {
        prop_assert_eq!(&result.payload, original);
        prop_assert_eq!(result.final_tokens, result.original_tokens);
    }
    if result.error {
        prop_assert!(result.payload == *original || output_tokens <= input_tokens);
    }

    Ok(())
}

proptest! {
    #![proptest_config(ProptestConfig::with_cases(128))]

    #[test]
    fn property_4_compression_never_increases_token_count(case in generated_case()) {
        assert_nonincreasing(case)?;
    }
}
