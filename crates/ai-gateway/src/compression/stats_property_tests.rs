use super::config::{CompressionConfig, EffectiveCompressionConfig, TimeBudgetConfig};
use super::pipeline::{CompressionPipeline, CompressionRequestMetadata};
use super::{CompressiblePayload, CompressionContext, CompressionLevel};
use crate::models::openai::{Message, OpenAIRequest};
use proptest::prelude::*;
use proptest::test_runner::TestCaseResult;
use serde_json::{Map, Value};

const SAVINGS_EPSILON: f64 = 1.0e-10;

fn compression_level_strategy() -> impl Strategy<Value = CompressionLevel> {
    prop_oneof![
        Just(CompressionLevel::None),
        Just(CompressionLevel::Lite),
        Just(CompressionLevel::Standard),
        Just(CompressionLevel::Aggressive),
        Just(CompressionLevel::Ultra),
        Just(CompressionLevel::Rtk),
        Just(CompressionLevel::Stacked),
    ]
}

fn prose_word() -> impl Strategy<Value = String> {
    proptest::string::string_regex("[a-z]{2,12}").expect("prose word regex must compile")
}

fn payload_strategy() -> impl Strategy<Value = CompressiblePayload> {
    (
        any::<bool>(),
        any::<u64>(),
        prop::collection::vec(prose_word(), 6..12),
        8usize..24,
    )
        .prop_map(|(compressible, case_id, words, repetitions)| {
            let request = if compressible {
                compressible_request(case_id, &words, repetitions)
            } else {
                unchanged_request(case_id, &words[0])
            };
            CompressiblePayload::from_openai_request(request)
        })
}

fn message(role: &str, content: String, extra: Map<String, Value>) -> Message {
    Message {
        role: role.to_owned(),
        content: Value::String(content),
        extra,
    }
}

fn compressible_request(case_id: u64, words: &[String], repetitions: usize) -> OpenAIRequest {
    let repeated_status = (0..repetitions)
        .map(|index| {
            format!(
                "modified: src/{}/{}_{}.rs    awaiting review",
                words[index % words.len()],
                case_id,
                index % 3
            )
        })
        .collect::<Vec<_>>()
        .join("\n");
    let mut command_extra = Map::new();
    command_extra.insert("command".to_owned(), Value::String("git status".to_owned()));

    OpenAIRequest {
        model: "gpt-4o".to_owned(),
        messages: vec![
            message(
                "system",
                format!("System policy {case_id} must remain exactly as supplied."),
                Map::new(),
            ),
            message(
                "user",
                format!(
                    "Could you please  actually review {} in order to make sure that it is able to work due to the fact that {} is very important?",
                    words[0], words[1]
                ),
                Map::new(),
            ),
            message(
                "assistant",
                format!(
                    "I think this is  actually a very detailed response about {}, and I hope this helps.",
                    words[2]
                ),
                Map::new(),
            ),
            message("tool", repeated_status, command_extra),
            message(
                "user",
                format!(
                    "Could you please  actually summarize {} and {} in order to finish?",
                    words[3], words[4]
                ),
                Map::new(),
            ),
        ],
        stream: false,
        temperature: None,
        max_tokens: None,
        extra: Map::new(),
    }
}

fn unchanged_request(case_id: u64, word: &str) -> OpenAIRequest {
    OpenAIRequest {
        model: "gpt-4o".to_owned(),
        messages: vec![message(
            "user",
            format!("```rust\nconst CASE_{case_id}: &str = \"{word}\";\n```"),
            Map::new(),
        )],
        stream: false,
        temperature: None,
        max_tokens: None,
        extra: Map::new(),
    }
}

fn effective(level: CompressionLevel) -> EffectiveCompressionConfig {
    EffectiveCompressionConfig {
        enabled: true,
        level,
        auto_threshold_tokens: u32::MAX,
        caveman_output: false,
    }
}

fn local_savings_oracle(original_tokens: u32, final_tokens: u32) -> f64 {
    if original_tokens == 0 {
        0.0
    } else {
        (1.0 - f64::from(final_tokens) / f64::from(original_tokens)) * 100.0
    }
}

fn savings_percent_from_counts(original_tokens: u32, final_tokens: u32) -> f64 {
    if original_tokens == 0 {
        0.0
    } else {
        f64::from(original_tokens.saturating_sub(final_tokens)) * 100.0 / f64::from(original_tokens)
    }
}

fn assert_nonnegative_duration_type(_: u64) {}

fn assert_stats_consistency(
    original: CompressiblePayload,
    level: CompressionLevel,
) -> TestCaseResult {
    let mut config = CompressionConfig {
        enabled: true,
        default_level: level,
        ..CompressionConfig::default()
    };
    config.time_budget_ms = TimeBudgetConfig {
        lite: 60_000,
        standard: 60_000,
        aggressive: 60_000,
        ultra: 60_000,
        rtk: 60_000,
        stacked: 60_000,
    };
    let pipeline = CompressionPipeline::from_config(config);
    let context = CompressionContext::new("gpt-4o", "property-test");
    let expected_original_tokens = context
        .token_counter
        .count_request(&original.clone().into_openai_request());
    let runtime = tokio::runtime::Builder::new_current_thread()
        .enable_time()
        .build()
        .expect("property-test runtime must build");
    let result = runtime.block_on(pipeline.compress_explicit(
        original.clone(),
        context.clone(),
        effective(level),
        CompressionRequestMetadata::default(),
    ));
    let expected_final_tokens = context
        .token_counter
        .count_request(&result.payload.clone().into_openai_request());

    prop_assert!(!result.timed_out);
    prop_assert!(!result.error);
    prop_assert_eq!(result.original_tokens, expected_original_tokens);
    prop_assert_eq!(result.final_tokens, expected_final_tokens);
    prop_assert!(result.final_tokens <= result.original_tokens);
    assert_nonnegative_duration_type(result.duration_ms);

    let applied_from_results = result
        .engine_results
        .iter()
        .filter(|engine| engine.applied)
        .map(|engine| engine.engine_name.as_str())
        .collect::<Vec<_>>();
    let applied_from_pipeline = result
        .engines_applied
        .iter()
        .map(String::as_str)
        .collect::<Vec<_>>();
    prop_assert_eq!(applied_from_results, applied_from_pipeline);

    if let Some(first) = result.engine_results.first() {
        prop_assert_eq!(first.tokens_before, result.original_tokens);
        for engine in &result.engine_results {
            prop_assert!(engine.tokens_after <= engine.tokens_before);
            assert_nonnegative_duration_type(engine.duration_ms);
        }
        for adjacent in result.engine_results.windows(2) {
            prop_assert_eq!(adjacent[0].tokens_after, adjacent[1].tokens_before);
        }
        prop_assert_eq!(
            result
                .engine_results
                .last()
                .expect("non-empty results have a final engine")
                .tokens_after,
            result.final_tokens
        );
    } else {
        prop_assert_eq!(result.final_tokens, result.original_tokens);
    }

    if result.final_tokens < result.original_tokens {
        prop_assert!(!result.engines_applied.is_empty());
    }
    if result.payload == original {
        prop_assert!(result.engines_applied.is_empty());
        prop_assert_eq!(result.final_tokens, result.original_tokens);
    }

    let computed = savings_percent_from_counts(result.original_tokens, result.final_tokens);
    let oracle = local_savings_oracle(result.original_tokens, result.final_tokens);
    prop_assert!((computed - oracle).abs() <= SAVINGS_EPSILON);

    Ok(())
}

proptest! {
    #![proptest_config(ProptestConfig::with_cases(128))]

    #[test]
    fn property_5_compression_statistics_mathematical_consistency(
        payload in payload_strategy(),
        level in compression_level_strategy(),
    ) {
        assert_stats_consistency(payload, level)?;
    }
}
