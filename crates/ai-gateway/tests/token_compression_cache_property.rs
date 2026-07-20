use ai_gateway::{
    compression::{
        config::{CompressionConfig, EffectiveCompressionConfig, TimeBudgetConfig},
        pipeline::{
            CacheDowngradeMetadata, CompressionPipeline, CompressionPipelineResult,
            CompressionRequestMetadata,
        },
        CompressiblePayload, CompressionContext, CompressionLevel,
    },
    models::openai::{Message, OpenAIRequest},
};
use proptest::{prelude::*, test_runner::TestCaseResult};
use serde_json::{json, Map, Value};

const LEVELS: [CompressionLevel; 6] = [
    CompressionLevel::Lite,
    CompressionLevel::Standard,
    CompressionLevel::Aggressive,
    CompressionLevel::Ultra,
    CompressionLevel::Rtk,
    CompressionLevel::Stacked,
];

#[derive(Debug)]
struct CacheCase {
    case_id: u64,
    words: Vec<String>,
    message_count: usize,
    marker_index: usize,
    marker_in_content: bool,
}

fn cache_case() -> impl Strategy<Value = CacheCase> {
    (
        any::<u64>(),
        prop::collection::vec(
            proptest::string::string_regex("[a-z]{3,12}").expect("word regex must compile"),
            24,
        ),
        3usize..9,
        any::<usize>(),
        any::<bool>(),
    )
        .prop_map(
            |(case_id, words, message_count, marker_seed, marker_in_content)| CacheCase {
                case_id,
                words,
                message_count,
                marker_index: marker_seed % (message_count - 1),
                marker_in_content,
            },
        )
}

fn prose(case: &CacheCase, index: usize) -> String {
    format!(
        "{} {} generated message {index} for case {}. This is intentionally verbose ordinary prose that can be safely normalized or shortened after the stable cached prefix boundary. {} {} provide repeated explanatory context for compression.",
        case.words[index % case.words.len()],
        case.words[(index + 1) % case.words.len()],
        case.case_id,
        case.words[(index + 2) % case.words.len()],
        case.words[(index + 3) % case.words.len()],
    )
}

fn request(case: &CacheCase) -> OpenAIRequest {
    let messages = (0..case.message_count)
        .map(|index| {
            let role = if index == 0 {
                "system"
            } else if index % 2 == 0 {
                "assistant"
            } else {
                "user"
            };
            let text = prose(case, index);
            let mut extra = Map::new();
            let content = if index == case.marker_index && case.marker_in_content {
                json!([{
                    "type": "text",
                    "text": text,
                    "cache_control": {"type": "ephemeral", "ttl": "5m"}
                }])
            } else {
                if index == case.marker_index {
                    extra.insert(
                        "cache_control".to_owned(),
                        json!({"type": "ephemeral", "ttl": "5m"}),
                    );
                }
                Value::String(text)
            };
            Message {
                role: role.to_owned(),
                content,
                extra,
            }
        })
        .collect();

    OpenAIRequest {
        model: "gpt-4o".to_owned(),
        messages,
        stream: false,
        temperature: Some(0.2),
        max_tokens: Some(512),
        extra: Map::new(),
    }
}

fn pipeline() -> CompressionPipeline {
    CompressionPipeline::from_config(CompressionConfig {
        enabled: true,
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

fn marker_values(request: &OpenAIRequest) -> Vec<(usize, Value)> {
    request
        .messages
        .iter()
        .enumerate()
        .filter_map(|(index, message)| {
            message
                .extra
                .get("cache_control")
                .cloned()
                .or_else(|| find_cache_marker(&message.content).cloned())
                .map(|marker| (index, marker))
        })
        .collect()
}

fn find_cache_marker(value: &Value) -> Option<&Value> {
    match value {
        Value::Array(values) => values.iter().find_map(find_cache_marker),
        Value::Object(object) => object
            .get("cache_control")
            .or_else(|| object.values().find_map(find_cache_marker)),
        _ => None,
    }
}

fn assert_result(
    original: &OpenAIRequest,
    boundary: usize,
    level: CompressionLevel,
    result: CompressionPipelineResult,
) -> TestCaseResult {
    let output = result.payload.into_openai_request();
    let original_prefix = serde_json::to_vec(&original.messages[..=boundary])
        .expect("original cached prefix must serialize");
    let output_prefix =
        serde_json::to_vec(&output.messages[..=boundary]).expect("output prefix must serialize");

    prop_assert_eq!(
        output_prefix,
        original_prefix,
        "{:?} changed cached prefix bytes",
        level
    );
    prop_assert_eq!(
        marker_values(&output),
        marker_values(original),
        "{:?} moved, removed, or changed a cache marker",
        level
    );
    prop_assert!(
        result.final_tokens <= result.original_tokens,
        "{level:?} increased tokens from {} to {}",
        result.original_tokens,
        result.final_tokens
    );

    let high_level = matches!(
        level,
        CompressionLevel::Aggressive
            | CompressionLevel::Ultra
            | CompressionLevel::Rtk
            | CompressionLevel::Stacked
    );
    if high_level {
        prop_assert!(result.cache_downgrade_applied);
        prop_assert_eq!(
            result.cache_downgrade,
            Some(CacheDowngradeMetadata {
                provider: "anthropic".to_owned(),
                requested_level: level,
                actual_prefix_level: CompressionLevel::None,
                boundary_message_index: boundary,
            })
        );
    } else {
        prop_assert!(!result.cache_downgrade_applied);
        prop_assert!(result.cache_downgrade.is_none());
    }

    Ok(())
}

fn assert_cache_property(case: CacheCase) -> TestCaseResult {
    let original = request(&case);
    let payload = CompressiblePayload::from_openai_request(original.clone());
    let pipeline = pipeline();
    let runtime = tokio::runtime::Builder::new_current_thread()
        .enable_time()
        .build()
        .expect("property-test runtime must build");

    for level in LEVELS {
        let result = runtime.block_on(pipeline.compress_explicit(
            payload.clone(),
            CompressionContext {
                context_window: u32::MAX,
                prompt_caching_enabled: true,
                ..CompressionContext::new(&original.model, "anthropic")
            },
            effective(level),
            CompressionRequestMetadata {
                request_id: format!("cache-property-{}-{level:?}", case.case_id),
                ..CompressionRequestMetadata::default()
            },
        ));
        assert_result(&original, case.marker_index, level, result)?;
    }

    Ok(())
}

proptest! {
    #![proptest_config(ProptestConfig::with_cases(128))]

    #[test]
    fn cached_prefix_is_preserved_through_production_compression(case in cache_case()) {
        assert_cache_property(case)?;
    }
}
