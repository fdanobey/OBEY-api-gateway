use super::config::{CompressionConfig, EffectiveCompressionConfig};
use super::engines::{CompressiblePayload, CompressionLevel};
use super::pipeline::{CompressionPipeline, CompressionRequestMetadata};
use super::CompressionContext;
use crate::models::openai::{Message, OpenAIRequest};
use proptest::prelude::*;
use proptest::test_runner::TestCaseResult;
use serde_json::{json, Map, Value};
use std::collections::{BTreeMap, BTreeSet};

#[derive(Debug, Clone, Copy)]
enum ToolProtocol {
    OpenAi,
    Anthropic,
}

#[derive(Debug, Clone)]
struct PairSeed {
    anthropic: bool,
    id: String,
    function_name: String,
    arguments: Value,
    age_gap: usize,
    intervening_noise: usize,
    noise_word: String,
}

#[derive(Debug, Clone, PartialEq, Eq)]
enum PairEvent {
    Call(String),
    Result(String),
}

impl PairEvent {
    fn id(&self) -> &str {
        match self {
            Self::Call(id) | Self::Result(id) => id,
        }
    }
}

#[derive(Debug, Clone)]
struct CallObservation {
    id: String,
    function_name: Option<String>,
    arguments: Option<Value>,
    message_position: usize,
}

#[derive(Debug, Clone)]
struct ResultObservation {
    id: String,
    function_name: Option<String>,
    message_position: usize,
}

#[derive(Debug, Default)]
struct ConversationSnapshot {
    calls: Vec<CallObservation>,
    results: Vec<ResultObservation>,
    events: Vec<PairEvent>,
    malformed_calls: usize,
    malformed_results: usize,
}

fn identifier(pattern: &str) -> impl Strategy<Value = String> {
    proptest::string::string_regex(pattern).expect("identifier regex must compile")
}

fn json_value_strategy() -> BoxedStrategy<Value> {
    let leaf = prop_oneof![
        Just(Value::Null),
        any::<bool>().prop_map(Value::Bool),
        (-100_000i64..100_000).prop_map(|number| Value::Number(number.into())),
        identifier("[A-Za-z0-9 _.-]{0,24}").prop_map(Value::String),
    ];

    leaf.prop_recursive(3, 32, 5, |inner| {
        prop_oneof![
            prop::collection::vec(inner.clone(), 0..5).prop_map(Value::Array),
            prop::collection::btree_map(identifier("[a-z][a-z0-9_]{0,10}"), inner, 0..5)
                .prop_map(|entries| Value::Object(entries.into_iter().collect())),
        ]
    })
    .boxed()
}

fn pair_seed_strategy() -> impl Strategy<Value = PairSeed> {
    (
        any::<bool>(),
        identifier("[A-Za-z0-9_-]{1,18}"),
        identifier("[a-z][a-z0-9_]{0,14}"),
        json_value_strategy(),
        0usize..4,
        0usize..3,
        identifier("[a-z]{2,12}"),
    )
        .prop_map(
            |(anthropic, id, function_name, arguments, age_gap, intervening_noise, noise_word)| {
                PairSeed {
                    anthropic,
                    id,
                    function_name,
                    arguments,
                    age_gap,
                    intervening_noise,
                    noise_word,
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

fn push_noise_turn(messages: &mut Vec<Message>, ordinal: usize, word: &str) {
    messages.push(plain_message(
        "user",
        format!(
            "Noise turn {ordinal} {word}: could you please actually inspect this ordinary request in order to help?"
        ),
    ));
    messages.push(plain_message(
        "assistant",
        format!(
            "Noise turn {ordinal} {word}: I think this is a very detailed ordinary response, and I hope this helps."
        ),
    ));
}

fn protocol_for(pair_index: usize, seed: &PairSeed) -> ToolProtocol {
    match pair_index {
        0 => ToolProtocol::OpenAi,
        1 => ToolProtocol::Anthropic,
        _ if seed.anthropic => ToolProtocol::Anthropic,
        _ => ToolProtocol::OpenAi,
    }
}

fn generated_request(
    case_id: u64,
    leading_turns: usize,
    trailing_turns: usize,
    seeds: &[PairSeed],
) -> OpenAIRequest {
    let mut messages = vec![plain_message(
        "system",
        format!("Pair integrity policy for case {case_id} must remain available."),
    )];
    let mut noise_ordinal = 0usize;

    for turn in 0..leading_turns {
        push_noise_turn(
            &mut messages,
            noise_ordinal,
            &seeds[turn % seeds.len()].noise_word,
        );
        noise_ordinal += 1;
    }

    for (pair_index, seed) in seeds.iter().enumerate() {
        for _ in 0..seed.age_gap {
            push_noise_turn(&mut messages, noise_ordinal, &seed.noise_word);
            noise_ordinal += 1;
        }

        let id = format!("{}-{case_id:x}-{pair_index}", seed.id);
        let function_name = format!("{}_{}", seed.function_name, pair_index);
        let arguments = json!({
            "payload": seed.arguments,
            "pair_index": pair_index,
            "enabled": pair_index % 2 == 0,
        });
        messages.push(plain_message(
            "user",
            format!(
                "Pair request {pair_index} {} asks for an ordinary tool operation.",
                seed.noise_word
            ),
        ));

        match protocol_for(pair_index, seed) {
            ToolProtocol::OpenAi => {
                let mut call_extra = Map::new();
                call_extra.insert(
                    "tool_calls".to_owned(),
                    json!([{
                        "id": id,
                        "type": "function",
                        "function": {
                            "name": function_name,
                            "arguments": arguments.to_string(),
                        },
                    }]),
                );
                messages.push(message(
                    "assistant",
                    Value::String(format!(
                        "Calling the requested operation for pair {pair_index}."
                    )),
                    call_extra,
                ));

                for noise_index in 0..seed.intervening_noise {
                    messages.push(plain_message(
                        "assistant",
                        format!(
                            "Intervening noise {noise_index} {} is ordinary and disposable.",
                            seed.noise_word
                        ),
                    ));
                }

                let mut result_extra = Map::new();
                result_extra.insert("tool_call_id".to_owned(), Value::String(id));
                result_extra.insert("name".to_owned(), Value::String(function_name));
                messages.push(message(
                    "tool",
                    Value::String(format!(
                        "progress: 10%\nprogress: 20%\nsrc/case_{pair_index}.rs:12: result {}\nstatus: success",
                        seed.noise_word
                    )),
                    result_extra,
                ));
            }
            ToolProtocol::Anthropic => {
                messages.push(message(
                    "assistant",
                    json!([
                        {
                            "type": "text",
                            "text": format!("Calling the requested operation for pair {pair_index}."),
                        },
                        {
                            "type": "tool_use",
                            "id": id,
                            "name": function_name,
                            "input": arguments,
                        }
                    ]),
                    Map::new(),
                ));

                for noise_index in 0..seed.intervening_noise {
                    messages.push(plain_message(
                        "assistant",
                        format!(
                            "Intervening noise {noise_index} {} is ordinary and disposable.",
                            seed.noise_word
                        ),
                    ));
                }

                messages.push(message(
                    "user",
                    json!([{
                        "type": "tool_result",
                        "tool_use_id": id,
                        "content": [{
                            "type": "text",
                            "text": format!(
                                "progress: 10%\nprogress: 20%\nsrc/case_{pair_index}.rs:12: result {}\nstatus: success",
                                seed.noise_word
                            ),
                        }],
                    }]),
                    Map::new(),
                ));
            }
        }

        messages.push(plain_message(
            "assistant",
            format!("Pair {pair_index} completed; this ordinary summary may be compressed."),
        ));
    }

    for turn in 0..trailing_turns {
        push_noise_turn(
            &mut messages,
            noise_ordinal,
            &seeds[(turn + leading_turns) % seeds.len()].noise_word,
        );
        noise_ordinal += 1;
    }

    OpenAIRequest {
        model: "gpt-4o".to_owned(),
        messages,
        stream: false,
        temperature: None,
        max_tokens: None,
        extra: Map::new(),
    }
}

fn parsed_arguments(value: Option<&Value>) -> Option<Value> {
    let text = value?.as_str()?;
    serde_json::from_str(text).ok()
}

fn snapshot(payload: &CompressiblePayload) -> ConversationSnapshot {
    let mut snapshot = ConversationSnapshot::default();

    for (message_position, message) in payload.messages.iter().enumerate() {
        if let Some(tool_calls) = message.extra.get("tool_calls").and_then(Value::as_array) {
            for call in tool_calls {
                let id = call.get("id").and_then(Value::as_str);
                let function = call.get("function");
                let function_name = function
                    .and_then(|value| value.get("name"))
                    .and_then(Value::as_str);
                let arguments = parsed_arguments(function.and_then(|value| value.get("arguments")));
                if let Some(id) = id {
                    snapshot.calls.push(CallObservation {
                        id: id.to_owned(),
                        function_name: function_name.map(str::to_owned),
                        arguments,
                        message_position,
                    });
                    snapshot.events.push(PairEvent::Call(id.to_owned()));
                } else {
                    snapshot.malformed_calls += 1;
                }
            }
        }

        if message.role == "tool" || message.extra.contains_key("tool_call_id") {
            if let Some(id) = message.extra.get("tool_call_id").and_then(Value::as_str) {
                snapshot.results.push(ResultObservation {
                    id: id.to_owned(),
                    function_name: message
                        .extra
                        .get("name")
                        .and_then(Value::as_str)
                        .map(str::to_owned),
                    message_position,
                });
                snapshot.events.push(PairEvent::Result(id.to_owned()));
            } else {
                snapshot.malformed_results += 1;
            }
        }

        if let Some(blocks) = message.content.as_value().as_array() {
            for block in blocks {
                let Some(object) = block.as_object() else {
                    continue;
                };
                match object.get("type").and_then(Value::as_str) {
                    Some("tool_use") => {
                        if let Some(id) = object.get("id").and_then(Value::as_str) {
                            snapshot.calls.push(CallObservation {
                                id: id.to_owned(),
                                function_name: object
                                    .get("name")
                                    .and_then(Value::as_str)
                                    .map(str::to_owned),
                                arguments: object.get("input").cloned(),
                                message_position,
                            });
                            snapshot.events.push(PairEvent::Call(id.to_owned()));
                        } else {
                            snapshot.malformed_calls += 1;
                        }
                    }
                    Some("tool_result") => {
                        if let Some(id) = object.get("tool_use_id").and_then(Value::as_str) {
                            snapshot.results.push(ResultObservation {
                                id: id.to_owned(),
                                function_name: None,
                                message_position,
                            });
                            snapshot.events.push(PairEvent::Result(id.to_owned()));
                        } else {
                            snapshot.malformed_results += 1;
                        }
                    }
                    _ => {}
                }
            }
        }
    }

    snapshot
}

fn calls_by_id(snapshot: &ConversationSnapshot) -> BTreeMap<String, &CallObservation> {
    snapshot
        .calls
        .iter()
        .map(|call| (call.id.clone(), call))
        .collect()
}

fn results_by_id(snapshot: &ConversationSnapshot) -> BTreeMap<String, &ResultObservation> {
    snapshot
        .results
        .iter()
        .map(|result| (result.id.clone(), result))
        .collect()
}

fn assert_pair_invariants(
    before: &CompressiblePayload,
    after: &CompressiblePayload,
    expected_pairs: usize,
) -> TestCaseResult {
    let before_snapshot = snapshot(before);
    let after_snapshot = snapshot(after);
    prop_assert_eq!(before_snapshot.malformed_calls, 0);
    prop_assert_eq!(before_snapshot.malformed_results, 0);
    prop_assert_eq!(after_snapshot.malformed_calls, 0);
    prop_assert_eq!(after_snapshot.malformed_results, 0);

    let before_calls = calls_by_id(&before_snapshot);
    let before_results = results_by_id(&before_snapshot);
    let after_calls = calls_by_id(&after_snapshot);
    let after_results = results_by_id(&after_snapshot);
    prop_assert_eq!(before_calls.len(), before_snapshot.calls.len());
    prop_assert_eq!(before_results.len(), before_snapshot.results.len());
    prop_assert_eq!(before_calls.len(), expected_pairs);
    prop_assert_eq!(before_results.len(), expected_pairs);
    prop_assert_eq!(after_calls.len(), after_snapshot.calls.len());
    prop_assert_eq!(after_results.len(), after_snapshot.results.len());

    let before_call_ids = before_calls.keys().cloned().collect::<BTreeSet<_>>();
    let before_result_ids = before_results.keys().cloned().collect::<BTreeSet<_>>();
    let after_call_ids = after_calls.keys().cloned().collect::<BTreeSet<_>>();
    let after_result_ids = after_results.keys().cloned().collect::<BTreeSet<_>>();
    prop_assert_eq!(&before_call_ids, &before_result_ids);
    prop_assert_eq!(&after_call_ids, &after_result_ids);
    prop_assert!(after_call_ids.is_subset(&before_call_ids));

    let retained_events = before_snapshot
        .events
        .iter()
        .filter(|event| after_call_ids.contains(event.id()))
        .cloned()
        .collect::<Vec<_>>();
    prop_assert_eq!(&after_snapshot.events, &retained_events);

    for id in &after_call_ids {
        let before_call = before_calls[id];
        let before_result = before_results[id];
        let after_call = after_calls[id];
        let after_result = after_results[id];
        prop_assert_eq!(&after_call.id, &before_call.id);
        prop_assert_eq!(&after_result.id, &before_result.id);
        prop_assert_eq!(&after_call.function_name, &before_call.function_name);
        prop_assert_eq!(&after_result.function_name, &before_result.function_name);
        prop_assert_eq!(&after_call.arguments, &before_call.arguments);
        prop_assert!(after_call.arguments.is_some());
        prop_assert!(after_call.message_position < after_result.message_position);

        let call_message = &after.messages[after_call.message_position];
        let result_message = &after.messages[after_result.message_position];
        prop_assert!(call_message.relationships.tool_call_ids.contains(id));
        prop_assert!(result_message
            .relationships
            .tool_result_for_ids
            .contains(id));
        prop_assert!(call_message
            .relationships
            .related_message_indices
            .contains(&result_message.original_index));
        prop_assert!(result_message
            .relationships
            .related_message_indices
            .contains(&call_message.original_index));
        prop_assert!(call_message.relationships.tool_names.contains(
            before_call
                .function_name
                .as_ref()
                .expect("generated calls always have function names")
        ));
    }

    Ok(())
}

fn all_named_levels() -> [CompressionLevel; 7] {
    [
        CompressionLevel::None,
        CompressionLevel::Lite,
        CompressionLevel::Standard,
        CompressionLevel::Aggressive,
        CompressionLevel::Ultra,
        CompressionLevel::Rtk,
        CompressionLevel::Stacked,
    ]
}

fn assert_tool_pair_integrity(
    case_id: u64,
    leading_turns: usize,
    trailing_turns: usize,
    target_budget: u32,
    seeds: Vec<PairSeed>,
) -> TestCaseResult {
    let request = generated_request(case_id, leading_turns, trailing_turns, &seeds);
    let original = CompressiblePayload::from_openai_request(request);
    let pair_ages = snapshot(&original)
        .calls
        .iter()
        .map(|call| original.messages[call.message_position].age)
        .collect::<BTreeSet<_>>();
    prop_assert!(pair_ages.len() > 1);

    let mut config = CompressionConfig::default();
    config.enabled = true;
    config.time_budget_ms.lite = 60_000;
    config.time_budget_ms.standard = 60_000;
    config.time_budget_ms.aggressive = 60_000;
    config.time_budget_ms.ultra = 60_000;
    config.time_budget_ms.rtk = 60_000;
    config.time_budget_ms.stacked = 60_000;
    let pipeline = CompressionPipeline::from_config(config);
    let runtime = tokio::runtime::Builder::new_current_thread()
        .enable_time()
        .build()
        .expect("property-test runtime must build");

    runtime.block_on(async move {
        for level in all_named_levels() {
            let mut context = CompressionContext::new("gpt-4o", "property-test");
            context.target_token_budget = Some(target_budget.max(1));
            let output = pipeline
                .compress_explicit(
                    original.clone(),
                    context.clone(),
                    EffectiveCompressionConfig {
                        enabled: true,
                        level,
                        auto_threshold_tokens: 0,
                        caveman_output: false,
                    },
                    CompressionRequestMetadata {
                        request_id: format!("property-2-{case_id}-{level:?}"),
                        ..CompressionRequestMetadata::default()
                    },
                )
                .await;

            prop_assert!(!output.timed_out);
            prop_assert!(output.errors.is_empty());
            assert_pair_invariants(&original, &output.payload, seeds.len())?;
            let recounted = context
                .token_counter
                .count_request(&output.payload.clone().into_openai_request());
            prop_assert_eq!(output.final_tokens, recounted);
            prop_assert!(output.final_tokens <= output.original_tokens);
        }
        Ok(())
    })
}

proptest! {
    #![proptest_config(ProptestConfig::with_cases(64))]

    #[test]
    fn property_2_tool_call_result_pair_integrity_across_all_named_levels(
        case_id in any::<u64>(),
        leading_turns in 0usize..4,
        trailing_turns in 1usize..4,
        target_budget in 1u32..96,
        seeds in prop::collection::vec(pair_seed_strategy(), 2..7),
    ) {
        assert_tool_pair_integrity(
            case_id,
            leading_turns,
            trailing_turns,
            target_budget,
            seeds,
        )?;
    }
}
