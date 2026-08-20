use super::engines::{aggressive::AggressiveEngine, ultra::UltraEngine, CompressibleMessage};
use super::{CompressiblePayload, CompressionContext, CompressionEngine};
use crate::models::openai::{Message, OpenAIRequest};
use proptest::prelude::*;
use proptest::test_runner::TestCaseResult;
use serde_json::{json, Map, Value};

fn prose_word() -> impl Strategy<Value = String> {
    proptest::string::string_regex("[a-z]{2,12}").expect("prose word regex must compile")
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

fn generated_conversation(
    system_count: usize,
    turns: usize,
    active_call_count: usize,
    case_id: u64,
    words: &[String],
) -> OpenAIRequest {
    let mut messages = Vec::new();

    for system_index in 0..system_count {
        messages.push(plain_message(
            "system",
            format!(
                "{} {} system guidance {} must remain exactly as supplied",
                words[system_index],
                words[system_index + 1],
                system_index
            ),
        ));
    }

    for turn in 0..turns {
        let word_offset = system_count * 2 + turn * 4;
        messages.push(plain_message(
            "user",
            format!(
                "{} {} user request {} asks for a careful ordinary answer",
                words[word_offset],
                words[word_offset + 1],
                turn
            ),
        ));

        let resolved_id = format!("resolved-{case_id}-{turn}");
        let mut assistant_extra = Map::new();
        assistant_extra.insert(
            "tool_calls".to_owned(),
            json!([{
                "id": resolved_id,
                "type": "function",
                "function": {
                    "name": "lookup",
                    "arguments": json!({
                        "case": case_id.to_string(),
                        "turn": turn,
                        "term": words[word_offset + 2],
                    })
                    .to_string(),
                },
            }]),
        );
        messages.push(message(
            "assistant",
            Value::String(format!(
                "{} assistant response {} requests an ordinary lookup",
                words[word_offset + 2],
                turn
            )),
            assistant_extra,
        ));

        let mut tool_extra = Map::new();
        tool_extra.insert("tool_call_id".to_owned(), Value::String(resolved_id));
        tool_extra.insert("name".to_owned(), Value::String("lookup".to_owned()));
        messages.push(message(
            "tool",
            Value::String(format!(
                "{} {} tool answer {} contains ordinary prose",
                words[word_offset + 2],
                words[word_offset + 3],
                turn
            )),
            tool_extra,
        ));

        messages.push(plain_message(
            "assistant",
            format!(
                "{} assistant summary {} gives a useful ordinary response",
                words[word_offset + 3],
                turn
            ),
        ));
    }

    let mut active_content = vec![json!({
        "type": "text",
        "text": format!(
            "{} {} assistant now requests pending work",
            words[words.len() - 2],
            words[words.len() - 1]
        ),
    })];
    for call_index in 0..active_call_count {
        active_content.push(json!({
            "type": "tool_use",
            "id": format!("active-{case_id}-{call_index}"),
            "name": "lookup",
            "input": {
                "case": case_id.to_string(),
                "call": call_index,
                "term": words[(call_index * 3) % words.len()],
            },
        }));
    }
    messages.push(message(
        "assistant",
        Value::Array(active_content),
        Map::new(),
    ));

    OpenAIRequest {
        model: "gpt-4o".to_owned(),
        messages,
        stream: false,
        temperature: None,
        max_tokens: None,
        extra: Map::new(),
    }
}

fn visible_prose_leaves(value: &Value) -> Vec<&str> {
    match value {
        Value::String(text) => vec![text],
        Value::Array(parts) => parts
            .iter()
            .filter_map(|part| {
                let object = part.as_object()?;
                match object.get("type").and_then(Value::as_str) {
                    Some("text" | "input_text" | "output_text") => {
                        object.get("text").and_then(Value::as_str)
                    }
                    _ => None,
                }
            })
            .collect(),
        _ => Vec::new(),
    }
}

fn tool_uses(message: &CompressibleMessage) -> Vec<(String, String, Value)> {
    message
        .content
        .as_value()
        .as_array()
        .into_iter()
        .flatten()
        .filter_map(|block| {
            let object = block.as_object()?;
            if object.get("type").and_then(Value::as_str) != Some("tool_use") {
                return None;
            }
            Some((
                object.get("id")?.as_str()?.to_owned(),
                object.get("name")?.as_str()?.to_owned(),
                object.get("input")?.clone(),
            ))
        })
        .collect()
}

fn message_by_original_index(
    payload: &CompressiblePayload,
    original_index: usize,
) -> Option<&CompressibleMessage> {
    payload
        .messages
        .iter()
        .find(|message| message.original_index == original_index)
}

fn assert_wire_message_unchanged(
    before: &CompressibleMessage,
    after: &CompressibleMessage,
) -> TestCaseResult {
    prop_assert_eq!(after.original_index, before.original_index);
    prop_assert_eq!(after.role.as_str(), before.role.as_str());
    prop_assert_eq!(&after.content, &before.content);
    prop_assert_eq!(&after.extra, &before.extra);
    Ok(())
}

fn assert_common_critical_invariants(
    before: &CompressiblePayload,
    after: &CompressiblePayload,
) -> TestCaseResult {
    for system in before.messages.iter().filter(|message| message.is_system()) {
        let retained = message_by_original_index(after, system.original_index);
        prop_assert!(retained.is_some());
        assert_wire_message_unchanged(system, retained.expect("system presence checked"))?;
    }

    let latest_user = before
        .messages
        .iter()
        .rev()
        .find(|message| message.role == "user")
        .expect("generated conversation must contain user messages");
    let retained_latest = message_by_original_index(after, latest_user.original_index);
    prop_assert!(retained_latest.is_some());
    assert_wire_message_unchanged(
        latest_user,
        retained_latest.expect("latest user presence checked"),
    )?;

    let active_before = before
        .messages
        .iter()
        .find(|message| !message.relationships.unresolved_tool_call_ids.is_empty())
        .expect("generated conversation must contain unresolved tool uses");
    let active_after = message_by_original_index(after, active_before.original_index);
    prop_assert!(active_after.is_some());
    let active_after = active_after.expect("active tool-use presence checked");
    assert_wire_message_unchanged(active_before, active_after)?;
    prop_assert_eq!(tool_uses(active_after), tool_uses(active_before));
    prop_assert_eq!(
        &active_after.relationships.unresolved_tool_call_ids,
        &active_before.relationships.unresolved_tool_call_ids
    );

    Ok(())
}

fn assert_critical_message_preservation(
    system_count: usize,
    turns: usize,
    active_call_count: usize,
    case_id: u64,
    words: Vec<String>,
) -> TestCaseResult {
    let context = CompressionContext::new("gpt-4o", "property-test");
    let request = generated_conversation(system_count, turns, active_call_count, case_id, &words);
    for message in &request.messages {
        for text in visible_prose_leaves(&message.content) {
            prop_assert!(context.protection_scanner.scan(text).is_empty());
        }
    }

    let input_tokens = context.token_counter.count_request(&request);
    let original = CompressiblePayload::from_openai_request(request);
    let active = original
        .messages
        .iter()
        .find(|message| !message.relationships.unresolved_tool_call_ids.is_empty())
        .expect("generated conversation must contain unresolved tool uses");
    prop_assert!(active.critical);
    prop_assert_eq!(
        active.relationships.unresolved_tool_call_ids.len(),
        active_call_count
    );
    prop_assert_eq!(tool_uses(active).len(), active_call_count);

    let runtime = tokio::runtime::Builder::new_current_thread()
        .build()
        .expect("property-test runtime must build");

    let mut aggressive = original.clone();
    let aggressive_result =
        runtime.block_on(AggressiveEngine::new().compress(&mut aggressive, &context));
    assert_common_critical_invariants(&original, &aggressive)?;
    let recent_users = original
        .messages
        .iter()
        .rev()
        .filter(|message| message.role == "user")
        .take(2)
        .collect::<Vec<_>>();
    prop_assert_eq!(recent_users.len(), 2);
    for recent_user in recent_users {
        let retained = message_by_original_index(&aggressive, recent_user.original_index);
        prop_assert!(retained.is_some());
        assert_wire_message_unchanged(
            recent_user,
            retained.expect("recent user presence checked"),
        )?;
    }
    let aggressive_tokens = context
        .token_counter
        .count_request(&aggressive.clone().into_openai_request());
    prop_assert_eq!(aggressive_result.tokens_before, input_tokens);
    prop_assert_eq!(aggressive_result.tokens_after, aggressive_tokens);
    prop_assert!(aggressive_tokens <= input_tokens);

    let current_critical = original
        .messages
        .iter()
        .filter(|message| message.critical)
        .cloned()
        .collect::<Vec<_>>();
    prop_assert!(current_critical.len() >= system_count + 2);
    let mut ultra = original.clone();
    let ultra_result = runtime.block_on(UltraEngine::new().compress(&mut ultra, &context));
    assert_common_critical_invariants(&original, &ultra)?;
    for critical in current_critical {
        let retained = message_by_original_index(&ultra, critical.original_index);
        prop_assert!(retained.is_some());
        assert_wire_message_unchanged(
            &critical,
            retained.expect("critical message presence checked"),
        )?;
    }
    let ultra_tokens = context
        .token_counter
        .count_request(&ultra.clone().into_openai_request());
    prop_assert_eq!(ultra_result.tokens_before, input_tokens);
    prop_assert_eq!(ultra_result.tokens_after, ultra_tokens);
    prop_assert!(ultra_tokens <= input_tokens);

    Ok(())
}

proptest! {
    #![proptest_config(ProptestConfig::with_cases(64))]

    #[test]
    fn property_3_critical_message_preservation(
        system_count in 1usize..3,
        turns in 3usize..8,
        active_call_count in 1usize..4,
        case_id in any::<u64>(),
        words in prop::collection::vec(prose_word(), 40),
    ) {
        assert_critical_message_preservation(
            system_count,
            turns,
            active_call_count,
            case_id,
            words,
        )?;
    }
}
