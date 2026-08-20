use super::engines::{aggressive::AggressiveEngine, lite::LiteEngine, standard::StandardEngine};
use super::{CompressiblePayload, CompressionContext, CompressionEngine};
use crate::models::openai::{Message, OpenAIRequest};
use proptest::prelude::*;
use proptest::test_runner::TestCaseResult;
use serde_json::{Map, Value};

fn prose_word() -> impl Strategy<Value = String> {
    proptest::string::string_regex("[a-z]{1,12}").expect("prose word regex must compile")
}

fn message(role: &str, content: String) -> Message {
    Message {
        role: role.to_owned(),
        content: Value::String(content),
        extra: Map::new(),
    }
}

fn generated_conversation(turns: usize, case_id: u64, words: &[String]) -> OpenAIRequest {
    let mut messages = vec![message(
        "system",
        format!(
            "System policy for case {case_id} must remain  exactly as written for every response."
        ),
    )];

    for turn in 0..turns {
        messages.push(message(
            "user",
            format!(
                "Case {case_id} turn {turn} {}: Could you please  actually take a look at this very detailed ordinary paragraph in order to make sure that it is able to remain useful due to the fact that it is really important?",
                words[turn * 2]
            ),
        ));
        messages.push(message(
            "assistant",
            format!(
                "Case {case_id} turn {turn} {}: I think this is  actually a very detailed response written in order to help, and I hope this helps.",
                words[turn * 2 + 1]
            ),
        ));
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

fn assert_progressive_aging(turns: usize, case_id: u64, words: Vec<String>) -> TestCaseResult {
    let mut context = CompressionContext::new("gpt-4o", "property-test");
    context.context_window = 0;
    context.target_token_budget = None;
    prop_assert_eq!(context.context_window, 0);
    prop_assert_eq!(context.target_token_budget, None);

    let request = generated_conversation(turns, case_id, &words);
    let input_tokens = context.token_counter.count_request(&request);
    let original = CompressiblePayload::from_openai_request(request);
    prop_assert!(original
        .messages
        .iter()
        .filter_map(|message| message.content.as_text())
        .all(|text| context.protection_scanner.scan(text).is_empty()));

    let mut lite_expected = original.clone();
    for message in &mut lite_expected.messages {
        message.cache_protected = !(3..=6).contains(&message.age);
    }

    let mut standard_expected = original.clone();
    for message in &mut standard_expected.messages {
        message.cache_protected = message.is_system() || message.age < 7;
    }

    let mut output = original.clone();
    let runtime = tokio::runtime::Builder::new_current_thread()
        .build()
        .expect("property-test runtime must build");
    let result = runtime.block_on(async {
        LiteEngine::new()
            .compress(&mut lite_expected, &context)
            .await;
        StandardEngine::new()
            .compress(&mut standard_expected, &context)
            .await;
        AggressiveEngine::new()
            .compress(&mut output, &context)
            .await
    });

    prop_assert_eq!(output.messages.len(), original.messages.len());
    for (position, (before, after)) in original
        .messages
        .iter()
        .zip(output.messages.iter())
        .enumerate()
    {
        prop_assert_eq!(after.role.as_str(), before.role.as_str());
        prop_assert_eq!(after.original_index, before.original_index);

        if before.is_system() || before.age <= 2 {
            prop_assert_eq!(&after.content, &before.content);
        } else if before.age <= 6 {
            prop_assert_eq!(&after.content, &lite_expected.messages[position].content);
        } else {
            prop_assert_eq!(
                &after.content,
                &standard_expected.messages[position].content
            );
        }
    }

    let system_before = original
        .messages
        .iter()
        .find(|message| message.is_system())
        .expect("generated conversation must contain a system message");
    let system_after = output
        .messages
        .iter()
        .find(|message| message.is_system())
        .expect("compressed conversation must retain its system message");
    prop_assert_eq!(&system_after.content, &system_before.content);

    let recent_user_positions = original
        .messages
        .iter()
        .enumerate()
        .rev()
        .filter(|(_, message)| message.role == "user")
        .take(2)
        .map(|(position, _)| position)
        .collect::<Vec<_>>();
    prop_assert_eq!(recent_user_positions.len(), 2);
    for position in recent_user_positions {
        prop_assert_eq!(output.messages[position].role.as_str(), "user");
        prop_assert_eq!(
            &output.messages[position].content,
            &original.messages[position].content
        );
    }

    let output_tokens = context
        .token_counter
        .count_request(&output.clone().into_openai_request());
    prop_assert_eq!(result.tokens_before, input_tokens);
    prop_assert_eq!(result.tokens_after, output_tokens);
    prop_assert!(output_tokens <= input_tokens);

    Ok(())
}

proptest! {
    #![proptest_config(ProptestConfig::with_cases(64))]

    #[test]
    fn property_11_progressive_aging_policy(
        turns in 8usize..17,
        case_id in any::<u64>(),
        words in prop::collection::vec(prose_word(), 32),
    ) {
        assert_progressive_aging(turns, case_id, words)?;
    }
}
