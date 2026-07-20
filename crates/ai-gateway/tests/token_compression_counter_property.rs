use ai_gateway::compression::token_counter::TokenCounter;
use ai_gateway::models::openai::{Message, OpenAIRequest};
use proptest::prelude::*;
use serde_json::{json, Map, Value};

fn message(role: &str, content: Value, extra: Map<String, Value>) -> Message {
    Message {
        role: role.to_owned(),
        content,
        extra,
    }
}

fn visible_value_tokens(counter: &TokenCounter, model: &str, value: &Value) -> u32 {
    match value {
        Value::Null => 0,
        Value::String(text) => counter.count_text(model, text),
        structured => counter.count_text(model, &structured.to_string()),
    }
}

fn visible_component_floor(counter: &TokenCounter, request: &OpenAIRequest) -> u32 {
    let mut components = vec![counter.count_text(&request.model, &request.model)];

    for message in &request.messages {
        components.push(counter.count_text(&request.model, &message.role));
        components.push(visible_value_tokens(
            counter,
            &request.model,
            &message.content,
        ));
        if !message.extra.is_empty() {
            components.push(counter.count_text(
                &request.model,
                &Value::Object(message.extra.clone()).to_string(),
            ));
        }
    }

    for (key, value) in &request.extra {
        if key != "tools" && key != "tool_choice" {
            components.push(counter.count_text(&request.model, key));
        }
        components.push(visible_value_tokens(counter, &request.model, value));
    }

    components.into_iter().fold(0u32, u32::saturating_add)
}

#[allow(clippy::too_many_arguments)]
fn complete_request(
    model: &str,
    system_text: &str,
    user_text: &str,
    assistant_text: &str,
    tool_result: &str,
    tool_name: &str,
    tool_description: &str,
    image_id: &str,
    schema_description: &str,
    request_tag: &str,
    stream: bool,
    temperature: f32,
    max_tokens: u32,
) -> OpenAIRequest {
    let call_id = format!("call_{request_tag}");
    let mut assistant_extra = Map::new();
    assistant_extra.insert(
        "tool_calls".to_owned(),
        json!([{
            "id": call_id,
            "type": "function",
            "function": {
                "name": tool_name,
                "arguments": json!({"query": user_text}).to_string()
            }
        }]),
    );

    let mut tool_extra = Map::new();
    tool_extra.insert("tool_call_id".to_owned(), json!(call_id));
    tool_extra.insert("name".to_owned(), json!(tool_name));

    let tools = json!([{
        "type": "function",
        "function": {
            "name": tool_name,
            "description": tool_description,
            "strict": true,
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {
                        "type": "string",
                        "description": schema_description
                    }
                },
                "required": ["query"],
                "additionalProperties": false
            }
        }
    }]);

    let mut extra = Map::new();
    extra.insert("tools".to_owned(), tools);
    extra.insert(
        "tool_choice".to_owned(),
        json!({"type": "function", "function": {"name": tool_name}}),
    );
    extra.insert(
        "response_format".to_owned(),
        json!({
            "type": "json_schema",
            "json_schema": {
                "name": format!("response_{request_tag}"),
                "strict": true,
                "schema": {
                    "type": "object",
                    "properties": {"answer": {"type": "string"}},
                    "required": ["answer"],
                    "additionalProperties": false
                }
            }
        }),
    );
    extra.insert(
        "metadata".to_owned(),
        json!({"request_tag": request_tag, "source": "property-test"}),
    );

    OpenAIRequest {
        model: model.to_owned(),
        messages: vec![
            message("system", json!(system_text), Map::new()),
            message(
                "user",
                json!([
                    {"type": "text", "text": user_text},
                    {
                        "type": "image_url",
                        "image_url": {
                            "url": format!("https://example.test/{image_id}.png"),
                            "detail": "low"
                        }
                    }
                ]),
                Map::new(),
            ),
            message("assistant", json!(assistant_text), assistant_extra),
            message(
                "tool",
                json!({"ok": true, "result": tool_result}),
                tool_extra,
            ),
        ],
        stream,
        temperature: Some(temperature),
        max_tokens: Some(max_tokens),
        extra,
    }
}

proptest! {
    #![proptest_config(ProptestConfig::with_cases(128))]

    #[test]
    fn complete_request_count_covers_visible_components_for_modern_and_fallback_models(
        modern_model in prop::sample::select(vec![
            "gpt-4o",
            "GPT-4O-mini",
            "o1-preview",
            "o3-mini",
            "chatgpt-4o-latest",
        ]),
        fallback_model in prop::sample::select(vec![
            "gpt-4",
            "gpt-3.5-turbo",
            "claude-3-5-sonnet",
            "unknown-provider-model",
        ]),
        system_text in ".{0,64}",
        user_text in ".{0,96}",
        assistant_text in ".{0,64}",
        tool_result in ".{0,96}",
        tool_name in "[a-z][a-z0-9_]{0,15}",
        tool_description in ".{0,64}",
        image_id in "[a-z0-9]{1,16}",
        schema_description in ".{0,64}",
        request_tag in "[a-z0-9]{1,16}",
        stream in any::<bool>(),
        temperature in 0.0f32..=2.0,
        max_tokens in 1u32..=16_384,
    ) {
        let counter = TokenCounter::new();

        for model in [modern_model, fallback_model] {
            let request = complete_request(
                model,
                &system_text,
                &user_text,
                &assistant_text,
                &tool_result,
                &tool_name,
                &tool_description,
                &image_id,
                &schema_description,
                &request_tag,
                stream,
                temperature,
                max_tokens,
            );
            let component_floor = visible_component_floor(&counter, &request);
            let first_count = counter.count_request(&request);
            let second_count = counter.count_request(&request);

            prop_assert!(component_floor > 0);
            prop_assert!(first_count > 0);
            prop_assert!(first_count >= component_floor);
            prop_assert_eq!(first_count, second_count);
        }
    }
}
