//! Property tests for the Responses API translation layer.
//!
//! Run with: `PROPTEST_CASES=64 cargo test -p ai-gateway --test responses_property_tests`
//!
//! Tests four invariants:
//! a) Text round-trip: no data loss through `translate()`
//! b) Item→message mapping round-trip for easy-input messages
//! c) Event-stream well-formedness under arbitrary SSE body splitting
//! d) ID / call_id stability for function-call and function-call-output items

use proptest::prelude::*;

use serde_json::{json, Value};

use ai_gateway::models::openai::OpenAIRequest;
use ai_gateway::responses::{
    translate, EasyInputContent, EasyInputMessage, FunctionCall, FunctionCallOutput,
    FunctionCallOutputContent, InputItem, ResponsesInput, ResponsesRequest,
    ResponsesSseEvent, ResponsesStreamTranslator, ResponsesUsage, TranslationContext,
    TypedInputItem,
};


// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

fn translation_ctx() -> TranslationContext<'static> {
    TranslationContext {
        resolved_model: "gpt-4o",
        model_supports_reasoning: false,
    }
}

fn make_request(input: ResponsesInput) -> ResponsesRequest {
    ResponsesRequest {
        model: "gpt-4o".to_string(),
        input,
        instructions: None,
        previous_response_id: None,
        store: false,
        metadata: None,
        temperature: None,
        top_p: None,
        max_output_tokens: None,
        truncation: None,
        parallel_tool_calls: None,
        reasoning: None,
        text: None,
        tools: Vec::new(),
        tool_choice: None,
        stream: false,
        stream_options: None,
        extra: serde_json::Map::new(),
    }
}

/// Event type tag string for assertion convenience.
fn event_type(event: &ResponsesSseEvent) -> &'static str {
    match event {
        ResponsesSseEvent::Created { .. } => "response.created",
        ResponsesSseEvent::InProgress { .. } => "response.in_progress",
        ResponsesSseEvent::Queued { .. } => "response.queued",
        ResponsesSseEvent::Completed { .. } => "response.completed",
        ResponsesSseEvent::Failed { .. } => "response.failed",
        ResponsesSseEvent::Incomplete { .. } => "response.incomplete",
        ResponsesSseEvent::OutputItemAdded { .. } => "response.output_item.added",
        ResponsesSseEvent::OutputItemDone { .. } => "response.output_item.done",
        ResponsesSseEvent::ContentPartAdded { .. } => "response.content_part.added",
        ResponsesSseEvent::ContentPartDone { .. } => "response.content_part.done",
        ResponsesSseEvent::OutputTextDelta { .. } => "response.output_text.delta",
        ResponsesSseEvent::OutputTextDone { .. } => "response.output_text.done",
        ResponsesSseEvent::RefusalDelta { .. } => "response.refusal.delta",
        ResponsesSseEvent::RefusalDone { .. } => "response.refusal.done",
        ResponsesSseEvent::FunctionCallArgumentsDelta { .. } => {
            "response.function_call_arguments.delta"
        }
        ResponsesSseEvent::FunctionCallArgumentsDone { .. } => {
            "response.function_call_arguments.done"
        }
        ResponsesSseEvent::ReasoningSummaryPartAdded { .. } => {
            "response.reasoning_summary_part.added"
        }
        ResponsesSseEvent::ReasoningSummaryTextDelta { .. } => {
            "response.reasoning_summary_text.delta"
        }
        ResponsesSseEvent::ReasoningTextDelta { .. } => "response.reasoning_text.delta",
    }
}

fn sequence_number(event: &ResponsesSseEvent) -> u64 {
    match event {
        ResponsesSseEvent::Created { sequence_number, .. }
        | ResponsesSseEvent::InProgress { sequence_number, .. }
        | ResponsesSseEvent::Queued { sequence_number, .. }
        | ResponsesSseEvent::Completed { sequence_number, .. }
        | ResponsesSseEvent::Failed { sequence_number, .. }
        | ResponsesSseEvent::Incomplete { sequence_number, .. }
        | ResponsesSseEvent::OutputItemAdded { sequence_number, .. }
        | ResponsesSseEvent::OutputItemDone { sequence_number, .. }
        | ResponsesSseEvent::ContentPartAdded { sequence_number, .. }
        | ResponsesSseEvent::ContentPartDone { sequence_number, .. }
        | ResponsesSseEvent::OutputTextDelta { sequence_number, .. }
        | ResponsesSseEvent::OutputTextDone { sequence_number, .. }
        | ResponsesSseEvent::RefusalDelta { sequence_number, .. }
        | ResponsesSseEvent::RefusalDone { sequence_number, .. }
        | ResponsesSseEvent::FunctionCallArgumentsDelta { sequence_number, .. }
        | ResponsesSseEvent::FunctionCallArgumentsDone { sequence_number, .. }
        | ResponsesSseEvent::ReasoningSummaryPartAdded { sequence_number, .. }
        | ResponsesSseEvent::ReasoningSummaryTextDelta { sequence_number, .. }
        | ResponsesSseEvent::ReasoningTextDelta { sequence_number, .. } => *sequence_number,
    }
}

/// Extract delta text from an `OutputTextDelta` event.
fn delta_text(event: &ResponsesSseEvent) -> Option<&str> {
    match event {
        ResponsesSseEvent::OutputTextDelta { delta, .. } => Some(delta),
        _ => None,
    }
}

/// Build a chat-completions SSE body that produces the given text content.
fn build_sse_body(text: &str) -> String {
    let mut body = String::new();
    body.push_str(&format!(
        "data: {}\n\n",
        serde_json::to_string(&json!({
            "id": "chatcmpl-test",
            "choices": [{"index": 0, "delta": {"role": "assistant"}}]
        }))
        .unwrap()
    ));
    body.push_str(&format!(
        "data: {}\n\n",
        serde_json::to_string(&json!({
            "id": "chatcmpl-test",
            "choices": [{"index": 0, "delta": {"content": text}}]
        }))
        .unwrap()
    ));
    body.push_str(&format!(
        "data: {}\n\n",
        serde_json::to_string(&json!({
            "id": "chatcmpl-test",
            "choices": [{"index": 0, "delta": {}, "finish_reason": "stop"}]
        }))
        .unwrap()
    ));
    body.push_str("data: [DONE]\n\n");
    body
}

/// Parse a raw SSE fragment, extracting complete `data: {json}` lines.
///
/// Returns the parsed JSON values and the leftover (incomplete) buffer.
fn parse_sse_fragment(
    buffer: &mut String,
    fragment: &str,
) -> Vec<Value> {
    buffer.push_str(fragment);
    let mut chunks = Vec::new();
    while let Some(pos) = buffer.find("\n\n") {
        let line: String = buffer[..pos].to_string();
        *buffer = buffer[pos + 2..].to_string();
        if let Some(json_str) = line.strip_prefix("data: ") {
            let trimmed = json_str.trim();
            if trimmed == "[DONE]" {
                continue;
            }
            if let Ok(val) = serde_json::from_str::<Value>(trimmed) {
                chunks.push(val);
            }
        }
    }
    chunks
}

/// Split the SSE body at `split_index`, simulating arbitrary TCP chunk
/// boundaries, and return all parseable chat-completion chunks.
fn split_and_parse_sse(body: &str, split_index: usize) -> Vec<Value> {
    let (first, second) = body.split_at(split_index);
    let mut buffer = String::new();
    let mut all_chunks = Vec::new();
    all_chunks.extend(parse_sse_fragment(&mut buffer, first));
    all_chunks.extend(parse_sse_fragment(&mut buffer, second));
    all_chunks
}

/// Feed all chat chunks through a fresh translator, collect all events
/// (including terminal events from `finish`).
fn translate_stream(chunks: &[Value]) -> Vec<ResponsesSseEvent> {
    let mut translator = ResponsesStreamTranslator::new("gpt-4o", None);
    let mut events = Vec::new();
    for chunk in chunks {
        events.extend(translator.feed_chunk(chunk));
    }
    events.extend(translator.finish(Some(ResponsesUsage {
        input_tokens: 5,
        output_tokens: 2,
        total_tokens: 7,
        ..Default::default()
    })));
    events
}

// ---------------------------------------------------------------------------
// Property tests
// ---------------------------------------------------------------------------

proptest! {
    #![proptest_config(ProptestConfig::with_cases(64))]

    // (a) Text round-trip: no data loss
    #[test]
    fn prop_text_roundtrip_no_data_loss(
        text in proptest::string::string_regex("[a-zA-Z0-9 .,!?'\";:-]{1,500}")
            .expect("text regex must compile"),
    ) {
        let req = make_request(ResponsesInput::Text(text.clone()));
        let result: OpenAIRequest = translate(&req, None, &translation_ctx())
            .expect("translation must succeed for valid text input");

        prop_assert_eq!(result.messages.len(), 1, "exactly one user message expected");
        prop_assert_eq!(&result.messages[0].role, "user");

        let content = &result.messages[0].content;
        let recovered = match content {
            Value::String(s) => s.clone(),
            _ => content.to_string(),
        };
        prop_assert_eq!(&recovered, &text, "text must survive round-trip without data loss");
    }

    // (b) Item→message mapping round-trip
    #[test]
    fn prop_item_message_mapping_roundtrip(
        text in proptest::string::string_regex("[a-zA-Z0-9 .,!?'-]{1,200}")
            .expect("text regex must compile"),
        role in prop::sample::select(vec!["user", "assistant"]),
    ) {
        let items = vec![InputItem::Easy(EasyInputMessage {
            content: EasyInputContent::Text(text.clone()),
            role: role.to_string(),
            phase: None,
            extra: serde_json::Map::new(),
        })];
        let req = make_request(ResponsesInput::Items(items));
        let result = translate(&req, None, &translation_ctx())
            .expect("translation must succeed for valid easy-input message");

        prop_assert_eq!(result.messages.len(), 1);
        prop_assert_eq!(&result.messages[0].role, role);

        let content = &result.messages[0].content;
        let recovered = match content {
            Value::String(s) => s.clone(),
            _ => content.to_string(),
        };
        prop_assert_eq!(&recovered, &text);
    }

    // (c) Event-stream well-formedness under arbitrary chunk splitting
    #[test]
    fn prop_event_stream_well_formed_under_arbitrary_split(
        split_index in 0usize..2000,
    ) {
        let text = "Hello world from the gateway";
        let body = build_sse_body(text);
        let max = body.len();
        let split = split_index.min(max);
        let chunks = split_and_parse_sse(&body, split);
        let events = translate_stream(&chunks);

        // Must contain at least the core event types.
        let types: Vec<&str> = events.iter().map(event_type).collect();
        let has_created = types.iter().any(|&t| t == "response.created");
        let has_delta = types.iter().any(|&t| t == "response.output_text.delta");
        let has_completed = types.iter().any(|&t| t == "response.completed");

        prop_assert!(has_created, "stream must contain response.created; got {:?}", types);
        prop_assert!(has_delta, "stream must contain response.output_text.delta; got {:?}", types);
        prop_assert!(has_completed, "stream must contain response.completed; got {:?}", types);

        // Sequence numbers must be strictly increasing.
        let seqs: Vec<u64> = events.iter().map(sequence_number).collect();
        for pair in seqs.windows(2) {
            prop_assert!(
                pair[1] > pair[0],
                "sequence numbers must be strictly increasing: {:?}",
                seqs
            );
        }

        // Concatenated delta text must equal the original.
        let assembled: String = events
            .iter()
            .filter_map(delta_text)
            .collect::<Vec<&str>>()
            .join("");
        prop_assert_eq!(
            &assembled, text,
            "concatenated delta text must equal original: {:?} vs {:?}",
            assembled, text
        );
    }

    // (d) ID / call_id stability for function-call and function-call-output
    #[test]
    fn prop_function_call_id_stability(
        call_id in proptest::string::string_regex("[a-zA-Z0-9_]{1,32}")
            .expect("call_id regex must compile"),
        name in proptest::string::string_regex("[a-z_]{1,20}")
            .expect("function name regex must compile"),
    ) {
        // --- FunctionCall → assistant message with tool_calls ---
        let fc_req = make_request(ResponsesInput::Items(vec![InputItem::Typed(
            TypedInputItem::FunctionCall(FunctionCall {
                id: None,
                call_id: call_id.clone(),
                name: name.clone(),
                arguments: r#"{"city":"Paris"}"#.to_string(),
                status: None,
                extra: serde_json::Map::new(),
            }),
        )]));
        let fc_result = translate(&fc_req, None, &translation_ctx())
            .expect("function_call translation must succeed");

        prop_assert_eq!(fc_result.messages.len(), 1);
        prop_assert_eq!(&fc_result.messages[0].role, "assistant");

        let tool_calls = fc_result.messages[0]
            .extra
            .get("tool_calls")
            .and_then(|v| v.as_array())
            .expect("assistant message must have tool_calls");
        prop_assert_eq!(tool_calls.len(), 1);
        prop_assert_eq!(
            &tool_calls[0]["id"],
            &json!(call_id),
            "tool_call id must equal the original call_id"
        );

        // --- FunctionCallOutput → tool message with tool_call_id ---
        let fco_req = make_request(ResponsesInput::Items(vec![InputItem::Typed(
            TypedInputItem::FunctionCallOutput(FunctionCallOutput {
                call_id: call_id.clone(),
                output: FunctionCallOutputContent::Text("sunny".to_string()),
                id: None,
                status: None,
                extra: serde_json::Map::new(),
            }),
        )]));
        let fco_result = translate(&fco_req, None, &translation_ctx())
            .expect("function_call_output translation must succeed");

        prop_assert_eq!(fco_result.messages.len(), 1);
        prop_assert_eq!(&fco_result.messages[0].role, "tool");
        prop_assert_eq!(
            fco_result.messages[0].extra.get("tool_call_id"),
            Some(&json!(call_id)),
            "tool_call_id must equal the original call_id"
        );
    }
}

// ---------------------------------------------------------------------------
// Unit tests (non-proptest) for deterministic edge cases of the split parser
// ---------------------------------------------------------------------------

#[test]
fn split_at_zero_recovers_all_chunks() {
    let body = build_sse_body("Hello world from the gateway");
    let chunks = split_and_parse_sse(&body, 0);
    assert_eq!(chunks.len(), 3, "should parse 3 chat chunks (role, content, finish)");
}

#[test]
fn split_at_end_recovers_all_chunks() {
    let body = build_sse_body("Hello world from the gateway");
    let chunks = split_and_parse_sse(&body, body.len());
    assert_eq!(chunks.len(), 3);
}

#[test]
fn split_at_middle_recovers_all_chunks() {
    let body = build_sse_body("Hello world from the gateway");
    let mid = body.len() / 2;
    let chunks = split_and_parse_sse(&body, mid);
    assert_eq!(chunks.len(), 3);
}

#[test]
fn stream_translation_produces_expected_events() {
    let text = "Hello world from the gateway";
    let body = build_sse_body(text);
    let chunks = split_and_parse_sse(&body, body.len() / 2);
    let events = translate_stream(&chunks);

    let types: Vec<&str> = events.iter().map(event_type).collect();
    assert!(
        types.iter().any(|&t| t == "response.created"),
        "must contain response.created"
    );
    assert!(
        types.iter().any(|&t| t == "response.output_text.delta"),
        "must contain response.output_text.delta"
    );
    assert!(
        types.iter().any(|&t| t == "response.completed"),
        "must contain response.completed"
    );

    let assembled: String = events
        .iter()
        .filter_map(delta_text)
        .collect::<Vec<&str>>()
        .join("");
    assert_eq!(assembled, text);
}
