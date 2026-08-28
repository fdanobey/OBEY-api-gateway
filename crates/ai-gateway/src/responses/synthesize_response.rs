//! Buffered Chat Completions → Responses API synthesizer.
//!
//! Converts a fully-buffered [`crate::models::openai::OpenAIResponse`] into a
//! [`ResponseObject`] for the `/v1/responses` front door. Non-streaming path
//! only; SSE synthesis is handled separately.
//!
//! Output item order is fixed: `reasoning` (if present) → assistant `message`
//! → `function_call` items (one per chat tool_call).

use crate::models::openai::{Message as ChatMessage, OpenAIResponse, Usage as ChatUsage};
use crate::responses::models::{
    IncompleteDetails, InputTokensDetails, OutputContentPart, OutputFunctionCall, OutputItem,
    OutputMessage, OutputReasoning, OutputTokensDetails, ReasoningConfig, ReasoningSummaryPart,
    ResponseError, ResponseObject, ResponsesUsage, TextConfig, ToolChoice, ToolDefinition,
};
use std::time::{SystemTime, UNIX_EPOCH};
use uuid::Uuid;

/// Request-side fields echoed back onto the synthesized response.
pub struct SynthesisContext<'a> {
    pub request_model: &'a str,
    pub request_instructions: Option<&'a str>,
    pub request_temperature: Option<f32>,
    pub request_top_p: Option<f32>,
    pub request_tools: &'a [ToolDefinition],
    pub request_tool_choice: Option<&'a ToolChoice>,
    pub request_metadata: Option<&'a serde_json::Map<String, serde_json::Value>>,
    pub request_store: bool,
    pub request_previous_response_id: Option<&'a str>,
    pub request_truncation: Option<&'a str>,
    pub request_text: Option<&'a TextConfig>,
    pub request_parallel_tool_calls: Option<bool>,
    pub request_reasoning: Option<&'a ReasoningConfig>,
}

/// Synthesize a Responses API response object from a buffered chat completion.
pub fn synthesize(chat: &OpenAIResponse, ctx: &SynthesisContext<'_>) -> ResponseObject {
    let choice = chat.choices.first();
    let finish_reason = choice.and_then(|c| c.finish_reason.as_deref());

    let (status, incomplete_details, error) = if choice.is_none() {
        // Sanitized error: no provider payload, no secrets.
        (
            "failed",
            None,
            Some(ResponseError {
                code: Some("no_choices".to_string()),
                message: Some("Upstream provider returned an empty response".to_string()),
                param: None,
                extra: Default::default(),
            }),
        )
    } else {
        let (status, details) = status_from_finish_reason(finish_reason);
        (status, details, None)
    };

    let mut output: Vec<OutputItem> = Vec::new();

    if let Some(choice) = choice {
        let message = &choice.message;

        if let Some(text) = reasoning_text(message) {
            output.push(OutputItem::Reasoning(OutputReasoning {
                id: new_id("rs_"),
                summary: vec![ReasoningSummaryPart {
                    r#type: Some("summary_text".to_string()),
                    text: Some(text),
                    extra: Default::default(),
                }],
                extra: Default::default(),
            }));
        }

        output.push(OutputItem::Message(OutputMessage {
            id: new_id("msg_"),
            role: "assistant".to_string(),
            status: "completed".to_string(),
            content: message_content_parts(message),
            extra: Default::default(),
        }));

        if let Some(tool_calls) = message.extra.get("tool_calls").and_then(|v| v.as_array()) {
            for tc in tool_calls {
                output.push(OutputItem::FunctionCall(OutputFunctionCall {
                    id: new_id("fc_"),
                    call_id: tc
                        .get("id")
                        .and_then(|v| v.as_str())
                        .unwrap_or_default()
                        .to_string(),
                    name: tc
                        .get("function")
                        .and_then(|f| f.get("name"))
                        .and_then(|v| v.as_str())
                        .unwrap_or_default()
                        .to_string(),
                    arguments: tc
                        .get("function")
                        .and_then(|f| f.get("arguments"))
                        .and_then(|v| v.as_str())
                        .unwrap_or_default()
                        .to_string(),
                    status: "completed".to_string(),
                    extra: Default::default(),
                }));
            }
        }
    }

    ResponseObject {
        id: new_id("resp_"),
        object: "response".to_string(),
        created_at: timestamp_or_now(chat.created),
        status: status.to_string(),
        error,
        incomplete_details,
        instructions: ctx.request_instructions.map(str::to_string),
        metadata: ctx.request_metadata.cloned(),
    model: {
        if chat.model.is_empty() {
            ctx.request_model.to_string()
        } else {
            chat.model.clone()
        }
    },
        output,
        parallel_tool_calls: ctx.request_parallel_tool_calls,
        previous_response_id: ctx.request_previous_response_id.map(str::to_string),
        reasoning: ctx.request_reasoning.cloned(),
        store: ctx.request_store,
        temperature: ctx.request_temperature,
        text: ctx.request_text.cloned(),
        tool_choice: ctx.request_tool_choice.cloned(),
        tools: ctx.request_tools.to_vec(),
        top_p: ctx.request_top_p,
        truncation: ctx.request_truncation.map(str::to_string),
        usage: Some(map_usage(&chat.usage)),
        extra: Default::default(),
    }
}

/// Map a chat `finish_reason` to a Responses status + incomplete details.
fn status_from_finish_reason(finish_reason: Option<&str>) -> (&'static str, Option<IncompleteDetails>) {
    match finish_reason {
        Some("length") => (
            "incomplete",
            Some(IncompleteDetails {
                reason: Some("max_output_tokens".to_string()),
            }),
        ),
        Some("content_filter") => (
            "incomplete",
            Some(IncompleteDetails {
                reason: Some("content_filter".to_string()),
            }),
        ),
        // `stop`, `tool_calls`, and anything unrecognized default to completed.
        _ => ("completed", None),
    }
}

/// Reasoning text from a chat message, if any.
///
/// DeepSeek-style providers use `reasoning_content`; others use `reasoning`.
fn reasoning_text(message: &ChatMessage) -> Option<String> {
    for key in ["reasoning_content", "reasoning"] {
        if let Some(text) = message.extra.get(key).and_then(|v| v.as_str()) {
            if !text.is_empty() {
                return Some(text.to_string());
            }
        }
    }
    None
}

/// Convert chat message content into Responses output content parts.
fn message_content_parts(message: &ChatMessage) -> Vec<OutputContentPart> {
    let mut parts = Vec::new();
    match &message.content {
        serde_json::Value::String(s) => {
            if !s.is_empty() {
                parts.push(OutputContentPart::OutputText {
                    text: s.clone(),
                    annotations: Some(Vec::new()),
                });
            }
        }
        serde_json::Value::Array(items) => {
            for item in items {
                match item.get("type").and_then(|t| t.as_str()) {
                    Some("text") => parts.push(OutputContentPart::OutputText {
                        text: item
                            .get("text")
                            .and_then(|v| v.as_str())
                            .unwrap_or_default()
                            .to_string(),
                        annotations: Some(Vec::new()),
                    }),
                    Some("refusal") => parts.push(OutputContentPart::Refusal {
                        refusal: item
                            .get("refusal")
                            .and_then(|v| v.as_str())
                            .unwrap_or_default()
                            .to_string(),
                    }),
                    _ => {}
                }
            }
        }
        _ => {}
    }
    // OpenAI-style assistants put the refusal in a dedicated field.
    if let Some(refusal) = message.extra.get("refusal").and_then(|v| v.as_str()) {
        if !refusal.is_empty() {
            parts.push(OutputContentPart::Refusal {
                refusal: refusal.to_string(),
            });
        }
    }
    parts
}

/// Map chat usage to Responses usage, extracting token detail breakdowns.
fn map_usage(usage: &ChatUsage) -> ResponsesUsage {
    let input_tokens_details = usage
        .extra
        .get("prompt_tokens_details")
        .and_then(|d| d.get("cached_tokens"))
        .and_then(|v| v.as_u64())
        .map(|cached_tokens| InputTokensDetails {
            cached_tokens,
            extra: Default::default(),
        });
    let output_tokens_details = usage
        .extra
        .get("completion_tokens_details")
        .and_then(|d| d.get("reasoning_tokens"))
        .and_then(|v| v.as_u64())
        .map(|reasoning_tokens| OutputTokensDetails {
            reasoning_tokens,
            extra: Default::default(),
        });
    ResponsesUsage {
        input_tokens: u64::from(usage.prompt_tokens),
        output_tokens: u64::from(usage.completion_tokens),
        total_tokens: u64::from(usage.total_tokens),
        input_tokens_details,
        output_tokens_details,
        extra: Default::default(),
    }
}

fn new_id(prefix: &str) -> String {
    format!("{prefix}{}", Uuid::new_v4().simple())
}

fn timestamp_or_now(created: i64) -> i64 {
    if created != 0 {
        created
    } else {
        SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .map(|d| d.as_secs() as i64)
            .unwrap_or_default()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn chat_from(value: serde_json::Value) -> OpenAIResponse {
        serde_json::from_value(value).expect("valid chat completion")
    }

    fn ctx() -> SynthesisContext<'static> {
        SynthesisContext {
            request_model: "request-alias",
            request_instructions: Some("be brief"),
            request_temperature: Some(0.5),
            request_top_p: Some(0.9),
            request_tools: &[],
            request_tool_choice: None,
            request_metadata: None,
            request_store: true,
            request_previous_response_id: None,
            request_truncation: None,
            request_text: None,
            request_parallel_tool_calls: None,
            request_reasoning: None,
        }
    }

    #[test]
    fn simple_text_response_synthesizes_output_text_message() {
        let chat = chat_from(serde_json::json!({
            "id": "chatcmpl-1",
            "model": "provider-model-x",
            "choices": [{
                "index": 0,
                "message": {"role": "assistant", "content": "Hello!"},
                "finish_reason": "stop"
            }],
            "usage": {"prompt_tokens": 10, "completion_tokens": 5, "total_tokens": 15}
        }));

        let resp = synthesize(&chat, &ctx());

        assert_eq!(resp.status, "completed");
        assert_eq!(resp.model, "provider-model-x");
        assert!(resp.id.starts_with("resp_"));
        assert!(resp.error.is_none());
        assert!(resp.incomplete_details.is_none());
        assert_eq!(resp.instructions.as_deref(), Some("be brief"));
        assert_eq!(resp.temperature, Some(0.5));
        assert_eq!(resp.top_p, Some(0.9));
        assert!(resp.store);

        assert_eq!(resp.output.len(), 1);
        let OutputItem::Message(msg) = &resp.output[0] else {
            panic!("expected message item");
        };
        assert!(msg.id.starts_with("msg_"));
        assert_eq!(msg.role, "assistant");
        assert_eq!(msg.status, "completed");
        assert_eq!(msg.content.len(), 1);
        match &msg.content[0] {
            OutputContentPart::OutputText { text, .. } => assert_eq!(text, "Hello!"),
            other => panic!("expected output_text, got {other:?}"),
        }

        let usage = resp.usage.expect("usage present");
        assert_eq!(usage.input_tokens, 10);
        assert_eq!(usage.output_tokens, 5);
        assert_eq!(usage.total_tokens, 15);
    }

    #[test]
    fn tool_calls_synthesize_function_call_items() {
        let chat = chat_from(serde_json::json!({
            "model": "provider-model-x",
            "choices": [{
                "index": 0,
                "message": {
                    "role": "assistant",
                    "content": null,
                    "tool_calls": [
                        {"id": "call_abc", "type": "function",
                         "function": {"name": "get_weather", "arguments": "{\"city\":\"Berlin\"}"}},
                        {"id": "call_def", "type": "function",
                         "function": {"name": "get_time", "arguments": "{}"}}
                    ]
                },
                "finish_reason": "tool_calls"
            }]
        }));

        let resp = synthesize(&chat, &ctx());

        assert_eq!(resp.status, "completed");
        assert_eq!(resp.output.len(), 3);
        assert!(matches!(&resp.output[0], OutputItem::Message(_)));

        let OutputItem::FunctionCall(fc1) = &resp.output[1] else {
            panic!("expected function_call");
        };
        assert!(fc1.id.starts_with("fc_"));
        assert_eq!(fc1.call_id, "call_abc");
        assert_eq!(fc1.name, "get_weather");
        assert_eq!(fc1.arguments, "{\"city\":\"Berlin\"}");
        assert_eq!(fc1.status, "completed");

        let OutputItem::FunctionCall(fc2) = &resp.output[2] else {
            panic!("expected function_call");
        };
        assert_eq!(fc2.call_id, "call_def");
        assert_eq!(fc2.name, "get_time");
    }

    #[test]
    fn finish_reason_maps_to_status_matrix() {
        for (finish_reason, status, reason) in [
            ("stop", "completed", None),
            ("tool_calls", "completed", None),
            ("length", "incomplete", Some("max_output_tokens")),
            ("content_filter", "incomplete", Some("content_filter")),
        ] {
            let chat = chat_from(serde_json::json!({
                "model": "m",
                "choices": [{
                    "index": 0,
                    "message": {"role": "assistant", "content": "x"},
                    "finish_reason": finish_reason
                }]
            }));
            let resp = synthesize(&chat, &ctx());
            assert_eq!(resp.status, status, "finish_reason={finish_reason}");
            assert_eq!(
                resp.incomplete_details.as_ref().and_then(|d| d.reason.clone()),
                reason.map(str::to_string),
                "finish_reason={finish_reason}"
            );
        }
    }

    #[test]
    fn usage_details_are_extracted() {
        let chat = chat_from(serde_json::json!({
            "model": "m",
            "choices": [{
                "index": 0,
                "message": {"role": "assistant", "content": "x"},
                "finish_reason": "stop"
            }],
            "usage": {
                "prompt_tokens": 100,
                "completion_tokens": 50,
                "total_tokens": 150,
                "prompt_tokens_details": {"cached_tokens": 40},
                "completion_tokens_details": {"reasoning_tokens": 25}
            }
        }));

        let usage = synthesize(&chat, &ctx()).usage.expect("usage");
        assert_eq!(usage.input_tokens_details.expect("input details").cached_tokens, 40);
        assert_eq!(
            usage.output_tokens_details.expect("output details").reasoning_tokens,
            25
        );
    }

    #[test]
    fn usage_details_absent_when_not_provided() {
        let chat = chat_from(serde_json::json!({
            "model": "m",
            "choices": [{
                "index": 0,
                "message": {"role": "assistant", "content": "x"},
                "finish_reason": "stop"
            }],
            "usage": {"prompt_tokens": 1, "completion_tokens": 1, "total_tokens": 2}
        }));

        let usage = synthesize(&chat, &ctx()).usage.expect("usage");
        assert!(usage.input_tokens_details.is_none());
        assert!(usage.output_tokens_details.is_none());
    }

    #[test]
    fn reasoning_content_becomes_first_output_item() {
        let chat = chat_from(serde_json::json!({
            "model": "deepseek-r1",
            "choices": [{
                "index": 0,
                "message": {
                    "role": "assistant",
                    "content": "The answer is 4.",
                    "reasoning_content": "thinking about 2+2"
                },
                "finish_reason": "stop"
            }]
        }));

        let resp = synthesize(&chat, &ctx());

        assert_eq!(resp.output.len(), 2);
        let OutputItem::Reasoning(reasoning) = &resp.output[0] else {
            panic!("expected reasoning first");
        };
        assert!(reasoning.id.starts_with("rs_"));
        assert_eq!(reasoning.summary.len(), 1);
        assert_eq!(reasoning.summary[0].text.as_deref(), Some("thinking about 2+2"));
        assert!(matches!(&resp.output[1], OutputItem::Message(_)));
    }

    #[test]
    fn empty_choices_yield_failed_status() {
        let chat = chat_from(serde_json::json!({
            "model": "m",
            "choices": []
        }));

        let resp = synthesize(&chat, &ctx());

        assert_eq!(resp.status, "failed");
        assert!(resp.output.is_empty());
        let error = resp.error.expect("error present");
        assert!(error.code.is_some());
        let message = error.message.expect("message present");
        assert!(!message.contains("sk-"), "no secrets in error");
        // Sanitized: no provider detail fields leak into the error object.
        assert!(error.extra.is_empty());
    }

    #[test]
    fn refusal_content_part_is_synthesized() {
        // Refusal as a typed content part.
        let chat = chat_from(serde_json::json!({
            "model": "m",
            "choices": [{
                "index": 0,
                "message": {
                    "role": "assistant",
                    "content": [
                        {"type": "refusal", "refusal": "I can't help with that."}
                    ]
                },
                "finish_reason": "stop"
            }]
        }));
        let resp = synthesize(&chat, &ctx());
        let OutputItem::Message(msg) = &resp.output[0] else {
            panic!("expected message");
        };
        match &msg.content[0] {
            OutputContentPart::Refusal { refusal } => {
                assert_eq!(refusal, "I can't help with that.");
            }
            other => panic!("expected refusal, got {other:?}"),
        }

        // Refusal as a dedicated assistant field.
        let chat = chat_from(serde_json::json!({
            "model": "m",
            "choices": [{
                "index": 0,
                "message": {
                    "role": "assistant",
                    "content": null,
                    "refusal": "Not allowed."
                },
                "finish_reason": "stop"
            }]
        }));
        let resp = synthesize(&chat, &ctx());
        let OutputItem::Message(msg) = &resp.output[0] else {
            panic!("expected message");
        };
        match &msg.content[0] {
            OutputContentPart::Refusal { refusal } => assert_eq!(refusal, "Not allowed."),
            other => panic!("expected refusal, got {other:?}"),
        }
    }

    #[test]
    fn mixed_text_and_refusal_parts_preserve_order() {
        let chat = chat_from(serde_json::json!({
            "model": "m",
            "choices": [{
                "index": 0,
                "message": {
                    "role": "assistant",
                    "content": [
                        {"type": "text", "text": "partial"},
                        {"type": "refusal", "refusal": "no more"}
                    ]
                },
                "finish_reason": "stop"
            }]
        }));
        let resp = synthesize(&chat, &ctx());
        let OutputItem::Message(msg) = &resp.output[0] else {
            panic!("expected message");
        };
        assert_eq!(msg.content.len(), 2);
        assert!(matches!(&msg.content[0], OutputContentPart::OutputText { text, .. } if text == "partial"));
        assert!(matches!(&msg.content[1], OutputContentPart::Refusal { refusal } if refusal == "no more"));
    }

    #[test]
    fn created_at_falls_back_to_now_when_missing() {
        let chat = chat_from(serde_json::json!({
            "model": "m",
            "created": 0,
            "choices": [{
                "index": 0,
                "message": {"role": "assistant", "content": "x"},
                "finish_reason": "stop"
            }]
        }));
        let before = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .unwrap()
            .as_secs() as i64;
        let resp = synthesize(&chat, &ctx());
        assert!(resp.created_at >= before, "created_at should be a fresh timestamp");

        let chat = chat_from(serde_json::json!({
            "model": "m",
            "created": 1234567890,
            "choices": [{
                "index": 0,
                "message": {"role": "assistant", "content": "x"},
                "finish_reason": "stop"
            }]
        }));
        assert_eq!(synthesize(&chat, &ctx()).created_at, 1234567890);
    }
}
