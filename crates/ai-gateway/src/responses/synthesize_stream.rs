//! Streaming event synthesizer: chat-completions SSE chunks → Responses events.
//!
//! Converts a parsed chat-completions streaming chunk sequence into the
//! canonical Responses API event stream:
//! `response.created → response.in_progress → output_item.added →
//! content_part.added → output_text.delta* → output_text.done →
//! content_part.done → output_item.done → response.completed`.
//!
//! Input is parsed chat SSE JSON (not raw bytes); arbitrary chunk splits
//! (partial tool_call argument fragments) are accumulated internally. Usage
//! arrives via [`ResponsesStreamTranslator::finish`] from the final usage
//! chunk or `[DONE]` sentinel handling in the caller.

use std::collections::HashMap;
use std::time::{SystemTime, UNIX_EPOCH};

use serde_json::Value;
use uuid::Uuid;

use super::models::{
    IncompleteDetails, OutputContentPart, OutputFunctionCall, OutputItem, OutputMessage,
    OutputReasoning, ReasoningSummaryPart, ResponseObject, ResponsesSseEvent, ResponsesUsage,
};

/// Per-tool-call accumulation state, keyed by the chat `tool_calls[].index`.
#[derive(Clone)]
struct FunctionCallState {
    item_id: String,
    call_id: String,
    name: String,
    arguments: String,
    output_index: u64,
}

/// Ordered output-item slot so `response.output` preserves arrival order.
enum ItemSlot {
    Reasoning,
    Message,
    FunctionCall { tool_index: usize },
}

/// Stateful translator from chat-completions stream chunks to Responses events.
pub struct ResponsesStreamTranslator {
    response_id: String,
    model: String,
    sequence_number: u64,
    created_ts: i64,
    first_chunk_processed: bool,
    instructions: Option<String>,

    ordered_items: Vec<ItemSlot>,
    current_message_item_id: Option<String>,
    message_output_index: Option<u64>,
    text_part_index: Option<u64>,
    refusal_part_index: Option<u64>,
    next_content_index: u64,
    text_buffer: String,
    refusal_buffer: String,

    reasoning_emitted: bool,
    reasoning_item_id: Option<String>,
    reasoning_output_index: Option<u64>,
    reasoning_buffer: String,

    current_function_call_items: HashMap<usize, FunctionCallState>,

    finish_reason: Option<String>,
}

impl ResponsesStreamTranslator {
    pub fn new(model: &str, request_instructions: Option<&str>) -> Self {
        Self {
            response_id: format!("resp_{}", Uuid::new_v4().simple()),
            model: model.to_string(),
            sequence_number: 0,
            created_ts: SystemTime::now()
                .duration_since(UNIX_EPOCH)
                .map(|d| d.as_secs() as i64)
                .unwrap_or_default(),
            first_chunk_processed: false,
            instructions: request_instructions.map(str::to_string),
            ordered_items: Vec::new(),
            current_message_item_id: None,
            message_output_index: None,
            text_part_index: None,
            refusal_part_index: None,
            next_content_index: 0,
            text_buffer: String::new(),
            refusal_buffer: String::new(),
            reasoning_emitted: false,
            reasoning_item_id: None,
            reasoning_output_index: None,
            reasoning_buffer: String::new(),
            current_function_call_items: HashMap::new(),
            finish_reason: None,
        }
    }

    /// Feed a parsed chat SSE chunk; returns 0+ synthesized Responses events.
    pub fn feed_chunk(&mut self, chunk: &Value) -> Vec<ResponsesSseEvent> {
        let mut events = Vec::new();

        let choice = chunk
            .get("choices")
            .and_then(Value::as_array)
            .and_then(|choices| choices.first().cloned());
        let Some(choice) = choice else {
            return events;
        };

        if let Some(reason) = choice.get("finish_reason").and_then(Value::as_str) {
            self.finish_reason = Some(reason.to_string());
        }

        // Lazy initialization: lifecycle events on the first choice-bearing
        // chunk, never at construction.
        if !self.first_chunk_processed {
            self.first_chunk_processed = true;
            events.push(ResponsesSseEvent::Created {
                sequence_number: self.next_seq(),
                response: self.base_response("in_progress"),
            });
            events.push(ResponsesSseEvent::InProgress {
                sequence_number: self.next_seq(),
                response: self.base_response("in_progress"),
            });
        }

        if let Some(delta) = choice.get("delta") {
            self.feed_delta(delta, &mut events);
        }

        events
    }

    /// Finalize the stream, emitting terminal events with usage.
    pub fn finish(mut self, usage: Option<ResponsesUsage>) -> Vec<ResponsesSseEvent> {
        let mut events = Vec::new();

        if !self.first_chunk_processed {
            self.first_chunk_processed = true;
            events.push(ResponsesSseEvent::Created {
                sequence_number: self.next_seq(),
                response: self.base_response("in_progress"),
            });
            events.push(ResponsesSseEvent::InProgress {
                sequence_number: self.next_seq(),
                response: self.base_response("in_progress"),
            });
        }

        // Text/refusal part completion.
        if let (Some(item_id), Some(output_index), Some(content_index)) = (
            self.current_message_item_id.clone(),
            self.message_output_index,
            self.text_part_index,
        ) {
            if !self.text_buffer.is_empty() {
                events.push(ResponsesSseEvent::OutputTextDone {
                    sequence_number: self.next_seq(),
                    item_id: item_id.clone(),
                    output_index,
                    content_index,
                    text: self.text_buffer.clone(),
                });
                events.push(ResponsesSseEvent::ContentPartDone {
                    sequence_number: self.next_seq(),
                    item_id,
                    output_index,
                    content_index,
                    part: OutputContentPart::OutputText {
                        text: self.text_buffer.clone(),
                        annotations: None,
                    },
                });
            }
        }
        if let (Some(item_id), Some(output_index), Some(content_index)) = (
            self.current_message_item_id.clone(),
            self.message_output_index,
            self.refusal_part_index,
        ) {
            if !self.refusal_buffer.is_empty() {
                events.push(ResponsesSseEvent::RefusalDone {
                    sequence_number: self.next_seq(),
                    item_id: item_id.clone(),
                    output_index,
                    content_index,
                    refusal: self.refusal_buffer.clone(),
                });
                events.push(ResponsesSseEvent::ContentPartDone {
                    sequence_number: self.next_seq(),
                    item_id,
                    output_index,
                    content_index,
                    part: OutputContentPart::Refusal {
                        refusal: self.refusal_buffer.clone(),
                    },
                });
            }
        }

        // Per-item completion, in arrival order.
        let ordered_items = std::mem::take(&mut self.ordered_items);
        for slot in &ordered_items {
            match slot {
                ItemSlot::Reasoning => {
                    if let (Some(item_id), Some(output_index)) =
                        (self.reasoning_item_id.clone(), self.reasoning_output_index)
                    {
                        events.push(ResponsesSseEvent::OutputItemDone {
                            sequence_number: self.next_seq(),
                            output_index,
                            item: OutputItem::Reasoning(self.build_reasoning_item(&item_id)),
                        });
                    }
                }
                ItemSlot::Message => {
                    if let (Some(item_id), Some(output_index)) = (
                        self.current_message_item_id.clone(),
                        self.message_output_index,
                    ) {
                        events.push(ResponsesSseEvent::OutputItemDone {
                            sequence_number: self.next_seq(),
                            output_index,
                            item: OutputItem::Message(self.build_message_item(&item_id)),
                        });
                    }
                }
ItemSlot::FunctionCall { tool_index } => {
if let Some(state) = self.current_function_call_items.get(tool_index).cloned() {
events.push(ResponsesSseEvent::FunctionCallArgumentsDone {
sequence_number: self.next_seq(),
item_id: state.item_id.clone(),
output_index: state.output_index,
arguments: state.arguments.clone(),
});
events.push(ResponsesSseEvent::OutputItemDone {
sequence_number: self.next_seq(),
output_index: state.output_index,
item: OutputItem::FunctionCall(OutputFunctionCall {
id: state.item_id.clone(),
call_id: state.call_id.clone(),
name: state.name.clone(),
arguments: state.arguments.clone(),
status: "completed".to_string(),
extra: Default::default(),
}),
});
}
}
            }
        }
        self.ordered_items = ordered_items;

        // Terminal event with the full response object and usage.
        let (status, incomplete_details) = self.terminal_status();
        let response = self.final_response(status, incomplete_details, usage);
        let terminal_sequence = self.next_seq();
        let terminal = if response.status == "incomplete" {
            ResponsesSseEvent::Incomplete {
                sequence_number: terminal_sequence,
                response,
            }
        } else {
            ResponsesSseEvent::Completed {
                sequence_number: terminal_sequence,
                response,
            }
        };
        events.push(terminal);

        events
    }

    // -- Delta dispatch ------------------------------------------------------

    fn feed_delta(&mut self, delta: &Value, events: &mut Vec<ResponsesSseEvent>) {
        // Reasoning must be handled before text (reasoning precedes answers).
        let reasoning = delta
            .get("reasoning_content")
            .or_else(|| delta.get("reasoning"))
            .and_then(Value::as_str);
        if let Some(reasoning) = reasoning.filter(|r| !r.is_empty()) {
            self.handle_reasoning(reasoning, events);
        }

        if let Some(content) = delta.get("content") {
            match content {
                Value::String(text) if !text.is_empty() => self.handle_text(text, events),
                Value::Array(parts) => {
                    for part in parts {
                        let kind = part.get("type").and_then(Value::as_str);
                        match kind {
                            Some("refusal") => {
                                if let Some(text) =
                                    part.get("refusal").and_then(Value::as_str).filter(|s| !s.is_empty())
                                {
                                    self.handle_refusal(text, events);
                                }
                            }
                            _ => {
                                if let Some(text) =
                                    part.get("text").and_then(Value::as_str).filter(|s| !s.is_empty())
                                {
                                    self.handle_text(text, events);
                                }
                            }
                        }
                    }
                }
                _ => {}
            }
        }

        if let Some(refusal) = delta
            .get("refusal")
            .and_then(Value::as_str)
            .filter(|r| !r.is_empty())
        {
            self.handle_refusal(refusal, events);
        }

        if let Some(tool_calls) = delta.get("tool_calls").and_then(Value::as_array) {
            for tool_call in tool_calls {
                self.handle_tool_call(tool_call, events);
            }
        }
    }

    fn handle_reasoning(&mut self, text: &str, events: &mut Vec<ResponsesSseEvent>) {
        if !self.reasoning_emitted {
            self.reasoning_emitted = true;
            let item_id = format!("rs_{}", Uuid::new_v4().simple());
            let output_index = self.ordered_items.len() as u64;
            self.ordered_items.push(ItemSlot::Reasoning);
            self.reasoning_item_id = Some(item_id.clone());
            self.reasoning_output_index = Some(output_index);
            events.push(ResponsesSseEvent::OutputItemAdded {
                sequence_number: self.next_seq(),
                output_index,
                item: OutputItem::Reasoning(OutputReasoning {
                    id: item_id.clone(),
                    summary: Vec::new(),
                    extra: Default::default(),
                }),
            });
            events.push(ResponsesSseEvent::ReasoningSummaryPartAdded {
                sequence_number: self.next_seq(),
                item_id: item_id.clone(),
                output_index,
                summary_index: 0,
                part: ReasoningSummaryPart {
                    r#type: Some("summary_text".to_string()),
                    text: Some(String::new()),
                    extra: Default::default(),
                },
            });
        }

        self.reasoning_buffer.push_str(text);
        let item_id = self.reasoning_item_id.clone().unwrap_or_default();
        let output_index = self.reasoning_output_index.unwrap_or_default();
        events.push(ResponsesSseEvent::ReasoningTextDelta {
            sequence_number: self.next_seq(),
            item_id: item_id.clone(),
            output_index,
            delta: text.to_string(),
        });
        events.push(ResponsesSseEvent::ReasoningSummaryTextDelta {
            sequence_number: self.next_seq(),
            item_id,
            output_index,
            summary_index: 0,
            delta: text.to_string(),
        });
    }

    fn handle_text(&mut self, text: &str, events: &mut Vec<ResponsesSseEvent>) {
        self.ensure_message_item(events);
        if self.text_part_index.is_none() {
            let content_index = self.next_content_index;
            self.next_content_index += 1;
            let item_id = self.current_message_item_id.clone().unwrap_or_default();
            let output_index = self.message_output_index.unwrap_or_default();
            events.push(ResponsesSseEvent::ContentPartAdded {
                sequence_number: self.next_seq(),
                item_id,
                output_index,
                content_index,
                part: OutputContentPart::OutputText {
                    text: String::new(),
                    annotations: None,
                },
            });
            self.text_part_index = Some(content_index);
        }
        let content_index = self.text_part_index.unwrap_or_default();

        self.text_buffer.push_str(text);
        events.push(ResponsesSseEvent::OutputTextDelta {
            sequence_number: self.next_seq(),
            item_id: self.current_message_item_id.clone().unwrap_or_default(),
            output_index: self.message_output_index.unwrap_or_default(),
            content_index,
            delta: text.to_string(),
        });
    }

    fn handle_refusal(&mut self, text: &str, events: &mut Vec<ResponsesSseEvent>) {
        self.ensure_message_item(events);
        if self.refusal_part_index.is_none() {
            let content_index = self.next_content_index;
            self.next_content_index += 1;
            let item_id = self.current_message_item_id.clone().unwrap_or_default();
            let output_index = self.message_output_index.unwrap_or_default();
            events.push(ResponsesSseEvent::ContentPartAdded {
                sequence_number: self.next_seq(),
                item_id,
                output_index,
                content_index,
                part: OutputContentPart::Refusal {
                    refusal: String::new(),
                },
            });
            self.refusal_part_index = Some(content_index);
        }
        let content_index = self.refusal_part_index.unwrap_or_default();

        self.refusal_buffer.push_str(text);
        events.push(ResponsesSseEvent::RefusalDelta {
            sequence_number: self.next_seq(),
            item_id: self.current_message_item_id.clone().unwrap_or_default(),
            output_index: self.message_output_index.unwrap_or_default(),
            content_index,
            delta: text.to_string(),
        });
    }

    fn handle_tool_call(&mut self, tool_call: &Value, events: &mut Vec<ResponsesSseEvent>) {
        let index = tool_call.get("index").and_then(Value::as_u64).unwrap_or(0) as usize;
        let id = tool_call.get("id").and_then(Value::as_str);
        let function = tool_call.get("function");
        let name = function
            .and_then(|f| f.get("name"))
            .and_then(Value::as_str);
        let arguments = function
            .and_then(|f| f.get("arguments"))
            .and_then(Value::as_str);

        if !self.current_function_call_items.contains_key(&index) {
            let item_id = format!("fc_{}", Uuid::new_v4().simple());
            let call_id = id
                .map(str::to_string)
                .unwrap_or_else(|| format!("call_{}", Uuid::new_v4().simple()));
            let name = name.unwrap_or_default().to_string();
            let output_index = self.ordered_items.len() as u64;
            self.ordered_items.push(ItemSlot::FunctionCall { tool_index: index });
            self.current_function_call_items.insert(
                index,
                FunctionCallState {
                    item_id: item_id.clone(),
                    call_id: call_id.clone(),
                    name: name.clone(),
                    arguments: String::new(),
                    output_index,
                },
            );
            events.push(ResponsesSseEvent::OutputItemAdded {
                sequence_number: self.next_seq(),
                output_index,
                item: OutputItem::FunctionCall(OutputFunctionCall {
                    id: item_id,
                    call_id,
                    name,
                    arguments: String::new(),
                    status: "in_progress".to_string(),
                    extra: Default::default(),
                }),
            });
        } else if let Some(name) = name {
            if let Some(state) = self.current_function_call_items.get_mut(&index) {
                state.name = name.to_string();
            }
        }

        if let Some(fragment) = arguments.filter(|a| !a.is_empty()) {
            let (item_id, output_index) = {
                let state = self
                    .current_function_call_items
                    .get_mut(&index)
                    .expect("tool call state inserted above");
                state.arguments.push_str(fragment);
                (state.item_id.clone(), state.output_index)
            };
            events.push(ResponsesSseEvent::FunctionCallArgumentsDelta {
                sequence_number: self.next_seq(),
                item_id,
                output_index,
                delta: fragment.to_string(),
            });
        }
    }

    // -- Helpers -------------------------------------------------------------

    fn ensure_message_item(&mut self, events: &mut Vec<ResponsesSseEvent>) {
        if self.current_message_item_id.is_none() {
            let item_id = format!("msg_{}", Uuid::new_v4().simple());
            let output_index = self.ordered_items.len() as u64;
            self.ordered_items.push(ItemSlot::Message);
            self.current_message_item_id = Some(item_id.clone());
            self.message_output_index = Some(output_index);
            events.push(ResponsesSseEvent::OutputItemAdded {
                sequence_number: self.next_seq(),
                output_index,
                item: OutputItem::Message(OutputMessage {
                    id: item_id,
                    role: "assistant".to_string(),
                    status: "in_progress".to_string(),
                    content: Vec::new(),
                    extra: Default::default(),
                }),
            });
        }
    }

    fn next_seq(&mut self) -> u64 {
        let sequence_number = self.sequence_number;
        self.sequence_number += 1;
        sequence_number
    }

    fn base_response(&self, status: &str) -> ResponseObject {
        ResponseObject {
            id: self.response_id.clone(),
            object: "response".to_string(),
            created_at: self.created_ts,
            status: status.to_string(),
            error: None,
            incomplete_details: None,
            instructions: self.instructions.clone(),
            metadata: None,
            model: self.model.clone(),
            output: Vec::new(),
            parallel_tool_calls: None,
            previous_response_id: None,
            reasoning: None,
            store: false,
            temperature: None,
            text: None,
            tool_choice: None,
            tools: Vec::new(),
            top_p: None,
            truncation: None,
            usage: None,
            extra: Default::default(),
        }
    }

    fn build_reasoning_item(&self, item_id: &str) -> OutputReasoning {
        OutputReasoning {
            id: item_id.to_string(),
            summary: vec![ReasoningSummaryPart {
                r#type: Some("summary_text".to_string()),
                text: Some(self.reasoning_buffer.clone()),
                extra: Default::default(),
            }],
            extra: Default::default(),
        }
    }

    fn build_message_item(&self, item_id: &str) -> OutputMessage {
        let mut content = Vec::new();
        if let Some(content_index) = self.text_part_index {
            if !self.text_buffer.is_empty() || self.refusal_part_index.is_none() {
                content.push((
                    content_index,
                    OutputContentPart::OutputText {
                        text: self.text_buffer.clone(),
                        annotations: None,
                    },
                ));
            }
        }
        if let Some(content_index) = self.refusal_part_index {
            if !self.refusal_buffer.is_empty() {
                content.push((
                    content_index,
                    OutputContentPart::Refusal {
                        refusal: self.refusal_buffer.clone(),
                    },
                ));
            }
        }
        content.sort_by_key(|(content_index, _)| *content_index);
        OutputMessage {
            id: item_id.to_string(),
            role: "assistant".to_string(),
            status: "completed".to_string(),
            content: content.into_iter().map(|(_, part)| part).collect(),
            extra: Default::default(),
        }
    }

    fn terminal_status(&self) -> (&'static str, Option<IncompleteDetails>) {
        match self.finish_reason.as_deref() {
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
            _ => ("completed", None),
        }
    }

    fn final_response(
        &self,
        status: &str,
        incomplete_details: Option<IncompleteDetails>,
        usage: Option<ResponsesUsage>,
    ) -> ResponseObject {
        let mut response = self.base_response(status);
        response.incomplete_details = incomplete_details;
        response.usage = usage;

        for slot in &self.ordered_items {
            match slot {
                ItemSlot::Reasoning => {
                    if let Some(item_id) = &self.reasoning_item_id {
                        response
                            .output
                            .push(OutputItem::Reasoning(self.build_reasoning_item(item_id)));
                    }
                }
                ItemSlot::Message => {
                    if let Some(item_id) = &self.current_message_item_id {
                        response
                            .output
                            .push(OutputItem::Message(self.build_message_item(item_id)));
                    }
                }
                ItemSlot::FunctionCall { tool_index } => {
                    if let Some(state) = self.current_function_call_items.get(tool_index) {
                        response
                            .output
                            .push(OutputItem::FunctionCall(OutputFunctionCall {
                                id: state.item_id.clone(),
                                call_id: state.call_id.clone(),
                                name: state.name.clone(),
                                arguments: state.arguments.clone(),
                                status: "completed".to_string(),
                                extra: Default::default(),
                            }));
                    }
                }
            }
        }

        response
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

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

    fn event_types(events: &[ResponsesSseEvent]) -> Vec<&'static str> {
        events.iter().map(event_type).collect()
    }

    fn sequence_numbers(events: &[ResponsesSseEvent]) -> Vec<u64> {
        events
            .iter()
            .map(|event| match event {
                ResponsesSseEvent::Created {
                    sequence_number, ..
                }
                | ResponsesSseEvent::InProgress {
                    sequence_number, ..
                }
                | ResponsesSseEvent::Queued {
                    sequence_number, ..
                }
                | ResponsesSseEvent::Completed {
                    sequence_number, ..
                }
                | ResponsesSseEvent::Failed {
                    sequence_number, ..
                }
                | ResponsesSseEvent::Incomplete {
                    sequence_number, ..
                }
                | ResponsesSseEvent::OutputItemAdded {
                    sequence_number, ..
                }
                | ResponsesSseEvent::OutputItemDone {
                    sequence_number, ..
                }
                | ResponsesSseEvent::ContentPartAdded {
                    sequence_number, ..
                }
                | ResponsesSseEvent::ContentPartDone {
                    sequence_number, ..
                }
                | ResponsesSseEvent::OutputTextDelta {
                    sequence_number, ..
                }
                | ResponsesSseEvent::OutputTextDone {
                    sequence_number, ..
                }
                | ResponsesSseEvent::RefusalDelta {
                    sequence_number, ..
                }
                | ResponsesSseEvent::RefusalDone {
                    sequence_number, ..
                }
                | ResponsesSseEvent::FunctionCallArgumentsDelta {
                    sequence_number, ..
                }
                | ResponsesSseEvent::FunctionCallArgumentsDone {
                    sequence_number, ..
                }
                | ResponsesSseEvent::ReasoningSummaryPartAdded {
                    sequence_number, ..
                }
                | ResponsesSseEvent::ReasoningSummaryTextDelta {
                    sequence_number, ..
                }
                | ResponsesSseEvent::ReasoningTextDelta {
                    sequence_number, ..
                } => *sequence_number,
            })
            .collect()
    }

    fn assert_sequence_discipline(events: &[ResponsesSseEvent]) {
        let numbers = sequence_numbers(events);
        assert_eq!(numbers.first(), Some(&0), "sequence numbers start at 0");
        for pair in numbers.windows(2) {
            assert_eq!(pair[1], pair[0] + 1, "sequence numbers increase by 1");
        }
    }

    #[test]
    fn text_only_stream_full_sequence() {
        let mut translator = ResponsesStreamTranslator::new("gpt-4o", Some("Be brief."));

        let mut events = Vec::new();
        events.extend(
            translator.feed_chunk(&json!({
                "id": "chatcmpl-1",
                "choices": [{"index": 0, "delta": {"role": "assistant"}}]
            })),
        );
        events.extend(
            translator.feed_chunk(&json!({
                "choices": [{"index": 0, "delta": {"content": "Hello"}}]
            })),
        );
        events.extend(
            translator.feed_chunk(&json!({
                "choices": [{"index": 0, "delta": {"content": " world"}, "finish_reason": null}]
            })),
        );
        events.extend(
            translator.feed_chunk(&json!({
                "choices": [{"index": 0, "delta": {}, "finish_reason": "stop"}]
            })),
        );
        events.extend(translator.finish(Some(ResponsesUsage {
            input_tokens: 5,
            output_tokens: 2,
            total_tokens: 7,
            ..Default::default()
        })));

        assert_eq!(
            event_types(&events),
            vec![
                "response.created",
                "response.in_progress",
                "response.output_item.added",
                "response.content_part.added",
                "response.output_text.delta",
                "response.output_text.delta",
                "response.output_text.done",
                "response.content_part.done",
                "response.output_item.done",
                "response.completed",
            ]
        );
        assert_sequence_discipline(&events);

        let ResponsesSseEvent::OutputTextDone { text, .. } = &events[6] else {
            panic!("expected output_text.done");
        };
        assert_eq!(text, "Hello world");

        let ResponsesSseEvent::Completed { response, .. } = events.last().unwrap() else {
            panic!("expected completed");
        };
        assert_eq!(response.status, "completed");
        assert_eq!(response.instructions.as_deref(), Some("Be brief."));
        assert_eq!(response.model, "gpt-4o");
        assert_eq!(response.output.len(), 1);
        let OutputItem::Message(message) = &response.output[0] else {
            panic!("expected message item");
        };
        assert_eq!(message.role, "assistant");
        assert_eq!(message.content.len(), 1);
        let OutputContentPart::OutputText { text, .. } = &message.content[0] else {
            panic!("expected output_text part");
        };
        assert_eq!(text, "Hello world");
        let usage = response.usage.as_ref().unwrap();
        assert_eq!(usage.total_tokens, 7);
    }

    #[test]
    fn parallel_tool_calls_interleaved() {
        let mut translator = ResponsesStreamTranslator::new("gpt-4o", None);

        let mut events = Vec::new();
        events.extend(translator.feed_chunk(&json!({
            "choices": [{"index": 0, "delta": {"role": "assistant"}}]
        })));
        events.extend(translator.feed_chunk(&json!({
            "choices": [{"index": 0, "delta": {"tool_calls": [
                {"index": 0, "id": "call_a", "type": "function",
                 "function": {"name": "get_weather", "arguments": "{\"ci"}}
            ]}}]
        })));
        events.extend(translator.feed_chunk(&json!({
            "choices": [{"index": 0, "delta": {"tool_calls": [
                {"index": 1, "id": "call_b", "type": "function",
                 "function": {"name": "get_time", "arguments": "{\"tz\""}}
            ]}}]
        })));
        events.extend(translator.feed_chunk(&json!({
            "choices": [{"index": 0, "delta": {"tool_calls": [
                {"index": 0, "function": {"arguments": "ty\":\"Paris\"}"}}
            ]}}]
        })));
        events.extend(translator.feed_chunk(&json!({
            "choices": [{"index": 0, "delta": {"tool_calls": [
                {"index": 1, "function": {"arguments": ":\"UTC\"}"}}
            ]}}]
        })));
        events.extend(translator.finish(None));

        let types = event_types(&events);
        assert_eq!(
            types
                .iter()
                .filter(|t| **t == "response.output_item.added")
                .count(),
            2,
            "one output_item.added per tool call"
        );

        let added_items: Vec<_> = events
            .iter()
            .filter_map(|event| match event {
                ResponsesSseEvent::OutputItemAdded { item, .. } => Some(item),
                _ => None,
            })
            .collect();
        let OutputItem::FunctionCall(first) = &added_items[0] else {
            panic!("expected function_call item");
        };
        assert_eq!(first.call_id, "call_a");
        assert_eq!(first.name, "get_weather");
        let OutputItem::FunctionCall(second) = &added_items[1] else {
            panic!("expected function_call item");
        };
        assert_eq!(second.call_id, "call_b");
        assert_eq!(second.name, "get_time");

        // Argument deltas interleave across the two tool calls.
        let deltas: Vec<_> = events
            .iter()
            .filter_map(|event| match event {
                ResponsesSseEvent::FunctionCallArgumentsDelta { item_id, delta, .. } => {
                    Some((item_id.clone(), delta.clone()))
                }
                _ => None,
            })
            .collect();
        assert_eq!(deltas.len(), 4);
        assert_eq!(
            deltas.iter().map(|(item_id, _)| item_id.as_str()).collect::<Vec<_>>(),
            vec![
                first.id.as_str(),
                second.id.as_str(),
                first.id.as_str(),
                second.id.as_str(),
            ],
            "argument deltas interleave per tool index"
        );

        let done_arguments: Vec<_> = events
            .iter()
            .filter_map(|event| match event {
                ResponsesSseEvent::FunctionCallArgumentsDone { item_id, arguments, .. } => {
                    Some((item_id.clone(), arguments.clone()))
                }
                _ => None,
            })
            .collect();
        assert_eq!(done_arguments.len(), 2);
        assert_eq!(done_arguments[0].1, "{\"city\":\"Paris\"}");
        assert_eq!(done_arguments[1].1, "{\"tz\":\"UTC\"}");

        let ResponsesSseEvent::Completed { response, .. } = events.last().unwrap() else {
            panic!("expected completed");
        };
        assert_eq!(response.output.len(), 2);
        assert_sequence_discipline(&events);
    }

    #[test]
    fn reasoning_precedes_text_events() {
        let mut translator = ResponsesStreamTranslator::new("deepseek-r1", None);

        let mut events = Vec::new();
        events.extend(translator.feed_chunk(&json!({
            "choices": [{"index": 0, "delta": {"reasoning_content": "step 1"}}]
        })));
        events.extend(translator.feed_chunk(&json!({
            "choices": [{"index": 0, "delta": {"reasoning_content": " step 2"}}]
        })));
        events.extend(translator.feed_chunk(&json!({
            "choices": [{"index": 0, "delta": {"content": "final answer"}}]
        })));
        events.extend(translator.finish(None));

        let types = event_types(&events);
        let reasoning_position = types
            .iter()
            .position(|t| *t == "response.reasoning_text.delta")
            .expect("reasoning delta emitted");
        let text_position = types
            .iter()
            .position(|t| *t == "response.output_text.delta")
            .expect("text delta emitted");
        assert!(reasoning_position < text_position);

        // Reasoning summary events accompany the reasoning text.
        assert!(types.contains(&"response.reasoning_summary_part.added"));
        assert!(types.contains(&"response.reasoning_summary_text.delta"));

        // Reasoning occupies output_index 0, message output_index 1.
        let reasoning_delta = events.iter().find_map(|event| match event {
            ResponsesSseEvent::ReasoningTextDelta { output_index, .. } => Some(*output_index),
            _ => None,
        });
        assert_eq!(reasoning_delta, Some(0));
        let text_delta = events.iter().find_map(|event| match event {
            ResponsesSseEvent::OutputTextDelta { output_index, .. } => Some(*output_index),
            _ => None,
        });
        assert_eq!(text_delta, Some(1));

        let ResponsesSseEvent::Completed { response, .. } = events.last().unwrap() else {
            panic!("expected completed");
        };
        assert_eq!(response.output.len(), 2);
        let OutputItem::Reasoning(reasoning) = &response.output[0] else {
            panic!("expected reasoning item first");
        };
        assert_eq!(
            reasoning.summary[0].text.as_deref(),
            Some("step 1 step 2"),
            "reasoning buffer assembled in order"
        );
        assert_sequence_discipline(&events);
    }

    #[test]
    fn refusal_delta_synthesized() {
        let mut translator = ResponsesStreamTranslator::new("gpt-4o", None);

        let mut events = Vec::new();
        events.extend(translator.feed_chunk(&json!({
            "choices": [{"index": 0, "delta": {"role": "assistant"}}]
        })));
        events.extend(translator.feed_chunk(&json!({
            "choices": [{"index": 0, "delta": {"refusal": "I cannot help"}}]
        })));
        events.extend(translator.finish(None));

        let types = event_types(&events);
        assert!(types.contains(&"response.refusal.delta"));
        assert!(types.contains(&"response.refusal.done"));

        let refusal_delta = events.iter().find_map(|event| match event {
            ResponsesSseEvent::RefusalDelta { delta, .. } => Some(delta.clone()),
            _ => None,
        });
        assert_eq!(refusal_delta.as_deref(), Some("I cannot help"));

        let ResponsesSseEvent::Completed { response, .. } = events.last().unwrap() else {
            panic!("expected completed");
        };
        let OutputItem::Message(message) = &response.output[0] else {
            panic!("expected message item");
        };
        let OutputContentPart::Refusal { refusal } = &message.content[0] else {
            panic!("expected refusal part");
        };
        assert_eq!(refusal, "I cannot help");
        assert_sequence_discipline(&events);
    }

    #[test]
    fn sequence_numbers_strictly_monotonic_across_mixed_stream() {
        let mut translator = ResponsesStreamTranslator::new("gpt-4o", None);
        let mut events = Vec::new();
        events.extend(
            translator.feed_chunk(&json!({
                "choices": [{"index": 0, "delta": {"reasoning": "think"}}]
            })),
        );
        events.extend(
            translator.feed_chunk(&json!({
                "choices": [{"index": 0, "delta": {"content": "hi"}}]
            })),
        );
        events.extend(
            translator.feed_chunk(&json!({
                "choices": [{"index": 0, "delta": {"tool_calls": [
                    {"index": 0, "id": "call_z", "function": {"name": "f", "arguments": "{}"}}
                ]}}]
            })),
        );
        events.extend(translator.finish(None));

        assert_sequence_discipline(&events);

        // response.completed carries the highest sequence number.
        let numbers = sequence_numbers(&events);
        let completed_position = event_types(&events)
            .iter()
            .rposition(|t| *t == "response.completed")
            .unwrap();
        assert_eq!(completed_position, numbers.len() - 1);
        assert_eq!(numbers[completed_position], *numbers.iter().max().unwrap());
    }

    #[test]
    fn tool_call_arguments_split_across_chunks() {
        let mut translator = ResponsesStreamTranslator::new("gpt-4o", None);

        // First fragment carries id + name and a partial argument string.
        let mut events = Vec::new();
        events.extend(translator.feed_chunk(&json!({
            "choices": [{"index": 0, "delta": {"tool_calls": [
                {"index": 0, "id": "call_split", "type": "function",
                 "function": {"name": "search", "arguments": "{\"q\":"}}
            ]}}]
        })));
        // Later fragments carry only argument continuations.
        events.extend(translator.feed_chunk(&json!({
            "choices": [{"index": 0, "delta": {"tool_calls": [
                {"index": 0, "function": {"arguments": "\"obey"}}
            ]}}]
        })));
        events.extend(translator.feed_chunk(&json!({
            "choices": [{"index": 0, "delta": {"tool_calls": [
                {"index": 0, "function": {"arguments": " router\"}"}}
            ]}}]
        })));
        events.extend(translator.finish(None));

        let types = event_types(&events);
        assert_eq!(
            types.iter().filter(|t| **t == "response.output_item.added").count(),
            1,
            "output_item.added emitted exactly once"
        );
        assert_eq!(
            types
                .iter()
                .filter(|t| **t == "response.function_call_arguments.delta")
                .count(),
            3,
            "one argument delta per fragment"
        );

        let done = events.iter().find_map(|event| match event {
            ResponsesSseEvent::FunctionCallArgumentsDone { arguments, .. } => {
                Some(arguments.clone())
            }
            _ => None,
        });
        assert_eq!(done.as_deref(), Some("{\"q\":\"obey router\"}"));

        let ResponsesSseEvent::Completed { response, .. } = events.last().unwrap() else {
            panic!("expected completed");
        };
        let OutputItem::FunctionCall(call) = &response.output[0] else {
            panic!("expected function_call item");
        };
        assert_eq!(call.call_id, "call_split");
        assert_eq!(call.name, "search");
        assert_eq!(call.arguments, "{\"q\":\"obey router\"}");
        assert_sequence_discipline(&events);
    }

    #[test]
    fn finish_reason_length_maps_to_incomplete() {
        let mut translator = ResponsesStreamTranslator::new("gpt-4o", None);
        let mut events = Vec::new();
        events.extend(translator.feed_chunk(&json!({
            "choices": [{"index": 0, "delta": {"content": "truncat"}}]
        })));
        events.extend(translator.feed_chunk(&json!({
            "choices": [{"index": 0, "delta": {}, "finish_reason": "length"}]
        })));
        events.extend(translator.finish(None));

        let terminal = events.last().unwrap();
        let ResponsesSseEvent::Incomplete { response, .. } = terminal else {
            panic!("expected response.incomplete for finish_reason=length");
        };
        assert_eq!(response.status, "incomplete");
        assert_eq!(
            response.incomplete_details.as_ref().and_then(|d| d.reason.as_deref()),
            Some("max_output_tokens")
        );
    }

    #[test]
    fn lazy_initialization_on_first_content_chunk() {
        // Empty-choices (usage-only) chunks never trigger initialization.
        let mut translator = ResponsesStreamTranslator::new("gpt-4o", None);
        assert!(translator
            .feed_chunk(&json!({"choices": [], "usage": {"total_tokens": 1}}))
            .is_empty());
        assert!(translator
            .feed_chunk(&json!({"choices": [], "usage": {"total_tokens": 2}}))
            .is_empty());

        let events = translator.finish(None);
        let types = event_types(&events);
        assert_eq!(
            types.first(),
            Some(&"response.created"),
            "finish on a never-started stream still emits lifecycle events"
        );
        assert_eq!(types.last(), Some(&"response.completed"));
        assert_sequence_discipline(&events);
    }
}
