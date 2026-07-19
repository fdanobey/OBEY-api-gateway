//! Responses API → Chat Completions response translator (streaming and non-streaming).
//!
//! Populated by tasks 13.1–13.2 of the `codex-backend-translation` spec.

use std::collections::HashMap;
use std::time::{SystemTime, UNIX_EPOCH};

use futures::StreamExt;
use serde_json::json;

use crate::codex::errors::CodexError;
use crate::codex::sse::{CompletedPayload, ResponsesEvent};

// ─── Helper structs ──────────────────────────────────────────────────────────

/// Accumulates tool-call fragments across streaming deltas.
#[derive(Debug, Clone)]
struct ToolCallBuilder {
    id: String,
    name: String,
    arguments: String,
}

/// Final usage counters extracted from the `CompletedPayload`.
#[derive(Debug, Clone, Copy)]
struct UsageOutput {
    prompt_tokens: u64,
    completion_tokens: u64,
    total_tokens: u64,
}

// ─── ResponseAccumulator ─────────────────────────────────────────────────────

/// Non-streaming accumulator that drains a `ResponsesEvent` stream and
/// assembles a Chat Completions JSON response body.
#[derive(Debug)]
pub struct ResponseAccumulator {
    response_id: Option<String>,
    model: Option<String>,
    text: String,
    tool_calls: Vec<ToolCallBuilder>,
    usage: Option<UsageOutput>,
    finish: Option<&'static str>,
    errored: bool,
}

impl ResponseAccumulator {
    fn new() -> Self {
        Self {
            response_id: None,
            model: None,
            text: String::new(),
            tool_calls: Vec::new(),
            usage: None,
            finish: None,
            errored: false,
        }
    }
}

// ─── finish_reason ───────────────────────────────────────────────────────────

/// Derives the Chat Completions `finish_reason` from a terminal
/// `CompletedPayload` (Req 4.12).
///
/// Priority:
/// 1. `"length"` if `incomplete_details.reason == "max_output_tokens"`
/// 2. `"tool_calls"` if any output item has `kind == "function_call"`
/// 3. `"stop"` otherwise
pub fn finish_reason(resp: &CompletedPayload) -> &'static str {
    if resp
        .incomplete_details
        .as_ref()
        .and_then(|d| d.reason.as_deref())
        == Some("max_output_tokens")
    {
        return "length";
    }

    if resp.output.iter().any(|item| item.kind == "function_call") {
        return "tool_calls";
    }

    "stop"
}

// ─── accumulate ──────────────────────────────────────────────────────────────

/// Drains the entire upstream `ResponsesEvent` stream and assembles a
/// Chat Completions JSON response body (non-streaming path).
///
/// Returns `Err(CodexError::UpstreamTerminatedEarly)` if the stream ends
/// without a terminal `Completed` or `Failed`/`Error` event (Req 5.7).
#[tracing::instrument(skip_all)]
pub async fn accumulate<S>(mut upstream: S) -> Result<serde_json::Value, CodexError>
where
    S: futures::Stream<Item = ResponsesEvent> + Unpin,
{
    let mut acc = ResponseAccumulator::new();
    let mut terminated = false;

    while let Some(event) = upstream.next().await {
        match event {
            ResponsesEvent::Created { id, model, .. } => {
                acc.response_id = Some(id);
                acc.model = Some(model);
            }

            ResponsesEvent::OutputTextDelta { delta, .. } => {
                acc.text.push_str(&delta);
            }

            ResponsesEvent::OutputItemAdded {
                item_id,
                kind,
                call_id,
                name,
                ..
            } if kind == "function_call" => {
                acc.tool_calls.push(ToolCallBuilder {
                    id: call_id.unwrap_or(item_id),
                    name: name.unwrap_or_default(),
                    arguments: String::new(),
                });
            }

            ResponsesEvent::FunctionCallArgumentsDelta { item_id, delta, .. } => {
                // Match by item_id — the tool call whose id matches,
                // or fall back to the last one pushed (streaming order).
                let idx = acc
                    .tool_calls
                    .iter()
                    .rposition(|tc| tc.id == item_id)
                    .or_else(|| {
                        if acc.tool_calls.is_empty() {
                            None
                        } else {
                            Some(acc.tool_calls.len() - 1)
                        }
                    });
                if let Some(i) = idx {
                    acc.tool_calls[i].arguments.push_str(&delta);
                }
            }

            ResponsesEvent::Completed { response, .. } => {
                // Authoritative finish reason from the payload.
                acc.finish = Some(finish_reason(&response));

                // Extract usage.
                if let Some(u) = &response.usage {
                    acc.usage = Some(UsageOutput {
                        prompt_tokens: u.input_tokens.unwrap_or(0),
                        completion_tokens: u.output_tokens.unwrap_or(0),
                        total_tokens: u.total_tokens.unwrap_or(0),
                    });
                }

                // For the non-streaming path, the CompletedPayload's output
                // items are authoritative for text content and tool calls.
                // Concatenate output_text content from message-type items.
                for item in &response.output {
                    if item.kind == "message" {
                        if let Some(contents) = &item.content {
                            for c in contents {
                                if c.kind == "output_text" {
                                    if let Some(t) = &c.text {
                                        // Only overwrite if we haven't
                                        // accumulated streaming deltas yet,
                                        // or append if text is already present.
                                        if acc.text.is_empty() {
                                            acc.text.push_str(t);
                                        }
                                    }
                                }
                            }
                        }
                    }
                }

                // Backfill tool_calls from CompletedPayload if streaming
                // didn't capture them (non-streaming path).
                if acc.tool_calls.is_empty() {
                    for item in &response.output {
                        if item.kind == "function_call" {
                            acc.tool_calls.push(ToolCallBuilder {
                                id: item
                                    .call_id
                                    .clone()
                                    .or_else(|| item.id.clone())
                                    .unwrap_or_default(),
                                name: item.name.clone().unwrap_or_default(),
                                arguments: item.arguments.clone().unwrap_or_default(),
                            });
                        }
                    }
                }

                // Backfill response_id / model if Created was missed.
                if acc.response_id.is_none() {
                    acc.response_id = Some(response.id);
                }
                if acc.model.is_none() {
                    acc.model = Some(response.model);
                }

                terminated = true;
            }

            ResponsesEvent::Failed { .. } | ResponsesEvent::Error { .. } => {
                acc.errored = true;
                acc.finish = Some("error");
                terminated = true;
            }

            // OutputTextDone, Other (reasoning items, native tool calls) → skip (Req 4.8, 4.13)
            ResponsesEvent::OutputTextDone { .. } | ResponsesEvent::Other { .. } => {}

            // OutputItemAdded for non-function_call kinds → skip
            ResponsesEvent::OutputItemAdded { .. } => {}
        }
    }

    if !terminated {
        return Err(CodexError::UpstreamTerminatedEarly);
    }

    // Override finish_reason when errored.
    let finish_reason_str = if acc.errored {
        "error"
    } else {
        acc.finish.unwrap_or("stop")
    };

    // Assemble the Chat Completions response JSON.
    let created_ts = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map(|d| d.as_secs())
        .unwrap_or(0);

    let response_id = acc.response_id.unwrap_or_default();
    let model = acc.model.unwrap_or_default();

    // Build message object.
    let has_tool_calls = !acc.tool_calls.is_empty();
    let has_text = !acc.text.is_empty();

    let content = if has_text {
        serde_json::Value::String(acc.text)
    } else if has_tool_calls {
        serde_json::Value::Null
    } else {
        // Empty response with no tool calls — content is empty string.
        serde_json::Value::String(String::new())
    };

    let mut message = serde_json::Map::new();
    message.insert("role".into(), json!("assistant"));
    message.insert("content".into(), content);

    if has_tool_calls {
        let tool_calls_json: Vec<serde_json::Value> = acc
            .tool_calls
            .iter()
            .map(|tc| {
                json!({
                    "id": tc.id,
                    "type": "function",
                    "function": {
                        "name": tc.name,
                        "arguments": tc.arguments,
                    }
                })
            })
            .collect();
        message.insert("tool_calls".into(), json!(tool_calls_json));
    }

    let choice = json!({
        "index": 0,
        "message": serde_json::Value::Object(message),
        "finish_reason": finish_reason_str,
    });

    let mut body = serde_json::Map::new();
    body.insert("id".into(), json!(format!("chatcmpl-{response_id}")));
    body.insert("object".into(), json!("chat.completion"));
    body.insert("created".into(), json!(created_ts));
    body.insert("model".into(), json!(model));
    body.insert("choices".into(), json!([choice]));

    if let Some(u) = acc.usage {
        body.insert(
            "usage".into(),
            json!({
                "prompt_tokens": u.prompt_tokens,
                "completion_tokens": u.completion_tokens,
                "total_tokens": u.total_tokens,
            }),
        );
    }

    Ok(serde_json::Value::Object(body))
}

// ─── stream_translate ────────────────────────────────────────────────────────

/// Internal state for the streaming translator state machine.
#[allow(dead_code)]
struct StreamTranslatorState {
    response_id: String,
    model: String,
    first_text_emitted: bool,
    tool_index_map: HashMap<String, u32>,
    next_tool_index: u32,
    finished: bool,
    created_ts: u64,
}

/// Translates a `ResponsesEvent` stream into a stream of SSE-formatted
/// Chat Completions chunk strings (`data: {...}\n\n`), suitable for piping
/// directly to an HTTP response body (streaming path).
///
/// Each yielded item is either a JSON chunk line or the terminal
/// `data: [DONE]\n\n` sentinel. The `resolved_model` is used as the
/// `model` field in every emitted chunk (Req 4.11).
#[tracing::instrument(skip_all)]
#[allow(dead_code)]
pub fn stream_translate<S>(
    upstream: S,
    resolved_model: String,
) -> impl futures::Stream<Item = Result<String, CodexError>>
where
    S: futures::Stream<Item = ResponsesEvent> + Unpin + Send + 'static,
{
    let created_ts = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map(|d| d.as_secs())
        .unwrap_or(0);

    let state = StreamTranslatorState {
        response_id: String::new(),
        model: resolved_model,
        first_text_emitted: false,
        tool_index_map: HashMap::new(),
        next_tool_index: 0,
        finished: false,
        created_ts,
    };

    futures::stream::unfold((upstream, state), |(mut upstream, mut state)| async move {
        if state.finished {
            return None;
        }

        loop {
            let event = match upstream.next().await {
                Some(ev) => ev,
                None => {
                    // Stream ended without terminal event.
                    if !state.finished {
                        state.finished = true;
                        return Some((
                            vec![Err(CodexError::UpstreamTerminatedEarly)],
                            (upstream, state),
                        ));
                    }
                    return None;
                }
            };

            let chunks = translate_event(&mut state, event);
            if !chunks.is_empty() {
                return Some((chunks, (upstream, state)));
            }
            // Event produced no output — loop to next event.
        }
    })
    .flat_map(futures::stream::iter)
}

/// Translates a single `ResponsesEvent` into zero or more SSE chunk strings.
#[allow(dead_code)]
fn translate_event(
    state: &mut StreamTranslatorState,
    event: ResponsesEvent,
) -> Vec<Result<String, CodexError>> {
    match event {
        ResponsesEvent::Created { id, .. } => {
            state.response_id = id;
            vec![]
        }

        ResponsesEvent::OutputTextDelta { delta, .. } => {
            let chunk = if !state.first_text_emitted {
                state.first_text_emitted = true;
                format_chunk(
                    state,
                    json!({"role": "assistant", "content": delta}),
                    serde_json::Value::Null,
                    None,
                )
            } else {
                format_chunk(
                    state,
                    json!({"content": delta}),
                    serde_json::Value::Null,
                    None,
                )
            };
            vec![Ok(chunk)]
        }

        ResponsesEvent::OutputItemAdded {
            item_id,
            kind,
            call_id,
            name,
            ..
        } if kind == "function_call" => {
            let idx = state.next_tool_index;
            state.next_tool_index += 1;
            state.tool_index_map.insert(item_id, idx);

            let delta = json!({
                "tool_calls": [{
                    "index": idx,
                    "id": call_id.unwrap_or_default(),
                    "type": "function",
                    "function": {
                        "name": name.unwrap_or_default(),
                        "arguments": ""
                    }
                }]
            });
            let chunk = format_chunk(state, delta, serde_json::Value::Null, None);
            vec![Ok(chunk)]
        }

        ResponsesEvent::FunctionCallArgumentsDelta { item_id, delta, .. } => {
            let idx = state
                .tool_index_map
                .get(&item_id)
                .copied()
                .unwrap_or(state.next_tool_index.saturating_sub(1));

            let delta_obj = json!({
                "tool_calls": [{
                    "index": idx,
                    "function": {
                        "arguments": delta
                    }
                }]
            });
            let chunk = format_chunk(state, delta_obj, serde_json::Value::Null, None);
            vec![Ok(chunk)]
        }

        ResponsesEvent::Completed { response, .. } => {
            state.finished = true;
            let reason = finish_reason(&response);

            let usage_val = response.usage.as_ref().map(|u| {
                json!({
                    "prompt_tokens": u.input_tokens.unwrap_or(0),
                    "completion_tokens": u.output_tokens.unwrap_or(0),
                    "total_tokens": u.total_tokens.unwrap_or(0)
                })
            });

            let chunk = format_chunk(
                state,
                json!({}),
                serde_json::Value::String(reason.to_owned()),
                usage_val,
            );
            vec![Ok(chunk), Ok("data: [DONE]\n\n".to_owned())]
        }

        ResponsesEvent::Failed { .. } | ResponsesEvent::Error { .. } => {
            state.finished = true;
            let chunk = format_chunk(
                state,
                json!({}),
                serde_json::Value::String("error".to_owned()),
                None,
            );
            vec![Ok(chunk), Ok("data: [DONE]\n\n".to_owned())]
        }

        // OutputTextDone, Other (reasoning, native tool calls), non-function_call OutputItemAdded → skip
        ResponsesEvent::OutputTextDone { .. }
        | ResponsesEvent::Other { .. }
        | ResponsesEvent::OutputItemAdded { .. } => vec![],
    }
}

/// Formats a single SSE chunk line: `data: <json>\n\n`.
#[allow(dead_code)]
fn format_chunk(
    state: &StreamTranslatorState,
    delta: serde_json::Value,
    finish_reason_val: serde_json::Value,
    usage: Option<serde_json::Value>,
) -> String {
    let mut choice = serde_json::Map::new();
    choice.insert("index".into(), json!(0));
    choice.insert("delta".into(), delta);
    choice.insert("finish_reason".into(), finish_reason_val);

    let mut body = serde_json::Map::new();
    body.insert(
        "id".into(),
        json!(format!("chatcmpl-{}", state.response_id)),
    );
    body.insert("object".into(), json!("chat.completion.chunk"));
    body.insert("created".into(), json!(state.created_ts));
    body.insert("model".into(), json!(&state.model));
    body.insert("choices".into(), json!([serde_json::Value::Object(choice)]));

    if let Some(u) = usage {
        body.insert("usage".into(), u);
    }

    let json_str =
        serde_json::to_string(&serde_json::Value::Object(body)).unwrap_or_else(|_| "{}".to_owned());
    format!("data: {json_str}\n\n")
}

#[cfg(test)]
mod property_tests {
    use super::*;
    use crate::codex::sse::{
        CompletedPayload, OutputItem, OutputItemContent, ResponsesEvent, Usage,
    };
    use futures::stream;
    use futures::StreamExt;
    use proptest::prelude::*;

    // ─── Strategies ──────────────────────────────────────────────────────────

    /// ASCII printable string up to 32 chars for text deltas.
    fn ascii_delta() -> impl Strategy<Value = String> {
        proptest::string::string_regex("[a-zA-Z0-9 ]{1,32}").unwrap()
    }

    /// Item ID from a bounded pool of 1..=4.
    fn item_id_pool() -> impl Strategy<Value = String> {
        (1u8..=4).prop_map(|i| format!("item_{}", i))
    }

    /// Generates a sorted Vec<ResponsesEvent> with unique sequence_numbers in 0..1000.
    /// Contains: Created first, then a mix of OutputTextDelta and
    /// OutputItemAdded(function_call)/FunctionCallArgumentsDelta, then Completed.
    ///
    /// Returns (events, expected_text) where expected_text is the concatenation
    /// of all OutputTextDelta deltas.
    fn event_sequence_with_tool_calls() -> impl Strategy<Value = (Vec<ResponsesEvent>, String)> {
        // Generate between 1 and 10 text deltas and 0..3 tool call groups
        let text_deltas = proptest::collection::vec(ascii_delta(), 1..=10);
        let tool_call_count = 0u8..=3;
        let tool_arg_deltas =
            proptest::collection::vec(proptest::collection::vec(ascii_delta(), 1..=3), 0..=3);

        (text_deltas, tool_call_count, tool_arg_deltas).prop_flat_map(
            |(deltas, tc_count, arg_groups)| {
                let tc_count = tc_count as usize;
                let arg_groups: Vec<Vec<String>> = arg_groups.into_iter().take(tc_count).collect();
                let deltas_clone = deltas.clone();
                let arg_groups_clone = arg_groups.clone();

                // We need item_ids for tool calls
                let tool_ids =
                    proptest::collection::vec(item_id_pool(), tc_count..=tc_count.max(1));

                tool_ids.prop_map(move |tool_ids| {
                    let mut events: Vec<ResponsesEvent> = Vec::new();
                    let mut seq: u64 = 0;

                    // Created event
                    events.push(ResponsesEvent::Created {
                        id: "resp_test".to_string(),
                        model: "test-model".to_string(),
                        sequence_number: seq,
                    });
                    seq += 1;

                    let expected_text: String = deltas_clone.concat();

                    // Interleave text deltas and tool call events
                    let mut text_idx = 0;
                    let mut tool_idx = 0;

                    // Emit half the text deltas first
                    let half = deltas_clone.len() / 2;
                    for i in 0..half {
                        events.push(ResponsesEvent::OutputTextDelta {
                            item_id: "msg_item".to_string(),
                            delta: deltas_clone[i].clone(),
                            sequence_number: seq,
                        });
                        seq += 1;
                        text_idx = i + 1;
                    }

                    // Emit tool call events
                    for t in 0..tc_count.min(tool_ids.len()) {
                        let tid = &tool_ids[t];
                        events.push(ResponsesEvent::OutputItemAdded {
                            item_id: tid.clone(),
                            kind: "function_call".to_string(),
                            call_id: Some(format!("call_{}", t)),
                            name: Some(format!("func_{}", t)),
                            sequence_number: seq,
                        });
                        seq += 1;

                        // Emit argument deltas for this tool call
                        if t < arg_groups_clone.len() {
                            for arg_delta in &arg_groups_clone[t] {
                                events.push(ResponsesEvent::FunctionCallArgumentsDelta {
                                    item_id: tid.clone(),
                                    delta: arg_delta.clone(),
                                    sequence_number: seq,
                                });
                                seq += 1;
                            }
                        }
                        tool_idx = t + 1;
                    }
                    let _ = tool_idx;

                    // Emit remaining text deltas
                    for i in text_idx..deltas_clone.len() {
                        events.push(ResponsesEvent::OutputTextDelta {
                            item_id: "msg_item".to_string(),
                            delta: deltas_clone[i].clone(),
                            sequence_number: seq,
                        });
                        seq += 1;
                    }

                    // Build CompletedPayload with the concatenated text
                    let completed_payload = CompletedPayload {
                        id: "resp_test".to_string(),
                        model: "test-model".to_string(),
                        output: {
                            let mut items = vec![OutputItem {
                                kind: "message".to_string(),
                                id: Some("msg_item".to_string()),
                                call_id: None,
                                name: None,
                                arguments: None,
                                content: Some(vec![OutputItemContent {
                                    kind: "output_text".to_string(),
                                    text: Some(expected_text.clone()),
                                }]),
                            }];
                            for t in 0..tc_count.min(tool_ids.len()) {
                                let args: String = if t < arg_groups_clone.len() {
                                    arg_groups_clone[t].concat()
                                } else {
                                    String::new()
                                };
                                items.push(OutputItem {
                                    kind: "function_call".to_string(),
                                    id: Some(tool_ids[t].clone()),
                                    call_id: Some(format!("call_{}", t)),
                                    name: Some(format!("func_{}", t)),
                                    arguments: Some(args),
                                    content: None,
                                });
                            }
                            items
                        },
                        usage: Some(Usage {
                            input_tokens: Some(10),
                            output_tokens: Some(5),
                            total_tokens: Some(15),
                        }),
                        incomplete_details: None,
                    };

                    events.push(ResponsesEvent::Completed {
                        response: completed_payload,
                        sequence_number: seq,
                    });

                    (events, expected_text)
                })
            },
        )
    }

    /// Generates a text-only event sequence (no function calls) for Property 5.
    /// Returns (events, expected_text).
    fn text_only_event_sequence() -> impl Strategy<Value = (Vec<ResponsesEvent>, String)> {
        let text_deltas = proptest::collection::vec(ascii_delta(), 1..=10);

        text_deltas.prop_map(|deltas| {
            let mut events: Vec<ResponsesEvent> = Vec::new();
            let mut seq: u64 = 0;

            let expected_text: String = deltas.concat();

            // Created event
            events.push(ResponsesEvent::Created {
                id: "resp_test".to_string(),
                model: "test-model".to_string(),
                sequence_number: seq,
            });
            seq += 1;

            // Text deltas
            for d in &deltas {
                events.push(ResponsesEvent::OutputTextDelta {
                    item_id: "msg_item".to_string(),
                    delta: d.clone(),
                    sequence_number: seq,
                });
                seq += 1;
            }

            // Completed with message output containing the full text
            let completed_payload = CompletedPayload {
                id: "resp_test".to_string(),
                model: "test-model".to_string(),
                output: vec![OutputItem {
                    kind: "message".to_string(),
                    id: Some("msg_item".to_string()),
                    call_id: None,
                    name: None,
                    arguments: None,
                    content: Some(vec![OutputItemContent {
                        kind: "output_text".to_string(),
                        text: Some(expected_text.clone()),
                    }]),
                }],
                usage: Some(Usage {
                    input_tokens: Some(10),
                    output_tokens: Some(5),
                    total_tokens: Some(15),
                }),
                incomplete_details: None,
            };

            events.push(ResponsesEvent::Completed {
                response: completed_payload,
                sequence_number: seq,
            });

            (events, expected_text)
        })
    }

    // ─── Helper: extract text content from streaming chunks ──────────────────

    /// Parses SSE chunk lines from stream_translate output and extracts
    /// the concatenated `delta.content` text.
    fn extract_streaming_text(chunks: &[String]) -> String {
        let mut text = String::new();
        for chunk in chunks {
            // Each chunk is "data: {...}\n\n" or "data: [DONE]\n\n"
            let data = chunk.trim();
            let json_str = match data.strip_prefix("data: ") {
                Some(s) => s.trim(),
                None => continue,
            };
            if json_str == "[DONE]" {
                continue;
            }
            if let Ok(v) = serde_json::from_str::<serde_json::Value>(json_str) {
                if let Some(content) = v
                    .get("choices")
                    .and_then(|c| c.get(0))
                    .and_then(|c| c.get("delta"))
                    .and_then(|d| d.get("content"))
                    .and_then(|c| c.as_str())
                {
                    text.push_str(content);
                }
            }
        }
        text
    }

    // ─── Property 4: Streaming Order & Content Preservation ──────────────────

    proptest! {
        #![proptest_config(ProptestConfig::with_cases(50))]

        /// For all synthetic event sequences with strictly increasing sequence_number:
        /// - Emitted chunks preserve source order
        /// - concat(delta.content for each emitted chunk) == concat(delta for each source OutputTextDelta)
        #[test]
        fn prop_streaming_order_and_content_preservation(
            (events, expected_text) in event_sequence_with_tool_calls()
        ) {
            let rt = tokio::runtime::Runtime::new().unwrap();
            rt.block_on(async {
                let upstream = stream::iter(events.clone());
                let output_stream = stream_translate(upstream, "test-model".to_string());

                let chunks: Vec<String> = output_stream
                    .filter_map(|r| async { r.ok() })
                    .collect()
                    .await;

                // 1. Verify chunks are non-empty (at least the terminal chunk)
                assert!(!chunks.is_empty(), "stream must emit at least one chunk");

                // 2. Extract text content from streaming chunks
                let streaming_text = extract_streaming_text(&chunks);

                // 3. Assert content preservation: streaming text == expected text
                assert_eq!(
                    streaming_text, expected_text,
                    "streaming text content must equal concatenation of all OutputTextDelta deltas"
                );

                // 4. Verify ordering: text chunks appear in the same order as source deltas
                // Extract individual content pieces from chunks in order
                let mut ordered_pieces: Vec<String> = Vec::new();
                for chunk in &chunks {
                    let data = chunk.trim();
                    let json_str = match data.strip_prefix("data: ") {
                        Some(s) => s.trim(),
                        None => continue,
                    };
                    if json_str == "[DONE]" {
                        continue;
                    }
                    if let Ok(v) = serde_json::from_str::<serde_json::Value>(json_str) {
                        if let Some(content) = v
                            .get("choices")
                            .and_then(|c| c.get(0))
                            .and_then(|c| c.get("delta"))
                            .and_then(|d| d.get("content"))
                            .and_then(|c| c.as_str())
                        {
                            ordered_pieces.push(content.to_string());
                        }
                    }
                }

                // The ordered pieces concatenated must equal expected_text
                let reconstructed: String = ordered_pieces.concat();
                assert_eq!(
                    reconstructed, expected_text,
                    "ordered content pieces must reconstruct the expected text"
                );
            });
        }
    }

    // ─── Property 5: Streaming / Non-Streaming Equivalence ───────────────────

    proptest! {
        #![proptest_config(ProptestConfig::with_cases(50))]

        /// For all synthetic event sequences terminating in Completed with a single
        /// message-type output item (no function_call items):
        /// concat(streaming_chunks.delta.content) == non_streaming_response.choices[0].message.content
        #[test]
        fn prop_streaming_non_streaming_equivalence(
            (events, expected_text) in text_only_event_sequence()
        ) {
            let rt = tokio::runtime::Runtime::new().unwrap();
            rt.block_on(async {
                // --- Streaming path ---
                let upstream_streaming = stream::iter(events.clone());
                let output_stream = stream_translate(upstream_streaming, "test-model".to_string());

                let chunks: Vec<String> = output_stream
                    .filter_map(|r| async { r.ok() })
                    .collect()
                    .await;

                let streaming_text = extract_streaming_text(&chunks);

                // --- Non-streaming path ---
                let upstream_accumulate = stream::iter(events.clone());
                let response_json = accumulate(upstream_accumulate).await
                    .expect("accumulate must succeed for valid event sequence");

                let non_streaming_text = response_json
                    .get("choices")
                    .and_then(|c| c.get(0))
                    .and_then(|c| c.get("message"))
                    .and_then(|m| m.get("content"))
                    .and_then(|c| c.as_str())
                    .unwrap_or("")
                    .to_string();

                // Assert equivalence
                assert_eq!(
                    streaming_text, non_streaming_text,
                    "streaming content must equal non-streaming content"
                );

                // Both must equal the expected text from the generator
                assert_eq!(
                    streaming_text, expected_text,
                    "both paths must produce the expected concatenated text"
                );
            });
        }
    }
}
