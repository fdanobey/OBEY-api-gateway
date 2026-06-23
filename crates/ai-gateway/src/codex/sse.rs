//! Hand-rolled SSE line parser for the Codex Responses API event stream.
//!
//! Per design §10, this is intentionally a minimal hand-rolled framer + JSON
//! dispatcher (no new dep): records are split on `\n\n`, every `data:` line
//! within a record is concatenated, the resulting payload is decoded as JSON,
//! and the top-level `type` field drives dispatch into a typed
//! [`ResponsesEvent`].
//!
//! Security: error variants emitted from this module are sanitized — see the
//! invariant on [`crate::codex::errors::CodexError`].

use serde::Deserialize;

use crate::codex::errors::CodexError;

/// Typed Codex Responses API event.
///
/// Variants and field names mirror the Data Models section of `design.md`
/// verbatim (Req 4.1–4.7, 13.4). The `Other` variant captures every native
/// server-side tool-call event family (`response.web_search_call.*`,
/// `response.code_interpreter_call.*`, `response.mcp_call.*`,
/// `response.file_search_call.*`, `response.image_generation_call.*`,
/// `response.custom_tool_call_input.*`) so the response translator can
/// silently skip them while preserving stream ordering — see Req 4.13 / 14.5.
#[derive(Debug, Clone)]
#[allow(dead_code)]
pub enum ResponsesEvent {
    Created {
        id: String,
        model: String,
        sequence_number: u64,
    },
    OutputTextDelta {
        item_id: String,
        delta: String,
        sequence_number: u64,
    },
    OutputTextDone {
        item_id: String,
        text: String,
        sequence_number: u64,
    },
    OutputItemAdded {
        item_id: String,
        kind: String,
        call_id: Option<String>,
        name: Option<String>,
        sequence_number: u64,
    },
    FunctionCallArgumentsDelta {
        item_id: String,
        delta: String,
        sequence_number: u64,
    },
    Completed {
        response: CompletedPayload,
        sequence_number: u64,
    },
    Failed {
        error: ErrorPayload,
        sequence_number: u64,
    },
    Error {
        error: ErrorPayload,
        sequence_number: u64,
    },
    Other {
        kind: String,
        sequence_number: u64,
    },
}

/// Aggregated terminal payload carried by `response.completed`.
///
/// Captures enough of the Codex output to drive `finish_reason` (Req 4.12)
/// and tool-call assembly (Req 5.4) in the response translator.
#[derive(Debug, Clone, Deserialize)]
pub struct CompletedPayload {
    pub id: String,
    pub model: String,
    #[serde(default)]
    pub output: Vec<OutputItem>,
    #[serde(default)]
    pub usage: Option<Usage>,
    #[serde(default)]
    pub incomplete_details: Option<IncompleteDetails>,
}

/// One element of `response.output` inside a `CompletedPayload`.
///
/// `kind` carries the JSON `type` discriminator (e.g. `"message"`,
/// `"function_call"`, `"reasoning"`); `serde(rename = "type")` does the
/// keyword-dodging.
#[derive(Debug, Clone, Deserialize)]
pub struct OutputItem {
    #[serde(rename = "type")]
    pub kind: String,
    #[serde(default)]
    pub id: Option<String>,
    #[serde(default)]
    pub call_id: Option<String>,
    #[serde(default)]
    pub name: Option<String>,
    #[serde(default)]
    pub arguments: Option<String>,
    #[serde(default)]
    pub content: Option<Vec<OutputItemContent>>,
}

/// One element inside `OutputItem.content`.
#[derive(Debug, Clone, Deserialize)]
pub struct OutputItemContent {
    #[serde(rename = "type")]
    pub kind: String,
    #[serde(default)]
    pub text: Option<String>,
}

/// Token usage as reported by the Codex backend on `response.completed`.
#[derive(Debug, Clone, Deserialize)]
pub struct Usage {
    #[serde(default)]
    pub input_tokens: Option<u64>,
    #[serde(default)]
    pub output_tokens: Option<u64>,
    #[serde(default)]
    pub total_tokens: Option<u64>,
}

/// `incomplete_details` block on `response.completed`; presence of
/// `reason == "max_output_tokens"` drives the `length` finish reason.
#[derive(Debug, Clone, Deserialize)]
pub struct IncompleteDetails {
    #[serde(default)]
    pub reason: Option<String>,
}

/// Sanitized error envelope carried by `response.failed` / `error` events.
///
/// `kind` mirrors the JSON `type` field. All fields are optional because
/// upstream Codex shapes vary across event flavours and we never want to
/// fail-stop the stream on a missing diagnostic field.
#[derive(Debug, Clone, Deserialize)]
#[allow(dead_code)]
pub struct ErrorPayload {
    #[serde(default)]
    pub message: Option<String>,
    #[serde(default)]
    pub code: Option<String>,
    #[serde(default, rename = "type")]
    pub kind: Option<String>,
}

/// Streaming SSE framer for the Codex Responses event stream.
///
/// Hand-rolled to keep the dependency surface tight (design §Dependencies);
/// the framer itself is <80 LOC and the per-event JSON dispatcher is pure
/// data shuffling. State is a single rolling UTF-8 buffer; partial chunks
/// across `feed` calls are stitched together until a record boundary
/// (`\n\n`) is found.
#[derive(Debug, Default)]
pub struct SseLineParser {
    buf: String,
}

impl SseLineParser {
    pub fn new() -> Self {
        Self { buf: String::new() }
    }

    /// Feed raw bytes from the upstream `text/event-stream` body. Returns
    /// zero or more typed events (or sanitized parse errors) corresponding
    /// to records that completed within this chunk.
    ///
    /// Malformed JSON in one record is reported as `Err(CodexError::Sse)`
    /// and the parser continues with the next record — a single corrupt
    /// frame must not poison the stream (Req 4.x stability invariant).
    pub fn feed(&mut self, chunk: &[u8]) -> Vec<Result<ResponsesEvent, CodexError>> {
        // Lossy decode keeps the framer robust against any non-UTF-8 byte
        // sequence (Codex always emits valid UTF-8, but we never want to
        // panic on a bad upstream).
        self.buf.push_str(&String::from_utf8_lossy(chunk));

        let mut out = Vec::new();
        while let Some(boundary) = self.buf.find("\n\n") {
            // Drain "[..boundary]\n\n" out of self.buf, leaving the trailing
            // partial behind for the next feed call.
            let rest = self.buf.split_off(boundary + 2);
            let record = std::mem::replace(&mut self.buf, rest);
            let record_body = &record[..record.len() - 2];

            // Walk every line; concatenate `data:` payloads (multi-line `data:`
            // is folded with `\n` per the SSE spec). Other line kinds
            // (`event:`, `retry:`, comment lines starting with `:`) are
            // ignored — Codex emits `event:` but we dispatch off the JSON
            // `type` field instead.
            let mut payload = String::new();
            for line in record_body.split('\n') {
                let line = line.strip_suffix('\r').unwrap_or(line);
                if let Some(rest) = line.strip_prefix("data:") {
                    let rest = rest.strip_prefix(' ').unwrap_or(rest);
                    if !payload.is_empty() {
                        payload.push('\n');
                    }
                    payload.push_str(rest);
                }
            }

            if payload.is_empty() {
                continue;
            }
            // The gateway emits its own `[DONE]` terminator downstream;
            // upstream Codex does not send one, but tolerating it here keeps
            // the parser robust against future upstream changes.
            if payload.trim() == "[DONE]" {
                continue;
            }

            match serde_json::from_str::<serde_json::Value>(&payload) {
                Ok(v) => out.push(dispatch_to_event(v)),
                Err(e) => out.push(Err(CodexError::Sse(format!(
                    "JSON decode failed at SSE record boundary: {}",
                    e
                )))),
            }
        }
        out
    }
}

/// Dispatch a JSON Value to the right `ResponsesEvent` variant based on the
/// top-level `type` field. Pure data shuffling; isolated from `feed` so the
/// framer stays under the 80 LOC budget called out in design §Dependencies.
fn dispatch_to_event(v: serde_json::Value) -> Result<ResponsesEvent, CodexError> {
    let event_type = v
        .get("type")
        .and_then(|t| t.as_str())
        .ok_or_else(|| CodexError::Sse("SSE event missing `type` field".into()))?
        .to_owned();
    // Always read sequence_number, even on `Other`, so downstream ordering
    // invariants (Req 13.4) hold uniformly across variants.
    let sequence_number = v
        .get("sequence_number")
        .and_then(|s| s.as_u64())
        .unwrap_or(0);

    match event_type.as_str() {
        "response.created" => {
            let resp = v.get("response").ok_or_else(|| {
                CodexError::Sse("response.created missing `response` object".into())
            })?;
            let id = resp
                .get("id")
                .and_then(|x| x.as_str())
                .ok_or_else(|| CodexError::Sse("response.created missing `response.id`".into()))?
                .to_owned();
            let model = resp
                .get("model")
                .and_then(|x| x.as_str())
                .ok_or_else(|| {
                    CodexError::Sse("response.created missing `response.model`".into())
                })?
                .to_owned();
            Ok(ResponsesEvent::Created { id, model, sequence_number })
        }
        "response.output_text.delta" => {
            let item_id = str_field(&v, "item_id", "output_text.delta")?;
            let delta = str_field(&v, "delta", "output_text.delta")?;
            Ok(ResponsesEvent::OutputTextDelta { item_id, delta, sequence_number })
        }
        "response.output_text.done" => {
            let item_id = str_field(&v, "item_id", "output_text.done")?;
            let text = str_field(&v, "text", "output_text.done")?;
            Ok(ResponsesEvent::OutputTextDone { item_id, text, sequence_number })
        }
        "response.output_item.added" => {
            let item = v.get("item").ok_or_else(|| {
                CodexError::Sse("output_item.added missing `item` object".into())
            })?;
            let kind = item
                .get("type")
                .and_then(|x| x.as_str())
                .ok_or_else(|| {
                    CodexError::Sse("output_item.added missing `item.type`".into())
                })?
                .to_owned();
            let item_id = item
                .get("id")
                .and_then(|x| x.as_str())
                .unwrap_or_default()
                .to_owned();
            let call_id = item.get("call_id").and_then(|x| x.as_str()).map(str::to_owned);
            let name = item.get("name").and_then(|x| x.as_str()).map(str::to_owned);
            Ok(ResponsesEvent::OutputItemAdded {
                item_id,
                kind,
                call_id,
                name,
                sequence_number,
            })
        }
        "response.function_call_arguments.delta" => {
            let item_id = str_field(&v, "item_id", "function_call_arguments.delta")?;
            let delta = str_field(&v, "delta", "function_call_arguments.delta")?;
            Ok(ResponsesEvent::FunctionCallArgumentsDelta {
                item_id,
                delta,
                sequence_number,
            })
        }
        "response.completed" => {
            let resp_v = v.get("response").cloned().ok_or_else(|| {
                CodexError::Sse("response.completed missing `response` object".into())
            })?;
            let response: CompletedPayload = serde_json::from_value(resp_v).map_err(|e| {
                CodexError::Sse(format!("response.completed payload decode failed: {}", e))
            })?;
            Ok(ResponsesEvent::Completed { response, sequence_number })
        }
        "response.failed" => {
            let error = decode_error_payload(&v, "response.failed")?;
            Ok(ResponsesEvent::Failed { error, sequence_number })
        }
        "error" => {
            let error = decode_error_payload(&v, "error")?;
            Ok(ResponsesEvent::Error { error, sequence_number })
        }
        _ => Ok(ResponsesEvent::Other {
            kind: event_type,
            sequence_number,
        }),
    }
}

fn str_field(v: &serde_json::Value, field: &str, ctx: &str) -> Result<String, CodexError> {
    v.get(field)
        .and_then(|x| x.as_str())
        .ok_or_else(|| CodexError::Sse(format!("{} missing `{}`", ctx, field)))
        .map(str::to_owned)
}

fn decode_error_payload(v: &serde_json::Value, ctx: &str) -> Result<ErrorPayload, CodexError> {
    match v.get("error").cloned() {
        None | Some(serde_json::Value::Null) => Ok(ErrorPayload {
            message: None,
            code: None,
            kind: None,
        }),
        Some(err_v) => serde_json::from_value(err_v)
            .map_err(|e| CodexError::Sse(format!("{} error envelope decode failed: {}", ctx, e))),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Smoke test: empty input produces no events. Full unit-test battery
    /// (record splitting across feeds, every variant, malformed-JSON
    /// resilience) lives in task 9.2.
    #[test]
    fn feed_empty_returns_no_events() {
        let mut p = SseLineParser::new();
        let out = p.feed(b"");
        assert!(out.is_empty());
    }

    /// Smoke test: a minimal `response.created` record decodes into the
    /// `Created` variant with the correct fields.
    #[test]
    fn feed_minimal_created_event_decodes() {
        let mut p = SseLineParser::new();
        let frame = b"event: response.created\n\
                      data: {\"type\":\"response.created\",\"sequence_number\":0,\
                      \"response\":{\"id\":\"resp_abc\",\"model\":\"gpt-5\"}}\n\n";
        let out = p.feed(frame);
        assert_eq!(out.len(), 1, "expected exactly one event");
        match out.into_iter().next().unwrap() {
            Ok(ResponsesEvent::Created { id, model, sequence_number }) => {
                assert_eq!(id, "resp_abc");
                assert_eq!(model, "gpt-5");
                assert_eq!(sequence_number, 0);
            }
            other => panic!("expected ResponsesEvent::Created, got {:?}", other),
        }
    }

    // ---- helper: drain a single event from a fresh parser fed one frame ----
    fn feed_one(frame: &[u8]) -> Result<ResponsesEvent, CodexError> {
        let mut p = SseLineParser::new();
        let mut out = p.feed(frame);
        assert_eq!(out.len(), 1, "expected exactly one event from frame");
        out.remove(0)
    }

    // 1. OutputTextDelta decode
    #[test]
    fn output_text_delta_decodes() {
        let frame = b"data: {\"type\":\"response.output_text.delta\",\
                      \"sequence_number\":3,\"item_id\":\"item_1\",\"delta\":\"Hel\"}\n\n";
        match feed_one(frame).expect("decode ok") {
            ResponsesEvent::OutputTextDelta { item_id, delta, sequence_number } => {
                assert_eq!(item_id, "item_1");
                assert_eq!(delta, "Hel");
                assert_eq!(sequence_number, 3);
            }
            other => panic!("expected OutputTextDelta, got {:?}", other),
        }
    }

    // 2. OutputTextDone decode
    #[test]
    fn output_text_done_decodes() {
        let frame = b"data: {\"type\":\"response.output_text.done\",\
                      \"sequence_number\":7,\"item_id\":\"item_1\",\"text\":\"Hello world\"}\n\n";
        match feed_one(frame).expect("decode ok") {
            ResponsesEvent::OutputTextDone { item_id, text, sequence_number } => {
                assert_eq!(item_id, "item_1");
                assert_eq!(text, "Hello world");
                assert_eq!(sequence_number, 7);
            }
            other => panic!("expected OutputTextDone, got {:?}", other),
        }
    }

    // 3. OutputItemAdded decode for function_call item type with call_id and name
    #[test]
    fn output_item_added_function_call_decodes() {
        let frame = b"data: {\"type\":\"response.output_item.added\",\
                      \"sequence_number\":11,\
                      \"item\":{\"type\":\"function_call\",\"id\":\"fc_1\",\
                      \"call_id\":\"call_abc\",\"name\":\"get_weather\"}}\n\n";
        match feed_one(frame).expect("decode ok") {
            ResponsesEvent::OutputItemAdded {
                item_id,
                kind,
                call_id,
                name,
                sequence_number,
            } => {
                assert_eq!(item_id, "fc_1");
                assert_eq!(kind, "function_call");
                assert_eq!(call_id.as_deref(), Some("call_abc"));
                assert_eq!(name.as_deref(), Some("get_weather"));
                assert_eq!(sequence_number, 11);
            }
            other => panic!("expected OutputItemAdded, got {:?}", other),
        }
    }

    // 4. FunctionCallArgumentsDelta decode
    #[test]
    fn function_call_arguments_delta_decodes() {
        let frame = b"data: {\"type\":\"response.function_call_arguments.delta\",\
                      \"sequence_number\":13,\"item_id\":\"fc_1\",\
                      \"delta\":\"{\\\"city\\\":\\\"Pa\"}\n\n";
        match feed_one(frame).expect("decode ok") {
            ResponsesEvent::FunctionCallArgumentsDelta {
                item_id,
                delta,
                sequence_number,
            } => {
                assert_eq!(item_id, "fc_1");
                assert_eq!(delta, "{\"city\":\"Pa");
                assert_eq!(sequence_number, 13);
            }
            other => panic!("expected FunctionCallArgumentsDelta, got {:?}", other),
        }
    }

    // 5. Completed decode with output (one message + one function_call), usage,
    //    incomplete_details {reason: "max_output_tokens"}
    #[test]
    fn completed_with_output_usage_and_incomplete_details_decodes() {
        let frame = b"data: {\"type\":\"response.completed\",\"sequence_number\":99,\
                      \"response\":{\"id\":\"resp_done\",\"model\":\"gpt-5.5\",\
                      \"output\":[\
                        {\"type\":\"message\",\"id\":\"msg_1\",\"content\":[\
                            {\"type\":\"output_text\",\"text\":\"Hello\"}]},\
                        {\"type\":\"function_call\",\"id\":\"fc_1\",\
                          \"call_id\":\"call_abc\",\"name\":\"get_weather\",\
                          \"arguments\":\"{\\\"city\\\":\\\"Paris\\\"}\"}\
                      ],\
                      \"usage\":{\"input_tokens\":10,\"output_tokens\":5,\"total_tokens\":15},\
                      \"incomplete_details\":{\"reason\":\"max_output_tokens\"}}}\n\n";
        match feed_one(frame).expect("decode ok") {
            ResponsesEvent::Completed { response, sequence_number } => {
                assert_eq!(sequence_number, 99);
                assert_eq!(response.id, "resp_done");
                assert_eq!(response.model, "gpt-5.5");
                assert_eq!(response.output.len(), 2);

                match (&response.output[0].kind[..], &response.output[1].kind[..]) {
                    ("message", "function_call") => {}
                    other => panic!("unexpected output kinds: {:?}", other),
                }

                // message item carries output_text content
                let msg = &response.output[0];
                let content = msg.content.as_ref().expect("message content present");
                assert_eq!(content.len(), 1);
                assert_eq!(content[0].kind, "output_text");
                assert_eq!(content[0].text.as_deref(), Some("Hello"));

                // function_call item carries call_id, name, arguments
                let fc = &response.output[1];
                assert_eq!(fc.call_id.as_deref(), Some("call_abc"));
                assert_eq!(fc.name.as_deref(), Some("get_weather"));
                assert_eq!(fc.arguments.as_deref(), Some("{\"city\":\"Paris\"}"));

                let usage = response.usage.expect("usage present");
                assert_eq!(usage.input_tokens, Some(10));
                assert_eq!(usage.output_tokens, Some(5));
                assert_eq!(usage.total_tokens, Some(15));

                let incomplete = response
                    .incomplete_details
                    .expect("incomplete_details present");
                assert_eq!(incomplete.reason.as_deref(), Some("max_output_tokens"));
            }
            other => panic!("expected Completed, got {:?}", other),
        }
    }

    // 6. Failed decode with full ErrorPayload
    #[test]
    fn failed_with_full_error_payload_decodes() {
        let frame = b"data: {\"type\":\"response.failed\",\"sequence_number\":42,\
                      \"error\":{\"message\":\"upstream blew up\",\
                      \"code\":\"server_error\",\"type\":\"api_error\"}}\n\n";
        match feed_one(frame).expect("decode ok") {
            ResponsesEvent::Failed { error, sequence_number } => {
                assert_eq!(sequence_number, 42);
                assert_eq!(error.message.as_deref(), Some("upstream blew up"));
                assert_eq!(error.code.as_deref(), Some("server_error"));
                assert_eq!(error.kind.as_deref(), Some("api_error"));
            }
            other => panic!("expected Failed, got {:?}", other),
        }
    }

    // 7. Error (top-level "error" type) decode
    #[test]
    fn top_level_error_decodes() {
        let frame = b"data: {\"type\":\"error\",\"sequence_number\":2,\
                      \"error\":{\"message\":\"bad token\",\"code\":\"invalid_request_error\",\
                      \"type\":\"invalid_request_error\"}}\n\n";
        match feed_one(frame).expect("decode ok") {
            ResponsesEvent::Error { error, sequence_number } => {
                assert_eq!(sequence_number, 2);
                assert_eq!(error.message.as_deref(), Some("bad token"));
                assert_eq!(error.code.as_deref(), Some("invalid_request_error"));
                assert_eq!(error.kind.as_deref(), Some("invalid_request_error"));
            }
            other => panic!("expected Error, got {:?}", other),
        }
    }

    // 8. Other captures unknown kind (e.g. response.web_search_call.in_progress)
    #[test]
    fn other_captures_unknown_kind() {
        let frame = b"data: {\"type\":\"response.web_search_call.in_progress\",\
                      \"sequence_number\":17}\n\n";
        match feed_one(frame).expect("decode ok") {
            ResponsesEvent::Other { kind, sequence_number } => {
                assert_eq!(kind, "response.web_search_call.in_progress");
                assert_eq!(sequence_number, 17);
            }
            other => panic!("expected Other, got {:?}", other),
        }
    }

    // 9. Record split inside boundary across two feed calls
    //    (the JSON payload is split mid-bytes between two chunks)
    #[test]
    fn record_split_inside_payload_across_feeds() {
        let mut p = SseLineParser::new();
        // First chunk delivers only part of the JSON payload — no boundary yet.
        let part1 = b"data: {\"type\":\"response.output_text.delta\",\
                      \"sequence_number\":4,\"item_id\":\"it";
        let out1 = p.feed(part1);
        assert!(out1.is_empty(), "no event yet, payload incomplete");

        // Second chunk completes the payload and the record boundary.
        let part2 = b"em_2\",\"delta\":\"lo \"}\n\n";
        let out2 = p.feed(part2);
        assert_eq!(out2.len(), 1);
        match out2.into_iter().next().unwrap().expect("decode ok") {
            ResponsesEvent::OutputTextDelta { item_id, delta, sequence_number } => {
                assert_eq!(item_id, "item_2");
                assert_eq!(delta, "lo ");
                assert_eq!(sequence_number, 4);
            }
            other => panic!("expected OutputTextDelta, got {:?}", other),
        }
    }

    // 10. Boundary split (`\n` then `\n` in two feeds)
    #[test]
    fn boundary_split_across_feeds() {
        let mut p = SseLineParser::new();
        // Whole record body + first newline of the boundary, but not the second.
        let part1 = b"data: {\"type\":\"response.output_text.delta\",\
                      \"sequence_number\":5,\"item_id\":\"it_3\",\"delta\":\"x\"}\n";
        let out1 = p.feed(part1);
        assert!(out1.is_empty(), "no event until second newline arrives");

        // Second `\n` completes the `\n\n` boundary on its own.
        let out2 = p.feed(b"\n");
        assert_eq!(out2.len(), 1);
        match out2.into_iter().next().unwrap().expect("decode ok") {
            ResponsesEvent::OutputTextDelta { item_id, delta, sequence_number } => {
                assert_eq!(item_id, "it_3");
                assert_eq!(delta, "x");
                assert_eq!(sequence_number, 5);
            }
            other => panic!("expected OutputTextDelta, got {:?}", other),
        }
    }

    // 11. Multi-line `data:` accumulation per SSE spec
    //     Two `data:` lines in one record are joined with `\n` and the
    //     result must parse as a single JSON value.
    #[test]
    fn multi_line_data_lines_concatenate_with_newline() {
        // Splitting JSON across two `data:` lines at a comma boundary yields a
        // payload of `{"type":"response.output_text.delta","sequence_number":6,\n"item_id":"x","delta":"y"}`
        // which is valid JSON (whitespace between tokens is fine).
        let frame = b"data: {\"type\":\"response.output_text.delta\",\"sequence_number\":6,\n\
                      data: \"item_id\":\"x\",\"delta\":\"y\"}\n\n";
        match feed_one(frame).expect("decode ok") {
            ResponsesEvent::OutputTextDelta { item_id, delta, sequence_number } => {
                assert_eq!(item_id, "x");
                assert_eq!(delta, "y");
                assert_eq!(sequence_number, 6);
            }
            other => panic!("expected OutputTextDelta, got {:?}", other),
        }
    }

    // 12. Malformed JSON record doesn't poison subsequent valid records
    #[test]
    fn malformed_json_does_not_poison_subsequent_records() {
        let mut p = SseLineParser::new();
        // First record: invalid JSON. Second record: a valid output_text.delta.
        let frames = b"data: {not valid json\n\n\
                       data: {\"type\":\"response.output_text.delta\",\
                       \"sequence_number\":8,\"item_id\":\"after\",\"delta\":\"ok\"}\n\n";
        let out = p.feed(frames);
        assert_eq!(out.len(), 2, "framer must emit one entry per record");

        match &out[0] {
            Err(CodexError::Sse(_)) => {}
            other => panic!("expected first record to be Sse error, got {:?}", other),
        }
        match &out[1] {
            Ok(ResponsesEvent::OutputTextDelta { item_id, delta, sequence_number }) => {
                assert_eq!(item_id, "after");
                assert_eq!(delta, "ok");
                assert_eq!(*sequence_number, 8);
            }
            other => panic!("expected second record to be OutputTextDelta, got {:?}", other),
        }
    }

    // 13. `[DONE]` sentinel tolerated (no event, no error)
    #[test]
    fn done_sentinel_is_silently_dropped() {
        let mut p = SseLineParser::new();
        // [DONE] alone produces no entries.
        let out1 = p.feed(b"data: [DONE]\n\n");
        assert!(out1.is_empty(), "[DONE] must not emit any event or error");

        // A subsequent valid record after [DONE] still decodes cleanly.
        let out2 = p.feed(
            b"data: {\"type\":\"response.output_text.delta\",\
              \"sequence_number\":1,\"item_id\":\"id\",\"delta\":\"d\"}\n\n",
        );
        assert_eq!(out2.len(), 1);
        match out2.into_iter().next().unwrap().expect("decode ok") {
            ResponsesEvent::OutputTextDelta { item_id, delta, sequence_number } => {
                assert_eq!(item_id, "id");
                assert_eq!(delta, "d");
                assert_eq!(sequence_number, 1);
            }
            other => panic!("expected OutputTextDelta, got {:?}", other),
        }
    }

    // 14. Non-data lines (`event:`, comment `:`, `retry:`) ignored
    #[test]
    fn non_data_lines_are_ignored() {
        // `event:`, comment line `:`, and `retry:` must all be discarded; only
        // the `data:` line drives event dispatch.
        let frame = b": this is a comment line\n\
                      event: response.output_text.delta\n\
                      retry: 1500\n\
                      data: {\"type\":\"response.output_text.delta\",\
                      \"sequence_number\":12,\"item_id\":\"only\",\"delta\":\"z\"}\n\n";
        match feed_one(frame).expect("decode ok") {
            ResponsesEvent::OutputTextDelta { item_id, delta, sequence_number } => {
                assert_eq!(item_id, "only");
                assert_eq!(delta, "z");
                assert_eq!(sequence_number, 12);
            }
            other => panic!("expected OutputTextDelta, got {:?}", other),
        }
    }
}
