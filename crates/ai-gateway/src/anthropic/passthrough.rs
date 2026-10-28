//! Native Anthropic passthrough side-channel for the Bedrock Mantle Messages
//! path (spec `anthropic-messages-front-door` decision D2, Req 4.3).
//!
//! The raw Anthropic request body and the client's `anthropic-version` /
//! `anthropic-beta` header values must reach the Mantle Messages adapter
//! verbatim, but they must NEVER appear in any other provider's upstream
//! request body. Markers stashed in `OpenAIRequest.extra` cannot guarantee
//! that: the generic buffered/pass-through dispatch serializes the whole
//! request (including flattened `extra`) inside the router, outside this
//! module's control. So the stash travels out-of-band in a task-local that
//! only the Messages adapter consults, and the only in-band artifact is the
//! relayed raw response parked under a reserved `OpenAIResponse.extra` key
//! that the `/v1/messages` handler short-circuits on before serialization.

use axum::http::HeaderMap;

use crate::error::GatewayError;
use crate::models::openai::{Choice, Message, OpenAIResponse, Usage};
use crate::router::cache_cost::token_field_from;

/// Reserved response-extras key carrying the verbatim upstream Anthropic JSON
/// (non-streaming wire). Never placed in any request body.
pub const VERBATIM_RESPONSE_KEY: &str = "x-obey-anthropic-verbatim-response";

/// Reserved response-extras key carrying the verbatim upstream Anthropic SSE
/// text (streaming wire). Never placed in any request body.
pub const VERBATIM_SSE_KEY: &str = "x-obey-anthropic-verbatim-sse";

/// Request-side passthrough context: the exact client JSON body (parsed once
/// as a `serde_json::Value`, never round-tripped through typed structs so
/// unknown fields survive) plus the two headers forwarded verbatim (Req 4.3).
#[derive(Debug, Clone)]
pub struct PassthroughContext {
    pub body: serde_json::Value,
    pub anthropic_version: String,
    pub anthropic_beta: Option<String>,
}

tokio::task_local! {
    /// Set by the `/v1/messages` handler for the duration of one dispatch.
    /// `None` (or an unset task-local, e.g. the plain chat path) means the
    /// legacy translated Messages adapter runs unchanged.
    pub static ANTHROPIC_PASSTHROUGH: Option<PassthroughContext>;
}

/// Runs a future under the passthrough task-local. Constructed through
/// [`Self::new`] by the `/v1/messages` handler.
pub struct AnthropicPassthroughScope {
    context: Option<PassthroughContext>,
}

impl AnthropicPassthroughScope {
    pub fn new(
        body: serde_json::Value,
        anthropic_version: String,
        anthropic_beta: Option<String>,
    ) -> Self {
        Self {
            context: Some(PassthroughContext {
                body,
                anthropic_version,
                anthropic_beta,
            }),
        }
    }

    /// Run `fut` with the passthrough context installed.
    pub async fn scope<F>(self, fut: F) -> F::Output
    where
        F: std::future::Future,
    {
        ANTHROPIC_PASSTHROUGH
            .scope(self.context, fut)
            .await
    }
}

/// Current passthrough context, if the request arrived through the
/// `/v1/messages` front door on this task.
pub fn current() -> Option<PassthroughContext> {
    ANTHROPIC_PASSTHROUGH
        .try_with(|value| value.clone())
        .ok()
        .flatten()
}

/// Deterministic single value for `anthropic-beta`: multiple arriving header
/// lines are joined with `,` in arrival order, matching the documented
/// comma-separated open-list form (Req 4.3).
pub fn beta_header_value(headers: &HeaderMap) -> Option<String> {
    let values: Vec<String> = headers
        .get_all("anthropic-beta")
        .iter()
        .filter_map(|value| value.to_str().ok())
        .map(str::to_string)
        .collect();
    if values.is_empty() {
        None
    } else {
        Some(values.join(","))
    }
}

/// Anthropic usage object → chat `Usage`, preserving the cache-token split in
/// `extra` so the router's cache-aware cost math (`extract_cache_usage`) and
/// the existing logging pipeline work unchanged (Req 6.2).
pub fn usage_to_chat(usage: &serde_json::Value) -> Usage {
    let field = |name: &str| {
        token_field_from(usage.get(name)).unwrap_or(0)
    };
    let input = field("input_tokens");
    let output = field("output_tokens");
    let cache_read = field("cache_read_input_tokens");
    let cache_creation = field("cache_creation_input_tokens");
    let mut extra = serde_json::Map::new();
    if cache_read > 0 {
        extra.insert(
            "cache_read_input_tokens".to_string(),
            serde_json::json!(cache_read),
        );
    }
    if cache_creation > 0 {
        extra.insert(
            "cache_creation_input_tokens".to_string(),
            serde_json::json!(cache_creation),
        );
    }
    let prompt_tokens = u32::try_from(input).unwrap_or(u32::MAX);
    let completion_tokens = u32::try_from(output).unwrap_or(u32::MAX);
    Usage {
        prompt_tokens,
        completion_tokens,
        total_tokens: prompt_tokens.saturating_add(completion_tokens),
        extra,
    }
}

/// Map an Anthropic `stop_reason` onto the chat `finish_reason` vocabulary so
/// the router's existing gates (truncation detection, tool-call repair) see a
/// familiar shape.
fn finish_from_stop_reason(stop_reason: &str) -> String {
    match stop_reason {
        "max_tokens" => "length".to_string(),
        "tool_use" => "tool_calls".to_string(),
        "content_filter" | "refusal" => "content_filter".to_string(),
        _ => "stop".to_string(),
    }
}

/// One parsed SSE frame (`event:` name plus `data:` payload).
struct SseFrame {
    event: Option<String>,
    data: String,
}

/// Split raw upstream SSE text into frames. Multi-line `data:` payloads are
/// joined with `\n` per the SSE spec; comment lines (`:`-prefixed) and
/// unknown fields are dropped.
fn parse_sse_frames(text: &str) -> Vec<SseFrame> {
    let mut frames = Vec::new();
    let mut event: Option<String> = None;
    let mut data: Vec<String> = Vec::new();
    let flush = |event: &mut Option<String>, data: &mut Vec<String>, frames: &mut Vec<SseFrame>| {
        if event.is_some() || !data.is_empty() {
            frames.push(SseFrame {
                event: event.take(),
                data: data.join("\n"),
            });
        }
        data.clear();
    };
    for line in text.lines() {
        let line = line.trim_end_matches('\r');
        if line.is_empty() {
            flush(&mut event, &mut data, &mut frames);
            continue;
        }
        if let Some(rest) = line.strip_prefix("event:") {
            event = Some(rest.trim_start().to_string());
        } else if let Some(rest) = line.strip_prefix("data:") {
            data.push(rest.trim_start().to_string());
        }
    }
    flush(&mut event, &mut data, &mut frames);
    frames
}

/// Accumulated chat-shaped view of an upstream Anthropic SSE stream. Only the
/// fields the router gates and the cost pipeline need are extracted; the raw
/// text is relayed to the client untouched.
struct StreamAccumulator {
    id: Option<String>,
    model: Option<String>,
    content: String,
    reasoning: String,
    tool_calls: Vec<serde_json::Value>,
    pending_tool: Option<PendingToolUse>,
    finish_reason: Option<String>,
    usage: Option<serde_json::Value>,
}

struct PendingToolUse {
    id: String,
    name: String,
    arguments: String,
}

impl StreamAccumulator {
    fn new() -> Self {
        Self {
            id: None,
            model: None,
            content: String::new(),
            reasoning: String::new(),
            tool_calls: Vec::new(),
            pending_tool: None,
            finish_reason: None,
            usage: None,
        }
    }

    fn feed(&mut self, event: &str, data: &str) {
        let Ok(value) = serde_json::from_str::<serde_json::Value>(data) else {
            return;
        };
        match event {
            "message_start" => {
                let message = value.get("message");
                self.id = message
                    .and_then(|m| m.get("id"))
                    .and_then(|v| v.as_str())
                    .map(str::to_string);
                self.model = message
                    .and_then(|m| m.get("model"))
                    .and_then(|v| v.as_str())
                    .map(str::to_string);
                if let Some(usage) = message.and_then(|m| m.get("usage")) {
                    self.usage = Some(usage.clone());
                }
            }
            "content_block_start" => {
                let block = value.get("content_block");
                if block
                    .and_then(|b| b.get("type"))
                    .and_then(|v| v.as_str())
                    == Some("tool_use")
                {
                    self.pending_tool = Some(PendingToolUse {
                        id: block
                            .and_then(|b| b.get("id"))
                            .and_then(|v| v.as_str())
                            .unwrap_or_default()
                            .to_string(),
                        name: block
                            .and_then(|b| b.get("name"))
                            .and_then(|v| v.as_str())
                            .unwrap_or_default()
                            .to_string(),
                        arguments: String::new(),
                    });
                }
            }
            "content_block_delta" => {
                // Anthropic delta shape is flat: {"type":"text_delta","text":..}
                // — the payload key is a sibling of `type`, not nested.
                let delta = value.get("delta");
                let delta_type = delta
                    .and_then(|d| d.get("type"))
                    .and_then(|v| v.as_str())
                    .unwrap_or_default();
                if delta_type == "text_delta" {
                    if let Some(text) = delta
                        .and_then(|d| d.get("text"))
                        .and_then(|v| v.as_str())
                    {
                        self.content.push_str(text);
                    }
                }
                if delta_type == "thinking_delta" {
                    if let Some(thinking) = delta
                        .and_then(|d| d.get("thinking"))
                        .and_then(|v| v.as_str())
                    {
                        self.reasoning.push_str(thinking);
                    }
                }
                if delta_type == "input_json_delta" {
                    if let Some(partial) = delta
                        .and_then(|d| d.get("partial_json"))
                        .and_then(|v| v.as_str())
                    {
                        if let Some(tool) = &mut self.pending_tool {
                            tool.arguments.push_str(partial);
                        }
                    }
                }
            }
            "content_block_stop" => {
                if let Some(tool) = self.pending_tool.take() {
                    self.tool_calls.push(tool_use_to_chat_call(
                        &tool.id,
                        &tool.name,
                        serde_json::Value::String(tool.arguments),
                    ));
                }
            }
            "message_delta" => {
                // Anthropic sends cumulative usage: output_tokens arrives
                // here, input_tokens only in message_start — merge fields.
                if let Some(usage) = value.get("usage").filter(|u| !u.is_null()) {
                    let merged = self
                        .usage
                        .take()
                        .unwrap_or_else(|| serde_json::Value::Object(Default::default()));
                    if let (Some(target), Some(source)) =
                        (merged.as_object(), usage.as_object())
                    {
                        let mut map = target.clone();
                        for (key, value) in source {
                            map.insert(key.clone(), value.clone());
                        }
                        self.usage = Some(serde_json::Value::Object(map));
                    } else {
                        self.usage = Some(usage.clone());
                    }
                }
                if let Some(stop) = value
                    .get("delta")
                    .and_then(|d| d.get("stop_reason"))
                    .and_then(|v| v.as_str())
                {
                    self.finish_reason = Some(finish_from_stop_reason(stop));
                }
            }
            _ => {}
        }
    }
}

/// Chat-shaped `tool_calls` entry from an Anthropic `tool_use` block. The
/// `input` object is serialized back to a JSON string (chat wire shape).
fn tool_use_to_chat_call(
    id: &str,
    name: &str,
    arguments: serde_json::Value,
) -> serde_json::Value {
    serde_json::json!({
        "id": id,
        "type": "function",
        "function": {
            "name": name,
            "arguments": match arguments {
                serde_json::Value::String(s) => s,
                other => other.to_string(),
            },
        },
    })
}

/// Assemble the chat-shaped `OpenAIResponse` the router expects, with the
/// verbatim payload parked under `stash_key`.
fn build_chat_response(
    model_fallback: &str,
    id: Option<String>,
    model: Option<String>,
    content: String,
    reasoning: String,
    tool_calls: Vec<serde_json::Value>,
    finish_reason: Option<String>,
    usage: Option<&serde_json::Value>,
    stash_key: &str,
    stash: serde_json::Value,
) -> OpenAIResponse {
    let mut message_extra = serde_json::Map::new();
    if !tool_calls.is_empty() {
        message_extra.insert(
            "tool_calls".to_string(),
            serde_json::Value::Array(tool_calls),
        );
    }
    if !reasoning.is_empty() {
        message_extra.insert(
            "reasoning_content".to_string(),
            serde_json::Value::String(reasoning),
        );
    }
    let finish_reason = finish_reason
        .unwrap_or_else(|| "stop".to_string());
    let mut response = OpenAIResponse {
        id: id.unwrap_or_default(),
        object: "chat.completion".to_string(),
        created: 0,
        model: model.unwrap_or_else(|| model_fallback.to_string()),
        choices: vec![Choice {
            index: 0,
            message: Message {
                role: "assistant".to_string(),
                content: serde_json::Value::String(content),
                extra: message_extra,
            },
            finish_reason: Some(finish_reason),
            extra: Default::default(),
        }],
        usage: usage.map(usage_to_chat).unwrap_or_default(),
        extra: Default::default(),
    };
    response.extra.insert(stash_key.to_string(), stash);
    response
}

/// Build the passthrough response from an upstream Anthropic wire body
/// (JSON or SSE text, decided by shape). The raw payload is preserved under
/// its reserved extras key for the handler to relay verbatim; the visible
/// chat shape exists only so the router's success gates, cost math, and
/// logging run exactly as on the translated path.
pub fn verbatim_response_from_wire(
    body: &str,
    model_fallback: &str,
    provider_name: &str,
) -> Result<OpenAIResponse, GatewayError> {
    if body.trim_start().starts_with('{') {
        let value: serde_json::Value = serde_json::from_str(body).map_err(|error| {
            GatewayError::Network(format!("Failed to parse Anthropic response: {}", error))
        })?;
        let mut content = String::new();
        let mut reasoning = String::new();
        let mut tool_calls = Vec::new();
        for block in value
            .get("content")
            .and_then(|v| v.as_array())
            .into_iter()
            .flatten()
        {
            match block.get("type").and_then(|v| v.as_str()) {
                Some("text") => {
                    if let Some(text) = block.get("text").and_then(|v| v.as_str()) {
                        content.push_str(text);
                    }
                }
                Some("thinking") => {
                    if let Some(text) = block.get("thinking").and_then(|v| v.as_str()) {
                        reasoning.push_str(text);
                    }
                }
                Some("tool_use") => {
                    tool_calls.push(tool_use_to_chat_call(
                        block.get("id").and_then(|v| v.as_str()).unwrap_or_default(),
                        block.get("name").and_then(|v| v.as_str()).unwrap_or_default(),
                        block.get("input").cloned().unwrap_or_default(),
                    ));
                }
                _ => {}
            }
        }
        let finish_reason = value
            .get("stop_reason")
            .and_then(|v| v.as_str())
            .map(finish_from_stop_reason);
        return Ok(build_chat_response(
            model_fallback,
            value.get("id").and_then(|v| v.as_str()).map(str::to_string),
            value.get("model").and_then(|v| v.as_str()).map(str::to_string),
            content,
            reasoning,
            tool_calls,
            finish_reason,
            value.get("usage"),
            VERBATIM_RESPONSE_KEY,
            value.clone(),
        ));
    }

    let mut accumulator = StreamAccumulator::new();
    for frame in parse_sse_frames(body) {
        let event_name = frame.event.as_deref().unwrap_or("");
        if event_name == "error" {
            let message = serde_json::from_str::<serde_json::Value>(&frame.data)
                .ok()
                .and_then(|value| {
                    value
                        .pointer("/error/message")
                        .and_then(|v| v.as_str())
                        .map(str::to_string)
                })
                .unwrap_or_else(|| "upstream Anthropic stream error".to_string());
            return Err(GatewayError::Provider {
                provider: provider_name.to_string(),
                message: format!("Anthropic stream error: {}", message),
                status_code: Some(502),
            });
        }
        if !frame.data.is_empty() {
            accumulator.feed(event_name, &frame.data);
        }
    }
    Ok(build_chat_response(
        model_fallback,
        accumulator.id,
        accumulator.model,
        accumulator.content,
        accumulator.reasoning,
        accumulator.tool_calls,
        accumulator.finish_reason,
        accumulator.usage.as_ref(),
        VERBATIM_SSE_KEY,
        serde_json::Value::String(body.to_string()),
    ))
}

/// Verbatim relay frames for the streaming handler: `(event, data)` pairs in
/// upstream order. Anthropic emits one `event:` + single-line `data:` per
/// frame, so re-emitting the pairs through axum's SSE encoder preserves the
/// client-visible stream; axum's `Event::data` newline-splitting would
/// corrupt a whole multi-frame body passed as one string. Event-less frames
/// are relayed as data-only frames (no invented event name) so the client
/// sees exactly the shape the upstream sent.
pub fn sse_relay_frames(text: &str) -> Vec<(String, String)> {
    parse_sse_frames(text)
        .into_iter()
        .filter(|frame| !frame.data.is_empty())
        .map(|frame| (frame.event.unwrap_or_default(), frame.data))
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn beta_header_value_joins_multiple_headers_with_comma() {
        let mut headers = HeaderMap::new();
        headers.append("anthropic-beta", "context-1m-2025-08-07".parse().unwrap());
        headers.append("anthropic-beta", "pdfs-2024-09-25".parse().unwrap());
        assert_eq!(
            beta_header_value(&headers).as_deref(),
            Some("context-1m-2025-08-07,pdfs-2024-09-25")
        );
    }

    #[test]
    fn beta_header_value_absent_when_not_sent() {
        assert_eq!(beta_header_value(&HeaderMap::new()), None);
    }

    #[test]
    fn usage_to_chat_maps_all_four_anthropic_fields() {
        let usage = usage_to_chat(&serde_json::json!({
            "input_tokens": 100,
            "output_tokens": 7,
            "cache_read_input_tokens": 40,
            "cache_creation_input_tokens": 10
        }));
        assert_eq!(usage.prompt_tokens, 100);
        assert_eq!(usage.completion_tokens, 7);
        assert_eq!(usage.total_tokens, 107);
        assert_eq!(
            usage.extra.get("cache_read_input_tokens").and_then(|v| v.as_u64()),
            Some(40)
        );
        assert_eq!(
            usage.extra.get("cache_creation_input_tokens").and_then(|v| v.as_u64()),
            Some(10)
        );
        // Router cost math must see the same split it computes for the chat path.
        let cache = crate::router::cache_cost::extract_cache_usage(&usage);
        assert_eq!(cache.cache_read_input_tokens, 40);
        assert_eq!(cache.cache_creation_input_tokens, 10);
    }

    #[test]
    fn json_wire_preserves_raw_body_and_extracts_usage() {
        let raw = serde_json::json!({
            "id": "msg_9",
            "type": "message",
            "role": "assistant",
            "model": "anthropic.claudesonnet",
            "content": [
                {"type": "thinking", "thinking": "pondering"},
                {"type": "text", "text": "hello "},
                {"type": "text", "text": "world"},
                {"type": "tool_use", "id": "toolu_1", "name": "get_weather",
                 "input": {"city": "Berlin"}}
            ],
            "stop_reason": "tool_use",
            "usage": {"input_tokens": 50, "output_tokens": 5}
        });
        let response =
            verbatim_response_from_wire(&raw.to_string(), "fallback", "bedrock").expect("parses");
        assert_eq!(response.extra.get(VERBATIM_RESPONSE_KEY), Some(&raw));
        assert_eq!(response.id, "msg_9");
        assert_eq!(response.model, "anthropic.claudesonnet");
        let choice = &response.choices[0];
        assert_eq!(choice.message.content, serde_json::json!("hello world"));
        assert_eq!(choice.finish_reason.as_deref(), Some("tool_calls"));
        assert_eq!(choice.message.extra.get("reasoning_content").and_then(|v| v.as_str()), Some("pondering"));
        let tool_calls = choice.message.extra.get("tool_calls").unwrap();
        assert_eq!(tool_calls[0]["id"], "toolu_1");
        assert_eq!(tool_calls[0]["function"]["name"], "get_weather");
        assert_eq!(tool_calls[0]["function"]["arguments"], "{\"city\":\"Berlin\"}");
        assert_eq!(response.usage.prompt_tokens, 50);
        assert_eq!(response.usage.completion_tokens, 5);
    }

    #[test]
    fn json_wire_tool_only_turn_passes_router_content_gate() {
        let raw = serde_json::json!({
            "id": "msg_t",
            "content": [{"type": "tool_use", "id": "t1", "name": "f", "input": {}}],
            "stop_reason": "tool_use",
            "usage": {"input_tokens": 3, "output_tokens": 1}
        });
        let response =
            verbatim_response_from_wire(&raw.to_string(), "m", "bedrock").expect("parses");
        let text = response.choices[0].message.content.as_str().unwrap_or_default();
        // No text content, but a well-formed tool call: the router's
        // response_has_content must accept this shape (it validates
        // id/type/function.name on each call).
        let calls = response.choices[0]
            .message
            .extra
            .get("tool_calls")
            .and_then(|v| v.as_array())
            .expect("tool calls present");
        assert_eq!(calls.len(), 1);
        assert!(text.is_empty());
    }

    #[test]
    fn sse_wire_accumulates_text_tools_and_usage() {
        let raw = concat!(
            "event: message_start\n",
            "data: {\"type\":\"message_start\",\"message\":{\"id\":\"msg_s\",\"model\":\"claude-x\",\"usage\":{\"input_tokens\":11,\"output_tokens\":1}}}\n\n",
            "event: content_block_start\n",
            "data: {\"type\":\"content_block_start\",\"index\":0,\"content_block\":{\"type\":\"tool_use\",\"id\":\"tu_1\",\"name\":\"search\"}}\n\n",
            "event: content_block_delta\n",
            "data: {\"type\":\"content_block_delta\",\"index\":0,\"delta\":{\"type\":\"input_json_delta\",\"partial_json\":\"{\\\"q\\\":\"}}\n\n",
            "event: content_block_delta\n",
            "data: {\"type\":\"content_block_delta\",\"index\":0,\"delta\":{\"type\":\"input_json_delta\",\"partial_json\":\"\\\"hi\\\"}\"}}\n\n",
            "event: content_block_stop\n",
            "data: {\"type\":\"content_block_stop\",\"index\":0}\n\n",
            "event: content_block_start\n",
            "data: {\"type\":\"content_block_start\",\"index\":1,\"content_block\":{\"type\":\"text\",\"text\":\"\"}}\n\n",
            "event: content_block_delta\n",
            "data: {\"type\":\"content_block_delta\",\"index\":1,\"delta\":{\"type\":\"text_delta\",\"text\":\"done\"}}\n\n",
            "event: ping\n",
            "data: {\"type\":\"ping\"}\n\n",
            "event: message_delta\n",
            "data: {\"type\":\"message_delta\",\"delta\":{\"stop_reason\":\"tool_use\"},\"usage\":{\"input_tokens\":12,\"output_tokens\":9,\"cache_read_input_tokens\":4}}\n\n",
            "event: message_stop\n",
            "data: {\"type\":\"message_stop\"}\n\n",
        );
        let response = verbatim_response_from_wire(raw, "m", "bedrock").expect("parses");
        assert_eq!(
            response.extra.get(VERBATIM_SSE_KEY).and_then(|v| v.as_str()),
            Some(raw)
        );
        let choice = &response.choices[0];
        assert_eq!(choice.message.content, serde_json::json!("done"));
        assert_eq!(choice.finish_reason.as_deref(), Some("tool_calls"));
        let calls = choice.message.extra.get("tool_calls").unwrap();
        assert_eq!(calls[0]["function"]["arguments"], "{\"q\":\"hi\"}");
        assert_eq!(response.usage.prompt_tokens, 12);
        assert_eq!(response.usage.completion_tokens, 9);
        assert_eq!(
            response.usage.extra.get("cache_read_input_tokens").and_then(|v| v.as_u64()),
            Some(4)
        );
    }

    #[test]
    fn sse_wire_error_frame_maps_to_provider_error() {
        let raw = "event: error\ndata: {\"type\":\"error\",\"error\":{\"type\":\"overloaded_error\",\"message\":\"saturated\"}}\n\n";
        let err = verbatim_response_from_wire(raw, "m", "bedrock").expect_err("error frame fails");
        match err {
            GatewayError::Provider { provider, status_code, message } => {
                assert_eq!(provider, "bedrock");
                assert_eq!(status_code, Some(502));
                assert!(message.contains("saturated"));
            }
            other => panic!("expected provider error, got {:?}", other),
        }
    }

#[test]
fn sse_relay_frames_splits_frames_preserving_event_names() {
    let raw = "event: message_start\ndata: {\"type\":\"message_start\"}\n\nevent: ping\ndata: {\"type\":\"ping\"}\n\n";
    let frames = sse_relay_frames(raw);
    assert_eq!(frames.len(), 2);
    assert_eq!(frames[0].0, "message_start");
    assert_eq!(frames[0].1, "{\"type\":\"message_start\"}");
    assert_eq!(frames[1].0, "ping");
}

#[test]
fn sse_relay_frames_event_less_frame_is_data_only() {
    let raw = "data: {\"type\":\"legacy\"}\n\n";
    let frames = sse_relay_frames(raw);
    assert_eq!(frames, vec![("".to_string(), "{\"type\":\"legacy\"}".to_string())]);
}

#[test]
fn sse_relay_frames_joins_multi_line_data_with_newlines() {
    let raw = "event: message_start\ndata: {\"a\":1,\ndata: \"b\":2}\n\n";
    let frames = sse_relay_frames(raw);
    assert_eq!(
        frames,
        vec![("message_start".to_string(), "{\"a\":1,\n\"b\":2}".to_string())]
    );
}

    #[test]
    fn task_local_is_none_outside_scope() {
        assert!(current().is_none());
    }

    #[tokio::test]
    async fn task_local_round_trips_inside_scope() {
        let ctx = PassthroughContext {
            body: serde_json::json!({"model": "m"}),
            anthropic_version: "2023-06-01".to_string(),
            anthropic_beta: Some("pdfs-2024-09-25".to_string()),
        };
        ANTHROPIC_PASSTHROUGH
            .scope(Some(ctx), async {
                let seen = current().expect("context visible inside scope");
                assert_eq!(seen.anthropic_version, "2023-06-01");
                assert_eq!(seen.anthropic_beta.as_deref(), Some("pdfs-2024-09-25"));
            })
            .await;
    }
}
