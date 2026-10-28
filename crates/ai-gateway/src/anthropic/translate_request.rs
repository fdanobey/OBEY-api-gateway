//! Anthropic Messages → Chat Completions request translation.
//!
//! Converts an [`AnthropicRequest`] into an [`OpenAIRequest`] for downstream
//! provider dispatch, mirroring the conventions of
//! [`crate::responses::translate_request`] (front door → Chat Completions).
//!
//! Translation rules (spec Requirement 1, 7):
//!
//! - `system` (string or text-block array) → a single **leading** `system`
//!   message (block texts joined with `\n`).
//! - `tool_result` user blocks → one `tool`-role message per block, carrying
//!   `tool_call_id` (the shape the existing pipeline expects: one tool
//!   message per call id, string content when text-only).
//! - assistant `tool_use` blocks → `tool_calls` entries on the assistant
//!   message (`arguments` = serialized `input`).
//! - replayed `thinking` / `redacted_thinking` blocks are preserved verbatim
//!   as content blocks — the reasoning-carrier convention
//!   [`crate::reasoning_compat::detect`] scans for; strip/preserve policy is
//!   applied per attempt downstream by [`crate::reasoning_compat::policy`].
//! - `thinking` config is emitted into `extra["thinking"]` in exactly the
//!   shape [`crate::reasoning_compat::normalize::read_client_spec`] reads;
//!   `disabled` emits nothing (no-injection rule).
//! - flat tools `{name, description, input_schema}` → nested OpenAI
//!   `tools[].function{name, description, parameters}`.
//! - `stop_sequences` → `stop`; `max_tokens`/`temperature`/`top_p`
//!   passthrough; `metadata.user_id` → `user`.
//! - images → `image_url` content parts (base64 becomes a `data:` URL).
//! - `cache_control` markers are dropped everywhere (block extras, message
//!   extras, tool definitions never copy theirs through), so
//!   [`crate::router::cache_inject`] sees a clean request and its own
//!   client-marker strip stays a no-op.
//! - hosted tool types, unknown `tools[].type` values, `document` content
//!   blocks, and `context_management` are rejected pre-dispatch with a
//!   field-naming message (mapped by the handler to 400
//!   `invalid_request_error`).
//!
//! Pure function module: no HTTP, no state, no clock.

use serde_json::{json, Map, Value};

use crate::anthropic::models::{
    AnthropicRequest, AnthropicToolChoice, AnthropicToolDefinition, AnthropicToolKind,
    ContentBlock, ImageSource, MessageContent, SystemPrompt, ThinkingConfig, ToolResultContent,
};
use crate::models::openai::{Message, OpenAIRequest};

/// Translate an Anthropic Messages request into a Chat Completions request.
///
/// # Errors
///
/// Returns a field-naming message for everything rejected pre-dispatch
/// (pattern `X is not supported`):
/// - hosted tool types (`web_search`, `computer`, `code_execution`,
///   `text_editor`, `bash`, `mcp_connector`, date-suffixed variants)
/// - unknown `tools[].type` values
/// - `document` content blocks
/// - `context_management`
pub fn translate(req: &AnthropicRequest) -> Result<OpenAIRequest, String> {
    reject_unsupported(req)?;

    let mut messages = Vec::new();

    if let Some(system) = &req.system {
        let text = system_text(system);
        if !text.is_empty() {
            messages.push(Message {
                role: "system".to_string(),
                content: Value::String(text),
                extra: Map::new(),
            });
        }
    }

    for message in &req.messages {
        translate_message(message, &mut messages);
    }

    let mut extra = Map::new();

    if !req.stop_sequences.is_empty() {
        extra.insert("stop".to_string(), json!(req.stop_sequences));
    }

    if !req.tools.is_empty() {
        extra.insert("tools".to_string(), json!(translate_tools(&req.tools)));
    }

    if let Some(choice) = &req.tool_choice {
        translate_tool_choice(choice, &mut extra);
    }

    if let Some(thinking) = &req.thinking {
        if let Some(value) = thinking_param(thinking) {
            extra.insert("thinking".to_string(), value);
        }
    }

    if let Some(user_id) = req.metadata.as_ref().and_then(|m| m.user_id.as_ref()) {
        extra.insert("user".to_string(), json!(user_id));
    }

    if let Some(top_p) = req.top_p {
        extra.insert("top_p".to_string(), json!(top_p));
    }

    // Pass remaining unknown fields through (e.g. `output_config.effort`,
    // which reasoning-compat normalization reads alongside `thinking`).
    // `context_management` never reaches here (rejected above).
    for (key, value) in &req.extra {
        extra.insert(key.clone(), value.clone());
    }

    Ok(OpenAIRequest {
        model: req.model.clone(),
        messages,
        stream: req.stream,
        temperature: req.temperature,
        max_tokens: Some(req.max_tokens),
        extra,
    })
}

/// Reject unsupported capabilities before any translation side effects.
fn reject_unsupported(req: &AnthropicRequest) -> Result<(), String> {
    if req.extra.contains_key("context_management") {
        return Err("context_management is not supported".to_string());
    }

    for tool in &req.tools {
        match tool.kind() {
            AnthropicToolKind::Custom => {}
            kind => return Err(format!("tool type '{}' is not supported", kind.name())),
        }
    }

    for message in &req.messages {
        if let MessageContent::Blocks(blocks) = &message.content {
            for block in blocks {
                if matches!(block, ContentBlock::Document { .. }) {
                    return Err("content block type 'document' is not supported".to_string());
                }
            }
        }
    }

    Ok(())
}

/// Join a `system` prompt (string or block array) into message text.
fn system_text(system: &SystemPrompt) -> String {
    match system {
        SystemPrompt::Text(text) => text.clone(),
        SystemPrompt::Blocks(blocks) => blocks
            .iter()
            .map(|block| block.text.as_str())
            .collect::<Vec<_>>()
            .join("\n"),
    }
}

/// Translate one Anthropic message, appending one or more chat messages.
///
/// `tool_result` blocks become `tool` messages immediately (in block order);
/// remaining blocks (`text`, `image`, `thinking`, `redacted_thinking`)
/// accumulate into one same-role message appended after them, carrying
/// `tool_calls` when `tool_use` blocks were present. This preserves the
/// chat invariant that tool messages directly follow the assistant
/// `tool_calls` message (Anthropic requires `tool_result` blocks first in a
/// user turn, so the split is order-faithful).
fn translate_message(message: &crate::anthropic::models::AnthropicMessage, out: &mut Vec<Message>) {
    match &message.content {
        MessageContent::Text(text) => out.push(Message {
            role: message.role.clone(),
            content: Value::String(text.clone()),
            extra: Map::new(),
        }),
        MessageContent::Blocks(blocks) => {
            let mut parts: Vec<Value> = Vec::new();
            let mut tool_calls: Vec<Value> = Vec::new();

            for block in blocks {
                match block {
                    ContentBlock::ToolResult {
                        tool_use_id,
                        content,
                        ..
                    } => out.push(tool_message(tool_use_id, content)),
                    ContentBlock::ToolUse { id, name, input, .. } => tool_calls.push(json!({
                        "id": id,
                        "type": "function",
                        "function": {
                            "name": name,
                            "arguments": input.to_string(),
                        },
                    })),
                    ContentBlock::Text { text, .. } => {
                        parts.push(json!({ "type": "text", "text": text }))
                    }
                    ContentBlock::Image { source, .. } => parts.push(image_part(source)),
                    ContentBlock::Thinking { thinking, signature, .. } => {
                        parts.push(json!({
                            "type": "thinking",
                            "thinking": thinking,
                            "signature": signature,
                        }))
                    }
                    ContentBlock::RedactedThinking { data, .. } => {
                        parts.push(json!({ "type": "redacted_thinking", "data": data }))
                    }
                    ContentBlock::Document { .. } => {}
                }
            }

            if parts.is_empty() && tool_calls.is_empty() {
                return;
            }

            let mut extra = Map::new();
            if !tool_calls.is_empty() {
                extra.insert("tool_calls".to_string(), Value::Array(tool_calls));
            }
            let content = if parts.is_empty() {
                Value::Null
            } else {
                Value::Array(parts)
            };
            out.push(Message {
                role: message.role.clone(),
                content,
                extra,
            });
        }
    }
}

/// Build the `tool`-role message for one `tool_result` block.
///
/// String content passes through as a string; block-list content becomes a
/// content array of `text` / `image_url` parts (images in tool results have
/// no string form). `is_error` has no Chat Completions carrier and is
/// dropped.
fn tool_message(tool_use_id: &str, content: &ToolResultContent) -> Message {
    let content = match content {
        ToolResultContent::Text(text) => Value::String(text.clone()),
        ToolResultContent::Blocks(blocks) => Value::Array(
            blocks
                .iter()
                .map(|block| match block {
                    crate::anthropic::models::TextOrImageBlock::Text { text, .. } => {
                        json!({ "type": "text", "text": text })
                    }
                    crate::anthropic::models::TextOrImageBlock::Image { source, .. } => {
                        image_part(source)
                    }
                })
                .collect(),
        ),
    };
    let mut extra = Map::new();
    extra.insert("tool_call_id".to_string(), json!(tool_use_id));
    Message {
        role: "tool".to_string(),
        content,
        extra,
    }
}

/// Translate an image `source` into a chat `image_url` content part.
/// Base64 sources become `data:<media_type>;base64,<data>` URLs.
fn image_part(source: &ImageSource) -> Value {
    let url = match source {
        ImageSource::Base64 { media_type, data, .. } => {
            format!("data:{media_type};base64,{data}")
        }
        ImageSource::Url { url, .. } => url.clone(),
    };
    json!({ "type": "image_url", "image_url": { "url": url } })
}

/// Flat Anthropic tool definitions → nested OpenAI `tools[].function`.
fn translate_tools(tools: &[AnthropicToolDefinition]) -> Vec<Value> {
    tools
        .iter()
        .map(|tool| {
            json!({
                "type": "function",
                "function": {
                    "name": tool.name.clone().unwrap_or_default(),
                    "description": tool.description.clone().unwrap_or_default(),
                    "parameters": tool.input_schema.clone().unwrap_or(json!({ "type": "object" })),
                },
            })
        })
        .collect()
}

/// `tool_choice` → the Chat Completions equivalent; Anthropic's
/// `disable_parallel_tool_use: true` maps to `parallel_tool_calls: false`.
fn translate_tool_choice(choice: &AnthropicToolChoice, extra: &mut Map<String, Value>) {
    let value = match choice {
        AnthropicToolChoice::Auto { .. } => json!("auto"),
        AnthropicToolChoice::Any { .. } => json!("required"),
        AnthropicToolChoice::None { .. } => json!("none"),
        AnthropicToolChoice::Tool { name, .. } => {
            json!({ "type": "function", "function": { "name": name } })
        }
    };
    extra.insert("tool_choice".to_string(), value);

    let disable_parallel = match choice {
        AnthropicToolChoice::Auto {
            disable_parallel_tool_use,
            ..
        }
        | AnthropicToolChoice::Any {
            disable_parallel_tool_use,
            ..
        }
        | AnthropicToolChoice::Tool {
            disable_parallel_tool_use,
            ..
        } => *disable_parallel_tool_use,
        AnthropicToolChoice::None { .. } => None,
    };
    if disable_parallel == Some(true) {
        extra.insert("parallel_tool_calls".to_string(), json!(false));
    }
}

/// `thinking` config → the `extra["thinking"]` shape
/// [`crate::reasoning_compat::normalize::read_client_spec`] parses.
/// `disabled` emits nothing (nothing was requested).
fn thinking_param(config: &ThinkingConfig) -> Option<Value> {
    match config {
        ThinkingConfig::Adaptive { .. } => Some(json!({ "type": "adaptive" })),
        ThinkingConfig::Enabled { budget_tokens, .. } => Some(json!({
            "type": "enabled",
            "budget_tokens": budget_tokens,
        })),
        ThinkingConfig::Disabled { .. } => None,
    }
}

#[cfg(test)]
mod tests {
    use super::translate;
    use crate::anthropic::models::AnthropicRequest;
    use serde_json::{json, Value};

    fn parse(body: Value) -> AnthropicRequest {
        serde_json::from_value(body).expect("test body parses as AnthropicRequest")
    }

    fn ok(body: Value) -> crate::models::openai::OpenAIRequest {
        translate(&parse(body)).expect("translation succeeds")
    }

    fn err(body: Value) -> String {
        translate(&parse(body)).expect_err("translation rejects")
    }

    fn minimal(body: Value) -> Value {
        let mut base = json!({ "model": "claude-group", "max_tokens": 1024, "messages": [] });
        let map = base.as_object_mut().unwrap();
        for (key, value) in body.as_object().unwrap() {
            map.insert(key.clone(), value.clone());
        }
        base
    }

    // ------------------------------------------------------------------
    // System
    // ------------------------------------------------------------------

    #[test]
    fn system_string_becomes_strictly_first_system_message() {
        let out = ok(minimal(json!({
            "system": "Be terse.",
            "messages": [{ "role": "user", "content": "Hello" }]
        })));
        assert_eq!(out.messages[0].role, "system");
        assert_eq!(out.messages[0].content, Value::String("Be terse.".into()));
        assert_eq!(out.messages[1].role, "user");
    }

    #[test]
    fn system_blocks_join_into_single_leading_system_message() {
        let out = ok(minimal(json!({
            "system": [
                { "type": "text", "text": "Part one." },
                { "type": "text", "text": "Part two." }
            ],
            "messages": [{ "role": "user", "content": "Hi" }]
        })));
        assert_eq!(out.messages[0].role, "system");
        assert_eq!(out.messages[0].content, Value::String("Part one.\nPart two.".into()));
        assert_eq!(out.messages.len(), 2);
    }

    #[test]
    fn no_system_yields_no_system_message() {
        let out = ok(minimal(json!({})));
        assert!(out.messages.is_empty());
    }

    // ------------------------------------------------------------------
    // Plain content
    // ------------------------------------------------------------------

    #[test]
    fn string_content_passthrough() {
        let out = ok(minimal(json!({
            "messages": [{ "role": "user", "content": "Hello" }]
        })));
        assert_eq!(out.messages[0].role, "user");
        assert_eq!(out.messages[0].content, Value::String("Hello".into()));
    }

    #[test]
    fn text_blocks_become_text_parts() {
        let out = ok(minimal(json!({
            "messages": [{
                "role": "user",
                "content": [
                    { "type": "text", "text": "a" },
                    { "type": "text", "text": "b" }
                ]
            }]
        })));
        assert_eq!(
            out.messages[0].content,
            json!([{ "type": "text", "text": "a" }, { "type": "text", "text": "b" }])
        );
    }

    // ------------------------------------------------------------------
    // Images
    // ------------------------------------------------------------------

    #[test]
    fn image_base64_becomes_data_url_part() {
        let out = ok(minimal(json!({
            "messages": [{
                "role": "user",
                "content": [
                    { "type": "text", "text": "what is this?" },
                    { "type": "image", "source": { "type": "base64", "media_type": "image/png", "data": "aGk=" } }
                ]
            }]
        })));
        assert_eq!(
            out.messages[0].content,
            json!([
                { "type": "text", "text": "what is this?" },
                { "type": "image_url", "image_url": { "url": "data:image/png;base64,aGk=" } }
            ])
        );
    }

    #[test]
    fn image_url_passthrough() {
        let out = ok(minimal(json!({
            "messages": [{
                "role": "user",
                "content": [
                    { "type": "image", "source": { "type": "url", "url": "https://x/i.png" } }
                ]
            }]
        })));
        assert_eq!(
            out.messages[0].content,
            json!([{ "type": "image_url", "image_url": { "url": "https://x/i.png" } }])
        );
    }

    // ------------------------------------------------------------------
    // Tool history
    // ------------------------------------------------------------------

    #[test]
    fn assistant_tool_use_becomes_tool_calls_entry() {
        let out = ok(minimal(json!({
            "messages": [{
                "role": "assistant",
                "content": [
                    { "type": "text", "text": "Checking." },
                    { "type": "tool_use", "id": "toolu_1", "name": "lookup", "input": { "q": "x" } }
                ]
            }]
        })));
        let msg = &out.messages[0];
        assert_eq!(msg.role, "assistant");
        assert_eq!(msg.content, json!([{ "type": "text", "text": "Checking." }]));
        let calls = msg.extra.get("tool_calls").and_then(Value::as_array).unwrap();
        assert_eq!(calls.len(), 1);
        assert_eq!(calls[0]["id"], "toolu_1");
        assert_eq!(calls[0]["type"], "function");
        assert_eq!(calls[0]["function"]["name"], "lookup");
        assert_eq!(calls[0]["function"]["arguments"], json!("{\"q\":\"x\"}"));
    }

    #[test]
    fn assistant_tool_use_only_gets_null_content() {
        let out = ok(minimal(json!({
            "messages": [{
                "role": "assistant",
                "content": [
                    { "type": "tool_use", "id": "toolu_2", "name": "run", "input": {} }
                ]
            }]
        })));
        assert_eq!(out.messages[0].content, Value::Null);
        assert!(out.messages[0].extra.contains_key("tool_calls"));
    }

    #[test]
    fn user_tool_result_becomes_tool_message_with_tool_call_id() {
        let out = ok(minimal(json!({
            "messages": [{
                "role": "user",
                "content": [
                    { "type": "tool_result", "tool_use_id": "toolu_1", "content": "42" }
                ]
            }]
        })));
        assert_eq!(out.messages.len(), 1);
        let msg = &out.messages[0];
        assert_eq!(msg.role, "tool");
        assert_eq!(msg.content, Value::String("42".into()));
        assert_eq!(
            msg.extra.get("tool_call_id").and_then(Value::as_str),
            Some("toolu_1")
        );
    }

    #[test]
    fn multi_turn_tool_history_round_trips() {
        let out = ok(minimal(json!({
            "messages": [
                { "role": "user", "content": "Weather in Paris?" },
                {
                    "role": "assistant",
                    "content": [
                        { "type": "thinking", "thinking": "need the tool", "signature": "sig_1" },
                        { "type": "tool_use", "id": "toolu_9", "name": "get_weather", "input": { "city": "Paris" } }
                    ]
                },
                {
                    "role": "user",
                    "content": [
                        { "type": "tool_result", "tool_use_id": "toolu_9", "content": "18C sunny" }
                    ]
                },
                { "role": "user", "content": "Thanks!" }
            ]
        })));
        let roles: Vec<&str> = out.messages.iter().map(|m| m.role.as_str()).collect();
        assert_eq!(roles, vec!["user", "assistant", "tool", "user"]);
        assert_eq!(
            out.messages[1].extra.get("tool_calls").unwrap()[0]["id"],
            "toolu_9"
        );
        assert_eq!(
            out.messages[2].extra.get("tool_call_id").and_then(Value::as_str),
            Some("toolu_9")
        );
        assert_eq!(out.messages[3].content, Value::String("Thanks!".into()));
    }

    #[test]
    fn consecutive_tool_results_become_one_tool_message_each() {
        let out = ok(minimal(json!({
            "messages": [{
                "role": "user",
                "content": [
                    { "type": "tool_result", "tool_use_id": "t1", "content": "a" },
                    { "type": "tool_result", "tool_use_id": "t2", "content": "b" }
                ]
            }]
        })));
        assert_eq!(out.messages.len(), 2);
        assert_eq!(out.messages[0].role, "tool");
        assert_eq!(out.messages[1].role, "tool");
        assert_eq!(
            out.messages[0].extra.get("tool_call_id").and_then(Value::as_str),
            Some("t1")
        );
        assert_eq!(
            out.messages[1].extra.get("tool_call_id").and_then(Value::as_str),
            Some("t2")
        );
    }

    #[test]
    fn tool_result_block_content_with_image_becomes_part_array() {
        let out = ok(minimal(json!({
            "messages": [{
                "role": "user",
                "content": [
                    {
                        "type": "tool_result",
                        "tool_use_id": "t1",
                        "content": [
                            { "type": "text", "text": "chart attached" },
                            { "type": "image", "source": { "type": "base64", "media_type": "image/jpeg", "data": "AA==" } }
                        ]
                    }
                ]
            }]
        })));
        let msg = &out.messages[0];
        assert_eq!(msg.role, "tool");
        assert_eq!(
            msg.content,
            json!([
                { "type": "text", "text": "chart attached" },
                { "type": "image_url", "image_url": { "url": "data:image/jpeg;base64,AA==" } }
            ])
        );
    }

    // ------------------------------------------------------------------
    // Reasoning carriers
    // ------------------------------------------------------------------

    #[test]
    fn replayed_thinking_block_preserved_verbatim() {
        let out = ok(minimal(json!({
            "messages": [{
                "role": "assistant",
                "content": [
                    { "type": "thinking", "thinking": "deep", "signature": "sig_abc" },
                    { "type": "text", "text": "answer" }
                ]
            }]
        })));
        assert_eq!(
            out.messages[0].content,
            json!([
                { "type": "thinking", "thinking": "deep", "signature": "sig_abc" },
                { "type": "text", "text": "answer" }
            ])
        );
    }

    #[test]
    fn replayed_redacted_thinking_block_preserved_verbatim() {
        let out = ok(minimal(json!({
            "messages": [{
                "role": "assistant",
                "content": [
                    { "type": "redacted_thinking", "data": "opaque" },
                    { "type": "text", "text": "answer" }
                ]
            }]
        })));
        assert_eq!(
            out.messages[0].content,
            json!([
                { "type": "redacted_thinking", "data": "opaque" },
                { "type": "text", "text": "answer" }
            ])
        );
    }

    #[test]
    fn thinking_only_assistant_message_survives_with_null_free_content() {
        let out = ok(minimal(json!({
            "messages": [{
                "role": "assistant",
                "content": [{ "type": "thinking", "thinking": "hmm", "signature": "s" }]
            }]
        })));
        assert_eq!(out.messages.len(), 1);
        assert_eq!(
            out.messages[0].content,
            json!([{ "type": "thinking", "thinking": "hmm", "signature": "s" }])
        );
    }

    #[test]
    fn thinking_matrix_adaptive_enabled_disabled_absent() {
        // adaptive
        let out = ok(minimal(json!({ "thinking": { "type": "adaptive" } })));
        assert_eq!(out.extra.get("thinking"), Some(&json!({ "type": "adaptive" })));

        // enabled with budget
        let out = ok(minimal(json!({ "thinking": { "type": "enabled", "budget_tokens": 2048 } })));
        assert_eq!(
            out.extra.get("thinking"),
            Some(&json!({ "type": "enabled", "budget_tokens": 2048 }))
        );

        // disabled → nothing emitted (no-injection rule)
        let out = ok(minimal(json!({ "thinking": { "type": "disabled" } })));
        assert!(out.extra.get("thinking").is_none());

        // absent
        let out = ok(minimal(json!({})));
        assert!(out.extra.get("thinking").is_none());
    }

    #[test]
    fn adaptive_effort_output_config_passthrough_for_reasoning_compat() {
        let out = ok(minimal(json!({
            "thinking": { "type": "adaptive" },
            "output_config": { "effort": "high" }
        })));
        assert_eq!(
            out.extra.get("output_config"),
            Some(&json!({ "effort": "high" }))
        );
    }

    // ------------------------------------------------------------------
    // Tools / tool_choice
    // ------------------------------------------------------------------

    #[test]
    fn flat_tools_flatten_to_nested_function_tools() {
        let out = ok(minimal(json!({
            "tools": [
                {
                    "name": "get_weather",
                    "description": "Get weather",
                    "input_schema": { "type": "object", "properties": { "city": { "type": "string" } } }
                },
                {
                    "type": "custom",
                    "name": "noop",
                    "input_schema": { "type": "object" }
                }
            ]
        })));
        let tools = out.extra.get("tools").and_then(Value::as_array).unwrap();
        assert_eq!(tools.len(), 2);
        assert_eq!(tools[0]["type"], "function");
        assert_eq!(tools[0]["function"]["name"], "get_weather");
        assert_eq!(tools[0]["function"]["description"], "Get weather");
        assert_eq!(
            tools[0]["function"]["parameters"],
            json!({ "type": "object", "properties": { "city": { "type": "string" } } })
        );
        assert_eq!(tools[1]["function"]["name"], "noop");
    }

    #[test]
    fn tool_defaults_when_optional_fields_missing() {
        let out = ok(minimal(json!({ "tools": [{ "name": "bare" }] })));
        let tools = out.extra.get("tools").and_then(Value::as_array).unwrap();
        assert_eq!(tools[0]["function"]["description"], "");
        assert_eq!(tools[0]["function"]["parameters"], json!({ "type": "object" }));
    }

    #[test]
    fn tool_choice_matrix() {
        for (anthropic, expected) in [
            (json!({ "type": "auto" }), json!("auto")),
            (json!({ "type": "any" }), json!("required")),
            (json!({ "type": "none" }), json!("none")),
            (
                json!({ "type": "tool", "name": "lookup" }),
                json!({ "type": "function", "function": { "name": "lookup" } }),
            ),
        ] {
            let out = ok(minimal(json!({ "tool_choice": anthropic })));
            assert_eq!(out.extra.get("tool_choice"), Some(&expected));
        }
    }

    #[test]
    fn disable_parallel_tool_use_maps_to_parallel_tool_calls_false() {
        let out = ok(minimal(json!({
            "tool_choice": { "type": "any", "disable_parallel_tool_use": true }
        })));
        assert_eq!(out.extra.get("parallel_tool_calls"), Some(&json!(false)));

        let out = ok(minimal(json!({
            "tool_choice": { "type": "auto" }
        })));
        assert!(out.extra.get("parallel_tool_calls").is_none());
    }

    // ------------------------------------------------------------------
    // Sampling / stops / metadata / stream
    // ------------------------------------------------------------------

    #[test]
    fn stop_sequences_map_to_stop() {
        let out = ok(minimal(json!({ "stop_sequences": ["END", "STOP"] })));
        assert_eq!(out.extra.get("stop"), Some(&json!(["END", "STOP"])));

        let out = ok(minimal(json!({})));
        assert!(out.extra.get("stop").is_none());
    }

    #[test]
    fn sampling_and_max_tokens_passthrough() {
        let out = ok(minimal(json!({
            "max_tokens": 4096,
            "temperature": 0.7,
            "top_p": 0.9,
            "stream": true
        })));
        assert_eq!(out.max_tokens, Some(4096));
        assert_eq!(out.temperature, Some(0.7));
        assert_eq!(
            out.extra.get("top_p").and_then(Value::as_f64),
            Some(f64::from(0.9_f32))
        );
        assert!(out.stream);
    }

    #[test]
    fn metadata_user_id_maps_to_user_field() {
        let out = ok(minimal(json!({ "metadata": { "user_id": "user_123" } })));
        assert_eq!(out.extra.get("user"), Some(&json!("user_123")));
    }

    #[test]
    fn model_group_name_passthrough() {
        let out = ok(minimal(json!({})));
        assert_eq!(out.model, "claude-group");
    }

    // ------------------------------------------------------------------
    // Rejections
    // ------------------------------------------------------------------

    #[test]
    fn hosted_tool_types_rejected_with_field_naming_message() {
        for (tool_type, stem) in [
            (json!("web_search_20250305"), "web_search"),
            (json!("computer_20250124"), "computer"),
            (json!("code_execution_20250522"), "code_execution"),
            (json!("text_editor_20250124"), "text_editor"),
            (json!("bash_20250124"), "bash"),
            (json!("mcp_connector_20250811"), "mcp_connector"),
            (json!("web_search"), "web_search"),
        ] {
            let message = err(minimal(json!({
                "tools": [{ "type": tool_type, "name": "x" }]
            })));
            assert_eq!(message, format!("tool type '{stem}' is not supported"));
        }
    }

    #[test]
    fn unknown_tool_type_rejected() {
        let message = err(minimal(json!({
            "tools": [{ "type": "future_tool_20990101", "name": "x" }]
        })));
        assert_eq!(message, "tool type 'future_tool_20990101' is not supported");
    }

    #[test]
    fn context_management_rejected() {
        let message = err(minimal(json!({
            "context_management": { "edits": [{ "type": "clear_tool_uses_20250919" }] }
        })));
        assert_eq!(message, "context_management is not supported");
    }

    #[test]
    fn document_content_block_rejected() {
        let message = err(minimal(json!({
            "messages": [{
                "role": "user",
                "content": [
                    { "type": "document", "source": { "type": "url", "url": "https://x/d.pdf" } }
                ]
            }]
        })));
        assert_eq!(message, "content block type 'document' is not supported");
    }

    #[test]
    fn rejections_fire_before_any_translation() {
        // A request that would also produce hosted-tool and document issues
        // reports the first rejection deterministically (context_management
        // is checked first).
        let message = err(minimal(json!({
            "context_management": {},
            "tools": [{ "type": "web_search_20250305" }]
        })));
        assert_eq!(message, "context_management is not supported");
    }

    // ------------------------------------------------------------------
    // cache_control
    // ------------------------------------------------------------------

    #[test]
    fn cache_control_markers_never_reach_the_translated_request() {
        let out = ok(minimal(json!({
            "system": [
                { "type": "text", "text": "sys", "cache_control": { "type": "ephemeral" } }
            ],
            "messages": [{
                "role": "user",
                "content": [
                    { "type": "text", "text": "hi", "cache_control": { "type": "ephemeral" } }
                ]
            }],
            "tools": [
                {
                    "name": "t",
                    "input_schema": { "type": "object" },
                    "cache_control": { "type": "ephemeral" }
                }
            ]
        })));
        let serialized = serde_json::to_string(&out).unwrap();
        assert!(!serialized.contains("cache_control"));
    }

    // ------------------------------------------------------------------
    // Edge shapes
    // ------------------------------------------------------------------

    #[test]
    fn empty_block_list_emits_no_message() {
        let out = ok(minimal(json!({
            "messages": [
                { "role": "user", "content": [] },
                { "role": "user", "content": "real" }
            ]
        })));
        assert_eq!(out.messages.len(), 1);
        assert_eq!(out.messages[0].content, Value::String("real".into()));
    }

    #[test]
    fn user_message_with_tool_result_and_text_splits_into_tool_then_user() {
        let out = ok(minimal(json!({
            "messages": [{
                "role": "user",
                "content": [
                    { "type": "tool_result", "tool_use_id": "t1", "content": "ok" },
                    { "type": "text", "text": "and also this" }
                ]
            }]
        })));
        let roles: Vec<&str> = out.messages.iter().map(|m| m.role.as_str()).collect();
        assert_eq!(roles, vec!["tool", "user"]);
        assert_eq!(out.messages[1].content, json!([{ "type": "text", "text": "and also this" }]));
    }
}
