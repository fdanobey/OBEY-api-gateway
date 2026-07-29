//! Chat Completions → Responses API request translator.
//!
//! Converts an [`OpenAIRequest`] into the Codex Responses API request envelope
//! described in the `codex-backend-translation` spec
//! (Requirements 2.x and 3.x; see `design.md` §2). The translator is
//! borrow-only and stateless — construct a fresh instance per request and
//! immediately drop it.
//!
//! # Security
//!
//! The translator never logs body bytes. The `tracing::instrument` annotation
//! skips `req`; the only emitted `info!` carries a sanitized summary
//! (endpoint path, model, message count, tool count, effort).

use std::collections::HashSet;

use serde_json::{json, Map, Value};
use tracing::{info, warn};

use crate::codex::errors::CodexError;
use crate::models::openai::{Message, OpenAIRequest};

/// Stateless borrow-only translator. Owns no buffers; each call to
/// [`Self::translate`] returns a freshly built `serde_json::Value` envelope.
///
/// Field shape matches `design.md` §Data Models exactly.
#[derive(Debug, Clone, Copy)]
pub struct ChatToResponsesTranslator<'a> {
    /// Resolved outgoing model name (pass-through of client model, or
    /// `codex_model_override` when configured).
    pub resolved_model: &'a str,
    /// Codex system prompt to attach as the Responses API `instructions`
    /// field (output of [`crate::codex::InstructionsStore::get`]).
    pub instructions: &'a str,
    /// Effort_Mapper output ∈ `{"low","medium","high","xhigh"}`. Only
    /// consulted when [`Self::emit_reasoning`] is true.
    pub mapped_effort: &'a str,
    /// Whether to emit the `reasoning` field on the envelope. Driven by
    /// [`crate::codex::is_reasoning`] — see Req 2.9.
    pub emit_reasoning: bool,
}

impl<'a> ChatToResponsesTranslator<'a> {
    /// Translate a Chat Completions request into a Responses API request
    /// envelope.
    ///
    /// # Errors
    ///
    /// * [`CodexError::UnsupportedFeature`] with `feature = "audio_output"`
    ///   when the request body contains a top-level `audio` field or has
    ///   `modalities` including `"audio"` (Req 14.3).
    /// * [`CodexError::UnsupportedFeature`] with `feature = "input_audio"`
    ///   when any user message contains an `input_audio` content part
    ///   (Req 14.3).
    #[tracing::instrument(skip(req, self))]
    pub fn translate(&self, req: &OpenAIRequest) -> Result<Value, CodexError> {
        // -- Req 14.3: reject audio output up front. -----------------------
        if req.extra.contains_key("audio") {
            return Err(CodexError::UnsupportedFeature {
                feature: "audio_output",
            });
        }
        if let Some(modalities) = req.extra.get("modalities").and_then(|v| v.as_array()) {
            if modalities.iter().any(|m| m.as_str() == Some("audio")) {
                return Err(CodexError::UnsupportedFeature {
                    feature: "audio_output",
                });
            }
        }

        let input = convert_messages(&req.messages)?;

        let tool_count = req
            .extra
            .get("tools")
            .and_then(|v| v.as_array())
            .map(|a| a.len())
            .unwrap_or(0);

        info!(
            endpoint = "/v1/chat/completions",
            model = self.resolved_model,
            message_count = req.messages.len(),
            tool_count = tool_count,
            effort = self.mapped_effort,
            "OpenAI Codex Prompt injected."
        );

        // -- Top-level envelope (Req 2.1, 2.2, 2.3, 2.6, 2.7, 2.8, 2.12) ---
        let mut env: Map<String, Value> = Map::new();
        env.insert("model".to_string(), json!(self.resolved_model));
        env.insert("instructions".to_string(), json!(self.instructions));
        env.insert("input".to_string(), Value::Array(input));
        env.insert("store".to_string(), json!(false));
        env.insert("stream".to_string(), json!(true));
        env.insert(
            "include".to_string(),
            json!(["reasoning.encrypted_content"]),
        );

        // tools — transform from Chat Completions format to Responses API format.
        // Chat Completions: {"type":"function","function":{"name":"x","description":"...","parameters":{...}}}
        // Responses API:    {"type":"function","name":"x","description":"...","parameters":{...}}
        if let Some(tools) = req.extra.get("tools").and_then(|v| v.as_array()) {
            let transformed: Vec<Value> = tools
                .iter()
                .map(|tool| {
                    if let Some(func) = tool.get("function") {
                        // Flatten: pull name, description, parameters up from function object
                        let mut t = serde_json::Map::new();
                        t.insert("type".to_string(), json!("function"));
                        if let Some(name) = func.get("name") {
                            t.insert("name".to_string(), name.clone());
                        }
                        if let Some(desc) = func.get("description") {
                            t.insert("description".to_string(), desc.clone());
                        }
                        if let Some(params) = func.get("parameters") {
                            t.insert("parameters".to_string(), params.clone());
                        }
                        if let Some(strict) = func.get("strict") {
                            t.insert("strict".to_string(), strict.clone());
                        }
                        Value::Object(t)
                    } else {
                        // Non-function tool types (web_search, etc.) — forward as-is
                        tool.clone()
                    }
                })
                .collect();
            env.insert("tools".to_string(), Value::Array(transformed));
        }
        // tool_choice — forward unchanged (Req 2.7)
        if let Some(tc) = req.extra.get("tool_choice") {
            env.insert("tool_choice".to_string(), tc.clone());
        }
        // temperature — the Codex backend does NOT accept temperature on the
        // Responses API (returns "Unsupported parameter: temperature").
        // Strip it unconditionally for Codex dispatch.
        // top_p — same treatment; strip to avoid similar rejection.
        // reasoning — emit only when caller flags the model as reasoning (Req 2.9)
        if self.emit_reasoning {
            env.insert(
                "reasoning".to_string(),
                json!({
                    "effort": self.mapped_effort,
                    "summary": "auto",
                }),
            );
        }

        // text {verbosity, format} — merged into a single object (Req 2.10, 2.13–2.17)
        let mut text_obj: Map<String, Value> = Map::new();
        if let Some(v) = req.extra.get("verbosity").and_then(|v| v.as_str()) {
            if matches!(v, "low" | "medium" | "high") {
                text_obj.insert("verbosity".to_string(), json!(v));
            }
        }
        if let Some(rf) = req.extra.get("response_format") {
            if let Some(format) = translate_response_format(rf) {
                text_obj.insert("format".to_string(), format);
            }
        }
        if !text_obj.is_empty() {
            env.insert("text".to_string(), Value::Object(text_obj));
        }

        // Note: max_output_tokens, max_completion_tokens, max_tokens, logprobs,
        // top_logprobs, n, seed, presence_penalty, frequency_penalty,
        // logit_bias, user, stop are stripped by construction — we never read
        // them from `req.extra` and never reference `req.max_tokens` (Req 2.4, 2.5).

        Ok(Value::Object(env))
    }
}

/// Translate a Chat Completions `response_format` value into the matching
/// Responses API `text.format` value (Req 2.13–2.17).
///
/// Returns `None` when the format should be omitted (`{type:"text"}` or
/// when the type is missing).
fn translate_response_format(rf: &Value) -> Option<Value> {
    let t = rf.get("type").and_then(|v| v.as_str());
    match t {
        // Req 2.15
        Some("text") | None => None,
        // Req 2.13
        Some("json_object") => Some(json!({"type": "json_object"})),
        // Req 2.14
        Some("json_schema") => {
            let js = rf.get("json_schema").cloned().unwrap_or(Value::Null);
            let name = js.get("name").cloned().unwrap_or(Value::Null);
            let schema = js.get("schema").cloned().unwrap_or(Value::Null);
            let strict = js.get("strict").cloned().unwrap_or(json!(false));
            Some(json!({
                "type": "json_schema",
                "name": name,
                "schema": schema,
                "strict": strict,
            }))
        }
        // Req 2.16 — forward unknown types unchanged + warn
        Some(other) => {
            warn!(
                unrecognized_type = other,
                "response_format.type not recognized; forwarding to Codex backend unchanged"
            );
            Some(rf.clone())
        }
    }
}

/// Walk the request messages in order and emit the matching Responses API
/// `input` items per Req 3.1–3.10. Tracks emitted `function_call` ids in a
/// `HashSet` so orphan `tool` messages can fall through to a user message
/// (Req 3.7).
fn convert_messages(messages: &[Message]) -> Result<Vec<Value>, CodexError> {
    let mut out: Vec<Value> = Vec::with_capacity(messages.len());
    let mut call_ids: HashSet<String> = HashSet::new();

    for msg in messages {
        match msg.role.as_str() {
            // Req 3.1 — System/developer messages are NOT emitted as input items.
            // The Codex backend rejects them with "System messages are not allowed".
            // System-level context goes exclusively in the `instructions` field.
            // Any client-supplied system messages are silently skipped here;
            // the InstructionsStore already provides the Codex system prompt.
            "system" | "developer" => {
                // Skip — handled via the `instructions` envelope field.
            }

            // Req 3.2 / 3.3
            "user" => match &msg.content {
                Value::Array(parts) => {
                    let content = convert_user_array_content(parts)?;
                    out.push(json!({
                        "type": "message",
                        "role": "user",
                        "content": content,
                    }));
                }
                _ => {
                    let text = msg.content_as_text();
                    out.push(json!({
                        "type": "message",
                        "role": "user",
                        "content": [{"type": "input_text", "text": text}],
                    }));
                }
            },

            // Req 3.4 / 3.5
            "assistant" => {
                let tool_calls = msg.extra.get("tool_calls").and_then(|v| v.as_array());
                match tool_calls {
                    Some(calls) if !calls.is_empty() => {
                        for c in calls {
                            let call_id = c
                                .get("id")
                                .and_then(|v| v.as_str())
                                .unwrap_or("")
                                .to_string();
                            let name = c
                                .get("function")
                                .and_then(|f| f.get("name"))
                                .and_then(|v| v.as_str())
                                .unwrap_or("")
                                .to_string();
                            let args = match c.get("function").and_then(|f| f.get("arguments")) {
                                Some(Value::String(s)) => s.clone(),
                                Some(other) => other.to_string(),
                                None => String::new(),
                            };
                            if !call_id.is_empty() {
                                call_ids.insert(call_id.clone());
                            }
                            out.push(json!({
                                "type": "function_call",
                                "call_id": call_id,
                                "name": name,
                                "arguments": args,
                            }));
                        }
                    }
                    _ => {
                        let text = msg.content_as_text();
                        // Req 3.4 specifies "non-empty string content"; skip
                        // entirely when empty so we don't emit an empty
                        // assistant turn.
                        if !text.is_empty() {
                            out.push(json!({
                                "type": "message",
                                "role": "assistant",
                                "content": [{"type": "output_text", "text": text}],
                            }));
                        }
                    }
                }
            }

            // Req 3.6 / 3.7
            "tool" => {
                let id = msg
                    .extra
                    .get("tool_call_id")
                    .and_then(|v| v.as_str())
                    .unwrap_or("")
                    .to_string();
                let text = msg.content_as_text();
                if !id.is_empty() && call_ids.contains(&id) {
                    out.push(json!({
                        "type": "function_call_output",
                        "call_id": id,
                        "output": text,
                    }));
                } else {
                    // Orphan — Req 3.7
                    out.push(json!({
                        "type": "message",
                        "role": "user",
                        "content": [{"type": "input_text", "text": text}],
                    }));
                }
            }

            // Forward any other role as a generic message (forward-compat).
            other => {
                let text = msg.content_as_text();
                out.push(json!({
                    "type": "message",
                    "role": other,
                    "content": [{"type": "input_text", "text": text}],
                }));
            }
        }
    }

    Ok(out)
}

/// Translate the `content` array of a `user` message (Req 3.3).
fn convert_user_array_content(parts: &[Value]) -> Result<Vec<Value>, CodexError> {
    let mut out: Vec<Value> = Vec::with_capacity(parts.len());
    for part in parts {
        let t = part.get("type").and_then(|v| v.as_str());
        match t {
            Some("text") => {
                let text = part
                    .get("text")
                    .cloned()
                    .unwrap_or_else(|| Value::String(String::new()));
                out.push(json!({"type": "input_text", "text": text}));
            }
            Some("image_url") => {
                let image_url = part
                    .get("image_url")
                    .and_then(|value| match value {
                        Value::String(url) => Some(url.clone()),
                        Value::Object(image) => {
                            image.get("url").and_then(Value::as_str).map(str::to_owned)
                        }
                        _ => None,
                    })
                    .ok_or_else(|| {
                        CodexError::Translation(
                            "image_url content part missing string image_url.url".to_string(),
                        )
                    })?;

                let detail = part
                    .get("image_url")
                    .and_then(Value::as_object)
                    .and_then(|image| image.get("detail"))
                    .or_else(|| part.get("detail"))
                    .cloned();

                let mut input_image = Map::new();
                input_image.insert("type".to_string(), json!("input_image"));
                input_image.insert("image_url".to_string(), json!(image_url));
                if let Some(detail) = detail {
                    input_image.insert("detail".to_string(), detail);
                }
                out.push(Value::Object(input_image));
            }
            // Req 14.3
            Some("input_audio") => {
                return Err(CodexError::UnsupportedFeature {
                    feature: "input_audio",
                });
            }
            // Forward-compat for unknown content parts.
            _ => {
                out.push(part.clone());
            }
        }
    }
    Ok(out)
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::{json, Map};

    fn req_with(messages: Vec<Message>) -> OpenAIRequest {
        OpenAIRequest {
            model: "gpt-5.1".into(),
            messages,
            stream: false,
            temperature: None,
            max_tokens: None,
            extra: Map::new(),
        }
    }

    fn msg(role: &str, content: Value) -> Message {
        Message {
            role: role.into(),
            content,
            extra: Map::new(),
        }
    }

    fn translator<'a>() -> ChatToResponsesTranslator<'a> {
        ChatToResponsesTranslator {
            resolved_model: "gpt-5.1",
            instructions: "you are codex",
            mapped_effort: "medium",
            emit_reasoning: true,
        }
    }

    #[test]
    fn top_level_envelope_shape_is_constant() {
        let req = req_with(vec![msg("user", json!("hi"))]);
        let out = translator().translate(&req).unwrap();
        assert_eq!(out["model"], json!("gpt-5.1"));
        assert_eq!(out["instructions"], json!("you are codex"));
        assert_eq!(out["store"], json!(false));
        assert_eq!(out["stream"], json!(true));
        assert_eq!(out["include"], json!(["reasoning.encrypted_content"]));
    }

    #[test]
    fn system_user_assistant_message_shapes() {
        // Post-integration behavior: system/developer messages are SKIPPED
        // (not emitted as input items) because the Codex backend rejects them.
        // Only user and assistant messages survive into `input`.
        let req = req_with(vec![
            msg("system", json!("rules")),
            msg("user", json!("hi")),
            msg("assistant", json!("hello")),
        ]);
        let out = translator().translate(&req).unwrap();
        let input = out["input"].as_array().unwrap();
        assert_eq!(
            input.len(),
            2,
            "system message should be skipped: {input:#?}"
        );
        assert_eq!(
            input[0],
            json!({"type":"message","role":"user",
                   "content":[{"type":"input_text","text":"hi"}]})
        );
        assert_eq!(
            input[1],
            json!({"type":"message","role":"assistant",
                   "content":[{"type":"output_text","text":"hello"}]})
        );
        // No item should carry role=="system".
        assert!(
            !input
                .iter()
                .any(|i| i.get("role").and_then(|r| r.as_str()) == Some("system")),
            "no system message should be in the output input array"
        );
    }

    #[test]
    fn assistant_tool_calls_emit_one_function_call_each() {
        let mut m = msg("assistant", Value::Null);
        m.extra.insert(
            "tool_calls".into(),
            json!([
                {"id":"c1","type":"function","function":{"name":"a","arguments":"{\"x\":1}"}},
                {"id":"c2","type":"function","function":{"name":"b","arguments":{"y":2}}},
            ]),
        );
        let req = req_with(vec![m]);
        let out = translator().translate(&req).unwrap();
        let input = out["input"].as_array().unwrap();
        assert_eq!(input.len(), 2);
        assert_eq!(input[0]["type"], json!("function_call"));
        assert_eq!(input[0]["call_id"], json!("c1"));
        assert_eq!(input[0]["name"], json!("a"));
        assert_eq!(input[0]["arguments"], json!("{\"x\":1}"));
        assert_eq!(input[1]["call_id"], json!("c2"));
        // Object args get JSON-stringified.
        assert_eq!(input[1]["arguments"].as_str().unwrap(), "{\"y\":2}");
    }

    #[test]
    fn tool_role_with_matching_call_id_emits_function_call_output() {
        let mut a = msg("assistant", Value::Null);
        a.extra.insert(
            "tool_calls".into(),
            json!([{"id":"c1","type":"function","function":{"name":"a","arguments":"{}"}}]),
        );
        let mut t = msg("tool", json!("42"));
        t.extra.insert("tool_call_id".into(), json!("c1"));
        let req = req_with(vec![a, t]);
        let out = translator().translate(&req).unwrap();
        let input = out["input"].as_array().unwrap();
        assert_eq!(input.len(), 2);
        assert_eq!(
            input[1],
            json!({"type":"function_call_output","call_id":"c1","output":"42"})
        );
    }

    #[test]
    fn tool_role_orphan_call_id_falls_back_to_user_message() {
        let mut t = msg("tool", json!("orphan-output"));
        t.extra
            .insert("tool_call_id".into(), json!("does-not-exist"));
        let req = req_with(vec![t]);
        let out = translator().translate(&req).unwrap();
        let input = out["input"].as_array().unwrap();
        assert_eq!(
            input[0],
            json!({"type":"message","role":"user",
                   "content":[{"type":"input_text","text":"orphan-output"}]})
        );
    }

    #[test]
    fn user_array_content_image_url_maps_to_input_image() {
        let m = msg(
            "user",
            json!([
                {"type":"text","text":"look"},
                {"type":"image_url","image_url":{"url":"https://x.example/p.png","detail":"high"}},
            ]),
        );
        let req = req_with(vec![m]);
        let out = translator().translate(&req).unwrap();
        let content = &out["input"][0]["content"];
        assert_eq!(content[0], json!({"type":"input_text","text":"look"}));
        assert_eq!(
            content[1],
            json!({"type":"input_image","image_url":"https://x.example/p.png","detail":"high"})
        );
    }

    #[test]
    fn user_array_content_string_image_url_is_preserved() {
        let m = msg(
            "user",
            json!([{"type":"image_url","image_url":"data:image/png;base64,AAEC"}]),
        );
        let req = req_with(vec![m]);
        let out = translator().translate(&req).unwrap();
        assert_eq!(
            out["input"][0]["content"][0],
            json!({"type":"input_image","image_url":"data:image/png;base64,AAEC"})
        );
    }

    #[test]
    fn user_array_content_image_url_without_url_returns_translation_error() {
        let m = msg(
            "user",
            json!([{"type":"image_url","image_url":{"detail":"low"}}]),
        );
        let req = req_with(vec![m]);
        match translator().translate(&req) {
            Err(CodexError::Translation(message)) => {
                assert!(message.contains("image_url.url"));
            }
            other => panic!("expected image URL translation error, got {other:?}"),
        }
    }

    #[test]
    fn user_array_content_input_audio_returns_unsupported_feature() {
        let m = msg(
            "user",
            json!([{"type":"input_audio","input_audio":{"data":"AAAA","format":"wav"}}]),
        );
        let req = req_with(vec![m]);
        match translator().translate(&req) {
            Err(CodexError::UnsupportedFeature {
                feature: "input_audio",
            }) => {}
            other => panic!("expected UnsupportedFeature(input_audio), got {other:?}"),
        }
    }

    #[test]
    fn audio_output_extra_field_returns_unsupported_feature() {
        let mut req = req_with(vec![msg("user", json!("hi"))]);
        req.extra
            .insert("audio".into(), json!({"voice":"alloy","format":"mp3"}));
        match translator().translate(&req) {
            Err(CodexError::UnsupportedFeature {
                feature: "audio_output",
            }) => {}
            other => panic!("expected UnsupportedFeature(audio_output), got {other:?}"),
        }
    }

    #[test]
    fn audio_output_modalities_audio_returns_unsupported_feature() {
        let mut req = req_with(vec![msg("user", json!("hi"))]);
        req.extra
            .insert("modalities".into(), json!(["text", "audio"]));
        match translator().translate(&req) {
            Err(CodexError::UnsupportedFeature {
                feature: "audio_output",
            }) => {}
            other => panic!("expected UnsupportedFeature(audio_output), got {other:?}"),
        }
    }

    #[test]
    fn response_format_text_is_omitted() {
        let mut req = req_with(vec![msg("user", json!("hi"))]);
        req.extra
            .insert("response_format".into(), json!({"type":"text"}));
        let out = translator().translate(&req).unwrap();
        assert!(out.get("text").is_none(), "text should be absent: {out:#}");
    }

    #[test]
    fn response_format_json_object_translates() {
        let mut req = req_with(vec![msg("user", json!("hi"))]);
        req.extra
            .insert("response_format".into(), json!({"type":"json_object"}));
        let out = translator().translate(&req).unwrap();
        assert_eq!(out["text"]["format"], json!({"type":"json_object"}));
    }

    #[test]
    fn response_format_json_schema_translates_with_default_strict() {
        let mut req = req_with(vec![msg("user", json!("hi"))]);
        req.extra.insert(
            "response_format".into(),
            json!({"type":"json_schema",
                   "json_schema":{"name":"Foo","schema":{"type":"object"}}}),
        );
        let out = translator().translate(&req).unwrap();
        assert_eq!(
            out["text"]["format"],
            json!({"type":"json_schema","name":"Foo",
                   "schema":{"type":"object"},"strict":false})
        );
    }

    #[test]
    fn response_format_json_schema_preserves_explicit_strict() {
        let mut req = req_with(vec![msg("user", json!("hi"))]);
        req.extra.insert(
            "response_format".into(),
            json!({"type":"json_schema",
                   "json_schema":{"name":"Foo","schema":{"x":1},"strict":true}}),
        );
        let out = translator().translate(&req).unwrap();
        assert_eq!(out["text"]["format"]["strict"], json!(true));
    }

    #[test]
    fn verbosity_propagates_when_recognized_else_omitted() {
        let mut req = req_with(vec![msg("user", json!("hi"))]);
        req.extra.insert("verbosity".into(), json!("medium"));
        let out = translator().translate(&req).unwrap();
        assert_eq!(out["text"]["verbosity"], json!("medium"));

        let mut req2 = req_with(vec![msg("user", json!("hi"))]);
        req2.extra.insert("verbosity".into(), json!("excessive"));
        let out2 = translator().translate(&req2).unwrap();
        assert!(out2.get("text").is_none(), "unknown verbosity dropped");
    }

    #[test]
    fn stripped_fields_absent_from_envelope() {
        let mut req = req_with(vec![msg("user", json!("hi"))]);
        req.max_tokens = Some(8000);
        for k in [
            "max_output_tokens",
            "max_completion_tokens",
            "max_tokens",
            "logprobs",
            "top_logprobs",
            "n",
            "seed",
            "presence_penalty",
            "frequency_penalty",
            "logit_bias",
            "user",
            "stop",
        ] {
            req.extra.insert(k.into(), json!("should-not-appear"));
        }
        let out = translator().translate(&req).unwrap();
        let obj = out.as_object().unwrap();
        for k in [
            "max_output_tokens",
            "max_completion_tokens",
            "max_tokens",
            "logprobs",
            "top_logprobs",
            "n",
            "seed",
            "presence_penalty",
            "frequency_penalty",
            "logit_bias",
            "user",
            "stop",
        ] {
            assert!(!obj.contains_key(k), "envelope leaked stripped field {k}");
        }
    }

    #[test]
    fn emit_reasoning_false_omits_reasoning_field() {
        let mut t = translator();
        t.emit_reasoning = false;
        let req = req_with(vec![msg("user", json!("hi"))]);
        let out = t.translate(&req).unwrap();
        assert!(out.get("reasoning").is_none(), "reasoning leaked: {out:#}");
    }

    #[test]
    fn emit_reasoning_true_includes_effort_and_summary() {
        let req = req_with(vec![msg("user", json!("hi"))]);
        let out = translator().translate(&req).unwrap();
        assert_eq!(
            out["reasoning"],
            json!({"effort":"medium","summary":"auto"})
        );
    }

    #[test]
    fn tools_pass_through_preserves_array_order_and_shape() {
        // Post-integration behavior: function tools are FLATTENED from
        // Chat Completions nested form `{"type":"function","function":{...}}`
        // to Responses API flat form `{"type":"function","name":...,...}`.
        // Non-function tool types are forwarded byte-for-byte unchanged.
        let mut req = req_with(vec![msg("user", json!("hi"))]);
        let tools = json!([
            {"type":"function","function":{"name":"f","description":"d","parameters":{"x":1},"strict":false}},
            {"type":"web_search"},
            {"type":"unknown-future","payload":{"q":1}},
        ]);
        req.extra.insert("tools".into(), tools);
        let out = translator().translate(&req).unwrap();
        let out_tools = out["tools"].as_array().unwrap();
        assert_eq!(out_tools.len(), 3, "all tools forwarded in order");

        // [0] function flattened: no `function` wrapper, fields hoisted.
        assert_eq!(
            out_tools[0],
            json!({
                "type": "function",
                "name": "f",
                "description": "d",
                "parameters": {"x": 1},
                "strict": false,
            })
        );
        assert!(
            out_tools[0].get("function").is_none(),
            "function tools must be flattened (no nested 'function' key)"
        );

        // [1] non-function: forwarded unchanged.
        assert_eq!(out_tools[1], json!({"type":"web_search"}));

        // [2] unknown-future: forwarded unchanged.
        assert_eq!(
            out_tools[2],
            json!({"type":"unknown-future","payload":{"q":1}})
        );
    }

    #[test]
    fn temperature_and_top_p_are_stripped() {
        // Post-integration behavior: the Codex backend rejects `temperature`
        // and `top_p` as "Unsupported parameter". The translator MUST strip
        // them from the outgoing envelope.
        let mut req = req_with(vec![msg("user", json!("hi"))]);
        req.temperature = Some(0.7);
        req.extra.insert("top_p".into(), json!(0.9));
        let out = translator().translate(&req).unwrap();
        let obj = out.as_object().unwrap();
        assert!(
            !obj.contains_key("temperature"),
            "temperature must be stripped from envelope: {out:#}"
        );
        assert!(
            !obj.contains_key("top_p"),
            "top_p must be stripped from envelope: {out:#}"
        );
    }

    #[test]
    fn no_id_field_emitted_on_input_items() {
        // Build one of every shape, then recursively scan for any "id" key.
        let mut a = msg("assistant", Value::Null);
        a.extra.insert(
            "tool_calls".into(),
            json!([{"id":"c1","type":"function","function":{"name":"x","arguments":"{}"}}]),
        );
        let mut t = msg("tool", json!("ok"));
        t.extra.insert("tool_call_id".into(), json!("c1"));
        let req = req_with(vec![
            msg("system", json!("s")),
            msg("user", json!("u")),
            msg(
                "user",
                json!([{"type":"text","text":"x"},
                       {"type":"image_url","image_url":{"url":"u"}}]),
            ),
            a,
            t,
        ]);
        let out = translator().translate(&req).unwrap();
        let input = &out["input"];
        fn has_id(v: &Value) -> bool {
            match v {
                Value::Object(map) => {
                    if map.contains_key("id") {
                        return true;
                    }
                    map.values().any(has_id)
                }
                Value::Array(items) => items.iter().any(has_id),
                _ => false,
            }
        }
        assert!(!has_id(input), "input emitted an id field: {input:#}");
    }

    #[test]
    fn message_order_preserved_across_mixed_roles() {
        // Post-integration behavior: system/developer messages are skipped
        // entirely (handled via the `instructions` envelope field), so they
        // do not appear in `input`. The remaining roles preserve order.
        let req = req_with(vec![
            msg("system", json!("s")),
            msg("user", json!("u1")),
            msg("assistant", json!("a1")),
            msg("user", json!("u2")),
        ]);
        let out = translator().translate(&req).unwrap();
        let roles: Vec<&str> = out["input"]
            .as_array()
            .unwrap()
            .iter()
            .map(|i| i["role"].as_str().unwrap())
            .collect();
        assert_eq!(roles, vec!["user", "assistant", "user"]);
    }
}

#[cfg(test)]
mod property_tests {
    //! Property-based tests for the request translator.
    //!
    //! **Validates: Requirements 13.2**
    //! **Validates: Requirements 13.6**
    //! **Validates: Requirements 13.8**

    use super::*;
    use proptest::prelude::*;
    use serde_json::{json, Map, Value};
    use std::collections::HashSet;

    /// Tagged union over the six legal source-message shapes from
    /// Req 3.1–3.6. Generated by [`shape_strategy`], then materialized into
    /// a [`Message`] via [`shape_to_message`] so the translator runs against
    /// real inputs.
    #[derive(Debug, Clone)]
    enum Shape {
        /// Req 3.1 — `system` or `developer` with string content.
        SystemOrDev { role: &'static str, text: String },
        /// Req 3.2 — `user` with string content.
        UserStr { text: String },
        /// Req 3.3 — `user` with array content of 0..=3 parts.
        UserArr { parts: Vec<Value> },
        /// Req 3.4 — `assistant` with non-empty string content.
        AssistantStr { text: String },
        /// Req 3.5 — `assistant` with `tool_calls` array of 1..=3 elements.
        AssistantTools { call_ids: Vec<String> },
        /// Req 3.6 — `tool` with a `tool_call_id` token. May or may not
        /// match a previously emitted assistant call id; both are legal
        /// under Property 2 (orphans fall through to user messages —
        /// Req 3.7 — but the assertion is about translated structure, not
        /// per-message kind mapping).
        Tool { call_id: String, output: String },
    }

    /// `[a-z]{1,8}` token pool used for call ids and function names.
    fn token() -> impl Strategy<Value = String> {
        "[a-z]{1,8}".prop_map(String::from)
    }

    /// One element of a user-message array `content` per Req 3.3.
    /// Restricted to `text` and `image_url` parts — `input_audio` parts
    /// would trigger [`CodexError::UnsupportedFeature`] (Req 14.3) and are
    /// out of scope for Property 2.
    fn user_part() -> impl Strategy<Value = Value> {
        prop_oneof![
            "[a-zA-Z]{0,32}".prop_map(|t| json!({"type": "text", "text": t})),
            "[a-z]{1,16}".prop_map(|u| {
                json!({
                    "type": "image_url",
                    "image_url": {"url": format!("https://x.example/{u}")},
                })
            }),
        ]
    }

    /// Strategy producing one of the six shapes uniformly.
    ///
    /// Assistant string content uses `[a-zA-Z]{1,32}` — guaranteed
    /// non-empty — so the translator's "assistant with empty content AND
    /// no tool_calls is silently dropped" branch never fires and the
    /// item-count formula is exact.
    fn shape_strategy() -> impl Strategy<Value = Shape> {
        prop_oneof![
            (
                prop_oneof![Just("system"), Just("developer")],
                "[a-zA-Z]{1,32}"
            )
                .prop_map(|(role, text)| Shape::SystemOrDev { role, text }),
            "[a-zA-Z]{1,32}".prop_map(|text| Shape::UserStr { text }),
            proptest::collection::vec(user_part(), 0..=3)
                .prop_map(|parts| Shape::UserArr { parts }),
            "[a-zA-Z]{1,32}".prop_map(|text| Shape::AssistantStr { text }),
            proptest::collection::vec(token(), 1..=3)
                .prop_map(|call_ids| Shape::AssistantTools { call_ids }),
            (token(), "[a-zA-Z]{0,32}")
                .prop_map(|(call_id, output)| Shape::Tool { call_id, output }),
        ]
    }

    /// Materialize a [`Shape`] into a real [`Message`] the translator
    /// can consume.
    fn shape_to_message(s: &Shape) -> Message {
        match s {
            Shape::SystemOrDev { role, text } => Message {
                role: (*role).to_string(),
                content: json!(text),
                extra: Map::new(),
            },
            Shape::UserStr { text } => Message {
                role: "user".into(),
                content: json!(text),
                extra: Map::new(),
            },
            Shape::UserArr { parts } => Message {
                role: "user".into(),
                content: Value::Array(parts.clone()),
                extra: Map::new(),
            },
            Shape::AssistantStr { text } => Message {
                role: "assistant".into(),
                content: json!(text),
                extra: Map::new(),
            },
            Shape::AssistantTools { call_ids } => {
                let arr: Vec<Value> = call_ids
                    .iter()
                    .map(|id| {
                        json!({
                            "id": id,
                            "type": "function",
                            "function": {"name": "fn", "arguments": "{}"},
                        })
                    })
                    .collect();
                let mut extra = Map::new();
                extra.insert("tool_calls".into(), Value::Array(arr));
                Message {
                    role: "assistant".into(),
                    content: Value::Null,
                    extra,
                }
            }
            Shape::Tool { call_id, output } => {
                let mut extra = Map::new();
                extra.insert("tool_call_id".into(), json!(call_id));
                Message {
                    role: "tool".into(),
                    content: json!(output),
                    extra,
                }
            }
        }
    }

    /// Recursive scan: returns `true` iff any object in the tree contains
    /// an `"id"` key. Mirrors the helper from the existing
    /// `no_id_field_emitted_on_input_items` unit test.
    fn has_id(v: &Value) -> bool {
        match v {
            Value::Object(map) => {
                if map.contains_key("id") {
                    return true;
                }
                map.values().any(has_id)
            }
            Value::Array(items) => items.iter().any(has_id),
            _ => false,
        }
    }

    fn translator<'a>() -> ChatToResponsesTranslator<'a> {
        ChatToResponsesTranslator {
            resolved_model: "gpt-5.1",
            instructions: "you are codex",
            mapped_effort: "medium",
            emit_reasoning: true,
        }
    }

    proptest! {
        /// **Property 2 — Message Shape Preservation**
        ///
        /// For all `OpenAIRequest` whose messages are any permutation of
        /// the six shapes in Req 3.1–3.6, the translated `input` array:
        ///
        /// (a) contains no `"id"` field anywhere (recursive scan)
        /// (b) preserves source message order
        /// (c) emits exactly one input item per non-assistant-tool-calls
        ///     message and one per `tool_calls` element for
        ///     assistant-tool-calls messages
        ///
        /// Tool messages with orphan `tool_call_id`s map to user messages
        /// per Req 3.7; both that branch and the matched-id
        /// `function_call_output` branch are exercised (orphans dominate
        /// because the token pool is wider than the per-request emitted
        /// id set).
        ///
        /// Validates: Requirements 13.2
        #[test]
        fn message_shape_preservation(
            shapes in proptest::collection::vec(shape_strategy(), 1..=8)
        ) {
            let messages: Vec<Message> = shapes.iter().map(shape_to_message).collect();
            let req = OpenAIRequest {
                model: "gpt-5.1".into(),
                messages,
                stream: false,
                temperature: None,
                max_tokens: None,
                extra: Map::new(),
            };

            let out = translator()
                .translate(&req)
                .expect("translate must succeed for the six legal shapes");
            let input = out["input"]
                .as_array()
                .expect("envelope.input must be an array")
                .clone();

            // (a) No `id` field anywhere in the translated array (Req 3.8).
            prop_assert!(
                !has_id(&Value::Array(input.clone())),
                "translated input contains an id field: {:#}",
                Value::Array(input.clone())
            );

            // (b) + (c) Lockstep walk: track call_ids emitted by previous
            // assistant tool_calls so tool-message expectations match the
            // translator's single forward pass. Note: system/developer
            // messages are skipped entirely (Codex backend rejects them),
            // so they advance the source-shape cursor without consuming
            // an output item.
            let mut idx: usize = 0;
            let mut emitted_call_ids: HashSet<String> = HashSet::new();
            for shape in &shapes {
                match shape {
                    Shape::SystemOrDev { .. } => {
                        // Skipped — not emitted as an input item.
                    }
                    Shape::UserStr { .. } | Shape::UserArr { .. } => {
                        prop_assert!(idx < input.len(), "ran out of items");
                        prop_assert_eq!(input[idx]["type"].as_str(), Some("message"));
                        prop_assert_eq!(input[idx]["role"].as_str(), Some("user"));
                        idx += 1;
                    }
                    Shape::AssistantStr { .. } => {
                        prop_assert!(idx < input.len(), "ran out of items");
                        prop_assert_eq!(input[idx]["type"].as_str(), Some("message"));
                        prop_assert_eq!(input[idx]["role"].as_str(), Some("assistant"));
                        idx += 1;
                    }
                    Shape::AssistantTools { call_ids } => {
                        for id in call_ids {
                            prop_assert!(
                                idx < input.len(),
                                "ran out of items mid-tool_calls"
                            );
                            prop_assert_eq!(
                                input[idx]["type"].as_str(),
                                Some("function_call")
                            );
                            prop_assert!(
                                input[idx].get("role").is_none(),
                                "function_call must not carry a role: {:#}",
                                input[idx]
                            );
                            idx += 1;
                            emitted_call_ids.insert(id.clone());
                        }
                    }
                    Shape::Tool { call_id, .. } => {
                        prop_assert!(idx < input.len(), "ran out of items");
                        if emitted_call_ids.contains(call_id) {
                            // Matched call_id → function_call_output (Req 3.6).
                            prop_assert_eq!(
                                input[idx]["type"].as_str(),
                                Some("function_call_output")
                            );
                            prop_assert!(
                                input[idx].get("role").is_none(),
                                "function_call_output must not carry a role: {:#}",
                                input[idx]
                            );
                        } else {
                            // Orphan → user message fallback (Req 3.7).
                            prop_assert_eq!(input[idx]["type"].as_str(), Some("message"));
                            prop_assert_eq!(input[idx]["role"].as_str(), Some("user"));
                        }
                        idx += 1;
                    }
                }
            }

            // (c) Exact item count — no extras, no shortfalls.
            prop_assert_eq!(
                idx,
                input.len(),
                "translated array length mismatch: consumed={}, total={}",
                idx,
                input.len()
            );
        }
    }

    // ----------------------------------------------------------------
    // Property 8 — Tools Pass-Through (Validates Requirements 13.8)
    // ----------------------------------------------------------------

    /// One opaque tool object. Carries a random `type` drawn from the
    /// known + "unknown-future" pool. When `type == "function"` it carries
    /// a Chat Completions–style nested `function` object (translator must
    /// flatten it). Other tool types never carry a `function` field, so
    /// they pass through byte-for-byte (Req 2.12, 14.2).
    fn tool_strategy() -> impl Strategy<Value = Value> {
        let type_strategy = prop::sample::select(vec![
            "function",
            "web_search",
            "file_search",
            "code_interpreter",
            "mcp",
            "custom",
            "unknown-future",
        ]);
        let function_payload = (
            "[a-zA-Z_][a-zA-Z0-9_]{0,16}",
            prop::option::of("[a-zA-Z]{0,32}"),
        )
            .prop_map(|(name, param_value)| {
                let mut params = Map::new();
                if let Some(v) = param_value {
                    params.insert("v".into(), Value::String(v));
                }
                json!({"name": name, "parameters": Value::Object(params)})
            });
        let body_strategy = prop::option::of("[a-zA-Z0-9 _-]{0,64}".prop_map(String::from));

        (type_strategy, function_payload, body_strategy).prop_map(|(t, function, body)| {
            let mut obj = Map::new();
            obj.insert("type".into(), Value::String(t.to_string()));
            // Only `function` tools carry a nested `function` object.
            if t == "function" {
                obj.insert("function".into(), function);
            }
            if let Some(b) = body {
                obj.insert("body".into(), Value::String(b));
            }
            Value::Object(obj)
        })
    }

    proptest! {
        /// **Property 8 — Tools Pass-Through (with function flattening)**
        ///
        /// For any `tools[]` array containing a mix of `type` values drawn
        /// from `{function, web_search, file_search, code_interpreter, mcp,
        /// custom, unknown-future}`, the translated Responses_Request's
        /// `tools` array preserves length and order, with the following
        /// per-element rule (Req 2.12, 14.2):
        ///
        /// * `type == "function"` → flattened: top-level `name`,
        ///   `description`, `parameters` lifted out of the nested
        ///   `function` object; no `function` wrapper remains.
        /// * any other `type` → forwarded byte-for-byte unchanged.
        ///
        /// Validates: Requirements 13.8
        #[test]
        fn tools_pass_through(
            input_tools in proptest::collection::vec(tool_strategy(), 1..=5)
        ) {
            let mut req = OpenAIRequest {
                model: "gpt-5.1".into(),
                messages: vec![Message {
                    role: "user".into(),
                    content: json!("hi"),
                    extra: Map::new(),
                }],
                stream: false,
                temperature: None,
                max_tokens: None,
                extra: Map::new(),
            };
            let tools_value = Value::Array(input_tools.clone());
            req.extra.insert("tools".into(), tools_value);

            let out = translator()
                .translate(&req)
                .expect("translate must succeed for arbitrary tool arrays");

            let out_tools = out["tools"]
                .as_array()
                .expect("tools must be an array")
                .clone();
            prop_assert_eq!(out_tools.len(), input_tools.len(), "tool count drift");

            for (i, (got, src)) in out_tools.iter().zip(input_tools.iter()).enumerate() {
                let src_type = src.get("type").and_then(|t| t.as_str());
                if src_type == Some("function") {
                    // Flattened shape: type=function, no nested `function`.
                    prop_assert_eq!(
                        got.get("type").and_then(|t| t.as_str()),
                        Some("function"),
                        "tool {} type drift",
                        i
                    );
                    prop_assert!(
                        got.get("function").is_none(),
                        "tool {} should be flattened (no nested 'function'): {:#}",
                        i,
                        got
                    );
                    // Hoisted fields (when present in source) must match.
                    if let Some(src_fn) = src.get("function") {
                        if let Some(name) = src_fn.get("name") {
                            prop_assert_eq!(
                                got.get("name"),
                                Some(name),
                                "tool {} name drift",
                                i
                            );
                        }
                        if let Some(params) = src_fn.get("parameters") {
                            prop_assert_eq!(
                                got.get("parameters"),
                                Some(params),
                                "tool {} parameters drift",
                                i
                            );
                        }
                        if let Some(desc) = src_fn.get("description") {
                            prop_assert_eq!(
                                got.get("description"),
                                Some(desc),
                                "tool {} description drift",
                                i
                            );
                        }
                    }
                } else {
                    // Non-function tools: byte-for-byte equality.
                    prop_assert_eq!(got, src, "non-function tool {} not forwarded as-is", i);
                }
            }
        }
    }

    // ----------------------------------------------------------------
    // Property 6 — Request Round-Trip (Text-Only) (Validates Req 13.6)
    // ----------------------------------------------------------------

    /// Test-only inverse of [`ChatToResponsesTranslator::translate`] restricted
    /// to the text-only subset covered by Property 6: roles
    /// `{system, user, assistant}` with string content, no tool calls,
    /// no image parts. Walks the Responses envelope's `input` array and
    /// reconstructs `(model, messages, temperature, top_p)`.
    fn responses_to_chat(envelope: &Value) -> (String, Vec<Message>, Option<f32>, Option<f64>) {
        let model = envelope
            .get("model")
            .and_then(|v| v.as_str())
            .unwrap_or("")
            .to_string();

        let mut messages: Vec<Message> = Vec::new();
        if let Some(items) = envelope.get("input").and_then(|v| v.as_array()) {
            for item in items {
                if item.get("type").and_then(|t| t.as_str()) != Some("message") {
                    continue;
                }
                let role = item
                    .get("role")
                    .and_then(|r| r.as_str())
                    .unwrap_or("")
                    .to_string();
                // Concatenate every input_text/output_text segment in order.
                let mut text = String::new();
                if let Some(parts) = item.get("content").and_then(|c| c.as_array()) {
                    for p in parts {
                        let pt = p.get("type").and_then(|t| t.as_str());
                        if matches!(pt, Some("input_text") | Some("output_text")) {
                            if let Some(s) = p.get("text").and_then(|t| t.as_str()) {
                                text.push_str(s);
                            }
                        }
                    }
                }
                messages.push(Message {
                    role,
                    content: json!(text),
                    extra: Map::new(),
                });
            }
        }

        let temperature = envelope
            .get("temperature")
            .and_then(|v| v.as_f64())
            .map(|x| x as f32);
        let top_p = envelope.get("top_p").and_then(|v| v.as_f64());

        (model, messages, temperature, top_p)
    }

    /// Build a translator pinned to a runtime `resolved_model` so the
    /// property's generated `model` survives the round-trip unchanged.
    fn translator_with_resolved(model: &str) -> ChatToResponsesTranslator<'_> {
        ChatToResponsesTranslator {
            resolved_model: model,
            instructions: "you are codex",
            mapped_effort: "medium",
            emit_reasoning: true,
        }
    }

    /// One text-only message: role drawn from `{user, assistant}`
    /// with non-empty ASCII content. Note: `system` is excluded because
    /// the translator skips system/developer messages entirely (the Codex
    /// backend rejects them); they cannot survive a round-trip.
    /// Non-empty content avoids the translator's silent-drop branch for
    /// empty assistant turns; ASCII avoids JSON-escape edge cases that
    /// would break literal byte equality.
    fn text_only_message_strategy() -> impl Strategy<Value = Message> {
        (
            prop_oneof![Just("user"), Just("assistant")],
            "[a-zA-Z0-9 ]{1,128}",
        )
            .prop_map(|(role, text)| Message {
                role: role.into(),
                content: json!(text),
                extra: Map::new(),
            })
    }

    proptest! {
        /// **Property 6 — Request Round-Trip (Text-Only)**
        ///
        /// For all `OpenAIRequest` restricted to roles `{user, assistant}`
        /// with string content (no tool calls, no image parts), translating
        /// to Responses_Request and inverting via the test-only
        /// [`responses_to_chat`] helper preserves
        /// `(model, messages after role/content normalization)`.
        ///
        /// Note: `temperature` and `top_p` are NOT preserved by round-trip
        /// — the Codex backend rejects them so the translator strips them
        /// unconditionally. They are still varied by the strategy to
        /// confirm stripping is total (the inverse must observe `None` on
        /// both sides regardless of input).
        ///
        /// `system` messages are excluded because the translator skips
        /// them (handled via the `instructions` envelope field).
        ///
        /// Validates: Requirements 13.6
        #[test]
        fn request_round_trip_text_only(
            model in "[a-z][a-z0-9.-]{1,32}",
            messages in proptest::collection::vec(text_only_message_strategy(), 1..=6),
            temperature in prop::option::of(0.0f32..=2.0f32),
            top_p in prop::option::of(0.0f64..=1.0f64),
        ) {
            let mut extra = Map::new();
            if let Some(tp) = top_p {
                extra.insert("top_p".into(), json!(tp));
            }
            let req = OpenAIRequest {
                model: model.clone(),
                messages: messages.clone(),
                stream: false,
                temperature,
                max_tokens: None,
                extra,
            };

            let envelope = translator_with_resolved(&model)
                .translate(&req)
                .expect("translate must succeed for the text-only subset");

            let (got_model, got_msgs, got_temp, got_top_p) = responses_to_chat(&envelope);

            prop_assert_eq!(&got_model, &model);
            prop_assert_eq!(got_msgs.len(), messages.len());
            for (i, (a, b)) in got_msgs.iter().zip(messages.iter()).enumerate() {
                prop_assert_eq!(&a.role, &b.role, "msg {} role drift", i);
                prop_assert_eq!(
                    a.content_as_text(),
                    b.content_as_text(),
                    "msg {} content drift",
                    i
                );
            }

            // Temperature and top_p are stripped by the translator (the
            // Codex backend rejects them as "Unsupported parameter"), so
            // the envelope never carries them and the inverse observes
            // `None` regardless of the input strategy.
            prop_assert!(
                got_temp.is_none(),
                "temperature must be stripped from envelope (got {:?})",
                got_temp
            );
            prop_assert!(
                got_top_p.is_none(),
                "top_p must be stripped from envelope (got {:?})",
                got_top_p
            );
        }
    }
}
