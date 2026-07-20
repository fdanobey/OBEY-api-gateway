//! Compression-engine interfaces and implementations.

use crate::compression::{protection::ProtectionScanner, token_counter::TokenCounter};
use crate::models::openai::{Message, OpenAIRequest};
use async_trait::async_trait;
use serde::{Deserialize, Serialize};
use serde_json::{Map, Value};
use std::{collections::HashMap, fmt, sync::Arc};

pub mod aggressive;
pub mod language_pack;
pub mod lite;
pub mod perplexity;
pub mod rtk;
pub mod standard;
pub mod tool_def;
pub mod ultra;

/// Sentinel age used for system messages, which are always compression-critical.
pub const SYSTEM_MESSAGE_AGE: usize = usize::MAX;

/// JSON message content retained without lossy conversion.
#[derive(Debug, Clone, Default, PartialEq, Serialize, Deserialize)]
#[serde(transparent)]
pub struct MessageContent(Value);

impl MessageContent {
    pub fn new(value: Value) -> Self {
        Self(value)
    }

    pub fn as_value(&self) -> &Value {
        &self.0
    }

    pub fn as_value_mut(&mut self) -> &mut Value {
        &mut self.0
    }

    pub fn into_value(self) -> Value {
        self.0
    }

    pub fn as_text(&self) -> Option<&str> {
        self.0.as_str()
    }

    /// Applies a transformation only to model-visible prose leaves.
    ///
    /// Root strings and text content blocks are eligible. Structured values,
    /// image URLs, tool inputs, call identifiers, and other provider-specific
    /// fields are never traversed as prose.
    pub fn transform_text_leaves<F>(&mut self, mut transform: F)
    where
        F: FnMut(&str) -> String,
    {
        Self::visit_text_leaves_mut(&mut self.0, true, &mut |text| {
            *text = transform(text);
        });
    }

    /// Visits mutable model-visible prose leaves without flattening content.
    pub fn for_each_text_leaf_mut<F>(&mut self, mut visitor: F)
    where
        F: FnMut(&mut String),
    {
        Self::visit_text_leaves_mut(&mut self.0, true, &mut visitor);
    }

    fn visit_text_leaves_mut<F>(value: &mut Value, root: bool, visitor: &mut F)
    where
        F: FnMut(&mut String),
    {
        match value {
            Value::String(text) if root => visitor(text),
            Value::Array(parts) if root => {
                for part in parts {
                    match part {
                        Value::String(text) => visitor(text),
                        Value::Object(_) => Self::visit_content_block_mut(part, visitor),
                        _ => {}
                    }
                }
            }
            Value::Object(_) if root => Self::visit_content_block_mut(value, visitor),
            _ => {}
        }
    }

    fn visit_content_block_mut<F>(value: &mut Value, visitor: &mut F)
    where
        F: FnMut(&mut String),
    {
        let Some(object) = value.as_object_mut() else {
            return;
        };
        let block_type = object.get("type").and_then(Value::as_str);

        match block_type {
            Some("text" | "input_text" | "output_text") => {
                if let Some(Value::String(text)) = object.get_mut("text") {
                    visitor(text);
                }
            }
            Some("tool_result") => {
                if let Some(content) = object.get_mut("content") {
                    Self::visit_text_leaves_mut(content, true, visitor);
                }
            }
            _ => {}
        }
    }
}

impl From<Value> for MessageContent {
    fn from(value: Value) -> Self {
        Self::new(value)
    }
}

impl From<MessageContent> for Value {
    fn from(content: MessageContent) -> Self {
        content.into_value()
    }
}

/// Derived tool-call links used to keep calls and results together.
#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub struct ToolRelationshipMetadata {
    pub tool_call_ids: Vec<String>,
    pub tool_result_for_ids: Vec<String>,
    pub tool_names: Vec<String>,
    /// Original message indices linked to this message.
    pub related_message_indices: Vec<usize>,
    /// Call IDs with no later matching result.
    pub unresolved_tool_call_ids: Vec<String>,
}

/// A message plus compression-only metadata.
#[derive(Debug, Clone, Default, PartialEq)]
pub struct CompressibleMessage {
    pub role: String,
    pub content: MessageContent,
    pub extra: Map<String, Value>,
    /// User-turn-relative age; zero is the latest user segment.
    pub age: usize,
    pub token_count: u32,
    /// True for every message at or before the last structural cache marker.
    pub cache_protected: bool,
    pub original_index: usize,
    /// System, latest-user, and unresolved tool-call messages are critical.
    pub critical: bool,
    pub relationships: ToolRelationshipMetadata,
}

impl CompressibleMessage {
    pub fn is_system(&self) -> bool {
        self.role == "system"
    }

    pub fn has_cache_marker(&self) -> bool {
        value_has_cache_marker(self.content.as_value())
            || self.extra.values().any(value_has_cache_marker)
            || self.extra.contains_key("cache_control")
    }
}

impl From<CompressibleMessage> for Message {
    fn from(message: CompressibleMessage) -> Self {
        Self {
            role: message.role,
            content: message.content.into_value(),
            extra: message.extra,
        }
    }
}

/// Input to the compression pipeline with all OpenAI wire data preserved.
#[derive(Debug, Clone, Default, PartialEq)]
pub struct CompressiblePayload {
    pub model: String,
    pub messages: Vec<CompressibleMessage>,
    pub stream: bool,
    pub temperature: Option<f32>,
    pub max_tokens: Option<u32>,
    /// The original top-level `tools` value, if present (including `null`).
    pub tool_definitions: Option<Value>,
    /// All flattened top-level fields except `tools`.
    pub extra: Map<String, Value>,
}

impl CompressiblePayload {
    pub fn from_openai_request(request: OpenAIRequest) -> Self {
        request.into()
    }

    pub fn into_openai_request(self) -> OpenAIRequest {
        self.into()
    }

    pub fn tools(&self) -> Option<&Value> {
        self.tool_definitions.as_ref()
    }

    /// Recomputes derived ages, cache boundaries, and tool relationships.
    /// Original indices and wire values remain unchanged.
    pub fn refresh_metadata(&mut self) {
        assign_message_ages(&mut self.messages);
        assign_cache_protection(&mut self.messages);
        assign_tool_relationships(&mut self.messages);
    }
}

impl From<OpenAIRequest> for CompressiblePayload {
    fn from(request: OpenAIRequest) -> Self {
        let OpenAIRequest {
            model,
            messages,
            stream,
            temperature,
            max_tokens,
            mut extra,
        } = request;
        let tool_definitions = extra.remove("tools");
        let counter = TokenCounter::new();
        let messages = messages
            .into_iter()
            .enumerate()
            .map(|(original_index, message)| CompressibleMessage {
                token_count: count_message_tokens(&counter, &model, &message),
                role: message.role,
                content: MessageContent::new(message.content),
                extra: message.extra,
                age: 0,
                cache_protected: false,
                original_index,
                critical: false,
                relationships: ToolRelationshipMetadata::default(),
            })
            .collect();

        let mut payload = Self {
            model,
            messages,
            stream,
            temperature,
            max_tokens,
            tool_definitions,
            extra,
        };
        payload.refresh_metadata();
        payload
    }
}

impl From<&OpenAIRequest> for CompressiblePayload {
    fn from(request: &OpenAIRequest) -> Self {
        request.clone().into()
    }
}

impl From<CompressiblePayload> for OpenAIRequest {
    fn from(payload: CompressiblePayload) -> Self {
        let mut extra = payload.extra;
        if let Some(tools) = payload.tool_definitions {
            extra.insert("tools".to_owned(), tools);
        }

        Self {
            model: payload.model,
            messages: payload.messages.into_iter().map(Message::from).collect(),
            stream: payload.stream,
            temperature: payload.temperature,
            max_tokens: payload.max_tokens,
            extra,
        }
    }
}

/// Contextual information available to compression engines.
#[derive(Clone)]
pub struct CompressionContext {
    pub model: String,
    pub context_window: u32,
    pub target_token_budget: Option<u32>,
    pub provider_name: String,
    pub prompt_caching_enabled: bool,
    pub language: String,
    pub protection_scanner: Arc<ProtectionScanner>,
    pub token_counter: Arc<TokenCounter>,
}

impl CompressionContext {
    pub fn new(model: impl Into<String>, provider_name: impl Into<String>) -> Self {
        Self {
            model: model.into(),
            provider_name: provider_name.into(),
            ..Self::default()
        }
    }
}

impl Default for CompressionContext {
    fn default() -> Self {
        Self {
            model: String::new(),
            context_window: 0,
            target_token_budget: None,
            provider_name: String::new(),
            prompt_caching_enabled: false,
            language: "en".to_owned(),
            protection_scanner: Arc::new(ProtectionScanner::default()),
            token_counter: Arc::new(TokenCounter::default()),
        }
    }
}

impl fmt::Debug for CompressionContext {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter
            .debug_struct("CompressionContext")
            .field("model", &self.model)
            .field("context_window", &self.context_window)
            .field("target_token_budget", &self.target_token_budget)
            .field("provider_name", &self.provider_name)
            .field("prompt_caching_enabled", &self.prompt_caching_enabled)
            .field("language", &self.language)
            .field("protection_scanner", &self.protection_scanner)
            .field("token_counter", &"TokenCounter")
            .finish()
    }
}

fn count_message_tokens(counter: &TokenCounter, model: &str, message: &Message) -> u32 {
    let content = match &message.content {
        Value::Null => 0,
        Value::String(text) => counter.count_text(model, text),
        structured => counter.count_text(model, &structured.to_string()),
    };
    let extra = if message.extra.is_empty() {
        0
    } else {
        counter.count_text(model, &Value::Object(message.extra.clone()).to_string())
    };

    4u32.saturating_add(counter.count_text(model, &message.role))
        .saturating_add(content)
        .saturating_add(extra)
}

fn assign_message_ages(messages: &mut [CompressibleMessage]) {
    let user_turns = messages
        .iter()
        .filter(|message| message.role == "user")
        .count();
    let mut seen_user_turns = 0usize;

    for message in messages {
        if message.role == "system" {
            message.age = SYSTEM_MESSAGE_AGE;
            message.critical = true;
            continue;
        }
        if message.role == "user" {
            seen_user_turns = seen_user_turns.saturating_add(1);
        }
        message.age = if user_turns == 0 {
            0
        } else {
            user_turns.saturating_sub(seen_user_turns.max(1))
        };
        message.critical = message.role == "user" && message.age == 0;
    }
}

fn assign_cache_protection(messages: &mut [CompressibleMessage]) {
    let boundary = messages
        .iter()
        .rposition(CompressibleMessage::has_cache_marker);
    for (index, message) in messages.iter_mut().enumerate() {
        message.cache_protected = boundary.is_some_and(|boundary| index <= boundary);
    }
}

fn assign_tool_relationships(messages: &mut [CompressibleMessage]) {
    for message in messages.iter_mut() {
        message.relationships = extract_tool_relationships(&message.content, &message.extra);
    }

    let mut calls_by_id: HashMap<String, Vec<usize>> = HashMap::new();
    for (position, message) in messages.iter().enumerate() {
        for call_id in &message.relationships.tool_call_ids {
            calls_by_id
                .entry(call_id.clone())
                .or_default()
                .push(position);
        }
    }

    let mut matched_calls = Vec::new();
    let mut pairs = Vec::new();
    for (result_position, message) in messages.iter().enumerate() {
        for result_id in &message.relationships.tool_result_for_ids {
            let Some(call_position) = calls_by_id.get(result_id).and_then(|positions| {
                positions
                    .iter()
                    .rev()
                    .copied()
                    .find(|position| *position < result_position)
            }) else {
                continue;
            };
            matched_calls.push((call_position, result_id.clone()));
            pairs.push((call_position, result_position));
        }
    }

    for (call_position, result_position) in pairs {
        let call_original_index = messages[call_position].original_index;
        let result_original_index = messages[result_position].original_index;
        push_unique(
            &mut messages[call_position]
                .relationships
                .related_message_indices,
            result_original_index,
        );
        push_unique(
            &mut messages[result_position]
                .relationships
                .related_message_indices,
            call_original_index,
        );
    }

    for (position, message) in messages.iter_mut().enumerate() {
        let call_ids = message.relationships.tool_call_ids.clone();
        for call_id in call_ids {
            if !matched_calls.iter().any(|(matched_position, matched_id)| {
                *matched_position == position && matched_id == &call_id
            }) {
                push_unique(&mut message.relationships.unresolved_tool_call_ids, call_id);
            }
        }
        if !message.relationships.unresolved_tool_call_ids.is_empty() {
            message.critical = true;
        }
    }
}

fn extract_tool_relationships(
    content: &MessageContent,
    extra: &Map<String, Value>,
) -> ToolRelationshipMetadata {
    let mut metadata = ToolRelationshipMetadata::default();

    if let Some(tool_calls) = extra.get("tool_calls").and_then(Value::as_array) {
        for call in tool_calls {
            if let Some(call_id) = call.get("id").and_then(Value::as_str) {
                push_unique(&mut metadata.tool_call_ids, call_id.to_owned());
            }
            if let Some(name) = call
                .get("function")
                .and_then(|function| function.get("name"))
                .and_then(Value::as_str)
                .or_else(|| call.get("name").and_then(Value::as_str))
            {
                push_unique(&mut metadata.tool_names, name.to_owned());
            }
        }
    }
    if let Some(result_id) = extra.get("tool_call_id").and_then(Value::as_str) {
        push_unique(&mut metadata.tool_result_for_ids, result_id.to_owned());
    }
    if let Some(name) = extra
        .get("function_call")
        .and_then(|function| function.get("name"))
        .and_then(Value::as_str)
    {
        push_unique(&mut metadata.tool_names, name.to_owned());
    }

    collect_content_relationships(content.as_value(), &mut metadata);
    metadata
}

fn collect_content_relationships(value: &Value, metadata: &mut ToolRelationshipMetadata) {
    match value {
        Value::Array(values) => {
            for value in values {
                collect_content_relationships(value, metadata);
            }
        }
        Value::Object(object) => {
            match object.get("type").and_then(Value::as_str) {
                Some("tool_use") => {
                    if let Some(call_id) = object.get("id").and_then(Value::as_str) {
                        push_unique(&mut metadata.tool_call_ids, call_id.to_owned());
                    }
                    if let Some(name) = object.get("name").and_then(Value::as_str) {
                        push_unique(&mut metadata.tool_names, name.to_owned());
                    }
                }
                Some("tool_result") => {
                    if let Some(result_id) = object.get("tool_use_id").and_then(Value::as_str) {
                        push_unique(&mut metadata.tool_result_for_ids, result_id.to_owned());
                    }
                }
                Some("function_call") => {
                    if let Some(call_id) = object
                        .get("call_id")
                        .or_else(|| object.get("id"))
                        .and_then(Value::as_str)
                    {
                        push_unique(&mut metadata.tool_call_ids, call_id.to_owned());
                    }
                    if let Some(name) = object.get("name").and_then(Value::as_str) {
                        push_unique(&mut metadata.tool_names, name.to_owned());
                    }
                }
                Some("function_call_output") => {
                    if let Some(result_id) = object.get("call_id").and_then(Value::as_str) {
                        push_unique(&mut metadata.tool_result_for_ids, result_id.to_owned());
                    }
                }
                _ => {}
            }
            for nested in object.values() {
                collect_content_relationships(nested, metadata);
            }
        }
        _ => {}
    }
}

fn value_has_cache_marker(value: &Value) -> bool {
    match value {
        Value::Array(values) => values.iter().any(value_has_cache_marker),
        Value::Object(object) => {
            object.contains_key("cache_control") || object.values().any(value_has_cache_marker)
        }
        _ => false,
    }
}

fn push_unique<T: PartialEq>(values: &mut Vec<T>, value: T) {
    if !values.contains(&value) {
        values.push(value);
    }
}

/// A single compression engine implementing one strategy.
#[async_trait]
pub trait CompressionEngine: Send + Sync {
    /// Returns the unique engine name.
    fn name(&self) -> &str;

    /// Compresses the payload and returns per-engine statistics.
    async fn compress(
        &self,
        payload: &mut CompressiblePayload,
        context: &CompressionContext,
    ) -> EngineResult;
}

/// Per-engine execution result.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct EngineResult {
    pub engine_name: String,
    pub tokens_before: u32,
    pub tokens_after: u32,
    pub duration_ms: u64,
    pub applied: bool,
}

/// Named compression preset used to resolve an ordered engine chain.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum CompressionLevel {
    None,
    #[default]
    Lite,
    Standard,
    Aggressive,
    Ultra,
    Rtk,
    Stacked,
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    fn request_from_json(value: Value) -> OpenAIRequest {
        serde_json::from_value(value).unwrap()
    }

    #[test]
    fn compression_level_defaults_to_lite() {
        assert_eq!(CompressionLevel::default(), CompressionLevel::Lite);
    }

    #[test]
    fn compression_levels_round_trip_as_snake_case() {
        let cases = [
            (CompressionLevel::None, "\"none\""),
            (CompressionLevel::Lite, "\"lite\""),
            (CompressionLevel::Standard, "\"standard\""),
            (CompressionLevel::Aggressive, "\"aggressive\""),
            (CompressionLevel::Ultra, "\"ultra\""),
            (CompressionLevel::Rtk, "\"rtk\""),
            (CompressionLevel::Stacked, "\"stacked\""),
        ];

        for (level, serialized) in cases {
            assert_eq!(serde_json::to_string(&level).unwrap(), serialized);
            assert_eq!(
                serde_json::from_str::<CompressionLevel>(serialized).unwrap(),
                level
            );
        }
    }

    #[test]
    fn openai_request_round_trip_preserves_all_wire_values_and_relationships() {
        let original_json = json!({
            "model": "gpt-4o",
            "stream": true,
            "temperature": 0.25,
            "max_tokens": 4096,
            "top_p": 0.91,
            "metadata": {"tenant": "alpha", "nested": [1, true, null]},
            "response_format": {"type": "json_schema", "json_schema": {"name": "answer"}},
            "tools": [{
                "type": "function",
                "function": {
                    "name": "lookup",
                    "description": "Look up a record",
                    "parameters": {
                        "type": "object",
                        "properties": {"id": {"type": "string"}},
                        "required": ["id"]
                    }
                },
                "provider_extra": "kept"
            }],
            "tool_choice": {"type": "function", "function": {"name": "lookup"}},
            "messages": [
                {
                    "role": "system",
                    "content": "Keep exact structure.",
                    "name": "policy",
                    "provider_field": {"a": 1}
                },
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": "Find this image"},
                        {"type": "image_url", "image_url": {"url": "data:image/png;base64,AAEC"}, "detail": "high"}
                    ],
                    "name": "customer"
                },
                {
                    "role": "assistant",
                    "content": [
                        {"type": "text", "text": "Calling lookup"},
                        {"type": "tool_use", "id": "anthropic_1", "name": "lookup", "input": {"id": "A-1"}}
                    ],
                    "tool_calls": [{
                        "id": "call_1",
                        "type": "function",
                        "function": {"name": "lookup", "arguments": "{\"id\":\"A-1\"}"},
                        "vendor": {"trace": true}
                    }],
                    "reasoning_details": [{"type": "summary", "text": "private extra"}]
                },
                {
                    "role": "tool",
                    "content": [
                        {"type": "tool_result", "tool_use_id": "anthropic_1", "content": [{"type": "text", "text": "Found"}, {"record": {"id": "A-1"}}]}
                    ],
                    "tool_call_id": "call_1",
                    "name": "lookup",
                    "vendor_status": {"ok": true}
                }
            ]
        });
        let request = request_from_json(original_json.clone());

        let payload = CompressiblePayload::from(request);

        assert!(payload.extra.get("tools").is_none());
        assert_eq!(
            payload.tool_definitions.as_ref(),
            original_json.get("tools")
        );
        assert!(payload
            .messages
            .iter()
            .all(|message| message.token_count > 0));
        assert_eq!(
            payload.messages[2].relationships.tool_call_ids,
            ["call_1", "anthropic_1"]
        );
        assert_eq!(
            payload.messages[2].relationships.related_message_indices,
            [3]
        );
        assert_eq!(
            payload.messages[3].relationships.related_message_indices,
            [2]
        );
        assert!(payload.messages[2]
            .relationships
            .unresolved_tool_call_ids
            .is_empty());

        let round_trip: OpenAIRequest = payload.into();
        assert_eq!(serde_json::to_value(round_trip).unwrap(), original_json);
    }

    #[test]
    fn message_age_is_relative_to_user_turn_segments() {
        let request = request_from_json(json!({
            "model": "test",
            "messages": [
                {"role": "system", "content": "policy"},
                {"role": "assistant", "content": "preamble"},
                {"role": "user", "content": "old question"},
                {"role": "assistant", "content": "old answer"},
                {"role": "tool", "content": "old tool output", "tool_call_id": "old"},
                {"role": "user", "content": "latest question"},
                {"role": "assistant", "content": "latest answer"},
                {"role": "tool", "content": "latest tool output", "tool_call_id": "latest"}
            ]
        }));

        let payload = CompressiblePayload::from(request);
        let ages: Vec<_> = payload.messages.iter().map(|message| message.age).collect();

        assert_eq!(ages, [SYSTEM_MESSAGE_AGE, 1, 1, 1, 1, 0, 0, 0]);
        assert!(payload.messages[0].critical);
        assert!(payload.messages[5].critical);
        assert!(!payload.messages[2].critical);
    }

    #[test]
    fn structural_cache_markers_protect_the_prefix_without_wire_changes() {
        let original_json = json!({
            "model": "claude-compatible",
            "stream": false,
            "messages": [
                {"role": "system", "content": "policy"},
                {
                    "role": "user",
                    "content": [{
                        "type": "text",
                        "text": "cached prompt",
                        "cache_control": {"type": "ephemeral", "ttl": "5m"}
                    }]
                },
                {"role": "assistant", "content": "uncached answer"},
                {
                    "role": "user",
                    "content": "second boundary",
                    "cache_control": {"type": "ephemeral"}
                },
                {"role": "assistant", "content": "after boundary"}
            ]
        });

        let payload = CompressiblePayload::from(request_from_json(original_json.clone()));
        let protected: Vec<_> = payload
            .messages
            .iter()
            .map(|message| message.cache_protected)
            .collect();

        assert_eq!(protected, [true, true, true, true, false]);
        let round_trip: OpenAIRequest = payload.into();
        assert_eq!(serde_json::to_value(round_trip).unwrap(), original_json);
    }

    #[test]
    fn text_leaf_transform_does_not_touch_structured_or_multimodal_fields() {
        let mut content = MessageContent::new(json!([
            {"type": "text", "text": "  visible text  ", "cache_control": {"type": "ephemeral"}},
            {"type": "image_url", "image_url": {"url": "https://example.test/a.png"}},
            {"type": "tool_use", "id": "call_1", "name": "lookup", "input": {"text": "do not alter"}},
            {"type": "tool_result", "tool_use_id": "call_1", "content": [{"type": "text", "text": "  result text  "}, {"text": "structured value"}]}
        ]));

        content.transform_text_leaves(|text| text.trim().to_uppercase());

        assert_eq!(content.as_value()[0]["text"], "VISIBLE TEXT");
        assert_eq!(
            content.as_value()[1]["image_url"]["url"],
            "https://example.test/a.png"
        );
        assert_eq!(content.as_value()[2]["input"]["text"], "do not alter");
        assert_eq!(content.as_value()[3]["content"][0]["text"], "RESULT TEXT");
        assert_eq!(
            content.as_value()[3]["content"][1]["text"],
            "structured value"
        );
    }
}
