//! Typed models for the Anthropic Messages API (`/v1/messages` front door).
//!
//! Gateway-owned protocol types mirroring the conventions of
//! [`crate::responses::models`] and [`crate::models::openai`]: known fields are
//! explicit, unknown fields are captured in `extra` maps via `#[serde(flatten)]`
//! and passed through, and nothing uses `deny_unknown_fields`.
//!
//! These types are wire-only: no translation or policy logic lives here.

use serde::{Deserialize, Serialize};
use serde_json::Value;

type Extra = serde_json::Map<String, Value>;

/// Generate a message id in the documented `msg_` + hyphenated UUID format
/// (shared by the buffered and streaming synthesizers so both surfaces emit
/// the same id shape).
pub fn new_message_id() -> String {
    format!("msg_{}", uuid::Uuid::new_v4())
}

// ---------------------------------------------------------------------------
// Request
// ---------------------------------------------------------------------------

/// Anthropic Messages API request body (`POST /v1/messages`).
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct AnthropicRequest {
    pub model: String,
    pub messages: Vec<AnthropicMessage>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub system: Option<SystemPrompt>,
    /// Required by the Anthropic Messages API.
    pub max_tokens: u32,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub temperature: Option<f32>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub top_p: Option<f32>,
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub stop_sequences: Vec<String>,
    #[serde(default)]
    pub stream: bool,
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub tools: Vec<AnthropicToolDefinition>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub tool_choice: Option<AnthropicToolChoice>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub thinking: Option<ThinkingConfig>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub metadata: Option<RequestMetadata>,
    /// Catch-all for anything not modelled above.
    #[serde(flatten)]
    pub extra: Extra,
}

/// One entry of `messages[]` (`user` or `assistant` role).
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct AnthropicMessage {
    pub role: String,
    pub content: MessageContent,
    /// Catch-all (e.g. future per-message fields).
    #[serde(flatten)]
    pub extra: Extra,
}

/// Message content: a bare string or a list of typed content blocks.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(untagged)]
pub enum MessageContent {
    Text(String),
    Blocks(Vec<ContentBlock>),
}

/// `system`: a bare string or an array of text blocks.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(untagged)]
pub enum SystemPrompt {
    Text(String),
    Blocks(Vec<SystemBlock>),
}

/// One block of a `system` block array.
///
/// `type` is kept as a raw string (Anthropic only documents `"text"` today) so
/// future block kinds still parse; the `text` field is the documented payload.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct SystemBlock {
    #[serde(rename = "type", default, skip_serializing_if = "Option::is_none")]
    pub kind: Option<String>,
    #[serde(default)]
    pub text: String,
    /// Catch-all (e.g. `cache_control`).
    #[serde(flatten)]
    pub extra: Extra,
}

/// Extended thinking configuration.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(tag = "type", rename_all = "snake_case")]
pub enum ThinkingConfig {
    Adaptive {
        #[serde(flatten)]
        extra: Extra,
    },
    Enabled {
        #[serde(default)]
        budget_tokens: u32,
        #[serde(flatten)]
        extra: Extra,
    },
    Disabled {
        #[serde(flatten)]
        extra: Extra,
    },
}

/// `metadata` on a request.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct RequestMetadata {
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub user_id: Option<String>,
    /// Catch-all.
    #[serde(flatten)]
    pub extra: Extra,
}

// ---------------------------------------------------------------------------
// Tools
// ---------------------------------------------------------------------------

/// Tool definition. Custom tools are flat `{name, description, input_schema}`
/// with no `type` (or `type: "custom"`); hosted tools carry a dated `type`
/// (e.g. `web_search_20250305`) plus their own config fields in `extra`.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct AnthropicToolDefinition {
    #[serde(rename = "type", default, skip_serializing_if = "Option::is_none")]
    pub kind: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub name: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub description: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub input_schema: Option<Value>,
    /// Catch-all for hosted tool config (e.g. `max_uses`, `allowed_domains`).
    #[serde(flatten)]
    pub extra: Extra,
}

/// The parsed kind of a tool definition, used to reject hosted tools by name.
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub enum AnthropicToolKind {
    Custom,
    WebSearch,
    Computer,
    CodeExecution,
    TextEditor,
    Bash,
    McpConnector,
    Other(String),
}

impl AnthropicToolKind {
    /// Stable name for the kind (used in rejection messages).
    pub fn name(&self) -> &str {
        match self {
            Self::Custom => "custom",
            Self::WebSearch => "web_search",
            Self::Computer => "computer",
            Self::CodeExecution => "code_execution",
            Self::TextEditor => "text_editor",
            Self::Bash => "bash",
            Self::McpConnector => "mcp_connector",
            Self::Other(name) => name,
        }
    }
}

/// Hosted tool `type` stems, longest first so `code_execution` wins over any
/// shorter prefix match.
const HOSTED_TOOL_STEMS: &[&str] = &[
    "code_execution",
    "mcp_connector",
    "web_search",
    "text_editor",
    "computer",
    "bash",
];

impl AnthropicToolDefinition {
    /// Classify the tool: absent/`"custom"` is custom; a known hosted stem
    /// (optionally date-suffixed, e.g. `web_search_20250305`) maps to its kind;
    /// anything else is passed through as [`AnthropicToolKind::Other`].
    pub fn kind(&self) -> AnthropicToolKind {
        let Some(kind) = self.kind.as_deref() else {
            return AnthropicToolKind::Custom;
        };
        if kind == "custom" {
            return AnthropicToolKind::Custom;
        }
        for stem in HOSTED_TOOL_STEMS {
            if kind == *stem || kind.starts_with(&format!("{stem}_")) {
                return match *stem {
                    "web_search" => AnthropicToolKind::WebSearch,
                    "computer" => AnthropicToolKind::Computer,
                    "code_execution" => AnthropicToolKind::CodeExecution,
                    "text_editor" => AnthropicToolKind::TextEditor,
                    "bash" => AnthropicToolKind::Bash,
                    "mcp_connector" => AnthropicToolKind::McpConnector,
                    _ => unreachable!("stem list is exhaustive for matches"),
                };
            }
        }
        AnthropicToolKind::Other(kind.to_string())
    }
}

/// `tool_choice` on a request.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(tag = "type", rename_all = "snake_case")]
pub enum AnthropicToolChoice {
    Auto {
        #[serde(default, skip_serializing_if = "Option::is_none")]
        disable_parallel_tool_use: Option<bool>,
        #[serde(flatten)]
        extra: Extra,
    },
    Any {
        #[serde(default, skip_serializing_if = "Option::is_none")]
        disable_parallel_tool_use: Option<bool>,
        #[serde(flatten)]
        extra: Extra,
    },
    None {
        #[serde(flatten)]
        extra: Extra,
    },
    Tool {
        name: String,
        #[serde(default, skip_serializing_if = "Option::is_none")]
        disable_parallel_tool_use: Option<bool>,
        #[serde(flatten)]
        extra: Extra,
    },
}

// ---------------------------------------------------------------------------
// Content blocks
// ---------------------------------------------------------------------------

/// Typed content block inside `messages[].content` and response `content[]`.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(tag = "type", rename_all = "snake_case")]
pub enum ContentBlock {
    Text {
        #[serde(default)]
        text: String,
        /// Catch-all (e.g. `citations`, `cache_control`).
        #[serde(flatten)]
        extra: Extra,
    },
    Image {
        source: ImageSource,
        #[serde(flatten)]
        extra: Extra,
    },
    ToolUse {
        id: String,
        name: String,
        #[serde(default)]
        input: Value,
        #[serde(flatten)]
        extra: Extra,
    },
    ToolResult {
        tool_use_id: String,
        #[serde(default)]
        content: ToolResultContent,
        #[serde(default, skip_serializing_if = "Option::is_none")]
        is_error: Option<bool>,
        #[serde(flatten)]
        extra: Extra,
    },
    Thinking {
        #[serde(default)]
        thinking: String,
        #[serde(default)]
        signature: String,
        #[serde(flatten)]
        extra: Extra,
    },
    RedactedThinking {
        #[serde(default)]
        data: String,
        #[serde(flatten)]
        extra: Extra,
    },
    Document {
        source: DocumentSource,
        /// Catch-all (e.g. `title`, `context`, `citations`).
        #[serde(flatten)]
        extra: Extra,
    },
}

impl ContentBlock {
    /// Wire `type` name of the block.
    pub fn name(&self) -> &'static str {
        match self {
            Self::Text { .. } => "text",
            Self::Image { .. } => "image",
            Self::ToolUse { .. } => "tool_use",
            Self::ToolResult { .. } => "tool_result",
            Self::Thinking { .. } => "thinking",
            Self::RedactedThinking { .. } => "redacted_thinking",
            Self::Document { .. } => "document",
        }
    }
}

/// `tool_result.content`: a bare string or a list of text/image blocks.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(untagged)]
pub enum ToolResultContent {
    Text(String),
    Blocks(Vec<TextOrImageBlock>),
}

impl Default for ToolResultContent {
    fn default() -> Self {
        Self::Text(String::new())
    }
}

/// Text or image block, the shared payload of `tool_result.content` and
/// document `content` sources.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(tag = "type", rename_all = "snake_case")]
pub enum TextOrImageBlock {
    Text {
        #[serde(default)]
        text: String,
        #[serde(flatten)]
        extra: Extra,
    },
    Image {
        source: ImageSource,
        #[serde(flatten)]
        extra: Extra,
    },
}

/// Image `source`: inline base64 data or a URL.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(tag = "type", rename_all = "snake_case")]
pub enum ImageSource {
    Base64 {
        media_type: String,
        data: String,
        #[serde(flatten)]
        extra: Extra,
    },
    Url {
        url: String,
        #[serde(flatten)]
        extra: Extra,
    },
}

/// Document `source`: base64, URL, plain text, or nested content blocks.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(tag = "type", rename_all = "snake_case")]
pub enum DocumentSource {
    Base64 {
        media_type: String,
        data: String,
        #[serde(flatten)]
        extra: Extra,
    },
    Url {
        url: String,
        #[serde(flatten)]
        extra: Extra,
    },
    /// `type: "text"` (e.g. `media_type: "text/plain"`).
    #[serde(rename = "text")]
    PlainText {
        media_type: String,
        text: String,
        #[serde(flatten)]
        extra: Extra,
    },
    /// `type: "content"`: a list of text/image blocks.
    Content {
        #[serde(default)]
        content: Vec<TextOrImageBlock>,
        #[serde(flatten)]
        extra: Extra,
    },
}

// ---------------------------------------------------------------------------
// Response
// ---------------------------------------------------------------------------

fn default_message_type() -> String {
    "message".to_string()
}

/// Anthropic Messages API response body.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct AnthropicResponse {
    #[serde(rename = "type", default = "default_message_type")]
    pub kind: String,
    pub id: String,
    /// Always `"assistant"` on responses.
    #[serde(default)]
    pub role: String,
    #[serde(default)]
    pub content: Vec<ContentBlock>,
    #[serde(default)]
    pub model: String,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub stop_reason: Option<StopReason>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub stop_sequence: Option<String>,
    #[serde(default)]
    pub usage: Usage,
    /// Catch-all (e.g. `container`, `context_management`).
    #[serde(flatten)]
    pub extra: Extra,
}

/// Token usage on a response (cumulative in `message_delta`).
#[derive(Debug, Clone, PartialEq, Default, Serialize, Deserialize)]
pub struct Usage {
    #[serde(default)]
    pub input_tokens: u64,
    #[serde(default)]
    pub output_tokens: u64,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub cache_creation_input_tokens: Option<u64>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub cache_read_input_tokens: Option<u64>,
    /// Catch-all (e.g. `cache_creation` object, `server_tool_use`).
    #[serde(flatten)]
    pub extra: Extra,
}

/// Why the model stopped generating.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum StopReason {
    EndTurn,
    MaxTokens,
    StopSequence,
    ToolUse,
    PauseTurn,
    Refusal,
}

// ---------------------------------------------------------------------------
// Errors
// ---------------------------------------------------------------------------

/// Anthropic error type strings (see the API errors documentation).
pub mod error_types {
    pub const INVALID_REQUEST_ERROR: &str = "invalid_request_error";
    pub const AUTHENTICATION_ERROR: &str = "authentication_error";
    pub const PERMISSION_ERROR: &str = "permission_error";
    pub const NOT_FOUND_ERROR: &str = "not_found_error";
    pub const REQUEST_TOO_LARGE: &str = "request_too_large";
    pub const RATE_LIMIT_ERROR: &str = "rate_limit_error";
    pub const API_ERROR: &str = "api_error";
    pub const OVERLOADED_ERROR: &str = "overloaded_error";
}

fn default_error_type() -> String {
    "error".to_string()
}

/// The `error` payload: `{"type": "...", "message": "..."}`.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct ErrorBody {
    #[serde(rename = "type", default)]
    pub kind: String,
    #[serde(default)]
    pub message: String,
    /// Catch-all.
    #[serde(flatten)]
    pub extra: Extra,
}

/// Top-level error response: `{"type":"error","error":{...}}`.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct AnthropicErrorResponse {
    #[serde(rename = "type", default = "default_error_type")]
    pub kind: String,
    pub error: ErrorBody,
    /// Catch-all.
    #[serde(flatten)]
    pub extra: Extra,
}

// ---------------------------------------------------------------------------
// SSE events
// ---------------------------------------------------------------------------

/// One SSE event data payload (the `data:` JSON of an Anthropic stream).
///
/// Event order: `message_start` → per block (`content_block_start` →
/// `content_block_delta`* → `content_block_stop`) → `message_delta` →
/// `message_stop`, with `ping` allowed anywhere and `error` on post-200
/// failures.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(tag = "type", rename_all = "snake_case")]
pub enum AnthropicSseEvent {
    MessageStart {
        message: AnthropicResponse,
        #[serde(flatten)]
        extra: Extra,
    },
    ContentBlockStart {
        #[serde(default)]
        index: usize,
        content_block: ContentBlock,
        #[serde(flatten)]
        extra: Extra,
    },
    ContentBlockDelta {
        #[serde(default)]
        index: usize,
        delta: ContentBlockDelta,
        #[serde(flatten)]
        extra: Extra,
    },
    ContentBlockStop {
        #[serde(default)]
        index: usize,
        #[serde(flatten)]
        extra: Extra,
    },
    MessageDelta {
        delta: MessageDeltaBody,
        #[serde(default, skip_serializing_if = "Option::is_none")]
        usage: Option<Usage>,
        #[serde(flatten)]
        extra: Extra,
    },
    MessageStop {
        #[serde(flatten)]
        extra: Extra,
    },
    Ping {
        #[serde(flatten)]
        extra: Extra,
    },
    Error {
        error: ErrorBody,
        #[serde(flatten)]
        extra: Extra,
    },
}

impl AnthropicSseEvent {
    /// Wire event name (the SSE `event:` line value).
    pub fn name(&self) -> &'static str {
        match self {
            Self::MessageStart { .. } => "message_start",
            Self::ContentBlockStart { .. } => "content_block_start",
            Self::ContentBlockDelta { .. } => "content_block_delta",
            Self::ContentBlockStop { .. } => "content_block_stop",
            Self::MessageDelta { .. } => "message_delta",
            Self::MessageStop { .. } => "message_stop",
            Self::Ping { .. } => "ping",
            Self::Error { .. } => "error",
        }
    }
}

/// Streaming delta payloads for `content_block_delta`.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(tag = "type", rename_all = "snake_case")]
pub enum ContentBlockDelta {
    TextDelta {
        #[serde(default)]
        text: String,
        #[serde(flatten)]
        extra: Extra,
    },
    InputJsonDelta {
        #[serde(default)]
        partial_json: String,
        #[serde(flatten)]
        extra: Extra,
    },
    ThinkingDelta {
        #[serde(default)]
        thinking: String,
        #[serde(flatten)]
        extra: Extra,
    },
    SignatureDelta {
        #[serde(default)]
        signature: String,
        #[serde(flatten)]
        extra: Extra,
    },
}

impl ContentBlockDelta {
    /// Wire `type` name of the delta.
    pub fn name(&self) -> &'static str {
        match self {
            Self::TextDelta { .. } => "text_delta",
            Self::InputJsonDelta { .. } => "input_json_delta",
            Self::ThinkingDelta { .. } => "thinking_delta",
            Self::SignatureDelta { .. } => "signature_delta",
        }
    }
}

/// `message_delta.delta` body (terminal stop reason/sequence).
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct MessageDeltaBody {
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub stop_reason: Option<StopReason>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub stop_sequence: Option<String>,
    /// Catch-all.
    #[serde(flatten)]
    pub extra: Extra,
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    fn rt<T: Serialize + serde::de::DeserializeOwned + PartialEq + std::fmt::Debug>(
        value: &T,
    ) -> T {
        let wire = serde_json::to_value(value).unwrap();
        serde_json::from_value(wire).unwrap()
    }

    fn sample_response() -> AnthropicResponse {
        AnthropicResponse {
            kind: default_message_type(),
            id: "msg_01ABC".to_string(),
            role: "assistant".to_string(),
            content: vec![
                ContentBlock::Thinking {
                    thinking: "hmm".to_string(),
                    signature: "sig_1".to_string(),
                    extra: Extra::new(),
                },
                ContentBlock::Text {
                    text: "Hello!".to_string(),
                    extra: Extra::new(),
                },
                ContentBlock::ToolUse {
                    id: "toolu_01A".to_string(),
                    name: "get_weather".to_string(),
                    input: json!({"city": "Tokyo"}),
                    extra: Extra::new(),
                },
            ],
            model: "claude-sonnet-4-5".to_string(),
            stop_reason: Some(StopReason::ToolUse),
            stop_sequence: None,
            usage: Usage {
                input_tokens: 12,
                output_tokens: 34,
                cache_creation_input_tokens: Some(3),
                cache_read_input_tokens: Some(4),
                extra: Extra::new(),
            },
            extra: Extra::new(),
        }
    }

    #[test]
    fn request_minimal_round_trip() {
        let req: AnthropicRequest = serde_json::from_value(json!({
            "model": "claude-sonnet-4-5",
            "max_tokens": 1024,
            "messages": [{"role": "user", "content": "Hello"}]
        }))
        .unwrap();

        assert_eq!(req.messages.len(), 1);
        assert_eq!(req.messages[0].content, MessageContent::Text("Hello".into()));
        assert!(!req.stream);
        assert_eq!(rt(&req), req);
    }

    #[test]
    fn request_full_round_trip() {
        let req: AnthropicRequest = serde_json::from_value(json!({
            "model": "claude-opus-4-6",
            "max_tokens": 8192,
            "system": [{"type": "text", "text": "Be terse."}],
            "messages": [
                {"role": "user", "content": [
                    {"type": "text", "text": "What is this?"},
                    {"type": "image", "source": {
                        "type": "base64", "media_type": "image/png", "data": "aGk="
                    }},
                    {"type": "document", "source": {
                        "type": "url", "url": "https://example.com/doc.pdf"
                    }}
                ]},
                {"role": "assistant", "content": [
                    {"type": "thinking", "thinking": "consider", "signature": "sig"},
                    {"type": "redacted_thinking", "data": "encrypted"},
                    {"type": "tool_use", "id": "toolu_1", "name": "lookup", "input": {"q": "x"}},
                    {"type": "text", "text": "Checking."}
                ]},
                {"role": "user", "content": [
                    {"type": "tool_result", "tool_use_id": "toolu_1", "is_error": true,
                     "content": [{"type": "text", "text": "nope"}]}
                ]}
            ],
            "temperature": 0.5,
            "top_p": 0.9,
            "stop_sequences": ["END", "STOP"],
            "stream": true,
            "tools": [
                {"name": "lookup", "description": "Look things up",
                 "input_schema": {"type": "object", "properties": {}}},
                {"type": "web_search_20250305", "name": "web_search", "max_uses": 3}
            ],
            "tool_choice": {"type": "tool", "name": "lookup", "disable_parallel_tool_use": true},
            "thinking": {"type": "enabled", "budget_tokens": 2048},
            "metadata": {"user_id": "user_123"},
            "future_field": {"nested": true}
        }))
        .unwrap();

        // Spot checks on the translated-in-place structures.
        assert_eq!(
            req.system,
            Some(SystemPrompt::Blocks(vec![SystemBlock {
                kind: Some("text".to_string()),
                text: "Be terse.".to_string(),
                extra: Extra::new(),
            }]))
        );
        assert_eq!(req.thinking, Some(ThinkingConfig::Enabled {
            budget_tokens: 2048,
            extra: Extra::new(),
        }));
        assert_eq!(req.metadata, Some(RequestMetadata {
            user_id: Some("user_123".to_string()),
            extra: Extra::new(),
        }));
        assert_eq!(req.tools.len(), 2);
        assert_eq!(req.tools[1].extra.get("max_uses"), Some(&json!(3)));

        // Unknown top-level field survives in extra.
        assert_eq!(req.extra.get("future_field"), Some(&json!({"nested": true})));

        let back = rt(&req);
        assert_eq!(back, req);
        // The unknown field must survive the round trip verbatim.
        assert_eq!(back.extra.get("future_field"), Some(&json!({"nested": true})));
    }

    #[test]
    fn system_accepts_string_and_blocks() {
        let s: AnthropicRequest = serde_json::from_value(json!({
            "model": "m", "max_tokens": 1,
            "messages": [], "system": "You are helpful."
        }))
        .unwrap();
        assert_eq!(s.system, Some(SystemPrompt::Text("You are helpful.".into())));

        let b: AnthropicRequest = serde_json::from_value(json!({
            "model": "m", "max_tokens": 1,
            "messages": [], "system": [{"type": "text", "text": "Be brief."}]
        }))
        .unwrap();
        assert!(matches!(b.system, Some(SystemPrompt::Blocks(_))));

        // Untyped system blocks (future kinds) must still parse.
        let u: AnthropicRequest = serde_json::from_value(json!({
            "model": "m", "max_tokens": 1,
            "messages": [], "system": [{"type": "future", "text": "hmm"}]
        }))
        .unwrap();
        assert!(matches!(u.system, Some(SystemPrompt::Blocks(_))));
    }

    #[test]
    fn tool_choice_modes_round_trip() {
        for body in [
            json!({"type": "auto"}),
            json!({"type": "any", "disable_parallel_tool_use": true}),
            json!({"type": "none"}),
            json!({"type": "tool", "name": "lookup"}),
        ] {
            let choice: AnthropicToolChoice = serde_json::from_value(body).unwrap();
            assert_eq!(rt(&choice), choice);
        }
    }

    #[test]
    fn thinking_config_round_trip() {
        for body in [
            json!({"type": "adaptive"}),
            json!({"type": "enabled", "budget_tokens": 10240}),
            json!({"type": "disabled"}),
        ] {
            let cfg: ThinkingConfig = serde_json::from_value(body).unwrap();
            assert_eq!(rt(&cfg), cfg);
        }
    }

    #[test]
    fn tool_kind_classification() {
        let tool = |kind: Option<&str>| AnthropicToolDefinition {
            kind: kind.map(str::to_string),
            name: Some("t".to_string()),
            description: None,
            input_schema: None,
            extra: Extra::new(),
        };

        assert_eq!(tool(None).kind(), AnthropicToolKind::Custom);
        assert_eq!(tool(Some("custom")).kind(), AnthropicToolKind::Custom);
        assert_eq!(
            tool(Some("web_search_20250305")).kind(),
            AnthropicToolKind::WebSearch
        );
        assert_eq!(tool(Some("web_search")).kind(), AnthropicToolKind::WebSearch);
        assert_eq!(
            tool(Some("computer_20250124")).kind(),
            AnthropicToolKind::Computer
        );
        assert_eq!(
            tool(Some("code_execution_20250522")).kind(),
            AnthropicToolKind::CodeExecution
        );
        assert_eq!(
            tool(Some("text_editor_20250124")).kind(),
            AnthropicToolKind::TextEditor
        );
        assert_eq!(tool(Some("bash_20250124")).kind(), AnthropicToolKind::Bash);
        assert_eq!(
            tool(Some("mcp_connector_20250811")).kind(),
            AnthropicToolKind::McpConnector
        );
        assert_eq!(
            tool(Some("bananas")).kind(),
            AnthropicToolKind::Other("bananas".to_string())
        );

        // Names are stable for rejection messages.
        assert_eq!(AnthropicToolKind::WebSearch.name(), "web_search");
        assert_eq!(AnthropicToolKind::McpConnector.name(), "mcp_connector");
        assert_eq!(AnthropicToolKind::Other("x".into()).name(), "x");
    }

    #[test]
    fn document_block_sources_parse() {
        let parse = |source: Value| -> ContentBlock {
            serde_json::from_value(json!({"type": "document", "source": source})).unwrap()
        };

        assert_eq!(
            parse(json!({"type": "base64", "media_type": "application/pdf", "data": "AAECAA=="})),
            ContentBlock::Document {
                source: DocumentSource::Base64 {
                    media_type: "application/pdf".to_string(),
                    data: "AAECAA==".to_string(),
                    extra: Extra::new(),
                },
                extra: Extra::new(),
            }
        );
        assert_eq!(
            parse(json!({"type": "url", "url": "https://example.com/a.pdf"})),
            ContentBlock::Document {
                source: DocumentSource::Url {
                    url: "https://example.com/a.pdf".to_string(),
                    extra: Extra::new(),
                },
                extra: Extra::new(),
            }
        );
        assert_eq!(
            parse(json!({"type": "text", "media_type": "text/plain", "text": "hello"})),
            ContentBlock::Document {
                source: DocumentSource::PlainText {
                    media_type: "text/plain".to_string(),
                    text: "hello".to_string(),
                    extra: Extra::new(),
                },
                extra: Extra::new(),
            }
        );
        assert_eq!(
            parse(json!({"type": "content", "content": [{"type": "text", "text": "part"}]})),
            ContentBlock::Document {
                source: DocumentSource::Content {
                    content: vec![TextOrImageBlock::Text {
                        text: "part".to_string(),
                        extra: Extra::new(),
                    }],
                    extra: Extra::new(),
                },
                extra: Extra::new(),
            }
        );
    }

    #[test]
    fn content_block_names() {
        let blocks = [
            ContentBlock::Text { text: String::new(), extra: Extra::new() },
            ContentBlock::Image {
                source: ImageSource::Url { url: "https://x/i.png".into(), extra: Extra::new() },
                extra: Extra::new(),
            },
            ContentBlock::ToolUse { id: "i".into(), name: "n".into(), input: json!({}), extra: Extra::new() },
            ContentBlock::ToolResult {
                tool_use_id: "i".into(),
                content: ToolResultContent::default(),
                is_error: None,
                extra: Extra::new(),
            },
            ContentBlock::Thinking { thinking: String::new(), signature: String::new(), extra: Extra::new() },
            ContentBlock::RedactedThinking { data: String::new(), extra: Extra::new() },
            ContentBlock::Document {
                source: DocumentSource::Url { url: "https://x/d.pdf".into(), extra: Extra::new() },
                extra: Extra::new(),
            },
        ];
        let expected = [
            "text",
            "image",
            "tool_use",
            "tool_result",
            "thinking",
            "redacted_thinking",
            "document",
        ];
        for (block, name) in blocks.iter().zip(expected) {
            assert_eq!(block.name(), name);
            assert_eq!(rt(block), *block);
        }
    }

    #[test]
    fn response_round_trip() {
        let resp = sample_response();
        let back = rt(&resp);
        assert_eq!(back, resp);

        let wire = serde_json::to_value(&resp).unwrap();
        assert_eq!(wire["type"], "message");
        assert_eq!(wire["stop_reason"], "tool_use");
        assert_eq!(wire["usage"]["cache_read_input_tokens"], 4);
    }

    #[test]
    fn response_minimal_defaults() {
        let resp: AnthropicResponse = serde_json::from_value(json!({
            "id": "msg_1",
            "content": [{"type": "text", "text": "hi"}],
            "usage": {"input_tokens": 1, "output_tokens": 2}
        }))
        .unwrap();

        assert_eq!(resp.kind, "message");
        assert_eq!(resp.role, "");
        assert_eq!(resp.stop_reason, None);
        assert_eq!(resp.usage.input_tokens, 1);
        assert_eq!(resp.usage.output_tokens, 2);
    }

    #[test]
    fn stop_reason_all_variants_round_trip() {
        for (wire, reason) in [
            (json!("end_turn"), StopReason::EndTurn),
            (json!("max_tokens"), StopReason::MaxTokens),
            (json!("stop_sequence"), StopReason::StopSequence),
            (json!("tool_use"), StopReason::ToolUse),
            (json!("pause_turn"), StopReason::PauseTurn),
            (json!("refusal"), StopReason::Refusal),
        ] {
            assert_eq!(serde_json::from_value::<StopReason>(wire.clone()).unwrap(), reason);
            assert_eq!(serde_json::to_value(reason).unwrap(), wire);
        }
    }

    #[test]
    fn error_response_round_trip() {
        let body = json!({
            "type": "error",
            "error": {"type": "rate_limit_error", "message": "Slow down."}
        });
        let resp: AnthropicErrorResponse = serde_json::from_value(body).unwrap();
        assert_eq!(resp.error.kind, error_types::RATE_LIMIT_ERROR);
        assert_eq!(rt(&resp), resp);

        let inner: ErrorBody =
            serde_json::from_value(json!({"type": "overloaded_error", "message": "busy"})).unwrap();
        assert_eq!(inner.kind, error_types::OVERLOADED_ERROR);
        assert_eq!(rt(&inner), inner);
    }

    #[test]
    fn sse_message_start_round_trip() {
        let event = AnthropicSseEvent::MessageStart {
            message: sample_response(),
            extra: Extra::new(),
        };
        let back = rt(&event);
        assert_eq!(back, event);
        assert_eq!(back.name(), "message_start");
        assert_eq!(serde_json::to_value(&event).unwrap()["type"], "message_start");
    }

    #[test]
    fn sse_content_block_start_round_trip() {
        for block in [
            ContentBlock::Text { text: "Hello".into(), extra: Extra::new() },
            ContentBlock::ToolUse {
                id: "toolu_1".into(),
                name: "lookup".into(),
                input: json!({"q": 1}),
                extra: Extra::new(),
            },
        ] {
            let event = AnthropicSseEvent::ContentBlockStart {
                index: 0,
                content_block: block,
                extra: Extra::new(),
            };
            assert_eq!(rt(&event), event);
            assert_eq!(event.name(), "content_block_start");
        }
    }

    #[test]
    fn sse_content_block_delta_all_variants_round_trip() {
        let deltas = [
            ContentBlockDelta::TextDelta { text: "Hello".into(), extra: Extra::new() },
            ContentBlockDelta::InputJsonDelta { partial_json: "{\"city\":".into(), extra: Extra::new() },
            ContentBlockDelta::ThinkingDelta { thinking: "deep".into(), extra: Extra::new() },
            ContentBlockDelta::SignatureDelta { signature: "sig_abc".into(), extra: Extra::new() },
        ];
        let names = ["text_delta", "input_json_delta", "thinking_delta", "signature_delta"];
        for (delta, name) in deltas.iter().zip(names) {
            assert_eq!(delta.name(), name);
            assert_eq!(rt(delta), *delta);

            let event = AnthropicSseEvent::ContentBlockDelta {
                index: 1,
                delta: delta.clone(),
                extra: Extra::new(),
            };
            let back = rt(&event);
            assert_eq!(back, event);
            assert_eq!(event.name(), "content_block_delta");
        }
    }

    #[test]
    fn sse_content_block_stop_round_trip() {
        let event = AnthropicSseEvent::ContentBlockStop { index: 2, extra: Extra::new() };
        assert_eq!(rt(&event), event);
        assert_eq!(event.name(), "content_block_stop");
    }

    #[test]
    fn sse_message_delta_round_trip() {
        let event = AnthropicSseEvent::MessageDelta {
            delta: MessageDeltaBody {
                stop_reason: Some(StopReason::EndTurn),
                stop_sequence: Some("\n\nHuman:".to_string()),
                extra: Extra::new(),
            },
            usage: Some(Usage {
                input_tokens: 10,
                output_tokens: 42,
                cache_creation_input_tokens: None,
                cache_read_input_tokens: Some(7),
                extra: Extra::new(),
            }),
            extra: Extra::new(),
        };
        let back = rt(&event);
        assert_eq!(back, event);
        assert_eq!(event.name(), "message_delta");

        let wire = serde_json::to_value(&event).unwrap();
        assert_eq!(wire["delta"]["stop_reason"], "end_turn");
        assert_eq!(wire["usage"]["output_tokens"], 42);
    }

    #[test]
    fn sse_message_stop_round_trip() {
        let event = AnthropicSseEvent::MessageStop { extra: Extra::new() };
        assert_eq!(rt(&event), event);
        assert_eq!(event.name(), "message_stop");
        assert_eq!(serde_json::to_value(&event).unwrap()["type"], "message_stop");
    }

    #[test]
    fn sse_ping_round_trip() {
        let event = AnthropicSseEvent::Ping { extra: Extra::new() };
        assert_eq!(rt(&event), event);
        assert_eq!(event.name(), "ping");
    }

    #[test]
    fn sse_error_round_trip() {
        let event = AnthropicSseEvent::Error {
            error: ErrorBody {
                kind: error_types::OVERLOADED_ERROR.to_string(),
                message: "Overloaded".to_string(),
                extra: Extra::new(),
            },
            extra: Extra::new(),
        };
        assert_eq!(rt(&event), event);
        assert_eq!(event.name(), "error");

        let wire = serde_json::to_value(&event).unwrap();
        assert_eq!(wire["type"], "error");
        assert_eq!(wire["error"]["type"], "overloaded_error");
    }
}
