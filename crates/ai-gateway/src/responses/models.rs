//! Typed models for the OpenAI Responses API (`/v1/responses` front door).
//!
//! These are gateway-owned protocol models, distinct from the Codex client
//! types in [`crate::codex`]. Known fields are explicit; unknown fields are
//! captured in `extra` maps via `#[serde(flatten)]` and passed through,
//! mirroring the conventions of [`crate::models::openai`].

use std::collections::HashMap;

use serde::{Deserialize, Serialize};

// ---------------------------------------------------------------------------
// Request
// ---------------------------------------------------------------------------

/// Responses API request body.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ResponsesRequest {
    pub model: String,
    #[serde(default)]
    pub input: ResponsesInput,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub instructions: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub previous_response_id: Option<String>,
    #[serde(default)]
    pub store: bool,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub metadata: Option<serde_json::Map<String, serde_json::Value>>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub temperature: Option<f32>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub top_p: Option<f32>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub max_output_tokens: Option<u32>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub truncation: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub parallel_tool_calls: Option<bool>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub reasoning: Option<ReasoningConfig>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub text: Option<TextConfig>,
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub tools: Vec<ToolDefinition>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub tool_choice: Option<ToolChoice>,
    #[serde(default)]
    pub stream: bool,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub stream_options: Option<StreamOptions>,
    /// Catch-all for anything not modelled above.
    #[serde(flatten)]
    pub extra: serde_json::Map<String, serde_json::Value>,
}

/// `input` accepts a bare string or a list of input items.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(untagged)]
pub enum ResponsesInput {
    Text(String),
    Items(Vec<InputItem>),
}

impl Default for ResponsesInput {
    fn default() -> Self {
        Self::Items(Vec::new())
    }
}

/// Reasoning configuration on a request.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ReasoningConfig {
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub effort: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub summary: Option<ReasoningSummaryConfig>,
    /// Catch-all (e.g. `generate_summary` on older shapes).
    #[serde(flatten)]
    pub extra: serde_json::Map<String, serde_json::Value>,
}

/// `reasoning.summary` configuration.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ReasoningSummaryConfig {
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub generate_summary: Option<String>,
    #[serde(flatten)]
    pub extra: serde_json::Map<String, serde_json::Value>,
}

/// `text` configuration on a request.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TextConfig {
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub verbosity: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub format: Option<TextFormat>,
    #[serde(flatten)]
    pub extra: serde_json::Map<String, serde_json::Value>,
}

/// `text.format`: plain text, JSON mode, or a JSON schema.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(tag = "type")]
pub enum TextFormat {
    #[serde(rename = "text")]
    Text,
    #[serde(rename = "json_object")]
    JsonObject,
    #[serde(rename = "json_schema")]
    JsonSchema {
        #[serde(default, skip_serializing_if = "Option::is_none")]
        name: Option<String>,
        #[serde(default, skip_serializing_if = "Option::is_none")]
        description: Option<String>,
        #[serde(default, skip_serializing_if = "Option::is_none")]
        schema: Option<serde_json::Value>,
        #[serde(default, skip_serializing_if = "Option::is_none")]
        strict: Option<bool>,
    },
}

/// Function tool definition (server-side tool types pass through via `extra`).
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ToolDefinition {
    #[serde(default)]
    pub r#type: String,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub name: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub description: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub parameters: Option<serde_json::Value>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub strict: Option<bool>,
    /// Catch-all for `function`, custom tool fields, hosted tool config, etc.
    #[serde(flatten)]
    pub extra: serde_json::Map<String, serde_json::Value>,
}

/// `tool_choice`: `"auto"`, `"none"`, `"required"`, or a named function pick.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(untagged)]
pub enum ToolChoice {
    Mode(String),
    Function {
        r#type: String,
        name: String,
    },
}

/// `stream_options` (e.g. `include_usage`).
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct StreamOptions {
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub include_usage: Option<bool>,
    #[serde(flatten)]
    pub extra: serde_json::Map<String, serde_json::Value>,
}

// ---------------------------------------------------------------------------
// Input items
// ---------------------------------------------------------------------------

/// One element of the `input` item list.
///
/// Untagged: bare `{role, content}` easy messages parse first; anything with
/// an explicit `type` discriminator falls through to the tagged inner enum.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(untagged)]
pub enum InputItem {
    Easy(EasyInputMessage),
    Typed(TypedInputItem),
}

/// Input items that carry an explicit `type` discriminator.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(tag = "type", rename_all = "snake_case")]
pub enum TypedInputItem {
    Message(Message),
    FunctionCall(FunctionCall),
    FunctionCallOutput(FunctionCallOutput),
    Reasoning(ReasoningItem),
    ItemReference(ItemReference),
}

/// Shorthand input message: content may be a string or a parts array.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct EasyInputMessage {
    #[serde(default)]
    pub content: EasyInputContent,
    #[serde(default)]
    pub role: String,
    /// Optional literal phase marker; pass-through only.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub phase: Option<String>,
    #[serde(flatten)]
    pub extra: serde_json::Map<String, serde_json::Value>,
}

/// Content of an easy input message: plain string or content-part array.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(untagged)]
pub enum EasyInputContent {
    Text(String),
    Parts(Vec<InputContentPart>),
}

impl Default for EasyInputContent {
    fn default() -> Self {
        Self::Text(String::new())
    }
}

/// Fully-typed input message with a content parts array.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Message {
    #[serde(default)]
    pub content: Vec<InputContentPart>,
    #[serde(default)]
    pub role: String,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub status: Option<String>,
    #[serde(flatten)]
    pub extra: serde_json::Map<String, serde_json::Value>,
}

/// A `function_call` input item.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct FunctionCall {
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub id: Option<String>,
    #[serde(default)]
    pub call_id: String,
    #[serde(default)]
    pub name: String,
    #[serde(default)]
    pub arguments: String,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub status: Option<String>,
    #[serde(flatten)]
    pub extra: serde_json::Map<String, serde_json::Value>,
}

/// A `function_call_output` input item.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct FunctionCallOutput {
    #[serde(default)]
    pub call_id: String,
    #[serde(default)]
    pub output: FunctionCallOutputContent,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub id: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub status: Option<String>,
    #[serde(flatten)]
    pub extra: serde_json::Map<String, serde_json::Value>,
}

/// `function_call_output.output`: plain string or content parts.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(untagged)]
pub enum FunctionCallOutputContent {
    Text(String),
    Parts(Vec<OutputContentPart>),
}

impl Default for FunctionCallOutputContent {
    fn default() -> Self {
        Self::Text(String::new())
    }
}

/// A `reasoning` input item.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ReasoningItem {
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub id: Option<String>,
    #[serde(default)]
    pub summary: Vec<ReasoningSummaryPart>,
    #[serde(flatten)]
    pub extra: serde_json::Map<String, serde_json::Value>,
}

/// One reasoning summary entry (request-side: bare text summaries).
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ReasoningSummaryPart {
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub r#type: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub text: Option<String>,
    #[serde(flatten)]
    pub extra: serde_json::Map<String, serde_json::Value>,
}

/// An `item_reference` input item.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ItemReference {
    #[serde(default)]
    pub id: String,
    #[serde(flatten)]
    pub extra: serde_json::Map<String, serde_json::Value>,
}

// ---------------------------------------------------------------------------
// Content parts
// ---------------------------------------------------------------------------

/// A request-side content part. Untagged for flexibility across providers.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(untagged)]
pub enum InputContentPart {
    Text(InputText),
    Image(InputImage),
    Other(serde_json::Value),
}

/// `{"type":"input_text","text":"..."}`.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct InputText {
    #[serde(default = "default_input_text_type")]
    pub r#type: String,
    #[serde(default)]
    pub text: String,
}

fn default_input_text_type() -> String {
    "input_text".to_string()
}

/// `{"type":"input_image","image_url":"...","detail":"..."}`.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct InputImage {
    #[serde(default = "default_input_image_type")]
    pub r#type: String,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub image_url: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub detail: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub file_id: Option<String>,
}

fn default_input_image_type() -> String {
    "input_image".to_string()
}

// ---------------------------------------------------------------------------
// Response object
// ---------------------------------------------------------------------------

/// The Responses API response object (`object: "response"`).
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ResponseObject {
    #[serde(default)]
    pub id: String,
    #[serde(default = "default_response_object_kind")]
    pub object: String,
    #[serde(default)]
    pub created_at: i64,
    #[serde(default = "default_response_status")]
    pub status: String,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub error: Option<ResponseError>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub incomplete_details: Option<IncompleteDetails>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub instructions: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub metadata: Option<serde_json::Map<String, serde_json::Value>>,
    #[serde(default)]
    pub model: String,
    #[serde(default)]
    pub output: Vec<OutputItem>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub parallel_tool_calls: Option<bool>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub previous_response_id: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub reasoning: Option<ReasoningConfig>,
    #[serde(default)]
    pub store: bool,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub temperature: Option<f32>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub text: Option<TextConfig>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub tool_choice: Option<ToolChoice>,
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub tools: Vec<ToolDefinition>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub top_p: Option<f32>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub truncation: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub usage: Option<ResponsesUsage>,
    /// Catch-all for `service_tier`, `user_id`, etc.
    #[serde(flatten)]
    pub extra: serde_json::Map<String, serde_json::Value>,
}

fn default_response_object_kind() -> String {
    "response".to_string()
}

fn default_response_status() -> String {
    "completed".to_string()
}

/// Error block on a failed response.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ResponseError {
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub code: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub message: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub param: Option<String>,
    #[serde(flatten)]
    pub extra: serde_json::Map<String, serde_json::Value>,
}

/// Why a response ended `incomplete`.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct IncompleteDetails {
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub reason: Option<String>,
}

/// Token usage on a response.
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct ResponsesUsage {
    #[serde(default)]
    pub input_tokens: u64,
    #[serde(default)]
    pub output_tokens: u64,
    #[serde(default)]
    pub total_tokens: u64,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub input_tokens_details: Option<InputTokensDetails>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub output_tokens_details: Option<OutputTokensDetails>,
    #[serde(flatten)]
    pub extra: serde_json::Map<String, serde_json::Value>,
}

/// Breakdown of input token billing.
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct InputTokensDetails {
    #[serde(default)]
    pub cached_tokens: u64,
    #[serde(flatten)]
    pub extra: serde_json::Map<String, serde_json::Value>,
}

/// Breakdown of output token billing.
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct OutputTokensDetails {
    #[serde(default)]
    pub reasoning_tokens: u64,
    #[serde(flatten)]
    pub extra: serde_json::Map<String, serde_json::Value>,
}

// ---------------------------------------------------------------------------
// Output items
// ---------------------------------------------------------------------------

/// One element of `response.output`, discriminated by its `type` field.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(tag = "type", rename_all = "snake_case")]
pub enum OutputItem {
    Message(OutputMessage),
    FunctionCall(OutputFunctionCall),
    Reasoning(OutputReasoning),
}

/// A `message` output item.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct OutputMessage {
    #[serde(default)]
    pub id: String,
    #[serde(default)]
    pub role: String,
    #[serde(default)]
    pub status: String,
    #[serde(default)]
    pub content: Vec<OutputContentPart>,
    #[serde(flatten)]
    pub extra: serde_json::Map<String, serde_json::Value>,
}

/// A content part of an output message.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(tag = "type", rename_all = "snake_case")]
pub enum OutputContentPart {
    OutputText {
        #[serde(default)]
        text: String,
        #[serde(default, skip_serializing_if = "Option::is_none")]
        annotations: Option<Vec<serde_json::Value>>,
    },
    Refusal {
        #[serde(default)]
        refusal: String,
    },
}

/// A `function_call` output item.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct OutputFunctionCall {
    #[serde(default)]
    pub id: String,
    #[serde(default)]
    pub call_id: String,
    #[serde(default)]
    pub name: String,
    #[serde(default)]
    pub arguments: String,
    #[serde(default)]
    pub status: String,
    #[serde(flatten)]
    pub extra: serde_json::Map<String, serde_json::Value>,
}

/// A `reasoning` output item.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct OutputReasoning {
    #[serde(default)]
    pub id: String,
    #[serde(default)]
    pub summary: Vec<ReasoningSummaryPart>,
    #[serde(flatten)]
    pub extra: serde_json::Map<String, serde_json::Value>,
}

// ---------------------------------------------------------------------------
// SSE events (gateway-owned)
// ---------------------------------------------------------------------------

/// Typed Responses API SSE event, tagged by the wire `type` field.
///
/// Gateway-owned and distinct from [`crate::codex::sse::ResponsesEvent`]:
/// that enum is a lossy view for the outbound Codex client, while this one
/// mirrors the full front-door wire contract the gateway must emit.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(tag = "type")]
pub enum ResponsesSseEvent {
    // Lifecycle
    #[serde(rename = "response.created")]
    Created {
        sequence_number: u64,
        response: ResponseObject,
    },
    #[serde(rename = "response.in_progress")]
    InProgress {
        sequence_number: u64,
        response: ResponseObject,
    },
    #[serde(rename = "response.queued")]
    Queued {
        sequence_number: u64,
        response: ResponseObject,
    },
    #[serde(rename = "response.completed")]
    Completed {
        sequence_number: u64,
        response: ResponseObject,
    },
    #[serde(rename = "response.failed")]
    Failed {
        sequence_number: u64,
        response: ResponseObject,
    },
    #[serde(rename = "response.incomplete")]
    Incomplete {
        sequence_number: u64,
        response: ResponseObject,
    },
    // Items
    #[serde(rename = "response.output_item.added")]
    OutputItemAdded {
        sequence_number: u64,
        output_index: u64,
        item: OutputItem,
    },
    #[serde(rename = "response.output_item.done")]
    OutputItemDone {
        sequence_number: u64,
        output_index: u64,
        item: OutputItem,
    },
    // Content parts
    #[serde(rename = "response.content_part.added")]
    ContentPartAdded {
        sequence_number: u64,
        item_id: String,
        output_index: u64,
        content_index: u64,
        part: OutputContentPart,
    },
    #[serde(rename = "response.content_part.done")]
    ContentPartDone {
        sequence_number: u64,
        item_id: String,
        output_index: u64,
        content_index: u64,
        part: OutputContentPart,
    },
    // Text
    #[serde(rename = "response.output_text.delta")]
    OutputTextDelta {
        sequence_number: u64,
        item_id: String,
        output_index: u64,
        content_index: u64,
        delta: String,
    },
    #[serde(rename = "response.output_text.done")]
    OutputTextDone {
        sequence_number: u64,
        item_id: String,
        output_index: u64,
        content_index: u64,
        text: String,
    },
    // Refusal
    #[serde(rename = "response.refusal.delta")]
    RefusalDelta {
        sequence_number: u64,
        item_id: String,
        output_index: u64,
        content_index: u64,
        delta: String,
    },
    #[serde(rename = "response.refusal.done")]
    RefusalDone {
        sequence_number: u64,
        item_id: String,
        output_index: u64,
        content_index: u64,
        refusal: String,
    },
    // Function calls
    #[serde(rename = "response.function_call_arguments.delta")]
    FunctionCallArgumentsDelta {
        sequence_number: u64,
        item_id: String,
        output_index: u64,
        delta: String,
    },
    #[serde(rename = "response.function_call_arguments.done")]
    FunctionCallArgumentsDone {
        sequence_number: u64,
        item_id: String,
        output_index: u64,
        arguments: String,
    },
    // Reasoning
    #[serde(rename = "response.reasoning_summary_part.added")]
    ReasoningSummaryPartAdded {
        sequence_number: u64,
        item_id: String,
        output_index: u64,
        summary_index: u64,
        part: ReasoningSummaryPart,
    },
    #[serde(rename = "response.reasoning_summary_text.delta")]
    ReasoningSummaryTextDelta {
        sequence_number: u64,
        item_id: String,
        output_index: u64,
        summary_index: u64,
        delta: String,
    },
    #[serde(rename = "response.reasoning_text.delta")]
    ReasoningTextDelta {
        sequence_number: u64,
        item_id: String,
        output_index: u64,
        delta: String,
    },
}

// ---------------------------------------------------------------------------
// Translation errors
// ---------------------------------------------------------------------------

/// A request feature the gateway cannot translate to the target protocol.
///
/// Sanitized for API surfaces: never carries raw provider payloads.
#[derive(Debug, Clone, thiserror::Error)]
pub enum ResponsesTranslationError {
    #[error("unsupported Responses feature: field `{field}` cannot be translated")]
    UnsupportedField { field: &'static str },
    #[error("unsupported Responses feature: `{feature}` is not supported")]
    UnsupportedFeature { feature: &'static str },
    #[error("invalid request: {reason}")]
    InvalidRequest { reason: String },
}

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

impl ResponsesRequest {
    /// Flatten `metadata` into owned key/value pairs (UI/logging helper).
    pub fn metadata_pairs(&self) -> HashMap<String, String> {
        self.metadata
            .as_ref()
            .map(|m| {
                m.iter()
                    .filter_map(|(k, v)| v.as_str().map(|s| (k.clone(), s.to_string())))
                    .collect()
            })
            .unwrap_or_default()
    }
}
