use serde::{Deserialize, Serialize};

/// OpenAI chat completion request
/// Known fields are explicit; everything else is captured in `extra` and passed through.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct OpenAIRequest {
    pub model: String,
    pub messages: Vec<Message>,
    #[serde(default)]
    pub stream: bool,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub temperature: Option<f32>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub max_tokens: Option<u32>,
    /// Catch-all for tools, tool_choice, response_format, top_p, etc.
    #[serde(flatten)]
    pub extra: serde_json::Map<String, serde_json::Value>,
}

/// Message in a chat completion request/response
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Message {
    pub role: String,
    #[serde(default)]
    pub content: serde_json::Value,
    /// Catch-all for tool_calls, tool_call_id, name, function_call, etc.
    #[serde(flatten)]
    pub extra: serde_json::Map<String, serde_json::Value>,
}

impl Message {
    /// Extract content as a plain text string.
    /// Handles both `"string"` and `[{"type":"text","text":"..."},...]` formats.
    pub fn content_as_text(&self) -> String {
        match &self.content {
            serde_json::Value::String(s) => s.clone(),
            serde_json::Value::Array(parts) => parts
                .iter()
                .filter_map(|p| {
                    if p.get("type").and_then(|t| t.as_str()) == Some("text") {
                        p.get("text")
                            .and_then(|t| t.as_str())
                            .map(|s| s.to_string())
                    } else {
                        None
                    }
                })
                .collect::<Vec<_>>()
                .join(""),
            serde_json::Value::Null => String::new(),
            other => other.to_string(),
        }
    }
}

/// OpenAI chat completion response
///
/// `id`, `object`, and `created` are optional to tolerate non-standard providers
/// (e.g. Nano-GPT) that return valid choices/usage but omit envelope metadata.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct OpenAIResponse {
    #[serde(default)]
    pub id: String,
    #[serde(default)]
    pub object: String,
    #[serde(default)]
    pub created: i64,
    #[serde(default)]
    pub model: String,
    #[serde(default)]
    pub choices: Vec<Choice>,
    #[serde(default)]
    pub usage: Usage,
    /// Catch-all for system_fingerprint, service_tier, etc.
    #[serde(flatten)]
    pub extra: serde_json::Map<String, serde_json::Value>,
}

/// Choice in a chat completion response
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Choice {
    #[serde(default)]
    pub index: u32,
    pub message: Message,
    pub finish_reason: Option<String>,
    /// Catch-all for logprobs, etc.
    #[serde(flatten)]
    pub extra: serde_json::Map<String, serde_json::Value>,
}

/// Token usage information
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct Usage {
    #[serde(default)]
    pub prompt_tokens: u32,
    #[serde(default)]
    pub completion_tokens: u32,
    #[serde(default)]
    pub total_tokens: u32,
    /// Catch-all for prompt_tokens_details, completion_tokens_details, etc.
    #[serde(flatten)]
    pub extra: serde_json::Map<String, serde_json::Value>,
}
