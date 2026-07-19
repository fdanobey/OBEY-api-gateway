#[allow(dead_code)]
pub mod bedrock;
#[allow(dead_code)]
pub mod groq;
#[allow(dead_code)]
pub mod lmstudio;
#[allow(dead_code)]
pub mod nvidia_nim;
#[allow(dead_code)]
pub mod ollama;
#[allow(dead_code)]
pub mod openai_compatible;
#[allow(dead_code)]
pub mod together;
#[allow(dead_code)]
pub mod vllm;

use async_trait::async_trait;
use futures::Stream;
use serde::{Deserialize, Serialize};
use std::pin::Pin;

use crate::error::GatewayError;
use crate::models::openai::{OpenAIRequest, OpenAIResponse};

/// Server-Sent Event for streaming responses
#[derive(Debug, Clone, Serialize, Deserialize)]
#[allow(dead_code)]
pub struct SSEEvent {
    /// Event type (e.g., "message", "error", "done")
    #[serde(skip_serializing_if = "Option::is_none")]
    pub event: Option<String>,
    /// Event data payload
    pub data: String,
    /// Optional event ID for client tracking
    #[serde(skip_serializing_if = "Option::is_none")]
    pub id: Option<String>,
}

#[allow(dead_code)]
impl SSEEvent {
    /// Create a new SSE event with data
    pub fn new(data: String) -> Self {
        Self {
            event: None,
            data,
            id: None,
        }
    }

    /// Create a comment event (for failover notifications)
    pub fn comment(comment: String) -> Self {
        Self {
            event: Some("comment".to_string()),
            data: comment,
            id: None,
        }
    }

    /// Create an error event
    pub fn error(message: &str) -> Self {
        Self {
            event: Some("error".to_string()),
            data: message.to_string(),
            id: None,
        }
    }

    /// Format as SSE protocol string
    pub fn to_sse_string(&self) -> String {
        let mut result = String::new();
        if let Some(event) = &self.event {
            result.push_str(&format!("event: {}\n", event));
        }
        if let Some(id) = &self.id {
            result.push_str(&format!("id: {}\n", id));
        }
        result.push_str(&format!("data: {}\n\n", self.data));
        result
    }
}

/// Model information returned by list_models
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Model {
    /// Model identifier
    pub id: String,
    /// Model object type (typically "model")
    pub object: String,
    /// Provider that owns this model
    pub owned_by: String,
    /// Optional creation timestamp
    #[serde(skip_serializing_if = "Option::is_none")]
    pub created: Option<i64>,
    /// Maximum context window in tokens (input + output combined)
    /// Populated from provider's /models endpoint when available
    #[serde(skip_serializing_if = "Option::is_none")]
    pub context_window: Option<u32>,
    /// Maximum output tokens for completions
    /// Populated from provider's /models endpoint when available
    #[serde(skip_serializing_if = "Option::is_none")]
    pub max_completion_tokens: Option<u32>,
    /// Whether this model supports image/vision inputs.
    /// Populated from provider's /models endpoint when available; defaults to false.
    #[serde(default, skip_serializing_if = "std::ops::Not::not")]
    pub supports_vision: bool,
}

/// Provider response wrapper
#[derive(Debug, Clone)]
#[allow(dead_code)]
pub struct ProviderResponse {
    /// The OpenAI-formatted response
    pub response: OpenAIResponse,
    /// Provider name that generated this response
    pub provider_name: String,
    /// Response latency in milliseconds
    pub latency_ms: u64,
}

/// Unified provider client trait
/// All provider implementations must implement this trait
#[async_trait]
#[allow(dead_code)]
pub trait ProviderClient: Send + Sync {
    /// Execute a non-streaming chat completion request
    ///
    /// # Arguments
    /// * `request` - OpenAI-formatted chat completion request
    ///
    /// # Returns
    /// * `Ok(ProviderResponse)` - Successful response with metadata
    /// * `Err(GatewayError)` - Provider error, network error, or translation error
    async fn chat_completion(
        &self,
        request: OpenAIRequest,
    ) -> Result<ProviderResponse, GatewayError>;

    /// Execute a streaming chat completion request
    ///
    /// # Arguments
    /// * `request` - OpenAI-formatted chat completion request with stream=true
    ///
    /// # Returns
    /// * `Ok(Stream)` - Stream of SSE events
    /// * `Err(GatewayError)` - Provider error, network error, or translation error
    async fn chat_completion_stream(
        &self,
        request: OpenAIRequest,
    ) -> Result<Pin<Box<dyn Stream<Item = Result<SSEEvent, GatewayError>> + Send>>, GatewayError>;

    /// List available models from this provider
    ///
    /// # Returns
    /// * `Ok(Vec<Model>)` - List of available models
    /// * `Err(GatewayError)` - Provider error or network error
    async fn list_models(&self) -> Result<Vec<Model>, GatewayError>;

    /// Get the provider name
    fn provider_name(&self) -> &str;
}
