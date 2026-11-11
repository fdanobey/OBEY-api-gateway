#[allow(dead_code)]
pub mod bedrock;
#[allow(dead_code)]
pub mod github_copilot;
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
    #[serde(alias = "context_length", alias = "max_model_len", skip_serializing_if = "Option::is_none")]
    pub context_window: Option<u32>,
    /// Maximum output tokens for completions
    /// Populated from provider's /models endpoint when available
    #[serde(alias = "max_output_tokens", skip_serializing_if = "Option::is_none")]
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

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_model_deserializes_context_length_alias() {
        let json =
            r#"{"id":"test-model","object":"model","owned_by":"test","context_length":200000}"#;
        let model: Model =
            serde_json::from_str(json).expect("context_length alias should deserialize");
        assert_eq!(model.context_window, Some(200000));
        assert_eq!(model.max_completion_tokens, None);
    }

    #[test]
    fn test_model_deserializes_max_model_len_alias() {
        let json =
            r#"{"id":"test-model","object":"model","owned_by":"test","max_model_len":131072}"#;
        let model: Model = serde_json::from_str(json).expect("max_model_len alias should deserialize");
        assert_eq!(model.context_window, Some(131072));
        assert_eq!(model.max_completion_tokens, None);
    }

    #[test]
    fn test_model_deserializes_max_output_tokens_alias() {
        let json =
            r#"{"id":"test-model","object":"model","owned_by":"test","max_output_tokens":8192}"#;
        let model: Model =
            serde_json::from_str(json).expect("max_output_tokens alias should deserialize");
        assert_eq!(model.max_completion_tokens, Some(8192));
        assert_eq!(model.context_window, None);
    }

    #[test]
    fn test_model_max_input_tokens_maps_to_nothing() {
        // Deliberate non-mapping: max_input_tokens is not an alias of either
        // field and must leave both as None.
        let json =
            r#"{"id":"test-model","object":"model","owned_by":"test","max_input_tokens":16384}"#;
        let model: Model = serde_json::from_str(json).expect("payload should still parse");
        assert_eq!(model.context_window, None);
        assert_eq!(model.max_completion_tokens, None);
    }

    #[test]
    fn test_model_ignores_unknown_fields() {
        let json = r#"{"id":"test-model","object":"model","owned_by":"test","architecture":{"modality":"text"},"top_pricing":0.5}"#;
        let model: Model =
            serde_json::from_str(json).expect("unknown extra fields should be ignored");
        assert_eq!(model.id, "test-model");
        assert_eq!(model.context_window, None);
        assert_eq!(model.max_completion_tokens, None);
    }

    #[test]
    fn test_model_deserializes_canonical_field_names() {
        let json = r#"{"id":"test-model","object":"model","owned_by":"test","context_window":65536,"max_completion_tokens":4096}"#;
        let model: Model =
            serde_json::from_str(json).expect("canonical names should deserialize");
        assert_eq!(model.context_window, Some(65536));
        assert_eq!(model.max_completion_tokens, Some(4096));
    }

    #[test]
    fn test_model_serializes_canonical_names_only() {
        let model = Model {
            id: "test-model".to_string(),
            object: "model".to_string(),
            owned_by: "test".to_string(),
            created: None,
            context_window: Some(200000),
            max_completion_tokens: Some(8192),
            supports_vision: false,
        };
        let value = serde_json::to_value(&model).expect("serialization should succeed");
        assert_eq!(value.get("id"), Some(&serde_json::json!("test-model")));
        assert_eq!(value.get("object"), Some(&serde_json::json!("model")));
        assert_eq!(value.get("owned_by"), Some(&serde_json::json!("test")));
        assert_eq!(
            value.get("context_window"),
            Some(&serde_json::json!(200000)),
            "serialized: {value}"
        );
        assert_eq!(
            value.get("max_completion_tokens"),
            Some(&serde_json::json!(8192)),
            "serialized: {value}"
        );
        assert!(
            value.get("context_length").is_none(),
            "alias context_length must not serialize: {value}"
        );
        assert!(
            value.get("max_model_len").is_none(),
            "alias max_model_len must not serialize: {value}"
        );
        assert!(
            value.get("max_output_tokens").is_none(),
            "alias max_output_tokens must not serialize: {value}"
        );
        assert!(
            value.get("created").is_none(),
            "None field created must be omitted: {value}"
        );
        assert!(
            value.get("supports_vision").is_none(),
            "false supports_vision must be omitted: {value}"
        );

        let round_tripped: Model =
            serde_json::from_value(value.clone()).expect("round-trip should deserialize");
        assert_eq!(round_tripped.context_window, Some(200000));
        assert_eq!(round_tripped.max_completion_tokens, Some(8192));
        assert_eq!(
            serde_json::to_value(&round_tripped).expect("re-serialization should succeed"),
            value
        );
    }

    #[test]
    fn test_model_conflicting_aliases_rejected_as_duplicate_field() {
        // serde's derived Deserialize maps a field and all its aliases to a
        // single Field variant, so a second occurrence under ANY name is a
        // hard "duplicate field" error — conflicting aliases do NOT silently
        // resolve last-field-wins at this type level.
        let json = r#"{"id":"test-model","object":"model","owned_by":"test","context_length":100000,"max_model_len":131072}"#;
        let result: Result<Model, _> = serde_json::from_str(json);
        let err = result.expect_err("duplicate field via aliases must be an error");
        assert!(
            err.to_string().contains("duplicate field `context_window`"),
            "unexpected error: {err}"
        );
    }
}
