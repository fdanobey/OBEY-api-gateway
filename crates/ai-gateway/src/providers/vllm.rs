use async_trait::async_trait;
use futures::Stream;
use std::collections::HashMap;
use std::pin::Pin;

use super::openai_compatible::OpenAICompatibleProvider;
use crate::error::GatewayError;
use crate::models::openai::OpenAIRequest;
use crate::providers::{Model, ProviderClient, ProviderResponse, SSEEvent};

/// vLLM provider client
/// Uses OpenAI-compatible API format
pub struct VLLMProvider {
    inner: OpenAICompatibleProvider,
}

impl VLLMProvider {
    /// Create a new vLLM provider client
    /// vLLM typically runs locally and exposes OpenAI-compatible endpoints
    pub fn new(
        name: String,
        base_url: String,
        api_key: Option<String>,
        max_connections: Option<u32>,
        timeout_seconds: Option<u64>,
        custom_headers: HashMap<String, String>,
    ) -> Result<Self, GatewayError> {
        let inner = OpenAICompatibleProvider::new(
            name,
            base_url,
            api_key.unwrap_or_default(),
            max_connections,
            timeout_seconds,
            custom_headers,
        )?;

        Ok(Self { inner })
    }
}

#[async_trait]
impl ProviderClient for VLLMProvider {
    async fn chat_completion(
        &self,
        request: OpenAIRequest,
    ) -> Result<ProviderResponse, GatewayError> {
        self.inner.chat_completion(request).await
    }

    async fn chat_completion_stream(
        &self,
        request: OpenAIRequest,
    ) -> Result<Pin<Box<dyn Stream<Item = Result<SSEEvent, GatewayError>> + Send>>, GatewayError>
    {
        self.inner.chat_completion_stream(request).await
    }

    async fn list_models(&self) -> Result<Vec<Model>, GatewayError> {
        self.inner.list_models().await
    }

    fn provider_name(&self) -> &str {
        self.inner.provider_name()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::collections::HashMap;

    #[test]
    fn test_vllm_provider_creation_without_key() {
        let provider = VLLMProvider::new(
            "vllm-local".to_string(),
            "http://localhost:8000/v1".to_string(),
            None,
            None,
            None,
            HashMap::new(),
        );

        assert!(provider.is_ok());
        let provider = provider.unwrap();
        assert_eq!(provider.provider_name(), "vllm-local");
    }

    #[test]
    fn test_vllm_provider_creation_with_key() {
        let provider = VLLMProvider::new(
            "vllm-remote".to_string(),
            "https://vllm.example.com/v1".to_string(),
            Some("test-key".to_string()),
            None,
            None,
            HashMap::new(),
        );

        assert!(provider.is_ok());
        assert_eq!(provider.unwrap().provider_name(), "vllm-remote");
    }
}
