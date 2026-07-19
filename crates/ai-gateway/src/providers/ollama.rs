use async_trait::async_trait;
use futures::Stream;
use std::collections::HashMap;
use std::pin::Pin;

use super::openai_compatible::OpenAICompatibleProvider;
use crate::error::GatewayError;
use crate::models::openai::OpenAIRequest;
use crate::providers::{Model, ProviderClient, ProviderResponse, SSEEvent};

/// Ollama provider client
/// Uses OpenAI-compatible API format
pub struct OllamaProvider {
    inner: OpenAICompatibleProvider,
}

impl OllamaProvider {
    /// Create a new Ollama provider client
    /// Ollama typically runs at http://localhost:11434 and doesn't require an API key
    pub fn new(
        name: String,
        base_url: String,
        max_connections: Option<u32>,
        timeout_seconds: Option<u64>,
        custom_headers: HashMap<String, String>,
    ) -> Result<Self, GatewayError> {
        let inner = OpenAICompatibleProvider::new(
            name,
            format!("{}/v1", base_url.trim_end_matches('/')),
            String::new(), // Ollama doesn't require API key
            max_connections,
            timeout_seconds,
            custom_headers,
        )?;

        Ok(Self { inner })
    }
}

#[async_trait]
impl ProviderClient for OllamaProvider {
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
    fn test_ollama_provider_creation() {
        let provider = OllamaProvider::new(
            "ollama-local".to_string(),
            "http://localhost:11434".to_string(),
            None,
            None,
            HashMap::new(),
        );

        assert!(provider.is_ok());
        let provider = provider.unwrap();
        assert_eq!(provider.provider_name(), "ollama-local");
    }

    #[test]
    fn test_ollama_provider_base_url_normalization() {
        let provider1 = OllamaProvider::new(
            "test".to_string(),
            "http://localhost:11434".to_string(),
            None,
            None,
            HashMap::new(),
        );
        let provider2 = OllamaProvider::new(
            "test".to_string(),
            "http://localhost:11434/".to_string(),
            None,
            None,
            HashMap::new(),
        );

        assert!(provider1.is_ok());
        assert!(provider2.is_ok());
    }
}
