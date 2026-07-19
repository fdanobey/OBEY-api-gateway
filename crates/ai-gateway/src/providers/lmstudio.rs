use async_trait::async_trait;
use futures::Stream;
use std::collections::HashMap;
use std::pin::Pin;

use super::openai_compatible::OpenAICompatibleProvider;
use crate::error::GatewayError;
use crate::models::openai::OpenAIRequest;
use crate::providers::{Model, ProviderClient, ProviderResponse, SSEEvent};

/// LM Studio provider client
/// Uses OpenAI-compatible API format
pub struct LMStudioProvider {
    inner: OpenAICompatibleProvider,
}

impl LMStudioProvider {
    /// Create a new LM Studio provider client
    /// LM Studio typically runs at http://localhost:1234/v1 and doesn't require an API key
    pub fn new(
        name: String,
        base_url: String,
        max_connections: Option<u32>,
        timeout_seconds: Option<u64>,
        custom_headers: HashMap<String, String>,
    ) -> Result<Self, GatewayError> {
        let inner = OpenAICompatibleProvider::new(
            name,
            base_url,
            String::new(), // LM Studio doesn't require API key
            max_connections,
            timeout_seconds,
            custom_headers,
        )?;

        Ok(Self { inner })
    }
}

#[async_trait]
impl ProviderClient for LMStudioProvider {
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
    fn test_lmstudio_provider_creation() {
        let provider = LMStudioProvider::new(
            "lmstudio-local".to_string(),
            "http://localhost:1234/v1".to_string(),
            None,
            None,
            HashMap::new(),
        );

        assert!(provider.is_ok());
        let provider = provider.unwrap();
        assert_eq!(provider.provider_name(), "lmstudio-local");
    }
}
