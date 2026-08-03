use async_trait::async_trait;
use futures::Stream;
use std::collections::HashMap;
use std::pin::Pin;

use super::openai_compatible::OpenAICompatibleProvider;
use crate::error::GatewayError;
use crate::models::openai::OpenAIRequest;
use crate::providers::{Model, ProviderClient, ProviderResponse, SSEEvent};

#[derive(Debug, Clone)]
pub struct NimFallbackModel {
    pub id: &'static str,
    pub owned_by: &'static str,
    pub supports_vision: bool,
    pub context_window: Option<u32>,
    pub max_completion_tokens: Option<u32>,
    pub source_url: &'static str,
}

// BEGIN NVIDIA NIM FALLBACK MODELS
/// Probe provenance: catalog=https://integrate.api.nvidia.com/v1/models; probed=2026-08-03T10:13:34.2943330Z; git_rev=ff65df5
pub const NVIDIA_NIM_FALLBACK_MODELS: &[NimFallbackModel] = &[
    NimFallbackModel {
        id: "openai/gpt-oss-120b",
        owned_by: "openai",
        supports_vision: false,
        context_window: Some(128000),
        max_completion_tokens: None,
        source_url: "https://build.nvidia.com/openai/gpt-oss-120b",
    },
    NimFallbackModel {
        id: "meta/llama-3.1-70b-instruct",
        owned_by: "meta",
        supports_vision: false,
        context_window: None,
        max_completion_tokens: None,
        source_url: "https://build.nvidia.com/meta/llama-3.1-70b-instruct",
    },
    NimFallbackModel {
        id: "google/diffusiongemma-26b-a4b-it",
        owned_by: "google",
        supports_vision: false,
        context_window: None,
        max_completion_tokens: None,
        source_url: "https://build.nvidia.com/google/diffusiongemma-26b-a4b-it",
    },
];
// END NVIDIA NIM FALLBACK MODELS

pub fn fallback_models() -> Vec<Model> {
    NVIDIA_NIM_FALLBACK_MODELS
        .iter()
        .map(|model| Model {
            id: model.id.to_string(),
            object: "model".to_string(),
            owned_by: model.owned_by.to_string(),
            created: None,
            context_window: model.context_window,
            max_completion_tokens: model.max_completion_tokens,
            supports_vision: model.supports_vision,
        })
        .collect()
}

/// NVIDIA NIM provider client
/// Uses OpenAI-compatible API format
pub struct NvidiaNIMProvider {
    inner: OpenAICompatibleProvider,
}

impl NvidiaNIMProvider {
    /// Create a new NVIDIA NIM provider client
    /// NVIDIA NIM API endpoint: https://integrate.api.nvidia.com/v1
    pub fn new(
        name: String,
        api_key: String,
        max_connections: Option<u32>,
        timeout_seconds: Option<u64>,
        custom_headers: HashMap<String, String>,
    ) -> Result<Self, GatewayError> {
        let inner = OpenAICompatibleProvider::new(
            name,
            "https://integrate.api.nvidia.com/v1".to_string(),
            api_key,
            max_connections,
            timeout_seconds,
            custom_headers,
        )?;

        Ok(Self { inner })
    }

    /// Create a new NVIDIA NIM provider with custom base URL
    pub fn new_with_base_url(
        name: String,
        base_url: String,
        api_key: String,
        max_connections: Option<u32>,
        timeout_seconds: Option<u64>,
        custom_headers: HashMap<String, String>,
    ) -> Result<Self, GatewayError> {
        let inner = OpenAICompatibleProvider::new(
            name,
            base_url,
            api_key,
            max_connections,
            timeout_seconds,
            custom_headers,
        )?;
        Ok(Self { inner })
    }
}

#[async_trait]
impl ProviderClient for NvidiaNIMProvider {
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
        match self.inner.list_models().await {
            Ok(models) if !models.is_empty() => Ok(models),
            Ok(_) => {
                tracing::warn!(
                    provider = self.provider_name(),
                    "NVIDIA NIM returned an empty model list; using built-in fallback catalog"
                );
                Ok(fallback_models())
            }
            Err(error) => {
                tracing::warn!(
                    provider = self.provider_name(),
                    error = %error,
                    "NVIDIA NIM model discovery failed; using built-in fallback catalog"
                );
                Ok(fallback_models())
            }
        }
    }

    fn provider_name(&self) -> &str {
        self.inner.provider_name()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::collections::HashMap;
    use wiremock::matchers::{method, path};
    use wiremock::{Mock, MockServer, ResponseTemplate};

    #[test]
    fn test_nvidia_nim_provider_creation() {
        let provider = NvidiaNIMProvider::new(
            "nvidia-nim".to_string(),
            "test-api-key".to_string(),
            None,
            None,
            HashMap::new(),
        );

        assert!(provider.is_ok());
        let provider = provider.unwrap();
        assert_eq!(provider.provider_name(), "nvidia-nim");
    }

    #[test]
    fn test_nvidia_nim_provider_with_custom_base_url() {
        let provider = NvidiaNIMProvider::new_with_base_url(
            "nvidia-custom".to_string(),
            "https://custom.nvidia.com/v1".to_string(),
            "test-key".to_string(),
            None,
            None,
            HashMap::new(),
        );

        assert!(provider.is_ok());
        assert_eq!(provider.unwrap().provider_name(), "nvidia-custom");
    }

    #[test]
    fn fallback_catalog_has_expected_models_and_metadata() {
        let models = fallback_models();
        let ids: Vec<&str> = models.iter().map(|model| model.id.as_str()).collect();

        assert_eq!(
            ids,
            vec![
                "openai/gpt-oss-120b",
                "meta/llama-3.1-70b-instruct",
                "nvidia/nemotron-3-nano",
            ]
        );
        assert!(models.iter().all(|model| !model.supports_vision));
        assert_eq!(models[0].context_window, Some(128_000));
    }

    #[tokio::test]
    async fn list_models_preserves_non_empty_live_catalog() {
        let server = MockServer::start().await;
        Mock::given(method("GET"))
            .and(path("/v1/models"))
            .respond_with(ResponseTemplate::new(200).set_body_json(serde_json::json!({
                "data": [{
                    "id": "publisher/live-model",
                    "object": "model",
                    "owned_by": "publisher"
                }]
            })))
            .mount(&server)
            .await;

        let provider = NvidiaNIMProvider::new_with_base_url(
            "nvidia-test".to_string(),
            format!("{}/v1", server.uri()),
            "test-key".to_string(),
            None,
            None,
            HashMap::new(),
        )
        .unwrap();

        let models = provider.list_models().await.unwrap();
        assert_eq!(models.len(), 1);
        assert_eq!(models[0].id, "publisher/live-model");
    }

    #[tokio::test]
    async fn list_models_falls_back_on_empty_live_catalog() {
        let server = MockServer::start().await;
        Mock::given(method("GET"))
            .and(path("/v1/models"))
            .respond_with(ResponseTemplate::new(200).set_body_json(serde_json::json!({
                "data": []
            })))
            .mount(&server)
            .await;

        let provider = NvidiaNIMProvider::new_with_base_url(
            "nvidia-test".to_string(),
            format!("{}/v1", server.uri()),
            "test-key".to_string(),
            None,
            None,
            HashMap::new(),
        )
        .unwrap();

        let models = provider.list_models().await.unwrap();
        assert_eq!(models.len(), NVIDIA_NIM_FALLBACK_MODELS.len());
        assert_eq!(models[0].id, "openai/gpt-oss-120b");
    }

    #[tokio::test]
    async fn list_models_falls_back_on_live_catalog_error() {
        let server = MockServer::start().await;
        Mock::given(method("GET"))
            .and(path("/v1/models"))
            .respond_with(ResponseTemplate::new(500))
            .mount(&server)
            .await;

        let provider = NvidiaNIMProvider::new_with_base_url(
            "nvidia-test".to_string(),
            format!("{}/v1", server.uri()),
            "test-key".to_string(),
            None,
            None,
            HashMap::new(),
        )
        .unwrap();

        let models = provider.list_models().await.unwrap();
        assert_eq!(models.len(), NVIDIA_NIM_FALLBACK_MODELS.len());
        assert_eq!(models[2].id, "nvidia/nemotron-3-nano");
    }
}
