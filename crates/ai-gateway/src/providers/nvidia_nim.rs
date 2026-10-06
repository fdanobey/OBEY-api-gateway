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
/// Probe provenance: catalog=https://integrate.api.nvidia.com/v1/models; probed=2026-10-05T17:47:55.3495924Z; git_rev=b496311
pub const NVIDIA_NIM_FALLBACK_MODELS: &[NimFallbackModel] = &[
    NimFallbackModel {
        id: "moonshotai/kimi-k3",
        owned_by: "moonshotai",
        supports_vision: true,
        context_window: Some(1000000),
        max_completion_tokens: None,
        source_url: "https://build.nvidia.com/moonshotai/kimi-k3",
    },
    NimFallbackModel {
        id: "meta/muse-glimmer-30b",
        owned_by: "meta",
        supports_vision: true,
        context_window: None,
        max_completion_tokens: None,
        source_url: "https://build.nvidia.com/meta/muse-glimmer-30b",
    },
    NimFallbackModel {
        id: "google/diffusiongemma-26b-a4b-it",
        owned_by: "google",
        supports_vision: true,
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

    /// Assert a single mapped `Model` satisfies the catalog invariants that must
    /// hold regardless of which models are curated: the id is a well-formed
    /// `owner/model` pair (exactly one `/`, non-empty segments), `owned_by` is
    /// non-empty, and the object tag is `"model"`.
    fn assert_model_well_formed(model: &Model) {
        let segments: Vec<&str> = model.id.split('/').collect();
        assert_eq!(
            segments.len(),
            2,
            "model id {:?} must be of the form owner/model",
            model.id
        );
        assert!(
            !segments[0].is_empty(),
            "model id {:?} has an empty owner segment",
            model.id
        );
        assert!(
            !segments[1].is_empty(),
            "model id {:?} has an empty model segment",
            model.id
        );
        assert!(
            !model.owned_by.is_empty(),
            "model {:?} has an empty owned_by",
            model.id
        );
        assert_eq!(model.object, "model", "model {:?} has wrong object tag", model.id);
    }

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

        // Structural invariants: the fallback catalog must be non-empty and map
        // 1:1 (in order) from the source const, with every entry well-formed.
        assert!(!models.is_empty(), "fallback catalog must not be empty");
        assert_eq!(
            models.len(),
            NVIDIA_NIM_FALLBACK_MODELS.len(),
            "mapping must preserve the number of curated entries"
        );

        for (i, model) in models.iter().enumerate() {
            assert_eq!(
                model.id, NVIDIA_NIM_FALLBACK_MODELS[i].id,
                "mapped id at index {i} must match the source const"
            );
            assert_eq!(
                model.owned_by, NVIDIA_NIM_FALLBACK_MODELS[i].owned_by,
                "mapped owned_by at index {i} must match the source const"
            );
            assert_model_well_formed(model);
        }

        // Validate the source const directly: every entry has a well-formed
        // owner/model id, a non-empty owned_by, and a source_url that points at
        // the NVIDIA build domain.
        for entry in NVIDIA_NIM_FALLBACK_MODELS {
            let segments: Vec<&str> = entry.id.split('/').collect();
            assert_eq!(
                segments.len(),
                2,
                "const id {:?} must be of the form owner/model",
                entry.id
            );
            assert!(
                !segments[0].is_empty() && !segments[1].is_empty(),
                "const id {:?} has an empty segment",
                entry.id
            );
            assert!(
                !entry.owned_by.is_empty(),
                "const entry {:?} has an empty owned_by",
                entry.id
            );
            assert!(
                entry.source_url.starts_with("https://build.nvidia.com"),
                "const entry {:?} source_url {:?} must point at build.nvidia.com",
                entry.id,
                entry.source_url
            );
        }
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
        assert!(!models.is_empty(), "fallback catalog must not be empty");
        assert_eq!(models.len(), NVIDIA_NIM_FALLBACK_MODELS.len());
        // On an empty live catalog the provider returns the fallback catalog in
        // source order; index 0 is read from the const, not hard-coded.
        assert_eq!(models[0].id, NVIDIA_NIM_FALLBACK_MODELS[0].id);
        assert_model_well_formed(&models[0]);
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
        assert!(!models.is_empty(), "fallback catalog must not be empty");
        assert_eq!(models.len(), NVIDIA_NIM_FALLBACK_MODELS.len());
        // On a live-catalog error the provider falls back in source order;
        // verify the full mapping generically rather than naming any model.
        for (i, model) in models.iter().enumerate() {
            assert_eq!(model.id, NVIDIA_NIM_FALLBACK_MODELS[i].id);
            assert_model_well_formed(model);
        }
    }
}
