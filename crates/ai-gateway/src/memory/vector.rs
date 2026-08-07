//! Optional Qdrant-backed semantic retrieval for persistent memory.

use std::collections::HashMap;
use std::sync::Arc;

use async_trait::async_trait;
use qdrant_client::qdrant::{
    vectors_config, CreateCollectionBuilder, Distance, PointStruct, SearchPointsBuilder,
    UpsertPointsBuilder, VectorParamsBuilder,
};
use qdrant_client::Qdrant;
use reqwest::Client;
use serde::Deserialize;
use uuid::Uuid;

use crate::config::Provider;

use super::{MemoryEntry, MemoryError, MemoryQdrantConfig};

pub(crate) const VECTOR_INDEX_TIMEOUT: std::time::Duration = std::time::Duration::from_secs(5);

#[derive(Debug, Clone, Copy, PartialEq)]
pub struct VectorMatch {
    pub id: Uuid,
    pub score: f32,
}

#[async_trait]
pub trait MemoryVectorTier: Send + Sync {
    async fn index(&self, entry: &MemoryEntry) -> Result<(), MemoryError>;
    async fn search(&self, query: &str, limit: usize) -> Result<Vec<VectorMatch>, MemoryError>;
}

pub struct QdrantMemoryVectorTier {
    qdrant: Arc<Qdrant>,
    http: Client,
    collection: String,
    similarity_threshold: f32,
    embedding_provider: String,
    embedding_model: String,
    embedding_base_url: String,
    embedding_api_key: String,
    vector_dimension_override: Option<u64>,
}

impl QdrantMemoryVectorTier {
    pub async fn new(
        config: &MemoryQdrantConfig,
        provider: &Provider,
    ) -> Result<Self, MemoryError> {
        if provider.auth_method.as_deref() == Some("oauth") {
            return Err(MemoryError::Qdrant(
                "OAuth embedding providers are unsupported for direct memory indexing".to_owned(),
            ));
        }
        if provider.provider_type == "bedrock" {
            return Err(MemoryError::Qdrant(
                "Bedrock embedding providers are unsupported by the OpenAI-compatible memory adapter"
                    .to_owned(),
            ));
        }
        let base_url = provider
            .base_url
            .as_deref()
            .map(normalize_embedding_base_url)
            .filter(|url| !url.is_empty())
            .ok_or_else(|| {
                MemoryError::Qdrant(format!(
                    "embedding provider '{}' has no HTTP base URL",
                    provider.name
                ))
            })?;
        let api_key = provider.resolve_api_key().unwrap_or_default();
        let qdrant_url = normalize_qdrant_url(&config.qdrant_url);
        let qdrant = Qdrant::from_url(&qdrant_url)
            .build()
            .map_err(|error| MemoryError::Qdrant(format!("Qdrant client build failed: {error}")))?;
        let tier = Self {
            qdrant: Arc::new(qdrant),
            http: Client::builder()
                .pool_max_idle_per_host(10)
                .build()
                .map_err(|error| {
                    MemoryError::Qdrant(format!("embedding HTTP client build failed: {error}"))
                })?,
            collection: config.qdrant_collection.clone(),
            similarity_threshold: config.similarity_threshold,
            embedding_provider: provider.name.clone(),
            embedding_model: config.embedding_model.clone(),
            embedding_base_url: base_url,
            embedding_api_key: api_key,
            vector_dimension_override: config.vector_dimension,
        };
        tier.ensure_collection().await?;
        Ok(tier)
    }

    async fn ensure_collection(&self) -> Result<(), MemoryError> {
        // Resolve dimension first (three-tier: override → lookup → probe)
        let dimension = if let Some(d) = self.vector_dimension_override {
            tracing::info!("Using manual vector_dimension override: {d}");
            d
        } else if let Some(d) = vector_dimension_for_model(&self.embedding_model) {
            d
        } else {
            tracing::warn!("Model not in lookup table, falling back to probe embedding");
            self.probe_dimension().await?
        };

        let exists = self
            .qdrant
            .collection_exists(&self.collection)
            .await
            .map_err(|error| {
                MemoryError::Qdrant(format!("Qdrant collection check failed: {error}"))
            })?;

        if exists {
            // Validate existing collection's vector dimension against resolved dimension
            let info = self
                .qdrant
                .collection_info(&self.collection)
                .await
                .map_err(|error| {
                    MemoryError::Qdrant(format!("Qdrant collection info query failed: {error}"))
                })?;

            let existing_dimension = info
                .result
                .and_then(|ci| ci.config)
                .and_then(|cfg| cfg.params)
                .and_then(|params| params.vectors_config)
                .and_then(|vc| vc.config)
                .and_then(|config| match config {
                    vectors_config::Config::Params(params) => Some(params.size),
                    vectors_config::Config::ParamsMap(_) => None,
                });

            if let Some(existing) = existing_dimension {
                if existing != dimension {
                    return Err(MemoryError::Qdrant(format!(
                        "Qdrant collection '{}' has vector dimension {} but the configured embedding model produces dimension {}. Recreate the collection or set vector_dimension = {} in config.",
                        self.collection, existing, dimension, dimension
                    )));
                }
            }
        } else {
            self.qdrant
                .create_collection(
                    CreateCollectionBuilder::new(&self.collection)
                        .vectors_config(VectorParamsBuilder::new(dimension, Distance::Cosine)),
                )
                .await
                .map_err(|error| {
                    MemoryError::Qdrant(format!("Qdrant collection creation failed: {error}"))
                })?;
        }
        Ok(())
    }

    async fn probe_dimension(&self) -> Result<u64, MemoryError> {
        let embedding = self.embed("dimension probe").await?;
        let dimension = embedding.len() as u64;
        tracing::info!(
            dimension,
            model = %self.embedding_model,
            "Probed embedding dimension for unrecognized model"
        );
        Ok(dimension)
    }

    async fn embed(&self, text: &str) -> Result<Vec<f32>, MemoryError> {
        let url = format!("{}/embeddings", self.embedding_base_url);
        let mut request = self.http.post(url).json(&serde_json::json!({
            "model": self.embedding_model,
            "input": text,
        }));
        if !self.embedding_api_key.is_empty() {
            request = request.bearer_auth(&self.embedding_api_key);
        }
        let response = request.send().await.map_err(|error| {
            MemoryError::Qdrant(format!(
                "embedding request to provider '{}' failed: {error}",
                self.embedding_provider
            ))
        })?;
        let status = response.status();
        if !status.is_success() {
            return Err(MemoryError::Qdrant(format!(
                "embedding provider '{}' returned HTTP {}",
                self.embedding_provider,
                status.as_u16()
            )));
        }
        let response: EmbeddingResponse = response.json().await.map_err(|error| {
            MemoryError::Qdrant(format!(
                "embedding response from '{}' was invalid: {error}",
                self.embedding_provider
            ))
        })?;
        response
            .data
            .into_iter()
            .next()
            .map(|item| item.embedding)
            .filter(|embedding| !embedding.is_empty())
            .ok_or_else(|| {
                MemoryError::Qdrant(format!(
                    "embedding response from '{}' contained no vector",
                    self.embedding_provider
                ))
            })
    }
}

#[async_trait]
impl MemoryVectorTier for QdrantMemoryVectorTier {
    async fn index(&self, entry: &MemoryEntry) -> Result<(), MemoryError> {
        let embedding = self.embed(&entry.content).await?;
        let mut payload: HashMap<String, serde_json::Value> = HashMap::new();
        payload.insert("memory_id".to_owned(), entry.id.to_string().into());
        payload.insert("namespace".to_owned(), entry.namespace.clone().into());
        self.qdrant
            .upsert_points(
                UpsertPointsBuilder::new(
                    &self.collection,
                    vec![PointStruct::new(entry.id.to_string(), embedding, payload)],
                )
                .wait(true),
            )
            .await
            .map_err(|error| MemoryError::Qdrant(format!("Qdrant upsert failed: {error}")))?;
        Ok(())
    }

    async fn search(&self, query: &str, limit: usize) -> Result<Vec<VectorMatch>, MemoryError> {
        if limit == 0 {
            return Ok(Vec::new());
        }
        let embedding = self.embed(query).await?;
        let result = self
            .qdrant
            .search_points(
                SearchPointsBuilder::new(&self.collection, embedding, limit as u64)
                    .score_threshold(self.similarity_threshold)
                    .with_payload(true),
            )
            .await
            .map_err(|error| MemoryError::Qdrant(format!("Qdrant search failed: {error}")))?;
        Ok(result
            .result
            .into_iter()
            .filter_map(|point| {
                let id = point
                    .payload
                    .get("memory_id")
                    .and_then(|value| value.as_str())
                    .and_then(|value| Uuid::parse_str(value).ok())?;
                Some(VectorMatch {
                    id,
                    score: point.score,
                })
            })
            .collect())
    }
}

#[derive(Debug, Deserialize)]
struct EmbeddingResponse {
    data: Vec<EmbeddingData>,
}

#[derive(Debug, Deserialize)]
struct EmbeddingData {
    embedding: Vec<f32>,
}

fn normalize_embedding_base_url(url: &str) -> String {
    let mut url = url.trim_end_matches('/').to_owned();
    if !url.ends_with("/v1") {
        url.push_str("/v1");
    }
    url
}

pub(crate) fn normalize_qdrant_url(url: &str) -> String {
    if url.contains(":6333") {
        url.replace(":6333", ":6334")
    } else {
        url.to_owned()
    }
}

fn vector_dimension_for_model(model: &str) -> Option<u64> {
    match model {
        "text-embedding-3-large" => Some(3072),
        "text-embedding-3-small" | "text-embedding-ada-002" => Some(1536),
        "nomic-embed-text" => Some(768),
        "all-MiniLM-L6-v2" => Some(384),
        "bge-small-en-v1.5" => Some(384),
        "bge-base-en-v1.5" => Some(768),
        "bge-large-en-v1.5" => Some(1024),
        "mxbai-embed-large-v1" => Some(1024),
        "e5-mistral-7b-instruct" => Some(4096),
        "voyage-large-2" => Some(1536),
        "voyage-code-2" => Some(1536),
        "embed-english-v3.0" => Some(1024),
        "embed-multilingual-v3.0" => Some(1024),
        _ => None,
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use proptest::prelude::*;

    #[test]
    fn normalizes_transport_urls() {
        assert_eq!(
            normalize_qdrant_url("http://localhost:6333"),
            "http://localhost:6334"
        );
        assert_eq!(
            normalize_embedding_base_url("http://localhost:8080"),
            "http://localhost:8080/v1"
        );
        assert_eq!(
            normalize_embedding_base_url("http://localhost:8080/v1/"),
            "http://localhost:8080/v1"
        );
    }

    // **Validates: Requirements 3.1, 3.2, 3.3, 3.4**
    //
    // Preservation: Known OpenAI models must continue to return their correct
    // dimensions from the lookup table without any probe call. This test MUST
    // PASS on both unfixed and fixed code.
    proptest! {
        #[test]
        fn preservation_known_openai_models_return_correct_dimension(
            (model, expected_dim) in prop::sample::select(vec![
                ("text-embedding-3-small", 1536u64),
                ("text-embedding-3-large", 3072u64),
                ("text-embedding-ada-002", 1536u64),
            ])
        ) {
            let resolved = vector_dimension_for_model(model);
            prop_assert_eq!(
                resolved,
                Some(expected_dim),
                "vector_dimension_for_model(\"{}\") returned {:?} but expected Some({}) (preservation violated)",
                model,
                resolved,
                expected_dim
            );
        }
    }

    // **Validates: Requirements 1.1, 1.2**
    //
    // Bug Condition Exploration: non-OpenAI models should return their actual
    // embedding dimension, NOT the default 1536. This test is EXPECTED TO FAIL
    // on unfixed code, proving the bug exists.
    proptest! {
        #[test]
        fn bug_condition_non_openai_models_return_correct_dimension(
            (model, expected_dim) in prop::sample::select(vec![
                ("nomic-embed-text", 768u64),
                ("all-MiniLM-L6-v2", 384u64),
                ("bge-small-en-v1.5", 384u64),
                ("bge-base-en-v1.5", 768u64),
                ("bge-large-en-v1.5", 1024u64),
                ("mxbai-embed-large-v1", 1024u64),
                ("e5-mistral-7b-instruct", 4096u64),
                ("embed-english-v3.0", 1024u64),
                ("embed-multilingual-v3.0", 1024u64),
            ])
        ) {
            let resolved = vector_dimension_for_model(model);
            prop_assert_eq!(
                resolved,
                Some(expected_dim),
                "vector_dimension_for_model(\"{}\") returned {:?} but expected Some({})",
                model,
                resolved,
                expected_dim
            );
        }
    }
}
