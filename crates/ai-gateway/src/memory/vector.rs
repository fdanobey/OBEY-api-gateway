//! Optional Qdrant-backed semantic retrieval for persistent memory.

use std::collections::HashMap;
use std::sync::Arc;

use async_trait::async_trait;
use qdrant_client::qdrant::{
    CreateCollectionBuilder, Distance, PointStruct, SearchPointsBuilder, UpsertPointsBuilder,
    VectorParamsBuilder,
};
use qdrant_client::Qdrant;
use reqwest::Client;
use serde::Deserialize;
use uuid::Uuid;

use crate::config::Provider;

use super::{MemoryEntry, MemoryError, MemoryQdrantConfig};

const DEFAULT_VECTOR_DIMENSION: u64 = 1536;
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
        };
        tier.ensure_collection().await?;
        Ok(tier)
    }

    async fn ensure_collection(&self) -> Result<(), MemoryError> {
        let exists = self
            .qdrant
            .collection_exists(&self.collection)
            .await
            .map_err(|error| {
                MemoryError::Qdrant(format!("Qdrant collection check failed: {error}"))
            })?;
        if !exists {
            self.qdrant
                .create_collection(
                    CreateCollectionBuilder::new(&self.collection).vectors_config(
                        VectorParamsBuilder::new(
                            vector_dimension_for_model(&self.embedding_model),
                            Distance::Cosine,
                        ),
                    ),
                )
                .await
                .map_err(|error| {
                    MemoryError::Qdrant(format!("Qdrant collection creation failed: {error}"))
                })?;
        }
        Ok(())
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

fn normalize_qdrant_url(url: &str) -> String {
    if url.contains(":6333") {
        url.replace(":6333", ":6334")
    } else {
        url.to_owned()
    }
}

fn vector_dimension_for_model(model: &str) -> u64 {
    match model {
        "text-embedding-3-large" => 3072,
        "text-embedding-3-small" | "text-embedding-ada-002" => 1536,
        _ => DEFAULT_VECTOR_DIMENSION,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

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
}
