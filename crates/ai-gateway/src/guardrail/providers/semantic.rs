//! Semantic guardrail provider (Req 7).
//!
//! Detects prompt-injection and policy-violating prompts using embedding-based
//! cosine similarity against stored allow/deny examples. It reuses the semantic
//! cache's embedding provider/model and its Qdrant instance (Req 7.5, 7.1),
//! querying two separate collections — one for allow examples and one for deny
//! examples. Cosine similarity is provided directly by Qdrant's
//! [`Distance::Cosine`] scoring, matching how [`crate::cache::semantic`] scores
//! matches.
//!
//! The core allow/deny [`semantic_decision`] rule is a pure function (Req 7.3,
//! 7.4, 7.8) extracted so it can be unit- and property-tested without any
//! network or Qdrant dependency. The [`GuardrailProvider::analyze`]
//! implementation is a thin wrapper that computes an embedding, queries both
//! collections, and feeds the best allow/deny scores into the decision rule.
//! Any embedding or Qdrant transport failure is surfaced as a
//! [`GuardrailProviderError`] and a WARN is logged (Req 7.6, 7.7); the engine's
//! failure-policy wrapper then applies fail_open / fail_close.

use std::sync::Arc;

use async_trait::async_trait;
use qdrant_client::qdrant::SearchPointsBuilder;
use qdrant_client::Qdrant;
use reqwest::Client;
use serde::Deserialize;

use crate::guardrail::provider::{Finding, GuardrailProvider, GuardrailProviderError};

/// Default allow-collection cosine-similarity threshold (Req 7.6).
pub const DEFAULT_ALLOW_THRESHOLD: f32 = 0.90;

/// Default deny-collection cosine-similarity threshold (Req 7.6).
pub const DEFAULT_DENY_THRESHOLD: f32 = 0.85;

/// Entity label reported when a deny example is matched but the deny payload
/// carries no explicit label.
const DEFAULT_DENY_LABEL: &str = "semantic_violation";

/// Outcome of the pure allow/deny decision rule.
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum SemanticDecision {
    /// Content is permitted; no finding is produced.
    Permit,
    /// Content is flagged as a deny violation with the matched deny score.
    Flag {
        /// Cosine similarity of the matched deny example.
        deny_score: f32,
    },
}

/// Pure allow/deny decision rule (Req 7.3, 7.4, 7.8).
///
/// Inputs are the best cosine similarity found in each collection (or `None`
/// when a collection produced no match), the configured thresholds, and a
/// `both_empty` flag that is `true` when neither the allow nor the deny
/// collection contains any stored examples.
///
/// Rules:
/// - When `both_empty`, always [`SemanticDecision::Permit`] (Req 7.8).
/// - A deny example is "active" when its similarity strictly exceeds the deny
///   threshold (Req 7.3, "exceeds"). If no deny example is active, permit.
/// - An allow example "meets" its threshold when its similarity is greater than
///   or equal to the allow threshold (Req 7.4).
/// - When a deny example is active AND no allow example meets its threshold,
///   flag a deny finding (Req 7.3).
/// - When a deny example is active AND an allow example also meets its
///   threshold, permit iff `allow_similarity >= deny_similarity`, otherwise flag
///   (Req 7.4).
pub fn semantic_decision(
    allow_similarity: Option<f32>,
    deny_similarity: Option<f32>,
    allow_threshold: f32,
    deny_threshold: f32,
    both_empty: bool,
) -> SemanticDecision {
    // Req 7.8: no stored examples at all → always permit.
    if both_empty {
        return SemanticDecision::Permit;
    }

    // A deny example must strictly exceed the deny threshold to be actionable
    // (Req 7.3, "exceeds").
    let deny_active = deny_similarity.is_some_and(|d| d > deny_threshold);
    if !deny_active {
        return SemanticDecision::Permit;
    }
    let deny_score = deny_similarity.expect("deny_active implies Some");

    // An allow example "meets its threshold" at or above the allow threshold
    // (Req 7.4).
    let allow_meets = allow_similarity.is_some_and(|a| a >= allow_threshold);
    if allow_meets {
        let allow_score = allow_similarity.expect("allow_meets implies Some");
        // Both thresholds satisfied: permit iff allow dominates deny (Req 7.4).
        if allow_score >= deny_score {
            SemanticDecision::Permit
        } else {
            SemanticDecision::Flag { deny_score }
        }
    } else {
        // Deny active, no allow meets → flag (Req 7.3).
        SemanticDecision::Flag { deny_score }
    }
}

/// Result of querying a single Qdrant collection for the top match.
#[derive(Debug, Clone, Default)]
struct CollectionMatch {
    /// Best cosine similarity, if the collection returned any point.
    score: Option<f32>,
    /// Entity label carried in the matched point's payload, if any.
    label: Option<String>,
}

/// Semantic guardrail provider backed by Qdrant + an embedding endpoint.
///
/// Holds a clone of the semantic cache's shared [`Qdrant`] client and embedding
/// configuration so it reuses the same infrastructure (Req 7.1, 7.5). The allow
/// and deny examples live in two separate collections within that instance.
pub struct SemanticProvider {
    /// Shared Qdrant client (reused from the semantic cache).
    qdrant_client: Arc<Qdrant>,
    /// HTTP client for embedding API calls.
    http_client: Client,
    /// Embedding provider name (for error/log context).
    embedding_provider: String,
    /// Embedding model identifier.
    embedding_model: String,
    /// Base URL for the embedding provider's API (OpenAI-compatible).
    embedding_base_url: String,
    /// API key for the embedding provider.
    embedding_api_key: String,
    /// Qdrant collection holding allow-example embeddings.
    allow_collection: String,
    /// Qdrant collection holding deny-example embeddings.
    deny_collection: String,
    /// Allow-collection cosine-similarity threshold (0.0–1.0).
    allow_threshold: f32,
    /// Deny-collection cosine-similarity threshold (0.0–1.0).
    deny_threshold: f32,
}

impl SemanticProvider {
    /// Construct a provider from explicit parts.
    ///
    /// `allow_threshold` / `deny_threshold` should already be resolved to their
    /// configured or default values ([`DEFAULT_ALLOW_THRESHOLD`],
    /// [`DEFAULT_DENY_THRESHOLD`]).
    #[allow(clippy::too_many_arguments)]
    pub fn new(
        qdrant_client: Arc<Qdrant>,
        http_client: Client,
        embedding_provider: String,
        embedding_model: String,
        embedding_base_url: String,
        embedding_api_key: String,
        allow_collection: String,
        deny_collection: String,
        allow_threshold: f32,
        deny_threshold: f32,
    ) -> Self {
        Self {
            qdrant_client,
            http_client,
            embedding_provider,
            embedding_model,
            embedding_base_url,
            embedding_api_key,
            allow_collection,
            deny_collection,
            allow_threshold,
            deny_threshold,
        }
    }

    /// Construct a provider reusing the semantic cache's embedding provider,
    /// model, and Qdrant instance (Req 7.1, 7.5).
    ///
    /// The allow/deny collections and thresholds are guardrail-specific; pass
    /// `None` for a threshold to use its default.
    pub fn from_semantic_cache(
        cache: &crate::cache::semantic::SemanticCache,
        allow_collection: String,
        deny_collection: String,
        allow_threshold: Option<f32>,
        deny_threshold: Option<f32>,
    ) -> Self {
        Self::new(
            Arc::clone(&cache.qdrant_client),
            cache.http_client.clone(),
            cache.embedding_provider.clone(),
            cache.embedding_model.clone(),
            cache.embedding_base_url.clone(),
            cache.embedding_api_key.clone(),
            allow_collection,
            deny_collection,
            allow_threshold.unwrap_or(DEFAULT_ALLOW_THRESHOLD),
            deny_threshold.unwrap_or(DEFAULT_DENY_THRESHOLD),
        )
    }

    /// Generate an embedding vector for `text` by calling the configured
    /// OpenAI-compatible `/embeddings` endpoint. Mirrors the semantic cache's
    /// embedding path (Req 7.5).
    async fn generate_embedding(&self, text: &str) -> Result<Vec<f32>, GuardrailProviderError> {
        let url = format!("{}/embeddings", self.embedding_base_url);
        let body = serde_json::json!({
            "model": self.embedding_model,
            "input": text,
        });

        let response = self
            .http_client
            .post(&url)
            .header("Content-Type", "application/json")
            .header(
                "Authorization",
                format!("Bearer {}", self.embedding_api_key),
            )
            .json(&body)
            .send()
            .await
            .map_err(|e| {
                GuardrailProviderError::Unreachable(format!(
                    "embedding request to provider '{}' failed: {}",
                    self.embedding_provider, e
                ))
            })?;

        let status = response.status();
        if !status.is_success() {
            let message = response
                .text()
                .await
                .unwrap_or_else(|_| "unknown error".to_string());
            return Err(GuardrailProviderError::UpstreamStatus {
                status: status.as_u16(),
                message,
            });
        }

        let parsed: EmbeddingResponse = response.json().await.map_err(|e| {
            GuardrailProviderError::MalformedResponse(format!(
                "failed to parse embedding response from '{}': {}",
                self.embedding_provider, e
            ))
        })?;

        parsed
            .data
            .into_iter()
            .next()
            .map(|d| d.embedding)
            .ok_or_else(|| {
                GuardrailProviderError::MalformedResponse(format!(
                    "embedding response from '{}' contained no data",
                    self.embedding_provider
                ))
            })
    }

    /// Query a single collection for its top match against `embedding`.
    ///
    /// Cosine similarity is returned directly by Qdrant's scoring
    /// (`Distance::Cosine`). Returns an empty [`CollectionMatch`] when the
    /// collection is empty or does not exist.
    async fn query_collection(
        &self,
        collection: &str,
        embedding: Vec<f32>,
    ) -> Result<CollectionMatch, GuardrailProviderError> {
        let search = self
            .qdrant_client
            .search_points(SearchPointsBuilder::new(collection, embedding, 1).with_payload(true))
            .await
            .map_err(|e| {
                GuardrailProviderError::Unreachable(format!(
                    "Qdrant search on collection '{}' failed: {}",
                    collection, e
                ))
            })?;

        let Some(point) = search.result.into_iter().next() else {
            return Ok(CollectionMatch::default());
        };

        let label = point
            .payload
            .get("entity_label")
            .and_then(|v| v.as_str())
            .map(|s| s.to_string());

        Ok(CollectionMatch {
            score: Some(point.score),
            label,
        })
    }
}

#[async_trait]
impl GuardrailProvider for SemanticProvider {
    async fn analyze(&self, content: &str) -> Result<Vec<Finding>, GuardrailProviderError> {
        // Compute the input embedding once and reuse it for both collections
        // (Req 7.2).
        let embedding = self.generate_embedding(content).await.inspect_err(|e| {
            tracing::warn!(
                provider = %self.embedding_provider,
                error = %e,
                "semantic guardrail embedding failed"
            );
        })?;

        let allow_match = self
            .query_collection(&self.allow_collection, embedding.clone())
            .await
            .inspect_err(|e| {
                tracing::warn!(
                    collection = %self.allow_collection,
                    error = %e,
                    "semantic guardrail allow-collection query failed"
                );
            })?;

        let deny_match = self
            .query_collection(&self.deny_collection, embedding)
            .await
            .inspect_err(|e| {
                tracing::warn!(
                    collection = %self.deny_collection,
                    error = %e,
                    "semantic guardrail deny-collection query failed"
                );
            })?;

        // Both collections empty (no points returned from either) → permit
        // (Req 7.8).
        let both_empty = allow_match.score.is_none() && deny_match.score.is_none();

        match semantic_decision(
            allow_match.score,
            deny_match.score,
            self.allow_threshold,
            self.deny_threshold,
            both_empty,
        ) {
            SemanticDecision::Permit => Ok(Vec::new()),
            SemanticDecision::Flag { deny_score } => {
                let entity_label = deny_match
                    .label
                    .unwrap_or_else(|| DEFAULT_DENY_LABEL.to_string());
                Ok(vec![Finding {
                    entity_label,
                    start: 0,
                    end: content.len(),
                    matched_text: None,
                    score: Some(deny_score),
                }])
            }
        }
    }

    fn provider_type(&self) -> &'static str {
        "semantic"
    }
}

/// Response from an OpenAI-compatible `/embeddings` endpoint.
#[derive(Debug, Deserialize)]
struct EmbeddingResponse {
    data: Vec<EmbeddingData>,
}

/// Single embedding entry in the response.
#[derive(Debug, Deserialize)]
struct EmbeddingData {
    embedding: Vec<f32>,
}

#[cfg(test)]
mod tests {
    use super::*;

    const ALLOW_T: f32 = DEFAULT_ALLOW_THRESHOLD; // 0.90
    const DENY_T: f32 = DEFAULT_DENY_THRESHOLD; // 0.85

    #[test]
    fn both_empty_always_permits() {
        // Even with scores present, an empty pair of collections permits.
        assert_eq!(
            semantic_decision(Some(0.99), Some(0.99), ALLOW_T, DENY_T, true),
            SemanticDecision::Permit
        );
        assert_eq!(
            semantic_decision(None, None, ALLOW_T, DENY_T, true),
            SemanticDecision::Permit
        );
    }

    #[test]
    fn deny_not_exceeding_threshold_permits() {
        // deny score == threshold is not "exceeds"; permit.
        assert_eq!(
            semantic_decision(None, Some(DENY_T), ALLOW_T, DENY_T, false),
            SemanticDecision::Permit
        );
        // deny below threshold; permit.
        assert_eq!(
            semantic_decision(None, Some(0.10), ALLOW_T, DENY_T, false),
            SemanticDecision::Permit
        );
        // no deny score at all; permit.
        assert_eq!(
            semantic_decision(Some(0.99), None, ALLOW_T, DENY_T, false),
            SemanticDecision::Permit
        );
    }

    #[test]
    fn deny_exceeds_without_allow_flags() {
        assert_eq!(
            semantic_decision(None, Some(0.95), ALLOW_T, DENY_T, false),
            SemanticDecision::Flag { deny_score: 0.95 }
        );
        // Allow present but below its threshold → does not "meet"; flag.
        assert_eq!(
            semantic_decision(Some(0.80), Some(0.95), ALLOW_T, DENY_T, false),
            SemanticDecision::Flag { deny_score: 0.95 }
        );
    }

    #[test]
    fn both_exceed_permit_when_allow_ge_deny() {
        // allow >= deny → permit.
        assert_eq!(
            semantic_decision(Some(0.97), Some(0.95), ALLOW_T, DENY_T, false),
            SemanticDecision::Permit
        );
        // allow == deny → permit (>=).
        assert_eq!(
            semantic_decision(Some(0.95), Some(0.95), ALLOW_T, DENY_T, false),
            SemanticDecision::Permit
        );
    }

    #[test]
    fn both_exceed_flag_when_allow_lt_deny() {
        assert_eq!(
            semantic_decision(Some(0.91), Some(0.96), ALLOW_T, DENY_T, false),
            SemanticDecision::Flag { deny_score: 0.96 }
        );
    }

    #[test]
    fn allow_meets_at_exact_threshold() {
        // allow exactly at threshold "meets" it; with allow(0.90) < deny(0.95) → flag.
        assert_eq!(
            semantic_decision(Some(0.90), Some(0.95), ALLOW_T, DENY_T, false),
            SemanticDecision::Flag { deny_score: 0.95 }
        );
        // allow at threshold and >= deny → permit.
        assert_eq!(
            semantic_decision(Some(0.90), Some(0.88), ALLOW_T, DENY_T, false),
            SemanticDecision::Permit
        );
    }

    // ------------------------------------------------------------------
    // Property 17: Semantic decision rule.
    //
    // **Validates: Requirements 7.3, 7.4, 7.8**
    //
    // For any pair of (allow_similarity, deny_similarity) scores against
    // configured thresholds:
    // - content is flagged as deny when the deny score strictly exceeds the
    //   deny threshold and no allow score meets its threshold (Req 7.3);
    // - when both scores exceed their thresholds, content is permitted iff
    //   allow_similarity >= deny_similarity (Req 7.4);
    // - when neither collection contains examples (both_empty), content is
    //   always permitted (Req 7.8).
    // ------------------------------------------------------------------
    use proptest::prelude::*;

    /// Independently-expressed oracle of the semantic decision rule. Written
    /// from the acceptance criteria (Req 7.3, 7.4, 7.8) rather than by reusing
    /// `semantic_decision`, so it can catch a regression in the implementation.
    ///
    /// Boundary semantics mirror the spec: deny is "exceeds" (strict `>`),
    /// allow "meets its threshold" (`>=`).
    fn oracle_decision(
        allow_similarity: Option<f32>,
        deny_similarity: Option<f32>,
        allow_threshold: f32,
        deny_threshold: f32,
        both_empty: bool,
    ) -> SemanticDecision {
        if both_empty {
            return SemanticDecision::Permit;
        }
        match deny_similarity {
            Some(deny) if deny > deny_threshold => {
                let allow_meets = matches!(allow_similarity, Some(a) if a >= allow_threshold);
                if allow_meets {
                    let allow = allow_similarity.unwrap();
                    if allow >= deny {
                        SemanticDecision::Permit
                    } else {
                        SemanticDecision::Flag { deny_score: deny }
                    }
                } else {
                    SemanticDecision::Flag { deny_score: deny }
                }
            }
            _ => SemanticDecision::Permit,
        }
    }

    /// Similarity score generator: `None` (no match in a collection) or a
    /// score in the valid cosine range `0.0..=1.0`.
    fn arb_similarity() -> impl Strategy<Value = Option<f32>> {
        prop_oneof![
            1 => Just(None),
            4 => (0.0f32..=1.0f32).prop_map(Some),
        ]
    }

    proptest! {
        #![proptest_config(ProptestConfig::with_cases(64))]

        #[test]
        fn prop_semantic_decision_matches_oracle(
            allow_similarity in arb_similarity(),
            deny_similarity in arb_similarity(),
            allow_threshold in 0.0f32..=1.0f32,
            deny_threshold in 0.0f32..=1.0f32,
            both_empty in any::<bool>(),
        ) {
            let actual = semantic_decision(
                allow_similarity,
                deny_similarity,
                allow_threshold,
                deny_threshold,
                both_empty,
            );
            let expected = oracle_decision(
                allow_similarity,
                deny_similarity,
                allow_threshold,
                deny_threshold,
                both_empty,
            );
            prop_assert_eq!(actual, expected);
        }

        /// Focused generator biased toward the decision boundaries: thresholds
        /// and scores drawn from a small shared grid so `==` boundary cases
        /// (deny strict `>` vs allow `>=`) are exercised frequently.
        #[test]
        fn prop_semantic_decision_boundaries(
            allow_idx in 0usize..=4,
            deny_idx in 0usize..=4,
            allow_t_idx in 0usize..=4,
            deny_t_idx in 0usize..=4,
            allow_none in any::<bool>(),
            deny_none in any::<bool>(),
            both_empty in any::<bool>(),
        ) {
            const GRID: [f32; 5] = [0.0, 0.85, 0.90, 0.95, 1.0];
            let allow_similarity = if allow_none { None } else { Some(GRID[allow_idx]) };
            let deny_similarity = if deny_none { None } else { Some(GRID[deny_idx]) };
            let allow_threshold = GRID[allow_t_idx];
            let deny_threshold = GRID[deny_t_idx];

            let actual = semantic_decision(
                allow_similarity,
                deny_similarity,
                allow_threshold,
                deny_threshold,
                both_empty,
            );
            let expected = oracle_decision(
                allow_similarity,
                deny_similarity,
                allow_threshold,
                deny_threshold,
                both_empty,
            );
            prop_assert_eq!(actual, expected);
        }
    }

    #[test]
    fn provider_type_is_semantic() {
        let provider = SemanticProvider::new(
            Arc::new(Qdrant::from_url("http://localhost:6334").build().unwrap()),
            Client::new(),
            "test-provider".to_string(),
            "text-embedding-3-small".to_string(),
            "http://localhost:8080/v1".to_string(),
            "test-key".to_string(),
            "guardrail_allow".to_string(),
            "guardrail_deny".to_string(),
            DEFAULT_ALLOW_THRESHOLD,
            DEFAULT_DENY_THRESHOLD,
        );
        assert_eq!(provider.provider_type(), "semantic");
    }
}
