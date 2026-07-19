//! OpenAI Moderation guardrail provider (Req 8.2).
//!
//! Translates content to/from the OpenAI Moderation API and produces one
//! [`Finding`] per flagged category, using the flagged category name as the
//! entity label (Req 8.2). The API key is resolved from the configured
//! `api_key_env`: it is first tried as an environment-variable name and, if
//! that variable is unset, used as a literal value — matching the repo's
//! `api_key_env` convention (see `router.rs` API-key resolution).
//!
//! A non-2xx status or malformed body surfaces as a [`GuardrailProviderError`];
//! the engine's failure-policy wrapper then applies fail_open / fail_close.

use std::collections::BTreeMap;
use std::time::Duration;

use async_trait::async_trait;
use reqwest::Client;
use serde::{Deserialize, Serialize};

use crate::guardrail::provider::{Finding, GuardrailProvider, GuardrailProviderError};

/// Default OpenAI Moderation endpoint.
pub const DEFAULT_ENDPOINT: &str = "https://api.openai.com/v1/moderations";

/// Default moderation model.
pub const DEFAULT_MODEL: &str = "omni-moderation-latest";

/// Resolve an API key from an `api_key_env` value, matching the repo
/// convention: try it as an environment-variable name first, and fall back to
/// treating the value itself as the literal key when the variable is unset.
pub fn resolve_api_key(api_key_env: &str) -> String {
    std::env::var(api_key_env).unwrap_or_else(|_| api_key_env.to_string())
}

/// A single result entry in the OpenAI Moderation response.
#[derive(Debug, Clone, Deserialize)]
struct ModerationResult {
    /// Per-category boolean flags.
    #[serde(default)]
    categories: BTreeMap<String, bool>,
    /// Per-category confidence scores.
    #[serde(default)]
    category_scores: BTreeMap<String, f32>,
}

/// The OpenAI Moderation response body.
#[derive(Debug, Clone, Deserialize)]
struct ModerationResponse {
    #[serde(default)]
    results: Vec<ModerationResult>,
}

/// Request payload sent to the moderation endpoint.
#[derive(Debug, Clone, Serialize)]
struct ModerationRequest<'a> {
    input: &'a str,
    model: &'a str,
}

/// Map flagged categories to [`Finding`]s (Req 8.2).
///
/// One finding is produced per category flagged `true`, with the category name
/// as `entity_label` and the corresponding `category_scores` value (if present)
/// as `score`. The span covers the whole analyzed content.
///
/// Pure helper, exposed for unit testing (task 7.9).
pub fn categories_to_findings(
    categories: &BTreeMap<String, bool>,
    category_scores: &BTreeMap<String, f32>,
    content_len: usize,
) -> Vec<Finding> {
    categories
        .iter()
        .filter(|(_, flagged)| **flagged)
        .map(|(name, _)| Finding {
            entity_label: name.clone(),
            start: 0,
            end: content_len,
            matched_text: None,
            score: category_scores.get(name).copied(),
        })
        .collect()
}

/// OpenAI Moderation guardrail provider.
pub struct OpenAiModerationProvider {
    /// Shared HTTP client.
    http_client: Client,
    /// Moderation endpoint URL.
    endpoint: String,
    /// Moderation model identifier.
    model: String,
    /// Resolved API key (see [`resolve_api_key`]).
    api_key: String,
    /// Per-request timeout.
    timeout: Duration,
}

impl OpenAiModerationProvider {
    /// Construct a provider.
    ///
    /// `api_key_env` is resolved via [`resolve_api_key`]. `endpoint`/`model`
    /// default to [`DEFAULT_ENDPOINT`]/[`DEFAULT_MODEL`] when `None`.
    /// `timeout_secs` defaults to 5 s.
    pub fn new(
        http_client: Client,
        endpoint: Option<String>,
        model: Option<String>,
        api_key_env: &str,
        timeout_secs: Option<u64>,
    ) -> Self {
        Self {
            http_client,
            endpoint: endpoint.unwrap_or_else(|| DEFAULT_ENDPOINT.to_string()),
            model: model.unwrap_or_else(|| DEFAULT_MODEL.to_string()),
            api_key: resolve_api_key(api_key_env),
            timeout: Duration::from_secs(timeout_secs.unwrap_or(5)),
        }
    }
}

#[async_trait]
impl GuardrailProvider for OpenAiModerationProvider {
    async fn analyze(&self, content: &str) -> Result<Vec<Finding>, GuardrailProviderError> {
        let payload = ModerationRequest {
            input: content,
            model: &self.model,
        };

        let response = self
            .http_client
            .post(&self.endpoint)
            .timeout(self.timeout)
            .header("Authorization", format!("Bearer {}", self.api_key))
            .json(&payload)
            .send()
            .await
            .map_err(|e| {
                GuardrailProviderError::Unreachable(format!(
                    "openai_moderation request to '{}' failed: {}",
                    self.endpoint, e
                ))
            })
            .inspect_err(|e| {
                tracing::warn!(endpoint = %self.endpoint, error = %e, "openai_moderation guardrail request failed");
            })?;

        let status = response.status();
        if !status.is_success() {
            let message = response
                .text()
                .await
                .unwrap_or_else(|_| "unknown error".to_string());
            let err = GuardrailProviderError::UpstreamStatus {
                status: status.as_u16(),
                message,
            };
            tracing::warn!(endpoint = %self.endpoint, error = %err, "openai_moderation guardrail returned non-2xx");
            return Err(err);
        }

        let parsed: ModerationResponse = response.json().await.map_err(|e| {
            let err = GuardrailProviderError::MalformedResponse(format!(
                "failed to parse openai_moderation response from '{}': {}",
                self.endpoint, e
            ));
            tracing::warn!(endpoint = %self.endpoint, error = %err, "openai_moderation guardrail returned malformed response");
            err
        })?;

        let mut findings = Vec::new();
        for result in &parsed.results {
            findings.extend(categories_to_findings(
                &result.categories,
                &result.category_scores,
                content.len(),
            ));
        }
        Ok(findings)
    }

    fn provider_type(&self) -> &'static str {
        "openai_moderation"
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn map_bool(pairs: &[(&str, bool)]) -> BTreeMap<String, bool> {
        pairs.iter().map(|(k, v)| (k.to_string(), *v)).collect()
    }

    fn map_score(pairs: &[(&str, f32)]) -> BTreeMap<String, f32> {
        pairs.iter().map(|(k, v)| (k.to_string(), *v)).collect()
    }

    #[test]
    fn maps_only_flagged_categories() {
        let categories = map_bool(&[("hate", true), ("violence", false), ("sexual", true)]);
        let scores = map_score(&[("hate", 0.9), ("sexual", 0.7)]);
        let findings = categories_to_findings(&categories, &scores, 42);

        // BTreeMap iteration is sorted: "hate" then "sexual".
        assert_eq!(findings.len(), 2);
        assert_eq!(findings[0].entity_label, "hate");
        assert_eq!(findings[0].score, Some(0.9));
        assert_eq!(findings[0].start, 0);
        assert_eq!(findings[0].end, 42);
        assert_eq!(findings[1].entity_label, "sexual");
        assert_eq!(findings[1].score, Some(0.7));
    }

    #[test]
    fn no_flagged_categories_yields_no_findings() {
        let categories = map_bool(&[("hate", false), ("violence", false)]);
        let findings = categories_to_findings(&categories, &BTreeMap::new(), 10);
        assert!(findings.is_empty());
    }

    #[test]
    fn flagged_category_without_score_maps_with_none() {
        let categories = map_bool(&[("harassment", true)]);
        let findings = categories_to_findings(&categories, &BTreeMap::new(), 5);
        assert_eq!(findings.len(), 1);
        assert_eq!(findings[0].entity_label, "harassment");
        assert_eq!(findings[0].score, None);
    }

    #[test]
    fn resolve_api_key_falls_back_to_literal() {
        // An almost-certainly-unset env var name falls back to the literal
        // value, matching the repo's api_key_env convention.
        let literal = "sk-literal-key-value-OBEY-UNSET";
        assert_eq!(resolve_api_key(literal), literal);
    }

    #[test]
    fn provider_type_is_openai_moderation() {
        let provider = OpenAiModerationProvider::new(Client::new(), None, None, "sk-test", None);
        assert_eq!(provider.provider_type(), "openai_moderation");
        assert_eq!(provider.endpoint, DEFAULT_ENDPOINT);
        assert_eq!(provider.model, DEFAULT_MODEL);
    }

    #[test]
    fn flagged_category_carries_score_through_when_present() {
        // A flagged category with a matching score entry carries that score
        // through to the finding (Req 8.2).
        let categories = map_bool(&[("self_harm", true)]);
        let scores = map_score(&[("self_harm", 0.42)]);
        let findings = categories_to_findings(&categories, &scores, 7);
        assert_eq!(findings.len(), 1);
        assert_eq!(findings[0].entity_label, "self_harm");
        assert_eq!(findings[0].score, Some(0.42));
    }

    /// Registry sharing (Req 8.5): a provider inserted as
    /// `Arc<dyn GuardrailProvider>` is shared across concurrent-request lookups
    /// — every `get()` returns a clone of the same underlying `Arc`.
    #[test]
    fn registry_shares_provider_instance_across_lookups() {
        use crate::guardrail::provider::{GuardrailProvider, ProviderRegistry};
        use std::sync::Arc;

        let mut registry = ProviderRegistry::new();
        let provider: Arc<dyn GuardrailProvider> = Arc::new(OpenAiModerationProvider::new(
            Client::new(),
            None,
            None,
            "sk-test",
            None,
        ));
        // strong_count is 1 while only the local binding holds it.
        assert_eq!(Arc::strong_count(&provider), 1);
        registry.insert("moderation", provider);

        let a = registry.get("moderation").expect("registered");
        let b = registry.get("moderation").expect("registered");

        // Both lookups resolve to the same instance shared behind an Arc.
        assert!(Arc::ptr_eq(&a, &b));
        assert_eq!(a.provider_type(), "openai_moderation");

        // The registry retains its own reference plus the two clones held here,
        // demonstrating a single shared instance rather than per-lookup copies.
        assert!(Arc::strong_count(&a) >= 3);
    }
}
