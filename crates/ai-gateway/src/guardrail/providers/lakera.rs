//! Lakera Guard guardrail provider (Req 8.2).
//!
//! Translates content to/from the Lakera Guard API and produces one
//! [`Finding`] per flagged category, using the flagged category name as the
//! entity label (Req 8.2). The API key is resolved from the configured
//! `api_key_env`: it is tried as an environment-variable name first and, if
//! unset, used as a literal value — matching the repo's `api_key_env`
//! convention (see `router.rs` API-key resolution).
//!
//! A non-2xx status or malformed body surfaces as a [`GuardrailProviderError`];
//! the engine's failure-policy wrapper then applies fail_open / fail_close.

use std::collections::BTreeMap;
use std::time::Duration;

use async_trait::async_trait;
use reqwest::Client;
use serde::{Deserialize, Serialize};

use crate::guardrail::provider::{Finding, GuardrailProvider, GuardrailProviderError};

/// Default Lakera Guard endpoint.
pub const DEFAULT_ENDPOINT: &str = "https://api.lakera.ai/v2/guard";

/// Resolve an API key from an `api_key_env` value: try it as an
/// environment-variable name first, falling back to the literal value when the
/// variable is unset (repo `api_key_env` convention).
pub fn resolve_api_key(api_key_env: &str) -> String {
    std::env::var(api_key_env).unwrap_or_else(|_| api_key_env.to_string())
}

/// A single result entry in the Lakera Guard response.
#[derive(Debug, Clone, Deserialize)]
struct LakeraResult {
    /// Per-category boolean flags (e.g. `prompt_injection`, `jailbreak`).
    #[serde(default)]
    categories: BTreeMap<String, bool>,
    /// Per-category confidence scores.
    #[serde(default)]
    category_scores: BTreeMap<String, f32>,
}

/// The Lakera Guard response body.
#[derive(Debug, Clone, Deserialize)]
struct LakeraResponse {
    #[serde(default)]
    results: Vec<LakeraResult>,
}

/// Request payload sent to the Lakera Guard endpoint.
#[derive(Debug, Clone, Serialize)]
struct LakeraRequest<'a> {
    input: &'a str,
}

/// Map flagged Lakera categories to [`Finding`]s (Req 8.2).
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

/// Lakera Guard guardrail provider.
pub struct LakeraProvider {
    /// Shared HTTP client.
    http_client: Client,
    /// Lakera Guard endpoint URL.
    endpoint: String,
    /// Resolved API key (see [`resolve_api_key`]).
    api_key: String,
    /// Per-request timeout.
    timeout: Duration,
}

impl LakeraProvider {
    /// Construct a provider.
    ///
    /// `api_key_env` is resolved via [`resolve_api_key`]. `endpoint` defaults to
    /// [`DEFAULT_ENDPOINT`] when `None`. `timeout_secs` defaults to 5 s.
    pub fn new(
        http_client: Client,
        endpoint: Option<String>,
        api_key_env: &str,
        timeout_secs: Option<u64>,
    ) -> Self {
        Self {
            http_client,
            endpoint: endpoint.unwrap_or_else(|| DEFAULT_ENDPOINT.to_string()),
            api_key: resolve_api_key(api_key_env),
            timeout: Duration::from_secs(timeout_secs.unwrap_or(5)),
        }
    }
}

#[async_trait]
impl GuardrailProvider for LakeraProvider {
    async fn analyze(&self, content: &str) -> Result<Vec<Finding>, GuardrailProviderError> {
        let payload = LakeraRequest { input: content };

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
                    "lakera request to '{}' failed: {}",
                    self.endpoint, e
                ))
            })
            .inspect_err(|e| {
                tracing::warn!(endpoint = %self.endpoint, error = %e, "lakera guardrail request failed");
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
            tracing::warn!(endpoint = %self.endpoint, error = %err, "lakera guardrail returned non-2xx");
            return Err(err);
        }

        let parsed: LakeraResponse = response.json().await.map_err(|e| {
            let err = GuardrailProviderError::MalformedResponse(format!(
                "failed to parse lakera response from '{}': {}",
                self.endpoint, e
            ));
            tracing::warn!(endpoint = %self.endpoint, error = %err, "lakera guardrail returned malformed response");
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
        "lakera"
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
    fn maps_flagged_categories_as_entity_labels() {
        let categories = map_bool(&[("jailbreak", false), ("prompt_injection", true)]);
        let scores = map_score(&[("prompt_injection", 0.99)]);
        let findings = categories_to_findings(&categories, &scores, 30);

        assert_eq!(findings.len(), 1);
        assert_eq!(findings[0].entity_label, "prompt_injection");
        assert_eq!(findings[0].score, Some(0.99));
        assert_eq!(findings[0].start, 0);
        assert_eq!(findings[0].end, 30);
    }

    #[test]
    fn no_flagged_categories_yields_no_findings() {
        let categories = map_bool(&[("prompt_injection", false)]);
        let findings = categories_to_findings(&categories, &BTreeMap::new(), 12);
        assert!(findings.is_empty());
    }

    #[test]
    fn resolve_api_key_falls_back_to_literal() {
        let literal = "lakera-literal-key-OBEY-UNSET";
        assert_eq!(resolve_api_key(literal), literal);
    }

    #[test]
    fn provider_type_is_lakera() {
        let provider = LakeraProvider::new(Client::new(), None, "lakera-key", None);
        assert_eq!(provider.provider_type(), "lakera");
        assert_eq!(provider.endpoint, DEFAULT_ENDPOINT);
    }

    #[test]
    fn flagged_category_without_score_maps_with_none() {
        // A flagged category with no matching score entry yields a finding with
        // no score, while the category name is preserved as the entity label.
        let categories = map_bool(&[("jailbreak", true)]);
        let findings = categories_to_findings(&categories, &BTreeMap::new(), 8);
        assert_eq!(findings.len(), 1);
        assert_eq!(findings[0].entity_label, "jailbreak");
        assert_eq!(findings[0].score, None);
        assert_eq!(findings[0].end, 8);
    }

    /// Registry sharing (Req 8.5): a Lakera provider inserted as
    /// `Arc<dyn GuardrailProvider>` is shared across concurrent-request lookups.
    #[test]
    fn registry_shares_provider_instance_across_lookups() {
        use crate::guardrail::provider::{GuardrailProvider, ProviderRegistry};
        use std::sync::Arc;

        let mut registry = ProviderRegistry::new();
        let provider: Arc<dyn GuardrailProvider> =
            Arc::new(LakeraProvider::new(Client::new(), None, "lakera-key", None));
        registry.insert("lakera", provider);

        let a = registry.get("lakera").expect("registered");
        let b = registry.get("lakera").expect("registered");

        assert!(Arc::ptr_eq(&a, &b));
        assert_eq!(a.provider_type(), "lakera");
    }
}
