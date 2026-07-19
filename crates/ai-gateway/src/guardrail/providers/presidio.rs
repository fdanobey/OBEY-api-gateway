//! Presidio-compatible guardrail provider (Req 6).
//!
//! POSTs `{ text, entities }` to a configurable Presidio-compatible HTTP
//! endpoint (Req 6.1) and maps returned entities into [`Finding`]s. Only
//! entities whose type appears in the configured entity list are mapped; every
//! other detected entity is ignored (Req 6.2). Entities whose confidence score
//! falls below the configured threshold (default 0.5, range 0.0–1.0) are
//! discarded (Req 6.6). The request timeout is clamped to the inclusive range
//! [1 s, 30 s] with a 5 s default (Req 6.5). Unreachable / non-2xx / malformed
//! schema / timeout all surface as a [`GuardrailProviderError`] with a WARN
//! log; the engine's failure-policy wrapper then applies fail_open / fail_close
//! (Req 6.4).

use std::collections::HashSet;
use std::time::Duration;

use async_trait::async_trait;
use reqwest::Client;
use serde::{Deserialize, Serialize};

use crate::guardrail::provider::{Finding, GuardrailProvider, GuardrailProviderError};

/// Minimum allowed Presidio request timeout in seconds (Req 6.5).
pub const MIN_TIMEOUT_SECS: u64 = 1;

/// Maximum allowed Presidio request timeout in seconds (Req 6.5).
pub const MAX_TIMEOUT_SECS: u64 = 30;

/// Default Presidio request timeout in seconds (Req 6.5).
pub const DEFAULT_TIMEOUT_SECS: u64 = 5;

/// Default minimum confidence score threshold (Req 6.6).
pub const DEFAULT_CONFIDENCE_THRESHOLD: f32 = 0.5;

/// Clamp a configured timeout (in seconds) into the inclusive range
/// [[`MIN_TIMEOUT_SECS`], [`MAX_TIMEOUT_SECS`]], defaulting to
/// [`DEFAULT_TIMEOUT_SECS`] when no value is supplied (Req 6.5).
///
/// This is a pure helper so it can be unit- and property-tested without any
/// network dependency.
pub fn clamp_timeout(configured_secs: Option<u64>) -> Duration {
    let secs = configured_secs
        .unwrap_or(DEFAULT_TIMEOUT_SECS)
        .clamp(MIN_TIMEOUT_SECS, MAX_TIMEOUT_SECS);
    Duration::from_secs(secs)
}

/// A single entity as returned by a Presidio-compatible `/analyze` endpoint.
///
/// Public so entity-filtering can be property-tested (task 7.2) against
/// deserialized results.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct PresidioEntity {
    /// Detected entity type (e.g. `EMAIL_ADDRESS`, `US_SSN`).
    pub entity_type: String,
    /// Byte-offset start within the analyzed text.
    pub start: usize,
    /// Byte-offset end (exclusive) within the analyzed text.
    pub end: usize,
    /// Confidence score reported by the Presidio service (0.0–1.0).
    pub score: f32,
}

/// Request payload sent to the Presidio `/analyze` endpoint (Req 6.1).
#[derive(Debug, Clone, Serialize)]
struct PresidioRequest<'a> {
    /// The content to analyze.
    text: &'a str,
    /// The configured entity types to detect.
    entities: &'a [String],
}

/// Filter a set of detected Presidio entities into [`Finding`]s (Req 6.2, 6.6).
///
/// A detected entity becomes a finding **iff** its `entity_type` is in
/// `configured_entities` **and** its `score` is greater than or equal to
/// `threshold`. All other entities are discarded.
///
/// Pure helper, exposed for unit/property testing (task 7.2).
pub fn filter_entities(
    entities: Vec<PresidioEntity>,
    configured_entities: &HashSet<String>,
    threshold: f32,
) -> Vec<Finding> {
    entities
        .into_iter()
        .filter(|e| configured_entities.contains(&e.entity_type) && e.score >= threshold)
        .map(|e| Finding {
            entity_label: e.entity_type,
            start: e.start,
            end: e.end,
            matched_text: None,
            score: Some(e.score),
        })
        .collect()
}

/// Presidio-compatible guardrail provider backed by an HTTP endpoint.
pub struct PresidioProvider {
    /// Shared HTTP client.
    http_client: Client,
    /// Presidio-compatible `/analyze` endpoint URL.
    endpoint: String,
    /// Entity types to detect and map (Req 6.3); others are ignored.
    entities: Vec<String>,
    /// Fast membership lookup for `entities`.
    entity_set: HashSet<String>,
    /// Minimum confidence score threshold (Req 6.6).
    confidence_threshold: f32,
    /// Per-request timeout, clamped to [1 s, 30 s] (Req 6.5).
    timeout: Duration,
}

impl PresidioProvider {
    /// Construct a provider.
    ///
    /// `confidence_threshold` defaults to [`DEFAULT_CONFIDENCE_THRESHOLD`] when
    /// `None` and is clamped into 0.0–1.0. `timeout_secs` is clamped via
    /// [`clamp_timeout`].
    pub fn new(
        http_client: Client,
        endpoint: String,
        entities: Vec<String>,
        confidence_threshold: Option<f32>,
        timeout_secs: Option<u64>,
    ) -> Self {
        let entity_set = entities.iter().cloned().collect();
        let confidence_threshold = confidence_threshold
            .unwrap_or(DEFAULT_CONFIDENCE_THRESHOLD)
            .clamp(0.0, 1.0);
        Self {
            http_client,
            endpoint,
            entities,
            entity_set,
            confidence_threshold,
            timeout: clamp_timeout(timeout_secs),
        }
    }
}

#[async_trait]
impl GuardrailProvider for PresidioProvider {
    async fn analyze(&self, content: &str) -> Result<Vec<Finding>, GuardrailProviderError> {
        let payload = PresidioRequest {
            text: content,
            entities: &self.entities,
        };

        let response = self
            .http_client
            .post(&self.endpoint)
            .timeout(self.timeout)
            .json(&payload)
            .send()
            .await
            .map_err(|e| {
                GuardrailProviderError::Unreachable(format!(
                    "Presidio request to '{}' failed: {}",
                    self.endpoint, e
                ))
            })
            .inspect_err(|e| {
                tracing::warn!(endpoint = %self.endpoint, error = %e, "presidio guardrail request failed");
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
            tracing::warn!(endpoint = %self.endpoint, error = %err, "presidio guardrail returned non-2xx");
            return Err(err);
        }

        let entities: Vec<PresidioEntity> = response.json().await.map_err(|e| {
            let err = GuardrailProviderError::MalformedResponse(format!(
                "failed to parse Presidio response from '{}': {}",
                self.endpoint, e
            ));
            tracing::warn!(endpoint = %self.endpoint, error = %err, "presidio guardrail returned malformed response");
            err
        })?;

        Ok(filter_entities(
            entities,
            &self.entity_set,
            self.confidence_threshold,
        ))
    }

    fn provider_type(&self) -> &'static str {
        "presidio"
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use proptest::prelude::*;

    fn configured(entities: &[&str]) -> HashSet<String> {
        entities.iter().map(|s| s.to_string()).collect()
    }

    fn entity(entity_type: &str, start: usize, end: usize, score: f32) -> PresidioEntity {
        PresidioEntity {
            entity_type: entity_type.to_string(),
            start,
            end,
            score,
        }
    }

    #[test]
    fn filter_keeps_configured_entities_above_threshold() {
        let entities = vec![
            entity("EMAIL_ADDRESS", 0, 5, 0.9),
            entity("US_SSN", 6, 15, 0.6),
        ];
        let findings = filter_entities(entities, &configured(&["EMAIL_ADDRESS", "US_SSN"]), 0.5);
        assert_eq!(findings.len(), 2);
        assert_eq!(findings[0].entity_label, "EMAIL_ADDRESS");
        assert_eq!(findings[0].score, Some(0.9));
        assert_eq!(findings[1].entity_label, "US_SSN");
    }

    #[test]
    fn filter_ignores_entities_not_in_configured_list() {
        // PERSON is detected but not configured → ignored (Req 6.2).
        let entities = vec![
            entity("PERSON", 0, 4, 0.99),
            entity("EMAIL_ADDRESS", 5, 10, 0.8),
        ];
        let findings = filter_entities(entities, &configured(&["EMAIL_ADDRESS"]), 0.5);
        assert_eq!(findings.len(), 1);
        assert_eq!(findings[0].entity_label, "EMAIL_ADDRESS");
    }

    #[test]
    fn filter_discards_below_threshold() {
        // Score below threshold is dropped; at-threshold is kept (>=) (Req 6.6).
        let entities = vec![
            entity("EMAIL_ADDRESS", 0, 5, 0.49),
            entity("EMAIL_ADDRESS", 6, 11, 0.50),
        ];
        let findings = filter_entities(entities, &configured(&["EMAIL_ADDRESS"]), 0.5);
        assert_eq!(findings.len(), 1);
        assert_eq!(findings[0].start, 6);
    }

    #[test]
    fn clamp_timeout_defaults_and_bounds() {
        assert_eq!(
            clamp_timeout(None),
            Duration::from_secs(DEFAULT_TIMEOUT_SECS)
        );
        // Below minimum clamps up to 1 s.
        assert_eq!(
            clamp_timeout(Some(0)),
            Duration::from_secs(MIN_TIMEOUT_SECS)
        );
        // Above maximum clamps down to 30 s.
        assert_eq!(
            clamp_timeout(Some(120)),
            Duration::from_secs(MAX_TIMEOUT_SECS)
        );
        // In-range value is preserved.
        assert_eq!(clamp_timeout(Some(10)), Duration::from_secs(10));
    }

    #[test]
    fn provider_type_is_presidio() {
        let provider = PresidioProvider::new(
            Client::new(),
            "http://presidio:3000/analyze".to_string(),
            vec!["EMAIL_ADDRESS".to_string()],
            None,
            None,
        );
        assert_eq!(provider.provider_type(), "presidio");
        assert_eq!(provider.confidence_threshold, DEFAULT_CONFIDENCE_THRESHOLD);
        assert_eq!(provider.timeout, Duration::from_secs(DEFAULT_TIMEOUT_SECS));
    }

    // ---- Property-based tests (proptest, >=100 cases) ----

    /// Candidate entity-type names; a subset is configured per case so both
    /// in-list and out-of-list types are exercised.
    const ENTITY_POOL: &[&str] = &[
        "EMAIL_ADDRESS",
        "US_SSN",
        "PERSON",
        "CREDIT_CARD",
        "PHONE_NUMBER",
        "IP_ADDRESS",
    ];

    /// Independent oracle for entity filtering (Property 15). Re-expresses the
    /// keep-iff rule without reusing `filter_entities` so the test is a genuine
    /// cross-check: keep an entity iff its type is configured AND its score is
    /// at or above the threshold, preserving input order.
    fn oracle_filter(
        entities: &[PresidioEntity],
        configured_entities: &HashSet<String>,
        threshold: f32,
    ) -> Vec<Finding> {
        let mut out = Vec::new();
        for e in entities {
            let type_configured = configured_entities.contains(&e.entity_type);
            let meets_threshold = e.score >= threshold;
            if type_configured && meets_threshold {
                out.push(Finding {
                    entity_label: e.entity_type.clone(),
                    start: e.start,
                    end: e.end,
                    matched_text: None,
                    score: Some(e.score),
                });
            }
        }
        out
    }

    /// Strategy producing a random detected entity: a type drawn from the pool,
    /// byte offsets, and a confidence score in the inclusive range 0.0..=1.0.
    fn entity_strategy() -> impl Strategy<Value = PresidioEntity> {
        (
            0usize..ENTITY_POOL.len(),
            0usize..1000,
            0usize..1000,
            0.0f32..=1.0f32,
        )
            .prop_map(|(type_idx, start, end, score)| PresidioEntity {
                entity_type: ENTITY_POOL[type_idx].to_string(),
                start,
                end,
                score,
            })
    }

    proptest! {
        #![proptest_config(ProptestConfig::with_cases(256))]

        /// Property 15: Presidio entity filtering and confidence threshold.
        /// For any parsed Presidio result, a detected entity becomes a finding
        /// iff its type is in the configured entity list AND its confidence is
        /// at or above the configured threshold; all others are discarded.
        /// Validated against an independent oracle.
        /// Validates: Requirements 6.2, 6.6
        #[test]
        fn prop_filter_entities_matches_oracle(
            entities in prop::collection::vec(entity_strategy(), 0..24),
            configured_idx in prop::collection::hash_set(0usize..ENTITY_POOL.len(), 0..=ENTITY_POOL.len()),
            threshold in 0.0f32..=1.0f32,
        ) {
            let configured_entities: HashSet<String> = configured_idx
                .iter()
                .map(|&i| ENTITY_POOL[i].to_string())
                .collect();

            let expected = oracle_filter(&entities, &configured_entities, threshold);
            let actual = filter_entities(entities.clone(), &configured_entities, threshold);

            prop_assert_eq!(actual.clone(), expected);

            // Reinforce the iff invariant directly against every input entity.
            for e in &entities {
                let should_keep =
                    configured_entities.contains(&e.entity_type) && e.score >= threshold;
                let kept = actual.iter().any(|f| {
                    f.entity_label == e.entity_type
                        && f.start == e.start
                        && f.end == e.end
                        && f.score == Some(e.score)
                });
                if should_keep {
                    prop_assert!(kept, "configured, at-or-above-threshold entity must be kept");
                }
            }

            // Every finding is justified by the keep rule (nothing spurious).
            for f in &actual {
                prop_assert!(
                    configured_entities.contains(&f.entity_label),
                    "finding type must be configured"
                );
                prop_assert!(
                    f.score.unwrap() >= threshold,
                    "finding score must be at or above threshold"
                );
            }
        }

        /// Property 16: Presidio timeout clamping. For any configured timeout,
        /// the effective timeout is clamped into [1 s, 30 s]; an absent value
        /// yields the 5 s default.
        /// Validates: Requirements 6.5
        #[test]
        fn prop_clamp_timeout_within_bounds(configured in prop::option::of(any::<u64>())) {
            let effective = clamp_timeout(configured);
            let secs = effective.as_secs();

            prop_assert!(
                (MIN_TIMEOUT_SECS..=MAX_TIMEOUT_SECS).contains(&secs),
                "effective timeout must be within [1 s, 30 s]"
            );

            let expected = match configured {
                None => DEFAULT_TIMEOUT_SECS,
                Some(v) => v.clamp(MIN_TIMEOUT_SECS, MAX_TIMEOUT_SECS),
            };
            prop_assert_eq!(secs, expected);
        }
    }
}
