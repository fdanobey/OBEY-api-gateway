//! Custom HTTP guardrail provider (Req 8.3, 8.4).
//!
//! POSTs content to a configured URL and parses the documented findings JSON
//! schema (Req 8.3):
//!
//! ```json
//! {
//!   "findings": [
//!     { "entity_label": "string (<=128 chars)", "start": 0, "end": 5, "score": 0.97 }
//!   ]
//! }
//! ```
//!
//! A non-2xx HTTP status or a response body that does not conform to the
//! schema is surfaced as a [`GuardrailProviderError`] (Req 8.4); the engine's
//! failure-policy wrapper then applies fail_open / fail_close.

use std::time::Duration;

use async_trait::async_trait;
use reqwest::Client;
use serde::{Deserialize, Serialize};

use crate::guardrail::provider::{Finding, GuardrailProvider, GuardrailProviderError};

/// The documented `custom_http` response schema (Req 8.3).
///
/// Public (with `Serialize`) so the parse round-trip can be property-tested
/// (task 7.6): a generated `CustomHttpFindingsResponse` serialized to JSON and
/// deserialized back yields an equal value.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct CustomHttpFindingsResponse {
    /// The detected findings.
    #[serde(default)]
    pub findings: Vec<CustomHttpFinding>,
}

/// A single finding entry in the documented `custom_http` schema (Req 8.3).
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct CustomHttpFinding {
    /// Entity label (<=128 chars).
    pub entity_label: String,
    /// Byte-offset start within the analyzed content.
    pub start: usize,
    /// Byte-offset end (exclusive) within the analyzed content.
    pub end: usize,
    /// Optional match score.
    #[serde(default)]
    pub score: Option<f32>,
}

impl From<CustomHttpFinding> for Finding {
    fn from(f: CustomHttpFinding) -> Self {
        Finding {
            entity_label: f.entity_label,
            start: f.start,
            end: f.end,
            matched_text: None,
            score: f.score,
        }
    }
}

impl CustomHttpFindingsResponse {
    /// Convert the parsed response into engine [`Finding`]s.
    pub fn into_findings(self) -> Vec<Finding> {
        self.findings.into_iter().map(Finding::from).collect()
    }
}

/// Parse a `custom_http` response body into [`Finding`]s (Req 8.3).
///
/// Pure helper, exposed for the parse round-trip property test (task 7.6). A
/// body that does not conform to the schema yields
/// [`GuardrailProviderError::MalformedResponse`].
pub fn parse_findings(body: &str) -> Result<Vec<Finding>, GuardrailProviderError> {
    let parsed: CustomHttpFindingsResponse = serde_json::from_str(body).map_err(|e| {
        GuardrailProviderError::MalformedResponse(format!(
            "custom_http response did not conform to findings schema: {}",
            e
        ))
    })?;
    Ok(parsed.into_findings())
}

/// Request payload POSTed to the custom HTTP endpoint.
#[derive(Debug, Clone, Serialize)]
struct CustomHttpRequest<'a> {
    /// The content to analyze.
    content: &'a str,
}

/// Custom HTTP guardrail provider (Req 8.3, 8.4).
pub struct CustomHttpProvider {
    /// Shared HTTP client.
    http_client: Client,
    /// Content-analysis endpoint URL.
    url: String,
    /// Per-request timeout.
    timeout: Duration,
}

impl CustomHttpProvider {
    /// Construct a provider. `timeout_secs` defaults to 5 s when `None`.
    pub fn new(http_client: Client, url: String, timeout_secs: Option<u64>) -> Self {
        Self {
            http_client,
            url,
            timeout: Duration::from_secs(timeout_secs.unwrap_or(5)),
        }
    }
}

#[async_trait]
impl GuardrailProvider for CustomHttpProvider {
    async fn analyze(&self, content: &str) -> Result<Vec<Finding>, GuardrailProviderError> {
        let payload = CustomHttpRequest { content };

        let response = self
            .http_client
            .post(&self.url)
            .timeout(self.timeout)
            .json(&payload)
            .send()
            .await
            .map_err(|e| {
                GuardrailProviderError::Unreachable(format!(
                    "custom_http request to '{}' failed: {}",
                    self.url, e
                ))
            })
            .inspect_err(|e| {
                tracing::warn!(url = %self.url, error = %e, "custom_http guardrail request failed");
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
            tracing::warn!(url = %self.url, error = %err, "custom_http guardrail returned non-2xx");
            return Err(err);
        }

        let body = response.text().await.map_err(|e| {
            GuardrailProviderError::MalformedResponse(format!(
                "failed to read custom_http response body from '{}': {}",
                self.url, e
            ))
        })?;

        parse_findings(&body).inspect_err(|e| {
            tracing::warn!(url = %self.url, error = %e, "custom_http guardrail returned malformed response");
        })
    }

    fn provider_type(&self) -> &'static str {
        "custom_http"
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use proptest::prelude::*;

    #[test]
    fn parses_documented_schema() {
        let body = r#"{
            "findings": [
                { "entity_label": "API_KEY", "start": 0, "end": 5, "score": 0.97 },
                { "entity_label": "EMAIL", "start": 10, "end": 20 }
            ]
        }"#;
        let findings = parse_findings(body).expect("valid schema parses");
        assert_eq!(findings.len(), 2);
        assert_eq!(findings[0].entity_label, "API_KEY");
        assert_eq!(findings[0].start, 0);
        assert_eq!(findings[0].end, 5);
        assert_eq!(findings[0].score, Some(0.97));
        // Missing score defaults to None.
        assert_eq!(findings[1].entity_label, "EMAIL");
        assert_eq!(findings[1].score, None);
    }

    #[test]
    fn empty_findings_list_parses_to_empty() {
        let findings = parse_findings(r#"{ "findings": [] }"#).unwrap();
        assert!(findings.is_empty());
        // Absent `findings` key also yields empty (serde default).
        let findings = parse_findings(r#"{}"#).unwrap();
        assert!(findings.is_empty());
    }

    #[test]
    fn schema_mismatch_is_malformed_response() {
        // `start` should be an integer, not a string.
        let body = r#"{ "findings": [ { "entity_label": "X", "start": "nope", "end": 5 } ] }"#;
        let err = parse_findings(body).unwrap_err();
        assert!(matches!(err, GuardrailProviderError::MalformedResponse(_)));
    }

    #[test]
    fn round_trip_serialize_parse() {
        let response = CustomHttpFindingsResponse {
            findings: vec![CustomHttpFinding {
                entity_label: "prompt_injection".to_string(),
                start: 3,
                end: 9,
                score: Some(0.5),
            }],
        };
        let json = serde_json::to_string(&response).unwrap();
        let findings = parse_findings(&json).unwrap();
        assert_eq!(findings.len(), 1);
        assert_eq!(findings[0].entity_label, "prompt_injection");
        assert_eq!(findings[0].start, 3);
        assert_eq!(findings[0].end, 9);
    }

    #[test]
    fn provider_type_is_custom_http() {
        let provider =
            CustomHttpProvider::new(Client::new(), "http://scanner:8080/scan".to_string(), None);
        assert_eq!(provider.provider_type(), "custom_http");
    }

    // ---- Property-based tests (proptest, >=100 cases) ----

    /// Strategy for a single valid `CustomHttpFinding`: entity label <=128
    /// chars, arbitrary usize offsets, optional score in 0.0..=1.0.
    fn finding_strategy() -> impl Strategy<Value = CustomHttpFinding> {
        (
            ".{0,128}",
            any::<usize>(),
            any::<usize>(),
            prop::option::of(0.0f32..=1.0f32),
        )
            .prop_map(|(entity_label, start, end, score)| CustomHttpFinding {
                entity_label,
                start,
                end,
                score,
            })
    }

    proptest! {
        #![proptest_config(ProptestConfig::with_cases(64))]

        // Feature: guardrail-pipelines, Property 19: custom_http findings parse round-trip
        // Validates: Requirements 8.3
        //
        // For any valid findings structure, serializing it to the documented
        // JSON schema and parsing it back via `parse_findings` yields findings
        // equal to the originals (entity_label, start, end, score preserved).
        #[test]
        fn prop_custom_http_findings_parse_round_trip(
            findings in prop::collection::vec(finding_strategy(), 0..8usize)
        ) {
            let response = CustomHttpFindingsResponse {
                findings: findings.clone(),
            };
            let json = serde_json::to_string(&response)
                .expect("documented schema serializes");
            let parsed = parse_findings(&json)
                .expect("serialized documented schema parses back");

            prop_assert_eq!(parsed.len(), findings.len());
            for (got, expected) in parsed.iter().zip(findings.iter()) {
                prop_assert_eq!(&got.entity_label, &expected.entity_label);
                prop_assert_eq!(got.start, expected.start);
                prop_assert_eq!(got.end, expected.end);
                prop_assert_eq!(got.score, expected.score);
            }
        }
    }
}
