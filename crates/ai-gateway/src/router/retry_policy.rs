//! Unified retry policy table shared by the buffered and streaming
//! pass-through dispatch stacks.

use crate::error::GatewayError;

/// What a dispatch loop should do with a failed attempt against one
/// provider candidate.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum RetryDecision {
    /// Apply `max_retries_per_provider` with jittered backoff (500, 502,
    /// 408, network/body-read errors).
    RetrySameProvider,
    /// This candidate contributes no more attempts; advance to the next
    /// candidate (timeout class per DD10, 429, 503, all other 4xx). When no
    /// candidate remains, return the error to the client per each stack's
    /// existing semantics.
    FailOver,
}

/// Classify a failed attempt with the single policy table both dispatch
/// stacks use.
///
/// Pure error-class classification only: no I/O, no config reads, no
/// locks. Mechanism-level one-shots (stream-param rejection retry, image
/// strip, OAuth refresh) and the bounded primary Retry-After wait remain
/// caller-owned decisions layered on top of [`RetryDecision::FailOver`].
pub fn classify(err: &GatewayError) -> RetryDecision {
    match err {
        GatewayError::Provider { status_code, .. } => match status_code {
            Some(504 | 524) => RetryDecision::FailOver,
            Some(429 | 503) => RetryDecision::FailOver,
            Some(408) => RetryDecision::RetrySameProvider,
            Some(code) if (400..500).contains(code) => RetryDecision::FailOver,
            _ => RetryDecision::RetrySameProvider,
        },
        GatewayError::Network(_) | GatewayError::Http(_) => RetryDecision::RetrySameProvider,
        GatewayError::TtfbTimeout(_)
        | GatewayError::TotalTimeout(_)
        | GatewayError::ChunkTimeout(_) => RetryDecision::FailOver,
        GatewayError::RateLimitExceeded(_)
        | GatewayError::SmartRoutingBudgetExceeded { .. } => RetryDecision::FailOver,
        GatewayError::CircuitBreakerOpen(_) | GatewayError::AllProvidersFailed(_) => {
            RetryDecision::FailOver
        }
        GatewayError::Configuration(_)
        | GatewayError::Database(_)
        | GatewayError::Cache(_)
        | GatewayError::Serialization(_) => RetryDecision::FailOver,
        GatewayError::InvalidRequest(_) | GatewayError::Authentication(_) => {
            RetryDecision::FailOver
        }
        GatewayError::ContextCapacityExceeded { .. } => RetryDecision::FailOver,
        GatewayError::GuardrailPolicyViolation { .. }
        | GatewayError::GuardrailInvalidAction
        | GatewayError::GuardrailUnavailable(_) => RetryDecision::FailOver,
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::error::AggregatedError;

    fn provider(status_code: Option<u16>) -> GatewayError {
        GatewayError::Provider {
            provider: "provider-a".to_string(),
            message: "boom".to_string(),
            status_code,
        }
    }

    #[test]
    fn timeout_class_fails_over_without_same_provider_retry() {
        assert_eq!(classify(&GatewayError::TtfbTimeout(30)), RetryDecision::FailOver);
        assert_eq!(classify(&GatewayError::TotalTimeout(300)), RetryDecision::FailOver);
        assert_eq!(classify(&GatewayError::ChunkTimeout(120)), RetryDecision::FailOver);
        assert_eq!(classify(&provider(Some(504))), RetryDecision::FailOver);
        assert_eq!(classify(&provider(Some(524))), RetryDecision::FailOver);
    }

    #[test]
    fn rate_limit_and_service_unavailable_fail_over() {
        assert_eq!(classify(&provider(Some(429))), RetryDecision::FailOver);
        assert_eq!(classify(&provider(Some(503))), RetryDecision::FailOver);
        assert_eq!(
            classify(&GatewayError::RateLimitExceeded("provider-a".to_string())),
            RetryDecision::FailOver
        );
        assert_eq!(
            classify(&GatewayError::GuardrailUnavailable(
                "guardrail timed out".to_string()
            )),
            RetryDecision::FailOver
        );
    }

    #[test]
    fn other_client_errors_fail_over() {
        for status in [400u16, 401, 403, 404, 413, 422, 499] {
            assert_eq!(
                classify(&provider(Some(status))),
                RetryDecision::FailOver,
                "status {status}"
            );
        }
        assert_eq!(
            classify(&GatewayError::InvalidRequest("bad request".to_string())),
            RetryDecision::FailOver
        );
        assert_eq!(
            classify(&GatewayError::Authentication("invalid key".to_string())),
            RetryDecision::FailOver
        );
        assert_eq!(
            classify(&GatewayError::ContextCapacityExceeded {
                estimated_requirement: 8000,
                largest_supported_context: Some(4096),
            }),
            RetryDecision::FailOver
        );
        assert_eq!(
            classify(&GatewayError::SmartRoutingBudgetExceeded {
                period: "daily".to_string()
            }),
            RetryDecision::FailOver
        );
        assert_eq!(
            classify(&GatewayError::GuardrailPolicyViolation {
                category: "pii".to_string()
            }),
            RetryDecision::FailOver
        );
        assert_eq!(
            classify(&GatewayError::GuardrailInvalidAction),
            RetryDecision::FailOver
        );
    }

    #[test]
    fn server_and_transport_errors_keep_retry_budget() {
        assert_eq!(
            classify(&provider(Some(500))),
            RetryDecision::RetrySameProvider
        );
        assert_eq!(
            classify(&provider(Some(502))),
            RetryDecision::RetrySameProvider
        );
        assert_eq!(
            classify(&provider(Some(501))),
            RetryDecision::RetrySameProvider
        );
        assert_eq!(
            classify(&provider(Some(408))),
            RetryDecision::RetrySameProvider
        );
        // No status code: dispatch wraps transport send failures this way.
        assert_eq!(
            classify(&provider(None)),
            RetryDecision::RetrySameProvider
        );
        // 2xx envelopes: body-read, error-in-200 and truncated-SSE retries.
        assert_eq!(
            classify(&provider(Some(200))),
            RetryDecision::RetrySameProvider
        );
        assert_eq!(
            classify(&GatewayError::Network("connection reset".to_string())),
            RetryDecision::RetrySameProvider
        );
        assert_eq!(
            classify(&GatewayError::Http("server error".to_string())),
            RetryDecision::RetrySameProvider
        );
    }

    #[test]
    fn gateway_state_errors_fail_over() {
        let serialization: serde_json::Error =
            serde_json::from_str::<serde_json::Value>("{").unwrap_err();
        assert_eq!(
            classify(&GatewayError::Configuration("invalid config".to_string())),
            RetryDecision::FailOver
        );
        assert_eq!(
            classify(&GatewayError::AllProvidersFailed(AggregatedError::new(
                vec![]
            ))),
            RetryDecision::FailOver
        );
        assert_eq!(
            classify(&GatewayError::CircuitBreakerOpen("provider-a".to_string())),
            RetryDecision::FailOver
        );
        assert_eq!(
            classify(&GatewayError::Database("db locked".to_string())),
            RetryDecision::FailOver
        );
        assert_eq!(
            classify(&GatewayError::Cache("cache unavailable".to_string())),
            RetryDecision::FailOver
        );
        assert_eq!(
            classify(&GatewayError::Serialization(serialization)),
            RetryDecision::FailOver
        );
    }

    #[test]
    fn stream_param_rejection_classifies_as_client_error_fail_over() {
        // A 400 rejecting `stream`/`stream_options` fails over in the table;
        // the one-shot stream:false retry stays a caller-owned mechanism.
        let err = GatewayError::Provider {
            provider: "provider-a".to_string(),
            message: "stream is not supported for this model".to_string(),
            status_code: Some(400),
        };
        assert_eq!(classify(&err), RetryDecision::FailOver);
    }
}
