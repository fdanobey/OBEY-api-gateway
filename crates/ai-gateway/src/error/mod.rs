use chrono::{DateTime, Utc};
use serde::Serialize;
use thiserror::Error;

/// Main gateway error type
#[derive(Debug, Error)]
#[allow(dead_code)]
pub enum GatewayError {
    #[error("Configuration error: {0}")]
    Configuration(String),

    #[error("Provider error: {provider} - {message}")]
    Provider {
        provider: String,
        message: String,
        status_code: Option<u16>,
    },

    #[error("All providers failed")]
    AllProvidersFailed(AggregatedError),

    #[error("Circuit breaker open for provider: {0}")]
    CircuitBreakerOpen(String),

    #[error("Rate limit exceeded for provider: {0}")]
    RateLimitExceeded(String),

    #[error("Request timeout after {0}s waiting for first byte from provider")]
    TtfbTimeout(u64),

    #[error("Request timeout after {0}s total round-trip time")]
    TotalTimeout(u64),

    #[error("Invalid request: {0}")]
    InvalidRequest(String),

    #[error("Authentication failed: {0}")]
    Authentication(String),

    #[error("Database error: {0}")]
    Database(String),

    #[error("Cache error: {0}")]
    Cache(String),

    #[error("Serialization error: {0}")]
    Serialization(#[from] serde_json::Error),

    #[error("HTTP error: {0}")]
    Http(String),

    #[error("Network error: {0}")]
    Network(String),

    /// A guardrail `block` action fired (pre- or post-call). Maps to HTTP 403.
    /// Carries only the triggering entity/category label, never raw content
    /// (Req 2.2, 3.1).
    #[error("Request blocked by guardrail policy: {category}")]
    GuardrailPolicyViolation { category: String },

    /// A guardrail stage declared an action invalid for its phase. Maps to
    /// HTTP 400 (Req 2.7).
    #[error("Invalid guardrail stage action")]
    GuardrailInvalidAction,

    /// A `fail_close` guardrail provider timed out or errored. Maps to HTTP 503
    /// (Req 2.9, 9.7).
    #[error("Guardrail unavailable: {0}")]
    GuardrailUnavailable(String),
}

/// Aggregated error containing all provider attempts
#[derive(Debug, Clone, Serialize)]
pub struct AggregatedError {
    pub attempts: Vec<ProviderAttempt>,
}

impl AggregatedError {
    pub fn new(attempts: Vec<ProviderAttempt>) -> Self {
        Self { attempts }
    }
}

impl std::fmt::Display for AggregatedError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(
            f,
            "All {} provider(s) failed: {}",
            self.attempts.len(),
            self.attempts
                .iter()
                .map(|a| format!("{}: {}", a.provider, a.error))
                .collect::<Vec<_>>()
                .join(", ")
        )
    }
}

impl std::error::Error for AggregatedError {}

impl GatewayError {
    /// Return the HTTP status code that this error maps to.
    pub fn status_code(&self) -> axum::http::StatusCode {
        use axum::http::StatusCode;
        match self {
            GatewayError::InvalidRequest(_) => StatusCode::BAD_REQUEST,
            GatewayError::Authentication(_) => StatusCode::UNAUTHORIZED,
            GatewayError::AllProvidersFailed(_) => StatusCode::BAD_GATEWAY,
            GatewayError::RateLimitExceeded(_) => StatusCode::TOO_MANY_REQUESTS,
            GatewayError::TtfbTimeout(_) => StatusCode::GATEWAY_TIMEOUT,
            GatewayError::TotalTimeout(_) => StatusCode::GATEWAY_TIMEOUT,
            GatewayError::CircuitBreakerOpen(_) => StatusCode::SERVICE_UNAVAILABLE,
            GatewayError::GuardrailPolicyViolation { .. } => StatusCode::FORBIDDEN,
            GatewayError::GuardrailInvalidAction => StatusCode::BAD_REQUEST,
            GatewayError::GuardrailUnavailable(_) => StatusCode::SERVICE_UNAVAILABLE,
            GatewayError::Provider { status_code, .. } => status_code
                .and_then(|c| StatusCode::from_u16(c).ok())
                .unwrap_or(StatusCode::BAD_GATEWAY),
            _ => StatusCode::INTERNAL_SERVER_ERROR,
        }
    }
}

/// Single provider attempt information
#[derive(Debug, Clone, Serialize)]
pub struct ProviderAttempt {
    pub provider: String,
    pub model: String,
    pub error: String,
    pub status_code: Option<u16>,
    pub timestamp: DateTime<Utc>,
}

impl ProviderAttempt {
    pub fn new(provider: String, model: String, error: String, status_code: Option<u16>) -> Self {
        Self {
            provider,
            model,
            error,
            status_code,
            timestamp: Utc::now(),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_aggregated_error_json_format() {
        // Validates: Requirements 25.6
        let attempts = vec![
            ProviderAttempt::new(
                "openai-primary".to_string(),
                "gpt-4".to_string(),
                "Rate limit exceeded".to_string(),
                Some(429),
            ),
            ProviderAttempt::new(
                "bedrock-us-east".to_string(),
                "claude-3".to_string(),
                "Service unavailable".to_string(),
                Some(503),
            ),
        ];

        let error = AggregatedError::new(attempts);
        let json = serde_json::to_value(&error).expect("Should serialize to JSON");

        assert!(json.is_object());
        assert!(json.get("attempts").is_some());
        assert!(json["attempts"].is_array());
        assert_eq!(json["attempts"].as_array().unwrap().len(), 2);
    }

    #[test]
    fn test_error_response_includes_all_provider_attempts() {
        // Validates: Requirements 25.2, 25.3, 25.4, 25.5
        let attempts = vec![
            ProviderAttempt::new(
                "provider1".to_string(),
                "model1".to_string(),
                "Error message 1".to_string(),
                Some(500),
            ),
            ProviderAttempt::new(
                "provider2".to_string(),
                "model2".to_string(),
                "Error message 2".to_string(),
                Some(502),
            ),
            ProviderAttempt::new(
                "provider3".to_string(),
                "model3".to_string(),
                "Error message 3".to_string(),
                None,
            ),
        ];

        let error = AggregatedError::new(attempts);
        let json = serde_json::to_value(&error).expect("Should serialize to JSON");

        let attempts_array = json["attempts"].as_array().unwrap();
        assert_eq!(attempts_array.len(), 3);

        // Verify first attempt includes all required fields
        let attempt1 = &attempts_array[0];
        assert_eq!(attempt1["provider"], "provider1");
        assert_eq!(attempt1["model"], "model1");
        assert_eq!(attempt1["error"], "Error message 1");
        assert_eq!(attempt1["status_code"], 500);
        assert!(attempt1.get("timestamp").is_some());

        // Verify second attempt
        let attempt2 = &attempts_array[1];
        assert_eq!(attempt2["provider"], "provider2");
        assert_eq!(attempt2["model"], "model2");
        assert_eq!(attempt2["error"], "Error message 2");
        assert_eq!(attempt2["status_code"], 502);
        assert!(attempt2.get("timestamp").is_some());

        // Verify third attempt with no status code
        let attempt3 = &attempts_array[2];
        assert_eq!(attempt3["provider"], "provider3");
        assert_eq!(attempt3["model"], "model3");
        assert_eq!(attempt3["error"], "Error message 3");
        assert!(attempt3["status_code"].is_null());
        assert!(attempt3.get("timestamp").is_some());
    }

    #[test]
    fn test_provider_attempt_includes_timestamp() {
        // Validates: Requirement 25.5
        let attempt = ProviderAttempt::new(
            "test-provider".to_string(),
            "test-model".to_string(),
            "test error".to_string(),
            Some(500),
        );

        let json = serde_json::to_value(&attempt).expect("Should serialize to JSON");

        assert!(json.get("timestamp").is_some());
        let timestamp_str = json["timestamp"].as_str().unwrap();
        assert!(!timestamp_str.is_empty());
    }

    #[test]
    fn test_aggregated_error_display_format() {
        // Validates: Requirement 25.7 - top-level error message
        let attempts = vec![
            ProviderAttempt::new(
                "provider1".to_string(),
                "model1".to_string(),
                "Connection timeout".to_string(),
                Some(504),
            ),
            ProviderAttempt::new(
                "provider2".to_string(),
                "model2".to_string(),
                "Internal error".to_string(),
                Some(500),
            ),
        ];

        let error = AggregatedError::new(attempts);
        let display_msg = format!("{}", error);

        assert!(display_msg.contains("All 2 provider(s) failed"));
        assert!(display_msg.contains("provider1: Connection timeout"));
        assert!(display_msg.contains("provider2: Internal error"));
    }

    #[test]
    fn test_empty_aggregated_error() {
        let error = AggregatedError::new(vec![]);
        let json = serde_json::to_value(&error).expect("Should serialize to JSON");

        assert!(json["attempts"].is_array());
        assert_eq!(json["attempts"].as_array().unwrap().len(), 0);
    }

    #[test]
    fn test_provider_attempt_with_optional_status_code() {
        // Test with status code
        let attempt_with_code = ProviderAttempt::new(
            "provider".to_string(),
            "model".to_string(),
            "error".to_string(),
            Some(429),
        );
        let json_with = serde_json::to_value(&attempt_with_code).unwrap();
        assert_eq!(json_with["status_code"], 429);

        // Test without status code
        let attempt_without_code = ProviderAttempt::new(
            "provider".to_string(),
            "model".to_string(),
            "error".to_string(),
            None,
        );
        let json_without = serde_json::to_value(&attempt_without_code).unwrap();
        assert!(json_without["status_code"].is_null());
    }
}
