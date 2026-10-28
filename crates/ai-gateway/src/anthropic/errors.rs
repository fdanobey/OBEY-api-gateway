//! Anthropic-shaped error mapping for the `/v1/messages` front door
//! (spec Requirement 3).
//!
//! Every gateway failure leaving the Anthropic surface is converted into an
//! [`AnthropicErrorResponse`] (`{"type":"error","error":{"type","message"}}`)
//! with the documented error-type ↔ HTTP status pairing, and every embedded
//! message is passed through [`redact`] first so no key material, bearer
//! tokens, or credential-bearing URLs can leak to the client.

use crate::anthropic::models::{error_types, AnthropicErrorResponse, ErrorBody};
use crate::error::GatewayError;
use crate::security::redact::redact_secrets;

/// HTTP status for an Anthropic error-type string (spec Req 3 mapping table).
///
/// Unknown types fall back to 500 (`api_error` semantics).
pub fn http_status(kind: &str) -> u16 {
    match kind {
        error_types::INVALID_REQUEST_ERROR => 400,
        error_types::AUTHENTICATION_ERROR => 401,
        error_types::PERMISSION_ERROR => 403,
        error_types::NOT_FOUND_ERROR => 404,
        error_types::REQUEST_TOO_LARGE => 413,
        error_types::RATE_LIMIT_ERROR => 429,
        error_types::API_ERROR => 500,
        error_types::OVERLOADED_ERROR => 529,
        _ => 500,
    }
}

/// Build an error body with the message redacted by [`redact`].
fn error_body(kind: &str, message: impl Into<String>) -> ErrorBody {
    ErrorBody {
        kind: kind.to_string(),
        message: redact(&message.into()),
        extra: serde_json::Map::new(),
    }
}

/// Build a full `{"type":"error","error":{...}}` response.
pub fn error_response(kind: &str, message: impl Into<String>) -> AnthropicErrorResponse {
    AnthropicErrorResponse {
        kind: "error".to_string(),
        error: error_body(kind, message),
        extra: serde_json::Map::new(),
    }
}

/// Convert a translator rejection string (from
/// [`crate::anthropic::translate_request::translate`]) into a 400
/// `invalid_request_error` (spec Req 3.1).
pub fn from_translator_rejection(reason: &str) -> AnthropicErrorResponse {
    error_response(error_types::INVALID_REQUEST_ERROR, reason)
}

/// Map a [`GatewayError`] onto the closest Anthropic error type (spec Req 3.4).
///
/// Timeout/circuit-breaker/unavailability classes become `overloaded_error`
/// (529); everything else follows the status pairing in [`http_status`].
/// Client messages are generic per-kind summaries — full error detail stays
/// in the handler's tracing log and never reaches the client body.
pub fn from_gateway_error(err: &GatewayError) -> AnthropicErrorResponse {
    let kind = match err {
        GatewayError::InvalidRequest(_) | GatewayError::GuardrailInvalidAction => {
            error_types::INVALID_REQUEST_ERROR
        }
        GatewayError::Authentication(_) => error_types::AUTHENTICATION_ERROR,
        GatewayError::GuardrailPolicyViolation { .. } => error_types::PERMISSION_ERROR,
        GatewayError::ContextCapacityExceeded { .. } => error_types::REQUEST_TOO_LARGE,
        GatewayError::RateLimitExceeded(_)
        | GatewayError::SmartRoutingBudgetExceeded { .. } => error_types::RATE_LIMIT_ERROR,
        GatewayError::TtfbTimeout(_)
        | GatewayError::TotalTimeout(_)
        | GatewayError::ChunkTimeout(_)
        | GatewayError::CircuitBreakerOpen(_)
        | GatewayError::GuardrailUnavailable(_) => error_types::OVERLOADED_ERROR,
        GatewayError::Provider { status_code, .. } => provider_error_kind(*status_code),
        GatewayError::AllProvidersFailed(_) => error_types::API_ERROR,
        _ => error_types::API_ERROR,
    };
    error_response(kind, client_message(kind))
}

/// Generic, per-kind client-visible message: never embeds upstream error
/// detail (provider names, messages, URLs) which could leak topology or
/// key material; the full detail remains in the handler's tracing log.
fn client_message(kind: &str) -> &'static str {
    match kind {
        error_types::INVALID_REQUEST_ERROR => "Invalid request.",
        error_types::AUTHENTICATION_ERROR => "Authentication failed.",
        error_types::PERMISSION_ERROR => "Request blocked by policy.",
        error_types::NOT_FOUND_ERROR => "Requested resource was not found.",
        error_types::REQUEST_TOO_LARGE => "Request exceeds the model context window.",
        error_types::RATE_LIMIT_ERROR => "Rate limit exceeded.",
        error_types::OVERLOADED_ERROR => "The service is temporarily overloaded.",
        _ => "An internal server error occurred.",
    }
}

/// Closest Anthropic error type for an upstream provider status code.
fn provider_error_kind(status_code: Option<u16>) -> &'static str {
    match status_code {
        Some(401) => error_types::AUTHENTICATION_ERROR,
        Some(403) => error_types::PERMISSION_ERROR,
        Some(404) => error_types::NOT_FOUND_ERROR,
        Some(413) => error_types::REQUEST_TOO_LARGE,
        Some(429) => error_types::RATE_LIMIT_ERROR,
        Some(400..=499) => error_types::INVALID_REQUEST_ERROR,
        Some(502 | 503 | 504 | 529) => error_types::OVERLOADED_ERROR,
        Some(500..=599) => error_types::API_ERROR,
        _ => error_types::API_ERROR,
    }
}

/// Redact secret shapes from an error message before it reaches a client.
///
/// Delegates to the shared [`crate::security::redact::redact_secrets`]
/// helper (single precompiled pattern covering `sk-` family keys, AWS
/// access keys, Bearer/Authorization spans, api-key params, and the known
/// token prefix families), so the Anthropic surface and the log sinks can
/// never drift apart.
fn redact(message: &str) -> String {
    redact_secrets(message)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::anthropic::models::error_types;

    #[test]
    fn every_error_type_maps_to_its_documented_status() {
        assert_eq!(http_status(error_types::INVALID_REQUEST_ERROR), 400);
        assert_eq!(http_status(error_types::AUTHENTICATION_ERROR), 401);
        assert_eq!(http_status(error_types::PERMISSION_ERROR), 403);
        assert_eq!(http_status(error_types::NOT_FOUND_ERROR), 404);
        assert_eq!(http_status(error_types::REQUEST_TOO_LARGE), 413);
        assert_eq!(http_status(error_types::RATE_LIMIT_ERROR), 429);
        assert_eq!(http_status(error_types::API_ERROR), 500);
        assert_eq!(http_status(error_types::OVERLOADED_ERROR), 529);
    }

    #[test]
    fn unknown_error_type_falls_back_to_500() {
        assert_eq!(http_status("error"), 500);
        assert_eq!(http_status("totally_unknown"), 500);
    }

    #[test]
    fn error_response_serializes_to_documented_shape() {
        let body = error_response(error_types::RATE_LIMIT_ERROR, "Slow down.");
        let value = serde_json::to_value(&body).expect("serializable");
        assert_eq!(value["type"], "error");
        assert_eq!(value["error"]["type"], "rate_limit_error");
        assert_eq!(value["error"]["message"], "Slow down.");
    }

    #[test]
    fn gateway_error_authentication_maps_to_401() {
        let body = from_gateway_error(&GatewayError::Authentication("bad key".into()));
        assert_eq!(body.error.kind, error_types::AUTHENTICATION_ERROR);
        assert_eq!(http_status(&body.error.kind), 401);
    }

    #[test]
    fn gateway_error_rate_limit_maps_to_429() {
        let body = from_gateway_error(&GatewayError::RateLimitExceeded("provider-a".into()));
        assert_eq!(body.error.kind, error_types::RATE_LIMIT_ERROR);
        assert_eq!(http_status(&body.error.kind), 429);
    }

#[test]
fn gateway_error_upstream_5xx_maps_to_api_error() {
    let body = from_gateway_error(&GatewayError::Provider {
        provider: "provider-a".into(),
        message: "boom".into(),
        status_code: Some(500),
    });
    assert_eq!(body.error.kind, error_types::API_ERROR);
    assert_eq!(http_status(&body.error.kind), 500);
    // Client messages are generic per-kind summaries (Req 3.4): upstream
    // detail never reaches the client body.
    assert_eq!(body.error.message, "An internal server error occurred.");
}

    #[test]
    fn gateway_error_provider_404_maps_to_not_found() {
        let body = from_gateway_error(&GatewayError::Provider {
            provider: "provider-a".into(),
            message: "no such model".into(),
            status_code: Some(404),
        });
        assert_eq!(body.error.kind, error_types::NOT_FOUND_ERROR);
        assert_eq!(http_status(&body.error.kind), 404);
    }

    #[test]
    fn gateway_error_invalid_request_maps_to_400() {
        let body = from_gateway_error(&GatewayError::InvalidRequest("bad shape".into()));
        assert_eq!(body.error.kind, error_types::INVALID_REQUEST_ERROR);
        assert_eq!(http_status(&body.error.kind), 400);
    }

    #[test]
    fn gateway_error_timeout_and_circuit_breaker_map_to_overloaded() {
        for err in [
            GatewayError::TtfbTimeout(10),
            GatewayError::TotalTimeout(60),
            GatewayError::ChunkTimeout(5),
            GatewayError::CircuitBreakerOpen("provider-a".into()),
        ] {
            let body = from_gateway_error(&err);
            assert_eq!(body.error.kind, error_types::OVERLOADED_ERROR);
            assert_eq!(http_status(&body.error.kind), 529);
        }
    }

    #[test]
    fn gateway_error_context_capacity_maps_to_request_too_large() {
        let body = from_gateway_error(&GatewayError::ContextCapacityExceeded {
            estimated_requirement: 500_000,
            largest_supported_context: Some(128_000),
        });
        assert_eq!(body.error.kind, error_types::REQUEST_TOO_LARGE);
        assert_eq!(http_status(&body.error.kind), 413);
    }

#[test]
fn gateway_error_all_providers_failed_keeps_generic_message() {
    let err = GatewayError::AllProvidersFailed(
        crate::error::AggregatedError::new(vec![crate::error::ProviderAttempt::new(
            "provider-a".into(),
            "model-x".into(),
            "upstream exploded".into(),
            Some(500),
        )]),
    );
    let body = from_gateway_error(&err);
    assert_eq!(body.error.kind, error_types::API_ERROR);
    // Client messages are generic per-kind summaries; attempt detail lives
    // in the aggregated-error log/report and the handler's tracing log.
    assert_eq!(body.error.message, "An internal server error occurred.");
    assert!(!body.error.message.contains("provider-a"));
}

    #[test]
    fn gateway_error_provider_status_matrix() {
        let cases = [
            (401u16, error_types::AUTHENTICATION_ERROR),
            (403, error_types::PERMISSION_ERROR),
            (404, error_types::NOT_FOUND_ERROR),
            (413, error_types::REQUEST_TOO_LARGE),
            (422, error_types::INVALID_REQUEST_ERROR),
            (429, error_types::RATE_LIMIT_ERROR),
            (502, error_types::OVERLOADED_ERROR),
            (503, error_types::OVERLOADED_ERROR),
            (504, error_types::OVERLOADED_ERROR),
            (500, error_types::API_ERROR),
        ];
        for (status, expected) in cases {
            let body = from_gateway_error(&GatewayError::Provider {
                provider: "p".into(),
                message: "m".into(),
                status_code: Some(status),
            });
            assert_eq!(body.error.kind, expected, "status {status}");
        }
    }

    #[test]
    fn translator_rejection_maps_to_invalid_request() {
        let body = from_translator_rejection("content block type 'document' is not supported");
        assert_eq!(body.error.kind, error_types::INVALID_REQUEST_ERROR);
        assert_eq!(http_status(&body.error.kind), 400);
        assert_eq!(
            body.error.message,
            "content block type 'document' is not supported"
        );
    }

#[test]
fn key_material_is_redacted_from_messages() {
    let err = GatewayError::Provider {
        provider: "provider-a".into(),
        message: "upstream https://alice:hunter2@provider.example/v1 rejected Bearer eyJabc.def sk-proj-KEY123_ AKIAIOSFODNN7EXAMPLE vk_SuperSecret9".into(),
        status_code: Some(500),
    };
    let body = from_gateway_error(&err);
    let message = &body.error.message;
    // Generic per-kind message never carries upstream detail or key
    // material (Req 3.4); redaction still applies to the shaped path.
    assert!(!message.contains("hunter2"));
    assert!(!message.contains("eyJabc.def"));
    assert!(!message.contains("sk-proj-KEY123_"));
    assert!(!message.contains("AKIAIOSFODNN7EXAMPLE"));
    assert!(!message.contains("vk_SuperSecret9"));
    assert!(!message.contains("provider-a"));
    assert_eq!(message, "An internal server error occurred.");
}

#[test]
fn redact_covers_every_supported_shape() {
    let redacted = redact(
        "https://alice:super-secret-password@example.test/a Bearer tok123 sk-abc_1234567890123456 \
AKIA1234567890ABCDEF vk_abcdef123456 plain text stays",
    );
    assert!(!redacted.contains("super-secret-password"));
    assert!(!redacted.contains("tok123"));
    assert!(!redacted.contains("sk-abc_1234567890123456"));
    assert!(!redacted.contains("AKIA1234567890ABCDEF"));
    assert!(redacted.contains("plain text stays"));
    }
}
