//! Error types for the virtual key management feature.
//!
//! [`KeyError`] is the top-level error returned by [`super::VirtualKeyManager`]
//! operations. It maps to HTTP status codes per the design's error mapping
//! table (validation → 400, not found → 404, revoked → 409, store/encryption →
//! 500). Field-level validation failures are carried by [`ValidationErrors`],
//! which serializes to the `{"errors": [{"field", "message"}]}` shape used by
//! the admin API.

use axum::http::StatusCode;
use axum::response::{IntoResponse, Response};
use axum::Json;
use serde::Serialize;
use serde_json::json;

use super::store::KeyStoreError;

/// A single field validation failure.
#[derive(Debug, Clone, PartialEq, Eq, Serialize)]
pub struct FieldError {
    pub field: String,
    pub message: String,
}

impl FieldError {
    pub fn new(field: impl Into<String>, message: impl Into<String>) -> Self {
        Self {
            field: field.into(),
            message: message.into(),
        }
    }
}

/// A collection of per-field validation errors.
///
/// Serializes as `{"errors": [{"field": "...", "message": "..."}]}` to match
/// the admin API contract (design: Requirement 10.3).
#[derive(Debug, Clone, PartialEq, Eq, Serialize)]
pub struct ValidationErrors {
    pub errors: Vec<FieldError>,
}

impl ValidationErrors {
    /// Create an empty error set.
    pub fn new() -> Self {
        Self { errors: Vec::new() }
    }

    /// Append a field error.
    pub fn push(&mut self, field: impl Into<String>, message: impl Into<String>) {
        self.errors.push(FieldError::new(field, message));
    }

    /// Whether any field errors have been recorded.
    pub fn is_empty(&self) -> bool {
        self.errors.is_empty()
    }

    /// Consume into a `Result`: `Ok(())` when empty, else `Err(self)`.
    pub fn into_result(self) -> Result<(), Self> {
        if self.is_empty() {
            Ok(())
        } else {
            Err(self)
        }
    }
}

impl Default for ValidationErrors {
    fn default() -> Self {
        Self::new()
    }
}

impl std::fmt::Display for ValidationErrors {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        let joined = self
            .errors
            .iter()
            .map(|e| format!("{}: {}", e.field, e.message))
            .collect::<Vec<_>>()
            .join("; ");
        write!(f, "{joined}")
    }
}

/// Top-level error for virtual key management operations.
#[derive(Debug, thiserror::Error)]
pub enum KeyError {
    /// The key identifier does not exist (HTTP 404).
    #[error("Key not found: {0}")]
    NotFound(String),

    /// The key is revoked and cannot be modified (HTTP 409).
    #[error("Key is revoked and cannot be modified")]
    KeyRevoked,

    /// One or more fields failed validation (HTTP 400).
    #[error("Validation error: {0}")]
    Validation(ValidationErrors),

    /// A persistence-layer error occurred (HTTP 500).
    #[error("Database error: {0}")]
    Store(#[from] KeyStoreError),

    /// Encryption of the key value failed (HTTP 500). Carries only a
    /// non-sensitive message; the plaintext key is never included.
    #[error("Encryption error: {0}")]
    Encryption(String),
}

/// Map a [`KeyError`] to its HTTP response per the design "HTTP Error Mapping"
/// table:
///
/// | Error                    | Status | Body                                       |
/// |--------------------------|--------|--------------------------------------------|
/// | `NotFound`               | 404    | `{"error": "Key not found"}`               |
/// | `KeyRevoked`             | 409    | `{"error": "Key is revoked and cannot ..."}` |
/// | `Validation(errors)`     | 400    | `{"errors": [{"field", "message"}]}`       |
/// | `Store` / `Encryption`   | 500    | `{"error": "Internal server error"}`       |
///
/// Internal store/encryption details are logged server-side but never leaked in
/// the response body (Req 10.3, 10.4; design HTTP Error Mapping).
impl IntoResponse for KeyError {
    fn into_response(self) -> Response {
        match self {
            KeyError::NotFound(_) => (
                StatusCode::NOT_FOUND,
                Json(json!({ "error": "Key not found" })),
            )
                .into_response(),
            KeyError::KeyRevoked => (
                StatusCode::CONFLICT,
                Json(json!({ "error": "Key is revoked and cannot be modified" })),
            )
                .into_response(),
            KeyError::Validation(errors) => (
                StatusCode::BAD_REQUEST,
                // Serializes to `{"errors": [{"field": "...", "message": "..."}]}`.
                Json(json!({ "errors": errors.errors })),
            )
                .into_response(),
            KeyError::Store(err) => {
                tracing::error!(error = %err, "virtual key store error");
                internal_error()
            }
            KeyError::Encryption(err) => {
                tracing::error!(error = %err, "virtual key encryption error");
                internal_error()
            }
        }
    }
}

/// Generic 500 response that does not leak internal error details.
fn internal_error() -> Response {
    (
        StatusCode::INTERNAL_SERVER_ERROR,
        Json(json!({ "error": "Internal server error" })),
    )
        .into_response()
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Smoke test: each `KeyError` variant maps to the documented HTTP status.
    /// Full body/endpoint assertions live with the admin API tests (task 11.2).
    #[test]
    fn key_error_status_mapping() {
        assert_eq!(
            KeyError::NotFound("x".into()).into_response().status(),
            StatusCode::NOT_FOUND
        );
        assert_eq!(
            KeyError::KeyRevoked.into_response().status(),
            StatusCode::CONFLICT
        );
        let mut errors = ValidationErrors::new();
        errors.push("name", "bad");
        assert_eq!(
            KeyError::Validation(errors).into_response().status(),
            StatusCode::BAD_REQUEST
        );
        assert_eq!(
            KeyError::Encryption("boom".into()).into_response().status(),
            StatusCode::INTERNAL_SERVER_ERROR
        );
    }
}
