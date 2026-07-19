//! `CodexError` type scoped to the Codex backend translation pipeline.
//!
//! # Security invariant
//!
//! **No variant payload may ever contain raw token bytes, access tokens,
//! refresh tokens, the `chatgpt-account-id` value, or raw upstream body
//! bytes.** `Translation` and `Sse` variants carry only sanitized context
//! strings (parse error descriptions, field names, etc.); callers that
//! construct these variants from upstream data MUST strip token-shaped
//! material first.
//!
//! This invariant mirrors the rule documented on [`crate::oauth::OAuthError`]
//! and satisfies Requirements 12.1 and 12.2 of the
//! `codex-backend-translation` spec.

use crate::oauth::OAuthError;

/// All failure modes of the Codex backend translation pipeline.
///
/// Every submodule under [`crate::codex`] returns `Result<_, CodexError>`;
/// the router translates these at the boundary into the crate-wide
/// `GatewayError` (HTTP 400 for [`CodexError::UnsupportedFeature`],
/// HTTP 502 for [`CodexError::AccountIdExtraction`],
/// [`CodexError::Translation`], [`CodexError::Sse`], and
/// [`CodexError::UpstreamTerminatedEarly`]).
///
/// The [`std::fmt::Display`] impl (provided by `thiserror`) is safe to log —
/// no variant embeds token or body bytes.
#[derive(Debug, thiserror::Error)]
#[allow(dead_code)]
pub enum CodexError {
    /// A Chat Completions request referenced a feature that the Codex
    /// translation layer does not support in this iteration (e.g. audio
    /// input content parts, audio output). Surfaced to the client as an
    /// HTTP 400 Chat-Completions-shaped error per Requirement 14.3.
    ///
    /// `feature` is a static discriminator (e.g. `"input_audio"`,
    /// `"audio_output"`) and never contains client payload bytes.
    #[error("unsupported Chat Completions feature for Codex backend: {feature}")]
    UnsupportedFeature { feature: &'static str },

    /// Extracting the `chatgpt_account_id` claim from the OAuth access
    /// token failed (malformed JWT, undecodable payload, missing claim).
    /// The wrapped [`OAuthError`] describes the failure mode without
    /// exposing token bytes — see the security note on that type.
    #[error("OAuth account-id extraction failed: {0}")]
    AccountIdExtraction(
        #[from]
        #[source]
        OAuthError,
    ),

    /// Chat-Completions → Responses request translation failed (e.g.
    /// malformed `response_format`, missing required field, JSON shape
    /// mismatch). The inner string is a sanitized diagnostic message;
    /// it MUST NOT include raw message content or tool-call arguments.
    #[error("request translation failed: {0}")]
    Translation(String),

    /// SSE framing or parse failure on the upstream `text/event-stream`
    /// body (invalid UTF-8 on a record, JSON decode failure at a record
    /// boundary, etc.). The inner string is a sanitized diagnostic
    /// message; it MUST NOT include raw body bytes beyond a short
    /// redacted snippet of the failing token or field name.
    #[error("Codex SSE parse error: {0}")]
    Sse(String),

    /// The upstream Responses stream closed before emitting a terminal
    /// `response.completed` or `response.failed` event. The router maps
    /// this to an HTTP 502 Chat-Completions-shaped error per
    /// Requirement 5.7.
    #[error("Codex upstream stream terminated before response.completed")]
    UpstreamTerminatedEarly,
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Guard against accidentally enlarging a variant payload to carry a
    /// heap blob that could contain token bytes. If a future change
    /// genuinely needs a larger payload, update this bound **and** document
    /// the redaction contract on that variant.
    #[test]
    fn variants_do_not_balloon() {
        // `AccountIdExtraction(OAuthError)` is the largest variant; it
        // must stay a thin wrapper rather than cloning token material.
        assert!(
            std::mem::size_of::<CodexError>() <= std::mem::size_of::<OAuthError>() + 16,
            "CodexError grew unexpectedly; audit variants for token payloads"
        );
    }

    #[test]
    fn display_unsupported_feature_uses_static_discriminator() {
        let err = CodexError::UnsupportedFeature {
            feature: "input_audio",
        };
        let rendered = format!("{err}");
        assert!(rendered.contains("input_audio"));
        assert!(!rendered.contains("Bearer"));
    }

    #[test]
    fn display_translation_surfaces_sanitized_context() {
        let err = CodexError::Translation("missing response_format.json_schema.name".into());
        let rendered = format!("{err}");
        assert!(rendered.contains("request translation failed"));
        assert!(rendered.contains("json_schema.name"));
    }

    #[test]
    fn display_sse_surfaces_sanitized_context() {
        let err = CodexError::Sse("record missing `event:` prefix".into());
        let rendered = format!("{err}");
        assert!(rendered.contains("Codex SSE parse error"));
    }

    #[test]
    fn display_upstream_terminated_early_is_stable() {
        let rendered = format!("{}", CodexError::UpstreamTerminatedEarly);
        assert_eq!(
            rendered,
            "Codex upstream stream terminated before response.completed"
        );
    }

    #[test]
    fn from_oauth_error_wraps_as_account_id_extraction() {
        let oauth = OAuthError::StateMismatch;
        let err: CodexError = oauth.into();
        match err {
            CodexError::AccountIdExtraction(_) => {}
            other => panic!("expected AccountIdExtraction, got {other:?}"),
        }
    }

    #[test]
    fn error_source_chain_exposes_oauth_cause() {
        use std::error::Error;
        let err: CodexError = OAuthError::StateMismatch.into();
        let source = err
            .source()
            .expect("source should be the wrapped OAuthError");
        assert!(source.to_string().contains("CSRF state mismatch"));
    }
}
