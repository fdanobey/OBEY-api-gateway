#![allow(dead_code)]
//! OAuth error type covering every failure mode of the OpenAI OAuth + PKCE
//! flow and the GitHub Copilot device flow.
//!
//! # Security invariant
//!
//! **Token values (access_token, refresh_token, code_verifier, authorization
//! code, PKCE verifiers, etc.) MUST NEVER appear in the `Display` or `Debug`
//! output of any variant.** Where a field could plausibly contain a secret,
//! the type either stores a placeholder (`<redacted>`) or stores only
//! non-sensitive metadata (status codes, sanitized body snippets, etc.).
//!
//! Callers that need to surface upstream error bodies must redact any
//! token-shaped content before constructing an [`OAuthError`] variant.

use crate::secrets::SecretError;

/// Placeholder string used anywhere a secret value would otherwise be rendered.
pub const REDACTED: &str = "<redacted>";

/// Maximum number of bytes of an upstream error response body retained in
/// `OAuthError` variants (256 bytes, UTF-8-boundary-safe).
pub const ERROR_BODY_MAX_BYTES: usize = 256;

/// Truncate a response body to at most [`ERROR_BODY_MAX_BYTES`] bytes on a
/// UTF-8 character boundary. Shared by the device flow and the Copilot
/// token exchange (mirrors the policy of `oauth/token.rs`).
pub(crate) fn truncate_error_body(body: String) -> String {
    if body.len() <= ERROR_BODY_MAX_BYTES {
        return body;
    }
    let mut end = ERROR_BODY_MAX_BYTES;
    while end > 0 && !body.is_char_boundary(end) {
        end -= 1;
    }
    let mut truncated = body;
    truncated.truncate(end);
    truncated
}

/// All failure modes of the OAuth login, token exchange, refresh, storage, and
/// callback subsystems.
///
/// The [`std::fmt::Display`] impl (provided by `thiserror`) is safe to log —
/// no variant includes raw token material.
#[derive(Debug, thiserror::Error)]
pub enum OAuthError {
    /// The platform refused to open the user's default browser.
    ///
    /// Callers should fall back to returning the authorization URL to the
    /// user for manual navigation.
    #[error("failed to open default browser: {0}")]
    BrowserOpenFailed(#[source] std::io::Error),

    /// The callback server did not receive the authorization redirect within
    /// the configured timeout (120 seconds per design).
    #[error("OAuth callback timed out after {timeout_secs} seconds")]
    CallbackTimeout { timeout_secs: u64 },

    /// The `state` query parameter on the callback did not match the value
    /// generated when the flow was initiated. Indicates a CSRF attempt or a
    /// stale/duplicate callback.
    #[error("CSRF state mismatch on OAuth callback")]
    StateMismatch,

    /// The callback URL did not contain a `code` parameter and no `error`
    /// parameter was present either — a malformed response from the provider.
    #[error("OAuth callback missing `code` parameter")]
    MissingAuthorizationCode,

    /// The authorization server returned an error via the callback query
    /// string (e.g. `error=access_denied`). The `error` code and optional
    /// `error_description` are from the provider and contain no token data.
    #[error("OAuth authorization error from provider: {error}{}",
        .description.as_deref().map(|d| format!(" ({d})")).unwrap_or_default()
    )]
    AuthorizationError {
        error: String,
        description: Option<String>,
    },

    /// The token endpoint returned a non-success HTTP status during the
    /// initial authorization-code exchange.
    ///
    /// `body` is a best-effort sanitized snippet of the response body; callers
    /// must ensure no token values leak into it (the OAuth token endpoint's
    /// error responses do not include tokens, but this field is still intended
    /// for short diagnostic messages only).
    #[error("token exchange failed with HTTP {status}: {body}")]
    TokenExchangeHttp { status: u16, body: String },

    /// A transport-level failure (DNS, TLS, connection reset, etc.) occurred
    /// during the initial authorization-code exchange.
    #[error("token exchange network error")]
    TokenExchangeNetwork(#[source] reqwest::Error),

    /// The provider returned 401/403 to a refresh request — the refresh token
    /// is no longer valid and the user must re-authenticate.
    #[error("OAuth session expired (refresh returned HTTP {status}); re-authentication required")]
    SessionExpired { status: u16 },

    /// The token endpoint returned a non-success HTTP status (other than
    /// 401/403) during a refresh attempt.
    #[error("token refresh failed with HTTP {status}: {body}")]
    TokenRefreshHttp { status: u16, body: String },

    /// Transport-level failure during a refresh attempt. Returned after the
    /// exponential-backoff retry budget has been exhausted.
    #[error("token refresh network error after retries exhausted")]
    TokenRefreshNetwork(#[source] reqwest::Error),

    /// The token response body could not be parsed into the expected JSON
    /// shape (missing required fields, wrong types, etc.).
    #[error("invalid token response payload: {0}")]
    InvalidTokenResponse(String),

    /// The on-disk token file exists but failed integrity checks (truncated,
    /// tampered, or format version mismatch). Treated as unauthenticated.
    #[error("stored OAuth token file is corrupted or unreadable")]
    CorruptedTokenFile,

    /// Filesystem I/O failure while reading or writing the token store.
    #[error("OAuth token store I/O error: {0}")]
    Io(#[from] std::io::Error),

    /// (De)serialization failure for the stored token payload.
    #[error("OAuth token (de)serialization error: {0}")]
    Serde(#[from] serde_json::Error),

    /// Encryption or decryption of the stored token blob failed. Wraps the
    /// existing [`SecretError`] type used elsewhere for at-rest crypto.
    #[error("OAuth token encryption error: {0}")]
    Crypto(#[from] SecretError),

    /// The loopback callback server could not bind to the requested port.
    #[error("failed to bind OAuth callback server on 127.0.0.1:{port}: {source}")]
    CallbackBindFailed {
        port: u16,
        #[source]
        source: std::io::Error,
    },

    /// The OAuth access token is not a well-formed JWT (segment count != 3).
    /// Returned by [`crate::codex::jwt::extract_chatgpt_account_id`].
    #[error("JWT access token is malformed (segment count != 3)")]
    JwtMalformed,

    /// The JWT payload segment failed to base64url-no-pad decode or the
    /// decoded bytes are not valid JSON.
    #[error("JWT payload failed to base64url-decode or parse as JSON")]
    JwtDecodeFailed,

    /// The JWT payload parsed successfully but the `chatgpt_account_id`
    /// claim is absent, non-string, or empty.
    #[error("JWT payload does not contain chatgpt_account_id claim")]
    JwtMissingAccountId,

    /// The user denied the authorization request during the GitHub device
    /// flow (terminal `access_denied` per RFC 8628 §3.5).
    ///
    /// Actionable guidance: re-run the login and approve the request at the
    /// verification URL; no further polling will happen for this grant.
    #[error("GitHub login was denied; re-run the login and approve the request at the verification URL")]
    DeviceFlowDenied,

    /// The device code expired before the user completed authorization
    /// (terminal `expired_token` per RFC 8628 §3.5, or local expiry guard).
    ///
    /// Actionable guidance: start a new login; the previous device code is
    /// no longer accepted by GitHub regardless of remaining poll budget.
    #[error("device code expired after {expires_in_secs} seconds; start a new login")]
    DeviceFlowExpired { expires_in_secs: u64 },

    /// The login was cancelled (watch channel signalled) before reaching a
    /// terminal state. Not an error condition on its own — surfaced so
    /// callers can distinguish deliberate cancellation from failures.
    #[error("device flow cancelled by user")]
    DeviceFlowCancelled,

    /// Transport-level failure (DNS, TLS, connection reset, etc.) during a
    /// GitHub device-flow request. Polling treats these as transient.
    #[error("GitHub device flow network error")]
    DeviceFlowNetwork(#[source] reqwest::Error),

    /// A GitHub device-flow endpoint returned a non-success HTTP status, or
    /// a success response whose payload was not the expected JSON shape
    /// (including unrecognized terminal error codes).
    ///
    /// `body` is a best-effort truncated snippet of the response body (or a
    /// parse-failure description), mirroring the token-exchange truncation
    /// rules; it must never contain token material.
    #[error("GitHub device flow HTTP {status}: {body}")]
    DeviceFlowHttp { status: u16, body: String },

    /// The Copilot token exchange returned 401/403: the GitHub credential
    /// is not entitled to Copilot (Req 3.3 terminal, Req 8.2 actionable).
    ///
    /// Carries only the HTTP status — the response body is deliberately
    /// dropped so no upstream content can leak token material (Req 10.2).
    /// Actionable guidance: the account needs an active Copilot
    /// subscription (Free, Pro, Pro+, Business, or Enterprise) with a
    /// Copilot Chat seat; PAT users must grant the `copilot` scope.
    #[error(
        "Copilot access missing — the account needs an active Copilot \
         subscription with a Copilot Chat seat (HTTP {status}); re-login \
         after enabling Copilot, or for a PAT add the copilot scope"
    )]
    CopilotEntitlementMissing { status: u16 },

    /// Transport-level failure (DNS, TLS, connection reset, etc.) during
    /// the Copilot token exchange. Treated as transient by the caller's
    /// single-retry policy and the background manager's backoff.
    #[error("Copilot token exchange network error")]
    CopilotTokenNetwork(#[source] reqwest::Error),

    /// The Copilot token exchange returned a non-success HTTP status
    /// (other than 401/403), or a success payload that was not the
    /// expected JSON shape.
    ///
    /// `body` is a best-effort truncated snippet of the response body (or a
    /// parse-failure description), mirroring the token-exchange truncation
    /// rules; it must never contain token material.
    #[error("Copilot token exchange failed with HTTP {status}: {body}")]
    CopilotTokenHttp { status: u16, body: String },
}
