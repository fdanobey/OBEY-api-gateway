//! OAuth token exchange and refresh for the OpenAI authorization server.
//!
//! This module hosts the token endpoint types and network operations used by
//! the OAuth flow:
//! - [`TokenResponse`]: deserialized response from the OpenAI token endpoint
//! - `exchange_code` (tasks 3.2 & 3.4): exchanges an authorization code +
//!   PKCE verifier for an access/refresh token pair, with a single retry
//!   after a 2s delay on transient network errors
//! - `refresh_token` (task 3.3): refreshes an expired access token using the
//!   stored refresh token
//!
//! JSON parsing / property tests are covered in tasks 3.5 – 3.6.
//!
//! The endpoint and client identifier below are taken from the OpenAI public
//! OAuth app registration used for ChatGPT subscription sign-in.

// Satisfies Requirement 1.1's location constraint of the
// `codex-backend-translation` spec: the JWT claim extractor is callable as
// `crate::oauth::token::extract_chatgpt_account_id`. The implementation lives
// in `crate::codex::jwt` to keep Codex-specific logic co-located.
#[allow(unused_imports)]
pub use crate::codex::jwt::extract_chatgpt_account_id;

use serde::Deserialize;
use std::fmt;
use std::time::Duration;

/// OpenAI OAuth 2.0 token endpoint.
pub const TOKEN_ENDPOINT: &str = "https://auth.openai.com/oauth/token";

/// Delay before the single retry of [`exchange_code`] on a transient network
/// error (Req 3.4).
const EXCHANGE_RETRY_DELAY: Duration = Duration::from_secs(2);

/// Public client identifier for the OpenAI OAuth app used by the router.
pub const OPENAI_CLIENT_ID: &str = "app_EMoamEEZ73f0CkXaXp7hrann";

/// Successful response body returned by the OpenAI token endpoint for both
/// `authorization_code` and `refresh_token` grants.
///
/// `Debug` is implemented manually to redact the `access_token` and
/// `refresh_token` fields so secrets cannot accidentally leak through
/// tracing, panic messages, or `{:?}` formatting.
#[derive(Deserialize)]
pub struct TokenResponse {
    pub access_token: String,
    pub refresh_token: Option<String>,
    pub expires_in: u64,
    pub token_type: String,
    #[serde(default)]
    pub scope: Option<String>,
}

impl fmt::Debug for TokenResponse {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("TokenResponse")
            .field("access_token", &"<redacted>")
            .field(
                "refresh_token",
                &self.refresh_token.as_ref().map(|_| "<redacted>"),
            )
            .field("expires_in", &self.expires_in)
            .field("token_type", &self.token_type)
            .field("scope", &self.scope)
            .finish()
    }
}

/// Maximum number of bytes of an error response body to retain in
/// [`OAuthError::TokenExchangeHttp`] / [`OAuthError::TokenRefreshHttp`].
///
/// Keeps diagnostic messages bounded and avoids shipping arbitrarily large
/// provider responses into logs or error surfaces.
const ERROR_BODY_MAX_BYTES: usize = 256;

/// Truncate a response body to at most [`ERROR_BODY_MAX_BYTES`] bytes on a
/// UTF-8 character boundary so the resulting `String` never panics on display.
fn truncate_error_body(body: String) -> String {
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

/// Exchange an authorization code for an access + refresh token pair using
/// PKCE (per [RFC 7636]).
///
/// This is the single-attempt implementation. External callers should use
/// [`exchange_code`], which wraps this with a single retry on transient
/// network errors (Req 3.4).
///
/// Posts a `grant_type=authorization_code` request to [`TOKEN_ENDPOINT`] with
/// the provided `code`, `code_verifier`, and `redirect_uri`.
///
/// # Security
///
/// - `code`, `code_verifier`, and the successful response body are **never**
///   logged. The `#[tracing::instrument]` annotation skips the secret-bearing
///   parameters and only records `redirect_uri` in the span.
/// - On HTTP error, the response body is truncated to
///   [`ERROR_BODY_MAX_BYTES`] bytes. The OpenAI token endpoint does not
///   include token values in its error responses.
///
/// # Retries
///
/// This function performs **no retries**; transport errors are returned
/// directly as [`OAuthError::TokenExchangeNetwork`]. The single-retry policy
/// is implemented by [`exchange_code`].
///
/// [RFC 7636]: https://datatracker.ietf.org/doc/html/rfc7636
#[tracing::instrument(
    skip(client, code, code_verifier),
    fields(redirect_uri = %redirect_uri),
)]
pub(crate) async fn exchange_code_once(
    client: &reqwest::Client,
    code: &str,
    code_verifier: &str,
    redirect_uri: &str,
) -> Result<TokenResponse, crate::oauth::error::OAuthError> {
    use crate::oauth::error::OAuthError;

    let params = [
        ("grant_type", "authorization_code"),
        ("client_id", OPENAI_CLIENT_ID),
        ("code", code),
        ("redirect_uri", redirect_uri),
        ("code_verifier", code_verifier),
    ];

    let response = client
        .post(TOKEN_ENDPOINT)
        .form(&params)
        .send()
        .await
        .map_err(OAuthError::TokenExchangeNetwork)?;

    let status = response.status();

    if status.is_success() {
        response
            .json::<TokenResponse>()
            .await
            .map_err(|e| OAuthError::InvalidTokenResponse(e.to_string()))
    } else {
        let body = response.text().await.unwrap_or_default();
        Err(OAuthError::TokenExchangeHttp {
            status: status.as_u16(),
            body: truncate_error_body(body),
        })
    }
}

/// Exchange an authorization code for an access + refresh token pair with a
/// single retry on transient network errors (Req 3.4).
///
/// Calls [`exchange_code_once`]. If it returns
/// [`OAuthError::TokenExchangeNetwork`], waits [`EXCHANGE_RETRY_DELAY`] and
/// retries exactly once. HTTP errors and parse errors return immediately
/// without retry.
///
/// # Security
///
/// See [`exchange_code_once`]; secret parameters are not logged.
#[tracing::instrument(
    skip(client, code, code_verifier),
    fields(redirect_uri = %redirect_uri),
)]
pub async fn exchange_code(
    client: &reqwest::Client,
    code: &str,
    code_verifier: &str,
    redirect_uri: &str,
) -> Result<TokenResponse, crate::oauth::error::OAuthError> {
    use crate::oauth::error::OAuthError;

    match exchange_code_once(client, code, code_verifier, redirect_uri).await {
        Ok(token) => Ok(token),
        Err(OAuthError::TokenExchangeNetwork(_)) => {
            tokio::time::sleep(EXCHANGE_RETRY_DELAY).await;
            exchange_code_once(client, code, code_verifier, redirect_uri).await
        }
        Err(other) => Err(other),
    }
}

/// Refresh an expired access token using the stored refresh token.
///
/// Posts a `grant_type=refresh_token` request to [`TOKEN_ENDPOINT`].
///
/// # Security
///
/// - `refresh_token` and the successful response body are **never** logged.
///   The `#[tracing::instrument]` annotation skips both the `client` and the
///   `refresh_token` argument so neither appears in any span.
/// - On HTTP error, the response body is truncated to
///   [`ERROR_BODY_MAX_BYTES`] bytes before being included in
///   [`OAuthError::TokenRefreshHttp`]. The OpenAI token endpoint does not
///   include token values in its error responses.
///
/// # Error mapping
///
/// - Transport failure → [`OAuthError::TokenRefreshNetwork`]
/// - HTTP 401 / 403    → [`OAuthError::SessionExpired`] (refresh token is
///   no longer valid; the user must re-authenticate — see Req 5.4)
/// - Other non-success → [`OAuthError::TokenRefreshHttp`]
/// - JSON parse error  → [`OAuthError::InvalidTokenResponse`]
///
/// # Retries
///
/// This function performs **no retries**. The exponential-backoff policy
/// described in the design doc (2s / 4s / 8s) is layered on top by the
/// background refresh task in task 5.x.
#[tracing::instrument(skip(client, refresh_token))]
pub async fn refresh_access_token(
    client: &reqwest::Client,
    refresh_token: &str,
) -> Result<TokenResponse, crate::oauth::error::OAuthError> {
    use crate::oauth::error::OAuthError;

    let params = [
        ("grant_type", "refresh_token"),
        ("client_id", OPENAI_CLIENT_ID),
        ("refresh_token", refresh_token),
    ];

    let response = client
        .post(TOKEN_ENDPOINT)
        .form(&params)
        .send()
        .await
        .map_err(OAuthError::TokenRefreshNetwork)?;

    let status = response.status();

    if status.is_success() {
        return response
            .json::<TokenResponse>()
            .await
            .map_err(|e| OAuthError::InvalidTokenResponse(e.to_string()));
    }

    if status.as_u16() == 401 || status.as_u16() == 403 {
        return Err(OAuthError::SessionExpired {
            status: status.as_u16(),
        });
    }

    let body = response.text().await.unwrap_or_default();
    Err(OAuthError::TokenRefreshHttp {
        status: status.as_u16(),
        body: truncate_error_body(body),
    })
}

#[cfg(test)]
mod tests {
    use super::TokenResponse;

    #[test]
    fn parse_valid_full_response() {
        let json = r#"{
            "access_token": "at-123",
            "refresh_token": "rt-456",
            "expires_in": 3600,
            "token_type": "Bearer",
            "scope": "openid profile email offline_access"
        }"#;

        let parsed: TokenResponse =
            serde_json::from_str(json).expect("valid full response should parse");

        assert_eq!(parsed.access_token, "at-123");
        assert_eq!(parsed.refresh_token.as_deref(), Some("rt-456"));
        assert_eq!(parsed.expires_in, 3600);
        assert_eq!(parsed.token_type, "Bearer");
        assert_eq!(
            parsed.scope.as_deref(),
            Some("openid profile email offline_access")
        );
    }

    #[test]
    fn parse_valid_minimal_response() {
        let json = r#"{
            "access_token": "at-only",
            "expires_in": 60,
            "token_type": "Bearer"
        }"#;

        let parsed: TokenResponse =
            serde_json::from_str(json).expect("minimal response should parse");

        assert_eq!(parsed.access_token, "at-only");
        assert!(parsed.refresh_token.is_none());
        assert_eq!(parsed.expires_in, 60);
        assert_eq!(parsed.token_type, "Bearer");
        assert!(parsed.scope.is_none());
    }

    #[test]
    fn parse_missing_access_token() {
        let json = r#"{
            "refresh_token": "rt-456",
            "expires_in": 3600,
            "token_type": "Bearer"
        }"#;

        let result = serde_json::from_str::<TokenResponse>(json);
        assert!(
            result.is_err(),
            "response without access_token must fail to parse"
        );
    }

    #[test]
    fn parse_wrong_type_expires_in() {
        let json = r#"{
            "access_token": "at-123",
            "expires_in": "not a number",
            "token_type": "Bearer"
        }"#;

        let result = serde_json::from_str::<TokenResponse>(json);
        assert!(result.is_err(), "expires_in as a string must fail to parse");
    }

    #[test]
    fn parse_empty_object() {
        let result = serde_json::from_str::<TokenResponse>("{}");
        assert!(result.is_err(), "empty object must fail to parse");
    }

    #[test]
    fn parse_malformed_json() {
        let result = serde_json::from_str::<TokenResponse>("not json at all");
        assert!(result.is_err(), "malformed JSON must fail to parse");
    }

    #[test]
    fn debug_redacts_tokens() {
        let token = TokenResponse {
            access_token: "secret-AT".to_string(),
            refresh_token: Some("secret-RT".to_string()),
            expires_in: 3600,
            token_type: "Bearer".to_string(),
            scope: None,
        };

        let rendered = format!("{:?}", token);

        assert!(
            rendered.contains("<redacted>"),
            "debug output should mark secrets as redacted: {rendered}"
        );
        assert!(
            !rendered.contains("secret-AT"),
            "debug output must not contain the raw access_token: {rendered}"
        );
        assert!(
            !rendered.contains("secret-RT"),
            "debug output must not contain the raw refresh_token: {rendered}"
        );
    }
}
