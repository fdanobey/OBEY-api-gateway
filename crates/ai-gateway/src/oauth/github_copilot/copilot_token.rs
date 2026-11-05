#![allow(dead_code)]
//! GitHub-token → Copilot-token exchange (task 4 of the
//! `github-copilot-oauth-provider` spec).
//!
//! [`exchange_copilot_token`] performs `GET` on the Copilot internal token
//! endpoint with the GitHub OAuth token as the Bearer credential (Req 3.1),
//! parses the `{token, expires_at, refresh_in, endpoints.api, sku}` payload,
//! and resolves the effective API base (`endpoints.api` override with the
//! `https://api.githubcopilot.com` fallback).
//!
//! # Retry policy
//!
//! A single retry after a 2 s delay on transient network errors, mirroring
//! `exchange_code` in `oauth/token.rs`. The deeper 2 s/4 s/8 s backoff of
//! Req 3.2 belongs to the background manager (task 5), not this call.
//! 401/403 are terminal immediately — no retry storm (Req 3.3).
//!
//! # Headers
//!
//! Only `Authorization: Bearer …` and `Accept: application/json` are sent
//! here. The first-party client header set is the adapter's concern
//! (task 7), not the exchange's.
//!
//! # Security invariants
//!
//! - The GitHub token and the returned Copilot token are never logged:
//!   the `#[tracing::instrument]` annotations skip every secret-bearing
//!   argument, and [`CopilotToken`] carries a manual redacted `Debug` impl.
//! - 401/403 map to [`OAuthError::CopilotEntitlementMissing`], which carries
//!   only the status code — no response body, no token material
//!   (Req 8.2 / Req 10.2).
//! - Other error bodies are truncated (256 bytes, UTF-8-boundary-safe)
//!   exactly like `oauth/token.rs` and `device_flow.rs`.

use std::fmt;
use std::time::Duration;

use serde::Deserialize;

use super::super::error::{truncate_error_body, OAuthError};

/// Copilot internal token endpoint (production constant, Req 3.1). Tests
/// inject a wiremock base URL through the `token_url` parameter instead.
pub const COPILOT_TOKEN_URL: &str = "https://api.github.com/copilot_internal/v2/token";

/// Default Copilot API base used when the exchange response carries no
/// `endpoints.api` override (Req 3.1 fallback).
pub const DEFAULT_COPILOT_API_BASE: &str = "https://api.githubcopilot.com";

/// Delay before the single retry of [`exchange_copilot_token`] on a
/// transient network error (mirrors `EXCHANGE_RETRY_DELAY` in
/// `oauth/token.rs`).
const EXCHANGE_RETRY_DELAY: Duration = Duration::from_secs(2);

/// `endpoints` object of the token exchange response.
///
/// Only `api` is modeled (Req 3.1); the live endpoint also returns fields
/// like `codex` and `individually` that the gateway does not use, and
/// unknown fields are ignored by serde.
#[derive(Clone, Deserialize)]
pub struct CopilotTokenEndpoints {
/// Copilot API base URL override (e.g. `https://api.githubcopilot.com`
/// or a regional override). Not a credential.
#[serde(default)]
pub api: Option<String>,
}

impl fmt::Debug for CopilotTokenEndpoints {
fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
f.debug_struct("CopilotTokenEndpoints")
            .field("api", &self.api)
            .finish()
    }
}

/// Successful response from `GET /copilot_internal/v2/token`.
///
/// `token` is secret-shaped (the short-lived Copilot API bearer) and is
/// redacted by the manual `Debug` impl; every other field is non-secret
/// metadata used for refresh scheduling and routing.
#[derive(Clone, Deserialize)]
pub struct CopilotToken {
    /// Short-lived Copilot API bearer token. Never logged.
    pub token: String,
    /// Expiry of [`Self::token`] as a Unix timestamp in seconds.
    pub expires_at: u64,
    /// Suggested seconds-before-expiry to re-exchange; the background
    /// manager clamps this against its own pre-expiry window.
    #[serde(default)]
    pub refresh_in: Option<u64>,
    /// Endpoint overrides returned by the exchange.
    #[serde(default)]
    pub endpoints: Option<CopilotTokenEndpoints>,
    /// Copilot SKU reported by the exchange, when present.
    #[serde(default)]
    pub sku: Option<String>,
}

impl fmt::Debug for CopilotToken {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("CopilotToken")
            .field("token", &"<redacted>")
            .field("expires_at", &self.expires_at)
            .field("refresh_in", &self.refresh_in)
            .field("endpoints", &self.endpoints)
            .field("sku", &self.sku)
            .finish()
    }
}

impl CopilotToken {
    /// Effective Copilot API base (Req 3.1): the `endpoints.api` override
    /// when it is an HTTPS URL with a non-empty host (or an HTTP URL on a
    /// loopback host — the wiremock servers tests use), otherwise
    /// [`DEFAULT_COPILOT_API_BASE`]. Rejecting non-HTTPS, non-loopback or
    /// hostless overrides before they become dispatch base URLs closes the
    /// SSRF-shaped hole of blindly trusting the exchange payload.
    pub fn api_base(&self) -> &str {
        self.endpoints
            .as_ref()
            .and_then(|endpoints| endpoints.api.as_deref())
            .filter(|api| {
                match reqwest::Url::parse(api) {
                    Ok(url) => {
                        let host_ok = url.host_str().is_some_and(|h| !h.is_empty());
                        let scheme_ok = url.scheme() == "https"
                            || (url.scheme() == "http"
                                && url.host_str().is_some_and(host_is_loopback));
                        if host_ok && scheme_ok {
                            true
                        } else {
                            tracing::warn!(
                                api,
                                "ignoring insecure Copilot endpoints.api override (must be https with a host, or http on a loopback host); using default base"
                            );
                            false
                        }
                    }
                    Err(_) => {
                        tracing::warn!(
                            api,
                            "ignoring unparseable Copilot endpoints.api override; using default base"
                        );
                        false
                    }
                }
            })
            .unwrap_or(DEFAULT_COPILOT_API_BASE)
    }
}

/// Loopback host check for the [`CopilotToken::api_base`] guard:
/// production GitHub endpoints are HTTPS-only, so plain HTTP is
/// accepted only when the host is loopback (`127.0.0.0/8`, `::1`, or
/// `localhost`) — the shape wiremock test servers use. The posture
/// against non-loopback HTTP overrides is unchanged.
fn host_is_loopback(host: &str) -> bool {
    let host = host.trim_matches(|c| c == '[' || c == ']');
    if host.eq_ignore_ascii_case("localhost") {
        return true;
    }
    if let Some(rest) = host.strip_prefix("127.") {
        let octets = rest.split('.').collect::<Vec<_>>();
        return host.split('.').count() == 4
            && octets
                .iter()
                .all(|octet| !octet.is_empty() && octet.parse::<u8>().is_ok());
    }
    host == "::1"
}

/// Single-attempt implementation of the exchange; external callers should
/// use [`exchange_copilot_token`], which wraps this with the single retry
/// on transient network errors.
///
/// # Error mapping
///
/// - Transport failure → [`OAuthError::CopilotTokenNetwork`]
/// - HTTP 401 / 403 → [`OAuthError::CopilotEntitlementMissing`] (terminal,
///   never retried — Req 3.3 / Req 8.2)
/// - Other non-success → [`OAuthError::CopilotTokenHttp`] (truncated body)
/// - Malformed JSON / missing required fields →
///   [`OAuthError::CopilotTokenHttp`] with the HTTP status and a truncated
///   parse description
#[tracing::instrument(skip(client, github_token))]
async fn exchange_copilot_token_once(
    client: &reqwest::Client,
    github_token: &str,
    token_url: &str,
) -> Result<CopilotToken, OAuthError> {
    let response = client
        .get(token_url)
        .bearer_auth(github_token)
        .header("accept", "application/json")
        .send()
        .await
        .map_err(OAuthError::CopilotTokenNetwork)?;

    let status = response.status();

    if !status.is_success() {
        let code = status.as_u16();
        if code == 401 || code == 403 {
            // Req 3.3 / Req 8.2: entitlement/auth failure is terminal and
            // carries only the status — the body is deliberately dropped so
            // no upstream content can leak token material (Req 10.2).
            return Err(OAuthError::CopilotEntitlementMissing { status: code });
        }
        let body = response.text().await.unwrap_or_default();
        return Err(OAuthError::CopilotTokenHttp {
            status: code,
            body: truncate_error_body(body),
        });
    }

    response
        .json::<CopilotToken>()
        .await
        .map_err(|e| OAuthError::CopilotTokenHttp {
            status: status.as_u16(),
            body: truncate_error_body(e.to_string()),
        })
}

/// Exchange a GitHub OAuth token (or PAT) for a short-lived Copilot API
/// token (Req 3.1), with a single retry on transient network errors.
///
/// `GET {token_url}` with `Authorization: Bearer {github_token}` and
/// `Accept: application/json`. Production callers pass [`COPILOT_TOKEN_URL`];
/// tests inject a wiremock base URL.
///
/// # Retries
///
/// Exactly one retry after [`EXCHANGE_RETRY_DELAY`] (2 s) when the first
/// attempt fails at the transport level, mirroring `exchange_code` in
/// `oauth/token.rs`. HTTP errors (including 401/403) and parse errors
/// return immediately without retry.
///
/// # Security
///
/// `github_token` and the successful response body are never logged; the
/// `#[tracing::instrument]` annotation skips both secret-bearing arguments.
#[tracing::instrument(skip(client, github_token))]
pub async fn exchange_copilot_token(
    client: &reqwest::Client,
    github_token: &str,
    token_url: &str,
) -> Result<CopilotToken, OAuthError> {
    match exchange_copilot_token_once(client, github_token, token_url).await {
        Ok(token) => Ok(token),
        Err(OAuthError::CopilotTokenNetwork(_)) => {
            tokio::time::sleep(EXCHANGE_RETRY_DELAY).await;
            exchange_copilot_token_once(client, github_token, token_url).await
        }
        Err(other) => Err(other),
    }
}

#[cfg(test)]
mod tests {
use super::*;
use crate::oauth::github_copilot::test_support::test_client;
use wiremock::matchers::{header, method, path};
use wiremock::{Mock, MockServer, ResponseTemplate};

const GITHUB_TOKEN: &str = "gho-test-github-token";
const COPILOT_BEARER: &str = "tid=test-copilot-token;exp=1770000000;sku=copilot_pro";
const TOKEN_PATH: &str = "/copilot_internal/v2/token";

fn token_url(server: &MockServer) -> String {
format!("{server}{TOKEN_PATH}", server = server.uri())
}

fn full_token_body() -> serde_json::Value {
        serde_json::json!({
            "token": COPILOT_BEARER,
            "expires_at": 1_770_000_000u64,
            "refresh_in": 1500u64,
            "endpoints": { "api": "https://api.example-copilot.test" },
            "sku": "copilot_pro",
            // Unknown fields from the live endpoint must be ignored.
            "chat_enabled": true,
            "code_quote_enabled": true
        })
    }

    /// Req 3.1: GET with Bearer GitHub token + Accept JSON; all documented
    /// fields parse; `endpoints.api` override is honored; Debug redacts the
    /// Copilot token.
    #[tokio::test]
    async fn exchange_request_shape_parses_all_fields_and_honors_endpoints_api() {
        let server = MockServer::start().await;
        Mock::given(method("GET"))
            .and(path(TOKEN_PATH))
            .and(header("authorization", format!("Bearer {GITHUB_TOKEN}")))
            .and(header("accept", "application/json"))
            .respond_with(ResponseTemplate::new(200).set_body_json(full_token_body()))
            .expect(1)
            .mount(&server)
            .await;

        let token =
            exchange_copilot_token(&test_client(), GITHUB_TOKEN, &token_url(&server))
                .await
                .expect("exchange should succeed");

        assert_eq!(token.token, COPILOT_BEARER);
        assert_eq!(token.expires_at, 1_770_000_000);
        assert_eq!(token.refresh_in, Some(1500));
        assert_eq!(token.sku.as_deref(), Some("copilot_pro"));
        assert_eq!(token.api_base(), "https://api.example-copilot.test");

        let rendered = format!("{token:?}");
        assert!(
            rendered.contains("<redacted>"),
            "debug output should mark the token redacted: {rendered}"
        );
        assert!(
            !rendered.contains(COPILOT_BEARER),
            "debug output must not contain the raw Copilot token: {rendered}"
        );

        server.verify().await;
    }

    /// Req 3.1 fallback: without `endpoints.api` the production default
    /// base is used — both when `endpoints` is absent and when it carries
    /// no usable `api` value.
    #[test]
    fn api_base_falls_back_to_production_default() {
        let absent: CopilotToken =
            serde_json::from_str(r#"{"token":"t","expires_at":1}"#).unwrap();
        assert_eq!(absent.api_base(), DEFAULT_COPILOT_API_BASE);

        let empty_api: CopilotToken =
            serde_json::from_str(r#"{"token":"t","expires_at":1,"endpoints":{}}"#)
                .unwrap();
        assert_eq!(empty_api.api_base(), DEFAULT_COPILOT_API_BASE);
    }

/// Req 3.1 (SSRF guard): an `endpoints.api` override is honored only
/// when it parses as an HTTPS URL with a non-empty host, or as an HTTP
/// URL whose host is loopback (wiremock test shape); insecure
/// non-loopback, hostless, or unparseable values fall back to the
/// production default instead of becoming the dispatch base URL.
#[test]
fn api_base_override_requires_https_url_with_host() {
    fn override_token(api: &str) -> CopilotToken {
        serde_json::from_str(&format!(
            r#"{{"token":"t","expires_at":1,"endpoints":{{"api":"{api}"}}}}"#
        ))
        .unwrap()
    }

    // Accepted: regional HTTPS override.
    assert_eq!(
        override_token("https://api.example-copilot.test").api_base(),
        "https://api.example-copilot.test"
    );

    // Accepted: HTTP on a loopback host (wiremock test servers).
    for good in [
        "http://127.0.0.1:12345",
        "http://localhost:12345",
        "http://[::1]:12345",
    ] {
        assert_eq!(
            override_token(good).api_base(),
            good,
            "loopback override {good:?} must be accepted"
        );
    }

    // Rejected: plain HTTP on a non-loopback host, hostless scheme-only,
    // relative path, and embedded credentials all fall back to the
    // default base.
    for bad in [
        "http://api.example-copilot.test",
        "http://10.0.0.1",
        "https://",
        "/evil/path",
        "not a url",
    ] {
        assert_eq!(
            override_token(bad).api_base(),
            DEFAULT_COPILOT_API_BASE,
            "override {bad:?} must be rejected"
        );
    }
}

    /// Req 10.2: a 200 payload without the required `token` field fails
    /// with an actionable error naming the missing field.
    #[tokio::test]
    async fn missing_token_field_is_actionable_error() {
        let server = MockServer::start().await;
        Mock::given(method("GET"))
            .and(path(TOKEN_PATH))
            .respond_with(ResponseTemplate::new(200).set_body_json(
                serde_json::json!({"expires_at": 123, "refresh_in": 456}),
            ))
            .expect(1)
            .mount(&server)
            .await;

        let result =
            exchange_copilot_token(&test_client(), GITHUB_TOKEN, &token_url(&server)).await;

        match result {
            Err(OAuthError::CopilotTokenHttp { status: 200, body }) => {
                assert!(
                    body.contains("token"),
                    "parse error must name the missing field, got: {body}"
                );
            }
            other => panic!("expected CopilotTokenHttp, got {other:?}"),
        }
        server.verify().await;
    }

    /// Req 3.3 / Req 8.2: 401 maps to the entitlement error variant with an
    /// actionable message, is terminal (exactly one request — no retry
    /// storm), and leaks neither token.
    #[tokio::test]
    async fn unauthorized_maps_to_entitlement_error_without_retry() {
        let server = MockServer::start().await;
        Mock::given(method("GET"))
            .and(path(TOKEN_PATH))
            .respond_with(ResponseTemplate::new(401).set_body_string("token is invalid"))
            .expect(1)
            .mount(&server)
            .await;

        let result =
            exchange_copilot_token(&test_client(), GITHUB_TOKEN, &token_url(&server)).await;

        match result {
            Err(err @ OAuthError::CopilotEntitlementMissing { status: 401 }) => {
                let message = err.to_string();
                assert!(
                    message.contains("Copilot access missing"),
                    "message must name the entitlement requirement: {message}"
                );
                assert!(
                    message.contains("Copilot Chat"),
                    "message must name the Copilot Chat seat requirement: {message}"
                );
                assert!(
                    !message.contains(GITHUB_TOKEN),
                    "message must not contain the GitHub token: {message}"
                );
            }
            other => panic!("expected CopilotEntitlementMissing, got {other:?}"),
        }
        // expect(1) proves the 401 was not retried.
        server.verify().await;
    }

    /// Req 3.3 / Req 8.2: 403 behaves identically to 401.
    #[tokio::test]
    async fn forbidden_maps_to_entitlement_error_without_retry() {
        let server = MockServer::start().await;
        Mock::given(method("GET"))
            .and(path(TOKEN_PATH))
            .respond_with(ResponseTemplate::new(403).set_body_string("forbidden"))
            .expect(1)
            .mount(&server)
            .await;

        let result =
            exchange_copilot_token(&test_client(), GITHUB_TOKEN, &token_url(&server)).await;

        match result {
            Err(OAuthError::CopilotEntitlementMissing { status: 403 }) => {}
            other => panic!("expected CopilotEntitlementMissing, got {other:?}"),
        }
        server.verify().await;
    }

    /// A transient network failure (client timeout on a stalled response)
    /// is retried exactly once after the 2 s delay and then succeeds.
    #[tokio::test]
    async fn transient_network_error_retried_once_then_succeeds() {
        let server = MockServer::start().await;
        // First attempt: response stalls past the client timeout.
        Mock::given(method("GET"))
            .and(path(TOKEN_PATH))
            .respond_with(
                ResponseTemplate::new(200).set_delay(Duration::from_secs(60)),
            )
            .up_to_n_times(1)
            .mount(&server)
            .await;
        // Second attempt: immediate success.
        Mock::given(method("GET"))
            .and(path(TOKEN_PATH))
            .respond_with(ResponseTemplate::new(200).set_body_json(full_token_body()))
            .expect(1)
            .mount(&server)
            .await;

        let client = reqwest::Client::builder()
            .timeout(Duration::from_millis(250))
            .build()
            .unwrap();

        let started = std::time::Instant::now();
        let token =
            exchange_copilot_token(&client, GITHUB_TOKEN, &token_url(&server))
                .await
                .expect("single retry after transient failure should succeed");

        assert_eq!(token.api_base(), "https://api.example-copilot.test");
        assert!(
            started.elapsed() >= EXCHANGE_RETRY_DELAY,
            "retry must wait the 2 s delay, took {:?}",
            started.elapsed()
        );
        assert_eq!(
            server.received_requests().await.unwrap().len(),
            2,
            "exactly one retry (two requests total)"
        );
        server.verify().await;
    }

    /// Non-entitlement HTTP errors keep a truncated body (256-byte cap,
    /// mirroring `oauth/token.rs`).
    #[tokio::test]
    async fn other_http_error_truncates_body() {
        let server = MockServer::start().await;
        Mock::given(method("GET"))
            .and(path(TOKEN_PATH))
            .respond_with(ResponseTemplate::new(500).set_body_string("x".repeat(1024)))
            .expect(1)
            .mount(&server)
            .await;

        let result =
            exchange_copilot_token(&test_client(), GITHUB_TOKEN, &token_url(&server)).await;

        match result {
            Err(OAuthError::CopilotTokenHttp { status: 500, body }) => {
                assert!(body.len() <= 256, "body must be truncated, got {}", body.len());
            }
            other => panic!("expected CopilotTokenHttp, got {other:?}"),
        }
        server.verify().await;
    }

    /// Req 8.2: the entitlement message is standalone-actionable and
    /// carries the diagnostic status.
    #[test]
    fn entitlement_error_message_names_copilot_chat_seat_requirement() {
        let message =
            OAuthError::CopilotEntitlementMissing { status: 403 }.to_string();
        assert!(message.contains("Copilot access missing"), "{message}");
        assert!(message.contains("Copilot Chat seat"), "{message}");
        assert!(message.contains("403"), "{message}");
    }
}
