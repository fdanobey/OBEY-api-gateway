//! HTTP client for the Jev classifier: System One evaluation and model
//! listing against the configured base URL.
//!
//! Bounded by every axis the gateway demands of outbound integrations:
//! per-call timeouts, at most two exponential-backoff retries on transient
//! statuses (429/529, honoring `Retry-After`), no retries on configuration
//! errors (401/403/422), and a response-body size cap. The API key is held
//! opaquely and never appears in logs, errors, or `Debug` output.

use std::fmt;
use std::time::Duration;

use super::models::{ModelsListing, SystemOneRequest, SystemOneResponse};

/// Transient statuses that merit a bounded retry.
const RETRYABLE_STATUSES: [u16; 2] = [429, 529];

/// Hard ceiling on evaluation response bodies (16 KiB is orders of
/// magnitude above any legitimate answer payload).
const MAX_RESPONSE_BODY_BYTES: usize = 16 * 1_024;
/// Hard ceiling for model listings; large catalogs exceed the answer cap.
const MAX_LISTING_BODY_BYTES: usize = 512 * 1_024;

/// Typed call outcomes mapped from HTTP/transport behavior.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum JevCallError {
    /// Transient failure after retries were exhausted (429/529 persisted).
    Transient { status: u16 },
    /// Authentication or permission rejected (401/403); operator action.
    Auth { status: u16 },
    /// Request rejected as invalid (422); configuration error.
    BadRequest { status: u16 },
    /// Unexpected status or transport failure.
    Unavailable { reason: String },
    /// The call exceeded its timeout budget.
    Timeout,
}

impl JevCallError {
    /// Content-free description safe for logs and admin surfaces.
    pub fn class(&self) -> &'static str {
        match self {
            Self::Transient { .. } => "transient",
            Self::Auth { .. } => "auth",
            Self::BadRequest { .. } => "bad_request",
            Self::Unavailable { .. } => "unavailable",
            Self::Timeout => "timeout",
        }
    }
}

impl fmt::Display for JevCallError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Transient { status } => write!(formatter, "transient failure (HTTP {status})"),
            Self::Auth { status } => write!(formatter, "authentication rejected (HTTP {status})"),
            Self::BadRequest { status } => write!(formatter, "request rejected (HTTP {status})"),
            Self::Unavailable { reason } => write!(formatter, "endpoint unavailable: {reason}"),
            Self::Timeout => write!(formatter, "call timed out"),
        }
    }
}

impl std::error::Error for JevCallError {}

/// Pooled HTTP client bound to one base URL and credential.
pub struct JevClient {
    http: reqwest::Client,
    base_url: String,
    authorization_header: String,
    timeout: Duration,
    max_attempts: u8,
    backoff: Duration,
}

impl fmt::Debug for JevClient {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter
            .debug_struct("JevClient")
            .field("base_url", &self.base_url)
            .field("timeout_ms", &self.timeout.as_millis())
            .field("max_attempts", &self.max_attempts)
            .field("authorization", &"<redacted>")
            .finish()
    }
}

impl JevClient {
    /// Build a client. `base_url` must already be validated (TLS, no query
    /// or fragment) by configuration validation.
    pub fn new(
        base_url: &str,
        api_key: &str,
        timeout: Duration,
        max_retries: u8,
        backoff: Duration,
    ) -> Result<Self, JevCallError> {
        let http = reqwest::Client::builder()
            .redirect(reqwest::redirect::Policy::none())
            .connect_timeout(timeout.min(Duration::from_secs(10)))
            .pool_max_idle_per_host(2)
            .pool_idle_timeout(Duration::from_secs(60))
            .build()
            .map_err(|error| JevCallError::Unavailable {
                reason: error.to_string(),
            })?;

        Ok(Self {
            http,
            base_url: base_url.trim_end_matches('/').to_string(),
            authorization_header: format!("Bearer {api_key}"),
            timeout,
            // Initial attempt + bounded retries.
            max_attempts: max_retries.saturating_add(1),
            backoff,
        })
    }

    /// Evaluate a System One request.
    pub async fn evaluate(
        &self,
        request: &SystemOneRequest,
    ) -> Result<SystemOneResponse, JevCallError> {
        let url = format!("{}/v1/systemone", self.base_url);
        let body = serde_json::to_vec(request).map_err(|error| JevCallError::Unavailable {
            reason: format!("serialize request: {error}"),
        })?;

        let response = self.execute_with_retries(url, body).await?;
        let bytes = read_capped_body(response, MAX_RESPONSE_BODY_BYTES).await?;
        serde_json::from_slice(&bytes).map_err(|error| JevCallError::Unavailable {
            reason: format!("decode response: {error}"),
        })
    }

    /// List models available at the endpoint.
    pub async fn list_models(&self) -> Result<ModelsListing, JevCallError> {
        let url = format!("{}/v1/models", self.base_url);
        let response = self.execute_with_retries(url, Vec::new()).await?;
        let bytes = read_capped_body(response, MAX_LISTING_BODY_BYTES).await?;
        serde_json::from_slice(&bytes).map_err(|error| JevCallError::Unavailable {
            reason: format!("decode listing: {error}"),
        })
    }

    /// One request with retry loop: POST when `body` is non-empty, GET
    /// otherwise. The whole loop is bounded by the per-call timeout.
    async fn execute_with_retries(
        &self,
        url: String,
        body: Vec<u8>,
    ) -> Result<reqwest::Response, JevCallError> {
        let deadline = tokio::time::Instant::now() + self.timeout;

        for attempt in 0..self.max_attempts.max(1) {
            let remaining = deadline.saturating_duration_since(tokio::time::Instant::now());
            if remaining.is_zero() {
                return Err(JevCallError::Timeout);
            }

            let mut request = self
                .http
                .request(
                    if body.is_empty() {
                        reqwest::Method::GET
                    } else {
                        reqwest::Method::POST
                    },
                    &url,
                )
                .timeout(remaining)
                .header(reqwest::header::AUTHORIZATION, &self.authorization_header);

            if !body.is_empty() {
                request = request.header(reqwest::header::CONTENT_TYPE, "application/json");
                request = request.body(body.clone());
            }

            let outcome = request.send().await;
            match outcome {
                Ok(response) => {
                    let status = response.status().as_u16();
                    if RETRYABLE_STATUSES.contains(&status) && attempt + 1 < self.max_attempts {
                        let delay = retry_delay(response.headers(), self.backoff, attempt);
                        // Honor Retry-After only within the remaining budget.
                        let wait = delay.min(
                            deadline
                                .saturating_duration_since(tokio::time::Instant::now())
                                .max(Duration::from_millis(1)),
                        );
                        tokio::time::sleep(wait).await;
                        continue;
                    }
                    return map_status(response, status);
                }
                Err(error) => {
                    if error.is_timeout() {
                        return Err(JevCallError::Timeout);
                    }
                    if attempt + 1 < self.max_attempts {
                        let wait = self
                            .backoff
                            .saturating_mul(1 << attempt)
                            .min(deadline.saturating_duration_since(tokio::time::Instant::now()));
                        if wait.is_zero() {
                            return Err(JevCallError::Unavailable {
                                reason: "budget exhausted".to_string(),
                            });
                        }
                        tokio::time::sleep(wait).await;
                        continue;
                    }
                    return Err(JevCallError::Unavailable {
                        reason: error.to_string(),
                    });
                }
            }
        }

        Err(JevCallError::Unavailable {
            reason: "retry budget exhausted".to_string(),
        })
    }
}

/// Map a non-retryable (or final) response status.
fn map_status(response: reqwest::Response, status: u16) -> Result<reqwest::Response, JevCallError> {
    match status {
        200..=299 => Ok(response),
        401 | 403 => Err(JevCallError::Auth { status }),
        422 => Err(JevCallError::BadRequest { status }),
        429 | 529 => Err(JevCallError::Transient { status }),
        _ => Err(JevCallError::Unavailable {
            reason: format!("HTTP {status}"),
        }),
    }
}

/// Compute the retry delay: `Retry-After` seconds when present, else
/// exponential backoff `base * 2^attempt`.
fn retry_delay(headers: &reqwest::header::HeaderMap, backoff: Duration, attempt: u8) -> Duration {
    if let Some(value) = headers
        .get(reqwest::header::RETRY_AFTER)
        .and_then(|value| value.to_str().ok())
    {
        if let Ok(seconds) = value.trim().parse::<u64>() {
            return Duration::from_secs(seconds);
        }
    }
    backoff.saturating_mul(1 << attempt)
}

/// Read a response body up to `cap` bytes; larger bodies are an error so a
/// hostile endpoint cannot exhaust memory.
async fn read_capped_body(
    response: reqwest::Response,
    cap: usize,
) -> Result<Vec<u8>, JevCallError> {
    let mut body = Vec::new();
    let mut stream = response;
    while let Some(chunk) = stream
        .chunk()
        .await
        .map_err(|error| JevCallError::Unavailable {
            reason: format!("read body: {error}"),
        })?
    {
        if body.len() + chunk.len() > cap {
            return Err(JevCallError::Unavailable {
                reason: "response body exceeded size cap".to_string(),
            });
        }
        body.extend_from_slice(&chunk);
    }
    Ok(body)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn debug_redacts_authorization() {
        let client = JevClient::new(
            "https://api.typesafe.ai",
            "super-secret-key",
            Duration::from_millis(1_000),
            2,
            Duration::from_millis(50),
        )
        .unwrap();

        let debug = format!("{client:?}");
        assert!(debug.contains("api.typesafe.ai"));
        assert!(!debug.contains("super-secret-key"));
        assert!(!debug.contains("Bearer"));
    }

    #[test]
    fn retry_delay_honors_retry_after_header() {
        let mut headers = reqwest::header::HeaderMap::new();
        headers.insert(
            reqwest::header::RETRY_AFTER,
            reqwest::header::HeaderValue::from_static("3"),
        );
        assert_eq!(
            retry_delay(&headers, Duration::from_millis(100), 0),
            Duration::from_secs(3)
        );
    }

    #[test]
    fn retry_delay_backs_off_exponentially() {
        let backoff = Duration::from_millis(100);
        assert_eq!(
            retry_delay(&reqwest::header::HeaderMap::new(), backoff, 0),
            Duration::from_millis(100)
        );
        assert_eq!(
            retry_delay(&reqwest::header::HeaderMap::new(), backoff, 1),
            Duration::from_millis(200)
        );
        assert_eq!(
            retry_delay(&reqwest::header::HeaderMap::new(), backoff, 2),
            Duration::from_millis(400)
        );
    }

    #[test]
    fn error_classes_are_content_free() {
        assert_eq!(JevCallError::Transient { status: 429 }.class(), "transient");
        assert_eq!(JevCallError::Auth { status: 401 }.class(), "auth");
        assert_eq!(
            JevCallError::BadRequest { status: 422 }.class(),
            "bad_request"
        );
        assert_eq!(
            JevCallError::Unavailable { reason: "x".into() }.class(),
            "unavailable"
        );
        assert_eq!(JevCallError::Timeout.class(), "timeout");
    }
}
