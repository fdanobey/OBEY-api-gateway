//! HTTP client for System One classifier families: evaluation and model
//! listing against the configured base URL.
//!
//! Bounded by every axis the gateway demands of outbound integrations:
//! per-call timeouts, bounded exponential-backoff retries on transient
//! statuses (family-defined — Jev retries 429/529, Laya also retries 503 —
//! honoring `Retry-After`), no retries on configuration errors
//! (401/403/422), a non-retryable payload-limit class for 413, and a
//! response-body size cap. The API key is held opaquely and never appears in
//! logs, errors, or `Debug` output.

use std::fmt;
use std::time::Duration;

use super::models::{ModelsListing, SystemOneRequest, SystemOneResponse};
use super::SystemOneFamily;

/// Hard ceiling on evaluation response bodies (16 KiB is orders of
/// magnitude above any legitimate answer payload).
const MAX_RESPONSE_BODY_BYTES: usize = 16 * 1_024;
/// Hard ceiling for model listings; large catalogs exceed the answer cap.
const MAX_LISTING_BODY_BYTES: usize = 512 * 1_024;

/// Typed call outcomes mapped from HTTP/transport behavior.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum SystemOneCallError {
    /// Transient failure after retries were exhausted (family statuses
    /// persisted, e.g. 429/529/503).
    Transient { status: u16 },
    /// Authentication or permission rejected (401/403); operator action.
    Auth { status: u16 },
    /// Request rejected as invalid (422); configuration error.
    BadRequest { status: u16 },
    /// Request body exceeded the endpoint payload limit (413); the caller
    /// should reduce the state budget.
    PayloadTooLarge,
    /// Unexpected status or transport failure.
    Unavailable { reason: String },
    /// The call exceeded its timeout budget.
    Timeout,
}

impl SystemOneCallError {
    /// Content-free description safe for logs and admin surfaces.
    pub fn class(&self) -> &'static str {
        match self {
            Self::Transient { .. } => "transient",
            Self::Auth { .. } => "auth",
            Self::BadRequest { .. } => "bad_request",
            Self::PayloadTooLarge => "payload_too_large",
            Self::Unavailable { .. } => "unavailable",
            Self::Timeout => "timeout",
        }
    }
}

impl fmt::Display for SystemOneCallError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Transient { status } => write!(formatter, "transient failure (HTTP {status})"),
            Self::Auth { status } => write!(formatter, "authentication rejected (HTTP {status})"),
            Self::BadRequest { status } => write!(formatter, "request rejected (HTTP {status})"),
            Self::PayloadTooLarge => write!(formatter, "payload exceeded endpoint limit (HTTP 413)"),
            Self::Unavailable { reason } => write!(formatter, "endpoint unavailable: {reason}"),
            Self::Timeout => write!(formatter, "call timed out"),
        }
    }
}

impl std::error::Error for SystemOneCallError {}

/// Pooled HTTP client bound to one base URL, credential, and family.
pub struct SystemOneClient {
    http: reqwest::Client,
    family: SystemOneFamily,
    base_url: String,
    authorization_header: String,
    timeout: Duration,
    max_attempts: u8,
    backoff: Duration,
}

impl fmt::Debug for SystemOneClient {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter
            .debug_struct("SystemOneClient")
            .field("family", &self.family.name)
            .field("base_url", &self.base_url)
            .field("timeout_ms", &self.timeout.as_millis())
            .field("max_attempts", &self.max_attempts)
            .field("authorization", &"<redacted>")
            .finish()
    }
}

impl SystemOneClient {
    /// Build a client. `base_url` should already be validated (no query or
    /// fragment) by configuration validation; this constructor additionally
    /// enforces that the scheme is HTTPS and rejects any other transport,
    /// except for loopback/private self-hosted addresses where plain HTTP is
    /// an explicit deployment choice.
    pub fn new(
        family: SystemOneFamily,
        base_url: &str,
        api_key: &str,
        timeout: Duration,
        max_retries: u8,
        backoff: Duration,
    ) -> Result<Self, SystemOneCallError> {
        // Defense in depth: the Bearer credential travels on every request, so
        // the transport must be TLS. Configuration validation already rejects
        // non-HTTPS endpoints; enforcing it here keeps the invariant local and
        // guarantees the token can never be sent over cleartext. Loopback and
        // private-network self-hosted endpoints are the documented exception.
        match reqwest::Url::parse(base_url.trim()) {
            Ok(url)
                if url.scheme() == "https" || Self::is_private_host(&url) => {}
            _ => {
                return Err(SystemOneCallError::Unavailable {
                    reason: "endpoint base URL must use https (TLS)".to_owned(),
                });
            }
        }

        let http = reqwest::Client::builder()
            .redirect(reqwest::redirect::Policy::none())
            .connect_timeout(timeout.min(Duration::from_secs(10)))
            .pool_max_idle_per_host(2)
            .pool_idle_timeout(Duration::from_secs(60))
            .build()
            .map_err(|error| SystemOneCallError::Unavailable {
                reason: error.to_string(),
            })?;

        Ok(Self {
            http,
            family,
            base_url: base_url.trim_end_matches('/').to_string(),
            authorization_header: format!("Bearer {api_key}"),
            timeout,
            // Initial attempt + bounded retries.
            max_attempts: max_retries.saturating_add(1),
            backoff,
        })
    }

    /// True when the URL's host is loopback or a private-network address —
    /// the documented exception allowing plain-HTTP self-hosted endpoints
    /// (credentials never leave the machine/network).
    fn is_private_host(url: &reqwest::Url) -> bool {
        use std::net::IpAddr;
        let Some(host) = url.host_str() else {
            return false;
        };
        let host = host.trim_start_matches('[').trim_end_matches(']');
        if host.eq_ignore_ascii_case("localhost") {
            return true;
        }
        match host.parse::<IpAddr>() {
            Ok(IpAddr::V4(ip)) => ip.is_loopback() || ip.is_private(),
            Ok(IpAddr::V6(ip)) => ip.is_loopback(),
            Err(_) => false,
        }
    }

    /// Evaluate a System One request.
    pub async fn evaluate(
        &self,
        request: &SystemOneRequest,
    ) -> Result<SystemOneResponse, SystemOneCallError> {
        let url = format!("{}/v1/systemone", self.base_url);
        let body = serde_json::to_vec(request).map_err(|error| SystemOneCallError::Unavailable {
            reason: format!("serialize request: {error}"),
        })?;

        let response = self.execute_with_retries(url, body).await?;
        let bytes = read_capped_body(response, MAX_RESPONSE_BODY_BYTES).await?;
        serde_json::from_slice(&bytes).map_err(|error| SystemOneCallError::Unavailable {
            reason: format!("decode response: {error}"),
        })
    }

    /// List models available at the endpoint.
    pub async fn list_models(&self) -> Result<ModelsListing, SystemOneCallError> {
        let url = format!("{}/v1/models", self.base_url);
        let response = self.execute_with_retries(url, Vec::new()).await?;
        let bytes = read_capped_body(response, MAX_LISTING_BODY_BYTES).await?;
        serde_json::from_slice(&bytes).map_err(|error| SystemOneCallError::Unavailable {
            reason: format!("decode listing: {error}"),
        })
    }

    /// One request with retry loop: POST when `body` is non-empty, GET
    /// otherwise. The whole loop is bounded by the per-call timeout.
    async fn execute_with_retries(
        &self,
        url: String,
        body: Vec<u8>,
    ) -> Result<reqwest::Response, SystemOneCallError> {
        let deadline = tokio::time::Instant::now() + self.timeout;

        for attempt in 0..self.max_attempts.max(1) {
            let remaining = deadline.saturating_duration_since(tokio::time::Instant::now());
            if remaining.is_zero() {
                return Err(SystemOneCallError::Timeout);
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
                    if self.family.retryable_statuses.contains(&status)
                        && attempt + 1 < self.max_attempts
                    {
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
                    return Self::map_status(&self.family, response, status);
                }
                Err(error) => {
                    if error.is_timeout() {
                        return Err(SystemOneCallError::Timeout);
                    }
                    if attempt + 1 < self.max_attempts {
                        let wait = self
                            .backoff
                            .saturating_mul(1 << attempt)
                            .min(deadline.saturating_duration_since(tokio::time::Instant::now()));
                        if wait.is_zero() {
                            return Err(SystemOneCallError::Unavailable {
                                reason: "budget exhausted".to_string(),
                            });
                        }
                        tokio::time::sleep(wait).await;
                        continue;
                    }
                    return Err(SystemOneCallError::Unavailable {
                        reason: error.to_string(),
                    });
                }
            }
        }

        Err(SystemOneCallError::Unavailable {
            reason: "retry budget exhausted".to_string(),
        })
    }
    /// Map a non-retryable (or final) response status.
    fn map_status(
        family: &SystemOneFamily,
        response: reqwest::Response,
        status: u16,
    ) -> Result<reqwest::Response, SystemOneCallError> {
        match status {
            200..=299 => Ok(response),
            401 | 403 => Err(SystemOneCallError::Auth { status }),
            413 => Err(SystemOneCallError::PayloadTooLarge),
            422 => Err(SystemOneCallError::BadRequest { status }),
            _ if family.retryable_statuses.contains(&status) => {
                Err(SystemOneCallError::Transient { status })
            }
            _ => Err(SystemOneCallError::Unavailable {
                reason: format!("HTTP {status}"),
            }),
        }
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
) -> Result<Vec<u8>, SystemOneCallError> {
    let mut body = Vec::new();
    let mut stream = response;
    while let Some(chunk) = stream
        .chunk()
        .await
        .map_err(|error| SystemOneCallError::Unavailable {
            reason: format!("read body: {error}"),
        })?
    {
        if body.len() + chunk.len() > cap {
            return Err(SystemOneCallError::Unavailable {
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
        let client = SystemOneClient::new(
            SystemOneFamily::JEV,
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
    fn debug_shows_family_name() {
        let client = SystemOneClient::new(
            SystemOneFamily::LAYA,
            "https://api.laya-ai.com",
            "k",
            Duration::from_millis(1_000),
            0,
            Duration::from_millis(50),
        )
        .unwrap();
        assert!(format!("{client:?}").contains("laya"));
    }

    #[test]
    fn non_https_base_url_is_rejected() {
        assert!(SystemOneClient::new(
            SystemOneFamily::LAYA,
            "http://api.laya-ai.com",
            "k",
            Duration::from_millis(1_000),
            0,
            Duration::from_millis(50),
        )
        .is_err());
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
        assert_eq!(SystemOneCallError::Transient { status: 429 }.class(), "transient");
        assert_eq!(SystemOneCallError::Auth { status: 401 }.class(), "auth");
        assert_eq!(
            SystemOneCallError::BadRequest { status: 422 }.class(),
            "bad_request"
        );
        assert_eq!(SystemOneCallError::PayloadTooLarge.class(), "payload_too_large");
        assert_eq!(
            SystemOneCallError::Unavailable { reason: "x".into() }.class(),
            "unavailable"
        );
        assert_eq!(SystemOneCallError::Timeout.class(), "timeout");
    }
}
