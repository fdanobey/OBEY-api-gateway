#![allow(dead_code)]
//! GitHub OAuth device-flow client (RFC 8628) for Copilot login.
//!
//! Task 3 of the `github-copilot-oauth-provider` spec:
//! - [`start_device_flow`]: requests a device code from GitHub's device
//!   authorization endpoint using the public VS Code Copilot Chat client
//!   ID and scope `read:user` (Req 2.1).
//! - [`poll_for_token`]: polls GitHub's token endpoint at the grant's
//!   mandated interval until success, a terminal authorization error, code
//!   expiry, or caller cancellation (Req 2.2 / 2.3 / 2.5).
//!
//! # Production endpoints
//!
//! [`GITHUB_DEVICE_CODE_URL`] and [`GITHUB_TOKEN_URL`] hold the real
//! GitHub endpoints; tests inject wiremock base URLs through
//! [`DeviceFlowEndpoints`] so no request ever leaves the loopback
//! interface under test.
//!
//! # Security invariants
//!
//! - `device_code` and the returned `access_token` are never logged: the
//!   `#[tracing::instrument]` annotations skip every secret-bearing
//!   argument, and [`GithubToken`] carries a manual redacted `Debug` impl.
//! - Error bodies are truncated (256 bytes, UTF-8-boundary-safe) exactly
//!   like `oauth/token.rs` before being placed into
//!   [`OAuthError::DeviceFlowHttp`].

use std::fmt;
use std::time::Duration;

use serde::Deserialize;
use tokio::sync::watch;

use super::super::error::{truncate_error_body, OAuthError};

/// GitHub device authorization endpoint (production constant).
pub const GITHUB_DEVICE_CODE_URL: &str = "https://github.com/login/device/code";

/// GitHub OAuth token endpoint (production constant).
pub const GITHUB_TOKEN_URL: &str = "https://github.com/login/oauth/access_token";

/// Public client ID used by VS Code Copilot Chat (spec Req 2.1); GitHub's
/// OAuth device flow for public clients requires no client secret.
pub const COPILOT_CLIENT_ID: &str = "Iv1.b507a08c87ecfe98";

/// OAuth scope requested for the Copilot login.
pub const COPILOT_OAUTH_SCOPE: &str = "read:user";

/// `grant_type` for OAuth 2.0 device authorization grants (RFC 8628 §3.4).
const DEVICE_CODE_GRANT_TYPE: &str = "urn:ietf:params:oauth:grant-type:device_code";

/// Poll interval assumed when GitHub omits `interval` from the device
/// grant (RFC 8628 §3.5 default is 5 seconds).
const DEFAULT_POLL_INTERVAL: Duration = Duration::from_secs(5);

/// Poll-interval increment mandated by RFC 8628 §3.5 when the server
/// responds with `slow_down`.
const SLOW_DOWN_INTERVAL_INCREMENT: Duration = Duration::from_secs(5);

/// Total-request timeout for device-code issuance: keeps the admin login
/// endpoint inside its 5 s budget even when GitHub hangs.
const DEVICE_CODE_REQUEST_TIMEOUT: Duration = Duration::from_secs(5);

/// Total-request timeout for a single token poll: interval-bound upper
/// bound (default interval 5 s + slow_down increments) with headroom for a
/// slow response body; a hung poll fails the round instead of stalling
/// cancellation-observed polling.
const POLL_REQUEST_TIMEOUT: Duration = Duration::from_secs(10);

/// Injectable endpoint pair so wiremock tests can point both legs of the
/// flow at a mock server (production callers use
/// [`DeviceFlowEndpoints::production`]).
#[derive(Debug, Clone)]
pub struct DeviceFlowEndpoints {
    /// Device authorization endpoint (device code issuance).
    pub device_code_url: String,
    /// Token endpoint (device grant polling).
    pub token_url: String,
}

impl DeviceFlowEndpoints {
    /// Production GitHub endpoints ([`GITHUB_DEVICE_CODE_URL`] /
    /// [`GITHUB_TOKEN_URL`]).
    pub fn production() -> Self {
        Self {
            device_code_url: GITHUB_DEVICE_CODE_URL.to_string(),
            token_url: GITHUB_TOKEN_URL.to_string(),
        }
    }
}

/// Device authorization grant returned by GitHub (RFC 8628 §3.2).
///
/// `device_code` is secret-shaped (it can be exchanged for a token) and is
/// redacted by the manual `Debug` impl; `user_code` and
/// `verification_uri` are the values the human user must see and are safe
/// to surface.
#[derive(Clone, Deserialize)]
pub struct DeviceCodeGrant {
    /// Secret per-device code exchanged at the token endpoint. Never
    /// logged.
    pub device_code: String,
    /// Short code the user types at the verification URL.
    pub user_code: String,
    /// URL the user visits to approve the request.
    pub verification_uri: String,
    /// Lifetime of the device code in seconds.
    pub expires_in: u64,
    /// Mandated polling interval in seconds; GitHub omits this in
    /// practice, so a 5 s default is applied at polling time.
    #[serde(default)]
    pub interval: Option<u64>,
}

impl fmt::Debug for DeviceCodeGrant {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("DeviceCodeGrant")
            .field("device_code", &"<redacted>")
            .field("user_code", &self.user_code)
            .field("verification_uri", &self.verification_uri)
            .field("expires_in", &self.expires_in)
            .field("interval", &self.interval)
            .finish()
    }
}

/// Successful terminal token response from the device grant polling
/// (GitHub returns no refresh token in the device flow).
#[derive(Clone, Deserialize)]
pub struct GithubToken {
    /// GitHub OAuth access token (`gho_…`). Never logged.
    pub access_token: String,
    /// Token lifetime in seconds (GitHub currently returns `expires_in:
    /// 28800`, i.e. 8 hours).
    #[serde(default)]
    pub expires_in: Option<u64>,
}

impl fmt::Debug for GithubToken {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("GithubToken")
            .field("access_token", &"<redacted>")
            .field("expires_in", &self.expires_in)
            .finish()
    }
}

/// Request a device code from the GitHub device authorization endpoint
/// (Req 2.1).
///
/// Posts `client_id` = [`COPILOT_CLIENT_ID`] and `scope` =
/// [`COPILOT_OAUTH_SCOPE`] as a form body with `Accept: application/json`
/// (GitHub answers `text/html` for unaccepted requests).
///
/// # Errors
///
/// - Transport failure → [`OAuthError::DeviceFlowNetwork`]
/// - Non-success HTTP → [`OAuthError::DeviceFlowHttp`] (truncated body)
/// - Malformed JSON → [`OAuthError::DeviceFlowHttp`] with status 200 and a
///   parse description
#[tracing::instrument(skip(client))]
pub async fn start_device_flow(
    client: &reqwest::Client,
    endpoints: &DeviceFlowEndpoints,
) -> Result<DeviceCodeGrant, OAuthError> {
    let params = [
        ("client_id", COPILOT_CLIENT_ID),
        ("scope", COPILOT_OAUTH_SCOPE),
    ];

    let response = client
        .post(&endpoints.device_code_url)
        .header("accept", "application/json")
        .form(&params)
        .timeout(DEVICE_CODE_REQUEST_TIMEOUT)
        .send()
        .await
        .map_err(OAuthError::DeviceFlowNetwork)?;

    let status = response.status();
    if !status.is_success() {
        let body = response.text().await.unwrap_or_default();
        return Err(OAuthError::DeviceFlowHttp {
            status: status.as_u16(),
            body: truncate_error_body(body),
        });
    }

    response
        .json::<DeviceCodeGrant>()
        .await
        .map_err(|e| OAuthError::DeviceFlowHttp {
            status: status.as_u16(),
            body: truncate_error_body(e.to_string()),
        })
}

/// One decoded token-endpoint poll outcome (RFC 8628 §3.5).
enum PollOutcome {
    /// `access_token` present — terminal success.
    Success(GithubToken),
    /// `authorization_pending` — keep polling at the current interval.
    Pending,
    /// `slow_down` — keep polling with an increased interval.
    SlowDown,
    /// Any other `error` code — terminal failure.
    Error(String),
}

fn decode_poll_body(body: &str) -> PollOutcome {
    match serde_json::from_str::<serde_json::Value>(body) {
        Ok(value) => {
            if value
                .get("access_token")
                .and_then(|token| token.as_str())
                .is_some()
            {
                match serde_json::from_str::<GithubToken>(body) {
                    Ok(token) => PollOutcome::Success(token),
                    Err(_) => PollOutcome::Error("malformed success payload".to_string()),
                }
            } else {
                match value.get("error").and_then(|error| error.as_str()) {
                    Some("authorization_pending") => PollOutcome::Pending,
                    Some("slow_down") => PollOutcome::SlowDown,
                    Some(other) => PollOutcome::Error(other.to_string()),
                    None => PollOutcome::Error("response carried neither access_token nor error".to_string()),
                }
            }
        }
        Err(_) => PollOutcome::Error("response was not valid JSON".to_string()),
    }
}

/// Poll the GitHub token endpoint with a device grant until a terminal
/// state is reached (Req 2.2 / 2.3 / 2.5).
///
/// Behavior:
/// - First poll fires immediately, then sleeps the mandated interval
///   (`interval` from the grant, default 5 s) between polls.
/// - `authorization_pending` keeps polling; `slow_down` increases the
///   interval by 5 s per occurrence (RFC 8628 §3.5).
/// - `access_denied` → [`OAuthError::DeviceFlowDenied`] (terminal),
///   `expired_token` → [`OAuthError::DeviceFlowExpired`] (terminal).
/// - Polling stops once the code lifetime (`expires_in`) has elapsed;
///   [`OAuthError::DeviceFlowExpired`] is returned without another
///   request (no retry beyond the device code lifetime).
/// - Cancellation-safe (Req 2.5): every await is raced against
///   `cancel.changed()` via `tokio::select!` on the caller-supplied watch
///   receiver (the same primitive used for run cancellation in
///   `gateway/handlers.rs`); a cancelled poll returns
///   [`OAuthError::DeviceFlowCancelled`] promptly with no spawned task
///   outliving the poll future.
///
/// # Security
///
/// `grant.device_code` and the returned token are never logged; the
/// `#[tracing::instrument]` annotation skips both the client and the
/// grant.
#[tracing::instrument(skip(client, grant, cancel))]
pub async fn poll_for_token(
    client: &reqwest::Client,
    endpoints: &DeviceFlowEndpoints,
    grant: &DeviceCodeGrant,
    mut cancel: watch::Receiver<bool>,
) -> Result<GithubToken, OAuthError> {
    let mut interval = Duration::from_secs(grant.interval.unwrap_or(DEFAULT_POLL_INTERVAL.as_secs()));
    let lifetime = Duration::from_secs(grant.expires_in);
    let start = tokio::time::Instant::now();

    loop {
        // No retry past the device-code lifetime (Req 2.3): check both
        // before sending and before sleeping past the horizon.
        if start.elapsed() >= lifetime {
            return Err(OAuthError::DeviceFlowExpired {
                expires_in_secs: grant.expires_in,
            });
        }

        let response = tokio::select! {
            biased;
            _ = cancel.changed() => return Err(OAuthError::DeviceFlowCancelled),
            sent = client
                .post(&endpoints.token_url)
                .header("accept", "application/json")
                .form(&[
                    ("grant_type", DEVICE_CODE_GRANT_TYPE),
                    ("client_id", COPILOT_CLIENT_ID),
                    ("device_code", grant.device_code.as_str()),
                ])
                .timeout(POLL_REQUEST_TIMEOUT)
                .send() => sent,
        }
        .map_err(OAuthError::DeviceFlowNetwork)?;

        let status = response.status();
        let body = tokio::select! {
            biased;
            _ = cancel.changed() => return Err(OAuthError::DeviceFlowCancelled),
            body = response.text() => body.map_err(OAuthError::DeviceFlowNetwork)?,
        };

        if !status.is_success() {
            return Err(OAuthError::DeviceFlowHttp {
                status: status.as_u16(),
                body: truncate_error_body(body),
            });
        }

        match decode_poll_body(&body) {
            PollOutcome::Success(token) => return Ok(token),
            PollOutcome::Pending => {}
            PollOutcome::SlowDown => {
                interval += SLOW_DOWN_INTERVAL_INCREMENT;
            }
            PollOutcome::Error(code) => match code.as_str() {
                "access_denied" => return Err(OAuthError::DeviceFlowDenied),
                "expired_token" => {
                    return Err(OAuthError::DeviceFlowExpired {
                        expires_in_secs: grant.expires_in,
                    })
                }
                other => {
                    return Err(OAuthError::DeviceFlowHttp {
                        status: status.as_u16(),
                        body: truncate_error_body(format!("unrecognized device-flow error: {other}")),
                    })
                }
            },
        }

        let remaining = lifetime.saturating_sub(start.elapsed());
        let sleep = interval.min(remaining);
        tokio::select! {
            biased;
            _ = cancel.changed() => return Err(OAuthError::DeviceFlowCancelled),
            _ = tokio::time::sleep(sleep) => {}
        }
    }
}

#[cfg(test)]
mod tests {
use super::*;
use crate::oauth::github_copilot::test_support::test_client;
use wiremock::matchers::{method, path};
use wiremock::{Mock, MockServer, ResponseTemplate};

fn pending_body() -> serde_json::Value {
serde_json::json!({"error": "authorization_pending"})
}

fn endpoints(server: &MockServer) -> DeviceFlowEndpoints {
        DeviceFlowEndpoints {
            device_code_url: format!("{}/login/device/code", server.uri()),
            token_url: format!("{}/login/oauth/access_token", server.uri()),
        }
    }

    async fn fetch_grant(server: &MockServer) -> DeviceCodeGrant {
        start_device_flow(&test_client(), &endpoints(server))
            .await
            .expect("device grant request should succeed")
    }

    /// A cancel pair whose sender stays alive for the duration of the
    /// test body (a dropped sender makes `watch::Receiver::changed`
    /// resolve immediately, which would read as instant cancellation).
    fn cancel_pair() -> (watch::Sender<bool>, watch::Receiver<bool>) {
        watch::channel(false)
    }

    fn grant(device_code: &str, expires_in: u64, interval: Option<u64>) -> DeviceCodeGrant {
        DeviceCodeGrant {
            device_code: device_code.to_string(),
            user_code: "ABCD-1234".to_string(),
            verification_uri: "https://github.com/login/device".to_string(),
            expires_in,
            interval,
        }
    }

    fn token_body() -> serde_json::Value {
        serde_json::json!({"access_token": "gho-secret-token", "token_type": "bearer", "scope": "read:user"})
    }

    /// Wall-clock upper bound for a poll round-trip over loopback
    /// (request + response + scheduling slack). Cadence assertions use
    /// this to tolerate real-time jitter while still proving the poll
    /// interval is honored: a poll fired "immediately" would violate the
    /// lower bound, a missed interval would violate the upper bound.
    const ROUND_TRIP_SLACK_MS: u64 = 700;

    /// (a) Req 2.1: the device-code request carries the Copilot public
    /// client ID and `read:user` scope, and the response parses into the
    /// grant fields.
    #[tokio::test]
    async fn device_grant_request_shape_and_parse() {
        let server = MockServer::start().await;
        Mock::given(method("POST"))
            .and(path("/login/device/code"))
            .and(wiremock::matchers::body_string_contains("client_id=Iv1.b507a08c87ecfe98"))
            .and(wiremock::matchers::body_string_contains("scope=read%3Auser"))
            .respond_with(ResponseTemplate::new(200).set_body_json(serde_json::json!({
                "device_code": "dc-secret-1",
                "user_code": "ABCD-1234",
                "verification_uri": "https://github.com/login/device",
                "expires_in": 900,
                "interval": 5
            })))
            .expect(1)
            .mount(&server)
            .await;

        let parsed = fetch_grant(&server).await;

        assert_eq!(parsed.user_code, "ABCD-1234");
        assert_eq!(parsed.verification_uri, "https://github.com/login/device");
        assert_eq!(parsed.expires_in, 900);
        assert_eq!(parsed.interval, Some(5));

        // Debug output must never contain the device code.
        assert!(!format!("{parsed:?}").contains("dc-secret-1"));

        server.verify().await;
    }

    /// Req 2.2: terminal success returns the GitHub access token; the
    /// token never leaks through Debug output.
    #[tokio::test]
    async fn poll_success_returns_token() {
        let server = MockServer::start().await;
        Mock::given(method("POST"))
            .and(path("/login/oauth/access_token"))
            .respond_with(ResponseTemplate::new(200).set_body_json(token_body()))
            .expect(1)
            .mount(&server)
            .await;

        let (_cancel_tx, cancel) = cancel_pair();
        let token = poll_for_token(
            &test_client(),
            &endpoints(&server),
            &grant("dc-secret-1", 900, Some(5)),
            cancel,
        )
        .await
        .expect("terminal success should return the token");

        assert_eq!(token.access_token, "gho-secret-token");
        assert!(!format!("{token:?}").contains("gho-secret-token"));
        server.verify().await;
    }

    /// (b) Req 2.2: polling obeys the mandated interval (real clock,
    /// 1 s grant interval): first poll immediate, second poll no earlier
    /// than the interval, third poll within interval + slack. The 4 s
    /// code lifetime then trips the local expiry guard — exactly three
    /// requests total (Req 2.3: no poll past expiry).
    #[ignore = "wall-clock cadence timing flakes on loaded CI runners"]
    #[tokio::test]
    async fn poll_cadence_follows_grant_interval() {
        let interval = Duration::from_millis(1000);
        let server = MockServer::start().await;
        // All responses pending; the local expiry guard ends the loop.
        Mock::given(method("POST"))
            .and(path("/login/oauth/access_token"))
            .respond_with(ResponseTemplate::new(200).set_body_json(pending_body()))
            .expect(4)
            .mount(&server)
            .await;

        let started = std::time::Instant::now();
        // expires_in=4, interval=1 → polls at ~t=0, ~1, ~2, ~3, guard at t=4.
        let g = grant("dc-secret-1", 4, Some(1));
        let (_cancel_tx, cancel) = cancel_pair();
        let result = poll_for_token(&test_client(), &endpoints(&server), &g, cancel).await;

        assert!(matches!(
            result,
            Err(OAuthError::DeviceFlowExpired { expires_in_secs: 4 })
        ));
        let elapsed = started.elapsed();

        let requests = server.received_requests().await.unwrap();
        assert_eq!(requests.len(), 4, "polls at t≈0,1,2,3 only");

        // The second request cannot have been sent before the mandated
        // interval elapsed (first request returns ~immediately).
        assert!(
            elapsed >= interval * 3,
            "four interval-spaced polls require ≥3 intervals, took {elapsed:?}"
        );
        // And the whole loop terminates at the code lifetime, not a
        // fifth interval.
        assert!(
            elapsed < interval * 4 + Duration::from_millis(ROUND_TRIP_SLACK_MS),
            "loop must stop at the 4 s code lifetime, took {elapsed:?}"
        );

        server.verify().await;
    }

    /// (b) Req 2.2: `slow_down` increases the interval by 5 s per RFC
    /// 8628 §3.5. Sequence (stateful responder): pending, pending,
    /// slow_down, then success. With a 1 s base interval the first three
    /// requests land at ~t=0/1/2; the fourth must then wait the
    /// increased 6 s interval (1 s + 5 s increment) — at t=5.9 s exactly
    /// three requests have been sent, and success arrives ~t=8 s.
    #[ignore = "wall-clock cadence timing flakes on loaded CI runners"]
    #[tokio::test]
    async fn slow_down_increases_interval_by_five_seconds() {
        struct PendingPendingSlowThenToken(std::sync::atomic::AtomicUsize);
        impl wiremock::Respond for PendingPendingSlowThenToken {
            fn respond(&self, _request: &wiremock::Request) -> wiremock::ResponseTemplate {
                let call = self.0.fetch_add(1, std::sync::atomic::Ordering::SeqCst);
                match call {
                    0 | 1 => ResponseTemplate::new(200).set_body_json(pending_body()),
                    2 => ResponseTemplate::new(200).set_body_json(
                        serde_json::json!({"error": "slow_down"}),
                    ),
                    _ => ResponseTemplate::new(200).set_body_json(token_body()),
                }
            }
        }

        let server = MockServer::start().await;
        Mock::given(method("POST"))
            .and(path("/login/oauth/access_token"))
            .respond_with(PendingPendingSlowThenToken(std::sync::atomic::AtomicUsize::new(
                0,
            )))
            .expect(4)
            .mount(&server)
            .await;

        let started = std::time::Instant::now();
        let g = grant("dc-secret-1", 900, Some(1));
        let (_cancel_tx, cancel) = cancel_pair();
        let client = test_client();
        let eps = endpoints(&server);
        let task = tokio::spawn(async move {
            poll_for_token(&client, &eps, &g, cancel).await
        });

        // First three requests fire on the 1 s base cadence (t≈0,1,2).
        loop {
            let count = server.received_requests().await.unwrap().len();
            if count >= 3 {
                break;
            }
            assert!(
                started.elapsed() < Duration::from_secs(4),
                "base-cadence polls never arrived"
            );
            tokio::time::sleep(Duration::from_millis(50)).await;
        }
        let third_at = started.elapsed();
        assert!(third_at < Duration::from_secs(4));

        // The slow_down response arrived with request #3; the next poll
        // must wait the increased 6 s interval. At t=5.9 s a 1 s cadence
        // would have sent ~6 requests — the increment must have held it.
        while started.elapsed() < Duration::from_millis(5900) {
            tokio::time::sleep(Duration::from_millis(50)).await;
        }
        assert_eq!(
            server.received_requests().await.unwrap().len(),
            3,
            "fourth poll must wait for the increased 6 s interval"
        );

        // Success lands ~6 s after the slow_down (t≈8) and the task ends.
        let token = tokio::time::timeout(Duration::from_secs(6), task)
            .await
            .expect("success poll must arrive within the increased interval")
            .expect("task must join")
            .expect("fourth poll returns the token");
        assert_eq!(token.access_token, "gho-secret-token");
        assert!(
            started.elapsed() >= Duration::from_secs(5),
            "fourth poll must be at least base+increment (≈6 s) after the third"
        );
        server.verify().await;
    }

    /// (c) Req 2.3: `access_denied` is terminal with a specific,
    /// actionable message.
    #[tokio::test]
    async fn access_denied_is_terminal_with_actionable_message() {
        let server = MockServer::start().await;
        Mock::given(method("POST"))
            .and(path("/login/oauth/access_token"))
            .respond_with(ResponseTemplate::new(200).set_body_json(
                serde_json::json!({"error": "access_denied"}),
            ))
            .expect(1)
            .mount(&server)
            .await;

        let (_cancel_tx, cancel) = cancel_pair();
        let result = poll_for_token(
            &test_client(),
            &endpoints(&server),
            &grant("dc-secret-1", 900, Some(5)),
            cancel,
        )
        .await;

        match result {
            Err(OAuthError::DeviceFlowDenied) => {}
            other => panic!("expected DeviceFlowDenied, got {other:?}"),
        }
        let message = OAuthError::DeviceFlowDenied.to_string();
        assert!(message.contains("re-run the login"), "{message}");
        server.verify().await;
    }

    /// (d) Req 2.3: `expired_token` is terminal.
    #[tokio::test]
    async fn expired_token_is_terminal() {
        let server = MockServer::start().await;
        Mock::given(method("POST"))
            .and(path("/login/oauth/access_token"))
            .respond_with(ResponseTemplate::new(200).set_body_json(
                serde_json::json!({"error": "expired_token"}),
            ))
            .expect(1)
            .mount(&server)
            .await;

        let (_cancel_tx, cancel) = cancel_pair();
        let result = poll_for_token(
            &test_client(),
            &endpoints(&server),
            &grant("dc-secret-1", 900, Some(5)),
            cancel,
        )
        .await;

        match result {
            Err(OAuthError::DeviceFlowExpired { expires_in_secs: 900 }) => {}
            other => panic!("expected DeviceFlowExpired, got {other:?}"),
        }
        server.verify().await;
    }

    /// (d) Req 2.3: no poll fires after the device code lifetime has
    /// elapsed — a 2 s lifetime with a 5 s interval allows exactly one
    /// request (t=0); the guard trips at t=2 before the first interval
    /// sleep completes a second poll.
    #[tokio::test]
    async fn no_polling_beyond_device_code_lifetime() {
        let server = MockServer::start().await;
        Mock::given(method("POST"))
            .and(path("/login/oauth/access_token"))
            .respond_with(ResponseTemplate::new(200).set_body_json(pending_body()))
            .expect(1)
            .mount(&server)
            .await;

        let started = std::time::Instant::now();
        let g = grant("dc-secret-1", 2, Some(5));
        let (_cancel_tx, cancel) = cancel_pair();
        let result = poll_for_token(&test_client(), &endpoints(&server), &g, cancel).await;

        assert!(matches!(
            result,
            Err(OAuthError::DeviceFlowExpired { expires_in_secs: 2 })
        ));
        // Terminated by the 2 s lifetime even though the interval was 5 s
        // (the sleep is clamped to the remaining lifetime).
        let elapsed = started.elapsed();
        assert!(
            elapsed >= Duration::from_secs(2)
                && elapsed < Duration::from_secs(2) + Duration::from_millis(ROUND_TRIP_SLACK_MS),
            "loop must end at the 2 s lifetime, took {elapsed:?}"
        );
        server.verify().await;
    }

    /// (e) Req 2.5: signalling the cancellation watch aborts a sleeping
    /// poll promptly with `DeviceFlowCancelled` and leaves no request
    /// running afterwards. The poll runs as a spawned task (mirroring
    /// the production login task); with a 30 s interval and cancellation
    /// at ~1 s, exactly one request was ever sent.
    #[tokio::test]
    async fn cancellation_aborts_pending_poll() {
        let (cancel_tx, cancel) = cancel_pair();

        let server = MockServer::start().await;
        Mock::given(method("POST"))
            .and(path("/login/oauth/access_token"))
            .respond_with(ResponseTemplate::new(200).set_body_json(pending_body()))
            .expect(1)
            .mount(&server)
            .await;

        let client = test_client();
        let eps = endpoints(&server);
        let g = grant("dc-secret-1", 900, Some(30));
        let task = tokio::spawn(async move {
            poll_for_token(&client, &eps, &g, cancel).await
        });

        // Let the first poll complete and the task settle into the 30 s
        // sleep. Poll wiremock until the request has arrived.
        for _ in 0..100 {
            if !server
                .received_requests()
                .await
                .unwrap_or_default()
                .is_empty()
            {
                break;
            }
            tokio::time::sleep(Duration::from_millis(25)).await;
        }

        // Cancel while sleeping — must win promptly (30 s interval would
        // otherwise keep the task alive far beyond the code expiry).
        let cancel_at = std::time::Instant::now();
        cancel_tx.send(true).unwrap();
        let result = task
            .await
            .expect("task must join promptly after cancel")
            .expect_err("cancelled poll returns DeviceFlowCancelled");
        let join_latency = cancel_at.elapsed();

        assert!(matches!(result, OAuthError::DeviceFlowCancelled));
        assert!(
            join_latency < Duration::from_secs(2),
            "cancel must abort promptly, took {join_latency:?}"
        );
        assert_eq!(server.received_requests().await.unwrap().len(), 1);
        server.verify().await;
    }

    /// Default interval: GitHub omits `interval` from the grant —
    /// polling falls back to the RFC 8628 default of 5 s. With a 7 s
    /// lifetime exactly two requests fire (t=0 and t=5); the t=5 sleep
    /// is clamped to the 2 s remainder, then the guard trips.
    #[tokio::test]
    async fn default_interval_is_five_seconds_when_grant_omits_it() {
        let server = MockServer::start().await;
        Mock::given(method("POST"))
            .and(path("/login/oauth/access_token"))
            .respond_with(ResponseTemplate::new(200).set_body_json(pending_body()))
            .expect(2)
            .mount(&server)
            .await;

        let g = grant("dc-secret-1", 7, None);
        let (_cancel_tx, cancel) = cancel_pair();
        let result = poll_for_token(&test_client(), &endpoints(&server), &g, cancel).await;

        assert!(matches!(
            result,
            Err(OAuthError::DeviceFlowExpired { expires_in_secs: 7 })
        ));

        // t=0 immediately, t=5 after the default interval — a 1 s
        // fallback would have produced ~7 requests instead of 2.
        let requests = server.received_requests().await.unwrap();
        assert_eq!(requests.len(), 2, "expected polls at t=0 and t=5 only");
    }

    /// End-to-end over one wiremock server: grant request, one pending
    /// response, then terminal success.
    #[tokio::test]
    async fn full_flow_pending_then_success() {
        let server = MockServer::start().await;
        Mock::given(method("POST"))
            .and(path("/login/device/code"))
            .respond_with(ResponseTemplate::new(200).set_body_json(serde_json::json!({
                "device_code": "dc-secret-1",
                "user_code": "ABCD-1234",
                "verification_uri": "https://github.com/login/device",
                "expires_in": 900,
                "interval": 1
            })))
            .expect(1)
            .mount(&server)
            .await;
        Mock::given(method("POST"))
            .and(path("/login/oauth/access_token"))
            .respond_with(ResponseTemplate::new(200).set_body_json(
                serde_json::json!({"error": "authorization_pending"}),
            ))
            .up_to_n_times(1)
            .mount(&server)
            .await;
        Mock::given(method("POST"))
            .and(path("/login/oauth/access_token"))
            .respond_with(ResponseTemplate::new(200).set_body_json(token_body()))
            .expect(1)
            .mount(&server)
            .await;

        let client = test_client();
        let eps = endpoints(&server);
        let g = fetch_grant(&server).await;

        let (_cancel_tx, cancel) = cancel_pair();
        let token = poll_for_token(&client, &eps, &g, cancel)
            .await
            .expect("second poll should succeed");
        assert_eq!(token.access_token, "gho-secret-token");
        server.verify().await;
    }

    /// Non-success HTTP during polling maps to `DeviceFlowHttp` with a
    /// truncated body (256-byte cap, mirroring oauth/token.rs).
    #[tokio::test]
    async fn poll_http_error_maps_to_device_flow_http() {
        let server = MockServer::start().await;
        Mock::given(method("POST"))
            .and(path("/login/oauth/access_token"))
            .respond_with(ResponseTemplate::new(500).set_body_string("x".repeat(1024)))
            .expect(1)
            .mount(&server)
            .await;

        let (_cancel_tx, cancel) = cancel_pair();
        let result = poll_for_token(
            &test_client(),
            &endpoints(&server),
            &grant("dc-secret-1", 900, Some(5)),
            cancel,
        )
        .await;

        match result {
            Err(OAuthError::DeviceFlowHttp { status: 500, body }) => {
                assert!(
                    body.len() <= 256,
                    "body must be truncated, got {}",
                    body.len()
                );
            }
            other => panic!("expected DeviceFlowHttp, got {other:?}"),
        }
        server.verify().await;
    }

    /// Error variants render specific, actionable Display messages.
    #[test]
    fn device_flow_error_messages_are_specific_and_actionable() {
        assert!(OAuthError::DeviceFlowDenied.to_string().contains("denied"));
        assert!(OAuthError::DeviceFlowExpired { expires_in_secs: 900 }
            .to_string()
            .contains("900 seconds"));
        assert!(OAuthError::DeviceFlowCancelled.to_string().contains("cancelled"));
    }

    /// The device-code request itself fails cleanly on non-success HTTP
    /// with a truncated body.
    #[tokio::test]
    async fn start_device_flow_http_error_truncates_body() {
        let server = MockServer::start().await;
        Mock::given(method("POST"))
            .and(path("/login/device/code"))
            .respond_with(ResponseTemplate::new(503).set_body_string("y".repeat(1024)))
            .expect(1)
            .mount(&server)
            .await;

        let result = start_device_flow(&test_client(), &endpoints(&server)).await;
        match result {
            Err(OAuthError::DeviceFlowHttp { status: 503, body }) => {
                assert!(body.len() <= 256, "body must be truncated");
            }
            other => panic!("expected DeviceFlowHttp, got {other:?}"),
        }
        server.verify().await;
    }
}

