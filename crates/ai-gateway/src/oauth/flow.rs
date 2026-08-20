#![allow(dead_code)]
//! OAuth flow orchestration and session state machine.
//!
//! This module owns the end-to-end lifecycle of the OpenAI OAuth 2.0
//! Authorization Code + PKCE flow. The responsibilities listed below are
//! introduced across subtasks 5.2–5.9 of the `openai-oauth-login` spec; this
//! file provides the scaffolding (state machine + [`OAuthFlow`] container) that
//! subsequent tasks build on.
//!
//! Orchestration responsibilities reserved for later subtasks:
//! - **5.2**: `OAuthSessionState` enum consumed by status reporting and the
//!   background refresh task.
//! - **5.3**: `initiate()` — generate PKCE material + state, build the
//!   authorization URL, launch the browser, start the callback server.
//! - **5.4**: authorization URL construction with all required query
//!   parameters (client_id, redirect_uri, scope, code_challenge,
//!   code_challenge_method, response_type, state).
//! - **5.5**: browser-launch fallback path — return the auth URL to the caller
//!   when `open::that()` fails.
//! - **5.6**: `OAuthManager` wrapping flow + token store + background refresh.
//! - **5.7**: background token-refresh task (60s cadence, refresh 5 min before
//!   expiry).
//! - **5.8**: exponential backoff on refresh failures (2s / 4s / 8s).
//! - **5.9**: property test for the backoff schedule.
//!
//! The state machine (design §2):
//!
//! ```text
//! Idle → Initiated → AwaitingCallback → ExchangingToken → Authenticated
//!                                     → TimedOut
//!                                     → Failed
//! ```

use std::time::Duration;

use serde::Serialize;
use tokio::task::JoinHandle;

use super::callback::AuthorizationCode;
use super::error::OAuthError;
use super::pkce;
use super::token::OPENAI_CLIENT_ID;

/// OpenAI OAuth 2.0 authorization endpoint (design §2, task 5.4).
pub(crate) const AUTHORIZATION_ENDPOINT: &str = "https://auth.openai.com/oauth/authorize";

/// Public session state reported by the `/admin/oauth/openai/status` endpoint
/// (design §7, Req 7.1).
///
/// Intentionally exposes only **safe metadata** — never raw token values.
/// Serialized with `#[serde(tag = "state", rename_all = "lowercase")]` so the
/// JSON shape is e.g. `{"state":"authenticated","expires_at":123,"scopes":"openid"}`.
#[derive(Debug, Clone, PartialEq, Eq, Serialize)]
#[serde(rename_all = "lowercase", tag = "state")]
pub enum OAuthSessionState {
    /// No tokens on disk; user has not logged in.
    Unauthenticated,
    /// Valid tokens present. `expires_at` is a Unix timestamp (seconds).
    Authenticated { expires_at: u64, scopes: String },
    /// Tokens exist but refresh failed past the backoff budget.
    Expired,
    /// A background refresh is in progress.
    Refreshing,
}

/// Lifecycle state of a single OAuth login attempt.
///
/// Mirrors the state machine documented in design §2. Terminal states are
/// [`FlowState::Authenticated`], [`FlowState::TimedOut`], and
/// [`FlowState::Failed`].
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum FlowState {
    /// No flow in progress.
    Idle,
    /// PKCE material and state generated; authorization URL built.
    Initiated,
    /// Browser launched (or URL surfaced for manual nav); callback server
    /// listening on the loopback interface.
    AwaitingCallback,
    /// Authorization code received; token exchange request in flight.
    ExchangingToken,
    /// Tokens received and persisted — terminal success state.
    Authenticated,
    /// Callback server timed out waiting for redirect — terminal failure.
    TimedOut,
    /// Any other failure (browser error, token-exchange error, CSRF
    /// mismatch, etc.) — terminal failure.
    Failed,
}

/// Outcome of a successful call to [`OAuthFlow::initiate`] (task 5.5).
///
/// Distinguishes the happy path (the default browser was launched) from the
/// manual-fallback path (browser launch failed and the caller should surface
/// the URL so the user can paste it into a browser themselves). Both variants
/// carry the same authorization URL; only the semantic distinction differs.
///
/// The authorization URL is **not secret** — it contains only public
/// parameters (client_id, redirect_uri, scope, PKCE challenge, state) — so it
/// is safe to log and safe to return via the admin API.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum InitiationOutcome {
    /// The platform's default browser was launched; the user is now stepping
    /// through the provider UI.
    BrowserOpened { auth_url: String },
    /// Browser launch failed; the caller must surface `auth_url` to the user
    /// so they can navigate manually.
    ManualNavigationRequired { auth_url: String },
}

/// Build the OpenAI authorization URL with every required query parameter
/// (task 5.4, design §2).
///
/// Parameters:
/// - `client_id`   — OAuth app identifier (typically [`OPENAI_CLIENT_ID`])
/// - `redirect_uri` — loopback callback, e.g. `http://localhost:8765/auth/callback`
/// - `scope`       — space-separated scope list, e.g.
///   `"openid profile email offline_access"`
/// - `code_challenge` — PKCE S256 challenge from [`pkce::generate_pkce`]
/// - `state`       — CSRF state from [`pkce::generate_state`]
///
/// `code_challenge_method` is hard-coded to `S256` and `response_type` to
/// `code`, per the PKCE Authorization Code flow.
///
/// URL encoding is handled by `reqwest::Url::parse_with_params`; parsing
/// cannot fail for a static URL plus a fixed parameter set, so this function
/// returns `String` directly.
pub(crate) fn build_authorization_url(
    client_id: &str,
    redirect_uri: &str,
    scope: &str,
    code_challenge: &str,
    state: &str,
) -> String {
    reqwest::Url::parse_with_params(
        AUTHORIZATION_ENDPOINT,
        &[
            ("client_id", client_id),
            ("redirect_uri", redirect_uri),
            ("scope", scope),
            ("code_challenge", code_challenge),
            ("code_challenge_method", "S256"),
            ("response_type", "code"),
            ("state", state),
        ],
    )
    .expect("AUTHORIZATION_ENDPOINT is a valid static URL")
    .to_string()
}

#[cfg(test)]
mod tests {
    use super::*;
    use proptest::prelude::*;

    // Feature: openai-oauth-login, Property 7 (variant): No Token Values in
    // Log-Formatted OAuthSessionState
    //
    // **Validates: Requirements 8.4**
    //
    // `OAuthSessionState` intentionally omits raw token material — it carries
    // only safe metadata (expiry timestamp, scope list). This property test
    // confirms that for *all* arbitrary token strings and *all* variants of
    // `OAuthSessionState`, neither the access-token nor the refresh-token
    // substring appears in the `Debug` output (`{:?}`). Since `Display` is
    // not implemented for `OAuthSessionState`, only `Debug` is tested.
    //
    // Token strings use a prefix ("tok_"/"ref_") and alphanumeric body of
    // 8–64 chars. This ensures they are long enough to avoid false positives
    // from coincidental substring matches in enum variant names, and distinct
    // from the `scopes` field which uses a different character pattern
    // (lowercase + spaces, mimicking real OAuth scopes like "openid profile").
    proptest! {
        #![proptest_config(ProptestConfig {
            cases: 64,
            .. ProptestConfig::default()
        })]

        #[test]
        fn no_token_values_in_debug_formatted_session_state(
            access_token in "tok_[A-Za-z0-9]{8,64}",
            refresh_token in "ref_[A-Za-z0-9]{8,64}",
            expires_at in any::<u64>(),
            scopes in "[a-z ]{0,64}",
        ) {
            // Build every variant of OAuthSessionState.
            let variants: Vec<OAuthSessionState> = vec![
                OAuthSessionState::Unauthenticated,
                OAuthSessionState::Authenticated { expires_at, scopes: scopes.clone() },
                OAuthSessionState::Expired,
                OAuthSessionState::Refreshing,
            ];

            for state in &variants {
                let debug_output = format!("{state:?}");

                // Neither token value must appear in the Debug representation.
                prop_assert!(
                    !debug_output.contains(&access_token),
                    "access_token {:?} leaked into Debug output: {}",
                    access_token,
                    debug_output
                );
                prop_assert!(
                    !debug_output.contains(&refresh_token),
                    "refresh_token {:?} leaked into Debug output: {}",
                    refresh_token,
                    debug_output
                );
            }
        }
    }
}

/// Orchestrator for a single OAuth login attempt.
///
/// Holds the current [`FlowState`], the shared `reqwest::Client` used for the
/// eventual token-exchange request, and the PKCE / CSRF material generated by
/// [`OAuthFlow::initiate`] (task 5.3) which is consumed during the
/// authorization-code exchange step.
pub struct OAuthFlow {
    state: FlowState,
    #[allow(dead_code)] // used by subtasks 5.3+
    http_client: reqwest::Client,
    /// PKCE `code_verifier` generated by [`OAuthFlow::initiate`] and consumed
    /// during the token-exchange step (task 5.3+). Stored as
    /// `Option<String>` so a completed or failed flow leaves no secret
    /// material behind in memory once consumed.
    pkce_verifier: Option<String>,
    /// CSRF `state` parameter generated at the start of the flow; the
    /// callback server verifies the provider-returned value against this.
    expected_state: Option<String>,
    /// Loopback redirect URI registered with the authorization server; the
    /// token-exchange request must present the same value.
    redirect_uri: Option<String>,
}

impl OAuthFlow {
    /// Creates a new flow in the [`FlowState::Idle`] state.
    ///
    /// The caller supplies the `reqwest::Client` so that the gateway's shared
    /// connection pool (with its configured timeouts and TLS settings) is
    /// reused for the token-exchange request.
    pub fn new(http_client: reqwest::Client) -> Self {
        Self {
            state: FlowState::Idle,
            http_client,
            pkce_verifier: None,
            expected_state: None,
            redirect_uri: None,
        }
    }

    /// Returns the current state of the flow.
    pub fn state(&self) -> &FlowState {
        &self.state
    }

    /// PKCE verifier captured by the most recent [`Self::initiate`] call, if any.
    ///
    /// Exposed `pub(crate)` so the token-exchange step can consume it; never
    /// exposed outside the crate because the verifier is secret.
    #[allow(dead_code)] // consumed by subsequent token-exchange subtask
    pub(crate) fn pkce_verifier(&self) -> Option<&str> {
        self.pkce_verifier.as_deref()
    }

    /// Redirect URI captured by the most recent [`Self::initiate`] call, if any.
    #[allow(dead_code)] // consumed by subsequent token-exchange subtask
    pub(crate) fn redirect_uri(&self) -> Option<&str> {
        self.redirect_uri.as_deref()
    }

    /// CSRF `state` value captured by the most recent [`Self::initiate`] call,
    /// if any. Used by the manual authorization-code entry path to validate
    /// the `state` pasted back by the user (headless / remote Docker case).
    pub(crate) fn expected_state(&self) -> Option<&str> {
        self.expected_state.as_deref()
    }

    /// Kick off an OAuth login attempt (task 5.3).
    ///
    /// Steps, per design §2:
    /// 1. Generate a PKCE verifier/challenge pair and a CSRF state value.
    /// 2. Persist the verifier, expected state, and redirect URI on `self` so
    ///    the token-exchange step (later subtask) can consume them.
    /// 3. Build the authorization URL via [`build_authorization_url`].
    /// 4. Attempt to launch the user's default browser with `open::that`.
    ///    Failure is **not** an error — per Req 1.7 / task 5.5, the caller
    ///    receives the URL back as
    ///    [`InitiationOutcome::ManualNavigationRequired`] so it can be
    ///    surfaced to the user.
    /// 5. Bind the loopback callback server on `callback_port` and spawn it
    ///    onto the tokio runtime; the returned [`JoinHandle`] resolves once
    ///    the server completes (either with the [`AuthorizationCode`] or with
    ///    an [`OAuthError`]). The server self-terminates on timeout or on
    ///    first callback receipt, so no external shutdown signal is needed.
    /// 6. Transition the flow through `Initiated` → `AwaitingCallback`.
    ///
    /// The authorization URL is logged at `INFO` level for operator
    /// visibility; it contains only public parameters (no secrets).
    pub async fn initiate(
        &mut self,
        _redirect_uri: &str,
        scope: &str,
        callback_port: u16,
        callback_timeout: Duration,
    ) -> Result<
        (
            InitiationOutcome,
            JoinHandle<Result<AuthorizationCode, OAuthError>>,
        ),
        OAuthError,
    > {
        // 1. Fresh PKCE + CSRF material for every attempt.
        let pkce_pair = pkce::generate_pkce();
        let state = pkce::generate_state();

        // 2. Bind the callback server FIRST so we know the actual port.
        //    Port 0 means the OS assigns an ephemeral port.
        //    In container environments (e.g. Docker), the callback must be
        //    reachable from the host browser, so we bind to 0.0.0.0 when
        //    OAUTH_CALLBACK_BIND_HOST is set (Dockerfile sets this).
        let bind_host: std::net::Ipv4Addr = std::env::var("OAUTH_CALLBACK_BIND_HOST")
            .ok()
            .and_then(|v| v.parse().ok())
            .unwrap_or(std::net::Ipv4Addr::LOCALHOST);
        let addr: std::net::SocketAddr = (bind_host, callback_port).into();
        let listener = tokio::net::TcpListener::bind(addr)
            .await
            .map_err(|source| OAuthError::CallbackBindFailed {
                port: callback_port,
                source,
            })?;
        let actual_port = listener
            .local_addr()
            .map(|a| a.port())
            .unwrap_or(callback_port);

        // 3. Build the redirect_uri with the REAL port.
        let redirect_uri = format!("http://localhost:{}/auth/callback", actual_port);

        // 4. Stash secrets + redirect URI for the later token-exchange step.
        self.pkce_verifier = Some(pkce_pair.code_verifier);
        self.expected_state = Some(state.clone());
        self.redirect_uri = Some(redirect_uri.clone());
        self.state = FlowState::Initiated;

        // 5. Build the authorization URL (task 5.4).
        let auth_url = build_authorization_url(
            OPENAI_CLIENT_ID,
            &redirect_uri,
            scope,
            &pkce_pair.code_challenge,
            &state,
        );

        tracing::info!(auth_url = %auth_url, "initiating OpenAI OAuth login");

        // 6. Attempt to launch the browser; fall back to manual navigation on
        //    failure (task 5.5). Either way we proceed to start the callback
        //    server — the user can still complete the flow manually.
        let outcome = match open::that(&auth_url) {
            Ok(()) => InitiationOutcome::BrowserOpened {
                auth_url: auth_url.clone(),
            },
            Err(err) => {
                tracing::warn!(
                    error = %err,
                    "failed to open default browser; returning URL for manual navigation"
                );
                InitiationOutcome::ManualNavigationRequired {
                    auth_url: auth_url.clone(),
                }
            }
        };

        // 7. Spawn the callback server using the pre-bound listener.
        let server_handle: JoinHandle<Result<AuthorizationCode, OAuthError>> = tokio::spawn(
            super::callback::start_callback_server_with_listener(listener, state, callback_timeout),
        );

        // 8. Advance the state machine.
        self.state = FlowState::AwaitingCallback;

        Ok((outcome, server_handle))
    }
}
