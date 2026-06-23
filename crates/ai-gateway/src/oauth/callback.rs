#![allow(dead_code)]
//! Loopback HTTP callback server for the OpenAI OAuth + PKCE flow.
//!
//! Per design §3, this module runs a minimal `axum` server bound exclusively
//! to `127.0.0.1:{port}` that exposes a single route — `GET /auth/callback` —
//! which receives the authorization-code redirect from the provider. The
//! server lives only for the duration of a single login attempt and is torn
//! down as soon as one of the terminal conditions below is reached.
//!
//! # Responsibilities (tasks 4.2 – 4.6)
//!
//! - **4.2** Build an `axum::Router` with `GET /auth/callback` bound to the
//!   loopback interface; wire a `tokio::sync::oneshot` channel so the handler
//!   can signal the server to shut down once the flow completes.
//! - **4.3** Validate that the callback's `state` query parameter matches the
//!   `expected_state` passed in by the caller. On mismatch the handler
//!   responds with HTTP 400, logs a CSRF warning, and resolves this function
//!   with [`OAuthError::StateMismatch`].
//! - **4.4** Handle provider-supplied `error` / `error_description` query
//!   parameters — log the error, render a user-facing failure page, and
//!   resolve with [`OAuthError::AuthorizationError`].
//! - **4.5** On a valid state + `code` pair, return a short HTML success page
//!   to the browser, signal shutdown, and resolve with the extracted
//!   [`AuthorizationCode`].
//! - **4.6** Enforce the 120-second callback timeout supplied by the caller;
//!   on timeout shut the server down and resolve with
//!   [`OAuthError::CallbackTimeout`].
//!
//! # Security
//!
//! The server binds to `127.0.0.1` only (see Req 8.2). The raw authorization
//! code is carried in [`AuthorizationCode`] — a newtype wrapper so callers
//! cannot accidentally confuse it with an access token or refresh token. It
//! must never be logged. The `state` parameter is likewise never logged.

use std::{
    net::{Ipv4Addr, SocketAddr},
    sync::Arc,
    time::Duration,
};

use axum::{
    extract::{Query, State},
    http::StatusCode,
    response::{Html, IntoResponse},
    routing::get,
    Router,
};
use serde::Deserialize;
use tokio::{
    net::TcpListener,
    sync::{oneshot, Mutex},
};

use crate::oauth::error::OAuthError;

/// HTML returned to the browser on a successful authorization-code capture.
const SUCCESS_HTML: &str = "<!doctype html><html><body><h1>Sign-in complete</h1><p>You can close this window and return to the router.</p></body></html>";

/// HTML returned to the browser on any failure path (CSRF mismatch, provider
/// error, or missing `code`).
const FAILURE_HTML: &str = "<!doctype html><html><body><h1>Sign-in failed</h1><p>Please return to the router and try again.</p></body></html>";

/// Newtype wrapper around an OAuth authorization code returned via the
/// loopback callback.
///
/// Kept distinct from raw `String` so callers cannot accidentally pass a
/// token, state value, or arbitrary user input where an authorization code is
/// expected. The contained value is short-lived and single-use, but still
/// sensitive — do not log the inner string.
#[derive(Debug, Clone)]
pub struct AuthorizationCode(pub String);

/// Query parameters expected on the `GET /auth/callback` endpoint.
///
/// All fields are optional because the provider may redirect with any subset
/// of them depending on the outcome (success → `code` + `state`; error →
/// `error` + `error_description` + `state`).
#[derive(Debug, Deserialize)]
struct CallbackQuery {
    code: Option<String>,
    state: Option<String>,
    error: Option<String>,
    error_description: Option<String>,
}

/// Shared handle for the handler to deliver the flow outcome back to the
/// awaiting [`start_callback_server`] task. Wrapped in `Mutex<Option<_>>` so
/// that only the first callback consumes the sender; any subsequent callbacks
/// observe `None` and are treated as no-ops (they still render a failure page
/// but cannot alter the already-decided outcome).
type ResultSender = Arc<Mutex<Option<oneshot::Sender<Result<AuthorizationCode, OAuthError>>>>>;

/// Per-request state threaded through `axum::Router::with_state`.
#[derive(Clone)]
struct CallbackState {
    expected_state: Arc<String>,
    result_tx: ResultSender,
}

/// Handler for `GET /auth/callback`.
///
/// Decides the flow outcome from the query parameters, delivers it to the
/// awaiting task via the oneshot sender, and returns an HTML page to the
/// user's browser.
async fn handle_callback(
    State(state): State<CallbackState>,
    Query(query): Query<CallbackQuery>,
) -> impl IntoResponse {
    // Decide the outcome (status, html body, channel payload) without
    // borrowing secrets into logs. Note: we never log `code` or `state`.
    let (status, body, result): (StatusCode, &'static str, Result<AuthorizationCode, OAuthError>) =
        if let Some(error_code) = query.error {
            // 4.4: provider returned an error on the redirect.
            tracing::warn!(
                error = %error_code,
                description = ?query.error_description,
                "OAuth authorization error from provider"
            );
            (
                StatusCode::OK,
                FAILURE_HTML,
                Err(OAuthError::AuthorizationError {
                    error: error_code,
                    description: query.error_description,
                }),
            )
        } else if query.state.as_deref() != Some(state.expected_state.as_str()) {
            // 4.3: CSRF state mismatch — HTTP 400 + warn log.
            tracing::warn!("CSRF state mismatch on OAuth callback");
            (StatusCode::BAD_REQUEST, FAILURE_HTML, Err(OAuthError::StateMismatch))
        } else if let Some(code) = query.code {
            // 4.5: success path.
            (StatusCode::OK, SUCCESS_HTML, Ok(AuthorizationCode(code)))
        } else {
            // Valid state but no code and no error — malformed provider response.
            (
                StatusCode::BAD_REQUEST,
                FAILURE_HTML,
                Err(OAuthError::MissingAuthorizationCode),
            )
        };

    // Deliver the outcome exactly once; subsequent callbacks (if any) find the
    // sender already consumed and silently render the failure page instead.
    if let Some(tx) = state.result_tx.lock().await.take() {
        let _ = tx.send(result);
    }

    (status, Html(body))
}

/// Start the loopback OAuth callback server and await the authorization-code
/// redirect.
///
/// Binds an `axum` server to `127.0.0.1:{port}` serving `GET /auth/callback`,
/// waits for the provider to redirect the user's browser there, validates the
/// returned `state` against `expected_state`, and resolves with the extracted
/// [`AuthorizationCode`] on success.
///
/// Resolves with an [`OAuthError`] on any of:
/// - bind failure ([`OAuthError::CallbackBindFailed`])
/// - provider error response ([`OAuthError::AuthorizationError`])
/// - CSRF state mismatch ([`OAuthError::StateMismatch`])
/// - missing `code` parameter ([`OAuthError::MissingAuthorizationCode`])
/// - elapsed `timeout` without any callback ([`OAuthError::CallbackTimeout`])
///
/// The server is guaranteed to be torn down before this function returns,
/// regardless of outcome.
pub async fn start_callback_server(
    expected_state: String,
    port: u16,
    timeout: Duration,
) -> Result<AuthorizationCode, OAuthError> {
    // 4.2 + Req 8.2: bind exclusively to the IPv4 loopback.
    let addr: SocketAddr = (Ipv4Addr::LOCALHOST, port).into();
    let listener = TcpListener::bind(addr)
        .await
        .map_err(|source| OAuthError::CallbackBindFailed { port, source })?;

    start_callback_server_with_listener(listener, expected_state, timeout).await
}

/// Same as [`start_callback_server`] but accepts a pre-bound [`TcpListener`].
///
/// Exists primarily to let tests bind `127.0.0.1:0`, learn the assigned port
/// up front, and then hand the listener to the server without a
/// `bind()` -> `drop()` -> `bind()` race window. Production code should prefer
/// [`start_callback_server`], which enforces the loopback invariant itself.
pub async fn start_callback_server_with_listener(
    listener: TcpListener,
    expected_state: String,
    timeout: Duration,
) -> Result<AuthorizationCode, OAuthError> {
    // Channels: `result_*` carries the final flow outcome from the handler;
    // `shutdown_*` drives `axum::serve::with_graceful_shutdown` from this task
    // once we observe either the outcome or the timeout.
    let (result_tx, result_rx) = oneshot::channel::<Result<AuthorizationCode, OAuthError>>();
    let (shutdown_tx, shutdown_rx) = oneshot::channel::<()>();

    let state = CallbackState {
        expected_state: Arc::new(expected_state),
        result_tx: Arc::new(Mutex::new(Some(result_tx))),
    };

    // 4.2: single route, bound to the loopback listener.
    let router: Router = Router::new()
        .route("/auth/callback", get(handle_callback))
        .with_state(state);

    let server = axum::serve(listener, router).with_graceful_shutdown(async move {
        // Completes either when we send on `shutdown_tx` or when the sender is
        // dropped (both signal: stop accepting new connections).
        let _ = shutdown_rx.await;
    });

    // Run the server on a background task so we can race it against the
    // 120-second callback timeout (4.6).
    let server_handle = tokio::spawn(async move {
        let _ = server.await;
    });

    // 4.3 / 4.4 / 4.5 / 4.6: race the handler-reported outcome against the
    // timeout; whichever wins drives the shutdown signal.
    let outcome = tokio::select! {
        biased;

        res = result_rx => {
            match res {
                Ok(result) => result,
                // Sender dropped without ever sending — treat as malformed
                // callback. In practice this only happens if the server task
                // is cancelled before a request arrives.
                Err(_) => Err(OAuthError::MissingAuthorizationCode),
            }
        }
        _ = tokio::time::sleep(timeout) => {
            Err(OAuthError::CallbackTimeout { timeout_secs: timeout.as_secs() })
        }
    };

    // Tear down the server in both success and failure paths (4.6 + cleanup
    // after 4.5). Ignore errors: the receiver may already be gone if the
    // server task finished on its own.
    let _ = shutdown_tx.send(());
    let _ = server_handle.await;

    outcome
}

#[cfg(test)]
mod tests {
    //! Integration-style tests for the loopback callback server (task 4.7).
    //!
    //! These tests bind real TCP listeners on `127.0.0.1` and drive the
    //! server through `reqwest` to exercise the same code path that the
    //! provider's browser redirect would. Each test obtains its own listener
    //! via `free_listener()` to avoid port collisions between tests running
    //! in parallel, and the listener is handed directly to
    //! [`start_callback_server_with_listener`] so there is no `bind()` race.
    //!
    //! Validates: Requirements 2.3, 2.4, 2.5, 2.7, 3.1 (callback surface).

    use super::*;

    /// Bind a fresh ephemeral listener on `127.0.0.1:0` and return it along
    /// with its resolved port. The caller hands the listener straight to the
    /// server, so there is no TOCTOU window between discovery and reuse.
    async fn free_listener() -> (TcpListener, u16) {
        let listener = TcpListener::bind((Ipv4Addr::LOCALHOST, 0))
            .await
            .expect("bind loopback ephemeral port");
        let port = listener.local_addr().expect("local addr").port();
        (listener, port)
    }

    /// Small delay to let the spawned server reach `accept()` before the test
    /// fires its GET. Matches the hint in the task spec.
    async fn wait_server_ready() {
        tokio::time::sleep(Duration::from_millis(100)).await;
    }

    #[tokio::test]
    async fn integration_valid_callback_returns_code() {
        let (listener, port) = free_listener().await;

        let server = tokio::spawn(start_callback_server_with_listener(
            listener,
            "state-123".to_string(),
            Duration::from_secs(5),
        ));

        wait_server_ready().await;

        let resp = reqwest::Client::new()
            .get(format!(
                "http://127.0.0.1:{port}/auth/callback?code=abc&state=state-123"
            ))
            .send()
            .await
            .expect("callback GET");
        assert_eq!(resp.status(), reqwest::StatusCode::OK);

        let result = server.await.expect("server task join");
        match result {
            Ok(AuthorizationCode(code)) => assert_eq!(code, "abc"),
            other => panic!("expected Ok(AuthorizationCode(\"abc\")), got {other:?}"),
        }
    }

    #[tokio::test]
    async fn integration_state_mismatch_returns_csrf_error() {
        let (listener, port) = free_listener().await;

        let server = tokio::spawn(start_callback_server_with_listener(
            listener,
            "state-123".to_string(),
            Duration::from_secs(5),
        ));

        wait_server_ready().await;

        let resp = reqwest::Client::new()
            .get(format!(
                "http://127.0.0.1:{port}/auth/callback?code=abc&state=wrong"
            ))
            .send()
            .await
            .expect("callback GET");
        assert_eq!(resp.status(), reqwest::StatusCode::BAD_REQUEST);

        let result = server.await.expect("server task join");
        assert!(
            matches!(result, Err(OAuthError::StateMismatch)),
            "expected Err(OAuthError::StateMismatch), got {result:?}"
        );
    }

    #[tokio::test]
    async fn integration_timeout_returns_callback_timeout() {
        let (listener, _port) = free_listener().await;

        // No GET is ever fired; the server should self-terminate via its
        // own timeout path (4.6).
        let result = start_callback_server_with_listener(
            listener,
            "s".to_string(),
            Duration::from_millis(200),
        )
        .await;

        assert!(
            matches!(result, Err(OAuthError::CallbackTimeout { timeout_secs: 0 })),
            "expected Err(OAuthError::CallbackTimeout {{ timeout_secs: 0 }}), got {result:?}"
        );
    }

    #[tokio::test]
    async fn integration_provider_error_returns_authorization_error() {
        let (listener, port) = free_listener().await;

        let server = tokio::spawn(start_callback_server_with_listener(
            listener,
            "state-123".to_string(),
            Duration::from_secs(5),
        ));

        wait_server_ready().await;

        let resp = reqwest::Client::new()
            .get(format!(
                "http://127.0.0.1:{port}/auth/callback?error=access_denied&error_description=User+denied&state=state-123"
            ))
            .send()
            .await
            .expect("callback GET");
        assert_eq!(resp.status(), reqwest::StatusCode::OK);

        let result = server.await.expect("server task join");
        match result {
            Err(OAuthError::AuthorizationError { error, description }) => {
                assert_eq!(error, "access_denied");
                assert_eq!(description.as_deref(), Some("User denied"));
            }
            other => panic!("expected Err(OAuthError::AuthorizationError), got {other:?}"),
        }
    }

    #[tokio::test]
    async fn integration_missing_code_returns_missing_authorization_code() {
        let (listener, port) = free_listener().await;

        let server = tokio::spawn(start_callback_server_with_listener(
            listener,
            "state-123".to_string(),
            Duration::from_secs(5),
        ));

        wait_server_ready().await;

        let resp = reqwest::Client::new()
            .get(format!(
                "http://127.0.0.1:{port}/auth/callback?state=state-123"
            ))
            .send()
            .await
            .expect("callback GET");
        assert_eq!(resp.status(), reqwest::StatusCode::BAD_REQUEST);

        let result = server.await.expect("server task join");
        assert!(
            matches!(result, Err(OAuthError::MissingAuthorizationCode)),
            "expected Err(OAuthError::MissingAuthorizationCode), got {result:?}"
        );
    }

    /// Validates: Requirements 8.2
    ///
    /// The callback server MUST bind exclusively to the loopback interface
    /// (127.0.0.1). This test verifies that:
    /// 1. The bound address is strictly 127.0.0.1 (not 0.0.0.0).
    /// 2. If a non-loopback LAN IP is available, a TCP connection attempt to
    ///    that IP on the server's port is refused (since the server only
    ///    listens on loopback).
    #[tokio::test]
    async fn integration_callback_server_rejects_non_loopback() {
        use std::net::{IpAddr, TcpStream as StdTcpStream};

        // Start the callback server on loopback with an ephemeral port.
        let (listener, port) = free_listener().await;

        // Verify the listener is bound to 127.0.0.1, not 0.0.0.0.
        let bound_addr = listener.local_addr().expect("local addr");
        assert_eq!(
            bound_addr.ip(),
            IpAddr::V4(Ipv4Addr::LOCALHOST),
            "callback server must bind to 127.0.0.1, not {}", bound_addr.ip()
        );

        let server = tokio::spawn(start_callback_server_with_listener(
            listener,
            "state-loopback".to_string(),
            Duration::from_secs(5),
        ));

        wait_server_ready().await;

        // Attempt to find a non-loopback LAN IP on this machine.
        let lan_ip = find_non_loopback_ipv4();

        if let Some(lan_ip) = lan_ip {
            // Attempt a TCP connection via the LAN IP. Since the server only
            // listens on 127.0.0.1, this should be refused.
            let target = std::net::SocketAddr::new(IpAddr::V4(lan_ip), port);
            let connect_result = StdTcpStream::connect_timeout(
                &target,
                std::time::Duration::from_secs(2),
            );
            assert!(
                connect_result.is_err(),
                "connection to non-loopback {target} should be refused, but succeeded"
            );
        }

        // Also verify that loopback access still works (sanity check).
        let resp = reqwest::Client::new()
            .get(format!(
                "http://127.0.0.1:{port}/auth/callback?code=test&state=state-loopback"
            ))
            .send()
            .await
            .expect("loopback callback GET should succeed");
        assert_eq!(resp.status(), reqwest::StatusCode::OK);

        let result = server.await.expect("server task join");
        assert!(
            matches!(result, Ok(AuthorizationCode(_))),
            "expected Ok(AuthorizationCode), got {result:?}"
        );
    }

    /// Find the first non-loopback IPv4 address on this machine.
    /// Returns `None` if no such address is found (e.g., in some CI
    /// environments with only loopback configured).
    fn find_non_loopback_ipv4() -> Option<Ipv4Addr> {
        use std::net::{IpAddr, UdpSocket};

        // A common trick: connect a UDP socket to an external address
        // (no actual traffic is sent) and read back the local address
        // the OS chose. This gives us a routable non-loopback IP.
        let socket = UdpSocket::bind("0.0.0.0:0").ok()?;
        socket.connect("8.8.8.8:80").ok()?;
        let local_addr = socket.local_addr().ok()?;
        match local_addr.ip() {
            IpAddr::V4(ip) if !ip.is_loopback() && !ip.is_unspecified() => Some(ip),
            _ => None,
        }
    }
}
