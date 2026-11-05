#![allow(dead_code)]
//! GitHub Copilot credential-lifecycle manager (task 5 of the
//! `github-copilot-oauth-provider` spec).
//!
//! [`CopilotTokenManager`] owns the whole credential lifecycle around the
//! task 2–4 primitives ([`device_flow`], [`copilot_token`], [`store`]):
//!
//! * **Session state machine** —
//!   [`CopilotSessionState::Unauthenticated`]/`Pending`/`Authenticated`/
//!   `Expired`/`Error`.
//! * **Login (Req 2.1–2.4, 2.6)** — `start_login` requests a device grant
//!   and spawns the poll; a second call while a login is in flight
//!   returns the SAME verification URL + user code (in-flight dedupe).
//!   On device-flow success the Copilot token exchange runs immediately,
//!   BEFORE the session flips to `Authenticated` (Req 2.4).
//! * **Background refresh (Req 3.2)** — 60 s tick; re-exchange when the
//!   cached Copilot token is within the 5-min pre-expiry window;
//!   transient failures back off `2 s / 4 s / 8 s` for at most 3
//!   attempts. All cadence constants come from
//!   [`crate::oauth::lifecycle`] (shared with the OpenAI manager).
//! * **Entitlement short-circuit (Req 3.3)** — an exchange 401/403 marks
//!   the session `Expired` immediately (no retry storm); the GitHub
//!   token stays on disk for a fast re-login.
//! * **Inline refresh (Req 3.4)** — [`CopilotTokenManager::get_copilot_token`]
//!   refreshes-and-retries the exchange ONCE inline when the cached
//!   token has just expired, before failing the attempt.
//! * **Logout** — cancels the refresh task (abort-safe), cancels any
//!   in-flight login, clears the in-memory cache, deletes the store
//!   file.
//!
//! # Endpoints injectability
//!
//! [`CopilotEndpoints`] bundles the device-flow endpoints plus the
//! exchange token URL; production wiring is
//! [`CopilotEndpoints::production`] and tests inject wiremock URLs so
//! the manager is fully drivable without real network access.
//!
//! # Security invariants
//!
//! * Token material is never logged (Req 8.4): GitHub token, Copilot
//!   bearer, and device codes only flow through
//!   `#[tracing::instrument(skip(...))]` boundaries and types with
//!   redacted `Debug` impls ([`CopilotStoredCredentials`],
//!   [`CopilotToken`], [`DeviceCodeGrant`], [`GithubToken`]).
//! * The in-memory token cache avoids per-request disk I/O; the
//!   encrypted file is only touched on login, refresh, load, logout.

use std::fmt;
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::{Arc, Mutex as StdMutex, Weak};
use std::time::SystemTime;

use serde::Serialize;
use tokio::sync::{watch, Mutex, RwLock};
use tokio::task::JoinHandle;

use super::super::error::OAuthError;
use super::super::lifecycle::{
    backoff_delay_secs, BACKOFF_MAX_ATTEMPTS, REFRESH_TICK_INTERVAL, REFRESH_WINDOW_SECS,
};
use super::copilot_token::{
    exchange_copilot_token, CopilotToken, COPILOT_TOKEN_URL, DEFAULT_COPILOT_API_BASE,
};
use super::device_flow::{
    poll_for_token, start_device_flow, DeviceCodeGrant, DeviceFlowEndpoints, GithubToken,
    COPILOT_OAUTH_SCOPE,
};
use super::store::{CopilotStoredCredentials, CopilotTokenStore};

/// Session state machine for the Copilot credential lifecycle (design
/// Component 3).
///
/// Mirrors [`super::super::flow::OAuthSessionState`] in spirit but keeps
/// a Copilot-specific shape: `Pending` covers an in-flight device-flow
/// login and `Error` covers non-terminal refresh exhaustion. Serialized
/// with a `state` tag for the admin status endpoint; carries only safe
/// metadata, never token material.
#[derive(Debug, Clone, PartialEq, Eq, Serialize)]
#[serde(rename_all = "lowercase", tag = "state")]
pub enum CopilotSessionState {
    /// No stored credentials; user has not logged in.
    Unauthenticated,
    /// A device-flow login is in flight (user code issued).
    Pending,
    /// Copilot token held. `expires_at` is a Unix timestamp (seconds).
    Authenticated { expires_at: u64 },
    /// Exchange returned 401/403 (entitlement gone) or the credential is
    /// otherwise dead. The GitHub token is retained for a fast re-login.
    Expired,
    /// Transient refresh failures exhausted the backoff budget;
    /// recoverable on the next tick or inline attempt.
    Error,
}

/// Injectable endpoint set for the Copilot manager: both device-flow
/// legs plus the exchange token URL. Production defaults come from the
/// constants modules; wiremock tests inject mock URLs.
#[derive(Debug, Clone)]
pub struct CopilotEndpoints {
    /// Device-flow endpoints (device code issuance + grant polling).
    pub device_flow: DeviceFlowEndpoints,
    /// Copilot token exchange URL (`copilot_internal/v2/token`).
    pub copilot_token_url: String,
}

impl CopilotEndpoints {
    /// Production GitHub/Copilot endpoints.
    pub fn production() -> Self {
        Self {
            device_flow: DeviceFlowEndpoints::production(),
            copilot_token_url: COPILOT_TOKEN_URL.to_string(),
        }
    }
}

/// Non-secret view of an in-flight device-flow login, returned by
/// [`CopilotTokenManager::start_login`] on both fresh and deduped calls.
///
/// The verification URL and user code are exactly the values the human
/// user must see; nothing secret-shaped is carried (the device code
/// stays inside the manager).
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct LoginPrompt {
    /// URL the user visits to approve the request.
    pub verification_uri: String,
    /// Short code the user types at the verification URL.
    pub user_code: String,
    /// Device-code lifetime in seconds (for UI countdowns).
    pub expires_in: u64,
}

/// What a call to [`CopilotTokenManager::start_login`] did.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum LoginStart {
    /// A fresh device flow was started; the poll task is in flight.
    Started(LoginPrompt),
    /// A login was already in flight (Req 2.6); the SAME verification
    /// URL and user code are returned without starting a second flow.
    InFlight(LoginPrompt),
}

/// In-memory snapshot of the credential material the manager needs on
/// the hot path.
///
/// Token fields are redacted by the manual `Debug` impl; `machine_id`
/// is an opaque non-secret identifier (see [`super::generate_machine_id`]).
#[derive(Clone)]
struct CopilotCredentialCache {
    /// Long-lived GitHub OAuth token (`gho_…`). Never logged.
    github_access_token: String,
    /// Short-lived Copilot API bearer. Never logged.
    copilot_token: String,
    /// Expiry of `copilot_token` as a Unix timestamp (seconds).
    copilot_expires_at: u64,
    /// Copilot API base URL (`endpoints.api` override or default).
    api_base: String,
    /// Stable machine id (generated at first login, persisted in the
    /// encrypted store).
    machine_id: String,
    /// Copilot SKU reported by the exchange, when present.
    sku: Option<String>,
    /// Space-separated OAuth scope string granted by GitHub.
    scopes: String,
}

impl fmt::Debug for CopilotCredentialCache {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("CopilotCredentialCache")
            .field("github_access_token", &"<redacted>")
            .field("copilot_token", &"<redacted>")
            .field("copilot_expires_at", &self.copilot_expires_at)
            .field("api_base", &self.api_base)
            .field("machine_id", &self.machine_id)
            .field("sku", &self.sku)
            .field("scopes", &self.scopes)
            .finish()
    }
}

/// In-flight device-flow login bookkeeping (Req 2.6 dedupe target).
struct InFlightLogin {
    prompt: LoginPrompt,
    /// Cancellation sender for the poll task.
    cancel: watch::Sender<bool>,
    /// Handle to the spawned poll task (aborted on logout).
    task: JoinHandle<()>,
}

impl InFlightLogin {
    /// Abort the poll task. Idempotent — tokio `JoinHandle::abort` on an
    /// already-finished task is a no-op.
    fn abort(&self) {
        self.task.abort();
    }
}

/// State of the login slot while [`CopilotTokenManager::start_login`]
/// runs its device-grant round-trip.
///
/// `Starting` marks that the (lock-free) network leg is in flight so a
/// concurrent caller dedupes onto it instead of issuing a second device
/// code (Req 2.6); it carries no prompt yet — deduped callers wait for
/// the grant to publish [`LoginSlot::InFlight`].
enum LoginSlot {
    /// Device-code issuance in progress; no prompt to show yet.
    Starting,
    /// Grant issued; the poll task is running with a publishable prompt.
    InFlight(InFlightLogin),
}

/// GitHub Copilot credential-lifecycle manager (design Component 3).
///
/// Cheap to share as an `Arc<CopilotTokenManager>`; all mutable state
/// sits behind `RwLock` / `Mutex`. Mirrors the OpenAI
/// [`super::super::OAuthManager`] orchestration patterns with
/// Copilot-specific semantics (device flow instead of PKCE callback,
/// exchange instead of refresh-token grant).
pub struct CopilotTokenManager {
    store: Arc<CopilotTokenStore>,
    session_state: Arc<RwLock<CopilotSessionState>>,
    /// In-memory credential cache — the hot path (get_copilot_token)
    /// never touches the disk (Req 3.2 "cache the token").
    cache: Arc<RwLock<Option<CopilotCredentialCache>>>,
    /// Login slot (Req 2.6 dedupe target). `None` when no login is
    /// pending. See [`LoginSlot`] for the two in-flight phases.
    login: Arc<Mutex<Option<LoginSlot>>>,
    /// Background refresh task handle; aborted on logout. A synchronous
    /// mutex: never held across an await, so async access is safe and
    /// `start_refresh_loop` can record the handle without a spawn hop.
    refresh_task: StdMutex<Option<JoinHandle<()>>>,
    /// Single-flight guard around the Copilot token exchange: N
    /// concurrent dispatches at cache-expiry coalesce onto ONE exchange
    /// instead of bursting the endpoint (GitHub 403s bursts).
    exchange_lock: Arc<Mutex<()>>,
    /// Last terminal device-flow login error message (Req 2.3),
    /// surfaced through the admin status snapshot. `None` after a
    /// successful login or logout.
    last_login_error: Arc<Mutex<Option<String>>>,
    /// Negative-result cache for `machine_id()`/`api_base()`: set once
    /// a store load comes back empty, so the per-request hot path in
    /// api-key mode (no OAuth session) is a pure in-memory read instead
    /// of disk + decrypt. Reset on login/logout/session load.
    store_known_empty: Arc<AtomicBool>,
    /// Weak self-reference for spawned child tasks (login/refresh) to
    /// reach manager methods that need `Arc<Self>` (refresh-loop
    /// restart) without forcing callers of [`Self::start_login`] to
    /// hold an Arc. Populated by [`Self::start_refresh_loop`] /
    /// [`Self::set_self_reference`]; stays `None` only until either is
    /// called (gateway wiring always calls `start_refresh_loop`).
    self_ref: StdMutex<Option<Weak<CopilotTokenManager>>>,
    http_client: reqwest::Client,
    /// Injectable endpoints (production constants by default; wiremock
    /// URLs under test).
    endpoints: RwLock<CopilotEndpoints>,
}

impl CopilotTokenManager {
    /// Construct a manager over an existing store and HTTP client with
    /// production endpoints.
    ///
    /// The initial state is
    /// [`CopilotSessionState::Unauthenticated`]; call
    /// [`Self::load_existing_session`] afterwards to reconcile with
    /// whatever is already on disk.
    pub fn new(store: CopilotTokenStore, http_client: reqwest::Client) -> Self {
        Self::with_endpoints(store, http_client, CopilotEndpoints::production())
    }

    /// Construct a manager with explicitly injected endpoints (tests
    /// point these at wiremock).
    pub fn with_endpoints(
        store: CopilotTokenStore,
        http_client: reqwest::Client,
        endpoints: CopilotEndpoints,
    ) -> Self {
        Self {
            store: Arc::new(store),
            session_state: Arc::new(RwLock::new(CopilotSessionState::Unauthenticated)),
            cache: Arc::new(RwLock::new(None)),
            login: Arc::new(Mutex::new(None)),
            refresh_task: StdMutex::new(None),
            exchange_lock: Arc::new(Mutex::new(())),
            last_login_error: Arc::new(Mutex::new(None)),
            store_known_empty: Arc::new(AtomicBool::new(false)),
            self_ref: StdMutex::new(None),
            http_client,
            endpoints: RwLock::new(endpoints),
        }
    }

    /// Read any credentials already on disk, prime the in-memory cache,
    /// and reconcile [`Self::session_state`].
    ///
    /// Mapping (mirrors the OpenAI manager's `load_existing_session`):
    /// * No file / unreadable blob → `Unauthenticated` (corrupt store
    ///   is a warn + unauthenticated, design Error Handling 7).
    /// * GitHub token present but no Copilot token or an expired one →
    ///   `Expired` (fast re-login path keeps the GitHub token).
    /// * Valid Copilot token → `Authenticated { expires_at }`.
    pub async fn load_existing_session(&self) -> Result<(), OAuthError> {
        let credentials = match self.store.load() {
            Ok(creds) => creds,
            Err(e) => {
                tracing::warn!(
                    error = %e,
                    "copilot: stored credential file is unreadable; treating as unauthenticated"
                );
                None
            }
        };

    let now = now_unix_secs();
    let mut cache = self.cache.write().await;
    let mut state = self.session_state.write().await;
    match credentials {
        None => {
            *cache = None;
            *state = CopilotSessionState::Unauthenticated;
            self.store_known_empty.store(true, Ordering::Release);
        }
        Some(creds) => {
            // The token/expiry pair is only a valid credential when BOTH
            // halves are present; a store file with `copilot_expires_at`
            // set but `copilot_token` null (or vice versa) must never
            // panic at startup — it maps to Expired (the GitHub token
            // is retained for a fast re-login).
            let copilot_alive = match (&creds.copilot_token, creds.copilot_expires_at) {
                (Some(_), Some(expires_at)) => expires_at > now,
                _ => false,
            };
            if copilot_alive {
                *cache = Some(CopilotCredentialCache {
                    github_access_token: creds.github_access_token,
                    copilot_token: creds
                        .copilot_token
                        .expect("copilot_alive implies copilot_token present"),
                    copilot_expires_at: creds
                        .copilot_expires_at
                        .expect("copilot_alive implies copilot_expires_at present"),
                    api_base: creds.api_base,
                    machine_id: creds.machine_id,
                    sku: creds.sku,
                    scopes: creds.scopes,
                });
                *state = CopilotSessionState::Authenticated {
                    expires_at: creds
                        .copilot_expires_at
                        .expect("copilot_alive implies copilot_expires_at present"),
                };
            } else {
                // GitHub token retained for fast re-login (Req 3.3).
                *cache = None;
                *state = CopilotSessionState::Expired;
            }
        }
    }
    Ok(())
    }

    /// Snapshot the current session state for read-only consumers
    /// (admin status endpoint, router eligibility checks).
    pub async fn session_state(&self) -> CopilotSessionState {
        self.session_state.read().await.clone()
    }

    /// Last terminal device-flow login error (Req 2.3), for the admin
    /// status snapshot. `None` when the last login succeeded, was
    /// cancelled, or the session was logged out / re-loaded.
    ///
    /// Carries only the actionable `Display` text of the terminal
    /// error (e.g. denial / device-code expiry) — never token material.
    pub async fn last_login_error(&self) -> Option<String> {
        self.last_login_error.lock().await.clone()
    }

    /// Start a device-flow login (Req 2.1 / 2.6).
    ///
    /// A fresh call requests a device grant and spawns the poll task;
    /// the session flips to `Pending` and the user-facing prompt is
    /// returned. If a login is already in flight (including the brief
    /// device-code issuance round-trip of another caller), the SAME
    /// verification URL and user code are returned
    /// ([`LoginStart::InFlight`]) without issuing a second device code.
    ///
    /// The login mutex is NOT held across the issuance round-trip
    /// (bounded by the device-flow client's 5 s timeout): the slot is
    /// reserved with a [`LoginSlot::Starting`] placeholder so cancel /
    /// logout are never blocked behind the network call, and the grant
    /// is published (or the reservation cleared) when it settles.
    ///    /// On device-flow success the poll task immediately performs the
    /// Copilot token exchange BEFORE flipping the session to
    /// `Authenticated` (Req 2.4); a denied/expired/cancelled flow
    /// returns the session to its pre-login state.
#[tracing::instrument(skip(self))]
pub async fn start_login(&self) -> Result<LoginStart, OAuthError> {
    // Req 2.6 dedupe, phase 1 — reserve the slot, return an
    // already-published prompt, or fall in behind another caller's
    // issuance round-trip.
    enum Role {
        /// We reserved the slot and perform the network leg.
        Issuer,
        /// An in-flight login is already published; dedupe onto it.
        Dedupe(LoginPrompt),
        /// Another caller is issuing a device code right now; wait
        /// (bounded by the issuance timeout) for their prompt.
        WaitExisting,
    }
    let role = {
        let mut login = self.login.lock().await;
        match login.as_ref() {
            Some(LoginSlot::InFlight(in_flight)) => {
                if !in_flight.task.is_finished() {
                    Role::Dedupe(in_flight.prompt.clone())
                } else {
                    // A finished task is indistinguishable from no
                    // login; clear the slot and become the issuer.
                    let finished = login.take();
                    if let Some(LoginSlot::InFlight(finished)) = finished {
                        finished.abort();
                    }
                    *login = Some(LoginSlot::Starting);
                    Role::Issuer
                }
            }
            Some(LoginSlot::Starting) => Role::WaitExisting,
            None => {
                *login = Some(LoginSlot::Starting);
                Role::Issuer
            }
        }
    };
    match role {
        Role::Dedupe(prompt) => return Ok(LoginStart::InFlight(prompt)),
        Role::WaitExisting => {
            // Another caller's issuance is in flight. Wait briefly for
            // them to publish (their round-trip is bounded by the
            // device-flow client's 5 s timeout) and dedupe onto the
            // published prompt. If their issuance failed (slot back to
            // None), start our own flow — the failed caller released
            // the reservation.
            for _ in 0..100 {
                tokio::time::sleep(std::time::Duration::from_millis(100)).await;
                let login = self.login.lock().await;
                match login.as_ref() {
                    Some(LoginSlot::InFlight(in_flight)) => {
                        return Ok(LoginStart::InFlight(in_flight.prompt.clone()));
                    }
                    Some(LoginSlot::Starting) => continue,
                    None => break,
                }
            }
            // Slot is free (their issuance failed or was cleared):
            // recurse to take the issuer role ourselves.
            return Box::pin(self.start_login()).await;
        }
        Role::Issuer => {}
    }
    // Lock released: cancel/logout run freely while the grant
    // round-trip is in flight.

    let endpoints = self.endpoints.read().await.clone();
    let grant = match start_device_flow(&self.http_client, &endpoints.device_flow).await {
        Ok(grant) => grant,
        Err(e) => {
            // Issuance failed: release the reservation so the next
            // caller can retry (or pick up another waiter's flow).
            self.clear_starting_slot().await;
            return Err(e);
        }
    };

    let prompt = LoginPrompt {
        verification_uri: grant.verification_uri.clone(),
        user_code: grant.user_code.clone(),
        expires_in: grant.expires_in,
    };

    // Publish the grant: re-acquire the slot and install the in-flight
    // login. Only proceed when OUR reservation survived (logout may
    // have cleared it mid-flight).
    let published = {
        let mut login = self.login.lock().await;
        match login.as_ref() {
            Some(LoginSlot::Starting) => {
                // Flip to Pending before spawning so a status poll
                // racing the spawn still observes the in-flight login.
                {
                    let mut state = self.session_state.write().await;
                    *state = CopilotSessionState::Pending;
                }

                let (cancel_tx, cancel_rx) = watch::channel(false);
                let task = {
                    let manager_endpoints = endpoints.clone();
                    let poll_endpoints = endpoints.device_flow.clone();
                    let http_client = self.http_client.clone();
                    let store = self.store.clone();
                    let session_state = self.session_state.clone();
                    let cache = self.cache.clone();
                    let login_slot = self.login.clone();
                    let last_login_error = self.last_login_error.clone();
                    let store_known_empty = self.store_known_empty.clone();
                    let manager_self = self.self_reference();
                    tokio::spawn(async move {
                        Self::run_login(
                            &http_client,
                            &manager_endpoints,
                            &poll_endpoints,
                            grant,
                            cancel_rx,
                            store,
                            session_state,
                            cache,
                            login_slot,
                            last_login_error,
                            manager_self,
                            store_known_empty,
                        )
                        .await;
                    })
                };
                *login = Some(LoginSlot::InFlight(InFlightLogin {
                    prompt: prompt.clone(),
                    cancel: cancel_tx,
                    task,
                }));
                true
            }
            _ => false,
        }
    };
    if !published {
        // Logout resolved while our issuance was in flight: nothing
        // was spawned, no task exists to abort, and the session was
        // never flipped to Pending by us. Surface a retryable error;
        // the slot semantics guarantee no orphaned device code.
        tracing::warn!("copilot: login slot changed during device-code issuance; aborting start");
        return Err(OAuthError::InvalidTokenResponse(
            "login state changed during device-code issuance; retry the login".to_string(),
        ));
    }

    Ok(LoginStart::Started(prompt))
}

    /// Clear a [`LoginSlot::Starting`] reservation left behind by a
    /// failed device-code issuance. Only clears the slot when it is
    /// still in the `Starting` phase — a published (or absent) slot is
    /// left untouched.
    async fn clear_starting_slot(&self) {
        let mut login = self.login.lock().await;
        if matches!(login.as_ref(), Some(LoginSlot::Starting)) {
            *login = None;
        }
    }

    /// Record the manager's own `Arc` so spawned child tasks (login /
    /// refresh) can call `Arc<Self>`-consuming methods such as the
    /// post-login refresh-loop restart. Called by
    /// [`Self::start_refresh_loop`] (gateway wiring always invokes it);
    /// direct construction in tests may call it explicitly.
    pub fn set_self_reference(self: &Arc<Self>) {
        let mut slot = self.self_ref.lock().expect("self_ref mutex poisoned");
        *slot = Some(Arc::downgrade(self));
    }

    /// Upgrade the recorded self-reference, if one was set.
    fn self_reference(&self) -> Option<Arc<CopilotTokenManager>> {
        self.self_ref
            .lock()
            .expect("self_ref mutex poisoned")
            .as_ref()
            .and_then(Weak::upgrade)
    }

    /// Cancel an in-flight login (admin cancel endpoint). Idempotent.
    ///
    /// Sends the cancellation signal and lets the poll task observe it
    /// (the poll races every await against the watch channel), restore
    /// the pre-login session state, and clear its own slot. The task is
    /// NOT aborted here so the graceful state restore always runs;
    /// [`Self::complete_logout`] force-aborts any remaining task after
    /// signalling.
    #[tracing::instrument(skip(self))]
    pub async fn cancel_login(&self) {
        let login = self.login.lock().await;
        if let Some(LoginSlot::InFlight(in_flight)) = login.as_ref() {
            let _ = in_flight.cancel.send(true);
        }
    }

    /// Body of the spawned login task (Req 2.2 / 2.3 / 2.4).
    ///
    /// Polls the device-flow token endpoint, then — on success —
    /// exchanges the GitHub token for a Copilot token before flipping
    /// the session to `Authenticated`. Terminal errors return the
    /// session to `Unauthenticated` unless a credential already exists
    /// on disk (then `Expired`), so a failed first login never leaves a
    /// misleading state behind. Terminal device-flow failures
    /// (denial / code expiry / endpoint errors) also persist their
    /// actionable message into `last_login_error` so the admin status
    /// endpoint can surface WHY the login ended (Req 2.3); a successful
    /// exchange or a deliberate cancellation clears it.
    #[tracing::instrument(skip(http_client, manager_endpoints, poll_endpoints, grant, cancel, store, session_state, cache, login_slot, last_login_error, manager_self, store_known_empty))]
    async fn run_login(
        http_client: &reqwest::Client,
        manager_endpoints: &CopilotEndpoints,
        poll_endpoints: &DeviceFlowEndpoints,
        grant: DeviceCodeGrant,
        cancel: watch::Receiver<bool>,
        store: Arc<CopilotTokenStore>,
        session_state: Arc<RwLock<CopilotSessionState>>,
        cache: Arc<RwLock<Option<CopilotCredentialCache>>>,
        login_slot: Arc<Mutex<Option<LoginSlot>>>,
        last_login_error: Arc<Mutex<Option<String>>>,
        manager_self: Option<Arc<CopilotTokenManager>>,
        store_known_empty: Arc<AtomicBool>,
    ) {
        let github_token = match poll_for_token(http_client, poll_endpoints, &grant, cancel).await
        {
            Ok(token) => token,
            Err(OAuthError::DeviceFlowCancelled) => {
                // Deliberate cancellation — restore the pre-login
                // state. Not a failure: leave any previous
                // last-login-error untouched (cancel is not a denial).
                Self::reset_login_state(&store, &session_state, &cache).await;
                return;
            }
            Err(e) => {
                tracing::warn!(error = %e, "copilot: device-flow login failed");
                // Req 2.3: surface the specific terminal outcome (e.g.
                // access_denied / expired_token) through the status
                // snapshot instead of dropping it on the floor.
                *last_login_error.lock().await = Some(e.to_string());
                Self::reset_login_state(&store, &session_state, &cache).await;
                return;
            }
        };

        // Req 2.4: exchange BEFORE reporting authenticated. A failed
        // exchange leaves the session not-authenticated (the GitHub
        // token is still persisted for a fast re-login attempt).
        match Self::exchange_and_persist(
            http_client,
            &manager_endpoints.copilot_token_url,
            &github_token,
            &store,
            &session_state,
            &cache,
            &store_known_empty,
        )
        .await
        {
            Ok(()) => {
                tracing::info!("copilot: login completed and session authenticated");
                // Successful login clears the surfaced login error
                // (Req 2.3) and revives the background refresh loop
                // (logout aborted the previous one; restart is
                // idempotent so no loops ever stack). Skipped only when
                // the manager was never wired with a self reference
                // (unit-test harnesses).
                *last_login_error.lock().await = None;
                if let Some(manager) = manager_self.as_ref() {
                    manager.restart_refresh_loop();
                }
            }
            Err(e) => {
                tracing::warn!(error = %e, "copilot: post-auth token exchange failed")
            }
        }

        // Clear the in-flight slot so a subsequent start_login begins a
        // fresh flow.
        let mut slot = login_slot.lock().await;
        if let Some(LoginSlot::InFlight(in_flight)) = slot.take() {
            in_flight.abort();
        }
    }

    /// Restore the session to its on-disk truth after a terminal login
    /// outcome (cancelled/denied/expired/failed exchange).
    async fn reset_login_state(
        store: &Arc<CopilotTokenStore>,
        session_state: &Arc<RwLock<CopilotSessionState>>,
        cache: &Arc<RwLock<Option<CopilotCredentialCache>>>,
    ) {
        let stored = store.load().unwrap_or(None);
        let now = now_unix_secs();
        let mut cache_guard = cache.write().await;
        let mut state = session_state.write().await;
        match stored {
            Some(creds) if creds.copilot_expires_at.map(|e| e > now).unwrap_or(false) => {
                *state = CopilotSessionState::Authenticated {
                    expires_at: creds.copilot_expires_at.expect("checked above"),
                };
            }
            Some(_) => {
                *cache_guard = None;
                *state = CopilotSessionState::Expired;
            }
            None => {
                *cache_guard = None;
                *state = CopilotSessionState::Unauthenticated;
            }
        }
    }

    /// Exchange the GitHub token for a Copilot token, persist the
    /// combined credential (generating + persisting the machine id on
    /// first login), prime the cache, and flip the session to
    /// `Authenticated`.
    ///
    /// Post-login bookkeeping (clearing the last-login error, resetting
    /// the empty-store flag, restarting the background refresh loop)
    /// lives in the caller ([`Self::run_login`]) which holds the
    /// manager self-reference.
    #[tracing::instrument(skip(http_client, github_token, store, session_state, cache, store_known_empty))]
    async fn exchange_and_persist(
        http_client: &reqwest::Client,
        copilot_token_url: &str,
        github_token: &GithubToken,
        store: &Arc<CopilotTokenStore>,
        session_state: &Arc<RwLock<CopilotSessionState>>,
        cache: &Arc<RwLock<Option<CopilotCredentialCache>>>,
        store_known_empty: &Arc<AtomicBool>,
    ) -> Result<(), OAuthError> {
        let copilot = exchange_copilot_token(
            http_client,
            &github_token.access_token,
            copilot_token_url,
        )
        .await?;

        // Machine id: reuse the one already in the store (stable across
        // restarts); generate + persist a fresh one on first login.
        let existing = store.load().unwrap_or(None);
        let machine_id = existing
            .map(|creds| creds.machine_id)
            .unwrap_or_else(super::generate_machine_id);

        let credentials = CopilotStoredCredentials {
            github_access_token: github_token.access_token.clone(),
            copilot_token: Some(copilot.token.clone()),
            copilot_expires_at: Some(copilot.expires_at),
            api_base: copilot.api_base().to_string(),
            machine_id,
            sku: copilot.sku.clone(),
            scopes: COPILOT_OAUTH_SCOPE.to_string(),
            updated_at: now_unix_secs(),
        };
        store.save(&credentials)?;
        store_known_empty.store(false, Ordering::Release);

        // Cache before flipping the state so a racing get_copilot_token
        // never observes Authenticated with an empty cache.
        *cache.write().await = Some(CopilotCredentialCache {
            github_access_token: credentials.github_access_token,
            copilot_token: copilot.token,
            copilot_expires_at: copilot.expires_at,
            api_base: credentials.api_base,
            machine_id: credentials.machine_id,
            sku: credentials.sku,
            scopes: credentials.scopes,
        });
        let mut state = session_state.write().await;
        *state = CopilotSessionState::Authenticated {
            expires_at: copilot.expires_at,
        };
        Ok(())
    }

    /// Resolve the Copilot API bearer token for dispatch (Req 3.4).
    ///
    /// * Session must be `Authenticated` (or `Error` — recoverable).
    /// * Hot path: serve from the in-memory cache when the token is
    /// still valid — no disk I/O, no network.
    /// * When the cached token has just expired, refresh-and-retry the
    /// exchange ONCE inline before failing. The exchange path is
    /// single-flight: concurrent callers at cache-expiry coalesce onto
    /// ONE exchange (bursting the endpoint earns a GitHub 403), with
    /// the losers re-reading the refreshed cache.
    /// * A 401/403 exchange marks the session `Expired` immediately;
    /// the GitHub token stays on disk for a fast re-login.
    #[tracing::instrument(skip(self))]
    pub async fn get_copilot_token(&self) -> Result<String, OAuthError> {
        match *self.session_state.read().await {
            CopilotSessionState::Authenticated { .. }
            | CopilotSessionState::Error => {}
            _ => {
                return Err(OAuthError::InvalidTokenResponse(
                    "copilot session is not authenticated".to_string(),
                ))
            }
        }

        {
            let cache = self.cache.read().await;
            if let Some(creds) = cache.as_ref() {
                if creds.copilot_expires_at > now_unix_secs() {
                    return Ok(creds.copilot_token.clone());
                }
                // Just expired → fall through to the inline refresh.
            } else {
                // Authenticated but empty cache (e.g. restart raced the
                // load): fall through to the inline refresh too.
            }
        }

        // Req 3.4: exactly ONE inline refresh-and-retry, single-flight.
        match self.refresh_single_flight().await {
            Ok(token) => Ok(token),
            Err(e) => {
                tracing::warn!(error = %e, "copilot: inline token refresh failed");
                Err(e)
            }
        }
    }

    /// Force an immediate Copilot token exchange, bypassing the
    /// unexpired-cache short-circuit (router 401-rescue hook).
    ///
    /// Marks the cached token as expired first, so the exchange cannot
    /// be skipped, then performs a single-flight
    /// [`Self::refresh_once`]: a revoked-but-unexpired cached token is
    /// replaced with a freshly exchanged one (or the session lands in
    /// `Expired` on a 401/403 entitlement rejection). Concurrent
    /// callers coalesce onto one exchange, same as the inline path.
    #[tracing::instrument(skip(self))]
    pub async fn force_refresh(&self) -> Result<String, OAuthError> {
        match *self.session_state.read().await {
            CopilotSessionState::Authenticated { .. }
            | CopilotSessionState::Error => {}
            _ => {
                return Err(OAuthError::InvalidTokenResponse(
                    "copilot session is not authenticated".to_string(),
                ))
            }
        }

        // Invalidate the cached bearer by treating it as expired; the
        // store copy keeps its real expiry (it will be overwritten by
        // the exchange below anyway).
        {
            let mut cache = self.cache.write().await;
            if let Some(creds) = cache.as_mut() {
                creds.copilot_expires_at = 0;
            }
        }

        self.refresh_single_flight().await
    }

    /// Single-flight wrapper around [`Self::refresh_once`] (M4): the
    /// first caller at cache-expiry performs the exchange while
    /// concurrent callers wait on the same mutex; after it completes,
    /// each waiter re-reads the cache and only exchanges again if the
    /// refreshed token is still unusable (it never is on the success
    /// path — the winner's cache write satisfies everyone).
    async fn refresh_single_flight(&self) -> Result<String, OAuthError> {
        let _guard = self.exchange_lock.lock().await;
        // Re-check under the lock: the winner may have refreshed while
        // we were queued behind the mutex.
        {
            let cache = self.cache.read().await;
            if let Some(creds) = cache.as_ref() {
                if creds.copilot_expires_at > now_unix_secs() {
                    return Ok(creds.copilot_token.clone());
                }
            }
        }
        self.refresh_once().await
    }

    /// Machine id for dispatch headers (stable across restarts).
    ///
    /// Returns the id from the cache, falling back to the store;
    /// `None` when no credential exists. Once a store load has come
    /// back empty, the negative result is cached in memory so the
    /// per-request hot path (api-key mode with no OAuth session) is a
    /// pure in-memory read; the flag resets on login/logout/session
    /// load.
    #[tracing::instrument(skip(self))]
    pub async fn machine_id(&self) -> Option<String> {
        if let Some(creds) = self.cache.read().await.as_ref() {
            return Some(creds.machine_id.clone());
        }
        if self.store_known_empty.load(Ordering::Acquire) {
            return None;
        }
        let loaded = self.store.load().ok().flatten().map(|c| c.machine_id);
        if loaded.is_none() {
            self.store_known_empty.store(true, Ordering::Release);
        }
        loaded
    }

    /// Copilot API base URL for dispatch (`endpoints.api` override or
    /// the production default).
    ///
    /// Same negative-result caching as [`Self::machine_id`]: after the
    /// first empty store load the hot path skips the disk round-trip.
    #[tracing::instrument(skip(self))]
    pub async fn api_base(&self) -> Option<String> {
        if let Some(creds) = self.cache.read().await.as_ref() {
            return Some(creds.api_base.clone());
        }
        if self.store_known_empty.load(Ordering::Acquire) {
            return Some(DEFAULT_COPILOT_API_BASE.to_string());
        }
        let loaded = self
            .store
            .load()
            .ok()
            .flatten()
            .map(|c| c.api_base)
            .filter(|base| !base.is_empty());
        if loaded.is_none() {
            self.store_known_empty.store(true, Ordering::Release);
        }
        loaded.or_else(|| Some(DEFAULT_COPILOT_API_BASE.to_string()))
    }

    /// Perform a single exchange attempt against the GitHub token
    /// (cached, falling back to the store) and apply the result.
    async fn refresh_once(&self) -> Result<String, OAuthError> {
        // Snapshot the cached identity so it survives the cache swap
        // below (machine id must stay stable across refreshes).
        let (github_token, machine_id, scopes) = {
            let cache = self.cache.read().await;
            match cache.as_ref() {
                Some(creds) => (
                    creds.github_access_token.clone(),
                    creds.machine_id.clone(),
                    creds.scopes.clone(),
                ),
                None => match self.store.load()? {
                    Some(stored) => (
                        stored.github_access_token,
                        stored.machine_id,
                        stored.scopes,
                    ),
                    None => {
                        return Err(OAuthError::InvalidTokenResponse(
                            "no stored GitHub token to exchange".to_string(),
                        ))
                    }
                },
            }
        };

        let token_url = self.endpoints.read().await.copilot_token_url.clone();
        match exchange_copilot_token(&self.http_client, &github_token, &token_url).await {
            Ok(copilot) => {
                self.apply_exchanged_token(&github_token, copilot, machine_id, scopes)
                    .await
            }
            Err(e @ OAuthError::CopilotEntitlementMissing { .. }) => {
                // Req 3.3: 401/403 → Expired immediately, no retry
                // storm; GitHub token retained on disk for fast
                // re-login.
                tracing::warn!(error = %e, "copilot: exchange rejected; marking session expired");
                let mut state = self.session_state.write().await;
                *state = CopilotSessionState::Expired;
                Err(e)
            }
            Err(e) => Err(e),
        }
    }

    /// Persist + cache a successful exchange result and flip the
    /// session to `Authenticated`.
    async fn apply_exchanged_token(
        &self,
        github_token: &str,
        copilot: CopilotToken,
        machine_id: String,
        scopes: String,
    ) -> Result<String, OAuthError> {
        let credentials = CopilotStoredCredentials {
            github_access_token: github_token.to_string(),
            copilot_token: Some(copilot.token.clone()),
            copilot_expires_at: Some(copilot.expires_at),
            api_base: copilot.api_base().to_string(),
            machine_id,
            sku: copilot.sku.clone(),
            scopes,
            updated_at: now_unix_secs(),
        };
        self.store.save(&credentials)?;

        *self.cache.write().await = Some(CopilotCredentialCache {
            github_access_token: credentials.github_access_token,
            copilot_token: credentials.copilot_token.expect("just set"),
            copilot_expires_at: copilot.expires_at,
            api_base: credentials.api_base,
            machine_id: credentials.machine_id,
            sku: credentials.sku,
            scopes: credentials.scopes,
        });
        let mut state = self.session_state.write().await;
        *state = CopilotSessionState::Authenticated {
            expires_at: copilot.expires_at,
        };
        Ok(copilot.token)
    }

    /// Logout (Req "complete_logout"): cancel any in-flight login,
    /// abort the background refresh task (abort-safe — no task leaks),
    /// clear the in-memory cache, delete the store file, and reset the
    /// session to `Unauthenticated`. Idempotent.
    ///
    /// The login task is aborted AND awaited before the store is
    /// deleted: a task past its exchange await could otherwise run its
    /// synchronous `store.save` after logout's `store.delete` and
    /// resurrect the session.
    #[tracing::instrument(skip(self))]
    pub async fn complete_logout(&self) -> Result<(), OAuthError> {
        // Signal cancellation first (lets the poll task wind down), then
        // abort whatever remains so no task outlives the logout — and
        // make the abort observable before the state is reset.
        self.cancel_login().await;
        self.stop_refresh_loop();
        let login_task = {
            let mut login = self.login.lock().await;
            match login.take() {
                Some(LoginSlot::InFlight(in_flight)) => {
                    in_flight.abort();
                    Some(in_flight.task)
                }
                Some(LoginSlot::Starting) => {
                    // A start_login issuance round-trip is in flight;
                    // it observes the cleared slot on completion (and
                    // aborts its task). Nothing to await here.
                    None
                }
                None => None,
            }
        };
        // Await the aborted login task to completion: once abort() is
        // called, this resolves promptly (JoinError::cancelled is
        // expected and ignored); a task already past its final await
        // still finishes its synchronous store work BEFORE we delete
        // the store below — that ordering is the point.
        if let Some(task) = login_task {
            let _ = task.await;
        }
        self.store.delete()?;
        *self.cache.write().await = None;
        let mut state = self.session_state.write().await;
        *state = CopilotSessionState::Unauthenticated;
        // Login error is stale after logout; the store is now empty so
        // prime the negative-result cache for the hot path.
        *self.last_login_error.lock().await = None;
        self.store_known_empty.store(true, Ordering::Release);
        Ok(())
    }

    /// Spawn the background refresh task (Req 3.2).
    ///
    /// Must be called on an `Arc<Self>`. Calling more than once
    /// replaces the stored handle; the previous task is NOT aborted
    /// automatically — call [`Self::stop_refresh_loop`] first for
    /// deterministic shutdown (logout and the idempotent
    /// [`Self::restart_refresh_loop`] do this).
    ///
    /// The handle is recorded synchronously (no extra spawn hop): an
    /// immediate stop/logout can never miss the handle and leak the
    /// loop.
    pub fn start_refresh_loop(self: Arc<Self>) {
        // Record the self reference first so login tasks spawned from
        // now on can restart this loop after a successful exchange.
        self.set_self_reference();
        let manager = self.clone();
        let handle = tokio::spawn(async move {
            manager.run_refresh_loop().await;
        });
        *self.refresh_task.lock().expect("refresh_task mutex poisoned") = Some(handle);
    }

    /// Idempotent (re)start of the background refresh loop: stop any
    /// existing loop first, then start a fresh one. Used after a
    /// successful login/exchange so a logout-then-re-login cycle keeps
    /// background refresh alive without stacking loops.
    fn restart_refresh_loop(self: &Arc<Self>) {
        self.stop_refresh_loop();
        self.clone().start_refresh_loop();
    }

    /// Abort the background refresh task if one is running.
    pub fn stop_refresh_loop(&self) {
        let mut slot = self.refresh_task.lock().expect("refresh_task mutex poisoned");
        if let Some(handle) = slot.take() {
            handle.abort();
        }
    }

    /// Body of the background refresh task (Req 3.2).
    ///
    /// Ticks every [`REFRESH_TICK_INTERVAL`] (60 s); when the cached
    /// Copilot token is within [`REFRESH_WINDOW_SECS`] (5 min) of
    /// expiry, re-exchanges with the shared
    /// 2 s/4 s/8 s × [`BACKOFF_MAX_ATTEMPTS`] backoff on transient
    /// failures. Exhausting the budget lands in
    /// [`CopilotSessionState::Error`] (recoverable on the next tick);
    /// 401/403 lands in [`CopilotSessionState::Expired`] immediately
    /// (Req 3.3) with the GitHub token retained.
    async fn run_refresh_loop(self: Arc<Self>) {
        let mut interval = tokio::time::interval(REFRESH_TICK_INTERVAL);
        // Skip the immediate first tick: wait a full cadence before the
        // first check so startup I/O doesn't race disk init (mirrors
        // the OpenAI manager).
        interval.tick().await;

        loop {
            interval.tick().await;

            let now = now_unix_secs();
            let (needs_refresh, github_token) = {
                let cache = self.cache.read().await;
                match cache.as_ref() {
                    Some(creds) => (
                        creds
                            .copilot_expires_at
                            .saturating_sub(now)
                            <= REFRESH_WINDOW_SECS,
                        creds.github_access_token.clone(),
                    ),
                    None => (false, String::new()),
                }
            };
            if !needs_refresh {
                continue;
            }

            match self.attempt_exchange_with_backoff(&github_token).await {
                Ok(copilot) => {
                    if let Err(e) = self
                        .apply_exchanged_token(
                            &github_token,
                            copilot,
                            // machine_id/scopes come from the store via
                            // refresh_once's sibling path; here we
                            // preserve the cached identity.
                            self.cached_machine_id_or_new().await,
                            self.cached_scopes_or_default().await,
                        )
                        .await
                    {
                        tracing::warn!(
                            error = %e,
                            "copilot refresh loop: failed to persist exchanged token"
                        );
                        let mut state = self.session_state.write().await;
                        *state = CopilotSessionState::Error;
                    }
                }
                Err(e) => {
                    tracing::warn!(
                        error = %e,
                        "copilot refresh loop: exchange failed; marking session error"
                    );
                    let mut state = self.session_state.write().await;
                    *state = if matches!(
                        e,
                        OAuthError::CopilotEntitlementMissing { .. }
                    ) {
                        CopilotSessionState::Expired
                    } else {
                        CopilotSessionState::Error
                    };
                }
            }
        }
    }

    /// Exchange with the shared backoff schedule on transient network
    /// failures (Req 3.2): delays `2 s / 4 s / 8 s`, at most
    /// [`BACKOFF_MAX_ATTEMPTS`] attempts. 401/403 and other HTTP/parse
    /// errors short-circuit immediately (Req 3.3 — retrying a terminal
    /// status only storms the endpoint).
    #[tracing::instrument(skip(self, github_token))]
    async fn attempt_exchange_with_backoff(
        &self,
        github_token: &str,
    ) -> Result<CopilotToken, OAuthError> {
        let token_url = self.endpoints.read().await.copilot_token_url.clone();
        let mut last_err: Option<OAuthError> = None;
        for attempt in 0..BACKOFF_MAX_ATTEMPTS {
            match exchange_copilot_token(&self.http_client, github_token, &token_url).await {
                Ok(token) => return Ok(token),
                Err(e @ OAuthError::CopilotEntitlementMissing { .. }) => return Err(e),
                Err(e @ OAuthError::CopilotTokenNetwork(_)) => {
                    last_err = Some(e);
                    tokio::time::sleep(std::time::Duration::from_secs(backoff_delay_secs(
                        attempt,
                    )))
                    .await;
                    continue;
                }
                Err(other) => return Err(other),
            }
        }
        Err(last_err.unwrap_or(OAuthError::CopilotTokenHttp {
            status: 0,
            body: "backoff exhausted".to_string(),
        }))
    }

    /// Cached machine id, or a freshly generated one when absent.
    async fn cached_machine_id_or_new(&self) -> String {
        if let Some(creds) = self.cache.read().await.as_ref() {
            return creds.machine_id.clone();
        }
        self.store
            .load()
            .ok()
            .flatten()
            .map(|c| c.machine_id)
            .unwrap_or_else(super::generate_machine_id)
    }

    /// Cached scopes string, or the Copilot default.
    async fn cached_scopes_or_default(&self) -> String {
        if let Some(creds) = self.cache.read().await.as_ref() {
            return creds.scopes.clone();
        }
        COPILOT_OAUTH_SCOPE.to_string()
    }
}

/// Current wall-clock time as seconds since the Unix epoch, clamped at 0
/// (same helper as the OpenAI manager).
fn now_unix_secs() -> u64 {
    SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .map(|d| d.as_secs())
        .unwrap_or(0)
}

#[cfg(test)]
mod tests {
use super::*;
use crate::oauth::github_copilot::test_support::{now_plus, test_client};
use crate::oauth::lifecycle::{
backoff_delay_secs, BACKOFF_MAX_ATTEMPTS, REFRESH_TICK_INTERVAL, REFRESH_WINDOW_SECS,
};
use proptest::prelude::*;
use wiremock::matchers::{method, path};
use wiremock::{Mock, MockServer, ResponseTemplate};

// ---- test scaffolding -------------------------------------------------

fn unique_store() -> CopilotTokenStore {
crate::oauth::github_copilot::test_support::unique_store("mgr_copilot_tokens")
}

fn endpoints(server: &MockServer) -> CopilotEndpoints {
        CopilotEndpoints {
            device_flow: DeviceFlowEndpoints {
                device_code_url: format!("{}/login/device/code", server.uri()),
                token_url: format!("{}/login/oauth/access_token", server.uri()),
            },
            copilot_token_url: format!("{}/copilot_internal/v2/token", server.uri()),
        }
    }

    fn device_grant_body() -> serde_json::Value {
        serde_json::json!({
            "device_code": "dc-test-1",
            "user_code": "ABCD-1234",
            "verification_uri": "https://github.com/login/device",
            "expires_in": 900,
            "interval": 0
        })
    }

    fn github_token_body() -> serde_json::Value {
        serde_json::json!({
            "access_token": "gho-test-github-token",
            "token_type": "bearer",
            "scope": "read:user"
        })
    }

    /// Exchange response with an absolute `expires_at`. Server-side
    /// short expiry values keep timing assertions independent of the
    /// shared lifecycle constants, per the task rules.
    fn copilot_token_body(expires_at: u64) -> serde_json::Value {
        serde_json::json!({
            "token": "tid=test-copilot-bearer",
            "expires_at": expires_at,
            "refresh_in": 1500,
            "endpoints": { "api": "https://api.example-copilot.test" },
"sku": "copilot_pro"
})
}

async fn mount_device_grant(server: &MockServer) {
        Mock::given(method("POST"))
            .and(path("/login/device/code"))
            .respond_with(ResponseTemplate::new(200).set_body_json(device_grant_body()))
            .expect(1)
            .mount(server)
            .await;
    }

    async fn mount_immediate_github_token(server: &MockServer) {
        Mock::given(method("POST"))
            .and(path("/login/oauth/access_token"))
            .respond_with(ResponseTemplate::new(200).set_body_json(github_token_body()))
            .expect(1)
            .mount(server)
            .await;
    }

    fn copilot_exchange_mock(expires_at: u64) -> Mock {
        Mock::given(method("GET"))
            .and(path("/copilot_internal/v2/token"))
            .respond_with(ResponseTemplate::new(200).set_body_json(copilot_token_body(
                expires_at,
            )))
            .up_to_n_times(1)
    }

    /// Drive a full successful login (device grant + immediate poll
    /// success + exchange) and return the manager once the session is
    /// authenticated.
    async fn logged_in_manager(
        server: &MockServer,
        copilot_expires_at: u64,
    ) -> CopilotTokenManager {
        mount_device_grant(server).await;
        mount_immediate_github_token(server).await;
        copilot_exchange_mock(copilot_expires_at)
            .expect(1)
            .mount(server)
            .await;

        let manager = CopilotTokenManager::with_endpoints(
            unique_store(),
            test_client(),
            endpoints(server),
        );
        manager.start_login().await.expect("login should start");

        // The login task flips the state only after the exchange (Req
        // 2.4); poll until authenticated (bounded).
        wait_for_state(&manager, CopilotSessionState::Authenticated { expires_at: 0 }, true).await;
        manager
    }

    /// Poll the session state until it matches (or deliberately fails to
    /// match) the expected variant shape. `expect_match = false` waits
    /// for the state to leave `Pending` and asserts it never became the
    /// expected variant.
    async fn wait_for_state(
        manager: &CopilotTokenManager,
        expected: CopilotSessionState,
        expect_match: bool,
    ) {
        let matches = |state: &CopilotSessionState| match (state, &expected) {
            (CopilotSessionState::Authenticated { .. }, CopilotSessionState::Authenticated { .. }) => true,
            (a, b) => a == b,
        };
        for _ in 0..300 {
            let state = manager.session_state().await;
            if expect_match {
                if matches(&state) {
                    return;
                }
            } else if !matches(&state) && state != CopilotSessionState::Pending {
                return;
            }
            tokio::time::sleep(std::time::Duration::from_millis(10)).await;
        }
        panic!(
            "session state did not settle as expected (expected_match={expect_match}, wanted ~{expected:?}, got {:?})",
            manager.session_state().await
        );
    }

    // ---- Req 2.4: exchange before authenticated --------------------------

    /// After device-flow success the Copilot exchange must happen BEFORE
    /// the session is reported authenticated, and the machine id is
    /// generated + persisted inside the encrypted store on first login.
    #[tokio::test]
    async fn login_performs_copilot_exchange_before_authenticated_and_persists_machine_id() {
        let server = MockServer::start().await;
        let manager = logged_in_manager(&server, now_plus(3600)).await;

        server.verify().await; // exactly one exchange call happened

        // Store now holds the full credential incl. a UUID-shaped
        // machine id generated at first login.
        let stored = manager.store.load().unwrap().expect("credentials stored");
        assert_eq!(stored.github_access_token, "gho-test-github-token");
        assert_eq!(stored.copilot_token.as_deref(), Some("tid=test-copilot-bearer"));
        assert_eq!(stored.copilot_expires_at, Some(now_plus(3600)));
        assert_eq!(stored.api_base, "https://api.example-copilot.test");
        assert_eq!(stored.scopes, "read:user");
        let machine_id = stored.machine_id;
        assert_eq!(
            machine_id.split('-').map(|g| g.len()).collect::<Vec<_>>(),
            vec![8, 4, 4, 4, 12],
            "machine id must be UUID-shaped"
        );

        // The same machine id and api base are served to dispatch.
        assert_eq!(manager.machine_id().await.as_deref(), Some(machine_id.as_str()));
        assert_eq!(
            manager.api_base().await.as_deref(),
            Some("https://api.example-copilot.test")
        );
    }

    /// A failed post-auth exchange must NOT flip the session to
    /// Authenticated (Req 2.4: exchange precedes the report).
    #[tokio::test]
    async fn failed_post_auth_exchange_leaves_session_not_authenticated() {
        let server = MockServer::start().await;
        mount_device_grant(&server).await;
        mount_immediate_github_token(&server).await;
        Mock::given(method("GET"))
            .and(path("/copilot_internal/v2/token"))
            .respond_with(ResponseTemplate::new(500).set_body_string("boom"))
            .expect(1)
            .mount(&server)
            .await;

        let manager = CopilotTokenManager::with_endpoints(
            unique_store(),
            test_client(),
            endpoints(&server),
        );
        manager.start_login().await.expect("login should start");

        // Wait for the login task to settle somewhere that is NOT
        // Authenticated.
        for _ in 0..300 {
            match manager.session_state().await {
                CopilotSessionState::Authenticated { .. } => {
                    panic!("session must not be authenticated when the exchange fails")
                }
                CopilotSessionState::Pending => {}
                _ => break,
            }
            tokio::time::sleep(std::time::Duration::from_millis(10)).await;
        }
        server.verify().await;
    }

    // ---- Req 2.6: in-flight login dedupe ---------------------------------

    /// A second `start_login` while a login is in flight returns the
    /// SAME verification URL + user code and does not request a second
    /// device code.
    #[tokio::test]
    async fn second_start_login_while_in_flight_returns_same_prompt() {
        let server = MockServer::start().await;
        mount_device_grant(&server).await;
        // Keep the poll pending forever (0 s interval, long code life)
        // so the login stays in flight across the dedupe call.
        Mock::given(method("POST"))
            .and(path("/login/oauth/access_token"))
            .respond_with(ResponseTemplate::new(200).set_body_json(
                serde_json::json!({"error": "authorization_pending"}),
            ))
            .mount(&server)
            .await;

        let manager = CopilotTokenManager::with_endpoints(
            unique_store(),
            test_client(),
            endpoints(&server),
        );

        let prompt_first = match manager.start_login().await.expect("first login") {
            LoginStart::Started(p) => p,
            LoginStart::InFlight(_) => panic!("first call must start a fresh flow"),
        };

        let prompt_second = match manager.start_login().await.expect("second login") {
            LoginStart::InFlight(p) => p,
            LoginStart::Started(_) => panic!("second call must dedupe, not start a new flow"),
        };

        assert_eq!(prompt_second, prompt_first);
        assert_eq!(prompt_second.user_code, "ABCD-1234");
        assert_eq!(
            prompt_second.verification_uri,
            "https://github.com/login/device"
        );

        // Exactly ONE device-code request despite two start_login calls.
        let grant_requests = server
            .received_requests()
            .await
            .unwrap()
            .into_iter()
            .filter(|r| r.url.path() == "/login/device/code")
            .count();
        assert_eq!(
            grant_requests, 1,
            "in-flight dedupe must not issue a second device code"
        );

        manager.cancel_login().await;
    }

    // ---- Req 3.4: inline refresh-once on get_copilot_token ----------------

    /// When the cached token has just expired, `get_copilot_token`
    /// refreshes-and-retries the exchange ONCE inline before failing
    /// (here: the retry succeeds, so a token comes back and the session
    /// returns to Authenticated with the far-future expiry; the next
    /// call is served from cache without another exchange).
    #[tokio::test]
    async fn get_copilot_token_refreshes_inline_once_when_cache_just_expired() {
        let server = MockServer::start().await;
        // Login with a token that expires immediately (now + 0 → the
        // cache predicate `expires_at > now` reads it as expired).
        let manager = logged_in_manager(&server, now_plus(0)).await;

        // Next exchange succeeds with a far-future token.
        copilot_exchange_mock(now_plus(3600))
            .expect(1)
            .mount(&server)
            .await;

        // Cached token is now expired → inline refresh-once (Req 3.4).
        let token = manager
            .get_copilot_token()
            .await
            .expect("inline refresh should yield a token");
        assert_eq!(token, "tid=test-copilot-bearer");

        // Session back to Authenticated with the far-future expiry.
        match manager.session_state().await {
            CopilotSessionState::Authenticated { expires_at } => {
                assert!(expires_at > now_unix_secs());
            }
            other => panic!("expected Authenticated, got {other:?}"),
        }

        // Second call is served from the cache — no extra exchange
        // (the expect(1) on the mock enforces this at verify time).
        let again = manager.get_copilot_token().await.expect("cached token");
        assert_eq!(again, "tid=test-copilot-bearer");
        server.verify().await;
    }

    // ---- Req 3.2 constants: refresh-before-expiry window -------------------

    /// The background loop consumes the shared lifecycle constants; this
    /// pins the documented policy (60 s tick, 5-min window, 3 attempts)
    /// and the window predicate arithmetic the loop performs.
    #[test]
    fn refresh_before_expiry_policy_constants_are_pinned() {
        assert_eq!(REFRESH_TICK_INTERVAL, std::time::Duration::from_secs(60));
        assert_eq!(REFRESH_WINDOW_SECS, 300);
        assert_eq!(BACKOFF_MAX_ATTEMPTS, 3);

        // The loop refreshes when expires_at - now <= window (saturating
        // subtraction: already-expired tokens are trivially due).
        let now = 1_000_000u64;
        let due_inside = now + REFRESH_WINDOW_SECS;
        let due_expired = now.saturating_sub(60);
        let not_due = now + REFRESH_WINDOW_SECS + 1;
        assert!(due_inside.saturating_sub(now) <= REFRESH_WINDOW_SECS);
        assert!(due_expired.saturating_sub(now) <= REFRESH_WINDOW_SECS);
        assert!(!(not_due.saturating_sub(now) <= REFRESH_WINDOW_SECS));
    }

    /// A non-transient exchange error (500) inside the inline path
    /// short-circuits: exactly one request, no retry storm.
    #[tokio::test]
    async fn non_transient_exchange_error_short_circuits_without_retry() {
        let server = MockServer::start().await;
        let manager = logged_in_manager(&server, now_plus(0)).await;

        Mock::given(method("GET"))
            .and(path("/copilot_internal/v2/token"))
            .respond_with(ResponseTemplate::new(500).set_body_string("boom"))
            .expect(1)
            .mount(&server)
            .await;

        let err = manager
            .get_copilot_token()
            .await
            .expect_err("exchange failure must surface");
        assert!(matches!(err, OAuthError::CopilotTokenHttp { status: 500, .. }));
        // Exactly one exchange request — no retry for a terminal 5xx.
        server.verify().await;
    }

    // ---- Req 3.3: 401/403 → Expired, GitHub token retained ----------------

    #[tokio::test]
    async fn exchange_401_marks_session_expired_immediately_and_keeps_github_token() {
        let server = MockServer::start().await;
        let manager = logged_in_manager(&server, now_plus(0)).await;

        Mock::given(method("GET"))
            .and(path("/copilot_internal/v2/token"))
            .respond_with(ResponseTemplate::new(401).set_body_string("no seat"))
            .expect(1)
            .mount(&server)
            .await;

        let err = manager
            .get_copilot_token()
            .await
            .expect_err("401 must fail the inline path");
        assert!(matches!(
            err,
            OAuthError::CopilotEntitlementMissing { status: 401 }
        ));

        assert_eq!(manager.session_state().await, CopilotSessionState::Expired);
        // Exactly one exchange request — 401 short-circuits, no retry storm.
        server.verify().await;

        // GitHub token retained on disk for fast re-login (Req 3.3).
        let stored = manager.store.load().unwrap().expect("credentials retained");
        assert_eq!(stored.github_access_token, "gho-test-github-token");
    }

    #[tokio::test]
    async fn exchange_403_marks_session_expired_immediately() {
        let server = MockServer::start().await;
        let manager = logged_in_manager(&server, now_plus(0)).await;

        Mock::given(method("GET"))
            .and(path("/copilot_internal/v2/token"))
            .respond_with(ResponseTemplate::new(403).set_body_string("forbidden"))
            .expect(1)
            .mount(&server)
            .await;

        assert!(matches!(
            manager.get_copilot_token().await,
            Err(OAuthError::CopilotEntitlementMissing { status: 403 })
        ));
        assert_eq!(manager.session_state().await, CopilotSessionState::Expired);
        server.verify().await;
    }

    // ---- load_existing_session ---------------------------------------------

    #[tokio::test]
    async fn load_existing_session_maps_store_states_to_session_states() {
        // Valid Copilot token → Authenticated (+ cache serves it with
        // zero requests and zero disk round-trips).
        {
            let store = unique_store();
            store
                .save(&CopilotStoredCredentials {
                    github_access_token: "gho-live".into(),
                    copilot_token: Some("tid=live".into()),
                    copilot_expires_at: Some(now_plus(3600)),
                    api_base: "https://api.example-copilot.test".into(),
                    machine_id: "6f9619ff-8b86-d011-b42d-00cf4fc964ff".into(),
                    sku: Some("copilot_pro".into()),
                    scopes: "read:user".into(),
                    updated_at: now_unix_secs(),
                })
                .unwrap();
            let server = MockServer::start().await;
            let manager =
                CopilotTokenManager::with_endpoints(store, test_client(), endpoints(&server));
            manager.load_existing_session().await.unwrap();
            assert!(matches!(
                manager.session_state().await,
                CopilotSessionState::Authenticated { .. }
            ));
            // Hot path serves from cache without touching the mock server.
            let token = manager.get_copilot_token().await.unwrap();
            assert_eq!(token, "tid=live");
            assert_eq!(server.received_requests().await.unwrap().len(), 0);
        }

        // Expired/absent Copilot token but GitHub token present →
        // Expired (fast re-login keeps the GitHub token).
        {
            let store = unique_store();
            store
                .save(&CopilotStoredCredentials {
                    github_access_token: "gho-live".into(),
                    copilot_token: None,
                    copilot_expires_at: None,
                    api_base: "https://api.example-copilot.test".into(),
                    machine_id: "6f9619ff-8b86-d011-b42d-00cf4fc964ff".into(),
                    sku: None,
                    scopes: "read:user".into(),
                    updated_at: now_unix_secs(),
                })
                .unwrap();
            let server = MockServer::start().await;
            let manager =
                CopilotTokenManager::with_endpoints(store, test_client(), endpoints(&server));
            manager.load_existing_session().await.unwrap();
            assert_eq!(manager.session_state().await, CopilotSessionState::Expired);
        }

        // No file → Unauthenticated.
        {
            let store = unique_store();
            let server = MockServer::start().await;
            let manager =
                CopilotTokenManager::with_endpoints(store, test_client(), endpoints(&server));
            manager.load_existing_session().await.unwrap();
            assert_eq!(manager.session_state().await, CopilotSessionState::Unauthenticated);
        }
    }

    // ---- logout clears everything ------------------------------------------

    #[tokio::test]
    async fn complete_logout_clears_cache_store_and_tasks() {
        let server = MockServer::start().await;
        let manager = logged_in_manager(&server, now_plus(3600)).await;

        // A refresh task is running; logout must abort it (abort-safe,
        // no leak) and delete the store file + in-memory cache.
        let cloned = Arc::new(manager);
        cloned.clone().start_refresh_loop();
        // Give the handle-recording spawn a moment.
        tokio::time::sleep(std::time::Duration::from_millis(50)).await;

        cloned.complete_logout().await.unwrap();

        assert_eq!(
            cloned.session_state().await,
            CopilotSessionState::Unauthenticated
        );
        assert!(
            !cloned.store.file_path().exists(),
            "store file must be deleted on logout"
        );
        assert!(cloned.store.load().unwrap().is_none());
        // In-memory cache cleared.
        assert!(cloned.cache.read().await.is_none());
        // get_copilot_token fails fast on the Unauthenticated state.
        assert!(cloned.get_copilot_token().await.is_err());
        // Logout is idempotent.
        cloned.complete_logout().await.unwrap();
    }

    /// A cancelled login returns the session to Unauthenticated (no
    /// stored credential) without leaking the poll task.
    #[tokio::test]
    async fn cancel_login_restores_unauthenticated_state() {
        let server = MockServer::start().await;
        mount_device_grant(&server).await;
        Mock::given(method("POST"))
            .and(path("/login/oauth/access_token"))
            .respond_with(ResponseTemplate::new(200).set_body_json(
                serde_json::json!({"error": "authorization_pending"}),
            ))
            .mount(&server)
            .await;

        let manager = CopilotTokenManager::with_endpoints(
            unique_store(),
            test_client(),
            endpoints(&server),
        );
        manager.start_login().await.expect("login should start");
        assert_eq!(manager.session_state().await, CopilotSessionState::Pending);

        manager.cancel_login().await;

        for _ in 0..300 {
            if manager.session_state().await == CopilotSessionState::Unauthenticated {
                return;
            }
            tokio::time::sleep(std::time::Duration::from_millis(10)).await;
        }
        panic!("cancelled login must restore Unauthenticated");
    }

    // ---- backoff schedule property test (shared formula) -------------------

    // Feature: github-copilot-oauth-provider, Req 3.2 (mirrors the
    // openai-oauth-login Property 8 test):
    //
    // ∀ attempt ∈ {0, 1, 2}. backoff_delay_secs(attempt) == 2^(attempt + 1)
    //
    // The manager consumes the same shared lifecycle constants as the
    // OpenAI manager, so this pins the Copilot-visible schedule
    // (2 s, 4 s, 8 s; max 3 attempts) independently.
    proptest! {
        #![proptest_config(ProptestConfig {
            cases: 16,
            .. ProptestConfig::default()
        })]

        #[test]
        fn prop_copilot_backoff_schedule(attempt in 0u32..=2u32) {
            let expected = 2u64.pow(attempt + 1);
            let actual = backoff_delay_secs(attempt);
            prop_assert_eq!(actual, expected);

            match attempt {
                0 => prop_assert_eq!(actual, 2),
                1 => prop_assert_eq!(actual, 4),
                2 => prop_assert_eq!(actual, 8),
                _ => unreachable!(),
            }
        }
    }

    // ---- production endpoint defaults --------------------------------------

    #[test]
    fn production_endpoints_use_github_and_copilot_constants() {
        let endpoints = CopilotEndpoints::production();
        assert_eq!(
            endpoints.device_flow.device_code_url,
            "https://github.com/login/device/code"
        );
        assert_eq!(
            endpoints.device_flow.token_url,
            "https://github.com/login/oauth/access_token"
        );
        assert_eq!(
            endpoints.copilot_token_url,
            "https://api.github.com/copilot_internal/v2/token"
        );
    }

// ---- security: no token material in Debug output ------------------------

#[test]
fn debug_output_of_cache_contains_no_token_material() {
let cache = CopilotCredentialCache {
github_access_token: "gho-secret".into(),
copilot_token: "tid=secret".into(),
copilot_expires_at: 1770000000,
api_base: "https://api.example-copilot.test".into(),
machine_id: "6f9619ff-8b86-d011-b42d-00cf4fc964ff".into(),
sku: Some("copilot_pro".into()),
scopes: "read:user".into(),
};
let rendered = format!("{cache:?}");
assert!(!rendered.contains("gho-secret"));
assert!(!rendered.contains("tid=secret"));
assert!(rendered.contains("<redacted>"));
}

// ---- M1: terminal device-flow errors surface via last_login_error -------

/// A terminal `access_denied` poll outcome resets the session but the
/// specific denial message is PERSISTED and surfaced through
/// `last_login_error()` (Req 2.3) instead of being dropped; a
/// subsequent successful login clears it.
#[tokio::test]
async fn denied_login_surfaces_specific_error_and_success_clears_it() {
let server = MockServer::start().await;
// Device grant serves BOTH flows (denied + retry).
Mock::given(method("POST"))
.and(path("/login/device/code"))
.respond_with(ResponseTemplate::new(200).set_body_json(device_grant_body()))
.expect(2)
.mount(&server)
.await;
// First poll answers with the terminal denial…
Mock::given(method("POST"))
.and(path("/login/oauth/access_token"))
.respond_with(ResponseTemplate::new(200).set_body_json(
serde_json::json!({"error": "access_denied"}),
))
.up_to_n_times(1)
.expect(1)
.mount(&server)
.await;
// …the retry flow's poll succeeds…
Mock::given(method("POST"))
.and(path("/login/oauth/access_token"))
.respond_with(ResponseTemplate::new(200).set_body_json(github_token_body()))
.expect(1)
.mount(&server)
.await;
// …and the retry's exchange authenticates.
copilot_exchange_mock(now_plus(3600))
.expect(1)
.mount(&server)
.await;

let manager = CopilotTokenManager::with_endpoints(
unique_store(),
test_client(),
endpoints(&server),
);
manager.start_login().await.expect("login should start");

// Terminal denial settles to Unauthenticated…
wait_for_state(&manager, CopilotSessionState::Unauthenticated, true).await;
// …with the SPECIFIC outcome surfaced (not a generic message).
let error = manager
.last_login_error()
.await
.expect("denial must surface a login error");
assert!(
error.to_lowercase().contains("denied"),
"surfaced error must name the denial, got: {error}"
);

// A subsequent successful login clears the surfaced error.
manager.start_login().await.expect("second login should start");
wait_for_state(&manager, CopilotSessionState::Authenticated { expires_at: 0 }, true).await;
assert!(
manager.last_login_error().await.is_none(),
"successful login clears the last-login error"
);
server.verify().await;
}

/// `expired_token` surfaces its own actionable message (distinct from
/// the denial wording).
#[tokio::test]
async fn expired_device_code_surfaces_expiry_error() {
let server = MockServer::start().await;
mount_device_grant(&server).await;
Mock::given(method("POST"))
.and(path("/login/oauth/access_token"))
.respond_with(ResponseTemplate::new(200).set_body_json(
serde_json::json!({"error": "expired_token"}),
))
.expect(1)
.mount(&server)
.await;

let manager = CopilotTokenManager::with_endpoints(
unique_store(),
test_client(),
endpoints(&server),
);
manager.start_login().await.expect("login should start");
wait_for_state(&manager, CopilotSessionState::Unauthenticated, true).await;

let error = manager
.last_login_error()
.await
.expect("expiry must surface a login error");
assert!(
error.to_lowercase().contains("expired"),
"surfaced error must name the expiry, got: {error}"
);
server.verify().await;
}

/// Cancellation is NOT a failure: cancel leaves any previously
/// surfaced error untouched and never sets one itself.
#[tokio::test]
async fn cancelled_login_does_not_set_login_error() {
let server = MockServer::start().await;
mount_device_grant(&server).await;
Mock::given(method("POST"))
.and(path("/login/oauth/access_token"))
.respond_with(ResponseTemplate::new(200).set_body_json(
serde_json::json!({"error": "authorization_pending"}),
))
.mount(&server)
.await;

let manager = CopilotTokenManager::with_endpoints(
unique_store(),
test_client(),
endpoints(&server),
);
manager.start_login().await.expect("login should start");
manager.cancel_login().await;
wait_for_state(&manager, CopilotSessionState::Unauthenticated, true).await;

assert!(
manager.last_login_error().await.is_none(),
"deliberate cancellation must not surface a login error"
);
}

/// Logout clears a previously surfaced login error.
#[tokio::test]
async fn logout_clears_login_error() {
let server = MockServer::start().await;
mount_device_grant(&server).await;
Mock::given(method("POST"))
.and(path("/login/oauth/access_token"))
.respond_with(ResponseTemplate::new(200).set_body_json(
serde_json::json!({"error": "access_denied"}),
))
.expect(1)
.mount(&server)
.await;

let manager = CopilotTokenManager::with_endpoints(
unique_store(),
test_client(),
endpoints(&server),
);
manager.start_login().await.expect("login should start");
wait_for_state(&manager, CopilotSessionState::Unauthenticated, true).await;
assert!(manager.last_login_error().await.is_some());

manager.complete_logout().await.expect("logout");
assert!(
manager.last_login_error().await.is_none(),
"logout clears the surfaced login error"
);
}

// ---- M3: force_refresh bypasses the unexpired-cache short-circuit -------

/// A revoked-but-unexpired cached token cannot be displaced by
/// `get_copilot_token` (cache hit), but `force_refresh` invalidates it
/// and exchanges immediately over wiremock, replacing the cache with
/// the fresh bearer (router 401-rescue hook).
#[tokio::test]
async fn force_refresh_replaces_unexpired_bogus_cached_token() {
let server = MockServer::start().await;
// Seed: login succeeds with an unexpired token…
let manager = logged_in_manager(&server, now_plus(3600)).await;
// …but the cached bearer is bogus (revoked upstream): a plain
// get_copilot_token serves it straight from the cache.
let bogus = manager.get_copilot_token().await.expect("cached token");
assert_eq!(bogus, "tid=test-copilot-bearer");

// force_refresh must exchange even though the cache is unexpired.
copilot_exchange_mock(now_plus(3600))
.expect(1)
.mount(&server)
.await;
let refreshed = manager
.force_refresh()
.await
.expect("force_refresh exchanges");
assert_eq!(refreshed, "tid=test-copilot-bearer");
server.verify().await; // exactly ONE exchange despite the live cache

// The cache was replaced (fresh expiry observable through state).
match manager.session_state().await {
CopilotSessionState::Authenticated { expires_at } => {
assert!(expires_at > now_unix_secs());
}
other => panic!("expected Authenticated, got {other:?}"),
}
}

// ---- M4: single-flight around the inline refresh -------------------------

/// Two concurrent `get_copilot_token` calls at cache-expiry coalesce
/// onto ONE exchange (the wiremock mock is mounted with expect(1) and
/// verified) — no burst, no spurious GitHub 403 → Expired.
#[tokio::test]
async fn concurrent_get_copilot_token_coalesces_to_single_exchange() {
let server = MockServer::start().await;
let manager = logged_in_manager(&server, now_plus(0)).await;

// Exactly ONE exchange serves both concurrent callers.
copilot_exchange_mock(now_plus(3600))
.expect(1)
.mount(&server)
.await;

let (a, b) = tokio::join!(manager.get_copilot_token(), manager.get_copilot_token());
assert_eq!(a.expect("first caller refreshed"), "tid=test-copilot-bearer");
assert_eq!(b.expect("second caller coalesced"), "tid=test-copilot-bearer");

server.verify().await; // exactly one exchange request
}

// ---- M6: half-present token/expiry pair must not panic -------------------

/// A store file with `copilot_expires_at` set but `copilot_token`
/// null maps to Expired at load — no startup panic.
#[tokio::test]
async fn load_with_expires_at_but_null_token_maps_to_expired() {
let store = unique_store();
store
.save(&CopilotStoredCredentials {
github_access_token: "gho-live".into(),
copilot_token: None,
copilot_expires_at: Some(now_plus(3600)),
api_base: "https://api.example-copilot.test".into(),
machine_id: "6f9619ff-8b86-d011-b42d-00cf4fc964ff".into(),
sku: None,
scopes: "read:user".into(),
updated_at: now_unix_secs(),
})
.unwrap();

let server = MockServer::start().await;
let manager =
CopilotTokenManager::with_endpoints(store, test_client(), endpoints(&server));
manager.load_existing_session().await.unwrap();
assert_eq!(
manager.session_state().await,
CopilotSessionState::Expired,
"half-present credential pair must map to Expired, not panic"
);
}

/// The mirror case (`copilot_token` set, `copilot_expires_at` null)
/// is equally non-viable → Expired.
#[tokio::test]
async fn load_with_token_but_null_expires_at_maps_to_expired() {
let store = unique_store();
store
.save(&CopilotStoredCredentials {
github_access_token: "gho-live".into(),
copilot_token: Some("tid=orphan".into()),
copilot_expires_at: None,
api_base: "https://api.example-copilot.test".into(),
machine_id: "6f9619ff-8b86-d011-b42d-00cf4fc964ff".into(),
sku: None,
scopes: "read:user".into(),
updated_at: now_unix_secs(),
})
.unwrap();

let server = MockServer::start().await;
let manager =
CopilotTokenManager::with_endpoints(store, test_client(), endpoints(&server));
manager.load_existing_session().await.unwrap();
assert_eq!(manager.session_state().await, CopilotSessionState::Expired);
}

// ---- M8: negative-result cache for machine_id/api_base -------------------

/// With no OAuth session (api-key mode), the first machine_id() load
/// hits the store; subsequent calls are served from the negative
/// cache. A store file written afterwards (login) resets the flag.
#[tokio::test]
async fn machine_id_and_api_base_cache_negative_store_result() {
let server = MockServer::start().await;
let manager = CopilotTokenManager::with_endpoints(
unique_store(),
test_client(),
endpoints(&server),
);

// Empty store: both resolve without credentials, prime the flag.
assert!(manager.machine_id().await.is_none());
assert_eq!(
manager.api_base().await.as_deref(),
Some("https://api.githubcopilot.com"),
"default api base with no stored credential"
);
assert!(
manager.store_known_empty.load(std::sync::atomic::Ordering::Acquire),
"empty load must prime the negative cache"
);

// Simulate an external write to the store (as-if login happened on
// another path): reset semantics are exercised through login/logout;
// here the flag must reflect the cached emptiness until invalidated.
manager.store_known_empty.store(false, std::sync::atomic::Ordering::Release);
// After invalidation the next call re-reads the (still empty) store.
assert!(manager.machine_id().await.is_none());
assert!(
manager.store_known_empty.load(std::sync::atomic::Ordering::Acquire),
"re-read of the still-empty store re-primes the flag"
);
}

// ---- M9: refresh handle recorded synchronously ---------------------------

/// start_refresh_loop records its handle synchronously: an immediate
/// stop_refresh_loop (logout race) observes and aborts the loop with
/// no sleep, no leak.
#[tokio::test]
async fn immediate_stop_after_start_aborts_loop_without_wait() {
let server = MockServer::start().await;
let manager = logged_in_manager(&server, now_plus(3600)).await;

let cloned = Arc::new(manager);
cloned.clone().start_refresh_loop();
// NO sleep: the handle must already be recorded.
cloned.stop_refresh_loop();
assert!(
cloned
.refresh_task
.lock()
.expect("refresh_task mutex")
.is_none(),
"immediate stop must have taken the handle"
);

// Idempotent second stop is a no-op.
cloned.stop_refresh_loop();
}

// ---- M2: refresh loop restarts after logout + re-login -------------------

/// Logout aborts the background refresh loop; a subsequent successful
/// login (exchange) restarts it — tokens keep being background
/// refreshed after re-login, and loops never stack.
#[tokio::test]
async fn relogin_after_logout_restarts_refresh_loop() {
let server = MockServer::start().await;
// Both flows (initial login + post-logout re-login) served by the
// same mounts: wiremock dispatches to the first matching mock, so
// each leg expects exactly two hits.
Mock::given(method("POST"))
.and(path("/login/device/code"))
.respond_with(ResponseTemplate::new(200).set_body_json(device_grant_body()))
.expect(2)
.mount(&server)
.await;
Mock::given(method("POST"))
.and(path("/login/oauth/access_token"))
.respond_with(ResponseTemplate::new(200).set_body_json(github_token_body()))
.expect(2)
.mount(&server)
.await;
Mock::given(method("GET"))
.and(path("/copilot_internal/v2/token"))
.respond_with(ResponseTemplate::new(200).set_body_json(copilot_token_body(
now_plus(3600),
)))
.expect(2)
.mount(&server)
.await;

let manager =
CopilotTokenManager::with_endpoints(unique_store(), test_client(), endpoints(&server));
manager.start_login().await.expect("login should start");
wait_for_state(&manager, CopilotSessionState::Authenticated { expires_at: 0 }, true).await;
let manager = Arc::new(manager);

// Gateway-boot equivalent: the loop is running.
manager.clone().start_refresh_loop();
assert!(
manager
.refresh_task
.lock()
.expect("refresh_task mutex")
.is_some()
);

// Logout aborts it.
manager.complete_logout().await.expect("logout");
assert!(
manager
.refresh_task
.lock()
.expect("refresh_task mutex")
.is_none()
);

// Re-login: after the exchange completes, the loop is alive again.
manager.start_login().await.expect("re-login starts");
wait_for_state(&manager.as_ref(), CopilotSessionState::Authenticated { expires_at: 0 }, true)
.await;
for _ in 0..300 {
if manager
.refresh_task
.lock()
.expect("refresh_task mutex")
.is_some()
{
server.verify().await; // both flows hit each leg exactly twice
return;
}
tokio::time::sleep(std::time::Duration::from_millis(10)).await;
}
panic!("re-login must restart the background refresh loop");
}
}
