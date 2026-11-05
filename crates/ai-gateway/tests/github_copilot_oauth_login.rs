//! Integration tests for the GitHub Copilot OAuth login flow (spec
//! `github-copilot-oauth-provider`, task 12; Req 2.1-2.6, 3.1-3.5).
//!
//! Drives the REAL credential-lifecycle components end to end over
//! wiremock: [`CopilotTokenManager`] (login, in-flight dedupe, cancel,
//! inline refresh, logout), the RFC 8628 device flow, the Copilot token
//! exchange, and the encrypted `CopilotTokenStore` — with every endpoint
//! pointed at the mock server via the manager's injectable
//! `CopilotEndpoints` (no request ever leaves the loopback interface).
//!
//! Isolation follows the store/manager unit-test pattern blessed by the
//! task: a process-wide `AI_GATEWAY_DATA_DIR` temp dir (created once via
//! `OnceLock`, master key generated eagerly) with a unique credential
//! file per test, so parallel tests never share `copilot_tokens.enc` or
//! the developer machine's real master key. This suite never constructs
//! a `GatewayServer`, so no SQLite databases are opened and
//! `common::isolate_databases` has nothing to isolate.
//!
//! Security: token literals (`gho_…`, `tid=…`, device codes) never
//! appear in assertion messages or failure output — comparisons are
//! boolean with static messages.

use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::OnceLock;
use std::time::{Duration, SystemTime};

use ai_gateway::oauth::github_copilot::device_flow::DeviceFlowEndpoints;
use ai_gateway::oauth::github_copilot::manager::{
 CopilotEndpoints, CopilotSessionState, CopilotTokenManager, LoginStart,
};
use ai_gateway::oauth::github_copilot::store::CopilotTokenStore;
use ai_gateway::oauth::OAuthError;
use wiremock::matchers::{method, path};
use wiremock::{Mock, MockServer, ResponseTemplate};

// ---- fixture constants (non-secret assertions never print these) -------

const DEVICE_CODE: &str = "dc-integration-1";
const USER_CODE: &str = "INTG-1234";
const VERIFICATION_URI: &str = "https://github.com/login/device";
const GITHUB_TOKEN: &str = "gho-integration-test-github-token";
const COPILOT_BEARER: &str = "tid=integration-test-copilot-bearer";
const COPILOT_API_BASE: &str = "https://api.example-copilot.test";

const DEVICE_CODE_PATH: &str = "/login/device/code";
const TOKEN_POLL_PATH: &str = "/login/oauth/access_token";
const EXCHANGE_PATH: &str = "/copilot_internal/v2/token";

// ---- isolation scaffolding (store/manager unit-test pattern) ------------

/// Hermetic secure-storage dir shared by the tests in this binary: point
/// `AI_GATEWAY_DATA_DIR` at a process-wide temp dir exactly once, create
/// the master key eagerly (check-then-create semantics make a lazy first
/// save racy), then hand each test a unique credential file inside it so
/// parallel tests never collide on `copilot_tokens.enc`.
fn secure_dir() -> &'static std::path::Path {
 static DIR: OnceLock<std::path::PathBuf> = OnceLock::new();
 DIR.get_or_init(|| {
  let dir = tempfile::tempdir().expect("temp secure dir");
  // Leak the buffer so the path stays valid for the process lifetime;
  // the OS reclaims temp content eventually (throwaway test keys).
  let leaked = Box::leak(Box::new(dir.keep()));
  std::env::set_var("AI_GATEWAY_DATA_DIR", &*leaked);
  let _ = ai_gateway::secrets::load_or_create_master_key();
  leaked.to_path_buf()
 })
}

/// Unique credential-store path for one test.
fn unique_store() -> CopilotTokenStore {
 static COUNTER: AtomicU64 = AtomicU64::new(0);
 let n = COUNTER.fetch_add(1, Ordering::SeqCst);
 CopilotTokenStore::new(secure_dir().join(format!("it_copilot_tokens_{n}.enc")))
}

fn test_client() -> reqwest::Client {
 reqwest::Client::builder().build().unwrap()
}

/// All three endpoint legs pointed at one wiremock server.
fn endpoints(server: &MockServer) -> CopilotEndpoints {
 CopilotEndpoints {
  device_flow: DeviceFlowEndpoints {
   device_code_url: format!("{}{}", server.uri(), DEVICE_CODE_PATH),
   token_url: format!("{}{}", server.uri(), TOKEN_POLL_PATH),
  },
  copilot_token_url: format!("{}{}", server.uri(), EXCHANGE_PATH),
 }
}

fn now_unix_secs() -> u64 {
 SystemTime::now()
  .duration_since(SystemTime::UNIX_EPOCH)
  .map(|d| d.as_secs())
  .unwrap_or(0)
}

fn now_plus(secs: u64) -> u64 {
 now_unix_secs().saturating_add(secs)
}

// ---- wiremock fixtures ---------------------------------------------------

fn device_grant_body(interval: u64, expires_in: u64) -> serde_json::Value {
 serde_json::json!({
  "device_code": DEVICE_CODE,
  "user_code": USER_CODE,
  "verification_uri": VERIFICATION_URI,
  "expires_in": expires_in,
  "interval": interval
 })
}

fn github_token_body() -> serde_json::Value {
 serde_json::json!({
  "access_token": GITHUB_TOKEN,
  "token_type": "bearer",
  "scope": "read:user"
 })
}

/// Exchange response with a server-side absolute `expires_at`; short
/// values drive the inline refresh path without touching the shared
/// lifecycle constants.
fn copilot_token_body(expires_at: u64) -> serde_json::Value {
 serde_json::json!({
  "token": COPILOT_BEARER,
  "expires_at": expires_at,
  "refresh_in": 1500,
  "endpoints": { "api": COPILOT_API_BASE },
  "sku": "copilot_pro"
 })
}

/// Device-code issuance mock, capped at one hit so a later mount on the
/// same path (re-login phases) takes over cleanly.
async fn mount_device_grant(server: &MockServer, interval: u64, expires_in: u64) {
 Mock::given(method("POST"))
  .and(path(DEVICE_CODE_PATH))
  .respond_with(ResponseTemplate::new(200).set_body_json(device_grant_body(
   interval, expires_in,
  )))
  .up_to_n_times(1)
  .expect(1)
  .mount(server)
  .await;
}

/// Poll mock answering the GitHub token immediately (device-flow leg
/// completes on the first poll).
async fn mount_immediate_github_token(server: &MockServer) {
 Mock::given(method("POST"))
  .and(path(TOKEN_POLL_PATH))
  .respond_with(ResponseTemplate::new(200).set_body_json(github_token_body()))
  .up_to_n_times(1)
  .expect(1)
  .mount(server)
  .await;
}

/// Poll mock answering a non-terminal pending payload forever (no hit
/// expectations: the biased cancel `select!` can cancel before the first
/// poll fires, and pending counts are asserted explicitly where they
/// matter).
async fn mount_poll_pending(server: &MockServer) {
 Mock::given(method("POST"))
  .and(path(TOKEN_POLL_PATH))
  .respond_with(ResponseTemplate::new(200).set_body_json(
   serde_json::json!({"error": "authorization_pending"}),
  ))
  .mount(server)
  .await;
}

/// Poll mock answering a terminal error payload exactly once; later
/// polls on the same path fall through to whatever mock is mounted
/// afterwards (re-login phases mount their own success response).
async fn mount_poll_error_once(server: &MockServer, error: &str) {
 Mock::given(method("POST"))
  .and(path(TOKEN_POLL_PATH))
  .respond_with(
   ResponseTemplate::new(200).set_body_json(serde_json::json!({"error": error})),
  )
  .up_to_n_times(1)
  .mount(server)
  .await;
}

/// Successful Copilot exchange mock, capped at one hit so sequential
/// flows (refresh, re-login) mount their own response next.
async fn mount_exchange(server: &MockServer, expires_at: u64) {
 Mock::given(method("GET"))
  .and(path(EXCHANGE_PATH))
  .respond_with(ResponseTemplate::new(200).set_body_json(copilot_token_body(
   expires_at,
  )))
  .up_to_n_times(1)
  .expect(1)
  .mount(server)
  .await;
}

/// Exchange mock answering 401 (entitlement gone), capped at one hit.
async fn mount_exchange_401(server: &MockServer) {
 Mock::given(method("GET"))
  .and(path(EXCHANGE_PATH))
  .respond_with(ResponseTemplate::new(401).set_body_string("no seat"))
  .up_to_n_times(1)
  .expect(1)
  .mount(server)
  .await;
}

/// Number of recorded requests whose path matches.
async fn count_requests(server: &MockServer, wanted: &str) -> usize {
 server
  .received_requests()
  .await
  .unwrap_or_default()
  .into_iter()
  .filter(|r| r.url.path() == wanted)
  .count()
}

// ---- state-wait helpers --------------------------------------------------

/// Poll the session state until it becomes `Authenticated` (any
/// `expires_at`), bounded at ~6 s.
async fn wait_for_authenticated(manager: &CopilotTokenManager) {
 for _ in 0..600 {
  if matches!(
   manager.session_state().await,
   CopilotSessionState::Authenticated { .. }
  ) {
   return;
  }
  tokio::time::sleep(Duration::from_millis(10)).await;
 }
 panic!(
  "session never became authenticated (last state: {:?})",
  manager.session_state().await
 );
}

/// Poll the session state until it equals `expected` exactly, bounded at
/// ~6 s.
async fn wait_for_terminal(manager: &CopilotTokenManager, expected: CopilotSessionState) {
 for _ in 0..600 {
  if manager.session_state().await == expected {
   return;
  }
  tokio::time::sleep(Duration::from_millis(10)).await;
 }
 panic!(
  "session never reached the expected terminal state {:?} (last state: {:?})",
  expected,
  manager.session_state().await
 );
}

// ---- happy path: login → cache → refresh → logout → re-login ------------

/// Full credential lifecycle over the real manager, device flow,
/// exchange, and encrypted store:
///
/// 1. Device code issued → poll succeeds → Copilot exchange runs
///    BEFORE the session reports `Authenticated` (Req 2.4: exactly one
///    exchange request by the time the state flips).
/// 2. Credentials are stored encrypted (file exists, `enc-v1:` blob,
///    not plaintext JSON, no token literals).
/// 3. With a short server-side expiry the cached token is already
///    stale → `get_copilot_token` drives the inline refresh-once path
///    (Req 3.4) and the next call is served from the cache (no extra
///    exchange).
/// 4. Logout wipes cache + file; a re-login performs a fresh flow.
#[tokio::test]
async fn full_lifecycle_login_cache_refresh_logout_relogin() {
 let server = MockServer::start().await;
 let store = unique_store();
 let manager =
  CopilotTokenManager::with_endpoints(store.clone(), test_client(), endpoints(&server));

 // -- login: short server-side expiry so the cache is stale immediately
 mount_device_grant(&server, 0, 900).await;
 mount_immediate_github_token(&server).await;
 mount_exchange(&server, now_plus(0)).await;

 let prompt = match manager.start_login().await.expect("login should start") {
  LoginStart::Started(p) => p,
  LoginStart::InFlight(_) => panic!("first login must start a fresh flow"),
 };
 assert!(prompt.user_code == USER_CODE, "prompt carries the issued user code");
 assert!(
  prompt.verification_uri == VERIFICATION_URI,
  "prompt carries the verification URL"
 );

 // Req 2.4: by the time the session is Authenticated, exactly one
 // exchange request has happened (exchange precedes the report).
 wait_for_authenticated(&manager).await;
 assert_eq!(
  count_requests(&server, EXCHANGE_PATH).await,
  1,
  "authenticated state implies the exchange already ran"
 );

 // Req 3.5: stored encrypted — file exists, enc-v1 blob, no plaintext
 // JSON, no token literals.
 let raw = std::fs::read_to_string(store.file_path()).expect("credential file exists");
 assert!(raw.starts_with("enc-v1:"), "store file must be an enc-v1 blob");
 assert!(
  serde_json::from_str::<serde_json::Value>(&raw).is_err(),
  "store file must not parse as plaintext JSON"
 );

 let stored = store.load().expect("load").expect("credentials stored");
 assert!(
  stored.github_access_token == GITHUB_TOKEN,
  "store holds the GitHub token"
 );
 assert!(
  stored.copilot_token.as_deref() == Some(COPILOT_BEARER),
  "store holds the Copilot bearer"
 );
 assert_eq!(stored.api_base, COPILOT_API_BASE);
 assert!(!stored.machine_id.is_empty(), "machine id generated at login");
 assert_eq!(
  manager.machine_id().await.as_deref(),
  Some(stored.machine_id.as_str()),
  "manager serves the persisted machine id"
 );

 // -- inline refresh (Req 3.4): cached token expired (server-side
 // short expiry) → one re-exchange with a far-future token.
 mount_exchange(&server, now_plus(3600)).await;
 let token = manager
  .get_copilot_token()
  .await
  .expect("inline refresh should yield a token");
 assert!(token == COPILOT_BEARER, "refreshed bearer matches the mock");
 match manager.session_state().await {
  CopilotSessionState::Authenticated { expires_at } => {
   assert!(expires_at > now_unix_secs(), "expiry moved forward after refresh");
  }
  other => panic!("expected Authenticated after refresh, got {:?}", other),
 }

 // -- cached hot path: the second call adds no exchange request (the
 // refresh mock's expect(1) enforces this at verify time).
 let cached = manager.get_copilot_token().await.expect("cached token");
 assert!(cached == COPILOT_BEARER, "cached bearer matches the mock");

 // -- logout: state reset, file gone, token unavailable.
 manager.complete_logout().await.expect("logout");
 assert_eq!(manager.session_state().await, CopilotSessionState::Unauthenticated);
 assert!(
  !store.file_path().exists(),
  "logout must delete the credential file"
 );
 assert!(
  matches!(manager.get_copilot_token().await, Err(OAuthError::InvalidTokenResponse(_))),
  "no token after logout"
 );

 // -- re-login performs a fresh flow end to end.
 mount_device_grant(&server, 0, 900).await;
 mount_immediate_github_token(&server).await;
 mount_exchange(&server, now_plus(3600)).await;
 match manager.start_login().await.expect("re-login should start") {
  LoginStart::Started(_) => {}
  LoginStart::InFlight(_) => panic!("re-login after logout must start a fresh flow"),
 }
 wait_for_authenticated(&manager).await;
 let re_token = manager.get_copilot_token().await.expect("token after re-login");
 assert!(re_token == COPILOT_BEARER, "re-login yields the exchanged bearer");

 server.verify().await; // every leg hit exactly the expected number of times
}

// ---- Req 2.3: denial is terminal, no infinite polling --------------------

/// `access_denied` terminates the login on the first poll: the state
/// settles to `Unauthenticated` (specific terminal outcome, never
/// `Pending` forever), exactly one poll request is made, and a fresh
/// login can be started immediately afterwards.
#[tokio::test]
async fn access_denied_is_terminal_without_infinite_polling() {
 let server = MockServer::start().await;
 let manager = CopilotTokenManager::with_endpoints(
  unique_store(),
  test_client(),
  endpoints(&server),
 );

 mount_device_grant(&server, 1, 900).await;
 mount_poll_error_once(&server, "access_denied").await;

 manager.start_login().await.expect("login should start");
 wait_for_terminal(&manager, CopilotSessionState::Unauthenticated).await;

 // Terminal on the FIRST poll response — no retry loop.
 assert_eq!(
  count_requests(&server, TOKEN_POLL_PATH).await,
  1,
  "access_denied must terminate polling on the first response"
 );
 // No exchange ever ran (authorization never succeeded).
 assert_eq!(
  count_requests(&server, EXCHANGE_PATH).await,
  0,
  "denial must not trigger a Copilot exchange"
 );

 // A fresh login is possible right away (previous flow fully wound down).
 mount_device_grant(&server, 0, 900).await;
 mount_immediate_github_token(&server).await;
 mount_exchange(&server, now_plus(3600)).await;
 match manager.start_login().await.expect("second login should start") {
  LoginStart::Started(_) => {}
  LoginStart::InFlight(_) => panic!("denied login must not linger as in-flight"),
 }
 wait_for_authenticated(&manager).await;
 server.verify().await;
}

// ---- Req 2.3 / 2.5: code expiry terminates polling -----------------------

/// Polling never outlives the device code: with `authorization_pending`
/// forever and a 1 s code lifetime, the poll stops at the lifetime
/// horizon (bounded request count, stable across a follow-up window).
#[tokio::test]
async fn polling_stops_at_device_code_lifetime() {
 let server = MockServer::start().await;
 let manager = CopilotTokenManager::with_endpoints(
  unique_store(),
  test_client(),
  endpoints(&server),
 );

 // 1 s interval, 1 s lifetime: first poll pending, then the lifetime
 // guard fires without another request.
 mount_device_grant(&server, 1, 1).await;
 mount_poll_pending(&server).await;

 manager.start_login().await.expect("login should start");
 wait_for_terminal(&manager, CopilotSessionState::Unauthenticated).await;

 let after_settle = count_requests(&server, TOKEN_POLL_PATH).await;
 assert!(
  after_settle <= 3,
  "poll count must stay bounded by the code lifetime"
 );

 // No retry beyond the code lifetime: the count is stable across a
 // window longer than the lifetime + poll interval.
 tokio::time::sleep(Duration::from_millis(1500)).await;
 assert_eq!(
  count_requests(&server, TOKEN_POLL_PATH).await,
  after_settle,
  "no polls after the device code lifetime elapsed"
 );
}

/// GitHub's terminal `expired_token` poll error ends the flow on the
/// first response (Req 2.3: specific, terminal — no further retries).
#[tokio::test]
async fn expired_token_poll_error_is_terminal() {
 let server = MockServer::start().await;
 let manager = CopilotTokenManager::with_endpoints(
  unique_store(),
  test_client(),
  endpoints(&server),
 );

 mount_device_grant(&server, 1, 900).await;
 mount_poll_error_once(&server, "expired_token").await;

 manager.start_login().await.expect("login should start");
 wait_for_terminal(&manager, CopilotSessionState::Unauthenticated).await;

 assert_eq!(
  count_requests(&server, TOKEN_POLL_PATH).await,
  1,
  "expired_token must terminate polling on the first response"
 );
 assert_eq!(
  count_requests(&server, EXCHANGE_PATH).await,
  0,
  "expired code must not trigger a Copilot exchange"
 );
}

// ---- Req 3.3: 401 short-circuit, GitHub token retained, re-login ---------

/// An exchange 401 marks the session `Expired` immediately with exactly
/// ONE request (no retry storm), the GitHub token stays on disk for a
/// fast re-login, and that re-login succeeds end to end.
#[tokio::test]
async fn exchange_401_short_circuits_to_expired_and_keeps_github_token() {
 let server = MockServer::start().await;
 let store = unique_store();
 let manager =
  CopilotTokenManager::with_endpoints(store.clone(), test_client(), endpoints(&server));

 // Login succeeds first (GitHub token persisted, session authenticated
 // with a short server-side expiry so the next get drives a refresh).
 mount_device_grant(&server, 0, 900).await;
 mount_immediate_github_token(&server).await;
 mount_exchange(&server, now_plus(0)).await;
 manager.start_login().await.expect("login should start");
 wait_for_authenticated(&manager).await;

 // The next exchange is a 401 (entitlement gone).
 mount_exchange_401(&server).await;
 let err = manager
  .get_copilot_token()
  .await
  .expect_err("401 must fail the inline refresh");
 assert!(
  matches!(err, OAuthError::CopilotEntitlementMissing { status: 401 }),
  "401 maps to the entitlement error"
 );
 assert_eq!(
  manager.session_state().await,
  CopilotSessionState::Expired,
  "401 marks the session Expired immediately"
 );

 // No retry storm: exactly one 401 exchange request, and the count is
 // stable across a window far longer than any backoff hop.
 assert_eq!(
  count_requests(&server, EXCHANGE_PATH).await,
  2,
  "exactly one login exchange + one 401 exchange, nothing more"
 );
 tokio::time::sleep(Duration::from_millis(1200)).await;
 assert_eq!(
  count_requests(&server, EXCHANGE_PATH).await,
  2,
  "401 must short-circuit — no retry storm"
 );

 // GitHub token retained on disk (Req 3.3) for the fast re-login.
 let stored = store.load().expect("load").expect("credentials retained");
 assert!(
  stored.github_access_token == GITHUB_TOKEN,
  "store still holds the GitHub token after the 401"
 );

 // Re-login is possible and fully succeeds over fresh mocks.
 mount_device_grant(&server, 0, 900).await;
 mount_immediate_github_token(&server).await;
 mount_exchange(&server, now_plus(3600)).await;
 match manager.start_login().await.expect("re-login should start") {
  LoginStart::Started(_) => {}
  LoginStart::InFlight(_) => panic!("expired session must allow a fresh login flow"),
 }
 wait_for_authenticated(&manager).await;
 let token = manager.get_copilot_token().await.expect("token after re-login");
 assert!(token == COPILOT_BEARER, "re-login exchanges a fresh bearer");
 server.verify().await;
}

// ---- cancel path ----------------------------------------------------------

/// Cancelling an in-flight login ends the poll task promptly and
/// restores `Unauthenticated`; no polls continue afterwards.
#[tokio::test]
async fn cancel_login_ends_promptly_and_restores_unauthenticated() {
 let server = MockServer::start().await;
 let manager = CopilotTokenManager::with_endpoints(
  unique_store(),
  test_client(),
  endpoints(&server),
 );

 // Pending forever (1 s cadence, long code life) so the login stays in
 // flight until cancelled.
 mount_device_grant(&server, 1, 900).await;
 mount_poll_pending(&server).await;

 manager.start_login().await.expect("login should start");
 assert_eq!(manager.session_state().await, CopilotSessionState::Pending);

 let cancel_at = std::time::Instant::now();
 manager.cancel_login().await;

 // Prompt: the graceful state restore lands well inside the poll
 // cadence budget.
 wait_for_terminal(&manager, CopilotSessionState::Unauthenticated).await;
 let restore_latency = cancel_at.elapsed();
 assert!(
  restore_latency < Duration::from_secs(3),
  "cancel must restore state promptly, took {:?}",
  restore_latency
 );

 // The task is really done: poll count stable across a window longer
 // than the poll interval.
 let after_cancel = count_requests(&server, TOKEN_POLL_PATH).await;
 tokio::time::sleep(Duration::from_millis(1300)).await;
 assert_eq!(
  count_requests(&server, TOKEN_POLL_PATH).await,
  after_cancel,
  "no polls after cancellation — task ended"
 );
}

// ---- corrupt store recovery -----------------------------------------------

/// A corrupt/unreadable store file is a warn + `Unauthenticated` on
/// load (no panic), and the next login writes a fresh state over it.
#[tokio::test]
async fn corrupt_store_recovers_to_unauthenticated_then_fresh_login() {
 let store = unique_store();
 std::fs::write(store.file_path(), b"this-is-not-an-enc-v1-blob").expect("seed corrupt file");

 let server = MockServer::start().await;
 let manager =
  CopilotTokenManager::with_endpoints(store.clone(), test_client(), endpoints(&server));

 // Corrupt blob → warn + Unauthenticated, NOT an error return or panic.
 manager
  .load_existing_session()
  .await
  .expect("corrupt store is recoverable, not fatal");
 assert_eq!(manager.session_state().await, CopilotSessionState::Unauthenticated);

 // A fresh login overwrites the corrupt file with a valid encrypted one.
 mount_device_grant(&server, 0, 900).await;
 mount_immediate_github_token(&server).await;
 mount_exchange(&server, now_plus(3600)).await;
 manager.start_login().await.expect("login should start");
 wait_for_authenticated(&manager).await;

 let stored = store.load().expect("load").expect("fresh credentials written");
 assert!(
  stored.github_access_token == GITHUB_TOKEN,
  "fresh login overwrote the corrupt store"
 );
 let raw = std::fs::read_to_string(store.file_path()).expect("read raw file");
 assert!(raw.starts_with("enc-v1:"), "overwritten store is a fresh enc-v1 blob");
 server.verify().await;
}

// ---- redaction (Req 8.4 / 3.5) --------------------------------------------

/// Token material never appears in Debug output of the stored
/// credentials or the login prompt, and never lands on disk in
/// plaintext.
#[tokio::test]
async fn no_token_literals_in_debug_output_or_store_file() {
 let server = MockServer::start().await;
 let store = unique_store();
 let manager =
  CopilotTokenManager::with_endpoints(store.clone(), test_client(), endpoints(&server));

 mount_device_grant(&server, 0, 900).await;
 mount_immediate_github_token(&server).await;
 mount_exchange(&server, now_plus(3600)).await;
 let prompt = match manager.start_login().await.expect("login should start") {
  LoginStart::Started(p) => p,
  LoginStart::InFlight(_) => panic!("first login must start a fresh flow"),
 };
 wait_for_authenticated(&manager).await;

 // Stored-credentials Debug: token fields redacted, literals absent.
 let stored = store.load().expect("load").expect("credentials");
 let creds_debug = format!("{:?}", stored);
 assert!(
  !creds_debug.contains(GITHUB_TOKEN) && !creds_debug.contains(COPILOT_BEARER),
  "credentials Debug must not contain token literals"
 );
 assert!(creds_debug.contains("<redacted>"), "token fields render as redacted");

 // Login prompt Debug: carries the user-facing code/URL, never the
 // secret device code.
 let prompt_debug = format!("{:?}", prompt);
 assert!(
  !prompt_debug.contains(DEVICE_CODE),
  "prompt Debug must not contain the device code"
 );
 assert!(prompt_debug.contains(USER_CODE), "prompt Debug keeps the safe user code");

 // Raw file: enc-v1 ciphertext, no plaintext token fragments.
 let raw = std::fs::read_to_string(store.file_path()).expect("read raw file");
 assert!(
  !raw.contains("gho-") && !raw.contains("tid="),
  "store file must not contain token material in plaintext"
 );
 assert!(
  !raw.contains(COPILOT_API_BASE),
  "store file must not contain the API base in plaintext"
 );
 server.verify().await;
}

// ---- Req 2.6: in-flight login dedupe --------------------------------------

/// Two concurrent `start_login` calls produce exactly ONE device-code
/// request and both callers observe the SAME user code (one `Started`,
/// one `InFlight`).
#[tokio::test]
async fn concurrent_start_login_dedupes_to_single_device_code_request() {
 let server = MockServer::start().await;
 let manager = CopilotTokenManager::with_endpoints(
  unique_store(),
  test_client(),
  endpoints(&server),
 );

 // Keep the poll pending (1 s cadence) so the login stays in flight
 // across both calls.
 mount_device_grant(&server, 1, 900).await;
 mount_poll_pending(&server).await;

 let (first, second) = tokio::join!(manager.start_login(), manager.start_login());
 let first = first.expect("first login");
 let second = second.expect("second login");

 let (started_prompt, in_flight_prompt) = match (first, second) {
  (LoginStart::Started(a), LoginStart::InFlight(b)) => (a, b),
  (LoginStart::InFlight(a), LoginStart::Started(b)) => (b, a),
  other => panic!(
   "expected one Started + one InFlight, got {:?}",
   other
  ),
 };
 assert_eq!(in_flight_prompt, started_prompt, "deduped prompt equals the original");
 assert!(
  in_flight_prompt.user_code == USER_CODE,
  "same user code returned to both callers"
 );

 // Exactly one device-code request despite two concurrent calls.
 assert_eq!(
  count_requests(&server, DEVICE_CODE_PATH).await,
  1,
  "in-flight dedupe must not issue a second device code"
 );

 // Wind the pending flow down.
 manager.cancel_login().await;
 wait_for_terminal(&manager, CopilotSessionState::Unauthenticated).await;
}
