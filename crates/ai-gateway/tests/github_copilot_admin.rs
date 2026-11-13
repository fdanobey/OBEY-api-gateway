//! Admin API integration tests for the GitHub Copilot OAuth endpoints
//! (spec `github-copilot-oauth-provider`, task 14; Reqs 2.1, 6.2, 6.4,
//! 7.1, 7.2).
//!
//! Drives the REAL gateway router (`GatewayServer::new` + `build_router`
//! + `tower::ServiceExt::oneshot`) against the five
//! `/admin/oauth/github-copilot/*` routes.
//!
//! # Wiremock wiring
//!
//! The gateway constructs its Copilot manager with PRODUCTION endpoints
//! (gateway/mod.rs `CopilotTokenManager::new`), so a login driven purely
//! through the gateway would hit real github.com. Instead each test
//! builds a real gateway, then swaps `AppState.copilot_manager` /
//! `copilot_usage_tracker` / `copilot_catalog` for instances whose
//! `CopilotEndpoints` point at a wiremock server — the exact Arcs the
//! admin handlers resolve — BEFORE `build_router()` captures the state.
//! No request ever leaves the loopback interface.
//!
//! # Isolation
//!
//! `AI_GATEWAY_DATA_DIR` is process-global and re-read per request (the
//! admin store is derived from `secrets::master_key_path()` on every
//! call), so every test takes a global mutex for its whole body and
//! points the env var at its own temp dir (master key created eagerly).
//! With `RUST_TEST_THREADS=2` the full-server tests serialize on that
//! mutex; each gateway's databases are additionally isolated via
//! `common::isolate_databases`.
//!
//! # Not-configured path
//!
//! The documented `configuration_error` envelope is gated on the
//! master-key wiring (`AppState.copilot_manager == None`), NOT on the
//! presence of a `github-copilot` provider in the config — account-level
//! OAuth is provider-independent by design. The not-configured test
//! therefore drives the envelope by clearing the wired manager/tracker
//! off a real built gateway, which exercises the production code path
//! (`copilot_admin_state` → `None` → `copilot_not_configured`) through
//! the real router.
//!
//! Security: token literals (`gho_…`, `tid=…`, device codes) never
//! appear in assertion messages — every redaction check is boolean with
//! a static message.

use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::{Arc, MutexGuard};

use axum::body::Body;
use axum::http::{Request, StatusCode};
use tower::ServiceExt;
use wiremock::matchers::{method, path};
use wiremock::{Mock, MockServer, ResponseTemplate};

use ai_gateway::gateway::GatewayServer;
use ai_gateway::oauth::github_copilot::catalog::CopilotCatalogSync;
use ai_gateway::oauth::github_copilot::device_flow::DeviceFlowEndpoints;
use ai_gateway::oauth::github_copilot::manager::{
    CopilotEndpoints, CopilotSessionState, CopilotTokenManager,
};
use ai_gateway::oauth::github_copilot::store::CopilotTokenStore;
use ai_gateway::oauth::UsageTracker;
use serde_json::json;

mod common;

// ---- fixture constants (never printed in assertion output) ---------------

const DEVICE_CODE: &str = "dc-admin-int-1";
const USER_CODE: &str = "ADMN-1234";
const VERIFICATION_URI: &str = "https://github.com/login/device";
const GITHUB_TOKEN: &str = "gho-admin-int-github-token";
const COPILOT_BEARER: &str = "tid=admin-int-copilot-bearer";

const LOGIN_PATH: &str = "/admin/oauth/github-copilot/login";
const STATUS_PATH: &str = "/admin/oauth/github-copilot/status";
const USAGE_PATH: &str = "/admin/oauth/github-copilot/usage";
const LOGOUT_PATH: &str = "/admin/oauth/github-copilot/logout";
const CANCEL_PATH: &str = "/admin/oauth/github-copilot/cancel";

// ---- env serialization + per-test isolation ------------------------------

/// Serializes every test in this binary: `AI_GATEWAY_DATA_DIR` is
/// process-global and re-read at request time, so the whole gateway
/// lifetime (construction → requests) must run while the env var points
/// at this test's dir.
static ENV_LOCK: std::sync::Mutex<()> = std::sync::Mutex::new(());

fn lock_env() -> MutexGuard<'static, ()> {
    ENV_LOCK
        .lock()
        .unwrap_or_else(|poisoned| poisoned.into_inner())
}

/// Real gateway over an isolated data dir (unique master key + credential
/// store) and isolated SQLite databases.
struct AdminHarness {
    _guard: MutexGuard<'static, ()>,
    data_dir: std::path::PathBuf,
    server: GatewayServer,
}

async fn gateway_harness() -> AdminHarness {
    let guard = lock_env();
    let dir = tempfile::tempdir().expect("temp data dir").keep();
    std::env::set_var("AI_GATEWAY_DATA_DIR", &dir);
    let _ = ai_gateway::secrets::load_or_create_master_key();

    let mut config = common::budget_test_config(None, None);
    common::isolate_databases(&mut config);
    let server = GatewayServer::new(config, None).await.expect("gateway");

    AdminHarness {
        _guard: guard,
        data_dir: dir,
        server,
    }
}

impl AdminHarness {
    /// Swap in a wiremock-backed manager over the SAME default store path
    /// the admin handlers derive from `master_key_path()` (so manager
    /// writes and status-handler `sku` reads observe one file).
    /// Must be called before the first `build_router()`.
    fn wire_copilot_manager(&mut self, endpoints: CopilotEndpoints) -> Arc<CopilotTokenManager> {
        let store = CopilotTokenStore::new(CopilotTokenStore::default_path(&self.data_dir));
        let manager = Arc::new(CopilotTokenManager::with_endpoints(
            store,
            reqwest::Client::new(),
            endpoints,
        ));
        self.server.state.copilot_manager = Some(Arc::clone(&manager));
        manager
    }

    /// Swap in a fresh usage tracker (seed it afterwards).
    fn wire_usage_tracker(&mut self) -> Arc<UsageTracker> {
        let tracker = Arc::new(UsageTracker::new());
        self.server.state.copilot_usage_tracker = Some(Arc::clone(&tracker));
        tracker
    }
}

/// All three Copilot OAuth endpoint legs pointed at one wiremock server.
fn endpoints(server: &MockServer) -> CopilotEndpoints {
    CopilotEndpoints {
        device_flow: DeviceFlowEndpoints {
            device_code_url: format!("{}/login/device/code", server.uri()),
            token_url: format!("{}/login/oauth/access_token", server.uri()),
        },
        copilot_token_url: format!("{}/copilot_internal/v2/token", server.uri()),
    }
}

// ---- request helpers ------------------------------------------------------

async fn send(app: axum::Router, req: Request<Body>) -> (StatusCode, serde_json::Value, String) {
    let resp = app.oneshot(req).await.expect("oneshot");
    let status = resp.status();
    let bytes = axum::body::to_bytes(resp.into_body(), 1024 * 1024)
        .await
        .expect("body");
    let raw = String::from_utf8_lossy(&bytes).into_owned();
    let json = if bytes.is_empty() {
        serde_json::Value::Null
    } else {
        serde_json::from_slice(&bytes).expect("JSON body")
    };
    (status, json, raw)
}

fn post(uri: &str) -> Request<Body> {
    Request::post(uri).body(Body::empty()).unwrap()
}

fn get(uri: &str) -> Request<Body> {
    Request::get(uri).body(Body::empty()).unwrap()
}

// ---- wiremock fixtures ----------------------------------------------------

async fn mount_grant_once(server: &MockServer, interval: u64) {
    Mock::given(method("POST"))
        .and(path("/login/device/code"))
        .respond_with(ResponseTemplate::new(200).set_body_json(json!({
            "device_code": DEVICE_CODE,
            "user_code": USER_CODE,
            "verification_uri": VERIFICATION_URI,
            "expires_in": 900,
            "interval": interval
        })))
        .up_to_n_times(1)
        .expect(1)
        .mount(server)
        .await;
}

/// Grant issuance capped + verified at exactly ONE hit across any number
/// of login calls (in-flight dedupe proof).
async fn mount_grant_exactly_once(server: &MockServer) {
    Mock::given(method("POST"))
        .and(path("/login/device/code"))
        .respond_with(ResponseTemplate::new(200).set_body_json(json!({
            "device_code": DEVICE_CODE,
            "user_code": USER_CODE,
            "verification_uri": VERIFICATION_URI,
            "expires_in": 900,
            "interval": 0
        })))
        .expect(1)
        .mount(server)
        .await;
}

async fn mount_poll_pending(server: &MockServer) {
    Mock::given(method("POST"))
        .and(path("/login/oauth/access_token"))
        .respond_with(ResponseTemplate::new(200).set_body_json(json!({
            "error": "authorization_pending"
        })))
        .expect(0..)
        .mount(server)
        .await;
}

/// Stateful token poll: answers `authorization_pending` until the
/// approval flag flips, then answers the GitHub token once (later calls
/// fail loudly via the mock-server verify). Gives the test a
/// deterministic, observable pending → authenticated transition.
struct PendingUntilApproved {
    state: Arc<PollState>,
}

struct PollState {
    approved: AtomicBool,
    served_success: AtomicBool,
}

impl wiremock::Respond for PendingUntilApproved {
    fn respond(&self, _request: &wiremock::Request) -> ResponseTemplate {
        if !self.state.approved.load(Ordering::SeqCst) {
            return ResponseTemplate::new(200).set_body_json(json!({
                "error": "authorization_pending"
            }));
        }
        if self.state.served_success.swap(true, Ordering::SeqCst) {
            return ResponseTemplate::new(500).set_body_string("unexpected extra poll");
        }
        ResponseTemplate::new(200).set_body_json(json!({
            "access_token": GITHUB_TOKEN,
            "token_type": "bearer",
            "scope": "read:user"
        }))
    }
}

async fn mount_exchange_once(server: &MockServer, expires_at: u64) {
    Mock::given(method("GET"))
        .and(path("/copilot_internal/v2/token"))
        .respond_with(ResponseTemplate::new(200).set_body_json(json!({
            "token": COPILOT_BEARER,
            "expires_at": expires_at,
            "refresh_in": 1500,
            "endpoints": { "api": server.uri() },
            "sku": "copilot_pro"
        })))
        .up_to_n_times(1)
        .expect(1)
        .mount(server)
        .await;
}

async fn mount_models_catalog(server: &MockServer) {
    Mock::given(method("GET"))
        .and(path("/models"))
        .respond_with(ResponseTemplate::new(200).set_body_json(json!({
            "data": [
                { "id": "gpt-5-codex", "model_picker_enabled": true },
                { "id": "gated-off", "model_picker_enabled": false }
            ]
        })))
        .expect(1..)
        .mount(server)
        .await;
}

// ---- shared assertions ----------------------------------------------------

/// Req 6.4 redaction, substring scan across a whole response body: no
/// GitHub token prefix, no Copilot bearer prefix, none of the fixture
/// token/device-code literals. Static messages only.
fn assert_redacted(context: &str, raw: &str) {
    assert!(
        !raw.contains("gho_"),
        "{context}: gho_ prefix must never appear"
    );
    assert!(
        !raw.contains("tid="),
        "{context}: tid= prefix must never appear"
    );
    assert!(
        !raw.contains(DEVICE_CODE),
        "{context}: device code must never appear"
    );
    assert!(
        !raw.contains(GITHUB_TOKEN),
        "{context}: GitHub access token must never appear"
    );
    assert!(
        !raw.contains(COPILOT_BEARER),
        "{context}: Copilot bearer must never appear"
    );
    // No token-bearing credential KEYS beyond the intended user_code
    // display field (exact quoted-key forms; `tokens` windows are not
    // credential fields and must not trip this).
    assert!(
        !raw.contains("\"device_code\""),
        "{context}: device_code key must never appear"
    );
    assert!(
        !raw.contains("\"access_token\""),
        "{context}: access_token key must never appear"
    );
    assert!(
        !raw.contains("\"github_access_token\""),
        "{context}: stored GitHub token key must never appear"
    );
    assert!(
        !raw.contains("\"copilot_token\""),
        "{context}: stored Copilot token key must never appear"
    );
}

fn sorted_keys(json: &serde_json::Value) -> Vec<String> {
    let mut keys: Vec<String> = json
        .as_object()
        .expect("response must be a JSON object")
        .keys()
        .cloned()
        .collect();
    keys.sort();
    keys
}

async fn wait_authenticated(manager: &CopilotTokenManager) {
    for _ in 0..400 {
        if matches!(
            manager.session_state().await,
            CopilotSessionState::Authenticated { .. }
        ) {
            return;
        }
        tokio::time::sleep(std::time::Duration::from_millis(25)).await;
    }
    panic!("session never reached Authenticated");
}

async fn wait_unauthenticated(manager: &CopilotTokenManager) {
    for _ in 0..400 {
        if manager.session_state().await == CopilotSessionState::Unauthenticated {
            return;
        }
        tokio::time::sleep(std::time::Duration::from_millis(25)).await;
    }
    panic!("session never returned to Unauthenticated");
}

fn now_plus(secs: u64) -> u64 {
    std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .map(|d| d.as_secs())
        .unwrap_or(0)
        .saturating_add(secs)
}

/// Seed a tracker with the short/weekly/premium header set (Req 7.1/7.2
/// shapes; premium used=37 limit=100 → remaining=63, is_percentage).
async fn seed_usage(tracker: &UsageTracker) {
    let mut short_headers = reqwest::header::HeaderMap::new();
    short_headers.insert("x-ratelimit-limit-requests", "50".parse().unwrap());
    short_headers.insert("x-ratelimit-remaining-requests", "48".parse().unwrap());
    short_headers.insert("x-ratelimit-reset-requests", "3h".parse().unwrap());
    tracker.update_from_headers(&short_headers).await;

    let mut weekly_headers = reqwest::header::HeaderMap::new();
    weekly_headers.insert("x-ratelimit-limit-tokens", "100000".parse().unwrap());
    weekly_headers.insert("x-ratelimit-remaining-tokens", "99900".parse().unwrap());
    weekly_headers.insert("x-ratelimit-reset-tokens", "6d".parse().unwrap());
    weekly_headers.insert("x-copilot-premium-requests-used", "37".parse().unwrap());
    weekly_headers.insert("x-copilot-premium-requests-limit", "100".parse().unwrap());
    weekly_headers.insert("x-copilot-premium-requests-reset", "1d12h".parse().unwrap());
    tracker.update_from_headers(&weekly_headers).await;
}

// ---------------------------------------------------------------------------
// Req 2.1 / 2.6: login shape + in-flight dedupe over the real router
// ---------------------------------------------------------------------------

/// POST login (fresh) → `{status: "started", verification_uri, user_code,
/// expires_in, expires_at}`; a second call while in flight → SAME
/// verification URI + user code with `status: "in_flight"` and exactly
/// ONE upstream device-code request. Both bodies redacted (Req 6.4).
#[tokio::test]
async fn login_shape_and_in_flight_dedupe_over_real_gateway() {
    let mut harness = gateway_harness().await;
    let wiremock_server = MockServer::start().await;
    mount_grant_exactly_once(&wiremock_server).await;
    mount_poll_pending(&wiremock_server).await;
    let manager = harness.wire_copilot_manager(endpoints(&wiremock_server));

    let app = harness.server.build_router();

    let (status_a, json_a, raw_a) = send(app.clone(), post(LOGIN_PATH)).await;
    assert_eq!(status_a, StatusCode::OK);
    assert_eq!(json_a["status"], "started");
    assert_eq!(json_a["user_code"], USER_CODE);
    assert_eq!(json_a["verification_uri"], VERIFICATION_URI);
    assert_eq!(json_a["expires_in"], 900);
    assert!(
        json_a["expires_at"].as_u64().is_some(),
        "login must carry a derived absolute expiry"
    );
    // Exact safe-metadata key set: user_code is the one intended display
    // field; nothing else rides along.
    assert_eq!(
        sorted_keys(&json_a),
        vec![
            "expires_at",
            "expires_in",
            "status",
            "user_code",
            "verification_uri"
        ]
    );
    assert_redacted("login (started)", &raw_a);

    let (status_b, json_b, raw_b) = send(app.clone(), post(LOGIN_PATH)).await;
    assert_eq!(status_b, StatusCode::OK);
    assert_eq!(json_b["status"], "in_flight");
    assert_eq!(json_b["user_code"], json_a["user_code"]);
    assert_eq!(json_b["verification_uri"], json_a["verification_uri"]);
    assert_eq!(json_b["expires_in"], json_a["expires_in"]);
    assert_redacted("login (in_flight)", &raw_b);

    // Req 2.6 dedupe: exactly one device-code issuance upstream.
    wiremock_server.verify().await;

    // Wind the pending poll down before releasing the env lock.
    let (status_c, json_c, _) = send(app.clone(), post(CANCEL_PATH)).await;
    assert_eq!(status_c, StatusCode::OK);
    assert_eq!(json_c["status"], "cancelled");
    wait_unauthenticated(&manager).await;
}

// ---------------------------------------------------------------------------
// Req 6.4: status shape + unauthenticated → pending → authenticated
// ---------------------------------------------------------------------------

/// GET status carries `{state, expires_at, sku, error, catalog}` at every
/// stage; the state transitions unauthenticated → pending →
/// authenticated are all observable over wiremock, with the catalog
/// object populated post-auth (model_count from the wiremock catalog).
#[tokio::test]
async fn status_transitions_and_shape_over_real_gateway() {
    let mut harness = gateway_harness().await;
    let wiremock_server = MockServer::start().await;
    let manager = harness.wire_copilot_manager(endpoints(&wiremock_server));
    harness.wire_usage_tracker();

    // Wire the catalog manually (the gateway only builds one when a
    // github-copilot provider exists in the config; the endpoints
    // themselves are provider-independent). No background sync spawn —
    // refresh is driven explicitly below.
    let catalog = Arc::new(CopilotCatalogSync::with_owner(
        Arc::clone(&manager),
        reqwest::Client::new(),
        "copilot-test-provider".to_string(),
    ));
    catalog.set_configured_models(vec![]).await;
    harness.server.state.copilot_catalog = Some(Arc::clone(&catalog));

    let app = harness.server.build_router();

    // -- unauthenticated
    let (status, json, raw) = send(app.clone(), get(STATUS_PATH)).await;
    assert_eq!(status, StatusCode::OK);
    assert_eq!(json["state"], "unauthenticated");
    assert!(json["expires_at"].is_null());
    assert!(json["sku"].is_null());
    assert!(
        json.get("error").is_some(),
        "error field must always be present"
    );
    assert_eq!(json["catalog"]["model_count"], 0);
    assert_eq!(
        sorted_keys(&json),
        vec!["catalog", "error", "expires_at", "sku", "state"]
    );
    assert_redacted("status (unauthenticated)", &raw);

    // -- pending: login started, authorization stays pending
    mount_grant_once(&wiremock_server, 1).await;
    let poll = Arc::new(PollState {
        approved: AtomicBool::new(false),
        served_success: AtomicBool::new(false),
    });
    Mock::given(method("POST"))
        .and(path("/login/oauth/access_token"))
        .respond_with(PendingUntilApproved {
            state: Arc::clone(&poll),
        })
        .expect(1..)
        .mount(&wiremock_server)
        .await;
    let (login_status, login_json, login_raw) = send(app.clone(), post(LOGIN_PATH)).await;
    assert_eq!(login_status, StatusCode::OK);
    assert_eq!(login_json["status"], "started");
    assert_redacted("login", &login_raw);

    let (status, json, raw) = send(app.clone(), get(STATUS_PATH)).await;
    assert_eq!(status, StatusCode::OK);
    assert_eq!(json["state"], "pending");
    assert_redacted("status (pending)", &raw);

    // -- authenticated: approve the grant, exchange, catalog fetch
    let copilot_expires_at = now_plus(3600);
    poll.approved.store(true, Ordering::SeqCst);
    mount_exchange_once(&wiremock_server, copilot_expires_at).await;
    mount_models_catalog(&wiremock_server).await;
    wait_authenticated(&manager).await;
    catalog.refresh_now().await;

    let (status, json, raw) = send(app.clone(), get(STATUS_PATH)).await;
    assert_eq!(status, StatusCode::OK);
    assert_eq!(json["state"], "authenticated");
    assert_eq!(json["expires_at"].as_u64(), Some(copilot_expires_at));
    assert_eq!(json["sku"], "copilot_pro");
    assert_eq!(json["catalog"]["model_count"], 1);
    assert_eq!(json["catalog"]["fetch_failed"], false);
    assert_redacted("status (authenticated)", &raw);

    wiremock_server.verify().await;
}

// ---------------------------------------------------------------------------
// Req 7.1 / 7.2: usage snapshot shape incl. premium percentage window
// ---------------------------------------------------------------------------

/// GET usage returns the `UsageSnapshot` shape `{short, weekly, premium,
/// updated_at}` with `premium.is_percentage = true` when seeded via the
/// Copilot premium-response headers.
#[tokio::test]
async fn usage_snapshot_shape_with_premium_percentage() {
    let mut harness = gateway_harness().await;
    // Idle mock server: endpoints are never driven, but any accidental
    // call would stay on loopback instead of reaching github.com.
    let wiremock_server = MockServer::start().await;
    harness.wire_copilot_manager(endpoints(&wiremock_server));
    let tracker = harness.wire_usage_tracker();
    seed_usage(&tracker).await;

    let app = harness.server.build_router();

    let (status, json, raw) = send(app.clone(), get(USAGE_PATH)).await;
    assert_eq!(status, StatusCode::OK);
    assert_eq!(
        sorted_keys(&json),
        vec!["premium", "short", "updated_at", "weekly"]
    );
    // Short window (3h reset < weekly threshold).
    assert_eq!(json["short"]["requests"]["limit"], 50);
    assert_eq!(json["short"]["requests"]["remaining"], 48);
    // Weekly window (6d reset ≥ weekly threshold).
    assert_eq!(json["weekly"]["tokens"]["limit"], 100000);
    assert_eq!(json["weekly"]["tokens"]["remaining"], 99900);
    // Premium percentage window (Req 7.2): used 37 of 100 → 63 left.
    assert_eq!(json["premium"]["limit"], 100);
    assert_eq!(json["premium"]["remaining"], 63);
    assert_eq!(json["premium"]["is_percentage"], true);
    assert!(json["updated_at"].as_u64().is_some_and(|t| t > 0));
    assert_redacted("usage", &raw);
}

// ---------------------------------------------------------------------------
// Req 6.2: logout clears the store + state; idempotent
// ---------------------------------------------------------------------------

/// POST logout on an authenticated session: the persisted store file is
/// deleted, the state returns to `unauthenticated`, the cached usage is
/// cleared, and a second logout still succeeds.
#[tokio::test]
async fn logout_clears_store_state_and_usage_idempotently() {
    let mut harness = gateway_harness().await;
    let wiremock_server = MockServer::start().await;
    let manager = harness.wire_copilot_manager(endpoints(&wiremock_server));
    let tracker = harness.wire_usage_tracker();

    // Full login: grant → token → exchange (far-future expiry). The
    // poll is approved immediately (deterministic first-poll success).
    mount_grant_once(&wiremock_server, 1).await;
    Mock::given(method("POST"))
        .and(path("/login/oauth/access_token"))
        .respond_with(PendingUntilApproved {
            state: Arc::new(PollState {
                approved: AtomicBool::new(true),
                served_success: AtomicBool::new(false),
            }),
        })
        .expect(1..)
        .mount(&wiremock_server)
        .await;
    mount_exchange_once(&wiremock_server, now_plus(3600)).await;

    let app = harness.server.build_router();
    let (login_status, login_json, _) = send(app.clone(), post(LOGIN_PATH)).await;
    assert_eq!(login_status, StatusCode::OK);
    assert_eq!(login_json["status"], "started");
    wait_authenticated(&manager).await;

    let store_path = CopilotTokenStore::default_path(&harness.data_dir);
    assert!(
        store_path.exists(),
        "login must persist the credential store"
    );

    // Seed cached usage so the logout's clear is observable.
    seed_usage(&tracker).await;
    let seeded = tracker.snapshot().await;
    assert!(seeded.premium.is_some());

    let (status, json, raw) = send(app.clone(), post(LOGOUT_PATH)).await;
    assert_eq!(status, StatusCode::OK);
    assert_eq!(json["status"], "logged_out");
    assert_redacted("logout", &raw);

    assert!(
        !store_path.exists(),
        "logout must delete the persisted credential file"
    );

    let (status, json, raw) = send(app.clone(), get(STATUS_PATH)).await;
    assert_eq!(status, StatusCode::OK);
    assert_eq!(json["state"], "unauthenticated");
    assert_redacted("status after logout", &raw);

    let (status, json, raw) = send(app.clone(), get(USAGE_PATH)).await;
    assert_eq!(status, StatusCode::OK);
    assert!(json["premium"].is_null(), "logout must clear cached usage");
    assert_eq!(json["updated_at"], 0);
    assert_redacted("usage after logout", &raw);

    // Idempotent.
    let (status, json, _) = send(app.clone(), post(LOGOUT_PATH)).await;
    assert_eq!(status, StatusCode::OK);
    assert_eq!(json["status"], "logged_out");

    wiremock_server.verify().await;
}

// ---------------------------------------------------------------------------
// Req 2.2: cancel restores the pre-login state
// ---------------------------------------------------------------------------

/// POST cancel on a pending login: `{status: "cancelled"}`, the session
/// restores to `unauthenticated`, and cancelling with nothing in flight
/// still returns 200.
#[tokio::test]
async fn cancel_cancels_pending_login_and_restores_state() {
    let mut harness = gateway_harness().await;
    let wiremock_server = MockServer::start().await;
    mount_grant_once(&wiremock_server, 1).await;
    mount_poll_pending(&wiremock_server).await;
    let manager = harness.wire_copilot_manager(endpoints(&wiremock_server));

    let app = harness.server.build_router();

    let (status, json, raw) = send(app.clone(), get(STATUS_PATH)).await;
    assert_eq!(status, StatusCode::OK);
    assert_eq!(json["state"], "unauthenticated");
    assert_redacted("status before login", &raw);

    let (status, json, _) = send(app.clone(), post(LOGIN_PATH)).await;
    assert_eq!(status, StatusCode::OK);
    assert_eq!(json["status"], "started");

    let (status, json, raw) = send(app.clone(), get(STATUS_PATH)).await;
    assert_eq!(status, StatusCode::OK);
    assert_eq!(json["state"], "pending");
    assert_redacted("status pending", &raw);

    let (status, json, raw) = send(app.clone(), post(CANCEL_PATH)).await;
    assert_eq!(status, StatusCode::OK);
    assert_eq!(json["status"], "cancelled");
    assert_redacted("cancel", &raw);

    // The poll task observes the cancel and restores the state.
    wait_unauthenticated(&manager).await;
    let (status, json, raw) = send(app.clone(), get(STATUS_PATH)).await;
    assert_eq!(status, StatusCode::OK);
    assert_eq!(json["state"], "unauthenticated");
    assert_redacted("status after cancel", &raw);

    // Idempotent: cancelling with nothing in flight still 200s.
    let (status, json, _) = send(app.clone(), post(CANCEL_PATH)).await;
    assert_eq!(status, StatusCode::OK);
    assert_eq!(json["status"], "cancelled");
}

// ---------------------------------------------------------------------------
// Not-configured path: configuration_error envelope
// ---------------------------------------------------------------------------

/// A gateway without the Copilot OAuth wiring (master-key gate never
/// constructed the manager — `AppState.copilot_manager == None`) serves
/// the documented `configuration_error` envelope on every endpoint.
#[tokio::test]
async fn not_configured_endpoints_serve_configuration_error_envelope() {
    let mut harness = gateway_harness().await;
    harness.server.state.copilot_manager = None;
    harness.server.state.copilot_usage_tracker = None;

    let app = harness.server.build_router();

    for (method_label, request) in [
        ("login", post(LOGIN_PATH)),
        ("cancel", post(CANCEL_PATH)),
        ("logout", post(LOGOUT_PATH)),
        ("status", get(STATUS_PATH)),
        ("usage", get(USAGE_PATH)),
    ] {
        let (status, json, raw) = send(app.clone(), request).await;
        assert_eq!(
            status,
            StatusCode::INTERNAL_SERVER_ERROR,
            "{method_label} must serve 500 when Copilot OAuth is not configured"
        );
        assert_eq!(
            json["error"]["type"], "configuration_error",
            "{method_label} must serve the configuration_error envelope"
        );
        assert_eq!(
            json["error"]["message"], "GitHub Copilot OAuth not configured",
            "{method_label} envelope message"
        );
        assert_redacted(method_label, &raw);
    }
}
