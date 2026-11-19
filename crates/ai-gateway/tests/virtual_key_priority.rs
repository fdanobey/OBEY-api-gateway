//! End-to-end integration tests for virtual-key priority classes
//! (spec: virtual-key-priority-classes, Task 6).
//!
//! Through-the-router coverage of the priority-ordered admission queue:
//! requests are dispatched through the full Axum router via
//! `tower::ServiceExt::oneshot()` (auth middleware → handler → router
//! admission gate → wiremock upstream) with virtual keys created in an
//! isolated keys.db. `GatewayServer::new` performs the production wiring
//! (`router.set_virtual_key_manager`), which the admission gate needs to
//! resolve each authenticated key's priority class.
//!
//! Timing discipline: assertions are structural (completion order via a
//! shared ordering vector, upstream hit counts, request-log rows) or carry
//! generous margins. The local bucket refills one token per
//! 60/rpm seconds (2 s at 30/min, 10 s at 6/min), which dwarfs both the
//! 750 ms launch gap between contended requests and the ~ms dispatch
//! latency, so no assertion depends on tight wall-clock behavior.

use std::sync::{Arc, Mutex};
use std::time::Duration;

use axum::body::Body;
use axum::http::{Request, StatusCode};
use tower::ServiceExt;
use wiremock::matchers::{method, path};
use wiremock::{Mock, MockServer, ResponseTemplate};

use ai_gateway::config::Config;
use ai_gateway::gateway::GatewayServer;
use ai_gateway::logger::LogEntry;
use ai_gateway::virtual_keys::models::{CreateKeyParams, UpdateKeyParams};

use crate::common;

/// Wall-clock budget per test: a hung dispatch fails the test instead of
/// hanging the suite.
const TEST_TIMEOUT: Duration = Duration::from_secs(30);
/// Launch offset between two contended requests: enough for the first
/// request to park on the admission queue (localhost dispatch plus one
/// keys.db read via the blocking pool) while far below the token refill
/// spacing, making queue-arrival order deterministic even under a
/// loaded CI host.
const LAUNCH_GAP: Duration = Duration::from_millis(750);

const GROUP: &str = "priority-group";
const ATTEMPT_PATH: &str = "/v1/chat/completions#attempt";
const OUTCOME_PATH: &str = "/v1/chat/completions";

// ---------------------------------------------------------------------------
// Harness helpers
// ---------------------------------------------------------------------------

fn completion_body() -> serde_json::Value {
    serde_json::json!({
        "id": "chatcmpl-priority-mock",
        "object": "chat.completion",
        "created": 1_700_000_000_i64,
        "model": "gpt-4",
        "choices": [{
            "index": 0,
            "message": { "role": "assistant", "content": "Hello from mock provider" },
            "finish_reason": "stop"
        }],
        "usage": { "prompt_tokens": 10, "completion_tokens": 5, "total_tokens": 15 }
    })
}

async fn mount_ok_upstream(server: &MockServer) {
    Mock::given(method("POST"))
        .and(path("/v1/chat/completions"))
        .respond_with(ResponseTemplate::new(200).set_body_json(completion_body()))
        .mount(server)
        .await;
}

/// One rate-limited `primary` (candidate 1) plus an optional unlimited
/// `backup` (candidate 2), with virtual-key enforcement in `optional` mode
/// (presented keys are authenticated, tracked, and priority-resolved).
fn priority_config(primary_uri: &str, backup_uri: Option<&str>, rpm: u32, max_wait_ms: u64) -> Config {
    let mut providers = vec![serde_json::json!({
        "name": "primary",
        "type": "openai",
        "base_url": primary_uri,
        "rate_limit_per_minute": rpm,
        "rate_limit_max_wait_ms": max_wait_ms,
    })];
    let mut models =
        vec![serde_json::json!({ "provider": "primary", "model": "gpt-4", "priority": 1 })];
    if let Some(uri) = backup_uri {
        providers.push(serde_json::json!({ "name": "backup", "type": "openai", "base_url": uri }));
        models.push(serde_json::json!({ "provider": "backup", "model": "gpt-4", "priority": 2 }));
    }
    let mut config: Config = serde_json::from_value(serde_json::json!({
        "server": { "host": "127.0.0.1", "port": 0 },
        "providers": providers,
        "model_groups": [{ "name": GROUP, "models": models }],
        "virtual_keys": { "enforcement": "optional" },
    }))
    .expect("priority-class test configuration should deserialize");
    // Repeated probe bodies must traverse the admission gate, not be served
    // by the exact-match cache or pinned to a provider by stickiness.
    config.exact_cache.enabled = false;
    config.cache_aware_routing.stickiness_ttl_seconds = 0;
    common::isolate_databases(&mut config);
    config
}

/// Default (all-unlimited) create params with an explicit priority class.
fn key_params(priority: u16) -> CreateKeyParams {
    CreateKeyParams {
        name: None,
        budget_limit_usd: None,
        token_budget: None,
        budget_window: None,
        requests_per_minute: None,
        tokens_per_minute: None,
        max_output_tokens: None,
        model_access: None,
        expires_in: None,
        loop_detection: None,
        priority: Some(priority),
    }
}

fn chat_request(bearer: &str, content: &str) -> Request<Body> {
    Request::post("/v1/chat/completions")
        .header("content-type", "application/json")
        .header("authorization", format!("Bearer {bearer}"))
        .body(Body::from(
            serde_json::json!({
                "model": GROUP,
                "messages": [{ "role": "user", "content": content }],
                "stream": false
            })
            .to_string(),
        ))
        .unwrap()
}

/// Drain the provider's local bucket so every subsequent dispatch must
/// queue for a refill (the fractional residue refills over seconds, so no
/// full token exists at return time).
async fn drain_provider_tokens(server: &GatewayServer, provider: &str) {
    let limiter = server.state.router.get_rate_limiter(provider).await;
    while limiter.consume().await {}
}

/// Align the drained bucket's refill phase for deterministic two-token
/// scenarios: single computed sleep until the fractional balance is
/// `offset` seconds PAST a full-token boundary, so the next full token is
/// `period - offset` seconds away and the one after that a further
/// `period`. Call right before launching contended requests.
async fn align_bucket_phase(server: &GatewayServer, provider: &str, rpm: u32, offset: Duration) {
    let period_secs = 60.0 / rpm as f64;
    let offset_secs = offset.as_secs_f64();
    let limiter = server.state.router.get_rate_limiter(provider).await;
    let tokens = limiter.get_tokens().await;
    // Seconds elapsed since the last full-token boundary.
    let since_boundary = (tokens - tokens.floor()) * period_secs;
    let wait_secs = if since_boundary < offset_secs {
        offset_secs - since_boundary
    } else {
        period_secs - since_boundary + offset_secs
    };
    tokio::time::sleep(Duration::from_secs_f64(wait_secs)).await;
}

async fn tokens_left(server: &GatewayServer, provider: &str) -> f64 {
    server
        .state
        .router
        .get_rate_limiter(provider)
        .await
        .get_tokens()
        .await
}

async fn chat_requests(server: &MockServer) -> usize {
    server
        .received_requests()
        .await
        .unwrap_or_default()
        .iter()
        .filter(|r| r.url.path() == "/v1/chat/completions")
        .count()
}

/// Dispatch `(label, key, delay)` jobs concurrently through `app` and
/// return `(label, status)` pairs in COMPLETION order (pushed under a
/// shared mutex by each task — JoinSet output order is not used).
async fn dispatch_ordered(
    app: axum::Router,
    jobs: Vec<(&'static str, String, Duration)>,
) -> Vec<(&'static str, StatusCode)> {
    let order: Arc<Mutex<Vec<(&'static str, StatusCode)>>> = Arc::new(Mutex::new(Vec::new()));
    let mut joins = Vec::new();
    for (label, key, delay) in jobs {
        let app = app.clone();
        let order = Arc::clone(&order);
        joins.push(tokio::spawn(async move {
            if !delay.is_zero() {
                tokio::time::sleep(delay).await;
            }
            let response = app
                .oneshot(chat_request(&key, label))
                .await
                .expect("oneshot dispatch");
            let status = response.status();
            let _ = axum::body::to_bytes(response.into_body(), 1024 * 1024)
                .await
                .expect("response body");
            order.lock().expect("completion-order mutex").push((label, status));
        }));
    }
 for join in joins {
 join.await.expect("dispatch task panicked");
 }
 let finish_order = order.lock().expect("completion-order mutex");
 finish_order.clone()
}

async fn logs(server: &GatewayServer) -> Vec<LogEntry> {
    let response = server
        .build_router()
        .oneshot(Request::get("/dashboard/logs").body(Body::empty()).unwrap())
        .await
        .unwrap();
    assert_eq!(response.status(), StatusCode::OK);
    let body = axum::body::to_bytes(response.into_body(), 4 * 1024 * 1024)
        .await
        .unwrap();
    serde_json::from_slice(&body).expect("dashboard logs should be LogEntry JSON")
}

/// Poll the request log until `done` holds or `timeout` passes.
async fn poll_logs(
    server: &GatewayServer,
    timeout: Duration,
    done: impl Fn(&[LogEntry]) -> bool,
) -> Vec<LogEntry> {
    let deadline = tokio::time::Instant::now() + timeout;
    loop {
        let rows = logs(server).await;
        if done(&rows) || tokio::time::Instant::now() >= deadline {
            return rows;
        }
        tokio::time::sleep(Duration::from_millis(50)).await;
    }
}

// ---------------------------------------------------------------------------
// (a) Contended ordering: priority beats arrival
// ---------------------------------------------------------------------------

/// Two keys (priority 0 and 10) contend for a provider whose bucket holds
/// no tokens (30/min → one refill every 2 s, wait budget 12 s). The
/// priority-0 request is launched first and parks on the queue first, but
/// the next two tokens must go to the priority-10 request first: completion
/// order is [high, low], both served by the primary, no failover.
#[tokio::test]
async fn contended_admission_serves_highest_priority_key_first() {
    tokio::time::timeout(TEST_TIMEOUT, async {
        let primary = MockServer::start().await;
        let backup = MockServer::start().await;
        mount_ok_upstream(&primary).await;
        mount_ok_upstream(&backup).await;
        let server = GatewayServer::new(
            priority_config(&primary.uri(), Some(&backup.uri()), 30, 12_000),
            None,
        )
        .await
        .unwrap();
        let manager = server.state.virtual_key_manager.clone();
        let low = manager.create_key(key_params(0)).await.unwrap();
        let high = manager.create_key(key_params(10)).await.unwrap();
        drain_provider_tokens(&server, "primary").await;

        let done = dispatch_ordered(
            server.build_router(),
            vec![
                ("low", low.key.clone(), Duration::ZERO),
                ("high", high.key.clone(), LAUNCH_GAP),
            ],
        )
        .await;

        assert_eq!(done.len(), 2);
        assert_eq!(done[0].0, "high", "completion order: {done:?}");
        assert_eq!(done[1].0, "low", "completion order: {done:?}");
        assert!(
            done.iter().all(|(_, status)| status.is_success()),
            "{done:?}"
        );
        assert_eq!(chat_requests(&primary).await, 2, "both waiters admitted");
        assert_eq!(chat_requests(&backup).await, 0, "no failover expected");
    })
    .await
    .expect("scenario finishes within the test budget");
}

// ---------------------------------------------------------------------------
// (b) Equal priorities keep arrival order
// ---------------------------------------------------------------------------

/// Two equal-priority (class 5) keys contend on a drained bucket; the
/// first-launched request parks first and must complete first (FIFO
/// tie-break). The 400 ms launch gap makes queue-arrival order
/// deterministic; only the completion ORDER is asserted, never wall clock.
#[tokio::test]
async fn equal_priority_keys_keep_arrival_order() {
    tokio::time::timeout(TEST_TIMEOUT, async {
        let primary = MockServer::start().await;
        let backup = MockServer::start().await;
        mount_ok_upstream(&primary).await;
        mount_ok_upstream(&backup).await;
        let server = GatewayServer::new(
            priority_config(&primary.uri(), Some(&backup.uri()), 30, 12_000),
            None,
        )
        .await
        .unwrap();
        let manager = server.state.virtual_key_manager.clone();
        let first = manager.create_key(key_params(5)).await.unwrap();
        let second = manager.create_key(key_params(5)).await.unwrap();
        drain_provider_tokens(&server, "primary").await;

        let done = dispatch_ordered(
            server.build_router(),
            vec![
                ("first", first.key.clone(), Duration::ZERO),
                ("second", second.key.clone(), LAUNCH_GAP),
            ],
        )
        .await;

        assert_eq!(done.len(), 2);
        assert_eq!(done[0].0, "first", "completion order: {done:?}");
        assert_eq!(done[1].0, "second", "completion order: {done:?}");
        assert!(
            done.iter().all(|(_, status)| status.is_success()),
            "{done:?}"
        );
        assert_eq!(chat_requests(&primary).await, 2);
        assert_eq!(chat_requests(&backup).await, 0);
    })
    .await
    .expect("scenario finishes within the test budget");
}

// ---------------------------------------------------------------------------
// (c) Low-priority waiter times out and fails over with a skip row
// ---------------------------------------------------------------------------

/// A priority-0 waiter is parked behind a priority-10 queue head on a
/// drained, phase-aligned 30/min bucket (one token every 2 s, phase set
/// 0.7 s past a boundary): the head's projected token (~1.3 s away, inside
/// the shared 2.4 s wait budget) is granted to the HEAD, and the waiter's
/// re-probe projects its next token ~2 s further out (~3.3 s absolute,
/// beyond its ~1.65 s remaining budget after the 0.75 s launch gap) — so
/// it must time out, skip the primary WITHOUT sending anything to it
/// (`rate_limit_skip` attempt row), and succeed on the backup. Budget
/// margins are >= 1 s on every decision.
#[tokio::test]
async fn low_priority_waiter_times_out_and_fails_over_with_rate_limit_skip_row() {
 tokio::time::timeout(TEST_TIMEOUT, async {
 let primary = MockServer::start().await;
 let backup = MockServer::start().await;
 mount_ok_upstream(&primary).await;
 mount_ok_upstream(&backup).await;
 let server = GatewayServer::new(
 priority_config(&primary.uri(), Some(&backup.uri()), 30, 2_400),
 None,
 )
 .await
 .unwrap();
 let manager = server.state.virtual_key_manager.clone();
 let low = manager.create_key(key_params(0)).await.unwrap();
 let high = manager.create_key(key_params(10)).await.unwrap();
 drain_provider_tokens(&server, "primary").await;
 // Park the bucket's phase at 0.7 s past a token boundary: the next
 // token (for the head) is 1.3 s away — well inside the 2.4 s
 // budget; the waiter's projection (~3.3 s) is well beyond its
 // ~1.65 s remaining budget.
 align_bucket_phase(&server, "primary", 30, Duration::from_millis(700)).await;

        // Head first: the high-priority request parks at the queue head;
        // the waiter only queues behind it after the launch gap.
        let app = server.build_router();
        let done = dispatch_ordered(
            app.clone(),
            vec![
                ("high", high.key.clone(), Duration::ZERO),
                ("low", low.key.clone(), LAUNCH_GAP),
            ],
        )
        .await;
        assert_eq!(done.len(), 2, "{done:?}");
        let low_entry = done
            .iter()
            .find(|(label, _)| *label == "low")
            .expect("low-priority request completed");
        assert!(low_entry.1.is_success(), "low fails over to backup: {done:?}");

        // The waiter times out (>= one 2 s refill cycle, well under the
        // server timeout) and nothing was ever sent to the primary by it.
        assert_eq!(
            chat_requests(&primary).await, 1,
            "only the priority-10 head reached the primary"
        );
        assert_eq!(chat_requests(&backup).await, 1);

        let rows = poll_logs(&server, Duration::from_secs(5), |rows| {
            rows.iter().any(|r| {
                r.path == OUTCOME_PATH && r.status_code == 200 && r.provider == "backup"
            })
        })
        .await;
        let skip = rows
            .iter()
            .find(|r| {
                r.path == ATTEMPT_PATH
                    && r.provider == "primary"
                    && r.error_class.as_deref() == Some("rate_limit_skip")
            })
            .expect("rate_limit_skip attempt row for the primary");
        assert_eq!(skip.status_code, 429);
    })
    .await
    .expect("scenario finishes within the test budget");
}

// ---------------------------------------------------------------------------
// (d) PATCHed priority applies to the next admission without restart
// ---------------------------------------------------------------------------

/// Two class-0 keys are created; one is PATCHed to class 50. The store and
/// the auth cache both serve 50 immediately, and the very next contended
/// admission orders the PATCHed key (launched SECOND) ahead of the
/// unpatched class-0 key (launched first).
#[tokio::test]
async fn patched_priority_applies_to_next_admission_without_restart() {
    tokio::time::timeout(TEST_TIMEOUT, async {
        let primary = MockServer::start().await;
        let backup = MockServer::start().await;
        mount_ok_upstream(&primary).await;
        mount_ok_upstream(&backup).await;
        let server = GatewayServer::new(
            priority_config(&primary.uri(), Some(&backup.uri()), 30, 12_000),
            None,
        )
        .await
        .unwrap();
        let manager = server.state.virtual_key_manager.clone();
        let patched = manager.create_key(key_params(0)).await.unwrap();
        let plain = manager.create_key(key_params(0)).await.unwrap();

        let info = manager
            .update_key(
                &patched.id,
                UpdateKeyParams {
                    priority: Some(Some(50)),
                    ..Default::default()
                },
            )
            .await
            .unwrap();
        assert_eq!(info.priority, 50, "stored priority after PATCH");
        assert_eq!(
            manager.authenticate(&patched.key).await.unwrap().priority,
            50,
            "auth cache must serve the PATCHed priority without restart"
        );

        drain_provider_tokens(&server, "primary").await;
        let done = dispatch_ordered(
            server.build_router(),
            vec![
                ("plain", plain.key.clone(), Duration::ZERO),
                ("patched", patched.key.clone(), LAUNCH_GAP),
            ],
        )
        .await;

        assert_eq!(done.len(), 2);
        assert_eq!(done[0].0, "patched", "completion order: {done:?}");
        assert_eq!(done[1].0, "plain", "completion order: {done:?}");
        assert!(
            done.iter().all(|(_, status)| status.is_success()),
            "{done:?}"
        );
        assert_eq!(chat_requests(&primary).await, 2);
        assert_eq!(chat_requests(&backup).await, 0);
    })
    .await
    .expect("scenario finishes within the test budget");
}

// ---------------------------------------------------------------------------
// (e) Exactly-one-token accounting under priority keys
// ---------------------------------------------------------------------------

/// ADR 0002 guard (mirror of `single_request_consumes_exactly_one_provider
/// _rate_token` in router.rs) with priority keys: a sequential and a
/// concurrent pair (classes 10 and 0) against a full 6/min bucket
/// (refill 0.1/s — negligible at test speed) must consume exactly one
/// token per upstream HTTP request, with no local skips and no failover.
#[tokio::test]
async fn exactly_one_token_per_request_with_priority_keys() {
    tokio::time::timeout(TEST_TIMEOUT, async {
        let primary = MockServer::start().await;
        let backup = MockServer::start().await;
        mount_ok_upstream(&primary).await;
        mount_ok_upstream(&backup).await;
        let server = GatewayServer::new(
            priority_config(&primary.uri(), Some(&backup.uri()), 6, 100),
            None,
        )
        .await
        .unwrap();
        let manager = server.state.virtual_key_manager.clone();
        let high = manager.create_key(key_params(10)).await.unwrap();
        let low = manager.create_key(key_params(0)).await.unwrap();

        let app = server.build_router();
        let before = tokens_left(&server, "primary").await;

        for (key, label) in [(&high, "seq-high"), (&low, "seq-low")] {
            let response = app
                .clone()
                .oneshot(chat_request(&key.key, label))
                .await
                .unwrap();
            assert_eq!(response.status(), StatusCode::OK, "{label}");
            let _ = axum::body::to_bytes(response.into_body(), 1024 * 1024)
                .await
                .unwrap();
        }

        let done = dispatch_ordered(
            app,
            vec![
                ("con-high", high.key.clone(), Duration::ZERO),
                ("con-low", low.key.clone(), Duration::ZERO),
            ],
        )
        .await;
        assert_eq!(done.len(), 2);
        assert!(
            done.iter().all(|(_, status)| status.is_success()),
            "{done:?}"
        );

        let sent = chat_requests(&primary).await;
        let after = tokens_left(&server, "primary").await;
        assert_eq!(sent, 4, "all four requests reach the primary");
        assert_eq!(
            (before - after).round() as usize,
            sent,
            "tokens consumed ({before:.3} -> {after:.3}) must equal upstream requests ({sent})"
        );
        assert_eq!(chat_requests(&backup).await, 0);

 let rows = poll_logs(&server, Duration::from_secs(5), |rows| {
 rows.iter()
 .filter(|r| r.path == OUTCOME_PATH && r.status_code == 200)
 .count()
 >= 4
 })
 .await;
 // Non-vacuous: the poll guarantees the four outcome rows, so the
 // skip-absence claim covers a log that provably contains this
 // test's requests.
 let outcomes = rows
 .iter()
 .filter(|r| r.path == OUTCOME_PATH && r.status_code == 200)
 .count();
 assert_eq!(outcomes, 4, "all four requests logged outcomes: {rows:?}");
 assert!(
 rows.iter()
 .all(|r| r.error_class.as_deref() != Some("rate_limit_skip")),
 "no local skip under a full bucket: {rows:?}"
 );
    })
    .await
    .expect("scenario finishes within the test budget");
}

// ---------------------------------------------------------------------------
// (f) Per-key RPM rejection is unchanged by priority
// ---------------------------------------------------------------------------

/// A key with `requests_per_minute = 1` and the maximum priority class
/// (1000) still gets its second rapid request rejected with 429 +
/// `Retry-After`: the admission priority class never bypasses the per-key
/// limiter (mirror of `rate_limit_returns_429_with_retry_after`).
#[tokio::test]
async fn per_key_rpm_rejection_is_independent_of_priority_class() {
    tokio::time::timeout(TEST_TIMEOUT, async {
        let primary = MockServer::start().await;
        mount_ok_upstream(&primary).await;
        let server = GatewayServer::new(
            priority_config(&primary.uri(), None, 60, 0),
            None,
        )
        .await
        .unwrap();
        let manager = server.state.virtual_key_manager.clone();
        let created = manager
            .create_key(CreateKeyParams {
                requests_per_minute: Some(1),
                priority: Some(1000),
                ..key_params(0)
            })
            .await
            .unwrap();

        let app = server.build_router();
        let first = app
            .clone()
            .oneshot(chat_request(&created.key, "rpm-first"))
            .await
            .unwrap();
        assert_eq!(first.status(), StatusCode::OK);
        let _ = axum::body::to_bytes(first.into_body(), 1024 * 1024)
            .await
            .unwrap();

        let second = app
            .oneshot(chat_request(&created.key, "rpm-second"))
            .await
            .unwrap();
        assert_eq!(
            second.status(),
            StatusCode::TOO_MANY_REQUESTS,
            "priority class must not bypass the per-key RPM limiter"
        );
        let retry_after = second
            .headers()
            .get(axum::http::header::RETRY_AFTER)
            .expect("Retry-After header must be present on rate-limit rejection");
        let secs: u64 = retry_after
            .to_str()
            .unwrap()
            .parse()
            .expect("Retry-After is integer seconds");
        assert!(secs >= 1, "Retry-After should be at least 1 second, got {secs}");
        let _ = axum::body::to_bytes(second.into_body(), 1024 * 1024)
            .await
            .unwrap();
    })
    .await
    .expect("scenario finishes within the test budget");
}
