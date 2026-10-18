//! Integration tests for the admin provider-budget and virtual-key spend
//! reset endpoints (persistent-budgets, Task 8).
//!
//! Covers:
//! - `GET /admin/providers/{name}/budget` (happy path + 404 unknown provider)
//! - `POST /admin/providers/{name}/budget/reset` (returns previous spend,
//!   clears exhausted state, zeroes store + metrics, preserves history)
//! - `POST /admin/keys/{id}/reset` (zeroes VK spend/tokens, resets
//!   window_start, preserves request_count/last_used_at/usage rows, 404)
//!
//! All requests go through the full admin router via
//! `tower::ServiceExt::oneshot()` — no live server. Databases are isolated
//! per test via `common::isolate_databases`.

use axum::body::Body;
use axum::http::{Request, StatusCode};
use tower::ServiceExt;

use ai_gateway::config::*;
use ai_gateway::gateway::GatewayServer;
use ai_gateway::virtual_keys::models::{CreateKeyParams, UsageRecord};

mod common;

const PROVIDER_NAME: &str = "budget-provider";

fn budget_test_config() -> Config {
    common::budget_test_config(
        None,
        Some(ProviderBudgetConfig {
            limit_usd: 50.0,
            reset_policy: Some(BudgetResetPolicy::Manual),
        }),
    )
}

async fn build_app(mut config: Config) -> (axum::Router, GatewayServer) {
    common::isolate_databases(&mut config);
    let server = GatewayServer::new(config, None).await.unwrap();
    let router = server.build_router();
    (router, server)
}

async fn send(app: axum::Router, req: Request<Body>) -> (StatusCode, serde_json::Value) {
    let resp = app.oneshot(req).await.unwrap();
    let status = resp.status();
    let bytes = axum::body::to_bytes(resp.into_body(), 1024 * 1024)
        .await
        .unwrap();
    let json = if bytes.is_empty() {
        serde_json::Value::Null
    } else {
        serde_json::from_slice(&bytes).unwrap()
    };
    (status, json)
}

// ---------------------------------------------------------------------------
// GET /admin/providers/{name}/budget
// ---------------------------------------------------------------------------

#[tokio::test]
async fn provider_budget_get_returns_state() {
    let (app, server) = build_app(budget_test_config()).await;
    // Accrue durable spend so the store row exists.
    let store = server
        .state
        .metrics
        .provider_budget_store()
        .expect("store attached");
    store
        .add_spend(PROVIDER_NAME, 12.5, chrono::Utc::now().timestamp())
        .unwrap();

    let req = Request::get(format!("/admin/providers/{PROVIDER_NAME}/budget"))
        .body(Body::empty())
        .unwrap();
    let (status, json) = send(app, req).await;

    assert_eq!(status, StatusCode::OK);
    assert_eq!(json["limit_usd"], 50.0);
    assert_eq!(json["spent_usd"], 12.5);
    assert_eq!(json["remaining_usd"], 37.5);
    assert_eq!(json["window"], "manual");
    assert_eq!(json["next_reset"], serde_json::Value::Null);
    assert_eq!(json["exhausted"], false);
}

#[tokio::test]
async fn provider_budget_get_marks_exhausted() {
    let (app, server) = build_app(budget_test_config()).await;
    let store = server
        .state
        .metrics
        .provider_budget_store()
        .expect("store attached");
    store
        .add_spend(PROVIDER_NAME, 55.0, chrono::Utc::now().timestamp())
        .unwrap();

    let req = Request::get(format!("/admin/providers/{PROVIDER_NAME}/budget"))
        .body(Body::empty())
        .unwrap();
    let (status, json) = send(app, req).await;

    assert_eq!(status, StatusCode::OK);
    assert_eq!(json["exhausted"], true);
    assert_eq!(json["remaining_usd"], 0.0);
}

#[tokio::test]
async fn provider_budget_get_unknown_provider_404() {
    let (app, _server) = build_app(budget_test_config()).await;
    let req = Request::get("/admin/providers/no-such-provider/budget")
        .body(Body::empty())
        .unwrap();
    let (status, json) = send(app, req).await;

    assert_eq!(status, StatusCode::NOT_FOUND);
    assert!(json["error"].is_object(), "GatewayError body shape: {json}");
}

// ---------------------------------------------------------------------------
// POST /admin/providers/{name}/budget/reset
// ---------------------------------------------------------------------------

#[tokio::test]
async fn provider_budget_reset_returns_previous_and_clears_exhausted() {
    let (app, server) = build_app(budget_test_config()).await;
    let store = server
        .state
        .metrics
        .provider_budget_store()
        .expect("store attached");
    store
        .add_spend(PROVIDER_NAME, 40.0, chrono::Utc::now().timestamp())
        .unwrap();
    // Mirror the router's exhaustion recording.
    server
        .state
        .metrics
        .record_provider_budget_exhausted(PROVIDER_NAME);
    // Seed the in-memory atomics too (write-through cache).
    server
        .state
        .metrics
        .restore_provider_budget_spend(PROVIDER_NAME, 40.0);

    let req = Request::post(format!("/admin/providers/{PROVIDER_NAME}/budget/reset"))
        .body(Body::empty())
        .unwrap();
    let (status, json) = send(app, req).await;

    assert_eq!(status, StatusCode::OK);
    assert_eq!(json["reset"], true);
    assert_eq!(json["provider"], PROVIDER_NAME);
    assert_eq!(json["previous_spend_usd"], 40.0);

    // Store row zeroed.
    let row = store
        .get_all()
        .unwrap()
        .into_iter()
        .find(|r| r.provider == PROVIDER_NAME)
        .expect("row preserved (spend-only, never deleted)");
    assert_eq!(row.current_spend_usd, 0.0);
    // Metrics atomics zeroed.
    assert_eq!(
        server
            .state
            .metrics
            .current_provider_cost_usd(PROVIDER_NAME),
        0.0
    );
    // Exhaustion marker cleared from the snapshot.
    let snap = server.state.metrics.snapshot();
    assert!(snap
        .budget_exhaustions_by_provider
        .iter()
        .all(|(p, _)| p != PROVIDER_NAME));
}

#[tokio::test]
async fn provider_budget_reset_unknown_provider_404() {
    let (app, _server) = build_app(budget_test_config()).await;
    let req = Request::post("/admin/providers/no-such-provider/budget/reset")
        .body(Body::empty())
        .unwrap();
    let (status, json) = send(app, req).await;

    assert_eq!(status, StatusCode::NOT_FOUND);
    assert!(json["error"].is_object(), "GatewayError body shape: {json}");
}

// ---------------------------------------------------------------------------
// C2/C1 remediation: known no-budget provider GET, fail-closed store reset
// ---------------------------------------------------------------------------

/// `budget-provider` with no budget block — known to config, no limit.
fn no_budget_test_config() -> Config {
    let mut config = budget_test_config();
    config.providers[0].budget = None;
    config
}

/// C4(b): GET on a KNOWN provider with no configured budget and no spend
/// row returns 200 with a zero-spend/no-limit payload (spend 0, limit
/// null, manual window, next_reset null) — not a 404.
#[tokio::test]
async fn provider_budget_get_known_no_budget_provider_returns_zero_state() {
    let (app, _server) = build_app(no_budget_test_config()).await;
    let req = Request::get(format!("/admin/providers/{PROVIDER_NAME}/budget"))
        .body(Body::empty())
        .unwrap();
    let (status, json) = send(app, req).await;

    assert_eq!(
        status,
        StatusCode::OK,
        "known provider with no budget row must return 200, not 404: {json}"
    );
    assert_eq!(json["provider"], PROVIDER_NAME);
    assert_eq!(json["spent_usd"], 0.0);
    assert!(
        json["limit_usd"].is_null(),
        "no configured limit must serialize as null: {json}"
    );
    assert!(
        json["remaining_usd"].is_null(),
        "no limit means no remaining value: {json}"
    );
    assert_eq!(json["window"], "manual");
    assert!(
        json["next_reset"].is_null(),
        "no window anchor means no next_reset: {json}"
    );
    assert_eq!(json["exhausted"], false);
}

/// C4(a): when the provider HAS a durable spend row and the store reset
/// fails (SQLite-level error), the reset endpoint must return a 500
/// `GatewayError::Database` and zero NOTHING — durable spend, in-memory
/// atomics, and the exhaustion marker all survive, so the spend cannot
/// silently resurrect after a restart.
///
/// The SQLite failure is injected deterministically: a second WAL-mode
/// connection takes the database write lock (`BEGIN IMMEDIATE`) and holds
/// it across the request, so the store's CAS `UPDATE` inside
/// `reset_window` exhausts its busy timeout and errors. (The alternative
/// — deleting the WAL-mode DB file — is unreliable under Windows' open
/// handle semantics, and the store's `Arc<Mutex<Connection>>` is not
/// reachable from tests for poisoning.)
#[tokio::test]
async fn provider_budget_reset_store_failure_is_fail_closed() {
    let dir = tempfile::tempdir().expect("temp dir").keep();
    let mut config = budget_test_config();
    // Inline `common::isolate_databases`, keeping the directory so the
    // sibling `budgets.db` path is known for the lock connection.
    config.logging.database_path = dir.join("logs.db").to_string_lossy().into_owned();
    config.virtual_keys.database_path = dir.join("keys.db").to_string_lossy().into_owned();
    let server = GatewayServer::new(config, None).await.unwrap();
    let app = server.build_router();

    let store = server
        .state
        .metrics
        .provider_budget_store()
        .expect("store attached");
    // The provider must HAVE a row for the failure to be fail-closed.
    store
        .add_spend(PROVIDER_NAME, 40.0, chrono::Utc::now().timestamp())
        .unwrap();
    // Seed in-memory state exactly like the router does on exhaustion.
    server
        .state
        .metrics
        .restore_provider_budget_spend(PROVIDER_NAME, 40.0);
    server
        .state
        .metrics
        .record_provider_budget_exhausted(PROVIDER_NAME);

    // Take the WAL write lock from a second connection and hold it for
    // the whole request. WAL readers (the store's SELECTs) still work;
    // only the write competes — exactly a store write failure.
    let lock_conn = rusqlite::Connection::open(dir.join("budgets.db")).expect("open budgets.db");
    lock_conn
        .execute_batch("BEGIN IMMEDIATE")
        .expect("acquire budgets.db write lock");

    let req = Request::post(format!("/admin/providers/{PROVIDER_NAME}/budget/reset"))
        .body(Body::empty())
        .unwrap();
    let (status, json) = send(app, req).await;

    assert_eq!(
        status,
        StatusCode::INTERNAL_SERVER_ERROR,
        "store failure must surface as a 5xx GatewayError, got {json}"
    );
    assert!(
        json["error"]["message"]
            .as_str()
            .unwrap_or_default()
            .contains("Failed to reset budget window"),
        "error message must name the failed reset: {json}"
    );

    // Fail-closed: NOTHING was zeroed.
    assert_eq!(
        server
            .state
            .metrics
            .current_provider_cost_usd(PROVIDER_NAME),
        40.0,
        "in-memory spend must survive a failed store reset"
    );
    let snap = server.state.metrics.snapshot();
    assert!(
        snap.budget_exhaustions_by_provider
            .iter()
            .any(|(p, _)| p == PROVIDER_NAME),
        "exhaustion marker must survive a failed store reset"
    );
    // Release the write lock, then verify the durable row survived.
    lock_conn
        .execute_batch("ROLLBACK")
        .expect("release budgets.db write lock");
    let row = store
        .get_all()
        .expect("store readable after lock release")
        .into_iter()
        .find(|r| r.provider == PROVIDER_NAME);
    assert_eq!(
        row.map(|r| r.current_spend_usd),
        Some(40.0),
        "durable spend must survive a failed store reset"
    );
}

// ---------------------------------------------------------------------------
// POST /admin/keys/{id}/reset
// ---------------------------------------------------------------------------

fn create_defaults() -> CreateKeyParams {
    CreateKeyParams {
        name: Some("reset-target".to_string()),
        budget_limit_usd: None,
        token_budget: None,
        budget_window: None,
        requests_per_minute: None,
        tokens_per_minute: None,
        model_access: None,
        expires_in: None,
        loop_detection: None,
    }
}

#[tokio::test]
async fn vk_reset_zeroes_spend_tokens_and_preserves_history() {
    let (app, server) = build_app(budget_test_config()).await;
    let manager = &server.state.virtual_key_manager;

    let created = manager.create_key(create_defaults()).await.unwrap();
    let key_id = created.id.clone();

    // Accrue usage: two rows + cumulative counters (spend, tokens,
    // request_count, last_used_at) exactly like the usage tracker does.
    for (input, output, cost) in [(10, 5, 0.001), (20, 7, 0.002)] {
        let record = UsageRecord {
            key_id: key_id.clone(),
            model_group: "budget-group".to_string(),
            model: "gpt-4.1-nano".to_string(),
            input_tokens: input,
            output_tokens: output,
            cost_usd: cost,
            timestamp: chrono::Utc::now(),
        };
        manager.record_usage(record).await.unwrap();
    }

    // Pre-state assertions (masked view carries the same counters).
    {
        let info = manager.get_key(&key_id).await.unwrap();
        assert_eq!(info.current_spend_usd, 0.003);
        assert_eq!(info.current_tokens_used, 42);
        assert_eq!(info.request_count, 2);
        assert!(info.last_used_at.is_some());
    }

    let req = Request::post(format!("/admin/keys/{key_id}/reset"))
        .body(Body::empty())
        .unwrap();
    let (status, json) = send(app, req).await;

    assert_eq!(status, StatusCode::OK);
    assert_eq!(json["reset"], true);
    assert_eq!(json["previous"]["spend_usd"], 0.003);
    assert_eq!(json["previous"]["tokens_used"], 42);

    // Post-state: counters zeroed, history intact.
    let info = manager.get_key(&key_id).await.unwrap();
    assert_eq!(info.current_spend_usd, 0.0);
    assert_eq!(info.current_tokens_used, 0);
    // PRESERVED: request_count, last_used_at, usage rows.
    assert_eq!(info.request_count, 2);
    assert!(info.last_used_at.is_some());
    let agg = manager
        .query_usage(
            &key_id,
            ai_gateway::virtual_keys::models::UsageQueryParams {
                start: chrono::DateTime::<chrono::Utc>::from_timestamp(0, 0).unwrap(),
                end: chrono::Utc::now() + chrono::Duration::days(1),
            },
        )
        .await
        .unwrap();
    assert_eq!(agg.total_requests, 2, "usage history rows preserved");
    assert_eq!(agg.total_spend_usd, 0.003);
}

/// Guard test (Task 10): the same VK reset semantics under the exact guard
/// name — a manual VK reset zeroes spend/tokens while usage history rows,
/// request_count, and last_used_at survive untouched.
#[tokio::test]
async fn vk_manual_reset_preserves_history() {
    let (app, server) = build_app(budget_test_config()).await;
    let manager = &server.state.virtual_key_manager;

    let created = manager.create_key(create_defaults()).await.unwrap();
    let key_id = created.id.clone();

    for (input, output, cost) in [(100, 40, 0.01), (5, 5, 0.005), (7, 3, 0.002)] {
        let record = UsageRecord {
            key_id: key_id.clone(),
            model_group: "budget-group".to_string(),
            model: "gpt-4.1-nano".to_string(),
            input_tokens: input,
            output_tokens: output,
            cost_usd: cost,
            timestamp: chrono::Utc::now(),
        };
        manager.record_usage(record).await.unwrap();
    }

    let req = Request::post(format!("/admin/keys/{key_id}/reset"))
        .body(Body::empty())
        .unwrap();
    let (status, json) = send(app, req).await;
    assert_eq!(status, StatusCode::OK);
    assert_eq!(json["reset"], true);

    // Counters zeroed; history preserved.
    let info = manager.get_key(&key_id).await.unwrap();
    assert_eq!(info.current_spend_usd, 0.0);
    assert_eq!(info.current_tokens_used, 0);
    assert_eq!(
        info.request_count, 3,
        "request_count is history, not a counter"
    );
    assert!(info.last_used_at.is_some(), "last_used_at preserved");
    let agg = manager
        .query_usage(
            &key_id,
            ai_gateway::virtual_keys::models::UsageQueryParams {
                start: chrono::DateTime::<chrono::Utc>::from_timestamp(0, 0).unwrap(),
                end: chrono::Utc::now() + chrono::Duration::days(1),
            },
        )
        .await
        .unwrap();
    assert_eq!(agg.total_requests, 3, "usage rows survive the manual reset");
    assert!((agg.total_spend_usd - 0.017).abs() < 1e-9);
}

#[tokio::test]
async fn vk_reset_unknown_key_404() {
    let (app, _server) = build_app(budget_test_config()).await;
    let req = Request::post("/admin/keys/no-such-id/reset")
        .body(Body::empty())
        .unwrap();
    let (status, json) = send(app, req).await;

    assert_eq!(status, StatusCode::NOT_FOUND);
    assert_eq!(json["error"], "Key not found");
}

// ---------------------------------------------------------------------------
// Guard (Task 10): POST budget/reset clears budget_skip on dispatch
// ---------------------------------------------------------------------------

/// Guard test (Task 10): after `POST /admin/providers/{name}/budget/reset`
/// on an exhausted (budget_skip'd) provider, a subsequent dispatch is NOT
/// skipped. Exercises the full router via oneshot against a wiremock
/// upstream — the same pattern as `budget_rollover.rs`, but driving the
/// skip through the real dispatch path both before and after the admin
/// reset.
#[tokio::test]
async fn manual_reset_clears_budget_skip() {
    use wiremock::matchers::{method as wm_method, path as wm_path};
    use wiremock::{Mock, MockServer, ResponseTemplate};

    let upstream = MockServer::start().await;
    let mut config = budget_test_config();
    config.providers[0].base_url = Some(format!("{}/v1", upstream.uri()));
    config.providers[0].api_key_env = None;
    common::isolate_databases(&mut config);
    let server = GatewayServer::new(config, None).await.unwrap();
    let app = server.build_router();

    let ok_body = serde_json::json!({
        "id": "chatcmpl-manual-reset-guard",
        "object": "chat.completion",
        "created": 1700000000_i64,
        "model": "gpt-4.1-nano",
        "choices": [{
            "index": 0,
            "message": {"role": "assistant", "content": "ok"},
            "finish_reason": "stop"
        }],
        "usage": {
            // 100k + 100k @ $30/$60 per million = $9.00 — enough to
            // attribute cost without re-exhausting the $50 limit.
            "prompt_tokens": 100_000,
            "completion_tokens": 100_000,
            "total_tokens": 200_000
        }
    });
    Mock::given(wm_method("POST"))
        .and(wm_path("/v1/chat/completions"))
        .respond_with(ResponseTemplate::new(200).set_body_json(ok_body))
        .expect(1)
        .mount(&upstream)
        .await;

    // Exhaust: $55 of durable + in-memory spend against the $50 limit;
    // the dispatch below records the budget_skip itself through the real
    // router path.
    let store = server
        .state
        .metrics
        .provider_budget_store()
        .expect("store attached");
    store
        .add_spend(PROVIDER_NAME, 55.0, chrono::Utc::now().timestamp())
        .unwrap();
    server.state.metrics.add_cost(PROVIDER_NAME, 55.0);

    let skip_req = Request::post("/v1/chat/completions")
        .header("content-type", "application/json")
        .body(Body::from(
            serde_json::to_string(&serde_json::json!({
                "model": "budget-group",
                "messages": [{"role": "user", "content": "pre-reset"}]
            }))
            .unwrap(),
        ))
        .unwrap();
    let resp = app.clone().oneshot(skip_req).await.unwrap();
    assert_eq!(
        resp.status(),
        StatusCode::BAD_GATEWAY,
        "exhausted provider must budget_skip before the reset"
    );
    let snap = server.state.metrics.snapshot();
    let skips_before = snap
        .budget_exhaustions_by_provider
        .iter()
        .find(|(provider, _)| provider == PROVIDER_NAME)
        .map(|(_, count)| *count)
        .unwrap_or(0);
    assert_eq!(skips_before, 1, "pre-reset dispatch must budget_skip once");

    // Manual reset through the admin endpoint.
    let reset_req = Request::post(format!("/admin/providers/{PROVIDER_NAME}/budget/reset"))
        .body(Body::empty())
        .unwrap();
    let (status, json) = send(app.clone(), reset_req).await;
    assert_eq!(status, StatusCode::OK);
    assert_eq!(json["reset"], true);

    // Post-reset dispatch is NOT skipped: spend zeroed in both truths, the
    // request reaches the upstream mock (expect(1) asserts exactly one
    // upstream hit on drop).
    let dispatch_req = Request::post("/v1/chat/completions")
        .header("content-type", "application/json")
        .body(Body::from(
            serde_json::to_string(&serde_json::json!({
                "model": "budget-group",
                "messages": [{"role": "user", "content": "post-reset"}]
            }))
            .unwrap(),
        ))
        .unwrap();
    let resp = app.oneshot(dispatch_req).await.unwrap();
    assert_eq!(
        resp.status(),
        StatusCode::OK,
        "manual budget reset must clear the budget_skip for later dispatches"
    );

    let in_memory = server
        .state
        .metrics
        .current_provider_cost_usd(PROVIDER_NAME);
    assert!(
        (in_memory - 9.0).abs() < 1e-6,
        "post-reset spend must be exactly the $9.00 post-reset request, got {in_memory}"
    );
}

// ---------------------------------------------------------------------------
// Guard: POST /admin/keys/{id}/reset invalidates the cached auth entry
// ---------------------------------------------------------------------------

/// `POST /admin/keys/{id}/reset` calls `invalidate_cache`, so a subsequent
/// request with the same key is budget-checked against the zeroed store
/// counters instead of a stale cached snapshot: a budgeted VK driven past
/// its limit (cached with the exhausted counters by an intermediate
/// request) passes again after the admin reset.
#[tokio::test]
async fn vk_reset_invalidates_cached_auth_state() {
    use wiremock::matchers::{method as wm_method, path as wm_path};
    use wiremock::{Mock, MockServer, ResponseTemplate};

    let upstream = MockServer::start().await;
    let mut config = budget_test_config();
    config.providers[0].base_url = Some(format!("{}/v1", upstream.uri()));
    config.virtual_keys.enforcement = ai_gateway::config::EnforcementMode::Required;
    common::isolate_databases(&mut config);
    let server = GatewayServer::new(config, None).await.unwrap();
    let app = server.build_router();

    Mock::given(wm_method("POST"))
        .and(wm_path("/v1/chat/completions"))
        .respond_with(ResponseTemplate::new(200).set_body_json(serde_json::json!({
            "id": "chatcmpl-vk-cache-reset",
            "object": "chat.completion",
            "created": 1700000000_i64,
            "model": "gpt-4.1-nano",
            "choices": [{
                "index": 0,
                "message": {"role": "assistant", "content": "ok"},
                "finish_reason": "stop"
            }],
            "usage": {
                "prompt_tokens": 100_000,
                "completion_tokens": 100_000,
                "total_tokens": 200_000
            }
        })))
        .expect(1)
        .mount(&upstream)
        .await;

    // Budgeted VK with a $1.00 limit.
    let manager = &server.state.virtual_key_manager;
    let created = manager
        .create_key(CreateKeyParams {
            budget_limit_usd: Some(1.0),
            ..create_defaults()
        })
        .await
        .unwrap();

    let chat = |content: &str| {
        Request::post("/v1/chat/completions")
            .header("content-type", "application/json")
            .header("authorization", format!("Bearer {}", created.key))
            .body(Body::from(
                serde_json::to_string(&serde_json::json!({
                    "model": "budget-group",
                    "messages": [{"role": "user", "content": content}]
                }))
                .unwrap(),
            ))
            .unwrap()
    };

    // Seed exhausted store counters and warm the auth cache with them via
    // a real request — no network hit: the middleware rejects before
    // dispatch, and the cached snapshot now carries the exhausted spend.
    manager
        .record_usage(UsageRecord {
            key_id: created.id.clone(),
            model_group: "budget-group".to_string(),
            model: "gpt-4.1-nano".to_string(),
            input_tokens: 1_000,
            output_tokens: 1_000,
            cost_usd: 2.0,
            timestamp: chrono::Utc::now(),
        })
        .await
        .unwrap();
    let resp = app.clone().oneshot(chat("cached-exhausted")).await.unwrap();
    assert_eq!(
        resp.status(),
        StatusCode::TOO_MANY_REQUESTS,
        "exhausted cached counters must 429 before the reset"
    );

    // Admin reset zeroes the store and drops the cached entry.
    let reset_req = Request::post(format!("/admin/keys/{}/reset", created.id))
        .body(Body::empty())
        .unwrap();
    let (status, json) = send(app.clone(), reset_req).await;
    assert_eq!(status, StatusCode::OK);
    assert_eq!(json["reset"], true);
    assert_eq!(json["previous"]["spend_usd"], 2.0);

    // The next authorized request must NOT be blocked by the stale cached
    // spend: it reaches the upstream mock (expect(1) asserts the hit).
    let resp = app.oneshot(chat("post-reset")).await.unwrap();
    assert_eq!(
        resp.status(),
        StatusCode::OK,
        "cached auth entry must be invalidated by the VK reset, got {}",
        resp.status()
    );
}
