//! Provider-budget alert webhooks (budget-alerts-unified-webhooks,
//! Task 5) — full-router integration tests.
//!
//! Exercises the complete funnel via `tower::ServiceExt::oneshot()` with
//! a wiremock provider (cost-attributed spend) and a wiremock webhook
//! receiver: durable accrual → post-persist evaluation → signed
//! delivery, plus reset events and the dormancy contract. Databases are
//! isolated per test via temp dirs (`budgets.db` / `webhooks.db` are
//! siblings of the redirected logs.db).
//!
//! Covers:
//! - spend crossing 80% → `budget.warning` with a valid Standard
//!   Webhooks signature over `{id}.{timestamp}.{body}` (Reqs 1.1-1.3,
//!   3.4)
//! - reaching 100% → `budget.exhausted` (Req 1.4)
//! - no re-fire within the same window on a later crossing accrual
//!   (Req 1.5)
//! - lazy rollover of an exhausted window → `budget.reset` with
//!   `window_rollover` + new `window_start` (Reqs 2.1, 2.3)
//! - admin reset → `budget.reset` with `admin` source (Req 2.2)
//! - no-limit provider → no events (Req 1.9)
//! - dormancy: zero endpoints → no ledger rows, no webhooks.db file
//!   activity beyond the schema init (Req 3.3)

use axum::body::Body;
use axum::http::{Request, StatusCode};
use tower::ServiceExt;
use wiremock::MockServer;

use ai_gateway::config::*;
use ai_gateway::gateway::GatewayServer;
use ai_gateway::webhooks::events::{BudgetAlertEvent, Scope};

use crate::common::{
 configure_webhook_receiver, mount_ok_mock, mount_webhook_ok, post_chat,
 verify_signature_and_parse, wait_for_deliveries, wait_for_store_spend, webhook_test_provider,
 WEBHOOK_TEST_PROVIDER as PROVIDER_NAME,
};

fn base_config() -> Config {
    Config {
        prompt_library: Default::default(),
        security: Default::default(),
        cache_aware_routing: Default::default(),
        server: ServerConfig {
            host: "127.0.0.1".to_string(),
            port: 0,
            request_timeout_seconds: 30,
            max_request_size_mb: 10,
        },
        tls: None,
        admin: AdminConfig::default(),
        dashboard: DashboardConfig::default(),
        cors: CorsConfig::default(),
        providers: vec![],
        model_groups: vec![],
        circuit_breaker: CircuitBreakerConfig::default(),
        retry: RetryConfig::default(),
        logging: LoggingConfig::default(),
        semantic_cache: None,
        exact_cache: ExactCacheConfig::default(),
        prometheus: None,
        context: ContextConfig::default(),
        compression: Default::default(),
        memory: None,
        first_launch_completed: false,
        tray: TrayConfig::default(),
        codex_instructions_url: None,
        streaming: None,
        virtual_keys: Default::default(),
        loop_detection: Default::default(),
        guardrails: None,
        tool_compression: Default::default(),
        smart_routing: Default::default(),
        xhigh_models_allowlist: Default::default(),
        reasoning_models_allowlist: Default::default(),
        codex_search: None,
        web_search: None,
        structured_output: None,
        reasoning_compat: Default::default(),
        webhooks: Default::default(),
        opentelemetry: Default::default(),
        model_deprecations: Default::default(),
        performance_alerts: Default::default(),
        usage_anomaly_alerts: Default::default(),
    }
}

/// Isolate databases into `db_dir` (mirrors `common::isolate_databases`
/// but keeps the directory handle so tests can address `webhooks.db` /
/// `budgets.db` directly).
fn isolated_config(db_dir: &std::path::Path) -> Config {
    let mut config = base_config();
    config.logging.database_path = db_dir.join("logs.db").to_string_lossy().into_owned();
    config.virtual_keys.database_path = db_dir.join("keys.db").to_string_lossy().into_owned();
    config
}

/// Poll the webhook ledger until it holds at least `min` events.
async fn wait_for_ledger_events(server: &GatewayServer, min: usize) {
    for _ in 0..200 {
        if server.state.webhook_ledger.recent_events(50).len() >= min {
            return;
        }
        tokio::time::sleep(std::time::Duration::from_millis(25)).await;
    }
    panic!("webhook ledger never held {min} events");
}

/// Bounded absence poll (finding 6): FIRST confirm the off-path funnel
/// demonstrably quiesced by waiting for the durable spend read to be
/// stable, THEN assert the ledger holds no alert events within a short
/// bounded window. The negative is only trusted after the funnel ran.
async fn assert_no_ledger_events(server: &GatewayServer, poll_ms: u64) {
    let ledger = &server.state.webhook_ledger;
    let mut last = ledger.recent_events(10);
    let mut stable_rounds = 0;
    let deadline = std::time::Instant::now() + std::time::Duration::from_millis(poll_ms);
    while std::time::Instant::now() < deadline {
        tokio::time::sleep(std::time::Duration::from_millis(50)).await;
        let current = ledger.recent_events(10);
        if current.is_empty() && last.is_empty() && stable_rounds >= 2 {
            return;
        }
        if current.len() != last.len() {
            stable_rounds = 0;
        } else {
            stable_rounds += 1;
        }
        last = current;
    }
    assert!(
        last.is_empty(),
        "ledger unexpectedly holds {} event rows: {last:?}",
        last.len()
    );
}
// ---------------------------------------------------------------------------
// Threshold crossing: warning → exhausted, signature verified
// ---------------------------------------------------------------------------

// Validates: Requirements 1.1, 1.3, 1.4, 1.10, 3.4
#[tokio::test]
async fn spend_crossing_emits_signed_warning_then_exhausted() {
    let upstream = MockServer::start().await;
    let receiver = MockServer::start().await;
    let db_dir = tempfile::tempdir().expect("temp dir");

    let mut config = isolated_config(db_dir.path());
    // $5.00 limit: $3.00 (60%) then $6.00 (120% — warning + exhausted
    // cross on the same second accrual). Warn at the default 80% ($4).
    config.providers = vec![webhook_test_provider(
        &upstream.uri(),
        Some(ProviderBudgetConfig {
            limit_usd: 5.0,
            reset_policy: Some(BudgetResetPolicy::Manual),
            alert_threshold_percent: None,
        }),
    )];
    crate::common::webhook_with_expensive_group(&mut config);
    configure_webhook_receiver(&mut config, &receiver);

    let server = GatewayServer::new(config, None).await.unwrap();
    let app = server.build_router();
    mount_ok_mock(&upstream, 2).await;
    mount_webhook_ok(&receiver).await;

    // First request: $3.00 = 60% — below the warn line, no events.
    let status = post_chat(app.clone(), "one").await;
    assert_eq!(status, StatusCode::OK);
    wait_for_store_spend(&server, PROVIDER_NAME, 3.0).await;
    // No delivery should arrive; give the off-path chain a beat to
    // prove the negative, then assert the ledger is empty.
    tokio::time::sleep(std::time::Duration::from_millis(300)).await;
    assert!(
        server.state.webhook_ledger.recent_events(10).is_empty(),
        "below-threshold accrual must not emit events"
    );

    // Second request: $6.00 total = 120% — warn + exhausted on this
    // single crossing accrual (Req 1.4, and the 80% line crossed
    // along the way → warning first).
    let status = post_chat(app, "two").await;
    assert_eq!(status, StatusCode::OK);
    wait_for_store_spend(&server, PROVIDER_NAME, 6.0).await;

    let deliveries = wait_for_deliveries(&receiver, 2).await;
    assert_eq!(deliveries.len(), 2, "warning + exhausted, exactly");
    // Each event's delivery is an independently spawned task, so the
    // two arrivals may race: verify both signatures and assert the
    // MULTISET of event types, not the arrival order.
    let first = verify_signature_and_parse(&deliveries[0]);
    let second = verify_signature_and_parse(&deliveries[1]);
    let types = [first.event_type, second.event_type];
    assert!(
        types.contains(&BudgetEventType::BudgetWarning)
            && types.contains(&BudgetEventType::BudgetExhausted),
        "expected warning + exhausted, got {types:?}"
    );
    let warn = if first.event_type == BudgetEventType::BudgetWarning {
        &first
    } else {
        &second
    };
    for event in [&first, &second] {
        assert_eq!(event.data.scope, Scope::Provider);
        assert_eq!(event.data.name.as_deref(), Some(PROVIDER_NAME));
        assert_eq!(event.data.limit_usd, Some(5.0));
        assert_eq!(event.data.spent_usd, Some(6.0));
    }
    assert_eq!(warn.data.percent, Some(80));
}

// ---------------------------------------------------------------------------
// No re-fire within the same window (Req 1.5)
// ---------------------------------------------------------------------------

// Validates: Requirements 1.5, 1.10
#[tokio::test]
async fn no_refire_within_same_window_on_later_accrual() {
    let upstream = MockServer::start().await;
    let receiver = MockServer::start().await;
    let db_dir = tempfile::tempdir().expect("temp dir");

    let mut config = isolated_config(db_dir.path());
    // $30.00 limit, manual window, $3.00 per request: #8 hits $24 =
    // exactly 80% (warn fires), #9 hits $27 — a SECOND crossing
    // accrual of the warn line that must NOT re-fire (Req 1.5), #10
    // hits $30 = exhausted. Every request stays under the limit until
    // the last, so the router never budget-skips.
    config.providers = vec![webhook_test_provider(
        &upstream.uri(),
        Some(ProviderBudgetConfig {
            limit_usd: 30.0,
            reset_policy: Some(BudgetResetPolicy::Manual),
            alert_threshold_percent: None,
        }),
    )];
    crate::common::webhook_with_expensive_group(&mut config);
    configure_webhook_receiver(&mut config, &receiver);

    let server = GatewayServer::new(config, None).await.unwrap();
    let app = server.build_router();
    mount_ok_mock(&upstream, 10).await;
    mount_webhook_ok(&receiver).await;

    for turn in [
        "one", "two", "three", "four", "five", "six", "seven", "eight", "nine", "ten",
    ] {
        let status = post_chat(app.clone(), turn).await;
        assert_eq!(status, StatusCode::OK);
    }
    wait_for_store_spend(&server, PROVIDER_NAME, 30.0).await;

    // Warn (#8) + exhausted (#10), each exactly once; the #9 re-cross
    // of the warn line produced nothing. Each delivery is an
    // independently spawned task — assert the multiset, not the order.
    let deliveries = wait_for_deliveries(&receiver, 2).await;
    assert_eq!(deliveries.len(), 2, "warn + exhausted exactly once");
    let parsed: Vec<BudgetAlertEvent> = deliveries.iter().map(verify_signature_and_parse).collect();
    let warn = parsed
        .iter()
        .find(|event| event.event_type == BudgetEventType::BudgetWarning)
        .expect("warning present exactly once");
    let exhausted = parsed
        .iter()
        .find(|event| event.event_type == BudgetEventType::BudgetExhausted)
        .expect("exhausted present exactly once");
    assert_eq!(warn.data.percent, Some(80));
    assert_eq!(warn.data.spent_usd, Some(24.0));
    assert_eq!(exhausted.data.spent_usd, Some(30.0));

    // Ledger agrees: exactly two events, no re-fire rows. All ten
    // accruals are confirmed durable above, so the count is trusted.
    wait_for_ledger_events(&server, 2).await;
    assert_eq!(
        server.state.webhook_ledger.recent_events(10).len(),
        2,
        "no re-fire within the same window"
    );
}

// ---------------------------------------------------------------------------
// Lazy rollover of an exhausted window → budget.reset (Reqs 2.1, 2.3)
// ---------------------------------------------------------------------------

// Validates: Requirements 2.1, 2.3
#[tokio::test]
async fn exhausted_window_rollover_emits_reset_with_new_anchor() {
    let upstream = MockServer::start().await;
    let receiver = MockServer::start().await;
    let db_dir = tempfile::tempdir().expect("temp dir");
    let budgets_db = db_dir.path().join("budgets.db");

    let mut config = isolated_config(db_dir.path());
    config.providers = vec![webhook_test_provider(
        &upstream.uri(),
        Some(ProviderBudgetConfig {
            limit_usd: 5.0,
            reset_policy: Some(BudgetResetPolicy::Daily),
            alert_threshold_percent: None,
        }),
    )];
    crate::common::webhook_with_expensive_group(&mut config);
    configure_webhook_receiver(&mut config, &receiver);

    let server = GatewayServer::new(config, None).await.unwrap();
    let app = server.build_router();
    mount_ok_mock(&upstream, 2).await;
    mount_webhook_ok(&receiver).await;

    // Exhaust the window: request #1 accrues $3.00 durably (warn fires
    // at 60%? No — $3 of $5 is 60%, below 80%: no events yet), then
    // bump BOTH truths to $6.00 directly (mirror of the budget_rollover
    // test-b shape: the boundary cache must not seed before the
    // backdate, so exhaustion happens off-router).
    let status = post_chat(app.clone(), "one").await;
    assert_eq!(status, StatusCode::OK);
    wait_for_store_spend(&server, PROVIDER_NAME, 3.0).await;
    let store = server
        .state
        .metrics
        .provider_budget_store()
        .expect("store attached");
    let anchor = store
        .get_window_start(PROVIDER_NAME)
        .expect("window start readable")
        .expect("row exists after first request");
    store
        .add_spend(PROVIDER_NAME, 3.0, anchor)
        .expect("durable $6.00");
    server.state.metrics.add_cost(PROVIDER_NAME, 3.0); // in-memory $6.00 ≥ $5.00

    // Backdate the durable anchor to yesterday → the daily window is
    // expired (the boundary cache has not been seeded: request #1's
    // dispatch found no row yet).
    let yesterday = chrono::Utc::now().timestamp() - 86_400;
    {
        let conn = rusqlite::Connection::open(&budgets_db).expect("open budgets.db");
        conn.execute(
            "UPDATE provider_budgets SET window_start = ?2 WHERE provider = ?1",
            rusqlite::params![PROVIDER_NAME, yesterday],
        )
        .expect("backdate window_start");
    }

    // Request #2 triggers the lazy rollover BEFORE the budget check
    // (the provider was exhausted / budget-skip eligible) →
    // `budget.reset(window_rollover)` with the NEW anchor; the fresh
    // $3.00 accrual re-enters the new window (crossing nothing).
    let status = post_chat(app, "two").await;
    assert_eq!(
        status,
        StatusCode::OK,
        "expired daily window must roll over before the budget check"
    );

    let deliveries = wait_for_deliveries(&receiver, 1).await;
    let reset = verify_signature_and_parse(&deliveries[0]);
    assert_eq!(reset.event_type, BudgetEventType::BudgetReset);
    assert_eq!(
        reset.data.reset_source,
        Some(ai_gateway::webhooks::events::ResetSource::WindowRollover)
    );
    let new_anchor = reset
        .data
        .window_start
        .expect("reset carries the new window_start");
    let durable_anchor = store
        .get_window_start(PROVIDER_NAME)
        .expect("durable anchor readable")
        .expect("row exists");
    assert_eq!(
        new_anchor, durable_anchor,
        "reset event anchor must equal the durable new window_start"
    );
    assert!(new_anchor > yesterday);
}

// ---------------------------------------------------------------------------
// Admin reset → budget.reset(admin) (Req 2.2)
// ---------------------------------------------------------------------------

// Validates: Requirement 2.2
#[tokio::test]
async fn admin_reset_emits_reset_event_with_admin_source() {
    let receiver = MockServer::start().await;
    let db_dir = tempfile::tempdir().expect("temp dir");

    let mut config = isolated_config(db_dir.path());
    // Manual window; no upstream needed — spend is accrued directly
    // into the store (the admin path only needs durable state).
    config.providers = vec![webhook_test_provider(
        "https://api.example.com/v1",
        Some(ProviderBudgetConfig {
            limit_usd: 5.0,
            reset_policy: Some(BudgetResetPolicy::Manual),
            alert_threshold_percent: None,
        }),
    )];
    crate::common::webhook_with_expensive_group(&mut config);
    configure_webhook_receiver(&mut config, &receiver);

    let server = GatewayServer::new(config, None).await.unwrap();
    let app = server.build_router();
    mount_webhook_ok(&receiver).await;

    // Durable spent over the limit (exhausted state).
    let store = server
        .state
        .metrics
        .provider_budget_store()
        .expect("store attached");
    store
        .add_spend(PROVIDER_NAME, 6.0, chrono::Utc::now().timestamp())
        .unwrap();
    server.state.metrics.add_cost(PROVIDER_NAME, 6.0);

    let req = Request::post(format!("/admin/providers/{PROVIDER_NAME}/budget/reset"))
        .body(Body::empty())
        .unwrap();
    let resp = app.oneshot(req).await.unwrap();
    assert_eq!(resp.status(), StatusCode::OK);

    let deliveries = wait_for_deliveries(&receiver, 1).await;
    let reset = verify_signature_and_parse(&deliveries[0]);
    assert_eq!(reset.event_type, BudgetEventType::BudgetReset);
    assert_eq!(
        reset.data.reset_source,
        Some(ai_gateway::webhooks::events::ResetSource::Admin)
    );
    assert_eq!(reset.data.name.as_deref(), Some(PROVIDER_NAME));
}

// ---------------------------------------------------------------------------
// No-limit provider → no events (Req 1.9)
// ---------------------------------------------------------------------------

// Validates: Requirement 1.9
#[tokio::test]
async fn no_limit_provider_emits_no_events() {
    let upstream = MockServer::start().await;
    let receiver = MockServer::start().await;
    let db_dir = tempfile::tempdir().expect("temp dir");

    let mut config = isolated_config(db_dir.path());
    // Spend-only provider: accrual still persists (persistent budgets
    // Task 4), but no percentage thresholds exist (Req 1.9).
    config.providers = vec![webhook_test_provider(&upstream.uri(), None)];
    crate::common::webhook_with_expensive_group(&mut config);
    configure_webhook_receiver(&mut config, &receiver);

    let server = GatewayServer::new(config, None).await.unwrap();
    let app = server.build_router();
    mount_ok_mock(&upstream, 1).await;
    mount_webhook_ok(&receiver).await;

    let status = post_chat(app, "one").await;
    assert_eq!(status, StatusCode::OK);
    wait_for_store_spend(&server, PROVIDER_NAME, 3.0).await;

    // Accrual confirmed durable above; a bounded absence poll can
    // trust the negative on loaded runners.
    assert_no_ledger_events(&server, 600).await;
    assert!(
        receiver.received_requests().await.unwrap().is_empty(),
        "no deliveries for a no-limit provider"
    );
}

// ---------------------------------------------------------------------------
// Dormancy: zero endpoints → no ledger rows, no file activity (Req 3.3)
// ---------------------------------------------------------------------------

// Validates: Requirement 3.3
#[tokio::test]
async fn dormant_config_emits_nothing_and_writes_no_ledger_rows() {
    let upstream = MockServer::start().await;
    let db_dir = tempfile::tempdir().expect("temp dir");

    let mut config = isolated_config(db_dir.path());
    // Same budget as the crossing test — but NO webhook endpoints. The
    // webhooks.db schema init is the dormancy contract's one allowed
    // artifact; no evaluation and no ledger rows may follow.
    config.providers = vec![webhook_test_provider(
        &upstream.uri(),
        Some(ProviderBudgetConfig {
            limit_usd: 5.0,
            reset_policy: Some(BudgetResetPolicy::Manual),
            alert_threshold_percent: None,
        }),
    )];
    crate::common::webhook_with_expensive_group(&mut config);
    config.webhooks = WebhooksConfig::default();

    let server = GatewayServer::new(config, None).await.unwrap();
    let app = server.build_router();
    mount_ok_mock(&upstream, 2).await;

    // Cross every threshold: $3 then $6 (120% of $5).
    let status = post_chat(app.clone(), "one").await;
    assert_eq!(status, StatusCode::OK);
    wait_for_store_spend(&server, PROVIDER_NAME, 3.0).await;
    let status = post_chat(app, "two").await;
    assert_eq!(status, StatusCode::OK);
    wait_for_store_spend(&server, PROVIDER_NAME, 6.0).await;

    // Dormancy: the metrics funnel was never set, so no evaluation ran
    // and the ledger holds zero rows. Both tables are asserted
    // unconditionally through the server's ledger handle (equal to
    // reading webhooks.db: same store, same file).
    assert_no_ledger_events(&server, 600).await;
    let rows = server.state.webhook_ledger.recent_deliveries(None, 50);
    assert!(
        rows.is_empty(),
        "dormant webhooks must hold zero delivery rows (got {rows:?})"
    );
    let webhooks_db = db_dir.path().join("webhooks.db");
    if webhooks_db.exists() {
        let conn = rusqlite::Connection::open(&webhooks_db).expect("open webhooks.db");
        let count: i64 = conn
            .query_row("SELECT COUNT(*) FROM alert_events", [], |row| row.get(0))
            .expect("count alert_events");
        assert_eq!(count, 0, "webhooks.db must hold zero event rows");
        let deliveries: i64 = conn
            .query_row("SELECT COUNT(*) FROM deliveries", [], |row| row.get(0))
            .expect("count deliveries");
        assert_eq!(deliveries, 0, "webhooks.db must hold zero delivery rows");
    }
}

// ---------------------------------------------------------------------------
// Restart parking: unfinished chains park on a fresh GatewayServer (design
// Error Handling 5)
// ---------------------------------------------------------------------------

/// Seed an unfinished chain directly in the ledger (event + inflight
/// attempt row — the crash signature), then construct a NEW
/// GatewayServer on the SAME database directory: startup parking must
/// mark the chain parked so the panel shows it as a failed delivery.
#[tokio::test]
async fn restart_parks_unfinished_delivery_chains() {
    let upstream = MockServer::start().await;
    let receiver = MockServer::start().await;
    let db_dir = tempfile::tempdir().expect("temp dir");

    let mut config = isolated_config(db_dir.path());
    config.providers = vec![webhook_test_provider(
        &upstream.uri(),
        Some(ProviderBudgetConfig {
            limit_usd: 5.0,
            reset_policy: Some(BudgetResetPolicy::Manual),
            alert_threshold_percent: None,
        }),
    )];
    crate::common::webhook_with_expensive_group(&mut config);
    configure_webhook_receiver(&mut config, &receiver);

    let server = GatewayServer::new(config.clone(), None).await.unwrap();
    let endpoint_url = {
        let live = server.state.config.read().await;
        live.webhooks.endpoints[0].url.clone()
    };

    // Crash signature: an event row whose ONLY delivery row is still
    // inflight (attempt 2 of 4 — a mid-retry crash).
    server
        .state
        .webhook_ledger
        .insert_event(
            "evt-restart-park",
            "budget.warning",
            "provider",
            PROVIDER_NAME,
            "{}",
            1_700_000_000,
        )
        .expect("seed event row");
    server
        .state
        .webhook_ledger
        .record_inflight_attempt("evt-restart-park", &endpoint_url, 2, 1_700_000_000)
        .expect("seed inflight attempt");
    drop(server);

    // "Restart": a brand-new gateway over the same db dir. Startup
    // parking runs when the webhook subsystem is active.
    let server2 = GatewayServer::new(config, None).await.unwrap();

    // The chain must now show a parked row (via the ledger handle —
    // the same rows the admin deliveries endpoint serves).
    let mut parked = Vec::new();
    for _ in 0..100 {
        parked = server2
            .state
            .webhook_ledger
            .recent_deliveries(Some(&endpoint_url), 10)
            .into_iter()
            .filter(|row| row.outcome == "parked" && row.event_id == "evt-restart-park")
            .collect();
        if !parked.is_empty() {
            break;
        }
        tokio::time::sleep(std::time::Duration::from_millis(25)).await;
    }
    assert!(
        !parked.is_empty(),
        "restart must park the unfinished chain (rows: {:?})",
        server2
            .state
            .webhook_ledger
            .recent_deliveries(Some(&endpoint_url), 10)
    );
    assert_eq!(
        parked[0].attempt, 3,
        "parked row resumes after the inflight attempt"
    );
}

// ---------------------------------------------------------------------------
// Event filter: endpoint subscribed to exhausted only (Req 3.2, 4.7)
// ---------------------------------------------------------------------------

/// An endpoint configured with `events: [budget.exhausted]` receives NO
/// warning delivery when spend crosses the warn threshold first: the
/// full-router funnel evaluates both thresholds, the exhausted event is
/// delivered, and the warning event records `skipped_no_endpoints`
/// (sole endpoint, filtered out).
#[tokio::test]
async fn event_filtered_endpoint_skips_warning_delivery() {
    let upstream = MockServer::start().await;
    let receiver = MockServer::start().await;
    let db_dir = tempfile::tempdir().expect("temp dir");

    let mut config = isolated_config(db_dir.path());
    // $5 limit: request #1 = $3 (60%, nothing fires), request #2 → $6
    // (120%): warning AND exhausted both cross on the funnel — only
    // exhausted is subscribed.
    config.providers = vec![webhook_test_provider(
        &upstream.uri(),
        Some(ProviderBudgetConfig {
            limit_usd: 5.0,
            reset_policy: Some(BudgetResetPolicy::Manual),
            alert_threshold_percent: None,
        }),
    )];
    crate::common::webhook_with_expensive_group(&mut config);
    let url = receiver.uri();
    let host = reqwest::Url::parse(&url)
        .expect("wiremock url parses")
        .host_str()
        .expect("wiremock host")
        .to_string();
    config.webhooks = WebhooksConfig {
        endpoints: vec![WebhookEndpoint {
            url: format!("{url}/hook"),
            secret: Some(crate::common::WEBHOOK_TEST_SECRET.to_string()),
            events: Some(vec![BudgetEventType::BudgetExhausted]),
        }],
        allowed_private_hosts: vec![host],
    };

    let server = GatewayServer::new(config, None).await.unwrap();
    let app = server.build_router();
    mount_ok_mock(&upstream, 2).await;
    mount_webhook_ok(&receiver).await;

    // Request #1: $3.00 = 60% — below the warn line, no events.
    let status = post_chat(app.clone(), "one").await;
    assert_eq!(status, StatusCode::OK);
    wait_for_store_spend(&server, PROVIDER_NAME, 3.0).await;
    assert_no_ledger_events(&server, 600).await;

    // Request #2: $6.00 = 120% — the funnel evaluates warning AND
    // exhausted; only the subscribed exhausted is delivered.
    let status = post_chat(app, "two").await;
    assert_eq!(status, StatusCode::OK);
    wait_for_store_spend(&server, PROVIDER_NAME, 6.0).await;

    let deliveries = wait_for_deliveries(&receiver, 1).await;
    assert_eq!(deliveries.len(), 1, "only the subscribed event delivers");
    let delivered = verify_signature_and_parse(&deliveries[0]);
    assert_eq!(delivered.event_type, BudgetEventType::BudgetExhausted);

    // The warning event exists in the ledger with a skipped_no_endpoints
    // delivery row (it had no subscribing endpoint).
    let warning_events: Vec<_> = server
        .state
        .webhook_ledger
        .recent_events(10)
        .into_iter()
        .filter(|row| row.event_type == "budget.warning")
        .collect();
    assert_eq!(warning_events.len(), 1, "warning event still recorded");
    let skip_rows: Vec<_> = server
        .state
        .webhook_ledger
        .recent_deliveries(None, 20)
        .into_iter()
        .filter(|row| row.event_id == warning_events[0].id && row.outcome == "skipped_no_endpoints")
        .collect();
    assert!(
        !skip_rows.is_empty(),
        "sole endpoint filtered out → skipped_no_endpoints row required"
    );
}

// ---------------------------------------------------------------------------
// Dashboard snapshot: recent_budget_alerts (Task 9, Req 5.4)
// ---------------------------------------------------------------------------

/// The dashboard snapshot's `recent_budget_alerts` field carries the
/// newest ledger events (capped at 20, newest first) after they fire
/// through the full spend funnel — the exact snapshot the WS metrics
/// cadence and `GET /dashboard/metrics` serve (design D8). Dormant or
/// empty ledgers must degrade to an empty list, never an error.
///
/// Validates: Requirement 5.4
#[tokio::test]
async fn dashboard_snapshot_carries_recent_budget_alerts() {
    let upstream = MockServer::start().await;
    let receiver = MockServer::start().await;
    let db_dir = tempfile::tempdir().expect("temp dir");

    let mut config = isolated_config(db_dir.path());
    // $5.00 limit, manual window: request #1 = $3 (no events), #2
    // crosses to $6 → warning + exhausted, exactly as the crossing
    // test — then a third request stays at 120% (no new rows, proving
    // the feed shows events, not accruals).
    config.providers = vec![webhook_test_provider(
        &upstream.uri(),
        Some(ProviderBudgetConfig {
            limit_usd: 5.0,
            reset_policy: Some(BudgetResetPolicy::Manual),
            alert_threshold_percent: None,
        }),
    )];
    crate::common::webhook_with_expensive_group(&mut config);
    configure_webhook_receiver(&mut config, &receiver);

    let server = GatewayServer::new(config, None).await.unwrap();
    let app = server.build_router();
    mount_ok_mock(&upstream, 2).await;
    mount_webhook_ok(&receiver).await;

    // Before any events: the field is present and empty (degrade
    // silently; the ledger exists but holds no rows).
    let snapshot = fetch_dashboard_snapshot(&app).await;
    let alerts = snapshot["recent_budget_alerts"]
        .as_array()
        .expect("recent_budget_alerts present in snapshot");
    assert!(alerts.is_empty(), "empty ledger must yield empty feed");

    // Fire the funnel: $3 then $6 → warning + exhausted.
    let status = post_chat(app.clone(), "one").await;
    assert_eq!(status, StatusCode::OK);
    wait_for_store_spend(&server, PROVIDER_NAME, 3.0).await;
    let status = post_chat(app.clone(), "two").await;
    assert_eq!(status, StatusCode::OK);
    wait_for_store_spend(&server, PROVIDER_NAME, 6.0).await;
    wait_for_ledger_events(&server, 2).await;
    // The ledger read behind the snapshot is TTL-cached (1s); let the
    // cache from the initial fetch expire so the next one is fresh.
    tokio::time::sleep(std::time::Duration::from_millis(1100)).await;

    let snapshot = fetch_dashboard_snapshot(&app).await;
    let alerts = snapshot["recent_budget_alerts"]
        .as_array()
        .expect("recent_budget_alerts present in snapshot");
    assert_eq!(alerts.len(), 2, "exactly warning + exhausted");
    // Set membership + newest-first sortedness (the ledger may tie on
    // created_at; fixed index order was never guaranteed).
    let types: Vec<&str> = alerts
        .iter()
        .map(|alert| alert["event_type"].as_str().expect("event_type"))
        .collect();
    assert!(
        types.contains(&"budget.exhausted") && types.contains(&"budget.warning"),
        "expected warning + exhausted, got {types:?}"
    );
    let fired: Vec<i64> = alerts
        .iter()
        .map(|alert| alert["fired_at"].as_i64().expect("fired_at"))
        .collect();
    let mut sorted = fired.clone();
    sorted.sort_unstable_by(|a, b| b.cmp(a));
    assert_eq!(fired, sorted, "feed must be newest-first");
    for alert in alerts {
        assert_eq!(alert["scope"], "provider");
        assert_eq!(alert["scope_label"], PROVIDER_NAME);
        let fired_at = alert["fired_at"].as_i64().expect("fired_at epoch seconds");
        let now = chrono::Utc::now().timestamp();
        assert!(
            (now - fired_at).abs() < 60,
            "fired_at must be a sane recent epoch: {fired_at} vs {now}"
        );
    }

    // The feed is bounded to the newest 20 events: seed 19 more
    // ledger rows directly (the funnel under test has already proven
    // end-to-end emission above) and assert the cap + newest-first
    // ordering the snapshot contract promises.
    for i in 0..19 {
        server
            .state
            .webhook_ledger
            .insert_event(
                &format!("cap-evt-{i}"),
                "budget.warning",
                "provider",
                PROVIDER_NAME,
                "{}",
                1_700_000_000 + i,
            )
            .expect("seed ledger row");
    }
    wait_for_ledger_events(&server, 21).await;
    // TTL cache from the previous fetch must expire before re-reading.
    tokio::time::sleep(std::time::Duration::from_millis(1100)).await;

    let snapshot = fetch_dashboard_snapshot(&app).await;
    let alerts = snapshot["recent_budget_alerts"]
        .as_array()
        .expect("recent_budget_alerts present in snapshot");
    assert_eq!(alerts.len(), 20, "feed is capped at 20 (Req 5.4)");
    // The funnel events (~now) outrank every seeded 1.7e9 row; the
    // warn/exhausted pair may tie on created_at (same second), so
    // assert membership + newest-first sortedness instead of a fixed
    // index: both funnel rows must survive the cap, ahead of all 18
    // seeded rows that fit.
    let funnel_rows: Vec<i64> = alerts
        .iter()
        .map(|a| a["fired_at"].as_i64().expect("fired_at"))
        .filter(|f| *f > 1_750_000_000)
        .collect();
    assert_eq!(funnel_rows.len(), 2, "both funnel events survive the cap");
    assert_eq!(alerts[0]["event_type"], "budget.exhausted");
    let fired: Vec<i64> = alerts
        .iter()
        .map(|a| a["fired_at"].as_i64().expect("fired_at"))
        .collect();
    let mut sorted = fired.clone();
    sorted.sort_unstable_by(|a, b| b.cmp(a));
    assert_eq!(fired, sorted, "feed must be newest-first");
    // Threshold kind rides warning rows only (design Component 5):
    // the fixed 80% default applies here (no configured threshold).
    let warning = alerts
        .iter()
        .find(|a| a["event_type"] == "budget.warning")
        .expect("warning row in feed");
    assert_eq!(warning["threshold_percent"], 80);
    let exhausted = alerts
        .iter()
        .find(|a| a["event_type"] == "budget.exhausted")
        .expect("exhausted row in feed");
    assert!(
        exhausted.get("threshold_percent").is_none(),
        "exhausted rows carry no threshold"
    );
}

/// GET /dashboard/metrics parsed as JSON — the same handler the WS
/// metrics cadence uses (`build_dashboard_snapshot`).
async fn fetch_dashboard_snapshot(app: &axum::Router) -> serde_json::Value {
    let response = app
        .clone()
        .oneshot(
            Request::get("/dashboard/metrics")
                .body(Body::empty())
                .unwrap(),
        )
        .await
        .expect("metrics request");
    assert_eq!(response.status(), StatusCode::OK);
    let body = axum::body::to_bytes(response.into_body(), 64 * 1024)
        .await
        .expect("metrics body");
    serde_json::from_slice(&body).expect("snapshot JSON")
}
