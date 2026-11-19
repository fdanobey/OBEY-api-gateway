//! Dashboard budget-status forecast surfaces (spec:
//! spend-pace-forecasting, Task 6).
//!
//! Exercises the same snapshot handler the WS metrics cadence and
//! `GET /dashboard/metrics` serve (`build_dashboard_snapshot`):
//! - provider statuses carry server-derived `forecast` fields
//! (windowed: pure projection; manual: trailing 30-day ledger run
//! rate through the Task 4 aggregate) — Reqs 1.6, 2.1-2.4;
//! - young windows surface `insufficient_data`, never a day-1
//! extrapolation (Req 1.4 via the dashboard surface);
//! - manual-window run-rate comes from SEEDED ledger rows (Req 1.3);
//! - VK statuses are the durable alert-safe rows with the same
//! forecast semantics;
//! - scopes without a configured limit carry no forecast at all.

use axum::body::Body;
use axum::http::{Request, StatusCode};
use tower::ServiceExt;

use ai_gateway::config::*;
use ai_gateway::gateway::GatewayServer;
use ai_gateway::virtual_keys::models::{BudgetWindow, CreateKeyParams};

use crate::common::budget_test_config;

const PROVIDER_NAME: &str = "budget-provider";
const DAY: i64 = 86_400;

/// Redirect logs/keys databases into `db_dir` (stable paths; budgets.db
/// lands as a sibling of logs.db per the gateway placement rule).
fn isolated_config(db_dir: &std::path::Path, budget: Option<ProviderBudgetConfig>) -> Config {
    let mut config = budget_test_config(Some(db_dir), budget);
    config.logging.retention_days = 0;
    config
}

/// Seed one attributed ledger row through the real logger write path.
fn seed_ledger_row(
    logger: &ai_gateway::logger::RequestLogger,
    timestamp: i64,
    provider: &str,
    cost: f64,
    key_id: Option<&str>,
) {
    let entry = crate::common::sample_log_entry(
        &format!("trace-{timestamp}-{provider}-{cost}"),
        timestamp,
        provider,
        "gpt-4.1-nano",
        cost,
        crate::common::LogEntryOverrides {
            virtual_key_id: key_id.map(str::to_string),
            ..Default::default()
        },
    );
    logger.log(entry).expect("seed log entry");
    logger.flush();
}

/// GET /dashboard/metrics parsed as JSON — the same handler the WS
/// metrics cadence uses (`build_dashboard_snapshot`).
async fn fetch_dashboard_snapshot(server: &GatewayServer) -> serde_json::Value {
    let response = server
        .build_router()
        .oneshot(
            Request::get("/dashboard/metrics")
                .body(Body::empty())
                .unwrap(),
        )
        .await
        .expect("metrics request");
    assert_eq!(response.status(), StatusCode::OK);
    let body = axum::body::to_bytes(response.into_body(), 512 * 1024)
        .await
        .expect("metrics body");
    serde_json::from_slice(&body).expect("snapshot JSON")
}

/// Manual provider budget with a limit: the forecast must be the
/// trailing 30-day run-rate projection over SEEDED ledger rows (Req
/// 1.3) — projected = spent + trailing 30-day spend when the ledger
/// covers ≥ 7 distinct UTC days inside the horizon.
#[tokio::test]
async fn manual_provider_forecast_uses_seeded_ledger_run_rate() {
    let dir = tempfile::tempdir().expect("temp dir");
    let config = isolated_config(
        dir.path(),
        Some(ProviderBudgetConfig {
            limit_usd: 50.0,
            reset_policy: None,
            alert_threshold_percent: None,
        }),
    );
    let server = GatewayServer::new(config, None).await.expect("server builds");

    // 10 distinct UTC days of provider traffic inside the trailing
    // horizon: $1/day => trailing 30-day spend $10 (history gate ≥ 7).
    let now = chrono::Utc::now().timestamp();
    for day in 0..10 {
        seed_ledger_row(
            &server.state.logger,
            now - day * DAY - 3_600,
            PROVIDER_NAME,
            1.0,
            None,
        );
    }
    // Durable current-window spend for the store row the status reads.
    let store = server
        .state
        .metrics
        .provider_budget_store()
        .expect("store attached");
    store
        .add_spend(PROVIDER_NAME, 5.0, now)
        .expect("spend persisted");
    server.state.metrics.add_cost(PROVIDER_NAME, 5.0);

    let snap = fetch_dashboard_snapshot(&server).await;
    let status = &snap["budget_status_by_provider"][PROVIDER_NAME];
    assert!(status.is_object(), "provider status present: {snap}");
    let forecast = &status["forecast"];
    assert!(
        forecast.is_object(),
        "manual scope with limit carries a forecast: {snap}"
    );
    assert_eq!(forecast["method"], "trailing_run_rate");
    // projected = spent (5) + trailing (10) = 15; limit 50 → within.
    assert_eq!(forecast["status"], "within_window");
    let projected = forecast["projected_usd"].as_f64().expect("projected_usd");
    assert!(
        (projected - 15.0).abs() < 1e-6,
        "projected {projected} must be spent(5) + trailing(10)"
    );
    assert!(
        forecast.get("projected_exhaustion_date").is_none()
            || forecast["projected_exhaustion_date"].is_null()
    );
}

/// Manual provider budget whose ledger history is under the 7-day
/// gate: explicit `insufficient_data`, never a thin-history
/// extrapolation.
#[tokio::test]
async fn manual_provider_short_history_is_insufficient_data() {
    let dir = tempfile::tempdir().expect("temp dir");
    let config = isolated_config(
        dir.path(),
        Some(ProviderBudgetConfig {
            limit_usd: 50.0,
            reset_policy: None,
            alert_threshold_percent: None,
        }),
    );
    let server = GatewayServer::new(config, None).await.expect("server builds");

    // Only 2 distinct UTC days of history — below the 7-day gate.
    let now = chrono::Utc::now().timestamp();
    for day in 0..2 {
        seed_ledger_row(
            &server.state.logger,
            now - day * DAY - 3_600,
            PROVIDER_NAME,
            3.0,
            None,
        );
    }
    let store = server
        .state
        .metrics
        .provider_budget_store()
        .expect("store attached");
    store
        .add_spend(PROVIDER_NAME, 6.0, now)
        .expect("spend persisted");
    server.state.metrics.add_cost(PROVIDER_NAME, 6.0);

    let snap = fetch_dashboard_snapshot(&server).await;
    let forecast = &snap["budget_status_by_provider"][PROVIDER_NAME]["forecast"];
    assert!(
        forecast.is_object(),
        "insufficient-data is still a forecast object: {snap}"
    );
    assert_eq!(forecast["status"], "insufficient_data");
    assert_eq!(forecast["method"], "trailing_run_rate");
    // Un-extrapolated spend, never a projection.
    assert_eq!(forecast["projected_usd"].as_f64(), Some(6.0));
}

/// Windowed (daily) provider budget: the forecast is pure window-pace
/// math (Req 1.2) — no ledger dependency — and a fresh window whose
/// elapsed share is under 5% reports `insufficient_data` (Req 1.4).
#[tokio::test]
async fn young_daily_window_is_insufficient_data_then_window_pace() {
        let dir = tempfile::tempdir().expect("temp dir");
        let config = isolated_config(
            dir.path(),
            Some(ProviderBudgetConfig {
                limit_usd: 100.0,
                reset_policy: Some(BudgetResetPolicy::Daily),
                alert_threshold_percent: None,
            }),
        );
    let server = GatewayServer::new(config, None).await.expect("server builds");

    let store = server
        .state
        .metrics
        .provider_budget_store()
        .expect("store attached");
    // Anchor the window start at NOW: the window is seconds old, far
    // under the 5% elapsed gate of a daily window.
    let now = chrono::Utc::now().timestamp();
    store
        .add_spend(PROVIDER_NAME, 20.0, now)
        .expect("spend persisted");
    server.state.metrics.add_cost(PROVIDER_NAME, 20.0);

    let snap = fetch_dashboard_snapshot(&server).await;
    let forecast = &snap["budget_status_by_provider"][PROVIDER_NAME]["forecast"];
    assert!(
        forecast.is_object(),
        "young window still carries a forecast object: {snap}"
    );
    assert_eq!(forecast["method"], "window_pace");
    assert_eq!(forecast["status"], "insufficient_data");
    assert_eq!(forecast["projected_usd"].as_f64(), Some(20.0));

 // Re-anchor the window so elapsed == remaining: the window end is
 // the next UTC midnight after `now` (the production boundary math),
 // so the anchor is `window_end - 2·(window_end - now)` — wait, no:
 // elapsed == remaining means now sits exactly mid-window, so the
 // anchor is `now - (window_end - now)` — deterministic 50% elapsed
 // of [anchor, window_end] at ANY wall clock (a fixed `now - 12h`
 // offset crosses midnight when the test runs before noon UTC and
 // skews the projection). $20 at the half-way mark projects to
 // exactly $40 — within the $100 limit. Raw SQL re-anchor because
 // the store API stamps "now".
 let window_end = {
 let mut boundary = chrono::Utc::now()
 .date_naive()
 .and_hms_opt(0, 0, 0)
 .unwrap()
 .and_utc()
 .timestamp()
 + DAY;
 if boundary <= now {
 boundary += DAY;
 }
 boundary
 };
 let remaining = window_end - now;
 let budgets_db = dir.path().join("budgets.db");
 {
 let conn = rusqlite::Connection::open(&budgets_db).expect("open budgets.db");
 conn.execute(
 "UPDATE provider_budgets SET window_start = ?1 WHERE provider = ?2",
 rusqlite::params![now - remaining, PROVIDER_NAME],
 )
 .expect("re-anchor window");
 }
 // The status derivation reads budget rows through a 1-second TTL
 // cache (design D8): no flush hook exists, so let the TTL expire
 // before re-fetching (finding 13: left as a sleep by design).
 tokio::time::sleep(std::time::Duration::from_millis(1100)).await;
 let snap = fetch_dashboard_snapshot(&server).await;
 let forecast = &snap["budget_status_by_provider"][PROVIDER_NAME]["forecast"];
 assert_eq!(forecast["status"], "within_window", "snapshot: {snap}");
 assert_eq!(forecast["method"], "window_pace");
 let projected = forecast["projected_usd"].as_f64().expect("projected_usd");
 // Elapsed == remaining ⇒ projected = spent + pace·remaining = $40
 // exactly; allow a hair of drift from second-granularity timestamps.
 assert!(
 (projected - 40.0).abs() < 1.0,
 "projected {projected} must be window-paced to ≈ $40 (elapsed == remaining)"
 );
}

/// A pace that reaches the limit inside the horizon reports the ISO
/// exhaustion date and whole days remaining (Req 2.3) — manual scope:
/// the 30-day horizon is anchored at `now`, so the projection is
/// wall-clock deterministic.
#[tokio::test]
async fn over_pace_scope_reports_exhaustion_date() {
    let dir = tempfile::tempdir().expect("temp dir");
    let config = isolated_config(
        dir.path(),
        Some(ProviderBudgetConfig {
            limit_usd: 12.0,
            reset_policy: None,
            alert_threshold_percent: None,
        }),
    );
    let server = GatewayServer::new(config, None).await.expect("server builds");

    // 10 days of history at $1/day (gate satisfied): run rate
    // $10/30d ≈ $0.33/day. Spent $5 against a $12 limit exhausts at
    // day 21 — inside the 30-day horizon; projected = 5 + 10 = 15.
    let now = chrono::Utc::now().timestamp();
    for day in 0..10 {
        seed_ledger_row(
            &server.state.logger,
            now - day * DAY - 3_600,
            PROVIDER_NAME,
            1.0,
            None,
        );
    }
    let store = server
        .state
        .metrics
        .provider_budget_store()
        .expect("store attached");
    store
        .add_spend(PROVIDER_NAME, 5.0, now)
        .expect("spend persisted");
    server.state.metrics.add_cost(PROVIDER_NAME, 5.0);

    let snap = fetch_dashboard_snapshot(&server).await;
    let forecast = &snap["budget_status_by_provider"][PROVIDER_NAME]["forecast"];
    assert_eq!(
        forecast["status"], "exhausts_within_window",
        "snapshot: {snap}"
    );
    assert_eq!(forecast["method"], "trailing_run_rate");
    let projected = forecast["projected_usd"].as_f64().expect("projected_usd");
    assert!((projected - 15.0).abs() < 1e-6, "projected {projected}");
    let date = forecast["projected_exhaustion_date"]
        .as_str()
        .expect("ISO exhaustion date");
    assert!(
        chrono::NaiveDate::parse_from_str(date, "%Y-%m-%d").is_ok(),
        "date must be ISO-8601: {date}"
    );
    let days = forecast["days_remaining"].as_u64().expect("days remaining");
    assert!(days >= 20 && days <= 22, "≈21 days, got {days}");
}

/// No configured limit → no forecast at all (Req 1.6 / 3.6: nothing to
/// project against).
#[tokio::test]
async fn no_limit_scope_carries_no_forecast() {
    let dir = tempfile::tempdir().expect("temp dir");
    let config = isolated_config(dir.path(), None);
    let server = GatewayServer::new(config, None).await.expect("server builds");

    let store = server
        .state
        .metrics
        .provider_budget_store()
        .expect("store attached");
    store
        .add_spend(PROVIDER_NAME, 12.34, chrono::Utc::now().timestamp())
        .expect("spend persisted");

    let snap = fetch_dashboard_snapshot(&server).await;
    let status = &snap["budget_status_by_provider"][PROVIDER_NAME];
    assert!(
        status.get("forecast").is_none() || status["forecast"].is_null(),
        "no limit must not project: {snap}"
    );
}

/// VK budget statuses: durable alert-safe rows with manual-window
/// trailing run-rate forecasts from seeded ledger rows attributed to
/// the key (Req 1.3), and window keys projected by window pace.
#[tokio::test]
async fn vk_budget_statuses_carry_forecasts() {
    let dir = tempfile::tempdir().expect("temp dir");
    let config = isolated_config(dir.path(), None);
    let server = GatewayServer::new(config, None).await.expect("server builds");

    // Manual key with a limit: 10 days of attributed ledger history
    // ($1/day) plus $2 durable current spend; limit $20 → projected 12
    // → within.
    let manual_key = server
        .state
        .virtual_key_manager
        .create_key(CreateKeyParams {
            name: Some("manual-key".to_string()),
            budget_limit_usd: Some(20.0),
            budget_window: None,
            ..default_create_params()
        })
        .await
        .expect("create manual key");
    let now = chrono::Utc::now().timestamp();
    for day in 0..10 {
        seed_ledger_row(
            &server.state.logger,
            now - day * DAY - 3_600,
            PROVIDER_NAME,
            1.0,
            Some(&manual_key.id),
        );
    }
    server
        .state
        .virtual_key_manager
        .record_usage(ai_gateway::virtual_keys::models::UsageRecord {
            key_id: manual_key.id.clone(),
            model_group: "budget-group".to_string(),
            model: "gpt-4.1-nano".to_string(),
            input_tokens: 0,
            output_tokens: 0,
            cost_usd: 2.0,
            timestamp: chrono::Utc::now(),
        })
        .await
        .expect("record manual-key usage");

    // Daily key with a limit: window-pace method regardless of the
    // wall clock (fresh or mid-window both render a forecast object;
    // the exact status depends on time-of-day, so assert method +
    // presence only).
    let daily_key = server
        .state
        .virtual_key_manager
        .create_key(CreateKeyParams {
            name: Some("daily-key".to_string()),
            budget_limit_usd: Some(8.0),
            budget_window: Some(BudgetWindow::Daily),
            ..default_create_params()
        })
        .await
        .expect("create daily key");
    server
        .state
        .virtual_key_manager
        .record_usage(ai_gateway::virtual_keys::models::UsageRecord {
            key_id: daily_key.id.clone(),
            model_group: "budget-group".to_string(),
            model: "gpt-4.1-nano".to_string(),
            input_tokens: 0,
            output_tokens: 0,
            cost_usd: 4.0,
            timestamp: chrono::Utc::now(),
        })
        .await
        .expect("record daily-key usage");

    let snap = fetch_dashboard_snapshot(&server).await;
    let vk = snap["budget_status_by_virtual_key"]
        .as_array()
        .expect("vk status list");
    assert_eq!(vk.len(), 2, "both keys carry statuses: {snap}");

    let manual = vk
        .iter()
        .find(|s| s["key_id"] == manual_key.id.as_str())
        .expect("manual key status");
    assert_eq!(manual["name"], "manual-key");
    assert_eq!(manual["window"], "manual");
    assert_eq!(manual["limit_usd"].as_f64(), Some(20.0));
    let forecast = &manual["forecast"];
    assert!(forecast.is_object(), "manual VK forecast present: {snap}");
    assert_eq!(forecast["method"], "trailing_run_rate");
    assert_eq!(forecast["status"], "within_window");
    let projected = forecast["projected_usd"].as_f64().expect("projected_usd");
    assert!(
        (projected - 12.0).abs() < 1e-6,
        "projected {projected} must be spent(2) + trailing(10)"
    );

    let daily = vk
        .iter()
        .find(|s| s["key_id"] == daily_key.id.as_str())
        .expect("daily key status");
    assert_eq!(daily["window"], "daily");
    let forecast = &daily["forecast"];
    assert!(forecast.is_object(), "daily VK forecast present: {snap}");
    assert_eq!(
        forecast["method"], "window_pace",
        "windowed VK uses window pace: {snap}"
    );
}

fn default_create_params() -> CreateKeyParams {
    CreateKeyParams {
        name: None,
        budget_limit_usd: None,
        token_budget: None,
        budget_window: None,
        requests_per_minute: None,
        tokens_per_minute: None,
        model_access: None,
        expires_in: None,
        loop_detection: None,
        priority: None,
    }
}
