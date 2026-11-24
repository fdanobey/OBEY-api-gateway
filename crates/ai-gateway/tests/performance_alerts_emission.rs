//! Performance-alert emission integration tests (performance-alerts
//! spec, Task 5): end-to-end cycles over a seeded 8-day requests
//! ledger, plus the dormancy, restart-dedup, ledger-failure, and
//! spawn-discipline contracts. Manual `run_cycle_at` orchestration
//! with an injected clock (no reliance on the wall clock passing the
//! 01:00 grace); `GatewayServer::new` for the real logger / ledger /
//! notifier plumbing. No live server, no bound ports.

use ai_gateway::config::*;
use ai_gateway::gateway::GatewayServer;
use ai_gateway::performance_alerts::emission::PerformanceAlertState;
use chrono::{DateTime, Duration, NaiveDate, SecondsFormat, Utc};
use wiremock::MockServer;

use crate::common;

const PROVIDER: &str = "budget-provider";
const MODEL: &str = "gpt-4.1-nano";

fn seeded_config(receiver: Option<&MockServer>) -> Config {
    let mut config = common::budget_test_config(None, None);
    common::isolate_databases(&mut config);
    config.logging.retention_days = 0;
    config.performance_alerts.enabled = true;
    if let Some(receiver) = receiver {
        common::configure_webhook_receiver(&mut config, receiver);
    }
    config
}

/// A `now` fixed 12:00 UTC on the day after the seeded window, so
/// the due day is always past its 01:00 grace.
fn cycle_clock(day_after: NaiveDate) -> DateTime<Utc> {
    day_after
        .and_hms_opt(12, 0, 0)
        .expect("noon always constructs")
        .and_utc()
}

fn epoch_of(day: NaiveDate) -> i64 {
    day.and_hms_opt(0, 0, 0)
        .expect("UTC midnight always constructs")
        .and_utc()
        .timestamp()
}

fn midnight_after(now: DateTime<Utc>, days: i64) -> NaiveDate {
    (now + Duration::days(days)).date_naive()
}

fn one_row(
    trace: &str,
    timestamp: i64,
    provider: &str,
    cost: f64,
    requested_model: Option<&str>,
) -> ai_gateway::logger::LogEntry {
    let mut entry = common::sample_log_entry(trace, timestamp, provider, MODEL, cost, Default::default());
    entry.requested_model = requested_model.map(str::to_owned);
    entry
}

/// Seed one row per baseline day for `baseline_provider` plus the
/// observed day on `observed_provider`.
async fn seed_window(
    server: &GatewayServer,
    due_day: NaiveDate,
    baseline_provider: &str,
    observed_provider: &str,
    baseline_cost: f64,
    observed_cost: f64,
) {
    let mut rows = Vec::new();
    for back in (1..=7).rev() {
        let day = due_day - Duration::days(back);
        rows.push(one_row(
            &format!("pa-base-{back}"),
            epoch_of(day) + 3_600,
            baseline_provider,
            baseline_cost,
            Some(MODEL),
        ));
    }
    rows.push(one_row(
        "pa-observed",
        epoch_of(due_day) + 3_600,
        observed_provider,
        observed_cost,
        Some(MODEL),
    ));
    for row in rows {
        server.state.logger.log(row).unwrap();
    }
    // Wait for the async writer commit so fetch_daily_stats sees them.
    for _ in 0..200 {
        let committed = server
            .state
            .logger
            .usage_since(observed_provider, MODEL, Utc::now() - Duration::days(9))
            .map(|(count, _)| count >= 1)
            .unwrap_or(false);
        if committed {
            return;
            }
        tokio::time::sleep(std::time::Duration::from_millis(25)).await;
    }
    panic!("seeded ledger rows never committed");
}

async fn build_server(config: Config) -> GatewayServer {
    GatewayServer::new(config, None).await.unwrap()
}

fn alert_state(server: &GatewayServer) -> std::sync::Arc<PerformanceAlertState> {
    server.state.performance_alert_state.clone()
}

fn performance_rows(server: &GatewayServer) -> Vec<ai_gateway::webhooks::ledger::AlertEventRow> {
    server
        .state
        .webhook_ledger
        .recent_events(200)
        .into_iter()
        .filter(|row| row.event_type == "performance.anomaly")
        .collect()
}

async fn wait_for_rows(
    server: &GatewayServer,
    min: usize,
) -> Vec<ai_gateway::webhooks::ledger::AlertEventRow> {
    for _ in 0..200 {
        let rows = performance_rows(server);
        if rows.len() >= min {
            return rows;
            }
        tokio::time::sleep(std::time::Duration::from_millis(25)).await;
    }
    panic!("performance.anomaly rows never reached {min}");
}

fn payload_of(row: &ai_gateway::webhooks::ledger::AlertEventRow) -> serde_json::Value {
    let envelope: serde_json::Value = serde_json::from_str(&row.payload_json).unwrap();
    envelope["data"].clone()
}

// ---------------------------------------------------------------------------
// (a) Seeded spike → exact expected events
// ---------------------------------------------------------------------------

/// Baseline: $10/day for 7 days on one provider; observed day: $100
/// (10x, +$90) — spend fires (>=3x and >=$50 over); traffic/error/
/// latency do not. The same rows roll up to the gateway scope (fires
/// spend) and the model-group dimension (fires spend), so the cycle
/// emits exactly three events.
#[tokio::test]
async fn seeded_spike_fires_expected_events() {
    let receiver = MockServer::start().await;
    common::mount_webhook_ok(&receiver).await;
    let config = seeded_config(Some(&receiver));
    let server = build_server(config).await;

    let now = Utc::now();
    let due_day = midnight_after(now, 0).pred_opt().unwrap();
    seed_window(&server, due_day, PROVIDER, PROVIDER, 10.0, 100.0).await;

    alert_state(&server)
        .run_cycle_at(cycle_clock(due_day.succ_opt().unwrap()))
        .await
        .unwrap();
    let rows = wait_for_rows(&server, 3).await;
    assert_eq!(rows.len(), 3, "exactly gateway + provider + model_group spend");

    let mut firings: Vec<(String, String, String)> = rows
        .iter()
        .map(|row| {
            let data = payload_of(row);
            (
                data["scope"].as_str().unwrap().to_string(),
                data["name"].as_str().unwrap().to_string(),
                data["signal"].as_str().unwrap().to_string(),
            )
        })
        .collect();
    firings.sort();
    firings.dedup();
    assert_eq!(
        firings,
        vec![
            ("gateway".to_string(), "gateway".to_string(), "spend".to_string()),
            ("model_group".to_string(), MODEL.to_string(), "spend".to_string()),
            ("provider".to_string(), PROVIDER.to_string(), "spend".to_string()),
        ]
    );

    for row in &rows {
        let data = payload_of(row);
        assert_eq!(data["evaluated_day"].as_str().unwrap(), due_day.to_string());
        assert_eq!(data["unit"].as_str().unwrap(), "usd");
        assert_eq!(data["threshold"].as_str().unwrap(), ">=3x baseline and >=$50 over");
        assert!(data["observed_value"].as_f64().unwrap() >= 100.0);
        assert!(data["baseline_value"].as_f64().unwrap() >= 10.0);
    }

    // Ledger scope column carries the new dimension values (Task 5
    // ledger contract).
    let scopes: Vec<(String, String)> = rows
        .iter()
        .map(|row| (row.scope.clone(), row.scope_label.clone()))
        .collect();
    assert!(scopes.contains(&(("gateway").to_string(), "gateway".to_string())));
    assert!(scopes.contains(&("provider".to_string(), PROVIDER.to_string())));
    assert!(scopes.contains(&("model_group".to_string(), MODEL.to_string())));
}

// ---------------------------------------------------------------------------
// (b) Dormant when no endpoints: no events, no writes
// ---------------------------------------------------------------------------

#[tokio::test]
async fn dormant_without_endpoints_emits_nothing() {
    let config = seeded_config(None);
    let server = build_server(config).await;

    let now = Utc::now();
    let due_day = midnight_after(now, 0).pred_opt().unwrap();
    seed_window(&server, due_day, PROVIDER, PROVIDER, 10.0, 100.0).await;

    alert_state(&server)
        .run_cycle_at(cycle_clock(due_day.succ_opt().unwrap()))
        .await
        .unwrap();
    assert!(
        performance_rows(&server).is_empty(),
        "dormant cycle writes no events"
    );
}

// ---------------------------------------------------------------------------
// (c) Disabled flag → skip; hot-reload re-enable → next cycle evaluates
// ---------------------------------------------------------------------------

#[tokio::test]
async fn disabled_flag_skips_and_reenable_evaluates() {
    let receiver = MockServer::start().await;
    common::mount_webhook_ok(&receiver).await;
    let mut config = seeded_config(Some(&receiver));
    config.performance_alerts.enabled = false;
    let server = build_server(config).await;

    let now = Utc::now();
    let due_day = midnight_after(now, 0).pred_opt().unwrap();
    seed_window(&server, due_day, PROVIDER, PROVIDER, 10.0, 100.0).await;

    let clock = cycle_clock(due_day.succ_opt().unwrap());
    alert_state(&server).run_cycle_at(clock).await.unwrap();
    assert!(performance_rows(&server).is_empty(), "disabled cycle skips");

    // Hot-reload re-enable: rewrite the shared config in place.
    {
        let mut guard = server.state.config.write().await;
        guard.performance_alerts.enabled = true;
    }
    alert_state(&server).run_cycle_at(clock).await.unwrap();
    let rows = wait_for_rows(&server, 3).await;
    assert_eq!(rows.len(), 3, "re-enabled cycle evaluates the due day");
}

// ---------------------------------------------------------------------------
// (d) Restart dedup: fresh state over a populated ledger suppresses
// ---------------------------------------------------------------------------

#[tokio::test]
async fn restart_with_populated_ledger_suppresses_refiring() {
    let receiver = MockServer::start().await;
    common::mount_webhook_ok(&receiver).await;
    let config = seeded_config(Some(&receiver));
    let server = build_server(config).await;

    let now = Utc::now();
    let due_day = midnight_after(now, 0).pred_opt().unwrap();
    seed_window(&server, due_day, PROVIDER, PROVIDER, 10.0, 100.0).await;

    let clock = cycle_clock(due_day.succ_opt().unwrap());
    alert_state(&server).run_cycle_at(clock).await.unwrap();
    let rows = wait_for_rows(&server, 3).await;
    assert_eq!(rows.len(), 3);

    // FRESH state sharing the same delivery ledger (a restart): the
    // day is deliberately NOT seeded from the newest ledger row
    // (fix M2 — that would mark a partially recorded day complete),
    // so the cycle re-evaluates and the PER-FINDING payload dedup
    // (scope+signal+evaluated_day) suppresses every finding the
    // ledger already holds. Nothing re-fires.
    let fresh = PerformanceAlertState::new(
        server.state.config.clone(),
        server.state.logger.clone(),
        server.state.webhook_ledger.clone(),
        server.state.webhook_notifier.clone(),
    );
    fresh.run_cycle_at(clock).await.unwrap();
    tokio::time::sleep(std::time::Duration::from_millis(200)).await;
    assert_eq!(
        performance_rows(&server).len(),
        3,
        "fresh state must not re-fire findings the ledger already holds"
    );
}

// ---------------------------------------------------------------------------
// (e) Ledger failure → cycle errs, day not advanced, same-state retry succeeds
// ---------------------------------------------------------------------------

#[tokio::test]
async fn ledger_failure_errs_and_same_state_retry_succeeds() {
    let receiver = MockServer::start().await;
    common::mount_webhook_ok(&receiver).await;
    let config = seeded_config(Some(&receiver));
    let server = build_server(config).await;

    let now = Utc::now();
    let due_day = midnight_after(now, 0).pred_opt().unwrap();
    seed_window(&server, due_day, PROVIDER, PROVIDER, 10.0, 100.0).await;

    // Broken ledger: a real store whose schema is dropped out from
    // under it via a second connection — every dedup query then
    // fails, which the cycle must surface as an error (design,
    // Error Handling 1).
    let broken_dir = tempfile::tempdir().unwrap();
    let broken_path = broken_dir.path().join("broken-webhooks.db");
    let broken_store = std::sync::Arc::new(
        ai_gateway::webhooks::ledger::WebhookLedgerStore::new(&broken_path).unwrap(),
    );
    {
        let conn = rusqlite::Connection::open(&broken_path).unwrap();
        conn.execute_batch("DROP TABLE alert_events; DROP TABLE deliveries;").unwrap();
    }

    // Same-state retry contract (fix M1 discipline): one state, the
    // SAME due day, cycled against the broken ledger first.
    let state = PerformanceAlertState::new(
        server.state.config.clone(),
        server.state.logger.clone(),
        broken_store.clone(),
        server.state.webhook_notifier.clone(),
    );
    let clock = cycle_clock(due_day.succ_opt().unwrap());
    let result = state.run_cycle_at(clock).await;
    assert!(
        result.is_err(),
        "cycle over a broken ledger must return an error"
    );
    // Behavioral day-not-advanced assertion: a second cycle against
    // the SAME state and day must still attempt the full evaluation
    // (not short-circuit), which fails again while the ledger stays
    // broken — proving `last_evaluated_day` never advanced.
    let result = state.run_cycle_at(clock).await;
    assert!(
        result.is_err(),
        "day must not have advanced after the failed cycle"
    );

    // Restore the schema (the store's init is idempotent), then the
    // SAME state's next cycle succeeds and fires the day. The
    // notifier must share the restored store so the ledger rows land
    // where the cycle confirms them.
    {
        let conn = rusqlite::Connection::open(&broken_path).unwrap();
        conn.execute_batch(
            "CREATE TABLE IF NOT EXISTS alert_events (
                id TEXT PRIMARY KEY,
                type TEXT NOT NULL,
                scope TEXT NOT NULL,
                scope_label TEXT NOT NULL,
                payload_json TEXT NOT NULL,
                created_at INTEGER NOT NULL
            );
            CREATE TABLE IF NOT EXISTS deliveries (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                event_id TEXT NOT NULL REFERENCES alert_events(id),
                endpoint_url TEXT NOT NULL,
                attempt INTEGER NOT NULL,
                outcome TEXT NOT NULL,
                status_code INTEGER,
                error TEXT,
                attempted_at INTEGER NOT NULL
            );",
        )
        .unwrap();
    }
    let restored_notifier = std::sync::Arc::new(
        ai_gateway::webhooks::delivery::WebhookNotifier::new(
            server.state.config.clone(),
            broken_store.clone(),
        )
        .await
        .unwrap(),
    );
    let state = PerformanceAlertState::new(
        server.state.config.clone(),
        server.state.logger.clone(),
        broken_store.clone(),
        restored_notifier,
    );
    state.run_cycle_at(clock).await.unwrap();
    let rows = wait_for_ledger_rows(&broken_store, 3).await;
    assert_eq!(rows.len(), 3, "same-state retry after schema restore fires the day");
}

/// Ledger-side twin of [`wait_for_rows`] for a standalone store.
async fn wait_for_ledger_rows(
    store: &std::sync::Arc<ai_gateway::webhooks::ledger::WebhookLedgerStore>,
    min: usize,
) -> Vec<ai_gateway::webhooks::ledger::AlertEventRow> {
    for _ in 0..200 {
        let rows = store
            .recent_events(200)
            .into_iter()
            .filter(|row| row.event_type == "performance.anomaly")
            .collect::<Vec<_>>();
        if rows.len() >= min {
            return rows;
        }
        tokio::time::sleep(std::time::Duration::from_millis(25)).await;
    }
    panic!("performance.anomaly rows never reached {min}");
}

// ---------------------------------------------------------------------------
// (f) Model-group scope events carry the model_group scope value
// ---------------------------------------------------------------------------

/// One provider's spend spikes on the observed day; the model-group
/// aggregate over the shared `requested_model` moves with it, so the
/// model_group scope fires alongside gateway and provider.
#[tokio::test]
async fn model_group_scope_fires_with_model_group_scope_value() {
    let receiver = MockServer::start().await;
    common::mount_webhook_ok(&receiver).await;
    let config = seeded_config(Some(&receiver));
    let server = build_server(config).await;

    let now = Utc::now();
    let due_day = midnight_after(now, 0).pred_opt().unwrap();
    // Model-group baseline: $5/day (one quiet provider); observed:
    // $200 on the same provider — merged aggregate fires spend.
    seed_window(&server, due_day, PROVIDER, PROVIDER, 5.0, 200.0).await;

    alert_state(&server)
        .run_cycle_at(cycle_clock(due_day.succ_opt().unwrap()))
        .await
        .unwrap();
    let rows = wait_for_rows(&server, 3).await;
    assert_eq!(rows.len(), 3, "gateway + provider + model_group findings");

    let model_group_rows: Vec<_> = rows
        .iter()
        .filter(|row| row.scope == "model_group")
        .collect();
    assert_eq!(model_group_rows.len(), 1, "one model_group row");
    let data = payload_of(model_group_rows[0]);
    assert_eq!(data["scope"].as_str().unwrap(), "model_group");
    assert_eq!(data["name"].as_str().unwrap(), MODEL);
    assert_eq!(model_group_rows[0].scope_label, MODEL);
}

// ---------------------------------------------------------------------------
// (g) No timer from GatewayServer::new
// ---------------------------------------------------------------------------

/// `GatewayServer::new` never spawns the performance-alert loop:
/// after construction (and a short wait), no performance.anomaly
/// rows exist even with endpoints configured — only
/// `start_with_shutdown` spawns the loop.
#[tokio::test]
async fn gateway_construction_starts_no_performance_alert_timer() {
    let receiver = MockServer::start().await;
    common::mount_webhook_ok(&receiver).await;
    let config = seeded_config(Some(&receiver));
    let server = build_server(config).await;

    tokio::time::sleep(std::time::Duration::from_millis(300)).await;
    assert!(
        performance_rows(&server).is_empty(),
        "construction alone must never fire the emission loop"
    );
}

// ---------------------------------------------------------------------------
// (h) Grace boundary: 00:59:59 skips, 01:00:00 fires
// ---------------------------------------------------------------------------

#[tokio::test]
async fn grace_boundary_second_before_fires_nothing_and_exact_grace_fires() {
    let receiver = MockServer::start().await;
    common::mount_webhook_ok(&receiver).await;
    let config = seeded_config(Some(&receiver));
    let server = build_server(config).await;

    let now = Utc::now();
    let due_day = midnight_after(now, 0).pred_opt().unwrap();
    seed_window(&server, due_day, PROVIDER, PROVIDER, 10.0, 100.0).await;

    let day_after = due_day.succ_opt().unwrap();
    let before_grace = day_after
        .and_hms_opt(0, 59, 59)
        .expect("00:59:59 always constructs")
        .and_utc();
    alert_state(&server).run_cycle_at(before_grace).await.unwrap();
    assert!(
        performance_rows(&server).is_empty(),
        "00:59:59 is before the 01:00 grace; nothing fires"
    );

    let at_grace = day_after
        .and_hms_opt(1, 0, 0)
        .expect("01:00:00 always constructs")
        .and_utc();
    alert_state(&server).run_cycle_at(at_grace).await.unwrap();
    let rows = wait_for_rows(&server, 3).await;
    assert_eq!(rows.len(), 3, "01:00:00 exactly is past the grace; the due day fires");
}

// ---------------------------------------------------------------------------
// (i) Partial-day restart: per-finding dedup finishes the missing findings
// ---------------------------------------------------------------------------

#[tokio::test]
async fn partial_day_restart_emits_only_missing_findings() {
    let receiver = MockServer::start().await;
    common::mount_webhook_ok(&receiver).await;
    let config = seeded_config(Some(&receiver));
    let server = build_server(config).await;

    let now = Utc::now();
    let due_day = midnight_after(now, 0).pred_opt().unwrap();
    seed_window(&server, due_day, PROVIDER, PROVIDER, 10.0, 100.0).await;

    // Pre-populate the delivery ledger with only ONE of the day's
    // three findings (a crashed process that fired, recorded, and
    // died before finishing the day): a gateway spend row built from
    // the same envelope shape the emission writes.
    let ledger = server.state.webhook_ledger.clone();
    let partial_payload = serde_json::json!({
        "id": uuid::Uuid::new_v4().to_string(),
        "type": "performance.anomaly",
        "created_at": Utc::now().to_rfc3339_opts(SecondsFormat::Secs, true),
        "signed": false,
        "data": {
            "signal": "spend",
            "scope": "gateway",
            "name": "gateway",
            "observed_value": 100.0,
            "baseline_value": 10.0,
            "unit": "usd",
            "evaluated_day": due_day.to_string(),
            "threshold": ">=3x baseline and >=$50 over"
        }
    });
    let payload_json = partial_payload.to_string();
    let created_at = epoch_of(due_day) + 3_600;
    let event_id = partial_payload["id"].as_str().unwrap().to_string();
    tokio::task::spawn_blocking(move || {
        ledger.insert_event(&event_id, "performance.anomaly", "gateway", "gateway", &payload_json, created_at)
    })
    .await
    .unwrap()
    .unwrap();

    // FRESH state (a restart) over the same ledger: the seed is gone
    // (fix M2), so the day re-evaluates and per-finding dedup emits
    // exactly the two MISSING findings — provider and model_group.
    let fresh = PerformanceAlertState::new(
        server.state.config.clone(),
        server.state.logger.clone(),
        server.state.webhook_ledger.clone(),
        server.state.webhook_notifier.clone(),
    );
    fresh
        .run_cycle_at(cycle_clock(due_day.succ_opt().unwrap()))
        .await
        .unwrap();
    let rows = wait_for_rows(&server, 3).await;
    assert_eq!(rows.len(), 3, "one pre-recorded + two re-emitted findings");
    let mut firings: Vec<(String, String)> = rows
        .iter()
        .map(|row| {
            let data = payload_of(row);
            (data["scope"].as_str().unwrap().to_string(), data["name"].as_str().unwrap().to_string())
        })
        .collect();
    firings.sort();
    firings.dedup();
    assert_eq!(
        firings,
        vec![
            ("gateway".to_string(), "gateway".to_string()),
            ("model_group".to_string(), MODEL.to_string()),
            ("provider".to_string(), PROVIDER.to_string()),
        ],
        "the already-recorded gateway finding must NOT be re-fired"
    );
}

// ---------------------------------------------------------------------------
// (j) Fresh install: empty 8-day window → Ok, no events, no panic
// ---------------------------------------------------------------------------

#[tokio::test]
async fn fresh_install_empty_window_is_ok_and_emits_nothing() {
    let receiver = MockServer::start().await;
    common::mount_webhook_ok(&receiver).await;
    let config = seeded_config(Some(&receiver));
    let server = build_server(config).await;

    // No seed_window: the requests ledger has zero rows in the
    // 8-day window (fresh install). The cycle materializes the zero
    // gateway scope (fix m7), evaluate's gates skip everything, and
    // the cycle completes Ok with no events.
    let now = Utc::now();
    let due_day = midnight_after(now, 0).pred_opt().unwrap();
    alert_state(&server)
        .run_cycle_at(cycle_clock(due_day.succ_opt().unwrap()))
        .await
        .unwrap();
    assert!(
        performance_rows(&server).is_empty(),
        "fresh-install cycle emits nothing"
    );
}