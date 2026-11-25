//! Usage-anomaly alert emission integration tests (usage-anomaly-alerts
//! spec, Task 7): end-to-end cycles over a seeded 8-day per-key requests
//! ledger, plus the dedup, restart, catch-up, dormancy, failure-retry,
//! request-path, and no-secrets contracts. Manual `run_cycle_at`
//! orchestration with an injected clock (no reliance on the wall clock
//! passing the 01:00 grace); `GatewayServer::new` for the real logger /
//! ledger / notifier plumbing. No live server, no bound ports.

use std::sync::Arc;

use ai_gateway::config::*;
use ai_gateway::gateway::GatewayServer;
use ai_gateway::usage_anomaly_alerts::emission::{UsageAnomalyAlertError, UsageAnomalyAlertState};
use ai_gateway::virtual_keys::models::CreateKeyParams;
use ai_gateway::webhooks::ledger::AlertEventRow;
use chrono::{DateTime, Duration, NaiveDate, Utc};
use wiremock::MockServer;

use crate::common;

const PROVIDER: &str = "budget-provider";
const MODEL: &str = "gpt-4.1-nano";
const NEW_MODEL: &str = "unseen-shiny-model";

const KEY_CLEAN: &str = "key-clean";
const KEY_SPEND: &str = "key-spend";
const KEY_TRAFFIC: &str = "key-traffic";
const KEY_TOKENS: &str = "key-tokens";
const KEY_ODD: &str = "key-odd";
const KEY_NEWMODEL: &str = "key-newmodel";
const KEY_PROBES: &str = "key-probes";
const KEY_CATCHUP: &str = "key-catchup";

fn seeded_config(receiver: Option<&MockServer>) -> Config {
    let mut config = common::budget_test_config(None, None);
    common::isolate_databases(&mut config);
    config.logging.retention_days = 0;
    config.usage_anomaly_alerts.enabled = true;
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

fn midnight_after(now: DateTime<Utc>, days: i64) -> NaiveDate {
    (now + Duration::days(days)).date_naive()
}

fn at(day: NaiveDate, hour: u32, minute: u32) -> i64 {
    day.and_hms_opt(hour, minute, 0)
        .expect("in-day time always constructs")
        .and_utc()
        .timestamp()
}

async fn build_server(config: Config) -> GatewayServer {
    GatewayServer::new(config, None).await.unwrap()
}

fn alert_state(server: &GatewayServer) -> Arc<UsageAnomalyAlertState> {
    server.state.usage_anomaly_alert_state.clone()
}

fn no_name_resolver() -> ai_gateway::usage_anomaly_alerts::emission::KeyNameResolver {
    Arc::new(|_key_id: &str| Option::<String>::None)
}

/// One attributed ledger row: `key_id` rides the `virtual_key_id`
/// override (persisted as the `key_id` column the aggregates group
/// on); status and requested_model vary per attack shape.
fn key_row(
    trace: &str,
    timestamp: i64,
    key_id: &str,
    cost: f64,
    input_tokens: i64,
    status: u16,
    requested_model: Option<&str>,
) -> ai_gateway::logger::LogEntry {
    let mut entry = common::sample_log_entry(
        trace,
        timestamp,
        PROVIDER,
        MODEL,
        cost,
        common::LogEntryOverrides {
            input_tokens: Some(input_tokens),
            virtual_key_id: Some(key_id.to_owned()),
            ..Default::default()
        },
    );
    entry.status_code = status;
    entry.requested_model = requested_model.map(str::to_owned);
    entry
}

/// Seed `count` attributed rows for one key on one day at a fixed
/// in-day time, all sharing the attack shape's per-row values.
fn log_key_day(
    server: &GatewayServer,
    key_id: &str,
    day: NaiveDate,
    count: usize,
    hour: u32,
    minute: u32,
    cost_each: f64,
    input_each: i64,
    status: u16,
    requested_model: Option<&str>,
    tag: &str,
) {
    let timestamp = at(day, hour, minute);
    for i in 0..count {
        let row = key_row(
            &format!("{tag}-{i}"),
            timestamp,
            key_id,
            cost_each,
            input_each,
            status,
            requested_model,
        );
        server.state.logger.log(row).unwrap();
    }
}

/// Wait for the async writer to commit every seeded row (the cycle's
/// aggregate reader must see the full window).
async fn wait_for_commit(server: &GatewayServer, min_rows: u64) {
    for _ in 0..400 {
        let committed = server
            .state
            .logger
            .usage_since(PROVIDER, MODEL, Utc::now() - Duration::days(9))
            .map(|(count, _)| count >= min_rows)
            .unwrap_or(false);
        if committed {
            return;
        }
        tokio::time::sleep(std::time::Duration::from_millis(25)).await;
    }
    panic!("seeded ledger rows never committed ({min_rows} expected)");
}

/// The 7-day baseline of quiet attack keys: every shape stays under
/// the 25-request new_activity gate so the early catch-up days (whose
/// own baselines are empty) can never fire.
fn seed_baseline_days(
    server: &GatewayServer,
    due_day: NaiveDate,
    key: &str,
    per_day: usize,
    cost_each: f64,
    input_each: i64,
    status: u16,
    requested_model: Option<&str>,
    tag: &str,
) {
    for back in 1..=7 {
        let day = due_day - Duration::days(back);
        log_key_day(
            server,
            key,
            day,
            per_day,
            12,
            0,
            cost_each,
            input_each,
            status,
            requested_model,
            tag,
        );
    }
}

/// Seed only the spend-spike shape (the cheap fixture the restart /
/// dormancy / failure / request-path tests reuse).
async fn seed_spike_only(server: &GatewayServer, due_day: NaiveDate, key: &str) {
    seed_baseline_days(server, due_day, key, 2, 5.0, 0, 200, Some(MODEL), "spike-base");
    log_key_day(server, key, due_day, 20, 12, 0, 5.02, 0, 200, Some(MODEL), "spike-obs");
    wait_for_commit(server, 34).await;
}

fn usage_rows(server: &GatewayServer) -> Vec<AlertEventRow> {
    server
        .state
        .webhook_ledger
        .recent_events(200)
        .into_iter()
        .filter(|row| row.event_type == "usage.anomaly")
        .collect()
}

async fn wait_for_usage_rows(server: &GatewayServer, min: usize) -> Vec<AlertEventRow> {
    for _ in 0..200 {
        let rows = usage_rows(server);
        if rows.len() >= min {
            return rows;
        }
        tokio::time::sleep(std::time::Duration::from_millis(25)).await;
    }
    panic!("usage.anomaly rows never reached {min}");
}

async fn wait_for_ledger_rows(
    store: &Arc<ai_gateway::webhooks::ledger::WebhookLedgerStore>,
    min: usize,
) -> Vec<AlertEventRow> {
    for _ in 0..200 {
        let rows = store
            .recent_events(200)
            .into_iter()
            .filter(|row| row.event_type == "usage.anomaly")
            .collect::<Vec<_>>();
        if rows.len() >= min {
            return rows;
        }
        tokio::time::sleep(std::time::Duration::from_millis(25)).await;
    }
    panic!("usage.anomaly rows never reached {min}");
}

fn payload_of(row: &AlertEventRow) -> serde_json::Value {
    let envelope: serde_json::Value = serde_json::from_str(&row.payload_json).unwrap();
    envelope["data"].clone()
}

fn key_params_defaults() -> CreateKeyParams {
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
        priority: None,
    }
}

// ---------------------------------------------------------------------------
// (a) Attack-shaped keys → exact expected signal set + payload contract
// ---------------------------------------------------------------------

/// 8-day fixture: one quiet key plus six attack shapes and one real
/// (created) dormant key reactivating on the due day. The first cycle
/// evaluates the whole 8-day window oldest-first; only the due day
/// fires — every attack shape is tuned so no early catch-up day can
/// cross a threshold (per-day baselines stay under the 25-request
/// new_activity gate; observed days are the only spikes).
///
/// Expected due-day firings, exactly one per attack key:
/// spend (3x+$50), traffic (10x+1000), tokens (10x+1M), odd_hours
/// (>=50% in baseline-quiet buckets, >=200), new_model (>=10 to an
/// absent model), probes (>=50 validation failures, >5x), and
/// new_activity (zero-baseline key, >=25). The clean key fires
/// nothing. The second cycle of the same day re-fires nothing.
#[tokio::test]
async fn attack_keys_fire_expected_signals() {
    let receiver = MockServer::start().await;
    common::mount_webhook_ok(&receiver).await;
    let config = seeded_config(Some(&receiver));
    let server = build_server(config).await;

    // A REAL virtual key for the dormant-reactivation shape: its id
    // attributes the ledger rows and its display name resolves into
    // the payload while the vk_ secret must never appear anywhere.
    let created = server
        .state
        .virtual_key_manager
        .create_key(CreateKeyParams {
            name: Some("attack-dormant".to_string()),
            ..key_params_defaults()
        })
        .await
        .unwrap();

    let now = Utc::now();
    let due_day = midnight_after(now, 0).pred_opt().unwrap();

    // Clean key: 4 req/day everywhere, no signal may fire.
    for back in (1..=7).rev() {
        log_key_day(
            &server,
            KEY_CLEAN,
            due_day - Duration::days(back),
            4,
            12,
            0,
            0.01,
            0,
            200,
            Some(MODEL),
            "clean-base",
        );
    }
    log_key_day(&server, KEY_CLEAN, due_day, 4, 12, 0, 0.01, 0, 200, Some(MODEL), "clean-obs");

    // Spend spike: $10/day baseline (2 x $5), $100.40 observed
    // (20 x $5.02) — >3x and >$50 over; traffic stays quiet (20 vs 2).
    seed_baseline_days(&server, due_day, KEY_SPEND, 2, 5.0, 0, 200, Some(MODEL), "spend-base");
    log_key_day(&server, KEY_SPEND, due_day, 20, 12, 0, 5.02, 0, 200, Some(MODEL), "spend-obs");

    // Traffic spike: 5/day baseline, 1052 observed — >10x and >=1000
    // over; tokens and cost stay ~flat so only traffic fires.
    seed_baseline_days(&server, due_day, KEY_TRAFFIC, 5, 0.01, 0, 200, Some(MODEL), "traffic-base");
    log_key_day(
        &server,
        KEY_TRAFFIC,
        due_day,
        1052,
        12,
        0,
        0.01,
        0,
        200,
        Some(MODEL),
        "traffic-obs",
    );

    // Token spike: 10k tokens/day baseline, one 1.02M-token observed
    // day — >10x and >=1M over; single request keeps traffic quiet.
    seed_baseline_days(&server, due_day, KEY_TOKENS, 1, 0.0, 10_000, 200, Some(MODEL), "tokens-base");
    log_key_day(
        &server,
        KEY_TOKENS,
        due_day,
        1,
        12,
        0,
        0.0,
        1_020_000,
        200,
        Some(MODEL),
        "tokens-obs",
    );

    // Odd hours: baseline 20/day all at 12:00 (bucket 3), observed
    // 200 all at 00:30 (bucket 0 — a baseline-under-2% bucket) —
    // 100% > 50% share and exactly the 200-request gate.
    seed_baseline_days(&server, due_day, KEY_ODD, 20, 0.001, 0, 200, Some(MODEL), "odd-base");
    log_key_day(&server, KEY_ODD, due_day, 200, 0, 30, 0.001, 0, 200, Some(MODEL), "odd-obs");

    // New model: baseline 3/day on MODEL, observed 10 requests to an
    // absent model — exactly the >=10 gate.
    seed_baseline_days(&server, due_day, KEY_NEWMODEL, 3, 0.01, 0, 200, Some(MODEL), "newmodel-base");
    log_key_day(
        &server,
        KEY_NEWMODEL,
        due_day,
        10,
        12,
        0,
        0.01,
        0,
        200,
        Some(NEW_MODEL),
        "newmodel-obs",
    );

    // Probes: baseline 3 validation failures/day (401, no output
    // tokens), observed 106 — >=50 absolute and >5x baseline; the
    // 106 requests stay under the traffic absolute leg.
    seed_baseline_days(&server, due_day, KEY_PROBES, 3, 0.0, 0, 401, Some(MODEL), "probes-base");
    log_key_day(&server, KEY_PROBES, due_day, 106, 12, 0, 0.0, 0, 401, Some(MODEL), "probes-obs");

    // Dormant reactivation: zero baseline (no rows before the due
    // day), 25 requests on it — the new_activity gate; no
    // requested_model so new_model cannot co-fire.
    log_key_day(
        &server,
        &created.id,
        due_day,
        25,
        12,
        0,
        0.01,
        0,
        200,
        None,
        "dormant-obs",
    );

    // Per-key arithmetic: clean 7*4+4=32, spend 7*2+20=34, traffic
    // 7*5+1052=1087, tokens 7*1+1=8, odd 7*20+200=340, newmodel
    // 7*3+10=31, probes 7*3+106=127, dormant 25 => 1684.
    wait_for_commit(&server, 1684).await;

    alert_state(&server)
        .run_cycle_at(cycle_clock(due_day.succ_opt().unwrap()))
        .await
        .unwrap();
    let rows = wait_for_usage_rows(&server, 7).await;
    assert_eq!(rows.len(), 7, "exactly one finding per attack key");

    let expected: Vec<(String, &str, &str, f64, f64)> = vec![
        (KEY_SPEND.to_string(), "spend", "usd", 100.4, 10.0),
        (KEY_TRAFFIC.to_string(), "traffic", "requests", 1052.0, 5.0),
        (KEY_TOKENS.to_string(), "tokens", "tokens", 1_020_000.0, 10_000.0),
        (KEY_ODD.to_string(), "odd_hours", "ratio", 1.0, 0.0),
        (KEY_NEWMODEL.to_string(), "new_model", "requests", 10.0, 0.0),
        (KEY_PROBES.to_string(), "probes", "requests", 106.0, 3.0),
        (created.id.clone(), "new_activity", "requests", 25.0, 0.0),
    ];
    let mut firings: Vec<(String, String)> = rows
        .iter()
        .map(|row| {
            let data = payload_of(row);
            (
                data["key_id"].as_str().unwrap().to_string(),
                data["signal"].as_str().unwrap().to_string(),
            )
        })
        .collect();
    firings.sort();
    let mut expected_keys: Vec<(String, String)> = expected
        .iter()
        .map(|(key, signal, _, _, _)| (key.clone(), signal.to_string()))
        .collect();
    expected_keys.sort();
    assert_eq!(firings, expected_keys, "exact (key, signal) set");

    for row in &rows {
        let data = payload_of(row);
        assert_eq!(data["scope"].as_str().unwrap(), "virtual_key");
        assert_eq!(data["evaluated_day"].as_str().unwrap(), due_day.to_string());
        assert!(!data["threshold"].as_str().unwrap().is_empty());
        let key_id = data["key_id"].as_str().unwrap();
        let signal = data["signal"].as_str().unwrap();
        let (_, _, unit, observed, baseline) = expected
            .iter()
            .find(|(key, sig, _, _, _)| key == key_id && *sig == signal)
            .cloned()
            .unwrap_or_else(|| panic!("unexpected firing {key_id}/{signal}"));
        assert_eq!(data["unit"].as_str().unwrap(), unit, "{key_id}/{signal}");
        let got_observed = data["observed_value"].as_f64().unwrap();
        let got_baseline = data["baseline_value"].as_f64().unwrap();
        assert!(
            (got_observed - observed).abs() < 0.01,
            "{key_id}/{signal} observed {got_observed} ~= {observed}"
        );
        assert!(
            (got_baseline - baseline).abs() < 1e-6,
            "{key_id}/{signal} baseline {got_baseline} ~= {baseline}"
        );
        // No-secrets invariant: the payload carries the key id and
        // (optionally) the display name — never the vk_ secret.
        assert!(!row.payload_json.contains("vk_"), "no vk_ material in {key_id}/{signal}");
        assert!(
            !row.payload_json.contains(&created.key),
            "raw key secret must never appear in {key_id}/{signal}"
        );
    }

    // The dormant key's display name resolved cosmetically.
    let dormant = rows
        .iter()
        .find(|row| payload_of(row)["signal"].as_str() == Some("new_activity"))
        .unwrap();
    assert_eq!(payload_of(dormant)["name"].as_str(), Some("attack-dormant"));

    // Second cycle of the same day: the last_evaluated_day
    // short-circuit plus per-finding dedup — nothing re-fires.
    alert_state(&server)
        .run_cycle_at(cycle_clock(due_day.succ_opt().unwrap()))
        .await
        .unwrap();
    tokio::time::sleep(std::time::Duration::from_millis(200)).await;
    assert_eq!(
        usage_rows(&server).len(),
        7,
        "second cycle of the same day must not re-fire"
    );
}

// ---------------------------------------------------------------------------
// (b) Restart semantics: fresh state over the same ledgers
// ---------------------------------------------------------------------------

/// A fresh `UsageAnomalyAlertState` (a restart) over the same ledger
/// and webhook ledger re-evaluates and suppresses every recorded
/// finding. Removing one recorded row first proves the day is NOT
/// seeded complete from the ledger rows: the fresh state re-opens the
/// day and re-fires exactly the deleted finding.
#[tokio::test]
async fn restart_reopens_partial_day_and_suppresses_recorded_findings() {
    let receiver = MockServer::start().await;
    common::mount_webhook_ok(&receiver).await;
    let config = seeded_config(Some(&receiver));
    let webhooks_db = std::path::Path::new(&config.logging.database_path)
        .parent()
        .unwrap()
        .join("webhooks.db");
    let server = build_server(config).await;

    let now = Utc::now();
    let due_day = midnight_after(now, 0).pred_opt().unwrap();
    seed_spike_only(&server, due_day, KEY_SPEND).await;

    let clock = cycle_clock(due_day.succ_opt().unwrap());
    alert_state(&server).run_cycle_at(clock).await.unwrap();
    let rows = wait_for_usage_rows(&server, 1).await;
    assert_eq!(rows.len(), 1);
    let spend_event_id = rows[0].id.clone();

    // Fresh state (a restart) over the same ledger + notifier: the
    // day is deliberately NOT adopted from the newest ledger row, so
    // the cycle re-evaluates — and the per-finding (key, signal,
    // day) dedup suppresses the recorded finding. Nothing re-fires.
    let fresh = UsageAnomalyAlertState::new(
        server.state.config.clone(),
        server.state.logger.clone(),
        no_name_resolver(),
        server.state.webhook_ledger.clone(),
        server.state.webhook_notifier.clone(),
    );
    fresh.run_cycle_at(clock).await.unwrap();
    tokio::time::sleep(std::time::Duration::from_millis(200)).await;
    assert_eq!(
        usage_rows(&server).len(),
        1,
        "fresh state must not re-fire findings the ledger already holds"
    );

    // A partially recorded day (one finding's row lost): because no
    // day is ever seeded complete from ledger rows, a fresh state
    // re-evaluates the day and re-fires exactly the missing finding.
    let webhooks_db = webhooks_db.clone();
    tokio::task::spawn_blocking(move || {
        let conn = rusqlite::Connection::open(&webhooks_db).unwrap();
        // Deliveries FK-reference the event row; remove the child
        // delivery rows first so the event row itself can go.
        conn.execute(
            "DELETE FROM deliveries WHERE event_id = ?1",
            rusqlite::params![spend_event_id],
        )
        .unwrap();
        conn.execute("DELETE FROM alert_events WHERE id = ?1", rusqlite::params![spend_event_id])
            .unwrap();
    })
    .await
    .unwrap();
    assert!(
        usage_rows(&server).is_empty(),
        "deleted row removed the only recorded firing"
    );
    let reopener = UsageAnomalyAlertState::new(
        server.state.config.clone(),
        server.state.logger.clone(),
        no_name_resolver(),
        server.state.webhook_ledger.clone(),
        server.state.webhook_notifier.clone(),
    );
    reopener.run_cycle_at(clock).await.unwrap();
    let rows = wait_for_usage_rows(&server, 1).await;
    assert_eq!(rows.len(), 1, "the missing finding re-fires after restart");
    let data = payload_of(&rows[0]);
    assert_eq!(data["key_id"].as_str().unwrap(), KEY_SPEND);
    assert_eq!(data["signal"].as_str().unwrap(), "spend");
    assert_eq!(data["evaluated_day"].as_str().unwrap(), due_day.to_string());
}

// ---------------------------------------------------------------------------
// (c) Multi-day catch-up: a one-day evaluation gap still evaluates the
//     skipped day while it is in-window, oldest-first
// ---------------------------------------------------------------------------

/// Cycle 1 (noon of D1) advances `last_evaluated_day` to D0. The
/// gateway then misses D2's noon tick, so D1 is never evaluated that
/// day. Cycle 2 (noon of D3) must process D1 (the skipped day, still
/// inside the 8-day window) before D2 — the D1 spend spike fires
/// with `evaluated_day` = D1.
#[tokio::test]
async fn multi_day_catch_up_evaluates_skipped_day() {
    let receiver = MockServer::start().await;
    common::mount_webhook_ok(&receiver).await;
    let config = seeded_config(Some(&receiver));
    let server = build_server(config).await;

    let now = Utc::now();
    let d2 = midnight_after(now, 0).pred_opt().unwrap(); // yesterday
    let d1 = d2.pred_opt().unwrap();
    let d0 = d1.pred_opt().unwrap();

    // Quiet 2/day baseline across [D0-6, D0] (d1's own 7-day window)
    // plus a quiet D2; the spike lands on D1 only. NOTE: d1's
    // baseline is [D1-7, D1] = [d0-6, d0], so the 7 flat days must
    // END at d0 — starting them 8 days back would starve d1's gate
    // (its window does not reach that far).
    for back in 0..7 {
        log_key_day(
            &server,
            KEY_CATCHUP,
            d0 - Duration::days(back),
            2,
            12,
            0,
            5.0,
            0,
            200,
            Some(MODEL),
            "catchup-base",
        );
    }
    log_key_day(&server, KEY_CATCHUP, d0, 2, 12, 0, 5.0, 0, 200, Some(MODEL), "catchup-d0");
    log_key_day(&server, KEY_CATCHUP, d1, 20, 12, 0, 5.02, 0, 200, Some(MODEL), "catchup-d1");
    log_key_day(&server, KEY_CATCHUP, d2, 2, 12, 0, 5.0, 0, 200, Some(MODEL), "catchup-d2");
    wait_for_commit(&server, 36).await;

    // Cycle 1 covers [D0-7, D0]: flat traffic, nothing fires.
    alert_state(&server).run_cycle_at(cycle_clock(d1)).await.unwrap();
    assert!(
        usage_rows(&server).is_empty(),
        "quiet days before the gap fire nothing"
    );

    // Cycle 2 catches up D1 (skipped, in-window) then D2.
    alert_state(&server)
        .run_cycle_at(cycle_clock(d2.succ_opt().unwrap()))
        .await
        .unwrap();
    let rows = wait_for_usage_rows(&server, 1).await;
    assert_eq!(rows.len(), 1, "only the skipped day's spike fires");
    let data = payload_of(&rows[0]);
    assert_eq!(data["evaluated_day"].as_str().unwrap(), d1.to_string());
    assert_eq!(data["signal"].as_str().unwrap(), "spend");
    assert_eq!(data["key_id"].as_str().unwrap(), KEY_CATCHUP);
}

// ---------------------------------------------------------------------------
// (d) Dormancy: no webhook endpoints → no ledger rows
// ---------------------------------------------------------------------------

#[tokio::test]
async fn dormant_without_endpoints_emits_nothing() {
    let config = seeded_config(None);
    let server = build_server(config).await;

    let now = Utc::now();
    let due_day = midnight_after(now, 0).pred_opt().unwrap();
    seed_spike_only(&server, due_day, KEY_SPEND).await;

    alert_state(&server)
        .run_cycle_at(cycle_clock(due_day.succ_opt().unwrap()))
        .await
        .unwrap();
    assert!(
        server.state.webhook_ledger.recent_events(200).is_empty(),
        "dormant cycle writes no ledger rows at all"
    );
}

// ---------------------------------------------------------------------------
// (e) Disabled flag → skip; hot-reload re-enable → next cycle evaluates
// ---------------------------------------------------------------------------

#[tokio::test]
async fn disabled_flag_skips_and_reenable_evaluates() {
    let receiver = MockServer::start().await;
    common::mount_webhook_ok(&receiver).await;
    let mut config = seeded_config(Some(&receiver));
    config.usage_anomaly_alerts.enabled = false;
    let server = build_server(config).await;

    let now = Utc::now();
    let due_day = midnight_after(now, 0).pred_opt().unwrap();
    seed_spike_only(&server, due_day, KEY_SPEND).await;

    let clock = cycle_clock(due_day.succ_opt().unwrap());
    alert_state(&server).run_cycle_at(clock).await.unwrap();
    assert!(usage_rows(&server).is_empty(), "disabled cycle skips");

    // Hot-reload re-enable: rewrite the shared config in place; the
    // day was never evaluated (dormancy leaves last_evaluated_day
    // untouched), so the next tick picks it up.
    {
        let mut guard = server.state.config.write().await;
        guard.usage_anomaly_alerts.enabled = true;
    }
    alert_state(&server).run_cycle_at(clock).await.unwrap();
    let rows = wait_for_usage_rows(&server, 1).await;
    assert_eq!(rows.len(), 1, "re-enabled cycle evaluates the due day");
}

// ---------------------------------------------------------------------------
// (f) Ledger failure → cycle errs, day not advanced, same-state retry
// ---------------------------------------------------------------------------

#[tokio::test]
async fn ledger_failure_errs_and_same_state_retry_succeeds() {
    let receiver = MockServer::start().await;
    common::mount_webhook_ok(&receiver).await;
    let config = seeded_config(Some(&receiver));
    let server = build_server(config).await;

    let now = Utc::now();
    let due_day = midnight_after(now, 0).pred_opt().unwrap();
    seed_spike_only(&server, due_day, KEY_SPEND).await;

    // Broken ledger: a real store whose schema is dropped out from
    // under it via a second connection — every dedup query then
    // fails, which the cycle must surface as an error.
    let broken_dir = tempfile::tempdir().unwrap();
    let broken_path = broken_dir.path().join("broken-webhooks.db");
    let broken_store = Arc::new(
        ai_gateway::webhooks::ledger::WebhookLedgerStore::new(&broken_path).unwrap(),
    );
    {
        let conn = rusqlite::Connection::open(&broken_path).unwrap();
        conn.execute_batch("DROP TABLE alert_events; DROP TABLE deliveries;").unwrap();
    }

    // Same-state retry contract: one state, the SAME due day, cycled
    // against the broken ledger first.
    let state = UsageAnomalyAlertState::new(
        server.state.config.clone(),
        server.state.logger.clone(),
        no_name_resolver(),
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

    // Restore the schema (the store's idempotent `init` on a new
    // handle recreates it), then a fresh state's next cycle
    // succeeds and fires the day. The notifier must share the
    // restored store so the ledger rows land where the cycle
    // confirms them.
    drop(state);
    ai_gateway::webhooks::ledger::WebhookLedgerStore::new(&broken_path).unwrap();
    let restored_notifier = Arc::new(
        ai_gateway::webhooks::delivery::WebhookNotifier::new(
            server.state.config.clone(),
            broken_store.clone(),
        )
        .await
        .unwrap(),
    );
    let state = UsageAnomalyAlertState::new(
        server.state.config.clone(),
        server.state.logger.clone(),
        no_name_resolver(),
        broken_store.clone(),
        restored_notifier,
    );
    state.run_cycle_at(clock).await.unwrap();
    let rows = wait_for_ledger_rows(&broken_store, 1).await;
    assert_eq!(rows.len(), 1, "same-state retry after schema restore fires the day");
    assert_eq!(payload_of(&rows[0])["signal"].as_str().unwrap(), "spend");
}

// ---------------------------------------------------------------------------
// (g) No request-path impact: the router serves a request during a cycle
// ---------------------------------------------------------------------------

#[tokio::test]
async fn router_serves_request_during_cycle() {
    let upstream = MockServer::start().await;
    common::mount_ok_mock(&upstream, 1).await;
    let receiver = MockServer::start().await;
    common::mount_webhook_ok(&receiver).await;
    let mut config = seeded_config(Some(&receiver));
    config.providers[0].base_url = Some(format!("{}/v1", upstream.uri()));
    let server = build_server(config).await;

    let now = Utc::now();
    let due_day = midnight_after(now, 0).pred_opt().unwrap();
    seed_spike_only(&server, due_day, KEY_SPEND).await;

    let app = server.build_router();
    let state = alert_state(&server);
    let cycle = state.run_cycle_at(cycle_clock(due_day.succ_opt().unwrap()));
    let (cycle_result, status) =
        tokio::join!(cycle, async { common::post_chat(app, "during-cycle").await });
    cycle_result.unwrap();
    assert_eq!(
        status,
        axum::http::StatusCode::OK,
        "router serves a request while the cycle runs"
    );
    let rows = wait_for_usage_rows(&server, 1).await;
    assert_eq!(rows.len(), 1, "cycle emitted its finding concurrently");
}

// ---------------------------------------------------------------------------
// (h) 01:00 grace gate: pre-grace cycle writes nothing and does not
// advance the day; a cycle at exactly 01:00:00 is due
// ---------------------------------------------------------------------------

#[tokio::test]
async fn grace_gate_holds_before_0100_and_fires_at_exactly_0100() {
    let receiver = MockServer::start().await;
    common::mount_webhook_ok(&receiver).await;
    let config = seeded_config(Some(&receiver));
    let server = build_server(config).await;

    let now = Utc::now();
    let due_day = midnight_after(now, 0).pred_opt().unwrap();
    seed_spike_only(&server, due_day, KEY_SPEND).await;

    // 00:30 UTC the day AFTER the due day: before the 01:00 grace,
    // so the cycle is a no-op — no ledger rows, day not advanced.
    let pre_grace = due_day
        .succ_opt()
        .unwrap()
        .and_hms_opt(0, 30, 0)
        .expect("00:30 always constructs")
        .and_utc();
    let state = alert_state(&server);
    state.run_cycle_at(pre_grace).await.unwrap();
    assert!(
        usage_rows(&server).is_empty(),
        "pre-grace cycle must not emit"
    );
    // Day not advanced: the SAME day stays due, proven behaviorally —
    // the next cycle at exactly 01:00:00 still evaluates it and fires.
    let at_grace = due_day
        .succ_opt()
        .unwrap()
        .and_hms_opt(1, 0, 0)
        .expect("01:00:00 always constructs")
        .and_utc();
    state.run_cycle_at(at_grace).await.unwrap();
    let rows = wait_for_usage_rows(&server, 1).await;
    assert_eq!(
        rows.len(),
        1,
        "cycle at exactly 01:00:00 is due and fires the day"
    );
    assert_eq!(payload_of(&rows[0])["key_id"].as_str().unwrap(), KEY_SPEND);
}

// ---------------------------------------------------------------------------
// (i) EmissionNotConfirmed: emitted rows never land in the ledger —
// the cycle errors and `last_evaluated_day` never advances
// ---------------------------------------------------------------------------

#[tokio::test]
async fn emission_not_confirmed_errs_and_day_not_advanced() {
    let receiver = MockServer::start().await;
    common::mount_webhook_ok(&receiver).await;
    let config = seeded_config(Some(&receiver));
    let server = build_server(config).await;

    let now = Utc::now();
    let due_day = midnight_after(now, 0).pred_opt().unwrap();
    seed_spike_only(&server, due_day, KEY_SPEND).await;

    // Deterministic never-lands ledger: a BEFORE INSERT trigger on
    // `alert_events` makes every emitted row insert ABORT (the
    // fire-and-forget seam swallows the failure) while every read —
    // the dedup lookup and each persistence-confirmation poll —
    // keeps succeeding and finds nothing. The cycle must exhaust
    // its bounded poll window and fail with EmissionNotConfirmed.
    let sink_dir = tempfile::tempdir().unwrap();
    let sink_path = sink_dir.path().join("sink-webhooks.db");
    let sink_store = Arc::new(
        ai_gateway::webhooks::ledger::WebhookLedgerStore::new(&sink_path).unwrap(),
    );
    {
        let conn = rusqlite::Connection::open(&sink_path).unwrap();
        conn.execute_batch(
            "CREATE TRIGGER reject_alert_event_inserts
             BEFORE INSERT ON alert_events
             BEGIN
                 SELECT RAISE(ABORT, 'sink rejects inserts');
             END;",
        )
        .unwrap();
    }
    let sink_notifier = Arc::new(
        ai_gateway::webhooks::delivery::WebhookNotifier::new(
            server.state.config.clone(),
            sink_store.clone(),
        )
        .await
        .unwrap(),
    );
    let state = Arc::new(UsageAnomalyAlertState::new(
        server.state.config.clone(),
        server.state.logger.clone(),
        no_name_resolver(),
        sink_store.clone(),
        sink_notifier,
    ));
    let clock = cycle_clock(due_day.succ_opt().unwrap());
    let result = state.run_cycle_at(clock).await;
    assert!(
        matches!(
            result.err(),
            Some(UsageAnomalyAlertError::EmissionNotConfirmed { .. })
        ),
        "cycle whose rows never land must fail with EmissionNotConfirmed"
    );

    // Day NOT advanced: with the trigger dropped, a second cycle on
    // the SAME state must still evaluate the SAME day and fire it —
    // had the failed cycle advanced `last_evaluated_day`, this
    // cycle would short-circuit and no row would ever land.
    {
        let conn = rusqlite::Connection::open(&sink_path).unwrap();
        conn.execute_batch("DROP TRIGGER reject_alert_event_inserts;")
            .unwrap();
    }
    state.run_cycle_at(clock).await.unwrap();
    let rows = wait_for_ledger_rows(&sink_store, 1).await;
    assert_eq!(rows.len(), 1, "day was not advanced; retry fires the finding");
    assert_eq!(payload_of(&rows[0])["signal"].as_str().unwrap(), "spend");
}

// ---------------------------------------------------------------------------
// (j) Catch-up boundary: with a multi-day gap, exactly the 8-day
// window `[newest_due - 7, newest_due]` is evaluated — the day at
// `newest_due - 7` is evaluated, `newest_due - 8` is dropped
// ---------------------------------------------------------------------------

#[tokio::test]
async fn catch_up_boundary_evaluates_minus7_and_drops_minus8() {
    let receiver = MockServer::start().await;
    common::mount_webhook_ok(&receiver).await;
    let config = seeded_config(Some(&receiver));
    let server = build_server(config).await;

    let now = Utc::now();
    let newest_due = midnight_after(now, 0).pred_opt().unwrap();
    let minus7 = newest_due - Duration::days(7);
    let minus8 = newest_due - Duration::days(8);

    // Identical spend spikes on separate keys: KEY_CATCHUP spikes on
    // `newest_due - 7` (the window's oldest day: evaluated), KEY_SPEND
    // spikes on `newest_due - 8` (outside: never evaluated). Each
    // day's own 7-day baseline is quiet 2/day, so ONLY the boundary
    // decides which fires. The minus8 rows reach 15 days back, beyond
    // the 9-day `usage_since` window `wait_for_commit` polls, so the
    // writer is trusted via one bounded wait on the newer rows only.
    for (day, key, tag) in [
        (minus7, KEY_CATCHUP, "boundary-in"),
        (minus8, KEY_SPEND, "boundary-out"),
    ] {
        // Baseline [day-7, day-1] quiet 2/day at $5.
        for back in 1..=7 {
            log_key_day(
                &server,
                key,
                day - Duration::days(back),
                2,
                12,
                0,
                5.0,
                0,
                200,
                Some(MODEL),
                &format!("{tag}-base"),
            );
        }
        log_key_day(&server, key, day, 20, 12, 0, 5.02, 0, 200, Some(MODEL), tag);
    }
    // The minus8 block reaches 15 days back, beyond the 9-day
    // `usage_since` window `wait_for_commit` polls, so the writer
    // is drained deterministically instead (flush checkpoints the
    // WAL after every queued entry lands).
    server.state.logger.flush();
    wait_for_commit(&server, 34).await;

    // One cycle with NO prior state: the whole 8-day window
    // `[newest_due - 7, newest_due]` is evaluated oldest-first.
    // minus7 is the window's oldest day (fires); minus8 is outside
    // (never evaluated, never fires).
    alert_state(&server)
        .run_cycle_at(cycle_clock(newest_due.succ_opt().unwrap()))
        .await
        .unwrap();
    let rows = wait_for_usage_rows(&server, 1).await;
    assert_eq!(rows.len(), 1, "only the in-window boundary day fires");
    assert_eq!(payload_of(&rows[0])["key_id"].as_str().unwrap(), KEY_CATCHUP);
    assert_eq!(
        payload_of(&rows[0])["evaluated_day"].as_str().unwrap(),
        minus7.to_string(),
        "newest_due - 7 is the oldest evaluated day"
    );
    let days: Vec<String> = rows
        .iter()
        .map(|row| payload_of(row)["evaluated_day"].as_str().unwrap().to_string())
        .collect();
    assert!(!days.contains(&minus8.to_string()), "newest_due - 8 is never evaluated");
}

// ---------------------------------------------------------------------------
// (k) In-flight guard: an overlapping run_cycle_at returns Ok early
// without duplicating ledger rows
// ---------------------------------------------------------------------------

#[tokio::test]
async fn overlapping_cycles_single_fire() {
    let receiver = MockServer::start().await;
    common::mount_webhook_ok(&receiver).await;
    let config = seeded_config(Some(&receiver));
    let server = build_server(config).await;

    let now = Utc::now();
    let due_day = midnight_after(now, 0).pred_opt().unwrap();
    seed_spike_only(&server, due_day, KEY_SPEND).await;

    // Two concurrent cycles on the SAME state: the in-flight guard
    // admits one; the overlapping tick returns Ok early. Both
    // complete Ok and exactly ONE finding row lands.
    let state = Arc::new(UsageAnomalyAlertState::new(
        server.state.config.clone(),
        server.state.logger.clone(),
        no_name_resolver(),
        server.state.webhook_ledger.clone(),
        server.state.webhook_notifier.clone(),
    ));
    let clock = cycle_clock(due_day.succ_opt().unwrap());
    let a = tokio::spawn({
        let state = Arc::clone(&state);
        async move { state.run_cycle_at(clock).await.is_ok() }
    });
    let b = tokio::spawn({
        let state = Arc::clone(&state);
        async move { state.run_cycle_at(clock).await.is_ok() }
    });
    let (ok_a, ok_b) = (a.await.unwrap(), b.await.unwrap());
    assert!(ok_a && ok_b, "both cycles resolve Ok (overlap returns early)");
    let rows = wait_for_usage_rows(&server, 1).await;
    assert_eq!(rows.len(), 1, "in-flight guard prevents duplicate firings");
}
