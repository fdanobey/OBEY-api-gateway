//! Forecast alert wiring (spec: spend-pace-forecasting, Task 5) —
//! accrual-hook integration tests.
//!
//! Proves the `budget.forecast.warning` funnel end-to-end through the
//! real off-path accrual chains, reusing the shared webhook rig in
//! `common` (wiremock upstream + wiremock receiver + isolated DBs):
//!
//! - provider scope: windowed over-pace fires exactly once per window;
//! a later same-window accrual never re-fires (Reqs 3.1-3.3);
//! payload carries the projection fields with `window_end` only
//! (Req 3.4)
//! - VK scope: windowed key fires once; admin reset re-arms (Req 3.3)
//! - manual provider: trailing run-rate projection from seeded ledger
//! rows fires once with `horizon_end` only (Reqs 3.1, 3.4)
//! - manual VK: trailing run-rate from key-attributed ledger rows
//! fires once with `horizon_end` only (Req 3.1)
//! - manual ledger failure: silent forecast skip while threshold
//! events still fire (Req 3.7).
//!
//! Windowed wall-clock determinism: the window anchor is BACKDATED
//! via SQL after the first accrual stamps the live period boundary —
//! the same backdating trick the rollover-rearm step always used —
//! to `max(period_start, now − 7/13 · remaining)`, pinning the
//! elapsed fraction in [5%, 35%] of the window for EVERY wall clock
//! (past the 5% projection gate, under the ~48% fraction where the
//! over-pace seed spend stays below the 80% warn line). The anchor
//! never leaves its calendar period, so no lazy rollover or VK auth
//! window reset interferes. Production boundary math is reused via
//! `metrics::current_window_start_for_policy` / `metrics::
//! next_reset_epoch` — no forked copies. The windowed assertions
//! ALWAYS execute: [`windowed_rig`] panics with a clear message in
//! the one corner where no policy qualifies (the first ~72 minutes
//! of a UTC Monday that is also the 1st/2nd of its month) instead of
//! silently skipping.

use axum::body::Body;
use axum::http::{Request, StatusCode};
use tower::ServiceExt;
use wiremock::MockServer;

use ai_gateway::config::*;
use ai_gateway::gateway::GatewayServer;
use ai_gateway::metrics::forecast::ForecastMethod;
use ai_gateway::virtual_keys::models::{BudgetWindow, CreateKeyParams, UsageRecord};
use ai_gateway::webhooks::events::{BudgetAlertEvent, Scope};

use crate::common::{
 budget_test_config, configure_webhook_receiver, mount_ok_mock, mount_webhook_ok, post_chat,
 post_chat_with_key, sample_log_entry, verify_signature_and_parse, wait_for_deliveries,
 wait_for_store_spend, webhook_test_provider, webhook_with_expensive_group, LogEntryOverrides,
 WEBHOOK_TEST_PROVIDER,
};

const PROVIDER_NAME: &str = WEBHOOK_TEST_PROVIDER;
const DAY: i64 = 86_400;
const LIMIT_USD: f64 = 100.0;

// ---------------------------------------------------------------------------
// Rig
// ---------------------------------------------------------------------------

/// The deterministic windowed setup (clock-independence fix): pick the
/// finest policy whose current period admits a same-period anchor
/// that (a) passes the 5% projection gate, (b) keeps the over-pace
/// seed spend and every later accrual total under the 80% warn line
/// (forecast-only events), and (c) whose elapsed-minus-one-period
/// re-arm anchor still projects over the limit. The anchor is
/// backdated to `max(period_start, now − 7/13 · remaining)`, pinning
/// the elapsed fraction in [5%, ~42%] of the window regardless of
/// when the test runs; it never leaves its calendar period, so no
/// lazy rollover or VK auth window reset interferes.
///
/// Returns `(policy, anchor, elapsed_s, window_len_s, boundary_epoch)`
/// where `boundary_epoch` is the projection's `window_end` (the next
/// UTC period boundary after `now`, exactly what
/// `next_reset_epoch(max(anchor, now))` computes in production).
///
/// Panics (never silently skips) when no policy qualifies — a rare
/// early-month pre-dawn corner (e.g. the first ~72 minutes of a UTC
/// Monday-the-1st/2nd) where every period is younger than its own
/// 5% gate; the message names the retry window.
fn windowed_rig() -> (BudgetResetPolicy, i64, i64, i64, i64) {
 let now = chrono::Utc::now().timestamp();
 for policy in [
 BudgetResetPolicy::Daily,
 BudgetResetPolicy::Weekly,
 BudgetResetPolicy::Monthly,
 ] {
 let start = ai_gateway::metrics::current_window_start_for_policy(policy);
 let Some(boundary) = ai_gateway::metrics::next_reset_epoch(policy, now) else {
 continue;
 };
 let remaining = boundary - now;
 // Backdate 7/13 of the remaining time (elapsed fraction 7/20 when
 // the clamp is inactive), clamped inside the current period.
 let backdate = (7 * remaining) / 13;
 let anchor = (now - backdate.max(0)).max(start);
 let elapsed = now - anchor;
 let window_len = boundary - anchor;
 // 5% projection gate in the exact integer form production uses
 // (`elapsed * 20 < duration` → insufficient data), plus strictly
 // inside the window.
 if elapsed <= 0 || window_len <= elapsed || elapsed * 20 < window_len {
 continue;
 }
 // Forecast-only bound: seed + every later accrual total (≤ seed+9)
 // must stay under the 80% warn line and the limit.
 let seed = overpace_seed_spend(LIMIT_USD, elapsed, window_len);
 if seed + 9.0 >= LIMIT_USD * 0.8 {
 continue;
 }
 // Re-arm crossing: the anchor backdated one more `elapsed` still
 // ends at the same boundary with doubled elapsed time; the pace
 // must still project past the limit there.
 let rearm_len = window_len + elapsed;
 let rearm_elapsed = 2 * elapsed;
 if rearm_elapsed * 20 < rearm_len {
 continue;
 }
 if (seed + 9.0) * rearm_len as f64 / rearm_elapsed as f64 <= LIMIT_USD {
 continue;
 }
 return (policy, anchor, elapsed, window_len, boundary);
 }
 panic!(
 "no windowed policy qualifies at this instant (early-month pre-dawn \
 corner; daily qualifies from 01:12 UTC and unclamped from 08:24 UTC) — \
 rerun the suite a few minutes later"
 );
}

/// Seed durable spend at the midpoint between the on-pace line
/// (`limit × elapsed/window`) and the limit: strictly over-pace
/// (projects past the limit within the window) while staying safely
/// under the exhausted line for any elapsed fraction in (0, 1).
fn overpace_seed_spend(limit: f64, elapsed_s: i64, window_s: i64) -> f64 {
 let fraction = elapsed_s as f64 / window_s as f64;
 limit * (fraction + (1.0 - fraction) / 2.0)
}

async fn wait_for_key_spend(server: &GatewayServer, key_id: &str, min_usd: f64) {
 for _ in 0..200 {
 if let Ok(Some(state)) = server
 .state
 .virtual_key_manager
 .key_budget_state(key_id)
 .await
 {
 if state.current_spend_usd >= min_usd {
 return;
 }
 }
 tokio::time::sleep(std::time::Duration::from_millis(50)).await;
 }
 panic!("durable spend for key {key_id} never reached {min_usd} USD");
}

/// Wait until the ledger holds exactly `expected` alert events and the
/// count is stable across three polls (bounded absence proof).
async fn assert_ledger_event_count(server: &GatewayServer, expected: usize, poll_ms: u64) {
 let ledger = &server.state.webhook_ledger;
 let mut stable = 0;
 let mut last = usize::MAX;
 let deadline = std::time::Instant::now() + std::time::Duration::from_millis(poll_ms);
 while std::time::Instant::now() < deadline {
 tokio::time::sleep(std::time::Duration::from_millis(50)).await;
 let rows = ledger.recent_events(50);
 let current = rows.len();
 if current != last {
 stable = 0;
 last = current;
 } else {
 stable += 1;
 if current == expected && stable >= 3 {
 return;
 }
 }
 }
 assert_eq!(
 last, expected,
 "ledger settled on {last} events, expected exactly {expected}"
 );
}

fn parse_forecast_event(deliveries: &[wiremock::Request]) -> BudgetAlertEvent {
 let events: Vec<BudgetAlertEvent> = deliveries.iter().map(verify_signature_and_parse).collect();
 events
 .into_iter()
 .find(|event| event.event_type == BudgetEventType::BudgetForecastWarning)
 .unwrap_or_else(|| panic!("a budget.forecast.warning delivery exists"))
}

// ---------------------------------------------------------------------------
// Provider scope: windowed over-pace fires once, re-arms on rollover
// ---------------------------------------------------------------------------

/// Validates: Reqs 3.1-3.4 (provider windowed projection, once per
/// window, rollover re-arm, payload fields)
#[tokio::test]
async fn provider_windowed_overpace_fires_forecast_once_and_rearms_on_rollover() {
 let (policy, anchor, elapsed, window_len, boundary) = windowed_rig();
 let dir = tempfile::tempdir().expect("temp dir");
 let upstream = MockServer::start().await;
 let receiver = MockServer::start().await;

 let mut config = budget_test_config(Some(dir.path()), None);
 config.providers = vec![webhook_test_provider(
 &upstream.uri(),
 Some(ProviderBudgetConfig {
 limit_usd: LIMIT_USD,
 reset_policy: Some(policy),
 alert_threshold_percent: None,
 }),
 )];
 webhook_with_expensive_group(&mut config);
 configure_webhook_receiver(&mut config, &receiver);
 let server = GatewayServer::new(config, None).await.expect("server builds");
 let app = server.build_router();
 mount_ok_mock(&upstream, 4).await;
 mount_webhook_ok(&receiver).await;

 let seed = overpace_seed_spend(LIMIT_USD, elapsed, window_len);
 // Every later accrual total must stay under the warn line
 // (forecast-only events) and the limit: elapsed fraction ≤ 35% ⇒
 // seed ≤ $67.50 ⇒ seed + $9 ≤ $76.50 < $80 warn < $100 limit.
 assert!(
 seed + 9.0 < LIMIT_USD * 0.8,
 "seed {seed} must stay under the warn line for forecast-only events"
 );

 // Accrual #1: $3.00 lands durable at the live period boundary (the
 // first accrual stamps `current_window_start_for_policy`). $3 at the
 // live elapsed fraction is on-pace: nothing fires.
 let status = post_chat(app.clone(), "one").await;
 assert_eq!(status, StatusCode::OK);
 wait_for_store_spend(&server, PROVIDER_NAME, 3.0).await;
 assert_ledger_event_count(&server, 0, 600).await;

 // Deterministic backdate (clock-independence fix): move the row's
 // anchor to the rig's mid-window instant — the same SQL the
 // rollover-rearm step uses. The anchor stays inside its calendar
 // period, and the boundary cache seeded from the live anchor keeps
 // the lazy rollover check short-circuited (not expired). Then top
 // the row up to the over-pace seed through the store's own CAS
 // upsert (anchored at the backdated `window_start`).
 {
 let budgets_db = dir.path().join("budgets.db");
 let conn = rusqlite::Connection::open(&budgets_db).expect("open budgets.db");
 conn.execute(
 "UPDATE provider_budgets SET window_start = ?2 WHERE provider = ?1",
 rusqlite::params![PROVIDER_NAME, anchor],
 )
 .expect("backdate window_start");
 let store = server
 .state
 .metrics
 .provider_budget_store()
 .expect("store attached");
 store
 .add_spend(PROVIDER_NAME, seed - 3.0, anchor)
 .expect("seed durable spend");
 }

 // Accrual #2: $3.00 more → durable ≈ seed + 3 → over-pace
 // projection crosses the limit within the window → fires once.
 let status = post_chat(app.clone(), "two").await;
 assert_eq!(status, StatusCode::OK);
 wait_for_store_spend(&server, PROVIDER_NAME, seed + 3.0).await;
 let deliveries = wait_for_deliveries(&receiver, 1).await;
 let event = parse_forecast_event(&deliveries);
 assert_eq!(event.data.scope, Scope::Provider);
 assert_eq!(event.data.name.as_deref(), Some(PROVIDER_NAME));
 assert_eq!(event.data.limit_usd, Some(LIMIT_USD));
 let spent = event.data.spent_usd.expect("spent_usd payload");
 assert!(
 (spent - (seed + 3.0)).abs() < 1e-6,
 "spent {spent} must be the seeded total plus the crossing accrual"
 );
 assert_eq!(event.data.window_start, Some(anchor));
 let forecast = event.data.forecast.as_ref().expect("forecast payload");
 assert_eq!(forecast.method, ForecastMethod::WindowPace);
 assert!(forecast.projected_usd > LIMIT_USD, "over-pace projection");
 assert_eq!(forecast.window_end, Some(boundary));
 assert!(
 forecast.horizon_end.is_none(),
 "windowed scope carries window_end only"
 );
 assert_eq!(forecast.threshold_kind, "forecast");

 // Accrual #3 (same window): no re-fire (once per window).
 let status = post_chat(app.clone(), "three").await;
 assert_eq!(status, StatusCode::OK);
 wait_for_store_spend(&server, PROVIDER_NAME, seed + 6.0).await;
 assert_ledger_event_count(&server, 1, 600).await;

 // Window-rollover re-arm (Req 3.3): move the row's anchor back one
 // more elapsed-window — the durable fact a rollover stamps (a NEW
 // (scope, anchor) key for the fired registry). The projection
 // window still ends at the same next boundary (it is computed from
 // max(anchor, now)), with double the elapsed time, so the over-pace
 // pace still crosses. Prior spend stayed under the limit, so no
 // budget.reset accompanies it.
 let rearm_anchor = anchor - elapsed;
 {
 let budgets_db = dir.path().join("budgets.db");
 let conn = rusqlite::Connection::open(&budgets_db).expect("open budgets.db");
 conn.execute(
 "UPDATE provider_budgets SET window_start = ?2 WHERE provider = ?1",
 rusqlite::params![PROVIDER_NAME, rearm_anchor],
 )
 .expect("backdate window_start for rearm");
 }
 let status = post_chat(app, "four").await;
 assert_eq!(status, StatusCode::OK);
 {
 // The accrual CAS'd into the re-armed anchor; the durable total is
 // prior spend + $3.
 let store = server
 .state
 .metrics
 .provider_budget_store()
 .expect("store attached");
 let deadline = std::time::Instant::now() + std::time::Duration::from_millis(5_000);
 loop {
 if let Ok(Some((spend, anchor_row))) = store.get_spend_and_window(PROVIDER_NAME) {
 if spend >= seed + 9.0 && anchor_row == rearm_anchor {
 break;
 }
 }
 assert!(
 std::time::Instant::now() < deadline,
 "accrual never landed in the re-armed window"
 );
 tokio::time::sleep(std::time::Duration::from_millis(50)).await;
 }
 }
 let deliveries = wait_for_deliveries(&receiver, 2).await;
 let forecast_events: Vec<BudgetAlertEvent> = deliveries
 .iter()
 .map(verify_signature_and_parse)
 .filter(|event| event.event_type == BudgetEventType::BudgetForecastWarning)
 .collect();
 assert_eq!(
 forecast_events.len(),
 2,
 "the new-window accrual re-armed the forecast threshold"
 );
 assert_eq!(
 forecast_events[1].data.window_start, Some(rearm_anchor),
 "second event carries the re-armed anchor"
 );
}

// ---------------------------------------------------------------------------
// Provider scope: manual run-rate + ledger-failure silent skip
// ---------------------------------------------------------------------------

/// Validates: Reqs 3.1, 3.4 (manual trailing run-rate fires with
/// horizon payload)
#[tokio::test]
async fn provider_manual_run_rate_fires_forecast_from_ledger() {
 let dir = tempfile::tempdir().expect("temp dir");
 let upstream = MockServer::start().await;
 let receiver = MockServer::start().await;
 let (server, app) = {
 let mut config = budget_test_config(Some(dir.path()), None);
 config.providers = vec![webhook_test_provider(
 &upstream.uri(),
 Some(ProviderBudgetConfig {
 // $14 limit: warn line $11.2 stays above both
 // accrual totals ($5, $8) — forecast-only events —
 // while projected 5 + 10 = $15 crosses it.
 limit_usd: 14.0,
 reset_policy: None,
 alert_threshold_percent: None,
 }),
 )];
 webhook_with_expensive_group(&mut config);
 configure_webhook_receiver(&mut config, &receiver);
 let server = GatewayServer::new(config, None).await.expect("server builds");
 let app = server.build_router();
 (server, app)
 };
 mount_ok_mock(&upstream, 2).await;
 mount_webhook_ok(&receiver).await;

 // 10 distinct recent UTC days at $1/day through the real logger
 // write path: trailing 30-day spend = $10, history_days = 10 ≥ 7.
 let now_epoch = chrono::Utc::now().timestamp();
 for day in 0..10 {
 let entry = sample_log_entry(
 &format!("trace-forecast-manual-{day}"),
 now_epoch - day * DAY - 3_600,
 PROVIDER_NAME,
 "gpt-4.1-nano",
 1.0,
 LogEntryOverrides::default(),
 );
 server.state.logger.log(entry).expect("seed log entry");
 }
 server.state.logger.flush();

 // Durable spend $2 (manual first accrual stamps "now").
 {
 let store = server
 .state
 .metrics
 .provider_budget_store()
 .expect("store attached");
 store
 .add_spend(PROVIDER_NAME, 2.0, now_epoch)
 .expect("seed durable spend");
 }

 // Accrual #1: +$3 → durable $5 → projected 5 + 10 = $15 > $14 →
 // ExhaustsWithinWindow fires. $5/$14 = 36% < 80% warn — forecast
 // only.
 let status = post_chat(app.clone(), "one").await;
 assert_eq!(status, StatusCode::OK);
 wait_for_store_spend(&server, PROVIDER_NAME, 5.0).await;
 let deliveries = wait_for_deliveries(&receiver, 1).await;
 let event = parse_forecast_event(&deliveries);
 assert_eq!(event.data.scope, Scope::Provider);
 let forecast = event.data.forecast.as_ref().expect("forecast payload");
 assert_eq!(forecast.method, ForecastMethod::TrailingRunRate);
 assert!(
 (forecast.projected_usd - 15.0).abs() < 1e-6,
 "projected = spent(5) + trailing(10)"
 );
 assert!(
 forecast.horizon_end.is_some() && forecast.window_end.is_none(),
 "manual scope carries horizon_end only"
 );

 // Same-window accrual: no re-fire ($8/$14 = 57%, still under the
 // warn line — the ledger count stays forecast-only).
 let status = post_chat(app, "two").await;
 assert_eq!(status, StatusCode::OK);
 wait_for_store_spend(&server, PROVIDER_NAME, 8.0).await;
 assert_ledger_event_count(&server, 1, 600).await;
}

/// Validates: Req 3.7 — an unreadable ledger (missing file) silently
/// skips the forecast while threshold events still fire.
#[tokio::test]
async fn provider_manual_ledger_failure_skips_forecast_silently() {
 let dir = tempfile::tempdir().expect("temp dir");
 let upstream = MockServer::start().await;
 let receiver = MockServer::start().await;
 let (server, app) = {
 let mut config = budget_test_config(Some(dir.path()), None);
 config.providers = vec![webhook_test_provider(
 &upstream.uri(),
 Some(ProviderBudgetConfig {
 limit_usd: 5.0,
 reset_policy: None,
 alert_threshold_percent: None,
 }),
 )];
 webhook_with_expensive_group(&mut config);
 configure_webhook_receiver(&mut config, &receiver);
 let server = GatewayServer::new(config, None).await.expect("server builds");
 let app = server.build_router();
 (server, app)
 };
 mount_ok_mock(&upstream, 2).await;
 mount_webhook_ok(&receiver).await;

 // Break the ledger path after server build: every manual forecast
 // read now fails to open → None → silent skip.
 {
 let mut config = server.state.config.write().await;
 config.logging.database_path = dir
 .path()
 .join("no-such-logs.db")
 .to_string_lossy()
 .into_owned();
 }

 // $3.00 = 60% (< 80%): nothing fires.
 let status = post_chat(app.clone(), "one").await;
 assert_eq!(status, StatusCode::OK);
 wait_for_store_spend(&server, PROVIDER_NAME, 3.0).await;
 assert_ledger_event_count(&server, 0, 600).await;

 // $6.00 = 120%: warning + exhausted fire; NO forecast event.
 let status = post_chat(app, "two").await;
 assert_eq!(status, StatusCode::OK);
 wait_for_store_spend(&server, PROVIDER_NAME, 6.0).await;
 let deliveries = wait_for_deliveries(&receiver, 2).await;
 let events: Vec<BudgetAlertEvent> = deliveries.iter().map(verify_signature_and_parse).collect();
 assert_eq!(events.len(), 2, "warn + exhausted exactly");
 for event in &events {
 assert_ne!(
 event.event_type,
 BudgetEventType::BudgetForecastWarning,
 "ledger failure must silently skip the forecast event"
 );
 }
 assert!(events.iter().any(|e| e.event_type == BudgetEventType::BudgetWarning));
 assert!(events.iter().any(|e| e.event_type == BudgetEventType::BudgetExhausted));
}

// ---------------------------------------------------------------------------
// VK scope
// ---------------------------------------------------------------------------

fn create_defaults() -> CreateKeyParams {
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

/// Validates: Reqs 3.1-3.4 (VK windowed projection fires once, admin
/// reset re-arms, payload fields)
#[tokio::test]
async fn vk_windowed_overpace_fires_once_and_rearms_on_admin_reset() {
 let (policy, anchor, elapsed, window_len, _boundary) = windowed_rig();
 let dir = tempfile::tempdir().expect("temp dir");
 let upstream = MockServer::start().await;
 let receiver = MockServer::start().await;

 let mut config = budget_test_config(Some(dir.path()), None);
 config.providers = vec![webhook_test_provider(&upstream.uri(), None)];
 webhook_with_expensive_group(&mut config);
 config.virtual_keys.enforcement = EnforcementMode::Required;
 configure_webhook_receiver(&mut config, &receiver);
 let server = GatewayServer::new(config, None).await.expect("server builds");
 let app = server.build_router();
 mount_ok_mock(&upstream, 4).await;
 mount_webhook_ok(&receiver).await;

 let created = server
 .state
 .virtual_key_manager
 .create_key(CreateKeyParams {
 name: Some("windowed-forecast-key".to_string()),
 budget_limit_usd: Some(LIMIT_USD),
 budget_window: Some(match policy {
 BudgetResetPolicy::Daily => BudgetWindow::Daily,
 BudgetResetPolicy::Weekly => BudgetWindow::Weekly,
 BudgetResetPolicy::Monthly => BudgetWindow::Monthly,
 BudgetResetPolicy::Manual => unreachable!(),
 }),
 ..create_defaults()
 })
 .await
 .expect("create key");

 let seed = overpace_seed_spend(LIMIT_USD, elapsed, window_len);
 assert!(
 seed + 6.0 < LIMIT_USD * 0.8,
 "seed {seed} must stay under the warn line for forecast-only events"
 );

 // Align the key's anchor to the rig's mid-window instant (creation
 // stamps wall-clock now; the rig anchor is inside the current
 // calendar period, so the auth-path lazy window reset never fires).
 {
 let keys_db = dir.path().join("keys.db");
 let conn = rusqlite::Connection::open(&keys_db).expect("open keys.db");
 conn.execute(
 "UPDATE virtual_keys SET window_start_timestamp = ?2 WHERE id = ?1",
 rusqlite::params![created.id, anchor],
 )
 .expect("align key window anchor");
 }

 // Accrual #1 through the real funnel: $3.00 durable — on-pace for
 // the seed target, nothing fires yet.
 let status = post_chat_with_key(app.clone(), &created.key, "one").await;
 assert_eq!(status, StatusCode::OK);
 wait_for_key_spend(&server, &created.id, 3.0).await;
 assert_ledger_event_count(&server, 0, 600).await;

 // Seed to the over-pace line via direct usage records; each
 // record's own off-path evaluation stays under the line until the
 // last one crosses it, firing the forecast exactly once.
 {
 let mut seeded = 3.0;
 let per_record = (seed - 3.0) / 8.0;
 for i in 0..8 {
 let cost = if i == 7 { seed - seeded } else { per_record };
 seeded += cost;
 server
 .state
 .virtual_key_manager
 .record_usage(UsageRecord {
 key_id: created.id.clone(),
 model_group: "budget-group".to_string(),
 model: "gpt-test".to_string(),
 input_tokens: 0,
 output_tokens: 0,
 cost_usd: cost,
 timestamp: chrono::Utc::now(),
 })
 .await
 .expect("seed key usage");
 tokio::time::sleep(std::time::Duration::from_millis(30)).await;
 }
 wait_for_key_spend(&server, &created.id, seed).await;
 }

 // Accrual #2: +$3 → durable ≈ seed + 3 → over-pace fires once.
 let status = post_chat_with_key(app.clone(), &created.key, "two").await;
 assert_eq!(status, StatusCode::OK);
 wait_for_key_spend(&server, &created.id, seed + 3.0).await;
 let deliveries = wait_for_deliveries(&receiver, 1).await;
 let event = parse_forecast_event(&deliveries);
 assert_eq!(event.data.scope, Scope::VirtualKey);
 assert_eq!(event.data.key_id.as_deref(), Some(created.id.as_str()));
 assert_eq!(event.data.name.as_deref(), Some("windowed-forecast-key"));
 let forecast = event.data.forecast.as_ref().expect("forecast payload");
 assert_eq!(forecast.method, ForecastMethod::WindowPace);
 assert!(forecast.projected_usd > LIMIT_USD);
 assert!(forecast.window_end.is_some() && forecast.horizon_end.is_none());

 // Same-window accrual: no additional forecast events.
 let status = post_chat_with_key(app.clone(), &created.key, "three").await;
 assert_eq!(status, StatusCode::OK);
 wait_for_key_spend(&server, &created.id, seed + 6.0).await;
 assert_ledger_event_count(&server, 1, 600).await;

 // Admin reset re-arms: budget.reset(admin) emitted; fresh young
 // window accrual fires nothing new.
 let reset_req = Request::post(format!("/admin/keys/{}/reset", created.id))
 .body(Body::empty())
 .unwrap();
 let resp = app.clone().oneshot(reset_req).await.unwrap();
 assert_eq!(resp.status(), StatusCode::OK);
 let deliveries = wait_for_deliveries(&receiver, 2).await;
 let events: Vec<BudgetAlertEvent> = deliveries.iter().map(verify_signature_and_parse).collect();
 assert!(
 events
 .iter()
 .any(|e| e.event_type == BudgetEventType::BudgetReset),
 "admin reset emitted budget.reset"
 );
 let status = post_chat_with_key(app, &created.key, "four").await;
 assert_eq!(status, StatusCode::OK);
 wait_for_key_spend(&server, &created.id, 3.0).await;
 let forecast_count = server
 .state
 .webhook_ledger
 .recent_events(50)
 .iter()
 .filter(|row| row.event_type == "budget.forecast.warning")
 .count();
 assert_eq!(forecast_count, 1, "forecast fired exactly once per window");
}

/// Validates: Reqs 3.1, 3.4 — a lifetime (manual-window) VK whose
/// key-attributed trailing 30-day run rate projects exhaustion inside
/// the horizon fires `budget.forecast.warning` exactly once with the
/// `horizon_end`-only payload shape.
#[tokio::test]
async fn vk_manual_run_rate_fires_forecast_from_ledger() {
 let dir = tempfile::tempdir().expect("temp dir");
 let upstream = MockServer::start().await;
 let receiver = MockServer::start().await;

 let mut config = budget_test_config(Some(dir.path()), None);
 config.providers = vec![webhook_test_provider(&upstream.uri(), None)];
 webhook_with_expensive_group(&mut config);
 config.virtual_keys.enforcement = EnforcementMode::Required;
 configure_webhook_receiver(&mut config, &receiver);
 let server = GatewayServer::new(config, None).await.expect("server builds");
 let app = server.build_router();
 mount_ok_mock(&upstream, 2).await;
 mount_webhook_ok(&receiver).await;

 let created = server
 .state
 .virtual_key_manager
 .create_key(CreateKeyParams {
 // No budget_window → lifetime scope, trailing run-rate path.
 name: Some("lifetime-forecast-key".to_string()),
 budget_limit_usd: Some(14.0),
 budget_window: None,
 ..create_defaults()
 })
 .await
 .expect("create key");

 // Seed the ledger like the dashboard forecast tests do: 10 distinct
 // recent UTC days at $1/day attributed to the key → trailing 30-day
 // spend $10, history_days 10 ≥ 7.
 let now_epoch = chrono::Utc::now().timestamp();
 for day in 0..10 {
 let entry = sample_log_entry(
 &format!("trace-vk-forecast-{day}"),
 now_epoch - day * DAY - 3_600,
 PROVIDER_NAME,
 "gpt-4.1-nano",
 1.0,
 LogEntryOverrides {
 virtual_key_id: Some(created.id.clone()),
 ..Default::default()
 },
 );
 server.state.logger.log(entry).expect("seed log entry");
 }
 server.state.logger.flush();

 // Durable key spend $2 (lifetime first accrual stamps "now"). The
 // record's own off-path evaluation: projected 2 + 10 = $12 < $14 —
 // under the limit, nothing fires yet.
 server
 .state
 .virtual_key_manager
 .record_usage(UsageRecord {
 key_id: created.id.clone(),
 model_group: "budget-group".to_string(),
 model: "gpt-test".to_string(),
 input_tokens: 0,
 output_tokens: 0,
 cost_usd: 2.0,
 timestamp: chrono::Utc::now(),
 })
 .await
 .expect("seed key usage");
 wait_for_key_spend(&server, &created.id, 2.0).await;
 assert_ledger_event_count(&server, 0, 600).await;

 // Accrual #1 through the real funnel: +$3 → durable $5, projected
 // 5 + 10 = $15 > $14 limit → fires once. $5/$14 = 36% < 80% warn —
 // forecast only.
 let status = post_chat_with_key(app.clone(), &created.key, "one").await;
 assert_eq!(status, StatusCode::OK);
 wait_for_key_spend(&server, &created.id, 5.0).await;
 let deliveries = wait_for_deliveries(&receiver, 1).await;
 let event = parse_forecast_event(&deliveries);
 assert_eq!(event.data.scope, Scope::VirtualKey);
 assert_eq!(event.data.key_id.as_deref(), Some(created.id.as_str()));
 assert_eq!(event.data.name.as_deref(), Some("lifetime-forecast-key"));
 let forecast = event.data.forecast.as_ref().expect("forecast payload");
 assert_eq!(forecast.method, ForecastMethod::TrailingRunRate);
 assert!(
 (forecast.projected_usd - 15.0).abs() < 1e-6,
 "projected = spent(5) + trailing(10)"
 );
 assert!(
 forecast.horizon_end.is_some() && forecast.window_end.is_none(),
 "manual VK scope carries horizon_end only"
 );

 // Same-window accrual: no re-fire ($8/$14 = 57% < 80% warn).
 let status = post_chat_with_key(app, &created.key, "two").await;
 assert_eq!(status, StatusCode::OK);
 wait_for_key_spend(&server, &created.id, 8.0).await;
 assert_ledger_event_count(&server, 1, 600).await;
}
