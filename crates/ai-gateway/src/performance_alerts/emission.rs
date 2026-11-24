//! Emission orchestrator and background loop for performance alerts
//! (spec: performance-alerts, design Component 3).
//!
//! Bridges the ledger aggregates ([`crate::performance_alerts::query`])
//! and the pure evaluation core ([`crate::performance_alerts::state`])
//! to the generalized webhook delivery seam
//! ([`WebhookNotifier::emit_value`], design D9): once the last
//! complete UTC day is past its 01:00 grace, the cycle fetches the
//! 8-day window for every dimension on the blocking pool, evaluates
//! each scope against its 7-day baseline, suppresses findings the
//! delivery ledger already recorded for that day (design D5 restart
//! dedup), and emits the rest as `performance.anomaly` envelopes.
//!
//! Dormancy contract (mirrors the deprecation emission): when
//! `performance_alerts.enabled` is false or
//! `config.webhooks.endpoints` is empty the cycle returns before any
//! query or write, and `last_evaluated_day` is left untouched so a
//! later re-enable resumes (design, Error Handling 4).
//!
//! Failure discipline (design, Error Handling 1): any ledger or
//! stats failure aborts the cycle with an error WITHOUT advancing
//! `last_evaluated_day`, so the hourly tick retries; a partially
//! emitted day is finished by the per-cycle ledger dedup, never
//! re-fired. Persistence is confirmed BEFORE the advance: the emit
//! seam is fire-and-forget, so the cycle re-reads the ledger with a
//! short bounded poll and fails itself when an intended firing has
//! no recorded row (fix M1). Restart safety is the same per-finding
//! dedup — no day is ever seeded from the ledger (that would mark a
//! partially recorded day complete and suppress its remaining
//! findings); a restart re-evaluates at most the pending day (three
//! bounded dimension queries, fix M2).
//!
//! Loop discipline (the `model_deprecations` scanner shape): the
//! state is built once in `GatewayServer::new`, and the hourly loop —
//! first tick immediate at startup (restart catch-up, design D7) —
//! is spawned ONLY in `start_with_shutdown`, never at construction,
//! so oneshot tests start no timers.

use std::collections::{HashMap, HashSet};
use std::sync::{Arc, Mutex};
use std::sync::atomic::{AtomicBool, Ordering};
use std::time::Duration;

use chrono::{DateTime, Days, NaiveDate, NaiveTime, SecondsFormat, Utc};
use rusqlite::{Connection, OpenFlags};
use tokio::sync::{RwLock, watch};

use crate::config::{BudgetEventType, Config};
use crate::logger::RequestLogger;
use crate::performance_alerts::query::{fetch_daily_stats, Dimension, QueryError, ScopeKey};
use crate::performance_alerts::state::{evaluate, DailyStats, Scope};
use crate::webhooks::delivery::WebhookNotifier;
use crate::webhooks::events::PerformanceAnomalyData;
use crate::webhooks::ledger::{AlertEventRow, WebhookLedgerError, WebhookLedgerStore};

/// Wire event type (ledger row + subscription filter key): the serde
/// name of [`BudgetEventType::PerformanceAnomaly`].
const EVENT_TYPE: &str = "performance.anomaly";

/// Background cadence between cycle attempts (design D7: hourly
/// tick; the `last_evaluated_day` short-circuit makes re-checks of an
/// already-evaluated day query-free).
const CYCLE_CADENCE: Duration = Duration::from_secs(3_600);

/// Polls of the delivery ledger while confirming persistence (fix
/// M1): ~10 × 200ms = ~2s total bounded wait for the fire-and-forget
/// emit seam to land its rows.
const PERSISTENCE_POLLS: usize = 10;
/// Sleep between persistence confirmation polls (fix M1).
const PERSISTENCE_POLL_INTERVAL: Duration = Duration::from_millis(200);

/// Cycle failure: ledger or stats query errors (and a panicked
/// blocking task) surface as errors so the caller retries on the
/// next tick — `last_evaluated_day` only advances on a fully
/// successful cycle.
#[derive(Debug, thiserror::Error)]
pub enum PerformanceAlertError {
    #[error("performance alerts ledger query failed: {0}")]
    Ledger(#[from] WebhookLedgerError),
    #[error("performance alerts stats query failed: {0}")]
    Stats(#[from] QueryError),
    #[error("performance alerts blocking task failed: {0}")]
    Task(#[from] tokio::task::JoinError),
    /// A finding the cycle intended to fire never landed in the
    /// delivery ledger (fix M1: persistence-before-advance).
    #[error(
        "performance alerts emission not confirmed in the ledger for {day}: missing {missing:?}"
    )]
    EmissionNotConfirmed {
        /// The due day whose firings were not all recorded.
        day: NaiveDate,
        /// `(scope, name, signal)` firings missing from the ledger.
        missing: Vec<(String, String, String)>,
    },
}

/// Emission state held by `GatewayServer` (design Component 3):
/// the config snapshot handle, the shared logger / delivery-ledger /
/// notifier Arcs, and the in-memory `last_evaluated_day`
/// short-circuit. The hourly loop is spawned only in
/// `start_with_shutdown`; construction never starts timers.
pub struct PerformanceAlertState {
    config: Arc<RwLock<Config>>,
    logger: Arc<RequestLogger>,
    ledger: Arc<WebhookLedgerStore>,
    notifier: Arc<WebhookNotifier>,
    /// Last UTC day this process evaluated end-to-end. Guarded by a
    /// std mutex: only a trivial copy happens under the lock — never
    /// held across an await.
    last_evaluated_day: Mutex<Option<NaiveDate>>,
    /// In-flight guard (fix E): `swap(true)` admits one cycle at a
    /// time; an overlapping tick returns `Ok` early with a debug log.
    cycle_in_flight: AtomicBool,
}

impl std::fmt::Debug for PerformanceAlertState {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("PerformanceAlertState").finish_non_exhaustive()
    }
}

impl PerformanceAlertState {
    /// Build the emission state on the same config, logger, ledger,
    /// and notifier Arcs the rest of `AppState` holds.
    pub fn new(
        config: Arc<RwLock<Config>>,
        logger: Arc<RequestLogger>,
        ledger: Arc<WebhookLedgerStore>,
        notifier: Arc<WebhookNotifier>,
    ) -> Self {
        Self {
            config,
            logger,
            ledger,
            notifier,
            last_evaluated_day: Mutex::new(None),
            cycle_in_flight: AtomicBool::new(false),
        }
    }

    /// One evaluation cycle against the wall clock (the background
    /// loop's call). See [`Self::run_cycle_at`] for the cycle
    /// contract; the clock is injected there so tests pin the due
    /// day instead of racing the 01:00 grace.
    pub async fn run_cycle(&self) -> Result<(), PerformanceAlertError> {
        self.run_cycle_at(Utc::now()).await
    }

    /// One evaluation cycle at `now` (design Component 3):
    ///
    /// - dormancy: `performance_alerts.enabled == false` or zero
    ///   webhook endpoints returns before any query or write;
    /// - due day: the last complete UTC day `D`, due once the clock
    ///   passed 01:00 UTC (grace); an already-evaluated `D`
    ///   short-circuits without queries;
    /// - evaluation: three grouped 8-day aggregates (`[D-7 .. D]`)
    ///   via `spawn_blocking`, grouped per scope into 7 baseline
    ///   days plus the observed day, then the pure core;
    /// - dedup: findings already recorded in the delivery ledger
    ///   for `D` (payload `evaluated_day` + scope + signal) are
    ///   skipped (design D5);
    /// - emission: the rest go through [`WebhookNotifier::emit_value`]
    ///   as `performance.anomaly` envelopes; skip reasons log at
    ///   `debug`;
    /// - confirmation (fix M1): after emitting, the ledger is
    ///   re-read and every intended firing must have a row before
    ///   the day advances; the fire-and-forget emit seam gets a
    ///   short bounded poll (~2s) to land its rows.
    ///
    /// `last_evaluated_day` advances only after a fully successful
    /// cycle; any ledger/stats/confirmation failure returns an error
    /// first, so the next tick retries (design, Error Handling 1).
    /// Overlapping cycles are serialized by an in-flight guard
    /// (fix E): a tick arriving while a cycle still runs returns
    /// `Ok` early.
    pub async fn run_cycle_at(&self, now: DateTime<Utc>) -> Result<(), PerformanceAlertError> {
        if self
            .cycle_in_flight
            .swap(true, Ordering::SeqCst)
        {
            tracing::debug!("performance alerts: cycle already in progress; overlapping tick skipped");
            return Ok(());
        }
        let guard = InFlightGuard(&self.cycle_in_flight);
        let result = self.run_cycle_inner(now).await;
        drop(guard);
        result
    }

    async fn run_cycle_inner(&self, now: DateTime<Utc>) -> Result<(), PerformanceAlertError> {
        // Config snapshot under the read lock; the guard drops at
        // the end of this block, before any ledger I/O (the
        // delivery.rs config-read discipline — no lock across await).
        let (enabled, dormant) = {
            let config = self.config.read().await;
            (
                config.performance_alerts.enabled,
                config.webhooks.endpoints.is_empty(),
            )
        };
        if !enabled || dormant {
            tracing::debug!(
                "performance alerts dormant (disabled or no webhook endpoints); cycle skipped"
            );
            return Ok(());
        }

        if now.time() < evaluation_grace() {
            tracing::debug!(
                "performance alerts: last complete UTC day not past the 01:00 grace; cycle skipped"
            );
            return Ok(());
        }
        let due_day = now
            .date_naive()
            .pred_opt()
            .expect("a date always has a predecessor");
        if *self.last_evaluated_day.lock().expect("last_evaluated_day mutex poisoned") == Some(due_day)
        {
            return Ok(());
        }

        let window_start = due_day
            .checked_sub_days(Days::new(7))
            .expect("week before the due day always constructs");

        // Blocking-pool aggregates over the requests ledger: one
        // read-only WAL reader connection serves all three
        // dimensions (the cost_reporting::query call discipline).
        let db_path = self.logger.database_path().to_string();
        let rows = tokio::task::spawn_blocking(
            move || -> Result<Vec<(ScopeKey, DailyStats)>, QueryError> {
                let conn = Connection::open_with_flags(
                    &db_path,
                    OpenFlags::SQLITE_OPEN_READ_ONLY | OpenFlags::SQLITE_OPEN_NO_MUTEX,
                )?;
                conn.busy_timeout(Duration::from_secs(5))?;
                let mut merged = Vec::new();
                for dimension in [
                    Dimension::Gateway,
                    Dimension::Provider,
                    Dimension::ModelGroup,
                ] {
                    merged.extend(fetch_daily_stats(
                        &conn,
                        dimension,
                        window_start,
                        due_day,
                    )?);
                }
                Ok(merged)
            },
        )
        .await
        .map_err(PerformanceAlertError::Task)??;

        // Group the daily rows per scope into 7 baseline days plus
        // the observed day; absent days read as zero traffic (the
        // evaluation core's documented empty-day semantics).
        let mut grouped: HashMap<ScopeKey, HashMap<NaiveDate, DailyStats>> = HashMap::new();
        let window_is_empty = rows.is_empty();
        for (key, stats) in rows {
            grouped.entry(key).or_default().insert(stats.day, stats);
        }
        // Empty-window guard (fix m7): a fresh install (or an idle
        // ledger) returns zero rows for EVERY dimension — evaluate
        // would then have no scope to run at all. Materialize one
        // all-zero gateway scope so the gates produce their skip
        // reasons (debug-logged); no finding can fire from zeros.
        // Only when the ENTIRE merged row set is empty — never
        // invent scopes when real rows exist.
        if window_is_empty {
            tracing::debug!(
                "performance alerts: empty 8-day window; materializing the zero gateway scope"
            );
            grouped.insert(
                ScopeKey {
                    scope: Scope::Gateway,
                    name: "gateway".to_string(),
                },
                HashMap::new(),
            );
        }
        let mut scopes: Vec<(ScopeKey, [DailyStats; 7], DailyStats)> = grouped
            .into_iter()
            .map(|(key, days)| {
                let mut baseline = [zero_stats(due_day); 7];
                for (index, slot) in baseline.iter_mut().enumerate() {
                    let day = due_day
                        .checked_sub_days(Days::new(
                            u64::try_from(7 - index).expect("baseline index below 7"),
                        ))
                        .expect("baseline day always constructs");
                    *slot = days.get(&day).copied().unwrap_or_else(|| zero_stats(day));
                }
                let observed = days
                    .get(&due_day)
                    .copied()
                    .unwrap_or_else(|| zero_stats(due_day));
                (key, baseline, observed)
            })
            .collect();
        scopes.sort_by(|a, b| {
            a.0.scope
                .as_str()
                .cmp(b.0.scope.as_str())
                .then_with(|| a.0.name.cmp(&b.0.name))
        });

        // Dedup ledger read (design D5): Result-returning, unlike
        // the degrade-to-empty dashboard reads — a failed lookup
        // must fail the cycle instead of re-firing alerts.
        let since_epoch = due_day
            .and_hms_opt(0, 0, 0)
            .expect("UTC midnight always constructs")
            .and_utc()
            .timestamp();
        let ledger = Arc::clone(&self.ledger);
        let existing = tokio::task::spawn_blocking(move || {
            ledger.events_of_type_since(EVENT_TYPE, since_epoch)
        })
        .await
        .map_err(PerformanceAlertError::Task)??;
        let recorded = recorded_firings(&existing, due_day);

        let mut intended: Vec<(String, String, String)> = Vec::new();
        for (key, baseline, observed) in &scopes {
            let (findings, skips) = evaluate(&key.scope, due_day, baseline, observed);
            for skip in &skips {
                tracing::debug!(
                    scope = %key.scope.as_str(),
                    name = %key.name,
                    skip = ?skip,
                    "performance alerts: signal check skipped"
                );
            }
            for finding in findings {
                let firing = (
                    key.scope.as_str().to_string(),
                    key.name.clone(),
                    finding.signal.as_str().to_string(),
                );
                if recorded.contains(&firing) {
                    tracing::debug!(
                        scope = %firing.0,
                        name = %firing.1,
                        signal = %firing.2,
                        "performance alerts: finding already recorded for the day; dedup suppressed"
                    );
                    continue;
                }
                intended.push(firing);
                let data = PerformanceAnomalyData {
                    signal: finding.signal,
                    scope: key.scope.as_str().to_string(),
                    name: key.name.clone(),
                    observed_value: finding.observed_value,
                    baseline_value: finding.baseline_value,
                    unit: finding.unit,
                    evaluated_day: due_day,
                    threshold: finding.threshold.to_string(),
                };
                let envelope = serde_json::json!({
                    "id": uuid::Uuid::new_v4().to_string(),
                    "type": serde_json::to_value(BudgetEventType::PerformanceAnomaly)
                        .expect("event type always serializes"),
                    "created_at": Utc::now().to_rfc3339_opts(SecondsFormat::Secs, true),
                    "signed": false,
                    "data": serde_json::to_value(&data).expect("payload always serializes"),
                });
                self.notifier.emit_value(EVENT_TYPE, envelope).await;
                tracing::info!(
                    scope = %key.scope.as_str(),
                    name = %key.name,
                    signal = %finding.signal.as_str(),
                    observed = finding.observed_value,
                    baseline = finding.baseline_value,
                    "performance anomaly alert emitted"
                );
            }
        }

        // Persistence-before-advance (fix M1): the emit seam is
        // fire-and-forget (`emit_value` spawns `fan_out`, which
        // records the ledger row off-path), so poll the ledger for a
        // short bounded window and require EVERY intended firing to
        // have a row (scope+signal+evaluated_day via the same
        // recorded_firings decode) before the day advances. Missing
        // rows fail the cycle — the hourly retry re-runs it while the
        // per-finding dedup suppresses whatever did land.
        if !intended.is_empty() {
            let mut confirmed = recorded.clone();
            let mut missing: Vec<(String, String, String)> =
                intended.iter().filter(|f| !confirmed.contains(*f)).cloned().collect();
            for _ in 0..PERSISTENCE_POLLS {
                if missing.is_empty() {
                    break;
                }
                tokio::time::sleep(PERSISTENCE_POLL_INTERVAL).await;
                let ledger = Arc::clone(&self.ledger);
                let recheck = tokio::task::spawn_blocking(move || {
                    ledger.events_of_type_since(EVENT_TYPE, since_epoch)
                })
                .await
                .map_err(PerformanceAlertError::Task)??;
                confirmed = recorded_firings(&recheck, due_day);
                missing = intended
                    .iter()
                    .filter(|f| !confirmed.contains(*f))
                    .cloned()
                    .collect();
            }
            if !missing.is_empty() {
                tracing::warn!(
                    day = %due_day,
                    missing = ?missing,
                    "performance alerts: emission not confirmed in the ledger; day not advanced"
                );
                return Err(PerformanceAlertError::EmissionNotConfirmed {
                    day: due_day,
                    missing,
                });
            }
        }

        *self.last_evaluated_day.lock().expect("last_evaluated_day mutex poisoned") =
            Some(due_day);
        Ok(())
    }

    /// Restart-safety diagnostic (fix M2): logs the newest recorded
    /// `performance.anomaly` row's `evaluated_day` for observability
    /// ONLY — the in-memory `last_evaluated_day` is deliberately
    /// NEVER seeded from the ledger: adopting the newest row's day
    /// would mark a partially recorded day complete and suppress its
    /// remaining findings after a restart. Restart safety comes from
    /// the per-finding ledger dedup inside the cycle; re-evaluation
    /// after restart is bounded (~3 dimension queries). Dormant
    /// configs skip the read entirely; a failed read is a warn, the
    /// cycle itself is unaffected.
    async fn log_last_recorded_day(&self) -> Result<(), PerformanceAlertError> {
        {
            let config = self.config.read().await;
            if !config.performance_alerts.enabled || config.webhooks.endpoints.is_empty() {
                return Ok(());
            }
        }
        let ledger = Arc::clone(&self.ledger);
        let newest = tokio::task::spawn_blocking(move || ledger.newest_event_of_type(EVENT_TYPE))
            .await
            .map_err(PerformanceAlertError::Task)??;
        if let Some(day) = newest.as_ref().and_then(evaluated_day_of) {
            tracing::debug!(
                day = %day,
                "performance alerts: newest recorded performance.anomaly day (restart re-evaluates via per-finding dedup)"
            );
        }
        Ok(())
    }

/// Spawn the background cadence loop (design D7): the first tick
/// fires immediately (startup catch-up; restart safety is the
/// per-finding ledger dedup inside the cycle, not a seeded day —
/// fix M2), subsequent ticks hourly; the loop exits when the
/// shutdown watch fires or its sender drops (the deprecation
/// scanner's `watch` pattern). Spawned ONLY in
/// `GatewayServer::start_with_shutdown` — never at construction,
/// so `GatewayServer::new` stays test-safe (no timers firing in
/// oneshot tests).
pub fn spawn_background(
        self: Arc<Self>,
        mut shutdown: watch::Receiver<bool>,
    ) -> tokio::task::JoinHandle<()> {
        tokio::spawn(async move {
            if let Err(error) = self.log_last_recorded_day().await {
                tracing::warn!(
                    error = %error,
                    "performance alerts: restart diagnostic read failed; per-finding dedup remains active"
                );
            }
            let mut interval = tokio::time::interval(CYCLE_CADENCE);
            interval.set_missed_tick_behavior(tokio::time::MissedTickBehavior::Delay);
            loop {
                tokio::select! {
                    changed = shutdown.changed() => {
                        if changed.is_err() || *shutdown.borrow() {
                            break;
                        }
                    }
                    _ = interval.tick() => {
                        if let Err(error) = self.run_cycle().await {
                            tracing::warn!(
                                error = %error,
                                "performance alerts cycle failed; retrying on the next tick"
                            );
                        }
                    }
                }
            }
            tracing::debug!("performance alerts background loop stopped");
        })
    }
}

/// RAII in-flight guard (fix E): releases the cycle slot even when
/// the cycle returns `Err` or panics mid-await (the flag is the
/// AtomicBool on [`PerformanceAlertState`]).
struct InFlightGuard<'a>(&'a AtomicBool);

impl Drop for InFlightGuard<'_> {
    fn drop(&mut self) {
        self.0.store(false, Ordering::SeqCst);
    }
}

/// Evaluation grace (design D7): the last complete UTC day counts as
/// due once the clock has passed 01:00 UTC, so a slow writer
/// flushing the day's tail cannot lose findings.
fn evaluation_grace() -> NaiveTime {
    NaiveTime::from_hms_opt(1, 0, 0).expect("01:00:00 always constructs")
}

/// Zero-traffic stand-in for a day absent from the aggregates (the
/// evaluation core treats it as no traffic that day).
fn zero_stats(day: NaiveDate) -> DailyStats {
    DailyStats {
        day,
        cost_micro_usd: 0,
        requests: 0,
        errors: 0,
        duration_ms_sum: 0,
    }
}

/// `(scope, name, signal)` firings recorded in the delivery ledger
/// for `day` — the D5 dedup key set, keyed off the payload (the
/// envelope is the source of truth).
fn recorded_firings(
    rows: &[AlertEventRow],
    day: NaiveDate,
) -> HashSet<(String, String, String)> {
    rows.iter()
        .filter_map(|row| {
            let data = payload_data_of(row)?;
            if payload_evaluated_day_of(&data)? != day {
                return None;
            }
            Some((
                data.get("scope")?.as_str()?.to_string(),
                data.get("name")?.as_str()?.to_string(),
                data.get("signal")?.as_str()?.to_string(),
            ))
        })
        .collect()
}

/// `evaluated_day` of one ledger row's payload, when present and
/// well-formed (the restart diagnostic reads the newest such row).
fn evaluated_day_of(row: &AlertEventRow) -> Option<NaiveDate> {
    payload_evaluated_day_of(&payload_data_of(row)?)
}

/// Parsed `data` object of a ledger row's serialized event payload
/// (fix n14: one shared decode for both payload-day readers).
fn payload_data_of(row: &AlertEventRow) -> Option<serde_json::Value> {
    let payload = serde_json::from_str::<serde_json::Value>(&row.payload_json).ok()?;
    Some(payload.get("data")?.clone())
}

/// `evaluated_day` field of a decoded payload `data` value (fix
/// n14: one shared decode for both payload-day readers).
fn payload_evaluated_day_of(data: &serde_json::Value) -> Option<NaiveDate> {
    NaiveDate::parse_from_str(data.get("evaluated_day")?.as_str()?, "%Y-%m-%d").ok()
}
