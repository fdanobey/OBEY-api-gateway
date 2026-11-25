//! Emission orchestrator and background loop for usage-anomaly
//! alerts (spec: usage-anomaly-alerts, design Component 4).
//!
//! Bridges the per-key ledger aggregates
//! ([`crate::usage_anomaly_alerts::query`]) and the pure
//! evaluation core ([`crate::usage_anomaly_alerts::state`]) to the
//! generalized webhook delivery seam ([`WebhookNotifier::emit_value`]):
//! once the last complete UTC day is past its 01:00 grace, the
//! cycle fetches per-key daily / hour-histogram / model
//! aggregates (exactly three queries per evaluated day, Req
//! 4.7) on the blocking pool, evaluates each key's day against
//! its own
//! 7-day baseline, suppresses findings the delivery ledger already
//! recorded for that day (per-finding dedup keyed on
//! `(key_id, signal, day)`), and emits the rest as
//! `usage.anomaly` envelopes with [`UsageAnomalyData`] payloads
//! (the display name is resolved cosmetically; the key id — never
//! the `vk_` secret — is the identity).
//!
//! Multi-day catch-up (design D8): every un-evaluated complete day
//! still inside the 8-day window `[newest_due - 7, newest_due]` is
//! evaluated oldest-first, each against the 7 days preceding it, so
//! gateway downtime never silently skips a day.
//!
//! Dormancy contract (mirrors the performance emission): when
//! `usage_anomaly_alerts.enabled` is false or
//! `config.webhooks.endpoints` is empty the cycle returns before
//! any query or write, and `last_evaluated_day` is left untouched
//! so a later re-enable resumes.
//!
//! Failure discipline: any ledger or stats failure aborts the
//! cycle with an error WITHOUT advancing `last_evaluated_day`, so
//! the hourly tick retries; a partially emitted day is finished by
//! the per-finding ledger dedup, never re-fired.
//! Persistence-before-advance (fix M1 discipline): the emit seam
//! is fire-and-forget, so the cycle re-reads the ledger with a
//! short bounded poll and fails itself when an intended firing has
//! no recorded row. Restart safety is the same per-finding dedup —
//! no day is ever seeded from the ledger (fix M2 discipline);
//! the newest recorded row is logged diagnostically ONLY.
//!
//! Loop discipline (the `model_deprecations` scanner shape): the
//! state is built once in `GatewayServer::new`, and the hourly
//! loop — first tick immediate at startup (restart catch-up via
//! the per-finding dedup) — is spawned ONLY in
//! `start_with_shutdown`, never at construction, so oneshot tests
//! start no timers.

use std::collections::{HashMap, HashSet};
use std::sync::{Arc, Mutex};
use std::sync::atomic::{AtomicBool, Ordering};
use std::time::Duration;

use chrono::{DateTime, Days, NaiveDate, NaiveTime, SecondsFormat, Utc};
use rusqlite::{Connection, OpenFlags};
use tokio::sync::{RwLock, watch};

use crate::config::{BudgetEventType, Config};
use crate::logger::RequestLogger;
use crate::usage_anomaly_alerts::query::{
    fetch_key_daily_stats, fetch_key_hour_histogram, fetch_key_model_usage, QueryError,
};
use crate::usage_anomaly_alerts::state::{
    evaluate, KeyDailyStats, KeyEvaluation, KeyHourHistogram, KeyModelUsage, KeyRef, Unit,
};
use crate::webhooks::delivery::WebhookNotifier;
use crate::webhooks::events::UsageAnomalyData;
use crate::webhooks::ledger::{AlertEventRow, WebhookLedgerError, WebhookLedgerStore};

/// Wire event type (ledger row + subscription filter key): the
/// serde name of [`BudgetEventType::UsageAnomaly`].
const EVENT_TYPE: &str = "usage.anomaly";

/// Background cadence between cycle attempts (hourly tick; the
/// `last_evaluated_day` short-circuit makes re-checks of an
/// already-evaluated day query-free).
const CYCLE_CADENCE: Duration = Duration::from_secs(3_600);

/// Polls of the delivery ledger while confirming persistence: ~10
/// × 200ms = ~2s total bounded wait for the fire-and-forget emit
/// seam to land its rows.
const PERSISTENCE_POLLS: usize = 10;
/// Sleep between persistence confirmation polls.
const PERSISTENCE_POLL_INTERVAL: Duration = Duration::from_millis(200);

/// Resolves one virtual-key id to its display name (cosmetic only
/// — never the `vk_` secret value). Backed by a blocking keys.db
/// read, so the cycle calls it only inside `spawn_blocking`; a
/// miss (unknown or deleted key) degrades to `None`, never fails
/// the cycle.
pub type KeyNameResolver = Arc<dyn Fn(&str) -> Option<String> + Send + Sync>;

/// Cycle failure: ledger or stats query errors (and a panicked
/// blocking task) surface as errors so the caller retries on the
/// next tick — `last_evaluated_day` only advances on a fully
/// successful cycle.
#[derive(Debug, thiserror::Error)]
pub enum UsageAnomalyAlertError {
    #[error("usage anomaly alerts ledger query failed: {0}")]
    Ledger(#[from] WebhookLedgerError),
    #[error("usage anomaly alerts stats query failed: {0}")]
    Stats(#[from] QueryError),
    #[error("usage anomaly alerts blocking task failed: {0}")]
    Task(#[from] tokio::task::JoinError),
    /// A finding the cycle intended to fire never landed in the
    /// delivery ledger (persistence-before-advance).
    #[error(
        "usage anomaly alerts emission not confirmed in the ledger for {day}: missing {missing:?}"
    )]
    EmissionNotConfirmed {
        /// The due day whose firings were not all recorded.
        day: NaiveDate,
        /// `(key_id, signal, day)` firings missing from the ledger.
        missing: Vec<(String, String, String)>,
    },
}

/// Per-key aggregates one evaluation day needs, fetched in a
/// single `spawn_blocking` pass over one read-only ledger
/// connection: exactly THREE aggregate queries (Req 4.7), each
/// day-resolved over `[day - 7, day]`.
struct DayAggregates {
    /// Per-(key, day) daily rows over `[day - 7, day]`.
    daily: Vec<(String, KeyDailyStats)>,
    /// Day-resolved hour histograms over `[day - 7, day]`.
    hour_histograms: Vec<(String, NaiveDate, KeyHourHistogram)>,
    /// Day-resolved per-(key, model) counts over `[day - 7, day]`.
    model_usage: Vec<(String, NaiveDate, KeyModelUsage)>,
    /// Cosmetic display names for the keys seen in `daily`.
    names: HashMap<String, String>,
}

/// Debug-observability counters for the per-cycle summary (Req
/// 6.2): accumulated across every day the cycle processed.
#[derive(Default)]
struct CycleSummary {
    keys: usize,
    findings: usize,
    skips: usize,
}

/// Emission state held by `GatewayServer` (design Component 4):
/// the config snapshot handle, the shared logger / key-name
/// resolver / delivery-ledger / notifier Arcs, and the in-memory
/// `last_evaluated_day` short-circuit. The hourly loop is spawned
/// only in `start_with_shutdown`; construction never starts
/// timers.
pub struct UsageAnomalyAlertState {
    config: Arc<RwLock<Config>>,
    logger: Arc<RequestLogger>,
    key_names: KeyNameResolver,
    ledger: Arc<WebhookLedgerStore>,
    notifier: Arc<WebhookNotifier>,
    /// Last UTC day this process evaluated end-to-end. Guarded by
    /// a std mutex: only a trivial copy happens under the lock —
    /// never held across an await.
    last_evaluated_day: Mutex<Option<NaiveDate>>,
    /// In-flight guard: `swap(true)` admits one cycle at a time;
    /// an overlapping tick returns `Ok` early with a debug log.
    cycle_in_flight: AtomicBool,
}

impl std::fmt::Debug for UsageAnomalyAlertState {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("UsageAnomalyAlertState")
            .finish_non_exhaustive()
    }
}

impl UsageAnomalyAlertState {
    /// Build the emission state on the same config, logger,
    /// ledger, and notifier Arcs the rest of `AppState` holds,
    /// plus the virtual-key name resolver.
    pub fn new(
        config: Arc<RwLock<Config>>,
        logger: Arc<RequestLogger>,
        key_names: KeyNameResolver,
        ledger: Arc<WebhookLedgerStore>,
        notifier: Arc<WebhookNotifier>,
    ) -> Self {
        Self {
            config,
            logger,
            key_names,
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
    pub async fn run_cycle(&self) -> Result<(), UsageAnomalyAlertError> {
        self.run_cycle_at(Utc::now()).await
    }

    /// One evaluation cycle at `now`:
    ///
    /// - dormancy: `usage_anomaly_alerts.enabled == false` or
    ///   zero webhook endpoints returns before any query or
    ///   write;
    /// - due days: the last complete UTC day is due once the
    ///   clock passed 01:00 UTC (grace); every un-evaluated
    ///   complete day still inside the 8-day window
    ///   `[newest_due - 7, newest_due]` is evaluated
    ///   oldest-first (multi-day catch-up), each against the 7
    ///   days preceding it; an already-evaluated newest day
    ///   short-circuits without queries;
    /// - evaluation: per-key daily rows plus day-resolved hour
    /// histograms and model counts via one `spawn_blocking`
    /// pass of exactly THREE aggregate queries per day (Req
    /// 4.7), split in memory into 7 baseline days plus the
    /// observed day, then the pure core;
    /// - dedup: findings already recorded in the delivery ledger
    ///   for the day (payload `key_id` + `signal` + day) are
    ///   skipped;
    /// - emission: the rest go through
    ///   [`WebhookNotifier::emit_value`] as `usage.anomaly`
    ///   envelopes; skip reasons and a per-cycle summary (keys
    ///   evaluated, findings emitted, skips) log at `debug`;
    /// - confirmation: after emitting, the ledger is re-read and
    ///   every intended firing must have a row before the day
    ///   advances; the fire-and-forget emit seam gets a short
    ///   bounded poll (~2s) to land its rows.
    ///
    /// `last_evaluated_day` advances only after a fully
    /// successful day; any ledger/stats/confirmation failure
    /// returns an error first, so the next tick retries (the
    /// per-finding dedup suppresses whatever did land).
    /// Overlapping cycles are serialized by an in-flight guard: a
    /// tick arriving while a cycle still runs returns `Ok` early.
    pub async fn run_cycle_at(&self, now: DateTime<Utc>) -> Result<(), UsageAnomalyAlertError> {
        if self
            .cycle_in_flight
            .swap(true, Ordering::SeqCst)
        {
            tracing::debug!("usage anomaly alerts: cycle already in progress; overlapping tick skipped");
            return Ok(());
        }
        let guard = InFlightGuard(&self.cycle_in_flight);
        let result = self.run_cycle_inner(now).await;
        drop(guard);
        result
    }

    async fn run_cycle_inner(&self, now: DateTime<Utc>) -> Result<(), UsageAnomalyAlertError> {
        // Config snapshot under the read lock; the guard drops at
        // the end of this block, before any ledger I/O (the
        // delivery.rs config-read discipline — no lock across
        // await).
        let (enabled, dormant) = {
            let config = self.config.read().await;
            (
                config.usage_anomaly_alerts.enabled,
                config.webhooks.endpoints.is_empty(),
            )
        };
        if !enabled || dormant {
            tracing::debug!(
                "usage anomaly alerts dormant (disabled or no webhook endpoints); cycle skipped"
            );
            return Ok(());
        }

        if now.time() < evaluation_grace() {
            tracing::debug!(
                "usage anomaly alerts: last complete UTC day not past the 01:00 grace; cycle skipped"
            );
            return Ok(());
        }
        let newest_due = now
            .date_naive()
            .pred_opt()
            .expect("a date always has a predecessor");
        // Multi-day catch-up bound (design D8): only the 8-day
        // window ending at the newest due day — older days fall
        // outside the ledger's dedup retention and are never
        // re-fired.
        let oldest_catch_up = newest_due
            .checked_sub_days(Days::new(7))
            .expect("week before the newest due day always constructs");
        let first = match *self
            .last_evaluated_day
            .lock()
            .expect("last_evaluated_day mutex poisoned")
        {
            Some(last_day) => last_day
                .succ_opt()
                .expect("a date always has a successor")
                .max(oldest_catch_up),
            None => oldest_catch_up,
        };
        if first > newest_due {
            return Ok(());
        }

        let db_path = self.logger.database_path().to_string();
        let mut summary = CycleSummary::default();
        let mut days_processed = 0usize;
        let mut day = first;
        while day <= newest_due {
            self.evaluate_one_day(day, &db_path, &mut summary).await?;
            days_processed += 1;
            day = day
                .succ_opt()
                .expect("a date always has a successor");
        }
        tracing::debug!(
            days = days_processed,
            keys = summary.keys,
            findings = summary.findings,
            skips = summary.skips,
            "usage anomaly alerts: cycle summary"
        );
        Ok(())
    }

    /// Evaluate one due day end-to-end (queries → pure core →
    /// dedup → emit → confirm) and advance `last_evaluated_day`
    /// only on full success. Abort leaves the day un-advanced so
    /// the hourly retry re-runs it while the per-finding dedup
    /// suppresses already-recorded findings.
    async fn evaluate_one_day(
        &self,
        day: NaiveDate,
        db_path: &str,
        summary: &mut CycleSummary,
    ) -> Result<(), UsageAnomalyAlertError> {
        // Blocking-pool aggregates over the requests ledger: one
        // read-only WAL reader connection serves exactly THREE
        // aggregate queries per evaluated day (Req 4.7, the
        // cost_reporting::query call discipline): daily stats,
        // hour histograms, and model usage each cover the whole
        // 8-day window `[day - 7, day]` day-resolved, and the
        // observed (== day) vs baseline (< day) split happens in
        // memory below.
        let db_path = db_path.to_string();
        let key_names = Arc::clone(&self.key_names);
        let aggregates = tokio::task::spawn_blocking(
            move || -> Result<DayAggregates, QueryError> {
                let conn = Connection::open_with_flags(
                    &db_path,
                    OpenFlags::SQLITE_OPEN_READ_ONLY | OpenFlags::SQLITE_OPEN_NO_MUTEX,
                )?;
                conn.busy_timeout(Duration::from_secs(5))?;
                let window_start = day
                    .checked_sub_days(Days::new(7))
                    .expect("week before the day always constructs");
                let daily = fetch_key_daily_stats(&conn, window_start, day)?;
                let hour_histograms = fetch_key_hour_histogram(&conn, window_start, day)?;
                let model_usage = fetch_key_model_usage(&conn, window_start, day)?;
                // Batch display-name resolution: collect the
                // distinct key ids first, then ONE resolver pass
                // inside this single blocking closure — one
                // blocking pass total, never one keys.db query
                // per key (N+1 fix).
                let mut distinct_ids: Vec<&str> = Vec::new();
                for (key_id, _) in &daily {
                    if !distinct_ids.contains(&key_id.as_str()) {
                        distinct_ids.push(key_id);
                    }
                }
                let mut names = HashMap::new();
                for key_id in distinct_ids {
                    if let Some(name) = key_names(key_id) {
                        names.insert(key_id.to_string(), name);
                    }
                }
                Ok(DayAggregates {
                    daily,
                    hour_histograms,
                    model_usage,
                    names,
                })
            },
        )
        .await
        .map_err(UsageAnomalyAlertError::Task)??;

        // Group the daily rows per key into 7 baseline days plus
        // the observed day; absent days read as zero traffic (the
        // evaluation core's documented empty-day semantics). A key
        // with no row anywhere in the 8-day window has nothing to
        // evaluate (no traffic can fire no finding).
        let mut grouped: HashMap<String, HashMap<NaiveDate, KeyDailyStats>> = HashMap::new();
        for (key_id, stats) in aggregates.daily {
            grouped
                .entry(key_id)
                .or_default()
                .insert(stats.day, stats);
        }
        let mut key_ids: Vec<String> = grouped.keys().cloned().collect();
        key_ids.sort();

        // Split the day-resolved aggregates in memory: observed
        // is `== day`, baseline sums the 7 days before it. A key
        // absent from the baseline side has NO behavioral rows
        // for the window — the `Option` stays `None` (behavioral
        // data absent) instead of being coerced to an empty set,
        // so new_model/odd_hours record the data-absent skip
        // rather than firing on an empty baseline.
        let mut observed_hist: HashMap<String, KeyHourHistogram> = HashMap::new();
        let mut baseline_hist: HashMap<String, KeyHourHistogram> = HashMap::new();
        for (key_id, row_day, hist) in aggregates.hour_histograms {
            if row_day == day {
                observed_hist.insert(key_id, hist);
            } else {
                let entry = baseline_hist.entry(key_id).or_insert(KeyHourHistogram {
                    buckets: [0; 6],
                });
                for (bucket, count) in entry.buckets.iter_mut().zip(hist.buckets.iter()) {
                    *bucket += count;
                }
            }
        }
        let mut observed_models: HashMap<String, Vec<KeyModelUsage>> = HashMap::new();
        let mut baseline_models: HashMap<String, Vec<KeyModelUsage>> = HashMap::new();
        for (key_id, row_day, usage) in aggregates.model_usage {
            if row_day == day {
                observed_models.entry(key_id).or_default().push(usage);
            } else {
                baseline_models.entry(key_id).or_default().push(usage);
            }
        }

        // Dedup ledger read: Result-returning, unlike the
        // degrade-to-empty dashboard reads — a failed lookup must
        // fail the cycle instead of re-firing alerts. Identity is
        // `(key_id, signal, day)`; the display name is never part
        // of the tuple.
        let ledger = Arc::clone(&self.ledger);
        let day_str = day.to_string();
        let recorded: HashSet<(String, String, String)> =
            tokio::task::spawn_blocking(move || ledger.fetch_recorded_firings(EVENT_TYPE, &day_str))
                .await
                .map_err(UsageAnomalyAlertError::Task)??;

        let mut intended: Vec<(String, String, String)> = Vec::new();
        for key_id in &key_ids {
            let days = &grouped[key_id];
            let key = KeyRef {
                key_id: key_id.clone(),
                name: aggregates.names.get(key_id).cloned(),
            };
            let mut baseline = [zero_stats(day); 7];
            for (index, slot) in baseline.iter_mut().enumerate() {
                let baseline_day = day
                    .checked_sub_days(Days::new(
                        u64::try_from(7 - index).expect("baseline index below 7"),
                    ))
                    .expect("baseline day always constructs");
                *slot = days
                    .get(&baseline_day)
                    .copied()
                    .unwrap_or_else(|| zero_stats(baseline_day));
            }
            let observed = days
                .get(&day)
                .copied()
                .unwrap_or_else(|| zero_stats(day));
            let evaluation = KeyEvaluation {
                key,
                observed,
                baseline: baseline.to_vec(),
                hour_histogram: observed_hist.get(key_id).copied(),
                baseline_hour_histogram: baseline_hist.get(key_id).copied(),
                observed_models: observed_models.get(key_id).cloned().unwrap_or_default(),
                baseline_models: baseline_models
                    .get(key_id)
                    .map(|usages| usages.iter().map(|usage| usage.model.clone()).collect()),
            };
            let (findings, skips) = evaluate(&evaluation, day);
            summary.keys += 1;
            for skip in &skips {
                summary.skips += 1;
                tracing::debug!(
                    key_id = %key_id,
                    skip = ?skip,
                    "usage anomaly alerts: signal check skipped"
                );
            }
            for finding in findings {
                let firing = (
                    finding.key.key_id.clone(),
                    finding.signal.as_str().to_string(),
                    day.to_string(),
                );
                if recorded.contains(&firing) {
                    tracing::debug!(
                        key_id = %firing.0,
                        signal = %firing.1,
                        day = %firing.2,
                        "usage anomaly alerts: finding already recorded for the day; dedup suppressed"
                    );
                    continue;
                }
                intended.push(firing);
                let data = UsageAnomalyData::new(
                    finding.signal.as_str().to_string(),
                    finding.key.key_id.clone(),
                    finding.key.name.clone(),
                    finding.observed_value,
                    finding.baseline_value,
                    unit_as_str(finding.unit).to_string(),
                    day.to_string(),
                    finding.detail.clone(),
                    finding.threshold.to_string(),
                );
                let envelope = serde_json::json!({
                    "id": uuid::Uuid::new_v4().to_string(),
                    "type": serde_json::to_value(BudgetEventType::UsageAnomaly)
                        .expect("event type always serializes"),
                    "created_at": Utc::now().to_rfc3339_opts(SecondsFormat::Secs, true),
                    "signed": false,
                    "data": serde_json::to_value(&data).expect("payload always serializes"),
                });
                self.notifier.emit_value(EVENT_TYPE, envelope).await;
                tracing::info!(
                    key_id = %finding.key.key_id,
                    signal = %finding.signal.as_str(),
                    day = %day,
                    observed = finding.observed_value,
                    baseline = finding.baseline_value,
                    "usage anomaly alert emitted"
                );
                summary.findings += 1;
            }
        }

        // Persistence-before-advance: the emit seam is
        // fire-and-forget (`emit_value` spawns `fan_out`, which
        // records the ledger row off-path), so poll the ledger for
        // a short bounded window and require EVERY intended firing
        // to have a row before the day advances. Missing rows fail
        // the cycle — the hourly retry re-runs it while the
        // per-finding dedup suppresses whatever did land.
        if !intended.is_empty() {
            let mut confirmed = recorded.clone();
            let mut missing: Vec<(String, String, String)> = intended
                .iter()
                .filter(|f| !confirmed.contains(*f))
                .cloned()
                .collect();
            // Immediate recheck before the first sleep: rows that
            // already landed (fast fan-out) need no 200ms wait;
            // each subsequent poll sleeps first.
            for poll in 0..PERSISTENCE_POLLS {
                if missing.is_empty() {
                    break;
                }
                if poll > 0 {
                    tokio::time::sleep(PERSISTENCE_POLL_INTERVAL).await;
                }
                let ledger = Arc::clone(&self.ledger);
                let day_str = day.to_string();
                let recheck = tokio::task::spawn_blocking(move || {
                    ledger.fetch_recorded_firings(EVENT_TYPE, &day_str)
                })
                .await
                .map_err(UsageAnomalyAlertError::Task)??;
                confirmed = recheck;
                missing = intended
                    .iter()
                    .filter(|f| !confirmed.contains(*f))
                    .cloned()
                    .collect();
            }
            if !missing.is_empty() {
                tracing::warn!(
                    day = %day,
                    missing = ?missing,
                    "usage anomaly alerts: emission not confirmed in the ledger; day not advanced"
                );
                return Err(UsageAnomalyAlertError::EmissionNotConfirmed { day, missing });
            }
        }

        *self
            .last_evaluated_day
            .lock()
            .expect("last_evaluated_day mutex poisoned") = Some(day);
        Ok(())
    }

    /// Restart-safety diagnostic (fix M2 discipline): logs the
    /// newest recorded `usage.anomaly` row's `evaluated_day` for
    /// observability ONLY — the in-memory `last_evaluated_day` is
    /// deliberately NEVER seeded from the ledger: adopting the
    /// newest row's day would mark a partially recorded day
    /// complete and suppress its remaining findings after a
    /// restart. Restart safety comes from the per-finding ledger
    /// dedup inside the cycle. Dormant configs skip the read
    /// entirely; a failed read is a warn, the cycle itself is
    /// unaffected.
    async fn log_last_recorded_day(&self) -> Result<(), UsageAnomalyAlertError> {
        {
            let config = self.config.read().await;
            if !config.usage_anomaly_alerts.enabled || config.webhooks.endpoints.is_empty() {
                return Ok(());
            }
        }
        let ledger = Arc::clone(&self.ledger);
        let newest = tokio::task::spawn_blocking(move || ledger.newest_event_of_type(EVENT_TYPE))
            .await
            .map_err(UsageAnomalyAlertError::Task)??;
        if let Some(day) = newest.as_ref().and_then(evaluated_day_of) {
            tracing::debug!(
                day = %day,
                "usage anomaly alerts: newest recorded usage.anomaly day (restart re-evaluates via per-finding dedup)"
            );
        }
        Ok(())
    }

    /// Spawn the background cadence loop: the first tick fires
    /// immediately (startup catch-up; restart safety is the
    /// per-finding ledger dedup inside the cycle, not a seeded
    /// day), subsequent ticks hourly; the loop exits when the
    /// shutdown watch fires or its sender drops (the deprecation
    /// scanner's `watch` pattern). Spawned ONLY in
    /// `GatewayServer::start_with_shutdown` — never at
    /// construction, so `GatewayServer::new` stays test-safe (no
    /// timers firing in oneshot tests).
    pub fn spawn_background(
        self: Arc<Self>,
        mut shutdown: watch::Receiver<bool>,
    ) -> tokio::task::JoinHandle<()> {
        tokio::spawn(async move {
            if let Err(error) = self.log_last_recorded_day().await {
                tracing::warn!(
                    error = %error,
                    "usage anomaly alerts: restart diagnostic read failed; per-finding dedup remains active"
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
                                "usage anomaly alerts cycle failed; retrying on the next tick"
                            );
                        }
                    }
                }
            }
            tracing::debug!("usage anomaly alerts background loop stopped");
        })
    }
}

/// RAII in-flight guard: releases the cycle slot even when the
/// cycle returns `Err` or panics mid-await (the flag is the
/// AtomicBool on [`UsageAnomalyAlertState`]).
struct InFlightGuard<'a>(&'a AtomicBool);

impl Drop for InFlightGuard<'_> {
    fn drop(&mut self) {
        self.0.store(false, Ordering::SeqCst);
    }
}

/// Evaluation grace: the last complete UTC day counts as due once
/// the clock has passed 01:00 UTC, so a slow writer flushing the
/// day's tail cannot lose findings.
fn evaluation_grace() -> NaiveTime {
    NaiveTime::from_hms_opt(1, 0, 0).expect("01:00:00 always constructs")
}

/// Zero-traffic stand-in for a day absent from the aggregates (the
/// evaluation core treats it as no traffic that day).
fn zero_stats(day: NaiveDate) -> KeyDailyStats {
    KeyDailyStats {
        day,
        cost_micro_usd: 0,
        requests: 0,
        validation_failures: 0,
        input_tokens: 0,
        output_tokens: 0,
    }
}

/// Wire unit name of a [`Unit`] (`usd` | `requests` | `tokens` |
/// `ratio`), matching the `UsageAnomalyData` payload contract.
fn unit_as_str(unit: Unit) -> &'static str {
    match unit {
        Unit::Usd => "usd",
        Unit::Requests => "requests",
        Unit::Tokens => "tokens",
        Unit::Ratio => "ratio",
    }
}

/// `evaluated_day` of one ledger row's payload, when present and
/// well-formed (the restart diagnostic reads the newest such row).
fn evaluated_day_of(row: &AlertEventRow) -> Option<NaiveDate> {
    payload_evaluated_day_of(&payload_data_of(row)?)
}

/// Parsed `data` object of a ledger row's serialized event payload.
fn payload_data_of(row: &AlertEventRow) -> Option<serde_json::Value> {
    let payload = serde_json::from_str::<serde_json::Value>(&row.payload_json).ok()?;
    Some(payload.get("data")?.clone())
}

/// `evaluated_day` field of a decoded payload `data` value (a
/// `YYYY-MM-DD` string in the usage-anomaly payload).
fn payload_evaluated_day_of(data: &serde_json::Value) -> Option<NaiveDate> {
    NaiveDate::parse_from_str(data.get("evaluated_day")?.as_str()?, "%Y-%m-%d").ok()
}
