//! Provider-budget alert evaluation hook (budget-alerts-unified-webhooks,
//! Task 5).
//!
//! The single funnel the router calls after a durable provider-spend
//! accrual (both `persist_provider_spend` call sites; the evaluation
//! itself is spawned off the request/response path by the router —
//! Req 1.10). Owns the in-memory [`AlertState`] registry and a clone of
//! the built-once [`WebhookNotifier`]; evaluation receives the durable
//! spend + window anchor as values so this module never imports
//! `metrics` (no module cycle: the dependency arrow points
//! router → webhooks only).
//!
//! Dormancy contract (Req 3.3): when `config.webhooks.endpoints` is
//! empty every method here is a no-op — no evaluation, no ledger
//! writes, no reset events.
//!
//! _Requirements: 1.1, 1.3-1.5, 1.8-1.10, 2.1, 2.2, 3.3_

use std::sync::Arc;

use tokio::sync::RwLock;

use crate::config::{BudgetResetPolicy, Config};
use crate::cost_reporting::query::{fetch_trailing_spend, TrailingScope};
use crate::metrics::forecast::{
    forecast_manual, forecast_windowed, Forecast, MANUAL_HORIZON_DAYS,
};
use crate::metrics::next_reset_epoch;
use crate::webhooks::alerts::{AlertState, EvalInput, ScopeRef};
use crate::webhooks::delivery::WebhookNotifier;
use crate::webhooks::events::ResetSource;

/// Provider-budget alert hook: shared alert-state registry plus the
/// built-once notifier.
///
/// Constructed once at `GatewayServer::new`, stored on `AppState`, and
/// attached to the router via `set_provider_budget_alerts`; every
/// method is invoked only from spawned off-path tasks.
#[derive(Clone)]
pub struct ProviderBudgetAlerts {
    notifier: Arc<WebhookNotifier>,
    state: Arc<AlertState>,
    /// Per-scope FIFO turnstile: the forecast derivation's ledger read
    /// (manual scopes) makes one evaluation async-await long enough
    /// that a later accrual's evaluation could overtake an earlier
    /// one's threshold check and win the fired-state registry race
    /// with a later spend snapshot (the threshold event then reports
    /// the OVERTAKEN accrual's total instead of the crossing
    /// accrual's). One turnstile per scope key serializes
    /// read-then-evaluate sequences in durable-write order.
    turns: Arc<std::sync::Mutex<std::collections::HashMap<String, Arc<tokio::sync::Mutex<()>>>>>,
}

impl ProviderBudgetAlerts {
    /// Wrap the built-once notifier. The [`AlertState`] registry lives
    /// here (not on the notifier) so the VK hooks (Task 6) can share
    /// the same registry instance for their own scope keys via
    /// [`Self::alert_state`].
    pub fn new(notifier: Arc<WebhookNotifier>) -> Self {
        Self {
            notifier,
            state: Arc::new(AlertState::default()),
            turns: Arc::new(std::sync::Mutex::new(Default::default())),
        }
    }

    /// The per-scope turnstile for `key`: serializes the async
    /// forecast derivation + threshold evaluation sequences for one
    /// scope so concurrent accrual evaluations cannot interleave.
    /// Lock-guarded map populate (insert path only); the mutex clone
    /// out is lock-free thereafter.
    fn turnstile(&self, key: String) -> Arc<tokio::sync::Mutex<()>> {
        let mut map = self.turns.lock().expect("alert turns map poisoned");
        map.entry(key)
            .or_insert_with(|| Arc::new(tokio::sync::Mutex::new(())))
            .clone()
    }

    /// The shared fired-state registry. Startup restart-seeding
    /// (`seed_from_recovered`) and the Task-6 VK hooks go through this
    /// handle so all scopes share one registry.
    pub fn alert_state(&self) -> &Arc<AlertState> {
        &self.state
    }

    /// Whether the webhook subsystem is active (≥1 endpoint configured,
    /// Reqs 1.1/3.3). Cheap read under the config read lock; only ever
    /// called from off-path spawned tasks.
    pub async fn is_active(&self, config: &Arc<RwLock<Config>>) -> bool {
        let guard = config.read().await;
        !guard.webhooks.endpoints.is_empty()
    }

    /// Evaluate thresholds for one provider after a durable accrual
    /// (Reqs 1.1, 1.3-1.5, 1.8-1.10): resolves limit / warn threshold /
    /// window label from config, runs the state machine against
    /// `durable_spend_usd` (the store value just persisted — the truth
    /// the emitted events must agree with), and emits every returned
    /// event through the notifier (which spawns delivery). No limit
    /// configured → no evaluation (Req 1.9).
    pub async fn evaluate_provider(
        &self,
        config: &Arc<RwLock<Config>>,
        provider: &str,
        durable_spend_usd: f64,
        window_anchor: i64,
    ) {
        // Dormancy contract first (Req 3.3): zero endpoints → no
        // evaluation, no ledger writes.
        if !self.is_active(config).await {
            return;
        }
        // Per-scope turnstile (see `turnstile`): hold the scope's
        // sequence lock across forecast derivation + threshold
        // evaluation so concurrent accrual evaluations for the same
        // provider run in durable-write order.
        let turnstile = self.turnstile(format!("p:{provider}"));
        let _turn = turnstile.lock().await;
        let (limit_usd, warn_percent, policy, logs_db_path) = {
            let guard = config.read().await;
            let Some(budget) = guard
                .providers
                .iter()
                .find(|p| p.name == provider)
                .and_then(|p| p.budget.clone())
            else {
                // No budget configured: no percentage events (Req 1.9).
                return;
            };
            if budget.limit_usd <= 0.0 {
                return;
            }
            (
                budget.limit_usd,
                // Per-provider override, default 80 (Req 1.8).
                budget.alert_threshold_percent.unwrap_or(80),
                budget.reset_policy.unwrap_or(BudgetResetPolicy::Manual),
                guard.logging.database_path.clone(),
            )
        };
        // Spend-pace-forecasting Component 4: derive the projection
        // from durable state alongside the threshold facts. Windowed
        // policies project by window pace to the boundary
        // `next_reset_epoch` computes for the row's anchor; manual
        // scopes use the trailing 30-day ledger run rate (Req 3.7:
        // a failed ledger read downgrades to `None` — silent skip,
        // never a failed accrual path).
        let (forecast, forecast_boundary) = match provider_forecast(
            policy,
            limit_usd,
            durable_spend_usd,
            window_anchor,
            &logs_db_path,
            provider,
        )
        .await
        {
            Some((forecast, boundary)) => (Some(forecast), Some(boundary)),
            None => (None, None),
        };
        let input = EvalInput {
            scope: ScopeRef::Provider {
                name: provider.to_string(),
            },
            limit_usd,
            spent_usd: durable_spend_usd,
            warn_percent,
            window_anchor,
            window: Some(window_label(policy)),
            forecast,
            forecast_boundary,
        };
        // evaluate() returns at most one warn + one exhausted event per
        // accrual (Reqs 1.3-1.5); emit() spawns the delivery task
        // (Req 1.10).
        for event in self.state.evaluate(&input) {
            self.notifier.emit(event).await;
        }
    }

    /// Record a provider budget reset and emit the `budget.reset` event
    /// (Reqs 2.1, 2.2): `note_reset` re-arms the scope's fired state
    /// and builds the event carrying the NEW `window_start` anchor and
    /// the reset source. Called from the lazy window-rollover path and
    /// the admin reset endpoint — both already off the response path.
    pub async fn note_provider_reset(
        &self,
        config: &Arc<RwLock<Config>>,
        provider: &str,
        source: ResetSource,
        new_window_start: i64,
    ) {
        // Dormancy contract (Req 3.3): no endpoints → no event, and the
        // registry re-arm is pointless without delivery — skip both.
        if !self.is_active(config).await {
            return;
        }
        let scope = ScopeRef::Provider {
            name: provider.to_string(),
        };
        let event = self.state.note_reset(scope, source, new_window_start);
        self.notifier.emit(event).await;
    }

    /// Evaluate thresholds for one virtual key after a durable accrual
    /// (Task 6; Reqs 1.2-1.4): the record_usage call site resolves the
    /// key's durable budget facts (limit, spend including the
    /// just-recorded usage, window anchor) and calls this from inside
    /// its existing off-path spawn (design D4 — never the request
    /// path). Warn threshold is the fixed default 80: virtual keys
    /// have no per-key override (design D9). All limit/window facts
    /// arrive as values, so config is read only for the dormancy
    /// check (Req 3.3). No limit configured (`limit_usd <= 0`) →
    /// no evaluation (Req 1.9).
    pub async fn evaluate_virtual_key(
        &self,
        config: &Arc<RwLock<Config>>,
        key_id: &str,
        key_name: &str,
        limit_usd: f64,
        spent_usd: f64,
        window_anchor: i64,
        window: Option<crate::virtual_keys::models::BudgetWindow>,
        logs_db_path: &str,
    ) {
        // Dormancy contract first (Req 3.3): zero endpoints → no
        // evaluation, no ledger writes — and no forecast derivation
        // either (the ledger read only happens when delivery is live).
        if !self.is_active(config).await {
            return;
        }
        if limit_usd <= 0.0 {
            return;
        }
        // Per-scope turnstile (see `turnstile`): same ordering
        // guarantee as the provider path for key scopes. The forecast
        // derivation runs INSIDE the turnstile (closure) so the
        // read-then-evaluate sequence stays serialized in durable-write
        // order and the blocking ledger read cannot start while dormant.
        let turnstile = self.turnstile(format!("k:{key_id}"));
        let _turn = turnstile.lock().await;
        let (forecast, forecast_boundary) = match crate::virtual_keys::usage::virtual_key_forecast(
            window.clone(),
            limit_usd,
            spent_usd,
            window_anchor,
            logs_db_path,
            key_id,
        )
        .await
        {
            Some((projection, boundary)) => (Some(projection), Some(boundary)),
            None => (None, None),
        };
        let input = EvalInput {
            scope: ScopeRef::VirtualKey {
                key_id: key_id.to_string(),
                name: key_name.to_string(),
            },
            limit_usd,
            spent_usd,
            // Fixed default warn (design D9): no per-key threshold config.
            warn_percent: 80,
            window_anchor,
            window: Some(match window {
                Some(crate::virtual_keys::models::BudgetWindow::Daily) => "daily".to_string(),
                Some(crate::virtual_keys::models::BudgetWindow::Weekly) => "weekly".to_string(),
                Some(crate::virtual_keys::models::BudgetWindow::Monthly) => "monthly".to_string(),
                None => "lifetime".to_string(),
            }),
            forecast,
            forecast_boundary,
        };
        // evaluate() returns at most one warn + one exhausted event per
        // accrual (Reqs 1.3-1.5); emit() spawns the delivery task
        // (Req 1.10).
        for event in self.state.evaluate(&input) {
            self.notifier.emit(event).await;
        }
    }

    /// Record a virtual-key budget reset and emit the `budget.reset`
    /// event (Task 6; Reqs 2.2, 2.3): mirrors
    /// [`Self::note_provider_reset`] for the VK admin spend-reset
    /// endpoint. `new_window_start` is the anchor the store just
    /// persisted; the emitted event carries it plus the reset source.
    pub async fn note_virtual_key_reset(
        &self,
        config: &Arc<RwLock<Config>>,
        key_id: &str,
        key_name: &str,
        source: ResetSource,
        new_window_start: i64,
    ) {
        // Dormancy contract (Req 3.3).
        if !self.is_active(config).await {
            return;
        }
        let scope = ScopeRef::VirtualKey {
            key_id: key_id.to_string(),
            name: key_name.to_string(),
        };
        let event = self.state.note_reset(scope, source, new_window_start);
        self.notifier.emit(event).await;
    }
}

/// Envelope `data.window` label for a reset policy.
fn window_label(policy: BudgetResetPolicy) -> String {
    match policy {
        BudgetResetPolicy::Daily => "daily",
        BudgetResetPolicy::Weekly => "weekly",
        BudgetResetPolicy::Monthly => "monthly",
        BudgetResetPolicy::Manual => "manual",
    }
    .to_string()
}

/// Conservatively-derived durable projection for restart re-derivation
/// seeding (spend-pace-forecasting Component 4): pure window-pace math
/// from the durable row's own facts — limit, spend, anchor — with no
/// ledger dependency, so the seed never under-reports a fired forecast
/// warning (a seeded `forecast` flag can only suppress re-announcement,
/// so the conservative choice for boundary-blur cases is to NOT seed
/// the flag; `None` is the conservative-safe projection here). Manual
/// scopes return `None` from this pure helper — the durable trailing
/// aggregate the gateway seeding sites compute (Req 3.7) is the manual
/// seed projection. A zero/absent anchor cannot anchor a window; fall
/// back to the policy's current window start (the dashboard guard) or
/// `None` when even that is unknown.
pub(crate) fn seed_projection(
    policy: BudgetResetPolicy,
    limit_usd: f64,
    spent_usd: f64,
    window_anchor: i64,
) -> Option<Forecast> {
    let now = chrono::Utc::now();
    if policy == BudgetResetPolicy::Manual {
        return None;
    }
    let anchor = if window_anchor <= 0 {
        crate::metrics::current_window_start_for_policy(policy)
    } else {
        window_anchor
    };
    if anchor <= 0 {
        return None;
    }
    let window_start = chrono::DateTime::from_timestamp(anchor, 0)?;
    let window_end_epoch =
        next_reset_epoch(policy, window_start.timestamp().max(now.timestamp()))?;
    let window_end = chrono::DateTime::from_timestamp(window_end_epoch, 0)?;
    Some(forecast_windowed(limit_usd, spent_usd, window_start, window_end, now))
}

/// Derive the provider-scope spend-pace projection from durable state
/// (spend-pace-forecasting Component 4; Reqs 3.1, 3.7). Windowed
/// policies are pure boundary math against the row's anchor — the
/// same `next_reset_epoch` the rollover check uses, so the projection
/// and the reset agree on the boundary. Manual scopes read the
/// trailing 30-day ledger aggregate through a `spawn_blocking`
/// read-only ledger connection (the call site already runs on a
/// spawned off-path task; the ledger read stays off the tokio
/// workers); any failure downgrades to `None` (silent skip, Req 3.7).
/// The blocking ledger read is bounded by a 250 ms timeout (matching
/// the dashboard's forecast bound) so the per-scope turnstile is
/// never head-of-line blocked; a timeout is a silent skip. One
/// `now_epoch` is captured up front and used for both the SQL horizon
/// and the projection so the manual path runs on a single clock. A
/// zero/absent window anchor cannot anchor a windowed projection;
/// fall back to the policy's current window start (the dashboard
/// guard) or `None` when even that is unknown.
async fn provider_forecast(
    policy: BudgetResetPolicy,
    limit_usd: f64,
    spent_usd: f64,
    window_anchor: i64,
    logs_db_path: &str,
    provider: &str,
) -> Option<(Forecast, i64)> {
    let now = chrono::Utc::now();
    if policy == BudgetResetPolicy::Manual {
        let now_epoch = now.timestamp();
        let provider = provider.to_string();
        let db_path = logs_db_path.to_string();
        let totals = match tokio::time::timeout(
            std::time::Duration::from_millis(250),
            tokio::task::spawn_blocking(move || -> Option<_> {
                let conn = rusqlite::Connection::open_with_flags(
                    db_path,
                    rusqlite::OpenFlags::SQLITE_OPEN_READ_ONLY
                        | rusqlite::OpenFlags::SQLITE_OPEN_NO_MUTEX,
                )
                .ok()?;
                fetch_trailing_spend(
                    &conn,
                    TrailingScope::Provider(&provider),
                    now_epoch,
                )
                .ok()
            })
        )
.await
{ Ok(Ok(Some(totals))) => totals, _ => return None };
        return Some((
            forecast_manual(
                limit_usd,
                spent_usd,
                totals.cost_micro_usd as f64 / 1_000_000.0,
                totals.history_days,
                now,
            ),
            now_epoch + MANUAL_HORIZON_DAYS * 86_400,
        ));
    }
    let anchor = if window_anchor <= 0 {
        crate::metrics::current_window_start_for_policy(policy)
    } else {
        window_anchor
    };
    if anchor <= 0 {
        return None;
    }
    let window_start = chrono::DateTime::from_timestamp(anchor, 0)?;
    let window_end_epoch =
        next_reset_epoch(policy, window_start.timestamp().max(now.timestamp()))?;
    let window_end = chrono::DateTime::from_timestamp(window_end_epoch, 0)?;
    Some((
        forecast_windowed(limit_usd, spent_usd, window_start, window_end, now),
        window_end_epoch,
    ))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn window_label_maps_policy() {
        assert_eq!(window_label(BudgetResetPolicy::Daily), "daily");
        assert_eq!(window_label(BudgetResetPolicy::Manual), "manual");
    }
}
