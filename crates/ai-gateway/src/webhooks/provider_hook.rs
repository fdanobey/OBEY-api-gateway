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
        }
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
        let (limit_usd, warn_percent, window) = {
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
                budget
                    .reset_policy
                    .map(window_label)
                    // Manual/unset policy: lifetime accumulation label.
                    .unwrap_or_else(|| "manual".to_string()),
            )
        };
        let input = EvalInput {
            scope: ScopeRef::Provider {
                name: provider.to_string(),
            },
            limit_usd,
            spent_usd: durable_spend_usd,
            warn_percent,
            window_anchor,
            window: Some(window),
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
        window: &str,
    ) {
        // Dormancy contract first (Req 3.3): zero endpoints → no
        // evaluation, no ledger writes.
        if !self.is_active(config).await {
            return;
        }
        if limit_usd <= 0.0 {
            return;
        }
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
            window: Some(window.to_string()),
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

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn window_label_maps_policy() {
        assert_eq!(window_label(BudgetResetPolicy::Daily), "daily");
        assert_eq!(window_label(BudgetResetPolicy::Manual), "manual");
    }
}
