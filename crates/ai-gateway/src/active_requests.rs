//! Live registry of in-flight (currently executing) requests for the dashboard.
//!
//! Unlike the atomic `active_requests` counter in [`crate::metrics::Metrics`], this
//! registry tracks *individual* requests so the dashboard can show what each active
//! connection is doing and why a particular model/provider is being used (primary
//! attempt, retry after a transient error, failover to another provider, or a
//! smart-routing cascade). Only active requests are retained; entries are removed when
//! the request completes.

use crate::error::ProviderAttempt;
use dashmap::DashMap;
use serde::{Deserialize, Serialize};
use std::sync::Arc;
use std::sync::Mutex;
use std::time::{Duration, SystemTime, UNIX_EPOCH};

/// Why the current `provider:model` target is in use for this in-flight request.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum ActivePhase {
    /// Entry created but the router has not begun an attempt yet.
    Pending,
    /// First attempt against the highest-priority provider/model.
    Primary,
    /// Retrying the *same* provider/model after a retryable error (e.g. 429/408).
    Retry,
    /// Moved to a *different* provider/model because the previous one failed.
    Failover,
    /// Smart-routing response-quality cascade escalated to another tier/version.
    Cascade,
}

impl ActivePhase {
    /// Short human label for the dashboard badge.
    #[allow(dead_code)] // public API; may be used by the dashboard frontend
    pub fn label(&self) -> &'static str {
        match self {
            ActivePhase::Pending => "pending",
            ActivePhase::Primary => "primary",
            ActivePhase::Retry => "retry",
            ActivePhase::Failover => "failover",
            ActivePhase::Cascade => "cascade",
        }
    }
}

/// Whether the in-flight request is a streaming or buffered (non-stream) chat completion.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum RequestKind {
    Chat,
    Stream,
}

/// Snapshot of a single in-flight request, surfaced to the dashboard.
///
/// Contains only operational metadata — never prompts, response bodies, or message
/// content — consistent with the dashboard's privacy guarantees.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ActiveRequestInfo {
    pub trace_id: String,
    pub requested_model: String,
    #[serde(default)]
    pub model_group: Option<String>,
    #[serde(default)]
    pub provider: Option<String>,
    #[serde(default)]
    pub model: Option<String>,
    #[serde(default)]
    pub attempt: usize,
    pub phase: ActivePhase,
    #[serde(default)]
    pub last_error: Option<String>,
    #[serde(default)]
    pub virtual_key_id: Option<String>,
    /// Epoch milliseconds when the request started.
    pub started_at_ms: i64,
    pub kind: RequestKind,
    /// The per-request Smart Routing decision line was already logged.
    #[serde(skip)]
    pub smart_routing_logged: bool,
    /// Ledger of failed upstream tries and skips, drained into `#attempt` log rows.
    #[serde(skip)]
    pub failed_attempts: Vec<ProviderAttempt>,
    /// A terminal outcome row was written for this request.
    #[serde(skip)]
    pub outcome_logged: bool,
    /// Breaker pre-filter skips were already recorded for this request.
    #[serde(skip)]
    pub breaker_skips_recorded: bool,
}

impl ActiveRequestInfo {
    #[allow(dead_code)] // public API; may be used by the dashboard frontend
    pub fn elapsed_ms(&self) -> i64 {
        let now = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .map(|d| d.as_millis() as i64)
            .unwrap_or(0);
        (now - self.started_at_ms).max(0)
    }
}

/// Cloneable handle the router mutates as a request progresses through attempts.
#[derive(Debug, Clone)]
pub struct ActiveRequestHandle(pub Arc<Mutex<ActiveRequestInfo>>);

impl ActiveRequestHandle {
    /// Set the resolved model group name (known once the router finds it).
    pub fn set_group(&self, group: &str) {
        if let Ok(mut info) = self.0.lock() {
            info.model_group = Some(group.to_string());
        }
    }

    /// Set the current target provider/model and the phase describing why it is active.
    pub fn set_target(&self, provider: &str, model: &str, phase: ActivePhase) {
        if let Ok(mut info) = self.0.lock() {
            info.provider = Some(provider.to_string());
            info.model = Some(model.to_string());
            info.phase = phase;
        }
    }

    /// Override only the phase (e.g. switching to cascade before a re-route).
    pub fn set_phase(&self, phase: ActivePhase) {
        if let Ok(mut info) = self.0.lock() {
            info.phase = phase;
        }
    }

    /// Record the running attempt count.
    pub fn set_attempt(&self, attempt: usize) {
        if let Ok(mut info) = self.0.lock() {
            info.attempt = attempt;
        }
    }

    /// Record the error from the preceding attempt (shown as context for a retry/failover).
    pub fn set_last_error(&self, error: &str) {
        if let Ok(mut info) = self.0.lock() {
            info.last_error = Some(error.to_string());
        }
    }

    /// Returns the trace id the first time it is called for this request and
    /// `None` afterwards, so re-plans do not log the routing decision twice.
    pub fn claim_smart_routing_log(&self) -> Option<String> {
        let mut info = self.0.lock().ok()?;
        if std::mem::replace(&mut info.smart_routing_logged, true) {
            return None;
        }
        Some(info.trace_id.clone())
    }

    /// Milliseconds since the request started (0 if the lock is poisoned).
    pub fn elapsed_ms(&self) -> i64 {
        self.0.lock().map(|info| info.elapsed_ms()).unwrap_or(0)
    }

    /// Append a failed upstream try (or skip) to the request's attempt ledger.
    pub fn record_failed_attempt(&self, attempt: ProviderAttempt) {
        if let Ok(mut info) = self.0.lock() {
            info.failed_attempts.push(attempt);
        }
    }

    /// Number of attempts currently held in the ledger.
    pub fn failed_attempt_count(&self) -> usize {
        self.0.lock().map(|info| info.failed_attempts.len()).unwrap_or(0)
    }

    /// Drain the ledger, leaving it empty so a second drain writes nothing.
    pub fn take_failed_attempts(&self) -> Vec<ProviderAttempt> {
        self.0
            .lock()
            .map(|mut info| std::mem::take(&mut info.failed_attempts))
            .unwrap_or_default()
    }

    /// Mark that a terminal outcome row was written for this request.
    pub fn mark_outcome_logged(&self) {
        if let Ok(mut info) = self.0.lock() {
            info.outcome_logged = true;
        }
    }

    /// Whether a terminal outcome row was already written.
    pub fn outcome_logged(&self) -> bool {
        self.0.lock().map(|info| info.outcome_logged).unwrap_or(false)
    }

    /// Returns `true` only the first time it is called for this request, so
    /// breaker pre-filter skips are recorded once even across re-routes.
    pub fn claim_breaker_skip_recording(&self) -> bool {
        match self.0.lock() {
            Ok(mut info) => !std::mem::replace(&mut info.breaker_skips_recorded, true),
            Err(_) => false,
        }
    }
}

/// Registry of all currently in-flight requests, keyed by trace id.
#[derive(Debug, Default)]
pub struct ActiveRequestRegistry {
    entries: DashMap<String, ActiveRequestHandle>,
}

impl ActiveRequestRegistry {
    pub fn new() -> Self {
        Self {
            entries: DashMap::new(),
        }
    }

    /// Insert a request and return its handle (also usable by the router to update state).
    pub fn register(&self, info: ActiveRequestInfo) -> ActiveRequestHandle {
        let handle = ActiveRequestHandle(Arc::new(Mutex::new(info.clone())));
        self.entries.insert(info.trace_id.clone(), handle.clone());
        handle
    }

    /// Remove a request once it has completed (called from the request guard's drop).
    pub fn deregister(&self, trace_id: &str) {
        self.entries.remove(trace_id);
    }

    /// Number of currently in-flight requests.
    #[allow(dead_code)] // public API; may be used by the dashboard frontend
    pub fn len(&self) -> usize {
        self.entries.len()
    }

    #[allow(dead_code)] // public API; may be used by the dashboard frontend
    pub fn is_empty(&self) -> bool {
        self.entries.is_empty()
    }

    /// Clone all in-flight entries, oldest first, for serialization to the dashboard.
    pub fn snapshot(&self) -> Vec<ActiveRequestInfo> {
        let mut list: Vec<ActiveRequestInfo> = self
            .entries
            .iter()
            .filter_map(|entry| entry.value().0.lock().ok().map(|info| info.clone()))
            .collect();
        list.sort_by_key(|info| info.started_at_ms);
        list
    }

    /// Drop entries older than `max_age`, used as a safety net against leaked registrations.
    pub fn sweep_stale(&self, max_age: Duration) {
        let max_ms = max_age.as_millis() as i64;
        let now = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .map(|d| d.as_millis() as i64)
            .unwrap_or(0);
        let stale: Vec<String> = self
            .entries
            .iter()
            .filter_map(|entry| {
                let info = entry.value().0.lock().ok()?;
                if now - info.started_at_ms > max_ms {
                    Some(entry.key().clone())
                } else {
                    None
                }
            })
            .collect();
        for key in stale {
            self.entries.remove(&key);
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn sample_info(trace_id: &str) -> ActiveRequestInfo {
        ActiveRequestInfo {
            trace_id: trace_id.to_string(),
            requested_model: "gpt-4".to_string(),
            model_group: None,
            provider: None,
            model: None,
            attempt: 0,
            phase: ActivePhase::Pending,
            last_error: None,
            virtual_key_id: None,
            started_at_ms: SystemTime::now()
                .duration_since(UNIX_EPOCH)
                .unwrap()
                .as_millis() as i64,
            kind: RequestKind::Chat,
            smart_routing_logged: false,
            failed_attempts: Vec::new(),
            outcome_logged: false,
            breaker_skips_recorded: false,
        }
    }

    #[test]
    fn failed_attempt_ledger_records_and_drains() {
        let reg = ActiveRequestRegistry::new();
        let handle = reg.register(sample_info("trace-ledger"));
        assert_eq!(handle.failed_attempt_count(), 0);

        handle.record_failed_attempt(
            ProviderAttempt::new("p1".into(), "m1".into(), "timeout".into(), Some(504))
                .with_duration(Duration::from_millis(1500))
                .with_error_class("ttfb_timeout"),
        );
        handle.clone().record_failed_attempt(ProviderAttempt::new(
            "p2".into(),
            "m2".into(),
            "boom".into(),
            Some(500),
        ));
        assert_eq!(handle.failed_attempt_count(), 2);

        let drained = handle.take_failed_attempts();
        assert_eq!(drained.len(), 2);
        assert_eq!(drained[0].error_class.as_deref(), Some("ttfb_timeout"));
        assert_eq!(drained[0].duration_ms, Some(1500));
        assert_eq!(drained[1].provider, "p2");
        assert_eq!(handle.failed_attempt_count(), 0);
        assert!(handle.take_failed_attempts().is_empty());

        assert!(!handle.outcome_logged());
        handle.mark_outcome_logged();
        assert!(handle.outcome_logged());

        assert!(handle.claim_breaker_skip_recording());
        assert!(!handle.claim_breaker_skip_recording());

        // The ledger never reaches the dashboard JSON.
        let json = serde_json::to_value(&reg.snapshot()[0]).unwrap();
        assert!(json.get("failed_attempts").is_none());
        assert!(json.get("outcome_logged").is_none());
    }

    #[test]
    fn smart_routing_log_is_claimed_once_per_request() {
        let handle = ActiveRequestRegistry::new().register(sample_info("trace-4"));
        assert_eq!(handle.claim_smart_routing_log().as_deref(), Some("trace-4"));
        assert_eq!(handle.claim_smart_routing_log(), None);
        assert_eq!(handle.clone().claim_smart_routing_log(), None);
    }

    #[test]
    fn register_then_snapshot_contains_entry() {
        let reg = ActiveRequestRegistry::new();
        reg.register(sample_info("trace-1"));
        let snap = reg.snapshot();
        assert_eq!(snap.len(), 1);
        assert_eq!(snap[0].trace_id, "trace-1");
    }

    #[test]
    fn deregister_removes_entry() {
        let reg = ActiveRequestRegistry::new();
        reg.register(sample_info("trace-2"));
        assert_eq!(reg.len(), 1);
        reg.deregister("trace-2");
        assert!(reg.is_empty());
    }

    #[test]
    fn handle_updates_propagate_to_snapshot() {
        let reg = ActiveRequestRegistry::new();
        let handle = reg.register(sample_info("trace-3"));
        handle.set_group("default");
        handle.set_target("openai", "gpt-4", ActivePhase::Primary);
        handle.set_attempt(2);
        handle.set_last_error("429 rate limited");
        let snap = reg.snapshot();
        assert_eq!(snap[0].model_group.as_deref(), Some("default"));
        assert_eq!(snap[0].provider.as_deref(), Some("openai"));
        assert_eq!(snap[0].attempt, 2);
        assert_eq!(snap[0].last_error.as_deref(), Some("429 rate limited"));
    }

    #[test]
    fn sweep_stale_removes_only_expired() {
        let reg = ActiveRequestRegistry::new();
        let old = sample_info("old");
        let old = ActiveRequestInfo {
            started_at_ms: 0,
            ..old
        };
        reg.register(old);
        reg.register(sample_info("new"));
        reg.sweep_stale(Duration::from_secs(60));
        let snap = reg.snapshot();
        assert_eq!(snap.len(), 1);
        assert_eq!(snap[0].trace_id, "new");
    }
}
