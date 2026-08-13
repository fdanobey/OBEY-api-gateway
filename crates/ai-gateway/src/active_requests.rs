//! Live registry of in-flight (currently executing) requests for the dashboard.
//!
//! Unlike the atomic `active_requests` counter in [`crate::metrics::Metrics`], this
//! registry tracks *individual* requests so the dashboard can show what each active
//! connection is doing and why a particular model/provider is being used (primary
//! attempt, retry after a transient error, failover to another provider, or a
//! smart-routing cascade). Only active requests are retained; entries are removed when
//! the request completes.

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
}

impl ActiveRequestInfo {
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
    pub fn len(&self) -> usize {
        self.entries.len()
    }

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
        }
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
