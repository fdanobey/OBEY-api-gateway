//! Live registry of in-flight (currently executing) requests for the dashboard.
//!
//! Unlike the atomic `active_requests` counter in [`crate::metrics::Metrics`], this
//! registry tracks *individual* requests so the dashboard can show what each active
//! connection is doing and why a particular model/provider is being used (primary
//! attempt, retry after a transient error, failover to another provider, or a
//! smart-routing cascade). Only active requests are retained; entries are removed when
//! the request completes.

use crate::error::ProviderAttempt;
use crate::reasoning_relay::RelayHandle;
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

/// One-to-two-word reason the request is momentarily busy on gateway-side work
/// that produces no client output. Rendered under the phase badge in the
/// dashboard's Reason column so invisible LLM calls and waits are explainable.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum StallReason {
    /// Pre-call guardrail scan (PII/secrets/etc.) before routing.
    GuardrailScan,
    /// Post-call guardrail scan on a completed provider response.
    GuardrailReview,
    /// Memory retrieval/injection before provider dispatch.
    MemoryRecall,
    /// Buffered dispatch: streaming upstream and reassembling SSE before the
    /// first client byte.
    Buffering,
    /// Codex search agent loop round (web search + resubmission).
    CodexSearch,
    /// Tool-disclosure synthetic call resolved locally, model resubmitted.
    ToolDisclosure,
    /// Same-provider continuation nudge after a reasoning-only turn.
    ReasoningNudge,
    /// Retry backoff sleep before the next same-provider attempt.
    RetryBackoff,
    /// Waiting for a local provider rate-limit token (admission).
    RateLimitWait,
    /// Semantic-cache (embedding) lookup before the first provider attempt.
    CacheLookup,
    /// Smart-routing planning: classification, internal cache, budget, tiers.
    SmartRouting,
}

impl StallReason {
    /// Short human label for the dashboard.
    #[allow(dead_code)] // public API; mirrored by STALL_REASON_LABELS in the dashboard frontend
    pub fn label(&self) -> &'static str {
        match self {
            StallReason::GuardrailScan => "guardrail scan",
            StallReason::GuardrailReview => "guardrail review",
            StallReason::MemoryRecall => "memory recall",
            StallReason::Buffering => "buffering",
            StallReason::CodexSearch => "codex search",
            StallReason::ToolDisclosure => "tool disclosure",
            StallReason::ReasoningNudge => "reasoning nudge",
            StallReason::RetryBackoff => "retry backoff",
            StallReason::RateLimitWait => "rate-limit wait",
            StallReason::CacheLookup => "cache lookup",
            StallReason::SmartRouting => "smart routing",
        }
    }
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
    /// Why the request is currently stalled on invisible gateway work, if it is.
    /// Cleared when the work finishes so the badge reflects only live stalls.
    #[serde(default)]
    pub stall_reason: Option<StallReason>,
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
    /// Tap for forwarding buffered reasoning deltas to a waiting client
    /// (never serialized to the dashboard).
    #[serde(skip)]
    pub reasoning_relay: Option<RelayHandle>,
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

    /// Mark the request as busy on gateway-side work that produces no client
    /// output. The dashboard shows `reason.label()` under the phase badge.
    ///
    /// Prefer [`Self::begin_stall`], which returns an RAII scope that restores
    /// the previous marker on exit and is exception-safe across `?`/early
    /// returns; this raw setter exists for the few sites that intentionally
    /// hold the marker until request completion.
    pub fn set_stall_reason(&self, reason: StallReason) {
        if let Ok(mut info) = self.0.lock() {
            info.stall_reason = Some(reason);
        }
    }

    /// Begin a scoped stall: mark the request busy on invisible gateway work
    /// and return an RAII guard that restores the previous marker (nesting
    /// aware) when dropped, including on `?`/early-return paths.
    pub fn begin_stall(&self, reason: StallReason) -> StallScope {
        let previous = self
            .0
            .lock()
            .ok()
            .map(|mut info| std::mem::replace(&mut info.stall_reason, Some(reason)))
            .flatten();
        StallScope {
            handle: Some(self.clone()),
            previous,
        }
    }

    /// Begin a scoped stall on an optional handle (no-op when absent).
    pub fn begin_stall_opt(
        handle: Option<&ActiveRequestHandle>,
        reason: StallReason,
) -> StallScope {
        match handle {
            Some(handle) => handle.begin_stall(reason),
            None => StallScope {
                handle: None,
                previous: None,
            },
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

    /// Attach a reasoning relay handle for this request.
    pub fn attach_reasoning_relay(&self, relay: RelayHandle) {
        if let Ok(mut info) = self.0.lock() {
            info.reasoning_relay = Some(relay);
        }
    }

    /// Detach the relay before routing without a receiver.
pub fn detach_reasoning_relay(&self) {
if let Ok(mut info) = self.0.lock() {
info.reasoning_relay = None;
}
}

    /// Clone the attached reasoning relay handle, if any.
    pub fn reasoning_relay(&self) -> Option<RelayHandle> {
        self.0
            .lock()
            .ok()
            .and_then(|info| info.reasoning_relay.clone())
    }
}

/// RAII stall scope created by [`ActiveRequestHandle::begin_stall`].
///
/// Dropping the scope restores the stall marker that was active before the
/// scope began (`None` when the request was not stalled), so nested scopes
/// (e.g. `buffering` inside a `codex search` round) unwind in LIFO order
/// without erasing a still-active outer reason, and error paths (`?`,
/// early returns) unwind the marker the same way as success paths.
#[derive(Debug)]
pub struct StallScope {
    handle: Option<ActiveRequestHandle>,
    previous: Option<StallReason>,
}

impl StallScope {
    /// End this scope early, restoring the previous marker.
    pub fn end(self) {
        drop(self)
    }
}

impl Drop for StallScope {
    fn drop(&mut self) {
        if let Some(handle) = &self.handle {
            if let Ok(mut info) = handle.0.lock() {
                info.stall_reason = self.previous;
            }
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
        stall_reason: None,
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
            reasoning_relay: None,
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
fn stall_reason_round_trips_and_serializes() {
    let reg = ActiveRequestRegistry::new();
    let handle = reg.register(sample_info("trace-stall"));
    assert_eq!(reg.snapshot()[0].stall_reason, None);

    handle.set_stall_reason(StallReason::GuardrailScan);
    let snap = reg.snapshot();
    assert_eq!(snap[0].stall_reason, Some(StallReason::GuardrailScan));
    // Serialized to the dashboard under the snake_case name.
    let json = serde_json::to_value(&snap[0]).unwrap();
    assert_eq!(json["stall_reason"], serde_json::json!("guardrail_scan"));
}

#[test]
fn stall_scope_restores_previous_reason_on_drop() {
    let reg = ActiveRequestRegistry::new();
    let handle = reg.register(sample_info("trace-scope"));

    // Outer scope, held across an inner scope's full lifetime.
    let outer = handle.begin_stall(StallReason::CodexSearch);
    assert_eq!(reg.snapshot()[0].stall_reason, Some(StallReason::CodexSearch));

    {
        // Inner scope overwrites the marker for its duration...
        let inner = handle.begin_stall(StallReason::Buffering);
        assert_eq!(reg.snapshot()[0].stall_reason, Some(StallReason::Buffering));
        // ...and restores the outer reason when it ends.
        inner.end();
    }
    assert_eq!(reg.snapshot()[0].stall_reason, Some(StallReason::CodexSearch));

    // Dropping the outer scope clears back to the pre-stall state.
    drop(outer);
    assert_eq!(reg.snapshot()[0].stall_reason, None);
}

#[test]
fn stall_scope_nests_same_reason_without_losing_outer() {
    let reg = ActiveRequestRegistry::new();
    let handle = reg.register(sample_info("trace-nest"));

    let outer = handle.begin_stall(StallReason::GuardrailScan);
    let inner = handle.begin_stall(StallReason::GuardrailScan);
    inner.end();
    // Same-reason nesting still keeps the outer marker set.
    assert_eq!(reg.snapshot()[0].stall_reason, Some(StallReason::GuardrailScan));
    drop(outer);
    assert_eq!(reg.snapshot()[0].stall_reason, None);
}

#[test]
fn begin_stall_opt_handles_missing_handle() {
    let scope = ActiveRequestHandle::begin_stall_opt(None, StallReason::Buffering);
    // No handle: a no-op scope that drops cleanly.
    scope.end();
}

#[test]
fn stall_reason_labels_are_short() {
    // The dashboard constraint: one or two words.
    for reason in [
        StallReason::GuardrailScan,
        StallReason::GuardrailReview,
        StallReason::MemoryRecall,
        StallReason::Buffering,
        StallReason::CodexSearch,
        StallReason::ToolDisclosure,
        StallReason::ReasoningNudge,
        StallReason::RetryBackoff,
        StallReason::RateLimitWait,
        StallReason::CacheLookup,
        StallReason::SmartRouting,
    ] {
        assert!(reason.label().split_whitespace().count() <= 2);
    }
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

    #[test]
    fn reasoning_relay_round_trips_through_handle() {
        let handle = ActiveRequestRegistry::new().register(sample_info("trace-relay"));
        assert!(handle.reasoning_relay().is_none());

        let (relay, mut rx) = RelayHandle::new();
        handle.attach_reasoning_relay(relay);
        let retrieved = handle.reasoning_relay().expect("relay attached");
        retrieved.try_relay(crate::reasoning_relay::RelayFrame::Reasoning {
            text: "thinking".to_string(),
        });
        assert_eq!(retrieved.sent(), 1);
        match rx.try_recv() {
            Ok(crate::reasoning_relay::RelayFrame::Reasoning { text }) => {
                assert_eq!(text, "thinking");
            }
            other => panic!("expected relay frame, got {:?}", other),
        }
    }

#[test]
fn reasoning_relay_default_is_none() {
let handle = ActiveRequestRegistry::new().register(sample_info("trace-no-relay"));
assert!(handle.reasoning_relay().is_none());
}

#[test]
fn detach_reasoning_relay_clears_attached_handle() {
let handle = ActiveRequestRegistry::new().register(sample_info("trace-detach"));
let (relay, _rx) = RelayHandle::new();
handle.attach_reasoning_relay(relay);
assert!(handle.reasoning_relay().is_some());
handle.detach_reasoning_relay();
assert!(handle.reasoning_relay().is_none());
}

    #[test]
    fn reasoning_relay_never_reaches_dashboard_json() {
        let reg = ActiveRequestRegistry::new();
        let handle = reg.register(sample_info("trace-relay-json"));
        let (relay, _rx) = RelayHandle::new();
        handle.attach_reasoning_relay(relay);
        let json = serde_json::to_string(&reg.snapshot()[0]).unwrap();
        assert!(!json.contains("reasoning_relay"));
    }
}
