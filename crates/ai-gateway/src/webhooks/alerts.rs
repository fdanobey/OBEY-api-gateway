//! Threshold state machine for budget-alert evaluation
//! (budget-alerts-unified-webhooks spec).
//!
//! This module owns the fired-state registry and decides, from
//! `(scope, limit, spent, window anchor, thresholds)`, whether a
//! [`BudgetAlertEvent`] fires. It is pure with respect to I/O — no config,
//! metrics, or async runtime dependencies — so the entire alert decision
//! logic is unit-testable and property-testable in isolation. Callers
//! (accrual paths, reset endpoints, startup recovery) own when to invoke it.
//!
//! Threshold semantics (Reqs 1.3-1.9):
//! - exhausted fires at `spent >= limit` (inclusive — same comparison as
//!   `budget_status_by_provider`'s `exhausted` derivation);
//! - warn fires at `spent >= warn% * limit` with `warn_percent` resolved by
//!   the caller (default 80; per-provider override `budget.alert_threshold_percent`
//!   — Req 1.8; virtual keys always pass the fixed default);
//! - each threshold fires at most once per (scope, window); a single jump
//!   from below-warn to exhausted emits `[warning, exhausted]` in order
//!   (Req 1.7);
//! - a reset (window rollover or admin) re-arms the scope by clearing
//! fired state only for windows OLDER than the reset's new anchor
//! (Reqs 1.6, 2.1, 2.2) — a concurrent accrual already evaluated under
//! the new anchor keeps its flags (no duplicate events);
//! - `evaluate` opportunistically evicts the scope's entries older
//! than the current window anchor, so composite (scope, anchor) keys
//! of warn-only windows (which never see a reset event) stay bounded.
//!
//! Registry is in-memory only (design D11): restart re-derivation via
//! [`AlertState::seed_from_recovered`] trades at most one duplicate
//! `budget.warning` per budget per mid-window restart for zero extra writes
//! on the accrual path; exhausted is always re-derived true so it never
//! re-announces. Seeds merge MONOTONICALLY into live entries (flags
//! OR'd, never cleared) and are skipped entirely when the live path
//! already tracks a newer anchor, so a delayed seed can neither erase
//! live fired state nor suppress post-startup crossings.
//!
//! _Requirements: 1.1-1.9, 2.1, 2.3_

use dashmap::DashMap;
use dashmap::mapref::entry::Entry;

use crate::webhooks::events::{BudgetAlertEvent, BudgetEventData, ResetSource, Scope};

/// Which budget a [`ScopeRef`] names: a provider by name, or a virtual key
/// by `(key_id, name)` (design "Component 1" interface note).
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub enum ScopeRef {
    Provider { name: String },
    VirtualKey { key_id: String, name: String },
}

impl ScopeRef {
    /// Stable identity string for the registry key: providers by name,
    /// keys by id (ids are unique and immutable; names are not).
    fn registry_id(&self) -> String {
        match self {
            ScopeRef::Provider { name } => format!("provider:{name}"),
            ScopeRef::VirtualKey { key_id, .. } => format!("virtual_key:{key_id}"),
        }
    }

    /// Envelope `data` fields (scope, name, key_id) for this scope.
    fn data_identity(&self) -> (Scope, Option<String>, Option<String>) {
        match self {
            ScopeRef::Provider { name } => {
                (Scope::Provider, Some(name.clone()), None)
            }
            ScopeRef::VirtualKey { key_id, name } => (
                Scope::VirtualKey,
                Some(name.clone()),
                Some(key_id.clone()),
            ),
        }
    }
}

/// Evaluation inputs for one accrual (design "Component 1"). `warn_percent`
/// is the already-resolved percentage (default 80; per-provider override
/// applied by the caller — Req 1.8). `window_anchor` is the current window's
/// UTC epoch-seconds start (the `window_start` concept from the budget
/// stores; 0 for manual/lifetime scopes, which re-arm only via admin reset).
#[derive(Debug, Clone, PartialEq)]
pub struct EvalInput {
    pub scope: ScopeRef,
    pub limit_usd: f64,
    pub spent_usd: f64,
    pub warn_percent: u8,
    pub window_anchor: i64,
    /// Window label for the envelope `data.window` field
    /// (`"daily" | "weekly" | "monthly" | "manual" | "lifetime"`).
    pub window: Option<String>,
}

/// Which thresholds have already fired for one (scope, window) entry.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
struct FiredState {
    warned: bool,
    exhausted: bool,
}

impl FiredState {
    /// Derive restart seeds from recovered spend: exhausted true where
    /// `spent >= limit`, warn true where `spent >= warn% * limit` (design
    /// "Restart re-derivation", D11).
    fn from_recovered(spent: f64, limit: f64, warn_pct: u8) -> Self {
        let warn_threshold = limit * f64::from(warn_pct) / 100.0;
        Self {
            warned: spent >= warn_threshold,
            exhausted: spent >= limit,
        }
    }
}

/// Registry key: `(scope identity, window anchor)` — fired state is
/// per-window, so a window rollover naturally starts a fresh key.
type RegistryKey = (String, i64);

/// In-memory fired-state registry + threshold evaluation
/// (design "Component 1").
///
/// All fired-state transitions go through the DashMap entry API — the
/// check-and-set happens atomically under the shard lock — so two
/// concurrent accruals crossing the same threshold produce exactly one
/// event (design "Atomicity").
///
/// No limit semantics (Req 1.9): the caller invokes `evaluate` only for
/// scopes with a configured `limit_usd > 0`. `evaluate` defensively returns
/// no events when `limit_usd <= 0` so an unconditional call site cannot
/// manufacture percentage events against an unconfigured budget.
#[derive(Debug, Default)]
pub struct AlertState {
    fired: DashMap<RegistryKey, FiredState>,
}

impl AlertState {
    /// Threshold evaluation (design "Component 1"): returns the events to
    /// fire for this accrual, in emission order — warning first, exhausted
    /// second when a single jump crosses both (Req 1.7). Each threshold
    /// fires at most once per (scope, window) (Reqs 1.5, 1.6); a new
    /// `window_anchor` is a new registry key, so a lazy rollover re-arms
    /// implicitly even before [`Self::note_reset`] is called.
pub fn evaluate(&self, input: &EvalInput) -> Vec<BudgetAlertEvent> {
// Req 1.9: no limit, no percentage events. The caller normally
// filters unconfigured scopes; this guard makes the function safe
// under unconditional invocation.
if input.limit_usd <= 0.0 {
return Vec::new();
}

// Stale-anchor eviction: reset/cleanup is gated on exhaustion, so a
// window that crossed warn but never exhausted would otherwise pin
// its composite (scope, anchor) key forever. Every evaluation for a
// scope is a free opportunity to drop that scope's entries older
// than the anchor being evaluated (the DashMap retain touches one
// shard per stale key; live keys stay untouched).
let scope_id = input.scope.registry_id();
self.fired
.retain(|(id, anchor), _| !(id == &scope_id && *anchor < input.window_anchor));

let mut events = Vec::new();
let key = (scope_id, input.window_anchor);
        let warn_threshold = input.limit_usd * f64::from(input.warn_percent) / 100.0;

        // Entry API: the read-modify-write of FiredState happens under the
        // shard lock, so concurrent evaluate() calls for the same key see
        // at most one transition per threshold (design "Atomicity").
        match self.fired.entry(key) {
            Entry::Occupied(mut occupied) => {
                let before = *occupied.get();
                Self::transition_occupied(
                    &mut occupied,
                    before,
                    input,
                    warn_threshold,
                    &mut events,
                );
            }
            Entry::Vacant(vacant) => {
                let mut state = FiredState::default();
                Self::transition_state(
                    &mut state,
                    input,
                    warn_threshold,
                    &mut events,
                );
                // Only insert when a threshold actually fired — below-warn
                // spend never allocates registry state.
                if state != FiredState::default() {
                    vacant.insert(state);
                }
            }
        }
        events
    }

    /// Threshold transitions shared by the occupied and vacant entry paths:
    /// warn first (Req 1.7 emission order; spent >= warn% of limit, Req
    /// 1.3), then exhausted (spent >= limit inclusive — the same
    /// comparison budget_status_by_provider applies, Req 1.4). Mutates
    /// `state` flags and pushes events for every threshold newly crossed.
    fn transition_state(
        state: &mut FiredState,
        input: &EvalInput,
        warn_threshold: f64,
        events: &mut Vec<BudgetAlertEvent>,
    ) {
        let (data_scope, name, key_id) = input.scope.data_identity();
        if !state.warned && input.spent_usd >= warn_threshold {
            state.warned = true;
            events.push(BudgetAlertEvent::warning(BudgetEventData {
                scope: data_scope,
                name: name.clone(),
                key_id: key_id.clone(),
                limit_usd: Some(input.limit_usd),
                spent_usd: Some(input.spent_usd),
                percent: Some(input.warn_percent),
                window: input.window.clone(),
                window_start: Some(input.window_anchor),
                reset_source: None,
            }));
        }
        if !state.exhausted && input.spent_usd >= input.limit_usd {
            state.exhausted = true;
            events.push(BudgetAlertEvent::exhausted(BudgetEventData {
                scope: data_scope,
                name,
                key_id,
                limit_usd: Some(input.limit_usd),
                spent_usd: Some(input.spent_usd),
                percent: None,
                window: input.window.clone(),
                window_start: Some(input.window_anchor),
                reset_source: None,
            }));
        }
    }

    /// Shared threshold transitions under an occupied entry (state visible
    /// to concurrent evaluations only after the shard lock releases).
    fn transition_occupied(
        occupied: &mut dashmap::mapref::entry::OccupiedEntry<'_, RegistryKey, FiredState>,
        before: FiredState,
        input: &EvalInput,
        warn_threshold: f64,
        events: &mut Vec<BudgetAlertEvent>,
    ) {
        let mut state = before;
        Self::transition_state(&mut state, input, warn_threshold, events);
        if state != before {
            occupied.insert(state);
        }
    }

/// Record a budget reset (Reqs 2.1-2.3): re-arms the scope by clearing
/// fired state ONLY for windows strictly older than the reset's
/// `window_start` (the retired anchor plus any stale ones) and returns
/// the `budget.reset` event whose `data` carries the NEW `window_start`
/// anchor and the reset source. Entries already evaluated at the new
/// (or a newer) anchor — e.g. a concurrent accrual that raced the
/// delayed reset processing — keep their flags, so a threshold that
/// already fired in the new window is not re-armed (no duplicate
/// events). Spend/percent fields stay `None` — the reset payload
/// identifies scope, source, and new anchor only (Req 2.3).
pub fn note_reset(
&self,
scope: ScopeRef,
source: ResetSource,
window_start: i64,
) -> BudgetAlertEvent {
// Re-arm: drop only entries whose window_anchor is OLDER than the
// incoming new window start. Anchors >= window_start belong to a
// window already evaluated by a live accrual and must survive.
let scope_id = scope.registry_id();
self.fired
.retain(|(id, anchor), _| !(id == &scope_id && *anchor < window_start));

        let (data_scope, name, key_id) = scope.data_identity();
        BudgetAlertEvent::reset(BudgetEventData {
            scope: data_scope,
            name,
            key_id,
            limit_usd: None,
            spent_usd: None,
            percent: None,
            window: None,
            window_start: Some(window_start),
            reset_source: Some(source),
        })
    }

/// Restart re-derivation (design "Component 1", D11): after budget
/// restore, seed fired state so an already-exhausted budget never
/// re-announces (exhausted seeded true at `spent >= limit`) and
/// already-warned budgets seed warn true at `spent >= warn% * limit`.
///
/// Anchor note: seeding pins the flags for the window anchor the
/// caller will evaluate under. Windowed scopes should be re-evaluated
/// with the anchor recovered from the budget store; manual/lifetime
/// scopes use anchor 0. When the recovered spend sits below the warn
/// line but the pre-restart process had already warned, the first
/// post-restart crossing re-fires the warning — at most one duplicate
/// warning per budget per restart, never a duplicate exhausted (D11).
///
/// Merge semantics (monotonic, anchor-guarded): the seed writes only
/// when the registry has NO live entry for the (scope, anchor) key.
/// A live entry means the post-restart process already evaluated an
/// accrual under this anchor — its flags are strictly fresher, so the
/// seed merges by OR'ing its recovered flags into them (a delayed seed
/// must never CLEAR a live-fired warning/exhausted, which would permit
/// duplicates) and never inserts over an entry whose anchor is newer
/// (a live transition under a newer window must not be suppressed by
/// a stale seed). When no live entry exists, the seed inserts only if
/// some threshold is already crossed — below-warn spend stays
/// unallocated exactly like a below-warn evaluate.
pub fn seed_from_recovered(
&self,
scope: ScopeRef,
spent: f64,
limit: f64,
warn_pct: u8,
window_anchor: i64,
) {
if limit <= 0.0 {
// No limit configured: nothing to seed (Req 1.9 path).
return;
}
let seeded = FiredState::from_recovered(spent, limit, warn_pct);
let scope_id = scope.registry_id();
// Entry API keeps the seed atomic with any concurrent accrual:
// whichever lands first wins, and a racing seed+accrual pair never
// loses flags (merge is OR; insert only on vacant, and only when a
// threshold is already crossed — a below-warn seed allocates nothing,
// exactly like a below-warn evaluate).
if seeded != FiredState::default() {
self.fired
.entry((scope_id, window_anchor))
.and_modify(|live| {
live.warned |= seeded.warned;
live.exhausted |= seeded.exhausted;
})
.or_insert(seeded);
}
}
}

#[cfg(test)]
mod tests {
    use super::*;

    fn provider_scope() -> ScopeRef {
        ScopeRef::Provider { name: "openai".to_string() }
    }

    fn key_scope() -> ScopeRef {
        ScopeRef::VirtualKey {
            key_id: "key_1".to_string(),
            name: "ci".to_string(),
        }
    }

    fn input(scope: &ScopeRef, limit: f64, spent: f64, warn: u8, anchor: i64) -> EvalInput {
        EvalInput {
            scope: scope.clone(),
            limit_usd: limit,
            spent_usd: spent,
            warn_percent: warn,
            window_anchor: anchor,
            window: Some("daily".to_string()),
        }
    }

    fn is_warn(e: &BudgetAlertEvent) -> bool {
        e.event_type == crate::config::BudgetEventType::BudgetWarning
    }

    fn is_exhausted(e: &BudgetAlertEvent) -> bool {
        e.event_type == crate::config::BudgetEventType::BudgetExhausted
    }

    // === Boundary inclusivity ==============================================
    //
    // spent == limit fires exhausted (same inclusivity as
    // budget_status_by_provider); spent just below warn fires nothing.

    // Validates: Requirements 1.3, 1.4
    #[test]
    fn boundary_spent_equals_limit_fires_exhausted() {
        let state = AlertState::default();
        // Exactly at the limit and exactly at the warn line: both fire.
        let events = state.evaluate(&input(&provider_scope(), 10.0, 10.0, 80, 100));
        assert_eq!(events.len(), 2);
        assert!(is_warn(&events[0]) && is_exhausted(&events[1]));
    }

    // Validates: Requirements 1.3, 1.4
    #[test]
    fn boundary_just_below_warn_fires_nothing() {
        let state = AlertState::default();
        // Warn line = 8.0; spent 7.999... stays below.
        let events = state.evaluate(&input(&provider_scope(), 10.0, 7.999_999, 80, 100));
        assert!(events.is_empty());
        // And the registry stayed untouched (no allocation on no-fire).
        assert!(state.fired.is_empty());
    }

    // === Single jump: warning then exhausted ===============================

    // Validates: Requirement 1.7
    #[test]
    fn single_jump_below_warn_to_exhausted_emits_warning_then_exhausted() {
        let state = AlertState::default();
        let events = state.evaluate(&input(&provider_scope(), 10.0, 10.0, 80, 100));
        assert_eq!(events.len(), 2);
        assert!(is_warn(&events[0]));
        assert!(is_exhausted(&events[1]));
        // Warning carries the percent; exhausted does not.
        assert_eq!(events[0].data.percent, Some(80));
        assert_eq!(events[1].data.percent, None);
        // Emission order is part of the contract: warning first.
        assert_eq!(
            serde_json::to_string(&events[0].event_type).expect("serializes"),
            r#""budget.warning""#
        );
    }

    // === Once per window per threshold =====================================

    // Validates: Requirements 1.5, 1.6
    #[test]
    fn threshold_fires_once_per_window() {
        let state = AlertState::default();
        let scope = provider_scope();
        // Cross warn.
        let first = state.evaluate(&input(&scope, 10.0, 8.0, 80, 100));
        assert_eq!(first.len(), 1);
        // Same and higher spend, same window: nothing new.
        let again = state.evaluate(&input(&scope, 10.0, 9.0, 80, 100));
        assert!(again.is_empty());
        // Cross exhausted: only the exhausted event.
        let exh = state.evaluate(&input(&scope, 10.0, 10.0, 80, 100));
        assert_eq!(exh.len(), 1);
        assert!(is_exhausted(&exh[0]));
        // Further spend in the same window: nothing.
        let more = state.evaluate(&input(&scope, 10.0, 12.0, 80, 100));
        assert!(more.is_empty());
    }

    // Validates: Requirement 1.5
    #[test]
    fn distinct_scopes_and_windows_are_independent() {
        let state = AlertState::default();
        let p = provider_scope();
        let k = key_scope();
        assert_eq!(state.evaluate(&input(&p, 10.0, 8.0, 80, 100)).len(), 1);
        // Same threshold, other scope: still fires.
        assert_eq!(state.evaluate(&input(&k, 10.0, 8.0, 80, 100)).len(), 1);
        // Same scope, new window (new anchor): warn re-arms implicitly.
        assert_eq!(state.evaluate(&input(&p, 10.0, 8.0, 80, 200)).len(), 1);
    }

    // === Re-arm after reset ================================================

    // Validates: Requirements 1.6, 2.1, 2.2, 2.3
    #[test]
    fn note_reset_rearms_and_returns_reset_event() {
        let state = AlertState::default();
        let scope = provider_scope();
        // Exhaust both thresholds under the old window.
        state.evaluate(&input(&scope, 10.0, 10.0, 80, 100));
        // Admin reset with a fresh anchor.
        let reset = state.note_reset(scope.clone(), ResetSource::Admin, 300);
        assert_eq!(
            reset.event_type,
            crate::config::BudgetEventType::BudgetReset
        );
        assert_eq!(reset.data.reset_source, Some(ResetSource::Admin));
        assert_eq!(reset.data.window_start, Some(300));
        assert_eq!(reset.data.scope, Scope::Provider);
        assert_eq!(reset.data.name.as_deref(), Some("openai"));
        assert_eq!(reset.data.key_id, None);
        // Post-reset crossing fires again (re-armed) under the new anchor.
        let re_crossed = state.evaluate(&input(&scope, 10.0, 10.0, 80, 300));
        assert_eq!(re_crossed.len(), 2);
        assert!(is_warn(&re_crossed[0]) && is_exhausted(&re_crossed[1]));
    }

// Validates: Requirements 1.6, 2.1 (window rollover re-arm path)
#[test]
fn note_reset_rollover_source_round_trips() {
let state = AlertState::default();
let scope = key_scope();
state.evaluate(&input(&scope, 5.0, 5.0, 80, 100));
let reset = state.note_reset(scope.clone(), ResetSource::WindowRollover, 200);
let json = serde_json::to_string(&reset).expect("serializes");
assert!(json.contains(r#""type":"budget.reset""#));
assert!(json.contains(r#""reset_source":"window_rollover""#));
assert!(json.contains(r#""key_id":"key_1""#));
assert!(json.contains(r#""scope":"virtual_key""#));
// Re-armed: crossing at the new anchor fires again.
assert_eq!(state.evaluate(&input(&scope, 5.0, 4.0, 80, 200)).len(), 1);
}

// Validates: Requirements 1.5, 1.6 (delayed reset never re-arms a live window)
#[test]
fn note_reset_does_not_clear_entries_at_or_newer_than_new_anchor() {
let state = AlertState::default();
let scope = provider_scope();
// Live accrual under the NEW window crosses warn (fires once).
assert_eq!(state.evaluate(&input(&scope, 10.0, 8.0, 80, 300)).len(), 1);
// Delayed reset for the retired 100 window arrives with new anchor
// 300 — it must NOT clear the 300 entry that already fired.
state.note_reset(scope.clone(), ResetSource::WindowRollover, 300);
// Further spend in the 300 window: warn stays fired (no duplicate).
assert!(state.evaluate(&input(&scope, 10.0, 9.0, 80, 300)).is_empty());
// The retired (strictly older) anchor WAS re-armed: a later
// evaluation at the old anchor fires again.
assert_eq!(
state.evaluate(&input(&scope, 10.0, 12.0, 80, 100)).len(),
2,
"retired (older) anchor was re-armed by the reset"
);
}

// Validates: bounded registry (stale warn-only windows get evicted)
#[test]
fn evaluate_evicts_stale_anchors_for_the_evaluated_scope() {
let state = AlertState::default();
let scope = provider_scope();
let other = key_scope();
// `scope` accumulates warn-only entries across several windows
// (windows that never exhaust also never see a reset event).
for anchor in [100, 200, 300, 400] {
state.evaluate(&input(&scope, 10.0, 8.0, 80, anchor));
}
// `other` has one entry at an old anchor.
state.evaluate(&input(&other, 10.0, 8.0, 80, 50));
// The next evaluation for `scope` at a newer anchor evicts only
// `scope`'s strictly older entries — the other scope keeps its own.
state.evaluate(&input(&scope, 10.0, 8.0, 80, 500));
let provider_entries = state
.fired
.iter()
.filter(|entry| entry.key().0 == scope.registry_id())
.count();
let other_entries = state
.fired
.iter()
.filter(|entry| entry.key().0 == other.registry_id())
.count();
assert_eq!(provider_entries, 1, "only the current anchor survives");
assert_eq!(other_entries, 1, "other scopes are untouched");
}

    // === Restart seeding ====================================================

// Validates: Requirements 1.5, 1.6 (restart re-derivation, design D11)
#[test]
fn seed_from_recovered_prevents_exhausted_refire() {
let state = AlertState::default();
let scope = provider_scope();
// Recovered at exhaustion, same window anchor as the next evaluate.
state.seed_from_recovered(scope.clone(), 10.0, 10.0, 80, 500);
let events = state.evaluate(&input(&scope, 10.0, 10.0, 80, 500));
// Warn was seeded true too (spent >= warn line): no duplicate.
assert!(events.is_empty());
}

// Validates: Requirements 1.5, 1.6 (restart re-derivation, design D11)
#[test]
fn seed_from_recovered_merges_monotonically_into_live_fired_state() {
// A live crossing already fired the warning; a delayed seed with
// below-warn recovered spend must NOT clear it (no duplicate).
let state = AlertState::default();
let scope = provider_scope();
let fired = state.evaluate(&input(&scope, 10.0, 8.0, 80, 500));
assert_eq!(fired.len(), 1);
// Stale seed (pre-restart spend below the warn line):
state.seed_from_recovered(scope.clone(), 3.0, 10.0, 80, 500);
// Same-window re-accrual stays silent — the live warned flag won.
assert!(state.evaluate(&input(&scope, 10.0, 9.0, 80, 500)).is_empty());
// The same holds for a live exhausted flag.
let state2 = AlertState::default();
state2.evaluate(&input(&scope, 10.0, 10.0, 80, 600));
state2.seed_from_recovered(scope.clone(), 1.0, 10.0, 80, 600);
assert!(state2.evaluate(&input(&scope, 10.0, 12.0, 80, 600)).is_empty());
}

// Validates: Requirements 1.5, 1.6 (stale seed never suppresses a live window)
#[test]
fn seed_from_recovered_does_not_overwrite_newer_live_anchor() {
let state = AlertState::default();
let scope = provider_scope();
// Live transition under the NEW window already warned + exhausted.
state.evaluate(&input(&scope, 10.0, 10.0, 80, 900));
// A delayed seed carrying the OLD (smaller) anchor arrives late —
// it merges into its own (scope, old-anchor) key and must not touch
// the live 900 entry.
state.seed_from_recovered(scope.clone(), 9.0, 10.0, 80, 500);
// The stale 500 entry got its own seed (spent 9 >= warn line).
assert!(state.evaluate(&input(&scope, 10.0, 9.5, 80, 500)).is_empty());
// The 900 window stays silent (flags survived the stale seed).
assert!(state.evaluate(&input(&scope, 10.0, 11.0, 80, 900)).is_empty());
}
// Validates: design D11 (at most one duplicate warning)
#[test]
fn seed_from_recovered_allows_at_most_one_duplicate_warning() {
        // Case A: seeding observed the warn state (spent over the line) —
        // no duplicate warning at all.
        let state = AlertState::default();
        let scope = provider_scope();
        state.seed_from_recovered(scope.clone(), 9.0, 10.0, 80, 0);
        assert!(state.evaluate(&input(&scope, 10.0, 9.0, 80, 0)).is_empty());

        // Case B (the D11 tolerance): pre-restart spend sat below the warn
        // line, post-restart accrual crosses it — warning fires once, then
        // never again for that window.
        let state2 = AlertState::default();
        state2.seed_from_recovered(scope.clone(), 7.0, 10.0, 80, 0);
        let once = state2.evaluate(&input(&scope, 10.0, 8.0, 80, 0));
        assert_eq!(once.len(), 1);
        assert!(is_warn(&once[0]));
        let twice = state2.evaluate(&input(&scope, 10.0, 9.0, 80, 0));
        assert!(twice.is_empty());
    }

    // Validates: Requirements 1.5, 1.6 (exhausted seed never re-announces)
    #[test]
    fn seed_from_recovered_never_reexhausts_even_above_limit() {
        let state = AlertState::default();
        let scope = key_scope();
        // Recovered above the limit; further accrual must stay silent on
        // exhausted (already seeded true).
        state.seed_from_recovered(scope.clone(), 12.0, 10.0, 80, 100);
        let events = state.evaluate(&input(&scope, 10.0, 13.0, 80, 100));
        assert!(events.is_empty());
    }

    // === Threshold override math ===========================================

    // Validates: Requirement 1.8
    #[test]
    fn warn_percent_override_changes_the_line() {
        let state = AlertState::default();
        let scope = provider_scope();
        // 50% override: 5.0/10.0 crosses.
        let events = state.evaluate(&input(&scope, 10.0, 5.0, 50, 100));
        assert_eq!(events.len(), 1);
        assert!(is_warn(&events[0]));
        assert_eq!(events[0].data.percent, Some(50));

        // Default 80: 5.0 does not cross.
        let state2 = AlertState::default();
        let none = state2.evaluate(&input(&scope, 10.0, 5.0, 80, 100));
        assert!(none.is_empty());
    }

    // Validates: Requirement 1.9
    #[test]
    fn no_limit_fires_nothing_even_when_called() {
        let state = AlertState::default();
        assert!(state
            .evaluate(&input(&provider_scope(), 0.0, 100.0, 80, 0))
            .is_empty());
        assert!(state
            .evaluate(&input(&provider_scope(), -1.0, 100.0, 80, 0))
            .is_empty());
    }

    // === Concurrent evaluation: exactly one event per threshold ===========

    // Validates: Requirements 1.5, 1.6 (design "Atomicity")
    #[test]
    fn concurrent_crossings_fire_exactly_one_event_per_threshold() {
        const THREADS: usize = 16;
        let state = std::sync::Arc::new(AlertState::default());
        let scope = std::sync::Arc::new(provider_scope());
        let mut handles = Vec::new();
        for _ in 0..THREADS {
            let state = std::sync::Arc::clone(&state);
            let scope = std::sync::Arc::clone(&scope);
            handles.push(std::thread::spawn(move || {
                state.evaluate(&input(&scope, 10.0, 10.0, 80, 100)).len()
            }));
        }
        let total: usize = handles
            .into_iter()
            .map(|h| h.join().expect("thread joins"))
            .sum();
        // The crossing includes both thresholds; each fires exactly once
        // across all concurrent evaluations (design "Atomicity").
        assert_eq!(total, 2);
    }

    // === Property-based tests ==============================================

    use proptest::prelude::*;

    proptest! {
        #![proptest_config(ProptestConfig::with_cases(64))]

        // Feature: budget-alerts-unified-webhooks, Property 1: monotonicity —
        // more spent (same limit/warn/window, fresh state) never yields
        // fewer events than less spent. Validates: Requirements 1.3, 1.4, 1.7
        #[test]
        fn prop_more_spent_never_fewer_events(
            limit in 1.0f64..=1000.0,
            spent_low in 0.0f64..=1000.0,
            spent_extra in 0.0f64..=1000.0,
            warn_pct in 1u8..=100,
            anchor in 0i64..=1_000_000,
        ) {
            let spent_high = spent_low + spent_extra;
            let scope = provider_scope();
            let low = AlertState::default()
                .evaluate(&input(&scope, limit, spent_low, warn_pct, anchor)).len();
            let high = AlertState::default()
                .evaluate(&input(&scope, limit, spent_high, warn_pct, anchor)).len();
            prop_assert!(high >= low);
        }

        // Feature: budget-alerts-unified-webhooks, Property 2: no double
        // fire — for any (limit, spent, warn_pct), a threshold that has
        // already fired never fires again in the same window, even as
        // spend keeps rising. Validates: Requirements 1.5, 1.6
        #[test]
        fn prop_no_threshold_fires_twice(
            limit in 1.0f64..=1000.0,
            spent in 0.0f64..=2000.0,
            warn_pct in 1u8..=100,
            anchor in 0i64..=1_000_000,
        ) {
            let state = AlertState::default();
            let scope = provider_scope();
            let first = state.evaluate(&input(&scope, limit, spent, warn_pct, anchor));
            let second = state.evaluate(&input(&scope, limit, spent, warn_pct, anchor));
            let third = state.evaluate(&input(&scope, limit, spent * 1.1 + 1.0, warn_pct, anchor));
            // Each threshold appears at most once across the whole
            // sequence (repeat evaluations with equal AND rising spend).
            let all = first.iter().chain(second.iter()).chain(third.iter());
            let warnings = all.clone().filter(|e| is_warn(e)).count();
            let exhaustions = all.filter(|e| is_exhausted(e)).count();
            prop_assert!(warnings <= 1);
            prop_assert!(exhaustions <= 1);
            // The repeat at identical spend is fully silent.
            prop_assert!(second.is_empty());
        }
    }
}
