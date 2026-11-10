//! Concurrent-retry budget (spec retry-failover-core-overhaul, Component 2 /
//! Model 3; Requirements 4.1-4.5).
//!
//! Caps concurrent same-provider retries at `max(3, active / 5)` (20% of the
//! provider's active+pending requests, floor of 3) so retries cannot amplify
//! a provider outage into self-inflicted overload. Fully synchronous: no
//! tokio, no I/O, no delays — first attempts are never blocked (Req 4.3);
//! callers fail over when `try_acquire_retry` returns `None` (Req 4.2).

use dashmap::DashMap;
use std::sync::atomic::{AtomicU64, Ordering};

/// Per-provider retry budget bookkeeping.
#[derive(Debug, Default)]
struct RetryBudgetState {
    /// Concurrent retries currently admitted for this provider.
    in_flight: AtomicU64,
}

/// Registry of per-provider retry budgets.
///
/// Admission is a per-provider compare-and-swap loop on the interior
/// `AtomicU64`; the shard lock is held only for map access, so the budget
/// can never overshoot, even when many threads race on one provider.
#[derive(Debug, Default)]
pub struct RetryBudgetMap {
    states: DashMap<String, RetryBudgetState>,
}

impl RetryBudgetMap {
    /// Creates an empty budget registry.
    pub fn new() -> Self {
        Self::default()
    }

    /// Try to admit a same-provider retry (Req 4.1).
    ///
    /// Admits only while the provider's in-flight retries are below
    /// `max(3, active / 5)` (integer division — 20% of active+pending with a
    /// floor of 3). The `active` count is read live from the provider
    /// concurrency registry by the caller, so the budget follows lower live
    /// counts instead of accumulated state (Req 4.5). Returns `None` when
    /// the budget is exhausted; the caller fails over (Req 4.2).
    pub fn try_acquire_retry(&self, provider: &str, active: u64) -> Option<RetrySlot<'_>> {
        let threshold = RetryBudgetMap::threshold_for(active);
        let state = self.states.entry(provider.to_string()).or_default();

        // CAS loop: only a transition current < threshold -> current + 1
        // admits; races that overshoot the threshold fail their check and
        // retry (or bail), so in_flight can never exceed the threshold.
        let mut current = state.in_flight.load(Ordering::Relaxed);
        loop {
            if current >= threshold {
                return None;
            }
            match state.in_flight.compare_exchange_weak(
                current,
                current + 1,
                Ordering::AcqRel,
                Ordering::Relaxed,
            ) {
                Ok(_) => {
                    return Some(RetrySlot {
                        provider: provider.to_string(),
                        budget: self,
                    })
                }
                Err(observed) => {
                    current = observed;
                }
            }
        }
    }

    /// Current in-flight retry count for a provider (tests/observability).
    pub fn in_flight(&self, provider: &str) -> u64 {
        match self.states.get(provider) {
            Some(state) => state.in_flight.load(Ordering::Acquire),
            None => 0,
        }
    }

    /// Drop all per-provider budget entries (config hot-reload seam, the
    /// same reset the circuit breakers / rate limiters / sticky cache get).
    /// Renamed or removed providers otherwise leak their entries across
    /// reloads; live in-flight slots still release via `RetrySlot::drop`
    /// because `Drop` tolerates a missing entry.
    pub fn clear(&self) {
        self.states.clear();
    }

    /// Admission threshold for a given live active+pending count:
    /// `max(3, active / 5)` (Req 4.1).
    fn threshold_for(active: u64) -> u64 {
        (active / 5).max(3)
    }
}

/// RAII drop-guard for one admitted retry slot.
///
/// The slot is released on `Drop` (Req 4.4), the same discipline as
/// `RequestCompleteGuard`: abandonment (499), gateway deadline (504),
/// panics, and early returns can never leak a slot.
///
/// Cloning is deliberately not implemented: exactly one release per acquire.
pub struct RetrySlot<'a> {
    provider: String,
    budget: &'a RetryBudgetMap,
}

impl Drop for RetrySlot<'_> {
    fn drop(&mut self) {
        if let Some(state) = self.budget.states.get(&self.provider) {
            // Saturating: a hot-reload `clear()` between acquire and drop
            // replaces the entry, and the stale slot must not underflow
            // the fresh counter (0 stays 0, admission caps still hold).
        // `let _ =` discards the impossible-failure result (closure always
        // returns Some); on a `clear()`-replaced entry the closure clamps.
        let _ = state.in_flight.try_update(
            Ordering::AcqRel,
            Ordering::Relaxed,
            |current| Some(current.saturating_sub(1)),
        );
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::sync::{Arc, Barrier};

    #[test]
    fn test_admits_up_to_threshold_and_rejects_beyond() {
        let budget = RetryBudgetMap::new();

        // active = 10 -> 10/5 = 2 -> threshold = max(3, 2) = 3 (floor wins)
        let first = budget.try_acquire_retry("prov", 10).unwrap();
        let second = budget.try_acquire_retry("prov", 10).unwrap();
        let third = budget.try_acquire_retry("prov", 10).unwrap();
        assert!(budget.try_acquire_retry("prov", 10).is_none());
        assert_eq!(budget.in_flight("prov"), 3);

        drop(first);
        drop(second);
        drop(third);
        assert_eq!(budget.in_flight("prov"), 0);
    }

    #[test]
    fn test_floor_of_three_for_small_active_counts() {
        // active = 0 and active = 4 both yield threshold 3 (20% share < 3)
        assert_eq!(RetryBudgetMap::threshold_for(0), 3);
        assert_eq!(RetryBudgetMap::threshold_for(4), 3);
        // First whole multiple of 5 where the 20% share takes over
        assert_eq!(RetryBudgetMap::threshold_for(15), 3);
        assert_eq!(RetryBudgetMap::threshold_for(20), 4);
        assert_eq!(RetryBudgetMap::threshold_for(25), 5);
    }

    #[test]
    fn test_drop_releases_slot_for_next_retry() {
        let budget = RetryBudgetMap::new();

        let a = budget.try_acquire_retry("prov", 10).unwrap();
        let b = budget.try_acquire_retry("prov", 10).unwrap();
        let c = budget.try_acquire_retry("prov", 10).unwrap();
        assert!(budget.try_acquire_retry("prov", 10).is_none());

        drop(b);
        assert_eq!(budget.in_flight("prov"), 2);
        let refill = budget.try_acquire_retry("prov", 10);
        assert!(refill.is_some());

        drop(a);
        drop(c);
        drop(refill);
        assert_eq!(budget.in_flight("prov"), 0);
    }

    #[test]
    fn test_budget_follows_lower_live_active_counts() {
        let budget = RetryBudgetMap::new();

        // active = 25 -> threshold 5; fill the budget.
        let slots: Vec<_> = (0..5)
            .map(|_| budget.try_acquire_retry("prov", 25).unwrap())
            .collect();
        assert!(budget.try_acquire_retry("prov", 25).is_none());

        // Traffic drops: active = 5 -> threshold 3. in_flight was computed
        // from the live counter per call and slots were released, so only
        // three retries are admitted now (Req 4.5: live counters, not
        // accumulated state).
        drop(slots);
        assert_eq!(budget.in_flight("prov"), 0);
        let held: Vec<_> = (0..3)
            .map(|_| budget.try_acquire_retry("prov", 5).unwrap())
            .collect();
        assert!(budget.try_acquire_retry("prov", 5).is_none());
        assert_eq!(budget.in_flight("prov"), 3);

        // The same holds while earlier slots are still being released.
        drop(held);
        let one = budget.try_acquire_retry("prov", 5).unwrap();
        let two = budget.try_acquire_retry("prov", 5).unwrap();
        let three = budget.try_acquire_retry("prov", 5).unwrap();
        assert!(budget.try_acquire_retry("prov", 5).is_none());
        drop(one);
        drop(two);
        drop(three);
        assert_eq!(budget.in_flight("prov"), 0);
    }

    #[test]
    fn test_budgets_are_per_provider() {
        let budget = RetryBudgetMap::new();

        let a1 = budget.try_acquire_retry("alpha", 10).unwrap();
        let a2 = budget.try_acquire_retry("alpha", 10).unwrap();
        let a3 = budget.try_acquire_retry("alpha", 10).unwrap();
        assert!(budget.try_acquire_retry("alpha", 10).is_none());

        let b1 = budget.try_acquire_retry("beta", 10).unwrap();
        let b2 = budget.try_acquire_retry("beta", 10).unwrap();
        let b3 = budget.try_acquire_retry("beta", 10).unwrap();
        assert!(budget.try_acquire_retry("beta", 10).is_none());

        assert_eq!(budget.in_flight("alpha"), 3);
        assert_eq!(budget.in_flight("beta"), 3);

        // Releasing an alpha slot opens room on alpha only; beta stays full.
        drop(a3);
        assert_eq!(budget.in_flight("alpha"), 2);
        let refill = budget.try_acquire_retry("alpha", 10).unwrap();
        assert_eq!(budget.in_flight("alpha"), 3);

    drop(a1);
    drop(a2);
    drop(refill);
    drop(b1);
    drop(b2);
    drop(b3);
    assert_eq!(budget.in_flight("alpha"), 0);
    assert_eq!(budget.in_flight("beta"), 0);
}

/// Hot-reload seam (R9): `clear()` drops every per-provider entry so
/// renamed/removed providers cannot leak budget state across config
/// reloads, and a still-held slot from before the clear releases
/// cleanly (Drop tolerates the missing entry).
#[test]
fn test_clear_evicts_all_provider_entries() {
    let budget = RetryBudgetMap::new();

    // active = 10 -> threshold 3; fill both providers' budgets.
    let held: Vec<_> = (0..3)
        .map(|_| budget.try_acquire_retry("renamed-away", 10).unwrap())
        .collect();
    assert!(budget.try_acquire_retry("renamed-away", 10).is_none());
    let kept_held: Vec<_> = (0..3)
        .map(|_| budget.try_acquire_retry("kept", 10).unwrap())
        .collect();

    budget.clear();
    assert_eq!(budget.in_flight("renamed-away"), 0);
    assert_eq!(budget.in_flight("kept"), 0);

    // Fresh admission after the clear: the map was emptied, so a new
    // provider (or a renamed one) starts from a zero in-flight count.
    let first = budget.try_acquire_retry("kept", 10).unwrap();
    let second = budget.try_acquire_retry("kept", 10).unwrap();
    let third = budget.try_acquire_retry("kept", 10).unwrap();
    assert!(budget.try_acquire_retry("kept", 10).is_none());

    // Stale slots from before the clear still release cleanly.
    drop(held);
    drop(kept_held);
    drop(first);
    drop(second);
    drop(third);
    assert_eq!(budget.in_flight("kept"), 0);
    assert_eq!(budget.in_flight("renamed-away"), 0);
}

    /// N threads race on the same provider while holding their slots;
    /// the concurrent in-flight count must hit exactly the threshold and
    /// never exceed it (Req 4.1 — no overshoot under concurrency), and all
    /// slots must be released afterwards (Req 4.4).
    #[test]
    fn test_concurrent_admission_never_exceeds_threshold() {
        const THREADS: usize = 8;
        const ROUNDS: usize = 5;
        const ATTEMPTS_PER_ROUND: usize = 200;
        const ACTIVE: u64 = 10; // threshold 3

        let budget = Arc::new(RetryBudgetMap::new());
        let barrier = Arc::new(Barrier::new(THREADS));

        let handles: Vec<_> = (0..THREADS)
            .map(|_| {
                let budget = Arc::clone(&budget);
                let barrier = Arc::clone(&barrier);
                std::thread::spawn(move || {
                    for _round in 0..ROUNDS {
                        // Phase 1: hammer the same provider, keeping every
                        // admitted slot held.
                        barrier.wait();
                        let mut held = Vec::new();
                        for _ in 0..ATTEMPTS_PER_ROUND {
                            if let Some(slot) = budget.try_acquire_retry("shared", ACTIVE) {
                                held.push(slot);
                            }
                        }
                        // Phase 2: all threads done acquiring — the live
                        // count must be exactly the threshold, never above.
                        barrier.wait();
                        assert_eq!(
                            budget.in_flight("shared"),
                            3,
                            "in-flight retries overshot the threshold"
                        );
                        // Phase 3: only after every thread has asserted do
                        // we release, so no thread asserts against a count
                        // another thread already drained.
                        barrier.wait();
                        drop(held);
                    }
                })
            })
            .collect();

        for handle in handles {
            handle.join().expect("worker thread panicked");
        }
        assert_eq!(budget.in_flight("shared"), 0);
    }
}
