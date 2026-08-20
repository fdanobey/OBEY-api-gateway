use dashmap::DashMap;
use std::collections::HashMap;
use std::sync::Arc;
use std::time::Duration;

/// Immutable routing-time latency snapshot.
/// Captures all known provider latencies and one fallback median in a single DashMap traversal.
#[derive(Debug, Clone)]
pub struct LatencySnapshot {
    known: HashMap<String, f64>,
    fallback_ms: f64,
}

impl LatencySnapshot {
    /// Get latency for a provider from the snapshot.
    /// Returns the fallback median if the provider was not in the known set.
    #[inline]
    pub fn get_latency(&self, provider: &str) -> f64 {
        self.known
            .get(provider)
            .copied()
            .unwrap_or(self.fallback_ms)
    }

    // Test-only accessors for asserting snapshot contents. Compiled out of
    // normal builds so the dual bin/lib targets stay warning-clean.
    #[cfg(test)]
    #[inline]
    pub fn has_latency(&self, provider: &str) -> bool {
        self.known.contains_key(provider)
    }

    #[cfg(test)]
    #[inline]
    pub fn fallback(&self) -> f64 {
        self.fallback_ms
    }

    #[cfg(test)]
    #[inline]
    pub fn known_count(&self) -> usize {
        self.known.len()
    }
}

/// Tracks per-provider latency using exponential moving average
#[derive(Debug, Clone)]
pub struct LatencyTracker {
    latencies: Arc<DashMap<String, f64>>,
    alpha: f64, // EMA smoothing factor (0.2 = 20% weight to new value)
}

impl LatencyTracker {
    /// Create a new LatencyTracker with default alpha of 0.2
    pub fn new() -> Self {
        Self {
            latencies: Arc::new(DashMap::new()),
            alpha: 0.2,
        }
    }

    /// Get latency for a provider in milliseconds
    /// Returns median of all providers if no history exists for this provider
    ///
    /// Production routing reads the immutable snapshot (`snapshot()` +
    /// `LatencySnapshot::get_latency`) instead of this live query, so the
    /// bin target considers it dead code. It remains part of the lib API:
    /// integration tests use it as the reference implementation when
    /// verifying snapshot equivalence.
    #[allow(dead_code)]
    #[inline]
    pub fn get_latency(&self, provider: &str) -> f64 {
        if let Some(latency) = self.latencies.get(provider) {
            *latency
        } else {
            self.calculate_median()
        }
    }

    /// Update latency for a provider using exponential moving average
    pub fn update_latency(&self, provider: &str, latency: Duration) {
        let latency_ms = latency.as_secs_f64() * 1000.0;

        self.latencies
            .entry(provider.to_string())
            .and_modify(|current| {
                *current = self.alpha * latency_ms + (1.0 - self.alpha) * *current;
            })
            .or_insert(latency_ms);
    }

    /// Create an immutable snapshot for provider selection.
    /// Traverses the DashMap once, collecting all known latencies and computing
    /// the fallback median in a single pass.
    pub fn snapshot(&self) -> LatencySnapshot {
        let mut values: Vec<f64> = Vec::new();
        let mut known: HashMap<String, f64> = HashMap::new();

        for entry in self.latencies.iter() {
            let latency = *entry.value();
            known.insert(entry.key().clone(), latency);
            values.push(latency);
        }

        let fallback_ms = if values.is_empty() {
            100.0 // Default 100ms if no providers tracked
        } else {
            values.sort_by(|a, b| a.partial_cmp(b).unwrap());
            let len = values.len();
            if len % 2 == 0 {
                (values[len / 2 - 1] + values[len / 2]) / 2.0
            } else {
                values[len / 2]
            }
        };

        LatencySnapshot { known, fallback_ms }
    }

    /// Calculate median latency of all tracked providers
    /// (only reachable via `get_latency`; see its note on the bin target)
    #[allow(dead_code)]
    fn calculate_median(&self) -> f64 {
        let mut values: Vec<f64> = self.latencies.iter().map(|entry| *entry.value()).collect();

        if values.is_empty() {
            return 100.0; // Default 100ms if no providers tracked
        }

        values.sort_by(|a, b| a.partial_cmp(b).unwrap());
        let len = values.len();

        if len % 2 == 0 {
            (values[len / 2 - 1] + values[len / 2]) / 2.0
        } else {
            values[len / 2]
        }
    }
}

impl Default for LatencyTracker {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_initial_latency_returns_default() {
        let tracker = LatencyTracker::new();
        assert_eq!(tracker.get_latency("unknown"), 100.0);
    }

    #[test]
    fn test_update_and_get_latency() {
        let tracker = LatencyTracker::new();
        tracker.update_latency("provider1", Duration::from_millis(50));
        assert_eq!(tracker.get_latency("provider1"), 50.0);
    }

    #[test]
    fn test_exponential_moving_average() {
        let tracker = LatencyTracker::new();
        tracker.update_latency("provider1", Duration::from_millis(100));
        tracker.update_latency("provider1", Duration::from_millis(200));

        // EMA: 0.2 * 200 + 0.8 * 100 = 120
        assert_eq!(tracker.get_latency("provider1"), 120.0);
    }

    #[test]
    fn test_median_fallback_single_provider() {
        let tracker = LatencyTracker::new();
        tracker.update_latency("provider1", Duration::from_millis(150));
        assert_eq!(tracker.get_latency("unknown"), 150.0);
    }

    #[test]
    fn test_median_fallback_multiple_providers() {
        let tracker = LatencyTracker::new();
        tracker.update_latency("provider1", Duration::from_millis(100));
        tracker.update_latency("provider2", Duration::from_millis(200));
        tracker.update_latency("provider3", Duration::from_millis(300));

        // Median of [100, 200, 300] = 200
        assert_eq!(tracker.get_latency("unknown"), 200.0);
    }

    #[test]
    fn test_median_fallback_even_count() {
        let tracker = LatencyTracker::new();
        tracker.update_latency("provider1", Duration::from_millis(100));
        tracker.update_latency("provider2", Duration::from_millis(300));

        // Median of [100, 300] = (100 + 300) / 2 = 200
        assert_eq!(tracker.get_latency("unknown"), 200.0);
    }

    #[test]
    fn test_snapshot_empty_history_default_fallback() {
        let tracker = LatencyTracker::new();
        let snap = tracker.snapshot();
        assert_eq!(snap.known_count(), 0);
        assert_eq!(snap.get_latency("any-provider"), 100.0);
        assert_eq!(snap.fallback(), 100.0);
        assert!(!snap.has_latency("any-provider"));
    }

    #[test]
    fn test_snapshot_complete_history() {
        let tracker = LatencyTracker::new();
        tracker.update_latency("provider1", Duration::from_millis(100));
        tracker.update_latency("provider2", Duration::from_millis(200));
        tracker.update_latency("provider3", Duration::from_millis(300));

        let snap = tracker.snapshot();
        assert_eq!(snap.known_count(), 3);
        assert!(snap.has_latency("provider1"));
        assert_eq!(snap.get_latency("provider1"), 100.0);
        assert_eq!(snap.get_latency("provider2"), 200.0);
        assert_eq!(snap.get_latency("provider3"), 300.0);
        // Fallback median of [100, 200, 300] = 200
        assert_eq!(snap.fallback(), 200.0);
        assert_eq!(snap.get_latency("unknown"), 200.0);
    }

    #[test]
    fn test_snapshot_partial_history_uses_fallback_for_unknown() {
        let tracker = LatencyTracker::new();
        tracker.update_latency("provider1", Duration::from_millis(100));
        tracker.update_latency("provider2", Duration::from_millis(300));

        let snap = tracker.snapshot();
        assert_eq!(snap.known_count(), 2);
        // Known provider uses tracked value
        assert_eq!(snap.get_latency("provider1"), 100.0);
        // Unknown provider uses fallback median (100+300)/2 = 200
        assert!(!snap.has_latency("unknown"));
        assert_eq!(snap.get_latency("unknown"), 200.0);
    }

    #[test]
    fn test_snapshot_isolated_from_subsequent_updates() {
        let tracker = LatencyTracker::new();
        tracker.update_latency("provider1", Duration::from_millis(100));
        let snap = tracker.snapshot();

        // Update after snapshot — snapshot must be unchanged (immutability)
        tracker.update_latency("provider1", Duration::from_millis(500));
        assert_eq!(snap.get_latency("provider1"), 100.0);

        // A fresh snapshot reflects the update.
        // EMA applies on the second update: 0.2*500 + 0.8*100 = 180.0
        let snap2 = tracker.snapshot();
        assert_eq!(snap2.get_latency("provider1"), 180.0);
    }

    #[test]
    fn test_snapshot_equivalent_to_get_latency() {
        let tracker = LatencyTracker::new();
        tracker.update_latency("p1", Duration::from_millis(50));
        tracker.update_latency("p2", Duration::from_millis(150));
        tracker.update_latency("p3", Duration::from_millis(250));
        tracker.update_latency("p4", Duration::from_millis(400));

        let snap = tracker.snapshot();
        for provider in ["p1", "p2", "p3", "p4", "unknown-a", "unknown-b"] {
            assert_eq!(
                snap.get_latency(provider),
                tracker.get_latency(provider),
                "snapshot and live lookup must agree for {provider}"
            );
        }
    }

    #[test]
    fn test_snapshot_single_provider_fallback() {
        let tracker = LatencyTracker::new();
        tracker.update_latency("only", Duration::from_millis(150));
        let snap = tracker.snapshot();
        assert_eq!(snap.fallback(), 150.0);
        assert_eq!(snap.get_latency("unknown"), 150.0);
    }
}

#[cfg(test)]
mod property_tests {
    use super::*;
    use proptest::prelude::*;

    proptest! {
            #![proptest_config(ProptestConfig {
                cases: 64,
                .. ProptestConfig::default()
            })]

            /// **Validates: Requirements 7.1, 7.3, 7.4**
            ///
            /// Property 23: Latency Tracking Update
            ///
            /// For any successful request to a provider, the latency tracker shall update
            /// that provider's average latency using exponential moving average.
            ///
            /// This property verifies:
            /// 1. First update sets the latency directly (no previous history)
            /// 2. Subsequent updates apply EMA formula: new_avg = alpha * new_value + (1 - alpha) * old_avg
            /// 3. The alpha value (0.2) correctly weights new vs old values
            /// 4. Updates are correctly tracked per provider independently
            #[test]
            fn prop_latency_tracking_update(
                provider_name in "[a-z]{3,10}",
                initial_latency_ms in 10u64..=5000,
                subsequent_latencies in prop::collection::vec(10u64..=5000, 1..=10)
            ) {
                let tracker = LatencyTracker::new();
                let alpha = 0.2;

                // First update: should set latency directly
                tracker.update_latency(&provider_name, Duration::from_millis(initial_latency_ms));
                let first_latency = tracker.get_latency(&provider_name);
                assert!((first_latency - initial_latency_ms as f64).abs() < 0.01,
                    "First update should set latency directly: expected {}, got {}",
                    initial_latency_ms, first_latency);

                // Subsequent updates: should apply EMA
                let mut expected_latency = initial_latency_ms as f64;
                for &new_latency_ms in &subsequent_latencies {
                    tracker.update_latency(&provider_name, Duration::from_millis(new_latency_ms));

                    // Calculate expected EMA: alpha * new + (1 - alpha) * old
                    expected_latency = alpha * (new_latency_ms as f64) + (1.0 - alpha) * expected_latency;

                    let actual_latency = tracker.get_latency(&provider_name);
                    assert!((actual_latency - expected_latency).abs() < 0.01,
                        "EMA calculation incorrect: expected {}, got {}",
                        expected_latency, actual_latency);
                }
            }

            /// Property: Multiple providers tracked independently
            ///
            /// Verifies that latency updates for different providers don't interfere with each other.
            #[test]
            fn prop_latency_tracking_independence(
                providers in prop::collection::vec("[a-z]{3,10}", 2..=5),
                latencies in prop::collection::vec(10u64..=5000, 2..=5)
            ) {
                prop_assume!(providers.len() == latencies.len());

                let tracker = LatencyTracker::new();

                // Update each provider with its latency
                for (provider, &latency_ms) in providers.iter().zip(latencies.iter()) {
                    tracker.update_latency(provider, Duration::from_millis(latency_ms));
                }

                // Verify each provider has its correct latency
                for (provider, &latency_ms) in providers.iter().zip(latencies.iter()) {
                    let actual = tracker.get_latency(provider);
                    assert!((actual - latency_ms as f64).abs() < 0.01,
                        "Provider {} should have latency {}, got {}",
                        provider, latency_ms, actual);
                }
            }

            /// Property: EMA converges toward new values
            ///
            /// Verifies that repeated updates with the same value cause the EMA to converge
            /// toward that value.
            #[test]
            fn prop_latency_ema_convergence(
                provider_name in "[a-z]{3,10}",
                initial_latency_ms in 100u64..=500,
                target_latency_ms in 1000u64..=2000,
                update_count in 5usize..=20
            ) {
                let tracker = LatencyTracker::new();

                // Set initial latency
                tracker.update_latency(&provider_name, Duration::from_millis(initial_latency_ms));

                // Apply multiple updates with target latency
                for _ in 0..update_count {
                    tracker.update_latency(&provider_name, Duration::from_millis(target_latency_ms));
                }

                let final_latency = tracker.get_latency(&provider_name);

                // After many updates, EMA should be closer to target than to initial
                let distance_to_target = (final_latency - target_latency_ms as f64).abs();
                let distance_to_initial = (final_latency - initial_latency_ms as f64).abs();

                assert!(distance_to_target < distance_to_initial,
                    "After {} updates, latency should be closer to target {} than initial {}: got {}",
                    update_count, target_latency_ms, initial_latency_ms, final_latency);
            }

            /// **Validates: Requirements 7.5**
            ///
            /// Property 24: Initial Latency Assumption
            ///
            /// For any provider with no latency history, the assumed latency shall be
            /// the median of all providers with latency history.
            ///
            /// This property verifies:
            /// 1. When no providers have history, default latency is 100ms
            /// 2. When providers have history, unknown provider gets median latency
            /// 3. Median calculation is correct for odd and even number of providers
            /// 4. Median is calculated from current latency values, not initial values
        #[test]
        fn prop_initial_latency_assumption(
            known in prop::collection::hash_set("[a-z]{3,10}", 0..=10)
                .prop_flat_map(|names| {
                    let count = names.len();
                    (Just(names), prop::collection::vec(10u64..=5000, count))
                }),
            unknown_provider in "[A-Z]{3,10}"
        ) {
            // Distinct provider names are required: the tracker keys latency by
            // provider name, so duplicate names would make the expected median
            // (computed over the raw latencies list) diverge from the tracker's
            // median over per-provider values.
            let known_providers: Vec<String> = known.0.into_iter().collect();
            let latencies = known.1;
            prop_assume!(!known_providers.contains(&unknown_provider.to_lowercase()));

                let tracker = LatencyTracker::new();

                // Case 1: No history - should return default 100ms
                if known_providers.is_empty() {
                    let latency = tracker.get_latency(&unknown_provider);
                    assert_eq!(latency, 100.0,
                        "With no provider history, unknown provider should get default 100ms, got {}",
                        latency);
                    return Ok(());
                }

                // Update known providers with their latencies
                for (provider, &latency_ms) in known_providers.iter().zip(latencies.iter()) {
                    tracker.update_latency(provider, Duration::from_millis(latency_ms));
                }

                // Calculate expected median
                let mut sorted_latencies = latencies.clone();
                sorted_latencies.sort_by(|a, b| a.partial_cmp(b).unwrap());
                let len = sorted_latencies.len();
                let expected_median = if len % 2 == 0 {
                    (sorted_latencies[len / 2 - 1] + sorted_latencies[len / 2]) as f64 / 2.0
                } else {
                    sorted_latencies[len / 2] as f64
                };

                // Case 2: With history - unknown provider should get median
                let actual_latency = tracker.get_latency(&unknown_provider);
                assert!((actual_latency - expected_median).abs() < 0.01,
                    "Unknown provider should get median latency: expected {}, got {}",
                    expected_median, actual_latency);

            // Case 3: Verify known providers still have their own latencies
            for (provider, &latency_ms) in known_providers.iter().zip(latencies.iter()) {
                let actual = tracker.get_latency(provider);
                assert!((actual - latency_ms as f64).abs() < 0.01,
                    "Known provider {} should retain its latency: expected {}, got {}",
                    provider, latency_ms, actual);
            }
            }

            /// Property: Snapshot equivalence with live lookup.
            ///
            /// **Validates: Requirements 2.2, 2.4, 2.6 (spec router-responsiveness-optimization)**
            ///
            /// For any set of known providers with latencies and any unknown provider,
            /// the immutable snapshot shall return exactly the same effective latency
            /// as the live `get_latency` path (known value, or one coherent fallback
            /// median / 100.0ms default when empty), and all unknown providers in one
            /// snapshot shall share the same fallback value.
            #[test]
            fn prop_snapshot_equivalent_to_live_lookup(
                known_providers in prop::collection::vec(("[a-z]{3,10}", 10u64..=5000), 0..=12),
                unknown_providers in prop::collection::vec("[a-z]{3,10}", 1..=4)
            ) {
                // Deduplicate by provider name BEFORE updating: a duplicate would
                // apply EMA in the tracker, so populate unique names only (first
                // update sets the value directly -> exact expected values).
                let unique: std::collections::BTreeMap<&str, u64> =
                    known_providers.iter().map(|(n, v)| (n.as_str(), *v)).collect();

                let tracker = LatencyTracker::new();
                for (name, latency_ms) in &unique {
                    tracker.update_latency(name, Duration::from_millis(*latency_ms));
                }

                let snap = tracker.snapshot();

                // Known providers: snapshot returns the exact tracked value.
                for (name, latency_ms) in &unique {
                    assert!(
                        (snap.get_latency(name) - *latency_ms as f64).abs() < 0.01,
                        "snapshot mismatch for known provider {name}"
                    );
                    assert!(snap.has_latency(name));
                }
                assert_eq!(snap.known_count(), unique.len());

                // Unknown providers: snapshot fallback equals live get_latency fallback.
                // Values use the tracker's own conversion (as_secs_f64()*1000.0) so the
                // expected median matches the stored f64 exactly.
                let mut sorted: Vec<f64> = unique
                    .values()
                    .map(|v| Duration::from_millis(*v).as_secs_f64() * 1000.0)
                    .collect();
                sorted.sort_by(|a, b| a.partial_cmp(b).unwrap());
                let expected_fallback = if sorted.is_empty() {
                    100.0
                } else {
                    let len = sorted.len();
                    if len % 2 == 0 {
                        (sorted[len / 2 - 1] + sorted[len / 2]) / 2.0
                    } else {
                        sorted[len / 2]
                    }
                };
                assert_eq!(snap.fallback(), expected_fallback);

                for unknown in &unknown_providers {
                    prop_assume!(!known_providers.iter().any(|(k, _)| k == unknown));
                    assert_eq!(
                        snap.get_latency(unknown),
                        tracker.get_latency(unknown),
                        "snapshot and live lookup disagree for unknown provider {unknown}"
                    );
                    assert!(!snap.has_latency(unknown));
                }

                // All unknown providers in one selection share the same fallback (Req 2.4).
                let first = snap.get_latency(&unknown_providers[0]);
                for unknown in &unknown_providers {
                    assert_eq!(snap.get_latency(unknown), first);
                }
            }
    }
}
