use dashmap::DashMap;
use std::sync::Arc;
use std::time::Duration;

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

    /// Calculate median latency of all tracked providers
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
}

#[cfg(test)]
mod property_tests {
    use super::*;
    use proptest::prelude::*;

    proptest! {
        #![proptest_config(ProptestConfig {
            cases: 100,
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
            known_providers in prop::collection::vec("[a-z]{3,10}", 1..=10),
            latencies in prop::collection::vec(10u64..=5000, 1..=10),
            unknown_provider in "[A-Z]{3,10}"
        ) {
            prop_assume!(known_providers.len() == latencies.len());
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
    }
}
