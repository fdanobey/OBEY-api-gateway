//! Usage tracking for the Tool Pruner stage.
//!
//! Provides `UsageTracker` which manages per-session tool call frequency and
//! per-API-key aggregate statistics with bounded memory (LRU eviction at
//! 10,000 entries per key).

use std::collections::HashMap;
use std::sync::Arc;

use dashmap::DashMap;

/// Re-export from types for convenience.
#[allow(unused_imports)]
pub use crate::tool_compression::types::ToolUsageMap;

/// Maximum per-API-key tool entries before LRU eviction.
const MAX_KEY_ENTRIES: usize = 10_000;

/// Tracks tool call frequency per session and per API key.
///
/// Backed by the `DashMap`-based shared state in `ToolCompressionState`.
/// This struct provides a higher-level API over the raw maps.
pub struct UsageTracker {
    /// Per-session tool usage: session_id → HashMap<tool_name, count>.
    session_usage: Arc<DashMap<String, HashMap<String, u64>>>,
    /// Per-session request counter: session_id → request_count.
    session_request_count: Arc<DashMap<String, u64>>,
    /// Per-API-key aggregate usage: api_key → HashMap<tool_name, (count, last_update)>.
    /// The `last_update` is a monotonic counter used for LRU eviction.
    key_usage: Arc<DashMap<String, KeyUsageState>>,
}

/// Internal state for per-API-key usage tracking with LRU eviction.
#[derive(Debug, Clone)]
pub struct KeyUsageState {
    /// tool_name → (call_count, last_update_tick)
    pub entries: HashMap<String, (u64, u64)>,
    /// Monotonically increasing tick for LRU ordering.
    pub tick: u64,
}

impl KeyUsageState {
    pub fn new() -> Self {
        Self {
            entries: HashMap::new(),
            tick: 0,
        }
    }

    /// Increment the count for a tool, updating the LRU tick.
    /// Evicts oldest entries if exceeding MAX_KEY_ENTRIES.
    pub fn record(&mut self, tool_name: &str) {
        self.tick += 1;
        let entry = self.entries.entry(tool_name.to_string()).or_insert((0, 0));
        entry.0 += 1;
        entry.1 = self.tick;

        // LRU eviction if over capacity
        if self.entries.len() > MAX_KEY_ENTRIES {
            self.evict();
        }
    }

    /// Get the call count for a tool (0 if not tracked).
    pub fn get_count(&self, tool_name: &str) -> u64 {
        self.entries.get(tool_name).map(|(c, _)| *c).unwrap_or(0)
    }

    /// Evict the least-recently-updated entries to bring size to MAX_KEY_ENTRIES.
    fn evict(&mut self) {
        if self.entries.len() <= MAX_KEY_ENTRIES {
            return;
        }

        let to_remove = self.entries.len() - MAX_KEY_ENTRIES;

        // Collect (tool_name, tick) and sort by tick ascending (oldest first)
        let mut by_tick: Vec<(String, u64)> = self
            .entries
            .iter()
            .map(|(k, (_, tick))| (k.clone(), *tick))
            .collect();
        by_tick.sort_unstable_by_key(|(_, tick)| *tick);

        for (name, _) in by_tick.into_iter().take(to_remove) {
            self.entries.remove(&name);
        }
    }
}

impl Default for KeyUsageState {
    fn default() -> Self {
        Self::new()
    }
}

impl UsageTracker {
    /// Create a new `UsageTracker` from the shared state maps.
    pub fn new(
        session_usage: Arc<DashMap<String, HashMap<String, u64>>>,
        session_request_count: Arc<DashMap<String, u64>>,
        key_usage: Arc<DashMap<String, KeyUsageState>>,
    ) -> Self {
        Self {
            session_usage,
            session_request_count,
            key_usage,
        }
    }

    /// Record a tool call for a session. Increments the call count.
    pub fn record_tool_call(&self, session_id: &str, tool_name: &str) {
        let mut entry = self
            .session_usage
            .entry(session_id.to_string())
            .or_default();
        *entry.value_mut().entry(tool_name.to_string()).or_insert(0) += 1;
    }

    /// Record a request for a session. Increments the request count.
    pub fn record_request(&self, session_id: &str) {
        let mut entry = self
            .session_request_count
            .entry(session_id.to_string())
            .or_default();
        *entry.value_mut() += 1;
    }

    /// Get current session usage map (tool_name → count).
    pub fn get_session_usage(&self, session_id: &str) -> HashMap<String, u64> {
        self.session_usage
            .get(session_id)
            .map(|r| r.value().clone())
            .unwrap_or_default()
    }

    /// Get the request count for a session.
    pub fn get_request_count(&self, session_id: &str) -> u64 {
        self.session_request_count
            .get(session_id)
            .map(|r| *r.value())
            .unwrap_or(0)
    }

    /// Get per-API-key aggregate call count for a tool.
    pub fn get_key_usage(&self, api_key_id: &str, tool_name: &str) -> u64 {
        self.key_usage
            .get(api_key_id)
            .map(|r| r.value().get_count(tool_name))
            .unwrap_or(0)
    }

    /// Record a tool call in the per-API-key aggregate (with LRU eviction).
    pub fn record_key_tool_call(&self, api_key_id: &str, tool_name: &str) {
        let mut entry = self
            .key_usage
            .entry(api_key_id.to_string())
            .or_insert_with(KeyUsageState::new);
        entry.value_mut().record(tool_name);
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn record_and_get_session_usage() {
        let tracker = UsageTracker::new(
            Arc::new(DashMap::new()),
            Arc::new(DashMap::new()),
            Arc::new(DashMap::new()),
        );

        tracker.record_tool_call("s1", "get_weather");
        tracker.record_tool_call("s1", "get_weather");
        tracker.record_tool_call("s1", "search");

        let usage = tracker.get_session_usage("s1");
        assert_eq!(usage.get("get_weather"), Some(&2));
        assert_eq!(usage.get("search"), Some(&1));
        assert_eq!(usage.get("unknown"), None);
    }

    #[test]
    fn record_and_get_request_count() {
        let tracker = UsageTracker::new(
            Arc::new(DashMap::new()),
            Arc::new(DashMap::new()),
            Arc::new(DashMap::new()),
        );

        tracker.record_request("s1");
        tracker.record_request("s1");
        tracker.record_request("s1");

        assert_eq!(tracker.get_request_count("s1"), 3);
        assert_eq!(tracker.get_request_count("s2"), 0);
    }

    #[test]
    fn key_usage_tracks_across_calls() {
        let tracker = UsageTracker::new(
            Arc::new(DashMap::new()),
            Arc::new(DashMap::new()),
            Arc::new(DashMap::new()),
        );

        tracker.record_key_tool_call("key1", "tool_a");
        tracker.record_key_tool_call("key1", "tool_a");
        tracker.record_key_tool_call("key1", "tool_b");

        assert_eq!(tracker.get_key_usage("key1", "tool_a"), 2);
        assert_eq!(tracker.get_key_usage("key1", "tool_b"), 1);
        assert_eq!(tracker.get_key_usage("key1", "tool_c"), 0);
    }

    #[test]
    fn key_usage_lru_eviction() {
        let key_usage = Arc::new(DashMap::new());
        let tracker = UsageTracker::new(
            Arc::new(DashMap::new()),
            Arc::new(DashMap::new()),
            key_usage.clone(),
        );

        // Insert MAX_KEY_ENTRIES + 5 tools
        for i in 0..(MAX_KEY_ENTRIES + 5) {
            tracker.record_key_tool_call("key1", &format!("tool_{i}"));
        }

        // Should have evicted oldest entries to stay at MAX_KEY_ENTRIES
        let state = key_usage.get("key1").unwrap();
        assert_eq!(state.entries.len(), MAX_KEY_ENTRIES);

        // The first 5 entries (tool_0..tool_4) should have been evicted
        assert_eq!(state.get_count("tool_0"), 0);
        assert_eq!(state.get_count("tool_4"), 0);
        // Later entries should still exist
        assert_eq!(state.get_count(&format!("tool_{}", MAX_KEY_ENTRIES + 4)), 1);
    }

    #[test]
    fn separate_sessions_independent() {
        let tracker = UsageTracker::new(
            Arc::new(DashMap::new()),
            Arc::new(DashMap::new()),
            Arc::new(DashMap::new()),
        );

        tracker.record_tool_call("s1", "tool_a");
        tracker.record_tool_call("s2", "tool_b");

        let usage_s1 = tracker.get_session_usage("s1");
        let usage_s2 = tracker.get_session_usage("s2");

        assert_eq!(usage_s1.get("tool_a"), Some(&1));
        assert_eq!(usage_s1.get("tool_b"), None);
        assert_eq!(usage_s2.get("tool_b"), Some(&1));
        assert_eq!(usage_s2.get("tool_a"), None);
    }
}

// ─── Property-Based Tests ─────────────────────────────────────────────────────

#[cfg(test)]
mod property_tests {
    use super::*;
    use proptest::prelude::*;

    // ─── Strategies ──────────────────────────────────────────────────────────

    /// Generate a vocabulary of tool names (small set to ensure collisions).
    fn tool_name_vocab() -> impl Strategy<Value = Vec<String>> {
        prop::collection::vec("[a-z_]{3,12}", 2..=20usize)
    }

    // ─── Property 7: Usage Tracker Accuracy and Memory Bound ─────────────────
    // Feature: tool-definition-compression
    // **Validates: Requirements 3.1, 3.2, 3.8**

    proptest! {
        #![proptest_config(ProptestConfig::with_cases(64))]

        /// Verify that after recording a random sequence of tool calls,
        /// the reported counts match the expected sum per tool name.
        #[test]
        fn prop_usage_tracker_count_accuracy(
            vocab in tool_name_vocab(),
            seed_calls in prop::collection::vec((0usize..20, 1u32..=5), 1..=100usize),
        ) {
            // Constrain indices to actual vocab size
            let calls: Vec<(usize, u32)> = seed_calls
                .into_iter()
                .map(|(idx, count)| (idx % vocab.len(), count))
                .collect();

            let tracker = UsageTracker::new(
                Arc::new(DashMap::new()),
                Arc::new(DashMap::new()),
                Arc::new(DashMap::new()),
            );

            // Compute expected counts
            let mut expected: HashMap<String, u64> = HashMap::new();
            for &(idx, count) in &calls {
                let name = &vocab[idx];
                for _ in 0..count {
                    tracker.record_tool_call("session_1", name);
                }
                *expected.entry(name.clone()).or_insert(0) += count as u64;
            }

            // Verify counts match
            let usage = tracker.get_session_usage("session_1");
            for (name, exp_count) in &expected {
                let actual = usage.get(name).copied().unwrap_or(0);
                prop_assert_eq!(
                    actual, *exp_count,
                    "Mismatch for tool '{}': expected {} got {}", name, exp_count, actual
                );
            }
        }

        /// Verify the LRU eviction bound is maintained by directly testing
        /// KeyUsageState with entries exceeding MAX_KEY_ENTRIES.
        #[test]
        fn prop_usage_tracker_lru_memory_bound(
            num_extra in 1u32..=100,
        ) {
            let mut state = KeyUsageState::new();
            let total_inserts = MAX_KEY_ENTRIES as u32 + num_extra;

            for i in 0..total_inserts {
                state.record(&format!("tool_{i}"));
            }

            // Assert memory bound: entries.len() <= MAX_KEY_ENTRIES
            prop_assert!(
                state.entries.len() <= MAX_KEY_ENTRIES,
                "LRU bound violated: {} entries (max {})",
                state.entries.len(),
                MAX_KEY_ENTRIES
            );

            // The most recent entries should still exist
            let last_tool = format!("tool_{}", total_inserts - 1);
            prop_assert!(
                state.get_count(&last_tool) > 0,
                "Most recent tool '{}' was evicted unexpectedly", last_tool
            );

            // The oldest entries should have been evicted
            prop_assert_eq!(
                state.get_count("tool_0"), 0,
                "Oldest tool 'tool_0' should have been evicted"
            );
        }
    }
}
