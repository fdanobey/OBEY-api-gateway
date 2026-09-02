//! Shared state container for the tool compression pipeline.
//!
//! `ToolCompressionState` lives in `AppState` as an `Arc<ToolCompressionState>`
//! and is accessed concurrently by the middleware and all pipeline stages.
//! All substates use `DashMap` for lock-free concurrent reads/writes.

use std::collections::{HashMap, HashSet};
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::Arc;

use dashmap::DashMap;

use super::stages::feedback_loop::FeedbackLoop;

use super::config::ToolCompressionConfig;
use super::types::ProviderCapabilityMap;
use super::usage::KeyUsageState;

/// Central shared state for the tool compression pipeline.
///
/// Each field represents a logically independent substate accessed by one or
/// more pipeline stages. All maps are keyed by session ID, API key, or tool/
/// namespace name as appropriate.
#[derive(Debug)]
pub struct ToolCompressionState {
    /// Per-session disclosed tool sets (session_id → HashSet<tool_name>).
    /// Used by Progressive_Disclosure_Engine to track which tools have had
    /// their full schemas revealed in a given session.
    pub disclosure_state: DashMap<String, HashSet<String>>,

    /// Per-session disclosed discovery targets (session_id → HashSet<canonical key>),
    /// where keys are `ns:<namespace>` or `tool:<tool_name>`. Tracks which synthetic
    /// drill-downs (`get_tools_in_namespace` / `get_tool_schema` / `ns_*`) have already
    /// been revealed this session so re-drills can be reminded from session cache
    /// instead of silently re-resolved, strengthening multi-turn memory.
    pub disclosure_targets: DashMap<String, HashSet<String>>,

    /// Per-session tool usage tracking (session_id → HashMap<tool_name, count>).
    /// Used by Tool_Pruner to identify unused tools for removal.
    pub session_usage: DashMap<String, HashMap<String, u64>>,

    /// Per-session request counter (session_id → request_count).
    /// Used by Tool_Pruner to compute call-to-request frequency ratios.
    pub session_request_count: DashMap<String, u64>,

    /// Per-API-key aggregate usage stats (api_key → KeyUsageState with LRU eviction).
    /// Used by Tool_Pruner for cross-session pruning decisions.
    pub key_usage: DashMap<String, KeyUsageState>,

    /// Per-session tool content hashes for cache placement (session_id → Vec<u64>).
    /// Used by Cache_Placement_Optimizer to detect stable vs changed tool defs.
    pub placement_state: DashMap<String, Vec<u64>>,

    /// Provider capability map (shared, initialized from config).
    /// Used by Schema_Minifier (nullable collapse) and Auto_Tuner.
    pub provider_caps: ProviderCapabilityMap,

    /// Placeholder for semantic retriever embeddings (populated in task 10.1).
    /// Keyed by tool_name → embedding vector.
    pub semantic_state: DashMap<String, Vec<f32>>,

    /// Feedback loop state per model group (group_name → placeholder for legacy compat).
    pub feedback_state: DashMap<String, ()>,

    /// Shared FeedbackLoop instance used by both the middleware and admin API.
    pub feedback_loop: Arc<FeedbackLoop>,

    /// Pre-computed compressed descriptions (tool_name → compressed_description).
    /// Used by Description_Truncator as a preferred source for truncated text.
    pub description_compressor: DashMap<String, String>,

    /// Namespace grouping state (namespace → Vec<tool_name>).
    /// Used by Namespace_Grouper for logical clustering in progressive disclosure.
    pub namespace_state: DashMap<String, Vec<String>>,

    /// Last-touch tick per tracked session, used for LRU eviction of the
    /// per-session maps above (session_id → tick).
    ///
    /// Those maps are keyed by a client-supplied `x-session-id` header (or a
    /// hashed Authorization bucket), so without a bound every distinct session
    /// ever seen stays resident for the process lifetime — an unbounded,
    /// caller-controlled memory leak. This registry keeps the session set capped
    /// at [`MAX_TRACKED_SESSIONS`].
    session_lru: DashMap<String, u64>,

    /// Monotonic counter supplying LRU ordering for [`Self::session_lru`].
    session_tick: AtomicU64,
}

/// Maximum number of sessions retained across the per-session maps.
///
/// Matches the default `loop_detection.max_sessions` so both per-session caches
/// have the same order-of-magnitude footprint. Not configurable on purpose: it is
/// a memory-safety backstop rather than a tuning knob.
pub const MAX_TRACKED_SESSIONS: usize = 10_000;

/// Sessions removed per eviction pass once the cap is exceeded.
///
/// Evicting a batch amortizes the O(n) scan over many insertions instead of
/// running it on every request once the map is full.
const SESSION_EVICTION_BATCH: usize = 128;

impl ToolCompressionState {
    /// Create a new `ToolCompressionState` from the given config.
    ///
    /// Initializes the `ProviderCapabilityMap` from defaults and merges any
    /// config-provided capability overrides.
    pub fn new(config: &ToolCompressionConfig) -> Self {
        let mut provider_caps = ProviderCapabilityMap::default();
        provider_caps.merge_overrides(&config.provider_overrides);

        let feedback_loop = Arc::new(FeedbackLoop::new(&config.feedback_loop, config.level));

        Self {
            disclosure_state: DashMap::new(),
            disclosure_targets: DashMap::new(),
            session_usage: DashMap::new(),
            session_request_count: DashMap::new(),
            key_usage: DashMap::new(),
            placement_state: DashMap::new(),
            provider_caps,
            semantic_state: DashMap::new(),
            feedback_state: DashMap::new(),
            feedback_loop,
            description_compressor: DashMap::new(),
            namespace_state: DashMap::new(),
            session_lru: DashMap::new(),
            session_tick: AtomicU64::new(0),
        }
    }

    /// Mark `session_id` as recently used, evicting least-recently-used sessions
    /// once more than [`MAX_TRACKED_SESSIONS`] are tracked.
    ///
    /// Call once per request as soon as the session id is resolved, before the
    /// per-session maps are read or written. Sessions are evicted as a unit
    /// across every per-session map so no map can outgrow the registry.
    pub fn touch_session(&self, session_id: &str) {
        let tick = self.session_tick.fetch_add(1, Ordering::Relaxed);
        self.session_lru.insert(session_id.to_string(), tick);

        if self.session_lru.len() > MAX_TRACKED_SESSIONS {
            self.evict_least_recently_used_sessions();
        }
    }

    /// Drop the oldest tracked sessions and all of their per-session state.
    fn evict_least_recently_used_sessions(&self) {
        let excess = self.session_lru.len().saturating_sub(MAX_TRACKED_SESSIONS);
        if excess == 0 {
            return;
        }
        let remove_count = excess.max(SESSION_EVICTION_BATCH);

        let mut by_tick: Vec<(String, u64)> = self
            .session_lru
            .iter()
            .map(|entry| (entry.key().clone(), *entry.value()))
            .collect();
        by_tick.sort_unstable_by_key(|(_, tick)| *tick);

        let evicted = by_tick.len().min(remove_count);
        for (session_id, _) in by_tick.into_iter().take(remove_count) {
            self.forget_session(&session_id);
        }

        tracing::debug!(
            evicted,
            retained = self.session_lru.len(),
            cap = MAX_TRACKED_SESSIONS,
            "Evicted least-recently-used tool compression sessions"
        );
    }

    /// Remove every trace of `session_id` from the per-session maps.
    fn forget_session(&self, session_id: &str) {
        self.disclosure_state.remove(session_id);
        self.disclosure_targets.remove(session_id);
        self.session_usage.remove(session_id);
        self.session_request_count.remove(session_id);
        self.placement_state.remove(session_id);
        self.session_lru.remove(session_id);
    }

    /// Number of sessions currently tracked. Used by tests and diagnostics.
    pub fn tracked_session_count(&self) -> usize {
        self.session_lru.len()
    }

    /// Reset all transient state on config hot-reload.
    ///
    /// Clears feedback loop states (circuit breaker reset), re-initializes
    /// provider capabilities, and clears pre-computed descriptions so they
    /// can be recomputed from new config.
    pub fn reset_on_reload(&self, new_config: &ToolCompressionConfig) {
        // Clear feedback loop states (circuit breaker reset)
        self.feedback_state.clear();
        self.feedback_loop.reset_all();

        // Clear pre-computed descriptions (will be recomputed for modified tools)
        self.description_compressor.clear();

        // Clear semantic state (embeddings will be recomputed for modified tools)
        self.semantic_state.clear();

        // Clear namespace state (groupings may have changed)
        self.namespace_state.clear();

        // Note: disclosure_state, session_usage, key_usage, and placement_state
        // are intentionally NOT cleared — they represent live session data that
        // should persist across config reloads. Their growth is bounded instead
        // by the `session_lru` registry (see `touch_session`), so persisting them
        // across reloads no longer implies unbounded retention.

        // Re-initialize provider capabilities from new config
        // (provider_caps is not behind a lock so we can't replace it directly;
        // individual lookups will use the existing map. A full rebuild would
        // require replacing the entire ToolCompressionState.)
        let _ = new_config; // provider_caps merge handled by caller if needed
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn new_creates_empty_state() {
        let config = ToolCompressionConfig::default();
        let state = ToolCompressionState::new(&config);

        assert!(state.disclosure_state.is_empty());
        assert!(state.disclosure_targets.is_empty());
        assert!(state.session_usage.is_empty());
        assert!(state.session_request_count.is_empty());
        assert!(state.key_usage.is_empty());
        assert!(state.placement_state.is_empty());
        assert!(state.semantic_state.is_empty());
        assert!(state.feedback_state.is_empty());
        assert!(state.description_compressor.is_empty());
        assert!(state.namespace_state.is_empty());
    }

    #[test]
    fn touch_session_bounds_the_per_session_maps() {
        let state = ToolCompressionState::new(&ToolCompressionConfig::default());

        // Simulate distinct caller-supplied session ids beyond the cap, writing
        // into every per-session map the way the pipeline stages do.
        let total = MAX_TRACKED_SESSIONS + SESSION_EVICTION_BATCH + 7;
        for index in 0..total {
            let session_id = format!("session-{index}");
            state.touch_session(&session_id);
            state
                .disclosure_state
                .entry(session_id.clone())
                .or_default()
                .insert("tool_a".to_string());
            state
                .disclosure_targets
                .entry(session_id.clone())
                .or_default()
                .insert("ns:alpha".to_string());
            state
                .session_usage
                .entry(session_id.clone())
                .or_default()
                .insert("tool_a".to_string(), 1);
            state.session_request_count.insert(session_id.clone(), 1);
            state.placement_state.insert(session_id, vec![7_u64]);
        }

        assert!(
            state.tracked_session_count() <= MAX_TRACKED_SESSIONS,
            "session registry must stay within the cap, got {}",
            state.tracked_session_count()
        );
        // Every per-session map must be bounded, not just the registry.
        for (label, len) in [
            ("disclosure_state", state.disclosure_state.len()),
            ("disclosure_targets", state.disclosure_targets.len()),
            ("session_usage", state.session_usage.len()),
            ("session_request_count", state.session_request_count.len()),
            ("placement_state", state.placement_state.len()),
        ] {
            assert!(
                len <= MAX_TRACKED_SESSIONS,
                "{label} grew past the cap: {len}"
            );
        }

        // The oldest session is gone; the newest survives.
        assert!(!state.disclosure_state.contains_key("session-0"));
        assert!(state
            .disclosure_state
            .contains_key(&format!("session-{}", total - 1)));
    }

    #[test]
    fn touching_an_existing_session_refreshes_it_without_growing() {
        let state = ToolCompressionState::new(&ToolCompressionConfig::default());

        state.touch_session("stable");
        state.touch_session("stable");
        state.touch_session("stable");

        assert_eq!(state.tracked_session_count(), 1);
    }

    #[test]
    fn eviction_removes_all_state_for_a_session() {
        let state = ToolCompressionState::new(&ToolCompressionConfig::default());
        state.touch_session("doomed");
        state
            .disclosure_state
            .entry("doomed".to_string())
            .or_default()
            .insert("tool_a".to_string());
        state.session_request_count.insert("doomed".to_string(), 3);
        state.placement_state.insert("doomed".to_string(), vec![1]);

        state.forget_session("doomed");

        assert_eq!(state.tracked_session_count(), 0);
        assert!(!state.disclosure_state.contains_key("doomed"));
        assert!(!state.session_request_count.contains_key("doomed"));
        assert!(!state.placement_state.contains_key("doomed"));
    }

    #[test]
    fn provider_caps_initialized_with_defaults() {
        let config = ToolCompressionConfig::default();
        let state = ToolCompressionState::new(&config);

        // Known providers should have non-conservative defaults
        let openai_caps = state.provider_caps.get("openai");
        assert!(openai_caps.supports_ref);
        assert!(openai_caps.supports_prompt_caching);

        // Unknown provider returns conservative
        let unknown = state.provider_caps.get("unknown_provider");
        assert!(!unknown.supports_ref);
        assert!(!unknown.supports_prompt_caching);
    }
}
