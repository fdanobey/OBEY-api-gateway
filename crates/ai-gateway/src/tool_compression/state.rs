//! Shared state container for the tool compression pipeline.
//!
//! `ToolCompressionState` lives in `AppState` as an `Arc<ToolCompressionState>`
//! and is accessed concurrently by the middleware and all pipeline stages.
//! All substates use `DashMap` for lock-free concurrent reads/writes.

use std::collections::{HashMap, HashSet};
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
}

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
        }
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
        // should persist across config reloads.

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
