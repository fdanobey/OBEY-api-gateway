//! Cache-Aware Placement Optimizer stage.
//!
//! Reorders tool definitions to maximize prefix cache hits by placing stable
//! (unchanged) tools before new/modified tools. Stable tools preserve their
//! previous ordering; new tools preserve their relative input order.
//!
//! On the first request in a session (no previous hashes), this stage is a
//! passthrough. After processing, it stores the current hash vector in the
//! context for the middleware to persist to session state.

use crate::tool_compression::config::{CompressionLevel, ToolCompressionConfig};
use crate::tool_compression::stage::CompressionStage;
use crate::tool_compression::types::{CompressionContext, ToolDefinition};

/// Maximum number of tool hashes stored per session (8 bytes × 200 = 1600 bytes).
const MAX_HASHES_PER_SESSION: usize = 200;

/// Cache-aware placement optimizer.
///
/// Partitions tools into stable (hash matches previous request) and new/modified,
/// then emits stable tools first (preserving their previous order) followed by
/// new tools (preserving their relative input order). This maximizes prefix
/// cache hits for providers that support prompt caching.
pub struct CachePlacementOptimizer;

impl CompressionStage for CachePlacementOptimizer {
    fn apply(
        &self,
        tools: &mut Vec<ToolDefinition>,
        ctx: &mut CompressionContext,
    ) -> u64 {
        // First request in session: passthrough, just store hashes for next time.
        let previous_hashes = match &ctx.previous_hashes {
            Some(hashes) if !hashes.is_empty() => hashes,
            _ => {
                // Store current hashes for next request (capped at max).
                ctx.previous_hashes = Some(
                    tools
                        .iter()
                        .take(MAX_HASHES_PER_SESSION)
                        .map(|t| t.content_hash)
                        .collect(),
                );
                return 0;
            }
        };

        // Build a position map from hash → index in previous_hashes for ordering.
        // If a hash appears multiple times, first occurrence wins.
        let prev_order: std::collections::HashMap<u64, usize> = previous_hashes
            .iter()
            .enumerate()
            .fold(std::collections::HashMap::new(), |mut map, (i, &h)| {
                map.entry(h).or_insert(i);
                map
            });

        // Partition: stable tools (hash in previous set) vs new/modified.
        let mut stable: Vec<(usize, ToolDefinition)> = Vec::new();
        let mut new_tools: Vec<ToolDefinition> = Vec::new();

        for tool in tools.drain(..) {
            if let Some(&prev_idx) = prev_order.get(&tool.content_hash) {
                stable.push((prev_idx, tool));
            } else {
                new_tools.push(tool);
            }
        }

        // Sort stable tools by their previous ordering position.
        stable.sort_by_key(|(prev_idx, _)| *prev_idx);

        // Reassemble: stable first (previous order), then new (relative input order).
        let reordered_count = stable.len();
        for (_, tool) in stable {
            tools.push(tool);
        }
        for tool in new_tools {
            tools.push(tool);
        }

        // Store current hash vector for next request (capped).
        ctx.previous_hashes = Some(
            tools
                .iter()
                .take(MAX_HASHES_PER_SESSION)
                .map(|t| t.content_hash)
                .collect(),
        );

        // Track strategy application if reordering actually happened.
        if reordered_count > 0 && reordered_count < tools.len() {
            ctx.strategies_applied
                .push("cache_placement_optimizer".to_string());
        }

        // This stage does not directly save tokens — it optimizes cache hits.
        // Return 0 as token savings (cache benefit is measured elsewhere).
        0
    }

    fn is_enabled(&self, config: &ToolCompressionConfig, level: CompressionLevel) -> bool {
        config.cache_placement
            && matches!(level, CompressionLevel::High | CompressionLevel::Max)
    }
}

// ─── Tests ────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;
    use crate::tool_compression::types::ProviderCaps;
    use serde_json::json;

    fn make_tool(name: &str, hash: u64) -> ToolDefinition {
        ToolDefinition {
            raw: json!({
                "type": "function",
                "function": { "name": name, "parameters": {} }
            }),
            name: name.to_string(),
            content_hash: hash,
        }
    }

    fn default_ctx_with_hashes(previous: Option<Vec<u64>>) -> CompressionContext {
        CompressionContext {
            level: CompressionLevel::High,
            provider_caps: ProviderCaps::conservative(),
            previous_hashes: previous,
            ..Default::default()
        }
    }

    #[test]
    fn first_request_passthrough() {
        let mut tools = vec![
            make_tool("a", 100),
            make_tool("b", 200),
            make_tool("c", 300),
        ];
        let mut ctx = default_ctx_with_hashes(None);

        let stage = CachePlacementOptimizer;
        let saved = stage.apply(&mut tools, &mut ctx);

        assert_eq!(saved, 0);
        // Order unchanged
        assert_eq!(tools[0].name, "a");
        assert_eq!(tools[1].name, "b");
        assert_eq!(tools[2].name, "c");
        // Hashes stored for next request
        assert_eq!(ctx.previous_hashes, Some(vec![100, 200, 300]));
        // No strategy applied on first request
        assert!(!ctx.strategies_applied.contains(&"cache_placement_optimizer".to_string()));
    }

    #[test]
    fn stable_tools_placed_first_in_previous_order() {
        // Previous request had: [a=100, b=200, c=300]
        // Current request has: [d=400, b=200, a=100] (new tool d, reordered a/b)
        let mut tools = vec![
            make_tool("d", 400),
            make_tool("b", 200),
            make_tool("a", 100),
        ];
        let mut ctx = default_ctx_with_hashes(Some(vec![100, 200, 300]));

        let stage = CachePlacementOptimizer;
        let saved = stage.apply(&mut tools, &mut ctx);

        assert_eq!(saved, 0);
        // Stable tools first in previous order: a(idx 0), b(idx 1)
        // Then new tools: d
        assert_eq!(tools[0].name, "a");
        assert_eq!(tools[1].name, "b");
        assert_eq!(tools[2].name, "d");
        assert!(ctx.strategies_applied.contains(&"cache_placement_optimizer".to_string()));
    }

    #[test]
    fn all_stable_no_reordering_marker() {
        // All tools match previous hashes — stable only, no new tools
        let mut tools = vec![
            make_tool("a", 100),
            make_tool("b", 200),
        ];
        let mut ctx = default_ctx_with_hashes(Some(vec![100, 200]));

        let stage = CachePlacementOptimizer;
        stage.apply(&mut tools, &mut ctx);

        // All tools are stable, none are new — no "reordering" happened
        assert!(!ctx.strategies_applied.contains(&"cache_placement_optimizer".to_string()));
    }

    #[test]
    fn all_new_tools_preserve_relative_order() {
        // Previous had [x=999], current has entirely different tools
        let mut tools = vec![
            make_tool("a", 100),
            make_tool("b", 200),
            make_tool("c", 300),
        ];
        let mut ctx = default_ctx_with_hashes(Some(vec![999]));

        let stage = CachePlacementOptimizer;
        stage.apply(&mut tools, &mut ctx);

        // No stable tools, so all are "new" — relative order preserved
        assert_eq!(tools[0].name, "a");
        assert_eq!(tools[1].name, "b");
        assert_eq!(tools[2].name, "c");
    }

    #[test]
    fn modified_tool_treated_as_new() {
        // Tool "b" changed hash from 200 to 201 — treated as new
        let mut tools = vec![
            make_tool("a", 100),
            make_tool("b", 201), // modified
            make_tool("c", 300),
        ];
        let mut ctx = default_ctx_with_hashes(Some(vec![100, 200, 300]));

        let stage = CachePlacementOptimizer;
        stage.apply(&mut tools, &mut ctx);

        // Stable: a(idx 0), c(idx 2) in previous order. New: b(modified)
        assert_eq!(tools[0].name, "a");
        assert_eq!(tools[1].name, "c");
        assert_eq!(tools[2].name, "b");
    }

    #[test]
    fn hashes_capped_at_max() {
        // Create more tools than MAX_HASHES_PER_SESSION
        let mut tools: Vec<ToolDefinition> = (0..250)
            .map(|i| make_tool(&format!("t{}", i), i as u64))
            .collect();
        let mut ctx = default_ctx_with_hashes(None);

        let stage = CachePlacementOptimizer;
        stage.apply(&mut tools, &mut ctx);

        // Should only store MAX_HASHES_PER_SESSION hashes
        let stored = ctx.previous_hashes.unwrap();
        assert_eq!(stored.len(), MAX_HASHES_PER_SESSION);
    }

    #[test]
    fn is_enabled_high_and_max_with_cache_placement_config() {
        let stage = CachePlacementOptimizer;
        let mut config = ToolCompressionConfig::default();
        config.cache_placement = true;

        assert!(!stage.is_enabled(&config, CompressionLevel::Low));
        assert!(!stage.is_enabled(&config, CompressionLevel::Medium));
        assert!(stage.is_enabled(&config, CompressionLevel::High));
        assert!(stage.is_enabled(&config, CompressionLevel::Max));
    }

    #[test]
    fn is_disabled_when_config_cache_placement_false() {
        let stage = CachePlacementOptimizer;
        let mut config = ToolCompressionConfig::default();
        config.cache_placement = false;

        assert!(!stage.is_enabled(&config, CompressionLevel::High));
        assert!(!stage.is_enabled(&config, CompressionLevel::Max));
    }

    #[test]
    fn empty_previous_hashes_treated_as_first_request() {
        let mut tools = vec![make_tool("a", 100)];
        let mut ctx = default_ctx_with_hashes(Some(vec![]));

        let stage = CachePlacementOptimizer;
        stage.apply(&mut tools, &mut ctx);

        // Empty vec treated same as None — passthrough
        assert_eq!(tools[0].name, "a");
        assert_eq!(ctx.previous_hashes, Some(vec![100]));
    }
}

// ─── Property Tests ───────────────────────────────────────────────────────────

#[cfg(test)]
mod property_tests {
    use super::*;
    use crate::tool_compression::types::ProviderCaps;
    use proptest::prelude::*;
    use serde_json::json;
    use std::collections::HashSet;

    // ─── Strategies ──────────────────────────────────────────────────────────

    fn make_tool(name: &str, hash: u64) -> ToolDefinition {
        ToolDefinition {
            raw: json!({
                "type": "function",
                "function": { "name": name, "parameters": {} }
            }),
            name: name.to_string(),
            content_hash: hash,
        }
    }

    /// Generate a vector of (name, hash) pairs representing a "previous" tool set (5-10 tools).
    fn previous_tool_set() -> impl Strategy<Value = Vec<(String, u64)>> {
        prop::collection::vec(
            ("[a-z]{2,6}", 1u64..10000),
            5..=10usize,
        )
        .prop_map(|entries| {
            // Ensure unique hashes by combining index
            entries
                .into_iter()
                .enumerate()
                .map(|(i, (name, h))| (format!("prev_{}_{}", name, i), h * 1000 + i as u64))
                .collect()
        })
    }

    /// Generate a "current" tool set that partially overlaps with previous.
    /// Returns (previous_tools, current_tools) where current has some stable and some new.
    fn consecutive_request_pair() -> impl Strategy<Value = (Vec<(String, u64)>, Vec<(String, u64)>)>
    {
        previous_tool_set().prop_flat_map(|prev| {
            let prev_len = prev.len();
            let prev_clone = prev.clone();
            // Choose how many tools to keep stable (1 to prev_len-1, ensuring at least 1 new)
            (1..prev_len).prop_flat_map(move |stable_count| {
                let prev_inner = prev_clone.clone();
                // Generate 1-5 new tools
                prop::collection::vec(
                    ("[a-z]{2,6}", 50000u64..99999),
                    1..=5usize,
                )
                .prop_map(move |new_entries| {
                    // Take `stable_count` tools from previous (keep their hashes)
                    let stable: Vec<(String, u64)> = prev_inner[..stable_count].to_vec();

                    // Create new tools with unique names/hashes
                    let new_tools: Vec<(String, u64)> = new_entries
                        .into_iter()
                        .enumerate()
                        .map(|(i, (name, h))| (format!("new_{}_{}", name, i), h * 100 + i as u64))
                        .collect();

                    // Current request: mix of new and stable in arbitrary order
                    // Interleave: new first, then stable (worst case for cache — optimizer should fix)
                    let mut current = new_tools.clone();
                    current.extend(stable);

                    (prev_inner.clone(), current)
                })
            })
        })
    }

    // ─── Property 9: Cache Placement Stability Ordering ──────────────────────
    // Feature: tool-definition-compression
    // **Validates: Requirements 6.1, 6.3, 6.4**
    //
    // Strategy:
    // 1. Generate a "previous" set of tools (5-10) with unique hashes
    // 2. Generate a "current" set that overlaps partially (some stable, some new)
    // 3. Apply CachePlacementOptimizer with previous hashes in context
    // 4. Verify:
    //    - All stable tools appear before all new tools in the output
    //    - Stable tools maintain their relative order from the previous request
    //    - New tools maintain their relative order from the current input
    //    - All tools from the input are present (no tools lost or duplicated)

    proptest! {
        #![proptest_config(ProptestConfig::with_cases(200))]

        #[test]
        fn cache_placement_stability_ordering(
            (prev_set, current_set) in consecutive_request_pair()
        ) {
            // Build previous hashes in order
            let previous_hashes: Vec<u64> = prev_set.iter().map(|(_, h)| *h).collect();

            // Build current tools
            let mut tools: Vec<ToolDefinition> = current_set
                .iter()
                .map(|(name, hash)| make_tool(name, *hash))
                .collect();

            let original_tools = tools.clone();
            let input_count = tools.len();

            // Build context with previous hashes
            let mut ctx = CompressionContext {
                level: CompressionLevel::High,
                provider_caps: ProviderCaps::conservative(),
                previous_hashes: Some(previous_hashes.clone()),
                ..Default::default()
            };

            // Apply the optimizer
            let stage = CachePlacementOptimizer;
            stage.apply(&mut tools, &mut ctx);

            // ── Assertion 1: No tools lost or duplicated ──
            prop_assert_eq!(
                tools.len(),
                input_count,
                "Output must contain exactly the same number of tools as input"
            );

            let output_names: Vec<&str> = tools.iter().map(|t| t.name.as_str()).collect();
            let input_names: HashSet<&str> = original_tools.iter().map(|t| t.name.as_str()).collect();
            let output_names_set: HashSet<&str> = output_names.iter().copied().collect();
            prop_assert_eq!(
                input_names,
                output_names_set,
                "Output must contain exactly the same tool names as input (no loss/duplication)"
            );

            // ── Classify output into stable and new ──
            let prev_hash_set: HashSet<u64> = previous_hashes.iter().copied().collect();
            let mut last_stable_idx: Option<usize> = None;
            let mut first_new_idx: Option<usize> = None;

            for (i, tool) in tools.iter().enumerate() {
                if prev_hash_set.contains(&tool.content_hash) {
                    last_stable_idx = Some(i);
                } else if first_new_idx.is_none() {
                    first_new_idx = Some(i);
                }
            }

            // ── Assertion 2: All stable tools before all new tools ──
            if let (Some(last_stable), Some(first_new)) = (last_stable_idx, first_new_idx) {
                prop_assert!(
                    last_stable < first_new,
                    "All stable tools must appear before all new tools. Last stable idx: {}, first new idx: {}",
                    last_stable,
                    first_new
                );
            }

            // ── Assertion 3: Stable tools preserve relative order from previous request ──
            let stable_tools: Vec<&ToolDefinition> = tools
                .iter()
                .filter(|t| prev_hash_set.contains(&t.content_hash))
                .collect();

            // Build a position map for previous hashes
            let prev_position: std::collections::HashMap<u64, usize> = previous_hashes
                .iter()
                .enumerate()
                .map(|(i, &h)| (h, i))
                .collect();

            for window in stable_tools.windows(2) {
                let pos_a = prev_position.get(&window[0].content_hash).unwrap();
                let pos_b = prev_position.get(&window[1].content_hash).unwrap();
                prop_assert!(
                    pos_a < pos_b,
                    "Stable tools must maintain relative order from previous request. \
                     Tool '{}' (prev pos {}) should come before '{}' (prev pos {})",
                    window[0].name,
                    pos_a,
                    window[1].name,
                    pos_b
                );
            }

            // ── Assertion 4: New tools preserve relative order from current input ──
            let new_tools_output: Vec<&str> = tools
                .iter()
                .filter(|t| !prev_hash_set.contains(&t.content_hash))
                .map(|t| t.name.as_str())
                .collect();

            let new_tools_input: Vec<&str> = original_tools
                .iter()
                .filter(|t| !prev_hash_set.contains(&t.content_hash))
                .map(|t| t.name.as_str())
                .collect();

            prop_assert_eq!(
                new_tools_output,
                new_tools_input,
                "New tools must preserve their relative order from the current input"
            );
        }
    }
}
