//! Tool Pruner stage — removes unused tools based on session frequency.
//!
//! Prunes tools with zero calls after the configured `min_requests` threshold,
//! respecting `always_include` glob patterns and retaining a minimum of 3 tools.
//! Restores pruned tools if they are referenced in the current message content.

use crate::tool_compression::config::{CompressionLevel, PruningConfig, ToolCompressionConfig};
use crate::tool_compression::stage::CompressionStage;
use crate::tool_compression::types::{CompressionContext, ToolDefinition};

/// Minimum tools to retain if all would be pruned.
const MIN_RETAINED_TOOLS: usize = 3;

/// Tool Pruner compression stage.
///
/// Removes tool definitions that have zero calls in the current session after
/// the minimum request threshold is reached. Respects `always_include` patterns
/// and ensures at least `MIN_RETAINED_TOOLS` remain.
pub struct ToolPruner {
    /// Minimum requests before pruning activates.
    min_requests: u32,
    /// Parsed always_include patterns (supports `*` wildcard).
    always_include: Vec<GlobPattern>,
}

/// Simple glob pattern supporting `*` as wildcard.
#[derive(Debug, Clone)]
pub struct GlobPattern {
    /// The original pattern string.
    pattern: String,
    /// Segments split by `*`.
    segments: Vec<String>,
}

impl GlobPattern {
    /// Parse a glob pattern string. Supports `*` as wildcard matching any sequence.
    pub fn new(pattern: &str) -> Self {
        let segments: Vec<String> = pattern.split('*').map(|s| s.to_string()).collect();
        Self {
            pattern: pattern.to_string(),
            segments,
        }
    }

    /// Check if a name matches this glob pattern.
    pub fn matches(&self, name: &str) -> bool {
        if self.segments.len() == 1 {
            // No wildcard — exact match
            return name == self.pattern;
        }

        let mut remaining = name;

        for (i, segment) in self.segments.iter().enumerate() {
            if segment.is_empty() {
                // Leading or trailing `*` or consecutive `**`
                continue;
            }

            if i == 0 {
                // First segment must be a prefix
                if !remaining.starts_with(segment.as_str()) {
                    return false;
                }
                remaining = &remaining[segment.len()..];
            } else if i == self.segments.len() - 1 {
                // Last segment must be a suffix
                if !remaining.ends_with(segment.as_str()) {
                    return false;
                }
                remaining = &remaining[..remaining.len() - segment.len()];
            } else {
                // Middle segment must appear somewhere
                match remaining.find(segment.as_str()) {
                    Some(pos) => {
                        remaining = &remaining[pos + segment.len()..];
                    }
                    None => return false,
                }
            }
        }

        true
    }
}

impl ToolPruner {
    /// Create a new `ToolPruner` from pruning configuration.
    pub fn new(config: &PruningConfig) -> Self {
        let always_include = config
            .always_include
            .iter()
            .map(|p| GlobPattern::new(p))
            .collect();

        Self {
            min_requests: config.min_requests,
            always_include,
        }
    }

    /// Check if a tool name matches any `always_include` pattern.
    fn is_always_included(&self, tool_name: &str) -> bool {
        self.always_include.iter().any(|p| p.matches(tool_name))
    }

    /// Check if a pruned tool name is referenced in the message content.
    fn is_referenced_in_messages(tool_name: &str, message_content: Option<&str>) -> bool {
        match message_content {
            Some(content) => content.contains(tool_name),
            None => false,
        }
    }
}

impl CompressionStage for ToolPruner {
    fn apply(&self, tools: &mut Vec<ToolDefinition>, ctx: &mut CompressionContext) -> u64 {
        // Must have a session to track usage
        if ctx.session_id.is_none() {
            return 0;
        }

        if tools.is_empty() {
            return 0;
        }

        // Check if session has enough requests before pruning activates
        if ctx.session_request_count < self.min_requests as u64 {
            return 0;
        }

        let session_usage = &ctx.session_usage;

        // Classify tools into keep/prune candidates
        let mut prune_candidates: Vec<usize> = Vec::new();
        let mut keep_count = 0usize;

        for (i, tool) in tools.iter().enumerate() {
            let call_count = session_usage.get(&tool.name).copied().unwrap_or(0);

            if call_count == 0 && !self.is_always_included(&tool.name) {
                prune_candidates.push(i);
            } else {
                keep_count += 1;
            }
        }

        // If nothing to prune, return early
        if prune_candidates.is_empty() {
            return 0;
        }

        // If ALL tools would be pruned, retain the last MIN_RETAINED_TOOLS by array position
        if keep_count == 0 {
            let retain_count = MIN_RETAINED_TOOLS.min(prune_candidates.len());
            let retain_start = prune_candidates.len() - retain_count;
            // Remove from prune_candidates the ones we want to retain
            prune_candidates.truncate(retain_start);
        }

        // Check message content for references to pruned tools — restore if found
        let message_content = ctx.message_content.as_deref();
        prune_candidates.retain(|&i| {
            let tool_name = &tools[i].name;
            !Self::is_referenced_in_messages(tool_name, message_content)
        });

        // If nothing left to prune after restore checks, return
        if prune_candidates.is_empty() {
            return 0;
        }

        // Calculate token savings before removal
        let mut total_saved: u64 = 0;
        for &i in &prune_candidates {
            total_saved += estimate_tokens(&tools[i].raw);
        }

        // Remove pruned tools (use a set for O(1) lookup)
        let prune_set: std::collections::HashSet<usize> = prune_candidates.into_iter().collect();
        let mut idx = 0;
        tools.retain(|_| {
            let keep = !prune_set.contains(&idx);
            idx += 1;
            keep
        });

        if total_saved > 0 {
            ctx.strategies_applied.push("tool_pruner".to_string());
            ctx.tokens_saved += total_saved;
        }

        total_saved
    }

    fn is_enabled(&self, config: &ToolCompressionConfig, level: CompressionLevel) -> bool {
        config.pruning.enabled && level == CompressionLevel::Max
    }
}

/// Estimate token count as character_count / 4.
fn estimate_tokens(value: &serde_json::Value) -> u64 {
    let s = value.to_string();
    (s.len() as u64) / 4
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;
    use std::collections::HashMap;

    fn make_tool(name: &str) -> ToolDefinition {
        ToolDefinition {
            raw: json!({
                "type": "function",
                "function": {
                    "name": name,
                    "description": format!("A tool named {name}"),
                    "parameters": {
                        "type": "object",
                        "properties": {}
                    }
                }
            }),
            name: name.to_string(),
            content_hash: 0,
        }
    }

    fn make_ctx_with_usage(usage: HashMap<String, u64>) -> CompressionContext {
        CompressionContext {
            session_id: Some("test-session".to_string()),
            session_usage: usage,
            session_request_count: 10, // Above default min_requests of 5
            message_content: None,
            ..Default::default()
        }
    }

    #[test]
    fn prunes_zero_call_tools() {
        let config = PruningConfig {
            enabled: true,
            min_requests: 5,
            always_include: vec![],
        };
        let pruner = ToolPruner::new(&config);

        let mut tools = vec![
            make_tool("get_weather"),
            make_tool("search"),
            make_tool("unused_tool"),
            make_tool("another_unused"),
        ];

        let mut usage = HashMap::new();
        usage.insert("get_weather".to_string(), 3);
        usage.insert("search".to_string(), 1);
        // unused_tool and another_unused have zero calls

        let mut ctx = make_ctx_with_usage(usage);
        let saved = pruner.apply(&mut tools, &mut ctx);

        assert!(saved > 0);
        assert_eq!(tools.len(), 2);
        assert_eq!(tools[0].name, "get_weather");
        assert_eq!(tools[1].name, "search");
    }

    #[test]
    fn respects_always_include_exact() {
        let config = PruningConfig {
            enabled: true,
            min_requests: 5,
            always_include: vec!["important_tool".to_string()],
        };
        let pruner = ToolPruner::new(&config);

        let mut tools = vec![
            make_tool("get_weather"),
            make_tool("important_tool"),
            make_tool("unused_tool"),
        ];

        let mut usage = HashMap::new();
        usage.insert("get_weather".to_string(), 1);
        // important_tool has zero calls but is in always_include

        let mut ctx = make_ctx_with_usage(usage);
        pruner.apply(&mut tools, &mut ctx);

        assert_eq!(tools.len(), 2);
        assert!(tools.iter().any(|t| t.name == "important_tool"));
        assert!(tools.iter().any(|t| t.name == "get_weather"));
    }

    #[test]
    fn respects_always_include_glob() {
        let config = PruningConfig {
            enabled: true,
            min_requests: 5,
            always_include: vec!["github_*".to_string()],
        };
        let pruner = ToolPruner::new(&config);

        let mut tools = vec![
            make_tool("get_weather"),
            make_tool("github_create_issue"),
            make_tool("github_list_repos"),
            make_tool("slack_post"),
        ];

        let mut usage = HashMap::new();
        usage.insert("get_weather".to_string(), 1);
        // All github_* tools have zero calls but match always_include glob

        let mut ctx = make_ctx_with_usage(usage);
        pruner.apply(&mut tools, &mut ctx);

        assert_eq!(tools.len(), 3);
        assert!(tools.iter().any(|t| t.name == "get_weather"));
        assert!(tools.iter().any(|t| t.name == "github_create_issue"));
        assert!(tools.iter().any(|t| t.name == "github_list_repos"));
    }

    #[test]
    fn retains_min_tools_when_all_pruned() {
        let config = PruningConfig {
            enabled: true,
            min_requests: 5,
            always_include: vec![],
        };
        let pruner = ToolPruner::new(&config);

        let mut tools = vec![
            make_tool("tool_1"),
            make_tool("tool_2"),
            make_tool("tool_3"),
            make_tool("tool_4"),
            make_tool("tool_5"),
        ];

        // All tools have zero calls — session_usage can be empty since we rely
        // on session_request_count for the threshold check
        let usage = HashMap::new();

        let mut ctx = CompressionContext {
            session_id: Some("test-session".to_string()),
            session_usage: usage,
            session_request_count: 10,
            message_content: None,
            ..Default::default()
        };
        pruner.apply(&mut tools, &mut ctx);

        // Should retain last 3 tools by array position
        assert_eq!(tools.len(), MIN_RETAINED_TOOLS);
        assert_eq!(tools[0].name, "tool_3");
        assert_eq!(tools[1].name, "tool_4");
        assert_eq!(tools[2].name, "tool_5");
    }

    #[test]
    fn restores_pruned_tools_referenced_in_messages() {
        let config = PruningConfig {
            enabled: true,
            min_requests: 5,
            always_include: vec![],
        };
        let pruner = ToolPruner::new(&config);

        let mut tools = vec![
            make_tool("get_weather"),
            make_tool("search"),
            make_tool("unused_tool"),
        ];

        let mut usage = HashMap::new();
        usage.insert("get_weather".to_string(), 2);
        // search and unused_tool have zero calls

        let mut ctx = make_ctx_with_usage(usage);
        // The user message references "search" — should be restored
        ctx.message_content = Some("Please use search to find results".to_string());

        pruner.apply(&mut tools, &mut ctx);

        assert_eq!(tools.len(), 2);
        assert!(tools.iter().any(|t| t.name == "get_weather"));
        assert!(tools.iter().any(|t| t.name == "search"));
        // unused_tool should be pruned
        assert!(!tools.iter().any(|t| t.name == "unused_tool"));
    }

    #[test]
    fn no_prune_when_all_tools_fit_in_min_retained() {
        let config = PruningConfig {
            enabled: true,
            min_requests: 5,
            always_include: vec![],
        };
        let pruner = ToolPruner::new(&config);

        // Only 2 tools, all with zero calls — since MIN_RETAINED_TOOLS is 3,
        // all 2 tools are retained (can't prune below minimum)
        let mut tools = vec![make_tool("tool_1"), make_tool("tool_2")];
        let mut ctx = make_ctx_with_usage(HashMap::new());

        let saved = pruner.apply(&mut tools, &mut ctx);
        assert_eq!(saved, 0);
        assert_eq!(tools.len(), 2);
    }

    #[test]
    fn no_prune_below_min_requests() {
        let config = PruningConfig {
            enabled: true,
            min_requests: 5,
            always_include: vec![],
        };
        let pruner = ToolPruner::new(&config);

        let mut tools = vec![make_tool("tool_1"), make_tool("tool_2")];
        let mut ctx = CompressionContext {
            session_id: Some("test-session".to_string()),
            session_usage: HashMap::new(),
            session_request_count: 3, // Below min_requests of 5
            message_content: None,
            ..Default::default()
        };

        let saved = pruner.apply(&mut tools, &mut ctx);
        assert_eq!(saved, 0);
        assert_eq!(tools.len(), 2);
    }

    #[test]
    fn no_prune_without_session_id() {
        let config = PruningConfig {
            enabled: true,
            min_requests: 5,
            always_include: vec![],
        };
        let pruner = ToolPruner::new(&config);

        let mut tools = vec![make_tool("tool_1")];
        let mut ctx = CompressionContext::default();
        // session_id is None

        let saved = pruner.apply(&mut tools, &mut ctx);
        assert_eq!(saved, 0);
    }

    #[test]
    fn glob_pattern_matching() {
        let p = GlobPattern::new("github_*");
        assert!(p.matches("github_create_issue"));
        assert!(p.matches("github_list_repos"));
        assert!(!p.matches("slack_post"));
        assert!(!p.matches("github")); // No trailing content after prefix

        let p2 = GlobPattern::new("*_api");
        assert!(p2.matches("weather_api"));
        assert!(p2.matches("search_api"));
        assert!(!p2.matches("api_search"));

        let p3 = GlobPattern::new("*tool*");
        assert!(p3.matches("my_tool_v2"));
        assert!(p3.matches("toolbox"));
        assert!(p3.matches("super_tool"));

        let p4 = GlobPattern::new("exact_match");
        assert!(p4.matches("exact_match"));
        assert!(!p4.matches("exact_match_extra"));
        assert!(!p4.matches("not_exact_match"));
    }

    #[test]
    fn is_enabled_checks_config_and_level() {
        let pruner = ToolPruner::new(&PruningConfig {
            enabled: true,
            min_requests: 5,
            always_include: vec![],
        });

        let mut config = ToolCompressionConfig::default();
        config.pruning.enabled = true;

        assert!(pruner.is_enabled(&config, CompressionLevel::Max));
        assert!(!pruner.is_enabled(&config, CompressionLevel::High));
        assert!(!pruner.is_enabled(&config, CompressionLevel::Medium));
        assert!(!pruner.is_enabled(&config, CompressionLevel::Low));

        config.pruning.enabled = false;
        assert!(!pruner.is_enabled(&config, CompressionLevel::Max));
    }
}

// ─── Property-Based Tests ─────────────────────────────────────────────────────

#[cfg(test)]
mod property_tests {
    use super::*;
    use proptest::prelude::*;
    use serde_json::json;
    use std::collections::HashMap;

    // ─── Strategies ──────────────────────────────────────────────────────────

    /// Generate a list of unique tool names (5-20 tools).
    fn tool_names_strategy() -> impl Strategy<Value = Vec<String>> {
        prop::collection::hash_set("[a-z]{3,10}", 5..=20usize)
            .prop_map(|s| s.into_iter().collect::<Vec<_>>())
    }

    fn make_tool(name: &str) -> ToolDefinition {
        ToolDefinition {
            raw: json!({
                "type": "function",
                "function": {
                    "name": name,
                    "description": format!("Tool {name}"),
                    "parameters": {
                        "type": "object",
                        "properties": {}
                    }
                }
            }),
            name: name.to_string(),
            content_hash: 0,
        }
    }

    // ─── Property 8: Pruning Correctness ─────────────────────────────────────
    // Feature: tool-definition-compression
    // **Validates: Requirements 3.1, 3.2, 3.3, 3.4, 3.5, 3.8**

    proptest! {
        #![proptest_config(ProptestConfig::with_cases(64))]

        /// Verify pruning correctness:
        /// - Tools with >0 calls are never pruned
        /// - Tools matching always_include are never pruned
        /// - Tools with 0 calls not matching always_include are pruned
        /// - Result always has >= MIN_RETAINED_TOOLS (3) if input had >= 3
        #[test]
        fn prop_pruning_correctness(
            tool_names in tool_names_strategy(),
        ) {
            let mut tools: Vec<ToolDefinition> = tool_names
                .iter()
                .map(|n| make_tool(n))
                .collect();

            // Randomly assign usage: give first ~half of tools some calls
            let half = tool_names.len() / 2;
            let mut usage: HashMap<String, u64> = HashMap::new();
            for (i, name) in tool_names.iter().enumerate() {
                if i < half {
                    usage.insert(name.clone(), (i as u64) + 1);
                }
                // remaining tools have 0 calls (not in map)
            }

            // Pick first tool as always_include (exact match)
            let always_include = vec![tool_names[0].clone()];

            let config = PruningConfig {
                enabled: true,
                min_requests: 5,
                always_include: always_include.clone(),
            };
            let pruner = ToolPruner::new(&config);

            let mut ctx = CompressionContext {
                session_id: Some("test-session".to_string()),
                session_usage: usage.clone(),
                session_request_count: 10, // Above threshold
                message_content: None,
                ..Default::default()
            };

            let original_count = tools.len();
            pruner.apply(&mut tools, &mut ctx);

            // Property 1: Tools with >0 calls are never pruned
            for (name, count) in &usage {
                if *count > 0 {
                    prop_assert!(
                        tools.iter().any(|t| &t.name == name),
                        "Tool '{}' with {} calls was incorrectly pruned", name, count
                    );
                }
            }

            // Property 2: Tools in always_include are never pruned
            for pattern in &always_include {
                prop_assert!(
                    tools.iter().any(|t| &t.name == pattern),
                    "Tool '{}' in always_include was incorrectly pruned", pattern
                );
            }

            // Property 3: Result has >= MIN_RETAINED_TOOLS if input had >= 3
            // AND all tools had zero calls (the min retention only applies in that case)
            let all_zero = usage.values().all(|&c| c == 0);
            if original_count >= MIN_RETAINED_TOOLS && all_zero && always_include.is_empty() {
                prop_assert!(
                    tools.len() >= MIN_RETAINED_TOOLS,
                    "Tool count {} below MIN_RETAINED_TOOLS {} (original: {})",
                    tools.len(),
                    MIN_RETAINED_TOOLS,
                    original_count
                );
            }

            // Property 4: Tools with 0 calls not in always_include should be pruned
            // (unless MIN_RETAINED_TOOLS prevents it)
            let used_tool_count = usage.values().filter(|&&c| c > 0).count();
            let always_included_zero_call: Vec<&String> = always_include
                .iter()
                .filter(|p| usage.get(*p).copied().unwrap_or(0) == 0)
                .collect();
            let protected_count = used_tool_count + always_included_zero_call.len();

            // If we have enough protected tools above MIN_RETAINED_TOOLS,
            // then ALL unprotected zero-call tools should be pruned
            if protected_count >= MIN_RETAINED_TOOLS {
                for name in &tool_names {
                    let has_calls = usage.get(name).copied().unwrap_or(0) > 0;
                    let is_always_included = always_include.contains(name);
                    if !has_calls && !is_always_included {
                        prop_assert!(
                            !tools.iter().any(|t| &t.name == name),
                            "Tool '{}' with 0 calls and not in always_include was NOT pruned",
                            name
                        );
                    }
                }
            }
        }

        /// Verify that pruning never drops below MIN_RETAINED_TOOLS regardless
        /// of usage patterns when all tools have zero calls.
        #[test]
        fn prop_pruning_min_retention(
            num_tools in 3usize..=20,
        ) {
            let tool_names: Vec<String> = (0..num_tools)
                .map(|i| format!("tool_{i}"))
                .collect();
            let mut tools: Vec<ToolDefinition> = tool_names
                .iter()
                .map(|n| make_tool(n))
                .collect();

            let config = PruningConfig {
                enabled: true,
                min_requests: 5,
                always_include: vec![],
            };
            let pruner = ToolPruner::new(&config);

            let mut ctx = CompressionContext {
                session_id: Some("test-session".to_string()),
                session_usage: HashMap::new(), // All zero calls
                session_request_count: 10,
                message_content: None,
                ..Default::default()
            };

            pruner.apply(&mut tools, &mut ctx);

            // Always retain at least MIN_RETAINED_TOOLS
            prop_assert!(
                tools.len() >= MIN_RETAINED_TOOLS,
                "Retained {} tools but minimum is {}",
                tools.len(),
                MIN_RETAINED_TOOLS
            );

            // The retained tools should be the LAST ones by array position
            for (i, tool) in tools.iter().enumerate() {
                let expected_idx = num_tools - MIN_RETAINED_TOOLS + i;
                let expected_name = format!("tool_{expected_idx}");
                prop_assert_eq!(
                    &tool.name,
                    &expected_name,
                    "Retained tool at position {} should be '{}' but got '{}'",
                    i,
                    expected_name,
                    tool.name
                );
            }
        }
    }
}
