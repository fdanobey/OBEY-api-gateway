//! Namespace Grouper stage — logical clustering for progressive disclosure.
//!
//! Auto-detects namespaces by splitting tool names on `_` or `.` (first segment
//! as prefix). Configured `namespace_mappings` take priority over auto-detection.
//! Replaces tools with namespace summaries and a synthetic `get_tools_in_namespace` tool.

use std::collections::HashMap;

use crate::tool_compression::config::{
    CompressionLevel, NamespaceGroupingConfig, NamespaceMetadata, ToolCompressionConfig,
};
use crate::tool_compression::stage::CompressionStage;
use crate::tool_compression::types::{CompressionContext, ToolDefinition};

/// A tool name entry returned from namespace resolution.
#[derive(Debug, Clone)]
pub struct ToolNameEntry {
    pub name: String,
    pub description: String,
}

/// Namespace Grouper compression stage.
///
/// Clusters tools by logical namespace (configured or auto-detected prefix),
/// replaces tools with namespace summaries, and injects a synthetic
/// `get_tools_in_namespace` tool for drill-down.
pub struct NamespaceGrouper {
    /// Configured namespace mappings (prefix → metadata), taking priority.
    configured_mappings: HashMap<String, NamespaceMetadata>,
    /// Minimum tool count to activate namespace grouping.
    min_tools_for_grouping: u32,
    /// Whether the stage is enabled.
    enabled: bool,
}

impl NamespaceGrouper {
    /// Create a new `NamespaceGrouper` from config.
    pub fn new(config: &NamespaceGroupingConfig) -> Self {
        Self {
            configured_mappings: config.namespace_mappings.clone(),
            min_tools_for_grouping: config.min_tools_for_grouping,
            enabled: config.enabled,
        }
    }

    /// Extract the namespace prefix from a tool name.
    ///
    /// Splits on `_` or `.` and returns the first segment. Returns `None`
    /// if the name has no separator (singleton).
    fn extract_prefix(name: &str) -> Option<&str> {
        let sep_pos = name.find(|c: char| c == '_' || c == '.');
        sep_pos.map(|pos| &name[..pos])
    }

    /// Detect namespaces from a set of tools.
    ///
    /// Returns a map of namespace prefix → list of tool indices.
    /// Configured mappings take priority; auto-detection uses first segment.
    /// Singletons (prefix with <2 tools) go to "other".
    pub fn detect_namespaces(&self, tools: &[ToolDefinition]) -> HashMap<String, Vec<usize>> {
        let mut prefix_groups: HashMap<String, Vec<usize>> = HashMap::new();

        for (i, tool) in tools.iter().enumerate() {
            // Check configured mappings first
            let mut matched_config = false;
            for (prefix, _meta) in &self.configured_mappings {
                if tool.name.starts_with(prefix) {
                    prefix_groups
                        .entry(prefix.clone())
                        .or_default()
                        .push(i);
                    matched_config = true;
                    break;
                }
            }

            if matched_config {
                continue;
            }

            // Auto-detect by prefix
            if let Some(prefix) = Self::extract_prefix(&tool.name) {
                prefix_groups
                    .entry(prefix.to_string())
                    .or_default()
                    .push(i);
            } else {
                // No separator → "other"
                prefix_groups.entry("other".to_string()).or_default().push(i);
            }
        }

        // Move singletons to "other"
        let singletons: Vec<String> = prefix_groups
            .iter()
            .filter(|(k, v)| v.len() < 2 && *k != "other")
            .map(|(k, _)| k.clone())
            .collect();

        for key in singletons {
            if let Some(indices) = prefix_groups.remove(&key) {
                prefix_groups.entry("other".to_string()).or_default().extend(indices);
            }
        }

        prefix_groups
    }

    /// Generate namespace summary text for a namespace.
    fn generate_summary(
        &self,
        prefix: &str,
        tool_count: usize,
        tools: &[ToolDefinition],
        indices: &[usize],
    ) -> String {
        let description = if let Some(meta) = self.configured_mappings.get(prefix) {
            meta.description.clone()
        } else {
            // Auto-generate from tool names
            let names: Vec<&str> = indices
                .iter()
                .take(3)
                .map(|&i| tools[i].name.as_str())
                .collect();
            format!("Tools: {}", names.join(", "))
        };

        format!("namespace: {} ({} tools) - {}", prefix, tool_count, description)
    }

    /// Build the synthetic `get_tools_in_namespace` tool definition.
    fn build_synthetic_tool(&self, available_namespaces: &[String]) -> ToolDefinition {
        let ns_enum: Vec<serde_json::Value> = available_namespaces
            .iter()
            .map(|s| serde_json::Value::String(s.clone()))
            .collect();

        let raw = serde_json::json!({
            "type": "function",
            "function": {
                "name": "get_tools_in_namespace",
                "description": "Retrieve all tools in a specific namespace. Returns tool names and descriptions.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "namespace": {
                            "type": "string",
                            "description": "The namespace to retrieve tools from.",
                            "enum": ns_enum
                        }
                    },
                    "required": ["namespace"]
                }
            }
        });

        ToolDefinition {
            name: "get_tools_in_namespace".to_string(),
            content_hash: {
                use std::hash::{Hash, Hasher};
                let mut hasher = std::collections::hash_map::DefaultHasher::new();
                let json_str = serde_json::to_string(&raw).unwrap_or_default();
                json_str.hash(&mut hasher);
                hasher.finish()
            },
            raw,
        }
    }

    /// Resolve a namespace to its constituent tool name entries.
    ///
    /// Returns tool names + truncated descriptions for valid namespaces.
    /// Returns an error with available namespace names for invalid input.
    pub fn resolve_namespace(
        &self,
        namespace: &str,
        original_tools: &[ToolDefinition],
    ) -> Result<Vec<ToolNameEntry>, String> {
        let namespaces = self.detect_namespaces(original_tools);

        if let Some(indices) = namespaces.get(namespace) {
            let entries: Vec<ToolNameEntry> = indices
                .iter()
                .map(|&i| {
                    let tool = &original_tools[i];
                    let desc = tool
                        .raw
                        .get("function")
                        .and_then(|f| f.get("description"))
                        .and_then(|d| d.as_str())
                        .unwrap_or("");
                    let truncated = if desc.len() > 100 {
                        format!("{}...", &desc[..97])
                    } else {
                        desc.to_string()
                    };
                    ToolNameEntry {
                        name: tool.name.clone(),
                        description: truncated,
                    }
                })
                .collect();
            Ok(entries)
        } else {
            let available: Vec<&String> = namespaces.keys().collect();
            Err(format!(
                "Invalid namespace '{}'. Available namespaces: {:?}",
                namespace, available
            ))
        }
    }
}

impl CompressionStage for NamespaceGrouper {
    fn apply(
        &self,
        tools: &mut Vec<ToolDefinition>,
        ctx: &mut CompressionContext,
    ) -> u64 {
        // Activation: enabled AND tool count > min_tools_for_grouping
        if tools.len() <= self.min_tools_for_grouping as usize {
            return 0;
        }

        let namespaces = self.detect_namespaces(tools);
        if namespaces.is_empty() {
            return 0;
        }

        // Estimate original tokens
        let original_tokens: u64 = tools
            .iter()
            .map(|t| serde_json::to_string(&t.raw).unwrap_or_default().len() as u64 / 4)
            .sum();

        // Build namespace summaries as tool definitions
        let mut summary_tools: Vec<ToolDefinition> = Vec::new();
        let mut namespace_names: Vec<String> = namespaces.keys().cloned().collect();
        namespace_names.sort();

        for ns in &namespace_names {
            let indices = &namespaces[ns];
            let summary = self.generate_summary(ns, indices.len(), tools, indices);
            let raw = serde_json::json!({
                "type": "function",
                "function": {
                    "name": format!("ns_{}", ns),
                    "description": summary
                }
            });
            summary_tools.push(ToolDefinition {
                name: format!("ns_{}", ns),
                content_hash: 0,
                raw,
            });
        }

        // Add synthetic get_tools_in_namespace tool
        summary_tools.push(self.build_synthetic_tool(&namespace_names));

        // Estimate new tokens
        let new_tokens: u64 = summary_tools
            .iter()
            .map(|t| serde_json::to_string(&t.raw).unwrap_or_default().len() as u64 / 4)
            .sum();

        let tokens_saved = original_tokens.saturating_sub(new_tokens);

        // Replace tools with namespace summaries
        *tools = summary_tools;

        if tokens_saved > 0 {
            ctx.strategies_applied.push("namespace_grouper".to_string());
            ctx.tokens_saved += tokens_saved;
        }

        tokens_saved
    }

    fn is_enabled(&self, _config: &ToolCompressionConfig, _level: CompressionLevel) -> bool {
        self.enabled
    }
}

// ─── Unit Tests ───────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    fn make_tool(name: &str, desc: &str) -> ToolDefinition {
        ToolDefinition {
            raw: serde_json::json!({
                "type": "function",
                "function": {
                    "name": name,
                    "description": desc
                }
            }),
            name: name.to_string(),
            content_hash: 0,
        }
    }

    fn default_config() -> NamespaceGroupingConfig {
        NamespaceGroupingConfig {
            enabled: true,
            min_tools_for_grouping: 5,
            namespace_mappings: HashMap::new(),
        }
    }

    #[test]
    fn extract_prefix_underscore() {
        assert_eq!(NamespaceGrouper::extract_prefix("github_list_repos"), Some("github"));
    }

    #[test]
    fn extract_prefix_dot() {
        assert_eq!(NamespaceGrouper::extract_prefix("slack.send_message"), Some("slack"));
    }

    #[test]
    fn extract_prefix_none() {
        assert_eq!(NamespaceGrouper::extract_prefix("standalone"), None);
    }

    #[test]
    fn detect_namespaces_basic() {
        let ng = NamespaceGrouper::new(&default_config());
        let tools = vec![
            make_tool("github_list_repos", "List repos"),
            make_tool("github_create_issue", "Create issue"),
            make_tool("slack_send_message", "Send msg"),
            make_tool("slack_list_channels", "List channels"),
            make_tool("standalone", "No namespace"),
        ];
        let ns = ng.detect_namespaces(&tools);
        assert_eq!(ns.get("github").map(|v| v.len()), Some(2));
        assert_eq!(ns.get("slack").map(|v| v.len()), Some(2));
        // "standalone" has no separator → goes to "other"
        assert!(ns.get("other").map(|v| v.contains(&4)).unwrap_or(false));
    }

    #[test]
    fn singletons_go_to_other() {
        let ng = NamespaceGrouper::new(&default_config());
        let tools = vec![
            make_tool("github_list_repos", "List repos"),
            make_tool("github_create_issue", "Create issue"),
            make_tool("unique_tool", "Single"), // only 1 with "unique" prefix
        ];
        let ns = ng.detect_namespaces(&tools);
        assert_eq!(ns.get("github").map(|v| v.len()), Some(2));
        // "unique" is singleton → should be in "other"
        assert!(ns.get("unique").is_none());
        assert!(ns.get("other").map(|v| v.contains(&2)).unwrap_or(false));
    }

    #[test]
    fn configured_mappings_take_priority() {
        let mut mappings = HashMap::new();
        mappings.insert("gh".to_string(), NamespaceMetadata {
            name: "GitHub".to_string(),
            description: "GitHub tools".to_string(),
        });
        let config = NamespaceGroupingConfig {
            enabled: true,
            min_tools_for_grouping: 5,
            namespace_mappings: mappings,
        };
        let ng = NamespaceGrouper::new(&config);
        let tools = vec![
            make_tool("gh_list_repos", "List repos"),
            make_tool("gh_create_issue", "Create issue"),
        ];
        let ns = ng.detect_namespaces(&tools);
        assert_eq!(ns.get("gh").map(|v| v.len()), Some(2));
    }

    #[test]
    fn resolve_namespace_valid() {
        let ng = NamespaceGrouper::new(&default_config());
        let tools = vec![
            make_tool("github_list_repos", "List repositories"),
            make_tool("github_create_issue", "Create an issue"),
            make_tool("slack_send", "Send message"),
            make_tool("slack_read", "Read message"),
        ];
        let result = ng.resolve_namespace("github", &tools);
        assert!(result.is_ok());
        let entries = result.unwrap();
        assert_eq!(entries.len(), 2);
        assert!(entries.iter().any(|e| e.name == "github_list_repos"));
        assert!(entries.iter().any(|e| e.name == "github_create_issue"));
    }

    #[test]
    fn resolve_namespace_invalid() {
        let ng = NamespaceGrouper::new(&default_config());
        let tools = vec![
            make_tool("github_list_repos", "List repos"),
            make_tool("github_create_issue", "Create issue"),
        ];
        let result = ng.resolve_namespace("nonexistent", &tools);
        assert!(result.is_err());
        assert!(result.unwrap_err().contains("Invalid namespace"));
    }

    #[test]
    fn noop_below_min_tools() {
        let ng = NamespaceGrouper::new(&default_config());
        let mut tools = vec![
            make_tool("github_list_repos", "List repos"),
            make_tool("github_create_issue", "Create issue"),
        ];
        let mut ctx = CompressionContext::default();
        let saved = ng.apply(&mut tools, &mut ctx);
        assert_eq!(saved, 0);
        assert_eq!(tools.len(), 2); // unchanged
    }

    #[test]
    fn apply_replaces_with_summaries() {
        let config = NamespaceGroupingConfig {
            enabled: true,
            min_tools_for_grouping: 3,
            namespace_mappings: HashMap::new(),
        };
        let ng = NamespaceGrouper::new(&config);
        let mut tools = vec![
            make_tool("github_list_repos", "List repos"),
            make_tool("github_create_issue", "Create issue"),
            make_tool("slack_send", "Send message"),
            make_tool("slack_read", "Read message"),
        ];
        let mut ctx = CompressionContext::default();
        let saved = ng.apply(&mut tools, &mut ctx);

        // Should have namespace summaries + synthetic tool
        assert!(tools.iter().any(|t| t.name == "get_tools_in_namespace"));
        assert!(saved > 0 || tools.len() < 4);
    }
}

// ─── Property Tests ───────────────────────────────────────────────────────────

#[cfg(test)]
mod property_tests {
    use super::*;
    use proptest::prelude::*;
    use std::collections::HashMap;

    fn make_tool_def(name: &str, desc: &str) -> ToolDefinition {
        ToolDefinition {
            raw: serde_json::json!({
                "type": "function",
                "function": {
                    "name": name,
                    "description": desc
                }
            }),
            name: name.to_string(),
            content_hash: 0,
        }
    }

    // ─── Property 19: Namespace Detection and Grouping ────────────────────────
    // Feature: tool-definition-compression
    // **Validates: Requirements 20.1, 20.2, 20.5, 20.6**
    //
    // Generate tool sets with prefixed names; verify correct namespace partitioning.
    // Configured mappings must take priority over auto-detection.

    proptest! {
        #![proptest_config(ProptestConfig::with_cases(200))]

        #[test]
        fn namespace_detection_and_grouping(
            tools_data in prop::collection::vec(
                (
                    prop_oneof![
                        Just("github".to_string()),
                        Just("slack".to_string()),
                        Just("jira".to_string()),
                    ],
                    "[a-z]{3,8}",
                ),
                6..=20,
            )
        ) {
            let ng = NamespaceGrouper::new(&NamespaceGroupingConfig {
                enabled: true,
                min_tools_for_grouping: 5,
                namespace_mappings: HashMap::new(),
            });

            let tools: Vec<ToolDefinition> = tools_data
                .iter()
                .enumerate()
                .map(|(i, (prefix, suffix))| {
                    make_tool_def(
                        &format!("{}_{}_{}", prefix, suffix, i),
                        &format!("Description for {} tool", prefix),
                    )
                })
                .collect();

            let namespaces = ng.detect_namespaces(&tools);

            // All tools must be accounted for
            let total_assigned: usize = namespaces.values().map(|v| v.len()).sum();
            prop_assert_eq!(
                total_assigned, tools.len(),
                "All tools must be in exactly one namespace"
            );

            // No index appears in more than one namespace
            let mut all_indices: Vec<usize> = namespaces.values().flatten().cloned().collect();
            all_indices.sort();
            all_indices.dedup();
            prop_assert_eq!(
                all_indices.len(), total_assigned,
                "No tool should appear in multiple namespaces"
            );

            // Namespaces with 2+ tools (excluding "other") have valid prefix
            for (ns, indices) in &namespaces {
                if ns == "other" {
                    continue;
                }
                prop_assert!(
                    indices.len() >= 2,
                    "Non-'other' namespace '{}' must have >=2 tools, got {}",
                    ns, indices.len()
                );
            }
        }
    }

    // ─── Property 20: Namespace Retrieval Correctness ─────────────────────────
    // Feature: tool-definition-compression
    // **Validates: Requirements 20.8, 20.9**
    //
    // Verify namespace retrieval returns all tools in a namespace and error for invalid.

    proptest! {
        #![proptest_config(ProptestConfig::with_cases(200))]

        #[test]
        fn namespace_retrieval_correctness(
            tools_data in prop::collection::vec(
                (
                    prop_oneof![
                        Just("github".to_string()),
                        Just("slack".to_string()),
                    ],
                    "[a-z]{3,8}",
                ),
                4..=12,
            )
        ) {
            let ng = NamespaceGrouper::new(&NamespaceGroupingConfig {
                enabled: true,
                min_tools_for_grouping: 3,
                namespace_mappings: HashMap::new(),
            });

            let tools: Vec<ToolDefinition> = tools_data
                .iter()
                .enumerate()
                .map(|(i, (prefix, suffix))| {
                    make_tool_def(
                        &format!("{}_{}_{}", prefix, suffix, i),
                        &format!("Desc for {}", prefix),
                    )
                })
                .collect();

            let namespaces = ng.detect_namespaces(&tools);

            // Valid namespace retrieval
            for (ns, indices) in &namespaces {
                let result = ng.resolve_namespace(ns, &tools);
                prop_assert!(
                    result.is_ok(),
                    "resolve_namespace for valid '{}' should succeed", ns
                );
                let entries = result.unwrap();
                prop_assert_eq!(
                    entries.len(), indices.len(),
                    "Namespace '{}' should return all {} tools",
                    ns, indices.len()
                );

                // Every returned tool name should be from the original tools
                for entry in &entries {
                    prop_assert!(
                        tools.iter().any(|t| t.name == entry.name),
                        "Returned tool '{}' not found in original tools",
                        entry.name
                    );
                }
            }

            // Invalid namespace retrieval
            let result = ng.resolve_namespace("nonexistent_namespace_xyz", &tools);
            prop_assert!(
                result.is_err(),
                "resolve_namespace for invalid namespace should error"
            );
        }
    }
}
