//! Progressive Disclosure Engine stage — two-tier tool listing with on-demand schema.
//!
//! When tool count > 5:
//! 1. Replaces full `tools` array with minimal listing: `{name, description_truncated_100}`
//! 2. Appends synthetic `get_tool_schema` tool definition
//! 3. For already-disclosed tools (tracked in session state), appends full schemas at END
//!
//! When tool count <= 5: bypasses (overhead exceeds savings).

use serde_json::{json, Value};

use crate::tool_compression::config::{CompressionLevel, ToolCompressionConfig};
use crate::tool_compression::stage::CompressionStage;
use crate::tool_compression::types::{CompressionContext, ToolDefinition};

// ─── Constants ────────────────────────────────────────────────────────────────

/// Minimum tool count to activate progressive disclosure.
/// Below this threshold, the overhead of the synthetic tool exceeds savings.
const MIN_TOOLS_FOR_DISCLOSURE: usize = 5;

/// Maximum length for truncated descriptions in the minimal listing.
const MAX_DESCRIPTION_LENGTH: usize = 100;

// ─── Synthetic tool definition ────────────────────────────────────────────────

/// Returns the synthetic `get_tool_schema` tool definition JSON.
fn synthetic_get_tool_schema() -> Value {
    json!({
        "type": "function",
        "function": {
            "name": "get_tool_schema",
            "description": "Retrieve the full parameter schema for a tool by name.",
            "parameters": {
                "type": "object",
                "required": ["tool_name"],
                "properties": {
                    "tool_name": {
                        "type": "string",
                        "description": "The name of the tool to get the schema for."
                    }
                }
            }
        }
    })
}

// ─── ProgressiveDisclosureEngine ──────────────────────────────────────────────

/// Progressive disclosure stage.
///
/// Converts the full tools array into a minimal name+description listing with
/// a synthetic `get_tool_schema` tool for on-demand schema retrieval. Previously
/// disclosed tools (tracked in `ctx.disclosed_tools`) have their full schemas
/// appended at the END of the array for prefix cache hits.
pub struct ProgressiveDisclosureEngine;

impl ProgressiveDisclosureEngine {
    pub fn new() -> Self {
        Self
    }

    /// Create a minimal listing entry for a tool: name + truncated description only.
    fn to_minimal_entry(tool: &ToolDefinition) -> Value {
        let description = extract_description(&tool.raw);
        let truncated = truncate_description(&description, MAX_DESCRIPTION_LENGTH);

        json!({
            "type": "function",
            "function": {
                "name": tool.name,
                "description": truncated
            }
        })
    }
}

/// Extract the function-level description from a tool's raw JSON value.
fn extract_description(raw: &Value) -> String {
    raw.get("function")
        .and_then(|f| f.get("description"))
        .and_then(|d| d.as_str())
        .unwrap_or("")
        .to_string()
}

/// Truncate a description to `max_len` characters, breaking at word boundary
/// when possible and appending "..." if truncated.
fn truncate_description(desc: &str, max_len: usize) -> String {
    if desc.len() <= max_len {
        return desc.to_string();
    }

    // Find the last space before max_len to break at word boundary
    let truncated = &desc[..max_len];
    if let Some(last_space) = truncated.rfind(' ') {
        if last_space > max_len / 2 {
            return format!("{}...", &desc[..last_space]);
        }
    }

    // Fall back to hard truncation
    format!("{}...", &desc[..max_len.saturating_sub(3)])
}

/// Resolve a `get_tool_schema` call against preserved original tools.
/// Returns `Ok(full_schema_json)` for valid tool names, or `Err(error_message)` for invalid ones.
pub fn resolve_get_tool_schema(
    tool_name: &str,
    original_tools: &[ToolDefinition],
) -> Result<Value, String> {
    // Search for the tool by name in the preserved originals
    for tool in original_tools {
        if tool.name == tool_name {
            return Ok(tool.raw.clone());
        }
    }

    // Tool not found — build error with available names
    let available: Vec<&str> = original_tools.iter().map(|t| t.name.as_str()).collect();
    Err(format!(
        "Tool '{}' not found. Available tools: {}",
        tool_name,
        available.join(", ")
    ))
}

impl CompressionStage for ProgressiveDisclosureEngine {
    fn apply(&self, tools: &mut Vec<ToolDefinition>, ctx: &mut CompressionContext) -> u64 {
        // Streaming requests cannot use the synthetic drill-down resolution loop,
        // so injecting get_tool_schema would leak to the client unresolved.
        if ctx.is_streaming {
            return 0;
        }

        // Bypass when tool count <= threshold
        if tools.len() <= MIN_TOOLS_FOR_DISCLOSURE {
            return 0;
        }

        let before_tokens: u64 = tools.iter().map(|t| estimate_tokens(&t.raw)).sum();

        // Build minimal listing entries for ALL tools
        let mut minimal_entries: Vec<ToolDefinition> = tools
            .iter()
            .map(|tool| {
                let minimal_raw = Self::to_minimal_entry(tool);
                let name = tool.name.clone();
                ToolDefinition {
                    raw: minimal_raw,
                    name,
                    content_hash: 0, // Hash not relevant for minimal entries
                }
            })
            .collect();

        // Append the synthetic get_tool_schema tool
        let synthetic_raw = synthetic_get_tool_schema();
        minimal_entries.push(ToolDefinition {
            raw: synthetic_raw,
            name: "get_tool_schema".to_string(),
            content_hash: 0,
        });

        // Collect full schemas for previously-disclosed tools (append at END for cache hits)
        let disclosed_full: Vec<ToolDefinition> = tools
            .iter()
            .filter(|tool| ctx.disclosed_tools.contains(&tool.name))
            .cloned()
            .collect();

        // Append disclosed full schemas at the end
        minimal_entries.extend(disclosed_full);

        // Replace the tools array with our minimal + disclosed listing
        let after_tokens: u64 = minimal_entries
            .iter()
            .map(|t| estimate_tokens(&t.raw))
            .sum();
        *tools = minimal_entries;

        // Record strategy
        ctx.strategies_applied
            .push("progressive_disclosure".to_string());

        before_tokens.saturating_sub(after_tokens)
    }

    fn is_enabled(&self, config: &ToolCompressionConfig, level: CompressionLevel) -> bool {
        // Enabled when progressive_disclosure config is true AND level is High or Max
        config.progressive_disclosure
            && matches!(level, CompressionLevel::High | CompressionLevel::Max)
    }
}

/// Estimate token count from a JSON value (chars / 4 approximation).
fn estimate_tokens(value: &Value) -> u64 {
    let serialized = serde_json::to_string(value).unwrap_or_default();
    (serialized.len() as u64) / 4
}

// ─── Tests ────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;
    use std::collections::HashSet;

    fn make_tool(name: &str, description: &str) -> ToolDefinition {
        let raw = json!({
            "type": "function",
            "function": {
                "name": name,
                "description": description,
                "parameters": {
                    "type": "object",
                    "required": ["query"],
                    "properties": {
                        "query": {
                            "type": "string",
                            "description": "The search query"
                        }
                    }
                }
            }
        });
        ToolDefinition {
            raw,
            name: name.to_string(),
            content_hash: 0,
        }
    }

    fn default_ctx() -> CompressionContext {
        CompressionContext {
            level: CompressionLevel::High,
            ..Default::default()
        }
    }

    #[test]
    fn bypass_when_5_or_fewer_tools() {
        let engine = ProgressiveDisclosureEngine::new();
        let mut tools = vec![
            make_tool("a", "Tool A"),
            make_tool("b", "Tool B"),
            make_tool("c", "Tool C"),
            make_tool("d", "Tool D"),
            make_tool("e", "Tool E"),
        ];
        let mut ctx = default_ctx();

        let saved = engine.apply(&mut tools, &mut ctx);
        assert_eq!(saved, 0);
        assert_eq!(tools.len(), 5); // Unchanged
    }

    #[test]
    fn applies_disclosure_when_more_than_5_tools() {
        let engine = ProgressiveDisclosureEngine::new();
        let mut tools = vec![
            make_tool("tool_1", "First tool description"),
            make_tool("tool_2", "Second tool description"),
            make_tool("tool_3", "Third tool description"),
            make_tool("tool_4", "Fourth tool description"),
            make_tool("tool_5", "Fifth tool description"),
            make_tool("tool_6", "Sixth tool description"),
        ];
        let mut ctx = default_ctx();

        let saved = engine.apply(&mut tools, &mut ctx);
        assert!(saved > 0);

        // Should have 6 minimal entries + 1 synthetic = 7
        assert_eq!(tools.len(), 7);

        // Last tool should be the synthetic get_tool_schema
        assert_eq!(tools[6].name, "get_tool_schema");

        // Minimal entries should not have parameters
        let func = tools[0].raw.get("function").unwrap();
        assert!(func.get("parameters").is_none());
    }

    #[test]
    fn includes_disclosed_tools_at_end() {
        let engine = ProgressiveDisclosureEngine::new();
        let mut tools = vec![
            make_tool("tool_1", "First tool"),
            make_tool("tool_2", "Second tool"),
            make_tool("tool_3", "Third tool"),
            make_tool("tool_4", "Fourth tool"),
            make_tool("tool_5", "Fifth tool"),
            make_tool("tool_6", "Sixth tool"),
        ];
        let mut ctx = default_ctx();
        // Mark tool_2 and tool_4 as previously disclosed
        ctx.disclosed_tools.insert("tool_2".to_string());
        ctx.disclosed_tools.insert("tool_4".to_string());

        engine.apply(&mut tools, &mut ctx);

        // 6 minimal + 1 synthetic + 2 disclosed = 9
        assert_eq!(tools.len(), 9);

        // The last two should be the disclosed full schemas
        let disclosed_names: Vec<&str> = tools[7..].iter().map(|t| t.name.as_str()).collect();
        assert!(disclosed_names.contains(&"tool_2"));
        assert!(disclosed_names.contains(&"tool_4"));

        // Disclosed tools should have parameters (full schema)
        let func = tools[7].raw.get("function").unwrap();
        assert!(func.get("parameters").is_some());
    }

    #[test]
    fn truncate_description_short() {
        let desc = "Short desc";
        assert_eq!(truncate_description(desc, 100), "Short desc");
    }

    #[test]
    fn truncate_description_long() {
        let desc = "This is a very long description that exceeds the maximum allowed length for the minimal listing and should be truncated appropriately at a word boundary";
        let result = truncate_description(desc, 100);
        assert!(result.len() <= 103); // 100 + "..."
        assert!(result.ends_with("..."));
    }

    #[test]
    fn resolve_valid_tool() {
        let tools = vec![
            make_tool("search", "Search the web"),
            make_tool("calculate", "Do math"),
        ];

        let result = resolve_get_tool_schema("search", &tools);
        assert!(result.is_ok());
        let schema = result.unwrap();
        assert_eq!(
            schema.get("function").unwrap().get("name").unwrap(),
            "search"
        );
    }

    #[test]
    fn resolve_invalid_tool_returns_error_with_names() {
        let tools = vec![
            make_tool("search", "Search the web"),
            make_tool("calculate", "Do math"),
        ];

        let result = resolve_get_tool_schema("nonexistent", &tools);
        assert!(result.is_err());
        let err = result.unwrap_err();
        assert!(err.contains("nonexistent"));
        assert!(err.contains("search"));
        assert!(err.contains("calculate"));
    }

    #[test]
    fn is_enabled_checks_config_and_level() {
        let engine = ProgressiveDisclosureEngine::new();

        // Enabled: progressive_disclosure true + High level
        let mut config = ToolCompressionConfig::default();
        config.progressive_disclosure = true;
        assert!(engine.is_enabled(&config, CompressionLevel::High));
        assert!(engine.is_enabled(&config, CompressionLevel::Max));

        // Disabled: progressive_disclosure false
        config.progressive_disclosure = false;
        assert!(!engine.is_enabled(&config, CompressionLevel::High));

        // Disabled: level too low
        config.progressive_disclosure = true;
        assert!(!engine.is_enabled(&config, CompressionLevel::Low));
        assert!(!engine.is_enabled(&config, CompressionLevel::Medium));
    }

    #[test]
    fn strategies_applied_recorded() {
        let engine = ProgressiveDisclosureEngine::new();
        let mut tools = vec![
            make_tool("a", "A"),
            make_tool("b", "B"),
            make_tool("c", "C"),
            make_tool("d", "D"),
            make_tool("e", "E"),
            make_tool("f", "F"),
        ];
        let mut ctx = default_ctx();

        engine.apply(&mut tools, &mut ctx);
        assert!(ctx
            .strategies_applied
            .contains(&"progressive_disclosure".to_string()));
    }
}

// ─── Property-Based Tests ─────────────────────────────────────────────────────

#[cfg(test)]
mod property_tests {
    use super::*;
    use proptest::prelude::*;
    use serde_json::json;

    // ─── Strategies ──────────────────────────────────────────────────────────

    /// Generate a unique tool name (lowercase alpha, 3-12 chars).
    fn tool_name_strategy() -> impl Strategy<Value = String> {
        "[a-z]{3,12}"
    }

    /// Generate a set of 6-15 unique tool names.
    fn tool_names_strategy() -> impl Strategy<Value = Vec<String>> {
        prop::collection::hash_set(tool_name_strategy(), 6..=15usize)
            .prop_map(|s| s.into_iter().collect::<Vec<_>>())
    }

    /// Generate a random description string.
    fn description_strategy() -> impl Strategy<Value = String> {
        "[A-Za-z ]{5,80}"
    }

    /// Generate a random JSON schema with varying complexity.
    fn schema_strategy() -> impl Strategy<Value = serde_json::Value> {
        (
            prop::collection::hash_set("[a-z_]{2,8}", 1..=5usize),
            prop::sample::select(vec!["string", "integer", "boolean", "number"]),
        )
            .prop_map(|(param_names, param_type)| {
                let mut properties = serde_json::Map::new();
                let required: Vec<String> = param_names.iter().take(2).cloned().collect();
                for name in &param_names {
                    properties.insert(
                        name.clone(),
                        json!({
                            "type": param_type,
                            "description": format!("Parameter {name}")
                        }),
                    );
                }
                json!({
                    "type": "object",
                    "required": required,
                    "properties": properties
                })
            })
    }

    /// Build a ToolDefinition with a given name, description, and schema.
    fn make_tool_with_schema(
        name: &str,
        description: &str,
        parameters: &serde_json::Value,
    ) -> ToolDefinition {
        let raw = json!({
            "type": "function",
            "function": {
                "name": name,
                "description": description,
                "parameters": parameters
            }
        });
        ToolDefinition {
            raw,
            name: name.to_string(),
            content_hash: 0,
        }
    }

    /// Generate an invalid tool name guaranteed not to be in the provided set.
    fn invalid_name_strategy() -> impl Strategy<Value = String> {
        // Prefix with "__invalid_" to guarantee no collision with [a-z]{3,12} names
        "[a-z]{3,8}".prop_map(|s| format!("__invalid_{s}"))
    }

    // ─── Property 10: Progressive Disclosure Round-Trip ──────────────────────
    // Feature: tool-definition-compression
    // **Validates: Requirements 5.2, 5.3, 5.4**

    proptest! {
        #![proptest_config(ProptestConfig::with_cases(200))]

        /// Property: For every tool in the original set, resolve_get_tool_schema
        /// returns the exact original schema (byte-identical JSON Value).
        /// For an invalid tool name, it returns an Err containing the invalid name
        /// and listing all available tool names.
        #[test]
        fn prop_progressive_disclosure_round_trip(
            tool_names in tool_names_strategy(),
            descriptions in prop::collection::vec(description_strategy(), 6..=15),
            schemas in prop::collection::vec(schema_strategy(), 6..=15),
            invalid_name in invalid_name_strategy(),
        ) {
            // Use the minimum of lengths to build tools
            let count = tool_names.len().min(descriptions.len()).min(schemas.len());
            let tool_names = &tool_names[..count];
            let descriptions = &descriptions[..count];
            let schemas = &schemas[..count];

            // Build original_tools
            let original_tools: Vec<ToolDefinition> = tool_names
                .iter()
                .enumerate()
                .map(|(i, name)| make_tool_with_schema(name, &descriptions[i], &schemas[i]))
                .collect();

            // ─── Round-trip: valid tool names ────────────────────────────────
            for tool in &original_tools {
                let result = resolve_get_tool_schema(&tool.name, &original_tools);
                prop_assert!(
                    result.is_ok(),
                    "resolve_get_tool_schema('{}') should succeed but got Err: {:?}",
                    tool.name,
                    result.err()
                );
                let returned_schema = result.unwrap();
                // Byte-identical: the returned Value equals the original raw Value
                prop_assert_eq!(
                    &returned_schema,
                    &tool.raw,
                    "Schema mismatch for tool '{}'",
                    tool.name
                );
            }

            // ─── Error case: invalid tool name ──────────────────────────────
            let err_result = resolve_get_tool_schema(&invalid_name, &original_tools);
            prop_assert!(
                err_result.is_err(),
                "resolve_get_tool_schema('{}') should fail for unknown tool",
                invalid_name
            );
            let err_msg = err_result.unwrap_err();

            // Error message must contain the invalid name
            prop_assert!(
                err_msg.contains(&invalid_name),
                "Error message should contain invalid name '{}', got: {}",
                invalid_name,
                err_msg
            );

            // Error message must list all available tool names
            for tool in &original_tools {
                prop_assert!(
                    err_msg.contains(&tool.name),
                    "Error message should list available tool '{}', got: {}",
                    tool.name,
                    err_msg
                );
            }
        }
    }
}
