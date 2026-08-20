//! Canonical Rewriter stage — transforms JSON Schema tool definitions into
//! a compact structured text format (EasyTool-style) for maximum token reduction.
//!
//! Only activates when explicitly enabled AND the target model matches the
//! `allowed_models` glob pattern list AND the provider `supports_canonical_format`.

use std::sync::Arc;

use dashmap::DashMap;
use serde_json::Value;

use crate::tool_compression::config::{CompressionLevel, ToolCompressionConfig};
use crate::tool_compression::stage::CompressionStage;
use crate::tool_compression::stages::pruner::GlobPattern;
use crate::tool_compression::types::{CompressionContext, ToolDefinition};

/// Canonical Rewriter compression stage.
///
/// Terminal pipeline stage that replaces JSON Schema tool definitions with a
/// compact structured text format. Stores original schemas for validation.
pub struct CanonicalRewriter {
    /// Parsed glob patterns for models permitted to receive canonical format.
    allowed_models: Vec<GlobPattern>,
    /// Storage for original schemas keyed by tool name (needed for tool-call validation).
    original_schemas: Arc<DashMap<String, Value>>,
}

impl CanonicalRewriter {
    /// Create a new `CanonicalRewriter` with the given allowed model patterns
    /// and shared schema storage.
    pub fn new(
        allowed_model_patterns: &[String],
        original_schemas: Arc<DashMap<String, Value>>,
    ) -> Self {
        let allowed_models = allowed_model_patterns
            .iter()
            .map(|p| GlobPattern::new(p))
            .collect();
        Self {
            allowed_models,
            original_schemas,
        }
    }

    /// Check if the given model name matches any allowed pattern.
    fn model_matches(&self, model: &str) -> bool {
        self.allowed_models.iter().any(|p| p.matches(model))
    }

    /// Convert a single tool definition to canonical text format.
    fn to_canonical(&self, tool: &ToolDefinition) -> String {
        let raw = &tool.raw;
        let func = raw.get("function").unwrap_or(raw);

        let name = func
            .get("name")
            .and_then(|v| v.as_str())
            .unwrap_or(&tool.name);

        let desc = func
            .get("description")
            .and_then(|v| v.as_str())
            .unwrap_or("");

        let params_str = self.format_params(func.get("parameters"));
        let returns_str = self.format_returns(func.get("returns"));

        let mut output = format!("tool: {name}\ndesc: {desc}");
        if !params_str.is_empty() {
            output.push_str(&format!("\nparams: {params_str}"));
        }
        if !returns_str.is_empty() {
            output.push_str(&format!("\nreturns: {returns_str}"));
        }
        output
    }

    /// Format parameters from a JSON Schema `parameters` object.
    fn format_params(&self, params: Option<&Value>) -> String {
        let Some(params) = params else {
            return String::new();
        };

        let properties = match params.get("properties") {
            Some(Value::Object(props)) => props,
            _ => return String::new(),
        };

        let required: Vec<&str> = params
            .get("required")
            .and_then(|v| v.as_array())
            .map(|arr| arr.iter().filter_map(|v| v.as_str()).collect())
            .unwrap_or_default();

        let mut param_strs: Vec<String> = Vec::new();

        for (key, schema) in properties.iter() {
            let is_required = required.contains(&key.as_str());
            self.format_param_entry(&mut param_strs, key, schema, is_required, None);
        }

        param_strs.join(", ")
    }

    /// Format a single parameter entry, flattening depth-1 nested objects.
    fn format_param_entry(
        &self,
        out: &mut Vec<String>,
        name: &str,
        schema: &Value,
        is_required: bool,
        parent_prefix: Option<&str>,
    ) {
        let full_name = match parent_prefix {
            Some(prefix) => format!("{prefix}.{name}"),
            None => name.to_string(),
        };

        let type_str = self.extract_type_string(schema);

        // Check if this is an object with properties at depth 0 (no parent) → flatten
        if parent_prefix.is_none() && is_object_with_properties(schema) {
            let nested_props = schema
                .get("properties")
                .and_then(|v| v.as_object())
                .unwrap();
            let nested_required: Vec<&str> = schema
                .get("required")
                .and_then(|v| v.as_array())
                .map(|arr| arr.iter().filter_map(|v| v.as_str()).collect())
                .unwrap_or_default();

            for (child_key, child_schema) in nested_props.iter() {
                let child_required = nested_required.contains(&child_key.as_str());
                // Only flatten depth-1; deeper nesting is preserved as JSON Schema
                if is_object_with_properties(child_schema) {
                    // Deeper than 1 level — emit as JSON Schema inline
                    let json_repr = serde_json::to_string(child_schema).unwrap_or_default();
                    let req_str = if child_required {
                        "required"
                    } else {
                        "optional"
                    };
                    out.push(format!("{full_name}.{child_key}({json_repr}, {req_str})"));
                } else {
                    self.format_param_entry(
                        out,
                        child_key,
                        child_schema,
                        child_required,
                        Some(&full_name),
                    );
                }
            }
            return;
        }

        let req_str = if is_required { "required" } else { "optional" };
        let default_str = schema
            .get("default")
            .map(|d| format!(", default={}", format_default_value(d)))
            .unwrap_or_default();

        out.push(format!("{full_name}({type_str}, {req_str}{default_str})"));
    }

    /// Extract a compact type representation from a JSON Schema value.
    fn extract_type_string(&self, schema: &Value) -> String {
        // Handle enum
        if let Some(enum_values) = schema.get("enum").and_then(|v| v.as_array()) {
            let vals: Vec<&str> = enum_values.iter().filter_map(|v| v.as_str()).collect();
            return format!("enum[{}]", vals.join(","));
        }

        // Handle type field
        let type_val = schema.get("type").and_then(|v| v.as_str()).unwrap_or("any");

        match type_val {
            "array" => {
                let items_type = schema
                    .get("items")
                    .map(|items| self.extract_type_string(items))
                    .unwrap_or_else(|| "any".to_string());
                format!("array[{items_type}]")
            }
            other => other.to_string(),
        }
    }

    /// Format the `returns` field from a schema's return type definition.
    fn format_returns(&self, returns: Option<&Value>) -> String {
        let Some(returns) = returns else {
            return String::new();
        };

        match returns {
            Value::Object(obj) => {
                if let Some(Value::Object(props)) = returns.get("properties") {
                    let fields: Vec<String> = props
                        .iter()
                        .map(|(k, v)| {
                            let t = self.extract_type_string(v);
                            format!("{k}: {t}")
                        })
                        .collect();
                    format!("{{{}}}", fields.join(", "))
                } else if obj.contains_key("type") {
                    self.extract_type_string(returns)
                } else {
                    serde_json::to_string(returns).unwrap_or_default()
                }
            }
            _ => String::new(),
        }
    }

    /// Build the canonical output JSON structure for a tool.
    fn build_canonical_output(&self, tool: &ToolDefinition, canonical_text: &str) -> Value {
        serde_json::json!({
            "type": "function",
            "function": {
                "name": tool.name,
                "description": canonical_text
            }
        })
    }
}

impl CompressionStage for CanonicalRewriter {
    fn apply(&self, tools: &mut Vec<ToolDefinition>, ctx: &mut CompressionContext) -> u64 {
        // Activation checks
        if self.allowed_models.is_empty() {
            return 0;
        }
        if !self.model_matches(&ctx.model) {
            return 0;
        }
        if !ctx.provider_caps.supports_canonical_format {
            return 0;
        }

        let mut tokens_saved: u64 = 0;

        for tool in tools.iter_mut() {
            // Store original schema before rewriting
            self.original_schemas
                .insert(tool.name.clone(), tool.raw.clone());

            // Estimate original tokens (chars / 4)
            let original_json = serde_json::to_string(&tool.raw).unwrap_or_default();
            let original_tokens = (original_json.len() as u64) / 4;

            // Generate canonical text
            let canonical_text = self.to_canonical(tool);

            // Build canonical output and replace raw
            let canonical_output = self.build_canonical_output(tool, &canonical_text);
            tool.raw = canonical_output;

            // Recompute hash for the new content
            let new_json = serde_json::to_string(&tool.raw).unwrap_or_default();
            tool.content_hash = compute_hash(&new_json);

            // Estimate new tokens
            let new_tokens = (new_json.len() as u64) / 4;
            tokens_saved += original_tokens.saturating_sub(new_tokens);
        }

        if tokens_saved > 0 {
            ctx.strategies_applied
                .push("canonical_rewriter".to_string());
            ctx.tokens_saved += tokens_saved;
        }

        tokens_saved
    }

    fn is_enabled(&self, config: &ToolCompressionConfig, level: CompressionLevel) -> bool {
        // Activate when: level == Max OR canonical_rewriting.enabled
        // The model/provider checks are done in apply() since they need runtime context
        if self.allowed_models.is_empty() {
            return false;
        }
        level == CompressionLevel::Max || config.canonical_rewriting.enabled
    }
}

// ─── Helpers ──────────────────────────────────────────────────────────────────

/// Compute a 64-bit hash of a string for content comparisons.
fn compute_hash(s: &str) -> u64 {
    use std::hash::{Hash, Hasher};
    let mut hasher = std::collections::hash_map::DefaultHasher::new();
    s.hash(&mut hasher);
    hasher.finish()
}

/// Check if a schema value is an object type with `properties`.
fn is_object_with_properties(schema: &Value) -> bool {
    let is_obj_type = schema
        .get("type")
        .and_then(|v| v.as_str())
        .map(|t| t == "object")
        .unwrap_or(false);
    is_obj_type && schema.get("properties").is_some()
}

/// Format a default value for display in the canonical format.
fn format_default_value(value: &Value) -> String {
    match value {
        Value::String(s) => s.clone(),
        Value::Number(n) => n.to_string(),
        Value::Bool(b) => b.to_string(),
        Value::Null => "null".to_string(),
        _ => serde_json::to_string(value).unwrap_or_else(|_| "?".to_string()),
    }
}

// ─── Tests ────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    fn make_tool(name: &str, raw: Value) -> ToolDefinition {
        let json_str = serde_json::to_string(&raw).unwrap();
        ToolDefinition {
            raw,
            name: name.to_string(),
            content_hash: compute_hash(&json_str),
        }
    }

    fn default_ctx_with_model(model: &str) -> CompressionContext {
        CompressionContext {
            level: CompressionLevel::Max,
            model: model.to_string(),
            provider_caps: crate::tool_compression::types::ProviderCaps {
                supports_canonical_format: true,
                ..crate::tool_compression::types::ProviderCaps::conservative()
            },
            ..Default::default()
        }
    }

    #[test]
    fn noop_when_allowed_models_empty() {
        let rewriter = CanonicalRewriter::new(&[], Arc::new(DashMap::new()));
        let mut tools = vec![make_tool(
            "test",
            serde_json::json!({
                "type": "function",
                "function": {"name": "test", "description": "A test tool"}
            }),
        )];
        let mut ctx = default_ctx_with_model("gpt-4");
        let saved = rewriter.apply(&mut tools, &mut ctx);
        assert_eq!(saved, 0);
    }

    #[test]
    fn noop_when_model_does_not_match() {
        let rewriter = CanonicalRewriter::new(&["gpt-4*".to_string()], Arc::new(DashMap::new()));
        let mut tools = vec![make_tool(
            "test",
            serde_json::json!({
                "type": "function",
                "function": {"name": "test", "description": "A test tool"}
            }),
        )];
        let mut ctx = default_ctx_with_model("claude-3");
        let saved = rewriter.apply(&mut tools, &mut ctx);
        assert_eq!(saved, 0);
    }

    #[test]
    fn noop_when_provider_does_not_support_canonical() {
        let rewriter = CanonicalRewriter::new(&["gpt-4*".to_string()], Arc::new(DashMap::new()));
        let mut tools = vec![make_tool(
            "test",
            serde_json::json!({
                "type": "function",
                "function": {"name": "test", "description": "A test tool"}
            }),
        )];
        let mut ctx = CompressionContext {
            level: CompressionLevel::Max,
            model: "gpt-4".to_string(),
            provider_caps: crate::tool_compression::types::ProviderCaps::conservative(),
            ..Default::default()
        };
        let saved = rewriter.apply(&mut tools, &mut ctx);
        assert_eq!(saved, 0);
    }

    #[test]
    fn rewrites_simple_tool() {
        let schemas = Arc::new(DashMap::new());
        let rewriter = CanonicalRewriter::new(&["gpt-*".to_string()], schemas.clone());

        let raw = serde_json::json!({
            "type": "function",
            "function": {
                "name": "get_weather",
                "description": "Get current weather for a location.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "location": {"type": "string"},
                        "units": {
                            "type": "string",
                            "enum": ["celsius", "fahrenheit"],
                            "default": "celsius"
                        }
                    },
                    "required": ["location"]
                }
            }
        });

        let mut tools = vec![make_tool("get_weather", raw.clone())];
        let mut ctx = default_ctx_with_model("gpt-4");
        let saved = rewriter.apply(&mut tools, &mut ctx);

        assert!(saved > 0);
        // Check canonical output structure
        let output = &tools[0].raw;
        assert_eq!(output["type"], "function");
        assert_eq!(output["function"]["name"], "get_weather");

        let desc = output["function"]["description"].as_str().unwrap();
        assert!(desc.contains("tool: get_weather"));
        assert!(desc.contains("desc: Get current weather for a location."));
        assert!(desc.contains("location(string, required)"));
        assert!(desc.contains("units(enum[celsius,fahrenheit], optional, default=celsius)"));

        // Original schema stored
        assert!(schemas.contains_key("get_weather"));
        assert_eq!(*schemas.get("get_weather").unwrap().value(), raw);

        // Strategy recorded
        assert!(ctx
            .strategies_applied
            .contains(&"canonical_rewriter".to_string()));
    }

    #[test]
    fn flattens_depth_1_nested_params() {
        let schemas = Arc::new(DashMap::new());
        let rewriter = CanonicalRewriter::new(&["*".to_string()], schemas.clone());

        let raw = serde_json::json!({
            "type": "function",
            "function": {
                "name": "configure",
                "description": "Configure settings.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "config": {
                            "type": "object",
                            "properties": {
                                "timeout": {"type": "integer", "default": 30},
                                "retries": {"type": "integer", "default": 3}
                            },
                            "required": ["timeout"]
                        }
                    },
                    "required": ["config"]
                }
            }
        });

        let mut tools = vec![make_tool("configure", raw)];
        let mut ctx = default_ctx_with_model("gpt-4");
        rewriter.apply(&mut tools, &mut ctx);

        let desc = tools[0].raw["function"]["description"].as_str().unwrap();
        assert!(desc.contains("config.timeout(integer, required, default=30)"));
        assert!(desc.contains("config.retries(integer, optional, default=3)"));
    }

    #[test]
    fn preserves_deep_nesting_as_json() {
        let schemas = Arc::new(DashMap::new());
        let rewriter = CanonicalRewriter::new(&["*".to_string()], schemas.clone());

        let deep_child = serde_json::json!({
            "type": "object",
            "properties": {
                "value": {"type": "string"}
            }
        });

        let raw = serde_json::json!({
            "type": "function",
            "function": {
                "name": "deep_tool",
                "description": "Has deep nesting.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "outer": {
                            "type": "object",
                            "properties": {
                                "inner": deep_child
                            }
                        }
                    },
                    "required": ["outer"]
                }
            }
        });

        let mut tools = vec![make_tool("deep_tool", raw)];
        let mut ctx = default_ctx_with_model("gpt-4");
        rewriter.apply(&mut tools, &mut ctx);

        let desc = tools[0].raw["function"]["description"].as_str().unwrap();
        // Deep nesting should contain JSON schema representation
        assert!(desc.contains("outer.inner("));
        // The JSON schema for the deep child should be present
        assert!(desc.contains("\"type\":\"object\""));
    }

    #[test]
    fn is_enabled_at_max_level() {
        let rewriter = CanonicalRewriter::new(&["gpt-*".to_string()], Arc::new(DashMap::new()));
        let config = ToolCompressionConfig::default();
        assert!(rewriter.is_enabled(&config, CompressionLevel::Max));
    }

    #[test]
    fn is_enabled_when_config_enabled() {
        let rewriter = CanonicalRewriter::new(&["gpt-*".to_string()], Arc::new(DashMap::new()));
        let config = ToolCompressionConfig {
            canonical_rewriting: crate::tool_compression::config::CanonicalRewritingConfig {
                enabled: true,
                allowed_models: vec!["gpt-*".to_string()],
            },
            ..Default::default()
        };
        assert!(rewriter.is_enabled(&config, CompressionLevel::Low));
    }

    #[test]
    fn not_enabled_when_allowed_models_empty() {
        let rewriter = CanonicalRewriter::new(&[], Arc::new(DashMap::new()));
        let config = ToolCompressionConfig::default();
        assert!(!rewriter.is_enabled(&config, CompressionLevel::Max));
    }

    #[test]
    fn glob_matching_works() {
        let rewriter = CanonicalRewriter::new(
            &["gpt-4*".to_string(), "claude-3*".to_string()],
            Arc::new(DashMap::new()),
        );
        assert!(rewriter.model_matches("gpt-4"));
        assert!(rewriter.model_matches("gpt-4-turbo"));
        assert!(rewriter.model_matches("claude-3-sonnet"));
        assert!(!rewriter.model_matches("gemini-pro"));
        assert!(!rewriter.model_matches("gpt-3.5-turbo"));
    }
}

// ─── Property Tests ───────────────────────────────────────────────────────────

#[cfg(test)]
mod property_tests {
    use super::*;
    use proptest::prelude::*;
    use serde_json::json;
    use std::sync::Arc;

    // ─── Strategies ──────────────────────────────────────────────────────────

    fn make_tool_def(name: &str, raw: Value) -> ToolDefinition {
        let json_str = serde_json::to_string(&raw).unwrap();
        ToolDefinition {
            raw,
            name: name.to_string(),
            content_hash: {
                use std::hash::{Hash, Hasher};
                let mut hasher = std::collections::hash_map::DefaultHasher::new();
                json_str.hash(&mut hasher);
                hasher.finish()
            },
        }
    }

    fn ctx_with_model(model: &str) -> CompressionContext {
        CompressionContext {
            level: CompressionLevel::Max,
            model: model.to_string(),
            provider_caps: crate::tool_compression::types::ProviderCaps {
                supports_canonical_format: true,
                ..crate::tool_compression::types::ProviderCaps::conservative()
            },
            ..Default::default()
        }
    }

    /// Generate random tool schemas with varying parameter types.
    fn arb_tool_schema() -> impl Strategy<Value = (String, Value)> {
        let name_strat = "[a-z_]{3,10}";
        let desc_strat = "[a-zA-Z ]{5,30}";
        let param_type_strat = prop_oneof![
            Just("string".to_string()),
            Just("integer".to_string()),
            Just("number".to_string()),
            Just("boolean".to_string()),
        ];

        (
            name_strat,
            desc_strat,
            prop::collection::vec((("[a-z_]{2,8}"), param_type_strat), 1..=4),
        )
            .prop_map(|(name, desc, params)| {
                let mut properties = serde_json::Map::new();
                let mut required = Vec::new();

                for (i, (param_name, param_type)) in params.iter().enumerate() {
                    properties.insert(param_name.clone(), json!({"type": param_type}));
                    if i == 0 {
                        required.push(json!(param_name.clone()));
                    }
                }

                let raw = json!({
                    "type": "function",
                    "function": {
                        "name": name,
                        "description": desc,
                        "parameters": {
                            "type": "object",
                            "properties": properties,
                            "required": required
                        }
                    }
                });

                (name, raw)
            })
    }

    /// Generate random model names.
    fn arb_model_name() -> impl Strategy<Value = String> {
        prop_oneof![
            Just("gpt-4-turbo".to_string()),
            Just("gpt-4o".to_string()),
            Just("claude-3-sonnet".to_string()),
            Just("gemini-pro".to_string()),
            Just("llama-3-70b".to_string()),
            "[a-z]{3,8}-[0-9]{1,2}".prop_map(|s| s),
        ]
    }

    /// Generate random glob patterns for allowed_models.
    fn arb_allowed_patterns() -> impl Strategy<Value = Vec<String>> {
        prop::collection::vec(
            prop_oneof![
                Just("gpt-*".to_string()),
                Just("claude-*".to_string()),
                Just("*".to_string()),
                Just("gemini-*".to_string()),
                "[a-z]{3,6}-*".prop_map(|s| s),
            ],
            1..=3,
        )
    }

    // ─── Property 12: Canonical Rewriting Round-Trip ──────────────────────────
    // Feature: tool-definition-compression
    // **Validates: Requirements 17.1, 17.4**
    //
    // For each tool rewritten, verify the original schema was stored in the
    // DashMap and is structurally equivalent to what was passed in.

    proptest! {
        #![proptest_config(ProptestConfig::with_cases(64))]

        #[test]
        fn canonical_rewriting_round_trip(
            schema_data in prop::collection::vec(arb_tool_schema(), 1..=5)
        ) {
            let schemas = Arc::new(DashMap::new());
            let rewriter = CanonicalRewriter::new(&["*".to_string()], schemas.clone());

            // Build tools from generated schemas
            let mut tools: Vec<ToolDefinition> = schema_data
                .iter()
                .enumerate()
                .map(|(i, (name, raw))| {
                    let unique_name = format!("{}_{}", name, i);
                    let mut raw_clone = raw.clone();
                    // Ensure unique name in raw
                    if let Some(func) = raw_clone.get_mut("function") {
                        func.as_object_mut().unwrap().insert(
                            "name".to_string(),
                            json!(unique_name),
                        );
                    }
                    make_tool_def(&unique_name, raw_clone)
                })
                .collect();

            // Store originals for comparison
            let originals: Vec<(String, Value)> = tools
                .iter()
                .map(|t| (t.name.clone(), t.raw.clone()))
                .collect();

            let mut ctx = ctx_with_model("gpt-4");
            rewriter.apply(&mut tools, &mut ctx);

            // Verify each original was stored in the DashMap
            for (name, original_raw) in &originals {
                prop_assert!(
                    schemas.contains_key(name),
                    "Original schema for '{}' must be stored in DashMap",
                    name
                );
                let stored = schemas.get(name).unwrap();
                prop_assert_eq!(
                    stored.value(),
                    original_raw,
                    "Stored schema for '{}' must be structurally equivalent to input",
                    name
                );
            }
        }
    }

    // ─── Property 13: Canonical Format Conditional Activation ─────────────────
    // Feature: tool-definition-compression
    // **Validates: Requirements 17.5, 17.6, 17.7, 17.8, 17.9**
    //
    // Generate random model names and allowed_models patterns; verify that when
    // model matches a pattern, tools ARE rewritten (output contains "tool:" prefix
    // in description); when model doesn't match, tools are NOT rewritten.

    proptest! {
        #![proptest_config(ProptestConfig::with_cases(64))]

        #[test]
        fn canonical_format_conditional_activation(
            model in arb_model_name(),
            patterns in arb_allowed_patterns(),
        ) {
            let schemas = Arc::new(DashMap::new());
            let rewriter = CanonicalRewriter::new(&patterns, schemas.clone());

            let raw = json!({
                "type": "function",
                "function": {
                    "name": "test_tool",
                    "description": "A test tool for validation.",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "input": {"type": "string"}
                        },
                        "required": ["input"]
                    }
                }
            });

            let mut tools = vec![make_tool_def("test_tool", raw.clone())];
            let mut ctx = ctx_with_model(&model);
            let original_raw = tools[0].raw.clone();

            rewriter.apply(&mut tools, &mut ctx);

            let model_matches = rewriter.model_matches(&model);

            if model_matches {
                // Tools should be rewritten: description should contain "tool:" prefix
                let desc = tools[0].raw["function"]["description"]
                    .as_str()
                    .unwrap_or("");
                prop_assert!(
                    desc.contains("tool:"),
                    "When model '{}' matches patterns {:?}, output description should contain 'tool:' prefix. Got: '{}'",
                    model,
                    patterns,
                    desc
                );
            } else {
                // Tools should NOT be rewritten: raw should be unchanged
                prop_assert_eq!(
                    &tools[0].raw,
                    &original_raw,
                    "When model '{}' does NOT match patterns {:?}, tools should be unchanged",
                    model,
                    patterns
                );
            }
        }
    }

    // ─── Property 14: Canonical Dot-Notation Flattening ──────────────────────
    // Feature: tool-definition-compression
    // **Validates: Requirements 17.1, 17.9**
    //
    // Generate schemas with nested objects at depth 1; verify the canonical text
    // contains dot-notation params (e.g., "parent.child(type, req)").

    proptest! {
        #![proptest_config(ProptestConfig::with_cases(64))]

        #[test]
        fn canonical_dot_notation_flattening(
            parent_name in "[a-z]{3,8}",
            child_name in "[a-z]{3,8}",
            child_type in prop_oneof![
                Just("string"),
                Just("integer"),
                Just("number"),
                Just("boolean"),
            ],
        ) {
            let schemas = Arc::new(DashMap::new());
            let rewriter = CanonicalRewriter::new(&["*".to_string()], schemas.clone());

            let raw = json!({
                "type": "function",
                "function": {
                    "name": "nested_tool",
                    "description": "Tool with nested params.",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            parent_name.clone(): {
                                "type": "object",
                                "properties": {
                                    child_name.clone(): {"type": child_type}
                                },
                                "required": [child_name.clone()]
                            }
                        },
                        "required": [parent_name.clone()]
                    }
                }
            });

            let mut tools = vec![make_tool_def("nested_tool", raw)];
            let mut ctx = ctx_with_model("gpt-4");
            rewriter.apply(&mut tools, &mut ctx);

            let desc = tools[0].raw["function"]["description"]
                .as_str()
                .unwrap_or("");

            // Should contain dot-notation: parent.child
            let dot_notation = format!("{}.{}", parent_name, child_name);
            prop_assert!(
                desc.contains(&dot_notation),
                "Canonical text should contain dot-notation '{}' for depth-1 nested params. Got: '{}'",
                dot_notation,
                desc
            );

            // Should contain the child's type
            prop_assert!(
                desc.contains(child_type),
                "Canonical text should contain child type '{}'. Got: '{}'",
                child_type,
                desc
            );
        }
    }
}
