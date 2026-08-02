//! Schema Minifier stage — strips redundant JSON Schema fields in-place.
//!
//! Removes `title`, `additionalProperties: false`, empty descriptions,
//! collapses single-element `anyOf`/`oneOf` wrappers, and converts nullable
//! unions to `nullable: true` when the provider supports it.

use serde_json::Value;

use crate::tool_compression::config::{CompressionLevel, MinificationConfig, ToolCompressionConfig};
use crate::tool_compression::stage::CompressionStage;
use crate::tool_compression::types::{CompressionContext, ProviderCaps, ToolDefinition};

/// Schema minification stage.
///
/// Recursively walks tool parameter schemas and removes redundant fields
/// without altering parameter semantics.
pub struct SchemaMinifier;

impl CompressionStage for SchemaMinifier {
    fn apply(&self, tools: &mut Vec<ToolDefinition>, ctx: &mut CompressionContext) -> u64 {
        let config = &ctx.provider_caps;
        let mut total_saved: u64 = 0;

        for tool in tools.iter_mut() {
            let before = estimate_tokens(&tool.raw);
            minify_value(&mut tool.raw, config);
            let after = estimate_tokens(&tool.raw);
            total_saved += before.saturating_sub(after);
        }

        if total_saved > 0 {
            ctx.strategies_applied.push("schema_minifier".to_string());
        }
        ctx.tokens_saved += total_saved;
        total_saved
    }

    fn is_enabled(&self, config: &ToolCompressionConfig, _level: CompressionLevel) -> bool {
        let m = &config.minification;
        m.remove_titles
            || m.collapse_single_unions
            || m.remove_additional_properties
            || m.remove_empty_descriptions
    }
}

/// Estimate token count as character_count / 4.
fn estimate_tokens(value: &Value) -> u64 {
    let s = value.to_string();
    (s.len() as u64) / 4
}

/// Recursively minify a JSON value in-place according to provider capabilities.
pub fn minify_value(value: &mut Value, provider_caps: &ProviderCaps) {
    match value {
        Value::Object(map) => {
            // Remove `title` fields
            map.remove("title");

            // Remove `additionalProperties: false`
            if let Some(Value::Bool(false)) = map.get("additionalProperties") {
                map.remove("additionalProperties");
            }

            // Remove empty/whitespace-only `description` fields
            if let Some(Value::String(desc)) = map.get("description") {
                if desc.trim().is_empty() {
                    map.remove("description");
                }
            }

            // Collapse single-element `anyOf`/`oneOf` and nullable unions
            for key in &["anyOf", "oneOf"] {
                if let Some(arr_val) = map.get(*key) {
                    if let Some(arr) = arr_val.as_array() {
                        let len = arr.len();

                        if len == 1 {
                            // Single-element wrapper — promote inner type
                            let inner = arr[0].clone();
                            map.remove(*key);
                            // Merge the inner object fields into the parent
                            if let Value::Object(inner_map) = inner {
                                for (k, v) in inner_map {
                                    map.insert(k, v);
                                }
                            }
                            // After promotion, recurse on the now-modified map
                            for (_k, v) in map.iter_mut() {
                                minify_value(v, provider_caps);
                            }
                            return;
                        } else if len == 2 {
                            // Check for nullable union pattern: [non-null-type, {type: "null"}]
                            // or [{type: "null"}, non-null-type]
                            if let Some((non_null_idx, _null_idx)) = find_nullable_pair(arr) {
                                if provider_caps.supports_nullable {
                                    let non_null = arr[non_null_idx].clone();
                                    map.remove(*key);
                                    // Merge non-null type fields into parent
                                    if let Value::Object(inner_map) = non_null {
                                        for (k, v) in inner_map {
                                            map.insert(k, v);
                                        }
                                    }
                                    map.insert(
                                        "nullable".to_string(),
                                        Value::Bool(true),
                                    );
                                    // Recurse on the modified map
                                    for (_k, v) in map.iter_mut() {
                                        minify_value(v, provider_caps);
                                    }
                                    return;
                                }
                                // Provider doesn't support nullable — leave unchanged,
                                // but still recurse into array elements below.
                            }
                        }
                    }
                }
            }

            // Recurse into all remaining child values
            for (_k, v) in map.iter_mut() {
                minify_value(v, provider_caps);
            }
        }
        Value::Array(arr) => {
            for item in arr.iter_mut() {
                minify_value(item, provider_caps);
            }
        }
        _ => {}
    }
}

/// Find a nullable pair in a 2-element array: one `{type: "null"}` and one non-null type.
/// Returns `Some((non_null_index, null_index))` or `None`.
fn find_nullable_pair(arr: &[Value]) -> Option<(usize, usize)> {
    if arr.len() != 2 {
        return None;
    }

    let is_null_type = |v: &Value| -> bool {
        v.as_object()
            .and_then(|obj| obj.get("type"))
            .and_then(|t| t.as_str())
            .map(|s| s == "null")
            .unwrap_or(false)
    };

    if is_null_type(&arr[0]) && !is_null_type(&arr[1]) {
        Some((1, 0))
    } else if is_null_type(&arr[1]) && !is_null_type(&arr[0]) {
        Some((0, 1))
    } else {
        None
    }
}

// ─── Tests ────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    fn caps_with_nullable() -> ProviderCaps {
        ProviderCaps {
            supports_nullable: true,
            ..ProviderCaps::conservative()
        }
    }

    fn caps_without_nullable() -> ProviderCaps {
        ProviderCaps::conservative()
    }

    #[test]
    fn removes_title_fields_recursively() {
        let mut val = json!({
            "type": "object",
            "title": "TopLevel",
            "properties": {
                "name": {
                    "type": "string",
                    "title": "Name Field"
                }
            }
        });
        minify_value(&mut val, &caps_without_nullable());
        assert!(val.get("title").is_none());
        assert!(val["properties"]["name"].get("title").is_none());
        assert_eq!(val["properties"]["name"]["type"], "string");
    }

    #[test]
    fn removes_additional_properties_false() {
        let mut val = json!({
            "type": "object",
            "additionalProperties": false,
            "properties": {
                "id": { "type": "integer" }
            }
        });
        minify_value(&mut val, &caps_without_nullable());
        assert!(val.get("additionalProperties").is_none());
        assert_eq!(val["type"], "object");
    }

    #[test]
    fn preserves_additional_properties_true() {
        let mut val = json!({
            "type": "object",
            "additionalProperties": true
        });
        minify_value(&mut val, &caps_without_nullable());
        assert_eq!(val["additionalProperties"], true);
    }

    #[test]
    fn removes_empty_description() {
        let mut val = json!({
            "type": "string",
            "description": ""
        });
        minify_value(&mut val, &caps_without_nullable());
        assert!(val.get("description").is_none());
    }

    #[test]
    fn removes_whitespace_only_description() {
        let mut val = json!({
            "type": "string",
            "description": "   \t\n  "
        });
        minify_value(&mut val, &caps_without_nullable());
        assert!(val.get("description").is_none());
    }

    #[test]
    fn preserves_non_empty_description() {
        let mut val = json!({
            "type": "string",
            "description": "A user name"
        });
        minify_value(&mut val, &caps_without_nullable());
        assert_eq!(val["description"], "A user name");
    }

    #[test]
    fn collapses_single_element_any_of() {
        let mut val = json!({
            "anyOf": [{"type": "string", "description": "name"}]
        });
        minify_value(&mut val, &caps_without_nullable());
        assert!(val.get("anyOf").is_none());
        assert_eq!(val["type"], "string");
        assert_eq!(val["description"], "name");
    }

    #[test]
    fn collapses_single_element_one_of() {
        let mut val = json!({
            "oneOf": [{"type": "integer", "format": "int64"}]
        });
        minify_value(&mut val, &caps_without_nullable());
        assert!(val.get("oneOf").is_none());
        assert_eq!(val["type"], "integer");
        assert_eq!(val["format"], "int64");
    }

    #[test]
    fn collapses_nullable_union_when_supported() {
        let mut val = json!({
            "anyOf": [
                {"type": "string"},
                {"type": "null"}
            ]
        });
        minify_value(&mut val, &caps_with_nullable());
        assert!(val.get("anyOf").is_none());
        assert_eq!(val["type"], "string");
        assert_eq!(val["nullable"], true);
    }

    #[test]
    fn collapses_nullable_union_reverse_order() {
        let mut val = json!({
            "oneOf": [
                {"type": "null"},
                {"type": "integer"}
            ]
        });
        minify_value(&mut val, &caps_with_nullable());
        assert!(val.get("oneOf").is_none());
        assert_eq!(val["type"], "integer");
        assert_eq!(val["nullable"], true);
    }

    #[test]
    fn leaves_nullable_union_unchanged_when_unsupported() {
        let mut val = json!({
            "anyOf": [
                {"type": "string"},
                {"type": "null"}
            ]
        });
        let original = val.clone();
        minify_value(&mut val, &caps_without_nullable());
        assert_eq!(val, original);
    }

    #[test]
    fn preserves_required_enum_type_properties_items_default_format() {
        let mut val = json!({
            "type": "object",
            "required": ["id", "name"],
            "properties": {
                "id": { "type": "integer", "format": "int64", "default": 0 },
                "status": { "type": "string", "enum": ["active", "inactive"] }
            },
            "items": { "type": "string" }
        });
        let expected = val.clone();
        minify_value(&mut val, &caps_without_nullable());
        assert_eq!(val, expected);
    }

    #[test]
    fn does_not_collapse_multi_element_any_of() {
        let mut val = json!({
            "anyOf": [
                {"type": "string"},
                {"type": "integer"}
            ]
        });
        let original = val.clone();
        minify_value(&mut val, &caps_without_nullable());
        assert_eq!(val, original);
    }

    #[test]
    fn deeply_nested_minification() {
        let mut val = json!({
            "type": "object",
            "title": "Root",
            "properties": {
                "config": {
                    "type": "object",
                    "title": "Config",
                    "additionalProperties": false,
                    "properties": {
                        "timeout": {
                            "title": "Timeout",
                            "anyOf": [{"type": "integer"}],
                            "description": ""
                        }
                    }
                }
            }
        });
        minify_value(&mut val, &caps_without_nullable());
        // Root title removed
        assert!(val.get("title").is_none());
        // Nested title removed
        let config = &val["properties"]["config"];
        assert!(config.get("title").is_none());
        assert!(config.get("additionalProperties").is_none());
        // Deeply nested: title removed, anyOf collapsed, empty desc removed
        let timeout = &config["properties"]["timeout"];
        assert!(timeout.get("title").is_none());
        assert!(timeout.get("anyOf").is_none());
        assert!(timeout.get("description").is_none());
        assert_eq!(timeout["type"], "integer");
    }

    #[test]
    fn idempotent_minification() {
        let mut val = json!({
            "type": "object",
            "title": "Test",
            "additionalProperties": false,
            "properties": {
                "x": {
                    "anyOf": [{"type": "string"}],
                    "title": "X",
                    "description": "  "
                }
            }
        });
        let caps = caps_with_nullable();
        minify_value(&mut val, &caps);
        let after_first = val.clone();
        minify_value(&mut val, &caps);
        assert_eq!(val, after_first, "minification should be idempotent");
    }

    #[test]
    fn stage_is_enabled_when_any_sub_feature_active() {
        let stage = SchemaMinifier;
        let mut config = ToolCompressionConfig::default();

        // Default: all minification sub-features true → enabled
        assert!(stage.is_enabled(&config, CompressionLevel::Low));

        // Disable all
        config.minification = MinificationConfig {
            remove_titles: false,
            collapse_single_unions: false,
            remove_additional_properties: false,
            remove_empty_descriptions: false,
        };
        assert!(!stage.is_enabled(&config, CompressionLevel::High));

        // Enable just one
        config.minification.remove_titles = true;
        assert!(stage.is_enabled(&config, CompressionLevel::Low));
    }
}

// ─── Property-Based Tests ─────────────────────────────────────────────────────

#[cfg(test)]
mod property_tests {
    use super::*;
    use proptest::prelude::*;
    use serde_json::{json, Map, Value};

    // ─── Strategies ──────────────────────────────────────────────────────────

    /// Generate a JSON Schema object with `title`, `additionalProperties: false`,
    /// and empty `description` fields injected at random depths (1-5).
    fn schema_with_removable_fields() -> impl Strategy<Value = Value> {
        // Generate depth 1-5, then build a nested schema
        (1u32..=5).prop_flat_map(|depth| {
            (
                prop::collection::vec("[a-z]{3,8}", 1..=3usize),
                prop::bool::ANY,
                prop::bool::ANY,
                prop::bool::ANY,
                Just(depth),
            )
                .prop_map(
                    move |(field_names, inject_title, inject_add_props, inject_empty_desc, depth)| {
                        build_nested_schema(
                            &field_names,
                            depth,
                            inject_title,
                            inject_add_props,
                            inject_empty_desc,
                        )
                    },
                )
        })
    }

    fn build_nested_schema(
        field_names: &[String],
        depth: u32,
        inject_title: bool,
        inject_add_props: bool,
        inject_empty_desc: bool,
    ) -> Value {
        let mut obj = Map::new();
        obj.insert("type".to_string(), json!("object"));

        if inject_title {
            obj.insert("title".to_string(), json!("SomeTitle"));
        }
        if inject_add_props {
            obj.insert("additionalProperties".to_string(), json!(false));
        }
        if inject_empty_desc {
            obj.insert("description".to_string(), json!(""));
        }

        if depth > 1 {
            let mut props = Map::new();
            for name in field_names {
                let inner = build_nested_schema(
                    field_names,
                    depth - 1,
                    inject_title,
                    inject_add_props,
                    inject_empty_desc,
                );
                props.insert(name.clone(), inner);
            }
            obj.insert("properties".to_string(), Value::Object(props));
        } else {
            let mut props = Map::new();
            for name in field_names {
                let mut leaf = Map::new();
                leaf.insert("type".to_string(), json!("string"));
                if inject_title {
                    leaf.insert("title".to_string(), json!("LeafTitle"));
                }
                if inject_empty_desc {
                    leaf.insert("description".to_string(), json!("   "));
                }
                props.insert(name.clone(), Value::Object(leaf));
            }
            obj.insert("properties".to_string(), Value::Object(props));
        }

        Value::Object(obj)
    }

    /// Generate a schema wrapped in a single-element `anyOf` or `oneOf`.
    fn single_union_schema() -> impl Strategy<Value = Value> {
        let inner_type = prop_oneof![
            Just(json!({"type": "string"})),
            Just(json!({"type": "integer"})),
            Just(json!({"type": "boolean"})),
            Just(json!({"type": "number", "format": "double"})),
            Just(json!({"type": "object", "properties": {"x": {"type": "string"}}})),
        ];

        (inner_type, prop::bool::ANY).prop_map(|(inner, use_any_of)| {
            let key = if use_any_of { "anyOf" } else { "oneOf" };
            json!({ key: [inner] })
        })
    }

    /// Generate an arbitrary tool-like schema for idempotence testing.
    fn arbitrary_schema() -> impl Strategy<Value = Value> {
        prop_oneof![
            // Simple schemas with removable fields
            schema_with_removable_fields(),
            // Single-union wrappers
            single_union_schema(),
            // Schemas with nullable patterns
            Just(json!({
                "type": "object",
                "title": "Nullable",
                "properties": {
                    "value": {
                        "anyOf": [{"type": "string"}, {"type": "null"}]
                    }
                }
            })),
            // Clean schemas (nothing to remove)
            Just(json!({
                "type": "object",
                "required": ["id"],
                "properties": {
                    "id": {"type": "integer"},
                    "name": {"type": "string", "description": "A name"}
                }
            })),
            // Multi-element anyOf (should NOT be collapsed)
            Just(json!({
                "anyOf": [{"type": "string"}, {"type": "integer"}]
            })),
            // Nested with mixed removable/non-removable
            Just(json!({
                "type": "object",
                "title": "Root",
                "additionalProperties": false,
                "properties": {
                    "nested": {
                        "type": "object",
                        "title": "Nested",
                        "description": "",
                        "properties": {
                            "leaf": {
                                "oneOf": [{"type": "boolean"}],
                                "title": "Leaf"
                            }
                        }
                    }
                }
            })),
        ]
    }

    // ─── Helper: recursively check no field with given key exists ─────────

    fn assert_no_field_anywhere(value: &Value, field_name: &str) -> bool {
        match value {
            Value::Object(map) => {
                if map.contains_key(field_name) {
                    return false;
                }
                map.values().all(|v| assert_no_field_anywhere(v, field_name))
            }
            Value::Array(arr) => arr.iter().all(|v| assert_no_field_anywhere(v, field_name)),
            _ => true,
        }
    }

    fn has_additional_properties_false(value: &Value) -> bool {
        match value {
            Value::Object(map) => {
                if let Some(Value::Bool(false)) = map.get("additionalProperties") {
                    return true;
                }
                map.values().any(|v| has_additional_properties_false(v))
            }
            Value::Array(arr) => arr.iter().any(|v| has_additional_properties_false(v)),
            _ => false,
        }
    }

    fn has_empty_description(value: &Value) -> bool {
        match value {
            Value::Object(map) => {
                if let Some(Value::String(desc)) = map.get("description") {
                    if desc.trim().is_empty() {
                        return true;
                    }
                }
                map.values().any(|v| has_empty_description(v))
            }
            Value::Array(arr) => arr.iter().any(|v| has_empty_description(v)),
            _ => false,
        }
    }

    // ─── Property 2: Minification Field Removal Completeness ──────────────
    // Feature: tool-definition-compression
    // **Validates: Requirements 1.1, 1.2, 1.3**

    proptest! {
        #![proptest_config(ProptestConfig::with_cases(200))]

        #[test]
        fn prop_field_removal_completeness(schema in schema_with_removable_fields()) {
            let caps = ProviderCaps::conservative();
            let mut value = schema;
            minify_value(&mut value, &caps);

            // No `title` fields remain anywhere in the tree
            prop_assert!(
                assert_no_field_anywhere(&value, "title"),
                "title field found after minification: {:?}", value
            );

            // No `additionalProperties: false` remains
            prop_assert!(
                !has_additional_properties_false(&value),
                "additionalProperties: false found after minification: {:?}", value
            );

            // No empty/whitespace descriptions remain
            prop_assert!(
                !has_empty_description(&value),
                "empty description found after minification: {:?}", value
            );
        }
    }

    // ─── Property 3: Single-Union Collapse Equivalence ────────────────────
    // Feature: tool-definition-compression
    // **Validates: Requirements 1.4, 1.5**

    proptest! {
        #![proptest_config(ProptestConfig::with_cases(200))]

        #[test]
        fn prop_single_union_collapse_equivalence(schema in single_union_schema()) {
            let caps = ProviderCaps::conservative();

            // Extract the expected inner type before minification
            let inner = if let Some(arr) = schema.get("anyOf").or_else(|| schema.get("oneOf")) {
                arr.as_array().unwrap()[0].clone()
            } else {
                unreachable!("strategy always produces anyOf or oneOf");
            };

            let mut value = schema.clone();
            minify_value(&mut value, &caps);

            // After collapse, no anyOf/oneOf wrapper should remain
            prop_assert!(
                value.get("anyOf").is_none(),
                "anyOf still present after collapse: {:?}", value
            );
            prop_assert!(
                value.get("oneOf").is_none(),
                "oneOf still present after collapse: {:?}", value
            );

            // The inner type's fields should be promoted to the top level
            // (after also minifying the inner type)
            let mut expected_inner = inner.clone();
            minify_value(&mut expected_inner, &caps);

            // All fields from the expected inner should be present in the result
            if let Value::Object(expected_map) = &expected_inner {
                for (k, v) in expected_map {
                    prop_assert_eq!(
                        value.get(k),
                        Some(v),
                        "field '{}' not promoted correctly", k
                    );
                }
            }
        }
    }

    // ─── Property 4: Minification Determinism (Idempotence) ───────────────
    // Feature: tool-definition-compression
    // **Validates: Requirements 1.9**

    proptest! {
        #![proptest_config(ProptestConfig::with_cases(200))]

        #[test]
        fn prop_minification_idempotence(schema in arbitrary_schema()) {
            let caps = ProviderCaps {
                supports_nullable: true,
                ..ProviderCaps::conservative()
            };

            // First pass
            let mut first_pass = schema.clone();
            minify_value(&mut first_pass, &caps);

            // Second pass
            let mut second_pass = first_pass.clone();
            minify_value(&mut second_pass, &caps);

            // Output after second pass must be identical to after first pass
            prop_assert_eq!(
                &second_pass,
                &first_pass,
                "minification is not idempotent!\nAfter 1st: {:?}\nAfter 2nd: {:?}",
                first_pass,
                second_pass
            );
        }
    }
}
