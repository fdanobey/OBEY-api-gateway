//! Schema Deduplicator stage — replaces duplicate parameter schemas with `$ref` references.
//!
//! 1. Normalizes each schema (sort keys alphabetically, strip already-removed fields)
//! 2. Computes a 64-bit hash of the canonical JSON serialization
//! 3. Groups schemas by hash — groups of size ≥ 2 are duplicates
//! 4. For each duplicate group: emits the schema once as a `$defs` entry,
//!    replaces occurrences with `{"$ref": "#/$defs/<hash_hex>"}`
//! 5. Checks net token savings — skips deduplication if reference overhead exceeds inline savings
//! 6. Only processes top-level parameters and depth-1 properties (not deeper)
//! 7. When provider does not support `$ref`, this stage is a no-op

use std::collections::hash_map::DefaultHasher;
use std::collections::{BTreeMap, HashMap};
use std::hash::{Hash, Hasher};

use serde_json::{Map, Value};

use crate::tool_compression::config::{CompressionLevel, ToolCompressionConfig};
use crate::tool_compression::stage::CompressionStage;
use crate::tool_compression::types::{CompressionContext, ToolDefinition};

/// Schema deduplication stage.
///
/// Identifies identical parameter schemas across tools and replaces duplicates
/// with JSON Schema `$ref` / `$defs` references to save tokens.
pub struct SchemaDeduplicator;

impl CompressionStage for SchemaDeduplicator {
    fn apply(&self, tools: &mut Vec<ToolDefinition>, ctx: &mut CompressionContext) -> u64 {
        // No-op when provider does not support $ref
        if !ctx.provider_caps.supports_ref {
            return 0;
        }

        // Collect schemas: (tool_index, property_key, canonical_json, hash, token_count)
        let mut schema_entries: Vec<SchemaEntry> = Vec::new();

        for (tool_idx, tool) in tools.iter().enumerate() {
            let properties = match tool
                .raw
                .pointer("/function/parameters/properties")
                .and_then(|v| v.as_object())
            {
                Some(props) => props,
                None => continue,
            };

            for (key, schema) in properties.iter() {
                // Only process top-level parameters and depth-1 properties
                let canonical = canonicalize(schema);
                let hash = compute_hash(&canonical);
                let token_estimate = estimate_tokens(&canonical);
                schema_entries.push(SchemaEntry {
                    tool_idx,
                    property_key: key.clone(),
                    canonical,
                    hash,
                    token_estimate,
                });
            }
        }

        // Group by hash — only groups with size ≥ 2 are candidates
        let mut groups: HashMap<u64, Vec<usize>> = HashMap::new();
        for (idx, entry) in schema_entries.iter().enumerate() {
            groups.entry(entry.hash).or_default().push(idx);
        }

        let mut total_saved: u64 = 0;
        let mut defs_to_inject: HashMap<String, Value> = HashMap::new();

        for (_hash, indices) in &groups {
            if indices.len() < 2 {
                continue;
            }

            let representative = &schema_entries[indices[0]];
            let inline_tokens = representative.token_estimate;
            let num_duplicates = indices.len() as u64;

            // Net savings calculation:
            // $ref reference ≈ 10 tokens overhead per reference
            // $defs entry ≈ inline_tokens + ~5 tokens overhead (key + structure)
            let ref_overhead_per_use: u64 = 10;
            let defs_entry_overhead: u64 = inline_tokens + 5;
            let total_without_dedup = num_duplicates * inline_tokens;
            let total_with_dedup = defs_entry_overhead + num_duplicates * ref_overhead_per_use;

            if total_with_dedup >= total_without_dedup {
                // Net savings ≤ 0, skip deduplication for this group
                continue;
            }

            let net_savings = total_without_dedup - total_with_dedup;
            let hash_hex = format!("{:016x}", representative.hash);
            let ref_value = Value::String(format!("#/$defs/{}", hash_hex));

            // Parse the canonical JSON back to a Value for $defs
            let schema_value: Value =
                serde_json::from_str(&representative.canonical).unwrap_or(Value::Null);
            defs_to_inject.insert(hash_hex.clone(), schema_value);

            // Replace all occurrences with $ref
            for &entry_idx in indices {
                let entry = &schema_entries[entry_idx];
                if let Some(props) = tools[entry.tool_idx]
                    .raw
                    .pointer_mut("/function/parameters/properties")
                    .and_then(|v| v.as_object_mut())
                {
                    let ref_obj = serde_json::json!({ "$ref": ref_value });
                    props.insert(entry.property_key.clone(), ref_obj);
                }
            }

            total_saved += net_savings;
        }

        // Inject $defs into each tool that has replaced schemas
        if !defs_to_inject.is_empty() {
            let defs_value =
                Value::Object(defs_to_inject.into_iter().collect::<Map<String, Value>>());

            for tool in tools.iter_mut() {
                if let Some(params) = tool
                    .raw
                    .pointer_mut("/function/parameters")
                    .and_then(|v| v.as_object_mut())
                {
                    // Only inject $defs if this tool has at least one $ref
                    let has_ref = params
                        .get("properties")
                        .and_then(|p| p.as_object())
                        .map(|props| {
                            props
                                .values()
                                .any(|v| v.as_object().and_then(|obj| obj.get("$ref")).is_some())
                        })
                        .unwrap_or(false);

                    if has_ref {
                        params.insert("$defs".to_string(), defs_value.clone());
                    }
                }
            }
        }

        if total_saved > 0 {
            ctx.strategies_applied
                .push("schema_deduplicator".to_string());
        }
        ctx.tokens_saved += total_saved;
        total_saved
    }

    fn is_enabled(&self, config: &ToolCompressionConfig, level: CompressionLevel) -> bool {
        // Enabled at Medium level and above when config.deduplication is true
        config.deduplication
            && matches!(
                level,
                CompressionLevel::Medium | CompressionLevel::High | CompressionLevel::Max
            )
    }
}

// ─── Internal types ───────────────────────────────────────────────────────────

struct SchemaEntry {
    tool_idx: usize,
    property_key: String,
    canonical: String,
    hash: u64,
    token_estimate: u64,
}

// ─── Internal helpers ─────────────────────────────────────────────────────────

/// Estimate token count as character_count / 4 (matching other stages).
fn estimate_tokens(s: &str) -> u64 {
    (s.len() as u64) / 4
}

/// Compute a 64-bit hash of a canonical JSON string using DefaultHasher.
fn compute_hash(canonical: &str) -> u64 {
    let mut hasher = DefaultHasher::new();
    canonical.hash(&mut hasher);
    hasher.finish()
}

/// Canonicalize a JSON value: sort all object keys alphabetically, produce
/// compact JSON. This ensures structurally identical schemas produce the same hash.
fn canonicalize(value: &Value) -> String {
    let normalized = normalize_value(value);
    // serde_json::to_string produces compact JSON without extra whitespace
    serde_json::to_string(&normalized).unwrap_or_default()
}

/// Recursively normalize a JSON value by sorting object keys into a BTreeMap.
fn normalize_value(value: &Value) -> Value {
    match value {
        Value::Object(map) => {
            let sorted: BTreeMap<&str, Value> = map
                .iter()
                .map(|(k, v)| (k.as_str(), normalize_value(v)))
                .collect();
            // Convert BTreeMap back to a serde_json Value (maintains insertion order)
            Value::Object(
                sorted
                    .into_iter()
                    .map(|(k, v)| (k.to_string(), v))
                    .collect(),
            )
        }
        Value::Array(arr) => Value::Array(arr.iter().map(normalize_value).collect()),
        other => other.clone(),
    }
}

// ─── Tests ────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;
    use crate::tool_compression::types::ProviderCaps;
    use serde_json::json;

    fn make_tool(name: &str, raw: Value) -> ToolDefinition {
        ToolDefinition {
            raw,
            name: name.to_string(),
            content_hash: 0,
        }
    }

    fn ctx_with_ref_support(supports_ref: bool) -> CompressionContext {
        CompressionContext {
            level: CompressionLevel::Medium,
            provider_caps: ProviderCaps {
                supports_ref,
                ..ProviderCaps::conservative()
            },
            ..Default::default()
        }
    }

    #[test]
    fn noop_when_provider_does_not_support_ref() {
        let raw = json!({
            "type": "function",
            "function": {
                "name": "tool_a",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "id": { "type": "string", "description": "identifier" }
                    }
                }
            }
        });
        let mut tools = vec![make_tool("tool_a", raw.clone()), make_tool("tool_b", raw)];
        let mut ctx = ctx_with_ref_support(false);

        let stage = SchemaDeduplicator;
        let saved = stage.apply(&mut tools, &mut ctx);

        assert_eq!(saved, 0);
        assert!(!ctx
            .strategies_applied
            .contains(&"schema_deduplicator".to_string()));
    }

    #[test]
    fn deduplicates_identical_schemas_across_tools() {
        let schema = json!({ "type": "string", "description": "A shared param schema that is long enough to benefit from dedup" });
        let tool_a = json!({
            "type": "function",
            "function": {
                "name": "tool_a",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "shared": schema.clone()
                    }
                }
            }
        });
        let tool_b = json!({
            "type": "function",
            "function": {
                "name": "tool_b",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "shared": schema.clone()
                    }
                }
            }
        });
        let tool_c = json!({
            "type": "function",
            "function": {
                "name": "tool_c",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "shared": schema.clone()
                    }
                }
            }
        });

        let mut tools = vec![
            make_tool("tool_a", tool_a),
            make_tool("tool_b", tool_b),
            make_tool("tool_c", tool_c),
        ];
        let mut ctx = ctx_with_ref_support(true);

        let stage = SchemaDeduplicator;
        let saved = stage.apply(&mut tools, &mut ctx);

        assert!(saved > 0);
        assert!(ctx
            .strategies_applied
            .contains(&"schema_deduplicator".to_string()));

        // Each tool should have $defs and a $ref in the shared property
        for tool in &tools {
            let props = tool
                .raw
                .pointer("/function/parameters/properties/shared")
                .unwrap();
            assert!(
                props.get("$ref").is_some(),
                "Expected $ref in tool {}",
                tool.name
            );

            let defs = tool.raw.pointer("/function/parameters/$defs").unwrap();
            assert!(defs.as_object().unwrap().len() == 1);
        }
    }

    #[test]
    fn does_not_deduplicate_unique_schemas() {
        let tool_a = json!({
            "type": "function",
            "function": {
                "name": "tool_a",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "name": { "type": "string" }
                    }
                }
            }
        });
        let tool_b = json!({
            "type": "function",
            "function": {
                "name": "tool_b",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "count": { "type": "integer" }
                    }
                }
            }
        });

        let mut tools = vec![make_tool("tool_a", tool_a), make_tool("tool_b", tool_b)];
        let mut ctx = ctx_with_ref_support(true);

        let stage = SchemaDeduplicator;
        let saved = stage.apply(&mut tools, &mut ctx);

        assert_eq!(saved, 0);
        // No $defs should be injected
        for tool in &tools {
            assert!(tool.raw.pointer("/function/parameters/$defs").is_none());
        }
    }

    #[test]
    fn skips_dedup_when_overhead_exceeds_savings() {
        // Very small schema: $ref overhead would exceed inline cost
        let schema = json!({ "type": "string" });
        let tool_a = json!({
            "type": "function",
            "function": {
                "name": "tool_a",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "x": schema.clone()
                    }
                }
            }
        });
        let tool_b = json!({
            "type": "function",
            "function": {
                "name": "tool_b",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "x": schema.clone()
                    }
                }
            }
        });

        let mut tools = vec![make_tool("tool_a", tool_a), make_tool("tool_b", tool_b)];
        let mut ctx = ctx_with_ref_support(true);

        let stage = SchemaDeduplicator;
        let saved = stage.apply(&mut tools, &mut ctx);

        // Small schema: 2 * inline_tokens vs (inline_tokens + 5) + 2 * 10
        // {"type":"string"} is 15 chars → 3 tokens. 2*3=6 vs (3+5)+2*10=28. Skip.
        assert_eq!(saved, 0);
        for tool in &tools {
            assert!(tool.raw.pointer("/function/parameters/$defs").is_none());
        }
    }

    #[test]
    fn canonicalization_sorts_keys() {
        let a = json!({ "type": "string", "description": "hello" });
        let b = json!({ "description": "hello", "type": "string" });
        assert_eq!(canonicalize(&a), canonicalize(&b));
    }

    #[test]
    fn is_enabled_medium_and_above_with_dedup_config() {
        let stage = SchemaDeduplicator;
        let mut config = ToolCompressionConfig::default();
        config.deduplication = true;

        assert!(!stage.is_enabled(&config, CompressionLevel::Low));
        assert!(stage.is_enabled(&config, CompressionLevel::Medium));
        assert!(stage.is_enabled(&config, CompressionLevel::High));
        assert!(stage.is_enabled(&config, CompressionLevel::Max));
    }

    #[test]
    fn is_disabled_when_config_dedup_false() {
        let stage = SchemaDeduplicator;
        let mut config = ToolCompressionConfig::default();
        config.deduplication = false;

        assert!(!stage.is_enabled(&config, CompressionLevel::Medium));
        assert!(!stage.is_enabled(&config, CompressionLevel::High));
    }

    #[test]
    fn normalize_produces_deterministic_output() {
        let v1 = json!({"b": 2, "a": 1, "c": {"z": true, "y": false}});
        let v2 = json!({"c": {"y": false, "z": true}, "a": 1, "b": 2});
        assert_eq!(canonicalize(&v1), canonicalize(&v2));
    }
}

#[cfg(test)]
mod property_tests {
    use super::*;
    use proptest::prelude::*;
    use serde_json::{json, Map, Value};

    // ─── Strategies ──────────────────────────────────────────────────────────

    /// Generate a random JSON leaf value (string, integer, or boolean).
    fn leaf_value() -> impl Strategy<Value = Value> {
        prop_oneof![
            "[a-z]{1,10}".prop_map(|s| json!(s)),
            (-1000i64..1000).prop_map(|n| json!(n)),
            any::<bool>().prop_map(|b| json!(b)),
        ]
    }

    /// Generate a random JSON object with 2-5 keys of type string/integer/boolean.
    fn random_schema_object() -> impl Strategy<Value = Value> {
        prop::collection::vec(("[a-z]{2,6}", leaf_value()), 2..=5usize).prop_map(|entries| {
            let mut map = Map::new();
            for (key, val) in entries {
                map.insert(key, val);
            }
            Value::Object(map)
        })
    }

    /// Generate a pair of JSON objects that are identical in content but have
    /// keys inserted in different (shuffled) order.
    fn shuffled_key_pair() -> impl Strategy<Value = (Value, Value)> {
        prop::collection::vec(("[a-z]{2,6}", leaf_value()), 2..=5usize).prop_flat_map(|entries| {
            // Deduplicate by key (keep first occurrence) to avoid
            // forward/reverse producing different values for the same key.
            let mut seen = std::collections::HashSet::new();
            let unique_entries: Vec<(String, Value)> = entries
                .into_iter()
                .filter(|(k, _)| seen.insert(k.clone()))
                .collect();
            let entries_clone = unique_entries.clone();
            // Create forward-order map
            let forward: Map<String, Value> = unique_entries.into_iter().collect();
            // Create reverse-order map (same keys/values, different insertion order)
            let reversed: Map<String, Value> = entries_clone.into_iter().rev().collect();
            Just((Value::Object(forward), Value::Object(reversed)))
        })
    }

    /// Generate two schemas that differ by at least one key or value.
    fn differing_schema_pair() -> impl Strategy<Value = (Value, Value)> {
        (random_schema_object(), "[a-z]{2,6}", leaf_value()).prop_map(
            |(base, extra_key, extra_val)| {
                let mut modified = base.clone();
                // Add/overwrite a key to ensure difference
                modified
                    .as_object_mut()
                    .unwrap()
                    .insert(format!("_diff_{}", extra_key), extra_val);
                (base, modified)
            },
        )
    }

    // ─── Property 6: Schema Hash Confluence (Order Independence) ─────────────
    // Feature: tool-definition-compression
    // **Validates: Requirements 4.3, 4.6**
    //
    // Strategy:
    // 1. Generate random JSON objects with 2-5 keys of type string/integer/boolean
    // 2. Create a copy with the same keys in shuffled order
    // 3. Call canonicalize() on both → verify hashes are identical
    // 4. Generate a second schema with at least one different value/key
    // 5. Verify the hashes are different

    proptest! {
        #![proptest_config(ProptestConfig::with_cases(64))]

        #[test]
        fn schema_hash_order_independent((ref a, ref b) in shuffled_key_pair()) {
            // Canonicalize both orderings
            let canonical_a = canonicalize(a);
            let canonical_b = canonicalize(b);

            // Canonical forms must be identical
            prop_assert_eq!(&canonical_a, &canonical_b,
                "Canonicalization should produce identical output regardless of key order");

            // Hashes must be identical
            let hash_a = compute_hash(&canonical_a);
            let hash_b = compute_hash(&canonical_b);
            prop_assert_eq!(hash_a, hash_b,
                "Hashes must be identical for structurally equivalent schemas with different key order");
        }

        #[test]
        fn differing_schemas_produce_different_hashes((ref a, ref b) in differing_schema_pair()) {
            let canonical_a = canonicalize(a);
            let canonical_b = canonicalize(b);

            // Canonical forms must differ (schemas are structurally different)
            prop_assert_ne!(&canonical_a, &canonical_b,
                "Differing schemas should have different canonical forms");

            // Hashes must differ (collision possible but extremely unlikely with 64-bit hash
            // and our constrained input space)
            let hash_a = compute_hash(&canonical_a);
            let hash_b = compute_hash(&canonical_b);
            prop_assert_ne!(hash_a, hash_b,
                "Different schemas should produce different hashes (collision extremely unlikely with constrained inputs)");
        }
    }
}
