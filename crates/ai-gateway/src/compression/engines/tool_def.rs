//! Tool-definition description compression with schema-preserving caching.

use super::{CompressiblePayload, CompressionContext, CompressionEngine, EngineResult};
use crate::description_utils::compress_description_fields;
use async_trait::async_trait;
use ring::digest;
use serde_json::Value;
use std::{
    collections::{hash_map::Entry, HashMap, HashSet},
    sync::{Arc, RwLock},
    time::Instant,
};

type ToolDefinitionHash = [u8; digest::SHA256_OUTPUT_LEN];
type ToolDefinitionCache = RwLock<HashMap<ToolDefinitionHash, Arc<Value>>>;

/// Tool-specific statistics returned alongside the shared engine result.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub struct ToolDefinitionCompressionReport {
    pub tool_definitions_tokens_saved: u32,
    pub cache_hit: bool,
}

/// Compresses only string-valued `description` fields in tool definitions.
///
/// Compressed tool sets are cached by the SHA-256 digest of their original JSON.
/// Provider names in `strict_schema_providers` are compared exactly and bypass
/// both compression and cache population.
#[derive(Debug)]
pub struct ToolDefinitionEngine {
    strict_schema_providers: HashSet<String>,
    cache: ToolDefinitionCache,
}

impl ToolDefinitionEngine {
    /// Creates an engine with an exact-name strict-schema provider allowlist.
    pub fn new<I, S>(strict_schema_providers: I) -> Self
    where
        I: IntoIterator<Item = S>,
        S: Into<String>,
    {
        Self {
            strict_schema_providers: strict_schema_providers
                .into_iter()
                .map(Into::into)
                .collect(),
            cache: RwLock::new(HashMap::new()),
        }
    }

    /// Compresses a payload and returns tool-specific savings for pipeline stats.
    pub async fn compress_with_report(
        &self,
        payload: &mut CompressiblePayload,
        context: &CompressionContext,
    ) -> (EngineResult, ToolDefinitionCompressionReport) {
        let started = Instant::now();
        let tokens_before = count_payload_tokens(payload, context);

        // Bypass: if the tool compression middleware already processed this
        // request, skip description compression to avoid double-processing.
        if context.tool_compression_applied {
            return (
                engine_result(started, tokens_before, tokens_before, false),
                ToolDefinitionCompressionReport::default(),
            );
        }

        let skipped = self
            .strict_schema_providers
            .contains(&context.provider_name)
            || payload
                .tool_definitions
                .as_ref()
                .is_none_or(tool_set_is_empty);

        if skipped {
            return (
                engine_result(started, tokens_before, tokens_before, false),
                ToolDefinitionCompressionReport::default(),
            );
        }

        let original_tools = payload
            .tool_definitions
            .as_ref()
            .expect("non-empty tool definitions were checked")
            .clone();
        let hash = hash_tool_definitions(&original_tools);
        let (compressed_tools, cache_hit) = self.cached_or_compress(hash, &original_tools);
        let changed = compressed_tools.as_ref() != &original_tools;

        if changed {
            payload.tool_definitions = Some(compressed_tools.as_ref().clone());
        }

        let candidate_tokens = count_payload_tokens(payload, context);
        if candidate_tokens > tokens_before {
            payload.tool_definitions = Some(original_tools);
            return (
                engine_result(started, tokens_before, tokens_before, false),
                ToolDefinitionCompressionReport {
                    tool_definitions_tokens_saved: 0,
                    cache_hit,
                },
            );
        }

        let tokens_saved = tokens_before.saturating_sub(candidate_tokens);
        (
            engine_result(started, tokens_before, candidate_tokens, changed),
            ToolDefinitionCompressionReport {
                tool_definitions_tokens_saved: tokens_saved,
                cache_hit,
            },
        )
    }

    /// Returns the number of distinct original tool sets currently cached.
    pub fn cached_tool_set_count(&self) -> usize {
        self.cache
            .read()
            .unwrap_or_else(std::sync::PoisonError::into_inner)
            .len()
    }

    fn cached_or_compress(&self, hash: ToolDefinitionHash, original: &Value) -> (Arc<Value>, bool) {
        if let Some(cached) = self
            .cache
            .read()
            .unwrap_or_else(std::sync::PoisonError::into_inner)
            .get(&hash)
            .cloned()
        {
            return (cached, true);
        }

        let mut compressed = original.clone();
        compress_description_fields(&mut compressed);
        let computed = Arc::new(compressed);
        let mut cache = self
            .cache
            .write()
            .unwrap_or_else(std::sync::PoisonError::into_inner);
        match cache.entry(hash) {
            Entry::Occupied(entry) => (Arc::clone(entry.get()), true),
            Entry::Vacant(entry) => {
                entry.insert(Arc::clone(&computed));
                (computed, false)
            }
        }
    }
}

impl Default for ToolDefinitionEngine {
    fn default() -> Self {
        Self::new(std::iter::empty::<String>())
    }
}

#[async_trait]
impl CompressionEngine for ToolDefinitionEngine {
    fn name(&self) -> &str {
        "tool_definitions"
    }

    async fn compress(
        &self,
        payload: &mut CompressiblePayload,
        context: &CompressionContext,
    ) -> EngineResult {
        self.compress_with_report(payload, context).await.0
    }
}

fn engine_result(
    started: Instant,
    tokens_before: u32,
    tokens_after: u32,
    applied: bool,
) -> EngineResult {
    EngineResult {
        engine_name: "tool_definitions".to_owned(),
        tokens_before,
        tokens_after,
        duration_ms: started.elapsed().as_millis().min(u64::MAX as u128) as u64,
        applied,
    }
}

fn count_payload_tokens(payload: &CompressiblePayload, context: &CompressionContext) -> u32 {
    context
        .token_counter
        .count_request(&payload.clone().into_openai_request())
}

fn tool_set_is_empty(tools: &Value) -> bool {
    tools.is_null() || tools.as_array().is_some_and(Vec::is_empty)
}

fn hash_tool_definitions(tools: &Value) -> ToolDefinitionHash {
    let serialized = serde_json::to_vec(tools).expect("serde_json::Value must serialize");
    let digest = digest::digest(&digest::SHA256, &serialized);
    let mut hash = [0; digest::SHA256_OUTPUT_LEN];
    hash.copy_from_slice(digest.as_ref());
    hash
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::models::openai::OpenAIRequest;
    use proptest::prelude::*;
    use serde_json::{json, Map};

    const LITERAL_SCHEMA_FIELDS: [&str; 5] = ["const", "default", "enum", "example", "examples"];

    fn payload(tools: Option<Value>) -> CompressiblePayload {
        let mut extra = Map::new();
        if let Some(tools) = tools {
            extra.insert("tools".to_owned(), tools);
        }
        CompressiblePayload::from(OpenAIRequest {
            model: "gpt-4o".to_owned(),
            messages: Vec::new(),
            stream: false,
            temperature: None,
            max_tokens: None,
            extra,
        })
    }

    fn context(provider: &str) -> CompressionContext {
        CompressionContext::new("gpt-4o", provider)
    }

    fn verbose_tools() -> Value {
        json!([{
            "type": "function",
            "function": {
                "name": "lookup_customer",
                "description": "This tool can be used to look up a customer by identifier. For example, pass customer_123. Note: archived customers may be unavailable. Look up a customer by identifier.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "customer_id": {
                            "type": "string",
                            "description": "The purpose of this function is to identify the customer record. Example: customer_123.",
                            "enum": ["customer_123", "customer_456"],
                            "default": "customer_123",
                            "const": "customer_123",
                            "examples": ["customer_123"]
                        }
                    },
                    "required": ["customer_id"],
                    "additionalProperties": false
                }
            }
        }])
    }

    fn assert_only_description_prose_changed(original: &Value, compressed: &Value) {
        match (original, compressed) {
            (Value::Object(original), Value::Object(compressed)) => {
                assert_eq!(original.len(), compressed.len());
                for (field, original_value) in original {
                    let compressed_value = compressed.get(field).expect("field must be preserved");
                    if field == "description"
                        && original_value.is_string()
                        && compressed_value.is_string()
                    {
                        continue;
                    }
                    if LITERAL_SCHEMA_FIELDS.contains(&field.as_str()) {
                        assert_eq!(compressed_value, original_value);
                    } else {
                        assert_only_description_prose_changed(original_value, compressed_value);
                    }
                }
            }
            (Value::Array(original), Value::Array(compressed)) => {
                assert_eq!(original.len(), compressed.len());
                for (original, compressed) in original.iter().zip(compressed) {
                    assert_only_description_prose_changed(original, compressed);
                }
            }
            _ => assert_eq!(compressed, original),
        }
    }

    #[tokio::test]
    async fn preserves_schema_and_changes_only_description_prose() {
        let original = verbose_tools();
        let mut payload = payload(Some(original.clone()));
        let (result, report) = ToolDefinitionEngine::default()
            .compress_with_report(&mut payload, &context("openai"))
            .await;
        let compressed = payload.tool_definitions.as_ref().unwrap();

        assert!(result.applied);
        assert!(report.tool_definitions_tokens_saved > 0);
        assert_only_description_prose_changed(&original, compressed);
        assert_eq!(compressed[0]["function"]["name"], "lookup_customer");
        assert_eq!(
            compressed[0]["function"]["parameters"]["required"],
            json!(["customer_id"])
        );
        assert_eq!(
            compressed[0]["function"]["parameters"]["properties"]["customer_id"]["type"],
            "string"
        );
    }

    #[tokio::test]
    async fn reports_full_payload_counts_and_description_savings() {
        let tools = verbose_tools();
        let mut payload = payload(Some(tools));
        let context = context("openai");
        let expected_before = count_payload_tokens(&payload, &context);

        let (result, report) = ToolDefinitionEngine::default()
            .compress_with_report(&mut payload, &context)
            .await;
        let expected_after = count_payload_tokens(&payload, &context);

        assert_eq!(result.tokens_before, expected_before);
        assert_eq!(result.tokens_after, expected_after);
        assert_eq!(
            report.tool_definitions_tokens_saved,
            expected_before - expected_after
        );
        assert!(expected_after < expected_before);
    }

    #[tokio::test]
    async fn removes_examples_only_from_description_strings() {
        let original = verbose_tools();
        let mut payload = payload(Some(original.clone()));
        ToolDefinitionEngine::default()
            .compress(&mut payload, &context("openai"))
            .await;
        let compressed = payload.tool_definitions.as_ref().unwrap();
        let property = &compressed[0]["function"]["parameters"]["properties"]["customer_id"];

        assert!(!property["description"]
            .as_str()
            .unwrap()
            .contains("Example"));
        assert_eq!(property["examples"], json!(["customer_123"]));
        assert_eq!(property["default"], "customer_123");
        assert_eq!(property["const"], "customer_123");
        assert_eq!(property["enum"], json!(["customer_123", "customer_456"]));
    }

    #[tokio::test]
    async fn reuses_sha256_cached_tool_sets_with_stable_hashes() {
        let original = verbose_tools();
        let engine = ToolDefinitionEngine::default();
        let first_hash = hash_tool_definitions(&original);
        let second_hash = hash_tool_definitions(&original.clone());
        let mut first = payload(Some(original.clone()));
        let mut second = payload(Some(original));

        let (_, first_report) = engine
            .compress_with_report(&mut first, &context("openai"))
            .await;
        let (_, second_report) = engine
            .compress_with_report(&mut second, &context("openai"))
            .await;

        assert_eq!(first_hash, second_hash);
        assert!(!first_report.cache_hit);
        assert!(second_report.cache_hit);
        assert_eq!(engine.cached_tool_set_count(), 1);
        assert_eq!(first.tool_definitions, second.tool_definitions);
    }

    #[tokio::test]
    async fn exact_strict_provider_name_skips_compression_and_cache() {
        let original = verbose_tools();
        let engine = ToolDefinitionEngine::new(["strict-provider"]);
        let mut strict = payload(Some(original.clone()));
        let (result, report) = engine
            .compress_with_report(&mut strict, &context("strict-provider"))
            .await;

        assert!(!result.applied);
        assert_eq!(strict.tool_definitions, Some(original.clone()));
        assert_eq!(report, ToolDefinitionCompressionReport::default());
        assert_eq!(engine.cached_tool_set_count(), 0);

        let mut similar = payload(Some(original));
        let (result, _) = engine
            .compress_with_report(&mut similar, &context("Strict-Provider"))
            .await;
        assert!(result.applied);
    }

    #[tokio::test]
    async fn missing_null_and_empty_tools_are_no_ops() {
        for tools in [None, Some(Value::Null), Some(json!([]))] {
            let mut payload = payload(tools);
            let original = payload.clone();
            let engine = ToolDefinitionEngine::default();
            let (result, report) = engine
                .compress_with_report(&mut payload, &context("openai"))
                .await;

            assert!(!result.applied);
            assert_eq!(result.tokens_before, result.tokens_after);
            assert_eq!(report, ToolDefinitionCompressionReport::default());
            assert_eq!(payload, original);
            assert_eq!(engine.cached_tool_set_count(), 0);
        }
    }

    #[tokio::test]
    async fn rolls_back_cached_candidate_when_full_payload_tokens_increase() {
        let original = json!([{
            "type": "function",
            "function": {
                "name": "run",
                "description": "Run.",
                "parameters": {"type": "object", "properties": {}}
            }
        }]);
        let mut expanded = original.clone();
        expanded[0]["function"]["description"] =
            Value::String("Expanded description. ".repeat(100));
        let engine = ToolDefinitionEngine::default();
        engine
            .cache
            .write()
            .unwrap()
            .insert(hash_tool_definitions(&original), Arc::new(expanded));
        let mut payload = payload(Some(original.clone()));

        let (result, report) = engine
            .compress_with_report(&mut payload, &context("openai"))
            .await;

        assert!(!result.applied);
        assert_eq!(result.tokens_before, result.tokens_after);
        assert_eq!(report.tool_definitions_tokens_saved, 0);
        assert!(report.cache_hit);
        assert_eq!(payload.tool_definitions, Some(original));
    }

    #[tokio::test]
    async fn compresses_nested_schema_descriptions_without_touching_literals() {
        let original = json!([{
            "type": "function",
            "function": {
                "name": "nested",
                "parameters": {
                    "$defs": {
                        "record": {
                            "type": "object",
                            "description": "This tool can be used to represent a nested record. Example: a nested record.",
                            "properties": {
                                "items": {
                                    "type": "array",
                                    "items": {
                                        "oneOf": [
                                            {"type": "string", "description": "This function is used to hold a textual item. Note: keep it short."},
                                            {"type": "integer", "description": "A numeric item."}
                                        ]
                                    },
                                    "default": {"description": "literal default must remain"},
                                    "examples": [{"description": "literal example must remain"}]
                                }
                            },
                            "required": ["items"]
                        }
                    },
                    "$ref": "#/$defs/record"
                }
            }
        }]);
        let mut payload = payload(Some(original.clone()));
        ToolDefinitionEngine::default()
            .compress(&mut payload, &context("openai"))
            .await;
        let compressed = payload.tool_definitions.as_ref().unwrap();

        assert_only_description_prose_changed(&original, compressed);
        assert_eq!(
            compressed[0]["function"]["parameters"]["$defs"]["record"]["properties"]["items"]
                ["default"],
            json!({"description": "literal default must remain"})
        );
        assert_eq!(
            compressed[0]["function"]["parameters"]["$defs"]["record"]["properties"]["items"]
                ["examples"],
            json!([{"description": "literal example must remain"}])
        );
    }

    fn schema_strategy() -> impl Strategy<Value = Value> {
        (
            "[a-z][a-z0-9_]{0,15}",
            proptest::collection::vec("[a-z][a-z0-9_]{0,12}", 1..6),
            proptest::collection::vec(any::<bool>(), 1..6),
        )
            .prop_map(|(name, property_names, required_flags)| {
                let mut properties = Map::new();
                let mut required = Vec::new();
                for (index, property_name) in property_names.into_iter().enumerate() {
                    let property_type = if index % 2 == 0 { "string" } else { "integer" };
                    properties.insert(
                        property_name.clone(),
                        json!({
                            "type": property_type,
                            "description": "This function can be used to provide a value. Example: a sample value. Note: examples are illustrative.",
                            "enum": ["alpha", "beta"],
                            "default": {"description": "literal"}
                        }),
                    );
                    if required_flags[index % required_flags.len()] {
                        required.push(property_name);
                    }
                }
                json!([{
                    "type": "function",
                    "function": {
                        "name": name,
                        "description": "This tool allows you to perform the generated operation. For example, call it with generated values. Warning: examples are illustrative.",
                        "parameters": {
                            "type": "object",
                            "properties": properties,
                            "required": required,
                            "additionalProperties": false
                        }
                    }
                }])
            })
    }

    proptest! {
        #![proptest_config(ProptestConfig::with_cases(128))]

        #[test]
        fn property_13_foundation_preserves_every_non_description_value(tools in schema_strategy()) {
            let mut compressed = tools.clone();
            compress_description_fields(&mut compressed);

            assert_only_description_prose_changed(&tools, &compressed);
            prop_assert_eq!(hash_tool_definitions(&tools), hash_tool_definitions(&tools.clone()));
        }
    }
}
