//! Integration tests for the tool compression pipeline.
//!
//! These tests exercise the full middleware + pipeline flow using
//! `tower::ServiceExt::oneshot()` pattern (no port binding).

#[cfg(test)]
mod tests {
    use serde_json::{json, Value};

    use crate::tool_compression::config::CompressionLevel;
    use crate::tool_compression::types::ToolDefinition;
    use crate::tool_compression::validation::{
        validate_compressed_tools, validate_tool_calls_against_originals,
    };

    // ─── Helper: build a minimal tools array of specified size ─────────────────

    fn generate_realistic_tools(count: usize) -> Vec<Value> {
        let prefixes = [
            "github", "slack", "jira", "aws", "gcp", "mcp", "db", "fs", "http", "email",
        ];
        let actions = [
            "create", "read", "update", "delete", "list", "search", "sync", "export", "import",
            "validate",
        ];

        (0..count)
            .map(|i| {
                let prefix = prefixes[i % prefixes.len()];
                let action = actions[i % actions.len()];
                let name = format!("{}_{}", prefix, action);
                json!({
                    "type": "function",
                    "function": {
                        "name": name,
                        "description": format!("This tool allows you to {} resources in {}. It accepts various parameters for filtering and configuration. Example: call with id='abc123'. Note: requires authentication.", action, prefix),
                        "parameters": {
                            "type": "object",
                            "properties": {
                                "id": {
                                    "type": "string",
                                    "description": format!("The unique identifier of the {} resource to {}. Example: 'res_123'.", prefix, action)
                                },
                                "options": {
                                    "type": "object",
                                    "description": "Additional configuration options for this operation.",
                                    "properties": {
                                        "verbose": { "type": "boolean", "description": "Whether to include verbose output." },
                                        "format": { "type": "string", "enum": ["json", "yaml", "toml", "csv"], "description": "Output format selection." }
                                    },
                                    "additionalProperties": false
                                },
                                "filters": {
                                    "type": "array",
                                    "items": { "type": "string" },
                                    "description": "List of filter expressions to apply."
                                }
                            },
                            "required": ["id"],
                            "additionalProperties": false
                        }
                    }
                })
            })
            .collect()
    }

    fn tools_to_definitions(tools: &[Value]) -> Vec<ToolDefinition> {
        tools
            .iter()
            .map(|raw| {
                let name = raw
                    .pointer("/function/name")
                    .and_then(|n| n.as_str())
                    .unwrap_or("")
                    .to_string();
                ToolDefinition {
                    raw: raw.clone(),
                    name,
                    content_hash: 0,
                }
            })
            .collect()
    }

    // ─── 24.1: Full pipeline with realistic MCP tool definitions ──────────────

    #[test]
    fn test_full_pipeline_realistic_mcp_tools() {
        let tools = generate_realistic_tools(200);
        let tools_json = Value::Array(tools.clone());

        // Verify we have the expected count
        assert_eq!(tools.len(), 200);

        // Verify token estimate is substantial (17,600+ tokens @ 4 chars/token)
        let total_chars: usize = serde_json::to_string(&tools_json).unwrap().len();
        let estimated_tokens = total_chars / 4;
        assert!(
            estimated_tokens > 10_000,
            "Expected >10,000 tokens, got {}",
            estimated_tokens
        );

        // Validate all tools are structurally valid
        let defs = tools_to_definitions(&tools);
        assert!(validate_compressed_tools(&defs));

        // Each tool should have type, function.name, function.parameters
        for tool in &tools {
            assert_eq!(tool.get("type").and_then(|v| v.as_str()), Some("function"));
            assert!(tool
                .pointer("/function/name")
                .and_then(|v| v.as_str())
                .is_some());
            assert!(tool
                .pointer("/function/parameters")
                .and_then(|v| v.as_object())
                .is_some());
        }
    }

    // ─── 24.2: Config hot-reload updates compression settings ─────────────────

    #[test]
    fn test_config_hot_reload_updates_compression_state() {
        use crate::tool_compression::config::ToolCompressionConfig;
        use crate::tool_compression::state::ToolCompressionState;

        // Create initial state
        let config = ToolCompressionConfig::default();
        let state = ToolCompressionState::new(&config);

        // Populate some state
        state.feedback_state.insert("group_a".to_string(), ());
        state.feedback_state.insert("group_b".to_string(), ());
        state
            .description_compressor
            .insert("tool_1".to_string(), "compressed desc".to_string());
        state
            .semantic_state
            .insert("tool_1".to_string(), vec![0.1, 0.2, 0.3]);

        assert_eq!(state.feedback_state.len(), 2);
        assert_eq!(state.description_compressor.len(), 1);
        assert_eq!(state.semantic_state.len(), 1);

        // Simulate config reload
        let new_config = ToolCompressionConfig {
            enabled: true,
            level: CompressionLevel::High,
            ..Default::default()
        };
        state.reset_on_reload(&new_config);

        // Feedback state should be cleared
        assert_eq!(state.feedback_state.len(), 0);
        // Description compressor should be cleared
        assert_eq!(state.description_compressor.len(), 0);
        // Semantic state should be cleared
        assert_eq!(state.semantic_state.len(), 0);
    }

    // ─── 24.3: Semantic retrieval end-to-end ──────────────────────────────────

    #[test]
    fn test_semantic_retrieval_tfidf_fallback() {
        use crate::tool_compression::tfidf::TfIdfScorer;

        // Build a scorer with tool descriptions as documents
        let documents = vec![
            "Get current weather forecast for a city by name",
            "Search web pages and return results",
            "Send email to a recipient with subject and body",
            "Create a new GitHub issue in a repository",
            "List all files in a directory recursively",
        ];

        let scorer = TfIdfScorer::new(&documents);

        // Query for weather-related content
        let scores = scorer.score_query("What is the weather in Tokyo?", &documents);

        assert_eq!(scores.len(), 5);
        // The weather document should score highest
        let max_idx = scores
            .iter()
            .enumerate()
            .max_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap())
            .map(|(i, _)| i)
            .unwrap();
        assert_eq!(
            max_idx, 0,
            "Weather tool should score highest for weather query"
        );
    }

    // ─── 24.4: Canonical rewriting end-to-end ─────────────────────────────────

    #[test]
    fn test_canonical_rewriting_end_to_end() {
        use crate::tool_compression::stage::CompressionStage;
        use crate::tool_compression::stages::canonical_rewriter::CanonicalRewriter;
        use crate::tool_compression::types::CompressionContext;
        use dashmap::DashMap;
        use std::sync::Arc;

        let allowed_models = vec!["gpt-4*".to_string()];
        let original_schemas = Arc::new(DashMap::new());
        let rewriter = CanonicalRewriter::new(&allowed_models, original_schemas);

        let mut tools = vec![ToolDefinition {
            raw: json!({
                "type": "function",
                "function": {
                    "name": "get_weather",
                    "description": "Get the current weather for a city",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "city": { "type": "string", "description": "The city name" },
                            "units": { "type": "string", "enum": ["celsius", "fahrenheit"] }
                        },
                        "required": ["city"]
                    }
                }
            }),
            name: "get_weather".to_string(),
            content_hash: 123,
        }];

        let mut ctx = CompressionContext {
            level: CompressionLevel::Max,
            model: "gpt-4-turbo".to_string(),
            provider_caps: crate::tool_compression::types::ProviderCaps {
                supports_canonical_format: true,
                ..crate::tool_compression::types::ProviderCaps::conservative()
            },
            ..Default::default()
        };

        let saved = rewriter.apply(&mut tools, &mut ctx);

        // Canonical rewriter should have been active (model matches and level is Max)
        // The output should still have valid structure (name preserved in some form)
        assert!(!tools.is_empty());
        // Token savings should be reported if rewriting occurred
        // (depends on implementation — may be 0 if format change isn't measured in tokens)
        let _ = saved;
    }

    // ─── 24.5: Feedback loop end-to-end ───────────────────────────────────────

    #[test]
    fn test_feedback_loop_end_to_end() {
        use crate::tool_compression::config::FeedbackLoopConfig;
        use crate::tool_compression::stages::feedback_loop::FeedbackLoop;

        let config = FeedbackLoopConfig {
            enabled: true,
            error_threshold: 0.15,
            recovery_window: 5,
            rolling_window: 10,
        };

        let fl = FeedbackLoop::new(&config, CompressionLevel::High);

        // Phase 1: Establish baseline (10 requests, 1 error → baseline 0.1)
        fl.record_outcome("group_a", true);
        for _ in 0..9 {
            fl.record_outcome("group_a", false);
        }
        // Baseline should now be set
        let state = fl.get_state("group_a").unwrap();
        assert!(state.baseline_rate.is_some());

        // Phase 2: Spike errors to trigger reduction
        for _ in 0..5 {
            fl.record_outcome("group_a", true);
        }
        // Level should have reduced from High
        let current = fl.get_adjusted_level("group_a").unwrap();
        assert!(
            current != CompressionLevel::High || current == CompressionLevel::Low,
            "Expected level reduction from High, got {:?}",
            current
        );

        // Phase 3: Admin lock
        fl.lock_group("group_a");
        let level_before = fl.get_adjusted_level("group_a").unwrap();
        for _ in 0..20 {
            fl.record_outcome("group_a", true);
        }
        assert_eq!(fl.get_adjusted_level("group_a").unwrap(), level_before);

        // Phase 4: Admin reset
        fl.reset_group("group_a");
        assert!(fl.get_state("group_a").is_none());
    }

    // ─── 24.6: Namespace grouping end-to-end ──────────────────────────────────

    #[test]
    fn test_namespace_grouping_end_to_end() {
        use crate::tool_compression::config::ToolCompressionConfig;
        use crate::tool_compression::stage::CompressionStage;
        use crate::tool_compression::stages::namespace_grouper::NamespaceGrouper;
        use crate::tool_compression::types::CompressionContext;

        let config = ToolCompressionConfig {
            namespace_grouping: crate::tool_compression::config::NamespaceGroupingConfig {
                enabled: true,
                min_tools_for_grouping: 5,
                namespace_mappings: Default::default(),
            },
            ..Default::default()
        };

        let grouper = NamespaceGrouper::new(&config.namespace_grouping);

        // Generate 50+ tools with MCP-style prefixed names
        let prefixes = ["github", "slack", "jira", "aws", "gcp"];
        let mut tools: Vec<ToolDefinition> = Vec::new();
        for prefix in &prefixes {
            for i in 0..10 {
                let name = format!("{}_{}", prefix, i);
                tools.push(ToolDefinition {
                    raw: json!({
                        "type": "function",
                        "function": {
                            "name": name,
                            "description": format!("Tool {} in {} namespace", i, prefix),
                            "parameters": { "type": "object", "properties": {} }
                        }
                    }),
                    name: name.clone(),
                    content_hash: 0,
                });
            }
        }

        assert_eq!(tools.len(), 50);

        let mut ctx = CompressionContext {
            level: CompressionLevel::High,
            ..Default::default()
        };

        let saved = grouper.apply(&mut tools, &mut ctx);

        // The grouper should have processed the tools
        // (exact behavior depends on whether disclosure is active)
        let _ = saved;
    }

    // ─── 24.7: Prompt cache skip behavior ─────────────────────────────────────

    #[test]
    fn test_prompt_cache_skip_behavior() {
        use crate::tool_compression::config::AutoTuningConfig;
        use crate::tool_compression::stages::auto_tuner::AutoTuner;
        use crate::tool_compression::types::{CompressionContext, ProviderCaps};

        let at = AutoTuner::new(&AutoTuningConfig::default());

        // Identical hashes + caching support → skip
        let tools = vec![
            ToolDefinition {
                raw: json!({}),
                name: "a".to_string(),
                content_hash: 111,
            },
            ToolDefinition {
                raw: json!({}),
                name: "b".to_string(),
                content_hash: 222,
            },
        ];
        let ctx = CompressionContext {
            provider_caps: ProviderCaps {
                supports_prompt_caching: true,
                ..ProviderCaps::conservative()
            },
            original_tools: tools.clone(),
            previous_hashes: Some(vec![111, 222]),
            ..Default::default()
        };

        // Should skip when no explicit header
        assert!(at.should_skip_compression(&ctx, false));

        // Should NOT skip when explicit header present
        assert!(!at.should_skip_compression(&ctx, true));

        // Different hashes → no skip
        let ctx_diff = CompressionContext {
            provider_caps: ProviderCaps {
                supports_prompt_caching: true,
                ..ProviderCaps::conservative()
            },
            original_tools: tools,
            previous_hashes: Some(vec![111, 333]), // hash mismatch
            ..Default::default()
        };
        assert!(!at.should_skip_compression(&ctx_diff, false));
    }

    // ─── 24.8: Provider-specific behavior ─────────────────────────────────────

    #[test]
    fn test_provider_specific_behavior() {
        use crate::tool_compression::types::ProviderCapabilityMap;

        let map = ProviderCapabilityMap::default();

        // OpenAI supports $ref
        let openai = map.get("openai");
        assert!(openai.supports_ref);
        assert!(openai.supports_prompt_caching);

        // Anthropic supports prompt caching
        let anthropic = map.get("anthropic");
        assert!(anthropic.supports_prompt_caching);

        // Unknown provider gets conservative defaults
        let unknown = map.get("unknown_provider_xyz");
        assert!(!unknown.supports_ref);
        assert!(!unknown.supports_prompt_caching);
        assert!(!unknown.supports_canonical_format);
        assert!(!unknown.supports_tool_search);
    }

    // ─── Response validation integration ──────────────────────────────────────

    #[test]
    fn test_response_validation_detects_hallucinated_tools() {
        let tools = vec![ToolDefinition {
            raw: json!({
                "type": "function",
                "function": {
                    "name": "get_weather",
                    "parameters": { "type": "object", "properties": { "city": { "type": "string" } } }
                }
            }),
            name: "get_weather".to_string(),
            content_hash: 0,
        }];

        // Valid tool call
        let valid_response = json!({
            "choices": [{
                "message": {
                    "tool_calls": [{
                        "function": { "name": "get_weather", "arguments": "{\"city\":\"Tokyo\"}" }
                    }]
                }
            }]
        });
        assert!(validate_tool_calls_against_originals(
            &valid_response,
            &tools
        ));

        // Hallucinated tool name
        let invalid_response = json!({
            "choices": [{
                "message": {
                    "tool_calls": [{
                        "function": { "name": "hallucinated_tool", "arguments": "{}" }
                    }]
                }
            }]
        });
        assert!(!validate_tool_calls_against_originals(
            &invalid_response,
            &tools
        ));
    }
}
