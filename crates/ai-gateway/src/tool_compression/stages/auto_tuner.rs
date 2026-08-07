//! Auto-Tuner stage — model tier detection and prompt cache skip logic.
//!
//! Classifies models into capability tiers via glob pattern matching and maps
//! tiers to default compression levels. Also detects prompt cache eligibility
//! when all tool hashes match the previous request.

use crate::tool_compression::config::{AutoTuningConfig, CompressionLevel, ToolCompressionConfig};
use crate::tool_compression::stage::CompressionStage;
use crate::tool_compression::stages::pruner::GlobPattern;
use crate::tool_compression::types::{CompressionContext, ToolDefinition};

/// Auto-Tuner compression stage.
///
/// Provides tier-based default compression levels and prompt cache skip detection.
/// The `apply` method is a no-op — this stage is consulted externally for level
/// resolution and skip decisions.
pub struct AutoTuner {
    /// Parsed glob patterns → tier level (1–3). First match wins.
    model_tiers: Vec<(GlobPattern, u8)>,
    /// Whether auto-tuning is enabled.
    enabled: bool,
}

impl AutoTuner {
    /// Create a new `AutoTuner` from config.
    pub fn new(config: &AutoTuningConfig) -> Self {
        let model_tiers: Vec<(GlobPattern, u8)> = config
            .model_tiers
            .iter()
            .map(|(pattern, &tier)| (GlobPattern::new(pattern), tier.clamp(1, 3)))
            .collect();

        Self {
            model_tiers,
            enabled: config.enabled,
        }
    }

    /// Get the tier-based default compression level for a model.
    ///
    /// Uses first-match semantics on glob patterns. If no pattern matches,
    /// defaults to Tier 2 (Medium).
    pub fn get_tier_level(&self, model: &str) -> CompressionLevel {
        for (pattern, tier) in &self.model_tiers {
            if pattern.matches(model) {
                return match tier {
                    1 => CompressionLevel::Low,
                    2 => CompressionLevel::Medium,
                    3 => CompressionLevel::High,
                    _ => CompressionLevel::Medium,
                };
            }
        }
        // Default: Tier 2 → Medium
        CompressionLevel::Medium
    }

    /// Determine whether compression should be skipped due to prompt cache hit.
    ///
    /// Returns `true` (skip) when:
    /// - Provider `supports_prompt_caching` is true
    /// - ALL current tool hashes match previous request hashes
    /// - No explicit `X-Tool-Compression-Level` header is present
    pub fn should_skip_compression(
        &self,
        ctx: &CompressionContext,
        has_explicit_header: bool,
    ) -> bool {
        // Explicit header overrides cache skip
        if has_explicit_header {
            return false;
        }

        // Provider must support prompt caching
        if !ctx.provider_caps.supports_prompt_caching {
            return false;
        }

        // Must have previous hashes to compare against
        let Some(previous) = &ctx.previous_hashes else {
            return false;
        };

        // Empty previous means first request — no skip
        if previous.is_empty() {
            return false;
        }

        // Compare current tool hashes against previous
        // Caller provides current hashes via original_tools content_hash
        let current_hashes: Vec<u64> = ctx.original_tools.iter().map(|t| t.content_hash).collect();

        // All hashes must match (same count and same values)
        if current_hashes.len() != previous.len() {
            return false;
        }

        current_hashes == *previous
    }

    /// Whether auto-tuning is enabled.
    pub fn is_auto_tuning_enabled(&self) -> bool {
        self.enabled
    }
}

impl CompressionStage for AutoTuner {
    fn apply(&self, _tools: &mut Vec<ToolDefinition>, _ctx: &mut CompressionContext) -> u64 {
        // No-op: AutoTuner is consulted by the middleware for level resolution,
        // not during pipeline execution.
        0
    }

    fn is_enabled(&self, config: &ToolCompressionConfig, _level: CompressionLevel) -> bool {
        config.auto_tuning.enabled
    }
}

// ─── Unit Tests ───────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;
    use std::collections::HashMap;

    fn config_with_tiers(tiers: Vec<(&str, u8)>) -> AutoTuningConfig {
        let model_tiers: HashMap<String, u8> =
            tiers.into_iter().map(|(k, v)| (k.to_string(), v)).collect();
        AutoTuningConfig {
            enabled: true,
            model_tiers,
        }
    }

    #[test]
    fn default_tier_is_medium() {
        let at = AutoTuner::new(&AutoTuningConfig::default());
        assert_eq!(at.get_tier_level("any-model"), CompressionLevel::Medium);
    }

    #[test]
    fn tier_1_maps_to_low() {
        let at = AutoTuner::new(&config_with_tiers(vec![("gpt-5*", 1)]));
        assert_eq!(at.get_tier_level("gpt-5-turbo"), CompressionLevel::Low);
    }

    #[test]
    fn tier_2_maps_to_medium() {
        let at = AutoTuner::new(&config_with_tiers(vec![("gpt-4*", 2)]));
        assert_eq!(at.get_tier_level("gpt-4-turbo"), CompressionLevel::Medium);
    }

    #[test]
    fn tier_3_maps_to_high() {
        let at = AutoTuner::new(&config_with_tiers(vec![("gpt-4o-mini*", 3)]));
        assert_eq!(at.get_tier_level("gpt-4o-mini"), CompressionLevel::High);
    }

    #[test]
    fn no_match_defaults_to_medium() {
        let at = AutoTuner::new(&config_with_tiers(vec![("gpt-5*", 1)]));
        assert_eq!(at.get_tier_level("claude-3"), CompressionLevel::Medium);
    }

    #[test]
    fn skip_when_hashes_match() {
        let at = AutoTuner::new(&AutoTuningConfig::default());
        let ctx = CompressionContext {
            provider_caps: crate::tool_compression::types::ProviderCaps {
                supports_prompt_caching: true,
                ..crate::tool_compression::types::ProviderCaps::conservative()
            },
            original_tools: vec![
                ToolDefinition {
                    raw: serde_json::json!({}),
                    name: "a".to_string(),
                    content_hash: 111,
                },
                ToolDefinition {
                    raw: serde_json::json!({}),
                    name: "b".to_string(),
                    content_hash: 222,
                },
            ],
            previous_hashes: Some(vec![111, 222]),
            ..Default::default()
        };
        assert!(at.should_skip_compression(&ctx, false));
    }

    #[test]
    fn no_skip_when_hashes_differ() {
        let at = AutoTuner::new(&AutoTuningConfig::default());
        let ctx = CompressionContext {
            provider_caps: crate::tool_compression::types::ProviderCaps {
                supports_prompt_caching: true,
                ..crate::tool_compression::types::ProviderCaps::conservative()
            },
            original_tools: vec![ToolDefinition {
                raw: serde_json::json!({}),
                name: "a".to_string(),
                content_hash: 111,
            }],
            previous_hashes: Some(vec![999]),
            ..Default::default()
        };
        assert!(!at.should_skip_compression(&ctx, false));
    }

    #[test]
    fn no_skip_when_explicit_header() {
        let at = AutoTuner::new(&AutoTuningConfig::default());
        let ctx = CompressionContext {
            provider_caps: crate::tool_compression::types::ProviderCaps {
                supports_prompt_caching: true,
                ..crate::tool_compression::types::ProviderCaps::conservative()
            },
            original_tools: vec![ToolDefinition {
                raw: serde_json::json!({}),
                name: "a".to_string(),
                content_hash: 111,
            }],
            previous_hashes: Some(vec![111]),
            ..Default::default()
        };
        assert!(!at.should_skip_compression(&ctx, true));
    }

    #[test]
    fn no_skip_when_no_caching_support() {
        let at = AutoTuner::new(&AutoTuningConfig::default());
        let ctx = CompressionContext {
            provider_caps: crate::tool_compression::types::ProviderCaps::conservative(),
            original_tools: vec![ToolDefinition {
                raw: serde_json::json!({}),
                name: "a".to_string(),
                content_hash: 111,
            }],
            previous_hashes: Some(vec![111]),
            ..Default::default()
        };
        assert!(!at.should_skip_compression(&ctx, false));
    }

    #[test]
    fn no_skip_when_no_previous_hashes() {
        let at = AutoTuner::new(&AutoTuningConfig::default());
        let ctx = CompressionContext {
            provider_caps: crate::tool_compression::types::ProviderCaps {
                supports_prompt_caching: true,
                ..crate::tool_compression::types::ProviderCaps::conservative()
            },
            original_tools: vec![ToolDefinition {
                raw: serde_json::json!({}),
                name: "a".to_string(),
                content_hash: 111,
            }],
            previous_hashes: None,
            ..Default::default()
        };
        assert!(!at.should_skip_compression(&ctx, false));
    }

    #[test]
    fn apply_is_noop() {
        let at = AutoTuner::new(&AutoTuningConfig::default());
        let mut tools = vec![ToolDefinition {
            raw: serde_json::json!({"type": "function"}),
            name: "test".to_string(),
            content_hash: 0,
        }];
        let mut ctx = CompressionContext::default();
        assert_eq!(at.apply(&mut tools, &mut ctx), 0);
    }

    #[test]
    fn is_enabled_follows_config() {
        let at = AutoTuner::new(&AutoTuningConfig::default());
        let config_enabled = ToolCompressionConfig {
            auto_tuning: AutoTuningConfig {
                enabled: true,
                ..Default::default()
            },
            ..Default::default()
        };
        let config_disabled = ToolCompressionConfig {
            auto_tuning: AutoTuningConfig {
                enabled: false,
                ..Default::default()
            },
            ..Default::default()
        };
        assert!(at.is_enabled(&config_enabled, CompressionLevel::Medium));
        assert!(!at.is_enabled(&config_disabled, CompressionLevel::Medium));
    }
}

// ─── Property Tests ───────────────────────────────────────────────────────────

#[cfg(test)]
mod property_tests {
    use super::*;
    use proptest::prelude::*;
    use std::collections::HashMap;

    // ─── Property 17: Auto-Tuner Tier Resolution ──────────────────────────────
    // Feature: tool-definition-compression
    // **Validates: Requirements 19.1, 19.2, 19.3**
    //
    // Generate random model names and tier maps; verify first-match semantics
    // and Tier 2 default when no pattern matches.

    fn arb_tier_map() -> impl Strategy<Value = HashMap<String, u8>> {
        prop::collection::hash_map(
            prop_oneof![
                Just("gpt-5*".to_string()),
                Just("gpt-4*".to_string()),
                Just("claude-3*".to_string()),
                Just("gemini*".to_string()),
                Just("llama*".to_string()),
            ],
            1u8..=3,
            0..=4,
        )
    }

    fn arb_model() -> impl Strategy<Value = String> {
        prop_oneof![
            Just("gpt-5-turbo".to_string()),
            Just("gpt-4-turbo".to_string()),
            Just("gpt-4o-mini".to_string()),
            Just("claude-3-sonnet".to_string()),
            Just("gemini-pro".to_string()),
            Just("llama-3-70b".to_string()),
            Just("unknown-model".to_string()),
            "[a-z]{4,8}-[0-9]".prop_map(|s| s),
        ]
    }

    proptest! {
        #![proptest_config(ProptestConfig::with_cases(300))]

        #[test]
        fn auto_tuner_tier_resolution(
            tier_map in arb_tier_map(),
            model in arb_model(),
        ) {
            let config = AutoTuningConfig {
                enabled: true,
                model_tiers: tier_map.clone(),
            };
            let at = AutoTuner::new(&config);
            let result = at.get_tier_level(&model);

            // Result must be a valid level
            let ord = match result {
                CompressionLevel::Low => 0u8,
                CompressionLevel::Medium => 1,
                CompressionLevel::High => 2,
                CompressionLevel::Max => 3,
            };
            prop_assert!(ord <= 2, "AutoTuner should not return Max, got {:?}", result);

            // Check if any pattern matches (first-match semantics)
            let mut matched = false;
            for (pattern, tier) in &at.model_tiers {
                if pattern.matches(&model) {
                    let expected = match tier {
                        1 => CompressionLevel::Low,
                        2 => CompressionLevel::Medium,
                        3 => CompressionLevel::High,
                        _ => CompressionLevel::Medium,
                    };
                    prop_assert_eq!(result, expected, "First match tier {} should map correctly", tier);
                    matched = true;
                    break;
                }
            }

            if !matched {
                // Default to Medium (Tier 2)
                prop_assert_eq!(result, CompressionLevel::Medium, "No match should default to Medium");
            }
        }
    }

    // ─── Property 18: Prompt Cache Skip Correctness ───────────────────────────
    // Feature: tool-definition-compression
    // **Validates: Requirements 19.6, 12.9**
    //
    // Generate consecutive requests with identical/different hashes; verify
    // skip behavior.

    proptest! {
        #![proptest_config(ProptestConfig::with_cases(300))]

        #[test]
        fn prompt_cache_skip_correctness(
            hashes in prop::collection::vec(1u64..=1000, 1..=10),
            modify_one in prop::bool::ANY,
            has_explicit_header in prop::bool::ANY,
            supports_caching in prop::bool::ANY,
        ) {
            let at = AutoTuner::new(&AutoTuningConfig::default());

            let tools: Vec<ToolDefinition> = hashes.iter().enumerate().map(|(i, &h)| {
                ToolDefinition {
                    raw: serde_json::json!({}),
                    name: format!("tool_{}", i),
                    content_hash: h,
                }
            }).collect();

            let mut previous = hashes.clone();
            if modify_one && !previous.is_empty() {
                // Change one hash to simulate a modified tool
                previous[0] = previous[0].wrapping_add(1);
            }

            let ctx = CompressionContext {
                provider_caps: crate::tool_compression::types::ProviderCaps {
                    supports_prompt_caching: supports_caching,
                    ..crate::tool_compression::types::ProviderCaps::conservative()
                },
                original_tools: tools,
                previous_hashes: Some(previous.clone()),
                ..Default::default()
            };

            let skip = at.should_skip_compression(&ctx, has_explicit_header);

            // Skip is only true when ALL conditions hold:
            // 1. No explicit header
            // 2. Provider supports caching
            // 3. Hashes match exactly
            let hashes_match = !modify_one;
            let expected_skip = !has_explicit_header && supports_caching && hashes_match;

            prop_assert_eq!(
                skip, expected_skip,
                "skip={} but expected={} (header={}, caching={}, match={})",
                skip, expected_skip, has_explicit_header, supports_caching, hashes_match
            );
        }
    }
}
