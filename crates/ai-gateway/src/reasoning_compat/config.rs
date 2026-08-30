//! Configuration for the reasoning-compatibility layer (design Component 5).
//!
//! Defines the configuration contract that the detect → policy → normalize →
//! cost pipeline consumes: effort-to-budget mapping, provider-family
//! classification, parameter shapes, and per-provider overrides.
//!
//! The layer is opt-out: `enabled: false` reproduces today's passthrough
//! behavior exactly. `enabled` defaults to `true` (the design describes the
//! feature as opt-out; the design does not pin a separate default).

use serde::{Deserialize, Serialize};
use std::collections::HashMap;

/// Minimum accepted `budget_tokens` value (Anthropic extended-thinking floor).
pub const MIN_REASONING_BUDGET_TOKENS: u32 = 1024;

/// Reasoning-compat configuration for cross-model failover safety.
///
/// When enabled (default: true), the gateway detects reasoning carriers in
/// prior assistant turns, strips them on cross-model transitions, normalizes
/// reasoning parameters to the target's accepted shape, and attributes
/// reasoning-token costs.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct ReasoningCompatConfig {
    /// Master enable flag. `false` reproduces exact current passthrough
    /// behavior (no detection, no strip, no normalization, no cost changes).
    #[serde(default = "default_enabled")]
    pub enabled: bool,

    /// Strip reasoning state on any model change. When true (default),
    /// prior-turn reasoning state is removed whenever the failover target
    /// model differs from the model that produced it. When false, reasoning
    /// state is preserved when source and target share a reasoning family.
    #[serde(default = "default_strip_on_model_change")]
    pub strip_on_model_change: bool,

    /// Attribute reasoning-token spend (including invisible reasoning) in
    /// logs, metrics, and cost calculations.
    #[serde(default = "default_attribute_reasoning_cost")]
    pub attribute_reasoning_cost: bool,

    /// Default mapping from `reasoning_effort` levels to Anthropic manual-mode
    /// `thinking.budget_tokens`. When the section is absent, the default map
    /// applies (minimal 1024, low 2048, medium 8192, high 16384, xhigh 32768).
    /// A custom map must define all five efforts and every budget must be
    /// >= 1024.
    #[serde(default)]
    pub effort_budget_map: EffortBudgetMap,

    /// Per-provider overrides. Keys must match configured provider names.
    #[serde(default)]
    pub per_provider: HashMap<String, ProviderReasoningOverride>,

    /// Track conversation-model affinity (prefix hash → resolved provider
    /// model id) to supply source-model attribution for preserve decisions.
    #[serde(default = "default_conversation_model_affinity")]
    pub conversation_model_affinity: bool,
}

fn default_enabled() -> bool {
    true
}

fn default_strip_on_model_change() -> bool {
    true
}

fn default_attribute_reasoning_cost() -> bool {
    true
}

fn default_conversation_model_affinity() -> bool {
    true
}

impl Default for ReasoningCompatConfig {
    fn default() -> Self {
        Self {
            enabled: default_enabled(),
            strip_on_model_change: default_strip_on_model_change(),
            attribute_reasoning_cost: default_attribute_reasoning_cost(),
            effort_budget_map: EffortBudgetMap::default(),
            per_provider: HashMap::new(),
            conversation_model_affinity: default_conversation_model_affinity(),
        }
    }
}

impl ReasoningCompatConfig {
    /// Validate the configuration against the set of configured providers.
    ///
    /// Checks:
    /// - every budget in the effort map is >= 1024
    /// - a custom effort map is complete (all five efforts present)
    /// - `per_provider` override keys reference known provider names
/// - per-provider override effort maps satisfy the same budget rules
pub fn validate(&self, known_providers: &[&str]) -> Result<(), Vec<String>> {
        let mut errors = Vec::new();

        if let Err(map_errors) = self.effort_budget_map.validate() {
            errors.extend(map_errors);
        }

        for (provider_name, override_config) in &self.per_provider {
            if !known_providers.contains(&provider_name.as_str()) {
                errors.push(format!(
                    "per_provider key '{provider_name}' does not match any configured provider"
                ));
            }
            if let Err(override_errors) = override_config.validate() {
                errors.extend(
                    override_errors
                        .into_iter()
                        .map(|error| format!("per_provider['{provider_name}']: {error}")),
                );
            }
        }

        if errors.is_empty() {
            Ok(())
        } else {
            Err(errors)
        }
    }
}

/// Reasoning effort level (OpenAI-style `reasoning_effort` values).
#[derive(
    Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash, Serialize, Deserialize,
)]
#[serde(rename_all = "lowercase")]
pub enum Effort {
    Minimal,
    Low,
    Medium,
    High,
    XHigh,
}

impl Effort {
    /// All efforts in ascending order.
    pub const ALL: [Effort; 5] = [
        Effort::Minimal,
        Effort::Low,
        Effort::Medium,
        Effort::High,
        Effort::XHigh,
    ];

    /// Parse an effort string (case-sensitive YAML value / case-insensitive
    /// fallback for request parameters). Used by validation and by the
    /// normalizer when reading client `reasoning_effort` values.
    pub fn parse(value: &str) -> Option<Self> {
        match value.trim().to_lowercase().as_str() {
            "minimal" => Some(Effort::Minimal),
            "low" => Some(Effort::Low),
            "medium" => Some(Effort::Medium),
            "high" => Some(Effort::High),
            "xhigh" => Some(Effort::XHigh),
            _ => None,
        }
    }

    /// Default budget for this effort level (minimal 1024, low 2048,
    /// medium 8192, high 16384, xhigh 32768).
    pub fn default_budget(self) -> u32 {
        match self {
            Effort::Minimal => 1024,
            Effort::Low => 2048,
            Effort::Medium => 8192,
            Effort::High => 16384,
            Effort::XHigh => 32768,
        }
    }
}

impl std::fmt::Display for Effort {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str(match self {
            Effort::Minimal => "minimal",
            Effort::Low => "low",
            Effort::Medium => "medium",
            Effort::High => "high",
            Effort::XHigh => "xhigh",
        })
    }
}

/// Mapping from reasoning effort levels to `budget_tokens` values, used when
/// normalizing OpenAI-style `reasoning_effort` to Anthropic manual-mode
/// `thinking.budget_tokens`.
///
/// Serialized as a plain mapping (`minimal: 1024`, ...). An absent section
/// deserializes to the default map; a custom section must list all five
/// efforts (enforced by [`EffortBudgetMap::validate`]).
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(transparent)]
pub struct EffortBudgetMap(pub HashMap<Effort, u32>);

impl Default for EffortBudgetMap {
    fn default() -> Self {
        Self(Effort::ALL.iter().map(|&e| (e, e.default_budget())).collect())
    }
}

impl EffortBudgetMap {
    /// Budget for an effort, falling back to the effort's default when the
    /// entry is absent.
    pub fn budget_for(&self, effort: Effort) -> u32 {
        self.0.get(&effort).copied().unwrap_or_else(|| effort.default_budget())
    }

/// Validate: complete (all five efforts) and every budget >= 1024.
pub fn validate(&self) -> Result<(), Vec<String>> {
        let mut errors = Vec::new();

        for effort in Effort::ALL {
            match self.0.get(&effort) {
                None => errors.push(format!(
                    "effort_budget_map is missing required effort '{effort}' \
                     (a custom map must define all five efforts: \
                     minimal, low, medium, high, xhigh)"
                )),
                Some(&budget) if budget < MIN_REASONING_BUDGET_TOKENS => errors.push(format!(
                    "effort_budget_map.{effort} = {budget} is below the minimum \
                     {MIN_REASONING_BUDGET_TOKENS}"
                )),
                Some(_) => {}
            }
        }

        if errors.is_empty() {
            Ok(())
        } else {
            Err(errors)
        }
    }
}

/// Per-provider reasoning-compat override.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct ProviderReasoningOverride {
    /// Replace the global effort-to-budget map for this provider.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub effort_budget_map: Option<EffortBudgetMap>,

    /// Override the global strip-on-model-change behavior for this provider.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub strip_on_model_change: Option<bool>,

    /// Override the global reasoning-cost attribution for this provider.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub normalize_reasoning_parameters: Option<bool>,
}

impl ProviderReasoningOverride {
/// Validate the override's effort map when present.
pub fn validate(&self) -> Result<(), Vec<String>> {
        match &self.effort_budget_map {
            Some(map) => map.validate(),
            None => Ok(()),
        }
    }
}

/// Reasoning family classification for a model.
///
/// Defined here (not in `detect.rs`) because [`crate::config::ProviderModel`]
/// carries it as optional config metadata; the classifier in `detect.rs`
/// (task 2) imports it from here.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum ReasoningFamily {
    /// Anthropic manual mode: `thinking: {type: "enabled", budget_tokens: N}`.
    /// Signed/encrypted thinking blocks; 400s on Claude 4.7+.
    AnthropicManual,
    /// Anthropic adaptive mode: `thinking: {type: "adaptive"}` +
    /// `output_config.effort`.
    AnthropicAdaptive,
    /// OpenAI o-series: `reasoning_effort`, usage in
    /// `completion_tokens_details.reasoning_tokens`.
    #[serde(rename = "openai_reasoning")]
    OpenAIReasoning,
    /// DeepSeek: assistant `reasoning_content` field.
    #[serde(rename = "deepseek")]
    DeepSeek,
    /// OpenRouter: assistant `reasoning` field, `reasoning: {max_tokens}`.
    #[serde(rename = "openrouter")]
    OpenRouter,
    /// Google Gemini.
    Gemini,
    /// xAI Grok: OpenAI-compatible `reasoning_effort`.
    #[serde(rename = "xai")]
    XAI,
    /// Unknown or no reasoning support.
    None,
}

/// Reasoning parameter shape a target model accepts; drives normalization.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum ReasoningParamShape {
    /// Anthropic manual: `thinking: {type: "enabled", budget_tokens: N}`
    /// (budget >= 1024 and < `max_tokens`).
    ThinkingBudget,
    /// Anthropic adaptive: `thinking: {type: "adaptive"}` +
    /// `output_config.effort`.
    Adaptive,
    /// OpenRouter: `reasoning: {max_tokens: N}`.
    ReasoningMaxTokens,
    /// OpenAI / xAI: `reasoning_effort: minimal|low|medium|high|xhigh`.
    ReasoningEffort,
    /// Target accepts no reasoning parameters.
    None,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn default_config_parses_from_empty_yaml() {
        let config: ReasoningCompatConfig =
            serde_yaml::from_str("{}").expect("empty section must parse");
        let expected = ReasoningCompatConfig::default();
        assert_eq!(config, expected);
        assert!(config.enabled);
        assert!(config.strip_on_model_change);
        assert!(config.attribute_reasoning_cost);
        assert!(config.conversation_model_affinity);
        assert_eq!(config.effort_budget_map.budget_for(Effort::Minimal), 1024);
        assert_eq!(config.effort_budget_map.budget_for(Effort::Low), 2048);
        assert_eq!(config.effort_budget_map.budget_for(Effort::Medium), 8192);
        assert_eq!(config.effort_budget_map.budget_for(Effort::High), 16384);
        assert_eq!(config.effort_budget_map.budget_for(Effort::XHigh), 32768);
        assert!(config.per_provider.is_empty());
    }

    #[test]
    fn disabled_flag_parses() {
        let config: ReasoningCompatConfig =
            serde_yaml::from_str("enabled: false").expect("disabled flag must parse");
        assert!(!config.enabled);
        // Remaining defaults still apply.
        assert_eq!(config.effort_budget_map, EffortBudgetMap::default());
    }

    #[test]
    fn custom_complete_effort_map_parses_and_validates() {
        let config: ReasoningCompatConfig = serde_yaml::from_str(
            "effort_budget_map:\n  minimal: 2048\n  low: 4096\n  medium: 8192\n  high: 16384\n  xhigh: 32768\n",
        )
        .expect("complete custom map must parse");
        assert_eq!(config.effort_budget_map.budget_for(Effort::Minimal), 2048);
        assert_eq!(config.effort_budget_map.budget_for(Effort::Low), 4096);
        assert!(config.validate(&[]).is_ok());
    }

    #[test]
    fn incomplete_effort_map_is_rejected() {
        let config: ReasoningCompatConfig = serde_yaml::from_str(
            "effort_budget_map:\n  minimal: 2048\n  high: 16384\n",
        )
        .expect("partial map parses at the serde layer");
        let errors = config.validate(&[]).unwrap_err();
        assert!(errors.iter().any(|e| e.contains("missing required effort 'low'")));
        assert!(errors.iter().any(|e| e.contains("missing required effort 'medium'")));
        assert!(errors.iter().any(|e| e.contains("missing required effort 'xhigh'")));
    }

    #[test]
    fn budget_below_minimum_is_rejected() {
        let config: ReasoningCompatConfig = serde_yaml::from_str(
            "effort_budget_map:\n  minimal: 512\n  low: 2048\n  medium: 8192\n  high: 16384\n  xhigh: 32768\n",
        )
        .expect("map with low budget parses at the serde layer");
        let errors = config.validate(&[]).unwrap_err();
        assert!(errors
            .iter()
            .any(|e| e.contains("effort_budget_map.minimal = 512 is below the minimum 1024")));
    }

    #[test]
    fn unknown_provider_override_is_rejected() {
        let config: ReasoningCompatConfig = serde_yaml::from_str(
            "per_provider:\n  ghost:\n    strip_on_model_change: false\n",
        )
        .expect("override parses at the serde layer");
        let errors = config.validate(&["openai", "anthropic"]).unwrap_err();
        assert!(errors
            .iter()
            .any(|e| e.contains("'ghost' does not match any configured provider")));
    }

    #[test]
    fn known_provider_override_with_valid_map_is_accepted() {
        let config: ReasoningCompatConfig = serde_yaml::from_str(
            "per_provider:\n  openai:\n    effort_budget_map:\n      minimal: 2048\n      low: 4096\n      medium: 8192\n      high: 16384\n      xhigh: 65536\n    strip_on_model_change: false\n    normalize_reasoning_parameters: true\n",
        )
        .expect("valid override must parse");
        assert!(config.validate(&["openai"]).is_ok());
        let override_config = config.per_provider.get("openai").unwrap();
        assert_eq!(override_config.strip_on_model_change, Some(false));
        assert_eq!(
            override_config.effort_budget_map.as_ref().unwrap().budget_for(Effort::XHigh),
            65536
        );
    }

    #[test]
    fn override_effort_map_inheritable_budget_checks() {
        let config: ReasoningCompatConfig = serde_yaml::from_str(
            "per_provider:\n  openai:\n    effort_budget_map:\n      minimal: 8\n      low: 2048\n      medium: 8192\n      high: 16384\n      xhigh: 32768\n",
        )
        .expect("override parses at the serde layer");
        let errors = config.validate(&["openai"]).unwrap_err();
        assert!(errors.iter().any(|e| e.contains("per_provider['openai']")));
        assert!(errors.iter().any(|e| e.contains("below the minimum")));
    }

    #[test]
    fn effort_parse_roundtrip() {
        for effort in Effort::ALL {
            assert_eq!(Effort::parse(&effort.to_string()), Some(effort));
        }
        assert_eq!(Effort::parse("MINIMAL"), Some(Effort::Minimal));
        assert_eq!(Effort::parse("XHigh"), Some(Effort::XHigh));
        assert_eq!(Effort::parse("bogus"), None);
    }

    #[test]
    fn effort_ordering_is_ascending() {
        assert!(Effort::Minimal < Effort::Low);
        assert!(Effort::Low < Effort::Medium);
        assert!(Effort::Medium < Effort::High);
        assert!(Effort::High < Effort::XHigh);
    }

    #[test]
    fn reasoning_family_serde_roundtrip() {
        let families = [
            (ReasoningFamily::AnthropicManual, "anthropic_manual"),
            (ReasoningFamily::AnthropicAdaptive, "anthropic_adaptive"),
            (ReasoningFamily::OpenAIReasoning, "openai_reasoning"),
            (ReasoningFamily::DeepSeek, "deepseek"),
            (ReasoningFamily::OpenRouter, "openrouter"),
            (ReasoningFamily::Gemini, "gemini"),
            (ReasoningFamily::XAI, "xai"),
            (ReasoningFamily::None, "none"),
        ];
        for (family, wire) in families {
            assert_eq!(serde_yaml::to_string(&family).unwrap().trim(), wire);
            let parsed: ReasoningFamily = serde_yaml::from_str(wire).unwrap();
            assert_eq!(parsed, family);
        }
    }

    #[test]
    fn reasoning_param_shape_serde_roundtrip() {
        let shapes = [
            (ReasoningParamShape::ThinkingBudget, "thinking_budget"),
            (ReasoningParamShape::Adaptive, "adaptive"),
            (ReasoningParamShape::ReasoningMaxTokens, "reasoning_max_tokens"),
            (ReasoningParamShape::ReasoningEffort, "reasoning_effort"),
            (ReasoningParamShape::None, "none"),
        ];
        for (shape, wire) in shapes {
            assert_eq!(serde_yaml::to_string(&shape).unwrap().trim(), wire);
            let parsed: ReasoningParamShape = serde_yaml::from_str(wire).unwrap();
            assert_eq!(parsed, shape);
        }
    }
}
