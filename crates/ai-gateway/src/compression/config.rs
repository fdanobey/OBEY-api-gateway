//! Compression configuration, validation, hierarchical resolution, and hot reload.

use std::collections::{HashMap, HashSet};
use std::fmt;
use std::sync::Arc;

use serde::{Deserialize, Serialize};
use tokio::sync::RwLock;

use super::engines::CompressionLevel;

const DEFAULT_LANGUAGE_PACKS_DIR: &str = "./language_packs";
const DEFAULT_PERPLEXITY_MODEL_PATH: &str = "./models/perplexity_scorer.onnx";
const MAX_COMPRESSION_RATIO: u8 = 20;

/// Engine names accepted in custom compression pipelines.
pub const KNOWN_ENGINE_NAMES: &[&str] = &[
    "lite",
    "standard",
    "aggressive",
    "ultra",
    "rtk",
    "perplexity",
    "tool_def",
    "language_pack",
];

/// Protection rules enabled when no explicit list is configured.
pub const DEFAULT_PROTECTION_RULES: &[&str] = &[
    "code_blocks",
    "urls",
    "file_paths",
    "json_structures",
    "identifiers",
    "math_expressions",
    "tool_definitions",
    "structured_tool_output",
];

/// Complete global token-compression configuration.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(default, deny_unknown_fields)]
pub struct CompressionConfig {
    pub enabled: bool,
    pub default_level: CompressionLevel,
    pub auto_threshold_tokens: u32,
    pub caveman_output: bool,
    pub compress_tool_definitions: bool,
    pub language: String,
    pub language_packs_dir: String,
    pub time_budget_ms: TimeBudgetConfig,
    pub protection_rules: Vec<String>,
    pub precompressed_contexts: Vec<PrecompressedEntry>,
    pub rtk: RtkConfig,
    pub perplexity: PerplexityConfig,
    pub custom_pipelines: HashMap<String, CustomPipelineConfig>,
}

impl Default for CompressionConfig {
    fn default() -> Self {
        Self {
            enabled: false,
            default_level: CompressionLevel::Lite,
            auto_threshold_tokens: 0,
            caveman_output: false,
            compress_tool_definitions: false,
            language: "en".to_owned(),
            language_packs_dir: DEFAULT_LANGUAGE_PACKS_DIR.to_owned(),
            time_budget_ms: TimeBudgetConfig::default(),
            protection_rules: default_protection_rules(),
            precompressed_contexts: Vec::new(),
            rtk: RtkConfig::default(),
            perplexity: PerplexityConfig::default(),
            custom_pipelines: HashMap::new(),
        }
    }
}

impl CompressionConfig {
    /// Resolves an effective request configuration one field at a time.
    ///
    /// Model-group values have precedence over provider values, which have
    /// precedence over the global configuration. A provider can therefore
    /// explicitly disable compression, and either override level can select
    /// [`CompressionLevel::None`].
    pub fn resolve(
        &self,
        provider: Option<&ProviderCompressionOverride>,
        model_group: Option<&ModelGroupCompressionOverride>,
    ) -> EffectiveCompressionConfig {
        EffectiveCompressionConfig {
            enabled: provider
                .and_then(|config| config.enabled)
                .unwrap_or(self.enabled),
            level: model_group
                .and_then(|config| config.level)
                .or_else(|| provider.and_then(|config| config.level))
                .unwrap_or(self.default_level),
            auto_threshold_tokens: model_group
                .and_then(|config| config.auto_threshold_tokens)
                .or_else(|| provider.and_then(|config| config.auto_threshold_tokens))
                .unwrap_or(self.auto_threshold_tokens),
            caveman_output: model_group
                .and_then(|config| config.caveman_output)
                .or_else(|| provider.and_then(|config| config.caveman_output))
                .unwrap_or(self.caveman_output),
        }
    }

    /// Validates all values that cannot be constrained by serde alone.
    pub fn validate(&self) -> CompressionValidationResult<()> {
        let mut errors = Vec::new();

        validate_nonempty("language", &self.language, &mut errors);
        validate_path("language_packs_dir", &self.language_packs_dir, &mut errors);
        self.time_budget_ms.validate(&mut errors);
        self.perplexity.validate(&mut errors);
        self.validate_protection_rules(&mut errors);
        self.validate_custom_pipelines(&mut errors);

        for (index, entry) in self.precompressed_contexts.iter().enumerate() {
            entry.validate(index, &mut errors);
        }

        if errors.is_empty() {
            Ok(())
        } else {
            Err(errors)
        }
    }

    fn validate_protection_rules(&self, errors: &mut Vec<CompressionConfigError>) {
        let known: HashSet<&str> = DEFAULT_PROTECTION_RULES.iter().copied().collect();
        let mut seen = HashSet::new();

        for (index, rule) in self.protection_rules.iter().enumerate() {
            let field = format!("protection_rules[{index}]");
            let rule = rule.trim();
            if rule.is_empty() {
                errors.push(CompressionConfigError::new(field, "must not be empty"));
            } else if !known.contains(rule) {
                errors.push(CompressionConfigError::new(
                    field,
                    format!(
                        "contains unknown rule `{rule}`; expected one of: {}",
                        DEFAULT_PROTECTION_RULES.join(", ")
                    ),
                ));
            } else if !seen.insert(rule) {
                errors.push(CompressionConfigError::new(
                    field,
                    format!("duplicates protection rule `{rule}`"),
                ));
            }
        }
    }

    fn validate_custom_pipelines(&self, errors: &mut Vec<CompressionConfigError>) {
        let known: HashSet<&str> = KNOWN_ENGINE_NAMES.iter().copied().collect();

        for (name, pipeline) in &self.custom_pipelines {
            let trimmed_name = name.trim();
            if trimmed_name.is_empty() {
                errors.push(CompressionConfigError::new(
                    "custom_pipelines",
                    "pipeline names must not be empty",
                ));
            } else if name.contains('\0') {
                errors.push(CompressionConfigError::new(
                    format!("custom_pipelines.{name}"),
                    "pipeline name must not contain NUL characters",
                ));
            }

            let field = format!("custom_pipelines.{name}.engines");
            if pipeline.engines.is_empty() {
                errors.push(CompressionConfigError::new(
                    field,
                    "must contain at least one engine",
                ));
                continue;
            }

            for (index, engine) in pipeline.engines.iter().enumerate() {
                if !known.contains(engine.as_str()) {
                    errors.push(CompressionConfigError::new(
                        format!("{field}[{index}]"),
                        format!(
                            "contains unknown engine `{engine}`; expected one of: {}",
                            KNOWN_ENGINE_NAMES.join(", ")
                        ),
                    ));
                }
            }
        }
    }
}

/// Per-provider fields that may override global compression behavior.
#[derive(Debug, Clone, Default, PartialEq, Eq, Serialize, Deserialize)]
#[serde(default, deny_unknown_fields)]
pub struct ProviderCompressionOverride {
    pub enabled: Option<bool>,
    pub level: Option<CompressionLevel>,
    pub auto_threshold_tokens: Option<u32>,
    pub caveman_output: Option<bool>,
}

/// Per-model-group fields that may override provider or global behavior.
#[derive(Debug, Clone, Default, PartialEq, Eq, Serialize, Deserialize)]
#[serde(default, deny_unknown_fields)]
pub struct ModelGroupCompressionOverride {
    pub level: Option<CompressionLevel>,
    pub auto_threshold_tokens: Option<u32>,
    pub caveman_output: Option<bool>,
}

/// Fully resolved request-scoped compression settings.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct EffectiveCompressionConfig {
    pub enabled: bool,
    pub level: CompressionLevel,
    pub auto_threshold_tokens: u32,
    pub caveman_output: bool,
}

/// Per-level compression time budgets in milliseconds.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(default, deny_unknown_fields)]
pub struct TimeBudgetConfig {
    pub lite: u64,
    pub standard: u64,
    pub aggressive: u64,
    pub ultra: u64,
    pub rtk: u64,
    pub stacked: u64,
}

impl Default for TimeBudgetConfig {
    fn default() -> Self {
        Self {
            lite: 500,
            standard: 500,
            aggressive: 2_000,
            ultra: 2_000,
            rtk: 2_000,
            stacked: 2_000,
        }
    }
}

impl TimeBudgetConfig {
    /// Returns the configured time budget for a named compression level.
    pub fn for_level(&self, level: CompressionLevel) -> Option<u64> {
        match level {
            CompressionLevel::None => None,
            CompressionLevel::Lite => Some(self.lite),
            CompressionLevel::Standard => Some(self.standard),
            CompressionLevel::Aggressive => Some(self.aggressive),
            CompressionLevel::Ultra => Some(self.ultra),
            CompressionLevel::Rtk => Some(self.rtk),
            CompressionLevel::Stacked => Some(self.stacked),
        }
    }

    fn validate(&self, errors: &mut Vec<CompressionConfigError>) {
        for (name, value) in [
            ("lite", self.lite),
            ("standard", self.standard),
            ("aggressive", self.aggressive),
            ("ultra", self.ultra),
            ("rtk", self.rtk),
            ("stacked", self.stacked),
        ] {
            if value == 0 {
                errors.push(CompressionConfigError::new(
                    format!("time_budget_ms.{name}"),
                    "must be greater than 0 milliseconds",
                ));
            }
        }
    }
}

/// RTK output grouping strategy.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum RtkGroupingStrategy {
    Aggressive,
    #[default]
    Balanced,
    Conservative,
}

/// RTK engine configuration.
#[derive(Debug, Clone, Default, PartialEq, Eq, Serialize, Deserialize)]
#[serde(default, deny_unknown_fields)]
pub struct RtkConfig {
    pub grouping_strategy: RtkGroupingStrategy,
}

/// Perplexity-based compression engine configuration.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(default, deny_unknown_fields)]
pub struct PerplexityConfig {
    pub enabled: bool,
    pub redundancy_threshold: f32,
    pub compression_ratio_target: u8,
    pub model_path: String,
}

impl Default for PerplexityConfig {
    fn default() -> Self {
        Self {
            enabled: false,
            redundancy_threshold: 0.5,
            compression_ratio_target: 5,
            model_path: DEFAULT_PERPLEXITY_MODEL_PATH.to_owned(),
        }
    }
}

impl PerplexityConfig {
    fn validate(&self, errors: &mut Vec<CompressionConfigError>) {
        if !self.redundancy_threshold.is_finite()
            || !(0.0..=1.0).contains(&self.redundancy_threshold)
        {
            errors.push(CompressionConfigError::new(
                "perplexity.redundancy_threshold",
                format!(
                    "must be a finite value between 0.0 and 1.0 inclusive; got {}",
                    self.redundancy_threshold
                ),
            ));
        }

        if !(1..=MAX_COMPRESSION_RATIO).contains(&self.compression_ratio_target) {
            errors.push(CompressionConfigError::new(
                "perplexity.compression_ratio_target",
                format!(
                    "must be between 1 and {MAX_COMPRESSION_RATIO} inclusive; got {}",
                    self.compression_ratio_target
                ),
            ));
        }

        validate_path("perplexity.model_path", &self.model_path, errors);
    }
}

/// Ordered engine list for a named custom pipeline.
#[derive(Debug, Clone, Default, PartialEq, Eq, Serialize, Deserialize)]
#[serde(default, deny_unknown_fields)]
pub struct CustomPipelineConfig {
    pub engines: Vec<String>,
}

/// Mapping from an original context source to its pre-compressed artifact.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct PrecompressedEntry {
    #[serde(alias = "path", alias = "source")]
    pub source_path: String,
    #[serde(alias = "compressed")]
    pub compressed_path: String,
    #[serde(default)]
    pub content_hash: Option<String>,
}

impl PrecompressedEntry {
    fn validate(&self, index: usize, errors: &mut Vec<CompressionConfigError>) {
        validate_path(
            &format!("precompressed_contexts[{index}].source_path"),
            &self.source_path,
            errors,
        );
        validate_path(
            &format!("precompressed_contexts[{index}].compressed_path"),
            &self.compressed_path,
            errors,
        );

        if let Some(content_hash) = &self.content_hash {
            validate_nonempty(
                &format!("precompressed_contexts[{index}].content_hash"),
                content_hash,
                errors,
            );
            if content_hash.contains('\0') {
                errors.push(CompressionConfigError::new(
                    format!("precompressed_contexts[{index}].content_hash"),
                    "must not contain NUL characters",
                ));
            }
        }
    }
}

/// A descriptive startup validation failure for one compression field.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct CompressionConfigError {
    pub field: String,
    pub message: String,
}

impl CompressionConfigError {
    fn new(field: impl Into<String>, message: impl Into<String>) -> Self {
        Self {
            field: field.into(),
            message: message.into(),
        }
    }
}

impl fmt::Display for CompressionConfigError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            formatter,
            "invalid compression.{}: {}",
            self.field, self.message
        )
    }
}

impl std::error::Error for CompressionConfigError {}

/// Result returned by compression configuration validation and hot reload.
pub type CompressionValidationResult<T> = Result<T, Vec<CompressionConfigError>>;

/// Shared compression state used by request handlers and hot reload.
pub type SharedCompressionConfig = Arc<RwLock<CompressionConfig>>;

/// Validates and wraps a startup configuration for concurrent access.
pub fn shared_compression_config(
    config: CompressionConfig,
) -> CompressionValidationResult<SharedCompressionConfig> {
    config.validate()?;
    Ok(Arc::new(RwLock::new(config)))
}

/// Atomically replaces live compression settings after validation succeeds.
pub async fn reload_compression_config(
    shared: &SharedCompressionConfig,
    replacement: CompressionConfig,
) -> CompressionValidationResult<()> {
    replacement.validate()?;
    *shared.write().await = replacement;
    Ok(())
}

fn default_protection_rules() -> Vec<String> {
    DEFAULT_PROTECTION_RULES
        .iter()
        .map(|rule| (*rule).to_owned())
        .collect()
}

fn validate_nonempty(field: &str, value: &str, errors: &mut Vec<CompressionConfigError>) {
    if value.trim().is_empty() {
        errors.push(CompressionConfigError::new(field, "must not be empty"));
    }
}

fn validate_path(field: &str, value: &str, errors: &mut Vec<CompressionConfigError>) {
    validate_nonempty(field, value, errors);
    if value.contains('\0') {
        errors.push(CompressionConfigError::new(
            field,
            "must not contain NUL characters",
        ));
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use proptest::prelude::*;

    #[test]
    fn serde_defaults_match_documented_configuration() {
        let config: CompressionConfig = serde_yaml::from_str("{}").unwrap();

        assert_eq!(config, CompressionConfig::default());
        assert!(!config.enabled);
        assert_eq!(config.default_level, CompressionLevel::Lite);
        assert_eq!(config.auto_threshold_tokens, 0);
        assert!(!config.caveman_output);
        assert!(!config.compress_tool_definitions);
        assert_eq!(config.language, "en");
        assert_eq!(config.language_packs_dir, "./language_packs");
        assert_eq!(config.time_budget_ms.lite, 500);
        assert_eq!(config.time_budget_ms.standard, 500);
        assert_eq!(config.time_budget_ms.aggressive, 2_000);
        assert_eq!(config.time_budget_ms.ultra, 2_000);
        assert_eq!(config.time_budget_ms.rtk, 2_000);
        assert_eq!(config.time_budget_ms.stacked, 2_000);
        assert_eq!(config.protection_rules, default_protection_rules());
        assert_eq!(config.rtk.grouping_strategy, RtkGroupingStrategy::Balanced);
        assert!(!config.perplexity.enabled);
        assert_eq!(config.perplexity.redundancy_threshold, 0.5);
        assert_eq!(config.perplexity.compression_ratio_target, 5);
        assert_eq!(
            config.perplexity.model_path,
            "./models/perplexity_scorer.onnx"
        );
        assert!(config.custom_pipelines.is_empty());
        assert!(config.precompressed_contexts.is_empty());
        config.validate().unwrap();
    }

    #[test]
    fn valid_full_configuration_deserializes_and_validates() {
        let config: CompressionConfig = serde_yaml::from_str(
            r#"
enabled: true
default_level: stacked
auto_threshold_tokens: 4096
caveman_output: true
compress_tool_definitions: true
language: de
language_packs_dir: ./packs
time_budget_ms:
  lite: 100
  standard: 200
  aggressive: 300
  ultra: 400
  rtk: 500
  stacked: 600
protection_rules:
  - code_blocks
  - urls
precompressed_contexts:
  - source_path: ./docs/source.md
    compressed_path: ./docs/source.compressed.md
    content_hash: abc123
rtk:
  grouping_strategy: conservative
perplexity:
  enabled: true
  redundancy_threshold: 0.75
  compression_ratio_target: 12
  model_path: ./models/scorer.onnx
custom_pipelines:
  terminal_then_prose:
    engines: [rtk, standard, lite]
"#,
        )
        .unwrap();

        config.validate().unwrap();
        assert!(config.enabled);
        assert_eq!(config.default_level, CompressionLevel::Stacked);
        assert_eq!(
            config.rtk.grouping_strategy,
            RtkGroupingStrategy::Conservative
        );
        assert_eq!(
            config.custom_pipelines["terminal_then_prose"].engines,
            ["rtk", "standard", "lite"]
        );
        assert_eq!(
            config.precompressed_contexts[0].content_hash.as_deref(),
            Some("abc123")
        );
    }

    #[test]
    fn invalid_levels_are_rejected_during_deserialization() {
        let global = serde_yaml::from_str::<CompressionConfig>("default_level: extreme");
        let provider = serde_yaml::from_str::<ProviderCompressionOverride>("level: extreme");
        let model_group = serde_yaml::from_str::<ModelGroupCompressionOverride>("level: extreme");

        for error in [
            global.unwrap_err().to_string(),
            provider.unwrap_err().to_string(),
            model_group.unwrap_err().to_string(),
        ] {
            assert!(error.contains("extreme"));
            assert!(error.contains("lite"));
            assert!(error.contains("none"));
        }
    }

    #[test]
    fn zero_time_budgets_are_rejected_with_field_names() {
        let mut config = CompressionConfig::default();
        config.time_budget_ms.lite = 0;
        config.time_budget_ms.stacked = 0;

        let errors = config.validate().unwrap_err();
        assert!(has_error(&errors, "time_budget_ms.lite", "greater than 0"));
        assert!(has_error(
            &errors,
            "time_budget_ms.stacked",
            "greater than 0"
        ));
    }

    #[test]
    fn invalid_perplexity_thresholds_are_rejected() {
        for threshold in [-0.01, 1.01, f32::NAN, f32::INFINITY] {
            let mut config = CompressionConfig::default();
            config.perplexity.redundancy_threshold = threshold;
            let errors = config.validate().unwrap_err();
            assert!(has_error(
                &errors,
                "perplexity.redundancy_threshold",
                "between 0.0 and 1.0"
            ));
        }
    }

    #[test]
    fn invalid_perplexity_ratios_are_rejected() {
        for ratio in [0, 21, u8::MAX] {
            let mut config = CompressionConfig::default();
            config.perplexity.compression_ratio_target = ratio;
            let errors = config.validate().unwrap_err();
            assert!(has_error(
                &errors,
                "perplexity.compression_ratio_target",
                "between 1 and 20"
            ));
        }
    }

    #[test]
    fn invalid_rtk_strategy_is_rejected_during_deserialization() {
        let error = serde_yaml::from_str::<CompressionConfig>("rtk:\n  grouping_strategy: maximum")
            .unwrap_err()
            .to_string();

        assert!(error.contains("maximum"));
        assert!(error.contains("balanced"));
    }

    #[test]
    fn custom_pipelines_require_nonempty_known_engine_lists() {
        let mut config = CompressionConfig::default();
        config
            .custom_pipelines
            .insert("empty".to_owned(), CustomPipelineConfig { engines: vec![] });
        config.custom_pipelines.insert(
            "unknown".to_owned(),
            CustomPipelineConfig {
                engines: vec!["lite".to_owned(), "invented".to_owned()],
            },
        );
        config.custom_pipelines.insert(
            "  ".to_owned(),
            CustomPipelineConfig {
                engines: vec!["rtk".to_owned()],
            },
        );

        let errors = config.validate().unwrap_err();
        assert!(has_error(
            &errors,
            "custom_pipelines.empty.engines",
            "at least one engine"
        ));
        assert!(has_error(
            &errors,
            "custom_pipelines.unknown.engines[1]",
            "unknown engine `invented`"
        ));
        assert!(has_error(
            &errors,
            "custom_pipelines",
            "pipeline names must not be empty"
        ));
    }

    #[test]
    fn obvious_invalid_paths_are_rejected_without_io() {
        let mut config = CompressionConfig::default();
        config.language_packs_dir = " ".to_owned();
        config.perplexity.model_path = "bad\0model".to_owned();
        config.precompressed_contexts.push(PrecompressedEntry {
            source_path: String::new(),
            compressed_path: "bad\0output".to_owned(),
            content_hash: Some(" ".to_owned()),
        });

        let errors = config.validate().unwrap_err();
        assert!(has_error(
            &errors,
            "language_packs_dir",
            "must not be empty"
        ));
        assert!(has_error(
            &errors,
            "perplexity.model_path",
            "NUL characters"
        ));
        assert!(has_error(
            &errors,
            "precompressed_contexts[0].source_path",
            "must not be empty"
        ));
        assert!(has_error(
            &errors,
            "precompressed_contexts[0].compressed_path",
            "NUL characters"
        ));
        assert!(has_error(
            &errors,
            "precompressed_contexts[0].content_hash",
            "must not be empty"
        ));
    }

    #[test]
    fn resolution_is_per_field_and_preserves_explicit_false_and_none() {
        let global = CompressionConfig {
            enabled: true,
            default_level: CompressionLevel::Ultra,
            auto_threshold_tokens: 1_000,
            ..CompressionConfig::default()
        };
        let provider = ProviderCompressionOverride {
            enabled: Some(false),
            level: Some(CompressionLevel::Standard),
            auto_threshold_tokens: None,
            caveman_output: None,
        };
        let model_group = ModelGroupCompressionOverride {
            level: Some(CompressionLevel::None),
            auto_threshold_tokens: Some(8_000),
            caveman_output: Some(true),
        };

        let effective = global.resolve(Some(&provider), Some(&model_group));

        assert!(!effective.enabled);
        assert_eq!(effective.level, CompressionLevel::None);
        assert_eq!(effective.auto_threshold_tokens, 8_000);
        assert!(effective.caveman_output);
    }

    #[test]
    fn resolution_falls_back_independently_at_each_level() {
        let global = CompressionConfig {
            enabled: false,
            default_level: CompressionLevel::Lite,
            auto_threshold_tokens: 100,
            ..CompressionConfig::default()
        };
        let provider = ProviderCompressionOverride {
            enabled: Some(true),
            level: Some(CompressionLevel::Aggressive),
            auto_threshold_tokens: Some(200),
            caveman_output: Some(true),
        };
        let model_group = ModelGroupCompressionOverride {
            level: None,
            auto_threshold_tokens: Some(300),
            caveman_output: None,
        };

        assert_eq!(
            global.resolve(None, None),
            EffectiveCompressionConfig {
                enabled: false,
                level: CompressionLevel::Lite,
                auto_threshold_tokens: 100,
                caveman_output: false,
            }
        );
        assert_eq!(
            global.resolve(Some(&provider), Some(&model_group)),
            EffectiveCompressionConfig {
                enabled: true,
                level: CompressionLevel::Aggressive,
                auto_threshold_tokens: 300,
                caveman_output: true,
            }
        );
    }

    #[tokio::test]
    async fn live_rwlock_update_is_visible_to_existing_readers() {
        let shared = shared_compression_config(CompressionConfig::default()).unwrap();
        let existing_reader = Arc::clone(&shared);
        let replacement = CompressionConfig {
            enabled: true,
            default_level: CompressionLevel::Aggressive,
            auto_threshold_tokens: 9_999,
            ..CompressionConfig::default()
        };

        reload_compression_config(&shared, replacement)
            .await
            .unwrap();

        let visible = existing_reader.read().await;
        assert!(visible.enabled);
        assert_eq!(visible.default_level, CompressionLevel::Aggressive);
        assert_eq!(visible.auto_threshold_tokens, 9_999);
    }

    #[tokio::test]
    async fn invalid_hot_reload_does_not_replace_live_configuration() {
        let original = CompressionConfig {
            enabled: true,
            ..CompressionConfig::default()
        };
        let shared = shared_compression_config(original.clone()).unwrap();
        let mut invalid = CompressionConfig::default();
        invalid.time_budget_ms.lite = 0;

        assert!(reload_compression_config(&shared, invalid).await.is_err());
        assert_eq!(*shared.read().await, original);
    }

    fn compression_level_strategy() -> impl Strategy<Value = CompressionLevel> {
        prop_oneof![
            Just(CompressionLevel::None),
            Just(CompressionLevel::Lite),
            Just(CompressionLevel::Standard),
            Just(CompressionLevel::Aggressive),
            Just(CompressionLevel::Ultra),
            Just(CompressionLevel::Rtk),
            Just(CompressionLevel::Stacked),
        ]
    }

    fn optional_level_strategy() -> impl Strategy<Value = Option<CompressionLevel>> {
        prop_oneof![Just(None), compression_level_strategy().prop_map(Some)]
    }

    proptest! {
        #![proptest_config(ProptestConfig::with_cases(64))]

        #[test]
        fn property_configuration_precedence_is_field_local(
            global_enabled in any::<bool>(),
            global_level in compression_level_strategy(),
            global_threshold in any::<u32>(),
            provider_enabled in proptest::option::of(any::<bool>()),
            provider_level in optional_level_strategy(),
            provider_threshold in proptest::option::of(any::<u32>()),
            provider_caveman in proptest::option::of(any::<bool>()),
            model_level in optional_level_strategy(),
            model_threshold in proptest::option::of(any::<u32>()),
            model_caveman in proptest::option::of(any::<bool>()),
        ) {
            let global = CompressionConfig {
                enabled: global_enabled,
                default_level: global_level,
                auto_threshold_tokens: global_threshold,
                ..CompressionConfig::default()
            };
            let provider = ProviderCompressionOverride {
                enabled: provider_enabled,
                level: provider_level,
                auto_threshold_tokens: provider_threshold,
                caveman_output: provider_caveman,
            };
            let model_group = ModelGroupCompressionOverride {
                level: model_level,
                auto_threshold_tokens: model_threshold,
                caveman_output: model_caveman,
            };

            let effective = global.resolve(Some(&provider), Some(&model_group));

            prop_assert_eq!(effective.enabled, provider_enabled.unwrap_or(global_enabled));
            prop_assert_eq!(
                effective.level,
                model_level.or(provider_level).unwrap_or(global_level)
            );
            prop_assert_eq!(
                effective.auto_threshold_tokens,
                model_threshold.or(provider_threshold).unwrap_or(global_threshold)
            );
            prop_assert_eq!(
                effective.caveman_output,
                model_caveman.or(provider_caveman).unwrap_or(global.caveman_output)
            );
        }
    }

    fn has_error(errors: &[CompressionConfigError], field: &str, message: &str) -> bool {
        errors
            .iter()
            .any(|error| error.field == field && error.message.contains(message))
    }
}
