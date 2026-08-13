//! Tool Definition Compression configuration types.
//!
//! All config structs implement `Default`, `Serialize`, `Deserialize` with
//! sane defaults so that an absent `tool_compression` section in config.yaml
//! results in the feature being completely disabled with zero performance impact.

use std::collections::HashMap;

use serde::{Deserialize, Serialize};

// ─── Helper default fns ───────────────────────────────────────────────────────

fn default_true() -> bool {
    true
}

fn default_compression_level() -> CompressionLevel {
    CompressionLevel::Medium
}

fn default_min_requests() -> u32 {
    5
}

fn default_min_preserve_length() -> u32 {
    20
}

fn default_tool_truncation() -> TruncationMode {
    TruncationMode::FirstSentence
}

fn default_param_truncation() -> ParamTruncationMode {
    ParamTruncationMode::Remove
}

fn default_embedding_model() -> String {
    "builtin-minilm".to_string()
}

fn default_top_k() -> u32 {
    20
}

fn default_similarity_threshold() -> f32 {
    0.3
}

fn default_frequency_weight() -> f32 {
    0.3
}

fn default_error_threshold() -> f32 {
    0.10
}

fn default_recovery_window() -> u32 {
    50
}

fn default_rolling_window() -> u32 {
    100
}

fn default_min_tools_for_grouping() -> u32 {
    10
}

fn default_description_method() -> DescriptionCompressionMethod {
    DescriptionCompressionMethod::Tfidf
}

// ─── Top-level ToolCompressionConfig ──────────────────────────────────────────

/// Complete tool definition compression configuration.
///
/// When absent from the gateway config, all fields default to disabled/inactive
/// values resulting in zero performance impact.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct ToolCompressionConfig {
    #[serde(default)]
    pub enabled: bool,

    #[serde(default = "default_compression_level")]
    pub level: CompressionLevel,

    #[serde(default)]
    pub progressive_disclosure: bool,

    #[serde(default)]
    pub pruning: PruningConfig,

    #[serde(default = "default_true")]
    pub cache_placement: bool,

    #[serde(default = "default_true")]
    pub deduplication: bool,

    #[serde(default)]
    pub minification: MinificationConfig,

    #[serde(default)]
    pub description_truncation: DescriptionTruncationConfig,

    #[serde(default)]
    pub semantic_retrieval: SemanticRetrievalConfig,

    #[serde(default)]
    pub canonical_rewriting: CanonicalRewritingConfig,

    #[serde(default)]
    pub feedback_loop: FeedbackLoopConfig,

    #[serde(default)]
    pub auto_tuning: AutoTuningConfig,

    #[serde(default)]
    pub namespace_grouping: NamespaceGroupingConfig,

    #[serde(default)]
    pub precomputed_descriptions: PrecomputedDescriptionsConfig,

    /// Per-model-group overrides. Group name → partial compression settings.
    #[serde(default)]
    pub model_group_overrides: HashMap<String, ToolCompressionOverride>,

    /// Enable post-stage validation in the pipeline (debug/development mode).
    /// When `true`, each pipeline stage's output is validated for structural
    /// correctness. Failures are logged as warnings without blocking the request.
    /// Default: false (zero cost when disabled).
    #[serde(default)]
    pub debug_validation: bool,

    /// Per-provider capability overrides. Provider name → partial capability settings.
    /// Merged on top of built-in defaults when constructing `ToolCompressionState`.
    #[serde(default)]
    pub provider_overrides: HashMap<String, crate::tool_compression::types::ProviderCapsOverlay>,
}

impl Default for ToolCompressionConfig {
    fn default() -> Self {
        Self {
            enabled: false,
            level: default_compression_level(),
            progressive_disclosure: false,
            pruning: PruningConfig::default(),
            cache_placement: true,
            deduplication: true,
            minification: MinificationConfig::default(),
            description_truncation: DescriptionTruncationConfig::default(),
            semantic_retrieval: SemanticRetrievalConfig::default(),
            canonical_rewriting: CanonicalRewritingConfig::default(),
            feedback_loop: FeedbackLoopConfig::default(),
            auto_tuning: AutoTuningConfig::default(),
            namespace_grouping: NamespaceGroupingConfig::default(),
            precomputed_descriptions: PrecomputedDescriptionsConfig::default(),
            model_group_overrides: HashMap::new(),
            debug_validation: false,
            provider_overrides: HashMap::new(),
        }
    }
}

// ─── CompressionLevel ─────────────────────────────────────────────────────────

/// Compression aggressiveness presets.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum CompressionLevel {
    Low,
    Medium,
    High,
    Max,
}

// ─── PruningConfig ────────────────────────────────────────────────────────────

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct PruningConfig {
    #[serde(default)]
    pub enabled: bool,

    /// Minimum session requests before pruning activates (range: 2–50).
    #[serde(default = "default_min_requests")]
    pub min_requests: u32,

    /// Tool name patterns (exact or glob) that are never pruned.
    #[serde(default)]
    pub always_include: Vec<String>,
}

impl Default for PruningConfig {
    fn default() -> Self {
        Self {
            enabled: false,
            min_requests: 5,
            always_include: Vec::new(),
        }
    }
}

// ─── MinificationConfig ───────────────────────────────────────────────────────

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct MinificationConfig {
    #[serde(default = "default_true")]
    pub remove_titles: bool,

    #[serde(default = "default_true")]
    pub collapse_single_unions: bool,

    #[serde(default = "default_true")]
    pub remove_additional_properties: bool,

    #[serde(default = "default_true")]
    pub remove_empty_descriptions: bool,
}

impl Default for MinificationConfig {
    fn default() -> Self {
        Self {
            remove_titles: true,
            collapse_single_unions: true,
            remove_additional_properties: true,
            remove_empty_descriptions: true,
        }
    }
}

// ─── DescriptionTruncationConfig ──────────────────────────────────────────────

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct DescriptionTruncationConfig {
    #[serde(default = "default_tool_truncation")]
    pub tool_level: TruncationMode,

    #[serde(default = "default_param_truncation")]
    pub parameter_level: ParamTruncationMode,

    #[serde(default = "default_true")]
    pub remove_examples: bool,

    /// Descriptions at or below this character count are preserved unchanged.
    #[serde(default = "default_min_preserve_length")]
    pub min_preserve_length: u32,
}

impl Default for DescriptionTruncationConfig {
    fn default() -> Self {
        Self {
            tool_level: default_tool_truncation(),
            parameter_level: default_param_truncation(),
            remove_examples: true,
            min_preserve_length: 20,
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum TruncationMode {
    None,
    FirstSentence,
    Remove,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum ParamTruncationMode {
    None,
    Remove,
}

// ─── SemanticRetrievalConfig ──────────────────────────────────────────────────

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct SemanticRetrievalConfig {
    #[serde(default)]
    pub enabled: bool,

    /// Embedding source:
    /// - `"builtin-minilm"` — local ONNX model via OnnxAssetManager
    /// - `"provider:<name>"` — external embedding API
    /// - `"precomputed"` — load from `precomputed_vectors_path`
    #[serde(default = "default_embedding_model")]
    pub embedding_model: String,

    /// Maximum tools to include in active set.
    #[serde(default = "default_top_k")]
    pub top_k: u32,

    /// Minimum hybrid score for inclusion (0.0–1.0).
    #[serde(default = "default_similarity_threshold")]
    pub similarity_threshold: f32,

    /// Weight for frequency vs semantic (0.0 = pure semantic, 1.0 = pure frequency).
    #[serde(default = "default_frequency_weight")]
    pub frequency_weight: f32,

    /// Optional path to pre-computed embedding vectors file.
    #[serde(default)]
    pub precomputed_vectors_path: Option<String>,
}

impl Default for SemanticRetrievalConfig {
    fn default() -> Self {
        Self {
            enabled: false,
            embedding_model: default_embedding_model(),
            top_k: 20,
            similarity_threshold: 0.3,
            frequency_weight: 0.3,
            precomputed_vectors_path: None,
        }
    }
}

// ─── CanonicalRewritingConfig ─────────────────────────────────────────────────

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct CanonicalRewritingConfig {
    #[serde(default)]
    pub enabled: bool,

    /// Glob patterns for model names allowed to receive canonical format.
    #[serde(default)]
    pub allowed_models: Vec<String>,
}

impl Default for CanonicalRewritingConfig {
    fn default() -> Self {
        Self {
            enabled: false,
            allowed_models: Vec::new(),
        }
    }
}

// ─── FeedbackLoopConfig ───────────────────────────────────────────────────────

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct FeedbackLoopConfig {
    #[serde(default = "default_true")]
    pub enabled: bool,

    /// Fractional increase over baseline that triggers level reduction.
    #[serde(default = "default_error_threshold")]
    pub error_threshold: f32,

    /// Sustained low-error requests before attempting level increase.
    #[serde(default = "default_recovery_window")]
    pub recovery_window: u32,

    /// Number of recent requests tracked per model group.
    #[serde(default = "default_rolling_window")]
    pub rolling_window: u32,
}

impl Default for FeedbackLoopConfig {
    fn default() -> Self {
        Self {
            enabled: true,
            error_threshold: 0.10,
            recovery_window: 50,
            rolling_window: 100,
        }
    }
}

// ─── AutoTuningConfig ─────────────────────────────────────────────────────────

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct AutoTuningConfig {
    #[serde(default = "default_true")]
    pub enabled: bool,

    /// Model name glob pattern → tier level (1–3).
    #[serde(default)]
    pub model_tiers: HashMap<String, u8>,
}

impl Default for AutoTuningConfig {
    fn default() -> Self {
        Self {
            enabled: true,
            model_tiers: HashMap::new(),
        }
    }
}

// ─── NamespaceGroupingConfig ──────────────────────────────────────────────────

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct NamespaceGroupingConfig {
    #[serde(default)]
    pub enabled: bool,

    /// Minimum tools required to activate namespace grouping.
    #[serde(default = "default_min_tools_for_grouping")]
    pub min_tools_for_grouping: u32,

    /// Explicit namespace mappings: tool name prefix → namespace metadata.
    #[serde(default)]
    pub namespace_mappings: HashMap<String, NamespaceMetadata>,
}

impl Default for NamespaceGroupingConfig {
    fn default() -> Self {
        Self {
            enabled: true,
            min_tools_for_grouping: 10,
            namespace_mappings: HashMap::new(),
        }
    }
}

/// Metadata for a configured namespace.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct NamespaceMetadata {
    pub name: String,
    pub description: String,
}

// ─── PrecomputedDescriptionsConfig ────────────────────────────────────────────

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct PrecomputedDescriptionsConfig {
    #[serde(default)]
    pub enabled: bool,

    /// Compression method: tfidf, manual, or model.
    #[serde(default = "default_description_method")]
    pub method: DescriptionCompressionMethod,

    /// Manual compressed descriptions: tool name → compressed text.
    #[serde(default)]
    pub descriptions: HashMap<String, String>,
}

impl Default for PrecomputedDescriptionsConfig {
    fn default() -> Self {
        Self {
            enabled: false,
            method: DescriptionCompressionMethod::Tfidf,
            descriptions: HashMap::new(),
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum DescriptionCompressionMethod {
    Tfidf,
    Manual,
    Model,
}

// ─── ToolCompressionOverride ──────────────────────────────────────────────────

/// Partial override for per-model-group configuration.
/// All fields are `Option` — only present fields override the global default.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize, Default)]
pub struct ToolCompressionOverride {
    pub enabled: Option<bool>,
    pub level: Option<CompressionLevel>,
    pub progressive_disclosure: Option<bool>,
    pub pruning: Option<PruningConfig>,
    pub cache_placement: Option<bool>,
    pub deduplication: Option<bool>,
    pub minification: Option<MinificationConfig>,
    pub description_truncation: Option<DescriptionTruncationConfig>,
    pub semantic_retrieval: Option<SemanticRetrievalConfig>,
    pub canonical_rewriting: Option<CanonicalRewritingConfig>,
    pub feedback_loop: Option<FeedbackLoopConfig>,
    pub auto_tuning: Option<AutoTuningConfig>,
    pub namespace_grouping: Option<NamespaceGroupingConfig>,
    pub precomputed_descriptions: Option<PrecomputedDescriptionsConfig>,
}

// ─── Tests ────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn default_config_is_disabled() {
        let cfg = ToolCompressionConfig::default();
        assert!(!cfg.enabled);
        assert_eq!(cfg.level, CompressionLevel::Medium);
        assert!(!cfg.progressive_disclosure);
        assert!(cfg.cache_placement);
        assert!(cfg.deduplication);
        assert!(cfg.model_group_overrides.is_empty());
    }

    #[test]
    fn default_pruning() {
        let p = PruningConfig::default();
        assert!(!p.enabled);
        assert_eq!(p.min_requests, 5);
        assert!(p.always_include.is_empty());
    }

    #[test]
    fn default_minification_all_true() {
        let m = MinificationConfig::default();
        assert!(m.remove_titles);
        assert!(m.collapse_single_unions);
        assert!(m.remove_additional_properties);
        assert!(m.remove_empty_descriptions);
    }

    #[test]
    fn default_description_truncation() {
        let dt = DescriptionTruncationConfig::default();
        assert_eq!(dt.tool_level, TruncationMode::FirstSentence);
        assert_eq!(dt.parameter_level, ParamTruncationMode::Remove);
        assert!(dt.remove_examples);
        assert_eq!(dt.min_preserve_length, 20);
    }

    #[test]
    fn default_semantic_retrieval() {
        let sr = SemanticRetrievalConfig::default();
        assert!(!sr.enabled);
        assert_eq!(sr.embedding_model, "builtin-minilm");
        assert_eq!(sr.top_k, 20);
        assert!((sr.similarity_threshold - 0.3).abs() < f32::EPSILON);
        assert!((sr.frequency_weight - 0.3).abs() < f32::EPSILON);
        assert!(sr.precomputed_vectors_path.is_none());
    }

    #[test]
    fn default_canonical_rewriting() {
        let cr = CanonicalRewritingConfig::default();
        assert!(!cr.enabled);
        assert!(cr.allowed_models.is_empty());
    }

    #[test]
    fn default_feedback_loop() {
        let fl = FeedbackLoopConfig::default();
        assert!(fl.enabled);
        assert!((fl.error_threshold - 0.10).abs() < f32::EPSILON);
        assert_eq!(fl.recovery_window, 50);
        assert_eq!(fl.rolling_window, 100);
    }

    #[test]
    fn default_auto_tuning() {
        let at = AutoTuningConfig::default();
        assert!(at.enabled);
        assert!(at.model_tiers.is_empty());
    }

    #[test]
    fn default_namespace_grouping() {
        let ng = NamespaceGroupingConfig::default();
        // Enabled by default: namespace grouping is the tool-count-reduction strategy
        // that keeps the provider-visible tool definitions under provider caps.
        assert!(ng.enabled);
        assert_eq!(ng.min_tools_for_grouping, 10);
        assert!(ng.namespace_mappings.is_empty());
    }

    #[test]
    fn default_precomputed_descriptions() {
        let pd = PrecomputedDescriptionsConfig::default();
        assert!(!pd.enabled);
        assert_eq!(pd.method, DescriptionCompressionMethod::Tfidf);
        assert!(pd.descriptions.is_empty());
    }

    #[test]
    fn deserialize_empty_yaml_produces_defaults() {
        let yaml = "{}";
        let cfg: ToolCompressionConfig = serde_yaml::from_str(yaml).unwrap();
        assert_eq!(cfg, ToolCompressionConfig::default());
    }

    #[test]
    fn deserialize_partial_yaml() {
        let yaml = r#"
enabled: true
level: high
pruning:
  enabled: true
  min_requests: 10
"#;
        let cfg: ToolCompressionConfig = serde_yaml::from_str(yaml).unwrap();
        assert!(cfg.enabled);
        assert_eq!(cfg.level, CompressionLevel::High);
        assert!(cfg.pruning.enabled);
        assert_eq!(cfg.pruning.min_requests, 10);
        // Remaining fields should be defaults
        assert!(cfg.cache_placement);
        assert!(cfg.deduplication);
    }

    #[test]
    fn deserialize_override_with_optional_fields() {
        let yaml = r#"
enabled: true
level: low
"#;
        let ovr: ToolCompressionOverride = serde_yaml::from_str(yaml).unwrap();
        assert_eq!(ovr.enabled, Some(true));
        assert_eq!(ovr.level, Some(CompressionLevel::Low));
        assert!(ovr.pruning.is_none());
        assert!(ovr.minification.is_none());
    }

    #[test]
    fn roundtrip_serialize_deserialize() {
        let cfg = ToolCompressionConfig {
            enabled: true,
            level: CompressionLevel::Max,
            progressive_disclosure: true,
            ..Default::default()
        };
        let yaml = serde_yaml::to_string(&cfg).unwrap();
        let restored: ToolCompressionConfig = serde_yaml::from_str(&yaml).unwrap();
        assert_eq!(cfg, restored);
    }
}
