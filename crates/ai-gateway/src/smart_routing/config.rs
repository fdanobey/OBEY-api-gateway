//! Smart model routing configuration and validation.

use std::collections::HashMap;
use std::fmt;

use serde::{Deserialize, Serialize};

/// Largest declared model context accepted by configuration validation.
///
/// `0` remains the explicit "unknown" sentinel. Ten million tokens leaves
/// headroom above current production model limits while rejecting accidental
/// byte counts, prices, and other implausible values entered as token limits.
pub const MAX_CONTEXT_WINDOW_TOKENS: u32 = 10_000_000;

const MAX_MODEL_PATH_CHARS: usize = 4096;
const MAX_MODEL_NAME_CHARS: usize = 256;
const MAX_STATE_PATH_CHARS: usize = 4096;
const MAX_EMBEDDING_MODEL_CHARS: usize = 256;
const MAX_GROUP_NAME_CHARS: usize = 256;
const MAX_CACHE_ENTRIES: usize = 1_000_000;
const MAX_TTL_SECS: u64 = 31_536_000;
const MAX_OPTIMIZER_INTERVAL_SECS: u64 = 604_800;
const MAX_TRAINING_BATCH_SIZE: usize = 4096;
const MAX_TRAINING_EPOCHS: usize = 1000;
const WEIGHT_SUM_TOLERANCE: f64 = 1.0e-6;

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(default)]
pub struct SmartRoutingConfig {
    #[serde(default)]
    pub enabled: bool,
    #[serde(default)]
    pub classifier: ClassifierMode,
    #[serde(default)]
    pub ml_model_path: Option<String>,
    #[serde(default)]
    pub classifier_model: Option<String>,
    #[serde(default = "default_cost_quality_threshold")]
    pub cost_quality_threshold: f64,
    #[serde(default)]
    pub cascade: CascadeConfig,
    #[serde(default)]
    pub tier_boundaries: TierBoundaries,
    #[serde(default)]
    pub heuristic_weights: Option<HeuristicWeights>,
    #[serde(default)]
    pub composite_weights: Option<CompositeWeights>,
    #[serde(default)]
    pub model_group_overrides: HashMap<String, SmartRoutingOverride>,
    #[serde(default)]
    pub streaming_cascade_mode: StreamingCascadeMode,
    #[serde(default)]
    pub allow_unknown_context_window: bool,
    #[serde(default = "default_reserved_output_tokens")]
    pub reserved_output_tokens: u32,
    #[serde(default = "default_provider_overhead_tokens")]
    pub provider_overhead_tokens: u32,
    #[serde(default = "default_context_safety_margin_tokens")]
    pub context_safety_margin_tokens: u32,
    #[serde(default)]
    pub online_optimizer: OnlineOptimizerConfig,
    #[serde(default)]
    pub semantic_cache: SmartRoutingSemanticCacheConfig,
    #[serde(default)]
    pub quality_evaluator: QualityEvaluatorConfig,
    #[serde(default)]
    pub budget_limits: HashMap<String, BudgetLimits>,
    #[serde(default)]
    pub ab_test: Option<ABTestConfig>,
    #[serde(default)]
    pub training: TrainingConfig,
}

impl Default for SmartRoutingConfig {
    fn default() -> Self {
        Self {
            enabled: false,
            classifier: ClassifierMode::Heuristic,
            ml_model_path: None,
            classifier_model: None,
            cost_quality_threshold: default_cost_quality_threshold(),
            cascade: CascadeConfig::default(),
            tier_boundaries: TierBoundaries::default(),
            heuristic_weights: None,
            composite_weights: None,
            model_group_overrides: HashMap::new(),
            streaming_cascade_mode: StreamingCascadeMode::Buffer,
            allow_unknown_context_window: false,
            reserved_output_tokens: default_reserved_output_tokens(),
            provider_overhead_tokens: default_provider_overhead_tokens(),
            context_safety_margin_tokens: default_context_safety_margin_tokens(),
            online_optimizer: OnlineOptimizerConfig::default(),
            semantic_cache: SmartRoutingSemanticCacheConfig::default(),
            quality_evaluator: QualityEvaluatorConfig::default(),
            budget_limits: HashMap::new(),
            ab_test: None,
            training: TrainingConfig::default(),
        }
    }
}

impl SmartRoutingConfig {
    /// Validate all global, per-group, experiment, budget, and training values.
    pub fn validate(&self) -> SmartRoutingValidationResult<()> {
        let mut errors = Vec::new();
        RoutingPolicySnapshot::from(self).validate_into("", &mut errors);
        self.training.validate_into("training", &mut errors);

        for (group, limits) in &self.budget_limits {
            validate_map_key("budget_limits", group, &mut errors);
            limits.validate_into(&format!("budget_limits.{group}"), &mut errors);
        }

        if let Some(ab_test) = &self.ab_test {
            ab_test.validate_into("ab_test", &mut errors);
        }

        for (group, config_override) in &self.model_group_overrides {
            validate_map_key("model_group_overrides", group, &mut errors);
            let effective = self.effective_for_group(group);
            RoutingPolicySnapshot::from(&effective)
                .validate_into(&format!("model_group_overrides.{group}"), &mut errors);
            effective.training.validate_into(
                &format!("model_group_overrides.{group}.training"),
                &mut errors,
            );
            config_override.validate_present_values_into(
                &format!("model_group_overrides.{group}"),
                &mut errors,
            );
        }

        validation_result(errors)
    }

    /// Resolve one model group's effective routing settings.
    ///
    /// Every present override field replaces the corresponding global field;
    /// omitted fields retain the global value. The returned snapshot clears the
    /// override map so applying it repeatedly cannot recurse or re-merge.
    pub fn effective_for_group(&self, model_group: &str) -> Self {
        let mut effective = self.clone();
        effective.model_group_overrides.clear();

        let Some(config_override) = self.model_group_overrides.get(model_group) else {
            return effective;
        };

        if let Some(value) = config_override.enabled {
            effective.enabled = value;
        }
        if let Some(value) = config_override.classifier {
            effective.classifier = value;
        }
        if let Some(value) = &config_override.ml_model_path {
            effective.ml_model_path = value.clone();
        }
        if let Some(value) = &config_override.classifier_model {
            effective.classifier_model = value.clone();
        }
        if let Some(value) = config_override.cost_quality_threshold {
            effective.cost_quality_threshold = value;
        }
        if let Some(value) = &config_override.cascade {
            effective.cascade = value.clone();
        }
        if let Some(value) = &config_override.tier_boundaries {
            effective.tier_boundaries = value.clone();
        }
        if let Some(value) = &config_override.heuristic_weights {
            effective.heuristic_weights = Some(value.clone());
        }
        if let Some(value) = &config_override.composite_weights {
            effective.composite_weights = Some(value.clone());
        }
        if let Some(value) = config_override.streaming_cascade_mode {
            effective.streaming_cascade_mode = value;
        }
        if let Some(value) = config_override.allow_unknown_context_window {
            effective.allow_unknown_context_window = value;
        }
        if let Some(value) = config_override.reserved_output_tokens {
            effective.reserved_output_tokens = value;
        }
        if let Some(value) = config_override.provider_overhead_tokens {
            effective.provider_overhead_tokens = value;
        }
        if let Some(value) = config_override.context_safety_margin_tokens {
            effective.context_safety_margin_tokens = value;
        }
        if let Some(value) = &config_override.online_optimizer {
            effective.online_optimizer = value.clone();
        }
        if let Some(value) = &config_override.semantic_cache {
            effective.semantic_cache = value.clone();
        }
        if let Some(value) = &config_override.quality_evaluator {
            effective.quality_evaluator = value.clone();
        }
        if let Some(value) = &config_override.training {
            effective.training = value.clone();
        }

        effective
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum ClassifierMode {
    Heuristic,
    Ml,
    Llm,
    Composite,
}

impl Default for ClassifierMode {
    fn default() -> Self {
        Self::Heuristic
    }
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(default)]
pub struct TierBoundaries {
    #[serde(default = "default_fast_max")]
    pub fast_max: f64,
    #[serde(default = "default_balanced_max")]
    pub balanced_max: f64,
}

impl Default for TierBoundaries {
    fn default() -> Self {
        Self {
            fast_max: default_fast_max(),
            balanced_max: default_balanced_max(),
        }
    }
}

impl TierBoundaries {
    fn validate_into(&self, scope: &str, errors: &mut Vec<SmartRoutingConfigError>) {
        validate_finite_open_unit(&field(scope, "fast_max"), self.fast_max, errors);
        validate_finite_open_unit(&field(scope, "balanced_max"), self.balanced_max, errors);
        if self.fast_max.is_finite()
            && self.balanced_max.is_finite()
            && self.fast_max >= self.balanced_max
        {
            errors.push(SmartRoutingConfigError::new(
                field(scope, "fast_max"),
                format!(
                    "is {}; expected a value smaller than balanced_max ({})",
                    self.fast_max, self.balanced_max
                ),
            ));
        }
    }
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(default)]
pub struct CascadeConfig {
    #[serde(default)]
    pub enabled: bool,
    #[serde(default = "default_max_escalations")]
    pub max_escalations: u8,
    #[serde(default = "default_min_response_tokens")]
    pub min_response_tokens: u32,
    #[serde(default = "default_early_signal_tokens")]
    pub early_signal_tokens: u32,
}

impl Default for CascadeConfig {
    fn default() -> Self {
        Self {
            enabled: false,
            max_escalations: default_max_escalations(),
            min_response_tokens: default_min_response_tokens(),
            early_signal_tokens: default_early_signal_tokens(),
        }
    }
}

impl CascadeConfig {
    fn validate_into(&self, scope: &str, errors: &mut Vec<SmartRoutingConfigError>) {
        if !(1..=2).contains(&self.max_escalations) {
            errors.push(SmartRoutingConfigError::new(
                field(scope, "max_escalations"),
                format!("is {}; expected an integer in 1..=2", self.max_escalations),
            ));
        }
        validate_u32_range(
            &field(scope, "min_response_tokens"),
            self.min_response_tokens,
            1,
            65_536,
            errors,
        );
        validate_u32_range(
            &field(scope, "early_signal_tokens"),
            self.early_signal_tokens,
            1,
            65_536,
            errors,
        );
    }
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(default)]
pub struct HeuristicWeights {
    #[serde(default = "default_equal_heuristic_weight")]
    pub message_count: f64,
    #[serde(default = "default_equal_heuristic_weight")]
    pub token_estimate: f64,
    #[serde(default = "default_equal_heuristic_weight")]
    pub code_blocks: f64,
    #[serde(default = "default_equal_heuristic_weight")]
    pub tool_calls: f64,
    #[serde(default = "default_equal_heuristic_weight")]
    pub math_expressions: f64,
    #[serde(default = "default_equal_heuristic_weight")]
    pub reasoning_keywords: f64,
}

impl Default for HeuristicWeights {
    fn default() -> Self {
        let equal = default_equal_heuristic_weight();
        Self {
            message_count: equal,
            token_estimate: equal,
            code_blocks: equal,
            tool_calls: equal,
            math_expressions: equal,
            reasoning_keywords: equal,
        }
    }
}

impl HeuristicWeights {
    fn validate_into(&self, scope: &str, errors: &mut Vec<SmartRoutingConfigError>) {
        let weights = [
            ("message_count", self.message_count),
            ("token_estimate", self.token_estimate),
            ("code_blocks", self.code_blocks),
            ("tool_calls", self.tool_calls),
            ("math_expressions", self.math_expressions),
            ("reasoning_keywords", self.reasoning_keywords),
        ];
        validate_weights(scope, &weights, errors);
    }
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(default)]
pub struct CompositeWeights {
    #[serde(default = "default_composite_heuristic_weight")]
    pub heuristic: f64,
    #[serde(default = "default_composite_ml_weight")]
    pub ml: f64,
}

impl Default for CompositeWeights {
    fn default() -> Self {
        Self {
            heuristic: default_composite_heuristic_weight(),
            ml: default_composite_ml_weight(),
        }
    }
}

impl CompositeWeights {
    fn validate_into(&self, scope: &str, errors: &mut Vec<SmartRoutingConfigError>) {
        validate_weights(
            scope,
            &[("heuristic", self.heuristic), ("ml", self.ml)],
            errors,
        );
    }
}

/// Per-model-group partial routing settings.
#[derive(Debug, Clone, Default, PartialEq, Serialize, Deserialize)]
#[serde(default)]
pub struct SmartRoutingOverride {
    #[serde(default)]
    pub enabled: Option<bool>,
    #[serde(default)]
    pub classifier: Option<ClassifierMode>,
    #[serde(default)]
    pub ml_model_path: Option<Option<String>>,
    #[serde(default)]
    pub classifier_model: Option<Option<String>>,
    #[serde(default)]
    pub cost_quality_threshold: Option<f64>,
    #[serde(default)]
    pub cascade: Option<CascadeConfig>,
    #[serde(default)]
    pub tier_boundaries: Option<TierBoundaries>,
    #[serde(default)]
    pub heuristic_weights: Option<HeuristicWeights>,
    #[serde(default)]
    pub composite_weights: Option<CompositeWeights>,
    #[serde(default)]
    pub streaming_cascade_mode: Option<StreamingCascadeMode>,
    #[serde(default)]
    pub allow_unknown_context_window: Option<bool>,
    #[serde(default)]
    pub reserved_output_tokens: Option<u32>,
    #[serde(default)]
    pub provider_overhead_tokens: Option<u32>,
    #[serde(default)]
    pub context_safety_margin_tokens: Option<u32>,
    #[serde(default)]
    pub online_optimizer: Option<OnlineOptimizerConfig>,
    #[serde(default)]
    pub semantic_cache: Option<SmartRoutingSemanticCacheConfig>,
    #[serde(default)]
    pub quality_evaluator: Option<QualityEvaluatorConfig>,
    #[serde(default)]
    pub training: Option<TrainingConfig>,
}

impl SmartRoutingOverride {
    fn validate_present_values_into(&self, scope: &str, errors: &mut Vec<SmartRoutingConfigError>) {
        if let Some(path) = self.ml_model_path.as_ref().and_then(Option::as_ref) {
            validate_optional_text(
                &field(scope, "ml_model_path"),
                path,
                MAX_MODEL_PATH_CHARS,
                errors,
            );
        }
        if let Some(model) = self.classifier_model.as_ref().and_then(Option::as_ref) {
            validate_optional_text(
                &field(scope, "classifier_model"),
                model,
                MAX_MODEL_NAME_CHARS,
                errors,
            );
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum StreamingCascadeMode {
    Buffer,
    EarlySignal,
}

impl Default for StreamingCascadeMode {
    fn default() -> Self {
        Self::Buffer
    }
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(default)]
pub struct OnlineOptimizerConfig {
    #[serde(default)]
    pub enabled: bool,
    #[serde(default = "default_optimizer_alpha")]
    pub alpha: f64,
    #[serde(default = "default_optimizer_interval_secs")]
    pub interval_secs: u64,
    #[serde(default)]
    pub state_path: Option<String>,
    #[serde(default = "default_optimizer_quality_threshold")]
    pub quality_threshold: f64,
}

impl Default for OnlineOptimizerConfig {
    fn default() -> Self {
        Self {
            enabled: false,
            alpha: default_optimizer_alpha(),
            interval_secs: default_optimizer_interval_secs(),
            state_path: None,
            quality_threshold: default_optimizer_quality_threshold(),
        }
    }
}

impl OnlineOptimizerConfig {
    fn validate_into(&self, scope: &str, errors: &mut Vec<SmartRoutingConfigError>) {
        validate_finite_range(
            &field(scope, "alpha"),
            self.alpha,
            f64::MIN_POSITIVE,
            1.0,
            "a finite value greater than 0.0 and at most 1.0",
            errors,
        );
        if !(1..=MAX_OPTIMIZER_INTERVAL_SECS).contains(&self.interval_secs) {
            errors.push(SmartRoutingConfigError::new(
                field(scope, "interval_secs"),
                format!(
                    "is {}; expected an integer in 1..={MAX_OPTIMIZER_INTERVAL_SECS}",
                    self.interval_secs
                ),
            ));
        }
        if let Some(path) = &self.state_path {
            validate_optional_text(
                &field(scope, "state_path"),
                path,
                MAX_STATE_PATH_CHARS,
                errors,
            );
        }
        validate_finite_closed_unit(
            &field(scope, "quality_threshold"),
            self.quality_threshold,
            errors,
        );
    }
}

/// Smart-routing's response cache settings, distinct from the gateway's
/// existing top-level semantic cache configuration.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(default)]
pub struct SmartRoutingSemanticCacheConfig {
    #[serde(default)]
    pub enabled: bool,
    #[serde(default = "default_similarity_threshold")]
    pub similarity_threshold: f64,
    #[serde(default = "default_semantic_cache_max_entries")]
    pub max_entries: usize,
    #[serde(default = "default_semantic_cache_ttl_secs")]
    pub ttl_secs: u64,
    #[serde(default = "default_embedding_model")]
    pub embedding_model: String,
    #[serde(default = "default_min_quality_score")]
    pub min_quality_score: f64,
}

/// Compatibility name for callers expecting the design's shorter type name.
pub type SemanticCacheConfig = SmartRoutingSemanticCacheConfig;

impl Default for SmartRoutingSemanticCacheConfig {
    fn default() -> Self {
        Self {
            enabled: false,
            similarity_threshold: default_similarity_threshold(),
            max_entries: default_semantic_cache_max_entries(),
            ttl_secs: default_semantic_cache_ttl_secs(),
            embedding_model: default_embedding_model(),
            min_quality_score: default_min_quality_score(),
        }
    }
}

impl SmartRoutingSemanticCacheConfig {
    fn validate_into(&self, scope: &str, errors: &mut Vec<SmartRoutingConfigError>) {
        validate_finite_closed_unit(
            &field(scope, "similarity_threshold"),
            self.similarity_threshold,
            errors,
        );
        if !(1..=MAX_CACHE_ENTRIES).contains(&self.max_entries) {
            errors.push(SmartRoutingConfigError::new(
                field(scope, "max_entries"),
                format!(
                    "is {}; expected an integer in 1..={MAX_CACHE_ENTRIES}",
                    self.max_entries
                ),
            ));
        }
        if !(1..=MAX_TTL_SECS).contains(&self.ttl_secs) {
            errors.push(SmartRoutingConfigError::new(
                field(scope, "ttl_secs"),
                format!(
                    "is {}; expected an integer in 1..={MAX_TTL_SECS}",
                    self.ttl_secs
                ),
            ));
        }
        validate_required_text(
            &field(scope, "embedding_model"),
            &self.embedding_model,
            MAX_EMBEDDING_MODEL_CHARS,
            errors,
        );
        validate_finite_closed_unit(
            &field(scope, "min_quality_score"),
            self.min_quality_score,
            errors,
        );
    }
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(default)]
pub struct QualityEvaluatorConfig {
    #[serde(default)]
    pub enabled: bool,
    #[serde(default = "default_quality_evaluator_threshold")]
    pub threshold: f64,
}

impl Default for QualityEvaluatorConfig {
    fn default() -> Self {
        Self {
            enabled: false,
            threshold: default_quality_evaluator_threshold(),
        }
    }
}

impl QualityEvaluatorConfig {
    fn validate_into(&self, scope: &str, errors: &mut Vec<SmartRoutingConfigError>) {
        validate_finite_closed_unit(&field(scope, "threshold"), self.threshold, errors);
    }
}

/// A non-recursive routing policy used by A/B experiment arms.
///
/// It deliberately excludes `ab_test`, `model_group_overrides`, budgets, and
/// training jobs, preventing recursively nested experiments and unbounded YAML.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(default)]
pub struct RoutingPolicySnapshot {
    #[serde(default)]
    pub enabled: bool,
    #[serde(default)]
    pub classifier: ClassifierMode,
    #[serde(default)]
    pub ml_model_path: Option<String>,
    #[serde(default)]
    pub classifier_model: Option<String>,
    #[serde(default = "default_cost_quality_threshold")]
    pub cost_quality_threshold: f64,
    #[serde(default)]
    pub cascade: CascadeConfig,
    #[serde(default)]
    pub tier_boundaries: TierBoundaries,
    #[serde(default)]
    pub heuristic_weights: Option<HeuristicWeights>,
    #[serde(default)]
    pub composite_weights: Option<CompositeWeights>,
    #[serde(default)]
    pub streaming_cascade_mode: StreamingCascadeMode,
    #[serde(default)]
    pub online_optimizer: OnlineOptimizerConfig,
    #[serde(default)]
    pub semantic_cache: SmartRoutingSemanticCacheConfig,
    #[serde(default)]
    pub quality_evaluator: QualityEvaluatorConfig,
}

impl Default for RoutingPolicySnapshot {
    fn default() -> Self {
        Self {
            enabled: false,
            classifier: ClassifierMode::Heuristic,
            ml_model_path: None,
            classifier_model: None,
            cost_quality_threshold: default_cost_quality_threshold(),
            cascade: CascadeConfig::default(),
            tier_boundaries: TierBoundaries::default(),
            heuristic_weights: None,
            composite_weights: None,
            streaming_cascade_mode: StreamingCascadeMode::Buffer,
            online_optimizer: OnlineOptimizerConfig::default(),
            semantic_cache: SmartRoutingSemanticCacheConfig::default(),
            quality_evaluator: QualityEvaluatorConfig::default(),
        }
    }
}

impl From<&SmartRoutingConfig> for RoutingPolicySnapshot {
    fn from(config: &SmartRoutingConfig) -> Self {
        Self {
            enabled: config.enabled,
            classifier: config.classifier,
            ml_model_path: config.ml_model_path.clone(),
            classifier_model: config.classifier_model.clone(),
            cost_quality_threshold: config.cost_quality_threshold,
            cascade: config.cascade.clone(),
            tier_boundaries: config.tier_boundaries.clone(),
            heuristic_weights: config.heuristic_weights.clone(),
            composite_weights: config.composite_weights.clone(),
            streaming_cascade_mode: config.streaming_cascade_mode,
            online_optimizer: config.online_optimizer.clone(),
            semantic_cache: config.semantic_cache.clone(),
            quality_evaluator: config.quality_evaluator.clone(),
        }
    }
}

impl RoutingPolicySnapshot {
    pub fn validate(&self) -> SmartRoutingValidationResult<()> {
        let mut errors = Vec::new();
        self.validate_into("", &mut errors);
        validation_result(errors)
    }

    fn validate_into(&self, scope: &str, errors: &mut Vec<SmartRoutingConfigError>) {
        validate_finite_closed_unit(
            &field(scope, "cost_quality_threshold"),
            self.cost_quality_threshold,
            errors,
        );
        self.tier_boundaries
            .validate_into(&field(scope, "tier_boundaries"), errors);
        self.cascade.validate_into(&field(scope, "cascade"), errors);

        if matches!(
            self.classifier,
            ClassifierMode::Ml | ClassifierMode::Composite
        ) && self
            .ml_model_path
            .as_deref()
            .is_none_or(|path| path.trim().is_empty())
        {
            errors.push(SmartRoutingConfigError::new(
                field(scope, "ml_model_path"),
                "is required when classifier is ml or composite",
            ));
        }
        if matches!(self.classifier, ClassifierMode::Llm)
            && self
                .classifier_model
                .as_deref()
                .is_none_or(|model| model.trim().is_empty())
        {
            errors.push(SmartRoutingConfigError::new(
                field(scope, "classifier_model"),
                "is required when classifier is llm",
            ));
        }
        if let Some(path) = &self.ml_model_path {
            validate_optional_text(
                &field(scope, "ml_model_path"),
                path,
                MAX_MODEL_PATH_CHARS,
                errors,
            );
        }
        if let Some(model) = &self.classifier_model {
            validate_optional_text(
                &field(scope, "classifier_model"),
                model,
                MAX_MODEL_NAME_CHARS,
                errors,
            );
        }
        if let Some(weights) = &self.heuristic_weights {
            weights.validate_into(&field(scope, "heuristic_weights"), errors);
        }
        if let Some(weights) = &self.composite_weights {
            weights.validate_into(&field(scope, "composite_weights"), errors);
        }
        self.online_optimizer
            .validate_into(&field(scope, "online_optimizer"), errors);
        self.semantic_cache
            .validate_into(&field(scope, "semantic_cache"), errors);
        self.quality_evaluator
            .validate_into(&field(scope, "quality_evaluator"), errors);
    }
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(default)]
pub struct ABTestConfig {
    #[serde(default)]
    pub control: RoutingPolicySnapshot,
    #[serde(default)]
    pub variant: RoutingPolicySnapshot,
    #[serde(default = "default_variant_percentage")]
    pub variant_percentage: f64,
}

impl Default for ABTestConfig {
    fn default() -> Self {
        Self {
            control: RoutingPolicySnapshot::default(),
            variant: RoutingPolicySnapshot::default(),
            variant_percentage: default_variant_percentage(),
        }
    }
}

impl ABTestConfig {
    fn validate_into(&self, scope: &str, errors: &mut Vec<SmartRoutingConfigError>) {
        self.control.validate_into(&field(scope, "control"), errors);
        self.variant.validate_into(&field(scope, "variant"), errors);
        validate_finite_range(
            &field(scope, "variant_percentage"),
            self.variant_percentage,
            f64::MIN_POSITIVE,
            1.0,
            "a finite value greater than 0.0 and at most 1.0",
            errors,
        );
    }
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(default)]
pub struct TrainingConfig {
    #[serde(default)]
    pub dataset_path: Option<String>,
    #[serde(default = "default_learning_rate")]
    pub learning_rate: f64,
    #[serde(default = "default_batch_size")]
    pub batch_size: usize,
    #[serde(default = "default_epochs")]
    pub epochs: usize,
    #[serde(default)]
    pub augmentation: bool,
}

impl Default for TrainingConfig {
    fn default() -> Self {
        Self {
            dataset_path: None,
            learning_rate: default_learning_rate(),
            batch_size: default_batch_size(),
            epochs: default_epochs(),
            augmentation: false,
        }
    }
}

impl TrainingConfig {
    fn validate_into(&self, scope: &str, errors: &mut Vec<SmartRoutingConfigError>) {
        if let Some(path) = &self.dataset_path {
            validate_optional_text(
                &field(scope, "dataset_path"),
                path,
                MAX_MODEL_PATH_CHARS,
                errors,
            );
        }
        validate_finite_range(
            &field(scope, "learning_rate"),
            self.learning_rate,
            f64::MIN_POSITIVE,
            1.0,
            "a finite value greater than 0.0 and at most 1.0",
            errors,
        );
        if !(1..=MAX_TRAINING_BATCH_SIZE).contains(&self.batch_size) {
            errors.push(SmartRoutingConfigError::new(
                field(scope, "batch_size"),
                format!(
                    "is {}; expected an integer in 1..={MAX_TRAINING_BATCH_SIZE}",
                    self.batch_size
                ),
            ));
        }
        if !(1..=MAX_TRAINING_EPOCHS).contains(&self.epochs) {
            errors.push(SmartRoutingConfigError::new(
                field(scope, "epochs"),
                format!(
                    "is {}; expected an integer in 1..={MAX_TRAINING_EPOCHS}",
                    self.epochs
                ),
            ));
        }
    }
}

#[derive(Debug, Clone, Default, PartialEq, Serialize, Deserialize)]
#[serde(default)]
pub struct BudgetLimits {
    #[serde(default)]
    pub hourly_limit_usd: Option<f64>,
    #[serde(default)]
    pub daily_limit_usd: Option<f64>,
    #[serde(default)]
    pub monthly_limit_usd: Option<f64>,
}

impl BudgetLimits {
    fn validate_into(&self, scope: &str, errors: &mut Vec<SmartRoutingConfigError>) {
        for (name, value) in [
            ("hourly_limit_usd", self.hourly_limit_usd),
            ("daily_limit_usd", self.daily_limit_usd),
            ("monthly_limit_usd", self.monthly_limit_usd),
        ] {
            if let Some(value) = value {
                if !value.is_finite() || value <= 0.0 {
                    errors.push(SmartRoutingConfigError::new(
                        field(scope, name),
                        format!("is {value}; expected a positive finite dollar amount"),
                    ));
                }
            }
        }
    }
}

/// One structured, field-addressable smart-routing configuration failure.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct SmartRoutingConfigError {
    pub field: String,
    pub message: String,
}

impl SmartRoutingConfigError {
    pub fn new(field: impl Into<String>, message: impl Into<String>) -> Self {
        Self {
            field: field.into(),
            message: message.into(),
        }
    }
}

impl fmt::Display for SmartRoutingConfigError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            formatter,
            "invalid smart_routing.{}: {}",
            self.field, self.message
        )
    }
}

impl std::error::Error for SmartRoutingConfigError {}

pub type SmartRoutingValidationResult<T> = Result<T, Vec<SmartRoutingConfigError>>;

fn validate_weights(
    scope: &str,
    weights: &[(&str, f64)],
    errors: &mut Vec<SmartRoutingConfigError>,
) {
    let mut total = 0.0;
    let mut all_valid = true;

    for (name, value) in weights {
        if !value.is_finite() || *value < 0.0 {
            all_valid = false;
            errors.push(SmartRoutingConfigError::new(
                field(scope, name),
                format!("is {value}; expected a finite non-negative weight"),
            ));
        } else {
            total += value;
        }
    }

    if all_valid && (total <= 0.0 || (total - 1.0).abs() > WEIGHT_SUM_TOLERANCE) {
        errors.push(SmartRoutingConfigError::new(
            scope,
            format!(
                "weights sum to {total}; expected a positive total equal to 1.0 within tolerance {WEIGHT_SUM_TOLERANCE}"
            ),
        ));
    }
}

fn validate_finite_open_unit(
    field_name: &str,
    value: f64,
    errors: &mut Vec<SmartRoutingConfigError>,
) {
    if !value.is_finite() || value <= 0.0 || value >= 1.0 {
        errors.push(SmartRoutingConfigError::new(
            field_name,
            format!("is {value}; expected a finite value strictly between 0.0 and 1.0"),
        ));
    }
}

fn validate_finite_closed_unit(
    field_name: &str,
    value: f64,
    errors: &mut Vec<SmartRoutingConfigError>,
) {
    validate_finite_range(
        field_name,
        value,
        0.0,
        1.0,
        "a finite value in 0.0..=1.0",
        errors,
    );
}

fn validate_finite_range(
    field_name: &str,
    value: f64,
    min: f64,
    max: f64,
    expected: &str,
    errors: &mut Vec<SmartRoutingConfigError>,
) {
    if !value.is_finite() || value < min || value > max {
        errors.push(SmartRoutingConfigError::new(
            field_name,
            format!("is {value}; expected {expected}"),
        ));
    }
}

fn validate_u32_range(
    field_name: &str,
    value: u32,
    min: u32,
    max: u32,
    errors: &mut Vec<SmartRoutingConfigError>,
) {
    if !(min..=max).contains(&value) {
        errors.push(SmartRoutingConfigError::new(
            field_name,
            format!("is {value}; expected an integer in {min}..={max}"),
        ));
    }
}

fn validate_optional_text(
    field_name: &str,
    value: &str,
    max_chars: usize,
    errors: &mut Vec<SmartRoutingConfigError>,
) {
    validate_required_text(field_name, value, max_chars, errors);
}

fn validate_required_text(
    field_name: &str,
    value: &str,
    max_chars: usize,
    errors: &mut Vec<SmartRoutingConfigError>,
) {
    if value.trim().is_empty() {
        errors.push(SmartRoutingConfigError::new(
            field_name,
            "must not be empty or whitespace-only",
        ));
    }
    if value.contains('\0') {
        errors.push(SmartRoutingConfigError::new(
            field_name,
            "must not contain NUL characters",
        ));
    }
    let length = value.chars().count();
    if length > max_chars {
        errors.push(SmartRoutingConfigError::new(
            field_name,
            format!("has {length} characters; expected at most {max_chars}"),
        ));
    }
}

fn validate_map_key(scope: &str, key: &str, errors: &mut Vec<SmartRoutingConfigError>) {
    validate_required_text(&format!("{scope}.{key}"), key, MAX_GROUP_NAME_CHARS, errors);
}

fn field(scope: &str, name: &str) -> String {
    if scope.is_empty() {
        name.to_string()
    } else {
        format!("{scope}.{name}")
    }
}

fn validation_result(errors: Vec<SmartRoutingConfigError>) -> SmartRoutingValidationResult<()> {
    if errors.is_empty() {
        Ok(())
    } else {
        Err(errors)
    }
}

fn default_cost_quality_threshold() -> f64 {
    0.5
}

fn default_fast_max() -> f64 {
    0.33
}

fn default_balanced_max() -> f64 {
    0.66
}

fn default_max_escalations() -> u8 {
    2
}

fn default_min_response_tokens() -> u32 {
    20
}

fn default_early_signal_tokens() -> u32 {
    50
}

fn default_reserved_output_tokens() -> u32 {
    1_024
}

fn default_provider_overhead_tokens() -> u32 {
    64
}

fn default_context_safety_margin_tokens() -> u32 {
    256
}

fn default_equal_heuristic_weight() -> f64 {
    1.0 / 6.0
}

fn default_composite_heuristic_weight() -> f64 {
    0.3
}

fn default_composite_ml_weight() -> f64 {
    0.7
}

fn default_optimizer_alpha() -> f64 {
    0.01
}

fn default_optimizer_interval_secs() -> u64 {
    600
}

fn default_optimizer_quality_threshold() -> f64 {
    0.5
}

fn default_similarity_threshold() -> f64 {
    0.95
}

fn default_semantic_cache_max_entries() -> usize {
    500
}

fn default_semantic_cache_ttl_secs() -> u64 {
    1800
}

fn default_embedding_model() -> String {
    "builtin-minilm".to_string()
}

fn default_min_quality_score() -> f64 {
    0.5
}

fn default_quality_evaluator_threshold() -> f64 {
    0.3
}

fn default_variant_percentage() -> f64 {
    0.1
}

fn default_learning_rate() -> f64 {
    5.0e-5
}

fn default_batch_size() -> usize {
    8
}

fn default_epochs() -> usize {
    3
}

#[cfg(test)]
mod tests {
    use super::*;
    use proptest::prelude::*;
    use std::cell::Cell;
    use std::sync::Arc;

    fn unit_fraction(value: u16) -> f64 {
        value as f64 / 1000.0
    }

    fn arb_classifier_settings(
    ) -> impl Strategy<Value = (ClassifierMode, Option<String>, Option<String>)> {
        prop_oneof![
            Just((ClassifierMode::Heuristic, None, None)),
            Just((
                ClassifierMode::Ml,
                Some("router-model.bin".to_string()),
                None,
            )),
            Just((
                ClassifierMode::Llm,
                None,
                Some("classifier-model".to_string()),
            )),
            Just((
                ClassifierMode::Composite,
                Some("router-model.bin".to_string()),
                Some("classifier-model".to_string()),
            )),
        ]
    }

    fn arb_tier_boundaries() -> impl Strategy<Value = TierBoundaries> {
        (1u16..999).prop_flat_map(|fast| {
            (fast + 1..1000).prop_map(move |balanced| TierBoundaries {
                fast_max: unit_fraction(fast),
                balanced_max: unit_fraction(balanced),
            })
        })
    }

    fn arb_cascade_config() -> impl Strategy<Value = CascadeConfig> {
        (any::<bool>(), 1u8..=2, 1u32..=65_536, 1u32..=65_536).prop_map(
            |(enabled, max_escalations, min_response_tokens, early_signal_tokens)| CascadeConfig {
                enabled,
                max_escalations,
                min_response_tokens,
                early_signal_tokens,
            },
        )
    }

    fn arb_online_optimizer_config() -> impl Strategy<Value = OnlineOptimizerConfig> {
        (
            any::<bool>(),
            1u16..=1000,
            1u64..=MAX_OPTIMIZER_INTERVAL_SECS,
            prop::option::of("[a-zA-Z0-9_./-]{1,40}"),
            0u16..=1000,
        )
            .prop_map(
                |(enabled, alpha, interval_secs, state_path, quality_threshold)| {
                    OnlineOptimizerConfig {
                        enabled,
                        alpha: unit_fraction(alpha),
                        interval_secs,
                        state_path,
                        quality_threshold: unit_fraction(quality_threshold),
                    }
                },
            )
    }

    fn arb_semantic_cache_config() -> impl Strategy<Value = SmartRoutingSemanticCacheConfig> {
        (
            any::<bool>(),
            0u16..=1000,
            1usize..=MAX_CACHE_ENTRIES,
            1u64..=MAX_TTL_SECS,
            "[a-zA-Z0-9_./-]{1,40}",
            0u16..=1000,
        )
            .prop_map(
                |(
                    enabled,
                    similarity_threshold,
                    max_entries,
                    ttl_secs,
                    embedding_model,
                    min_quality_score,
                )| SmartRoutingSemanticCacheConfig {
                    enabled,
                    similarity_threshold: unit_fraction(similarity_threshold),
                    max_entries,
                    ttl_secs,
                    embedding_model,
                    min_quality_score: unit_fraction(min_quality_score),
                },
            )
    }

    fn arb_training_config() -> impl Strategy<Value = TrainingConfig> {
        (
            prop::option::of("[a-zA-Z0-9_./-]{1,40}"),
            1u16..=1000,
            1usize..=MAX_TRAINING_BATCH_SIZE,
            1usize..=MAX_TRAINING_EPOCHS,
            any::<bool>(),
        )
            .prop_map(
                |(dataset_path, learning_rate, batch_size, epochs, augmentation)| TrainingConfig {
                    dataset_path,
                    learning_rate: unit_fraction(learning_rate),
                    batch_size,
                    epochs,
                    augmentation,
                },
            )
    }

    fn arb_valid_smart_routing_config() -> impl Strategy<Value = SmartRoutingConfig> {
        (
            any::<bool>(),
            arb_classifier_settings(),
            0u16..=1000,
            arb_tier_boundaries(),
            arb_cascade_config(),
            (any::<bool>(), any::<bool>()),
            any::<bool>(),
            arb_online_optimizer_config(),
            arb_semantic_cache_config(),
            (any::<bool>(), 0u16..=1000),
            arb_training_config(),
        )
            .prop_map(
                |(
                    enabled,
                    (classifier, ml_model_path, classifier_model),
                    cost_quality_threshold,
                    tier_boundaries,
                    cascade,
                    (include_heuristic_weights, include_composite_weights),
                    early_signal,
                    online_optimizer,
                    semantic_cache,
                    (quality_evaluator_enabled, quality_threshold),
                    training,
                )| SmartRoutingConfig {
                    enabled,
                    classifier,
                    ml_model_path,
                    classifier_model,
                    cost_quality_threshold: unit_fraction(cost_quality_threshold),
                    cascade,
                    tier_boundaries,
                    heuristic_weights: include_heuristic_weights.then(HeuristicWeights::default),
                    composite_weights: include_composite_weights.then(CompositeWeights::default),
                    streaming_cascade_mode: if early_signal {
                        StreamingCascadeMode::EarlySignal
                    } else {
                        StreamingCascadeMode::Buffer
                    },
                    online_optimizer,
                    semantic_cache,
                    quality_evaluator: QualityEvaluatorConfig {
                        enabled: quality_evaluator_enabled,
                        threshold: unit_fraction(quality_threshold),
                    },
                    training,
                    ..Default::default()
                },
            )
    }

    fn arb_smart_routing_override() -> impl Strategy<Value = SmartRoutingOverride> {
        (
            prop::option::of(any::<bool>()),
            prop::option::of(arb_classifier_settings()),
            prop::option::of(0u16..=1000),
            prop::option::of(arb_cascade_config()),
            prop::option::of(arb_tier_boundaries()),
            prop::option::of(any::<bool>()),
            prop::option::of(any::<bool>()),
            prop::option::of(any::<bool>()),
            prop::option::of(arb_online_optimizer_config()),
            prop::option::of(arb_semantic_cache_config()),
            prop::option::of((any::<bool>(), 0u16..=1000)),
            prop::option::of(arb_training_config()),
        )
            .prop_map(
                |(
                    enabled,
                    classifier_settings,
                    cost_quality_threshold,
                    cascade,
                    tier_boundaries,
                    include_heuristic_weights,
                    include_composite_weights,
                    early_signal,
                    online_optimizer,
                    semantic_cache,
                    quality_evaluator,
                    training,
                )| {
                    let (classifier, ml_model_path, classifier_model) = classifier_settings
                        .map(|(classifier, ml_model_path, classifier_model)| {
                            (
                                Some(classifier),
                                Some(ml_model_path),
                                Some(classifier_model),
                            )
                        })
                        .unwrap_or((None, None, None));
                    SmartRoutingOverride {
                        enabled,
                        classifier,
                        ml_model_path,
                        classifier_model,
                        cost_quality_threshold: cost_quality_threshold.map(unit_fraction),
                        cascade,
                        tier_boundaries,
                        heuristic_weights: include_heuristic_weights
                            .map(|include| include.then_some(HeuristicWeights::default()))
                            .flatten(),
                        composite_weights: include_composite_weights
                            .map(|include| include.then_some(CompositeWeights::default()))
                            .flatten(),
                        streaming_cascade_mode: early_signal.map(|early_signal| {
                            if early_signal {
                                StreamingCascadeMode::EarlySignal
                            } else {
                                StreamingCascadeMode::Buffer
                            }
                        }),
                        online_optimizer,
                        semantic_cache,
                        quality_evaluator: quality_evaluator.map(|(enabled, threshold)| {
                            QualityEvaluatorConfig {
                                enabled,
                                threshold: unit_fraction(threshold),
                            }
                        }),
                        training,
                        ..Default::default()
                    }
                },
            )
    }

    fn publish_atomic_snapshot<T, Error>(
        current: &mut Arc<T>,
        build: impl FnOnce() -> Result<T, Error>,
    ) -> Result<(), Error> {
        let replacement = Arc::new(build()?);
        *current = replacement;
        Ok(())
    }

    fn assert_has_field(errors: &[SmartRoutingConfigError], expected: &str) {
        assert!(
            errors.iter().any(|error| error.field == expected),
            "missing {expected}: {errors:?}"
        );
    }

    #[test]
    fn defaults_are_disabled_and_empty_yaml_is_backward_compatible() {
        let config: SmartRoutingConfig = serde_yaml::from_str("{}").unwrap();
        assert_eq!(config, SmartRoutingConfig::default());
        assert!(!config.enabled);
        assert_eq!(config.classifier, ClassifierMode::Heuristic);
        assert_eq!(config.streaming_cascade_mode, StreamingCascadeMode::Buffer);
        assert!(!config.online_optimizer.enabled);
        assert!(!config.semantic_cache.enabled);
        assert!(config.model_group_overrides.is_empty());
        assert!(config.budget_limits.is_empty());
        assert!(config.ab_test.is_none());
        assert!(config.validate().is_ok());
    }

    #[test]
    fn empty_weight_objects_use_equal_and_composite_defaults() {
        let config: SmartRoutingConfig =
            serde_yaml::from_str("heuristic_weights: {}\ncomposite_weights: {}\n").unwrap();
        assert_eq!(config.heuristic_weights, Some(HeuristicWeights::default()));
        assert_eq!(config.composite_weights, Some(CompositeWeights::default()));
        assert!(config.validate().is_ok());
    }

    #[test]
    fn representative_valid_config_is_accepted() {
        let yaml = r#"
enabled: true
classifier: composite
ml_model_path: ./models/router.safetensors
classifier_model: cheap-classifier
cost_quality_threshold: 0.6
tier_boundaries:
  fast_max: 0.25
  balanced_max: 0.75
cascade:
  enabled: true
  max_escalations: 2
  min_response_tokens: 10
  early_signal_tokens: 50
heuristic_weights:
  message_count: 0.1
  token_estimate: 0.2
  code_blocks: 0.2
  tool_calls: 0.2
  math_expressions: 0.2
  reasoning_keywords: 0.1
composite_weights:
  heuristic: 0.3
  ml: 0.7
streaming_cascade_mode: early_signal
online_optimizer:
  enabled: true
  alpha: 0.01
  interval_secs: 600
  quality_threshold: 0.5
semantic_cache:
  enabled: true
  similarity_threshold: 0.95
  max_entries: 500
  ttl_secs: 1800
  embedding_model: builtin-minilm
  min_quality_score: 0.5
quality_evaluator:
  enabled: true
  threshold: 0.3
budget_limits:
  default:
    hourly_limit_usd: 5.0
    daily_limit_usd: 50.0
    monthly_limit_usd: 500.0
training:
  learning_rate: 0.00005
  batch_size: 8
  epochs: 3
"#;
        let config: SmartRoutingConfig = serde_yaml::from_str(yaml).unwrap();
        assert!(config.validate().is_ok());
    }

    #[test]
    fn non_finite_and_out_of_range_values_are_rejected() {
        let mut config = SmartRoutingConfig::default();
        config.cost_quality_threshold = f64::NAN;
        config.tier_boundaries.fast_max = f64::INFINITY;
        config.quality_evaluator.threshold = f64::INFINITY;
        config.semantic_cache.similarity_threshold = -0.1;
        config.semantic_cache.min_quality_score = f64::NEG_INFINITY;
        config.cascade.max_escalations = 0;
        config.cascade.min_response_tokens = 0;
        config.training.learning_rate = f64::NEG_INFINITY;

        let errors = config.validate().unwrap_err();
        for expected in [
            "cost_quality_threshold",
            "tier_boundaries.fast_max",
            "quality_evaluator.threshold",
            "semantic_cache.similarity_threshold",
            "semantic_cache.min_quality_score",
            "cascade.max_escalations",
            "cascade.min_response_tokens",
            "training.learning_rate",
        ] {
            assert_has_field(&errors, expected);
        }
    }

    #[test]
    fn invalid_boundaries_are_rejected() {
        for boundaries in [
            TierBoundaries {
                fast_max: 0.7,
                balanced_max: 0.6,
            },
            TierBoundaries {
                fast_max: 0.0,
                balanced_max: 0.6,
            },
            TierBoundaries {
                fast_max: 0.3,
                balanced_max: 1.0,
            },
            TierBoundaries {
                fast_max: f64::NAN,
                balanced_max: 0.6,
            },
        ] {
            let config = SmartRoutingConfig {
                tier_boundaries: boundaries,
                ..Default::default()
            };
            assert!(config.validate().is_err());
        }
    }

    proptest! {
            #![proptest_config(ProptestConfig::with_cases(256))]

            #[test]
            fn invalid_tier_boundaries_are_rejected(fast_max in any::<f64>(), balanced_max in any::<f64>()) {
            let boundaries_are_valid = fast_max.is_finite()
            && balanced_max.is_finite()
            && fast_max > 0.0
            && balanced_max < 1.0
            && fast_max < balanced_max;
            prop_assume!(!boundaries_are_valid);

            let config = SmartRoutingConfig {
            tier_boundaries: TierBoundaries {
            fast_max,
            balanced_max,
            },
            ..Default::default()
            };

            let errors = config.validate().expect_err("invalid tier boundaries must be rejected");
            prop_assert!(errors.iter().any(|error| error.field.starts_with("tier_boundaries.")));
            }

        #[test]
        fn valid_smart_routing_config_serde_round_trip(config in arb_valid_smart_routing_config()) {
        prop_assert!(config.validate().is_ok());

        let json = serde_json::to_string(&config).expect("valid config must serialize as JSON");
        let json_round_trip: SmartRoutingConfig =
        serde_json::from_str(&json).expect("serialized JSON must deserialize");
        prop_assert_eq!(&json_round_trip, &config);
        prop_assert!(json_round_trip.validate().is_ok());

        let yaml = serde_yaml::to_string(&config).expect("valid config must serialize as YAML");
        let yaml_round_trip: SmartRoutingConfig =
        serde_yaml::from_str(&yaml).expect("serialized YAML must deserialize");
        prop_assert_eq!(&yaml_round_trip, &config);
        prop_assert!(yaml_round_trip.validate().is_ok());
        }

        #[test]
        fn property_18_override_present_fields_win_and_global_absent_fields_fill(
        global in arb_valid_smart_routing_config(),
        config_override in arb_smart_routing_override(),
        ) {
        let mut config = global.clone();
        config.model_group_overrides.insert("property-group".to_string(), config_override.clone());

        let effective = config.effective_for_group("property-group");

        prop_assert_eq!(effective.enabled, config_override.enabled.unwrap_or(global.enabled));
        prop_assert_eq!(effective.classifier, config_override.classifier.unwrap_or(global.classifier));
        prop_assert_eq!(
        &effective.ml_model_path,
        config_override.ml_model_path.as_ref().unwrap_or(&global.ml_model_path)
        );
        prop_assert_eq!(
        &effective.classifier_model,
        config_override.classifier_model.as_ref().unwrap_or(&global.classifier_model)
        );
        prop_assert_eq!(
        effective.cost_quality_threshold,
        config_override.cost_quality_threshold.unwrap_or(global.cost_quality_threshold)
        );
        prop_assert_eq!(
        &effective.cascade,
        config_override.cascade.as_ref().unwrap_or(&global.cascade)
        );
        prop_assert_eq!(
        &effective.tier_boundaries,
        config_override.tier_boundaries.as_ref().unwrap_or(&global.tier_boundaries)
        );
    prop_assert_eq!(
    effective.heuristic_weights.as_ref(),
    config_override
    .heuristic_weights
    .as_ref()
    .or(global.heuristic_weights.as_ref())
    );
    prop_assert_eq!(
    effective.composite_weights.as_ref(),
    config_override
    .composite_weights
    .as_ref()
    .or(global.composite_weights.as_ref())
    );
        prop_assert_eq!(
        effective.streaming_cascade_mode,
        config_override
        .streaming_cascade_mode
        .unwrap_or(global.streaming_cascade_mode)
        );
        prop_assert_eq!(
        effective.allow_unknown_context_window,
        config_override
        .allow_unknown_context_window
        .unwrap_or(global.allow_unknown_context_window)
        );
        prop_assert_eq!(
        effective.reserved_output_tokens,
        config_override
        .reserved_output_tokens
        .unwrap_or(global.reserved_output_tokens)
        );
        prop_assert_eq!(
        effective.provider_overhead_tokens,
        config_override
        .provider_overhead_tokens
        .unwrap_or(global.provider_overhead_tokens)
        );
        prop_assert_eq!(
        effective.context_safety_margin_tokens,
        config_override
        .context_safety_margin_tokens
        .unwrap_or(global.context_safety_margin_tokens)
        );
        prop_assert_eq!(
        &effective.online_optimizer,
        config_override.online_optimizer.as_ref().unwrap_or(&global.online_optimizer)
        );
        prop_assert_eq!(
        &effective.semantic_cache,
        config_override.semantic_cache.as_ref().unwrap_or(&global.semantic_cache)
        );
        prop_assert_eq!(
        &effective.quality_evaluator,
        config_override.quality_evaluator.as_ref().unwrap_or(&global.quality_evaluator)
        );
        prop_assert_eq!(
        &effective.training,
        config_override.training.as_ref().unwrap_or(&global.training)
        );
        prop_assert!(effective.model_group_overrides.is_empty());
        }
        }

    #[test]
    fn invalid_heuristic_and_composite_weights_are_rejected() {
        let mut config = SmartRoutingConfig {
            heuristic_weights: Some(HeuristicWeights::default()),
            composite_weights: Some(CompositeWeights::default()),
            ..Default::default()
        };
        config.heuristic_weights.as_mut().unwrap().message_count = -0.1;
        config.composite_weights.as_mut().unwrap().ml = f64::INFINITY;
        let errors = config.validate().unwrap_err();
        assert_has_field(&errors, "heuristic_weights.message_count");
        assert_has_field(&errors, "composite_weights.ml");

        config.heuristic_weights = Some(HeuristicWeights {
            message_count: 0.0,
            token_estimate: 0.0,
            code_blocks: 0.0,
            tool_calls: 0.0,
            math_expressions: 0.0,
            reasoning_keywords: 0.0,
        });
        config.composite_weights = Some(CompositeWeights {
            heuristic: 0.4,
            ml: 0.4,
        });
        let errors = config.validate().unwrap_err();
        assert_has_field(&errors, "heuristic_weights");
        assert_has_field(&errors, "composite_weights");
    }

    #[test]
    fn ml_and_composite_modes_require_a_model_path() {
        for classifier in [ClassifierMode::Ml, ClassifierMode::Composite] {
            let config = SmartRoutingConfig {
                classifier,
                ml_model_path: None,
                ..Default::default()
            };
            assert_has_field(&config.validate().unwrap_err(), "ml_model_path");
        }
    }

    #[test]
    fn llm_mode_requires_a_classifier_model() {
        let config = SmartRoutingConfig {
            classifier: ClassifierMode::Llm,
            classifier_model: None,
            ..Default::default()
        };
        assert_has_field(&config.validate().unwrap_err(), "classifier_model");
    }

    #[test]
    fn invalid_budget_and_optimizer_values_are_rejected() {
        let mut config = SmartRoutingConfig::default();
        config.budget_limits.insert(
            "default".to_string(),
            BudgetLimits {
                hourly_limit_usd: Some(0.0),
                daily_limit_usd: Some(f64::NAN),
                monthly_limit_usd: Some(f64::INFINITY),
            },
        );
        config.online_optimizer.alpha = 0.0;
        config.online_optimizer.interval_secs = 0;
        let errors = config.validate().unwrap_err();
        for expected in [
            "budget_limits.default.hourly_limit_usd",
            "budget_limits.default.daily_limit_usd",
            "budget_limits.default.monthly_limit_usd",
            "online_optimizer.alpha",
            "online_optimizer.interval_secs",
        ] {
            assert_has_field(&errors, expected);
        }
    }

    #[test]
    fn override_merge_uses_present_values_and_global_fallbacks() {
        let mut config = SmartRoutingConfig {
            enabled: true,
            classifier_model: Some("cheap-classifier".to_string()),
            cost_quality_threshold: 0.4,
            tier_boundaries: TierBoundaries {
                fast_max: 0.2,
                balanced_max: 0.8,
            },
            ..Default::default()
        };
        config.model_group_overrides.insert(
            "coding".to_string(),
            SmartRoutingOverride {
                enabled: Some(false),
                classifier: Some(ClassifierMode::Llm),
                cost_quality_threshold: Some(0.9),
                cascade: Some(CascadeConfig {
                    enabled: true,
                    ..Default::default()
                }),
                ..Default::default()
            },
        );

        let effective = config.effective_for_group("coding");
        assert!(!effective.enabled);
        assert_eq!(effective.classifier, ClassifierMode::Llm);
        assert_eq!(effective.cost_quality_threshold, 0.9);
        assert!(effective.cascade.enabled);
        assert_eq!(effective.tier_boundaries, config.tier_boundaries);
        assert!(effective.model_group_overrides.is_empty());

        let untouched = config.effective_for_group("other");
        assert!(untouched.enabled);
        assert_eq!(untouched.cost_quality_threshold, 0.4);
        assert!(untouched.model_group_overrides.is_empty());
    }

    #[test]
    fn atomic_snapshot_rolls_back_invalid_candidate_before_optional_builder_runs() {
        let old = Arc::new(SmartRoutingConfig {
            enabled: true,
            cost_quality_threshold: 0.25,
            ..Default::default()
        });
        let mut current = Arc::clone(&old);
        let optional_builder_ran = Cell::new(false);
        let invalid_candidate = SmartRoutingConfig {
            cost_quality_threshold: f64::NAN,
            ..Default::default()
        };

        let result = publish_atomic_snapshot(&mut current, || {
            invalid_candidate.validate().map_err(|_| "validation")?;
            optional_builder_ran.set(true);
            Ok::<_, &str>(invalid_candidate)
        });

        assert_eq!(result, Err("validation"));
        assert!(!optional_builder_ran.get());
        assert!(Arc::ptr_eq(&current, &old));
    }

    #[test]
    fn atomic_snapshot_rolls_back_when_optional_component_builder_fails() {
        let old = Arc::new(SmartRoutingConfig {
            enabled: true,
            cost_quality_threshold: 0.25,
            ..Default::default()
        });
        let mut current = Arc::clone(&old);
        let candidate = SmartRoutingConfig {
            enabled: false,
            cost_quality_threshold: 0.75,
            ..Default::default()
        };

        let result = publish_atomic_snapshot(&mut current, || {
            candidate.validate().map_err(|_| "validation")?;
            Err::<SmartRoutingConfig, _>("optional-component")
        });

        assert_eq!(result, Err("optional-component"));
        assert!(Arc::ptr_eq(&current, &old));
        assert!(current.enabled);
        assert_eq!(current.cost_quality_threshold, 0.25);
    }

    #[test]
    fn override_effective_policy_is_validated() {
        let mut config = SmartRoutingConfig::default();
        config.model_group_overrides.insert(
            "ml-group".to_string(),
            SmartRoutingOverride {
                classifier: Some(ClassifierMode::Ml),
                ..Default::default()
            },
        );
        assert_has_field(
            &config.validate().unwrap_err(),
            "model_group_overrides.ml-group.ml_model_path",
        );
    }

    #[test]
    fn ab_test_policy_is_non_recursive_and_validated() {
        let config: SmartRoutingConfig = serde_yaml::from_str(
            r#"
ab_test:
  control:
    enabled: true
  variant:
    enabled: true
    cost_quality_threshold: 0.8
  variant_percentage: 0.1
"#,
        )
        .unwrap();
        assert!(config.validate().is_ok());
        let serialized = serde_yaml::to_string(&config).unwrap();
        assert_eq!(serialized.matches("ab_test:").count(), 1);

        let mut invalid = config;
        invalid.ab_test.as_mut().unwrap().variant_percentage = f64::NAN;
        assert_has_field(
            &invalid.validate().unwrap_err(),
            "ab_test.variant_percentage",
        );
    }
}
