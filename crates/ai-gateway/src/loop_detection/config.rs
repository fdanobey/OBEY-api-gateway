use serde::{Deserialize, Serialize};
use std::fmt;

const WEIGHT_SUM_TOLERANCE: f32 = 0.001;
const MAX_BREAK_INSTRUCTION_TEMPLATE_LENGTH: usize = 2_000;

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct LoopDetectionConfig {
    #[serde(default)]
    pub enabled: bool,
    #[serde(default = "default_session_timeout_minutes")]
    pub session_timeout_minutes: u32,
    #[serde(default = "default_max_sessions")]
    pub max_sessions: u32,
    #[serde(default = "default_history_depth")]
    pub history_depth: u32,
    #[serde(default)]
    pub thresholds: ThresholdConfig,
    #[serde(default)]
    pub consecutive_counts: ConsecutiveCountConfig,
    #[serde(default)]
    pub weights: SignalWeights,
    #[serde(default = "default_throttle_delay_seconds")]
    pub throttle_delay_seconds: u32,
    #[serde(default)]
    pub injection_strategy: InjectionStrategy,
    #[serde(default = "default_ema_alpha")]
    pub ema_alpha: f32,
    #[serde(default = "default_eviction_interval_seconds")]
    pub eviction_interval_seconds: u32,
    #[serde(default = "default_token_velocity_threshold")]
    pub token_velocity_threshold: f32,
    #[serde(default = "default_cost_velocity_threshold")]
    pub cost_velocity_threshold: f64,
    #[serde(default)]
    pub break_instruction_template: Option<String>,
}

impl Default for LoopDetectionConfig {
    fn default() -> Self {
        Self {
            enabled: false,
            session_timeout_minutes: default_session_timeout_minutes(),
            max_sessions: default_max_sessions(),
            history_depth: default_history_depth(),
            thresholds: ThresholdConfig::default(),
            consecutive_counts: ConsecutiveCountConfig::default(),
            weights: SignalWeights::default(),
            throttle_delay_seconds: default_throttle_delay_seconds(),
            injection_strategy: InjectionStrategy::default(),
            ema_alpha: default_ema_alpha(),
            eviction_interval_seconds: default_eviction_interval_seconds(),
            token_velocity_threshold: default_token_velocity_threshold(),
            cost_velocity_threshold: default_cost_velocity_threshold(),
            break_instruction_template: None,
        }
    }
}

impl LoopDetectionConfig {
    pub fn validate(&self) -> Result<(), Vec<LoopDetectionConfigError>> {
        let mut errors = Vec::new();

        validate_u32_range(
            &mut errors,
            "session_timeout_minutes",
            self.session_timeout_minutes,
            1,
            1_440,
        );
        validate_u32_range(
            &mut errors,
            "max_sessions",
            self.max_sessions,
            100,
            1_000_000,
        );
        validate_u32_range(&mut errors, "history_depth", self.history_depth, 2, 50);
        validate_u32_range(
            &mut errors,
            "throttle_delay_seconds",
            self.throttle_delay_seconds,
            1,
            30,
        );
        validate_u32_range(
            &mut errors,
            "eviction_interval_seconds",
            self.eviction_interval_seconds,
            10,
            3_600,
        );
        validate_f32_range(&mut errors, "ema_alpha", self.ema_alpha, 0.01, 1.0);

        if !self.token_velocity_threshold.is_finite() || self.token_velocity_threshold <= 0.0 {
            errors.push(LoopDetectionConfigError::InvalidRange {
                field: "token_velocity_threshold",
                value: self.token_velocity_threshold.to_string(),
                expected: "a positive finite value".to_string(),
            });
        }
        if !self.cost_velocity_threshold.is_finite() || self.cost_velocity_threshold <= 0.0 {
            errors.push(LoopDetectionConfigError::InvalidRange {
                field: "cost_velocity_threshold",
                value: self.cost_velocity_threshold.to_string(),
                expected: "a positive finite value".to_string(),
            });
        }

        self.thresholds.validate(&mut errors);
        self.consecutive_counts.validate(&mut errors);
        self.weights.validate(&mut errors);
        if let Some(template) = self.break_instruction_template.as_deref() {
            if let Err(error) = Self::validate_break_instruction_template(template) {
                errors.push(error);
            }
        }

        if errors.is_empty() {
            Ok(())
        } else {
            Err(errors)
        }
    }

    pub fn validate_break_instruction_template(
        template: &str,
    ) -> Result<(), LoopDetectionConfigError> {
        let length = template.chars().count();
        if length == 0 || length > MAX_BREAK_INSTRUCTION_TEMPLATE_LENGTH {
            return Err(LoopDetectionConfigError::InvalidBreakInstructionTemplate { length });
        }
        Ok(())
    }
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct ThresholdConfig {
    #[serde(default = "default_warn_confidence")]
    pub warn_confidence: f32,
    #[serde(default = "default_throttle_confidence")]
    pub throttle_confidence: f32,
    #[serde(default = "default_inject_confidence")]
    pub inject_confidence: f32,
    #[serde(default = "default_hardstop_confidence")]
    pub hardstop_confidence: f32,
}

impl Default for ThresholdConfig {
    fn default() -> Self {
        Self {
            warn_confidence: default_warn_confidence(),
            throttle_confidence: default_throttle_confidence(),
            inject_confidence: default_inject_confidence(),
            hardstop_confidence: default_hardstop_confidence(),
        }
    }
}

impl ThresholdConfig {
    fn validate(&self, errors: &mut Vec<LoopDetectionConfigError>) {
        for (field, value) in [
            ("thresholds.warn_confidence", self.warn_confidence),
            ("thresholds.throttle_confidence", self.throttle_confidence),
            ("thresholds.inject_confidence", self.inject_confidence),
            ("thresholds.hardstop_confidence", self.hardstop_confidence),
        ] {
            validate_f32_range(errors, field, value, 0.0, 1.0);
        }

        if !(self.warn_confidence < self.throttle_confidence
            && self.throttle_confidence < self.inject_confidence
            && self.inject_confidence < self.hardstop_confidence)
        {
            errors.push(LoopDetectionConfigError::ThresholdOrder {
                warn: self.warn_confidence,
                throttle: self.throttle_confidence,
                inject: self.inject_confidence,
                hardstop: self.hardstop_confidence,
            });
        }
    }
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct ConsecutiveCountConfig {
    #[serde(default = "default_warn_count")]
    pub warn: u32,
    #[serde(default = "default_throttle_count")]
    pub throttle: u32,
    #[serde(default = "default_inject_count")]
    pub inject: u32,
    #[serde(default = "default_hardstop_count")]
    pub hardstop: u32,
}

impl Default for ConsecutiveCountConfig {
    fn default() -> Self {
        Self {
            warn: default_warn_count(),
            throttle: default_throttle_count(),
            inject: default_inject_count(),
            hardstop: default_hardstop_count(),
        }
    }
}

impl ConsecutiveCountConfig {
    fn validate(&self, errors: &mut Vec<LoopDetectionConfigError>) {
        for (field, value) in [
            ("consecutive_counts.warn", self.warn),
            ("consecutive_counts.throttle", self.throttle),
            ("consecutive_counts.inject", self.inject),
            ("consecutive_counts.hardstop", self.hardstop),
        ] {
            validate_u32_range(errors, field, value, 1, 100);
        }

        if !(self.warn <= self.throttle
            && self.throttle <= self.inject
            && self.inject <= self.hardstop)
        {
            errors.push(LoopDetectionConfigError::ConsecutiveCountOrder {
                warn: self.warn,
                throttle: self.throttle,
                inject: self.inject,
                hardstop: self.hardstop,
            });
        }
    }
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct SignalWeights {
    #[serde(default = "default_content_similarity_weight")]
    pub content_similarity: f32,
    #[serde(default = "default_tool_call_repetition_weight")]
    pub tool_call_repetition: f32,
    #[serde(default = "default_response_stagnation_weight")]
    pub response_stagnation: f32,
    #[serde(default = "default_token_velocity_weight")]
    pub token_velocity: f32,
    #[serde(default = "default_error_cycling_weight")]
    pub error_cycling: f32,
    #[serde(default = "default_context_growth_weight")]
    pub context_growth: f32,
    #[serde(default = "default_cost_velocity_weight")]
    pub cost_velocity: f32,
}

impl Default for SignalWeights {
    fn default() -> Self {
        Self {
            content_similarity: default_content_similarity_weight(),
            tool_call_repetition: default_tool_call_repetition_weight(),
            response_stagnation: default_response_stagnation_weight(),
            token_velocity: default_token_velocity_weight(),
            error_cycling: default_error_cycling_weight(),
            context_growth: default_context_growth_weight(),
            cost_velocity: default_cost_velocity_weight(),
        }
    }
}

impl SignalWeights {
    pub fn sum(&self) -> f32 {
        self.content_similarity
            + self.tool_call_repetition
            + self.response_stagnation
            + self.token_velocity
            + self.error_cycling
            + self.context_growth
            + self.cost_velocity
    }

    fn validate(&self, errors: &mut Vec<LoopDetectionConfigError>) {
        for (field, value) in [
            ("weights.content_similarity", self.content_similarity),
            ("weights.tool_call_repetition", self.tool_call_repetition),
            ("weights.response_stagnation", self.response_stagnation),
            ("weights.token_velocity", self.token_velocity),
            ("weights.error_cycling", self.error_cycling),
            ("weights.context_growth", self.context_growth),
            ("weights.cost_velocity", self.cost_velocity),
        ] {
            validate_f32_range(errors, field, value, 0.0, 1.0);
        }

        let sum = self.sum();
        if !sum.is_finite() || (sum - 1.0).abs() > WEIGHT_SUM_TOLERANCE {
            errors.push(LoopDetectionConfigError::WeightSum { sum });
        }
    }
}

#[derive(Debug, Clone, Default, PartialEq, Serialize, Deserialize)]
pub struct VkLoopConfig {
    #[serde(default)]
    pub thresholds: Option<ThresholdConfig>,
    #[serde(default)]
    pub consecutive_counts: Option<ConsecutiveCountConfig>,
    #[serde(default)]
    pub weights: Option<SignalWeights>,
    #[serde(default)]
    pub throttle_delay_seconds: Option<u32>,
    #[serde(default)]
    pub injection_strategy: Option<InjectionStrategy>,
    #[serde(default)]
    pub break_instruction_template: Option<String>,
}

impl VkLoopConfig {
    pub fn merge(
        &self,
        global: &LoopDetectionConfig,
    ) -> Result<LoopDetectionConfig, Vec<LoopDetectionConfigError>> {
        let mut effective = global.clone();
        if let Some(thresholds) = &self.thresholds {
            effective.thresholds = thresholds.clone();
        }
        if let Some(counts) = &self.consecutive_counts {
            effective.consecutive_counts = counts.clone();
        }
        if let Some(weights) = &self.weights {
            effective.weights = weights.clone();
        }
        if let Some(delay) = self.throttle_delay_seconds {
            effective.throttle_delay_seconds = delay;
        }
        if let Some(strategy) = self.injection_strategy {
            effective.injection_strategy = strategy;
        }
        if let Some(template) = &self.break_instruction_template {
            effective.break_instruction_template = Some(template.clone());
        }
        effective.validate()?;
        Ok(effective)
    }
}

#[derive(Debug, Clone, Copy, Default, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum InjectionStrategy {
    #[default]
    SystemPromptAppend,
    ContextAware,
}

#[derive(Debug, Clone, PartialEq)]
pub enum LoopDetectionConfigError {
    InvalidRange {
        field: &'static str,
        value: String,
        expected: String,
    },
    ThresholdOrder {
        warn: f32,
        throttle: f32,
        inject: f32,
        hardstop: f32,
    },
    ConsecutiveCountOrder {
        warn: u32,
        throttle: u32,
        inject: u32,
        hardstop: u32,
    },
    WeightSum {
        sum: f32,
    },
    InvalidBreakInstructionTemplate {
        length: usize,
    },
}

impl fmt::Display for LoopDetectionConfigError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::InvalidRange { field, value, expected } => {
                write!(formatter, "loop_detection.{field} is {value}; expected {expected}")
            }
            Self::ThresholdOrder { warn, throttle, inject, hardstop } => write!(
                formatter,
                "loop_detection thresholds must be strictly ascending (warn {warn} < throttle {throttle} < inject {inject} < hardstop {hardstop})"
            ),
            Self::ConsecutiveCountOrder { warn, throttle, inject, hardstop } => write!(
                formatter,
                "loop_detection consecutive counts must be non-decreasing (warn {warn} <= throttle {throttle} <= inject {inject} <= hardstop {hardstop})"
            ),
            Self::WeightSum { sum } => write!(
                formatter,
                "loop_detection weights must sum to 1.0 within ±{WEIGHT_SUM_TOLERANCE}; got {sum}"
            ),
            Self::InvalidBreakInstructionTemplate { length } => write!(
                formatter,
                "loop_detection.break_instruction_template must contain 1..={MAX_BREAK_INSTRUCTION_TEMPLATE_LENGTH} characters; got {length}"
            ),
        }
    }
}

impl std::error::Error for LoopDetectionConfigError {}

fn validate_u32_range(
    errors: &mut Vec<LoopDetectionConfigError>,
    field: &'static str,
    value: u32,
    min: u32,
    max: u32,
) {
    if !(min..=max).contains(&value) {
        errors.push(LoopDetectionConfigError::InvalidRange {
            field,
            value: value.to_string(),
            expected: format!("a value in {min}..={max}"),
        });
    }
}

fn validate_f32_range(
    errors: &mut Vec<LoopDetectionConfigError>,
    field: &'static str,
    value: f32,
    min: f32,
    max: f32,
) {
    if !value.is_finite() || !(min..=max).contains(&value) {
        errors.push(LoopDetectionConfigError::InvalidRange {
            field,
            value: value.to_string(),
            expected: format!("a finite value in {min}..={max}"),
        });
    }
}

const fn default_session_timeout_minutes() -> u32 {
    30
}
const fn default_max_sessions() -> u32 {
    10_000
}
const fn default_history_depth() -> u32 {
    5
}
const fn default_throttle_delay_seconds() -> u32 {
    2
}
const fn default_eviction_interval_seconds() -> u32 {
    60
}
fn default_ema_alpha() -> f32 {
    0.3
}
fn default_token_velocity_threshold() -> f32 {
    10_000.0
}
fn default_cost_velocity_threshold() -> f64 {
    0.5
}
fn default_warn_confidence() -> f32 {
    0.3
}
fn default_throttle_confidence() -> f32 {
    0.5
}
fn default_inject_confidence() -> f32 {
    0.7
}
fn default_hardstop_confidence() -> f32 {
    0.9
}
const fn default_warn_count() -> u32 {
    3
}
const fn default_throttle_count() -> u32 {
    5
}
const fn default_inject_count() -> u32 {
    7
}
const fn default_hardstop_count() -> u32 {
    10
}
fn default_content_similarity_weight() -> f32 {
    0.25
}
fn default_tool_call_repetition_weight() -> f32 {
    0.20
}
fn default_response_stagnation_weight() -> f32 {
    0.15
}
fn default_token_velocity_weight() -> f32 {
    0.10
}
fn default_error_cycling_weight() -> f32 {
    0.15
}
fn default_context_growth_weight() -> f32 {
    0.10
}
fn default_cost_velocity_weight() -> f32 {
    0.05
}
