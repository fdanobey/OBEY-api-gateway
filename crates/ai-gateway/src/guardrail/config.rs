//! Guardrail pipeline configuration types.
//!
//! These types deserialize the top-level `guardrails` configuration section
//! (see the design document's "Data Models" section). They follow the crate's
//! established config conventions: `#[serde(default)]` on optional fields,
//! `default_*` helper functions for scalar defaults, and
//! `#[serde(rename_all = "snake_case")]` on enums.
//!
//! This module only defines the configuration data model. Validation lives in
//! [`crate::config::validation`] and runtime behavior lives in the sibling
//! guardrail modules.

use serde::{Deserialize, Serialize};
use std::collections::HashMap;

/// Top-level guardrail configuration (`guardrails` section).
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize, Default)]
pub struct GuardrailConfig {
    /// Declared provider instances, keyed by their unique `name`.
    #[serde(default)]
    pub providers: Vec<GuardrailProviderConfig>,
    /// Named pipeline definitions. Each pipeline requires at least one stage
    /// (enforced during validation).
    #[serde(default)]
    pub pipelines: Vec<PipelineConfig>,
    /// Name of the pipeline designated as the Global_Default_Pipeline, applied
    /// to every request regardless of bindings. Optional.
    #[serde(default)]
    pub global_default_pipeline: Option<String>,
    /// Bindings that attach pipelines to virtual keys, model groups, or routes.
    #[serde(default)]
    pub bindings: GuardrailBindings,
}

/// A named, ordered sequence of guardrail stages.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct PipelineConfig {
    /// Unique pipeline name (uniqueness enforced during validation).
    pub name: String,
    /// Ordered stages. At least one stage is required (enforced during validation).
    pub stages: Vec<StageConfig>,
    /// Optional per-pipeline override of the redaction-notice instruction text.
    /// When absent, `DEFAULT_REDACTION_NOTICE_INSTRUCTION` is used (Req 4.8, 4.9).
    #[serde(default)]
    pub redaction_notice_instruction: Option<String>,
    /// How the redaction-notice instruction is inserted (Req 4.10).
    /// Default: `separate`.
    #[serde(default)]
    pub instruction_insertion_mode: InstructionInsertionMode,
    /// Per-pipeline "failover if refusal is detected" toggle (Req 12.4).
    /// Default: `false` (disabled). The effective toggle is `true` if the binding
    /// OR the resolved pipeline sets it.
    #[serde(default)]
    pub failover_on_refusal: bool,
    /// Optional per-pipeline override of the Refusal_Phrase_List (Req 12.2, 12.13).
    /// When absent (`None`), `DEFAULT_REFUSAL_PHRASES` is used. Each entry is a
    /// case-insensitive regex; validation rejects empty or uncompilable entries.
    #[serde(default)]
    pub refusal_phrase_list: Option<Vec<String>>,
}

/// How the redaction-notice system instruction is inserted into the request.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize, Default)]
#[serde(rename_all = "snake_case")]
pub enum InstructionInsertionMode {
    /// Insert the instruction as a new system message before all existing messages.
    #[default]
    Separate,
    /// Merge the instruction into the existing leading system message (prepend with
    /// a blank-line separator). Falls back to `Separate` if no leading system message
    /// exists.
    Merged,
}

/// A single guardrail stage within a pipeline.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct StageConfig {
    /// Human-readable stage name, used for metric labels and logs.
    pub name: String,
    /// Reference to a declared [`GuardrailProviderConfig::name`].
    pub provider: String,
    /// Whether the stage runs pre-call or post-call.
    pub phase: StagePhase,
    /// Enforcement action taken when the stage's provider reports findings.
    pub action: PolicyAction,
}

/// Enforcement action taken when a guardrail rule matches.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum PolicyAction {
    Allow,
    Block,
    Mask,
    Redact,
    ReplaceWithPolicyMessage,
}

/// The point in the request lifecycle at which a stage executes.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum StagePhase {
    PreCall,
    PostCall,
}

/// Behavior when a provider fails or times out.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum FailurePolicy {
    /// Skip the failing stage and continue the pipeline.
    FailOpen,
    /// Halt the pipeline and reject the request with a guardrail-unavailable error.
    FailClose,
}

/// Declared guardrail provider instance.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct GuardrailProviderConfig {
    /// Unique provider name referenced by [`StageConfig::provider`].
    pub name: String,
    /// Provider backend discriminant.
    #[serde(rename = "type")]
    pub provider_type: GuardrailProviderType,
    /// Failure policy applied on provider error or timeout (required).
    pub failure_policy: FailurePolicy,
    /// Per-call timeout in seconds. Defaults to
    /// [`default_provider_timeout_secs`].
    #[serde(default = "default_provider_timeout_secs")]
    pub timeout_seconds: u64,
    /// Type-specific settings (regex patterns, presidio endpoint, etc.),
    /// flattened into the provider entry.
    #[serde(flatten)]
    pub settings: ProviderSettings,
}

/// Supported guardrail provider backends.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum GuardrailProviderType {
    Regex,
    Presidio,
    OpenaiModeration,
    Lakera,
    CustomHttp,
    Semantic,
}

/// Type-specific provider settings, flattened into each provider entry.
///
/// All fields are optional so a single struct can represent the settings for
/// any [`GuardrailProviderType`]. Each provider implementation reads only the
/// fields relevant to its type; cross-field validity is enforced during
/// configuration validation.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize, Default)]
pub struct ProviderSettings {
    /// Regex provider: named patterns (max 256, enforced during validation).
    #[serde(default)]
    pub patterns: Vec<RegexPatternConfig>,
    /// Presidio / custom_http provider: analysis endpoint URL.
    #[serde(default)]
    pub endpoint: Option<String>,
    /// custom_http provider: content-analysis URL (alias for `endpoint`).
    #[serde(default)]
    pub url: Option<String>,
    /// Presidio provider: entity types to detect (at least one required).
    #[serde(default)]
    pub entities: Vec<String>,
    /// Presidio provider: minimum confidence score threshold (0.0–1.0).
    #[serde(default)]
    pub confidence_threshold: Option<f32>,
    /// Semantic provider: allow-collection cosine-similarity threshold (0.0–1.0).
    #[serde(default)]
    pub allow_threshold: Option<f32>,
    /// Semantic provider: deny-collection cosine-similarity threshold (0.0–1.0).
    #[serde(default)]
    pub deny_threshold: Option<f32>,
    /// Semantic provider: Qdrant collection holding allow-example embeddings.
    #[serde(default)]
    pub allow_collection: Option<String>,
    /// Semantic provider: Qdrant collection holding deny-example embeddings.
    #[serde(default)]
    pub deny_collection: Option<String>,
    /// openai_moderation / lakera provider: env var name (or literal) holding
    /// the upstream API key.
    #[serde(default)]
    pub api_key_env: Option<String>,
}

/// A single named regex rule for the regex guardrail provider.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct RegexPatternConfig {
    /// Unique pattern name, surfaced in compile errors and warnings.
    pub name: String,
    /// The regex string, compiled at configuration load time.
    pub regex: String,
    /// Entity label reported on matches (e.g. `API_KEY`, `US_SSN`).
    pub entity: String,
    /// Whether this pattern is an allow-list or deny-list rule.
    #[serde(default)]
    pub mode: RegexRuleMode,
}

/// Regex rule mode. Allow-list matches take precedence over deny-list matches.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize, Default)]
#[serde(rename_all = "snake_case")]
pub enum RegexRuleMode {
    Allow,
    #[default]
    Deny,
}

/// Pipeline bindings by target. Each value references a defined pipeline name.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize, Default)]
pub struct GuardrailBindings {
    /// Virtual-key id → pipeline name.
    #[serde(default)]
    pub virtual_keys: HashMap<String, String>,
    /// Model-group name → pipeline name.
    #[serde(default)]
    pub model_groups: HashMap<String, String>,
    /// Route path → pipeline name.
    #[serde(default)]
    pub routes: HashMap<String, String>,
    /// Per-binding "failover if refusal is detected" toggle (Req 12.4). A binding
    /// target present here with value `true` enables refusal-failover for that target
    /// regardless of the resolved pipeline's own toggle. Default: absent (disabled).
    #[serde(default)]
    pub failover_on_refusal: HashMap<String, bool>,
}

/// Default per-provider `analyze` timeout in seconds (Req 8.6).
pub fn default_provider_timeout_secs() -> u64 {
    5
}
