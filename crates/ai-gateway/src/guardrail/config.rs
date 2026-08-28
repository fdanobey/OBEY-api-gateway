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
    /// Maximum number of distinct PII values tracked for re-injection per request.
    /// Values beyond this cap are still redacted but not restored downstream.
    /// Defaults to 256. Range: 1–10000. (Req 4.3, 4.12).
    #[serde(default = "default_max_reinjection_entries")]
    pub max_reinjection_entries: usize,
}

fn default_max_reinjection_entries() -> usize {
    256
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
    /// Per-pipeline tuning for the `tool_result` phase (indirect-injection
    /// defense). Controls whether JSON-object tool-result content is
    /// serialized and scanned. Default: enabled.
    #[serde(default)]
    pub tool_result: ToolResultPhaseConfig,
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
///
/// The four phases compose into two execution windows:
/// - Pre-call window (inbound request): `pre_call` → `tool_result` → `tool_call`
///   (inbound assistant-history tool calls).
/// - Post-call window (outbound response): `post_call` → `tool_call`
///   (assistant-emitted tool calls, including assembled SSE streams).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum StagePhase {
    /// All message content slots, all roles (behavior unchanged).
    PreCall,
    /// Inbound `role:"tool"` message content (string, text parts, and — when
    /// enabled — compact-JSON serialization of object content).
    ToolResult,
    /// Outbound assistant `tool_calls` (function name + decoded arguments);
    /// also applied to inbound assistant-history tool calls.
    ToolCall,
    /// Assistant response text (behavior unchanged).
    PostCall,
}

impl StagePhase {
    /// Stable label used in metrics, logs, and admin UI (matches serde form).
    pub fn as_str(self) -> &'static str {
        match self {
            StagePhase::PreCall => "pre_call",
            StagePhase::ToolResult => "tool_result",
            StagePhase::ToolCall => "tool_call",
            StagePhase::PostCall => "post_call",
        }
    }
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
    /// Local, deterministic Unicode steganography detector (zero-width
    /// characters, tag-char ASCII smuggling, bidi controls, mixed-script
    /// homoglyph confusables). No network I/O.
    UnicodeStego,
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
    /// Presidio provider: analyzer language code. Defaults to `en`.
    #[serde(default)]
    pub language: Option<String>,
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
    /// unicode_stego provider: category toggles and suppression thresholds.
    /// Ignored by other provider types.
    #[serde(default)]
    pub unicode_stego: UnicodeStegoSettings,
}

/// Settings for the `unicode_stego` provider (indirect-injection defense).
///
/// Detection categories are independently toggleable so operators can, e.g.,
/// alert on tag-character smuggling while masking zero-width noise. Findings
/// below a category's suppression threshold (character count) are dropped to
/// avoid false positives on benign typography (Req 2.x of the
/// indirect-injection-defense spec).
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct UnicodeStegoSettings {
    /// Detect Unicode tag characters U+E0000–U+E007F (ASCII smuggling).
    /// Default: `true`.
    #[serde(default = "default_true")]
    pub detect_tag_chars: bool,
    /// Detect zero-width / format characters (U+200B–U+200D, U+2060–U+2064,
    /// U+FEFF, U+180E, U+061C, U+00AD, U+FE00–U+FE0F). Default: `true`.
    #[serde(default = "default_true")]
    pub detect_zero_width: bool,
    /// Detect bidi controls (U+202A–U+202E, U+2066–U+2069). Default: `true`.
    #[serde(default = "default_true")]
    pub detect_bidi: bool,
    /// Detect TR39-style mixed-script homoglyph confusables (moderate
    /// profile). Default: `true`.
    #[serde(default = "default_true")]
    pub detect_mixed_script: bool,
    /// Suppress `zero_width` findings covering fewer than this many
    /// characters. Default: `4`. Range: 0–1000.
    #[serde(default = "default_zero_width_threshold")]
    pub zero_width_threshold: u32,
    /// Suppress `unicode_tag` findings covering fewer than this many
    /// characters. Default: `0` (any tag character is a finding). Range: 0–1000.
    #[serde(default)]
    pub tag_chars_threshold: u32,
    /// Suppress `bidi_control` findings covering fewer than this many
    /// characters. Default: `0`. Range: 0–1000.
    #[serde(default)]
    pub bidi_threshold: u32,
}

impl Default for UnicodeStegoSettings {
    fn default() -> Self {
        Self {
            detect_tag_chars: true,
            detect_zero_width: true,
            detect_bidi: true,
            detect_mixed_script: true,
            zero_width_threshold: default_zero_width_threshold(),
            tag_chars_threshold: 0,
            bidi_threshold: 0,
        }
    }
}

/// Shared "defaults to true" helper for serde.
fn default_true() -> bool {
    true
}

fn default_zero_width_threshold() -> u32 {
    4
}

/// Per-pipeline tuning for the `tool_result` phase.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct ToolResultPhaseConfig {
    /// Serialize non-string/non-array `role:"tool"` content (JSON objects,
    /// nested documents) as compact JSON and scan the serialization.
    /// Default: `true`.
    #[serde(default = "default_true")]
    pub scan_json_content: bool,
}

impl Default for ToolResultPhaseConfig {
    fn default() -> Self {
        Self {
            scan_json_content: true,
        }
    }
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

#[cfg(test)]
mod tests {
    //! Indirect-injection defense, task 1.4 — config parse (accept/reject),
    //! matrix-adjacent deserialization, and defaults for the new phases and
    //! provider settings.

    use super::*;

    #[test]
    fn stage_phase_new_variants_deserialize_snake_case() {
        assert_eq!(
            serde_yaml::from_str::<StagePhase>("tool_result").unwrap(),
            StagePhase::ToolResult
        );
        assert_eq!(
            serde_yaml::from_str::<StagePhase>("tool_call").unwrap(),
            StagePhase::ToolCall
        );
        // Existing variants unchanged.
        assert_eq!(
            serde_yaml::from_str::<StagePhase>("pre_call").unwrap(),
            StagePhase::PreCall
        );
        assert_eq!(
            serde_yaml::from_str::<StagePhase>("post_call").unwrap(),
            StagePhase::PostCall
        );
        // Unknown phase still rejected.
        assert!(serde_yaml::from_str::<StagePhase>("mid_call").is_err());
    }

    #[test]
    fn stage_phase_as_str_matches_serde_form() {
        assert_eq!(StagePhase::ToolResult.as_str(), "tool_result");
        assert_eq!(StagePhase::ToolCall.as_str(), "tool_call");
        assert_eq!(StagePhase::PreCall.as_str(), "pre_call");
        assert_eq!(StagePhase::PostCall.as_str(), "post_call");
    }

    #[test]
    fn unicode_stego_provider_type_deserializes() {
        assert_eq!(
            serde_yaml::from_str::<GuardrailProviderType>("unicode_stego").unwrap(),
            GuardrailProviderType::UnicodeStego
        );
    }

    #[test]
    fn unicode_stego_settings_defaults() {
        let settings: UnicodeStegoSettings = serde_yaml::from_str("{}").unwrap();
        assert_eq!(settings, UnicodeStegoSettings::default());
        assert!(settings.detect_tag_chars);
        assert!(settings.detect_zero_width);
        assert!(settings.detect_bidi);
        assert!(settings.detect_mixed_script);
        assert_eq!(settings.zero_width_threshold, 4);
        assert_eq!(settings.tag_chars_threshold, 0);
        assert_eq!(settings.bidi_threshold, 0);
    }

    #[test]
    fn unicode_stego_settings_rejects_unknown_keys() {
        let yaml = "detect_tag_chars: true\nbogus_flag: true\n";
        assert!(serde_yaml::from_str::<UnicodeStegoSettings>(yaml).is_err());
    }

    #[test]
    fn provider_entry_flattens_unicode_stego_section() {
        let yaml = r"
name: stego-local
type: unicode_stego
failure_policy: fail_open
unicode_stego:
  detect_zero_width: false
  zero_width_threshold: 8
";
        let provider: GuardrailProviderConfig = serde_yaml::from_str(yaml).unwrap();
        assert_eq!(provider.provider_type, GuardrailProviderType::UnicodeStego);
        assert!(!provider.settings.unicode_stego.detect_zero_width);
        assert_eq!(provider.settings.unicode_stego.zero_width_threshold, 8);
        // Other categories keep defaults.
        assert!(provider.settings.unicode_stego.detect_tag_chars);
    }

    #[test]
    fn pipeline_tool_result_gate_defaults_on() {
        let yaml = "name: p\nstages: []\n";
        let pipeline: PipelineConfig = serde_yaml::from_str(yaml).unwrap();
        assert!(pipeline.tool_result.scan_json_content);

        let yaml_off = "name: p\nstages: []\ntool_result:\n  scan_json_content: false\n";
        let pipeline: PipelineConfig = serde_yaml::from_str(yaml_off).unwrap();
        assert!(!pipeline.tool_result.scan_json_content);
    }

    #[test]
    fn pipeline_tool_result_section_rejects_unknown_keys() {
        let yaml = "name: p\nstages: []\ntool_result:\n  scan_json: true\n";
        assert!(serde_yaml::from_str::<PipelineConfig>(yaml).is_err());
    }

    #[test]
    fn old_style_config_deserializes_unchanged() {
        // A pre-feature guardrails YAML must round-trip identically (Req 5.2).
        let yaml = r#"
providers:
  - name: scanner
    type: regex
    failure_policy: fail_close
    patterns:
      - name: key
        regex: "sk-[A-Za-z0-9]+"
        entity: API_KEY
        mode: deny
pipelines:
  - name: standard
    stages:
      - name: block-keys
        provider: scanner
        phase: pre_call
        action: block
global_default_pipeline: standard
"#;
        let config: GuardrailConfig = serde_yaml::from_str(yaml).unwrap();
        assert_eq!(config.providers.len(), 1);
        assert_eq!(config.pipelines.len(), 1);
        assert_eq!(config.pipelines[0].stages[0].phase, StagePhase::PreCall);
        // New sections took defaults, not values.
        assert_eq!(
            config.providers[0].settings.unicode_stego,
            UnicodeStegoSettings::default()
        );
        assert!(config.pipelines[0].tool_result.scan_json_content);
    }
}
