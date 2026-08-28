use super::*;
use std::env;
use std::path::{Path, PathBuf};

use crate::guardrail::GuardrailConfig;
use crate::memory::MemoryConfigError;
use crate::secrets;
use crate::smart_routing::config::MAX_CONTEXT_WINDOW_TOKENS;
use crate::structured_output::config::StructuredOutputConfigError;

const DEFAULT_CONFIG_TEMPLATE: &str = include_str!("../../config.example.yaml");

#[derive(Debug, thiserror::Error)]
#[allow(dead_code)]
pub enum ValidationError {
    #[error("Missing required field: {0}")]
    MissingField(String),

    #[error("Invalid value for field '{field}': {value}. Expected: {expected}")]
    InvalidValue {
        field: String,
        value: String,
        expected: String,
    },

    #[error("Port number must be in range 1-65535, got: {0}")]
    InvalidPort(u16),

    #[error("Timeout value must be positive, got: {0}")]
    InvalidTimeout(u64),

    #[error("At least one provider must be configured")]
    NoProviders,

    #[error("Model group '{0}' must contain at least one model")]
    EmptyModelGroup(String),

    #[error("Model in group '{group}' is missing provider field")]
    MissingProviderField { group: String },

    #[error("Model in group '{group}' is missing model identifier field")]
    MissingModelField { group: String },

    #[error("Environment variable '{0}' is not set")]
    MissingEnvVar(String),

    #[error("Bedrock provider '{0}' requires a region to be configured")]
    MissingBedrockRegion(String),

    #[error(
        "Codex field '{field}' is only valid on oauth+openai providers (provider: '{provider}')"
    )]
    InvalidCodexField { provider: String, field: String },

    // --- Guardrail pipeline validation (Req 1.1, 1.2, 1.9, 1.10, 6.3, 6.6, 7.6, 5.1) ---
    #[error("Guardrail pipeline name must be non-empty")]
    GuardrailEmptyPipelineName,

    #[error("Duplicate guardrail pipeline name: '{0}'")]
    GuardrailDuplicatePipeline(String),

    #[error("Guardrail pipeline '{0}' must contain at least one stage")]
    GuardrailEmptyPipeline(String),

    #[error("Duplicate guardrail provider name: '{0}'")]
    GuardrailDuplicateProvider(String),

    #[error("Guardrail pipeline '{pipeline}' stage {stage_index} references undeclared provider '{provider}'")]
    GuardrailUndeclaredProvider {
        pipeline: String,
        stage_index: usize,
        provider: String,
    },

    #[error("Guardrail binding target '{target}' references undefined pipeline '{pipeline}'")]
    GuardrailUndefinedBindingPipeline { target: String, pipeline: String },

    #[error("Guardrail global_default_pipeline references undefined pipeline '{0}'")]
    GuardrailUndefinedGlobalDefault(String),

    #[error("Guardrail provider '{provider}' (presidio) requires at least one entity type")]
    GuardrailPresidioNoEntities { provider: String },

    #[error("Guardrail provider '{provider}' field '{field}' must be within {min}..={max}, got: {value}")]
    GuardrailThresholdOutOfRange {
        provider: String,
        field: String,
        value: f32,
        min: f32,
        max: f32,
    },

    #[error("Guardrail provider '{provider}' (regex) declares {count} patterns, exceeding the maximum of {max}")]
    GuardrailTooManyPatterns {
        provider: String,
        count: usize,
        max: usize,
    },

#[error("Guardrail pipeline '{pipeline_name}' stage {stage_index} uses action '{action}' which is invalid for phase '{phase}'")]
GuardrailInvalidPhaseAction {
    pipeline_name: String,
    stage_index: usize,
    phase: String,
    action: String,
},

#[error("Guardrail provider '{provider}' unicode_stego field '{field}' must be within {min}..={max}, got: {value}")]
GuardrailStegoThresholdOutOfRange {
    provider: String,
    field: String,
    value: u32,
    min: u32,
    max: u32,
},

#[error("Guardrail pipeline '{pipeline_name}' refusal_phrase_list entry {index} is invalid: {reason} (pattern: '{pattern}')")]
GuardrailInvalidRefusalPhrase {
        pipeline_name: String,
        index: usize,
        pattern: String,
        reason: String,
    },
}

pub type ValidationResult<T> = Result<T, Vec<ValidationError>>;

/// Resolve configuration file path using priority order:
/// 1. --config CLI flag
/// 2. CONFIG_PATH environment variable
/// 3. ./config.yaml
/// 4. %APPDATA%/ai-gateway/config.yaml (Windows only)
pub fn resolve_config_path(cli_path: Option<PathBuf>) -> PathBuf {
    // Priority 1: CLI flag
    if let Some(path) = cli_path {
        return path;
    }

    // Priority 2: CONFIG_PATH env var
    if let Ok(path) = env::var("CONFIG_PATH") {
        return PathBuf::from(path);
    }

    // Priority 3: ./config.yaml
    let local_path = PathBuf::from("./config.yaml");
    if local_path.exists() {
        return local_path;
    }

    // Priority 4: %APPDATA%/ai-gateway/config.yaml (Windows)
    #[cfg(target_os = "windows")]
    {
        if let Ok(appdata) = env::var("APPDATA") {
            let appdata_path = PathBuf::from(appdata)
                .join("ai-gateway")
                .join("config.yaml");
            if appdata_path.exists() {
                return appdata_path;
            }
        }
    }

    // Default fallback
    local_path
}

impl Config {
    pub fn validate(&self) -> ValidationResult<()> {
        let mut errors = Vec::new();

        // Validate port range (21.9)
        if self.server.port == 0 {
            errors.push(ValidationError::InvalidPort(self.server.port));
        }

        // Validate timeout values (21.10)
        if self.server.request_timeout_seconds == 0 {
            errors.push(ValidationError::InvalidTimeout(
                self.server.request_timeout_seconds,
            ));
        }

        // Validate compression configuration before it reaches the live request path.
        if let Err(compression_errors) = self.compression.validate() {
            errors.extend(compression_errors.into_iter().map(|error| {
                ValidationError::InvalidValue {
                    field: error.field,
                    value: error.message,
                    expected: "a valid token compression configuration".to_string(),
                }
            }));
        }

        // Validate loop detection before the configuration reaches runtime state.
        if let Err(loop_errors) = self.loop_detection.validate() {
            errors.extend(
                loop_errors
                    .into_iter()
                    .map(|error| ValidationError::InvalidValue {
                        field: "loop_detection".to_string(),
                        value: error.to_string(),
                        expected: "a valid loop detection configuration".to_string(),
                    }),
            );
        }

        let configured_provider_names: std::collections::HashSet<&str> = self
            .providers
            .iter()
            .map(|provider| provider.name.as_str())
            .collect();
        let configured_model_group_names: std::collections::HashSet<&str> = self
            .model_groups
            .iter()
            .map(|group| group.name.as_str())
            .collect();

        if let Err(smart_routing_errors) = self.smart_routing.validate() {
            errors.extend(smart_routing_errors.into_iter().map(|error| {
                ValidationError::InvalidValue {
                    field: format!("smart_routing.{}", error.field),
                    value: error.message,
                    expected: "a valid smart routing configuration".to_string(),
                }
            }));
        }

        for group in self.smart_routing.model_group_overrides.keys() {
            if !configured_model_group_names.contains(group.as_str()) {
                errors.push(ValidationError::InvalidValue {
                    field: format!("smart_routing.model_group_overrides.{group}"),
                    value: group.clone(),
                    expected: "the exact name of a configured model group".to_string(),
                });
            }
        }
        for group in self.smart_routing.budget_limits.keys() {
            if !configured_model_group_names.contains(group.as_str()) {
                errors.push(ValidationError::InvalidValue {
                    field: format!("smart_routing.budget_limits.{group}"),
                    value: group.clone(),
                    expected: "the exact name of a configured model group".to_string(),
                });
            }
        }

        if let Some(memory) = &self.memory {
            self.validate_memory_config("memory", memory.validate(), &mut errors);
            self.validate_memory_config(
                "memory",
                memory.validate_with_provider_names(configured_provider_names.iter().copied()),
                &mut errors,
            );

            if let Some(qdrant) = &memory.qdrant {
                let embedding_provider = qdrant.embedding_provider.trim();
                if !embedding_provider.is_empty()
                    && !configured_provider_names.contains(embedding_provider)
                {
                    errors.push(ValidationError::InvalidValue {
                        field: "memory.qdrant.embedding_provider".to_string(),
                        value: qdrant.embedding_provider.clone(),
                        expected: "the exact name of a configured provider".to_string(),
                    });
                }
            }
        }

        if let Some(structured_output) = &self.structured_output {
            self.validate_structured_output_config(
                "structured_output",
                structured_output.validate(),
                &mut errors,
            );
            self.warn_for_unknown_structured_output_providers(
                "structured_output",
                &structured_output.passthrough_providers,
                &configured_provider_names,
            );
        }

        // Validate at least one provider (21.7)
        if self.providers.is_empty() {
            errors.push(ValidationError::NoProviders);
        }

        // Validate provider timeouts and env vars
        for provider in &self.providers {
            if let Some(memory) = &provider.memory {
                self.validate_memory_config(
                    &format!("providers.{}.memory", provider.name),
                    memory.validate(),
                    &mut errors,
                );
            }

            if provider.timeout_seconds == 0 {
                errors.push(ValidationError::InvalidTimeout(provider.timeout_seconds));
            }

            if let Some(t) = provider.ttfb_timeout_seconds {
                if t == 0 {
                    errors.push(ValidationError::InvalidTimeout(t));
                }
            }

            if let Some(t) = provider.total_timeout_seconds {
                if t == 0 {
                    errors.push(ValidationError::InvalidTimeout(t));
                }
            }

            if provider.connection_pool.max_idle_per_host == 0 {
                errors.push(ValidationError::InvalidValue {
                    field: format!(
                        "providers.{}.connection_pool.max_idle_per_host",
                        provider.name
                    ),
                    value: provider.connection_pool.max_idle_per_host.to_string(),
                    expected: "a positive integer".to_string(),
                });
            }

            if provider.connection_pool.idle_timeout_seconds == 0 {
                errors.push(ValidationError::InvalidValue {
                    field: format!(
                        "providers.{}.connection_pool.idle_timeout_seconds",
                        provider.name
                    ),
                    value: provider.connection_pool.idle_timeout_seconds.to_string(),
                    expected: "a positive integer".to_string(),
                });
            }

            if let Some(budget) = &provider.budget {
                if !budget.limit_usd.is_finite() || budget.limit_usd <= 0.0 {
                    errors.push(ValidationError::InvalidValue {
                        field: format!("providers.{}.budget.limit_usd", provider.name),
                        value: budget.limit_usd.to_string(),
                        expected: "a positive finite dollar amount".to_string(),
                    });
                }
            }

            // Warn about missing API key env vars but don't block startup (configurable via UI)
            if let Some(ref env_var) = provider.api_key_env {
                if !env_var.is_empty()
                    && secrets::is_env_var_reference(env_var)
                    && env::var(env_var).is_err()
                    && provider.resolved_api_key.is_none()
                {
                    tracing::warn!("Environment variable '{}' for provider '{}' is not set â€” provider will be unavailable until configured", env_var, provider.name);
                }
            }

            if provider.api_key_encrypted.is_some() && provider.resolved_api_key.is_none() {
                tracing::warn!(
                    "Encrypted API key for provider '{}' could not be resolved â€” provider will be unavailable until the key is re-entered",
                    provider.name
                );
            }

            if let Some(ref env_var) = provider.api_secret_env {
                if !env_var.is_empty() && env::var(env_var).is_err() {
                    tracing::warn!("Environment variable '{}' for provider '{}' is not set â€” provider will be unavailable until configured", env_var, provider.name);
                }
            }

            // OAuth auth_method is only supported for OpenAI providers (6.1, 6.5)
            if provider.auth_method.as_deref() == Some("oauth")
                && provider.provider_type != "openai"
            {
                errors.push(ValidationError::InvalidValue {
                    field: format!("providers.{}.auth_method", provider.name),
                    value: "oauth".to_string(),
                    expected: "auth_method 'oauth' is only supported for provider_type 'openai'"
                        .to_string(),
                });
            }

            // Codex-specific field validation (Req 10.6)
            let is_codex_capable = provider.auth_method.as_deref() == Some("oauth")
                && provider.provider_type == "openai";

            if !is_codex_capable {
                if provider.codex_base_url_override.is_some() {
                    errors.push(ValidationError::InvalidCodexField {
                        provider: provider.name.clone(),
                        field: "codex_base_url_override".to_string(),
                    });
                }
                if provider.codex_model_override.is_some() {
                    errors.push(ValidationError::InvalidCodexField {
                        provider: provider.name.clone(),
                        field: "codex_model_override".to_string(),
                    });
                }
                if provider.instructions_override.is_some() {
                    errors.push(ValidationError::InvalidCodexField {
                        provider: provider.name.clone(),
                        field: "instructions_override".to_string(),
                    });
                }
            }

            // Req 10.9 â€” codex_model_override must be non-empty when set
            if let Some(ref m) = provider.codex_model_override {
                if m.trim().is_empty() {
                    errors.push(ValidationError::InvalidCodexField {
                        provider: provider.name.clone(),
                        field: "codex_model_override (must be non-empty)".to_string(),
                    });
                }
            }

            // Req 12.6 â€” ToS warning when Codex provider active with admin auth disabled
            if is_codex_capable && !self.admin.auth.enabled {
                tracing::warn!(
                    provider = %provider.name,
                    "Codex provider active with admin auth disabled â€” ToS risk: \
                     a shared ChatGPT session over an unauthenticated admin panel \
                     violates OpenAI terms. Enable admin.auth.enabled = true."
                );
            }

            // Bedrock-specific validation (9.1, 9.2)
            if provider.provider_type == "bedrock" {
                if provider.region.is_none() {
                    errors.push(ValidationError::MissingBedrockRegion(provider.name.clone()));
                }

                if !provider.has_api_key_configured() {
                    tracing::warn!(
                        "Bedrock provider '{}' has no API key configured â€” authentication is required for Bedrock Mantle endpoints",
                        provider.name
                    );
                }

                if provider.custom_vpc_endpoint
                    && provider.base_url.as_deref().unwrap_or("").is_empty()
                {
                    errors.push(ValidationError::InvalidValue {
                        field: format!("providers.{}.base_url", provider.name),
                        value: "empty".to_string(),
                        expected: "a URL when custom_vpc_endpoint is enabled".to_string(),
                    });
                }
            }
        }

        if !self.retry.jitter_ratio.is_finite() || !(0.0..=1.0).contains(&self.retry.jitter_ratio) {
            errors.push(ValidationError::InvalidValue {
                field: "retry.jitter_ratio".to_string(),
                value: self.retry.jitter_ratio.to_string(),
                expected: "a number between 0.0 and 1.0".to_string(),
            });
        }

        if self.tray.splash_duration_ms == 0 {
            errors.push(ValidationError::InvalidValue {
                field: "tray.splash_duration_ms".to_string(),
                value: self.tray.splash_duration_ms.to_string(),
                expected: "a positive integer in milliseconds".to_string(),
            });
        }

        // Validate streaming config (Req 7.4, 7.5).
        // The whole `streaming` section is re-read on hot-reload via
        // `apply_runtime_config_update`, which validates through this path.
        if let Some(ref streaming) = self.streaming {
            // keepalive_interval_seconds must be within 0â€“60 (0 = disabled).
            if streaming.keepalive_interval_seconds > 60 {
                errors.push(ValidationError::InvalidValue {
                    field: "streaming.keepalive_interval_seconds".to_string(),
                    value: streaming.keepalive_interval_seconds.to_string(),
                    expected: "a value between 0 and 60 seconds".to_string(),
                });
            }

            // chunk_timeout_seconds must be at least 5 seconds.
            if streaming.chunk_timeout_seconds < 5 {
                errors.push(ValidationError::InvalidValue {
                    field: "streaming.chunk_timeout_seconds".to_string(),
                    value: streaming.chunk_timeout_seconds.to_string(),
                    expected: "at least 5 seconds".to_string(),
                });
            }
        }

        if let Some(ref codex_search) = self.codex_search {
            if let Err(msg) = codex_search.validate() {
                errors.push(ValidationError::InvalidValue {
                    field: "codex_search".to_string(),
                    value: msg,
                    expected: "a valid codex search configuration".to_string(),
                });
            }

            let has_codex_provider = self
                .providers
                .iter()
                .any(|p| p.auth_method.as_deref() == Some("oauth") && p.provider_type == "openai");

            if codex_search.effective_enabled(has_codex_provider) && !has_codex_provider {
                tracing::warn!(
    "codex_search.enabled is true but no Codex (oauth+openai) provider is configured â€” \
    search tools will not be injected and no upstream search calls will be made"
    );
            }
        }

        // Validate admin auth env vars (21.5)
        if self.admin.auth.enabled {
            if let Some(ref env_var) = self.admin.auth.username_env {
                if env::var(env_var).is_err() {
                    tracing::warn!("Admin auth env var '{}' is not set â€” admin auth will be disabled until configured", env_var);
                }
            }
            if let Some(ref env_var) = self.admin.auth.password_env {
                if env::var(env_var).is_err() {
                    tracing::warn!("Admin auth env var '{}' is not set â€” admin auth will be disabled until configured", env_var);
                }
            }
        }

        // Validate model groups (21.8)
        for group in &self.model_groups {
            if let Some(memory) = &group.memory {
                self.validate_memory_config(
                    &format!("model_groups.{}.memory", group.name),
                    memory.validate(),
                    &mut errors,
                );
            }

            if group.models.is_empty() {
                errors.push(ValidationError::EmptyModelGroup(group.name.clone()));
            }

            if let Some(structured_output) = &group.structured_output {
                let scope = format!("model_groups.{}.structured_output", group.name);
                self.validate_structured_output_config(
                    &scope,
                    structured_output.validate(),
                    &mut errors,
                );
                if let Some(passthrough_providers) =
                    structured_output.passthrough_providers.as_deref()
                {
                    self.warn_for_unknown_structured_output_providers(
                        &scope,
                        passthrough_providers,
                        &configured_provider_names,
                    );
                }
            }

            // Validate each model has provider and model fields (4.3)
            for model in &group.models {
                if model.provider.is_empty() {
                    errors.push(ValidationError::MissingProviderField {
                        group: group.name.clone(),
                    });
                }
                if model.model.is_empty() {
                    errors.push(ValidationError::MissingModelField {
                        group: group.name.clone(),
                    });
                }
                if model.context_window > MAX_CONTEXT_WINDOW_TOKENS {
                    errors.push(ValidationError::InvalidValue {
                        field: format!(
                            "model_groups.{}.models.{}.context_window",
                            group.name, model.model
                        ),
                        value: model.context_window.to_string(),
                        expected: format!(
                            "0 for unknown, or a token count in 1..={MAX_CONTEXT_WINDOW_TOKENS}"
                        ),
                    });
                }

                let mut specializations = std::collections::HashSet::new();
                for specialization in &model.specializations {
                    if !specializations.insert(*specialization) {
                        errors.push(ValidationError::InvalidValue {
                            field: format!(
                                "model_groups.{}.models.{}.specializations",
                                group.name, model.model
                            ),
                            value: format!("duplicate {specialization:?}"),
                            expected: "each task type at most once".to_string(),
                        });
                    }
                }
            }
        }

        // Validate guardrail pipelines (Req 1.1, 1.2, 1.9, 1.10, 6.3, 6.6, 7.6, 8.7).
        // Runs only when the opt-in `guardrails` section is present; an absent
        // section disables all guardrail processing and skips validation.
        if let Some(ref guardrails) = self.guardrails {
            self.validate_guardrails(guardrails, &mut errors);
        }

        // Validate context truncation strategy (F1)
        let valid_strategies = ["remove_oldest", "sliding_window"];
        if !valid_strategies.contains(&self.context.truncation_strategy.as_str()) {
            errors.push(ValidationError::InvalidValue {
                field: "context.truncation_strategy".to_string(),
                value: self.context.truncation_strategy.clone(),
                expected: "one of: remove_oldest, sliding_window".to_string(),
            });
        }

        if errors.is_empty() {
            Ok(())
        } else {
            Err(errors)
        }
    }

    fn validate_memory_config(
        &self,
        scope: &str,
        result: Result<(), Vec<MemoryConfigError>>,
        errors: &mut Vec<ValidationError>,
    ) {
        if let Err(memory_errors) = result {
            errors.extend(
                memory_errors
                    .into_iter()
                    .map(|error| ValidationError::InvalidValue {
                        field: format!("{scope}.{}", error.field),
                        value: error.message,
                        expected: "a valid persistent memory configuration".to_string(),
                    }),
            );
        }
    }

    fn validate_structured_output_config(
        &self,
        scope: &str,
        result: Result<(), Vec<StructuredOutputConfigError>>,
        errors: &mut Vec<ValidationError>,
    ) {
        if let Err(structured_output_errors) = result {
            errors.extend(structured_output_errors.into_iter().map(|error| {
                let field = match &error {
                    StructuredOutputConfigError::InvalidRange { field, .. } => {
                        format!("{scope}.{field}")
                    }
                    StructuredOutputConfigError::TooManyPassthroughProviders { .. } => {
                        format!("{scope}.passthrough_providers")
                    }
                    StructuredOutputConfigError::PassthroughProviderTooLong { index, .. } => {
                        format!("{scope}.passthrough_providers[{index}]")
                    }
                };

                ValidationError::InvalidValue {
                    field,
                    value: error.to_string(),
                    expected: "a valid structured output configuration".to_string(),
                }
            }));
        }
    }

    fn warn_for_unknown_structured_output_providers(
        &self,
        scope: &str,
        passthrough_providers: &[String],
        configured_provider_names: &std::collections::HashSet<&str>,
    ) {
        for provider_name in passthrough_providers {
            if !configured_provider_names.contains(provider_name.as_str()) {
                tracing::warn!(
                    scope,
                    provider = %provider_name,
                    "Structured-output passthrough provider does not match a configured provider"
                );
            }
        }
    }

    /// Validate the opt-in guardrail configuration section.
    ///
    /// Collects every violation into `errors` (matching the accumulate-then-
    /// report pattern used throughout [`Config::validate`]) so a single load
    /// surfaces all problems at once.
    fn validate_guardrails(&self, guardrails: &GuardrailConfig, errors: &mut Vec<ValidationError>) {
        use crate::guardrail::{GuardrailProviderType, PolicyAction, StagePhase};
        use std::collections::HashSet;

        /// Maximum regex patterns per regex provider (Req 5.1).
        const MAX_REGEX_PATTERNS: usize = 256;

        /// Maximum stego suppression threshold (indirect-injection defense).
        const MAX_STEGO_THRESHOLD: u32 = 1000;

        /// Phase × action validity matrix (design §2.4). `replace_with_policy_message`
        /// is invalid for the two inbound phases; all other actions are valid
        /// everywhere.
        fn phase_allows_action(phase: StagePhase, action: PolicyAction) -> bool {
            match (phase, action) {
                (
                    StagePhase::PreCall | StagePhase::ToolResult,
                    PolicyAction::ReplaceWithPolicyMessage,
                ) => false,
                _ => true,
            }
        }

        // Collect declared provider names and detect duplicates. Provider-type
        // specific settings are validated in the same pass.
        let mut provider_names: HashSet<&str> = HashSet::new();
        for provider in &guardrails.providers {
            if !provider_names.insert(provider.name.as_str()) {
                errors.push(ValidationError::GuardrailDuplicateProvider(
                    provider.name.clone(),
                ));
            }

            // Presidio: entity list must be non-empty (Req 6.3).
            if provider.provider_type == GuardrailProviderType::Presidio
                && provider.settings.entities.is_empty()
            {
                errors.push(ValidationError::GuardrailPresidioNoEntities {
                    provider: provider.name.clone(),
                });
            }

            // Threshold ranges must be within 0.0..=1.0:
            //   - presidio confidence_threshold (Req 6.6)
            //   - semantic allow/deny thresholds (Req 7.6)
            let thresholds = [
                (
                    "confidence_threshold",
                    provider.settings.confidence_threshold,
                ),
                ("allow_threshold", provider.settings.allow_threshold),
                ("deny_threshold", provider.settings.deny_threshold),
            ];
            for (field, value) in thresholds {
                if let Some(v) = value {
                    if !v.is_finite() || !(0.0..=1.0).contains(&v) {
                        errors.push(ValidationError::GuardrailThresholdOutOfRange {
                            provider: provider.name.clone(),
                            field: field.to_string(),
                            value: v,
                            min: 0.0,
                            max: 1.0,
                        });
                    }
                }
            }

            // Regex provider: pattern count cap (Req 5.1).
            if provider.provider_type == GuardrailProviderType::Regex
                && provider.settings.patterns.len() > MAX_REGEX_PATTERNS
            {
                errors.push(ValidationError::GuardrailTooManyPatterns {
                    provider: provider.name.clone(),
                    count: provider.settings.patterns.len(),
                    max: MAX_REGEX_PATTERNS,
                });
            }

            // unicode_stego provider: suppression thresholds must be within
            // 0..=1000 (indirect-injection defense, task 1.3).
            if provider.provider_type == GuardrailProviderType::UnicodeStego {
                let stego = &provider.settings.unicode_stego;
                let thresholds = [
                    (
                        "zero_width_threshold",
                        stego.zero_width_threshold,
                    ),
                    ("tag_chars_threshold", stego.tag_chars_threshold),
                    ("bidi_threshold", stego.bidi_threshold),
                ];
                for (field, value) in thresholds {
                    if value > MAX_STEGO_THRESHOLD {
                        errors.push(ValidationError::GuardrailStegoThresholdOutOfRange {
                            provider: provider.name.clone(),
                            field: field.to_string(),
                            value,
                            min: 0,
                            max: MAX_STEGO_THRESHOLD,
                        });
                    }
                }
            }

            // Req 8.7: each provider must declare a failure_policy. The field is
            // a required, non-Option `FailurePolicy`, so its presence (and
            // validity) is structurally guaranteed at deserialization time;
            // no additional runtime check is required here.
        }

        // Collect defined pipeline names, enforcing uniqueness, non-empty names,
        // and at least one stage per pipeline (Req 1.1). Stage provider
        // references are validated against the declared provider set (Req 1.2,
        // 1.9). `PolicyAction` and `StagePhase` are enums, so invalid action or
        // phase values are rejected at deserialization; validity is structurally
        // guaranteed here.
        let mut pipeline_names: HashSet<&str> = HashSet::new();
        for pipeline in &guardrails.pipelines {
            if pipeline.name.trim().is_empty() {
                errors.push(ValidationError::GuardrailEmptyPipelineName);
            }
            if !pipeline_names.insert(pipeline.name.as_str()) {
                errors.push(ValidationError::GuardrailDuplicatePipeline(
                    pipeline.name.clone(),
                ));
            }

            if pipeline.stages.is_empty() {
                errors.push(ValidationError::GuardrailEmptyPipeline(
                    pipeline.name.clone(),
                ));
            }

            for (stage_index, stage) in pipeline.stages.iter().enumerate() {
                if !provider_names.contains(stage.provider.as_str()) {
                    errors.push(ValidationError::GuardrailUndeclaredProvider {
                        pipeline: pipeline.name.clone(),
                        stage_index,
                        provider: stage.provider.clone(),
                    });
                }

                // Phase × action validity matrix (design §2.4, task 1.3).
                if !phase_allows_action(stage.phase, stage.action) {
                    errors.push(ValidationError::GuardrailInvalidPhaseAction {
                        pipeline_name: pipeline.name.clone(),
                        stage_index,
                        phase: stage.phase.as_str().to_string(),
                        action: serde_json::to_string(&stage.action)
                            .unwrap_or_default()
                            .trim_matches('"')
                            .to_string(),
                    });
                }
            }

            // Req 12.2, 12.13: validate refusal_phrase_list override entries.
            // Each entry must be non-empty and compilable as a case-insensitive regex.
            if let Some(ref phrases) = pipeline.refusal_phrase_list {
                for (index, pattern) in phrases.iter().enumerate() {
                    if pattern.is_empty() {
                        errors.push(ValidationError::GuardrailInvalidRefusalPhrase {
                            pipeline_name: pipeline.name.clone(),
                            index,
                            pattern: pattern.clone(),
                            reason: "empty refusal phrase".to_string(),
                        });
                    } else if let Err(e) = regex::RegexBuilder::new(pattern)
                        .case_insensitive(true)
                        .build()
                    {
                        errors.push(ValidationError::GuardrailInvalidRefusalPhrase {
                            pipeline_name: pipeline.name.clone(),
                            index,
                            pattern: pattern.clone(),
                            reason: e.to_string(),
                        });
                    }
                }
            }
        }

        // global_default_pipeline, if set, must reference a defined pipeline.
        if let Some(ref default_name) = guardrails.global_default_pipeline {
            if !pipeline_names.contains(default_name.as_str()) {
                errors.push(ValidationError::GuardrailUndefinedGlobalDefault(
                    default_name.clone(),
                ));
            }
        }

        // Every binding (virtual_keys / model_groups / routes) must reference a
        // defined pipeline; undefined â†’ error identifying binding target and
        // pipeline name (Req 1.10).
        let binding_groups: [(&str, &std::collections::HashMap<String, String>); 3] = [
            ("virtual_keys", &guardrails.bindings.virtual_keys),
            ("model_groups", &guardrails.bindings.model_groups),
            ("routes", &guardrails.bindings.routes),
        ];
        for (kind, map) in binding_groups {
            for (target, pipeline_name) in map {
                if !pipeline_names.contains(pipeline_name.as_str()) {
                    errors.push(ValidationError::GuardrailUndefinedBindingPipeline {
                        target: format!("{kind}.{target}"),
                        pipeline: pipeline_name.clone(),
                    });
                }
            }
        }
    }
}

pub fn load_and_validate_config(path: &Path) -> Result<Config, String> {
    // Check if file exists (21.1)
    if !path.exists() {
        return Err(format!(
            "Configuration file not found at expected path: {}",
            path.display()
        ));
    }

    // Read file
    let contents = std::fs::read_to_string(path)
        .map_err(|e| format!("Failed to read configuration file: {}", e))?;

    // Parse YAML (21.2)
    let mut config: Config =
        serde_yaml::from_str(&contents).map_err(|e| format!("Invalid YAML syntax: {}", e))?;

    for provider in &mut config.providers {
        // Handle api_key
        provider.resolved_api_key = None;

        if let Some(encrypted) = provider.api_key_encrypted.as_deref() {
            match secrets::decrypt_provider_secret(encrypted) {
                Ok(decrypted) => provider.resolved_api_key = Some(decrypted),
                Err(error) => tracing::warn!(
                    provider = %provider.name,
                    error = %error,
                    "Failed to decrypt provider API key"
                ),
            }
        } else if let Some(api_key_env) = provider.api_key_env.as_deref() {
            if secrets::looks_like_plaintext_secret(api_key_env) {
                provider.resolved_api_key = Some(api_key_env.to_string());
            }
        }

        // Handle api_secret
        provider.resolved_api_secret = None;

        if let Some(encrypted) = provider.api_secret_encrypted.as_deref() {
            match secrets::decrypt_provider_secret(encrypted) {
                Ok(decrypted) => provider.resolved_api_secret = Some(decrypted),
                Err(error) => tracing::warn!(
                    provider = %provider.name,
                    error = %error,
                    "Failed to decrypt provider API secret"
                ),
            }
        } else if let Some(api_secret_env) = provider.api_secret_env.as_deref() {
            if secrets::looks_like_plaintext_secret(api_secret_env) {
                provider.resolved_api_secret = Some(api_secret_env.to_string());
            }
        }
    }

    // Validate configuration (21.3, 21.4, 41.2, 41.4)
    config.validate().map_err(|errors| {
        let error_messages: Vec<String> = errors.iter().map(|e| e.to_string()).collect();
        format!(
            "Configuration validation failed:\n  - {}",
            error_messages.join("\n  - ")
        )
    })?;

    Ok(config)
}

pub fn bootstrap_config_if_missing(path: &Path) -> Result<bool, String> {
    if path.exists() {
        return Ok(false);
    }

    if let Some(parent) = path.parent() {
        if !parent.as_os_str().is_empty() {
            std::fs::create_dir_all(parent)
                .map_err(|e| format!("Failed to create configuration directory: {}", e))?;
        }
    }

    std::fs::write(path, DEFAULT_CONFIG_TEMPLATE)
        .map_err(|e| format!("Failed to create default configuration file: {}", e))?;

    Ok(true)
}

pub fn save_config(path: &Path, config: &Config) -> Result<(), String> {
    let yaml = serde_yaml::to_string(config)
        .map_err(|e| format!("Failed to serialize configuration: {}", e))?;

    if let Some(parent) = path.parent() {
        if !parent.as_os_str().is_empty() {
            std::fs::create_dir_all(parent)
                .map_err(|e| format!("Failed to create configuration directory: {}", e))?;
        }
    }

    std::fs::write(path, yaml).map_err(|e| format!("Failed to write configuration file: {}", e))
}

#[cfg(test)]
mod property_tests {
    use super::*;
    use proptest::prelude::*;
    use std::io::Write;
    use std::sync::{Arc, Mutex};

    #[derive(Clone)]
    struct CapturedWriter(Arc<Mutex<Vec<u8>>>);

    impl Write for CapturedWriter {
        fn write(&mut self, buffer: &[u8]) -> std::io::Result<usize> {
            self.0.lock().unwrap().extend_from_slice(buffer);
            Ok(buffer.len())
        }

        fn flush(&mut self) -> std::io::Result<()> {
            Ok(())
        }
    }

    // Helper to create minimal valid config
    fn minimal_valid_config() -> Config {
        Config {
            server: ServerConfig {
                host: "0.0.0.0".to_string(),
                port: 8080,
                request_timeout_seconds: 30,
                max_request_size_mb: 10,
            },
            tls: None,
            admin: AdminConfig::default(),
            dashboard: DashboardConfig::default(),
            cors: CorsConfig::default(),
            providers: vec![Provider {
                name: "test-provider".to_string(),
                provider_type: "openai".to_string(),
                base_url: Some("https://api.openai.com/v1".to_string()),
                api_key_env: None, // No env var required for test
                api_key_encrypted: None,
                api_secret_env: None,
                api_secret_encrypted: None,
                auth_method: None,
                resolved_api_key: None,
                resolved_api_secret: None,
                region: None,
                timeout_seconds: 30,
                ttfb_timeout_seconds: None,
                total_timeout_seconds: None,
                max_connections: 100,
                rate_limit_per_minute: 0,
                custom_headers: Default::default(),
                connection_pool: ProviderConnectionPoolConfig::default(),
                budget: None,
                manual_models: vec![],
                global_inference_profile: false,
                cross_region_inference: false,
                custom_vpc_endpoint: false,
                prompt_caching: false,
                compression: None,
                memory: None,
                reasoning: true,
                codex_base_url_override: None,
                codex_model_override: None,
                instructions_override: None,
                max_rate_limit_cooldown_seconds: None,
            }],
            model_groups: vec![ModelGroup {
                name: "test-group".to_string(),
                version_fallback_enabled: false,
                compression: None,
                memory: None,
                structured_output: None,
                models: vec![ProviderModel {
                    provider: "test-provider".to_string(),
                    model: "gpt-4".to_string(),
                    cost_per_million_input_tokens: 10.0,
                    cost_per_million_output_tokens: 30.0,
                    priority: 100,
                    structured_output_passthrough: None,
                    tier: None,
                    context_window: 0,
                    specializations: vec![],
                }],
            }],
            circuit_breaker: CircuitBreakerConfig::default(),
            retry: RetryConfig::default(),
            logging: LoggingConfig::default(),
            semantic_cache: None,
            exact_cache: ExactCacheConfig::default(),
            prometheus: None,
            context: ContextConfig::default(),
            compression: Default::default(),
            memory: None,
            first_launch_completed: false,
            tray: TrayConfig::default(),
            codex_instructions_url: None,
            streaming: None,
            virtual_keys: Default::default(),
            loop_detection: Default::default(),
            structured_output: None,
            guardrails: None,
            tool_compression: Default::default(),
            smart_routing: Default::default(),
            xhigh_models_allowlist: Default::default(),
            reasoning_models_allowlist: Default::default(),
            codex_search: None,
        }
    }

    #[test]
    fn smart_routing_unknown_group_keys_are_rejected() {
        let mut config = minimal_valid_config();
        config.smart_routing.model_group_overrides.insert(
            "missing-override".to_string(),
            crate::smart_routing::config::SmartRoutingOverride::default(),
        );
        config.smart_routing.budget_limits.insert(
            "missing-budget".to_string(),
            crate::smart_routing::config::BudgetLimits {
                hourly_limit_usd: Some(1.0),
                ..Default::default()
            },
        );

        let errors = config.validate().unwrap_err();
        assert!(errors.iter().any(|error| matches!(
            error,
            ValidationError::InvalidValue { field, .. }
                if field == "smart_routing.model_group_overrides.missing-override"
        )));
        assert!(errors.iter().any(|error| matches!(
            error,
            ValidationError::InvalidValue { field, .. }
                if field == "smart_routing.budget_limits.missing-budget"
        )));
    }

    #[test]
    fn provider_model_context_and_duplicate_specializations_are_rejected() {
        let mut config = minimal_valid_config();
        let model = &mut config.model_groups[0].models[0];
        model.context_window = MAX_CONTEXT_WINDOW_TOKENS + 1;
        model.specializations = vec![
            crate::smart_routing::tier::TaskType::CodeGeneration,
            crate::smart_routing::tier::TaskType::CodeGeneration,
        ];

        let errors = config.validate().unwrap_err();
        assert!(errors.iter().any(|error| matches!(
            error,
            ValidationError::InvalidValue { field, .. } if field.ends_with(".context_window")
        )));
        assert!(errors.iter().any(|error| matches!(
            error,
            ValidationError::InvalidValue { field, .. } if field.ends_with(".specializations")
        )));
    }

    #[test]
    fn provider_model_zero_and_plausible_context_windows_are_valid() {
        let mut config = minimal_valid_config();
        config.model_groups[0].models[0].context_window = 0;
        assert!(config.validate().is_ok());
        config.model_groups[0].models[0].context_window = 1_000_000;
        assert!(config.validate().is_ok());
    }

    #[test]
    fn test_load_config_resolves_plaintext_provider_key_to_runtime_only() {
        let tempdir = tempfile::tempdir().unwrap();
        let path = tempdir.path().join("config.yaml");

        std::fs::write(
            &path,
            r#"server:
  host: "127.0.0.1"
  port: 8080
  request_timeout_seconds: 30
  max_request_size_mb: 10
providers:
  - name: "openai"
    type: "openai"
    base_url: "https://api.openai.com/v1"
    api_key_env: "sk-test-12345678901234567890"
    timeout_seconds: 30
model_groups:
  - name: "default"
    version_fallback_enabled: false
    models:
      - provider: "openai"
        model: "gpt-4"
        priority: 100
"#,
        )
        .unwrap();

        let config = load_and_validate_config(&path).unwrap();
        assert_eq!(
            config.providers[0].resolved_api_key.as_deref(),
            Some("sk-test-12345678901234567890")
        );
        assert!(!config.first_launch_completed);
        assert_eq!(config.tray, TrayConfig::default());
    }

    #[test]
    fn test_save_config_persists_tray_fields() {
        let tempdir = tempfile::tempdir().unwrap();
        let path = tempdir.path().join("nested").join("config.yaml");

        let mut config = minimal_valid_config();
        config.first_launch_completed = true;
        config.tray = TrayConfig {
            show_notifications: false,
            auto_open_browser: false,
            splash_duration_ms: 1500,
        };

        save_config(&path, &config).unwrap();

        let reloaded = load_and_validate_config(&path).unwrap();
        assert!(reloaded.first_launch_completed);
        assert_eq!(reloaded.tray, config.tray);
    }

    // Feature: ai-gateway, Property 11: API Key Storage
    // **Validates: Requirements 12.8, 19.2**
    proptest! {
        #[test]
        fn prop_api_keys_stored_as_env_var_names(
            api_key in "[A-Z_][A-Z0-9_]{0,50}",
            api_secret in "[A-Z_][A-Z0-9_]{0,50}",
        ) {
            // Property: For any configuration file, API keys shall be stored only as
            // environment variable names, never as literal values.

            let mut config = minimal_valid_config();

            // Set API key and secret as environment variable names
            config.providers[0].api_key_env = Some(api_key.clone());
            config.providers[0].api_secret_env = Some(api_secret.clone());

            // Serialize to YAML (simulating config file storage)
            let yaml = serde_yaml::to_string(&config).unwrap();

            // Verify that the YAML contains the env var names
            prop_assert!(yaml.contains(&api_key),
                "Config should contain env var name: {}", api_key);
            prop_assert!(yaml.contains(&api_secret),
                "Config should contain env var name: {}", api_secret);

            // Verify that the YAML does NOT contain literal API key patterns
            // Common API key patterns: sk-..., Bearer ..., etc.
            let literal_key_patterns = [
                "sk-[a-zA-Z0-9]{20,}",  // OpenAI style
                "Bearer [a-zA-Z0-9]{20,}",  // Bearer token
                "[a-f0-9]{32,}",  // Hex keys
                "AKIA[A-Z0-9]{16}",  // AWS access key
            ];

            for pattern in &literal_key_patterns {
                let re = regex::Regex::new(pattern).unwrap();
                prop_assert!(!re.is_match(&yaml),
                    "Config should not contain literal API key matching pattern: {}", pattern);
            }
        }

        #[test]
        fn prop_admin_auth_stored_as_env_var_names(
            username_env in "[A-Z_][A-Z0-9_]{0,50}",
            password_env in "[A-Z_][A-Z0-9_]{0,50}",
        ) {
            // Property: Admin credentials should also be stored as env var names

            let mut config = minimal_valid_config();
            config.admin.auth.enabled = true;
            config.admin.auth.username_env = Some(username_env.clone());
            config.admin.auth.password_env = Some(password_env.clone());

            // Serialize to YAML
            let yaml = serde_yaml::to_string(&config).unwrap();

            // Verify env var names are present
            prop_assert!(yaml.contains(&username_env),
                "Config should contain username env var: {}", username_env);
            prop_assert!(yaml.contains(&password_env),
                "Config should contain password env var: {}", password_env);

            // Verify no literal passwords (common patterns)
            let password_patterns = [
                r#"password:\s*"[^"]{8,}""#,  // Quoted password
                r#"password:\s*[a-zA-Z0-9]{8,}"#,  // Unquoted password
            ];

            for pattern in &password_patterns {
                let re = regex::Regex::new(pattern).unwrap();
                prop_assert!(!re.is_match(&yaml),
                    "Config should not contain literal password matching pattern: {}", pattern);
            }
        }

        #[test]
        fn prop_custom_headers_may_contain_env_refs(
            header_value in "[A-Z_][A-Z0-9_]{0,50}",
        ) {
            // Property: Custom headers can reference env vars but should not contain literal secrets

            let mut config = minimal_valid_config();
            config.providers[0].custom_headers.insert(
                "X-API-Key".to_string(),
                format!("${{{}}}", header_value)  // ${ENV_VAR} format
            );

            let yaml = serde_yaml::to_string(&config).unwrap();

            // Should contain the env var reference
            prop_assert!(yaml.contains(&header_value),
                "Config should contain env var reference: {}", header_value);
        }
    }

    // Feature: ai-gateway, Property 9: Configuration Validation Rejection
    // **Validates: Requirements 12.6, 12.7, 21.1-21.4, 41.2, 41.4, 41.5**
    proptest! {
        #[test]
        fn prop_invalid_port_rejected(port in prop::num::u16::ANY) {
            if port == 0 {
                let mut config = minimal_valid_config();
                config.server.port = port;

                let result = config.validate();
                prop_assert!(result.is_err(), "Port 0 should be rejected");
            }
        }

        #[test]
        fn prop_zero_timeout_rejected(timeout in prop::num::u64::ANY) {
            if timeout == 0 {
                let mut config = minimal_valid_config();
                config.server.request_timeout_seconds = timeout;

                let result = config.validate();
                prop_assert!(result.is_err(), "Zero timeout should be rejected");
            }
        }

        #[test]
        fn prop_no_providers_rejected(_dummy in prop::num::u8::ANY) {
            let mut config = minimal_valid_config();
            config.providers.clear();

            let result = config.validate();
            prop_assert!(result.is_err(), "Config with no providers should be rejected");
        }

        #[test]
        fn prop_empty_model_group_rejected(_dummy in prop::num::u8::ANY) {
            let mut config = minimal_valid_config();
            config.model_groups[0].models.clear();

            let result = config.validate();
            prop_assert!(result.is_err(), "Model group with no models should be rejected");
        }

        #[test]
        fn prop_missing_provider_field_rejected(_dummy in prop::num::u8::ANY) {
            let mut config = minimal_valid_config();
            config.model_groups[0].models[0].provider = String::new();

            let result = config.validate();
            prop_assert!(result.is_err(), "Model with empty provider should be rejected");
        }

        #[test]
        fn prop_missing_model_field_rejected(_dummy in prop::num::u8::ANY) {
            let mut config = minimal_valid_config();
            config.model_groups[0].models[0].model = String::new();

            let result = config.validate();
            prop_assert!(result.is_err(), "Model with empty model identifier should be rejected");
        }

        #[test]
        fn prop_valid_config_accepted(
            port in 1u16..=65535u16,
            timeout in 1u64..=3600u64,
        ) {
            let mut config = minimal_valid_config();
            config.server.port = port;
            config.server.request_timeout_seconds = timeout;

            let result = config.validate();
            prop_assert!(result.is_ok(), "Valid config should be accepted: {:?}", result);
        }
    }

    // Feature: openai-oauth-login, OAuth auth_method validation
    // **Validates: Requirements 6.1, 6.5**

    #[test]
    fn test_oauth_auth_method_rejected_for_non_openai_provider() {
        let mut config = minimal_valid_config();
        config.providers[0].provider_type = "bedrock".to_string();
        config.providers[0].region = Some("us-east-1".to_string());
        config.providers[0].auth_method = Some("oauth".to_string());

        let result = config.validate();
        assert!(
            result.is_err(),
            "auth_method 'oauth' on non-openai provider should be rejected"
        );
        let errors = result.unwrap_err();
        assert!(
            errors.iter().any(|e| match e {
                ValidationError::InvalidValue { field, value, .. } =>
                    field.contains("auth_method") && value == "oauth",
                _ => false,
            }),
            "Should contain InvalidValue error for auth_method"
        );
    }

    // Streaming Reliability, Task 1.3: StreamingConfig validation
    // **Validates: Requirements 7.4**

    #[test]
    fn valid_enabled_memory_config_is_accepted() {
        let mut config = minimal_valid_config();
        config.memory = Some(crate::memory::MemoryConfig {
            enabled: true,
            auto_extract_enabled: true,
            auto_extract_provider: "test-provider".to_string(),
            auto_extract_model: "gpt-4o-mini".to_string(),
            ..Default::default()
        });

        assert!(config.validate().is_ok());
    }

    #[test]
    fn memory_local_validation_errors_are_aggregated_with_scoped_fields() {
        let mut config = minimal_valid_config();
        config.memory = Some(crate::memory::MemoryConfig {
            database_path: " ".to_string(),
            max_injection_tokens: 10_001,
            auto_extract_enabled: true,
            auto_extract_provider: " ".to_string(),
            auto_extract_model: String::new(),
            auto_extract_min_turns: 0,
            decay_schedule_hours: 0,
            max_memories_per_namespace: 0,
            ..Default::default()
        });

        let errors = config.validate().unwrap_err();
        let fields: Vec<&str> = errors
            .iter()
            .filter_map(|error| match error {
                ValidationError::InvalidValue { field, .. } if field.starts_with("memory.") => {
                    Some(field.as_str())
                }
                _ => None,
            })
            .collect();

        for field in [
            "memory.database_path",
            "memory.max_injection_tokens",
            "memory.auto_extract_provider",
            "memory.auto_extract_model",
            "memory.auto_extract_min_turns",
            "memory.decay_schedule_hours",
            "memory.max_memories_per_namespace",
        ] {
            assert!(fields.contains(&field), "missing {field}: {errors:?}");
        }
    }

    #[test]
    fn missing_memory_provider_references_are_rejected_at_field_level() {
        let mut config = minimal_valid_config();
        config.memory = Some(crate::memory::MemoryConfig {
            auto_extract_enabled: true,
            auto_extract_provider: "missing-auto".to_string(),
            auto_extract_model: "extractor".to_string(),
            qdrant: Some(crate::memory::MemoryQdrantConfig {
                qdrant_url: "https://qdrant.example.com".to_string(),
                embedding_provider: "missing-embedding".to_string(),
                embedding_model: "embedder".to_string(),
                ..Default::default()
            }),
            ..Default::default()
        });

        let errors = config.validate().unwrap_err();
        assert!(errors.iter().any(|error| matches!(
            error,
            ValidationError::InvalidValue { field, value, .. }
                if field == "memory.auto_extract_provider" && value.contains("missing-auto")
        )));
        assert!(errors.iter().any(|error| matches!(
            error,
            ValidationError::InvalidValue { field, value, .. }
                if field == "memory.qdrant.embedding_provider" && value == "missing-embedding"
        )));
    }

    #[test]
    fn invalid_provider_and_model_group_memory_overrides_are_aggregated() {
        let mut config = minimal_valid_config();
        config.providers[0].memory = Some(crate::memory::ProviderMemoryOverride {
            max_injection_tokens: Some(10_001),
            ..Default::default()
        });
        config.model_groups[0].memory = Some(crate::memory::ModelGroupMemoryOverride {
            max_injection_tokens: Some(10_001),
            ..Default::default()
        });

        let errors = config.validate().unwrap_err();
        assert!(errors.iter().any(|error| matches!(
            error,
            ValidationError::InvalidValue { field, .. }
                if field == "providers.test-provider.memory.max_injection_tokens"
        )));
        assert!(errors.iter().any(|error| matches!(
            error,
            ValidationError::InvalidValue { field, .. }
                if field == "model_groups.test-group.memory.max_injection_tokens"
        )));
    }

    #[test]
    fn test_streaming_keepalive_above_max_rejected() {
        let mut config = minimal_valid_config();
        config.streaming = Some(StreamingConfig {
            keepalive_interval_seconds: 61,
            ..StreamingConfig::default()
        });

        let result = config.validate();
        assert!(
            result.is_err(),
            "keepalive_interval_seconds > 60 should be rejected"
        );
        let errors = result.unwrap_err();
        assert!(
            errors.iter().any(|e| matches!(
                e,
                ValidationError::InvalidValue { field, .. }
                    if field == "streaming.keepalive_interval_seconds"
            )),
            "Should contain InvalidValue error for keepalive_interval_seconds"
        );
    }

    #[test]
    fn structured_output_validation_aggregates_global_and_all_group_errors() {
        let mut config = minimal_valid_config();
        config.structured_output = Some(crate::structured_output::config::StructuredOutputConfig {
            max_retries: 6,
            retry_temperature: 3.0,
            ..Default::default()
        });
        config.model_groups[0].structured_output =
            Some(crate::structured_output::config::StructuredOutputOverride {
                max_retries: Some(7),
                retry_temperature: Some(f32::NAN),
                ..Default::default()
            });

        let errors = config.validate().unwrap_err();
        let structured_output_fields: Vec<&str> = errors
            .iter()
            .filter_map(|error| match error {
                ValidationError::InvalidValue { field, .. }
                    if field.contains("structured_output") =>
                {
                    Some(field.as_str())
                }
                _ => None,
            })
            .collect();

        assert_eq!(structured_output_fields.len(), 4);
        assert!(structured_output_fields.contains(&"structured_output.max_retries"));
        assert!(structured_output_fields.contains(&"structured_output.retry_temperature"));
        assert!(structured_output_fields
            .contains(&"model_groups.test-group.structured_output.max_retries"));
        assert!(structured_output_fields
            .contains(&"model_groups.test-group.structured_output.retry_temperature"));
    }

    #[test]
    fn unknown_structured_output_passthrough_provider_warns_without_failing_validation() {
        use tracing::subscriber::with_default;

        let output = Arc::new(Mutex::new(Vec::new()));
        let subscriber = tracing_subscriber::fmt()
            .with_ansi(false)
            .without_time()
            .with_writer({
                let output = Arc::clone(&output);
                move || CapturedWriter(Arc::clone(&output))
            })
            .finish();

        let mut config = minimal_valid_config();
        config.structured_output = Some(crate::structured_output::config::StructuredOutputConfig {
            passthrough_providers: vec!["not-configured".to_string()],
            ..Default::default()
        });
        config.model_groups[0].structured_output =
            Some(crate::structured_output::config::StructuredOutputOverride {
                passthrough_providers: Some(vec!["also-not-configured".to_string()]),
                ..Default::default()
            });

        assert!(with_default(subscriber, || config.validate()).is_ok());

        let warnings = String::from_utf8(output.lock().unwrap().clone()).unwrap();
        assert_eq!(
            warnings
                .matches(
                    "Structured-output passthrough provider does not match a configured provider"
                )
                .count(),
            2
        );
        assert!(warnings.contains("not-configured"));
        assert!(warnings.contains("also-not-configured"));
        assert!(!warnings.contains("schema"));
    }

    #[test]
    fn test_streaming_keepalive_at_max_accepted() {
        let mut config = minimal_valid_config();
        config.streaming = Some(StreamingConfig {
            keepalive_interval_seconds: 60,
            ..StreamingConfig::default()
        });

        let result = config.validate();
        assert!(
            result.is_ok(),
            "keepalive_interval_seconds == 60 should be accepted: {:?}",
            result
        );
    }

    #[test]
    fn test_streaming_keepalive_zero_accepted() {
        // 0 disables keep-alive (axum default) and is within the valid range.
        let mut config = minimal_valid_config();
        config.streaming = Some(StreamingConfig {
            keepalive_interval_seconds: 0,
            ..StreamingConfig::default()
        });

        let result = config.validate();
        assert!(
            result.is_ok(),
            "keepalive_interval_seconds == 0 should be accepted: {:?}",
            result
        );
    }

    #[test]
    fn test_streaming_chunk_timeout_below_min_rejected() {
        let mut config = minimal_valid_config();
        config.streaming = Some(StreamingConfig {
            chunk_timeout_seconds: 4,
            ..StreamingConfig::default()
        });

        let result = config.validate();
        assert!(
            result.is_err(),
            "chunk_timeout_seconds < 5 should be rejected"
        );
        let errors = result.unwrap_err();
        assert!(
            errors.iter().any(|e| matches!(
                e,
                ValidationError::InvalidValue { field, .. }
                    if field == "streaming.chunk_timeout_seconds"
            )),
            "Should contain InvalidValue error for chunk_timeout_seconds"
        );
    }

    #[test]
    fn test_streaming_chunk_timeout_at_min_accepted() {
        let mut config = minimal_valid_config();
        config.streaming = Some(StreamingConfig {
            chunk_timeout_seconds: 5,
            ..StreamingConfig::default()
        });

        let result = config.validate();
        assert!(
            result.is_ok(),
            "chunk_timeout_seconds == 5 should be accepted: {:?}",
            result
        );
    }

    #[test]
    fn test_streaming_default_config_accepted() {
        let mut config = minimal_valid_config();
        config.streaming = Some(StreamingConfig::default());

        let result = config.validate();
        assert!(
            result.is_ok(),
            "Default StreamingConfig should be accepted: {:?}",
            result
        );
    }

    #[test]
    fn test_oauth_auth_method_accepted_for_openai_provider() {
        let mut config = minimal_valid_config();
        config.providers[0].provider_type = "openai".to_string();
        config.providers[0].auth_method = Some("oauth".to_string());

        let result = config.validate();
        assert!(
            result.is_ok(),
            "auth_method 'oauth' on openai provider should be accepted: {:?}",
            result
        );
    }

    #[test]
    fn test_no_auth_method_accepted_for_any_provider() {
        let mut config = minimal_valid_config();
        config.providers[0].provider_type = "anthropic".to_string();
        config.providers[0].auth_method = None;

        let result = config.validate();
        assert!(
            result.is_ok(),
            "No auth_method should be accepted for any provider: {:?}",
            result
        );
    }

    // Feature: bedrock-ui-integration, Bedrock validation rules
    // **Validates: Requirements 9.1, 9.2**

    #[test]
    fn test_bedrock_provider_without_region_rejected() {
        let mut config = minimal_valid_config();
        config.providers[0].provider_type = "bedrock".to_string();
        config.providers[0].region = None;

        let result = config.validate();
        assert!(
            result.is_err(),
            "Bedrock provider without region should be rejected"
        );
        let errors = result.unwrap_err();
        assert!(
            errors
                .iter()
                .any(|e| matches!(e, ValidationError::MissingBedrockRegion(_))),
            "Should contain MissingBedrockRegion error"
        );
    }

    #[test]
    fn test_bedrock_provider_with_region_accepted() {
        let mut config = minimal_valid_config();
        config.providers[0].provider_type = "bedrock".to_string();
        config.providers[0].region = Some("us-east-1".to_string());

        let result = config.validate();
        assert!(
            result.is_ok(),
            "Bedrock provider with region should be accepted: {:?}",
            result
        );
    }

    #[test]
    fn test_non_bedrock_provider_without_region_accepted() {
        let mut config = minimal_valid_config();
        config.providers[0].provider_type = "openai".to_string();
        config.providers[0].region = None;

        let result = config.validate();
        assert!(
            result.is_ok(),
            "Non-bedrock provider without region should be accepted: {:?}",
            result
        );
    }

    // Feature: bedrock-ui-integration, Property 4: Provider config Bedrock fields round-trip through serialization
    // **Validates: Requirements 8.1, 3.2**
    proptest! {
        #![proptest_config(proptest::prelude::ProptestConfig::with_cases(64))]

        #[test]
        fn prop_provider_bedrock_fields_roundtrip(
            global_inference_profile in proptest::bool::ANY,
            prompt_caching in proptest::bool::ANY,
            reasoning in proptest::bool::ANY,
            region in proptest::option::of(
                proptest::sample::select(vec![
                    "us-east-1", "us-west-2",
                    "eu-west-1", "eu-west-3", "eu-central-1",
                    "ap-northeast-1", "ap-southeast-1", "ap-southeast-2", "ap-south-1",
                    "sa-east-1", "ca-central-1", "us-gov-west-1",
                ])
            ),
        ) {
            let provider = Provider {
                name: "bedrock-test".to_string(),
                provider_type: "bedrock".to_string(),
                base_url: Some("https://bedrock-mantle.us-east-1.api.aws/v1".to_string()),
                api_key_env: None,
                api_key_encrypted: None,
                api_secret_env: None,
                api_secret_encrypted: None,
                auth_method: None,
                resolved_api_key: None,
                resolved_api_secret: None,
                region: region.map(|s| s.to_string()),
                timeout_seconds: 30,
                ttfb_timeout_seconds: None,
                total_timeout_seconds: None,
                max_connections: 100,
                rate_limit_per_minute: 0,
                custom_headers: Default::default(),
                connection_pool: ProviderConnectionPoolConfig::default(),
                budget: None,
                manual_models: vec![],
                global_inference_profile,
                cross_region_inference: false,
                custom_vpc_endpoint: false,
                prompt_caching,
                compression: None,
                memory: None,
                reasoning,
                codex_base_url_override: None,
                codex_model_override: None,
                instructions_override: None,
                max_rate_limit_cooldown_seconds: None,
            };

            // Serialize to YAML
            let yaml = serde_yaml::to_string(&provider).unwrap();

            // Deserialize back
            let deserialized: Provider = serde_yaml::from_str(&yaml).unwrap();

            // Assert all Bedrock-specific fields round-trip identically
            prop_assert_eq!(
                deserialized.global_inference_profile, global_inference_profile,
                "global_inference_profile mismatch after round-trip"
            );
            prop_assert_eq!(
                deserialized.prompt_caching, prompt_caching,
                "prompt_caching mismatch after round-trip"
            );
            prop_assert_eq!(
                deserialized.reasoning, reasoning,
                "reasoning mismatch after round-trip"
            );
            prop_assert_eq!(
                deserialized.region, region.map(|s| s.to_string()),
                "region mismatch after round-trip"
            );
        }
    }

    // Feature: openai-oauth-login, Property 6: Backward Compatibility
    // For all valid Provider YAML configs lacking `auth_method`, deserialization
    // succeeds and `auth_method` is `None`.
    // **Validates: Requirements 6.5**
    proptest! {
        #![proptest_config(proptest::prelude::ProptestConfig::with_cases(64))]

        #[test]
        fn prop_provider_without_auth_method_deserializes_with_none(
            name in "[a-z][a-z0-9_-]{0,20}",
            provider_type in proptest::sample::select(vec![
                "openai", "anthropic", "bedrock", "google", "azure", "custom",
            ]),
            base_url in proptest::option::of("https://api\\.example\\.com/v1"),
            api_key_env in proptest::option::of("[A-Z_][A-Z0-9_]{2,20}"),
            timeout in 1u64..=600u64,
            max_connections in 1u32..=500u32,
        ) {
            // Build a YAML string that does NOT include `auth_method`
            let mut yaml = format!(
                "name: \"{name}\"\ntype: \"{provider_type}\"\ntimeout_seconds: {timeout}\nmax_connections: {max_connections}\n"
            );
            if let Some(url) = &base_url {
                yaml.push_str(&format!("base_url: \"{url}\"\n"));
            }
            if let Some(key) = &api_key_env {
                yaml.push_str(&format!("api_key_env: \"{key}\"\n"));
            }

            // Deserialize â€” must succeed
            let provider: Provider = serde_yaml::from_str(&yaml)
                .expect("Provider YAML without auth_method must deserialize successfully");

            // auth_method must be None (backward compatible default)
            prop_assert_eq!(
                provider.auth_method, None,
                "Provider without auth_method field must deserialize with auth_method == None"
            );

            // Sanity: other fields parsed correctly
            prop_assert_eq!(&provider.name, &name);
            prop_assert_eq!(&provider.provider_type, provider_type);
            prop_assert_eq!(provider.timeout_seconds, timeout);
            prop_assert_eq!(provider.max_connections, max_connections);
        }
    }

    // Feature: codex-backend-translation, Property 6: Config Backward Compatibility
    // For all YAML provider configs lacking the three new Codex fields,
    // deserialization succeeds and each new Option<String> field equals None.
    // **Validates: Requirements 10.6, 10.9**
    proptest! {
        #![proptest_config(ProptestConfig::with_cases(50))]

        #[test]
        fn prop_config_backward_compat(
            name in "[a-z]{3,10}",
            provider_type in prop_oneof![
                Just("openai".to_string()),
                Just("anthropic".to_string()),
                Just("bedrock".to_string()),
                Just("ollama".to_string()),
            ],
            auth_method in prop_oneof![
                Just(None::<String>),
                Just(Some("api_key".to_string())),
                Just(Some("oauth".to_string())),
            ],
        ) {
            // Build a YAML string WITHOUT the three new Codex fields
            let auth_line = match &auth_method {
                Some(m) => format!("auth_method: {m}\n"),
                None => String::new(),
            };
            let yaml = format!(
                "name: {name}\ntype: {provider_type}\n{auth_line}timeout_seconds: 30\nmax_connections: 100\n"
            );

            let provider: Provider = serde_yaml::from_str(&yaml).unwrap();
            prop_assert_eq!(provider.codex_base_url_override, None);
            prop_assert_eq!(provider.codex_model_override, None);
            prop_assert_eq!(provider.instructions_override, None);
        }
    }

    // Feature: guardrail-pipelines, Task 3.1: guardrail configuration validation
    // **Validates: Requirements 1.1, 1.2, 1.9, 1.10, 6.3, 6.6, 7.6, 8.7**

    use crate::guardrail::{
        FailurePolicy, GuardrailBindings, GuardrailConfig, GuardrailProviderConfig,
        GuardrailProviderType, InstructionInsertionMode, PipelineConfig, PolicyAction,
        ProviderSettings, RegexPatternConfig, RegexRuleMode, StageConfig, StagePhase,
    };

    fn regex_provider(name: &str) -> GuardrailProviderConfig {
        GuardrailProviderConfig {
            name: name.to_string(),
            provider_type: GuardrailProviderType::Regex,
            failure_policy: FailurePolicy::FailClose,
            timeout_seconds: 5,
            settings: ProviderSettings {
                patterns: vec![RegexPatternConfig {
                    name: "key".to_string(),
                    regex: "sk-[A-Za-z0-9]+".to_string(),
                    entity: "API_KEY".to_string(),
                    mode: RegexRuleMode::Deny,
                }],
                ..Default::default()
            },
        }
    }

    fn stage(name: &str, provider: &str) -> StageConfig {
        StageConfig {
            name: name.to_string(),
            provider: provider.to_string(),
            phase: StagePhase::PreCall,
            action: PolicyAction::Block,
        }
    }

    /// A minimal guardrail config: one regex provider, one single-stage pipeline.
    fn minimal_guardrails() -> GuardrailConfig {
        GuardrailConfig {
            providers: vec![regex_provider("scanner")],
            pipelines: vec![PipelineConfig {
                name: "standard".to_string(),
                stages: vec![stage("block-keys", "scanner")],
                redaction_notice_instruction: None,
                instruction_insertion_mode: InstructionInsertionMode::default(),
                failover_on_refusal: false,
                refusal_phrase_list: None,
                tool_result: crate::guardrail::config::ToolResultPhaseConfig::default(),
            }],
            global_default_pipeline: None,
            bindings: GuardrailBindings::default(),
            ..Default::default()
        }
    }

    fn config_with_guardrails(guardrails: GuardrailConfig) -> Config {
        let mut config = minimal_valid_config();
        config.guardrails = Some(guardrails);
        config
    }

    #[test]
    fn test_guardrail_valid_config_accepted() {
        let config = config_with_guardrails(minimal_guardrails());
        let result = config.validate();
        assert!(
            result.is_ok(),
            "Valid guardrail config should be accepted: {:?}",
            result
        );
    }

    #[test]
    fn test_guardrail_none_section_accepted() {
        // Absent guardrails section disables validation entirely.
        let config = minimal_valid_config();
        assert!(config.guardrails.is_none());
        assert!(config.validate().is_ok());
    }

    // Feature: indirect-injection-defense, task 1.3/1.4 — phase × action
    // matrix and unicode_stego threshold bounds.

    #[test]
    fn test_guardrail_matrix_rejects_replace_in_inbound_phases() {
        for phase in [StagePhase::PreCall, StagePhase::ToolResult] {
            let mut guardrails = minimal_guardrails();
            guardrails.pipelines[0].stages = vec![StageConfig {
                name: "bad".to_string(),
                provider: "scanner".to_string(),
                phase,
                action: PolicyAction::ReplaceWithPolicyMessage,
            }];
            let config = config_with_guardrails(guardrails);
            let errors = config.validate().unwrap_err();
            assert!(
                errors.iter().any(|e| matches!(
                    e,
                    ValidationError::GuardrailInvalidPhaseAction {
                        phase: p, ..
                    } if p == phase.as_str()
                )),
                "phase {phase:?} + replace_with_policy_message must be rejected: {errors:?}"
            );
        }
    }

    #[test]
    fn test_guardrail_matrix_accepts_all_actions_in_outbound_and_tool_call_phases() {
        for phase in [StagePhase::PostCall, StagePhase::ToolCall] {
            for action in [
                PolicyAction::Allow,
                PolicyAction::Block,
                PolicyAction::Mask,
                PolicyAction::Redact,
                PolicyAction::ReplaceWithPolicyMessage,
            ] {
                let mut guardrails = minimal_guardrails();
                guardrails.pipelines[0].stages = vec![StageConfig {
                    name: "ok".to_string(),
                    provider: "scanner".to_string(),
                    phase,
                    action,
                }];
                let config = config_with_guardrails(guardrails);
                assert!(
                    config.validate().is_ok(),
                    "phase {phase:?} + {action:?} must be accepted"
                );
            }
        }
    }

    #[test]
    fn test_guardrail_matrix_accepts_field_actions_in_inbound_phases() {
        // allow/block/mask/redact are valid in every phase (design §2.4 rows
        // 1–2).
        for phase in [StagePhase::PreCall, StagePhase::ToolResult] {
            for action in [
                PolicyAction::Allow,
                PolicyAction::Block,
                PolicyAction::Mask,
                PolicyAction::Redact,
            ] {
                let mut guardrails = minimal_guardrails();
                guardrails.pipelines[0].stages = vec![StageConfig {
                    name: "ok".to_string(),
                    provider: "scanner".to_string(),
                    phase,
                    action,
                }];
                let config = config_with_guardrails(guardrails);
                assert!(
                    config.validate().is_ok(),
                    "phase {phase:?} + {action:?} must be accepted"
                );
            }
        }
    }

    fn stego_provider(name: &str) -> GuardrailProviderConfig {
        GuardrailProviderConfig {
            name: name.to_string(),
            provider_type: GuardrailProviderType::UnicodeStego,
            failure_policy: FailurePolicy::FailOpen,
            timeout_seconds: 5,
            settings: ProviderSettings::default(),
        }
    }

    #[test]
    fn test_guardrail_stego_defaults_accepted() {
        let mut guardrails = minimal_guardrails();
        guardrails.providers.push(stego_provider("stego"));
        guardrails.pipelines[0].stages.push(StageConfig {
            name: "tool-result-scan".to_string(),
            provider: "stego".to_string(),
            phase: StagePhase::ToolResult,
            action: PolicyAction::Mask,
        });
        let config = config_with_guardrails(guardrails);
        assert!(config.validate().is_ok());
    }

    #[test]
    fn test_guardrail_stego_threshold_out_of_range_rejected() {
        for field in ["zero_width_threshold", "tag_chars_threshold", "bidi_threshold"] {
            let mut guardrails = minimal_guardrails();
            let mut provider = stego_provider("stego");
            let yaml = format!("{field}: 1001");
            provider.settings.unicode_stego =
                serde_yaml::from_str(&yaml).expect("valid stego settings");
            guardrails.providers.push(provider);
            let config = config_with_guardrails(guardrails);
            let errors = config.validate().unwrap_err();
            assert!(
                errors.iter().any(|e| matches!(
                    e,
                    ValidationError::GuardrailStegoThresholdOutOfRange {
                        field: f, value: 1001, ..
                    } if f == field
                )),
                "{field} = 1001 must be rejected: {errors:?}"
            );
        }
    }

    #[test]
    fn test_guardrail_stego_threshold_upper_bound_accepted() {
        let mut guardrails = minimal_guardrails();
        let mut provider = stego_provider("stego");
        provider.settings.unicode_stego = serde_yaml::from_str(
            "zero_width_threshold: 1000\ntag_chars_threshold: 1000\nbidi_threshold: 1000\n",
        )
        .expect("valid stego settings");
        guardrails.providers.push(provider);
        let config = config_with_guardrails(guardrails);
        assert!(config.validate().is_ok());
    }

    #[test]
    fn test_guardrail_empty_pipeline_rejected() {
        let mut guardrails = minimal_guardrails();
        guardrails.pipelines[0].stages.clear();
        let config = config_with_guardrails(guardrails);

        let errors = config.validate().unwrap_err();
        assert!(
            errors.iter().any(|e| matches!(
                e,
                ValidationError::GuardrailEmptyPipeline(name) if name == "standard"
            )),
            "Should reject a pipeline with no stages: {:?}",
            errors
        );
    }

    #[test]
    fn test_guardrail_duplicate_pipeline_rejected() {
        let mut guardrails = minimal_guardrails();
        guardrails.pipelines.push(PipelineConfig {
            name: "standard".to_string(),
            stages: vec![stage("dup", "scanner")],
            redaction_notice_instruction: None,
            instruction_insertion_mode: InstructionInsertionMode::default(),
            failover_on_refusal: false,
            refusal_phrase_list: None,
            tool_result: crate::guardrail::config::ToolResultPhaseConfig::default(),
        });
        let config = config_with_guardrails(guardrails);

        let errors = config.validate().unwrap_err();
        assert!(
            errors.iter().any(|e| matches!(
                e,
                ValidationError::GuardrailDuplicatePipeline(name) if name == "standard"
            )),
            "Should reject duplicate pipeline names: {:?}",
            errors
        );
    }

    #[test]
    fn test_guardrail_undeclared_provider_rejected_with_context() {
        let mut guardrails = minimal_guardrails();
        // Stage index 1 references an undeclared provider.
        guardrails.pipelines[0]
            .stages
            .push(stage("second", "missing-provider"));
        let config = config_with_guardrails(guardrails);

        let errors = config.validate().unwrap_err();
        assert!(
            errors.iter().any(|e| matches!(
                e,
                ValidationError::GuardrailUndeclaredProvider { pipeline, stage_index, provider }
                    if pipeline == "standard" && *stage_index == 1 && provider == "missing-provider"
            )),
            "Should identify pipeline, stage index, and provider name: {:?}",
            errors
        );
    }

    #[test]
    fn test_guardrail_undefined_binding_pipeline_rejected() {
        let mut guardrails = minimal_guardrails();
        guardrails
            .bindings
            .model_groups
            .insert("gpt-4o-group".to_string(), "nonexistent".to_string());
        let config = config_with_guardrails(guardrails);

        let errors = config.validate().unwrap_err();
        assert!(
            errors.iter().any(|e| matches!(
                e,
                ValidationError::GuardrailUndefinedBindingPipeline { target, pipeline }
                    if target == "model_groups.gpt-4o-group" && pipeline == "nonexistent"
            )),
            "Should identify binding target and undefined pipeline: {:?}",
            errors
        );
    }

    #[test]
    fn test_guardrail_binding_to_defined_pipeline_accepted() {
        let mut guardrails = minimal_guardrails();
        guardrails
            .bindings
            .routes
            .insert("/v1/chat/completions".to_string(), "standard".to_string());
        let config = config_with_guardrails(guardrails);
        assert!(config.validate().is_ok());
    }

    #[test]
    fn test_guardrail_undefined_global_default_rejected() {
        let mut guardrails = minimal_guardrails();
        guardrails.global_default_pipeline = Some("ghost".to_string());
        let config = config_with_guardrails(guardrails);

        let errors = config.validate().unwrap_err();
        assert!(
            errors.iter().any(|e| matches!(
                e,
                ValidationError::GuardrailUndefinedGlobalDefault(name) if name == "ghost"
            )),
            "Should reject global_default_pipeline referencing an undefined pipeline: {:?}",
            errors
        );
    }

    #[test]
    fn test_guardrail_defined_global_default_accepted() {
        let mut guardrails = minimal_guardrails();
        guardrails.global_default_pipeline = Some("standard".to_string());
        let config = config_with_guardrails(guardrails);
        assert!(config.validate().is_ok());
    }

    #[test]
    fn test_guardrail_presidio_empty_entities_rejected() {
        let mut guardrails = minimal_guardrails();
        guardrails.providers.push(GuardrailProviderConfig {
            name: "pii".to_string(),
            provider_type: GuardrailProviderType::Presidio,
            failure_policy: FailurePolicy::FailOpen,
            timeout_seconds: 5,
            settings: ProviderSettings {
                endpoint: Some("http://presidio:3000/analyze".to_string()),
                entities: vec![], // empty â†’ rejected (Req 6.3)
                ..Default::default()
            },
        });
        let config = config_with_guardrails(guardrails);

        let errors = config.validate().unwrap_err();
        assert!(
            errors.iter().any(|e| matches!(
                e,
                ValidationError::GuardrailPresidioNoEntities { provider } if provider == "pii"
            )),
            "Should reject presidio provider with empty entity list: {:?}",
            errors
        );
    }

    #[test]
    fn test_guardrail_confidence_threshold_out_of_range_rejected() {
        let mut guardrails = minimal_guardrails();
        guardrails.providers.push(GuardrailProviderConfig {
            name: "pii".to_string(),
            provider_type: GuardrailProviderType::Presidio,
            failure_policy: FailurePolicy::FailOpen,
            timeout_seconds: 5,
            settings: ProviderSettings {
                endpoint: Some("http://presidio:3000/analyze".to_string()),
                entities: vec!["EMAIL_ADDRESS".to_string()],
                confidence_threshold: Some(1.5), // out of range (Req 6.6)
                ..Default::default()
            },
        });
        let config = config_with_guardrails(guardrails);

        let errors = config.validate().unwrap_err();
        assert!(
            errors.iter().any(|e| matches!(
                e,
                ValidationError::GuardrailThresholdOutOfRange { provider, field, .. }
                    if provider == "pii" && field == "confidence_threshold"
            )),
            "Should reject confidence_threshold outside 0.0..=1.0: {:?}",
            errors
        );
    }

    #[test]
    fn test_guardrail_semantic_thresholds_out_of_range_rejected() {
        let mut guardrails = minimal_guardrails();
        guardrails.providers.push(GuardrailProviderConfig {
            name: "semantic".to_string(),
            provider_type: GuardrailProviderType::Semantic,
            failure_policy: FailurePolicy::FailOpen,
            timeout_seconds: 5,
            settings: ProviderSettings {
                allow_threshold: Some(-0.1), // out of range (Req 7.6)
                deny_threshold: Some(2.0),   // out of range (Req 7.6)
                ..Default::default()
            },
        });
        let config = config_with_guardrails(guardrails);

        let errors = config.validate().unwrap_err();
        let out_of_range: Vec<&str> = errors
            .iter()
            .filter_map(|e| match e {
                ValidationError::GuardrailThresholdOutOfRange { field, .. } => Some(field.as_str()),
                _ => None,
            })
            .collect();
        assert!(
            out_of_range.contains(&"allow_threshold") && out_of_range.contains(&"deny_threshold"),
            "Should reject both semantic thresholds outside 0.0..=1.0: {:?}",
            errors
        );
    }

    #[test]
    fn test_guardrail_threshold_at_bounds_accepted() {
        let mut guardrails = minimal_guardrails();
        guardrails.providers.push(GuardrailProviderConfig {
            name: "semantic".to_string(),
            provider_type: GuardrailProviderType::Semantic,
            failure_policy: FailurePolicy::FailOpen,
            timeout_seconds: 5,
            settings: ProviderSettings {
                allow_threshold: Some(0.0),
                deny_threshold: Some(1.0),
                ..Default::default()
            },
        });
        let config = config_with_guardrails(guardrails);
        assert!(
            config.validate().is_ok(),
            "Thresholds at 0.0 and 1.0 bounds should be accepted"
        );
    }

    #[test]
    fn test_guardrail_regex_pattern_cap_rejected() {
        let mut guardrails = minimal_guardrails();
        let patterns: Vec<RegexPatternConfig> = (0..257)
            .map(|i| RegexPatternConfig {
                name: format!("p{i}"),
                regex: "a".to_string(),
                entity: "X".to_string(),
                mode: RegexRuleMode::Deny,
            })
            .collect();
        guardrails.providers[0].settings.patterns = patterns;
        let config = config_with_guardrails(guardrails);

        let errors = config.validate().unwrap_err();
        assert!(
            errors.iter().any(|e| matches!(
                e,
                ValidationError::GuardrailTooManyPatterns { provider, count, max }
                    if provider == "scanner" && *count == 257 && *max == 256
            )),
            "Should reject regex provider exceeding 256 patterns: {:?}",
            errors
        );
    }

    #[test]
    fn test_guardrail_duplicate_provider_rejected() {
        let mut guardrails = minimal_guardrails();
        guardrails.providers.push(regex_provider("scanner"));
        let config = config_with_guardrails(guardrails);

        let errors = config.validate().unwrap_err();
        assert!(
            errors.iter().any(|e| matches!(
                e,
                ValidationError::GuardrailDuplicateProvider(name) if name == "scanner"
            )),
            "Should reject duplicate provider names: {:?}",
            errors
        );
    }

    // Feature: guardrail-pipelines, Task 16.3: Refusal_Phrase_List override validation
    // **Validates: Requirements 12.2, 12.13**

    #[test]
    fn test_guardrail_refusal_phrase_list_empty_entry_rejected() {
        let mut guardrails = minimal_guardrails();
        guardrails.pipelines[0].refusal_phrase_list = Some(vec![
            "i can't help".to_string(),
            "".to_string(), // empty â†’ rejected (Req 12.13)
        ]);
        let config = config_with_guardrails(guardrails);

        let errors = config.validate().unwrap_err();
        assert!(
            errors.iter().any(|e| matches!(
                e,
                ValidationError::GuardrailInvalidRefusalPhrase { pipeline_name, index, reason, .. }
                    if pipeline_name == "standard" && *index == 1 && reason == "empty refusal phrase"
            )),
            "Should reject empty refusal phrase entry: {:?}",
            errors
        );
    }

    #[test]
    fn test_guardrail_refusal_phrase_list_invalid_regex_rejected() {
        let mut guardrails = minimal_guardrails();
        guardrails.pipelines[0].refusal_phrase_list = Some(vec![
            "i can'?t (help|assist)".to_string(), // valid
            "(unclosed".to_string(),              // invalid regex â†’ rejected (Req 12.13)
        ]);
        let config = config_with_guardrails(guardrails);

        let errors = config.validate().unwrap_err();
        assert!(
            errors.iter().any(|e| matches!(
                e,
                ValidationError::GuardrailInvalidRefusalPhrase { pipeline_name, index, pattern, .. }
                    if pipeline_name == "standard" && *index == 1 && pattern == "(unclosed"
            )),
            "Should reject uncompilable regex refusal phrase entry: {:?}",
            errors
        );
    }

    #[test]
    fn test_guardrail_refusal_phrase_list_valid_entries_accepted() {
        let mut guardrails = minimal_guardrails();
        guardrails.pipelines[0].refusal_phrase_list = Some(vec![
            "i can'?t (help|assist) with".to_string(),
            "i'?m (sorry|unable)".to_string(),
            "as an ai".to_string(),
        ]);
        let config = config_with_guardrails(guardrails);
        assert!(
            config.validate().is_ok(),
            "Valid refusal phrase list should be accepted"
        );
    }

    #[test]
    fn test_guardrail_refusal_phrase_list_none_accepted() {
        // When refusal_phrase_list is None, default list is used â€” no validation needed.
        let guardrails = minimal_guardrails();
        assert!(guardrails.pipelines[0].refusal_phrase_list.is_none());
        let config = config_with_guardrails(guardrails);
        assert!(config.validate().is_ok());
    }

    // ---------------------------------------------------------------------
    // Feature: guardrail-pipelines, Task 3.2 â€” Property 3: Pipeline
    // configuration validation.
    // **Validates: Requirements 1.1, 1.2, 1.9, 1.10**
    //
    // For any generated guardrail configuration, validation succeeds *if and
    // only if* every pipeline name is unique, every pipeline has at least one
    // stage, every stage references a declared provider, and every binding
    // references a defined pipeline; otherwise it fails with an error that
    // identifies the offending pipeline name / stage index / provider name /
    // binding target.
    //
    // Generators are deliberately constrained so the ONLY possible validation
    // failures are the four Property-3 conditions: the three declared providers
    // are always unique and valid; pipeline names are always non-empty; there
    // is no global default; and no thresholds / entities / pattern caps are
    // exercised. This lets the test independently recompute expected validity
    // and match it exactly (both directions of the iff). `PolicyAction` and
    // `StagePhase` are enums, so an invalid action/phase is unrepresentable and
    // rejected at deserialization â€” validity is structurally guaranteed and not
    // separately generated here.

    /// Provider names declared by every generated config.
    const P3_DECLARED_PROVIDERS: [&str; 3] = ["p0", "p1", "p2"];
    /// Candidate provider references for stages; the last two are undeclared.
    const P3_PROVIDER_REFS: [&str; 5] = ["p0", "p1", "p2", "ghostA", "ghostB"];
    /// Candidate pipeline names (a small pool so duplicates arise naturally).
    const P3_PIPELINE_NAMES: [&str; 3] = ["pipeA", "pipeB", "pipeC"];
    /// Candidate binding pipeline references; `undefX` is never a defined name.
    const P3_BINDING_REFS: [&str; 4] = ["pipeA", "pipeB", "pipeC", "undefX"];

    #[derive(Debug, Clone)]
    struct P3Stage {
        provider_ref: String,
        action: PolicyAction,
        phase: StagePhase,
    }

    #[derive(Debug, Clone)]
    struct P3Pipeline {
        name: String,
        stages: Vec<P3Stage>,
    }

    #[derive(Debug, Clone)]
    struct P3Binding {
        kind: usize, // 0 = virtual_keys, 1 = model_groups, 2 = routes
        target: String,
        pipeline_ref: String,
    }

    fn p3_arb_action() -> impl Strategy<Value = PolicyAction> {
        prop::sample::select(vec![
            PolicyAction::Allow,
            PolicyAction::Block,
            PolicyAction::Mask,
            PolicyAction::Redact,
            PolicyAction::ReplaceWithPolicyMessage,
        ])
    }

    fn p3_arb_phase() -> impl Strategy<Value = StagePhase> {
        prop::sample::select(vec![
            StagePhase::PreCall,
            StagePhase::PostCall,
            StagePhase::ToolResult,
            StagePhase::ToolCall,
        ])
    }

    fn p3_arb_stage() -> impl Strategy<Value = P3Stage> {
        (
            0usize..P3_PROVIDER_REFS.len(),
            p3_arb_action(),
            p3_arb_phase(),
        )
            .prop_map(|(pidx, action, phase)| P3Stage {
                provider_ref: P3_PROVIDER_REFS[pidx].to_string(),
                action,
                phase,
            })
    }

    fn p3_arb_pipeline() -> impl Strategy<Value = P3Pipeline> {
        (
            0usize..P3_PIPELINE_NAMES.len(),
            prop::collection::vec(p3_arb_stage(), 0..=3),
        )
            .prop_map(|(nidx, stages)| P3Pipeline {
                name: P3_PIPELINE_NAMES[nidx].to_string(),
                stages,
            })
    }

    fn p3_arb_binding() -> impl Strategy<Value = P3Binding> {
        (0usize..3, 0usize..4, 0usize..P3_BINDING_REFS.len()).prop_map(|(kind, tidx, ridx)| {
            P3Binding {
                kind,
                target: format!("t{tidx}"),
                pipeline_ref: P3_BINDING_REFS[ridx].to_string(),
            }
        })
    }

    proptest! {
        #![proptest_config(ProptestConfig::with_cases(64))]

        #[test]
        fn prop_guardrail_config_validation_iff(
            gen_pipelines in prop::collection::vec(p3_arb_pipeline(), 1..=4),
            gen_bindings in prop::collection::vec(p3_arb_binding(), 0..=4),
        ) {
            use std::collections::{HashMap, HashSet};

            // --- Build the guardrail config from the generated data. ---
            let providers: Vec<GuardrailProviderConfig> =
                P3_DECLARED_PROVIDERS.iter().map(|n| regex_provider(n)).collect();

            let pipeline_configs: Vec<PipelineConfig> = gen_pipelines
                .iter()
                .map(|p| PipelineConfig {
                    name: p.name.clone(),
                    stages: p
                        .stages
                        .iter()
                        .enumerate()
                        .map(|(i, s)| StageConfig {
                            name: format!("s{i}"),
                            provider: s.provider_ref.clone(),
                            phase: s.phase,
                            action: s.action,
                        })
                        .collect(),
                    redaction_notice_instruction: None,
                    instruction_insertion_mode: InstructionInsertionMode::default(),
                    failover_on_refusal: false,
                    refusal_phrase_list: None,
                    tool_result: crate::guardrail::config::ToolResultPhaseConfig::default(),
                })
                .collect();

            let mut binding_cfg = GuardrailBindings::default();
            for b in &gen_bindings {
                let map = match b.kind {
                    0 => &mut binding_cfg.virtual_keys,
                    1 => &mut binding_cfg.model_groups,
                    _ => &mut binding_cfg.routes,
                };
                map.insert(b.target.clone(), b.pipeline_ref.clone());
            }

            let guardrails = GuardrailConfig {
                providers,
                pipelines: pipeline_configs,
                global_default_pipeline: None,
                bindings: binding_cfg.clone(),
                ..Default::default()
            };
            let config = config_with_guardrails(guardrails);

            // --- Independently recompute expected validity + offending items. ---
            let declared: HashSet<&str> = P3_DECLARED_PROVIDERS.iter().copied().collect();
            let defined: HashSet<&str> =
                gen_pipelines.iter().map(|p| p.name.as_str()).collect();

            // Req 1.1: duplicate pipeline names.
            let mut seen: HashSet<&str> = HashSet::new();
            let mut dup_names: HashSet<&str> = HashSet::new();
            for p in &gen_pipelines {
                if !seen.insert(p.name.as_str()) {
                    dup_names.insert(p.name.as_str());
                }
            }

            // Req 1.1: pipelines must contain at least one stage.
            let empty_names: HashSet<&str> = gen_pipelines
                .iter()
                .filter(|p| p.stages.is_empty())
                .map(|p| p.name.as_str())
                .collect();

            // Req 1.2 / 1.9: stages referencing an undeclared provider,
            // identified by (pipeline name, stage index, provider name).
            let mut undeclared: Vec<(String, usize, String)> = Vec::new();
            for p in &gen_pipelines {
                for (i, s) in p.stages.iter().enumerate() {
                    if !declared.contains(s.provider_ref.as_str()) {
                        undeclared.push((p.name.clone(), i, s.provider_ref.clone()));
                    }
                }
            }

            // Req 1.10: bindings referencing an undefined pipeline, identified
            // by (binding target, pipeline name).
            let mut undefined_bindings: Vec<(String, String)> = Vec::new();
            let binding_kinds: [(&str, &HashMap<String, String>); 3] = [
                ("virtual_keys", &binding_cfg.virtual_keys),
                ("model_groups", &binding_cfg.model_groups),
                ("routes", &binding_cfg.routes),
            ];
            for (kind, map) in binding_kinds {
                for (target, pref) in map {
                    if !defined.contains(pref.as_str()) {
                        undefined_bindings.push((format!("{kind}.{target}"), pref.clone()));
                    }
                }
            }

            // Design §2.4: phase × action matrix — `replace_with_policy_message`
            // is invalid for the two inbound phases.
            let matrix_violations: Vec<(&str, usize)> = gen_pipelines
                .iter()
                .flat_map(|p| p.stages.iter().enumerate().map(move |(i, s)| (p, i, s)))
                .filter(|(_, _, s)| {
                    matches!(
                        (s.phase, s.action),
                        (
                            StagePhase::PreCall | StagePhase::ToolResult,
                            PolicyAction::ReplaceWithPolicyMessage
                        )
                    )
                })
                .map(|(p, i, _)| (p.name.as_str(), i))
                .collect();

            let expected_valid = dup_names.is_empty()
                && empty_names.is_empty()
                && undeclared.is_empty()
                && undefined_bindings.is_empty()
                && matrix_violations.is_empty();

            // --- Property: validation succeeds iff no offending condition. ---
            let result = config.validate();
            prop_assert_eq!(
                result.is_ok(),
                expected_valid,
                "validate().is_ok() must equal expected_valid; errors: {:?}",
                result.as_ref().err()
            );

            // --- When invalid, each offending item is identified with context. ---
            if let Err(errors) = result {
                for name in &dup_names {
                    prop_assert!(
                        errors.iter().any(|e| matches!(
                            e,
                            ValidationError::GuardrailDuplicatePipeline(n) if n == name
                        )),
                        "missing GuardrailDuplicatePipeline for {:?} in {:?}",
                        name, errors
                    );
                }
                for name in &empty_names {
                    prop_assert!(
                        errors.iter().any(|e| matches!(
                            e,
                            ValidationError::GuardrailEmptyPipeline(n) if n == name
                        )),
                        "missing GuardrailEmptyPipeline for {:?} in {:?}",
                        name, errors
                    );
                }
                for (pname, sidx, prov) in &undeclared {
                    prop_assert!(
                        errors.iter().any(|e| matches!(
                            e,
                            ValidationError::GuardrailUndeclaredProvider { pipeline, stage_index, provider }
                                if pipeline == pname && stage_index == sidx && provider == prov
                        )),
                        "missing GuardrailUndeclaredProvider for ({:?}, {}, {:?}) in {:?}",
                        pname, sidx, prov, errors
                    );
                }
                for (target, pref) in &undefined_bindings {
                    prop_assert!(
                        errors.iter().any(|e| matches!(
                            e,
                            ValidationError::GuardrailUndefinedBindingPipeline { target: t, pipeline }
                                if t == target && pipeline == pref
                        )),
                        "missing GuardrailUndefinedBindingPipeline for ({:?}, {:?}) in {:?}",
                        target, pref, errors
                    );
                }
            }
        }
    }

    // -------------------------------------------------------------------------
    // Feature: guardrail-pipelines, Task 16.6 â€” Property 34: Refusal
    // phrase-list validation rejects malformed entries.
    // **Validates: Requirements 12.13**
    //
    // For any generated refusal_phrase_list, validation rejects the config iff
    // the list contains an empty entry OR an uncompilable regex entry, and
    // identifies the offending entry via GuardrailInvalidRefusalPhrase. A valid,
    // non-empty list of compilable regex/phrase entries passes validation.

    /// Generate a string that is a valid regex when compiled case-insensitively.
    /// Uses a fixed pool of known-valid patterns to avoid slow filtering.
    fn arb_valid_regex_phrase() -> impl Strategy<Value = String> {
        prop::sample::select(vec![
            "i can'?t (help|assist) with".to_string(),
            "i'?m (sorry|unable)".to_string(),
            "as an ai".to_string(),
            "i must decline".to_string(),
            "i cannot comply".to_string(),
            "not able to".to_string(),
            "refuse to".to_string(),
            "will not help".to_string(),
            "against my (policy|guidelines)".to_string(),
            "harmful content".to_string(),
        ])
    }

    /// Generate a string that is NOT compilable as a regex.
    fn arb_invalid_regex() -> impl Strategy<Value = String> {
        // Unbalanced parens/brackets produce regex compile errors.
        prop::sample::select(vec![
            "[invalid".to_string(),
            "(unclosed".to_string(),
            "(?P<bad".to_string(),
            "[z-a]".to_string(),
            "***".to_string(),
            "+start".to_string(),
            "(?:".to_string(),
        ])
    }

    /// The three categories an entry can fall into.
    #[derive(Debug, Clone)]
    enum PhraseEntry {
        Valid(String),
        Empty,
        InvalidRegex(String),
    }

    fn arb_phrase_entry() -> impl Strategy<Value = PhraseEntry> {
        prop_oneof![
            3 => arb_valid_regex_phrase().prop_map(PhraseEntry::Valid),
            1 => Just(PhraseEntry::Empty),
            1 => arb_invalid_regex().prop_map(PhraseEntry::InvalidRegex),
        ]
    }

    proptest! {
        #![proptest_config(ProptestConfig::with_cases(64))]

        #[test]
        fn prop_refusal_phrase_list_validation_rejects_malformed(
            entries in prop::collection::vec(arb_phrase_entry(), 1..=8),
        ) {
            // Build refusal_phrase_list from generated entries.
            let phrase_list: Vec<String> = entries
                .iter()
                .map(|e| match e {
                    PhraseEntry::Valid(s) => s.clone(),
                    PhraseEntry::Empty => String::new(),
                    PhraseEntry::InvalidRegex(s) => s.clone(),
                })
                .collect();

            let mut guardrails = minimal_guardrails();
            guardrails.pipelines[0].refusal_phrase_list = Some(phrase_list.clone());
            let config = config_with_guardrails(guardrails);

            // Independently compute which entries are malformed.
            let malformed_indices: Vec<(usize, &str)> = entries
                .iter()
                .enumerate()
                .filter_map(|(i, e)| match e {
                    PhraseEntry::Empty => Some((i, "empty")),
                    PhraseEntry::InvalidRegex(_) => Some((i, "regex_error")),
                    PhraseEntry::Valid(_) => None,
                })
                .collect();

            let expected_valid = malformed_indices.is_empty();
            let result = config.validate();

            // Property: validation passes iff no malformed entry exists.
            prop_assert_eq!(
                result.is_ok(),
                expected_valid,
                "validate().is_ok() = {} but expected_valid = {}; phrase_list: {:?}",
                result.is_ok(),
                expected_valid,
                phrase_list
            );

            // When invalid, each malformed entry triggers GuardrailInvalidRefusalPhrase.
            if let Err(ref errors) = result {
                for (idx, kind) in &malformed_indices {
                    let found = errors.iter().any(|e| match e {
                        ValidationError::GuardrailInvalidRefusalPhrase {
                            pipeline_name,
                            index,
                            pattern,
                            reason,
                        } => {
                            pipeline_name == "standard"
                                && *index == *idx
                                && pattern == &phrase_list[*idx]
                                && match *kind {
                                    "empty" => reason.contains("empty"),
                                    _ => !reason.is_empty(), // regex error has non-empty reason
                                }
                        }
                        _ => false,
                    });
                    prop_assert!(
                        found,
                        "missing GuardrailInvalidRefusalPhrase for index {} ({:?}) in {:?}",
                        idx,
                        kind,
                        errors
                    );
                }
            }
        }
    }
}
