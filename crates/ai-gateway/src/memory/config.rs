//! Persistent-memory configuration and field-level validation.

use std::collections::HashSet;
use std::fmt;

use serde::{Deserialize, Serialize};

use super::InjectionStrategy;

const MAX_DATABASE_PATH_CHARS: usize = 4_096;
const MAX_INJECTION_TOKENS: u32 = 10_000;
const MAX_AUTO_EXTRACT_MIN_TURNS: u32 = 100;
const MAX_DECAY_SCHEDULE_HOURS: u32 = 8_760;
const MAX_MEMORIES_PER_NAMESPACE: u32 = 100_000;
const MAX_QDRANT_COLLECTION_CHARS: usize = 255;

/// Global persistent-memory settings from the top-level `memory` section.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(default, deny_unknown_fields)]
pub struct MemoryConfig {
    pub enabled: bool,
    pub database_path: String,
    pub injection_strategy: InjectionStrategy,
    pub max_injection_tokens: u32,
    pub auto_extract_enabled: bool,
    pub auto_extract_provider: String,
    pub auto_extract_model: String,
    pub auto_extract_min_turns: u32,
    pub decay_schedule_hours: u32,
    pub max_memories_per_namespace: u32,
    pub allow_sensitive_storage: bool,
    pub show_feedback: bool,
    pub qdrant: Option<MemoryQdrantConfig>,
    pub default_prompts: Vec<String>,
    pub custom_sensitive_patterns: Vec<String>,
}

impl Default for MemoryConfig {
    fn default() -> Self {
        Self {
            enabled: false,
            database_path: "./memory.db".to_owned(),
            injection_strategy: InjectionStrategy::SystemPromptPrefix,
            max_injection_tokens: 500,
            auto_extract_enabled: false,
            auto_extract_provider: String::new(),
            auto_extract_model: String::new(),
            auto_extract_min_turns: 4,
            decay_schedule_hours: 24,
            max_memories_per_namespace: 1_000,
            allow_sensitive_storage: false,
            show_feedback: true,
            qdrant: None,
            default_prompts: Vec::new(),
            custom_sensitive_patterns: Vec::new(),
        }
    }
}

impl MemoryConfig {
    /// Resolves the request-scoped memory settings one field at a time.
    ///
    /// Model-group values have precedence over provider values, which have
    /// precedence over the global configuration. Explicit `false` and zero token
    /// values are preserved at every override level.
    pub fn resolve(
        &self,
        provider: Option<&ProviderMemoryOverride>,
        model_group: Option<&ModelGroupMemoryOverride>,
    ) -> EffectiveMemoryConfig {
        EffectiveMemoryConfig {
            enabled: model_group
                .and_then(|config| config.enabled)
                .or_else(|| provider.and_then(|config| config.enabled))
                .unwrap_or(self.enabled),
            injection_strategy: model_group
                .and_then(|config| config.injection_strategy)
                .or_else(|| provider.and_then(|config| config.injection_strategy))
                .unwrap_or(self.injection_strategy),
            max_injection_tokens: model_group
                .and_then(|config| config.max_injection_tokens)
                .or_else(|| provider.and_then(|config| config.max_injection_tokens))
                .unwrap_or(self.max_injection_tokens),
            show_feedback: model_group
                .and_then(|config| config.show_feedback)
                .or_else(|| provider.and_then(|config| config.show_feedback))
                .unwrap_or(self.show_feedback),
        }
    }

    /// Validates all local memory fields and cross-field dependencies.
    ///
    /// Provider existence is intentionally checked separately by
    /// [`Self::validate_with_provider_names`] once the parent configuration has
    /// access to the complete provider list.
    pub fn validate(&self) -> MemoryValidationResult<()> {
        let mut errors = Vec::new();

        validate_database_path(&self.database_path, &mut errors);
        validate_u32_range(
            "max_injection_tokens",
            self.max_injection_tokens,
            0,
            MAX_INJECTION_TOKENS,
            &mut errors,
        );
        validate_u32_range(
            "auto_extract_min_turns",
            self.auto_extract_min_turns,
            1,
            MAX_AUTO_EXTRACT_MIN_TURNS,
            &mut errors,
        );
        validate_u32_range(
            "decay_schedule_hours",
            self.decay_schedule_hours,
            1,
            MAX_DECAY_SCHEDULE_HOURS,
            &mut errors,
        );
        validate_u32_range(
            "max_memories_per_namespace",
            self.max_memories_per_namespace,
            1,
            MAX_MEMORIES_PER_NAMESPACE,
            &mut errors,
        );

        if self.auto_extract_enabled {
            validate_nonempty(
                "auto_extract_provider",
                &self.auto_extract_provider,
                "must not be empty when auto_extract_enabled is true",
                &mut errors,
            );
            validate_nonempty(
                "auto_extract_model",
                &self.auto_extract_model,
                "must not be empty when auto_extract_enabled is true",
                &mut errors,
            );
        }

        if let Some(qdrant) = &self.qdrant {
            qdrant.validate_into(&mut errors);
        }

        validation_result(errors)
    }

    /// Validates local fields and resolves the auto-extraction provider name.
    ///
    /// The Wave 1 parent integration should pass every configured provider name
    /// from the top-level configuration into this method.
    pub fn validate_with_provider_names<'a, I>(
        &self,
        provider_names: I,
    ) -> MemoryValidationResult<()>
    where
        I: IntoIterator<Item = &'a str>,
    {
        let mut errors = self.validate().err().unwrap_or_default();

        if self.auto_extract_enabled && !self.auto_extract_provider.trim().is_empty() {
            let configured: HashSet<&str> = provider_names.into_iter().collect();
            if !configured.contains(self.auto_extract_provider.trim()) {
                errors.push(MemoryConfigError::new(
                    "auto_extract_provider",
                    format!(
                        "has unresolved value {:?}; expected the name of a configured provider",
                        self.auto_extract_provider
                    ),
                ));
            }
        }

        validation_result(errors)
    }
}

/// Optional Qdrant vector retrieval settings.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(default, deny_unknown_fields)]
pub struct MemoryQdrantConfig {
    pub qdrant_url: String,
    pub qdrant_collection: String,
    pub similarity_threshold: f32,
    pub embedding_provider: String,
    pub embedding_model: String,
    pub fts_weight: f32,
    pub vector_weight: f32,
    /// Manual override for the embedding vector dimension.
    /// When set, bypasses both the lookup table and the probe-on-first-store
    /// logic, using this value directly when creating or validating a Qdrant
    /// collection. Existing configs without this field deserialize as `None`.
    pub vector_dimension: Option<u64>,
}

impl Default for MemoryQdrantConfig {
    fn default() -> Self {
        Self {
            qdrant_url: String::new(),
            qdrant_collection: "obey_memories".to_owned(),
            similarity_threshold: 0.7,
            embedding_provider: String::new(),
            embedding_model: String::new(),
            fts_weight: 0.4,
            vector_weight: 0.6,
            vector_dimension: None,
        }
    }
}

impl MemoryQdrantConfig {
    /// Validates URL, collection, threshold, embedding, and weight fields.
    pub fn validate(&self) -> MemoryValidationResult<()> {
        let mut errors = Vec::new();
        self.validate_into(&mut errors);
        validation_result(errors)
    }

    fn validate_into(&self, errors: &mut Vec<MemoryConfigError>) {
        validate_http_url("qdrant.qdrant_url", &self.qdrant_url, errors);
        validate_char_range(
            "qdrant.qdrant_collection",
            &self.qdrant_collection,
            1,
            MAX_QDRANT_COLLECTION_CHARS,
            errors,
        );
        validate_f32_range(
            "qdrant.similarity_threshold",
            self.similarity_threshold,
            0.0,
            1.0,
            errors,
        );
        validate_nonempty(
            "qdrant.embedding_provider",
            &self.embedding_provider,
            "must not be empty when Qdrant is configured",
            errors,
        );
        validate_nonempty(
            "qdrant.embedding_model",
            &self.embedding_model,
            "must not be empty when Qdrant is configured",
            errors,
        );
        validate_weight("qdrant.fts_weight", self.fts_weight, errors);
        validate_weight("qdrant.vector_weight", self.vector_weight, errors);

        let weight_sum = self.fts_weight + self.vector_weight;
        if self.fts_weight.is_finite() && self.vector_weight.is_finite() && weight_sum <= 0.0 {
            errors.push(MemoryConfigError::new(
                "qdrant.fts_weight/qdrant.vector_weight",
                format!(
                    "sum is {weight_sum}; expected finite non-negative weights with a sum greater than 0"
                ),
            ));
        }

        if let Some(d) = self.vector_dimension {
            if d == 0 || d > 65536 {
                errors.push(MemoryConfigError::new(
                    "qdrant.vector_dimension",
                    format!("has value {d}; expected a value in 1..=65536"),
                ));
            }
        }
    }
}

/// Provider-scoped fields that may override global memory behavior.
#[derive(Debug, Clone, Default, PartialEq, Serialize, Deserialize)]
#[serde(default, deny_unknown_fields)]
pub struct ProviderMemoryOverride {
    pub enabled: Option<bool>,
    pub injection_strategy: Option<InjectionStrategy>,
    pub max_injection_tokens: Option<u32>,
    pub show_feedback: Option<bool>,
}

impl ProviderMemoryOverride {
    pub fn validate(&self) -> MemoryValidationResult<()> {
        validate_override_tokens(self.max_injection_tokens)
    }
}

/// Model-group-scoped fields that may override provider or global behavior.
#[derive(Debug, Clone, Default, PartialEq, Serialize, Deserialize)]
#[serde(default, deny_unknown_fields)]
pub struct ModelGroupMemoryOverride {
    pub enabled: Option<bool>,
    pub injection_strategy: Option<InjectionStrategy>,
    pub max_injection_tokens: Option<u32>,
    pub show_feedback: Option<bool>,
}

impl ModelGroupMemoryOverride {
    pub fn validate(&self) -> MemoryValidationResult<()> {
        validate_override_tokens(self.max_injection_tokens)
    }
}

/// Fully resolved memory settings for one request.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct EffectiveMemoryConfig {
    pub enabled: bool,
    pub injection_strategy: InjectionStrategy,
    pub max_injection_tokens: u32,
    pub show_feedback: bool,
}

/// One structured, field-addressable memory configuration failure.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct MemoryConfigError {
    pub field: String,
    pub message: String,
}

impl MemoryConfigError {
    pub fn new(field: impl Into<String>, message: impl Into<String>) -> Self {
        Self {
            field: field.into(),
            message: message.into(),
        }
    }
}

impl fmt::Display for MemoryConfigError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(formatter, "invalid memory.{}: {}", self.field, self.message)
    }
}

impl std::error::Error for MemoryConfigError {}

/// Aggregated result returned by memory configuration validation.
pub type MemoryValidationResult<T> = Result<T, Vec<MemoryConfigError>>;

fn validate_override_tokens(value: Option<u32>) -> MemoryValidationResult<()> {
    let mut errors = Vec::new();
    if let Some(value) = value {
        validate_u32_range(
            "max_injection_tokens",
            value,
            0,
            MAX_INJECTION_TOKENS,
            &mut errors,
        );
    }
    validation_result(errors)
}

fn validate_database_path(value: &str, errors: &mut Vec<MemoryConfigError>) {
    validate_nonempty("database_path", value, "must not be empty", errors);
    if value.contains('\0') {
        errors.push(MemoryConfigError::new(
            "database_path",
            format!(
                "contains a NUL character in {value:?}; expected a path without NUL characters"
            ),
        ));
    }
    validate_max_chars("database_path", value, MAX_DATABASE_PATH_CHARS, errors);
}

fn validate_nonempty(
    field: &str,
    value: &str,
    constraint: &str,
    errors: &mut Vec<MemoryConfigError>,
) {
    if value.trim().is_empty() {
        errors.push(MemoryConfigError::new(
            field,
            format!("has value {value:?}; {constraint}"),
        ));
    }
}

fn validate_max_chars(field: &str, value: &str, max: usize, errors: &mut Vec<MemoryConfigError>) {
    let length = value.chars().count();
    if length > max {
        errors.push(MemoryConfigError::new(
            field,
            format!("has {length} characters; expected at most {max} characters"),
        ));
    }
}

fn validate_char_range(
    field: &str,
    value: &str,
    min: usize,
    max: usize,
    errors: &mut Vec<MemoryConfigError>,
) {
    let length = value.chars().count();
    if !(min..=max).contains(&length) {
        errors.push(MemoryConfigError::new(
            field,
            format!("has value {value:?} ({length} characters); expected {min}..={max} characters"),
        ));
    }
}

fn validate_u32_range(
    field: &str,
    value: u32,
    min: u32,
    max: u32,
    errors: &mut Vec<MemoryConfigError>,
) {
    if !(min..=max).contains(&value) {
        errors.push(MemoryConfigError::new(
            field,
            format!("has value {value}; expected a value in {min}..={max}"),
        ));
    }
}

fn validate_f32_range(
    field: &str,
    value: f32,
    min: f32,
    max: f32,
    errors: &mut Vec<MemoryConfigError>,
) {
    if !value.is_finite() || !(min..=max).contains(&value) {
        errors.push(MemoryConfigError::new(
            field,
            format!("has value {value}; expected a finite value in {min}..={max}"),
        ));
    }
}

fn validate_weight(field: &str, value: f32, errors: &mut Vec<MemoryConfigError>) {
    if !value.is_finite() || value < 0.0 {
        errors.push(MemoryConfigError::new(
            field,
            format!("has value {value}; expected a finite non-negative value"),
        ));
    }
}

fn validate_http_url(field: &str, value: &str, errors: &mut Vec<MemoryConfigError>) {
    let remainder = value
        .strip_prefix("http://")
        .or_else(|| value.strip_prefix("https://"));
    let Some(remainder) = remainder else {
        errors.push(MemoryConfigError::new(
            field,
            format!("has value {value:?}; expected an absolute HTTP or HTTPS URL"),
        ));
        return;
    };

    let authority = remainder.split(['/', '?', '#']).next().unwrap_or_default();
    let invalid_characters = authority
        .chars()
        .any(|character| character.is_whitespace() || character.is_control());
    let has_credentials = authority.contains('@');
    let host = authority_host(authority);

    if authority.is_empty() || host.is_empty() || invalid_characters || has_credentials {
        errors.push(MemoryConfigError::new(
            field,
            format!(
                "has value {value:?}; expected an HTTP or HTTPS URL with a host and no credentials"
            ),
        ));
    }
}

fn authority_host(authority: &str) -> &str {
    if authority.starts_with('[') {
        return authority
            .find(']')
            .map(|end| &authority[1..end])
            .unwrap_or_default();
    }
    authority.split(':').next().unwrap_or_default()
}

fn validation_result(errors: Vec<MemoryConfigError>) -> MemoryValidationResult<()> {
    if errors.is_empty() {
        Ok(())
    } else {
        Err(errors)
    }
}

#[cfg(test)]
mod tests {
    use proptest::prelude::*;

    use super::*;

    fn has_error(errors: &[MemoryConfigError], field: &str, message: &str) -> bool {
        errors
            .iter()
            .any(|error| error.field == field && error.message.contains(message))
    }

    fn qdrant_config() -> MemoryQdrantConfig {
        MemoryQdrantConfig {
            qdrant_url: "https://qdrant.example.com:6333".to_owned(),
            embedding_provider: "embeddings".to_owned(),
            embedding_model: "text-embedding-3-small".to_owned(),
            ..MemoryQdrantConfig::default()
        }
    }

    proptest! {
        #![proptest_config(ProptestConfig::with_cases(128))]

        #[test]
        fn property_configuration_precedence_is_field_local(
            global_enabled in any::<bool>(),
            provider_enabled in proptest::option::of(any::<bool>()),
            model_enabled in proptest::option::of(any::<bool>()),
            global_strategy_synthetic in any::<bool>(),
            provider_strategy_synthetic in proptest::option::of(any::<bool>()),
            model_strategy_synthetic in proptest::option::of(any::<bool>()),
            global_tokens in 0u32..=MAX_INJECTION_TOKENS,
            provider_tokens in proptest::option::of(0u32..=MAX_INJECTION_TOKENS),
            model_tokens in proptest::option::of(0u32..=MAX_INJECTION_TOKENS),
            global_feedback in any::<bool>(),
            provider_feedback in proptest::option::of(any::<bool>()),
            model_feedback in proptest::option::of(any::<bool>()),
        ) {
            let strategy = |synthetic| if synthetic {
                InjectionStrategy::SyntheticMessage
            } else {
                InjectionStrategy::SystemPromptPrefix
            };
            let global = MemoryConfig {
                enabled: global_enabled,
                injection_strategy: strategy(global_strategy_synthetic),
                max_injection_tokens: global_tokens,
                show_feedback: global_feedback,
                ..MemoryConfig::default()
            };
            let provider = ProviderMemoryOverride {
                enabled: provider_enabled,
                injection_strategy: provider_strategy_synthetic.map(strategy),
                max_injection_tokens: provider_tokens,
                show_feedback: provider_feedback,
            };
            let model = ModelGroupMemoryOverride {
                enabled: model_enabled,
                injection_strategy: model_strategy_synthetic.map(strategy),
                max_injection_tokens: model_tokens,
                show_feedback: model_feedback,
            };

            let effective = global.resolve(Some(&provider), Some(&model));

            prop_assert_eq!(
                effective.enabled,
                model_enabled.or(provider_enabled).unwrap_or(global_enabled)
            );
            prop_assert_eq!(
                effective.injection_strategy,
                model_strategy_synthetic
                    .or(provider_strategy_synthetic)
                    .map(strategy)
                    .unwrap_or_else(|| strategy(global_strategy_synthetic))
            );
            prop_assert_eq!(
                effective.max_injection_tokens,
                model_tokens.or(provider_tokens).unwrap_or(global_tokens)
            );
            prop_assert_eq!(
                effective.show_feedback,
                model_feedback.or(provider_feedback).unwrap_or(global_feedback)
            );
        }
    }

    #[test]
    fn serde_defaults_match_documented_values() {
        let config: MemoryConfig = serde_yaml::from_str("{}").unwrap();

        assert_eq!(config, MemoryConfig::default());
        assert!(!config.enabled);
        assert_eq!(config.database_path, "./memory.db");
        assert_eq!(
            config.injection_strategy,
            InjectionStrategy::SystemPromptPrefix
        );
        assert_eq!(config.max_injection_tokens, 500);
        assert!(!config.auto_extract_enabled);
        assert!(config.auto_extract_provider.is_empty());
        assert!(config.auto_extract_model.is_empty());
        assert_eq!(config.auto_extract_min_turns, 4);
        assert_eq!(config.decay_schedule_hours, 24);
        assert_eq!(config.max_memories_per_namespace, 1_000);
        assert!(!config.allow_sensitive_storage);
        assert!(config.show_feedback);
        assert!(config.qdrant.is_none());
        assert!(config.default_prompts.is_empty());
        assert!(config.custom_sensitive_patterns.is_empty());
        config.validate().unwrap();
    }

    #[test]
    fn ranges_are_inclusive_and_zero_tokens_disable_without_invalidating() {
        for tokens in [0, MAX_INJECTION_TOKENS] {
            let config = MemoryConfig {
                max_injection_tokens: tokens,
                auto_extract_min_turns: if tokens == 0 { 1 } else { 100 },
                decay_schedule_hours: if tokens == 0 { 1 } else { 8_760 },
                max_memories_per_namespace: if tokens == 0 { 1 } else { 100_000 },
                ..MemoryConfig::default()
            };
            config.validate().unwrap();
        }

        for (field, config) in [
            (
                "max_injection_tokens",
                MemoryConfig {
                    max_injection_tokens: 10_001,
                    ..MemoryConfig::default()
                },
            ),
            (
                "auto_extract_min_turns",
                MemoryConfig {
                    auto_extract_min_turns: 0,
                    ..MemoryConfig::default()
                },
            ),
            (
                "decay_schedule_hours",
                MemoryConfig {
                    decay_schedule_hours: 8_761,
                    ..MemoryConfig::default()
                },
            ),
            (
                "max_memories_per_namespace",
                MemoryConfig {
                    max_memories_per_namespace: 100_001,
                    ..MemoryConfig::default()
                },
            ),
        ] {
            assert!(has_error(
                &config.validate().unwrap_err(),
                field,
                "expected"
            ));
        }

        assert_eq!(
            ProviderMemoryOverride {
                max_injection_tokens: Some(0),
                ..ProviderMemoryOverride::default()
            }
            .validate(),
            Ok(())
        );
        assert!(ModelGroupMemoryOverride {
            max_injection_tokens: Some(10_001),
            ..ModelGroupMemoryOverride::default()
        }
        .validate()
        .is_err());
    }

    #[test]
    fn database_path_constraints_are_aggregated() {
        let config = MemoryConfig {
            database_path: format!("{}\0", "x".repeat(MAX_DATABASE_PATH_CHARS)),
            ..MemoryConfig::default()
        };
        let errors = config.validate().unwrap_err();

        assert!(has_error(&errors, "database_path", "NUL"));
        assert!(has_error(&errors, "database_path", "at most 4096"));
        assert!(MemoryConfig {
            database_path: " ".to_owned(),
            ..MemoryConfig::default()
        }
        .validate()
        .is_err());
    }

    #[test]
    fn auto_extract_requires_provider_model_and_resolved_provider() {
        let config = MemoryConfig {
            auto_extract_enabled: true,
            auto_extract_provider: " ".to_owned(),
            auto_extract_model: String::new(),
            ..MemoryConfig::default()
        };
        let errors = config.validate().unwrap_err();
        assert!(has_error(
            &errors,
            "auto_extract_provider",
            "enabled is true"
        ));
        assert!(has_error(&errors, "auto_extract_model", "enabled is true"));

        let config = MemoryConfig {
            auto_extract_enabled: true,
            auto_extract_provider: "missing".to_owned(),
            auto_extract_model: "extractor-model".to_owned(),
            ..MemoryConfig::default()
        };
        assert!(has_error(
            &config
                .validate_with_provider_names(["openai", "anthropic"])
                .unwrap_err(),
            "auto_extract_provider",
            "unresolved value"
        ));
        config
            .validate_with_provider_names(["missing", "openai"])
            .unwrap();
    }

    #[test]
    fn invalid_qdrant_schemes_hosts_and_credentials_are_rejected() {
        for url in [
            "grpc://qdrant.example.com",
            "http://",
            "https:///collections",
            "https://user:pass@qdrant.example.com",
            "HTTP://qdrant.example.com",
        ] {
            let mut qdrant = qdrant_config();
            qdrant.qdrant_url = url.to_owned();
            let config = MemoryConfig {
                qdrant: Some(qdrant),
                ..MemoryConfig::default()
            };
            assert!(has_error(
                &config.validate().unwrap_err(),
                "qdrant.qdrant_url",
                "HTTP or HTTPS URL"
            ));
        }
    }

    #[test]
    fn qdrant_threshold_weights_collection_and_embedding_fields_are_validated() {
        let mut qdrant = qdrant_config();
        qdrant.qdrant_collection = String::new();
        qdrant.similarity_threshold = f32::NAN;
        qdrant.embedding_provider = " ".to_owned();
        qdrant.embedding_model = String::new();
        qdrant.fts_weight = -0.1;
        qdrant.vector_weight = f32::INFINITY;
        let errors = MemoryConfig {
            qdrant: Some(qdrant),
            ..MemoryConfig::default()
        }
        .validate()
        .unwrap_err();

        for field in [
            "qdrant.qdrant_collection",
            "qdrant.similarity_threshold",
            "qdrant.embedding_provider",
            "qdrant.embedding_model",
            "qdrant.fts_weight",
            "qdrant.vector_weight",
        ] {
            assert!(errors.iter().any(|error| error.field == field), "{field}");
        }

        let mut qdrant = qdrant_config();
        qdrant.fts_weight = 0.0;
        qdrant.vector_weight = 0.0;
        let errors = MemoryConfig {
            qdrant: Some(qdrant),
            ..MemoryConfig::default()
        }
        .validate()
        .unwrap_err();
        assert!(has_error(
            &errors,
            "qdrant.fts_weight/qdrant.vector_weight",
            "sum greater than 0"
        ));
    }

    #[test]
    fn injection_strategy_deserializes_as_snake_case_through_all_configs() {
        let global: MemoryConfig =
            serde_yaml::from_str("injection_strategy: synthetic_message").unwrap();
        let provider: ProviderMemoryOverride =
            serde_yaml::from_str("injection_strategy: system_prompt_prefix").unwrap();
        let model: ModelGroupMemoryOverride =
            serde_yaml::from_str("injection_strategy: synthetic_message").unwrap();

        assert_eq!(
            global.injection_strategy,
            InjectionStrategy::SyntheticMessage
        );
        assert_eq!(
            provider.injection_strategy,
            Some(InjectionStrategy::SystemPromptPrefix)
        );
        assert_eq!(
            model.injection_strategy,
            Some(InjectionStrategy::SyntheticMessage)
        );
    }

    #[test]
    fn valid_full_configuration_deserializes_and_validates() {
        let config: MemoryConfig = serde_yaml::from_str(
            r#"
enabled: true
database_path: ./state/memory.db
injection_strategy: synthetic_message
max_injection_tokens: 10000
auto_extract_enabled: true
auto_extract_provider: openai
auto_extract_model: gpt-4.1-mini
auto_extract_min_turns: 12
decay_schedule_hours: 48
max_memories_per_namespace: 5000
allow_sensitive_storage: true
show_feedback: false
default_prompts: ["default assistant"]
custom_sensitive_patterns: ["SECRET-[0-9]+"]
qdrant:
  qdrant_url: https://qdrant.example.com:6333
  qdrant_collection: project_memories
  similarity_threshold: 0.25
  embedding_provider: openai
  embedding_model: text-embedding-3-small
  fts_weight: 0.3
  vector_weight: 0.7
"#,
        )
        .unwrap();

        config.validate_with_provider_names(["openai"]).unwrap();
        assert_eq!(
            config.injection_strategy,
            InjectionStrategy::SyntheticMessage
        );
        assert_eq!(config.qdrant.unwrap().qdrant_collection, "project_memories");
    }

    #[test]
    fn resolution_is_field_local_with_model_group_precedence() {
        let global = MemoryConfig {
            enabled: true,
            injection_strategy: InjectionStrategy::SystemPromptPrefix,
            max_injection_tokens: 500,
            show_feedback: true,
            ..MemoryConfig::default()
        };
        let provider = ProviderMemoryOverride {
            enabled: Some(false),
            injection_strategy: Some(InjectionStrategy::SyntheticMessage),
            max_injection_tokens: Some(750),
            show_feedback: Some(false),
        };
        let model_group = ModelGroupMemoryOverride {
            enabled: Some(true),
            injection_strategy: None,
            max_injection_tokens: Some(0),
            show_feedback: None,
        };

        assert_eq!(
            global.resolve(Some(&provider), Some(&model_group)),
            EffectiveMemoryConfig {
                enabled: true,
                injection_strategy: InjectionStrategy::SyntheticMessage,
                max_injection_tokens: 0,
                show_feedback: false,
            }
        );
    }

    #[test]
    fn resolution_falls_back_independently_and_preserves_explicit_false() {
        let global = MemoryConfig {
            enabled: true,
            injection_strategy: InjectionStrategy::SyntheticMessage,
            max_injection_tokens: 250,
            show_feedback: true,
            ..MemoryConfig::default()
        };
        let provider = ProviderMemoryOverride {
            enabled: Some(false),
            max_injection_tokens: Some(0),
            ..ProviderMemoryOverride::default()
        };
        let model_group = ModelGroupMemoryOverride {
            show_feedback: Some(false),
            ..ModelGroupMemoryOverride::default()
        };

        assert_eq!(
            global.resolve(Some(&provider), Some(&model_group)),
            EffectiveMemoryConfig {
                enabled: false,
                injection_strategy: InjectionStrategy::SyntheticMessage,
                max_injection_tokens: 0,
                show_feedback: false,
            }
        );
        assert_eq!(
            global.resolve(None, None),
            EffectiveMemoryConfig {
                enabled: true,
                injection_strategy: InjectionStrategy::SyntheticMessage,
                max_injection_tokens: 250,
                show_feedback: true,
            }
        );
    }

    #[test]
    fn override_validation_accepts_bounds_and_rejects_out_of_range_tokens() {
        for tokens in [0, MAX_INJECTION_TOKENS] {
            ProviderMemoryOverride {
                max_injection_tokens: Some(tokens),
                ..ProviderMemoryOverride::default()
            }
            .validate()
            .unwrap();
            ModelGroupMemoryOverride {
                max_injection_tokens: Some(tokens),
                ..ModelGroupMemoryOverride::default()
            }
            .validate()
            .unwrap();
        }

        for errors in [
            ProviderMemoryOverride {
                max_injection_tokens: Some(MAX_INJECTION_TOKENS + 1),
                ..ProviderMemoryOverride::default()
            }
            .validate()
            .unwrap_err(),
            ModelGroupMemoryOverride {
                max_injection_tokens: Some(MAX_INJECTION_TOKENS + 1),
                ..ModelGroupMemoryOverride::default()
            }
            .validate()
            .unwrap_err(),
        ] {
            assert!(has_error(
                &errors,
                "max_injection_tokens",
                "expected a value in 0..=10000"
            ));
        }
    }

    #[test]
    fn unknown_fields_are_rejected_at_every_level() {
        assert!(serde_yaml::from_str::<MemoryConfig>("surprise: true").is_err());
        assert!(serde_yaml::from_str::<MemoryQdrantConfig>("surprise: true").is_err());
        assert!(serde_yaml::from_str::<ProviderMemoryOverride>("surprise: true").is_err());
        assert!(serde_yaml::from_str::<ModelGroupMemoryOverride>("surprise: true").is_err());
    }

    // **Validates: Requirements 3.1, 3.2, 3.3, 3.4**
    //
    // Preservation: MemoryQdrantConfig YAML without vector_dimension field
    // deserializes correctly. Currently the struct has no such field, so any
    // valid YAML for the current fields works. After the fix adds the field,
    // this test verifies it defaults to None (backward compatible).
    #[test]
    fn preservation_qdrant_config_without_vector_dimension_deserializes() {
        let yaml = r#"
qdrant_url: https://qdrant.example.com:6333
qdrant_collection: obey_memories
similarity_threshold: 0.7
embedding_provider: openai
embedding_model: text-embedding-3-small
fts_weight: 0.4
vector_weight: 0.6
"#;
        let config: MemoryQdrantConfig = serde_yaml::from_str(yaml).unwrap();
        assert_eq!(config.qdrant_url, "https://qdrant.example.com:6333");
        assert_eq!(config.qdrant_collection, "obey_memories");
        assert!((config.similarity_threshold - 0.7).abs() < f32::EPSILON);
        assert_eq!(config.embedding_provider, "openai");
        assert_eq!(config.embedding_model, "text-embedding-3-small");
        assert!((config.fts_weight - 0.4).abs() < f32::EPSILON);
        assert!((config.vector_weight - 0.6).abs() < f32::EPSILON);
        assert_eq!(config.vector_dimension, None);
    }

    // **Validates: Requirements 3.4**
    //
    // Preservation: MemoryQdrantConfig::default() remains stable. This captures
    // the current default field values to guard against accidental regressions.
    #[test]
    fn preservation_qdrant_config_default_unchanged() {
        let defaults = MemoryQdrantConfig::default();
        assert_eq!(defaults.qdrant_url, "");
        assert_eq!(defaults.qdrant_collection, "obey_memories");
        assert!((defaults.similarity_threshold - 0.7).abs() < f32::EPSILON);
        assert_eq!(defaults.embedding_provider, "");
        assert_eq!(defaults.embedding_model, "");
        assert!((defaults.fts_weight - 0.4).abs() < f32::EPSILON);
        assert!((defaults.vector_weight - 0.6).abs() < f32::EPSILON);
        assert_eq!(defaults.vector_dimension, None);
    }

    // **Validates: Requirements 3.3, 3.4**
    //
    // Preservation: Validation continues to pass for valid MemoryQdrantConfig
    // without a vector_dimension field. After the fix, this proves configs
    // without the optional field remain valid.
    #[test]
    fn preservation_qdrant_config_validation_passes_without_vector_dimension() {
        let config = MemoryQdrantConfig {
            qdrant_url: "https://qdrant.example.com:6333".to_owned(),
            qdrant_collection: "obey_memories".to_owned(),
            similarity_threshold: 0.7,
            embedding_provider: "openai".to_owned(),
            embedding_model: "text-embedding-3-small".to_owned(),
            fts_weight: 0.4,
            vector_weight: 0.6,
            vector_dimension: None,
        };
        config.validate().unwrap();
    }

    // **Validates: Requirements 3.4**
    //
    // Preservation: Empty/default YAML deserializes to MemoryQdrantConfig::default()
    // confirming serde(default) behavior is stable.
    #[test]
    fn preservation_qdrant_config_empty_yaml_matches_default() {
        let config: MemoryQdrantConfig = serde_yaml::from_str("{}").unwrap();
        assert_eq!(config, MemoryQdrantConfig::default());
    }
}
