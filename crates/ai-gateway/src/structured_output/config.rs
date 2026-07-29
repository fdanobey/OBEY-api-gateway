use serde::{Deserialize, Serialize};
use std::fmt;

const MAX_RETRIES: u8 = 5;
const MIN_RETRY_TEMPERATURE: f32 = 0.0;
const MAX_RETRY_TEMPERATURE: f32 = 2.0;
const MAX_PASSTHROUGH_PROVIDERS: usize = 50;
const MAX_PROVIDER_NAME_LENGTH: usize = 128;

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct StructuredOutputConfig {
    #[serde(default = "default_true")]
    pub enabled: bool,
    #[serde(default = "default_max_retries")]
    pub max_retries: u8,
    #[serde(default)]
    pub retry_temperature: f32,
    #[serde(default)]
    pub passthrough_providers: Vec<String>,
}

impl Default for StructuredOutputConfig {
    fn default() -> Self {
        Self {
            enabled: default_true(),
            max_retries: default_max_retries(),
            retry_temperature: 0.0,
            passthrough_providers: Vec::new(),
        }
    }
}

impl StructuredOutputConfig {
    pub fn merge(&self, config_override: &StructuredOutputOverride) -> Self {
        Self {
            enabled: config_override.enabled.unwrap_or(self.enabled),
            max_retries: config_override.max_retries.unwrap_or(self.max_retries),
            retry_temperature: config_override
                .retry_temperature
                .unwrap_or(self.retry_temperature),
            passthrough_providers: config_override
                .passthrough_providers
                .clone()
                .unwrap_or_else(|| self.passthrough_providers.clone()),
        }
    }

    pub fn validate(&self) -> Result<(), Vec<StructuredOutputConfigError>> {
        validate_values(
            self.max_retries,
            self.retry_temperature,
            &self.passthrough_providers,
        )
    }
}

#[derive(Debug, Clone, Default, PartialEq, Serialize, Deserialize)]
pub struct StructuredOutputOverride {
    #[serde(default)]
    pub enabled: Option<bool>,
    #[serde(default)]
    pub max_retries: Option<u8>,
    #[serde(default)]
    pub retry_temperature: Option<f32>,
    #[serde(default)]
    pub passthrough_providers: Option<Vec<String>>,
}

impl StructuredOutputOverride {
    pub fn validate(&self) -> Result<(), Vec<StructuredOutputConfigError>> {
        let mut errors = Vec::new();

        if let Some(max_retries) = self.max_retries {
            validate_max_retries(&mut errors, max_retries);
        }
        if let Some(retry_temperature) = self.retry_temperature {
            validate_retry_temperature(&mut errors, retry_temperature);
        }
        if let Some(passthrough_providers) = self.passthrough_providers.as_deref() {
            validate_passthrough_providers(&mut errors, passthrough_providers);
        }

        validation_result(errors)
    }
}

#[derive(Debug, Clone, PartialEq)]
pub struct EffectiveConfig {
    pub enabled: bool,
    pub max_retries: u8,
    pub retry_temperature: f32,
    pub passthrough: bool,
}

#[derive(Debug, Clone, PartialEq)]
pub enum StructuredOutputConfigError {
    InvalidRange {
        field: &'static str,
        value: String,
        expected: &'static str,
    },
    TooManyPassthroughProviders {
        count: usize,
        max: usize,
    },
    PassthroughProviderTooLong {
        index: usize,
        value: String,
        length: usize,
        max: usize,
    },
}

impl fmt::Display for StructuredOutputConfigError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::InvalidRange {
                field,
                value,
                expected,
            } => write!(
                formatter,
                "structured_output.{field} is {value}; expected {expected}"
            ),
            Self::TooManyPassthroughProviders { count, max } => write!(
                formatter,
                "structured_output.passthrough_providers has {count} entries; expected at most {max} entries"
            ),
            Self::PassthroughProviderTooLong {
                index,
                value,
                length,
                max,
            } => write!(
                formatter,
                "structured_output.passthrough_providers[{index}] is {value:?} ({length} characters); expected at most {max} characters"
            ),
        }
    }
}

impl std::error::Error for StructuredOutputConfigError {}

fn validate_values(
    max_retries: u8,
    retry_temperature: f32,
    passthrough_providers: &[String],
) -> Result<(), Vec<StructuredOutputConfigError>> {
    let mut errors = Vec::new();
    validate_max_retries(&mut errors, max_retries);
    validate_retry_temperature(&mut errors, retry_temperature);
    validate_passthrough_providers(&mut errors, passthrough_providers);
    validation_result(errors)
}

fn validate_max_retries(errors: &mut Vec<StructuredOutputConfigError>, value: u8) {
    if value > MAX_RETRIES {
        errors.push(StructuredOutputConfigError::InvalidRange {
            field: "max_retries",
            value: value.to_string(),
            expected: "a value in 0..=5",
        });
    }
}

fn validate_retry_temperature(errors: &mut Vec<StructuredOutputConfigError>, value: f32) {
    if !value.is_finite() || !(MIN_RETRY_TEMPERATURE..=MAX_RETRY_TEMPERATURE).contains(&value) {
        errors.push(StructuredOutputConfigError::InvalidRange {
            field: "retry_temperature",
            value: value.to_string(),
            expected: "a finite value in 0.0..=2.0",
        });
    }
}

fn validate_passthrough_providers(
    errors: &mut Vec<StructuredOutputConfigError>,
    providers: &[String],
) {
    if providers.len() > MAX_PASSTHROUGH_PROVIDERS {
        errors.push(StructuredOutputConfigError::TooManyPassthroughProviders {
            count: providers.len(),
            max: MAX_PASSTHROUGH_PROVIDERS,
        });
    }

    for (index, provider) in providers.iter().enumerate() {
        let length = provider.chars().count();
        if length > MAX_PROVIDER_NAME_LENGTH {
            errors.push(StructuredOutputConfigError::PassthroughProviderTooLong {
                index,
                value: provider.clone(),
                length,
                max: MAX_PROVIDER_NAME_LENGTH,
            });
        }
    }
}

fn validation_result(
    errors: Vec<StructuredOutputConfigError>,
) -> Result<(), Vec<StructuredOutputConfigError>> {
    if errors.is_empty() {
        Ok(())
    } else {
        Err(errors)
    }
}

const fn default_true() -> bool {
    true
}

const fn default_max_retries() -> u8 {
    1
}

#[cfg(test)]
mod tests {
    use super::*;
    use proptest::prelude::*;

    fn provider_list_strategy() -> impl Strategy<Value = Vec<String>> {
        prop::collection::vec("[a-z][a-z0-9_-]{0,31}", 0..=8)
    }

    fn max_retries_strategy() -> impl Strategy<Value = u8> {
        prop_oneof![3 => 0u8..=MAX_RETRIES, 5 => (MAX_RETRIES + 1)..=u8::MAX]
    }

    fn retry_temperature_strategy() -> impl Strategy<Value = f32> {
        prop_oneof![
            4 => MIN_RETRY_TEMPERATURE..=MAX_RETRY_TEMPERATURE,
            3 => any::<u32>().prop_map(f32::from_bits),
            1 => prop::sample::select(vec![
                f32::NAN,
                f32::INFINITY,
                f32::NEG_INFINITY,
                f32::MIN,
                f32::MAX,
                f32::from_bits(MAX_RETRY_TEMPERATURE.to_bits() + 1),
            ]),
        ]
    }

    #[test]
    fn serde_defaults_match_documented_values() {
        let config: StructuredOutputConfig = serde_json::from_str("{}").unwrap();

        assert_eq!(config, StructuredOutputConfig::default());
        assert!(config.enabled);
        assert_eq!(config.max_retries, 1);
        assert_eq!(config.retry_temperature, 0.0);
        assert!(config.passthrough_providers.is_empty());
    }

    #[test]
    fn merge_replaces_only_explicit_override_fields() {
        let global = StructuredOutputConfig {
            enabled: true,
            max_retries: 1,
            retry_temperature: 0.25,
            passthrough_providers: vec!["global-provider".to_string()],
        };
        let config_override = StructuredOutputOverride {
            enabled: Some(false),
            max_retries: None,
            retry_temperature: Some(1.5),
            passthrough_providers: None,
        };

        let merged = global.merge(&config_override);

        assert!(!merged.enabled);
        assert_eq!(merged.max_retries, 1);
        assert_eq!(merged.retry_temperature, 1.5);
        assert_eq!(
            merged.passthrough_providers,
            vec!["global-provider".to_string()]
        );
    }

    #[test]
    fn merge_honors_explicit_empty_passthrough_provider_list() {
        let global = StructuredOutputConfig {
            passthrough_providers: vec!["global-provider".to_string()],
            ..StructuredOutputConfig::default()
        };
        let config_override = StructuredOutputOverride {
            passthrough_providers: Some(Vec::new()),
            ..StructuredOutputOverride::default()
        };

        assert!(global
            .merge(&config_override)
            .passthrough_providers
            .is_empty());
    }

    #[test]
    fn validation_accepts_inclusive_bounds() {
        for max_retries in [0, 5] {
            for retry_temperature in [0.0, 2.0] {
                let config = StructuredOutputConfig {
                    max_retries,
                    retry_temperature,
                    passthrough_providers: vec!["p".repeat(128); 50],
                    ..StructuredOutputConfig::default()
                };

                assert_eq!(config.validate(), Ok(()));
            }
        }
    }

    #[test]
    fn validation_aggregates_all_bounds_errors() {
        let mut passthrough_providers = vec!["p".repeat(129); 50];
        passthrough_providers.push("valid-provider".to_string());
        let config = StructuredOutputConfig {
            max_retries: 6,
            retry_temperature: f32::INFINITY,
            passthrough_providers,
            ..StructuredOutputConfig::default()
        };

        let errors = config.validate().unwrap_err();

        assert_eq!(errors.len(), 53);
        assert!(errors.iter().any(|error| matches!(
            error,
            StructuredOutputConfigError::InvalidRange {
                field: "max_retries",
                value,
                expected: "a value in 0..=5",
            } if value == "6"
        )));
        assert!(errors.iter().any(|error| matches!(
            error,
            StructuredOutputConfigError::InvalidRange {
                field: "retry_temperature",
                expected: "a finite value in 0.0..=2.0",
                ..
            }
        )));
        assert!(errors.iter().any(|error| matches!(
            error,
            StructuredOutputConfigError::TooManyPassthroughProviders { count: 51, max: 50 }
        )));
        assert_eq!(
            errors
                .iter()
                .filter(|error| matches!(
                    error,
                    StructuredOutputConfigError::PassthroughProviderTooLong {
                        length: 129,
                        max: 128,
                        ..
                    }
                ))
                .count(),
            50
        );

        let messages: Vec<String> = errors.iter().map(ToString::to_string).collect();
        assert!(messages
            .iter()
            .any(|message| message.contains("max_retries is 6") && message.contains("0..=5")));
        assert!(messages
            .iter()
            .any(|message| message.contains("retry_temperature")
                && message.contains("finite value in 0.0..=2.0")));
        assert!(messages
            .iter()
            .any(|message| message.contains("has 51 entries") && message.contains("at most 50")));
        assert!(messages
            .iter()
            .any(|message| message.contains("129 characters") && message.contains("at most 128")));
    }

    #[test]
    fn override_validation_checks_only_explicit_fields_and_rejects_nan() {
        assert_eq!(StructuredOutputOverride::default().validate(), Ok(()));

        let config_override = StructuredOutputOverride {
            max_retries: Some(9),
            retry_temperature: Some(f32::NAN),
            passthrough_providers: Some(vec!["p".repeat(129); 51]),
            ..StructuredOutputOverride::default()
        };

        assert_eq!(config_override.validate().unwrap_err().len(), 54);
    }

    proptest! {
        #![proptest_config(ProptestConfig::with_cases(256))]

        /// Property 10: Config Override Merge
        /// Validates: Requirements 5.2
        #[test]
        fn property_10_config_override_merge(
            global_enabled in any::<bool>(),
            global_max_retries in any::<u8>(),
            global_retry_temperature in any::<f32>(),
            global_passthrough_providers in provider_list_strategy(),
            override_enabled in prop::option::of(any::<bool>()),
            override_max_retries in prop::option::of(any::<u8>()),
            override_retry_temperature in prop::option::of(any::<f32>()),
            override_passthrough_providers in prop::option::of(provider_list_strategy()),
        ) {
            let global = StructuredOutputConfig {
                enabled: global_enabled,
                max_retries: global_max_retries,
                retry_temperature: global_retry_temperature,
                passthrough_providers: global_passthrough_providers,
            };
            let config_override = StructuredOutputOverride {
                enabled: override_enabled,
                max_retries: override_max_retries,
                retry_temperature: override_retry_temperature,
                passthrough_providers: override_passthrough_providers.clone(),
            };

            let merged = global.merge(&config_override);

            prop_assert_eq!(merged.enabled, override_enabled.unwrap_or(global.enabled));
            prop_assert_eq!(
                merged.max_retries,
                override_max_retries.unwrap_or(global.max_retries)
            );
            let expected_temperature =
                override_retry_temperature.unwrap_or(global.retry_temperature);
            prop_assert_eq!(
                merged.retry_temperature.to_bits(),
                expected_temperature.to_bits()
            );
            prop_assert_eq!(
                merged.passthrough_providers,
                override_passthrough_providers
                    .unwrap_or_else(|| global.passthrough_providers.clone())
            );
        }

        /// Property 11: Config Validation Bounds
        /// Validates: Requirements 5.5
        #[test]
        fn property_11_config_validation_bounds(
            max_retries in max_retries_strategy(),
            retry_temperature in retry_temperature_strategy(),
            passthrough_providers in provider_list_strategy(),
        ) {
            let config = StructuredOutputConfig {
                max_retries,
                retry_temperature,
                passthrough_providers,
                ..StructuredOutputConfig::default()
            };
            let retries_are_valid = max_retries <= MAX_RETRIES;
            let temperature_is_valid = retry_temperature.is_finite()
                && (MIN_RETRY_TEMPERATURE..=MAX_RETRY_TEMPERATURE)
                    .contains(&retry_temperature);

            prop_assert_eq!(
                config.validate().is_ok(),
                retries_are_valid && temperature_is_valid
            );

            let retries_only = StructuredOutputOverride {
                max_retries: Some(max_retries),
                ..StructuredOutputOverride::default()
            };
            prop_assert_eq!(retries_only.validate().is_ok(), retries_are_valid);

            let temperature_only = StructuredOutputOverride {
                retry_temperature: Some(retry_temperature),
                ..StructuredOutputOverride::default()
            };
            prop_assert_eq!(temperature_only.validate().is_ok(), temperature_is_valid);
        }
    }
}
