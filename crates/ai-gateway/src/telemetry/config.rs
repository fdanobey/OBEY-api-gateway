use serde::{Deserialize, Serialize};
use std::collections::HashMap;

use crate::config::resolve_env_var_in_string;

/// OpenTelemetry trace export configuration (opentelemetry-trace-export
/// spec, Req 1). Absent section applies [`TelemetryConfig::default`]:
/// fully disabled, no OTLP network activity.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct TelemetryConfig {
    #[serde(default)]
    pub enabled: bool,
    #[serde(default)]
    pub endpoint: Option<String>,
    #[serde(default)]
    pub headers: HashMap<String, String>,
    #[serde(default = "default_sampling_ratio")]
    pub sampling_ratio: f64,
    #[serde(default)]
    pub capture_content: bool,
    #[serde(default = "default_service_name")]
    pub service_name: String,
}

impl Default for TelemetryConfig {
    fn default() -> Self {
        Self {
            enabled: false,
            endpoint: None,
            headers: HashMap::new(),
            sampling_ratio: default_sampling_ratio(),
            capture_content: false,
            service_name: default_service_name(),
        }
    }
}

fn default_sampling_ratio() -> f64 {
    1.0
}

fn default_service_name() -> String {
    "ai-gateway".to_string()
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct TelemetryConfigError {
    pub field: String,
    pub value: String,
    pub expected: String,
}

impl TelemetryConfig {
    /// Validate the section (Req 1.2, 1.3, 1.5): the endpoint is only
    /// required when enabled, but every field is validated even when
    /// disabled so a broken dormant config is reported.
    pub fn validate(&self) -> Result<(), Vec<TelemetryConfigError>> {
        let mut errors = Vec::new();

        match self.endpoint.as_deref() {
            None | Some("") => {
                if self.enabled {
                    errors.push(TelemetryConfigError {
                        field: "opentelemetry.endpoint".to_string(),
                        value: self.endpoint.clone().unwrap_or_default(),
                        expected:
                            "a non-empty http/https URL (required when opentelemetry.enabled is true)"
                                .to_string(),
                    });
                }
            }
            Some(endpoint) => {
                let scheme_ok = reqwest::Url::parse(endpoint)
                    .map(|url| matches!(url.scheme(), "http" | "https"))
                    .unwrap_or(false);
                if !scheme_ok {
                    errors.push(TelemetryConfigError {
                        field: "opentelemetry.endpoint".to_string(),
                        value: endpoint.to_string(),
                        expected: "a valid http/https URL".to_string(),
                    });
                }
            }
        }

        if !(0.0..=1.0).contains(&self.sampling_ratio) {
            errors.push(TelemetryConfigError {
                field: "opentelemetry.sampling_ratio".to_string(),
                value: self.sampling_ratio.to_string(),
                expected: "a number in the range [0.0, 1.0]".to_string(),
            });
        }

        if errors.is_empty() {
            Ok(())
        } else {
            Err(errors)
        }
    }

    /// Resolve header values with `${ENV_VAR}` substitution (Req 1.4),
    /// mirroring the provider custom-header pattern.
    pub fn resolve_headers(&self) -> HashMap<String, String> {
        self.headers
            .iter()
            .map(|(key, value)| (key.clone(), resolve_env_var_in_string(value)))
            .collect()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn absent_section_defaults_keep_telemetry_disabled() {
        let config: TelemetryConfig = serde_yaml::from_str("{}").unwrap();
        assert!(!config.enabled);
        assert_eq!(config.endpoint, None);
        assert!(config.headers.is_empty());
        assert_eq!(config.sampling_ratio, 1.0);
        assert!(!config.capture_content);
        assert_eq!(config.service_name, "ai-gateway");
        assert_eq!(TelemetryConfig::default(), config);
        assert!(config.validate().is_ok());
    }

    #[test]
    fn partial_section_applies_field_defaults() {
        let config: TelemetryConfig = serde_yaml::from_str(
            "enabled: true\nendpoint: http://localhost:4318/v1/traces\n",
        )
        .unwrap();
        assert!(config.enabled);
        assert_eq!(config.sampling_ratio, 1.0);
        assert!(!config.capture_content);
        assert_eq!(config.service_name, "ai-gateway");
        assert!(config.validate().is_ok());
    }

    #[test]
    fn snake_case_keys_parse() {
        let config: TelemetryConfig = serde_yaml::from_str(
            "sampling_ratio: 0.25\ncapture_content: true\nservice_name: custom-edge\n",
        )
        .unwrap();
        assert_eq!(config.sampling_ratio, 0.25);
        assert!(config.capture_content);
        assert_eq!(config.service_name, "custom-edge");
        assert!(config.validate().is_ok());
    }

    #[test]
    fn header_env_references_are_resolved() {
        // Drop guard keeps the env mutation safe under parallel test
        // threads: the var is removed even when an assertion panics.
        struct EnvVarGuard;
        impl Drop for EnvVarGuard {
            fn drop(&mut self) {
                std::env::remove_var("OTEL_TEST_AUTH_TOKEN");
            }
        }
        let _guard = EnvVarGuard;
        std::env::set_var("OTEL_TEST_AUTH_TOKEN", "token123");
        let config: TelemetryConfig = serde_yaml::from_str(
            "headers:\n  authorization: \"Basic ${OTEL_TEST_AUTH_TOKEN}\"\n",
        )
        .unwrap();
        let headers = config.resolve_headers();
        assert_eq!(
            headers.get("authorization").map(String::as_str),
            Some("Basic token123")
        );
    }

    #[test]
    fn unresolved_env_references_are_left_verbatim() {
        let config: TelemetryConfig = serde_yaml::from_str(
            "headers:\n  authorization: \"Basic ${OTEL_TEST_UNSET_VAR_9C31}\"\n",
        )
        .unwrap();
        let headers = config.resolve_headers();
        assert_eq!(
            headers.get("authorization").map(String::as_str),
            Some("Basic ${OTEL_TEST_UNSET_VAR_9C31}")
        );
    }
}
