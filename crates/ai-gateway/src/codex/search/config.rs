//! Configuration for the Codex Search feature.

use serde::{Deserialize, Serialize};
use std::time::Duration;

const DEFAULT_SEARCH_BASE_URL: &str = "https://chatgpt.com/backend-api/codex/alpha/search";
const DEFAULT_TIMEOUT_SECONDS: u64 = 15;
const DEFAULT_MAX_ITERATIONS: u32 = 5;

#[derive(Debug, Clone, Default, Serialize, Deserialize, PartialEq)]
pub struct CodexSearchConfig {
    #[serde(default)]
    pub enabled: Option<bool>,
    #[serde(default)]
    pub base_url: Option<String>,
    #[serde(default)]
    pub timeout_seconds: Option<u64>,
    #[serde(default)]
    pub max_iterations: Option<u32>,
}

impl CodexSearchConfig {
    pub fn validate(&self) -> Result<(), String> {
        if let Some(t) = self.timeout_seconds {
            if !(1..=120).contains(&t) {
                return Err("timeout_seconds must be between 1 and 120".to_string());
            }
        }
        if let Some(m) = self.max_iterations {
            if !(1..=20).contains(&m) {
                return Err("max_iterations must be between 1 and 20".to_string());
            }
        }
        if let Some(url) = &self.base_url {
            let valid = (url.starts_with("http://") && url.len() > "http://".len())
                || (url.starts_with("https://") && url.len() > "https://".len());
            if !valid {
                return Err("base_url must be an HTTP or HTTPS URL".to_string());
            }
        }
        Ok(())
    }

    pub fn effective_enabled(&self, has_codex_provider: bool) -> bool {
        self.enabled.unwrap_or(has_codex_provider)
    }

    pub fn effective_timeout(&self) -> Duration {
        Duration::from_secs(self.timeout_seconds.unwrap_or(DEFAULT_TIMEOUT_SECONDS))
    }

    pub fn effective_max_iterations(&self) -> u32 {
        self.max_iterations.unwrap_or(DEFAULT_MAX_ITERATIONS)
    }

    pub fn effective_base_url(&self) -> String {
        self.base_url
            .clone()
            .unwrap_or_else(|| DEFAULT_SEARCH_BASE_URL.to_string())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use proptest::prelude::*;

    #[test]
    fn timeout_at_boundaries_accepted() {
        let low = CodexSearchConfig {
            timeout_seconds: Some(1),
            ..Default::default()
        };
        assert!(low.validate().is_ok());

        let high = CodexSearchConfig {
            timeout_seconds: Some(120),
            ..Default::default()
        };
        assert!(high.validate().is_ok());
    }

    #[test]
    fn timeout_below_range_rejected() {
        let cfg = CodexSearchConfig {
            timeout_seconds: Some(0),
            ..Default::default()
        };
        assert!(cfg.validate().is_err());
    }

    #[test]
    fn timeout_above_range_rejected() {
        let cfg = CodexSearchConfig {
            timeout_seconds: Some(121),
            ..Default::default()
        };
        assert!(cfg.validate().is_err());
    }

    #[test]
    fn iterations_at_boundaries_accepted() {
        let low = CodexSearchConfig {
            max_iterations: Some(1),
            ..Default::default()
        };
        assert!(low.validate().is_ok());

        let high = CodexSearchConfig {
            max_iterations: Some(20),
            ..Default::default()
        };
        assert!(high.validate().is_ok());
    }

    #[test]
    fn iterations_below_range_rejected() {
        let cfg = CodexSearchConfig {
            max_iterations: Some(0),
            ..Default::default()
        };
        assert!(cfg.validate().is_err());
    }

    #[test]
    fn iterations_above_range_rejected() {
        let cfg = CodexSearchConfig {
            max_iterations: Some(21),
            ..Default::default()
        };
        assert!(cfg.validate().is_err());
    }

    #[test]
    fn http_url_accepted() {
        let cfg = CodexSearchConfig {
            base_url: Some("http://localhost:8080/search".to_string()),
            ..Default::default()
        };
        assert!(cfg.validate().is_ok());
    }

    #[test]
    fn https_url_accepted() {
        let cfg = CodexSearchConfig {
            base_url: Some("https://chatgpt.com/backend-api/codex/alpha/search".to_string()),
            ..Default::default()
        };
        assert!(cfg.validate().is_ok());
    }

    #[test]
    fn non_http_url_rejected() {
        let cfg = CodexSearchConfig {
            base_url: Some("ftp://example.com".to_string()),
            ..Default::default()
        };
        assert!(cfg.validate().is_err());
    }

    #[test]
    fn non_url_rejected() {
        let cfg = CodexSearchConfig {
            base_url: Some("not a url".to_string()),
            ..Default::default()
        };
        assert!(cfg.validate().is_err());
    }

    #[test]
    fn effective_enabled_defaults_to_has_codex_provider() {
        let cfg = CodexSearchConfig::default();
        assert!(cfg.effective_enabled(true));
    }

    #[test]
    fn effective_enabled_defaults_false_without_codex_provider() {
        let cfg = CodexSearchConfig::default();
        assert!(!cfg.effective_enabled(false));
    }

    #[test]
    fn effective_enabled_respects_explicit_false() {
        let cfg = CodexSearchConfig {
            enabled: Some(false),
            ..Default::default()
        };
        assert!(!cfg.effective_enabled(true));
    }

    #[test]
    fn effective_timeout_defaults() {
        let cfg = CodexSearchConfig::default();
        assert_eq!(cfg.effective_timeout(), Duration::from_secs(15));
    }

    #[test]
    fn effective_max_iterations_defaults() {
        let cfg = CodexSearchConfig::default();
        assert_eq!(cfg.effective_max_iterations(), 5);
    }

    #[test]
    fn effective_base_url_defaults() {
        let cfg = CodexSearchConfig::default();
        assert_eq!(
            cfg.effective_base_url(),
            "https://chatgpt.com/backend-api/codex/alpha/search"
        );
    }

    proptest! {
        #![proptest_config(proptest::prelude::ProptestConfig::with_cases(100))]

        // Feature: codex-search, Property 10: Config validation rejects out-of-range values
        #[test]
        fn prop_config_validation_rejects_out_of_range(
            timeout in 0u64..=200u64,
            iterations in 0u32..=50u32,
        ) {
            let cfg = CodexSearchConfig {
                timeout_seconds: Some(timeout),
                max_iterations: Some(iterations),
                ..Default::default()
            };
            let result = cfg.validate();
            let timeout_valid = (1..=120).contains(&timeout);
            let iterations_valid = (1..=20).contains(&iterations);
            if timeout_valid && iterations_valid {
                prop_assert!(result.is_ok(), "validate should pass for timeout={} iterations={}", timeout, iterations);
            } else {
                prop_assert!(result.is_err(), "validate should fail for timeout={} iterations={}", timeout, iterations);
            }
        }
    }
}
