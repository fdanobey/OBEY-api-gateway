//! Configuration for the provider-agnostic web search feature.

use serde::{Deserialize, Serialize};

const DEFAULT_TIMEOUT_SECONDS: u64 = 15;
const DEFAULT_MAX_ITERATIONS: u32 = 5;
const DEFAULT_OUTPUT_TO_CHAT: bool = true;

#[derive(Debug, Clone, Default, Serialize, Deserialize, PartialEq)]
pub struct WebSearchConfig {
    #[serde(default)]
    pub enabled: Option<bool>,
    #[serde(default)]
    pub backend: Option<String>,
    #[serde(default)]
    pub api_key_env: Option<String>,
    #[serde(default)]
    pub base_url: Option<String>,
    #[serde(default)]
    pub timeout_seconds: Option<u64>,
    #[serde(default)]
    pub max_iterations: Option<u32>,
    #[serde(default)]
    pub output_to_chat: Option<bool>,
}

impl WebSearchConfig {
    pub fn validate(&self) -> Result<(), String> {
        if let Some(backend) = &self.backend {
            if !matches!(
                backend.as_str(),
                "tavily" | "brave" | "exa" | "searxng" | "codex"
            ) {
                return Err(format!(
                    "web_search.backend must be one of: tavily, brave, exa, searxng, codex (got '{}')",
                    backend
                ));
            }
        }
        if !(1..=120).contains(&self.effective_timeout_seconds()) {
            return Err("web_search.timeout_seconds must be between 1 and 120".to_string());
        }
        if !(1..=20).contains(&self.effective_max_iterations()) {
            return Err("web_search.max_iterations must be between 1 and 20".to_string());
        }
        if let Some(url) = &self.base_url {
            let parsed = reqwest::Url::parse(url.trim());
            let valid = parsed.as_ref().map_or(false, |u| {
                matches!(u.scheme(), "http" | "https") && u.host_str().is_some()
            });
            if !valid {
                return Err("web_search.base_url must be an absolute HTTP or HTTPS URL with a host".to_string());
            }
        }
        if self.effective_enabled() {
            let backend = match self.backend.as_deref() {
                Some(backend) => backend,
                None => {
                    return Err(
                        "web_search.backend is required when web_search.enabled is true".to_string(),
                    )
                }
            };
            if matches!(backend, "tavily" | "brave" | "exa")
                && self
                    .api_key_env
                    .as_deref()
                    .map_or(true, |key| key.trim().is_empty())
            {
                return Err(format!(
                    "web_search.api_key_env is required when web_search.backend is '{}'",
                    backend
                ));
            }
            if backend == "searxng"
                && self
                    .base_url
                    .as_deref()
                    .map_or(true, |url| url.trim().is_empty())
            {
                return Err(
                    "web_search.base_url is required when web_search.backend is 'searxng'"
                        .to_string(),
                );
            }
        }
        Ok(())
    }

    pub fn effective_enabled(&self) -> bool {
        match self.enabled {
            Some(explicit) => explicit,
            None => self.backend.is_some(),
        }
    }

    /// When enabled, executed search tool results are also appended to the
    /// assistant message content so they persist in the visible chat
    /// history instead of living only in tool-call metadata.
    pub fn effective_output_to_chat(&self) -> bool {
        self.output_to_chat.unwrap_or(DEFAULT_OUTPUT_TO_CHAT)
    }

    pub fn effective_timeout_seconds(&self) -> u64 {
        self.timeout_seconds.unwrap_or(DEFAULT_TIMEOUT_SECONDS)
    }

    pub fn effective_max_iterations(&self) -> u32 {
        self.max_iterations.unwrap_or(DEFAULT_MAX_ITERATIONS)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use proptest::prelude::*;

    #[test]
    fn backend_presence_implies_enabled() {
        let cfg = WebSearchConfig {
            backend: Some("tavily".to_string()),
            api_key_env: Some("TAVILY_API_KEY".to_string()),
            ..Default::default()
        };
        assert!(cfg.effective_enabled());
    }

    #[test]
    fn absent_backend_and_enabled_defaults_to_disabled() {
        let cfg = WebSearchConfig::default();
        assert!(!cfg.effective_enabled());
    }

    #[test]
    fn explicit_false_disables_even_with_backend() {
        let cfg = WebSearchConfig {
            enabled: Some(false),
            backend: Some("tavily".to_string()),
            ..Default::default()
        };
        assert!(!cfg.effective_enabled());
    }

    #[test]
    fn explicit_true_without_backend_rejected() {
        let cfg = WebSearchConfig {
            enabled: Some(true),
            ..Default::default()
        };
        let err = cfg.validate().unwrap_err();
        assert!(err.contains("web_search.backend is required"));
    }

    #[test]
    fn unknown_backend_rejected() {
        let cfg = WebSearchConfig {
            backend: Some("google".to_string()),
            ..Default::default()
        };
        let err = cfg.validate().unwrap_err();
        assert!(err.contains("web_search.backend must be one of"));
    }

    #[test]
    fn minimal_tavily_with_api_key_env_valid() {
        let cfg = WebSearchConfig {
            backend: Some("tavily".to_string()),
            api_key_env: Some("TAVILY_API_KEY".to_string()),
            ..Default::default()
        };
        assert!(cfg.validate().is_ok());
    }

    #[test]
    fn minimal_searxng_with_base_url_valid() {
        let cfg = WebSearchConfig {
            backend: Some("searxng".to_string()),
            base_url: Some("http://localhost:8080/search".to_string()),
            ..Default::default()
        };
        assert!(cfg.validate().is_ok());
    }

    #[test]
    fn minimal_codex_bare_backend_valid() {
        let cfg = WebSearchConfig {
            backend: Some("codex".to_string()),
            ..Default::default()
        };
        assert!(cfg.validate().is_ok());
    }

    #[test]
    fn keyed_backends_require_api_key_env() {
        for backend in ["tavily", "brave", "exa"] {
            let cfg = WebSearchConfig {
                backend: Some(backend.to_string()),
                ..Default::default()
            };
            let err = cfg.validate().unwrap_err();
            assert!(
                err.contains("web_search.api_key_env is required"),
                "{} should require api_key_env",
                backend
            );
        }
    }

    #[test]
    fn searxng_without_base_url_rejected() {
        let cfg = WebSearchConfig {
            backend: Some("searxng".to_string()),
            ..Default::default()
        };
        let err = cfg.validate().unwrap_err();
        assert!(err.contains("web_search.base_url is required"));
    }

    #[test]
    fn searxng_requirements_waived_when_explicitly_disabled() {
        let cfg = WebSearchConfig {
            enabled: Some(false),
            backend: Some("searxng".to_string()),
            ..Default::default()
        };
        assert!(cfg.validate().is_ok());
    }

    #[test]
    fn timeout_at_boundaries_accepted() {
        let low = WebSearchConfig {
            timeout_seconds: Some(1),
            ..Default::default()
        };
        assert!(low.validate().is_ok());

        let high = WebSearchConfig {
            timeout_seconds: Some(120),
            ..Default::default()
        };
        assert!(high.validate().is_ok());
    }

    #[test]
    fn timeout_below_range_rejected() {
        let cfg = WebSearchConfig {
            timeout_seconds: Some(0),
            ..Default::default()
        };
        let err = cfg.validate().unwrap_err();
        assert!(err.contains("web_search.timeout_seconds"));
    }

    #[test]
    fn timeout_above_range_rejected() {
        let cfg = WebSearchConfig {
            timeout_seconds: Some(121),
            ..Default::default()
        };
        assert!(cfg.validate().is_err());
    }

    #[test]
    fn iterations_at_boundaries_accepted() {
        let low = WebSearchConfig {
            max_iterations: Some(1),
            ..Default::default()
        };
        assert!(low.validate().is_ok());

        let high = WebSearchConfig {
            max_iterations: Some(20),
            ..Default::default()
        };
        assert!(high.validate().is_ok());
    }

    #[test]
    fn iterations_below_range_rejected() {
        let cfg = WebSearchConfig {
            max_iterations: Some(0),
            ..Default::default()
        };
        let err = cfg.validate().unwrap_err();
        assert!(err.contains("web_search.max_iterations"));
    }

    #[test]
    fn iterations_above_range_rejected() {
        let cfg = WebSearchConfig {
            max_iterations: Some(21),
            ..Default::default()
        };
        assert!(cfg.validate().is_err());
    }

    #[test]
    fn http_url_accepted() {
        let cfg = WebSearchConfig {
            base_url: Some("http://localhost:8080/search".to_string()),
            ..Default::default()
        };
        assert!(cfg.validate().is_ok());
    }

    #[test]
    fn https_url_accepted() {
        let cfg = WebSearchConfig {
            base_url: Some("https://api.tavily.com/search".to_string()),
            ..Default::default()
        };
        assert!(cfg.validate().is_ok());
    }

    #[test]
    fn non_http_url_rejected() {
        let cfg = WebSearchConfig {
            base_url: Some("ftp://example.com".to_string()),
            ..Default::default()
        };
        let err = cfg.validate().unwrap_err();
        assert!(err.contains("web_search.base_url"), "{err}");
    }

    #[test]
    fn non_url_rejected() {
        let cfg = WebSearchConfig {
            base_url: Some("not a url".to_string()),
            ..Default::default()
        };
        let err = cfg.validate().unwrap_err();
        assert!(err.contains("web_search.base_url"), "{err}");
    }

    #[test]
    fn scheme_only_http_prefix_without_host_rejected() {
        let cfg = WebSearchConfig {
            base_url: Some("http://".to_string()),
            ..Default::default()
        };
        assert!(cfg.validate().is_err());
    }

    #[test]
    fn scheme_relative_url_rejected() {
        let cfg = WebSearchConfig {
            base_url: Some("//api.example.com/search".to_string()),
            ..Default::default()
        };
        assert!(cfg.validate().is_err());
    }

    #[test]
    fn scheme_only_text_rejected() {
        let cfg = WebSearchConfig {
            base_url: Some("https:".to_string()),
            ..Default::default()
        };
        assert!(cfg.validate().is_err());
    }

    #[test]
    fn base_url_error_names_field() {
        let cfg = WebSearchConfig {
            base_url: Some("gopher://example.com".to_string()),
            ..Default::default()
        };
        let err = cfg.validate().unwrap_err();
        assert!(err.starts_with("web_search.base_url"), "{err}");
    }

    #[test]
    fn url_with_userinfo_and_host_accepted() {
        let cfg = WebSearchConfig {
            base_url: Some("http://user:pass@localhost:8080/search".to_string()),
            ..Default::default()
        };
        assert!(cfg.validate().is_ok());
    }

    #[test]
    fn effective_timeout_seconds_defaults_to_15() {
        let cfg = WebSearchConfig::default();
        assert_eq!(cfg.effective_timeout_seconds(), 15);
    }

    #[test]
    fn effective_max_iterations_defaults_to_5() {
        let cfg = WebSearchConfig::default();
        assert_eq!(cfg.effective_max_iterations(), 5);
    }

    #[test]
    fn effective_output_to_chat_defaults_true() {
        let cfg = WebSearchConfig::default();
        assert!(cfg.effective_output_to_chat());
    }

    #[test]
    fn effective_output_to_chat_respects_explicit_false() {
        let cfg = WebSearchConfig {
            output_to_chat: Some(false),
            ..Default::default()
        };
        assert!(!cfg.effective_output_to_chat());
    }

    #[test]
    fn serde_ignores_unknown_fields_and_defaults_missing() {
        let cfg: WebSearchConfig =
            serde_yaml::from_str("backend: tavily\napi_key_env: KEY\nfuture_field: 42\n").unwrap();
        assert_eq!(cfg.backend.as_deref(), Some("tavily"));
        assert_eq!(cfg.enabled, None);
        assert_eq!(cfg.timeout_seconds, None);
        assert_eq!(cfg.max_iterations, None);
    }

    #[test]
    fn empty_section_deserializes_to_default() {
        let cfg: WebSearchConfig = serde_yaml::from_str("{}").unwrap();
        assert_eq!(cfg, WebSearchConfig::default());
    }

    proptest! {
        #![proptest_config(proptest::prelude::ProptestConfig::with_cases(100))]

        // Feature: provider-agnostic-web-search-injection, Task 1:
        // config validation rejects out-of-range values
        #[test]
        fn prop_config_validation_rejects_out_of_range(
            timeout in 0u64..=200u64,
            iterations in 0u32..=50u32,
        ) {
            let cfg = WebSearchConfig {
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
