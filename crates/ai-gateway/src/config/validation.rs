use super::*;
use std::path::{Path, PathBuf};
use std::env;

use crate::secrets;

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

    #[error("Codex field '{field}' is only valid on oauth+openai providers (provider: '{provider}')")]
    InvalidCodexField { provider: String, field: String },
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
            errors.push(ValidationError::InvalidTimeout(self.server.request_timeout_seconds));
        }
        
        // Validate at least one provider (21.7)
        if self.providers.is_empty() {
            errors.push(ValidationError::NoProviders);
        }
        
        // Validate provider timeouts and env vars
        for provider in &self.providers {
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
                    field: format!("providers.{}.connection_pool.max_idle_per_host", provider.name),
                    value: provider.connection_pool.max_idle_per_host.to_string(),
                    expected: "a positive integer".to_string(),
                });
            }

            if provider.connection_pool.idle_timeout_seconds == 0 {
                errors.push(ValidationError::InvalidValue {
                    field: format!("providers.{}.connection_pool.idle_timeout_seconds", provider.name),
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
                    tracing::warn!("Environment variable '{}' for provider '{}' is not set — provider will be unavailable until configured", env_var, provider.name);
                }
            }

            if provider.api_key_encrypted.is_some() && provider.resolved_api_key.is_none() {
                tracing::warn!(
                    "Encrypted API key for provider '{}' could not be resolved — provider will be unavailable until the key is re-entered",
                    provider.name
                );
            }
            
            if let Some(ref env_var) = provider.api_secret_env {
                if !env_var.is_empty() && env::var(env_var).is_err() {
                    tracing::warn!("Environment variable '{}' for provider '{}' is not set — provider will be unavailable until configured", env_var, provider.name);
                }
            }

            // OAuth auth_method is only supported for OpenAI providers (6.1, 6.5)
            if provider.auth_method.as_deref() == Some("oauth") && provider.provider_type != "openai" {
                errors.push(ValidationError::InvalidValue {
                    field: format!("providers.{}.auth_method", provider.name),
                    value: "oauth".to_string(),
                    expected: "auth_method 'oauth' is only supported for provider_type 'openai'".to_string(),
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

            // Req 10.9 — codex_model_override must be non-empty when set
            if let Some(ref m) = provider.codex_model_override {
                if m.trim().is_empty() {
                    errors.push(ValidationError::InvalidCodexField {
                        provider: provider.name.clone(),
                        field: "codex_model_override (must be non-empty)".to_string(),
                    });
                }
            }

            // Req 12.6 — ToS warning when Codex provider active with admin auth disabled
            if is_codex_capable && !self.admin.auth.enabled {
                tracing::warn!(
                    provider = %provider.name,
                    "Codex provider active with admin auth disabled — ToS risk: \
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
                        "Bedrock provider '{}' has no API key configured — authentication is required for Bedrock Mantle endpoints",
                        provider.name
                    );
                }

                if provider.custom_vpc_endpoint && provider.base_url.as_deref().unwrap_or("").is_empty() {
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
            // keepalive_interval_seconds must be within 0–60 (0 = disabled).
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
        
        // Validate admin auth env vars (21.5)
        if self.admin.auth.enabled {
            if let Some(ref env_var) = self.admin.auth.username_env {
                if env::var(env_var).is_err() {
                    tracing::warn!("Admin auth env var '{}' is not set — admin auth will be disabled until configured", env_var);
                }
            }
            if let Some(ref env_var) = self.admin.auth.password_env {
                if env::var(env_var).is_err() {
                    tracing::warn!("Admin auth env var '{}' is not set — admin auth will be disabled until configured", env_var);
                }
            }
        }
        
        // Validate model groups (21.8)
        for group in &self.model_groups {
            if group.models.is_empty() {
                errors.push(ValidationError::EmptyModelGroup(group.name.clone()));
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
            }
        }
        
        if errors.is_empty() {
            Ok(())
        } else {
            Err(errors)
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
    let mut config: Config = serde_yaml::from_str(&contents)
        .map_err(|e| format!("Invalid YAML syntax: {}", e))?;

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
    config.validate()
        .map_err(|errors| {
            let error_messages: Vec<String> = errors.iter()
                .map(|e| e.to_string())
                .collect();
            format!("Configuration validation failed:\n  - {}", error_messages.join("\n  - "))
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

    std::fs::write(path, yaml)
        .map_err(|e| format!("Failed to write configuration file: {}", e))
}

#[cfg(test)]
mod property_tests {
    use super::*;
    use proptest::prelude::*;
    
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
                api_key_env: None,  // No env var required for test
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
                reasoning: true,
                codex_base_url_override: None,
                codex_model_override: None,
                instructions_override: None,
                max_rate_limit_cooldown_seconds: None,
            }],
            model_groups: vec![ModelGroup {
                name: "test-group".to_string(),
                version_fallback_enabled: false,
                models: vec![ProviderModel {
                    provider: "test-provider".to_string(),
                    model: "gpt-4".to_string(),
                    cost_per_million_input_tokens: 10.0,
                    cost_per_million_output_tokens: 30.0,
                    priority: 100,
                }],
            }],
            circuit_breaker: CircuitBreakerConfig::default(),
            retry: RetryConfig::default(),
            logging: LoggingConfig::default(),
            semantic_cache: None,
            exact_cache: ExactCacheConfig::default(),
            prometheus: None,
            context: ContextConfig::default(),
            first_launch_completed: false,
            tray: TrayConfig::default(),
            codex_instructions_url: None,
            streaming: None,
        }
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
        assert!(result.is_err(), "auth_method 'oauth' on non-openai provider should be rejected");
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
    fn test_streaming_keepalive_above_max_rejected() {
        let mut config = minimal_valid_config();
        config.streaming = Some(StreamingConfig {
            keepalive_interval_seconds: 61,
            ..StreamingConfig::default()
        });

        let result = config.validate();
        assert!(result.is_err(), "keepalive_interval_seconds > 60 should be rejected");
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
    fn test_streaming_keepalive_at_max_accepted() {
        let mut config = minimal_valid_config();
        config.streaming = Some(StreamingConfig {
            keepalive_interval_seconds: 60,
            ..StreamingConfig::default()
        });

        let result = config.validate();
        assert!(result.is_ok(), "keepalive_interval_seconds == 60 should be accepted: {:?}", result);
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
        assert!(result.is_ok(), "keepalive_interval_seconds == 0 should be accepted: {:?}", result);
    }

    #[test]
    fn test_streaming_chunk_timeout_below_min_rejected() {
        let mut config = minimal_valid_config();
        config.streaming = Some(StreamingConfig {
            chunk_timeout_seconds: 4,
            ..StreamingConfig::default()
        });

        let result = config.validate();
        assert!(result.is_err(), "chunk_timeout_seconds < 5 should be rejected");
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
        assert!(result.is_ok(), "chunk_timeout_seconds == 5 should be accepted: {:?}", result);
    }

    #[test]
    fn test_streaming_default_config_accepted() {
        let mut config = minimal_valid_config();
        config.streaming = Some(StreamingConfig::default());

        let result = config.validate();
        assert!(result.is_ok(), "Default StreamingConfig should be accepted: {:?}", result);
    }

    #[test]
    fn test_oauth_auth_method_accepted_for_openai_provider() {
        let mut config = minimal_valid_config();
        config.providers[0].provider_type = "openai".to_string();
        config.providers[0].auth_method = Some("oauth".to_string());

        let result = config.validate();
        assert!(result.is_ok(), "auth_method 'oauth' on openai provider should be accepted: {:?}", result);
    }

    #[test]
    fn test_no_auth_method_accepted_for_any_provider() {
        let mut config = minimal_valid_config();
        config.providers[0].provider_type = "anthropic".to_string();
        config.providers[0].auth_method = None;

        let result = config.validate();
        assert!(result.is_ok(), "No auth_method should be accepted for any provider: {:?}", result);
    }

    // Feature: bedrock-ui-integration, Bedrock validation rules
    // **Validates: Requirements 9.1, 9.2**

    #[test]
    fn test_bedrock_provider_without_region_rejected() {
        let mut config = minimal_valid_config();
        config.providers[0].provider_type = "bedrock".to_string();
        config.providers[0].region = None;

        let result = config.validate();
        assert!(result.is_err(), "Bedrock provider without region should be rejected");
        let errors = result.unwrap_err();
        assert!(
            errors.iter().any(|e| matches!(e, ValidationError::MissingBedrockRegion(_))),
            "Should contain MissingBedrockRegion error"
        );
    }

    #[test]
    fn test_bedrock_provider_with_region_accepted() {
        let mut config = minimal_valid_config();
        config.providers[0].provider_type = "bedrock".to_string();
        config.providers[0].region = Some("us-east-1".to_string());

        let result = config.validate();
        assert!(result.is_ok(), "Bedrock provider with region should be accepted: {:?}", result);
    }

    #[test]
    fn test_non_bedrock_provider_without_region_accepted() {
        let mut config = minimal_valid_config();
        config.providers[0].provider_type = "openai".to_string();
        config.providers[0].region = None;

        let result = config.validate();
        assert!(result.is_ok(), "Non-bedrock provider without region should be accepted: {:?}", result);
    }

    // Feature: bedrock-ui-integration, Property 4: Provider config Bedrock fields round-trip through serialization
    // **Validates: Requirements 8.1, 3.2**
    proptest! {
        #![proptest_config(proptest::prelude::ProptestConfig::with_cases(100))]

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
        #![proptest_config(proptest::prelude::ProptestConfig::with_cases(200))]

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

            // Deserialize — must succeed
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
}
