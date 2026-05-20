use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::sync::LazyLock;

use crate::secrets;

pub mod validation;
pub use validation::{bootstrap_config_if_missing, load_and_validate_config, resolve_config_path, save_config};

/// Pre-compiled regex for environment variable substitution
/// Compiled once at startup using LazyLock for thread-safe lazy initialization
static ENV_VAR_REGEX: LazyLock<regex::Regex> = LazyLock::new(|| {
    regex::Regex::new(r"\$\{([A-Z_][A-Z0-9_]*)\}")
        .expect("Invalid regex pattern for environment variable substitution")
});

/// Resolve environment variable references in a string
/// Supports ${ENV_VAR} syntax
fn resolve_env_var_in_string(value: &str) -> String {
    let mut result = value.to_string();
    
    for cap in ENV_VAR_REGEX.captures_iter(value) {
        let env_var = &cap[1];
        if let Ok(env_value) = std::env::var(env_var) {
            result = result.replace(&format!("${{{}}}", env_var), &env_value);
        }
    }
    
    result
}

/// Main configuration structure for the OBEY-API gateway
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct Config {
    pub server: ServerConfig,
    #[serde(default)]
    pub tls: Option<TlsConfig>,
    #[serde(default)]
    pub admin: AdminConfig,
    #[serde(default)]
    pub dashboard: DashboardConfig,
    #[serde(default)]
    pub cors: CorsConfig,
    pub providers: Vec<Provider>,
    pub model_groups: Vec<ModelGroup>,
    #[serde(default)]
    pub circuit_breaker: CircuitBreakerConfig,
    #[serde(default)]
    pub retry: RetryConfig,
    #[serde(default)]
    pub logging: LoggingConfig,
    #[serde(default)]
    pub semantic_cache: Option<SemanticCacheConfig>,
    /// Tier-1 in-memory exact-match response cache.
    /// Defaults to enabled — see [`ExactCacheConfig::default`].
    #[serde(default)]
    pub exact_cache: ExactCacheConfig,
    #[serde(default)]
    pub prometheus: Option<PrometheusConfig>,
    #[serde(default)]
    pub context: ContextConfig,
    #[serde(default)]
    pub first_launch_completed: bool,
    #[serde(default)]
    pub tray: TrayConfig,
    /// Optional override for the Codex system-prompt source URL.
    /// Defaults to [`crate::codex::instructions::DEFAULT_CODEX_INSTRUCTIONS_URL`]
    /// at fetch time when absent. See [`crate::codex::instructions`].
    #[serde(default)]
    pub codex_instructions_url: Option<String>,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct TrayConfig {
    #[serde(default = "default_true")]
    pub show_notifications: bool,
    #[serde(default = "default_true")]
    pub auto_open_browser: bool,
    #[serde(default = "default_splash_duration")]
    pub splash_duration_ms: u64,
}

impl Default for TrayConfig {
    fn default() -> Self {
        Self {
            show_notifications: true,
            auto_open_browser: true,
            splash_duration_ms: default_splash_duration(),
        }
    }
}

fn default_splash_duration() -> u64 {
    3000
}

/// Server configuration
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct ServerConfig {
    pub host: String,
    pub port: u16,
    #[serde(default = "default_request_timeout")]
    pub request_timeout_seconds: u64,
    #[serde(default = "default_max_request_size")]
    pub max_request_size_mb: u64,
}

fn default_request_timeout() -> u64 {
    30
}

fn default_ttfb_timeout() -> u64 {
    30
}

fn default_total_timeout() -> u64 {
    300
}

fn default_thinking_ttfb_timeout() -> u64 {
    120
}

fn default_thinking_total_timeout() -> u64 {
    600
}

/// Returns `true` when the model name matches known reasoning / thinking model
/// patterns across providers (OpenAI o-series, DeepSeek-R1, Claude reasoning, QwQ, etc.).
pub fn is_thinking_model(model: &str) -> bool {
    let m = model.to_lowercase();

    // OpenAI o-series reasoning models (o1, o1-preview, o1-mini, o3, o3-mini, o4-mini, etc.)
    if (m.starts_with("o1") || m.starts_with("o3") || m.starts_with("o4"))
        && m.chars().nth(2).map_or(true, |c| c == '-' || c == ' ')
    {
        return true;
    }

    // DeepSeek reasoning
    if m.contains("deepseek-r1") || m.contains("deepseek-reasoner") {
        return true;
    }

    // Qwen QwQ reasoning
    if m.contains("qwq") {
        return true;
    }

    // Claude reasoning models (claude-3.5-sonnet-v2+, claude-3-opus, claude-4+)
    if m.contains("claude") {
        if m.contains("claude-3-5-sonnet") {
            if let Some(pos) = m.find("-v") {
                let after_v = &m[pos + 2..];
                if let Some(version) = after_v.chars().next().and_then(|c| c.to_digit(10)) {
                    if version >= 2 {
                        return true;
                    }
                }
            }
        }
        if m.contains("claude-3-opus") || m.contains("claude-4") || m.contains("claude-5") {
            return true;
        }
    }

    false
}

fn default_max_request_size() -> u64 {
    10
}

/// TLS configuration
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct TlsConfig {
    pub enabled: bool,
    pub cert_path: String,
    pub key_path: String,
}

/// Admin panel configuration
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct AdminConfig {
    #[serde(default = "default_true")]
    pub enabled: bool,
    #[serde(default = "default_admin_path")]
    pub path: String,
    #[serde(default)]
    pub auth: AdminAuthConfig,
}

impl Default for AdminConfig {
    fn default() -> Self {
        Self {
            enabled: true,
            path: "/admin".to_string(),
            auth: AdminAuthConfig::default(),
        }
    }
}

fn default_admin_path() -> String {
    "/admin".to_string()
}

/// Admin authentication configuration
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct AdminAuthConfig {
    #[serde(default)]
    pub enabled: bool,
    pub username_env: Option<String>,
    pub password_env: Option<String>,
}

impl Default for AdminAuthConfig {
    fn default() -> Self {
        Self {
            enabled: false,
            username_env: None,
            password_env: None,
        }
    }
}

impl AdminAuthConfig {
    /// Resolve admin username from environment variable at runtime
    /// Returns None if no username_env is configured or the environment variable is unset
    pub fn resolve_username(&self) -> Option<String> {
        self.username_env
            .as_ref()
            .and_then(|env_var| std::env::var(env_var).ok())
    }

    /// Resolve admin password from environment variable at runtime
    /// Returns None if no password_env is configured or the environment variable is unset
    pub fn resolve_password(&self) -> Option<String> {
        self.password_env
            .as_ref()
            .and_then(|env_var| std::env::var(env_var).ok())
    }
}

/// Dashboard configuration
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct DashboardConfig {
    #[serde(default = "default_true")]
    pub enabled: bool,
    #[serde(default = "default_dashboard_path")]
    pub path: String,
    #[serde(default = "default_metrics_interval")]
    pub metrics_update_interval_seconds: u64,
}

impl Default for DashboardConfig {
    fn default() -> Self {
        Self {
            enabled: true,
            path: "/dashboard".to_string(),
            metrics_update_interval_seconds: 1,
        }
    }
}

fn default_dashboard_path() -> String {
    "/dashboard".to_string()
}

fn default_metrics_interval() -> u64 {
    1
}

fn default_true() -> bool {
    true
}

/// CORS configuration
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct CorsConfig {
    #[serde(default)]
    pub enabled: bool,
    #[serde(default = "default_allowed_origins")]
    pub allowed_origins: Vec<String>,
    #[serde(default = "default_allowed_methods")]
    pub allowed_methods: Vec<String>,
    #[serde(default = "default_allowed_headers")]
    pub allowed_headers: Vec<String>,
}

impl Default for CorsConfig {
    fn default() -> Self {
        Self {
            enabled: false,
            allowed_origins: vec!["*".to_string()],
            allowed_methods: vec!["GET".to_string(), "POST".to_string(), "OPTIONS".to_string()],
            allowed_headers: vec!["Content-Type".to_string(), "Authorization".to_string()],
        }
    }
}

fn default_allowed_origins() -> Vec<String> {
    vec!["*".to_string()]
}

fn default_allowed_methods() -> Vec<String> {
    vec!["GET".to_string(), "POST".to_string(), "OPTIONS".to_string()]
}

fn default_allowed_headers() -> Vec<String> {
    vec!["Content-Type".to_string(), "Authorization".to_string()]
}

/// Provider configuration
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct Provider {
    pub name: String,
    #[serde(rename = "type")]
    pub provider_type: String,
    pub base_url: Option<String>,
    pub api_key_env: Option<String>,
    #[serde(default)]
    pub api_key_encrypted: Option<String>,
    pub api_secret_env: Option<String>,
    #[serde(default)]
    pub api_secret_encrypted: Option<String>,
    /// Authentication method override. When set to `Some("oauth")`, the router
    /// resolves the Bearer token from the OAuth token store instead of
    /// `api_key_env` / `api_key_encrypted`. Currently only supported for
    /// `provider_type: openai`. See [`crate::oauth`].
    #[serde(default)]
    pub auth_method: Option<String>,
    #[serde(skip, default)]
    pub resolved_api_key: Option<String>,
    #[serde(skip, default)]
    pub resolved_api_secret: Option<String>,
    pub region: Option<String>,
    /// Legacy single timeout field — used as `total_timeout_seconds` when the
    /// split fields are not set. Kept for backward compatibility.
    #[serde(default = "default_request_timeout")]
    pub timeout_seconds: u64,
    /// Time-to-first-byte timeout: maximum seconds to wait for the provider to
    /// start sending a response (first byte / thinking stream start).
    /// Default: 30 s (standard models), 120 s (thinking models).
    #[serde(default)]
    pub ttfb_timeout_seconds: Option<u64>,
    /// Total round-trip timeout: maximum seconds for the entire request
    /// (including reading the full response body).
    /// Default: 300 s (standard models), 600 s (thinking models).
    #[serde(default)]
    pub total_timeout_seconds: Option<u64>,
    #[serde(default = "default_max_connections")]
    pub max_connections: u32,
    #[serde(default)]
    pub rate_limit_per_minute: u32,
    #[serde(default)]
    pub custom_headers: HashMap<String, String>,
    #[serde(default)]
    pub connection_pool: ProviderConnectionPoolConfig,
    #[serde(default)]
    pub budget: Option<ProviderBudgetConfig>,
    /// Manually specified models for this provider.
    /// These are always included in /v1/models responses, even if the
    /// provider's own /v1/models endpoint returns nothing.
    #[serde(default)]
    pub manual_models: Vec<String>,
    /// Enable global inference (Bedrock only).
    /// When true, AWS auto-selects the optimal region for the request.
    /// Uses inference profile ARNs instead of model IDs.
    #[serde(default)]
    pub global_inference_profile: bool,
    /// Enable cross-region inference (Bedrock only).
    /// When true, model IDs are prefixed with the region group (e.g., "us.") to
    /// route requests across multiple regions for higher throughput.
    #[serde(default)]
    pub cross_region_inference: bool,
    /// Enable prompt caching for supported Bedrock models (Claude 3.5+).
    /// When true, the gateway includes prompt caching headers in requests.
    #[serde(default)]
    pub prompt_caching: bool,
    /// Use a custom VPC endpoint for Bedrock (Bedrock only).
    /// When true, the base_url field is used as-is instead of auto-generating
    /// the Bedrock Mantle endpoint from the region.
    #[serde(default)]
    pub custom_vpc_endpoint: bool,
    /// Enable reasoning/extended thinking for supported Bedrock models.
    /// When true, the gateway includes the `thinking` parameter in requests
    /// to models that support extended thinking (Claude 3.5 Sonnet+).
    #[serde(default = "default_true")]
    pub reasoning: bool,

    /// Override the Codex backend URL. Only honoured for oauth+openai providers.
    #[serde(default)]
    pub codex_base_url_override: Option<String>,

    /// Pin the Codex model independent of the client's requested model.
    /// May be any non-empty string the Codex backend accepts.
    #[serde(default)]
    pub codex_model_override: Option<String>,

    /// Replace the Codex system prompt entirely for this provider.
    #[serde(default)]
    pub instructions_override: Option<String>,

    /// Per-provider maximum cooldown (seconds) for upstream-driven
    /// rate-limit windows. Use this to opt a provider into very long
    /// cooldowns without changing the global default.
    ///
    /// Concrete examples:
    /// - Nano-GPT (weekly account quotas): `604800` (7d).
    /// - OpenAI / Anthropic (minute-bucket): leave unset; the global
    ///   default of 24h is enough headroom.
    ///
    /// Falls back to `RetryConfig.max_rate_limit_cooldown_seconds` when
    /// unset.
    #[serde(default)]
    pub max_rate_limit_cooldown_seconds: Option<u64>,
}

impl Provider {
    /// Resolve the effective TTFB timeout for a given model.
    /// Priority: explicit `ttfb_timeout_seconds` > model-aware default.
    pub fn effective_ttfb_timeout(&self, model: &str) -> u64 {
        if let Some(t) = self.ttfb_timeout_seconds {
            return t;
        }
        if is_thinking_model(model) {
            default_thinking_ttfb_timeout()
        } else {
            default_ttfb_timeout()
        }
    }

    /// Resolve the effective total (round-trip) timeout for a given model.
    /// Priority: explicit `total_timeout_seconds` > legacy `timeout_seconds`
    /// (when non-default) > model-aware default.
    pub fn effective_total_timeout(&self, model: &str) -> u64 {
        if let Some(t) = self.total_timeout_seconds {
            return t;
        }
        // If the user set the legacy field to something other than the old
        // default (30), honour it as the total timeout for backward compat.
        if self.timeout_seconds != default_request_timeout() {
            return self.timeout_seconds;
        }
        if is_thinking_model(model) {
            default_thinking_total_timeout()
        } else {
            default_total_timeout()
        }
    }

    /// Resolve API key from environment variable at runtime
    /// Returns None if no api_key_env is configured or the environment variable is unset
    pub fn resolve_api_key(&self) -> Option<String> {
        if let Some(resolved) = self.resolved_api_key.as_ref() {
            if !resolved.is_empty() {
                return Some(resolved.clone());
            }
        }

        self.api_key_env.as_ref().and_then(|env_var| {
            if env_var.is_empty() || !secrets::is_env_var_reference(env_var) {
                None
            } else {
                std::env::var(env_var).ok()
            }
        })
    }

    pub fn has_encrypted_api_key(&self) -> bool {
        self.api_key_encrypted
            .as_deref()
            .is_some_and(secrets::is_encrypted_secret)
    }

    pub fn has_plaintext_api_key_input(&self) -> bool {
        self.api_key_env
            .as_deref()
            .is_some_and(secrets::looks_like_plaintext_secret)
    }

    pub fn has_api_key_configured(&self) -> bool {
        self.resolve_api_key().is_some()
            || self.has_encrypted_api_key()
            || self
                .api_key_env
                .as_deref()
                .is_some_and(secrets::is_env_var_reference)
    }

    /// Resolve API secret from environment variable at runtime
    /// Returns None if no api_secret_env is configured or the environment variable is unset
    pub fn resolve_api_secret(&self) -> Option<String> {
        if let Some(resolved) = self.resolved_api_secret.as_ref() {
            if !resolved.is_empty() {
                return Some(resolved.clone());
            }
        }

        self.api_secret_env.as_ref().and_then(|env_var| {
            if env_var.is_empty() || !secrets::is_env_var_reference(env_var) {
                None
            } else {
                std::env::var(env_var).ok()
            }
        })
    }

    pub fn has_encrypted_api_secret(&self) -> bool {
        self.api_secret_encrypted
            .as_deref()
            .is_some_and(secrets::is_encrypted_secret)
    }

    pub fn has_plaintext_api_secret_input(&self) -> bool {
        self.api_secret_env
            .as_deref()
            .is_some_and(secrets::looks_like_plaintext_secret)
    }

    pub fn has_api_secret_configured(&self) -> bool {
        self.resolve_api_secret().is_some()
            || self.has_encrypted_api_secret()
            || self
                .api_secret_env
                .as_deref()
                .is_some_and(secrets::is_env_var_reference)
    }

    /// Resolve custom headers with environment variable substitution
    /// Supports ${ENV_VAR} syntax in header values
    pub fn resolve_custom_headers(&self) -> HashMap<String, String> {
        self.custom_headers
            .iter()
            .map(|(key, value)| {
                let resolved_value = resolve_env_var_in_string(value);
                (key.clone(), resolved_value)
            })
            .collect()
    }
}

fn default_max_connections() -> u32 {
    100
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum BudgetResetPolicy {
    Manual,
}

impl Default for BudgetResetPolicy {
    fn default() -> Self {
        Self::Manual
    }
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct ProviderConnectionPoolConfig {
    #[serde(default = "default_pool_max_idle_per_host")]
    pub max_idle_per_host: u32,
    #[serde(default = "default_pool_idle_timeout_seconds")]
    pub idle_timeout_seconds: u64,
}

impl Default for ProviderConnectionPoolConfig {
    fn default() -> Self {
        Self {
            max_idle_per_host: default_pool_max_idle_per_host(),
            idle_timeout_seconds: default_pool_idle_timeout_seconds(),
        }
    }
}

fn default_pool_max_idle_per_host() -> u32 {
    10
}

fn default_pool_idle_timeout_seconds() -> u64 {
    90
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct ProviderBudgetConfig {
    pub limit_usd: f64,
    #[serde(default)]
    pub reset_policy: BudgetResetPolicy,
}

/// Model group configuration
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct ModelGroup {
    pub name: String,
    #[serde(default)]
    pub version_fallback_enabled: bool,
    pub models: Vec<ProviderModel>,
}

/// Provider model configuration
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct ProviderModel {
    pub provider: String,
    pub model: String,
    #[serde(default)]
    pub cost_per_million_input_tokens: f64,
    #[serde(default)]
    pub cost_per_million_output_tokens: f64,
    #[serde(default = "default_priority")]
    pub priority: u32,
}

fn default_priority() -> u32 {
    100
}

impl ProviderModel {
    /// Calculate total cost for a request
    #[inline]
    pub fn total_cost(&self) -> f64 {
        self.cost_per_million_input_tokens + self.cost_per_million_output_tokens
    }
}

/// Circuit breaker configuration
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct CircuitBreakerConfig {
    #[serde(default = "default_failure_threshold")]
    pub failure_threshold: u32,
    #[serde(default = "default_backoff_sequence")]
    pub backoff_sequence_seconds: Vec<u64>,
    #[serde(default = "default_success_threshold")]
    pub success_threshold: u32,
}

impl Default for CircuitBreakerConfig {
    fn default() -> Self {
        Self {
            failure_threshold: 3,
            backoff_sequence_seconds: vec![5, 10, 20, 40, 300],
            success_threshold: 1,
        }
    }
}

fn default_failure_threshold() -> u32 {
    3
}

fn default_backoff_sequence() -> Vec<u64> {
    vec![5, 10, 20, 40, 300]
}

fn default_success_threshold() -> u32 {
    1
}

/// Retry configuration
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct RetryConfig {
    #[serde(default = "default_max_retries")]
    pub max_retries_per_provider: u32,
    #[serde(default = "default_retry_backoff")]
    pub backoff_sequence_seconds: Vec<u64>,
    #[serde(default = "default_retry_jitter_enabled")]
    pub jitter_enabled: bool,
    #[serde(default = "default_retry_jitter_ratio")]
    pub jitter_ratio: f64,
    /// Default upper bound (seconds) for an upstream-driven rate-limit
    /// cooldown. A provider that returns 429 with `Retry-After: 86400` is
    /// kept out of failover for up to this many seconds.
    ///
    /// Default: 86400 (24h). Cap covers daily quotas without burning
    /// weekly quotas. Override per-provider via
    /// `Provider.max_rate_limit_cooldown_seconds` for accounts with
    /// longer reset windows (e.g. Nano-GPT weekly).
    #[serde(default = "default_max_rate_limit_cooldown")]
    pub max_rate_limit_cooldown_seconds: u64,
    /// Cooldown applied when the upstream signals a rate-limit but
    /// provides no usable `Retry-After` / `retry_after_ms` value.
    /// Default: 30s.
    #[serde(default = "default_rate_limit_cooldown_default")]
    pub default_rate_limit_cooldown_seconds: u64,
}

impl Default for RetryConfig {
    fn default() -> Self {
        Self {
            max_retries_per_provider: 1,
            backoff_sequence_seconds: vec![1, 2, 4],
            jitter_enabled: true,
            jitter_ratio: 0.2,
            max_rate_limit_cooldown_seconds: default_max_rate_limit_cooldown(),
            default_rate_limit_cooldown_seconds: default_rate_limit_cooldown_default(),
        }
    }
}

fn default_max_retries() -> u32 {
    1
}

fn default_retry_backoff() -> Vec<u64> {
    vec![1, 2, 4]
}

fn default_retry_jitter_enabled() -> bool {
    true
}

fn default_retry_jitter_ratio() -> f64 {
    0.2
}

fn default_max_rate_limit_cooldown() -> u64 {
    24 * 60 * 60 // 24h: covers daily quotas, leaves weekly for per-provider override
}

fn default_rate_limit_cooldown_default() -> u64 {
    30
}

/// Logging configuration
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct LoggingConfig {
    #[serde(default = "default_log_level")]
    pub level: String,
    #[serde(default = "default_database_path")]
    pub database_path: String,
    #[serde(default)]
    pub request_body_logging: bool,
    #[serde(default)]
    pub response_body_logging: bool,
    #[serde(default = "default_max_body_size")]
    pub max_body_size_bytes: usize,
    #[serde(default = "default_excluded_fields")]
    pub excluded_fields: Vec<String>,
    #[serde(default = "default_retention_days")]
    pub retention_days: u32,
    #[serde(default = "default_cleanup_schedule")]
    pub cleanup_schedule_hours: u32,
}

impl Default for LoggingConfig {
    fn default() -> Self {
        Self {
            level: "info".to_string(),
            database_path: "./logs.db".to_string(),
            request_body_logging: false,
            response_body_logging: false,
            max_body_size_bytes: 10000,
            excluded_fields: vec!["api_key".to_string(), "authorization".to_string()],
            retention_days: 30,
            cleanup_schedule_hours: 24,
        }
    }
}

fn default_log_level() -> String {
    "info".to_string()
}

fn default_database_path() -> String {
    "./logs.db".to_string()
}

fn default_max_body_size() -> usize {
    10000
}

fn default_excluded_fields() -> Vec<String> {
    vec!["api_key".to_string(), "authorization".to_string()]
}

fn default_retention_days() -> u32 {
    30
}

fn default_cleanup_schedule() -> u32 {
    24
}

/// Semantic cache configuration
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct SemanticCacheConfig {
    pub enabled: bool,
    pub qdrant_url: String,
    #[serde(default = "default_collection_name")]
    pub collection_name: String,
    #[serde(default = "default_similarity_threshold")]
    pub similarity_threshold: f32,
    pub embedding_provider: String,
    pub embedding_model: String,
    #[serde(default = "default_ttl")]
    pub ttl_seconds: u64,
    #[serde(default = "default_max_cache_size")]
    pub max_cache_size: usize,
}

fn default_collection_name() -> String {
    "OBEY_API_cache".to_string()
}

fn default_similarity_threshold() -> f32 {
    0.95
}

fn default_ttl() -> u64 {
    3600
}

fn default_max_cache_size() -> usize {
    10000
}

/// Tier-1 in-memory exact-match cache configuration.
///
/// Catches byte-for-byte identical retries (and stream/non-stream variants
/// of the same request) with zero embedding overhead.  Independent of the
/// optional [`SemanticCacheConfig`].
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct ExactCacheConfig {
    /// Master switch.  Default: enabled.
    #[serde(default = "default_true")]
    pub enabled: bool,
    /// Maximum number of cached entries before oldest-first eviction.
    #[serde(default = "default_exact_max_entries")]
    pub max_entries: usize,
    /// Per-entry TTL in seconds.
    #[serde(default = "default_exact_ttl")]
    pub ttl_seconds: u64,
    /// Maximum temperature for a request to be considered cacheable.
    /// Higher temperatures imply non-determinism and are not cached.
    #[serde(default = "default_exact_temperature_threshold")]
    pub temperature_threshold: f32,
}

impl Default for ExactCacheConfig {
    fn default() -> Self {
        Self {
            enabled: true,
            max_entries: default_exact_max_entries(),
            ttl_seconds: default_exact_ttl(),
            temperature_threshold: default_exact_temperature_threshold(),
        }
    }
}

fn default_exact_max_entries() -> usize {
    5000
}

fn default_exact_ttl() -> u64 {
    3600
}

fn default_exact_temperature_threshold() -> f32 {
    0.15
}

/// Prometheus metrics configuration
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct PrometheusConfig {
    pub enabled: bool,
    #[serde(default = "default_metrics_path")]
    pub path: String,
}

fn default_metrics_path() -> String {
    "/metrics".to_string()
}

/// Context management configuration
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct ContextConfig {
    /// Enable automatic context truncation
    #[serde(default = "default_true")]
    pub enabled: bool,
    /// Truncation strategy: "remove_oldest" or "sliding_window"
    #[serde(default = "default_truncation_strategy")]
    pub truncation_strategy: String,
    /// Window size for sliding window strategy (number of messages to keep)
    #[serde(default = "default_sliding_window_size")]
    pub sliding_window_size: usize,
    /// Cache TTL for model capabilities in seconds
    #[serde(default = "default_capabilities_cache_ttl")]
    pub capabilities_cache_ttl_seconds: u64,
    /// Maximum retries for context truncation
    #[serde(default = "default_max_truncation_retries")]
    pub max_truncation_retries: usize,
    /// Default context window (tokens) used when model capabilities are unknown
    /// and no limit can be parsed from the provider error. Defaults to 32768.
    #[serde(default = "default_context_window")]
    pub default_context_window: u32,
}

impl Default for ContextConfig {
    fn default() -> Self {
        Self {
            enabled: true,
            truncation_strategy: "remove_oldest".to_string(),
            sliding_window_size: 10,
            capabilities_cache_ttl_seconds: 3600,
            max_truncation_retries: 3,
            default_context_window: 32768,
        }
    }
}

fn default_truncation_strategy() -> String {
    "remove_oldest".to_string()
}

fn default_sliding_window_size() -> usize {
    10
}

fn default_capabilities_cache_ttl() -> u64 {
    3600
}

fn default_max_truncation_retries() -> usize {
    3
}

fn default_context_window() -> u32 {
    32768
}

#[cfg(test)]
mod runtime_resolution_tests {
    use super::*;
    use std::env;

    #[test]
    fn test_provider_resolve_api_key() {
        env::set_var("TEST_API_KEY", "sk-test123");
        
        let provider = Provider {
            name: "test".to_string(),
            provider_type: "openai".to_string(),
            base_url: None,
            api_key_env: Some("TEST_API_KEY".to_string()),
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
            custom_headers: HashMap::new(),
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
        };
        
        assert_eq!(provider.resolve_api_key(), Some("sk-test123".to_string()));
        
        env::remove_var("TEST_API_KEY");
    }

    #[test]
    fn test_provider_resolve_api_key_none() {
        let provider = Provider {
            name: "test".to_string(),
            provider_type: "openai".to_string(),
            base_url: None,
            api_key_env: None,
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
            custom_headers: HashMap::new(),
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
        };
        
        assert_eq!(provider.resolve_api_key(), None);
    }

    #[test]
    fn test_provider_resolve_custom_headers() {
        env::set_var("CUSTOM_TOKEN", "token123");
        
        let mut headers = HashMap::new();
        headers.insert("X-API-Key".to_string(), "${CUSTOM_TOKEN}".to_string());
        headers.insert("X-Static".to_string(), "static-value".to_string());
        
        let provider = Provider {
            name: "test".to_string(),
            provider_type: "openai".to_string(),
            base_url: None,
            api_key_env: None,
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
            custom_headers: headers,
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
        };
        
        let resolved = provider.resolve_custom_headers();
        assert_eq!(resolved.get("X-API-Key"), Some(&"token123".to_string()));
        assert_eq!(resolved.get("X-Static"), Some(&"static-value".to_string()));
        
        env::remove_var("CUSTOM_TOKEN");
    }

    #[test]
    fn test_provider_resolve_api_key_prefers_runtime_secret() {
        let provider = Provider {
            name: "test".to_string(),
            provider_type: "openai".to_string(),
            base_url: None,
            api_key_env: Some("OPENAI_API_KEY".to_string()),
            api_key_encrypted: Some("enc-v1:abc:def".to_string()),
            api_secret_env: None,
            api_secret_encrypted: None,
            auth_method: None,
            resolved_api_key: Some("decrypted-key".to_string()),
            resolved_api_secret: None,
            region: None,
            timeout_seconds: 30,
            ttfb_timeout_seconds: None,
            total_timeout_seconds: None,
            max_connections: 100,
            rate_limit_per_minute: 0,
            custom_headers: HashMap::new(),
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
        };

        assert_eq!(provider.resolve_api_key(), Some("decrypted-key".to_string()));
    }

    #[test]
    fn test_provider_plaintext_secret_detection() {
        let provider = Provider {
            name: "test".to_string(),
            provider_type: "openai".to_string(),
            base_url: None,
            api_key_env: Some("sk-test-secret-12345678901234567890".to_string()),
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
            custom_headers: HashMap::new(),
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
        };

        assert!(provider.has_plaintext_api_key_input());
        assert!(!provider.resolve_api_key().is_some());
    }

    #[test]
    fn test_admin_auth_resolve_credentials() {
        env::set_var("ADMIN_USER", "admin");
        env::set_var("ADMIN_PASS", "secret");
        
        let auth = AdminAuthConfig {
            enabled: true,
            username_env: Some("ADMIN_USER".to_string()),
            password_env: Some("ADMIN_PASS".to_string()),
        };
        
        assert_eq!(auth.resolve_username(), Some("admin".to_string()));
        assert_eq!(auth.resolve_password(), Some("secret".to_string()));
        
        env::remove_var("ADMIN_USER");
        env::remove_var("ADMIN_PASS");
    }
}
