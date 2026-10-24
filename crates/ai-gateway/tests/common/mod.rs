//! Shared test utilities for integration tests.
//!
//! Each integration-test file in `tests/` is compiled as its own crate, so
//! they cannot share code through `src/`. A file at `tests/common/mod.rs` is
//! the Cargo-blessed way to share helpers across test binaries.
//!
//! ## Purpose
//!
//! Every `GatewayServer::new` opens three SQLite databases:
//! - `logging.database_path` (default `./logs.db`)
//! - a sibling `assistants.db`
//! - `virtual_keys.database_path` (default `./keys.db`)
//!
//! When multiple tests run concurrently inside the *same* test binary, their
//! `GatewayServer` instances share the same default files, producing
//! SQL-level lock contention and file cleanup races. This module's
//! [`isolate_databases`] redirects every database into a unique temporary
//! directory so no two servers ever touch the same file.
//!
//! Every test binary embeds this module in full but uses only a subset of
//! its helpers; unused ones would trip `dead_code` in the other binaries
//! (e.g. `budget_persistence` uses [`budget_test_config`] but keeps stable
//! database paths instead of calling [`isolate_databases`]). The lint is
//! scoped here rather than at each embedding site.
#![allow(dead_code)]

use ai_gateway::config::*;

/// Redirect the logging / assistants / virtual-key SQLite databases into a
/// unique temporary directory so parallel tests never share `./logs.db`.
///
/// The directory is **intentionally leaked**: the gateway holds open SQLite
/// handles for the lifetime of the test process, and removing the directory
/// before those handles close causes Windows delete-failures. The OS temp
/// cleaner reclaims the (tiny) files eventually.
pub fn isolate_databases(config: &mut Config) {
    let dir = tempfile::tempdir()
        .expect("create temp dir for isolated test databases")
        .keep();
    config.logging.database_path = dir.join("logs.db").to_string_lossy().into_owned();
    config.virtual_keys.database_path = dir.join("keys.db").to_string_lossy().into_owned();
}

/// Minimal valid single-provider `Config` shared by the budget test binaries
/// (`budget_persistence`, `admin_budget_reset`): one OpenAI provider named
/// `budget-provider` in one `budget-group` exposing `gpt-4.1-nano` at
/// $30/$60 per million tokens.
///
/// - `db_dir`: `Some(dir)` redirects the logs/keys databases into `dir`
///   (stable paths across `GatewayServer` rebuilds — the persistence
///   tests); `None` keeps the defaults, for callers that apply
///   [`isolate_databases`] themselves.
/// - `budget`: the provider's budget block; `None` yields a spend-only
///   provider (no limit).
pub fn budget_test_config(
    db_dir: Option<&std::path::Path>,
    budget: Option<ProviderBudgetConfig>,
) -> Config {
    let mut config = Config {
        security: Default::default(),
        cache_aware_routing: Default::default(),
        server: ServerConfig {
            host: "127.0.0.1".to_string(),
            port: 0,
            request_timeout_seconds: 30,
            max_request_size_mb: 10,
        },
        tls: None,
        admin: AdminConfig::default(),
        dashboard: DashboardConfig::default(),
        cors: CorsConfig::default(),
        providers: vec![Provider {
            name: "budget-provider".to_string(),
            provider_type: "openai".to_string(),
            base_url: Some("https://api.example.com/v1".to_string()),
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
            buffered_upstream_streaming: None,
            rate_limit_max_wait_ms: None,
            max_connections: 10,
            rate_limit_per_minute: 0,
            custom_headers: Default::default(),
            user_agent: None,
            connection_pool: ProviderConnectionPoolConfig::default(),
            budget,
            manual_models: vec![],
            global_inference_profile: false,
            cross_region_inference: false,
            custom_vpc_endpoint: false,
            prompt_caching: false,
            compression: None,
            reasoning: true,
            codex_base_url_override: None,
            codex_model_override: None,
            instructions_override: None,
            max_rate_limit_cooldown_seconds: None,
            memory: None,
        }],
        model_groups: vec![ModelGroup {
            name: "budget-group".to_string(),
            version_fallback_enabled: false,
            compression: None,
            structured_output: None,
            memory: None,
            models: vec![ProviderModel {
                provider: "budget-provider".to_string(),
                model: "gpt-4.1-nano".to_string(),
                cost_per_million_input_tokens: 30.0,
                cost_per_million_output_tokens: 60.0,
                cache_support: None,
                cache_min_tokens: None,
                cost_per_million_cache_read_input_tokens: None,
                cost_per_million_cache_creation_input_tokens: None,
                priority: 100,
                structured_output_passthrough: None,
                tier: None,
                context_window: 0,
                specializations: vec![],
                cost_per_million_reasoning_tokens: None,
                reasoning_family: None,
                reasoning_parameter: None,
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
        first_launch_completed: false,
        tray: TrayConfig::default(),
        codex_instructions_url: None,
        streaming: None,
        virtual_keys: Default::default(),
        loop_detection: Default::default(),
        guardrails: None,
        tool_compression: Default::default(),
        smart_routing: Default::default(),
        xhigh_models_allowlist: Default::default(),
        reasoning_models_allowlist: Default::default(),
        codex_search: None,
        structured_output: None,
        memory: None,
        reasoning_compat: Default::default(),
    };
    if let Some(dir) = db_dir {
        config.logging.database_path = dir.join("logs.db").to_string_lossy().into_owned();
        config.virtual_keys.database_path = dir.join("keys.db").to_string_lossy().into_owned();
    }
    config
}
