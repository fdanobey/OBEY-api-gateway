#![cfg(feature = "tray")]

use std::time::Duration;

use ai_gateway::config::{
    AdminConfig, Config, ContextConfig, CorsConfig, DashboardConfig, ExactCacheConfig,
    LoggingConfig, ModelGroup, Provider, ProviderConnectionPoolConfig, ProviderModel, RetryConfig,
    ServerConfig, TrayConfig,
};
use ai_gateway::tray::{
    SingleInstanceGuard, SplashScreen, TrayAssets, TrayManager, TrayMenuAction,
};

fn test_config() -> Config {
    Config {
        server: ServerConfig {
            host: "127.0.0.1".to_string(),
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
            base_url: Some("http://localhost:11434".to_string()),
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
            max_connections: 10,
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
            name: "default".to_string(),
            version_fallback_enabled: false,
            compression: None,
            structured_output: None,
            memory: None,
            models: vec![ProviderModel {
                provider: "test-provider".to_string(),
                model: "gpt-4".to_string(),
                cost_per_million_input_tokens: 0.0,
                cost_per_million_output_tokens: 0.0,
                priority: 100,
                structured_output_passthrough: None,
            }],
        }],
        circuit_breaker: Default::default(),
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
        guardrails: None,
        tool_compression: Default::default(),
        structured_output: None,
    }
}

#[tokio::test]
async fn test_tray_manager_first_launch_flow() {
    let mut manager = TrayManager::with_assets(test_config(), TrayAssets::default())
        .await
        .unwrap();

    assert!(manager.is_first_launch().await);
    manager.show_first_launch_experience().await.unwrap();
    manager.mark_first_launch_complete().await;
    assert!(!manager.is_first_launch().await);
}

#[tokio::test]
async fn test_tray_manager_updates_server_status_and_menu() {
    let mut manager = TrayManager::new(test_config()).await.unwrap();

    manager.set_server_running(true).await;
    let status = manager.server_status().await;
    assert!(status.is_running);
    assert!(manager
        .tray_menu()
        .items()
        .iter()
        .any(|item| item.label.contains("Server Running")));

    let status = manager.server_status().await;
    assert_eq!(status.dashboard_url(), "http://127.0.0.1:8080/dashboard/");
}

#[tokio::test]
async fn test_tray_menu_quit_requests_shutdown() {
    let manager = TrayManager::new(test_config()).await.unwrap();
    assert!(!manager.shutdown_requested());

    manager
        .handle_menu_action(TrayMenuAction::Quit)
        .await
        .unwrap();
    assert!(manager.shutdown_requested());
}

#[tokio::test]
async fn test_splash_screen_reads_logo_dimensions() {
    let assets = TrayAssets::resolve().unwrap();
    let mut splash = SplashScreen::new(assets.logo_path, Duration::from_millis(10));
    splash.show().await.unwrap();
    assert!(splash.is_visible());
    assert!(splash.dimensions().is_some());
    splash.animate_to_tray().await;
    splash.close();
    assert!(!splash.is_visible());
}

#[test]
fn test_single_instance_guard_acquires_once() {
    let guard = SingleInstanceGuard::acquire("obey-api-gateway-test").unwrap();
    assert!(!guard.is_already_running());
}
