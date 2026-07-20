// The tray feature is still being wired up: several helper methods and fields
// are defined ahead of their call sites. Allow dead_code across the tray
// submodules so an in-progress `--features tray` build stays warning-clean.
#[allow(dead_code)]
mod icon;
#[allow(dead_code)]
mod manager;
#[allow(dead_code)]
mod menu;
#[allow(dead_code)]
mod notification;
mod single_instance;
#[allow(dead_code)]
mod splash;

use std::fs;
use std::net::TcpListener;
use std::path::{Path, PathBuf};

use crate::config::Config;

pub use icon::TrayIconHandle;
pub use manager::TrayManager;
pub use menu::{TrayMenu, TrayMenuAction};
pub use notification::NotificationManager;
pub use single_instance::{InstanceError, SingleInstanceGuard};
pub use splash::SplashScreen;

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ServerStatus {
    pub is_running: bool,
    pub address: String,
    pub admin_path: String,
    pub dashboard_path: String,
}

impl ServerStatus {
    pub fn new(config: &Config) -> Self {
        Self {
            is_running: false,
            address: format!(
                "{}:{}",
                client_host_for_bind_host(&config.server.host),
                config.server.port
            ),
            admin_path: config.admin.path.clone(),
            dashboard_path: config.dashboard.path.clone(),
        }
    }

    pub fn dashboard_url(&self) -> String {
        format!(
            "http://{}{}",
            self.address,
            normalized_route_prefix(&self.dashboard_path)
        )
    }

    pub fn admin_url(&self) -> String {
        format!(
            "http://{}{}",
            self.address,
            normalized_route_prefix(&self.admin_path)
        )
    }
}

pub fn prepare_startup_config(mut config: Config, config_path: &Path) -> Result<Config, TrayError> {
    let selected_port = select_available_port(&config.server.host, config.server.port)?;

    if selected_port != config.server.port {
        let configured_port = config.server.port;
        config.server.port = selected_port;
        crate::config::save_config(config_path, &config).map_err(TrayError::Configuration)?;

        tracing::warn!(
            bind_host = %config.server.host,
            configured_port,
            selected_port,
            config_path = %config_path.display(),
            "Configured server port was unavailable; updated configuration to use a fallback port"
        );
    }

    Ok(config)
}

pub fn client_host_for_bind_host(host: &str) -> &str {
    match host {
        "0.0.0.0" | "::" | "[::]" => "127.0.0.1",
        _ => host,
    }
}

fn select_available_port(host: &str, preferred_port: u16) -> Result<u16, TrayError> {
    if port_is_available(host, preferred_port) {
        return Ok(preferred_port);
    }

    let higher_ports =
        ((preferred_port as u32 + 1).max(1024)..=u16::MAX as u32).map(|port| port as u16);
    let lower_ports = (1024..preferred_port.max(1024)).map(|port| port as u16);

    for candidate in higher_ports.chain(lower_ports) {
        if port_is_available(host, candidate) {
            return Ok(candidate);
        }
    }

    Err(TrayError::Io(std::io::Error::new(
        std::io::ErrorKind::AddrNotAvailable,
        format!("No available fallback port found for host {host}"),
    )))
}

fn port_is_available(host: &str, port: u16) -> bool {
    TcpListener::bind((host, port)).is_ok()
}

fn normalized_route_prefix(path: &str) -> String {
    if path == "/" {
        return "/".to_string();
    }

    let trimmed = path.trim_end_matches('/');
    if trimmed.is_empty() {
        "/".to_string()
    } else {
        format!("{}/", trimmed)
    }
}

#[derive(Debug, Clone)]
pub struct TrayAssets {
    pub icon_path: PathBuf,
    pub logo_path: PathBuf,
}

const EMBEDDED_ICON_BYTES: &[u8] = include_bytes!("../../../../Assets/icon.ico");
const EMBEDDED_LOGO_BYTES: &[u8] = include_bytes!("../../../../Assets/logo.jpg");

impl Default for TrayAssets {
    fn default() -> Self {
        Self::resolve().unwrap_or_else(|_| Self {
            icon_path: PathBuf::from("Assets/icon.ico"),
            logo_path: PathBuf::from("Assets/logo.jpg"),
        })
    }
}

impl TrayAssets {
    pub fn resolve() -> Result<Self, TrayError> {
        Ok(Self {
            icon_path: resolve_or_extract_asset("icon.ico", EMBEDDED_ICON_BYTES)?,
            logo_path: resolve_or_extract_asset("logo.jpg", EMBEDDED_LOGO_BYTES)?,
        })
    }
}

fn resolve_or_extract_asset(file_name: &str, contents: &[u8]) -> Result<PathBuf, TrayError> {
    let mut candidates = Vec::new();

    if let Ok(exe_path) = std::env::current_exe() {
        if let Some(exe_dir) = exe_path.parent() {
            candidates.push(exe_dir.join("Assets").join(file_name));
        }
    }

    if let Ok(current_dir) = std::env::current_dir() {
        candidates.push(current_dir.join("Assets").join(file_name));
    }

    #[cfg(target_os = "windows")]
    if let Ok(appdata) = std::env::var("APPDATA") {
        candidates.push(
            PathBuf::from(appdata)
                .join("ai-gateway")
                .join("Assets")
                .join(file_name),
        );
    }

    for candidate in &candidates {
        if candidate.exists() {
            return Ok(candidate.clone());
        }
    }

    let output_path = candidates
        .into_iter()
        .next()
        .unwrap_or_else(|| PathBuf::from("Assets").join(file_name));

    if let Some(parent) = output_path.parent() {
        fs::create_dir_all(parent)?;
    }

    fs::write(&output_path, contents)?;
    Ok(output_path)
}

#[derive(Debug, thiserror::Error)]
pub enum TrayError {
    #[error("tray icon error: {0}")]
    Icon(String),
    #[error("tray menu error: {0}")]
    Menu(String),
    #[error("configuration error: {0}")]
    Configuration(String),
    #[error("tray notification error: {0}")]
    Notification(String),
    #[allow(dead_code)] // reserved error variant; not yet constructed
    #[error("splash screen error: {0}")]
    Splash(String),
    #[error("single instance error: {0}")]
    SingleInstance(#[from] InstanceError),
    #[error("I/O error: {0}")]
    Io(#[from] std::io::Error),
    #[error("image error: {0}")]
    Image(#[from] image::ImageError),
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::config::{
        AdminConfig, CircuitBreakerConfig, ContextConfig, CorsConfig, DashboardConfig,
        ExactCacheConfig, LoggingConfig, ModelGroup, Provider, ProviderConnectionPoolConfig,
        ProviderModel, RetryConfig, ServerConfig, TrayConfig,
    };

    fn test_config(host: &str, port: u16) -> Config {
        Config {
            server: ServerConfig {
                host: host.to_string(),
                port,
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
                models: vec![ProviderModel {
                    provider: "test-provider".to_string(),
                    model: "gpt-4".to_string(),
                    cost_per_million_input_tokens: 0.0,
                    cost_per_million_output_tokens: 0.0,
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
            compression: Default::default(),
            first_launch_completed: false,
            tray: TrayConfig::default(),
            codex_instructions_url: None,
            streaming: None,
            virtual_keys: Default::default(),
            loop_detection: Default::default(),
            guardrails: None,
        }
    }

    #[test]
    fn test_server_status_uses_loopback_host_for_wildcard_bind_address() {
        let status = ServerStatus::new(&test_config("0.0.0.0", 8080));

        assert_eq!(status.address, "127.0.0.1:8080");
        assert_eq!(status.dashboard_url(), "http://127.0.0.1:8080/dashboard/");
        assert_eq!(status.admin_url(), "http://127.0.0.1:8080/admin/");
    }

    #[test]
    fn test_prepare_startup_config_reassigns_busy_port_and_persists_it() {
        let listener = TcpListener::bind(("127.0.0.1", 0)).unwrap();
        let occupied_port = listener.local_addr().unwrap().port();
        let tempdir = tempfile::tempdir().unwrap();
        let config_path = tempdir.path().join("config.yaml");
        let config = test_config("127.0.0.1", occupied_port);

        crate::config::save_config(&config_path, &config).unwrap();

        let updated = prepare_startup_config(config, &config_path).unwrap();
        let reloaded = crate::config::load_and_validate_config(&config_path).unwrap();

        assert_ne!(updated.server.port, occupied_port);
        assert_eq!(reloaded.server.port, updated.server.port);
        assert!(port_is_available("127.0.0.1", updated.server.port));
    }
}
