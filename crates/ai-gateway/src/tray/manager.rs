use std::path::PathBuf;
use std::process::Command;
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::Arc;
use std::time::Duration;

use tokio::sync::{
    mpsc::{unbounded_channel, UnboundedReceiver},
    RwLock,
};

use crate::config::Config;

use super::{
    NotificationManager, ServerStatus, SingleInstanceGuard, SplashScreen, TrayAssets, TrayError,
    TrayIconHandle, TrayMenu, TrayMenuAction,
};

pub struct TrayManager {
    config: Arc<RwLock<Config>>,
    assets: TrayAssets,
    status: Arc<RwLock<ServerStatus>>,
    tray_icon: TrayIconHandle,
    tray_menu: TrayMenu,
    splash_screen: SplashScreen,
    shutdown_requested: Arc<AtomicBool>,
    instance_guard: Option<SingleInstanceGuard>,
    menu_events: UnboundedReceiver<TrayMenuAction>,
}

impl TrayManager {
    pub async fn new(config: Config) -> Result<Self, TrayError> {
        Self::with_assets(config, TrayAssets::resolve()?).await
    }

    pub async fn with_assets(config: Config, assets: TrayAssets) -> Result<Self, TrayError> {
        let status = ServerStatus::new(&config);
        let splash_duration = Duration::from_millis(config.tray.splash_duration_ms);
        let (menu_tx, menu_events) = unbounded_channel();

        Ok(Self {
            config: Arc::new(RwLock::new(config)),
            tray_icon: TrayIconHandle::new(assets.icon_path.clone(), menu_tx)?,
            tray_menu: TrayMenu::new(&status),
            splash_screen: SplashScreen::new(assets.logo_path.clone(), splash_duration),
            assets,
            status: Arc::new(RwLock::new(status)),
            shutdown_requested: Arc::new(AtomicBool::new(false)),
            instance_guard: None,
            menu_events,
        })
    }

    pub fn attach_instance_guard(&mut self, guard: SingleInstanceGuard) {
        self.instance_guard = Some(guard);
    }

    pub async fn run(&mut self) -> Result<(), TrayError> {
        tracing::info!(icon = %self.assets.icon_path.display(), "Tray manager running");

        while !self.shutdown_requested.load(Ordering::Relaxed) {
            tokio::select! {
                event = self.menu_events.recv() => {
                    if let Some(action) = event {
                        self.handle_menu_action(action).await?;
                    }
                }
                _ = tokio::time::sleep(Duration::from_millis(100)) => {}
            }
        }

        Ok(())
    }

    pub async fn show_first_launch_experience(&mut self) -> Result<(), TrayError> {
        if !self.is_first_launch().await {
            return Ok(());
        }

        self.splash_screen.show().await?;
        self.splash_screen.animate_to_tray().await;
        self.splash_screen.close();

        let status = self.status.read().await.clone();
        if self.config.read().await.tray.show_notifications {
            NotificationManager::notify_server_started(&status.address)?;
        }

        Ok(())
    }

    pub async fn is_first_launch(&self) -> bool {
        !self.config.read().await.first_launch_completed
    }

    pub async fn mark_first_launch_complete(&self) {
        self.config.write().await.first_launch_completed = true;
    }

    pub async fn set_server_running(&mut self, running: bool) {
        let mut status = self.status.write().await;
        status.is_running = running;
        self.tray_menu.update_status(&status);
        self.tray_icon.set_tooltip(if running {
            format!("OBEY API Gateway running at {}", status.address)
        } else {
            "OBEY API Gateway stopped".to_string()
        });
    }

    pub async fn server_status(&self) -> ServerStatus {
        self.status.read().await.clone()
    }

    pub async fn open_dashboard(&self) -> Result<(), TrayError> {
        let status = self.server_status().await;
        let url = status.dashboard_url();
        tracing::info!(
            address = %status.address,
            dashboard_path = %status.dashboard_path,
            is_running = status.is_running,
            url = %url,
            "Tray requested dashboard browser launch"
        );
        wait_for_http_ready(&url).await;
        open_in_browser(&url)
    }

    pub async fn handle_menu_action(&self, action: TrayMenuAction) -> Result<(), TrayError> {
        match action {
            TrayMenuAction::OpenDashboard => self.open_dashboard().await,
            TrayMenuAction::ViewServerStatus => {
                let status = self.server_status().await;
                if self.config.read().await.tray.show_notifications {
                    NotificationManager::notify_server_status(&status.address, status.is_running)?;
                }
                tracing::info!(address = %status.address, is_running = status.is_running, "Tray status requested");
                Ok(())
            }
            TrayMenuAction::Quit => {
                self.request_shutdown();
                Ok(())
            }
        }
    }

    pub fn request_shutdown(&self) {
        self.shutdown_requested.store(true, Ordering::Relaxed);
    }

    pub fn shutdown_requested(&self) -> bool {
        self.shutdown_requested.load(Ordering::Relaxed)
    }

    pub fn tray_menu(&self) -> &TrayMenu {
        &self.tray_menu
    }

    pub fn tray_icon(&self) -> &TrayIconHandle {
        &self.tray_icon
    }

    pub fn splash_logo_path(&self) -> &PathBuf {
        &self.assets.logo_path
    }

    pub fn notify_server_error(&self, error: &str) -> Result<(), TrayError> {
        NotificationManager::notify_server_error(error)
    }

    pub fn signal_existing_instance(&self) {
        if let Some(guard) = &self.instance_guard {
            if guard.is_already_running() {
                guard.bring_to_front();
            }
        }
    }
}

fn open_in_browser(url: &str) -> Result<(), TrayError> {
    #[cfg(target_os = "windows")]
    {
        Command::new("cmd")
            .args(["/C", "start", "", url])
            .spawn()
            .map_err(TrayError::Io)?;
        return Ok(());
    }

    #[cfg(target_os = "macos")]
    {
        Command::new("open")
            .arg(url)
            .spawn()
            .map_err(TrayError::Io)?;
        return Ok(());
    }

    #[cfg(all(unix, not(target_os = "macos")))]
    {
        Command::new("xdg-open")
            .arg(url)
            .spawn()
            .map_err(TrayError::Io)?;
        return Ok(());
    }

    #[allow(unreachable_code)]
    Err(TrayError::Io(std::io::Error::new(
        std::io::ErrorKind::Unsupported,
        "browser launch is not supported on this platform",
    )))
}

async fn wait_for_http_ready(url: &str) {
    let client = reqwest::Client::builder()
        .timeout(Duration::from_millis(500))
        .build();

    let Ok(client) = client else {
        tracing::warn!(url = %url, "Failed to create HTTP client while waiting for dashboard URL readiness");
        return;
    };

    tracing::info!(url = %url, "Waiting for dashboard URL to become reachable before browser launch");

    for attempt in 1..=20 {
        match client.get(url).send().await {
            Ok(response)
                if response.status().is_success() || response.status().is_redirection() =>
            {
                tracing::info!(attempt, status = %response.status(), url = %url, "Dashboard URL responded successfully before browser launch");
                return;
            }
            Ok(response) if response.status() == reqwest::StatusCode::NOT_FOUND => {
                tracing::warn!(attempt, status = %response.status(), url = %url, "Dashboard URL responded with 404 while waiting for browser launch");
            }
            Ok(response) => {
                tracing::debug!(attempt, status = %response.status(), url = %url, "Dashboard URL responded but is not ready for browser launch");
            }
            Err(error) => {
                tracing::debug!(attempt, %error, url = %url, "Dashboard URL not reachable yet");
            }
        }
        tokio::time::sleep(Duration::from_millis(250)).await;
    }

    tracing::warn!(url = %url, "Timed out waiting for dashboard URL readiness; opening browser anyway");
}
