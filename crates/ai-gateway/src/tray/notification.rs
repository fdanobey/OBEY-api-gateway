use super::TrayError;

#[cfg(target_os = "windows")]
use winrt_notification::{Duration as ToastDuration, Sound, Toast};

#[derive(Debug, Default, Clone, Copy)]
pub struct NotificationManager;

impl NotificationManager {
    pub fn notify_server_started(address: &str) -> Result<(), TrayError> {
        #[cfg(target_os = "windows")]
        show_windows_toast("OBEY API Gateway", "Server is now running", address)?;

        tracing::info!(%address, "Tray notification: server started");
        Ok(())
    }

    pub fn notify_server_status(address: &str, is_running: bool) -> Result<(), TrayError> {
        let status = if is_running { "Running" } else { "Stopped" };

        #[cfg(target_os = "windows")]
        show_windows_toast("OBEY API Gateway status", status, address)?;

        tracing::info!(%address, %status, "Tray notification: server status");
        Ok(())
    }

    pub fn notify_server_error(error: &str) -> Result<(), TrayError> {
        #[cfg(target_os = "windows")]
        show_windows_toast("OBEY API Gateway error", error, "")?;

        tracing::warn!(%error, "Tray notification: server error");
        Ok(())
    }

    pub fn notify_already_running() -> Result<(), TrayError> {
        #[cfg(target_os = "windows")]
        show_windows_toast(
            "OBEY API Gateway",
            "Application is already running",
            "Use the tray icon to interact with it",
        )?;

        tracing::info!("Tray notification: application is already running");
        Ok(())
    }
}

#[cfg(target_os = "windows")]
fn show_windows_toast(title: &str, line1: &str, line2: &str) -> Result<(), TrayError> {
    let mut toast = Toast::new(Toast::POWERSHELL_APP_ID)
        .title(title)
        .text1(line1)
        .sound(Some(Sound::Default))
        .duration(ToastDuration::Short);

    if !line2.is_empty() {
        toast = toast.text2(line2);
    }

    toast
        .show()
        .map_err(|error| TrayError::Notification(error.to_string()))
}
