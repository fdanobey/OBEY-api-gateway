use std::path::{Path, PathBuf};

#[cfg(target_os = "windows")]
use std::os::windows::ffi::OsStrExt;

#[cfg(target_os = "windows")]
use tokio::sync::mpsc::UnboundedSender;

#[cfg(target_os = "windows")]
use tray_item::{IconSource, TrayItem};

#[cfg(target_os = "windows")]
use winapi::um::winuser::{LoadImageW, IMAGE_ICON, LR_DEFAULTSIZE, LR_LOADFROMFILE};

use super::{TrayError, TrayMenuAction};

pub struct TrayIconHandle {
    icon_path: PathBuf,
    tooltip: String,
    #[cfg(target_os = "windows")]
    tray: TrayItem,
}

impl TrayIconHandle {
    pub fn new(
        icon_path: impl Into<PathBuf>,
        #[cfg(target_os = "windows")] event_tx: UnboundedSender<TrayMenuAction>,
    ) -> Result<Self, TrayError> {
        let icon_path = icon_path.into();
        if !icon_path.exists() {
            return Err(TrayError::Icon(format!(
                "Tray icon file not found: {}",
                icon_path.display()
            )));
        }

        #[cfg(target_os = "windows")]
        let mut tray = {
            let mut tray = TrayItem::new("OBEY API Gateway", load_icon_source(&icon_path)?)
                .map_err(|error| TrayError::Icon(error.to_string()))?;

            let open_dashboard_tx = event_tx.clone();
            tray.add_menu_item("Open Admin Dashboard", move || {
                let _ = open_dashboard_tx.send(TrayMenuAction::OpenDashboard);
            })
            .map_err(|error| TrayError::Menu(error.to_string()))?;

            let status_tx = event_tx.clone();
            tray.add_menu_item("View Server Status", move || {
                let _ = status_tx.send(TrayMenuAction::ViewServerStatus);
            })
            .map_err(|error| TrayError::Menu(error.to_string()))?;

            let quit_tx = event_tx.clone();
            tray.add_menu_item("Quit", move || {
                let _ = quit_tx.send(TrayMenuAction::Quit);
            })
            .map_err(|error| TrayError::Menu(error.to_string()))?;

            tray
        };

        #[cfg(target_os = "windows")]
        tray.inner_mut()
            .set_tooltip("OBEY API Gateway")
            .map_err(|error| TrayError::Icon(error.to_string()))?;

        Ok(Self {
            icon_path,
            tooltip: "OBEY API Gateway".to_string(),
            #[cfg(target_os = "windows")]
            tray,
        })
    }

    pub fn set_tooltip(&mut self, tooltip: impl Into<String>) {
        let tooltip = tooltip.into();

        #[cfg(target_os = "windows")]
        if let Err(error) = self.tray.inner_mut().set_tooltip(&tooltip) {
            tracing::warn!(%error, "Failed to update tray tooltip");
        }

        self.tooltip = tooltip;
    }

    pub fn update_icon(&mut self, icon_path: impl AsRef<Path>) -> Result<(), TrayError> {
        let icon_path = icon_path.as_ref();
        if !icon_path.exists() {
            return Err(TrayError::Icon(format!(
                "Updated tray icon file not found: {}",
                icon_path.display()
            )));
        }

        self.icon_path = icon_path.to_path_buf();

        #[cfg(target_os = "windows")]
        self.tray
            .set_icon(load_icon_source(&self.icon_path)?)
            .map_err(|error| TrayError::Icon(error.to_string()))?;

        Ok(())
    }

    pub fn tooltip(&self) -> &str {
        &self.tooltip
    }

    pub fn icon_path(&self) -> &Path {
        &self.icon_path
    }
}

#[cfg(target_os = "windows")]
fn load_icon_source(icon_path: &Path) -> Result<IconSource, TrayError> {
    let mut wide_path: Vec<u16> = icon_path.as_os_str().encode_wide().collect();
    wide_path.push(0);

    let handle = unsafe {
        LoadImageW(
            std::ptr::null_mut(),
            wide_path.as_ptr(),
            IMAGE_ICON,
            0,
            0,
            LR_LOADFROMFILE | LR_DEFAULTSIZE,
        )
    };

    if handle.is_null() {
        return Err(TrayError::Icon(format!(
            "Failed to load tray icon from {}",
            icon_path.display()
        )));
    }

    Ok(IconSource::RawIcon(handle as _))
}
