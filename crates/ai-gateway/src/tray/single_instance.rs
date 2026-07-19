#[cfg(target_os = "windows")]
use std::ffi::OsStr;
#[cfg(target_os = "windows")]
use std::os::windows::ffi::OsStrExt;

#[cfg(target_os = "windows")]
use winapi::shared::minwindef::FALSE;
#[cfg(target_os = "windows")]
use winapi::shared::winerror::ERROR_ALREADY_EXISTS;
#[cfg(target_os = "windows")]
use winapi::um::errhandlingapi::GetLastError;
#[cfg(target_os = "windows")]
use winapi::um::handleapi::{CloseHandle, INVALID_HANDLE_VALUE};
#[cfg(target_os = "windows")]
use winapi::um::synchapi::CreateMutexW;
#[cfg(target_os = "windows")]
use winapi::um::winnt::HANDLE;

#[derive(Debug, thiserror::Error)]
pub enum InstanceError {
    #[error("failed to create application mutex")]
    MutexCreationFailed,
}

#[derive(Debug)]
pub struct SingleInstanceGuard {
    app_id: String,
    already_running: bool,
    #[cfg(target_os = "windows")]
    handle: HANDLE,
}

impl SingleInstanceGuard {
    pub fn acquire(app_id: &str) -> Result<Self, InstanceError> {
        #[cfg(target_os = "windows")]
        {
            let mut wide: Vec<u16> = OsStr::new(app_id).encode_wide().collect();
            wide.push(0);

            let handle = unsafe { CreateMutexW(std::ptr::null_mut(), FALSE, wide.as_ptr()) };
            if handle.is_null() || handle == INVALID_HANDLE_VALUE {
                return Err(InstanceError::MutexCreationFailed);
            }

            let already_running = unsafe { GetLastError() } == ERROR_ALREADY_EXISTS;

            return Ok(Self {
                app_id: app_id.to_string(),
                already_running,
                handle,
            });
        }

        #[cfg(not(target_os = "windows"))]
        {
            Ok(Self {
                app_id: app_id.to_string(),
                already_running: false,
            })
        }
    }

    pub fn is_already_running(&self) -> bool {
        self.already_running
    }

    pub fn bring_to_front(&self) {
        tracing::info!(app_id = %self.app_id, "Single-instance bring_to_front requested");
    }
}

impl Drop for SingleInstanceGuard {
    fn drop(&mut self) {
        #[cfg(target_os = "windows")]
        unsafe {
            if !self.handle.is_null() && self.handle != INVALID_HANDLE_VALUE {
                CloseHandle(self.handle);
            }
        }
    }
}
