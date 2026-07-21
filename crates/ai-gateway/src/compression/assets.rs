//! Secure installation and inspection of ONNX model/runtime assets.

use std::{
    io::Read,
    path::{Path, PathBuf},
    sync::Arc,
    time::Duration,
};

use futures::StreamExt;
use serde::Serialize;
use sha2::{Digest, Sha256};
use tokio::{fs, io::AsyncWriteExt, sync::Mutex};

const MODEL_REVISION: &str = "eacbdc589d039a1a39f76a92844979ad4e266bcd";
const MODEL_BASE_URL: &str =
    "https://huggingface.co/chopratejas/kompress-small/resolve/eacbdc589d039a1a39f76a92844979ad4e266bcd";
const RUNTIME_VERSION: &str = "1.22.0";
const RUNTIME_BASE_URL: &str = "https://github.com/microsoft/onnxruntime/releases/download/v1.22.0";
const DOWNLOAD_TIMEOUT: Duration = Duration::from_secs(15 * 60);
const RUNTIME_DIRECTORY: &str = ".onnxruntime";

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize)]
#[serde(rename_all = "snake_case")]
pub enum RuntimeMode {
    Dynamic,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize)]
pub struct OnnxAssetStatus {
    pub configured_model_path: String,
    pub resolved_model_path: String,
    pub model_available: bool,
    pub model_valid: bool,
    pub tokenizer_available: bool,
    pub external_data_available: bool,
    pub runtime_available: bool,
    pub runtime_mode: RuntimeMode,
    pub runtime_path: String,
    pub ready: bool,
    pub missing_assets: Vec<String>,
    pub invalid_assets: Vec<String>,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize)]
pub struct OnnxInstallResult {
    pub installed: Vec<String>,
    pub skipped: Vec<String>,
    pub status: OnnxAssetStatus,
}

#[derive(Debug, thiserror::Error)]
pub enum OnnxAssetError {
    #[error("ONNX model path must not be empty or contain NUL characters")]
    InvalidModelPath,
    #[error("ONNX assets are not available for target {target}")]
    UnsupportedPlatform { target: String },
    #[error("unsafe ONNX asset destination `{path}`: expected a regular file or a missing path")]
    UnsafeDestination { path: PathBuf },
    #[error("failed to {operation} ONNX asset `{path}`: {source}")]
    Io {
        operation: &'static str,
        path: PathBuf,
        #[source]
        source: std::io::Error,
    },
    #[error("failed to download ONNX asset `{asset}`: {message}")]
    Download { asset: String, message: String },
    #[error("ONNX asset `{asset}` exceeded its {maximum_bytes} byte limit")]
    TooLarge { asset: String, maximum_bytes: u64 },
    #[error("ONNX asset `{asset}` failed integrity validation")]
    Integrity { asset: String },
    #[error("failed to unpack ONNX runtime: {0}")]
    Archive(String),
}

impl OnnxAssetError {
    pub fn is_permission_denied(&self) -> bool {
        matches!(self, Self::Io { source, .. } if source.kind() == std::io::ErrorKind::PermissionDenied)
    }
}

#[derive(Debug, Clone, Copy)]
enum Destination {
    Graph,
    ExternalData,
    Tokenizer,
}

#[derive(Debug, Clone)]
struct ManagedFile {
    name: &'static str,
    url: String,
    sha256: &'static str,
    exact_size: u64,
    destination: Destination,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum RuntimeArchiveKind {
    Zip,
    TarGz,
}

#[derive(Debug, Clone)]
struct RuntimeArchive {
    name: &'static str,
    url: String,
    sha256: &'static str,
    exact_size: u64,
    kind: RuntimeArchiveKind,
    library_suffix: &'static str,
}

#[derive(Clone)]
pub struct OnnxAssetManager {
    client: reqwest::Client,
    install_lock: Arc<Mutex<()>>,
    files: Arc<Vec<ManagedFile>>,
    runtime: RuntimeArchive,
}

impl std::fmt::Debug for OnnxAssetManager {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        formatter
            .debug_struct("OnnxAssetManager")
            .field("model_revision", &MODEL_REVISION)
            .field("runtime_version", &RUNTIME_VERSION)
            .finish_non_exhaustive()
    }
}

impl OnnxAssetManager {
    pub fn new() -> Result<Self, OnnxAssetError> {
        let client = reqwest::Client::builder()
            .timeout(DOWNLOAD_TIMEOUT)
            .redirect(reqwest::redirect::Policy::custom(|attempt| {
                if attempt.url().scheme() != "https" {
                    attempt.stop()
                } else if attempt.previous().len() > 10 {
                    attempt.error("too many asset redirects")
                } else {
                    attempt.follow()
                }
            }))
            .build()
            .map_err(|error| OnnxAssetError::Download {
                asset: "HTTP client".to_owned(),
                message: error.to_string(),
            })?;
        Ok(Self {
            client,
            install_lock: Arc::new(Mutex::new(())),
            files: Arc::new(production_files()),
            runtime: production_runtime()?,
        })
    }

    pub fn resolve_model_path(model_path: &Path) -> Result<PathBuf, OnnxAssetError> {
        validate_model_path(model_path)?;
        if model_path.is_absolute() {
            Ok(model_path.to_path_buf())
        } else {
            std::env::current_dir()
                .map(|directory| directory.join(model_path))
                .map_err(|source| OnnxAssetError::Io {
                    operation: "resolve",
                    path: model_path.to_path_buf(),
                    source,
                })
        }
    }

    pub fn tokenizer_path(model_path: &Path) -> PathBuf {
        model_parent(model_path).join("tokenizer.json")
    }

    pub fn external_data_path(model_path: &Path) -> PathBuf {
        model_parent(model_path).join("model.onnx.data")
    }

    pub fn runtime_library_path(model_path: &Path) -> Result<PathBuf, OnnxAssetError> {
        production_runtime()?;
        Ok(model_parent(model_path)
            .join(RUNTIME_DIRECTORY)
            .join(runtime_library_file_name()))
    }

    pub async fn status(&self, model_path: &Path) -> Result<OnnxAssetStatus, OnnxAssetError> {
        let resolved = Self::resolve_model_path(model_path)?;
        let graph = inspect_file(&resolved, expected_size(&self.files, Destination::Graph)).await?;
        let tokenizer_path = Self::tokenizer_path(&resolved);
        let tokenizer = inspect_file(
            &tokenizer_path,
            expected_size(&self.files, Destination::Tokenizer),
        )
        .await?;
        let external_data_path = Self::external_data_path(&resolved);
        let external_data = inspect_file(
            &external_data_path,
            expected_size(&self.files, Destination::ExternalData),
        )
        .await?;
        let runtime_path = model_parent(&resolved)
            .join(RUNTIME_DIRECTORY)
            .join(runtime_library_file_name());
        let runtime = inspect_file(&runtime_path, None).await?;

        let mut missing_assets = Vec::new();
        let mut invalid_assets = Vec::new();
        classify_status("model", graph, &mut missing_assets, &mut invalid_assets);
        classify_status(
            "external_data",
            external_data,
            &mut missing_assets,
            &mut invalid_assets,
        );
        classify_status(
            "tokenizer",
            tokenizer,
            &mut missing_assets,
            &mut invalid_assets,
        );
        classify_status("runtime", runtime, &mut missing_assets, &mut invalid_assets);

        Ok(OnnxAssetStatus {
            configured_model_path: model_path.display().to_string(),
            resolved_model_path: resolved.display().to_string(),
            model_available: graph != FileState::Missing,
            model_valid: graph == FileState::Valid,
            tokenizer_available: tokenizer == FileState::Valid,
            external_data_available: external_data == FileState::Valid,
            runtime_available: runtime == FileState::Valid,
            runtime_mode: RuntimeMode::Dynamic,
            runtime_path: runtime_path.display().to_string(),
            ready: missing_assets.is_empty() && invalid_assets.is_empty(),
            missing_assets,
            invalid_assets,
        })
    }

    pub async fn install(&self, model_path: &Path) -> Result<OnnxInstallResult, OnnxAssetError> {
        let _guard = self.install_lock.lock().await;
        let resolved = Self::resolve_model_path(model_path)?;
        let parent = model_parent(&resolved);
        fs::create_dir_all(&parent)
            .await
            .map_err(|source| OnnxAssetError::Io {
                operation: "create model directory for",
                path: parent.clone(),
                source,
            })?;

        let mut installed = Vec::new();
        let mut skipped = Vec::new();
        for asset in self.files.iter() {
            let destination = destination_path(asset.destination, &resolved);
            match inspect_file(&destination, Some(asset.exact_size)).await? {
                FileState::Valid => skipped.push(asset.name.to_owned()),
                FileState::Missing | FileState::Invalid => {
                    self.download_file(asset, &destination).await?;
                    installed.push(asset.name.to_owned());
                }
            }
        }

        let runtime_path = parent
            .join(RUNTIME_DIRECTORY)
            .join(runtime_library_file_name());
        match inspect_file(&runtime_path, None).await? {
            FileState::Valid => skipped.push("runtime".to_owned()),
            FileState::Missing | FileState::Invalid => {
                self.install_runtime(&parent).await?;
                installed.push("runtime".to_owned());
            }
        }

        let status = self.status(model_path).await?;
        Ok(OnnxInstallResult {
            installed,
            skipped,
            status,
        })
    }

    async fn download_file(
        &self,
        asset: &ManagedFile,
        destination: &Path,
    ) -> Result<(), OnnxAssetError> {
        self.download_to_file(
            asset.name,
            &asset.url,
            asset.exact_size,
            asset.sha256,
            destination,
        )
        .await
    }

    async fn install_runtime(&self, model_parent: &Path) -> Result<(), OnnxAssetError> {
        let runtime_dir = model_parent.join(RUNTIME_DIRECTORY);
        fs::create_dir_all(&runtime_dir)
            .await
            .map_err(|source| OnnxAssetError::Io {
                operation: "create runtime directory for",
                path: runtime_dir.clone(),
                source,
            })?;
        let archive_path = runtime_dir.join(format!(".runtime-{}.archive", uuid::Uuid::new_v4()));
        self.download_to_file(
            self.runtime.name,
            &self.runtime.url,
            self.runtime.exact_size,
            self.runtime.sha256,
            &archive_path,
        )
        .await?;
        let kind = self.runtime.kind;
        let suffix = self.runtime.library_suffix;
        let extraction_path = archive_path.clone();
        let library = tokio::task::spawn_blocking(move || {
            extract_runtime_library_from_path(&extraction_path, kind, suffix)
        })
        .await
        .map_err(|error| OnnxAssetError::Archive(error.to_string()))?;
        let _ = fs::remove_file(&archive_path).await;
        let library = library?;
        atomic_write(&runtime_dir.join(runtime_library_file_name()), &library).await
    }

    async fn download_to_file(
        &self,
        name: &str,
        url: &str,
        exact_size: u64,
        expected_sha256: &str,
        destination: &Path,
    ) -> Result<(), OnnxAssetError> {
        let response =
            self.client
                .get(url)
                .send()
                .await
                .map_err(|error| OnnxAssetError::Download {
                    asset: name.to_owned(),
                    message: error.to_string(),
                })?;
        if !response.status().is_success() {
            return Err(OnnxAssetError::Download {
                asset: name.to_owned(),
                message: format!("upstream HTTP {}", response.status()),
            });
        }
        if response.url().scheme() != "https" {
            return Err(OnnxAssetError::Download {
                asset: name.to_owned(),
                message: "asset redirect resolved to a non-HTTPS URL".to_owned(),
            });
        }

        let parent = model_parent(destination);
        fs::create_dir_all(&parent)
            .await
            .map_err(|source| OnnxAssetError::Io {
                operation: "create download directory for",
                path: destination.to_path_buf(),
                source,
            })?;
        validate_replaceable_destination(destination).await?;
        let temporary = parent.join(format!(
            ".{}.{}.download",
            destination
                .file_name()
                .and_then(|name| name.to_str())
                .unwrap_or("onnx"),
            uuid::Uuid::new_v4()
        ));
        let mut file = fs::File::create(&temporary)
            .await
            .map_err(|source| OnnxAssetError::Io {
                operation: "create temporary download for",
                path: destination.to_path_buf(),
                source,
            })?;
        let mut stream = response.bytes_stream();
        let mut received = 0u64;
        let mut digest = Sha256::new();
        let download_result = async {
            while let Some(chunk) = stream.next().await {
                let chunk = chunk.map_err(|error| OnnxAssetError::Download {
                    asset: name.to_owned(),
                    message: error.to_string(),
                })?;
                received = received.saturating_add(chunk.len() as u64);
                if received > exact_size {
                    return Err(OnnxAssetError::TooLarge {
                        asset: name.to_owned(),
                        maximum_bytes: exact_size,
                    });
                }
                digest.update(&chunk);
                file.write_all(&chunk)
                    .await
                    .map_err(|source| OnnxAssetError::Io {
                        operation: "write temporary download for",
                        path: destination.to_path_buf(),
                        source,
                    })?;
            }
            file.flush().await.map_err(|source| OnnxAssetError::Io {
                operation: "flush temporary download for",
                path: destination.to_path_buf(),
                source,
            })?;
            drop(file);
            if received != exact_size || format!("{:x}", digest.finalize()) != expected_sha256 {
                return Err(OnnxAssetError::Integrity {
                    asset: name.to_owned(),
                });
            }
            activate_temporary(&temporary, destination).await
        }
        .await;
        if download_result.is_err() {
            let _ = fs::remove_file(&temporary).await;
        }
        download_result
    }
}

fn production_files() -> Vec<ManagedFile> {
    vec![
        ManagedFile {
            name: "model",
            url: format!("{MODEL_BASE_URL}/model.onnx"),
            sha256: "75e35f4cf3e9e7352b33c8b38999f1ea6c4d9c296da954de48b08d1d6ed1e098",
            exact_size: 532_452,
            destination: Destination::Graph,
        },
        ManagedFile {
            name: "external_data",
            url: format!("{MODEL_BASE_URL}/model.onnx.data"),
            sha256: "61fd594d55fb74c339e3156b143317a4e67c2355a36d392bda0bc77922be9e5e",
            exact_size: 275_120_128,
            destination: Destination::ExternalData,
        },
        ManagedFile {
            name: "tokenizer",
            url: format!("{MODEL_BASE_URL}/tokenizer.json"),
            sha256: "6c8aaa9a542084f2457eab775d4eeb51f92a70c0fd9de28d5edb0ddec3c08d30",
            exact_size: 3_583_228,
            destination: Destination::Tokenizer,
        },
    ]
}

fn production_runtime() -> Result<RuntimeArchive, OnnxAssetError> {
    let target = format!("{}-{}", std::env::consts::OS, std::env::consts::ARCH);
    let runtime = match (std::env::consts::OS, std::env::consts::ARCH) {
        ("windows", "x86_64") => RuntimeArchive {
            name: "runtime",
            url: format!("{RUNTIME_BASE_URL}/onnxruntime-win-x64-{RUNTIME_VERSION}.zip"),
            sha256: "174c616efc0271194488642a72f1a514e01487da4dfe84c49296d66e40ebe0da",
            exact_size: 72_368_545,
            kind: RuntimeArchiveKind::Zip,
            library_suffix: "/lib/onnxruntime.dll",
        },
        ("linux", "x86_64") => RuntimeArchive {
            name: "runtime",
            url: format!("{RUNTIME_BASE_URL}/onnxruntime-linux-x64-{RUNTIME_VERSION}.tgz"),
            sha256: "8344d55f93d5bc5021ce342db50f62079daf39aaafb5d311a451846228be49b3",
            exact_size: 7_798_730,
            kind: RuntimeArchiveKind::TarGz,
            library_suffix: "/lib/libonnxruntime.so.1.22.0",
        },
        ("linux", "aarch64") => RuntimeArchive {
            name: "runtime",
            url: format!("{RUNTIME_BASE_URL}/onnxruntime-linux-aarch64-{RUNTIME_VERSION}.tgz"),
            sha256: "bb76395092d150b52c7092dc6b8f2fe4d80f0f3bf0416d2f269193e347e24702",
            exact_size: 6_849_865,
            kind: RuntimeArchiveKind::TarGz,
            library_suffix: "/lib/libonnxruntime.so.1.22.0",
        },
        _ => return Err(OnnxAssetError::UnsupportedPlatform { target }),
    };
    Ok(runtime)
}

fn runtime_library_file_name() -> &'static str {
    if cfg!(target_os = "windows") {
        "onnxruntime.dll"
    } else if cfg!(target_os = "macos") {
        "libonnxruntime.dylib"
    } else {
        "libonnxruntime.so"
    }
}

fn destination_path(destination: Destination, model_path: &Path) -> PathBuf {
    match destination {
        Destination::Graph => model_path.to_path_buf(),
        Destination::ExternalData => OnnxAssetManager::external_data_path(model_path),
        Destination::Tokenizer => OnnxAssetManager::tokenizer_path(model_path),
    }
}

fn expected_size(files: &[ManagedFile], destination: Destination) -> Option<u64> {
    files
        .iter()
        .find(|file| {
            std::mem::discriminant(&file.destination) == std::mem::discriminant(&destination)
        })
        .map(|file| file.exact_size)
}

fn model_parent(model_path: &Path) -> PathBuf {
    model_path
        .parent()
        .filter(|parent| !parent.as_os_str().is_empty())
        .unwrap_or_else(|| Path::new("."))
        .to_path_buf()
}

fn validate_model_path(path: &Path) -> Result<(), OnnxAssetError> {
    if path.as_os_str().is_empty() || path.to_string_lossy().contains('\0') {
        Err(OnnxAssetError::InvalidModelPath)
    } else {
        Ok(())
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum FileState {
    Missing,
    Valid,
    Invalid,
}

async fn inspect_file(path: &Path, exact_size: Option<u64>) -> Result<FileState, OnnxAssetError> {
    match fs::symlink_metadata(path).await {
        Ok(metadata) if metadata.file_type().is_symlink() || !metadata.is_file() => {
            Ok(FileState::Invalid)
        }
        Ok(metadata) if exact_size.is_some_and(|size| metadata.len() != size) => {
            Ok(FileState::Invalid)
        }
        Ok(_) => Ok(FileState::Valid),
        Err(error) if error.kind() == std::io::ErrorKind::NotFound => Ok(FileState::Missing),
        Err(source) => Err(OnnxAssetError::Io {
            operation: "inspect",
            path: path.to_path_buf(),
            source,
        }),
    }
}

fn classify_status(
    name: &str,
    state: FileState,
    missing: &mut Vec<String>,
    invalid: &mut Vec<String>,
) {
    match state {
        FileState::Missing => missing.push(name.to_owned()),
        FileState::Invalid => invalid.push(name.to_owned()),
        FileState::Valid => {}
    }
}

async fn validate_replaceable_destination(path: &Path) -> Result<(), OnnxAssetError> {
    if let Ok(metadata) = fs::symlink_metadata(path).await {
        if metadata.file_type().is_symlink() || !metadata.is_file() {
            return Err(OnnxAssetError::UnsafeDestination {
                path: path.to_path_buf(),
            });
        }
    }
    Ok(())
}

async fn activate_temporary(temporary: &Path, destination: &Path) -> Result<(), OnnxAssetError> {
    validate_replaceable_destination(destination).await?;
    if fs::metadata(destination).await.is_ok() {
        fs::remove_file(destination)
            .await
            .map_err(|source| OnnxAssetError::Io {
                operation: "replace",
                path: destination.to_path_buf(),
                source,
            })?;
    }
    fs::rename(temporary, destination)
        .await
        .map_err(|source| OnnxAssetError::Io {
            operation: "activate",
            path: destination.to_path_buf(),
            source,
        })
}

async fn atomic_write(path: &Path, bytes: &[u8]) -> Result<(), OnnxAssetError> {
    validate_replaceable_destination(path).await?;
    let parent = model_parent(path);
    fs::create_dir_all(&parent)
        .await
        .map_err(|source| OnnxAssetError::Io {
            operation: "create directory for",
            path: path.to_path_buf(),
            source,
        })?;
    let temporary = parent.join(format!(
        ".{}.{}.tmp",
        path.file_name()
            .and_then(|name| name.to_str())
            .unwrap_or("onnx"),
        uuid::Uuid::new_v4()
    ));
    let result = async {
        let mut file = fs::File::create(&temporary)
            .await
            .map_err(|source| OnnxAssetError::Io {
                operation: "create temporary file for",
                path: path.to_path_buf(),
                source,
            })?;
        file.write_all(bytes)
            .await
            .map_err(|source| OnnxAssetError::Io {
                operation: "write temporary file for",
                path: path.to_path_buf(),
                source,
            })?;
        file.flush().await.map_err(|source| OnnxAssetError::Io {
            operation: "flush temporary file for",
            path: path.to_path_buf(),
            source,
        })?;
        drop(file);
        activate_temporary(&temporary, path).await
    }
    .await;
    if result.is_err() {
        let _ = fs::remove_file(&temporary).await;
    }
    result
}

fn extract_runtime_library_from_path(
    archive_path: &Path,
    kind: RuntimeArchiveKind,
    suffix: &str,
) -> Result<Vec<u8>, OnnxAssetError> {
    let file = std::fs::File::open(archive_path).map_err(|source| OnnxAssetError::Io {
        operation: "open runtime archive",
        path: archive_path.to_path_buf(),
        source,
    })?;
    extract_runtime_library(file, kind, suffix)
}

fn extract_runtime_library<R: Read + std::io::Seek>(
    archive: R,
    kind: RuntimeArchiveKind,
    suffix: &str,
) -> Result<Vec<u8>, OnnxAssetError> {
    match kind {
        RuntimeArchiveKind::Zip => {
            let mut archive = zip::ZipArchive::new(archive)
                .map_err(|error| OnnxAssetError::Archive(error.to_string()))?;
            for index in 0..archive.len() {
                let mut entry = archive
                    .by_index(index)
                    .map_err(|error| OnnxAssetError::Archive(error.to_string()))?;
                if entry.name().replace('\\', "/").ends_with(suffix) && entry.is_file() {
                    let mut bytes = Vec::with_capacity(entry.size() as usize);
                    entry
                        .read_to_end(&mut bytes)
                        .map_err(|error| OnnxAssetError::Archive(error.to_string()))?;
                    return Ok(bytes);
                }
            }
        }
        RuntimeArchiveKind::TarGz => {
            let decoder = flate2::read::GzDecoder::new(archive);
            let mut archive = tar::Archive::new(decoder);
            let entries = archive
                .entries()
                .map_err(|error| OnnxAssetError::Archive(error.to_string()))?;
            for entry in entries {
                let mut entry =
                    entry.map_err(|error| OnnxAssetError::Archive(error.to_string()))?;
                let path = entry
                    .path()
                    .map_err(|error| OnnxAssetError::Archive(error.to_string()))?;
                if path.to_string_lossy().replace('\\', "/").ends_with(suffix)
                    && entry.header().entry_type().is_file()
                {
                    let mut bytes = Vec::new();
                    entry
                        .read_to_end(&mut bytes)
                        .map_err(|error| OnnxAssetError::Archive(error.to_string()))?;
                    return Ok(bytes);
                }
            }
        }
    }
    Err(OnnxAssetError::Archive(format!(
        "runtime library `{suffix}` was not found"
    )))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn companion_paths_are_beside_configured_graph() {
        let graph = Path::new("models/custom-name.onnx");
        assert_eq!(
            OnnxAssetManager::tokenizer_path(graph),
            PathBuf::from("models/tokenizer.json")
        );
        assert_eq!(
            OnnxAssetManager::external_data_path(graph),
            PathBuf::from("models/model.onnx.data")
        );
    }

    #[tokio::test]
    async fn status_reports_missing_assets_without_creating_files() {
        let directory = tempfile::tempdir().unwrap();
        let model = directory.path().join("model.onnx");
        let manager = OnnxAssetManager::new().unwrap();
        let status = manager.status(&model).await.unwrap();
        assert!(!status.ready);
        assert_eq!(
            status.missing_assets,
            ["model", "external_data", "tokenizer", "runtime"]
        );
        assert!(!model.exists());
    }

    #[tokio::test]
    async fn status_rejects_wrong_sizes_and_unsafe_objects() {
        let directory = tempfile::tempdir().unwrap();
        let model = directory.path().join("model.onnx");
        fs::write(&model, b"wrong").await.unwrap();
        fs::create_dir(directory.path().join("tokenizer.json"))
            .await
            .unwrap();
        let manager = OnnxAssetManager::new().unwrap();
        let status = manager.status(&model).await.unwrap();
        assert!(status.invalid_assets.contains(&"model".to_owned()));
        assert!(status.invalid_assets.contains(&"tokenizer".to_owned()));
    }

    #[tokio::test]
    async fn atomic_write_replaces_regular_file_and_leaves_no_temp() {
        let directory = tempfile::tempdir().unwrap();
        let destination = directory.path().join("asset.bin");
        fs::write(&destination, b"old").await.unwrap();
        atomic_write(&destination, b"new").await.unwrap();
        assert_eq!(fs::read(&destination).await.unwrap(), b"new");
        let mut entries = fs::read_dir(directory.path()).await.unwrap();
        let mut names = Vec::new();
        while let Some(entry) = entries.next_entry().await.unwrap() {
            names.push(entry.file_name());
        }
        assert_eq!(names, [std::ffi::OsString::from("asset.bin")]);
    }

    #[tokio::test]
    async fn atomic_write_rejects_directory_destination() {
        let directory = tempfile::tempdir().unwrap();
        let destination = directory.path().join("asset.bin");
        fs::create_dir(&destination).await.unwrap();
        assert!(matches!(
            atomic_write(&destination, b"new").await,
            Err(OnnxAssetError::UnsafeDestination { .. })
        ));
    }

    #[test]
    fn production_manifest_is_pinned_and_https_only() {
        for file in production_files() {
            assert!(file.url.starts_with("https://"));
            assert_eq!(file.sha256.len(), 64);
            assert!(file.exact_size > 0);
        }
        let runtime = production_runtime().unwrap();
        assert!(runtime.url.starts_with("https://"));
        assert_eq!(runtime.sha256.len(), 64);
    }

    #[test]
    fn downloaded_runtime_archive_contains_expected_library() {
        let runtime = production_runtime().unwrap();
        let fixture = std::env::var("ONNX_RUNTIME_ARCHIVE_FIXTURE").ok();
        if let Some(path) = fixture {
            let file = std::fs::File::open(path).unwrap();
            assert!(
                !extract_runtime_library(file, runtime.kind, runtime.library_suffix)
                    .unwrap()
                    .is_empty()
            );
        }
    }
}
