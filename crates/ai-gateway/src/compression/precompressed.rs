//! Safe management of offline pre-compressed context artifacts.

use std::{
    collections::HashSet,
    ffi::OsString,
    fmt::Write as _,
    fs::{self, File, OpenOptions},
    io::{self, Read, Write},
    path::{Component, Path, PathBuf},
    sync::atomic::{AtomicU64, Ordering},
};

use chrono::{DateTime, Utc};
use ring::digest::{digest, SHA256};
use serde::{Deserialize, Serialize};
use thiserror::Error;

use super::{config::PrecompressedEntry, CompressionLevel};

/// Current on-disk metadata schema version.
pub const PRECOMPRESSED_FORMAT_VERSION: u32 = 1;
/// Default upper bound for an original or compressed context file.
pub const DEFAULT_MAX_CONTEXT_FILE_BYTES: u64 = 16 * 1024 * 1024;
/// Required target reduction for natural-language pre-compression.
pub const TARGET_REDUCTION_PERCENT: u64 = 40;

const MAX_METADATA_FILE_BYTES: u64 = 64 * 1024;
static TEMP_FILE_SEQUENCE: AtomicU64 = AtomicU64::new(0);

/// Metadata stored at `<compressed-path>.metadata.json`.
///
/// The sidecar intentionally contains no source text, compressed text, paths, or
/// credentials.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct PrecompressedMetadata {
    pub format_version: u32,
    pub original_tokens: u64,
    pub compressed_tokens: u64,
    pub level: CompressionLevel,
    pub timestamp: DateTime<Utc>,
    pub source_sha256: String,
}

impl PrecompressedMetadata {
    /// Builds metadata from the original bytes and validates its token counts.
    pub fn for_source(
        original_tokens: u64,
        compressed_tokens: u64,
        level: CompressionLevel,
        timestamp: DateTime<Utc>,
        source: &[u8],
    ) -> Result<Self, PrecompressedError> {
        let metadata = Self {
            format_version: PRECOMPRESSED_FORMAT_VERSION,
            original_tokens,
            compressed_tokens,
            level,
            timestamp,
            source_sha256: sha256_hex(source),
        };
        metadata.validate()?;
        Ok(metadata)
    }

    /// Validates schema compatibility, hash shape, and token-count invariants.
    pub fn validate(&self) -> Result<(), PrecompressedError> {
        if self.format_version != PRECOMPRESSED_FORMAT_VERSION {
            return Err(PrecompressedError::InvalidMetadata(format!(
                "unsupported format version {}; expected {PRECOMPRESSED_FORMAT_VERSION}",
                self.format_version
            )));
        }
        if !is_sha256_hex(&self.source_sha256) {
            return Err(PrecompressedError::InvalidMetadata(
                "source_sha256 must be exactly 64 hexadecimal characters".to_owned(),
            ));
        }
        if self.compressed_tokens > self.original_tokens {
            return Err(PrecompressedError::InvalidMetadata(
                "compressed_tokens must not exceed original_tokens".to_owned(),
            ));
        }
        Ok(())
    }

    /// Reports whether the recorded counts meet the 40% reduction target.
    pub fn meets_target_reduction(&self) -> bool {
        meets_target_reduction(self.original_tokens, self.compressed_tokens)
    }
}

/// Why an original context must be sent through runtime compression instead.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum PrecompressedFallbackReason {
    MissingArtifact,
    MissingMetadata,
    InvalidArtifact,
    InvalidMetadata,
    ArtifactTooLarge,
    MetadataTooLarge,
    InvalidConfiguredHash,
    ConfiguredHashMismatch,
    SourceModified,
}

/// Whether the loaded content is a direct pre-compressed hit or a fallback.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum PrecompressedLoadStatus {
    Hit,
    Stale(PrecompressedFallbackReason),
    RuntimeFallback(PrecompressedFallbackReason),
}

impl PrecompressedLoadStatus {
    pub fn is_hit(self) -> bool {
        matches!(self, Self::Hit)
    }

    pub fn is_stale(self) -> bool {
        matches!(self, Self::Stale(_))
    }

    pub fn uses_runtime_fallback(self) -> bool {
        !self.is_hit()
    }
}

/// Content selected for request construction.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct PrecompressedLoad {
    /// Pre-compressed text on a hit, otherwise the untouched original text.
    pub content: String,
    pub status: PrecompressedLoadStatus,
    /// Present only when a validated pre-compressed artifact is used.
    pub metadata: Option<PrecompressedMetadata>,
}

impl PrecompressedLoad {
    pub fn used_precompressed(&self) -> bool {
        self.status.is_hit()
    }
}

/// Errors that prevent safe pre-compressed context management.
#[derive(Debug, Error)]
pub enum PrecompressedError {
    #[error("pre-compressed context root is not a directory: {0}")]
    InvalidRoot(PathBuf),
    #[error("invalid pre-compressed context path `{path}`: {message}")]
    InvalidPath { path: PathBuf, message: String },
    #[error("path traversal is not allowed: {0}")]
    PathTraversal(PathBuf),
    #[error("path resolves outside the configured root: {0}")]
    OutsideRoot(PathBuf),
    #[error("path is not a regular file: {0}")]
    NotRegularFile(PathBuf),
    #[error("file `{path}` is too large ({actual} bytes; maximum {maximum})")]
    FileTooLarge {
        path: PathBuf,
        actual: u64,
        maximum: u64,
    },
    #[error("configured source is not registered: {0}")]
    EntryNotConfigured(PathBuf),
    #[error("duplicate configured source: {0}")]
    DuplicateSource(PathBuf),
    #[error("input and compressed output resolve to the same path: {0}")]
    SameInputOutput(PathBuf),
    #[error("pre-compressed context is not valid UTF-8: {0}")]
    InvalidUtf8(PathBuf),
    #[error("invalid pre-compressed metadata: {0}")]
    InvalidMetadata(String),
    #[error("failed to access `{path}`: {source}")]
    Io {
        path: PathBuf,
        #[source]
        source: io::Error,
    },
    #[error("failed to serialize pre-compressed metadata: {0}")]
    Serialize(#[from] serde_json::Error),
}

#[derive(Debug, Clone)]
struct ManagedEntry {
    config: PrecompressedEntry,
    source_path: PathBuf,
    compressed_path: PathBuf,
    metadata_path: PathBuf,
}

/// Root-constrained loader and writer for configured pre-compressed contexts.
#[derive(Debug, Clone)]
pub struct PrecompressedManager {
    canonical_root: PathBuf,
    entries: Vec<ManagedEntry>,
    max_file_bytes: u64,
}

impl PrecompressedManager {
    /// Creates a manager using [`DEFAULT_MAX_CONTEXT_FILE_BYTES`].
    pub fn new(
        root: impl AsRef<Path>,
        entries: impl IntoIterator<Item = PrecompressedEntry>,
    ) -> Result<Self, PrecompressedError> {
        Self::with_max_file_size(root, entries, DEFAULT_MAX_CONTEXT_FILE_BYTES)
    }

    /// Creates a manager with an explicit non-zero file-size cap.
    pub fn with_max_file_size(
        root: impl AsRef<Path>,
        entries: impl IntoIterator<Item = PrecompressedEntry>,
        max_file_bytes: u64,
    ) -> Result<Self, PrecompressedError> {
        let root = root.as_ref();
        let canonical_root = canonicalize(root)?;
        if !canonical_root.is_dir() {
            return Err(PrecompressedError::InvalidRoot(canonical_root));
        }
        if max_file_bytes == 0 {
            return Err(PrecompressedError::InvalidPath {
                path: canonical_root,
                message: "file-size cap must be greater than zero".to_owned(),
            });
        }

        let mut managed_entries = Vec::new();
        let mut sources = HashSet::new();
        for config in entries {
            let source_path =
                resolve_regular_file(&canonical_root, Path::new(&config.source_path))?;
            let compressed_path =
                constrain_path(&canonical_root, Path::new(&config.compressed_path))?;
            if paths_refer_to_same_file(&source_path, &compressed_path) {
                return Err(PrecompressedError::SameInputOutput(source_path));
            }
            if !sources.insert(source_path.clone()) {
                return Err(PrecompressedError::DuplicateSource(source_path));
            }
            let metadata_path = metadata_path_for(&compressed_path);
            ensure_path_under_root(&canonical_root, &metadata_path)?;
            managed_entries.push(ManagedEntry {
                config,
                source_path,
                compressed_path,
                metadata_path,
            });
        }

        Ok(Self {
            canonical_root,
            entries: managed_entries,
            max_file_bytes,
        })
    }

    pub fn root(&self) -> &Path {
        &self.canonical_root
    }

    pub fn max_file_bytes(&self) -> u64 {
        self.max_file_bytes
    }

    /// Loads a configured source reference.
    ///
    /// A hit returns the artifact directly. Missing, malformed, oversized, or
    /// stale artifacts return the untouched source with a status suitable for
    /// invoking the normal runtime compression pipeline.
    pub fn load(
        &self,
        source_reference: impl AsRef<Path>,
    ) -> Result<PrecompressedLoad, PrecompressedError> {
        let source_path = resolve_regular_file(&self.canonical_root, source_reference.as_ref())?;
        let entry = self
            .entries
            .iter()
            .find(|entry| entry.source_path == source_path)
            .ok_or_else(|| PrecompressedError::EntryNotConfigured(source_path.clone()))?;

        let source_bytes = read_regular_file(&source_path, self.max_file_bytes)
            .map_err(|error| error.into_public(source_path.clone()))?;
        let source_hash = sha256_hex(&source_bytes);
        let original = String::from_utf8(source_bytes)
            .map_err(|_| PrecompressedError::InvalidUtf8(source_path.clone()))?;

        let artifact = match read_regular_file(&entry.compressed_path, self.max_file_bytes) {
            Ok(bytes) => match String::from_utf8(bytes) {
                Ok(content) => content,
                Err(_) => {
                    return Ok(self.fallback(
                        entry,
                        original,
                        PrecompressedFallbackReason::InvalidArtifact,
                    ))
                }
            },
            Err(error) => {
                let reason = match error {
                    FileReadError::Missing => PrecompressedFallbackReason::MissingArtifact,
                    FileReadError::TooLarge { .. } => PrecompressedFallbackReason::ArtifactTooLarge,
                    FileReadError::NotRegular | FileReadError::Io(_) => {
                        PrecompressedFallbackReason::InvalidArtifact
                    }
                };
                return Ok(self.fallback(entry, original, reason));
            }
        };

        let metadata_bytes = match read_regular_file(&entry.metadata_path, MAX_METADATA_FILE_BYTES)
        {
            Ok(bytes) => bytes,
            Err(error) => {
                let reason = match error {
                    FileReadError::Missing => PrecompressedFallbackReason::MissingMetadata,
                    FileReadError::TooLarge { .. } => PrecompressedFallbackReason::MetadataTooLarge,
                    FileReadError::NotRegular | FileReadError::Io(_) => {
                        PrecompressedFallbackReason::InvalidMetadata
                    }
                };
                return Ok(self.fallback(entry, original, reason));
            }
        };
        let metadata: PrecompressedMetadata = match serde_json::from_slice(&metadata_bytes) {
            Ok(metadata) => metadata,
            Err(_) => {
                return Ok(self.fallback(
                    entry,
                    original,
                    PrecompressedFallbackReason::InvalidMetadata,
                ))
            }
        };
        if metadata.validate().is_err() {
            return Ok(self.fallback(
                entry,
                original,
                PrecompressedFallbackReason::InvalidMetadata,
            ));
        }

        if let Some(configured_hash) = entry.config.content_hash.as_deref() {
            if !is_sha256_hex(configured_hash) {
                return Ok(self.fallback(
                    entry,
                    original,
                    PrecompressedFallbackReason::InvalidConfiguredHash,
                ));
            }
            if !source_hash.eq_ignore_ascii_case(configured_hash) {
                return Ok(self.stale(
                    entry,
                    original,
                    PrecompressedFallbackReason::ConfiguredHashMismatch,
                ));
            }
        }
        if !source_hash.eq_ignore_ascii_case(&metadata.source_sha256) {
            return Ok(self.stale(entry, original, PrecompressedFallbackReason::SourceModified));
        }

        Ok(PrecompressedLoad {
            content: artifact,
            status: PrecompressedLoadStatus::Hit,
            metadata: Some(metadata),
        })
    }

    /// Atomically writes an artifact and its deterministic metadata sidecar.
    ///
    /// Both files are fully staged as temporary siblings before either final
    /// path is replaced. Existing files are backed up and restored if a commit
    /// step fails.
    pub fn write_atomic(
        &self,
        source_path: impl AsRef<Path>,
        compressed_path: impl AsRef<Path>,
        compressed_content: &[u8],
        metadata: &PrecompressedMetadata,
    ) -> Result<PathBuf, PrecompressedError> {
        metadata.validate()?;
        let source_path = resolve_regular_file(&self.canonical_root, source_path.as_ref())?;
        let compressed_path = constrain_path(&self.canonical_root, compressed_path.as_ref())?;
        let metadata_path = metadata_path_for(&compressed_path);
        ensure_path_under_root(&self.canonical_root, &metadata_path)?;
        ensure_writable_file_target(&compressed_path)?;
        ensure_writable_file_target(&metadata_path)?;

        if paths_refer_to_same_file(&source_path, &compressed_path)
            || paths_refer_to_same_file(&source_path, &metadata_path)
        {
            return Err(PrecompressedError::SameInputOutput(source_path));
        }
        if compressed_content.len() as u64 > self.max_file_bytes {
            return Err(PrecompressedError::FileTooLarge {
                path: compressed_path,
                actual: compressed_content.len() as u64,
                maximum: self.max_file_bytes,
            });
        }

        let source = read_regular_file(&source_path, self.max_file_bytes)
            .map_err(|error| error.into_public(source_path.clone()))?;
        if !sha256_hex(&source).eq_ignore_ascii_case(&metadata.source_sha256) {
            return Err(PrecompressedError::InvalidMetadata(
                "source_sha256 does not match the current input file".to_owned(),
            ));
        }

        let mut sidecar = serde_json::to_vec_pretty(metadata)?;
        sidecar.push(b'\n');
        if sidecar.len() as u64 > MAX_METADATA_FILE_BYTES {
            return Err(PrecompressedError::FileTooLarge {
                path: metadata_path,
                actual: sidecar.len() as u64,
                maximum: MAX_METADATA_FILE_BYTES,
            });
        }

        let mut artifact_temp = StagedFile::create(&compressed_path, compressed_content)?;
        let mut metadata_temp = StagedFile::create(&metadata_path, &sidecar)?;
        commit_staged_pair(
            &mut artifact_temp,
            &compressed_path,
            &mut metadata_temp,
            &metadata_path,
        )?;
        Ok(metadata_path)
    }

    fn fallback(
        &self,
        entry: &ManagedEntry,
        original: String,
        reason: PrecompressedFallbackReason,
    ) -> PrecompressedLoad {
        tracing::warn!(
            source_path = %entry.source_path.display(),
            compressed_path = %entry.compressed_path.display(),
            ?reason,
            "Pre-compressed context unavailable; using runtime compression fallback"
        );
        PrecompressedLoad {
            content: original,
            status: PrecompressedLoadStatus::RuntimeFallback(reason),
            metadata: None,
        }
    }

    fn stale(
        &self,
        entry: &ManagedEntry,
        original: String,
        reason: PrecompressedFallbackReason,
    ) -> PrecompressedLoad {
        tracing::warn!(
            source_path = %entry.source_path.display(),
            compressed_path = %entry.compressed_path.display(),
            ?reason,
            "Pre-compressed context is stale; using runtime compression fallback"
        );
        PrecompressedLoad {
            content: original,
            status: PrecompressedLoadStatus::Stale(reason),
            metadata: None,
        }
    }
}

/// Writes an artifact using the default cap without constructing a long-lived manager.
pub fn write_precompressed_atomic(
    root: impl AsRef<Path>,
    source_path: impl AsRef<Path>,
    compressed_path: impl AsRef<Path>,
    compressed_content: &[u8],
    metadata: &PrecompressedMetadata,
) -> Result<PathBuf, PrecompressedError> {
    PrecompressedManager::new(root, std::iter::empty())?.write_atomic(
        source_path,
        compressed_path,
        compressed_content,
        metadata,
    )
}

/// Returns the deterministic sidecar path for a compressed artifact.
pub fn metadata_path_for(compressed_path: impl AsRef<Path>) -> PathBuf {
    let compressed_path = compressed_path.as_ref();
    let mut sidecar_name: OsString = compressed_path
        .file_name()
        .unwrap_or_default()
        .to_os_string();
    sidecar_name.push(".metadata.json");
    compressed_path.with_file_name(sidecar_name)
}

/// Computes a lowercase SHA-256 digest using `ring`.
pub fn sha256_hex(content: &[u8]) -> String {
    let digest = digest(&SHA256, content);
    let mut encoded = String::with_capacity(64);
    for byte in digest.as_ref() {
        let _ = write!(encoded, "{byte:02x}");
    }
    encoded
}

/// Evaluates the 40% target using integer arithmetic without claiming that a
/// caller's compression operation achieved it.
pub fn meets_target_reduction(original_tokens: u64, compressed_tokens: u64) -> bool {
    original_tokens > 0
        && compressed_tokens <= original_tokens
        && u128::from(compressed_tokens) * 100
            <= u128::from(original_tokens) * (100 - TARGET_REDUCTION_PERCENT) as u128
}

fn is_sha256_hex(value: &str) -> bool {
    value.len() == 64 && value.bytes().all(|byte| byte.is_ascii_hexdigit())
}

fn canonicalize(path: &Path) -> Result<PathBuf, PrecompressedError> {
    path.canonicalize()
        .map_err(|source| PrecompressedError::Io {
            path: path.to_path_buf(),
            source,
        })
}

fn validate_path(path: &Path) -> Result<(), PrecompressedError> {
    if path.as_os_str().is_empty() {
        return Err(PrecompressedError::InvalidPath {
            path: path.to_path_buf(),
            message: "path must not be empty".to_owned(),
        });
    }
    if path.to_string_lossy().contains('\0') {
        return Err(PrecompressedError::InvalidPath {
            path: path.to_path_buf(),
            message: "path must not contain NUL characters".to_owned(),
        });
    }
    if path
        .components()
        .any(|component| matches!(component, Component::ParentDir))
    {
        return Err(PrecompressedError::PathTraversal(path.to_path_buf()));
    }
    if !path.is_absolute()
        && path
            .components()
            .any(|component| matches!(component, Component::Prefix(_) | Component::RootDir))
    {
        return Err(PrecompressedError::InvalidPath {
            path: path.to_path_buf(),
            message: "relative path contains an absolute component".to_owned(),
        });
    }
    Ok(())
}

fn candidate_path(root: &Path, path: &Path) -> Result<PathBuf, PrecompressedError> {
    validate_path(path)?;
    Ok(if path.is_absolute() {
        path.to_path_buf()
    } else {
        root.join(path)
    })
}

fn resolve_regular_file(root: &Path, path: &Path) -> Result<PathBuf, PrecompressedError> {
    let candidate = candidate_path(root, path)?;
    let link_metadata =
        fs::symlink_metadata(&candidate).map_err(|source| PrecompressedError::Io {
            path: candidate.clone(),
            source,
        })?;
    if !link_metadata.file_type().is_file() {
        return Err(PrecompressedError::NotRegularFile(candidate));
    }
    let canonical = canonicalize(&candidate)?;
    ensure_path_under_root(root, &canonical)?;
    Ok(canonical)
}

fn constrain_path(root: &Path, path: &Path) -> Result<PathBuf, PrecompressedError> {
    let candidate = candidate_path(root, path)?;
    let file_name = candidate
        .file_name()
        .ok_or_else(|| PrecompressedError::InvalidPath {
            path: candidate.clone(),
            message: "path must name a file".to_owned(),
        })?;
    let parent = candidate
        .parent()
        .ok_or_else(|| PrecompressedError::InvalidPath {
            path: candidate.clone(),
            message: "path must have a parent directory".to_owned(),
        })?;
    let canonical_parent = canonicalize(parent)?;
    ensure_path_under_root(root, &canonical_parent)?;
    let constrained = canonical_parent.join(file_name);

    if fs::symlink_metadata(&constrained).is_ok() {
        if let Ok(canonical_target) = constrained.canonicalize() {
            ensure_path_under_root(root, &canonical_target)?;
        }
    }
    Ok(constrained)
}

fn ensure_path_under_root(root: &Path, path: &Path) -> Result<(), PrecompressedError> {
    if path.starts_with(root) {
        Ok(())
    } else {
        Err(PrecompressedError::OutsideRoot(path.to_path_buf()))
    }
}

fn paths_refer_to_same_file(left: &Path, right: &Path) -> bool {
    if left == right {
        return true;
    }
    match (left.canonicalize(), right.canonicalize()) {
        (Ok(left), Ok(right)) => left == right,
        _ => false,
    }
}

fn ensure_writable_file_target(path: &Path) -> Result<(), PrecompressedError> {
    match fs::symlink_metadata(path) {
        Ok(metadata) if metadata.file_type().is_file() => Ok(()),
        Ok(_) => Err(PrecompressedError::NotRegularFile(path.to_path_buf())),
        Err(error) if error.kind() == io::ErrorKind::NotFound => Ok(()),
        Err(source) => Err(PrecompressedError::Io {
            path: path.to_path_buf(),
            source,
        }),
    }
}

#[derive(Debug)]
enum FileReadError {
    Missing,
    NotRegular,
    TooLarge { actual: u64, maximum: u64 },
    Io(io::Error),
}

impl FileReadError {
    fn into_public(self, path: PathBuf) -> PrecompressedError {
        match self {
            Self::Missing => PrecompressedError::Io {
                path,
                source: io::Error::new(io::ErrorKind::NotFound, "file not found"),
            },
            Self::NotRegular => PrecompressedError::NotRegularFile(path),
            Self::TooLarge { actual, maximum } => PrecompressedError::FileTooLarge {
                path,
                actual,
                maximum,
            },
            Self::Io(source) => PrecompressedError::Io { path, source },
        }
    }
}

fn read_regular_file(path: &Path, maximum: u64) -> Result<Vec<u8>, FileReadError> {
    let link_metadata = fs::symlink_metadata(path).map_err(|error| {
        if error.kind() == io::ErrorKind::NotFound {
            FileReadError::Missing
        } else {
            FileReadError::Io(error)
        }
    })?;
    if !link_metadata.file_type().is_file() {
        return Err(FileReadError::NotRegular);
    }
    if link_metadata.len() > maximum {
        return Err(FileReadError::TooLarge {
            actual: link_metadata.len(),
            maximum,
        });
    }

    let file = File::open(path).map_err(FileReadError::Io)?;
    let mut bytes = Vec::with_capacity(link_metadata.len() as usize);
    file.take(maximum.saturating_add(1))
        .read_to_end(&mut bytes)
        .map_err(FileReadError::Io)?;
    if bytes.len() as u64 > maximum {
        return Err(FileReadError::TooLarge {
            actual: bytes.len() as u64,
            maximum,
        });
    }
    Ok(bytes)
}

#[derive(Debug)]
struct StagedFile {
    path: PathBuf,
    active: bool,
}

impl StagedFile {
    fn create(target: &Path, content: &[u8]) -> Result<Self, PrecompressedError> {
        for _ in 0..32 {
            let path = unique_sibling_path(target, "tmp")?;
            match OpenOptions::new().write(true).create_new(true).open(&path) {
                Ok(mut file) => {
                    if let Err(source) = file.write_all(content).and_then(|_| file.sync_all()) {
                        let _ = fs::remove_file(&path);
                        return Err(PrecompressedError::Io { path, source });
                    }
                    return Ok(Self { path, active: true });
                }
                Err(error) if error.kind() == io::ErrorKind::AlreadyExists => continue,
                Err(source) => return Err(PrecompressedError::Io { path, source }),
            }
        }
        Err(PrecompressedError::Io {
            path: target.to_path_buf(),
            source: io::Error::new(
                io::ErrorKind::AlreadyExists,
                "could not allocate a unique temporary sibling",
            ),
        })
    }

    fn disarm(&mut self) {
        self.active = false;
    }
}

impl Drop for StagedFile {
    fn drop(&mut self) {
        if self.active {
            let _ = fs::remove_file(&self.path);
        }
    }
}

fn unique_sibling_path(target: &Path, suffix: &str) -> Result<PathBuf, PrecompressedError> {
    let parent = target
        .parent()
        .ok_or_else(|| PrecompressedError::InvalidPath {
            path: target.to_path_buf(),
            message: "target must have a parent directory".to_owned(),
        })?;
    let file_name = target
        .file_name()
        .ok_or_else(|| PrecompressedError::InvalidPath {
            path: target.to_path_buf(),
            message: "target must name a file".to_owned(),
        })?
        .to_string_lossy();
    let sequence = TEMP_FILE_SEQUENCE.fetch_add(1, Ordering::Relaxed);
    Ok(parent.join(format!(
        ".{file_name}.{}.{}.{}",
        std::process::id(),
        sequence,
        suffix
    )))
}

fn move_existing_to_backup(path: &Path) -> Result<Option<PathBuf>, PrecompressedError> {
    match fs::symlink_metadata(path) {
        Ok(metadata) if !metadata.file_type().is_file() => {
            return Err(PrecompressedError::NotRegularFile(path.to_path_buf()))
        }
        Ok(_) => {}
        Err(error) if error.kind() == io::ErrorKind::NotFound => return Ok(None),
        Err(source) => {
            return Err(PrecompressedError::Io {
                path: path.to_path_buf(),
                source,
            })
        }
    }
    for _ in 0..32 {
        let backup = unique_sibling_path(path, "backup")?;
        if backup.exists() {
            continue;
        }
        match fs::rename(path, &backup) {
            Ok(()) => return Ok(Some(backup)),
            Err(error) if error.kind() == io::ErrorKind::AlreadyExists => continue,
            Err(source) => {
                return Err(PrecompressedError::Io {
                    path: path.to_path_buf(),
                    source,
                })
            }
        }
    }
    Err(PrecompressedError::Io {
        path: path.to_path_buf(),
        source: io::Error::new(
            io::ErrorKind::AlreadyExists,
            "could not allocate a unique backup sibling",
        ),
    })
}

fn restore_backup(backup: Option<&PathBuf>, target: &Path) {
    if let Some(backup) = backup {
        let _ = fs::rename(backup, target);
    }
}

fn commit_staged_pair(
    artifact_temp: &mut StagedFile,
    artifact_path: &Path,
    metadata_temp: &mut StagedFile,
    metadata_path: &Path,
) -> Result<(), PrecompressedError> {
    if artifact_path == metadata_path {
        return Err(PrecompressedError::InvalidPath {
            path: artifact_path.to_path_buf(),
            message: "artifact and metadata paths must be distinct".to_owned(),
        });
    }
    let artifact_backup = move_existing_to_backup(artifact_path)?;
    let metadata_backup = match move_existing_to_backup(metadata_path) {
        Ok(backup) => backup,
        Err(error) => {
            restore_backup(artifact_backup.as_ref(), artifact_path);
            return Err(error);
        }
    };

    if let Err(source) = fs::rename(&artifact_temp.path, artifact_path) {
        restore_backup(artifact_backup.as_ref(), artifact_path);
        restore_backup(metadata_backup.as_ref(), metadata_path);
        return Err(PrecompressedError::Io {
            path: artifact_path.to_path_buf(),
            source,
        });
    }
    artifact_temp.disarm();

    if let Err(source) = fs::rename(&metadata_temp.path, metadata_path) {
        let _ = fs::remove_file(artifact_path);
        restore_backup(artifact_backup.as_ref(), artifact_path);
        restore_backup(metadata_backup.as_ref(), metadata_path);
        return Err(PrecompressedError::Io {
            path: metadata_path.to_path_buf(),
            source,
        });
    }
    metadata_temp.disarm();

    if let Some(backup) = artifact_backup {
        let _ = fs::remove_file(backup);
    }
    if let Some(backup) = metadata_backup {
        let _ = fs::remove_file(backup);
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use tempfile::TempDir;

    const ORIGINAL: &str = "This is the original natural language context with technical details.";
    const COMPRESSED: &str = "Original context; technical details.";

    fn write(path: impl AsRef<Path>, content: impl AsRef<[u8]>) {
        fs::write(path, content).unwrap();
    }

    fn metadata(source: &[u8]) -> PrecompressedMetadata {
        PrecompressedMetadata::for_source(100, 55, CompressionLevel::Standard, Utc::now(), source)
            .unwrap()
    }

    fn entry(content_hash: Option<String>) -> PrecompressedEntry {
        PrecompressedEntry {
            source_path: "source.txt".to_owned(),
            compressed_path: "source.txt.compressed".to_owned(),
            content_hash,
        }
    }

    fn fixture(write_artifact: bool, write_metadata: bool) -> (TempDir, PrecompressedMetadata) {
        let directory = TempDir::new().unwrap();
        write(directory.path().join("source.txt"), ORIGINAL);
        let metadata = metadata(ORIGINAL.as_bytes());
        if write_artifact {
            write(directory.path().join("source.txt.compressed"), COMPRESSED);
        }
        if write_metadata {
            write(
                directory.path().join("source.txt.compressed.metadata.json"),
                serde_json::to_vec(&metadata).unwrap(),
            );
        }
        (directory, metadata)
    }

    #[test]
    fn valid_artifact_is_returned_directly() {
        let (directory, metadata) = fixture(true, true);
        let manager = PrecompressedManager::new(
            directory.path(),
            [entry(Some(metadata.source_sha256.clone()))],
        )
        .unwrap();

        let loaded = manager.load("source.txt").unwrap();

        assert_eq!(loaded.content, COMPRESSED);
        assert_eq!(loaded.status, PrecompressedLoadStatus::Hit);
        assert_eq!(loaded.metadata, Some(metadata));
        assert!(loaded.used_precompressed());
    }

    #[test]
    fn modified_source_is_stale_and_returns_original() {
        let (directory, _) = fixture(true, true);
        let manager = PrecompressedManager::new(directory.path(), [entry(None)]).unwrap();
        let modified = "The source changed after pre-compression.";
        write(directory.path().join("source.txt"), modified);

        let loaded = manager.load("source.txt").unwrap();

        assert_eq!(loaded.content, modified);
        assert_eq!(
            loaded.status,
            PrecompressedLoadStatus::Stale(PrecompressedFallbackReason::SourceModified)
        );
        assert!(loaded.metadata.is_none());
    }

    #[test]
    fn missing_artifact_or_metadata_returns_runtime_fallback() {
        let (missing_artifact, _) = fixture(false, false);
        let manager = PrecompressedManager::new(missing_artifact.path(), [entry(None)]).unwrap();
        let loaded = manager.load("source.txt").unwrap();
        assert_eq!(loaded.content, ORIGINAL);
        assert_eq!(
            loaded.status,
            PrecompressedLoadStatus::RuntimeFallback(PrecompressedFallbackReason::MissingArtifact)
        );

        let (missing_metadata, _) = fixture(true, false);
        let manager = PrecompressedManager::new(missing_metadata.path(), [entry(None)]).unwrap();
        let loaded = manager.load("source.txt").unwrap();
        assert_eq!(loaded.content, ORIGINAL);
        assert_eq!(
            loaded.status,
            PrecompressedLoadStatus::RuntimeFallback(PrecompressedFallbackReason::MissingMetadata)
        );
    }

    #[test]
    fn configured_hash_mismatch_is_stale() {
        let (directory, _) = fixture(true, true);
        let manager = PrecompressedManager::new(
            directory.path(),
            [entry(Some(sha256_hex(b"different source")))],
        )
        .unwrap();

        let loaded = manager.load("source.txt").unwrap();

        assert_eq!(loaded.content, ORIGINAL);
        assert_eq!(
            loaded.status,
            PrecompressedLoadStatus::Stale(PrecompressedFallbackReason::ConfiguredHashMismatch)
        );
    }

    #[test]
    fn malformed_metadata_returns_runtime_fallback() {
        let (directory, _) = fixture(true, false);
        write(
            directory.path().join("source.txt.compressed.metadata.json"),
            br#"{"format_version":1,"original_tokens":"secret content"}"#,
        );
        let manager = PrecompressedManager::new(directory.path(), [entry(None)]).unwrap();

        let loaded = manager.load("source.txt").unwrap();

        assert_eq!(loaded.content, ORIGINAL);
        assert_eq!(
            loaded.status,
            PrecompressedLoadStatus::RuntimeFallback(PrecompressedFallbackReason::InvalidMetadata)
        );
    }

    #[test]
    fn traversal_and_outside_root_are_rejected() {
        let root = TempDir::new().unwrap();
        let outside = TempDir::new().unwrap();
        write(root.path().join("source.txt"), ORIGINAL);
        write(outside.path().join("outside.txt"), ORIGINAL);

        let traversal = PrecompressedManager::new(
            root.path(),
            [PrecompressedEntry {
                source_path: "source.txt".to_owned(),
                compressed_path: "../escape.txt".to_owned(),
                content_hash: None,
            }],
        );
        assert!(matches!(
            traversal,
            Err(PrecompressedError::PathTraversal(_))
        ));

        let outside_root = PrecompressedManager::new(
            root.path(),
            [PrecompressedEntry {
                source_path: outside.path().join("outside.txt").display().to_string(),
                compressed_path: "compressed.txt".to_owned(),
                content_hash: None,
            }],
        );
        assert!(matches!(
            outside_root,
            Err(PrecompressedError::OutsideRoot(_))
        ));
    }

    #[test]
    fn same_input_and_output_are_refused() {
        let directory = TempDir::new().unwrap();
        write(directory.path().join("source.txt"), ORIGINAL);
        let same_entry = PrecompressedEntry {
            source_path: "source.txt".to_owned(),
            compressed_path: "source.txt".to_owned(),
            content_hash: None,
        };
        assert!(matches!(
            PrecompressedManager::new(directory.path(), [same_entry]),
            Err(PrecompressedError::SameInputOutput(_))
        ));

        let metadata = metadata(ORIGINAL.as_bytes());
        assert!(matches!(
            write_precompressed_atomic(
                directory.path(),
                "source.txt",
                "source.txt",
                COMPRESSED.as_bytes(),
                &metadata
            ),
            Err(PrecompressedError::SameInputOutput(_))
        ));
        assert_eq!(
            fs::read_to_string(directory.path().join("source.txt")).unwrap(),
            ORIGINAL
        );
    }

    #[test]
    fn atomic_write_creates_complete_pair_and_leaves_no_partial_files() {
        let directory = TempDir::new().unwrap();
        write(directory.path().join("source.txt"), ORIGINAL);
        let metadata = metadata(ORIGINAL.as_bytes());

        let sidecar = write_precompressed_atomic(
            directory.path(),
            "source.txt",
            "compressed.txt",
            COMPRESSED.as_bytes(),
            &metadata,
        )
        .unwrap();
        assert_eq!(
            fs::read_to_string(directory.path().join("compressed.txt")).unwrap(),
            COMPRESSED
        );
        assert_eq!(
            serde_json::from_slice::<PrecompressedMetadata>(&fs::read(&sidecar).unwrap()).unwrap(),
            metadata
        );
        assert!(fs::read_dir(directory.path()).unwrap().all(|entry| {
            let name = entry.unwrap().file_name().to_string_lossy().into_owned();
            !name.contains(".tmp") && !name.contains(".backup")
        }));

        let invalid = PrecompressedMetadata {
            compressed_tokens: 101,
            ..metadata
        };
        let failed = write_precompressed_atomic(
            directory.path(),
            "source.txt",
            "failed.txt",
            COMPRESSED.as_bytes(),
            &invalid,
        );
        assert!(matches!(
            failed,
            Err(PrecompressedError::InvalidMetadata(_))
        ));
        assert!(!directory.path().join("failed.txt").exists());
        assert!(!directory.path().join("failed.txt.metadata.json").exists());
    }

    #[test]
    fn metadata_round_trips_without_content_or_paths() {
        let metadata = metadata(ORIGINAL.as_bytes());
        let serialized = serde_json::to_string(&metadata).unwrap();
        let round_trip: PrecompressedMetadata = serde_json::from_str(&serialized).unwrap();

        assert_eq!(round_trip, metadata);
        assert!(!serialized.contains(ORIGINAL));
        assert!(!serialized.contains(COMPRESSED));
        assert!(!serialized.contains("source.txt"));
        assert_eq!(
            metadata_path_for("context.md.compressed"),
            PathBuf::from("context.md.compressed.metadata.json")
        );
    }

    #[test]
    fn target_evaluator_reports_actual_math_without_faking_reduction() {
        assert!(meets_target_reduction(100, 60));
        assert!(meets_target_reduction(5, 3));
        assert!(!meets_target_reduction(100, 61));
        assert!(!meets_target_reduction(0, 0));
        assert!(!meets_target_reduction(100, 101));

        let achieved = PrecompressedMetadata::for_source(
            100,
            60,
            CompressionLevel::Standard,
            Utc::now(),
            ORIGINAL.as_bytes(),
        )
        .unwrap();
        let missed = PrecompressedMetadata::for_source(
            100,
            61,
            CompressionLevel::Standard,
            Utc::now(),
            ORIGINAL.as_bytes(),
        )
        .unwrap();
        assert!(achieved.meets_target_reduction());
        assert!(!missed.meets_target_reduction());
    }
}
