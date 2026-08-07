#![cfg(feature = "ml-router")]

//! Secure preference-data training orchestration.
//!
//! This module validates datasets and model artifacts but deliberately does not
//! implement training or augmentation. A real backend must be injected through
//! [`TrainingBackend`].

use std::fs::{self, File};
use std::io::{self, Read};
use std::path::{Component, Path, PathBuf};
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::{Arc, Mutex as StdMutex};
use std::time::Duration;

use async_trait::async_trait;
use serde::{Deserialize, Serialize};
use thiserror::Error;
use tokio::sync::{Mutex, Notify, OwnedSemaphorePermit, RwLock, Semaphore};
use tokio::task::JoinHandle;
use uuid::Uuid;

use super::ml_classifier::{ArtifactLoader, ValidatedArtifact};

pub const MIN_PREFERENCE_PAIRS: usize = 100;
pub const MAX_DATASET_BYTES: u64 = 64 * 1024 * 1024;
pub const MAX_TRAINING_BATCH_SIZE: usize = 4096;
pub const MAX_TRAINING_EPOCHS: usize = 1000;
pub const MAX_AUGMENTATIONS_PER_PAIR: usize = 4;
pub const MAX_AUGMENTATION_TEMPERATURE: f64 = 2.0;

#[derive(Debug, Clone, Copy, PartialEq, Eq, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum PreferenceLabel {
    Strong,
    Weak,
}

#[derive(Debug, Deserialize)]
#[serde(deny_unknown_fields)]
struct PreferencePair {
    query: String,
    strong_response: String,
    weak_response: String,
    label: PreferenceLabel,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ValidatedDataset {
    path: PathBuf,
    pair_count: usize,
    size_bytes: u64,
}

impl ValidatedDataset {
    pub fn path(&self) -> &Path {
        &self.path
    }

    pub fn pair_count(&self) -> usize {
        self.pair_count
    }

    pub fn size_bytes(&self) -> u64 {
        self.size_bytes
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
pub struct TrainingHyperparameters {
    pub learning_rate: f64,
    pub batch_size: usize,
    pub epochs: usize,
}

impl TrainingHyperparameters {
    fn validate(self) -> Result<(), TrainingError> {
        if !self.learning_rate.is_finite() || !(0.0..=1.0).contains(&self.learning_rate) {
            return Err(TrainingError::InvalidRequest(
                "learning_rate must be finite, greater than 0, and at most 1".to_owned(),
            ));
        }
        if self.learning_rate == 0.0 {
            return Err(TrainingError::InvalidRequest(
                "learning_rate must be finite, greater than 0, and at most 1".to_owned(),
            ));
        }
        if !(1..=MAX_TRAINING_BATCH_SIZE).contains(&self.batch_size) {
            return Err(TrainingError::InvalidRequest(format!(
                "batch_size must be in 1..={MAX_TRAINING_BATCH_SIZE}"
            )));
        }
        if !(1..=MAX_TRAINING_EPOCHS).contains(&self.epochs) {
            return Err(TrainingError::InvalidRequest(format!(
                "epochs must be in 1..={MAX_TRAINING_EPOCHS}"
            )));
        }
        Ok(())
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
pub struct AugmentationPolicy {
    pub enabled: bool,
    pub paraphrases_per_pair: usize,
    pub paraphrase_temperature: f64,
    pub back_translation: bool,
}

impl Default for AugmentationPolicy {
    fn default() -> Self {
        Self {
            enabled: false,
            paraphrases_per_pair: 0,
            paraphrase_temperature: 1.0,
            back_translation: false,
        }
    }
}

impl AugmentationPolicy {
    fn validate(self) -> Result<(), TrainingError> {
        if self.paraphrases_per_pair > MAX_AUGMENTATIONS_PER_PAIR {
            return Err(TrainingError::InvalidRequest(format!(
                "paraphrases_per_pair must be at most {MAX_AUGMENTATIONS_PER_PAIR}"
            )));
        }
        if !self.paraphrase_temperature.is_finite()
            || !(0.0..=MAX_AUGMENTATION_TEMPERATURE).contains(&self.paraphrase_temperature)
        {
            return Err(TrainingError::InvalidRequest(format!(
                "paraphrase_temperature must be finite and in 0..={MAX_AUGMENTATION_TEMPERATURE}"
            )));
        }
        if !self.enabled && (self.paraphrases_per_pair != 0 || self.back_translation) {
            return Err(TrainingError::InvalidRequest(
                "augmentation settings require augmentation to be enabled".to_owned(),
            ));
        }
        Ok(())
    }
}

#[derive(Debug, Clone)]
pub struct TrainingCancellationToken {
    cancelled: Arc<AtomicBool>,
    notification: Arc<Notify>,
}

impl TrainingCancellationToken {
    fn new() -> Self {
        Self {
            cancelled: Arc::new(AtomicBool::new(false)),
            notification: Arc::new(Notify::new()),
        }
    }

    pub fn cancel(&self) {
        self.cancelled.store(true, Ordering::Release);
        self.notification.notify_waiters();
    }

    pub fn is_cancelled(&self) -> bool {
        self.cancelled.load(Ordering::Acquire)
    }

    pub async fn cancelled(&self) {
        if self.is_cancelled() {
            return;
        }
        let notified = self.notification.notified();
        if self.is_cancelled() {
            return;
        }
        notified.await;
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
pub struct TrainingProgress {
    pub percent: f64,
    pub current_epoch: usize,
    pub loss: Option<f64>,
    pub estimated_seconds_remaining: Option<u64>,
}

impl Default for TrainingProgress {
    fn default() -> Self {
        Self {
            percent: 0.0,
            current_epoch: 0,
            loss: None,
            estimated_seconds_remaining: None,
        }
    }
}

#[derive(Debug, Clone)]
pub struct TrainingProgressReporter {
    progress: Arc<StdMutex<TrainingProgress>>,
}

impl TrainingProgressReporter {
    fn new() -> Self {
        Self {
            progress: Arc::new(StdMutex::new(TrainingProgress::default())),
        }
    }

    pub fn update(&self, progress: TrainingProgress) -> Result<(), TrainingBackendError> {
        if !progress.percent.is_finite() || !(0.0..=100.0).contains(&progress.percent) {
            return Err(TrainingBackendError::InvalidProgress);
        }
        if progress.loss.is_some_and(|loss| !loss.is_finite()) {
            return Err(TrainingBackendError::InvalidProgress);
        }
        let mut current = self
            .progress
            .lock()
            .map_err(|_| TrainingBackendError::Internal)?;
        if progress.percent < current.percent || progress.current_epoch < current.current_epoch {
            return Err(TrainingBackendError::InvalidProgress);
        }
        *current = progress;
        Ok(())
    }

    fn snapshot(&self) -> TrainingProgress {
        self.progress
            .lock()
            .map(|progress| *progress)
            .unwrap_or_default()
    }
}

#[derive(Debug, Clone)]
pub struct TrainingBackendRequest {
    pub dataset: ValidatedDataset,
    pub output_artifact_path: PathBuf,
    pub hyperparameters: TrainingHyperparameters,
    pub augmentation: AugmentationPolicy,
    pub cancellation: TrainingCancellationToken,
    pub progress: TrainingProgressReporter,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Error)]
pub enum TrainingBackendError {
    #[error("training backend failed")]
    Failed,
    #[error("training backend does not support the requested operation")]
    Unsupported,
    #[error("training was cancelled")]
    Cancelled,
    #[error("training backend reported invalid progress")]
    InvalidProgress,
    #[error("training backend internal state is unavailable")]
    Internal,
}

#[async_trait]
pub trait TrainingBackend: Send + Sync {
    async fn train(&self, request: TrainingBackendRequest) -> Result<(), TrainingBackendError>;
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct TrainingJobId(String);

impl TrainingJobId {
    pub fn as_str(&self) -> &str {
        &self.0
    }
}

#[derive(Debug, Clone)]
pub struct TrainingRequest {
    pub dataset_path: PathBuf,
    pub hyperparameters: TrainingHyperparameters,
    pub augmentation: AugmentationPolicy,
}

impl TrainingRequest {
    fn validate(&self) -> Result<(), TrainingError> {
        self.hyperparameters.validate()?;
        self.augmentation.validate()?;
        if self.augmentation.enabled {
            return Err(TrainingError::Unsupported);
        }
        Ok(())
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize)]
#[serde(rename_all = "snake_case")]
pub enum TrainingJobState {
    Running,
    Cancelling,
    Completed,
    Failed,
    Cancelled,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize)]
#[serde(rename_all = "snake_case")]
pub enum TrainingFailureCategory {
    Dataset,
    Unsupported,
    Backend,
    Artifact,
    Filesystem,
    Cancelled,
    Internal,
}

#[derive(Debug, Clone, Serialize)]
pub struct TrainingJobStatus {
    pub job_id: String,
    pub state: TrainingJobState,
    pub progress: TrainingProgress,
    pub failure: Option<TrainingFailureCategory>,
}

#[derive(Debug, Error)]
pub enum TrainingError {
    #[error("unsafe training path: {0}")]
    UnsafePath(String),
    #[error("training dataset does not exist")]
    DatasetMissing,
    #[error("training dataset is not a regular file")]
    DatasetNotFile,
    #[error("training dataset exceeds the {MAX_DATASET_BYTES} byte limit")]
    DatasetTooLarge,
    #[error("training dataset line {line} is not a valid preference record")]
    InvalidDatasetRecord { line: usize },
    #[error("training dataset line {line} contains an empty required field")]
    EmptyDatasetField { line: usize },
    #[error("training dataset has {actual} records; at least {MIN_PREFERENCE_PAIRS} are required")]
    InsufficientData { actual: usize },
    #[error("invalid training request: {0}")]
    InvalidRequest(String),
    #[error("a training job is already running")]
    Busy,
    #[error("training is unsupported because no real backend is configured")]
    Unsupported,
    #[error("training job was not found")]
    JobNotFound,
    #[error("training filesystem operation failed")]
    Filesystem,
    #[error("training task could not be scheduled")]
    Runtime,
}

struct JobRecord {
    id: TrainingJobId,
    state: TrainingJobState,
    failure: Option<TrainingFailureCategory>,
    cancellation: TrainingCancellationToken,
    progress: TrainingProgressReporter,
}

struct TrainingManagerInner {
    dataset_root: PathBuf,
    artifact_root: PathBuf,
    backend: Option<Arc<dyn TrainingBackend>>,
    timeout: Duration,
    semaphore: Arc<Semaphore>,
    job: Mutex<Option<JobRecord>>,
    active_artifact: RwLock<Option<Arc<ValidatedArtifact>>>,
}

#[derive(Clone)]
pub struct TrainingManager {
    inner: Arc<TrainingManagerInner>,
    task: Arc<StdMutex<Option<JoinHandle<()>>>>,
}

impl TrainingManager {
    pub fn new(
        dataset_root: impl AsRef<Path>,
        artifact_root: impl AsRef<Path>,
        backend: Option<Arc<dyn TrainingBackend>>,
    ) -> Result<Self, TrainingError> {
        Self::with_timeout(
            dataset_root,
            artifact_root,
            backend,
            default_training_timeout(),
        )
    }

    pub fn with_timeout(
        dataset_root: impl AsRef<Path>,
        artifact_root: impl AsRef<Path>,
        backend: Option<Arc<dyn TrainingBackend>>,
        timeout: Duration,
    ) -> Result<Self, TrainingError> {
        if timeout.is_zero() {
            return Err(TrainingError::InvalidRequest(
                "training timeout must be greater than zero".to_owned(),
            ));
        }
        let dataset_root = validate_existing_directory(dataset_root.as_ref())?;
        let artifact_root = prepare_artifact_root(artifact_root.as_ref())?;
        cleanup_temporary_artifacts(&artifact_root);
        Ok(Self {
            inner: Arc::new(TrainingManagerInner {
                dataset_root,
                artifact_root,
                backend,
                timeout,
                semaphore: Arc::new(Semaphore::new(1)),
                job: Mutex::new(None),
                active_artifact: RwLock::new(None),
            }),
            task: Arc::new(StdMutex::new(None)),
        })
    }

    pub async fn start(&self, request: TrainingRequest) -> Result<TrainingJobId, TrainingError> {
        request.validate()?;
        let backend = self
            .inner
            .backend
            .clone()
            .ok_or(TrainingError::Unsupported)?;
        let permit = self
            .inner
            .semaphore
            .clone()
            .try_acquire_owned()
            .map_err(|_| TrainingError::Busy)?;
        {
            let job = self.inner.job.lock().await;
            if job.as_ref().is_some_and(|job| {
                matches!(
                    job.state,
                    TrainingJobState::Running | TrainingJobState::Cancelling
                )
            }) {
                return Err(TrainingError::Busy);
            }
        }

        let dataset_root = self.inner.dataset_root.clone();
        let dataset_path = request.dataset_path.clone();
        let dataset =
            tokio::task::spawn_blocking(move || validate_dataset(&dataset_root, &dataset_path))
                .await
                .map_err(|_| TrainingError::Runtime)??;

        let id = TrainingJobId(Uuid::new_v4().to_string());
        let cancellation = TrainingCancellationToken::new();
        let progress = TrainingProgressReporter::new();
        let temporary_artifact = self
            .inner
            .artifact_root
            .join(format!(".training-{}.tmp", id.as_str()));
        fs::create_dir(&temporary_artifact).map_err(|_| TrainingError::Filesystem)?;

        {
            let mut job = self.inner.job.lock().await;
            *job = Some(JobRecord {
                id: id.clone(),
                state: TrainingJobState::Running,
                failure: None,
                cancellation: cancellation.clone(),
                progress: progress.clone(),
            });
        }

        let inner = Arc::clone(&self.inner);
        let job_id = id.clone();
        let task = tokio::spawn(async move {
            run_training_job(
                inner,
                backend,
                permit,
                job_id,
                dataset,
                temporary_artifact,
                request.hyperparameters,
                request.augmentation,
                cancellation,
                progress,
            )
            .await;
        });
        let mut task_slot = match self.task.lock() {
            Ok(task_slot) => task_slot,
            Err(_) => {
                task.abort();
                return Err(TrainingError::Runtime);
            }
        };
        if let Some(previous) = task_slot.take() {
            if !previous.is_finished() {
                task.abort();
                return Err(TrainingError::Busy);
            }
        }
        *task_slot = Some(task);
        Ok(id)
    }

    pub async fn status(&self, id: &TrainingJobId) -> Result<TrainingJobStatus, TrainingError> {
        let job = self.inner.job.lock().await;
        let job = job
            .as_ref()
            .filter(|job| job.id == *id)
            .ok_or(TrainingError::JobNotFound)?;
        Ok(TrainingJobStatus {
            job_id: job.id.0.clone(),
            state: job.state,
            progress: job.progress.snapshot(),
            failure: job.failure,
        })
    }

    pub async fn cancel(&self, id: &TrainingJobId) -> Result<(), TrainingError> {
        let mut job = self.inner.job.lock().await;
        let job = job
            .as_mut()
            .filter(|job| job.id == *id)
            .ok_or(TrainingError::JobNotFound)?;
        if job.state == TrainingJobState::Running {
            job.state = TrainingJobState::Cancelling;
            job.cancellation.cancel();
        }
        Ok(())
    }

    pub async fn active_artifact(&self) -> Option<Arc<ValidatedArtifact>> {
        self.inner.active_artifact.read().await.clone()
    }
}

impl Drop for TrainingManager {
    fn drop(&mut self) {
        if Arc::strong_count(&self.task) != 1 {
            return;
        }
        if let Ok(job) = self.inner.job.try_lock() {
            if let Some(job) = job.as_ref() {
                job.cancellation.cancel();
            }
        }
        if let Ok(mut task) = self.task.lock() {
            if let Some(task) = task.take() {
                task.abort();
            }
        }
        cleanup_temporary_artifacts(&self.inner.artifact_root);
    }
}

#[allow(clippy::too_many_arguments)]
async fn run_training_job(
    inner: Arc<TrainingManagerInner>,
    backend: Arc<dyn TrainingBackend>,
    _permit: OwnedSemaphorePermit,
    job_id: TrainingJobId,
    dataset: ValidatedDataset,
    temporary_artifact: PathBuf,
    hyperparameters: TrainingHyperparameters,
    augmentation: AugmentationPolicy,
    cancellation: TrainingCancellationToken,
    progress: TrainingProgressReporter,
) {
    let request = TrainingBackendRequest {
        dataset,
        output_artifact_path: temporary_artifact.clone(),
        hyperparameters,
        augmentation,
        cancellation: cancellation.clone(),
        progress,
    };

    let backend_future = backend.train(request);
    tokio::pin!(backend_future);
    let timeout = tokio::time::sleep(inner.timeout);
    tokio::pin!(timeout);
    let backend_result = tokio::select! {
        biased;
        _ = cancellation.cancelled() => Err(TrainingFailureCategory::Cancelled),
        _ = &mut timeout => {
            cancellation.cancel();
            Err(TrainingFailureCategory::Backend)
        }
        result = &mut backend_future => result.map_err(|error| match error {
            TrainingBackendError::Unsupported => TrainingFailureCategory::Unsupported,
            TrainingBackendError::Cancelled => TrainingFailureCategory::Cancelled,
            TrainingBackendError::Failed | TrainingBackendError::InvalidProgress => {
                TrainingFailureCategory::Backend
            }
            TrainingBackendError::Internal => TrainingFailureCategory::Internal,
        }),
    };
    let result = if cancellation.is_cancelled()
        && !matches!(backend_result, Err(TrainingFailureCategory::Backend))
    {
        Err(TrainingFailureCategory::Cancelled)
    } else if let Err(category) = backend_result {
        Err(category)
    } else {
        activate_artifact(&inner, &job_id, &temporary_artifact, &cancellation).await
    };

    let _ = fs::remove_dir_all(&temporary_artifact);
    let mut job = inner.job.lock().await;
    if let Some(job) = job.as_mut().filter(|job| job.id == job_id) {
        match result {
            Ok(()) => {
                job.state = TrainingJobState::Completed;
                job.failure = None;
            }
            Err(TrainingFailureCategory::Cancelled) => {
                job.state = TrainingJobState::Cancelled;
                job.failure = Some(TrainingFailureCategory::Cancelled);
            }
            Err(category) => {
                job.state = TrainingJobState::Failed;
                job.failure = Some(category);
            }
        }
    }
}

async fn activate_artifact(
    inner: &TrainingManagerInner,
    job_id: &TrainingJobId,
    temporary_artifact: &Path,
    cancellation: &TrainingCancellationToken,
) -> Result<(), TrainingFailureCategory> {
    reject_symlinks_recursively(temporary_artifact)
        .map_err(|_| TrainingFailureCategory::Artifact)?;
    ArtifactLoader::load(temporary_artifact).map_err(|_| TrainingFailureCategory::Artifact)?;
    if cancellation.is_cancelled() {
        return Err(TrainingFailureCategory::Cancelled);
    }

    let final_artifact = inner
        .artifact_root
        .join(format!("trained-{}", job_id.as_str()));
    fs::rename(temporary_artifact, &final_artifact)
        .map_err(|_| TrainingFailureCategory::Filesystem)?;
    let validated = match ArtifactLoader::load(&final_artifact) {
        Ok(artifact) => artifact,
        Err(_) => {
            let _ = fs::remove_dir_all(&final_artifact);
            return Err(TrainingFailureCategory::Artifact);
        }
    };
    if cancellation.is_cancelled() {
        let _ = fs::remove_dir_all(&final_artifact);
        return Err(TrainingFailureCategory::Cancelled);
    }

    *inner.active_artifact.write().await = Some(Arc::new(validated));
    Ok(())
}

pub fn validate_dataset(
    allowed_root: impl AsRef<Path>,
    requested_path: impl AsRef<Path>,
) -> Result<ValidatedDataset, TrainingError> {
    let canonical_root = validate_existing_directory(allowed_root.as_ref())?;
    let relative_path = requested_path.as_ref();
    validate_relative_path(relative_path)?;
    reject_symlinks_between(&canonical_root, relative_path)?;

    let candidate = canonical_root.join(relative_path);
    let metadata = fs::symlink_metadata(&candidate).map_err(|error| {
        if error.kind() == io::ErrorKind::NotFound {
            TrainingError::DatasetMissing
        } else {
            TrainingError::Filesystem
        }
    })?;
    if metadata.file_type().is_symlink() {
        return Err(TrainingError::UnsafePath(
            "symbolic links are not allowed".to_owned(),
        ));
    }
    if !metadata.is_file() {
        return Err(TrainingError::DatasetNotFile);
    }
    if metadata.len() > MAX_DATASET_BYTES {
        return Err(TrainingError::DatasetTooLarge);
    }

    let canonical_path = candidate
        .canonicalize()
        .map_err(|_| TrainingError::Filesystem)?;
    if !canonical_path.starts_with(&canonical_root) {
        return Err(TrainingError::UnsafePath(
            "dataset must remain within the configured root".to_owned(),
        ));
    }

    let mut bytes = Vec::with_capacity(metadata.len() as usize);
    File::open(&canonical_path)
        .map_err(|_| TrainingError::Filesystem)?
        .take(MAX_DATASET_BYTES + 1)
        .read_to_end(&mut bytes)
        .map_err(|_| TrainingError::Filesystem)?;
    if bytes.len() as u64 > MAX_DATASET_BYTES {
        return Err(TrainingError::DatasetTooLarge);
    }
    let text =
        std::str::from_utf8(&bytes).map_err(|_| TrainingError::InvalidDatasetRecord { line: 1 })?;
    let mut pair_count = 0_usize;
    for (index, line) in text.lines().enumerate() {
        let line_number = index + 1;
        if line.trim().is_empty() {
            return Err(TrainingError::InvalidDatasetRecord { line: line_number });
        }
        let pair: PreferencePair = serde_json::from_str(line)
            .map_err(|_| TrainingError::InvalidDatasetRecord { line: line_number })?;
        if pair.query.trim().is_empty()
            || pair.strong_response.trim().is_empty()
            || pair.weak_response.trim().is_empty()
        {
            return Err(TrainingError::EmptyDatasetField { line: line_number });
        }
        let _ = pair.label;
        pair_count = pair_count.saturating_add(1);
    }
    if pair_count < MIN_PREFERENCE_PAIRS {
        return Err(TrainingError::InsufficientData { actual: pair_count });
    }

    Ok(ValidatedDataset {
        path: canonical_path,
        pair_count,
        size_bytes: bytes.len() as u64,
    })
}

fn validate_existing_directory(path: &Path) -> Result<PathBuf, TrainingError> {
    let metadata = fs::symlink_metadata(path).map_err(|error| {
        if error.kind() == io::ErrorKind::NotFound {
            TrainingError::DatasetMissing
        } else {
            TrainingError::Filesystem
        }
    })?;
    if metadata.file_type().is_symlink() {
        return Err(TrainingError::UnsafePath(
            "symbolic-link roots are not allowed".to_owned(),
        ));
    }
    if !metadata.is_dir() {
        return Err(TrainingError::UnsafePath(
            "configured root must be a directory".to_owned(),
        ));
    }
    path.canonicalize().map_err(|_| TrainingError::Filesystem)
}

fn prepare_artifact_root(path: &Path) -> Result<PathBuf, TrainingError> {
    if !path.exists() {
        fs::create_dir_all(path).map_err(|_| TrainingError::Filesystem)?;
    }
    validate_existing_directory(path)
}

fn validate_relative_path(path: &Path) -> Result<(), TrainingError> {
    let text = path.to_string_lossy();
    let portable_segments_are_safe = !text.is_empty()
        && !text.starts_with(['/', '\\'])
        && !text.contains('\0')
        && !text.contains(':')
        && text
            .split(['/', '\\'])
            .all(|segment| !segment.is_empty() && segment != "." && segment != "..");
    let native_components_are_safe = path
        .components()
        .all(|component| matches!(component, Component::Normal(_)));
    if !portable_segments_are_safe || !native_components_are_safe {
        return Err(TrainingError::UnsafePath(
            "dataset path must be a safe relative path".to_owned(),
        ));
    }
    Ok(())
}

fn reject_symlinks_between(root: &Path, relative_path: &Path) -> Result<(), TrainingError> {
    let mut candidate = root.to_path_buf();
    for component in relative_path.components() {
        if let Component::Normal(segment) = component {
            candidate.push(segment);
            let metadata = fs::symlink_metadata(&candidate).map_err(|error| {
                if error.kind() == io::ErrorKind::NotFound {
                    TrainingError::DatasetMissing
                } else {
                    TrainingError::Filesystem
                }
            })?;
            if metadata.file_type().is_symlink() {
                return Err(TrainingError::UnsafePath(
                    "symbolic links are not allowed".to_owned(),
                ));
            }
        }
    }
    Ok(())
}

fn reject_symlinks_recursively(root: &Path) -> io::Result<()> {
    let metadata = fs::symlink_metadata(root)?;
    if metadata.file_type().is_symlink() {
        return Err(io::Error::new(
            io::ErrorKind::InvalidData,
            "symbolic links are not allowed",
        ));
    }
    if metadata.is_dir() {
        for entry in fs::read_dir(root)? {
            reject_symlinks_recursively(&entry?.path())?;
        }
    }
    Ok(())
}

fn cleanup_temporary_artifacts(artifact_root: &Path) {
    let Ok(entries) = fs::read_dir(artifact_root) else {
        return;
    };
    for entry in entries.flatten() {
        let file_name = entry.file_name();
        let file_name = file_name.to_string_lossy();
        if file_name.starts_with(".training-") && file_name.ends_with(".tmp") {
            let path = entry.path();
            if fs::symlink_metadata(&path)
                .is_ok_and(|metadata| metadata.is_dir() && !metadata.file_type().is_symlink())
            {
                let _ = fs::remove_dir_all(path);
            }
        }
    }
}

pub fn default_training_timeout() -> Duration {
    Duration::from_secs(60 * 60)
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::future::Future;
    use tempfile::TempDir;

    const RAW_QUERY: &str = "raw-query-secret-7f918a";
    const RAW_STRONG_RESPONSE: &str = "raw-strong-response-secret-34cdd2";
    const RAW_WEAK_RESPONSE: &str = "raw-weak-response-secret-913bef";

    fn preference_record(label: &str) -> String {
        serde_json::json!({
        "query": RAW_QUERY,
        "strong_response": RAW_STRONG_RESPONSE,
        "weak_response": RAW_WEAK_RESPONSE,
        "label": label,
        })
        .to_string()
    }

    fn write_dataset(root: &Path, relative_path: &str, records: &[String]) -> PathBuf {
        let path = root.join(relative_path);
        if let Some(parent) = path.parent() {
            fs::create_dir_all(parent).unwrap();
        }
        fs::write(&path, records.join("\n")).unwrap();
        path
    }

    fn records(count: usize) -> Vec<String> {
        (0..count)
            .map(|index| preference_record(if index % 2 == 0 { "strong" } else { "weak" }))
            .collect()
    }

    fn valid_hyperparameters() -> TrainingHyperparameters {
        TrainingHyperparameters {
            learning_rate: 0.001,
            batch_size: 16,
            epochs: 3,
        }
    }

    fn training_request(dataset_path: impl Into<PathBuf>) -> TrainingRequest {
        TrainingRequest {
            dataset_path: dataset_path.into(),
            hyperparameters: valid_hyperparameters(),
            augmentation: AugmentationPolicy::default(),
        }
    }

    fn valid_manifest() -> &'static str {
        r#"{
"format": "obey.smart-routing.ml-classifier",
"version": 1,
"model_family": "bert_sequence_classification",
"tokenizer_path": "tokenizer.json",
"weights_path": "model.safetensors"
}"#
    }

    fn write_artifact(root: &Path, manifest: &str) {
        fs::create_dir_all(root).unwrap();
        fs::write(root.join("tokenizer.json"), b"tokenizer").unwrap();
        fs::write(root.join("model.safetensors"), b"weights").unwrap();
        fs::write(root.join("manifest.json"), manifest).unwrap();
    }

    fn test_manager(workspace: &TempDir) -> TrainingManager {
        let dataset_root = workspace.path().join("datasets");
        let artifact_root = workspace.path().join("artifacts");
        fs::create_dir(&dataset_root).unwrap();
        TrainingManager::new(dataset_root, artifact_root, None).unwrap()
    }

    async fn assert_finishes<F>(future: F)
    where
        F: Future<Output = ()>,
    {
        tokio::time::timeout(Duration::from_secs(2), future)
            .await
            .expect("operation should finish promptly");
    }

    #[test]
    fn malformed_jsonl_reports_the_exact_line() {
        let directory = TempDir::new().unwrap();
        let mut dataset_records = records(MIN_PREFERENCE_PAIRS - 1);
        dataset_records.push("{not-json".to_owned());
        write_dataset(directory.path(), "malformed.jsonl", &dataset_records);

        let error = validate_dataset(directory.path(), "malformed.jsonl").unwrap_err();

        assert!(matches!(
            error,
            TrainingError::InvalidDatasetRecord {
                line: MIN_PREFERENCE_PAIRS
            }
        ));
    }

    #[test]
    fn fewer_than_one_hundred_pairs_are_rejected() {
        let directory = TempDir::new().unwrap();
        write_dataset(
            directory.path(),
            "small.jsonl",
            &records(MIN_PREFERENCE_PAIRS - 1),
        );

        let error = validate_dataset(directory.path(), "small.jsonl").unwrap_err();

        assert!(matches!(
        error,
        TrainingError::InsufficientData {
        actual
        } if actual == MIN_PREFERENCE_PAIRS - 1
        ));
    }

    #[test]
    fn supported_labels_validate_and_unknown_labels_fail_closed() {
        let directory = TempDir::new().unwrap();
        write_dataset(
            directory.path(),
            "supported-labels.jsonl",
            &records(MIN_PREFERENCE_PAIRS),
        );
        let validated = validate_dataset(directory.path(), "supported-labels.jsonl").unwrap();
        assert_eq!(validated.pair_count(), MIN_PREFERENCE_PAIRS);

        let mut invalid_records = records(MIN_PREFERENCE_PAIRS);
        invalid_records[37] = preference_record("unreviewed");
        write_dataset(directory.path(), "invalid-label.jsonl", &invalid_records);

        let error = validate_dataset(directory.path(), "invalid-label.jsonl").unwrap_err();
        assert!(matches!(
            error,
            TrainingError::InvalidDatasetRecord { line: 38 }
        ));
    }

    #[test]
    fn empty_required_fields_are_rejected_without_echoing_content() {
        let directory = TempDir::new().unwrap();
        let mut dataset_records = records(MIN_PREFERENCE_PAIRS);
        dataset_records[4] = serde_json::json!({
        "query": "   ",
        "strong_response": RAW_STRONG_RESPONSE,
        "weak_response": RAW_WEAK_RESPONSE,
        "label": "strong",
        })
        .to_string();
        write_dataset(directory.path(), "empty-field.jsonl", &dataset_records);

        let error = validate_dataset(directory.path(), "empty-field.jsonl").unwrap_err();
        let display = error.to_string();
        let debug = format!("{error:?}");

        assert!(matches!(
            error,
            TrainingError::EmptyDatasetField { line: 5 }
        ));
        for raw_value in [RAW_QUERY, RAW_STRONG_RESPONSE, RAW_WEAK_RESPONSE] {
            assert!(!display.contains(raw_value));
            assert!(!debug.contains(raw_value));
        }
    }

    #[test]
    fn dataset_paths_must_be_relative_regular_files_within_the_root() {
        let directory = TempDir::new().unwrap();
        let outside = TempDir::new().unwrap();
        let valid_path = write_dataset(
            directory.path(),
            "nested/preferences.jsonl",
            &records(MIN_PREFERENCE_PAIRS),
        );
        write_dataset(
            outside.path(),
            "outside.jsonl",
            &records(MIN_PREFERENCE_PAIRS),
        );

        let validated = validate_dataset(directory.path(), "nested/preferences.jsonl").unwrap();
        assert_eq!(validated.path(), valid_path.canonicalize().unwrap());
        assert!(matches!(
            validate_dataset(directory.path(), "../outside.jsonl"),
            Err(TrainingError::UnsafePath(_))
        ));
        assert!(matches!(
            validate_dataset(directory.path(), &valid_path),
            Err(TrainingError::UnsafePath(_))
        ));
        assert!(matches!(
            validate_dataset(directory.path(), "nested"),
            Err(TrainingError::DatasetNotFile)
        ));
    }

    #[cfg(unix)]
    fn create_file_symlink(target: &Path, link: &Path) -> io::Result<()> {
        std::os::unix::fs::symlink(target, link)
    }

    #[cfg(windows)]
    fn create_file_symlink(target: &Path, link: &Path) -> io::Result<()> {
        std::os::windows::fs::symlink_file(target, link)
    }

    #[test]
    fn dataset_symlinks_are_rejected() {
        let directory = TempDir::new().unwrap();
        let target = write_dataset(
            directory.path(),
            "target.jsonl",
            &records(MIN_PREFERENCE_PAIRS),
        );
        let link = directory.path().join("linked.jsonl");
        if let Err(error) = create_file_symlink(&target, &link) {
            if cfg!(windows)
                && (error.kind() == io::ErrorKind::PermissionDenied
                    || error.raw_os_error() == Some(1314))
            {
                return;
            }
            panic!("failed to create test symlink: {error}");
        }

        let error = validate_dataset(directory.path(), "linked.jsonl").unwrap_err();

        assert!(matches!(error, TrainingError::UnsafePath(_)));
        assert!(!error.to_string().contains(RAW_QUERY));
    }

    #[tokio::test]
    async fn cancellation_wakes_waiters_and_is_idempotent() {
        let token = TrainingCancellationToken::new();
        let waiting_token = token.clone();
        let waiter = tokio::spawn(async move {
            waiting_token.cancelled().await;
        });
        tokio::task::yield_now().await;

        assert!(!token.is_cancelled());
        token.cancel();
        token.cancel();
        assert!(token.is_cancelled());
        assert_finishes(async { waiter.await.unwrap() }).await;
        assert_finishes(token.cancelled()).await;
    }

    #[tokio::test]
    async fn malformed_artifacts_never_become_active() {
        let workspace = TempDir::new().unwrap();
        let manager = test_manager(&workspace);
        let candidate = manager.inner.artifact_root.join(".training-invalid.tmp");
        write_artifact(&candidate, "{not-json");
        let cancellation = TrainingCancellationToken::new();
        let job_id = TrainingJobId("invalid-artifact".to_owned());

        let result = activate_artifact(&manager.inner, &job_id, &candidate, &cancellation).await;

        assert_eq!(result, Err(TrainingFailureCategory::Artifact));
        assert!(manager.active_artifact().await.is_none());
        assert!(!manager
            .inner
            .artifact_root
            .join("trained-invalid-artifact")
            .exists());
    }

    #[tokio::test]
    async fn successful_activation_atomically_publishes_a_validated_artifact() {
        let workspace = TempDir::new().unwrap();
        let manager = test_manager(&workspace);
        let candidate = manager.inner.artifact_root.join(".training-valid.tmp");
        write_artifact(&candidate, valid_manifest());
        let cancellation = TrainingCancellationToken::new();
        let job_id = TrainingJobId("valid-artifact".to_owned());

        activate_artifact(&manager.inner, &job_id, &candidate, &cancellation)
            .await
            .unwrap();

        let expected_root = manager
            .inner
            .artifact_root
            .join("trained-valid-artifact")
            .canonicalize()
            .unwrap();
        let active = manager.active_artifact().await.unwrap();
        assert_eq!(active.root(), expected_root);
        assert!(!candidate.exists());
        assert!(active.tokenizer_path().is_file());
        assert!(active.weights_path().is_file());
    }

    #[tokio::test]
    async fn failed_swap_rolls_back_and_keeps_the_current_artifact() {
        let workspace = TempDir::new().unwrap();
        let manager = test_manager(&workspace);
        let current_root = manager.inner.artifact_root.join("current");
        write_artifact(&current_root, valid_manifest());
        let current = Arc::new(ArtifactLoader::load(&current_root).unwrap());
        *manager.inner.active_artifact.write().await = Some(current.clone());

        let candidate = manager.inner.artifact_root.join(".training-rollback.tmp");
        write_artifact(&candidate, "{not-json");
        let cancellation = TrainingCancellationToken::new();
        let job_id = TrainingJobId("rollback".to_owned());

        let result = activate_artifact(&manager.inner, &job_id, &candidate, &cancellation).await;

        assert_eq!(result, Err(TrainingFailureCategory::Artifact));
        let active = manager.active_artifact().await.unwrap();
        assert!(Arc::ptr_eq(&active, &current));
        assert_eq!(active.root(), current.root());
        assert!(!manager
            .inner
            .artifact_root
            .join("trained-rollback")
            .exists());
    }

    #[tokio::test]
    async fn cancellation_before_activation_keeps_the_current_artifact() {
        let workspace = TempDir::new().unwrap();
        let manager = test_manager(&workspace);
        let current_root = manager.inner.artifact_root.join("current");
        write_artifact(&current_root, valid_manifest());
        let current = Arc::new(ArtifactLoader::load(&current_root).unwrap());
        *manager.inner.active_artifact.write().await = Some(current.clone());

        let candidate = manager.inner.artifact_root.join(".training-cancelled.tmp");
        write_artifact(&candidate, valid_manifest());
        let cancellation = TrainingCancellationToken::new();
        cancellation.cancel();
        let job_id = TrainingJobId("cancelled".to_owned());

        let result = activate_artifact(&manager.inner, &job_id, &candidate, &cancellation).await;

        assert_eq!(result, Err(TrainingFailureCategory::Cancelled));
        assert!(Arc::ptr_eq(
            &manager.active_artifact().await.unwrap(),
            &current
        ));
        assert!(candidate.exists());
        assert!(!manager
            .inner
            .artifact_root
            .join("trained-cancelled")
            .exists());
    }

    #[test]
    fn augmentation_limits_are_enforced_before_backend_selection() {
        let too_many = TrainingRequest {
            augmentation: AugmentationPolicy {
                enabled: true,
                paraphrases_per_pair: MAX_AUGMENTATIONS_PER_PAIR + 1,
                paraphrase_temperature: 1.0,
                back_translation: false,
            },
            ..training_request("preferences.jsonl")
        };
        assert!(matches!(
        too_many.validate(),
        Err(TrainingError::InvalidRequest(message))
        if message.contains("paraphrases_per_pair")
        ));

        let too_hot = TrainingRequest {
            augmentation: AugmentationPolicy {
                enabled: true,
                paraphrases_per_pair: 1,
                paraphrase_temperature: f64::INFINITY,
                back_translation: false,
            },
            ..training_request("preferences.jsonl")
        };
        assert!(matches!(
        too_hot.validate(),
        Err(TrainingError::InvalidRequest(message))
        if message.contains("paraphrase_temperature")
        ));

        let disabled_with_options = TrainingRequest {
            augmentation: AugmentationPolicy {
                enabled: false,
                paraphrases_per_pair: 1,
                paraphrase_temperature: 1.0,
                back_translation: true,
            },
            ..training_request("preferences.jsonl")
        };
        assert!(matches!(
        disabled_with_options.validate(),
        Err(TrainingError::InvalidRequest(message))
        if message.contains("require augmentation")
        ));
    }

    #[test]
    fn bounded_augmentation_is_explicitly_unsupported() {
        let request = TrainingRequest {
            augmentation: AugmentationPolicy {
                enabled: true,
                paraphrases_per_pair: MAX_AUGMENTATIONS_PER_PAIR,
                paraphrase_temperature: MAX_AUGMENTATION_TEMPERATURE,
                back_translation: true,
            },
            ..training_request("preferences.jsonl")
        };

        assert!(matches!(
            request.validate(),
            Err(TrainingError::Unsupported)
        ));
    }

    #[test]
    fn dataset_debug_and_errors_never_include_raw_preference_data() {
        let directory = TempDir::new().unwrap();
        write_dataset(
            directory.path(),
            "private.jsonl",
            &records(MIN_PREFERENCE_PAIRS),
        );
        let dataset = validate_dataset(directory.path(), "private.jsonl").unwrap();
        let backend_request = TrainingBackendRequest {
            dataset,
            output_artifact_path: directory.path().join("artifact"),
            hyperparameters: valid_hyperparameters(),
            augmentation: AugmentationPolicy::default(),
            cancellation: TrainingCancellationToken::new(),
            progress: TrainingProgressReporter::new(),
        };
        let request_debug = format!("{backend_request:?}");

        let mut invalid_records = records(MIN_PREFERENCE_PAIRS);
        invalid_records[0] = preference_record("not-a-label");
        write_dataset(directory.path(), "private-invalid.jsonl", &invalid_records);
        let error = validate_dataset(directory.path(), "private-invalid.jsonl").unwrap_err();
        let error_output = format!("{error:?} {error}");

        for raw_value in [RAW_QUERY, RAW_STRONG_RESPONSE, RAW_WEAK_RESPONSE] {
            assert!(!request_debug.contains(raw_value));
            assert!(!error_output.contains(raw_value));
        }
    }
}
