#![cfg(feature = "ml-router")]

//! Validation and ONNX inference for the optional ML smart router.
//!
//! A supported artifact is a directory containing `manifest.json` with this
//! exact versioned contract:
//!
//! ```json
//! {
//! "format": "obey.smart-routing.ml-classifier",
//! "version": 1,
//! "model_family": "bert_sequence_classification",
//! "tokenizer_path": "tokenizer.json",
//! "weights_path": "model.onnx",
//! "checksums": {
//! "tokenizer_sha256": "<optional 64-character lowercase or uppercase hex>",
//! "weights_sha256": "<optional 64-character lowercase or uppercase hex>"
//! }
//! }
//! ```
//!
//! Both `model.safetensors` (Candle) and `model.onnx` (ONNX Runtime) weights
//! are accepted. When ONNX weights are detected, [`OnnxClassifierBackend`]
//! loads the model via the same `ort` + `tokenizers` stack used by the
//! compression perplexity engine. The ONNX Runtime shared library is resolved
//! from the artifact directory's `.onnxruntime/` subfolder, mirroring the
//! existing `OnnxAssetManager` pattern.

use std::fmt;
use std::fs::{self, File};
use std::io::{self, Read};
use std::path::{Component, Path, PathBuf};

use async_trait::async_trait;
use serde::Deserialize;
use sha2::{Digest, Sha256};
use thiserror::Error;

pub const ARTIFACT_MANIFEST_FILE: &str = "manifest.json";
pub const SUPPORTED_ARTIFACT_FORMAT: &str = "obey.smart-routing.ml-classifier";
pub const SUPPORTED_ARTIFACT_VERSION: u32 = 1;
pub const SUPPORTED_MODEL_FAMILY: &str = "bert_sequence_classification";
pub const MAX_CLASSIFIER_TOKENS: usize = 512;

const MAX_MANIFEST_BYTES: u64 = 64 * 1024;
const SHA256_HEX_LENGTH: usize = 64;
const ONNX_WEIGHTS_SUFFIX: &str = ".onnx";

#[derive(Debug, Clone, PartialEq, Eq, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct ArtifactManifest {
    pub format: String,
    pub version: u32,
    pub model_family: String,
    pub tokenizer_path: String,
    pub weights_path: String,
    #[serde(default)]
    pub checksums: ArtifactChecksums,
}

#[derive(Debug, Clone, Default, PartialEq, Eq, Deserialize)]
#[serde(default, deny_unknown_fields)]
pub struct ArtifactChecksums {
    pub tokenizer_sha256: Option<String>,
    pub weights_sha256: Option<String>,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ValidatedArtifact {
    root: PathBuf,
    manifest: ArtifactManifest,
    tokenizer_path: PathBuf,
    weights_path: PathBuf,
}

impl ValidatedArtifact {
    pub fn root(&self) -> &Path {
        &self.root
    }

    pub fn manifest(&self) -> &ArtifactManifest {
        &self.manifest
    }

    pub fn tokenizer_path(&self) -> &Path {
        &self.tokenizer_path
    }

    pub fn weights_path(&self) -> &Path {
        &self.weights_path
    }
}

#[derive(Debug, Error)]
pub enum MlClassifierError {
    #[error("ML classifier artifact is missing: {path}")]
    MissingArtifact { path: PathBuf },

    #[error("ML classifier artifact path is not a directory: {path}")]
    InvalidArtifactRoot { path: PathBuf },

    #[error("failed to read ML classifier artifact {path}: {source}")]
    ArtifactIo {
        path: PathBuf,
        #[source]
        source: io::Error,
    },

    #[error("ML classifier manifest is invalid at {path}: {reason}")]
    InvalidManifest { path: PathBuf, reason: String },

    #[error("ML classifier artifact is incompatible: {reason}")]
    IncompatibleArtifact { reason: String },

    #[error("unsafe ML classifier artifact path in {field}: {path}")]
    UnsafeArtifactPath { field: &'static str, path: String },

    #[error("ML classifier artifact file is missing for {field}: {path}")]
    MissingArtifactFile { field: &'static str, path: PathBuf },

    #[error("ML classifier artifact file is invalid for {field}: {path}")]
    InvalidArtifactFile { field: &'static str, path: PathBuf },

    #[error("invalid SHA-256 checksum in {field}: expected 64 hexadecimal characters")]
    InvalidChecksum { field: &'static str },

    #[error("SHA-256 checksum mismatch for {field}: {path}")]
    ChecksumMismatch { field: &'static str, path: PathBuf },

    #[error("ML classifier backend failed: {0}")]
    Backend(String),

    #[error("ML classifier backend returned a non-finite score: {0}")]
    NonFiniteScore(f32),
}

pub struct ArtifactLoader;

impl ArtifactLoader {
    pub fn load(root: impl AsRef<Path>) -> Result<ValidatedArtifact, MlClassifierError> {
        let requested_root = root.as_ref();
        if !requested_root.exists() {
            return Err(MlClassifierError::MissingArtifact {
                path: requested_root.to_path_buf(),
            });
        }
        if !requested_root.is_dir() {
            return Err(MlClassifierError::InvalidArtifactRoot {
                path: requested_root.to_path_buf(),
            });
        }

        let canonical_root =
            requested_root
                .canonicalize()
                .map_err(|source| MlClassifierError::ArtifactIo {
                    path: requested_root.to_path_buf(),
                    source,
                })?;
        let manifest_path = canonical_root.join(ARTIFACT_MANIFEST_FILE);
        let manifest_bytes = read_manifest(&manifest_path)?;
        let manifest: ArtifactManifest =
            serde_json::from_slice(&manifest_bytes).map_err(|error| {
                MlClassifierError::InvalidManifest {
                    path: manifest_path.clone(),
                    reason: error.to_string(),
                }
            })?;

        validate_compatibility(&manifest)?;
        validate_checksum_text(
            "checksums.tokenizer_sha256",
            manifest.checksums.tokenizer_sha256.as_deref(),
        )?;
        validate_checksum_text(
            "checksums.weights_sha256",
            manifest.checksums.weights_sha256.as_deref(),
        )?;

        let tokenizer_path =
            resolve_artifact_file(&canonical_root, "tokenizer_path", &manifest.tokenizer_path)?;
        let weights_path =
            resolve_artifact_file(&canonical_root, "weights_path", &manifest.weights_path)?;

        verify_checksum(
            "tokenizer_path",
            &tokenizer_path,
            manifest.checksums.tokenizer_sha256.as_deref(),
        )?;
        verify_checksum(
            "weights_path",
            &weights_path,
            manifest.checksums.weights_sha256.as_deref(),
        )?;

        Ok(ValidatedArtifact {
            root: canonical_root,
            manifest,
            tokenizer_path,
            weights_path,
        })
    }
}

pub trait MlClassifierBackend: Sized {
    fn load(artifact: &ValidatedArtifact) -> Result<Self, MlClassifierError>;

    fn score(&self, text: &str) -> Result<f32, MlClassifierError>;
}

// ---------------------------------------------------------------------------
// ONNX Runtime backend
// ---------------------------------------------------------------------------

use std::sync::{Mutex, OnceLock};

use ort::{session::Session, value::Tensor};
use tokenizers::{EncodeInput, Tokenizer, TruncationParams};

static ML_ORT_INITIALIZED: OnceLock<Result<(), String>> = OnceLock::new();

fn ensure_runtime(runtime_path: &Path) -> Result<(), String> {
    ML_ORT_INITIALIZED
        .get_or_init(|| {
            ort::init_from(runtime_path.display().to_string())
                .with_name("obey-api-gateway-ml-router")
                .commit()
                .map(|_| ())
                .map_err(|error| error.to_string())
        })
        .clone()
}

/// ONNX Runtime-backed classifier. Reuses the same `ort` + `tokenizers`
/// infrastructure as the compression perplexity engine.
pub struct OnnxClassifierBackend {
    tokenizer: Tokenizer,
    session: Mutex<Session>,
}

impl std::fmt::Debug for OnnxClassifierBackend {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("OnnxClassifierBackend")
            .field("name", &"obey-smart-routing-onnx")
            .finish_non_exhaustive()
    }
}

impl MlClassifierBackend for OnnxClassifierBackend {
    fn load(artifact: &ValidatedArtifact) -> Result<Self, MlClassifierError> {
        let weights_path = artifact.weights_path();
        if !weights_path.is_file() {
            return Err(MlClassifierError::MissingArtifactFile {
                field: "weights_path",
                path: weights_path.to_path_buf(),
            });
        }

        let artifact_root = artifact.root();
        let runtime_dir = artifact_root.join(".onnxruntime");
        let runtime_library_name = if cfg!(target_os = "windows") {
            "onnxruntime.dll"
        } else if cfg!(target_os = "macos") {
            "libonnxruntime.dylib"
        } else {
            "libonnxruntime.so"
        };
        let runtime_path = runtime_dir.join(runtime_library_name);
        if !runtime_path.is_file() {
            return Err(MlClassifierError::Backend(format!(
                "ONNX Runtime not found at {}; install runtime assets alongside the ML model",
                runtime_path.display()
            )));
        }
        ensure_runtime(&runtime_path).map_err(MlClassifierError::Backend)?;

        let tokenizer_path = artifact.tokenizer_path();
        let mut tokenizer = Tokenizer::from_file(tokenizer_path).map_err(|error| {
            MlClassifierError::Backend(format!(
                "failed to load tokenizer `{}`: {error}",
                tokenizer_path.display()
            ))
        })?;
        tokenizer
            .with_truncation(Some(TruncationParams {
                max_length: MAX_CLASSIFIER_TOKENS,
                ..TruncationParams::default()
            }))
            .map_err(|error| {
                MlClassifierError::Backend(format!(
                    "failed to configure tokenizer truncation: {error}"
                ))
            })?;

        let session = Session::builder()
            .and_then(|builder| builder.commit_from_file(weights_path))
            .map_err(|error| {
                MlClassifierError::Backend(format!(
                    "failed to load model `{}`: {error}",
                    weights_path.display()
                ))
            })?;

        Ok(Self {
            tokenizer,
            session: Mutex::new(session),
        })
    }

    fn score(&self, text: &str) -> Result<f32, MlClassifierError> {
        if text.trim().is_empty() {
            return Ok(0.0);
        }
        let encoding = self
            .tokenizer
            .encode(EncodeInput::Single(text.to_owned().into()), true)
            .map_err(|error| MlClassifierError::Backend(format!("tokenization failed: {error}")))?;

        let sequence_length = encoding.get_ids().len();
        if sequence_length == 0 {
            return Ok(0.0);
        }

        let input_ids: Vec<i64> = encoding.get_ids().iter().map(|v| i64::from(*v)).collect();
        let attention_mask: Vec<i64> = encoding
            .get_attention_mask()
            .iter()
            .map(|v| i64::from(*v))
            .collect();

        let input_ids_tensor =
            Tensor::from_array(([1usize, sequence_length], input_ids)).map_err(|error| {
                MlClassifierError::Backend(format!("input tensor creation failed: {error}"))
            })?;
        let attention_mask_tensor = Tensor::from_array(([1usize, sequence_length], attention_mask))
            .map_err(|error| {
                MlClassifierError::Backend(format!("attention tensor creation failed: {error}"))
            })?;

        let mut session = self
            .session
            .lock()
            .unwrap_or_else(|poisoned| poisoned.into_inner());
        let outputs = session
            .run(ort::inputs![
                "input_ids" => input_ids_tensor,
                "attention_mask" => attention_mask_tensor
            ])
            .map_err(|error| MlClassifierError::Backend(format!("inference failed: {error}")))?;

        let output = outputs
            .get("logits")
            .or_else(|| outputs.get("output"))
            .ok_or_else(|| MlClassifierError::Backend("model returned no output".to_string()))?;

        let (_shape, logits) = output
            .try_extract_tensor::<f32>()
            .map_err(|error| MlClassifierError::Backend(format!("output extraction: {error}")))?;

        Ok(logits_to_complexity(&logits))
    }
}

/// Map raw model logits to a complexity score in `[0.0, 1.0]`.
///
/// Three layouts are supported:
/// 1. Single regression logit → sigmoid
/// 2. Two-class logits `[keep, complex]` → softmax → probability of complex
/// 3. Three-class logits `[simple, moderate, complex]` → weighted expectation
fn logits_to_complexity(logits: &[f32]) -> f32 {
    match logits.len() {
        0 => 0.5,
        1 => {
            let z = logits[0];
            if !z.is_finite() {
                return 0.5;
            }
            1.0 / (1.0 + (-z).exp())
        }
        2 => {
            let (a, b) = (logits[0], logits[1]);
            if !a.is_finite() || !b.is_finite() {
                return 0.5;
            }
            let ea = a.exp();
            let eb = b.exp();
            let sum = ea + eb;
            if sum == 0.0 || !sum.is_finite() {
                0.5
            } else {
                eb / sum
            }
        }
        _ => {
            let targets = [0.15_f32, 0.50, 0.85];
            let mut max = f32::NEG_INFINITY;
            for &v in &logits[..3] {
                if v.is_finite() && v > max {
                    max = v;
                }
            }
            if !max.is_finite() {
                return 0.5;
            }
            let mut sum_exp = 0.0_f32;
            let mut weighted = 0.0_f32;
            for (i, &logit) in logits[..3].iter().enumerate() {
                let clamped = if logit.is_finite() { logit } else { max };
                let exp = (clamped - max).exp();
                sum_exp += exp;
                weighted += targets[i] * exp;
            }
            if sum_exp == 0.0 || !sum_exp.is_finite() {
                0.5
            } else {
                weighted / sum_exp
            }
        }
    }
}

pub struct MlClassifier<B> {
    backend: B,
}

impl<B: MlClassifierBackend> MlClassifier<B> {
    pub fn load(artifact_root: impl AsRef<Path>) -> Result<Self, MlClassifierError> {
        let artifact = ArtifactLoader::load(artifact_root)?;
        let backend = B::load(&artifact)?;
        Ok(Self { backend })
    }

    pub fn score(&self, text: &str) -> Result<NormalizedScore, MlClassifierError> {
        NormalizedScore::new(self.backend.score(text)?)
    }
}

#[derive(Debug, Clone, Copy, PartialEq, PartialOrd)]
pub struct NormalizedScore(f32);

impl NormalizedScore {
    pub fn new(score: f32) -> Result<Self, MlClassifierError> {
        if !score.is_finite() {
            return Err(MlClassifierError::NonFiniteScore(score));
        }
        Ok(Self(score.clamp(0.0, 1.0)))
    }

    pub fn value(self) -> f32 {
        self.0
    }
}

// ---------------------------------------------------------------------------
// SmartRouter integration adapter
// ---------------------------------------------------------------------------

use crate::smart_routing::heuristic::visit_text_content;
use crate::smart_routing::{
    ClassifierFailure, ClassifierInput, ClassifierOutput, OptionalClassifier,
};

/// Adapter wrapping [`MlClassifier<OnnxClassifierBackend>`] as a thread-safe
/// [`OptionalClassifier`] suitable for injection into [`SmartRouter`].
pub struct OnnxMlAdapter {
    inner: std::sync::Mutex<MlClassifier<OnnxClassifierBackend>>,
}

impl OnnxMlAdapter {
    pub fn load(artifact_root: impl AsRef<Path>) -> Result<Self, MlClassifierError> {
        Ok(Self {
            inner: std::sync::Mutex::new(MlClassifier::<OnnxClassifierBackend>::load(
                artifact_root,
            )?),
        })
    }
}

impl std::fmt::Debug for OnnxMlAdapter {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("OnnxMlAdapter").finish_non_exhaustive()
    }
}

#[async_trait]
impl OptionalClassifier for OnnxMlAdapter {
    async fn classify(
        &self,
        input: ClassifierInput<'_>,
    ) -> Result<ClassifierOutput, ClassifierFailure> {
        let text = extract_request_text(input.request);
        let classifier = self
            .inner
            .lock()
            .unwrap_or_else(|poisoned| poisoned.into_inner());
        match classifier.score(&text) {
            Ok(score) => Ok(ClassifierOutput {
                score: f64::from(score.value()),
            }),
            Err(error) => {
                tracing::warn!(error = %error, "ML classifier inference failed");
                Err(ClassifierFailure::Backend)
            }
        }
    }
}

fn extract_request_text(request: &crate::models::openai::OpenAIRequest) -> String {
    let mut text = String::with_capacity(1024);
    for message in &request.messages {
        visit_text_content(message, |part| {
            if !text.is_empty() {
                text.push(' ');
            }
            let remaining = MAX_CLASSIFIER_INPUT_CHARS.saturating_sub(text.len());
            if remaining == 0 {
                return;
            }
            if part.len() <= remaining {
                text.push_str(part);
            } else {
                text.push_str(&part[..remaining]);
            }
        });
    }
    text
}

const MAX_CLASSIFIER_INPUT_CHARS: usize = 8_192;

pub fn truncate_token_ids(token_ids: &mut Vec<u32>) -> bool {
    if token_ids.len() <= MAX_CLASSIFIER_TOKENS {
        return false;
    }
    token_ids.truncate(MAX_CLASSIFIER_TOKENS);
    true
}

fn read_manifest(path: &Path) -> Result<Vec<u8>, MlClassifierError> {
    let metadata = fs::metadata(path).map_err(|source| {
        if source.kind() == io::ErrorKind::NotFound {
            MlClassifierError::MissingArtifactFile {
                field: "manifest",
                path: path.to_path_buf(),
            }
        } else {
            MlClassifierError::ArtifactIo {
                path: path.to_path_buf(),
                source,
            }
        }
    })?;
    if !metadata.is_file() {
        return Err(MlClassifierError::InvalidArtifactFile {
            field: "manifest",
            path: path.to_path_buf(),
        });
    }
    if metadata.len() > MAX_MANIFEST_BYTES {
        return Err(MlClassifierError::InvalidManifest {
            path: path.to_path_buf(),
            reason: format!("manifest exceeds {MAX_MANIFEST_BYTES} bytes"),
        });
    }
    fs::read(path).map_err(|source| MlClassifierError::ArtifactIo {
        path: path.to_path_buf(),
        source,
    })
}

fn validate_compatibility(manifest: &ArtifactManifest) -> Result<(), MlClassifierError> {
    if manifest.format != SUPPORTED_ARTIFACT_FORMAT {
        return Err(MlClassifierError::IncompatibleArtifact {
            reason: format!(
                "unsupported format {:?}; expected {:?}",
                manifest.format, SUPPORTED_ARTIFACT_FORMAT
            ),
        });
    }
    if manifest.version != SUPPORTED_ARTIFACT_VERSION {
        return Err(MlClassifierError::IncompatibleArtifact {
            reason: format!(
                "unsupported version {}; expected {}",
                manifest.version, SUPPORTED_ARTIFACT_VERSION
            ),
        });
    }
    if manifest.model_family != SUPPORTED_MODEL_FAMILY {
        return Err(MlClassifierError::IncompatibleArtifact {
            reason: format!(
                "unsupported model family {:?}; expected {:?}",
                manifest.model_family, SUPPORTED_MODEL_FAMILY
            ),
        });
    }
    Ok(())
}

fn resolve_artifact_file(
    canonical_root: &Path,
    field: &'static str,
    relative_path: &str,
) -> Result<PathBuf, MlClassifierError> {
    validate_relative_path(field, relative_path)?;
    let candidate = canonical_root.join(relative_path);
    let canonical_path = candidate.canonicalize().map_err(|source| {
        if source.kind() == io::ErrorKind::NotFound {
            MlClassifierError::MissingArtifactFile {
                field,
                path: candidate.clone(),
            }
        } else {
            MlClassifierError::ArtifactIo {
                path: candidate.clone(),
                source,
            }
        }
    })?;

    if !canonical_path.starts_with(canonical_root) {
        return Err(MlClassifierError::UnsafeArtifactPath {
            field,
            path: relative_path.to_owned(),
        });
    }
    if !canonical_path.is_file() {
        return Err(MlClassifierError::InvalidArtifactFile {
            field,
            path: canonical_path,
        });
    }
    Ok(canonical_path)
}

fn validate_relative_path(field: &'static str, path_text: &str) -> Result<(), MlClassifierError> {
    let path = Path::new(path_text);
    let portable_segments_are_safe = !path_text.is_empty()
        && !path_text.starts_with(['/', '\\'])
        && !path_text.contains('\0')
        && !path_text.contains(':')
        && path_text
            .split(['/', '\\'])
            .all(|segment| !segment.is_empty() && segment != "." && segment != "..");
    let native_components_are_safe = path
        .components()
        .all(|component| matches!(component, Component::Normal(_)));

    if !portable_segments_are_safe || !native_components_are_safe {
        return Err(MlClassifierError::UnsafeArtifactPath {
            field,
            path: path_text.to_owned(),
        });
    }
    Ok(())
}

fn validate_checksum_text(
    field: &'static str,
    checksum: Option<&str>,
) -> Result<(), MlClassifierError> {
    if let Some(checksum) = checksum {
        if checksum.len() != SHA256_HEX_LENGTH
            || !checksum.bytes().all(|byte| byte.is_ascii_hexdigit())
        {
            return Err(MlClassifierError::InvalidChecksum { field });
        }
    }
    Ok(())
}

fn verify_checksum(
    field: &'static str,
    path: &Path,
    expected: Option<&str>,
) -> Result<(), MlClassifierError> {
    let Some(expected) = expected else {
        return Ok(());
    };

    let mut file = File::open(path).map_err(|source| MlClassifierError::ArtifactIo {
        path: path.to_path_buf(),
        source,
    })?;
    let mut digest = Sha256::new();
    let mut buffer = [0_u8; 16 * 1024];
    loop {
        let bytes_read =
            file.read(&mut buffer)
                .map_err(|source| MlClassifierError::ArtifactIo {
                    path: path.to_path_buf(),
                    source,
                })?;
        if bytes_read == 0 {
            break;
        }
        digest.update(&buffer[..bytes_read]);
    }
    let actual: String = digest
        .finalize()
        .iter()
        .map(|byte| format!("{byte:02x}"))
        .collect();
    if !actual.eq_ignore_ascii_case(expected) {
        return Err(MlClassifierError::ChecksumMismatch {
            field,
            path: path.to_path_buf(),
        });
    }
    Ok(())
}

impl fmt::Display for NormalizedScore {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        self.0.fmt(formatter)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::fs;

    use tempfile::TempDir;

    fn write_artifact(root: &Path, manifest: &str) {
        fs::write(root.join("tokenizer.json"), b"tokenizer").unwrap();
        fs::write(root.join("model.safetensors"), b"weights").unwrap();
        fs::write(root.join(ARTIFACT_MANIFEST_FILE), manifest).unwrap();
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

    #[test]
    fn validates_supported_artifact() {
        let directory = TempDir::new().unwrap();
        write_artifact(directory.path(), valid_manifest());

        let artifact = ArtifactLoader::load(directory.path()).unwrap();

        assert_eq!(artifact.manifest().version, SUPPORTED_ARTIFACT_VERSION);
        assert_eq!(artifact.manifest().model_family, SUPPORTED_MODEL_FAMILY);
        assert!(artifact.tokenizer_path().is_absolute());
        assert!(artifact.weights_path().is_absolute());
    }

    #[test]
    fn missing_artifact_is_explicit() {
        let directory = TempDir::new().unwrap();
        let missing = directory.path().join("not-present");

        let error = ArtifactLoader::load(&missing).unwrap_err();

        assert!(matches!(error, MlClassifierError::MissingArtifact { path } if path == missing));
    }

    #[test]
    fn missing_manifest_is_explicit() {
        let directory = TempDir::new().unwrap();

        let error = ArtifactLoader::load(directory.path()).unwrap_err();

        assert!(matches!(
        error,
        MlClassifierError::MissingArtifactFile {
        field: "manifest",
        path
        } if path.ends_with(ARTIFACT_MANIFEST_FILE)
        ));
    }

    #[test]
    fn missing_model_is_explicit() {
        let directory = TempDir::new().unwrap();
        write_artifact(directory.path(), valid_manifest());
        fs::remove_file(directory.path().join("model.safetensors")).unwrap();

        let error = ArtifactLoader::load(directory.path()).unwrap_err();

        assert!(matches!(
        error,
        MlClassifierError::MissingArtifactFile {
        field: "weights_path",
        path
        } if path.ends_with("model.safetensors")
        ));
    }

    #[test]
    fn invalid_artifact_root_is_explicit() {
        let directory = TempDir::new().unwrap();
        let artifact_path = directory.path().join("artifact-file");
        fs::write(&artifact_path, b"not a directory").unwrap();

        let error = ArtifactLoader::load(&artifact_path).unwrap_err();

        assert!(matches!(
        error,
        MlClassifierError::InvalidArtifactRoot { path } if path == artifact_path
        ));
    }

    #[test]
    fn invalid_manifest_is_explicit() {
        let directory = TempDir::new().unwrap();
        write_artifact(directory.path(), "{not-json");

        let error = ArtifactLoader::load(directory.path()).unwrap_err();

        assert!(matches!(error, MlClassifierError::InvalidManifest { .. }));
    }

    #[test]
    fn incompatible_format_is_refused() {
        let directory = TempDir::new().unwrap();
        let manifest = valid_manifest().replace(
            SUPPORTED_ARTIFACT_FORMAT,
            "obey.smart-routing.unsupported-classifier",
        );
        write_artifact(directory.path(), &manifest);

        let error = ArtifactLoader::load(directory.path()).unwrap_err();

        assert!(matches!(
        error,
        MlClassifierError::IncompatibleArtifact { reason }
        if reason.contains("unsupported format")
        ));
    }

    #[test]
    fn incompatible_version_is_refused() {
        let directory = TempDir::new().unwrap();
        let manifest = valid_manifest().replace("\"version\": 1", "\"version\": 2");
        write_artifact(directory.path(), &manifest);

        let error = ArtifactLoader::load(directory.path()).unwrap_err();

        assert!(matches!(
        error,
        MlClassifierError::IncompatibleArtifact { reason }
        if reason.contains("unsupported version")
        ));
    }

    #[test]
    fn unsafe_paths_are_refused() {
        let directory = TempDir::new().unwrap();
        let manifest = valid_manifest().replace("tokenizer.json", "../tokenizer.json");
        write_artifact(directory.path(), &manifest);

        let error = ArtifactLoader::load(directory.path()).unwrap_err();

        assert!(matches!(
            error,
            MlClassifierError::UnsafeArtifactPath {
                field: "tokenizer_path",
                ..
            }
        ));
    }

    #[test]
    fn checksum_mismatch_is_refused() {
        let directory = TempDir::new().unwrap();
        let manifest = r#"{
            "format": "obey.smart-routing.ml-classifier",
            "version": 1,
            "model_family": "bert_sequence_classification",
            "tokenizer_path": "tokenizer.json",
            "weights_path": "model.safetensors",
            "checksums": {
                "tokenizer_sha256": "0000000000000000000000000000000000000000000000000000000000000000"
            }
        }"#;
        write_artifact(directory.path(), manifest);

        let error = ArtifactLoader::load(directory.path()).unwrap_err();

        assert!(matches!(
            error,
            MlClassifierError::ChecksumMismatch {
                field: "tokenizer_path",
                ..
            }
        ));
    }

    #[test]
    fn finite_scores_are_clamped() {
        assert_eq!(NormalizedScore::new(-0.25).unwrap().value(), 0.0);
        assert_eq!(NormalizedScore::new(0.25).unwrap().value(), 0.25);
        assert_eq!(NormalizedScore::new(1.25).unwrap().value(), 1.0);
    }

    #[test]
    fn non_finite_scores_are_rejected() {
        assert!(matches!(
        NormalizedScore::new(f32::NAN),
        Err(MlClassifierError::NonFiniteScore(value)) if value.is_nan()
        ));
        assert!(matches!(
        NormalizedScore::new(f32::INFINITY),
        Err(MlClassifierError::NonFiniteScore(value)) if value == f32::INFINITY
        ));
        assert!(matches!(
        NormalizedScore::new(f32::NEG_INFINITY),
        Err(MlClassifierError::NonFiniteScore(value)) if value == f32::NEG_INFINITY
        ));
    }

    #[test]
    fn token_ids_are_truncated_to_backend_limit() {
        let mut token_ids: Vec<u32> = (0..600).collect();

        assert!(truncate_token_ids(&mut token_ids));
        assert_eq!(token_ids.len(), MAX_CLASSIFIER_TOKENS);
        assert_eq!(token_ids.last(), Some(&511));
        assert!(!truncate_token_ids(&mut token_ids));
    }
}
