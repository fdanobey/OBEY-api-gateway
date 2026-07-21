//! ONNX-backed token retention scorer for the perplexity engine.

use std::{
    path::Path,
    sync::{Arc, Mutex, OnceLock},
};

use ort::{session::Session, value::Tensor};
use tokenizers::{EncodeInput, Tokenizer, TruncationParams};

use super::perplexity::{
    RedundancyScorer, RedundancyScorerLoader, ScoreRequest, ScorerError, ScorerKind,
    ScorerMetadata, TokenCandidate,
};
use crate::compression::assets::OnnxAssetManager;

const MODEL_NAME: &str = "kompress-small-onnx";
const DEFAULT_MAX_SEQUENCE_LENGTH: usize = 8_192;
static ORT_INITIALIZED: OnceLock<Result<(), String>> = OnceLock::new();

#[derive(Debug, Default)]
pub struct OnnxRedundancyScorerLoader;

impl OnnxRedundancyScorerLoader {
    pub fn new() -> Self {
        Self
    }
}

impl RedundancyScorerLoader for OnnxRedundancyScorerLoader {
    fn load(&self, model_path: &Path) -> Result<Arc<dyn RedundancyScorer>, ScorerError> {
        if !model_path.is_file() {
            return Err(ScorerError::ModelAssetUnavailable {
                path: model_path.to_owned(),
            });
        }
        let tokenizer_path = OnnxAssetManager::tokenizer_path(model_path);
        let external_data_path = OnnxAssetManager::external_data_path(model_path);
        for required in [&tokenizer_path, &external_data_path] {
            if !required.is_file() {
                return Err(ScorerError::ModelAssetUnavailable {
                    path: required.to_path_buf(),
                });
            }
        }
        let runtime_path = OnnxAssetManager::runtime_library_path(model_path).map_err(|error| {
            ScorerError::RuntimeUnavailable {
                path: model_path.to_owned(),
                reason: error.to_string(),
            }
        })?;
        if !runtime_path.is_file() {
            return Err(ScorerError::RuntimeUnavailable {
                path: runtime_path,
                reason: "install the ONNX runtime from the admin Compression page".to_owned(),
            });
        }
        ensure_runtime(&runtime_path).map_err(|reason| ScorerError::RuntimeUnavailable {
            path: runtime_path,
            reason,
        })?;

        let mut tokenizer =
            Tokenizer::from_file(&tokenizer_path).map_err(|error| ScorerError::ScoringFailed {
                scorer: MODEL_NAME.to_owned(),
                reason: format!(
                    "failed to load tokenizer `{}`: {error}",
                    tokenizer_path.display()
                ),
            })?;
        tokenizer
            .with_truncation(Some(TruncationParams {
                max_length: DEFAULT_MAX_SEQUENCE_LENGTH,
                ..TruncationParams::default()
            }))
            .map_err(|error| ScorerError::ScoringFailed {
                scorer: MODEL_NAME.to_owned(),
                reason: format!("failed to configure tokenizer truncation: {error}"),
            })?;
        let session = Session::builder()
            .and_then(|builder| builder.commit_from_file(model_path))
            .map_err(|error| ScorerError::ScoringFailed {
                scorer: MODEL_NAME.to_owned(),
                reason: format!("failed to load model `{}`: {error}", model_path.display()),
            })?;

        Ok(Arc::new(OnnxRedundancyScorer {
            tokenizer,
            session: Mutex::new(session),
        }))
    }
}

fn ensure_runtime(runtime_path: &Path) -> Result<(), String> {
    ORT_INITIALIZED
        .get_or_init(|| {
            ort::init_from(runtime_path.display().to_string())
                .with_name("obey-api-gateway")
                .commit()
                .map(|_| ())
                .map_err(|error| error.to_string())
        })
        .clone()
}

pub struct OnnxRedundancyScorer {
    tokenizer: Tokenizer,
    session: Mutex<Session>,
}

impl std::fmt::Debug for OnnxRedundancyScorer {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        formatter
            .debug_struct("OnnxRedundancyScorer")
            .field("name", &MODEL_NAME)
            .finish_non_exhaustive()
    }
}

impl RedundancyScorer for OnnxRedundancyScorer {
    fn metadata(&self) -> Result<ScorerMetadata, ScorerError> {
        Ok(ScorerMetadata {
            name: MODEL_NAME.to_owned(),
            kind: ScorerKind::ExternalModel,
        })
    }

    fn score_coarse(&self, request: &ScoreRequest<'_>) -> Result<f32, ScorerError> {
        if request.text.trim().is_empty() {
            return Ok(0.0);
        }
        let encoding = self
            .tokenizer
            .encode(EncodeInput::Single(request.text.into()), true)
            .map_err(scoring_error)?;
        let scores = self.run_encoding(&encoding)?;
        let attention = encoding.get_attention_mask();
        let special = encoding.get_special_tokens_mask();
        let mut total = 0.0f32;
        let mut count = 0usize;
        for (index, score) in scores.into_iter().enumerate() {
            if attention.get(index).copied().unwrap_or_default() != 0
                && special.get(index).copied().unwrap_or(1) == 0
            {
                total += score;
                count += 1;
            }
        }
        Ok(if count == 0 {
            1.0
        } else {
            total / count as f32
        })
    }

    fn score_tokens(
        &self,
        _request: &ScoreRequest<'_>,
        tokens: &[TokenCandidate<'_>],
    ) -> Result<Vec<f32>, ScorerError> {
        if tokens.is_empty() {
            return Ok(Vec::new());
        }
        let words = tokens.iter().map(|token| token.text).collect::<Vec<_>>();
        let encoding = self
            .tokenizer
            .encode(EncodeInput::Single(words.into()), true)
            .map_err(scoring_error)?;
        let subword_scores = self.run_encoding(&encoding)?;
        Ok(aggregate_word_scores(
            tokens.len(),
            encoding.get_word_ids(),
            &subword_scores,
        ))
    }
}

impl OnnxRedundancyScorer {
    fn run_encoding(&self, encoding: &tokenizers::Encoding) -> Result<Vec<f32>, ScorerError> {
        let sequence_length = encoding.get_ids().len();
        let input_ids = encoding
            .get_ids()
            .iter()
            .map(|value| i64::from(*value))
            .collect::<Vec<_>>();
        let attention_mask = encoding
            .get_attention_mask()
            .iter()
            .map(|value| i64::from(*value))
            .collect::<Vec<_>>();
        let input_ids = Tensor::from_array(([1usize, sequence_length], input_ids))
            .map_err(|error| scoring_error(error.to_string()))?;
        let attention_mask = Tensor::from_array(([1usize, sequence_length], attention_mask))
            .map_err(|error| scoring_error(error.to_string()))?;
        let mut session = self
            .session
            .lock()
            .unwrap_or_else(|poisoned| poisoned.into_inner());
        let outputs = session
            .run(ort::inputs![
                "input_ids" => input_ids,
                "attention_mask" => attention_mask
            ])
            .map_err(|error| scoring_error(error.to_string()))?;
        let output = outputs
            .get("logits")
            .or_else(|| outputs.get("output"))
            .ok_or_else(|| invalid_output("model returned no logits output"))?;
        let (shape, logits) = output
            .try_extract_tensor::<f32>()
            .map_err(|error| invalid_output(error.to_string()))?;
        logits_to_retention_scores(shape, logits, sequence_length)
    }
}

fn logits_to_retention_scores(
    shape: &[i64],
    logits: &[f32],
    sequence_length: usize,
) -> Result<Vec<f32>, ScorerError> {
    if shape.len() != 3 || shape[0] != 1 || shape[1] != sequence_length as i64 || shape[2] < 2 {
        return Err(invalid_output(format!(
            "expected logits shaped [1, {sequence_length}, >=2], got {shape:?}"
        )));
    }
    let classes = shape[2] as usize;
    if logits.len() < sequence_length.saturating_mul(classes) {
        return Err(invalid_output("logits buffer is shorter than its shape"));
    }
    Ok((0..sequence_length)
        .map(|index| {
            let removable = logits[index * classes];
            let keep = logits[index * classes + 1];
            binary_softmax_keep(removable, keep)
        })
        .collect())
}

fn binary_softmax_keep(removable: f32, keep: f32) -> f32 {
    if !removable.is_finite() || !keep.is_finite() {
        return 1.0;
    }
    let maximum = removable.max(keep);
    let remove_exp = (removable - maximum).exp();
    let keep_exp = (keep - maximum).exp();
    keep_exp / (remove_exp + keep_exp)
}

fn aggregate_word_scores(
    token_count: usize,
    word_ids: &[Option<u32>],
    subword_scores: &[f32],
) -> Vec<f32> {
    let mut sums = vec![0.0f32; token_count];
    let mut counts = vec![0usize; token_count];
    for (word_id, score) in word_ids.iter().zip(subword_scores) {
        let Some(index) = word_id.and_then(|value| usize::try_from(value).ok()) else {
            continue;
        };
        if index < token_count {
            sums[index] += score.clamp(0.0, 1.0);
            counts[index] += 1;
        }
    }
    sums.into_iter()
        .zip(counts)
        .map(
            |(sum, count)| {
                if count == 0 {
                    1.0
                } else {
                    sum / count as f32
                }
            },
        )
        .collect()
}

fn scoring_error(error: impl ToString) -> ScorerError {
    ScorerError::ScoringFailed {
        scorer: MODEL_NAME.to_owned(),
        reason: error.to_string(),
    }
}

fn invalid_output(reason: impl ToString) -> ScorerError {
    ScorerError::InvalidOutput {
        scorer: MODEL_NAME.to_owned(),
        reason: reason.to_string(),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn binary_softmax_is_stable_and_bounded() {
        assert!((binary_softmax_keep(0.0, 0.0) - 0.5).abs() < 0.0001);
        assert!(binary_softmax_keep(-1000.0, 1000.0) > 0.9999);
        assert_eq!(binary_softmax_keep(f32::NAN, 1.0), 1.0);
    }

    #[test]
    fn word_scores_average_subwords_and_retain_unmatched_words() {
        let scores = aggregate_word_scores(
            3,
            &[None, Some(0), Some(0), Some(1), None],
            &[0.0, 0.2, 0.6, 0.9, 0.0],
        );
        assert_eq!(scores, vec![0.4, 0.9, 1.0]);
    }

    #[test]
    fn logits_shape_is_validated() {
        assert!(logits_to_retention_scores(&[1, 2, 2], &[0.0, 1.0, 1.0, 0.0], 2).is_ok());
        assert!(logits_to_retention_scores(&[1, 2], &[0.0; 4], 2).is_err());
        assert!(logits_to_retention_scores(&[1, 3, 2], &[0.0; 6], 2).is_err());
    }

    #[test]
    fn loader_reports_missing_model_without_initializing_runtime() {
        let directory = tempfile::tempdir().unwrap();
        let model = directory.path().join("missing.onnx");
        assert!(matches!(
            OnnxRedundancyScorerLoader::new().load(&model),
            Err(ScorerError::ModelAssetUnavailable { path }) if path == model
        ));
    }
}
