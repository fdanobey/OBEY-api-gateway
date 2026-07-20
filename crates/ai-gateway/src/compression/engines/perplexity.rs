//! Perplexity-style coarse-to-fine compression.
//!
//! This module intentionally does not bundle an ONNX runtime or model asset. Real
//! model-backed scoring requires an external asset plus an implementation of
//! [`RedundancyScorerLoader`] supplied by the embedding application. The loader is
//! cached with [`OnceLock`], so a model is loaded at most once and the compression
//! pipeline does not need to change when an ONNX feature is added later.
//!
//! [`HeuristicFallbackScorer`] is a deterministic, lightweight fallback. It is
//! explicitly identified as a heuristic and makes no ONNX, model-accuracy, or
//! 0.85 calibration-set preservation claim. Callers that require a model should
//! use [`PerplexityEngine::require_external_model`]; until a runtime/asset-backed
//! loader is supplied, that engine reports unavailability and safely passes the
//! original payload through unchanged.

use super::{
    CompressibleMessage, CompressiblePayload, CompressionContext, CompressionEngine, EngineResult,
};
use crate::compression::config::PerplexityConfig;
use async_trait::async_trait;
use serde_json::Value;
use std::{
    cmp::Ordering,
    collections::HashSet,
    error::Error,
    fmt,
    path::{Path, PathBuf},
    sync::{Arc, Mutex, OnceLock},
    time::Instant,
};

const DEFAULT_REDUNDANCY_THRESHOLD: f32 = 0.5;
const DEFAULT_COMPRESSION_RATIO_TARGET: u8 = 5;
const MAX_COMPRESSION_RATIO_TARGET: u8 = 20;

/// Whether scores come from an external model or the named lightweight fallback.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ScorerKind {
    ExternalModel,
    HeuristicFallback,
}

/// Runtime identity exposed by the engine availability API.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ScorerMetadata {
    pub name: String,
    pub kind: ScorerKind,
}

/// Explicit scorer/model failures. Any such failure causes a safe pass-through.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum ScorerError {
    ModelAssetUnavailable { path: PathBuf },
    RuntimeUnavailable { path: PathBuf, reason: String },
    ScoringFailed { scorer: String, reason: String },
    InvalidOutput { scorer: String, reason: String },
}

impl fmt::Display for ScorerError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::ModelAssetUnavailable { path } => {
                write!(
                    formatter,
                    "scoring model asset is unavailable: {}",
                    path.display()
                )
            }
            Self::RuntimeUnavailable { path, reason } => write!(
                formatter,
                "scoring runtime is unavailable for {}: {reason}",
                path.display()
            ),
            Self::ScoringFailed { scorer, reason } => {
                write!(formatter, "scorer `{scorer}` failed: {reason}")
            }
            Self::InvalidOutput { scorer, reason } => {
                write!(
                    formatter,
                    "scorer `{scorer}` returned invalid output: {reason}"
                )
            }
        }
    }
}

impl Error for ScorerError {}

/// Scoring granularity used by the coarse-to-fine pipeline.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ScoringGranularity {
    Message,
    Sentence,
}

/// Context supplied for message, sentence, and token scoring.
#[derive(Debug, Clone, Copy)]
pub struct ScoreRequest<'a> {
    pub text: &'a str,
    pub query: Option<&'a str>,
    pub role: &'a str,
    pub granularity: ScoringGranularity,
}

/// A UTF-8-safe token candidate within a low-information prose region.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct TokenCandidate<'a> {
    pub text: &'a str,
    pub start: usize,
    pub end: usize,
    pub query_overlap: bool,
}

/// Extensible scoring boundary for heuristic or future external model scorers.
///
/// Scores are retention/importance values in `0.0..=1.0`; values below the
/// configured redundancy threshold are removable candidates. Implementations
/// must return one token score for each input token.
pub trait RedundancyScorer: Send + Sync {
    fn metadata(&self) -> Result<ScorerMetadata, ScorerError>;

    fn score_coarse(&self, request: &ScoreRequest<'_>) -> Result<f32, ScorerError>;

    fn score_tokens(
        &self,
        request: &ScoreRequest<'_>,
        tokens: &[TokenCandidate<'_>],
    ) -> Result<Vec<f32>, ScorerError>;
}

/// Loads an external scorer implementation from a configured model asset.
pub trait RedundancyScorerLoader: Send + Sync {
    fn load(&self, model_path: &Path) -> Result<Arc<dyn RedundancyScorer>, ScorerError>;
}

/// Lazily loads and process-caches an external scorer exactly once.
pub struct CachedModelScorer {
    model_path: PathBuf,
    loader: Arc<dyn RedundancyScorerLoader>,
    scorer: OnceLock<Result<Arc<dyn RedundancyScorer>, ScorerError>>,
}

impl CachedModelScorer {
    pub fn new(model_path: impl Into<PathBuf>, loader: Arc<dyn RedundancyScorerLoader>) -> Self {
        Self {
            model_path: model_path.into(),
            loader,
            scorer: OnceLock::new(),
        }
    }

    pub fn model_path(&self) -> &Path {
        &self.model_path
    }

    pub fn availability(&self) -> Result<ScorerMetadata, ScorerError> {
        self.loaded()?.metadata()
    }

    fn loaded(&self) -> Result<Arc<dyn RedundancyScorer>, ScorerError> {
        self.scorer
            .get_or_init(|| self.loader.load(&self.model_path))
            .clone()
    }
}

impl fmt::Debug for CachedModelScorer {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter
            .debug_struct("CachedModelScorer")
            .field("model_path", &self.model_path)
            .field("loaded", &self.scorer.get().is_some())
            .finish_non_exhaustive()
    }
}

impl RedundancyScorer for CachedModelScorer {
    fn metadata(&self) -> Result<ScorerMetadata, ScorerError> {
        self.availability()
    }

    fn score_coarse(&self, request: &ScoreRequest<'_>) -> Result<f32, ScorerError> {
        self.loaded()?.score_coarse(request)
    }

    fn score_tokens(
        &self,
        request: &ScoreRequest<'_>,
        tokens: &[TokenCandidate<'_>],
    ) -> Result<Vec<f32>, ScorerError> {
        self.loaded()?.score_tokens(request, tokens)
    }
}

#[derive(Debug)]
struct ExternalRuntimeRequiredLoader;

impl RedundancyScorerLoader for ExternalRuntimeRequiredLoader {
    fn load(&self, model_path: &Path) -> Result<Arc<dyn RedundancyScorer>, ScorerError> {
        if !model_path.is_file() {
            return Err(ScorerError::ModelAssetUnavailable {
                path: model_path.to_owned(),
            });
        }

        Err(ScorerError::RuntimeUnavailable {
            path: model_path.to_owned(),
            reason: "no external ONNX/runtime scorer implementation was supplied".to_owned(),
        })
    }
}

/// Validated perplexity-engine settings.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct PerplexityEngineConfig {
    redundancy_threshold: f32,
    compression_ratio_target: u8,
}

impl PerplexityEngineConfig {
    pub fn new(
        redundancy_threshold: f32,
        compression_ratio_target: u8,
    ) -> Result<Self, PerplexityConfigurationError> {
        if !redundancy_threshold.is_finite() || !(0.0..=1.0).contains(&redundancy_threshold) {
            return Err(PerplexityConfigurationError::RedundancyThreshold(
                redundancy_threshold,
            ));
        }
        if !(1..=MAX_COMPRESSION_RATIO_TARGET).contains(&compression_ratio_target) {
            return Err(PerplexityConfigurationError::CompressionRatioTarget(
                compression_ratio_target,
            ));
        }

        Ok(Self {
            redundancy_threshold,
            compression_ratio_target,
        })
    }

    pub fn redundancy_threshold(self) -> f32 {
        self.redundancy_threshold
    }

    pub fn compression_ratio_target(self) -> u8 {
        self.compression_ratio_target
    }
}

impl Default for PerplexityEngineConfig {
    fn default() -> Self {
        Self {
            redundancy_threshold: DEFAULT_REDUNDANCY_THRESHOLD,
            compression_ratio_target: DEFAULT_COMPRESSION_RATIO_TARGET,
        }
    }
}

impl TryFrom<&PerplexityConfig> for PerplexityEngineConfig {
    type Error = PerplexityConfigurationError;

    fn try_from(config: &PerplexityConfig) -> Result<Self, Self::Error> {
        Self::new(config.redundancy_threshold, config.compression_ratio_target)
    }
}

/// Configuration errors kept separate from scorer availability failures.
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum PerplexityConfigurationError {
    RedundancyThreshold(f32),
    CompressionRatioTarget(u8),
}

impl fmt::Display for PerplexityConfigurationError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::RedundancyThreshold(value) => write!(
                formatter,
                "redundancy threshold must be finite and between 0.0 and 1.0; got {value}"
            ),
            Self::CompressionRatioTarget(value) => write!(
                formatter,
                "compression ratio target must be between 1 and {MAX_COMPRESSION_RATIO_TARGET}; got {value}"
            ),
        }
    }
}

impl Error for PerplexityConfigurationError {}

/// Deterministic lightweight scorer, explicitly not an ONNX/model substitute.
#[derive(Debug, Default, Clone, Copy)]
pub struct HeuristicFallbackScorer;

impl RedundancyScorer for HeuristicFallbackScorer {
    fn metadata(&self) -> Result<ScorerMetadata, ScorerError> {
        Ok(ScorerMetadata {
            name: "heuristic-fallback".to_owned(),
            kind: ScorerKind::HeuristicFallback,
        })
    }

    fn score_coarse(&self, request: &ScoreRequest<'_>) -> Result<f32, ScorerError> {
        let terms = normalized_terms(request.text);
        if terms.is_empty() {
            return Ok(0.0);
        }

        let unique = terms.iter().collect::<HashSet<_>>().len() as f32 / terms.len() as f32;
        let meaningful = terms
            .iter()
            .filter(|term| !is_low_information_term(term))
            .count() as f32
            / terms.len() as f32;
        let query_terms = query_terms(request.query);
        let overlap = if query_terms.is_empty() {
            0.0
        } else {
            terms
                .iter()
                .filter(|term| query_terms.contains(term.as_str()))
                .count() as f32
                / terms.len() as f32
        };

        Ok((0.05 + unique * 0.35 + meaningful * 0.4 + overlap * 0.4).clamp(0.0, 1.0))
    }

    fn score_tokens(
        &self,
        _request: &ScoreRequest<'_>,
        tokens: &[TokenCandidate<'_>],
    ) -> Result<Vec<f32>, ScorerError> {
        Ok(tokens
            .iter()
            .map(|token| {
                let normalized = normalize_term(token.text);
                if token.query_overlap {
                    1.0
                } else if normalized.is_empty() {
                    0.2
                } else if is_low_information_term(&normalized) {
                    0.1
                } else if token.text.chars().any(|character| character.is_numeric())
                    || token.text.contains('_')
                    || token
                        .text
                        .chars()
                        .skip(1)
                        .any(|character| character.is_uppercase())
                {
                    0.95
                } else if normalized.len() >= 7 {
                    0.75
                } else {
                    0.45
                }
            })
            .collect())
    }
}

/// Coarse-to-fine compression engine backed by an injected scorer.
pub struct PerplexityEngine {
    config: PerplexityEngineConfig,
    scorer: Arc<dyn RedundancyScorer>,
    last_error: Mutex<Option<ScorerError>>,
}

impl PerplexityEngine {
    /// Constructs an engine around an already shared/cached scorer implementation.
    pub fn with_scorer(config: PerplexityEngineConfig, scorer: Arc<dyn RedundancyScorer>) -> Self {
        Self {
            config,
            scorer,
            last_error: Mutex::new(None),
        }
    }

    /// Constructs the explicitly named deterministic heuristic fallback.
    pub fn heuristic_fallback(config: PerplexityEngineConfig) -> Self {
        Self::with_scorer(config, Arc::new(HeuristicFallbackScorer))
    }

    /// Constructs a model-required engine with a caller-supplied, load-once loader.
    pub fn with_model_loader(
        config: PerplexityEngineConfig,
        model_path: impl Into<PathBuf>,
        loader: Arc<dyn RedundancyScorerLoader>,
    ) -> Self {
        Self::with_scorer(config, Arc::new(CachedModelScorer::new(model_path, loader)))
    }

    /// Constructs an unavailable-until-integrated model engine.
    ///
    /// This checks the configured asset path and never silently substitutes the
    /// heuristic scorer or represents heuristic output as model inference.
    pub fn require_external_model(
        config: PerplexityEngineConfig,
        model_path: impl Into<PathBuf>,
    ) -> Self {
        Self::with_model_loader(config, model_path, Arc::new(ExternalRuntimeRequiredLoader))
    }

    pub fn config(&self) -> PerplexityEngineConfig {
        self.config
    }

    /// Forces lazy initialization and reports scorer/model availability.
    pub fn availability(&self) -> Result<ScorerMetadata, ScorerError> {
        self.scorer.metadata()
    }

    /// Returns the most recent availability/scoring error, if any.
    pub fn last_error(&self) -> Option<ScorerError> {
        self.last_error
            .lock()
            .unwrap_or_else(|poisoned| poisoned.into_inner())
            .clone()
    }

    fn set_last_error(&self, error: Option<ScorerError>) {
        *self
            .last_error
            .lock()
            .unwrap_or_else(|poisoned| poisoned.into_inner()) = error;
    }
}

impl fmt::Debug for PerplexityEngine {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter
            .debug_struct("PerplexityEngine")
            .field("config", &self.config)
            .field("availability", &self.availability())
            .finish_non_exhaustive()
    }
}

#[async_trait]
impl CompressionEngine for PerplexityEngine {
    fn name(&self) -> &str {
        "perplexity"
    }

    async fn compress(
        &self,
        payload: &mut CompressiblePayload,
        context: &CompressionContext,
    ) -> EngineResult {
        let started = Instant::now();
        let original = payload.clone();
        let tokens_before = count_payload_tokens(&original, context);
        self.set_last_error(None);

        if let Err(error) = self.availability() {
            self.set_last_error(Some(error));
            return unchanged_result(self.name(), tokens_before, started);
        }

        let query = latest_user_query(&original);
        let query_terms = query_terms(query.as_deref());
        let mut changed = false;
        let compression = compress_payload(
            payload,
            context,
            self.scorer.as_ref(),
            self.config,
            query.as_deref(),
            &query_terms,
            &mut changed,
        );

        if let Err(error) = compression {
            *payload = original;
            self.set_last_error(Some(error));
            return unchanged_result(self.name(), tokens_before, started);
        }

        if changed {
            payload.refresh_metadata();
        }
        let candidate_tokens = count_payload_tokens(payload, context);
        if !changed || candidate_tokens >= tokens_before {
            *payload = original;
            return unchanged_result(self.name(), tokens_before, started);
        }

        refresh_message_token_counts(payload, context);
        EngineResult {
            engine_name: self.name().to_owned(),
            tokens_before,
            tokens_after: candidate_tokens,
            duration_ms: elapsed_millis(started),
            applied: true,
        }
    }
}

fn compress_payload(
    payload: &mut CompressiblePayload,
    context: &CompressionContext,
    scorer: &dyn RedundancyScorer,
    config: PerplexityEngineConfig,
    query: Option<&str>,
    query_terms: &HashSet<String>,
    changed: &mut bool,
) -> Result<(), ScorerError> {
    for message in &mut payload.messages {
        if message_is_protected(message) {
            continue;
        }

        let message_text = visible_text(message.content.as_value());
        if message_text.trim().is_empty() {
            continue;
        }
        let request = ScoreRequest {
            text: &message_text,
            query,
            role: &message.role,
            granularity: ScoringGranularity::Message,
        };
        let message_score = validate_coarse_score(scorer, scorer.score_coarse(&request)?)?;
        if message_score >= config.redundancy_threshold {
            continue;
        }

        let role = message.role.clone();
        let mut scoring_error = None;
        message.content.for_each_text_leaf_mut(|text| {
            if scoring_error.is_some() {
                return;
            }
            let transformed = context
                .protection_scanner
                .transform_unprotected(text, |segment| {
                    match compress_prose_segment(segment, &role, query, query_terms, scorer, config)
                    {
                        Ok(output) => output,
                        Err(error) => {
                            scoring_error = Some(error);
                            segment.to_owned()
                        }
                    }
                });
            if transformed != *text {
                *text = transformed;
                *changed = true;
            }
        });
        if let Some(error) = scoring_error {
            return Err(error);
        }
    }

    Ok(())
}

fn compress_prose_segment(
    segment: &str,
    role: &str,
    query: Option<&str>,
    query_terms: &HashSet<String>,
    scorer: &dyn RedundancyScorer,
    config: PerplexityEngineConfig,
) -> Result<String, ScorerError> {
    let mut output = String::with_capacity(segment.len());
    let mut cursor = 0;

    for sentence in sentence_ranges(segment) {
        output.push_str(&segment[cursor..sentence.start]);
        let sentence_text = &segment[sentence.clone()];
        let request = ScoreRequest {
            text: sentence_text,
            query,
            role,
            granularity: ScoringGranularity::Sentence,
        };
        let coarse_score = validate_coarse_score(scorer, scorer.score_coarse(&request)?)?;
        if coarse_score < config.redundancy_threshold {
            output.push_str(&compress_low_information_sentence(
                sentence_text,
                role,
                query,
                query_terms,
                scorer,
                config,
            )?);
        } else {
            output.push_str(sentence_text);
        }
        cursor = sentence.end;
    }
    output.push_str(&segment[cursor..]);

    Ok(output)
}

fn compress_low_information_sentence(
    sentence: &str,
    role: &str,
    query: Option<&str>,
    query_terms: &HashSet<String>,
    scorer: &dyn RedundancyScorer,
    config: PerplexityEngineConfig,
) -> Result<String, ScorerError> {
    let tokens = token_candidates(sentence, query_terms);
    if tokens.len() < 2 || config.compression_ratio_target <= 1 {
        return Ok(sentence.to_owned());
    }

    let request = ScoreRequest {
        text: sentence,
        query,
        role,
        granularity: ScoringGranularity::Sentence,
    };
    let scores = scorer.score_tokens(&request, &tokens)?;
    if scores.len() != tokens.len() {
        return Err(invalid_output(
            scorer,
            format!(
                "expected {} token scores, received {}",
                tokens.len(),
                scores.len()
            ),
        ));
    }
    if scores.iter().any(|score| !score.is_finite()) {
        return Err(invalid_output(scorer, "token score was not finite"));
    }

    let minimum_to_keep = tokens
        .len()
        .div_ceil(config.compression_ratio_target as usize);
    let maximum_to_remove = tokens.len().saturating_sub(minimum_to_keep);
    let mut removable: Vec<_> = scores
        .iter()
        .copied()
        .enumerate()
        .filter(|(index, score)| {
            !tokens[*index].query_overlap && *score < config.redundancy_threshold
        })
        .collect();
    removable.sort_by(|(left_index, left_score), (right_index, right_score)| {
        left_score
            .partial_cmp(right_score)
            .unwrap_or(Ordering::Equal)
            .then_with(|| left_index.cmp(right_index))
    });
    removable.truncate(maximum_to_remove);
    if removable.is_empty() {
        return Ok(sentence.to_owned());
    }

    let removed: HashSet<_> = removable.into_iter().map(|(index, _)| index).collect();
    let mut output = String::with_capacity(sentence.len());
    let mut cursor = 0;
    for (index, token) in tokens.iter().enumerate() {
        if removed.contains(&index) {
            output.push_str(&sentence[cursor..token.start]);
            cursor = token.end;
        }
    }
    output.push_str(&sentence[cursor..]);

    Ok(cleanup_removed_tokens(&output))
}

fn validate_coarse_score(scorer: &dyn RedundancyScorer, score: f32) -> Result<f32, ScorerError> {
    if score.is_finite() {
        Ok(score.clamp(0.0, 1.0))
    } else {
        Err(invalid_output(scorer, "coarse score was not finite"))
    }
}

fn invalid_output(scorer: &dyn RedundancyScorer, reason: impl Into<String>) -> ScorerError {
    let name = scorer
        .metadata()
        .map(|metadata| metadata.name)
        .unwrap_or_else(|_| "unknown".to_owned());
    ScorerError::InvalidOutput {
        scorer: name,
        reason: reason.into(),
    }
}

fn message_is_protected(message: &CompressibleMessage) -> bool {
    message.cache_protected
        || message.critical
        || matches!(message.role.as_str(), "system" | "tool" | "function")
        || !message.relationships.tool_call_ids.is_empty()
        || !message.relationships.tool_result_for_ids.is_empty()
        || !message.relationships.related_message_indices.is_empty()
        || !message.relationships.unresolved_tool_call_ids.is_empty()
}

fn latest_user_query(payload: &CompressiblePayload) -> Option<String> {
    payload
        .messages
        .iter()
        .rev()
        .find(|message| message.role == "user")
        .map(|message| visible_text(message.content.as_value()))
        .filter(|query| !query.trim().is_empty())
}

fn visible_text(value: &Value) -> String {
    match value {
        Value::String(text) => text.clone(),
        Value::Array(parts) => parts
            .iter()
            .filter_map(visible_content_block_text)
            .collect::<Vec<_>>()
            .join("\n"),
        Value::Object(_) => visible_content_block_text(value).unwrap_or_default(),
        _ => String::new(),
    }
}

fn visible_content_block_text(value: &Value) -> Option<String> {
    let object = value.as_object()?;
    match object.get("type").and_then(Value::as_str) {
        Some("text" | "input_text" | "output_text") => object
            .get("text")
            .and_then(Value::as_str)
            .map(str::to_owned),
        Some("tool_result") => object.get("content").map(visible_text),
        _ => None,
    }
}

fn token_candidates<'a>(text: &'a str, query_terms: &HashSet<String>) -> Vec<TokenCandidate<'a>> {
    let mut tokens = Vec::new();
    let mut start = None;
    let mut word_kind = false;

    for (index, character) in text.char_indices() {
        if character.is_whitespace() {
            if let Some(token_start) = start.take() {
                push_token(text, token_start, index, query_terms, &mut tokens);
            }
            continue;
        }

        let current_word_kind =
            character.is_alphanumeric() || matches!(character, '_' | '-' | '\'' | '’');
        if let Some(token_start) = start {
            if current_word_kind != word_kind {
                push_token(text, token_start, index, query_terms, &mut tokens);
                start = Some(index);
            }
        } else {
            start = Some(index);
        }
        word_kind = current_word_kind;
    }
    if let Some(token_start) = start {
        push_token(text, token_start, text.len(), query_terms, &mut tokens);
    }

    tokens
}

fn push_token<'a>(
    text: &'a str,
    start: usize,
    end: usize,
    query_terms: &HashSet<String>,
    tokens: &mut Vec<TokenCandidate<'a>>,
) {
    let token = &text[start..end];
    let normalized = normalize_term(token);
    tokens.push(TokenCandidate {
        text: token,
        start,
        end,
        query_overlap: !normalized.is_empty() && query_terms.contains(&normalized),
    });
}

fn sentence_ranges(text: &str) -> Vec<std::ops::Range<usize>> {
    let mut ranges = Vec::new();
    let mut start = 0;
    let mut iterator = text.char_indices().peekable();

    while let Some((index, character)) = iterator.next() {
        let boundary = matches!(character, '.' | '!' | '?' | '\n');
        if !boundary {
            continue;
        }
        let mut end = index + character.len_utf8();
        while let Some((next_index, next_character)) = iterator.peek().copied() {
            if !next_character.is_whitespace() || next_character == '\n' {
                break;
            }
            iterator.next();
            end = next_index + next_character.len_utf8();
        }
        if start < end {
            ranges.push(start..end);
        }
        start = end;
    }
    if start < text.len() {
        ranges.push(start..text.len());
    }
    if ranges.is_empty() && !text.is_empty() {
        ranges.push(0..text.len());
    }

    ranges
}

fn cleanup_removed_tokens(text: &str) -> String {
    let mut output = String::with_capacity(text.len());
    let mut pending_horizontal_space = false;

    for character in text.chars() {
        if matches!(character, ' ' | '\t') {
            pending_horizontal_space = true;
            continue;
        }
        if character == '\n' || character == '\r' {
            while output.ends_with(' ') {
                output.pop();
            }
            output.push(character);
            pending_horizontal_space = false;
            continue;
        }
        if pending_horizontal_space
            && !output.is_empty()
            && !output.ends_with(['\n', '\r', ' '])
            && !matches!(
                character,
                ',' | '.' | ';' | ':' | '!' | '?' | ')' | ']' | '}'
            )
        {
            output.push(' ');
        }
        output.push(character);
        pending_horizontal_space = false;
    }

    output
}

fn normalized_terms(text: &str) -> Vec<String> {
    text.split_whitespace()
        .map(normalize_term)
        .filter(|term| !term.is_empty())
        .collect()
}

fn query_terms(query: Option<&str>) -> HashSet<String> {
    query
        .map(normalized_terms)
        .unwrap_or_default()
        .into_iter()
        .collect()
}

fn normalize_term(text: &str) -> String {
    text.chars()
        .filter(|character| character.is_alphanumeric() || *character == '_')
        .flat_map(char::to_lowercase)
        .collect()
}

fn is_low_information_term(term: &str) -> bool {
    matches!(
        term,
        "a" | "an"
            | "the"
            | "and"
            | "or"
            | "but"
            | "so"
            | "just"
            | "really"
            | "very"
            | "basically"
            | "actually"
            | "simply"
            | "perhaps"
            | "maybe"
            | "please"
            | "kind"
            | "sort"
            | "of"
            | "to"
            | "in"
            | "on"
            | "at"
            | "for"
            | "with"
            | "is"
            | "are"
            | "was"
            | "were"
            | "be"
            | "been"
    )
}

fn count_payload_tokens(payload: &CompressiblePayload, context: &CompressionContext) -> u32 {
    context
        .token_counter
        .count_request(&payload.clone().into_openai_request())
}

fn refresh_message_token_counts(payload: &mut CompressiblePayload, context: &CompressionContext) {
    let model = if payload.model.is_empty() {
        context.model.as_str()
    } else {
        payload.model.as_str()
    };
    for message in &mut payload.messages {
        let content_tokens = match message.content.as_value() {
            Value::Null => 0,
            Value::String(text) => context.token_counter.count_text(model, text),
            structured => context
                .token_counter
                .count_text(model, &structured.to_string()),
        };
        let extra_tokens = if message.extra.is_empty() {
            0
        } else {
            context
                .token_counter
                .count_text(model, &Value::Object(message.extra.clone()).to_string())
        };
        message.token_count = 4u32
            .saturating_add(context.token_counter.count_text(model, &message.role))
            .saturating_add(content_tokens)
            .saturating_add(extra_tokens);
    }
}

fn unchanged_result(engine_name: &str, tokens: u32, started: Instant) -> EngineResult {
    EngineResult {
        engine_name: engine_name.to_owned(),
        tokens_before: tokens,
        tokens_after: tokens,
        duration_ms: elapsed_millis(started),
        applied: false,
    }
}

fn elapsed_millis(started: Instant) -> u64 {
    started.elapsed().as_millis().min(u64::MAX as u128) as u64
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::models::openai::OpenAIRequest;
    use serde_json::{json, Value};
    use std::sync::atomic::{AtomicUsize, Ordering as AtomicOrdering};
    use std::time::Duration;

    #[derive(Debug)]
    struct MockScorer {
        coarse_calls: Mutex<Vec<(ScoringGranularity, String)>>,
        token_calls: AtomicUsize,
        delay: Duration,
    }

    impl MockScorer {
        fn new() -> Self {
            Self {
                coarse_calls: Mutex::new(Vec::new()),
                token_calls: AtomicUsize::new(0),
                delay: Duration::ZERO,
            }
        }

        fn with_delay(delay: Duration) -> Self {
            Self {
                delay,
                ..Self::new()
            }
        }

        fn coarse_calls(&self) -> Vec<(ScoringGranularity, String)> {
            self.coarse_calls.lock().unwrap().clone()
        }
    }

    impl RedundancyScorer for MockScorer {
        fn metadata(&self) -> Result<ScorerMetadata, ScorerError> {
            Ok(ScorerMetadata {
                name: "deterministic-mock-model".to_owned(),
                kind: ScorerKind::ExternalModel,
            })
        }

        fn score_coarse(&self, request: &ScoreRequest<'_>) -> Result<f32, ScorerError> {
            if !self.delay.is_zero() {
                std::thread::sleep(self.delay);
            }
            self.coarse_calls
                .lock()
                .unwrap()
                .push((request.granularity, request.text.to_owned()));
            Ok(
                if request.granularity == ScoringGranularity::Sentence
                    && request.text.contains("important sentence")
                {
                    0.9
                } else {
                    0.1
                },
            )
        }

        fn score_tokens(
            &self,
            _request: &ScoreRequest<'_>,
            tokens: &[TokenCandidate<'_>],
        ) -> Result<Vec<f32>, ScorerError> {
            self.token_calls.fetch_add(1, AtomicOrdering::SeqCst);
            Ok(tokens
                .iter()
                .map(|token| {
                    if normalize_term(token.text) == "essential" {
                        0.9
                    } else {
                        0.1
                    }
                })
                .collect())
        }
    }

    struct CountingLoader {
        loads: AtomicUsize,
        scorer: Arc<dyn RedundancyScorer>,
    }

    impl RedundancyScorerLoader for CountingLoader {
        fn load(&self, _model_path: &Path) -> Result<Arc<dyn RedundancyScorer>, ScorerError> {
            self.loads.fetch_add(1, AtomicOrdering::SeqCst);
            Ok(self.scorer.clone())
        }
    }

    fn payload(value: Value) -> CompressiblePayload {
        let request: OpenAIRequest = serde_json::from_value(value).unwrap();
        CompressiblePayload::from(request)
    }

    fn context() -> CompressionContext {
        CompressionContext::new("gpt-4o", "test")
    }

    fn engine_with(scorer: Arc<dyn RedundancyScorer>) -> PerplexityEngine {
        PerplexityEngine::with_scorer(PerplexityEngineConfig::default(), scorer)
    }

    #[tokio::test]
    async fn mock_scorer_retains_essential_and_removes_low_scoring_tokens() {
        let scorer = Arc::new(MockScorer::new());
        let engine = engine_with(scorer);
        let mut input = payload(json!({
            "model": "gpt-4o",
            "messages": [
                {"role": "assistant", "content": "removable essential removable removable removable"},
                {"role": "user", "content": "answer concisely"}
            ]
        }));

        let result = engine.compress(&mut input, &context()).await;
        let output = input.messages[0].content.as_text().unwrap();

        assert!(result.applied);
        assert!(output.contains("essential"));
        assert!(!output.contains("removable"));
    }

    #[tokio::test]
    async fn coarse_to_fine_scores_messages_then_sentences_and_only_low_regions_tokens() {
        let scorer = Arc::new(MockScorer::new());
        let engine = engine_with(scorer.clone());
        let mut input = payload(json!({
            "model": "gpt-4o",
            "messages": [
                {"role": "assistant", "content": "important sentence remains intact"},
                {"role": "assistant", "content": "important sentence stays. removable removable essential removable."},
                {"role": "user", "content": "current query"}
            ]
        }));

        engine.compress(&mut input, &context()).await;
        let calls = scorer.coarse_calls();
        let granularities: Vec<_> = calls.iter().map(|(granularity, _)| *granularity).collect();

        assert_eq!(
            granularities,
            [
                ScoringGranularity::Message,
                ScoringGranularity::Sentence,
                ScoringGranularity::Message,
                ScoringGranularity::Sentence,
                ScoringGranularity::Sentence,
            ]
        );
        assert!(calls.iter().any(|(granularity, text)| *granularity
            == ScoringGranularity::Sentence
            && text.contains("important sentence")));
        assert_eq!(scorer.token_calls.load(AtomicOrdering::SeqCst), 1);
        assert_eq!(
            input.messages[0].content.as_text(),
            Some("important sentence remains intact")
        );
        assert!(input.messages[1]
            .content
            .as_text()
            .unwrap()
            .contains("important sentence stays."));
    }

    #[tokio::test]
    async fn query_overlap_is_retained_even_when_mock_scores_every_token_low() {
        let engine = engine_with(Arc::new(MockScorer::new()));
        let mut input = payload(json!({
            "model": "gpt-4o",
            "messages": [
                {"role": "assistant", "content": "remove remove zebra remove remove remove"},
                {"role": "user", "content": "Explain zebra behavior"}
            ]
        }));

        let result = engine.compress(&mut input, &context()).await;

        assert!(result.applied);
        assert!(input.messages[0]
            .content
            .as_text()
            .unwrap()
            .contains("zebra"));
        assert_eq!(
            input.messages[1].content.as_text(),
            Some("Explain zebra behavior")
        );
    }

    #[tokio::test]
    async fn ratio_configuration_is_bounded_and_removal_never_exceeds_target() {
        assert!(matches!(
            PerplexityEngineConfig::new(0.5, 0),
            Err(PerplexityConfigurationError::CompressionRatioTarget(0))
        ));
        assert!(matches!(
            PerplexityEngineConfig::new(0.5, 21),
            Err(PerplexityConfigurationError::CompressionRatioTarget(21))
        ));
        let config = PerplexityEngineConfig::new(0.5, 20).unwrap();
        let engine = PerplexityEngine::with_scorer(config, Arc::new(MockScorer::new()));
        let sentence = (0..40)
            .map(|index| format!("word{index}"))
            .collect::<Vec<_>>()
            .join(" ");
        let mut input = payload(json!({
            "model": "gpt-4o",
            "messages": [
                {"role": "assistant", "content": sentence},
                {"role": "user", "content": "query"}
            ]
        }));

        engine.compress(&mut input, &context()).await;
        let remaining = token_candidates(
            input.messages[0].content.as_text().unwrap(),
            &HashSet::new(),
        );

        assert!(
            remaining.len() >= 2,
            "20x target must retain at least ceil(40/20)"
        );
    }

    #[tokio::test]
    async fn protected_regions_and_structural_messages_remain_byte_identical() {
        let protected_parts = [
            "```rust\nlet  value = essential;\n```\n",
            "https://example.test/a?q=1",
            r"C:\Users\a\main.rs",
            "/usr/local/bin/tool",
            r#"{"exact": true}"#,
            "camelCase",
            "$x + y$",
        ];
        let protected = format!(
            "{}{} {} {} {} {} {}",
            protected_parts[0],
            protected_parts[1],
            protected_parts[2],
            protected_parts[3],
            protected_parts[4],
            protected_parts[5],
            protected_parts[6]
        );
        let system = "system exact";
        let cached = "cached exact";
        let latest_user = "latest exact request";
        let tool = "tool exact output";
        let engine = engine_with(Arc::new(MockScorer::new()));
        let mut input = payload(json!({
            "model": "gpt-4o",
            "messages": [
                {"role": "system", "content": system},
                {"role": "assistant", "content": cached, "cache_control": {"type": "ephemeral"}},
                {"role": "assistant", "content": format!("remove remove\n{protected}\nremove remove")},
                {"role": "tool", "content": tool, "tool_call_id": "call-1"},
                {"role": "user", "content": latest_user}
            ]
        }));

        engine.compress(&mut input, &context()).await;
        let compressed = input.messages[2].content.as_text().unwrap();

        assert_eq!(input.messages[0].content.as_text(), Some(system));
        assert_eq!(input.messages[1].content.as_text(), Some(cached));
        assert_eq!(input.messages[3].content.as_text(), Some(tool));
        assert_eq!(input.messages[4].content.as_text(), Some(latest_user));
        for part in protected_parts {
            assert!(compressed.contains(part), "protected bytes changed: {part}");
        }
    }

    #[tokio::test]
    async fn unavailable_required_model_reports_error_and_safely_passes_through() {
        let missing = std::env::temp_dir().join(format!(
            "missing-perplexity-model-{}.onnx",
            std::process::id()
        ));
        let engine = PerplexityEngine::require_external_model(
            PerplexityEngineConfig::default(),
            missing.clone(),
        );
        let mut input = payload(json!({
            "model": "gpt-4o",
            "messages": [
                {"role": "assistant", "content": "remove remove remove remove"},
                {"role": "user", "content": "query"}
            ]
        }));
        let original = input.clone();

        assert_eq!(
            engine.availability(),
            Err(ScorerError::ModelAssetUnavailable { path: missing })
        );
        let result = engine.compress(&mut input, &context()).await;

        assert!(!result.applied);
        assert_eq!(result.tokens_before, result.tokens_after);
        assert_eq!(input, original);
        assert!(matches!(
            engine.last_error(),
            Some(ScorerError::ModelAssetUnavailable { .. })
        ));
    }

    #[tokio::test]
    async fn cached_model_loader_and_scorer_are_reused_across_requests() {
        let scorer: Arc<dyn RedundancyScorer> = Arc::new(MockScorer::new());
        let loader = Arc::new(CountingLoader {
            loads: AtomicUsize::new(0),
            scorer,
        });
        let engine = PerplexityEngine::with_model_loader(
            PerplexityEngineConfig::default(),
            "injected-model.asset",
            loader.clone(),
        );

        assert!(engine.availability().is_ok());
        for _ in 0..2 {
            let mut input = payload(json!({
                "model": "gpt-4o",
                "messages": [
                    {"role": "assistant", "content": "remove remove essential remove"},
                    {"role": "user", "content": "query"}
                ]
            }));
            engine.compress(&mut input, &context()).await;
        }

        assert_eq!(loader.loads.load(AtomicOrdering::SeqCst), 1);
    }

    #[tokio::test]
    async fn compression_never_increases_tokens_and_reports_measured_duration() {
        let scorer = Arc::new(MockScorer::with_delay(Duration::from_millis(2)));
        let engine = engine_with(scorer);
        let mut input = payload(json!({
            "model": "gpt-4o",
            "messages": [
                {"role": "assistant", "content": "remove remove essential remove remove remove"},
                {"role": "user", "content": "query"}
            ]
        }));

        let result = engine.compress(&mut input, &context()).await;

        assert!(result.tokens_after <= result.tokens_before);
        assert!(result.duration_ms >= 2);
    }

    #[tokio::test]
    async fn heuristic_fallback_is_explicitly_identified_as_non_model() {
        let engine = PerplexityEngine::heuristic_fallback(PerplexityEngineConfig::default());

        assert_eq!(
            engine.availability().unwrap(),
            ScorerMetadata {
                name: "heuristic-fallback".to_owned(),
                kind: ScorerKind::HeuristicFallback,
            }
        );
    }
}
