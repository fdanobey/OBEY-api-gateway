//! Language-aware prose compression with pluggable, safely loaded rule packs.

use super::{CompressiblePayload, CompressionContext, CompressionEngine, EngineResult};
use async_trait::async_trait;
use regex::Regex;
use serde::{Deserialize, Serialize};
use serde_json::Value;
use std::{
    collections::{BTreeMap, HashMap},
    fs::File,
    io::{self, Read},
    path::{Path, PathBuf},
    sync::{Arc, LazyLock, RwLock},
    time::Instant,
};

/// Maximum accepted on-disk language-pack size (256 KiB).
pub const MAX_LANGUAGE_PACK_BYTES: u64 = 256 * 1024;
const DETECTION_CONFIDENCE_THRESHOLD: f32 = 0.7;
const MAX_RULES_PER_PACK: usize = 2_048;
const MAX_RULE_TEXT_BYTES: usize = 256;

/// Language-pack compression strength.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum LanguagePackLevel {
    #[default]
    Light,
    Full,
    Maximum,
}

/// Transform groups enabled at one language-pack level.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub struct LanguagePackLevelRules {
    pub remove_fillers: bool,
    pub condense_verbose_phrases: bool,
    pub remove_hedges: bool,
    pub remove_articles: bool,
    pub apply_abbreviations: bool,
}

impl LanguagePackLevelRules {
    const LIGHT: Self = Self {
        remove_fillers: true,
        condense_verbose_phrases: false,
        remove_hedges: false,
        remove_articles: false,
        apply_abbreviations: false,
    };

    const FULL: Self = Self {
        remove_fillers: true,
        condense_verbose_phrases: true,
        remove_hedges: true,
        remove_articles: true,
        apply_abbreviations: false,
    };

    const MAXIMUM: Self = Self {
        remove_fillers: true,
        condense_verbose_phrases: true,
        remove_hedges: true,
        remove_articles: true,
        apply_abbreviations: true,
    };
}

/// Per-pack behavior for light, full, and maximum compression.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct LanguagePackLevels {
    pub light: LanguagePackLevelRules,
    pub full: LanguagePackLevelRules,
    pub maximum: LanguagePackLevelRules,
}

impl LanguagePackLevels {
    fn rules(&self, level: LanguagePackLevel) -> LanguagePackLevelRules {
        match level {
            LanguagePackLevel::Light => self.light,
            LanguagePackLevel::Full => self.full,
            LanguagePackLevel::Maximum => self.maximum,
        }
    }
}

impl Default for LanguagePackLevels {
    fn default() -> Self {
        Self {
            light: LanguagePackLevelRules::LIGHT,
            full: LanguagePackLevelRules::FULL,
            maximum: LanguagePackLevelRules::MAXIMUM,
        }
    }
}

/// Serializable language-specific compression rules.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct LanguagePack {
    pub code: String,
    pub name: String,
    #[serde(default)]
    pub filler_words: Vec<String>,
    #[serde(default)]
    pub verbose_phrases: BTreeMap<String, String>,
    #[serde(default)]
    pub hedges: Vec<String>,
    #[serde(default)]
    pub removable_articles: Vec<String>,
    /// Maximum-level replacements such as `and` -> `&` or `with` -> `w/`.
    #[serde(default)]
    pub abbreviations: BTreeMap<String, String>,
    #[serde(default)]
    pub levels: LanguagePackLevels,
}

impl LanguagePack {
    fn validate(&self, requested_code: Option<&str>) -> Result<(), LanguagePackError> {
        let normalized_code = normalize_language_code(&self.code)?;
        if normalized_code != self.code.to_ascii_lowercase() {
            return Err(LanguagePackError::InvalidPack(
                "pack code must be a normalized language code".to_owned(),
            ));
        }
        if requested_code.is_some_and(|requested| requested != normalized_code) {
            return Err(LanguagePackError::InvalidPack(format!(
                "pack code `{normalized_code}` does not match requested language `{}`",
                requested_code.unwrap_or_default()
            )));
        }
        if self.name.trim().is_empty() {
            return Err(LanguagePackError::InvalidPack(
                "pack name must not be empty".to_owned(),
            ));
        }

        let rule_count = self
            .filler_words
            .len()
            .saturating_add(self.verbose_phrases.len())
            .saturating_add(self.hedges.len())
            .saturating_add(self.removable_articles.len())
            .saturating_add(self.abbreviations.len());
        if rule_count > MAX_RULES_PER_PACK {
            return Err(LanguagePackError::InvalidPack(format!(
                "pack contains {rule_count} rules; maximum is {MAX_RULES_PER_PACK}"
            )));
        }

        for rule in self
            .filler_words
            .iter()
            .chain(self.hedges.iter())
            .chain(self.removable_articles.iter())
            .chain(self.verbose_phrases.keys())
            .chain(self.verbose_phrases.values())
            .chain(self.abbreviations.keys())
            .chain(self.abbreviations.values())
        {
            if rule.len() > MAX_RULE_TEXT_BYTES {
                return Err(LanguagePackError::InvalidPack(format!(
                    "rule exceeds {MAX_RULE_TEXT_BYTES} bytes"
                )));
            }
        }
        for rule in self
            .filler_words
            .iter()
            .chain(self.hedges.iter())
            .chain(self.removable_articles.iter())
            .chain(self.verbose_phrases.keys())
            .chain(self.abbreviations.keys())
        {
            if rule.trim().is_empty() {
                return Err(LanguagePackError::InvalidPack(
                    "match rules must not be empty".to_owned(),
                ));
            }
        }

        Ok(())
    }
}

static ENGLISH_PACK: LazyLock<LanguagePack> = LazyLock::new(|| LanguagePack {
    code: "en".to_owned(),
    name: "English".to_owned(),
    filler_words: [
        "actually",
        "basically",
        "clearly",
        "essentially",
        "honestly",
        "just",
        "literally",
        "maybe",
        "obviously",
        "perhaps",
        "please",
        "really",
        "simply",
        "very",
    ]
    .into_iter()
    .map(str::to_owned)
    .collect(),
    verbose_phrases: BTreeMap::from([
        ("at this point in time".to_owned(), "now".to_owned()),
        ("due to the fact that".to_owned(), "because".to_owned()),
        ("for the purpose of".to_owned(), "to".to_owned()),
        ("has the ability to".to_owned(), "can".to_owned()),
        ("in order to".to_owned(), "to".to_owned()),
        ("is able to".to_owned(), "can".to_owned()),
        ("make use of".to_owned(), "use".to_owned()),
        ("take a look at".to_owned(), "inspect".to_owned()),
        ("you need to".to_owned(), "".to_owned()),
        ("you should".to_owned(), "".to_owned()),
    ]),
    hedges: [
        "happy to help with that",
        "i hope this helps",
        "i think",
        "i believe",
        "let me know if you have any questions",
    ]
    .into_iter()
    .map(str::to_owned)
    .collect(),
    removable_articles: ["a", "an", "the"].into_iter().map(str::to_owned).collect(),
    abbreviations: BTreeMap::from([
        ("additional information".to_owned(), "details".to_owned()),
        ("and".to_owned(), "&".to_owned()),
        ("for example".to_owned(), "e.g.".to_owned()),
        ("versus".to_owned(), "vs".to_owned()),
        ("with".to_owned(), "w/".to_owned()),
        ("without".to_owned(), "w/o".to_owned()),
    ]),
    levels: LanguagePackLevels::default(),
});

/// Returns the process-wide built-in English pack.
pub fn english_language_pack() -> &'static LanguagePack {
    &ENGLISH_PACK
}

/// Safe language-pack loading failures.
#[derive(Debug, thiserror::Error)]
pub enum LanguagePackError {
    #[error("language-pack directory contains a NUL character")]
    NulPath,
    #[error("language-pack directory is not a directory: {0}")]
    InvalidDirectory(PathBuf),
    #[error("invalid language-pack name `{0}`")]
    InvalidLanguage(String),
    #[error("language-pack extension must be .yaml, .yml, or .json")]
    UnsupportedExtension,
    #[error("language pack `{0}` was not found")]
    MissingPack(String),
    #[error("language-pack path escapes configured directory: {0}")]
    PathEscape(PathBuf),
    #[error("language-pack path is not a regular file: {0}")]
    NotAFile(PathBuf),
    #[error("language pack is too large ({actual} bytes; maximum {maximum})")]
    Oversized { actual: u64, maximum: u64 },
    #[error("failed to access language pack `{path}`: {source}")]
    Io {
        path: PathBuf,
        #[source]
        source: io::Error,
    },
    #[error("failed to parse language pack `{path}`: {message}")]
    Parse { path: PathBuf, message: String },
    #[error("invalid language pack: {0}")]
    InvalidPack(String),
}

/// Loader rooted at a canonical configured directory.
#[derive(Debug, Clone)]
pub struct LanguagePackLoader {
    canonical_directory: PathBuf,
}

impl LanguagePackLoader {
    pub fn new(directory: impl AsRef<Path>) -> Result<Self, LanguagePackError> {
        let directory = directory.as_ref();
        if path_contains_nul(directory) {
            return Err(LanguagePackError::NulPath);
        }
        let canonical_directory =
            directory
                .canonicalize()
                .map_err(|source| LanguagePackError::Io {
                    path: directory.to_path_buf(),
                    source,
                })?;
        if !canonical_directory.is_dir() {
            return Err(LanguagePackError::InvalidDirectory(canonical_directory));
        }
        Ok(Self {
            canonical_directory,
        })
    }

    pub fn directory(&self) -> &Path {
        &self.canonical_directory
    }

    /// Loads `<language>.yaml`, `<language>.yml`, or `<language>.json`.
    /// Explicit supported extensions are also accepted.
    pub fn load(&self, language: &str) -> Result<LanguagePack, LanguagePackError> {
        if language.contains('\0') {
            return Err(LanguagePackError::InvalidLanguage(language.to_owned()));
        }
        let requested_path = Path::new(language);
        let extension = requested_path
            .extension()
            .and_then(|value| value.to_str())
            .map(str::to_ascii_lowercase);
        let stem = if extension.is_some() {
            requested_path
                .file_stem()
                .and_then(|value| value.to_str())
                .ok_or_else(|| LanguagePackError::InvalidLanguage(language.to_owned()))?
        } else {
            language
        };
        let requested_code = normalize_language_code(stem)?;

        let candidate_names = match extension.as_deref() {
            Some("yaml" | "yml" | "json") => vec![language.to_owned()],
            Some(_) => return Err(LanguagePackError::UnsupportedExtension),
            None => ["yaml", "yml", "json"]
                .into_iter()
                .map(|extension| format!("{requested_code}.{extension}"))
                .collect(),
        };

        let mut first_non_missing_error = None;
        for candidate_name in candidate_names {
            let candidate = self.canonical_directory.join(candidate_name);
            match self.load_candidate(&candidate, &requested_code) {
                Ok(pack) => return Ok(pack),
                Err(LanguagePackError::Io { source, .. })
                    if source.kind() == io::ErrorKind::NotFound => {}
                Err(error) => {
                    first_non_missing_error = Some(error);
                    break;
                }
            }
        }

        Err(first_non_missing_error.unwrap_or(LanguagePackError::MissingPack(requested_code)))
    }

    fn load_candidate(
        &self,
        candidate: &Path,
        requested_code: &str,
    ) -> Result<LanguagePack, LanguagePackError> {
        let link_metadata =
            std::fs::symlink_metadata(candidate).map_err(|source| LanguagePackError::Io {
                path: candidate.to_path_buf(),
                source,
            })?;
        if !link_metadata.file_type().is_file() && !link_metadata.file_type().is_symlink() {
            return Err(LanguagePackError::NotAFile(candidate.to_path_buf()));
        }
        let canonical = candidate
            .canonicalize()
            .map_err(|source| LanguagePackError::Io {
                path: candidate.to_path_buf(),
                source,
            })?;
        if !canonical.starts_with(&self.canonical_directory) {
            return Err(LanguagePackError::PathEscape(canonical));
        }
        let extension = canonical
            .extension()
            .and_then(|value| value.to_str())
            .map(str::to_ascii_lowercase);
        if !matches!(extension.as_deref(), Some("yaml" | "yml" | "json")) {
            return Err(LanguagePackError::UnsupportedExtension);
        }

        let mut file = File::open(&canonical).map_err(|source| LanguagePackError::Io {
            path: canonical.clone(),
            source,
        })?;
        let metadata = file.metadata().map_err(|source| LanguagePackError::Io {
            path: canonical.clone(),
            source,
        })?;
        if !metadata.is_file() {
            return Err(LanguagePackError::NotAFile(canonical));
        }
        if metadata.len() > MAX_LANGUAGE_PACK_BYTES {
            return Err(LanguagePackError::Oversized {
                actual: metadata.len(),
                maximum: MAX_LANGUAGE_PACK_BYTES,
            });
        }

        let mut bytes = Vec::with_capacity(metadata.len() as usize);
        file.by_ref()
            .take(MAX_LANGUAGE_PACK_BYTES + 1)
            .read_to_end(&mut bytes)
            .map_err(|source| LanguagePackError::Io {
                path: canonical.clone(),
                source,
            })?;
        if bytes.len() as u64 > MAX_LANGUAGE_PACK_BYTES {
            return Err(LanguagePackError::Oversized {
                actual: bytes.len() as u64,
                maximum: MAX_LANGUAGE_PACK_BYTES,
            });
        }

        let pack = match extension.as_deref() {
            Some("json") => {
                serde_json::from_slice(&bytes).map_err(|error| LanguagePackError::Parse {
                    path: canonical.clone(),
                    message: error.to_string(),
                })?
            }
            Some("yaml" | "yml") => {
                serde_yaml::from_slice(&bytes).map_err(|error| LanguagePackError::Parse {
                    path: canonical.clone(),
                    message: error.to_string(),
                })?
            }
            _ => return Err(LanguagePackError::UnsupportedExtension),
        };
        let pack: LanguagePack = pack;
        pack.validate(Some(requested_code))?;
        Ok(pack)
    }
}

/// How a language choice was made.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum LanguageDetectionSource {
    Explicit,
    Automatic,
    Fallback,
}

/// Deterministic lightweight language-detection result.
#[derive(Debug, Clone, PartialEq)]
pub struct LanguageDetection {
    pub language: String,
    pub confidence: f32,
    pub source: LanguageDetectionSource,
}

impl LanguageDetection {
    fn explicit(language: String) -> Self {
        Self {
            language,
            confidence: 1.0,
            source: LanguageDetectionSource::Explicit,
        }
    }
}

/// Detects a few common scripts and marker words. It is intentionally not a
/// general-purpose language classifier.
pub fn detect_language(text: &str) -> LanguageDetection {
    let mut letters = 0usize;
    let mut cyrillic = 0usize;
    let mut han = 0usize;
    let mut hiragana_katakana = 0usize;
    let mut hangul = 0usize;

    for character in text.chars() {
        if !character.is_alphabetic() {
            continue;
        }
        letters += 1;
        match character {
            '\u{0400}'..='\u{052F}' => cyrillic += 1,
            '\u{3040}'..='\u{30FF}' => hiragana_katakana += 1,
            '\u{3400}'..='\u{4DBF}' | '\u{4E00}'..='\u{9FFF}' => han += 1,
            '\u{AC00}'..='\u{D7AF}' => hangul += 1,
            _ => {}
        }
    }

    if letters > 0 {
        if hiragana_katakana * 2 >= letters {
            return automatic_detection("ja", 0.96);
        }
        if hangul * 2 >= letters {
            return automatic_detection("ko", 0.95);
        }
        if cyrillic * 2 >= letters {
            return automatic_detection("ru", 0.92);
        }
        if han * 2 >= letters {
            return automatic_detection("zh", 0.90);
        }
    }

    let words: Vec<String> = text
        .split(|character: char| !character.is_alphabetic() && character != '\'')
        .filter(|word| !word.is_empty())
        .map(str::to_lowercase)
        .collect();
    let candidates = [
        ("en", ENGLISH_MARKERS),
        ("es", SPANISH_MARKERS),
        ("fr", FRENCH_MARKERS),
        ("de", GERMAN_MARKERS),
    ];
    let mut scores: Vec<(&str, usize)> = candidates
        .into_iter()
        .map(|(language, markers)| {
            let score = words
                .iter()
                .filter(|word| markers.contains(&word.as_str()))
                .count();
            (language, score)
        })
        .collect();
    scores.sort_by(|left, right| right.1.cmp(&left.1).then_with(|| left.0.cmp(right.0)));
    let (language, best) = scores[0];
    let runner_up = scores.get(1).map_or(0, |score| score.1);
    let confidence = match best {
        0 => 0.0,
        1 => 0.56,
        2 => 0.72,
        3 => 0.84,
        _ => 0.92,
    } - if runner_up == best && best > 0 {
        0.12
    } else {
        0.0
    };

    if confidence >= DETECTION_CONFIDENCE_THRESHOLD {
        automatic_detection(language, confidence)
    } else {
        LanguageDetection {
            language: "en".to_owned(),
            confidence,
            source: LanguageDetectionSource::Fallback,
        }
    }
}

const ENGLISH_MARKERS: &[&str] = &[
    "and", "are", "because", "for", "from", "please", "that", "the", "this", "to", "with",
];
const SPANISH_MARKERS: &[&str] = &[
    "como", "con", "de", "el", "en", "es", "la", "para", "por", "que", "una",
];
const FRENCH_MARKERS: &[&str] = &[
    "avec", "dans", "de", "des", "est", "et", "la", "le", "les", "pour", "que",
];
const GERMAN_MARKERS: &[&str] = &[
    "das", "der", "die", "ein", "eine", "für", "ist", "mit", "und", "von", "zu",
];

fn automatic_detection(language: &str, confidence: f32) -> LanguageDetection {
    LanguageDetection {
        language: language.to_owned(),
        confidence,
        source: LanguageDetectionSource::Automatic,
    }
}

/// Resolves an explicit configured language or performs auto-detection.
pub fn resolve_language(text: &str, configured: Option<&str>) -> LanguageDetection {
    match configured.map(str::trim) {
        Some(language) if !language.is_empty() && !language.eq_ignore_ascii_case("auto") => {
            match normalize_language_code(language) {
                Ok(language) => LanguageDetection::explicit(language),
                Err(error) => {
                    tracing::warn!(%error, language, "Invalid configured language; falling back to English");
                    LanguageDetection {
                        language: "en".to_owned(),
                        confidence: 0.0,
                        source: LanguageDetectionSource::Fallback,
                    }
                }
            }
        }
        _ => detect_language(text),
    }
}

/// Compression engine backed by a built-in or external language pack.
#[derive(Debug, Clone)]
pub struct LanguagePackEngine {
    level: LanguagePackLevel,
    configured_language: Option<String>,
    loader: Option<LanguagePackLoader>,
    loaded_packs: Arc<RwLock<HashMap<String, Arc<LanguagePack>>>>,
}

impl LanguagePackEngine {
    /// Creates an engine that uses `CompressionContext::language` (`auto` opts
    /// into detection; the context default explicitly selects English).
    pub fn new(level: LanguagePackLevel) -> Self {
        Self {
            level,
            configured_language: None,
            loader: None,
            loaded_packs: Arc::new(RwLock::new(HashMap::new())),
        }
    }

    pub fn with_language(level: LanguagePackLevel, language: impl Into<String>) -> Self {
        let mut engine = Self::new(level);
        engine.configured_language = Some(language.into());
        engine
    }

    pub fn from_directory(
        level: LanguagePackLevel,
        language: Option<String>,
        directory: impl AsRef<Path>,
    ) -> Result<Self, LanguagePackError> {
        Ok(Self {
            level,
            configured_language: language,
            loader: Some(LanguagePackLoader::new(directory)?),
            loaded_packs: Arc::new(RwLock::new(HashMap::new())),
        })
    }

    /// Non-panicking configuration entry point. Invalid/missing directories are
    /// warned about and leave the built-in English fallback active.
    pub fn from_config(
        level: LanguagePackLevel,
        language: Option<String>,
        directory: impl AsRef<Path>,
    ) -> Self {
        match Self::from_directory(level, language.clone(), directory.as_ref()) {
            Ok(engine) => engine,
            Err(error) => {
                tracing::warn!(
                    %error,
                    directory = %directory.as_ref().display(),
                    "Language-pack directory unavailable; using built-in English"
                );
                Self {
                    level,
                    configured_language: language,
                    loader: None,
                    loaded_packs: Arc::new(RwLock::new(HashMap::new())),
                }
            }
        }
    }

    pub fn level(&self) -> LanguagePackLevel {
        self.level
    }

    fn pack_for(&self, language: &str) -> Arc<LanguagePack> {
        if language == "en" {
            return Arc::new(english_language_pack().clone());
        }
        if let Ok(cache) = self.loaded_packs.read() {
            if let Some(pack) = cache.get(language) {
                return Arc::clone(pack);
            }
        }

        let loaded = self
            .loader
            .as_ref()
            .and_then(|loader| match loader.load(language) {
                Ok(pack) => Some(Arc::new(pack)),
                Err(error) => {
                    tracing::warn!(
                        %error,
                        language,
                        "Language pack unavailable; falling back to built-in English"
                    );
                    None
                }
            });
        let Some(pack) = loaded else {
            return Arc::new(english_language_pack().clone());
        };
        if let Ok(mut cache) = self.loaded_packs.write() {
            cache.insert(language.to_owned(), Arc::clone(&pack));
        }
        pack
    }
}

impl Default for LanguagePackEngine {
    fn default() -> Self {
        Self::new(LanguagePackLevel::Light)
    }
}

#[async_trait]
impl CompressionEngine for LanguagePackEngine {
    fn name(&self) -> &str {
        "language_pack"
    }

    async fn compress(
        &self,
        payload: &mut CompressiblePayload,
        context: &CompressionContext,
    ) -> EngineResult {
        let started = Instant::now();
        let original = payload.clone();
        let tokens_before = count_payload_tokens(&original, context);
        let context_language = self
            .configured_language
            .as_deref()
            .or((context.language != "auto").then_some(context.language.as_str()));
        let mut changed = false;

        for message in &mut payload.messages {
            if message.cache_protected || matches!(message.role.as_str(), "system" | "tool") {
                continue;
            }
            let assistant = message.role == "assistant";
            transform_prose_leaves(message.content.as_value_mut(), &mut |text| {
                let detection = resolve_language(text, context_language);
                let pack = self.pack_for(&detection.language);
                let transformed = context
                    .protection_scanner
                    .transform_unprotected(text, |segment| {
                        transform_segment(segment, &pack, self.level, assistant)
                    });
                if transformed != *text {
                    *text = transformed;
                    changed = true;
                }
            });
        }

        if changed {
            payload.refresh_metadata();
            refresh_message_token_counts(payload, context);
        }
        let candidate_tokens = count_payload_tokens(payload, context);
        let applied = changed && candidate_tokens <= tokens_before;
        if candidate_tokens > tokens_before {
            *payload = original;
        }
        let tokens_after = if candidate_tokens > tokens_before {
            tokens_before
        } else {
            candidate_tokens
        };

        EngineResult {
            engine_name: self.name().to_owned(),
            tokens_before,
            tokens_after,
            duration_ms: started.elapsed().as_millis().min(u64::MAX as u128) as u64,
            applied,
        }
    }
}

fn transform_prose_leaves<F>(value: &mut Value, transform: &mut F)
where
    F: FnMut(&mut String),
{
    match value {
        Value::String(text) => transform(text),
        Value::Array(parts) => {
            for part in parts {
                match part {
                    Value::String(text) => transform(text),
                    Value::Object(object)
                        if matches!(
                            object.get("type").and_then(Value::as_str),
                            Some("text" | "input_text" | "output_text")
                        ) =>
                    {
                        if let Some(Value::String(text)) = object.get_mut("text") {
                            transform(text);
                        }
                    }
                    _ => {}
                }
            }
        }
        Value::Object(object)
            if matches!(
                object.get("type").and_then(Value::as_str),
                Some("text" | "input_text" | "output_text")
            ) =>
        {
            if let Some(Value::String(text)) = object.get_mut("text") {
                transform(text);
            }
        }
        _ => {}
    }
}

fn transform_segment(
    segment: &str,
    pack: &LanguagePack,
    level: LanguagePackLevel,
    assistant: bool,
) -> String {
    let rules = pack.levels.rules(level);
    let mut output = segment.to_owned();

    if rules.condense_verbose_phrases {
        output = replace_map(&output, &pack.verbose_phrases);
    }
    if rules.remove_fillers {
        output = remove_terms(&output, &pack.filler_words, false);
    }
    if rules.remove_hedges && assistant {
        output = remove_terms(&output, &pack.hedges, true);
    }
    if rules.remove_articles {
        output = remove_terms(&output, &pack.removable_articles, false);
    }
    if rules.apply_abbreviations {
        output = replace_map(&output, &pack.abbreviations);
    }

    if output == segment {
        output
    } else {
        normalize_artifacts(&output)
    }
}

fn replace_map(text: &str, replacements: &BTreeMap<String, String>) -> String {
    let mut ordered: Vec<_> = replacements.iter().collect();
    ordered.sort_by(|left, right| {
        right
            .0
            .len()
            .cmp(&left.0.len())
            .then_with(|| left.0.cmp(right.0))
    });
    let mut output = text.to_owned();
    for (phrase, replacement) in ordered {
        if let Ok(regex) = term_regex(phrase, false) {
            let replacement = replacement_with_boundaries(replacement);
            output = regex
                .replace_all(&output, replacement.as_str())
                .into_owned();
        }
    }
    output
}

fn remove_terms(text: &str, terms: &[String], trailing_punctuation: bool) -> String {
    let mut ordered = terms.to_vec();
    ordered.sort_by(|left, right| right.len().cmp(&left.len()).then_with(|| left.cmp(right)));
    let mut output = text.to_owned();
    for term in ordered {
        if let Ok(regex) = term_regex(&term, trailing_punctuation) {
            output = regex
                .replace_all(&output, replacement_with_boundaries("").as_str())
                .into_owned();
        }
    }
    output
}

fn term_regex(term: &str, trailing_punctuation: bool) -> Result<Regex, regex::Error> {
    let escaped = regex::escape(term.trim());
    let suffix = if trailing_punctuation {
        r#"(?:[\t ]*[,.;:!?])?(?P<after>[^\p{L}\p{N}_]|$)"#
    } else {
        r#"(?P<after>[^\p{L}\p{N}_]|$)"#
    };
    Regex::new(&format!(
        r"(?iu)(?P<before>^|[^\p{{L}}\p{{N}}_]){escaped}{suffix}"
    ))
}

fn replacement_with_boundaries(replacement: &str) -> String {
    if replacement.is_empty() {
        "${before}${after}".to_owned()
    } else {
        format!("${{before}}{replacement}${{after}}")
    }
}

static HORIZONTAL_WHITESPACE: LazyLock<Regex> =
    LazyLock::new(|| Regex::new(r"[\t ]{2,}").expect("static whitespace regex must compile"));
static SPACE_BEFORE_PUNCTUATION: LazyLock<Regex> = LazyLock::new(|| {
    Regex::new(r"[\t ]+([,.;:!?])").expect("static punctuation regex must compile")
});
static EMPTY_PUNCTUATION: LazyLock<Regex> = LazyLock::new(|| {
    Regex::new(r"(?m)(^|[.!?]\s+)[,;:]\s*").expect("static punctuation regex must compile")
});

fn normalize_artifacts(text: &str) -> String {
    let output = HORIZONTAL_WHITESPACE.replace_all(text, " ").into_owned();
    let output = SPACE_BEFORE_PUNCTUATION
        .replace_all(&output, "$1")
        .into_owned();
    let output = EMPTY_PUNCTUATION.replace_all(&output, "$1").into_owned();
    output.trim().to_owned()
}

fn normalize_language_code(language: &str) -> Result<String, LanguagePackError> {
    let language = language.trim().to_ascii_lowercase();
    let valid = !language.is_empty()
        && language.len() <= 35
        && language
            .bytes()
            .all(|byte| byte.is_ascii_alphanumeric() || byte == b'-')
        && language
            .as_bytes()
            .first()
            .is_some_and(u8::is_ascii_alphabetic)
        && language
            .as_bytes()
            .last()
            .is_some_and(u8::is_ascii_alphanumeric)
        && !language.contains("--");
    if valid {
        Ok(language)
    } else {
        Err(LanguagePackError::InvalidLanguage(language))
    }
}

fn path_contains_nul(path: &Path) -> bool {
    path.as_os_str().to_string_lossy().contains('\0')
}

fn count_payload_tokens(payload: &CompressiblePayload, context: &CompressionContext) -> u32 {
    context
        .token_counter
        .count_request(&payload.clone().into_openai_request())
}

fn refresh_message_token_counts(payload: &mut CompressiblePayload, context: &CompressionContext) {
    for message in &mut payload.messages {
        let content_tokens = match message.content.as_value() {
            Value::Null => 0,
            Value::String(text) => context.token_counter.count_text(&context.model, text),
            structured => context
                .token_counter
                .count_text(&context.model, &structured.to_string()),
        };
        let extra_tokens = if message.extra.is_empty() {
            0
        } else {
            context.token_counter.count_text(
                &context.model,
                &Value::Object(message.extra.clone()).to_string(),
            )
        };
        message.token_count = 4u32
            .saturating_add(
                context
                    .token_counter
                    .count_text(&context.model, &message.role),
            )
            .saturating_add(content_tokens)
            .saturating_add(extra_tokens);
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::models::openai::OpenAIRequest;
    use serde_json::{json, Map};
    use std::fs;
    use tempfile::tempdir;

    fn payload(value: Value) -> CompressiblePayload {
        let request: OpenAIRequest = serde_json::from_value(value).unwrap();
        CompressiblePayload::from(request)
    }

    fn context(language: &str) -> CompressionContext {
        let mut context = CompressionContext::new("gpt-4o", "test");
        context.language = language.to_owned();
        context
    }

    fn custom_pack_yaml() -> &'static str {
        r#"
code: es
name: Spanish test pack
filler_words: [realmente, básicamente]
verbose_phrases:
  con el fin de: para
  usted debe: ""
hedges: [espero que esto ayude]
removable_articles: [el, la, los, las]
abbreviations:
  y: "&"
"#
    }

    #[derive(Clone, Copy)]
    struct FixedCounter(u32);

    impl FixedCounter {
        fn count_payload(self, _payload: &CompressiblePayload) -> u32 {
            self.0
        }
    }

    #[derive(Debug, Clone)]
    struct HarnessMessage {
        role: String,
        content: Value,
        extra: Map<String, Value>,
        cache_protected: bool,
    }

    #[derive(Debug, Clone)]
    struct HarnessPayload {
        messages: Vec<HarnessMessage>,
        tool_definitions: Option<Value>,
    }

    fn harness_message(role: &str, content: Value) -> HarnessMessage {
        HarnessMessage {
            role: role.to_owned(),
            content,
            extra: Map::new(),
            cache_protected: false,
        }
    }

    fn compress_harness(
        engine: &LanguagePackEngine,
        payload: &mut HarnessPayload,
        configured: Option<&str>,
    ) -> bool {
        let before = serde_json::to_string(&json!({
            "messages": payload.messages.iter().map(|message| json!({
                "role": message.role,
                "content": message.content,
                "extra": message.extra,
                "cache_protected": message.cache_protected,
            })).collect::<Vec<_>>(),
            "tools": payload.tool_definitions,
        }))
        .unwrap();
        let original = payload.clone();
        let mut changed = false;
        for message in &mut payload.messages {
            if message.cache_protected || matches!(message.role.as_str(), "system" | "tool") {
                continue;
            }
            let assistant = message.role == "assistant";
            let scanner = crate::compression::protection::ProtectionScanner::default();
            transform_prose_leaves(&mut message.content, &mut |text| {
                let detection = resolve_language(text, configured);
                let pack = engine.pack_for(&detection.language);
                let transformed = scanner.transform_unprotected(text, |segment| {
                    transform_segment(segment, &pack, engine.level, assistant)
                });
                if transformed != *text {
                    *text = transformed;
                    changed = true;
                }
            });
        }
        let after = serde_json::to_string(&json!({
            "messages": payload.messages.iter().map(|message| json!({
                "role": message.role,
                "content": message.content,
                "extra": message.extra,
                "cache_protected": message.cache_protected,
            })).collect::<Vec<_>>(),
            "tools": payload.tool_definitions,
        }))
        .unwrap();
        if after.len() > before.len() {
            *payload = original;
            false
        } else {
            changed
        }
    }

    #[test]
    fn transform_rules_cover_all_levels_without_regex_panics() {
        let source = "Basically, I hope this helps. You should use the tool and additional information in order to finish.";
        let pack = english_language_pack();
        let light = transform_segment(source, pack, LanguagePackLevel::Light, true);
        let full = transform_segment(source, pack, LanguagePackLevel::Full, true);
        let maximum = transform_segment(source, pack, LanguagePackLevel::Maximum, true);

        assert_eq!(light, "I hope this helps. You should use the tool and additional information in order to finish.");
        assert_eq!(full, "use tool and additional information to finish.");
        assert_eq!(maximum, "use tool & details to finish.");
        assert_ne!(light, full);
        assert_ne!(full, maximum);
        assert_eq!(
            maximum,
            transform_segment(source, pack, LanguagePackLevel::Maximum, true)
        );
        assert_eq!(
            FixedCounter(7).count_payload(&payload(json!({"model":"gpt-4o","messages":[]}))),
            7
        );
    }

    #[test]
    fn harness_applies_custom_pack_and_preserves_structural_data() {
        let directory = tempdir().unwrap();
        fs::write(directory.path().join("es.yaml"), custom_pack_yaml()).unwrap();
        let engine = LanguagePackEngine::from_directory(
            LanguagePackLevel::Maximum,
            Some("es".to_owned()),
            directory.path(),
        )
        .unwrap();
        let tools = json!([{"name":"actuallyImportant","description":"Please retain"}]);
        let mut cached = harness_message("user", json!("Realmente preserve cached prose."));
        cached.cache_protected = true;
        let mut payload = HarnessPayload {
            messages: vec![
                harness_message("system", json!("Realmente preserve system prose.")),
                cached,
                harness_message("tool", json!("Realmente preserve tool prose.")),
                harness_message(
                    "assistant",
                    json!([
                        {"type":"text","text":"Realmente, espero que esto ayude. Usted debe usar el sistema y la herramienta con el fin de terminar."},
                        {"type":"tool_result","content":[{"type":"text","text":"Realmente preserve nested tool prose."}]}
                    ]),
                ),
            ],
            tool_definitions: Some(tools.clone()),
        };

        assert!(compress_harness(&engine, &mut payload, Some("es")));
        assert_eq!(payload.tool_definitions, Some(tools));
        assert_eq!(
            payload.messages[0].content,
            json!("Realmente preserve system prose.")
        );
        assert_eq!(
            payload.messages[1].content,
            json!("Realmente preserve cached prose.")
        );
        assert_eq!(
            payload.messages[2].content,
            json!("Realmente preserve tool prose.")
        );
        assert_eq!(
            payload.messages[3].content[0]["text"],
            "usar sistema & herramienta para terminar."
        );
        assert_eq!(
            payload.messages[3].content[1]["content"][0]["text"],
            "Realmente preserve nested tool prose."
        );
    }

    #[test]
    fn harness_preserves_protected_technical_regions() {
        let protected = [
            "```rust\nlet actual_value = reallyImportant;\n```",
            "https://example.test/actually/path?q=really",
            "/usr/local/very/important.rs",
            r"C:\Users\alice\actually.rs",
            r#"{"actually":"very", "please":true}"#,
            "camelCaseIdentifier",
            "snake_case_identifier",
        ];
        let source = format!(
            "Please actually inspect these:\n{}\n{}\n{}\n{}\n{}\n{}\n{}\nReally finish.",
            protected[0],
            protected[1],
            protected[2],
            protected[3],
            protected[4],
            protected[5],
            protected[6]
        );
        let engine = LanguagePackEngine::with_language(LanguagePackLevel::Maximum, "en");
        let mut payload = HarnessPayload {
            messages: vec![harness_message("user", json!(source))],
            tool_definitions: None,
        };

        assert!(compress_harness(&engine, &mut payload, Some("en")));
        let output = payload.messages[0].content.as_str().unwrap();
        for expected in protected {
            assert!(
                output.contains(expected),
                "missing protected bytes: {expected}"
            );
        }
    }

    #[test]
    fn harness_rolls_back_expanding_rule() {
        let directory = tempdir().unwrap();
        fs::write(
            directory.path().join("xx.json"),
            serde_json::to_vec(&json!({
                "code":"xx",
                "name":"Expanding test",
                "verbose_phrases":{"ok":"an intentionally enormous replacement phrase repeated many times"}
            }))
            .unwrap(),
        )
        .unwrap();
        let engine = LanguagePackEngine::from_directory(
            LanguagePackLevel::Full,
            Some("xx".to_owned()),
            directory.path(),
        )
        .unwrap();
        let mut payload = HarnessPayload {
            messages: vec![harness_message("user", json!("ok"))],
            tool_definitions: None,
        };
        let original = payload.clone();

        assert!(!compress_harness(&engine, &mut payload, Some("xx")));
        assert_eq!(payload.messages[0].content, original.messages[0].content);
    }

    #[test]
    fn built_in_english_pack_has_all_required_rule_groups_and_levels() {
        let pack = english_language_pack();
        assert_eq!(pack.code, "en");
        assert_eq!(pack.name, "English");
        assert!(!pack.filler_words.is_empty());
        assert!(!pack.verbose_phrases.is_empty());
        assert!(!pack.hedges.is_empty());
        assert!(!pack.removable_articles.is_empty());
        assert_ne!(pack.levels.light, pack.levels.full);
        assert_ne!(pack.levels.full, pack.levels.maximum);
        pack.validate(Some("en")).unwrap();
    }

    #[tokio::test]
    async fn light_full_and_maximum_are_distinct_and_deterministic() {
        let source = "Basically, I hope this helps. You should use the tool and additional information in order to finish.";
        let mut outputs = Vec::new();
        for level in [
            LanguagePackLevel::Light,
            LanguagePackLevel::Full,
            LanguagePackLevel::Maximum,
        ] {
            let mut value = payload(json!({
                "model": "gpt-4o",
                "messages": [{"role": "assistant", "content": source}]
            }));
            let result = LanguagePackEngine::with_language(level, "en")
                .compress(&mut value, &context("en"))
                .await;
            assert!(result.applied);
            assert!(result.tokens_after <= result.tokens_before);
            outputs.push(value.messages[0].content.as_text().unwrap().to_owned());
        }

        assert_eq!(outputs[0], "I hope this helps. You should use the tool and additional information in order to finish.");
        assert_eq!(outputs[1], "use tool and additional information to finish.");
        assert_eq!(outputs[2], "use tool & details to finish.");
        assert!(outputs.windows(2).all(|pair| pair[0] != pair[1]));
    }

    #[test]
    fn explicit_auto_and_low_confidence_fallback_are_reported() {
        let explicit = resolve_language("ambiguous", Some("es"));
        assert_eq!(explicit.language, "es");
        assert_eq!(explicit.confidence, 1.0);
        assert_eq!(explicit.source, LanguageDetectionSource::Explicit);

        let automatic = resolve_language("el sistema es para la casa y una persona", Some("auto"));
        assert_eq!(automatic.language, "es");
        assert!(automatic.confidence >= DETECTION_CONFIDENCE_THRESHOLD);
        assert_eq!(automatic.source, LanguageDetectionSource::Automatic);

        let fallback = resolve_language("zxqv blorp", None);
        assert_eq!(fallback.language, "en");
        assert!(fallback.confidence < DETECTION_CONFIDENCE_THRESHOLD);
        assert_eq!(fallback.source, LanguageDetectionSource::Fallback);

        let script = detect_language("これは日本語の文章です");
        assert_eq!(script.language, "ja");
        assert!(script.confidence > 0.9);
    }

    #[tokio::test]
    async fn loads_and_applies_custom_pack_from_configured_directory() {
        let directory = tempdir().unwrap();
        fs::write(directory.path().join("es.yaml"), custom_pack_yaml()).unwrap();
        let engine = LanguagePackEngine::from_directory(
            LanguagePackLevel::Maximum,
            Some("es".to_owned()),
            directory.path(),
        )
        .unwrap();
        let mut value = payload(json!({
            "model": "gpt-4o",
            "messages": [{"role": "assistant", "content": "Realmente, espero que esto ayude. Usted debe usar el sistema y la herramienta con el fin de terminar."}]
        }));

        let result = engine.compress(&mut value, &context("en")).await;

        assert!(result.applied);
        assert_eq!(
            value.messages[0].content.as_text(),
            Some("usar sistema & herramienta para terminar.")
        );
    }

    #[test]
    fn loader_rejects_traversal_nul_non_files_and_unsupported_extensions() {
        let directory = tempdir().unwrap();
        fs::write(directory.path().join("en.yaml"), "code: en\nname: test\n").unwrap();
        fs::create_dir(directory.path().join("fr.yaml")).unwrap();
        fs::write(directory.path().join("de.txt"), "code: de\nname: test\n").unwrap();
        let loader = LanguagePackLoader::new(directory.path()).unwrap();

        assert!(matches!(
            loader.load("../en"),
            Err(LanguagePackError::InvalidLanguage(_))
        ));
        assert!(matches!(
            loader.load("en\0.yaml"),
            Err(LanguagePackError::InvalidLanguage(_))
        ));
        assert!(matches!(
            loader.load("fr.yaml"),
            Err(LanguagePackError::NotAFile(_))
        ));
        assert!(matches!(
            loader.load("de.txt"),
            Err(LanguagePackError::UnsupportedExtension)
        ));
    }

    #[cfg(unix)]
    #[test]
    fn loader_rejects_symlink_escape() {
        use std::os::unix::fs::symlink;

        let directory = tempdir().unwrap();
        let outside = tempdir().unwrap();
        let target = outside.path().join("es.yaml");
        fs::write(&target, custom_pack_yaml()).unwrap();
        symlink(&target, directory.path().join("es.yaml")).unwrap();
        let loader = LanguagePackLoader::new(directory.path()).unwrap();

        assert!(matches!(
            loader.load("es"),
            Err(LanguagePackError::PathEscape(_))
        ));
    }

    #[cfg(windows)]
    #[test]
    fn loader_rejects_symlink_escape_when_symlinks_are_available() {
        use std::os::windows::fs::symlink_file;

        let directory = tempdir().unwrap();
        let outside = tempdir().unwrap();
        let target = outside.path().join("es.yaml");
        fs::write(&target, custom_pack_yaml()).unwrap();
        if symlink_file(&target, directory.path().join("es.yaml")).is_err() {
            return;
        }
        let loader = LanguagePackLoader::new(directory.path()).unwrap();
        assert!(matches!(
            loader.load("es"),
            Err(LanguagePackError::PathEscape(_))
        ));
    }

    #[test]
    fn loader_rejects_malformed_and_oversized_files() {
        let directory = tempdir().unwrap();
        fs::write(directory.path().join("es.yaml"), "code: [not valid").unwrap();
        fs::write(
            directory.path().join("fr.json"),
            vec![b' '; MAX_LANGUAGE_PACK_BYTES as usize + 1],
        )
        .unwrap();
        let loader = LanguagePackLoader::new(directory.path()).unwrap();

        assert!(matches!(
            loader.load("es"),
            Err(LanguagePackError::Parse { .. })
        ));
        assert!(matches!(
            loader.load("fr"),
            Err(LanguagePackError::Oversized { .. })
        ));
    }

    #[tokio::test]
    async fn preserves_code_paths_urls_json_and_identifiers_byte_for_byte() {
        let protected = [
            "```rust\nlet actual_value = reallyImportant;\n```",
            "https://example.test/actually/path?q=really",
            "/usr/local/very/important.rs",
            r"C:\Users\alice\actually.rs",
            r#"{"actually":"very", "please":true}"#,
            "camelCaseIdentifier",
            "snake_case_identifier",
        ];
        let source = format!(
            "Please actually inspect these:\n{}\n{}\n{}\n{}\n{}\n{}\n{}\nReally finish.",
            protected[0],
            protected[1],
            protected[2],
            protected[3],
            protected[4],
            protected[5],
            protected[6]
        );
        let mut value = payload(json!({
            "model": "gpt-4o",
            "messages": [{"role": "user", "content": source}]
        }));

        let result = LanguagePackEngine::with_language(LanguagePackLevel::Maximum, "en")
            .compress(&mut value, &context("en"))
            .await;
        let output = value.messages[0].content.as_text().unwrap();

        assert!(result.applied);
        for expected in protected {
            assert!(
                output.contains(expected),
                "missing protected bytes: {expected}"
            );
        }
        assert!(!output.contains("Please actually inspect"));
        assert!(!output.contains("Really finish"));
    }

    #[tokio::test]
    async fn skips_system_cache_tool_roles_tool_blocks_and_schema_data() {
        let tools = json!([{
            "type": "function",
            "function": {
                "name": "actuallyImportant",
                "description": "Please actually retain this very verbose schema.",
                "parameters": {"type": "object", "properties": {"please": {"type": "string"}}}
            }
        }]);
        let mut value = payload(json!({
            "model": "gpt-4o",
            "tools": tools,
            "messages": [
                {"role": "system", "content": "Please actually preserve this very important policy."},
                {"role": "user", "content": [{"type": "text", "text": "Please actually preserve cached prose.", "cache_control": {"type": "ephemeral"}}]},
                {"role": "tool", "tool_call_id": "call_1", "content": "Please actually preserve tool prose."},
                {"role": "assistant", "content": [
                    {"type": "text", "text": "Please actually compress visible prose."},
                    {"type": "tool_result", "tool_use_id": "call_1", "content": [{"type": "text", "text": "Please actually preserve nested tool data."}]}
                ]}
            ]
        }));
        let original_tools = value.tool_definitions.clone();
        let original_prefix: Vec<_> = value.messages[..3]
            .iter()
            .map(|message| message.content.clone())
            .collect();

        let result = LanguagePackEngine::with_language(LanguagePackLevel::Full, "en")
            .compress(&mut value, &context("en"))
            .await;

        assert!(result.applied);
        assert_eq!(value.tool_definitions, original_tools);
        assert_eq!(value.messages[0].content, original_prefix[0]);
        assert_eq!(value.messages[1].content, original_prefix[1]);
        assert_eq!(value.messages[2].content, original_prefix[2]);
        assert_eq!(
            value.messages[3].content.as_value()[0]["text"],
            "compress visible prose."
        );
        assert_eq!(
            value.messages[3].content.as_value()[1]["content"][0]["text"],
            "Please actually preserve nested tool data."
        );
    }

    #[tokio::test]
    async fn missing_pack_falls_back_without_panicking_and_counts_are_accurate() {
        let directory = tempdir().unwrap();
        let engine = LanguagePackEngine::from_directory(
            LanguagePackLevel::Full,
            Some("zz".to_owned()),
            directory.path(),
        )
        .unwrap();
        let mut value = payload(json!({
            "model": "gpt-4o",
            "messages": [{"role": "user", "content": "Please actually use the tool in order to finish."}]
        }));
        let context = context("zz");
        let expected_before = count_payload_tokens(&value, &context);

        let result = engine.compress(&mut value, &context).await;
        let actual_after = count_payload_tokens(&value, &context);

        assert!(result.applied);
        assert_eq!(result.tokens_before, expected_before);
        assert_eq!(result.tokens_after, actual_after);
        assert!(result.tokens_after <= result.tokens_before);
    }

    #[tokio::test]
    async fn never_increases_tokens_and_rolls_back_expanding_custom_rules() {
        let directory = tempdir().unwrap();
        fs::write(
            directory.path().join("xx.json"),
            serde_json::to_vec(&json!({
                "code": "xx",
                "name": "Expanding test",
                "verbose_phrases": {"ok": "an intentionally enormous replacement phrase repeated many times"}
            }))
            .unwrap(),
        )
        .unwrap();
        let engine = LanguagePackEngine::from_directory(
            LanguagePackLevel::Full,
            Some("xx".to_owned()),
            directory.path(),
        )
        .unwrap();
        let mut value = payload(json!({
            "model": "gpt-4o",
            "messages": [{"role": "user", "content": "ok"}]
        }));
        let original = value.clone();

        let result = engine.compress(&mut value, &context("xx")).await;

        assert_eq!(value, original);
        assert!(!result.applied);
        assert_eq!(result.tokens_after, result.tokens_before);
    }
}
