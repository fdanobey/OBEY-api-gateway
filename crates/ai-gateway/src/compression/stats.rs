//! Compression statistics and safe structured logging.

use super::{
    pipeline::CompressionPipelineResult, CompressionLevel, EngineResult as PipelineEngineResult,
};
use regex::Regex;
use serde::{Deserialize, Serialize, Serializer};
use serde_json::Value;
use std::sync::OnceLock;

/// Maximum UTF-8 byte length retained for a request identifier.
pub const MAX_REQUEST_ID_LEN: usize = 128;
/// Maximum UTF-8 byte length retained for a provider label.
pub const MAX_PROVIDER_LEN: usize = 64;
/// Maximum UTF-8 byte length retained for a model label.
pub const MAX_MODEL_LEN: usize = 128;
/// Maximum UTF-8 byte length retained for an engine label.
pub const MAX_ENGINE_LABEL_LEN: usize = 64;

const REDACTED: &str = "[REDACTED]";

/// Serializable, content-free statistics for one compression engine invocation.
#[derive(Debug, Clone, PartialEq, Deserialize)]
pub struct CompressionEngineStats {
    pub engine_name: String,
    pub tokens_before: u32,
    pub tokens_after: u32,
    pub tokens_saved: u32,
    pub savings_percent: f64,
    pub duration_ms: u64,
    pub applied: bool,
}

impl From<&PipelineEngineResult> for CompressionEngineStats {
    fn from(result: &PipelineEngineResult) -> Self {
        let tokens_saved = result.tokens_before.saturating_sub(result.tokens_after);
        Self {
            engine_name: sanitize_operational_metadata(&result.engine_name, MAX_ENGINE_LABEL_LEN),
            tokens_before: result.tokens_before,
            tokens_after: result.tokens_after,
            tokens_saved,
            savings_percent: savings_percent(result.tokens_before, result.tokens_after),
            duration_ms: result.duration_ms,
            applied: result.applied,
        }
    }
}

/// Complete content-free compression statistics for a request.
#[derive(Debug, Clone, PartialEq, Deserialize)]
pub struct CompressionStats {
    pub request_id: String,
    pub level: CompressionLevel,
    pub engines_applied: Vec<String>,
    pub original_tokens: u32,
    pub compressed_tokens: u32,
    pub savings_percent: f64,
    pub compression_time_ms: u64,
    pub auto_triggered: bool,
    pub cache_downgrade_applied: bool,
    pub tool_definitions_tokens_saved: u32,
    pub caveman_applied: bool,
    pub timed_out: bool,
    pub error: bool,
    pub provider: String,
    pub model: String,
    pub engine_results: Vec<CompressionEngineStats>,
}

impl CompressionStats {
    /// Returns the saturating number of tokens saved by the full pipeline.
    pub fn tokens_saved(&self) -> u32 {
        self.original_tokens.saturating_sub(self.compressed_tokens)
    }

    /// Builds safe operational statistics without retaining request or response content.
    pub fn from_pipeline_result(
        result: &CompressionPipelineResult,
        caveman_applied: bool,
        provider: impl AsRef<str>,
        model: impl AsRef<str>,
    ) -> Self {
        Self {
            request_id: sanitize_operational_metadata(&result.request_id, MAX_REQUEST_ID_LEN),
            level: result.level,
            engines_applied: result
                .engines_applied
                .iter()
                .map(|engine| sanitize_operational_metadata(engine, MAX_ENGINE_LABEL_LEN))
                .collect(),
            original_tokens: result.original_tokens,
            compressed_tokens: result.final_tokens,
            savings_percent: savings_percent(result.original_tokens, result.final_tokens),
            compression_time_ms: result.duration_ms,
            auto_triggered: result.auto_triggered,
            cache_downgrade_applied: result.cache_downgrade_applied,
            tool_definitions_tokens_saved: result.tool_definitions_tokens_saved,
            caveman_applied,
            timed_out: result.timed_out,
            error: result.error,
            provider: sanitize_operational_metadata(provider.as_ref(), MAX_PROVIDER_LEN),
            model: sanitize_operational_metadata(model.as_ref(), MAX_MODEL_LEN),
            engine_results: result
                .engine_results
                .iter()
                .map(CompressionEngineStats::from)
                .collect(),
        }
    }

    /// Alternate argument order for callers that group provider/model metadata first.
    pub fn from_pipeline_result_with_metadata(
        result: &CompressionPipelineResult,
        provider: impl AsRef<str>,
        model: impl AsRef<str>,
        caveman_applied: bool,
    ) -> Self {
        Self::from_pipeline_result(result, caveman_applied, provider, model)
    }

    /// Serializes this content-free event for logger and dashboard integrations.
    pub fn to_json(&self) -> serde_json::Result<String> {
        serde_json::to_string(self)
    }

    /// Serializes this content-free event as a JSON value.
    pub fn to_json_value(&self) -> serde_json::Result<Value> {
        serde_json::to_value(self)
    }

    /// Emits one structured INFO event containing safe operational metadata only.
    pub fn log(&self) {
        let sanitized_request_id =
            sanitize_operational_metadata(&self.request_id, MAX_REQUEST_ID_LEN);
        let sanitized_provider = sanitize_operational_metadata(&self.provider, MAX_PROVIDER_LEN);
        let sanitized_model = sanitize_operational_metadata(&self.model, MAX_MODEL_LEN);
        let sanitized_engines_applied = self
            .engines_applied
            .iter()
            .map(|engine| sanitize_operational_metadata(engine, MAX_ENGINE_LABEL_LEN))
            .collect::<Vec<_>>();
        let sanitized_engine_results = self
            .engine_results
            .iter()
            .cloned()
            .map(|mut result| {
                result.engine_name =
                    sanitize_operational_metadata(&result.engine_name, MAX_ENGINE_LABEL_LEN);
                result
            })
            .collect::<Vec<_>>();
        let engines_applied =
            serde_json::to_string(&sanitized_engines_applied).unwrap_or_else(|_| "[]".to_owned());
        let engine_results =
            serde_json::to_string(&sanitized_engine_results).unwrap_or_else(|_| "[]".to_owned());

        tracing::info!(
            target: "ai_gateway::compression_stats",
            request_id = %sanitized_request_id,
            provider = %sanitized_provider,
            model = %sanitized_model,
            level = ?self.level,
            original_tokens = self.original_tokens,
            compressed_tokens = self.compressed_tokens,
            tokens_saved = self.tokens_saved(),
            savings_percent = self.savings_percent,
            compression_time_ms = self.compression_time_ms,
            auto_triggered = self.auto_triggered,
            cache_downgrade_applied = self.cache_downgrade_applied,
            tool_definitions_tokens_saved = self.tool_definitions_tokens_saved,
            caveman_applied = self.caveman_applied,
            timed_out = self.timed_out,
            error = self.error,
            engines_applied = %engines_applied,
            engine_results = %engine_results,
            "Compression statistics"
        );
    }
}

impl Serialize for CompressionEngineStats {
    fn serialize<S>(&self, serializer: S) -> Result<S::Ok, S::Error>
    where
        S: Serializer,
    {
        use serde::ser::SerializeStruct;

        let mut state = serializer.serialize_struct("CompressionEngineStats", 7)?;
        state.serialize_field(
            "engine_name",
            &sanitize_operational_metadata(&self.engine_name, MAX_ENGINE_LABEL_LEN),
        )?;
        state.serialize_field("tokens_before", &self.tokens_before)?;
        state.serialize_field("tokens_after", &self.tokens_after)?;
        state.serialize_field(
            "tokens_saved",
            &self.tokens_before.saturating_sub(self.tokens_after),
        )?;
        state.serialize_field(
            "savings_percent",
            &savings_percent(self.tokens_before, self.tokens_after),
        )?;
        state.serialize_field("duration_ms", &self.duration_ms)?;
        state.serialize_field("applied", &self.applied)?;
        state.end()
    }
}

impl Serialize for CompressionStats {
    fn serialize<S>(&self, serializer: S) -> Result<S::Ok, S::Error>
    where
        S: Serializer,
    {
        use serde::ser::SerializeStruct;

        let engines_applied = self
            .engines_applied
            .iter()
            .map(|engine| sanitize_operational_metadata(engine, MAX_ENGINE_LABEL_LEN))
            .collect::<Vec<_>>();
        let engine_results = self
            .engine_results
            .iter()
            .cloned()
            .map(|mut result| {
                result.engine_name =
                    sanitize_operational_metadata(&result.engine_name, MAX_ENGINE_LABEL_LEN);
                result
            })
            .collect::<Vec<_>>();

        let mut state = serializer.serialize_struct("CompressionStats", 18)?;
        state.serialize_field(
            "request_id",
            &sanitize_operational_metadata(&self.request_id, MAX_REQUEST_ID_LEN),
        )?;
        state.serialize_field("level", &self.level)?;
        state.serialize_field("engines_applied", &engines_applied)?;
        state.serialize_field("original_tokens", &self.original_tokens)?;
        state.serialize_field("compressed_tokens", &self.compressed_tokens)?;
        state.serialize_field("tokens_saved", &self.tokens_saved())?;
        state.serialize_field(
            "savings_percent",
            &savings_percent(self.original_tokens, self.compressed_tokens),
        )?;
        state.serialize_field("compression_time_ms", &self.compression_time_ms)?;
        state.serialize_field("auto_triggered", &self.auto_triggered)?;
        state.serialize_field("cache_downgrade_applied", &self.cache_downgrade_applied)?;
        state.serialize_field(
            "tool_definitions_tokens_saved",
            &self.tool_definitions_tokens_saved,
        )?;
        state.serialize_field("caveman_applied", &self.caveman_applied)?;
        state.serialize_field("timed_out", &self.timed_out)?;
        state.serialize_field("error", &self.error)?;
        state.serialize_field(
            "provider",
            &sanitize_operational_metadata(&self.provider, MAX_PROVIDER_LEN),
        )?;
        state.serialize_field(
            "model",
            &sanitize_operational_metadata(&self.model, MAX_MODEL_LEN),
        )?;
        state.serialize_field("engine_results", &engine_results)?;
        state.end()
    }
}

impl From<(&CompressionPipelineResult, bool, &str, &str)> for CompressionStats {
    fn from(
        (result, caveman_applied, provider, model): (&CompressionPipelineResult, bool, &str, &str),
    ) -> Self {
        Self::from_pipeline_result(result, caveman_applied, provider, model)
    }
}

impl From<(&CompressionPipelineResult, &str, &str, bool)> for CompressionStats {
    fn from(
        (result, provider, model, caveman_applied): (&CompressionPipelineResult, &str, &str, bool),
    ) -> Self {
        Self::from_pipeline_result(result, caveman_applied, provider, model)
    }
}

/// Removes log-forging characters and secrets, then caps the UTF-8 byte length.
pub fn sanitize_operational_metadata(value: &str, max_len: usize) -> String {
    let without_controls = value
        .chars()
        .map(|character| {
            if character.is_control() {
                ' '
            } else {
                character
            }
        })
        .collect::<String>();
    let redacted = redact_secrets(&without_controls);
    truncate_utf8(&redacted, max_len)
}

fn savings_percent(original_tokens: u32, compressed_tokens: u32) -> f64 {
    if original_tokens == 0 {
        return 0.0;
    }

    f64::from(original_tokens.saturating_sub(compressed_tokens)) * 100.0
        / f64::from(original_tokens)
}

fn redact_secrets(value: &str) -> String {
    static URL_PASSWORD: OnceLock<Regex> = OnceLock::new();
    static BEARER: OnceLock<Regex> = OnceLock::new();
    static OPENAI_KEY: OnceLock<Regex> = OnceLock::new();
    static AWS_ACCESS_KEY: OnceLock<Regex> = OnceLock::new();

    let value = URL_PASSWORD
        .get_or_init(|| {
            Regex::new(r"(?i)([a-z][a-z0-9+.-]*://[^/@:\s]+:)[^@/\s]+(@)")
                .expect("URL-password redaction regex must compile")
        })
        .replace_all(value, format!("$1{REDACTED}$2"));
    let value = BEARER
        .get_or_init(|| {
            Regex::new(r"(?i)\bbearer\s+[^\s,;]+")
                .expect("bearer-token redaction regex must compile")
        })
        .replace_all(&value, REDACTED);
    let value = OPENAI_KEY
        .get_or_init(|| {
            Regex::new(r"(?i)\bsk-[a-z0-9_-]*").expect("OpenAI-key redaction regex must compile")
        })
        .replace_all(&value, REDACTED);
    AWS_ACCESS_KEY
        .get_or_init(|| {
            Regex::new(r"(?i)\bAKIA[a-z0-9_-]*").expect("AWS-key redaction regex must compile")
        })
        .replace_all(&value, REDACTED)
        .into_owned()
}

fn truncate_utf8(value: &str, max_len: usize) -> String {
    if value.len() <= max_len {
        return value.to_owned();
    }

    let mut end = 0;
    for (index, character) in value.char_indices() {
        let next = index + character.len_utf8();
        if next > max_len {
            break;
        }
        end = next;
    }
    value[..end].to_owned()
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{
        compression::{
            pipeline::{CompressionPipelineError, CompressionPipelineResult},
            CompressiblePayload, EngineResult,
        },
        models::openai::OpenAIRequest,
    };
    use proptest::prelude::*;
    use serde_json::{Map, Value};

    fn pipeline_result() -> CompressionPipelineResult {
        CompressionPipelineResult {
            payload: CompressiblePayload::from_openai_request(OpenAIRequest {
                model: "RAW_CONTENT_MUST_NOT_APPEAR".to_owned(),
                messages: Vec::new(),
                stream: false,
                temperature: None,
                max_tokens: None,
                extra: Map::from_iter([(
                    "content".to_owned(),
                    Value::String("RAW_CONTENT_MUST_NOT_APPEAR".to_owned()),
                )]),
            }),
            request_id: "request-123".to_owned(),
            level: CompressionLevel::Standard,
            engine_results: vec![
                EngineResult {
                    engine_name: "lite".to_owned(),
                    tokens_before: 1_000,
                    tokens_after: 800,
                    duration_ms: 5,
                    applied: true,
                },
                EngineResult {
                    engine_name: "standard".to_owned(),
                    tokens_before: 800,
                    tokens_after: 600,
                    duration_ms: 7,
                    applied: true,
                },
            ],
            original_tokens: 1_000,
            final_tokens: 600,
            engines_applied: vec!["lite".to_owned(), "standard".to_owned()],
            duration_ms: 12,
            timed_out: false,
            error: false,
            errors: Vec::new(),
            auto_triggered: true,
            auto_trigger_decision: None,
            cache_downgrade_applied: true,
            cache_downgrade: None,
            tool_definitions_tokens_saved: 25,
            tool_definitions_compressed: true,
            tool_definitions_cache_hit: false,
        }
    }

    #[test]
    fn converts_all_pipeline_fields_and_calculations() {
        let result = pipeline_result();
        let stats = CompressionStats::from_pipeline_result(&result, true, "openai", "gpt-4.1");

        assert_eq!(stats.request_id, "request-123");
        assert_eq!(stats.level, CompressionLevel::Standard);
        assert_eq!(stats.engines_applied, ["lite", "standard"]);
        assert_eq!(stats.original_tokens, 1_000);
        assert_eq!(stats.compressed_tokens, 600);
        assert_eq!(stats.tokens_saved(), 400);
        assert_eq!(stats.savings_percent, 40.0);
        assert_eq!(stats.compression_time_ms, 12);
        assert!(stats.auto_triggered);
        assert!(stats.cache_downgrade_applied);
        assert_eq!(stats.tool_definitions_tokens_saved, 25);
        assert!(stats.caveman_applied);
        assert!(!stats.timed_out);
        assert!(!stats.error);
        assert_eq!(stats.provider, "openai");
        assert_eq!(stats.model, "gpt-4.1");
        assert_eq!(stats.engine_results.len(), 2);
        assert_eq!(stats.engine_results[0].tokens_saved, 200);
        assert_eq!(stats.engine_results[0].savings_percent, 20.0);
    }

    #[test]
    fn zero_original_tokens_has_zero_savings() {
        let mut result = pipeline_result();
        result.original_tokens = 0;
        result.final_tokens = 0;
        result.engine_results.clear();
        result.engines_applied.clear();

        let stats = CompressionStats::from_pipeline_result(&result, false, "", "");

        assert_eq!(stats.tokens_saved(), 0);
        assert_eq!(stats.savings_percent, 0.0);
        assert!(stats.savings_percent.is_finite());
    }

    #[test]
    fn inconsistent_increased_counts_saturate_savings_at_zero() {
        let mut result = pipeline_result();
        result.original_tokens = 10;
        result.final_tokens = 20;

        let stats = CompressionStats::from_pipeline_result(&result, false, "", "");

        assert_eq!(stats.tokens_saved(), 0);
        assert_eq!(stats.savings_percent, 0.0);
    }

    #[test]
    fn preserves_timeout_and_error_outcomes_without_error_text() {
        let mut result = pipeline_result();
        result.timed_out = true;
        result.error = true;
        result.errors = vec![CompressionPipelineError::InvalidCustomPipeline {
            name: "RAW_ERROR_NAME".to_owned(),
            reason: "RAW_ERROR_REASON".to_owned(),
        }];

        let stats = CompressionStats::from_pipeline_result(&result, false, "openai", "model");
        let json = stats.to_json().unwrap();

        assert!(stats.timed_out);
        assert!(stats.error);
        assert!(!json.contains("RAW_ERROR_NAME"));
        assert!(!json.contains("RAW_ERROR_REASON"));
    }

    #[test]
    fn serialized_stats_never_include_raw_payload_content() {
        let stats =
            CompressionStats::from_pipeline_result(&pipeline_result(), false, "provider", "model");
        let json = stats.to_json().unwrap();
        let value = stats.to_json_value().unwrap();

        assert!(!json.contains("RAW_CONTENT_MUST_NOT_APPEAR"));
        assert!(value.get("payload").is_none());
        assert!(value.get("content").is_none());
    }

    #[test]
    fn serialized_stats_sanitize_publicly_mutated_metadata() {
        let mut stats =
            CompressionStats::from_pipeline_result(&pipeline_result(), false, "provider", "model");
        stats.request_id = "Bearer post-construction-secret\r\nforged=true".to_owned();
        stats.provider = "https://user:password@example.com".to_owned();
        stats.model = "sk-post-construction-secret".to_owned();
        stats.engines_applied = vec!["AKIAIOSFODNN7EXAMPLE".to_owned()];
        stats.engine_results[0].engine_name = "engine\u{001b}forged".to_owned();

        let json = stats.to_json().unwrap();

        assert!(!json.contains("post-construction-secret"));
        assert!(!json.contains("password"));
        assert!(!json.contains("AKIAIOSFODNN7EXAMPLE"));
        assert!(!json.contains("\\r"));
        assert!(!json.contains("\\n"));
        assert!(!json.contains("\\u001b"));
    }

    #[test]
    fn sanitizes_control_characters_and_malicious_labels() {
        let mut result = pipeline_result();
        result.request_id = "request\r\nforged=true\u{0000}".to_owned();
        result.engines_applied = vec!["lite\nforged=true\u{001b}".to_owned()];
        result.engine_results[0].engine_name = "engine\r\nspoof".to_owned();

        let stats = CompressionStats::from_pipeline_result(
            &result,
            false,
            "provider\r\nadmin=true",
            "model\u{0007}name",
        );
        let json = stats.to_json().unwrap();

        for value in [
            &stats.request_id,
            &stats.provider,
            &stats.model,
            &stats.engines_applied[0],
            &stats.engine_results[0].engine_name,
        ] {
            assert!(!value.chars().any(char::is_control));
        }
        assert!(!json.contains("\\r"));
        assert!(!json.contains("\\n"));
        assert!(!json.contains("\\u0000"));
        assert!(!json.contains("\\u001b"));
    }

    #[test]
    fn redacts_all_supported_secret_shapes() {
        let mut result = pipeline_result();
        result.request_id = "id sk-proj-AbCd_123".to_owned();
        result.engines_applied = vec!["Bearer eyJhbGciOi.secret".to_owned()];
        result.engine_results[0].engine_name = "AKIAIOSFODNN7EXAMPLE".to_owned();

        let stats = CompressionStats::from_pipeline_result(
            &result,
            false,
            "https://user:hunter2@example.com/api",
            "prefix AKIA1234567890ABCDEF suffix",
        );
        let json = stats.to_json().unwrap();

        assert!(!json.contains("sk-proj-AbCd_123"));
        assert!(!json.contains("Bearer"));
        assert!(!json.contains("eyJhbGciOi.secret"));
        assert!(!json.contains("AKIAIOSFODNN7EXAMPLE"));
        assert!(!json.contains("AKIA1234567890ABCDEF"));
        assert!(!json.contains("hunter2"));
        assert!(json.matches(REDACTED).count() >= 5);
    }

    #[test]
    fn bounds_all_operational_metadata_by_utf8_bytes() {
        let mut result = pipeline_result();
        result.request_id = "é".repeat(MAX_REQUEST_ID_LEN);
        result.engines_applied = vec!["é".repeat(MAX_ENGINE_LABEL_LEN)];
        result.engine_results[0].engine_name = "é".repeat(MAX_ENGINE_LABEL_LEN);

        let stats = CompressionStats::from_pipeline_result(
            &result,
            false,
            "é".repeat(MAX_PROVIDER_LEN),
            "é".repeat(MAX_MODEL_LEN),
        );

        assert!(stats.request_id.len() <= MAX_REQUEST_ID_LEN);
        assert!(stats.provider.len() <= MAX_PROVIDER_LEN);
        assert!(stats.model.len() <= MAX_MODEL_LEN);
        assert!(stats.engines_applied[0].len() <= MAX_ENGINE_LABEL_LEN);
        assert!(stats.engine_results[0].engine_name.len() <= MAX_ENGINE_LABEL_LEN);
        assert!(stats.request_id.is_char_boundary(stats.request_id.len()));
    }

    proptest! {
        #![proptest_config(ProptestConfig::with_cases(64))]

        #[test]
        fn property_5_stats_formula_and_consistency(
            original_tokens in any::<u32>(),
            compressed_tokens in any::<u32>(),
        ) {
            let expected_saved = original_tokens.saturating_sub(compressed_tokens);
            let percent = savings_percent(original_tokens, compressed_tokens);

            prop_assert_eq!(expected_saved, original_tokens.saturating_sub(compressed_tokens));
            prop_assert!(percent.is_finite());
            prop_assert!((0.0..=100.0).contains(&percent));
            if original_tokens == 0 || compressed_tokens >= original_tokens {
                prop_assert_eq!(percent, 0.0);
            } else {
                let expected = f64::from(expected_saved) * 100.0 / f64::from(original_tokens);
                prop_assert!((percent - expected).abs() <= f64::EPSILON);
            }
        }
    }
}
