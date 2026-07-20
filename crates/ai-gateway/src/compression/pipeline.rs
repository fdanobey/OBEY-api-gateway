//! Compression pipeline orchestration.

use std::{collections::HashMap, fmt, sync::Arc, time::Instant};

use tokio::task::JoinError;

use super::{
    config::{
        CompressionConfig, EffectiveCompressionConfig, SharedCompressionConfig, KNOWN_ENGINE_NAMES,
    },
    engines::{
        aggressive::AggressiveEngine,
        language_pack::{LanguagePackEngine, LanguagePackLevel},
        lite::LiteEngine,
        perplexity::{PerplexityEngine, PerplexityEngineConfig},
        rtk::RtkEngine,
        standard::StandardEngine,
        tool_def::{ToolDefinitionCompressionReport, ToolDefinitionEngine},
        ultra::UltraEngine,
        CompressiblePayload, CompressionContext, CompressionEngine, CompressionLevel, EngineResult,
    },
    protection::ProtectionScanner,
    token_counter::TokenCounter,
};

const LITE: &str = "lite";
const STANDARD: &str = "standard";
const AGGRESSIVE: &str = "aggressive";
const ULTRA: &str = "ultra";
const RTK: &str = "rtk";
const PERPLEXITY: &str = "perplexity";
const TOOL_DEF: &str = "tool_def";
const LANGUAGE_PACK: &str = "language_pack";

/// Request-scoped values needed by the pipeline and later statistics emission.
#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub struct CompressionRequestMetadata {
    pub request_id: String,
    pub custom_pipeline: Option<String>,
    /// Selects auto-only execution for callers using [`CompressionPipeline::compress`].
    /// Prefer [`CompressionPipeline::compress_auto`] for new call sites.
    pub auto_triggered: bool,
    /// Populated by the pipeline when a cache-aware downgrade is actually applied.
    pub cache_downgrade_applied: bool,
}

/// Why an automatically evaluated request did or did not trigger compression.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum AutoTriggerReason {
    ExplicitlyDisabled,
    LevelNone,
    ThresholdDisabled,
    AtOrBelowThreshold,
    ThresholdExceeded,
}

/// Complete, deterministic auto-trigger decision made before provider dispatch.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct AutoTriggerDecision {
    pub original_tokens: u32,
    pub threshold_tokens: u32,
    pub level: CompressionLevel,
    pub should_compress: bool,
    pub auto_triggered: bool,
    pub reason: AutoTriggerReason,
}

/// Decides whether an auto-only compression invocation should run.
///
/// Explicit pipeline invocations may still run below the threshold. Automatic
/// invocations require an enabled effective config, a non-`none` level, a
/// positive threshold, and an original token count strictly greater than it.
pub fn decide_compression(
    original_tokens: u32,
    effective: &EffectiveCompressionConfig,
) -> AutoTriggerDecision {
    let (should_compress, reason) = if !effective.enabled {
        (false, AutoTriggerReason::ExplicitlyDisabled)
    } else if effective.level == CompressionLevel::None {
        (false, AutoTriggerReason::LevelNone)
    } else if effective.auto_threshold_tokens == 0 {
        (false, AutoTriggerReason::ThresholdDisabled)
    } else if original_tokens > effective.auto_threshold_tokens {
        (true, AutoTriggerReason::ThresholdExceeded)
    } else {
        (false, AutoTriggerReason::AtOrBelowThreshold)
    };

    AutoTriggerDecision {
        original_tokens,
        threshold_tokens: effective.auto_threshold_tokens,
        level: effective.level,
        should_compress,
        auto_triggered: should_compress,
        reason,
    }
}

/// Structured record of the byte-stable treatment applied to a cached prefix.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct CacheDowngradeMetadata {
    pub provider: String,
    pub requested_level: CompressionLevel,
    /// Cached messages are preserved exactly, so their actual level is `none`.
    pub actual_prefix_level: CompressionLevel,
    pub boundary_message_index: usize,
}

/// A recoverable pipeline failure. The returned payload remains safe to forward.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum CompressionPipelineError {
    UnknownCustomPipeline { name: String },
    InvalidCustomPipeline { name: String, reason: String },
    EnginePanicked { engine: String },
    EngineTaskCancelled { engine: String },
}

impl fmt::Display for CompressionPipelineError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::UnknownCustomPipeline { name } => {
                write!(formatter, "unknown custom compression pipeline `{name}`")
            }
            Self::InvalidCustomPipeline { name, reason } => {
                write!(
                    formatter,
                    "invalid custom compression pipeline `{name}`: {reason}"
                )
            }
            Self::EnginePanicked { engine } => {
                write!(formatter, "compression engine `{engine}` panicked")
            }
            Self::EngineTaskCancelled { engine } => {
                write!(
                    formatter,
                    "compression engine task `{engine}` was cancelled"
                )
            }
        }
    }
}

impl std::error::Error for CompressionPipelineError {}

/// Complete pipeline output, including attempted per-engine work and rollback state.
#[derive(Debug, Clone, PartialEq)]
pub struct CompressionPipelineResult {
    pub payload: CompressiblePayload,
    pub request_id: String,
    pub level: CompressionLevel,
    pub engine_results: Vec<EngineResult>,
    pub original_tokens: u32,
    pub final_tokens: u32,
    pub engines_applied: Vec<String>,
    pub duration_ms: u64,
    pub timed_out: bool,
    pub error: bool,
    pub errors: Vec<CompressionPipelineError>,
    pub auto_triggered: bool,
    pub auto_trigger_decision: Option<AutoTriggerDecision>,
    pub cache_downgrade_applied: bool,
    pub cache_downgrade: Option<CacheDowngradeMetadata>,
    pub tool_definitions_tokens_saved: u32,
    pub tool_definitions_compressed: bool,
    pub tool_definitions_cache_hit: bool,
}

impl CompressionPipelineResult {
    #[allow(clippy::too_many_arguments)]
    fn unchanged(
        payload: CompressiblePayload,
        original_tokens: u32,
        level: CompressionLevel,
        metadata: &CompressionRequestMetadata,
        auto_trigger_decision: Option<AutoTriggerDecision>,
        cache_downgrade: Option<CacheDowngradeMetadata>,
        started: Instant,
        errors: Vec<CompressionPipelineError>,
    ) -> Self {
        Self {
            payload,
            request_id: metadata.request_id.clone(),
            level,
            engine_results: Vec::new(),
            original_tokens,
            final_tokens: original_tokens,
            engines_applied: Vec::new(),
            duration_ms: elapsed_millis(started),
            timed_out: false,
            error: !errors.is_empty(),
            errors,
            auto_triggered: auto_trigger_decision.is_some_and(|decision| decision.auto_triggered),
            auto_trigger_decision,
            cache_downgrade_applied: cache_downgrade.is_some(),
            cache_downgrade,
            tool_definitions_tokens_saved: 0,
            tool_definitions_compressed: false,
            tool_definitions_cache_hit: false,
        }
    }
}

#[derive(Clone)]
enum RegisteredEngine {
    General(Arc<dyn CompressionEngine>),
    ToolDefinitions(Arc<ToolDefinitionEngine>),
}

impl RegisteredEngine {
    async fn compress(
        &self,
        payload: &mut CompressiblePayload,
        context: &CompressionContext,
    ) -> (EngineResult, ToolDefinitionCompressionReport) {
        match self {
            Self::General(engine) => (
                engine.compress(payload, context).await,
                ToolDefinitionCompressionReport::default(),
            ),
            Self::ToolDefinitions(engine) => engine.compress_with_report(payload, context).await,
        }
    }
}

/// Ordered compression-engine orchestrator.
///
/// Configuration is snapshotted before execution, so the live `RwLock` is never
/// held while an engine future is awaited.
pub struct CompressionPipeline {
    engines: HashMap<String, RegisteredEngine>,
    token_counter: Arc<TokenCounter>,
    protection_scanner: Arc<ProtectionScanner>,
    config: SharedCompressionConfig,
}

impl fmt::Debug for CompressionPipeline {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        let mut engine_names = self.engines.keys().cloned().collect::<Vec<_>>();
        engine_names.sort();
        formatter
            .debug_struct("CompressionPipeline")
            .field("engines", &engine_names)
            .field("token_counter", &"TokenCounter")
            .field("protection_scanner", &self.protection_scanner)
            .finish_non_exhaustive()
    }
}

impl CompressionPipeline {
    /// Builds a production pipeline from a validated, hot-reloadable config.
    pub async fn new(config: SharedCompressionConfig) -> Self {
        let snapshot = { config.read().await.clone() };
        Self::from_shared_config(config, &snapshot)
    }

    /// Builds a standalone pipeline and wraps the supplied config for live use.
    pub fn from_config(config: CompressionConfig) -> Self {
        let shared = Arc::new(tokio::sync::RwLock::new(config.clone()));
        Self::from_shared_config(shared, &config)
    }

    /// Builds a pipeline with injected engines for deterministic orchestration tests.
    pub fn with_engines(
        config: SharedCompressionConfig,
        token_counter: Arc<TokenCounter>,
        protection_scanner: Arc<ProtectionScanner>,
        engines: HashMap<String, Arc<dyn CompressionEngine>>,
    ) -> Self {
        Self {
            engines: engines
                .into_iter()
                .map(|(name, engine)| (name, RegisteredEngine::General(engine)))
                .collect(),
            token_counter,
            protection_scanner,
            config,
        }
    }

    fn from_shared_config(config: SharedCompressionConfig, snapshot: &CompressionConfig) -> Self {
        let token_counter = Arc::new(TokenCounter::new());
        let protection_scanner = Arc::new(ProtectionScanner::default());
        let perplexity_config = PerplexityEngineConfig::try_from(&snapshot.perplexity)
            .unwrap_or_else(|_| PerplexityEngineConfig::default());
        let tool_definitions = Arc::new(ToolDefinitionEngine::default());
        let mut engines = HashMap::new();

        insert_engine(&mut engines, LITE, Arc::new(LiteEngine::new()));
        insert_engine(&mut engines, STANDARD, Arc::new(StandardEngine::new()));
        insert_engine(&mut engines, AGGRESSIVE, Arc::new(AggressiveEngine::new()));
        insert_engine(&mut engines, ULTRA, Arc::new(UltraEngine::new()));
        insert_engine(
            &mut engines,
            RTK,
            Arc::new(RtkEngine::from_config(&snapshot.rtk)),
        );
        insert_engine(
            &mut engines,
            PERPLEXITY,
            Arc::new(PerplexityEngine::heuristic_fallback(perplexity_config)),
        );
        insert_engine(
            &mut engines,
            LANGUAGE_PACK,
            Arc::new(LanguagePackEngine::from_config(
                LanguagePackLevel::Light,
                Some(snapshot.language.clone()),
                &snapshot.language_packs_dir,
            )),
        );
        engines.insert(
            TOOL_DEF.to_owned(),
            RegisteredEngine::ToolDefinitions(tool_definitions),
        );

        Self {
            engines,
            token_counter,
            protection_scanner,
            config,
        }
    }

    /// Returns the exact fixed chain for a named compression level.
    pub fn engine_names_for_level(level: CompressionLevel) -> &'static [&'static str] {
        match level {
            CompressionLevel::None => &[],
            CompressionLevel::Lite => &[LITE],
            CompressionLevel::Standard => &[LITE, STANDARD],
            CompressionLevel::Aggressive => &[LITE, STANDARD, AGGRESSIVE],
            CompressionLevel::Ultra => &[LITE, STANDARD, AGGRESSIVE, ULTRA],
            CompressionLevel::Rtk => &[RTK],
            CompressionLevel::Stacked => &[RTK, STANDARD],
        }
    }

    /// Resolves the shared engine instances for a fixed named level.
    pub fn resolve_engines(&self, level: CompressionLevel) -> Vec<Arc<dyn CompressionEngine>> {
        Self::engine_names_for_level(level)
            .iter()
            .filter_map(|name| match self.engines.get(*name) {
                Some(RegisteredEngine::General(engine)) => Some(Arc::clone(engine)),
                Some(RegisteredEngine::ToolDefinitions(engine)) => {
                    let engine: Arc<dyn CompressionEngine> = engine.clone();
                    Some(engine)
                }
                None => None,
            })
            .collect()
    }

    /// Compresses an explicitly enabled outgoing payload before upstream dispatch.
    ///
    /// Explicit execution ignores `auto_threshold_tokens`; it only requires an
    /// enabled effective configuration and a non-`none` level.
    pub async fn compress_explicit(
        &self,
        payload: CompressiblePayload,
        context: CompressionContext,
        effective: EffectiveCompressionConfig,
        mut metadata: CompressionRequestMetadata,
    ) -> CompressionPipelineResult {
        metadata.auto_triggered = false;
        self.compress_with_mode(payload, context, effective, metadata, false)
            .await
    }

    /// Counts tokens and applies compression only when auto-trigger rules pass.
    pub async fn compress_auto(
        &self,
        payload: CompressiblePayload,
        context: CompressionContext,
        effective: EffectiveCompressionConfig,
        mut metadata: CompressionRequestMetadata,
    ) -> CompressionPipelineResult {
        metadata.auto_triggered = true;
        self.compress_with_mode(payload, context, effective, metadata, true)
            .await
    }

    /// Compresses an outgoing payload fully before its caller dispatches upstream.
    ///
    /// `metadata.auto_triggered` selects auto-only threshold behavior for backwards
    /// compatibility. New call sites should prefer `compress_explicit` or
    /// `compress_auto` so intent is unambiguous.
    pub async fn compress(
        &self,
        payload: CompressiblePayload,
        context: CompressionContext,
        effective: EffectiveCompressionConfig,
        metadata: CompressionRequestMetadata,
    ) -> CompressionPipelineResult {
        let auto_only = metadata.auto_triggered;
        self.compress_with_mode(payload, context, effective, metadata, auto_only)
            .await
    }

    async fn compress_with_mode(
        &self,
        payload: CompressiblePayload,
        mut context: CompressionContext,
        effective: EffectiveCompressionConfig,
        metadata: CompressionRequestMetadata,
        auto_only: bool,
    ) -> CompressionPipelineResult {
        let started = Instant::now();
        let mut original = payload;
        original.refresh_metadata();
        let original_tokens = count_tokens(&self.token_counter, &original);
        let auto_trigger_decision =
            auto_only.then(|| decide_compression(original_tokens, &effective));
        let config = { self.config.read().await.clone() };

        if let Some(decision) = auto_trigger_decision {
            tracing::debug!(
                request_id = %metadata.request_id,
                provider = %context.provider_name,
                original_tokens = decision.original_tokens,
                threshold_tokens = decision.threshold_tokens,
                level = ?decision.level,
                enabled = effective.enabled,
                triggered = decision.auto_triggered,
                reason = ?decision.reason,
                "Evaluated automatic compression trigger"
            );
            if !decision.should_compress {
                return CompressionPipelineResult::unchanged(
                    original,
                    original_tokens,
                    effective.level,
                    &metadata,
                    auto_trigger_decision,
                    None,
                    started,
                    Vec::new(),
                );
            }
        } else if !effective.enabled || effective.level == CompressionLevel::None {
            return CompressionPipelineResult::unchanged(
                original,
                original_tokens,
                effective.level,
                &metadata,
                None,
                None,
                started,
                Vec::new(),
            );
        }

        let cache_downgrade = cache_downgrade_for(&original, &context, effective.level);
        if let Some(downgrade) = cache_downgrade.as_ref() {
            tracing::info!(
                request_id = %metadata.request_id,
                provider = %downgrade.provider,
                requested_level = ?downgrade.requested_level,
                actual_prefix_level = ?downgrade.actual_prefix_level,
                boundary_message_index = downgrade.boundary_message_index,
                cache_downgrade_applied = true,
                "Preserving cached prompt prefix byte-for-byte"
            );
        }

        let mut engine_names = match self.resolve_engine_names(&config, effective.level, &metadata)
        {
            Ok(names) => names,
            Err(error) => {
                return CompressionPipelineResult::unchanged(
                    original,
                    original_tokens,
                    effective.level,
                    &metadata,
                    auto_trigger_decision,
                    cache_downgrade,
                    started,
                    vec![error],
                );
            }
        };

        if config.compress_tool_definitions
            && !engine_names.is_empty()
            && !engine_names.iter().any(|name| name == TOOL_DEF)
        {
            engine_names.insert(0, TOOL_DEF.to_owned());
        }

        context.token_counter = Arc::clone(&self.token_counter);
        context.protection_scanner = Arc::clone(&self.protection_scanner);
        context.language.clone_from(&config.language);

        let budget_level = if effective.level == CompressionLevel::None {
            CompressionLevel::Stacked
        } else {
            effective.level
        };
        let budget_ms = config.time_budget_ms.for_level(budget_level);
        let deadline = budget_ms.map(|milliseconds| {
            tokio::time::Instant::now() + std::time::Duration::from_millis(milliseconds)
        });
        let mut current = original.clone();
        let mut current_tokens = original_tokens;
        let mut engine_results = Vec::new();
        let mut engines_applied = Vec::new();
        let mut errors = Vec::new();
        let mut tool_definitions_tokens_saved = 0u32;
        let mut tool_definitions_compressed = false;
        let mut tool_definitions_cache_hit = false;

        for engine_name in engine_names {
            let Some(engine) = self.engines.get(&engine_name).cloned() else {
                errors.push(CompressionPipelineError::InvalidCustomPipeline {
                    name: metadata
                        .custom_pipeline
                        .clone()
                        .unwrap_or_else(|| format!("{budget_level:?}").to_ascii_lowercase()),
                    reason: format!("engine `{engine_name}` is unavailable"),
                });
                continue;
            };
            let before = current.clone();
            let before_tokens = current_tokens;
            let engine_context = context.clone();
            let task_engine_name = engine_name.clone();
            let engine_started = Instant::now();
            let mut task = tokio::spawn(async move {
                let mut candidate = before;
                let (result, tool_report) = engine.compress(&mut candidate, &engine_context).await;
                (candidate, result, tool_report)
            });

            let task_result = if let Some(deadline) = deadline {
                let remaining = deadline.saturating_duration_since(tokio::time::Instant::now());
                match tokio::time::timeout(remaining, &mut task).await {
                    Ok(result) => result,
                    Err(_) => {
                        task.abort();
                        let _ = task.await;
                        tracing::warn!(
                            request_id = %metadata.request_id,
                            engine = %task_engine_name,
                            budget_ms = budget_ms.unwrap_or_default(),
                            "Compression time budget exceeded; forwarding original payload"
                        );
                        return CompressionPipelineResult {
                            payload: original,
                            request_id: metadata.request_id,
                            level: effective.level,
                            engine_results,
                            original_tokens,
                            final_tokens: original_tokens,
                            engines_applied,
                            duration_ms: elapsed_millis(started),
                            timed_out: true,
                            error: false,
                            errors,
                            auto_triggered: auto_trigger_decision
                                .is_some_and(|decision| decision.auto_triggered),
                            auto_trigger_decision,
                            cache_downgrade_applied: cache_downgrade.is_some(),
                            cache_downgrade,
                            tool_definitions_tokens_saved,
                            tool_definitions_compressed,
                            tool_definitions_cache_hit,
                        };
                    }
                }
            } else {
                task.await
            };

            let (candidate, _reported, tool_report) = match task_result {
                Ok(completed) => completed,
                Err(join_error) => {
                    let error = join_error_to_pipeline_error(&engine_name, &join_error);
                    tracing::warn!(
                        request_id = %metadata.request_id,
                        engine = %engine_name,
                        error = %error,
                        "Compression engine failed; continuing from its pre-engine payload"
                    );
                    errors.push(error);
                    continue;
                }
            };

            let candidate_tokens = count_tokens(&self.token_counter, &candidate);
            let accepted = candidate_tokens <= before_tokens;
            let applied = accepted && candidate != current;
            let tokens_after = if accepted {
                candidate_tokens
            } else {
                before_tokens
            };
            let normalized = EngineResult {
                engine_name: engine_name.clone(),
                tokens_before: before_tokens,
                tokens_after,
                duration_ms: elapsed_millis(engine_started),
                applied,
            };

            if accepted {
                current = candidate;
                current_tokens = candidate_tokens;
                if engine_name == TOOL_DEF {
                    tool_definitions_tokens_saved = tool_definitions_tokens_saved
                        .saturating_add(tool_report.tool_definitions_tokens_saved);
                    tool_definitions_compressed |= applied;
                    tool_definitions_cache_hit |= tool_report.cache_hit;
                }
            }
            if applied {
                engines_applied.push(engine_name);
            }
            engine_results.push(normalized);

            if context
                .target_token_budget
                .is_some_and(|target| current_tokens <= target)
            {
                break;
            }
        }

        if current_tokens > original_tokens {
            current = original;
            current_tokens = original_tokens;
            engines_applied.clear();
            tool_definitions_tokens_saved = 0;
            tool_definitions_compressed = false;
            tool_definitions_cache_hit = false;
        }

        let savings_tokens = original_tokens.saturating_sub(current_tokens);
        let savings_percent = if original_tokens == 0 {
            0.0
        } else {
            f64::from(savings_tokens) * 100.0 / f64::from(original_tokens)
        };
        tracing::info!(
            request_id = %metadata.request_id,
            provider = %context.provider_name,
            model = %context.model,
            level = ?effective.level,
            original_tokens,
            final_tokens = current_tokens,
            savings_tokens,
            savings_percent,
            threshold_tokens = effective.auto_threshold_tokens,
            auto_triggered = auto_trigger_decision
                .is_some_and(|decision| decision.auto_triggered),
            cache_downgrade_applied = cache_downgrade.is_some(),
            "Compression pipeline completed"
        );

        CompressionPipelineResult {
            payload: current,
            request_id: metadata.request_id,
            level: effective.level,
            engine_results,
            original_tokens,
            final_tokens: current_tokens,
            engines_applied,
            duration_ms: elapsed_millis(started),
            timed_out: false,
            error: !errors.is_empty(),
            errors,
            auto_triggered: auto_trigger_decision.is_some_and(|decision| decision.auto_triggered),
            auto_trigger_decision,
            cache_downgrade_applied: cache_downgrade.is_some(),
            cache_downgrade,
            tool_definitions_tokens_saved,
            tool_definitions_compressed,
            tool_definitions_cache_hit,
        }
    }

    fn resolve_engine_names(
        &self,
        config: &CompressionConfig,
        level: CompressionLevel,
        metadata: &CompressionRequestMetadata,
    ) -> Result<Vec<String>, CompressionPipelineError> {
        let Some(custom_name) = metadata.custom_pipeline.as_deref() else {
            return Ok(Self::engine_names_for_level(level)
                .iter()
                .map(|name| (*name).to_owned())
                .collect());
        };

        let custom = config.custom_pipelines.get(custom_name).ok_or_else(|| {
            CompressionPipelineError::UnknownCustomPipeline {
                name: custom_name.to_owned(),
            }
        })?;
        if let Err(config_errors) = config.validate() {
            let reason = config_errors
                .into_iter()
                .map(|error| error.to_string())
                .collect::<Vec<_>>()
                .join("; ");
            return Err(CompressionPipelineError::InvalidCustomPipeline {
                name: custom_name.to_owned(),
                reason,
            });
        }
        if custom.engines.is_empty() {
            return Err(CompressionPipelineError::InvalidCustomPipeline {
                name: custom_name.to_owned(),
                reason: "engine list must not be empty".to_owned(),
            });
        }
        for engine in &custom.engines {
            if !KNOWN_ENGINE_NAMES.contains(&engine.as_str()) || !self.engines.contains_key(engine)
            {
                return Err(CompressionPipelineError::InvalidCustomPipeline {
                    name: custom_name.to_owned(),
                    reason: format!("engine `{engine}` is unknown or unavailable"),
                });
            }
        }

        Ok(custom.engines.clone())
    }
}

fn insert_engine<T>(engines: &mut HashMap<String, RegisteredEngine>, name: &str, engine: Arc<T>)
where
    T: CompressionEngine + 'static,
{
    engines.insert(name.to_owned(), RegisteredEngine::General(engine));
}

fn count_tokens(counter: &TokenCounter, payload: &CompressiblePayload) -> u32 {
    counter.count_request(&payload.clone().into_openai_request())
}

fn cache_downgrade_for(
    payload: &CompressiblePayload,
    context: &CompressionContext,
    requested_level: CompressionLevel,
) -> Option<CacheDowngradeMetadata> {
    if !context.prompt_caching_enabled || !requires_cache_downgrade(requested_level) {
        return None;
    }

    let boundary_message_index = payload
        .messages
        .iter()
        .rposition(|message| message.cache_protected)?;
    Some(CacheDowngradeMetadata {
        provider: context.provider_name.clone(),
        requested_level,
        actual_prefix_level: CompressionLevel::None,
        boundary_message_index,
    })
}

fn requires_cache_downgrade(level: CompressionLevel) -> bool {
    matches!(
        level,
        CompressionLevel::Aggressive
            | CompressionLevel::Ultra
            | CompressionLevel::Rtk
            | CompressionLevel::Stacked
    )
}

fn elapsed_millis(started: Instant) -> u64 {
    started.elapsed().as_millis().min(u64::MAX as u128) as u64
}

fn join_error_to_pipeline_error(engine: &str, error: &JoinError) -> CompressionPipelineError {
    if error.is_panic() {
        CompressionPipelineError::EnginePanicked {
            engine: engine.to_owned(),
        }
    } else {
        CompressionPipelineError::EngineTaskCancelled {
            engine: engine.to_owned(),
        }
    }
}

#[cfg(test)]
mod tests {
    use std::sync::Mutex;

    use async_trait::async_trait;
    use serde_json::{json, Value};

    use super::*;
    use crate::compression::config::{CustomPipelineConfig, TimeBudgetConfig};
    use crate::models::openai::OpenAIRequest;

    #[derive(Clone)]
    enum TestAction {
        Replace(String),
        Append(String),
        Sleep(std::time::Duration),
        Panic,
        Noop,
    }

    struct TestEngine {
        name: String,
        action: TestAction,
        calls: Arc<Mutex<Vec<String>>>,
    }

    impl TestEngine {
        fn new(name: &str, action: TestAction, calls: Arc<Mutex<Vec<String>>>) -> Self {
            Self {
                name: name.to_owned(),
                action,
                calls,
            }
        }
    }

    #[async_trait]
    impl CompressionEngine for TestEngine {
        fn name(&self) -> &str {
            &self.name
        }

        async fn compress(
            &self,
            payload: &mut CompressiblePayload,
            context: &CompressionContext,
        ) -> EngineResult {
            self.calls.lock().unwrap().push(self.name.clone());
            let before = count_tokens(&context.token_counter, payload);
            match &self.action {
                TestAction::Replace(text) => set_first_content(payload, text.clone()),
                TestAction::Append(text) => {
                    if let Some(Value::String(content)) = payload
                        .messages
                        .first_mut()
                        .map(|message| message.content.as_value_mut())
                    {
                        content.push_str(text);
                    }
                }
                TestAction::Sleep(duration) => tokio::time::sleep(*duration).await,
                TestAction::Panic => panic!("injected engine panic"),
                TestAction::Noop => {}
            }
            let after = count_tokens(&context.token_counter, payload);
            EngineResult {
                engine_name: self.name.clone(),
                tokens_before: before,
                tokens_after: after,
                duration_ms: 0,
                applied: before != after,
            }
        }
    }

    fn set_first_content(payload: &mut CompressiblePayload, text: String) {
        if let Some(message) = payload.messages.first_mut() {
            *message.content.as_value_mut() = Value::String(text);
        }
    }

    fn payload() -> CompressiblePayload {
        let request: OpenAIRequest = serde_json::from_value(json!({
            "model": "gpt-4",
            "messages": [{
                "role": "user",
                "content": "This is a deliberately long payload with many repeated words and details for compression testing. ".repeat(20)
            }]
        }))
        .unwrap();
        request.into()
    }

    fn tool_payload() -> CompressiblePayload {
        let request: OpenAIRequest = serde_json::from_value(json!({
            "model": "gpt-4",
            "messages": [{"role": "user", "content": "Use the lookup tool."}],
            "tools": [{
                "type": "function",
                "function": {
                    "name": "lookup",
                    "description": "The purpose of this tool is to look up a record. For example, use it when a record identifier is available. Note: this is a deliberately verbose caveat.",
                    "parameters": {"type": "object", "properties": {"id": {"type": "string"}}}
                }
            }]
        }))
        .unwrap();
        request.into()
    }

    fn effective(level: CompressionLevel) -> EffectiveCompressionConfig {
        EffectiveCompressionConfig {
            enabled: true,
            level,
            auto_threshold_tokens: 0,
            caveman_output: false,
        }
    }

    fn context(target: Option<u32>) -> CompressionContext {
        CompressionContext {
            model: "gpt-4".to_owned(),
            target_token_budget: target,
            ..CompressionContext::default()
        }
    }

    fn caching_context(provider: &str) -> CompressionContext {
        CompressionContext {
            model: "claude-test".to_owned(),
            provider_name: provider.to_owned(),
            prompt_caching_enabled: true,
            ..CompressionContext::default()
        }
    }

    fn cached_payload() -> CompressiblePayload {
        let suffix = (0..80)
            .map(|index| format!("Compiling cached_suffix_crate_{index}   v1.0.0"))
            .chain(std::iter::once("Finished   release   target(s)".to_owned()))
            .collect::<Vec<_>>()
            .join("\n");
        let request: OpenAIRequest = serde_json::from_value(json!({
            "model": "claude-test",
            "messages": [
                {"role": "user", "content": "cached prefix   must remain exactly stable"},
                {
                    "role": "assistant",
                    "content": [{
                        "type": "text",
                        "text": "cache boundary   remains exact",
                        "cache_control": {"type": "ephemeral"}
                    }]
                },
                {
                    "role": "tool",
                    "content": suffix,
                    "command": "cargo build --release"
                }
            ]
        }))
        .unwrap();
        request.into()
    }

    fn injected_pipeline(
        config: CompressionConfig,
        engines: Vec<(&str, Arc<dyn CompressionEngine>)>,
    ) -> CompressionPipeline {
        CompressionPipeline::with_engines(
            Arc::new(tokio::sync::RwLock::new(config)),
            Arc::new(TokenCounter::new()),
            Arc::new(ProtectionScanner::default()),
            engines
                .into_iter()
                .map(|(name, engine)| (name.to_owned(), engine))
                .collect(),
        )
    }

    #[test]
    fn trigger_decision_uses_strict_threshold_and_hard_disables() {
        let enabled = EffectiveCompressionConfig {
            enabled: true,
            level: CompressionLevel::Lite,
            auto_threshold_tokens: 100,
            caveman_output: false,
        };
        assert_eq!(
            decide_compression(100, &enabled).reason,
            AutoTriggerReason::AtOrBelowThreshold
        );
        assert!(!decide_compression(100, &enabled).should_compress);
        assert!(decide_compression(101, &enabled).should_compress);

        let none = EffectiveCompressionConfig {
            level: CompressionLevel::None,
            ..enabled
        };
        assert_eq!(
            decide_compression(u32::MAX, &none).reason,
            AutoTriggerReason::LevelNone
        );
        let disabled = EffectiveCompressionConfig {
            enabled: false,
            ..enabled
        };
        assert_eq!(
            decide_compression(u32::MAX, &disabled).reason,
            AutoTriggerReason::ExplicitlyDisabled
        );
    }

    #[tokio::test]
    async fn explicit_execution_ignores_auto_threshold_but_auto_execution_does_not() {
        let calls = Arc::new(Mutex::new(Vec::new()));
        let pipeline = injected_pipeline(
            CompressionConfig::default(),
            vec![(
                "lite",
                Arc::new(TestEngine::new(
                    "lite",
                    TestAction::Replace("short".to_owned()),
                    Arc::clone(&calls),
                )),
            )],
        );
        let original = payload();
        let original_tokens = count_tokens(&TokenCounter::new(), &original);
        let effective = EffectiveCompressionConfig {
            enabled: true,
            level: CompressionLevel::Lite,
            auto_threshold_tokens: original_tokens,
            caveman_output: false,
        };

        let auto = pipeline
            .compress_auto(
                original.clone(),
                context(None),
                effective,
                CompressionRequestMetadata::default(),
            )
            .await;
        assert_eq!(auto.payload, original);
        assert!(!auto.auto_triggered);
        assert!(calls.lock().unwrap().is_empty());

        let explicit = pipeline
            .compress_explicit(
                original,
                context(None),
                effective,
                CompressionRequestMetadata::default(),
            )
            .await;
        assert_eq!(*calls.lock().unwrap(), ["lite"]);
        assert!(explicit.final_tokens < explicit.original_tokens);
    }

    #[tokio::test]
    async fn cached_prefix_is_byte_stable_at_every_level_and_suffix_can_compress() {
        let pipeline = CompressionPipeline::from_config(CompressionConfig::default());
        let original = cached_payload();
        let prefix = original.messages[..=1].to_vec();

        for level in [
            CompressionLevel::Lite,
            CompressionLevel::Standard,
            CompressionLevel::Aggressive,
            CompressionLevel::Ultra,
            CompressionLevel::Rtk,
            CompressionLevel::Stacked,
        ] {
            let result = pipeline
                .compress_explicit(
                    original.clone(),
                    caching_context("anthropic"),
                    effective(level),
                    CompressionRequestMetadata::default(),
                )
                .await;

            assert_eq!(result.payload.messages[..=1], prefix);
            if requires_cache_downgrade(level) {
                assert!(result.cache_downgrade_applied);
                assert_eq!(
                    result.cache_downgrade,
                    Some(CacheDowngradeMetadata {
                        provider: "anthropic".to_owned(),
                        requested_level: level,
                        actual_prefix_level: CompressionLevel::None,
                        boundary_message_index: 1,
                    })
                );
            } else {
                assert!(!result.cache_downgrade_applied);
                assert!(result.cache_downgrade.is_none());
            }
        }

        let suffix_result = pipeline
            .compress_explicit(
                original.clone(),
                caching_context("anthropic"),
                effective(CompressionLevel::Rtk),
                CompressionRequestMetadata::default(),
            )
            .await;
        assert_ne!(suffix_result.payload.messages[2], original.messages[2]);
    }

    #[tokio::test]
    async fn cache_marker_without_provider_support_has_no_downgrade_event() {
        let pipeline = CompressionPipeline::from_config(CompressionConfig::default());
        let result = pipeline
            .compress_explicit(
                cached_payload(),
                CompressionContext::new("claude-test", "custom"),
                effective(CompressionLevel::Aggressive),
                CompressionRequestMetadata::default(),
            )
            .await;

        assert!(!result.cache_downgrade_applied);
        assert!(result.cache_downgrade.is_none());
    }

    #[test]
    fn named_levels_resolve_to_exact_fixed_chains() {
        assert_eq!(
            CompressionPipeline::engine_names_for_level(CompressionLevel::Lite),
            ["lite"]
        );
        assert_eq!(
            CompressionPipeline::engine_names_for_level(CompressionLevel::Standard),
            ["lite", "standard"]
        );
        assert_eq!(
            CompressionPipeline::engine_names_for_level(CompressionLevel::Aggressive),
            ["lite", "standard", "aggressive"]
        );
        assert_eq!(
            CompressionPipeline::engine_names_for_level(CompressionLevel::Ultra),
            ["lite", "standard", "aggressive", "ultra"]
        );
        assert_eq!(
            CompressionPipeline::engine_names_for_level(CompressionLevel::Rtk),
            ["rtk"]
        );
        assert_eq!(
            CompressionPipeline::engine_names_for_level(CompressionLevel::Stacked),
            ["rtk", "standard"]
        );
        assert!(CompressionPipeline::engine_names_for_level(CompressionLevel::None).is_empty());
    }

    #[tokio::test]
    async fn custom_pipeline_preserves_order_and_cumulative_math() {
        let calls = Arc::new(Mutex::new(Vec::new()));
        let mut config = CompressionConfig::default();
        config.enabled = true;
        config.custom_pipelines.insert(
            "ordered".to_owned(),
            CustomPipelineConfig {
                engines: vec!["rtk".to_owned(), "standard".to_owned()],
            },
        );
        let pipeline = injected_pipeline(
            config,
            vec![
                (
                    "rtk",
                    Arc::new(TestEngine::new(
                        "rtk",
                        TestAction::Replace("short payload".to_owned()),
                        Arc::clone(&calls),
                    )),
                ),
                (
                    "standard",
                    Arc::new(TestEngine::new(
                        "standard",
                        TestAction::Replace("short".to_owned()),
                        Arc::clone(&calls),
                    )),
                ),
            ],
        );
        let result = pipeline
            .compress(
                payload(),
                context(None),
                effective(CompressionLevel::Stacked),
                CompressionRequestMetadata {
                    custom_pipeline: Some("ordered".to_owned()),
                    ..CompressionRequestMetadata::default()
                },
            )
            .await;

        assert_eq!(*calls.lock().unwrap(), ["rtk", "standard"]);
        assert_eq!(result.engine_results.len(), 2);
        assert_eq!(
            result.engine_results[0].tokens_after,
            result.engine_results[1].tokens_before
        );
        assert_eq!(result.final_tokens, result.engine_results[1].tokens_after);
        assert!(result.final_tokens <= result.original_tokens);
    }

    #[tokio::test]
    async fn target_budget_short_circuits_remaining_engines() {
        let calls = Arc::new(Mutex::new(Vec::new()));
        let first = Arc::new(TestEngine::new(
            "lite",
            TestAction::Replace("tiny".to_owned()),
            Arc::clone(&calls),
        ));
        let second = Arc::new(TestEngine::new(
            "standard",
            TestAction::Noop,
            Arc::clone(&calls),
        ));
        let pipeline = injected_pipeline(
            CompressionConfig::default(),
            vec![("lite", first), ("standard", second)],
        );
        let result = pipeline
            .compress(
                payload(),
                context(Some(100)),
                effective(CompressionLevel::Standard),
                CompressionRequestMetadata::default(),
            )
            .await;

        assert_eq!(*calls.lock().unwrap(), ["lite"]);
        assert_eq!(result.engine_results.len(), 1);
    }

    #[tokio::test]
    async fn timeout_returns_exact_original_payload() {
        let calls = Arc::new(Mutex::new(Vec::new()));
        let mut config = CompressionConfig::default();
        config.time_budget_ms = TimeBudgetConfig {
            lite: 10,
            ..TimeBudgetConfig::default()
        };
        let pipeline = injected_pipeline(
            config,
            vec![(
                "lite",
                Arc::new(TestEngine::new(
                    "lite",
                    TestAction::Sleep(std::time::Duration::from_millis(100)),
                    calls,
                )),
            )],
        );
        let original = payload();
        let result = pipeline
            .compress(
                original.clone(),
                context(None),
                effective(CompressionLevel::Lite),
                CompressionRequestMetadata::default(),
            )
            .await;

        assert!(result.timed_out);
        assert_eq!(result.payload, original);
        assert_eq!(result.final_tokens, result.original_tokens);
    }

    #[tokio::test]
    async fn panic_restores_pre_engine_payload_and_continues() {
        let calls = Arc::new(Mutex::new(Vec::new()));
        let mut config = CompressionConfig::default();
        config.custom_pipelines.insert(
            "recover".to_owned(),
            CustomPipelineConfig {
                engines: vec![
                    "lite".to_owned(),
                    "standard".to_owned(),
                    "aggressive".to_owned(),
                ],
            },
        );
        let pipeline = injected_pipeline(
            config,
            vec![
                (
                    "lite",
                    Arc::new(TestEngine::new(
                        "lite",
                        TestAction::Replace("first compressed payload".to_owned()),
                        Arc::clone(&calls),
                    )),
                ),
                (
                    "standard",
                    Arc::new(TestEngine::new(
                        "standard",
                        TestAction::Panic,
                        Arc::clone(&calls),
                    )),
                ),
                (
                    "aggressive",
                    Arc::new(TestEngine::new(
                        "aggressive",
                        TestAction::Replace("final".to_owned()),
                        Arc::clone(&calls),
                    )),
                ),
            ],
        );
        let result = pipeline
            .compress(
                payload(),
                context(None),
                effective(CompressionLevel::Aggressive),
                CompressionRequestMetadata {
                    custom_pipeline: Some("recover".to_owned()),
                    ..CompressionRequestMetadata::default()
                },
            )
            .await;

        assert_eq!(*calls.lock().unwrap(), ["lite", "standard", "aggressive"]);
        assert!(result.error);
        assert!(matches!(
            result.errors.as_slice(),
            [CompressionPipelineError::EnginePanicked { engine }] if engine == "standard"
        ));
        assert_eq!(result.engines_applied, ["lite", "aggressive"]);
    }

    #[tokio::test]
    async fn invalid_custom_pipeline_is_rejected_before_execution() {
        let calls = Arc::new(Mutex::new(Vec::new()));
        let mut config = CompressionConfig::default();
        config.custom_pipelines.insert(
            "invalid".to_owned(),
            CustomPipelineConfig {
                engines: vec!["invented".to_owned()],
            },
        );
        let pipeline = injected_pipeline(
            config,
            vec![(
                "lite",
                Arc::new(TestEngine::new(
                    "lite",
                    TestAction::Noop,
                    Arc::clone(&calls),
                )),
            )],
        );
        let original = payload();
        let result = pipeline
            .compress(
                original.clone(),
                context(None),
                effective(CompressionLevel::Lite),
                CompressionRequestMetadata {
                    custom_pipeline: Some("invalid".to_owned()),
                    ..CompressionRequestMetadata::default()
                },
            )
            .await;

        assert!(result.error);
        assert!(calls.lock().unwrap().is_empty());
        assert_eq!(result.payload, original);
    }

    #[tokio::test]
    async fn none_executes_no_engines() {
        let calls = Arc::new(Mutex::new(Vec::new()));
        let pipeline = injected_pipeline(
            CompressionConfig::default(),
            vec![(
                "lite",
                Arc::new(TestEngine::new(
                    "lite",
                    TestAction::Noop,
                    Arc::clone(&calls),
                )),
            )],
        );
        let original = payload();
        let result = pipeline
            .compress(
                original.clone(),
                context(None),
                effective(CompressionLevel::None),
                CompressionRequestMetadata::default(),
            )
            .await;

        assert!(calls.lock().unwrap().is_empty());
        assert_eq!(result.payload, original);
        assert!(result.engine_results.is_empty());
    }

    #[tokio::test]
    async fn token_increasing_engine_is_rolled_back_exactly() {
        let calls = Arc::new(Mutex::new(Vec::new()));
        let pipeline = injected_pipeline(
            CompressionConfig::default(),
            vec![(
                "lite",
                Arc::new(TestEngine::new(
                    "lite",
                    TestAction::Append(" expanded content".repeat(1_000)),
                    calls,
                )),
            )],
        );
        let original = payload();
        let result = pipeline
            .compress(
                original.clone(),
                context(None),
                effective(CompressionLevel::Lite),
                CompressionRequestMetadata::default(),
            )
            .await;

        assert_eq!(result.payload, original);
        assert_eq!(result.final_tokens, result.original_tokens);
        assert!(result.engines_applied.is_empty());
        assert_eq!(
            result.engine_results[0].tokens_before,
            result.engine_results[0].tokens_after
        );
        assert!(!result.engine_results[0].applied);
    }

    #[tokio::test]
    async fn configured_tool_auxiliary_reports_savings_and_cache_hits() {
        let mut config = CompressionConfig::default();
        config.enabled = true;
        config.compress_tool_definitions = true;
        let pipeline = CompressionPipeline::from_config(config);
        let original = tool_payload();
        let first = pipeline
            .compress(
                original.clone(),
                context(None),
                effective(CompressionLevel::Lite),
                CompressionRequestMetadata::default(),
            )
            .await;
        let second = pipeline
            .compress(
                original,
                context(None),
                effective(CompressionLevel::Lite),
                CompressionRequestMetadata::default(),
            )
            .await;

        assert!(first.tool_definitions_compressed);
        assert!(first.tool_definitions_tokens_saved > 0);
        assert!(!first.tool_definitions_cache_hit);
        assert!(second.tool_definitions_cache_hit);
        assert_eq!(first.engine_results[0].engine_name, "tool_def");
    }
}
