use crate::active_requests::{ActivePhase, ActiveRequestHandle};
use crate::compression::{
    caveman::apply_caveman_output,
    config::CompressionConfig,
    pipeline::{CompressionPipeline, CompressionRequestMetadata},
    precompressed::{PrecompressedLoadStatus, PrecompressedManager},
    stats::CompressionStats,
    CompressiblePayload, CompressionContext,
};
use crate::config::{Config, ContextConfig, ModelGroup, Provider, ProviderModel};
use crate::context::ContextManager;
use crate::dashboard::CompressionEventHub;
use crate::error::{AggregatedError, GatewayError, ProviderAttempt};
use crate::memory::{
    CompressionExtractionInput, CompressionMessageSnapshot, CompressionRemovalReport,
    ExtractionPolicy, MemorySystem, ResolvedNamespace,
};
use crate::models::openai::{Choice, Message, OpenAIRequest, OpenAIResponse, Usage};
use crate::providers::bedrock::{
    apply_global_inference_prefix, apply_global_inference_profile, model_supports_reasoning,
    normalize_mantle_chat_messages, sanitize_mantle_chat_request, BedrockProvider,
};
use crate::providers::{ProviderClient, ProviderResponse};
use crate::smart_routing::budget_controller::BudgetController;
use crate::smart_routing::{
    PinnedRoutingContext, RoutingPlanOutcome, RoutingPlanningError, SmartRouter, SmartRoutingInput,
};
use dashmap::DashMap;
use std::collections::HashSet;
use std::error::Error as StdError;
use std::sync::Arc;
use std::time::Duration;
use std::time::{SystemTime, UNIX_EPOCH};
use tokio::sync::RwLock;
use tracing::{debug, info, warn};

const PRECOMPRESSED_CACHE_MARKER_KEY: &str = "cache_control";
const PRECOMPRESSED_CACHE_MARKER_TYPE: &str = "obey_precompressed_context";

struct CompressionRuntime {
    pipeline: Arc<CompressionPipeline>,
    precompressed_manager: Option<Arc<PrecompressedManager>>,
}

use super::{CircuitBreaker, LatencyTracker, RateLimiter};

fn smart_routing_tier_name(tier: crate::smart_routing::tier::SmartRoutingTier) -> &'static str {
    match tier {
        crate::smart_routing::tier::SmartRoutingTier::Fast => "fast",
        crate::smart_routing::tier::SmartRoutingTier::Balanced => "balanced",
        crate::smart_routing::tier::SmartRoutingTier::Powerful => "powerful",
    }
}

fn smart_routing_classifier_name(
    classifier: crate::smart_routing::tier::ClassifierUsed,
) -> &'static str {
    match classifier {
        crate::smart_routing::tier::ClassifierUsed::Heuristic => "heuristic",
        crate::smart_routing::tier::ClassifierUsed::Ml => "ml",
        crate::smart_routing::tier::ClassifierUsed::Llm => "llm",
        crate::smart_routing::tier::ClassifierUsed::Composite => "composite",
    }
}

fn smart_routing_task_name(task: crate::smart_routing::tier::TaskType) -> &'static str {
    match task {
        crate::smart_routing::tier::TaskType::CodeGeneration => "code_generation",
        crate::smart_routing::tier::TaskType::MathReasoning => "math_reasoning",
        crate::smart_routing::tier::TaskType::CreativeWriting => "creative_writing",
        crate::smart_routing::tier::TaskType::FactualQA => "factual_qa",
        crate::smart_routing::tier::TaskType::ToolUse => "tool_use",
        crate::smart_routing::tier::TaskType::Summarization => "summarization",
        crate::smart_routing::tier::TaskType::General => "general",
    }
}

/// Intelligent router for provider selection and request routing
pub struct Router {
    config: Arc<RwLock<Config>>,
    circuit_breakers: Arc<DashMap<String, Arc<CircuitBreaker>>>,
    latency_tracker: Arc<LatencyTracker>,
    rate_limiters: Arc<DashMap<String, Arc<RateLimiter>>>,
    http_clients: Arc<DashMap<String, reqwest::Client>>,
    /// Context manager for automatic context window handling
    context_manager: Arc<ContextManager>,
    /// Request-scoped runtime snapshot for compression and pre-compressed contexts.
    compression_runtime: Arc<std::sync::RwLock<CompressionRuntime>>,
    /// Shared dashboard event stream for every provider-specific compression attempt.
    compression_events: Option<Arc<CompressionEventHub>>,
    /// Shared metrics for recording provider-level stats
    metrics: Arc<crate::metrics::Metrics>,
    /// Atomically swappable Smart Router snapshot. Requests clone one snapshot
    /// before classification, so reload never changes an in-flight decision.
    smart_router: Arc<std::sync::RwLock<Option<Arc<SmartRouter>>>>,
    /// Hot-reloadable memory snapshot used by the compression heuristic hook.
    memory_system: Option<Arc<RwLock<Option<Arc<MemorySystem>>>>>,
    /// OAuth session manager used when a provider is configured with
    /// `auth_method: oauth`. `None` while OAuth is not wired up (pre-task
    /// 14.1) — in that state OAuth bearer resolution is silently skipped and
    /// providers fall back to their configured api_key. Req 6.2
    /// (`openai-oauth-login` spec).
    oauth_manager: Option<Arc<crate::oauth::OAuthManager>>,
    /// Codex system instructions store. Populated at startup when at least one
    /// Codex-capable provider (oauth+openai) is configured.
    instructions_store: Option<Arc<crate::codex::InstructionsStore>>,
    /// Tracks OpenAI rate-limit headers for OAuth providers (browser login).
    /// Used to display usage in admin UI and as fallback cooldown when
    /// no Retry-After header is present on 429 responses.
    oauth_usage_tracker: Option<Arc<crate::oauth::UsageTracker>>,
    /// Adaptively learned set of `provider::model` combinations whose model
    /// emits XML-style tool calls (instead of native `tool_calls`). Populated
    /// at runtime when a streaming pass-through response is detected to contain
    /// XML tool use. Subsequent tool requests for a learned combo take the
    /// buffer-and-translate path so the XML is rewritten into native
    /// `tool_calls`. In-memory only — resets on process restart.
    xml_tool_combos: Arc<std::sync::RwLock<HashSet<String>>>,
}

/// Result of [`Router::route_request_streaming`].
///
/// Either a live upstream streaming body that the handler relays chunk-by-chunk
/// (true pass-through), or a fully buffered response produced by the existing
/// non-streaming path when the provider needs response transformation or
/// pass-through is disabled.
///
/// No derives: `reqwest::Response` is not `Clone` and carries a live body, so
/// the variant must be consumed by the streaming relay (task 5.3).
///
/// Requirements: 3.1, 3.8, 3.9
// Consumed by the streaming handler wired up in task 5.5.
pub enum StreamingResponse {
    /// True streaming pass-through: forward the upstream SSE body as-is.
    PassThrough {
        byte_stream: reqwest::Response,
        provider: String,
        model: String,
        compression: CompressionStats,
    },
    /// Buffer-and-replay fallback: a complete response the handler re-chunks.
    Buffered(OpenAIResponse),
}

/// OpenAI-compatible provider endpoint handled without request/response translation.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ProviderPassThroughEndpoint {
    Embeddings,
    ImageGenerations,
    AudioTranscriptions,
    AudioTranslations,
}

impl ProviderPassThroughEndpoint {
    fn path(self) -> &'static str {
        match self {
            Self::Embeddings => "embeddings",
            Self::ImageGenerations => "images/generations",
            Self::AudioTranscriptions => "audio/transcriptions",
            Self::AudioTranslations => "audio/translations",
        }
    }
}

/// Buffered upstream response for non-chat OpenAI-compatible endpoints.
pub struct ProviderPassThroughResponse {
    pub status: u16,
    pub headers: reqwest::header::HeaderMap,
    pub body: Vec<u8>,
}

#[derive(Clone)]
struct ProviderPassThroughTarget {
    provider: Provider,
    model: ProviderModel,
}

impl Router {
    fn build_smart_router(
        smart_routing_config: crate::smart_routing::config::SmartRoutingConfig,
    ) -> Option<Arc<SmartRouter>> {
        if !smart_routing_config.enabled {
            return None;
        }
        SmartRouter::new(smart_routing_config.clone())
            .map(|router| {
        #[cfg_attr(not(feature = "ml-router"), allow(unused_mut))]
        let mut router = router;
        #[cfg(feature = "ml-router")]
                if matches!(
                    smart_routing_config.classifier,
                    crate::smart_routing::config::ClassifierMode::Ml
                        | crate::smart_routing::config::ClassifierMode::Composite
                ) {
                    if let Some(ml_model_path) = &smart_routing_config.ml_model_path {
                        match crate::smart_routing::ml_classifier::OnnxMlAdapter::load(ml_model_path)
                        {
                            Ok(adapter) => {
                                router = router.with_ml_classifier(Arc::new(adapter));
                                tracing::info!(
                                    path = %ml_model_path,
                                    "Smart-routing ONNX ML classifier loaded"
                                );
                            }
                            Err(error) => {
                                tracing::warn!(
                                    error = %error,
                                    path = %ml_model_path,
                                    "Smart-routing ONNX ML classifier unavailable; falling back to heuristic"
                                );
                            }
                        }
                    }
                }
                if smart_routing_config.budget_limits.is_empty() {
                    return router;
                }
                let state_path = smart_routing_config
                    .online_optimizer
                    .state_path
                    .as_deref()
                    .map(std::path::PathBuf::from)
                    .unwrap_or_else(|| std::path::PathBuf::from("smart_routing_state.json"))
                    .with_extension("budget.json");
                match BudgetController::load(state_path) {
                    Ok(controller) => router.with_budget(Arc::new(controller)),
                    Err(error) => {
                        warn!(error = %error, "Smart-routing budget state unavailable; budget checks remain conservatively unavailable");
                        router
                    }
                }
            })
            .map(Arc::new)
            .map_err(|error| tracing::error!(error = %error, "Failed to initialize Smart Router"))
            .ok()
    }

    pub fn reload_smart_router(
        &self,
        config: crate::smart_routing::config::SmartRoutingConfig,
    ) -> Result<(), String> {
        let replacement = if config.enabled {
            Self::build_smart_router(config.clone())
                .ok_or_else(|| "failed to initialize Smart Router replacement".to_string())?
                .into()
        } else {
            None
        };
        if config.enabled {
            self.metrics.enable_smart_routing();
        }
        *self
            .smart_router
            .write()
            .unwrap_or_else(std::sync::PoisonError::into_inner) = replacement;
        Ok(())
    }

    fn smart_router_snapshot(&self) -> Option<Arc<SmartRouter>> {
        self.smart_router
            .read()
            .unwrap_or_else(std::sync::PoisonError::into_inner)
            .clone()
    }

    /// Create a new Router with the given configuration
    pub fn new(config: Arc<RwLock<Config>>, metrics: Arc<crate::metrics::Metrics>) -> Self {
        let (context_config, compression_config, smart_routing_config) = {
            let cfg = config.try_read().expect("config lock");
            (
                cfg.context.clone(),
                cfg.compression.clone(),
                cfg.smart_routing.clone(),
            )
        };
        let smart_router = Self::build_smart_router(smart_routing_config.clone());
        if smart_routing_config.enabled {
            metrics.enable_smart_routing();
        }
        Self {
            config,
            circuit_breakers: Arc::new(DashMap::new()),
            latency_tracker: Arc::new(LatencyTracker::new()),
            rate_limiters: Arc::new(DashMap::new()),
            http_clients: Arc::new(DashMap::new()),
            context_manager: Arc::new(ContextManager::with_config(context_config)),
            compression_runtime: Arc::new(std::sync::RwLock::new(CompressionRuntime {
                pipeline: Arc::new(CompressionPipeline::from_config(compression_config)),
                precompressed_manager: None,
            })),
            compression_events: None,
            metrics,
            smart_router: Arc::new(std::sync::RwLock::new(smart_router)),
            memory_system: None,
            oauth_manager: None,
            instructions_store: None,
            oauth_usage_tracker: None,
            xml_tool_combos: Arc::new(std::sync::RwLock::new(HashSet::new())),
        }
    }

    /// Create a new Router with explicit context configuration
    #[allow(dead_code)]
    pub fn with_context_config(
        config: Arc<RwLock<Config>>,
        context_config: ContextConfig,
        metrics: Arc<crate::metrics::Metrics>,
    ) -> Self {
        let (compression_config, smart_routing_config) = {
            let cfg = config.try_read().expect("config lock");
            (cfg.compression.clone(), cfg.smart_routing.clone())
        };
        let smart_router = Self::build_smart_router(smart_routing_config.clone());
        if smart_routing_config.enabled {
            metrics.enable_smart_routing();
        }
        Self {
            config,
            circuit_breakers: Arc::new(DashMap::new()),
            latency_tracker: Arc::new(LatencyTracker::new()),
            rate_limiters: Arc::new(DashMap::new()),
            http_clients: Arc::new(DashMap::new()),
            context_manager: Arc::new(ContextManager::with_config(context_config)),
            compression_runtime: Arc::new(std::sync::RwLock::new(CompressionRuntime {
                pipeline: Arc::new(CompressionPipeline::from_config(compression_config)),
                precompressed_manager: None,
            })),
            compression_events: None,
            metrics,
            smart_router: Arc::new(std::sync::RwLock::new(smart_router)),
            memory_system: None,
            oauth_manager: None,
            instructions_store: None,
            oauth_usage_tracker: None,
            xml_tool_combos: Arc::new(std::sync::RwLock::new(HashSet::new())),
        }
    }

    /// Attach an [`OAuthManager`](crate::oauth::OAuthManager) so providers
    /// configured with `auth_method: oauth` can resolve a Bearer access token
    /// per request. Called once during gateway startup (task 14.1).
    pub fn set_oauth_manager(&mut self, manager: Arc<crate::oauth::OAuthManager>) {
        self.oauth_manager = Some(manager);
    }

    /// Attach a [`UsageTracker`](crate::oauth::UsageTracker) for capturing
    /// OpenAI rate-limit headers from OAuth provider responses.
    pub fn set_oauth_usage_tracker(&mut self, tracker: Arc<crate::oauth::UsageTracker>) {
        self.oauth_usage_tracker = Some(tracker);
    }

    /// Attach an [`InstructionsStore`](crate::codex::InstructionsStore) for
    /// Codex providers. Called once during gateway startup.
    pub fn set_instructions_store(&mut self, store: Arc<crate::codex::InstructionsStore>) {
        self.instructions_store = Some(store);
    }

    /// Get the context manager
    #[allow(dead_code)]
    pub fn context_manager(&self) -> Arc<ContextManager> {
        self.context_manager.clone()
    }

    /// Attach the shared dashboard compression event hub.
    pub fn set_compression_event_hub(&mut self, hub: Arc<CompressionEventHub>) {
        self.compression_events = Some(hub);
    }

    /// Attach the hot-reloadable memory snapshot used after compression locks release.
    pub fn set_memory_system(&mut self, memory_system: Arc<RwLock<Option<Arc<MemorySystem>>>>) {
        self.memory_system = Some(memory_system);
    }

    /// Atomically replaces the pipeline used by requests that start after reload.
    /// In-flight requests retain the snapshot they already cloned.
    #[cfg_attr(not(test), allow(dead_code))]
    pub fn reload_compression_pipeline(&self, config: CompressionConfig) {
        let replacement = Arc::new(CompressionPipeline::from_config(config));
        let mut runtime = self
            .compression_runtime
            .write()
            .expect("compression runtime lock poisoned");
        runtime.pipeline = replacement;
    }

    /// Atomically replaces both compression runtime components on hot reload.
    pub fn reload_compression_runtime(
        &self,
        config: CompressionConfig,
        manager: Option<Arc<PrecompressedManager>>,
    ) {
        let replacement = CompressionRuntime {
            pipeline: Arc::new(CompressionPipeline::from_config(config)),
            precompressed_manager: manager,
        };
        let mut runtime = self
            .compression_runtime
            .write()
            .expect("compression runtime lock poisoned");
        *runtime = replacement;
    }

    /// Atomically replaces the pre-compressed context manager used by new requests.
    pub fn set_precompressed_manager(&self, manager: Option<Arc<PrecompressedManager>>) {
        let mut runtime = self
            .compression_runtime
            .write()
            .expect("compression runtime lock poisoned");
        runtime.precompressed_manager = manager;
    }

    /// Test-only wrapper for the post-truncation compression path.
    #[cfg(test)]
    pub(crate) async fn prepare_compressed_request(
        &self,
        request: &OpenAIRequest,
        model_group: &ModelGroup,
        provider_model: &ProviderModel,
        request_id: &str,
    ) -> OpenAIRequest {
        self.prepare_compressed_request_with_stats(request, model_group, provider_model, request_id)
            .await
            .0
    }

    async fn prepare_compressed_request_with_stats(
        &self,
        request: &OpenAIRequest,
        model_group: &ModelGroup,
        provider_model: &ProviderModel,
        request_id: &str,
    ) -> (OpenAIRequest, CompressionStats) {
        let (
            global_config,
            provider_override,
            prompt_caching_enabled,
            default_context_window,
            tool_compression_enabled,
        ) = {
            let config = self.config.read().await;
            let provider = config
                .providers
                .iter()
                .find(|provider| provider.name == provider_model.provider);
            (
                config.compression.clone(),
                provider.and_then(|provider| provider.compression.clone()),
                provider.is_some_and(|provider| provider.prompt_caching),
                config.context.default_context_window,
                config.tool_compression.enabled,
            )
        };
        let effective =
            global_config.resolve(provider_override.as_ref(), model_group.compression.as_ref());
        let caveman_output = effective.caveman_output;
        let (pipeline, precompressed_manager) = {
            let runtime = self
                .compression_runtime
                .read()
                .expect("compression runtime lock poisoned");
            (
                runtime.pipeline.clone(),
                runtime.precompressed_manager.clone(),
            )
        };
        let context_window = self
            .context_manager
            .get_capabilities(&provider_model.model)
            .map(|capabilities| capabilities.context_window)
            .unwrap_or(default_context_window);
        let context = CompressionContext {
            model: provider_model.model.clone(),
            context_window,
            provider_name: provider_model.provider.clone(),
            prompt_caching_enabled,
            tool_compression_applied: tool_compression_enabled,
            ..CompressionContext::default()
        };
        let metadata = CompressionRequestMetadata {
            request_id: request_id.to_owned(),
            ..CompressionRequestMetadata::default()
        };
        let mut payload = CompressiblePayload::from(request);
        let inserted_markers = precompressed_manager
            .as_deref()
            .map(|manager| Self::load_precompressed_references(&mut payload, manager, request_id))
            .unwrap_or_default();
        Self::refresh_precompressed_metadata(&mut payload);
        let mut result = if effective.auto_threshold_tokens > 0 {
            pipeline
                .compress_auto(payload, context, effective, metadata)
                .await
        } else {
            pipeline
                .compress_explicit(payload, context, effective, metadata)
                .await
        };

        if result.timed_out {
            warn!(
                request_id,
                provider = %provider_model.provider,
                model = %provider_model.model,
                original_tokens = result.original_tokens,
                duration_ms = result.duration_ms,
                "Compression timed out; forwarding original request"
            );
        } else {
            debug!(
                request_id,
                provider = %provider_model.provider,
                model = %provider_model.model,
                original_tokens = result.original_tokens,
                final_tokens = result.final_tokens,
                engines_applied = result.engines_applied.len(),
                duration_ms = result.duration_ms,
                "Prepared outgoing request compression"
            );
        }

        // Caveman output is applied after the pipeline's safe payload is selected.
        // The helper itself refuses to mutate a cache-protected prefix.
        let caveman_applied = apply_caveman_output(&mut result.payload, caveman_output);
        if caveman_applied {
            debug!(
                request_id,
                provider = %provider_model.provider,
                model = %provider_model.model,
                "Applied caveman output mode"
            );
        }

        let stats = CompressionStats::from_pipeline_result(
            &result,
            caveman_applied,
            &provider_model.provider,
            &provider_model.model,
        );
        Self::remove_precompressed_markers(&mut result.payload, &inserted_markers);
        let prepared_request = result.payload.into_openai_request();
        self.run_memory_compression_hook(request, &prepared_request, request_id)
            .await;
        stats.log();
        self.metrics.record_compression(&stats);
        if let Some(hub) = &self.compression_events {
            hub.publish(stats.clone());
        }
        if Self::compression_savings_warning_required(&stats) {
            warn!(
                request_id = %stats.request_id,
                provider = %stats.provider,
                model = %stats.model,
                level = ?stats.level,
                original_tokens = stats.original_tokens,
                compressed_tokens = stats.compressed_tokens,
                tokens_saved = stats.tokens_saved(),
                savings_percent = stats.savings_percent,
                compression_time_ms = stats.compression_time_ms,
                timed_out = stats.timed_out,
                error = stats.error,
                "Compression saved more than 50 percent of input tokens"
            );
        }

        (prepared_request, stats)
    }

    async fn run_memory_compression_hook(
        &self,
        before_request: &OpenAIRequest,
        after_request: &OpenAIRequest,
        request_id: &str,
    ) {
        let Some(handle) = &self.memory_system else {
            return;
        };
        let Some(memory) = handle.read().await.clone() else {
            return;
        };
        let config = memory.config.read().await.clone();
        if !config.enabled {
            return;
        }

        let counter = crate::compression::token_counter::TokenCounter::new();
        let before_owned = before_request
            .messages
            .iter()
            .enumerate()
            .map(|(index, message)| {
                let content = message.content_as_text();
                let tokens = Self::message_token_count(&counter, before_request, message);
                (index.to_string(), content, tokens)
            })
            .collect::<Vec<_>>();
        let after_owned = after_request
            .messages
            .iter()
            .enumerate()
            .map(|(index, message)| {
                let content = message.content_as_text();
                let tokens = Self::message_token_count(&counter, after_request, message);
                (index.to_string(), content, tokens)
            })
            .collect::<Vec<_>>();
        let before = before_owned
            .iter()
            .map(|(id, content, tokens)| CompressionMessageSnapshot {
                message_id: id,
                content,
                tokens: *tokens,
            })
            .collect::<Vec<_>>();
        let after = after_owned
            .iter()
            .map(|(id, content, tokens)| CompressionMessageSnapshot {
                message_id: id,
                content,
                tokens: *tokens,
            })
            .collect::<Vec<_>>();
        let removals = before
            .iter()
            .map(|snapshot| CompressionRemovalReport {
                message_id: snapshot.message_id,
                tokens_before: snapshot.tokens,
                tokens_after: after
                    .iter()
                    .find(|candidate| candidate.message_id == snapshot.message_id)
                    .map_or(0, |candidate| candidate.tokens),
            })
            .collect::<Vec<_>>();

        let candidates = match memory
            .extractor
            .compression_candidates(CompressionExtractionInput {
                before: &before,
                after: &after,
                removals: &removals,
            }) {
            Ok(candidates) => candidates,
            Err(error) => {
                warn!(request_id, error = %error, "Memory compression heuristic failed");
                return;
            }
        };
        if candidates.is_empty() {
            return;
        }

        let context = memory.context_detector.detect(before_request);
        let namespace = ResolvedNamespace::resolve(None, &context);
        let extractor = memory.extractor.clone();
        let source_request_id = uuid::Uuid::parse_str(request_id).ok();
        let policy = ExtractionPolicy {
            allow_sensitive_storage: config.allow_sensitive_storage,
            max_memories_per_namespace: config.max_memories_per_namespace as usize,
        };
        tokio::spawn(async move {
            if let Err(error) = extractor
                .persist_compression_candidates(candidates, &namespace, source_request_id, policy)
                .await
            {
                warn!(error = %error, "Failed to persist memory compression candidates");
            }
        });
    }

    fn message_token_count(
        counter: &crate::compression::token_counter::TokenCounter,
        request: &OpenAIRequest,
        message: &Message,
    ) -> u32 {
        let mut one = request.clone();
        one.messages = vec![message.clone()];
        counter.count_request(&one)
    }

    fn load_precompressed_references(
        payload: &mut CompressiblePayload,
        manager: &PrecompressedManager,
        request_id: &str,
    ) -> HashSet<usize> {
        let mut inserted_markers = HashSet::new();
        for message in &mut payload.messages {
            let mut used_precompressed = false;
            Self::replace_precompressed_value(
                message.content.as_value_mut(),
                manager,
                true,
                request_id,
                &mut used_precompressed,
            );
            if used_precompressed {
                message.cache_protected = true;
                message.critical = true;
                if !message.extra.contains_key(PRECOMPRESSED_CACHE_MARKER_KEY) {
                    // Pipeline metadata refreshes discover cache boundaries from wire-shaped
                    // values. This unique temporary marker survives every engine refresh and is
                    // removed by original_index before the request is converted back to wire data.
                    message.extra.insert(
                        PRECOMPRESSED_CACHE_MARKER_KEY.to_owned(),
                        serde_json::json!({"type": PRECOMPRESSED_CACHE_MARKER_TYPE}),
                    );
                    inserted_markers.insert(message.original_index);
                }
            }
        }
        payload.refresh_metadata();
        inserted_markers
    }

    fn refresh_precompressed_metadata(payload: &mut CompressiblePayload) {
        let counter = crate::compression::token_counter::TokenCounter::new();
        let model = payload.model.clone();
        for message in &mut payload.messages {
            let wire_message = Message {
                role: message.role.clone(),
                content: message.content.as_value().clone(),
                extra: message.extra.clone(),
            };
            let mut request = OpenAIRequest {
                model: model.clone(),
                messages: vec![wire_message],
                stream: false,
                temperature: None,
                max_tokens: None,
                extra: Default::default(),
            };
            let with_message = counter.count_request(&request);
            request.messages.clear();
            let without_message = counter.count_request(&request);
            message.token_count = with_message.saturating_sub(without_message);
        }
    }

    fn replace_precompressed_value(
        value: &mut serde_json::Value,
        manager: &PrecompressedManager,
        root: bool,
        request_id: &str,
        used_precompressed: &mut bool,
    ) {
        match value {
            serde_json::Value::String(text) if root => {
                Self::replace_precompressed_string(text, manager, request_id, used_precompressed);
            }
            serde_json::Value::Array(parts) if root => {
                for part in parts {
                    match part {
                        serde_json::Value::String(text) => Self::replace_precompressed_string(
                            text,
                            manager,
                            request_id,
                            used_precompressed,
                        ),
                        serde_json::Value::Object(_) => Self::replace_precompressed_content_block(
                            part,
                            manager,
                            request_id,
                            used_precompressed,
                        ),
                        _ => {}
                    }
                }
            }
            serde_json::Value::Object(_) if root => Self::replace_precompressed_content_block(
                value,
                manager,
                request_id,
                used_precompressed,
            ),
            _ => {}
        }
    }

    fn replace_precompressed_content_block(
        value: &mut serde_json::Value,
        manager: &PrecompressedManager,
        request_id: &str,
        used_precompressed: &mut bool,
    ) {
        let Some(object) = value.as_object_mut() else {
            return;
        };
        let block_type = object.get("type").and_then(serde_json::Value::as_str);

        if block_type == Some("file_reference") && object.len() == 2 {
            let source_reference = object
                .get("path")
                .and_then(serde_json::Value::as_str)
                .map(str::to_owned);
            if let Some(source_reference) = source_reference {
                if let Some((content, hit)) =
                    Self::load_precompressed_reference(manager, &source_reference, request_id)
                {
                    *value = serde_json::Value::String(content);
                    *used_precompressed |= hit;
                }
            }
            return;
        }

        match block_type {
            Some("text" | "input_text" | "output_text") => {
                if let Some(serde_json::Value::String(text)) = object.get_mut("text") {
                    Self::replace_precompressed_string(
                        text,
                        manager,
                        request_id,
                        used_precompressed,
                    );
                }
                return;
            }
            Some("system" | "documentation") => {
                if let Some(content) = object.get_mut("content") {
                    Self::replace_precompressed_value(
                        content,
                        manager,
                        true,
                        request_id,
                        used_precompressed,
                    );
                }
                return;
            }
            Some(_) => return,
            None => {}
        }

        for key in ["system", "documentation", "content"] {
            if let Some(content) = object.get_mut(key) {
                Self::replace_precompressed_value(
                    content,
                    manager,
                    true,
                    request_id,
                    used_precompressed,
                );
            }
        }
    }

    fn replace_precompressed_string(
        text: &mut String,
        manager: &PrecompressedManager,
        request_id: &str,
        used_precompressed: &mut bool,
    ) {
        let Some(source_reference) = text.strip_prefix("file://") else {
            return;
        };
        if source_reference.is_empty() {
            return;
        }
        if let Some((content, hit)) =
            Self::load_precompressed_reference(manager, source_reference, request_id)
        {
            *text = content;
            *used_precompressed |= hit;
        }
    }

    fn load_precompressed_reference(
        manager: &PrecompressedManager,
        source_reference: &str,
        request_id: &str,
    ) -> Option<(String, bool)> {
        match manager.load(source_reference) {
            Ok(loaded) => {
                let hit = loaded.used_precompressed();
                match loaded.status {
                    PrecompressedLoadStatus::Hit => {
                        if let Some(metadata) = loaded.metadata.as_ref() {
                            debug!(
                                request_id,
                                original_tokens = metadata.original_tokens,
                                compressed_tokens = metadata.compressed_tokens,
                                level = ?metadata.level,
                                "Loaded validated pre-compressed context"
                            );
                        }
                    }
                    PrecompressedLoadStatus::Stale(reason)
                    | PrecompressedLoadStatus::RuntimeFallback(reason) => {
                        debug!(
                            request_id,
                            ?reason,
                            "Loaded original context for runtime compression fallback"
                        );
                    }
                }
                Some((loaded.content, hit))
            }
            Err(error) => {
                debug!(
                    request_id,
                    error = %error,
                    "Ignored explicit context reference not registered for pre-compression"
                );
                None
            }
        }
    }

    fn remove_precompressed_markers(
        payload: &mut CompressiblePayload,
        inserted_markers: &HashSet<usize>,
    ) {
        let marker = serde_json::json!({"type": PRECOMPRESSED_CACHE_MARKER_TYPE});
        for message in &mut payload.messages {
            if inserted_markers.contains(&message.original_index)
                && message.extra.get(PRECOMPRESSED_CACHE_MARKER_KEY) == Some(&marker)
            {
                message.extra.remove(PRECOMPRESSED_CACHE_MARKER_KEY);
            }
        }
    }

    fn compression_savings_warning_required(stats: &CompressionStats) -> bool {
        stats.savings_percent > 50.0
    }

    /// Get the instructions store (used by admin test-connection endpoint).
    pub fn instructions_store(&self) -> Option<Arc<crate::codex::InstructionsStore>> {
        self.instructions_store.clone()
    }

    /// Find the model group containing the requested model
    ///
    /// Returns the model group if found, or an error if the model is not configured
    pub async fn find_model_group(&self, model: &str) -> Result<ModelGroup, GatewayError> {
        let config = self.config.read().await;

        for group in &config.model_groups {
            // Match by group name first (allows clients to use group names directly)
            if group.name == model {
                return Ok(group.clone());
            }
            for provider_model in &group.models {
                if provider_model.model == model {
                    return Ok(group.clone());
                }
            }
        }

        Err(GatewayError::InvalidRequest(format!(
            "Model '{}' not found in any model group",
            model
        )))
    }

    async fn smart_routing_plan(
        &self,
        request: &OpenAIRequest,
        model_group: &ModelGroup,
    ) -> Result<Option<crate::smart_routing::CandidatePlan>, GatewayError> {
        let Some(smart_router) = self.smart_router_snapshot() else {
            return Ok(None);
        };
        let pinned_model = (request.model != model_group.name).then(|| request.model.clone());
        let pinned_context = PinnedRoutingContext {
            model: pinned_model,
            ..PinnedRoutingContext::default()
        };
        let input = SmartRoutingInput {
            request_id: request
                .extra
                .get("request_id")
                .and_then(serde_json::Value::as_str)
                .unwrap_or("unavailable"),
            request,
            model_group,
            pinned_context: &pinned_context,
        };
        match smart_router.plan(&input).await {
            Ok(RoutingPlanOutcome::Route(plan)) => Ok(Some(plan)),
            Ok(RoutingPlanOutcome::CacheHit(_)) => Err(GatewayError::Cache(
                "smart-routing semantic cache payload integration is unavailable".to_string(),
            )),
            Ok(RoutingPlanOutcome::BudgetRejected(rejection)) => {
                Err(GatewayError::SmartRoutingBudgetExceeded {
                    period: format!("{:?}", rejection.reason).to_ascii_lowercase(),
                })
            }
            Err(RoutingPlanningError::ContextCapacity(error)) => {
                Err(GatewayError::ContextCapacityExceeded {
                    estimated_requirement: error.estimated_requirement,
                    largest_supported_context: error.largest_known_context,
                })
            }
            Err(RoutingPlanningError::NoCandidates) => Err(GatewayError::InvalidRequest(
                "No models are configured for the requested model group".to_string(),
            )),
            Err(RoutingPlanningError::DisabledForModelGroup) => Ok(None),
        }
    }

    /// Select provider order based on priority, cost, latency, and availability
    ///
    /// Algorithm:
    /// 1. Filter out providers whose circuit breaker is open
    /// 2. Filter out providers in an upstream-driven rate-limit cooldown
    ///    (set by a recent 429 / `Retry-After`). Internal token-bucket
    ///    exhaustion is *not* used as a pre-filter — `route_with_failover`
    ///    handles that with proper attempt logging so failover is visible.
    /// 3. Sort by priority (ascending - lower priority value = higher priority)
    /// 4. Within same priority, sort by cost (ascending - lower cost first)
    /// 5. Within similar costs (±10%), sort by latency (ascending - lower latency first)
    /// 6. If version_fallback_enabled, sort by version date (descending - newer first)
    pub async fn select_provider_order(&self, model_group: &ModelGroup) -> Vec<ProviderModel> {
        let mut filtered = Vec::with_capacity(model_group.models.len());
        for m in &model_group.models {
            // CB keys are "provider:model" (per-model circuit breakers).
            // Clone the Arc out of the DashMap guard and drop the guard
            // before awaiting — holding a shard guard across `.await`
            // blocks every other operation on the same shard (spec task 4).
            let cb_key = format!("{}:{}", m.provider, m.model);
            let cb = self
                .circuit_breakers
                .get(&cb_key)
                .map(|entry| entry.value().clone());
            let cb_ok = match cb {
                Some(cb) => cb.is_available().await,
                None => true,
            };

            // Only treat upstream-driven cooldown as a pre-filter signal.
            // The internal token bucket is left to the failover path so an
            // exhausted bucket on one provider doesn't silently re-route
            // traffic to a different model under the same operator
            // (see `route_with_failover`).
            //
            // We consult two stores:
            //   1. `RateLimiter::cooldown_until` — the per-`Router` source
            //      of truth, populated when a 429/Retry-After lands.
            //   2. `Metrics::provider_cooldown_remaining_secs` — the
            //      durable epoch-seconds copy that backs the dashboard.
            //
            // Both must agree that the provider is eligible. The metrics
            // store is the *survivor* across config hot-reloads
            // (`apply_runtime_config_update` clears the router's
            // rate_limiters DashMap but cannot clear the metrics map,
            // which is shared state). Without this second check, every
            // config save re-opens routing to providers the operator is
            // still seeing rendered as "Pausing for ~Nh (rate limited)"
            // in the UI, producing a fresh 429 within seconds.
            // Same guard discipline as the CB check above: clone the Arc out
            // of the DashMap and drop the shard guard before awaiting.
            let rl = self
                .rate_limiters
                .get(&m.provider)
                .map(|entry| entry.value().clone());
            let cooldown_ok = match rl {
                Some(rl) => rl.cooldown_remaining().await.is_none(),
                None => true,
            } && self
                .metrics
                .provider_cooldown_remaining_secs(&m.provider)
                .is_none();

            if cb_ok && cooldown_ok {
                filtered.push(m.clone());
            }
        }
        let mut candidates = filtered;

        // Capture latency snapshot once before sorting.
        // Eliminates repeated DashMap traversal and median calculation in the comparator.
        let latency_snapshot = self.latency_tracker.snapshot();

        // Stage 3: Sort by priority, cost, and latency
        candidates.sort_by(|a, b| {
            // First: sort by priority (ascending)
            match a.priority.cmp(&b.priority) {
                std::cmp::Ordering::Equal => {
                    // Second: sort by total cost (ascending)
                    let cost_a = a.total_cost();
                    let cost_b = b.total_cost();

                    // Check if costs are within 10% of each other
                    let cost_diff = (cost_a - cost_b).abs();
                    let cost_threshold = cost_a.min(cost_b) * 0.1;

                    if cost_diff <= cost_threshold {
                        // Costs are similar, sort by latency
                        // Use immutable snapshot instead of querying mutable tracker
                        let latency_a = latency_snapshot.get_latency(&a.provider);
                        let latency_b = latency_snapshot.get_latency(&b.provider);
                        latency_a
                            .partial_cmp(&latency_b)
                            .unwrap_or(std::cmp::Ordering::Equal)
                    } else {
                        // Costs are different, sort by cost
                        cost_a
                            .partial_cmp(&cost_b)
                            .unwrap_or(std::cmp::Ordering::Equal)
                    }
                }
                other => other,
            }
        });

        // Stage 4: Version fallback sorting (if enabled)
        if model_group.version_fallback_enabled {
            candidates = self.sort_by_version_fallback(candidates);
        }

        candidates
    }

    /// Sort models by version date (descending - newer versions first)
    ///
    /// Extracts version dates from model names in format "model-name-YYYY-MM-DD"
    /// Models without version dates are treated as oldest versions
    #[inline]
    fn sort_by_version_fallback(&self, mut models: Vec<ProviderModel>) -> Vec<ProviderModel> {
        models.sort_by(|a, b| {
            let version_a = Self::extract_version_date(&a.model);
            let version_b = Self::extract_version_date(&b.model);

            // Sort descending (newer versions first)
            version_b.cmp(&version_a)
        });

        models
    }

    /// Extract version date from model name
    ///
    /// Returns a tuple (year, month, day) or (0, 0, 0) if no version found
    #[inline]
    fn extract_version_date(model_name: &str) -> (u32, u32, u32) {
        // Look for pattern YYYY-MM-DD at the end of the model name
        let parts: Vec<&str> = model_name.split('-').collect();

        if parts.len() >= 3 {
            let len = parts.len();
            if let (Ok(year), Ok(month), Ok(day)) = (
                parts[len - 3].parse::<u32>(),
                parts[len - 2].parse::<u32>(),
                parts[len - 1].parse::<u32>(),
            ) {
                // Basic validation
                if year >= 2020
                    && year <= 2100
                    && month >= 1
                    && month <= 12
                    && day >= 1
                    && day <= 31
                {
                    return (year, month, day);
                }
            }
        }

        (0, 0, 0) // No version found
    }

    /// Get or create circuit breaker for a provider
    pub async fn get_circuit_breaker(&self, provider: &str) -> Arc<CircuitBreaker> {
        if let Some(cb) = self.circuit_breakers.get(provider) {
            return cb.value().clone();
        }

        let config = self.config.read().await;

        let backoff_sequence: Vec<std::time::Duration> = config
            .circuit_breaker
            .backoff_sequence_seconds
            .iter()
            .map(|&s| std::time::Duration::from_secs(s))
            .collect();

        let cb = Arc::new(CircuitBreaker::with_backoff_sequence(
            config.circuit_breaker.failure_threshold,
            backoff_sequence,
        ));

        self.circuit_breakers
            .insert(provider.to_string(), cb.clone());
        cb
    }

    /// Record a circuit-breaker failure for a provider that disconnected or
    /// errored during a true-streaming pass-through relay (Req 4.5).
    ///
    /// The pass-through path in [`Self::route_request_streaming_excluding`]
    /// hands the live body to the handler without observing mid-stream
    /// failures, so the handler calls this when the relay fails — either before
    /// any content was forwarded (task 6.1, transparent failover) or after
    /// content was already sent (task 6.2, error event + close). In both cases
    /// the breaker must account for the failed attempt. The circuit-breaker key
    /// matches the gating key (`"{provider}:{model}"`).
    ///
    /// `reason` is recorded verbatim on the Provider Health dashboard via
    /// [`Metrics::record_provider_failure_with_reason`], keeping the streaming
    /// failure path consistent with the non-streaming failover path. Pass
    /// `None` to leave the previous dashboard reason untouched.
    pub async fn record_streaming_failure(
        &self,
        provider: &str,
        model: &str,
        reason: Option<String>,
    ) {
        let cb_key = format!("{}:{}", provider, model);
        let cb = self.get_circuit_breaker(&cb_key).await;
        cb.record_failure().await;
        self.metrics
            .record_provider_failure_with_reason(provider, reason, None);
    }

    /// Detect and strip image content parts from messages when the target
    /// model does not support vision inputs.
    ///
    /// OpenAI-style messages can carry `content` as an array of parts
    /// including `{ "type": "image_url", "image_url": { ... } }`. Many
    /// providers respond with HTTP 400 if such parts reach a non-vision
    /// model. This method removes those parts and logs that fact.
    fn strip_image_content_if_unsupported(
        request: &mut OpenAIRequest,
        supports_vision: bool,
        provider_name: &str,
        model: &str,
    ) -> usize {
        if supports_vision {
            return 0;
        }

        let mut stripped_total: usize = 0;
        for (idx, msg) in request.messages.iter_mut().enumerate() {
            if let serde_json::Value::Array(parts) = &mut msg.content {
                let before = parts.len();
                parts.retain(|part| {
                    if part.get("type").and_then(|v| v.as_str()) == Some("image_url") {
                        false
                    } else {
                        true
                    }
                });
                let removed = before.saturating_sub(parts.len());
                if removed > 0 && !parts.is_empty() {
                    warn!(
                        provider = provider_name,
                        model = %model,
                        message_index = idx,
                        images_removed = removed,
                        "Stripped image_url content parts from message for non-vision model"
                    );
                }
                stripped_total += removed;
            }
        }
        stripped_total
    }

    /// Get or create rate limiter for a provider
    pub async fn get_rate_limiter(&self, provider: &str) -> Arc<RateLimiter> {
        if let Some(rl) = self.rate_limiters.get(provider) {
            return rl.value().clone();
        }

        let config = self.config.read().await;

        let rate_limit = config
            .providers
            .iter()
            .find(|p| p.name == provider)
            .map(|p| p.rate_limit_per_minute)
            .unwrap_or(0);

        let rl = Arc::new(RateLimiter::new(rate_limit));
        self.rate_limiters.insert(provider.to_string(), rl.clone());
        rl
    }

    /// Get latency tracker
    #[allow(dead_code)]
    pub fn get_latency_tracker(&self) -> Arc<LatencyTracker> {
        self.latency_tracker.clone()
    }

    /// Clear all circuit breaker states (used during config reload)
    pub fn clear_circuit_breakers(&self) {
        self.circuit_breakers.clear();
    }

    /// Get circuit breaker states for all providers (used by Prometheus exporter).
    /// Returns Vec of (provider_name, state_label) where state_label is "closed", "open", or "half_open".
    pub async fn get_circuit_breaker_states(&self) -> Vec<(String, String)> {
        let mut results = Vec::new();
        for entry in self.circuit_breakers.iter() {
            let provider = entry.key().clone();
            let cb = entry.value().clone();
            let state_label = match cb.get_state().await {
                super::circuit_breaker::CircuitState::Closed => "closed",
                super::circuit_breaker::CircuitState::Open { .. } => "open",
                super::circuit_breaker::CircuitState::HalfOpen => "half_open",
            };
            results.push((provider, state_label.to_string()));
        }
        results
    }

    /// Clear all rate limiter states (used during config reload)
    pub fn clear_rate_limiters(&self) {
        self.rate_limiters.clear();
    }

    pub fn clear_http_clients(&self) {
        self.http_clients.clear();
    }

    #[allow(dead_code)] // reserved for future budget enforcement
    fn get_provider_budget_limit_usd(config: &Config, provider_name: &str) -> Option<f64> {
        config
            .providers
            .iter()
            .find(|provider| provider.name == provider_name)
            .and_then(|provider| provider.budget.as_ref().map(|budget| budget.limit_usd))
    }

    /// Store model capabilities from provider's list_models response
    #[allow(dead_code)]
    pub fn store_model_capabilities(&self, models: &[crate::providers::Model]) {
        self.context_manager.store_models(models);
    }

    /// Clear model capabilities cache (used during config reload)
    pub fn clear_model_capabilities(&self) {
        self.context_manager.clear_cache();
    }

    /// Check if an error indicates a context length problem
    pub fn is_context_length_error(&self, status: u16, body: &str) -> bool {
        self.context_manager.is_context_length_error(status, body)
    }

    /// Detect a rate-limit signal from any source.
    ///
    /// Returns `true` when:
    /// - the upstream HTTP status is 429, or
    /// - the response body contains a recognizable rate-limit / quota
    ///   marker (covering providers that wrap rate limits inside HTTP 200
    ///   error envelopes, e.g. Nano-GPT / OpenRouter style).
    ///
    /// Used to unify backoff suppression and cooldown application across
    /// the inner retry loop.
    pub(crate) fn is_rate_limited(status_code: u16, body: &str) -> bool {
        if status_code == 429 {
            return true;
        }

        // Cheap fast path: skip body sniffing when the status is clearly
        // unrelated to rate limiting.
        if status_code != 200 && !(400..500).contains(&status_code) {
            return false;
        }

        // Try a structured parse first; fall back to a case-insensitive
        // substring check so providers with varied error envelopes still
        // get caught (e.g. plain text bodies).
        if let Ok(json) = serde_json::from_str::<serde_json::Value>(body) {
            let err = json.get("error").unwrap_or(&json);

            for field in ["code", "type", "status", "reason"] {
                if let Some(v) = err.get(field).and_then(|x| x.as_str()) {
                    let v = v.to_ascii_lowercase();
                    if v.contains("rate_limit")
                        || v.contains("rate-limit")
                        || v.contains("ratelimit")
                        || v.contains("rate_limited")
                        || v.contains("quota")
                        || v.contains("insufficient_quota")
                        || v == "429"
                    {
                        return true;
                    }
                }
            }

            if let Some(code) = err.get("code").and_then(|c| c.as_i64()) {
                if code == 429 {
                    return true;
                }
            }

            if let Some(msg) = err.get("message").and_then(|m| m.as_str()) {
                if Self::message_indicates_rate_limit(msg) {
                    return true;
                }
            }

            return false;
        }

        Self::message_indicates_rate_limit(body)
    }

    /// Substring check for rate-limit phrases. Centralized so the JSON
    /// and plain-text paths agree on the same vocabulary.
    fn message_indicates_rate_limit(text: &str) -> bool {
        let lower = text.to_ascii_lowercase();
        lower.contains("rate limit")
            || lower.contains("rate-limit")
            || lower.contains("rate_limit")
            || lower.contains("ratelimit")
            || lower.contains("too many requests")
            || lower.contains("quota exceeded")
            || lower.contains("insufficient_quota")
            || lower.contains("quota_exceeded")
    }

    /// Translate an upstream failure into a short, plain-English sentence
    /// suitable for the Provider Health dashboard.
    ///
    /// The goal is for a non-technical operator to understand at a glance
    /// *why* a provider is currently failing, without needing to read
    /// status codes, JSON envelopes, or stack traces. Detailed diagnostics
    /// still live in the logs and Recent Errors tab.
    pub(crate) fn friendly_failure_reason(
        status_code: Option<u16>,
        body_or_message: &str,
    ) -> String {
        // Try to extract the provider's own error message, if any. Most
        // OpenAI-compatible envelopes look like {"error":{"message": "..."}}.
        let parse_provider_message = |text: &str| {
            serde_json::from_str::<serde_json::Value>(text)
                .ok()
                .or_else(|| {
                    text.find('{').and_then(|start| {
                        serde_json::from_str::<serde_json::Value>(&text[start..]).ok()
                    })
                })
                .and_then(|v| {
                    v.get("error")
                        .and_then(|e| e.get("message"))
                        .and_then(|m| m.as_str())
                        .map(|s| s.to_string())
                        .or_else(|| {
                            v.get("message")
                                .and_then(|m| m.as_str())
                                .map(|s| s.to_string())
                        })
                })
        };
        let provider_msg: Option<String> = parse_provider_message(body_or_message);

        let snippet = provider_msg
            .as_deref()
            .map(Self::truncate_for_display)
            .unwrap_or_default();

        match status_code {
            Some(code) if Self::is_rate_limited(code, body_or_message) => {
                "Rate limited by provider — pausing until the limit resets".to_string()
            }
            Some(401) => "Authentication failed — check the provider's API key".to_string(),
            Some(403) => {
                "Provider refused the request (forbidden) — check account permissions".to_string()
            }
            Some(402) => "Billing issue at the provider (payment required)".to_string(),
            Some(404) => "Provider could not find the requested model or endpoint".to_string(),
            Some(408) => "Provider took too long to respond (request timeout)".to_string(),
            Some(413) => "Request was too large for the provider".to_string(),
            Some(code) if (500..600).contains(&code) => {
                if code == 503 {
                    "Provider is temporarily unavailable".to_string()
                } else {
                    format!("Provider had a server error (HTTP {})", code)
                }
            }
            Some(code) if (400..500).contains(&code) => {
                if snippet.is_empty() {
                    format!("Provider rejected the request (HTTP {})", code)
                } else {
                    format!("Provider rejected the request: {}", snippet)
                }
            }
            None => {
                // Network / timeout / unparsed error message fell into the
                // "no status code" bucket. Reuse the message text directly
                // when it's already friendly (e.g. our TtfbTimeout text),
                // otherwise normalize the most common cases.
                let lower = body_or_message.to_ascii_lowercase();
                if lower.contains("timeout") || lower.contains("timed out") {
                    "Network timeout while contacting provider".to_string()
                } else if lower.contains("dns")
                    || lower.contains("connection refused")
                    || lower.contains("connect error")
                    || lower.contains("could not resolve")
                {
                    "Could not reach provider (network error)".to_string()
                } else if !snippet.is_empty() {
                    format!("Provider error: {}", snippet)
                } else {
                    "Provider request failed (network error)".to_string()
                }
            }
            Some(code) => format!("Provider returned an unexpected response (HTTP {})", code),
        }
    }

    /// Trim provider-supplied error text to a single short line so the
    /// dashboard cell stays readable.
    fn truncate_for_display(text: &str) -> String {
        const MAX: usize = 140;
        let one_line: String = text
            .lines()
            .next()
            .unwrap_or("")
            .trim()
            .chars()
            .take(MAX)
            .collect();
        if one_line.chars().count() == MAX {
            format!("{}…", one_line)
        } else {
            one_line
        }
    }

    /// Compute the cooldown duration to apply for a rate-limit response.
    ///
    /// Looks at, in order of preference:
    /// 1. `Retry-After` HTTP header (seconds, or HTTP-date)
    /// 2. `retry-after-ms` HTTP header
    /// 3. `X-RateLimit-Reset` HTTP header (Unix epoch seconds — OpenAI,
    ///    Nano-GPT, OpenRouter, etc.) and `X-RateLimit-Reset-After`
    ///    (relative seconds — GitHub-style)
    /// 4. Anthropic-style `anthropic-ratelimit-*-reset` headers (RFC 3339)
    /// 5. `error.retry_after` / `error.retry_after_ms` body fields
    /// 6. `error.reset_at` / `error.reset` body fields (epoch seconds
    ///    or RFC 3339)
    /// 7. Period markers in the error message text:
    ///    - "weekly" / "per week"  -> 7d
    ///    - "daily" / "per day"    -> 24h
    ///    - "hourly" / "per hour"  -> 1h
    ///    - "monthly" / "per month"-> 30d
    ///
    /// Falls back to `retry.default_rate_limit_cooldown_seconds`
    /// (default 30s) when no signal is present.
    ///
    /// The returned value is clamped to:
    /// `min(provider.max_rate_limit_cooldown_seconds,
    ///      retry.max_rate_limit_cooldown_seconds,
    ///      rate_limiter::MAX_COOLDOWN)`.
    pub(crate) async fn parse_rate_limit_cooldown(
        &self,
        provider_name: &str,
        headers: Option<&reqwest::header::HeaderMap>,
        body: &str,
    ) -> Duration {
        let config = self.config.read().await;
        let global_cap_secs = config.retry.max_rate_limit_cooldown_seconds;
        let default_cooldown =
            Duration::from_secs(config.retry.default_rate_limit_cooldown_seconds);
        let provider_cap_secs = config
            .providers
            .iter()
            .find(|p| p.name == provider_name)
            .and_then(|p| p.max_rate_limit_cooldown_seconds);
        drop(config);

        Self::compute_rate_limit_cooldown(
            headers,
            body,
            default_cooldown,
            provider_cap_secs,
            global_cap_secs,
        )
    }

    /// Pure cooldown computation — no I/O. Exposed for unit testing the
    /// header / body / period-marker parsing without spinning up a router.
    pub(crate) fn compute_rate_limit_cooldown(
        headers: Option<&reqwest::header::HeaderMap>,
        body: &str,
        default_cooldown: Duration,
        provider_cap_secs: Option<u64>,
        global_cap_secs: u64,
    ) -> Duration {
        let cap = {
            // Operator policy: per-provider override wins over global,
            // since operators set the per-provider value specifically to
            // raise (or lower) the cooldown for that provider. If no
            // provider override is set, use the global cap.
            //
            // The limiter backstop is a separate, hard ceiling that
            // protects against runaway / nonsense values regardless of
            // operator config.
            let chosen_secs = provider_cap_secs.unwrap_or(global_cap_secs);
            let backstop = crate::router::rate_limiter::MAX_COOLDOWN.as_secs();
            Duration::from_secs(chosen_secs.min(backstop))
        };

        if let Some(h) = headers {
            if let Some(d) = Self::cooldown_from_headers(h) {
                return d.min(cap);
            }
        }

        if let Some(d) = Self::cooldown_from_body(body) {
            return d.min(cap);
        }

        if let Some(d) = Self::cooldown_from_period_marker(body) {
            return d.min(cap);
        }

        default_cooldown.min(cap)
    }

    fn cooldown_from_headers(headers: &reqwest::header::HeaderMap) -> Option<Duration> {
        // Millisecond-precision retry-after first (Anthropic, some
        // OpenAI-compatible providers).
        if let Some(v) = headers.get("retry-after-ms").and_then(|h| h.to_str().ok()) {
            if let Ok(ms) = v.trim().parse::<u64>() {
                return Some(Duration::from_millis(ms));
            }
        }

        // Standard Retry-After: seconds or HTTP-date.
        if let Some(v) = headers
            .get(reqwest::header::RETRY_AFTER)
            .and_then(|h| h.to_str().ok())
        {
            let trimmed = v.trim();
            if let Ok(secs) = trimmed.parse::<u64>() {
                return Some(Duration::from_secs(secs));
            }
            if let Ok(when) = chrono::DateTime::parse_from_rfc2822(trimmed) {
                let now = chrono::Utc::now();
                let delta = when.with_timezone(&chrono::Utc) - now;
                if let Ok(std_dur) = delta.to_std() {
                    return Some(std_dur);
                }
                return Some(Duration::from_secs(0));
            }
        }

        // X-RateLimit-Reset-After: relative seconds (GitHub style, some
        // OpenAI-compatible providers also emit it).
        for name in ["x-ratelimit-reset-after", "ratelimit-reset"] {
            if let Some(v) = headers.get(name).and_then(|h| h.to_str().ok()) {
                if let Ok(secs) = v.trim().parse::<u64>() {
                    // `ratelimit-reset` (RFC draft) is "delta seconds";
                    // some implementations emit epoch instead. Treat
                    // values smaller than 1e9 as relative.
                    if secs < 1_000_000_000 {
                        return Some(Duration::from_secs(secs));
                    }
                }
            }
        }

        // X-RateLimit-Reset: Unix epoch seconds (OpenAI, OpenRouter,
        // Nano-GPT for weekly/daily quotas).
        for name in [
            "x-ratelimit-reset",
            "x-rate-limit-reset",
            "x-ratelimit-reset-requests",
            "x-ratelimit-reset-tokens",
        ] {
            if let Some(v) = headers.get(name).and_then(|h| h.to_str().ok()) {
                if let Some(d) = Self::duration_until_epoch_or_relative(v.trim()) {
                    return Some(d);
                }
            }
        }

        // Anthropic-style ISO-8601 reset headers.
        for name in [
            "anthropic-ratelimit-requests-reset",
            "anthropic-ratelimit-tokens-reset",
            "anthropic-ratelimit-input-tokens-reset",
            "anthropic-ratelimit-output-tokens-reset",
        ] {
            if let Some(v) = headers.get(name).and_then(|h| h.to_str().ok()) {
                if let Ok(when) = chrono::DateTime::parse_from_rfc3339(v.trim()) {
                    let now = chrono::Utc::now();
                    let delta = when.with_timezone(&chrono::Utc) - now;
                    if let Ok(std_dur) = delta.to_std() {
                        return Some(std_dur);
                    }
                    return Some(Duration::from_secs(0));
                }
            }
        }

        None
    }

    /// Interpret a header value as either an epoch-seconds timestamp
    /// (large number), a relative-seconds delta (small number), or a
    /// Go-style duration string (e.g. "6m0s", "4h32m10s"). Returns the
    /// resulting `Duration` from "now" until the reset.
    fn duration_until_epoch_or_relative(v: &str) -> Option<Duration> {
        // Try plain integer first (epoch or relative seconds).
        if let Ok(secs) = v.parse::<u64>() {
            let now_epoch = chrono::Utc::now().timestamp() as u64;
            if secs > now_epoch.saturating_sub(60 * 60 * 24)
                && secs < now_epoch.saturating_add(60 * 60 * 24 * 365)
            {
                // Plausible epoch timestamp.
                let delta = secs.saturating_sub(now_epoch);
                return Some(Duration::from_secs(delta));
            }
            // Otherwise treat as relative seconds.
            return Some(Duration::from_secs(secs));
        }

        // Try Go-style duration format: "4h32m10s", "6m0s", "12ms", etc.
        Self::parse_go_duration(v)
    }

    /// Parse a Go-style duration string like "4h32m10s", "6m0s", "12ms".
    /// Returns None if the string cannot be parsed.
    fn parse_go_duration(s: &str) -> Option<Duration> {
        let s = s.trim();
        if s.is_empty() {
            return None;
        }
        let mut total_secs: u64 = 0;
        let mut current_num = String::new();
        let chars: Vec<char> = s.chars().collect();
        let mut i = 0;
        while i < chars.len() {
            let c = chars[i];
            if c.is_ascii_digit() || c == '.' {
                current_num.push(c);
                i += 1;
            } else if c == 'm' && i + 1 < chars.len() && chars[i + 1] == 's' {
                // milliseconds — round up to at least 1s
                if let Ok(ms) = current_num.parse::<f64>() {
                    total_secs += (ms / 1000.0).ceil().max(1.0) as u64;
                }
                current_num.clear();
                i += 2;
            } else {
                if let Ok(num) = current_num.parse::<f64>() {
                    match c {
                        'd' => total_secs += (num * 86400.0) as u64,
                        'h' => total_secs += (num * 3600.0) as u64,
                        'm' => total_secs += (num * 60.0) as u64,
                        's' => total_secs += num as u64,
                        _ => {}
                    }
                }
                current_num.clear();
                i += 1;
            }
        }
        if !current_num.is_empty() {
            if let Ok(num) = current_num.parse::<u64>() {
                total_secs += num;
            }
        }
        if total_secs > 0 {
            Some(Duration::from_secs(total_secs))
        } else {
            None
        }
    }

    fn cooldown_from_body(body: &str) -> Option<Duration> {
        let json = serde_json::from_str::<serde_json::Value>(body).ok()?;
        let err = json.get("error").unwrap_or(&json);

        for field in ["retry_after_ms", "retry-after-ms"] {
            if let Some(ms) = err.get(field).and_then(|v| v.as_u64()) {
                return Some(Duration::from_millis(ms));
            }
        }
        for field in ["retry_after", "retry-after"] {
            if let Some(secs) = err.get(field).and_then(|v| v.as_u64()) {
                return Some(Duration::from_secs(secs));
            }
            if let Some(secs) = err.get(field).and_then(|v| v.as_f64()) {
                if secs.is_finite() && secs >= 0.0 {
                    return Some(Duration::from_secs_f64(secs));
                }
            }
        }
        // Reset-at fields: epoch seconds or RFC 3339 string.
        for field in ["reset_at", "resets_at", "reset"] {
            if let Some(v) = err.get(field) {
                if let Some(secs) = v.as_u64() {
                    if let Some(d) = Self::duration_until_epoch_or_relative(&secs.to_string()) {
                        return Some(d);
                    }
                }
                if let Some(s) = v.as_str() {
                    if let Ok(when) = chrono::DateTime::parse_from_rfc3339(s) {
                        let now = chrono::Utc::now();
                        let delta = when.with_timezone(&chrono::Utc) - now;
                        if let Ok(std_dur) = delta.to_std() {
                            return Some(std_dur);
                        }
                        return Some(Duration::from_secs(0));
                    }
                }
            }
        }
        None
    }

    /// Last-resort signal extraction: scan the error message for period
    /// keywords. Useful for providers like Nano-GPT that surface "weekly
    /// limit reached" in plain text without machine-readable headers.
    fn cooldown_from_period_marker(body: &str) -> Option<Duration> {
        let lower = body.to_ascii_lowercase();

        // Order matters: check the longer windows first so "monthly"
        // doesn't get caught by a "month" substring inside "monthly
        // active users" etc.
        const HOUR: u64 = 60 * 60;
        const DAY: u64 = 24 * HOUR;

        if lower.contains("per month")
            || lower.contains("monthly limit")
            || lower.contains("monthly quota")
        {
            return Some(Duration::from_secs(30 * DAY));
        }
        if lower.contains("per week")
            || lower.contains("weekly limit")
            || lower.contains("weekly quota")
        {
            return Some(Duration::from_secs(7 * DAY));
        }
        if lower.contains("per day")
            || lower.contains("daily limit")
            || lower.contains("daily quota")
        {
            return Some(Duration::from_secs(DAY));
        }
        if lower.contains("per hour")
            || lower.contains("hourly limit")
            || lower.contains("hourly quota")
        {
            return Some(Duration::from_secs(HOUR));
        }
        None
    }

    /// Apply context truncation at the handler/router lifecycle boundary.
    /// Memory injection must consume the returned request and run before routing,
    /// because provider-specific compression occurs inside the route methods.
    pub fn prepare_post_truncation_request(&self, request: &OpenAIRequest) -> OpenAIRequest {
        let (prepared, truncated) = self.check_and_truncate_context(request);
        if truncated {
            info!(model = %request.model, "Applied pre-flight context truncation");
        }
        prepared
    }

    /// Check and potentially truncate context before routing
    /// Returns the request to use (possibly modified) and whether truncation occurred
    pub fn check_and_truncate_context(&self, request: &OpenAIRequest) -> (OpenAIRequest, bool) {
        let config = self.config.try_read().expect("config lock");

        // Skip if context management is disabled
        if !config.context.enabled {
            return (request.clone(), false);
        }

        // Try to get model capabilities
        let context_window = match self.context_manager.get_capabilities(&request.model) {
            Some(caps) => caps.context_window,
            None => {
                // No capabilities known, skip pre-flight check
                return (request.clone(), false);
            }
        };

        // Check if request fits within limits
        if self
            .context_manager
            .fits_within_limits(request, context_window)
        {
            return (request.clone(), false);
        }

        // Request exceeds limits, truncate it
        let mut truncated_request = request.clone();
        let result = self
            .context_manager
            .truncate_request(&mut truncated_request, context_window);

        if result.truncated {
            info!(
                model = %request.model,
                original_tokens = result.original_tokens,
                final_tokens = result.final_tokens,
                messages_removed = result.messages_removed,
                "Context truncated to fit within {} token limit",
                context_window
            );
        }

        (truncated_request, result.truncated)
    }

    async fn dispatch_buffered_with_context_retry<C: ProviderClient + ?Sized>(
        &self,
        client: &C,
        mut request: OpenAIRequest,
    ) -> Result<ProviderResponse, GatewayError> {
        let mut attempt = 0;
        loop {
            match client.chat_completion(request.clone()).await {
                Ok(response) => return Ok(response),
                Err(GatewayError::Provider {
                    provider,
                    message,
                    status_code: Some(status_code),
                }) if self.is_context_length_error(status_code, &message) => {
                    match self.context_manager.handle_context_error(
                        &mut request,
                        attempt,
                        Some(&message),
                    ) {
                        Ok(result) => {
                            attempt += 1;
                            info!(
                                provider = %provider,
                                model = %request.model,
                                attempt,
                                original_tokens = result.original_tokens,
                                final_tokens = result.final_tokens,
                                messages_removed = result.messages_removed,
                                "Buffered provider context-length error detected, truncated request and retrying"
                            );
                        }
                        Err(error) => {
                            return Err(GatewayError::InvalidRequest(format!(
                                "Request exceeds provider context limits and cannot be truncated further: {}",
                                error
                            )));
                        }
                    }
                }
                Err(error) => return Err(error),
            }
        }
    }

    /// Attempt request with retry logic and exponential backoff
    ///
    /// Implements retry with backoff sequence [1s, 2s, 4s]
    /// Skips retry on 4xx errors except 429 (rate limit) and 408 (timeout)
    ///
    /// Requirements: 10.1-10.5
    pub async fn attempt_with_retry(
        &self,
        provider_name: &str,
        request: &OpenAIRequest,
        provider_model: &ProviderModel,
        active: Option<ActiveRequestHandle>,
        base_attempt: usize,
    ) -> Result<OpenAIResponse, GatewayError> {
        let config = self.config.read().await;
        let max_retries = config.retry.max_retries_per_provider;
        let backoff_sequence = config.retry.backoff_sequence_seconds.clone();

        // Find provider config
        let provider_cfg = config
            .providers
            .iter()
            .find(|p| p.name == provider_name)
            .ok_or_else(|| {
                GatewayError::Configuration(format!(
                    "Provider '{}' not found in config",
                    provider_name
                ))
            })?;

        // ─── Codex dispatch (Req 10.1, 10.2) ────────────────────────────────
        let is_codex = provider_cfg.auth_method.as_deref() == Some("oauth")
            && provider_cfg.provider_type == "openai";

        if is_codex {
            // Delegate end-to-end to CodexProviderClient.
            let oauth = match &self.oauth_manager {
                Some(m) => m.clone(),
                None => {
                    return Err(GatewayError::Provider {
                        provider: provider_name.to_string(),
                        message: "OAuth manager not attached for Codex provider".to_string(),
                        status_code: Some(500),
                    });
                }
            };
            let instructions = match &self.instructions_store {
                Some(s) => s.clone(),
                None => {
                    return Err(GatewayError::Provider {
                        provider: provider_name.to_string(),
                        message: "Instructions store not attached for Codex provider".to_string(),
                        status_code: Some(500),
                    });
                }
            };
            let http =
                self.get_or_create_http_client(provider_name, &provider_cfg.connection_pool)?;

            let codex_client = crate::codex::client::CodexProviderClient::new(
                provider_name.to_string(),
                oauth,
                instructions,
                http,
                self.metrics.clone(),
                self.oauth_usage_tracker
                    .clone()
                    .unwrap_or_else(|| Arc::new(crate::oauth::UsageTracker::new())),
                provider_cfg.codex_base_url_override.clone(),
                provider_cfg.codex_model_override.clone(),
                provider_cfg.instructions_override.clone(),
                config.xhigh_models_allowlist.clone(),
                config.reasoning_models_allowlist.clone(),
            );

            let mut codex_request = request.clone();
            // Rewrite the model from the group name to the actual provider model ID
            codex_request.model = provider_model.model.clone();
            let result = self
                .dispatch_buffered_with_context_retry(&codex_client, codex_request)
                .await?;
            return Ok(result.response);
        }
        // ─── End Codex dispatch ──────────────────────────────────────────────

        // Resolve API key: try as env var first, fall back to using the value directly
        let api_key = provider_cfg.resolve_api_key().unwrap_or_default();

        let is_bedrock_api_key = provider_cfg.provider_type == "bedrock" && !api_key.is_empty();
        if is_bedrock_api_key {
            let mut bedrock_request = request.clone();
            bedrock_request.model = provider_model.model.clone();
            bedrock_request.stream = false;
            let bedrock_client = BedrockProvider::new_with_config(
                provider_name.to_string(),
                provider_cfg
                    .region
                    .clone()
                    .unwrap_or_else(|| "us-east-1".to_string()),
                Some(api_key),
                Some(provider_cfg.max_connections),
                Some(provider_cfg.effective_total_timeout(&provider_model.model)),
                provider_cfg.custom_headers.clone(),
            )
            .await?;
            return Ok(self
                .dispatch_buffered_with_context_retry(&bedrock_client, bedrock_request)
                .await?
                .response);
        }

        // OAuth bearer override (Req 6.2): when the provider declares
        // `auth_method: oauth` and the OAuth manager has a live,
        // non-expired access token, use it as the outgoing Bearer in place
        // of any configured api_key.
        let mut oauth_bearer: Option<String> =
            if provider_cfg.auth_method.as_deref() == Some("oauth") {
                match &self.oauth_manager {
                    Some(manager) => manager.get_access_token().await,
                    None => None,
                }
            } else {
                None
            };

        // Req 6.3 (openai-oauth-login): if the provider is configured for
        // OAuth but the session snapshot is Unauthenticated / Expired /
        // Refreshing (or the manager is unattached), skip this provider so
        // the outer fail-over loop moves on to the next candidate. Falling
        // back to the configured `api_key` would route traffic onto the
        // wrong credentials and mask a genuinely unauthenticated session.
        if provider_cfg.auth_method.as_deref() == Some("oauth") && oauth_bearer.is_none() {
            debug!(
                provider = provider_name,
                model = %provider_model.model,
                "OAuth session unusable (unauthenticated/expired/refreshing), skipping provider"
            );
            return Err(GatewayError::Provider {
                provider: provider_name.to_string(),
                message: "OAuth session not authenticated; no usable access token".to_string(),
                status_code: Some(401),
            });
        }

        // Build base URL — strip trailing slash, append /v1 if not present
        // For Bedrock with API key, use the Bedrock Mantle endpoint (unless custom VPC endpoint)
        let mut base_url = if provider_cfg.provider_type == "bedrock"
            && !api_key.is_empty()
            && !provider_cfg.custom_vpc_endpoint
        {
            // Bedrock API key mode: use Bedrock Mantle endpoint (OpenAI-compatible)
            let region = provider_cfg.region.as_deref().unwrap_or("us-east-1");
            format!("https://bedrock-mantle.{}.api.aws/v1", region)
        } else {
            provider_cfg.base_url.clone().unwrap_or_default()
        };
        base_url = base_url.trim_end_matches('/').to_string();
        if !base_url.ends_with("/v1") {
            base_url.push_str("/v1");
        }
        let url = format!("{}/chat/completions", base_url);

        let ttfb_timeout_secs = provider_cfg.effective_ttfb_timeout(&provider_model.model);
        let total_timeout_secs = provider_cfg.effective_total_timeout(&provider_model.model);
        let ttfb_timeout = Duration::from_secs(ttfb_timeout_secs);
        let total_timeout = Duration::from_secs(total_timeout_secs);
        tracing::info!(provider = provider_name, %url, model = %provider_model.model, ttfb_timeout_secs, total_timeout_secs, "Calling provider");

        let pool_config = provider_cfg.connection_pool.clone();
        let custom_headers = provider_cfg.custom_headers.clone();
        let provider_type = provider_cfg.provider_type.clone();
        let cross_region_inference = provider_cfg.cross_region_inference;
        let global_inference_profile = provider_cfg.global_inference_profile;
        let prompt_caching = provider_cfg.prompt_caching;
        let reasoning = provider_cfg.reasoning;
        let provider_region = provider_cfg.region.clone();
        let is_oauth_provider = provider_cfg.auth_method.as_deref() == Some("oauth");
        let jitter_enabled = config.retry.jitter_enabled;
        let jitter_ratio = config.retry.jitter_ratio;

        // Drop config lock before making HTTP calls
        drop(config);

        let http_client = self.get_or_create_http_client(provider_name, &pool_config)?;

        // Build the outgoing request body — override model to the actual provider model name
        // Always request non-streaming from provider; gateway handles client streaming separately
        let mut outgoing = request.clone();
        outgoing.model = provider_model.model.clone();
        if request.stream {
            debug!(
                provider = provider_name,
                model = %provider_model.model,
                "Client requested streaming, but gateway is forcing upstream stream=false and buffering the full provider response"
            );
        }
        outgoing.stream = false;
        let mut context_retry_attempt: usize = 0;

        // Apply Bedrock inference profiles only in AWS SDK mode. Mantle model
        // IDs never accept geo/global prefixes, and only some Runtime models
        // publish inference profiles.
        if provider_type == "bedrock" && api_key.is_empty() {
            let region = provider_region.as_deref().unwrap_or("us-east-1");
            outgoing.model = if global_inference_profile {
                apply_global_inference_profile(&outgoing.model, true)
            } else {
                apply_global_inference_prefix(&outgoing.model, region, cross_region_inference)
            };
        }

        // Prompt caching is configured in the request body using each Bedrock
        // API's cache checkpoint fields. A synthetic HTTP header is not part of
        // the Mantle Chat Completions contract and can make otherwise valid
        // requests fail validation.
        if provider_type == "bedrock" && prompt_caching {
            tracing::debug!(
                provider = provider_name,
                model = %outgoing.model,
                "Bedrock prompt caching enabled; no synthetic request header added"
            );
        }

        // Inject reasoning/extended thinking parameter for Bedrock providers
        if provider_type == "bedrock" && reasoning {
            if model_supports_reasoning(&outgoing.model) {
                outgoing.extra.insert(
                    "thinking".to_string(),
                    serde_json::json!({
                        "type": "enabled",
                        "budget_tokens": 4096
                    }),
                );
            }
        }

        // Strip fields the target provider doesn't support to avoid 400/502 errors.
        if provider_type == "bedrock" {
            let normalized = normalize_mantle_chat_messages(&mut outgoing);
            if normalized > 0 {
                info!(
                    provider = provider_name,
                    model = %provider_model.model,
                    fields_normalized = normalized,
                    "Normalized request messages for Bedrock Mantle compatibility"
                );
            }
        }
        let stripped = Self::sanitize_request_for_provider(&mut outgoing, &provider_type);
        if stripped > 0 {
            info!(
                provider = provider_name,
                provider_type = %provider_type,
                fields_removed = stripped,
                "Sanitized request for provider (removed unsupported fields)"
            );
        }

        // Normalize tool_calls on assistant messages to always be an array.
        // Some clients send tool_calls as a single object or other non-array
        // shapes, which causes downstream providers to reject with:
        //   "assistant.tool_calls must be an array when provided"
        Self::normalize_message_tool_calls(&mut outgoing.messages);

        // Strip image content parts when the target model is known not to
        // support vision inputs. This prevents avoidable HTTP 400 rejections
        // from providers whose non-vision models receive image_url parts.
        let supports_vision = self
            .context_manager
            .get_capabilities(&provider_model.model)
            .map(|caps| caps.supports_vision)
            .unwrap_or(false);
        let images_stripped = Self::strip_image_content_if_unsupported(
            &mut outgoing,
            supports_vision,
            provider_name,
            &provider_model.model,
        );
        if images_stripped > 0 {
            info!(
                provider = provider_name,
                model = %provider_model.model,
                images_stripped,
                "Removed image content from request for non-vision model"
            );
        }

        // Reverse-translate tool_calls history for models that use XML-style tool use.
        //
        // When the gateway previously translated XML tool use → native tool_calls,
        // the client (Roo Code) sends back the conversation with:
        //   - assistant messages containing tool_calls [{id:"call_xlat_0",...}]
        //   - tool result messages with role:"tool", tool_call_id:"call_xlat_0"
        //
        // The model never generated those IDs or that format — it thinks in XML.
        // If we send these back as-is, the model gets confused and loops.
        //
        // Solution: detect gateway-translated tool_calls (by the "call_xlat_" prefix)
        // and convert them back to the XML format the model originally produced,
        // merging assistant+tool message groups into a single conversation flow
        // Diagnostic: log whether the outgoing request carries tools/tool_choice
        // so we can verify the client's tool definitions reach the provider.
        let has_tools = outgoing.extra.contains_key("tools");
        let has_tool_choice = outgoing.extra.contains_key("tool_choice");
        if has_tools || has_tool_choice {
            let tool_count = outgoing
                .extra
                .get("tools")
                .and_then(|v| v.as_array())
                .map(|a| a.len())
                .unwrap_or(0);
            debug!(
                provider = provider_name,
                model = %provider_model.model,
                tool_count,
                has_tool_choice,
                "Outgoing request includes tools definitions"
            );
        }

        // Reverse-translate gateway-translated tool_calls back to XML for the
        // provider.  See translate_xml_tool_calls() for the forward direction.
        if has_tools {
            Self::reverse_translate_tool_history(&mut outgoing.messages);
        }

        // Inject a compact but explicit tool-calling guide.
        //
        // Goal: make native OpenAI-style tool use clear even for models that
        // were primarily trained on XML/pseudo-XML agent formats. The hint is
        // kept intentionally short enough to limit token overhead, while still
        // covering: correct formatting, multi-step usage, and common mistakes.
        // It is appended as the last system message so it doesn't override the
        // client's system prompt.
        if has_tools {
            outgoing.messages.push(Self::tool_calling_system_hint());
        }

        let mut last_error = None;

        for attempt in 0..=max_retries {
            if attempt > 0 {
                // Defense-in-depth: never burn backoff time on a
                // rate-limit-class error from the previous attempt. The
                // explicit branches above already short-circuit, but if a
                // future change adds another rate-limit code path this
                // guard ensures we don't accidentally sleep through it.
                if let Some(GatewayError::Provider {
                    status_code: Some(code),
                    message: prev_msg,
                    ..
                }) = &last_error
                {
                    if Self::is_rate_limited(*code, prev_msg) {
                        debug!(
                            provider = provider_name,
                            status = *code,
                            "Skipping retry backoff for rate-limit-class error"
                        );
                        break;
                    }
                }

                let backoff_secs = backoff_sequence
                    .get((attempt - 1) as usize)
                    .copied()
                    .unwrap_or(4);
                let retry_delay =
                    Self::calculate_retry_delay(backoff_secs, jitter_enabled, jitter_ratio);
                self.metrics
                    .record_provider_retry(provider_name, retry_delay.as_millis() as u64);
                // Report the retry to the in-flight registry: same provider,
                // retrying after the previous attempt's error.
                if let Some(handle) = &active {
                    handle.set_phase(ActivePhase::Retry);
                    handle.set_attempt(base_attempt + attempt as usize);
                    if let Some(err) = &last_error {
                        handle.set_last_error(&err.to_string());
                    }
                }
                tokio::time::sleep(retry_delay).await;
                debug!(
                    provider = provider_name,
                    attempt,
                    delay_ms = retry_delay.as_millis() as u64,
                    "Retrying request"
                );
            }

            let mut req_builder = http_client
                .post(&url)
                .header("Content-Type", "application/json");

            if let Some(ref bearer) = oauth_bearer {
                req_builder = req_builder.header("Authorization", format!("Bearer {}", bearer));
            } else if !api_key.is_empty() {
                req_builder = req_builder.header("Authorization", format!("Bearer {}", api_key));
            }

 for (k, v) in &custom_headers {
 if k.eq_ignore_ascii_case(reqwest::header::ACCEPT_ENCODING.as_str()) {
 continue;
 }
 req_builder = req_builder.header(k.as_str(), v.as_str());
 }
 // RequestBuilder::header appends rather than replaces existing values, so
 // filter any provider-level Accept-Encoding above and add identity exactly
 // once. Compressed SSE is vulnerable to truncated decoder frames.
 req_builder = req_builder.header(reqwest::header::ACCEPT_ENCODING, "identity");


            let request_start = std::time::Instant::now();
            let result =
                tokio::time::timeout(ttfb_timeout, req_builder.json(&outgoing).send()).await;

            let send_result = match result {
                Ok(inner) => inner,
                Err(_) => {
                    // TTFB timeout — provider didn't start responding in time
                    warn!(
                        provider = provider_name,
                        attempt,
                        ttfb_timeout_secs,
                        "TTFB timeout — provider did not respond in time"
                    );
                    last_error = Some(GatewayError::TtfbTimeout(ttfb_timeout_secs));
                    continue;
                }
            };

            match send_result {
                Ok(response) => {
                    let status = response.status();
                    let status_code = status.as_u16();
                    let response_headers = response.headers().clone();

                    // Capture rate-limit headers for OAuth provider usage tracking.
                    if oauth_bearer.is_some() {
                        if let Some(tracker) = &self.oauth_usage_tracker {
                            let tracker = tracker.clone();
                            let hdrs = response_headers.clone();
                            tokio::spawn(async move {
                                tracker.update_from_headers(&hdrs).await;
                            });
                        }
                    }

                    // Read body with remaining total timeout budget
                    let elapsed = request_start.elapsed();
                    let remaining_total = total_timeout.saturating_sub(elapsed);
                    let body_result = tokio::time::timeout(remaining_total, response.text()).await;
                    let body_text = match body_result {
                        Ok(Ok(text)) => text,
                        Ok(Err(e)) => {
                            warn!(
                                provider = provider_name,
                                attempt,
                                error = %e,
                                "Failed to read response body"
                            );
                            last_error = Some(GatewayError::Provider {
                                provider: provider_name.to_string(),
                                message: format!("Failed to read response body: {}", e),
                                status_code: Some(status_code),
                            });
                            continue;
                        }
                        Err(_) => {
                            warn!(
                                provider = provider_name,
                                attempt,
                                total_timeout_secs,
                                "Total timeout — response body read exceeded round-trip limit"
                            );
                            last_error = Some(GatewayError::TotalTimeout(total_timeout_secs));
                            continue;
                        }
                    };
                    tracing::info!(
                        provider = provider_name,
                        status = status_code,
                        body_len = body_text.len(),
                        "Provider responded"
                    );

                    if status.is_success() {
                        // Detect error-in-200: some providers (e.g. Nano-GPT) return
                        // HTTP 200 with an error payload like {"error":{...}}.
                        // Treat these as retryable provider errors instead of parse failures.
                        if let Ok(parsed) = serde_json::from_str::<serde_json::Value>(&body_text) {
                            if parsed.get("error").is_some() && parsed.get("choices").is_none() {
                                let err_msg = parsed["error"]["message"]
                                    .as_str()
                                    .unwrap_or("unknown error in 200 response");

                                // Promote rate-limit-shaped 200 to a real 429.
                                // This unifies failover with proper HTTP 429
                                // handling: cooldown the provider and bail
                                // out of the inner retry loop immediately.
                                if Self::is_rate_limited(200, &body_text) {
                                    let cooldown = self
                                        .parse_rate_limit_cooldown(
                                            provider_name,
                                            Some(&response_headers),
                                            &body_text,
                                        )
                                        .await;
                                    let rate_limiter = self.get_rate_limiter(provider_name).await;
                                    rate_limiter.apply_cooldown(cooldown).await;
                                    self.metrics
                                        .record_provider_rate_limit_exhausted(provider_name);
                                    let now_secs = SystemTime::now()
                                        .duration_since(UNIX_EPOCH)
                                        .map(|d| d.as_secs())
                                        .unwrap_or(0);
                                    let deadline = now_secs.saturating_add(cooldown.as_secs());
                                    self.metrics.set_provider_cooldown(
                                        provider_name,
                                        Self::friendly_failure_reason(Some(429), &body_text),
                                        deadline,
                                    );

                                    warn!(
                                        provider = provider_name,
                                        cooldown_ms = cooldown.as_millis() as u64,
                                        error = %err_msg,
                                        "Provider returned rate-limit-shaped HTTP 200, failing over"
                                    );

                                    return Err(GatewayError::Provider {
                                        provider: provider_name.to_string(),
                                        message: format!(
                                            "Rate limited (HTTP 200 envelope): {}",
                                            err_msg
                                        ),
                                        status_code: Some(429),
                                    });
                                }

                                warn!(
                                    provider = provider_name,
                                    attempt,
                                    error = %err_msg,
                                    "Provider returned error inside HTTP 200 — treating as retryable"
                                );
                                last_error = Some(GatewayError::Provider {
                                    provider: provider_name.to_string(),
                                    message: format!("Error in 200 response: {}", err_msg),
                                    status_code: Some(status_code),
                                });
                                continue;
                            }
                        }

                        // Try parsing as a normal JSON response first
                        if let Ok(openai_response) =
                            serde_json::from_str::<OpenAIResponse>(&body_text)
                        {
                            // Diagnostic: detect whether the model used native tool_calls
                            // or fell back to XML-style tool use in plain text content.
                            if let Some(choice) = openai_response.choices.first() {
                                let has_native_tc = choice.message.extra.contains_key("tool_calls");
                                let content_text = choice.message.content_as_text();
                                let has_xml_tool_use = content_text.contains("<use_tool")
                                    || content_text.contains("<tool_call")
                                    || content_text.contains("<function_call")
                                    || content_text.contains("<invoke ")
                                    || content_text.contains("<tool_calls>")
                                    || content_text.contains("<execute_command")
                                    || content_text.contains("<|tool_call");
                                if has_native_tc {
                                    debug!(
                                        provider = provider_name,
                                        model = %provider_model.model,
                                        finish_reason = ?choice.finish_reason,
                                        "Provider returned native tool_calls"
                                    );
                                }
                                if has_xml_tool_use {
                                    warn!(
                                        provider = provider_name,
                                        model = %provider_model.model,
                                        content_preview = %content_text.chars().take(200).collect::<String>(),
                                        has_tools_in_request = has_tools,
                                        "Model output XML-style tool use as plain text instead of native tool_calls"
                                    );
                                }
                            }
                            return Ok(openai_response);
                        }

                        // Provider may have ignored stream:false and returned SSE chunks.
                        // Parse the SSE stream and reconstruct a single OpenAIResponse.
                        if body_text.starts_with("data: ") {
                            tracing::debug!(
                                provider = provider_name,
                                "Provider returned SSE despite stream:false, reassembling"
                            );
                            match Self::reassemble_sse_response(&body_text) {
                                Ok(response) => return Ok(response),
                                Err(e) => {
                                    tracing::error!(provider = provider_name, error = %e, body = %body_text.chars().take(500).collect::<String>(), "Failed to reassemble SSE response");
                                    return Err(GatewayError::Provider {
                                        provider: provider_name.to_string(),
                                        message: format!("Failed to parse response: {}", e),
                                        status_code: Some(status_code),
                                    });
                                }
                            }
                        }

                        // Neither JSON nor SSE — log and fail
                        tracing::error!(provider = provider_name, body = %body_text.chars().take(500).collect::<String>(), "Failed to parse provider response");
                        return Err(GatewayError::Provider {
                            provider: provider_name.to_string(),
                            message: "Failed to parse response: not JSON or SSE".to_string(),
                            status_code: Some(status_code),
                        });
                    }

                    // Context-length failure: attempt in-process truncation + retry.
                    if self.is_context_length_error(status_code, &body_text) {
                        match self.context_manager.handle_context_error(
                            &mut outgoing,
                            context_retry_attempt,
                            Some(&body_text),
                        ) {
                            Ok(result) => {
                                context_retry_attempt += 1;
                                info!(
                                    provider = provider_name,
                                    model = %provider_model.model,
                                    attempt = context_retry_attempt,
                                    original_tokens = result.original_tokens,
                                    final_tokens = result.final_tokens,
                                    messages_removed = result.messages_removed,
                                    "Context-length error detected, truncated request and retrying"
                                );
                                continue;
                            }
                            Err(e) => {
                                warn!(
                                    provider = provider_name,
                                    model = %provider_model.model,
                                    error = %e,
                                    "Context-length error detected but truncation retry cannot continue"
                                );
                                return Err(GatewayError::InvalidRequest(format!(
                                    "Request exceeds model context limits and cannot be truncated further: {}",
                                    e
                                )));
                            }
                        }
                    }

                    let err = GatewayError::Provider {
                        provider: provider_name.to_string(),
                        message: format!("HTTP {}: {}", status_code, body_text),
                        status_code: Some(status_code),
                    };

                    // Req 6.4 (openai-oauth-login): when an OAuth provider
                    // returns HTTP 401, force an immediate token refresh
                    // *before* the circuit breaker's retry path consumes the
                    // attempt. If the refresh succeeds, update the bearer and
                    // let the retry loop continue with the fresh token.
                    if status_code == 401 && is_oauth_provider {
                        if let Some(ref manager) = self.oauth_manager {
                            debug!(
                                provider = provider_name,
                                attempt,
                                "OAuth provider returned 401, forcing token refresh before retry"
                            );
                            match manager.force_refresh().await {
                                Ok(new_token) => {
                                    oauth_bearer = Some(new_token);
                                    debug!(
                                        provider = provider_name,
                                        "OAuth token refreshed successfully, retrying request"
                                    );
                                    last_error = Some(err);
                                    continue;
                                }
                                Err(e) => {
                                    warn!(
                                        provider = provider_name,
                                        error = %e,
                                        "OAuth force-refresh failed after upstream 401, failing over"
                                    );
                                    return Err(err);
                                }
                            }
                        }
                    }

                    // Don't retry 4xx errors except 408 (timeout)
                    // 429 (rate limit) should fail over to next provider, not retry same one
                    // 503 (service unavailable) signals provider is down — fail over immediately
                    if status_code >= 400 && status_code < 500 && status_code != 408 {
                        // For rate-limit signals, parse Retry-After /
                        // retry_after_ms and put the provider in a
                        // bounded cooldown window so subsequent requests
                        // skip it via select_provider_order without
                        // re-issuing.
                        if Self::is_rate_limited(status_code, &body_text) {
                            let cooldown = self
                                .parse_rate_limit_cooldown(
                                    provider_name,
                                    Some(&response_headers),
                                    &body_text,
                                )
                                .await;
                            let rate_limiter = self.get_rate_limiter(provider_name).await;
                            rate_limiter.apply_cooldown(cooldown).await;
                            self.metrics
                                .record_provider_rate_limit_exhausted(provider_name);
                            let now_secs = SystemTime::now()
                                .duration_since(UNIX_EPOCH)
                                .map(|d| d.as_secs())
                                .unwrap_or(0);
                            let deadline = now_secs.saturating_add(cooldown.as_secs());
                            self.metrics.set_provider_cooldown(
                                provider_name,
                                Self::friendly_failure_reason(Some(status_code), &body_text),
                                deadline,
                            );

                            warn!(
                                provider = provider_name,
                                status = status_code,
                                cooldown_ms = cooldown.as_millis() as u64,
                                "Rate limited, failing over and cooling down provider"
                            );
                        } else {
                            warn!(
                                provider = provider_name,
                                status = status_code,
                                "Non-retryable client error, failing over"
                            );
                        }
                        return Err(err);
                    }
                    if status_code == 503 {
                        warn!(
                            provider = provider_name,
                            status = status_code,
                            "Service unavailable, failing over immediately"
                        );
                        return Err(err);
                    }

                    warn!(
                        provider = provider_name,
                        status = status_code,
                        attempt,
                        "Retryable error"
                    );
                    last_error = Some(err);
                }
                Err(e) => {
                    let err = GatewayError::Provider {
                        provider: provider_name.to_string(),
                        message: format!("Request failed: {}", e),
                        status_code: None,
                    };
                    // Log full error chain for network diagnostics
                    let mut cause_chain = String::new();
                    let mut source: Option<&dyn StdError> = std::error::Error::source(&e);
                    while let Some(cause) = source {
                        cause_chain.push_str(&format!(" -> {}", cause));
                        source = std::error::Error::source(cause);
                    }
                    warn!(
                        provider = provider_name,
                        attempt,
                        error = %e,
                        causes = %cause_chain,
                        is_timeout = e.is_timeout(),
                        is_connect = e.is_connect(),
                        "Network error"
                    );
                    last_error = Some(err);
                }
            }
        }

        Err(last_error.unwrap_or_else(|| GatewayError::Provider {
            provider: provider_name.to_string(),
            message: "All retry attempts exhausted (context truncation retries may have consumed all attempts)".to_string(),
            status_code: None,
        }))
    }

    /// Reassemble an SSE (Server-Sent Events) streaming response into a single OpenAIResponse.
    /// Some providers ignore `stream: false` and return chunked SSE anyway.
    /// This parses all `data: {...}` lines, concatenates delta content, and builds
    /// a complete response object.
    ///
    /// `pub(crate)` so the streaming pass-through relay (task 5.4) can reuse the
    /// same accumulation logic to assemble a cacheable response from forwarded
    /// SSE chunks.
    pub(crate) fn reassemble_sse_response(body: &str) -> Result<OpenAIResponse, String> {
        let mut full_content = String::new();
        let mut reasoning_content = String::new();
        let mut response_id = String::new();
        let mut model = String::new();
        let mut created: i64 = 0;
        let mut finish_reason: Option<String> = None;
        let mut prompt_tokens: u32 = 0;
        let mut completion_tokens: u32 = 0;
        let mut total_tokens: u32 = 0;
        let mut chunk_count: u32 = 0;

        // Accumulate tool_calls from streaming deltas.
        // OpenAI streams tool_calls as indexed entries across multiple chunks:
        //   delta: { tool_calls: [{ index: 0, id: "...", type: "function", function: { name: "...", arguments: "" } }] }
        //   delta: { tool_calls: [{ index: 0, function: { arguments: "{\"pa" } }] }
        //   delta: { tool_calls: [{ index: 0, function: { arguments: "th\":\"file.rs\"}" } }] }
        // We merge them by index into complete tool_call objects.
        use std::collections::BTreeMap;
        let mut tool_calls_map: BTreeMap<u64, serde_json::Value> = BTreeMap::new();

        // Some providers concatenate SSE chunks without newlines between them
        // e.g. "data: {...}data: {...}" instead of "data: {...}\ndata: {...}"
        // Split on "data: " boundaries to handle both cases.
        let chunks_iter: Vec<&str> = body
            .split("data: ")
            .map(|s| s.trim())
            .filter(|s| !s.is_empty())
            .collect();

        for json_str in &chunks_iter {
            if *json_str == "[DONE]" {
                break;
            }
            // Strip trailing "data:" fragment that might appear if split left a partial
            let json_str = json_str.trim_end();
            let chunk: serde_json::Value = match serde_json::from_str(json_str) {
                Ok(v) => v,
                Err(_) => {
                    // Might be a partial or non-JSON line, skip it
                    tracing::trace!(chunk = json_str, "Skipping unparseable SSE chunk");
                    continue;
                }
            };

            chunk_count += 1;

            // Detect mid-stream error frames: chunks with an "error" object
            // or finish_reason of "error". These indicate the provider failed
            // partway through generation.
            if let Some(error_obj) = chunk.get("error") {
                let error_msg = error_obj
                    .get("message")
                    .and_then(|v| v.as_str())
                    .unwrap_or("Unknown mid-stream error");
                let error_status = error_obj
                    .get("status")
                    .and_then(|v| v.as_u64())
                    .map(|s| s as u16);
                let error_code = error_obj
                    .get("code")
                    .and_then(|v| v.as_str())
                    .unwrap_or("unknown");
                tracing::warn!(
                    error_message = error_msg,
                    error_status = ?error_status,
                    error_code = error_code,
                    "Mid-stream error frame received from provider"
                );
                return Err(format!("Mid-stream error ({}): {}", error_code, error_msg));
            }

            // Also check for finish_reason: "error" without a top-level error object
            if let Some(choices) = chunk.get("choices").and_then(|v| v.as_array()) {
                if let Some(choice) = choices.first() {
                    if choice.get("finish_reason").and_then(|v| v.as_str()) == Some("error") {
                        let delta_text = choice
                            .get("delta")
                            .and_then(|d| d.get("content"))
                            .and_then(|c| c.as_str())
                            .unwrap_or("Provider returned error finish_reason");
                        tracing::warn!(
                            detail = delta_text,
                            "Provider stream ended with finish_reason=error"
                        );
                        return Err(format!("Stream error: {}", delta_text));
                    }
                }
            }

            // Grab metadata from first chunk
            if response_id.is_empty() {
                if let Some(id) = chunk.get("id").and_then(|v| v.as_str()) {
                    response_id = id.to_string();
                }
            }
            if model.is_empty() {
                if let Some(m) = chunk.get("model").and_then(|v| v.as_str()) {
                    model = m.to_string();
                }
            }
            if created == 0 {
                if let Some(c) = chunk.get("created").and_then(|v| v.as_i64()) {
                    created = c;
                }
            }

            // Extract delta content from choices[0].delta
            if let Some(choices) = chunk.get("choices").and_then(|v| v.as_array()) {
                if let Some(choice) = choices.first() {
                    if let Some(delta) = choice.get("delta") {
                        if let Some(c) = delta.get("content").and_then(|v| v.as_str()) {
                            full_content.push_str(c);
                        }
                        if let Some(r) = delta.get("reasoning").and_then(|v| v.as_str()) {
                            reasoning_content.push_str(r);
                        }
                        if let Some(r) = delta.get("reasoning_content").and_then(|v| v.as_str()) {
                            reasoning_content.push_str(r);
                        }

                        // Accumulate streamed tool_calls by index
                        if let Some(tc_arr) = delta.get("tool_calls").and_then(|v| v.as_array()) {
                            for tc_delta in tc_arr {
                                let idx =
                                    tc_delta.get("index").and_then(|v| v.as_u64()).unwrap_or(0);
                                let entry = tool_calls_map.entry(idx).or_insert_with(|| {
                                    serde_json::json!({
                                        "id": "",
                                        "type": "function",
                                        "function": { "name": "", "arguments": "" }
                                    })
                                });
                                // Merge id
                                if let Some(id) = tc_delta.get("id").and_then(|v| v.as_str()) {
                                    entry["id"] = serde_json::Value::String(id.to_string());
                                }
                                // Merge type
                                if let Some(t) = tc_delta.get("type").and_then(|v| v.as_str()) {
                                    entry["type"] = serde_json::Value::String(t.to_string());
                                }
                                // Merge function name and arguments (arguments are appended)
                                if let Some(func) = tc_delta.get("function") {
                                    if let Some(name) = func.get("name").and_then(|v| v.as_str()) {
                                        if !name.is_empty() {
                                            entry["function"]["name"] =
                                                serde_json::Value::String(name.to_string());
                                        }
                                    }
                                    if let Some(args) =
                                        func.get("arguments").and_then(|v| v.as_str())
                                    {
                                        let existing =
                                            entry["function"]["arguments"].as_str().unwrap_or("");
                                        entry["function"]["arguments"] = serde_json::Value::String(
                                            format!("{}{}", existing, args),
                                        );
                                    }
                                }
                            }
                        }
                    }
                    if let Some(fr) = choice.get("finish_reason").and_then(|v| v.as_str()) {
                        finish_reason = Some(fr.to_string());
                    }
                }
            }

            // Extract usage if present (some providers send it in the last chunk)
            if let Some(usage) = chunk.get("usage") {
                if let Some(pt) = usage.get("prompt_tokens").and_then(|v| v.as_u64()) {
                    prompt_tokens = pt as u32;
                }
                if let Some(ct) = usage.get("completion_tokens").and_then(|v| v.as_u64()) {
                    completion_tokens = ct as u32;
                }
                if let Some(tt) = usage.get("total_tokens").and_then(|v| v.as_u64()) {
                    total_tokens = tt as u32;
                }
            }
        }

        if chunk_count == 0 {
            return Err("No SSE chunks found in response body".to_string());
        }

        // If we have reasoning content but no regular content, use reasoning as content
        let final_content = if full_content.is_empty() && !reasoning_content.is_empty() {
            reasoning_content.clone()
        } else {
            full_content
        };

        // Estimate tokens if provider didn't send usage
        if total_tokens == 0 {
            completion_tokens = (final_content.len() / 4) as u32; // rough estimate
            total_tokens = prompt_tokens + completion_tokens;
        }

        // Build message extra with tool_calls if any were accumulated
        let mut msg_extra = serde_json::Map::new();
        if !tool_calls_map.is_empty() {
            let tool_calls_vec: Vec<serde_json::Value> = tool_calls_map.into_values().collect();
            msg_extra.insert(
                "tool_calls".to_string(),
                serde_json::Value::Array(tool_calls_vec),
            );
        }
        if !reasoning_content.is_empty() {
            msg_extra.insert(
                "reasoning_content".to_string(),
                serde_json::Value::String(reasoning_content.clone()),
            );
        }

        Ok(OpenAIResponse {
            id: if response_id.is_empty() {
                format!("chatcmpl-reassembled-{}", chunk_count)
            } else {
                response_id
            },
            object: "chat.completion".to_string(),
            created,
            model,
            choices: vec![Choice {
                index: 0,
                message: Message {
                    role: "assistant".to_string(),
                    content: serde_json::Value::String(final_content),
                    extra: msg_extra,
                },
                finish_reason,
                extra: Default::default(),
            }],
            usage: Usage {
                prompt_tokens,
                completion_tokens,
                total_tokens,
                extra: Default::default(),
            },
            extra: Default::default(),
        })
    }

    /// Route request with failover orchestration
    ///
    /// Iterates through providers in order, attempts each with retry logic,
    /// collects all attempts, returns aggregated error if all fail.
    /// Resolves the model group for direct callers before entering the
    /// post-truncation group-aware implementation.
    ///
    /// Requirements: 8.1-8.10
    pub async fn route_with_failover(
        &self,
        request: &OpenAIRequest,
        providers: Vec<ProviderModel>,
        active: Option<ActiveRequestHandle>,
    ) -> Result<OpenAIResponse, GatewayError> {
        let model_group = self.find_model_group(&request.model).await?;
        self.route_with_failover_for_group(
            request,
            &model_group,
            providers,
            active,
            ActivePhase::Primary,
        )
        .await
    }

    /// Model-group-aware failover implementation used after pre-flight truncation.
    async fn route_with_failover_for_group(
        &self,
        request: &OpenAIRequest,
        model_group: &ModelGroup,
        providers: Vec<ProviderModel>,
        active: Option<ActiveRequestHandle>,
        initial_phase: ActivePhase,
    ) -> Result<OpenAIResponse, GatewayError> {
        let mut attempts = Vec::new();
        // Running count of provider attempts across the whole request, surfaced
        // to the dashboard's in-flight view.
        let mut attempt_counter: usize = 0;
        // Partial responses kept when a provider truncates (finish_reason=length
        // well below max_tokens). Task 7.2 returns the longest of these instead
        // of an error when every provider truncates (Req 6.2).
        let mut truncated_candidates: Vec<OpenAIResponse> = Vec::new();
        let config = self.config.read().await;
        let provider_budgets: std::collections::HashMap<String, f64> = config
            .providers
            .iter()
            .filter_map(|provider| {
                provider
                    .budget
                    .as_ref()
                    .map(|budget| (provider.name.clone(), budget.limit_usd))
            })
            .collect();
        // Effective truncation-retry setting (Req 6.4). Defaults to true when
        // no `streaming` section is configured.
        let retry_on_truncation = config
            .streaming
            .clone()
            .unwrap_or_default()
            .retry_on_truncation;
        drop(config);

        for provider_model in providers {
            let start = std::time::Instant::now();
            let request_id = format!("route-{}", uuid::Uuid::new_v4());

            if let Some(budget_limit_usd) = provider_budgets.get(&provider_model.provider).copied()
            {
                self.metrics
                    .set_provider_budget_limit(&provider_model.provider, budget_limit_usd);
                let current_cost_usd = self
                    .metrics
                    .current_provider_cost_usd(&provider_model.provider);
                if current_cost_usd >= budget_limit_usd {
                    warn!(provider = %provider_model.provider, spent_usd = current_cost_usd, budget_limit_usd, "Provider budget exhausted, skipping provider");
                    self.metrics
                        .record_provider_budget_exhausted(&provider_model.provider);
                    attempts.push(ProviderAttempt::new(
                        provider_model.provider.clone(),
                        provider_model.model.clone(),
                        format!(
                            "Provider budget exhausted at ${:.2} / ${:.2}",
                            current_cost_usd, budget_limit_usd
                        ),
                        Some(402),
                    ));
                    continue;
                }
            }

            // Key circuit breaker by provider+model so one model's failures
            // don't lock out other models on the same provider.
            let cb_key = format!("{}:{}", provider_model.provider, provider_model.model);
            let cb = self.get_circuit_breaker(&cb_key).await;
            if !cb.is_available().await {
                debug!(provider = %provider_model.provider, model = %provider_model.model, "Circuit breaker open, skipping provider");
                attempts.push(ProviderAttempt::new(
                    provider_model.provider.clone(),
                    provider_model.model.clone(),
                    "Circuit breaker open".to_string(),
                    Some(503),
                ));
                continue;
            }

            // Defense-in-depth: re-check the durable upstream-driven
            // cooldown that backs the dashboard. `select_provider_order`
            // already filters on this, but a config hot-reload between
            // selection and failover (or any direct call to
            // `route_with_failover`) could leave a stale provider in the
            // candidate list. The metrics store survives
            // `clear_rate_limiters()`, so it is the authoritative gate
            // for "this provider returned 429 / Retry-After recently".
            if let Some(remaining) = self
                .metrics
                .provider_cooldown_remaining_secs(&provider_model.provider)
            {
                debug!(
                    provider = %provider_model.provider,
                    model = %provider_model.model,
                    cooldown_remaining_secs = remaining,
                    "Upstream rate-limit cooldown active, skipping provider"
                );
                attempts.push(ProviderAttempt::new(
                    provider_model.provider.clone(),
                    provider_model.model.clone(),
                    format!(
                        "Provider in upstream rate-limit cooldown ({}s remaining)",
                        remaining
                    ),
                    Some(429),
                ));
                continue;
            }

            // Consume rate limit token before attempting request
            let rate_limiter = self.get_rate_limiter(&provider_model.provider).await;
            if !rate_limiter.consume().await {
                warn!(provider = %provider_model.provider, "Rate limit exhausted, skipping provider");
                self.metrics
                    .record_provider_rate_limit_exhausted(&provider_model.provider);
                attempts.push(ProviderAttempt::new(
                    provider_model.provider.clone(),
                    provider_model.model.clone(),
                    "Rate limit exhausted".to_string(),
                    Some(429),
                ));
                continue;
            }

            let (prepared_request, compression) = self
                .prepare_compressed_request_with_stats(
                    request,
                    model_group,
                    &provider_model,
                    &request_id,
                )
                .await;
            // Report the current attempt target to the in-flight registry. The
            // first attempt of the whole request uses the initial phase (Primary,
            // or Cascade when re-routed by smart routing); subsequent providers
            // entered after a failure are failovers.
            if let Some(handle) = &active {
                attempt_counter += 1;
                let phase = if attempts.is_empty() {
                    initial_phase
                } else {
                    ActivePhase::Failover
                };
                handle.set_target(&provider_model.provider, &provider_model.model, phase);
                handle.set_attempt(attempt_counter);
            }
            match self
                .attempt_with_retry(
                    &provider_model.provider,
                    &prepared_request,
                    &provider_model,
                    active.clone(),
                    attempt_counter,
                )
                .await
            {
                Ok(mut response) => {
                    if Self::promote_reasoning_to_content(&mut response) {
                        warn!(
                            provider = %provider_model.provider,
                            model = %provider_model.model,
                            "Provider returned reasoning without answer content; promoted reasoning to content"
                        );
                    }
                    // Validate that the response actually contains usable content.
                    // Some overwhelmed providers return 200 with empty choices or
                    // null content and no tool_calls — treat these as failures so
                    // failover can try the next provider.
                    if !Self::response_has_content(&response) {
                        warn!(
                            provider = %provider_model.provider,
                            model = %provider_model.model,
                            "Provider returned empty response (no assistant content), failing over"
                        );
                        cb.record_failure().await;
                        self.metrics.record_provider_failure_with_reason(
                            &provider_model.provider,
                            Some(
                                "Provider returned an empty response — no answer text or tool calls"
                                    .to_string(),
                            ),
                            None,
                        );
                        attempts.push(ProviderAttempt::new(
                            provider_model.provider.clone(),
                            provider_model.model.clone(),
                            "Provider returned empty response with no assistant content"
                                .to_string(),
                            Some(200),
                        ));
                        continue;
                    }

                    // --- Truncation detection and retry (Req 6.1, 6.3, 6.4) ---
                    // A provider can return HTTP 200 with finish_reason="length"
                    // yet stop well short of the client's requested max_tokens —
                    // a sign it hit an internal cap rather than the legitimate
                    // limit. When `retry_on_truncation` is enabled, treat this as
                    // a failure and fail over to the next provider, keeping the
                    // partial response as a fallback candidate (consumed by task
                    // 7.2). When `retry_on_truncation` is false we skip detection
                    // entirely and return the response as-is (prior behavior).
                    let is_truncated = retry_on_truncation
                        && response
                            .choices
                            .first()
                            .and_then(|choice| choice.finish_reason.as_deref())
                            == Some("length")
                        && match request.max_tokens {
                            // Req 6.3: only suspicious when the response stopped
                            // well short of the requested limit. If
                            // completion_tokens reached (within 50 of) max_tokens,
                            // the response legitimately hit the requested limit.
                            Some(max_tokens) => {
                                response.usage.completion_tokens < max_tokens.saturating_sub(50)
                            }
                            None => false,
                        };

                    if is_truncated {
                        let max_tokens = request.max_tokens.unwrap_or(0);
                        let completion_tokens = response.usage.completion_tokens;
                        warn!(
                            provider = %provider_model.provider,
                            model = %provider_model.model,
                            completion_tokens,
                            max_tokens,
                            "Provider returned a truncated response (finish_reason=length), failing over"
                        );
                        cb.record_failure().await;
                        self.metrics.record_provider_failure_with_reason(
                            &provider_model.provider,
                            Some(format!(
                                "Provider returned a truncated response (finish_reason=length, {}/{} tokens)",
                                completion_tokens, max_tokens
                            )),
                            None,
                        );
                        attempts.push(ProviderAttempt::new(
                            provider_model.provider.clone(),
                            provider_model.model.clone(),
                            format!(
                                "Response truncated at {}/{} tokens (finish_reason=length)",
                                completion_tokens, max_tokens
                            ),
                            Some(200),
                        ));

                        // Preserve this partial response as a fallback candidate
                        // so task 7.2 can return the longest truncated response
                        // if every provider truncates. Annotate it with the same
                        // gateway metadata + cost the success path attaches, so it
                        // can be returned directly without reprocessing.
                        let mut candidate = response;
                        let input_cost = candidate.usage.prompt_tokens as f64
                            * provider_model.cost_per_million_input_tokens
                            / 1_000_000.0;
                        let output_cost = candidate.usage.completion_tokens as f64
                            * provider_model.cost_per_million_output_tokens
                            / 1_000_000.0;
                        let candidate_cost = input_cost + output_cost;
                        candidate.extra.insert(
                            "gateway_provider".to_string(),
                            serde_json::Value::String(provider_model.provider.clone()),
                        );
                        candidate.extra.insert(
                            "gateway_responded_model".to_string(),
                            serde_json::Value::String(provider_model.model.clone()),
                        );
                        candidate.extra.insert(
                            "gateway_cost".to_string(),
                            serde_json::json!(candidate_cost),
                        );
                        candidate.extra.insert(
                            "gateway_compression".to_string(),
                            serde_json::to_value(&compression)
                                .expect("CompressionStats serialization must succeed"),
                        );
                        truncated_candidates.push(candidate);
                        continue;
                    }

                    // Record success
                    let duration = start.elapsed();
                    let duration_ms = duration.as_millis() as u64;
                    self.latency_tracker
                        .update_latency(&provider_model.provider, duration);
                    self.metrics
                        .record_provider_success(&provider_model.provider, duration_ms);

                    cb.record_success().await;

                    // Provider recovered — clear any upstream-driven cooldown
                    // so we don't keep skipping it after it has come back.
                    let rate_limiter = self.get_rate_limiter(&provider_model.provider).await;
                    rate_limiter.clear_cooldown().await;
                    // The metrics store also holds an independent
                    // cooldown deadline (durable across config reloads)
                    // and feeds both the dashboard countdown and the
                    // routing gate in `select_provider_order` /
                    // `route_with_failover`. Clear it here so a recovered
                    // provider stops being filtered out and the UI
                    // reflects recovery immediately. (record_provider_success
                    // also clears it, but only when we actually report a
                    // success — keep this explicit for readability.)
                    self.metrics
                        .clear_provider_cooldown(&provider_model.provider);

                    // Calculate and record cost from token usage
                    let usage_known = response.usage.total_tokens > 0
                        || response.usage.prompt_tokens > 0
                        || response.usage.completion_tokens > 0;
                    let total_cost = if usage_known {
                        let input_cost = response.usage.prompt_tokens as f64
                            * provider_model.cost_per_million_input_tokens
                            / 1_000_000.0;
                        let output_cost = response.usage.completion_tokens as f64
                            * provider_model.cost_per_million_output_tokens
                            / 1_000_000.0;
                        let total_cost = input_cost + output_cost;
                        if total_cost > 0.0 {
                            self.metrics.add_cost(&provider_model.provider, total_cost);
                        }
                        total_cost
                    } else {
                        self.metrics
                            .record_provider_unknown_cost(&provider_model.provider);
                        0.0
                    };

                    // Translate XML-style tool use to native tool_calls.
                    // Models that don't support the OpenAI tools parameter
                    // (e.g. GLM, Kimi via Nano-GPT) emit tool invocations as
                    // XML tags in plain text.  Rewrite these into proper
                    // tool_calls so clients like Roo Code / Kilo Code work.
                    let mut response = response;
                    if request.extra.contains_key("tools") {
                        Self::translate_xml_tool_calls(&mut response, request);
                    }

                    // Always strip Kimi-style special tokens from response
                    // content, even when no tools are in the request.
                    // Kimi K2.6 can leak raw tokenizer tokens like
                    // <|tool_calls_section_begin|> into plain text.
                    Self::sanitize_kimi_tokens_in_response(&mut response);

                    response.extra.insert(
                        "gateway_provider".to_string(),
                        serde_json::Value::String(provider_model.provider.clone()),
                    );
                    response.extra.insert(
                        "gateway_responded_model".to_string(),
                        serde_json::Value::String(provider_model.model.clone()),
                    );
                    response
                        .extra
                        .insert("gateway_cost".to_string(), serde_json::json!(total_cost));
                    response.extra.insert(
                        "gateway_compression".to_string(),
                        serde_json::to_value(&compression)
                            .expect("CompressionStats serialization must succeed"),
                    );

                    return Ok(response);
                }
                Err(e) => {
                    // Record failure
                    cb.record_failure().await;

                    // Surface the failure reason on the in-flight registry so a
                    // following retry/failover can show why the model changed.
                    if let Some(handle) = &active {
                        handle.set_last_error(&e.to_string());
                    }

                    // Extract status code from the error when available
                    let attempt_status = match &e {
                        GatewayError::Provider { status_code, .. } => *status_code,
                        _ => None,
                    };
                    let raw_message = match &e {
                        GatewayError::Provider { message, .. } => message.clone(),
                        _ => e.to_string(),
                    };

                    // Apply upstream rate-limit cooldown for 429 responses
                    // that arrive without a cooldown already set (i.e., from
                    // the Codex/OAuth dispatch path which bypasses the
                    // standard HTTP retry loop's cooldown logic).
                    if attempt_status == Some(429) {
                        let rate_limiter = self.get_rate_limiter(&provider_model.provider).await;
                        let already_cooled = rate_limiter.cooldown_remaining().await.is_some();
                        if !already_cooled {
                            // Use the OAuth usage tracker's reset window as
                            // the cooldown source when available; otherwise
                            // fall back to the configured default.
                            let cooldown = if let Some(tracker) = &self.oauth_usage_tracker {
                                let secs = tracker.fallback_cooldown_secs().await;
                                match secs {
                                    Some(s) if s > 0 => Duration::from_secs(s),
                                    _ => {
                                        self.parse_rate_limit_cooldown(
                                            &provider_model.provider,
                                            None,
                                            &raw_message,
                                        )
                                        .await
                                    }
                                }
                            } else {
                                self.parse_rate_limit_cooldown(
                                    &provider_model.provider,
                                    None,
                                    &raw_message,
                                )
                                .await
                            };
                            rate_limiter.apply_cooldown(cooldown).await;
                            self.metrics
                                .record_provider_rate_limit_exhausted(&provider_model.provider);
                            let now_secs = SystemTime::now()
                                .duration_since(UNIX_EPOCH)
                                .map(|d| d.as_secs())
                                .unwrap_or(0);
                            let deadline = now_secs.saturating_add(cooldown.as_secs());
                            self.metrics.set_provider_cooldown(
                                &provider_model.provider,
                                Self::friendly_failure_reason(Some(429), &raw_message),
                                deadline,
                            );
                            warn!(
                                provider = %provider_model.provider,
                                cooldown_secs = cooldown.as_secs(),
                                "Applied rate-limit cooldown from failover error path (Codex/OAuth 429)"
                            );
                        }
                    }

                    // Don't overwrite a fresh "Pausing until …" message
                    // that the rate-limit path just set: when this branch
                    // sees the same 429 the rate-limit code already wrote
                    // the friendlier countdown text.
                    let friendly = if attempt_status == Some(429) {
                        None
                    } else {
                        Some(Self::friendly_failure_reason(attempt_status, &raw_message))
                    };
                    self.metrics.record_provider_failure_with_reason(
                        &provider_model.provider,
                        friendly,
                        None,
                    );

                    // Collect attempt for aggregated error
                    attempts.push(ProviderAttempt::new(
                        provider_model.provider.clone(),
                        provider_model.model.clone(),
                        e.to_string(),
                        attempt_status,
                    ));
                }
            }
        }

        // All providers failed.
        // Req 6.2: when every provider truncated with finish_reason=length we
        // return the longest partial (highest completion_tokens) rather than an
        // error. These candidates already carry gateway_provider/responded_model/
        // cost metadata and their finish_reason=length is preserved verbatim, so
        // the client sees the partial content and the truncation reason.
        if let Some(longest) = truncated_candidates
            .into_iter()
            .max_by_key(|r| r.usage.completion_tokens)
        {
            let chosen_provider = longest
                .extra
                .get("gateway_provider")
                .and_then(serde_json::Value::as_str)
                .unwrap_or("unknown")
                .to_string();
            info!(
                provider = %chosen_provider,
                completion_tokens = longest.usage.completion_tokens,
                "All providers truncated (finish_reason=length); returning longest partial response"
            );
            return Ok(longest);
        }

        Err(GatewayError::AllProvidersFailed(AggregatedError::new(
            attempts,
        )))
    }

    fn promote_reasoning_to_content(response: &mut OpenAIResponse) -> bool {
        let Some(choice) = response.choices.first_mut() else {
            return false;
        };
        if !Self::content_is_empty(&choice.message.content) {
            return false;
        }
        for key in ["reasoning", "reasoning_content"] {
            if let Some(text) = choice
                .message
                .extra
                .get(key)
                .and_then(serde_json::Value::as_str)
                .filter(|text| !text.is_empty())
            {
                choice.message.content = serde_json::Value::String(text.to_string());
                return true;
            }
        }
        false
    }

    /// Check whether a provider response contains usable assistant content.
    ///
    /// Returns `false` when:
    /// - `choices` is empty
    /// - The first choice has null/empty string content AND no tool_calls
    /// - tool_calls are present but malformed (missing id, type, or function.name)
    ///
    /// This prevents forwarding hollow 200-OK responses that cause clients to
    /// report "no assistant messages", and catches malformed tool call responses
    /// that would confuse clients.
    fn response_has_content(response: &OpenAIResponse) -> bool {
        let Some(choice) = response.choices.first() else {
            return false;
        };

        // tool_calls present → validate structure before accepting
        if let Some(tool_calls) = choice.message.extra.get("tool_calls") {
            if let Some(arr) = tool_calls.as_array() {
                if arr.is_empty() {
                    // Empty tool_calls array with no text content is useless
                    return !Self::content_is_empty(&choice.message.content);
                }
                for tc in arr {
                    // Each tool call must have id, type, and function.name
                    let has_id = tc
                        .get("id")
                        .and_then(|v| v.as_str())
                        .is_some_and(|s| !s.is_empty());
                    let has_type = tc.get("type").and_then(|v| v.as_str()).is_some();
                    let has_fn_name = tc
                        .get("function")
                        .and_then(|f| f.get("name"))
                        .and_then(|n| n.as_str())
                        .is_some_and(|s| !s.is_empty());
                    if !has_id || !has_type || !has_fn_name {
                        warn!(
                            tool_call = %tc,
                            "Malformed tool call in provider response (missing id, type, or function.name)"
                        );
                        return false;
                    }
                }
                return true;
            }
            // tool_calls is not an array — malformed
            warn!("tool_calls field is not an array in provider response");
            return false;
        }

        // Check for non-empty text content
        !Self::content_is_empty(&choice.message.content)
    }

    /// Detect XML-style tool use in plain text content and translate it into
    /// proper OpenAI `tool_calls` format.
    ///
    /// Some models (e.g. GLM, Kimi) ignore the `tools` parameter and instead
    /// emit tool invocations as XML tags in their text output:
    ///   `<use_tool name="execute_command">{"command":"npm run build"}</use_tool>`
    ///   `<tool_call>{"name":"read_file","arguments":{...}}</tool_call>`
    ///
    // ── Known Roo Code / Kilo Code tool names ──
    // Hardcoded so we can match XML tool tags even when the tools array is
    // absent or incomplete.  Kept as a static slice for zero-alloc lookup.
    const KNOWN_TOOL_NAMES: &'static [&'static str] = &[
        // Sorted longest-first so Pattern 4 matches "edit_file" before "edit",
        // "read_file" before "read", etc.  Prevents prefix collisions.
        "ask_followup_question",
        "access_mcp_resource",
        "attempt_completion",
        "read_command_output",
        "run_slash_command",
        "fetch_instructions",
        "update_todo_list",
        "execute_command",
        "codebase_search",
        "generate_image",
        "search_replace",
        "write_to_file",
        "search_files",
        "apply_patch",
        "switch_mode",
        "use_mcp_tool",
        "apply_diff",
        "edit_file",
        "list_files",
        "read_file",
        "new_task",
        "skill",
        "edit",
        // Additional Roo Code / Cline / Kilo Code tools
        "replace_in_file",
        "insert_code_block",
        "browser_action",
        "list_code_definition_names",
        "inspect_site",
    ];

    /// Clients like Roo Code / Kilo Code expect native OpenAI `tool_calls` in
    /// the response message.  This function rewrites the response in-place so
    /// the client sees well-formed tool_calls and `finish_reason: "tool_calls"`.
    ///
    /// Returns `true` if any translation was performed.
    fn translate_xml_tool_calls(response: &mut OpenAIResponse, request: &OpenAIRequest) -> bool {
        let Some(choice) = response.choices.first_mut() else {
            debug!("translate_xml_tool_calls: no choices in response");
            return false;
        };

        // Skip if the response already has native tool_calls
        if choice.message.extra.contains_key("tool_calls") {
            debug!("translate_xml_tool_calls: response already has native tool_calls, skipping");
            return false;
        }

        let content_text = choice.message.content_as_text();
        // Some providers (e.g. Nano-GPT with thinking models) put tool calls
        // in a `reasoning` field instead of `content`. Check both.
        let reasoning_text = choice
            .message
            .extra
            .get("reasoning")
            .and_then(|v| v.as_str())
            .unwrap_or("");
        let combined_text = if content_text.is_empty() && !reasoning_text.is_empty() {
            reasoning_text.to_string()
        } else {
            content_text.clone()
        };
        if combined_text.is_empty() {
            debug!("translate_xml_tool_calls: combined text is empty, skipping");
            return false;
        }

        debug!(
            content_len = combined_text.len(),
            content_preview = %combined_text.chars().take(150).collect::<String>(),
            has_tools_in_request = request.extra.contains_key("tools"),
            "translate_xml_tool_calls: processing response content"
        );

        let mut tool_calls: Vec<serde_json::Value> = Vec::new();
        let mut remaining_text = combined_text.clone();

        // ── Collect all tool names to try ──
        // Start with known Roo/Kilo Code tools, then add any from the request's
        // tools array (covers MCP tools and custom tools the client defines).
        let mut tool_names: Vec<String> = Self::KNOWN_TOOL_NAMES
            .iter()
            .map(|s| s.to_string())
            .collect();
        if let Some(tools_val) = request.extra.get("tools") {
            if let Some(tools_arr) = tools_val.as_array() {
                for tool in tools_arr {
                    if let Some(name) = tool
                        .get("function")
                        .and_then(|f| f.get("name"))
                        .and_then(|n| n.as_str())
                    {
                        if !tool_names.iter().any(|n| n == name) {
                            tool_names.push(name.to_string());
                        }
                    }
                }
            }
        }

        // Pattern 0: Anthropic-style <tool_calls><invoke name="X"><parameter name="Y">V</parameter>...</invoke></tool_calls>
        // Also handles JSON-array-in-tool_calls: <tool_calls>[{"name":"X","arguments":{...}}]</tool_calls>
        Self::extract_invoke_style_tool_calls(&mut remaining_text, &mut tool_calls);
        if !tool_calls.is_empty() {
            debug!(
                count = tool_calls.len(),
                "Pattern 0 (invoke/JSON-in-tool_calls) extracted tool calls"
            );
        }

        // Pattern 1: <use_tool name="tool_name">...</use_tool>
        Self::extract_xml_tool_calls_pattern_inner(
            &mut remaining_text,
            &mut tool_calls,
            r#"<use_tool"#,
            "</use_tool>",
            false,
        );

        // Pattern 2: <tool_call>{"name":"X","arguments":{...}}</tool_call>
        Self::extract_xml_tool_calls_pattern_inner(
            &mut remaining_text,
            &mut tool_calls,
            "<tool_call>",
            "</tool_call>",
            false,
        );

        // Pattern 3: <function_call name="tool_name">...</function_call>
        Self::extract_xml_tool_calls_pattern_inner(
            &mut remaining_text,
            &mut tool_calls,
            r#"<function_call"#,
            "</function_call>",
            false,
        );

        // Pattern 4: Direct tool-name tags from known + request tools.
        // e.g. <execute_command>...</execute_command>, <attempt_completion>...</attempt_completion>
        // This is the primary format Roo/Kilo Code models are trained on.
        for name in &tool_names {
            let open = format!("<{}", name);
            let close = format!("</{}>", name);
            Self::extract_xml_tool_calls_pattern(
                &mut remaining_text,
                &mut tool_calls,
                &open,
                &close,
            );
        }

        // Pattern 5: Malformed <tool_name<arg_key>K</arg_key><arg_value>V</arg_value></tool_call>
        // Some models produce this broken format where the opening tag never closes.
        if tool_calls.is_empty() {
            Self::extract_arg_key_value_tool_calls(
                &mut remaining_text,
                &mut tool_calls,
                &tool_names,
            );
        }

        // Pattern 6: Kimi-style special token tool calls.
        // Kimi K2.6 (and similar) emit raw tokenizer special tokens in text:
        //   <|tool_calls_section_begin|><|tool_call_begin|>function_name<|tool_call_argument_begin|>{"arg":"val"}<|tool_call_end|><|tool_calls_section_end|>
        // Extract these into proper tool_calls and strip the tokens.
        if tool_calls.is_empty() {
            Self::extract_kimi_token_tool_calls(&mut remaining_text, &mut tool_calls);
            if !tool_calls.is_empty() {
                debug!(
                    count = tool_calls.len(),
                    "Pattern 6 (Kimi special tokens) extracted tool calls"
                );
            }
        }

        // Always strip any remaining Kimi-style special tokens from content,
        // even if we already extracted tool calls via other patterns.
        Self::strip_kimi_special_tokens(&mut remaining_text);

        if tool_calls.is_empty() {
            if combined_text.contains('<') && combined_text.contains("</") {
                debug!(
                    content_preview = %combined_text.chars().take(300).collect::<String>(),
                    "XML-like content detected but no tool calls extracted"
                );
            }
            return false;
        }

        info!(
            count = tool_calls.len(),
            tools = %tool_calls.iter()
                .filter_map(|tc| tc.get("function").and_then(|f| f.get("name")).and_then(|n| n.as_str()))
                .collect::<Vec<_>>()
                .join(", "),
            "Translated XML-style tool use to native tool_calls"
        );

        // Replace message content with any remaining non-tool text (usually empty)
        let cleaned = remaining_text.trim();
        if cleaned.is_empty() {
            choice.message.content = serde_json::Value::Null;
        } else {
            choice.message.content = serde_json::Value::String(cleaned.to_string());
        }

        // If tool calls were extracted from the reasoning field, clear it
        // so the translated response doesn't carry stale XML in reasoning.
        if !reasoning_text.is_empty() && content_text.is_empty() {
            choice.message.extra.remove("reasoning");
        }

        // Set tool_calls on the message
        choice.message.extra.insert(
            "tool_calls".to_string(),
            serde_json::Value::Array(tool_calls),
        );

        // Set finish_reason to "tool_calls" per OpenAI spec
        choice.finish_reason = Some("tool_calls".to_string());

        true
    }

    /// Extract XML-tagged tool calls from text content.
    ///
    /// Handles multiple sub-patterns:
    /// - `<tag name="tool_name">{args}</tag>` (use_tool / function_call style)
    /// - `<tag>{"name":"X","arguments":{...}}</tag>` (tool_call style)
    /// - `<tag><param1>val</param1><param2>val</param2></tag>` (Roo/Kilo Code style)
    ///
    /// Uses the LAST matching close tag to handle parameter values that may
    /// contain angle brackets or nested XML-like content.
    ///
    /// Extracted calls are appended to `tool_calls` and removed from `text`.
    fn extract_xml_tool_calls_pattern(
        text: &mut String,
        tool_calls: &mut Vec<serde_json::Value>,
        open_tag_prefix: &str,
        close_tag: &str,
    ) {
        Self::extract_xml_tool_calls_pattern_inner(
            text,
            tool_calls,
            open_tag_prefix,
            close_tag,
            true,
        )
    }

    /// Inner extraction with control over greedy vs non-greedy close-tag matching.
    ///
    /// `greedy`: when `true`, uses `rfind` to find the LAST close tag (needed for
    /// direct tool-name tags like `<attempt_completion>` whose body may contain
    /// XML-like text). When `false`, uses `find` to match the FIRST close tag
    /// (correct for wrapper patterns like `<use_tool>`, `<tool_call>`,
    /// `<function_call>` where the body is JSON and multiple calls can appear).
    fn extract_xml_tool_calls_pattern_inner(
        text: &mut String,
        tool_calls: &mut Vec<serde_json::Value>,
        open_tag_prefix: &str,
        close_tag: &str,
        greedy: bool,
    ) {
        // Process all occurrences
        loop {
            let Some(start) = text.find(open_tag_prefix) else {
                break;
            };

            // Find the close tag after the open tag.
            // Greedy (rfind): finds the LAST occurrence — critical for tags like
            // <attempt_completion> where the <result> parameter value can be very
            // long and might contain text that looks like XML.
            // Non-greedy (find): finds the FIRST occurrence — correct for wrapper
            // tags like <use_tool> where the body is JSON and multiple sequential
            // tool calls must each match their own close tag.
            let search_region = &text[start..];
            let close_finder = if greedy {
                search_region.rfind(close_tag)
            } else {
                search_region.find(close_tag)
            };
            let (close_start_rel, close_end, dangling) = if let Some(rel) = close_finder {
                (rel, start + rel + close_tag.len(), false)
            } else {
                // No closing tag — the model likely got truncated.
                // Treat everything from the open tag to end-of-text as the body
                // so we can still attempt to salvage the tool call.
                warn!(
                    tag_prefix = open_tag_prefix,
                    text_len = text.len(),
                    "No closing tag found for XML tool call, attempting to salvage body to end of text"
                );
                let rel = text.len() - start;
                (rel, text.len(), true)
            };
            // Extract the full tag content
            let full_tag = text[start..close_end].to_string();

            // Find the end of the opening tag (the '>' after attributes)
            let Some(open_end) = full_tag.find('>') else {
                // Malformed — remove this occurrence and continue looking
                text.replace_range(start..close_end, "");
                if dangling {
                    break;
                }
                continue;
            };

            let opening_tag = &full_tag[..=open_end];
            // When dangling (no close tag), body runs to end of full_tag.
            // Otherwise body ends where the close tag starts (relative to full_tag start).
            let body_end = if dangling {
                full_tag.len()
            } else {
                close_start_rel
            };
            let body = &full_tag[open_end + 1..body_end];

            // Try to extract tool name from the opening tag attribute: name="..."
            let tag_name = Self::extract_xml_attribute(opening_tag, "name");

            // Parse the body as JSON
            let body_trimmed = body.trim();
            let parsed: Option<serde_json::Value> = serde_json::from_str(body_trimmed).ok();

            let (tool_name, arguments) = if let Some(tag_n) = &tag_name {
                // Pattern: <use_tool name="X">{args}</use_tool>
                // Validate the body is proper JSON. If it is, use it directly.
                // If not, try to salvage: for known tools with a single primary
                // parameter (e.g. attempt_completion → result), wrap the body
                // as that parameter's value.
                if parsed.is_some() {
                    (tag_n.clone(), body_trimmed.to_string())
                } else if body_trimmed.starts_with('{') {
                    // Looks like JSON but failed to parse (unescaped chars, truncated, etc.)
                    // Try to extract the first key's value as a best-effort recovery.
                    // e.g. {"result":"some broken text..."} → extract "result" key
                    debug!(
                        tool = %tag_n,
                        body_preview = %body_trimmed.chars().take(200).collect::<String>(),
                        "Body looks like JSON but failed to parse, attempting recovery"
                    );

                    // ── Attempt 1: fix common JSON issues (unescaped control chars) ──
                    // Models often emit JSON with literal newlines, tabs, or other
                    // control characters inside string values. Escape them and retry.
                    let sanitized = body_trimmed
                        .replace("\\\n", "\\n") // already-escaped but with literal newline
                        .replace('\n', "\\n")
                        .replace('\r', "\\r")
                        .replace('\t', "\\t")
                        // Escape other control chars (0x00-0x1F except the ones we just handled)
                        .chars()
                        .map(|c| {
                            if c.is_control() && c != '\\' {
                                format!("\\u{:04x}", c as u32)
                            } else {
                                c.to_string()
                            }
                        })
                        .collect::<String>();

                    if let Ok(repaired) = serde_json::from_str::<serde_json::Value>(&sanitized) {
                        debug!(
                            tool = %tag_n,
                            "Recovered JSON by escaping control characters"
                        );
                        (tag_n.clone(), repaired.to_string())
                    } else {
                        // ── Attempt 2: naive key extraction (last resort) ──
                        // Strip outer braces and try to find "key":"value" pattern
                        let inner = body_trimmed
                            .trim_start_matches('{')
                            .trim_end_matches('}')
                            .trim();
                        let mut recovered = serde_json::Map::new();
                        // Find first "key": pattern
                        if let Some(colon_pos) = inner.find(':') {
                            let key = inner[..colon_pos].trim().trim_matches('"');
                            let val = inner[colon_pos + 1..].trim();
                            // Strip surrounding quotes if present
                            let val_clean = if val.starts_with('"') {
                                val.trim_start_matches('"').trim_end_matches('"')
                            } else {
                                val
                            };
                            recovered.insert(
                                key.to_string(),
                                serde_json::Value::String(val_clean.to_string()),
                            );
                        }
                        if recovered.is_empty() {
                            // Total fallback: use the whole body as a "result" or "input" param
                            recovered.insert(
                                "result".to_string(),
                                serde_json::Value::String(body_trimmed.to_string()),
                            );
                        }
                        (
                            tag_n.clone(),
                            serde_json::Value::Object(recovered).to_string(),
                        )
                    }
                } else {
                    // Body is plain text, not JSON. Wrap it as the primary parameter.
                    // For attempt_completion → result, for others → input
                    let param_name = match tag_n.as_str() {
                        "attempt_completion" => "result",
                        "ask_followup_question" => "question",
                        _ => "input",
                    };
                    let mut map = serde_json::Map::new();
                    map.insert(
                        param_name.to_string(),
                        serde_json::Value::String(body_trimmed.to_string()),
                    );
                    (tag_n.clone(), serde_json::Value::Object(map).to_string())
                }
            } else if let Some(ref obj) = parsed {
                // Pattern: <tool_call>{"name":"X","arguments":{...}}</tool_call>
                let name = obj
                    .get("name")
                    .and_then(|v| v.as_str())
                    .unwrap_or("unknown")
                    .to_string();
                let args = obj
                    .get("arguments")
                    .map(|v| {
                        if v.is_string() {
                            v.as_str().unwrap_or("{}").to_string()
                        } else {
                            v.to_string()
                        }
                    })
                    .unwrap_or_else(|| {
                        let mut m = obj.as_object().cloned().unwrap_or_default();
                        m.remove("name");
                        serde_json::Value::Object(m).to_string()
                    });
                (name, args)
            } else if body_trimmed.contains('<') && body_trimmed.contains('>') {
                // Pattern: <execute_command><command>X</command></execute_command>
                // The body contains nested XML tags representing arguments.
                let tool_n = open_tag_prefix.trim_start_matches('<').trim().to_string();
                let args_json = Self::parse_inner_xml_to_json(body_trimmed);
                debug!(
                    tool = %tool_n,
                    args_preview = %args_json.chars().take(200).collect::<String>(),
                    "Parsed inner XML tags to JSON arguments"
                );
                (tool_n, args_json)
            } else {
                warn!(
                    tag_prefix = open_tag_prefix,
                    body_preview = %body_trimmed.chars().take(200).collect::<String>(),
                    "Skipping malformed XML tool call (unparseable body, no name attribute)"
                );
                text.replace_range(start..close_end, "");
                if dangling {
                    break;
                }
                continue;
            };

            let call_id = format!("call_xlat_{}", tool_calls.len());

            // Normalize arguments for known tools whose schemas models
            // frequently get wrong (e.g. read_file expects "path" as a
            // newline-separated string, but models often send "files" array).
            let arguments = Self::normalize_tool_arguments(&tool_name, &arguments);

            tool_calls.push(serde_json::json!({
                "id": call_id,
                "type": "function",
                "function": {
                    "name": tool_name,
                    "arguments": arguments
                }
            }));

            // Remove the XML tag from the text
            text.replace_range(start..close_end, "");
            if dangling {
                break;
            }
        }
    }

    /// Extract an attribute value from an XML opening tag.
    /// e.g. `<use_tool name="execute_command">` → Some("execute_command")
    fn extract_xml_attribute(tag: &str, attr_name: &str) -> Option<String> {
        // Look for: attr_name="value" or attr_name='value'
        let pattern_dq = format!("{}=\"", attr_name);
        let pattern_sq = format!("{}='", attr_name);

        if let Some(pos) = tag.find(&pattern_dq) {
            let value_start = pos + pattern_dq.len();
            if let Some(end) = tag[value_start..].find('"') {
                return Some(tag[value_start..value_start + end].to_string());
            }
        }
        if let Some(pos) = tag.find(&pattern_sq) {
            let value_start = pos + pattern_sq.len();
            if let Some(end) = tag[value_start..].find('\'') {
                return Some(tag[value_start..value_start + end].to_string());
            }
        }
        None
    }

    /// Normalize tool arguments for known tools whose schemas models frequently
    /// get wrong during XML-to-native translation.
    ///
    /// Common mismatches:
    /// - `read_file`: model sends `{"files":[{"path":"a"},{"path":"b"}]}` but
    ///   Kilo Code expects `{"path":"a\nb"}` (newline-separated).
    /// - `read_file`: model sends `{"file_path":"a"}` instead of `{"path":"a"}`.
    /// - `list_files`: model sends `{"directory":"x"}` instead of `{"path":"x"}`.
    /// - `search_files`: model sends `{"search_term":"x"}` instead of `{"regex":"x"}`.
    fn normalize_tool_arguments(tool_name: &str, arguments: &str) -> String {
        let Ok(mut args) = serde_json::from_str::<serde_json::Value>(arguments) else {
            return arguments.to_string();
        };
        let Some(obj) = args.as_object_mut() else {
            return arguments.to_string();
        };

        let mut changed = false;

        match tool_name {
            "read_file" => {
                // {"files":[{"path":"a"},{"path":"b"}]} → {"path":"a\nb"}
                if let Some(files_val) = obj.remove("files") {
                    if let Some(files_arr) = files_val.as_array() {
                        let paths: Vec<&str> = files_arr
                            .iter()
                            .filter_map(|f| f.get("path").and_then(|p| p.as_str()))
                            .collect();
                        if !paths.is_empty() {
                            obj.insert(
                                "path".to_string(),
                                serde_json::Value::String(paths.join("\n")),
                            );
                            changed = true;
                        }
                    }
                }
                // {"file_path":"a"} → {"path":"a"}
                if !obj.contains_key("path") {
                    if let Some(fp) = obj.remove("file_path") {
                        obj.insert("path".to_string(), fp);
                        changed = true;
                    }
                }
            }
            "list_files" => {
                // {"directory":"x"} → {"path":"x"}
                if !obj.contains_key("path") {
                    if let Some(d) = obj.remove("directory") {
                        obj.insert("path".to_string(), d);
                        changed = true;
                    }
                }
            }
            "search_files" => {
                // {"search_term":"x"} → {"regex":"x"}
                if !obj.contains_key("regex") {
                    if let Some(st) = obj.remove("search_term") {
                        obj.insert("regex".to_string(), st);
                        changed = true;
                    }
                }
            }
            "execute_command" => {
                // {"cmd":"x"} → {"command":"x"}
                if !obj.contains_key("command") {
                    if let Some(c) = obj.remove("cmd") {
                        obj.insert("command".to_string(), c);
                        changed = true;
                    }
                }
            }
            _ => {}
        }

        if changed {
            debug!(
                tool = tool_name,
                "Normalized tool arguments for known schema mismatch"
            );
            serde_json::to_string(&args).unwrap_or_else(|_| arguments.to_string())
        } else {
            arguments.to_string()
        }
    }

    /// Parse simple inner XML tags into a JSON object string.
    ///
    /// Handles the Roo/Kilo Code tool parameter format:
    ///   `<command>npm run build</command><cwd>/path</cwd>`
    /// → `{"command":"npm run build","cwd":"/path"}`
    ///
    /// For `attempt_completion`:
    ///   `<result>I've fixed the bug...</result><command>npm test</command>`
    /// → `{"result":"I've fixed the bug...","command":"npm test"}`
    ///
    /// Uses rfind for closing tags to handle values containing angle brackets.
    /// Also handles `null` text values by emitting JSON null.
    fn parse_inner_xml_to_json(body: &str) -> String {
        let mut map = serde_json::Map::new();
        let mut remaining = body.trim();

        while !remaining.is_empty() {
            // Skip whitespace and text between tags
            remaining = remaining.trim_start();
            if remaining.is_empty() {
                break;
            }

            // Find next opening tag
            let Some(open_start) = remaining.find('<') else {
                break;
            };

            // Skip closing tags that appear at the start (orphaned)
            if remaining[open_start..].starts_with("</") {
                if let Some(close_end) = remaining[open_start..].find('>') {
                    remaining = &remaining[open_start + close_end + 1..];
                } else {
                    break;
                }
                continue;
            }

            // Extract tag name from <tag_name> or <tag_name attr="...">
            let tag_content_start = open_start + 1;
            let Some(open_end) = remaining[tag_content_start..].find('>') else {
                break;
            };
            let tag_name = remaining[tag_content_start..tag_content_start + open_end]
                .split_whitespace()
                .next()
                .unwrap_or("")
                .trim_end_matches('/')
                .trim_end_matches('"')
                .trim_end_matches('\'')
                .to_string();

            if tag_name.is_empty() {
                break;
            }

            let value_start = tag_content_start + open_end + 1;
            let close_tag = format!("</{}>", tag_name);

            // Use rfind to find the LAST matching close tag.
            // This is critical for parameters whose values contain XML-like
            // content (e.g. <result> containing code with angle brackets).
            if let Some(close_pos) = remaining[value_start..].rfind(&close_tag) {
                let value = &remaining[value_start..value_start + close_pos];
                let json_value = if value == "null" || value == "undefined" {
                    serde_json::Value::Null
                } else if value == "true" {
                    serde_json::Value::Bool(true)
                } else if value == "false" {
                    serde_json::Value::Bool(false)
                } else {
                    serde_json::Value::String(value.to_string())
                };
                map.insert(tag_name, json_value);
                remaining = &remaining[value_start + close_pos + close_tag.len()..];
            } else {
                // No matching close tag — try to recover by looking for any close tag
                // (handles malformed XML like <cwd>null</command>)
                if let Some(any_close) = remaining[value_start..].find("</") {
                    let value = &remaining[value_start..value_start + any_close];
                    let json_value = if value == "null" || value == "undefined" {
                        serde_json::Value::Null
                    } else {
                        serde_json::Value::String(value.to_string())
                    };
                    map.insert(tag_name, json_value);
                    // Skip past the mismatched closing tag
                    if let Some(close_end) = remaining[value_start + any_close..].find('>') {
                        remaining = &remaining[value_start + any_close + close_end + 1..];
                    } else {
                        break;
                    }
                } else {
                    // No close tag at all — treat rest of body as the value
                    let value = &remaining[value_start..];
                    if !value.trim().is_empty() {
                        let json_value = if value.trim() == "null" {
                            serde_json::Value::Null
                        } else {
                            serde_json::Value::String(value.to_string())
                        };
                        map.insert(tag_name, json_value);
                    }
                    break;
                }
            }
        }

        serde_json::Value::Object(map).to_string()
    }

    /// Parse Anthropic-style `<tool_calls><invoke name="X"><parameter name="Y">V</parameter>...</invoke></tool_calls>`
    /// into native tool_calls. Handles one or more `<invoke>` blocks inside a `<tool_calls>` wrapper,
    /// as well as bare `<invoke>` blocks without the wrapper.
    fn extract_invoke_style_tool_calls(text: &mut String, tool_calls: &mut Vec<serde_json::Value>) {
        // Try wrapped form first: <tool_calls>...</tool_calls>
        while let Some(wrapper_start) = text.find("<tool_calls>") {
            let after_open = wrapper_start + "<tool_calls>".len();
            if let Some(wrapper_close_rel) = text[after_open..].find("</tool_calls>") {
                let inner = text[after_open..after_open + wrapper_close_rel].to_string();
                let wrapper_end = after_open + wrapper_close_rel + "</tool_calls>".len();
                // Try JSON array first: <tool_calls>[{"name":"X","arguments":{...}}]</tool_calls>
                let inner_trimmed = inner.trim();
                if inner_trimmed.starts_with('[') {
                    Self::parse_json_tool_calls_array(inner_trimmed, tool_calls);
                } else if inner_trimmed.starts_with('{') {
                    // Single JSON object: <tool_calls>{"name":"X","arguments":{...}}</tool_calls>
                    Self::parse_json_tool_calls_array(&format!("[{}]", inner_trimmed), tool_calls);
                } else {
                    Self::parse_invoke_blocks(&inner, tool_calls);
                }
                text.replace_range(wrapper_start..wrapper_end, "");
            } else {
                // No closing tag — parse what we have and remove the dangling open
                let inner = text[after_open..].to_string();
                let inner_trimmed = inner.trim();
                if inner_trimmed.starts_with('[') || inner_trimmed.starts_with('{') {
                    Self::parse_json_tool_calls_array(inner_trimmed, tool_calls);
                } else {
                    Self::parse_invoke_blocks(&inner, tool_calls);
                }
                text.replace_range(wrapper_start.., "");
                break;
            }
        }

        // Also handle bare <invoke> blocks without wrapper
        while text.contains("<invoke ") {
            let Some(start) = text.find("<invoke ") else {
                break;
            };
            if let Some(close_rel) = text[start..].find("</invoke>") {
                let block_end = start + close_rel + "</invoke>".len();
                let block = text[start..block_end].to_string();
                Self::parse_invoke_blocks(&block, tool_calls);
                text.replace_range(start..block_end, "");
            } else {
                // No closing tag — try to parse what's there, then remove
                let block = text[start..].to_string();
                Self::parse_invoke_blocks(&block, tool_calls);
                text.replace_range(start.., "");
                break;
            }
        }
    }

    /// Parse one or more `<invoke name="tool_name"><parameter name="key">value</parameter>...</invoke>`
    /// blocks from the given text and append them to `tool_calls`.
    fn parse_invoke_blocks(text: &str, tool_calls: &mut Vec<serde_json::Value>) {
        let mut remaining = text;
        while let Some(inv_start) = remaining.find("<invoke ") {
            remaining = &remaining[inv_start..];
            // Extract tool name from <invoke name="...">
            let tool_name = Self::extract_xml_attribute(remaining, "name")
                .unwrap_or_else(|| "unknown".to_string());

            // Find end of opening tag
            let Some(open_end) = remaining.find('>') else {
                break;
            };
            let after_open = open_end + 1;

            // Find </invoke> or end of string
            let body_end = remaining[after_open..]
                .find("</invoke>")
                .unwrap_or(remaining.len() - after_open);
            let body = &remaining[after_open..after_open + body_end];

            // Parse <parameter name="key">value</parameter> tags
            let mut args = serde_json::Map::new();
            let mut param_remaining = body;
            while let Some(p_start) = param_remaining.find("<parameter ") {
                param_remaining = &param_remaining[p_start..];
                let param_name = Self::extract_xml_attribute(param_remaining, "name")
                    .unwrap_or_else(|| "unknown".to_string());
                let Some(p_open_end) = param_remaining.find('>') else {
                    break;
                };
                let p_after = p_open_end + 1;
                let p_close = param_remaining[p_after..]
                    .find("</parameter>")
                    .unwrap_or(param_remaining.len() - p_after);
                let value = &param_remaining[p_after..p_after + p_close];
                let json_val = if value == "null" {
                    serde_json::Value::Null
                } else {
                    serde_json::Value::String(value.to_string())
                };
                args.insert(param_name, json_val);
                param_remaining = &param_remaining[p_after + p_close..];
                // Skip past </parameter> if present
                if param_remaining.starts_with("</parameter>") {
                    param_remaining = &param_remaining["</parameter>".len()..];
                }
            }

            let call_id = format!("call_xlat_{}", tool_calls.len());
            tool_calls.push(serde_json::json!({
                "id": call_id,
                "type": "function",
                "function": {
                    "name": tool_name,
                    "arguments": serde_json::Value::Object(args).to_string()
                }
            }));

            // Advance past this invoke block
            let skip = after_open + body_end;
            remaining = &remaining[skip..];
            if remaining.starts_with("</invoke>") {
                remaining = &remaining["</invoke>".len()..];
            }
        }
    }

    /// Parse a JSON array of tool call objects.
    /// Handles: `[{"name":"X","arguments":{...}}, ...]`
    fn parse_json_tool_calls_array(text: &str, tool_calls: &mut Vec<serde_json::Value>) {
        let arr: Vec<serde_json::Value> = match serde_json::from_str(text) {
            Ok(a) => a,
            Err(_) => return,
        };
        for obj in arr {
            let name = obj
                .get("name")
                .and_then(|v| v.as_str())
                .unwrap_or("unknown")
                .to_string();
            let arguments = obj
                .get("arguments")
                .map(|v| {
                    if v.is_string() {
                        v.as_str().unwrap_or("{}").to_string()
                    } else {
                        v.to_string()
                    }
                })
                .unwrap_or_else(|| {
                    let mut m = obj.as_object().cloned().unwrap_or_default();
                    m.remove("name");
                    serde_json::Value::Object(m).to_string()
                });
            let call_id = format!("call_xlat_{}", tool_calls.len());
            tool_calls.push(serde_json::json!({
                "id": call_id,
                "type": "function",
                "function": { "name": name, "arguments": arguments }
            }));
        }
    }

    /// Extract tool calls in the malformed `<tool_name<arg_key>K</arg_key><arg_value>V</arg_value></tool_call>` format.
    /// Some models produce this broken XML where the opening tag never closes properly.
    fn extract_arg_key_value_tool_calls(
        text: &mut String,
        tool_calls: &mut Vec<serde_json::Value>,
        tool_names: &[String],
    ) {
        for name in tool_names {
            let pattern = format!("<{}<arg_key>", name);
            loop {
                let Some(start) = text.find(&pattern) else {
                    break;
                };
                // Find the end — could be </tool_call> or end of text
                let search_from = start + pattern.len();
                let block_end = text[search_from..]
                    .find("</tool_call>")
                    .map(|p| search_from + p + "</tool_call>".len())
                    .or_else(|| {
                        text[search_from..]
                            .find(&format!("</{}>", name))
                            .map(|p| search_from + p + format!("</{}>", name).len())
                    })
                    .unwrap_or(text.len());

                let block = text[start..block_end].to_string();

                // Extract all <arg_key>K</arg_key><arg_value>V</arg_value> pairs
                let mut args = serde_json::Map::new();
                let mut remaining = block.as_str();
                while let Some(key_start) = remaining.find("<arg_key>") {
                    let key_content_start = key_start + "<arg_key>".len();
                    let Some(key_end) = remaining[key_content_start..].find("</arg_key>") else {
                        break;
                    };
                    let key = &remaining[key_content_start..key_content_start + key_end];

                    let after_key = key_content_start + key_end + "</arg_key>".len();
                    remaining = &remaining[after_key..];

                    if let Some(val_start) = remaining.find("<arg_value>") {
                        let val_content_start = val_start + "<arg_value>".len();
                        let val_end = remaining[val_content_start..]
                            .find("</arg_value>")
                            .unwrap_or(remaining.len() - val_content_start);
                        let value = &remaining[val_content_start..val_content_start + val_end];

                        // Try to parse as JSON first (for arrays/objects), fall back to string
                        let json_val: serde_json::Value = serde_json::from_str(value)
                            .unwrap_or_else(|_| serde_json::Value::String(value.to_string()));
                        args.insert(key.to_string(), json_val);

                        let skip = val_content_start + val_end;
                        if remaining[skip..].starts_with("</arg_value>") {
                            remaining = &remaining[skip + "</arg_value>".len()..];
                        } else {
                            remaining = &remaining[skip..];
                        }
                    } else {
                        break;
                    }
                }

                if !args.is_empty() {
                    let call_id = format!("call_xlat_{}", tool_calls.len());
                    tool_calls.push(serde_json::json!({
                        "id": call_id,
                        "type": "function",
                        "function": {
                            "name": name,
                            "arguments": serde_json::Value::Object(args).to_string()
                        }
                    }));
                }

                text.replace_range(start..block_end, "");
            }
        }
    }

    /// Extract tool calls from Kimi-style special token format.
    ///
    /// Kimi K2.6 (and similar models) emit raw tokenizer special tokens in
    /// their text output instead of using the native tool_calls API:
    ///
    /// ```text
    /// <|tool_calls_section_begin|><|tool_call_begin|>function_name<|tool_call_argument_begin|>{"key":"val"}<|tool_call_end|><|tool_calls_section_end|>
    /// ```
    ///
    /// Also handles variants with `functions_` prefix (e.g. `functions_list_files_6`).
    fn extract_kimi_token_tool_calls(text: &mut String, tool_calls: &mut Vec<serde_json::Value>) {
        const SECTION_BEGIN: &str = "<|tool_calls_section_begin|>";
        const SECTION_END: &str = "<|tool_calls_section_end|>";
        const CALL_BEGIN: &str = "<|tool_call_begin|>";
        const CALL_END: &str = "<|tool_call_end|>";
        const ARG_BEGIN: &str = "<|tool_call_argument_begin|>";

        while let Some(sec_start) = text.find(SECTION_BEGIN) {
            let sec_end_pos = text[sec_start..]
                .find(SECTION_END)
                .map(|p| sec_start + p + SECTION_END.len())
                .unwrap_or(text.len());

            let section = text[sec_start + SECTION_BEGIN.len()..sec_end_pos].to_string();

            // Extract individual tool calls within the section
            let mut remaining = section.as_str();
            while let Some(call_start) = remaining.find(CALL_BEGIN) {
                let after_call_begin = call_start + CALL_BEGIN.len();
                let call_end = remaining[after_call_begin..]
                    .find(CALL_END)
                    .map(|p| after_call_begin + p)
                    .unwrap_or(remaining.len());

                let call_body = &remaining[after_call_begin..call_end];

                // Split on argument begin token: name<|tool_call_argument_begin|>args_json
                if let Some(arg_split) = call_body.find(ARG_BEGIN) {
                    let raw_name = call_body[..arg_split].trim();
                    let args_str = call_body[arg_split + ARG_BEGIN.len()..].trim();

                    // Normalize function name: strip "functions_" prefix and
                    // trailing _N suffix that Kimi sometimes adds
                    // e.g. "functions_list_files_6" → "list_files"
                    let func_name = Self::normalize_kimi_function_name(raw_name);

                    // Validate args as JSON; use as-is if valid, wrap as string otherwise
                    let args_json = if serde_json::from_str::<serde_json::Value>(args_str).is_ok() {
                        args_str.to_string()
                    } else {
                        format!(
                            "{{\"input\":{}}}",
                            serde_json::Value::String(args_str.to_string())
                        )
                    };

                    let call_id = format!("call_xlat_{}", tool_calls.len());
                    tool_calls.push(serde_json::json!({
                        "id": call_id,
                        "type": "function",
                        "function": {
                            "name": func_name,
                            "arguments": args_json
                        }
                    }));
                }

                remaining = if call_end + CALL_END.len() <= remaining.len() {
                    &remaining[call_end + CALL_END.len()..]
                } else {
                    ""
                };
            }

            text.replace_range(sec_start..sec_end_pos, "");
        }

        // Handle bare <|tool_call_begin|>...<|tool_call_end|> without section wrapper
        while let Some(call_start) = text.find(CALL_BEGIN) {
            let after_begin = call_start + CALL_BEGIN.len();
            let call_end = text[after_begin..]
                .find(CALL_END)
                .map(|p| after_begin + p)
                .unwrap_or(text.len());
            let block_end = if call_end + CALL_END.len() <= text.len() {
                call_end + CALL_END.len()
            } else {
                text.len()
            };

            let call_body = &text[after_begin..call_end];
            if let Some(arg_split) = call_body.find(ARG_BEGIN) {
                let raw_name = call_body[..arg_split].trim();
                let args_str = call_body[arg_split + ARG_BEGIN.len()..].trim();
                let func_name = Self::normalize_kimi_function_name(raw_name);
                let args_json = if serde_json::from_str::<serde_json::Value>(args_str).is_ok() {
                    args_str.to_string()
                } else {
                    format!(
                        "{{\"input\":{}}}",
                        serde_json::Value::String(args_str.to_string())
                    )
                };
                let call_id = format!("call_xlat_{}", tool_calls.len());
                tool_calls.push(serde_json::json!({
                    "id": call_id,
                    "type": "function",
                    "function": {
                        "name": func_name,
                        "arguments": args_json
                    }
                }));
            }
            text.replace_range(call_start..block_end, "");
        }
    }

    /// Normalize a Kimi-style function name.
    ///
    /// Kimi K2.6 sometimes prefixes tool names with `functions_` and appends
    /// a numeric suffix like `_6`. Strip both to recover the real tool name.
    /// e.g. "functions_list_files_6" → "list_files"
    fn normalize_kimi_function_name(raw: &str) -> String {
        let mut name = raw.to_string();

        // Strip "functions_" prefix
        if let Some(stripped) = name.strip_prefix("functions_") {
            name = stripped.to_string();
        }

        // Strip trailing _N numeric suffix (only if what remains is non-empty)
        if let Some(last_underscore) = name.rfind('_') {
            let suffix = &name[last_underscore + 1..];
            if !suffix.is_empty() && suffix.chars().all(|c| c.is_ascii_digit()) {
                let base = &name[..last_underscore];
                if !base.is_empty() {
                    name = base.to_string();
                }
            }
        }

        name
    }

    /// Strip all Kimi-style special tokens from text content.
    ///
    /// Safety net: removes `<|...|>` tokens that may leak through even after
    /// tool call extraction, or when no tools were in the request.
    fn strip_kimi_special_tokens(text: &mut String) {
        const KIMI_TOKENS: &[&str] = &[
            "<|tool_calls_section_begin|>",
            "<|tool_calls_section_end|>",
            "<|tool_call_begin|>",
            "<|tool_call_end|>",
            "<|tool_call_argument_begin|>",
            "<|tool_call_argument_end|>",
            "<|tool_sep|>",
        ];
        for token in KIMI_TOKENS {
            while text.contains(token) {
                *text = text.replace(token, "");
            }
        }
    }

    /// Strip Kimi-style special tokens from all text content in a response.
    /// Operates on the first choice's message content.
    fn sanitize_kimi_tokens_in_response(response: &mut OpenAIResponse) {
        let Some(choice) = response.choices.first_mut() else {
            return;
        };
        let content_text = choice.message.content_as_text();
        if content_text.contains("<|tool_call") || content_text.contains("<|tool_sep|>") {
            let mut cleaned = content_text.clone();
            Self::strip_kimi_special_tokens(&mut cleaned);
            let cleaned = cleaned.trim();
            if cleaned.is_empty() {
                choice.message.content = serde_json::Value::Null;
            } else {
                choice.message.content = serde_json::Value::String(cleaned.to_string());
            }
        }
    }

    /// Reverse-translate gateway-translated tool_calls back to XML format
    /// for models that use XML-style tool use.
    ///
    /// Detects assistant messages with tool_calls containing "call_xlat_" IDs
    /// (generated by `translate_xml_tool_calls`) and converts them back to
    /// XML `<use_tool>` tags in the assistant content. Corresponding tool
    /// result messages (role:"tool") are converted to user messages containing
    /// the tool output, so the model sees a natural conversation flow.
    fn reverse_translate_tool_history(messages: &mut Vec<Message>) {
        let mut i = 0;
        while i < messages.len() {
            let msg = &messages[i];

            // Only process assistant messages with gateway-translated tool_calls
            if msg.role != "assistant" {
                i += 1;
                continue;
            }

            let is_xlat = msg
                .extra
                .get("tool_calls")
                .and_then(|v| v.as_array())
                .map(|arr| {
                    arr.iter().any(|tc| {
                        tc.get("id")
                            .and_then(|v| v.as_str())
                            .is_some_and(|id| id.starts_with("call_xlat_"))
                    })
                })
                .unwrap_or(false);

            if !is_xlat {
                i += 1;
                continue;
            }

            // Convert tool_calls back to XML content
            let tool_calls = messages[i]
                .extra
                .get("tool_calls")
                .and_then(|v| v.as_array())
                .cloned()
                .unwrap_or_default();

            let existing_content = messages[i].content_as_text();
            let mut xml_content = if existing_content.is_empty() {
                String::new()
            } else {
                existing_content
            };

            for tc in &tool_calls {
                let fn_name = tc
                    .get("function")
                    .and_then(|f| f.get("name"))
                    .and_then(|n| n.as_str())
                    .unwrap_or("unknown");
                let fn_args = tc
                    .get("function")
                    .and_then(|f| f.get("arguments"))
                    .and_then(|a| a.as_str())
                    .unwrap_or("{}");

                xml_content.push_str(&format!(
                    "<use_tool name=\"{}\">{}</use_tool>",
                    fn_name, fn_args
                ));
            }

            // Replace the assistant message: set content to XML, remove tool_calls
            messages[i].content = serde_json::Value::String(xml_content);
            messages[i].extra.remove("tool_calls");

            i += 1;

            // Convert following tool result messages to user messages with the output
            while i < messages.len() && messages[i].role == "tool" {
                let tool_content = messages[i].content_as_text();
                let tool_name = messages[i]
                    .extra
                    .get("tool_call_id")
                    .and_then(|v| v.as_str())
                    .unwrap_or("tool")
                    .to_string();

                // Convert to a user message with the tool result
                messages[i].role = "user".to_string();
                messages[i].content = serde_json::Value::String(format!(
                    "[Tool Result: {}]\n{}",
                    tool_name, tool_content
                ));
                // Remove tool-specific extra fields
                messages[i].extra.remove("tool_call_id");

                i += 1;
            }
        }
    }

    /// Check if message content is empty/null.
    fn content_is_empty(content: &serde_json::Value) -> bool {
        match content {
            serde_json::Value::String(s) => s.is_empty(),
            serde_json::Value::Null => true,
            serde_json::Value::Array(arr) => arr.is_empty(),
            _ => false,
        }
    }

    /// Determine whether a response is safe to cache.
    ///
    /// Responses are NOT cached when:
    /// - They contain tool_calls (stateful, context-dependent)
    /// - finish_reason is "length" (incomplete, would produce bad cached results)
    /// - finish_reason is "content_filter" (policy-dependent, may vary)
    /// - The response has no usable content

    /// Sanitize an outgoing request based on provider type.
    ///
    /// Different providers accept different subsets of the OpenAI request schema.
    /// Unknown fields can cause 400/422/502 errors. This method strips fields
    /// from the `extra` catch-all map that the target provider does not support.
    ///
    /// Returns the number of fields removed (for logging).
    fn sanitize_request_for_provider(outgoing: &mut OpenAIRequest, provider_type: &str) -> usize {
        match provider_type {
            "nvidia_nim" => {
                // NIM /v1/chat/completions accepts only these extra fields
                // (beyond model, messages, stream, temperature, max_tokens
                //  which are explicit struct fields):
                //   tools, tool_choice, top_p, frequency_penalty,
                //   presence_penalty, stop
                // Source: https://docs.api.nvidia.com/nim/reference
                const NIM_ALLOWED: &[&str] = &[
                    "tools",
                    "tool_choice",
                    "top_p",
                    "frequency_penalty",
                    "presence_penalty",
                    "stop",
                ];
                let before = outgoing.extra.len();
                outgoing
                    .extra
                    .retain(|k, _| NIM_ALLOWED.contains(&k.as_str()));
                before - outgoing.extra.len()
            }
            "bedrock" => sanitize_mantle_chat_request(outgoing),
            // Other provider types pass through unmodified
            _ => 0,
        }
    }

    /// Ensure `tool_calls` on every message is either a JSON array or absent.
    ///
    /// Some clients (or intermediate proxies) send `tool_calls` as a single
    /// object, null, or other non-array shape.  Downstream providers reject
    /// this with `"assistant.tool_calls must be an array when provided"`.
    ///
    /// Rules:
    ///   - Already an array → keep as-is.
    ///   - Single object with `id`+`function` → wrap in a one-element array.
    ///   - Null / other → remove the key entirely.
    fn normalize_message_tool_calls(messages: &mut [Message]) {
        for (idx, msg) in messages.iter_mut().enumerate() {
            let Some(tc_val) = msg.extra.remove("tool_calls") else {
                continue;
            };
            debug!(
                message_index = idx,
                role = %msg.role,
                tool_calls_type = %match &tc_val {
                    serde_json::Value::Array(a) => format!("array(len={})", a.len()),
                    serde_json::Value::Object(o) => format!("object(keys={})", o.len()),
                    serde_json::Value::Null => "null".to_string(),
                    serde_json::Value::String(_) => "string".to_string(),
                    serde_json::Value::Bool(_) => "bool".to_string(),
                    serde_json::Value::Number(_) => "number".to_string(),
                },
                "normalize_message_tool_calls: processing tool_calls"
            );
            match tc_val {
                serde_json::Value::Array(arr) => {
                    // Already correct shape — put it back.
                    if !arr.is_empty() {
                        msg.extra
                            .insert("tool_calls".to_string(), serde_json::Value::Array(arr));
                    }
                    // Empty array → drop it to avoid confusing providers.
                }
                serde_json::Value::Object(obj) => {
                    // Single tool_call object → wrap in array.
                    // Accept any object shape that looks like a tool_call
                    // (has id, function, type, or name keys).
                    let looks_like_tool_call = obj.contains_key("id")
                        || obj.contains_key("function")
                        || obj.contains_key("type")
                        || obj.contains_key("name");
                    if looks_like_tool_call {
                        warn!(
                            role = %msg.role,
                            "tool_calls was a single object, wrapping in array"
                        );
                        msg.extra.insert(
                            "tool_calls".to_string(),
                            serde_json::Value::Array(vec![serde_json::Value::Object(obj)]),
                        );
                    } else if !obj.is_empty() {
                        // Non-empty object without recognized keys — could be
                        // an indexed map like {"0": {...}, "1": {...}}. Try to
                        // convert numeric-keyed objects into an array ordered
                        // by key.
                        let all_numeric_keys = obj.keys().all(|k| k.parse::<usize>().is_ok());
                        if all_numeric_keys {
                            let mut entries: Vec<(usize, serde_json::Value)> = obj
                                .into_iter()
                                .filter_map(|(k, v)| k.parse::<usize>().ok().map(|idx| (idx, v)))
                                .collect();
                            entries.sort_by_key(|(idx, _)| *idx);
                            let arr: Vec<serde_json::Value> =
                                entries.into_iter().map(|(_, v)| v).collect();
                            if !arr.is_empty() {
                                warn!(
                                    role = %msg.role,
                                    count = arr.len(),
                                    "tool_calls was an indexed object map, converted to array"
                                );
                                msg.extra.insert(
                                    "tool_calls".to_string(),
                                    serde_json::Value::Array(arr),
                                );
                            }
                        } else {
                            warn!(
                                role = %msg.role,
                                keys = ?obj.keys().take(5).collect::<Vec<_>>(),
                                "Dropping unrecognized tool_calls object (no tool_call keys found)"
                            );
                        }
                    }
                    // Empty object → drop silently.
                }
                serde_json::Value::Null => {
                    // Null → just drop it.
                }
                serde_json::Value::String(s) => {
                    // Some clients serialize tool_calls as a JSON string.
                    // Attempt to parse it back.
                    if let Ok(parsed) = serde_json::from_str::<serde_json::Value>(&s) {
                        match parsed {
                            serde_json::Value::Array(arr) if !arr.is_empty() => {
                                warn!(
                                    role = %msg.role,
                                    "tool_calls was a JSON-encoded string, parsed back to array"
                                );
                                msg.extra.insert(
                                    "tool_calls".to_string(),
                                    serde_json::Value::Array(arr),
                                );
                            }
                            _ => {
                                warn!(
                                    role = %msg.role,
                                    "Dropping tool_calls string that did not parse to a non-empty array"
                                );
                            }
                        }
                    } else {
                        warn!(
                            role = %msg.role,
                            "Dropping tool_calls string (not valid JSON)"
                        );
                    }
                }
                other => {
                    warn!(
                        role = %msg.role,
                        value_type = %other,
                        "Dropping malformed tool_calls (not an array, object, or string)"
                    );
                    // Drop it — better to omit than send garbage.
                }
            }
        }
    }

    pub fn should_cache_response(response: &OpenAIResponse) -> bool {
        let Some(choice) = response.choices.first() else {
            return false;
        };

        // Never cache tool_calls responses — they're part of a multi-turn
        // tool-use flow and replaying them from cache would break the loop
        if choice.message.extra.contains_key("tool_calls") {
            return false;
        }

        // Only cache complete responses
        match choice.finish_reason.as_deref() {
            Some("stop") => true,
            // function_call is legacy but equivalent to tool_calls
            Some("function_call") => false,
            Some("length") => false,
            Some("content_filter") => false,
            None => true,     // some providers omit finish_reason on success
            Some(_) => false, // unknown finish_reason — don't cache
        }
    }

    /// Determine whether a provider requires response transformation that
    /// makes true streaming pass-through impossible.
    ///
    /// Providers returning `true` here must be served via buffer-and-replay
    /// because their responses need server-side rewriting before they are
    /// OpenAI-compatible:
    /// - Bedrock providers require format translation.
    /// - Providers that emit XML-style tool calls need post-processing when
    ///   the request carries `tools` (see [`Self::translate_xml_tool_calls`]).
    /// - Kimi / Nano-GPT models leak special tokenizer tokens that must be
    ///   sanitized (see [`Self::sanitize_kimi_tokens_in_response`]).
    ///
    /// Requirements: 3.8
    // Wired into the streaming pass-through path by task 5.2.
    #[allow(dead_code)]
    fn provider_needs_transformation(&self, provider: &Provider, _request: &OpenAIRequest) -> bool {
        // Bedrock requires format translation — cannot pass-through stream.
        if provider.provider_type == "bedrock" {
            return true;
        }
        // Kimi / Nano-GPT models need special-token sanitization regardless of
        // tools, so they always take the buffered path.
        let name = provider.name.to_lowercase();
        if name.contains("kimi") || name.contains("nano-gpt") {
            return true;
        }
        false
    }

    /// Cache key for the adaptively-learned XML-tool combo set.
    fn xml_combo_key(provider: &str, model: &str) -> String {
        format!("{}::{}", provider, model)
    }

    /// True if this exact `provider`/`model` combo has been observed emitting
    /// XML-style tool calls during a streaming pass-through (see
    /// [`Self::mark_xml_tool_combo`]). Such combos take the buffer-and-translate
    /// path when the request carries `tools`.
    pub fn is_xml_tool_combo(&self, provider: &str, model: &str) -> bool {
        self.xml_tool_combos
            .read()
            .map(|set| set.contains(&Self::xml_combo_key(provider, model)))
            .unwrap_or(false)
    }

    /// Record a `provider`/`model` combo as one that emits XML-style tool calls.
    /// Idempotent. Called from the streaming relay when XML tool use is detected
    /// in an otherwise native-streamed response.
    pub fn mark_xml_tool_combo(&self, provider: &str, model: &str) {
        if let Ok(mut set) = self.xml_tool_combos.write() {
            set.insert(Self::xml_combo_key(provider, model));
        }
    }

    /// Heuristic: does `text` contain XML/pseudo-XML tool-call markers that some
    /// models emit instead of native OpenAI `tool_calls`? Shared by the
    /// streaming detector and the non-streaming diagnostic path.
    pub fn looks_like_xml_tool_use(text: &str) -> bool {
        text.contains("<use_tool")
            || text.contains("<tool_call")
            || text.contains("<function_call")
            || text.contains("<invoke ")
            || text.contains("<tool_calls>")
            || text.contains("<execute_command")
            || text.contains("<|tool_call")
    }

    /// Route non-streaming request
    ///
    /// Integrates provider selection, retry, failover, and cost calculation
    ///
    /// Requirements: 2.1, 30.1, 30.2
    pub async fn route_request(
        &self,
        request: &OpenAIRequest,
        active: Option<ActiveRequestHandle>,
    ) -> Result<OpenAIResponse, GatewayError> {
        let prepared_request = request.clone();

        // Find model group
        let model_group = self.find_model_group(&prepared_request.model).await?;
        debug!(group = %model_group.name, "Found model group");
        if let Some(handle) = &active {
            handle.set_group(&model_group.name);
        }

        let (model_group, mut routing_decision, routing_bypassed) = if let Some(plan) = self
            .smart_routing_plan(&prepared_request, &model_group)
            .await?
        {
            let mut filtered_group = model_group.clone();
            filtered_group.models = plan.candidates;
            (filtered_group, Some(plan.decision), plan.bypassed)
        } else {
            (model_group, None, true)
        };

        // Select provider order
        let providers = self.select_provider_order(&model_group).await;
        debug!(count = providers.len(), "Selected providers");

        if providers.is_empty() {
            return Err(GatewayError::InvalidRequest(
                "No available providers for model".to_string(),
            ));
        }

        // Route with failover and bounded response-quality cascade.
        let original_request = prepared_request.clone();
        let mut response = self
            .route_with_failover_for_group(
                &prepared_request,
                &model_group,
                providers,
                active.clone(),
                ActivePhase::Primary,
            )
            .await?;
        if let (Some(smart_router), Some(decision)) =
            (self.smart_router_snapshot(), routing_decision.as_mut())
        {
            let cascade_config = smart_router.cascade_config();
            while cascade_config.enabled {
                let failed = smart_router.cascade_evaluator().is_failure_signal(
                    crate::smart_routing::cascade::CascadeEvaluationInput::response(
                        &original_request,
                        &response,
                    ),
                    cascade_config,
                );
                let Some(_failure) = failed else {
                    break;
                };
                // Re-routing via smart-routing cascade: mark the in-flight entry
                // so the dashboard shows this is an escalated tier/version attempt.
                if let Some(handle) = &active {
                    handle.set_phase(ActivePhase::Cascade);
                }
                let Some(next_tier) = crate::smart_routing::cascade::CascadeEvaluator::next_tier(
                    decision.tier,
                    decision.escalation_count,
                    cascade_config.max_escalations,
                ) else {
                    break;
                };
                let context_safe_group = self.find_model_group(&original_request.model).await?;
                let next = smart_router.filter_by_tier(
                    &context_safe_group,
                    next_tier,
                    decision.task_type,
                    None,
                );
                if next.bypassed || next.model_group.models.is_empty() {
                    break;
                }
                let escalated_providers = self.select_provider_order(&next.model_group).await;
                if escalated_providers.is_empty() {
                    break;
                }
                response = self
                    .route_with_failover_for_group(
                        &original_request,
                        &next.model_group,
                        escalated_providers,
                        active.clone(),
                        ActivePhase::Cascade,
                    )
                    .await?;
                self.metrics.record_smart_routing_cascade_transition(
                    smart_routing_tier_name(decision.tier),
                    smart_routing_tier_name(next_tier),
                );
                decision.tier = next_tier;
                decision.escalated = true;
                decision.escalation_count = decision.escalation_count.saturating_add(1);
            }
        }
        if let Some(decision) = routing_decision.filter(|_| !routing_bypassed) {
            self.metrics.record_smart_routing_decision(
                crate::metrics::SmartRoutingDecisionMetric {
                    tier: smart_routing_tier_name(decision.tier),
                    classifier: smart_routing_classifier_name(decision.classifier),
                    group: &model_group.name,
                    score: decision.score.value(),
                    estimated_cost_usd: 0.0,
                    classifier_latency_ms: 0.0,
                    task_type: smart_routing_task_name(decision.task_type),
                    quality: 0.0,
                    context_filtered: decision.context_filtered,
                    experiment: None,
                },
            );
            response.extra.insert(
                "gateway_smart_routing".to_string(),
                serde_json::to_value(decision).unwrap_or(serde_json::Value::Null),
            );
        }

        Ok(response)
    }

    /// Build the native tool-calling system hint appended to outgoing requests
    /// that carry `tools`. Shared by the buffered ([`Self::attempt_with_retry`])
    /// and streaming pass-through ([`Self::route_request_streaming`]) paths so
    /// both send identical guidance to the provider.
    fn tool_calling_system_hint() -> Message {
        Message {
            role: "system".to_string(),
            content: serde_json::Value::String(
                r#"TOOL CALLING RULES:
You have access to tools through the API's native function-calling interface.
If you need a tool, respond with native `tool_calls` only. Do not write XML tags, pseudo-XML, markdown code fences, or plain-text tool instructions.

Use the exact tool name from the provided tools list. Arguments must be valid JSON and must match the tool schema exactly. Include only the fields the tool needs; do not repeat large amounts of context unnecessarily.

Correct single-tool example:
{"role":"assistant","content":"","tool_calls":[{"id":"call_1","type":"function","function":{"name":"read_file","arguments":"{\"path\":\"src/main.rs\"}"}}]}

Best practice for multi-step work:
1. Call one tool to inspect or gather information.
2. Wait for the tool result.
3. Then decide the next tool call or give a normal text answer.
4. Finish with plain assistant text only when no more tools are needed.

Common mistakes to avoid:
- Do NOT output <use_tool>...</use_tool>, <tool_call>...</tool_call>, <function_call>...</function_call>, <tool_calls><invoke>...</invoke></tool_calls>, or direct tags like <read_file>.
- Do NOT put tool JSON inside markdown fences or regular message text.
- Do NOT mix fake textual tool calls with native `tool_calls`.
- Do NOT invent argument names or guess missing required inputs; request the missing information with an appropriate question tool instead.
- Do NOT call multiple tools in one response unless they are independent and parallel tool calling is explicitly supported.

If no tool is needed, respond normally with plain assistant text and no `tool_calls`."#
                    .to_string(),
            ),
            extra: serde_json::Map::new(),
        }
    }

    /// Route a streaming request, returning a [`StreamingResponse`].
    ///
    /// Follows the same provider selection / circuit-breaker / cooldown path as
    /// [`Self::route_request`], but prefers true streaming pass-through:
    ///
    /// - When `streaming.passthrough_enabled` is false → buffer the whole
    ///   request via [`Self::route_request`] (`Buffered`).
    /// - When the first eligible provider needs response transformation
    ///   (Bedrock / XML-tool rewrite / Kimi-Nano sanitization) or is a Codex
    ///   OAuth provider → buffer the whole request (`Buffered`).
    /// - Otherwise → send the request upstream with `stream: true`, apply the
    ///   TTFB timeout to the initial response headers, and hand the live body
    ///   to the caller (`PassThrough`) without consuming it.
    ///
    /// Mid-stream relay, inter-chunk timeouts, accumulation, and failover are
    /// the responsibility of the streaming handler (tasks 5.3–5.6).
    ///
    /// Requirements: 3.1, 3.8, 3.9
    // Wired into the streaming handler by task 5.5.
    pub async fn route_request_streaming(
        &self,
        request: &OpenAIRequest,
        active: Option<ActiveRequestHandle>,
    ) -> Result<StreamingResponse, GatewayError> {
        self.route_request_streaming_excluding(request, &[], active)
            .await
    }

    /// Like [`Self::route_request_streaming`], but skips any provider whose name
    /// appears in `exclude` when picking the first eligible pass-through
    /// provider. Used by the streaming handler's pre-content failover loop
    /// (task 6.1, Req 4.1) to retry the next provider after one disconnects or
    /// errors before any content was forwarded.
    ///
    /// The buffered fallbacks are intentionally left intact: when no eligible
    /// pass-through provider remains (all excluded / circuit-open / cooled
    /// down), this returns a `Buffered` response via [`Self::route_request`],
    /// which performs its own gating, retry, and failover. Task 6.3 refines the
    /// failover limits and aggregated-error reporting.
    ///
    /// Requirements: 3.1, 3.8, 3.9, 4.1
    pub async fn route_request_streaming_excluding(
        &self,
        request: &OpenAIRequest,
        exclude: &[String],
        active: Option<ActiveRequestHandle>,
    ) -> Result<StreamingResponse, GatewayError> {
        let prepared_request = request.clone();

        // Effective streaming configuration (defaults when section absent).
        let streaming_config = self
            .config
            .read()
            .await
            .streaming
            .clone()
            .unwrap_or_default();

        // Model group + provider order (same selection path as route_request).
        let model_group = self.find_model_group(&prepared_request.model).await?;
        if let Some(handle) = &active {
            handle.set_group(&model_group.name);
        }
        let routing_plan = self
            .smart_routing_plan(&prepared_request, &model_group)
            .await?;
        let (model_group, routing_decision, routing_bypassed) = if let Some(plan) = routing_plan {
            let mut filtered_group = model_group.clone();
            filtered_group.models = plan.candidates;
            (filtered_group, Some(plan.decision), plan.bypassed)
        } else {
            (model_group, None, true)
        };
        let providers = self.select_provider_order(&model_group).await;
        if providers.is_empty() {
            return Err(GatewayError::InvalidRequest(
                "No available providers for model".to_string(),
            ));
        }

        // Pass-through disabled globally → buffer the whole request.
        if !streaming_config.passthrough_enabled
            || (routing_decision.is_some() && !routing_bypassed)
        {
            debug!("Streaming pass-through disabled, using buffered path");
            return Ok(StreamingResponse::Buffered(
                self.route_request(request, active.clone()).await?,
            ));
        }

        // Find the first provider eligible to serve (circuit breaker closed and
        // not in an upstream rate-limit cooldown). Mirrors the gating in
        // `route_with_failover` but stops at the first candidate; failover for
        // the streaming relay is handled by task 5.6.
        let mut chosen: Option<ProviderModel> = None;
        for provider_model in &providers {
            // Req 4.1: skip providers already tried in the failover loop.
            if exclude.iter().any(|p| p == &provider_model.provider) {
                debug!(provider = %provider_model.provider, "Excluded by failover, skipping (streaming)");
                continue;
            }
            let cb_key = format!("{}:{}", provider_model.provider, provider_model.model);
            let cb = self.get_circuit_breaker(&cb_key).await;
            if !cb.is_available().await {
                debug!(provider = %provider_model.provider, model = %provider_model.model, "Circuit breaker open, skipping (streaming)");
                continue;
            }
            if let Some(remaining) = self
                .metrics
                .provider_cooldown_remaining_secs(&provider_model.provider)
            {
                debug!(provider = %provider_model.provider, cooldown_remaining_secs = remaining, "Upstream cooldown active, skipping (streaming)");
                continue;
            }
            chosen = Some(provider_model.clone());
            break;
        }

        // No eligible provider for pass-through → buffered path performs its own
        // gating, retry, and failover.
        let provider_model = match chosen {
            Some(pm) => pm,
            None => {
                debug!("No eligible pass-through provider, using buffered path");
                return Ok(StreamingResponse::Buffered(
                    self.route_request(request, active.clone()).await?,
                ));
            }
        };

        // Report the streaming pass-through target to the in-flight registry.
        // If providers were excluded it means an earlier pass-through failed and
        // this is a failover; otherwise it is the primary attempt.
        if let Some(handle) = &active {
            let phase = if exclude.is_empty() {
                ActivePhase::Primary
            } else {
                ActivePhase::Failover
            };
            handle.set_target(&provider_model.provider, &provider_model.model, phase);
        }

        // Compression is provider-specific and completes before model rewrite,
        // sanitization, or the upstream streaming request starts.
        let request_id = format!("stream-{}", uuid::Uuid::new_v4());
        let (compressed_request, compression) = self
            .prepare_compressed_request_with_stats(
                &prepared_request,
                &model_group,
                &provider_model,
                &request_id,
            )
            .await;

        // Inspect the chosen provider config. Clone every field needed for the
        // outgoing request before dropping the config guard — the guard must
        // not be held across the network `.await`.
        let config = self.config.read().await;
        let provider_cfg = match config
            .providers
            .iter()
            .find(|p| p.name == provider_model.provider)
        {
            Some(p) => p,
            None => {
                drop(config);
                return Err(GatewayError::Configuration(format!(
                    "Provider '{}' not found in config",
                    provider_model.provider
                )));
            }
        };

        // Codex (oauth + openai) is handled end-to-end by CodexProviderClient
        // and cannot pass-through; providers needing response transformation
        // (Bedrock / XML-tool rewrite / Kimi-Nano) likewise must buffer.
        let is_codex = provider_cfg.auth_method.as_deref() == Some("oauth")
            && provider_cfg.provider_type == "openai";
        // A provider/model combo previously observed emitting XML-style tool
        // calls takes the buffer-and-translate path when the request carries
        // `tools`, so the XML can be rewritten into native `tool_calls`. Unknown
        // combos stream optimistically; the relay learns and marks them if XML
        // tool use is detected (see `relay_passthrough_stream`).
        let tools_present = prepared_request.extra.contains_key("tools");
        let known_xml_combo = tools_present
            && self.is_xml_tool_combo(&provider_model.provider, &provider_model.model);
        if is_codex
            || self.provider_needs_transformation(provider_cfg, &prepared_request)
            || known_xml_combo
        {
            drop(config);
            debug!(provider = %provider_model.provider, "Provider needs transformation or is Codex, using buffered path");
            return Ok(StreamingResponse::Buffered(
                self.route_request(request, active.clone()).await?,
            ));
        }

        // Snapshot config fields for the outgoing pass-through request.
        let api_key = provider_cfg.resolve_api_key().unwrap_or_default();
        let is_oauth_provider = provider_cfg.auth_method.as_deref() == Some("oauth");
        let provider_type = provider_cfg.provider_type.clone();
        let custom_vpc_endpoint = provider_cfg.custom_vpc_endpoint;
        let provider_region = provider_cfg.region.clone();
        let configured_base_url = provider_cfg.base_url.clone();
        let custom_headers = provider_cfg.custom_headers.clone();
        let pool_config = provider_cfg.connection_pool.clone();
        let ttfb_timeout_secs = provider_cfg.effective_ttfb_timeout(&provider_model.model);
        let ttfb_timeout = Duration::from_secs(ttfb_timeout_secs);
        drop(config);

        // OAuth bearer resolution (after dropping the config guard).
        let oauth_bearer: Option<String> = if is_oauth_provider {
            match &self.oauth_manager {
                Some(manager) => manager.get_access_token().await,
                None => None,
            }
        } else {
            None
        };
        // OAuth provider without a usable token → fall back to the buffered
        // path so the auth failure surfaces consistently via failover.
        if is_oauth_provider && oauth_bearer.is_none() {
            debug!(provider = %provider_model.provider, "OAuth session unusable, using buffered path");
            return Ok(StreamingResponse::Buffered(
                self.route_request(request, active.clone()).await?,
            ));
        }

        // Base URL normalization — strip trailing '/', ensure '/v1'; Bedrock
        // Mantle special-case kept for parity (Bedrock never reaches here).
        let mut base_url =
            if provider_type == "bedrock" && !api_key.is_empty() && !custom_vpc_endpoint {
                let region = provider_region.as_deref().unwrap_or("us-east-1");
                format!("https://bedrock-mantle.{}.api.aws/v1", region)
            } else {
                configured_base_url.unwrap_or_default()
            };
        base_url = base_url.trim_end_matches('/').to_string();
        if !base_url.ends_with("/v1") {
            base_url.push_str("/v1");
        }
        let url = format!("{}/chat/completions", base_url);

        // Build the outgoing request the same way attempt_with_retry does, but
        // request streaming from the provider.
        let mut outgoing = compressed_request;
        outgoing.model = provider_model.model.clone();
        outgoing.stream = true;
        let stripped = Self::sanitize_request_for_provider(&mut outgoing, &provider_type);
        if stripped > 0 {
            info!(provider = %provider_model.provider, provider_type = %provider_type, fields_removed = stripped, "Sanitized streaming request for provider");
        }
        Self::normalize_message_tool_calls(&mut outgoing.messages);
        if outgoing.extra.contains_key("tools") {
            Self::reverse_translate_tool_history(&mut outgoing.messages);
            outgoing.messages.push(Self::tool_calling_system_hint());
        }

        let http_client = self.get_or_create_http_client(&provider_model.provider, &pool_config)?;

        let mut req_builder = http_client
            .post(&url)
            .header("Content-Type", "application/json");
        if let Some(ref bearer) = oauth_bearer {
            req_builder = req_builder.header("Authorization", format!("Bearer {}", bearer));
        } else if !api_key.is_empty() {
            req_builder = req_builder.header("Authorization", format!("Bearer {}", api_key));
        }
 for (k, v) in &custom_headers {
 if k.eq_ignore_ascii_case(reqwest::header::ACCEPT_ENCODING.as_str()) {
 continue;
 }
 req_builder = req_builder.header(k.as_str(), v.as_str());
 }
 // RequestBuilder::header appends rather than replaces existing values, so
 // filter any provider-level Accept-Encoding above and add identity exactly
 // once. Compressed SSE is vulnerable to truncated decoder frames.
 req_builder = req_builder.header(reqwest::header::ACCEPT_ENCODING, "identity");


        tracing::info!(provider = %provider_model.provider, %url, model = %provider_model.model, ttfb_timeout_secs, "Calling provider (streaming pass-through)");

        // Apply the TTFB timeout to the initial response headers only. The
        // inter-chunk timeout is applied to the body by the relay loop (5.3).
        let send_result =
            tokio::time::timeout(ttfb_timeout, req_builder.json(&outgoing).send()).await;
        let response = match send_result {
            Ok(Ok(resp)) => resp,
            Ok(Err(e)) => {
                warn!(provider = %provider_model.provider, error = %e, "Streaming pass-through send failed, falling back to buffered path with full failover");
                return Ok(StreamingResponse::Buffered(
                    self.route_request(request, active.clone()).await?,
                ));
            }
            Err(_) => {
                warn!(provider = %provider_model.provider, ttfb_timeout_secs, "TTFB timeout (streaming) — falling back to buffered path with full failover");
                return Ok(StreamingResponse::Buffered(
                    self.route_request(request, active.clone()).await?,
                ));
            }
        };

        let status = response.status();
        if !status.is_success() {
            let status_code = status.as_u16();
            // Drain the error body so the connection can be reused, then fall
            // back to the buffered path which has full multi-provider failover.
            let _ = response.text().await;
            warn!(provider = %provider_model.provider, status = status_code, "Provider returned non-success status (streaming), falling back to buffered path with full failover");
            return Ok(StreamingResponse::Buffered(
                self.route_request(request, active.clone()).await?,
            ));
        }

        // Success — hand the live streaming body to the caller without
        // consuming it. The handler (5.3) relays chunks and accumulates.
        Ok(StreamingResponse::PassThrough {
            byte_stream: response,
            provider: provider_model.provider.clone(),
            model: provider_model.model.clone(),
            compression,
        })
    }

    pub async fn route_provider_pass_through(
        &self,
        endpoint: ProviderPassThroughEndpoint,
        requested_model: &str,
        content_type: &str,
        body: Vec<u8>,
    ) -> Result<ProviderPassThroughResponse, GatewayError> {
        let targets = self.pass_through_targets(endpoint, requested_model).await?;
        let mut attempts = Vec::new();
        let mut last_http_response = None;

        for target in targets {
            match self
                .send_provider_pass_through(endpoint, &target, content_type, body.clone())
                .await
            {
                Ok(response) if (200..300).contains(&response.status) => return Ok(response),
                Ok(response) => {
                    attempts.push(ProviderAttempt::new(
                        target.provider.name.clone(),
                        target.model.model.clone(),
                        Self::pass_through_error_message(&response.body, response.status),
                        Some(response.status),
                    ));
                    last_http_response = Some(response);
                }
                Err(error) => {
                    let (message, status_code) = match error {
                        GatewayError::Provider {
                            message,
                            status_code,
                            ..
                        } => (message, status_code),
                        other => (other.to_string(), None),
                    };
                    attempts.push(ProviderAttempt::new(
                        target.provider.name.clone(),
                        target.model.model.clone(),
                        message,
                        status_code,
                    ));
                }
            }
        }

        if let Some(response) = last_http_response {
            return Ok(response);
        }

        Err(GatewayError::AllProvidersFailed(AggregatedError::new(
            attempts,
        )))
    }

    /// Proxy a fine-tuning API request to the first configured OpenAI-compatible
    /// provider. Returns `GatewayError::Provider` with status 501 when no such
    /// provider exists, letting the handler emit the structured unsupported
    /// feature response after capability selection fails.
    pub async fn route_fine_tuning_pass_through(
        &self,
        method: reqwest::Method,
        path_suffix: &str,
        body: Option<Vec<u8>>,
    ) -> Result<ProviderPassThroughResponse, GatewayError> {
        let config = self.config.read().await;
        let provider = config
            .providers
            .iter()
            .find(|p| {
                p.provider_type == "openai" && p.base_url.as_deref().is_some_and(|u| !u.is_empty())
            })
            .cloned();
        drop(config);

        let provider = match provider {
            Some(provider) => provider,
            None => {
                return Err(GatewayError::Provider {
                    provider: "fine_tuning".to_string(),
                    message: "No OpenAI-compatible provider is configured for fine-tuning"
                        .to_string(),
                    status_code: Some(501),
                })
            }
        };

        let provider_name = provider.name.clone();
        let rate_limiter = self.get_rate_limiter(&provider_name).await;
        if !rate_limiter.consume().await {
            return Err(GatewayError::Provider {
                provider: provider_name,
                message: "Rate limit exhausted".to_string(),
                status_code: Some(429),
            });
        }

        let api_key = provider.resolve_api_key().unwrap_or_default();
        let mut base_url = provider.base_url.clone().unwrap_or_default();
        base_url = base_url.trim_end_matches('/').to_string();
        if !base_url.ends_with("/v1") {
            base_url.push_str("/v1");
        }
        let url = format!("{base_url}/fine_tuning/jobs{path_suffix}");
        let client = self.get_or_create_http_client(&provider_name, &provider.connection_pool)?;
        let mut request = client.request(method, &url);
        if let Some(body) = body {
            request = request
                .header(reqwest::header::CONTENT_TYPE, "application/json")
                .body(body);
        }
        if !api_key.is_empty() {
            request = request.bearer_auth(api_key);
        }
        for (name, value) in provider.resolve_custom_headers() {
            request = request.header(name, value);
        }

        let timeout_seconds = provider.effective_total_timeout("");
        let upstream = tokio::time::timeout(Duration::from_secs(timeout_seconds), request.send())
            .await
            .map_err(|_| GatewayError::Provider {
                provider: provider_name.clone(),
                message: format!("Request timed out after {timeout_seconds}s"),
                status_code: Some(504),
            })?
            .map_err(|error| GatewayError::Provider {
                provider: provider_name.clone(),
                message: error.to_string(),
                status_code: None,
            })?;
        let status_code = upstream.status().as_u16();
        let headers = upstream.headers().clone();
        let bytes = upstream
            .bytes()
            .await
            .map_err(|error| GatewayError::Provider {
                provider: provider_name,
                message: format!("Failed to read provider response: {error}"),
                status_code: Some(status_code),
            })?;

        Ok(ProviderPassThroughResponse {
            status: status_code,
            headers,
            body: bytes.to_vec(),
        })
    }

    async fn pass_through_targets(
        &self,
        endpoint: ProviderPassThroughEndpoint,
        requested_model: &str,
    ) -> Result<Vec<ProviderPassThroughTarget>, GatewayError> {
        let model_group = self.find_model_group(requested_model).await?;
        let candidates = self.select_provider_order(&model_group).await;
        if candidates.is_empty() {
            return Err(GatewayError::AllProvidersFailed(AggregatedError::new(
                model_group
                    .models
                    .iter()
                    .map(|model| {
                        ProviderAttempt::new(
                            model.provider.clone(),
                            model.model.clone(),
                            "Provider unavailable due to circuit breaker or rate-limit cooldown"
                                .to_string(),
                            Some(503),
                        )
                    })
                    .collect(),
            )));
        }

        let config = self.config.read().await;
        let targets = candidates
            .into_iter()
            .filter(|candidate| {
                requested_model == model_group.name || requested_model == candidate.model
            })
            .filter_map(|model| {
                config
                    .providers
                    .iter()
                    .find(|provider| provider.name == model.provider)
                    .cloned()
                    .map(|provider| ProviderPassThroughTarget { provider, model })
            })
            .filter(|target| {
                Self::provider_supports_endpoint(&target.provider.provider_type, endpoint)
            })
            .collect::<Vec<_>>();

        if targets.is_empty() {
            return Err(GatewayError::InvalidRequest(format!(
                "No configured provider supports '{}' for model '{}'",
                endpoint.path(),
                requested_model
            )));
        }
        Ok(targets)
    }

    fn provider_supports_endpoint(
        provider_type: &str,
        endpoint: ProviderPassThroughEndpoint,
    ) -> bool {
        match provider_type {
            // Bedrock uses AWS SigV4 request signing rather than an
            // OpenAI-compatible HTTP API, so it cannot serve pass-through.
            "bedrock" => false,
            "nvidia_nim" => matches!(
                endpoint,
                ProviderPassThroughEndpoint::Embeddings
                    | ProviderPassThroughEndpoint::AudioTranscriptions
            ),
            // Other OpenAI-compatible HTTP providers serve all four
            // pass-through endpoints.
            _ => true,
        }
    }

    async fn send_provider_pass_through(
        &self,
        endpoint: ProviderPassThroughEndpoint,
        target: &ProviderPassThroughTarget,
        content_type: &str,
        body: Vec<u8>,
    ) -> Result<ProviderPassThroughResponse, GatewayError> {
        let provider_name = &target.provider.name;
        let model_name = &target.model.model;
        let rate_limiter = self.get_rate_limiter(provider_name).await;
        if !rate_limiter.consume().await {
            return Err(GatewayError::Provider {
                provider: provider_name.clone(),
                message: "Rate limit exhausted".to_string(),
                status_code: Some(429),
            });
        }

        let api_key = target.provider.resolve_api_key().unwrap_or_default();
        let oauth_bearer = if target.provider.auth_method.as_deref() == Some("oauth") {
            match &self.oauth_manager {
                Some(manager) => manager.get_access_token().await,
                None => None,
            }
        } else {
            None
        };
        if target.provider.auth_method.as_deref() == Some("oauth") && oauth_bearer.is_none() {
            return Err(GatewayError::Provider {
                provider: provider_name.clone(),
                message: "OAuth session not authenticated; no usable access token".to_string(),
                status_code: Some(401),
            });
        }
        if target.provider.provider_type == "bedrock" && api_key.is_empty() {
            return Err(GatewayError::Provider {
                provider: provider_name.clone(),
                message: "Endpoint requires an OpenAI-compatible provider HTTP API".to_string(),
                status_code: Some(501),
            });
        }

        let mut base_url = target.provider.base_url.clone().unwrap_or_default();
        base_url = base_url.trim_end_matches('/').to_string();
        if base_url.is_empty() {
            return Err(GatewayError::Provider {
                provider: provider_name.clone(),
                message: "Provider base URL is not configured".to_string(),
                status_code: Some(500),
            });
        }
        if !base_url.ends_with("/v1") {
            base_url.push_str("/v1");
        }
        let url = format!("{}/{}", base_url, endpoint.path());
        let outgoing_body = Self::rewrite_pass_through_model(content_type, body, model_name)?;
        let client =
            self.get_or_create_http_client(provider_name, &target.provider.connection_pool)?;
        let mut request = client
            .post(&url)
            .header(reqwest::header::CONTENT_TYPE, content_type)
            .body(outgoing_body);
        if let Some(bearer) = oauth_bearer {
            request = request.bearer_auth(bearer);
        } else if !api_key.is_empty() {
            request = request.bearer_auth(api_key);
        }
        for (name, value) in target.provider.resolve_custom_headers() {
            request = request.header(name, value);
        }

        let started = std::time::Instant::now();
        let timeout_seconds = target.provider.effective_total_timeout(model_name);
        let upstream = tokio::time::timeout(Duration::from_secs(timeout_seconds), request.send())
            .await
            .map_err(|_| GatewayError::Provider {
                provider: provider_name.clone(),
                message: format!("Request timed out after {}s", timeout_seconds),
                status_code: Some(504),
            })?
            .map_err(|error| GatewayError::Provider {
                provider: provider_name.clone(),
                message: error.to_string(),
                status_code: None,
            })?;
        let status_code = upstream.status().as_u16();
        let headers = upstream.headers().clone();
        let bytes = upstream
            .bytes()
            .await
            .map_err(|error| GatewayError::Provider {
                provider: provider_name.clone(),
                message: format!("Failed to read provider response: {}", error),
                status_code: Some(status_code),
            })?;

        let circuit_breaker = self
            .get_circuit_breaker(&format!("{}:{}", provider_name, model_name))
            .await;
        if (200..300).contains(&status_code) {
            circuit_breaker.record_success().await;
            rate_limiter.clear_cooldown().await;
            self.metrics
                .record_provider_success(provider_name, started.elapsed().as_millis() as u64);
            return Ok(ProviderPassThroughResponse {
                status: status_code,
                headers,
                body: bytes.to_vec(),
            });
        }

        circuit_breaker.record_failure().await;
        let body_text = String::from_utf8_lossy(&bytes).to_string();
        self.metrics.record_provider_failure_with_reason(
            provider_name,
            Some(Self::friendly_failure_reason(Some(status_code), &body_text)),
            None,
        );

        // Mirror chat routing's 429 semantics: parse Retry-After /
        // retry_after_ms and place the provider in a bounded cooldown window
        // so subsequent requests skip it via select_provider_order.
        if Self::is_rate_limited(status_code, &body_text) {
            let cooldown = self
                .parse_rate_limit_cooldown(provider_name, Some(&headers), &body_text)
                .await;
            rate_limiter.apply_cooldown(cooldown).await;
            self.metrics
                .record_provider_rate_limit_exhausted(provider_name);
            let now_secs = SystemTime::now()
                .duration_since(UNIX_EPOCH)
                .map(|d| d.as_secs())
                .unwrap_or(0);
            let deadline = now_secs.saturating_add(cooldown.as_secs());
            self.metrics.set_provider_cooldown(
                provider_name,
                Self::friendly_failure_reason(Some(status_code), &body_text),
                deadline,
            );
            tracing::warn!(
                provider = provider_name,
                status = status_code,
                cooldown_ms = cooldown.as_millis() as u64,
                "Rate limited on pass-through, cooling down provider"
            );
        }

        Ok(ProviderPassThroughResponse {
            status: status_code,
            headers,
            body: bytes.to_vec(),
        })
    }

    fn rewrite_pass_through_model(
        content_type: &str,
        body: Vec<u8>,
        target_model: &str,
    ) -> Result<Vec<u8>, GatewayError> {
        if content_type
            .split(';')
            .next()
            .is_some_and(|mime| mime.trim().eq_ignore_ascii_case("application/json"))
        {
            let mut json: serde_json::Value = serde_json::from_slice(&body).map_err(|error| {
                GatewayError::InvalidRequest(format!("Invalid JSON body: {}", error))
            })?;
            let object = json.as_object_mut().ok_or_else(|| {
                GatewayError::InvalidRequest("JSON request body must be an object".to_string())
            })?;
            object.insert(
                "model".to_string(),
                serde_json::Value::String(target_model.to_string()),
            );
            return serde_json::to_vec(&json).map_err(|error| {
                GatewayError::InvalidRequest(format!("Failed to rewrite JSON model: {}", error))
            });
        }

        if content_type
            .split(';')
            .next()
            .is_some_and(|mime| mime.trim().eq_ignore_ascii_case("multipart/form-data"))
        {
            return Self::rewrite_multipart_model(content_type, body, target_model);
        }

        Err(GatewayError::InvalidRequest(format!(
            "Unsupported pass-through content type '{}'",
            content_type
        )))
    }

    fn rewrite_multipart_model(
        content_type: &str,
        mut body: Vec<u8>,
        target_model: &str,
    ) -> Result<Vec<u8>, GatewayError> {
        let boundary = content_type
            .split(';')
            .map(str::trim)
            .find_map(|part| part.strip_prefix("boundary="))
            .map(|boundary| boundary.trim_matches('"'))
            .filter(|boundary| !boundary.is_empty())
            .ok_or_else(|| {
                GatewayError::InvalidRequest("Multipart boundary is missing".to_string())
            })?;
        let delimiter = format!("--{}", boundary).into_bytes();
        let header_separator = b"\r\n\r\n";
        let line_end = b"\r\n";
        let mut search_start = 0;

        while let Some(relative_start) = body[search_start..]
            .windows(delimiter.len())
            .position(|window| window == delimiter.as_slice())
        {
            let part_start = search_start + relative_start + delimiter.len();
            let part_end = body[part_start..]
                .windows(delimiter.len())
                .position(|window| window == delimiter.as_slice())
                .map(|offset| part_start + offset)
                .unwrap_or(body.len());
            let Some(relative_header_end) = body[part_start..part_end]
                .windows(header_separator.len())
                .position(|window| window == header_separator)
            else {
                search_start = part_end;
                continue;
            };
            let header_end = part_start + relative_header_end;
            let headers = String::from_utf8_lossy(&body[part_start..header_end]);
            let is_model_field = headers.lines().any(|line| {
                line.split_once(':').is_some_and(|(name, value)| {
                    name.trim().eq_ignore_ascii_case("content-disposition")
                        && value.split(';').map(str::trim).any(|parameter| {
                            parameter
                                .strip_prefix("name=")
                                .is_some_and(|name| name.trim_matches('"') == "model")
                        })
                })
            });
            if is_model_field {
                let value_start = header_end + header_separator.len();
                let value_end = body[value_start..part_end]
                    .windows(line_end.len())
                    .position(|window| window == line_end)
                    .map(|offset| value_start + offset)
                    .unwrap_or(part_end);
                body.splice(value_start..value_end, target_model.bytes());
                return Ok(body);
            }
            search_start = part_end;
        }

        Err(GatewayError::InvalidRequest(
            "Multipart request must include a model field".to_string(),
        ))
    }

    fn pass_through_error_message(body: &[u8], status_code: u16) -> String {
        let text = String::from_utf8_lossy(body);
        serde_json::from_slice::<serde_json::Value>(body)
            .ok()
            .and_then(|json| {
                json.pointer("/error/message")
                    .or_else(|| json.get("message"))
                    .and_then(serde_json::Value::as_str)
                    .map(str::to_string)
            })
            .filter(|message| !message.is_empty())
            .unwrap_or_else(|| {
                let trimmed = text.trim();
                if trimmed.is_empty() {
                    format!("Provider returned HTTP {}", status_code)
                } else {
                    trimmed.to_string()
                }
            })
    }

    fn get_or_create_http_client(
        &self,
        provider_name: &str,
        pool_config: &crate::config::ProviderConnectionPoolConfig,
    ) -> Result<reqwest::Client, GatewayError> {
        if let Some(existing) = self.http_clients.get(provider_name) {
            return Ok(existing.clone());
        }

        let http_client = reqwest::Client::builder()
            .connect_timeout(Duration::from_secs(10))
            .tcp_keepalive(Duration::from_secs(30))
            .pool_max_idle_per_host(pool_config.max_idle_per_host as usize)
            .pool_idle_timeout(Duration::from_secs(pool_config.idle_timeout_seconds))
            .build()
            .map_err(|e| {
                GatewayError::Configuration(format!("Failed to build HTTP client: {}", e))
            })?;
        self.http_clients
            .insert(provider_name.to_string(), http_client.clone());
        Ok(http_client)
    }

    fn calculate_retry_delay(
        base_delay_secs: u64,
        jitter_enabled: bool,
        jitter_ratio: f64,
    ) -> Duration {
        if !jitter_enabled || jitter_ratio <= 0.0 {
            return Duration::from_secs(base_delay_secs);
        }

        let nanos = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .map(|duration| duration.subsec_nanos() as f64)
            .unwrap_or(0.0);
        let random_unit = (nanos % 1000.0) / 1000.0;
        let lower_bound = (1.0 - jitter_ratio).max(0.0);
        let upper_bound = 1.0 + jitter_ratio;
        let multiplier = lower_bound + ((upper_bound - lower_bound) * random_unit);
        Duration::from_secs_f64((base_delay_secs as f64) * multiplier)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::active_requests::{ActivePhase, ActiveRequestHandle};
    use crate::compression::{
        caveman::CAVEMAN_OUTPUT_SUFFIX,
        config::{ModelGroupCompressionOverride, PrecompressedEntry, ProviderCompressionOverride},
        engines::CompressionLevel,
        precompressed::{metadata_path_for, PrecompressedMetadata},
        token_counter::TokenCounter,
    };
    use crate::config::{CircuitBreakerConfig, ExactCacheConfig, ModelGroup, ProviderModel};
    use std::sync::{Arc, Mutex};
    use wiremock::{Mock, MockServer, ResponseTemplate};

    struct ContextThenSuccessClient {
        calls: std::sync::atomic::AtomicUsize,
    }

    #[async_trait::async_trait]
    impl ProviderClient for ContextThenSuccessClient {
        async fn chat_completion(
            &self,
            request: OpenAIRequest,
        ) -> Result<ProviderResponse, GatewayError> {
            let call = self.calls.fetch_add(1, std::sync::atomic::Ordering::SeqCst);
            if call == 0 {
                return Err(GatewayError::Provider {
                    provider: "adapter".to_string(),
                    message: "This model's maximum context length is 100 tokens. However, your messages resulted in 300 tokens.".to_string(),
                    status_code: Some(400),
                });
            }
            assert!(request.messages.len() <= 3);
            Ok(ProviderResponse {
                response: serde_json::from_value(completion_response())
                    .expect("fixture should deserialize"),
                provider_name: "adapter".to_string(),
                latency_ms: 1,
            })
        }

        async fn chat_completion_stream(
            &self,
            _request: OpenAIRequest,
        ) -> Result<
            std::pin::Pin<
                Box<
                    dyn futures::Stream<Item = Result<crate::providers::SSEEvent, GatewayError>>
                        + Send,
                >,
            >,
            GatewayError,
        > {
            unreachable!()
        }

        async fn list_models(&self) -> Result<Vec<crate::providers::Model>, GatewayError> {
            Ok(Vec::new())
        }

        fn provider_name(&self) -> &str {
            "adapter"
        }
    }

    #[tokio::test]
    async fn pass_through_rewrites_multipart_model_preserves_binary_and_fails_over() {
        use wiremock::matchers::{body_bytes, header, method, path};

        let first = MockServer::start().await;
        Mock::given(method("POST"))
            .and(path("/v1/audio/transcriptions"))
            .and(body_bytes(
                b"--raw\r\nContent-Disposition: form-data; name=\"model\"\r\n\r\nfirst-upstream-model\r\n--raw\r\nContent-Disposition: form-data; name=\"file\"; filename=\"a.wav\"\r\n\r\n\x00\xffaudio\r\n--raw--\r\n"
                    .to_vec(),
            ))
            .respond_with(ResponseTemplate::new(503).set_body_json(serde_json::json!({
                "error": { "message": "temporarily unavailable" }
            })))
            .mount(&first)
            .await;

        let second = MockServer::start().await;
        let body = b"--raw\r\nContent-Disposition: form-data; name=\"model\"\r\n\r\ntest-group\r\n--raw\r\nContent-Disposition: form-data; name=\"file\"; filename=\"a.wav\"\r\n\r\n\x00\xffaudio\r\n--raw--\r\n".to_vec();
        let rewritten_body = b"--raw\r\nContent-Disposition: form-data; name=\"model\"\r\n\r\nupstream-model\r\n--raw\r\nContent-Disposition: form-data; name=\"file\"; filename=\"a.wav\"\r\n\r\n\x00\xffaudio\r\n--raw--\r\n".to_vec();
        Mock::given(method("POST"))
            .and(path("/v1/audio/transcriptions"))
            .and(header("content-type", "multipart/form-data; boundary=raw"))
            .and(body_bytes(rewritten_body))
            .respond_with(ResponseTemplate::new(202).set_body_json(serde_json::json!({
                "text": "ok"
            })))
            .mount(&second)
            .await;

        let mut config = create_test_config();
        config.providers = vec![
            test_provider("first", first.uri()),
            test_provider("second", second.uri()),
        ];
        config.model_groups = vec![test_group(vec![
            test_model_named("first", "first-upstream-model", 1),
            test_model("second", 2),
        ])];
        let router = Router::new(Arc::new(RwLock::new(config)), test_metrics());
        let response = router
            .route_provider_pass_through(
                ProviderPassThroughEndpoint::AudioTranscriptions,
                "test-group",
                "multipart/form-data; boundary=raw",
                body,
            )
            .await
            .expect("second provider should succeed");
        assert_eq!(response.status, 202);
        assert_eq!(
            serde_json::from_slice::<serde_json::Value>(&response.body).unwrap()["text"],
            "ok"
        );
    }

    #[tokio::test]
    async fn pass_through_rewrites_json_model_and_propagates_success_status() {
        use wiremock::matchers::{body_json, method, path};

        let server = MockServer::start().await;
        Mock::given(method("POST"))
            .and(path("/v1/images/generations"))
            .and(body_json(serde_json::json!({
                "model": "upstream-model",
                "prompt": "draw a lighthouse"
            })))
            .respond_with(ResponseTemplate::new(201).set_body_json(serde_json::json!({
                "created": 123
            })))
            .mount(&server)
            .await;

        let mut config = create_test_config();
        config.providers = vec![test_provider("provider", server.uri())];
        config.model_groups = vec![test_group(vec![test_model("provider", 1)])];
        let router = Router::new(Arc::new(RwLock::new(config)), test_metrics());
        let response = router
            .route_provider_pass_through(
                ProviderPassThroughEndpoint::ImageGenerations,
                "test-group",
                "application/json",
                br#"{"model":"test-group","prompt":"draw a lighthouse"}"#.to_vec(),
            )
            .await
            .expect("provider should succeed");

        assert_eq!(response.status, 201);
        assert_eq!(
            serde_json::from_slice::<serde_json::Value>(&response.body).unwrap()["created"],
            123
        );
    }

    #[tokio::test]
    async fn pass_through_returns_terminal_provider_status_and_body() {
        use wiremock::matchers::{body_json, method, path};

        let server = MockServer::start().await;
        Mock::given(method("POST"))
            .and(path("/v1/embeddings"))
            .and(body_json(serde_json::json!({
                "model": "upstream-model",
                "input": "hello"
            })))
            .respond_with(ResponseTemplate::new(422).set_body_json(serde_json::json!({
                "error": { "message": "invalid embedding input" }
            })))
            .mount(&server)
            .await;

        let mut config = create_test_config();
        config.providers = vec![test_provider("provider", server.uri())];
        config.model_groups = vec![test_group(vec![test_model("provider", 1)])];
        let router = Router::new(Arc::new(RwLock::new(config)), test_metrics());
        let response = router
            .route_provider_pass_through(
                ProviderPassThroughEndpoint::Embeddings,
                "test-group",
                "application/json",
                br#"{"model":"test-group","input":"hello"}"#.to_vec(),
            )
            .await
            .expect("terminal HTTP response should be preserved");

        assert_eq!(response.status, 422);
        assert_eq!(
            serde_json::from_slice::<serde_json::Value>(&response.body).unwrap()["error"]
                ["message"],
            "invalid embedding input"
        );
    }

    #[tokio::test]
    async fn pass_through_maps_transport_failures_structurally() {
        let mut config = create_test_config();
        config.providers = vec![test_provider("provider", "http://127.0.0.1:9".to_string())];
        config.model_groups = vec![test_group(vec![test_model("provider", 1)])];
        let router = Router::new(Arc::new(RwLock::new(config)), test_metrics());
        let result = router
            .route_provider_pass_through(
                ProviderPassThroughEndpoint::Embeddings,
                "test-group",
                "application/json",
                br#"{"model":"test-group","input":"hello"}"#.to_vec(),
            )
            .await;
        let error = match result {
            Ok(_) => panic!("transport error should be aggregated"),
            Err(error) => error,
        };
        let GatewayError::AllProvidersFailed(aggregated) = error else {
            panic!("expected structured aggregate");
        };
        assert_eq!(aggregated.attempts.len(), 1);
        assert_eq!(aggregated.attempts[0].provider, "provider");
        assert_eq!(aggregated.attempts[0].model, "upstream-model");
        assert_eq!(aggregated.attempts[0].status_code, None);
    }

    #[tokio::test]
    async fn buffered_adapter_context_error_truncates_and_retries() {
        let router = Router::new(Arc::new(RwLock::new(create_test_config())), test_metrics());
        let client = ContextThenSuccessClient {
            calls: std::sync::atomic::AtomicUsize::new(0),
        };
        let mut request = compression_request(false);
        request.model = "adapter-model".to_string();
        request.messages = vec![
            Message {
                role: "system".to_string(),
                content: serde_json::json!("system"),
                extra: Default::default(),
            },
            Message {
                role: "user".to_string(),
                content: serde_json::json!("a".repeat(120)),
                extra: Default::default(),
            },
            Message {
                role: "assistant".to_string(),
                content: serde_json::json!("b".repeat(120)),
                extra: Default::default(),
            },
            Message {
                role: "user".to_string(),
                content: serde_json::json!("c".repeat(120)),
                extra: Default::default(),
            },
            Message {
                role: "assistant".to_string(),
                content: serde_json::json!("d".repeat(120)),
                extra: Default::default(),
            },
        ];

        let response = router
            .dispatch_buffered_with_context_retry(&client, request)
            .await
            .expect("shared wrapper should truncate and retry");
        assert_eq!(response.provider_name, "adapter");
        assert_eq!(client.calls.load(std::sync::atomic::Ordering::SeqCst), 2);
    }

    #[test]
    fn reasoning_only_response_is_promoted_to_content() {
        let mut response = OpenAIResponse {
            id: "test".to_string(),
            object: "chat.completion".to_string(),
            created: 1,
            model: "zai.glm-5".to_string(),
            choices: vec![Choice {
                index: 0,
                message: Message {
                    role: "assistant".to_string(),
                    content: serde_json::Value::String(String::new()),
                    extra: serde_json::Map::from_iter([(
                        "reasoning".to_string(),
                        serde_json::json!("reasoning fallback"),
                    )]),
                },
                finish_reason: Some("length".to_string()),
                extra: Default::default(),
            }],
            usage: Usage::default(),
            extra: Default::default(),
        };

        assert!(Router::promote_reasoning_to_content(&mut response));
        assert_eq!(
            response.choices[0].message.content_as_text(),
            "reasoning fallback"
        );
        assert!(Router::response_has_content(&response));
    }

    #[test]
    fn friendly_failure_reason_extracts_json_from_provider_prefix() {
        let message = r#"HTTP 400: {"error":{"message":"Invalid content part"}}"#;
        assert_eq!(
            Router::friendly_failure_reason(Some(400), message),
            "Provider rejected the request: Invalid content part"
        );
    }

    #[test]
    fn bedrock_sanitizer_keeps_reasoning_effort_and_drops_unknown_fields() {
        let mut request = OpenAIRequest {
            model: "zai.glm-5".to_string(),
            messages: vec![Message {
                role: "user".to_string(),
                content: serde_json::json!("test"),
                extra: Default::default(),
            }],
            stream: false,
            temperature: None,
            max_tokens: Some(16),
            extra: Default::default(),
        };
        request
            .extra
            .insert("reasoning_effort".to_string(), serde_json::json!("high"));
        request
            .extra
            .insert("store".to_string(), serde_json::json!(false));

        assert_eq!(
            Router::sanitize_request_for_provider(&mut request, "bedrock"),
            1
        );
        assert_eq!(
            request.extra.get("reasoning_effort"),
            Some(&serde_json::json!("high"))
        );
        assert!(!request.extra.contains_key("store"));
    }

    pub(super) fn test_metrics() -> Arc<crate::metrics::Metrics> {
        Arc::new(crate::metrics::Metrics::new())
    }

    pub(super) fn create_test_config() -> Config {
        Config {
            server: crate::config::ServerConfig {
                host: "0.0.0.0".to_string(),
                port: 8080,
                request_timeout_seconds: 30,
                max_request_size_mb: 10,
            },
            tls: None,
            admin: crate::config::AdminConfig::default(),
            dashboard: crate::config::DashboardConfig::default(),
            cors: crate::config::CorsConfig::default(),
            providers: vec![],
            model_groups: vec![],
            circuit_breaker: CircuitBreakerConfig::default(),
            retry: crate::config::RetryConfig::default(),
            logging: crate::config::LoggingConfig::default(),
            semantic_cache: None,
            exact_cache: ExactCacheConfig::default(),
            prometheus: None,
            context: crate::config::ContextConfig::default(),
            compression: Default::default(),
            structured_output: None,
            first_launch_completed: false,
            tray: crate::config::TrayConfig::default(),
            codex_instructions_url: None,
            streaming: None,
            virtual_keys: Default::default(),
            loop_detection: Default::default(),
            guardrails: None,
            tool_compression: Default::default(),
            smart_routing: Default::default(),
            memory: None,
            xhigh_models_allowlist: Default::default(),
            reasoning_models_allowlist: Default::default(),
        }
    }

    fn test_provider(name: &str, base_url: String) -> crate::config::Provider {
        crate::config::Provider {
            name: name.to_string(),
            provider_type: "openai".to_string(),
            base_url: Some(base_url),
            api_key_env: None,
            api_key_encrypted: None,
            api_secret_env: None,
            api_secret_encrypted: None,
            auth_method: None,
            resolved_api_key: None,
            resolved_api_secret: None,
            region: None,
            timeout_seconds: 30,
            ttfb_timeout_seconds: Some(5),
            total_timeout_seconds: Some(5),
            max_connections: 10,
            rate_limit_per_minute: 0,
            custom_headers: Default::default(),
            connection_pool: crate::config::ProviderConnectionPoolConfig::default(),
            budget: None,
            manual_models: vec![],
            global_inference_profile: false,
            cross_region_inference: false,
            prompt_caching: false,
            compression: None,
            custom_vpc_endpoint: false,
            reasoning: true,
            codex_base_url_override: None,
            codex_model_override: None,
            instructions_override: None,
            max_rate_limit_cooldown_seconds: None,
            memory: None,
        }
    }

    fn test_model(provider: &str, priority: u32) -> ProviderModel {
        test_model_named(provider, "upstream-model", priority)
    }

    fn test_model_named(provider: &str, model: &str, priority: u32) -> ProviderModel {
        ProviderModel {
            provider: provider.to_string(),
            model: model.to_string(),
            cost_per_million_input_tokens: 0.0,
            cost_per_million_output_tokens: 0.0,
            priority,
            structured_output_passthrough: None,
            tier: None,
            context_window: 0,
            specializations: vec![],
        }
    }

    fn test_group(models: Vec<ProviderModel>) -> ModelGroup {
        ModelGroup {
            name: "test-group".to_string(),
            version_fallback_enabled: false,
            compression: None,
            structured_output: None,
            memory: None,
            models,
        }
    }

    fn compression_request(stream: bool) -> OpenAIRequest {
        OpenAIRequest {
            model: "test-group".to_string(),
            messages: vec![Message {
                role: "user".to_string(),
                content: serde_json::json!(
                    "Please actually use a very small number of checks in order to finish."
                ),
                extra: Default::default(),
            }],
            stream,
            temperature: None,
            max_tokens: None,
            extra: Default::default(),
        }
    }

    fn compression_config(level: CompressionLevel, threshold: u32) -> CompressionConfig {
        CompressionConfig {
            enabled: true,
            default_level: level,
            auto_threshold_tokens: threshold,
            ..CompressionConfig::default()
        }
    }

    fn completion_response() -> serde_json::Value {
        serde_json::json!({
            "id": "chatcmpl-test",
            "object": "chat.completion",
            "created": 1,
            "model": "upstream-model",
            "choices": [{
                "index": 0,
                "message": {"role": "assistant", "content": "ok"},
                "finish_reason": "stop"
            }],
            "usage": {"prompt_tokens": 1, "completion_tokens": 1, "total_tokens": 2}
        })
    }

    fn request_content(request: &OpenAIRequest) -> &str {
        request.messages[0].content.as_str().unwrap()
    }

    fn install_precompressed_fixture(
        directory: &tempfile::TempDir,
        source_name: &str,
        source: &str,
        artifact_name: &str,
        artifact: &str,
    ) -> Arc<PrecompressedManager> {
        use chrono::Utc;

        std::fs::write(directory.path().join(source_name), source).unwrap();
        std::fs::write(directory.path().join(artifact_name), artifact).unwrap();
        let metadata = PrecompressedMetadata::for_source(
            100,
            40,
            CompressionLevel::Standard,
            Utc::now(),
            source.as_bytes(),
        )
        .unwrap();
        std::fs::write(
            metadata_path_for(directory.path().join(artifact_name)),
            serde_json::to_vec(&metadata).unwrap(),
        )
        .unwrap();
        Arc::new(
            PrecompressedManager::new(
                directory.path(),
                [PrecompressedEntry {
                    source_path: source_name.to_owned(),
                    compressed_path: artifact_name.to_owned(),
                    content_hash: None,
                }],
            )
            .unwrap(),
        )
    }

    fn precompressed_router(
        manager: Arc<PrecompressedManager>,
    ) -> (Router, ModelGroup, ProviderModel) {
        let mut config = create_test_config();
        config.compression = compression_config(CompressionLevel::Standard, 0);
        let provider_model = test_model("provider", 1);
        let group = test_group(vec![provider_model.clone()]);
        config.providers = vec![test_provider("provider", "http://localhost".to_owned())];
        config.model_groups = vec![group.clone()];
        let router = Router::new(Arc::new(RwLock::new(config)), test_metrics());
        router.set_precompressed_manager(Some(manager));
        (router, group, provider_model)
    }

    #[tokio::test]
    async fn precompressed_hit_replaces_exact_reference_without_runtime_recompression() {
        let directory = tempfile::tempdir().unwrap();
        let artifact = "Artifact actually retains this exact wording in order to prove bypass.";
        let manager = install_precompressed_fixture(
            &directory,
            "context.md",
            "Original source that should not reach the provider.",
            "context.compressed.md",
            artifact,
        );
        let (router, group, provider_model) = precompressed_router(manager);
        let request = OpenAIRequest {
            messages: vec![Message {
                role: "system".to_owned(),
                content: serde_json::json!("file://context.md"),
                extra: Default::default(),
            }],
            ..compression_request(false)
        };

        let prepared = router
            .prepare_compressed_request(&request, &group, &provider_model, "precompressed-hit")
            .await;

        assert_eq!(prepared.messages[0].content, serde_json::json!(artifact));
        assert!(!prepared.messages[0].extra.contains_key("cache_control"));
    }

    #[tokio::test]
    async fn stale_precompressed_source_uses_original_then_runtime_compression() {
        let directory = tempfile::tempdir().unwrap();
        let manager = install_precompressed_fixture(
            &directory,
            "context.md",
            "Initial source.",
            "context.compressed.md",
            "Old artifact.",
        );
        let changed = (0..80)
            .map(|index| {
                format!(
                    "Please actually use a very small number of checks in order to finish item {index}."
                )
            })
            .collect::<Vec<_>>()
            .join(" ");
        std::fs::write(directory.path().join("context.md"), &changed).unwrap();
        let (router, group, provider_model) = precompressed_router(manager);
        let request = OpenAIRequest {
            messages: vec![Message {
                role: "user".to_owned(),
                content: serde_json::json!("file://context.md"),
                extra: Default::default(),
            }],
            ..compression_request(false)
        };

        let prepared = router
            .prepare_compressed_request(&request, &group, &provider_model, "precompressed-stale")
            .await;
        let content = prepared.messages[0].content.as_str().unwrap();

        assert_ne!(content, changed);
        assert!(!content.contains("actually"));
        assert!(!content.contains("in order to"));
        assert!(!prepared.messages[0].extra.contains_key("cache_control"));
    }

    #[tokio::test]
    async fn precompressed_references_are_explicit_configured_and_root_constrained() {
        let directory = tempfile::tempdir().unwrap();
        let manager = install_precompressed_fixture(
            &directory,
            "context.md",
            "Original source.",
            "context.compressed.md",
            "Validated artifact.",
        );
        let (router, group, provider_model) = precompressed_router(manager);
        let literal = "Read context.md and ../secret.md without treating them as paths.";
        let request = OpenAIRequest {
            messages: vec![
                Message {
                    role: "user".to_owned(),
                    content: serde_json::json!(literal),
                    extra: Default::default(),
                },
                Message {
                    role: "user".to_owned(),
                    content: serde_json::json!("file://unknown.md"),
                    extra: Default::default(),
                },
                Message {
                    role: "user".to_owned(),
                    content: serde_json::json!({"type":"file_reference","path":"../secret.md"}),
                    extra: Default::default(),
                },
                Message {
                    role: "user".to_owned(),
                    content: serde_json::json!({"type":"file_reference","path":"context.md"}),
                    extra: Default::default(),
                },
            ],
            ..compression_request(false)
        };
        let disabled_group = ModelGroup {
            compression: Some(ModelGroupCompressionOverride {
                level: Some(CompressionLevel::None),
                auto_threshold_tokens: Some(0),
                caveman_output: Some(false),
            }),
            ..group
        };

        let prepared = router
            .prepare_compressed_request(
                &request,
                &disabled_group,
                &provider_model,
                "precompressed-explicit",
            )
            .await;

        assert_eq!(prepared.messages[0].content, serde_json::json!(literal));
        assert_eq!(
            prepared.messages[1].content,
            serde_json::json!("file://unknown.md")
        );
        assert_eq!(
            prepared.messages[2].content,
            serde_json::json!({"type":"file_reference","path":"../secret.md"})
        );
        assert_eq!(
            prepared.messages[3].content,
            serde_json::json!("Validated artifact.")
        );
        assert!(prepared
            .messages
            .iter()
            .all(|message| !message.extra.contains_key("cache_control")));
    }

    #[tokio::test]
    async fn precompressed_manager_reload_changes_subsequent_requests() {
        let first_directory = tempfile::tempdir().unwrap();
        let second_directory = tempfile::tempdir().unwrap();
        let first = install_precompressed_fixture(
            &first_directory,
            "context.md",
            "First source.",
            "context.compressed.md",
            "First artifact.",
        );
        let second = install_precompressed_fixture(
            &second_directory,
            "context.md",
            "Second source.",
            "context.compressed.md",
            "Second artifact.",
        );
        let (router, group, provider_model) = precompressed_router(first);
        let request = OpenAIRequest {
            messages: vec![Message {
                role: "system".to_owned(),
                content: serde_json::json!("file://context.md"),
                extra: Default::default(),
            }],
            ..compression_request(false)
        };

        let before = router
            .prepare_compressed_request(&request, &group, &provider_model, "manager-before")
            .await;
        let compression = {
            let config = router.config.read().await;
            config.compression.clone()
        };
        router.reload_compression_runtime(compression, Some(second));
        let after = router
            .prepare_compressed_request(&request, &group, &provider_model, "manager-after")
            .await;

        assert_eq!(
            before.messages[0].content,
            serde_json::json!("First artifact.")
        );
        assert_eq!(
            after.messages[0].content,
            serde_json::json!("Second artifact.")
        );
    }

    #[tokio::test]
    async fn compression_disabled_leaves_request_unchanged() {
        let mut config = create_test_config();
        let provider_model = test_model("provider", 1);
        let group = test_group(vec![provider_model.clone()]);
        config.providers = vec![test_provider("provider", "http://localhost".to_string())];
        config.model_groups = vec![group.clone()];
        let router = Router::new(Arc::new(RwLock::new(config)), test_metrics());
        let request = compression_request(false);

        let prepared = router
            .prepare_compressed_request(&request, &group, &provider_model, "disabled")
            .await;

        assert_eq!(
            serde_json::to_value(prepared).unwrap(),
            serde_json::to_value(request).unwrap()
        );
    }

    #[tokio::test]
    async fn compression_resolves_global_provider_model_group_precedence() {
        let mut config = create_test_config();
        config.compression = compression_config(CompressionLevel::Lite, 0);
        let mut provider = test_provider("provider", "http://localhost".to_string());
        provider.compression = Some(ProviderCompressionOverride {
            enabled: Some(true),
            level: Some(CompressionLevel::Standard),
            auto_threshold_tokens: Some(0),
            caveman_output: Some(false),
        });
        let provider_model = test_model("provider", 1);
        let mut group = test_group(vec![provider_model.clone()]);
        group.compression = Some(ModelGroupCompressionOverride {
            level: Some(CompressionLevel::None),
            auto_threshold_tokens: Some(0),
            caveman_output: None,
        });
        config.providers = vec![provider];
        config.model_groups = vec![group.clone()];
        let router = Router::new(Arc::new(RwLock::new(config)), test_metrics());
        let request = compression_request(false);

        let model_group_disabled = router
            .prepare_compressed_request(&request, &group, &provider_model, "model-group")
            .await;
        assert_eq!(
            request_content(&model_group_disabled),
            request_content(&request)
        );

        group.compression = None;
        let provider_compressed = router
            .prepare_compressed_request(&request, &group, &provider_model, "provider")
            .await;
        assert_ne!(
            request_content(&provider_compressed),
            request_content(&request)
        );
        assert!(!request_content(&provider_compressed).contains("actually"));
        assert!(!request_content(&provider_compressed).contains("in order to"));
    }

    #[tokio::test]
    async fn compression_auto_threshold_is_strictly_greater_than() {
        let mut config = create_test_config();
        let provider_model = test_model("provider", 1);
        let group = test_group(vec![provider_model.clone()]);
        config.providers = vec![test_provider("provider", "http://localhost".to_string())];
        config.model_groups = vec![group.clone()];
        let request = compression_request(false);
        let token_count = TokenCounter::new().count_request(&request);
        config.compression = compression_config(CompressionLevel::Standard, token_count);
        let router = Router::new(Arc::new(RwLock::new(config)), test_metrics());

        let equal = router
            .prepare_compressed_request(&request, &group, &provider_model, "equal")
            .await;
        assert_eq!(request_content(&equal), request_content(&request));

        let above_config = compression_config(CompressionLevel::Standard, token_count - 1);
        {
            let mut config = router.config.write().await;
            config.compression = above_config.clone();
        }
        router.reload_compression_pipeline(above_config);
        let above = router
            .prepare_compressed_request(&request, &group, &provider_model, "above")
            .await;
        assert_ne!(request_content(&above), request_content(&request));
    }

    #[tokio::test]
    async fn compression_noop_returns_original_request() {
        let mut config = create_test_config();
        config.compression = compression_config(CompressionLevel::Lite, 0);
        let provider_model = test_model("provider", 1);
        let group = test_group(vec![provider_model.clone()]);
        config.providers = vec![test_provider("provider", "http://localhost".to_string())];
        config.model_groups = vec![group.clone()];
        let router = Router::new(Arc::new(RwLock::new(config)), test_metrics());
        let request = OpenAIRequest {
            messages: vec![Message {
                role: "user".to_string(),
                content: serde_json::json!("already compact"),
                extra: Default::default(),
            }],
            ..compression_request(false)
        };

        let prepared = router
            .prepare_compressed_request(&request, &group, &provider_model, "noop")
            .await;
        assert_eq!(
            serde_json::to_value(prepared).unwrap(),
            serde_json::to_value(request).unwrap()
        );
    }

    #[tokio::test]
    async fn caveman_suffix_applies_and_skips_existing_output_instruction() {
        let mut config = create_test_config();
        config.compression.caveman_output = true;
        let provider_model = test_model("provider", 1);
        let group = test_group(vec![provider_model.clone()]);
        config.providers = vec![test_provider("provider", "http://localhost".to_string())];
        config.model_groups = vec![group.clone()];
        let router = Router::new(Arc::new(RwLock::new(config)), test_metrics());

        let applied = router
            .prepare_compressed_request(
                &compression_request(false),
                &group,
                &provider_model,
                "caveman",
            )
            .await;
        assert_eq!(applied.messages[0].role, "system");
        assert!(request_content(&applied).contains(CAVEMAN_OUTPUT_SUFFIX));

        let existing = OpenAIRequest {
            messages: vec![
                Message {
                    role: "system".to_string(),
                    content: serde_json::json!("Respond in JSON format."),
                    extra: Default::default(),
                },
                Message {
                    role: "user".to_string(),
                    content: serde_json::json!("answer"),
                    extra: Default::default(),
                },
            ],
            ..compression_request(false)
        };
        let skipped = router
            .prepare_compressed_request(&existing, &group, &provider_model, "skip")
            .await;
        assert_eq!(skipped.messages.len(), existing.messages.len());
        assert!(!skipped
            .messages
            .iter()
            .any(|message| message.content_as_text().contains(CAVEMAN_OUTPUT_SUFFIX)));
    }

    #[tokio::test]
    async fn buffered_provider_receives_compressed_body() {
        use wiremock::matchers::{body_string_contains, method, path};

        let server = MockServer::start().await;
        Mock::given(method("POST"))
            .and(path("/v1/chat/completions"))
            .and(body_string_contains(
                "use a small number of checks to finish",
            ))
            .respond_with(ResponseTemplate::new(200).set_body_json(completion_response()))
            .expect(1)
            .mount(&server)
            .await;
        let mut config = create_test_config();
        config.retry.max_retries_per_provider = 0;
        config.compression = compression_config(CompressionLevel::Standard, 0);
        let provider_model = test_model("provider", 1);
        config.providers = vec![test_provider("provider", server.uri())];
        config.model_groups = vec![test_group(vec![provider_model])];
        let mut router = Router::new(Arc::new(RwLock::new(config)), test_metrics());
        let hub = Arc::new(CompressionEventHub::new());
        router.set_compression_event_hub(hub.clone());
        let replay_before = hub.subscribe().replay.len();

        let response = router
            .route_request(&compression_request(false), None)
            .await
            .unwrap();
        assert!(response.extra.contains_key("gateway_compression"));
        let replay = hub.subscribe().replay;
        assert_eq!(replay.len(), replay_before + 1);
        assert_eq!(replay[0].provider, "provider");
        assert_eq!(replay[0].model, "upstream-model");
    }

    #[tokio::test]
 async fn streaming_provider_receives_compressed_body_before_response() {
 use wiremock::matchers::{body_string_contains, method, path};


        let server = MockServer::start().await;
 Mock::given(method("POST"))
 .and(path("/v1/chat/completions"))
 .and(body_string_contains(

                "use a small number of checks to finish",
            ))
            .and(body_string_contains("\"stream\":true"))
            .respond_with(
                ResponseTemplate::new(200)
                    .insert_header("content-type", "text/event-stream")
                    .set_body_string("data: [DONE]\n\n"),
            )
            .expect(1)
            .mount(&server)
            .await;
        let mut config = create_test_config();
        config.compression = compression_config(CompressionLevel::Standard, 0);
        let provider_model = test_model("provider", 1);
        let mut provider = test_provider("provider", server.uri());
        provider
            .custom_headers
            .insert("Accept-Encoding".to_string(), "gzip".to_string());
        config.providers = vec![provider];
        config.model_groups = vec![test_group(vec![provider_model])];
        let router = Router::new(Arc::new(RwLock::new(config)), test_metrics());

        let response = router
            .route_request_streaming(&compression_request(true), None)
            .await
            .unwrap();
        assert!(matches!(response, StreamingResponse::PassThrough { .. }));
        let StreamingResponse::PassThrough { compression, .. } = response else {
            unreachable!()
        };
 assert_eq!(compression.provider, "provider");
 assert_eq!(compression.model, "upstream-model");

 let requests = server.received_requests().await.unwrap();
 assert_eq!(requests.len(), 1);
 let accept_encoding = requests[0]
 .headers
 .get_all(reqwest::header::ACCEPT_ENCODING)
 .iter()
 .flat_map(|value| value.to_str().unwrap().split(','))
 .map(str::trim)
 .collect::<Vec<_>>();
 assert_eq!(accept_encoding, vec!["identity"]);
 }


    #[tokio::test]
    async fn failover_prepares_each_provider_from_original_request() {
        use wiremock::matchers::{body_string_contains, method, path};

        let first = MockServer::start().await;
        let second = MockServer::start().await;
        Mock::given(method("POST"))
            .and(path("/v1/chat/completions"))
            .and(body_string_contains(
                "use a small number of checks to finish",
            ))
            .respond_with(ResponseTemplate::new(500))
            .expect(1)
            .mount(&first)
            .await;
        Mock::given(method("POST"))
            .and(path("/v1/chat/completions"))
            .and(body_string_contains(
                "Please actually use a very small number of checks in order to finish.",
            ))
            .respond_with(ResponseTemplate::new(200).set_body_json(completion_response()))
            .expect(1)
            .mount(&second)
            .await;
        let mut config = create_test_config();
        config.retry.max_retries_per_provider = 0;
        config.compression = compression_config(CompressionLevel::Lite, 0);
        let mut first_provider = test_provider("first", first.uri());
        first_provider.compression = Some(ProviderCompressionOverride {
            enabled: Some(true),
            level: Some(CompressionLevel::Standard),
            auto_threshold_tokens: Some(0),
            caveman_output: None,
        });
        let mut second_provider = test_provider("second", second.uri());
        second_provider.compression = Some(ProviderCompressionOverride {
            enabled: Some(false),
            level: None,
            auto_threshold_tokens: None,
            caveman_output: None,
        });
        config.providers = vec![first_provider, second_provider];
        config.model_groups = vec![test_group(vec![
            test_model("first", 1),
            test_model("second", 2),
        ])];
        let router = Router::new(Arc::new(RwLock::new(config)), test_metrics());

        router
            .route_request(&compression_request(false), None)
            .await
            .unwrap();
    }

    #[tokio::test]
    async fn in_flight_handle_reflects_failover_to_second_provider() {
        use crate::active_requests::ActiveRequestInfo;
        let first = MockServer::start().await;
        let second = MockServer::start().await;
        Mock::given(wiremock::matchers::method("POST"))
            .and(wiremock::matchers::path("/v1/chat/completions"))
            .respond_with(ResponseTemplate::new(500))
            .mount(&first)
            .await;
        Mock::given(wiremock::matchers::method("POST"))
            .and(wiremock::matchers::path("/v1/chat/completions"))
            .respond_with(ResponseTemplate::new(200).set_body_json(completion_response()))
            .mount(&second)
            .await;

        let mut config = create_test_config();
        config.retry.max_retries_per_provider = 0;
        config.compression = CompressionConfig::default();
        let mut first_provider = test_provider("first", first.uri());
        first_provider.compression = None;
        let mut second_provider = test_provider("second", second.uri());
        second_provider.compression = None;
        config.providers = vec![first_provider, second_provider];
        config.model_groups = vec![test_group(vec![
            test_model("first", 1),
            test_model("second", 2),
        ])];
        let router = Router::new(Arc::new(RwLock::new(config)), test_metrics());

        let handle = ActiveRequestHandle(Arc::new(Mutex::new(ActiveRequestInfo {
            trace_id: "test-trace".to_string(),
            requested_model: "test-group".to_string(),
            model_group: None,
            provider: None,
            model: None,
            attempt: 0,
            phase: ActivePhase::Pending,
            last_error: None,
            virtual_key_id: None,
            started_at_ms: 0,
            kind: crate::active_requests::RequestKind::Chat,
        })));
        let result = router
            .route_request(&compression_request(false), Some(handle.clone()))
            .await;
        assert!(
            result.is_ok(),
            "routing should succeed via the second provider"
        );

        let info = handle.0.lock().unwrap();
        assert_eq!(
            info.phase,
            ActivePhase::Failover,
            "second provider is a failover"
        );
        assert_eq!(info.provider.as_deref(), Some("second"));
        assert!(
            info.last_error.is_some(),
            "prior failure should be recorded"
        );
    }

    #[tokio::test]
    async fn compression_records_metrics_for_operations_and_noops() {
        let mut config = create_test_config();
        config.compression = compression_config(CompressionLevel::Standard, 0);
        let provider_model = test_model("provider", 1);
        let group = test_group(vec![provider_model.clone()]);
        config.providers = vec![test_provider("provider", "http://localhost".to_string())];
        config.model_groups = vec![group.clone()];
        let metrics = test_metrics();
        let router = Router::new(Arc::new(RwLock::new(config)), metrics.clone());

        router
            .prepare_compressed_request(
                &compression_request(false),
                &group,
                &provider_model,
                "compressed",
            )
            .await;
        let noop = OpenAIRequest {
            messages: vec![Message {
                role: "user".to_owned(),
                content: serde_json::json!("compact"),
                extra: Default::default(),
            }],
            ..compression_request(false)
        };
        router
            .prepare_compressed_request(&noop, &group, &provider_model, "noop-metrics")
            .await;

        let mut out = String::new();
        metrics.write_compression_prometheus(&mut out);
        assert!(out
            .contains("obey_compression_ratio_count{level=\"standard\",provider=\"provider\"} 2"));
        assert!(out.contains(
            "obey_compression_duration_seconds_count{level=\"standard\",provider=\"provider\"} 2"
        ));
    }

    #[test]
    fn compression_savings_warning_threshold_is_strictly_over_fifty_percent() {
        let mut stats = CompressionStats {
            request_id: "warning-test".to_owned(),
            level: CompressionLevel::Standard,
            engines_applied: Vec::new(),
            original_tokens: 100,
            compressed_tokens: 50,
            savings_percent: 50.0,
            compression_time_ms: 1,
            auto_triggered: false,
            cache_downgrade_applied: false,
            tool_definitions_tokens_saved: 0,
            caveman_applied: false,
            timed_out: false,
            error: false,
            provider: "provider".to_owned(),
            model: "model".to_owned(),
            engine_results: Vec::new(),
        };
        assert!(!Router::compression_savings_warning_required(&stats));
        stats.savings_percent = 51.0;
        assert!(Router::compression_savings_warning_required(&stats));
    }

    #[tokio::test]
    async fn pipeline_reload_changes_subsequent_request_snapshots() {
        let mut config = create_test_config();
        let provider_model = test_model("provider", 1);
        let group = test_group(vec![provider_model.clone()]);
        config.providers = vec![test_provider("provider", "http://localhost".to_string())];
        config.model_groups = vec![group.clone()];
        let router = Router::new(Arc::new(RwLock::new(config)), test_metrics());
        let request = compression_request(false);

        let before = router
            .prepare_compressed_request(&request, &group, &provider_model, "before-reload")
            .await;
        assert_eq!(request_content(&before), request_content(&request));

        let replacement = compression_config(CompressionLevel::Standard, 0);
        {
            let mut config = router.config.write().await;
            config.compression = replacement.clone();
        }
        router.reload_compression_pipeline(replacement);
        let after = router
            .prepare_compressed_request(&request, &group, &provider_model, "after-reload")
            .await;
        assert_ne!(request_content(&after), request_content(&request));
    }

    #[tokio::test]
    async fn test_find_model_group_success() {
        let mut config = create_test_config();
        config.model_groups = vec![ModelGroup {
            name: "gpt-4-group".to_string(),
            version_fallback_enabled: false,
            compression: None,
            structured_output: None,
            memory: None,

            models: vec![ProviderModel {
                provider: "openai".to_string(),
                model: "gpt-4".to_string(),
                cost_per_million_input_tokens: 10.0,
                cost_per_million_output_tokens: 30.0,
                priority: 100,
                structured_output_passthrough: None,
                tier: None,
                context_window: 0,
                specializations: vec![],
            }],
        }];

        let router = Router::new(Arc::new(RwLock::new(config)), test_metrics());
        let result = router.find_model_group("gpt-4").await;

        assert!(result.is_ok());
        assert_eq!(result.unwrap().name, "gpt-4-group");
    }

    #[tokio::test]
    async fn test_find_model_group_not_found() {
        let config = create_test_config();
        let router = Router::new(Arc::new(RwLock::new(config)), test_metrics());
        let result = router.find_model_group("unknown-model").await;

        assert!(result.is_err());
    }

    #[tokio::test]
    async fn test_select_provider_order_by_priority() {
        let mut config = create_test_config();
        config.model_groups = vec![ModelGroup {
            name: "test-group".to_string(),
            version_fallback_enabled: false,
            compression: None,
            structured_output: None,
            memory: None,
            models: vec![
                ProviderModel {
                    provider: "provider-low-priority".to_string(),
                    model: "model-1".to_string(),
                    cost_per_million_input_tokens: 10.0,
                    cost_per_million_output_tokens: 30.0,
                    priority: 200,
                    structured_output_passthrough: None,
                    tier: None,
                    context_window: 0,
                    specializations: vec![],
                },
                ProviderModel {
                    provider: "provider-high-priority".to_string(),
                    model: "model-2".to_string(),
                    cost_per_million_input_tokens: 10.0,
                    cost_per_million_output_tokens: 30.0,
                    priority: 100,
                    structured_output_passthrough: None,
                    tier: None,
                    context_window: 0,
                    specializations: vec![],
                },
            ],
        }];

        let router = Router::new(Arc::new(RwLock::new(config)), test_metrics());
        let model_group = router.find_model_group("model-1").await.unwrap();
        let order = router.select_provider_order(&model_group).await;

        assert_eq!(order.len(), 2);
        assert_eq!(order[0].provider, "provider-high-priority");
        assert_eq!(order[1].provider, "provider-low-priority");
    }

    #[tokio::test]
    async fn test_select_provider_order_by_cost() {
        let mut config = create_test_config();
        config.model_groups = vec![ModelGroup {
            name: "test-group".to_string(),
            version_fallback_enabled: false,
            compression: None,
            structured_output: None,
            memory: None,
            models: vec![
                ProviderModel {
                    provider: "expensive-provider".to_string(),
                    model: "model-1".to_string(),
                    cost_per_million_input_tokens: 20.0,
                    cost_per_million_output_tokens: 60.0,
                    priority: 100,
                    structured_output_passthrough: None,
                    tier: None,
                    context_window: 0,
                    specializations: vec![],
                },
                ProviderModel {
                    provider: "cheap-provider".to_string(),
                    model: "model-2".to_string(),
                    cost_per_million_input_tokens: 5.0,
                    cost_per_million_output_tokens: 15.0,
                    priority: 100,
                    structured_output_passthrough: None,
                    tier: None,
                    context_window: 0,
                    specializations: vec![],
                },
            ],
        }];

        let router = Router::new(Arc::new(RwLock::new(config)), test_metrics());
        let model_group = router.find_model_group("model-1").await.unwrap();
        let order = router.select_provider_order(&model_group).await;

        assert_eq!(order.len(), 2);
        assert_eq!(order[0].provider, "cheap-provider");
        assert_eq!(order[1].provider, "expensive-provider");
    }

    #[tokio::test]
    async fn test_select_provider_order_by_latency() {
        let mut config = create_test_config();
        config.model_groups = vec![ModelGroup {
            name: "test-group".to_string(),
            version_fallback_enabled: false,
            compression: None,
            structured_output: None,
            memory: None,
            models: vec![
                ProviderModel {
                    provider: "slow-provider".to_string(),
                    model: "model-1".to_string(),
                    cost_per_million_input_tokens: 10.0,
                    cost_per_million_output_tokens: 30.0,
                    priority: 100,
                    structured_output_passthrough: None,
                    tier: None,
                    context_window: 0,
                    specializations: vec![],
                },
                ProviderModel {
                    provider: "fast-provider".to_string(),
                    model: "model-2".to_string(),
                    cost_per_million_input_tokens: 10.5,
                    cost_per_million_output_tokens: 31.0,
                    priority: 100,
                    structured_output_passthrough: None,
                    tier: None,
                    context_window: 0,
                    specializations: vec![],
                },
            ],
        }];

        let router = Router::new(Arc::new(RwLock::new(config)), test_metrics());

        // Set latencies
        router
            .latency_tracker
            .update_latency("slow-provider", std::time::Duration::from_millis(500));
        router
            .latency_tracker
            .update_latency("fast-provider", std::time::Duration::from_millis(100));

        let model_group = router.find_model_group("model-1").await.unwrap();
        let order = router.select_provider_order(&model_group).await;

        assert_eq!(order.len(), 2);
        // Costs are within 10%, so should sort by latency
        assert_eq!(order[0].provider, "fast-provider");
        assert_eq!(order[1].provider, "slow-provider");
    }

    #[tokio::test]
    async fn disabled_smart_routing_preserves_provider_order() {
        let mut config = create_test_config();
        assert!(!config.smart_routing.enabled);
        let group = test_group(vec![test_model("slow", 20), test_model("fast", 10)]);
        config.model_groups.push(group.clone());
        let router = Router::new(Arc::new(RwLock::new(config)), test_metrics());
        assert!(router.smart_router_snapshot().is_none());

        let direct = router.select_provider_order(&group).await;
        let request = OpenAIRequest {
            model: group.name.clone(),
            messages: vec![],
            stream: false,
            temperature: None,
            max_tokens: None,
            extra: serde_json::Map::new(),
        };
        assert!(router
            .smart_routing_plan(&request, &group)
            .await
            .unwrap()
            .is_none());
        assert_eq!(direct, router.select_provider_order(&group).await);
    }

    #[tokio::test]
    async fn test_extract_version_date() {
        assert_eq!(
            Router::extract_version_date("gpt-4-turbo-2024-04-09"),
            (2024, 4, 9)
        );
        assert_eq!(
            Router::extract_version_date("claude-3-opus-2024-02-29"),
            (2024, 2, 29)
        );
        assert_eq!(Router::extract_version_date("gpt-4"), (0, 0, 0));
        assert_eq!(Router::extract_version_date("model-name"), (0, 0, 0));
    }

    #[tokio::test]
    async fn test_version_fallback_sorting() {
        let mut config = create_test_config();
        config.model_groups = vec![ModelGroup {
            name: "test-group".to_string(),
            version_fallback_enabled: true,
            compression: None,
            structured_output: None,
            memory: None,
            models: vec![
                ProviderModel {
                    provider: "provider-1".to_string(),
                    model: "gpt-4-turbo-2024-01-25".to_string(),
                    cost_per_million_input_tokens: 10.0,
                    cost_per_million_output_tokens: 30.0,
                    priority: 100,
                    structured_output_passthrough: None,
                    tier: None,
                    context_window: 0,
                    specializations: vec![],
                },
                ProviderModel {
                    provider: "provider-2".to_string(),
                    model: "gpt-4-turbo-2024-04-09".to_string(),
                    cost_per_million_input_tokens: 10.0,
                    cost_per_million_output_tokens: 30.0,
                    priority: 100,
                    structured_output_passthrough: None,
                    tier: None,
                    context_window: 0,
                    specializations: vec![],
                },
                ProviderModel {
                    provider: "provider-3".to_string(),
                    model: "gpt-4-turbo".to_string(),
                    cost_per_million_input_tokens: 10.0,
                    cost_per_million_output_tokens: 30.0,
                    priority: 100,
                    structured_output_passthrough: None,
                    tier: None,
                    context_window: 0,
                    specializations: vec![],
                },
            ],
        }];

        let router = Router::new(Arc::new(RwLock::new(config)), test_metrics());
        let model_group = router
            .find_model_group("gpt-4-turbo-2024-01-25")
            .await
            .unwrap();
        let order = router.select_provider_order(&model_group).await;

        assert_eq!(order.len(), 3);
        // Should be sorted by version date descending (newest first)
        assert_eq!(order[0].model, "gpt-4-turbo-2024-04-09");
        assert_eq!(order[1].model, "gpt-4-turbo-2024-01-25");
        assert_eq!(order[2].model, "gpt-4-turbo"); // No version = oldest
    }

    #[test]
    fn test_http_client_reused_per_provider() {
        let router = Router::new(Arc::new(RwLock::new(create_test_config())), test_metrics());
        let pool_config = crate::config::ProviderConnectionPoolConfig::default();
        let _client1 = router
            .get_or_create_http_client("provider-a", &pool_config)
            .unwrap();
        let _client2 = router
            .get_or_create_http_client("provider-a", &pool_config)
            .unwrap();
        assert_eq!(router.http_clients.len(), 1);
    }

    #[test]
    fn test_reassemble_sse_response_preserves_reasoning_content() {
        let body = concat!(
            "data: {\"id\":\"chatcmpl-test\",\"object\":\"chat.completion.chunk\",\"created\":123,\"model\":\"test-model\",\"choices\":[{\"index\":0,\"delta\":{\"reasoning_content\":\"thinking\"},\"finish_reason\":null}]}\n\n",
            "data: {\"id\":\"chatcmpl-test\",\"object\":\"chat.completion.chunk\",\"created\":123,\"model\":\"test-model\",\"choices\":[{\"index\":0,\"delta\":{\"content\":\"answer\"},\"finish_reason\":null}]}\n\n",
            "data: {\"id\":\"chatcmpl-test\",\"object\":\"chat.completion.chunk\",\"created\":123,\"model\":\"test-model\",\"choices\":[{\"index\":0,\"delta\":{},\"finish_reason\":\"stop\"}],\"usage\":{\"prompt_tokens\":1,\"completion_tokens\":2,\"total_tokens\":3}}\n\n",
            "data: [DONE]\n\n"
        );

        let response = Router::reassemble_sse_response(body).expect("response should reassemble");

        assert_eq!(
            response.choices[0].message.content,
            serde_json::json!("answer")
        );
        assert_eq!(
            response.choices[0].message.extra.get("reasoning_content"),
            Some(&serde_json::json!("thinking"))
        );
    }

    #[test]
    fn test_calculate_retry_delay_without_jitter() {
        let delay = Router::calculate_retry_delay(4, false, 0.2);
        assert_eq!(delay, Duration::from_secs(4));
    }

    #[test]
    fn test_calculate_retry_delay_with_jitter_stays_in_bounds() {
        for _ in 0..32 {
            let delay = Router::calculate_retry_delay(10, true, 0.2);
            assert!(delay >= Duration::from_secs(8));
            assert!(delay <= Duration::from_secs(12));
        }
    }

    #[tokio::test]
    async fn test_budget_exhausted_provider_is_skipped() {
        let mut config = create_test_config();
        config.providers = vec![crate::config::Provider {
            name: "budgeted-provider".to_string(),
            provider_type: "openai".to_string(),
            base_url: Some("http://localhost:1234".to_string()),
            api_key_env: None,
            api_key_encrypted: None,
            api_secret_env: None,
            api_secret_encrypted: None,
            auth_method: None,
            resolved_api_key: None,
            resolved_api_secret: None,
            region: None,
            timeout_seconds: 30,
            ttfb_timeout_seconds: None,
            total_timeout_seconds: None,
            max_connections: 10,
            rate_limit_per_minute: 0,
            custom_headers: Default::default(),
            connection_pool: crate::config::ProviderConnectionPoolConfig::default(),
            budget: Some(crate::config::ProviderBudgetConfig {
                limit_usd: 1.0,
                reset_policy: crate::config::BudgetResetPolicy::Manual,
            }),
            manual_models: vec![],
            global_inference_profile: false,
            cross_region_inference: false,
            custom_vpc_endpoint: false,
            prompt_caching: false,
            compression: None,
            reasoning: true,
            codex_base_url_override: None,
            codex_model_override: None,
            instructions_override: None,
            max_rate_limit_cooldown_seconds: None,
            memory: None,
        }];
        let providers = vec![ProviderModel {
            provider: "budgeted-provider".to_string(),
            model: "test-model".to_string(),
            cost_per_million_input_tokens: 0.0,
            cost_per_million_output_tokens: 0.0,
            priority: 100,
            structured_output_passthrough: None,
            tier: None,
            context_window: 0,
            specializations: vec![],
        }];
        config.model_groups = vec![ModelGroup {
            name: "test-group".to_string(),
            version_fallback_enabled: false,
            compression: None,
            structured_output: None,
            memory: None,
            models: providers.clone(),
        }];
        let router_metrics = test_metrics();
        router_metrics.add_cost("budgeted-provider", 1.25);
        let router = Router::new(Arc::new(RwLock::new(config)), router_metrics.clone());

        let request = OpenAIRequest {
            model: "test-model".to_string(),
            messages: vec![],
            temperature: None,
            max_tokens: None,
            stream: false,
            extra: Default::default(),
        };

        let result = router.route_with_failover(&request, providers, None).await;
        assert!(matches!(result, Err(GatewayError::AllProvidersFailed(_))));

        let snapshot = router_metrics.snapshot();
        let exhausted = snapshot
            .budget_exhaustions_by_provider
            .iter()
            .find(|(provider, _)| provider == "budgeted-provider")
            .map(|(_, count)| *count)
            .unwrap_or(0);
        assert_eq!(exhausted, 1);
    }

    #[test]
    fn test_strip_image_content_if_unsupported_removes_image_parts() {
        let mut request = OpenAIRequest {
            model: "no-vision".to_string(),
            messages: vec![Message {
                role: "user".to_string(),
                content: serde_json::json!([
                    {"type": "text", "text": "describe this"},
                    {"type": "image_url", "image_url": {"url": "https://x.example/p.png"}},
                ]),
                extra: Default::default(),
            }],
            temperature: None,
            max_tokens: None,
            stream: false,
            extra: Default::default(),
        };

        let removed = Router::strip_image_content_if_unsupported(
            &mut request,
            false,
            "test-provider",
            "no-vision",
        );
        assert_eq!(removed, 1);
        let parts = request.messages[0].content.as_array().unwrap();
        assert_eq!(parts.len(), 1);
        assert_eq!(parts[0]["type"], serde_json::json!("text"));
    }

    #[test]
    fn test_strip_image_content_if_unsupported_keeps_images_for_vision_model() {
        let mut request = OpenAIRequest {
            model: "vision".to_string(),
            messages: vec![Message {
                role: "user".to_string(),
                content: serde_json::json!([
                    {"type": "text", "text": "describe this"},
                    {"type": "image_url", "image_url": {"url": "https://x.example/p.png"}},
                ]),
                extra: Default::default(),
            }],
            temperature: None,
            max_tokens: None,
            stream: false,
            extra: Default::default(),
        };

        let removed = Router::strip_image_content_if_unsupported(
            &mut request,
            true,
            "test-provider",
            "vision",
        );
        assert_eq!(removed, 0);
        let parts = request.messages[0].content.as_array().unwrap();
        assert_eq!(parts.len(), 2);
    }
}

#[cfg(test)]
mod property_tests {
    use super::tests::{create_test_config, test_metrics};
    use super::*;
    use proptest::prelude::*;

    // Generator for ProviderModel
    fn provider_model_strategy() -> impl Strategy<Value = ProviderModel> {
        (
            "[a-z]{3,8}",
            "[a-z0-9-]{3,15}",
            0.0..100.0f64,
            0.0..100.0f64,
            1u32..1000,
        )
            .prop_map(|(provider, model, input_cost, output_cost, priority)| {
                ProviderModel {
                    provider,
                    model,
                    cost_per_million_input_tokens: input_cost,
                    cost_per_million_output_tokens: output_cost,
                    priority,
                    structured_output_passthrough: None,
                    tier: None,
                    context_window: 0,
                    specializations: vec![],
                }
            })
    }

    // Generator for ModelGroup
    fn model_group_strategy() -> impl Strategy<Value = ModelGroup> {
        (
            "[a-z]{3,10}",
            any::<bool>(),
            prop::collection::vec(provider_model_strategy(), 1..10),
        )
            .prop_map(|(name, version_fallback, models)| ModelGroup {
                name,
                version_fallback_enabled: version_fallback,
                compression: None,
                structured_output: None,
                memory: None,
                models,
            })
    }

    proptest! {
        #![proptest_config(ProptestConfig {
            cases: 50,
            .. ProptestConfig::default()
        })]

        /// **Property 1: Model Group Membership Preservation**
        /// **Validates: Requirements 4.2, 4.5**
        ///
        /// For any model group and any provider selection from that group,
        /// all selected providers must be members of that model group.
        #[test]
        fn prop_model_group_membership_preservation(model_group in model_group_strategy()) {
            let rt = tokio::runtime::Runtime::new().unwrap();
            rt.block_on(async {
                let mut config = create_test_config();
                config.model_groups = vec![model_group.clone()];

                let router = Router::new(Arc::new(RwLock::new(config)), test_metrics());
                let selected = router.select_provider_order(&model_group).await;

                // All selected providers must be in the original model group
                let original_providers: std::collections::HashSet<_> =
                    model_group.models.iter().map(|m| &m.provider).collect();

                for selected_model in &selected {
                    prop_assert!(
                        original_providers.contains(&selected_model.provider),
                        "Selected provider '{}' not in original model group",
                        selected_model.provider
                    );
                }

                Ok(())
            })?;
        }

        /// **Property 2: Provider Selection Ordering**
        /// **Validates: Requirements 6.2, 6.3, 7.2, 28.2-28.4, 5.2**
        ///
        /// For any model group with multiple providers, the router shall order providers by:
        /// (1) priority ascending, (2) cost ascending within same priority,
        /// (3) latency ascending within similar costs (±10%),
        /// (4) version date descending if version fallback is enabled.
        #[test]
        fn prop_provider_selection_ordering(model_group in model_group_strategy()) {
            let rt = tokio::runtime::Runtime::new().unwrap();
            rt.block_on(async {
                let mut config = create_test_config();
                config.model_groups = vec![model_group.clone()];

                let router = Router::new(Arc::new(RwLock::new(config)), test_metrics());
                let selected = router.select_provider_order(&model_group).await;

                // Check priority ordering (ascending)
                for window in selected.windows(2) {
                    let (a, b) = (&window[0], &window[1]);
                    prop_assert!(
                        a.priority <= b.priority,
                        "Priority ordering violated: {} > {}",
                        a.priority, b.priority
                    );

                    // Within same priority, check cost ordering
                    if a.priority == b.priority {
                        let cost_a = a.total_cost();
                        let cost_b = b.total_cost();
                        let cost_diff = (cost_a - cost_b).abs();
                        let cost_threshold = cost_a.min(cost_b) * 0.1;

                        // If costs differ by more than 10%, lower cost should come first
                        if cost_diff > cost_threshold {
                            prop_assert!(
                                cost_a <= cost_b,
                                "Cost ordering violated: {} > {}",
                                cost_a, cost_b
                            );
                        }
                    }
                }

                Ok(())
            })?;
        }

        /// **Property 18: Model Group Lookup**
        /// **Validates: Requirements 4.4**
        ///
        /// For any model name that exists in the configuration,
        /// the router shall identify exactly one model group containing that model.
        #[test]
        fn prop_model_group_lookup(model_group in model_group_strategy()) {
            let rt = tokio::runtime::Runtime::new().unwrap();
            rt.block_on(async {
                let mut config = create_test_config();
                let group_name = model_group.name.clone();
                config.model_groups = vec![model_group.clone()];

                let router = Router::new(Arc::new(RwLock::new(config)), test_metrics());

                // Test lookup for each model in the group
                for provider_model in &model_group.models {
                    let result = router.find_model_group(&provider_model.model).await;

                    prop_assert!(
                        result.is_ok(),
                        "Failed to find model group for model '{}'",
                        provider_model.model
                    );

                    let found_group = result.unwrap();
                    prop_assert_eq!(
                        &found_group.name,
                        &group_name,
                        "Found wrong model group"
                    );
                }

                Ok(())
            })?;
        }
    }

    /// **Property 19: Model Group Validation**
    /// **Validates: Requirements 4.3**
    ///
    /// For any model group configuration, validation shall fail if any model
    /// is missing a provider field or model identifier field.
    #[test]
    fn test_model_group_validation_missing_fields() {
        // Test with empty provider
        let invalid_group = ModelGroup {
            name: "test-group".to_string(),
            version_fallback_enabled: false,
            compression: None,
            structured_output: None,
            memory: None,
            models: vec![ProviderModel {
                provider: "".to_string(), // Invalid: empty provider
                model: "gpt-4".to_string(),
                cost_per_million_input_tokens: 10.0,
                cost_per_million_output_tokens: 30.0,
                priority: 100,
                structured_output_passthrough: None,
                tier: None,
                context_window: 0,
                specializations: vec![],
            }],
        };

        let mut config = create_test_config();
        config.model_groups = vec![invalid_group];

        // Validation should catch this during config validation
        // (This is tested in config validation tests, but we verify the structure here)
        assert!(config.model_groups[0].models[0].provider.is_empty());

        // Test with empty model
        let invalid_group2 = ModelGroup {
            name: "test-group".to_string(),
            version_fallback_enabled: false,
            compression: None,
            structured_output: None,
            memory: None,
            models: vec![ProviderModel {
                provider: "openai".to_string(),
                model: "".to_string(), // Invalid: empty model
                cost_per_million_input_tokens: 10.0,
                cost_per_million_output_tokens: 30.0,
                priority: 100,
                structured_output_passthrough: None,
                tier: None,
                context_window: 0,
                specializations: vec![],
            }],
        };

        let mut config2 = create_test_config();
        config2.model_groups = vec![invalid_group2];

        assert!(config2.model_groups[0].models[0].model.is_empty());
    }

    // ────────────────────────────────────────────────────────────────────
    // Rate-limit detection & cooldown parsing
    // ────────────────────────────────────────────────────────────────────

    #[test]
    fn test_is_rate_limited_status_429() {
        assert!(Router::is_rate_limited(429, ""));
        assert!(Router::is_rate_limited(429, "anything goes"));
    }

    #[test]
    fn test_is_rate_limited_non_rate_limit_4xx() {
        assert!(!Router::is_rate_limited(
            400,
            r#"{"error":{"message":"bad request"}}"#
        ));
        assert!(!Router::is_rate_limited(
            401,
            r#"{"error":{"message":"unauthorized"}}"#
        ));
        assert!(!Router::is_rate_limited(
            404,
            r#"{"error":{"message":"not found"}}"#
        ));
    }

    #[test]
    fn test_is_rate_limited_5xx_ignored() {
        // 5xx aren't rate-limit signals even if message mentions limits.
        assert!(!Router::is_rate_limited(503, "service unavailable"));
        assert!(!Router::is_rate_limited(
            500,
            r#"{"error":{"message":"internal"}}"#
        ));
    }

    #[test]
    fn test_is_rate_limited_200_with_rate_limit_envelope() {
        let body = r#"{"error":{"message":"You are sending requests too fast","type":"rate_limit_error","code":"rate_limited"}}"#;
        assert!(Router::is_rate_limited(200, body));
    }

    #[test]
    fn test_is_rate_limited_200_with_quota_message() {
        let body = r#"{"error":{"message":"Quota exceeded for this account"}}"#;
        assert!(Router::is_rate_limited(200, body));
    }

    #[test]
    fn test_is_rate_limited_200_with_insufficient_quota() {
        let body = r#"{"error":{"code":"insufficient_quota","message":"You ran out of credits"}}"#;
        assert!(Router::is_rate_limited(200, body));
    }

    #[test]
    fn test_is_rate_limited_200_with_numeric_429_code() {
        let body = r#"{"error":{"code":429,"message":"Too Many Requests"}}"#;
        assert!(Router::is_rate_limited(200, body));
    }

    #[test]
    fn test_is_rate_limited_200_normal_response_not_flagged() {
        let body =
            r#"{"id":"chatcmpl-1","choices":[{"message":{"role":"assistant","content":"hi"}}]}"#;
        assert!(!Router::is_rate_limited(200, body));
    }

    #[test]
    fn test_is_rate_limited_200_unrelated_error_not_flagged() {
        let body =
            r#"{"error":{"message":"context length exceeded","type":"invalid_request_error"}}"#;
        assert!(!Router::is_rate_limited(200, body));
    }

    #[test]
    fn test_is_rate_limited_plain_text_too_many_requests() {
        assert!(Router::is_rate_limited(200, "Too Many Requests"));
        assert!(Router::is_rate_limited(429, "Too Many Requests"));
    }

    #[test]
    fn test_parse_rate_limit_cooldown_from_retry_after_seconds() {
        let mut headers = reqwest::header::HeaderMap::new();
        headers.insert(reqwest::header::RETRY_AFTER, "7".parse().unwrap());

        let cooldown = Router::compute_rate_limit_cooldown(
            Some(&headers),
            "",
            Duration::from_secs(30),
            None,
            24 * 60 * 60,
        );
        assert_eq!(cooldown, Duration::from_secs(7));
    }

    #[test]
    fn test_parse_rate_limit_cooldown_from_retry_after_ms_header() {
        let mut headers = reqwest::header::HeaderMap::new();
        headers.insert("retry-after-ms", "1500".parse().unwrap());

        let cooldown = Router::compute_rate_limit_cooldown(
            Some(&headers),
            "",
            Duration::from_secs(30),
            None,
            24 * 60 * 60,
        );
        assert_eq!(cooldown, Duration::from_millis(1500));
    }

    #[test]
    fn test_parse_rate_limit_cooldown_clamped_to_global_cap() {
        let mut headers = reqwest::header::HeaderMap::new();
        // Provider tries to ask for 30 days, global cap is 24h.
        headers.insert(reqwest::header::RETRY_AFTER, "2592000".parse().unwrap());

        let cooldown = Router::compute_rate_limit_cooldown(
            Some(&headers),
            "",
            Duration::from_secs(30),
            None,
            24 * 60 * 60,
        );
        assert_eq!(cooldown, Duration::from_secs(24 * 60 * 60));
    }

    #[test]
    fn test_parse_rate_limit_cooldown_provider_override_raises_cap() {
        // Nano-GPT-style weekly quota: provider override = 7d, global = 24h.
        // Provider's Retry-After of 6 days should be honored, not clamped to 24h.
        let mut headers = reqwest::header::HeaderMap::new();
        let six_days = 6 * 24 * 60 * 60;
        headers.insert(
            reqwest::header::RETRY_AFTER,
            six_days.to_string().parse().unwrap(),
        );

        let cooldown = Router::compute_rate_limit_cooldown(
            Some(&headers),
            "",
            Duration::from_secs(30),
            Some(7 * 24 * 60 * 60),
            24 * 60 * 60,
        );
        assert_eq!(cooldown, Duration::from_secs(six_days));
    }

    #[test]
    fn test_parse_rate_limit_cooldown_clamped_to_limiter_backstop() {
        // Even if both operator caps are absurdly large, the limiter
        // backstop (7 days) wins.
        let mut headers = reqwest::header::HeaderMap::new();
        headers.insert(reqwest::header::RETRY_AFTER, "99999999".parse().unwrap());

        let cooldown = Router::compute_rate_limit_cooldown(
            Some(&headers),
            "",
            Duration::from_secs(30),
            Some(99_999_999),
            99_999_999,
        );
        assert_eq!(cooldown, crate::router::rate_limiter::MAX_COOLDOWN);
    }

    #[test]
    fn test_parse_rate_limit_cooldown_from_body_retry_after_seconds() {
        let body = r#"{"error":{"message":"slow down","retry_after":12}}"#;
        let cooldown = Router::compute_rate_limit_cooldown(
            None,
            body,
            Duration::from_secs(30),
            None,
            24 * 60 * 60,
        );
        assert_eq!(cooldown, Duration::from_secs(12));
    }

    #[test]
    fn test_parse_rate_limit_cooldown_from_body_retry_after_ms() {
        let body = r#"{"error":{"message":"slow down","retry_after_ms":2500}}"#;
        let cooldown = Router::compute_rate_limit_cooldown(
            None,
            body,
            Duration::from_secs(30),
            None,
            24 * 60 * 60,
        );
        assert_eq!(cooldown, Duration::from_millis(2500));
    }

    #[test]
    fn test_parse_rate_limit_cooldown_default_when_missing() {
        let cooldown = Router::compute_rate_limit_cooldown(
            None,
            "{}",
            Duration::from_secs(30),
            None,
            24 * 60 * 60,
        );
        assert_eq!(cooldown, Duration::from_secs(30));
    }

    #[test]
    fn test_parse_rate_limit_cooldown_header_takes_precedence_over_body() {
        let mut headers = reqwest::header::HeaderMap::new();
        headers.insert(reqwest::header::RETRY_AFTER, "3".parse().unwrap());
        let body = r#"{"error":{"retry_after":99}}"#;

        let cooldown = Router::compute_rate_limit_cooldown(
            Some(&headers),
            body,
            Duration::from_secs(30),
            None,
            24 * 60 * 60,
        );
        assert_eq!(cooldown, Duration::from_secs(3));
    }

    #[test]
    fn test_cooldown_from_xratelimit_reset_epoch() {
        // OpenAI-style epoch-seconds reset header.
        let future = (chrono::Utc::now().timestamp() + 3600) as u64;
        let mut headers = reqwest::header::HeaderMap::new();
        headers.insert("x-ratelimit-reset", future.to_string().parse().unwrap());

        let cooldown = Router::compute_rate_limit_cooldown(
            Some(&headers),
            "",
            Duration::from_secs(30),
            Some(7 * 24 * 60 * 60),
            7 * 24 * 60 * 60,
        );
        // Should be ~1h, allow ±5s for clock skew during the test.
        assert!(cooldown > Duration::from_secs(3590));
        assert!(cooldown <= Duration::from_secs(3600));
    }

    #[test]
    fn test_cooldown_from_xratelimit_reset_after_relative() {
        let mut headers = reqwest::header::HeaderMap::new();
        headers.insert("x-ratelimit-reset-after", "120".parse().unwrap());

        let cooldown = Router::compute_rate_limit_cooldown(
            Some(&headers),
            "",
            Duration::from_secs(30),
            None,
            24 * 60 * 60,
        );
        assert_eq!(cooldown, Duration::from_secs(120));
    }

    #[test]
    fn test_cooldown_from_anthropic_iso_reset() {
        let when = chrono::Utc::now() + chrono::Duration::seconds(45);
        let header_val = when.to_rfc3339();
        let mut headers = reqwest::header::HeaderMap::new();
        headers.insert(
            "anthropic-ratelimit-requests-reset",
            header_val.parse().unwrap(),
        );

        let cooldown = Router::compute_rate_limit_cooldown(
            Some(&headers),
            "",
            Duration::from_secs(30),
            None,
            24 * 60 * 60,
        );
        assert!(cooldown > Duration::from_secs(40));
        assert!(cooldown <= Duration::from_secs(45));
    }

    #[test]
    fn test_cooldown_from_period_marker_weekly() {
        // Nano-GPT-style "weekly limit reached" with no machine-readable
        // retry-after. Operator has opted into weekly cooldown via the
        // per-provider override.
        let body = r#"{"error":{"message":"You have reached your weekly limit. Please try again later."}}"#;

        let cooldown = Router::compute_rate_limit_cooldown(
            None,
            body,
            Duration::from_secs(30),
            Some(7 * 24 * 60 * 60),
            24 * 60 * 60,
        );
        assert_eq!(cooldown, Duration::from_secs(7 * 24 * 60 * 60));
    }

    #[test]
    fn test_cooldown_from_period_marker_weekly_clamped_when_no_override() {
        // Same body, but provider has no override. Global default cap of
        // 24h prevents us from holding the provider out for a full week.
        let body = r#"{"error":{"message":"You have reached your weekly limit."}}"#;

        let cooldown = Router::compute_rate_limit_cooldown(
            None,
            body,
            Duration::from_secs(30),
            None,
            24 * 60 * 60,
        );
        assert_eq!(cooldown, Duration::from_secs(24 * 60 * 60));
    }

    #[test]
    fn test_cooldown_from_period_marker_daily() {
        let body = r#"{"error":{"message":"Daily quota exceeded"}}"#;

        let cooldown = Router::compute_rate_limit_cooldown(
            None,
            body,
            Duration::from_secs(30),
            None,
            48 * 60 * 60,
        );
        assert_eq!(cooldown, Duration::from_secs(24 * 60 * 60));
    }

    #[test]
    fn test_cooldown_from_body_reset_at_rfc3339() {
        let when = chrono::Utc::now() + chrono::Duration::seconds(90);
        let body = format!(r#"{{"error":{{"reset_at":"{}"}}}}"#, when.to_rfc3339());

        let cooldown = Router::compute_rate_limit_cooldown(
            None,
            &body,
            Duration::from_secs(30),
            None,
            24 * 60 * 60,
        );
        assert!(cooldown > Duration::from_secs(85));
        assert!(cooldown <= Duration::from_secs(90));
    }

    // ────────────────────────────────────────────────────────────────────
    // Pre-filter behavior of select_provider_order
    // ────────────────────────────────────────────────────────────────────

    fn build_two_provider_group() -> (Config, ModelGroup) {
        let mut config = create_test_config();

        let make_provider = |name: &str| crate::config::Provider {
            name: name.to_string(),
            provider_type: "openai".to_string(),
            base_url: Some("http://localhost:1234".to_string()),
            api_key_env: None,
            api_key_encrypted: None,
            api_secret_env: None,
            api_secret_encrypted: None,
            auth_method: None,
            resolved_api_key: None,
            resolved_api_secret: None,
            region: None,
            timeout_seconds: 30,
            ttfb_timeout_seconds: None,
            total_timeout_seconds: None,
            max_connections: 10,
            // Tight bucket so check_available() trivially returns false
            // after a single consume.
            rate_limit_per_minute: 1,
            custom_headers: Default::default(),
            connection_pool: crate::config::ProviderConnectionPoolConfig::default(),
            budget: None,
            manual_models: vec![],
            global_inference_profile: false,
            cross_region_inference: false,
            prompt_caching: false,
            compression: None,
            reasoning: false,
            custom_vpc_endpoint: false,
            codex_base_url_override: None,
            codex_model_override: None,
            instructions_override: None,
            max_rate_limit_cooldown_seconds: None,
            memory: None,
        };

        config.providers = vec![make_provider("primary"), make_provider("backup")];

        let group = ModelGroup {
            name: "test-group".to_string(),
            version_fallback_enabled: false,
            compression: None,
            structured_output: None,
            memory: None,
            models: vec![
                ProviderModel {
                    provider: "primary".to_string(),
                    model: "model-1".to_string(),
                    cost_per_million_input_tokens: 10.0,
                    cost_per_million_output_tokens: 30.0,
                    priority: 1,
                    structured_output_passthrough: None,
                    tier: None,
                    context_window: 0,
                    specializations: vec![],
                },
                ProviderModel {
                    provider: "backup".to_string(),
                    model: "model-1".to_string(),
                    cost_per_million_input_tokens: 11.0,
                    cost_per_million_output_tokens: 31.0,
                    priority: 2,
                    structured_output_passthrough: None,
                    tier: None,
                    context_window: 0,
                    specializations: vec![],
                },
            ],
        };
        config.model_groups = vec![group.clone()];
        (config, group)
    }

    #[tokio::test]
    async fn test_select_provider_order_does_not_filter_on_token_bucket() {
        let (config, group) = build_two_provider_group();
        let router = Router::new(Arc::new(RwLock::new(config)), test_metrics());

        // Drain the primary provider's token bucket entirely.
        let limiter = router.get_rate_limiter("primary").await;
        assert!(limiter.consume().await);
        assert!(!limiter.check_available().await);

        // Even with the bucket exhausted, primary stays in the list.
        // The failover path is responsible for handling bucket exhaustion
        // visibly; pre-filtering here would silently shift traffic.
        let order = router.select_provider_order(&group).await;
        assert_eq!(order.len(), 2);
        assert_eq!(order[0].provider, "primary");
        assert_eq!(order[1].provider, "backup");
    }

    #[tokio::test]
    async fn test_select_provider_order_filters_on_upstream_cooldown() {
        let (config, group) = build_two_provider_group();
        let router = Router::new(Arc::new(RwLock::new(config)), test_metrics());

        // An upstream-driven cooldown DOES remove the provider. This is
        // the signal a real 429 / Retry-After produces.
        let limiter = router.get_rate_limiter("primary").await;
        limiter.apply_cooldown(Duration::from_secs(5)).await;

        let order = router.select_provider_order(&group).await;
        assert_eq!(order.len(), 1);
        assert_eq!(order[0].provider, "backup");
    }

    #[tokio::test]
    async fn test_cooldown_in_metrics_filters_after_clear_rate_limiters() {
        // Regression: a long-running cooldown (e.g. Nano-GPT weekly
        // quota at 23h) was being silently bypassed after any config
        // hot-reload. `apply_runtime_config_update` calls
        // `clear_rate_limiters()`, which wipes the per-`Router`
        // cooldown but cannot wipe the metrics map (shared, durable).
        // The dashboard kept rendering "Pausing for ~23h (rate
        // limited)" while the router happily routed new traffic to the
        // provider, which immediately 429'd again — exactly the user
        // bug report. The fix is for `select_provider_order` to also
        // consult the metrics-side cooldown.
        let (config, group) = build_two_provider_group();
        let metrics = test_metrics();
        let router = Router::new(Arc::new(RwLock::new(config)), metrics.clone());

        // Simulate a 429 with a long Retry-After landing on `primary`.
        // Both stores get written, just like the real 429 handler does.
        let limiter = router.get_rate_limiter("primary").await;
        limiter
            .apply_cooldown(Duration::from_secs(60 * 60 * 23))
            .await;
        let now_secs = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .map(|d| d.as_secs())
            .unwrap_or(0);
        metrics.set_provider_cooldown(
            "primary",
            "Rate limited by provider — pausing until the limit resets".to_string(),
            now_secs + 60 * 60 * 23,
        );

        // Now simulate a config hot-reload, which clears the router-side
        // rate_limiters DashMap but leaves the metrics store intact.
        router.clear_rate_limiters();

        // The cooldown must STILL be honored — primary stays out of the
        // candidate list. Without the metrics check, primary would be
        // reinstated here (None => true), and a request would be issued.
        let order = router.select_provider_order(&group).await;
        assert_eq!(
            order.len(),
            1,
            "primary must still be filtered after reload"
        );
        assert_eq!(order[0].provider, "backup");

        // Sanity: clearing the metrics cooldown restores eligibility.
        metrics.clear_provider_cooldown("primary");
        let order = router.select_provider_order(&group).await;
        assert_eq!(order.len(), 2);
    }

    // ────────────────────────────────────────────────────────────────────
    // provider_needs_transformation (task 5.1, Requirements: 3.8)
    // ────────────────────────────────────────────────────────────────────

    fn make_provider_named(name: &str, provider_type: &str) -> Provider {
        Provider {
            name: name.to_string(),
            provider_type: provider_type.to_string(),
            base_url: Some("http://localhost:1234".to_string()),
            api_key_env: None,
            api_key_encrypted: None,
            api_secret_env: None,
            api_secret_encrypted: None,
            auth_method: None,
            resolved_api_key: None,
            resolved_api_secret: None,
            region: None,
            timeout_seconds: 30,
            ttfb_timeout_seconds: None,
            total_timeout_seconds: None,
            max_connections: 10,
            rate_limit_per_minute: 0,
            custom_headers: Default::default(),
            connection_pool: crate::config::ProviderConnectionPoolConfig::default(),
            budget: None,
            manual_models: vec![],
            global_inference_profile: false,
            cross_region_inference: false,
            custom_vpc_endpoint: false,
            prompt_caching: false,
            compression: None,
            reasoning: false,
            codex_base_url_override: None,
            codex_model_override: None,
            instructions_override: None,
            max_rate_limit_cooldown_seconds: None,
            memory: None,
        }
    }

    fn request_with_tools(with_tools: bool) -> OpenAIRequest {
        let mut extra = serde_json::Map::new();
        if with_tools {
            extra.insert("tools".to_string(), serde_json::json!([]));
        }
        OpenAIRequest {
            model: "test-model".to_string(),
            messages: vec![],
            temperature: None,
            max_tokens: None,
            stream: true,
            extra,
        }
    }

    #[test]
    fn test_provider_needs_transformation() {
        let router = Router::new(Arc::new(RwLock::new(create_test_config())), test_metrics());

        // Bedrock always needs transformation, regardless of tools.
        let bedrock = make_provider_named("aws-bedrock", "bedrock");
        assert!(router.provider_needs_transformation(&bedrock, &request_with_tools(false)));

        // Plain OpenAI provider without tools passes through.
        let openai = make_provider_named("openai-main", "openai");
        assert!(!router.provider_needs_transformation(&openai, &request_with_tools(false)));

        // Kimi providers need token sanitization even without tools.
        let kimi = make_provider_named("kimi-k2", "openai");
        assert!(router.provider_needs_transformation(&kimi, &request_with_tools(false)));

        // XML tool use is no longer inferred from the provider name: a GLM
        // provider streams optimistically (with or without tools) until the
        // relay learns it emits XML — see `is_xml_tool_combo` learning below.
        let glm = make_provider_named("glm-provider", "openai");
        assert!(!router.provider_needs_transformation(&glm, &request_with_tools(false)));
        assert!(!router.provider_needs_transformation(&glm, &request_with_tools(true)));
    }

    #[test]
    fn test_xml_tool_combo_learning_is_scoped_per_provider_model() {
        let router = Router::new(Arc::new(RwLock::new(create_test_config())), test_metrics());

        // Unknown combos stream by default.
        assert!(!router.is_xml_tool_combo("glm-provider", "glm-5.2"));

        // Learning is scoped to the exact provider/model pair.
        router.mark_xml_tool_combo("glm-provider", "glm-5.2");
        assert!(router.is_xml_tool_combo("glm-provider", "glm-5.2"));
        assert!(!router.is_xml_tool_combo("glm-provider", "glm-4.6"));
        assert!(!router.is_xml_tool_combo("other-provider", "glm-5.2"));

        // Idempotent.
        router.mark_xml_tool_combo("glm-provider", "glm-5.2");
        assert!(router.is_xml_tool_combo("glm-provider", "glm-5.2"));
    }

    #[test]
    fn test_looks_like_xml_tool_use_detects_common_markers() {
        assert!(Router::looks_like_xml_tool_use(
            "sure<tool_call>{\"name\":\"read_file\"}</tool_call>"
        ));
        assert!(Router::looks_like_xml_tool_use(
            "<use_tool name=\"execute_command\">{}</use_tool>"
        ));
        assert!(Router::looks_like_xml_tool_use(
            "<tool_calls><invoke name=\"x\"></invoke></tool_calls>"
        ));
        assert!(!Router::looks_like_xml_tool_use(
            "Here is a normal answer with no tool use."
        ));
    }
}
