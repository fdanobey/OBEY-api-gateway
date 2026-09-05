use crate::active_requests::{ActivePhase, ActiveRequestHandle};
use crate::compression::{
    caveman::apply_caveman_output,
    config::CompressionConfig,
    pipeline::{CompressionPipeline, CompressionRequestMetadata},
    precompressed::{PrecompressedLoadStatus, PrecompressedManager},
    stats::CompressionStats,
    CompressiblePayload, CompressionContext,
};
use crate::config::{
    CacheAwareRouting, Config, ContextConfig, ModelGroup, Provider, ProviderModel,
    PromptCacheSupport,
};
use crate::context::ContextManager;
use crate::dashboard::CompressionEventHub;
use crate::error::{AggregatedError, GatewayError, ProviderAttempt};
use crate::memory::{
    CompressionExtractionInput, CompressionMessageSnapshot, CompressionRemovalReport,
    ExtractionPolicy, MemorySystem, ResolvedNamespace,
};
use crate::models::openai::{Choice, Message, OpenAIRequest, OpenAIResponse, Usage};
use crate::providers::bedrock::{
    apply_global_inference_prefix, apply_global_inference_profile,
    is_duplicate_compaction_trigger_error, model_supports_reasoning, normalize_mantle_chat_messages,
    normalize_mantle_compaction_triggers, sanitize_mantle_chat_request, BedrockProvider,
};
use crate::providers::{ProviderClient, ProviderResponse};
use crate::reasoning_compat::{self, AttemptReport};
use crate::smart_routing::budget_controller::BudgetController;
use crate::smart_routing::{
    PinnedRoutingContext, RoutingPlanOutcome, RoutingPlanningError, SmartRouter, SmartRoutingInput,
};
use dashmap::DashMap;
use std::collections::HashSet;
use std::error::Error as StdError;
use std::sync::{Arc, Mutex};
use std::time::Duration;
use std::time::{SystemTime, UNIX_EPOCH};
use tokio::sync::{Notify, RwLock};
use tracing::{debug, info, warn};

const PRECOMPRESSED_CACHE_MARKER_KEY: &str = "cache_control";
const PRECOMPRESSED_CACHE_MARKER_TYPE: &str = "obey_precompressed_context";



/// Adapter that lets the Codex Search agent loop resubmit through the
/// normal dispatch pipeline (`attempt_with_retry`) for any provider.
/// Resubmissions reuse the same model rewrite, retry, and translation
/// logic as first attempts; interception itself lives at the failover
/// layer, so re-entry cannot recurse into another interception.
struct SearchResubmitter<'a> {
    router: &'a Router,
    provider_name: &'a str,
    provider_model: &'a ProviderModel,
    active: Option<ActiveRequestHandle>,
    base_attempt: usize,
}

#[async_trait::async_trait]
impl ProviderClient for SearchResubmitter<'_> {
    async fn chat_completion(
        &self,
        request: OpenAIRequest,
    ) -> Result<ProviderResponse, GatewayError> {
        let response = self
            .router
            .attempt_with_retry(
                self.provider_name,
                &request,
                self.provider_model,
                self.active.clone(),
                self.base_attempt,
            )
            .await?;
        Ok(ProviderResponse {
            response,
            provider_name: self.provider_name.to_string(),
            latency_ms: 0,
        })
    }

    async fn chat_completion_stream(
        &self,
        _request: OpenAIRequest,
    ) -> Result<
        std::pin::Pin<
            Box<dyn futures::Stream<Item = Result<crate::providers::SSEEvent, GatewayError>> + Send>,
        >,
        GatewayError,
    > {
        Err(GatewayError::Provider {
            provider: self.provider_name.to_string(),
            message: "streaming is not supported for search resubmission".to_string(),
            status_code: Some(500),
        })
    }

    async fn list_models(&self) -> Result<Vec<crate::providers::Model>, GatewayError> {
        Ok(Vec::new())
    }

    fn provider_name(&self) -> &str {
        self.provider_name
    }
}

#[derive(Debug)]
struct ProviderConcurrencyState {
    limit: usize,
    in_flight: usize,
}

#[derive(Debug)]
struct ProviderConcurrencyLimiter {
    state: Mutex<ProviderConcurrencyState>,
    notify: Notify,
}

impl ProviderConcurrencyLimiter {
    fn new(limit: u32) -> Self {
        Self {
            state: Mutex::new(ProviderConcurrencyState {
                limit: limit as usize,
                in_flight: 0,
            }),
            notify: Notify::new(),
        }
    }

    fn update_limit(&self, limit: u32) {
        let mut state = self
            .state
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner);
        let limit = limit.max(1) as usize;
        let increased = limit > state.limit;
        state.limit = limit;
        drop(state);
        if increased {
            self.notify.notify_waiters();
        }
    }

    fn try_acquire(self: &Arc<Self>) -> Option<ProviderConcurrencyPermit> {
        let mut state = self
            .state
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner);
        if state.in_flight >= state.limit {
            return None;
        }
        state.in_flight += 1;
        Some(ProviderConcurrencyPermit {
            limiter: Arc::clone(self),
        })
    }

    async fn acquire(self: &Arc<Self>) -> ProviderConcurrencyPermit {
        loop {
            let notified = self.notify.notified();
            tokio::pin!(notified);
            notified.as_mut().enable();
            {
                let mut state = self
                    .state
                    .lock()
                    .unwrap_or_else(std::sync::PoisonError::into_inner);
                if state.in_flight < state.limit {
                    state.in_flight += 1;
                    return ProviderConcurrencyPermit {
                        limiter: Arc::clone(self),
                    };
                }
            }
            notified.await;
        }
    }
}

#[derive(Debug)]
pub struct ProviderConcurrencyPermit {
    limiter: Arc<ProviderConcurrencyLimiter>,
}

impl Drop for ProviderConcurrencyPermit {
    fn drop(&mut self) {
        let mut state = self
            .limiter
            .state
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner);
        state.in_flight = state.in_flight.saturating_sub(1);
        drop(state);
        self.limiter.notify.notify_waiters();
    }
}

/// Longest a request will queue for one of `provider_cfg`'s in-flight slots.
///
/// Derived from the provider's own effective total timeout: if a slot cannot be
/// obtained within the time that provider is permitted to spend on an entire
/// request, waiting longer cannot yield a useful response. Using the configured
/// value rather than a constant keeps deliberately slow local providers (e.g.
/// Ollama with a long `total_timeout_seconds`) working, while still guaranteeing
/// the wait terminates. Floored at one second so a zero config cannot produce a
/// non-blocking spin.
fn provider_slot_wait(provider_cfg: &Provider, model: &str) -> Duration {
    Duration::from_secs(provider_cfg.effective_total_timeout(model).max(1))
}

/// 503 returned when a provider's in-flight slots stay exhausted.
fn provider_saturated_error(provider_cfg: &Provider, waited: Duration) -> GatewayError {
    GatewayError::Provider {
        provider: provider_cfg.name.clone(),
        message: format!(
            "Provider '{}' has no free in-flight slot after {}s (max_connections = {}). \
             Raise `max_connections` for this provider, lower its `total_timeout_seconds`, \
             or reduce concurrent load.",
            provider_cfg.name,
            waited.as_secs(),
            provider_cfg.max_connections
        ),
        status_code: Some(503),
    }
}

struct CompressionRuntime {
    pipeline: Arc<CompressionPipeline>,
    precompressed_manager: Option<Arc<PrecompressedManager>>,
}

use super::cache_cost::{compute_actual_cost, extract_cache_usage};
use super::cache_inject::{inject_explicit_cache_breakpoints, CacheInjectorConfig};
use super::sticky_cache::StickyCache;
use super::{CircuitBreaker, LatencyTracker, RateLimiter};

/// Computes the prompt-cache cost savings of a completed response in
/// whole cents versus its uncached baseline (Req 4.2). `actual_cost`
/// is the cache-aware dollar cost from [`compute_actual_cost`]; the
/// baseline prices every prompt token at the model's base input rate
/// and every completion token at the output rate — i.e. what the
/// request would have cost with zero cache. Can be negative when a
/// cache-creation premium exceeds the read discount.
fn cache_savings_cents(model: &ProviderModel, usage: &Usage, actual_cost: f64) -> i64 {
let baseline_cost = (usage.prompt_tokens as f64 * model.cost_per_million_input_tokens
+ usage.completion_tokens as f64 * model.cost_per_million_output_tokens)
/ 1_000_000.0;
((baseline_cost - actual_cost) * 100.0).round() as i64
}

/// Records prompt-cache token telemetry for a provider response
/// (Req 4.2): the extracted cache token split plus the savings
/// versus the uncached baseline. Keeps the success-path call sites
/// one-liners.
fn record_cache_usage(
metrics: &crate::metrics::Metrics,
provider: &str,
model: &ProviderModel,
usage: &Usage,
actual_cost: f64,
) {
metrics.add_cache_usage(
provider,
extract_cache_usage(usage),
cache_savings_cents(model, usage, actual_cost),
);
}

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
    provider_concurrency_limiters: Arc<DashMap<String, Arc<ProviderConcurrencyLimiter>>>,
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
    /// Codex Search Prometheus metrics (tool executions + latency).
    search_metrics: Arc<crate::codex::search::metrics::SearchMetrics>,
/// Adaptively learned set of `provider::model` combinations whose model
/// emits XML-style tool calls (instead of native `tool_calls`). Populated
/// at runtime when a response is detected to contain XML tool use.
/// Subsequent tool requests for a learned combo take the buffer-and-
/// translate path and receive the tool-calling hint. Entries are sticky
/// for the process lifetime: un-marking a combo mid-session would make
/// the hint appear and disappear between turns, which models flag as a
/// prompt-injection pattern. In-memory only — resets on process restart.
    xml_tool_combos: Arc<std::sync::RwLock<HashSet<String>>>,

    /// `provider::model` combos observed ending a *streamed* turn with reasoning
    /// only — no answer text and no tool call. These take the buffered path for
    /// tools-bearing requests so the degenerate-turn failover in
    /// [`Self::route_with_failover_for_group`] can actually run: during a live
    /// relay the provider's terminal chunk has already reached the client by the
    /// time the shortfall is detectable, so there is nothing left to retry.
    /// In-memory only — resets on process restart.
    degenerate_stream_combos: Arc<std::sync::RwLock<HashSet<String>>>,
    /// Prompt-cache sticky routing affinity (prefix hash → last successful
    /// provider/model, TTL-bounded). Zero-TTL when cache-aware routing is
    /// disabled, so lookups always miss (Req 1.4, prompt-cache-routing spec).
    sticky_cache: StickyCache,
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
        concurrency_permit: ProviderConcurrencyPermit,
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
/// Builds the sticky-routing cache from the effective cache-aware
/// routing config. A disabled feature (or a zero stickiness TTL)
/// produces a zero-TTL cache whose lookups always miss (Req 1.4).
/// The reasoning-compat conversation-model-affinity feature (Task 6)
/// rides the same prefix→provider entries, so it also needs a live
/// TTL; `stickiness_ttl_seconds` (default 300) is the shared knob.
fn sticky_cache_from_config(
cache_aware_routing: &CacheAwareRouting,
reasoning_compat: &reasoning_compat::ReasoningCompatConfig,
) -> StickyCache {
let affinity_enabled =
reasoning_compat.enabled && reasoning_compat.conversation_model_affinity;
if (cache_aware_routing.enabled || affinity_enabled)
&& cache_aware_routing.stickiness_ttl_seconds > 0
{
StickyCache::new(Duration::from_secs(cache_aware_routing.stickiness_ttl_seconds))
} else {
StickyCache::new(Duration::ZERO)
}
}

pub fn new(config: Arc<RwLock<Config>>, metrics: Arc<crate::metrics::Metrics>) -> Self {
let (context_config, compression_config, smart_routing_config, cache_aware_routing, reasoning_compat) = {
let cfg = config.try_read().expect("config lock");
(
cfg.context.clone(),
cfg.compression.clone(),
cfg.smart_routing.clone(),
cfg.cache_aware_routing.clone(),
cfg.reasoning_compat.clone(),
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
            provider_concurrency_limiters: Arc::new(DashMap::new()),
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
            search_metrics: Arc::new(crate::codex::search::metrics::SearchMetrics::new()),
            xml_tool_combos: Arc::new(std::sync::RwLock::new(HashSet::new())),
            degenerate_stream_combos: Arc::new(std::sync::RwLock::new(HashSet::new())),
sticky_cache: Self::sticky_cache_from_config(&cache_aware_routing, &reasoning_compat),
}
}

/// Create a new Router with explicit context configuration
#[allow(dead_code)]
pub fn with_context_config(
config: Arc<RwLock<Config>>,
context_config: ContextConfig,
metrics: Arc<crate::metrics::Metrics>,
) -> Self {
let (compression_config, smart_routing_config, cache_aware_routing, reasoning_compat) = {
let cfg = config.try_read().expect("config lock");
(
cfg.compression.clone(),
cfg.smart_routing.clone(),
cfg.cache_aware_routing.clone(),
cfg.reasoning_compat.clone(),
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
            provider_concurrency_limiters: Arc::new(DashMap::new()),
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
search_metrics: Arc::new(crate::codex::search::metrics::SearchMetrics::new()),
            xml_tool_combos: Arc::new(std::sync::RwLock::new(HashSet::new())),
            degenerate_stream_combos: Arc::new(std::sync::RwLock::new(HashSet::new())),
sticky_cache: Self::sticky_cache_from_config(&cache_aware_routing, &reasoning_compat),
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

    /// Get the Codex Search metrics handle for Prometheus exposition.
    pub fn search_metrics(&self) -> Arc<crate::codex::search::metrics::SearchMetrics> {
        self.search_metrics.clone()
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

/// Get the OAuth manager (used by Responses API Codex native dispatch).
pub fn oauth_manager(&self) -> Option<Arc<crate::oauth::OAuthManager>> {
self.oauth_manager.clone()
}

/// Get the OAuth usage tracker (used by Responses API Codex native dispatch).
pub fn oauth_usage_tracker(&self) -> Option<Arc<crate::oauth::UsageTracker>> {
self.oauth_usage_tracker.clone()
}

/// Get or create an HTTP client for a provider.
pub fn get_http_client(&self, provider_name: &str, config: &crate::config::ProviderConnectionPoolConfig) -> Result<reqwest::Client, GatewayError> {
self.get_or_create_http_client(provider_name, config)
}

/// Resolve the first candidate provider for a model group.
/// Returns `(provider_config, provider_model)` if found.
/// Used by the Responses API handler to detect Codex providers.
pub async fn first_provider_for_model(
&self,
model: &str,
) -> Result<Option<(Provider, ProviderModel)>, GatewayError> {
let model_group = self.find_model_group(model).await?;
let candidates = self.select_provider_order(&model_group).await;

let Some(first) = candidates.into_iter().next() else {
return Ok(None);
};

let config = self.config.read().await;
let provider_cfg = config
.providers
.iter()
.find(|p| p.name == first.provider)
.cloned();

Ok(provider_cfg.map(|p| (p, first)))
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
    /// 4. Within same priority, sort by cost (ascending - lower cost first).
    ///    When `cache_aware_routing.enabled` and `cost_sort_hit_rate > 0.0`,
    ///    the cost key blends cache-read and uncached input prices at that
    ///    assumed hit rate (Req 3.1); otherwise it is the uncached cost.
    /// 5. Within similar costs (±10%), sort by latency (ascending - lower latency first)
    /// 6. If version_fallback_enabled, re-sort the whole candidate list by
    ///    version date (descending - newer first). This is intentionally the
    ///    DOMINANT key, overriding priority/cost/latency: the newest dated
    ///    model is preferred even over a lower-priority (or cheaper) undated
    ///    model, and undated models keep their relative order after all
    ///    dated ones. See `test_version_fallback_sorting`.
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

        // Cache-aware cost sort key (Req 3.1): when the feature is enabled
        // and a positive hit rate is configured, the comparator weighs
        // cache-read vs uncached input prices at that assumed hit rate.
        // Hoisted out of the closure so the config lock is acquired exactly
        // once (never per comparison) and the sort stays deterministic even
        // if a hot-reload lands mid-sort. `None` keeps the uncached
        // `total_cost()` behavior (Req 3.2).
        let assumed_hit_rate = {
            let config = self.config.read().await;
            let cache_cfg = &config.cache_aware_routing;
            if cache_cfg.enabled && cache_cfg.cost_sort_hit_rate > 0.0 {
                Some(cache_cfg.cost_sort_hit_rate)
            } else {
                None
            }
        };

        // Stage 3: Sort by priority, cost, and latency
        candidates.sort_by(|a, b| {
            // First: sort by priority (ascending)
            match a.priority.cmp(&b.priority) {
                std::cmp::Ordering::Equal => {
                    // Second: sort by total cost (ascending). With an assumed
                    // cache hit rate this blends cache-read and uncached
                    // input pricing; without one it falls back to the
                    // uncached `total_cost()`.
                    let cost_a = match assumed_hit_rate {
                        Some(hit_rate) => a.total_cost_with_hit_rate(Some(hit_rate)),
                        None => a.total_cost(),
                    };
                    let cost_b = match assumed_hit_rate {
                        Some(hit_rate) => b.total_cost_with_hit_rate(Some(hit_rate)),
                        None => b.total_cost(),
                    };

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

/// Whether sticky/affinity routing is active right now: the cache-aware
/// feature flag must be enabled with a positive stickiness TTL
/// (`stickiness_ttl_seconds: 0` disables stickiness, Req 1.4), OR the
/// reasoning-compat conversation-model-affinity feature must be on
/// (Task 6) — it records and consults the same prefix→provider entries
/// for source-model attribution in strip/preserve decisions.
async fn sticky_routing_enabled(&self) -> bool {
let config = self.config.read().await;
let cache_cfg = &config.cache_aware_routing;
let reasoning_cfg = &config.reasoning_compat;
(cache_cfg.enabled
|| (reasoning_cfg.enabled && reasoning_cfg.conversation_model_affinity))
&& cache_cfg.stickiness_ttl_seconds > 0
}

    /// Promotes the sticky provider for `request`'s conversation prefix to
    /// the head of `candidates` (Req 1.2).
    ///
    /// MUST be applied to the output of [`Self::select_provider_order`] —
    /// after the circuit-breaker and cooldown gates have filtered the list —
    /// so a sticky entry for an unhealthy provider is simply not promoted
    /// and normal priority/cost/latency routing serves the request
    /// (Req 1.3). The entry is deliberately left in the cache (never
    /// deleted on a gate miss) so stickiness resumes once the breaker
    /// closes. Promotion moves the entry's provider/model to index 0 and
    /// never removes other candidates.
    async fn promote_sticky_provider(
        &self,
        request: &OpenAIRequest,
        candidates: &mut Vec<ProviderModel>,
    ) {
        if !self.sticky_routing_enabled().await {
            debug!(reason = "disabled", "sticky_skipped");
            return;
        }
        let prefix_hash = StickyCache::compute_prefix_hash(request);
        // `get` lazily evicts expired entries, so an expired affinity is
        // indistinguishable from a miss here.
        let Some(entry) = self.sticky_cache.get(prefix_hash) else {
            debug!(prefix_hash, reason = "miss", "sticky_skipped");
            return;
        };
        let Some(index) = candidates
            .iter()
            .position(|pm| pm.provider == entry.provider_id && pm.model == entry.model_id)
        else {
            debug!(
                provider = %entry.provider_id,
                prefix_hash,
                reason = "not-in-candidates",
                "sticky_skipped"
            );
            return;
        };
        if index > 0 {
            let promoted = candidates.remove(index);
            candidates.insert(0, promoted);
        }
        debug!(
            provider = %entry.provider_id,
            model = %entry.model_id,
            prefix_hash,
            last_cache_read_tokens =
                entry.last_success_usage.map(|u| u.cache_read_input_tokens),
            "sticky_promoted"
        );
    }

    /// Upserts the sticky-routing affinity entry after a successful
    /// response (Req 1.1). The prefix hash is computed from the ORIGINAL
    /// client request (pre-transform), so affinity stays stable across the
    /// provider-specific mutations applied to the outgoing copy.
    async fn record_sticky_success(
        &self,
        request: &OpenAIRequest,
        provider: &str,
        model: &str,
        usage: &Usage,
    ) {
        if !self.sticky_routing_enabled().await {
            return;
        }
let prefix_hash = StickyCache::compute_prefix_hash(request);
self.sticky_cache.insert(
prefix_hash,
provider.to_string(),
model.to_string(),
Some(extract_cache_usage(usage)),
);
debug!(provider = %provider, model = %model, prefix_hash, "sticky_recorded");
}

/// Source-model attribution for the reasoning-compat strip/preserve
/// policy (reasoning-failover-compat spec, Req 6.4): resolves the
/// conversation-prefix affinity entry to a policy
/// [`ModelRef`](reasoning_compat::policy::ModelRef) describing the
/// provider + model that last served this conversation.
///
/// Returns `None` when the affinity feature is off (zero overhead: no
/// prefix hash, no lookup) or when no fresh affinity entry exists for
/// this prefix (first turn, TTL expired, or config hot-reload cleared
/// the cache) — the policy then falls back to family matching.
/// Synchronous and lock-free: a single DashMap read.
fn model_affinity_source(
&self,
request: &OpenAIRequest,
reasoning_compat_cfg: &reasoning_compat::ReasoningCompatConfig,
) -> Option<reasoning_compat::policy::ModelRef> {
if !reasoning_compat_cfg.conversation_model_affinity {
return None;
}
let prefix_hash = StickyCache::compute_prefix_hash(request);
let (provider, model) = self.sticky_cache.get_model_affinity(prefix_hash)?;
Some(reasoning_compat::policy::ModelRef {
provider,
family: reasoning_compat::detect::classify_family(&model),
model,
})
}

    /// Applies prompt-cache decorations to an outgoing provider request:
    /// gateway-computed `cache_control` breakpoints for explicit-cache
    /// providers (Req 2.1) and a deterministic OpenRouter session id derived
    /// from the conversation prefix hash so intermediary stickiness aligns
    /// with gateway stickiness (Req 1.5). No-op unless cache-aware routing
    /// is enabled. Synchronous and lock-free: the sticky-cache lookup is a
    /// DashMap read with no await points.
    fn apply_cache_routing_decorations(
        &self,
        outgoing: &mut OpenAIRequest,
        request: &OpenAIRequest,
        provider_model: &ProviderModel,
        is_openrouter: bool,
        cache_cfg: &CacheAwareRouting,
    ) {
        if !cache_cfg.enabled {
            return;
        }
        let prefix_hash = StickyCache::compute_prefix_hash(request);
        if is_openrouter {
            // Deterministic session id from the prefix hash: same
            // conversation prefix → same OpenRouter affinity bucket.
            let session_id = format!("obey-{:016x}", prefix_hash);
            outgoing
                .extra
                .insert("session_id".to_string(), serde_json::json!(session_id));
            debug!(session_id = %session_id, prefix_hash, "openrouter_session_id_attached");
        }
        if let Some(PromptCacheSupport::Explicit { max_breakpoints }) = &provider_model.cache_support
        {
            let cache_min_tokens = provider_model
                .cache_min_tokens
                .unwrap_or(cache_cfg.default_cache_min_tokens);
            let prior_usage = self
                .sticky_cache
                .get(prefix_hash)
                .and_then(|entry| entry.last_success_usage);
            match inject_explicit_cache_breakpoints(
                outgoing,
                &CacheInjectorConfig {
                    max_breakpoints: *max_breakpoints,
                    cache_min_tokens,
                },
                prior_usage.as_ref(),
            ) {
                Ok(()) => {
                    debug!(
                        provider = %provider_model.provider,
                        model = %provider_model.model,
                        prefix_hash,
                        max_breakpoints,
                        "cache_breakpoints_injected"
                    );
                }
                Err(err) => {
                    // Injection failing its own post-condition must never
                    // fail the request — send without markers.
                    warn!(
                        provider = %provider_model.provider,
                        error = %err,
                        "cache_breakpoint_injection_failed_sending_without_markers"
                    );
                }
            }
        }
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

    /// True when a dispatch error is rate-limit-class: an HTTP 429, or an
    /// error-in-200 envelope promoted to 429, or any 200/4xx body carrying a
    /// recognizable rate-limit/quota marker.
    ///
    /// Rate-limit-class failures are governed exclusively by the dedicated
    /// upstream cooldown (`RateLimiter::apply_cooldown` plus the durable
    /// metrics cooldown store, both enforced as routing gates by
    /// `select_provider_order` / `route_with_failover_for_group`): a
    /// rate-limited provider is *paused*, not unhealthy, so these errors must
    /// not count toward the circuit-breaker failure threshold.
    fn is_rate_limit_class_error(e: &GatewayError) -> bool {
        match e {
            GatewayError::Provider {
                status_code,
                message,
                ..
            } => Self::is_rate_limited(status_code.unwrap_or(0), message),
            _ => false,
        }
    }

    /// Suspicious truncation detector (Req 6.1, 6.3, 6.4).
    ///
    /// A `finish_reason: "length"` response is only treated as a suspicious
    /// truncation when the provider actually reported token usage AND the
    /// completion stopped well short (more than 50 tokens below) of the
    /// client's requested `max_tokens`. Providers that omit `usage` default
    /// to `completion_tokens == 0`; reading that as "stopped short" would
    /// trigger false failover (duplicate token spend, doubled latency) and
    /// spurious circuit-breaker failures on legitimate length-capped
    /// responses.
    fn is_suspicious_truncation(response: &OpenAIResponse, max_tokens: Option<u32>) -> bool {
        if response
            .choices
            .first()
            .and_then(|choice| choice.finish_reason.as_deref())
            != Some("length")
        {
            return false;
        }
        let Some(max_tokens) = max_tokens else {
            return false;
        };
        let completion_tokens = response.usage.completion_tokens;
        completion_tokens > 0 && completion_tokens < max_tokens.saturating_sub(50)
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

    /// Record a successful true-streaming pass-through relay — the mirror of
    /// [`Self::record_streaming_failure`] for the success path.
    ///
    /// The buffered success path (`route_with_failover_for_group`) closes the
    /// circuit breaker, updates latency, accrues cost, and clears any
    /// upstream-driven cooldown. Pass-through streams finish inside the
    /// handler's relay, so without this hook a streaming-only provider would
    /// never reset its breaker (one isolated failure after recovery would
    /// re-open it at escalated backoff), never feed the latency tracker that
    /// backs cost-band tiebreaks, and never clear a stale rate-limit cooldown.
    ///
    /// The handler passes the relay's reassembled `usage` (zeros when the
    /// provider omitted usage frames); cost is accrued from the configured
    /// per-model rates only when token usage is actually known, mirroring the
    /// buffered path's `usage_known` handling.
    ///
    /// `request` is the original client request; it is used to upsert the
    /// cache-aware sticky-routing affinity entry (Req 1.1) so the next turn
    /// of the same conversation prefix prefers this provider.
    pub async fn record_streaming_success(
        &self,
        request: &OpenAIRequest,
        provider: &str,
        model: &str,
        duration: std::time::Duration,
        usage: &Usage,
    ) {
        let cb_key = format!("{}:{}", provider, model);
        let cb = self.get_circuit_breaker(&cb_key).await;
        cb.record_success().await;

        let duration_ms = duration.as_millis() as u64;
        self.latency_tracker.update_latency(provider, duration);
        self.metrics.record_provider_success(provider, duration_ms);

        // Provider recovered — clear the upstream-driven cooldown in both
        // stores so it stops being filtered out (mirrors the buffered
        // success path).
        let rate_limiter = self.get_rate_limiter(provider).await;
        rate_limiter.clear_cooldown().await;
        self.metrics.clear_provider_cooldown(provider);

        let usage_known =
            usage.total_tokens > 0 || usage.prompt_tokens > 0 || usage.completion_tokens > 0;
        if !usage_known {
            self.metrics.record_provider_unknown_cost(provider);
            return;
        }
        // Cache-aware actual cost (Req 3.5): price the reassembled usage
        // split (uncached / cache-read / cache-creation) at the model's
        // configured per-million rates. Falls back to base-price math,
        // bit-identical to the previous formula, when the usage carries
        // no cache fields.
        {
            let config = self.config.read().await;
            let model_entry = config
                .model_groups
                .iter()
                .find_map(|group| {
                    group
                        .models
                        .iter()
                        .find(|m| m.provider == provider && m.model == model)
                });
        match model_entry {
            Some(model_entry) => {
                let cost = compute_actual_cost(model_entry, usage);
                self.metrics.add_cost(provider, cost);
                record_cache_usage(&self.metrics, provider, model_entry, usage, cost);
                // Reasoning-token attribution (Req 4.7) for pass-through
                // streams: the relay's reassembled usage carries the
                // provider's reasoning-token field; price it at the
                // dedicated (or output-fallback) rate when attribution is
                // enabled. No response object exists here, so there are no
                // gateway_* extras to attach — metrics only.
                let reasoning_usage =
                    reasoning_compat::cost::extract_reasoning_usage(usage);
                if reasoning_usage.reasoning_tokens > 0
                    && config.reasoning_compat.attribute_reasoning_cost
                {
                    let reasoning_cost = reasoning_compat::cost::reasoning_cost(
                        model_entry,
                        reasoning_usage.reasoning_tokens,
                    );
                    self.metrics.add_reasoning_usage(
                        provider,
                        u64::from(reasoning_usage.reasoning_tokens),
                        reasoning_cost,
                    );
                }
            }
            None => self.metrics.record_provider_unknown_cost(provider),
        }
        }

        // Cache-aware sticky routing (Req 1.1): upsert the prefix→provider
        // affinity from the original client request so the next turn of
        // this conversation prefers this provider.
        self.record_sticky_success(request, provider, model, usage)
            .await;
    }

/// Detect and strip image content parts from messages when the target
/// model does not support vision inputs.
///
/// OpenAI-style messages can carry `content` as an array of parts
/// including `{ "type": "image_url", "image_url": { ... } }`. Many
/// providers respond with HTTP 400 if such parts reach a non-vision
/// model. This method removes those parts and logs that fact.
///
/// Recognizes the common image part type spellings across client
/// libraries (`image_url`, `image`, `input_image`) at every nesting
/// depth — including image parts inside a `tool_result` part's own
/// `content` array, which clients send when a tool returned an image.
/// When stripping empties a content array entirely (top-level or
/// nested), a short text placeholder is inserted so the provider never
/// sees an empty content array.
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
            let removed = Self::strip_image_parts_recursive(parts, idx, provider_name, model);
            if removed > 0 && parts.is_empty() {
                parts.push(serde_json::json!({
                    "type": "text",
                    "text": "[image content removed: model does not support image inputs]"
                }));
            }
            stripped_total += removed;
        }
    }
    stripped_total
}

/// Recursively remove image content parts from a `content` array and
/// from nested `content` arrays inside surviving parts (e.g.
/// `{"type":"tool_result","content":[{"type":"image_url",...}]}`).
/// Returns the total number of image parts removed at every depth.
fn strip_image_parts_recursive(
    parts: &mut Vec<serde_json::Value>,
    message_index: usize,
    provider_name: &str,
    model: &str,
) -> usize {
    let mut stripped: usize = 0;

    // First recurse into nested `content` arrays so images buried inside
    // non-image parts (tool results, custom part shapes) are removed too.
    for part in parts.iter_mut() {
        if let Some(nested_value) = part.get_mut("content") {
            if let serde_json::Value::Array(nested) = nested_value {
                let nested_removed =
                    Self::strip_image_parts_recursive(nested, message_index, provider_name, model);
                if nested_removed > 0 && nested.is_empty() {
                    nested.push(serde_json::json!({
                        "type": "text",
                        "text": "[image content removed: model does not support image inputs]"
                    }));
                }
                stripped += nested_removed;
            }
        }
    }

    // Then remove image parts at this level.
    let before = parts.len();
    parts.retain(|part| {
        !matches!(
            part.get("type").and_then(|v| v.as_str()),
            Some("image_url") | Some("image") | Some("input_image")
        )
    });
    let removed = before.saturating_sub(parts.len());
    if removed > 0 {
        warn!(
            provider = provider_name,
            model = %model,
            message_index = message_index,
            images_removed = removed,
            "Stripped image content parts from message for non-vision model"
        );
    }
    stripped + removed
}

/// Check whether an upstream rejection is caused by image content the
/// model cannot accept, regardless of what the capabilities cache
/// believed.
///
/// Providers phrase this differently ("This model does not support
/// image inputs", "invalid content type: image", "does not support
/// vision", …) and always with a 4xx status. A case-insensitive
/// substring check over the body keeps the detection tolerant of the
/// varied error envelopes.
fn is_unsupported_image_error(status_code: u16, body: &str) -> bool {
    (400..500).contains(&status_code) && Self::is_unsupported_image_phrasing(body)
}

/// Phrase-level detection of "model cannot accept image inputs" error
/// text, independent of the HTTP status. Used for real 4xx rejections
/// and for the error-inside-HTTP-200 envelopes some providers return.
fn is_unsupported_image_phrasing(body: &str) -> bool {
    let lower = body.to_ascii_lowercase();
    lower.contains("does not support image")
        || lower.contains("not support image inputs")
        || lower.contains("unsupported image")
        || lower.contains("image inputs")
        || lower.contains("image content")
        || lower.contains("input_image")
        || lower.contains("does not support vision")
        || lower.contains("vision is not supported")
        || lower.contains("not a vision model")
        || lower.contains("only supports text")
    }

    /// Check whether an upstream 400 is an Anthropic-style thinking/budget
    /// validation failure (reasoning-failover-compat spec, Req 6.2).
    ///
    /// The provider error body mentions `thinking` or `budget_tokens` when
    /// the request's extended-thinking state or parameters failed
    /// validation (e.g. `thinking.budget_tokens` >= `max_tokens`, adaptive
    /// models rejecting `type: "enabled"`, or signature mismatches on
    /// replayed thinking blocks). Matched on the body text, never on the
    /// bare status, so unrelated 400s keep their normal classification.
    fn is_thinking_validation_error(status_code: u16, body: &str) -> bool {
        if status_code != 400 {
            return false;
        }
        let lower = body.to_ascii_lowercase();
        lower.contains("thinking") || lower.contains("budget_tokens")
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

    /// Reset the circuit breaker for a specific provider key.
    /// Returns `true` if the breaker existed and was removed.
    pub fn reset_circuit_breaker(&self, provider_key: &str) -> bool {
        self.circuit_breakers.remove(provider_key).is_some()
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

    /// Clear all sticky-routing affinity entries (used during config
    /// reload, mirroring the circuit-breaker / rate-limiter resets).
    pub fn clear_sticky_cache(&self) {
        self.sticky_cache.clear();
    }

    /// Sweeps expired prefix-affinity entries (maintenance complement to
    /// the lazy TTL eviction in `StickyCache::get`; called periodically
    /// by the gateway background task so dead conversations don't
    /// accumulate).
    pub fn evict_expired_sticky_entries(&self) {
        self.sticky_cache.evict_expired();
    }

    fn provider_concurrency_limiter(
        &self,
        provider_name: &str,
        limit: u32,
    ) -> Arc<ProviderConcurrencyLimiter> {
        use dashmap::mapref::entry::Entry;

        let limiter = match self
            .provider_concurrency_limiters
            .entry(provider_name.to_string())
        {
            Entry::Occupied(entry) => Arc::clone(entry.get()),
            Entry::Vacant(entry) => {
                let limiter = Arc::new(ProviderConcurrencyLimiter::new(limit.max(1)));
                entry.insert(Arc::clone(&limiter));
                limiter
            }
        };
        limiter.update_limit(limit);
        limiter
    }

    /// Acquire one provider-wide in-flight request slot, giving up after `wait`.
    ///
    /// The permit is released automatically when it is dropped, including
    /// cancellation and error paths. `None` means the provider stayed saturated
    /// for the whole window; callers must surface that as a 503 rather than
    /// continuing to wait.
    ///
    /// The bound matters because a permit is held for the entire upstream call —
    /// including the full duration of a streaming relay. With a small
    /// `max_connections` and a large `total_timeout_seconds`, an unbounded wait
    /// let requests queue behind saturated slots indefinitely with no response
    /// and no diagnostic, which is indistinguishable from a gateway hang.
    async fn acquire_provider_concurrency_within(
        &self,
        provider_name: &str,
        limit: u32,
        wait: Duration,
    ) -> Option<ProviderConcurrencyPermit> {
        let limiter = self.provider_concurrency_limiter(provider_name, limit);
        // `timeout` only drops the pending `acquire` future; a permit produced by
        // the final poll is returned as `Ok`, so no permit can be leaked here.
        tokio::time::timeout(wait, limiter.acquire()).await.ok()
    }

    fn try_acquire_provider_concurrency(
        &self,
        provider_name: &str,
        limit: u32,
    ) -> Option<ProviderConcurrencyPermit> {
        self.provider_concurrency_limiter(provider_name, limit)
            .try_acquire()
    }

    /// Unbounded-in-practice acquire used by the limiter mechanics tests.
    ///
    /// Production code must go through [`Self::acquire_provider_slot_or_reject`]
    /// so saturation surfaces as a 503 instead of an open-ended wait.
    #[cfg(test)]
    async fn acquire_provider_concurrency(
        &self,
        provider_name: &str,
        limit: u32,
    ) -> ProviderConcurrencyPermit {
        self.acquire_provider_concurrency_within(provider_name, limit, Duration::from_secs(30))
            .await
            .expect("test slot acquisition must not time out")
    }

    /// Acquire a slot or fail with a 503 describing the saturated provider.
    async fn acquire_provider_slot_or_reject(
        &self,
        provider_cfg: &Provider,
        model: &str,
    ) -> Result<ProviderConcurrencyPermit, GatewayError> {
        let wait = provider_slot_wait(provider_cfg, model);
        match self
            .acquire_provider_concurrency_within(
                &provider_cfg.name,
                provider_cfg.max_connections,
                wait,
            )
            .await
        {
            Some(permit) => Ok(permit),
            None => {
                warn!(
                    provider = %provider_cfg.name,
                    model = %model,
                    max_connections = provider_cfg.max_connections,
                    waited_seconds = wait.as_secs(),
                    "Provider in-flight slot unavailable within the wait window; rejecting"
                );
                Err(provider_saturated_error(provider_cfg, wait))
            }
        }
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
            Some(code) if (200..300).contains(&code) => {
                // A 2xx status here means the provider returned an error
                // payload (or an unparseable body) inside a success
                // envelope. The caller-supplied text carries the detail —
                // "Error in 200 response: <provider message>" from the
                // error-in-200 detection, or "Failed to parse response:
                // <reason>" from SSE/JSON parse failures. Surface that
                // detail so the dashboard shows the real cause instead of
                // a confusing "unexpected response (HTTP 200)".
                const ERR_IN_200: &str = "Error in 200 response: ";
                const PARSE_FAIL: &str = "Failed to parse response: ";
                let trimmed = body_or_message.trim();
                if let Some(detail) = trimmed.strip_prefix(ERR_IN_200) {
                    format!(
                        "Provider error (in HTTP {}): {}",
                        code,
                        Self::truncate_for_display(detail)
                    )
                } else if let Some(detail) = trimmed.strip_prefix(PARSE_FAIL) {
                    format!(
                        "Provider sent an unparseable response (HTTP {}): {}",
                        code,
                        Self::truncate_for_display(detail)
                    )
                } else if !snippet.is_empty() {
                    format!("Provider error (in HTTP {}): {}", code, snippet)
                } else {
                    format!("Provider returned an error inside a HTTP {} response", code)
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
        // One-shot reactive image strip: the Codex and Bedrock dispatch
        // paths delegate through this wrapper and never run the proactive
        // strip pass in `dispatch_attempts_under_permit`. When the provider
        // rejects with an "image inputs not supported" error (stale or
        // absent capabilities cache), strip the image parts and retry the
        // same provider once — mirroring the standard buffered retry loop.
        let mut image_strip_retry_done = false;
        // One-shot backstop for the Bedrock Mantle single-trigger invariant: the
        // `normalize_mantle_compaction_triggers` seam already de-dupes on every
        // Mantle dispatch, but if a future client shape evades the seam and the
        // provider rejects with "only one 'compaction_trigger' ...", re-run the
        // seam and retry the same provider once. Guarded so it fires at most once
        // and only for this specific rejection (clause 3.9).
        let mut duplicate_trigger_retry_done = false;
        loop {
            match client.chat_completion(request.clone()).await {
                Ok(response) => return Ok(response),
                Err(GatewayError::Provider {
                    provider,
                    message,
                    status_code: Some(status_code),
                }) if !image_strip_retry_done
                    && Self::is_unsupported_image_error(status_code, &message) =>
                {
                    let model = request.model.clone();
                    let removed = Self::strip_image_content_if_unsupported(
                        &mut request,
                        false,
                        &provider,
                        &model,
                    );
                    if removed > 0 {
                        image_strip_retry_done = true;
                        info!(
                            provider = %provider,
                            model = %model,
                            status = status_code,
                            images_removed = removed,
                            "Provider rejected image inputs — stripped images and retrying same provider"
                        );
                        continue;
                    }
                    return Err(GatewayError::Provider {
                        provider,
                        message,
                        status_code: Some(status_code),
                    });
                }
                Err(GatewayError::Provider {
                    provider,
                    message,
                    status_code: Some(status_code),
                }) if !duplicate_trigger_retry_done
                    && is_duplicate_compaction_trigger_error(status_code, &message) =>
                {
                    // Backstop for the Bedrock Mantle single-trigger invariant.
                    // Re-run the shape-complete de-dup seam; retry the same
                    // provider once only if it actually removed a trigger.
                    let normalization = normalize_mantle_compaction_triggers(&mut request);
                    if normalization.removed > 0 {
                        duplicate_trigger_retry_done = true;
                        info!(
                            provider = %provider,
                            model = %request.model,
                            status = status_code,
                            triggers_removed = normalization.removed,
                            "Provider rejected duplicate compaction_trigger — normalized and retrying same provider"
                        );
                        continue;
                    }
                    // Nothing to repair: surface the original error unchanged so
                    // outer failover behaves exactly as today (clause 3.9).
                    return Err(GatewayError::Provider {
                        provider,
                        message,
                        status_code: Some(status_code),
                    });
                }
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
        self.attempt_with_retry_with_permit(
            provider_name,
            request,
            provider_model,
            active,
            base_attempt,
            None,
        )
        .await
    }

    async fn attempt_with_retry_with_permit(
        &self,
        provider_name: &str,
        request: &OpenAIRequest,
        provider_model: &ProviderModel,
        active: Option<ActiveRequestHandle>,
        base_attempt: usize,
        concurrency_permit: Option<ProviderConcurrencyPermit>,
    ) -> Result<OpenAIResponse, GatewayError> {
        let provider_cfg = {
            let config = self.config.read().await;
            config
                .providers
                .iter()
                .find(|p| p.name == provider_name)
                .cloned()
                .ok_or_else(|| {
                    GatewayError::Configuration(format!(
                        "Provider '{}' not found in config",
                        provider_name
                    ))
                })?
        };
        let _concurrency_permit = match concurrency_permit {
            Some(permit) => permit,
            None => {
                self.acquire_provider_slot_or_reject(&provider_cfg, &provider_model.model)
                    .await?
            }
        };
        self.dispatch_attempts_under_permit(
            provider_name,
            request,
            provider_model,
            active,
            base_attempt,
            provider_cfg,
        )
        .await
    }

    /// Returns the effective Codex Search configuration when the feature is
    /// enabled and a valid OpenAI OAuth token exists, `None` otherwise.
    ///
    /// Unlike the Codex dispatch path, this check is independent of the
    /// serving provider: any model group can invoke gateway search while a
    /// valid token is available. The token validity check awaits token
    /// refresh, so callers must not hold the config lock across it.
    async fn codex_search_ready(&self) -> Option<crate::codex::search::CodexSearchConfig> {
        let cfg = self
            .config
            .read()
            .await
            .codex_search
            .clone()
            .unwrap_or_default();
        let token_available = match &self.oauth_manager {
            Some(manager) => manager.get_access_token().await.is_some(),
            None => false,
        };
        if cfg.effective_enabled(token_available) {
            Some(cfg)
        } else {
            None
        }
    }

    /// Inject `codex_search`/`codex_web` tool definitions into a prepared
    /// request when the search feature is active. Runs after tool
    /// compression so the injected definitions are never compressed away.
    async fn maybe_inject_search_tools(&self, request: &mut OpenAIRequest) {
        if self.codex_search_ready().await.is_some() {
            crate::codex::search::injector::ToolInjector::inject(request, true, true);
        }
    }

    /// Run the Codex Search agent loop when a buffered provider response
    /// contains gateway-injected search tool calls. No-ops for Codex
    /// providers (already intercepted during Codex dispatch) and for
    /// responses without search tool calls.
    async fn maybe_intercept_search_tools(
        &self,
        provider_model: &ProviderModel,
        request: &OpenAIRequest,
        mut response: OpenAIResponse,
        active: Option<ActiveRequestHandle>,
        base_attempt: usize,
    ) -> Result<OpenAIResponse, GatewayError> {
        // XML-prone models (GLM, Kimi, Qwen, DeepSeek) emit tool calls as
        // XML text rather than native tool_calls. Translate first so
        // gateway search calls become detectable; otherwise they slip past
        // this hook and leak to the client (which doesn't know codex_search).
        Self::translate_xml_tool_calls(&mut response, request);

        let has_gateway_call = response
            .choices
            .first()
            .and_then(|c| c.message.extra.get("tool_calls"))
            .and_then(|v| v.as_array())
            .map(|calls| {
                calls.iter().any(|tc| {
                    tc.get("function")
                        .and_then(|f| f.get("name"))
                        .and_then(|n| n.as_str())
                        .map(|n| n == "codex_search" || n == "codex_web")
                        .unwrap_or(false)
                })
            })
            .unwrap_or(false);
        if !has_gateway_call {
            return Ok(response);
        }

        // Past this point the response carries a gateway-injected search call the
        // client never declared, so EVERY exit path must strip it. A client that
        // receives `codex_search` cannot execute it and cannot produce the
        // matching `role: tool` reply, so its agent loop deadlocks with no error
        // to show — the request just stops.
        //
        // These two bail-outs are reachable in normal operation because readiness
        // is evaluated independently at injection time and again here: an OAuth
        // token that expires or starts refreshing during the provider round trip,
        // or an `/admin/config/reload` that disables the feature mid-request,
        // lands exactly here holding a call that was legitimately injected.
        let Some(search_cfg) = self.codex_search_ready().await else {
            warn!(
                provider = %provider_model.provider,
                "Codex search became unavailable between tool injection and interception; stripping the gateway tool call the client cannot execute"
            );
            Self::discard_unexecutable_search_call(&mut response);
            return Ok(response);
        };
        let Some(oauth) = self.oauth_manager.clone() else {
            warn!(
                provider = %provider_model.provider,
                "No OAuth manager attached to execute the gateway search call; stripping it rather than leaking it to the client"
            );
            Self::discard_unexecutable_search_call(&mut response);
            return Ok(response);
        };
        let (pool_config, budget) = {
            let config = self.config.read().await;
            let pool = config
                .providers
                .iter()
                .find(|p| p.name == provider_model.provider)
                .map(|p| p.connection_pool.clone());
            // The search loop is a single request as far as the client is
            // concerned, so it gets the same ceiling the gateway applies to any
            // one request. It needs its own copy because on streaming requests
            // the global deadline middleware does not cover this work at all —
            // that middleware only bounds producing response headers, and the SSE
            // handler returns its body handle before the router is ever called.
            let budget = Duration::from_secs(
                crate::runtime_limits::effective_request_timeout_seconds(&config),
            );
            (pool, budget)
        };
        let http = match pool_config {
            Some(pool) => self.get_or_create_http_client(&provider_model.provider, &pool)?,
            None => {
                let default_pool = crate::config::ProviderConnectionPoolConfig::default();
                self.get_or_create_http_client(&provider_model.provider, &default_pool)?
            }
        };
        let usage_tracker = self
            .oauth_usage_tracker
            .clone()
            .unwrap_or_else(|| Arc::new(crate::oauth::UsageTracker::new()));
        let executor = Arc::new(crate::codex::search::executor::SearchExecutor::new(
            http,
            oauth,
            usage_tracker,
            self.search_metrics.clone(),
            search_cfg.effective_base_url(),
            search_cfg.effective_timeout(),
        ));
        let interceptor = crate::codex::search::interceptor::ToolInterceptor::new(
            executor,
            search_cfg.effective_max_iterations(),
            search_cfg.effective_output_to_chat(),
            budget,
        );
        let resubmitter = SearchResubmitter {
            router: self,
            provider_name: provider_model.provider.as_str(),
            provider_model,
            active,
            base_attempt,
        };
        let intercepted = interceptor
            .intercept(&resubmitter, request.clone(), response)
            .await?;
        let mut final_response = intercepted.response;

        // Safety net. The interceptor strips its own tool calls on the way out,
        // which can empty the `tool_calls` array and leave a turn with no content
        // either. `response_has_content` already ran *before* interception, so
        // nothing downstream catches it, and the SSE synthesizer renders such a
        // turn as a bare `finish_reason` chunk — which a harness reads as a
        // finished, empty turn and stops on.
        if Self::backfill_empty_turn(
            &mut final_response,
            "The gateway ran your web search, but this turn produced no answer text and no tool call. \
             Continue the task with your own tools, or say what information you still need.",
        ) {
            warn!(
                provider = %provider_model.provider,
                model = %provider_model.model,
                iteration_limit_reached = intercepted.iteration_limit_reached,
                "Codex search interception left an empty assistant turn; backfilled an explanatory message so the client can continue"
            );
        }
        Ok(final_response)
    }

    async fn dispatch_attempts_under_permit(
        &self,
        provider_name: &str,
        request: &OpenAIRequest,
        provider_model: &ProviderModel,
        active: Option<ActiveRequestHandle>,
        base_attempt: usize,
        provider_cfg: Provider,
    ) -> Result<OpenAIResponse, GatewayError> {
        let config = self.config.read().await;
        let max_retries = config.retry.max_retries_per_provider;
        let backoff_sequence = config.retry.backoff_sequence_seconds.clone();

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
                oauth.clone(),
                instructions,
                http.clone(),
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

            let codex_search_config = config.codex_search.clone().unwrap_or_default();
            // Wall-clock ceiling for the search agent loop — the same one the
            // failover-layer interception uses. Read here while the config guard
            // is still held, since the loop itself runs after it is dropped.
            let codex_search_budget = Duration::from_secs(
                crate::runtime_limits::effective_request_timeout_seconds(&config),
            );
            // Drop config lock before making HTTP calls. The dispatch below
            // awaits the full upstream round-trip; holding the write-preferring
            // read guard across it lets any queued config writer (hot-reload,
            // tray, memory settings) stall every subsequent request gateway-wide
            // for the duration of the slowest in-flight Codex call. All config
            // values needed here were cloned above.
            drop(config);

            let mut codex_request = request.clone();
            // Rewrite the model from the group name to the actual provider model ID
            codex_request.model = provider_model.model.clone();

            let search_enabled = codex_search_config.effective_enabled(true);
            let oauth_active = oauth.get_access_token().await.is_some();
            crate::codex::search::injector::ToolInjector::inject(
                &mut codex_request,
                oauth_active,
                search_enabled,
            );

            let result = self
                .dispatch_buffered_with_context_retry(&codex_client, codex_request.clone())
                .await?;

            if search_enabled && oauth_active {
                let usage_tracker = self
                    .oauth_usage_tracker
                    .clone()
                    .unwrap_or_else(|| Arc::new(crate::oauth::UsageTracker::new()));
                let executor = Arc::new(crate::codex::search::executor::SearchExecutor::new(
                    http,
                    oauth,
                    usage_tracker,
                    self.search_metrics.clone(),
                    codex_search_config.effective_base_url(),
                    codex_search_config.effective_timeout(),
                ));
let interceptor = crate::codex::search::interceptor::ToolInterceptor::new(
executor,
codex_search_config.effective_max_iterations(),
codex_search_config.effective_output_to_chat(),
codex_search_budget,
);
                let intercepted = interceptor
                    .intercept(&codex_client, codex_request, result.response)
                    .await?;
                let mut final_response = intercepted.response;
                // Stripping the gateway's own tool calls can leave a turn with
                // neither content nor tool calls, which clients read as a
                // finished, empty reply and stop on.
                Self::backfill_empty_turn(
                    &mut final_response,
                    "The gateway ran your web search, but this turn produced no answer text and no tool call. \
                     Continue the task with your own tools, or say what information you still need.",
                );
                return Ok(final_response);
            }

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
let reasoning_compat_cfg = config.reasoning_compat.clone();
        let provider_region = provider_cfg.region.clone();
        let is_oauth_provider = provider_cfg.auth_method.as_deref() == Some("oauth");
        let jitter_enabled = config.retry.jitter_enabled;
        let jitter_ratio = config.retry.jitter_ratio;
        let cache_aware_routing_cfg = config.cache_aware_routing.clone();
        let configured_base_url = provider_cfg.base_url.clone();

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

        // Reasoning-compat per-attempt stage (reasoning-failover-compat
        // spec, Req 6.1/6.2): detect reasoning carriers on the ORIGINAL
        // request (they live in prior assistant turns, pre-transform),
        // strip/preserve per source→target transition, and normalize the
        // client's reasoning parameter into the target family's accepted
        // shape. Runs inside the per-provider dispatch so every failover
        // attempt is transformed against its own target. `enabled: false`
        // skips everything (exact passthrough, Bedrock legacy block below
        // unchanged) and only emits a debug note when carriers were seen.
let mut reasoning_report: Option<AttemptReport> = None;
if reasoning_compat_cfg.enabled {
// Source-model attribution (Task 6): the provider + model that
// last served this conversation prefix, from the sticky cache.
// None on a miss — the policy then falls back to family matching.
let source_ref = self.model_affinity_source(request, &reasoning_compat_cfg);
let report = reasoning_compat::prepare_attempt(
&mut outgoing,
request,
source_ref,
provider_model,
&reasoning_compat_cfg,
);
            if report.strip.messages_touched > 0 || report.strip.thinking_blocks > 0 {
                reasoning_compat::policy::log_strip_action(
                    &report.strip,
                    report.decision,
                    &crate::router::trace_id::generate_trace_id(None),
                );
            }
            reasoning_report = Some(report);
        } else {
            let footprint = reasoning_compat::detect::detect(&request.messages);
            if !footprint.is_empty() {
                debug!(
                    provider = provider_name,
                    model = %provider_model.model,
                    "Reasoning carriers detected in request but reasoning_compat is disabled; forwarding unmodified"
                );
            }
        }

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

        // Inject reasoning/extended thinking parameter for Bedrock providers.
        // Legacy passthrough block: when reasoning_compat is enabled, the
        // normalization stage above already emitted the correct parameter
        // shape (honoring the provider `reasoning` flag via the target's
        // family/shape resolution), so the hardcoded budget_tokens: 4096
        // injection must not run.
        if provider_type == "bedrock" && reasoning && !reasoning_compat_cfg.enabled {
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

// Conditionally inject the tool-calling guide.
//
// Goal: make native OpenAI-style tool use clear for models that were
// primarily trained on XML/pseudo-XML agent formats — without taxing
// every capable model with uncached prompt tokens (and guidance that
// conflicts with parallel tool calling).
//
// Injection targets **learned XML combos only** — models actually
// observed emitting XML-style tool use — so models that call tools
// natively never see unexplained instructions. Learned combos keep the
// hint for the process lifetime; toggling it mid-session is the context
// mutation that models flag as prompt injection. The hint is inserted
// directly after the client's system prompt (see
// [`Self::insert_tool_calling_hint`]) — not appended at the tail, where
// a system message after user/tool content reads as an injection
// attempt.
        let inject_tool_hint =
            has_tools && self.should_inject_tool_hint(provider_name, &provider_model.model);
        if inject_tool_hint {
            debug!(
                provider = provider_name,
                model = %provider_model.model,
                "Injecting tool-calling system hint (learned XML combo)"
            );
            Self::insert_tool_calling_hint(&mut outgoing.messages);
        }

        // Prompt-cache decorations (Req 2.1 / 1.5): explicit-cache
        // providers get gateway-computed `cache_control` breakpoints;
        // OpenRouter gets a deterministic session id aligned with the
        // gateway's prefix-hash stickiness. Applied after all message
        // mutations so the marker positions reflect the final wire body.
        let is_openrouter = provider_type.eq_ignore_ascii_case("openrouter")
            || configured_base_url
                .as_deref()
                .map_or(false, |u| u.contains("openrouter.ai"));
        self.apply_cache_routing_decorations(
            &mut outgoing,
            request,
            provider_model,
            is_openrouter,
            &cache_aware_routing_cfg,
        );

let mut last_error = None;
// One-shot reactive image strip: when the provider rejects the request
// with an "image inputs not supported" error despite the proactive
// strip pass (stale/incorrect capabilities), remove the images and
// retry the same provider once without waiting through backoff.
        let mut image_strip_retry_done = false;
        // One-shot reasoning-compat 400 recovery (see the 4xx branch
        // below): an Anthropic-style thinking/budget_tokens validation 400
        // triggers one aggressive strip + retry of the same provider.
        let mut reasoning_strip_retry_done = false;
        let mut skip_next_backoff = false;

for attempt in 0..=max_retries {
    let skip_backoff_this_attempt = skip_next_backoff;
    skip_next_backoff = false;
    if attempt > 0 && !skip_backoff_this_attempt {
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

			// Image-input rejection inside a 200 envelope: same rescue
			// as the 4xx branch below — strip image parts (including
			// nested ones) and retry the same provider once instead of
			// failing over with images still attached.
			if !image_strip_retry_done
				&& Self::is_unsupported_image_phrasing(&body_text)
			{
				let removed = Self::strip_image_content_if_unsupported(
					&mut outgoing,
					false,
					provider_name,
					&provider_model.model,
				);
				if removed > 0 {
					image_strip_retry_done = true;
					skip_next_backoff = true;
					info!(
						provider = provider_name,
						model = %provider_model.model,
						images_removed = removed,
						"Provider rejected image inputs (in HTTP 200 envelope) — stripped images and retrying same provider"
					);
					continue;
				}
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
                        if let Ok(mut openai_response) =
                            serde_json::from_str::<OpenAIResponse>(&body_text)
                        {
// Diagnostic: detect whether the model used native tool_calls
// or fell back to XML-style tool use in plain text content.
// This is the buffered-path feed of the same adaptive signal the
// streaming relay uses: XML output marks the combo (hint +
// buffer-and-translate). Native tool_calls need no action —
// learned combos are intentionally sticky so the injected hint
// never appears/disappears between turns of a conversation.
if let Some(choice) = openai_response.choices.first() {
    let has_native_tc = choice.message.extra.contains_key("tool_calls");
    let content_text = choice.message.content_as_text();
    // XML tool use can land in a reasoning carrier rather than `content`
    // (GLM emits `reasoning_content`), so scan both or the combo is never
    // learned and every turn for it keeps losing the tool call.
    let reasoning_text = Self::reasoning_text(choice).unwrap_or_default();
    let has_xml_tool_use = Self::looks_like_xml_tool_use(&content_text)
        || Self::looks_like_xml_tool_use(reasoning_text);
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
    if has_tools && has_xml_tool_use {
        self.mark_xml_tool_combo(provider_name, &provider_model.model);
    }
}
                            if has_tools {
            openai_response.extra.insert(
                "gateway_tool_hint_injected".to_string(),
                serde_json::json!(inject_tool_hint),
            );
        }
        // Reasoning-compat log telemetry (Req 4.6): attach the compat
        // stage's actions (counts/families only) where the report exists;
        // the failover success path and the handler pass it through and
        // strip it before the client sees the response.
        if let Some(actions) = reasoning_report.and_then(AttemptReport::actions_json) {
            openai_response.extra.insert(
                "gateway_reasoning_compat_actions".to_string(),
                serde_json::Value::String(actions),
            );
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
                Ok(mut response) => {
                    if has_tools {
                        response.extra.insert(
                            "gateway_tool_hint_injected".to_string(),
                            serde_json::json!(inject_tool_hint),
                        );
                    }
                    if let Some(actions) = reasoning_report.and_then(AttemptReport::actions_json) {
                        response.extra.insert(
                            "gateway_reasoning_compat_actions".to_string(),
                            serde_json::Value::String(actions),
                        );
                    }
                    return Ok(response);
                }
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
    // Image-input rejection: the provider refused image content even
    // though the proactive strip pass believed it safe (stale or
    // incorrect capabilities cache). Strip every image part and retry
    // the same provider immediately — bounded to one shot per request.
    if !image_strip_retry_done && Self::is_unsupported_image_error(status_code, &body_text) {
        let removed = Self::strip_image_content_if_unsupported(
            &mut outgoing,
            false,
            provider_name,
            &provider_model.model,
        );
                    if removed > 0 {
                        image_strip_retry_done = true;
                        skip_next_backoff = true;
                        info!(
                            provider = provider_name,
                            model = %provider_model.model,
                            status = status_code,
                            images_removed = removed,
                            "Provider rejected image inputs — stripped images and retrying same provider"
                        );
                        last_error = Some(err);
                        continue;
                    }
                }

                // Reasoning-compat 400 recovery (reasoning-failover-compat
                // spec, Req 6.2): an Anthropic-style thinking/budget_tokens
                // validation 400 means the request carried reasoning state
                // or params the target rejected. Classify as non-retryable
                // in-provider (fail over) with a `reasoning_compat` tagged
                // diagnostic, but first one-shot an aggressive strip of
                // every reasoning carrier and retry the same provider
                // without backoff.
                if reasoning_compat_cfg.enabled
                    && !reasoning_strip_retry_done
                    && Self::is_thinking_validation_error(status_code, &body_text)
                {
                    let strip_report = reasoning_compat::policy::apply(
                        &mut outgoing,
                        reasoning_compat::policy::StripDecision::StripAll,
                    );
                    if strip_report.messages_touched > 0 {
                        reasoning_strip_retry_done = true;
                        skip_next_backoff = true;
                        info!(
                            provider = provider_name,
                            model = %provider_model.model,
                            status = status_code,
                            thinking_blocks = strip_report.thinking_blocks,
                            redacted_thinking_blocks = strip_report.redacted_thinking_blocks,
                            fields_removed = strip_report.fields_removed,
                            "[reasoning_compat] thinking validation 400 — aggressively stripped reasoning carriers, retrying same provider"
                        );
                        last_error = Some(GatewayError::Provider {
                            provider: provider_name.to_string(),
                            message: format!(
                                "[reasoning_compat] HTTP {}: thinking-parameter validation failed",
                                status_code
                            ),
                            status_code: Some(status_code),
                        });
                        continue;
                    }
                }

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
                // Tag thinking-validation 400s with the reasoning_compat
                // diagnostic so the aggregated per-attempt record shows why
                // the provider was skipped (Req 6.2).
                if reasoning_compat_cfg.enabled
                    && Self::is_thinking_validation_error(status_code, &body_text)
                {
                    return Err(GatewayError::Provider {
                        provider: provider_name.to_string(),
                        message: format!(
                            "[reasoning_compat] HTTP {}: {}",
                            status_code, body_text
                        ),
                        status_code: Some(status_code),
                    });
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
                                let idx = match tc_delta.get("index").and_then(|v| v.as_u64()) {
                                    Some(explicit) => explicit,
                                    None => Self::implicit_tool_call_index(
                                        &tool_calls_map,
                                        tc_delta,
                                    ),
                                };
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

        // Reasoning stays in its own carrier and is NOT copied into `content`.
        // Copying it made a turn that only thought look identical to a finished
        // answer, which defeated the degenerate-turn detection on the success
        // path and let agentic clients stop mid-work. `promote_reasoning_to_content`
        // still performs that fallback deliberately, but only where it is safe
        // (no tools in play) or as a last resort after failover is exhausted.
        let final_content = full_content;

        // Estimate tokens if provider didn't send usage. Reasoning counts:
        // it is generated output even when it never reaches `content`.
        if total_tokens == 0 {
            completion_tokens = ((final_content.len() + reasoning_content.len()) / 4) as u32;
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

        let mut response = OpenAIResponse {
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
        };

        // Streamed tool-call deltas routinely omit the `id`, and the merge above
        // seeds it empty; downstream validation reads an empty id as a malformed
        // call and throws the whole turn away. Mint one, and align a `stop` /
        // absent finish_reason with the tool call that is actually present.
        Self::repair_tool_calls(&mut response);

        Ok(response)
    }

    /// Slot a `tool_calls` delta that arrived without an `index` field.
    ///
    /// OpenAI always indexes streamed tool-call deltas, but some
    /// OpenAI-compatible providers omit `index` entirely. Folding every such
    /// delta into slot 0 concatenated the argument fragments of unrelated
    /// parallel calls into one unparseable string and silently dropped all but
    /// the first call.
    ///
    /// A delta that identifies a different call than the one in progress — by
    /// `id`, or by `function.name` when no id is supplied — opens the next slot.
    /// Anything else (a bare `arguments` fragment, or a repeat of the same
    /// identity) continues the call in progress. Parallel calls to the *same*
    /// tool with neither indices nor ids are genuinely indistinguishable on the
    /// wire; those still merge, which is the conservative outcome.
    fn implicit_tool_call_index(
        tool_calls: &std::collections::BTreeMap<u64, serde_json::Value>,
        delta: &serde_json::Value,
    ) -> u64 {
        fn field(value: &serde_json::Value, key: &str) -> String {
            value
                .get(key)
                .and_then(|v| v.as_str())
                .unwrap_or_default()
                .to_string()
        }
        fn fn_name(value: &serde_json::Value) -> String {
            value
                .get("function")
                .and_then(|f| f.get("name"))
                .and_then(|v| v.as_str())
                .unwrap_or_default()
                .to_string()
        }

        let Some((last_index, last_entry)) = tool_calls.iter().next_back() else {
            return 0;
        };

        let delta_id = field(delta, "id");
        let delta_name = fn_name(delta);
        // Pure continuation fragment: no identity of its own.
        if delta_id.is_empty() && delta_name.is_empty() {
            return *last_index;
        }

        let last_id = field(last_entry, "id");
        let identifies_new_call = if !delta_id.is_empty() && !last_id.is_empty() {
            delta_id != last_id
        } else if !delta_name.is_empty() {
            let last_name = fn_name(last_entry);
            !last_name.is_empty() && delta_name != last_name
        } else {
            // Supplying the id for the call already in progress.
            false
        };

        if identifies_new_call {
            last_index + 1
        } else {
            *last_index
        }
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
        // Turns that carried only reasoning — no answer text, no tool call.
        // These fail over (see the success arm below) because a tool-using
        // client cannot act on them, but they are kept so a request where every
        // provider stops after thinking still returns something rather than a
        // hard error.
 let mut reasoning_only_candidates: Vec<OpenAIResponse> = Vec::new();
 // One-shot guard for the reasoning continuation nudge (see the
 // reasoning-only branch in the success arm below): per request, a
 // single stalled turn whose thinking announces an action ("let me
 // ...") is retried on the same provider with a nudge before
 // failover gives up on it.
 let mut reasoning_nudge_done = false;
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
    // Reasoning-compat knobs for the buffered success path below (Req 4.6/
    // 4.7). Cloned before the guard is dropped, mirroring the other
    // snapshots.
    let reasoning_compat_cfg = config.reasoning_compat.clone();
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

            let (mut prepared_request, compression) = self
                .prepare_compressed_request_with_stats(
                    request,
                    model_group,
                    &provider_model,
                    &request_id,
                )
                .await;
            // Inject gateway search tools after compression so the injected
            // definitions are never compressed away. Applies to every
            // provider type; the Codex dispatch path re-injects
            // idempotently for itself.
            self.maybe_inject_search_tools(&mut prepared_request).await;
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
                    // ---- Repair the turn before judging it ----------------
                    // Order matters here. Each of these steps can turn what
                    // looks like a dead-end turn into a usable one, so judging
                    // emptiness first is how real tool calls get discarded.
                    let tools_present = prepared_request.extra.contains_key("tools");

                    // An inline `<think>…</think>` prefix becomes a proper
                    // reasoning carrier, so deliberation is never mistaken for
                    // the model's answer.
                    if Self::split_think_tags(&mut response) {
                        debug!(
                            provider = %provider_model.provider,
                            model = %provider_model.model,
                            "Split inline <think> block out of assistant content"
                        );
                    }

                    // Recover tool calls the model encoded as XML/text in
                    // `content` or in a reasoning carrier. This also runs later
                    // on the success path; doing it here means a recovered call
                    // counts as real work in the checks below, and the later
                    // call is a no-op once `tool_calls` exists.
                    if tools_present {
                        Self::translate_xml_tool_calls(&mut response, &prepared_request);
                    }

                    // Mint missing tool-call ids and align finish_reason so a
                    // structurally sloppy but perfectly usable call survives
                    // validation instead of being failed over.
                    Self::repair_tool_calls(&mut response);

                    // ---- Judge the turn ----------------------------------
                    // A turn carrying only reasoning is not an answer for a
                    // tool-using client; it is a turn that stopped mid-work.
                    // Promoting the chain of thought into `content` (the old
                    // unconditional behavior) made it indistinguishable from a
                    // finished reply, so the harness rendered the thinking and
                    // stopped. Fail over instead, keeping it as a last resort.
                    if tools_present && Self::reasoning_only_turn(&response) {
                        // One-shot continuation nudge: the turn's thinking
                        // announced an action ("let me ...") but stopped
                        // before emitting it. Retry the same provider once
                        // with the stalled turn echoed back plus a user-role
                        // instruction to act, before failing over. The nudge
                        // exists only in the resubmitted request — the
                        // client never sees it.
                        let revived = if !reasoning_nudge_done
                            && Self::reasoning_states_intent(&response)
                        {
                            reasoning_nudge_done = true;
                            info!(
                                provider = %provider_model.provider,
                                model = %provider_model.model,
                                "Reasoning-only turn announces an action; retrying same provider with a continuation nudge"
                            );
                            self.nudge_reasoning_continuation(
                                &provider_model,
                                &prepared_request,
                                &response,
                                active.clone(),
                                attempt_counter,
                            )
                            .await
                        } else {
                            None
                        }
                        .and_then(|mut resumed| {
                            // Same repair pipeline as a first attempt so a
                            // recovered tool call counts as real work.
                            Self::split_think_tags(&mut resumed);
                            if tools_present {
                                Self::translate_xml_tool_calls(
                                    &mut resumed,
                                    &prepared_request,
                                );
                            }
                            Self::repair_tool_calls(&mut resumed);
                            (!Self::reasoning_only_turn(&resumed)
                                && Self::response_has_content(&resumed))
                                .then_some(resumed)
                        });

                        if let Some(resumed) = revived {
                            info!(
                                provider = %provider_model.provider,
                                model = %provider_model.model,
                                "Continuation nudge revived the reasoning-only turn"
                            );
                            response = resumed;
                            // Fall through to the normal success path below.
                        } else {
                            warn!(
                                provider = %provider_model.provider,
                                model = %provider_model.model,
                                finish_reason = ?response
                                    .choices
                                    .first()
                                    .and_then(|c| c.finish_reason.as_deref()),
                                "Provider returned reasoning with no answer and no tool call; failing over"
                            );
                            cb.record_failure().await;
                            self.metrics.record_provider_failure_with_reason(
                                &provider_model.provider,
                                Some(
                                    "Provider stopped after its reasoning block — no answer text and no tool call"
                                        .to_string(),
                                ),
                                None,
                            );
                            attempts.push(ProviderAttempt::new(
                                provider_model.provider.clone(),
                                provider_model.model.clone(),
                                "Provider returned reasoning only (no answer text, no tool call)"
                                    .to_string(),
                                Some(200),
                            ));
                            // Annotate with the same gateway metadata the
                            // success path attaches so the last-resort
                            // return below needs no reprocessing.
                            let mut candidate = response;
                            let candidate_cost =
                                compute_actual_cost(&provider_model, &candidate.usage);
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
                            reasoning_only_candidates.push(candidate);
                            continue;
                        }
                    }

                    // With no tools in play there is no agent loop to stall, so
                    // surfacing the reasoning beats returning nothing at all.
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

            // --- Codex Search agent loop (any provider) ---
            // When the buffered response contains gateway-injected search
            // tool calls, execute them against the Codex search endpoint
            // with the gateway's OpenAI OAuth token and resubmit through
            // the normal dispatch pipeline until the model produces a
            // final answer. No-ops for Codex providers (intercepted during
            // Codex dispatch) and for responses without search tool calls.
            response = self
                .maybe_intercept_search_tools(
                    &provider_model,
                    &prepared_request,
                    response,
                    active.clone(),
                    attempt_counter,
                )
                .await?;

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
                        && Self::is_suspicious_truncation(&response, request.max_tokens);

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
        // Cache-aware actual cost (Req 3.5/3.7): the partial
        // response's usage already carries the provider's cache
        // token split, so price it with the same formula as the
        // success path. Base-price identical when no cache fields.
        let candidate_cost = compute_actual_cost(&provider_model, &candidate.usage);
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

        // Cache-aware actual cost (Req 3.5): price the response's usage
        // split (uncached / cache-read / cache-creation) at the model's
        // per-million rates. Bit-identical to the previous base-price
        // formula when the usage carries no cache fields, so providers
        // that never report cache telemetry are unaffected.
        let usage_known = response.usage.total_tokens > 0
            || response.usage.prompt_tokens > 0
            || response.usage.completion_tokens > 0;
    let total_cost = if usage_known {
        let cost = compute_actual_cost(&provider_model, &response.usage);
        if cost > 0.0 {
            self.metrics.add_cost(&provider_model.provider, cost);
        }
        record_cache_usage(
            &self.metrics,
            &provider_model.provider,
            &provider_model,
            &response.usage,
            cost,
        );
        cost
    } else {
        self.metrics
            .record_provider_unknown_cost(&provider_model.provider);
        0.0
    };

    // Reasoning-token attribution (Req 4.7): extract the reasoning /
    // thinking token count (any carrier shape, never double-counted) and
    // accrue the provider metric + dedicated reasoning cost when the
    // attribution knob is on.
    let reasoning_usage = reasoning_compat::cost::extract_reasoning_usage(&response.usage);
    if reasoning_usage.reasoning_tokens > 0 && reasoning_compat_cfg.attribute_reasoning_cost {
        let reasoning_cost = reasoning_compat::cost::reasoning_cost(
            &provider_model,
            reasoning_usage.reasoning_tokens,
        );
        self.metrics.add_reasoning_usage(
            &provider_model.provider,
            u64::from(reasoning_usage.reasoning_tokens),
            reasoning_cost,
        );
    }

        // Cache-aware sticky routing (Req 1.1): remember which provider
        // served this conversation prefix so the next turn is promoted back
        // to it. The hash is computed from the original client request
        // (`request`), not the provider-mutated outgoing copy.
        self.record_sticky_success(
            request,
            &provider_model.provider,
            &provider_model.model,
            &response.usage,
        )
        .await;

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
        // Prompt-cache log telemetry (Req 4.3): attach the cache token
        // split, realized savings, and the prefix affinity hash so the
        // request logger can persist them. Only present when the provider
        // actually reported cached tokens.
        {
            let cache_usage = extract_cache_usage(&response.usage);
            if cache_usage.cache_read_input_tokens > 0 || cache_usage.cache_creation_input_tokens > 0
            {
                let prefix_hash = StickyCache::compute_prefix_hash(request);
                let savings = cache_savings_cents(&provider_model, &response.usage, total_cost);
                response.extra.insert(
                    "gateway_cache_read_tokens".to_string(),
                    serde_json::json!(cache_usage.cache_read_input_tokens as i64),
                );
                response.extra.insert(
                    "gateway_cache_creation_tokens".to_string(),
                    serde_json::json!(cache_usage.cache_creation_input_tokens as i64),
                );
                response.extra.insert(
                    "gateway_cache_savings_cents".to_string(),
                    serde_json::json!(savings),
                );
                response.extra.insert(
                    "gateway_prefix_hash".to_string(),
                    serde_json::json!(crate::logger::encode_prefix_hash(prefix_hash)),
                );
            }
        }
        response.extra.insert(
            "gateway_compression".to_string(),
            serde_json::to_value(&compression)
                .expect("CompressionStats serialization must succeed"),
        );
        // Reasoning-compat log telemetry (Req 4.6/4.7): expose the
        // per-request reasoning-token count so the request logger can
        // persist it. The compat stage's actions JSON
        // (`gateway_reasoning_compat_actions`) is attached by the
        // per-attempt dispatch right where the report exists. The handler
        // strips both keys before returning the response to the client.
        if reasoning_usage.reasoning_tokens > 0 {
            response.extra.insert(
                "gateway_reasoning_tokens".to_string(),
                serde_json::json!(reasoning_usage.reasoning_tokens),
            );
        }

        return Ok(response);
                }
                Err(e) => {
                    // Record failure — except rate-limit-class errors (HTTP 429
                    // and rate-limit-shaped error-in-200 envelopes). Those are
                    // governed exclusively by the dedicated upstream cooldown
                    // applied below (and enforced as a routing gate by
                    // `select_provider_order` / the defense-in-depth check
                    // above), so a rate-limited provider is paused without its
                    // circuit-breaker health also being eroded: three 429s must
                    // not open the breaker on top of the cooldown.
                    if !Self::is_rate_limit_class_error(&e) {
                        cb.record_failure().await;
                    }

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

        // Every provider stopped after its reasoning block. Failing over was
        // worth trying — usually another provider completes the turn — but once
        // the chain is exhausted, surfacing the model's thinking beats returning
        // a hard error, which is what the client saw before this fallback
        // existed. The reasoning is promoted into `content` so the harness has
        // something to render.
        if let Some(mut best) = reasoning_only_candidates
            .into_iter()
            .max_by_key(|response| {
                response
                    .choices
                    .first()
                    .and_then(Self::reasoning_text)
                    .map(str::len)
                    .unwrap_or(0)
            })
        {
            let chosen_provider = best
                .extra
                .get("gateway_provider")
                .and_then(serde_json::Value::as_str)
                .unwrap_or("unknown")
                .to_string();
            Self::promote_reasoning_to_content(&mut best);
            warn!(
                provider = %chosen_provider,
                "All providers returned reasoning without an answer or tool call; returning the longest reasoning as content"
            );
            return Ok(best);
        }

        Err(GatewayError::AllProvidersFailed(AggregatedError::new(
            attempts,
        )))
    }

    /// The non-empty reasoning carrier on a choice, if any.
    ///
    /// `reasoning` is the Nano-GPT/OpenRouter spelling; `reasoning_content` is
    /// the DeepSeek-style spelling used by GLM and friends. Both are checked
    /// everywhere the gateway reasons about reasoning, because a provider only
    /// ever populates one of them and code that checks a single key silently
    /// misses half the ecosystem.
    fn reasoning_text(choice: &Choice) -> Option<&str> {
        for key in ["reasoning", "reasoning_content"] {
            if let Some(text) = choice
                .message
                .extra
                .get(key)
                .and_then(serde_json::Value::as_str)
                .filter(|text| !text.trim().is_empty())
            {
                return Some(text);
            }
        }
        None
    }

    /// Split a leading `<think>…</think>` block out of assistant content into
    /// the `reasoning_content` carrier. Returns `true` when content was rewritten.
    ///
    /// Some reasoning models — the GLM family in particular — inline their chain
    /// of thought in `content` rather than using a dedicated reasoning field.
    /// Left in place it reads as the assistant's answer, so a turn that only
    /// thought and never acted looks like a finished reply and an agentic client
    /// stops mid-work. Extracting it lets [`Self::reasoning_only_turn`] see the
    /// turn for what it is, and lets the XML tool-call extractors work on the
    /// model's real output instead of its deliberation.
    ///
    /// Only a block at the very start of the content counts as thinking — that
    /// is where these models emit it — so prose or code that merely mentions
    /// `<think>` further in is left alone. An unclosed `<think>` (the model was
    /// cut off mid-thought) makes the whole remainder reasoning.
    fn split_think_tags(response: &mut OpenAIResponse) -> bool {
        const OPEN: &str = "<think>";
        const CLOSE: &str = "</think>";

        let Some(choice) = response.choices.first_mut() else {
            return false;
        };
        let text = choice.message.content_as_text();
        let trimmed = text.trim_start();
        if !trimmed.starts_with(OPEN) {
            return false;
        }

        let body = &trimmed[OPEN.len()..];
        let (thought, rest) = match body.find(CLOSE) {
            Some(end) => (&body[..end], &body[end + CLOSE.len()..]),
            None => (body, ""),
        };

        let thought = thought.trim();
        if !thought.is_empty() {
            let existing = choice
                .message
                .extra
                .get("reasoning_content")
                .and_then(serde_json::Value::as_str)
                .unwrap_or("");
            let merged = if existing.is_empty() {
                thought.to_string()
            } else {
                format!("{existing}{thought}")
            };
            choice.message.extra.insert(
                "reasoning_content".to_string(),
                serde_json::Value::String(merged),
            );
        }

        let rest = rest.trim();
        choice.message.content = if rest.is_empty() {
            serde_json::Value::Null
        } else {
            serde_json::Value::String(rest.to_string())
        };
        true
    }

    /// Repair structurally incomplete `tool_calls` so a usable tool call is
    /// never thrown away by [`Self::response_has_content`].
    ///
    /// `id` is a correlation token the client echoes back on the tool result;
    /// when a provider omits it (common on OpenAI-compatible gateways, and the
    /// streamed-delta merge in [`Self::reassemble_sse_response`] seeds it empty)
    /// the gateway can mint one safely. Before this, a missing id failed
    /// validation and the entire turn — tool call included — was discarded and
    /// failed over, which is one of the ways an agent loop stalls with no
    /// visible error.
    ///
    /// A call with no `function.name` is genuinely unusable and is left for
    /// validation to reject.
    ///
    /// Also normalizes `finish_reason` to `tool_calls` when the provider
    /// reported `stop` or omitted it entirely, because clients branch on
    /// `finish_reason` to decide whether to run a tool. `length` and
    /// `content_filter` are preserved — those carry information this function
    /// must not erase (truncation detection reads `length`).
    fn repair_tool_calls(response: &mut OpenAIResponse) -> bool {
        let Some(choice) = response.choices.first_mut() else {
            return false;
        };
        let Some(calls) = choice
            .message
            .extra
            .get_mut("tool_calls")
            .and_then(serde_json::Value::as_array_mut)
        else {
            return false;
        };
        if calls.is_empty() {
            return false;
        }

        let mut repaired = false;
        for call in calls.iter_mut() {
            let Some(object) = call.as_object_mut() else {
                continue;
            };
            let has_name = object
                .get("function")
                .and_then(|f| f.get("name"))
                .and_then(serde_json::Value::as_str)
                .is_some_and(|name| !name.is_empty());
            if !has_name {
                continue;
            }
            let has_id = object
                .get("id")
                .and_then(serde_json::Value::as_str)
                .is_some_and(|id| !id.is_empty());
            if !has_id {
                object.insert(
                    "id".to_string(),
                    serde_json::Value::String(format!(
                        "call_{}",
                        uuid::Uuid::new_v4().simple()
                    )),
                );
                repaired = true;
            }
            if object
                .get("type")
                .and_then(serde_json::Value::as_str)
                .is_none()
            {
                object.insert("type".to_string(), serde_json::json!("function"));
                repaired = true;
            }
        }

        if matches!(choice.finish_reason.as_deref(), None | Some("stop")) {
            choice.finish_reason = Some("tool_calls".to_string());
            repaired = true;
        }

        repaired
    }

    /// True when a provider turn produced reasoning and nothing else: no answer
    /// text and no tool call.
    ///
    /// For a tool-using client this is not an answer, it is a turn that stopped
    /// mid-work. Handing it back as a completed reply is what makes a harness
    /// like Kilo Code render the chain of thought and stop.
    fn reasoning_only_turn(response: &OpenAIResponse) -> bool {
        let Some(choice) = response.choices.first() else {
            return false;
        };
        if !Self::content_is_empty(&choice.message.content) {
            return false;
        }
        let has_tool_calls = choice
            .message
            .extra
            .get("tool_calls")
            .and_then(serde_json::Value::as_array)
            .is_some_and(|calls| !calls.is_empty());
        if has_tool_calls {
            return false;
        }
        Self::reasoning_text(choice).is_some()
    }

    fn promote_reasoning_to_content(response: &mut OpenAIResponse) -> bool {
        let Some(choice) = response.choices.first_mut() else {
            return false;
        };
        if !Self::content_is_empty(&choice.message.content) {
            return false;
        }
        let Some(text) = Self::reasoning_text(choice).map(str::to_string) else {
            return false;
        };
 choice.message.content = serde_json::Value::String(text);
        true
 }

    /// Lowercase substrings that mark a reasoning block as having ended
    /// mid-plan: the model announced an action it never emitted. Matched
    /// only against the reasoning carrier of an already reasoning-only
    /// turn, so a false positive costs one wasted nudge round-trip, not a
    /// wrong response.
    const REASONING_INTENT_MARKERS: &[&str] = &[
        "let me",
        "let's",
        "i'll",
        "i will",
        "i am going",
        "i'm going",
        "i need to",
        "i want to",
        "i should",
        "i must",
        "i could",
        "i can use",
        "we need to",
        "we should",
        "we will",
        "next step",
    ];

    const REASONING_CONTINUATION_NUDGE: &str = "Your previous turn ended inside your private \
reasoning: you announced an action (for example \"let me ...\") but emitted no answer text and \
no tool call. The user cannot see your reasoning, so nothing happened. Continue the original \
task now: emit your next native tool call to act on your plan, or write the final answer as \
visible content. Do not restate your plan and do not end your turn without doing one of the two.";

    /// True when the response is a reasoning-only turn whose thinking
    /// announces an action ("let me ...", "I'll ..."), i.e. the model
    /// stopped mid-work rather than concluding.
    fn reasoning_states_intent(response: &OpenAIResponse) -> bool {
        let Some(choice) = response.choices.first() else {
            return false;
        };
        let Some(text) = Self::reasoning_text(choice) else {
            return false;
        };
        let lower = text.to_lowercase();
        Self::REASONING_INTENT_MARKERS
            .iter()
            .any(|marker| lower.contains(marker))
    }

    /// One-shot same-provider retry for a reasoning-only turn that
    /// announced an action. Appends the stalled turn (its thinking echoed
    /// as capped assistant content) and a user-role continuation
    /// instruction to a clone of the prepared request, then resubmits
    /// through the normal dispatch pipeline. Returns the provider's
    /// response, or `None` on dispatch failure. Judging the revived turn
    /// (repair pipeline + usability check) stays with the caller so it can
    /// still fall back to the reasoning-only failover path.
    async fn nudge_reasoning_continuation(
        &self,
        provider_model: &ProviderModel,
        prepared_request: &OpenAIRequest,
        reasoning_only: &OpenAIResponse,
        active: Option<ActiveRequestHandle>,
        attempt_counter: usize,
    ) -> Option<OpenAIResponse> {
        let mut nudged = prepared_request.clone();
        if let Some(choice) = reasoning_only.choices.first() {
            // Echo the stalled turn back as the assistant message, with the
            // thinking carried as plain content: the reasoning-compat strip
            // can remove `reasoning`/`reasoning_content` carriers from
            // outgoing messages (and drop a message left empty by the
            // strip), but plain assistant content survives to every
            // provider. Capped so a long thinking block cannot balloon the
            // resubmission.
            let echo = Self::reasoning_text(choice)
                .map(|text| text.chars().take(2000).collect::<String>())
                .unwrap_or_default();
            let content = if echo.trim().is_empty() {
                choice.message.content.clone()
            } else {
                serde_json::Value::String(echo)
            };
            nudged.messages.push(Message {
                role: "assistant".to_string(),
                content,
                extra: Default::default(),
            });
        }
        nudged.messages.push(Message {
            role: "user".to_string(),
            content: serde_json::Value::String(Self::REASONING_CONTINUATION_NUDGE.to_string()),
            extra: Default::default(),
        });
        match self
            .attempt_with_retry(
                &provider_model.provider,
                &nudged,
                provider_model,
                active,
                attempt_counter,
            )
            .await
        {
            Ok(response) => Some(response),
            Err(err) => {
                warn!(
                    provider = %provider_model.provider,
                    model = %provider_model.model,
                    error = %err,
                    "Reasoning continuation nudge failed; falling over"
                );
                None
            }
        }
    }

    /// Remove a gateway search tool call the gateway is not going to execute, and
    /// make sure a usable turn remains.
    ///
    /// Leaking `codex_search`/`codex_web` to a client that never declared them is
    /// a hard stall rather than a visible failure: the harness cannot run the
    /// tool, cannot produce the `role: tool` reply the protocol now demands, and
    /// has no error to surface.
    fn discard_unexecutable_search_call(response: &mut OpenAIResponse) {
        crate::codex::search::interceptor::strip_gateway_tool_calls(response);
        Self::backfill_empty_turn(
            response,
            "The gateway's web-search tool was requested but is currently unavailable, so no search ran this turn. \
             Continue the task with your own tools, or say what information you still need.",
        );
    }

    /// Give an otherwise-empty assistant turn something to say. Returns `true`
    /// when `note` was written.
    ///
    /// A turn with neither content nor tool calls is rendered by the SSE
    /// synthesizer as a lone terminal `finish_reason` chunk. Clients read that as
    /// a completed but empty reply and stop — indistinguishable from the task
    /// actually being finished. An explanation keeps the agent able to proceed.
    fn backfill_empty_turn(response: &mut OpenAIResponse, note: &str) -> bool {
        let Some(choice) = response.choices.first_mut() else {
            return false;
        };
        let has_tool_calls = choice
            .message
            .extra
            .get("tool_calls")
            .and_then(serde_json::Value::as_array)
            .is_some_and(|calls| !calls.is_empty());
        if has_tool_calls || !Self::content_is_empty(&choice.message.content) {
            return false;
        }
        choice.message.content = serde_json::Value::String(note.to_string());
        choice.finish_reason = Some("stop".to_string());
        true
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
        // Some providers put tool calls in a reasoning carrier instead of
        // `content`: Nano-GPT thinking models use `reasoning`, DeepSeek-style
        // providers and the GLM family use `reasoning_content`. Checking only
        // `reasoning` left GLM tool calls unrecoverable, so the turn arrived at
        // the client as a thinking block with no tool to run.
        let reasoning_key = ["reasoning", "reasoning_content"].into_iter().find(|key| {
            choice
                .message
                .extra
                .get(*key)
                .and_then(|v| v.as_str())
                .is_some_and(|text| !text.is_empty())
        });
        let reasoning_text = reasoning_key
            .and_then(|key| choice.message.extra.get(key))
            .and_then(|v| v.as_str())
            .unwrap_or("")
            .to_string();
        let combined_text = if content_text.is_empty() && !reasoning_text.is_empty() {
            reasoning_text.clone()
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

        // If tool calls were extracted from the reasoning field, clear whichever
        // carrier they came from so the translated response doesn't ship stale
        // XML back to the client as a thinking block.
        if !reasoning_text.is_empty() && content_text.is_empty() {
            if let Some(key) = reasoning_key {
                choice.message.extra.remove(key);
            }
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
            // Whitespace-only content is not an answer. Providers that stop
            // after their thinking phase often emit a stray newline as the
            // whole message body; treating that as content lets a turn that
            // did no work look like a finished reply.
            serde_json::Value::String(s) => s.trim().is_empty(),
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

        // Never cache a turn with no answer text. A provider that stopped after
        // its reasoning block produces exactly this shape, and caching it would
        // replay that dead end for every identical prefix — turning a transient
        // provider hiccup into a permanent one.
        if Self::content_is_empty(&choice.message.content) {
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
/// Idempotent. Called from the streaming relay and the buffered-response
/// diagnostic when XML tool use is detected. Combos are sticky for the
/// process lifetime — see [`Self::is_xml_tool_combo`].
pub fn mark_xml_tool_combo(&self, provider: &str, model: &str) {
    if let Ok(mut set) = self.xml_tool_combos.write() {
        set.insert(Self::xml_combo_key(provider, model));
    }
}

    /// True if this `provider`/`model` combo has been observed ending a streamed
    /// turn with reasoning only. Such combos take the buffer-and-retry path when
    /// the request carries `tools` — see [`Self::degenerate_stream_combos`].
    pub fn is_degenerate_stream_combo(&self, provider: &str, model: &str) -> bool {
        self.degenerate_stream_combos
            .read()
            .map(|set| set.contains(&Self::xml_combo_key(provider, model)))
            .unwrap_or(false)
    }

    /// Record a `provider`/`model` combo as one that ends streamed turns with
    /// reasoning and no answer. Idempotent; sticky for the process lifetime, for
    /// the same reason the XML combos are: flipping transport mode back and
    /// forth mid-conversation is worse than committing to the safe path.
    pub fn mark_degenerate_stream_combo(&self, provider: &str, model: &str) {
        if let Ok(mut set) = self.degenerate_stream_combos.write() {
            set.insert(Self::xml_combo_key(provider, model));
        }
    }

/// True when the tool-calling system hint should be injected for a
/// tools-bearing request to this combo.
///
/// Only **learned combos** — combos observed emitting XML-style tool use
/// (streaming relay or buffered diagnostic) — get the hint, and it stays
/// enabled for the process lifetime: toggling guidance on and off
/// mid-conversation is exactly the context mutation that makes models
/// flag injected instructions as a prompt-injection attack in their
/// reasoning. Everything else (including known XML-prone families)
/// starts clean; a first XML-flavored response is repaired transparently
/// by `translate_xml_tool_calls` and marks the combo for subsequent
/// requests.
pub(crate) fn should_inject_tool_hint(&self, provider: &str, model: &str) -> bool {
    self.is_xml_tool_combo(provider, model)
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
        self.route_request_with_exclusions(request, active, &[])
            .await
    }

    /// Like [`Self::route_request`], but removes any `provider:model` entry
    /// listed in `exclude` from the initial failover chain. Used by the
    /// streaming pass-through path so a provider that just returned a
    /// rate-limit response is not immediately retried by the buffered
    /// fallback: the applied cooldown also gates selection, but the explicit
    /// exclusion makes the guarantee airtight even when the parsed cooldown
    /// window is near zero. Smart-routing cascade escalation re-derives its
    /// own candidates and is not filtered.
    pub(crate) async fn route_request_with_exclusions(
        &self,
        request: &OpenAIRequest,
        active: Option<ActiveRequestHandle>,
        exclude: &[String],
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
        let mut providers = self.select_provider_order(&model_group).await;
        // Cache-aware sticky routing (Req 1.2): promote the provider that
        // last served this conversation prefix to the head of the list.
        // Applied after `select_provider_order`'s health gates so an
        // unhealthy sticky provider falls through to normal routing.
        self.promote_sticky_provider(&prepared_request, &mut providers)
            .await;
        // Drop explicitly excluded provider:model entries (streaming 429
        // fallback — see `route_request_streaming_excluding`).
        let providers = if exclude.is_empty() {
            providers
        } else {
            providers
                .into_iter()
                .filter(|pm| {
                    let key = format!("{}:{}", pm.provider, pm.model);
                    !exclude.iter().any(|k| k == &key)
                })
                .collect()
        };
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

/// Build the native tool-calling system hint inserted into outgoing
/// requests that carry `tools` for learned XML combos. Shared by the
/// buffered ([`Self::attempt_with_retry`]) and streaming pass-through
/// ([`Self::route_request_streaming`]) paths so both send identical
/// guidance to the provider.
///
/// The text is deliberately short, attributed (`[gateway]`), and
/// positively framed, and it contains no XML tag examples: long
/// anonymous imperative blocks — especially ones enumerating forbidden
/// markup — match the fingerprints models associate with prompt-injection
/// payloads and get flagged in reasoning output, eroding user trust.
fn tool_calling_system_hint() -> Message {
    Message {
        role: "system".to_string(),
        content: serde_json::Value::String(
            "[gateway] Tools on this endpoint are available through the \
             API's native function-calling interface. When a tool would \
             help, emit a native `tool_calls` payload using the exact \
             names and argument schemas from the provided tools list. \
             When no tool is needed, reply with plain text."
                .to_string(),
        ),
        extra: serde_json::Map::new(),
    }
}

/// Insert the tool-calling hint directly after the last system message
/// so it reads as operator guidance in the trusted system region of the
/// prompt. A system message appended at the tail — after user and tool
/// content, immediately before generation — is the canonical
/// prompt-injection position and is what reasoning models flag as
/// suspicious. When the conversation has no system message, the hint
/// becomes the first message. Positioning right after the system block
/// also keeps the prefix stable for provider prompt caching.
fn insert_tool_calling_hint(messages: &mut Vec<Message>) {
    let hint = Self::tool_calling_system_hint();
    let pos = messages
        .iter()
        .rposition(|message| message.role == "system")
        .map_or(0, |index| index + 1);
    messages.insert(pos, hint);
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

    /// Like [`Self::route_request_streaming`], but skips any `provider:model`
    /// entry listed in `exclude` when picking the first eligible pass-through
    /// provider. Used by the streaming handler's pre-content failover loop
    /// (task 6.1, Req 4.1) to retry the next provider after one disconnects or
    /// errors before any content was forwarded. Exclusion is keyed per
    /// `provider:model` — matching the circuit-breaker key — so a provider
    /// that offers several models in the group remains eligible via its other
    /// models after one of them failed.
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
        let mut providers = self.select_provider_order(&model_group).await;
        // Cache-aware sticky routing (Req 1.2): promote the sticky provider
        // for this prefix after the health gates inside
        // `select_provider_order` (mirrors the buffered path).
        self.promote_sticky_provider(&prepared_request, &mut providers)
            .await;
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
        let mut chosen_without_capacity: Option<(ProviderModel, Provider)> = None;
        let mut chosen: Option<(ProviderModel, ProviderConcurrencyPermit)> = None;
        for provider_model in &providers {
            // Req 4.1: skip provider:model entries already tried in the
            // failover loop. Keyed per provider:model (same as the circuit
            // breaker) so other models from the same provider stay eligible.
            let candidate_key = format!("{}:{}", provider_model.provider, provider_model.model);
            if exclude.iter().any(|p| p == &candidate_key) {
                debug!(provider = %provider_model.provider, model = %provider_model.model, "Excluded by failover, skipping (streaming)");
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
            let provider_cfg = {
                let config = self.config.read().await;
                config
                    .providers
                    .iter()
                    .find(|provider| provider.name == provider_model.provider)
                    .cloned()
            };
            let Some(provider_cfg) = provider_cfg else {
                continue;
            };
            let Some(concurrency_permit) = self.try_acquire_provider_concurrency(
                &provider_model.provider,
                provider_cfg.max_connections,
            ) else {
                debug!(provider = %provider_model.provider, model = %provider_model.model, "Provider concurrency limit reached, checking next pass-through candidate");
                chosen_without_capacity
                    .get_or_insert_with(|| (provider_model.clone(), provider_cfg));
                continue;
            };
            chosen = Some((provider_model.clone(), concurrency_permit));
            break;
        }

        // No pass-through slot is immediately available. Preserve provider
        // order by waiting for the first saturated candidate, then use the
        // normal buffered route so cross-provider failover remains intact.
        let (provider_model, concurrency_permit) = match chosen {
            Some(chosen) => chosen,
            None => {
                let Some((provider_model, provider_cfg)) = chosen_without_capacity else {
                    debug!("No eligible pass-through provider, using buffered path");
                    return Ok(StreamingResponse::Buffered(
                        self.route_request(request, active.clone()).await?,
                    ));
                };
                debug!(provider = %provider_model.provider, model = %provider_model.model, "Waiting for provider concurrency before buffered dispatch");
                drop(provider_cfg);
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

    // Codex Search is active → route to the buffered path BEFORE running
    // compression. Streaming pass-through cannot intercept tool calls to
    // execute search server-side, and without injection the model never
    // sees the codex_search/codex_web tools. The buffered path injects the
    // tool definitions after its own compression and intercepts/executes
    // any search tool calls before returning the final response. Checking
    // here avoids burning a compression pass that the buffered path will
    // redo anyway.
    if self.codex_search_ready().await.is_some() {
        debug!(
            provider = %provider_model.provider,
            "Codex search active — routing streaming request through buffered path for tool-call interception"
        );
        drop(concurrency_permit);
        return Ok(StreamingResponse::Buffered(
            self.route_request(request, active.clone()).await?,
        ));
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
    let provider_cfg = {
            let config = self.config.read().await;
            config
                .providers
                .iter()
                .find(|p| p.name == provider_model.provider)
                .cloned()
        };
        let Some(provider_cfg) = provider_cfg else {
            return Err(GatewayError::Configuration(format!(
                "Provider '{}' not found in config",
                provider_model.provider
            )));
        };

        // Codex (oauth + openai) is handled end-to-end by CodexProviderClient
        // and cannot pass-through; providers needing response transformation
        // (Bedrock / XML-tool rewrite / Kimi-Nano) likewise must buffer.
        let is_codex = provider_cfg.auth_method.as_deref() == Some("oauth")
            && provider_cfg.provider_type == "openai";
        // INVARIANT: a Bedrock request must never be dispatched from the
        // pass-through path. That path posts to `{base_url}/chat/completions`
        // unconditionally and would send a `MantleApi::Responses` model to the
        // wrong endpoint. Bedrock's own dispatch seam (dispatch_mantle /
        // dispatch_mantle_stream) selects the endpoint per model and de-dupes
        // compaction triggers, so Bedrock streams must always go through the
        // buffered route (which calls `route_request` -> the Bedrock provider).
        // This is enforced structurally by the `|| is_bedrock` gate below, not
        // left to `provider_needs_transformation` returning true incidentally.
        let is_bedrock = provider_cfg.provider_type == "bedrock";
        // A provider/model combo previously observed emitting XML-style tool
        // calls takes the buffer-and-translate path when the request carries
        // `tools`, so the XML can be rewritten into native `tool_calls`. Unknown
        // combos stream optimistically; the relay learns and marks them if XML
        // tool use is detected (see `relay_passthrough_stream`).
        let tools_present = prepared_request.extra.contains_key("tools");
        let known_xml_combo = tools_present
            && self.is_xml_tool_combo(&provider_model.provider, &provider_model.model);
        // A combo that has already ended a streamed turn with reasoning and no
        // tool call also buffers, so the degenerate-turn failover can run. On a
        // live relay the terminal chunk is long gone before the shortfall is
        // detectable, leaving nothing to retry.
        let known_degenerate_combo = tools_present
            && self.is_degenerate_stream_combo(&provider_model.provider, &provider_model.model);
        if is_codex
            || is_bedrock
            || self.provider_needs_transformation(&provider_cfg, &prepared_request)
            || known_xml_combo
            || known_degenerate_combo
        {
            debug!(provider = %provider_model.provider, "Provider needs transformation or is Codex, using buffered path with full failover");
            drop(concurrency_permit);
            return Ok(StreamingResponse::Buffered(
                self.route_request(request, active.clone()).await?,
            ));
        }

        // Snapshot config fields for the outgoing pass-through request.
        let api_key = provider_cfg.resolve_api_key().unwrap_or_default();
        let is_oauth_provider = provider_cfg.auth_method.as_deref() == Some("oauth");
        let provider_type = provider_cfg.provider_type.clone();
        let configured_base_url = provider_cfg.base_url.clone();
        let custom_headers = provider_cfg.custom_headers.clone();
        let pool_config = provider_cfg.connection_pool.clone();
        let ttfb_timeout_secs = provider_cfg.effective_ttfb_timeout(&provider_model.model);
        let ttfb_timeout = Duration::from_secs(ttfb_timeout_secs);

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
            drop(concurrency_permit);
            return Ok(StreamingResponse::Buffered(
                self.route_request(request, active.clone()).await?,
            ));
        }

        // Consume an internal rate-limit token for this streaming dispatch,
        // mirroring the buffered path (`route_with_failover_for_group`):
        // streaming traffic must not bypass `rate_limit_per_minute`, or a
        // streaming-heavy workload would never be throttled locally and the
        // provider would answer with real 429s. On exhaustion, drop to the
        // buffered path — its own consume() gate then skips this provider
        // with proper attempt logging and fails over to the next candidate.
        // (The token is consumed here, after the transformation checks, so
        // requests that end up buffered anyway are not double-charged.)
        let rate_limiter = self.get_rate_limiter(&provider_model.provider).await;
        if !rate_limiter.consume().await {
            warn!(provider = %provider_model.provider, "Rate limit exhausted before streaming dispatch, using buffered path");
            self.metrics
                .record_provider_rate_limit_exhausted(&provider_model.provider);
            drop(concurrency_permit);
            return Ok(StreamingResponse::Buffered(
                self.route_request(request, active.clone()).await?,
            ));
        }

        // Base URL normalization — strip trailing '/', ensure '/v1'. Bedrock
        // never reaches this pass-through path (it takes the buffered gate
        // above via `|| is_bedrock`), so no Mantle special-case is needed here.
        let mut base_url = configured_base_url.as_deref().unwrap_or_default().to_string();
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
// Reasoning-compat per-attempt stage (reasoning-failover-compat
// spec, Req 6.1/6.2), mirroring the buffered dispatch path: detect
// carriers on the original request, strip/preserve for this
// target, normalize reasoning params. Source-model attribution
// (Task 6) comes from the sticky prefix affinity; None on a miss.
// Pass-through streams carry no response object, so the report is
// only logged here; usage metrics land in `record_streaming_success`.
let reasoning_compat_cfg = self.config.read().await.reasoning_compat.clone();
if reasoning_compat_cfg.enabled {
let source_ref = self.model_affinity_source(request, &reasoning_compat_cfg);
let report = reasoning_compat::prepare_attempt(
&mut outgoing,
request,
source_ref,
&provider_model,
&reasoning_compat_cfg,
);
            if report.strip.messages_touched > 0 || report.strip.thinking_blocks > 0 {
                reasoning_compat::policy::log_strip_action(
                    &report.strip,
                    report.decision,
                    &crate::router::trace_id::generate_trace_id(None),
                );
            }
        }
        let stripped = Self::sanitize_request_for_provider(&mut outgoing, &provider_type);
        if stripped > 0 {
            info!(provider = %provider_model.provider, provider_type = %provider_type, fields_removed = stripped, "Sanitized streaming request for provider");
        }
        Self::normalize_message_tool_calls(&mut outgoing.messages);

        // Strip image content parts when the target model is known not to
        // support vision inputs. Mirrors the buffered path so streaming
        // pass-through does not burn a 4xx round-trip before the buffered
        // fallback's reactive strip rescues the request.
        let supports_vision = self
            .context_manager
            .get_capabilities(&provider_model.model)
            .map(|caps| caps.supports_vision)
            .unwrap_or(false);
        let images_stripped = Self::strip_image_content_if_unsupported(
            &mut outgoing,
            supports_vision,
            &provider_model.provider,
            &provider_model.model,
        );
        if images_stripped > 0 {
            info!(
                provider = %provider_model.provider,
                model = %provider_model.model,
                images_stripped,
                "Removed image content from streaming request for non-vision model"
            );
        }
        if outgoing.extra.contains_key("tools") {
            Self::reverse_translate_tool_history(&mut outgoing.messages);
            // Same conditional policy as the buffered path: only learned XML
            // combos get the hint. `known_xml_combo` above already routed
            // learned combos to the buffered path, so a hint here only fires
            // when the exclusion list skipped that check for this candidate.
            if self.should_inject_tool_hint(&provider_model.provider, &provider_model.model) {
                debug!(
                    provider = %provider_model.provider,
                    model = %provider_model.model,
                    "Injecting tool-calling system hint (streaming, learned XML combo)"
                );
                Self::insert_tool_calling_hint(&mut outgoing.messages);
            }
        }

        // Prompt-cache decorations (Req 2.1 / 1.5), mirroring the
        // buffered dispatch path: explicit-cache breakpoints and the
        // OpenRouter session id, applied after all message mutations.
        let cache_aware_routing_cfg = self.config.read().await.cache_aware_routing.clone();
        let is_openrouter = provider_type.eq_ignore_ascii_case("openrouter")
            || configured_base_url
                .as_deref()
                .map_or(false, |u| u.contains("openrouter.ai"));
        self.apply_cache_routing_decorations(
            &mut outgoing,
            request,
            &provider_model,
            is_openrouter,
            &cache_aware_routing_cfg,
        );

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
        // The permit acquired during candidate selection moves with the live
        // response so it remains held until the downstream relay finishes.
        let send_result =
            tokio::time::timeout(ttfb_timeout, req_builder.json(&outgoing).send()).await;
        let response = match send_result {
            Ok(Ok(resp)) => resp,
            Ok(Err(e)) => {
                warn!(provider = %provider_model.provider, error = %e, "Streaming pass-through send failed, falling back to buffered path with full failover");
                drop(concurrency_permit);
                return Ok(StreamingResponse::Buffered(
                    self.route_request(request, active.clone()).await?,
                ));
            }
            Err(_) => {
                warn!(provider = %provider_model.provider, ttfb_timeout_secs, "TTFB timeout (streaming) — falling back to buffered path with full failover");
                drop(concurrency_permit);
                return Ok(StreamingResponse::Buffered(
                    self.route_request(request, active.clone()).await?,
                ));
            }
        };

        let status = response.status();
        if !status.is_success() {
            let status_code = status.as_u16();
            // Capture headers before draining so Retry-After style signals
            // can feed the cooldown parser.
            let response_headers = response.headers().clone();
            let body_text = response.text().await.unwrap_or_default();
            if Self::is_rate_limited(status_code, &body_text) {
                // Rate-limit response on the streaming pass-through: apply
                // the dedicated upstream cooldown (RateLimiter + durable
                // metrics store, both consulted by the routing gates) and
                // exclude this provider:model from the buffered fallback so
                // it is not immediately retried while the cooldown is
                // active. No circuit-breaker failure is recorded — a
                // rate-limited provider is paused, not unhealthy.
                let cooldown = self
                    .parse_rate_limit_cooldown(
                        &provider_model.provider,
                        Some(&response_headers),
                        &body_text,
                    )
                    .await;
                let rate_limiter = self.get_rate_limiter(&provider_model.provider).await;
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
                    Self::friendly_failure_reason(Some(status_code), &body_text),
                    deadline,
                );
                warn!(
                    provider = %provider_model.provider,
                    status = status_code,
                    cooldown_ms = cooldown.as_millis() as u64,
                    "Rate-limit response on streaming pass-through; cooldown applied, failing over via buffered path"
                );
                drop(concurrency_permit);
                let mut excluded = exclude.to_vec();
                let failed_key = format!("{}:{}", provider_model.provider, provider_model.model);
                if !excluded.contains(&failed_key) {
                    excluded.push(failed_key);
                }
                return Ok(StreamingResponse::Buffered(
                    self.route_request_with_exclusions(request, active.clone(), &excluded)
                        .await?,
                ));
            }
    // Non-rate-limit failure: the body was drained above so the
    // connection can be reused; fall back to the buffered path
    // which has full multi-provider failover.
    // Context-length errors: attempt truncation before falling back.
    if self.is_context_length_error(status_code, &body_text) {
        let mut truncated_request = request.clone();
        match self.context_manager.handle_context_error(
            &mut truncated_request,
            0,
            Some(&body_text),
        ) {
            Ok(result) => {
                info!(
                    provider = %provider_model.provider,
                    model = %provider_model.model,
                    original_tokens = result.original_tokens,
                    final_tokens = result.final_tokens,
                    messages_removed = result.messages_removed,
                    "Context-length error on streaming pass-through, truncated request and retrying via buffered path"
                );
                drop(concurrency_permit);
                return Ok(StreamingResponse::Buffered(
                    self.route_request(&truncated_request, active.clone()).await?,
                ));
            }
            Err(e) => {
                warn!(
                    provider = %provider_model.provider,
                    model = %provider_model.model,
                    error = %e,
                    "Context-length error on streaming pass-through but truncation cannot continue"
                );
            }
        }
    }
    warn!(provider = %provider_model.provider, status = status_code, "Provider returned non-success status (streaming), falling back to buffered path with full failover");
    drop(concurrency_permit);
    return Ok(StreamingResponse::Buffered(
        self.route_request(request, active.clone()).await?,
    ));
}

        // Success — hand the live streaming body and permit to the caller.
        // The handler keeps both alive until the relay finishes or is dropped.
        Ok(StreamingResponse::PassThrough {
            byte_stream: response,
            provider: provider_model.provider.clone(),
            model: provider_model.model.clone(),
            compression,
            concurrency_permit,
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
        // Fine-tuning dispatch is not model-specific, so the slot wait resolves
        // from the provider's non-thinking default.
        let _concurrency_permit = self.acquire_provider_slot_or_reject(&provider, "").await?;
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
        let _concurrency_permit = self
            .acquire_provider_slot_or_reject(&target.provider, model_name)
            .await?;
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
    use std::time::Duration as StdDuration;
    use wiremock::{Mock, MockServer, ResponseTemplate};

    #[tokio::test]
    async fn provider_concurrency_limit_serializes_and_releases_waiters() {
        let router = Arc::new(Router::new(
            Arc::new(RwLock::new(create_test_config())),
            test_metrics(),
        ));

        let first = router.acquire_provider_concurrency("electron", 1).await;
        let waiting_router = Arc::clone(&router);
        let mut second = tokio::spawn(async move {
            waiting_router
                .acquire_provider_concurrency("electron", 1)
                .await
        });

        assert!(
            tokio::time::timeout(StdDuration::from_millis(50), &mut second)
                .await
                .is_err()
        );
        drop(first);
        let second_permit = tokio::time::timeout(StdDuration::from_secs(1), second)
            .await
            .expect("waiting request should acquire after release")
            .expect("waiting task should complete");
        drop(second_permit);
    }

    #[tokio::test]
    async fn provider_slot_wait_times_out_instead_of_waiting_forever() {
        let router = Router::new(Arc::new(RwLock::new(create_test_config())), test_metrics());

        let _held = router.acquire_provider_concurrency("electron", 1).await;

        // A saturated provider must yield `None` once the wait window elapses
        // rather than parking the caller indefinitely.
        let denied = router
            .acquire_provider_concurrency_within("electron", 1, StdDuration::from_millis(50))
            .await;

        assert!(
            denied.is_none(),
            "saturated provider must not hand out a permit"
        );
    }

    #[tokio::test]
    async fn saturated_provider_is_rejected_with_503() {
        let router = Router::new(Arc::new(RwLock::new(create_test_config())), test_metrics());
        let mut provider_cfg =
            test_provider("electron", "http://localhost:1/v1".to_string());
        // Saturate the single slot, then force a short wait window so the
        // rejection path runs without a long test delay.
        provider_cfg.max_connections = 1;
        provider_cfg.total_timeout_seconds = Some(1);

        let _held = router
            .acquire_provider_slot_or_reject(&provider_cfg, "gpt-4")
            .await
            .expect("first request should get the only slot");

        let error = router
            .acquire_provider_slot_or_reject(&provider_cfg, "gpt-4")
            .await
            .expect_err("second request must be rejected, not queued forever");

        match error {
            GatewayError::Provider {
                status_code,
                provider,
                message,
            } => {
                assert_eq!(status_code, Some(503));
                assert_eq!(provider, "electron");
                assert!(
                    message.contains("max_connections"),
                    "message should name the knob to change, got: {message}"
                );
            }
            other => panic!("expected a provider 503, got {other:?}"),
        }
    }

    #[tokio::test]
    async fn provider_slot_wait_is_derived_from_the_provider_total_timeout() {
        let mut provider_cfg =
            test_provider("electron", "http://localhost:1/v1".to_string());

        provider_cfg.total_timeout_seconds = Some(800);
        assert_eq!(
            provider_slot_wait(&provider_cfg, "gpt-4"),
            StdDuration::from_secs(800)
        );

        // A zero configuration must still produce a blocking wait, not a spin.
        provider_cfg.total_timeout_seconds = Some(0);
        assert_eq!(
            provider_slot_wait(&provider_cfg, "gpt-4"),
            StdDuration::from_secs(1)
        );
    }

    #[tokio::test]
    async fn provider_concurrency_limits_are_independent_per_provider() {
        let router = Router::new(Arc::new(RwLock::new(create_test_config())), test_metrics());

        let _electron = router.acquire_provider_concurrency("electron", 1).await;
        tokio::time::timeout(
            StdDuration::from_millis(50),
            router.acquire_provider_concurrency("other", 1),
        )
        .await
        .expect("one provider must not block a different provider");
    }

    #[tokio::test]
    async fn provider_concurrency_limit_increase_wakes_waiters() {
        let router = Arc::new(Router::new(
            Arc::new(RwLock::new(create_test_config())),
            test_metrics(),
        ));

        let _first = router.acquire_provider_concurrency("electron", 1).await;
        let waiting_router = Arc::clone(&router);
        let mut second = tokio::spawn(async move {
            waiting_router
                .acquire_provider_concurrency("electron", 1)
                .await
        });
        assert!(
            tokio::time::timeout(StdDuration::from_millis(50), &mut second)
                .await
                .is_err()
        );

        let limiter = router.provider_concurrency_limiter("electron", 2);
        let second_permit = tokio::time::timeout(StdDuration::from_secs(1), second)
            .await
            .expect("increased limit should wake a waiting request")
            .expect("waiting task should complete");
        assert_eq!(limiter.state.lock().unwrap().limit, 2);
        drop(second_permit);
    }

    #[tokio::test]
    async fn provider_concurrency_limit_decrease_applies_after_in_flight_drains() {
        let router = Arc::new(Router::new(
            Arc::new(RwLock::new(create_test_config())),
            test_metrics(),
        ));

        let first = router.acquire_provider_concurrency("electron", 2).await;
        let second = router.acquire_provider_concurrency("electron", 2).await;
        let waiting_router = Arc::clone(&router);
        let mut third = tokio::spawn(async move {
            waiting_router
                .acquire_provider_concurrency("electron", 1)
                .await
        });
        assert!(
            tokio::time::timeout(StdDuration::from_millis(50), &mut third)
                .await
                .is_err()
        );

        drop(first);
        assert!(
            tokio::time::timeout(StdDuration::from_millis(50), &mut third)
                .await
                .is_err()
        );
        drop(second);
        let third_permit = tokio::time::timeout(StdDuration::from_secs(1), third)
            .await
            .expect("lower limit should apply once old requests drain")
            .expect("waiting task should complete");
        drop(third_permit);
    }

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

struct ImageRejectThenSuccessClient {
    calls: std::sync::atomic::AtomicUsize,
}

#[async_trait::async_trait]
impl ProviderClient for ImageRejectThenSuccessClient {
    async fn chat_completion(
        &self,
        request: OpenAIRequest,
    ) -> Result<ProviderResponse, GatewayError> {
        let call = self.calls.fetch_add(1, std::sync::atomic::Ordering::SeqCst);
        let carries_images = request.messages.iter().any(|msg| {
            msg.content
                .as_array()
                .map(|parts| {
                    parts.iter().any(|part| {
                        matches!(
                            part.get("type").and_then(|v| v.as_str()),
                            Some("image_url") | Some("image") | Some("input_image")
                        )
                    })
                })
                .unwrap_or(false)
        });
        if call == 0 {
            assert!(carries_images, "first call should carry image parts");
            return Err(GatewayError::Provider {
                provider: "adapter".to_string(),
                message: r#"Upstream HTTP 400: {"error":{"message":"This model does not support image inputs."}}"#.to_string(),
                status_code: Some(400),
            });
        }
        assert!(!carries_images, "retry must not carry image parts");
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
async fn buffered_adapter_image_error_strips_and_retries() {
    let router = Router::new(Arc::new(RwLock::new(create_test_config())), test_metrics());
    let client = ImageRejectThenSuccessClient {
        calls: std::sync::atomic::AtomicUsize::new(0),
    };
    let mut request = compression_request(false);
    request.model = "adapter-model".to_string();
    request.messages = vec![
        Message {
            role: "user".to_string(),
            content: serde_json::json!([
                {"type": "text", "text": "what is in this picture?"},
                {"type": "image_url", "image_url": {"url": "data:image/png;base64,aGVsbG8="}}
            ]),
            extra: Default::default(),
        },
    ];

    let response = router
        .dispatch_buffered_with_context_retry(&client, request)
        .await
        .expect("shared wrapper should strip images and retry");
    assert_eq!(response.provider_name, "adapter");
    assert_eq!(client.calls.load(std::sync::atomic::Ordering::SeqCst), 2);
}

// ── Backstop: duplicate-compaction-trigger repair-and-retry (task 7.2) ──
//
// Mirrors the image-strip client pattern: a test ProviderClient that counts
// calls, rejects the first attempt with the observed Bedrock Mantle 400 body,
// and succeeds on the retry. Determinism comes from driving
// `dispatch_buffered_with_context_retry` directly rather than fighting the
// dispatch seam, which already de-dupes before the provider is reached.

/// Count compaction-trigger sites in a chat request's message content arrays.
/// Sufficient for these tests, whose duplicates live as content parts.
fn count_trigger_content_parts(request: &OpenAIRequest) -> usize {
    request
        .messages
        .iter()
        .filter_map(|msg| msg.content.as_array())
        .flatten()
        .filter(|part| {
            part.get("type").and_then(|v| v.as_str()) == Some("compaction_trigger")
        })
        .count()
}

struct DuplicateTriggerThenSuccessClient {
    calls: std::sync::atomic::AtomicUsize,
}

#[async_trait::async_trait]
impl ProviderClient for DuplicateTriggerThenSuccessClient {
    async fn chat_completion(
        &self,
        request: OpenAIRequest,
    ) -> Result<ProviderResponse, GatewayError> {
        let call = self.calls.fetch_add(1, std::sync::atomic::Ordering::SeqCst);
        if call == 0 {
            assert!(
                count_trigger_content_parts(&request) > 1,
                "first call should carry more than one compaction_trigger"
            );
            return Err(GatewayError::Provider {
                provider: "adapter".to_string(),
                message: "Only one 'compaction_trigger' item may be provided.".to_string(),
                status_code: Some(400),
            });
        }
        assert_eq!(
            count_trigger_content_parts(&request),
            1,
            "retry must carry exactly one compaction_trigger"
        );
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
async fn buffered_adapter_duplicate_trigger_normalizes_and_retries() {
    let router = Router::new(Arc::new(RwLock::new(create_test_config())), test_metrics());
    let client = DuplicateTriggerThenSuccessClient {
        calls: std::sync::atomic::AtomicUsize::new(0),
    };
    let mut request = compression_request(false);
    request.model = "adapter-model".to_string();
    // Two compaction-trigger content parts across two messages: the seam
    // re-run inside the retry arm keeps the last and removes the first.
    request.messages = vec![
        Message {
            role: "user".to_string(),
            content: serde_json::json!([{"type": "compaction_trigger"}]),
            extra: Default::default(),
        },
        Message {
            role: "user".to_string(),
            content: serde_json::json!([{"type": "compaction_trigger"}]),
            extra: Default::default(),
        },
    ];

    let response = router
        .dispatch_buffered_with_context_retry(&client, request)
        .await
        .expect("shared wrapper should normalize triggers and retry");
    assert_eq!(response.provider_name, "adapter");
    // Exactly two upstream calls: the rejected first and the repaired retry.
    assert_eq!(client.calls.load(std::sync::atomic::Ordering::SeqCst), 2);
}

struct UnrelatedRejectClient {
    calls: std::sync::atomic::AtomicUsize,
}

#[async_trait::async_trait]
impl ProviderClient for UnrelatedRejectClient {
    async fn chat_completion(
        &self,
        _request: OpenAIRequest,
    ) -> Result<ProviderResponse, GatewayError> {
        self.calls.fetch_add(1, std::sync::atomic::Ordering::SeqCst);
        Err(GatewayError::Provider {
            provider: "adapter".to_string(),
            message: "invalid request".to_string(),
            status_code: Some(400),
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
async fn buffered_adapter_unrelated_400_surfaces_without_extra_attempt() {
    let router = Router::new(Arc::new(RwLock::new(create_test_config())), test_metrics());
    let client = UnrelatedRejectClient {
        calls: std::sync::atomic::AtomicUsize::new(0),
    };
    let mut request = compression_request(false);
    request.model = "adapter-model".to_string();
    // Even with duplicate triggers present, an unrelated 400 must not trip the
    // duplicate-trigger arm — the guard requires the specific rejection phrase.
    request.messages = vec![
        Message {
            role: "user".to_string(),
            content: serde_json::json!([{"type": "compaction_trigger"}]),
            extra: Default::default(),
        },
        Message {
            role: "user".to_string(),
            content: serde_json::json!([{"type": "compaction_trigger"}]),
            extra: Default::default(),
        },
    ];

    let error = router
        .dispatch_buffered_with_context_retry(&client, request)
        .await
        .expect_err("unrelated 400 should surface, not be repaired");
    match error {
        GatewayError::Provider {
            status_code: Some(400),
            ..
        } => {}
        other => panic!("expected provider 400 to surface, got {other:?}"),
    }
    // Exactly one upstream call: no repair retry for an unrelated failure.
    assert_eq!(client.calls.load(std::sync::atomic::Ordering::SeqCst), 1);
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

    // ── Degenerate turns: reasoning with no answer and no tool call ──
    //
    // The failure these guard against: a provider (GLM in the field report)
    // finishes its thinking phase and stops. Handed back verbatim, the turn is
    // indistinguishable from a completed reply, so an agentic harness renders
    // the chain of thought and stops working mid-task.

    /// A response shaped like a provider turn: `content` plus message extras.
    fn turn(
        content: serde_json::Value,
        extra: serde_json::Map<String, serde_json::Value>,
    ) -> OpenAIResponse {
        OpenAIResponse {
            id: "chatcmpl-turn".to_string(),
            object: "chat.completion".to_string(),
            created: 1,
            model: "glm-4.6".to_string(),
            choices: vec![Choice {
                index: 0,
                message: Message {
                    role: "assistant".to_string(),
                    content,
                    extra,
                },
                finish_reason: Some("stop".to_string()),
                extra: Default::default(),
            }],
            usage: Usage::default(),
            extra: Default::default(),
        }
    }

    fn extras(pairs: &[(&str, serde_json::Value)]) -> serde_json::Map<String, serde_json::Value> {
        pairs
            .iter()
            .map(|(key, value)| ((*key).to_string(), value.clone()))
            .collect()
    }

    fn request_carrying_tools() -> OpenAIRequest {
        let mut extra = serde_json::Map::new();
        extra.insert(
            "tools".to_string(),
            serde_json::json!([{
                "type": "function",
                "function": {"name": "read_file", "parameters": {}}
            }]),
        );
        OpenAIRequest {
            model: "test-model".to_string(),
            messages: vec![],
            temperature: None,
            max_tokens: None,
            stream: false,
            extra,
        }
    }

    #[test]
    fn reasoning_only_turn_is_detected() {
        let thinking = extras(&[(
            "reasoning_content",
            serde_json::json!("I should read the file"),
        )]);

        assert!(Router::reasoning_only_turn(&turn(
            serde_json::Value::Null,
            thinking.clone()
        )));
        // Whitespace-only content is still not an answer.
        assert!(Router::reasoning_only_turn(&turn(
            serde_json::json!("\n  \n"),
            thinking.clone()
        )));
        // The `reasoning` spelling counts as well as `reasoning_content`.
        assert!(Router::reasoning_only_turn(&turn(
            serde_json::Value::Null,
            extras(&[("reasoning", serde_json::json!("thinking"))])
        )));

        // Real answer text means the turn completed.
        assert!(!Router::reasoning_only_turn(&turn(
            serde_json::json!("here you go"),
            thinking.clone()
        )));
        // A tool call is work, even with no answer text alongside it.
        let mut with_call = thinking;
        with_call.insert(
            "tool_calls".to_string(),
            serde_json::json!([{
                "id": "call_1",
                "type": "function",
                "function": {"name": "read_file", "arguments": "{}"}
            }]),
        );
        assert!(!Router::reasoning_only_turn(&turn(
            serde_json::Value::Null,
            with_call
        )));
        // No reasoning either: a plain hollow response, handled by
        // `response_has_content` rather than here.
        assert!(!Router::reasoning_only_turn(&turn(
            serde_json::Value::Null,
            extras(&[])
        )));
    }

    #[test]
    fn reasoning_states_intent_matches_announced_actions() {
        // The field-report shape: thinking ends on "let me ...".
        let thinking = extras(&[(
            "reasoning_content",
            serde_json::json!("Let me read the config file first."),
        )]);
        assert!(Router::reasoning_states_intent(&turn(
            serde_json::Value::Null,
            thinking
        )));

        // First-person variants.
        for text in [
            "I'll inspect the schema.",
            "I will query the endpoint.",
            "I need to check the logs.",
            "First, I should look at the diff.",
            "The next step is to open the file.",
        ] {
            let thinking = extras(&[("reasoning_content", serde_json::json!(text))]);
            assert!(
                Router::reasoning_states_intent(&turn(serde_json::Value::Null, thinking)),
                "must match: {text}"
            );
        }

        // Concluded thinking without an announced action: no nudge.
        let done = extras(&[(
            "reasoning_content",
            serde_json::json!("The analysis is complete and consistent."),
        )]);
        assert!(!Router::reasoning_states_intent(&turn(
            serde_json::Value::Null,
            done
        )));
        // No reasoning carrier at all.
        assert!(!Router::reasoning_states_intent(&turn(
            serde_json::json!("answer"),
            extras(&[])
        )));
    }

    /// Serves a reasoning-only turn (whose thinking announces an action)
    /// for the original request and a real tool call once the gateway's
    /// continuation nudge arrives.
    struct ReasoningThenToolCall;
    impl wiremock::Respond for ReasoningThenToolCall {
        fn respond(&self, request: &wiremock::Request) -> wiremock::ResponseTemplate {
            let body = String::from_utf8_lossy(request.body.as_slice());
            if body.contains("private reasoning") {
                ResponseTemplate::new(200).set_body_json(serde_json::json!({
                    "id": "chatcmpl-nudged",
                    "object": "chat.completion",
                    "created": 2,
                    "model": "upstream-model",
                    "choices": [{
                        "index": 0,
                        "message": {
                            "role": "assistant",
                            "content": null,
                            "tool_calls": [{
                                "id": "call_1",
                                "type": "function",
                                "function": {
                                    "name": "read_file",
                                    "arguments": "{\"path\": \"config.yaml\"}"
                                }
                            }]
                        },
                        "finish_reason": "tool_calls"
                    }],
                    "usage": {"prompt_tokens": 5, "completion_tokens": 7, "total_tokens": 12}
                }))
            } else {
                ResponseTemplate::new(200).set_body_json(serde_json::json!({
                    "id": "chatcmpl-stalled",
                    "object": "chat.completion",
                    "created": 1,
                    "model": "upstream-model",
                    "choices": [{
                        "index": 0,
                        "message": {
                            "role": "assistant",
                            "content": null,
                            "reasoning_content": "Let me read the config file to check the setting."
                        },
                        "finish_reason": "stop"
                    }],
                    "usage": {"prompt_tokens": 5, "completion_tokens": 4, "total_tokens": 9}
                }))
            }
        }
    }

    #[tokio::test]
    async fn reasoning_intent_nudge_revives_stalled_turn() {
        use wiremock::matchers::{method, path};

        let server = MockServer::start().await;
        Mock::given(method("POST"))
            .and(path("/v1/chat/completions"))
            .respond_with(ReasoningThenToolCall)
            .mount(&server)
            .await;

        let mut config = create_test_config();
        config.retry.max_retries_per_provider = 0;
        config.providers = vec![test_provider("first", server.uri())];
        config.model_groups = vec![test_group(vec![test_model("first", 1)])];
        let router = Router::new(Arc::new(RwLock::new(config)), test_metrics());

        // A tool-using request so the reasoning-only turn is judged as a
        // stalled agent turn rather than promoted to content.
        let mut request = compression_request(false);
        request.extra.insert(
            "tools".to_string(),
            serde_json::json!([{
                "type": "function",
                "function": {
                    "name": "read_file",
                    "description": "Read a file",
                    "parameters": {"type": "object", "properties": {}}
                }
            }]),
        );

        let response = router.route_request(&request, None).await.unwrap();

        // The nudge revived the turn into a real tool call.
        let tool_calls = response
            .choices
            .first()
            .unwrap()
            .message
            .extra
            .get("tool_calls")
            .and_then(|v| v.as_array())
            .expect("revived turn must carry the tool call");
        assert_eq!(
            tool_calls[0]
                .get("function")
                .and_then(|f| f.get("name"))
                .and_then(|n| n.as_str()),
            Some("read_file")
        );

        // Exactly two upstream hits: the stalled attempt + the nudge.
        let received = server.received_requests().await.unwrap();
        assert_eq!(received.len(), 2, "nudge must retry the same provider once");
        // The nudged request echoed the stalled assistant turn and ended
        // with the user-role continuation instruction.
        let nudged_body = String::from_utf8_lossy(&received[1].body);
        let nudged: serde_json::Value = serde_json::from_str(&nudged_body).unwrap();
        let messages = nudged["messages"].as_array().unwrap();
        let last = messages.last().unwrap();
        assert_eq!(last["role"], "user");
        assert!(
            last["content"]
                .as_str()
                .unwrap()
                .contains("private reasoning")
        );
        let prev = &messages[messages.len() - 2];
        assert_eq!(prev["role"], "assistant");
        // The stalled thinking is echoed back as assistant content (the
        // reasoning-compat strip would remove the `reasoning_content`
        // carrier from the outgoing request).
        assert!(prev["content"]
            .as_str()
            .unwrap()
            .contains("Let me read"));
    }

    #[test]
    fn split_think_tags_moves_inline_thinking_into_reasoning() {
        let mut response = turn(
            serde_json::json!("<think>weighing options</think>the answer"),
            extras(&[]),
        );

        assert!(Router::split_think_tags(&mut response));
        assert_eq!(response.choices[0].message.content_as_text(), "the answer");
        assert_eq!(
            response.choices[0].message.extra.get("reasoning_content"),
            Some(&serde_json::json!("weighing options"))
        );
    }

    /// A think block with nothing after it is the degenerate turn in disguise:
    /// once split out, the emptiness check can finally see it.
    #[test]
    fn split_think_tags_exposes_a_thinking_only_turn() {
        let mut response = turn(
            serde_json::json!("<think>just thinking</think>"),
            extras(&[]),
        );

        assert!(Router::split_think_tags(&mut response));
        assert!(Router::reasoning_only_turn(&response));
    }

    /// Cut off mid-thought: no closing tag, so the whole remainder is reasoning.
    #[test]
    fn split_think_tags_handles_an_unclosed_block() {
        let mut response = turn(serde_json::json!("<think>cut off"), extras(&[]));

        assert!(Router::split_think_tags(&mut response));
        assert!(response.choices[0].message.content.is_null());
        assert_eq!(
            response.choices[0].message.extra.get("reasoning_content"),
            Some(&serde_json::json!("cut off"))
        );
    }

    /// Only a leading block counts as thinking. Content that merely mentions the
    /// tag — documentation or a code sample — must survive untouched.
    #[test]
    fn split_think_tags_ignores_mid_content_mentions() {
        let body = "Use `<think>` to open a reasoning block.";
        let mut response = turn(serde_json::json!(body), extras(&[]));

        assert!(!Router::split_think_tags(&mut response));
        assert_eq!(response.choices[0].message.content_as_text(), body);
    }

    /// Regression: a provider that omits the tool-call `id` used to lose the
    /// entire turn to validation and fail over, destroying a usable tool call.
    #[test]
    fn repair_tool_calls_mints_missing_id_and_aligns_finish_reason() {
        let mut response = turn(
            serde_json::Value::Null,
            extras(&[(
                "tool_calls",
                serde_json::json!([{
                    "id": "",
                    "function": {"name": "read_file", "arguments": "{\"path\":\"a.rs\"}"}
                }]),
            )]),
        );
        assert!(
            !Router::response_has_content(&response),
            "an empty id is rejected before repair"
        );

        assert!(Router::repair_tool_calls(&mut response));

        let call = &response.choices[0].message.extra["tool_calls"][0];
        assert!(call["id"].as_str().unwrap().starts_with("call_"));
        assert_eq!(call["type"], "function");
        assert_eq!(
            response.choices[0].finish_reason.as_deref(),
            Some("tool_calls"),
            "a tool call present means finish_reason must say so"
        );
        assert!(Router::response_has_content(&response));
    }

    /// `length` is what truncation detection reads, so a tool call must not
    /// overwrite it.
    #[test]
    fn repair_tool_calls_preserves_truncation_finish_reason() {
        let mut response = turn(
            serde_json::Value::Null,
            extras(&[(
                "tool_calls",
                serde_json::json!([{
                    "id": "call_1",
                    "type": "function",
                    "function": {"name": "read_file", "arguments": "{"}
                }]),
            )]),
        );
        response.choices[0].finish_reason = Some("length".to_string());

        Router::repair_tool_calls(&mut response);

        assert_eq!(response.choices[0].finish_reason.as_deref(), Some("length"));
    }

    /// GLM puts XML tool calls in `reasoning_content`. Reading only `reasoning`
    /// meant they were never recovered, so the turn reached the client as a
    /// thinking block with no tool to run.
    #[test]
    fn translate_xml_tool_calls_reads_the_reasoning_content_carrier() {
        let mut response = turn(
            serde_json::Value::Null,
            extras(&[(
                "reasoning_content",
                serde_json::json!("<read_file><path>src/main.rs</path></read_file>"),
            )]),
        );

        assert!(Router::translate_xml_tool_calls(
            &mut response,
            &request_carrying_tools()
        ));

        let calls = response.choices[0].message.extra["tool_calls"]
            .as_array()
            .unwrap();
        assert_eq!(calls.len(), 1);
        assert_eq!(calls[0]["function"]["name"], "read_file");
        assert_eq!(
            response.choices[0].finish_reason.as_deref(),
            Some("tool_calls")
        );
        // The consumed carrier is cleared so stale XML is not shipped as thinking.
        assert!(!response.choices[0]
            .message
            .extra
            .contains_key("reasoning_content"));
    }

    /// Reassembly must leave reasoning in its own carrier. Copying it into
    /// `content` was what made a thinking-only turn look like a real answer.
    #[test]
    fn reassemble_sse_response_keeps_reasoning_out_of_content() {
        let body = concat!(
            "data: {\"id\":\"c\",\"object\":\"chat.completion.chunk\",\"created\":1,\"model\":\"glm\",\"choices\":[{\"index\":0,\"delta\":{\"reasoning_content\":\"thinking hard\"}}]}\n\n",
            "data: {\"id\":\"c\",\"object\":\"chat.completion.chunk\",\"created\":1,\"model\":\"glm\",\"choices\":[{\"index\":0,\"delta\":{},\"finish_reason\":\"stop\"}]}\n\n",
            "data: [DONE]\n\n"
        );

        let response = Router::reassemble_sse_response(body).unwrap();

        assert!(Router::content_is_empty(&response.choices[0].message.content));
        assert_eq!(
            response.choices[0].message.extra.get("reasoning_content"),
            Some(&serde_json::json!("thinking hard"))
        );
        assert!(Router::reasoning_only_turn(&response));
        // And it must never be cached, or the dead end replays for every
        // identical prefix.
        assert!(!Router::should_cache_response(&response));
    }

    /// Providers that omit `index` used to have every parallel tool call folded
    /// into slot 0, concatenating unrelated arguments into one invalid blob and
    /// losing all but the first call.
    #[test]
    fn reassemble_sse_response_separates_unindexed_parallel_tool_calls() {
        let body = concat!(
            "data: {\"choices\":[{\"index\":0,\"delta\":{\"tool_calls\":[{\"id\":\"call_a\",\"function\":{\"name\":\"read_file\",\"arguments\":\"\"}}]}}]}\n\n",
            "data: {\"choices\":[{\"index\":0,\"delta\":{\"tool_calls\":[{\"function\":{\"arguments\":\"{\\\"path\\\":\\\"a.rs\\\"}\"}}]}}]}\n\n",
            "data: {\"choices\":[{\"index\":0,\"delta\":{\"tool_calls\":[{\"id\":\"call_b\",\"function\":{\"name\":\"read_file\",\"arguments\":\"\"}}]}}]}\n\n",
            "data: {\"choices\":[{\"index\":0,\"delta\":{\"tool_calls\":[{\"function\":{\"arguments\":\"{\\\"path\\\":\\\"b.rs\\\"}\"}}]}}]}\n\n",
            "data: [DONE]\n\n"
        );

        let response = Router::reassemble_sse_response(body).unwrap();

        let calls = response.choices[0].message.extra["tool_calls"]
            .as_array()
            .unwrap();
        assert_eq!(calls.len(), 2, "each call keeps its own slot");
        assert_eq!(calls[0]["id"], "call_a");
        assert_eq!(calls[0]["function"]["arguments"], "{\"path\":\"a.rs\"}");
        assert_eq!(calls[1]["id"], "call_b");
        assert_eq!(calls[1]["function"]["arguments"], "{\"path\":\"b.rs\"}");
        assert_eq!(
            response.choices[0].finish_reason.as_deref(),
            Some("tool_calls")
        );
    }

    /// A learned degenerate-stream combo is what routes the next tools-bearing
    /// request to the buffered path, where failover can actually run.
    #[test]
    fn degenerate_stream_combos_are_learned_per_provider_model() {
        let router = Router::new(Arc::new(RwLock::new(create_test_config())), test_metrics());

        assert!(!router.is_degenerate_stream_combo("zai", "glm-4.6"));

        router.mark_degenerate_stream_combo("zai", "glm-4.6");

        assert!(router.is_degenerate_stream_combo("zai", "glm-4.6"));
        // Learning is scoped to the exact combo, not the provider.
        assert!(!router.is_degenerate_stream_combo("zai", "glm-4.5"));
        assert!(!router.is_degenerate_stream_combo("other", "glm-4.6"));
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
    fn friendly_failure_reason_surfaces_error_in_200_detail() {
        let message = "Error in 200 response: Model overloaded, please retry";
        assert_eq!(
            Router::friendly_failure_reason(Some(200), message),
            "Provider error (in HTTP 200): Model overloaded, please retry"
        );
    }

    #[test]
    fn friendly_failure_reason_surfaces_parse_failure_in_200() {
        let message = "Failed to parse response: not JSON or SSE";
        assert_eq!(
            Router::friendly_failure_reason(Some(200), message),
            "Provider sent an unparseable response (HTTP 200): not JSON or SSE"
        );
    }

    #[test]
    fn friendly_failure_reason_generic_2xx_without_detail() {
        assert_eq!(
            Router::friendly_failure_reason(Some(204), "gateway dropped body"),
            "Provider returned an error inside a HTTP 204 response"
        );
        // A JSON error envelope embedded in the text still wins for 2xx.
        let message = r#"HTTP 200: {"error":{"message":"insufficient credits"}}"#;
        assert_eq!(
            Router::friendly_failure_reason(Some(200), message),
            "Provider error (in HTTP 200): insufficient credits"
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
codex_search: None,
cache_aware_routing: Default::default(),
reasoning_compat: Default::default(),
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
cache_min_tokens: None,
cache_support: None,
cost_per_million_cache_read_input_tokens: None,
cost_per_million_cache_creation_input_tokens: None,
cost_per_million_reasoning_tokens: None,
reasoning_family: None,
reasoning_parameter: None,
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

    // ------------------------------------------------------------------
    // Task 9 preservation — non-Bedrock pass-through keeps every trigger
    // (clause 3.3). On the live path a trigger-bearing request to an
    // `openai`/`openai_compatible` provider never touches the Bedrock seam
    // (`normalize_mantle_compaction_triggers` is only reached from the Bedrock
    // dispatch). The provider-specific sanitizer is the only outgoing transform,
    // and its `openai` arm is a no-op, so both triggers must survive.
    // ------------------------------------------------------------------
    #[test]
    fn openai_sanitize_preserves_all_compaction_triggers() {
        let mut outgoing = OpenAIRequest {
            model: "gpt-4o".to_string(),
            messages: vec![
                Message {
                    role: "user".to_string(),
                    content: serde_json::json!([{"type": "compaction_trigger"}]),
                    extra: Default::default(),
                },
                Message {
                    role: "assistant".to_string(),
                    content: serde_json::json!([
                        {"type": "text", "text": "hi"},
                        {"type": "compaction_trigger"}
                    ]),
                    extra: Default::default(),
                },
            ],
            stream: false,
            temperature: None,
            max_tokens: None,
            extra: Default::default(),
        };
        let before = serde_json::to_value(&outgoing).unwrap();

        // The `openai` arm removes no fields and touches no triggers.
        let removed = Router::sanitize_request_for_provider(&mut outgoing, "openai");
        assert_eq!(removed, 0, "openai sanitize must remove nothing");

        // Count triggers across content parts — both must survive intact.
        let trigger_count: usize = outgoing
            .messages
            .iter()
            .filter_map(|m| m.content.as_array())
            .flatten()
            .filter(|p| {
                p.get("type").and_then(serde_json::Value::as_str) == Some("compaction_trigger")
            })
            .count();
        assert_eq!(trigger_count, 2, "both compaction triggers must pass through untouched");
        assert_eq!(
            serde_json::to_value(&outgoing).unwrap(),
            before,
            "openai pass-through must leave the request byte-identical"
        );
    }

    // ------------------------------------------------------------------
    // Task 9 preservation — streaming transport decision for Bedrock
    // (clause 3.6). A streaming request to a `bedrock` provider must take the
    // buffered gate (task 8's structural `is_bedrock` early return), so it can
    // NEVER resolve to a `PassThrough` relay. The companion non-Bedrock case
    // (returns `PassThrough`) is `streaming_provider_receives_compressed_body_before_response`.
    //
    // The Bedrock buffered route dispatches through the real provider (the
    // Mantle base URL is derived from the region, not the config, so it cannot
    // be redirected to wiremock). The route therefore errors on the network,
    // which `route_request_streaming` surfaces as `Err` — but the decisive fact
    // is that the outcome is never `Ok(PassThrough { .. })`. That distinguishes
    // the buffered gate from the pass-through path structurally.
    // ------------------------------------------------------------------
    #[tokio::test]
    async fn bedrock_streaming_request_takes_buffered_path() {
        let mut config = create_test_config();
        config.retry.max_retries_per_provider = 0;
        let mut provider = test_provider("bedrock-provider", "https://unused.example".to_string());
        provider.provider_type = "bedrock".to_string();
        provider.region = Some("us-east-1".to_string());
        // A short total timeout bounds the buffered network attempt.
        provider.total_timeout_seconds = Some(2);
        provider.ttfb_timeout_seconds = Some(2);
        config.providers = vec![provider];
        config.model_groups = vec![test_group(vec![test_model_named(
            "bedrock-provider",
            "openai.gpt-oss-120b",
            1,
        )])];
        let router = Router::new(Arc::new(RwLock::new(config)), test_metrics());

        let request = OpenAIRequest {
            model: "test-group".to_string(),
            messages: vec![Message {
                role: "user".to_string(),
                content: serde_json::json!("hello"),
                extra: Default::default(),
            }],
            stream: true,
            temperature: None,
            max_tokens: None,
            extra: Default::default(),
        };

        let outcome = router.route_request_streaming(&request, None).await;
        // The buffered gate fired: the result is either a Buffered response (if
        // the upstream somehow answered) or an Err from the buffered route — but
        // NEVER a pass-through relay.
        match outcome {
            Ok(StreamingResponse::Buffered(_)) => {}
            Ok(StreamingResponse::PassThrough { .. }) => {
                panic!("Bedrock streaming must not use the pass-through relay")
            }
            Err(_) => {
                // Buffered route attempted and failed on the network — the
                // buffered gate was still taken (a pass-through would have
                // returned Ok(PassThrough) before any buffered dispatch).
            }
        }
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
                cache_min_tokens: None,
                cache_support: None,
                cost_per_million_cache_read_input_tokens: None,
                cost_per_million_cache_creation_input_tokens: None,
                cost_per_million_reasoning_tokens: None,
                reasoning_family: None,
                reasoning_parameter: None,
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
                    cache_min_tokens: None,
                    cache_support: None,
                    cost_per_million_cache_read_input_tokens: None,
                    cost_per_million_cache_creation_input_tokens: None,
                    cost_per_million_reasoning_tokens: None,
                    reasoning_family: None,
                    reasoning_parameter: None,
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
                    cache_min_tokens: None,
                    cache_support: None,
                    cost_per_million_cache_read_input_tokens: None,
                    cost_per_million_cache_creation_input_tokens: None,
                    cost_per_million_reasoning_tokens: None,
                    reasoning_family: None,
                    reasoning_parameter: None,
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
                    cache_min_tokens: None,
                    cache_support: None,
                    cost_per_million_cache_read_input_tokens: None,
                    cost_per_million_cache_creation_input_tokens: None,
                    cost_per_million_reasoning_tokens: None,
                    reasoning_family: None,
                    reasoning_parameter: None,
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
                    cache_min_tokens: None,
                    cache_support: None,
                    cost_per_million_cache_read_input_tokens: None,
                    cost_per_million_cache_creation_input_tokens: None,
                    cost_per_million_reasoning_tokens: None,
                    reasoning_family: None,
                    reasoning_parameter: None,
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
                    cache_min_tokens: None,
                    cache_support: None,
                    cost_per_million_cache_read_input_tokens: None,
                    cost_per_million_cache_creation_input_tokens: None,
                    cost_per_million_reasoning_tokens: None,
                    reasoning_family: None,
                    reasoning_parameter: None,
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
                    cache_min_tokens: None,
                    cache_support: None,
                    cost_per_million_cache_read_input_tokens: None,
                    cost_per_million_cache_creation_input_tokens: None,
                    cost_per_million_reasoning_tokens: None,
                    reasoning_family: None,
                    reasoning_parameter: None,
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
                    cache_min_tokens: None,
                    cache_support: None,
                    cost_per_million_cache_read_input_tokens: None,
                    cost_per_million_cache_creation_input_tokens: None,
                    cost_per_million_reasoning_tokens: None,
                    reasoning_family: None,
                    reasoning_parameter: None,
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
                    cache_min_tokens: None,
                    cache_support: None,
                    cost_per_million_cache_read_input_tokens: None,
                    cost_per_million_cache_creation_input_tokens: None,
                    cost_per_million_reasoning_tokens: None,
                    reasoning_family: None,
                    reasoning_parameter: None,
                },
                ProviderModel {
                    provider: "provider-3".to_string(),
                    model: "gpt-4-turbo".to_string(),
                    cost_per_million_input_tokens: 10.0,
                    cost_per_million_output_tokens: 30.0,
                    // Best (lowest) priority — the dated models must still
                    // outrank it: version date is the DOMINANT ordering key
                    // when version_fallback_enabled (decided behavior).
                    priority: 1,
                    structured_output_passthrough: None,
                    tier: None,
                    context_window: 0,
                    specializations: vec![],
                    cache_min_tokens: None,
                    cache_support: None,
                    cost_per_million_cache_read_input_tokens: None,
                    cost_per_million_cache_creation_input_tokens: None,
                    cost_per_million_reasoning_tokens: None,
                    reasoning_family: None,
                    reasoning_parameter: None,
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
        // Should be sorted by version date descending (newest first), even
        // over a lower-priority undated model.
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
            cache_min_tokens: None,
            cache_support: None,
            cost_per_million_cache_read_input_tokens: None,
            cost_per_million_cache_creation_input_tokens: None,
            cost_per_million_reasoning_tokens: None,
            reasoning_family: None,
            reasoning_parameter: None,
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

#[test]
fn test_strip_image_content_if_unsupported_inserts_placeholder_for_image_only_message() {
    let mut request = OpenAIRequest {
        model: "no-vision".to_string(),
        messages: vec![Message {
            role: "user".to_string(),
            content: serde_json::json!([
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
    assert!(parts[0]["text"].as_str().unwrap().contains("image"));
}

    #[test]
    fn test_strip_image_content_if_unsupported_removes_variant_image_part_types() {
        let mut request = OpenAIRequest {
            model: "no-vision".to_string(),
            messages: vec![Message {
                role: "user".to_string(),
                content: serde_json::json!([
                    {"type": "text", "text": "describe this"},
                    {"type": "image", "source": {"type": "base64"}},
                    {"type": "input_image", "image_url": "https://x.example/p.png"},
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
        assert_eq!(removed, 2);
        let parts = request.messages[0].content.as_array().unwrap();
        assert_eq!(parts.len(), 1);
        assert_eq!(parts[0]["type"], serde_json::json!("text"));
    }

    #[test]
    fn test_strip_image_content_if_unsupported_removes_nested_tool_result_images() {
        // Images nested inside a tool_result part's own content array
        // must be stripped too — a top-level-only pass leaves them
        // behind and the provider rejects the retry again.
        let mut request = OpenAIRequest {
            model: "no-vision".to_string(),
            messages: vec![Message {
                role: "user".to_string(),
                content: serde_json::json!([
                    {"type": "text", "text": "screenshot attached"},
                    {"type": "tool_result", "tool_use_id": "call_1", "content": [
                        {"type": "text", "text": "tool output follows"},
                        {"type": "image_url", "image_url": {"url": "data:image/png;base64,aGVsbG8="}}
                    ]},
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
        assert_eq!(removed, 1, "nested image must be counted and removed");
        let parts = request.messages[0].content.as_array().unwrap();
        assert_eq!(parts.len(), 2, "text + tool_result parts survive");
        let nested = parts[1]["content"].as_array().unwrap();
        assert_eq!(nested.len(), 1, "nested text part survives");
        assert_eq!(nested[0]["type"], serde_json::json!("text"));
    }

    #[test]
    fn test_strip_image_content_if_unsupported_nested_only_image_gets_placeholder() {
        // A tool_result whose nested content is ONLY an image gets a text
        // placeholder so the provider never sees an empty content array.
        let mut request = OpenAIRequest {
            model: "no-vision".to_string(),
            messages: vec![Message {
                role: "user".to_string(),
                content: serde_json::json!([
                    {"type": "tool_result", "tool_use_id": "call_1", "content": [
                        {"type": "image_url", "image_url": {"url": "data:image/png;base64,aGVsbG8="}}
                    ]},
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
        let nested = parts[0]["content"].as_array().unwrap();
        assert_eq!(nested.len(), 1, "placeholder inserted into nested array");
        assert_eq!(nested[0]["type"], serde_json::json!("text"));
    }

    #[test]
    fn test_is_unsupported_image_error_matches_provider_phrasings() {
        assert!(Router::is_unsupported_image_error(
            400,
            "This model does not support image inputs."
        ));
        assert!(Router::is_unsupported_image_error(
            400,
            r#"{"error":{"message":"Invalid request: image content is not supported for this model"}}"#
        ));
        assert!(Router::is_unsupported_image_error(
            422,
            "model does not support vision"
        ));
        assert!(!Router::is_unsupported_image_error(
            400,
            "invalid model identifier"
        ));
        assert!(!Router::is_unsupported_image_error(500, "image inputs"));
        assert!(!Router::is_unsupported_image_error(
            200,
            "This model does not support image inputs."
        ));
    }

    #[test]
    fn test_is_unsupported_image_phrasing_detects_200_envelope_rejections() {
        // The phrase check is status-independent: used for real 4xx
        // rejections AND error-inside-HTTP-200 envelopes.
        assert!(Router::is_unsupported_image_phrasing(
            "This model does not support image inputs."
        ));
        assert!(Router::is_unsupported_image_phrasing(
            r#"{"error":{"message":"This model does not support image inputs."}}"#
        ));
assert!(!Router::is_unsupported_image_phrasing(
"invalid model identifier"
));
}

// --- Reasoning-compat conversation-model affinity (Task 6) ---

fn affinity_router(reasoning_compat: crate::reasoning_compat::ReasoningCompatConfig) -> Router {
let mut config = create_test_config();
config.reasoning_compat = reasoning_compat;
Router::new(Arc::new(RwLock::new(config)), test_metrics())
}

fn affinity_conversation() -> OpenAIRequest {
OpenAIRequest {
model: "test-group".to_string(),
messages: vec![
Message {
role: "assistant".to_string(),
content: serde_json::json!([
{"type": "thinking", "thinking": "deep", "signature": "sig"},
{"type": "text", "text": "partial answer"}
]),
extra: Default::default(),
},
Message {
role: "user".to_string(),
content: serde_json::json!("continue"),
extra: Default::default(),
},
],
stream: false,
temperature: None,
max_tokens: None,
extra: Default::default(),
}
}

#[test]
fn model_affinity_source_resolves_entry_to_model_ref() {
let router = affinity_router(Default::default());
let request = affinity_conversation();
let cfg = crate::reasoning_compat::ReasoningCompatConfig::default();

let prefix_hash = StickyCache::compute_prefix_hash(&request);
router.sticky_cache.insert(
prefix_hash,
"anthropic".to_string(),
"claude-sonnet-4-5".to_string(),
None,
);

let source = router
.model_affinity_source(&request, &cfg)
.expect("fresh affinity entry resolves to a source ModelRef");
assert_eq!(source.provider, "anthropic");
assert_eq!(source.model, "claude-sonnet-4-5");
assert_eq!(
source.family,
crate::reasoning_compat::detect::classify_family("claude-sonnet-4-5")
);
}

#[test]
fn model_affinity_source_is_none_on_miss_or_disabled() {
let request = affinity_conversation();
let cfg = crate::reasoning_compat::ReasoningCompatConfig::default();

// No entry for this prefix → miss → None (no source attribution).
let router = affinity_router(Default::default());
assert!(router.model_affinity_source(&request, &cfg).is_none());

// Affinity flag off → no lookup at all, even with a fresh entry.
let router = affinity_router(crate::reasoning_compat::ReasoningCompatConfig {
conversation_model_affinity: false,
..Default::default()
});
let prefix_hash = StickyCache::compute_prefix_hash(&request);
router.sticky_cache.insert(
prefix_hash,
"anthropic".to_string(),
"claude-sonnet-4-5".to_string(),
None,
);
assert!(router.model_affinity_source(&request, &cfg).is_none());
}

#[test]
fn affinity_hit_same_model_preserves_reasoning_state() {
let router = affinity_router(Default::default());
let request = affinity_conversation();
let cfg = crate::reasoning_compat::ReasoningCompatConfig::default();

let prefix_hash = StickyCache::compute_prefix_hash(&request);
router.sticky_cache.insert(
prefix_hash,
"anthropic".to_string(),
"claude-sonnet-4-5".to_string(),
None,
);
let source = router.model_affinity_source(&request, &cfg).unwrap();

// Same resolved provider + model (mid-tool-loop continuation): the
// signed thinking blocks must survive verbatim.
let target = test_model_named("anthropic", "claude-sonnet-4-5", 1);
let mut outgoing = request.clone();
let report = reasoning_compat::prepare_attempt(
&mut outgoing,
&request,
Some(source),
&target,
&cfg,
);
assert_eq!(
report.decision,
reasoning_compat::policy::StripDecision::Preserve
);
assert_eq!(report.strip, reasoning_compat::policy::StripReport::default());
assert_eq!(
serde_json::to_value(&outgoing.messages).unwrap(),
serde_json::to_value(&request.messages).unwrap()
);
}

#[test]
fn affinity_hit_cross_family_strips_all_reasoning_state() {
let router = affinity_router(Default::default());
let request = affinity_conversation();
let cfg = crate::reasoning_compat::ReasoningCompatConfig::default();

let prefix_hash = StickyCache::compute_prefix_hash(&request);
router.sticky_cache.insert(
prefix_hash,
"anthropic".to_string(),
"claude-sonnet-4-5".to_string(),
None,
);
let source = router.model_affinity_source(&request, &cfg).unwrap();

let target = test_model_named("deepseek", "deepseek-reasoner", 1);
let mut outgoing = request.clone();
let report = reasoning_compat::prepare_attempt(
&mut outgoing,
&request,
Some(source),
&target,
&cfg,
);
assert_eq!(
report.decision,
reasoning_compat::policy::StripDecision::StripAll
);
assert_eq!(report.strip.thinking_blocks, 1);
assert!(outgoing.messages[0]
.content
.as_array()
.unwrap()
.iter()
.all(|block| block["type"] != "thinking"));
}

#[test]
fn affinity_miss_cross_family_strips_with_unknown_attribution() {
let router = affinity_router(Default::default());
let request = affinity_conversation();
let cfg = crate::reasoning_compat::ReasoningCompatConfig::default();

// No affinity entry: attribution unknown, conservative strip on a
// cross-family target.
assert!(router.model_affinity_source(&request, &cfg).is_none());
let target = test_model_named("deepseek", "deepseek-reasoner", 1);
let mut outgoing = request.clone();
let report =
reasoning_compat::prepare_attempt(&mut outgoing, &request, None, &target, &cfg);
assert_eq!(
report.decision,
reasoning_compat::policy::StripDecision::StripAttributionUnknown
);
assert_eq!(report.strip.thinking_blocks, 1);
}

#[tokio::test]
async fn sticky_routing_gate_widens_to_reasoning_affinity() {
// Reasoning affinity on, cache-aware routing off (its default): the
// gate is active and successes record affinity entries.
let router = affinity_router(Default::default());
assert!(router.sticky_routing_enabled().await);
assert!(!router.config.read().await.cache_aware_routing.enabled);

let request = affinity_conversation();
let usage = crate::models::openai::Usage::default();
router
.record_sticky_success(&request, "anthropic", "claude-sonnet-4-5", &usage)
.await;
let prefix_hash = StickyCache::compute_prefix_hash(&request);
assert_eq!(
router.sticky_cache.get_model_affinity(prefix_hash),
Some(("anthropic".to_string(), "claude-sonnet-4-5".to_string()))
);

// Both features off: gate closed, zero-TTL cache, nothing recorded.
let router = affinity_router(crate::reasoning_compat::ReasoningCompatConfig {
enabled: false,
conversation_model_affinity: false,
..Default::default()
});
assert!(!router.sticky_routing_enabled().await);
router
.record_sticky_success(&request, "anthropic", "claude-sonnet-4-5", &usage)
.await;
let prefix_hash = StickyCache::compute_prefix_hash(&request);
assert!(router.sticky_cache.get_model_affinity(prefix_hash).is_none());
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
                    cache_min_tokens: None,
                    cache_support: None,
                    cost_per_million_cache_read_input_tokens: None,
                    cost_per_million_cache_creation_input_tokens: None,
                    cost_per_million_reasoning_tokens: None,
                    reasoning_family: None,
                    reasoning_parameter: None,
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
    /// (4) version date descending when version fallback is enabled — a
    ///     DOMINANT re-sort that overrides (1)-(3) (see
    ///     `test_version_fallback_sorting`). The generated model names carry
    ///     no `YYYY-MM-DD` suffix, so for the generated groups the priority /
    ///     cost invariants below hold unconditionally.
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
                cache_min_tokens: None,
                cache_support: None,
                cost_per_million_cache_read_input_tokens: None,
                cost_per_million_cache_creation_input_tokens: None,
                cost_per_million_reasoning_tokens: None,
                reasoning_family: None,
                reasoning_parameter: None,
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
                cache_min_tokens: None,
                cache_support: None,
                cost_per_million_cache_read_input_tokens: None,
                cost_per_million_cache_creation_input_tokens: None,
                cost_per_million_reasoning_tokens: None,
                reasoning_family: None,
                reasoning_parameter: None,
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
    fn test_is_rate_limit_class_error_429() {
        let err = GatewayError::Provider {
            provider: "p".to_string(),
            message: "Too Many Requests".to_string(),
            status_code: Some(429),
        };
        assert!(Router::is_rate_limit_class_error(&err));
    }

    #[test]
    fn test_is_rate_limit_class_error_promoted_200_envelope() {
        // Error-in-200 envelopes are promoted to status 429 by the dispatch
        // loop; the promoted error must be recognized as rate-limit-class.
        let err = GatewayError::Provider {
            provider: "p".to_string(),
            message: "Rate limited (HTTP 200 envelope): slow down".to_string(),
            status_code: Some(429),
        };
        assert!(Router::is_rate_limit_class_error(&err));
    }

    #[test]
    fn test_is_rate_limit_class_error_non_rate_limit() {
        for status in [400u16, 401, 404, 500, 503] {
            let err = GatewayError::Provider {
                provider: "p".to_string(),
                message: format!("HTTP {} failure", status),
                status_code: Some(status),
            };
            assert!(
                !Router::is_rate_limit_class_error(&err),
                "status {} must not be rate-limit-class",
                status
            );
        }
        // 5xx with rate-limit wording stays non-rate-limit-class (5xx are
        // health failures, not pause signals).
        let err = GatewayError::Provider {
            provider: "p".to_string(),
            message: "rate limit hit while overloaded".to_string(),
            status_code: Some(503),
        };
        assert!(!Router::is_rate_limit_class_error(&err));
    }

    #[test]
    fn test_is_rate_limit_class_error_non_provider_error() {
        let err = GatewayError::TtfbTimeout(30);
        assert!(!Router::is_rate_limit_class_error(&err));
    }

    fn truncation_test_response(
        finish_reason: Option<&str>,
        completion_tokens: u32,
    ) -> OpenAIResponse {
        OpenAIResponse {
            id: "test".to_string(),
            object: "chat.completion".to_string(),
            created: 0,
            model: "test-model".to_string(),
            choices: vec![Choice {
                index: 0,
                message: Message {
                    role: "assistant".to_string(),
                    content: serde_json::Value::String("partial".to_string()),
                    extra: serde_json::Map::new(),
                },
                finish_reason: finish_reason.map(|s| s.to_string()),
                extra: serde_json::Map::new(),
            }],
            usage: Usage {
                prompt_tokens: 10,
                completion_tokens,
                total_tokens: 10 + completion_tokens,
                extra: serde_json::Map::new(),
            },
            extra: serde_json::Map::new(),
        }
    }

    #[test]
    fn test_is_suspicious_truncation_detected() {
        let response = truncation_test_response(Some("length"), 100);
        assert!(Router::is_suspicious_truncation(&response, Some(4096)));
    }

    #[test]
    fn test_is_suspicious_truncation_near_limit_not_suspicious() {
        // completion_tokens within 50 of max_tokens is a legitimate stop.
        let response = truncation_test_response(Some("length"), 4060);
        assert!(!Router::is_suspicious_truncation(&response, Some(4096)));
    }

    #[test]
    fn test_is_suspicious_truncation_finish_reason_stop() {
        let response = truncation_test_response(Some("stop"), 100);
        assert!(!Router::is_suspicious_truncation(&response, Some(4096)));
    }

    #[test]
    fn test_is_suspicious_truncation_no_max_tokens() {
        let response = truncation_test_response(Some("length"), 100);
        assert!(!Router::is_suspicious_truncation(&response, None));
    }

    #[test]
    fn test_is_suspicious_truncation_missing_usage_not_suspicious() {
        // Providers that omit `usage` default to completion_tokens == 0; that
        // must NOT be read as "stopped far short" (false failover + spurious
        // circuit-breaker failure on a legitimate length-capped response).
        let mut response = truncation_test_response(Some("length"), 0);
        response.usage = Usage::default();
        assert!(!Router::is_suspicious_truncation(&response, Some(4096)));
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
                    cache_min_tokens: None,
                    cache_support: None,
                    cost_per_million_cache_read_input_tokens: None,
                    cost_per_million_cache_creation_input_tokens: None,
                    cost_per_million_reasoning_tokens: None,
                    reasoning_family: None,
                    reasoning_parameter: None,
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
                    cache_min_tokens: None,
                    cache_support: None,
                    cost_per_million_cache_read_input_tokens: None,
                    cost_per_million_cache_creation_input_tokens: None,
                    cost_per_million_reasoning_tokens: None,
                    reasoning_family: None,
                    reasoning_parameter: None,
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

#[test]
fn test_tool_hint_injection_targets_learned_combos_only() {
    let router = Router::new(Arc::new(RwLock::new(create_test_config())), test_metrics());

    // Unknown models never see the hint: zero prompt overhead, no
    // unexplained instructions for models that call tools natively.
    assert!(!router.should_inject_tool_hint("openai-main", "gpt-4o"));
    assert!(!router.should_inject_tool_hint("lite", "claude-sonnet-4-5"));
    assert!(!router.should_inject_tool_hint("local", "llama3.1-8b"));

    // Former built-in XML-prone families also start clean now — the
    // hint is demoted to learned combos only, with
    // `translate_xml_tool_calls` repairing any first XML-flavored
    // response transparently.
    assert!(!router.should_inject_tool_hint("x", "Kimi-K2-Instruct"));
    assert!(!router.should_inject_tool_hint("x", "qwen2.5-72b-instruct"));
    assert!(!router.should_inject_tool_hint("x", "glm-4.6"));
    assert!(!router.should_inject_tool_hint("x", "deepseek-v3"));

    // Learned combos get it regardless of family — and stay learned:
    // the hint must not appear/disappear between conversation turns.
    router.mark_xml_tool_combo("weird-provider", "some-model");
    assert!(router.should_inject_tool_hint("weird-provider", "some-model"));
    assert!(router.is_xml_tool_combo("weird-provider", "some-model"));
}

#[test]
fn test_tool_hint_text_is_attributed_and_xml_free() {
    // The hint must read as attributed infrastructure guidance, not an
    // anonymous imperative block: attributed, positively framed, and
    // free of the XML tag litany that matches prompt-injection
    // fingerprints (and would trip `looks_like_xml_tool_use`).
    let content = match Router::tool_calling_system_hint().content {
        serde_json::Value::String(text) => text,
        other => panic!("hint content must be a string, got: {other:?}"),
    };
    assert!(content.starts_with("[gateway]"));
    assert!(!content.contains('<'));
    assert!(!Router::looks_like_xml_tool_use(&content));
    assert!(content.len() < 500, "hint should stay short, got {} bytes", content.len());
}

#[test]
fn test_insert_tool_calling_hint_positions_after_system_block() {
    let msg = |role: &str| Message {
        role: role.to_string(),
        content: serde_json::Value::String(format!("{role} content")),
        extra: serde_json::Map::new(),
    };

    // With a system prompt present, the hint lands directly after the
    // last system message — never at the tail after user/tool content.
    let mut messages = vec![msg("system"), msg("user"), msg("assistant"), msg("user")];
    Router::insert_tool_calling_hint(&mut messages);
    assert_eq!(messages.len(), 5);
    assert_eq!(messages[1].role, "system");
    assert_eq!(messages[2].role, "user");
    assert_eq!(messages[4].role, "user");
    match messages[1].content {
        serde_json::Value::String(ref text) => assert!(text.starts_with("[gateway]")),
        ref other => panic!("hint content must be a string, got: {other:?}"),
    }

    // Multiple system messages: insert after the last one.
    let mut messages = vec![msg("system"), msg("system"), msg("user")];
    Router::insert_tool_calling_hint(&mut messages);
    assert_eq!(messages[2].role, "system");
    assert!(match messages[2].content {
        serde_json::Value::String(ref text) => text.starts_with("[gateway]"),
        ref other => panic!("hint content must be a string, got: {other:?}"),
    });

// No system message at all: the hint becomes the first message.
let mut messages = vec![msg("user"), msg("assistant")];
Router::insert_tool_calling_hint(&mut messages);
assert_eq!(messages[0].role, "system");
assert_eq!(messages[1].role, "user");
}
}
