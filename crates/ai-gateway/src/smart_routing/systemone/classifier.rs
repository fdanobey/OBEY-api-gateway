//! The shared System One classifier facade: the seam that plugs a System One
//! decision-model family (Jev, Laya) into the smart-routing pipeline as an
//! `OptionalClassifier`.
//!
//! Classification flow: SimHash score cache -> model discovery (TTL-cached) ->
//! System One evaluation under a timeout -> pure composition with confidence
//! gating. Every failure mode maps to the bounded `ClassifierFailure`
//! taxonomy so the orchestrator falls back exactly as it does for the ML and
//! LLM classifiers. No payload content ever reaches logs, metrics, or errors.

use std::collections::{HashMap, VecDeque};
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::{Arc, Mutex, MutexGuard};
use std::time::{Duration, Instant};

use async_trait::async_trait;

use crate::models::openai::OpenAIRequest;
use crate::smart_routing::config::{JevFallbackPolicy, JevTrustOverride};
use crate::smart_routing::tier::TaskType;
use crate::smart_routing::{
    ClassifierFailure, ClassifierInput, ClassifierOutput, OptionalClassifier, WarnRateLimiter,
};

use super::client::SystemOneClient;
use super::compose::{self, ComposeResult, JevTrustConfig};
use super::discovery::{
    DiscoveryError, ModelDiscovery, ModelsFetcher, ResolutionOrigin, ResolvedModel,
};
use super::models::{ModelsListing, SystemOneRequest};
use super::rubric;
use super::SystemOneFamily;

/// Score-cache capacity mirroring `LlmClassifier`.
pub const SCORE_CACHE_CAPACITY: usize = 1_000;
/// Score-cache TTL mirroring `LlmClassifier`.
pub const SCORE_CACHE_TTL: Duration = Duration::from_secs(5 * 60);
/// Timeout hard cap per requirements (config may be lower).
const TIMEOUT_HARD_CAP: Duration = Duration::from_millis(2_000);
/// Minimum char budget after a 413-driven reduction.
const MIN_CHAR_BUDGET: usize = 256;
/// Char-budget reduction factor applied on payload-limit responses.
const CHAR_BUDGET_REDUCTION: usize = 2;

/// Family-neutral endpoint settings extracted from a family config.
#[derive(Debug, Clone)]
pub struct SystemOneEndpointSettings {
    pub base_url: String,
    pub timeout_ms: u16,
    pub min_confidence: f64,
    pub min_task_confidence: f64,
    pub retry_max_attempts: u8,
    pub retry_backoff_ms: u16,
    pub char_budget: usize,
    pub discovery_ttl_secs: u64,
    pub dimension_weights: crate::smart_routing::config::DimensionWeights,
    pub fallback_policy: JevFallbackPolicy,
    pub model: String,
}

/// Extra detail produced by a successful classification, consumed by the
/// orchestrator for telemetry and decision records.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct SystemOneClassificationDetail {
    pub task_type: TaskType,
    pub confidence: f64,
}

/// Fetcher that records whether discovery actually needs a network fetch.
/// `ModelDiscovery::resolve` calls `fetch` only after the pinned check and
/// the TTL cache miss, so `attempted == false` proves the outcome was
/// served without any fetch — the Req 5.2 fast path.
struct FetchProbe {
    attempted: AtomicBool,
}

impl ModelsFetcher for FetchProbe {
    fn fetch(&self) -> Result<ModelsListing, String> {
        self.attempted.store(true, Ordering::SeqCst);
        Err("probe".to_string())
    }
}

/// Fetcher serving an already-awaited listing result, so
/// `ModelDiscovery::resolve` performs its full select/cache/fallback logic
/// on the request path without a blocking network call (Req 5.1).
struct PrefetchedFetcher {
    outcome: Result<ModelsListing, String>,
}

impl ModelsFetcher for PrefetchedFetcher {
    fn fetch(&self) -> Result<ModelsListing, String> {
        // `resolve` consumes exactly one fetch; clone preserves the
        // original for a hypothetical second call.
        self.outcome.clone()
    }
}

/// LRU + TTL score cache, the `LlmClassifier` pattern applied verbatim.
#[derive(Debug)]
struct ScoreCache {
    state: Mutex<CacheState>,
    capacity: usize,
    ttl: Duration,
}

#[derive(Debug)]
struct CacheState {
    entries: HashMap<u64, CacheEntry>,
    lru: VecDeque<u64>,
}

#[derive(Debug, Clone, Copy)]
struct CacheEntry {
    detail: CachedClassification,
    expires_at: Instant,
}

/// Everything needed to serve a decision from cache.
#[derive(Debug, Clone, Copy, PartialEq)]
struct CachedClassification {
    score: f64,
    task_type: TaskType,
    confidence: f64,
    blended: bool,
}

impl ScoreCache {
    fn new(capacity: usize, ttl: Duration) -> Self {
        Self {
            state: Mutex::new(CacheState {
                entries: HashMap::with_capacity(capacity),
                lru: VecDeque::with_capacity(capacity),
            }),
            capacity,
            ttl,
        }
    }

    fn get(&self, fingerprint: u64) -> Option<CachedClassification> {
        self.lock().get(fingerprint, Instant::now())
    }

    fn insert(&self, fingerprint: u64, detail: CachedClassification) {
        self.lock()
            .insert(fingerprint, detail, Instant::now(), self.ttl, self.capacity);
    }

    fn lock(&self) -> MutexGuard<'_, CacheState> {
        self.state
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner)
    }
}

impl CacheState {
    fn get(&mut self, fingerprint: u64, now: Instant) -> Option<CachedClassification> {
        let entry = self.entries.get(&fingerprint).copied()?;
        if entry.expires_at <= now {
            self.remove(fingerprint);
            return None;
        }
        self.touch(fingerprint);
        Some(entry.detail)
    }

    fn insert(
        &mut self,
        fingerprint: u64,
        detail: CachedClassification,
        now: Instant,
        ttl: Duration,
        capacity: usize,
    ) {
        self.remove_expired(now);
        self.remove(fingerprint);
        while self.entries.len() >= capacity {
            let Some(oldest) = self.lru.pop_front() else {
                break;
            };
            self.entries.remove(&oldest);
        }
        self.entries.insert(
            fingerprint,
            CacheEntry {
                detail,
                expires_at: now + ttl,
            },
        );
        self.lru.push_back(fingerprint);
    }

    fn remove_expired(&mut self, now: Instant) {
        self.entries.retain(|_, entry| entry.expires_at > now);
        self.lru
            .retain(|fingerprint| self.entries.contains_key(fingerprint));
    }

    fn remove(&mut self, fingerprint: u64) {
        self.entries.remove(&fingerprint);
        self.lru.retain(|cached| *cached != fingerprint);
    }

    fn touch(&mut self, fingerprint: u64) {
        self.lru.retain(|cached| *cached != fingerprint);
        self.lru.push_back(fingerprint);
    }
}

/// The shared facade. Construct via [`SystemOneClassifier::new`] after config
/// validation; the family descriptor supplies defaults and behavior
/// (retryable statuses, confidence selection, discovery defaults).
pub struct SystemOneClassifier {
    family: SystemOneFamily,
    client: Arc<SystemOneClient>,
    discovery: Arc<ModelDiscovery>,
    /// Serializes discovery resolution (probe → fetch → replay) for this
    /// classifier so the probe's transient sentinel entry is never
    /// observable by another caller: concurrent resolves wait here and then
    /// serve the real cached outcome.
    discovery_gate: Arc<tokio::sync::Mutex<()>>,
    /// The listing fetch used on a confirmed cache miss; injectable in
    /// tests so discovery semantics are provable without network.
    discovery_fetch: ListingFetch,
    settings: SystemOneEndpointSettings,
    trust: JevTrustConfig,
    timeout: Duration,
    cache: ScoreCache,
    /// Char budget reduced after a payload-limit response (413); bounded
    /// below by [`MIN_CHAR_BUDGET`].
    effective_char_budget: Mutex<usize>,
    call_failure_warnings: WarnRateLimiter,
}

impl SystemOneClassifier {
    /// Build from a family descriptor and resolved endpoint settings. The
    /// API key must already be resolved (empty keys are rejected by config
    /// validation).
    pub fn new(
        family: SystemOneFamily,
        settings: SystemOneEndpointSettings,
        api_key: &str,
    ) -> Result<Self, ClassifierFailure> {
        if api_key.trim().is_empty() {
            return Err(ClassifierFailure::Unavailable);
        }
        let timeout = Duration::from_millis(u64::from(settings.timeout_ms)).min(TIMEOUT_HARD_CAP);
        let trust = JevTrustConfig {
            min_confidence: settings.min_confidence,
            min_task_confidence: settings.min_task_confidence,
            fallback_policy: settings.fallback_policy,
        };

        let client = Arc::new(
            SystemOneClient::new(
                family,
                &settings.base_url,
                api_key,
                timeout,
                settings.retry_max_attempts,
                Duration::from_millis(u64::from(settings.retry_backoff_ms)),
            )
            .map_err(|_| ClassifierFailure::Backend)?,
        );
        let discovery = Arc::new(ModelDiscovery::new(
            family,
            &settings.model,
            settings.discovery_ttl_secs,
        ));
        let effective_char_budget = Mutex::new(settings.char_budget);

    Ok(Self {
        family,
        cache: ScoreCache::new(SCORE_CACHE_CAPACITY, SCORE_CACHE_TTL),
        client: Arc::clone(&client),
        discovery,
        discovery_gate: Arc::new(tokio::sync::Mutex::new(())),
        discovery_fetch: ListingFetch::Client(client),
        settings,
            trust,
            timeout,
            effective_char_budget,
            call_failure_warnings: WarnRateLimiter::default(),
        })
    }

    /// Best-effort startup resolution so the first request does not pay the
    /// discovery cost and operators see the resolved model in logs. Spawned
    /// because router construction is synchronous; the fetch runs on the
    /// async runtime (no tokio worker is blocked or removed from rotation).
    /// Priming goes through the same resolve helper as the request path so
    /// the pinned/cached fast path and negative-cache TTL semantics apply.
    pub fn prime_discovery(&self) {
        let resolver = DiscoveryResolver {
            discovery: Arc::clone(&self.discovery),
            gate: Arc::clone(&self.discovery_gate),
            fetch: self.discovery_fetch.clone(),
        };
        let family = self.family;
        if let Ok(handle) = tokio::runtime::Handle::try_current() {
            handle.spawn(async move {
                let outcome = resolver.resolve().await;
                log_priming(family.name, outcome);
            });
        }
        // Without a runtime there is nothing to prime onto; discovery
        // defers to the first request as before.
    }

    /// Invalidate cached discovery (hot-reload semantics).
    pub fn invalidate_discovery(&self) {
        self.discovery.invalidate();
    }

    /// Last resolved model for status surfaces; `None` before first
    /// resolution. Reads the cache only; never fetches or blocks on network.
    pub fn resolved_model(&self) -> Option<String> {
        self.discovery.resolved_model()
    }

    /// The family this classifier serves (`jev` | `laya`).
    pub fn family(&self) -> SystemOneFamily {
        self.family
    }

    /// The active family-selected confidence gate (`min_confidence`).
    pub fn confidence_gate(&self) -> f64 {
        self.trust.min_confidence
    }

    /// The configured base URL (safe for status surfaces; contains no
    /// credential material).
    pub fn base_url(&self) -> &str {
        &self.settings.base_url
    }

    fn trust(&self) -> JevTrustConfig {
        self.trust
    }

    fn trust_for(&self, policy: Option<JevTrustOverride>) -> JevTrustConfig {
        policy.map_or_else(
            || self.trust(),
            |trust| JevTrustConfig {
                min_confidence: trust.min_confidence,
                min_task_confidence: trust.min_task_confidence,
                fallback_policy: trust.fallback_policy,
            },
        )
    }

    /// Halve the effective char budget after a payload-limit response, down
    /// to the floor. Returns the budget used for the next state assembly.
    fn reduce_char_budget(&self) -> usize {
        let mut budget = self
            .effective_char_budget
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner);
        *budget = (*budget / CHAR_BUDGET_REDUCTION).max(MIN_CHAR_BUDGET);
        *budget
    }

    /// Classify with full detail; the trait method is a thin wrapper.
    pub async fn classify_with_detail(
        &self,
        input: ClassifierInput<'_>,
    ) -> Result<(ClassifierOutput, SystemOneClassificationDetail), ClassifierFailure> {
        let trust = self.trust_for(input.jev_trust);
        let fingerprint = request_fingerprint(input.request) ^ trust_fingerprint(&trust);
        if let Some(cached) = self.cache.get(fingerprint) {
            return Ok((
                ClassifierOutput {
                    score: cached.score,
                    task_type: Some(cached.task_type),
                    confidence: Some(cached.confidence),
                    resolved_model: self.resolved_model(),
                    discovery_refreshed: false,
                    blended: cached.blended,
                },
                SystemOneClassificationDetail {
                    task_type: cached.task_type,
                    confidence: cached.confidence,
                },
            ));
        }

        let resolved = self
            .resolve_discovery()
            .await
            .map_err(map_discovery_failure)?;
        let discovery_refreshed = matches!(resolved.origin, ResolutionOrigin::Auto);

        let char_budget = *self
            .effective_char_budget
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner);
        let request = SystemOneRequest {
            state: rubric::build_state(input.request, char_budget),
            model: resolved.model.clone(),
            questions: rubric::build_questions(),
        };

        let response = match tokio::time::timeout(self.timeout, self.client.evaluate(&request))
            .await
            .map_err(|_| ClassifierFailure::Timeout)?
        {
            Ok(response) => response,
            Err(super::client::SystemOneCallError::PayloadTooLarge) => {
                // Reduce the budget for subsequent requests and fail this
                // one; the orchestrator falls back to the heuristic chain.
                let reduced = self.reduce_char_budget();
                tracing::warn!(
                    family = self.family.name,
                    char_budget = reduced,
                    "System One endpoint rejected the payload; reduced char budget for subsequent requests"
                );
                return Err(ClassifierFailure::Backend);
            }
            Err(other) => {
                // Content-free: error class and HTTP status only.
                if self
                    .call_failure_warnings
                    .allow(self.family.name, other.class())
                {
                    let status = match &other {
                        super::client::SystemOneCallError::Transient { status }
                        | super::client::SystemOneCallError::Auth { status }
                        | super::client::SystemOneCallError::BadRequest { status, .. } => {
                            Some(*status)
                        }
                        _ => None,
                    };
                    tracing::warn!(
                        family = %self.family.name,
                        error_class = %other.class(),
                        status,
                        "System One classifier call failed (rate-limited, 60s)"
                    );
                }
                return Err(map_call_failure(other));
            }
        };

        let result = compose::compose(
            &response.answers,
            &self.settings.dimension_weights,
            &trust,
            &self.family,
        );
        match result {
            ComposeResult::Ok {
                score,
                task_type,
                confidence,
            } => {
                let detail = SystemOneClassificationDetail {
                    task_type,
                    confidence,
                };
                self.cache.insert(
                    fingerprint,
                    CachedClassification {
                        score: score.value(),
                        task_type,
                        confidence,
                        blended: false,
                    },
                );
                Ok((
                    ClassifierOutput {
                        score: score.value(),
                        task_type: Some(task_type),
                        confidence: Some(confidence),
                        resolved_model: Some(resolved.model),
                        discovery_refreshed,
                        blended: false,
                    },
                    detail,
                ))
            }
            ComposeResult::LowConfidence { score, confidence } => {
                // Blend policy resolves here so the orchestrator sees one
                // uniform success/failure contract: blending yields a usable
                // score; strict fallback reports low confidence upward.
                match trust.fallback_policy {
                    JevFallbackPolicy::Blend => {
                        let blended = compose::blend_scores(
                            score,
                            input.heuristic_score,
                            confidence,
                            trust.min_confidence,
                        );
                        let detail = SystemOneClassificationDetail {
                            task_type: input.heuristic_task_type,
                            confidence,
                        };
                        // Blend results are cached like any other decision.
                        self.cache.insert(
                            fingerprint,
                            CachedClassification {
                                score: blended.value(),
                                task_type: input.heuristic_task_type,
                                confidence,
                                blended: true,
                            },
                        );
                        Ok((
                            ClassifierOutput {
                                score: blended.value(),
                                task_type: Some(input.heuristic_task_type),
                                confidence: Some(confidence),
                                resolved_model: Some(resolved.model),
                                discovery_refreshed,
                                blended: true,
                            },
                            detail,
                        ))
                    }
                    JevFallbackPolicy::Fallback => {
                        // The failure enum carries no value; surface the
                        // score that missed the threshold for tuning.
                        tracing::debug!(
                            family = %self.family.name,
                            confidence,
                            min_confidence = trust.min_confidence,
                            "System One low confidence; using heuristic"
                        );
                        Err(ClassifierFailure::LowConfidence)
                    }
                }
            }
            ComposeResult::Invalid => Err(ClassifierFailure::InvalidOutput),
        }
    }
}

#[async_trait]
impl OptionalClassifier for SystemOneClassifier {
    async fn classify(
        &self,
        input: ClassifierInput<'_>,
    ) -> Result<ClassifierOutput, ClassifierFailure> {
        self.classify_with_detail(input)
            .await
            .map(|(output, _)| output)
    }
}

impl SystemOneClassifier {
    /// Resolve model discovery without blocking a tokio worker (Req 5.1).
    async fn resolve_discovery(&self) -> Result<ResolvedModel, DiscoveryError> {
        DiscoveryResolver {
            discovery: Arc::clone(&self.discovery),
            gate: Arc::clone(&self.discovery_gate),
            fetch: self.discovery_fetch.clone(),
        }
        .resolve()
        .await
    }
}

/// Gated discovery resolution shared by the request path and priming.
///
/// A probe resolve runs first, mirroring `ModelDiscovery::resolve`'s sync
/// pinned/cached fast path. When the probe's fetcher is never invoked the
/// outcome came from a pin or a fresh (including negatively cached) entry,
/// so it is returned as-is — no network await, nothing spawned (Req 5.2).
/// Only a confirmed miss (the probe's fetcher fired) awaits the real
/// listing fetch and replays the outcome through
/// `ModelDiscovery::resolve`, dropping the probe's sentinel entry first so
/// TTL caching, negative caching, and the Laya missing-listing fallback
/// all run through discovery's own unchanged logic (Req 5.1).
///
/// The whole miss path (probe → invalidate → await fetch → replay) runs
/// behind a per-classifier async gate. The probe stores a sentinel
/// `"probe"` failure via `ModelDiscovery::resolve`'s own caching; the gate
/// guarantees no other caller of this classifier resolves between the
/// store and the `invalidate()`: concurrent callers queue on the gate and
/// then observe only REAL outcomes — a fresh resolution or a negative
/// cache entry written by the fetch path. From every other caller's point
/// of view the probe is read-only against the cache.
struct DiscoveryResolver {
    discovery: Arc<ModelDiscovery>,
    gate: Arc<tokio::sync::Mutex<()>>,
    fetch: ListingFetch,
}

impl DiscoveryResolver {
    async fn resolve(&self) -> Result<ResolvedModel, DiscoveryError> {
        let _guard = self.gate.lock().await;
        let probe = FetchProbe {
            attempted: AtomicBool::new(false),
        };
        let outcome = self.discovery.resolve(&probe);
        if !probe.attempted.load(Ordering::SeqCst) {
            return outcome;
        }
        self.discovery.invalidate();
        let fetcher = PrefetchedFetcher {
            outcome: self.fetch.fetch().await,
        };
        self.discovery.resolve(&fetcher)
    }
}

/// The real listing fetch performed once the probe confirms a cache miss.
/// Production awaits the client's native async `list_models`; tests
/// inject a deterministic counted fetch so discovery behavior is provable
/// without network.
#[derive(Clone)]
enum ListingFetch {
    /// Production: `SystemOneClient::list_models`.
    Client(Arc<SystemOneClient>),
    /// Test seam: a deterministic, call-counted fetch (F4).
    #[cfg(test)]
    Injected(Arc<tests::CountingFetcher>),
}

impl ListingFetch {
    async fn fetch(&self) -> Result<ModelsListing, String> {
        match self {
            Self::Client(client) => client.list_models().await.map_err(|e| e.to_string()),
            #[cfg(test)]
            Self::Injected(fetcher) => fetcher.fetch().await,
        }
    }
}

/// Priming outcome logging, unchanged from the blocking version.
fn log_priming(family: &'static str, outcome: Result<ResolvedModel, DiscoveryError>) {
    match outcome {
        Ok(resolved) => tracing::info!(
            family,
            model = %resolved.model,
            origin = ?resolved.origin,
            "System One classifier ready"
        ),
        Err(DiscoveryError::NoModelAtEndpoint) => {
            // ModelDiscovery already emitted its rate-limited warning.
        }
        Err(error) => tracing::warn!(
            family,
            error = %format!("{error:?}"),
            "System One classifier model discovery deferred to first request"
        ),
    }
}

fn map_discovery_failure(error: DiscoveryError) -> ClassifierFailure {
    match error {
        DiscoveryError::NoModelAtEndpoint => ClassifierFailure::NoJevAtEndpoint,
        DiscoveryError::ModelUnavailable { .. } => ClassifierFailure::Unavailable,
        DiscoveryError::Listing(_) => ClassifierFailure::Unavailable,
    }
}

fn map_call_failure(error: super::client::SystemOneCallError) -> ClassifierFailure {
    match error {
        super::client::SystemOneCallError::Transient { .. } => ClassifierFailure::Backend,
        super::client::SystemOneCallError::Auth { .. } => ClassifierFailure::Backend,
        super::client::SystemOneCallError::BadRequest { .. } => ClassifierFailure::InvalidOutput,
        // 413 is handled by the caller (char-budget reduction) and never
        // reaches this mapping in the classify path.
        super::client::SystemOneCallError::PayloadTooLarge => ClassifierFailure::Backend,
        super::client::SystemOneCallError::Unavailable { .. } => ClassifierFailure::Unavailable,
        super::client::SystemOneCallError::Timeout => ClassifierFailure::Timeout,
    }
}

/// Request fingerprint: identical rules to `LlmClassifier::request_fingerprint`
/// so cache behavior is consistent across classifier backends.
fn request_fingerprint(request: &OpenAIRequest) -> u64 {
    let mut fingerprint = crate::loop_detection::simhash::compute(&request.model);
    for message in &request.messages {
        if message.role.eq_ignore_ascii_case("tool") {
            continue;
        }

        fingerprint = mix_fingerprint(
            fingerprint,
            crate::loop_detection::simhash::compute(&message.role),
        );
        rubric::visit_message_text(message, |text| {
            fingerprint =
                mix_fingerprint(fingerprint, crate::loop_detection::simhash::compute(text));
        });
    }
    fingerprint
}

fn mix_fingerprint(current: u64, next: u64) -> u64 {
    current.rotate_left(13) ^ next.wrapping_mul(0x9e37_79b9_7f4a_7c15)
}

/// Mix per-call trust thresholds into the cache key so model groups with
/// different trust overrides never share a cached decision.
fn trust_fingerprint(trust: &JevTrustConfig) -> u64 {
    let mut fingerprint = trust.min_confidence.to_bits();
    fingerprint = mix_fingerprint(fingerprint, trust.min_task_confidence.to_bits());
    fingerprint = mix_fingerprint(
        fingerprint,
        match trust.fallback_policy {
            JevFallbackPolicy::Fallback => 0u64,
            JevFallbackPolicy::Blend => 1u64,
        },
    );
    fingerprint
}

#[cfg(test)]
mod tests {
    use serde_json::{json, Map, Value};

    use super::*;
    use crate::models::openai::Message;
    use crate::smart_routing::systemone::models::ModelEntry;

    fn settings() -> SystemOneEndpointSettings {
        SystemOneEndpointSettings {
            base_url: "https://api.typesafe.ai".to_string(),
            timeout_ms: 500,
            min_confidence: 0.60,
            min_task_confidence: 0.50,
            retry_max_attempts: 2,
            retry_backoff_ms: 1,
            char_budget: 1_024,
            discovery_ttl_secs: 3_600,
            dimension_weights: crate::smart_routing::config::DimensionWeights::default(),
            fallback_policy: JevFallbackPolicy::Fallback,
            model: "auto".to_string(),
        }
    }

    fn request(content: Value) -> OpenAIRequest {
        OpenAIRequest {
            model: "model".to_string(),
            messages: vec![Message {
                role: "user".to_string(),
                content,
                extra: Map::new(),
            }],
            stream: false,
            temperature: None,
            max_tokens: None,
            extra: Map::new(),
        }
    }

    fn test_model_group() -> crate::config::ModelGroup {
        crate::config::ModelGroup {
            name: "group".to_string(),
            version_fallback_enabled: false,
            compression: None,
            memory: None,
            structured_output: None,
            models: Vec::new(),
        }
    }

    #[test]
    fn build_rejects_unresolvable_key() {
        assert!(matches!(
            SystemOneClassifier::new(SystemOneFamily::JEV, settings(), ""),
            Err(ClassifierFailure::Unavailable)
        ));
    }

    #[test]
    fn char_budget_reduction_is_bounded() {
        let classifier =
            SystemOneClassifier::new(SystemOneFamily::LAYA, settings(), "key").unwrap();
        assert_eq!(classifier.reduce_char_budget(), 512);
        assert_eq!(classifier.reduce_char_budget(), 256);
        assert_eq!(classifier.reduce_char_budget(), 256, "floor is respected");
    }

    #[tokio::test]
    async fn classify_without_network_fails_gracefully() {
        // The default base URL is unreachable in tests; every outcome must be
        // a typed failure, never a panic. Timeout is the expected class since
        // discovery is deferred inside the classify timeout budget.
        let classifier = SystemOneClassifier::new(SystemOneFamily::JEV, settings(), "key").unwrap();
        let request = request(json!("hello"));

        let model_group = test_model_group();
        let pinned_context = crate::smart_routing::PinnedRoutingContext::default();

        let outcome = classifier
            .classify_with_detail(ClassifierInput {
                request: &request,
                model_group: &model_group,
                pinned_context: &pinned_context,
                heuristic_score: crate::smart_routing::tier::ComplexityScore::new(0.5),
                heuristic_task_type: TaskType::General,
                jev_trust: None,
            })
            .await;
        assert!(outcome.is_err());
    }

    /// Req 5.2: a pinned model serves discovery synchronously — the probe
    /// fetcher never fires, so no network future is created and nothing is
    /// spawned. Behavioral proof: resolution succeeds without any fetch.
    #[tokio::test]
    async fn discovery_fast_path_serves_pin_without_fetch() {
        let mut pinned = settings();
        pinned.model = "jev-1.13.0".to_string();
        let classifier = SystemOneClassifier::new(SystemOneFamily::JEV, pinned, "key").unwrap();

        let resolved = classifier.resolve_discovery().await.unwrap();
        assert_eq!(resolved.model, "jev-1.13.0");
        assert_eq!(resolved.origin, ResolutionOrigin::Pinned);
    }

/// Req 5.2: a fresh cached resolution serves discovery without a fetch —
/// the probe fetcher never fires on the second call.
#[tokio::test]
async fn discovery_fast_path_serves_cache_without_fetch() {
    // Prime the cache by hand (no network): the listing path runs through
    // ModelDiscovery's own logic.
    let classifier = SystemOneClassifier::new(SystemOneFamily::JEV, settings(), "key").unwrap();
    let listing = ModelsListing {
        data: vec![ModelEntry {
            id: "jev-1.13.0".to_string(),
            owned_by: None,
        }],
    };
    let primed = crate::smart_routing::systemone::discovery::ModelDiscovery::resolve(
        &classifier.discovery,
        &PrefetchedFetcher {
            outcome: Ok(listing),
        },
    )
    .unwrap();
    assert_eq!(primed.origin, ResolutionOrigin::Auto);

    // Cache hit: served as Cached without any fetch attempt.
    let resolved = classifier.resolve_discovery().await.unwrap();
    assert_eq!(resolved.model, "jev-1.13.0");
    assert_eq!(resolved.origin, ResolutionOrigin::Cached);
}

/// Deterministic, call-counted listing fetch for discovery tests: no
/// network, replays a fixed outcome, and records how many times the
/// real-fetch seam fired.
pub(crate) struct CountingFetcher {
    listing: Result<ModelsListing, String>,
    calls: std::sync::atomic::AtomicUsize,
}

impl CountingFetcher {
    fn listing(ids: &[&str]) -> Self {
        Self {
            listing: Ok(ModelsListing {
                data: ids
                    .iter()
                    .map(|id| ModelEntry {
                        id: id.to_string(),
                        owned_by: None,
                    })
                    .collect(),
            }),
            calls: std::sync::atomic::AtomicUsize::new(0),
        }
    }

    fn failing(message: &str) -> Self {
        Self {
            listing: Err(message.to_string()),
            calls: std::sync::atomic::AtomicUsize::new(0),
        }
    }

    fn calls(&self) -> usize {
        self.calls.load(Ordering::SeqCst)
    }

    pub(crate) async fn fetch(&self) -> Result<ModelsListing, String> {
        self.calls.fetch_add(1, Ordering::SeqCst);
        self.listing.clone()
    }
}

/// Build a classifier whose miss-path listing fetch is the injected
/// `CountingFetcher` instead of the network client.
fn classifier_with_fetch(
    fetcher: Arc<CountingFetcher>,
) -> SystemOneClassifier {
    let mut classifier = SystemOneClassifier::new(SystemOneFamily::JEV, settings(), "key").unwrap();
    classifier.discovery_fetch = ListingFetch::Injected(fetcher);
    classifier
}

/// Req 5.1 + 4 (F4a): a successful miss-path discovery resolves via the
/// async fetch, caches the resolution, and the next resolve is served
/// from the TTL cache with zero additional fetch calls.
#[tokio::test]
async fn discovery_success_populates_cache_and_cache_hit_skips_refetch() {
    let fetcher = Arc::new(CountingFetcher::listing(&["jev-1.12.0", "jev-1.13.0"]));
    let classifier = classifier_with_fetch(Arc::clone(&fetcher));

    // Miss → real fetch → Auto resolution cached.
    let first = classifier.resolve_discovery().await.unwrap();
    assert_eq!(first.model, "jev-1.13.0");
    assert_eq!(first.origin, ResolutionOrigin::Auto);
    assert_eq!(fetcher.calls(), 1);

    // Cache hit inside the TTL: no refetch.
    let second = classifier.resolve_discovery().await.unwrap();
    assert_eq!(second.model, "jev-1.13.0");
    assert_eq!(second.origin, ResolutionOrigin::Cached);
    assert_eq!(fetcher.calls(), 1);
}

/// Req 5.1 + 4 (F4c): a failed miss-path discovery negative-caches the
/// typed failure; a subsequent resolve within the TTL errors again
/// WITHOUT a refetch.
#[tokio::test]
async fn discovery_failure_is_negatively_cached_within_ttl() {
    let fetcher = Arc::new(CountingFetcher::failing("endpoint unreachable"));
    let classifier = classifier_with_fetch(Arc::clone(&fetcher));

    let first = classifier.resolve_discovery().await;
    assert!(matches!(first, Err(DiscoveryError::Listing(_))));
    assert_eq!(fetcher.calls(), 1);

    // Served from the negative cache: same typed error, no refetch.
    let second = classifier.resolve_discovery().await;
    assert!(matches!(second, Err(DiscoveryError::Listing(_))));
    assert_eq!(fetcher.calls(), 1);
}

/// F3 (via F4d): concurrent resolves on a cache miss never observe the
/// probe's transient sentinel failure — the per-classifier gate
/// serializes probe → invalidate → fetch → replay, so every caller sees
/// either the real resolution or the real (negative-)cached outcome, and
/// the real fetch fires exactly once.
#[tokio::test]
async fn concurrent_misses_never_observe_probe_sentinel() {
    let fetcher = Arc::new(CountingFetcher::listing(&["jev-1.13.0"]));
    let classifier = Arc::new(classifier_with_fetch(Arc::clone(&fetcher)));

    let mut resolves = Vec::new();
    for _ in 0..8 {
        let classifier = Arc::clone(&classifier);
        resolves.push(tokio::spawn(async move {
            classifier.resolve_discovery().await
        }));
    }
    let outcomes = futures::future::join_all(resolves).await;

    // Exactly one real fetch; every caller resolved the real model and
    // never saw the probe's `Listing("probe")` sentinel.
    assert_eq!(fetcher.calls(), 1);
    for outcome in outcomes {
        let resolved = outcome.unwrap();
        assert_eq!(resolved.unwrap().model, "jev-1.13.0");
    }
}


    #[test]
    fn fingerprint_ignores_tool_messages() {
        let mut with_tool = request(json!("same"));
        with_tool.messages.push(Message {
            role: "tool".to_string(),
            content: Value::String("different tool output".to_string()),
            extra: Map::new(),
        });
        let without_tool = request(json!("same"));
        assert_eq!(
            request_fingerprint(&with_tool),
            request_fingerprint(&without_tool)
        );
        assert_ne!(
            request_fingerprint(&request(json!("same"))),
            request_fingerprint(&request(json!("different")))
        );
    }

    #[test]
    fn cache_is_lru_bounded_and_ttl_aware() {
        let cache = ScoreCache::new(2, Duration::from_secs(60));
        let detail = |score| CachedClassification {
            score,
            task_type: TaskType::General,
            confidence: 0.9,
            blended: false,
        };
        let now = Instant::now();
        let mut state = cache.lock();
        state.insert(1, detail(0.1), now, Duration::from_secs(60), 2);
        state.insert(2, detail(0.2), now, Duration::from_secs(60), 2);
        assert_eq!(state.get(1, now).map(|d| d.score), Some(0.1));
        state.insert(3, detail(0.3), now, Duration::from_secs(60), 2);
        assert_eq!(state.get(2, now), None);
        assert_eq!(state.get(1, now).map(|d| d.score), Some(0.1));
        assert_eq!(state.get(3, now).map(|d| d.score), Some(0.3));
        assert_eq!(state.get(1, now + Duration::from_secs(61)), None);
    }
}
