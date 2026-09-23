//! The Jev classifier facade: the seam that plugs the System One decision
//! model into the smart-routing pipeline as an `OptionalClassifier`.
//!
//! Classification flow: SimHash score cache -> model discovery (TTL-cached) ->
//! System One evaluation under a timeout -> pure composition with confidence
//! gating. Every failure mode maps to the bounded `ClassifierFailure`
//! taxonomy so the orchestrator falls back exactly as it does for the ML and
//! LLM classifiers. No payload content ever reaches logs, metrics, or errors.

use std::collections::{HashMap, VecDeque};
use std::sync::{Arc, Mutex, MutexGuard};
use std::time::{Duration, Instant};

use async_trait::async_trait;

use crate::models::openai::OpenAIRequest;
use crate::smart_routing::config::{JevConfig, JevFallbackPolicy, JevTrustOverride};
use crate::smart_routing::tier::TaskType;
use crate::smart_routing::{
    ClassifierFailure, ClassifierInput, ClassifierOutput, OptionalClassifier,
};

use crate::smart_routing::jev::client::JevClient;
use crate::smart_routing::jev::compose::{self, ComposeResult, JevTrustConfig};
use crate::smart_routing::jev::discovery::{
    DiscoveryError, ModelDiscovery, ModelsFetcher, ResolutionOrigin,
};
use crate::smart_routing::jev::models::SystemOneRequest;
use crate::smart_routing::jev::rubric;

/// Score-cache capacity mirroring `LlmClassifier`.
pub const JEV_SCORE_CACHE_CAPACITY: usize = 1_000;
/// Score-cache TTL mirroring `LlmClassifier`.
pub const JEV_SCORE_CACHE_TTL: Duration = Duration::from_secs(5 * 60);
/// Timeout hard cap per requirements (config may be lower).
const JEV_TIMEOUT_HARD_CAP: Duration = Duration::from_millis(2_000);

/// Extra detail produced by a successful Jev classification, consumed by the
/// orchestrator for telemetry and decision records.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct JevClassificationDetail {
    pub task_type: TaskType,
    pub confidence: f64,
}

/// Adapter bridging the async `JevClient` to the sync `ModelsFetcher`
/// boundary. Runs inside `tokio::task::block_in_place` during resolution so
/// the synchronous discovery cache stays simple.
struct ClientFetcher {
    client: Arc<JevClient>,
}

impl ModelsFetcher for ClientFetcher {
    fn fetch(&self) -> Result<super::models::ModelsListing, String> {
        // Discovery shares the evaluation timeout budget; the caller wraps
        // this in the per-call timeout already.
        futures::executor::block_on(self.client.list_models()).map_err(|error| error.to_string())
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

/// The facade. Construct via [`JevClassifier::new`] after config validation.
pub struct JevClassifier {
    client: Arc<JevClient>,
    discovery: ModelDiscovery,
    fetcher: ClientFetcher,
    config: JevConfig,
    trust: JevTrustConfig,
    timeout: Duration,
    cache: ScoreCache,
}

impl JevClassifier {
    /// Build from validated configuration. The API key must already be
    /// resolved (empty keys are rejected by config validation).
    pub fn new(config: JevConfig) -> Result<Self, ClassifierFailure> {
        let api_key = config
            .resolve_api_key()
            .ok_or(ClassifierFailure::Unavailable)?;
        let timeout = Duration::from_millis(u64::from(config.timeout_ms)).min(JEV_TIMEOUT_HARD_CAP);
        let trust = JevTrustConfig {
            min_confidence: config.min_confidence,
            min_task_confidence: config.min_task_confidence,
            fallback_policy: config.fallback_policy,
        };

        let client = Arc::new(
            JevClient::new(
                &config.base_url,
                &api_key,
                timeout,
                config.retry.max_attempts,
                Duration::from_millis(u64::from(config.retry.backoff_ms)),
            )
            .map_err(|_| ClassifierFailure::Backend)?,
        );
        let discovery = ModelDiscovery::new(&config);
        let fetcher = ClientFetcher {
            client: client.clone(),
        };

        Ok(Self {
            cache: ScoreCache::new(JEV_SCORE_CACHE_CAPACITY, JEV_SCORE_CACHE_TTL),
            client,
            discovery,
            fetcher,
            trust,
            timeout,
            config,
        })
    }

    /// Best-effort startup resolution so the first request does not pay the
    /// discovery cost and operators see the resolved model in logs.
    pub fn prime_discovery(&self) {
        let outcome = tokio::task::block_in_place(|| self.discovery.resolve(&self.fetcher));
        match outcome {
            Ok(resolved) => tracing::info!(
                model = %resolved.model,
                origin = ?resolved.origin,
                "Jev classifier ready"
            ),
            Err(DiscoveryError::NoJevAtEndpoint) => {
                // ModelDiscovery already emitted its rate-limited warning.
            }
            Err(error) => tracing::warn!(
                error = %format!("{error:?}"),
                "Jev classifier model discovery deferred to first request"
            ),
        }
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

    /// Classify with full detail; the trait method is a thin wrapper.
    pub async fn classify_with_detail(
        &self,
        input: ClassifierInput<'_>,
    ) -> Result<(ClassifierOutput, JevClassificationDetail), ClassifierFailure> {
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
                },
                JevClassificationDetail {
                    task_type: cached.task_type,
                    confidence: cached.confidence,
                },
            ));
        }

        let resolved = tokio::task::block_in_place(|| self.discovery.resolve(&self.fetcher))
            .map_err(map_discovery_failure)?;
        let discovery_refreshed = matches!(resolved.origin, ResolutionOrigin::Auto);

        let request = SystemOneRequest {
            state: rubric::build_state(input.request, &self.config),
            model: resolved.model.clone(),
            questions: rubric::build_questions(),
        };

        let response = tokio::time::timeout(self.timeout, self.client.evaluate(&request))
            .await
            .map_err(|_| ClassifierFailure::Timeout)?
            .map_err(map_call_failure)?;

        let result = compose::compose(&response.answers, &self.config.dimension_weights, &trust);
        match result {
            ComposeResult::Ok {
                score,
                task_type,
                confidence,
            } => {
                let detail = JevClassificationDetail {
                    task_type,
                    confidence,
                };
                self.cache.insert(
                    fingerprint,
                    CachedClassification {
                        score: score.value(),
                        task_type,
                        confidence,
                    },
                );
                Ok((
                    ClassifierOutput {
                        score: score.value(),
                        task_type: Some(task_type),
                        confidence: Some(confidence),
                        resolved_model: Some(resolved.model),
                        discovery_refreshed,
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
                        let detail = JevClassificationDetail {
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
                            },
                        );
                        Ok((
                            ClassifierOutput {
                                score: blended.value(),
                                task_type: Some(input.heuristic_task_type),
                                confidence: Some(confidence),
                                resolved_model: Some(resolved.model),
                                discovery_refreshed,
                            },
                            detail,
                        ))
                    }
                    JevFallbackPolicy::Fallback => Err(ClassifierFailure::LowConfidence),
                }
            }
            ComposeResult::Invalid => Err(ClassifierFailure::InvalidOutput),
        }
    }
}

#[async_trait]
impl OptionalClassifier for JevClassifier {
    async fn classify(
        &self,
        input: ClassifierInput<'_>,
    ) -> Result<ClassifierOutput, ClassifierFailure> {
        self.classify_with_detail(input)
            .await
            .map(|(output, _)| output)
    }
}

fn map_discovery_failure(error: DiscoveryError) -> ClassifierFailure {
    match error {
        DiscoveryError::NoJevAtEndpoint => ClassifierFailure::NoJevAtEndpoint,
        DiscoveryError::ModelUnavailable { .. } => ClassifierFailure::Unavailable,
        DiscoveryError::Listing(_) => ClassifierFailure::Unavailable,
    }
}

fn map_call_failure(error: super::client::JevCallError) -> ClassifierFailure {
    match error {
        super::client::JevCallError::Transient { .. } => ClassifierFailure::Backend,
        super::client::JevCallError::Auth { .. } => ClassifierFailure::Backend,
        super::client::JevCallError::BadRequest { .. } => ClassifierFailure::InvalidOutput,
        super::client::JevCallError::Unavailable { .. } => ClassifierFailure::Unavailable,
        super::client::JevCallError::Timeout => ClassifierFailure::Timeout,
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
        super::rubric::visit_message_text(message, |text| {
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
/// different `jev_trust` overrides never share a cached decision.
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

    fn config() -> JevConfig {
        JevConfig {
            api_key: Some("test-key".to_string()),
            ..JevConfig::default()
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
        let mut config = config();
        config.api_key = None;
        config.api_key_env = None;
        assert!(matches!(
            JevClassifier::new(config),
            Err(ClassifierFailure::Unavailable)
        ));
    }

    #[test]
    fn cache_is_lru_bounded_and_ttl_aware() {
        let cache = ScoreCache::new(2, Duration::from_secs(60));
        let detail = |score| CachedClassification {
            score,
            task_type: TaskType::General,
            confidence: 0.9,
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

    #[tokio::test(flavor = "multi_thread")]
    async fn classify_without_network_fails_gracefully() {
        // The default base URL is unreachable in tests; every outcome must be
        // a typed failure, never a panic. Timeout is the expected class since
        // discovery is deferred inside the classify timeout budget.
        let classifier = JevClassifier::new(config()).unwrap();
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

    #[test]
    fn fingerprint_ignores_tool_messages() {
        let mut with_tool = request(json!("same"));
        with_tool.messages.push(Message {
            role: "tool".to_string(),
            content: Value::String("different tool output".to_string()),
            extra: Map::new(),
        });
        let without_tool = request(json!("same"));

        // Equal up to SimHash collision resistance for identical content;
        // deterministic for identical inputs.
        assert_eq!(
            request_fingerprint(&with_tool),
            request_fingerprint(&without_tool)
        );
        assert_ne!(
            request_fingerprint(&request(json!("alpha"))),
            request_fingerprint(&request(json!("beta")))
        );
    }
}
