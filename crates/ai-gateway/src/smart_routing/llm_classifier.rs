//! LLM-backed complexity classification with a small content-free SimHash cache.

use std::collections::{HashMap, VecDeque};
use std::fmt;
use std::sync::{Arc, Mutex, MutexGuard};
use std::time::{Duration, Instant};

use async_trait::async_trait;
use serde_json::Value;

use crate::loop_detection::simhash;
use crate::models::openai::{Message, OpenAIRequest};
use crate::smart_routing::{
    ClassifierFailure, ClassifierInput, ClassifierOutput, OptionalClassifier,
};

pub const DEFAULT_CLASSIFIER_TIMEOUT: Duration = Duration::from_millis(2_000);
pub const CLASSIFIER_CACHE_CAPACITY: usize = 1_000;
pub const CLASSIFIER_CACHE_TTL: Duration = Duration::from_secs(5 * 60);
pub const MAX_CLASSIFIER_PROMPT_TOKENS: usize = 100;
pub const CLASSIFIER_INTERNAL_TAG: &str = "smart-routing-llm-classifier";

const MAX_CLASSIFIER_SOURCE_CHARS: usize = 1_024;
const MAX_CLASSIFIER_MODEL_CHARS: usize = 256;
const CLASSIFIER_MAX_OUTPUT_TOKENS: u16 = 4;
const CLASSIFIER_PROMPT_PREFIX: &str = "Classify the request complexity. Reply with exactly one label: SIMPLE, MODERATE, or COMPLEX. Consider reasoning depth, tools, code, math, and required steps.\nRequest:\n";

/// Fixed, content-free metadata that prevents an internal classifier call from
/// recursively entering request-processing features.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct ClassifierBypassMetadata {
    pub internal_tag: &'static str,
    pub bypass_smart_routing: bool,
    pub bypass_semantic_cache: bool,
    pub bypass_cascade: bool,
    pub bypass_loop_detection: bool,
    pub bypass_memory: bool,
}

impl ClassifierBypassMetadata {
    pub const fn internal() -> Self {
        Self {
            internal_tag: CLASSIFIER_INTERNAL_TAG,
            bypass_smart_routing: true,
            bypass_semantic_cache: true,
            bypass_cascade: true,
            bypass_loop_detection: true,
            bypass_memory: true,
        }
    }
}

/// Owned request passed to the injected classifier adapter.
///
/// The prompt is intentionally short-lived. Implementations must not log or
/// retain it. The model is always supplied by `LlmClassifier`, never by the
/// request being classified.
#[derive(Clone)]
pub struct ClassifierRequest {
    pub model: String,
    pub prompt: String,
    pub max_output_tokens: u16,
    pub temperature: f32,
    pub metadata: ClassifierBypassMetadata,
}

impl fmt::Debug for ClassifierRequest {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter
            .debug_struct("ClassifierRequest")
            .field("model", &self.model)
            .field("prompt", &"<redacted>")
            .field("max_output_tokens", &self.max_output_tokens)
            .field("temperature", &self.temperature)
            .field("metadata", &self.metadata)
            .finish()
    }
}

/// Bounded labels accepted from a classifier adapter.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ClassifierLabel {
    Simple,
    Moderate,
    Complex,
}

impl ClassifierLabel {
    pub fn parse(value: &str) -> Result<Self, ClassifierInvokerError> {
        match value.trim() {
            value if value.eq_ignore_ascii_case("SIMPLE") => Ok(Self::Simple),
            value if value.eq_ignore_ascii_case("MODERATE") => Ok(Self::Moderate),
            value if value.eq_ignore_ascii_case("COMPLEX") => Ok(Self::Complex),
            _ => Err(ClassifierInvokerError::InvalidOutput),
        }
    }

    pub const fn score(self) -> f64 {
        match self {
            Self::Simple => 0.15,
            Self::Moderate => 0.50,
            Self::Complex => 0.85,
        }
    }
}

/// Bounded classifier response. Raw provider output is parsed by the adapter
/// and never enters the cache.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct ClassifierResponse {
    pub label: ClassifierLabel,
}

impl ClassifierResponse {
    pub fn parse(value: &str) -> Result<Self, ClassifierInvokerError> {
        Ok(Self {
            label: ClassifierLabel::parse(value)?,
        })
    }
}

/// Content-free invocation failures suitable for orchestrator fallback.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ClassifierInvokerError {
    Unavailable,
    Timeout,
    InvalidOutput,
    Backend,
}

impl From<ClassifierInvokerError> for ClassifierFailure {
    fn from(error: ClassifierInvokerError) -> Self {
        match error {
            ClassifierInvokerError::Unavailable => Self::Unavailable,
            ClassifierInvokerError::Timeout => Self::Timeout,
            ClassifierInvokerError::InvalidOutput => Self::InvalidOutput,
            ClassifierInvokerError::Backend => Self::Backend,
        }
    }
}

/// Injectable async boundary implemented by the parent provider adapter.
///
/// Implementations must honor the internal bypass metadata and must not route
/// this request recursively through `SmartRouter`.
#[async_trait]
pub trait ClassifierInvoker: Send + Sync {
    async fn invoke(
        &self,
        request: ClassifierRequest,
    ) -> Result<ClassifierResponse, ClassifierInvokerError>;
}

#[derive(Debug, Clone, Copy)]
struct CacheEntry {
    score: f64,
    expires_at: Instant,
}

#[derive(Debug)]
struct CacheState {
    entries: HashMap<u64, CacheEntry>,
    lru: VecDeque<u64>,
}

impl CacheState {
    fn new(capacity: usize) -> Self {
        Self {
            entries: HashMap::with_capacity(capacity),
            lru: VecDeque::with_capacity(capacity),
        }
    }

    fn get(&mut self, fingerprint: u64, now: Instant) -> Option<f64> {
        let entry = self.entries.get(&fingerprint).copied()?;
        if entry.expires_at <= now {
            self.remove(fingerprint);
            return None;
        }

        self.touch(fingerprint);
        Some(entry.score)
    }

    fn insert(
        &mut self,
        fingerprint: u64,
        score: f64,
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
                score,
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

#[derive(Debug)]
struct ClassifierCache {
    state: Mutex<CacheState>,
    capacity: usize,
    ttl: Duration,
}

impl ClassifierCache {
    fn new() -> Self {
        Self::with_limits(CLASSIFIER_CACHE_CAPACITY, CLASSIFIER_CACHE_TTL)
    }

    fn with_limits(capacity: usize, ttl: Duration) -> Self {
        Self {
            state: Mutex::new(CacheState::new(capacity)),
            capacity,
            ttl,
        }
    }

    fn get(&self, fingerprint: u64) -> Option<f64> {
        self.lock().get(fingerprint, Instant::now())
    }

    fn insert(&self, fingerprint: u64, score: f64) {
        self.lock()
            .insert(fingerprint, score, Instant::now(), self.ttl, self.capacity);
    }

    fn lock(&self) -> MutexGuard<'_, CacheState> {
        self.state
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner)
    }
}

/// LLM-backed optional classifier with a fixed model and exact-SimHash LRU.
pub struct LlmClassifier {
    classifier_model: String,
    invoker: Arc<dyn ClassifierInvoker>,
    timeout: Duration,
    cache: ClassifierCache,
}

impl LlmClassifier {
    pub fn new(classifier_model: impl Into<String>, invoker: Arc<dyn ClassifierInvoker>) -> Self {
        let classifier_model = classifier_model.into();
        assert!(
            !classifier_model.trim().is_empty(),
            "classifier model must not be empty"
        );
        assert!(
            classifier_model.chars().count() <= MAX_CLASSIFIER_MODEL_CHARS,
            "classifier model exceeds {MAX_CLASSIFIER_MODEL_CHARS} characters"
        );

        Self {
            classifier_model,
            invoker,
            timeout: DEFAULT_CLASSIFIER_TIMEOUT,
            cache: ClassifierCache::new(),
        }
    }

    pub fn with_timeout(mut self, timeout: Duration) -> Self {
        assert!(
            !timeout.is_zero(),
            "classifier timeout must be greater than zero"
        );
        self.timeout = timeout;
        self
    }

    pub fn classifier_model(&self) -> &str {
        &self.classifier_model
    }

    fn classifier_request(&self, request: &OpenAIRequest) -> ClassifierRequest {
        ClassifierRequest {
            model: self.classifier_model.clone(),
            prompt: build_classifier_prompt(request),
            max_output_tokens: CLASSIFIER_MAX_OUTPUT_TOKENS,
            temperature: 0.0,
            metadata: ClassifierBypassMetadata::internal(),
        }
    }
}

#[async_trait]
impl OptionalClassifier for LlmClassifier {
    async fn classify(
        &self,
        input: ClassifierInput<'_>,
    ) -> Result<ClassifierOutput, ClassifierFailure> {
        let fingerprint = request_fingerprint(input.request);
        if let Some(score) = self.cache.get(fingerprint) {
            return Ok(ClassifierOutput { score });
        }

        let request = self.classifier_request(input.request);
        let response = tokio::time::timeout(self.timeout, self.invoker.invoke(request))
            .await
            .map_err(|_| ClassifierFailure::Timeout)?
            .map_err(ClassifierFailure::from)?;
        let score = response.label.score();
        self.cache.insert(fingerprint, score);
        Ok(ClassifierOutput { score })
    }
}

fn request_fingerprint(request: &OpenAIRequest) -> u64 {
    let mut fingerprint = simhash::compute(&request.model);
    for message in &request.messages {
        if message.role.eq_ignore_ascii_case("tool") {
            continue;
        }

        fingerprint = mix_fingerprint(fingerprint, simhash::compute(&message.role));
        visit_text_content(message, |text| {
            fingerprint = mix_fingerprint(fingerprint, simhash::compute(text));
        });
    }
    fingerprint
}

fn mix_fingerprint(current: u64, next: u64) -> u64 {
    current.rotate_left(13) ^ next.wrapping_mul(0x9e37_79b9_7f4a_7c15)
}

fn build_classifier_prompt(request: &OpenAIRequest) -> String {
    let mut prompt = String::with_capacity(
        CLASSIFIER_PROMPT_PREFIX.len() + MAX_CLASSIFIER_SOURCE_CHARS.min(256),
    );
    prompt.push_str(CLASSIFIER_PROMPT_PREFIX);

    if let Some(message) = request
        .messages
        .iter()
        .rev()
        .find(|message| message.role.eq_ignore_ascii_case("user"))
    {
        append_text_content(&mut prompt, message, MAX_CLASSIFIER_SOURCE_CHARS);
    } else if let Some(message) = request
        .messages
        .iter()
        .rev()
        .find(|message| !message.role.eq_ignore_ascii_case("tool"))
    {
        append_text_content(&mut prompt, message, MAX_CLASSIFIER_SOURCE_CHARS);
    }

    truncate_prompt_to_token_limit(&mut prompt);
    prompt
}

fn visit_text_content(message: &Message, mut visit: impl FnMut(&str)) {
    match &message.content {
        Value::String(text) => visit(text),
        Value::Array(parts) => {
            for part in parts {
                if part.get("type").and_then(Value::as_str) == Some("tool_result") {
                    continue;
                }
                if let Some(text) = part.get("text").and_then(Value::as_str) {
                    visit(text);
                }
            }
        }
        _ => {}
    }
}

fn append_text_content(target: &mut String, message: &Message, maximum_chars: usize) {
    let mut remaining = maximum_chars;
    visit_text_content(message, |text| {
        if remaining == 0 {
            return;
        }
        let appended = append_chars(target, text, remaining);
        remaining = remaining.saturating_sub(appended);
    });
}

fn append_chars(target: &mut String, source: &str, maximum_chars: usize) -> usize {
    let mut appended = 0;
    for character in source.chars().take(maximum_chars) {
        target.push(character);
        appended += 1;
    }
    appended
}

fn truncate_prompt_to_token_limit(prompt: &mut String) {
    let tokenizer = tiktoken_rs::cl100k_base_singleton();
    let prefix_len = CLASSIFIER_PROMPT_PREFIX.len();

    while tokenizer.encode_with_special_tokens(prompt).len() > MAX_CLASSIFIER_PROMPT_TOKENS
        && prompt.len() > prefix_len
    {
        prompt.pop();
    }

    debug_assert!(
        tokenizer.encode_with_special_tokens(prompt).len() <= MAX_CLASSIFIER_PROMPT_TOKENS
    );
}

#[cfg(test)]
mod tests {
    use std::sync::atomic::{AtomicUsize, Ordering};

    use proptest::prelude::*;
    use proptest::test_runner::Config as ProptestConfig;
    use serde_json::{json, Map};

    use super::*;
    use crate::config::ModelGroup;
    use crate::smart_routing::tier::{ComplexityScore, TaskType};
    use crate::smart_routing::PinnedRoutingContext;

    #[derive(Debug, Clone, Copy)]
    enum MockResult {
        Label(ClassifierLabel),
        Error(ClassifierInvokerError),
    }

    #[derive(Debug, Clone, PartialEq, Eq)]
    struct RequestObservation {
        model: String,
        prompt_tokens: usize,
        metadata: ClassifierBypassMetadata,
        max_output_tokens: u16,
    }

    struct MockInvoker {
        calls: AtomicUsize,
        result: MockResult,
        delay: Duration,
        observation: Mutex<Option<RequestObservation>>,
    }

    impl MockInvoker {
        fn label(label: ClassifierLabel) -> Arc<Self> {
            Arc::new(Self {
                calls: AtomicUsize::new(0),
                result: MockResult::Label(label),
                delay: Duration::ZERO,
                observation: Mutex::new(None),
            })
        }

        fn error(error: ClassifierInvokerError) -> Arc<Self> {
            Arc::new(Self {
                calls: AtomicUsize::new(0),
                result: MockResult::Error(error),
                delay: Duration::ZERO,
                observation: Mutex::new(None),
            })
        }

        fn delayed(label: ClassifierLabel, delay: Duration) -> Arc<Self> {
            Arc::new(Self {
                calls: AtomicUsize::new(0),
                result: MockResult::Label(label),
                delay,
                observation: Mutex::new(None),
            })
        }
    }

    #[async_trait]
    impl ClassifierInvoker for MockInvoker {
        async fn invoke(
            &self,
            request: ClassifierRequest,
        ) -> Result<ClassifierResponse, ClassifierInvokerError> {
            self.calls.fetch_add(1, Ordering::SeqCst);
            let observation = RequestObservation {
                model: request.model,
                prompt_tokens: tiktoken_rs::cl100k_base_singleton()
                    .encode_with_special_tokens(&request.prompt)
                    .len(),
                metadata: request.metadata,
                max_output_tokens: request.max_output_tokens,
            };
            *self
                .observation
                .lock()
                .unwrap_or_else(std::sync::PoisonError::into_inner) = Some(observation);

            if !self.delay.is_zero() {
                tokio::time::sleep(self.delay).await;
            }

            match self.result {
                MockResult::Label(label) => Ok(ClassifierResponse { label }),
                MockResult::Error(error) => Err(error),
            }
        }
    }

    fn request(content: Value) -> OpenAIRequest {
        OpenAIRequest {
            model: "caller-controlled-model".to_owned(),
            messages: vec![Message {
                role: "user".to_owned(),
                content,
                extra: Map::new(),
            }],
            stream: false,
            temperature: None,
            max_tokens: None,
            extra: Map::new(),
        }
    }

    fn model_group() -> ModelGroup {
        ModelGroup {
            name: "test-group".to_owned(),
            version_fallback_enabled: false,
            compression: None,
            memory: None,
            structured_output: None,
            models: Vec::new(),
        }
    }

    fn input<'a>(
        request: &'a OpenAIRequest,
        model_group: &'a ModelGroup,
        pinned_context: &'a PinnedRoutingContext,
    ) -> ClassifierInput<'a> {
        ClassifierInput {
            request,
            model_group,
            pinned_context,
            heuristic_score: ComplexityScore::new(0.5),
            heuristic_task_type: TaskType::General,
        }
    }

    #[test]
    fn labels_parse_strictly_and_map_to_required_scores() {
        assert_eq!(ClassifierLabel::parse(" SIMPLE ").unwrap().score(), 0.15);
        assert_eq!(ClassifierLabel::parse("moderate").unwrap().score(), 0.50);
        assert_eq!(ClassifierLabel::parse("COMPLEX").unwrap().score(), 0.85);
        assert_eq!(
            ClassifierLabel::parse("The request is complex"),
            Err(ClassifierInvokerError::InvalidOutput)
        );
    }

    #[tokio::test]
    async fn sends_fixed_model_bypass_metadata_and_bounded_prompt() {
        let invoker = MockInvoker::label(ClassifierLabel::Moderate);
        let classifier = LlmClassifier::new("fixed-classifier", invoker.clone());
        let request = request(Value::String("é".repeat(2_000)));
        let model_group = model_group();
        let pinned_context = PinnedRoutingContext::default();

        let output = classifier
            .classify(input(&request, &model_group, &pinned_context))
            .await
            .unwrap();

        assert_eq!(output.score, 0.50);
        let observation = invoker
            .observation
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner)
            .clone()
            .unwrap();
        assert_eq!(observation.model, "fixed-classifier");
        assert!(observation.prompt_tokens <= MAX_CLASSIFIER_PROMPT_TOKENS);
        assert_eq!(observation.metadata, ClassifierBypassMetadata::internal());
        assert_eq!(observation.max_output_tokens, CLASSIFIER_MAX_OUTPUT_TOKENS);
    }

    #[tokio::test]
    async fn exact_simhash_cache_hit_avoids_second_invocation() {
        let invoker = MockInvoker::label(ClassifierLabel::Simple);
        let classifier = LlmClassifier::new("fixed-classifier", invoker.clone());
        let request = request(json!("Summarize this short sentence."));
        let model_group = model_group();
        let pinned_context = PinnedRoutingContext::default();

        let first = classifier
            .classify(input(&request, &model_group, &pinned_context))
            .await
            .unwrap();
        let second = classifier
            .classify(input(&request, &model_group, &pinned_context))
            .await
            .unwrap();

        assert_eq!(first, second);
        assert_eq!(invoker.calls.load(Ordering::SeqCst), 1);
    }

    proptest! {
    #![proptest_config(ProptestConfig::with_cases(256))]

    #[test]
    fn property_5_cache_capacity_never_exceeds_1000(
    insertions in prop::collection::vec((any::<u64>(), 0u8..=2), 1_001..=1_200),
    ) {
    let now = Instant::now();
    let mut state = CacheState::new(CLASSIFIER_CACHE_CAPACITY);

    for (fingerprint, score_index) in insertions {
    let score = [0.15, 0.50, 0.85][usize::from(score_index)];
    state.insert(
    fingerprint,
    score,
    now,
    CLASSIFIER_CACHE_TTL,
    CLASSIFIER_CACHE_CAPACITY,
    );
    prop_assert!(state.entries.len() <= CLASSIFIER_CACHE_CAPACITY);
    prop_assert!(state.lru.len() <= CLASSIFIER_CACHE_CAPACITY);
    }
    }

    #[test]
    fn property_6_exact_hit_returns_cached_score_without_invoker_call(
    content in prop::collection::vec(any::<char>(), 0..=256)
    .prop_map(|characters| characters.into_iter().collect::<String>()),
    cached_score_basis_points in 0u16..=10_000,
    ) {
    let invoker = MockInvoker::error(ClassifierInvokerError::Backend);
    let classifier = LlmClassifier::new("fixed-classifier", invoker.clone());
    let request = request(Value::String(content));
    let model_group = model_group();
    let pinned_context = PinnedRoutingContext::default();
    let cached_score = f64::from(cached_score_basis_points) / 10_000.0;
    classifier
    .cache
    .insert(request_fingerprint(&request), cached_score);

    let output = futures::executor::block_on(
    classifier.classify(input(&request, &model_group, &pinned_context)),
    )
    .unwrap();

    prop_assert_eq!(output.score, cached_score);
    prop_assert_eq!(invoker.calls.load(Ordering::SeqCst), 0);
    }
    }

    #[tokio::test]
    async fn timeout_is_typed_for_orchestrator_fallback() {
        let invoker = MockInvoker::delayed(ClassifierLabel::Complex, Duration::from_millis(50));
        let classifier =
            LlmClassifier::new("fixed-classifier", invoker).with_timeout(Duration::from_millis(1));
        let request = request(json!("Solve a difficult proof."));
        let model_group = model_group();
        let pinned_context = PinnedRoutingContext::default();

        assert_eq!(
            classifier
                .classify(input(&request, &model_group, &pinned_context))
                .await,
            Err(ClassifierFailure::Timeout)
        );
    }

    #[tokio::test]
    async fn invoker_failures_remain_bounded_and_typed() {
        for (source, expected) in [
            (
                ClassifierInvokerError::Unavailable,
                ClassifierFailure::Unavailable,
            ),
            (ClassifierInvokerError::Timeout, ClassifierFailure::Timeout),
            (
                ClassifierInvokerError::InvalidOutput,
                ClassifierFailure::InvalidOutput,
            ),
            (ClassifierInvokerError::Backend, ClassifierFailure::Backend),
        ] {
            let classifier = LlmClassifier::new("fixed-classifier", MockInvoker::error(source));
            let request = request(json!("Classify this."));
            let model_group = model_group();
            let pinned_context = PinnedRoutingContext::default();
            assert_eq!(
                classifier
                    .classify(input(&request, &model_group, &pinned_context))
                    .await,
                Err(expected)
            );
        }
    }

    #[test]
    fn cache_is_lru_bounded_and_ttl_aware() {
        let now = Instant::now();
        let mut state = CacheState::new(2);
        state.insert(1, 0.15, now, Duration::from_secs(60), 2);
        state.insert(2, 0.50, now, Duration::from_secs(60), 2);
        assert_eq!(state.get(1, now), Some(0.15));
        state.insert(3, 0.85, now, Duration::from_secs(60), 2);

        assert_eq!(state.get(2, now), None);
        assert_eq!(state.get(1, now), Some(0.15));
        assert_eq!(state.get(3, now), Some(0.85));
        assert_eq!(state.entries.len(), 2);

        assert_eq!(state.get(1, now + Duration::from_secs(61)), None);
        assert_eq!(state.get(3, now + Duration::from_secs(61)), None);
        assert!(state.entries.is_empty());
        assert!(state.lru.is_empty());
    }

    #[test]
    fn request_debug_redacts_content() {
        let request = ClassifierRequest {
            model: "fixed-classifier".to_owned(),
            prompt: "secret raw request content".to_owned(),
            max_output_tokens: CLASSIFIER_MAX_OUTPUT_TOKENS,
            temperature: 0.0,
            metadata: ClassifierBypassMetadata::internal(),
        };

        let debug = format!("{request:?}");
        assert!(debug.contains("<redacted>"));
        assert!(!debug.contains("secret raw request content"));
    }
}
