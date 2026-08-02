//! Explicit and provider-assisted extraction for persistent memories.
//!
//! This module deliberately has no router dependency. Callers translate their
//! request messages into [`ExtractionMessage`] values and invoke explicit
//! extraction only after the response has been delivered. Automatic extraction
//! uses an injected provider trait rather than recursively calling the public
//! HTTP API.

use std::collections::HashSet;
use std::ops::Range;
use std::sync::{Arc, LazyLock};
use std::time::Duration;

use async_trait::async_trait;
use regex::Regex;
use tokio::sync::Semaphore;
use tokio::task::JoinHandle;
use uuid::Uuid;

use crate::compression::protection::{ProtectedRange, ProtectionScanner};

use super::metrics::{MemoryMetrics, StoreMethod};
use super::sensitive::{SensitiveContentScanner, SensitiveScanOptions};
use super::store::{MemoryStore, NewMemoryEntry};
use super::{MemoryEntry, MemoryError, MemoryType, ResolvedNamespace};

const MIN_MEMORY_CHARS: usize = 5;
const MAX_MEMORY_CHARS: usize = 4_096;
pub const MEMORY_EXTRACTION_INTERNAL_TAG: &str = "memory_extraction";

static TRIGGER_RE: LazyLock<Regex> = LazyLock::new(|| {
    Regex::new(
r"(?i)\b(my preference is|remember this|keep in mind|always use|never use|save this|note that|i prefer)\b",
)
.expect("memory trigger regex must compile")
});

static DECISION_RE: LazyLock<Regex> = LazyLock::new(|| {
    Regex::new(r"(?i)\b(we decided|the approach is|going with|let(?:'|’)s use|the plan is)\b")
        .expect("compression decision regex must compile")
});

static PREFERENCE_RE: LazyLock<Regex> = LazyLock::new(|| {
    Regex::new(r"(?i)\b(i prefer|my preference is|always use|never use)\b")
        .expect("compression preference regex must compile")
});

static BARE_UNIX_PATH_RE: LazyLock<Regex> = LazyLock::new(|| {
    Regex::new(r#"(?:^|[\s(\[{'"`])(?P<path>(?:\.{0,2}/)?(?:[A-Za-z0-9_.-]+/)+[A-Za-z0-9_.-]+)"#)
        .expect("bare Unix path regex must compile")
});

const MAX_COMPRESSION_CANDIDATES: usize = 10;

/// Message roles understood by extraction.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ExtractionRole {
    User,
    Assistant,
    Other,
}

/// Origin distinguishes caller conversation from gateway-injected wrappers.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ExtractionMessageOrigin {
    Caller,
    Injected,
}

/// Router-independent conversation message used by both extraction paths.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ExtractionMessage {
    pub role: ExtractionRole,
    pub content: String,
    pub origin: ExtractionMessageOrigin,
}

impl ExtractionMessage {
    pub fn caller(role: ExtractionRole, content: impl Into<String>) -> Self {
        Self {
            role,
            content: content.into(),
            origin: ExtractionMessageOrigin::Caller,
        }
    }

    pub fn injected(role: ExtractionRole, content: impl Into<String>) -> Self {
        Self {
            role,
            content: content.into(),
            origin: ExtractionMessageOrigin::Injected,
        }
    }
}

/// Per-call storage controls resolved by the parent orchestration layer.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct ExtractionPolicy {
    pub allow_sensitive_storage: bool,
    pub max_memories_per_namespace: usize,
}

impl Default for ExtractionPolicy {
    fn default() -> Self {
        Self {
            allow_sensitive_storage: false,
            max_memories_per_namespace: 1_000,
        }
    }
}

/// Observable result shared by explicit and automatic extraction.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub struct ExtractionCounts {
    pub stored: u32,
    pub rejected: u32,
    pub sensitive_rejected: u32,
    pub duplicates_skipped: u32,
}

/// One structured memory returned by an internal extraction provider.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct StructuredMemoryCandidate {
    pub content: String,
    pub memory_type: MemoryType,
}

/// Parent-supplied message state on either side of one compression pass.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct CompressionMessageSnapshot<'a> {
    pub message_id: &'a str,
    pub content: &'a str,
    pub tokens: u32,
}

/// Parent-supplied accounting for a message removed or reduced by compression.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct CompressionRemovalReport<'a> {
    pub message_id: &'a str,
    pub tokens_before: u32,
    pub tokens_after: u32,
}

/// Borrowed input to the lock-free compression heuristic.
#[derive(Debug, Clone, Copy)]
pub struct CompressionExtractionInput<'a> {
    pub before: &'a [CompressionMessageSnapshot<'a>],
    pub after: &'a [CompressionMessageSnapshot<'a>],
    pub removals: &'a [CompressionRemovalReport<'a>],
}

/// Provider call payload. The tag and bypass bit must be preserved by adapters.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct MemoryExtractionProviderRequest {
    pub provider: String,
    pub model: String,
    pub messages: Vec<ExtractionMessage>,
    pub internal_tag: &'static str,
    pub bypass_memory: bool,
}

/// Sanitized provider failure. Adapters must not include conversation content.
#[derive(Debug, Clone, PartialEq, Eq, thiserror::Error)]
#[error("{message}")]
pub struct MemoryExtractionProviderError {
    pub message: String,
}

/// Internal, mockable provider boundary for structured fact extraction.
#[async_trait]
pub trait MemoryExtractionProvider: Send + Sync {
    async fn extract(
        &self,
        request: MemoryExtractionProviderRequest,
    ) -> Result<Vec<StructuredMemoryCandidate>, MemoryExtractionProviderError>;
}

/// Owned input for an automatic extraction job.
#[derive(Debug, Clone)]
pub struct AsyncExtractionRequest {
    pub enabled: bool,
    pub provider: String,
    pub model: String,
    pub minimum_turns: u32,
    pub messages: Vec<ExtractionMessage>,
    pub namespace: ResolvedNamespace,
    pub source_request_id: Option<Uuid>,
    pub policy: ExtractionPolicy,
}

/// Reason an automatic extraction job was not queued.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum AsyncExtractionSkipReason {
    Disabled,
    InsufficientTurns,
}

/// Completion state of a queued automatic extraction job.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum AsyncExtractionOutcome {
    Completed(ExtractionCounts),
    ProviderFailed,
    TimedOut,
}

/// Scheduling result. Ignoring a spawned handle detaches the owned task.
pub enum AsyncExtractionSchedule {
    NotScheduled(AsyncExtractionSkipReason),
    Spawned(JoinHandle<AsyncExtractionOutcome>),
}

/// Extracts and persists explicit or provider-produced memory candidates.
pub struct MemoryExtractor {
    store: Arc<MemoryStore>,
    sensitive_scanner: Arc<SensitiveContentScanner>,
    provider: Arc<dyn MemoryExtractionProvider>,
    provider_semaphore: Arc<Semaphore>,
    provider_timeout: Duration,
    metrics: Arc<MemoryMetrics>,
    stored_entry_callback: Option<Arc<dyn Fn(MemoryEntry) + Send + Sync>>,
}

impl MemoryExtractor {
    pub fn new(
        store: Arc<MemoryStore>,
        sensitive_scanner: Arc<SensitiveContentScanner>,
        provider: Arc<dyn MemoryExtractionProvider>,
        maximum_concurrent_provider_calls: usize,
        provider_timeout: Duration,
    ) -> Result<Self, MemoryError> {
        Self::with_metrics(
            store,
            sensitive_scanner,
            provider,
            maximum_concurrent_provider_calls,
            provider_timeout,
            Arc::new(MemoryMetrics::new()),
        )
    }

    pub(crate) fn with_metrics(
        store: Arc<MemoryStore>,
        sensitive_scanner: Arc<SensitiveContentScanner>,
        provider: Arc<dyn MemoryExtractionProvider>,
        maximum_concurrent_provider_calls: usize,
        provider_timeout: Duration,
        metrics: Arc<MemoryMetrics>,
    ) -> Result<Self, MemoryError> {
        if maximum_concurrent_provider_calls == 0 {
            return Err(MemoryError::Config(
                "maximum concurrent memory extraction calls must be at least 1".to_owned(),
            ));
        }
        if provider_timeout.is_zero() {
            return Err(MemoryError::Config(
                "memory extraction provider timeout must be greater than zero".to_owned(),
            ));
        }
        Ok(Self {
            store,
            sensitive_scanner,
            provider,
            provider_semaphore: Arc::new(Semaphore::new(maximum_concurrent_provider_calls)),
            provider_timeout,
            metrics,
            stored_entry_callback: None,
        })
    }

    pub(crate) fn with_stored_entry_callback(
        mut self,
        callback: Arc<dyn Fn(MemoryEntry) + Send + Sync>,
    ) -> Self {
        self.stored_entry_callback = Some(callback);
        self
    }

    /// Scan caller-authored user messages for all defined explicit triggers.
    ///
    /// Sensitive scanning always completes before the synchronous SQLite write.
    pub async fn extract_explicit(
        &self,
        messages: &[ExtractionMessage],
        namespace: &ResolvedNamespace,
        source_request_id: Option<Uuid>,
        policy: ExtractionPolicy,
    ) -> Result<ExtractionCounts, MemoryError> {
        let candidates = messages
            .iter()
            .filter(|message| {
                message.role == ExtractionRole::User
                    && message.origin == ExtractionMessageOrigin::Caller
            })
            .flat_map(|message| explicit_candidates(&message.content))
            .collect::<Vec<_>>();

        self.store_candidates(
            candidates,
            namespace,
            source_request_id,
            policy,
            StoreMethod::Explicit,
        )
        .await
    }

    /// Extract candidates from messages removed or reduced by more than 50%.
    ///
    /// This method performs no database or async work and is safe to call while the
    /// parent compression hook holds its own locks. The returned candidates must be
    /// persisted only after those locks are released.
    pub fn compression_candidates(
        &self,
        input: CompressionExtractionInput<'_>,
    ) -> Result<Vec<StructuredMemoryCandidate>, MemoryError> {
        let scanner = ProtectionScanner::new()
            .map_err(|error| MemoryError::Extraction(format!("protection scanner: {error}")))?;
        Ok(compression_candidates_with_scanner(input, &scanner))
    }

    /// Persist previously collected compression candidates after lock release.
    pub async fn persist_compression_candidates(
        &self,
        candidates: Vec<StructuredMemoryCandidate>,
        namespace: &ResolvedNamespace,
        source_request_id: Option<Uuid>,
        policy: ExtractionPolicy,
    ) -> Result<ExtractionCounts, MemoryError> {
        self.store_candidates(
            candidates,
            namespace,
            source_request_id,
            policy,
            StoreMethod::Heuristic,
        )
        .await
    }

    /// Queue owned automatic extraction work after response delivery.
    ///
    /// The caller controls ordering by invoking this method only after delivery.
    /// The returned handle is optional for production callers and useful to tests.
    pub fn spawn_after_delivery(
        &self,
        mut request: AsyncExtractionRequest,
    ) -> AsyncExtractionSchedule {
        if !request.enabled {
            return AsyncExtractionSchedule::NotScheduled(AsyncExtractionSkipReason::Disabled);
        }

        request.messages.retain(|message| {
            matches!(
                message.role,
                ExtractionRole::User | ExtractionRole::Assistant
            ) && message.origin == ExtractionMessageOrigin::Caller
        });
        let qualifying_turns = request.messages.len() as u32;
        if qualifying_turns < request.minimum_turns {
            return AsyncExtractionSchedule::NotScheduled(
                AsyncExtractionSkipReason::InsufficientTurns,
            );
        }

        let store = self.store.clone();
        let scanner = self.sensitive_scanner.clone();
        let provider = self.provider.clone();
        let metrics = self.metrics.clone();
        let stored_entry_callback = self.stored_entry_callback.clone();
        let semaphore = self.provider_semaphore.clone();
        let provider_timeout = self.provider_timeout;

        AsyncExtractionSchedule::Spawned(tokio::spawn(async move {
            let permit = match semaphore.acquire_owned().await {
                Ok(permit) => permit,
                Err(_) => {
                    tracing::warn!("memory extraction semaphore closed");
                    return AsyncExtractionOutcome::ProviderFailed;
                }
            };
            let provider_request = MemoryExtractionProviderRequest {
                provider: request.provider,
                model: request.model,
                messages: request.messages,
                internal_tag: MEMORY_EXTRACTION_INTERNAL_TAG,
                bypass_memory: true,
            };
            let provider_result =
                tokio::time::timeout(provider_timeout, provider.extract(provider_request)).await;
            drop(permit);

            let candidates = match provider_result {
                Err(_) => {
                    tracing::warn!("memory extraction provider timed out");
                    return AsyncExtractionOutcome::TimedOut;
                }
                Ok(Err(error)) => {
                    tracing::warn!(error = %error, "memory extraction provider failed");
                    return AsyncExtractionOutcome::ProviderFailed;
                }
                Ok(Ok(candidates)) => candidates,
            };

            match store_candidates(
                &store,
                &scanner,
                &metrics,
                candidates,
                &request.namespace,
                request.source_request_id,
                request.policy,
                StoreMethod::AsyncLlm,
                stored_entry_callback.as_deref(),
            )
            .await
            {
                Ok(counts) => AsyncExtractionOutcome::Completed(counts),
                Err(error) => {
                    tracing::warn!(error = %error, "memory extraction storage failed");
                    AsyncExtractionOutcome::ProviderFailed
                }
            }
        }))
    }

    async fn store_candidates(
        &self,
        candidates: Vec<StructuredMemoryCandidate>,
        namespace: &ResolvedNamespace,
        source_request_id: Option<Uuid>,
        policy: ExtractionPolicy,
        method: StoreMethod,
    ) -> Result<ExtractionCounts, MemoryError> {
        store_candidates(
            &self.store,
            &self.sensitive_scanner,
            &self.metrics,
            candidates,
            namespace,
            source_request_id,
            policy,
            method,
            self.stored_entry_callback.as_deref(),
        )
        .await
    }
}

pub(crate) fn compression_candidates_with_scanner(
    input: CompressionExtractionInput<'_>,
    scanner: &ProtectionScanner,
) -> Vec<StructuredMemoryCandidate> {
    let after_by_id = input
        .after
        .iter()
        .map(|snapshot| (snapshot.message_id, snapshot))
        .collect::<std::collections::HashMap<_, _>>();
    let removal_by_id = input
        .removals
        .iter()
        .map(|report| (report.message_id, report))
        .collect::<std::collections::HashMap<_, _>>();
    let mut ranked = Vec::new();
    let mut sequence = 0usize;

    for before in input.before {
        let (tokens_before, tokens_after) = removal_by_id
            .get(before.message_id)
            .map(|report| (report.tokens_before, report.tokens_after))
            .unwrap_or_else(|| {
                (
                    before.tokens,
                    after_by_id
                        .get(before.message_id)
                        .map(|snapshot| snapshot.tokens)
                        .unwrap_or(0),
                )
            });
        if tokens_before == 0 || tokens_after.saturating_mul(2) >= tokens_before {
            continue;
        }

        let protected = scanner.scan(before.content);
        let code_ranges = code_block_ranges(before.content);
        for range in &code_ranges {
            push_ranked_candidate(
                &mut ranked,
                &mut sequence,
                0,
                MemoryType::Context,
                &before.content[range.clone()],
            );
        }
        for range in protected {
            if code_ranges.iter().any(|code| ranges_overlap(code, &range)) {
                continue;
            }
            let text = before.content[range].trim();
            if looks_like_path(text) {
                push_ranked_candidate(&mut ranked, &mut sequence, 3, MemoryType::Context, text);
            }
        }
        for matched in DECISION_RE.find_iter(before.content) {
            let sentence = sentence_containing(before.content, matched.start(), matched.end());
            push_ranked_candidate(
                &mut ranked,
                &mut sequence,
                1,
                MemoryType::Decision,
                sentence,
            );
        }
        for matched in PREFERENCE_RE.find_iter(before.content) {
            let sentence = sentence_containing(before.content, matched.start(), matched.end());
            push_ranked_candidate(
                &mut ranked,
                &mut sequence,
                2,
                MemoryType::Preference,
                sentence,
            );
        }
        for captures in BARE_UNIX_PATH_RE.captures_iter(before.content) {
            if let Some(path) = captures.name("path") {
                push_ranked_candidate(
                    &mut ranked,
                    &mut sequence,
                    3,
                    MemoryType::Context,
                    path.as_str(),
                );
            }
        }
    }

    ranked.sort_by_key(|candidate| (candidate.priority, candidate.sequence));
    let mut unique = HashSet::new();
    ranked
        .into_iter()
        .filter(|candidate| unique.insert(candidate.candidate.content.clone()))
        .take(MAX_COMPRESSION_CANDIDATES)
        .map(|candidate| candidate.candidate)
        .collect()
}

struct RankedCompressionCandidate {
    priority: u8,
    sequence: usize,
    candidate: StructuredMemoryCandidate,
}

fn push_ranked_candidate(
    ranked: &mut Vec<RankedCompressionCandidate>,
    sequence: &mut usize,
    priority: u8,
    memory_type: MemoryType,
    content: &str,
) {
    let content = content.trim();
    if (MIN_MEMORY_CHARS..=MAX_MEMORY_CHARS).contains(&content.chars().count()) {
        ranked.push(RankedCompressionCandidate {
            priority,
            sequence: *sequence,
            candidate: StructuredMemoryCandidate {
                content: content.to_owned(),
                memory_type,
            },
        });
        *sequence = sequence.saturating_add(1);
    }
}

fn code_block_ranges(content: &str) -> Vec<Range<usize>> {
    let fence = Regex::new(
        r"(?ms)^ {0,3}(?P<marker>`{3,}|~{3,})[^\r\n]*\r?\n.*?^ {0,3}(?:`{3,}|~{3,})[^\r\n]*",
    )
    .expect("code fence regex must compile");
    let indented =
        Regex::new(r"(?m)(?:^ {4}[^\r\n]*(?:\r?\n|$))+").expect("indented code regex must compile");
    let mut ranges = fence
        .find_iter(content)
        .chain(indented.find_iter(content))
        .map(|matched| matched.range())
        .collect::<Vec<_>>();
    ranges.sort_by_key(|range| range.start);
    ranges
}

fn sentence_containing(content: &str, start: usize, end: usize) -> &str {
    let sentence_start = content[..start]
        .rfind(['.', '!', '?', '\n', '\r'])
        .map(|index| index + 1)
        .unwrap_or(0);
    let sentence_end = content[end..]
        .char_indices()
        .find_map(|(offset, character)| {
            matches!(character, '.' | '!' | '?' | '\n' | '\r')
                .then_some(end + offset + character.len_utf8())
        })
        .unwrap_or(content.len());
    content[sentence_start..sentence_end].trim()
}

fn looks_like_path(text: &str) -> bool {
    text.starts_with('/')
        || text.starts_with("./")
        || text.starts_with("../")
        || text.as_bytes().get(1) == Some(&b':')
        || (text.contains('/') && !text.contains(char::is_whitespace))
}

fn ranges_overlap(left: &Range<usize>, right: &ProtectedRange) -> bool {
    left.start < right.end && right.start < left.end
}

pub(crate) fn explicit_candidates(message: &str) -> Vec<StructuredMemoryCandidate> {
    let matches = TRIGGER_RE.find_iter(message).collect::<Vec<_>>();
    if let Some(trigger) = matches
        .first()
        .filter(|trigger| message[..trigger.start()].trim().is_empty())
    {
        return vec![StructuredMemoryCandidate {
            content: message.trim().to_owned(),
            memory_type: classify_trigger(trigger.as_str()),
        }];
    }

    matches
        .into_iter()
        .map(|trigger| {
            let sentence_end = message[trigger.end()..]
                .char_indices()
                .find_map(|(offset, character)| {
                    matches!(character, '.' | '!' | '?' | '\n' | '\r')
                        .then_some(trigger.end() + offset + character.len_utf8())
                })
                .unwrap_or(message.len());
            StructuredMemoryCandidate {
                content: message[trigger.start()..sentence_end].trim().to_owned(),
                memory_type: classify_trigger(trigger.as_str()),
            }
        })
        .collect()
}

fn classify_trigger(trigger: &str) -> MemoryType {
    if trigger.eq_ignore_ascii_case("i prefer")
        || trigger.eq_ignore_ascii_case("always use")
        || trigger.eq_ignore_ascii_case("never use")
        || trigger.eq_ignore_ascii_case("my preference is")
    {
        MemoryType::Preference
    } else if trigger.eq_ignore_ascii_case("save this") {
        MemoryType::Context
    } else {
        MemoryType::Fact
    }
}

fn namespace_for_type<'a>(namespace: &'a ResolvedNamespace, memory_type: MemoryType) -> &'a str {
    match memory_type {
        MemoryType::Preference => &namespace.user_scope,
        MemoryType::Fact | MemoryType::Context | MemoryType::Decision => namespace
            .context_scope
            .as_deref()
            .unwrap_or(&namespace.user_scope),
    }
}

async fn store_candidates(
    store: &MemoryStore,
    scanner: &SensitiveContentScanner,
    metrics: &MemoryMetrics,
    candidates: Vec<StructuredMemoryCandidate>,
    namespace: &ResolvedNamespace,
    source_request_id: Option<Uuid>,
    policy: ExtractionPolicy,
    method: StoreMethod,
    stored_entry_callback: Option<&(dyn Fn(MemoryEntry) + Send + Sync)>,
) -> Result<ExtractionCounts, MemoryError> {
    let mut counts = ExtractionCounts::default();

    for candidate in candidates {
        let content = candidate.content.trim();
        let character_count = content.chars().count();
        if !(MIN_MEMORY_CHARS..=MAX_MEMORY_CHARS).contains(&character_count) {
            counts.rejected = counts.rejected.saturating_add(1);
            tracing::warn!(
                character_count,
                "discarding memory extraction outside content length limits"
            );
            continue;
        }

        let target_namespace = namespace_for_type(namespace, candidate.memory_type);
        if store.find_duplicate(target_namespace, content)?.is_some() {
            counts.duplicates_skipped = counts.duplicates_skipped.saturating_add(1);
            continue;
        }

        let scan = scanner
            .scan_with_options(
                content,
                SensitiveScanOptions {
                    allow_sensitive_storage: policy.allow_sensitive_storage,
                    ..SensitiveScanOptions::default()
                },
            )
            .await
            .map_err(|error| MemoryError::Extraction(error.to_string()))?;
        if scan.contains_sensitive {
            counts.rejected = counts.rejected.saturating_add(1);
            counts.sensitive_rejected = counts.sensitive_rejected.saturating_add(1);
            tracing::warn!(
                namespace = target_namespace,
                sensitive_match_count = scan.match_count,
                sensitive_sources = ?scan.sources,
                "discarding sensitive memory extraction"
            );
            continue;
        }

        let entry = store.store_entry(
            NewMemoryEntry {
                namespace: target_namespace.to_owned(),
                content: content.to_owned(),
                memory_type: candidate.memory_type,
                source_request_id,
            },
            Some(policy.max_memories_per_namespace),
        )?;
        if let Some(callback) = stored_entry_callback {
            callback(entry);
        }
        counts.stored = counts.stored.saturating_add(1);
        metrics.record_store(method);
    }

    Ok(counts)
}

#[cfg(test)]
mod tests {
    use std::path::Path;
    use std::sync::atomic::{AtomicUsize, Ordering};

    use tokio::sync::Notify;

    use super::*;

    struct MockProvider {
        response: MockResponse,
        calls: AtomicUsize,
        active: AtomicUsize,
        maximum_active: AtomicUsize,
        entered: Notify,
        release: Notify,
    }

    enum MockResponse {
        Candidates(Vec<StructuredMemoryCandidate>),
        Failure,
        Block,
    }

    impl MockProvider {
        fn new(response: MockResponse) -> Self {
            Self {
                response,
                calls: AtomicUsize::new(0),
                active: AtomicUsize::new(0),
                maximum_active: AtomicUsize::new(0),
                entered: Notify::new(),
                release: Notify::new(),
            }
        }

        fn record_active(&self, active: usize) {
            let mut observed = self.maximum_active.load(Ordering::SeqCst);
            while active > observed {
                match self.maximum_active.compare_exchange(
                    observed,
                    active,
                    Ordering::SeqCst,
                    Ordering::SeqCst,
                ) {
                    Ok(_) => break,
                    Err(current) => observed = current,
                }
            }
        }
    }

    #[async_trait]
    impl MemoryExtractionProvider for MockProvider {
        async fn extract(
            &self,
            request: MemoryExtractionProviderRequest,
        ) -> Result<Vec<StructuredMemoryCandidate>, MemoryExtractionProviderError> {
            assert_eq!(request.internal_tag, MEMORY_EXTRACTION_INTERNAL_TAG);
            assert!(request.bypass_memory);
            assert!(request.messages.iter().all(|message| {
                matches!(
                    message.role,
                    ExtractionRole::User | ExtractionRole::Assistant
                ) && message.origin == ExtractionMessageOrigin::Caller
            }));
            self.calls.fetch_add(1, Ordering::SeqCst);
            match &self.response {
                MockResponse::Candidates(candidates) => Ok(candidates.clone()),
                MockResponse::Failure => Err(MemoryExtractionProviderError {
                    message: "mock provider unavailable".to_owned(),
                }),
                MockResponse::Block => {
                    let active = self.active.fetch_add(1, Ordering::SeqCst) + 1;
                    self.record_active(active);
                    self.entered.notify_one();
                    self.release.notified().await;
                    self.active.fetch_sub(1, Ordering::SeqCst);
                    Ok(Vec::new())
                }
            }
        }
    }

    fn namespace() -> ResolvedNamespace {
        ResolvedNamespace {
            user_scope: "user::test".to_owned(),
            context_scope: Some("user::test::project::abc".to_owned()),
        }
    }

    fn test_extractor(
        provider: Arc<MockProvider>,
        concurrency: usize,
        timeout: Duration,
    ) -> (MemoryExtractor, Arc<MemoryStore>) {
        let store = Arc::new(MemoryStore::new(Path::new(":memory:")).unwrap());
        let scanner = Arc::new(SensitiveContentScanner::new(&[], None).unwrap());
        let extractor =
            MemoryExtractor::new(store.clone(), scanner, provider, concurrency, timeout).unwrap();
        (extractor, store)
    }

    fn async_request(messages: Vec<ExtractionMessage>) -> AsyncExtractionRequest {
        AsyncExtractionRequest {
            enabled: true,
            provider: "internal-provider".to_owned(),
            model: "extractor-model".to_owned(),
            minimum_turns: 2,
            messages,
            namespace: namespace(),
            source_request_id: Some(Uuid::from_u128(7)),
            policy: ExtractionPolicy::default(),
        }
    }

    async fn await_spawned(schedule: AsyncExtractionSchedule) -> AsyncExtractionOutcome {
        match schedule {
            AsyncExtractionSchedule::Spawned(handle) => handle.await.unwrap(),
            AsyncExtractionSchedule::NotScheduled(reason) => {
                panic!("expected spawned extraction, got {reason:?}")
            }
        }
    }

    #[tokio::test]
    async fn explicit_triggers_are_case_insensitive_and_classified() {
        let provider = Arc::new(MockProvider::new(MockResponse::Candidates(Vec::new())));
        let (extractor, store) = test_extractor(provider, 1, Duration::from_secs(1));
        let messages = vec![
            ExtractionMessage::caller(
                ExtractionRole::User,
                "I PREFER tabs and concise diagnostics.",
            ),
            ExtractionMessage::caller(
                ExtractionRole::User,
                "Before the task, REMEMBER THIS repository targets Rust. SaVe ThIs build context.",
            ),
            ExtractionMessage::caller(ExtractionRole::Assistant, "always use ignored"),
            ExtractionMessage::injected(ExtractionRole::User, "never use injected wrappers"),
        ];

        let counts = extractor
            .extract_explicit(&messages, &namespace(), None, ExtractionPolicy::default())
            .await
            .unwrap();

        assert_eq!(counts.stored, 3);
        let user_entries = store.list_entries("user::test", 10, 0).unwrap().entries;
        assert_eq!(user_entries.len(), 1);
        assert_eq!(user_entries[0].memory_type, MemoryType::Preference);
        assert_eq!(user_entries[0].content, messages[0].content);
        let context_entries = store
            .list_entries("user::test::project::abc", 10, 0)
            .unwrap()
            .entries;
        assert!(context_entries.iter().any(|entry| {
            entry.memory_type == MemoryType::Fact
                && entry.content == "REMEMBER THIS repository targets Rust."
        }));
        assert!(context_entries.iter().any(|entry| {
            entry.memory_type == MemoryType::Context && entry.content == "SaVe ThIs build context."
        }));
    }

    #[tokio::test]
    async fn explicit_extraction_rejects_sensitive_and_skips_exact_duplicates() {
        let provider = Arc::new(MockProvider::new(MockResponse::Candidates(Vec::new())));
        let (extractor, store) = test_extractor(provider, 1, Duration::from_secs(1));
        let messages = vec![
            ExtractionMessage::caller(
                ExtractionRole::User,
                "Please remember this URL https://user:password@example.invalid.",
            ),
            ExtractionMessage::caller(ExtractionRole::User, "Note that stable fact."),
            ExtractionMessage::caller(ExtractionRole::User, "Note that stable fact."),
        ];

        let first = extractor
            .extract_explicit(&messages, &namespace(), None, ExtractionPolicy::default())
            .await
            .unwrap();
        assert_eq!(first.stored, 1);
        assert_eq!(first.rejected, 1);
        assert_eq!(first.sensitive_rejected, 1);
        assert_eq!(first.duplicates_skipped, 1);

        let second = extractor
            .extract_explicit(&messages, &namespace(), None, ExtractionPolicy::default())
            .await
            .unwrap();
        assert_eq!(second.stored, 0);
        assert_eq!(second.rejected, 1);
        assert_eq!(second.duplicates_skipped, 2);
        assert_eq!(
            store.namespace_count("user::test::project::abc").unwrap(),
            1
        );
    }

    #[tokio::test]
    async fn async_threshold_filters_roles_and_provider_candidates_use_namespaces() {
        let provider = Arc::new(MockProvider::new(MockResponse::Candidates(vec![
            StructuredMemoryCandidate {
                content: "Prefer concise diagnostics".to_owned(),
                memory_type: MemoryType::Preference,
            },
            StructuredMemoryCandidate {
                content: "The project uses SQLite".to_owned(),
                memory_type: MemoryType::Fact,
            },
        ])));
        let (extractor, store) = test_extractor(provider.clone(), 1, Duration::from_secs(1));
        let mut request = async_request(vec![
            ExtractionMessage::caller(ExtractionRole::Other, "system wrapper"),
            ExtractionMessage::injected(ExtractionRole::User, "memory wrapper"),
            ExtractionMessage::caller(ExtractionRole::User, "question"),
        ]);
        assert!(matches!(
            extractor.spawn_after_delivery(request.clone()),
            AsyncExtractionSchedule::NotScheduled(AsyncExtractionSkipReason::InsufficientTurns)
        ));

        request.messages.push(ExtractionMessage::caller(
            ExtractionRole::Assistant,
            "answer",
        ));
        let outcome = await_spawned(extractor.spawn_after_delivery(request)).await;
        assert_eq!(
            outcome,
            AsyncExtractionOutcome::Completed(ExtractionCounts {
                stored: 2,
                ..ExtractionCounts::default()
            })
        );
        assert_eq!(provider.calls.load(Ordering::SeqCst), 1);
        assert_eq!(store.namespace_count("user::test").unwrap(), 1);
        assert_eq!(
            store.namespace_count("user::test::project::abc").unwrap(),
            1
        );
    }

    #[tokio::test]
    async fn async_provider_failure_and_timeout_are_non_fatal() {
        let failure = Arc::new(MockProvider::new(MockResponse::Failure));
        let (failure_extractor, _) = test_extractor(failure, 1, Duration::from_secs(1));
        let messages = vec![
            ExtractionMessage::caller(ExtractionRole::User, "question"),
            ExtractionMessage::caller(ExtractionRole::Assistant, "answer"),
        ];
        assert_eq!(
            await_spawned(failure_extractor.spawn_after_delivery(async_request(messages.clone())))
                .await,
            AsyncExtractionOutcome::ProviderFailed
        );

        let blocked = Arc::new(MockProvider::new(MockResponse::Block));
        let (timeout_extractor, _) = test_extractor(blocked, 1, Duration::from_millis(20));
        assert_eq!(
            await_spawned(timeout_extractor.spawn_after_delivery(async_request(messages))).await,
            AsyncExtractionOutcome::TimedOut
        );
    }

    #[tokio::test]
    async fn automatic_extraction_is_background_owned_and_concurrency_bounded() {
        let provider = Arc::new(MockProvider::new(MockResponse::Block));
        let (extractor, _) = test_extractor(provider.clone(), 1, Duration::from_secs(2));
        let messages = vec![
            ExtractionMessage::caller(ExtractionRole::User, "question"),
            ExtractionMessage::caller(ExtractionRole::Assistant, "answer"),
        ];

        let first = match extractor.spawn_after_delivery(async_request(messages.clone())) {
            AsyncExtractionSchedule::Spawned(handle) => handle,
            AsyncExtractionSchedule::NotScheduled(reason) => panic!("not spawned: {reason:?}"),
        };
        let second = match extractor.spawn_after_delivery(async_request(messages)) {
            AsyncExtractionSchedule::Spawned(handle) => handle,
            AsyncExtractionSchedule::NotScheduled(reason) => panic!("not spawned: {reason:?}"),
        };

        tokio::time::timeout(Duration::from_millis(200), provider.entered.notified())
            .await
            .unwrap();
        tokio::task::yield_now().await;
        assert_eq!(provider.calls.load(Ordering::SeqCst), 1);
        assert_eq!(provider.maximum_active.load(Ordering::SeqCst), 1);

        provider.release.notify_one();
        tokio::time::timeout(Duration::from_millis(200), provider.entered.notified())
            .await
            .unwrap();
        assert_eq!(provider.calls.load(Ordering::SeqCst), 2);
        assert_eq!(provider.maximum_active.load(Ordering::SeqCst), 1);
        provider.release.notify_one();

        assert!(matches!(
            first.await.unwrap(),
            AsyncExtractionOutcome::Completed(_)
        ));
        assert!(matches!(
            second.await.unwrap(),
            AsyncExtractionOutcome::Completed(_)
        ));
    }
}

#[cfg(test)]
#[path = "property_tests.rs"]
mod property_tests;
