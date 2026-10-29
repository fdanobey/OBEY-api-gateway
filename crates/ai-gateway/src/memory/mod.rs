//! Core types for the persistent memory store.
//!
//! Feature implementations live in sibling modules. Only modules whose files
//! exist in the current implementation wave are declared here.

pub mod admin;
pub mod config;
pub mod consolidation;
pub mod context_detector;
pub mod decay;
pub mod extraction_adapter;
pub mod extractor;
pub mod injector;
pub mod metrics;
pub mod namespace;
pub mod scoring;
pub mod sensitive;
pub mod store;
pub mod vector;

pub use config::{
    EffectiveMemoryConfig, MemoryConfig, MemoryConfigError, MemoryQdrantConfig,
    MemoryValidationResult, ModelGroupMemoryOverride, ProviderMemoryOverride,
};
pub use context_detector::{sanitize_label, ContextDetector, DetectedContext};
pub use consolidation::{
    group_candidates, parse_decisions, run_consolidation, ConsolidationContext, ConsolidationCounts,
    ConsolidationError, Decision,
};
pub use decay::{ConsolidationHook, DecayScheduler, EvictionEventPublisher, VectorRetryCallback};
pub use extraction_adapter::GatewayExtractionAdapter;
pub use extractor::{
    AsyncExtractionOutcome, AsyncExtractionRequest, AsyncExtractionSchedule,
    AsyncExtractionSkipReason, CompressionExtractionInput, CompressionMessageSnapshot,
    CompressionRemovalReport, ExtractionCounts, ExtractionMessage, ExtractionMessageOrigin,
    ExtractionPolicy, ExtractionRole, MemoryExtractionProvider, MemoryExtractionProviderError,
    MemoryExtractionProviderRequest, MemoryExtractor, StructuredMemoryCandidate,
    MEMORY_EXTRACTION_INTERNAL_TAG,
};
pub use injector::{
    available_budget, format_memories, merge_retrieval_scores, retrieve_with_vector_fallback,
    strip_formatting, MemoryInjector,
};
pub use metrics::{ConsolidationLastPass, MemoryMetrics};
pub use namespace::{sanitize_vk_id, validate_namespace, MAX_NAMESPACE_CHARS, MAX_VK_ID_CHARS};
pub use scoring::{apply_decay, compute_score, recency_boost};
pub use sensitive::{
    PiiProviderErrorKind, PiiProviderFailurePolicy, SensitiveContentScanner, SensitiveMatchSource,
    SensitiveScanError, SensitiveScanOptions, SensitiveScanResult,
};
pub use store::{
    MemoryEntryInput, MemoryEntryPage, MemoryNamespace, MemoryStats, MemoryStore, NewMemoryEntry,
};
pub use vector::{MemoryVectorTier, QdrantMemoryVectorTier, VectorMatch};

#[cfg(test)]
mod property_tests;
#[cfg(test)]
mod store_property_tests;

use std::path::Path;
use std::sync::{Arc, Mutex, Once};
use std::time::Duration;

use async_trait::async_trait;
use chrono::{DateTime, Utc};
use serde::{Deserialize, Serialize};
use tokio::sync::RwLock;
use uuid::Uuid;

use crate::compression::token_counter::TokenCounter;
use crate::guardrail::provider::GuardrailProvider;
use crate::models::openai::OpenAIRequest;

use decay::DisabledConsolidationHook;

/// Errors produced by memory configuration, persistence, and processing.
#[derive(Debug, thiserror::Error)]
pub enum MemoryError {
    /// A SQLite persistence operation failed.
    #[error("memory store error: {0}")]
    Store(#[from] rusqlite::Error),

    /// Memory content exceeded the allowed character count.
    #[error("memory content is too long: {length} characters (maximum {max})")]
    ContentTooLong { length: usize, max: usize },

    /// Memory content did not meet the minimum character count.
    #[error("memory content is too short: {length} characters (minimum {min})")]
    ContentTooShort { length: usize, min: usize },

    /// Memory configuration is invalid.
    #[error("memory configuration error: {0}")]
    Config(String),

    /// The request conflicts with the current state of the store.
    ///
    /// Carries a sanitized, client-safe static message (no internal detail)
    /// and maps to HTTP 409 at the admin layer, matching the conflict-status
    /// convention of the supersede endpoint.
    #[error("{0}")]
    Conflict(&'static str),

    /// An optional Qdrant operation failed.
    #[error("memory Qdrant error: {0}")]
    Qdrant(String),

    /// Memory extraction failed.
    #[error("memory extraction error: {0}")]
    Extraction(String),

    /// A blocking store task was cancelled or panicked before completing.
    ///
    /// Synchronous SQLite work is dispatched to the blocking thread pool so it
    /// cannot occupy an async worker; this is distinct from a SQL-level failure.
    #[error("memory blocking task error: {0}")]
    TaskFailed(String),
}

/// Run a blocking memory-store operation on the blocking thread pool.
///
/// [`MemoryStore`] holds a `std::sync::Mutex<Connection>` and every method does
/// synchronous SQLite I/O, including FTS5 scans and write transactions. Awaiting
/// those inline parks a Tokio worker for the lock wait plus the query; because a
/// single connection serializes all callers, sufficient concurrency can occupy
/// every worker and stall the gateway. Mirrors the decay module's `run_cycle`,
/// which already dispatched its store work this way.
async fn spawn_store_task<T, F>(task: F) -> Result<T, MemoryError>
where
    F: FnOnce() -> Result<T, MemoryError> + Send + 'static,
    T: Send + 'static,
{
    match tokio::task::spawn_blocking(task).await {
        Ok(result) => result,
        Err(error) => Err(MemoryError::TaskFailed(error.to_string())),
    }
}

/// Administrative creation was rejected before or during persistence.
#[derive(Debug, thiserror::Error)]
pub enum AdminCreateError {
    #[error("namespace must use non-empty ASCII segments separated by '::'")]
    InvalidNamespace,
    #[error("{0}")]
    InvalidContent(String),
    #[error("content contains sensitive information")]
    SensitiveContent,
    #[error("sensitive-content scan failed: {0}")]
    Scan(String),
    #[error(transparent)]
    Memory(#[from] MemoryError),
}

/// Minimum whitespace-separated words for admin-edited and
/// consolidation-merged content, mirroring the extraction-grade rule in
/// `extractor.rs` (`MIN_MEMORY_WORDS`, private there). Single words and bare
/// paths carry no reusable meaning on their own.
pub(crate) const MIN_MEMORY_WORD_COUNT: usize = 3;

/// Shared extraction-grade word-count check for admin-edited and
/// consolidation-merged content; character limits are enforced separately by
/// each caller.
pub(crate) fn validate_memory_content_words(content: &str) -> Result<(), AdminCreateError> {
    if content.split_whitespace().count() < MIN_MEMORY_WORD_COUNT {
        return Err(AdminCreateError::InvalidContent(
            "content must be at least 3 words".to_owned(),
        ));
    }
    Ok(())
}

/// Memory classification used for namespace assignment and decay behavior.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum MemoryType {
    Preference,
    Fact,
    Context,
    Decision,
}

/// A single persistent memory record.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct MemoryEntry {
    pub id: Uuid,
    pub namespace: String,
    pub content: String,
    pub memory_type: MemoryType,
    pub relevance_score: f64,
    pub created_at: DateTime<Utc>,
    pub last_accessed_at: DateTime<Utc>,
    pub access_count: u64,
    pub source_request_id: Option<Uuid>,
    /// When the entry content was last edited; `None` for never-edited entries.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub updated_at: Option<i64>,
    /// Entry that superseded this one; `None` while the entry is active.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub superseded_by: Option<Uuid>,
}

/// Context classification produced by automatic context detection.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum ContextType {
    /// A project detected from file paths, carrying its 16-character hash.
    Project(String),
    /// An agent detected from a system prompt, carrying its 16-character hash.
    Agent(String),
    /// User-level fallback with relevance-only retrieval.
    User,
}

/// User and optional context scopes used for isolated retrieval.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ResolvedNamespace {
    pub user_scope: String,
    pub context_scope: Option<String>,
}

/// Placement strategy for recalled memories in an outgoing request.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum InjectionStrategy {
    SystemPromptPrefix,
    SyntheticMessage,
}

/// A ranked memory candidate with its estimated injection cost.
#[derive(Debug, Clone, PartialEq)]
pub struct ScoredMemory {
    pub entry: MemoryEntry,
    pub final_score: f64,
    pub estimated_tokens: u32,
}

/// Memory operation metadata collected for a single request cycle.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub struct InjectionResult {
    pub memories_injected: u32,
    pub injection_tokens: u32,
    pub memories_stored: u32,
    pub sensitive_rejected: u32,
}

/// Result of one request-side memory orchestration pass.
#[derive(Debug, Clone, PartialEq)]
pub struct MemoryRequestResult {
    pub context: ContextType,
    pub namespace: ResolvedNamespace,
    pub injection: InjectionResult,
}

/// Result of response-side explicit extraction and background scheduling.
pub struct MemoryResponseResult {
    pub extraction: ExtractionCounts,
    pub automatic_extraction: AsyncExtractionSchedule,
}

struct DisabledExtractionProvider;

#[async_trait]
impl MemoryExtractionProvider for DisabledExtractionProvider {
    async fn extract(
        &self,
        _request: MemoryExtractionProviderRequest,
    ) -> Result<Vec<StructuredMemoryCandidate>, MemoryExtractionProviderError> {
        Err(MemoryExtractionProviderError {
            message: "automatic memory extraction provider is disabled".to_owned(),
        })
    }
}

static SENSITIVE_STORAGE_WARNING: Once = Once::new();

/// Runs one bounded consolidation pass after each decay cycle (requirement
/// 3.6) and backs the on-demand admin trigger (requirement 3.8).
struct MemoryConsolidationRunner {
    store: Arc<MemoryStore>,
    provider: Arc<dyn MemoryExtractionProvider>,
    scanner: Arc<SensitiveContentScanner>,
    vector_tier: Arc<RwLock<Option<Arc<dyn MemoryVectorTier>>>>,
    config: Arc<RwLock<MemoryConfig>>,
    metrics: Arc<MemoryMetrics>,
    /// Serializes passes: the post-decay hook and the manual admin trigger
    /// share one gate so two concurrent passes can never read the same
    /// active sources and double-merge them.
    pass_gate: Arc<tokio::sync::Mutex<()>>,
}

impl MemoryConsolidationRunner {
    /// Single construction path for the startup hook, the hot-reload hook,
    /// and the manual admin trigger so the three call sites cannot drift.
    fn new(
        store: Arc<MemoryStore>,
        provider: Arc<dyn MemoryExtractionProvider>,
        scanner: Arc<SensitiveContentScanner>,
        vector_tier: Arc<RwLock<Option<Arc<dyn MemoryVectorTier>>>>,
        config: Arc<RwLock<MemoryConfig>>,
        metrics: Arc<MemoryMetrics>,
        pass_gate: Arc<tokio::sync::Mutex<()>>,
    ) -> Self {
        Self {
            store,
            provider,
            scanner,
            vector_tier,
            config,
            metrics,
            pass_gate,
        }
    }

    /// Convenience wrapper mirroring [`Self::new`] for callers holding a
    /// [`MemorySystem`] reference, such as the manual admin trigger and the
    /// hot-reload hook replacement.
    fn from_system(system: &MemorySystem, pass_gate: Arc<tokio::sync::Mutex<()>>) -> Self {
        Self::new(
            Arc::clone(&system.store),
            Arc::clone(&system.extraction_provider),
            Arc::clone(&system.sensitive_scanner),
            Arc::clone(&system.vector_tier),
            Arc::clone(&system.config),
            Arc::clone(&system.metrics),
            pass_gate,
        )
    }

    /// Build the context for one pass from the current config snapshot.
    ///
    /// `None` when consolidation is disabled (background passes only — a
    /// manual trigger passes `require_enabled = false` so an explicit
    /// operator request runs even while background consolidation is off) or
    /// when the provider/model pair does not resolve through the requirement
    /// 3.7 fallback chain.
    async fn context_for_pass(&self, require_enabled: bool) -> Option<ConsolidationContext> {
        let config = self.config.read().await.clone();
        if require_enabled && !config.consolidation_enabled {
            return None;
        }
        let Some((provider_name, model)) = config.resolve_consolidation_pair() else {
            return None;
        };
        let vector_tier = self.vector_tier.read().await.clone();
        Some(ConsolidationContext {
            store: Arc::clone(&self.store),
            provider: Arc::clone(&self.provider),
            scanner: Arc::clone(&self.scanner),
            vector_tier,
            allow_sensitive_storage: config.allow_sensitive_storage,
            provider_name,
            model,
            maximum_concurrent_provider_calls: 1,
            provider_timeout: Duration::from_secs(120),
        })
    }

    /// Run one pass, record its counts (requirement 3.6), and log completion.
    async fn execute_pass(
        &self,
        context: ConsolidationContext,
    ) -> Result<ConsolidationCounts, MemoryError> {
        let counts = run_consolidation(context).await?;
        tracing::info!(
            merged = counts.merged,
            superseded = counts.superseded,
            kept = counts.kept,
            skipped_batches = counts.skipped_batches,
            provider_errors = counts.provider_errors,
            "memory consolidation pass completed"
        );
        self.metrics.record_consolidation_pass(counts);
        Ok(counts)
    }
}

#[async_trait]
impl ConsolidationHook for MemoryConsolidationRunner {
    async fn run_consolidation_pass(&self) {
        // Hold the shared pass gate for the whole pass so the manual admin
        // trigger cannot start a concurrent pass over the same sources.
        let _gate = self.pass_gate.lock().await;
        let Some(context) = self.context_for_pass(true).await else {
            return;
        };
        if let Err(error) = self.execute_pass(context).await {
            tracing::warn!(error = %error, "memory consolidation pass failed")
        }
    }
}

struct MemoryVectorRetryCallback {
    store: Arc<MemoryStore>,
    vector_tier: Arc<RwLock<Option<Arc<dyn MemoryVectorTier>>>>,
}

#[async_trait]
impl VectorRetryCallback for MemoryVectorRetryCallback {
    async fn retry_pending(&self) -> Result<u64, MemoryError> {
        let Some(tier) = self.vector_tier.read().await.clone() else {
            return Ok(0);
        };
        let _ = self.store.mark_unconfigured_vector_entries_pending()?;
        let candidates = self.store.list_pending_vector_retries(Utc::now())?;
        let mut indexed = 0_u64;
        for candidate in candidates {
            let attempted_at = Utc::now();
            match tokio::time::timeout(vector::VECTOR_INDEX_TIMEOUT, tier.index(&candidate.entry))
                .await
            {
                Ok(Ok(())) => {
                    self.store
                        .mark_vector_indexed(candidate.entry.id, attempted_at)?;
                    indexed = indexed.saturating_add(1);
                }
                Ok(Err(error)) => mark_vector_retry(
                    &self.store,
                    candidate.entry.id,
                    attempted_at,
                    &error.to_string(),
                ),
                Err(_) => mark_vector_retry(
                    &self.store,
                    candidate.entry.id,
                    attempted_at,
                    "vector indexing timed out",
                ),
            }
        }
        Ok(indexed)
    }
}

/// Coordinates persistent storage, context detection, injection, extraction, and decay.
pub struct MemorySystem {
    pub store: Arc<MemoryStore>,
    pub context_detector: Arc<ContextDetector>,
    pub injector: Arc<MemoryInjector>,
    pub extractor: Arc<MemoryExtractor>,
    pub metrics: Arc<MemoryMetrics>,
    pub config: Arc<RwLock<MemoryConfig>>,
    vector_tier: Arc<RwLock<Option<Arc<dyn MemoryVectorTier>>>>,
    vector_index_semaphore: Arc<tokio::sync::Semaphore>,
    sensitive_scanner: Arc<SensitiveContentScanner>,
    decay_scheduler: Mutex<DecayScheduler>,
    extraction_provider: Arc<dyn MemoryExtractionProvider>,
    extraction_provider_available: bool,
    /// Shared consolidation pass gate: one holder at a time across the
    /// post-decay hook and the manual admin trigger.
    consolidation_pass_gate: Arc<tokio::sync::Mutex<()>>,
}

impl MemorySystem {
    /// Initializes SQLite and starts the decay scheduler.
    ///
    /// The extraction provider may be omitted only while automatic extraction is
    /// disabled. Wave 10's gateway adapter can be supplied through this boundary
    /// without coupling the memory module to router dispatch.
    pub async fn new(
        config: MemoryConfig,
        pii_provider: Option<Arc<dyn GuardrailProvider>>,
        extraction_provider: Option<Arc<dyn MemoryExtractionProvider>>,
    ) -> Result<Self, MemoryError> {
        Self::new_with_vector(config, pii_provider, extraction_provider, None).await
    }

    pub async fn new_with_vector(
        config: MemoryConfig,
        pii_provider: Option<Arc<dyn GuardrailProvider>>,
        extraction_provider: Option<Arc<dyn MemoryExtractionProvider>>,
        vector_tier: Option<Arc<dyn MemoryVectorTier>>,
    ) -> Result<Self, MemoryError> {
        validate_config(&config)?;
        if config.auto_extract_enabled && extraction_provider.is_none() {
            return Err(MemoryError::Config(
                "auto_extract_enabled requires a memory extraction provider adapter".to_owned(),
            ));
        }
        warn_sensitive_storage_once(&config);

        let metrics = Arc::new(MemoryMetrics::new());
        let store = Arc::new(MemoryStore::with_metrics(
            Path::new(&config.database_path),
            metrics.clone(),
        )?);
        let context_detector = Arc::new(ContextDetector::new(config.default_prompts.clone()));
        let injector = Arc::new(MemoryInjector::with_metrics(
            store.clone(),
            TokenCounter::new(),
            metrics.clone(),
        ));
        let scanner = Arc::new(
            SensitiveContentScanner::new(&config.custom_sensitive_patterns, pii_provider)
                .map_err(|error| MemoryError::Config(error.to_string()))?,
        );
        let extraction_provider_available = extraction_provider.is_some();
        let provider = extraction_provider.unwrap_or_else(|| Arc::new(DisabledExtractionProvider));
        let vector_tier = Arc::new(RwLock::new(vector_tier));
        let vector_index_semaphore = Arc::new(tokio::sync::Semaphore::new(4));
        let callback_store = store.clone();
        let callback_tier = vector_tier.clone();
        let callback_semaphore = vector_index_semaphore.clone();
        let extractor = Arc::new(
            MemoryExtractor::with_metrics(
                store.clone(),
                scanner.clone(),
                Arc::clone(&provider),
                1,
                Duration::from_secs(30),
                metrics.clone(),
            )?
            .with_stored_entry_callback(Arc::new(move |entry| {
                schedule_vector_index_task(
                    callback_store.clone(),
                    callback_tier.clone(),
                    callback_semaphore.clone(),
                    entry,
                );
            })),
        );
        let config = Arc::new(RwLock::new(config));
        let consolidation_pass_gate = Arc::new(tokio::sync::Mutex::new(()));
        let consolidation_hook: Arc<dyn ConsolidationHook> =
            if config.read().await.consolidation_enabled {
                Arc::new(MemoryConsolidationRunner::new(
                    Arc::clone(&store),
                    Arc::clone(&provider),
                    Arc::clone(&scanner),
                    Arc::clone(&vector_tier),
                    Arc::clone(&config),
                    Arc::clone(&metrics),
                    Arc::clone(&consolidation_pass_gate),
                ))
            } else {
                Arc::new(DisabledConsolidationHook)
            };
        let mut decay_scheduler = DecayScheduler::with_parts(
            (*store).clone(),
            config.read().await.decay_schedule_hours,
            config.read().await.max_memories_per_namespace as usize,
            metrics.clone(),
            consolidation_hook,
        )?;
        decay_scheduler.set_vector_retry_callback(Arc::new(MemoryVectorRetryCallback {
            store: store.clone(),
            vector_tier: vector_tier.clone(),
        }));

        Ok(Self {
            store,
            context_detector,
            injector,
            extractor,
            metrics,
            config,
            vector_tier,
            vector_index_semaphore,
            sensitive_scanner: scanner.clone(),
            decay_scheduler: Mutex::new(decay_scheduler),
            extraction_provider: provider,
            extraction_provider_available,
            consolidation_pass_gate,
        })
    }

    /// Validates and synchronously stores one administrator-created entry.
    pub async fn admin_create(
        &self,
        namespace: String,
        content: String,
        memory_type: MemoryType,
    ) -> Result<MemoryEntry, AdminCreateError> {
        if !validate_namespace(&namespace) {
            return Err(AdminCreateError::InvalidNamespace);
        }
        let content_length = content.chars().count();
        if content_length < 5 {
            return Err(AdminCreateError::InvalidContent(
                "content must be at least 5 characters".to_owned(),
            ));
        }
        if content_length > 4096 {
            return Err(AdminCreateError::InvalidContent(
                "content must be at most 4096 characters".to_owned(),
            ));
        }

        let config = self.config.read().await.clone();
        let scan = self
            .sensitive_scanner
            .scan_with_options(
                &content,
                SensitiveScanOptions {
                    allow_sensitive_storage: config.allow_sensitive_storage,
                    ..SensitiveScanOptions::default()
                },
            )
            .await
            .map_err(|error| AdminCreateError::Scan(error.to_string()))?;
        if scan.contains_sensitive {
            return Err(AdminCreateError::SensitiveContent);
        }

        let entry = self
            .store
            .store_entry(
                NewMemoryEntry {
                    namespace,
                    content,
                    memory_type,
                    source_request_id: None,
                },
                Some(config.max_memories_per_namespace as usize),
            )
            .map_err(AdminCreateError::Memory)?;
        self.schedule_vector_index(entry.clone()).await;
        Ok(entry)
    }

    /// Validates and synchronously edits one stored entry, then re-indexes it.
    ///
    /// The same length and sensitive-content screening applied to
    /// [`MemorySystem::admin_create`] runs before any write; on success the
    /// old vector point is removed and the updated content is re-indexed.
    /// Returns `Ok(None)` when `id` does not match a stored entry; editing an
    /// entry that a consolidation merge already superseded is rejected with a
    /// conflict error (the entry is dead weight and must not be re-indexed);
    /// vector work is best-effort and never fails the store commit.
    pub async fn admin_edit(
        &self,
        id: Uuid,
        content: &str,
        memory_type: &str,
    ) -> Result<Option<MemoryEntry>, AdminCreateError> {
        let content_length = content.chars().count();
        if content_length < 5 {
            return Err(AdminCreateError::InvalidContent(
                "content must be at least 5 characters".to_owned(),
            ));
        }
    if content_length > 4096 {
        return Err(AdminCreateError::InvalidContent(
            "content must be at most 4096 characters".to_owned(),
        ));
    }
    validate_memory_content_words(content)?;

    let config = self.config.read().await.clone();
    let scan = self
        .sensitive_scanner
        .scan_with_options(
            content,
            SensitiveScanOptions {
                allow_sensitive_storage: config.allow_sensitive_storage,
                ..SensitiveScanOptions::default()
            },
        )
        .await
        .map_err(|error| AdminCreateError::Scan(error.to_string()))?;
    if scan.contains_sensitive {
        return Err(AdminCreateError::SensitiveContent);
    }

    // Reject superseded entries before any write: fetch and update run
        // in one blocking task against the same connection so a concurrent
        // supersession cannot slip between the check and the update; the
        // conflict error maps to 409 at the admin layer (the entry is dead
        // weight and must not be re-indexed).
        let store = Arc::clone(&self.store);
        let content = content.to_owned();
        let memory_type = memory_type.to_owned();
        let updated = match spawn_store_task(move || {
            let existing = store.get_entry_by_id(id)?;
            match existing {
                None => Ok(None),
                Some(entry) => {
                if entry.superseded_by.is_some() {
                    return Err(MemoryError::Conflict(
                        "memory entry is already superseded",
                    ));
                }
                    store.update_entry(id, &content, &memory_type)
                }
            }
        })
        .await
        {
            Ok(updated) => updated,
            Err(error) => return Err(AdminCreateError::Memory(error)),
        };
        match updated {
            None => Ok(None),
            Some(entry) => {
                self.schedule_vector_delete(entry.id).await;
                self.schedule_vector_index(entry.clone()).await;
                Ok(Some(entry))
            }
        }
    }

    /// Deletes one entry, then removes its vector point best-effort.
    ///
    /// Returns `Ok(false)` when `id` does not match a stored entry; vector
    /// work never fails the store commit.
    pub async fn admin_delete(&self, id: Uuid) -> Result<bool, MemoryError> {
        let store = Arc::clone(&self.store);
        let deleted = spawn_store_task(move || store.delete_entry(id)).await?;
        if deleted {
            self.schedule_vector_delete(id).await;
        }
        Ok(deleted)
    }

    /// Clears a namespace, then removes its vector points best-effort.
    ///
    /// Vector work never fails the store commit.
 pub async fn admin_clear(&self, namespace: &str) -> Result<u64, MemoryError> {
 let store = Arc::clone(&self.store);
 let namespace_for_store = namespace.to_owned();
 let deleted =
 spawn_store_task(move || store.delete_namespace(&namespace_for_store)).await?;
 self.schedule_vector_delete_namespace(namespace.to_owned())
 .await;
 Ok(deleted)
 }

    /// Runs one consolidation pass immediately and returns its counts (Req
    /// 3.8), sharing the context construction, execution, and metric recording
    /// of the post-decay background hook (Req 3.6) so both paths cannot drift.
    ///
    /// A manual operator trigger runs even while `consolidation_enabled` is
    /// false, as long as the provider/model pair resolves; it is the caller's
    /// responsibility to answer 503 when the pair does not resolve. When a
    /// background (or other manual) pass is already running, this answers a
    /// conflict error instead of racing it; the sanitized `Conflict` message
    /// maps to HTTP 409 at the admin layer without leaking any internal
    /// detail.
    pub async fn run_consolidation_now(&self) -> Result<ConsolidationCounts, MemoryError> {
        let runner =
            MemoryConsolidationRunner::from_system(self, Arc::clone(&self.consolidation_pass_gate));
        let Ok(_gate) = runner.pass_gate.try_lock() else {
            return Err(MemoryError::Conflict("consolidation already in progress"));
        };
        let context = runner
            .context_for_pass(false)
            .await
            .ok_or_else(|| MemoryError::Config("consolidation provider is not configured".to_owned()))?;
        runner.execute_pass(context).await
    }

    pub fn set_eviction_publisher(
        &self,
        publisher: Arc<dyn EvictionEventPublisher>,
    ) -> Result<(), MemoryError> {
        self.decay_scheduler
            .lock()
            .map_err(|_| {
                MemoryError::Config("memory decay scheduler mutex was poisoned".to_owned())
            })?
            .set_eviction_publisher(publisher);
        Ok(())
    }

    /// Detects context, resolves the virtual-key namespace, and injects memories.
    #[allow(clippy::too_many_arguments)]
    pub async fn process_request(
        &self,
        request: &mut OpenAIRequest,
        query: &str,
        model_context_window: u32,
        post_truncation_tokens: u32,
        effective_config: EffectiveMemoryConfig,
        vk_id: Option<&str>,
    ) -> Result<MemoryRequestResult, MemoryError> {
        let detected = self.context_detector.detect_with_label(request);
        let context = detected.context;
        self.metrics
            .record_project_detection(context_namespace_type(&context));
        let namespace = ResolvedNamespace::resolve(vk_id, &context);
        let context_kind = match &context {
            ContextType::Project(_) => "project",
            ContextType::Agent(_) => "agent",
            ContextType::User => "user",
        };
        // `upsert_namespace_labels` is a synchronous SQLite write and
        // `retrieve_lexical` runs an FTS5 + bm25 scan; both take the single
        // connection mutex. Run them on the blocking pool so a slow query cannot
        // park an async worker — with memory enabled this sits on the path of
        // every chat request, so inline execution let concurrent traffic occupy
        // every worker thread at once.
        let store = Arc::clone(&self.store);
        let label_scope = namespace
            .context_scope
            .as_deref()
            .unwrap_or(&namespace.user_scope)
            .to_owned();
        let display_name = detected.display_name.clone();
        spawn_store_task(move || {
            store.upsert_namespace_labels(
                &label_scope,
                context_kind,
                display_name.as_deref(),
                None,
            )
        })
        .await?;

        let injection = if effective_config.enabled {
            let injector = Arc::clone(&self.injector);
            let retrieval_namespace = namespace.clone();
            let retrieval_query = query.to_owned();
            let lexical = spawn_store_task(move || {
                injector.retrieve_lexical(&retrieval_namespace, &retrieval_query, None)
            })
            .await?;
            let candidates = if let (Some(tier), Some(qdrant)) = (
                self.vector_tier.read().await.clone(),
                self.config.read().await.qdrant.clone(),
            ) {
                retrieve_with_vector_fallback(
                    &self.store,
                    tier.as_ref(),
                    &namespace,
                    query,
                    lexical,
                    qdrant.fts_weight,
                    qdrant.vector_weight,
                    std::time::Duration::from_secs(qdrant.retrieval_timeout_seconds),
                )
                .await
            } else {
                lexical
            };
            self.injector.inject_candidates(
                request,
                candidates,
                effective_config.injection_strategy,
                model_context_window,
                post_truncation_tokens,
                effective_config.max_injection_tokens,
            )?
        } else {
            InjectionResult::default()
        };

        Ok(MemoryRequestResult {
            context,
            namespace,
            injection,
        })
    }

    /// Performs synchronous explicit extraction for feedback-sensitive paths.
    pub async fn extract_explicit_response(
        &self,
        messages: &[ExtractionMessage],
        namespace: &ResolvedNamespace,
        request_id: Option<Uuid>,
    ) -> Result<ExtractionCounts, MemoryError> {
        let config = self.config.read().await.clone();
        if !config.enabled {
            return Ok(ExtractionCounts::default());
        }
        let extraction = self
            .extractor
            .extract_explicit(
                messages,
                namespace,
                request_id,
                ExtractionPolicy {
                    allow_sensitive_storage: config.allow_sensitive_storage,
                    max_memories_per_namespace: config.max_memories_per_namespace as usize,
                },
            )
            .await?;
        self.schedule_unindexed_entries().await?;
        Ok(extraction)
    }

    /// Queue provider extraction after the caller has assembled the response.
    pub async fn schedule_automatic_extraction(
        &self,
        messages: &[ExtractionMessage],
        response_content: &str,
        namespace: &ResolvedNamespace,
        request_id: Option<Uuid>,
    ) -> AsyncExtractionSchedule {
        let config = self.config.read().await.clone();
        if !config.enabled {
            return AsyncExtractionSchedule::NotScheduled(AsyncExtractionSkipReason::Disabled);
        }
        let mut automatic_messages = messages.to_vec();
        automatic_messages.push(ExtractionMessage::caller(
            ExtractionRole::Assistant,
            response_content,
        ));
        self.extractor.spawn_after_delivery(AsyncExtractionRequest {
            enabled: config.auto_extract_enabled,
            provider: config.auto_extract_provider,
            model: config.auto_extract_model,
            minimum_turns: config.auto_extract_min_turns,
            messages: automatic_messages,
            namespace: namespace.clone(),
            source_request_id: request_id,
            policy: ExtractionPolicy {
                allow_sensitive_storage: config.allow_sensitive_storage,
                max_memories_per_namespace: config.max_memories_per_namespace as usize,
            },
        })
    }

    /// Performs explicit extraction and then queues optional provider extraction.
    ///
    /// The caller must invoke this method only after the response has been
    /// delivered. This API never claims delivery occurred; invocation is the
    /// ordering boundary, and automatic work is spawned only after explicit
    /// extraction completes.
    pub async fn process_response(
        &self,
        messages: &[ExtractionMessage],
        response_content: &str,
        namespace: &ResolvedNamespace,
        request_id: Option<Uuid>,
    ) -> Result<MemoryResponseResult, MemoryError> {
        let config = self.config.read().await.clone();
        if !config.enabled {
            return Ok(MemoryResponseResult {
                extraction: ExtractionCounts::default(),
                automatic_extraction: AsyncExtractionSchedule::NotScheduled(
                    AsyncExtractionSkipReason::Disabled,
                ),
            });
        }

        let policy = ExtractionPolicy {
            allow_sensitive_storage: config.allow_sensitive_storage,
            max_memories_per_namespace: config.max_memories_per_namespace as usize,
        };
        let extraction = self
            .extractor
            .extract_explicit(messages, namespace, request_id, policy)
            .await?;
        let mut automatic_messages = messages.to_vec();
        automatic_messages.push(ExtractionMessage::caller(
            ExtractionRole::Assistant,
            response_content,
        ));
        let automatic_extraction = self.extractor.spawn_after_delivery(AsyncExtractionRequest {
            enabled: config.auto_extract_enabled,
            provider: config.auto_extract_provider,
            model: config.auto_extract_model,
            minimum_turns: config.auto_extract_min_turns,
            messages: automatic_messages,
            namespace: namespace.clone(),
            source_request_id: request_id,
            policy,
        });

        Ok(MemoryResponseResult {
            extraction,
            automatic_extraction,
        })
    }

    async fn schedule_unindexed_entries(&self) -> Result<(), MemoryError> {
        if self.vector_tier.read().await.is_none() {
            return Ok(());
        }
        let _ = self.store.mark_unconfigured_vector_entries_pending()?;
        for candidate in self.store.list_pending_vector_retries(Utc::now())? {
            self.schedule_vector_index(candidate.entry).await;
        }
        Ok(())
    }

    pub async fn set_vector_tier(&self, tier: Option<Arc<dyn MemoryVectorTier>>) {
        *self.vector_tier.write().await = tier;
    }

    pub async fn schedule_vector_index(&self, entry: MemoryEntry) {
        schedule_vector_index_task(
            self.store.clone(),
            self.vector_tier.clone(),
            self.vector_index_semaphore.clone(),
            entry,
        );
    }

    /// Best-effort removal of one entry's vector point; never fails callers.
    pub async fn schedule_vector_delete(&self, id: Uuid) {
        schedule_vector_delete_task(self.vector_tier.clone(), id).await;
    }

    /// Best-effort removal of a namespace's vector points; never fails callers.
    pub async fn schedule_vector_delete_namespace(&self, namespace: String) {
        schedule_vector_delete_namespace_task(self.vector_tier.clone(), namespace).await;
    }

    pub async fn retry_pending_vector_indexes(&self) -> Result<u64, MemoryError> {
        MemoryVectorRetryCallback {
            store: self.store.clone(),
            vector_tier: self.vector_tier.clone(),
        }
        .retry_pending()
        .await
    }

    /// Applies a validated hot reload while preserving the open store.
    ///
    /// Changing the SQLite path requires a process restart. Every successful
    /// reload replaces the config snapshot and restarts the scheduler once.
    pub async fn reload(&self, new_config: MemoryConfig) -> Result<(), MemoryError> {
        validate_config(&new_config)?;
        if new_config.auto_extract_enabled && !self.extraction_provider_available {
            return Err(MemoryError::Config(
                "auto_extract_enabled requires a memory extraction provider adapter".to_owned(),
            ));
        }

        let current_path = self.config.read().await.database_path.clone();
        if new_config.database_path != current_path {
            return Err(MemoryError::Config(
                "database_path change requires a gateway restart".to_owned(),
            ));
        }
        warn_sensitive_storage_once(&new_config);

        let previous_qdrant = self.config.read().await.qdrant.clone();
        *self.config.write().await = new_config.clone();
        {
            let mut scheduler = self.decay_scheduler.lock().map_err(|_| {
                MemoryError::Config("memory decay scheduler mutex was poisoned".to_owned())
            })?;
        // Refresh the consolidation hook from the new config before the
        // restart re-spawns the background task. The shared pass gate is
        // carried over so an in-flight pass still excludes the manual trigger.
        let hook: Arc<dyn ConsolidationHook> = if new_config.consolidation_enabled {
            Arc::new(MemoryConsolidationRunner::from_system(
                self,
                Arc::clone(&self.consolidation_pass_gate),
            ))
        } else {
            Arc::new(DisabledConsolidationHook)
        };
            scheduler.set_consolidation_hook(hook);
            scheduler.restart_decay_scheduler(
                new_config.decay_schedule_hours,
                new_config.max_memories_per_namespace as usize,
            )?;
        }
        if previous_qdrant != new_config.qdrant {
            *self.vector_tier.write().await = None;
        }
        Ok(())
    }
}

fn schedule_vector_index_task(
    store: Arc<MemoryStore>,
    vector_tier: Arc<RwLock<Option<Arc<dyn MemoryVectorTier>>>>,
    semaphore: Arc<tokio::sync::Semaphore>,
    entry: MemoryEntry,
) {
    tokio::spawn(async move {
        let Some(tier) = vector_tier.read().await.clone() else {
            return;
        };
        if let Err(error) = store.mark_vector_pending(entry.id) {
            tracing::warn!(entry_id = %entry.id, error = %error, "failed to mark memory vector pending");
            return;
        }
        let permit = match semaphore.acquire_owned().await {
            Ok(permit) => permit,
            Err(_) => return,
        };
        let attempted_at = Utc::now();
        let result = tokio::time::timeout(vector::VECTOR_INDEX_TIMEOUT, tier.index(&entry)).await;
        drop(permit);
        match result {
            Ok(Ok(())) => {
                if let Err(error) = store.mark_vector_indexed(entry.id, attempted_at) {
                    tracing::warn!(entry_id = %entry.id, error = %error, "failed to mark memory vector indexed");
                }
            }
            Ok(Err(error)) => {
                mark_vector_retry(&store, entry.id, attempted_at, &error.to_string());
            }
            Err(_) => {
                mark_vector_retry(&store, entry.id, attempted_at, "vector indexing timed out");
            }
        }
    });
}

fn mark_vector_retry(store: &MemoryStore, id: Uuid, attempted_at: DateTime<Utc>, error: &str) {
    let next_retry_at = attempted_at + chrono::Duration::minutes(5);
    if let Err(store_error) = store.mark_vector_retry(id, attempted_at, next_retry_at, error) {
        tracing::warn!(entry_id = %id, error = %store_error, "failed to mark memory vector retry");
    } else {
        tracing::warn!(entry_id = %id, error = %error, "memory vector indexing failed; retry scheduled");
    }
}

/// Await one best-effort tier delete, logging sanitized failures inline.
async fn run_vector_delete(result: Result<(), MemoryError>, key: impl std::fmt::Display) {
    if let Err(error) = result {
        tracing::warn!(key = %key, error = %error, "memory vector delete failed");
    }
}

async fn schedule_vector_delete_task(
    vector_tier: Arc<RwLock<Option<Arc<dyn MemoryVectorTier>>>>,
    id: Uuid,
) {
    let Some(tier) = vector_tier.read().await.clone() else {
        return;
    };
    let result = tokio::time::timeout(vector::VECTOR_INDEX_TIMEOUT, tier.delete_entry(id)).await;
    match result {
        Ok(result) => run_vector_delete(result, id).await,
        Err(_) => tracing::warn!(entry_id = %id, "memory vector delete timed out"),
    }
}

async fn schedule_vector_delete_namespace_task(
    vector_tier: Arc<RwLock<Option<Arc<dyn MemoryVectorTier>>>>,
    namespace: String,
) {
    let Some(tier) = vector_tier.read().await.clone() else {
        return;
    };
    let result =
        tokio::time::timeout(vector::VECTOR_INDEX_TIMEOUT, tier.delete_namespace(&namespace)).await;
    match result {
        Ok(result) => run_vector_delete(result, namespace).await,
        Err(_) => tracing::warn!(namespace = %namespace, "memory vector namespace delete timed out"),
    }
}

fn context_namespace_type(context: &ContextType) -> metrics::NamespaceType {
    match context {
        ContextType::Project(_) => metrics::NamespaceType::Project,
        ContextType::Agent(_) => metrics::NamespaceType::Agent,
        ContextType::User => metrics::NamespaceType::User,
    }
}

fn validate_config(config: &MemoryConfig) -> Result<(), MemoryError> {
    config.validate().map_err(|errors| {
        MemoryError::Config(
            errors
                .into_iter()
                .map(|error| error.to_string())
                .collect::<Vec<_>>()
                .join("; "),
        )
    })
}

fn warn_sensitive_storage_once(config: &MemoryConfig) {
    if config.allow_sensitive_storage {
        SENSITIVE_STORAGE_WARNING.call_once(|| {
            tracing::warn!(
                "memory sensitive-content protection is bypassed; sensitive data may be persisted"
            );
        });
    }
}

/// Format the optional user-visible memory activity suffix.
///
/// Structured-output suppression and `show_feedback` are caller policy; this
/// helper suppresses feedback when:
/// - All counters are zero, OR
/// - Only injection occurred (no storage/rejection) and it is NOT the first
///   message in the thread. This avoids noisy per-message feedback when the
///   system is simply re-injecting existing memories without learning anything
///   new.
pub fn format_feedback_suffix(
    injected: u32,
    stored: u32,
    sensitive_rejected: u32,
    is_thread_start: bool,
) -> Option<String> {
    if injected == 0 && stored == 0 && sensitive_rejected == 0 {
        return None;
    }

    // Only show injection-only feedback at thread start; mid-conversation
    // messages only get feedback when new memories are stored or rejected.
    if stored == 0 && sensitive_rejected == 0 && !is_thread_start {
        return None;
    }

    let mut suffix = format!("\n\n---\n📝 {injected} memories injected | {stored} memories stored");
    if sensitive_rejected > 0 {
        suffix.push_str(&format!(
            " ({sensitive_rejected} rejected: sensitive content)"
        ));
    }
    Some(suffix)
}

/// Shared test doubles for vector tiers used across this module's tests.
///
/// Kept as a `#[cfg(test)]` submodule of this file so every `mod.rs` test
/// imports one fake instead of drifting copies; other test files still use
/// their own fakes (cross-file unification is a follow-up).
#[cfg(test)]
pub(crate) mod test_support {
    use std::sync::Mutex;

    use async_trait::async_trait;
    use uuid::Uuid;

    use super::{MemoryEntry, MemoryError, MemoryVectorTier, VectorMatch};

    /// Records index/delete calls for sequencing assertions.
    pub(crate) struct RecordingVectorTier {
        pub fail_deletes: bool,
        pub events: Mutex<Vec<String>>,
    }

    impl RecordingVectorTier {
        pub fn new() -> Self {
            Self {
                fail_deletes: false,
                events: Mutex::new(Vec::new()),
            }
        }

        pub fn events(&self) -> Vec<String> {
            self.events.lock().unwrap().clone()
        }
    }

    #[async_trait]
    impl MemoryVectorTier for RecordingVectorTier {
        async fn index(&self, entry: &MemoryEntry) -> Result<(), MemoryError> {
            self.events
                .lock()
                .unwrap()
                .push(format!("index:{}", entry.id));
            Ok(())
        }

        async fn search(
            &self,
            _query: &str,
            _limit: usize,
        ) -> Result<Vec<VectorMatch>, MemoryError> {
            Ok(Vec::new())
        }

        async fn delete_entry(&self, id: Uuid) -> Result<(), MemoryError> {
            self.events.lock().unwrap().push(format!("delete:{id}"));
            if self.fail_deletes {
                return Err(MemoryError::Qdrant("mock delete failure".to_owned()));
            }
            Ok(())
        }

        async fn delete_namespace(&self, namespace: &str) -> Result<(), MemoryError> {
            self.events
                .lock()
                .unwrap()
                .push(format!("delete_namespace:{namespace}"));
            if self.fail_deletes {
                return Err(MemoryError::Qdrant("mock delete failure".to_owned()));
            }
            Ok(())
        }
    }
}

#[cfg(test)]
mod tests {
    use std::sync::atomic::{AtomicUsize, Ordering};

    use serde_json::{json, Map};
    use tempfile::tempdir;

    use super::test_support::RecordingVectorTier;
    use super::*;

    struct FailingVectorTier;

    #[async_trait]
    impl MemoryVectorTier for FailingVectorTier {
        async fn index(&self, _entry: &MemoryEntry) -> Result<(), MemoryError> {
            Err(MemoryError::Qdrant("mock index failure".to_owned()))
        }

        async fn search(
            &self,
            _query: &str,
            _limit: usize,
        ) -> Result<Vec<VectorMatch>, MemoryError> {
            Err(MemoryError::Qdrant("mock search failure".to_owned()))
        }
    }

    struct SuccessfulVectorTier;

    #[async_trait]
    impl MemoryVectorTier for SuccessfulVectorTier {
        async fn index(&self, _entry: &MemoryEntry) -> Result<(), MemoryError> {
            Ok(())
        }

        async fn search(
            &self,
            _query: &str,
            _limit: usize,
        ) -> Result<Vec<VectorMatch>, MemoryError> {
            Ok(Vec::new())
        }
    }

    struct TestExtractionProvider {
        calls: AtomicUsize,
    }

    #[async_trait]
    impl MemoryExtractionProvider for TestExtractionProvider {
        async fn extract(
            &self,
            _request: MemoryExtractionProviderRequest,
        ) -> Result<Vec<StructuredMemoryCandidate>, MemoryExtractionProviderError> {
            self.calls.fetch_add(1, Ordering::SeqCst);
            Ok(Vec::new())
        }
    }

    /// Extraction provider whose calls park while paused, used to hold a
    /// consolidation pass mid-flight for gate-contention tests. Unpaused
    /// calls return immediately with an empty candidate list.
    struct PausableExtractionProvider {
        calls: AtomicUsize,
        paused: Mutex<bool>,
        release: tokio::sync::Notify,
    }

    impl Default for PausableExtractionProvider {
        fn default() -> Self {
            Self {
                calls: AtomicUsize::new(0),
                paused: Mutex::new(false),
                release: tokio::sync::Notify::new(),
            }
        }
    }

    impl PausableExtractionProvider {
        /// Park subsequent provider calls until the returned guard drops.
        async fn hold(&self) -> ReleaseGuard<'_> {
            *self.paused.lock().unwrap() = true;
            ReleaseGuard { provider: self }
        }
    }

    struct ReleaseGuard<'a> {
        provider: &'a PausableExtractionProvider,
    }

    impl Drop for ReleaseGuard<'_> {
        fn drop(&mut self) {
            *self.provider.paused.lock().unwrap() = false;
            // notify_one stores a permit when nothing is parked yet, so a
            // release that races the call's parking cannot be lost.
            self.provider.release.notify_one();
        }
    }

    #[async_trait]
    impl MemoryExtractionProvider for PausableExtractionProvider {
        async fn extract(
            &self,
            _request: MemoryExtractionProviderRequest,
        ) -> Result<Vec<StructuredMemoryCandidate>, MemoryExtractionProviderError> {
            self.calls.fetch_add(1, Ordering::SeqCst);
            while *self.paused.lock().unwrap() {
                self.release.notified().await;
            }
            Ok(Vec::new())
        }
    }

    async fn wait_for_calls(provider: &PausableExtractionProvider, expected: usize) {
        tokio::time::timeout(Duration::from_secs(5), async {
            while provider.calls.load(Ordering::SeqCst) < expected {
                tokio::task::yield_now().await;
            }
        })
        .await
        .expect("provider calls must be recorded");
    }

    fn request(content: &str) -> OpenAIRequest {
        OpenAIRequest {
            model: "gpt-4o-mini".to_owned(),
            messages: vec![crate::models::openai::Message {
                role: "user".to_owned(),
                content: json!(content),
                extra: Map::new(),
            }],
            stream: false,
            temperature: None,
            max_tokens: None,
            extra: Map::new(),
        }
    }

    async fn system(config: MemoryConfig) -> MemorySystem {
        MemorySystem::new(config, None, None).await.unwrap()
    }

    #[tokio::test]
    async fn memory_system_initializes_sqlite_and_disabled_calls_are_noops() {
        let temp = tempdir().unwrap();
        let database_path = temp.path().join("memory.db");
        let config = MemoryConfig {
            database_path: database_path.to_string_lossy().into_owned(),
            ..MemoryConfig::default()
        };
        let system = system(config).await;
        assert!(database_path.is_file());

        let mut outgoing = request("alpha project");
        let original = serde_json::to_value(&outgoing).unwrap();
        let result = system
            .process_request(
                &mut outgoing,
                "alpha project",
                128_000,
                10,
                EffectiveMemoryConfig {
                    enabled: false,
                    injection_strategy: InjectionStrategy::SystemPromptPrefix,
                    max_injection_tokens: 500,
                    show_feedback: true,
                },
                Some("vk"),
            )
            .await
            .unwrap();
        assert_eq!(result.injection, InjectionResult::default());
        assert_eq!(serde_json::to_value(outgoing).unwrap(), original);

        let response = system
            .process_response(
                &[ExtractionMessage::caller(
                    ExtractionRole::User,
                    "Remember this disabled fact.",
                )],
                "answer",
                &result.namespace,
                None,
            )
            .await
            .unwrap();
        assert_eq!(response.extraction, ExtractionCounts::default());
        assert!(matches!(
            response.automatic_extraction,
            AsyncExtractionSchedule::NotScheduled(AsyncExtractionSkipReason::Disabled)
        ));
        assert_eq!(system.store.stats().unwrap().total_count, 0);
    }

    #[tokio::test]
    async fn admin_insert_survives_vector_failure_and_retry_transitions_to_indexed() {
        let config = MemoryConfig {
            enabled: true,
            database_path: ":memory:".to_owned(),
            qdrant: Some(MemoryQdrantConfig {
                qdrant_url: "http://localhost:6333".to_owned(),
                embedding_provider: "mock".to_owned(),
                embedding_model: "text-embedding-3-small".to_owned(),
                ..Default::default()
            }),
            ..Default::default()
        };
        let system =
            MemorySystem::new_with_vector(config, None, None, Some(Arc::new(FailingVectorTier)))
                .await
                .unwrap();
        let entry = system
            .admin_create(
                "user::vector".to_owned(),
                "Vector indexing remains asynchronous.".to_owned(),
                MemoryType::Fact,
            )
            .await
            .unwrap();
        tokio::time::timeout(Duration::from_secs(1), async {
            loop {
                if system
                    .store
                    .vector_status(entry.id)
                    .unwrap()
                    .is_some_and(|status| status.0 == "retry")
                {
                    break;
                }
                tokio::task::yield_now().await;
            }
        })
        .await
        .unwrap();
        assert!(system.store.get_entry_by_id(entry.id).unwrap().is_some());
        let retry = system.store.vector_status(entry.id).unwrap().unwrap();
        assert_eq!(retry.0, "retry");
        assert_eq!(retry.1, 1);
        assert!(retry.2.unwrap().contains("mock index failure"));

        system
            .set_vector_tier(Some(Arc::new(SuccessfulVectorTier)))
            .await;
        system
            .store
            .connection()
            .unwrap()
            .execute(
                "UPDATE memories SET vector_next_retry_at = NULL WHERE id = ?1",
                [entry.id.to_string()],
            )
            .unwrap();
    assert_eq!(system.retry_pending_vector_indexes().await.unwrap(), 1);
    assert_eq!(
        system.store.vector_status(entry.id).unwrap().unwrap().0,
        "indexed"
    );
    }

    fn vector_config() -> MemoryConfig {
        MemoryConfig {
            enabled: true,
            database_path: ":memory:".to_owned(),
            qdrant: Some(MemoryQdrantConfig {
                qdrant_url: "http://localhost:6333".to_owned(),
                embedding_provider: "mock".to_owned(),
                embedding_model: "text-embedding-3-small".to_owned(),
                ..Default::default()
            }),
            ..Default::default()
        }
    }

    async fn wait_for_events(tier: &RecordingVectorTier, expected: usize) -> Vec<String> {
        tokio::time::timeout(Duration::from_secs(2), async {
            loop {
                let events = tier.events();
                if events.len() >= expected {
                    return events;
                }
                tokio::task::yield_now().await;
            }
        })
        .await
        .expect("tier events must be recorded")
    }

    #[tokio::test]
    async fn admin_edit_rejects_disallowed_content_before_write() {
        let system = system(MemoryConfig {
            enabled: true,
            database_path: ":memory:".to_owned(),
            ..Default::default()
        })
        .await;
        let entry = system
            .admin_create(
                "user::edit".to_owned(),
                "Original safe stored content.".to_owned(),
                MemoryType::Fact,
            )
            .await
            .unwrap();

    let rejected = system
        .admin_edit(entry.id, "api key sk-ABCDEFGHIJKLMNOPQRSTUVWXY", "fact")
        .await
        .unwrap_err();
    assert!(matches!(rejected, AdminCreateError::SensitiveContent));
    let short = system
        .admin_edit(entry.id, "tiny two", "fact")
        .await
        .unwrap_err();
    assert!(matches!(short, AdminCreateError::InvalidContent(_)));
    let single_word = system
        .admin_edit(entry.id, "sufficiently-long-word", "fact")
        .await
        .unwrap_err();
    assert!(matches!(single_word, AdminCreateError::InvalidContent(_)));
    let stored = system.store.get_entry_by_id(entry.id).unwrap().unwrap();
    assert_eq!(stored.content, "Original safe stored content.");
}

    #[tokio::test]
    async fn admin_edit_updates_store_and_tier_observes_delete_then_reindex() {
        let tier = Arc::new(RecordingVectorTier::new());
        let system = MemorySystem::new_with_vector(
            vector_config(),
            None,
            None,
            Some(tier.clone() as Arc<dyn MemoryVectorTier>),
        )
        .await
        .unwrap();
    let entry = system
        .admin_create(
            "user::edit".to_owned(),
            "Vector indexing remains asynchronous.".to_owned(),
            MemoryType::Fact,
        )
        .await
        .unwrap();
    let create_events = wait_for_events(&tier, 1).await;
    assert_eq!(create_events[0], format!("index:{}", entry.id));

    let updated = system
        .admin_edit(entry.id, "Edited content stays safe.", "decision")
        .await
        .unwrap()
        .expect("entry must exist");
    assert_eq!(updated.content, "Edited content stays safe.");
    assert_eq!(updated.memory_type, MemoryType::Decision);
    assert!(updated.updated_at.is_some());

    let events = wait_for_events(&tier, 3).await;
    assert_eq!(events[1], format!("delete:{}", entry.id));
    assert_eq!(events[2], format!("index:{}", entry.id));
        assert_eq!(
            system.store.get_entry_by_id(entry.id).unwrap().unwrap(),
            updated
        );
    }

    #[tokio::test]
    async fn admin_edit_returns_none_for_missing_id() {
        let system = system(MemoryConfig {
            enabled: true,
            database_path: ":memory:".to_owned(),
            ..Default::default()
        })
        .await;
        assert!(
            system
                .admin_edit(Uuid::from_u128(4242), "Missing entry content.", "fact")
                .await
                .unwrap()
                .is_none()
        );
    }

    #[tokio::test]
    async fn admin_edit_rejects_superseded_entry_without_write_or_reindex() {
        let tier = Arc::new(RecordingVectorTier::new());
        let system = MemorySystem::new_with_vector(
            vector_config(),
            None,
            None,
            Some(tier.clone() as Arc<dyn MemoryVectorTier>),
        )
        .await
        .unwrap();
        let source = system
            .admin_create(
                "user::superseded-edit".to_owned(),
                "First entry that will be superseded.".to_owned(),
                MemoryType::Fact,
            )
            .await
            .unwrap();
        let survivor = system
            .admin_create(
                "user::superseded-edit".to_owned(),
                "Second entry that supersedes the first.".to_owned(),
                MemoryType::Fact,
            )
            .await
            .unwrap();
        let marked = system
            .store
            .mark_superseded(&[source.id], survivor.id)
            .unwrap();
        assert_eq!(marked, 1);

        let error = system
            .admin_edit(source.id, "Attempting to edit dead weight.", "fact")
            .await
            .unwrap_err();
    match error {
        AdminCreateError::Memory(MemoryError::Conflict(message)) => {
            assert_eq!(message, "memory entry is already superseded");
        }
        other => panic!("expected superseded-entry conflict, got {other:?}"),
    }

        // No write, no re-index: content unchanged and no new tier events
        // beyond the two create-index calls.
        let stored = system.store.get_entry_by_id(source.id).unwrap().unwrap();
        assert_eq!(stored.content, "First entry that will be superseded.");
        let events = wait_for_events(&tier, 2).await;
        assert_eq!(events.len(), 2);
    }

    #[tokio::test]
    async fn run_consolidation_now_rejects_while_a_pass_is_running() {
        let provider = Arc::new(PausableExtractionProvider::default());
        let config = MemoryConfig {
            enabled: true,
            database_path: ":memory:".to_owned(),
            consolidation_enabled: true,
            consolidation_provider: "provider".to_owned(),
            consolidation_model: "model".to_owned(),
            ..Default::default()
        };
        let system =
            MemorySystem::new_with_vector(config, None, Some(provider.clone()), None)
                .await
                .unwrap();
        seed_near_duplicates(&system);

        // Park the provider so the background pass holds the gate while the
        // manual trigger tries to start.
        let release = provider.hold().await;
        let background = {
            let runner = MemoryConsolidationRunner::from_system(
                &system,
                Arc::clone(&system.consolidation_pass_gate),
            );
            tokio::spawn(async move { runner.run_consolidation_pass().await })
        };
        wait_for_calls(&provider, 1).await;

        let error = system.run_consolidation_now().await.unwrap_err();
        assert!(error.to_string().contains("consolidation already in progress"));

        drop(release);
        background.await.unwrap();

        // Gate released: a manual pass runs again.
        system.run_consolidation_now().await.unwrap();
        assert!(provider.calls.load(Ordering::SeqCst) >= 2);
    }

    #[tokio::test]
    async fn manual_pass_released_gate_lets_background_hook_run() {
        let provider = Arc::new(PausableExtractionProvider::default());
        let config = MemoryConfig {
            enabled: true,
            database_path: ":memory:".to_owned(),
            consolidation_enabled: true,
            consolidation_provider: "provider".to_owned(),
            consolidation_model: "model".to_owned(),
            ..Default::default()
        };
        let system =
            MemorySystem::new_with_vector(config, None, Some(provider.clone()), None)
                .await
                .unwrap();
        seed_near_duplicates(&system);

        // Manual pass completes while no background pass holds the gate.
        system.run_consolidation_now().await.unwrap();
        assert_eq!(provider.calls.load(Ordering::SeqCst), 1);

        // The background hook then serializes behind the same gate instead
        // of racing the completed manual pass.
        let runner = MemoryConsolidationRunner::from_system(
            &system,
            Arc::clone(&system.consolidation_pass_gate),
        );
        runner.run_consolidation_pass().await;
        assert_eq!(provider.calls.load(Ordering::SeqCst), 2);
    }

    #[tokio::test]
    async fn admin_edit_survives_vector_delete_failure() {
        let tier = Arc::new(RecordingVectorTier {
            fail_deletes: true,
            events: Mutex::new(Vec::new()),
        });
        let system = MemorySystem::new_with_vector(
            vector_config(),
            None,
            None,
            Some(tier.clone() as Arc<dyn MemoryVectorTier>),
        )
        .await
        .unwrap();
        let entry = system
            .admin_create(
                "user::edit".to_owned(),
                "Vector indexing remains asynchronous.".to_owned(),
                MemoryType::Fact,
            )
            .await
            .unwrap();
        wait_for_events(&tier, 1).await;

        let updated = system
            .admin_edit(entry.id, "Edited content stays safe.", "fact")
            .await
            .unwrap()
            .expect("edit must succeed despite tier delete failure");
        assert_eq!(updated.content, "Edited content stays safe.");
        let events = wait_for_events(&tier, 3).await;
        assert_eq!(events[1], format!("delete:{}", entry.id));
        assert!(system.store.get_entry_by_id(entry.id).unwrap().is_some());
    }

    #[tokio::test]
    async fn default_trait_delete_methods_are_noops() {
        struct BareTier;
        #[async_trait]
        impl MemoryVectorTier for BareTier {
            async fn index(&self, _entry: &MemoryEntry) -> Result<(), MemoryError> {
                Ok(())
            }
            async fn search(
                &self,
                _query: &str,
                _limit: usize,
            ) -> Result<Vec<VectorMatch>, MemoryError> {
                Ok(Vec::new())
            }
        }
        let tier = BareTier;
        tier.delete_entry(Uuid::new_v4()).await.unwrap();
        tier.delete_namespace("user::bare").await.unwrap();
    }

    #[tokio::test]
    async fn admin_delete_and_clear_trigger_tier_deletes_best_effort() {
        let tier = Arc::new(RecordingVectorTier::new());
        let system = MemorySystem::new_with_vector(
            vector_config(),
            None,
            None,
            Some(tier.clone() as Arc<dyn MemoryVectorTier>),
        )
        .await
        .unwrap();
        let first = system
            .admin_create(
                "user::purge".to_owned(),
                "First purge candidate content.".to_owned(),
                MemoryType::Fact,
            )
            .await
            .unwrap();
        let second = system
            .admin_create(
                "user::purge".to_owned(),
                "Second purge candidate content.".to_owned(),
                MemoryType::Fact,
            )
            .await
            .unwrap();
        wait_for_events(&tier, 2).await;

        assert!(system.admin_delete(first.id).await.unwrap());
        assert!(!system.admin_delete(first.id).await.unwrap());
        assert_eq!(system.admin_clear("user::purge").await.unwrap(), 1);
        assert_eq!(
            system.store.get_entry_by_id(second.id).unwrap(),
            None,
            "second entry must be cleared with the namespace"
        );

        let events = wait_for_events(&tier, 4).await;
        assert_eq!(events[2], format!("delete:{}", first.id));
        assert_eq!(events[3], "delete_namespace:user::purge");
    }

    #[test]
    fn normalized_merge_uses_normalized_weight_sum() {
        let first = sample_entry();
        let mut second = sample_entry();
        second.id = Uuid::from_u128(2);
        second.content = "Second vector candidate".to_owned();
        let merged = merge_retrieval_scores(
            vec![ScoredMemory {
                entry: first.clone(),
                final_score: 2.0,
                estimated_tokens: 0,
            }],
            vec![(second.clone(), 0.5)],
            2.0,
            1.0,
        );
        assert_eq!(merged[0].entry.id, first.id);
        assert!((merged[0].final_score - (2.0 / 3.0)).abs() < 1.0e-9);
        assert!((merged[1].final_score - (1.0 / 3.0)).abs() < 1.0e-9);
    }

    #[tokio::test]
    async fn vector_retrieval_failure_returns_fts_unchanged() {
        let store = MemoryStore::new(Path::new(":memory:")).unwrap();
        let lexical = vec![ScoredMemory {
            entry: sample_entry(),
            final_score: 0.75,
            estimated_tokens: 0,
        }];
        let namespace = ResolvedNamespace {
            user_scope: lexical[0].entry.namespace.clone(),
            context_scope: None,
        };
        let returned = retrieve_with_vector_fallback(
            &store,
            &FailingVectorTier,
            &namespace,
            "query",
            lexical.clone(),
            0.4,
            0.6,
            std::time::Duration::from_secs(5),
        )
        .await;
        assert_eq!(returned, lexical);
    }

    #[tokio::test]
    async fn auto_extract_requires_adapter_when_enabled() {
        let config = MemoryConfig {
            enabled: true,
            database_path: ":memory:".to_owned(),
            auto_extract_enabled: true,
            auto_extract_provider: "provider".to_owned(),
            auto_extract_model: "model".to_owned(),
            ..MemoryConfig::default()
        };
        let error = match MemorySystem::new(config, None, None).await {
            Ok(_) => panic!("automatic extraction unexpectedly initialized without an adapter"),
            Err(error) => error,
        };
        assert!(error
            .to_string()
            .contains("requires a memory extraction provider"));
    }

    #[tokio::test]
    async fn reload_preserves_store_and_rejects_database_path_changes() {
        let mut config = MemoryConfig {
            enabled: true,
            database_path: ":memory:".to_owned(),
            ..MemoryConfig::default()
        };
        let system = system(config.clone()).await;
        let stored = system
            .store
            .store_entry(
                NewMemoryEntry {
                    namespace: "user::reload".to_owned(),
                    content: "Persist across configuration reloads.".to_owned(),
                    memory_type: MemoryType::Fact,
                    source_request_id: None,
                },
                None,
            )
            .unwrap();

        config.max_injection_tokens = 321;
        config.decay_schedule_hours = 2;
        system.reload(config.clone()).await.unwrap();
        assert_eq!(system.config.read().await.max_injection_tokens, 321);
        assert!(system.store.get_entry_by_id(stored.id).unwrap().is_some());

        config.database_path = "other-memory.db".to_owned();
        let error = system.reload(config).await.unwrap_err();
        assert!(error.to_string().contains("requires a gateway restart"));
        assert!(system.store.get_entry_by_id(stored.id).unwrap().is_some());
    }

    #[tokio::test]
    async fn decay_cycle_triggers_consolidation_when_enabled_and_not_when_disabled() {
        let provider = Arc::new(TestExtractionProvider {
            calls: AtomicUsize::new(0),
        });

        let config = MemoryConfig {
            enabled: true,
            database_path: ":memory:".to_owned(),
            consolidation_enabled: true,
            consolidation_provider: "provider".to_owned(),
            consolidation_model: "model".to_owned(),
            ..MemoryConfig::default()
        };
        let system = MemorySystem::new_with_vector(config, None, Some(provider.clone()), None)
            .await
            .unwrap();
        seed_near_duplicates(&system);
        assert_eq!(provider.calls.load(Ordering::SeqCst), 0);

        let scheduler = system.decay_scheduler.lock().unwrap();
        scheduler.run_once().await.unwrap();
        drop(scheduler);
        assert_eq!(provider.calls.load(Ordering::SeqCst), 1);
        assert!(system.metrics.last_consolidation_pass().is_some());

        // Disabled (default): no provider call on decay completion.
        let disabled_provider = Arc::new(TestExtractionProvider {
            calls: AtomicUsize::new(0),
        });
        let config = MemoryConfig {
            enabled: true,
            database_path: ":memory:".to_owned(),
            ..MemoryConfig::default()
        };
        let disabled = MemorySystem::new_with_vector(
            config,
            None,
            Some(disabled_provider.clone()),
            None,
        )
        .await
        .unwrap();
        seed_near_duplicates(&disabled);
        let scheduler = disabled.decay_scheduler.lock().unwrap();
        scheduler.run_once().await.unwrap();
        drop(scheduler);
        assert_eq!(disabled_provider.calls.load(Ordering::SeqCst), 0);
        assert!(disabled.metrics.last_consolidation_pass().is_none());
    }

    fn seed_near_duplicates(system: &MemorySystem) {
        for content in [
            "The deploy target is fly.io eu-west",
            "The deploy target is fly.io eu-west region",
        ] {
            system
                .store
                .store_entry(
                    NewMemoryEntry {
                        namespace: "user::consolidation".to_owned(),
                        content: content.to_owned(),
                        memory_type: MemoryType::Fact,
                        source_request_id: None,
                    },
                    None,
                )
                .unwrap();
        }
    }

    #[tokio::test]
    async fn reload_applies_consolidation_settings_without_dropping_store() {
        let provider = Arc::new(TestExtractionProvider {
            calls: AtomicUsize::new(0),
        });
        let config = MemoryConfig {
            enabled: true,
            database_path: ":memory:".to_owned(),
            ..MemoryConfig::default()
        };
        let system = MemorySystem::new_with_vector(config.clone(), None, Some(provider.clone()), None)
            .await
            .unwrap();
        seed_near_duplicates(&system);
        let stored = system
            .store
            .store_entry(
                NewMemoryEntry {
                    namespace: "user::consolidation-reload".to_owned(),
                    content: "Survives consolidation hot reload.".to_owned(),
                    memory_type: MemoryType::Fact,
                    source_request_id: None,
                },
                None,
            )
            .unwrap();

        let mut enabled = config.clone();
        enabled.consolidation_enabled = true;
        enabled.consolidation_provider = "provider".to_owned();
        enabled.consolidation_model = "model".to_owned();
        system.reload(enabled).await.unwrap();
        assert!(system.config.read().await.consolidation_enabled);

        let scheduler = system.decay_scheduler.lock().unwrap();
        scheduler.run_once().await.unwrap();
        drop(scheduler);
        assert_eq!(provider.calls.load(Ordering::SeqCst), 1);
        assert!(system.store.get_entry_by_id(stored.id).unwrap().is_some());
    }

    #[tokio::test]
    async fn request_and_response_round_trip_injects_then_extracts() {
        let config = MemoryConfig {
            enabled: true,
            database_path: ":memory:".to_owned(),
            ..MemoryConfig::default()
        };
        let system = system(config).await;
        system
            .store
            .store_entry(
                NewMemoryEntry {
                    namespace: "user::roundtrip".to_owned(),
                    content: "Use alpha project stable Rust conventions.".to_owned(),
                    memory_type: MemoryType::Decision,
                    source_request_id: None,
                },
                None,
            )
            .unwrap();

        let mut outgoing = request("alpha project Rust");
        let request_result = system
            .process_request(
                &mut outgoing,
                "alpha project Rust",
                128_000,
                100,
                EffectiveMemoryConfig {
                    enabled: true,
                    injection_strategy: InjectionStrategy::SystemPromptPrefix,
                    max_injection_tokens: 500,
                    show_feedback: true,
                },
                Some("roundtrip"),
            )
            .await
            .unwrap();
        assert_eq!(request_result.injection.memories_injected, 1);

        let response_result = system
            .process_response(
                &[ExtractionMessage::caller(
                    ExtractionRole::User,
                    "Remember this beta project uses SQLite.",
                )],
                "Acknowledged.",
                &request_result.namespace,
                Some(Uuid::from_u128(99)),
            )
            .await
            .unwrap();
        assert_eq!(response_result.extraction.stored, 1);
        assert!(matches!(
            response_result.automatic_extraction,
            AsyncExtractionSchedule::NotScheduled(AsyncExtractionSkipReason::Disabled)
        ));
        assert_eq!(system.store.namespace_count("user::roundtrip").unwrap(), 2);
    }

    #[tokio::test]
    async fn configured_adapter_is_used_for_background_extraction() {
        let provider = Arc::new(TestExtractionProvider {
            calls: AtomicUsize::new(0),
        });
        let config = MemoryConfig {
            enabled: true,
            database_path: ":memory:".to_owned(),
            auto_extract_enabled: true,
            auto_extract_provider: "provider".to_owned(),
            auto_extract_model: "model".to_owned(),
            auto_extract_min_turns: 2,
            ..MemoryConfig::default()
        };
        let system = MemorySystem::new(config, None, Some(provider.clone()))
            .await
            .unwrap();
        let result = system
            .process_response(
                &[ExtractionMessage::caller(ExtractionRole::User, "question")],
                "answer",
                &ResolvedNamespace::resolve(Some("adapter"), &ContextType::User),
                None,
            )
            .await
            .unwrap();
        let handle = match result.automatic_extraction {
            AsyncExtractionSchedule::Spawned(handle) => handle,
            AsyncExtractionSchedule::NotScheduled(reason) => panic!("not spawned: {reason:?}"),
        };
        assert!(matches!(
            handle.await.unwrap(),
            AsyncExtractionOutcome::Completed(_)
        ));
        assert_eq!(provider.calls.load(Ordering::SeqCst), 1);
    }

    fn sample_entry() -> MemoryEntry {
        let created_at = DateTime::<Utc>::from_timestamp(1_700_000_000, 0)
            .expect("sample timestamp must be valid");
        let last_accessed_at = DateTime::<Utc>::from_timestamp(1_700_000_100, 0)
            .expect("sample timestamp must be valid");

        MemoryEntry {
            id: Uuid::from_u128(1),
            namespace: "user::default::project::0123456789abcdef".to_string(),
            content: "Use repository serde conventions.".to_string(),
            memory_type: MemoryType::Decision,
            relevance_score: 0.75,
            created_at,
            last_accessed_at,
            access_count: 3,
            source_request_id: Some(Uuid::from_u128(2)),
            updated_at: None,
            superseded_by: None,
        }
    }

    #[test]
    fn feedback_suffix_formats_activity_and_sensitive_rejections() {
        // All zeros → always None regardless of thread position
        assert_eq!(format_feedback_suffix(0, 0, 0, true), None);
        assert_eq!(format_feedback_suffix(0, 0, 0, false), None);

        // Storage occurred → shown regardless of thread position
        assert_eq!(
            format_feedback_suffix(2, 1, 0, true).as_deref(),
            Some("\n\n---\n📝 2 memories injected | 1 memories stored")
        );
        assert_eq!(
            format_feedback_suffix(2, 1, 0, false).as_deref(),
            Some("\n\n---\n📝 2 memories injected | 1 memories stored")
        );

        // Sensitive rejection → shown regardless of thread position
        assert_eq!(
            format_feedback_suffix(0, 0, 1, true).as_deref(),
            Some(
                "\n\n---\n📝 0 memories injected | 0 memories stored (1 rejected: sensitive content)"
            )
        );
        assert_eq!(
            format_feedback_suffix(0, 0, 1, false).as_deref(),
            Some(
                "\n\n---\n📝 0 memories injected | 0 memories stored (1 rejected: sensitive content)"
            )
        );

        // Injection only at thread start → shown
        assert_eq!(
            format_feedback_suffix(14, 0, 0, true).as_deref(),
            Some("\n\n---\n📝 14 memories injected | 0 memories stored")
        );

        // Injection only mid-conversation → suppressed
        assert_eq!(format_feedback_suffix(14, 0, 0, false), None);
    }

    #[test]
    fn memory_type_serde_uses_snake_case() {
        let cases = [
            (MemoryType::Preference, "\"preference\""),
            (MemoryType::Fact, "\"fact\""),
            (MemoryType::Context, "\"context\""),
            (MemoryType::Decision, "\"decision\""),
        ];

        for (memory_type, serialized) in cases {
            assert_eq!(serde_json::to_string(&memory_type).unwrap(), serialized);
            assert_eq!(
                serde_json::from_str::<MemoryType>(serialized).unwrap(),
                memory_type
            );
        }
    }

    #[test]
    fn injection_strategy_serde_uses_snake_case() {
        let cases = [
            (
                InjectionStrategy::SystemPromptPrefix,
                "\"system_prompt_prefix\"",
            ),
            (InjectionStrategy::SyntheticMessage, "\"synthetic_message\""),
        ];

        for (strategy, serialized) in cases {
            assert_eq!(serde_json::to_string(&strategy).unwrap(), serialized);
            assert_eq!(
                serde_json::from_str::<InjectionStrategy>(serialized).unwrap(),
                strategy
            );
        }
    }

    #[test]
    fn core_structs_preserve_field_semantics() {
        let entry = sample_entry();
        let scored = ScoredMemory {
            entry: entry.clone(),
            final_score: 1.125,
            estimated_tokens: 12,
        };
        let namespace = ResolvedNamespace {
            user_scope: "user::default".to_string(),
            context_scope: Some(entry.namespace.clone()),
        };
        let result = InjectionResult {
            memories_injected: 1,
            injection_tokens: scored.estimated_tokens,
            memories_stored: 2,
            sensitive_rejected: 3,
        };

        assert_eq!(scored.entry, entry);
        assert_eq!(scored.final_score, 1.125);
        assert_eq!(namespace.user_scope, "user::default");
        assert_eq!(
            namespace.context_scope.as_deref(),
            Some(scored.entry.namespace.as_str())
        );
        assert_eq!(result.memories_injected, 1);
        assert_eq!(result.injection_tokens, 12);
        assert_eq!(result.memories_stored, 2);
        assert_eq!(result.sensitive_rejected, 3);
        assert_eq!(
            ContextType::Project("0123456789abcdef".into()),
            ContextType::Project("0123456789abcdef".into())
        );
        assert_ne!(
            ContextType::Agent("0123456789abcdef".into()),
            ContextType::User
        );
    }

    #[test]
    fn memory_entry_serde_round_trip_preserves_fields() {
        let entry = sample_entry();
        let serialized = serde_json::to_string(&entry).unwrap();
        let deserialized: MemoryEntry = serde_json::from_str(&serialized).unwrap();

        assert_eq!(deserialized, entry);
    }

    #[test]
    fn content_errors_report_character_limits_without_content() {
        let too_long = MemoryError::ContentTooLong {
            length: 4_097,
            max: 4_096,
        };
        let too_short = MemoryError::ContentTooShort { length: 4, min: 5 };

        assert_eq!(
            too_long.to_string(),
            "memory content is too long: 4097 characters (maximum 4096)"
        );
        assert_eq!(
            too_short.to_string(),
            "memory content is too short: 4 characters (minimum 5)"
        );
    }

    #[test]
    fn rusqlite_errors_convert_to_store_errors() {
        let error: MemoryError = rusqlite::Error::InvalidQuery.into();

        assert!(matches!(
            error,
            MemoryError::Store(rusqlite::Error::InvalidQuery)
        ));
    }
}
