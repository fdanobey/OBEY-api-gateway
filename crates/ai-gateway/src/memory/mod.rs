//! Core types for the persistent memory store.
//!
//! Feature implementations live in sibling modules. Only modules whose files
//! exist in the current implementation wave are declared here.

pub mod admin;
pub mod config;
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
pub use extraction_adapter::GatewayExtractionAdapter;
pub use context_detector::ContextDetector;
pub use decay::{DecayScheduler, VectorRetryCallback};
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
pub use metrics::MemoryMetrics;
pub use namespace::{sanitize_vk_id, validate_namespace, MAX_NAMESPACE_CHARS, MAX_VK_ID_CHARS};
pub use scoring::{apply_decay, compute_score, recency_boost};
pub use sensitive::{
    PiiProviderErrorKind, PiiProviderFailurePolicy, SensitiveContentScanner, SensitiveMatchSource,
    SensitiveScanError, SensitiveScanOptions, SensitiveScanResult,
};
pub use store::{
    MemoryEntryInput, MemoryEntryPage, MemoryStats, MemoryStore, NewMemoryEntry, ProjectNamespace,
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

    /// An optional Qdrant operation failed.
    #[error("memory Qdrant error: {0}")]
    Qdrant(String),

    /// Memory extraction failed.
    #[error("memory extraction error: {0}")]
    Extraction(String),
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
    extraction_provider_available: bool,
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
                provider,
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
        let mut decay_scheduler = DecayScheduler::with_metrics(
            (*store).clone(),
            config.decay_schedule_hours,
            config.max_memories_per_namespace as usize,
            metrics.clone(),
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
            config: Arc::new(RwLock::new(config)),
            vector_tier,
            vector_index_semaphore,
            sensitive_scanner: scanner.clone(),
            decay_scheduler: Mutex::new(decay_scheduler),
            extraction_provider_available,
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
        let context = self.context_detector.detect(request);
        self.metrics
            .record_project_detection(context_namespace_type(&context));
        let namespace = ResolvedNamespace::resolve(vk_id, &context);
        let injection = if effective_config.enabled {
            let lexical = self.injector.retrieve_lexical(&namespace, query, None)?;
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
        self.decay_scheduler
            .lock()
            .map_err(|_| {
                MemoryError::Config("memory decay scheduler mutex was poisoned".to_owned())
            })?
            .restart_decay_scheduler(
                new_config.decay_schedule_hours,
                new_config.max_memories_per_namespace as usize,
            )?;
        *self.config.write().await = new_config.clone();
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
/// helper only suppresses an all-zero result.
pub fn format_feedback_suffix(
    injected: u32,
    stored: u32,
    sensitive_rejected: u32,
) -> Option<String> {
    if injected == 0 && stored == 0 && sensitive_rejected == 0 {
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

#[cfg(test)]
mod tests {
    use std::sync::atomic::{AtomicUsize, Ordering};

    use serde_json::{json, Map};
    use tempfile::tempdir;

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
        }
    }

    #[test]
    fn feedback_suffix_formats_activity_and_sensitive_rejections() {
        assert_eq!(format_feedback_suffix(0, 0, 0), None);
        assert_eq!(
            format_feedback_suffix(2, 1, 0).as_deref(),
            Some("\n\n---\n📝 2 memories injected | 1 memories stored")
        );
        assert_eq!(
            format_feedback_suffix(0, 0, 1).as_deref(),
            Some(
                "\n\n---\n📝 0 memories injected | 0 memories stored (1 rejected: sensitive content)"
            )
        );
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
