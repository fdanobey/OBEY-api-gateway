//! In-memory semantic response index for smart routing.
//!
//! The index retains embeddings and content-free metadata only. Response bytes
//! are delegated to an injected encrypted store and are addressed by opaque
//! references that are never formatted or logged by this module.

use std::collections::{HashMap, VecDeque};
use std::fmt;
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::{Arc, Mutex, MutexGuard};
use std::time::{Duration, Instant};

use async_trait::async_trait;
use serde_json::{Map, Value};
use sha2::{Digest, Sha256};

use crate::models::openai::{Message, OpenAIRequest};

use super::tier::{ComplexityScore, RoutingDecision, SmartRoutingTier};
use super::{SemanticCacheFailure, SemanticCacheHit, SemanticCacheLookup, SemanticRoutingCache};

const MAX_IDENTITY_LABEL_BYTES: usize = 256;
const MIN_VECTOR_NORM: f64 = 1.0e-12;

/// A one-way tenant and authorization-context digest.
///
/// A cache must be constructed per digest. Raw tenant or credential material
/// cannot be represented by this type and is therefore never retained in the
/// index.
#[derive(Clone, Copy, PartialEq, Eq, Hash)]
pub struct TenantAuthHash([u8; 32]);

impl TenantAuthHash {
    pub const fn from_digest(digest: [u8; 32]) -> Self {
        Self(digest)
    }

    pub const fn as_bytes(&self) -> &[u8; 32] {
        &self.0
    }
}

impl fmt::Debug for TenantAuthHash {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter.write_str("TenantAuthHash([redacted])")
    }
}

/// Store-owned reference to encrypted response bytes.
#[derive(Clone, PartialEq, Eq, Hash)]
pub struct OpaquePayloadRef(Vec<u8>);

impl OpaquePayloadRef {
    pub fn new(value: Vec<u8>) -> Result<Self, PayloadStoreError> {
        if value.is_empty() {
            return Err(PayloadStoreError::new(
                PayloadStoreErrorKind::InvalidReference,
            ));
        }
        Ok(Self(value))
    }

    pub fn as_bytes(&self) -> &[u8] {
        &self.0
    }
}

impl fmt::Debug for OpaquePayloadRef {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter.write_str("OpaquePayloadRef([redacted])")
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum EmbeddingProviderErrorKind {
    Unavailable,
    Timeout,
    Rejected,
    Backend,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct EmbeddingProviderError {
    pub kind: EmbeddingProviderErrorKind,
}

impl EmbeddingProviderError {
    pub const fn new(kind: EmbeddingProviderErrorKind) -> Self {
        Self { kind }
    }
}

/// Injectable embedding boundary. Implementations must not route recursively
/// through the semantic cache.
#[async_trait]
pub trait EmbeddingProvider: Send + Sync {
    async fn embed(&self, semantic_text: &str) -> Result<Vec<f32>, EmbeddingProviderError>;
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum PayloadStoreErrorKind {
    Unavailable,
    Timeout,
    Encryption,
    InvalidReference,
    Backend,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct PayloadStoreError {
    pub kind: PayloadStoreErrorKind,
}

impl PayloadStoreError {
    pub const fn new(kind: PayloadStoreErrorKind) -> Self {
        Self { kind }
    }
}

/// Mandatory encrypted payload boundary.
///
/// `put` must encrypt before persistence and return an opaque reference. The
/// cache never keeps a copy of `plaintext` after this call returns.
#[async_trait]
pub trait EncryptedPayloadStore: Send + Sync {
    async fn put(
        &self,
        plaintext: &[u8],
        ttl: Duration,
    ) -> Result<OpaquePayloadRef, PayloadStoreError>;

    async fn get(
        &self,
        payload_ref: &OpaquePayloadRef,
    ) -> Result<Option<Vec<u8>>, PayloadStoreError>;

    async fn delete(&self, payload_ref: &OpaquePayloadRef) -> Result<(), PayloadStoreError>;
}

#[derive(Debug, Clone, PartialEq)]
pub struct SemanticCacheConfig {
    pub max_entries: usize,
    pub ttl: Duration,
    pub similarity_threshold: f64,
    pub min_quality_score: f64,
    /// Stable provider/model/version label. It is hashed into every scope key.
    pub embedding_namespace: String,
    /// Routing policy revision. It is hashed into every scope key.
    pub policy_version: String,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum SemanticCacheConfigError {
    ZeroCapacity,
    ZeroTtl,
    InvalidSimilarityThreshold,
    InvalidMinimumQuality,
    InvalidEmbeddingNamespace,
    InvalidPolicyVersion,
}

impl SemanticCacheConfig {
    fn validate(&self) -> Result<(), SemanticCacheConfigError> {
        if self.max_entries == 0 {
            return Err(SemanticCacheConfigError::ZeroCapacity);
        }
        if self.ttl.is_zero() {
            return Err(SemanticCacheConfigError::ZeroTtl);
        }
        if !is_unit_interval(self.similarity_threshold) {
            return Err(SemanticCacheConfigError::InvalidSimilarityThreshold);
        }
        if !is_unit_interval(self.min_quality_score) {
            return Err(SemanticCacheConfigError::InvalidMinimumQuality);
        }
        validate_identity_label(&self.embedding_namespace)
            .map_err(|_| SemanticCacheConfigError::InvalidEmbeddingNamespace)?;
        validate_identity_label(&self.policy_version)
            .map_err(|_| SemanticCacheConfigError::InvalidPolicyVersion)?;
        Ok(())
    }
}

fn is_unit_interval(value: f64) -> bool {
    value.is_finite() && (0.0..=1.0).contains(&value)
}

fn validate_identity_label(value: &str) -> Result<(), ()> {
    if value.trim().is_empty() || value.len() > MAX_IDENTITY_LABEL_BYTES {
        Err(())
    } else {
        Ok(())
    }
}

/// Request facts unavailable from the OpenAI request itself.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct CacheRequestContext {
    /// True only when a requested stream has been fully buffered and validated.
    pub response_buffered: bool,
    /// True when user/session-specific state affected the response.
    pub personalized: bool,
    /// True when generation was constrained to deterministic behavior.
    pub deterministic: bool,
}

impl CacheRequestContext {
    pub fn conservative(request: &OpenAIRequest) -> Self {
        Self {
            response_buffered: !request.stream,
            personalized: request_is_personalized(request),
            deterministic: request_is_deterministic(request),
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct CacheEligibilityPolicy {
    pub allow_tools: bool,
    pub allow_unbuffered_streams: bool,
    pub allow_personalized: bool,
    pub allow_nondeterministic: bool,
}

impl Default for CacheEligibilityPolicy {
    fn default() -> Self {
        Self {
            allow_tools: false,
            allow_unbuffered_streams: false,
            allow_personalized: false,
            allow_nondeterministic: false,
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum CacheIneligibility {
    Tools,
    UnbufferedStream,
    Personalized,
    Nondeterministic,
    EmptySemanticInput,
    QualityBelowMinimum,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ResponseProvenance {
    pub model: String,
    pub tier: SmartRoutingTier,
}

#[derive(Debug, Clone, PartialEq)]
pub struct SemanticMetadataHit {
    pub entry_id: u64,
    pub similarity: ComplexityScore,
    pub quality_score: ComplexityScore,
    pub decision: RoutingDecision,
    pub provenance: ResponseProvenance,
}

#[derive(Debug, Clone, PartialEq)]
pub enum MetadataLookupOutcome {
    Hit(SemanticMetadataHit),
    Miss,
    Ineligible(CacheIneligibility),
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum MetadataLookupFailure {
    Embedding(EmbeddingProviderErrorKind),
    InvalidEmbedding,
    EmbeddingDimensionMismatch,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum PayloadRetrievalFailure {
    Store(PayloadStoreErrorKind),
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum SemanticInsertFailure {
    Embedding(EmbeddingProviderErrorKind),
    InvalidEmbedding,
    EmbeddingDimensionMismatch,
    Store(PayloadStoreErrorKind),
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum SemanticInsertOutcome {
    Inserted { entry_id: u64 },
    Ineligible(CacheIneligibility),
}

pub struct MetadataLookupRequest<'a> {
    pub request: &'a OpenAIRequest,
    pub model_group: &'a str,
    pub context: CacheRequestContext,
}

pub struct SemanticInsertRequest<'a> {
    pub request: &'a OpenAIRequest,
    pub model_group: &'a str,
    pub context: CacheRequestContext,
    pub quality_score: ComplexityScore,
    pub decision: RoutingDecision,
    pub provenance: ResponseProvenance,
    pub response_payload: &'a [u8],
}

#[derive(Clone)]
struct IndexEntry {
    id: u64,
    request_digest: [u8; 32],
    scope_key: [u8; 32],
    embedding: Arc<[f32]>,
    quality_score: ComplexityScore,
    decision: RoutingDecision,
    provenance: ResponseProvenance,
    payload_ref: OpaquePayloadRef,
    expires_at: Instant,
}

struct IndexState {
    entries: HashMap<u64, IndexEntry>,
    lru: VecDeque<u64>,
    embedding_dimension: Option<usize>,
}

impl IndexState {
    fn new(capacity: usize) -> Self {
        Self {
            entries: HashMap::with_capacity(capacity),
            lru: VecDeque::with_capacity(capacity),
            embedding_dimension: None,
        }
    }

    fn validate_dimension(&mut self, dimension: usize) -> Result<(), ()> {
        match self.embedding_dimension {
            Some(expected) if expected != dimension => Err(()),
            Some(_) => Ok(()),
            None => {
                self.embedding_dimension = Some(dimension);
                Ok(())
            }
        }
    }

    fn remove_expired(&mut self, now: Instant) -> Vec<OpaquePayloadRef> {
        let expired = self
            .entries
            .iter()
            .filter_map(|(id, entry)| (entry.expires_at <= now).then_some(*id))
            .collect::<Vec<_>>();
        expired
            .into_iter()
            .filter_map(|id| self.remove(id).map(|entry| entry.payload_ref))
            .collect()
    }

    fn insert(&mut self, entry: IndexEntry, capacity: usize) -> Vec<OpaquePayloadRef> {
        let mut removed = Vec::new();
        if let Some(existing_id) = self.entries.iter().find_map(|(id, existing)| {
            (existing.request_digest == entry.request_digest).then_some(*id)
        }) {
            if let Some(existing) = self.remove(existing_id) {
                removed.push(existing.payload_ref);
            }
        }
        while self.entries.len() >= capacity {
            let Some(oldest) = self.lru.front().copied() else {
                break;
            };
            if let Some(evicted) = self.remove(oldest) {
                removed.push(evicted.payload_ref);
            }
        }
        self.lru.push_back(entry.id);
        self.entries.insert(entry.id, entry);
        removed
    }

    fn remove(&mut self, id: u64) -> Option<IndexEntry> {
        self.lru.retain(|cached| *cached != id);
        self.entries.remove(&id)
    }

    fn touch(&mut self, id: u64) {
        self.lru.retain(|cached| *cached != id);
        self.lru.push_back(id);
    }

    fn payload_ref(&mut self, id: u64, now: Instant) -> PayloadRefLookup {
        let Some(entry) = self.entries.get(&id) else {
            return PayloadRefLookup::Missing;
        };
        if entry.expires_at <= now {
            return self
                .remove(id)
                .map_or(PayloadRefLookup::Missing, |expired| {
                    PayloadRefLookup::Expired(expired.payload_ref)
                });
        }
        let payload_ref = entry.payload_ref.clone();
        self.touch(id);
        PayloadRefLookup::Live(payload_ref)
    }
}

enum PayloadRefLookup {
    Live(OpaquePayloadRef),
    Expired(OpaquePayloadRef),
    Missing,
}

/// Bounded, true-LRU, TTL-aware in-memory semantic index.
pub struct InMemorySemanticCache {
    config: SemanticCacheConfig,
    tenant_auth_hash: TenantAuthHash,
    eligibility: CacheEligibilityPolicy,
    embedding_provider: Arc<dyn EmbeddingProvider>,
    payload_store: Arc<dyn EncryptedPayloadStore>,
    next_id: AtomicU64,
    state: Mutex<IndexState>,
}

impl InMemorySemanticCache {
    pub fn new(
        config: SemanticCacheConfig,
        tenant_auth_hash: TenantAuthHash,
        embedding_provider: Arc<dyn EmbeddingProvider>,
        payload_store: Arc<dyn EncryptedPayloadStore>,
    ) -> Result<Self, SemanticCacheConfigError> {
        config.validate()?;
        Ok(Self {
            state: Mutex::new(IndexState::new(config.max_entries)),
            config,
            tenant_auth_hash,
            eligibility: CacheEligibilityPolicy::default(),
            embedding_provider,
            payload_store,
            next_id: AtomicU64::new(1),
        })
    }

    pub fn with_eligibility_policy(mut self, eligibility: CacheEligibilityPolicy) -> Self {
        self.eligibility = eligibility;
        self
    }

    pub async fn lookup_metadata(
        &self,
        input: MetadataLookupRequest<'_>,
    ) -> Result<MetadataLookupOutcome, MetadataLookupFailure> {
        if let Some(reason) = self.ineligibility(input.request, input.context, None) {
            return Ok(MetadataLookupOutcome::Ineligible(reason));
        }
        let semantic_text = semantic_text(input.request);
        if semantic_text.is_empty() {
            return Ok(MetadataLookupOutcome::Ineligible(
                CacheIneligibility::EmptySemanticInput,
            ));
        }
        let scope_key = self.scope_key(input.request, input.model_group);
        let embedding = self
            .embedding_provider
            .embed(&semantic_text)
            .await
            .map_err(|error| MetadataLookupFailure::Embedding(error.kind))?;
        validate_embedding(&embedding).map_err(|_| MetadataLookupFailure::InvalidEmbedding)?;

        let now = Instant::now();
        let (outcome, expired) = {
            let mut state = self.lock_state();
            state
                .validate_dimension(embedding.len())
                .map_err(|_| MetadataLookupFailure::EmbeddingDimensionMismatch)?;
            let expired = state.remove_expired(now);
            let best = state
                .entries
                .values()
                .filter(|entry| {
                    entry.scope_key == scope_key
                        && entry.quality_score.value() >= self.config.min_quality_score
                })
                .filter_map(|entry| {
                    cosine_similarity(&embedding, &entry.embedding)
                        .map(|similarity| (entry, similarity))
                })
                .filter(|(_, similarity)| *similarity >= self.config.similarity_threshold)
                .max_by(
                    |(left_entry, left_similarity), (right_entry, right_similarity)| {
                        left_similarity
                            .total_cmp(right_similarity)
                            .then_with(|| left_entry.id.cmp(&right_entry.id))
                    },
                )
                .map(|(entry, similarity)| {
                    let mut decision = entry.decision.clone();
                    decision.cache_hit = true;
                    SemanticMetadataHit {
                        entry_id: entry.id,
                        similarity: ComplexityScore::new(similarity),
                        quality_score: entry.quality_score,
                        decision,
                        provenance: entry.provenance.clone(),
                    }
                });
            if let Some(hit) = &best {
                state.touch(hit.entry_id);
            }
            (
                best.map_or(MetadataLookupOutcome::Miss, MetadataLookupOutcome::Hit),
                expired,
            )
        };
        self.delete_payloads(expired).await;
        Ok(outcome)
    }

    pub async fn insert(
        &self,
        input: SemanticInsertRequest<'_>,
    ) -> Result<SemanticInsertOutcome, SemanticInsertFailure> {
        if let Some(reason) =
            self.ineligibility(input.request, input.context, Some(input.quality_score))
        {
            return Ok(SemanticInsertOutcome::Ineligible(reason));
        }
        let semantic_text = semantic_text(input.request);
        if semantic_text.is_empty() {
            return Ok(SemanticInsertOutcome::Ineligible(
                CacheIneligibility::EmptySemanticInput,
            ));
        }
        let scope_key = self.scope_key(input.request, input.model_group);
        let request_digest = request_digest(input.request, &scope_key);
        let embedding = self
            .embedding_provider
            .embed(&semantic_text)
            .await
            .map_err(|error| SemanticInsertFailure::Embedding(error.kind))?;
        validate_embedding(&embedding).map_err(|_| SemanticInsertFailure::InvalidEmbedding)?;
        {
            let mut state = self.lock_state();
            state
                .validate_dimension(embedding.len())
                .map_err(|_| SemanticInsertFailure::EmbeddingDimensionMismatch)?;
        }
        let payload_ref = self
            .payload_store
            .put(input.response_payload, self.config.ttl)
            .await
            .map_err(|error| SemanticInsertFailure::Store(error.kind))?;
        let entry_id = self.next_entry_id();
        let now = Instant::now();
        let entry = IndexEntry {
            id: entry_id,
            request_digest,
            scope_key,
            embedding: embedding.into(),
            quality_score: input.quality_score,
            decision: input.decision,
            provenance: input.provenance,
            payload_ref,
            expires_at: now + self.config.ttl,
        };
        let removed = {
            let mut state = self.lock_state();
            let mut removed = state.remove_expired(now);
            removed.extend(state.insert(entry, self.config.max_entries));
            removed
        };
        self.delete_payloads(removed).await;
        Ok(SemanticInsertOutcome::Inserted { entry_id })
    }

    /// Retrieve decrypted response bytes separately from metadata lookup.
    pub async fn retrieve_payload(
        &self,
        entry_id: u64,
    ) -> Result<Option<Vec<u8>>, PayloadRetrievalFailure> {
        let payload_ref = self.lock_state().payload_ref(entry_id, Instant::now());
        match payload_ref {
            PayloadRefLookup::Live(payload_ref) => self
                .payload_store
                .get(&payload_ref)
                .await
                .map_err(|error| PayloadRetrievalFailure::Store(error.kind)),
            PayloadRefLookup::Expired(payload_ref) => {
                let _ = self.payload_store.delete(&payload_ref).await;
                Ok(None)
            }
            PayloadRefLookup::Missing => Ok(None),
        }
    }

    pub fn len(&self) -> usize {
        self.lock_state().entries.len()
    }

    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }

    fn ineligibility(
        &self,
        request: &OpenAIRequest,
        context: CacheRequestContext,
        quality_score: Option<ComplexityScore>,
    ) -> Option<CacheIneligibility> {
        if request_uses_tools(request) && !self.eligibility.allow_tools {
            return Some(CacheIneligibility::Tools);
        }
        if request.stream
            && !context.response_buffered
            && !self.eligibility.allow_unbuffered_streams
        {
            return Some(CacheIneligibility::UnbufferedStream);
        }
        if context.personalized && !self.eligibility.allow_personalized {
            return Some(CacheIneligibility::Personalized);
        }
        if !context.deterministic && !self.eligibility.allow_nondeterministic {
            return Some(CacheIneligibility::Nondeterministic);
        }
        if quality_score.is_some_and(|score| score.value() < self.config.min_quality_score) {
            return Some(CacheIneligibility::QualityBelowMinimum);
        }
        None
    }

    fn scope_key(&self, request: &OpenAIRequest, model_group: &str) -> [u8; 32] {
        let mut hasher = Sha256::new();
        hash_field(
            &mut hasher,
            b"tenant_auth",
            self.tenant_auth_hash.as_bytes(),
        );
        hash_field(&mut hasher, b"model_group", model_group.as_bytes());
        hash_field(
            &mut hasher,
            b"embedding_namespace",
            self.config.embedding_namespace.as_bytes(),
        );
        hash_field(
            &mut hasher,
            b"policy_version",
            self.config.policy_version.as_bytes(),
        );
        let response_fields = canonical_response_fields(request);
        hash_field(
            &mut hasher,
            b"response_fields",
            canonical_json(&response_fields).as_bytes(),
        );
        hasher.finalize().into()
    }

    fn next_entry_id(&self) -> u64 {
        loop {
            let id = self.next_id.fetch_add(1, Ordering::Relaxed);
            if id != 0 {
                return id;
            }
        }
    }

    fn lock_state(&self) -> MutexGuard<'_, IndexState> {
        self.state
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner)
    }

    async fn delete_payloads(&self, payload_refs: Vec<OpaquePayloadRef>) {
        for payload_ref in payload_refs {
            let _ = self.payload_store.delete(&payload_ref).await;
        }
    }
}

#[async_trait]
impl SemanticRoutingCache for InMemorySemanticCache {
    async fn lookup(
        &self,
        input: SemanticCacheLookup<'_>,
    ) -> Result<Option<SemanticCacheHit>, SemanticCacheFailure> {
        let context = CacheRequestContext::conservative(input.request);
        match self
            .lookup_metadata(MetadataLookupRequest {
                request: input.request,
                model_group: &input.model_group.name,
                context,
            })
            .await
        {
            Ok(MetadataLookupOutcome::Hit(hit)) => Ok(Some(SemanticCacheHit {
                entry_id: hit.entry_id,
                similarity: hit.similarity,
                quality_score: hit.quality_score,
                decision: hit.decision,
            })),
            Ok(MetadataLookupOutcome::Miss | MetadataLookupOutcome::Ineligible(_)) => Ok(None),
            Err(MetadataLookupFailure::Embedding(
                EmbeddingProviderErrorKind::Unavailable | EmbeddingProviderErrorKind::Timeout,
            )) => Err(SemanticCacheFailure::Unavailable),
            Err(_) => Err(SemanticCacheFailure::Backend),
        }
    }
}

fn request_uses_tools(request: &OpenAIRequest) -> bool {
    request.extra.contains_key("tools")
        || request.extra.contains_key("tool_choice")
        || request.messages.iter().any(|message| {
            message.role.eq_ignore_ascii_case("tool")
                || message.extra.contains_key("tool_calls")
                || message.extra.contains_key("function_call")
        })
}

fn request_is_personalized(request: &OpenAIRequest) -> bool {
    [
        "user",
        "session_id",
        "personalization",
        "personalized",
        "memory",
        "memory_context",
    ]
    .iter()
    .any(|key| request.extra.contains_key(*key))
}

fn request_is_deterministic(request: &OpenAIRequest) -> bool {
    request.temperature == Some(0.0)
        && request
            .extra
            .get("top_p")
            .and_then(Value::as_f64)
            .is_none_or(|top_p| top_p == 0.0 || top_p == 1.0)
}

fn semantic_text(request: &OpenAIRequest) -> String {
    let mut semantic = String::new();
    for message in &request.messages {
        if message.role.eq_ignore_ascii_case("system")
            || message.role.eq_ignore_ascii_case("developer")
            || message.role.eq_ignore_ascii_case("tool")
        {
            continue;
        }
        let content = message.content_as_text();
        if content.trim().is_empty() {
            continue;
        }
        if !semantic.is_empty() {
            semantic.push('\n');
        }
        semantic.push_str(&message.role);
        semantic.push(':');
        semantic.push_str(&content);
    }
    semantic
}

fn canonical_response_fields(request: &OpenAIRequest) -> Value {
    let mut root = Map::new();
    root.insert("temperature".to_owned(), option_f32(request.temperature));
    root.insert(
        "max_tokens".to_owned(),
        request.max_tokens.map_or(Value::Null, Value::from),
    );

    let mut messages = Vec::new();
    for message in &request.messages {
        if message.role.eq_ignore_ascii_case("system")
            || message.role.eq_ignore_ascii_case("developer")
        {
            messages.push(canonical_message(message));
        }
    }
    root.insert("instructions".to_owned(), Value::Array(messages));
    root.insert("extra".to_owned(), Value::Object(request.extra.clone()));
    canonicalize(Value::Object(root))
}

fn canonical_message(message: &Message) -> Value {
    let mut value = Map::new();
    value.insert("role".to_owned(), Value::String(message.role.clone()));
    value.insert("content".to_owned(), message.content.clone());
    value.insert("extra".to_owned(), Value::Object(message.extra.clone()));
    canonicalize(Value::Object(value))
}

fn option_f32(value: Option<f32>) -> Value {
    value
        .and_then(|number| serde_json::Number::from_f64(number as f64))
        .map_or(Value::Null, Value::Number)
}

fn canonicalize(value: Value) -> Value {
    match value {
        Value::Object(map) => {
            let mut fields = map.into_iter().collect::<Vec<_>>();
            fields.sort_by(|left, right| left.0.cmp(&right.0));
            Value::Object(
                fields
                    .into_iter()
                    .map(|(key, value)| (key, canonicalize(value)))
                    .collect(),
            )
        }
        Value::Array(values) => Value::Array(values.into_iter().map(canonicalize).collect()),
        other => other,
    }
}

fn canonical_json(value: &Value) -> String {
    serde_json::to_string(value).expect("JSON values are always serializable")
}

fn request_digest(request: &OpenAIRequest, scope_key: &[u8; 32]) -> [u8; 32] {
    let mut hasher = Sha256::new();
    hash_field(&mut hasher, b"scope", scope_key);
    hash_field(
        &mut hasher,
        b"request",
        canonical_json(&canonicalize(
            serde_json::to_value(request).expect("OpenAI requests are serializable"),
        ))
        .as_bytes(),
    );
    hasher.finalize().into()
}

fn hash_field(hasher: &mut Sha256, label: &[u8], value: &[u8]) {
    hasher.update((label.len() as u64).to_be_bytes());
    hasher.update(label);
    hasher.update((value.len() as u64).to_be_bytes());
    hasher.update(value);
}

fn validate_embedding(embedding: &[f32]) -> Result<(), ()> {
    if embedding.is_empty() || embedding.iter().any(|value| !value.is_finite()) {
        return Err(());
    }
    let norm = embedding
        .iter()
        .map(|value| f64::from(*value) * f64::from(*value))
        .sum::<f64>();
    if norm <= MIN_VECTOR_NORM {
        Err(())
    } else {
        Ok(())
    }
}

fn cosine_similarity(left: &[f32], right: &[f32]) -> Option<f64> {
    if left.len() != right.len() || left.is_empty() {
        return None;
    }
    let mut dot = 0.0_f64;
    let mut left_norm = 0.0_f64;
    let mut right_norm = 0.0_f64;
    for (left_value, right_value) in left.iter().zip(right) {
        let left_value = f64::from(*left_value);
        let right_value = f64::from(*right_value);
        dot += left_value * right_value;
        left_norm += left_value * left_value;
        right_norm += right_value * right_value;
    }
    let denominator = left_norm.sqrt() * right_norm.sqrt();
    (denominator > MIN_VECTOR_NORM).then_some((dot / denominator).clamp(-1.0, 1.0))
}

#[cfg(test)]
mod tests {
    use std::collections::{HashMap, VecDeque};
    use std::sync::atomic::{AtomicUsize, Ordering};

    use proptest::prelude::*;
    use proptest::test_runner::{Config as ProptestConfig, TestRunner};
    use serde_json::{json, Map};

    use super::*;
    use crate::config::{ModelGroup, ProviderModel};
    use crate::smart_routing::config::{ClassifierMode, SmartRoutingConfig};
    use crate::smart_routing::tier::{ClassifierUsed, TaskType};
    use crate::smart_routing::{
        PinnedRoutingContext, RoutingPlanOutcome, SmartRouter, SmartRoutingInput,
    };

    struct MockEmbeddingProvider {
        calls: AtomicUsize,
    }

    #[async_trait]
    impl EmbeddingProvider for MockEmbeddingProvider {
        async fn embed(&self, semantic_text: &str) -> Result<Vec<f32>, EmbeddingProviderError> {
            self.calls.fetch_add(1, Ordering::SeqCst);
            if semantic_text.contains("different") {
                Ok(vec![0.0, 1.0])
            } else {
                Ok(vec![1.0, 0.0])
            }
        }
    }

    struct FixedEmbeddingProvider {
        responses: Mutex<VecDeque<Result<Vec<f32>, EmbeddingProviderError>>>,
        calls: AtomicUsize,
    }

    impl FixedEmbeddingProvider {
        fn new(responses: Vec<Result<Vec<f32>, EmbeddingProviderError>>) -> Self {
            Self {
                responses: Mutex::new(responses.into()),
                calls: AtomicUsize::new(0),
            }
        }
    }

    #[async_trait]
    impl EmbeddingProvider for FixedEmbeddingProvider {
        async fn embed(&self, _semantic_text: &str) -> Result<Vec<f32>, EmbeddingProviderError> {
            self.calls.fetch_add(1, Ordering::SeqCst);
            self.responses
                .lock()
                .unwrap()
                .pop_front()
                .expect("a fixed embedding response for every call")
        }
    }

    #[derive(Default)]
    struct MockEncryptedStore {
        next: AtomicUsize,
        put_calls: AtomicUsize,
        get_calls: AtomicUsize,
        delete_calls: AtomicUsize,
        values: Mutex<HashMap<Vec<u8>, Vec<u8>>>,
        fail_put: bool,
        fail_get: bool,
    }

    #[async_trait]
    impl EncryptedPayloadStore for MockEncryptedStore {
        async fn put(
            &self,
            plaintext: &[u8],
            _ttl: Duration,
        ) -> Result<OpaquePayloadRef, PayloadStoreError> {
            self.put_calls.fetch_add(1, Ordering::SeqCst);
            if self.fail_put {
                return Err(PayloadStoreError::new(PayloadStoreErrorKind::Encryption));
            }
            let key = self
                .next
                .fetch_add(1, Ordering::SeqCst)
                .to_be_bytes()
                .to_vec();
            self.values
                .lock()
                .unwrap()
                .insert(key.clone(), plaintext.to_vec());
            OpaquePayloadRef::new(key)
        }

        async fn get(
            &self,
            payload_ref: &OpaquePayloadRef,
        ) -> Result<Option<Vec<u8>>, PayloadStoreError> {
            self.get_calls.fetch_add(1, Ordering::SeqCst);
            if self.fail_get {
                return Err(PayloadStoreError::new(PayloadStoreErrorKind::Unavailable));
            }
            Ok(self
                .values
                .lock()
                .unwrap()
                .get(payload_ref.as_bytes())
                .cloned())
        }

        async fn delete(&self, payload_ref: &OpaquePayloadRef) -> Result<(), PayloadStoreError> {
            self.delete_calls.fetch_add(1, Ordering::SeqCst);
            self.values.lock().unwrap().remove(payload_ref.as_bytes());
            Ok(())
        }
    }

    fn config(max_entries: usize, ttl: Duration) -> SemanticCacheConfig {
        SemanticCacheConfig {
            max_entries,
            ttl,
            similarity_threshold: 0.9,
            min_quality_score: 0.7,
            embedding_namespace: "embedding-provider/model-v1".to_owned(),
            policy_version: "routing-policy-v1".to_owned(),
        }
    }

    fn request(text: &str) -> OpenAIRequest {
        OpenAIRequest {
            model: "logical-group".to_owned(),
            messages: vec![Message {
                role: "user".to_owned(),
                content: Value::String(text.to_owned()),
                extra: Map::new(),
            }],
            stream: false,
            temperature: Some(0.0),
            max_tokens: Some(100),
            extra: Map::new(),
        }
    }

    fn decision() -> RoutingDecision {
        RoutingDecision {
            score: ComplexityScore::new(0.4),
            adjusted_score: ComplexityScore::new(0.4),
            tier: SmartRoutingTier::Balanced,
            task_type: TaskType::General,
            classifier: ClassifierUsed::Heuristic,
            escalated: false,
            escalation_count: 0,
            cache_hit: false,
            budget_downgraded: false,
            context_filtered: false,
        }
    }

    fn context(request: &OpenAIRequest) -> CacheRequestContext {
        CacheRequestContext::conservative(request)
    }

    fn cache(
        max_entries: usize,
        ttl: Duration,
    ) -> (
        InMemorySemanticCache,
        Arc<MockEmbeddingProvider>,
        Arc<MockEncryptedStore>,
    ) {
        let embeddings = Arc::new(MockEmbeddingProvider {
            calls: AtomicUsize::new(0),
        });
        let store = Arc::new(MockEncryptedStore::default());
        let cache = InMemorySemanticCache::new(
            config(max_entries, ttl),
            TenantAuthHash::from_digest([7; 32]),
            embeddings.clone(),
            store.clone(),
        )
        .unwrap();
        (cache, embeddings, store)
    }

    async fn insert_with(
        cache: &InMemorySemanticCache,
        request: &OpenAIRequest,
        quality_score: f64,
        provenance: ResponseProvenance,
        payload: &[u8],
    ) -> SemanticInsertOutcome {
        cache
            .insert(SemanticInsertRequest {
                request,
                model_group: "logical-group",
                context: context(request),
                quality_score: ComplexityScore::new(quality_score),
                decision: decision(),
                provenance,
                response_payload: payload,
            })
            .await
            .unwrap()
    }

    async fn insert(cache: &InMemorySemanticCache, request: &OpenAIRequest) -> u64 {
        let outcome = insert_with(
            cache,
            request,
            0.9,
            ResponseProvenance {
                model: "provider-model".to_owned(),
                tier: SmartRoutingTier::Balanced,
            },
            b"encrypted-by-store",
        )
        .await;
        let SemanticInsertOutcome::Inserted { entry_id } = outcome else {
            panic!("test request should be cacheable");
        };
        entry_id
    }

    async fn lookup(
        cache: &InMemorySemanticCache,
        request: &OpenAIRequest,
    ) -> MetadataLookupOutcome {
        cache
            .lookup_metadata(MetadataLookupRequest {
                request,
                model_group: "logical-group",
                context: context(request),
            })
            .await
            .unwrap()
    }

    fn cache_with_provider(
        cache_config: SemanticCacheConfig,
        tenant: [u8; 32],
        embedding_provider: Arc<dyn EmbeddingProvider>,
        store: Arc<MockEncryptedStore>,
    ) -> InMemorySemanticCache {
        InMemorySemanticCache::new(
            cache_config,
            TenantAuthHash::from_digest(tenant),
            embedding_provider,
            store,
        )
        .unwrap()
    }

    fn model_group() -> ModelGroup {
        ModelGroup {
            name: "logical-group".to_owned(),
            version_fallback_enabled: false,
            compression: None,
            memory: None,
            structured_output: None,
            models: vec![ProviderModel {
                provider: "provider".to_owned(),
                model: "provider-model".to_owned(),
                cost_per_million_input_tokens: 0.0,
                cost_per_million_output_tokens: 0.0,
                priority: 100,
                structured_output_passthrough: None,
                tier: Some(SmartRoutingTier::Balanced),
                context_window: 16_384,
                specializations: Vec::new(),
            }],
        }
    }

    fn smart_router_config(
        cache_enabled: bool,
        similarity_threshold: f64,
        min_quality: f64,
    ) -> SmartRoutingConfig {
        let mut routing = SmartRoutingConfig {
            enabled: true,
            classifier: ClassifierMode::Heuristic,
            ..SmartRoutingConfig::default()
        };
        routing.semantic_cache.enabled = cache_enabled;
        routing.semantic_cache.similarity_threshold = similarity_threshold;
        routing.semantic_cache.min_quality_score = min_quality;
        routing
    }

    #[tokio::test]
    async fn lookup_returns_metadata_and_payload_is_retrieved_separately() {
        let (cache, embeddings, store) = cache(4, Duration::from_secs(60));
        let original = request("explain rust ownership");
        let entry_id = insert(&cache, &original).await;
        let paraphrase = request("please explain ownership in rust");

        let outcome = cache
            .lookup_metadata(MetadataLookupRequest {
                request: &paraphrase,
                model_group: "logical-group",
                context: context(&paraphrase),
            })
            .await
            .unwrap();
        let MetadataLookupOutcome::Hit(hit) = outcome else {
            panic!("expected semantic hit");
        };
        assert_eq!(hit.entry_id, entry_id);
        assert_eq!(hit.provenance.model, "provider-model");
        assert_eq!(hit.provenance.tier, SmartRoutingTier::Balanced);
        assert!(hit.decision.cache_hit);
        assert_eq!(embeddings.calls.load(Ordering::SeqCst), 2);
        assert_eq!(store.put_calls.load(Ordering::SeqCst), 1);
        assert_eq!(
            cache.retrieve_payload(entry_id).await.unwrap(),
            Some(b"encrypted-by-store".to_vec())
        );
    }

    #[tokio::test]
    async fn canonical_scope_separates_system_instructions() {
        let (cache, _, _) = cache(4, Duration::from_secs(60));
        let mut original = request("same question");
        original.messages.insert(
            0,
            Message {
                role: "system".to_owned(),
                content: json!("answer as a lawyer"),
                extra: Map::new(),
            },
        );
        insert(&cache, &original).await;
        let mut changed = original.clone();
        changed.messages[0].content = json!("answer as a physician");

        assert_eq!(
            cache
                .lookup_metadata(MetadataLookupRequest {
                    request: &changed,
                    model_group: "logical-group",
                    context: context(&changed),
                })
                .await
                .unwrap(),
            MetadataLookupOutcome::Miss
        );
    }

    #[tokio::test]
    async fn access_updates_true_lru_order() {
        let (cache, _, _) = cache(2, Duration::from_secs(60));
        let first = request("first");
        let second = request("different second");
        let first_id = insert(&cache, &first).await;
        let second_id = insert(&cache, &second).await;
        let _ = cache
            .lookup_metadata(MetadataLookupRequest {
                request: &first,
                model_group: "logical-group",
                context: context(&first),
            })
            .await
            .unwrap();
        let third = request("third");
        insert(&cache, &third).await;

        assert!(cache.retrieve_payload(first_id).await.unwrap().is_some());
        assert!(cache.retrieve_payload(second_id).await.unwrap().is_none());
        assert_eq!(cache.len(), 2);
    }

    #[tokio::test]
    async fn expired_entries_are_not_returned() {
        let (cache, _, store) = cache(2, Duration::from_millis(50));
        let request = request("expires");
        let entry_id = insert(&cache, &request).await;
        tokio::time::sleep(Duration::from_millis(100)).await;

        assert_eq!(lookup(&cache, &request).await, MetadataLookupOutcome::Miss);
        assert!(cache.retrieve_payload(entry_id).await.unwrap().is_none());
        assert!(cache.is_empty());
        assert_eq!(store.delete_calls.load(Ordering::SeqCst), 1);
        assert!(store.values.lock().unwrap().is_empty());
    }

    #[tokio::test]
    async fn ineligible_requests_bypass_without_provider_or_store_calls() {
        let (cache, embeddings, store) = cache(2, Duration::from_secs(60));
        let mut tools = request("use a tool");
        tools.extra.insert("tools".to_owned(), json!([]));
        let mut stream = request("unbuffered stream");
        stream.stream = true;
        let mut personalized = request("personalized");
        personalized
            .extra
            .insert("user".to_owned(), json!("user-1"));
        let mut nondeterministic = request("sampled");
        nondeterministic.temperature = Some(0.7);
        let cases = [
            (tools, CacheIneligibility::Tools),
            (stream, CacheIneligibility::UnbufferedStream),
            (personalized, CacheIneligibility::Personalized),
            (nondeterministic, CacheIneligibility::Nondeterministic),
        ];

        for (request, expected) in cases {
            assert_eq!(
                cache
                    .insert(SemanticInsertRequest {
                        request: &request,
                        model_group: "logical-group",
                        context: context(&request),
                        quality_score: ComplexityScore::new(0.9),
                        decision: decision(),
                        provenance: ResponseProvenance {
                            model: "provider-model".to_owned(),
                            tier: SmartRoutingTier::Balanced,
                        },
                        response_payload: b"must-not-store",
                    })
                    .await
                    .unwrap(),
                SemanticInsertOutcome::Ineligible(expected)
            );
            assert_eq!(
                lookup(&cache, &request).await,
                MetadataLookupOutcome::Ineligible(expected)
            );
        }
        assert_eq!(embeddings.calls.load(Ordering::SeqCst), 0);
        assert_eq!(store.put_calls.load(Ordering::SeqCst), 0);
        assert_eq!(store.get_calls.load(Ordering::SeqCst), 0);
    }

    #[tokio::test]
    async fn disabled_smart_router_never_calls_cache() {
        let (cache, embeddings, store) = cache(2, Duration::from_secs(60));
        let request = request("disabled cache");
        let group = model_group();
        let pinned = PinnedRoutingContext::default();
        let router = SmartRouter::new(smart_router_config(false, 0.0, 0.0))
            .unwrap()
            .with_cache(Arc::new(cache));

        let outcome = router
            .plan(&SmartRoutingInput {
                request_id: "request-1",
                request: &request,
                model_group: &group,
                pinned_context: &pinned,
            })
            .await
            .unwrap();

        assert!(matches!(outcome, RoutingPlanOutcome::Route(_)));
        assert_eq!(embeddings.calls.load(Ordering::SeqCst), 0);
        assert_eq!(store.get_calls.load(Ordering::SeqCst), 0);
    }

    #[tokio::test]
    async fn failed_encrypted_store_insertion_never_indexes_payload() {
        let embeddings = Arc::new(MockEmbeddingProvider {
            calls: AtomicUsize::new(0),
        });
        let store = Arc::new(MockEncryptedStore {
            fail_put: true,
            ..MockEncryptedStore::default()
        });
        let cache = InMemorySemanticCache::new(
            config(2, Duration::from_secs(60)),
            TenantAuthHash::from_digest([9; 32]),
            embeddings,
            store,
        )
        .unwrap();
        let request = request("cannot persist");

        assert_eq!(
            cache
                .insert(SemanticInsertRequest {
                    request: &request,
                    model_group: "logical-group",
                    context: context(&request),
                    quality_score: ComplexityScore::new(0.9),
                    decision: decision(),
                    provenance: ResponseProvenance {
                        model: "provider-model".to_owned(),
                        tier: SmartRoutingTier::Balanced,
                    },
                    response_payload: b"plaintext",
                })
                .await,
            Err(SemanticInsertFailure::Store(
                PayloadStoreErrorKind::Encryption
            ))
        );
        assert!(cache.is_empty());
    }

    #[tokio::test]
    async fn embedding_failures_are_typed_and_never_touch_encrypted_storage() {
        let insert_provider = Arc::new(FixedEmbeddingProvider::new(vec![Err(
            EmbeddingProviderError::new(EmbeddingProviderErrorKind::Timeout),
        )]));
        let insert_store = Arc::new(MockEncryptedStore::default());
        let insert_cache = cache_with_provider(
            config(2, Duration::from_secs(60)),
            [9; 32],
            insert_provider,
            insert_store.clone(),
        );
        let request = request("embedding failure");

        assert_eq!(
            insert_cache
                .insert(SemanticInsertRequest {
                    request: &request,
                    model_group: "logical-group",
                    context: context(&request),
                    quality_score: ComplexityScore::new(0.9),
                    decision: decision(),
                    provenance: ResponseProvenance {
                        model: "provider-model".to_owned(),
                        tier: SmartRoutingTier::Balanced,
                    },
                    response_payload: b"must-not-store",
                })
                .await,
            Err(SemanticInsertFailure::Embedding(
                EmbeddingProviderErrorKind::Timeout
            ))
        );
        assert!(insert_cache.is_empty());
        assert_eq!(insert_store.put_calls.load(Ordering::SeqCst), 0);

        let lookup_provider = Arc::new(FixedEmbeddingProvider::new(vec![Err(
            EmbeddingProviderError::new(EmbeddingProviderErrorKind::Unavailable),
        )]));
        let lookup_store = Arc::new(MockEncryptedStore::default());
        let lookup_cache = cache_with_provider(
            config(2, Duration::from_secs(60)),
            [9; 32],
            lookup_provider,
            lookup_store.clone(),
        );
        assert_eq!(
            lookup_cache
                .lookup_metadata(MetadataLookupRequest {
                    request: &request,
                    model_group: "logical-group",
                    context: context(&request),
                })
                .await,
            Err(MetadataLookupFailure::Embedding(
                EmbeddingProviderErrorKind::Unavailable
            ))
        );
        assert_eq!(lookup_store.get_calls.load(Ordering::SeqCst), 0);
    }

    #[tokio::test]
    async fn tenant_policy_and_embedding_semantics_scopes_are_isolated() {
        let shared_store = Arc::new(MockEncryptedStore::default());
        let make_cache = |tenant, namespace: &str, policy: &str| {
            let mut cache_config = config(4, Duration::from_secs(60));
            cache_config.embedding_namespace = namespace.to_owned();
            cache_config.policy_version = policy.to_owned();
            InMemorySemanticCache::new(
                cache_config,
                TenantAuthHash::from_digest(tenant),
                Arc::new(MockEmbeddingProvider {
                    calls: AtomicUsize::new(0),
                }),
                shared_store.clone(),
            )
            .unwrap()
        };
        let tenant_a = make_cache([1; 32], "embedding-v1", "policy-v1");
        let tenant_b = make_cache([2; 32], "embedding-v1", "policy-v1");
        let embedding_v2 = make_cache([1; 32], "embedding-v2", "policy-v1");
        let policy_v2 = make_cache([1; 32], "embedding-v1", "policy-v2");
        let request = request("same semantic request");
        let tenant_a_scope = tenant_a.scope_key(&request, "logical-group");
        assert_ne!(
            tenant_a_scope,
            tenant_b.scope_key(&request, "logical-group")
        );
        assert_ne!(
            tenant_a_scope,
            embedding_v2.scope_key(&request, "logical-group")
        );
        assert_ne!(
            tenant_a_scope,
            policy_v2.scope_key(&request, "logical-group")
        );
        insert(&tenant_a, &request).await;

        assert!(matches!(
            lookup(&tenant_a, &request).await,
            MetadataLookupOutcome::Hit(_)
        ));
        assert_eq!(
            lookup(&tenant_b, &request).await,
            MetadataLookupOutcome::Miss
        );
        assert_eq!(
            lookup(&embedding_v2, &request).await,
            MetadataLookupOutcome::Miss
        );
        assert_eq!(
            lookup(&policy_v2, &request).await,
            MetadataLookupOutcome::Miss
        );
    }

    #[tokio::test]
    async fn encrypted_store_retrieval_failure_is_typed_without_removing_metadata() {
        let provider = Arc::new(MockEmbeddingProvider {
            calls: AtomicUsize::new(0),
        });
        let store = Arc::new(MockEncryptedStore {
            fail_get: true,
            ..MockEncryptedStore::default()
        });
        let cache = cache_with_provider(
            config(2, Duration::from_secs(60)),
            [9; 32],
            provider,
            store.clone(),
        );
        let request = request("payload retrieval failure");
        let entry_id = insert(&cache, &request).await;

        assert_eq!(
            cache.retrieve_payload(entry_id).await,
            Err(PayloadRetrievalFailure::Store(
                PayloadStoreErrorKind::Unavailable
            ))
        );
        assert_eq!(cache.len(), 1);
        assert_eq!(store.get_calls.load(Ordering::SeqCst), 1);
    }

    #[tokio::test]
    async fn replacement_and_lru_eviction_cleanup_payloads() {
        let (cache, _, store) = cache(2, Duration::from_secs(60));
        let first = request("first");
        let second = request("different second");
        let third = request("third");
        let old_first_id = insert(&cache, &first).await;
        let new_first_id = insert(&cache, &first).await;
        assert_ne!(old_first_id, new_first_id);
        assert_eq!(store.delete_calls.load(Ordering::SeqCst), 1);
        assert!(cache
            .retrieve_payload(old_first_id)
            .await
            .unwrap()
            .is_none());

        let second_id = insert(&cache, &second).await;
        let _ = lookup(&cache, &first).await;
        insert(&cache, &third).await;
        assert!(cache
            .retrieve_payload(new_first_id)
            .await
            .unwrap()
            .is_some());
        assert!(cache.retrieve_payload(second_id).await.unwrap().is_none());
        assert_eq!(store.delete_calls.load(Ordering::SeqCst), 2);
        assert_eq!(store.values.lock().unwrap().len(), 2);
    }

    #[tokio::test]
    async fn metadata_hit_preserves_provenance_and_marks_decision_as_cached() {
        let (cache, _, _) = cache(2, Duration::from_secs(60));
        let request = request("provenance");
        let expected = ResponseProvenance {
            model: "specific-provider-model".to_owned(),
            tier: SmartRoutingTier::Powerful,
        };
        insert_with(&cache, &request, 0.95, expected.clone(), b"response").await;

        let MetadataLookupOutcome::Hit(hit) = lookup(&cache, &request).await else {
            panic!("expected metadata hit");
        };
        assert_eq!(hit.provenance, expected);
        assert_eq!(hit.quality_score, ComplexityScore::new(0.95));
        assert!(hit.decision.cache_hit);
        assert!(!decision().cache_hit);
    }

    #[tokio::test]
    async fn property_29_cosine_similarity_threshold_has_256_cases() {
        let strategy = (
            0.01f64..=0.99,
            prop::sample::select(vec![-1.0_f32, 1.0_f32]),
            prop::sample::select(vec![-1.0_f32, 1.0_f32]),
        );
        let mut runner = TestRunner::new(ProptestConfig::with_cases(256));
        let generated_cases = Mutex::new(Vec::with_capacity(256));
        runner
            .run(&strategy, |generated_case| {
                generated_cases.lock().unwrap().push(generated_case);
                Ok(())
            })
            .unwrap();

        for (threshold, first_sign, second_sign) in generated_cases.into_inner().unwrap() {
            let angle = threshold.acos();
            let entry_vector = vec![1.0, 0.0];
            let at_threshold = vec![threshold as f32, angle.sin() as f32 * first_sign];
            let actual_at_threshold = cosine_similarity(&entry_vector, &at_threshold).unwrap();
            let configured_threshold = actual_at_threshold.min(threshold);
            let below = configured_threshold * 0.5;
            let below_angle = below.acos();
            let below_threshold = vec![below as f32, below_angle.sin() as f32 * second_sign];
            let provider = Arc::new(FixedEmbeddingProvider::new(vec![
                Ok(entry_vector),
                Ok(at_threshold),
                Ok(below_threshold),
            ]));
            let store = Arc::new(MockEncryptedStore::default());
            let mut cache_config = config(1, Duration::from_secs(60));
            cache_config.similarity_threshold = configured_threshold;
            let cache = cache_with_provider(cache_config, [9; 32], provider, store);
            let request = request("threshold property");
            insert(&cache, &request).await;

            let MetadataLookupOutcome::Hit(hit) = lookup(&cache, &request).await else {
                panic!("cosine similarity at the configured threshold must hit");
            };
            assert!(hit.similarity.value() + 1.0e-12 >= configured_threshold);
            assert_eq!(lookup(&cache, &request).await, MetadataLookupOutcome::Miss);
        }
    }

    #[tokio::test]
    async fn property_30_quality_filter_has_256_cases() {
        let strategy = (0.0f64..=1.0, 0.0f64..=1.0);
        let mut runner = TestRunner::new(ProptestConfig::with_cases(256));
        let generated_cases = Mutex::new(Vec::with_capacity(256));
        runner
            .run(&strategy, |generated_case| {
                generated_cases.lock().unwrap().push(generated_case);
                Ok(())
            })
            .unwrap();

        for (minimum, quality) in generated_cases.into_inner().unwrap() {
            let provider = Arc::new(MockEmbeddingProvider {
                calls: AtomicUsize::new(0),
            });
            let store = Arc::new(MockEncryptedStore::default());
            let mut cache_config = config(1, Duration::from_secs(60));
            cache_config.min_quality_score = minimum;
            let cache = cache_with_provider(cache_config, [9; 32], provider.clone(), store.clone());
            let request = request("quality property");
            let outcome = insert_with(
                &cache,
                &request,
                quality,
                ResponseProvenance {
                    model: "quality-model".to_owned(),
                    tier: SmartRoutingTier::Fast,
                },
                b"quality-response",
            )
            .await;

            if quality < minimum {
                assert_eq!(
                    outcome,
                    SemanticInsertOutcome::Ineligible(CacheIneligibility::QualityBelowMinimum)
                );
                assert!(cache.is_empty());
                assert_eq!(provider.calls.load(Ordering::SeqCst), 0);
                assert_eq!(store.put_calls.load(Ordering::SeqCst), 0);
            } else {
                assert!(matches!(outcome, SemanticInsertOutcome::Inserted { .. }));
                let MetadataLookupOutcome::Hit(hit) = lookup(&cache, &request).await else {
                    panic!("quality at or above minimum must be eligible for lookup");
                };
                assert_eq!(hit.quality_score, ComplexityScore::new(quality));
            }
        }
    }

    #[test]
    fn tenant_hash_debug_is_redacted() {
        assert_eq!(
            format!("{:?}", TenantAuthHash::from_digest([42; 32])),
            "TenantAuthHash([redacted])"
        );
        assert_eq!(
            format!("{:?}", OpaquePayloadRef::new(vec![1, 2, 3]).unwrap()),
            "OpaquePayloadRef([redacted])"
        );
    }
}
