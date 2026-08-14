//! SQLite-backed persistence for cross-session memories.
//!
//! SQLite is the source of truth. The external-content FTS5 table mirrors the
//! memory content through triggers, while vector indexing state is persisted so
//! a later Qdrant integration can retry failed work without losing entries.
//! Timestamps are stored as UTC epoch seconds (`INTEGER`), matching `KeyStore`.

use std::collections::BTreeMap;
use std::fmt;
use std::path::{Path, PathBuf};
use std::sync::{Arc, Mutex, MutexGuard};

use chrono::{DateTime, Utc};
use rusqlite::{params, Connection, OptionalExtension, Row};
use uuid::Uuid;

use super::metrics::{MemoryMetrics, NamespaceType};
use super::namespace::validate_namespace as is_valid_namespace;
use super::{MemoryEntry, MemoryError, MemoryType, ScoredMemory};

const SCHEMA_VERSION: i64 = 3;
pub const DEFAULT_MAX_MEMORIES_PER_NAMESPACE: usize = 1_000;
const MAX_LIST_LIMIT: usize = 200;
const MAX_RETRIEVAL_CANDIDATES: usize = 50;
const DEFAULT_MIN_RELEVANCE_THRESHOLD: f64 = 0.1;
pub(crate) const CONTEXT_SCOPE_BOOST: f64 = 1.5;
const SECONDS_PER_DAY: f64 = 86_400.0;

#[derive(Debug, Clone, PartialEq)]
pub struct NewMemoryEntry {
    pub namespace: String,
    pub content: String,
    pub memory_type: MemoryType,
    pub source_request_id: Option<Uuid>,
}

#[derive(Debug, Clone, PartialEq)]
pub enum MemoryEntryInput {
    New(NewMemoryEntry),
    Entry(MemoryEntry),
}

impl From<NewMemoryEntry> for MemoryEntryInput {
    fn from(value: NewMemoryEntry) -> Self {
        Self::New(value)
    }
}

impl From<MemoryEntry> for MemoryEntryInput {
    fn from(value: MemoryEntry) -> Self {
        Self::Entry(value)
    }
}

#[derive(Debug, Clone, PartialEq)]
pub struct MemoryEntryPage {
    pub entries: Vec<MemoryEntry>,
    pub total_count: u64,
    pub limit: usize,
    pub offset: usize,
}

#[derive(Debug, Clone, PartialEq)]
pub struct MemoryStats {
    pub total_count: u64,
    pub memories_per_namespace: BTreeMap<String, u64>,
    pub average_relevance_score: f64,
    pub storage_size_bytes: Option<u64>,
    pub last_decay_cycle: Option<DateTime<Utc>>,
}

#[derive(Debug, Clone, PartialEq)]
pub struct MemoryNamespace {
    pub namespace: String,
    pub context_kind: String,
    pub display_name: Option<String>,
    pub client_name: Option<String>,
    pub entry_count: u64,
    pub last_activity: DateTime<Utc>,
}

#[derive(Debug, Clone, PartialEq)]
pub struct NamespaceEviction {
    pub namespace: String,
    pub evicted_count: u64,
    pub lowest_evicted_score: f64,
}

#[derive(Debug, Clone, PartialEq)]
pub struct DecayCycleResult {
    pub completed_at: DateTime<Utc>,
    pub decayed_count: u64,
    pub evicted_count: u64,
    pub vector_retry_pending_count: u64,
    pub namespace_evictions: Vec<NamespaceEviction>,
}

#[derive(Debug, Clone, PartialEq)]
pub struct VectorRetryCandidate {
    pub entry: MemoryEntry,
    pub retry_count: u64,
    pub last_attempt_at: Option<DateTime<Utc>>,
    pub next_retry_at: Option<DateTime<Utc>>,
    pub last_error: Option<String>,
}

/// SQLite-backed memory store. Clones share one synchronous connection.
#[derive(Clone)]
pub struct MemoryStore {
    conn: Arc<Mutex<Connection>>,
    db_path: Option<PathBuf>,
    metrics: Arc<MemoryMetrics>,
}

impl MemoryStore {
    /// Open or create a memory database and bring its schema up to date.
    pub fn new(db_path: &Path) -> Result<Self, MemoryError> {
        Self::with_metrics(db_path, Arc::new(MemoryMetrics::new()))
    }

    pub(crate) fn with_metrics(
        db_path: &Path,
        metrics: Arc<MemoryMetrics>,
    ) -> Result<Self, MemoryError> {
        if let Some(parent) = db_path.parent() {
            if !parent.as_os_str().is_empty() {
                std::fs::create_dir_all(parent).map_err(store_external_error)?;
            }
        }

        let mut conn = Connection::open(db_path)?;
        Self::verify_fts5(&conn)?;
        Self::initialize_schema(&mut conn)?;
        let db_path = (db_path != Path::new(":memory:")).then(|| db_path.to_path_buf());

        Ok(Self {
            conn: Arc::new(Mutex::new(conn)),
            db_path,
            metrics,
        })
    }

    /// Lock the synchronous connection for later storage operations.
    ///
    /// Callers must release the guard before reaching an `.await` point.
    pub(crate) fn connection(&self) -> Result<MutexGuard<'_, Connection>, MemoryError> {
        self.conn.lock().map_err(|_| {
            store_external_error(ConnectionPoisoned(
                "memory SQLite connection mutex was poisoned",
            ))
        })
    }

    pub fn store_entry(
        &self,
        input: impl Into<MemoryEntryInput>,
        max_memories_per_namespace: Option<usize>,
    ) -> Result<MemoryEntry, MemoryError> {
        let entry = match input.into() {
            MemoryEntryInput::Entry(entry) => entry,
            MemoryEntryInput::New(input) => {
                let now = Utc::now();
                MemoryEntry {
                    id: Uuid::new_v4(),
                    namespace: input.namespace,
                    content: input.content,
                    memory_type: input.memory_type,
                    relevance_score: 1.0,
                    created_at: now,
                    last_accessed_at: now,
                    access_count: 0,
                    source_request_id: input.source_request_id,
                }
            }
        };
        validate_entry(&entry)?;
        let cap = max_memories_per_namespace.unwrap_or(DEFAULT_MAX_MEMORIES_PER_NAMESPACE);
        if cap == 0 {
            return Err(MemoryError::Config(
                "max_memories_per_namespace must be at least 1".to_string(),
            ));
        }
        let cap = i64::try_from(cap).map_err(store_external_error)?;
        let mut conn = self.connection()?;
        let transaction = conn.transaction()?;
        transaction.execute(
            "INSERT INTO memories (
                id, namespace, content, memory_type, relevance_score,
                created_at, last_accessed_at, access_count, source_request_id
             ) VALUES (?1, ?2, ?3, ?4, ?5, ?6, ?7, ?8, ?9)",
            params![
                entry.id.to_string(),
                entry.namespace,
                entry.content,
                memory_type_as_str(entry.memory_type),
                entry.relevance_score,
                entry.created_at.timestamp(),
                entry.last_accessed_at.timestamp(),
                i64::try_from(entry.access_count).map_err(store_external_error)?,
                entry.source_request_id.map(|id| id.to_string()),
            ],
        )?;
        transaction.execute(
            "DELETE FROM memories
             WHERE id IN (
                SELECT id FROM memories
                WHERE namespace = ?1
                ORDER BY relevance_score ASC, last_accessed_at ASC,
                         created_at ASC, id ASC
                LIMIT CASE
                    WHEN (SELECT count(*) FROM memories WHERE namespace = ?1) > ?2
                    THEN (SELECT count(*) FROM memories WHERE namespace = ?1) - ?2
                    ELSE 0
                END
             )",
            params![entry.namespace, cap],
        )?;
        transaction.commit()?;
        Ok(entry)
    }

    pub fn get_entry_by_id(&self, id: Uuid) -> Result<Option<MemoryEntry>, MemoryError> {
        let conn = self.connection()?;
        conn.query_row(
            "SELECT id, namespace, content, memory_type, relevance_score,
                    created_at, last_accessed_at, access_count, source_request_id
             FROM memories WHERE id = ?1",
            [id.to_string()],
            memory_entry_from_row,
        )
        .optional()
        .map_err(Into::into)
    }

    pub fn delete_entry(&self, id: Uuid) -> Result<bool, MemoryError> {
        Ok(self
            .connection()?
            .execute("DELETE FROM memories WHERE id = ?1", [id.to_string()])?
            != 0)
    }

    pub fn delete_namespace(&self, namespace: &str) -> Result<u64, MemoryError> {
        validate_namespace(namespace)?;
        let deleted = self
            .connection()?
            .execute("DELETE FROM memories WHERE namespace = ?1", [namespace])?;
        u64::try_from(deleted).map_err(store_external_error)
    }

    pub fn list_entries(
        &self,
        namespace: &str,
        limit: usize,
        offset: usize,
    ) -> Result<MemoryEntryPage, MemoryError> {
        validate_namespace(namespace)?;
        let limit = limit.min(MAX_LIST_LIMIT);
        let sql_limit = i64::try_from(limit).map_err(store_external_error)?;
        let sql_offset = i64::try_from(offset).map_err(store_external_error)?;
        let conn = self.connection()?;
        let total_count = query_u64(
            &conn,
            "SELECT count(*) FROM memories WHERE namespace = ?1",
            [namespace],
        )?;
        let mut statement = conn.prepare(
            "SELECT id, namespace, content, memory_type, relevance_score,
                    created_at, last_accessed_at, access_count, source_request_id
             FROM memories WHERE namespace = ?1
             ORDER BY created_at DESC, id ASC LIMIT ?2 OFFSET ?3",
        )?;
        let entries = statement
            .query_map(
                params![namespace, sql_limit, sql_offset],
                memory_entry_from_row,
            )?
            .collect::<Result<Vec<_>, _>>()?;
        Ok(MemoryEntryPage {
            entries,
            total_count,
            limit,
            offset,
        })
    }

    pub fn namespace_count(&self, namespace: &str) -> Result<u64, MemoryError> {
        validate_namespace(namespace)?;
        let conn = self.connection()?;
        query_u64(
            &conn,
            "SELECT count(*) FROM memories WHERE namespace = ?1",
            [namespace],
        )
    }

    pub fn mark_unconfigured_vector_entries_pending(&self) -> Result<u64, MemoryError> {
        let updated = self.connection()?.execute(
            "UPDATE memories SET vector_index_status = 'pending'
WHERE vector_index_status = 'not_configured'",
            [],
        )?;
        u64::try_from(updated).map_err(store_external_error)
    }

    pub fn mark_all_vector_entries_pending(&self) -> Result<u64, MemoryError> {
        let updated = self
            .connection()?
            .execute("UPDATE memories SET vector_index_status = 'pending'", [])?;
        u64::try_from(updated).map_err(store_external_error)
    }

    pub fn mark_vector_pending(&self, id: Uuid) -> Result<bool, MemoryError> {
        Ok(self.connection()?.execute(
            "UPDATE memories
             SET vector_index_status = 'pending', vector_next_retry_at = NULL,
                 vector_last_error = NULL
             WHERE id = ?1",
            [id.to_string()],
        )? != 0)
    }

    pub fn mark_vector_indexed(
        &self,
        id: Uuid,
        attempted_at: DateTime<Utc>,
    ) -> Result<bool, MemoryError> {
        Ok(self.connection()?.execute(
            "UPDATE memories
             SET vector_index_status = 'indexed', vector_last_attempt_at = ?2,
                 vector_next_retry_at = NULL, vector_last_error = NULL
             WHERE id = ?1",
            params![id.to_string(), attempted_at.timestamp()],
        )? != 0)
    }

    pub fn mark_vector_retry(
        &self,
        id: Uuid,
        attempted_at: DateTime<Utc>,
        next_retry_at: DateTime<Utc>,
        error: &str,
    ) -> Result<bool, MemoryError> {
        let sanitized_error = sanitize_vector_error(error);
        Ok(self.connection()?.execute(
            "UPDATE memories
             SET vector_index_status = 'retry',
                 vector_retry_count = vector_retry_count + 1,
                 vector_last_attempt_at = ?2, vector_next_retry_at = ?3,
                 vector_last_error = ?4
             WHERE id = ?1",
            params![
                id.to_string(),
                attempted_at.timestamp(),
                next_retry_at.timestamp(),
                sanitized_error,
            ],
        )? != 0)
    }

    pub fn vector_status(
        &self,
        id: Uuid,
    ) -> Result<Option<(String, u64, Option<String>)>, MemoryError> {
        self.connection()?
            .query_row(
                "SELECT vector_index_status, vector_retry_count, vector_last_error
                 FROM memories WHERE id = ?1",
                [id.to_string()],
                |row| Ok((row.get(0)?, checked_db_u64(row.get(1)?, 1)?, row.get(2)?)),
            )
            .optional()
            .map_err(Into::into)
    }

    pub fn get_entries_by_ids(&self, ids: &[Uuid]) -> Result<Vec<MemoryEntry>, MemoryError> {
        let mut entries = Vec::with_capacity(ids.len());
        for id in ids {
            if let Some(entry) = self.get_entry_by_id(*id)? {
                entries.push(entry);
            }
        }
        Ok(entries)
    }

    pub fn retrieve(
        &self,
        user_scope: &str,
        context_scope: Option<&str>,
        query: &str,
        minimum_relevance_threshold: Option<f64>,
    ) -> Result<Vec<ScoredMemory>, MemoryError> {
        validate_namespace(user_scope)?;
        if let Some(context_scope) = context_scope {
            validate_namespace(context_scope)?;
        }
        let Some(fts_query) = fts_query_from_user_text(query) else {
            return Ok(Vec::new());
        };
        let threshold = minimum_relevance_threshold.unwrap_or(DEFAULT_MIN_RELEVANCE_THRESHOLD);
        if !threshold.is_finite() || threshold < 0.0 {
            return Err(MemoryError::Config(
                "minimum relevance threshold must be finite and non-negative".to_string(),
            ));
        }
        let now = Utc::now();
        let conn = self.connection()?;
        let scope_candidate_limit = i64::try_from(MAX_RETRIEVAL_CANDIDATES)
            .map_err(store_external_error)?
            .checked_mul(if context_scope.is_some() { 2 } else { 1 })
            .ok_or_else(|| store_external_error(NumericOverflow))?;
        let mut statement = conn.prepare(
            "SELECT m.id, m.namespace, m.content, m.memory_type, m.relevance_score,
                    m.created_at, m.last_accessed_at, m.access_count, m.source_request_id,
                    bm25(memories_fts) AS raw_rank
             FROM memories_fts
             JOIN memories m ON m.rowid = memories_fts.rowid
             WHERE memories_fts MATCH ?1
               AND (m.namespace = ?2 OR (?3 IS NOT NULL AND m.namespace = ?3))
             ORDER BY raw_rank ASC, m.rowid ASC
             LIMIT ?4",
        )?;
        let rows = statement.query_map(
            params![fts_query, user_scope, context_scope, scope_candidate_limit],
            |row| Ok((memory_entry_from_row(row)?, row.get::<_, f64>(9)?)),
        )?;
        let mut candidates = Vec::new();
        for row in rows {
            let (entry, raw_rank) = row?;
            let normalized_rank = normalize_fts_rank(raw_rank);
            let elapsed_seconds = now
                .signed_duration_since(entry.last_accessed_at)
                .num_milliseconds() as f64
                / 1_000.0;
            let days_since_access = (elapsed_seconds / SECONDS_PER_DAY).max(0.0);
            let recency_boost = 1.0 / (1.0 + days_since_access * 0.1);
            let scope_boost = if context_scope == Some(entry.namespace.as_str()) {
                CONTEXT_SCOPE_BOOST
            } else {
                1.0
            };
            let final_score = normalized_rank * entry.relevance_score * recency_boost * scope_boost;
            if final_score.is_finite() && final_score >= threshold {
                candidates.push(ScoredMemory {
                    entry,
                    final_score,
                    estimated_tokens: 0,
                });
            }
        }

        candidates.sort_by(|left, right| right.final_score.total_cmp(&left.final_score));
        let mut deduplicated = Vec::with_capacity(candidates.len());
        for candidate in candidates {
            let trimmed = candidate.entry.content.trim();
            if let Some(existing_index) = deduplicated
                .iter()
                .position(|existing: &ScoredMemory| existing.entry.content.trim() == trimmed)
            {
                let candidate_is_context =
                    context_scope == Some(candidate.entry.namespace.as_str());
                let existing_is_context =
                    context_scope == Some(deduplicated[existing_index].entry.namespace.as_str());
                if candidate_is_context && !existing_is_context {
                    deduplicated[existing_index] = candidate;
                }
            } else {
                deduplicated.push(candidate);
            }
        }
        deduplicated.sort_by(|left, right| right.final_score.total_cmp(&left.final_score));
        deduplicated.truncate(MAX_RETRIEVAL_CANDIDATES);
        let namespace_type = context_scope
            .map(NamespaceType::from_namespace)
            .unwrap_or_else(|| NamespaceType::from_namespace(user_scope));
        self.metrics.record_retrievals(
            namespace_type,
            u64::try_from(deduplicated.len()).unwrap_or(u64::MAX),
        );
        Ok(deduplicated)
    }

    pub fn find_duplicate(
        &self,
        namespace: &str,
        content: &str,
    ) -> Result<Option<MemoryEntry>, MemoryError> {
        validate_namespace(namespace)?;
        let needle = content.trim();
        let conn = self.connection()?;
        let mut statement = conn.prepare(
            "SELECT id, namespace, content, memory_type, relevance_score,
                    created_at, last_accessed_at, access_count, source_request_id
             FROM memories WHERE namespace = ?1 ORDER BY created_at ASC, id ASC",
        )?;
        let mut rows = statement.query([namespace])?;
        while let Some(row) = rows.next()? {
            let entry = memory_entry_from_row(row)?;
            if entry.content.trim() == needle {
                return Ok(Some(entry));
            }
        }
        Ok(None)
    }

    pub fn update_access_metadata(
        &self,
        ids: &[Uuid],
        accessed_at: DateTime<Utc>,
    ) -> Result<u64, MemoryError> {
        let mut conn = self.connection()?;
        let transaction = conn.transaction()?;
        let mut updated = 0_u64;
        {
            let mut statement = transaction.prepare(
                "UPDATE memories
                 SET access_count = access_count + 1,
                     last_accessed_at = ?1,
                     relevance_score = min(1.0, relevance_score + 0.1)
                 WHERE id = ?2",
            )?;
            for id in ids {
                updated = updated
                    .checked_add(
                        u64::try_from(
                            statement.execute(params![accessed_at.timestamp(), id.to_string()])?,
                        )
                        .map_err(store_external_error)?,
                    )
                    .ok_or_else(|| store_external_error(NumericOverflow))?;
            }
        }
        transaction.commit()?;
        Ok(updated)
    }

    pub fn stats(&self) -> Result<MemoryStats, MemoryError> {
        let conn = self.connection()?;
        let (total_count, average_relevance_score): (i64, f64) = conn.query_row(
            "SELECT count(*), coalesce(avg(relevance_score), 0.0) FROM memories",
            [],
            |row| Ok((row.get(0)?, row.get(1)?)),
        )?;
        let mut statement = conn.prepare(
            "SELECT namespace, count(*) FROM memories GROUP BY namespace ORDER BY namespace ASC",
        )?;
        let rows = statement.query_map([], |row| {
            Ok((row.get::<_, String>(0)?, checked_db_u64(row.get(1)?, 1)?))
        })?;
        let memories_per_namespace = rows.collect::<Result<BTreeMap<_, _>, _>>()?;
        let last_decay_cycle = metadata_timestamp(&conn, "last_decay_cycle")?;
        drop(statement);
        drop(conn);
        let storage_size_bytes = self
            .db_path
            .as_ref()
            .map(std::fs::metadata)
            .transpose()
            .map_err(store_external_error)?
            .map(|metadata| metadata.len());
        Ok(MemoryStats {
            total_count: u64::try_from(total_count).map_err(store_external_error)?,
            memories_per_namespace,
            average_relevance_score,
            storage_size_bytes,
            last_decay_cycle,
        })
    }

    pub fn list_namespaces(&self) -> Result<Vec<MemoryNamespace>, MemoryError> {
        let conn = self.connection()?;
        let mut statement = conn.prepare(
            "SELECT m.namespace,
                    CASE WHEN instr(m.namespace, '::project::') > 0 THEN 'project'
                         WHEN instr(m.namespace, '::agent::') > 0 THEN 'agent'
                         ELSE 'user' END,
                    n.display_name, n.client_name, count(*), max(m.last_accessed_at)
             FROM memories m
             LEFT JOIN memory_namespaces n ON n.namespace = m.namespace
             GROUP BY m.namespace, n.display_name, n.client_name
             ORDER BY m.namespace ASC",
        )?;
        let namespaces = statement
            .query_map([], |row| {
                Ok(MemoryNamespace {
                    namespace: row.get(0)?,
                    context_kind: row.get(1)?,
                    display_name: row.get(2)?,
                    client_name: row.get(3)?,
                    entry_count: checked_db_u64(row.get(4)?, 4)?,
                    last_activity: timestamp_from_db(row.get(5)?, 5)?,
                })
            })?
            .collect::<Result<Vec<_>, _>>()?;
        Ok(namespaces)
    }

    pub fn upsert_namespace_labels(
        &self,
        namespace: &str,
        context_kind: &str,
        display_name: Option<&str>,
        client_name: Option<&str>,
    ) -> Result<(), MemoryError> {
        validate_namespace(namespace)?;
        if !matches!(context_kind, "project" | "agent" | "user") {
            return Err(MemoryError::Config("invalid context kind".to_owned()));
        }
        self.connection()?.execute(
            "INSERT INTO memory_namespaces(namespace, context_kind, display_name, client_name)
             VALUES (?1, ?2, ?3, ?4)
             ON CONFLICT(namespace) DO UPDATE SET
               context_kind = excluded.context_kind,
               display_name = coalesce(excluded.display_name, memory_namespaces.display_name),
               client_name = coalesce(excluded.client_name, memory_namespaces.client_name)",
            params![namespace, context_kind, display_name, client_name],
        )?;
        Ok(())
    }

    pub fn run_decay_cycle(
        &self,
        max_memories_per_namespace: usize,
        completed_at: DateTime<Utc>,
    ) -> Result<DecayCycleResult, MemoryError> {
        if max_memories_per_namespace == 0 {
            return Err(MemoryError::Config(
                "max_memories_per_namespace must be at least 1".to_string(),
            ));
        }
        let cap = i64::try_from(max_memories_per_namespace).map_err(store_external_error)?;
        let mut conn = self.connection()?;
        let transaction = conn.transaction()?;
        let decayed_count = u64::try_from(transaction.execute(
            "UPDATE memories SET relevance_score = max(0.0, relevance_score * CASE memory_type
WHEN 'preference' THEN 0.99
WHEN 'fact' THEN 0.95
WHEN 'context' THEN 0.85
WHEN 'decision' THEN 0.98
END)",
            [],
        )?)
        .map_err(store_external_error)?;
        let mut namespace_evictions = Vec::new();
        {
            let mut statement = transaction.prepare(
                "SELECT namespace, count(*) - ?1 AS excess
FROM memories
GROUP BY namespace
HAVING count(*) > ?1
ORDER BY namespace ASC",
            )?;
            let over_cap = statement
                .query_map([cap], |row| {
                    Ok((row.get::<_, String>(0)?, row.get::<_, i64>(1)?))
                })?
                .collect::<Result<Vec<_>, _>>()?;
            drop(statement);
            let mut eviction_statement = transaction.prepare(
                "SELECT relevance_score FROM memories
WHERE namespace = ?1
ORDER BY relevance_score ASC, last_accessed_at ASC, created_at ASC, id ASC
LIMIT ?2",
            )?;
            let mut delete_statement = transaction.prepare(
                "DELETE FROM memories WHERE id IN (
SELECT id FROM memories
WHERE namespace = ?1
ORDER BY relevance_score ASC, last_accessed_at ASC, created_at ASC, id ASC
LIMIT ?2
)",
            )?;
            for (namespace, excess) in over_cap {
                let lowest_evicted_score =
                    eviction_statement.query_row(params![namespace, excess], |row| row.get(0))?;
                let evicted_count =
                    u64::try_from(delete_statement.execute(params![namespace, excess])?)
                        .map_err(store_external_error)?;
                namespace_evictions.push(NamespaceEviction {
                    namespace,
                    evicted_count,
                    lowest_evicted_score,
                });
            }
        }
        let vector_retry_pending_count = query_u64(
            &transaction,
            "SELECT count(*) FROM memories
WHERE vector_index_status IN ('pending', 'retry')
AND (vector_next_retry_at IS NULL OR vector_next_retry_at <= ?1)",
            [completed_at.timestamp()],
        )?;
        transaction.execute(
            "INSERT INTO memory_metadata(key, value) VALUES ('last_decay_cycle', ?1)
ON CONFLICT(key) DO UPDATE SET value = excluded.value",
            [completed_at.timestamp().to_string()],
        )?;
        transaction.commit()?;
        let evicted_count = namespace_evictions
            .iter()
            .try_fold(0_u64, |total, eviction| {
                total
                    .checked_add(eviction.evicted_count)
                    .ok_or_else(|| store_external_error(NumericOverflow))
            })?;
        Ok(DecayCycleResult {
            completed_at,
            decayed_count,
            evicted_count,
            vector_retry_pending_count,
            namespace_evictions,
        })
    }

    pub fn list_pending_vector_retries(
        &self,
        as_of: DateTime<Utc>,
    ) -> Result<Vec<VectorRetryCandidate>, MemoryError> {
        let conn = self.connection()?;
        let mut statement = conn.prepare(
            "SELECT id, namespace, content, memory_type, relevance_score,
created_at, last_accessed_at, access_count, source_request_id,
vector_retry_count, vector_last_attempt_at, vector_next_retry_at, vector_last_error
FROM memories
WHERE vector_index_status IN ('pending', 'retry')
AND (vector_next_retry_at IS NULL OR vector_next_retry_at <= ?1)
ORDER BY coalesce(vector_next_retry_at, 0) ASC, id ASC",
        )?;
        let candidates = statement
            .query_map([as_of.timestamp()], |row| {
                Ok(VectorRetryCandidate {
                    entry: memory_entry_from_row(row)?,
                    retry_count: checked_db_u64(row.get(9)?, 9)?,
                    last_attempt_at: optional_timestamp_from_db(row.get(10)?, 10)?,
                    next_retry_at: optional_timestamp_from_db(row.get(11)?, 11)?,
                    last_error: row.get(12)?,
                })
            })?
            .collect::<Result<Vec<_>, _>>()?;
        Ok(candidates)
    }

    pub fn record_decay_cycle(&self, completed_at: DateTime<Utc>) -> Result<(), MemoryError> {
        self.connection()?.execute(
            "INSERT INTO memory_metadata(key, value) VALUES ('last_decay_cycle', ?1)
             ON CONFLICT(key) DO UPDATE SET value = excluded.value",
            [completed_at.timestamp().to_string()],
        )?;
        Ok(())
    }

    fn verify_fts5(conn: &Connection) -> Result<(), MemoryError> {
        conn.execute_batch(
            "CREATE VIRTUAL TABLE temp.memory_fts5_startup_probe USING fts5(content);
             DROP TABLE temp.memory_fts5_startup_probe;",
        )?;
        Ok(())
    }

    fn initialize_schema(conn: &mut Connection) -> Result<(), MemoryError> {
        let transaction = conn.transaction()?;
        transaction.execute_batch(
            "CREATE TABLE IF NOT EXISTS memories (
                id TEXT PRIMARY KEY,
                namespace TEXT NOT NULL CHECK(length(namespace) BETWEEN 1 AND 256),
                content TEXT NOT NULL CHECK(length(content) BETWEEN 1 AND 4096),
                memory_type TEXT NOT NULL
                    CHECK(memory_type IN ('preference', 'fact', 'context', 'decision')),
                relevance_score REAL NOT NULL DEFAULT 1.0
                    CHECK(relevance_score >= 0.0 AND relevance_score <= 1.0),
                created_at INTEGER NOT NULL,
                last_accessed_at INTEGER NOT NULL,
                access_count INTEGER NOT NULL DEFAULT 0 CHECK(access_count >= 0),
                source_request_id TEXT,
                vector_index_status TEXT NOT NULL DEFAULT 'not_configured'
                    CHECK(vector_index_status IN ('not_configured', 'pending', 'indexed', 'retry')),
                vector_retry_count INTEGER NOT NULL DEFAULT 0 CHECK(vector_retry_count >= 0),
                vector_last_attempt_at INTEGER,
                vector_next_retry_at INTEGER,
                vector_last_error TEXT
            );
        CREATE TABLE IF NOT EXISTS memory_metadata (
            key TEXT PRIMARY KEY,
            value TEXT NOT NULL
        );
        CREATE TABLE IF NOT EXISTS memory_namespaces (
            namespace TEXT PRIMARY KEY,
            context_kind TEXT NOT NULL CHECK(context_kind IN ('project', 'agent', 'user')),
            display_name TEXT CHECK(display_name IS NULL OR length(display_name) BETWEEN 1 AND 64),
            client_name TEXT CHECK(client_name IS NULL OR length(client_name) BETWEEN 1 AND 64)
        );",
        )?;
        Self::migrate_schema(&transaction)?;
        transaction.execute_batch(
            "CREATE VIRTUAL TABLE IF NOT EXISTS memories_fts USING fts5(
                content, content='memories', content_rowid='rowid'
            );
            DROP TRIGGER IF EXISTS memories_ai;
            DROP TRIGGER IF EXISTS memories_ad;
            DROP TRIGGER IF EXISTS memories_au;
            CREATE TRIGGER memories_ai AFTER INSERT ON memories BEGIN
                INSERT INTO memories_fts(rowid, content) VALUES (new.rowid, new.content);
            END;
            CREATE TRIGGER memories_ad AFTER DELETE ON memories BEGIN
                INSERT INTO memories_fts(memories_fts, rowid, content)
                VALUES ('delete', old.rowid, old.content);
            END;
            CREATE TRIGGER memories_au AFTER UPDATE ON memories BEGIN
                INSERT INTO memories_fts(memories_fts, rowid, content)
                VALUES ('delete', old.rowid, old.content);
                INSERT INTO memories_fts(rowid, content) VALUES (new.rowid, new.content);
            END;
            CREATE INDEX IF NOT EXISTS idx_memories_namespace ON memories(namespace);
            CREATE INDEX IF NOT EXISTS idx_memories_namespace_score
                ON memories(namespace, relevance_score DESC, id);
            CREATE INDEX IF NOT EXISTS idx_memories_created_at ON memories(created_at, id);
            CREATE INDEX IF NOT EXISTS idx_memories_last_accessed ON memories(last_accessed_at, id);
            CREATE INDEX IF NOT EXISTS idx_memories_type ON memories(memory_type, id);
            CREATE INDEX IF NOT EXISTS idx_memories_vector_retry
                ON memories(vector_index_status, vector_next_retry_at, id);
            INSERT INTO memories_fts(memories_fts) VALUES ('rebuild');",
        )?;
        transaction.pragma_update(None, "user_version", SCHEMA_VERSION)?;
        transaction.commit()?;
        Ok(())
    }

    fn migrate_schema(conn: &Connection) -> Result<(), MemoryError> {
        for (column_name, alter_sql) in [
            ("vector_index_status", "ALTER TABLE memories ADD COLUMN vector_index_status TEXT NOT NULL DEFAULT 'not_configured' CHECK(vector_index_status IN ('not_configured', 'pending', 'indexed', 'retry'))"),
            ("vector_retry_count", "ALTER TABLE memories ADD COLUMN vector_retry_count INTEGER NOT NULL DEFAULT 0 CHECK(vector_retry_count >= 0)"),
            ("vector_last_attempt_at", "ALTER TABLE memories ADD COLUMN vector_last_attempt_at INTEGER"),
            ("vector_next_retry_at", "ALTER TABLE memories ADD COLUMN vector_next_retry_at INTEGER"),
            ("vector_last_error", "ALTER TABLE memories ADD COLUMN vector_last_error TEXT"),
        ] {
            Self::add_column_if_missing(conn, column_name, alter_sql)?;
        }
        Ok(())
    }

    fn add_column_if_missing(
        conn: &Connection,
        column_name: &str,
        alter_sql: &str,
    ) -> Result<(), MemoryError> {
        let mut statement = conn.prepare("PRAGMA table_info(memories)")?;
        let columns = statement.query_map([], |row| row.get::<_, String>(1))?;
        for column in columns {
            if column? == column_name {
                return Ok(());
            }
        }
        drop(statement);
        conn.execute(alter_sql, [])?;
        Ok(())
    }
}

fn sanitize_vector_error(error: &str) -> String {
    let first_line = error.lines().next().unwrap_or("vector indexing failed");
    first_line.chars().take(512).collect()
}

fn fts_query_from_user_text(query: &str) -> Option<String> {
    let tokens = query
        .split_whitespace()
        .map(|token| token.trim_matches(|character: char| !character.is_alphanumeric()))
        .filter(|token| !token.is_empty())
        .map(|token| format!("\"{}\"", token.replace('"', "\"\"")))
        .collect::<Vec<_>>();
    (!tokens.is_empty()).then(|| tokens.join(" OR "))
}

fn normalize_fts_rank(raw_rank: f64) -> f64 {
    if !raw_rank.is_finite() {
        return 0.0;
    }
    let positive_rank = if raw_rank < 0.0 {
        -raw_rank
    } else {
        1.0 / (1.0 + raw_rank)
    };
    positive_rank / (positive_rank + 1.0e-6)
}

fn validate_entry(entry: &MemoryEntry) -> Result<(), MemoryError> {
    let length = entry.content.chars().count();
    if length < 5 {
        return Err(MemoryError::ContentTooShort { length, min: 5 });
    }
    if length > 4_096 {
        return Err(MemoryError::ContentTooLong { length, max: 4_096 });
    }
    validate_namespace(&entry.namespace)?;
    if !entry.relevance_score.is_finite() || !(0.0..=1.0).contains(&entry.relevance_score) {
        return Err(MemoryError::Config(
            "relevance_score must be finite and between 0.0 and 1.0".to_string(),
        ));
    }
    i64::try_from(entry.access_count).map_err(store_external_error)?;
    Ok(())
}

fn validate_namespace(namespace: &str) -> Result<(), MemoryError> {
    if !is_valid_namespace(namespace) {
        return Err(MemoryError::Config(
            "namespace must use safe non-empty ASCII segments separated by '::' and contain at most 256 characters".to_string(),
        ));
    }
    Ok(())
}

fn memory_entry_from_row(row: &Row<'_>) -> rusqlite::Result<MemoryEntry> {
    let id = uuid_from_db(row.get(0)?, 0)?;
    let memory_type = memory_type_from_db(&row.get::<_, String>(3)?, 3)?;
    let relevance_score: f64 = row.get(4)?;
    if !relevance_score.is_finite() || !(0.0..=1.0).contains(&relevance_score) {
        return Err(invalid_stored_value(4, "invalid relevance_score"));
    }
    let source_request_id = row
        .get::<_, Option<String>>(8)?
        .map(|value| uuid_from_db(value, 8))
        .transpose()?;
    Ok(MemoryEntry {
        id,
        namespace: row.get(1)?,
        content: row.get(2)?,
        memory_type,
        relevance_score,
        created_at: timestamp_from_db(row.get(5)?, 5)?,
        last_accessed_at: timestamp_from_db(row.get(6)?, 6)?,
        access_count: checked_db_u64(row.get(7)?, 7)?,
        source_request_id,
    })
}

fn memory_type_as_str(memory_type: MemoryType) -> &'static str {
    match memory_type {
        MemoryType::Preference => "preference",
        MemoryType::Fact => "fact",
        MemoryType::Context => "context",
        MemoryType::Decision => "decision",
    }
}

fn memory_type_from_db(value: &str, column: usize) -> rusqlite::Result<MemoryType> {
    match value {
        "preference" => Ok(MemoryType::Preference),
        "fact" => Ok(MemoryType::Fact),
        "context" => Ok(MemoryType::Context),
        "decision" => Ok(MemoryType::Decision),
        _ => Err(invalid_stored_value(column, "invalid memory_type")),
    }
}

fn uuid_from_db(value: String, column: usize) -> rusqlite::Result<Uuid> {
    Uuid::parse_str(&value).map_err(|error| {
        rusqlite::Error::FromSqlConversionFailure(
            column,
            rusqlite::types::Type::Text,
            Box::new(error),
        )
    })
}

fn timestamp_from_db(value: i64, column: usize) -> rusqlite::Result<DateTime<Utc>> {
    DateTime::<Utc>::from_timestamp(value, 0)
        .ok_or_else(|| invalid_stored_value(column, "timestamp is outside the supported range"))
}

fn optional_timestamp_from_db(
    value: Option<i64>,
    column: usize,
) -> rusqlite::Result<Option<DateTime<Utc>>> {
    value
        .map(|value| timestamp_from_db(value, column))
        .transpose()
}

fn checked_db_u64(value: i64, column: usize) -> rusqlite::Result<u64> {
    u64::try_from(value).map_err(|_| invalid_stored_value(column, "negative unsigned value"))
}

fn invalid_stored_value(column: usize, message: &'static str) -> rusqlite::Error {
    rusqlite::Error::FromSqlConversionFailure(
        column,
        rusqlite::types::Type::Text,
        Box::new(InvalidStoredValue(message)),
    )
}

fn query_u64<P: rusqlite::Params>(
    conn: &Connection,
    sql: &str,
    params: P,
) -> Result<u64, MemoryError> {
    let value: i64 = conn.query_row(sql, params, |row| row.get(0))?;
    u64::try_from(value).map_err(store_external_error)
}

fn metadata_timestamp(conn: &Connection, key: &str) -> Result<Option<DateTime<Utc>>, MemoryError> {
    let value = conn
        .query_row(
            "SELECT value FROM memory_metadata WHERE key = ?1",
            [key],
            |row| row.get::<_, String>(0),
        )
        .optional()?;
    value
        .map(|value| {
            let seconds = value.parse::<i64>().map_err(store_external_error)?;
            DateTime::<Utc>::from_timestamp(seconds, 0).ok_or_else(|| {
                store_external_error(InvalidStoredValue("invalid metadata timestamp"))
            })
        })
        .transpose()
}

#[derive(Debug)]
struct ConnectionPoisoned(&'static str);

impl fmt::Display for ConnectionPoisoned {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter.write_str(self.0)
    }
}

impl std::error::Error for ConnectionPoisoned {}

#[derive(Debug)]
struct InvalidStoredValue(&'static str);

impl fmt::Display for InvalidStoredValue {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter.write_str(self.0)
    }
}

impl std::error::Error for InvalidStoredValue {}

#[derive(Debug)]
struct NumericOverflow;

impl fmt::Display for NumericOverflow {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter.write_str("numeric overflow")
    }
}

impl std::error::Error for NumericOverflow {}

fn store_external_error(error: impl std::error::Error + Send + Sync + 'static) -> MemoryError {
    rusqlite::Error::ToSqlConversionFailure(Box::new(error)).into()
}

#[cfg(test)]
mod tests {
    use std::collections::BTreeSet;

    use tempfile::tempdir;

    use super::*;

    fn in_memory_store() -> MemoryStore {
        MemoryStore::new(Path::new(":memory:")).expect("in-memory memory store should initialize")
    }

    fn at(seconds: i64) -> DateTime<Utc> {
        DateTime::<Utc>::from_timestamp(seconds, 0).unwrap()
    }

    fn entry(id: u128, namespace: &str, content: &str) -> MemoryEntry {
        MemoryEntry {
            id: Uuid::from_u128(id),
            namespace: namespace.to_string(),
            content: content.to_string(),
            memory_type: MemoryType::Fact,
            relevance_score: 0.5,
            created_at: at(1_700_000_000 + id as i64),
            last_accessed_at: at(1_700_000_100 + id as i64),
            access_count: id as u64,
            source_request_id: Some(Uuid::from_u128(id + 100)),
        }
    }

    fn schema_objects(store: &MemoryStore, object_type: &str) -> BTreeSet<String> {
        let conn = store.connection().unwrap();
        let mut statement = conn
            .prepare("SELECT name FROM sqlite_master WHERE type = ?1")
            .unwrap();
        statement
            .query_map([object_type], |row| row.get(0))
            .unwrap()
            .collect::<Result<_, _>>()
            .unwrap()
    }

    #[test]
    fn creates_schema_and_database_file() {
        let temp = tempdir().unwrap();
        let path = temp.path().join("nested/memory.db");
        let store = MemoryStore::new(&path).unwrap();
        assert!(path.is_file());
        assert!(schema_objects(&store, "table").contains("memories_fts"));
        assert!(schema_objects(&store, "table").contains("memory_metadata"));
        assert_eq!(
            store
                .connection()
                .unwrap()
                .pragma_query_value(None, "user_version", |row| row.get::<_, i64>(0))
                .unwrap(),
            SCHEMA_VERSION
        );
    }

    #[test]
    fn complete_crud_round_trip() {
        let store = in_memory_store();
        let expected = entry(1, "user::one", "remember this exact fact");
        assert_eq!(store.store_entry(expected.clone(), None).unwrap(), expected);
        assert_eq!(
            store.get_entry_by_id(expected.id).unwrap(),
            Some(expected.clone())
        );
        assert_eq!(store.namespace_count("user::one").unwrap(), 1);
        assert!(store.delete_entry(expected.id).unwrap());
        assert!(!store.delete_entry(expected.id).unwrap());
        assert_eq!(store.get_entry_by_id(expected.id).unwrap(), None);

        let created = store
            .store_entry(
                NewMemoryEntry {
                    namespace: "user::one".to_string(),
                    content: "created from input".to_string(),
                    memory_type: MemoryType::Decision,
                    source_request_id: None,
                },
                None,
            )
            .unwrap();
        assert_eq!(created.relevance_score, 1.0);
        assert_eq!(created.access_count, 0);
    }

    #[test]
    fn validates_unicode_boundaries_and_numeric_fields() {
        let store = in_memory_store();
        store
            .store_entry(entry(1, "unicode", "界界界界界"), None)
            .unwrap();
        store
            .store_entry(entry(2, &"a".repeat(256), &"界".repeat(4_096)), None)
            .unwrap();
        assert!(matches!(
            store.store_entry(entry(3, "界", "界界界界"), None),
            Err(MemoryError::ContentTooShort { length: 4, .. })
        ));
        assert!(matches!(
            store.store_entry(entry(4, "界", &"界".repeat(4_097)), None),
            Err(MemoryError::ContentTooLong { length: 4_097, .. })
        ));
        assert!(store
            .store_entry(entry(5, "", "valid content"), None)
            .is_err());
        assert!(store
            .store_entry(entry(6, &"界".repeat(257), "valid content"), None)
            .is_err());
        for relevance in [f64::NAN, f64::INFINITY, -0.1, 1.1] {
            let mut invalid = entry(7, "user::one", "valid content");
            invalid.relevance_score = relevance;
            assert!(store.store_entry(invalid, None).is_err());
        }
    }

    #[test]
    fn preserves_exact_namespace_isolation_and_delimiters() {
        let store = in_memory_store();
        for (id, namespace) in [
            (1, "user::alpha"),
            (2, "user::alpha-other"),
            (3, "user::alpha::project::x"),
        ] {
            store
                .store_entry(entry(id, namespace, "same valid content"), None)
                .unwrap();
        }
        assert_eq!(store.namespace_count("user::alpha").unwrap(), 1);
        assert_eq!(store.namespace_count("user::alpha-other").unwrap(), 1);
        assert_eq!(store.delete_namespace("user::alpha").unwrap(), 1);
        assert_eq!(store.namespace_count("user::alpha-other").unwrap(), 1);
        assert_eq!(store.namespace_count("user::alpha::project::x").unwrap(), 1);
    }

    #[test]
    fn paginates_with_total_and_caps_limit() {
        let store = in_memory_store();
        for id in 1..=5 {
            store
                .store_entry(entry(id, "paged", &format!("content number {id}")), None)
                .unwrap();
        }
        let page = store.list_entries("paged", 2, 1).unwrap();
        assert_eq!(page.total_count, 5);
        assert_eq!(page.entries.len(), 2);
        assert_eq!(page.limit, 2);
        assert_eq!(page.offset, 1);
        assert_eq!(store.list_entries("paged", 1_000, 0).unwrap().limit, 200);
    }

    #[test]
    fn finds_trimmed_duplicate_only_in_exact_namespace() {
        let store = in_memory_store();
        let expected = entry(1, "exact", "  Unicode duplicate 界  ");
        store.store_entry(expected.clone(), None).unwrap();
        store
            .store_entry(entry(2, "other", "Unicode duplicate 界"), None)
            .unwrap();
        assert_eq!(
            store
                .find_duplicate("exact", "\nUnicode duplicate 界\t")
                .unwrap(),
            Some(expected)
        );
        assert_eq!(
            store
                .find_duplicate("missing", "Unicode duplicate 界")
                .unwrap(),
            None
        );
    }

    #[test]
    fn access_update_is_transactional_and_caps_boost() {
        let store = in_memory_store();
        let mut first = entry(1, "access", "first valid content");
        first.relevance_score = 0.95;
        let second = entry(2, "access", "second valid content");
        store.store_entry(first.clone(), None).unwrap();
        store.store_entry(second.clone(), None).unwrap();
        let accessed_at = at(1_800_000_000);
        assert_eq!(
            store
                .update_access_metadata(&[first.id, second.id, Uuid::from_u128(99)], accessed_at)
                .unwrap(),
            2
        );
        let first = store.get_entry_by_id(first.id).unwrap().unwrap();
        assert_eq!(first.access_count, 2);
        assert_eq!(first.last_accessed_at, accessed_at);
        assert_eq!(first.relevance_score, 1.0);
    }

    #[test]
    fn eviction_uses_all_deterministic_ties() {
        let store = in_memory_store();
        for id in [4, 3, 2, 1] {
            let mut tied = entry(id, "ties", &format!("tie content {id}"));
            tied.relevance_score = 0.5;
            tied.created_at = at(100);
            tied.last_accessed_at = at(100);
            store.store_entry(tied, Some(3)).unwrap();
        }
        assert_eq!(store.namespace_count("ties").unwrap(), 3);
        assert_eq!(store.get_entry_by_id(Uuid::from_u128(1)).unwrap(), None);
        assert!(store.get_entry_by_id(Uuid::from_u128(2)).unwrap().is_some());
    }

    #[test]
    fn decay_cycle_updates_all_types_evicts_deterministically_and_records_metadata() {
        let store = in_memory_store();
        let completed_at = at(1_800_000_000);
        let cases = [
            (1, "user::one", MemoryType::Preference, 0.8, 0.792),
            (2, "user::one", MemoryType::Fact, 0.7, 0.665),
            (3, "user::one", MemoryType::Context, 0.6, 0.51),
            (4, "user::two", MemoryType::Decision, 0.5, 0.49),
        ];
        for (id, namespace, memory_type, score, _) in cases {
            let mut memory = entry(id, namespace, "valid decay content");
            memory.memory_type = memory_type;
            memory.relevance_score = score;
            store.store_entry(memory, Some(10)).unwrap();
        }

        let result = store.run_decay_cycle(2, completed_at).unwrap();
        assert_eq!(result.completed_at, completed_at);
        assert_eq!(result.decayed_count, 4);
        assert_eq!(result.evicted_count, 1);
        assert_eq!(result.vector_retry_pending_count, 0);
        assert_eq!(
            result.namespace_evictions,
            vec![NamespaceEviction {
                namespace: "user::one".to_string(),
                evicted_count: 1,
                lowest_evicted_score: 0.51,
            }]
        );
        assert_eq!(store.get_entry_by_id(Uuid::from_u128(3)).unwrap(), None);
        for (id, _, _, _, expected) in cases {
            if id != 3 {
                let actual = store
                    .get_entry_by_id(Uuid::from_u128(id))
                    .unwrap()
                    .unwrap()
                    .relevance_score;
                assert!((actual - expected).abs() < 1e-12);
            }
        }
        assert_eq!(store.stats().unwrap().last_decay_cycle, Some(completed_at));
    }

    #[test]
    fn pending_vector_retry_listing_is_due_and_deterministic() {
        let store = in_memory_store();
        for id in [2_u128, 1] {
            store
                .store_entry(entry(id, "user::one", "valid retry content"), None)
                .unwrap();
        }
        let conn = store.connection().unwrap();
        conn.execute(
            "UPDATE memories SET vector_index_status = 'retry', vector_retry_count = 2,
vector_last_attempt_at = 90, vector_next_retry_at = 100, vector_last_error = 'offline'
WHERE id = ?1",
            [Uuid::from_u128(2).to_string()],
        )
        .unwrap();
        conn.execute(
            "UPDATE memories SET vector_index_status = 'pending', vector_next_retry_at = 200
WHERE id = ?1",
            [Uuid::from_u128(1).to_string()],
        )
        .unwrap();
        drop(conn);

        let retries = store.list_pending_vector_retries(at(150)).unwrap();
        assert_eq!(retries.len(), 1);
        assert_eq!(retries[0].entry.id, Uuid::from_u128(2));
        assert_eq!(retries[0].retry_count, 2);
        assert_eq!(retries[0].last_attempt_at, Some(at(90)));
        assert_eq!(retries[0].next_retry_at, Some(at(100)));
        assert_eq!(retries[0].last_error.as_deref(), Some("offline"));
    }

    #[test]
    fn stats_and_projects_include_decay_and_file_size() {
        let temp = tempdir().unwrap();
        let store = MemoryStore::new(&temp.path().join("memory.db")).unwrap();
        store
            .store_entry(
                entry(1, "user::a::project::hash", "project valid content"),
                None,
            )
            .unwrap();
        store
            .store_entry(
                entry(2, "user::a::agent::hash", "agent valid content"),
                None,
            )
            .unwrap();
        let decay = at(1_900_000_000);
        store.record_decay_cycle(decay).unwrap();
        let stats = store.stats().unwrap();
        assert_eq!(stats.total_count, 2);
        assert_eq!(stats.memories_per_namespace.len(), 2);
        assert_eq!(stats.average_relevance_score, 0.5);
        assert!(stats.storage_size_bytes.unwrap() > 0);
        assert_eq!(stats.last_decay_cycle, Some(decay));
        let namespaces = store.list_namespaces().unwrap();
        assert_eq!(namespaces.len(), 2);
        assert_eq!(namespaces[0].context_kind, "agent");
        assert_eq!(namespaces[1].context_kind, "project");
        assert_eq!(namespaces[1].entry_count, 1);
        assert_eq!(namespaces[1].last_activity, at(1_700_000_101));
    }

    #[test]
    fn invalid_stored_values_return_memory_errors() {
        let store = in_memory_store();
        let id = Uuid::from_u128(1);
        store
            .connection()
            .unwrap()
            .execute(
                "INSERT INTO memories (
                    id, namespace, content, memory_type, relevance_score,
                    created_at, last_accessed_at, access_count, source_request_id
                 ) VALUES (?1, 'raw', 'valid raw content', 'fact', 0.5, 0, 0, 0, NULL)",
                [id.to_string()],
            )
            .unwrap();
        store
            .connection()
            .unwrap()
            .execute("PRAGMA ignore_check_constraints = ON", [])
            .unwrap();
        store
            .connection()
            .unwrap()
            .execute(
                "UPDATE memories SET memory_type = 'corrupt' WHERE id = ?1",
                [id.to_string()],
            )
            .unwrap();
        assert!(matches!(
            store.get_entry_by_id(id),
            Err(MemoryError::Store(_))
        ));

        store
            .connection()
            .unwrap()
            .execute(
                "UPDATE memories SET memory_type = 'fact', id = 'not-a-uuid'",
                [],
            )
            .unwrap();
        assert!(matches!(
            store.list_entries("raw", 50, 0),
            Err(MemoryError::Store(_))
        ));
    }

    #[test]
    fn retrieval_ranks_normalized_fts_relevance_descending() {
        let store = in_memory_store();
        let now = Utc::now();
        let mut exact = entry(1, "user::rank", "rust sqlite indexing");
        exact.relevance_score = 1.0;
        exact.last_accessed_at = now;
        let mut partial = entry(2, "user::rank", "rust general notes");
        partial.relevance_score = 1.0;
        partial.last_accessed_at = now;
        store.store_entry(exact.clone(), None).unwrap();
        store.store_entry(partial, None).unwrap();
        let results = store
            .retrieve("user::rank", None, "rust sqlite", Some(0.0))
            .unwrap();
        assert_eq!(results[0].entry.id, exact.id);
        assert!(results[0].final_score > results[1].final_score);
        assert!(results.iter().all(|result| result.final_score.is_finite()));
    }

    #[test]
    fn retrieval_uses_exact_dual_scope_isolation() {
        let store = in_memory_store();
        for (id, namespace) in [
            (1, "user::alpha"),
            (2, "user::alpha::project::one"),
            (3, "user::alpha-other"),
            (4, "user::alpha::project::one-extra"),
        ] {
            let mut saved = entry(id, namespace, &format!("isolated retrieval item {id}"));
            saved.relevance_score = 1.0;
            saved.last_accessed_at = Utc::now();
            store.store_entry(saved, None).unwrap();
        }
        let results = store
            .retrieve(
                "user::alpha",
                Some("user::alpha::project::one"),
                "isolated retrieval",
                Some(0.0),
            )
            .unwrap();
        assert_eq!(results.len(), 2);
        assert!(results.iter().all(|result| {
            result.entry.namespace == "user::alpha"
                || result.entry.namespace == "user::alpha::project::one"
        }));
    }

    #[test]
    fn retrieval_deduplicates_trimmed_content_in_favor_of_context() {
        let store = in_memory_store();
        let now = Utc::now();
        let mut user = entry(1, "user::dupe", " shared searchable memory ");
        user.relevance_score = 1.0;
        user.last_accessed_at = now;
        let mut context = entry(2, "user::dupe::project::ctx", "shared searchable memory\n");
        context.relevance_score = 0.5;
        context.last_accessed_at = now;
        store.store_entry(user, None).unwrap();
        store.store_entry(context.clone(), None).unwrap();
        let results = store
            .retrieve(
                "user::dupe",
                Some("user::dupe::project::ctx"),
                "shared searchable",
                Some(0.0),
            )
            .unwrap();
        assert_eq!(results.len(), 1);
        assert_eq!(results[0].entry.id, context.id);
    }

    #[test]
    fn retrieval_filters_threshold_and_caps_total_candidates() {
        let store = in_memory_store();
        for id in 1..=60 {
            let mut saved = entry(id, "user::cap", &format!("capped searchable memory {id}"));
            saved.relevance_score = 1.0;
            saved.last_accessed_at = Utc::now();
            store.store_entry(saved, None).unwrap();
        }
        assert!(store
            .retrieve("user::cap", None, "capped searchable", Some(2.0))
            .unwrap()
            .is_empty());
        assert_eq!(
            store
                .retrieve("user::cap", None, "capped searchable", Some(0.0))
                .unwrap()
                .len(),
            MAX_RETRIEVAL_CANDIDATES
        );
    }

    #[test]
    fn retrieval_quotes_malformed_queries_and_rejects_empty_tokens() {
        let store = in_memory_store();
        let mut saved = entry(1, "user::syntax", "operator syntax searchable");
        saved.relevance_score = 1.0;
        saved.last_accessed_at = Utc::now();
        store.store_entry(saved, None).unwrap();
        assert!(!store
            .retrieve(
                "user::syntax",
                None,
                "operator OR (syntax: searchable*",
                Some(0.0)
            )
            .unwrap()
            .is_empty());
        assert!(store
            .retrieve("user::syntax", None, "  !!! (( ))  ", None)
            .unwrap()
            .is_empty());
    }

    #[test]
    fn retrieval_recency_boost_prefers_recent_access() {
        let store = in_memory_store();
        let now = Utc::now();
        let mut old = entry(1, "user::recent", "identical recency terms old");
        old.relevance_score = 1.0;
        old.last_accessed_at = now - chrono::Duration::days(30);
        let mut recent = entry(2, "user::recent", "identical recency terms recent");
        recent.relevance_score = 1.0;
        recent.last_accessed_at = now;
        store.store_entry(old, None).unwrap();
        store.store_entry(recent.clone(), None).unwrap();
        let results = store
            .retrieve("user::recent", None, "identical recency terms", Some(0.0))
            .unwrap();
        assert_eq!(results[0].entry.id, recent.id);
        assert!(results[0].final_score > results[1].final_score);
    }

    #[test]
    fn fts_triggers_follow_crud() {
        let store = in_memory_store();
        let saved = store
            .store_entry(entry(1, "fts", "alpha searchable phrase"), None)
            .unwrap();
        let count = |query: &str| {
            store
                .connection()
                .unwrap()
                .query_row(
                    "SELECT count(*) FROM memories_fts WHERE memories_fts MATCH ?1",
                    [query],
                    |row| row.get::<_, i64>(0),
                )
                .unwrap()
        };
        assert_eq!(count("searchable"), 1);
        store.delete_entry(saved.id).unwrap();
        assert_eq!(count("searchable"), 0);
    }

    #[test]
    fn migrates_legacy_schema_without_losing_fts_content() {
        let mut conn = Connection::open_in_memory().unwrap();
        conn.execute_batch(
            "CREATE TABLE memories (
                id TEXT PRIMARY KEY, namespace TEXT NOT NULL, content TEXT NOT NULL,
                memory_type TEXT NOT NULL, relevance_score REAL NOT NULL,
                created_at INTEGER NOT NULL, last_accessed_at INTEGER NOT NULL,
                access_count INTEGER NOT NULL, source_request_id TEXT
            );
            INSERT INTO memories VALUES (
                '00000000-0000-0000-0000-000000000001', 'user::default',
                'legacy searchable content', 'fact', 1.0, 1700000000, 1700000000, 0, NULL
            );",
        )
        .unwrap();
        MemoryStore::verify_fts5(&conn).unwrap();
        MemoryStore::initialize_schema(&mut conn).unwrap();
        let store = MemoryStore {
            conn: Arc::new(Mutex::new(conn)),
            db_path: None,
            metrics: Arc::new(MemoryMetrics::new()),
        };
        assert!(schema_objects(&store, "table").contains("memory_metadata"));
        assert_eq!(
            store
                .connection()
                .unwrap()
                .query_row(
                    "SELECT count(*) FROM memories_fts WHERE memories_fts MATCH 'legacy'",
                    [],
                    |row| row.get::<_, i64>(0)
                )
                .unwrap(),
            1
        );
    }
}
