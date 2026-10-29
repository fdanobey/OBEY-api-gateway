//! Opt-in background consolidation of near-duplicate and contradictory memories.
//!
//! The engine groups active entries within a namespace deterministically, asks
//! the extraction provider for strict-JSON merge/supersede/keep decisions, and
//! applies each merge transactionally. Grouping and parsing are pure functions;
//! provider failure or invalid decisions skip a batch before writes. A write
//! failure after prior decisions is reported with the actual committed counts.

use std::collections::HashSet;
use std::sync::Arc;
use std::time::Duration;

use serde::{Deserialize, Serialize};
use uuid::Uuid;

use super::extractor::{
    ExtractionMessage, ExtractionRole, MemoryExtractionProvider,
    MemoryExtractionProviderRequest,
};
use super::sensitive::{SensitiveContentScanner, SensitiveScanOptions};
use super::store::MemoryStore;
use super::vector::VECTOR_INDEX_TIMEOUT;
use super::{
    spawn_store_task, MemoryEntry, MemoryError, MemoryType, MemoryVectorTier,
    MEMORY_EXTRACTION_INTERNAL_TAG,
};

/// Maximum entries sent to the provider in one consolidation batch.
const MAX_ENTRIES_PER_PROVIDER_CALL: usize = 40;

/// Minimum normalized token-set similarity for two entries to group.
const SIMILARITY_THRESHOLD: f64 = 0.6;

/// Maximum entries examined per namespace in one pass.
const MAX_ENTRIES_PER_NAMESPACE: usize = 1_000;

/// Errors produced while parsing consolidation decisions.
#[derive(Debug, Clone, PartialEq, Eq, thiserror::Error)]
pub enum ConsolidationError {
    #[error("consolidation decisions are not valid JSON")]
    MalformedJson,
    #[error("consolidation decision {index} does not match the required schema")]
    SchemaMismatch { index: usize },
    #[error("consolidation decision {index} references an unknown entry id")]
    UnknownId { index: usize },
}

/// One provider decision about a candidate group.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum Decision {
    /// Fold every source into one new entry superseding them all.
    Merge {
        source_ids: Vec<Uuid>,
        merged_content: String,
    },
    /// Mark one entry as superseded by another active entry.
    Supersede {
        superseded_id: Uuid,
        superseded_by_id: Uuid,
    },
    /// Leave the entry untouched.
    Keep { id: Uuid },
}

/// Strictly-typed provider decision as returned in JSON.
#[derive(Debug, Deserialize)]
#[serde(tag = "action", rename_all = "snake_case")]
enum RawDecision {
    Merge {
        source_ids: Vec<String>,
        merged_content: String,
    },
    Supersede {
        superseded_id: String,
        superseded_by_id: String,
    },
    Keep { id: String },
}

/// Counts recorded for one consolidation pass.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq, Serialize, Deserialize)]
pub struct ConsolidationCounts {
    pub merged: u32,
    pub superseded: u32,
    pub kept: u32,
    pub skipped_batches: u32,
    pub provider_errors: u32,
}

/// Group active entries into provider-call batches.
///
/// Entries are bucketed by namespace and memory type, then chained into groups
/// with a fixed normalized token-set similarity rule. Groups holding two or
/// more entries are near-duplicate batches; singletons are returned as
/// one-entry "kept" groups so callers can account for them. Pure: no I/O and
/// no provider involvement.
pub fn group_candidates(entries: &[MemoryEntry]) -> Vec<Vec<MemoryEntry>> {
    let mut buckets: Vec<(String, MemoryType, Vec<usize>)> = Vec::new();
    for (position, entry) in entries.iter().enumerate() {
        let index = match buckets
            .iter()
            .position(|(namespace, memory_type, _)| {
                namespace == &entry.namespace && *memory_type == entry.memory_type
            }) {
            Some(index) => index,
            None => {
                buckets.push((entry.namespace.clone(), entry.memory_type, Vec::new()));
                buckets.len() - 1
            }
        };
        buckets[index].2.push(position);
    }
    buckets.sort_by(|left, right| {
        left.0.cmp(&right.0)
            .then_with(|| memory_type_rank(left.1).cmp(&memory_type_rank(right.1)))
    });

    let mut groups = Vec::new();
    for (_, _, positions) in buckets {
        let mut remaining: Vec<usize> = positions;
        // Preserve input order inside each bucket so grouping is deterministic.
        remaining.sort_unstable();
        while !remaining.is_empty() {
            let first = remaining[0];
            let mut group = vec![first];
            let mut rest = Vec::with_capacity(remaining.len() - 1);
            for &candidate in &remaining[1..] {
                if normalized_similarity(&entries[first].content, &entries[candidate].content)
                    >= SIMILARITY_THRESHOLD
                {
                    group.push(candidate);
                } else {
                    rest.push(candidate);
                }
            }
            remaining = rest;
            groups.push(
                group
                    .into_iter()
                    .map(|position| entries[position].clone())
                    .collect::<Vec<_>>(),
            );
        }
    }
    groups
}

fn memory_type_rank(memory_type: MemoryType) -> u8 {
    match memory_type {
        MemoryType::Preference => 0,
        MemoryType::Fact => 1,
        MemoryType::Context => 2,
        MemoryType::Decision => 3,
    }
}

/// Normalized token-set similarity on lowercased alphanumeric tokens.
fn normalized_similarity(left: &str, right: &str) -> f64 {
    let left_tokens = normalized_tokens(left);
    let right_tokens = normalized_tokens(right);
    if left_tokens.is_empty() || right_tokens.is_empty() {
        return 0.0;
    }
    let shared = left_tokens.intersection(&right_tokens).count();
    shared as f64 / (left_tokens.len() + right_tokens.len() - shared) as f64
}

fn normalized_tokens(content: &str) -> HashSet<String> {
    content
        .split(|character: char| !character.is_ascii_alphanumeric())
        .filter(|token| !token.is_empty())
        .map(|token| token.to_ascii_lowercase())
        .collect()
}

/// Parse provider output into decisions, accepting only strict JSON.
///
/// Every referenced id must appear in `valid_ids`; malformed JSON, unknown
/// schema shapes, and unknown ids are errors so the caller skips the batch.
pub fn parse_decisions(
    json: &str,
    valid_ids: &HashSet<Uuid>,
) -> Result<Vec<Decision>, ConsolidationError> {
    let raw = serde_json::from_str::<Vec<serde_json::Value>>(json)
        .map_err(|_| ConsolidationError::MalformedJson)?;
    let mut decisions = Vec::with_capacity(raw.len());
    for (index, value) in raw.iter().enumerate() {
        if !value.is_object() {
            return Err(ConsolidationError::SchemaMismatch { index });
        }
        let decision = match serde_json::from_value::<RawDecision>(value.clone()) {
            Ok(raw) => raw,
            Err(_) => return Err(ConsolidationError::SchemaMismatch { index }),
        };
        let decision = match decision {
            RawDecision::Merge {
                source_ids,
                merged_content,
            } => {
                let mut ids = Vec::with_capacity(source_ids.len());
                for id in &source_ids {
                    ids.push(
                        parse_valid_id(id, valid_ids)
                            .ok_or(ConsolidationError::UnknownId { index })?,
                    );
                }
                if ids.is_empty() || merged_content.trim().is_empty() {
                    return Err(ConsolidationError::SchemaMismatch { index });
                }
                Decision::Merge {
                    source_ids: ids,
                    merged_content,
                }
            }
            RawDecision::Supersede {
                superseded_id,
                superseded_by_id,
            } => Decision::Supersede {
                superseded_id: parse_valid_id(&superseded_id, valid_ids)
                    .ok_or(ConsolidationError::UnknownId { index })?,
                superseded_by_id: parse_valid_id(&superseded_by_id, valid_ids)
                    .ok_or(ConsolidationError::UnknownId { index })?,
            },
            RawDecision::Keep { id } => Decision::Keep {
                id: parse_valid_id(&id, valid_ids)
                    .ok_or(ConsolidationError::UnknownId { index })?,
            },
        };
        decisions.push(decision);
    }
    Ok(decisions)
}

fn parse_valid_id(raw: &str, valid_ids: &HashSet<Uuid>) -> Option<Uuid> {
    let id = Uuid::parse_str(raw).ok()?;
    valid_ids.contains(&id).then_some(id)
}

/// Inputs for one consolidation pass over every namespace.
#[derive(Clone)]
pub struct ConsolidationContext {
    pub store: Arc<MemoryStore>,
    pub provider: Arc<dyn MemoryExtractionProvider>,
    pub scanner: Arc<SensitiveContentScanner>,
    pub vector_tier: Option<Arc<dyn MemoryVectorTier>>,
    pub allow_sensitive_storage: bool,
    pub provider_name: String,
    pub model: String,
    pub maximum_concurrent_provider_calls: usize,
    pub provider_timeout: Duration,
}

/// Run one consolidation pass across all namespaces.
///
/// Each namespace is grouped deterministically and dispatched in bounded
/// batches; provider errors and parse failures skip their batch with sanitized
/// logging. Merged content passes the same sensitive scan and validation as
/// manual creation before any store write.
pub async fn run_consolidation(
    context: ConsolidationContext,
) -> Result<ConsolidationCounts, MemoryError> {
    if context.maximum_concurrent_provider_calls == 0 {
        return Err(MemoryError::Config(
            "maximum concurrent consolidation calls must be at least 1".to_owned(),
        ));
    }
    if context.provider_timeout.is_zero() {
        return Err(MemoryError::Config(
            "consolidation provider timeout must be greater than zero".to_owned(),
        ));
    }
    let namespaces = {
        let store = Arc::clone(&context.store);
        spawn_store_task(move || store.list_namespaces()).await?
    };

    let mut counts = ConsolidationCounts::default();
    for namespace in namespaces {
        let namespace_name = namespace.namespace.clone();
        let mut entries = Vec::new();
        while entries.len() < MAX_ENTRIES_PER_NAMESPACE {
            let offset = entries.len();
            let page = {
                let store = Arc::clone(&context.store);
                let namespace = namespace_name.clone();
                spawn_store_task(move || {
                    store.list_entries(&namespace, MAX_ENTRIES_PER_NAMESPACE - offset, offset)
                })
                .await?
            };
            let page_len = page.entries.len();
            entries.extend(page.entries);
            if page_len < page.limit || entries.len() >= page.total_count as usize {
                break;
            }
        }
        let groups = group_candidates(&entries);
        for mut group in groups {
            if group.len() < 2 {
                counts.kept += 1;
                continue;
            }
            // Hard cap per provider call; overflow entries are not sent and
            // stay untouched, so they are accounted as kept.
            if group.len() > MAX_ENTRIES_PER_PROVIDER_CALL {
            let overflow = group.len() - MAX_ENTRIES_PER_PROVIDER_CALL;
            counts.kept = saturating_count(counts.kept as usize + overflow);
            group.truncate(MAX_ENTRIES_PER_PROVIDER_CALL);
            }
            let result = process_batch(&context, &group).await;
            match result {
                BatchOutcome::Applied(applied) => {
                    counts.merged = saturating_count(counts.merged as usize + applied.merged);
                    counts.superseded =
                        saturating_count(counts.superseded as usize + applied.superseded);
                    counts.kept = saturating_count(counts.kept as usize + applied.kept);
                }
                BatchOutcome::ProviderError => counts.provider_errors += 1,
                BatchOutcome::BatchSkipped => counts.skipped_batches += 1,
            }
        }
    }
    Ok(counts)
}

enum BatchOutcome {
    Applied(AppliedCounts),
    ProviderError,
    BatchSkipped,
}

#[derive(Default)]
struct AppliedCounts {
    merged: usize,
    superseded: usize,
    kept: usize,
}

fn saturating_count(count: usize) -> u32 {
    count.min(u32::MAX as usize) as u32
}

/// Ask the provider for decisions on one batch and apply them.
///
/// The provider returns raw decision JSON through its candidate contents (the
/// shared [`MemoryExtractionProvider`] boundary is text-in/text-out). All
/// decisions are checked before writing; if a later write fails, committed
/// decisions retain their counts and the remaining decisions are abandoned.
async fn process_batch(context: &ConsolidationContext, group: &[MemoryEntry]) -> BatchOutcome {
    let valid_ids = group.iter().map(|entry| entry.id).collect::<HashSet<_>>();
    let request = MemoryExtractionProviderRequest {
        provider: context.provider_name.clone(),
        model: context.model.clone(),
        messages: vec![ExtractionMessage::injected(
            ExtractionRole::User,
            batch_prompt(group),
        )],
        internal_tag: MEMORY_EXTRACTION_INTERNAL_TAG,
        bypass_memory: true,
    };

    let provider_result = tokio::time::timeout(
        context.provider_timeout,
        context.provider.clone().extract(request),
    )
    .await;
    let raw_output = match provider_result {
        Err(_) => {
            tracing::warn!("memory consolidation provider timed out");
            return BatchOutcome::ProviderError;
        }
        Ok(Err(_error)) => {
            tracing::warn!("memory consolidation provider failed");
            return BatchOutcome::ProviderError;
        }
        Ok(Ok(candidates)) => candidates
            .into_iter()
            .map(|candidate| candidate.content)
            .collect::<Vec<_>>()
            .join(""),
    };

    let decisions = match parse_decisions(&raw_output, &valid_ids) {
        Ok(decisions) => decisions,
        Err(error) => {
            tracing::warn!(error = %error, batch_size = group.len(), "memory consolidation decisions rejected");
            return BatchOutcome::BatchSkipped;
        }
    };

    for decision in &decisions {
        if let Decision::Merge { source_ids, merged_content } = decision {
            if !validate_merge(context, group, source_ids, merged_content).await {
                return BatchOutcome::BatchSkipped;
            }
        }
    }

    let mut applied = AppliedCounts::default();
    for decision in decisions {
        if !apply_decision(context, group, decision, &mut applied).await {
            if applied.merged == 0 && applied.superseded == 0 && applied.kept == 0 {
                return BatchOutcome::BatchSkipped;
            }
            tracing::warn!(merged = applied.merged, superseded = applied.superseded, kept = applied.kept, "memory consolidation batch partially applied");
            break;
        }
    }
    BatchOutcome::Applied(applied)
}

async fn apply_decision(context: &ConsolidationContext, group: &[MemoryEntry], decision: Decision, applied: &mut AppliedCounts) -> bool {
    match decision {
        Decision::Merge { source_ids, merged_content } => {
            let Some(marked) = apply_merge(context, group, &source_ids, &merged_content).await else {
                return false;
            };
            applied.merged += 1;
            applied.superseded += marked;
        }
        Decision::Supersede { superseded_id, superseded_by_id } => {
            let marked = match spawn_store_task({
                let store = Arc::clone(&context.store);
                move || store.mark_superseded(&[superseded_id], superseded_by_id)
            }).await {
                Ok(marked) => marked,
                Err(_) => {
                    tracing::warn!("memory consolidation supersede failed");
                    return false;
                }
            };
            applied.superseded += marked;
            if marked != 0 { schedule_vector_delete(context, superseded_id).await; }
        }
        Decision::Keep { id } => {
            if group.iter().any(|entry| entry.id == id) {
                applied.kept += 1;
            }
        }
    }
    true
}

/// Validate a merge before any batch write.
async fn validate_merge(context: &ConsolidationContext, group: &[MemoryEntry], source_ids: &[Uuid], merged_content: &str) -> bool {
    let sources = source_ids.iter().filter_map(|id| group.iter().find(|entry| &entry.id == id)).collect::<Vec<_>>();
    if sources.len() != source_ids.len() || sources.is_empty() || source_ids.iter().collect::<HashSet<_>>().len() != source_ids.len() {
        tracing::warn!("memory consolidation merge has invalid sources");
        return false;
    }
    let namespace = &sources[0].namespace;
    let memory_type = sources[0].memory_type;
    if sources.iter().any(|entry| &entry.namespace != namespace || entry.memory_type != memory_type) {
        tracing::warn!("memory consolidation merge mixes namespaces or types");
        return false;
    }
    if !(5..=4096).contains(&merged_content.chars().count()) {
        tracing::warn!("memory consolidation merged content has invalid length");
        return false;
    }
    let scan = match context.scanner.scan_with_options(merged_content, SensitiveScanOptions {
        allow_sensitive_storage: context.allow_sensitive_storage,
        ..SensitiveScanOptions::default()
    }).await {
        Ok(scan) => scan,
        Err(_) => {
            tracing::warn!("memory consolidation sensitive scan failed");
            return false;
        }
    };
    if scan.contains_sensitive {
        tracing::warn!("memory consolidation merged content rejected as sensitive");
        return false;
    }
    true
}

/// Persist one prevalidated merge atomically and update vectors.
///
/// Returns the number of sources actually marked superseded. The store
/// helper's transaction requires every source UPDATE to affect exactly one
/// row (or the whole merge rolls back), so a successful commit marks
/// exactly `source_ids.len()` rows.
async fn apply_merge(context: &ConsolidationContext, group: &[MemoryEntry], source_ids: &[Uuid], merged_content: &str) -> Option<usize> {
    let first = group.iter().find(|entry| entry.id == source_ids[0])?;
    let namespace = first.namespace.clone();
    let memory_type = first.memory_type;
    let merged = {
        let store = Arc::clone(&context.store);
        let content = merged_content.to_owned();
        let source_ids = source_ids.to_vec();
        match spawn_store_task(move || store.create_merged_entry_with_supersession(&namespace, &content, memory_type, &source_ids)).await {
            Ok(merged) => merged,
            Err(_) => {
                tracing::warn!("memory consolidation merge write failed");
                return None;
            }
        }
    };
    schedule_vector_index(context, merged).await;
    for id in source_ids { schedule_vector_delete(context, *id).await; }
    Some(source_ids.len())
}

/// Best-effort vector index of a merged entry; sanitized logging only.
async fn schedule_vector_index(context: &ConsolidationContext, entry: MemoryEntry) {
    let Some(tier) = context.vector_tier.as_ref() else {
        return;
    };
    let id = entry.id;
    let result = tokio::time::timeout(VECTOR_INDEX_TIMEOUT, tier.index(&entry)).await;
    match result {
        Ok(Ok(())) => {}
        Ok(Err(error)) => {
            tracing::warn!(entry_id = %id, error = %error, "memory consolidation vector index failed")
        }
        Err(_) => tracing::warn!(entry_id = %id, "memory consolidation vector index timed out"),
    }
}

/// Best-effort vector delete of a superseded entry; sanitized logging only.
async fn schedule_vector_delete(context: &ConsolidationContext, id: Uuid) {
    let Some(tier) = context.vector_tier.as_ref() else {
        return;
    };
    let result = tokio::time::timeout(VECTOR_INDEX_TIMEOUT, tier.delete_entry(id)).await;
    match result {
        Ok(Ok(())) => {}
        Ok(Err(error)) => {
            tracing::warn!(entry_id = %id, error = %error, "memory consolidation vector delete failed")
        }
        Err(_) => tracing::warn!(entry_id = %id, "memory consolidation vector delete timed out"),
    }
}

/// Render one batch as compact JSON for the provider prompt.
///
/// Content is embedded because the provider must see it to merge; this is the
/// same data the extraction path already sends to the configured provider.
fn batch_prompt(group: &[MemoryEntry]) -> String {
    let entries = group
        .iter()
        .map(|entry| {
            serde_json::json!({
                "id": entry.id.to_string(),
                "content": entry.content,
            })
        })
        .collect::<Vec<_>>();
    serde_json::json!({
        "entries": entries,
        "instructions": "Decide for each entry group: merge near-duplicates into one entry, \
supersede contradictions, or keep. Respond ONLY as a JSON array of decisions: \
[{\"action\":\"merge\",\"source_ids\":[...],\"merged_content\":\"...\"}, \
{\"action\":\"supersede\",\"superseded_id\":\"...\",\"superseded_by_id\":\"...\"}, \
{\"action\":\"keep\",\"id\":\"...\"}]",
    })
    .to_string()
}

#[cfg(test)]
mod tests {
    use super::*;
    use async_trait::async_trait;
    use std::sync::atomic::{AtomicUsize, Ordering};

    use crate::memory::extractor::MemoryExtractionProviderError;
    use crate::memory::store::NewMemoryEntry;
    use crate::memory::{StructuredMemoryCandidate, VectorMatch};

    fn entry(id: u128, namespace: &str, memory_type: MemoryType, content: &str) -> MemoryEntry {
        let now = chrono::Utc::now();
        MemoryEntry {
            id: Uuid::from_u128(id),
            namespace: namespace.to_owned(),
            content: content.to_owned(),
            memory_type,
            relevance_score: 1.0,
            created_at: now,
            last_accessed_at: now,
            access_count: 0,
            source_request_id: None,
            updated_at: None,
            superseded_by: None,
        }
    }

    #[test]
    fn near_duplicates_in_same_namespace_and_type_group() {
        let entries = vec![
            entry(1, "user::a", MemoryType::Fact, "The deploy target is fly.io eu-west"),
            entry(
                2,
                "user::a",
                MemoryType::Fact,
                "The deploy target is fly.io eu-west region",
            ),
            entry(3, "user::a", MemoryType::Fact, "Rust is the primary language"),
        ];
        let groups = group_candidates(&entries);
        assert_eq!(groups.len(), 2);
        let batch = groups
            .iter()
            .find(|group| group.len() == 2)
            .expect("near-duplicates must form one batch");
        assert_eq!(batch[0].id, Uuid::from_u128(1));
        assert_eq!(batch[1].id, Uuid::from_u128(2));
    }

    #[test]
    fn different_type_or_namespace_never_group() {
        let entries = vec![
            entry(1, "user::a", MemoryType::Fact, "The deploy target is fly.io eu-west"),
            entry(
                2,
                "user::a",
                MemoryType::Preference,
                "The deploy target is fly.io eu-west",
            ),
            entry(3, "user::b", MemoryType::Fact, "The deploy target is fly.io eu-west"),
        ];
        let groups = group_candidates(&entries);
        assert_eq!(groups.len(), 3);
        assert!(groups.iter().all(|group| group.len() == 1));
    }

    #[test]
    fn grouping_is_deterministic_across_runs() {
        let entries = vec![
            entry(1, "user::a", MemoryType::Fact, "alpha beta gamma delta"),
            entry(2, "user::a", MemoryType::Fact, "alpha beta gamma epsilon"),
            entry(3, "user::a", MemoryType::Fact, "unrelated single entry here"),
        ];
        let first = group_candidates(&entries);
        let second = group_candidates(&entries);
        assert_eq!(first, second);
    }

    fn valid_ids(ids: &[u128]) -> HashSet<Uuid> {
        ids.iter().map(|&id| Uuid::from_u128(id)).collect()
    }

    #[test]
    fn parse_accepts_merge_supersede_and_keep() {
        let ids = valid_ids(&[1, 2, 3]);
        let json = format!(
            "[{{\"action\":\"merge\",\"source_ids\":[\"{}\",\"{}\"],\"merged_content\":\"combined\"}}, \
             {{\"action\":\"supersede\",\"superseded_id\":\"{}\",\"superseded_by_id\":\"{}\"}}, \
             {{\"action\":\"keep\",\"id\":\"{}\"}}]",
            Uuid::from_u128(1),
            Uuid::from_u128(2),
            Uuid::from_u128(3),
            Uuid::from_u128(1),
            Uuid::from_u128(2),
        );
        let decisions = parse_decisions(&json, &ids).unwrap();
        assert_eq!(
            decisions,
            vec![
                Decision::Merge {
                    source_ids: vec![Uuid::from_u128(1), Uuid::from_u128(2)],
                    merged_content: "combined".to_owned(),
                },
                Decision::Supersede {
                    superseded_id: Uuid::from_u128(3),
                    superseded_by_id: Uuid::from_u128(1),
                },
                Decision::Keep {
                    id: Uuid::from_u128(2),
                },
            ]
        );
    }

    #[test]
    fn parse_rejects_malformed_json_and_wrong_schema() {
        let ids = valid_ids(&[1]);
        assert_eq!(
            parse_decisions("not json", &ids).unwrap_err(),
            ConsolidationError::MalformedJson
        );
        let wrong_schema = format!("[{{\"action\":\"vanish\",\"id\":\"{}\"}}]", Uuid::from_u128(1));
        assert_eq!(
            parse_decisions(&wrong_schema, &ids).unwrap_err(),
            ConsolidationError::SchemaMismatch { index: 0 }
        );
    }

    #[test]
    fn parse_rejects_unknown_ids() {
        let ids = valid_ids(&[1]);
        let unknown = format!("[{{\"action\":\"keep\",\"id\":\"{}\"}}]", Uuid::from_u128(99));
        assert_eq!(
            parse_decisions(&unknown, &ids).unwrap_err(),
            ConsolidationError::UnknownId { index: 0 }
        );
        let unparseable = "[{\"action\":\"keep\",\"id\":\"zero\"}]";
        assert_eq!(
            parse_decisions(unparseable, &ids).unwrap_err(),
            ConsolidationError::UnknownId { index: 0 }
        );
    }

    #[test]
    fn parse_rejects_empty_merge_sources_and_blank_content() {
        let ids = valid_ids(&[1]);
        let no_sources = "[{\"action\":\"merge\",\"source_ids\":[],\"merged_content\":\"x\"}]";
        assert_eq!(
            parse_decisions(no_sources, &ids).unwrap_err(),
            ConsolidationError::SchemaMismatch { index: 0 }
        );
        let blank = format!(
            "[{{\"action\":\"merge\",\"source_ids\":[\"{}\"],\"merged_content\":\"   \"}}]",
            Uuid::from_u128(1)
        );
        assert_eq!(
            parse_decisions(&blank, &ids).unwrap_err(),
            ConsolidationError::SchemaMismatch { index: 0 }
        );
    }

    /// Fake provider returning fixed candidate contents; records call count
    /// and the largest message payload seen (mirrors extractor.rs fakes).
    struct FakeConsolidationProvider {
        contents: std::sync::Mutex<Vec<String>>,
        fail: bool,
        calls: AtomicUsize,
        max_payload_entries: AtomicUsize,
    }

    impl FakeConsolidationProvider {
        fn decisions(raw_json: String) -> Self {
            Self {
                contents: std::sync::Mutex::new(vec![raw_json]),
                fail: false,
                calls: AtomicUsize::new(0),
                max_payload_entries: AtomicUsize::new(0),
            }
        }

        fn failing() -> Self {
            Self {
                contents: std::sync::Mutex::new(Vec::new()),
                fail: true,
                calls: AtomicUsize::new(0),
                max_payload_entries: AtomicUsize::new(0),
            }
        }

        fn calls(&self) -> usize {
            self.calls.load(Ordering::SeqCst)
        }
    }

    #[async_trait]
    impl MemoryExtractionProvider for FakeConsolidationProvider {
        async fn extract(
            &self,
            request: MemoryExtractionProviderRequest,
        ) -> Result<Vec<StructuredMemoryCandidate>, MemoryExtractionProviderError> {
            assert_eq!(request.internal_tag, MEMORY_EXTRACTION_INTERNAL_TAG);
            assert!(request.bypass_memory);
            self.calls.fetch_add(1, Ordering::SeqCst);
            if self.fail {
                return Err(MemoryExtractionProviderError {
                    message: "mock consolidation provider unavailable".to_owned(),
                });
            }
            let payload: serde_json::Value =
                serde_json::from_str(&request.messages[0].content).unwrap();
            let size = payload["entries"].as_array().unwrap().len();
            self.max_payload_entries
                .fetch_max(size, Ordering::SeqCst);
            Ok(self
                .contents
                .lock()
                .unwrap()
                .iter()
                .map(|content| StructuredMemoryCandidate {
                    content: content.clone(),
                    memory_type: MemoryType::Fact,
                })
                .collect())
        }
    }

    /// Recording vector tier observing index and delete operations.
    #[derive(Default)]
    struct RecordingTier {
        indexed: std::sync::Mutex<Vec<Uuid>>,
        deleted: std::sync::Mutex<Vec<Uuid>>,
    }

    #[async_trait]
    impl MemoryVectorTier for RecordingTier {
        async fn index(&self, entry: &MemoryEntry) -> Result<(), MemoryError> {
            self.indexed.lock().unwrap().push(entry.id);
            Ok(())
        }

        async fn search(&self, _query: &str, _limit: usize) -> Result<Vec<VectorMatch>, MemoryError> {
            Ok(Vec::new())
        }

        async fn delete_entry(&self, id: Uuid) -> Result<(), MemoryError> {
            self.deleted.lock().unwrap().push(id);
            Ok(())
        }
    }

    async fn test_context(
        provider: Arc<FakeConsolidationProvider>,
        tier: Option<Arc<RecordingTier>>,
    ) -> (ConsolidationContext, Arc<MemoryStore>) {
        let store = Arc::new(MemoryStore::new(std::path::Path::new(":memory:")).unwrap());
        let scanner = Arc::new(SensitiveContentScanner::new(&[], None).unwrap());
        let context = ConsolidationContext {
            store: store.clone(),
            provider,
            scanner,
            vector_tier: tier.map(|tier| tier as Arc<dyn MemoryVectorTier>),
            allow_sensitive_storage: false,
            provider_name: "internal-provider".to_owned(),
            model: "consolidator-model".to_owned(),
            maximum_concurrent_provider_calls: 1,
            provider_timeout: Duration::from_secs(2),
        };
        (context, store)
    }

    /// Seed two active near-duplicate entries directly into the store.
    async fn seed_near_duplicates(store: &MemoryStore) -> (Uuid, Uuid) {
        let first = store
            .store_entry(
                NewMemoryEntry {
                    namespace: "user::test".to_owned(),
                    content: "The deploy target is fly.io eu-west".to_owned(),
                    memory_type: MemoryType::Fact,
                    source_request_id: None,
                },
                None,
            )
            .unwrap();
        let second = store
            .store_entry(
                NewMemoryEntry {
                    namespace: "user::test".to_owned(),
                    content: "The deploy target is fly.io eu-west region".to_owned(),
                    memory_type: MemoryType::Fact,
                    source_request_id: None,
                },
                None,
            )
            .unwrap();
        (first.id, second.id)
    }

    fn merge_decisions_json(sources: &[Uuid]) -> String {
        let ids = sources
            .iter()
            .map(|id| format!("\"{id}\""))
            .collect::<Vec<_>>()
            .join(",");
        format!(
            "[{{\"action\":\"merge\",\"source_ids\":[{ids}],\"merged_content\":\"Deployments run on fly.io in eu-west.\"}}]"
        )
    }

    #[tokio::test]
    async fn happy_path_merges_duplicates_and_supersedes_sources() {
        let provider = Arc::new(FakeConsolidationProvider::decisions(String::new()));
        let tier = Arc::new(RecordingTier::default());
        let (mut context, store) = test_context(provider.clone(), Some(tier.clone())).await;
        let (first, second) = seed_near_duplicates(&store).await;

        // The provider sees the batch and answers with a merge decision.
        *provider.contents.lock().unwrap() = vec![merge_decisions_json(&[first, second])];
        let counts = run_consolidation(context.clone()).await.unwrap();

        assert_eq!(counts.merged, 1);
        assert_eq!(counts.superseded, 2);
        assert_eq!(counts.kept, 0);
        assert_eq!(counts.skipped_batches, 0);
        assert_eq!(counts.provider_errors, 0);

        let active = store.list_entries("user::test", 10, 0).unwrap();
        assert_eq!(active.entries.len(), 1);
        assert_eq!(active.entries[0].content, "Deployments run on fly.io in eu-west.");
        let archived = store
            .list_entries_with_superseded("user::test", 10, 0, true)
            .unwrap();
        assert_eq!(archived.entries.len(), 3);
        for source in [first, second] {
            let entry = store.get_entry_by_id(source).unwrap().unwrap();
            assert_eq!(entry.superseded_by, Some(active.entries[0].id));
        }

        // Vector tier observed the merged index plus both source deletes.
        let indexed = tier.indexed.lock().unwrap().clone();
        let deleted = tier.deleted.lock().unwrap().clone();
        assert_eq!(indexed, vec![active.entries[0].id]);
        assert!(deleted.contains(&first) && deleted.contains(&second));

        context.provider = Arc::new(FakeConsolidationProvider::failing());
    }

    #[tokio::test]
    async fn provider_error_skips_batch_and_continues_other_namespaces() {
        let provider = Arc::new(FakeConsolidationProvider::failing());
        let (context, store) = test_context(provider.clone(), None).await;
        let (first, _second) = seed_near_duplicates(&store).await;
        store
            .store_entry(
                NewMemoryEntry {
                    namespace: "user::other".to_owned(),
                    content: "Completely different project context entry".to_owned(),
                    memory_type: MemoryType::Fact,
                    source_request_id: None,
                },
                None,
            )
            .unwrap();

        let counts = run_consolidation(context).await.unwrap();

        assert_eq!(counts.provider_errors, 1);
        assert_eq!(counts.skipped_batches, 0);
        assert_eq!(counts.kept, 1);
        // Nothing changed in the failing namespace.
        assert!(store.get_entry_by_id(first).unwrap().is_some());
        assert_eq!(store.list_entries("user::test", 10, 0).unwrap().entries.len(), 2);
        assert_eq!(provider.calls(), 1);
    }

    #[tokio::test]
    async fn malformed_decisions_skip_the_batch() {
        let provider = Arc::new(FakeConsolidationProvider::decisions("not json".to_owned()));
        let (context, store) = test_context(provider, None).await;
        seed_near_duplicates(&store).await;

        let counts = run_consolidation(context).await.unwrap();

        assert_eq!(counts.skipped_batches, 1);
        assert_eq!(counts.merged, 0);
        assert_eq!(store.list_entries("user::test", 10, 0).unwrap().entries.len(), 2);
    }

    #[tokio::test]
    async fn sensitive_merged_content_is_skipped() {
        let provider = Arc::new(FakeConsolidationProvider::decisions(String::new()));
        let (mut context, store) = test_context(provider.clone(), None).await;
        let (first, second) = seed_near_duplicates(&store).await;
        *provider.contents.lock().unwrap() = vec![format!(
            "[{{\"action\":\"merge\",\"source_ids\":[\"{first}\",\"{second}\"],\"merged_content\":\"The api key is sk-abcdefghijklmnopqrst\"}}]"
        )];

        let counts = run_consolidation(context.clone()).await.unwrap();

        assert_eq!(counts.skipped_batches, 1);
        assert_eq!(counts.merged, 0);
        assert_eq!(store.list_entries("user::test", 10, 0).unwrap().entries.len(), 2);
        assert!(store.get_entry_by_id(first).unwrap().unwrap().superseded_by.is_none());

        // allow_sensitive_storage bypasses the scan and the merge applies.
        context.allow_sensitive_storage = true;
        let counts = run_consolidation(context).await.unwrap();
        assert_eq!(counts.merged, 1);
    }

#[tokio::test]
async fn batches_are_capped_at_forty_entries_per_provider_call() {
    let provider = Arc::new(FakeConsolidationProvider::decisions("[]".to_owned()));
    let (context, store) = test_context(provider.clone(), None).await;
    for index in 0..45 {
        store
            .store_entry(
                NewMemoryEntry {
                    namespace: "user::cap".to_owned(),
                    content: format!("Shared alpha beta gamma token number {index}"),
                    memory_type: MemoryType::Fact,
                    source_request_id: None,
                },
                None,
            )
            .unwrap();
    }

    let counts = run_consolidation(context).await.unwrap();

    assert_eq!(provider.calls(), 1);
    assert_eq!(
        provider.max_payload_entries.load(Ordering::SeqCst),
        MAX_ENTRIES_PER_PROVIDER_CALL
    );
    // One capped batch (provider answered []) plus the five overflow
    // entries fall outside the cap and are accounted as kept.
    assert_eq!(counts.kept, 5);
    assert_eq!(counts.skipped_batches, 0);
}

#[tokio::test]
async fn namespaces_beyond_the_list_limit_are_fully_paginated() {
    // MAX_LIST_LIMIT clamps every list_entries page to 200 rows; a
    // near-duplicate pair seeded as the two oldest entries lands on the
    // second page (listing is created_at DESC). Without pagination the pair
    // is never listed, never grouped, and the provider is never called.
    let provider = Arc::new(FakeConsolidationProvider::decisions(String::new()));
    let (context, store) = test_context(provider.clone(), None).await;

    let base = 1_700_000_000i64;
    let filler_count = crate::memory::store::MAX_LIST_LIMIT;
    for index in 0..filler_count {
        store
            .store_entry(
                MemoryEntry {
                    id: Uuid::from_u128(1_000 + index as u128),
                    namespace: "user::paged".to_owned(),
                    content: format!("omega{index} sigma{index} tau{index} rho{index}"),
                    memory_type: MemoryType::Fact,
                    relevance_score: 1.0,
                    created_at: chrono::DateTime::<chrono::Utc>::from_timestamp(
                        base + index as i64,
                        0,
                    )
                    .unwrap(),
                    last_accessed_at: chrono::DateTime::<chrono::Utc>::from_timestamp(
                        base + index as i64,
                        0,
                    )
                    .unwrap(),
                    access_count: 0,
                    source_request_id: None,
                    updated_at: None,
                    superseded_by: None,
                },
                None,
            )
            .unwrap();
    }
    let pair_base = base - 10;
    let mut pair_ids = Vec::new();
    for offset in 0..2 {
        let content = if offset == 0 {
            "The deploy target is fly.io eu-west".to_owned()
        } else {
            "The deploy target is fly.io eu-west region".to_owned()
        };
        let entry = store
            .store_entry(
                MemoryEntry {
                    id: Uuid::from_u128(2_000 + offset),
                    namespace: "user::paged".to_owned(),
                    content,
                    memory_type: MemoryType::Fact,
                    relevance_score: 1.0,
                    created_at: chrono::DateTime::<chrono::Utc>::from_timestamp(
                        pair_base - offset as i64,
                        0,
                    )
                    .unwrap(),
                    last_accessed_at: chrono::DateTime::<chrono::Utc>::from_timestamp(
                        pair_base - offset as i64,
                        0,
                    )
                    .unwrap(),
                    access_count: 0,
                    source_request_id: None,
                    updated_at: None,
                    superseded_by: None,
                },
                None,
            )
            .unwrap();
        pair_ids.push(entry.id);
    }

    *provider.contents.lock().unwrap() = vec![merge_decisions_json(&pair_ids)];
    let counts = run_consolidation(context).await.unwrap();

    assert_eq!(provider.calls(), 1, "the second-page pair must reach the provider");
    assert_eq!(counts.merged, 1);
    assert_eq!(counts.superseded, 2);
    // The dissimilar fillers stay kept (one group each); the pair was merged.
    assert_eq!(counts.kept as usize, filler_count);
    let active = store.list_entries("user::paged", 10, 0).unwrap();
    assert_eq!(active.total_count, 201);
}

#[tokio::test]
async fn batch_with_any_invalid_merge_is_skipped_before_writes() {
    // The whole decision set is validated before any write: a valid merge
    // followed by an invalid (sensitive) merge must skip the batch with
    // nothing persisted, not apply the first merge and then stop.
    let provider = Arc::new(FakeConsolidationProvider::decisions(String::new()));
    let (context, store) = test_context(provider.clone(), None).await;
    let (first, second) = seed_near_duplicates(&store).await;
    // A second near-duplicate pair in another namespace shares the batch?
    // No: batches never mix namespaces, so put both decisions in one batch
    // by making the single group's decision list contain two merges: one
    // valid for the real pair and one whose sources are the same pair but
    // with sensitive merged content.
    let ids = [first, second];
    let id_list = ids
        .iter()
        .map(|id| format!("\"{id}\""))
        .collect::<Vec<_>>()
        .join(",");
    *provider.contents.lock().unwrap() = vec![format!(
        "[{{\"action\":\"merge\",\"source_ids\":[{id_list}],\"merged_content\":\"Deployments run on fly.io in eu-west.\"}}, \
          {{\"action\":\"merge\",\"source_ids\":[{id_list}],\"merged_content\":\"The api key is sk-abcdefghijklmnopqrst\"}}]"
    )];

    let counts = run_consolidation(context).await.unwrap();

    assert_eq!(counts.skipped_batches, 1);
    assert_eq!(counts.merged, 0);
    assert_eq!(counts.superseded, 0);
    assert_eq!(store.list_entries("user::test", 10, 0).unwrap().entries.len(), 2);
    assert!(store.get_entry_by_id(first).unwrap().unwrap().superseded_by.is_none());
}
}
