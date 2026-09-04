//! Budget-aware retrieval and request injection for persistent memories.

use std::collections::HashMap;
use std::sync::Arc;

use chrono::Utc;
use serde_json::Value;
use uuid::Uuid;

use crate::compression::token_counter::TokenCounter;
use crate::models::openai::{Message, OpenAIRequest};

use super::metrics::MemoryMetrics;
use super::vector::MemoryVectorTier;
use super::{
    InjectionResult, InjectionStrategy, MemoryEntry, MemoryError, MemoryStore, MemoryType,
    ResolvedNamespace, ScoredMemory,
};

const RECALLED_MEMORIES_HEADER: &str = "[Recalled memories]";
const RECALLED_MEMORIES_FOOTER: &str = "[End memories]";
const DEFAULT_OUTPUT_RESERVE_DIVISOR: u32 = 4;

/// Retrieves, selects, and injects memories into OpenAI-compatible requests.
///
/// The store is shared across request handlers. Token counting is deterministic
/// and uses the same model-aware tokenizer as request compression.
pub struct MemoryInjector {
    store: Arc<MemoryStore>,
    token_counter: TokenCounter,
    metrics: Arc<MemoryMetrics>,
}

impl MemoryInjector {
    pub fn new(store: Arc<MemoryStore>, token_counter: TokenCounter) -> Self {
        Self::with_metrics(store, token_counter, Arc::new(MemoryMetrics::new()))
    }

    pub(crate) fn with_metrics(
        store: Arc<MemoryStore>,
        token_counter: TokenCounter,
        metrics: Arc<MemoryMetrics>,
    ) -> Self {
        Self {
            store,
            token_counter,
            metrics,
        }
    }

    pub fn store(&self) -> &Arc<MemoryStore> {
        &self.store
    }

    pub fn retrieve_lexical(
        &self,
        namespace: &ResolvedNamespace,
        query: &str,
        minimum_relevance_threshold: Option<f64>,
    ) -> Result<Vec<ScoredMemory>, MemoryError> {
        self.store.retrieve(
            &namespace.user_scope,
            namespace.context_scope.as_deref(),
            query,
            minimum_relevance_threshold,
        )
    }

    /// Retrieve relevant memories and inject the highest-scoring set that fits.
    ///
    /// `post_truncation_tokens` must describe the request immediately before
    /// memory injection. The output reserve is the smaller of a configured
    /// output limit and 25% of the context window; when no output limit is
    /// configured, 25% is reserved. Access metadata is updated transactionally
    /// only after selection and successful request mutation.
    #[allow(clippy::too_many_arguments)]
    pub fn inject(
        &self,
        request: &mut OpenAIRequest,
        namespace: &ResolvedNamespace,
        query: &str,
        strategy: InjectionStrategy,
        model_context_window: u32,
        post_truncation_tokens: u32,
        max_injection_tokens: u32,
        minimum_relevance_threshold: Option<f64>,
    ) -> Result<InjectionResult, MemoryError> {
        let budget = available_budget(
            request,
            model_context_window,
            post_truncation_tokens,
            max_injection_tokens,
        );
        if budget == 0 {
            return Ok(InjectionResult::default());
        }

        let candidates = self.store.retrieve(
            &namespace.user_scope,
            namespace.context_scope.as_deref(),
            query,
            minimum_relevance_threshold,
        )?;
        let selected = self.select_within_budget(&request.model, candidates, budget);
        let Some(formatted) = format_memories(&selected) else {
            return Ok(InjectionResult::default());
        };
        let injection_tokens = self.token_counter.count_text(&request.model, &formatted);
        debug_assert!(injection_tokens <= budget);

        inject_formatted(request, strategy, formatted);
        let selected_ids = selected
            .iter()
            .map(|memory| memory.entry.id)
            .collect::<Vec<_>>();
        self.store
            .update_access_metadata(&selected_ids, Utc::now())?;
        self.metrics
            .record_injection_tokens(u64::from(injection_tokens));

        Ok(InjectionResult {
            memories_injected: selected.len().min(u32::MAX as usize) as u32,
            injection_tokens,
            ..InjectionResult::default()
        })
    }

    pub fn inject_candidates(
        &self,
        request: &mut OpenAIRequest,
        candidates: Vec<ScoredMemory>,
        strategy: InjectionStrategy,
        model_context_window: u32,
        post_truncation_tokens: u32,
        max_injection_tokens: u32,
    ) -> Result<InjectionResult, MemoryError> {
        let budget = available_budget(
            request,
            model_context_window,
            post_truncation_tokens,
            max_injection_tokens,
        );
        if budget == 0 {
            return Ok(InjectionResult::default());
        }
        let selected = self.select_within_budget(&request.model, candidates, budget);
        let Some(formatted) = format_memories(&selected) else {
            return Ok(InjectionResult::default());
        };
        let injection_tokens = self.token_counter.count_text(&request.model, &formatted);
        inject_formatted(request, strategy, formatted);
        let selected_ids = selected
            .iter()
            .map(|memory| memory.entry.id)
            .collect::<Vec<_>>();
        self.store
            .update_access_metadata(&selected_ids, Utc::now())?;
        self.metrics
            .record_injection_tokens(u64::from(injection_tokens));
        Ok(InjectionResult {
            memories_injected: selected.len().min(u32::MAX as usize) as u32,
            injection_tokens,
            ..InjectionResult::default()
        })
    }

    pub(crate) fn select_within_budget(
        &self,
        model: &str,
        mut candidates: Vec<ScoredMemory>,
        budget: u32,
    ) -> Vec<ScoredMemory> {
        candidates.sort_by(|left, right| right.final_score.total_cmp(&left.final_score));
        let mut selected = Vec::new();

        for mut candidate in candidates {
            let line = format_memory_line(&candidate);
            candidate.estimated_tokens = self.token_counter.count_text(model, &line);

            selected.push(candidate);
            let formatted_tokens = format_memories(&selected)
                .map(|block| self.token_counter.count_text(model, &block))
                .unwrap_or(0);
            if formatted_tokens > budget {
                selected.pop();
            }
        }

        selected
    }
}

pub fn merge_retrieval_scores(
    lexical: Vec<ScoredMemory>,
    vector_entries: Vec<(MemoryEntry, f32)>,
    fts_weight: f32,
    vector_weight: f32,
) -> Vec<ScoredMemory> {
    use std::collections::HashMap;

    let weight_sum = f64::from(fts_weight + vector_weight);
    if !weight_sum.is_finite() || weight_sum <= 0.0 {
        return lexical;
    }
    let normalized_fts_weight = f64::from(fts_weight) / weight_sum;
    let normalized_vector_weight = f64::from(vector_weight) / weight_sum;
    let lexical_max = lexical
        .iter()
        .map(|candidate| candidate.final_score)
        .filter(|score| score.is_finite() && *score > 0.0)
        .fold(0.0_f64, f64::max);
    let vector_max = vector_entries
        .iter()
        .map(|(_, score)| f64::from(*score).max(0.0))
        .filter(|score| score.is_finite())
        .fold(0.0_f64, f64::max);
    let mut merged: HashMap<Uuid, (MemoryEntry, f64, f64)> = HashMap::new();
    for candidate in lexical {
        let score = if lexical_max > 0.0 {
            (candidate.final_score / lexical_max).clamp(0.0, 1.0)
        } else {
            0.0
        };
        merged.insert(candidate.entry.id, (candidate.entry, score, 0.0));
    }
    for (entry, raw_score) in vector_entries {
        let score = if vector_max > 0.0 {
            (f64::from(raw_score).max(0.0) / vector_max).clamp(0.0, 1.0)
        } else {
            0.0
        };
        merged
            .entry(entry.id)
            .and_modify(|existing| existing.2 = score)
            .or_insert((entry, 0.0, score));
    }
    let mut result = merged
        .into_values()
        .map(|(entry, lexical_score, vector_score)| ScoredMemory {
            entry,
            final_score: lexical_score * normalized_fts_weight
                + vector_score * normalized_vector_weight,
            estimated_tokens: 0,
        })
        .collect::<Vec<_>>();
    result.sort_by(|left, right| right.final_score.total_cmp(&left.final_score));
    result.truncate(50);
    result
}

#[allow(clippy::too_many_arguments)]
pub async fn retrieve_with_vector_fallback(
    store: &MemoryStore,
    tier: &dyn MemoryVectorTier,
    namespace: &ResolvedNamespace,
    query: &str,
    lexical: Vec<ScoredMemory>,
    fts_weight: f32,
    vector_weight: f32,
    timeout: std::time::Duration,
) -> Vec<ScoredMemory> {
    let matches = match tokio::time::timeout(timeout, tier.search(query, 50)).await {
        Ok(Ok(matches)) => matches,
        Ok(Err(error)) => {
            tracing::warn!(error = %error, "memory vector retrieval failed; using FTS5 results");
            return lexical;
        }
        Err(_) => {
            tracing::warn!(
                timeout_ms = timeout.as_millis() as u64,
                "memory vector retrieval timed out; using FTS5 results"
            );
            return lexical;
        }
    };
    let scores = matches
        .into_iter()
        .filter(|matched| matched.score.is_finite())
        .map(|matched| (matched.id, matched.score))
        .collect::<HashMap<_, _>>();
    let ids = scores.keys().copied().collect::<Vec<_>>();
    let entries = match store.get_entries_by_ids(&ids) {
        Ok(entries) => entries,
        Err(error) => {
            tracing::warn!(error = %error, "memory vector candidates could not be loaded; using FTS5 results");
            return lexical;
        }
    };
    let vector_entries = entries
        .into_iter()
        .filter(|entry| {
            entry.namespace == namespace.user_scope
                || namespace.context_scope.as_deref() == Some(entry.namespace.as_str())
        })
        .filter_map(|entry| scores.get(&entry.id).map(|score| (entry, *score)))
        .collect();
    merge_retrieval_scores(lexical, vector_entries, fts_weight, vector_weight)
}

/// Calculate the safe memory budget for a post-truncation request.
///
/// All arithmetic saturates, so an overcommitted context returns zero rather
/// than wrapping. The result never exceeds `max_injection_tokens`.
pub fn available_budget(
    request: &OpenAIRequest,
    model_context_window: u32,
    post_truncation_tokens: u32,
    max_injection_tokens: u32,
) -> u32 {
    if max_injection_tokens == 0 {
        return 0;
    }

    let default_reserve = model_context_window / DEFAULT_OUTPUT_RESERVE_DIVISOR;
    let output_reserve = configured_output_limit(request)
        .map(|configured| configured.min(default_reserve))
        .unwrap_or(default_reserve);
    model_context_window
        .saturating_sub(post_truncation_tokens)
        .saturating_sub(output_reserve)
        .min(max_injection_tokens)
}

/// Format memories using the gateway-authored injection envelope.
///
/// Empty input produces `None`; non-empty input uses exactly one tagged line
/// per memory and no trailing newline after the footer.
pub fn format_memories(memories: &[ScoredMemory]) -> Option<String> {
    if memories.is_empty() {
        return None;
    }

    let mut block = String::from(RECALLED_MEMORIES_HEADER);
    for memory in memories {
        block.push('\n');
        block.push_str(&format_memory_line(memory));
    }
    block.push('\n');
    block.push_str(RECALLED_MEMORIES_FOOTER);
    Some(block)
}

/// Remove only one complete gateway-authored, single-memory representation.
///
/// Accepted input is either one exact `- [type] content` line or a three-line
/// wrapper containing exactly that one line between the exact header/footer.
/// Multi-memory wrappers, partial wrappers, surrounding text, unknown tags,
/// and delimiter text appearing inside memory content are returned unchanged.
pub fn strip_formatting(content: &str) -> String {
    if let Some(stripped) = strip_tagged_line(content) {
        return stripped.to_owned();
    }

    let mut lines = content.split('\n');
    let Some(header) = lines.next() else {
        return content.to_owned();
    };
    let Some(memory_line) = lines.next() else {
        return content.to_owned();
    };
    let Some(footer) = lines.next() else {
        return content.to_owned();
    };
    if lines.next().is_none()
        && header == RECALLED_MEMORIES_HEADER
        && footer == RECALLED_MEMORIES_FOOTER
    {
        if let Some(stripped) = strip_tagged_line(memory_line) {
            return stripped.to_owned();
        }
    }

    content.to_owned()
}

fn configured_output_limit(request: &OpenAIRequest) -> Option<u32> {
    [
        request.max_tokens,
        json_u32(request.extra.get("max_output_tokens")),
        json_u32(request.extra.get("max_completion_tokens")),
    ]
    .into_iter()
    .flatten()
    .min()
}

fn json_u32(value: Option<&Value>) -> Option<u32> {
    value
        .and_then(Value::as_u64)
        .and_then(|value| u32::try_from(value).ok())
}

fn format_memory_line(memory: &ScoredMemory) -> String {
    format!(
        "- [{}] {}",
        memory_type_tag(memory.entry.memory_type),
        memory.entry.content
    )
}

fn memory_type_tag(memory_type: MemoryType) -> &'static str {
    match memory_type {
        MemoryType::Preference => "preference",
        MemoryType::Fact => "fact",
        MemoryType::Context => "context",
        MemoryType::Decision => "decision",
    }
}

fn strip_tagged_line(content: &str) -> Option<&str> {
    ["preference", "fact", "context", "decision"]
        .into_iter()
        .find_map(|tag| content.strip_prefix(&format!("- [{tag}] ")))
}

fn inject_formatted(request: &mut OpenAIRequest, strategy: InjectionStrategy, formatted: String) {
    match strategy {
        InjectionStrategy::SystemPromptPrefix => inject_system_prompt_prefix(request, formatted),
        InjectionStrategy::SyntheticMessage => insert_before_first_user(request, formatted),
    }
}

fn inject_system_prompt_prefix(request: &mut OpenAIRequest, formatted: String) {
    if let Some(system_index) = request
        .messages
        .iter()
        .position(|message| message.role == "system")
    {
        if let Value::String(existing) = &mut request.messages[system_index].content {
            if existing.is_empty() {
                *existing = formatted;
            } else {
                *existing = format!("{formatted}\n\n{existing}");
            }
            return;
        }

        request
            .messages
            .insert(system_index, system_message(formatted));
        return;
    }

    request.messages.insert(0, system_message(formatted));
}

fn insert_before_first_user(request: &mut OpenAIRequest, formatted: String) {
    let insertion_index = request
        .messages
        .iter()
        .position(|message| message.role == "user")
        .unwrap_or(request.messages.len());
    request
        .messages
        .insert(insertion_index, system_message(formatted));
}

fn system_message(content: String) -> Message {
    Message {
        role: "system".to_owned(),
        content: Value::String(content),
        extra: Default::default(),
    }
}

#[cfg(test)]
mod tests {
    use std::path::Path;

    use chrono::{TimeZone, Utc};
    use serde_json::{json, Map};
    use uuid::Uuid;

    use super::*;
    use crate::memory::{MemoryEntry, NewMemoryEntry};

    fn request(messages: Vec<Message>) -> OpenAIRequest {
        OpenAIRequest {
            model: "gpt-4o-mini".to_owned(),
            messages,
            stream: false,
            temperature: None,
            max_tokens: None,
            extra: Map::new(),
        }
    }

    fn message(role: &str, content: Value) -> Message {
        Message {
            role: role.to_owned(),
            content,
            extra: Map::new(),
        }
    }

    fn scored(id: u128, memory_type: MemoryType, content: &str, score: f64) -> ScoredMemory {
        let timestamp = Utc
            .with_ymd_and_hms(2026, 7, 31, 12, 0, 0)
            .single()
            .expect("fixed timestamp must be valid");
        ScoredMemory {
            entry: MemoryEntry {
                id: Uuid::from_u128(id),
                namespace: "user::default".to_owned(),
                content: content.to_owned(),
                memory_type,
                relevance_score: 0.5,
                created_at: timestamp,
                last_accessed_at: timestamp,
                access_count: 0,
                source_request_id: None,
            },
            final_score: score,
            estimated_tokens: 0,
        }
    }

    fn injector() -> MemoryInjector {
        MemoryInjector::new(
            Arc::new(MemoryStore::new(Path::new(":memory:")).expect("store must open")),
            TokenCounter::new(),
        )
    }

    #[test]
    fn budget_underflow_saturates_to_zero() {
        let request = request(Vec::new());

        assert_eq!(available_budget(&request, 100, 101, 500), 0);
        assert_eq!(available_budget(&request, 100, 90, 500), 0);
    }

    #[test]
    fn configured_output_uses_smaller_reserve_and_max_injection_cap() {
        let mut request = request(Vec::new());
        request.max_tokens = Some(10);

        assert_eq!(available_budget(&request, 100, 20, 500), 70);
        assert_eq!(available_budget(&request, 100, 20, 12), 12);
        request.max_tokens = Some(80);
        assert_eq!(available_budget(&request, 100, 20, 500), 55);
    }

    #[test]
    fn greedy_selection_skips_oversized_and_continues() {
        let injector = injector();
        let oversized = scored(1, MemoryType::Fact, &"oversized ".repeat(200), 3.0);
        let fitting = scored(2, MemoryType::Decision, "Use stable Rust.", 2.0);
        let fitting_tokens = injector.token_counter.count_text(
            "gpt-4o-mini",
            &format_memories(std::slice::from_ref(&fitting)).unwrap(),
        );

        let selected = injector.select_within_budget(
            "gpt-4o-mini",
            vec![fitting.clone(), oversized],
            fitting_tokens,
        );

        assert_eq!(selected.len(), 1);
        assert_eq!(selected[0].entry.id, fitting.entry.id);
        assert!(selected[0].estimated_tokens > 0);
    }

    #[test]
    fn formatting_preserves_delimiter_text_inside_content() {
        let memory = scored(
            1,
            MemoryType::Context,
            "Keep [End memories] and [Recalled memories] literally.",
            1.0,
        );
        let formatted = format_memories(std::slice::from_ref(&memory)).unwrap();

        assert_eq!(
            formatted,
            "[Recalled memories]\n- [context] Keep [End memories] and [Recalled memories] literally.\n[End memories]"
        );
        assert_eq!(strip_formatting(&formatted), memory.entry.content);
        assert_eq!(
            strip_formatting("prefix [Recalled memories]\ntext\n[End memories] suffix"),
            "prefix [Recalled memories]\ntext\n[End memories] suffix"
        );
        assert_eq!(
            strip_formatting("[Recalled memories]\n- [fact] one\n- [decision] two\n[End memories]"),
            "[Recalled memories]\n- [fact] one\n- [decision] two\n[End memories]"
        );
    }

    #[test]
    fn system_prefix_prepends_strings_and_preserves_array_content() {
        let block = format_memories(&[scored(1, MemoryType::Fact, "Remember this.", 1.0)]).unwrap();
        let mut string_request = request(vec![message("system", json!("Existing."))]);
        inject_formatted(
            &mut string_request,
            InjectionStrategy::SystemPromptPrefix,
            block.clone(),
        );
        assert_eq!(
            string_request.messages[0].content,
            Value::String(format!("{block}\n\nExisting."))
        );

        let array = json!([{"type": "text", "text": "Existing."}]);
        let mut array_request = request(vec![
            message("system", array.clone()),
            message("user", json!("Question")),
        ]);
        inject_formatted(
            &mut array_request,
            InjectionStrategy::SystemPromptPrefix,
            block.clone(),
        );
        assert_eq!(array_request.messages.len(), 3);
        assert_eq!(array_request.messages[0].content, Value::String(block));
        assert_eq!(array_request.messages[1].content, array);
    }

    #[test]
    fn synthetic_message_is_inserted_before_first_user() {
        let block = format_memories(&[scored(1, MemoryType::Fact, "Remember this.", 1.0)]).unwrap();
        let mut request = request(vec![
            message("system", json!("Existing.")),
            message("assistant", json!("Earlier.")),
            message("user", json!("Question")),
        ]);

        inject_formatted(
            &mut request,
            InjectionStrategy::SyntheticMessage,
            block.clone(),
        );

        assert_eq!(request.messages[2].role, "system");
        assert_eq!(request.messages[2].content, Value::String(block));
        assert_eq!(request.messages[3].role, "user");
    }

    #[test]
    fn zero_budget_is_a_no_op_without_access_update() {
        let store = Arc::new(MemoryStore::new(Path::new(":memory:")).expect("store must open"));
        let entry = store
            .store_entry(
                NewMemoryEntry {
                    namespace: "user::default".to_owned(),
                    content: "Remember alpha project conventions.".to_owned(),
                    memory_type: MemoryType::Fact,
                    source_request_id: None,
                },
                None,
            )
            .unwrap();
        let injector = MemoryInjector::new(store.clone(), TokenCounter::new());
        let mut request = request(vec![message("user", json!("alpha project"))]);
        let original = serde_json::to_value(&request).unwrap();

        let result = injector
            .inject(
                &mut request,
                &ResolvedNamespace {
                    user_scope: "user::default".to_owned(),
                    context_scope: None,
                },
                "alpha project",
                InjectionStrategy::SystemPromptPrefix,
                128_000,
                100,
                0,
                Some(0.0),
            )
            .unwrap();

        assert_eq!(result, InjectionResult::default());
        assert_eq!(serde_json::to_value(&request).unwrap(), original);
        let unchanged = store.get_entry_by_id(entry.id).unwrap().unwrap();
        assert_eq!(unchanged.access_count, 0);
        assert_eq!(unchanged.relevance_score, 1.0);
    }

    #[test]
    fn successful_injection_updates_access_count_and_boosts_relevance() {
        let store = Arc::new(MemoryStore::new(Path::new(":memory:")).expect("store must open"));
        let mut entry = scored(
            10,
            MemoryType::Decision,
            "Use alpha project stable Rust conventions.",
            1.0,
        )
        .entry;
        entry.relevance_score = 0.5;
        let stored = store.store_entry(entry, None).unwrap();
        let injector = MemoryInjector::new(store.clone(), TokenCounter::new());
        let mut request = request(vec![message("user", json!("alpha project Rust"))]);

        let result = injector
            .inject(
                &mut request,
                &ResolvedNamespace {
                    user_scope: "user::default".to_owned(),
                    context_scope: None,
                },
                "alpha project Rust",
                InjectionStrategy::SyntheticMessage,
                128_000,
                100,
                500,
                Some(0.0),
            )
            .unwrap();

        assert_eq!(result.memories_injected, 1);
        assert!(result.injection_tokens <= 500);
        let updated = store.get_entry_by_id(stored.id).unwrap().unwrap();
        assert_eq!(updated.access_count, 1);
        assert_eq!(updated.relevance_score, 0.6);
        assert!(updated.last_accessed_at >= stored.last_accessed_at);
    }
}
