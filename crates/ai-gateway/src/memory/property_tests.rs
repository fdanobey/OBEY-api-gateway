use std::path::Path;
use std::sync::Arc;

use chrono::{Duration, TimeZone, Utc};
use proptest::prelude::*;
use tokio::runtime::Builder;
use uuid::Uuid;

use super::*;
use crate::compression::protection::ProtectionScanner;
use crate::compression::token_counter::TokenCounter;
use crate::memory::extractor::{
    compression_candidates_with_scanner, explicit_candidates, CompressionExtractionInput,
    CompressionMessageSnapshot, CompressionRemovalReport,
};
use crate::memory::injector::{format_memories, strip_formatting, MemoryInjector};
use crate::memory::scoring::{apply_decay, compute_score};
use crate::memory::store::CONTEXT_SCOPE_BOOST;
use crate::memory::{MemoryEntry, MemoryStore, ScoredMemory, SensitiveMatchSource};

fn timestamp() -> chrono::DateTime<Utc> {
    Utc.with_ymd_and_hms(2026, 7, 31, 12, 0, 0)
        .single()
        .unwrap()
}

fn scored(id: u128, memory_type: MemoryType, content: String, score: f64) -> ScoredMemory {
    ScoredMemory {
        entry: MemoryEntry {
            id: Uuid::from_u128(id),
            namespace: "user::property".to_owned(),
            content,
            memory_type,
            relevance_score: 0.75,
            created_at: timestamp(),
            last_accessed_at: timestamp(),
            access_count: 0,
            source_request_id: None,
        },
        final_score: score,
        estimated_tokens: 0,
    }
}

fn memory_type(index: usize) -> MemoryType {
    [
        MemoryType::Preference,
        MemoryType::Fact,
        MemoryType::Context,
        MemoryType::Decision,
    ][index]
}

fn sensitive_case(kind: usize, suffix: &str) -> String {
    match kind {
        0 => format!("sk-{suffix}"),
        1 => format!("AKIA{}", &suffix.to_ascii_uppercase()[..16]),
        2 => format!("Bearer {suffix}"),
        _ => format!("https://user:{suffix}@example.invalid/path"),
    }
}

proptest! {
    #![proptest_config(ProptestConfig::with_cases(64))]

    #[test]
    fn property_trigger_phrase_classification(
        trigger_index in 0usize..8,
        suffix in "[A-Za-z0-9 ]{5,80}",
        uppercase in any::<bool>(),
    ) {
        let (trigger, expected) = [
            ("remember this", MemoryType::Fact),
            ("I prefer", MemoryType::Preference),
            ("save this", MemoryType::Context),
            ("note that", MemoryType::Fact),
            ("keep in mind", MemoryType::Fact),
            ("always use", MemoryType::Preference),
            ("never use", MemoryType::Preference),
            ("my preference is", MemoryType::Preference),
        ][trigger_index];
        let trigger = if uppercase { trigger.to_uppercase() } else { trigger.to_owned() };
        let candidates = explicit_candidates(&format!("{trigger} {suffix}."));
        prop_assert_eq!(candidates.len(), 1);
        prop_assert_eq!(candidates[0].memory_type, expected);
        prop_assert!(candidates[0].content.to_ascii_lowercase().starts_with(&trigger.to_ascii_lowercase()));
    }

    #[test]
    fn property_sensitive_builtins_reject(
        kind in 0usize..4,
        suffix in "[A-Za-z0-9]{24}",
    ) {
        let content = sensitive_case(kind, &suffix);
        let scanner = SensitiveContentScanner::new(&[], None).unwrap();
        let runtime = Builder::new_current_thread().enable_all().build().unwrap();
        let result = runtime.block_on(scanner.scan(&content)).unwrap();
        prop_assert!(result.contains_sensitive);
        prop_assert!(result.match_count >= 1);
        prop_assert!(result.sources.contains(&SensitiveMatchSource::SecretPattern));
    }

    #[test]
    fn property_greedy_budget_never_overflows_and_skips_oversized(
        short_words in 1usize..20,
        long_words in 80usize..240,
        budget_padding in 0u32..20,
    ) {
        let injector = MemoryInjector::new(
            Arc::new(MemoryStore::new(Path::new(":memory:")).unwrap()),
            TokenCounter::new(),
        );
        let fitting = scored(1, MemoryType::Decision, "fit ".repeat(short_words), 2.0);
        let oversized = scored(2, MemoryType::Fact, "oversized ".repeat(long_words), 3.0);
        let counter = TokenCounter::new();
        let fitting_cost = counter.count_text(
            "gpt-4o-mini",
            &format_memories(std::slice::from_ref(&fitting)).unwrap(),
        );
        let budget = fitting_cost.saturating_add(budget_padding);
        let selected = injector.select_within_budget(
            "gpt-4o-mini",
            vec![fitting.clone(), oversized],
            budget,
        );
        let actual = format_memories(&selected)
            .map(|block| counter.count_text("gpt-4o-mini", &block))
            .unwrap_or(0);
        prop_assert!(actual <= budget);
        prop_assert!(selected.iter().any(|candidate| candidate.entry.id == fitting.entry.id));
    }

    #[test]
    fn property_decay_and_scoring_formula(
        rank in 0.0f64..10_000.0,
        relevance in 0.0f64..=1.0,
        elapsed_hours in 0i64..24_000,
        type_index in 0usize..4,
    ) {
        let now = timestamp();
        let last_accessed = now - Duration::hours(elapsed_hours);
        let user_score = compute_score(rank, relevance, last_accessed, now, false);
        let context_score = compute_score(rank, relevance, last_accessed, now, true);
        let expected_recency = 1.0 / (1.0 + elapsed_hours as f64 / 24.0 * 0.1);
        let expected = rank * relevance * expected_recency;
        prop_assert!((user_score - expected).abs() <= expected.abs().max(1.0) * 1e-12);
        prop_assert!((context_score - user_score * CONTEXT_SCOPE_BOOST).abs() <= user_score.abs().max(1.0) * 1e-12);
        let multiplier = [0.99, 0.95, 0.85, 0.98][type_index];
        prop_assert_eq!(apply_decay(relevance, memory_type(type_index)), relevance * multiplier);
    }

    #[test]
    fn property_format_round_trip_with_delimiter_content(
        type_index in 0usize..4,
        prefix in "[A-Za-z0-9 ]{0,40}",
        suffix in "[A-Za-z0-9 ]{0,40}",
        delimiter_index in 0usize..2,
    ) {
        let delimiter = ["[Recalled memories]", "[End memories]"][delimiter_index];
        let content = format!("{prefix}{delimiter}{suffix}");
        let memory = scored(1, memory_type(type_index), content.clone(), 1.0);
        let formatted = format_memories(&[memory]).unwrap();
        prop_assert_eq!(strip_formatting(&formatted), content);
    }

    #[test]
    fn property_scope_boost_is_exact(
        rank in 0.0f64..10_000.0,
        relevance in 0.0f64..=1.0,
        elapsed_days in 0i64..1_000,
    ) {
        let now = timestamp();
        let last_accessed = now - Duration::days(elapsed_days);
        let user = compute_score(rank, relevance, last_accessed, now, false);
        let context = compute_score(rank, relevance, last_accessed, now, true);
        prop_assert!((context - user * 1.5).abs() <= user.abs().max(1.0) * 1e-12);
    }

    #[test]
    fn property_namespace_eviction_maintains_cap(
        cap in 1usize..25,
        extra in 1usize..25,
    ) {
        let store = MemoryStore::new(Path::new(":memory:")).unwrap();
        for index in 0..cap + extra {
            let mut entry = scored(
                index as u128 + 1,
                MemoryType::Fact,
                format!("eviction property content {index}"),
                1.0,
            ).entry;
            entry.relevance_score = index as f64 / (cap + extra) as f64;
            store.store_entry(entry, Some(cap)).unwrap();
            prop_assert!(store.namespace_count("user::property").unwrap() <= cap as u64);
        }
        prop_assert_eq!(store.namespace_count("user::property").unwrap(), cap as u64);
    }
}

#[test]
fn compression_candidates_prioritize_and_cap_without_persistence() {
    let mut content = String::new();
    for index in 0..12 {
        content.push_str(&format!("```rust\nfn preserved_{index}() {{}}\n```\n"));
    }
    content.push_str("We decided to use SQLite. I prefer concise output. src/memory/store.rs");
    let before = [CompressionMessageSnapshot {
        message_id: "m1",
        content: &content,
        tokens: 100,
    }];
    let removals = [CompressionRemovalReport {
        message_id: "m1",
        tokens_before: 100,
        tokens_after: 0,
    }];
    let candidates = compression_candidates_with_scanner(
        CompressionExtractionInput {
            before: &before,
            after: &[],
            removals: &removals,
        },
        &ProtectionScanner::new().unwrap(),
    );
    assert_eq!(candidates.len(), 10);
    assert!(candidates
        .iter()
        .all(|candidate| candidate.memory_type == MemoryType::Context));
}
