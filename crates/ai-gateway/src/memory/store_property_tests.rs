use chrono::{DateTime, Utc};
use proptest::prelude::*;
use uuid::Uuid;

use super::{MemoryEntry, MemoryError, MemoryStore, MemoryType};

fn at(seconds: i64) -> DateTime<Utc> {
    DateTime::<Utc>::from_timestamp(seconds, 0).unwrap()
}

fn entry(id: u128, namespace: String, content: String) -> MemoryEntry {
    MemoryEntry {
        id: Uuid::from_u128(id),
        namespace,
        content,
        memory_type: MemoryType::Fact,
        relevance_score: 0.75,
        created_at: at(1_700_000_000),
        last_accessed_at: at(1_700_000_100),
        access_count: 7,
        source_request_id: Some(Uuid::from_u128(id.saturating_add(1))),
    }
}

fn safe_segment() -> impl Strategy<Value = String> {
    "[A-Za-z0-9_-]{1,39}".prop_map(|value| value)
}

fn valid_content() -> impl Strategy<Value = String> {
    prop::collection::vec(
        prop_oneof![Just('a'), Just('界'), Just(' '), Just('z')],
        5..256,
    )
    .prop_map(|chars| chars.into_iter().collect())
}

proptest! {
    #![proptest_config(ProptestConfig::with_cases(100))]

    #[test]
    fn property_store_round_trip(
        id in any::<u128>(),
        segment in safe_segment(),
        content in valid_content(),
    ) {
        let store = MemoryStore::new(std::path::Path::new(":memory:")).unwrap();
        let expected = entry(id, format!("user::{segment}"), content);
        let stored = store.store_entry(expected.clone(), None).unwrap();
        prop_assert_eq!(stored, expected.clone());
        prop_assert_eq!(store.get_entry_by_id(expected.id).unwrap(), Some(expected));
    }

    #[test]
    fn property_content_boundaries(length in 1usize..4200) {
        let store = MemoryStore::new(std::path::Path::new(":memory:")).unwrap();
        let candidate = entry(1, "user::bounds".to_string(), "界".repeat(length));
        let result = store.store_entry(candidate, None);
        if (5..=4096).contains(&length) {
            prop_assert!(result.is_ok());
        } else if length < 5 {
            prop_assert!(result.is_err());
            prop_assert!(matches!(result.unwrap_err(), MemoryError::ContentTooShort { .. }), "short content must be rejected");
        } else {
            prop_assert!(result.is_err());
            prop_assert!(matches!(result.unwrap_err(), MemoryError::ContentTooLong { .. }), "long content must be rejected");
        }
    }

    #[test]
    fn property_exact_namespace_isolation(
        left in safe_segment(),
        right in safe_segment().prop_filter("distinct namespaces", |value| !value.is_empty()),
    ) {
        let right = if right == left { format!("{right}-other") } else { right };
        let store = MemoryStore::new(std::path::Path::new(":memory:")).unwrap();
        let left_namespace = format!("user::{left}");
        let right_namespace = format!("user::{right}");
        store.store_entry(entry(1, left_namespace.clone(), "shared searchable memory".to_string()), None).unwrap();
        let results = store.retrieve(&right_namespace, None, "shared searchable memory", Some(0.0)).unwrap();
        prop_assert!(results.iter().all(|result| result.entry.namespace == right_namespace));
    }
}

#[test]
fn property_retrieval_result_cap() {
    let store = MemoryStore::new(std::path::Path::new(":memory:")).unwrap();
    for id in 1..=100u128 {
        store
            .store_entry(
                entry(
                    id,
                    "user::cap".to_string(),
                    format!("common searchable memory number {id}"),
                ),
                Some(200),
            )
            .unwrap();
    }
    let results = store
        .retrieve("user::cap", None, "common searchable memory", Some(0.0))
        .unwrap();
    assert!(results.len() <= 50);
}
