use ai_gateway::loop_detection::{
    eviction::insert_bounded, RequestRecord, ResponseDescriptor, SessionState,
};
use dashmap::DashMap;
use proptest::prelude::*;
use std::time::{Duration, Instant};

fn record(sequence: u32, timestamp: Instant) -> RequestRecord {
    RequestRecord {
        content_simhash: u64::from(sequence),
        tool_call_fingerprint: Some(u64::from(sequence)),
        context_token_count: sequence,
        new_information_tokens: 1,
        token_count: sequence,
        cost: f64::from(sequence),
        has_tool_calls: true,
        tool_names: vec![],
        timestamp,
    }
}

proptest! {
    #![proptest_config(ProptestConfig::with_cases(128))]

    // Feature: agent-loop-detection, Property 10: History Depth Bounding
    #[test]
    fn prop_history_is_bounded(history_depth in 1usize..=50, request_count in 0u32..=500) {
        let mut session = SessionState::new(None, history_depth);
        let start = Instant::now();
        for sequence in 0..request_count {
            session.record_request(&record(sequence, start + Duration::from_millis(sequence.into())));
            session.record_response(ResponseDescriptor {
                token_count: sequence,
                block_type_hash: u64::from(sequence),
                is_error: false,
            });
        }
        let expected = request_count.min(history_depth as u32) as usize;
        prop_assert_eq!(session.request_hashes.len(), expected);
        prop_assert_eq!(session.tool_fingerprints.len(), expected);
        prop_assert_eq!(session.response_descriptors.len(), expected);
        prop_assert_eq!(session.timestamps.len(), expected);
        if expected > 0 {
            prop_assert_eq!(*session.request_hashes.back().unwrap(), u64::from(request_count - 1));
        }
    }

    // Feature: agent-loop-detection, Property 11: Max Sessions Capacity Invariant
    #[test]
    fn prop_capacity_is_never_exceeded(max_sessions in 1usize..=100, creations in 1usize..=500) {
        let sessions = DashMap::new();
        for sequence in 0..creations {
            let mut state = SessionState::new(None, 5);
            state.last_active = Instant::now() + Duration::from_millis(sequence as u64);
            insert_bounded(&sessions, format!("session-{sequence}"), state, max_sessions, None);
            prop_assert!(sessions.len() <= max_sessions);
        }
    }
}

#[test]
fn capacity_evicts_least_recently_active() {
    let sessions = DashMap::new();
    let now = Instant::now();
    let mut old = SessionState::new(None, 5);
    old.last_active = now - Duration::from_secs(10);
    let mut new = SessionState::new(None, 5);
    new.last_active = now;
    insert_bounded(&sessions, "old".into(), old, 2, None);
    insert_bounded(&sessions, "new".into(), new, 2, None);
    let evicted = insert_bounded(&sessions, "latest".into(), SessionState::new(None, 5), 2, None);
    assert_eq!(evicted.as_deref(), Some("old"));
    assert!(!sessions.contains_key("old"));
}
