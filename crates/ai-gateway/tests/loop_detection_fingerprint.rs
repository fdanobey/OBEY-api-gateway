use ai_gateway::loop_detection::fingerprint::{FingerprintTracker, ToolCall, ToolCallFingerprint};
use proptest::prelude::*;

proptest! {
    #![proptest_config(ProptestConfig::with_cases(64))]

    // Feature: agent-loop-detection, Property 9: Tool Call Fingerprint Scoring Monotonicity
    #[test]
    fn prop_identical_fingerprint_scoring(repetitions in 1usize..=20) {
        let mut tracker = FingerprintTracker::default();
        let fingerprint = Some(42);
        let mut scores = Vec::new();
        for _ in 0..repetitions {
            scores.push(tracker.observe(fingerprint));
        }

        let expected = match repetitions {
            1 => 0.0,
            2 => 0.4,
            3 => 0.7,
            _ => 1.0,
        };
        prop_assert_eq!(*scores.last().unwrap(), expected);
        prop_assert!(scores.windows(2).all(|window| window[0] <= window[1]));
    }
}

#[test]
fn argument_values_do_not_change_fingerprint() {
    let left = serde_json::json!([{
        "function": {"name": "read_file", "arguments": "{\"path\":\"a\",\"offset\":1}"}
    }]);
    let right = serde_json::json!([{
        "function": {"name": "read_file", "arguments": "{\"offset\":999,\"path\":\"b\"}"}
    }]);
    assert_eq!(
        ToolCallFingerprint::from_json(&left),
        ToolCallFingerprint::from_json(&right)
    );
}

#[test]
fn ordered_function_names_change_fingerprint() {
    let first = vec![
        ToolCall {
            function_name: "read".into(),
            argument_keys: vec!["path".into()],
        },
        ToolCall {
            function_name: "write".into(),
            argument_keys: vec!["path".into()],
        },
    ];
    let second = vec![first[1].clone(), first[0].clone()];
    assert_ne!(
        ToolCallFingerprint::compute(&first),
        ToolCallFingerprint::compute(&second)
    );
}

#[test]
fn no_tool_calls_do_not_update_tracker() {
    let mut tracker = FingerprintTracker::default();
    assert_eq!(tracker.observe(Some(7)), 0.0);
    assert_eq!(tracker.observe(None), 0.0);
    assert_eq!(tracker.consecutive_count(), 1);
    assert_eq!(tracker.observe(Some(7)), 0.4);
}

#[test]
fn changed_fingerprint_resets_count() {
    let mut tracker = FingerprintTracker::default();
    assert_eq!(tracker.observe(Some(7)), 0.0);
    assert_eq!(tracker.observe(Some(7)), 0.4);
    assert_eq!(tracker.observe(Some(8)), 0.0);
    assert_eq!(tracker.consecutive_count(), 1);
    assert_eq!(tracker.last_fingerprint(), Some(8));
}
