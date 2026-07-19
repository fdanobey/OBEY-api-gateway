use ai_gateway::loop_detection::simhash::{
    compute, compute_messages, hamming_distance, similarity,
};
use proptest::prelude::*;

proptest! {
    #![proptest_config(ProptestConfig::with_cases(128))]

    // Feature: agent-loop-detection, Property 8: SimHash Identical Content Identity
    #[test]
    fn prop_identical_content_identity(content in ".{0,2000}") {
        let left = compute(&content);
        let right = compute(&content);
        prop_assert_eq!(left, right);
        prop_assert_eq!(hamming_distance(left, right), 0);
        prop_assert_eq!(similarity(left, right), 1.0);
    }

    #[test]
    fn prop_similarity_is_bounded(left in any::<u64>(), right in any::<u64>()) {
        let score = similarity(left, right);
        prop_assert!((0.0..=1.0).contains(&score));
    }
}

#[test]
fn normalization_and_stop_words_preserve_identity() {
    let left = compute("The QUICK   brown fox and the agile hound");
    let right = compute("quick brown fox agile hound");
    assert_eq!(left, right);
}

#[test]
fn zero_token_content_returns_zero_hash() {
    assert_eq!(compute("the and or"), 0);
    assert_eq!(similarity(0, 0), 1.0);
}

#[test]
fn tool_result_blocks_are_excluded() {
    let messages = serde_json::json!([
        {"role": "user", "content": "analyze alpha beta gamma"},
        {"role": "assistant", "content": [
            {"type": "text", "text": "delta epsilon zeta"},
            {"type": "tool_result", "text": "secret tool output changes"}
        ]},
        {"role": "tool", "content": "another tool payload"}
    ]);
    assert_eq!(
        compute_messages(&messages),
        compute("analyze alpha beta gamma delta epsilon zeta")
    );
}

#[test]
fn distinct_content_is_not_identical() {
    let left = compute("compile rust gateway middleware behavior");
    let right = compute("paint watercolor mountain landscape");
    assert!(similarity(left, right) < 1.0);
}
