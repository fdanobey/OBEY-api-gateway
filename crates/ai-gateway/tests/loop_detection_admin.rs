use ai_gateway::loop_detection::{
    admin::reset_session_state, EnforcementLevel, RequestRecord, ResponseDescriptor, SessionState,
};
use proptest::prelude::*;
use std::time::Instant;

proptest! {
    #![proptest_config(ProptestConfig::with_cases(128))]

    // Feature: agent-loop-detection, Property 17: Admin Reset Produces Baseline State
    #[test]
    fn prop_admin_reset_produces_baseline(
        request_count in 1u32..=1000,
        confidence in 0.0f32..=1.0,
        total_tokens in any::<u64>(),
        total_cost in 0.0f64..1_000_000.0,
    ) {
        let mut session = SessionState::new(Some("vk-id".into()), 5);
        session.request_count = request_count;
        session.smoothed_confidence = confidence;
        session.peak_confidence = confidence;
        session.total_tokens = total_tokens;
        session.total_cost = total_cost;
        session.enforcement_level = EnforcementLevel::HardStop;
        session.dominant_signal = "error_cycling";
        session.request_hashes.push_back(42);
        session.tool_fingerprints.push_back(7);
        session.record_response(ResponseDescriptor { token_count: 10, block_type_hash: 2, is_error: true });
        session.record_request(&RequestRecord {
            content_simhash: 1,
            tool_call_fingerprint: Some(1),
            context_token_count: 1,
            new_information_tokens: 1,
            token_count: 1,
            cost: 1.0,
            has_tool_calls: true,
            tool_names: vec!["tool".into()],
            discovery_keys: vec![],
            timestamp: Instant::now(),
        });

        reset_session_state(&mut session);

        prop_assert_eq!(session.vk_id.as_deref(), Some("vk-id"));
        prop_assert_eq!(session.history_depth(), 5);
        prop_assert_eq!(session.request_count, 0);
        prop_assert_eq!(session.smoothed_confidence, 0.0);
        prop_assert_eq!(session.enforcement_level, EnforcementLevel::None);
        prop_assert_eq!(session.dominant_signal, "none");
        prop_assert!(session.request_hashes.is_empty());
        prop_assert!(session.tool_fingerprints.is_empty());
        prop_assert!(session.response_descriptors.is_empty());
        prop_assert!(session.escalation_history.is_empty());
    }
}
