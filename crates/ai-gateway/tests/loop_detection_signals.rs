use ai_gateway::loop_detection::{
    signals::{SignalComputer, SignalValues},
    LoopDetectionConfig, RequestRecord, ResponseDescriptor, SessionState,
};
use proptest::prelude::*;
use std::time::Instant;

fn request(tool_fingerprint: Option<u64>) -> RequestRecord {
    RequestRecord {
        content_simhash: 123,
        tool_call_fingerprint: tool_fingerprint,
        context_token_count: 100,
        new_information_tokens: 10,
        token_count: 100,
        cost: 0.01,
        has_tool_calls: tool_fingerprint.is_some(),
        tool_names: Vec::new(),
        timestamp: Instant::now(),
    }
}

proptest! {
    #![proptest_config(ProptestConfig::with_cases(128))]

    // Feature: agent-loop-detection, Property 16: Signal Non-Applicability Returns Zero
    #[test]
    fn prop_sessions_with_fewer_than_two_requests_return_zero(
        hash in any::<u64>(),
        tokens in any::<u32>(),
        context in any::<u32>(),
        cost in 0.0f64..1000.0,
    ) {
        let mut current = request(Some(hash));
        current.content_simhash = hash;
        current.token_count = tokens;
        current.context_token_count = context;
        current.cost = cost;
        let session = SessionState::new(Some("vk".into()), 5);

        prop_assert_eq!(
            SignalComputer::compute(
                &session,
                &current,
                None,
                &LoopDetectionConfig::default(),
                Some(cost),
            ),
            SignalValues::default()
        );
    }
}

#[test]
fn no_tool_calls_produce_zero_tool_repetition() {
    let mut session = SessionState::new(Some("vk".into()), 5);
    session.request_count = 2;
    session.tool_fingerprints.extend([7, 7]);

    let signals = SignalComputer::compute(
        &session,
        &request(None),
        None,
        &LoopDetectionConfig::default(),
        None,
    );
    assert_eq!(signals.tool_call_repetition, 0.0);
}

#[test]
fn no_response_produces_zero_response_stagnation_and_error_cycling() {
    let mut session = SessionState::new(Some("vk".into()), 5);
    session.request_count = 2;

    let signals = SignalComputer::compute(
        &session,
        &request(Some(7)),
        None,
        &LoopDetectionConfig::default(),
        None,
    );
    assert_eq!(signals.response_stagnation, 0.0);
    assert_eq!(signals.error_cycling, 0.0);
}

#[test]
fn prior_responses_drive_stagnation_and_error_retry_signals() {
    let mut session = SessionState::new(Some("vk".into()), 5);
    session.request_count = 3;
    session.request_hashes.extend([123, 123, 123]);
    session.response_descriptors.extend([
        ResponseDescriptor { token_count: 100, block_type_hash: 9, is_error: true },
        ResponseDescriptor { token_count: 101, block_type_hash: 9, is_error: true },
        ResponseDescriptor { token_count: 99, block_type_hash: 9, is_error: true },
    ]);
    session.error_retry_cycles = 2;

    let signals = SignalComputer::compute(
        &session,
        &request(Some(7)),
        None,
        &LoopDetectionConfig::default(),
        None,
    );
    assert_eq!(signals.response_stagnation, 0.6);
    assert_eq!(signals.error_cycling, 1.0);
}

#[test]
fn all_signal_values_are_bounded() {
    let mut session = SessionState::new(Some("vk".into()), 5);
    session.request_count = 10;
    session.request_hashes.extend([123, 123]);
    session.tool_fingerprints.extend([7, 7, 7]);
    session.context_token_counts.push_back(1);
    session.error_retry_cycles = 10;

    let mut current = request(Some(7));
    current.context_token_count = u32::MAX;
    current.new_information_tokens = 0;
    current.token_count = u32::MAX;
    let signals = SignalComputer::compute(
        &session,
        &current,
        Some(&ai_gateway::loop_detection::ResponseDescriptor {
            token_count: 100,
            block_type_hash: 9,
            is_error: true,
        }),
        &LoopDetectionConfig::default(),
        Some(f64::MAX),
    );

    for (_, value) in signals.iter() {
        assert!((0.0..=1.0).contains(&value));
    }
}
