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
        discovery_keys: Vec::new(),
        timestamp: Instant::now(),
    }
}

proptest! {
    #![proptest_config(ProptestConfig::with_cases(64))]

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
        ResponseDescriptor {
            token_count: 100,
            block_type_hash: 9,
            is_error: true,
        },
        ResponseDescriptor {
            token_count: 101,
            block_type_hash: 9,
            is_error: true,
        },
        ResponseDescriptor {
            token_count: 99,
            block_type_hash: 9,
            is_error: true,
        },
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

#[test]
fn discovery_loop_raises_tool_call_repetition() {
    let mut session = SessionState::new(Some("vk".into()), 8);

    // Seed two ordinary requests so the signal computer is out of its warm-up gate.
    session.record_request(&request(None));
    session.record_request(&request(None));

    // Mirror the middleware order: compute runs BEFORE record_request, so the signal
    // never sees the current request's own discovery keys in the history yet.
    // First disclosure of namespace `fs`: history is still empty, so no repeat yet.
    let mut first = request(None);
    first.discovery_keys = vec!["ns:fs".into()];
    let first_signals = SignalComputer::compute(
        &session,
        &first,
        None,
        &LoopDetectionConfig::default(),
        None,
    );
    assert_eq!(first_signals.tool_call_repetition, 0.0);
    session.record_request(&first);

    // An unrelated request in between (so the re-drill is NOT consecutive).
    session.record_request(&request(None));

    // Re-drill of `ns:fs`: the non-consecutive discovery repeat must be detected.
    let mut redrill = request(None);
    redrill.discovery_keys = vec!["ns:fs".into()];
    let signals = SignalComputer::compute(
        &session,
        &redrill,
        None,
        &LoopDetectionConfig::default(),
        None,
    );
    assert!(
        signals.tool_call_repetition > 0.0,
        "non-consecutive synthetic discovery re-drill should be monitored"
    );
    session.record_request(&redrill);

    // A genuine (non-discovery) tool call must not be flagged by this mechanism.
    let mut real = request(None);
    real.discovery_keys = vec![];
    let real_signals =
        SignalComputer::compute(&session, &real, None, &LoopDetectionConfig::default(), None);
    assert_eq!(real_signals.tool_call_repetition, 0.0);
}
