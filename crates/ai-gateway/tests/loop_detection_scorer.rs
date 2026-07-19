use ai_gateway::loop_detection::{scorer::ConfidenceScorer, signals::SignalValues, SignalWeights};
use proptest::prelude::*;

fn signals(values: [f32; 7]) -> SignalValues {
    SignalValues {
        content_similarity: values[0],
        tool_call_repetition: values[1],
        response_stagnation: values[2],
        token_velocity: values[3],
        error_cycling: values[4],
        context_growth: values[5],
        cost_velocity: values[6],
    }
}

proptest! {
    #![proptest_config(ProptestConfig::with_cases(128))]

    // Feature: agent-loop-detection, Property 1: Confidence Score Normalization
    #[test]
    fn prop_confidence_is_normalized(values in prop::array::uniform7(-10.0f32..=10.0)) {
        let result = ConfidenceScorer::score(
            &signals(values),
            &SignalWeights::default(),
            0.5,
            0.3,
            3,
        );
        prop_assert!((0.0..=1.0).contains(&result.raw_confidence));
        prop_assert!((0.0..=1.0).contains(&result.confidence));
    }

    // Feature: agent-loop-detection, Property 2: EMA Monotone Convergence
    #[test]
    fn prop_ema_moves_monotonically_toward_constant(
        target in 0.0f32..=1.0,
        initial in 0.0f32..=1.0,
        alpha in 0.01f32..=1.0,
    ) {
        let constant = signals([target; 7]);
        let mut current = initial;
        let initial_distance = (initial - target).abs();
        for _ in 0..20 {
            let next = ConfidenceScorer::score(
                &constant,
                &SignalWeights::default(),
                current,
                alpha,
                3,
            ).confidence;
            if current < target {
                prop_assert!(next + 1e-6 >= current && next <= target + 1e-6);
            } else if current > target {
                prop_assert!(next <= current + 1e-6 && next + 1e-6 >= target);
            }
            current = next;
        }
        prop_assert!((current - target).abs() <= initial_distance + f32::EPSILON);
    }

    // Feature: agent-loop-detection, Property 3: Suppression Below Minimum Requests
    #[test]
    fn prop_fewer_than_three_requests_are_suppressed(
        values in prop::array::uniform7(any::<f32>()),
        request_count in 0usize..3,
    ) {
        let result = ConfidenceScorer::score(
            &signals(values),
            &SignalWeights::default(),
            1.0,
            1.0,
            request_count,
        );
        prop_assert_eq!(result.confidence, 0.0);
        prop_assert_eq!(result.raw_confidence, 0.0);
        prop_assert_eq!(result.dominant_signal, "none");
    }
}

#[test]
fn dominant_signal_uses_weighted_contribution() {
    let result = ConfidenceScorer::score(
        &SignalValues {
            content_similarity: 0.5,
            tool_call_repetition: 1.0,
            ..SignalValues::default()
        },
        &SignalWeights::default(),
        0.0,
        1.0,
        3,
    );
    assert_eq!(result.dominant_signal, "tool_call_repetition");
}
