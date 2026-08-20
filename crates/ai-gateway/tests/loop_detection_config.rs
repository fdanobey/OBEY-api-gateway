use ai_gateway::loop_detection::{
    LoopDetectionConfig, LoopDetectionConfigError, SignalWeights, ThresholdConfig,
};
use proptest::prelude::*;

proptest! {
    #![proptest_config(ProptestConfig::with_cases(64))]

    // Feature: agent-loop-detection, Property 5: Threshold Ordering Validation
    #[test]
    fn prop_threshold_ordering_validation(
        warn in 0.0f32..=1.0,
        throttle in 0.0f32..=1.0,
        inject in 0.0f32..=1.0,
        hardstop in 0.0f32..=1.0,
    ) {
        let config = LoopDetectionConfig {
            thresholds: ThresholdConfig {
                warn_confidence: warn,
                throttle_confidence: throttle,
                inject_confidence: inject,
                hardstop_confidence: hardstop,
            },
            ..LoopDetectionConfig::default()
        };
        let expected_valid = warn < throttle && throttle < inject && inject < hardstop;
        let result = config.validate();

        prop_assert_eq!(result.is_ok(), expected_valid);
        if !expected_valid {
            prop_assert!(
                result.unwrap_err().iter().any(|error| matches!(
                    error,
                    LoopDetectionConfigError::ThresholdOrder { .. }
                )),
                "misordered thresholds must report a threshold-order error"
            );
        }
    }

    // Feature: agent-loop-detection, Property 6: Weight Sum Validation
    #[test]
    fn prop_weight_sum_validation(values in prop::array::uniform7(-0.25f32..=1.25)) {
        let weights = SignalWeights {
            content_similarity: values[0],
            tool_call_repetition: values[1],
            response_stagnation: values[2],
            token_velocity: values[3],
            error_cycling: values[4],
            context_growth: values[5],
            cost_velocity: values[6],
        };
        let expected_valid = values.iter().all(|value| (0.0..=1.0).contains(value))
            && (values.iter().sum::<f32>() - 1.0).abs() <= 0.001;
        let result = LoopDetectionConfig {
            weights,
            ..LoopDetectionConfig::default()
        }
        .validate();

        prop_assert_eq!(result.is_ok(), expected_valid);
    }

    // Feature: agent-loop-detection, Property 15: Break Instruction Template Validation
    #[test]
    fn prop_break_instruction_template_validation(length in 0usize..=2_100) {
        let template = "x".repeat(length);
        let result = LoopDetectionConfig::validate_break_instruction_template(&template);

        prop_assert_eq!(result.is_ok(), (1..=2_000).contains(&length));
    }
}

#[test]
fn defaults_deserialize_from_empty_section() {
    let config: LoopDetectionConfig = serde_yaml::from_str("{}\n").unwrap();
    assert_eq!(config, LoopDetectionConfig::default());
    assert!(config.validate().is_ok());
}

#[test]
fn injection_strategy_uses_documented_snake_case_values() {
    let config: LoopDetectionConfig =
        serde_yaml::from_str("injection_strategy: context_aware\n").unwrap();
    assert_eq!(
        config.injection_strategy,
        ai_gateway::loop_detection::InjectionStrategy::ContextAware
    );
}
