use ai_gateway::loop_detection::{
    InjectionStrategy, LoopDetectionConfig, SignalWeights, VkLoopConfig,
};
use proptest::prelude::*;

proptest! {
    #![proptest_config(ProptestConfig::with_cases(128))]

    // Feature: agent-loop-detection, Property 14: Per-VK Override Inheritance
    #[test]
    fn prop_partial_override_inherits_unspecified_fields(
        delay in 1u32..=30,
        context_weight in 0.0f32..=1.0,
    ) {
        let mut global = LoopDetectionConfig::default();
        global.weights.context_growth = context_weight;
        global.weights.content_similarity += 0.10 - context_weight;
        prop_assume!(global.weights.content_similarity >= 0.0 && global.validate().is_ok());
        let override_config = VkLoopConfig {
            throttle_delay_seconds: Some(delay),
            injection_strategy: Some(InjectionStrategy::ContextAware),
            ..Default::default()
        };
        let merged = override_config.merge(&global).unwrap();

        prop_assert_eq!(merged.throttle_delay_seconds, delay);
        prop_assert_eq!(merged.injection_strategy, InjectionStrategy::ContextAware);
        prop_assert_eq!(&merged.weights, &global.weights);
        prop_assert_eq!(&merged.thresholds, &global.thresholds);
        prop_assert_eq!(merged.history_depth, global.history_depth);
    }
}

#[test]
fn invalid_weight_override_is_rejected() {
    let override_config = VkLoopConfig {
        weights: Some(SignalWeights {
            content_similarity: 1.0,
            ..SignalWeights::default()
        }),
        ..Default::default()
    };
    assert!(override_config
        .merge(&LoopDetectionConfig::default())
        .is_err());
}

#[test]
fn full_override_replaces_only_supported_fields() {
    let global = LoopDetectionConfig::default();
    let override_config: VkLoopConfig = serde_json::from_value(serde_json::json!({
        "throttle_delay_seconds": 10,
        "injection_strategy": "context_aware",
        "break_instruction_template": "escape now"
    }))
    .unwrap();
    let merged = override_config.merge(&global).unwrap();
    assert_eq!(merged.throttle_delay_seconds, 10);
    assert_eq!(merged.injection_strategy, InjectionStrategy::ContextAware);
    assert_eq!(
        merged.break_instruction_template.as_deref(),
        Some("escape now")
    );
    assert_eq!(
        merged.session_timeout_minutes,
        global.session_timeout_minutes
    );
}
