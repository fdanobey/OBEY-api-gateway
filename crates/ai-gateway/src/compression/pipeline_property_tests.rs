use super::config::{
    CompressionConfig, EffectiveCompressionConfig, ModelGroupCompressionOverride,
    ProviderCompressionOverride,
};
use super::engines::CompressionLevel;
use super::pipeline::{decide_compression, AutoTriggerReason};
use proptest::prelude::*;

fn compression_level_strategy() -> impl Strategy<Value = CompressionLevel> {
    prop_oneof![
        Just(CompressionLevel::None),
        Just(CompressionLevel::Lite),
        Just(CompressionLevel::Standard),
        Just(CompressionLevel::Aggressive),
        Just(CompressionLevel::Ultra),
        Just(CompressionLevel::Rtk),
        Just(CompressionLevel::Stacked),
    ]
}

fn optional_level_strategy() -> impl Strategy<Value = Option<CompressionLevel>> {
    prop_oneof![Just(None), compression_level_strategy().prop_map(Some)]
}

proptest! {
    #![proptest_config(ProptestConfig::with_cases(64))]

    #[test]
    fn property_6_effective_config_drives_pipeline_decision_with_field_local_precedence(
        global_enabled in any::<bool>(),
        global_level in compression_level_strategy(),
        global_threshold in any::<u32>(),
        provider_enabled in proptest::option::of(any::<bool>()),
        provider_level in optional_level_strategy(),
        provider_threshold in proptest::option::of(any::<u32>()),
        provider_caveman in proptest::option::of(any::<bool>()),
        model_level in optional_level_strategy(),
        model_threshold in proptest::option::of(any::<u32>()),
        model_caveman in proptest::option::of(any::<bool>()),
        original_tokens in any::<u32>(),
    ) {
        let global = CompressionConfig {
            enabled: global_enabled,
            default_level: global_level,
            auto_threshold_tokens: global_threshold,
            ..CompressionConfig::default()
        };
        let provider = ProviderCompressionOverride {
            enabled: provider_enabled,
            level: provider_level,
            auto_threshold_tokens: provider_threshold,
            caveman_output: provider_caveman,
        };
        let model_group = ModelGroupCompressionOverride {
            level: model_level,
            auto_threshold_tokens: model_threshold,
            caveman_output: model_caveman,
        };

        let effective = global.resolve(Some(&provider), Some(&model_group));
        let expected = EffectiveCompressionConfig {
            enabled: provider_enabled.unwrap_or(global_enabled),
            level: model_level.or(provider_level).unwrap_or(global_level),
            auto_threshold_tokens: model_threshold
                .or(provider_threshold)
                .unwrap_or(global_threshold),
            caveman_output: model_caveman
                .or(provider_caveman)
                .unwrap_or(global.caveman_output),
        };
        prop_assert_eq!(effective, expected);

        let decision = decide_compression(original_tokens, &effective);
        let expected_trigger = expected.enabled
            && expected.level != CompressionLevel::None
            && expected.auto_threshold_tokens > 0
            && original_tokens > expected.auto_threshold_tokens;
        prop_assert_eq!(decision.should_compress, expected_trigger);
        prop_assert_eq!(decision.threshold_tokens, expected.auto_threshold_tokens);
        prop_assert_eq!(decision.level, expected.level);
    }

    #[test]
    fn property_7_auto_threshold_trigger_correctness(
        threshold in 1u32..u32::MAX,
        relation in 0u8..3,
        enabled in any::<bool>(),
        level in compression_level_strategy(),
    ) {
        let original_tokens = match relation {
            0 => threshold.saturating_sub(1),
            1 => threshold,
            _ => threshold.saturating_add(1),
        };
        let effective = EffectiveCompressionConfig {
            enabled,
            level,
            auto_threshold_tokens: threshold,
            caveman_output: false,
        };
        let decision = decide_compression(original_tokens, &effective);
        let expected = enabled
            && level != CompressionLevel::None
            && original_tokens > threshold;

        prop_assert_eq!(decision.should_compress, expected);
        prop_assert_eq!(decision.auto_triggered, expected);
        prop_assert_eq!(decision.original_tokens, original_tokens);
        if !enabled {
            prop_assert_eq!(decision.reason, AutoTriggerReason::ExplicitlyDisabled);
        } else if level == CompressionLevel::None {
            prop_assert_eq!(decision.reason, AutoTriggerReason::LevelNone);
        } else if original_tokens > threshold {
            prop_assert_eq!(decision.reason, AutoTriggerReason::ThresholdExceeded);
        } else {
            prop_assert_eq!(decision.reason, AutoTriggerReason::AtOrBelowThreshold);
        }
    }
}
