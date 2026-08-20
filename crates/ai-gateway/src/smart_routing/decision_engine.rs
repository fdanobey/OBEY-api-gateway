use super::config::TierBoundaries;
use super::tier::{ComplexityScore, SmartRoutingTier};

/// Stateless cost/quality adjustment and tier-selection engine.
#[derive(Debug, Default, Clone, Copy)]
pub struct DecisionEngine;

/// Score adjustment and tier selected for a routing decision.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Decision {
    pub score: ComplexityScore,
    pub adjusted_score: ComplexityScore,
    pub tier: SmartRoutingTier,
}

impl DecisionEngine {
    /// Apply the configured cost/quality bias to a complexity score.
    ///
    /// Non-finite thresholds use the neutral `0.5` bias. Finite values are
    /// normalized to the supported range so this method remains safe even if
    /// called with configuration that has not passed validation.
    pub fn adjust_score(score: ComplexityScore, threshold: f64) -> ComplexityScore {
        let threshold = normalize_threshold(threshold);
        ComplexityScore::new(score.value() * (1.0 + (threshold - 0.5)))
    }

    /// Map a threshold-adjusted score to a configured capability tier.
    pub fn select_tier(
        &self,
        score: ComplexityScore,
        boundaries: &TierBoundaries,
        cost_quality_threshold: f64,
    ) -> SmartRoutingTier {
        self.decide(score, boundaries, cost_quality_threshold).tier
    }

    /// Produce the complete score adjustment and tier selection in one pass.
    pub fn decide(
        &self,
        score: ComplexityScore,
        boundaries: &TierBoundaries,
        cost_quality_threshold: f64,
    ) -> Decision {
        let adjusted_score = Self::adjust_score(score, cost_quality_threshold);
        let boundaries = normalize_boundaries(boundaries);
        let value = adjusted_score.value();
        let tier = if value < boundaries.fast_max {
            SmartRoutingTier::Fast
        } else if value <= boundaries.balanced_max {
            SmartRoutingTier::Balanced
        } else {
            SmartRoutingTier::Powerful
        };

        Decision {
            score,
            adjusted_score,
            tier,
        }
    }
}

fn normalize_threshold(threshold: f64) -> f64 {
    if threshold.is_finite() {
        threshold.clamp(0.0, 1.0)
    } else {
        0.5
    }
}

fn normalize_boundaries(boundaries: &TierBoundaries) -> TierBoundaries {
    if boundaries.fast_max.is_finite()
        && boundaries.balanced_max.is_finite()
        && boundaries.fast_max > 0.0
        && boundaries.fast_max < boundaries.balanced_max
        && boundaries.balanced_max < 1.0
    {
        boundaries.clone()
    } else {
        TierBoundaries::default()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use proptest::prelude::*;

    const EPSILON: f64 = 1.0e-12;

    fn unit_fraction(value: u16) -> f64 {
        value as f64 / 1000.0
    }

    fn arb_score() -> impl Strategy<Value = ComplexityScore> {
        (0u16..=1000).prop_map(|value| ComplexityScore::new(unit_fraction(value)))
    }

    fn arb_valid_boundaries() -> impl Strategy<Value = TierBoundaries> {
        (1u16..999).prop_flat_map(|fast| {
            (fast + 1..1000).prop_map(move |balanced| TierBoundaries {
                fast_max: unit_fraction(fast),
                balanced_max: unit_fraction(balanced),
            })
        })
    }

    fn boundaries() -> TierBoundaries {
        TierBoundaries {
            fast_max: 0.33,
            balanced_max: 0.66,
        }
    }

    fn select(score: f64) -> SmartRoutingTier {
        DecisionEngine.select_tier(ComplexityScore::new(score), &boundaries(), 0.5)
    }

    fn tier_rank(tier: SmartRoutingTier) -> u8 {
        match tier {
            SmartRoutingTier::Fast => 0,
            SmartRoutingTier::Balanced => 1,
            SmartRoutingTier::Powerful => 2,
        }
    }

    proptest! {
    #![proptest_config(ProptestConfig::with_cases(64))]

    #[test]
    fn property_7_tier_mapping_is_complete(
    score in arb_score(),
    boundaries in arb_valid_boundaries(),
    ) {
    let value = score.value();
    let tier = DecisionEngine.select_tier(score, &boundaries, 0.5);
    let expected = if value < boundaries.fast_max {
    SmartRoutingTier::Fast
    } else if value <= boundaries.balanced_max {
    SmartRoutingTier::Balanced
    } else {
    SmartRoutingTier::Powerful
    };

    prop_assert!(boundaries.fast_max > 0.0);
    prop_assert!(boundaries.fast_max < boundaries.balanced_max);
    prop_assert!(boundaries.balanced_max < 1.0);
    prop_assert_eq!(tier, expected);
    }

    #[test]
    fn property_15_cost_quality_formula_is_monotonic_and_neutral_is_no_op(
    score in arb_score(),
    first_threshold in 0u16..=1000,
    second_threshold in 0u16..=1000,
    boundaries in arb_valid_boundaries(),
    ) {
    let low_threshold = unit_fraction(first_threshold.min(second_threshold));
    let high_threshold = unit_fraction(first_threshold.max(second_threshold));
    let low = DecisionEngine.decide(score, &boundaries, low_threshold);
    let high = DecisionEngine.decide(score, &boundaries, high_threshold);
    let neutral = DecisionEngine.decide(score, &boundaries, 0.5);
    let expected_low = ComplexityScore::new(
    score.value() * (1.0 + (low_threshold - 0.5)),
    );
    let expected_high = ComplexityScore::new(
    score.value() * (1.0 + (high_threshold - 0.5)),
    );

    prop_assert_eq!(low.adjusted_score, expected_low);
    prop_assert_eq!(high.adjusted_score, expected_high);
    prop_assert!(low.adjusted_score <= high.adjusted_score);
    prop_assert!(tier_rank(low.tier) <= tier_rank(high.tier));
    prop_assert_eq!(neutral.adjusted_score, score);
    let neutral_expected_tier = if score.value() < boundaries.fast_max {
    SmartRoutingTier::Fast
    } else if score.value() <= boundaries.balanced_max {
    SmartRoutingTier::Balanced
    } else {
    SmartRoutingTier::Powerful
    };
    prop_assert_eq!(neutral.tier, neutral_expected_tier);
    }
    }

    #[test]
    fn maps_exact_tier_boundaries() {
        assert_eq!(select(0.0), SmartRoutingTier::Fast);
        assert_eq!(select(0.33 - EPSILON), SmartRoutingTier::Fast);
        assert_eq!(select(0.33), SmartRoutingTier::Balanced);
        assert_eq!(select(0.66), SmartRoutingTier::Balanced);
        assert_eq!(select(0.66 + EPSILON), SmartRoutingTier::Powerful);
        assert_eq!(select(1.0), SmartRoutingTier::Powerful);
    }

    #[test]
    fn threshold_adjustment_is_neutral_at_midpoint() {
        let score = ComplexityScore::new(0.72);
        assert_eq!(DecisionEngine::adjust_score(score, 0.5), score);
    }

    #[test]
    fn threshold_adjustment_is_monotonic_for_examples() {
        let score = ComplexityScore::new(0.6);
        let low = DecisionEngine::adjust_score(score, 0.0);
        let neutral = DecisionEngine::adjust_score(score, 0.5);
        let high = DecisionEngine::adjust_score(score, 1.0);

        assert_eq!(low.value(), 0.3);
        assert_eq!(neutral.value(), 0.6);
        assert!((high.value() - 0.9).abs() < f64::EPSILON);
        assert!(low < neutral);
        assert!(neutral < high);

        let engine = DecisionEngine;
        assert_eq!(
            engine.select_tier(score, &boundaries(), 0.0),
            SmartRoutingTier::Fast
        );
        assert_eq!(
            engine.select_tier(score, &boundaries(), 0.5),
            SmartRoutingTier::Balanced
        );
        assert_eq!(
            engine.select_tier(score, &boundaries(), 1.0),
            SmartRoutingTier::Powerful
        );
    }

    #[test]
    fn adjustment_clamps_to_unit_interval() {
        assert_eq!(
            DecisionEngine::adjust_score(ComplexityScore::new(1.0), 1.0).value(),
            1.0
        );
        assert_eq!(
            DecisionEngine::adjust_score(ComplexityScore::new(0.0), 0.0).value(),
            0.0
        );
    }

    #[test]
    fn decision_exposes_original_and_adjusted_scores() {
        let score = ComplexityScore::new(0.6);
        let decision = DecisionEngine.decide(score, &boundaries(), 1.0);

        assert_eq!(decision.score, score);
        assert!((decision.adjusted_score.value() - 0.9).abs() < f64::EPSILON);
        assert_eq!(decision.tier, SmartRoutingTier::Powerful);
    }

    #[test]
    fn non_finite_thresholds_use_neutral_bias() {
        let score = ComplexityScore::new(0.6);

        for threshold in [f64::NAN, f64::INFINITY, f64::NEG_INFINITY] {
            assert_eq!(DecisionEngine::adjust_score(score, threshold), score);
        }
    }

    #[test]
    fn non_finite_boundaries_fall_back_to_defaults() {
        let invalid = TierBoundaries {
            fast_max: f64::NAN,
            balanced_max: f64::INFINITY,
        };

        assert_eq!(
            DecisionEngine.select_tier(ComplexityScore::new(0.5), &invalid, 0.5),
            SmartRoutingTier::Balanced
        );
    }
}
