use crate::loop_detection::{config::SignalWeights, signals::SignalValues};

#[derive(Debug, Clone, Copy, PartialEq)]
pub struct ScoreResult {
    pub confidence: f32,
    pub raw_confidence: f32,
    pub dominant_signal: &'static str,
}

pub struct ConfidenceScorer;

impl ConfidenceScorer {
    pub fn score(
        signals: &SignalValues,
        weights: &SignalWeights,
        previous_ema: f32,
        alpha: f32,
        request_count: usize,
    ) -> ScoreResult {
        if request_count < 3 {
            return ScoreResult {
                confidence: 0.0,
                raw_confidence: 0.0,
                dominant_signal: "none",
            };
        }

        let contributions = [
            (
                "content_similarity",
                clamp_signal(signals.content_similarity) * weights.content_similarity,
            ),
            (
                "tool_call_repetition",
                clamp_signal(signals.tool_call_repetition) * weights.tool_call_repetition,
            ),
            (
                "response_stagnation",
                clamp_signal(signals.response_stagnation) * weights.response_stagnation,
            ),
            (
                "token_velocity",
                clamp_signal(signals.token_velocity) * weights.token_velocity,
            ),
            (
                "error_cycling",
                clamp_signal(signals.error_cycling) * weights.error_cycling,
            ),
            (
                "context_growth",
                clamp_signal(signals.context_growth) * weights.context_growth,
            ),
            (
                "cost_velocity",
                clamp_signal(signals.cost_velocity) * weights.cost_velocity,
            ),
        ];
        let weight_sum = weights.sum();
        let weighted_sum: f32 = contributions
            .iter()
            .map(|(_, contribution)| contribution)
            .sum();
        let raw_confidence = if weight_sum.is_finite() && weight_sum > 0.0 {
            (weighted_sum / weight_sum).clamp(0.0, 1.0)
        } else {
            0.0
        };
        let alpha = alpha.clamp(0.01, 1.0);
        let previous_ema = previous_ema.clamp(0.0, 1.0);
        let confidence = (alpha * raw_confidence + (1.0 - alpha) * previous_ema).clamp(0.0, 1.0);
        let maximum_contribution = contributions
            .iter()
            .map(|(_, contribution)| *contribution)
            .max_by(f32::total_cmp)
            .unwrap_or(0.0);
        let tool_error_tie = maximum_contribution > 0.0
            && (contributions[1].1 - maximum_contribution).abs() <= f32::EPSILON
            && (contributions[4].1 - maximum_contribution).abs() <= f32::EPSILON;
        let dominant_signal = if tool_error_tie {
            "none"
        } else {
            contributions
                .iter()
                .max_by(|left, right| left.1.total_cmp(&right.1))
                .filter(|(_, contribution)| *contribution > 0.0)
                .map_or("none", |(name, _)| *name)
        };

        ScoreResult {
            confidence,
            raw_confidence,
            dominant_signal,
        }
    }
}

fn clamp_signal(value: f32) -> f32 {
    if value.is_nan() {
        0.0
    } else {
        value.clamp(0.0, 1.0)
    }
}
