//! Pure composition of System One answers into a complexity score and task
//! type: per-dimension normalization, weighting, and confidence gating.
//! Family-neutral except for confidence-field selection, which is supplied
//! by the family descriptor.
//!
//! No I/O here — everything is deterministic and property-testable. The
//! caller owns fallback behavior for `LowConfidence` and `Invalid` results.

use std::collections::HashMap;

use crate::smart_routing::config::{DimensionWeights, JevFallbackPolicy};
use crate::smart_routing::tier::{ComplexityScore, TaskType};

use super::models::Answer;
use super::rubric::{DIMENSION_LEVELS, DIMENSION_QUESTION_IDS, TASK_TYPE_QUESTION_ID};
use super::SystemOneFamily;

/// Trust thresholds applied to composed answers.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct JevTrustConfig {
    pub min_confidence: f64,
    pub min_task_confidence: f64,
    pub fallback_policy: JevFallbackPolicy,
}

/// Result of composing a System One response.
#[derive(Debug, Clone, PartialEq)]
pub enum ComposeResult {
    /// Composite score, task type, and the composite confidence.
    Ok {
        score: ComplexityScore,
        task_type: TaskType,
        confidence: f64,
    },
    /// Composite confidence is below `min_confidence`; the caller must apply
    /// the fallback policy (fallback chain or blend with the heuristic score).
    LowConfidence {
        score: ComplexityScore,
        confidence: f64,
    },
    /// A required answer was missing or malformed; the caller falls back.
    Invalid,
}

/// Compose answers into a complexity score and task type.
///
/// Each dimension's `score / (levels - 1)` normalizes to 0..=1; the weighted
/// mean (weights normalized to sum 1) forms the composite. The composite
/// confidence is the family-selected confidence of the least-confident
/// dimension — a chain is as strong as its weakest signal, which keeps
/// gating conservative.
pub fn compose(
    answers: &HashMap<String, Answer>,
    weights: &DimensionWeights,
    trust: &JevTrustConfig,
    family: &SystemOneFamily,
) -> ComposeResult {
    let mut total_weight = 0.0;
    let mut weighted_sum = 0.0;
    let mut min_confidence = f64::INFINITY;

    for id in DIMENSION_QUESTION_IDS {
        let Some(Answer::Score { score, .. }) = answers.get(id) else {
            return ComposeResult::Invalid;
        };
        let score = *score;
        let Some(confidence) = answers.get(id).and_then(|a| a.gate_confidence(family)) else {
            return ComposeResult::Invalid;
        };
        if !score.is_finite() || !confidence.is_finite() || !(0.0..=1.0).contains(&confidence) {
            return ComposeResult::Invalid;
        }

        let normalized = (score / (DIMENSION_LEVELS as f64 - 1.0)).clamp(0.0, 1.0);
        let (_, weight) = weights
            .as_slice()
            .into_iter()
            .find(|(name, _)| *name == id)
            .expect("dimension ids match weight names");
        weighted_sum += normalized * weight;
        total_weight += weight;
        min_confidence = min_confidence.min(confidence);
    }

    if !total_weight.is_finite() || total_weight <= 0.0 {
        return ComposeResult::Invalid;
    }

    let score = ComplexityScore::new(weighted_sum / total_weight);
    let confidence = min_confidence;

    if confidence < trust.min_confidence {
        return ComposeResult::LowConfidence { score, confidence };
    }

    let task_type = match answers.get(TASK_TYPE_QUESTION_ID) {
        Some(answer @ Answer::Choice { choice, .. }) => {
            let task_confidence = answer
                .gate_confidence(family)
                .or_else(|| answer.confidence());
            match task_confidence {
                Some(confidence) if confidence >= trust.min_task_confidence => {
                    parse_task_type(choice).unwrap_or(TaskType::General)
                }
                _ => TaskType::General,
            }
        }
        // Below the task-confidence gate (or missing/malformed choice): the
        // caller keeps the heuristic task type; `General` is a placeholder
        // the facade replaces.
        _ => TaskType::General,
    };

    ComposeResult::Ok {
        score,
        task_type,
        confidence,
    }
}

/// Blend a low-confidence System One score with the heuristic score.
///
/// The blend weight is the confidence expressed as a fraction of the
/// threshold, so closer-to-threshold confidences trust the classifier more.
/// At zero confidence this returns the heuristic score unchanged.
pub fn blend_scores(
    systemone_score: ComplexityScore,
    heuristic_score: ComplexityScore,
    confidence: f64,
    min_confidence: f64,
) -> ComplexityScore {
    let denominator = if min_confidence > 0.0 {
        min_confidence
    } else {
        1.0
    };
    let systemone_weight = (confidence / denominator).clamp(0.0, 1.0);
    ComplexityScore::new(
        systemone_score.value() * systemone_weight
            + heuristic_score.value() * (1.0 - systemone_weight),
    )
}

/// Map a task-type option key to `TaskType`; unknown keys are None.
pub fn parse_task_type(choice: &str) -> Option<TaskType> {
    match choice {
        "code_generation" => Some(TaskType::CodeGeneration),
        "math_reasoning" => Some(TaskType::MathReasoning),
        "creative_writing" => Some(TaskType::CreativeWriting),
        "factual_qa" => Some(TaskType::FactualQA),
        "tool_use" => Some(TaskType::ToolUse),
        "summarization" => Some(TaskType::Summarization),
        "general" => Some(TaskType::General),
        _ => None,
    }
}

/// Build a Score answer map entry helper for tests and callers.
pub fn dimension_answer(score: f64, confidence: f64) -> Answer {
    Answer::Score {
        score,
        confidence,
        answer_confidence: None,
        probabilities: HashMap::new(),
        legend: HashMap::new(),
    }
}

/// Build a Laya-shaped Score answer (with `answer_confidence`) for tests and
/// callers.
pub fn dimension_answer_with_gate_confidence(
    score: f64,
    confidence: f64,
    answer_confidence: Option<f64>,
) -> Answer {
    Answer::Score {
        score,
        confidence,
        answer_confidence,
        probabilities: HashMap::new(),
        legend: HashMap::new(),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use proptest::prelude::*;

    fn trust() -> JevTrustConfig {
        JevTrustConfig {
            min_confidence: 0.60,
            min_task_confidence: 0.50,
            fallback_policy: JevFallbackPolicy::Fallback,
        }
    }
    fn full_answers(dimension_score: f64, dimension_confidence: f64) -> HashMap<String, Answer> {
        let mut answers = HashMap::new();
        for id in DIMENSION_QUESTION_IDS {
            answers.insert(
                id.to_string(),
                dimension_answer(dimension_score, dimension_confidence),
            );
        }
        answers.insert(
            TASK_TYPE_QUESTION_ID.to_string(),
            Answer::Choice {
                choice: "code_generation".to_string(),
                probabilities: HashMap::new(),
                confidence: 0.9,
                answer_confidence: None,
            },
        );
        answers
    }

    #[test]
    fn composes_uniform_high_confidence_answers() {
        let answers = full_answers(2.0, 0.9);
        let weights = DimensionWeights::default();
        let result = compose(&answers, &weights, &trust(), &SystemOneFamily::JEV);
        let ComposeResult::Ok {
            score,
            task_type,
            confidence,
        } = result
        else {
            panic!("expected ok compose, got {result:?}");
        };
        assert!((score.value() - 2.0 / 3.0).abs() < 1.0e-9);
        assert_eq!(task_type, TaskType::CodeGeneration);
        assert!((confidence - 0.9).abs() < f64::EPSILON);
    }

    #[test]
    fn laya_gates_on_answer_confidence_when_present() {
        let mut answers = HashMap::new();
        for id in DIMENSION_QUESTION_IDS {
            answers.insert(
                id.to_string(),
                dimension_answer_with_gate_confidence(2.0, 0.30, Some(0.95)),
            );
        }
        answers.insert(
            TASK_TYPE_QUESTION_ID.to_string(),
            Answer::Choice {
                choice: "code_generation".to_string(),
                probabilities: HashMap::new(),
                confidence: 0.2,
                answer_confidence: Some(0.9),
            },
        );
        let result = compose(
            &answers,
            &DimensionWeights::default(),
            &trust(),
            &SystemOneFamily::LAYA,
        );
        let ComposeResult::Ok { confidence, .. } = result else {
            panic!("expected ok compose, got {result:?}");
        };
        assert!((confidence - 0.95).abs() < f64::EPSILON);
    }

    #[test]
    fn laya_low_answer_confidence_falls_back_even_with_high_confidence_field() {
        let mut answers = HashMap::new();
        for id in DIMENSION_QUESTION_IDS {
            answers.insert(
                id.to_string(),
                dimension_answer_with_gate_confidence(2.0, 0.99, Some(0.10)),
            );
        }
        answers.insert(
            TASK_TYPE_QUESTION_ID.to_string(),
            Answer::Choice {
                choice: "code_generation".to_string(),
                probabilities: HashMap::new(),
                confidence: 0.9,
                answer_confidence: None,
            },
        );
        let result = compose(
            &answers,
            &DimensionWeights::default(),
            &trust(),
            &SystemOneFamily::LAYA,
        );
        assert!(matches!(
            result,
            ComposeResult::LowConfidence { confidence, .. } if (confidence - 0.10).abs() < f64::EPSILON
        ));
    }

    #[test]
    fn laya_missing_answer_confidence_falls_back_to_confidence() {
        let answers = full_answers(2.0, 0.9);
        let result = compose(
            &answers,
            &DimensionWeights::default(),
            &trust(),
            &SystemOneFamily::LAYA,
        );
        let ComposeResult::Ok { confidence, .. } = result else {
            panic!("expected ok compose, got {result:?}");
        };
        assert!((confidence - 0.9).abs() < f64::EPSILON);
    }

    #[test]
    fn weights_are_normalized_and_skewed_weights_steering_matters() {
        let mut answers = full_answers(3.0, 0.9);
        answers.insert(
            "reasoning_depth".to_string(),
            dimension_answer(0.0, 0.9),
        );

        let mut weights = DimensionWeights::default();
        weights.reasoning_depth = 0.0;
        let result = compose(&answers, &weights, &trust(), &SystemOneFamily::JEV);
        let ComposeResult::Ok { score, .. } = result else {
            panic!("expected ok compose, got {result:?}");
        };
        assert!((score.value() - 1.0).abs() < 1.0e-9, "zero-weight dimension ignored");
    }

    #[test]
    fn lowest_dimension_confidence_gates_the_composite() {
        let mut answers = full_answers(2.0, 0.9);
        answers.insert("domain_load".to_string(), dimension_answer(2.0, 0.61));
        let result = compose(
            &answers,
            &DimensionWeights::default(),
            &trust(),
            &SystemOneFamily::JEV,
        );
        assert!(matches!(
            result,
            ComposeResult::Ok { confidence, .. } if (confidence - 0.61).abs() < f64::EPSILON
        ));

        answers.insert("domain_load".to_string(), dimension_answer(2.0, 0.59));
        let result = compose(
            &answers,
            &DimensionWeights::default(),
            &trust(),
            &SystemOneFamily::JEV,
        );
        assert!(matches!(
            result,
            ComposeResult::LowConfidence { confidence, .. } if (confidence - 0.59).abs() < f64::EPSILON
        ));
    }

    #[test]
    fn missing_or_malformed_answers_are_invalid() {
        let mut answers = full_answers(2.0, 0.9);
        answers.remove("domain_load");
        assert_eq!(
            compose(
                &answers,
                &DimensionWeights::default(),
                &trust(),
                &SystemOneFamily::JEV
            ),
            ComposeResult::Invalid
        );

        let mut nan_answer = full_answers(2.0, 0.9);
        nan_answer.insert(
            "domain_load".to_string(),
            dimension_answer(f64::NAN, 0.9),
        );
        assert_eq!(
            compose(
                &nan_answer,
                &DimensionWeights::default(),
                &trust(),
                &SystemOneFamily::JEV
            ),
            ComposeResult::Invalid
        );

        let mut out_of_range = full_answers(2.0, 0.9);
        out_of_range.insert(
            "domain_load".to_string(),
            dimension_answer(2.0, 1.5),
        );
        assert_eq!(
            compose(
                &out_of_range,
                &DimensionWeights::default(),
                &trust(),
                &SystemOneFamily::JEV
            ),
            ComposeResult::Invalid
        );

        let mut wrong_type = full_answers(2.0, 0.9);
        wrong_type.insert(
            "domain_load".to_string(),
            Answer::Choice {
                choice: "general".to_string(),
                probabilities: HashMap::new(),
                confidence: 0.9,
                answer_confidence: None,
            },
        );
        assert_eq!(
            compose(
                &wrong_type,
                &DimensionWeights::default(),
                &trust(),
                &SystemOneFamily::JEV
            ),
            ComposeResult::Invalid
        );
    }

    #[test]
    fn low_task_choice_confidence_keeps_general_placeholder() {
        let mut answers = full_answers(2.0, 0.9);
        answers.insert(
            TASK_TYPE_QUESTION_ID.to_string(),
            Answer::Choice {
                choice: "math_reasoning".to_string(),
                probabilities: HashMap::new(),
                confidence: 0.25,
                answer_confidence: None,
            },
        );
        let result = compose(
            &answers,
            &DimensionWeights::default(),
            &trust(),
            &SystemOneFamily::JEV,
        );
        let ComposeResult::Ok { task_type, .. } = result else {
            panic!("expected ok compose, got {result:?}");
        };
        assert_eq!(task_type, TaskType::General);
    }

    #[test]
    fn blend_returns_heuristic_at_zero_confidence_and_jev_at_threshold() {
        let heuristic = ComplexityScore::new(0.25);
        let systemone = ComplexityScore::new(0.75);
        let zero = blend_scores(systemone, heuristic, 0.0, 0.60);
        assert!((zero.value() - 0.25).abs() < 1.0e-9);
        let full = blend_scores(systemone, heuristic, 0.60, 0.60);
        assert!((full.value() - 0.75).abs() < 1.0e-9);
        let mid = blend_scores(systemone, heuristic, 0.30, 0.60);
        assert!((mid.value() - 0.5).abs() < 1.0e-9);
    }

    #[test]
    fn task_type_parsing_covers_all_options() {
        assert_eq!(parse_task_type("code_generation"), Some(TaskType::CodeGeneration));
        assert_eq!(parse_task_type("math_reasoning"), Some(TaskType::MathReasoning));
        assert_eq!(parse_task_type("creative_writing"), Some(TaskType::CreativeWriting));
        assert_eq!(parse_task_type("factual_qa"), Some(TaskType::FactualQA));
        assert_eq!(parse_task_type("tool_use"), Some(TaskType::ToolUse));
        assert_eq!(parse_task_type("summarization"), Some(TaskType::Summarization));
        assert_eq!(parse_task_type("general"), Some(TaskType::General));
        assert_eq!(parse_task_type("unknown_option"), None);
    }

    proptest! {
        #![proptest_config(ProptestConfig::with_cases(64))]

        #[test]
        fn property_composed_scores_stay_in_unit_interval(
            dimension_scores in prop::collection::vec(0.0f64..=3.0, 6),
            dimension_confidences in prop::collection::vec(0.0f64..=1.0, 6),
            weight_values in prop::collection::vec(0.0f64..=10.0, 6),
        ) {
            let mut answers = HashMap::new();
            for (index, id) in DIMENSION_QUESTION_IDS.iter().enumerate() {
                answers.insert(
                    id.to_string(),
                    dimension_answer(dimension_scores[index], dimension_confidences[index]),
                );
            }
            answers.insert(
                TASK_TYPE_QUESTION_ID.to_string(),
                Answer::Choice {
                    choice: "general".to_string(),
                    probabilities: HashMap::new(),
                    confidence: 0.9,
                    answer_confidence: None,
                },
            );

            let pairs = DimensionWeights {
                reasoning_depth: weight_values[0],
                tool_coupling: weight_values[1],
                context_synthesis: weight_values[2],
                output_precision: weight_values[3],
                domain_load: weight_values[4],
                ambiguity: weight_values[5],
            };

            match compose(&answers, &pairs, &trust(), &SystemOneFamily::JEV) {
                ComposeResult::Ok { score, .. } | ComposeResult::LowConfidence { score, .. } => {
                    prop_assert!((0.0..=1.0).contains(&score.value()));
                    prop_assert!(score.value().is_finite());
                }
                ComposeResult::Invalid => {
                    // All-zero weights are the only valid-input Invalid case.
                    prop_assert!(weight_values.iter().all(|weight| *weight == 0.0));
                }
            }
        }

        #[test]
        fn property_missing_answers_never_panic(
            drop_index in 0usize..6,
        ) {
            let mut answers = HashMap::new();
            for (index, id) in DIMENSION_QUESTION_IDS.iter().enumerate() {
                if index != drop_index {
                    answers.insert(id.to_string(), dimension_answer(2.0, 0.9));
                }
            }
            let result = compose(
                &answers,
                &DimensionWeights::default(),
                &trust(),
                &SystemOneFamily::JEV,
            );
            prop_assert_eq!(result, ComposeResult::Invalid);
        }
    }
}
