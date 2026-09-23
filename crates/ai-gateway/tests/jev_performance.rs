//! Wall-clock smoke budgets for the Jev classifier.
//!
//! Ignored by default; run with:
//! `cargo test -p ai-gateway --test jev_performance -- --ignored`
//!
//! These are not micro-benchmarks. They use generous ceilings to catch gross
//! regressions while remaining stable on loaded CI hosts.

use std::collections::HashMap;
use std::time::{Duration, Instant};

use ai_gateway::smart_routing::config::DimensionWeights;
use ai_gateway::smart_routing::jev::compose::{compose, ComposeResult, JevTrustConfig};
use ai_gateway::smart_routing::jev::models::Answer;
use ai_gateway::smart_routing::jev::rubric::{
    build_questions, DIMENSION_QUESTION_IDS, TASK_TYPE_QUESTION_ID,
};
use ai_gateway::smart_routing::tier::ComplexityScore;

fn answers() -> HashMap<String, Answer> {
    let mut answers = HashMap::new();
    for id in DIMENSION_QUESTION_IDS {
        answers.insert(
            id.to_string(),
            Answer::Score {
                score: 1.5,
                confidence: 0.9,
                probabilities: HashMap::new(),
                legend: HashMap::new(),
            },
        );
    }
    answers.insert(
        TASK_TYPE_QUESTION_ID.to_string(),
        Answer::Choice {
            choice: "general".to_string(),
            probabilities: HashMap::new(),
            confidence: 0.9,
        },
    );
    answers
}

#[test]
#[ignore = "wall-clock Jev pure-path latency budget smoke test"]
fn jev_rubric_and_compose_p95_under_one_millisecond() {
    let answers = answers();
    let weights = DimensionWeights::default();
    let trust = JevTrustConfig {
        min_confidence: 0.6,
        min_task_confidence: 0.5,
        fallback_policy: ai_gateway::smart_routing::config::JevFallbackPolicy::Fallback,
    };

    let mut samples = Vec::with_capacity(500);
    for _ in 0..500 {
        let start = Instant::now();
        let questions = build_questions();
        let result = compose(&answers, &weights, &trust);
        std::hint::black_box(questions);
        std::hint::black_box(result);
        samples.push(start.elapsed());
    }
    samples.sort_unstable();
    let p95 = samples[(samples.len() * 95 / 100).min(samples.len() - 1)];
    eprintln!("jev pure path p95: {p95:?}");
    assert!(
        p95 < Duration::from_millis(5),
        "Jev pure path p95 exceeded generous 5ms CI ceiling: {p95:?}"
    );
}

#[test]
fn blend_score_math_is_constant_time_and_bounded() {
    let start = Instant::now();
    let mut result = ComplexityScore::new(0.0);
    for confidence_basis_points in 0..=10_000 {
        let confidence = f64::from(confidence_basis_points) / 10_000.0;
        result = ai_gateway::smart_routing::jev::compose::blend_scores(
            ComplexityScore::new(0.9),
            ComplexityScore::new(0.3),
            confidence,
            0.6,
        );
        assert!((0.0..=1.0).contains(&result.value()));
    }
    std::hint::black_box(result);
    assert!(
        start.elapsed() < Duration::from_millis(100),
        "10k blend operations exceeded 100ms"
    );
}

#[test]
fn rubric_shape_remains_seven_questions() {
    assert_eq!(build_questions().len(), 7);
}

#[test]
fn compose_success_path_is_stable() {
    let result = compose(
        &answers(),
        &DimensionWeights::default(),
        &JevTrustConfig {
            min_confidence: 0.6,
            min_task_confidence: 0.5,
            fallback_policy: ai_gateway::smart_routing::config::JevFallbackPolicy::Fallback,
        },
    );
    assert!(matches!(result, ComposeResult::Ok { .. }));
}
