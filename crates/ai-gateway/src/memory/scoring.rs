//! Retrieval scoring and relevance decay helpers.

use chrono::{DateTime, Utc};

use super::MemoryType;

const DAYS_TO_RECENCY_WEIGHT: f64 = 0.1;
const SECONDS_PER_DAY: f64 = 86_400.0;
const CONTEXT_SCOPE_BOOST: f64 = 1.5;

/// Compute a retrieval score from rank, relevance, recency, and scope.
///
/// Negative and non-finite rank values contribute no score. Relevance is
/// clamped to `[0.0, 1.0]`, and an overflowing composition saturates at the
/// largest finite `f64`.
pub fn compute_score(
    fts5_rank: f64,
    relevance_score: f64,
    last_accessed_at: DateTime<Utc>,
    now: DateTime<Utc>,
    is_context_scoped: bool,
) -> f64 {
    let rank = finite_non_negative(fts5_rank);
    let relevance = finite_relevance(relevance_score);
    let scope_boost = if is_context_scoped {
        CONTEXT_SCOPE_BOOST
    } else {
        1.0
    };
    let score = rank * relevance * recency_boost(last_accessed_at, now) * scope_boost;

    if score.is_finite() {
        score
    } else if score.is_sign_positive() {
        f64::MAX
    } else {
        0.0
    }
}

/// Return the recency multiplier for a last-access timestamp.
///
/// Sub-day precision is retained. Future timestamps are treated as current,
/// so the multiplier always remains finite and in `(0.0, 1.0]`.
pub fn recency_boost(last_accessed_at: DateTime<Utc>, now: DateTime<Utc>) -> f64 {
    let elapsed_seconds = now
        .signed_duration_since(last_accessed_at)
        .num_milliseconds() as f64
        / 1_000.0;
    let days_since_access = (elapsed_seconds / SECONDS_PER_DAY).max(0.0);

    1.0 / (1.0 + days_since_access * DAYS_TO_RECENCY_WEIGHT)
}

/// Apply the type-specific decay multiplier to a relevance score.
///
/// Invalid and negative scores become `0.0`; positive scores are clamped to
/// the relevance range before decay, keeping the result finite in `[0.0, 1.0]`.
pub fn apply_decay(relevance_score: f64, memory_type: MemoryType) -> f64 {
    let multiplier = match memory_type {
        MemoryType::Preference => 0.99,
        MemoryType::Fact => 0.95,
        MemoryType::Context => 0.85,
        MemoryType::Decision => 0.98,
    };

    finite_relevance(relevance_score) * multiplier
}

fn finite_non_negative(value: f64) -> f64 {
    if value.is_finite() {
        value.max(0.0)
    } else {
        0.0
    }
}

fn finite_relevance(value: f64) -> f64 {
    if value.is_finite() {
        value.clamp(0.0, 1.0)
    } else if value == f64::INFINITY {
        1.0
    } else {
        0.0
    }
}

#[cfg(test)]
mod tests {
    use chrono::{Duration, TimeZone};
    use proptest::prelude::*;

    use super::*;

    fn timestamp() -> DateTime<Utc> {
        Utc.with_ymd_and_hms(2026, 7, 31, 12, 0, 0)
            .single()
            .expect("fixed test timestamp must be valid")
    }

    #[test]
    fn recency_uses_fractional_days() {
        let now = timestamp();
        let twelve_hours_ago = now - Duration::hours(12);

        assert!((recency_boost(twelve_hours_ago, now) - (1.0 / 1.05)).abs() < 1e-12);
    }

    #[test]
    fn future_access_never_boosts_above_one() {
        let now = timestamp();

        assert_eq!(recency_boost(now + Duration::days(30), now), 1.0);
        assert_eq!(recency_boost(now, now), 1.0);
    }

    #[test]
    fn context_scope_multiplier_is_exactly_one_point_five() {
        let now = timestamp();
        let user_score = compute_score(0.8, 0.75, now, now, false);
        let context_score = compute_score(0.8, 0.75, now, now, true);

        assert_eq!(context_score, user_score * 1.5);
    }

    #[test]
    fn memory_type_decay_multipliers_are_exact() {
        let cases = [
            (MemoryType::Preference, 0.99),
            (MemoryType::Fact, 0.95),
            (MemoryType::Context, 0.85),
            (MemoryType::Decision, 0.98),
        ];

        for (memory_type, expected) in cases {
            assert_eq!(apply_decay(1.0, memory_type), expected);
        }
    }

    #[test]
    fn public_helpers_clamp_invalid_inputs_to_finite_ranges() {
        let now = timestamp();
        let invalid_values = [f64::NAN, f64::NEG_INFINITY, -1.0];

        for value in invalid_values {
            assert_eq!(compute_score(value, 1.0, now, now, false), 0.0);
            assert_eq!(apply_decay(value, MemoryType::Preference), 0.0);
        }
        assert_eq!(apply_decay(f64::INFINITY, MemoryType::Fact), 0.95);
        assert!(compute_score(f64::MAX, 1.0, now, now, true).is_finite());
    }

    proptest! {
        #![proptest_config(ProptestConfig::with_cases(64))]

        #[test]
        fn score_composes_formula_for_valid_inputs(
            rank in 0.0f64..1_000_000.0,
            relevance in 0.0f64..=1.0,
            elapsed_millis in 0i64..31_536_000_000i64,
            is_context_scoped in any::<bool>(),
        ) {
            let now = timestamp();
            let last_accessed_at = now - Duration::milliseconds(elapsed_millis);
            let days = elapsed_millis as f64 / 1_000.0 / SECONDS_PER_DAY;
            let expected_recency = 1.0 / (1.0 + days * 0.1);
            let expected_scope = if is_context_scoped { 1.5 } else { 1.0 };
            let expected = rank * relevance * expected_recency * expected_scope;
            let actual = compute_score(
                rank,
                relevance,
                last_accessed_at,
                now,
                is_context_scoped,
            );

            prop_assert!(actual.is_finite());
            prop_assert!(actual >= 0.0);
            prop_assert!((actual - expected).abs() <= expected.abs().max(1.0) * 1e-12);
        }

        #[test]
        fn decay_matches_each_type_and_stays_clamped(
            relevance in -10.0f64..10.0,
            memory_type_index in 0usize..4,
        ) {
            let (memory_type, multiplier) = [
                (MemoryType::Preference, 0.99),
                (MemoryType::Fact, 0.95),
                (MemoryType::Context, 0.85),
                (MemoryType::Decision, 0.98),
            ][memory_type_index];
            let actual = apply_decay(relevance, memory_type);
            let expected = relevance.clamp(0.0, 1.0) * multiplier;

            prop_assert!(actual.is_finite());
            prop_assert!((0.0..=1.0).contains(&actual));
            prop_assert_eq!(actual, expected);
        }
    }
}
