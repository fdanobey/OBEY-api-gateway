//! Token-capacity filtering for smart-routing candidates.

use crate::config::ProviderModel;

/// Token counts reserved around a request before selecting a routing tier.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct ContextRequirement {
    pub input_tokens: u64,
    pub reserved_output_tokens: u64,
    pub provider_overhead_tokens: u64,
    pub safety_margin_tokens: u64,
}

impl ContextRequirement {
    /// Return the total required capacity, saturating if any addition overflows.
    pub fn estimated_tokens(self) -> u64 {
        self.input_tokens
            .checked_add(self.reserved_output_tokens)
            .and_then(|total| total.checked_add(self.provider_overhead_tokens))
            .and_then(|total| total.checked_add(self.safety_margin_tokens))
            .unwrap_or(u64::MAX)
    }
}

/// Successful capacity filtering performed before tier selection.
#[derive(Debug, Clone, PartialEq)]
pub struct EligibleModels {
    pub models: Vec<ProviderModel>,
    pub excluded_count: usize,
    pub largest_known_context: Option<u32>,
    pub estimated_requirement: u64,
}

/// Structured capacity failure that callers can map to HTTP 413.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct NoSafeCandidate {
    pub excluded_count: usize,
    pub largest_known_context: Option<u32>,
    pub estimated_requirement: u64,
}

/// Result of filtering candidates by their declared context capacity.
#[derive(Debug, Clone, PartialEq)]
pub enum ContextFilterResult {
    /// At least one candidate is safe under the selected unknown-capacity policy.
    Eligible(EligibleModels),
    /// The caller supplied no candidates to filter.
    NoCandidates {
        excluded_count: usize,
        largest_known_context: Option<u32>,
        estimated_requirement: u64,
    },
    /// Candidates were supplied, but every candidate was excluded as unsafe.
    NoSafeCandidate(NoSafeCandidate),
}

/// Filter models by token capacity before any tier selection occurs.
///
/// A `context_window` of zero is an unknown capacity and is retained only when
/// `allow_unknown_context_window` is explicitly enabled. Known capacities below
/// the estimated requirement are always excluded from the returned clone set.
pub fn filter_by_context_capacity(
    candidates: &[ProviderModel],
    requirement: ContextRequirement,
    allow_unknown_context_window: bool,
) -> ContextFilterResult {
    let estimated_requirement = requirement.estimated_tokens();
    let largest_known_context = candidates
        .iter()
        .filter_map(|candidate| (candidate.context_window > 0).then_some(candidate.context_window))
        .max();

    if candidates.is_empty() {
        return ContextFilterResult::NoCandidates {
            excluded_count: 0,
            largest_known_context,
            estimated_requirement,
        };
    }

    let models: Vec<_> = candidates
        .iter()
        .filter(|candidate| {
            if candidate.context_window == 0 {
                allow_unknown_context_window
            } else {
                u64::from(candidate.context_window) >= estimated_requirement
            }
        })
        .cloned()
        .collect();
    let excluded_count = candidates.len().saturating_sub(models.len());

    if models.is_empty() {
        ContextFilterResult::NoSafeCandidate(NoSafeCandidate {
            excluded_count,
            largest_known_context,
            estimated_requirement,
        })
    } else {
        ContextFilterResult::Eligible(EligibleModels {
            models,
            excluded_count,
            largest_known_context,
            estimated_requirement,
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::smart_routing::tier::{SmartRoutingTier, TaskType};
    use proptest::prelude::*;

    fn model(name: &str, context_window: u32, tier: SmartRoutingTier) -> ProviderModel {
        ProviderModel {
            provider: "test".to_string(),
            model: name.to_string(),
            cost_per_million_input_tokens: 0.0,
            cost_per_million_output_tokens: 0.0,
            priority: 100,
            structured_output_passthrough: None,
            tier: Some(tier),
            context_window,
            specializations: Vec::<TaskType>::new(),
        }
    }

    fn requirement(
        input_tokens: u64,
        reserved_output_tokens: u64,
        provider_overhead_tokens: u64,
        safety_margin_tokens: u64,
    ) -> ContextRequirement {
        ContextRequirement {
            input_tokens,
            reserved_output_tokens,
            provider_overhead_tokens,
            safety_margin_tokens,
        }
    }

    #[test]
    fn exact_fit_is_safe() {
        let candidates = [model("exact", 1_000, SmartRoutingTier::Fast)];

        let result = filter_by_context_capacity(&candidates, requirement(700, 200, 50, 50), false);

        let ContextFilterResult::Eligible(filtered) = result else {
            panic!("exact-fit candidate should be eligible");
        };
        assert_eq!(filtered.models, candidates);
        assert_eq!(filtered.excluded_count, 0);
        assert_eq!(filtered.largest_known_context, Some(1_000));
        assert_eq!(filtered.estimated_requirement, 1_000);
    }

    #[test]
    fn overflow_saturates_requirement_and_excludes_known_capacity() {
        let candidates = [model("known", u32::MAX, SmartRoutingTier::Powerful)];

        let result =
            filter_by_context_capacity(&candidates, requirement(u64::MAX - 1, 1, 1, 1), false);

        assert_eq!(
            result,
            ContextFilterResult::NoSafeCandidate(NoSafeCandidate {
                excluded_count: 1,
                largest_known_context: Some(u32::MAX),
                estimated_requirement: u64::MAX,
            })
        );
    }

    #[test]
    fn all_unsafe_returns_structured_failure() {
        let candidates = [
            model("fast", 1_000, SmartRoutingTier::Fast),
            model("balanced", 2_000, SmartRoutingTier::Balanced),
        ];

        let result =
            filter_by_context_capacity(&candidates, requirement(1_900, 100, 50, 50), false);

        assert_eq!(
            result,
            ContextFilterResult::NoSafeCandidate(NoSafeCandidate {
                excluded_count: 2,
                largest_known_context: Some(2_000),
                estimated_requirement: 2_100,
            })
        );
    }

    #[test]
    fn empty_input_is_distinct_from_all_unsafe() {
        let result = filter_by_context_capacity(&[], requirement(10, 20, 30, 40), false);

        assert_eq!(
            result,
            ContextFilterResult::NoCandidates {
                excluded_count: 0,
                largest_known_context: None,
                estimated_requirement: 100,
            }
        );
    }

    #[test]
    fn unknown_capacity_requires_explicit_allow_policy() {
        let candidates = [model("unknown", 0, SmartRoutingTier::Fast)];
        let required = requirement(100, 100, 10, 10);

        assert!(matches!(
            filter_by_context_capacity(&candidates, required, false),
            ContextFilterResult::NoSafeCandidate(NoSafeCandidate {
                excluded_count: 1,
                largest_known_context: None,
                estimated_requirement: 220,
            })
        ));

        let ContextFilterResult::Eligible(filtered) =
            filter_by_context_capacity(&candidates, required, true)
        else {
            panic!("explicit policy should retain unknown capacity");
        };
        assert_eq!(filtered.models, candidates);
        assert_eq!(filtered.excluded_count, 0);
        assert_eq!(filtered.largest_known_context, None);
    }

    #[test]
    fn unsafe_adjacent_tier_cannot_be_reintroduced() {
        let candidates = [
            model("fast-safe", 4_000, SmartRoutingTier::Fast),
            model("balanced-unsafe", 1_000, SmartRoutingTier::Balanced),
            model("powerful-safe", 8_000, SmartRoutingTier::Powerful),
        ];

        let ContextFilterResult::Eligible(filtered) =
            filter_by_context_capacity(&candidates, requirement(1_500, 500, 100, 100), false)
        else {
            panic!("safe candidates should remain");
        };

        assert_eq!(filtered.excluded_count, 1);
        assert_eq!(filtered.largest_known_context, Some(8_000));
        assert_eq!(
            filtered
                .models
                .iter()
                .map(|candidate| candidate.model.as_str())
                .collect::<Vec<_>>(),
            vec!["fast-safe", "powerful-safe"]
        );
    }

    proptest! {
    #![proptest_config(ProptestConfig::with_cases(256))]

    #[test]
    fn property_31_known_insufficient_windows_are_excluded(
    (required_window, insufficient_windows) in (2u32..=u32::MAX, 1usize..16).prop_flat_map(
    |(required_window, count)| {
    prop::collection::vec(1u32..required_window, count)
    .prop_map(move |windows| (required_window, windows))
    },
    ),
    safe_offsets in prop::collection::vec(0u32..=1_000_000, 1..16),
    ) {
    let mut candidates = insufficient_windows
    .iter()
    .enumerate()
    .map(|(index, &context_window)| {
    model(
    &format!("insufficient-{index}"),
    context_window,
    SmartRoutingTier::Fast,
    )
    })
    .collect::<Vec<_>>();
    let safe_models = safe_offsets
    .iter()
    .enumerate()
    .map(|(index, &offset)| {
    model(
    &format!("safe-{index}"),
    required_window.saturating_add(offset),
    SmartRoutingTier::Balanced,
    )
    })
    .collect::<Vec<_>>();
    candidates.extend(safe_models.iter().cloned());

    let result = filter_by_context_capacity(
    &candidates,
    requirement(u64::from(required_window), 0, 0, 0),
    false,
    );

    let ContextFilterResult::Eligible(filtered) = result else {
    prop_assert!(false, "known sufficient candidates must remain eligible");
    return Ok(());
    };
    prop_assert_eq!(filtered.excluded_count, insufficient_windows.len());
    prop_assert_eq!(&filtered.models, &safe_models);
    prop_assert!(filtered
    .models
    .iter()
    .all(|candidate| candidate.context_window >= required_window));
    prop_assert_eq!(
    filtered.largest_known_context,
    candidates.iter().map(|candidate| candidate.context_window).max()
    );
    prop_assert_eq!(filtered.estimated_requirement, u64::from(required_window));
    }

    #[test]
    fn property_32_oversized_requests_return_no_safe_structured_result(
    context_windows in prop::collection::vec(1u32..=u32::MAX, 1..32),
    required in prop_oneof![
    (u64::from(u32::MAX) + 1..=u64::from(u32::MAX) + 1_000_000)
    .prop_map(|input_tokens| requirement(input_tokens, 0, 0, 0)),
    (0u64..=1_000_000).prop_map(|overflow_delta| {
    requirement(u64::MAX - overflow_delta, overflow_delta + 1, 1, 1)
    }),
    ],
    ) {
    let candidates = context_windows
    .iter()
    .enumerate()
    .map(|(index, &context_window)| {
    model(
    &format!("known-{index}"),
    context_window,
    SmartRoutingTier::Powerful,
    )
    })
    .collect::<Vec<_>>();
    let estimated_requirement = required.estimated_tokens();

    let result = filter_by_context_capacity(&candidates, required, false);

    prop_assert_eq!(
    result,
    ContextFilterResult::NoSafeCandidate(NoSafeCandidate {
    excluded_count: candidates.len(),
    largest_known_context: context_windows.iter().copied().max(),
    estimated_requirement,
    })
    );
    }
    }
}
