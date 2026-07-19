//! Reasoning-effort mapping between Chat Completions and Responses API semantics.
//!
//! The Chat Completions API accepts the extension values `minimal` and
//! `none` in `reasoning_effort` (both collapsing to `low` on the Codex
//! backend), the standard `low` / `medium` / `high` tier, and the
//! OBEY-specific `xhigh` extension that is only legal for
//! [`Xhigh_Models`][crate::codex::model_map::XHIGH_MODEL_PATTERNS]. The
//! Responses API `reasoning.effort` field accepts `{low, medium, high,
//! xhigh}` — which is the return-type domain enforced by construction
//! here (Req 9.7).
//!
//! See `.kiro/specs/codex-backend-translation/design.md` §6 for the
//! mapping table and `.kiro/specs/codex-backend-translation/requirements.md`
//! Requirement 9 for the authoritative acceptance criteria.

/// Map a Chat Completions `reasoning_effort` value onto a Responses API
/// `reasoning.effort` value.
///
/// Semantics:
/// * `None` → `"medium"` (Req 9.6).
/// * `Some("minimal")` or `Some("none")` → `"low"` (Req 9.1 / 9.2).
/// * `Some("low" | "medium" | "high")` → forwarded unchanged (Req 9.3).
/// * `Some("xhigh")` → `"xhigh"` when
///   [`is_xhigh`][super::model_map::is_xhigh]`(resolved_model, allowlist)`
///   holds (Req 9.4); otherwise downgraded to `"high"` (Req 9.5).
/// * Any other `Some(_)` (unknown value, empty string, wrong case,
///   whitespace variants) → `"medium"` as the safe default.
///
/// The return value is always one of `{"low", "medium", "high", "xhigh"}`
/// and `"xhigh"` is gated on the resolved-model heuristic, which
/// together satisfy Req 9.7 by construction.
pub fn map_effort(
    req_effort: Option<&str>,
    resolved_model: &str,
    xhigh_allowlist: &[String],
) -> &'static str {
    match req_effort {
        None => "medium",                        // Req 9.6
        Some("minimal") | Some("none") => "low", // Req 9.1 / 9.2
        Some("low") => "low",
        Some("medium") => "medium",
        Some("high") => "high", // Req 9.3
        Some("xhigh") if super::model_map::is_xhigh(resolved_model, xhigh_allowlist) => {
            "xhigh" // Req 9.4
        }
        Some("xhigh") => "high", // Req 9.5
        Some(_) => "medium",     // unknown → safe default
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    // ---- Req 9.1 — "minimal" → "low" ----
    #[test]
    fn minimal_maps_to_low() {
        assert_eq!(map_effort(Some("minimal"), "gpt-5.1", &[]), "low");
        assert_eq!(map_effort(Some("minimal"), "gpt-5.5", &[]), "low");
    }

    // ---- Req 9.2 — "none" → "low" ----
    #[test]
    fn none_string_maps_to_low() {
        assert_eq!(map_effort(Some("none"), "gpt-5.1", &[]), "low");
        assert_eq!(map_effort(Some("none"), "gpt-5.5-codex", &[]), "low");
    }

    // ---- Req 9.3 — {low, medium, high} pass through unchanged ----
    #[test]
    fn standard_tiers_pass_through_unchanged() {
        assert_eq!(map_effort(Some("low"), "gpt-5.1", &[]), "low");
        assert_eq!(map_effort(Some("medium"), "gpt-5.1", &[]), "medium");
        assert_eq!(map_effort(Some("high"), "gpt-5.1", &[]), "high");
        // The resolved model and allowlist do not affect these arms.
        assert_eq!(map_effort(Some("high"), "gpt-5.5", &[]), "high");
    }

    // ---- Req 9.4 — "xhigh" on an Xhigh_Models match → "xhigh" ----
    #[test]
    fn xhigh_on_xhigh_capable_model_returns_xhigh() {
        // Pattern-based membership.
        assert_eq!(map_effort(Some("xhigh"), "gpt-5.5", &[]), "xhigh");
        assert_eq!(map_effort(Some("xhigh"), "gpt-5.2-codex", &[]), "xhigh");
        assert_eq!(map_effort(Some("xhigh"), "gpt-5.1-codex-max", &[]), "xhigh");
        assert_eq!(map_effort(Some("xhigh"), "gpt-5.4", &[]), "xhigh");
    }

    #[test]
    fn xhigh_on_allowlist_only_model_returns_xhigh() {
        let allow = vec!["custom-xhigh-model".to_string()];
        assert_eq!(
            map_effort(Some("xhigh"), "custom-xhigh-model", &allow),
            "xhigh"
        );
    }

    // ---- Req 9.5 — "xhigh" on a non-Xhigh model downgrades to "high" ----
    #[test]
    fn xhigh_on_non_xhigh_model_downgrades_to_high() {
        assert_eq!(map_effort(Some("xhigh"), "gpt-5.1", &[]), "high");
        assert_eq!(map_effort(Some("xhigh"), "codex-mini-latest", &[]), "high");
        assert_eq!(map_effort(Some("xhigh"), "gpt-4o", &[]), "high");
        // Allowlist miss — exact-match is required on the allowlist path.
        let allow = vec!["custom-xhigh-model".to_string()];
        assert_eq!(
            map_effort(Some("xhigh"), "custom-xhigh-model-v2", &allow),
            "high"
        );
    }

    // ---- Req 9.6 — missing `reasoning_effort` → "medium" ----
    #[test]
    fn missing_reasoning_effort_maps_to_medium() {
        assert_eq!(map_effort(None, "gpt-5.1", &[]), "medium");
        assert_eq!(map_effort(None, "gpt-5.5", &[]), "medium");
        assert_eq!(map_effort(None, "", &[]), "medium");
    }

    // ---- Req 9.7 invariant — unknown values fall back to "medium" ----
    #[test]
    fn unknown_values_fall_back_to_medium() {
        // Empty string, wrong case, unknown token, trailing whitespace —
        // all are outside the documented vocabulary and must not leak
        // through. The return value must remain in
        // {low, medium, high, xhigh}, and specifically "medium" for the
        // unknown arm per the design document's mapping table.
        for bad in [
            "",        // empty string
            "Low",     // wrong case
            "LOW",     // wrong case
            "HIGHEST", // not in the vocabulary
            "xhigh ",  // trailing whitespace
            " xhigh",  // leading whitespace
            "x-high",  // hyphen variant
            "very-high",
            "ultra",
        ] {
            let got = map_effort(Some(bad), "gpt-5.5", &[]);
            assert_eq!(
                got, "medium",
                "unknown effort value {bad:?} should fall back to \"medium\""
            );
        }
    }

    // ---- Req 9.7 invariant — return value is always in the documented set ----
    #[test]
    fn return_value_is_always_in_allowed_set() {
        let allowed = ["low", "medium", "high", "xhigh"];
        let efforts: [Option<&str>; 9] = [
            None,
            Some("minimal"),
            Some("none"),
            Some("low"),
            Some("medium"),
            Some("high"),
            Some("xhigh"),
            Some(""),
            Some("garbage"),
        ];
        let models = [
            "gpt-5.1",
            "gpt-5.5",
            "gpt-5.5-codex",
            "gpt-5.2-codex",
            "gpt-5.1-codex-max",
            "codex-mini-latest",
            "xyz-unknown",
            "",
        ];
        for e in efforts {
            for m in models {
                let got = map_effort(e, m, &[]);
                assert!(
                    allowed.contains(&got),
                    "map_effort({e:?}, {m:?}, &[]) = {got:?} is not in {allowed:?}"
                );
                if got == "xhigh" {
                    assert!(
                        super::super::model_map::is_xhigh(m, &[]),
                        "xhigh returned for non-xhigh model {m:?}"
                    );
                }
            }
        }
    }
}

#[cfg(test)]
mod property_tests {
    //! Property-based tests for [`map_effort`].
    //!
    //! Validates: Requirements 13.3

    use super::*;
    use proptest::prelude::*;

    /// Strategy producing every `reasoning_effort` value in the documented
    /// vocabulary plus the "missing" case (`None`).
    fn effort_strategy() -> impl Strategy<Value = Option<&'static str>> {
        prop_oneof![
            Just(None),
            Just(Some("none")),
            Just(Some("minimal")),
            Just(Some("low")),
            Just(Some("medium")),
            Just(Some("high")),
            Just(Some("xhigh")),
        ]
    }

    /// Strategy drawing from a synthesized set of resolved model names
    /// covering both `is_xhigh`-positive and `is_xhigh`-negative rows.
    fn model_strategy() -> impl Strategy<Value = &'static str> {
        prop::sample::select(vec![
            "gpt-5.1",
            "gpt-5.2",
            "gpt-5.4",
            "gpt-5.5",
            "gpt-5.5-codex",
            "gpt-5.2-codex",
            "gpt-5.1-codex",
            "gpt-5.1-codex-max",
            "codex-mini-latest",
            "gpt-6.0-hypothetical",
            "xyz-unknown",
        ])
    }

    proptest! {
        /// **Property 3 — Effort Mapping Invariant**
        ///
        /// For all `(reasoning_effort, resolved_model)` drawn from the
        /// documented vocabulary and synthesized model set:
        ///
        /// 1. `map_effort(effort, model, &[])` always returns a value in
        ///    `{"low", "medium", "high", "xhigh"}`.
        /// 2. `"xhigh"` is only returned when
        ///    `is_xhigh(model, &[])` holds.
        ///
        /// Validates: Requirements 13.3
        #[test]
        fn effort_mapping_invariant(
            effort in effort_strategy(),
            model in model_strategy(),
        ) {
            let got = map_effort(effort, model, &[]);
            prop_assert!(
                matches!(got, "low" | "medium" | "high" | "xhigh"),
                "map_effort({:?}, {:?}, &[]) = {:?} is outside the allowed set",
                effort, model, got
            );
            if got == "xhigh" {
                prop_assert!(
                    super::super::model_map::is_xhigh(model, &[]),
                    "xhigh returned for non-xhigh model {:?}",
                    model
                );
            }
        }
    }
}
