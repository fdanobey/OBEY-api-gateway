//! Model resolution and reasoning/xhigh classification helpers.
//!
//! Per the `codex-backend-translation` spec, the gateway does not maintain
//! a hard-coded allow-list of Codex-supported models. Instead:
//!
//! * [`resolve_model`] is a pass-through that forwards the client-supplied
//!   model value unchanged unless the operator configured a
//!   `codex_model_override` on the provider (Req 8.1, 8.2, 8.3, 8.6).
//! * [`is_xhigh`] and [`is_reasoning`] are heuristic substring tests against
//!   [`XHIGH_MODEL_PATTERNS`] and [`REASONING_MODEL_PATTERNS`] respectively,
//!   augmented by operator-configured allowlists for new model releases
//!   whose names do not match the built-in patterns (Req 8.4, 10.9, 10.10).
//!
//! The heuristics are deliberately loose so new OpenAI model releases
//! (e.g. `gpt-5.5`, future `gpt-6.*`) can be dispatched without a code
//! change. Operators extend membership via the `xhigh_models_allowlist` and
//! `reasoning_models_allowlist` fields in gateway config.

/// Heuristic set used by Req 8.4 / Req 9.4 to decide whether a resolved
/// outgoing model accepts `xhigh` reasoning effort.
///
/// Membership is substring-based: a model string matches if any entry in
/// this slice is a substring of the model name. The order is the insertion
/// order documented in the spec and is preserved verbatim.
pub const XHIGH_MODEL_PATTERNS: &[&str] = &[
    "gpt-5.2",
    "gpt-5.1-codex-max",
    "gpt-5.4",
    "gpt-5.5",
    "gpt-5.2-codex",
    "gpt-5.5-codex",
];

/// Heuristic set used by Req 2.9 / Req 8.4 to decide whether to emit the
/// `reasoning` field at all on the outgoing Responses_Request.
///
/// Membership is substring-based. The order is the insertion order
/// documented in the spec and is preserved verbatim.
pub const REASONING_MODEL_PATTERNS: &[&str] = &["codex", "gpt-5", "gpt-6", "o1", "o3", "o4"];

/// Forward the client-supplied model unchanged unless a
/// `codex_model_override` is configured on the provider.
///
/// The lifetime parameter `'a` ties the returned reference to the longer of
/// the two input lifetimes so callers can use the result without cloning.
///
/// _Requirements: 8.1, 8.2, 8.3, 8.6._
#[inline]
pub fn resolve_model<'a>(client_model: &'a str, override_: Option<&'a str>) -> &'a str {
    override_.unwrap_or(client_model)
}

/// Returns `true` when `model` is considered Xhigh-capable.
///
/// A model is Xhigh-capable when either:
/// * any pattern in [`XHIGH_MODEL_PATTERNS`] is a substring of `model`, OR
/// * `model` exactly matches any entry in the operator-provided
///   `allowlist`.
///
/// _Requirements: 8.4, 10.9, 10.10._
pub fn is_xhigh(model: &str, allowlist: &[String]) -> bool {
    XHIGH_MODEL_PATTERNS.iter().any(|p| model.contains(p)) || allowlist.iter().any(|a| a == model)
}

/// Returns `true` when `model` should receive a `reasoning` field on the
/// outgoing Responses_Request.
///
/// A model is reasoning-capable when either:
/// * any pattern in [`REASONING_MODEL_PATTERNS`] is a substring of `model`,
///   OR
/// * `model` exactly matches any entry in the operator-provided
///   `allowlist`.
///
/// _Requirements: 2.9, 8.4, 10.9, 10.10._
pub fn is_reasoning(model: &str, allowlist: &[String]) -> bool {
    REASONING_MODEL_PATTERNS.iter().any(|p| model.contains(p))
        || allowlist.iter().any(|a| a == model)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn resolve_model_passes_through_when_no_override() {
        assert_eq!(resolve_model("gpt-5.1", None), "gpt-5.1");
        assert_eq!(resolve_model("gpt-5.5-codex", None), "gpt-5.5-codex");
        assert_eq!(resolve_model("", None), "");
    }

    #[test]
    fn resolve_model_uses_override_when_present() {
        assert_eq!(
            resolve_model("gpt-5.1", Some("codex-mini-latest")),
            "codex-mini-latest"
        );
        // Empty-string override is honoured verbatim — validation of
        // non-emptiness is the config layer's responsibility (Req 10.9).
        assert_eq!(resolve_model("gpt-5.1", Some("")), "");
    }

    #[test]
    fn is_xhigh_matches_known_xhigh_strings() {
        assert!(is_xhigh("gpt-5.5", &[]));
        assert!(is_xhigh("gpt-5.2-codex", &[]));
        assert!(is_xhigh("gpt-5.1-codex-max", &[]));
        assert!(is_xhigh("gpt-5.4", &[]));
        // Substring match — a longer model name that contains a pattern
        // should still match (e.g. vendor suffixes).
        assert!(is_xhigh("openai/gpt-5.5-preview", &[]));
    }

    #[test]
    fn is_xhigh_rejects_non_xhigh_models() {
        assert!(!is_xhigh("gpt-5.1", &[]));
        assert!(!is_xhigh("gpt-4o", &[]));
        assert!(!is_xhigh("codex-mini-latest", &[]));
        assert!(!is_xhigh("o3-mini", &[]));
        assert!(!is_xhigh("", &[]));
    }

    #[test]
    fn is_xhigh_honours_allowlist_only_entries() {
        let allow = vec!["custom-xhigh-model".to_string()];
        assert!(is_xhigh("custom-xhigh-model", &allow));
        // Allowlist is exact-match, not substring.
        assert!(!is_xhigh("custom-xhigh-model-v2", &allow));
        assert!(!is_xhigh("custom-xhigh", &allow));
    }

    #[test]
    fn is_reasoning_matches_gpt_5_family() {
        assert!(is_reasoning("gpt-5.1", &[]));
        assert!(is_reasoning("gpt-5.5", &[]));
        assert!(is_reasoning("gpt-5", &[]));
    }

    #[test]
    fn is_reasoning_matches_codex_family() {
        assert!(is_reasoning("codex-mini-latest", &[]));
        assert!(is_reasoning("gpt-5.2-codex", &[]));
    }

    #[test]
    fn is_reasoning_matches_o_series_and_gpt_6() {
        assert!(is_reasoning("o1", &[]));
        assert!(is_reasoning("o3-mini", &[]));
        assert!(is_reasoning("o4-preview", &[]));
        assert!(is_reasoning("gpt-6.0-hypothetical", &[]));
    }

    #[test]
    fn is_reasoning_rejects_unknown_models() {
        assert!(!is_reasoning("gpt-4o", &[]));
        assert!(!is_reasoning("gpt-4-turbo", &[]));
        assert!(!is_reasoning("claude-3-opus", &[]));
        assert!(!is_reasoning("xyz-unknown", &[]));
        assert!(!is_reasoning("", &[]));
    }

    #[test]
    fn is_reasoning_honours_allowlist_only_entries() {
        let allow = vec!["future-reasoning-model".to_string()];
        assert!(is_reasoning("future-reasoning-model", &allow));
        assert!(!is_reasoning("future-reasoning-model-v2", &allow));
        assert!(!is_reasoning("gpt-4o", &allow));
    }

    #[test]
    fn pattern_arrays_preserve_documented_order() {
        // The spec pins these orderings; a re-sort would be a silent
        // semantic change.
        assert_eq!(
            XHIGH_MODEL_PATTERNS,
            &[
                "gpt-5.2",
                "gpt-5.1-codex-max",
                "gpt-5.4",
                "gpt-5.5",
                "gpt-5.2-codex",
                "gpt-5.5-codex",
            ]
        );
        assert_eq!(
            REASONING_MODEL_PATTERNS,
            &["codex", "gpt-5", "gpt-6", "o1", "o3", "o4"]
        );
    }
}
