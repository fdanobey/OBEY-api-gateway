//! Output-cap clamp core for the response-output-caps feature.
//!
//! Pure resolution and application of per-key / per-group output-token
//! ceilings at the handler seam, before routing. Ceiling-only: declared
//! carriers are clamped to `min(value, cap)`, never raised; when no
//! carrier is declared the typed `max_tokens` field is injected with the
//! effective cap.

use crate::models::openai::OpenAIRequest;

/// Result of applying an output cap to a request.
///
/// Content-free by construction: only the effective cap and the
/// pre-clamp values of declared carriers, never request content.
#[derive(Debug, Clone, serde::Serialize)]
pub struct OutputCapApplication {
    /// The effective cap that was applied.
    pub applied_value: u32,
    /// Declared carriers with their pre-clamp values, in the order
    /// `max_tokens`, `max_completion_tokens`, `max_output_tokens`.
    /// Empty means the cap was injected (`max_tokens` was absent).
    pub originals: Vec<(&'static str, u32)>,
}

/// Resolve the effective output cap from the key and group caps.
///
/// Returns the minimum of the set caps, or `None` when both are unset
/// (feature off; callers skip clamping entirely).
pub fn resolve_output_cap(key_cap: Option<u32>, group_cap: Option<u32>) -> Option<u32> {
    match (key_cap, group_cap) {
        (Some(k), Some(g)) => Some(k.min(g)),
        (Some(k), None) => Some(k),
        (None, Some(g)) => Some(g),
        (None, None) => None,
    }
}

/// Apply an output cap to a request's output-token carriers.
///
/// Carriers: typed `max_tokens` plus `extra["max_completion_tokens"]`
/// and `extra["max_output_tokens"]` when present as numbers. Each
/// declared carrier is clamped to `min(value, cap)` and never raised.
/// When all carriers are absent, typed `max_tokens = cap` is injected.
/// Non-numeric extra values are skipped without modification.
pub fn apply_output_cap(req: &mut OpenAIRequest, cap: u32) -> OutputCapApplication {
    let mut originals: Vec<(&'static str, u32)> = Vec::new();

    if let Some(original) = req.max_tokens {
        originals.push(("max_tokens", original));
        req.max_tokens = Some(original.min(cap));
    }

    for key in ["max_completion_tokens", "max_output_tokens"] {
        if let Some(value) = req.extra.get(key) {
            if let Some(original) = value.as_u64() {
                let original = u32::try_from(original).unwrap_or(u32::MAX);
                originals.push((match key {
                    "max_completion_tokens" => "max_completion_tokens",
                    _ => "max_output_tokens",
                }, original));
                let clamped = original.min(cap);
                req.extra.insert(key.to_string(), serde_json::json!(clamped));
            }
        }
    }

    if originals.is_empty() {
        req.max_tokens = Some(cap);
    }

    OutputCapApplication {
        applied_value: cap,
        originals,
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    fn req() -> OpenAIRequest {
        OpenAIRequest {
            model: "test-model".to_string(),
            messages: vec![],
            stream: false,
            temperature: None,
            max_tokens: None,
            extra: serde_json::Map::new(),
        }
    }

    #[test]
    fn test_resolve_output_cap() {
        assert_eq!(resolve_output_cap(None, None), None);
        assert_eq!(resolve_output_cap(Some(100), None), Some(100));
        assert_eq!(resolve_output_cap(None, Some(50)), Some(50));
        assert_eq!(resolve_output_cap(Some(100), Some(50)), Some(50));
        assert_eq!(resolve_output_cap(Some(50), Some(100)), Some(50));
        assert_eq!(resolve_output_cap(Some(100), Some(100)), Some(100));
    }

    #[test]
    fn test_apply_each_carrier_alone() {
        let cap = 50u32;

        let mut r = req();
        r.max_tokens = Some(100);
        let app = apply_output_cap(&mut r, cap);
        assert_eq!(r.max_tokens, Some(50));
        assert_eq!(r.extra.get("max_completion_tokens"), None);
        assert_eq!(r.extra.get("max_output_tokens"), None);
        assert_eq!(app.applied_value, 50);
        assert_eq!(app.originals, vec![("max_tokens", 100)]);

        let mut r = req();
        r.extra.insert("max_completion_tokens".to_string(), json!(100));
        let app = apply_output_cap(&mut r, cap);
        assert_eq!(r.extra.get("max_completion_tokens"), Some(&json!(50)));
        assert_eq!(r.max_tokens, None);
        assert_eq!(app.originals, vec![("max_completion_tokens", 100)]);

        let mut r = req();
        r.extra.insert("max_output_tokens".to_string(), json!(100));
        let app = apply_output_cap(&mut r, cap);
        assert_eq!(r.extra.get("max_output_tokens"), Some(&json!(50)));
        assert_eq!(r.max_tokens, None);
        assert_eq!(app.originals, vec![("max_output_tokens", 100)]);
    }

    #[test]
    fn test_apply_mixed_carriers() {
        let mut r = req();
        r.max_tokens = Some(30);
        r.extra.insert("max_completion_tokens".to_string(), json!(80));
        r.extra.insert("max_output_tokens".to_string(), json!(200));
        let app = apply_output_cap(&mut r, 50);
        assert_eq!(r.max_tokens, Some(30));
        assert_eq!(r.extra.get("max_completion_tokens"), Some(&json!(50)));
        assert_eq!(r.extra.get("max_output_tokens"), Some(&json!(50)));
        assert_eq!(app.applied_value, 50);
        assert_eq!(
            app.originals,
            vec![
                ("max_tokens", 30),
                ("max_completion_tokens", 80),
                ("max_output_tokens", 200),
            ]
        );
    }

    #[test]
    fn test_apply_ceiling_only_never_raises() {
        let mut r = req();
        r.max_tokens = Some(10);
        r.extra.insert("max_completion_tokens".to_string(), json!(20));
        r.extra.insert("max_output_tokens".to_string(), json!(5));
        let app = apply_output_cap(&mut r, 100);
        assert_eq!(r.max_tokens, Some(10));
        assert_eq!(r.extra.get("max_completion_tokens"), Some(&json!(20)));
        assert_eq!(r.extra.get("max_output_tokens"), Some(&json!(5)));
        assert_eq!(app.applied_value, 100);
        assert_eq!(app.originals.len(), 3);
    }

    #[test]
    fn test_apply_injects_when_all_absent() {
        let mut r = req();
        let app = apply_output_cap(&mut r, 64);
        assert_eq!(r.max_tokens, Some(64));
        assert!(r.extra.get("max_completion_tokens").is_none());
        assert!(r.extra.get("max_output_tokens").is_none());
        assert_eq!(app.applied_value, 64);
        assert!(app.originals.is_empty());
    }

    #[test]
    fn test_apply_injects_when_only_non_numeric_extras_present() {
        let mut r = req();
        r.extra.insert("max_completion_tokens".to_string(), json!("unlimited"));
        r.extra.insert("max_output_tokens".to_string(), json!(null));
        let app = apply_output_cap(&mut r, 32);
        assert_eq!(r.max_tokens, Some(32));
        assert_eq!(r.extra.get("max_completion_tokens"), Some(&json!("unlimited")));
        assert_eq!(r.extra.get("max_output_tokens"), Some(&json!(null)));
        assert!(app.originals.is_empty());
    }

    #[test]
    fn test_apply_cap_equal_to_value() {
        let mut r = req();
        r.max_tokens = Some(50);
        r.extra.insert("max_completion_tokens".to_string(), json!(50));
        let app = apply_output_cap(&mut r, 50);
        assert_eq!(r.max_tokens, Some(50));
        assert_eq!(r.extra.get("max_completion_tokens"), Some(&json!(50)));
        assert_eq!(app.applied_value, 50);
        assert_eq!(app.originals, vec![("max_tokens", 50), ("max_completion_tokens", 50)]);
    }

    #[test]
    fn test_apply_skips_float_extras() {
        let mut r = req();
        r.extra.insert("max_completion_tokens".to_string(), json!(1.5));
        r.extra.insert("max_output_tokens".to_string(), json!("high"));
        let app = apply_output_cap(&mut r, 10);
        assert_eq!(r.max_tokens, Some(10));
        assert_eq!(r.extra.get("max_completion_tokens"), Some(&json!(1.5)));
        assert_eq!(r.extra.get("max_output_tokens"), Some(&json!("high")));
        assert!(app.originals.is_empty());
    }

    use proptest::prelude::*;

    proptest! {
        #[test]
        fn prop_resolve_returns_min_of_set_caps(
            key_cap in proptest::option::of(0u32..=100_000),
            group_cap in proptest::option::of(0u32..=100_000),
        ) {
            let resolved = resolve_output_cap(key_cap, group_cap);
            match (key_cap, group_cap) {
                (Some(k), Some(g)) => prop_assert_eq!(resolved, Some(k.min(g))),
                (Some(k), None) => prop_assert_eq!(resolved, Some(k)),
                (None, Some(g)) => prop_assert_eq!(resolved, Some(g)),
                (None, None) => prop_assert_eq!(resolved, None),
            }
        }

        #[test]
        fn prop_apply_never_raises_any_carrier(
            max_tokens in proptest::option::of(0u32..=100_000),
            max_completion in proptest::option::of(0u64..=100_000),
            max_output in proptest::option::of(0u64..=100_000),
            cap in 0u32..=100_000,
        ) {
            let mut r = req();
            r.max_tokens = max_tokens;
            if let Some(v) = max_completion {
                r.extra.insert("max_completion_tokens".to_string(), json!(v));
            }
            if let Some(v) = max_output {
                r.extra.insert("max_output_tokens".to_string(), json!(v));
            }

            let app = apply_output_cap(&mut r, cap);

            if let Some(original) = max_tokens {
                prop_assert!(r.max_tokens.unwrap() <= original);
            }
            for (key, original) in [("max_completion_tokens", max_completion), ("max_output_tokens", max_output)] {
                if let Some(original) = original {
                    let after = r.extra.get(key).unwrap().as_u64().unwrap();
                    prop_assert!(after <= original);
                    prop_assert!(after <= cap as u64);
                }
            }

            if max_tokens.is_none() && max_completion.is_none() && max_output.is_none() {
                prop_assert_eq!(r.max_tokens, Some(cap));
                prop_assert!(app.originals.is_empty());
            } else {
                let mut declared = 0;
                if max_tokens.is_some() {
                    declared += 1;
                }
                if max_completion.is_some() {
                    declared += 1;
                }
                if max_output.is_some() {
                    declared += 1;
                }
                prop_assert_eq!(app.originals.len(), declared);
            }
            prop_assert_eq!(app.applied_value, cap);
        }
    }
}
