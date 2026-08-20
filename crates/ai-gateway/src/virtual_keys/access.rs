//! Model access restriction for virtual keys.
//!
//! Implements [`VirtualKeyManager::check_model_access`], the pure membership
//! check that decides whether a virtual key may use a requested model. The
//! check is deliberately minimal: it compares the requested model name against
//! the key's permitted list using case-sensitive exact matching.
//!
//! Group expansion and configuration-existence handling (Req 6.5 — ignoring
//! list entries that do not correspond to a real config model/group) are
//! deferred to the middleware integration task (13.1), where the gateway's
//! model-group configuration is available. At this layer we only know the
//! permitted names and the requested model, so "ignore non-existent entries"
//! reduces to a pure set-membership test: a request is permitted iff the
//! requested model appears in the list. Entries that reference nothing real are
//! simply never matched and thus never cause a rejection on their own.

use super::models::AuthenticatedKey;
use super::VirtualKeyManager;

/// Error returned when a virtual key is not permitted to use a model.
///
/// Maps to HTTP 403 with body
/// `{"error": "Model not permitted", "model": "...", "allowed": [...]}`
/// (design: HTTP Error Mapping table).
#[derive(Debug, Clone, PartialEq, Eq, thiserror::Error)]
pub enum AccessError {
    /// The requested `model` is not present in the key's access list.
    /// `allowed` carries the key's permitted scope for the response body.
    #[error("Model {model} not permitted for this key")]
    ModelDenied { model: String, allowed: Vec<String> },
}

impl VirtualKeyManager {
    /// Check whether `key` is permitted to use `model`.
    ///
    /// Semantics (Req 6.1, 6.2, 6.3, 6.5):
    /// - If the key has no access list (`model_access` is `None`), all models
    ///   are permitted → `Ok(())`.
    /// - Otherwise the requested `model` must appear in the list using a
    ///   case-sensitive exact match → `Ok(())`.
    /// - If the model is absent, return [`AccessError::ModelDenied`] carrying
    ///   the requested model name and the permitted list.
    ///
    /// This is a pure membership check. List entries that do not correspond to
    /// a real configured model/group are ignored implicitly (they never match),
    /// satisfying Req 6.5 without needing the gateway configuration here. Group
    /// name expansion is wired in the middleware integration task (13.1).
    ///
    /// Note: an empty access list permits no models (every request is denied).
    /// Key create/update validation forbids an empty `model_access` list, so
    /// this state should not arise in practice; the deny-all behavior is the
    /// safe default should it occur.
    ///
    /// Enforcement ordering (Req 6.4 — model access is checked after
    /// authentication and before budget/rate-limit checks) is the middleware's
    /// responsibility (task 13.1); this method is the pure check only.
    ///
    /// _Requirements: 6.1, 6.2, 6.3, 6.4, 6.5_
    pub fn check_model_access(
        &self,
        key: &AuthenticatedKey,
        model: &str,
    ) -> Result<(), AccessError> {
        match &key.model_access {
            // No list configured → access to all models (Req 6.3).
            None => Ok(()),
            // Case-sensitive exact membership (Req 6.1).
            Some(allowed) => {
                if allowed.iter().any(|entry| entry == model) {
                    Ok(())
                } else {
                    // Req 6.2: reject with the denied model and permitted scope.
                    Err(AccessError::ModelDenied {
                        model: model.to_string(),
                        allowed: allowed.clone(),
                    })
                }
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::virtual_keys::models::KeyStatus;
    use proptest::prelude::*;
    use std::path::Path;

    /// Build an [`AuthenticatedKey`] with the given `model_access` list; all
    /// other constraints are unset (irrelevant to model access checks).
    fn key_with_access(model_access: Option<Vec<String>>) -> AuthenticatedKey {
        AuthenticatedKey {
            id: "test-key".to_string(),
            name: None,
            status: KeyStatus::Active,
            budget_limit_usd: None,
            token_budget: None,
            budget_window: None,
            current_spend_usd: 0.0,
            current_tokens_used: 0,
            window_start: None,
            requests_per_minute: None,
            tokens_per_minute: None,
            model_access,
            expires_at: None,
            loop_detection: None,
        }
    }

    fn manager() -> (VirtualKeyManager, tempfile::NamedTempFile) {
        let temp = tempfile::NamedTempFile::new().unwrap();
        let mgr = VirtualKeyManager::new(Path::new(temp.path())).unwrap();
        (mgr, temp)
    }

    /// Req 6.1: a model present in the list is permitted (exact match).
    #[test]
    fn model_in_list_permitted() {
        let (mgr, _tmp) = manager();
        let key = key_with_access(Some(vec!["gpt-4".to_string(), "claude-3-opus".to_string()]));
        assert!(mgr.check_model_access(&key, "gpt-4").is_ok());
        assert!(mgr.check_model_access(&key, "claude-3-opus").is_ok());
    }

    /// Req 6.2: a model absent from the list is denied, carrying the requested
    /// model name and the permitted scope.
    #[test]
    fn model_not_in_list_denied() {
        let (mgr, _tmp) = manager();
        let allowed = vec!["gpt-4".to_string()];
        let key = key_with_access(Some(allowed.clone()));

        let err = mgr.check_model_access(&key, "gpt-3.5-turbo").unwrap_err();
        assert_eq!(
            err,
            AccessError::ModelDenied {
                model: "gpt-3.5-turbo".to_string(),
                allowed,
            }
        );
    }

    /// Req 6.1: matching is case-sensitive — a case-mismatched name is denied.
    #[test]
    fn model_match_is_case_sensitive() {
        let (mgr, _tmp) = manager();
        let key = key_with_access(Some(vec!["GPT-4".to_string()]));
        assert!(mgr.check_model_access(&key, "gpt-4").is_err());
        assert!(mgr.check_model_access(&key, "GPT-4").is_ok());
    }

    /// Req 6.3: no access list configured permits every model.
    #[test]
    fn none_list_permits_all() {
        let (mgr, _tmp) = manager();
        let key = key_with_access(None);
        assert!(mgr.check_model_access(&key, "gpt-4").is_ok());
        assert!(mgr.check_model_access(&key, "anything-at-all").is_ok());
    }

    /// Edge case: an empty list matches nothing and therefore denies all
    /// models. Create/update validation forbids empty `model_access`, so this
    /// is a defensive check documenting the deny-all fallback.
    #[test]
    fn empty_list_denies_all() {
        let (mgr, _tmp) = manager();
        let key = key_with_access(Some(vec![]));
        let err = mgr.check_model_access(&key, "gpt-4").unwrap_err();
        assert_eq!(
            err,
            AccessError::ModelDenied {
                model: "gpt-4".to_string(),
                allowed: vec![],
            }
        );
    }

    proptest! {
        #![proptest_config(ProptestConfig { cases: 64, ..ProptestConfig::default() })]

        // Feature: virtual-key-management, Property 12: For a key with a
        // Model_Access_List and any requested model name, check_model_access
        // permits IFF the model appears in the list (case-sensitive exact
        // match). For a key with model_access = None, ALL models are permitted.
        // Validates: Requirements 6.1, 6.2, 6.3
        #[test]
        fn prop_model_access_membership_iff(
            list in prop::collection::vec("[A-Za-z0-9._/-]{1,16}", 1..8),
            requested in "[A-Za-z0-9._/-]{1,16}",
        ) {
            let (mgr, _tmp) = manager();
            let key = key_with_access(Some(list.clone()));
            let result = mgr.check_model_access(&key, &requested);

            if list.iter().any(|entry| entry == &requested) {
                // In the list → permitted.
                prop_assert!(result.is_ok());
            } else {
                // Not in the list → denied with the requested model + full scope.
                prop_assert_eq!(
                    result.unwrap_err(),
                    AccessError::ModelDenied {
                        model: requested.clone(),
                        allowed: list.clone(),
                    }
                );
            }
        }

        // Feature: virtual-key-management, Property 12: a key with
        // model_access = None permits every requested model.
        // Validates: Requirements 6.3
        #[test]
        fn prop_none_list_permits_all(requested in "\\PC{0,32}") {
            let (mgr, _tmp) = manager();
            let key = key_with_access(None);
            prop_assert!(mgr.check_model_access(&key, &requested).is_ok());
        }

        // Feature: virtual-key-management, Property 12: matching is
        // case-sensitive — a differing-case variant absent from the list is
        // denied, while the exact-case entry is permitted.
        // Validates: Requirements 6.1, 6.2
        #[test]
        fn prop_model_access_case_sensitive(model in "[a-z][a-z0-9]{0,15}") {
            let upper = model.to_ascii_uppercase();
            // First char is always an ascii lowercase letter, so the uppercase
            // variant is guaranteed to differ from the original.
            prop_assume!(upper != model);

            let (mgr, _tmp) = manager();
            // List holds only the uppercase variant.
            let key = key_with_access(Some(vec![upper.clone()]));

            // Lower-case request is not an exact match → denied.
            prop_assert_eq!(
                mgr.check_model_access(&key, &model).unwrap_err(),
                AccessError::ModelDenied {
                    model: model.clone(),
                    allowed: vec![upper.clone()],
                }
            );
            // Exact-case request → permitted.
            prop_assert!(mgr.check_model_access(&key, &upper).is_ok());
        }
    }
}
