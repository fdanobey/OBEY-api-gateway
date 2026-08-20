//! Key update and lifecycle operations for [`VirtualKeyManager`].
//!
//! Task 10.1 implements partial updates ([`VirtualKeyManager::update_key`])
//! with `Option<Option<T>>` semantics: an omitted field (`None`) is left
//! unchanged, `Some(Some(v))` sets a value, and `Some(None)` clears a nullable
//! limit. Updates are validated per the design "Validation Rules" (with the
//! update-specific `tokens_per_minute` ceiling of 100,000,000 from Req 7.2),
//! rejected for revoked keys (HTTP 409, Req 7.8), and — on success —
//! invalidate the authentication cache and drop the per-key rate limiter so new
//! constraints take effect on the next request (Req 7.5).
//!
//! The [`stored_to_info`] helper converts a persistence-layer
//! [`StoredVirtualKey`] into the masked, non-sensitive [`VirtualKeyInfo`] view
//! returned by update/list/info responses (reused by task 10.2).

use chrono::Utc;

use super::models::{
    validate_name, ExpiresIn, KeyStatus, ListKeysParams, PaginatedKeys, UpdateKeyParams,
    VirtualKeyInfo,
};
use super::store::{KeyUpdates, StoredVirtualKey};
use super::{KeyError, ValidationErrors, VirtualKeyManager};

// Validation bounds for updates (design "Validation Rules"). Re-declared
// locally to keep the module self-contained and because the `tokens_per_minute`
// ceiling for updates (Req 7.2) differs from key creation.
const BUDGET_USD_MIN: f64 = 0.01;
const BUDGET_USD_MAX: f64 = 999_999_999.99;
const TOKEN_BUDGET_MIN: u64 = 1;
const TOKEN_BUDGET_MAX: u64 = 999_999_999;
const RPM_MIN: u32 = 1;
const RPM_MAX: u32 = 100_000;
const TPM_MIN: u64 = 1;
/// Update-time tokens-per-minute ceiling per Req 7.2 (1 - 100,000,000), which
/// intentionally exceeds the 10,000,000 ceiling applied at key creation.
const TPM_UPDATE_MAX: u64 = 100_000_000;

// List pagination bounds at the manager level (design: default 50, max 100).
/// Page size applied when a caller passes `limit == 0`.
const LIST_DEFAULT_LIMIT: u32 = 50;
/// Hard ceiling on the manager-level page size.
const LIST_MAX_LIMIT: u32 = 100;

impl VirtualKeyManager {
    /// Apply a partial update to the key identified by `id`.
    ///
    /// Only fields present in `params` are changed; omitted fields
    /// (`None`) retain their current values (Req 7.1). A nullable field set to
    /// `Some(None)` clears the corresponding limit. Increasing a budget never
    /// resets the current spend/token counters (Req 7.3).
    ///
    /// Returns [`KeyError::NotFound`] when `id` is unknown (HTTP 404),
    /// [`KeyError::KeyRevoked`] when the target key is revoked (HTTP 409,
    /// Req 7.8), and [`KeyError::Validation`] when any provided field is out of
    /// range (HTTP 400, Req 7.9).
    ///
    /// _Requirements: 7.1, 7.2, 7.3, 7.4, 7.5, 7.6, 7.7, 7.8, 7.9_
    pub async fn update_key(
        &self,
        id: &str,
        params: UpdateKeyParams,
    ) -> Result<VirtualKeyInfo, KeyError> {
        // 1. Load the existing record (404 when absent).
        let stored = self
            .store
            .get_key_by_id(id)?
            .ok_or_else(|| KeyError::NotFound(id.to_string()))?;

        // 2. Revoked keys are immutable (Req 7.8, HTTP 409).
        if stored.status == KeyStatus::Revoked {
            return Err(KeyError::KeyRevoked);
        }

        // 3. Validate every provided field (Req 7.9, HTTP 400).
        validate_update_params(&params)?;

        // 4. Translate into a store-layer `KeyUpdates`.
        //
        // Expiration (Req 7.6/7.7): `Never` removes any expiration
        // (`Some(None)`); a duration variant is computed relative to now
        // (`Some(Some(now + d))`); an omitted field leaves it unchanged.
        let now = Utc::now();
        let expires_at = match &params.expires_in {
            None => None,
            Some(ExpiresIn::Never) => Some(None),
            Some(other) => Some(other.to_duration().map(|d| now + d)),
        };

        let updates = KeyUpdates {
            name: params.name.clone(),
            status: None,
            budget_limit_usd: params.budget_limit_usd,
            token_budget: params.token_budget,
            budget_window: params.budget_window.clone(),
            requests_per_minute: params.requests_per_minute,
            tokens_per_minute: params.tokens_per_minute,
            model_access_list: params.model_access.clone(),
            loop_detection: params.loop_detection.clone(),
            expires_at,
            // Req 7.3: never touch usage counters or window start on update.
            window_start: None,
            current_spend_usd: None,
            current_tokens_used: None,
            last_used_at: None,
            request_count: None,
        };

        // 5. Persist.
        self.store.update_key(id, &updates)?;

        // 6. Invalidate the auth cache (keyed by the SHA-256 hash) and drop the
        // per-key rate limiter so updated RPM/TPM constraints take effect on the
        // next request (Req 7.5).
        self.invalidate_cache(&stored.key_hash);
        self.rate_limiters.remove(&stored.id);

        // 7. Re-load and return the masked, non-sensitive view.
        let updated = self
            .store
            .get_key_by_id(id)?
            .ok_or_else(|| KeyError::NotFound(id.to_string()))?;
        Ok(stored_to_info(&updated))
    }

    /// List stored keys as a page of masked [`VirtualKeyInfo`] views plus the
    /// total count in the store.
    ///
    /// The requested `limit` is clamped to the manager-level bounds: a `limit`
    /// of `0` falls back to the default page size ([`LIST_DEFAULT_LIMIT`], 50)
    /// and any larger value is capped at [`LIST_MAX_LIMIT`] (100). Results are
    /// ordered by creation date descending (enforced by the store) and each
    /// entry exposes only the masked key prefix, never the full key value
    /// (Req 8.1, 8.5).
    ///
    /// _Requirements: 8.1, 8.5_
    pub async fn list_keys(&self, params: ListKeysParams) -> Result<PaginatedKeys, KeyError> {
        let limit = if params.limit == 0 {
            LIST_DEFAULT_LIMIT
        } else {
            params.limit.min(LIST_MAX_LIMIT)
        };

        let (stored, total) = self.store.list_keys(limit, params.offset)?;
        let keys = stored.iter().map(stored_to_info).collect();
        Ok(PaginatedKeys { keys, total })
    }

    /// Return the masked [`VirtualKeyInfo`] for a single key.
    ///
    /// Returns [`KeyError::NotFound`] (HTTP 404) when `id` is unknown. The
    /// response exposes only the masked key prefix; the full or encrypted key
    /// value is never included (Req 8.2, 8.5).
    ///
    /// _Requirements: 8.2, 8.5_
    pub async fn get_key(&self, id: &str) -> Result<VirtualKeyInfo, KeyError> {
        let stored = self
            .store
            .get_key_by_id(id)?
            .ok_or_else(|| KeyError::NotFound(id.to_string()))?;
        Ok(stored_to_info(&stored))
    }

    /// Revoke a key, marking its status as [`KeyStatus::Revoked`].
    ///
    /// Idempotent per Req 8.6: if the key is already revoked or expired, the
    /// current info is returned unchanged and without error. Otherwise the
    /// status is set to revoked, the authentication cache entry and per-key rate
    /// limiter are dropped so subsequent requests are rejected (Req 8.3), and
    /// the updated masked info is returned.
    ///
    /// Returns [`KeyError::NotFound`] (HTTP 404) when `id` is unknown.
    ///
    /// _Requirements: 8.3, 8.5, 8.6_
    pub async fn revoke_key(&self, id: &str) -> Result<VirtualKeyInfo, KeyError> {
        let stored = self
            .store
            .get_key_by_id(id)?
            .ok_or_else(|| KeyError::NotFound(id.to_string()))?;

        // Req 8.6: already-revoked or expired keys are returned as-is.
        if matches!(stored.status, KeyStatus::Revoked | KeyStatus::Expired) {
            return Ok(stored_to_info(&stored));
        }

        self.store.update_key(
            id,
            &KeyUpdates {
                status: Some(KeyStatus::Revoked),
                ..Default::default()
            },
        )?;

        // Invalidate the auth cache and drop the rate limiter so the revocation
        // takes effect on the next request (Req 8.3).
        self.invalidate_cache(&stored.key_hash);
        self.rate_limiters.remove(&stored.id);

        let updated = self
            .store
            .get_key_by_id(id)?
            .ok_or_else(|| KeyError::NotFound(id.to_string()))?;
        Ok(stored_to_info(&updated))
    }

    /// Delete a key and its usage history from the store.
    ///
    /// Removes the key record; associated usage rows are removed via the
    /// `ON DELETE CASCADE` foreign key (Req 8.4). The authentication cache entry
    /// and per-key rate limiter are dropped so no stale state remains. Returns
    /// [`KeyError::NotFound`] (HTTP 404) when `id` is unknown, consistent with
    /// the other lifecycle operations.
    ///
    /// _Requirements: 8.4_
    pub async fn delete_key(&self, id: &str) -> Result<(), KeyError> {
        let stored = self
            .store
            .get_key_by_id(id)?
            .ok_or_else(|| KeyError::NotFound(id.to_string()))?;

        self.store.delete_key(id)?;
        self.invalidate_cache(&stored.key_hash);
        self.rate_limiters.remove(&stored.id);
        Ok(())
    }
}

/// Convert a persistence-layer [`StoredVirtualKey`] into the masked
/// [`VirtualKeyInfo`] returned by update/list/info responses.
///
/// Only the stored `key_prefix` (first 8 characters) is exposed; the full or
/// encrypted key value is never included (Req 8.5).
pub(crate) fn stored_to_info(stored: &StoredVirtualKey) -> VirtualKeyInfo {
    VirtualKeyInfo {
        id: stored.id.clone(),
        key_prefix: stored.key_prefix.clone(),
        name: stored.name.clone(),
        status: stored.status.clone(),
        budget_limit_usd: stored.budget_limit_usd,
        token_budget: stored.token_budget,
        budget_window: stored.budget_window.clone(),
        current_spend_usd: stored.current_spend_usd,
        current_tokens_used: stored.current_tokens_used,
        requests_per_minute: stored.requests_per_minute,
        tokens_per_minute: stored.tokens_per_minute,
        model_access: stored.model_access_list.clone(),
        loop_detection: stored.loop_detection.clone(),
        request_count: stored.request_count,
        created_at: stored.created_at,
        expires_at: stored.expires_at,
        last_used_at: stored.last_used_at,
    }
}

/// Validate all provided [`UpdateKeyParams`] fields per the design "Validation
/// Rules". Only `Some(Some(v))` values are range-checked; clearing a limit
/// (`Some(None)`) and omitting a field (`None`) skip validation. Failures are
/// accumulated so callers receive a complete field-error set.
fn validate_update_params(params: &UpdateKeyParams) -> Result<(), KeyError> {
    let mut errors = ValidationErrors::new();

    if let Some(Some(name)) = &params.name {
        if validate_name(name).is_err() {
            errors.push("name", "name must be between 1 and 128 characters");
        }
    }

    if let Some(Some(budget)) = params.budget_limit_usd {
        if !budget.is_finite() || !(BUDGET_USD_MIN..=BUDGET_USD_MAX).contains(&budget) {
            errors.push(
                "budget_limit_usd",
                "budget must be between 0.01 and 999999999.99",
            );
        }
    }

    if let Some(Some(token_budget)) = params.token_budget {
        if !(TOKEN_BUDGET_MIN..=TOKEN_BUDGET_MAX).contains(&token_budget) {
            errors.push(
                "token_budget",
                "token_budget must be between 1 and 999999999",
            );
        }
    }

    if let Some(Some(rpm)) = params.requests_per_minute {
        if !(RPM_MIN..=RPM_MAX).contains(&rpm) {
            errors.push(
                "requests_per_minute",
                "requests_per_minute must be between 1 and 100000",
            );
        }
    }

    if let Some(Some(tpm)) = params.tokens_per_minute {
        if !(TPM_MIN..=TPM_UPDATE_MAX).contains(&tpm) {
            errors.push(
                "tokens_per_minute",
                "tokens_per_minute must be between 1 and 100000000",
            );
        }
    }

    if let Some(Some(model_access)) = &params.model_access {
        if model_access.is_empty() {
            errors.push("model_access", "model_access must be a non-empty list");
        }
    }

    if let Some(Some(loop_detection)) = &params.loop_detection {
        if let Err(loop_errors) =
            loop_detection.merge(&crate::loop_detection::LoopDetectionConfig::default())
        {
            for error in loop_errors {
                errors.push("loop_detection", error.to_string());
            }
        }
    }

    errors.into_result().map_err(KeyError::Validation)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::virtual_keys::models::{CreateKeyParams, ExpiresIn};
    use tempfile::NamedTempFile;

    fn manager() -> (VirtualKeyManager, NamedTempFile) {
        let tmp = NamedTempFile::new().unwrap();
        let mgr = VirtualKeyManager::new(tmp.path()).unwrap();
        (mgr, tmp)
    }

    fn create_defaults() -> CreateKeyParams {
        CreateKeyParams {
            name: None,
            budget_limit_usd: None,
            token_budget: None,
            budget_window: None,
            requests_per_minute: None,
            tokens_per_minute: None,
            model_access: None,
            expires_in: None,
            loop_detection: None,
        }
    }

    /// Req 7.1: a partial update changes only the named field; omitted fields
    /// keep their previous values.
    #[tokio::test]
    async fn update_preserves_omitted_fields() {
        let (mgr, _tmp) = manager();
        let created = mgr
            .create_key(CreateKeyParams {
                name: Some("orig".into()),
                budget_limit_usd: Some(100.0),
                requests_per_minute: Some(60),
                ..create_defaults()
            })
            .await
            .unwrap();

        let params = UpdateKeyParams {
            name: Some(Some("renamed".into())),
            ..Default::default()
        };
        let info = mgr.update_key(&created.id, params).await.unwrap();

        assert_eq!(info.name.as_deref(), Some("renamed"));
        assert_eq!(info.budget_limit_usd, Some(100.0));
        assert_eq!(info.requests_per_minute, Some(60));
    }

    /// Req 7.3: increasing the budget leaves the current spend/token counters
    /// untouched.
    #[tokio::test]
    async fn budget_increase_preserves_spend() {
        let (mgr, _tmp) = manager();
        let created = mgr
            .create_key(CreateKeyParams {
                budget_limit_usd: Some(10.0),
                ..create_defaults()
            })
            .await
            .unwrap();
        mgr.store
            .update_usage_counters(&created.id, 5.0, 100)
            .unwrap();

        let params = UpdateKeyParams {
            budget_limit_usd: Some(Some(50.0)),
            ..Default::default()
        };
        let info = mgr.update_key(&created.id, params).await.unwrap();

        assert_eq!(info.budget_limit_usd, Some(50.0));
        assert!((info.current_spend_usd - 5.0).abs() < 1e-9);
        assert_eq!(info.current_tokens_used, 100);
    }

    /// Req 7.7: setting `expires_in` to `never` removes any existing
    /// expiration.
    #[tokio::test]
    async fn expires_in_never_removes_expiration() {
        let (mgr, _tmp) = manager();
        let created = mgr
            .create_key(CreateKeyParams {
                expires_in: Some(ExpiresIn::OneMonth),
                ..create_defaults()
            })
            .await
            .unwrap();
        assert!(created.expires_at.is_some());

        let params = UpdateKeyParams {
            expires_in: Some(ExpiresIn::Never),
            ..Default::default()
        };
        let info = mgr.update_key(&created.id, params).await.unwrap();

        assert!(info.expires_at.is_none());
    }

    /// Req 7.6: setting `expires_in` to a duration computes a new expiration
    /// relative to now.
    #[tokio::test]
    async fn expires_in_duration_sets_expiration() {
        let (mgr, _tmp) = manager();
        let created = mgr.create_key(create_defaults()).await.unwrap();
        assert!(created.expires_at.is_none());

        let before = Utc::now();
        let params = UpdateKeyParams {
            expires_in: Some(ExpiresIn::OneDay),
            ..Default::default()
        };
        let info = mgr.update_key(&created.id, params).await.unwrap();

        let expires_at = info.expires_at.expect("expiration set");
        // Roughly one day out (allowing for test execution time drift).
        let delta = expires_at - before;
        assert!(delta >= chrono::Duration::hours(23));
        assert!(delta <= chrono::Duration::hours(25));
    }

    /// Req 7.8: updating a revoked key is rejected with a conflict (HTTP 409).
    #[tokio::test]
    async fn update_revoked_key_conflicts() {
        let (mgr, _tmp) = manager();
        let created = mgr.create_key(create_defaults()).await.unwrap();
        mgr.store
            .update_key(
                &created.id,
                &KeyUpdates {
                    status: Some(KeyStatus::Revoked),
                    ..Default::default()
                },
            )
            .unwrap();

        let err = mgr
            .update_key(&created.id, UpdateKeyParams::default())
            .await
            .unwrap_err();
        assert!(matches!(err, KeyError::KeyRevoked));
    }

    /// Req 7.9: an out-of-range field is rejected with a structured validation
    /// error (HTTP 400).
    #[tokio::test]
    async fn invalid_field_rejected() {
        let (mgr, _tmp) = manager();
        let created = mgr.create_key(create_defaults()).await.unwrap();

        let params = UpdateKeyParams {
            budget_limit_usd: Some(Some(0.0)), // below the 0.01 minimum
            ..Default::default()
        };
        let err = mgr.update_key(&created.id, params).await.unwrap_err();

        let KeyError::Validation(errors) = err else {
            panic!("expected validation error");
        };
        assert!(errors.errors.iter().any(|e| e.field == "budget_limit_usd"));
    }

    /// Updating an unknown id returns not found (HTTP 404).
    #[tokio::test]
    async fn update_missing_key_not_found() {
        let (mgr, _tmp) = manager();
        let err = mgr
            .update_key("does-not-exist", UpdateKeyParams::default())
            .await
            .unwrap_err();
        assert!(matches!(err, KeyError::NotFound(_)));
    }

    // --- Task 10.2: list / get / revoke / delete ----------------------------

    /// Req 8.1: `limit == 0` falls back to the default page size and the total
    /// reflects the full store count.
    #[tokio::test]
    async fn list_keys_clamps_zero_to_default_and_reports_total() {
        let (mgr, _tmp) = manager();
        for _ in 0..3 {
            mgr.create_key(create_defaults()).await.unwrap();
        }

        let page = mgr
            .list_keys(ListKeysParams {
                limit: 0,
                offset: 0,
            })
            .await
            .unwrap();

        assert_eq!(page.total, 3);
        assert_eq!(page.keys.len(), 3);
    }

    /// Req 8.1: an oversized `limit` is capped at the manager-level maximum
    /// (100), while the total still reports every stored key.
    #[tokio::test]
    async fn list_keys_caps_limit_at_max() {
        let (mgr, _tmp) = manager();
        for _ in 0..5 {
            mgr.create_key(create_defaults()).await.unwrap();
        }

        // A limit far above the cap still succeeds; with only 5 keys the page
        // returns all of them and the cap does not truncate the result here.
        let page = mgr
            .list_keys(ListKeysParams {
                limit: 10_000,
                offset: 0,
            })
            .await
            .unwrap();

        assert_eq!(page.total, 5);
        assert_eq!(page.keys.len(), 5);
    }

    /// Req 8.2 / 8.5: `get_key` returns the masked prefix and never the full
    /// key value.
    #[tokio::test]
    async fn get_key_returns_masked_prefix() {
        let (mgr, _tmp) = manager();
        let created = mgr.create_key(create_defaults()).await.unwrap();

        let info = mgr.get_key(&created.id).await.unwrap();

        // Prefix is exactly the first 8 chars of the full key value.
        assert_eq!(info.key_prefix.len(), 8);
        assert_eq!(info.key_prefix, created.key[..8]);
        assert!(created.key.starts_with(&info.key_prefix));
    }

    /// Req 8.2: `get_key` on an unknown id returns not found (HTTP 404).
    #[tokio::test]
    async fn get_key_missing_not_found() {
        let (mgr, _tmp) = manager();
        let err = mgr.get_key("nope").await.unwrap_err();
        assert!(matches!(err, KeyError::NotFound(_)));
    }

    /// Req 8.3: revoking an active key sets its status to revoked.
    #[tokio::test]
    async fn revoke_sets_status_revoked() {
        let (mgr, _tmp) = manager();
        let created = mgr.create_key(create_defaults()).await.unwrap();

        let info = mgr.revoke_key(&created.id).await.unwrap();
        assert_eq!(info.status, KeyStatus::Revoked);

        let stored = mgr.store.get_key_by_id(&created.id).unwrap().unwrap();
        assert_eq!(stored.status, KeyStatus::Revoked);
    }

    /// Req 8.6: revoking an already-revoked key is idempotent — it returns the
    /// current status without error and without modification.
    #[tokio::test]
    async fn revoke_already_revoked_is_idempotent() {
        let (mgr, _tmp) = manager();
        let created = mgr.create_key(create_defaults()).await.unwrap();

        let first = mgr.revoke_key(&created.id).await.unwrap();
        assert_eq!(first.status, KeyStatus::Revoked);

        // Second revoke succeeds and remains revoked.
        let second = mgr.revoke_key(&created.id).await.unwrap();
        assert_eq!(second.status, KeyStatus::Revoked);
    }

    /// Req 8.3: revoking an unknown id returns not found (HTTP 404).
    #[tokio::test]
    async fn revoke_missing_not_found() {
        let (mgr, _tmp) = manager();
        let err = mgr.revoke_key("nope").await.unwrap_err();
        assert!(matches!(err, KeyError::NotFound(_)));
    }

    /// Req 8.4: deleting a key removes the record; a subsequent `get_key`
    /// returns not found.
    #[tokio::test]
    async fn delete_removes_key() {
        let (mgr, _tmp) = manager();
        let created = mgr.create_key(create_defaults()).await.unwrap();

        mgr.delete_key(&created.id).await.unwrap();

        let err = mgr.get_key(&created.id).await.unwrap_err();
        assert!(matches!(err, KeyError::NotFound(_)));
        assert!(mgr.store.get_key_by_id(&created.id).unwrap().is_none());
    }

    /// Req 8.4: deleting an unknown id returns not found (HTTP 404).
    #[tokio::test]
    async fn delete_missing_not_found() {
        let (mgr, _tmp) = manager();
        let err = mgr.delete_key("nope").await.unwrap_err();
        assert!(matches!(err, KeyError::NotFound(_)));
    }

    // --- Task 10.3: property-based tests ------------------------------------

    use proptest::prelude::*;

    /// Build a fresh single-threaded runtime for one proptest case. proptest
    /// test bodies are synchronous, so each case drives the async manager API
    /// via `block_on`. A fresh runtime + fresh `manager()` keeps every case
    /// isolated.
    fn rt() -> tokio::runtime::Runtime {
        tokio::runtime::Runtime::new().unwrap()
    }

    proptest! {
        #![proptest_config(ProptestConfig {
            cases: 64,
            .. ProptestConfig::default()
        })]

        // Feature: virtual-key-management, Property 13: Partial Update Preservation
        //
        // For any virtual key and any partial update touching a subset of
        // fields, only the provided fields change; omitted fields retain their
        // previous values.
        //
        // **Validates: Requirements 7.1, 7.3**

        /// A partial update that touches a random subset of {name, budget}
        /// changes only those fields and preserves everything else (here the
        /// fixed requests_per_minute).
        #[test]
        fn prop_partial_update_preserves_omitted_fields(
            name0 in "[a-zA-Z0-9 ]{1,32}",
            budget0 in 0.01f64..=1000.0,
            update_name in any::<bool>(),
            name1 in "[a-zA-Z0-9 ]{1,32}",
            update_budget in any::<bool>(),
            budget1 in 0.01f64..=1000.0,
        ) {
            let (updated, exp_name, exp_budget) = rt().block_on(async {
                let (mgr, _tmp) = manager();
                let created = mgr
                    .create_key(CreateKeyParams {
                        name: Some(name0.clone()),
                        budget_limit_usd: Some(budget0),
                        requests_per_minute: Some(60),
                        ..create_defaults()
                    })
                    .await
                    .unwrap();

                let params = UpdateKeyParams {
                    name: if update_name { Some(Some(name1.clone())) } else { None },
                    budget_limit_usd: if update_budget { Some(Some(budget1)) } else { None },
                    ..Default::default()
                };
                let updated = mgr.update_key(&created.id, params).await.unwrap();

                let exp_name = if update_name { name1.clone() } else { name0.clone() };
                let exp_budget = if update_budget { budget1 } else { budget0 };
                (updated, exp_name, exp_budget)
            });

            // Provided fields changed to the new value; omitted fields kept old.
            prop_assert_eq!(updated.name.as_deref(), Some(exp_name.as_str()));
            prop_assert!(
                (updated.budget_limit_usd.unwrap() - exp_budget).abs() < 1e-9
            );
            // requests_per_minute was never in any update payload.
            prop_assert_eq!(updated.requests_per_minute, Some(60));
        }

        /// Increasing the budget never changes the current spend or token
        /// counters (Req 7.3).
        #[test]
        fn prop_budget_increase_preserves_spend(
            budget0 in 0.01f64..=500.0,
            spend in 0.0f64..=100.0,
            tokens in 0u64..=100_000,
            increase in 0.01f64..=499.0,
        ) {
            let (spend_after, tokens_after, budget_after) = rt().block_on(async {
                let (mgr, _tmp) = manager();
                let created = mgr
                    .create_key(CreateKeyParams {
                        budget_limit_usd: Some(budget0),
                        ..create_defaults()
                    })
                    .await
                    .unwrap();
                mgr.store
                    .update_usage_counters(&created.id, spend, tokens as i64)
                    .unwrap();

                let new_budget = budget0 + increase;
                let info = mgr
                    .update_key(
                        &created.id,
                        UpdateKeyParams {
                            budget_limit_usd: Some(Some(new_budget)),
                            ..Default::default()
                        },
                    )
                    .await
                    .unwrap();
                (info.current_spend_usd, info.current_tokens_used, info.budget_limit_usd)
            });

            prop_assert!((spend_after - spend).abs() < 1e-6);
            prop_assert_eq!(tokens_after, tokens);
            prop_assert!(budget_after.is_some());
        }

        // Feature: virtual-key-management, Property 14: Key Masking
        //
        // For any created key, get_key/list_keys expose a key_prefix equal to
        // exactly the first 8 characters of the full key value, and the full
        // key value never appears in the serialized VirtualKeyInfo.
        //
        // **Validates: Requirements 8.1, 8.5**

        /// The masked prefix is the first 8 chars of the full key, and the full
        /// key value is absent from the JSON view (from both get and list).
        #[test]
        fn prop_key_masking(_seed in 0u64..1_000) {
            let (full_key, get_prefix, list_prefix, json_get, json_list) =
                rt().block_on(async {
                    let (mgr, _tmp) = manager();
                    let created = mgr.create_key(create_defaults()).await.unwrap();

                    let g = mgr.get_key(&created.id).await.unwrap();
                    let page = mgr
                        .list_keys(ListKeysParams { limit: 0, offset: 0 })
                        .await
                        .unwrap();
                    let l = page.keys.into_iter().next().unwrap();

                    let json_get = serde_json::to_string(&g).unwrap();
                    let json_list = serde_json::to_string(&l).unwrap();
                    (created.key, g.key_prefix, l.key_prefix, json_get, json_list)
                });

            prop_assert!(full_key.len() >= 8);
            prop_assert_eq!(&get_prefix, &full_key[..8]);
            prop_assert_eq!(&list_prefix, &full_key[..8]);
            // The full secret must never leak into the masked view.
            prop_assert!(!json_get.contains(&full_key));
            prop_assert!(!json_list.contains(&full_key));
        }

        // Feature: virtual-key-management, Property 16: Pagination Correctness
        //
        // For any collection of N keys and any (limit, offset), the returned
        // page contains exactly min(effective_limit, N - offset) keys and the
        // total equals N. The manager clamps limit to <=100 and treats limit==0
        // as the default 50, so the expectation uses the effective limit.
        //
        // **Validates: Requirements 10.5**

        /// Page length matches the effective-limit arithmetic and total == N.
        #[test]
        fn prop_pagination_correctness(
            n in 0usize..=25,
            limit in 1u32..=200,
            offset in 0u32..=25,
        ) {
            let (page_len, total) = rt().block_on(async {
                let (mgr, _tmp) = manager();
                for _ in 0..n {
                    mgr.create_key(create_defaults()).await.unwrap();
                }
                let page = mgr
                    .list_keys(ListKeysParams { limit, offset })
                    .await
                    .unwrap();
                (page.keys.len(), page.total)
            });

            let effective = if limit == 0 { 50 } else { limit.min(100) };
            let remaining = (n as u32).saturating_sub(offset);
            let expected = effective.min(remaining) as usize;

            prop_assert_eq!(page_len, expected);
            prop_assert_eq!(total, n as u64);
        }
    }
}
