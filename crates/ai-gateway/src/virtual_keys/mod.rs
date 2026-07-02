//! Virtual key management: caller authentication and usage governance.
//!
//! This module owns the full lifecycle of gateway-issued API keys: generation,
//! authentication, constraint enforcement (budgets, rate limits, model access),
//! usage tracking, and administrative CRUD.
//!
//! Submodules are added incrementally by later tasks (`store`, `auth`,
//! `budget`, `rate_limiter`, `usage`, `admin`). Task 1.1 establishes the module
//! structure, the data models, and the [`VirtualKeyManager`] skeleton.

pub mod access;
pub mod admin;
pub mod auth;
pub mod budget;
pub mod errors;
pub mod lifecycle;
pub mod models;
pub mod rate_limiter;
pub mod store;
pub mod usage;

use std::path::Path;
use std::sync::{Arc, Mutex};

use base64::engine::general_purpose::URL_SAFE_NO_PAD;
use base64::Engine as _;
use chrono::Utc;
use dashmap::DashMap;
use rand::rngs::OsRng;
use rand::RngCore as _;
use uuid::Uuid;

use crate::secrets;
use models::{
    validate_name, AuthenticatedKey, BudgetWindow, CachedKey, CreateKeyParams, CreateKeyResponse,
    KeyStatus, UsageAggregate, UsageQueryParams, UsageRecord,
};

pub use access::AccessError;
pub use auth::AuthError;
pub use budget::BudgetError;
pub use errors::{FieldError, KeyError, ValidationErrors};
pub use rate_limiter::{PerKeyRateLimiter, RateLimitError, RetryAfterSeconds};
pub use store::{KeyStore, KeyStoreError, KeyUpdates, StoredVirtualKey};
pub use usage::{compute_cost, UsageTracker};

// --- Key generation / validation constants -----------------------------------

/// Number of random bytes used to generate a virtual key (256 bits of entropy).
const KEY_RANDOM_BYTES: usize = 32;
/// Prefix applied to every generated virtual key.
const KEY_PREFIX: &str = "vk_";
/// Number of leading characters retained for masked display.
const KEY_PREFIX_DISPLAY_LEN: usize = 8;

// Validation bounds (design "Validation Rules").
const BUDGET_USD_MIN: f64 = 0.01;
const BUDGET_USD_MAX: f64 = 999_999_999.99;
const TOKEN_BUDGET_MIN: u64 = 1;
const TOKEN_BUDGET_MAX: u64 = 999_999_999;
const RPM_MIN: u32 = 1;
const RPM_MAX: u32 = 100_000;
const TPM_MIN: u64 = 1;
const TPM_MAX: u64 = 10_000_000;

// --- Manager ------------------------------------------------------------------

/// Central coordinator that owns the key store, authentication cache, usage
/// tracker, and per-key rate limiters.
///
/// Task 2.1 implements [`VirtualKeyManager::new`] and
/// [`VirtualKeyManager::create_key`]; remaining methods arrive in later tasks.
pub struct VirtualKeyManager {
    store: Arc<KeyStore>,
    cache: Arc<DashMap<String, CachedKey>>,
    usage_tracker: Arc<UsageTracker>,
    /// Per-key rate limiters, keyed by virtual key id. Interior mutability via
    /// `Mutex` since each limiter mutates on every request (token bucket /
    /// rolling window).
    rate_limiters: Arc<DashMap<String, Arc<Mutex<PerKeyRateLimiter>>>>,
}

impl VirtualKeyManager {
    /// Construct a manager backed by the SQLite key store at `db_path`.
    ///
    /// Opens (or creates) the store, initializes the authentication cache, and
    /// installs placeholder usage-tracker / rate-limiter collaborators that
    /// later tasks replace with full implementations.
    pub fn new(db_path: &Path) -> Result<Self, KeyStoreError> {
        let store = Arc::new(KeyStore::new(db_path)?);
        let usage_tracker = Arc::new(UsageTracker::new(Arc::clone(&store)));
        Ok(Self {
            store,
            cache: Arc::new(DashMap::new()),
            usage_tracker,
            rate_limiters: Arc::new(DashMap::new()),
        })
    }

    /// Generate, encrypt, and persist a new virtual key.
    ///
    /// Validates all constraint fields, generates a cryptographically secure
    /// key value (`vk_` + 43 base64url chars), stores its SHA-256 hash and an
    /// encrypted copy of the value, and returns the plaintext key exactly once.
    ///
    /// _Requirements: 1.1, 1.2, 1.3, 1.4, 1.5, 1.6, 1.7, 1.8, 1.9_
    pub async fn create_key(
        &self,
        params: CreateKeyParams,
    ) -> Result<CreateKeyResponse, KeyError> {
        validate_create_params(&params)?;

        // 1-4. Generate `vk_<43 base64url chars>` from 32 secure random bytes.
        let full_key = generate_key_value();
        // 5. SHA-256 hash for lookup indexing (hex).
        let key_hash = sha256_hex(&full_key);
        // 6. Masked prefix: first 8 chars of the full key (incl. `vk_`).
        let key_prefix: String = full_key.chars().take(KEY_PREFIX_DISPLAY_LEN).collect();
        // 7. Encrypt the full value at rest. Never log the plaintext.
        let encrypted_key = secrets::encrypt_provider_secret(&full_key)
            .map_err(|e| KeyError::Encryption(e.to_string()))?;

        let now = Utc::now();
        // Expiration: `None` (field omitted) and `Some(Never)` both yield no
        // expiration; other variants add their fixed duration to `now`.
        let expires_at = params
            .expires_in
            .as_ref()
            .and_then(|e| e.to_duration())
            .map(|d| now + d);
        // Window start is only meaningful when a budget window is configured.
        let window_start = params.budget_window.as_ref().map(|_: &BudgetWindow| now);

        let stored = StoredVirtualKey {
            id: Uuid::new_v4().to_string(),
            key_hash,
            key_prefix,
            encrypted_key,
            name: params.name.clone(),
            status: KeyStatus::Active,
            budget_limit_usd: params.budget_limit_usd,
            token_budget: params.token_budget,
            budget_window: params.budget_window.clone(),
            current_spend_usd: 0.0,
            current_tokens_used: 0,
            window_start,
            requests_per_minute: params.requests_per_minute,
            tokens_per_minute: params.tokens_per_minute,
            model_access_list: params.model_access.clone(),
            expires_at,
            created_at: now,
            last_used_at: None,
            request_count: 0,
        };

        // Persist. On failure, no partial record exists (single INSERT).
        self.store.create_key(&stored)?;

        Ok(CreateKeyResponse {
            id: stored.id,
            key: full_key,
            name: stored.name,
            status: KeyStatus::Active,
            created_at: now,
            expires_at,
        })
    }

    /// Remove a cached authentication entry by its SHA-256 hash.
    ///
    /// Used by update/revoke/delete flows (later tasks) so that mutated keys
    /// are re-validated against the store on their next request rather than
    /// served from a stale cache entry.
    pub fn invalidate_cache(&self, key_hash: &str) {
        self.cache.remove(key_hash);
    }

    /// Compute the lookup hash for a raw key value.
    ///
    /// Exposed so callers holding the plaintext key (e.g. revoke/delete paths)
    /// can derive the cache key to pass to [`Self::invalidate_cache`].
    pub fn hash_key(raw_key: &str) -> String {
        sha256_hex(raw_key)
    }

    /// Enforce the per-key rate limits (RPM then TPM) for an authenticated key.
    ///
    /// Gets or creates the key's [`PerKeyRateLimiter`] (keyed by `key.id`) from
    /// the key's `requests_per_minute` / `tokens_per_minute` constraints, then:
    /// checks the RPM token bucket, checks the TPM rolling window, and — only
    /// on success — consumes one RPM token. Returns [`RateLimitError`] carrying
    /// the `Retry-After` seconds when either limit is exceeded.
    ///
    /// _Requirements: 5.1, 5.2, 5.3, 5.4, 5.5_
    pub fn check_rate_limit(&self, key: &AuthenticatedKey) -> Result<(), RateLimitError> {
        // Fast path: no limits configured means nothing to track.
        if key.requests_per_minute.is_none() && key.tokens_per_minute.is_none() {
            return Ok(());
        }

        let limiter = self
            .rate_limiters
            .entry(key.id.clone())
            .or_insert_with(|| {
                Arc::new(Mutex::new(PerKeyRateLimiter::new(
                    key.requests_per_minute,
                    key.tokens_per_minute,
                )))
            })
            .clone();

        // A poisoned lock only means a prior panic while holding it; recover the
        // guard so rate limiting stays available rather than propagating panics.
        let mut guard = limiter.lock().unwrap_or_else(|e| e.into_inner());

        // Check RPM without consuming, then TPM; consume an RPM token only when
        // both checks pass so a TPM rejection does not burn request budget.
        guard
            .check_rpm()
            .map_err(|retry_after_seconds| RateLimitError::RpmExceeded {
                retry_after_seconds,
            })?;
        guard
            .check_tpm()
            .map_err(|retry_after_seconds| RateLimitError::TpmExceeded {
                retry_after_seconds,
            })?;
        guard.consume_rpm();
        Ok(())
    }

    /// Record post-response token consumption into a key's TPM rolling window.
    ///
    /// Called after a provider response with the actual `input + output` token
    /// count so the rolling 60-second window reflects real usage (Req 5.3). No
    /// effect when the key has no limiter (no RPM/TPM configured) or no TPM
    /// window.
    pub fn record_tpm_usage(&self, key_id: &str, tokens: u64) {
        if let Some(limiter) = self.rate_limiters.get(key_id) {
            let limiter = limiter.clone();
            let mut guard = limiter.lock().unwrap_or_else(|e| e.into_inner());
            guard.record_tpm(tokens);
        }
    }

    /// Record a completed request's usage into the key store.
    ///
    /// Inserts the usage row and advances the key's cumulative counters
    /// (spend, tokens, request count, last-used) via the [`UsageTracker`]. The
    /// caller supplies a fully-formed [`UsageRecord`]; deciding whether to skip
    /// recording on missing provider usage (Req 3.6) or to record an estimate
    /// (Req 4.5) is the caller's responsibility.
    ///
    /// _Requirements: 3.5, 4.4, 9.1, 9.6_
    pub async fn record_usage(&self, record: UsageRecord) -> Result<(), KeyError> {
        self.usage_tracker.record(record)?;
        Ok(())
    }

    /// Aggregate a key's usage over the inclusive `[start, end]` range.
    ///
    /// Returns [`KeyError::NotFound`] when `id` does not identify a stored key
    /// (Req 9.3); otherwise returns summed spend/token totals and request count,
    /// with zero values when no records fall in the range (Req 9.2, 9.4).
    ///
    /// _Requirements: 9.2, 9.3, 9.4_
    pub async fn query_usage(
        &self,
        id: &str,
        params: UsageQueryParams,
    ) -> Result<UsageAggregate, KeyError> {
        // Distinguish "unknown key" (404) from "known key, no usage" (zeros).
        if self.store.get_key_by_id(id)?.is_none() {
            return Err(KeyError::NotFound(id.to_string()));
        }
        let aggregate = self
            .usage_tracker
            .query_aggregate(id, params.start, params.end)?;
        Ok(aggregate)
    }
}

/// Generate a virtual key value: `vk_` followed by 43 URL-safe base64
/// characters derived from 32 cryptographically secure random bytes.
fn generate_key_value() -> String {
    let mut bytes = [0u8; KEY_RANDOM_BYTES];
    OsRng.fill_bytes(&mut bytes);
    let encoded = URL_SAFE_NO_PAD.encode(bytes);
    format!("{KEY_PREFIX}{encoded}")
}

/// Compute the lowercase hex SHA-256 digest of `input`.
pub(crate) fn sha256_hex(input: &str) -> String {
    use std::fmt::Write as _;
    let digest = ring::digest::digest(&ring::digest::SHA256, input.as_bytes());
    let mut out = String::with_capacity(digest.as_ref().len() * 2);
    for byte in digest.as_ref() {
        // Writing to a String is infallible.
        let _ = write!(out, "{byte:02x}");
    }
    out
}

/// Validate all [`CreateKeyParams`] fields per the design "Validation Rules".
/// Accumulates every failure so callers receive a complete field-error set.
fn validate_create_params(params: &CreateKeyParams) -> Result<(), KeyError> {
    let mut errors = ValidationErrors::new();

    if let Some(name) = &params.name {
        if validate_name(name).is_err() {
            errors.push("name", "name must be between 1 and 128 characters");
        }
    }

    if let Some(budget) = params.budget_limit_usd {
        if !budget.is_finite() || !(BUDGET_USD_MIN..=BUDGET_USD_MAX).contains(&budget) {
            errors.push(
                "budget_limit_usd",
                "budget must be between 0.01 and 999999999.99",
            );
        }
    }

    if let Some(token_budget) = params.token_budget {
        if !(TOKEN_BUDGET_MIN..=TOKEN_BUDGET_MAX).contains(&token_budget) {
            errors.push(
                "token_budget",
                "token_budget must be between 1 and 999999999",
            );
        }
    }

    if let Some(rpm) = params.requests_per_minute {
        if !(RPM_MIN..=RPM_MAX).contains(&rpm) {
            errors.push(
                "requests_per_minute",
                "requests_per_minute must be between 1 and 100000",
            );
        }
    }

    if let Some(tpm) = params.tokens_per_minute {
        if !(TPM_MIN..=TPM_MAX).contains(&tpm) {
            errors.push(
                "tokens_per_minute",
                "tokens_per_minute must be between 1 and 10000000",
            );
        }
    }

    if let Some(model_access) = &params.model_access {
        if model_access.is_empty() {
            errors.push("model_access", "model_access must be a non-empty list");
        }
    }

    errors.into_result().map_err(KeyError::Validation)
}

#[cfg(test)]
mod tests {
    use super::*;
    use models::{BudgetWindow, ExpiresIn};
    use tempfile::NamedTempFile;

    fn temp_manager() -> (VirtualKeyManager, NamedTempFile) {
        let temp = NamedTempFile::new().unwrap();
        let mgr = VirtualKeyManager::new(temp.path()).unwrap();
        (mgr, temp)
    }

    fn defaults() -> CreateKeyParams {
        CreateKeyParams {
            name: None,
            budget_limit_usd: None,
            token_budget: None,
            budget_window: None,
            requests_per_minute: None,
            tokens_per_minute: None,
            model_access: None,
            expires_in: None,
        }
    }

    /// Req 1.6: omitting all optional fields yields an active key with no
    /// limits, no expiration, and all-model access, persisted and retrievable.
    #[tokio::test]
    async fn create_key_with_all_defaults() {
        let (mgr, _tmp) = temp_manager();

        let resp = mgr.create_key(defaults()).await.unwrap();

        // Key format: `vk_` + 43 base64url chars == 46 chars (Req 1.2).
        assert!(resp.key.starts_with("vk_"));
        assert_eq!(resp.key.len(), 46);
        assert_eq!(resp.status, KeyStatus::Active);
        assert!(resp.name.is_none());
        assert!(resp.expires_at.is_none());

        // Persisted with unlimited constraints (Req 1.6) and encrypted value
        // that is not the plaintext (Req 1.3).
        let stored = mgr.store.get_key_by_id(&resp.id).unwrap().unwrap();
        assert_eq!(stored.status, KeyStatus::Active);
        assert!(stored.budget_limit_usd.is_none());
        assert!(stored.token_budget.is_none());
        assert!(stored.budget_window.is_none());
        assert!(stored.requests_per_minute.is_none());
        assert!(stored.tokens_per_minute.is_none());
        assert!(stored.model_access_list.is_none());
        assert!(stored.expires_at.is_none());
        assert!(stored.window_start.is_none());
        assert_eq!(stored.current_spend_usd, 0.0);
        assert_eq!(stored.current_tokens_used, 0);
        assert_eq!(stored.request_count, 0);
        assert_eq!(stored.key_prefix, &resp.key[..8]);
        assert_ne!(stored.encrypted_key, resp.key);

        // Hash lookup resolves to the same record.
        let by_hash = mgr
            .store
            .get_key_by_hash(&sha256_hex(&resp.key))
            .unwrap()
            .unwrap();
        assert_eq!(by_hash.id, resp.id);

        // Encrypted value decrypts back to the plaintext (Req 1.3).
        let decrypted = secrets::decrypt_provider_secret(&stored.encrypted_key).unwrap();
        assert_eq!(decrypted, resp.key);
    }

    /// Req 1.5: a duration `expires_in` records `created_at + duration`.
    #[tokio::test]
    async fn create_key_sets_expiration_and_window_start() {
        let (mgr, _tmp) = temp_manager();
        let params = CreateKeyParams {
            expires_in: Some(ExpiresIn::OneDay),
            budget_window: Some(BudgetWindow::Daily),
            ..defaults()
        };

        let resp = mgr.create_key(params).await.unwrap();
        let expires_at = resp.expires_at.expect("expiration set");
        assert_eq!(expires_at - resp.created_at, chrono::Duration::days(1));

        let stored = mgr.store.get_key_by_id(&resp.id).unwrap().unwrap();
        assert!(stored.window_start.is_some());
        assert_eq!(stored.budget_window, Some(BudgetWindow::Daily));
    }

    /// Req 1.8 / validation rules: out-of-range fields produce per-field errors
    /// and no record is persisted.
    #[tokio::test]
    async fn create_key_validation_failure_reports_fields() {
        let (mgr, _tmp) = temp_manager();
        let params = CreateKeyParams {
            name: Some(String::new()),          // too short
            budget_limit_usd: Some(0.0),        // below 0.01
            token_budget: Some(0),              // below 1
            requests_per_minute: Some(0),       // below 1
            tokens_per_minute: Some(20_000_000), // above 10_000_000
            model_access: Some(vec![]),         // empty list
            ..defaults()
        };

        let err = mgr.create_key(params).await.unwrap_err();
        let KeyError::Validation(errors) = err else {
            panic!("expected validation error, got {err:?}");
        };

        let fields: Vec<&str> = errors.errors.iter().map(|e| e.field.as_str()).collect();
        assert!(fields.contains(&"name"));
        assert!(fields.contains(&"budget_limit_usd"));
        assert!(fields.contains(&"token_budget"));
        assert!(fields.contains(&"requests_per_minute"));
        assert!(fields.contains(&"tokens_per_minute"));
        assert!(fields.contains(&"model_access"));

        // Nothing persisted on validation failure.
        let (page, total) = mgr.store.list_keys(10, 0).unwrap();
        assert_eq!(total, 0);
        assert!(page.is_empty());
    }

    /// Two successive generations produce distinct keys, hashes, and ids.
    #[tokio::test]
    async fn create_key_generates_unique_values() {
        let (mgr, _tmp) = temp_manager();
        let a = mgr.create_key(defaults()).await.unwrap();
        let b = mgr.create_key(defaults()).await.unwrap();
        assert_ne!(a.key, b.key);
        assert_ne!(a.id, b.id);
    }
}

#[cfg(test)]
mod property_tests {
    use super::*;
    use proptest::prelude::*;

    /// Returns true iff every character is a URL-safe base64 character
    /// (`A-Z`, `a-z`, `0-9`, `-`, `_`).
    fn is_url_safe_base64(s: &str) -> bool {
        s.chars()
            .all(|c| c.is_ascii_alphanumeric() || c == '-' || c == '_')
    }

    proptest! {
        #![proptest_config(ProptestConfig {
            cases: 100,
            .. ProptestConfig::default()
        })]

        // Feature: virtual-key-management, Property 1: Key Generation Produces Valid Keys
        //
        // For any invocation of the key generation function, the resulting key
        // value SHALL be at least 43 characters long, contain only URL-safe
        // base64 characters (A-Z, a-z, 0-9, -, _), and be prefixed with `vk_`.
        //
        // **Validates: Requirements 1.2**

        /// `generate_key_value()` takes no input; the `_seed` argument only
        /// drives proptest to invoke generation across many independent cases,
        /// exercising the underlying `OsRng` randomness. The invariant must hold
        /// every case.
        #[test]
        fn prop_key_generation_produces_valid_keys(_seed in any::<u32>()) {
            let key = generate_key_value();

            // Spec minimum length is 43 (actual is 46: `vk_` + 43 chars).
            prop_assert!(
                key.len() >= 43,
                "key too short: len={} key={:?}",
                key.len(),
                key
            );

            // Must be prefixed with `vk_`.
            prop_assert!(key.starts_with("vk_"), "missing vk_ prefix: {:?}", key);

            // The portion after `vk_` must be URL-safe base64 only.
            let body = &key["vk_".len()..];
            prop_assert!(
                is_url_safe_base64(body),
                "non-url-safe chars in key body: {:?}",
                body
            );
        }
    }

    proptest! {
        #![proptest_config(ProptestConfig {
            cases: 100,
            .. ProptestConfig::default()
        })]

        // Feature: virtual-key-management, Property 2: Key Encryption Round-Trip
        //
        // For any generated virtual key value, encrypting it with the secrets
        // module and then decrypting the ciphertext SHALL produce the original
        // key value.
        //
        // **Validates: Requirements 1.3**

        /// `decrypt_provider_secret(encrypt_provider_secret(key)) == key` for
        /// freshly generated keys. The `_seed` argument only drives repeated
        /// proptest cases over distinct generated keys.
        #[test]
        fn prop_key_encryption_round_trip(_seed in any::<u32>()) {
            let key = generate_key_value();

            let ciphertext = secrets::encrypt_provider_secret(&key)
                .expect("encryption should succeed");
            // Ciphertext must not be the plaintext.
            prop_assert_ne!(&ciphertext, &key);

            let decrypted = secrets::decrypt_provider_secret(&ciphertext)
                .expect("decryption should succeed");
            prop_assert_eq!(decrypted, key);
        }
    }
}
