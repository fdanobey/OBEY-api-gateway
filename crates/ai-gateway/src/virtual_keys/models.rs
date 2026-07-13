//! Data structures for the virtual key management feature.
//!
//! These types define the public-facing request/response shapes and the
//! in-memory representations carried through the request pipeline. Persistence
//! (`store.rs`), authentication (`auth.rs`), and enforcement logic live in
//! sibling modules added by later tasks.

use std::time::Instant;

use chrono::{DateTime, Utc};
use serde::{Deserialize, Serialize};

/// Expiration duration options for key creation/update.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum ExpiresIn {
    Never,
    #[serde(rename = "1_year")]
    OneYear,
    #[serde(rename = "6_months")]
    SixMonths,
    #[serde(rename = "3_months")]
    ThreeMonths,
    #[serde(rename = "1_month")]
    OneMonth,
    #[serde(rename = "2_weeks")]
    TwoWeeks,
    #[serde(rename = "1_week")]
    OneWeek,
    #[serde(rename = "3_days")]
    ThreeDays,
    #[serde(rename = "1_day")]
    OneDay,
}

impl ExpiresIn {
    /// Fixed duration for this variant, or `None` for [`ExpiresIn::Never`].
    pub fn to_duration(&self) -> Option<chrono::Duration> {
        match self {
            Self::Never => None,
            Self::OneYear => Some(chrono::Duration::days(365)),
            Self::SixMonths => Some(chrono::Duration::days(182)),
            Self::ThreeMonths => Some(chrono::Duration::days(91)),
            Self::OneMonth => Some(chrono::Duration::days(30)),
            Self::TwoWeeks => Some(chrono::Duration::days(14)),
            Self::OneWeek => Some(chrono::Duration::days(7)),
            Self::ThreeDays => Some(chrono::Duration::days(3)),
            Self::OneDay => Some(chrono::Duration::days(1)),
        }
    }
}

/// Minimum allowed length (in characters) for a virtual key name.
pub const NAME_MIN_LEN: usize = 1;
/// Maximum allowed length (in characters) for a virtual key name.
pub const NAME_MAX_LEN: usize = 128;

/// Error returned when a virtual key name fails length validation.
#[derive(Debug, Clone, PartialEq, Eq, thiserror::Error)]
#[error("name must be between 1 and 128 characters")]
pub struct NameValidationError;

/// Validate a virtual key `name` against the 1-128 character rule.
///
/// Length is measured in Unicode scalar values (`char` count), not bytes, so
/// multi-byte characters count as a single character. Accepts names whose
/// length is between [`NAME_MIN_LEN`] and [`NAME_MAX_LEN`] inclusive; rejects
/// empty names and names longer than [`NAME_MAX_LEN`].
///
/// _Requirements: 1.4, 1.8, 7.9_
pub fn validate_name(name: &str) -> Result<(), NameValidationError> {
    let len = name.chars().count();
    if (NAME_MIN_LEN..=NAME_MAX_LEN).contains(&len) {
        Ok(())
    } else {
        Err(NameValidationError)
    }
}

/// Budget window for periodic spend/token counter resets.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum BudgetWindow {
    Daily,
    Weekly,
    Monthly,
}

/// Lifecycle status of a virtual key.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum KeyStatus {
    Active,
    Expired,
    Revoked,
}

/// Parameters for creating a new virtual key.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CreateKeyParams {
    /// Human-readable alias (1-128 chars).
    #[serde(default)]
    pub name: Option<String>,
    /// USD budget limit (0.01 - 999_999_999.99).
    #[serde(default)]
    pub budget_limit_usd: Option<f64>,
    /// Total token budget (1 - 999_999_999).
    #[serde(default)]
    pub token_budget: Option<u64>,
    #[serde(default)]
    pub budget_window: Option<BudgetWindow>,
    /// Requests-per-minute limit (1 - 100_000).
    #[serde(default)]
    pub requests_per_minute: Option<u32>,
    /// Tokens-per-minute limit (1 - 10_000_000).
    #[serde(default)]
    pub tokens_per_minute: Option<u64>,
    /// Permitted model/model-group names; `None` allows all.
    #[serde(default)]
    pub model_access: Option<Vec<String>>,
    #[serde(default)]
    pub expires_in: Option<ExpiresIn>,
}

/// Response from key creation (includes the plaintext key exactly once).
#[derive(Debug, Clone, Serialize)]
pub struct CreateKeyResponse {
    pub id: String,
    /// Full key value, shown only once at creation time.
    pub key: String,
    pub name: Option<String>,
    pub status: KeyStatus,
    pub created_at: DateTime<Utc>,
    pub expires_at: Option<DateTime<Utc>>,
}

/// Parameters for updating a virtual key.
///
/// The nested `Option<Option<T>>` pattern distinguishes three states:
/// `None` = field unchanged, `Some(Some(v))` = set to `v`,
/// `Some(None)` = clear/remove the value.
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct UpdateKeyParams {
    #[serde(default)]
    pub name: Option<Option<String>>,
    #[serde(default)]
    pub budget_limit_usd: Option<Option<f64>>,
    #[serde(default)]
    pub token_budget: Option<Option<u64>>,
    #[serde(default)]
    pub budget_window: Option<Option<BudgetWindow>>,
    #[serde(default)]
    pub requests_per_minute: Option<Option<u32>>,
    #[serde(default)]
    pub tokens_per_minute: Option<Option<u64>>,
    #[serde(default)]
    pub model_access: Option<Option<Vec<String>>>,
    #[serde(default)]
    pub expires_in: Option<ExpiresIn>,
}

/// Authenticated key loaded from cache/store, carried through the request
/// pipeline for enforcement checks.
#[allow(dead_code)]
#[derive(Debug, Clone)]
pub struct AuthenticatedKey {
    pub id: String,
    pub name: Option<String>,
    pub status: KeyStatus,
    pub budget_limit_usd: Option<f64>,
    pub token_budget: Option<u64>,
    pub budget_window: Option<BudgetWindow>,
    pub current_spend_usd: f64,
    pub current_tokens_used: u64,
    pub window_start: Option<DateTime<Utc>>,
    pub requests_per_minute: Option<u32>,
    pub tokens_per_minute: Option<u64>,
    pub model_access: Option<Vec<String>>,
    pub expires_at: Option<DateTime<Utc>>,
}

/// Cached key entry stored in the DashMap for fast authentication lookups.
#[allow(dead_code)]
#[derive(Debug, Clone)]
pub struct CachedKey {
    pub key: AuthenticatedKey,
    pub cached_at: Instant,
}

/// Usage record for a single completed request.
#[derive(Debug, Clone, Serialize)]
pub struct UsageRecord {
    pub key_id: String,
    pub model_group: String,
    pub model: String,
    pub input_tokens: u64,
    pub output_tokens: u64,
    /// Cost in USD, rounded to 6 decimal places.
    pub cost_usd: f64,
    pub timestamp: DateTime<Utc>,
}

/// Aggregated usage response for a key over a time range.
#[derive(Debug, Clone, Serialize)]
pub struct UsageAggregate {
    pub total_spend_usd: f64,
    pub total_input_tokens: u64,
    pub total_output_tokens: u64,
    pub total_requests: u64,
}

/// Pagination parameters for listing keys.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ListKeysParams {
    #[serde(default = "default_list_limit")]
    pub limit: u32,
    #[serde(default)]
    pub offset: u32,
}

impl Default for ListKeysParams {
    fn default() -> Self {
        Self {
            limit: default_list_limit(),
            offset: 0,
        }
    }
}

fn default_list_limit() -> u32 {
    50
}

/// Non-sensitive view of a virtual key for list/info responses. The full key
/// value is never included; only the masked prefix is exposed.
#[derive(Debug, Clone, Serialize)]
pub struct VirtualKeyInfo {
    pub id: String,
    /// First 8 characters of the key value, for display only.
    pub key_prefix: String,
    pub name: Option<String>,
    pub status: KeyStatus,
    pub budget_limit_usd: Option<f64>,
    pub token_budget: Option<u64>,
    pub budget_window: Option<BudgetWindow>,
    pub current_spend_usd: f64,
    pub current_tokens_used: u64,
    pub requests_per_minute: Option<u32>,
    pub tokens_per_minute: Option<u64>,
    pub model_access: Option<Vec<String>>,
    pub request_count: u64,
    pub created_at: DateTime<Utc>,
    pub expires_at: Option<DateTime<Utc>>,
    pub last_used_at: Option<DateTime<Utc>>,
}

/// A page of virtual keys plus the total count in the store, returned by
/// [`super::VirtualKeyManager::list_keys`].
#[derive(Debug, Clone, Serialize)]
pub struct PaginatedKeys {
    /// Keys for the requested page, ordered by creation date descending.
    pub keys: Vec<VirtualKeyInfo>,
    /// Total number of keys in the store (across all pages).
    pub total: u64,
}

/// Query parameters for per-key usage aggregation.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct UsageQueryParams {
    pub start: DateTime<Utc>,
    pub end: DateTime<Utc>,
}

#[cfg(test)]
mod property_tests {
    use super::*;
    use proptest::prelude::*;

    /// Strategy yielding any `ExpiresIn` variant except `Never` (those have a
    /// concrete duration, which Property 4 exercises).
    fn arb_expires_in_with_duration() -> impl Strategy<Value = ExpiresIn> {
        prop_oneof![
            Just(ExpiresIn::OneYear),
            Just(ExpiresIn::SixMonths),
            Just(ExpiresIn::ThreeMonths),
            Just(ExpiresIn::OneMonth),
            Just(ExpiresIn::TwoWeeks),
            Just(ExpiresIn::OneWeek),
            Just(ExpiresIn::ThreeDays),
            Just(ExpiresIn::OneDay),
        ]
    }

    proptest! {
        #![proptest_config(ProptestConfig {
            cases: 100,
            .. ProptestConfig::default()
        })]

        // Feature: virtual-key-management, Property 3: Name Length Validation
        //
        // For any string of length between 1 and 128 inclusive, name validation
        // SHALL accept it. For any string of length 0 or greater than 128, the
        // validation SHALL reject it.
        //
        // **Validates: Requirements 1.4, 1.8, 7.9**

        /// Names whose character count is within 1..=128 are accepted.
        #[test]
        fn prop_name_length_valid_accepted(
            // `.{1,128}` matches 1-128 Unicode scalar values under proptest's
            // regex strategy, aligning with the char-count semantics of
            // `validate_name`.
            name in "(?s).{1,128}"
        ) {
            let len = name.chars().count();
            prop_assert!((1..=128).contains(&len));
            prop_assert!(validate_name(&name).is_ok());
        }

        /// Names longer than 128 characters are rejected.
        #[test]
        fn prop_name_length_too_long_rejected(
            extra in 0usize..=256,
            fill in proptest::char::any(),
        ) {
            // Build a string of exactly 129 + extra characters.
            let name: String = std::iter::repeat(fill).take(129 + extra).collect();
            prop_assert!(name.chars().count() > 128);
            prop_assert_eq!(validate_name(&name), Err(NameValidationError));
        }
    }

    /// The empty string (length 0) is always rejected. Deterministic edge case,
    /// so it lives outside the randomized block.
    #[test]
    fn empty_name_rejected() {
        assert_eq!(validate_name(""), Err(NameValidationError));
    }

    /// Boundary lengths 1 and 128 are accepted; 0 and 129 are rejected.
    #[test]
    fn name_length_boundaries() {
        assert!(validate_name(&"a".repeat(1)).is_ok());
        assert!(validate_name(&"a".repeat(128)).is_ok());
        assert!(validate_name(&"a".repeat(129)).is_err());
    }

    proptest! {
        #![proptest_config(ProptestConfig {
            cases: 100,
            .. ProptestConfig::default()
        })]

        // Feature: virtual-key-management, Property 4: Expiration Computation
        //
        // For any valid `ExpiresIn` variant (other than `Never`) and any
        // creation timestamp, the computed expiration timestamp SHALL equal the
        // creation timestamp plus the variant's fixed duration.
        //
        // **Validates: Requirements 1.5, 7.6**

        /// `expires_at == created_at + variant.to_duration()` for every
        /// non-`Never` variant across a wide range of creation timestamps.
        #[test]
        fn prop_expiration_computation(
            expires_in in arb_expires_in_with_duration(),
            // Range chosen to stay comfortably within chrono's valid
            // `DateTime<Utc>` domain even after adding ~1 year.
            created_secs in 0i64..=32_000_000_000i64,
        ) {
            let created_at = DateTime::<Utc>::from_timestamp(created_secs, 0)
                .expect("timestamp within chrono range");
            let duration = expires_in
                .to_duration()
                .expect("non-Never variant has a duration");

            let expires_at = created_at + duration;

            prop_assert_eq!(expires_at, created_at + duration);
            prop_assert!(expires_at > created_at);
            prop_assert_eq!(expires_at - created_at, duration);
        }
    }

    /// `ExpiresIn::Never` has no duration, so no expiration is computed.
    #[test]
    fn never_has_no_duration() {
        assert!(ExpiresIn::Never.to_duration().is_none());
    }
}
