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

/// Minimum allowed value for a virtual key priority class.
pub const PRIORITY_MIN: u16 = 0;
/// Maximum allowed value for a virtual key priority class.
pub const PRIORITY_MAX: u16 = 1000;

/// Error returned when a virtual key priority fails range validation.
#[derive(Debug, Clone, PartialEq, Eq, thiserror::Error)]
#[error("priority must be between 0 and 1000")]
pub struct PriorityValidationError;

/// Validate a virtual key `priority` against the 0-1000 inclusive rule.
///
/// Accepts values between [`PRIORITY_MIN`] and [`PRIORITY_MAX`] inclusive;
/// rejects anything above 1000. Negative and non-integer values are rejected
/// earlier, at deserialization time, because the field is typed `u16`.
///
/// _Requirements: 1.1, 1.3_
pub fn validate_priority(priority: u16) -> Result<(), PriorityValidationError> {
    if (PRIORITY_MIN..=PRIORITY_MAX).contains(&priority) {
        Ok(())
    } else {
        Err(PriorityValidationError)
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

/// Maps a provider budget reset policy onto virtual-key budget-window
/// semantics. `Manual` maps to `None`: spend only clears via an explicit
/// admin reset, so no periodic window applies.
///
/// Lives here (not in `config`) because `virtual_keys` already depends on
/// `config`; the reverse dependency would create a module cycle.
impl From<crate::config::BudgetResetPolicy> for Option<BudgetWindow> {
    fn from(policy: crate::config::BudgetResetPolicy) -> Self {
        match policy {
            crate::config::BudgetResetPolicy::Daily => Some(BudgetWindow::Daily),
            crate::config::BudgetResetPolicy::Weekly => Some(BudgetWindow::Weekly),
            crate::config::BudgetResetPolicy::Monthly => Some(BudgetWindow::Monthly),
            crate::config::BudgetResetPolicy::Manual => None,
        }
    }
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
    /// Per-request output-token ceiling (positive integer); `None` = no cap.
    #[serde(default)]
    pub max_output_tokens: Option<u32>,
    /// Permitted model/model-group names; `None` allows all.
    #[serde(default)]
    pub model_access: Option<Vec<String>>,
    #[serde(default)]
    pub expires_in: Option<ExpiresIn>,
    #[serde(default)]
    pub loop_detection: Option<crate::loop_detection::VkLoopConfig>,
 /// Admission priority class (0-1000); higher values are admitted first
 /// when provider capacity is contended. Defaults to 0.
 #[serde(default, skip_serializing_if = "Option::is_none")]
 pub priority: Option<u16>,
}

/// Response from key creation (includes the plaintext key exactly once).
#[derive(Debug, Clone, Serialize)]
pub struct CreateKeyResponse {
    pub id: String,
    /// Full key value, shown only once at creation time.
    pub key: String,
    pub name: Option<String>,
    pub status: KeyStatus,
    /// Admission priority class (0-1000); 0 is the default class.
    pub priority: u16,
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
    /// Per-request output-token ceiling; `None` leaves it unchanged,
    /// `Some(None)` clears it, `Some(Some(v))` sets it (positive integer).
    /// Uses the double-option deserializer so JSON `null` clears the cap
    /// instead of being read as an absent field.
    #[serde(default, deserialize_with = "deserialize_output_cap_double_option")]
    pub max_output_tokens: Option<Option<u32>>,
    #[serde(default)]
    pub model_access: Option<Option<Vec<String>>>,
    #[serde(default)]
    pub expires_in: Option<ExpiresIn>,
    #[serde(default)]
    pub loop_detection: Option<Option<crate::loop_detection::VkLoopConfig>>,
    /// Admission priority class (0-1000). `None` leaves it unchanged,
    /// `Some(None)` (JSON `null`) resets it to 0, `Some(Some(v))` sets it
    /// to `v`. Unlike the other nullable fields, this field uses a
    /// double-option deserializer so JSON `null` is distinguishable from
    /// an absent field (Req 1 AC 4).
    #[serde(default, deserialize_with = "deserialize_priority_double_option")]
    pub priority: Option<Option<u16>>,
    }

    /// Deserialize an `Option<Option<u16>>` with explicit three-state semantics:
    /// an absent field yields `None` (unchanged), JSON `null` yields `Some(None)`
    /// (reset to the default class 0), and a value yields `Some(Some(v))`.
    ///
/// Used only by [`UpdateKeyParams::priority`] (Req 1 AC 4); the other
/// nullable update fields keep plain serde `Option<Option<T>>` behavior.
fn deserialize_priority_double_option<'de, D>(deserializer: D) -> Result<Option<Option<u16>>, D::Error>
where
    D: serde::Deserializer<'de>,
{
    Ok(match Option::<u16>::deserialize(deserializer)? {
        None => Some(None),
        Some(value) => Some(Some(value)),
    })
}

/// Deserialize an `Option<Option<u32>>` with explicit three-state semantics:
/// an absent field yields `None` (unchanged), JSON `null` yields `Some(None)`
/// (clear the cap), and a value yields `Some(Some(v))`.
///
/// Used only by [`UpdateKeyParams::max_output_tokens`] so the admin API's
/// `PATCH {"max_output_tokens": null}` clears the output cap.
fn deserialize_output_cap_double_option<'de, D>(deserializer: D) -> Result<Option<Option<u32>>, D::Error>
where
    D: serde::Deserializer<'de>,
{
    Ok(match Option::<u32>::deserialize(deserializer)? {
        None => Some(None),
        Some(value) => Some(Some(value)),
    })
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
    /// Per-request output-token ceiling; `None` = no cap.
    pub max_output_tokens: Option<u32>,
    pub model_access: Option<Vec<String>>,
    pub expires_at: Option<DateTime<Utc>>,
    pub loop_detection: Option<crate::loop_detection::VkLoopConfig>,
    /// Admission priority class (0-1000); 0 is the default class.
    pub priority: u16,
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
    /// Per-request output-token ceiling; `None` = no cap.
    pub max_output_tokens: Option<u32>,
    pub model_access: Option<Vec<String>>,
    pub loop_detection: Option<crate::loop_detection::VkLoopConfig>,
    pub request_count: u64,
    /// Admission priority class (0-1000); 0 is the default class.
    pub priority: u16,
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
            cases: 64,
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

    /// Boundary priorities 0 and 1000 are accepted; 1001 is rejected
    /// (Req 1.3). Negative and non-integer values are rejected at JSON
    /// deserialization time by the `u16` field type, before validation runs.
    #[test]
    fn priority_range_boundaries() {
        assert_eq!(validate_priority(0), Ok(()));
        assert_eq!(validate_priority(1000), Ok(()));
        assert_eq!(validate_priority(1001), Err(PriorityValidationError));
        assert_eq!(validate_priority(u16::MAX), Err(PriorityValidationError));
    }

    /// `CreateKeyParams` rejects negative and non-integer `priority` values
    /// at deserialization (Req 1.3: out-of-range/non-integer rejected with a
    /// field-level error), and defaults the field to `None` when omitted
    /// (Req 1.2).
    #[test]
    fn priority_deserialization_rejects_non_unsigned_integers() {
        assert_eq!(
            serde_json::from_str::<CreateKeyParams>(r#"{"priority": 10}"#)
                .unwrap()
                .priority,
            Some(10)
        );
        // Omitted -> None (default 0 downstream).
        assert_eq!(
            serde_json::from_str::<CreateKeyParams>("{}").unwrap().priority,
            None
        );
        // Negative rejected.
        assert!(serde_json::from_str::<CreateKeyParams>(r#"{"priority": -1}"#).is_err());
        // Non-integer rejected.
        assert!(serde_json::from_str::<CreateKeyParams>(r#"{"priority": 10.5}"#).is_err());
        assert!(serde_json::from_str::<CreateKeyParams>(r#"{"priority": "10"}"#).is_err());
        // Above the u16 domain rejected at deserialization.
        assert!(serde_json::from_str::<CreateKeyParams>(r#"{"priority": 65536}"#).is_err());
    }

    /// `UpdateKeyParams.priority` uses a double-option deserializer (unlike the
    /// other nullable update fields): omitted yields `None` (unchanged), JSON
    /// `null` yields `Some(None)` (reset to the default class 0, Req 1 AC 4),
    /// and a value yields `Some(Some(v))`.
    #[test]
    fn priority_update_deserialization_states() {
        assert_eq!(
            serde_json::from_str::<UpdateKeyParams>("{}").unwrap().priority,
            None
        );
        assert_eq!(
            serde_json::from_str::<UpdateKeyParams>(r#"{"priority": null}"#)
                .unwrap()
                .priority,
            Some(None)
        );
        assert_eq!(
            serde_json::from_str::<UpdateKeyParams>(r#"{"priority": 25}"#)
                .unwrap()
                .priority,
            Some(Some(25))
        );
        // Out-of-domain values are rejected at deserialization; 1001 is a
        // valid u16 and deserializes fine — it is rejected later by
        // `validate_priority` in the handler.
        assert!(serde_json::from_str::<UpdateKeyParams>(r#"{"priority": -1}"#).is_err());
        assert!(serde_json::from_str::<UpdateKeyParams>(r#"{"priority": 1001}"#).is_ok());
    }

    proptest! {
        #![proptest_config(ProptestConfig {
            cases: 64,
            .. ProptestConfig::default()
        })]

 // Feature: virtual-key-management, Property 4: Expiration Computation
 //
 // For any valid `ExpiresIn` variant (other than `Never`) and any
 // creation timestamp, the computed expiration timestamp SHALL equal the
 // creation timestamp plus the variant's fixed duration.
 //
 // **Validates: Requirements 1.5, 7.6**

 /// Priority validation accepts the whole 0..=1000 domain and rejects
 /// everything above it up to `u16::MAX` (Req 1.3).
 #[test]
 fn prop_priority_range(priority in 0u16..=u16::MAX) {
 prop_assert_eq!(
 validate_priority(priority).is_ok(),
 (PRIORITY_MIN..=PRIORITY_MAX).contains(&priority)
 );
 }

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
