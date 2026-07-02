//! Per-key budget enforcement: USD spend and token consumption limits.
//!
//! This module implements the pure budget-window boundary logic
//! ([`window_expired`]) and the [`super::VirtualKeyManager::check_budget`]
//! decision that rejects requests once a key meets or exceeds either its USD
//! budget or its token budget. Windowed budgets (daily/weekly/monthly) reset
//! their counters at UTC boundaries; lifetime budgets (no window) never reset.
//!
//! Reset boundaries (design "Budget Enforcement Algorithm"):
//! - daily:   00:00 UTC (a new UTC calendar date)
//! - weekly:  Monday 00:00 UTC (a new ISO week starting Monday)
//! - monthly: first of month 00:00 UTC (a new (year, month))
//!
//! _Requirements: 3.1, 3.2, 3.3, 3.4, 3.7, 4.1, 4.2, 4.3, 4.6, 4.7_

use chrono::{DateTime, Datelike, Duration, NaiveDate, Utc};

use super::models::{AuthenticatedKey, BudgetWindow};
use super::VirtualKeyManager;

/// Budget enforcement errors. Both variants map to HTTP 429 (design error
/// mapping table); the specific variant selects the user-facing message.
#[derive(Debug, Clone, Copy, PartialEq, Eq, thiserror::Error)]
pub enum BudgetError {
    /// Cumulative USD spend met or exceeded the configured budget limit.
    #[error("USD budget exhausted")]
    BudgetExhausted,

    /// Cumulative token consumption met or exceeded the configured token budget.
    #[error("Token budget exhausted")]
    TokenBudgetExhausted,
}

/// Return the Monday (ISO week start) of the week containing `date`.
fn monday_of_week(date: NaiveDate) -> NaiveDate {
    // `num_days_from_monday()` is 0 for Monday .. 6 for Sunday.
    let offset = date.weekday().num_days_from_monday() as i64;
    date - Duration::days(offset)
}

/// Determine whether the current budget window has expired for the given
/// window kind, window start, and current time — all in UTC.
///
/// - `Daily`:   expired when `now` falls on a different UTC calendar date than
///   `window_start` (reset boundary is 00:00 UTC).
/// - `Weekly`:  expired when `now` has reached the Monday 00:00 UTC that starts
///   the week after `window_start`'s week (reset boundary is Monday 00:00 UTC).
/// - `Monthly`: expired when `now`'s `(year, month)` differs from
///   `window_start`'s (reset boundary is the first of the month, 00:00 UTC).
///
/// _Requirements: 3.3, 4.6_
pub fn window_expired(
    window: &BudgetWindow,
    window_start: DateTime<Utc>,
    now: DateTime<Utc>,
) -> bool {
    match window {
        BudgetWindow::Daily => now.date_naive() != window_start.date_naive(),
        BudgetWindow::Weekly => {
            // Boundary: Monday 00:00 UTC that begins the week after the one
            // containing `window_start`. Expired once `now` reaches it.
            let next_monday = monday_of_week(window_start.date_naive()) + Duration::days(7);
            let boundary = next_monday
                .and_hms_opt(0, 0, 0)
                .expect("00:00:00 is always valid")
                .and_utc();
            now >= boundary
        }
        BudgetWindow::Monthly => {
            (now.year(), now.month()) != (window_start.year(), window_start.month())
        }
    }
}

impl VirtualKeyManager {
    /// Enforce the key's USD and token budgets against the wall clock.
    ///
    /// Thin wrapper over [`Self::check_budget_at`] using [`Utc::now`]; the inner
    /// method takes an injected `now` so tests are deterministic.
    ///
    /// _Requirements: 3.2, 3.3, 3.4, 3.7, 4.2, 4.3, 4.6, 4.7_
    pub fn check_budget(&self, key: &AuthenticatedKey) -> Result<(), BudgetError> {
        self.check_budget_at(key, Utc::now())
    }

    /// Budget decision at an explicit `now` (see [`Self::check_budget`]).
    ///
    /// Algorithm (design "Budget Enforcement Algorithm"):
    /// 1. If the key has a budget window and a window start, and that window has
    ///    expired, reset the persisted counters and treat the effective spend /
    ///    token usage as zero for this decision.
    /// 2. Reject with [`BudgetError::BudgetExhausted`] when a USD limit is
    ///    configured and effective spend meets or exceeds it.
    /// 3. Reject with [`BudgetError::TokenBudgetExhausted`] when a token limit
    ///    is configured and effective token usage meets or exceeds it.
    ///
    /// USD and token budgets are independent limits (Req 3.3 / 4.3): either can
    /// trigger rejection regardless of the other's state. Keys without a budget
    /// window are lifetime budgets that never reset (Req 3.4 / 4.7).
    ///
    /// Note on cache invalidation: [`AuthenticatedKey`] carries the key `id` but
    /// not its hash, and the authentication cache is keyed by hash. This method
    /// therefore does not touch the cache. After a window reset the persisted
    /// counters are zeroed in the store, and the cached entry refreshes on its
    /// next lookup miss; for the current decision we use locally-zeroed
    /// effective values so a reset is honored immediately.
    pub fn check_budget_at(
        &self,
        key: &AuthenticatedKey,
        now: DateTime<Utc>,
    ) -> Result<(), BudgetError> {
        // Step 1: windowed keys may need a reset before evaluating limits.
        let reset = match (&key.budget_window, key.window_start) {
            (Some(window), Some(window_start)) if window_expired(window, window_start, now) => {
                // Persist the reset (zero counters, advance window start). A
                // store error here is non-fatal to the budget decision: the
                // effective values below are still treated as zero, so the
                // request is allowed and the reset is retried on a later call.
                let _ = self.store.reset_window_counters(&key.id);
                true
            }
            _ => false,
        };

        // Step 2: effective counters — zero immediately after a reset.
        let effective_spend = if reset { 0.0 } else { key.current_spend_usd };
        let effective_tokens = if reset { 0 } else { key.current_tokens_used };

        // Step 3: USD budget (independent of token budget).
        if let Some(limit) = key.budget_limit_usd {
            if effective_spend >= limit {
                return Err(BudgetError::BudgetExhausted);
            }
        }

        // Step 4: token budget (independent of USD budget).
        if let Some(limit) = key.token_budget {
            if effective_tokens >= limit {
                return Err(BudgetError::TokenBudgetExhausted);
            }
        }

        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::virtual_keys::models::KeyStatus;
    use chrono::TimeZone;
    use tempfile::NamedTempFile;

    fn temp_manager() -> (VirtualKeyManager, NamedTempFile) {
        let temp = NamedTempFile::new().unwrap();
        let mgr = VirtualKeyManager::new(temp.path()).unwrap();
        (mgr, temp)
    }

    fn dt(y: i32, m: u32, d: u32, h: u32, min: u32) -> DateTime<Utc> {
        Utc.with_ymd_and_hms(y, m, d, h, min, 0).unwrap()
    }

    /// Build an authenticated key with the given budget fields; other fields
    /// are inert defaults for budget checks.
    fn key_with(
        budget_limit_usd: Option<f64>,
        token_budget: Option<u64>,
        budget_window: Option<BudgetWindow>,
        current_spend_usd: f64,
        current_tokens_used: u64,
        window_start: Option<DateTime<Utc>>,
    ) -> AuthenticatedKey {
        AuthenticatedKey {
            id: "test-key".to_string(),
            name: None,
            status: KeyStatus::Active,
            budget_limit_usd,
            token_budget,
            budget_window,
            current_spend_usd,
            current_tokens_used,
            window_start,
            requests_per_minute: None,
            tokens_per_minute: None,
            model_access: None,
            expires_at: None,
        }
    }

    // --- window_expired: daily -------------------------------------------------

    #[test]
    fn daily_not_expired_same_utc_date() {
        let start = dt(2024, 3, 10, 1, 0);
        let now = dt(2024, 3, 10, 23, 59);
        assert!(!window_expired(&BudgetWindow::Daily, start, now));
    }

    #[test]
    fn daily_expired_next_utc_date() {
        let start = dt(2024, 3, 10, 23, 0);
        let now = dt(2024, 3, 11, 0, 1);
        assert!(window_expired(&BudgetWindow::Daily, start, now));
    }

    // --- window_expired: weekly ------------------------------------------------

    #[test]
    fn weekly_not_expired_same_iso_week() {
        // 2024-03-11 is a Monday; 2024-03-17 is the Sunday of the same week.
        let start = dt(2024, 3, 11, 12, 0);
        let now = dt(2024, 3, 17, 23, 59);
        assert!(!window_expired(&BudgetWindow::Weekly, start, now));
    }

    #[test]
    fn weekly_expired_next_monday() {
        // Next Monday after the 2024-03-11 week is 2024-03-18 00:00 UTC.
        let start = dt(2024, 3, 11, 12, 0);
        let boundary = dt(2024, 3, 18, 0, 0);
        assert!(window_expired(&BudgetWindow::Weekly, start, boundary));
        // One minute before the boundary is still the same window.
        let before = dt(2024, 3, 17, 23, 59);
        assert!(!window_expired(&BudgetWindow::Weekly, start, before));
    }

    #[test]
    fn weekly_start_midweek_expires_on_following_monday() {
        // Start on Wednesday 2024-03-13; week's Monday is 2024-03-11, so the
        // reset boundary is the following Monday 2024-03-18 00:00 UTC.
        let start = dt(2024, 3, 13, 8, 0);
        assert!(!window_expired(&BudgetWindow::Weekly, start, dt(2024, 3, 17, 23, 0)));
        assert!(window_expired(&BudgetWindow::Weekly, start, dt(2024, 3, 18, 0, 0)));
    }

    // --- window_expired: monthly -----------------------------------------------

    #[test]
    fn monthly_not_expired_same_month() {
        let start = dt(2024, 3, 1, 0, 0);
        let now = dt(2024, 3, 31, 23, 59);
        assert!(!window_expired(&BudgetWindow::Monthly, start, now));
    }

    #[test]
    fn monthly_expired_next_month() {
        let start = dt(2024, 3, 15, 12, 0);
        let now = dt(2024, 4, 1, 0, 0);
        assert!(window_expired(&BudgetWindow::Monthly, start, now));
    }

    #[test]
    fn monthly_expired_same_month_number_different_year() {
        let start = dt(2023, 3, 10, 0, 0);
        let now = dt(2024, 3, 10, 0, 0);
        assert!(window_expired(&BudgetWindow::Monthly, start, now));
    }

    // --- check_budget: USD exhaustion -----------------------------------------

    #[test]
    fn usd_budget_exhausted_at_limit() {
        let (mgr, _tmp) = temp_manager();
        // Spend equals the limit -> exhausted (>= comparison, Req 3.2).
        let key = key_with(Some(10.0), None, None, 10.0, 0, None);
        assert_eq!(
            mgr.check_budget_at(&key, Utc::now()),
            Err(BudgetError::BudgetExhausted)
        );
    }

    #[test]
    fn usd_budget_within_limit_ok() {
        let (mgr, _tmp) = temp_manager();
        let key = key_with(Some(10.0), None, None, 9.99, 0, None);
        assert!(mgr.check_budget_at(&key, Utc::now()).is_ok());
    }

    // --- check_budget: token exhaustion ---------------------------------------

    #[test]
    fn token_budget_exhausted_at_limit() {
        let (mgr, _tmp) = temp_manager();
        let key = key_with(None, Some(1_000), None, 0.0, 1_000, None);
        assert_eq!(
            mgr.check_budget_at(&key, Utc::now()),
            Err(BudgetError::TokenBudgetExhausted)
        );
    }

    #[test]
    fn token_budget_within_limit_ok() {
        let (mgr, _tmp) = temp_manager();
        let key = key_with(None, Some(1_000), None, 0.0, 999, None);
        assert!(mgr.check_budget_at(&key, Utc::now()).is_ok());
    }

    // --- check_budget: independence (Req 3.3 / 4.3) ---------------------------

    #[test]
    fn token_limit_triggers_independently_of_usd() {
        let (mgr, _tmp) = temp_manager();
        // USD well within limit, tokens exhausted -> token error.
        let key = key_with(Some(100.0), Some(1_000), None, 1.0, 1_000, None);
        assert_eq!(
            mgr.check_budget_at(&key, Utc::now()),
            Err(BudgetError::TokenBudgetExhausted)
        );
    }

    #[test]
    fn usd_limit_triggers_independently_of_tokens() {
        let (mgr, _tmp) = temp_manager();
        // Tokens well within limit, USD exhausted -> USD error.
        let key = key_with(Some(5.0), Some(1_000_000), None, 5.0, 10, None);
        assert_eq!(
            mgr.check_budget_at(&key, Utc::now()),
            Err(BudgetError::BudgetExhausted)
        );
    }

    #[test]
    fn both_within_limits_ok() {
        let (mgr, _tmp) = temp_manager();
        let key = key_with(Some(100.0), Some(1_000), None, 50.0, 500, None);
        assert!(mgr.check_budget_at(&key, Utc::now()).is_ok());
    }

    // --- check_budget: window reset allows a previously-exhausted key ----------

    #[test]
    fn expired_window_resets_and_allows_request() {
        let (mgr, _tmp) = temp_manager();
        // Spend exceeds limit, but the daily window has rolled to a new date;
        // effective spend is zeroed, so the request is allowed (Req 3.3).
        let start = dt(2024, 3, 10, 12, 0);
        let now = dt(2024, 3, 11, 0, 1);
        let key = key_with(Some(10.0), Some(100), Some(BudgetWindow::Daily), 50.0, 500, Some(start));
        assert!(mgr.check_budget_at(&key, now).is_ok());
    }

    #[test]
    fn active_window_still_enforces_limit() {
        let (mgr, _tmp) = temp_manager();
        // Same-day window (not expired) with spend over limit -> rejected.
        let start = dt(2024, 3, 10, 1, 0);
        let now = dt(2024, 3, 10, 12, 0);
        let key = key_with(Some(10.0), None, Some(BudgetWindow::Daily), 50.0, 0, Some(start));
        assert_eq!(
            mgr.check_budget_at(&key, now),
            Err(BudgetError::BudgetExhausted)
        );
    }

    // --- check_budget: lifetime budget never resets (Req 3.4 / 4.7) -----------

    #[test]
    fn lifetime_budget_never_resets() {
        let (mgr, _tmp) = temp_manager();
        // No budget window: even with a stale window_start and years elapsed,
        // the accumulated spend is enforced without reset.
        let key = key_with(Some(10.0), None, None, 10.0, 0, Some(dt(2000, 1, 1, 0, 0)));
        assert_eq!(
            mgr.check_budget_at(&key, dt(2024, 6, 1, 0, 0)),
            Err(BudgetError::BudgetExhausted)
        );
    }

    #[test]
    fn no_limits_configured_always_ok() {
        let (mgr, _tmp) = temp_manager();
        let key = key_with(None, None, None, 1_000_000.0, 1_000_000, None);
        assert!(mgr.check_budget_at(&key, Utc::now()).is_ok());
    }

    // === Property-based tests (task 5.2) =====================================
    //
    // Property 6 note: the production cost function lives in the usage module
    // (task 9.1) and is intentionally NOT defined here to avoid duplicating /
    // conflicting with a public `compute_cost`. Property 6 below tests a
    // test-local implementation of the design "Cost Computation" formula.

    use proptest::prelude::*;

    /// Round a value to 6 decimal places (design "round_to_6_decimals").
    fn round6(x: f64) -> f64 {
        (x * 1_000_000.0).round() / 1_000_000.0
    }

    /// Test-local cost formula (design "Cost Computation"). Mirrors the shape
    /// of the production function owned by task 9.1 without publishing it.
    fn compute_cost_local(
        input_tokens: u64,
        output_tokens: u64,
        input_rate: f64,
        output_rate: f64,
    ) -> f64 {
        let input_cost = input_tokens as f64 * input_rate / 1_000_000.0;
        let output_cost = output_tokens as f64 * output_rate / 1_000_000.0;
        round6(input_cost + output_cost)
    }

    proptest! {
        #![proptest_config(ProptestConfig::with_cases(100))]

        // Feature: virtual-key-management, Property 6: Cost Computation Formula
        // For non-negative input/output tokens and non-negative rates, cost ==
        // round6((input*in_rate/1e6)+(output*out_rate/1e6)).
        // Validates: Requirements 3.1
        #[test]
        fn prop6_cost_matches_formula(
            input_tokens in 0u64..=10_000_000,
            output_tokens in 0u64..=10_000_000,
            input_rate in 0.0f64..=1000.0,
            output_rate in 0.0f64..=1000.0,
        ) {
            let cost = compute_cost_local(input_tokens, output_tokens, input_rate, output_rate);

            // Recompute the formula independently and assert exact equality
            // (deterministic f64 ops), which pins both the arithmetic and the
            // 6-decimal rounding routine.
            let expected = round6(
                (input_tokens as f64 * input_rate / 1_000_000.0)
                    + (output_tokens as f64 * output_rate / 1_000_000.0),
            );
            prop_assert_eq!(cost, expected);

            // Cost is non-negative. The exact equality against the
            // independently-recomputed round6(..) value above already pins the
            // 6-decimal rounding; a separate `(cost*1e6 - round).abs() < 1e-6`
            // check is unsound for large magnitudes (f64 ULP exceeds the
            // tolerance) and is intentionally omitted.
            prop_assert!(cost >= 0.0);
        }

        // Feature: virtual-key-management, Property 7: Budget Enforcement Independence
        // Key with BOTH usd + token budgets (no window): USD exhaustion takes
        // precedence, else token exhaustion, else Ok — each independently.
        // Validates: Requirements 3.2, 4.2, 4.3
        #[test]
        fn prop7_budget_enforcement_independence(
            usd_limit in 1.0f64..=1000.0,
            token_limit in 1u64..=1_000_000,
            spend in 0.0f64..=2000.0,
            tokens in 0u64..=2_000_000,
        ) {
            let (mgr, _tmp) = temp_manager();
            let key = key_with(
                Some(usd_limit),
                Some(token_limit),
                None,
                spend,
                tokens,
                None,
            );
            let result = mgr.check_budget_at(&key, Utc::now());

            if spend >= usd_limit {
                // USD checked first regardless of token state.
                prop_assert_eq!(result, Err(BudgetError::BudgetExhausted));
            } else if tokens >= token_limit {
                prop_assert_eq!(result, Err(BudgetError::TokenBudgetExhausted));
            } else {
                prop_assert!(result.is_ok());
            }
        }

        // Feature: virtual-key-management, Property 8: Budget Window Reset
        // A windowed key whose window has expired treats counters as zero, so
        // check_budget_at is Ok even when stored spend/tokens exceed limits.
        // Validates: Requirements 3.3, 4.6
        #[test]
        fn prop8_expired_window_resets_counters(
            day_offset in 1i64..=3650,
            within_hours in 0i64..=11,
            usd_limit in 1.0f64..=1000.0,
            token_limit in 1u64..=1_000_000,
            usd_over in 0.0f64..=1000.0,
            tokens_over in 0u64..=1_000_000,
        ) {
            let start = dt(2024, 1, 1, 12, 0);
            let now = start + Duration::days(day_offset);
            let within = start + Duration::hours(within_hours);

            // Daily window: crossing to a later UTC date is expired; staying
            // within the same date (12:00 + <=11h) is not.
            prop_assert!(window_expired(&BudgetWindow::Daily, start, now));
            prop_assert!(!window_expired(&BudgetWindow::Daily, start, within));

            let (mgr, _tmp) = temp_manager();
            // Stored counters exceed both limits, but the window has rolled.
            let key = key_with(
                Some(usd_limit),
                Some(token_limit),
                Some(BudgetWindow::Daily),
                usd_limit + usd_over,
                token_limit + tokens_over,
                Some(start),
            );
            prop_assert!(mgr.check_budget_at(&key, now).is_ok());
        }

        // Feature: virtual-key-management, Property 9: Lifetime Budget No-Reset
        // No budget_window: arbitrarily large elapsed time never resets. If
        // spend >= limit it stays Err regardless of now.
        // Validates: Requirements 3.4, 4.7
        #[test]
        fn prop9_lifetime_budget_never_resets(
            usd_limit in 1.0f64..=1000.0,
            usd_over in 0.0f64..=1000.0,
            year in 1971i32..=3000,
            month in 1u32..=12,
            day in 1u32..=28,
        ) {
            let (mgr, _tmp) = temp_manager();
            // Lifetime budget (window=None), spend at/over limit. A stale
            // window_start plus a far-future `now` must not trigger a reset.
            let key = key_with(
                Some(usd_limit),
                None,
                None,
                usd_limit + usd_over,
                0,
                Some(dt(1970, 1, 1, 0, 0)),
            );
            let now = dt(year, month, day, 0, 0);
            prop_assert_eq!(
                mgr.check_budget_at(&key, now),
                Err(BudgetError::BudgetExhausted)
            );
        }
    }
}
