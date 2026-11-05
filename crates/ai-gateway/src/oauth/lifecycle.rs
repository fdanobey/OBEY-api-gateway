#![allow(dead_code)]
//! Shared OAuth lifecycle constants — the single source of truth for the
//! background-refresh cadence used by BOTH credential managers:
//! the OpenAI [`super::OAuthManager`] and the GitHub Copilot
//! [`super::github_copilot::manager::CopilotTokenManager`].
//!
//! Per design Decision 3 of the `github-copilot-oauth-provider` spec, the
//! two managers share **constants only** — each keeps its own isolated
//! orchestration (tick loop, backoff loop, cache, session state). No
//! generic framework lives here; just the numeric policy both managers
//! must agree on:
//!
//! * **Tick cadence** — 60 s between refresh checks (OpenAI Req 5.2,
//!   Copilot Req 3.2).
//! * **Pre-expiry window** — refresh any token within 5 min (300 s) of
//!   expiry.
//! * **Backoff schedule** — `2^(attempt + 1)` seconds (`2 s, 4 s, 8 s`),
//!   at most 3 attempts, on transient failures.

use std::time::Duration;

/// Background-refresh tick cadence (OpenAI Req 5.2 / Copilot Req 3.2:
/// "check at a regular interval").
pub(crate) const REFRESH_TICK_INTERVAL: Duration = Duration::from_secs(60);

/// Pre-expiry refresh window in seconds (OpenAI Req 5.2 / Copilot Req 3.2:
/// "within 5 minutes of expiry").
pub(crate) const REFRESH_WINDOW_SECS: u64 = 300;

/// Number of refresh attempts before marking the session as expired
/// (OpenAI Req 5.5 / Copilot Req 3.2: "up to 3 attempts").
pub(crate) const BACKOFF_MAX_ATTEMPTS: u32 = 3;

/// Exponential-backoff delay in seconds before refresh attempt `attempt`
/// (OpenAI Req 5.5 / Property 8 and Copilot Req 3.2:
/// `2^(attempt + 1)` seconds).
///
/// `attempt = 0 → 2 s`, `1 → 4 s`, `2 → 8 s`. Shared by both managers;
/// the property tests in `oauth/manager.rs`, `oauth/lifecycle.rs`, and
/// `oauth/github_copilot/manager.rs` pin the formula independently.
pub(crate) fn backoff_delay_secs(attempt: u32) -> u64 {
    2u64.pow(attempt + 1)
}

#[cfg(test)]
mod tests {
    use super::*;
    use proptest::prelude::*;

    // Feature: shared OAuth lifecycle policy (openai-oauth-login Req 5.5 /
    // github-copilot-oauth-provider Req 3.2), Property 8:
    //
    // ∀ attempt ∈ {0, 1, 2}. backoff_delay_secs(attempt) == 2^(attempt + 1)
    //
    // The valid domain is exactly `{0, 1, 2}` because
    // BACKOFF_MAX_ATTEMPTS is 3. A tight proptest range exercises each
    // point with room to spare; the three literal schedule values
    // (2, 4, 8) are pinned so a regression that keeps the exponential
    // shape but shifts the base would still be caught. The tick/window
    // constants are pinned too, since both managers consume them and a
    // silent drift would desynchronize Copilot from the OpenAI policy.
    proptest! {
        #![proptest_config(ProptestConfig {
            cases: 16,
            .. ProptestConfig::default()
        })]

        #[test]
        fn prop_backoff_schedule(attempt in 0u32..=2u32) {
            let expected = 2u64.pow(attempt + 1);
            let actual = backoff_delay_secs(attempt);
            prop_assert_eq!(actual, expected);

            match attempt {
                0 => prop_assert_eq!(actual, 2),
                1 => prop_assert_eq!(actual, 4),
                2 => prop_assert_eq!(actual, 8),
                _ => unreachable!(),
            }
        }
    }

    #[test]
    fn shared_constants_pin_the_documented_policy() {
        assert_eq!(REFRESH_TICK_INTERVAL, Duration::from_secs(60));
        assert_eq!(REFRESH_WINDOW_SECS, 300);
        assert_eq!(BACKOFF_MAX_ATTEMPTS, 3);
    }
}
