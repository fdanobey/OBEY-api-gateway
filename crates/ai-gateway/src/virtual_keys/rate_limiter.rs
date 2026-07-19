//! Per-key rate limiting: RPM token-bucket and TPM rolling window.
//!
//! [`PerKeyRateLimiter`] combines two independent limiters scoped to a single
//! virtual key:
//!
//! * an optional **RPM** token bucket (`requests_per_minute`): burst capacity
//!   equal to the configured RPM, refilling at `RPM / 60` tokens per second
//!   (Req 5.1, 5.2);
//! * an optional **TPM** rolling window (`tokens_per_minute`): a 60-second
//!   sliding window over actual token consumption reported by provider
//!   responses (Req 5.3, 5.4).
//!
//! All time-dependent methods have an `_at(now: Instant)` form so tests can
//! advance a synthetic clock deterministically; the public wrappers call
//! [`Instant::now`]. `Retry-After` values are the integer number of seconds,
//! rounded **up** (ceil), until capacity becomes available.

use std::collections::VecDeque;
use std::time::{Duration, Instant};

/// Number of seconds a caller should wait before retrying, as reported in the
/// `Retry-After` header. Always the ceiling of the true wait time.
pub type RetryAfterSeconds = u32;

/// Length of the TPM rolling window.
const TPM_WINDOW: Duration = Duration::from_secs(60);
/// Seconds per minute; the RPM bucket refills at `rpm / 60` tokens per second.
const SECONDS_PER_MINUTE: f64 = 60.0;

/// Rate-limit rejection reasons carrying the computed `Retry-After` seconds.
///
/// Maps to HTTP 429 with a `Retry-After` header per the design error table.
#[derive(Debug, Clone, PartialEq, Eq, thiserror::Error)]
pub enum RateLimitError {
    /// The per-minute request budget (token bucket) is exhausted.
    #[error("Requests per minute exceeded")]
    RpmExceeded {
        retry_after_seconds: RetryAfterSeconds,
    },
    /// The rolling 60-second token consumption meets or exceeds the limit.
    #[error("Tokens per minute exceeded")]
    TpmExceeded {
        retry_after_seconds: RetryAfterSeconds,
    },
}

/// Token bucket for `requests_per_minute`.
///
/// Uses the same algorithm as the provider-level `RateLimiter`: continuous
/// refill based on elapsed wall-clock time, capped at `capacity`.
#[derive(Debug, Clone)]
struct RpmBucket {
    /// Burst capacity == configured requests-per-minute.
    capacity: f64,
    /// Currently available tokens (fractional).
    tokens: f64,
    /// Refill rate in tokens per second (`rpm / 60`).
    refill_per_sec: f64,
    /// Timestamp of the last refill computation.
    last_refill: Instant,
}

impl RpmBucket {
    fn new(rpm: u32, now: Instant) -> Self {
        let capacity = rpm as f64;
        Self {
            capacity,
            tokens: capacity,
            refill_per_sec: rpm as f64 / SECONDS_PER_MINUTE,
            last_refill: now,
        }
    }

    /// Add tokens accrued since the last refill, capped at capacity.
    fn refill(&mut self, now: Instant) {
        if now <= self.last_refill {
            return;
        }
        let elapsed = now.duration_since(self.last_refill).as_secs_f64();
        let added = elapsed * self.refill_per_sec;
        if added > 0.0 {
            self.tokens = (self.tokens + added).min(self.capacity);
            self.last_refill = now;
        }
    }

    /// Ok when at least one token is available; otherwise the ceil seconds
    /// until a full token refills.
    fn check(&mut self, now: Instant) -> Result<(), RetryAfterSeconds> {
        self.refill(now);
        if self.tokens >= 1.0 {
            Ok(())
        } else {
            let needed = 1.0 - self.tokens;
            let seconds = needed / self.refill_per_sec;
            Err(ceil_secs(seconds).max(1))
        }
    }

    /// Deduct one token (clamped at zero). Callers `check` first.
    fn consume(&mut self, now: Instant) {
        self.refill(now);
        self.tokens = (self.tokens - 1.0).max(0.0);
    }
}

/// Rolling 60-second window for `tokens_per_minute`.
///
/// Stores `(timestamp, tokens)` entries and maintains their running sum;
/// entries older than 60 seconds are pruned lazily on each operation.
#[derive(Debug, Clone)]
struct TpmWindow {
    limit: u64,
    entries: VecDeque<(Instant, u64)>,
    current_sum: u64,
}

impl TpmWindow {
    fn new(limit: u64) -> Self {
        Self {
            limit,
            entries: VecDeque::new(),
            current_sum: 0,
        }
    }

    /// Drop entries whose age is >= 60 seconds, decrementing the running sum.
    fn prune(&mut self, now: Instant) {
        while let Some(&(ts, tokens)) = self.entries.front() {
            if now.duration_since(ts) >= TPM_WINDOW {
                self.entries.pop_front();
                self.current_sum = self.current_sum.saturating_sub(tokens);
            } else {
                break;
            }
        }
    }

    /// Ok while the rolling sum is below the limit; otherwise the ceil seconds
    /// until enough of the oldest entries expire to drop the sum below `limit`.
    fn check(&mut self, now: Instant) -> Result<(), RetryAfterSeconds> {
        self.prune(now);
        if self.current_sum < self.limit {
            return Ok(());
        }

        // Walk oldest→newest, accounting for tokens that will expire. Once the
        // remaining sum would drop below the limit, the wait is until that
        // entry ages out of the window (its timestamp + 60s).
        let mut remaining = self.current_sum;
        for &(ts, tokens) in &self.entries {
            remaining = remaining.saturating_sub(tokens);
            if remaining < self.limit {
                let expiry = ts + TPM_WINDOW;
                let wait = expiry.saturating_duration_since(now).as_secs_f64();
                return Err(ceil_secs(wait).max(1));
            }
        }
        // Fallback: after everything expires the window is empty (< limit).
        // Use the newest entry's expiry as the wait bound.
        let wait = self
            .entries
            .back()
            .map(|&(ts, _)| {
                (ts + TPM_WINDOW)
                    .saturating_duration_since(now)
                    .as_secs_f64()
            })
            .unwrap_or(0.0);
        Err(ceil_secs(wait).max(1))
    }

    /// Record `tokens` consumed at `now`, pruning stale entries first.
    fn record(&mut self, now: Instant, tokens: u64) {
        self.prune(now);
        if tokens == 0 {
            return;
        }
        self.entries.push_back((now, tokens));
        self.current_sum = self.current_sum.saturating_add(tokens);
    }
}

/// Round a non-negative seconds value up to the next whole second.
fn ceil_secs(seconds: f64) -> u32 {
    if seconds <= 0.0 {
        return 0;
    }
    seconds.ceil() as u32
}

/// Per-key rate limiter combining an RPM token bucket and a TPM rolling window.
///
/// Either limiter is absent when the corresponding key constraint is unset, in
/// which case its `check_*` always succeeds and its mutation is a no-op.
#[derive(Debug, Clone, Default)]
pub struct PerKeyRateLimiter {
    rpm_bucket: Option<RpmBucket>,
    tpm_window: Option<TpmWindow>,
}

impl PerKeyRateLimiter {
    /// Build a limiter from optional RPM/TPM constraints. The RPM bucket starts
    /// full (burst capacity == `rpm`).
    pub fn new(rpm: Option<u32>, tpm: Option<u64>) -> Self {
        Self::new_at(rpm, tpm, Instant::now())
    }

    /// [`Self::new`] with an explicit clock, for deterministic tests.
    pub fn new_at(rpm: Option<u32>, tpm: Option<u64>, now: Instant) -> Self {
        Self {
            rpm_bucket: rpm.map(|r| RpmBucket::new(r, now)),
            tpm_window: tpm.map(TpmWindow::new),
        }
    }

    /// Check RPM availability without consuming a token.
    pub fn check_rpm(&mut self) -> Result<(), RetryAfterSeconds> {
        self.check_rpm_at(Instant::now())
    }

    /// [`Self::check_rpm`] with an explicit clock.
    pub fn check_rpm_at(&mut self, now: Instant) -> Result<(), RetryAfterSeconds> {
        match self.rpm_bucket.as_mut() {
            Some(bucket) => bucket.check(now),
            None => Ok(()),
        }
    }

    /// Consume one RPM token. No-op when RPM is unlimited. Call after a
    /// successful [`Self::check_rpm`].
    pub fn consume_rpm(&mut self) {
        self.consume_rpm_at(Instant::now());
    }

    /// [`Self::consume_rpm`] with an explicit clock.
    pub fn consume_rpm_at(&mut self, now: Instant) {
        if let Some(bucket) = self.rpm_bucket.as_mut() {
            bucket.consume(now);
        }
    }

    /// Check TPM availability against the rolling 60-second window.
    pub fn check_tpm(&mut self) -> Result<(), RetryAfterSeconds> {
        self.check_tpm_at(Instant::now())
    }

    /// [`Self::check_tpm`] with an explicit clock.
    pub fn check_tpm_at(&mut self, now: Instant) -> Result<(), RetryAfterSeconds> {
        match self.tpm_window.as_mut() {
            Some(window) => window.check(now),
            None => Ok(()),
        }
    }

    /// Record `tokens` of consumption into the rolling window. No-op when TPM
    /// is unlimited.
    pub fn record_tpm(&mut self, tokens: u64) {
        self.record_tpm_at(Instant::now(), tokens);
    }

    /// [`Self::record_tpm`] with an explicit clock.
    pub fn record_tpm_at(&mut self, now: Instant, tokens: u64) {
        if let Some(window) = self.tpm_window.as_mut() {
            window.record(now, tokens);
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    // --- RPM token bucket ----------------------------------------------------

    /// Req 5.1: an RPM=R bucket permits an initial burst of exactly R requests,
    /// then rejects.
    #[test]
    fn rpm_allows_initial_burst_up_to_capacity() {
        let now = Instant::now();
        let mut limiter = PerKeyRateLimiter::new_at(Some(3), None, now);

        for _ in 0..3 {
            assert!(limiter.check_rpm_at(now).is_ok());
            limiter.consume_rpm_at(now);
        }
        // Fourth request within the same instant is rejected.
        assert!(limiter.check_rpm_at(now).is_err());
    }

    /// Req 5.1: after exhaustion the bucket refills at R/60 tokens per second.
    /// With RPM=60 (1 token/sec), one token is back after 1 second.
    #[test]
    fn rpm_refills_at_configured_rate() {
        let start = Instant::now();
        let mut limiter = PerKeyRateLimiter::new_at(Some(60), None, start);

        // Drain the full burst of 60.
        for _ in 0..60 {
            limiter.consume_rpm_at(start);
        }
        assert!(limiter.check_rpm_at(start).is_err());

        // After 1 second exactly one token (60/60) has refilled.
        let later = start + Duration::from_secs(1);
        assert!(limiter.check_rpm_at(later).is_ok());
    }

    /// Req 5.2: Retry-After is the ceil seconds until >= 1 token is available.
    /// RPM=60 → 1 token/sec, empty bucket → 1 second.
    #[test]
    fn rpm_retry_after_is_ceiled() {
        let start = Instant::now();
        let mut limiter = PerKeyRateLimiter::new_at(Some(60), None, start);
        for _ in 0..60 {
            limiter.consume_rpm_at(start);
        }

        // 0.5s after draining, 0.5 token has refilled; need 0.5 more → 0.5s,
        // ceil → 1 second.
        let mid = start + Duration::from_millis(500);
        match limiter.check_rpm_at(mid) {
            Err(secs) => assert_eq!(secs, 1),
            Ok(()) => panic!("bucket should be empty"),
        }
    }

    /// Slow refill rate rounds the wait up. RPM=6 → 0.1 token/sec; from empty,
    /// one full token needs 10 seconds.
    #[test]
    fn rpm_retry_after_ceils_slow_refill() {
        let start = Instant::now();
        let mut limiter = PerKeyRateLimiter::new_at(Some(6), None, start);
        for _ in 0..6 {
            limiter.consume_rpm_at(start);
        }
        match limiter.check_rpm_at(start) {
            Err(secs) => assert_eq!(secs, 10),
            Ok(()) => panic!("bucket should be empty"),
        }
    }

    /// Unlimited RPM (None) never blocks.
    #[test]
    fn rpm_unlimited_always_ok() {
        let now = Instant::now();
        let mut limiter = PerKeyRateLimiter::new_at(None, None, now);
        for _ in 0..1000 {
            assert!(limiter.check_rpm_at(now).is_ok());
            limiter.consume_rpm_at(now);
        }
    }

    // --- TPM rolling window --------------------------------------------------

    /// Req 5.3/5.4: once the rolling sum meets the limit, further checks are
    /// rejected until entries expire from the 60s window.
    #[test]
    fn tpm_rejects_when_window_sum_reaches_limit() {
        let start = Instant::now();
        let mut limiter = PerKeyRateLimiter::new_at(None, Some(1000), start);

        assert!(limiter.check_tpm_at(start).is_ok());
        limiter.record_tpm_at(start, 600);
        assert!(limiter.check_tpm_at(start).is_ok());
        limiter.record_tpm_at(start, 400); // sum == 1000 == limit
        assert!(limiter.check_tpm_at(start).is_err());
    }

    /// Req 5.4: Retry-After reflects when enough tokens expire to drop below
    /// the limit. A single entry at t0 expires 60s later.
    #[test]
    fn tpm_retry_after_until_entries_expire() {
        let start = Instant::now();
        let mut limiter = PerKeyRateLimiter::new_at(None, Some(1000), start);
        limiter.record_tpm_at(start, 1000); // at limit

        // 10s in, the lone entry expires at start+60 → 50s remaining.
        let t = start + Duration::from_secs(10);
        match limiter.check_tpm_at(t) {
            Err(secs) => assert_eq!(secs, 50),
            Ok(()) => panic!("window should be at limit"),
        }
    }

    /// Rolling window frees capacity once entries age past 60 seconds.
    #[test]
    fn tpm_window_slides_and_frees_capacity() {
        let start = Instant::now();
        let mut limiter = PerKeyRateLimiter::new_at(None, Some(1000), start);
        limiter.record_tpm_at(start, 1000);
        assert!(limiter.check_tpm_at(start).is_err());

        // At exactly 60s the first entry has aged out (age >= 60).
        let expired = start + Duration::from_secs(60);
        assert!(limiter.check_tpm_at(expired).is_ok());
    }

    /// Partial expiry: only entries old enough are pruned, so the wait targets
    /// the oldest entry needed to drop below the limit.
    #[test]
    fn tpm_partial_expiry_targets_oldest_needed_entry() {
        let start = Instant::now();
        let mut limiter = PerKeyRateLimiter::new_at(None, Some(1000), start);
        limiter.record_tpm_at(start, 600); // expires at start+60
        let later = start + Duration::from_secs(30);
        limiter.record_tpm_at(later, 400); // sum 1000, expires at start+90

        // At t=40s: sum still 1000 (nothing expired). Dropping the 600 entry
        // (expiry start+60) leaves 400 < 1000 → wait = 20s.
        let t = start + Duration::from_secs(40);
        match limiter.check_tpm_at(t) {
            Err(secs) => assert_eq!(secs, 20),
            Ok(()) => panic!("window should be at limit"),
        }
    }

    /// Unlimited TPM (None) never blocks regardless of recorded tokens.
    #[test]
    fn tpm_unlimited_always_ok() {
        let now = Instant::now();
        let mut limiter = PerKeyRateLimiter::new_at(None, None, now);
        limiter.record_tpm_at(now, 10_000_000);
        assert!(limiter.check_tpm_at(now).is_ok());
    }
}

#[cfg(test)]
mod property_tests {
    use super::*;
    use proptest::prelude::*;

    proptest! {
        #![proptest_config(ProptestConfig {
            cases: 100,
            .. ProptestConfig::default()
        })]

        // Feature: virtual-key-management, Property 10: For any virtual key with
        // requests_per_minute = R, the token bucket SHALL allow at most R requests
        // in an initial burst, and after exhaustion, the bucket SHALL refill at a
        // rate of R/60 tokens per second. When fewer than 1 token is available, the
        // response SHALL include a Retry-After header with the integer number of
        // seconds (rounded up) until at least 1 token becomes available.
        // Validates: Requirements 5.1, 5.2
        #[test]
        fn prop10_token_bucket_burst_and_refill(rpm in 1u32..=1500) {
            let base = Instant::now();

            // --- Burst capacity: exactly R requests allowed at a fixed instant,
            // the (R+1)th is rejected. ---
            let mut limiter = PerKeyRateLimiter::new_at(Some(rpm), None, base);
            for _ in 0..rpm {
                prop_assert!(limiter.check_rpm_at(base).is_ok());
                limiter.consume_rpm_at(base);
            }
            prop_assert!(
                limiter.check_rpm_at(base).is_err(),
                "the (R+1)th request in a fixed-instant burst must be rejected"
            );

            // --- Retry-After on a fully-drained bucket: >= 1, and waiting that
            // many seconds makes at least one token available (the reported wait
            // is sufficient). Bucket is empty at `base` from the burst above. ---
            match limiter.check_rpm_at(base) {
                Err(retry) => {
                    prop_assert!(retry >= 1, "Retry-After must be at least 1 second");
                    // From an empty bucket a single token needs 60/R seconds, so
                    // the ceil wait never exceeds 60 seconds (plus FP slack).
                    prop_assert!(retry <= 61, "Retry-After bounded by ~60s for R>=1");
                    let after = base + Duration::from_secs(u64::from(retry));
                    prop_assert!(
                        limiter.check_rpm_at(after).is_ok(),
                        "after waiting Retry-After seconds a token must be available"
                    );
                }
                Ok(()) => prop_assert!(false, "drained bucket must reject"),
            }

            // --- Monotonic refill is bounded by burst capacity: after enough time
            // to fully refill (>= 60s), the bucket again allows exactly R and no
            // more, proving refill never exceeds capacity R. ---
            let mut limiter2 = PerKeyRateLimiter::new_at(Some(rpm), None, base);
            for _ in 0..rpm {
                limiter2.consume_rpm_at(base);
            }
            let refilled = base + Duration::from_secs(120);
            for _ in 0..rpm {
                prop_assert!(limiter2.check_rpm_at(refilled).is_ok());
                limiter2.consume_rpm_at(refilled);
            }
            prop_assert!(
                limiter2.check_rpm_at(refilled).is_err(),
                "refill must be capped at burst capacity R"
            );
        }
    }

    proptest! {
        #![proptest_config(ProptestConfig {
            cases: 100,
            .. ProptestConfig::default()
        })]

        // Feature: virtual-key-management, Property 11: For any virtual key with
        // tokens_per_minute = T and any sequence of requests with token counts
        // within a 60-second window, the gateway SHALL reject requests when the sum
        // of tokens consumed in the rolling 60-second window meets or exceeds T,
        // returning HTTP 429 with a Retry-After header. Once all entries age out of
        // the 60-second window, capacity is available again.
        // Validates: Requirements 5.3, 5.4
        #[test]
        fn prop11_rolling_window_tpm(
            tpm in 1u64..=10_000_000,
            amounts in proptest::collection::vec(0u64..=2_000_000, 1..=8),
        ) {
            let base = Instant::now();
            let mut limiter = PerKeyRateLimiter::new_at(None, Some(tpm), base);

            let mut running: u64 = 0;
            for &amt in &amounts {
                // Before recording: check reflects the current rolling sum. All
                // entries share `base`, so nothing has expired yet.
                if running < tpm {
                    prop_assert!(
                        limiter.check_tpm_at(base).is_ok(),
                        "while rolling sum < T the check must pass"
                    );
                } else {
                    match limiter.check_tpm_at(base) {
                        Err(retry) => {
                            // All entries recorded at `base` expire at base+60, so
                            // the wait until the sum drops below T is exactly 60s.
                            prop_assert!(retry >= 1 && retry <= 60);
                            prop_assert_eq!(retry, 60);
                        }
                        Ok(()) => prop_assert!(false, "sum >= T must reject"),
                    }
                }
                limiter.record_tpm_at(base, amt);
                running = running.saturating_add(amt);
            }

            // Final state at `base` matches the cumulative sum.
            if running >= tpm {
                match limiter.check_tpm_at(base) {
                    Err(retry) => prop_assert!(retry >= 1 && retry <= 60),
                    Ok(()) => prop_assert!(false, "final sum >= T must reject"),
                }
            } else {
                prop_assert!(limiter.check_tpm_at(base).is_ok());
            }

            // After 60 seconds every entry has aged out of the window, so the sum
            // is 0 (< T since T >= 1) and the check passes again.
            let expired = base + Duration::from_secs(60);
            prop_assert!(
                limiter.check_tpm_at(expired).is_ok(),
                "after the 60s window all entries expire and checks pass"
            );
        }
    }
}
