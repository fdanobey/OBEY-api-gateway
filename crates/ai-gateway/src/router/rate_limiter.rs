use std::sync::Arc;
use std::time::{Duration, Instant};
use tokio::sync::Mutex;

/// Hard upper bound that any cooldown is clamped to before being applied.
///
/// This is a safety backstop, not a policy: it bounds against negative,
/// nonsense, or malicious upstream values, but is intentionally large
/// enough to accommodate weekly-quota providers (e.g. Nano-GPT) that
/// legitimately want a multi-day cooldown.
///
/// The *policy* cap (per-provider / global) is enforced earlier, in
/// `Router::parse_rate_limit_cooldown`. By the time `apply_cooldown` is
/// called, the value should already match operator policy.
pub const MAX_COOLDOWN: Duration = Duration::from_secs(7 * 24 * 60 * 60);

/// Internal state for the rate limiter, protected by a single mutex
/// to avoid potential deadlocks from acquiring multiple locks
#[derive(Debug)]
struct RateLimiterState {
    /// Current number of available tokens
    tokens: f64,
    /// Last refill timestamp
    last_refill: Instant,
    /// Optional upstream-driven cooldown deadline. While `Some(deadline)`
    /// and `now < deadline`, the limiter reports unavailable so failover
    /// skips this provider without round-tripping it.
    cooldown_until: Option<Instant>,
}

/// Token bucket rate limiter for per-provider rate limiting
#[derive(Debug)]
pub struct RateLimiter {
    /// Maximum number of tokens (requests per minute)
    capacity: u32,
    /// Requests per minute limit
    requests_per_minute: u32,
    /// Combined state protected by a single mutex to prevent deadlocks
    state: Arc<Mutex<RateLimiterState>>,
}

impl RateLimiter {
    /// Create a new rate limiter with specified requests per minute
    ///
    /// # Arguments
    /// * `requests_per_minute` - Maximum requests allowed per minute (0 = unlimited)
    pub fn new(requests_per_minute: u32) -> Self {
        Self {
            capacity: requests_per_minute,
            requests_per_minute,
            state: Arc::new(Mutex::new(RateLimiterState {
                tokens: requests_per_minute as f64,
                last_refill: Instant::now(),
                cooldown_until: None,
            })),
        }
    }

    /// Returns `Some(remaining)` if the limiter is in an upstream-driven
    /// cooldown, `None` otherwise. Used by failover to skip providers that
    /// recently returned a rate-limit signal without re-issuing the request.
    pub async fn cooldown_remaining(&self) -> Option<Duration> {
        let state = self.state.lock().await;
        match state.cooldown_until {
            Some(deadline) => {
                let now = Instant::now();
                if now < deadline {
                    Some(deadline - now)
                } else {
                    None
                }
            }
            None => None,
        }
    }

    /// Apply an upstream-driven cooldown. Subsequent `check_available` /
    /// `consume` / `cooldown_remaining` calls report the provider as
    /// unavailable until `duration` has elapsed.
    ///
    /// If a longer cooldown is already in effect it is preserved; otherwise
    /// the new (clamped) deadline replaces it. The cooldown duration is
    /// always clamped to [`MAX_COOLDOWN`].
    pub async fn apply_cooldown(&self, duration: Duration) {
        let clamped = duration.min(MAX_COOLDOWN);
        let new_deadline = Instant::now() + clamped;
        let mut state = self.state.lock().await;
        match state.cooldown_until {
            Some(existing) if existing >= new_deadline => {
                // Keep the longer cooldown.
            }
            _ => {
                state.cooldown_until = Some(new_deadline);
            }
        }
    }

    /// Clear any active upstream cooldown. Called on successful provider
    /// responses so we don't carry stale rate-limit windows across config
    /// reloads or after the upstream recovers early.
    pub async fn clear_cooldown(&self) {
        let mut state = self.state.lock().await;
        state.cooldown_until = None;
    }

    /// Check if a request can be made without consuming a token
    ///
    /// Returns true if tokens are available (or the bucket is unlimited)
    /// AND the limiter is not in an upstream-driven cooldown.
    #[allow(dead_code)]
    pub async fn check_available(&self) -> bool {
        let mut state = self.state.lock().await;

        // Honor upstream-driven cooldown regardless of bucket capacity.
        if let Some(deadline) = state.cooldown_until {
            if Instant::now() < deadline {
                return false;
            }
            state.cooldown_until = None;
        }

        // Unlimited rate limit
        if self.requests_per_minute == 0 {
            return true;
        }

        self.refill_tokens_internal(&mut state);
        state.tokens >= 1.0
    }

    /// Consume a token for a request
    ///
    /// Returns true if a token was consumed (or the bucket is unlimited)
    /// AND the limiter is not in an upstream-driven cooldown.
    pub async fn consume(&self) -> bool {
        let mut state = self.state.lock().await;

        // Honor upstream-driven cooldown regardless of bucket capacity.
        if let Some(deadline) = state.cooldown_until {
            if Instant::now() < deadline {
                return false;
            }
            state.cooldown_until = None;
        }

        // Unlimited rate limit
        if self.requests_per_minute == 0 {
            return true;
        }

        self.refill_tokens_internal(&mut state);

        if state.tokens >= 1.0 {
            state.tokens -= 1.0;
            true
        } else {
            false
        }
    }

    /// Refill tokens based on elapsed time since last refill
    /// Internal method that operates on already-locked state
    fn refill_tokens_internal(&self, state: &mut RateLimiterState) {
        let now = Instant::now();
        let elapsed = now.duration_since(state.last_refill);

        // Calculate tokens to add based on elapsed time
        // tokens_per_second = requests_per_minute / 60
        let tokens_per_second = self.requests_per_minute as f64 / 60.0;
        let tokens_to_add = elapsed.as_secs_f64() * tokens_per_second;

        if tokens_to_add > 0.0 {
            state.tokens = (state.tokens + tokens_to_add).min(self.capacity as f64);
            state.last_refill = now;
        }
    }

    /// Get current token count (for testing/monitoring)
    #[allow(dead_code)]
    pub async fn get_tokens(&self) -> f64 {
        let mut state = self.state.lock().await;
        self.refill_tokens_internal(&mut state);
        state.tokens
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use tokio::time::sleep;

    #[tokio::test]
    async fn test_cooldown_blocks_unlimited_bucket() {
        // Even providers with no per-minute limit must honor an
        // upstream-driven cooldown window.
        let limiter = RateLimiter::new(0);
        assert!(limiter.consume().await);

        limiter.apply_cooldown(Duration::from_millis(80)).await;
        assert!(!limiter.check_available().await);
        assert!(!limiter.consume().await);
        assert!(limiter.cooldown_remaining().await.is_some());

        sleep(Duration::from_millis(120)).await;
        assert!(limiter.cooldown_remaining().await.is_none());
        assert!(limiter.check_available().await);
        assert!(limiter.consume().await);
    }

    #[tokio::test]
    async fn test_cooldown_blocks_token_bucket() {
        let limiter = RateLimiter::new(60);
        assert!(limiter.consume().await);

        limiter.apply_cooldown(Duration::from_millis(80)).await;
        assert!(!limiter.consume().await);

        sleep(Duration::from_millis(120)).await;
        assert!(limiter.consume().await);
    }

    #[tokio::test]
    async fn test_cooldown_keeps_longer_existing_window() {
        let limiter = RateLimiter::new(0);
        limiter.apply_cooldown(Duration::from_secs(5)).await;
        limiter.apply_cooldown(Duration::from_millis(10)).await;

        // Should still be in the longer (5s) cooldown.
        let remaining = limiter.cooldown_remaining().await.unwrap();
        assert!(remaining > Duration::from_millis(500));
    }

    #[tokio::test]
    async fn test_cooldown_clamped_to_max() {
        let limiter = RateLimiter::new(0);
        // Anything beyond MAX_COOLDOWN (7 days) is the limiter's hard
        // backstop — anything larger gets clamped here. Operator-policy
        // caps are enforced earlier in the router.
        let huge = MAX_COOLDOWN + Duration::from_secs(60);
        limiter.apply_cooldown(huge).await;

        let remaining = limiter.cooldown_remaining().await.unwrap();
        assert!(remaining <= MAX_COOLDOWN);
        assert!(remaining > MAX_COOLDOWN - Duration::from_secs(1));
    }

    #[tokio::test]
    async fn test_clear_cooldown_restores_availability() {
        let limiter = RateLimiter::new(0);
        limiter.apply_cooldown(Duration::from_secs(30)).await;
        assert!(!limiter.check_available().await);

        limiter.clear_cooldown().await;
        assert!(limiter.cooldown_remaining().await.is_none());
        assert!(limiter.check_available().await);
    }

    #[tokio::test]
    async fn test_unlimited_rate_limit() {
        let limiter = RateLimiter::new(0);

        // Should always allow requests
        for _ in 0..100 {
            assert!(limiter.check_available().await);
            assert!(limiter.consume().await);
        }
    }

    #[tokio::test]
    async fn test_rate_limit_enforcement() {
        let limiter = RateLimiter::new(60); // 60 requests per minute

        // Should allow up to capacity
        for _ in 0..60 {
            assert!(limiter.consume().await);
        }

        // Should reject when exhausted
        assert!(!limiter.check_available().await);
        assert!(!limiter.consume().await);
    }

    #[tokio::test]
    async fn test_token_refill() {
        let limiter = RateLimiter::new(60); // 60 requests per minute = 1 per second

        // Consume all tokens
        for _ in 0..60 {
            assert!(limiter.consume().await);
        }

        assert!(!limiter.check_available().await);

        // Wait for 1 second to refill 1 token
        sleep(Duration::from_millis(1100)).await;

        assert!(limiter.check_available().await);
        assert!(limiter.consume().await);

        // Should be exhausted again
        assert!(!limiter.check_available().await);
    }

    #[tokio::test]
    async fn test_check_available_does_not_consume() {
        let limiter = RateLimiter::new(60);

        // Consume all but one token
        for _ in 0..59 {
            assert!(limiter.consume().await);
        }

        // Check multiple times without consuming
        assert!(limiter.check_available().await);
        assert!(limiter.check_available().await);
        assert!(limiter.check_available().await);

        // Should still have 1 token available
        assert!(limiter.consume().await);

        // Now should be exhausted
        assert!(!limiter.check_available().await);
    }

    #[tokio::test]
    async fn test_token_refill_caps_at_capacity() {
        let limiter = RateLimiter::new(10);

        // Consume 5 tokens
        for _ in 0..5 {
            assert!(limiter.consume().await);
        }

        // Wait long enough to refill more than capacity
        sleep(Duration::from_secs(2)).await;

        // Should have capacity tokens, not more
        let tokens = limiter.get_tokens().await;
        assert!(
            tokens <= 10.0,
            "Tokens should be capped at capacity: {}",
            tokens
        );
    }

    #[tokio::test]
    async fn test_fractional_token_accumulation() {
        let limiter = RateLimiter::new(60); // 1 token per second

        // Consume all tokens
        for _ in 0..60 {
            assert!(limiter.consume().await);
        }

        // Wait for 0.5 seconds (should accumulate 0.5 tokens)
        sleep(Duration::from_millis(500)).await;
        assert!(!limiter.check_available().await); // Not enough for 1 request

        // Wait another 0.6 seconds (total 1.1 tokens)
        sleep(Duration::from_millis(600)).await;
        assert!(limiter.check_available().await); // Now have >= 1 token
        assert!(limiter.consume().await);
    }

    // Feature: ai-gateway, Property 40: Rate Limit Enforcement
    // **Validates: Requirements 44.2, 44.3**
    #[cfg(test)]
    mod proptests {
        use super::*;
        use proptest::prelude::*;

        proptest! {
            #![proptest_config(ProptestConfig {
                cases: 20,
                max_shrink_iters: 100,
                .. ProptestConfig::default()
            })]

            #[test]
            fn prop_rate_limit_enforced_in_60s_window(
                rate_limit in 10u32..100u32,
                burst_size in 1usize..20usize,
            ) {
                let rt = tokio::runtime::Runtime::new().unwrap();
                rt.block_on(async {
                    let limiter = RateLimiter::new(rate_limit);
                    let start = std::time::Instant::now();
                    let mut successful_requests = 0u32;

                    // Attempt burst_size requests immediately
                    for _ in 0..burst_size.min(rate_limit as usize) {
                        if limiter.consume().await {
                            successful_requests += 1;
                        }
                    }

                    // Verify we don't exceed rate limit in initial burst
                    prop_assert!(successful_requests <= rate_limit);

                    // Wait 1 second and try more requests
                    tokio::time::sleep(Duration::from_secs(1)).await;

                    let expected_refill = (rate_limit as f64 / 60.0).ceil() as u32;
                    let mut second_batch = 0u32;

                    for _ in 0..expected_refill + 5 {
                        if limiter.consume().await {
                            second_batch += 1;
                        }
                    }

                    // Total requests in ~1 second should not exceed rate_limit + expected_refill
                    let total = successful_requests + second_batch;
                    let elapsed = start.elapsed().as_secs_f64();
                    let max_allowed = (rate_limit as f64 * elapsed / 60.0).ceil() as u32 + rate_limit;

                    prop_assert!(total <= max_allowed,
                        "Rate limit violated: {} requests in {:.2}s (limit: {} req/min, max_allowed: {})",
                        total, elapsed, rate_limit, max_allowed);

                    Ok(())
                })?;
            }
        }
    }
}
