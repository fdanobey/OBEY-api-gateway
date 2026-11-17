use std::cmp::Reverse;
use std::collections::BTreeMap;
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::{Arc, Weak};
use std::time::Duration;
use tokio::sync::{Mutex, Notify};
use tokio::time::Instant;

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

/// Ordered-admission key for one queued waiter (spec
/// virtual-key-priority-classes, Requirement 2).
///
/// `BTreeMap` iterates keys ascending, and `Reverse(priority)` makes the
/// ordering `(priority DESC, arrival ASC)`: the first key is the queue
/// head — highest priority, earliest arrival among equals.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord)]
struct WaiterKey {
    priority: Reverse<u16>,
    arrival: u64,
}

/// A waiter parked on the priority wait-queue. The queue holds only a
/// [`Weak`] reference: a waiter that times out removes itself, and a
/// waiter whose task is cancelled simply dangles and is drained on the
/// next grant pass — neither can strand a token.
///
/// `granted` is set (and `woken` notified) while the limiter state lock
/// is held, so a late arrival can never barge between the head-check
/// and the grant.
#[derive(Debug, Default)]
struct Waiter {
    granted: AtomicBool,
    woken: Notify,
}

/// Internal state for the rate limiter, protected by a single mutex
/// to avoid potential deadlocks from acquiring multiple locks.
///
/// The token bucket and the priority wait-queue live in this ONE lock
/// domain: "queue head AND token available" (grant) and "token
/// available AND queue empty" (fast path) are decided atomically under
/// the same mutex.
#[derive(Debug, Default)]
struct RateLimiterState {
    /// Current number of available tokens
    tokens: f64,
    /// Last refill timestamp
    last_refill: Option<Instant>,
    /// Optional upstream-driven cooldown deadline. While `Some(deadline)`
    /// and `now < deadline`, the limiter reports unavailable so failover
    /// skips this provider without round-tripping it.
    cooldown_until: Option<Instant>,
    /// Priority wait-queue, ordered head-first (see [`WaiterKey`]).
    waiters: BTreeMap<WaiterKey, Weak<Waiter>>,
    /// Monotonic arrival counter for FIFO tie-breaks among equal
    /// priorities.
    next_arrival: u64,
}

/// Why the local limiter refused a request. The request was not sent.
#[derive(Debug, Clone, PartialEq)]
pub struct RateLimitShortfall {
    /// Configured `rate_limit_per_minute`.
    pub requests_per_minute: u32,
    /// Bucket balance at the time of the check (negative after debits).
    pub tokens_available: f64,
    /// Time until one token is available (or the cooldown ends).
    pub retry_in: Duration,
    /// True when an upstream-driven cooldown, not the bucket, refused.
    pub cooldown: bool,
}

impl std::fmt::Display for RateLimitShortfall {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        if self.cooldown {
            write!(
                f,
                "local cooldown after an upstream rate limit ({:.0}s remaining); request not sent to provider",
                self.retry_in.as_secs_f64().ceil()
            )
        } else {
            write!(
                f,
                "local rate limit {}/min exhausted ({:.2} tokens available); next token in {:.1}s; request not sent to provider",
                self.requests_per_minute,
                self.tokens_available,
                self.retry_in.as_secs_f64()
            )
        }
    }
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
                last_refill: Some(Instant::now()),
                cooldown_until: None,
                waiters: BTreeMap::new(),
                next_arrival: 0,
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

    /// If an upstream-driven cooldown is active, `Some(shortfall)` for it
    /// (and the wait-queue is NOT granted through it); otherwise clears an
    /// expired deadline and returns `None`. Internal, call-site equivalent
    /// of the inline checks the public gates run.
    fn active_cooldown_shortfall(
        &self,
        state: &mut RateLimiterState,
    ) -> Option<RateLimitShortfall> {
        let deadline = state.cooldown_until?;
        let now = Instant::now();
        if now < deadline {
            Some(RateLimitShortfall {
                requests_per_minute: self.requests_per_minute,
                tokens_available: state.tokens,
                retry_in: deadline - now,
                cooldown: true,
            })
        } else {
            state.cooldown_until = None;
            None
        }
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
        self.try_acquire().await.is_ok()
    }

    /// Take one token for a new upstream request, or report why not.
    ///
    /// Same gate as [`Self::consume`]; the error carries the bucket state so a
    /// local skip can never be mistaken for an upstream 429 (ADR 0002).
    ///
    /// When priority waiters are queued, a free token belongs to the queue
    /// head (`priority DESC, arrival ASC`): this call hands the token to
    /// the head and reports the resulting shortfall to its own caller
    /// instead of barging ahead of the queue.
    pub async fn try_acquire(&self) -> Result<(), RateLimitShortfall> {
        let mut state = self.state.lock().await;

        // Honor upstream-driven cooldown regardless of bucket capacity.
        if let Some(shortfall) = self.active_cooldown_shortfall(&mut state) {
            return Err(shortfall);
        }

        // Unlimited rate limit
        if self.requests_per_minute == 0 {
            return Ok(());
        }

        self.refill_tokens_internal(&mut state);

        if state.tokens >= 1.0 {
            // Tokens that just became available serve the queue head
            // first; with an empty queue this is the uncontended fast
            // path and the loop below grants nothing.
            self.grant_available_tokens_to_queue(&mut state);
            if state.tokens >= 1.0 {
                state.tokens -= 1.0;
                return Ok(());
            }
        }
        Err(self.bucket_shortfall(&state))
    }

    /// Snapshot shortfall for a bucket-exhausted limiter (>= 1 token away).
    fn bucket_shortfall(&self, state: &RateLimiterState) -> RateLimitShortfall {
        let tokens_per_second = self.requests_per_minute as f64 / 60.0;
        RateLimitShortfall {
            requests_per_minute: self.requests_per_minute,
            tokens_available: state.tokens,
            retry_in: Duration::from_secs_f64((1.0 - state.tokens) / tokens_per_second),
            cooldown: false,
        }
    }

    /// Hand every available token to queued waiters, head first, until the
    /// bucket drops below one token or the queue empties. Direct handoff:
    /// the token is debited here, `granted` is set and the head notified
    /// while the state lock is held, so no other arrival can take the
    /// token between the head-check and the wake. Waiters whose tasks were
    /// cancelled (dangling [`Weak`]) are drained without consuming a
    /// token.
    ///
    /// Never grants through a live upstream cooldown: a token that appears
    /// while the cooldown holds (refill, `refund`) stays in the bucket
    /// until the cooldown expires and a gate re-probes.
    fn grant_available_tokens_to_queue(&self, state: &mut RateLimiterState) {
        if let Some(deadline) = state.cooldown_until {
        if Instant::now() < deadline {
        return;
        }
        }
    while state.tokens >= 1.0 {
            let Some((_, waiter)) = state.waiters.pop_first() else {
                break;
            };
            let Some(waiter) = waiter.upgrade() else {
                // Cancelled waiter: no token spent, try the next head.
                continue;
            };
            waiter.granted.store(true, Ordering::Release);
            waiter.woken.notify_one();
            state.tokens -= 1.0;
        }
    }

    /// Take one token, waiting up to `max_wait` for the bucket to refill.
    ///
    /// Returns how long it waited. Never waits through an upstream cooldown,
    /// and gives up at once when the next token is further away than the
    /// remaining wait budget. The bucket mutex is released while sleeping, so
    /// callers must not hold any other lock or permit across this call.
    ///
    /// Priority class 0 (see [`Self::acquire_within_priority`]).
    pub async fn acquire_within(&self, max_wait: Duration) -> Result<Duration, RateLimitShortfall> {
        self.acquire_within_priority(0, max_wait).await
    }

    /// Priority-aware bounded admission wait (spec
    /// virtual-key-priority-classes, Requirement 2).
    ///
    /// Like [`Self::acquire_within`], but when no token is immediately
    /// available the waiter joins an ordered queue granted in
    /// `(priority DESC, arrival ASC)` order instead of racing in a
    /// sleep-and-recheck loop: only the queue head is granted a token
    /// when one refills (or is refunded), so a higher-priority late
    /// arrival is still admitted before lower-priority waiters. When
    /// the wait budget is exhausted, or an upstream cooldown is active,
    /// the existing [`RateLimitShortfall`] is returned unchanged.
    /// `max_wait == 0` bypasses the queue and probes
    /// [`Self::try_acquire`] directly.
    ///
    /// No capacity is reserved or partitioned by priority class: the
    /// bucket, refill, debit and cooldown semantics are untouched.
    pub async fn acquire_within_priority(
        &self,
        priority: u16,
        max_wait: Duration,
    ) -> Result<Duration, RateLimitShortfall> {
        if max_wait == Duration::ZERO {
            return self.try_acquire().await.map(|_| Duration::ZERO);
        }
        let started = Instant::now();
        // Fast path, identical to `acquire_within`: token available AND
        // queue empty (decided atomically in `try_acquire` under the one
        // state lock) → immediate grant, zero queueing.
        let mut shortfall = match self.try_acquire().await {
            Ok(()) => return Ok(Duration::ZERO),
            Err(shortfall) => shortfall,
        };
        if shortfall.cooldown {
            // Never wait through an upstream cooldown.
            return Err(shortfall);
        }

        let waiter = Arc::new(Waiter::default());
        let key = {
            let mut state = self.state.lock().await;
            // A cooldown or a refill may have happened since the probe.
            if let Some(cooldown) = self.active_cooldown_shortfall(&mut state) {
                return Err(cooldown);
            }
            if self.requests_per_minute != 0 {
                self.refill_tokens_internal(&mut state);
                self.grant_available_tokens_to_queue(&mut state);
                if state.tokens >= 1.0 {
                    // The queue is drained now, so this is the
                    // uncontended path, not a barge.
                    state.tokens -= 1.0;
                    return Ok(started.elapsed());
                }
            } else {
                // Unlimited bucket cannot have been exhausted; the probe
                // would have granted. Defensive, unreachable in practice.
                return Ok(started.elapsed());
            }
            let key = WaiterKey {
                priority: Reverse(priority),
                arrival: state.next_arrival,
            };
            state.next_arrival += 1;
            state.waiters.insert(key, Arc::downgrade(&waiter));
            shortfall = self.bucket_shortfall(&state);
            key
        };

        loop {
            // Give up at once when the next token cannot arrive inside
            // the remaining wait budget; same policy as `acquire_within`.
            let remaining = max_wait.saturating_sub(started.elapsed());
            if shortfall.cooldown || shortfall.retry_in > remaining {
                if self.dequeue_if_not_granted(&key, &waiter).await {
                    // A grant won the race with the timeout.
                    return Ok(started.elapsed());
                }
                return Err(shortfall);
            }
            // Park until the projected token, a head grant, or the
            // budget boundary — whichever comes first.
            let wait_for = (shortfall.retry_in + Duration::from_millis(1)).min(remaining);
            tokio::select! {
                () = waiter.woken.notified() => {}
                () = tokio::time::sleep(wait_for) => {}
            }
            if waiter.granted.load(Ordering::Acquire) {
                return Ok(started.elapsed());
            }
            // Timer boundary (or spurious wake): re-probe under the one
            // lock, refilling and letting available tokens drain to the
            // queue head — possibly this waiter.
            let mut state = self.state.lock().await;
            if let Some(cooldown) = self.active_cooldown_shortfall(&mut state) {
                drop(state);
                if self.dequeue_if_not_granted(&key, &waiter).await {
                    return Ok(started.elapsed());
                }
                return Err(cooldown);
            }
            self.refill_tokens_internal(&mut state);
            self.grant_available_tokens_to_queue(&mut state);
            if waiter.granted.load(Ordering::Acquire) {
                return Ok(started.elapsed());
            }
            shortfall = self.bucket_shortfall(&state);
        }
    }

    /// Remove a timed-out waiter from the queue. Returns `true` when a
    /// grant had already been handed to it: grants remove the waiter and
    /// set `granted` while holding the state lock, so checking after
    /// removal under the same lock loses no grant and admits no
    /// double-spend.
    async fn dequeue_if_not_granted(&self, key: &WaiterKey, waiter: &Waiter) -> bool {
        let mut state = self.state.lock().await;
        state.waiters.remove(key);
        waiter.granted.load(Ordering::Acquire)
    }

    /// Charge one token for a continuation request (retry, Codex Search
    /// round, nudge) of a request that was already admitted. Never blocks and
    /// never refuses: the balance may go negative, which delays the next
    /// admission instead of failing work already in flight (ADR 0002).
    pub async fn debit(&self) {
        if self.requests_per_minute == 0 {
            return;
        }
        let mut state = self.state.lock().await;
        self.refill_tokens_internal(&mut state);
        state.tokens -= 1.0;
    }

    /// Return one admission token taken for a candidate that was never
    /// dispatched (ADR 0002: exactly one token per real upstream request — a
    /// gate-skipped candidate consumes nothing).
    ///
    /// The inverse of [`Self::debit`]: adds one token back, capped at
    /// capacity, then immediately hands it to the priority queue head if
    /// anyone is waiting — a refunded token serves the head before it is
    /// available to new arrivals. No-op for unlimited buckets
    /// (`requests_per_minute == 0` never consumed one).
    pub async fn refund(&self) {
        if self.requests_per_minute == 0 {
            return;
        }
        let mut state = self.state.lock().await;
        self.refill_tokens_internal(&mut state);
        state.tokens = (state.tokens + 1.0).min(self.capacity as f64);
        self.grant_available_tokens_to_queue(&mut state);
    }

    /// Refill tokens based on elapsed time since last refill
    /// Internal method that operates on already-locked state
    fn refill_tokens_internal(&self, state: &mut RateLimiterState) {
        let Some(last_refill) = state.last_refill else {
            // Unlimited bucket: nothing to refill.
            return;
        };
        let now = Instant::now();
        let elapsed = now.duration_since(last_refill);

        // Calculate tokens to add based on elapsed time
        // tokens_per_second = requests_per_minute / 60
        let tokens_per_second = self.requests_per_minute as f64 / 60.0;
        let tokens_to_add = elapsed.as_secs_f64() * tokens_per_second;

        if tokens_to_add > 0.0 {
            state.tokens = (state.tokens + tokens_to_add).min(self.capacity as f64);
            state.last_refill = Some(now);
        }
    }

    /// Get current token count (for testing/monitoring)
    #[allow(dead_code)]
    pub async fn get_tokens(&self) -> f64 {
        let mut state = self.state.lock().await;
        self.refill_tokens_internal(&mut state);
        state.tokens
    }

    /// Number of waiters currently parked on the priority queue
    /// (for testing/monitoring).
    #[cfg(test)]
    async fn queued_waiter_count(&self) -> usize {
        self.state.lock().await.waiters.len()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use tokio::time::sleep;

    #[tokio::test(start_paused = true)]
    async fn test_cooldown_blocks_unlimited_bucket() {
        let limiter = RateLimiter::new(0);
        assert!(limiter.consume().await);

        limiter.apply_cooldown(Duration::from_millis(80)).await;
        assert!(!limiter.check_available().await);
        assert!(!limiter.consume().await);
        assert!(limiter.cooldown_remaining().await.is_some());

        tokio::time::advance(Duration::from_millis(120)).await;
        assert!(limiter.cooldown_remaining().await.is_none());
        assert!(limiter.check_available().await);
        assert!(limiter.consume().await);
    }

    #[tokio::test(start_paused = true)]
    async fn test_cooldown_blocks_token_bucket() {
        let limiter = RateLimiter::new(60);
        assert!(limiter.consume().await);

        limiter.apply_cooldown(Duration::from_millis(80)).await;
        assert!(!limiter.consume().await);

        tokio::time::advance(Duration::from_millis(120)).await;
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

    #[tokio::test(start_paused = true)]
    async fn test_token_refill() {
        let limiter = RateLimiter::new(60); // 60 requests per minute = 1 per second

        // Consume all tokens
        for _ in 0..60 {
            assert!(limiter.consume().await);
        }

        assert!(!limiter.check_available().await);

        // Wait for 1 second to refill 1 token (fast-forwarded: paused clock)
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

    #[tokio::test(start_paused = true)]
    async fn test_token_refill_caps_at_capacity() {
        let limiter = RateLimiter::new(10);

        // Consume 5 tokens
        for _ in 0..5 {
            assert!(limiter.consume().await);
        }

        // Wait long enough to refill more than capacity (fast-forwarded)
        sleep(Duration::from_secs(2)).await;

        // Should have capacity tokens, not more
        let tokens = limiter.get_tokens().await;
        assert!(
            tokens <= 10.0,
            "Tokens should be capped at capacity: {}",
            tokens
        );
    }

    #[tokio::test(start_paused = true)]
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

    #[tokio::test(start_paused = true)]
    async fn acquire_within_waits_for_one_token_inside_budget() {
        let limiter = RateLimiter::new(8); // one token per 7.5 s
        assert_eq!(
            limiter.acquire_within(Duration::from_secs(10)).await,
            Ok(Duration::ZERO),
            "an immediate grant reports no wait"
        );
        for _ in 0..7 {
            assert!(limiter.consume().await);
        }
        let waited = limiter
            .acquire_within(Duration::from_secs(10))
            .await
            .expect("token refills within the 10 s budget");
        assert!(waited >= Duration::from_millis(7_500), "{waited:?}");
        assert!(waited < Duration::from_millis(7_600), "{waited:?}");
    }

    #[tokio::test(start_paused = true)]
    async fn acquire_within_refuses_when_token_is_beyond_budget() {
        let limiter = RateLimiter::new(2); // one token per 30 s
        assert!(limiter.consume().await);
        assert!(limiter.consume().await);
        let shortfall = limiter
            .acquire_within(Duration::from_secs(10))
            .await
            .expect_err("30 s refill exceeds the 10 s budget");
        assert!(!shortfall.cooldown);
        assert_eq!(shortfall.requests_per_minute, 2);
        let message = shortfall.to_string();
        assert!(
            message.starts_with("local rate limit 2/min exhausted"),
            "{message}"
        );
        assert!(message.contains("next token in 30.0s"), "{message}");
        assert!(
            message.ends_with("request not sent to provider"),
            "{message}"
        );
    }

    #[tokio::test(start_paused = true)]
    async fn acquire_within_never_waits_through_cooldown() {
        let limiter = RateLimiter::new(60);
        limiter.apply_cooldown(Duration::from_millis(50)).await;
        let shortfall = limiter
            .acquire_within(Duration::from_secs(10))
            .await
            .expect_err("cooldown is not waited out");
        assert!(shortfall.cooldown);
        assert!(shortfall.to_string().starts_with("local cooldown"));
    }

    #[tokio::test(start_paused = true)]
    async fn debit_charges_even_when_empty_and_delays_next_admission() {
        let limiter = RateLimiter::new(60); // one token per second
        for _ in 0..60 {
            assert!(limiter.consume().await);
        }
        limiter.debit().await;
        assert!((limiter.get_tokens().await + 1.0).abs() < 1e-6);
        // Two tokens of refill are now needed before the next admission.
        let shortfall = limiter.try_acquire().await.unwrap_err();
        assert!((shortfall.retry_in.as_secs_f64() - 2.0).abs() < 0.01);

        let unlimited = RateLimiter::new(0);
        unlimited.debit().await;
        assert!(unlimited.consume().await);
    }

    // ---- Priority wait-queue (spec virtual-key-priority-classes, Req 2) ----

    /// Grants waiters in (priority DESC, arrival ASC) order: the
    /// higher-priority late arrival is admitted first.
    #[tokio::test(start_paused = true)]
    async fn priority_queue_grants_highest_priority_head_first() {
        let limiter = Arc::new(RateLimiter::new(2)); // one token per 30 s
        assert!(limiter.consume().await);
        assert!(limiter.consume().await);

        // Low priority arrives first, high priority second.
        let low = {
            let limiter = Arc::clone(&limiter);
            tokio::spawn(async move {
                limiter
                    .acquire_within_priority(0, Duration::from_secs(120))
                    .await
            })
        };
        tokio::task::yield_now().await;
        let high = {
            let limiter = Arc::clone(&limiter);
            tokio::spawn(async move {
                limiter
                    .acquire_within_priority(5, Duration::from_secs(120))
                    .await
            })
        };
        tokio::task::yield_now().await;
        assert_eq!(limiter.queued_waiter_count().await, 2);

        // One token refills: the high-priority head is granted.
        sleep(Duration::from_secs(31)).await;
        assert!(high.is_finished(), "queue head (priority 5) was granted");
        assert!(!low.is_finished(), "priority 0 waiter is still queued");
        assert_eq!(limiter.queued_waiter_count().await, 1);

        let high_waited = high.await.unwrap().expect("granted");
        assert!(high_waited >= Duration::from_secs(30), "{high_waited:?}");
        assert!(high_waited < Duration::from_secs(31), "{high_waited:?}");

        // The next token goes to the remaining waiter.
        sleep(Duration::from_secs(31)).await;
        let low_waited = low.await.unwrap().expect("granted");
        assert!(low_waited >= Duration::from_secs(60), "{low_waited:?}");
        assert!(low_waited < Duration::from_secs(61), "{low_waited:?}");
        assert_eq!(limiter.queued_waiter_count().await, 0);
    }

    /// Equal priorities are granted in arrival (FIFO) order.
    #[tokio::test(start_paused = true)]
    async fn priority_queue_ties_grant_in_arrival_order() {
        let limiter = Arc::new(RateLimiter::new(1));
        assert!(limiter.consume().await);

        let first = {
            let limiter = Arc::clone(&limiter);
            tokio::spawn(async move {
                limiter
                    .acquire_within_priority(3, Duration::from_secs(180))
                    .await
            })
        };
        tokio::task::yield_now().await;
        let second = {
            let limiter = Arc::clone(&limiter);
            tokio::spawn(async move {
                limiter
                    .acquire_within_priority(3, Duration::from_secs(180))
                    .await
            })
        };
        tokio::task::yield_now().await;

        sleep(Duration::from_secs(61)).await;
        assert!(first.is_finished(), "equal priority: earlier arrival first");
        assert!(!second.is_finished(), "later arrival still queued");

        let first_waited = first.await.unwrap().expect("granted");
        assert!(first_waited >= Duration::from_secs(60), "{first_waited:?}");
        assert!(first_waited < Duration::from_secs(61), "{first_waited:?}");

        sleep(Duration::from_secs(61)).await;
        let second_waited = second.await.unwrap().expect("granted");
        assert!(
            second_waited >= Duration::from_secs(120),
            "{second_waited:?}"
        );
        assert!(
            second_waited < Duration::from_secs(121),
            "{second_waited:?}"
        );
    }

    /// A lower-priority waiter's budget expires while every refilled
    /// token keeps going to a newly-arrived higher-priority head: it
    /// fails over with the existing shortfall, exactly like an
    /// over-budget wait.
    #[tokio::test(start_paused = true)]
    async fn priority_queue_lower_priority_times_out_under_contention() {
        let limiter = Arc::new(RateLimiter::new(2)); // one token per 30 s
        assert!(limiter.consume().await);
        assert!(limiter.consume().await);

        let low = {
            let limiter = Arc::clone(&limiter);
            tokio::spawn(async move {
                limiter
                    .acquire_within_priority(0, Duration::from_secs(110))
                    .await
            })
        };
        tokio::task::yield_now().await;
        let mut highs = vec![{
            let limiter = Arc::clone(&limiter);
            tokio::spawn(async move {
                limiter
                    .acquire_within_priority(9, Duration::from_secs(60))
                    .await
            })
        }];
        tokio::task::yield_now().await;

        // Every refilled token is taken by a fresh higher-priority
        // arrival (grants at t≈30/60/90); after the third preemption
        // the low-priority waiter's remaining budget (~20 s) no longer
        // covers the next 30 s refill, so it bails with the shortfall.
        for _ in 0..3 {
            sleep(Duration::from_secs(31)).await;
            let high = highs.pop().unwrap();
            let waited = high.await.unwrap().expect("head granted");
            assert!(waited >= Duration::from_secs(20), "{waited:?}");
            assert!(waited < Duration::from_secs(31), "{waited:?}");
            highs.push({
                let limiter = Arc::clone(&limiter);
                tokio::spawn(async move {
                    limiter
                        .acquire_within_priority(9, Duration::from_secs(60))
                        .await
                })
            });
            tokio::task::yield_now().await;
        }

        let shortfall = low.await.unwrap().expect_err("budget exhausted");
        assert!(!shortfall.cooldown);
        assert_eq!(shortfall.requests_per_minute, 2);
        assert!(
            shortfall.retry_in >= Duration::from_secs(29),
            "{:?}",
            shortfall.retry_in
        );
        // The last high-priority arrival holds the granted token.
        let last_high = highs.pop().unwrap();
        let waited = last_high.await.unwrap().expect("granted");
        assert!(waited >= Duration::from_secs(20), "{waited:?}");
        assert!(waited < Duration::from_secs(31), "{waited:?}");
        assert_eq!(limiter.queued_waiter_count().await, 0);
    }

    /// Uncontended fast path: token available + empty queue → immediate
    /// grant reporting zero wait, and nothing is ever queued.
    #[tokio::test(start_paused = true)]
    async fn priority_queue_fast_path_immediate_grant() {
        let limiter = RateLimiter::new(5);
        assert_eq!(
            limiter
                .acquire_within_priority(7, Duration::from_secs(10))
                .await,
            Ok(Duration::ZERO)
        );
        assert_eq!(limiter.queued_waiter_count().await, 0);
        assert_eq!(
            limiter.acquire_within(Duration::from_secs(10)).await,
            Ok(Duration::ZERO)
        );
        assert_eq!(limiter.queued_waiter_count().await, 0);
    }

    /// A refunded admission token is handed to the queued head before it
    /// is available to new arrivals (router.rs `refund()` call sites).
    #[tokio::test(start_paused = true)]
    async fn priority_queue_refund_grants_queued_head() {
    let limiter = Arc::new(RateLimiter::new(1));
    assert!(limiter.consume().await);

    let waiter = {
    let limiter = Arc::clone(&limiter);
    tokio::spawn(async move {
    limiter
    .acquire_within_priority(7, Duration::from_secs(120))
    .await
    })
    };
    tokio::task::yield_now().await;
    assert_eq!(limiter.queued_waiter_count().await, 1);

    // The gate-skipped candidate returns its token: the queued head
    // gets it, not the next new arrival.
    limiter.refund().await;
    let waited = waiter.await.unwrap().expect("refunded token granted");
    assert!(waited < Duration::from_secs(1), "{waited:?}");
    assert_eq!(limiter.queued_waiter_count().await, 0);
    // The refunded token is gone: no second grant from it.
    assert!(limiter.try_acquire().await.is_err());
    }

    /// Refund (and refill) never grant through a live cooldown: the
    /// refunded token sits in the bucket while the cooldown deadline is
    /// live, and is handed to the queued head once the cooldown has
    /// expired.
    #[tokio::test(start_paused = true)]
    async fn priority_queue_refund_holds_grant_through_cooldown() {
    let limiter = Arc::new(RateLimiter::new(1)); // one token per 60 s
    assert!(limiter.consume().await);

    // The waiter's next scheduled wake (projected refill) is at ~60 s —
    // past the 30 s cooldown below, so it stays parked unless a grant
    // reaches it.
    let waiter = {
    let limiter = Arc::clone(&limiter);
    tokio::spawn(async move {
    limiter
    .acquire_within_priority(7, Duration::from_secs(300))
    .await
    })
    };
    tokio::task::yield_now().await;
    assert_eq!(limiter.queued_waiter_count().await, 1);

    // Cooldown lands, then a token is refunded: the token is banked but
    // no grant is made while the deadline is live.
    limiter.apply_cooldown(Duration::from_secs(30)).await;
    limiter.refund().await;
    tokio::task::yield_now().await;
    assert!(!waiter.is_finished(), "no grant during cooldown");
    assert_eq!(limiter.queued_waiter_count().await, 1);
    assert!(
    limiter.get_tokens().await >= 1.0,
    "refunded token is banked in the bucket"
    );

    // Cooldown expires at t=30 s; the waiter is still parked until its
    // projected-token wake at ~60 s, and nothing grants early.
    sleep(Duration::from_secs(25)).await; // t ≈ 25 s, cooldown live
    assert!(!waiter.is_finished());
    sleep(Duration::from_secs(10)).await; // t ≈ 35 s, cooldown expired
    assert!(!waiter.is_finished(), "no re-probe before the wake");
    assert_eq!(limiter.queued_waiter_count().await, 1);

    // At the wake the cooldown is gone and the banked token goes to the
    // queued head.
    sleep(Duration::from_secs(30)).await; // t ≈ 65 s
    let waited = waiter.await.unwrap().expect("granted after expiry");
    assert!(waited >= Duration::from_secs(60), "{waited:?}");
    assert!(waited < Duration::from_secs(61), "{waited:?}");
    assert_eq!(limiter.queued_waiter_count().await, 0);
    // The banked token went to the head, not back to new arrivals.
    assert!(limiter.try_acquire().await.is_err());
    }

    /// Zero-budget acquires bypass the queue entirely: direct
    /// `try_acquire`, no registration, existing shortfall on failure.
    #[tokio::test(start_paused = true)]
    async fn priority_queue_zero_budget_bypasses_queue() {
        // Token available: immediate grant without queueing.
        let full = RateLimiter::new(5);
        assert_eq!(
            full.acquire_within_priority(9, Duration::ZERO).await,
            Ok(Duration::ZERO)
        );
        assert_eq!(full.queued_waiter_count().await, 0);

        // Bucket empty with a waiter parked: a zero-budget acquire must
        // not register (and cannot steal the head's future token).
        let limiter = Arc::new(RateLimiter::new(1));
        assert!(limiter.consume().await);
        let waiter = {
            let limiter = Arc::clone(&limiter);
            tokio::spawn(async move {
                limiter
                    .acquire_within_priority(5, Duration::from_secs(120))
                    .await
            })
        };
        tokio::task::yield_now().await;
        assert_eq!(limiter.queued_waiter_count().await, 1);

        let shortfall = limiter
            .acquire_within_priority(9, Duration::ZERO)
            .await
            .expect_err("zero budget does not wait");
        assert!(!shortfall.cooldown);
        assert_eq!(limiter.queued_waiter_count().await, 1, "no registration");

        limiter.refund().await;
        let waited = waiter.await.unwrap().expect("head still gets the token");
        assert!(waited < Duration::from_secs(1), "{waited:?}");
    }

    /// Debit keeps its never-block/never-refuse semantics and does not
    /// mint a token for the queue; an upstream cooldown applied while
    /// waiting is never waited through.
    #[tokio::test(start_paused = true)]
    async fn priority_queue_debit_and_cooldown_invariants() {
        let limiter = Arc::new(RateLimiter::new(1)); // one token per 60 s
        assert!(limiter.consume().await);

        let waiter = {
            let limiter = Arc::clone(&limiter);
            tokio::spawn(async move {
                limiter
                    .acquire_within_priority(5, Duration::from_secs(300))
                    .await
            })
        };
        tokio::task::yield_now().await;

        // A debit goes negative and must not grant the waiter.
        limiter.debit().await;
        tokio::task::yield_now().await;
        assert!(!waiter.is_finished(), "debit does not grant the queue");

        // The waiter wakes at its projected token time, finds the
        // balance negative (debit pushed it there), and keeps waiting.
        sleep(Duration::from_secs(61)).await;
        assert!(!waiter.is_finished(), "no token was minted for the queue");

        // A cooldown that lands while waiting is not waited through:
        // the waiter bails with the cooldown shortfall at its next
        // scheduled wake-up.
        limiter.apply_cooldown(Duration::from_secs(300)).await;
        // Force the next wake: the waiter re-probes after its current
        // sleep (up to 60 s away with a drained bucket).
        sleep(Duration::from_secs(61)).await;
        let shortfall = waiter.await.unwrap().expect_err("cooldown not waited out");
        assert!(shortfall.cooldown);
        assert_eq!(limiter.queued_waiter_count().await, 0);
    }

    /// A cancelled waiter's dead queue entry is drained without spending
    /// a token; the next head receives the grant instead.
    #[tokio::test(start_paused = true)]
    async fn priority_queue_cancels_drained_without_token_spend() {
        let limiter = Arc::new(RateLimiter::new(1));
        assert!(limiter.consume().await);

        let cancelled = {
            let limiter = Arc::clone(&limiter);
            tokio::spawn(async move {
                limiter
                    .acquire_within_priority(9, Duration::from_secs(120))
                    .await
            })
        };
        tokio::task::yield_now().await;
        let live = {
            let limiter = Arc::clone(&limiter);
            tokio::spawn(async move {
                limiter
                    .acquire_within_priority(1, Duration::from_secs(120))
                    .await
            })
        };
        tokio::task::yield_now().await;
        cancelled.abort();
        tokio::task::yield_now().await;
        assert_eq!(limiter.queued_waiter_count().await, 2, "entry drained lazily");

        // The token skips the dead head and grants the live waiter.
        limiter.refund().await;
        let waited = live.await.unwrap().expect("live head granted");
        assert!(waited < Duration::from_secs(1), "{waited:?}");
        assert_eq!(limiter.queued_waiter_count().await, 0);
    }
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
            // Paused clock: the 1s wait fast-forwards instantly while the
            // limiter still observes a full second of virtual elapsed time
            // (tokio::time::Instant), keeping the refill math deterministic.
            let rt = tokio::runtime::Builder::new_current_thread()
                .enable_all()
                .start_paused(true)
                .build()
                .unwrap();
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
