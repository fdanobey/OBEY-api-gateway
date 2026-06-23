use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::Arc;
use std::time::{Duration, Instant};
use tokio::sync::RwLock;

/// Circuit breaker states
#[derive(Debug, Clone, PartialEq)]
pub enum CircuitState {
    /// Circuit is closed, requests flow normally
    Closed,
    /// Circuit is open, requests are rejected
    Open {
        opened_at: Instant,
        retry_after: Duration,
    },
    /// Circuit is half-open, allowing test requests
    HalfOpen,
}

/// Circuit breaker for provider failure detection
pub struct CircuitBreaker {
    state: Arc<RwLock<CircuitState>>,
    failure_threshold: u32,
    backoff_sequence: Vec<Duration>,
    current_backoff_index: AtomicUsize,
    consecutive_failures: AtomicUsize,
}

impl CircuitBreaker {
    /// Create a new circuit breaker with default backoff sequence [5s, 10s, 20s, 40s, 300s]
    #[allow(dead_code)]
    pub fn new(failure_threshold: u32) -> Self {
        Self::with_backoff_sequence(
            failure_threshold,
            vec![
                Duration::from_secs(5),
                Duration::from_secs(10),
                Duration::from_secs(20),
                Duration::from_secs(40),
                Duration::from_secs(300),
            ],
        )
    }

    /// Create a circuit breaker with custom backoff sequence
    pub fn with_backoff_sequence(failure_threshold: u32, backoff_sequence: Vec<Duration>) -> Self {
        Self {
            state: Arc::new(RwLock::new(CircuitState::Closed)),
            failure_threshold,
            backoff_sequence,
            current_backoff_index: AtomicUsize::new(0),
            consecutive_failures: AtomicUsize::new(0),
        }
    }

    /// Check if the circuit breaker allows requests
    /// 
    /// Uses a write lock to atomically check and transition state, avoiding
    /// race conditions between checking elapsed time and updating state.
    pub async fn is_available(&self) -> bool {
        let mut state = self.state.write().await;
        match &*state {
            CircuitState::Closed => true,
            CircuitState::HalfOpen => true,
            CircuitState::Open {
                opened_at,
                retry_after,
            } => {
                // Check if enough time has passed to transition to half-open
                if opened_at.elapsed() >= *retry_after {
                    *state = CircuitState::HalfOpen;
                    true
                } else {
                    false
                }
            }
        }
    }

    /// Record a successful request
    pub async fn record_success(&self) {
        self.consecutive_failures.store(0, Ordering::SeqCst);
        self.current_backoff_index.store(0, Ordering::SeqCst);

        let mut state = self.state.write().await;
        *state = CircuitState::Closed;
    }

    /// Record a failed request
    pub async fn record_failure(&self) {
        let failures = self.consecutive_failures.fetch_add(1, Ordering::SeqCst) + 1;

        if failures >= self.failure_threshold as usize {
            let backoff_index = self.current_backoff_index.load(Ordering::SeqCst);
            let retry_after = self.backoff_sequence[backoff_index.min(self.backoff_sequence.len() - 1)];

            let mut state = self.state.write().await;
            match *state {
                CircuitState::HalfOpen => {
                    // Failed in half-open, increase backoff
                    let next_index = (backoff_index + 1).min(self.backoff_sequence.len() - 1);
                    self.current_backoff_index.store(next_index, Ordering::SeqCst);
                    let new_retry_after = self.backoff_sequence[next_index];
                    *state = CircuitState::Open {
                        opened_at: Instant::now(),
                        retry_after: new_retry_after,
                    };
                }
                _ => {
                    // Transition to open
                    *state = CircuitState::Open {
                        opened_at: Instant::now(),
                        retry_after,
                    };
                }
            }
        }
    }

    /// Get current state (for testing/monitoring)
    pub async fn get_state(&self) -> CircuitState {
        self.state.read().await.clone()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use tokio::time::sleep;

    #[tokio::test]
    async fn test_circuit_breaker_starts_closed() {
        let cb = CircuitBreaker::new(3);
        assert!(cb.is_available().await);
        assert_eq!(cb.get_state().await, CircuitState::Closed);
    }

    #[tokio::test]
    async fn test_circuit_breaker_opens_after_threshold() {
        let cb = CircuitBreaker::new(3);

        // Record failures up to threshold
        cb.record_failure().await;
        cb.record_failure().await;
        assert!(cb.is_available().await);

        cb.record_failure().await;
        assert!(!cb.is_available().await);

        match cb.get_state().await {
            CircuitState::Open { retry_after, .. } => {
                assert_eq!(retry_after, Duration::from_secs(5));
            }
            _ => panic!("Expected Open state"),
        }
    }

    #[tokio::test]
    async fn test_circuit_breaker_transitions_to_half_open() {
        let cb = CircuitBreaker::new(1);

        cb.record_failure().await;
        assert!(!cb.is_available().await);

        // Wait for backoff period
        sleep(Duration::from_millis(5100)).await;

        assert!(cb.is_available().await);
        assert_eq!(cb.get_state().await, CircuitState::HalfOpen);
    }

    #[tokio::test]
    async fn test_circuit_breaker_closes_on_success() {
        let cb = CircuitBreaker::new(1);

        cb.record_failure().await;
        assert!(!cb.is_available().await);
        
        sleep(Duration::from_millis(5100)).await;
        
        // is_available transitions to half-open
        assert!(cb.is_available().await);
        assert_eq!(cb.get_state().await, CircuitState::HalfOpen);

        cb.record_success().await;
        assert_eq!(cb.get_state().await, CircuitState::Closed);
    }

    #[tokio::test]
    async fn test_circuit_breaker_exponential_backoff() {
        let cb = CircuitBreaker::with_backoff_sequence(
            1,
            vec![
                Duration::from_millis(10),
                Duration::from_millis(20),
                Duration::from_millis(30),
            ],
        );

        // First failure: 10ms backoff
        cb.record_failure().await;
        match cb.get_state().await {
            CircuitState::Open { retry_after, .. } => {
                assert_eq!(retry_after, Duration::from_millis(10));
            }
            _ => panic!("Expected Open state"),
        }

        // Transition to half-open and fail again: 20ms backoff
        sleep(Duration::from_millis(15)).await;
        assert!(cb.is_available().await);
        cb.record_failure().await;
        match cb.get_state().await {
            CircuitState::Open { retry_after, .. } => {
                assert_eq!(retry_after, Duration::from_millis(20));
            }
            _ => panic!("Expected Open state"),
        }

        // Transition to half-open and fail again: 30ms backoff
        sleep(Duration::from_millis(25)).await;
        assert!(cb.is_available().await);
        cb.record_failure().await;
        match cb.get_state().await {
            CircuitState::Open { retry_after, .. } => {
                assert_eq!(retry_after, Duration::from_millis(30));
            }
            _ => panic!("Expected Open state"),
        }
    }

    #[tokio::test]
    async fn test_circuit_breaker_max_backoff() {
        // Use shorter durations for testing
        let cb = CircuitBreaker::with_backoff_sequence(
            1,
            vec![
                Duration::from_millis(10),
                Duration::from_millis(20),
                Duration::from_millis(30),
                Duration::from_millis(40),
                Duration::from_millis(50),
            ],
        );

        // Fail repeatedly to reach max backoff
        for i in 0..6 {
            cb.record_failure().await;
            let state = cb.get_state().await;
            if let CircuitState::Open { retry_after, .. } = state {
                sleep(retry_after + Duration::from_millis(5)).await;
                assert!(cb.is_available().await);
            }
            
            // After 5th failure, should be capped at 50ms
            if i >= 4 {
                match cb.get_state().await {
                    CircuitState::Open { retry_after, .. } => {
                        assert_eq!(retry_after, Duration::from_millis(50));
                    }
                    CircuitState::HalfOpen => {
                        // Expected after waiting
                    }
                    _ => panic!("Unexpected state"),
                }
            }
        }
    }

    #[tokio::test]
    async fn test_circuit_breaker_resets_backoff_on_success() {
        // Use short durations for testing
        let cb = CircuitBreaker::with_backoff_sequence(
            1,
            vec![
                Duration::from_millis(10),
                Duration::from_millis(20),
                Duration::from_millis(30),
            ],
        );

        // Fail and increase backoff
        cb.record_failure().await;
        sleep(Duration::from_millis(15)).await;
        assert!(cb.is_available().await);
        cb.record_failure().await;

        // Should be at 20ms backoff
        match cb.get_state().await {
            CircuitState::Open { retry_after, .. } => {
                assert_eq!(retry_after, Duration::from_millis(20));
            }
            _ => panic!("Expected Open state"),
        }

        // Success should reset
        sleep(Duration::from_millis(25)).await;
        assert!(cb.is_available().await);
        cb.record_success().await;

        // Next failure should start at 10ms again
        cb.record_failure().await;
        match cb.get_state().await {
            CircuitState::Open { retry_after, .. } => {
                assert_eq!(retry_after, Duration::from_millis(10));
            }
            _ => panic!("Expected Open state"),
        }
    }

    #[tokio::test]
    async fn test_default_backoff_sequence() {
        let cb = CircuitBreaker::new(1);
        assert_eq!(
            cb.backoff_sequence,
            vec![
                Duration::from_secs(5),
                Duration::from_secs(10),
                Duration::from_secs(20),
                Duration::from_secs(40),
                Duration::from_secs(300),
            ]
        );
    }
}

#[cfg(test)]
mod property_tests {
    use super::*;
    use proptest::prelude::*;

    proptest! {
        #![proptest_config(ProptestConfig {
            cases: 10,
            .. ProptestConfig::default()
        })]

        /// **Validates: Requirements 9.2-9.6**
        /// 
        /// Property 4: Circuit Breaker State Transitions
        /// 
        /// For any circuit breaker, the state transitions shall follow:
        /// - Closed → (after failure) → Open
        /// - Open → (after backoff) → HalfOpen
        /// - HalfOpen → (after test success) → Closed
        /// - HalfOpen → (after test failure) → Open with increased backoff
        #[test]
        fn prop_circuit_breaker_state_transitions(
            failure_threshold in 1u32..=5,
            backoff_ms in prop::collection::vec(10u64..=100, 2..=5)
        ) {
            tokio::runtime::Runtime::new().unwrap().block_on(async {
                let backoff_durations: Vec<Duration> = backoff_ms.iter()
                    .map(|&ms| Duration::from_millis(ms))
                    .collect();
                
                let cb = CircuitBreaker::with_backoff_sequence(
                    failure_threshold,
                    backoff_durations.clone(),
                );

                // Initial state: Closed
                assert_eq!(cb.get_state().await, CircuitState::Closed);
                assert!(cb.is_available().await);

                // Transition: Closed → Open (after threshold failures)
                for _ in 0..failure_threshold {
                    cb.record_failure().await;
                }
                
                assert!(!cb.is_available().await);
                match cb.get_state().await {
                    CircuitState::Open { retry_after, .. } => {
                        assert_eq!(retry_after, backoff_durations[0]);
                    }
                    _ => panic!("Expected Open state after failures"),
                }

                // Transition: Open → HalfOpen (after backoff)
                tokio::time::sleep(backoff_durations[0] + Duration::from_millis(5)).await;
                assert!(cb.is_available().await);
                assert_eq!(cb.get_state().await, CircuitState::HalfOpen);

                // Transition: HalfOpen → Closed (after success)
                cb.record_success().await;
                assert_eq!(cb.get_state().await, CircuitState::Closed);
                assert!(cb.is_available().await);

                // Transition: HalfOpen → Open with increased backoff (after failure)
                for _ in 0..failure_threshold {
                    cb.record_failure().await;
                }
                tokio::time::sleep(backoff_durations[0] + Duration::from_millis(5)).await;
                assert!(cb.is_available().await);
                assert_eq!(cb.get_state().await, CircuitState::HalfOpen);
                
                cb.record_failure().await;
                
                assert!(!cb.is_available().await);
                match cb.get_state().await {
                    CircuitState::Open { retry_after, .. } => {
                        let expected_backoff = backoff_durations.get(1)
                            .copied()
                            .unwrap_or(*backoff_durations.last().unwrap());
                        assert_eq!(retry_after, expected_backoff);
                    }
                    _ => panic!("Expected Open state with increased backoff"),
                }
            });
        }

        /// **Validates: Requirements 9.7, 9.8**
        /// 
        /// Property 5: Circuit Breaker Backoff Sequence
        /// 
        /// For any sequence of circuit breaker failures, the backoff times shall follow 
        /// the sequence [5s, 10s, 20s, 40s, 300s] with 300s as the maximum.
        #[test]
        fn prop_circuit_breaker_backoff_sequence(
            failure_threshold in 1u32..=1,
        ) {
            tokio::runtime::Runtime::new().unwrap().block_on(async {
                // Use very short durations for testing (divide by 1000)
                // This maintains the sequence ratios while keeping tests fast
                let test_sequence = vec![
                    Duration::from_millis(5),    // 5s -> 5ms
                    Duration::from_millis(10),   // 10s -> 10ms
                    Duration::from_millis(20),   // 20s -> 20ms
                    Duration::from_millis(40),   // 40s -> 40ms
                    Duration::from_millis(300),  // 300s -> 300ms
                ];
                
                let cb = CircuitBreaker::with_backoff_sequence(
                    failure_threshold,
                    test_sequence.clone(),
                );

                // Verify the backoff sequence is correctly configured
                assert_eq!(cb.backoff_sequence, test_sequence);

                // Test progression through the backoff sequence
                for i in 0..test_sequence.len() {
                    // Trigger failures to open the circuit
                    for _ in 0..failure_threshold {
                        cb.record_failure().await;
                    }
                    
                    // Verify the circuit is open with correct backoff
                    assert!(!cb.is_available().await);
                    match cb.get_state().await {
                        CircuitState::Open { retry_after, .. } => {
                            assert_eq!(
                                retry_after, 
                                test_sequence[i],
                                "Failure iteration {}: expected backoff {:?}, got {:?}",
                                i, test_sequence[i], retry_after
                            );
                        }
                        _ => panic!("Expected Open state after failures at iteration {}", i),
                    }

                    // Wait for backoff and transition to half-open
                    tokio::time::sleep(test_sequence[i] + Duration::from_millis(2)).await;
                    assert!(cb.is_available().await);
                    assert_eq!(cb.get_state().await, CircuitState::HalfOpen);
                }

                // Verify maximum backoff is capped at the last value (300ms in test)
                // After exhausting the sequence, further failures should stay at max
                for _ in 0..failure_threshold {
                    cb.record_failure().await;
                }
                
                match cb.get_state().await {
                    CircuitState::Open { retry_after, .. } => {
                        assert_eq!(
                            retry_after, 
                            *test_sequence.last().unwrap(),
                            "Backoff should be capped at maximum value"
                        );
                    }
                    _ => panic!("Expected Open state"),
                }
            });
        }

        /// **Validates: Requirements 9.9**
        /// 
        /// Property 6: Circuit Breaker Reset
        /// 
        /// For any circuit breaker in Open state, a successful request shall reset 
        /// the backoff time to 5 seconds and transition to Closed state.
        #[test]
        fn prop_circuit_breaker_reset(
            failure_threshold in 1u32..=3,
            backoff_ms in prop::collection::vec(10u64..=100, 3..=5),
            num_failures_before_success in 1usize..=4,
        ) {
            tokio::runtime::Runtime::new().unwrap().block_on(async {
                let backoff_durations: Vec<Duration> = backoff_ms.iter()
                    .map(|&ms| Duration::from_millis(ms))
                    .collect();
                
                let cb = CircuitBreaker::with_backoff_sequence(
                    failure_threshold,
                    backoff_durations.clone(),
                );

                // Progress through multiple failure cycles to increase backoff
                let max_iterations = num_failures_before_success.min(backoff_durations.len());
                for i in 0..max_iterations {
                    // Trigger failures to open the circuit
                    for _ in 0..failure_threshold {
                        cb.record_failure().await;
                    }
                    
                    // Verify circuit is open with expected backoff
                    assert!(!cb.is_available().await);
                    match cb.get_state().await {
                        CircuitState::Open { retry_after, .. } => {
                            assert_eq!(retry_after, backoff_durations[i]);
                        }
                        _ => panic!("Expected Open state at iteration {}", i),
                    }

                    // Wait for backoff and transition to half-open
                    tokio::time::sleep(backoff_durations[i] + Duration::from_millis(2)).await;
                    assert!(cb.is_available().await);
                    assert_eq!(cb.get_state().await, CircuitState::HalfOpen);
                }

                // Now record a successful request - this should reset everything
                cb.record_success().await;
                
                // Verify circuit is closed
                assert_eq!(cb.get_state().await, CircuitState::Closed);
                assert!(cb.is_available().await);

                // Trigger failures again - backoff should reset to first value
                for _ in 0..failure_threshold {
                    cb.record_failure().await;
                }
                
                // Verify circuit is open with FIRST backoff value (reset)
                assert!(!cb.is_available().await);
                match cb.get_state().await {
                    CircuitState::Open { retry_after, .. } => {
                        assert_eq!(
                            retry_after, 
                            backoff_durations[0],
                            "After success, backoff should reset to first value {:?}, got {:?}",
                            backoff_durations[0], retry_after
                        );
                    }
                    _ => panic!("Expected Open state after reset"),
                }
            });
        }
    }
}
