use crate::compression::{
    stats::{sanitize_operational_metadata, CompressionStats, MAX_PROVIDER_LEN},
    CompressionLevel,
};
use crate::structured_output::metrics::StructuredOutputMetrics;
use dashmap::DashMap;
use serde::{Deserialize, Serialize};
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::{Arc, Mutex};
use std::time::{SystemTime, UNIX_EPOCH};

/// Thread-safe metrics tracking for the gateway
#[derive(Debug)]
pub struct Metrics {
    /// Total request count since startup
    request_count: AtomicU64,
    /// Sum of all response times in milliseconds (for average calculation)
    total_response_time_ms: AtomicU64,
    /// Number of completed requests (for average calculation)
    completed_requests: AtomicU64,
    /// Timestamp of last request (for rate calculation)
    last_request_time: AtomicU64,
    /// Request count in last minute window
    requests_last_minute: AtomicU64,
    /// Currently active/in-flight requests
    active_requests: AtomicU64,
    /// Cumulative cost in dollars
    cumulative_cost_cents: AtomicU64, // Store as cents to avoid float atomics
    /// Per-provider metrics
    provider_health: Arc<DashMap<String, ProviderHealth>>,
    /// Per-provider cost tracking
    cost_by_provider_cents: Arc<DashMap<String, AtomicU64>>,
    /// Per-provider retry counts
    retry_count_by_provider: Arc<DashMap<String, AtomicU64>>,
    /// Total retry delay accumulated per provider in milliseconds
    retry_delay_ms_by_provider: Arc<DashMap<String, AtomicU64>>,
    /// Configured provider budget limits in cents
    budget_limit_by_provider_cents: Arc<DashMap<String, AtomicU64>>,
    /// Per-provider budget exhaustion counts
    budget_exhaustions_by_provider: Arc<DashMap<String, AtomicU64>>,
    /// Per-provider unknown-cost response counts
    unknown_cost_by_provider: Arc<DashMap<String, AtomicU64>>,
    /// Per-provider rate-limit exhaustion counts
    rate_limit_exhaustions_by_provider: Arc<DashMap<String, AtomicU64>>,
    /// Cache statistics
    cache_hits: AtomicU64,
    cache_misses: AtomicU64,
    /// Guardrail stage execution counter, keyed by
    /// (pipeline, stage, provider_type, action). `action` is one of
    /// `pass`, `block`, `redact`, `mask`, `replace_with_policy_message`, `error`
    /// (Req 11.1).
    guardrail_stage_executions: Arc<DashMap<GuardrailStageCounterKey, AtomicU64>>,
    /// Guardrail stage latency histogram, keyed by
    /// (pipeline, stage, provider_type), with fixed bucket boundaries
    /// (Req 11.2).
    guardrail_stage_latency: Arc<DashMap<GuardrailLatencyKey, GuardrailLatencyHistogram>>,
    /// Refusal detection counter, keyed by (pipeline, signal) where
    /// signal ∈ {phrase, tool_omission} (Req 12.11).
    guardrail_refusal_detected: Arc<DashMap<(String, String), AtomicU64>>,
    /// Refusal failover outcome counter, keyed by (pipeline, outcome) where
    /// outcome ∈ {recovered, exhausted} (Req 12.11).
    guardrail_refusal_failover: Arc<DashMap<(String, String), AtomicU64>>,
    /// Structured output validation, retry, and latency metrics.
    structured_output: Arc<StructuredOutputMetrics>,
    /// Compression tokens saved counter, keyed by bounded (level, provider).
    compression_tokens_saved: Arc<DashMap<CompressionMetricKey, AtomicU64>>,
    /// Compression ratio histogram, keyed by bounded (level, provider).
    compression_ratio: Arc<DashMap<CompressionMetricKey, CompressionHistogram>>,
    /// Compression duration histogram, keyed by bounded (level, provider).
    compression_duration_seconds: Arc<DashMap<CompressionMetricKey, CompressionHistogram>>,
}

/// Bounded label set for compression metrics: (level, provider).
type CompressionMetricKey = (String, String);

/// Compression ratios are `compressed_tokens / original_tokens`, clamped to
/// `[0, 1]`; an empty original request has ratio `1.0`.
const COMPRESSION_RATIO_BUCKETS: [f64; 8] = [0.1, 0.25, 0.5, 0.75, 0.9, 0.95, 0.99, 1.0];
/// Compression durations in seconds, covering sub-millisecond through slow runs.
const COMPRESSION_DURATION_BUCKETS_SECONDS: [f64; 10] =
    [0.001, 0.005, 0.01, 0.025, 0.05, 0.1, 0.25, 0.5, 1.0, 5.0];
const MAX_COMPRESSION_LEVEL_LABEL_LEN: usize = 16;
const MAX_COMPRESSION_PROVIDER_LABEL_LEN: usize = 64;

/// Histogram state with non-cumulative buckets, rendered cumulatively.
#[derive(Debug)]
struct CompressionHistogram {
    buckets: Box<[AtomicU64]>,
    count: AtomicU64,
    sum_micros: AtomicU64,
}

impl CompressionHistogram {
    fn new(bucket_count: usize) -> Self {
        Self {
            buckets: (0..bucket_count)
                .map(|_| AtomicU64::new(0))
                .collect::<Vec<_>>()
                .into_boxed_slice(),
            count: AtomicU64::new(0),
            sum_micros: AtomicU64::new(0),
        }
    }

    fn observe(&self, value: f64, buckets: &[f64]) {
        let observation = if value.is_finite() {
            value.max(0.0)
        } else {
            0.0
        };
        saturating_atomic_add(&self.count, 1);
        saturating_atomic_add(
            &self.sum_micros,
            (observation * 1_000_000.0).round().min(u64::MAX as f64) as u64,
        );
        if let Some(index) = buckets.iter().position(|boundary| observation <= *boundary) {
            saturating_atomic_add(&self.buckets[index], 1);
        }
    }
}

fn saturating_atomic_add(target: &AtomicU64, amount: u64) {
    let _ = target.fetch_update(Ordering::Relaxed, Ordering::Relaxed, |current| {
        Some(current.saturating_add(amount))
    });
}

fn compression_level_label(level: CompressionLevel) -> &'static str {
    match level {
        CompressionLevel::None => "none",
        CompressionLevel::Lite => "lite",
        CompressionLevel::Standard => "standard",
        CompressionLevel::Aggressive => "aggressive",
        CompressionLevel::Ultra => "ultra",
        CompressionLevel::Rtk => "rtk",
        CompressionLevel::Stacked => "stacked",
    }
}

fn bounded_label(value: &str, max_bytes: usize) -> String {
    let mut end = 0;
    for (index, character) in value.char_indices() {
        let next = index + character.len_utf8();
        if next > max_bytes {
            break;
        }
        end = next;
    }
    if value.len() <= max_bytes {
        value.to_owned()
    } else {
        value[..end].to_owned()
    }
}

fn escape_prometheus_label(value: &str) -> String {
    let mut escaped = String::with_capacity(value.len());
    for character in value.chars() {
        match character {
            '\\' => escaped.push_str("\\\\"),
            '"' => escaped.push_str("\\\""),
            '\n' => escaped.push_str("\\n"),
            '\r' => escaped.push_str("\\r"),
            character if character.is_control() => escaped.push(' '),
            character => escaped.push(character),
        }
    }
    escaped
}

/// Label set for the guardrail stage execution counter (Req 11.1).
type GuardrailStageCounterKey = (String, String, String, String);

/// Label set for the guardrail stage latency histogram (Req 11.2).
type GuardrailLatencyKey = (String, String, String);

/// Upper bucket boundaries (inclusive, milliseconds) for the guardrail stage
/// latency histogram (Req 11.2). Observations greater than the last boundary
/// fall only into the implicit `+Inf` bucket.
const GUARDRAIL_LATENCY_BUCKETS_MS: [f64; 10] = [
    5.0, 10.0, 25.0, 50.0, 100.0, 250.0, 500.0, 1000.0, 2500.0, 5000.0,
];

/// Per-label-set latency histogram. Bucket counts are stored non-cumulatively
/// and rendered cumulatively in Prometheus exposition. The latency sum is kept
/// in microseconds to preserve sub-millisecond precision without float atomics.
#[derive(Debug)]
struct GuardrailLatencyHistogram {
    /// Non-cumulative per-bucket observation counts, aligned with
    /// `GUARDRAIL_LATENCY_BUCKETS_MS`.
    buckets: [AtomicU64; 10],
    /// Total observation count (equals the `+Inf` bucket in exposition).
    count: AtomicU64,
    /// Sum of observed latencies in microseconds.
    sum_micros: AtomicU64,
}

impl GuardrailLatencyHistogram {
    fn new() -> Self {
        Self {
            buckets: std::array::from_fn(|_| AtomicU64::new(0)),
            count: AtomicU64::new(0),
            sum_micros: AtomicU64::new(0),
        }
    }

    /// Record a single latency observation in milliseconds.
    fn observe(&self, latency_ms: f64) {
        let clamped = latency_ms.max(0.0);
        self.count.fetch_add(1, Ordering::Relaxed);
        self.sum_micros
            .fetch_add((clamped * 1000.0).round() as u64, Ordering::Relaxed);
        if let Some(idx) = GUARDRAIL_LATENCY_BUCKETS_MS
            .iter()
            .position(|&boundary| clamped <= boundary)
        {
            self.buckets[idx].fetch_add(1, Ordering::Relaxed);
        }
    }
}

/// Per-provider health tracking
#[derive(Debug)]
pub struct ProviderHealth {
    /// Total requests to this provider
    pub total_requests: AtomicU64,
    /// Successful requests
    pub successful_requests: AtomicU64,
    /// Failed requests
    pub failed_requests: AtomicU64,
    /// Sum of response times for average calculation
    pub total_response_time_ms: AtomicU64,
    /// Last successful request timestamp (Unix epoch seconds)
    pub last_success_timestamp: AtomicU64,
    /// Last failed request timestamp (Unix epoch seconds)
    pub last_failure_timestamp: AtomicU64,
    /// Human-friendly description of the most recent failure. Cleared
    /// when the provider next succeeds. Surfaced by the dashboard so
    /// operators can see why a provider is currently failing without
    /// digging through logs.
    pub last_failure_reason: Mutex<Option<String>>,
    /// Unix epoch seconds at which an upstream-driven cooldown
    /// (e.g. a `Retry-After` from a 429) expires. `None` when no
    /// cooldown is active. Used to render a countdown in the UI.
    pub cooldown_until_timestamp: AtomicU64,
}

/// Snapshot of current metrics for serialization
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MetricsSnapshot {
    pub request_count: u64,
    pub avg_response_time_ms: f64,
    pub request_rate_per_min: f64,
    pub provider_health: Vec<ProviderHealthSnapshot>,
    pub active_requests: u64,
    pub cumulative_cost: f64,
    pub cost_by_provider: Vec<(String, f64)>,
    #[serde(default)]
    pub retry_count_by_provider: Vec<(String, u64)>,
    #[serde(default)]
    pub retry_delay_ms_by_provider: Vec<(String, u64)>,
    #[serde(default)]
    pub budget_limit_by_provider: Vec<(String, f64)>,
    #[serde(default)]
    pub budget_exhaustions_by_provider: Vec<(String, u64)>,
    #[serde(default)]
    pub unknown_cost_by_provider: Vec<(String, u64)>,
    #[serde(default)]
    pub rate_limit_exhaustions_by_provider: Vec<(String, u64)>,
    pub cache_hit_rate: Option<f64>,
    /// Per-model circuit breaker states: Vec of (key, state) where key is "provider:model"
    /// and state is "closed", "open", or "half_open".
    #[serde(default)]
    pub circuit_breaker_states: Vec<(String, String)>,
}

impl MetricsSnapshot {
    /// Enrich provider health entries with circuit breaker states.
    ///
    /// `cb_states` is a list of `(key, state_label)` where key is
    /// `"provider:model"` and state_label is `"closed"`, `"open"`, or `"half_open"`.
    ///
    /// For each provider in the snapshot, we check if any circuit breaker
    /// key starting with that provider name is open/half_open.
    #[allow(dead_code)] // Called from dashboard handlers via axum routing
    pub fn enrich_circuit_breaker_states(&mut self, cb_states: &[(String, String)]) {
        for ph in &mut self.provider_health {
            // Find the matching circuit breaker state.
            // CB keys are "provider:model" now; match by provider name prefix.
            let worst_state = cb_states
                .iter()
                .filter(|(key, _)| key.starts_with(&ph.provider))
                .map(|(_, state)| state.as_str())
                .fold("closed", |worst, s| match (worst, s) {
                    (_, "open") | ("open", _) => "open",
                    (_, "half_open") | ("half_open", _) => "half_open",
                    _ => "closed",
                });
            ph.circuit_breaker_state = worst_state.to_string();
        }
    }
}

/// Serializable provider health snapshot
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ProviderHealthSnapshot {
    pub provider: String,
    pub total_requests: u64,
    pub successful_requests: u64,
    pub failed_requests: u64,
    pub success_rate: f64,
    pub avg_response_time_ms: f64,
    pub last_success_timestamp: Option<u64>,
    pub last_failure_timestamp: Option<u64>,
    pub status: HealthStatus,
    /// Circuit breaker state: "closed", "open", or "half_open".
    /// Populated externally by the dashboard/handler layer since Metrics
    /// doesn't own the circuit breakers.
    #[serde(default = "default_cb_state")]
    pub circuit_breaker_state: String,
    /// Human-friendly description of the most recent failure (e.g.
    /// "Rate limited by provider — pausing for ~30s", "Provider returned
    /// an authentication error", "Network timeout"). `None` once the
    /// provider has succeeded again.
    #[serde(default)]
    pub last_failure_reason: Option<String>,
    /// Unix epoch seconds at which an upstream-driven cooldown ends.
    /// Only set while a cooldown is active. The dashboard can use this
    /// to render a live countdown.
    #[serde(default)]
    pub cooldown_until_timestamp: Option<u64>,
}

fn default_cb_state() -> String {
    "closed".to_string()
}

#[derive(Debug, Clone, Copy, Serialize, Deserialize, PartialEq, Eq)]
#[serde(rename_all = "lowercase")]
pub enum HealthStatus {
    Healthy,
    Degraded,
    Unhealthy,
}

impl Metrics {
    pub fn new() -> Self {
        Self {
            request_count: AtomicU64::new(0),
            total_response_time_ms: AtomicU64::new(0),
            completed_requests: AtomicU64::new(0),
            last_request_time: AtomicU64::new(0),
            requests_last_minute: AtomicU64::new(0),
            active_requests: AtomicU64::new(0),
            cumulative_cost_cents: AtomicU64::new(0),
            provider_health: Arc::new(DashMap::new()),
            cost_by_provider_cents: Arc::new(DashMap::new()),
            retry_count_by_provider: Arc::new(DashMap::new()),
            retry_delay_ms_by_provider: Arc::new(DashMap::new()),
            budget_limit_by_provider_cents: Arc::new(DashMap::new()),
            budget_exhaustions_by_provider: Arc::new(DashMap::new()),
            unknown_cost_by_provider: Arc::new(DashMap::new()),
            rate_limit_exhaustions_by_provider: Arc::new(DashMap::new()),
            cache_hits: AtomicU64::new(0),
            cache_misses: AtomicU64::new(0),
            guardrail_stage_executions: Arc::new(DashMap::new()),
            guardrail_stage_latency: Arc::new(DashMap::new()),
            guardrail_refusal_detected: Arc::new(DashMap::new()),
            guardrail_refusal_failover: Arc::new(DashMap::new()),
            structured_output: Arc::new(StructuredOutputMetrics::new()),
            compression_tokens_saved: Arc::new(DashMap::new()),
            compression_ratio: Arc::new(DashMap::new()),
            compression_duration_seconds: Arc::new(DashMap::new()),
        }
    }

    /// Increment request count and mark request as active
    #[inline]
    pub fn start_request(&self) {
        self.request_count.fetch_add(1, Ordering::Relaxed);
        self.active_requests.fetch_add(1, Ordering::Relaxed);

        let now = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .unwrap()
            .as_secs();
        self.last_request_time.store(now, Ordering::Relaxed);
        self.requests_last_minute.fetch_add(1, Ordering::Relaxed);
    }

    /// Record completed request with response time
    #[inline]
    pub fn complete_request(&self, duration_ms: u64) {
        let mut current = self.active_requests.load(Ordering::Relaxed);
        loop {
            if current == 0 {
                tracing::warn!(
                    duration_ms,
                    "Ignoring request completion because active request count is already zero"
                );
                return;
            }
            match self.active_requests.compare_exchange_weak(
                current,
                current - 1,
                Ordering::Relaxed,
                Ordering::Relaxed,
            ) {
                Ok(_) => break,
                Err(actual) => current = actual,
            }
        }
        self.total_response_time_ms
            .fetch_add(duration_ms, Ordering::Relaxed);
        self.completed_requests.fetch_add(1, Ordering::Relaxed);
    }

    /// Record successful provider request
    pub fn record_provider_success(&self, provider: &str, duration_ms: u64) {
        let health = self
            .provider_health
            .entry(provider.to_string())
            .or_insert_with(|| ProviderHealth {
                total_requests: AtomicU64::new(0),
                successful_requests: AtomicU64::new(0),
                failed_requests: AtomicU64::new(0),
                total_response_time_ms: AtomicU64::new(0),
                last_success_timestamp: AtomicU64::new(0),
                last_failure_timestamp: AtomicU64::new(0),
                last_failure_reason: Mutex::new(None),
                cooldown_until_timestamp: AtomicU64::new(0),
            });

        health.total_requests.fetch_add(1, Ordering::Relaxed);
        health.successful_requests.fetch_add(1, Ordering::Relaxed);
        health
            .total_response_time_ms
            .fetch_add(duration_ms, Ordering::Relaxed);

        let now = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .unwrap()
            .as_secs();
        health.last_success_timestamp.store(now, Ordering::Relaxed);

        // Provider is healthy again — clear the last failure reason and any
        // active cooldown marker so the dashboard reflects recovery.
        if let Ok(mut reason) = health.last_failure_reason.lock() {
            *reason = None;
        }
        health.cooldown_until_timestamp.store(0, Ordering::Relaxed);
    }

    /// Record failed provider request
    #[allow(dead_code)]
    pub fn record_provider_failure(&self, provider: &str) {
        self.record_provider_failure_with_reason(provider, None, None);
    }

    /// Record failed provider request with an optional human-friendly
    /// reason and an optional cooldown deadline (Unix epoch seconds).
    ///
    /// `reason` is shown verbatim on the Provider Health dashboard,
    /// so the caller is responsible for keeping it user-readable.
    /// Pass `None` to leave the previous reason untouched, or pass
    /// `Some("")` to clear it.
    pub fn record_provider_failure_with_reason(
        &self,
        provider: &str,
        reason: Option<String>,
        cooldown_until_unix: Option<u64>,
    ) {
        let health = self
            .provider_health
            .entry(provider.to_string())
            .or_insert_with(|| ProviderHealth {
                total_requests: AtomicU64::new(0),
                successful_requests: AtomicU64::new(0),
                failed_requests: AtomicU64::new(0),
                total_response_time_ms: AtomicU64::new(0),
                last_success_timestamp: AtomicU64::new(0),
                last_failure_timestamp: AtomicU64::new(0),
                last_failure_reason: Mutex::new(None),
                cooldown_until_timestamp: AtomicU64::new(0),
            });

        health.total_requests.fetch_add(1, Ordering::Relaxed);
        health.failed_requests.fetch_add(1, Ordering::Relaxed);

        let now = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .unwrap()
            .as_secs();
        health.last_failure_timestamp.store(now, Ordering::Relaxed);

        if let Some(text) = reason {
            if let Ok(mut slot) = health.last_failure_reason.lock() {
                *slot = if text.is_empty() { None } else { Some(text) };
            }
        }
        if let Some(deadline) = cooldown_until_unix {
            health
                .cooldown_until_timestamp
                .store(deadline, Ordering::Relaxed);
        }
    }

    /// Note that an upstream-driven cooldown is active for `provider`,
    /// without recording a new failure (the failure was already recorded
    /// in the same code path). Updates the dashboard's "last failure
    /// reason" to reflect the rate-limit pause and stores the deadline.
    pub fn set_provider_cooldown(&self, provider: &str, reason: String, cooldown_until_unix: u64) {
        let health = self
            .provider_health
            .entry(provider.to_string())
            .or_insert_with(|| ProviderHealth {
                total_requests: AtomicU64::new(0),
                successful_requests: AtomicU64::new(0),
                failed_requests: AtomicU64::new(0),
                total_response_time_ms: AtomicU64::new(0),
                last_success_timestamp: AtomicU64::new(0),
                last_failure_timestamp: AtomicU64::new(0),
                last_failure_reason: Mutex::new(None),
                cooldown_until_timestamp: AtomicU64::new(0),
            });

        if let Ok(mut slot) = health.last_failure_reason.lock() {
            *slot = Some(reason);
        }
        health
            .cooldown_until_timestamp
            .store(cooldown_until_unix, Ordering::Relaxed);
    }

    /// Number of seconds remaining on an upstream-driven cooldown for
    /// `provider`, or `None` if no cooldown is active.
    ///
    /// This is the routing-side authority for "is this provider paused
    /// because it returned a 429 / Retry-After recently?". The metrics
    /// store survives `Router::clear_rate_limiters()` (which is called
    /// on every config hot-reload), so the routing gate must consult
    /// it in addition to the per-`Router` `RateLimiter::cooldown_until`.
    /// Without that, a config save (admin UI / `/admin/config/reload`)
    /// silently re-routes traffic to a provider the dashboard is still
    /// showing as "Pausing for ~23h (rate limited)".
    pub fn provider_cooldown_remaining_secs(&self, provider: &str) -> Option<u64> {
        let entry = self.provider_health.get(provider)?;
        let deadline = entry
            .value()
            .cooldown_until_timestamp
            .load(Ordering::Relaxed);
        if deadline == 0 {
            return None;
        }
        let now = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .map(|d| d.as_secs())
            .unwrap_or(0);
        if deadline > now {
            Some(deadline - now)
        } else {
            None
        }
    }

    /// Clear `provider`'s cooldown deadline and last-failure reason.
    /// Called by the routing layer when a request to the provider
    /// succeeds, so the dashboard reflects recovery immediately.
    pub fn clear_provider_cooldown(&self, provider: &str) {
        if let Some(entry) = self.provider_health.get(provider) {
            entry
                .value()
                .cooldown_until_timestamp
                .store(0, Ordering::Relaxed);
            if let Ok(mut slot) = entry.value().last_failure_reason.lock() {
                *slot = None;
            }
        }
    }

    /// Add cost to cumulative total and per-provider tracking
    pub fn add_cost(&self, provider: &str, cost: f64) {
        let cost_cents = (cost * 100.0) as u64;
        self.cumulative_cost_cents
            .fetch_add(cost_cents, Ordering::Relaxed);

        self.cost_by_provider_cents
            .entry(provider.to_string())
            .or_insert_with(|| AtomicU64::new(0))
            .fetch_add(cost_cents, Ordering::Relaxed);
    }

    /// Record a retry and its applied delay for a provider.
    pub fn record_provider_retry(&self, provider: &str, delay_ms: u64) {
        self.retry_count_by_provider
            .entry(provider.to_string())
            .or_insert_with(|| AtomicU64::new(0))
            .fetch_add(1, Ordering::Relaxed);

        self.retry_delay_ms_by_provider
            .entry(provider.to_string())
            .or_insert_with(|| AtomicU64::new(0))
            .fetch_add(delay_ms, Ordering::Relaxed);
    }

    pub fn set_provider_budget_limit(&self, provider: &str, budget_limit_usd: f64) {
        let budget_limit_cents = (budget_limit_usd * 100.0) as u64;
        self.budget_limit_by_provider_cents
            .entry(provider.to_string())
            .or_insert_with(|| AtomicU64::new(0))
            .store(budget_limit_cents, Ordering::Relaxed);
    }

    pub fn current_provider_cost_usd(&self, provider: &str) -> f64 {
        self.cost_by_provider_cents
            .get(provider)
            .map(|value| value.load(Ordering::Relaxed) as f64 / 100.0)
            .unwrap_or(0.0)
    }

    pub fn record_provider_budget_exhausted(&self, provider: &str) {
        self.budget_exhaustions_by_provider
            .entry(provider.to_string())
            .or_insert_with(|| AtomicU64::new(0))
            .fetch_add(1, Ordering::Relaxed);
    }

    pub fn record_provider_unknown_cost(&self, provider: &str) {
        self.unknown_cost_by_provider
            .entry(provider.to_string())
            .or_insert_with(|| AtomicU64::new(0))
            .fetch_add(1, Ordering::Relaxed);
    }

    pub fn record_provider_rate_limit_exhausted(&self, provider: &str) {
        self.rate_limit_exhaustions_by_provider
            .entry(provider.to_string())
            .or_insert_with(|| AtomicU64::new(0))
            .fetch_add(1, Ordering::Relaxed);
    }

    /// Record a single guardrail stage execution: bumps the execution counter
    /// for `(pipeline, stage, provider_type, action)` and observes `latency_ms`
    /// in the latency histogram for `(pipeline, stage, provider_type)`
    /// (Req 11.1, 11.2).
    ///
    /// `action` must be one of `pass`, `block`, `redact`, `mask`,
    /// `replace_with_policy_message`, or `error`. Recording is best-effort and
    /// never fails the calling request (Req 11.7).
    pub fn record_guardrail_stage(
        &self,
        pipeline: &str,
        stage: &str,
        provider_type: &str,
        action: &str,
        latency_ms: f64,
    ) {
        let counter_key = (
            pipeline.to_string(),
            stage.to_string(),
            provider_type.to_string(),
            action.to_string(),
        );
        self.guardrail_stage_executions
            .entry(counter_key)
            .or_insert_with(|| AtomicU64::new(0))
            .fetch_add(1, Ordering::Relaxed);

        let latency_key = (
            pipeline.to_string(),
            stage.to_string(),
            provider_type.to_string(),
        );
        self.guardrail_stage_latency
            .entry(latency_key)
            .or_insert_with(GuardrailLatencyHistogram::new)
            .observe(latency_ms);
    }

    /// Increment the refusal-detected counter for `(pipeline, signal)` where
    /// `signal` ∈ {phrase, tool_omission} (Req 12.11). Best-effort: never
    /// fails the request (Req 11.7).
    pub fn record_guardrail_refusal_detected(&self, pipeline: &str, signal: &str) {
        let key = (pipeline.to_string(), signal.to_string());
        self.guardrail_refusal_detected
            .entry(key)
            .or_insert_with(|| AtomicU64::new(0))
            .fetch_add(1, Ordering::Relaxed);
    }

    /// Increment the refusal-failover counter for `(pipeline, outcome)` where
    /// `outcome` ∈ {recovered, exhausted} (Req 12.11). Best-effort: never
    /// fails the request (Req 11.7).
    #[allow(dead_code)] // public API; called from GuardrailEngine::record_refusal_failover
    pub fn record_guardrail_refusal_failover(&self, pipeline: &str, outcome: &str) {
        let key = (pipeline.to_string(), outcome.to_string());
        self.guardrail_refusal_failover
            .entry(key)
            .or_insert_with(|| AtomicU64::new(0))
            .fetch_add(1, Ordering::Relaxed);
    }

    /// Append guardrail counter and histogram metrics to `out` in Prometheus
    /// text exposition format, using the `obey_api_guardrail_` name prefix
    /// (Req 11.5). Emits nothing when no guardrail stages have executed.
    ///
    /// Called by the `prometheus_metrics` endpoint handler so guardrail metrics
    /// are exposed alongside the existing gateway metrics.
    pub fn write_guardrail_prometheus(&self, out: &mut String) {
        // Counter: obey_api_guardrail_stage_executions_total (Req 11.1)
        if !self.guardrail_stage_executions.is_empty() {
            // Sort for deterministic exposition ordering.
            let mut rows: Vec<(GuardrailStageCounterKey, u64)> = self
                .guardrail_stage_executions
                .iter()
                .map(|e| (e.key().clone(), e.value().load(Ordering::Relaxed)))
                .collect();
            rows.sort_by(|a, b| a.0.cmp(&b.0));

            out.push_str(
                "# HELP obey_api_guardrail_stage_executions_total \
Total guardrail stage executions by pipeline, stage, provider, and action\n",
            );
            out.push_str("# TYPE obey_api_guardrail_stage_executions_total counter\n");
            for ((pipeline, stage, provider, action), value) in &rows {
                out.push_str(&format!(
                    "obey_api_guardrail_stage_executions_total{{pipeline=\"{}\",stage=\"{}\",provider=\"{}\",action=\"{}\"}} {}\n",
                    pipeline, stage, provider, action, value
                ));
            }
        }

        // Histogram: obey_api_guardrail_stage_latency_ms (Req 11.2)
        if !self.guardrail_stage_latency.is_empty() {
            let mut keys: Vec<GuardrailLatencyKey> = self
                .guardrail_stage_latency
                .iter()
                .map(|e| e.key().clone())
                .collect();
            keys.sort();

            out.push_str(
                "# HELP obey_api_guardrail_stage_latency_ms \
Guardrail stage latency in milliseconds by pipeline, stage, and provider\n",
            );
            out.push_str("# TYPE obey_api_guardrail_stage_latency_ms histogram\n");
            for key in &keys {
                let entry = match self.guardrail_stage_latency.get(key) {
                    Some(entry) => entry,
                    None => continue,
                };
                let hist = entry.value();
                let (pipeline, stage, provider) = key;

                let mut cumulative = 0u64;
                for (idx, boundary) in GUARDRAIL_LATENCY_BUCKETS_MS.iter().enumerate() {
                    cumulative += hist.buckets[idx].load(Ordering::Relaxed);
                    out.push_str(&format!(
                        "obey_api_guardrail_stage_latency_ms_bucket{{pipeline=\"{}\",stage=\"{}\",provider=\"{}\",le=\"{}\"}} {}\n",
                        pipeline, stage, provider, boundary, cumulative
                    ));
                }
                let total = hist.count.load(Ordering::Relaxed);
                out.push_str(&format!(
                    "obey_api_guardrail_stage_latency_ms_bucket{{pipeline=\"{}\",stage=\"{}\",provider=\"{}\",le=\"+Inf\"}} {}\n",
                    pipeline, stage, provider, total
                ));
                let sum_ms = hist.sum_micros.load(Ordering::Relaxed) as f64 / 1000.0;
                out.push_str(&format!(
                    "obey_api_guardrail_stage_latency_ms_sum{{pipeline=\"{}\",stage=\"{}\",provider=\"{}\"}} {}\n",
                    pipeline, stage, provider, sum_ms
                ));
                out.push_str(&format!(
                    "obey_api_guardrail_stage_latency_ms_count{{pipeline=\"{}\",stage=\"{}\",provider=\"{}\"}} {}\n",
                    pipeline, stage, provider, total
                ));
            }
        }

        // Counter: obey_api_guardrail_refusal_detected_total (Req 12.11)
        if !self.guardrail_refusal_detected.is_empty() {
            let mut rows: Vec<((String, String), u64)> = self
                .guardrail_refusal_detected
                .iter()
                .map(|e| (e.key().clone(), e.value().load(Ordering::Relaxed)))
                .collect();
            rows.sort_by(|a, b| a.0.cmp(&b.0));

            out.push_str(
                "# HELP obey_api_guardrail_refusal_detected_total \
Total refusal detections by pipeline and signal\n",
            );
            out.push_str("# TYPE obey_api_guardrail_refusal_detected_total counter\n");
            for ((pipeline, signal), value) in &rows {
                out.push_str(&format!(
                    "obey_api_guardrail_refusal_detected_total{{pipeline=\"{}\",signal=\"{}\"}} {}\n",
                    pipeline, signal, value
                ));
            }
        }

        // Counter: obey_api_guardrail_refusal_failover_total (Req 12.11)
        if !self.guardrail_refusal_failover.is_empty() {
            let mut rows: Vec<((String, String), u64)> = self
                .guardrail_refusal_failover
                .iter()
                .map(|e| (e.key().clone(), e.value().load(Ordering::Relaxed)))
                .collect();
            rows.sort_by(|a, b| a.0.cmp(&b.0));

            out.push_str(
                "# HELP obey_api_guardrail_refusal_failover_total \
Total refusal failover outcomes by pipeline and outcome\n",
            );
            out.push_str("# TYPE obey_api_guardrail_refusal_failover_total counter\n");
            for ((pipeline, outcome), value) in &rows {
                out.push_str(&format!(
                    "obey_api_guardrail_refusal_failover_total{{pipeline=\"{}\",outcome=\"{}\"}} {}\n",
                    pipeline, outcome, value
                ));
            }
        }
    }

    pub fn structured_output_metrics(&self) -> Arc<StructuredOutputMetrics> {
        Arc::clone(&self.structured_output)
    }

    /// Record a structured output validation outcome.
    pub fn record_structured_output_validation(&self, provider: &str, model: &str, status: &str) {
        self.structured_output
            .record_structured_output_validation(provider, model, status);
    }

    /// Record a structured output retry outcome.
    pub fn record_structured_output_retry(&self, provider: &str, model: &str, outcome: &str) {
        self.structured_output
            .record_structured_output_retry(provider, model, outcome);
    }

    /// Observe structured output validation and retry latency in milliseconds.
    pub fn observe_structured_output_latency(&self, provider: &str, model: &str, latency_ms: f64) {
        self.structured_output
            .observe_structured_output_latency(provider, model, latency_ms);
    }

    /// Append structured output metrics in Prometheus text format.
    pub fn write_structured_output_prometheus(&self, out: &mut String) {
        self.structured_output
            .write_structured_output_prometheus(out);
    }

    /// Record content-free compression metrics for one pipeline operation.
    pub fn record_compression(&self, stats: &CompressionStats) {
        let key = (
            bounded_label(
                compression_level_label(stats.level),
                MAX_COMPRESSION_LEVEL_LABEL_LEN,
            ),
            bounded_label(
                &sanitize_operational_metadata(&stats.provider, MAX_PROVIDER_LEN),
                MAX_COMPRESSION_PROVIDER_LABEL_LEN,
            ),
        );
        let tokens_saved = u64::from(stats.tokens_saved());
        let ratio = if stats.original_tokens == 0 {
            1.0
        } else {
            (f64::from(stats.compressed_tokens) / f64::from(stats.original_tokens)).clamp(0.0, 1.0)
        };
        let duration_seconds = stats.compression_time_ms as f64 / 1000.0;

        let counter = self
            .compression_tokens_saved
            .entry(key.clone())
            .or_insert_with(|| AtomicU64::new(0));
        saturating_atomic_add(counter.value(), tokens_saved);
        self.compression_ratio
            .entry(key.clone())
            .or_insert_with(|| CompressionHistogram::new(COMPRESSION_RATIO_BUCKETS.len()))
            .observe(ratio, &COMPRESSION_RATIO_BUCKETS);
        self.compression_duration_seconds
            .entry(key)
            .or_insert_with(|| {
                CompressionHistogram::new(COMPRESSION_DURATION_BUCKETS_SECONDS.len())
            })
            .observe(duration_seconds, &COMPRESSION_DURATION_BUCKETS_SECONDS);
    }

    /// Append compression counter and histograms in Prometheus text format.
    /// The ratio is `compressed_tokens / original_tokens` in `[0, 1]`; when
    /// `original_tokens` is zero, the observed ratio is `1.0`.
    pub fn write_compression_prometheus(&self, out: &mut String) {
        if !self.compression_tokens_saved.is_empty() {
            let mut rows: Vec<(CompressionMetricKey, u64)> = self
                .compression_tokens_saved
                .iter()
                .map(|entry| (entry.key().clone(), entry.value().load(Ordering::Relaxed)))
                .collect();
            rows.sort_by(|left, right| left.0.cmp(&right.0));
            out.push_str("# HELP obey_compression_tokens_saved_total Total tokens saved by compression operations\n");
            out.push_str("# TYPE obey_compression_tokens_saved_total counter\n");
            for ((level, provider), value) in rows {
                out.push_str(&format!(
                    "obey_compression_tokens_saved_total{{level=\"{}\",provider=\"{}\"}} {}\n",
                    escape_prometheus_label(&level),
                    escape_prometheus_label(&provider),
                    value
                ));
            }
        }
        self.write_compression_histogram(
            out,
            "obey_compression_ratio",
            "Compressed token ratio (compressed_tokens / original_tokens; 1.0 when original_tokens is zero)",
            &self.compression_ratio,
            &COMPRESSION_RATIO_BUCKETS,
        );
        self.write_compression_histogram(
            out,
            "obey_compression_duration_seconds",
            "Compression operation duration in seconds",
            &self.compression_duration_seconds,
            &COMPRESSION_DURATION_BUCKETS_SECONDS,
        );
    }

    fn write_compression_histogram(
        &self,
        out: &mut String,
        name: &str,
        help: &str,
        values: &DashMap<CompressionMetricKey, CompressionHistogram>,
        buckets: &[f64],
    ) {
        if values.is_empty() {
            return;
        }
        let mut keys: Vec<CompressionMetricKey> =
            values.iter().map(|entry| entry.key().clone()).collect();
        keys.sort();
        out.push_str(&format!("# HELP {name} {help}\n"));
        out.push_str(&format!("# TYPE {name} histogram\n"));
        for key in keys {
            let Some(entry) = values.get(&key) else {
                continue;
            };
            let histogram = entry.value();
            let level = escape_prometheus_label(&key.0);
            let provider = escape_prometheus_label(&key.1);
            let mut cumulative = 0u64;
            for (index, boundary) in buckets.iter().enumerate() {
                cumulative =
                    cumulative.saturating_add(histogram.buckets[index].load(Ordering::Relaxed));
                out.push_str(&format!(
                    "{name}_bucket{{level=\"{level}\",provider=\"{provider}\",le=\"{boundary}\"}} {cumulative}\n"
                ));
            }
            let count = histogram.count.load(Ordering::Relaxed);
            out.push_str(&format!(
                "{name}_bucket{{level=\"{level}\",provider=\"{provider}\",le=\"+Inf\"}} {count}\n"
            ));
            let sum = histogram.sum_micros.load(Ordering::Relaxed) as f64 / 1_000_000.0;
            out.push_str(&format!(
                "{name}_sum{{level=\"{level}\",provider=\"{provider}\"}} {sum}\n"
            ));
            out.push_str(&format!(
                "{name}_count{{level=\"{level}\",provider=\"{provider}\"}} {count}\n"
            ));
        }
    }

    /// Record cache hit
    #[inline]
    pub fn record_cache_hit(&self) {
        self.cache_hits.fetch_add(1, Ordering::Relaxed);
    }

    /// Record cache miss
    #[inline]
    pub fn record_cache_miss(&self) {
        self.cache_misses.fetch_add(1, Ordering::Relaxed);
    }

    /// Get current metrics snapshot
    pub fn snapshot(&self) -> MetricsSnapshot {
        let completed = self.completed_requests.load(Ordering::Relaxed);
        let avg_response_time_ms = if completed > 0 {
            self.total_response_time_ms.load(Ordering::Relaxed) as f64 / completed as f64
        } else {
            0.0
        };

        let provider_health: Vec<ProviderHealthSnapshot> = self
            .provider_health
            .iter()
            .map(|entry| {
                let provider = entry.key().clone();
                let health = entry.value();

                let total = health.total_requests.load(Ordering::Relaxed);
                let successful = health.successful_requests.load(Ordering::Relaxed);
                let failed = health.failed_requests.load(Ordering::Relaxed);

                let success_rate = if total > 0 {
                    successful as f64 / total as f64
                } else {
                    0.0
                };

                let avg_response_time_ms = if successful > 0 {
                    health.total_response_time_ms.load(Ordering::Relaxed) as f64 / successful as f64
                } else {
                    0.0
                };

                let last_success = health.last_success_timestamp.load(Ordering::Relaxed);
                let last_failure = health.last_failure_timestamp.load(Ordering::Relaxed);

                let status = if success_rate >= 0.9 {
                    HealthStatus::Healthy
                } else if success_rate >= 0.5 {
                    HealthStatus::Degraded
                } else {
                    HealthStatus::Unhealthy
                };

                let last_failure_reason = health
                    .last_failure_reason
                    .lock()
                    .ok()
                    .and_then(|guard| guard.clone());

                let cooldown_until_raw = health.cooldown_until_timestamp.load(Ordering::Relaxed);
                let now_secs = SystemTime::now()
                    .duration_since(UNIX_EPOCH)
                    .map(|d| d.as_secs())
                    .unwrap_or(0);
                // Treat past cooldowns as inactive: don't render stale
                // "Pausing until..." text on the dashboard.
                let cooldown_until_timestamp = if cooldown_until_raw > now_secs {
                    Some(cooldown_until_raw)
                } else {
                    None
                };

                ProviderHealthSnapshot {
                    provider,
                    total_requests: total,
                    successful_requests: successful,
                    failed_requests: failed,
                    success_rate,
                    avg_response_time_ms,
                    last_success_timestamp: if last_success > 0 {
                        Some(last_success)
                    } else {
                        None
                    },
                    last_failure_timestamp: if last_failure > 0 {
                        Some(last_failure)
                    } else {
                        None
                    },
                    status,
                    circuit_breaker_state: "closed".to_string(),
                    last_failure_reason,
                    cooldown_until_timestamp,
                }
            })
            .collect();

        let cost_by_provider: Vec<(String, f64)> = self
            .cost_by_provider_cents
            .iter()
            .map(|entry| {
                let provider = entry.key().clone();
                let cents = entry.value().load(Ordering::Relaxed);
                (provider, cents as f64 / 100.0)
            })
            .collect();

        let retry_count_by_provider: Vec<(String, u64)> = self
            .retry_count_by_provider
            .iter()
            .map(|entry| (entry.key().clone(), entry.value().load(Ordering::Relaxed)))
            .collect();

        let retry_delay_ms_by_provider: Vec<(String, u64)> = self
            .retry_delay_ms_by_provider
            .iter()
            .map(|entry| (entry.key().clone(), entry.value().load(Ordering::Relaxed)))
            .collect();

        let budget_limit_by_provider: Vec<(String, f64)> = self
            .budget_limit_by_provider_cents
            .iter()
            .map(|entry| {
                (
                    entry.key().clone(),
                    entry.value().load(Ordering::Relaxed) as f64 / 100.0,
                )
            })
            .collect();

        let budget_exhaustions_by_provider: Vec<(String, u64)> = self
            .budget_exhaustions_by_provider
            .iter()
            .map(|entry| (entry.key().clone(), entry.value().load(Ordering::Relaxed)))
            .collect();

        let unknown_cost_by_provider: Vec<(String, u64)> = self
            .unknown_cost_by_provider
            .iter()
            .map(|entry| (entry.key().clone(), entry.value().load(Ordering::Relaxed)))
            .collect();

        let rate_limit_exhaustions_by_provider: Vec<(String, u64)> = self
            .rate_limit_exhaustions_by_provider
            .iter()
            .map(|entry| (entry.key().clone(), entry.value().load(Ordering::Relaxed)))
            .collect();

        let cache_hits = self.cache_hits.load(Ordering::Relaxed);
        let cache_misses = self.cache_misses.load(Ordering::Relaxed);
        let cache_hit_rate = if cache_hits + cache_misses > 0 {
            Some(cache_hits as f64 / (cache_hits + cache_misses) as f64)
        } else {
            None
        };

        MetricsSnapshot {
            request_count: self.request_count.load(Ordering::Relaxed),
            avg_response_time_ms,
            request_rate_per_min: self.requests_last_minute.load(Ordering::Relaxed) as f64,
            provider_health,
            active_requests: self.active_requests.load(Ordering::Relaxed),
            cumulative_cost: self.cumulative_cost_cents.load(Ordering::Relaxed) as f64 / 100.0,
            cost_by_provider,
            retry_count_by_provider,
            retry_delay_ms_by_provider,
            budget_limit_by_provider,
            budget_exhaustions_by_provider,
            unknown_cost_by_provider,
            rate_limit_exhaustions_by_provider,
            cache_hit_rate,
            circuit_breaker_states: Vec::new(),
        }
    }

    /// Reset per-minute request counter (should be called every minute)
    pub fn reset_minute_counter(&self) {
        self.requests_last_minute.store(0, Ordering::Relaxed);
    }

    /// Log a final metrics snapshot during graceful shutdown (Req 18.3).
    pub fn flush(&self) {
        let snapshot = self.snapshot();
        tracing::info!(
            request_count = snapshot.request_count,
            active_requests = snapshot.active_requests,
            cumulative_cost = snapshot.cumulative_cost,
            "Metrics flushed at shutdown"
        );
    }
}

impl Default for Metrics {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_metrics_initialization() {
        let metrics = Metrics::new();
        let snapshot = metrics.snapshot();

        assert_eq!(snapshot.request_count, 0);
        assert_eq!(snapshot.active_requests, 0);
        assert_eq!(snapshot.cumulative_cost, 0.0);
        assert_eq!(snapshot.cache_hit_rate, None);
    }

    #[test]
    fn test_request_tracking() {
        let metrics = Metrics::new();

        metrics.start_request();
        assert_eq!(metrics.snapshot().request_count, 1);
        assert_eq!(metrics.snapshot().active_requests, 1);

        metrics.complete_request(100);
        assert_eq!(metrics.snapshot().active_requests, 0);
        assert_eq!(metrics.snapshot().avg_response_time_ms, 100.0);
    }

    #[test]
    fn test_request_completion_does_not_underflow_active_requests() {
        let metrics = Metrics::new();

        metrics.complete_request(100);
        assert_eq!(metrics.snapshot().active_requests, 0);
        assert_eq!(metrics.snapshot().avg_response_time_ms, 0.0);

        metrics.start_request();
        metrics.complete_request(50);
        metrics.complete_request(75);
        let snapshot = metrics.snapshot();
        assert_eq!(snapshot.active_requests, 0);
        assert_eq!(snapshot.avg_response_time_ms, 50.0);
    }

    #[test]
    fn test_provider_health_tracking() {
        let metrics = Metrics::new();

        metrics.record_provider_success("provider1", 50);
        metrics.record_provider_success("provider1", 150);
        metrics.record_provider_failure("provider1");

        let snapshot = metrics.snapshot();
        let health = snapshot
            .provider_health
            .iter()
            .find(|h| h.provider == "provider1")
            .unwrap();

        assert_eq!(health.total_requests, 3);
        assert_eq!(health.successful_requests, 2);
        assert_eq!(health.failed_requests, 1);
        assert!((health.success_rate - 0.666).abs() < 0.01);
        assert_eq!(health.avg_response_time_ms, 100.0);
        assert_eq!(health.status, HealthStatus::Degraded);
    }

    #[test]
    fn test_cost_tracking() {
        let metrics = Metrics::new();

        metrics.add_cost("provider1", 0.05);
        metrics.add_cost("provider2", 0.10);
        metrics.add_cost("provider1", 0.03);

        let snapshot = metrics.snapshot();
        assert_eq!(snapshot.cumulative_cost, 0.18);

        let provider1_cost = snapshot
            .cost_by_provider
            .iter()
            .find(|(p, _)| p == "provider1")
            .map(|(_, c)| *c)
            .unwrap();
        assert_eq!(provider1_cost, 0.08);
    }

    #[test]
    fn test_retry_tracking() {
        let metrics = Metrics::new();

        metrics.record_provider_retry("provider1", 1200);
        metrics.record_provider_retry("provider1", 800);

        let snapshot = metrics.snapshot();
        let retry_count = snapshot
            .retry_count_by_provider
            .iter()
            .find(|(p, _)| p == "provider1")
            .map(|(_, c)| *c)
            .unwrap();
        let retry_delay = snapshot
            .retry_delay_ms_by_provider
            .iter()
            .find(|(p, _)| p == "provider1")
            .map(|(_, c)| *c)
            .unwrap();

        assert_eq!(retry_count, 2);
        assert_eq!(retry_delay, 2000);
    }

    #[test]
    fn test_budget_and_unknown_cost_tracking() {
        let metrics = Metrics::new();

        metrics.set_provider_budget_limit("provider1", 12.5);
        metrics.record_provider_budget_exhausted("provider1");
        metrics.record_provider_unknown_cost("provider1");
        metrics.record_provider_rate_limit_exhausted("provider1");

        let snapshot = metrics.snapshot();
        let budget_limit = snapshot
            .budget_limit_by_provider
            .iter()
            .find(|(p, _)| p == "provider1")
            .map(|(_, c)| *c)
            .unwrap();
        let budget_exhaustions = snapshot
            .budget_exhaustions_by_provider
            .iter()
            .find(|(p, _)| p == "provider1")
            .map(|(_, c)| *c)
            .unwrap();
        let unknown_cost = snapshot
            .unknown_cost_by_provider
            .iter()
            .find(|(p, _)| p == "provider1")
            .map(|(_, c)| *c)
            .unwrap();
        let rate_limit_exhaustions = snapshot
            .rate_limit_exhaustions_by_provider
            .iter()
            .find(|(p, _)| p == "provider1")
            .map(|(_, c)| *c)
            .unwrap();

        assert_eq!(budget_limit, 12.5);
        assert_eq!(budget_exhaustions, 1);
        assert_eq!(unknown_cost, 1);
        assert_eq!(rate_limit_exhaustions, 1);
    }

    #[test]
    fn test_cache_hit_rate() {
        let metrics = Metrics::new();

        metrics.record_cache_hit();
        metrics.record_cache_hit();
        metrics.record_cache_miss();

        let snapshot = metrics.snapshot();
        assert_eq!(snapshot.cache_hit_rate, Some(2.0 / 3.0));
    }

    #[test]
    fn structured_output_prometheus_is_exposed_through_metrics() {
        let metrics = Metrics::new();
        metrics.record_structured_output_validation("openai", "gpt-4o", "pass");
        metrics.record_structured_output_retry("openai", "gpt-4o", "recovered");
        metrics.observe_structured_output_latency("openai", "gpt-4o", 12.5);

        let mut out = String::new();
        metrics.write_structured_output_prometheus(&mut out);

        assert!(out.contains(
            "# HELP obey_api_structured_output_validations_total Structured output validation outcomes by provider, model, and status"
        ));
        assert!(out.contains("# TYPE obey_api_structured_output_validations_total counter"));
        assert!(out.contains(
            "obey_api_structured_output_validations_total{provider=\"openai\",model=\"gpt-4o\",status=\"pass\"} 1"
        ));
        assert!(out.contains("# TYPE obey_api_structured_output_retries_total counter"));
        assert!(out.contains(
            "obey_api_structured_output_retries_total{provider=\"openai\",model=\"gpt-4o\",outcome=\"recovered\"} 1"
        ));
        assert!(out.contains("# TYPE obey_api_structured_output_latency_ms histogram"));
        assert!(out.contains(
            "obey_api_structured_output_latency_ms_count{provider=\"openai\",model=\"gpt-4o\"} 1"
        ));
    }

    #[test]
    fn test_guardrail_stage_counter_exposition() {
        let metrics = Metrics::new();

        metrics.record_guardrail_stage("standard", "pii-redact", "presidio", "redact", 12.0);
        metrics.record_guardrail_stage("standard", "pii-redact", "presidio", "redact", 8.0);
        metrics.record_guardrail_stage("standard", "secret-block", "regex", "block", 3.0);

        let mut out = String::new();
        metrics.write_guardrail_prometheus(&mut out);

        // Counter type + prefixed name (Req 11.1, 11.5)
        assert!(out.contains("# TYPE obey_api_guardrail_stage_executions_total counter"));
        // Two executions of the redact stage, one of the block stage.
        assert!(out.contains(
            "obey_api_guardrail_stage_executions_total{pipeline=\"standard\",stage=\"pii-redact\",provider=\"presidio\",action=\"redact\"} 2"
        ));
        assert!(out.contains(
            "obey_api_guardrail_stage_executions_total{pipeline=\"standard\",stage=\"secret-block\",provider=\"regex\",action=\"block\"} 1"
        ));
    }

    #[test]
    fn test_guardrail_stage_latency_histogram_exposition() {
        let metrics = Metrics::new();

        // One observation at 8ms (falls in le="10"), one at 300ms (le="500").
        metrics.record_guardrail_stage("standard", "pii-redact", "presidio", "pass", 8.0);
        metrics.record_guardrail_stage("standard", "pii-redact", "presidio", "pass", 300.0);

        let mut out = String::new();
        metrics.write_guardrail_prometheus(&mut out);

        assert!(out.contains("# TYPE obey_api_guardrail_stage_latency_ms histogram"));
        // Design bucket boundaries are present.
        assert!(out.contains(
            "obey_api_guardrail_stage_latency_ms_bucket{pipeline=\"standard\",stage=\"pii-redact\",provider=\"presidio\",le=\"5\"} 0"
        ));
        // Cumulative: 8ms counted at le="10".
        assert!(out.contains(
            "obey_api_guardrail_stage_latency_ms_bucket{pipeline=\"standard\",stage=\"pii-redact\",provider=\"presidio\",le=\"10\"} 1"
        ));
        // Cumulative: both observations counted by le="500".
        assert!(out.contains(
            "obey_api_guardrail_stage_latency_ms_bucket{pipeline=\"standard\",stage=\"pii-redact\",provider=\"presidio\",le=\"500\"} 2"
        ));
        assert!(out.contains(
            "obey_api_guardrail_stage_latency_ms_bucket{pipeline=\"standard\",stage=\"pii-redact\",provider=\"presidio\",le=\"+Inf\"} 2"
        ));
        assert!(out.contains(
            "obey_api_guardrail_stage_latency_ms_count{pipeline=\"standard\",stage=\"pii-redact\",provider=\"presidio\"} 2"
        ));
        // Sum = 8 + 300 = 308 ms.
        assert!(out.contains(
            "obey_api_guardrail_stage_latency_ms_sum{pipeline=\"standard\",stage=\"pii-redact\",provider=\"presidio\"} 308"
        ));
    }

    #[test]
    fn test_guardrail_prometheus_empty_when_no_stages() {
        let metrics = Metrics::new();
        let mut out = String::new();
        metrics.write_guardrail_prometheus(&mut out);
        assert!(out.is_empty(), "no guardrail stages should emit no metrics");
    }

    #[test]
    fn test_guardrail_latency_over_max_bucket_only_in_inf() {
        let metrics = Metrics::new();
        // 6000ms exceeds the last boundary (5000) → only in +Inf/count.
        metrics.record_guardrail_stage("p", "s", "regex", "error", 6000.0);

        let mut out = String::new();
        metrics.write_guardrail_prometheus(&mut out);

        assert!(out.contains(
            "obey_api_guardrail_stage_latency_ms_bucket{pipeline=\"p\",stage=\"s\",provider=\"regex\",le=\"5000\"} 0"
        ));
        assert!(out.contains(
            "obey_api_guardrail_stage_latency_ms_bucket{pipeline=\"p\",stage=\"s\",provider=\"regex\",le=\"+Inf\"} 1"
        ));
    }

    fn compression_stats(
        level: CompressionLevel,
        provider: &str,
        original_tokens: u32,
        compressed_tokens: u32,
        duration_ms: u64,
    ) -> CompressionStats {
        CompressionStats {
            request_id: "metrics-test".to_owned(),
            level,
            engines_applied: Vec::new(),
            original_tokens,
            compressed_tokens,
            savings_percent: if original_tokens == 0 {
                0.0
            } else {
                f64::from(original_tokens.saturating_sub(compressed_tokens)) * 100.0
                    / f64::from(original_tokens)
            },
            compression_time_ms: duration_ms,
            auto_triggered: false,
            cache_downgrade_applied: false,
            tool_definitions_tokens_saved: 0,
            caveman_applied: false,
            timed_out: false,
            error: false,
            provider: provider.to_owned(),
            model: "model".to_owned(),
            engine_results: Vec::new(),
        }
    }

    #[test]
    fn compression_prometheus_accumulates_counter_and_histograms() {
        let metrics = Metrics::new();
        metrics.record_compression(&compression_stats(
            CompressionLevel::Standard,
            "openai",
            100,
            50,
            25,
        ));
        metrics.record_compression(&compression_stats(
            CompressionLevel::Standard,
            "openai",
            100,
            90,
            200,
        ));

        let mut out = String::new();
        metrics.write_compression_prometheus(&mut out);

        assert!(out.contains("# TYPE obey_compression_tokens_saved_total counter"));
        assert!(out.contains(
            "obey_compression_tokens_saved_total{level=\"standard\",provider=\"openai\"} 60"
        ));
        assert!(out.contains("# TYPE obey_compression_ratio histogram"));
        assert!(out.contains(
            "obey_compression_ratio_bucket{level=\"standard\",provider=\"openai\",le=\"0.5\"} 1"
        ));
        assert!(out.contains(
            "obey_compression_ratio_bucket{level=\"standard\",provider=\"openai\",le=\"0.9\"} 2"
        ));
        assert!(out.contains(
            "obey_compression_ratio_bucket{level=\"standard\",provider=\"openai\",le=\"+Inf\"} 2"
        ));
        assert!(
            out.contains("obey_compression_ratio_sum{level=\"standard\",provider=\"openai\"} 1.4")
        );
        assert!(
            out.contains("obey_compression_ratio_count{level=\"standard\",provider=\"openai\"} 2")
        );
        assert!(out.contains("# TYPE obey_compression_duration_seconds histogram"));
        assert!(out.contains(
            "obey_compression_duration_seconds_bucket{level=\"standard\",provider=\"openai\",le=\"0.025\"} 1"
        ));
        assert!(out.contains(
            "obey_compression_duration_seconds_bucket{level=\"standard\",provider=\"openai\",le=\"0.25\"} 2"
        ));
        assert!(out.contains(
            "obey_compression_duration_seconds_sum{level=\"standard\",provider=\"openai\"} 0.225"
        ));
        assert!(out.contains(
            "obey_compression_duration_seconds_count{level=\"standard\",provider=\"openai\"} 2"
        ));
    }

    #[test]
    fn compression_prometheus_uses_one_for_empty_original_ratio() {
        let metrics = Metrics::new();
        metrics.record_compression(&compression_stats(
            CompressionLevel::None,
            "provider",
            0,
            0,
            0,
        ));

        let mut out = String::new();
        metrics.write_compression_prometheus(&mut out);

        assert!(out.contains(
            "obey_compression_ratio_bucket{level=\"none\",provider=\"provider\",le=\"0.99\"} 0"
        ));
        assert!(out.contains(
            "obey_compression_ratio_bucket{level=\"none\",provider=\"provider\",le=\"1\"} 1"
        ));
        assert!(out.contains("obey_compression_ratio_sum{level=\"none\",provider=\"provider\"} 1"));
    }

    #[test]
    fn compression_prometheus_escapes_and_bounds_labels() {
        let metrics = Metrics::new();
        let provider = format!("{}\\\"\nrest", "x".repeat(60));
        metrics.record_compression(&compression_stats(
            CompressionLevel::Lite,
            &provider,
            10,
            5,
            1,
        ));

        let mut out = String::new();
        metrics.write_compression_prometheus(&mut out);

        assert!(out.contains("level=\"lite\",provider=\""));
        assert!(out.contains("\\\\\\\" r"));
        assert!(!out.contains("\nrest"));
        assert_eq!(escape_prometheus_label("a\\\"\n\r"), "a\\\\\\\"\\n\\r");
    }

    #[test]
    fn compression_prometheus_is_empty_before_recording() {
        let metrics = Metrics::new();
        let mut out = String::new();
        metrics.write_compression_prometheus(&mut out);
        assert!(out.is_empty());
    }

    // Property 33: Cost Calculation
    // **Validates: Requirements 30.1, 30.2**
    use proptest::prelude::*;

    proptest! {
        #![proptest_config(ProptestConfig {
            cases: 100,
            .. ProptestConfig::default()
        })]

        #[test]
        fn prop_cost_calculation_cumulative_equals_sum(
            costs in prop::collection::vec(("[a-z]{1,5}", 0.0f64..1000.0f64), 1..20)
        ) {
            let metrics = Metrics::new();
            let mut expected_total_cents: u64 = 0;
            let mut expected_by_provider: std::collections::HashMap<String, u64> = std::collections::HashMap::new();

            // Add costs and track expected values using same cents conversion as implementation
            for (provider, cost) in &costs {
                metrics.add_cost(provider, *cost);
                let cost_cents = (*cost * 100.0) as u64;
                expected_total_cents += cost_cents;
                *expected_by_provider.entry(provider.clone()).or_insert(0) += cost_cents;
            }

            let snapshot = metrics.snapshot();

            // Verify cumulative cost equals sum of all individual costs
            let expected_total = expected_total_cents as f64 / 100.0;
            assert!((snapshot.cumulative_cost - expected_total).abs() < f64::EPSILON,
                "Cumulative cost {} should equal sum of individual costs {}",
                snapshot.cumulative_cost, expected_total);

            // Verify per-provider costs sum correctly
            for (provider, expected_cents) in &expected_by_provider {
                let expected_cost = *expected_cents as f64 / 100.0;
                let actual_cost = snapshot.cost_by_provider.iter()
                    .find(|(p, _)| p == provider)
                    .map(|(_, c)| *c)
                    .unwrap_or(0.0);

                assert!((actual_cost - expected_cost).abs() < f64::EPSILON,
                    "Provider {} cost {} should equal sum of its costs {}",
                    provider, actual_cost, expected_cost);
            }

            // Verify sum of per-provider costs equals cumulative cost
            let provider_sum_cents: u64 = expected_by_provider.values().sum();
            assert_eq!(provider_sum_cents, expected_total_cents,
                "Sum of per-provider cents should equal cumulative cents");
        }

        // Property 34: Provider Health Status
        // **Validates: Requirements 31.5-31.8**
        #[test]
        fn prop_provider_health_status_derived_from_success_rate(
            successes in 0u64..1000,
            failures in 0u64..1000,
        ) {
            let metrics = Metrics::new();
            let provider = "test-provider";

            // Record success and failure counts
            for _ in 0..successes {
                metrics.record_provider_success(provider, 100);
            }
            for _ in 0..failures {
                metrics.record_provider_failure(provider);
            }

            let snapshot = metrics.snapshot();
            let health = snapshot.provider_health.iter()
                .find(|h| h.provider == provider)
                .expect("Provider should exist in snapshot");

            let total = successes + failures;

            // Skip if no requests (undefined success rate)
            if total == 0 {
                return Ok(());
            }

            let success_rate = successes as f64 / total as f64;

            // Verify health status matches success rate thresholds
            let expected_status = if success_rate >= 0.9 {
                HealthStatus::Healthy
            } else if success_rate >= 0.5 {
                HealthStatus::Degraded
            } else {
                HealthStatus::Unhealthy
            };

            assert_eq!(
                health.status, expected_status,
                "Provider with {} successes and {} failures (success_rate={:.2}) should have status {:?}, got {:?}",
                successes, failures, success_rate, expected_status, health.status
            );

            // Verify success rate calculation
            assert!(
                (health.success_rate - success_rate).abs() < 0.0001,
                "Success rate should be {:.4}, got {:.4}",
                success_rate, health.success_rate
            );
        }
    }

    #[test]
    fn test_guardrail_refusal_detected_counter_exposition() {
        let metrics = Metrics::new();

        metrics.record_guardrail_refusal_detected("safety-pipeline", "phrase");
        metrics.record_guardrail_refusal_detected("safety-pipeline", "phrase");
        metrics.record_guardrail_refusal_detected("safety-pipeline", "tool_omission");

        let mut out = String::new();
        metrics.write_guardrail_prometheus(&mut out);

        assert!(out.contains("# TYPE obey_api_guardrail_refusal_detected_total counter"));
        assert!(out.contains(
            "obey_api_guardrail_refusal_detected_total{pipeline=\"safety-pipeline\",signal=\"phrase\"} 2"
        ));
        assert!(out.contains(
            "obey_api_guardrail_refusal_detected_total{pipeline=\"safety-pipeline\",signal=\"tool_omission\"} 1"
        ));
    }

    #[test]
    fn test_guardrail_refusal_failover_counter_exposition() {
        let metrics = Metrics::new();

        metrics.record_guardrail_refusal_failover("safety-pipeline", "recovered");
        metrics.record_guardrail_refusal_failover("safety-pipeline", "exhausted");
        metrics.record_guardrail_refusal_failover("safety-pipeline", "recovered");

        let mut out = String::new();
        metrics.write_guardrail_prometheus(&mut out);

        assert!(out.contains("# TYPE obey_api_guardrail_refusal_failover_total counter"));
        assert!(out.contains(
            "obey_api_guardrail_refusal_failover_total{pipeline=\"safety-pipeline\",outcome=\"exhausted\"} 1"
        ));
        assert!(out.contains(
            "obey_api_guardrail_refusal_failover_total{pipeline=\"safety-pipeline\",outcome=\"recovered\"} 2"
        ));
    }

    #[test]
    fn test_guardrail_refusal_prometheus_empty_when_no_refusals() {
        let metrics = Metrics::new();
        let mut out = String::new();
        metrics.write_guardrail_prometheus(&mut out);
        // No refusal counters should appear.
        assert!(!out.contains("refusal_detected"));
        assert!(!out.contains("refusal_failover"));
    }
}
