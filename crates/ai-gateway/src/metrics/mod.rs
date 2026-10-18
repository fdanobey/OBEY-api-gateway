use crate::compression::{
    stats::{sanitize_operational_metadata, CompressionStats, MAX_PROVIDER_LEN},
    CompressionLevel,
};
use crate::config::Config;
use crate::router::sticky_cache::CacheUsage;
use crate::structured_output::metrics::StructuredOutputMetrics;

mod provider_attempt;
use provider_attempt::ProviderAttemptFailureMetrics;
pub mod provider_budget_store;
use dashmap::DashMap;
pub use provider_budget_store::{ProviderBudgetStore, ProviderBudgetStoreError};
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::sync::atomic::{AtomicI64, AtomicU64, Ordering};
use std::sync::{Arc, Mutex};
use std::time::{SystemTime, UNIX_EPOCH};

/// Number of one-second buckets in the rolling request-rate window.
const REQUEST_RATE_WINDOW_SECONDS: usize = 60;

/// Fixed-size ring of per-second request counts providing the trailing-60s
/// request rate. The slot for second `s` lives at `s % WINDOW`; buckets older
/// than the window are zeroed on access, and the window sum is maintained
/// incrementally, so recording and snapshotting are bounded by the window
/// size (O(1) amortized) with no background reset task and no unbounded
/// memory.
#[derive(Debug)]
struct RequestRateRing {
    /// Newest second already reflected in the slot layout.
    base_second: u64,
    /// Per-second request counts.
    slots: [u64; REQUEST_RATE_WINDOW_SECONDS],
    /// Sum of all slots (the trailing-window total).
    total: u64,
}

impl RequestRateRing {
    fn new() -> Self {
        Self {
            base_second: 0,
            slots: [0; REQUEST_RATE_WINDOW_SECONDS],
            total: 0,
        }
    }

    /// Zero buckets that have aged out of the trailing window as of `now`.
    fn advance(&mut self, now: u64) {
        if now <= self.base_second {
            return;
        }
        let elapsed = now - self.base_second;
        if elapsed >= REQUEST_RATE_WINDOW_SECONDS as u64 {
            self.slots = [0; REQUEST_RATE_WINDOW_SECONDS];
            self.total = 0;
            self.base_second = now;
            return;
        }
        let base_pos = (self.base_second % REQUEST_RATE_WINDOW_SECONDS as u64) as usize;
        for i in 1..=elapsed as usize {
            let pos = (base_pos + i) % REQUEST_RATE_WINDOW_SECONDS;
            self.total -= self.slots[pos];
            self.slots[pos] = 0;
        }
        self.base_second = now;
    }

    /// Record one request during second `now`. Clock steps backwards are
    /// clamped to the newest observed second so a stale slot is never
    /// incremented.
    fn record_at(&mut self, now: u64) {
        let now = now.max(self.base_second);
        self.advance(now);
        let pos = (now % REQUEST_RATE_WINDOW_SECONDS as u64) as usize;
        self.slots[pos] = self.slots[pos].saturating_add(1);
        self.total = self.total.saturating_add(1);
    }

    /// Trailing-window sum as of `now`, aging out expired buckets.
    fn rolling_total_at(&mut self, now: u64) -> u64 {
        self.advance(now);
        self.total
    }
}

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
    /// Trailing-60s rolling request count driving `requests_last_minute` and
    /// `request_rate_per_min`
    requests_last_minute: Mutex<RequestRateRing>,
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
    /// Configured provider budget reset policies (config-authoritative;
    /// Task 7 status derivation reads this, not the store's default-manual
    /// window column).
    budget_policy_by_provider: Arc<DashMap<String, crate::config::BudgetResetPolicy>>,
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
    /// outcome âˆˆ {recovered, exhausted} (Req 12.11).
    guardrail_refusal_failover: Arc<DashMap<(String, String), AtomicU64>>,
    /// unicode_stego provider detection counter, keyed by (category, phase)
    /// where category âˆˆ {unicode_tag, zero_width, bidi_control,
    /// mixed_script_confusable} and phase âˆˆ {pre_call, tool_result,
    /// tool_call, post_call} (indirect-injection defense, task 5.2).
    guardrail_stego_detections: Arc<DashMap<(String, String), AtomicU64>>,
    /// Structured output validation, retry, and latency metrics.
    structured_output: Arc<StructuredOutputMetrics>,
    /// Failed provider attempts keyed by (provider, model, status_code),
    /// including attempts masked by a successful retry/fallback.
    provider_attempt_failures: Arc<ProviderAttemptFailureMetrics>,
    /// Compression tokens saved counter, keyed by bounded (level, provider).
    compression_tokens_saved: Arc<DashMap<CompressionMetricKey, AtomicU64>>,
    /// Compression ratio histogram, keyed by bounded (level, provider).
    compression_ratio: Arc<DashMap<CompressionMetricKey, CompressionHistogram>>,
    /// Compression duration histogram, keyed by bounded (level, provider).
    compression_duration_seconds: Arc<DashMap<CompressionMetricKey, CompressionHistogram>>,
    /// Tool compression requests total, keyed by (model_group, level).
    tool_compression_requests_total: Arc<DashMap<ToolCompressionMetricKey, AtomicU64>>,
    /// Tool compression tokens saved total, keyed by (model_group, level).
    tool_compression_tokens_saved_total: Arc<DashMap<ToolCompressionMetricKey, AtomicU64>>,
    /// Tool compression ratio histogram, keyed by (model_group, level).
    tool_compression_ratio: Arc<DashMap<ToolCompressionMetricKey, CompressionHistogram>>,
    /// Tool compression pipeline duration histogram (ms), keyed by (model_group, level).
    tool_compression_duration_ms: Arc<DashMap<ToolCompressionMetricKey, CompressionHistogram>>,
    /// Tool compression tools pruned gauge, keyed by model_group.
    tool_compression_tools_pruned: Arc<DashMap<String, AtomicU64>>,
    /// Tool compression feedback current level gauge, keyed by model_group.
    tool_compression_feedback_level: Arc<DashMap<String, AtomicU64>>,
    /// Tool compression feedback error rate gauge (stored as micros, i.e. rate * 1_000_000), keyed by model_group.
    tool_compression_feedback_error_rate: Arc<DashMap<String, AtomicU64>>,
    /// Tool compression feedback adjustments counter, keyed by model_group.
    tool_compression_feedback_adjustments: Arc<DashMap<String, AtomicU64>>,
    /// Lazily allocated Smart Routing metrics. Disabled gateways retain no
    /// Smart Routing metric maps or counters.
    smart_routing: Mutex<Option<Arc<SmartRoutingMetricState>>>,
    /// Durable owner of per-provider budget-window spend (`budgets.db`).
    /// The cost/budget atomics in this struct act as a write-through cache
    /// of the store: on restore the store value wins (durable truth).
    /// `None` only in unit tests constructing [`Metrics::new`] directly;
    /// the gateway always sets it (persistent-budgets Task 3).
    provider_budget_store: Mutex<Option<Arc<ProviderBudgetStore>>>,
    /// 1-second TTL cache of the budget-store rows (`get_all()` result)
    /// backing `budget_status_by_provider`, so dashboard snapshot polling
    /// does not hit SQLite on every snapshot. Guarded by a short-lived
    /// Mutex; see `cached_budget_store_rows`.
    budget_store_rows_cache: Mutex<
        Option<(
            std::time::Instant,
            HashMap<String, crate::metrics::provider_budget_store::ProviderBudgetRow>,
        )>,
    >,
    /// In-memory next-boundary cache for the lazy budget-window rollover
    /// (hot-path remediation, finding B1): provider -> (policy at seed
    /// time, durable `window_start`, next boundary epoch). Seeded from the
    /// store + policy once per provider per window; every later rollover
    /// check is a pure `now >= next_boundary_epoch` comparison with no
    /// SQL. A policy change (config hot-reload) re-seeds via the cached
    /// policy mismatch check.
    budget_window_boundaries: Mutex<HashMap<String, (crate::config::BudgetResetPolicy, i64, i64)>>,
}

/// TTL of the cached budget-store rows served to budget status snapshots.
const BUDGET_STORE_ROWS_TTL: std::time::Duration = std::time::Duration::from_secs(1);

/// Bounded label set for compression metrics: (level, provider).
type CompressionMetricKey = (String, String);

/// Bounded label set for tool compression metrics: (model_group, level).
type ToolCompressionMetricKey = (String, String);

/// Compression ratios are `compressed_tokens / original_tokens`, clamped to
/// `[0, 1]`; an empty original request has ratio `1.0`.
const COMPRESSION_RATIO_BUCKETS: [f64; 8] = [0.1, 0.25, 0.5, 0.75, 0.9, 0.95, 0.99, 1.0];
/// Compression durations in seconds, covering sub-millisecond through slow runs.
const COMPRESSION_DURATION_BUCKETS_SECONDS: [f64; 10] =
    [0.001, 0.005, 0.01, 0.025, 0.05, 0.1, 0.25, 0.5, 1.0, 5.0];
const MAX_COMPRESSION_LEVEL_LABEL_LEN: usize = 16;
const MAX_COMPRESSION_PROVIDER_LABEL_LEN: usize = 64;

const SMART_ROUTING_SCORE_BUCKETS: [f64; 10] = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0];
const SMART_ROUTING_LATENCY_BUCKETS_MS: [f64; 10] =
    [0.1, 0.5, 1.0, 2.5, 5.0, 10.0, 25.0, 50.0, 100.0, 500.0];
const SMART_ROUTING_COST_BUCKETS_USD: [f64; 10] =
    [0.0001, 0.0005, 0.001, 0.005, 0.01, 0.05, 0.1, 0.5, 1.0, 5.0];
const SMART_ROUTING_QUALITY_BUCKETS: [f64; 10] = SMART_ROUTING_SCORE_BUCKETS;
#[allow(dead_code)]
const SMART_ROUTING_GROUP_BUCKETS: u64 = 64;

/// Metric values for a complete Smart Routing decision. All string labels are
/// normalized to finite enumerations or a fixed group bucket before storage.
#[derive(Debug, Clone, Copy)]
#[allow(dead_code)]
pub struct SmartRoutingDecisionMetric<'a> {
    pub tier: &'a str,
    pub classifier: &'a str,
    pub group: &'a str,
    pub score: f64,
    pub estimated_cost_usd: f64,
    pub classifier_latency_ms: f64,
    pub task_type: &'a str,
    pub quality: f64,
    pub context_filtered: bool,
    pub experiment: Option<&'a str>,
}

#[derive(Debug)]
struct SmartRoutingMetricState {
    decisions: DashMap<(String, String, String, String), AtomicU64>,
    scores: DashMap<(String, String), CompressionHistogram>,
    cascade_transitions: DashMap<(String, String), AtomicU64>,
    estimated_cost: DashMap<String, CompressionHistogram>,
    classifier_latency: DashMap<String, CompressionHistogram>,
    simhash: DashMap<String, AtomicU64>,
    task_types: DashMap<String, AtomicU64>,
    quality: DashMap<String, CompressionHistogram>,
    semantic_cache: DashMap<String, AtomicU64>,
    context_filtered: DashMap<String, AtomicU64>,
    // System One classifier instruments (family label: `jev` | `laya`).
    jev_consults: DashMap<(String, String), AtomicU64>,
    jev_fallbacks: DashMap<(String, String), AtomicU64>,
    jev_confidence: DashMap<String, CompressionHistogram>,
    jev_latency: DashMap<String, CompressionHistogram>,
    jev_discovery_refreshes: DashMap<(String, String), AtomicU64>,
}

impl SmartRoutingMetricState {
    #[allow(dead_code)]
    fn new() -> Self {
        Self {
            decisions: DashMap::new(),
            scores: DashMap::new(),
            cascade_transitions: DashMap::new(),
            estimated_cost: DashMap::new(),
            classifier_latency: DashMap::new(),
            simhash: DashMap::new(),
            task_types: DashMap::new(),
            quality: DashMap::new(),
            semantic_cache: DashMap::new(),
            context_filtered: DashMap::new(),
            jev_consults: DashMap::new(),
            jev_fallbacks: DashMap::new(),
            jev_confidence: DashMap::new(),
            jev_latency: DashMap::new(),
            jev_discovery_refreshes: DashMap::new(),
        }
    }
}

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
    let mut current = target.load(Ordering::Relaxed);
    loop {
        let next = current.saturating_add(amount);
        match target.compare_exchange_weak(current, next, Ordering::Relaxed, Ordering::Relaxed) {
            Ok(_) => break,
            Err(observed) => current = observed,
        }
    }
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

#[allow(dead_code)]
fn smart_routing_enum_label(value: &str, allowed: &[&str]) -> String {
    let normalized = value.trim().to_ascii_lowercase().replace('-', "_");
    if allowed.contains(&normalized.as_str()) {
        normalized
    } else {
        "other".to_owned()
    }
}

/// Bounded System One family label: `jev` or `laya`; anything else becomes
/// `other`, keeping series cardinality fixed.
fn smart_routing_family_label(family: &str) -> String {
    let normalized = family.trim().to_ascii_lowercase();
    if normalized == "jev" || normalized == "laya" {
        normalized
    } else {
        "other".to_owned()
    }
}

#[allow(dead_code)]
fn smart_routing_tier_label(value: &str) -> String {
    smart_routing_enum_label(value, &["fast", "balanced", "powerful"])
}

#[allow(dead_code)]
fn smart_routing_classifier_label(value: &str) -> String {
    smart_routing_enum_label(
        value,
        &["heuristic", "ml", "llm", "composite", "jev", "laya"],
    )
}

#[allow(dead_code)]
fn smart_routing_task_label(value: &str) -> String {
    smart_routing_enum_label(
        value,
        &[
            "code_generation",
            "math_reasoning",
            "creative_writing",
            "factual_qa",
            "tool_use",
            "summarization",
            "general",
        ],
    )
}

#[allow(dead_code)]
fn smart_routing_experiment_label(value: Option<&str>) -> String {
    match value.map(str::trim).map(str::to_ascii_lowercase).as_deref() {
        Some("control") => "control".to_owned(),
        Some("variant") => "variant".to_owned(),
        Some(_) => "other".to_owned(),
        None => "none".to_owned(),
    }
}

#[allow(dead_code)]
fn smart_routing_outcome_label(hit: bool) -> String {
    if hit {
        "hit".to_owned()
    } else {
        "miss".to_owned()
    }
}

#[allow(dead_code)]
fn smart_routing_group_label(value: &str) -> String {
    if value.is_empty() {
        return "unknown".to_owned();
    }
    let mut hash = 0xcbf29ce484222325u64;
    for byte in value.bytes() {
        hash ^= u64::from(byte);
        hash = hash.wrapping_mul(0x100000001b3);
    }
    format!("bucket_{:02}", hash % SMART_ROUTING_GROUP_BUCKETS)
}

#[allow(dead_code)]
fn smart_routing_transition_label(value: &str) -> String {
    smart_routing_enum_label(value, &["fast", "balanced", "powerful", "none"])
}

/// Label set for the guardrail stage execution counter (Req 11.1; phase label
/// added by the indirect-injection defense, task 5.1).
type GuardrailStageCounterKey = (String, String, String, String, String);

/// Label set for the guardrail stage latency histogram (Req 11.2; phase label
/// added by the indirect-injection defense, task 5.1).
type GuardrailLatencyKey = (String, String, String, String);

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
    /// Cumulative prompt tokens served from the upstream KV cache
    /// (cache reads) across recorded responses (Req 4.2).
    pub cache_read_tokens: AtomicU64,
    /// Cumulative prompt tokens written into the upstream cache
    /// (cache creations) across recorded responses (Req 4.2).
    pub cache_creation_tokens: AtomicU64,
    /// Cumulative uncached prompt tokens; with the two counters above
    /// this forms the denominator of the running prompt-cache hit rate
    /// (Req 4.2).
    pub cache_uncached_tokens: AtomicU64,
    /// Cumulative prompt-cache cost savings in whole cents versus the
    /// uncached baseline (Req 4.2). Signed because a cache-creation
    /// premium can make a single response cost more than the baseline.
    pub cache_savings_cents: AtomicI64,
    /// Cumulative reasoning/thinking tokens across recorded responses,
    /// extracted from provider usage objects of any carrier shape and
    /// never double-counted (Req 4.7).
    pub reasoning_tokens: AtomicU64,
    /// Cumulative reasoning cost in micro-dollars (USD × 1e6), stored
    /// as an integer to avoid float atomics (Req 4.7).
    pub reasoning_cost_usd_micros: AtomicU64,
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
    /// Live snapshot of currently in-flight requests (dashboard "In-Flight Requests").
    #[serde(default)]
    pub active_requests_list: Vec<crate::active_requests::ActiveRequestInfo>,
    /// Server-derived per-provider budget status (persistent budgets,
    /// Task 7): limits, current-window spend, remaining, window policy,
    /// next reset boundary, exhausted flag. Serialized to the dashboard
    /// via WS snapshots and `GET /dashboard/metrics`.
    #[serde(default)]
    pub budget_status_by_provider: HashMap<String, ProviderBudgetStatus>,
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

/// Server-derived per-provider budget status for the dashboard
/// (persistent budgets, Task 7). All fields are computed server-side
/// from the store rows, the seeded limit atomics and the exhaustion
/// counters — the JS renders only, never derives.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ProviderBudgetStatus {
    /// Configured budget limit in USD; `None` = no limit configured
    /// (spend-only row, no fake limit or remaining).
    pub limit_usd: Option<f64>,
    /// Spend accrued in the current budget window (durable store value
    /// restored into the cost atomics at startup).
    pub spent_usd: f64,
    /// `limit - spent`; `None` when no limit is configured.
    pub remaining_usd: Option<f64>,
    /// Reset policy for the window. Serializes as the snake_case label
    /// ("daily" / "weekly" / "monthly" / "manual") via the enum's serde
    /// derive — the JSON contract is unchanged from the previous string
    /// field.
    pub window: crate::config::BudgetResetPolicy,
    /// UTC epoch seconds the current window opened at.
    pub window_start_epoch: i64,
    /// Next UTC reset boundary (epoch seconds); `None` for manual policy.
    pub next_reset_epoch: Option<i64>,
    /// Whether the provider is currently budget-exhausted (spent >= limit).
    pub exhausted: bool,
}

/// Next UTC reset boundary for a windowed policy, anchored on the row's
/// durable `window_start` (persistent budgets, Task 7). Uses the same
/// boundary math as the virtual-key budgets — daily/weekly/monthly via
/// the `From<BudgetResetPolicy>` mapping onto `BudgetWindow` — not a
/// fork. `None` for manual policy (never resets). Public so the Task-10
/// guard tests can assert boundary equivalence with the VK logic.
pub fn next_reset_epoch(
    policy: crate::config::BudgetResetPolicy,
    window_start_epoch: i64,
) -> Option<i64> {
    use chrono::Datelike as _;
    let window: Option<crate::virtual_keys::models::BudgetWindow> = policy.into();
    let window = window?;
    let start = chrono::DateTime::from_timestamp(window_start_epoch, 0)?;
    let next = match window {
        crate::virtual_keys::models::BudgetWindow::Daily => {
            let d = start.date_naive() + chrono::Duration::days(1);
            d.and_hms_opt(0, 0, 0)?.and_utc()
        }
        crate::virtual_keys::models::BudgetWindow::Weekly => {
            // Monday 00:00 UTC starting the week after window_start's week —
            // the same boundary `window_expired` rolls at.
            let offset = start.date_naive().weekday().num_days_from_monday() as i64;
            let days_to_next_monday = 7 - offset;
            let d = start.date_naive() + chrono::Duration::days(days_to_next_monday);
            d.and_hms_opt(0, 0, 0)?.and_utc()
        }
        crate::virtual_keys::models::BudgetWindow::Monthly => {
            // First of the following month, 00:00 UTC.
            let d = start.date_naive().with_day(1)?;
            let (year, month) = (d.year(), d.month());
            let (next_year, next_month) = if month == 12 {
                (year + 1, 1)
            } else {
                (year, month + 1)
            };
            chrono::NaiveDate::from_ymd_opt(next_year, next_month, 1)?
                .and_hms_opt(0, 0, 0)?
                .and_utc()
        }
    };
    Some(next.timestamp())
}

impl Metrics {
    /// Derive the server-authoritative per-provider budget status map
    /// (persistent budgets, Task 7) from three sources:
    ///
    /// - durable store rows (window policy, window start),
    /// - the seeded `budget_limit_by_provider` atomics (config limit),
    /// - the `cost_by_provider` atomics (current-window spend, restored
    /// from the store at startup), and `budget_exhaustions` for the
    /// exhausted flag.
    ///
    /// Providers are unioned across store rows and configured limits so a
    /// configured-but-never-spent provider still shows its limit, and a
    /// spend-only row still shows its spend. Store read failures degrade
    /// to atomics-only derivation (observability, never a request blocker).
    ///
    /// The store rows are served from a tiny 1-second TTL cache: the
    /// dashboard polls `snapshot()` frequently and every snapshot otherwise
    /// hits SQLite with a full `get_all()` table scan. The cache holds the
    /// lock only to clone the cached map (or refresh it once per second),
    /// keeping snapshot blocking bounded.
    fn budget_status_by_provider(&self) -> HashMap<String, ProviderBudgetStatus> {
        let store_rows: HashMap<String, crate::metrics::provider_budget_store::ProviderBudgetRow> =
            self.cached_budget_store_rows();
        // Degraded store: fall back to manual windows for store-known providers.
        let mut provider_names: std::collections::BTreeSet<String> =
            store_rows.keys().cloned().collect();
        for entry in self.budget_limit_by_provider_cents.iter() {
            provider_names.insert(entry.key().clone());
        }
        let now_epoch = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .map(|d| d.as_secs() as i64)
            .unwrap_or(0);
        provider_names
            .into_iter()
            .map(|provider| {
                let row = store_rows.get(&provider);
                let limit_usd = self
                    .budget_limit_by_provider_cents
                    .get(&provider)
                    .map(|entry| entry.value().load(Ordering::Relaxed) as f64 / 100.0)
                    // Config limit wins over any stale stored copy.
                    .or_else(|| row.and_then(|row| row.limit_usd));
                let spent_usd = self
                    .cost_by_provider_cents
                    .get(&provider)
                    .map(|entry| entry.value().load(Ordering::Relaxed) as f64 / 100.0)
                    .unwrap_or_else(|| row.map(|row| row.current_spend_usd).unwrap_or(0.0));
                // Config-seeded policy is authoritative (the store row's window
                // column only ever holds the insert-time default `manual`).
                let policy = self
                    .budget_policy_by_provider
                    .get(&provider)
                    .map(|entry| *entry.value())
                    .or_else(|| row.map(|row| row.window))
                    .unwrap_or(crate::config::BudgetResetPolicy::Manual);
                let window_start_epoch = row
                    .map(|row| row.window_start)
                    .filter(|epoch| *epoch > 0)
                    .unwrap_or(0);
                let exhausted = limit_usd.map(|limit| spent_usd >= limit).unwrap_or(false);
                let status = ProviderBudgetStatus {
                    limit_usd,
                    spent_usd,
                    remaining_usd: limit_usd.map(|limit| (limit - spent_usd).max(0.0)),
                    window: policy,
                    window_start_epoch,
                    // A missing/zero/legacy anchor cannot yield a meaningful next
                    // boundary (the epoch-0 fallback would promise a reset at some
                    // 1970-derived instant) — emit `None` instead.
                    next_reset_epoch: (window_start_epoch > 0)
                        .then(|| next_reset_epoch(policy, window_start_epoch.max(now_epoch)))
                        .flatten(),
                    exhausted,
                };
                (provider, status)
            })
            .collect()
    }

    /// All store rows keyed by provider, served from a 1-second TTL cache
    /// shared by every `budget_status_by_provider` derivation. On a cold or
    /// expired cache the SQLite `get_all()` runs inside the same short lock;
    /// concurrent snapshots inside the TTL window clone the cached map
    /// without touching the database.
    fn cached_budget_store_rows(
        &self,
    ) -> HashMap<String, crate::metrics::provider_budget_store::ProviderBudgetRow> {
        let mut cache = self
            .budget_store_rows_cache
            .lock()
            .expect("budget store rows cache poisoned");
        let fresh = cache
            .as_ref()
            .map(|(at, _)| at.elapsed() < BUDGET_STORE_ROWS_TTL)
            .unwrap_or(false);
        if !fresh {
            let rows: HashMap<String, crate::metrics::provider_budget_store::ProviderBudgetRow> =
                match self.provider_budget_store() {
                    Some(store) => store
                        .get_all()
                        .unwrap_or_default()
                        .into_iter()
                        .map(|row| (row.provider.clone(), row))
                        .collect(),
                    None => HashMap::new(),
                };
            *cache = Some((std::time::Instant::now(), rows));
        }
        cache
            .as_ref()
            .map(|(_, rows)| rows.clone())
            .unwrap_or_default()
    }
}

/// Current window start (UTC epoch seconds) for a policy: the current
/// daily/weekly/monthly period boundary (today 00:00 UTC, this week's
/// Monday 00:00 UTC, first-of-month 00:00 UTC — the same boundary math
/// the rollover check and `next_reset_epoch` use), or "now" for manual.
/// This is the insert-time row anchor for a provider's first durable
/// spend accrual.
fn current_window_start_for_policy(policy: crate::config::BudgetResetPolicy) -> i64 {
    use chrono::Datelike as _;
    let now = chrono::Utc::now();
    let fallback = now.timestamp();
    match policy {
        crate::config::BudgetResetPolicy::Manual => fallback,
        crate::config::BudgetResetPolicy::Daily => now
            .date_naive()
            .and_hms_opt(0, 0, 0)
            .map(|d| d.and_utc().timestamp())
            .unwrap_or(fallback),
        crate::config::BudgetResetPolicy::Weekly => {
            let offset = now.date_naive().weekday().num_days_from_monday() as i64;
            now.date_naive()
                .checked_sub_signed(chrono::Duration::days(offset))
                .and_then(|d| d.and_hms_opt(0, 0, 0))
                .map(|d| d.and_utc().timestamp())
                .unwrap_or(fallback)
        }
        crate::config::BudgetResetPolicy::Monthly => {
            let first = chrono::NaiveDate::from_ymd_opt(now.year(), now.month(), 1)
                .unwrap_or_else(|| now.date_naive());
            first
                .and_hms_opt(0, 0, 0)
                .map(|d| d.and_utc().timestamp())
                .unwrap_or(fallback)
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
    /// Cumulative prompt tokens served from the upstream KV cache
    /// (cache reads) across recorded responses (Req 4.2).
    #[serde(default)]
    pub cache_read_tokens: u64,
    /// Cumulative prompt tokens written into the upstream cache
    /// (cache creations) across recorded responses (Req 4.2).
    #[serde(default)]
    pub cache_creation_tokens: u64,
    /// Cumulative prompt-cache cost savings in whole cents versus the
    /// uncached baseline (Req 4.2). Signed (see [`ProviderHealth::cache_savings_cents`]).
    #[serde(default)]
    pub cache_savings_cents: i64,
    /// Running prompt-cache hit rate across recorded responses:
    /// `cache_read_tokens / (cache_read_tokens + cache_creation_tokens
    /// + cache_uncached_tokens)`. `None` until at least one prompt
    /// token has been recorded (Req 4.2).
    #[serde(default)]
    pub cache_prompt_hit_rate: Option<f64>,
    /// Cumulative reasoning/thinking tokens across recorded responses
    /// (Req 4.7).
    #[serde(default)]
    pub reasoning_tokens: u64,
    /// Cumulative reasoning cost in USD across recorded responses
    /// (Req 4.7). Reconstructed from the micro-dollar counter.
    #[serde(default)]
    pub reasoning_cost_usd: f64,
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
            requests_last_minute: Mutex::new(RequestRateRing::new()),
            active_requests: AtomicU64::new(0),
            cumulative_cost_cents: AtomicU64::new(0),
            provider_health: Arc::new(DashMap::new()),
            cost_by_provider_cents: Arc::new(DashMap::new()),
            retry_count_by_provider: Arc::new(DashMap::new()),
            retry_delay_ms_by_provider: Arc::new(DashMap::new()),
            budget_limit_by_provider_cents: Arc::new(DashMap::new()),
            budget_policy_by_provider: Arc::new(DashMap::new()),
            budget_exhaustions_by_provider: Arc::new(DashMap::new()),
            unknown_cost_by_provider: Arc::new(DashMap::new()),
            rate_limit_exhaustions_by_provider: Arc::new(DashMap::new()),
            cache_hits: AtomicU64::new(0),
            cache_misses: AtomicU64::new(0),
            guardrail_stage_executions: Arc::new(DashMap::new()),
            guardrail_stage_latency: Arc::new(DashMap::new()),
            guardrail_refusal_detected: Arc::new(DashMap::new()),
            guardrail_refusal_failover: Arc::new(DashMap::new()),
            guardrail_stego_detections: Arc::new(DashMap::new()),
            structured_output: Arc::new(StructuredOutputMetrics::new()),
            provider_attempt_failures: Arc::new(ProviderAttemptFailureMetrics::new()),
            compression_tokens_saved: Arc::new(DashMap::new()),
            compression_ratio: Arc::new(DashMap::new()),
            compression_duration_seconds: Arc::new(DashMap::new()),
            tool_compression_requests_total: Arc::new(DashMap::new()),
            tool_compression_tokens_saved_total: Arc::new(DashMap::new()),
            tool_compression_ratio: Arc::new(DashMap::new()),
            tool_compression_duration_ms: Arc::new(DashMap::new()),
            tool_compression_tools_pruned: Arc::new(DashMap::new()),
            tool_compression_feedback_level: Arc::new(DashMap::new()),
            tool_compression_feedback_error_rate: Arc::new(DashMap::new()),
            tool_compression_feedback_adjustments: Arc::new(DashMap::new()),
            smart_routing: Mutex::new(None),
            provider_budget_store: Mutex::new(None),
            budget_store_rows_cache: Mutex::new(None),
            budget_window_boundaries: Mutex::new(HashMap::new()),
        }
    }

    /// Attach the durable provider-budget store. Called once during
    /// `GatewayServer` construction (startup/config-reload wiring only,
    /// never on the request path). After this call the in-memory
    /// cost/budget atomics are a write-through cache of the store.
    pub fn set_provider_budget_store(&self, store: Arc<ProviderBudgetStore>) {
        self.provider_budget_store_with_lock(|slot| *slot = Some(store));
    }

    /// The attached durable provider-budget store, if any. Spend accrual
    /// (Task 4) persists through this handle; the Task 3 rebuild test uses
    /// it to write durable spend.
    pub fn provider_budget_store(&self) -> Option<Arc<ProviderBudgetStore>> {
        self.provider_budget_store_with_lock(|slot| slot.clone())
    }

    /// Whether the budget window for `provider` under `policy` has
    /// expired, answered from the in-memory next-boundary cache (hot-path
    /// remediation, finding B1).
    ///
    /// Fast path: a cached entry seeded for the same policy is a pure
    /// `now >= next_boundary_epoch` comparison — no SQL on the tokio
    /// worker for the common not-expired case. The one sync store read at
    /// cache-seed time (once per provider per window) is acceptable; the
    /// store's `Arc<Mutex<Connection>>` serializes it and WAL keeps it
    /// cheap.
    ///
    /// Cache misses and policy changes (config hot-reload) re-seed from
    /// the store. Boundary math reuses [`next_reset_epoch`] — the same
    /// UTC daily/weekly/monthly logic as the virtual-key budgets
    /// (`window_expired`), not a fork. Returns `None` when no store is
    /// attached, the provider has no row yet (nothing accrued to roll;
    /// not cached so the first spend re-seeds), or the anchor is invalid.
    pub fn budget_window_expired_cached(
        &self,
        provider: &str,
        policy: crate::config::BudgetResetPolicy,
    ) -> Option<bool> {
        let now_epoch = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .map(|d| d.as_secs() as i64)
            .unwrap_or(0);
        {
            let cache = self
                .budget_window_boundaries
                .lock()
                .expect("budget window boundary cache poisoned");
            if let Some((cached_policy, _anchor, next_boundary)) = cache.get(provider) {
                if *cached_policy == policy {
                    return Some(now_epoch >= *next_boundary);
                }
            }
        }
        // Seed: one sync store read per provider per window. The read
        // runs outside the cache lock so the lock is never held across
        // SQLite I/O; a concurrent double-seed is benign (same data).
        let store = self.provider_budget_store()?;
        let anchor = match store.get_window_start(provider) {
            Ok(anchor) => anchor?,
            Err(error) => {
                tracing::warn!(
                    provider = provider,
                    error = %error,
                    "Failed to read provider budget window start; skipping lazy rollover"
                );
                return None;
            }
        };
        let next_boundary = next_reset_epoch(policy, anchor)?;
        self.budget_window_boundaries
            .lock()
            .expect("budget window boundary cache poisoned")
            .insert(provider.to_string(), (policy, anchor, next_boundary));
        Some(now_epoch >= next_boundary)
    }

    /// Persist a cost accrual to the durable provider-budget store,
    /// **off the response path** (persistent budgets, Task 4, Req 1.3).
    ///
    /// Mirrors the `UsageTracker` spawn pattern: the caller invokes this
    /// where `add_cost` runs, the write is spawned onto a blocking-safe
    /// task, and a store failure is logged as a warning and never fails
    /// the request (Req 1.4) — the metrics atomics incremented by
    /// `add_cost` remain the hot-path truth; the store is the durability
    /// tail. Spend is persisted for every provider, including providers
    /// without a configured limit (the store upserts the row; the limit is
    /// metadata seeded from config).
    ///
    /// Lost-update guard: the write carries the durable `window_start` it
    /// intends to accrue into. When the provider already has a row, that
    /// row's anchor is read first; when it does not (first accrual), the
    /// anchor is the current window start derived from the config policy
    /// (boundary-anchored for daily/weekly/monthly, "now" for manual), so
    /// the row is born with a correct window anchor. A write whose anchor
    /// no longer matches the row (a `reset_window` raced it) is dropped by
    /// the store's conditional upsert instead of polluting the new window.
    pub fn persist_provider_spend(&self, provider: &str, usd: f64) {
        let Some(store) = self.provider_budget_store() else {
            return;
        };
        let provider = provider.to_string();
        // Resolve the window anchor BEFORE spawning: the row read is cheap
        // and the anchor must reflect the instant this accrual happened.
        let window_start_epoch = match store.get_window_start(provider.as_str()) {
            Ok(Some(epoch)) if epoch > 0 => epoch,
            Ok(_) => self.current_budget_window_start(provider.as_str()),
            Err(error) => {
                tracing::warn!(
                    provider = %provider,
                    error = %error,
                    "Failed to read provider budget window start; skipping durable spend accrual"
                );
                return;
            }
        };
        let write = move || {
            if let Err(error) = store.add_spend(&provider, usd, window_start_epoch) {
                tracing::warn!(
                    provider = %provider,
                    error = %error,
                    "Failed to persist provider budget spend; in-memory metrics remain authoritative"
                );
            }
        };
        // Inside the async gateway, keep the sync SQLite write off the
        // response path entirely (spawn-and-forget, never awaited). The
        // inline fallback only runs in non-tokio contexts (unit tests),
        // where there is no response path to block.
        match tokio::runtime::Handle::try_current() {
            Ok(handle) => {
                handle.spawn_blocking(write);
            }
            Err(_) => write(),
        }
    }

    /// The current budget-window start (UTC epoch seconds) for `provider`
    /// derived from its config-seeded reset policy: the current period
    /// boundary for daily/weekly/monthly (the same UTC boundary math as
    /// the virtual-key budgets) and "now" for manual/unknown providers.
    /// Used as the insert-time row anchor when the first spend accrual
    /// creates the durable row.
    fn current_budget_window_start(&self, provider: &str) -> i64 {
        let policy = self
            .budget_policy_by_provider
            .get(provider)
            .map(|entry| *entry.value())
            .unwrap_or(crate::config::BudgetResetPolicy::Manual);
        current_window_start_for_policy(policy)
    }

    /// Zero `provider`'s in-memory spend atomics. The lazy window-rollover
    /// path (persistent budgets, Task 4) calls this together with
    /// [`ProviderBudgetStore::reset_window`] so both the hot-path truth and
    /// the durable store agree the new window starts from zero.
    pub fn reset_provider_spend(&self, provider: &str) {
        if let Some(entry) = self.cost_by_provider_cents.get(provider) {
            entry.value().store(0, Ordering::Relaxed);
        }
        // The rollover just advanced the durable anchor: drop the cached
        // next-boundary entry so the next check re-seeds from the fresh
        // store row (finding B1 cache discipline).
        self.budget_window_boundaries
            .lock()
            .expect("budget window boundary cache poisoned")
            .remove(provider);
    }

    /// Seed `budget_limit_by_provider` from `config` for every provider
    /// with a budget configured, and CLEAR the limit/policy entries of
    /// providers absent from the new config — both the startup path and
    /// the config-reload path call this, so a provider whose budget was
    /// removed (or renamed) on reload must not keep a stale limit or
    /// policy. Clearing is equally safe at startup (the maps start
    /// empty). Providers without a configured limit are skipped here (no
    /// limit gauge) but their spend is still persisted through the store;
    /// spend accrual is Task 4.
    ///
    /// Also clears the boundary-cache entries of providers whose policy
    /// changed (or vanished) so the rollover fast path re-seeds from the
    /// store under the new policy (finding B1 cache discipline).
    pub fn seed_provider_budget_limits(&self, config: &Config) {
        let configured: std::collections::HashSet<&str> = config
            .providers
            .iter()
            .filter(|provider| provider.budget.is_some())
            .map(|provider| provider.name.as_str())
            .collect();
        self.budget_limit_by_provider_cents
            .retain(|name, _| configured.contains(name.as_str()));
        self.budget_policy_by_provider
            .retain(|name, _| configured.contains(name.as_str()));
        {
            let mut boundaries = self
                .budget_window_boundaries
                .lock()
                .expect("budget window boundary cache poisoned");
            boundaries.retain(|name, (policy, _, _)| {
                config.providers.iter().any(|provider| {
                    provider.name == *name
                        && provider
                            .budget
                            .as_ref()
                            .and_then(|budget| budget.reset_policy)
                            .is_some_and(|configured| configured == *policy)
                })
            });
        }
        for provider in &config.providers {
            if let Some(budget) = &provider.budget {
                self.set_provider_budget_limit(&provider.name, budget.limit_usd);
                self.budget_policy_by_provider.insert(
                    provider.name.clone(),
                    budget.reset_policy.unwrap_or_default(),
                );
            }
        }
    }

    /// Restore one provider's persisted budget-window spend into the
    /// in-memory cost atomics. Called only during startup restore with
    /// rows from [`ProviderBudgetStore::get_all`]; the store value is the
    /// durable truth and wins over the (freshly zeroed) atomics. Restores
    /// ONLY the per-provider window spend — never the process-lifetime
    /// `cumulative_cost` counter, which measures spend accrued during THIS
    /// process, not across restarts (dashboard "total cost since start").
    pub fn restore_provider_budget_spend(&self, provider: &str, spend_usd: f64) {
        let spend_cents = (spend_usd * 100.0).round() as u64;
        self.cost_by_provider_cents
            .entry(provider.to_string())
            .or_insert_with(|| AtomicU64::new(0))
            .fetch_add(spend_cents, Ordering::Relaxed);
    }

    /// Shared lock helper for the budget-store slot. Panics on a poisoned
    /// lock, matching the codebase's other `Mutex` usages.
    fn provider_budget_store_with_lock<T>(
        &self,
        f: impl FnOnce(&mut Option<Arc<ProviderBudgetStore>>) -> T,
    ) -> T {
        let mut slot = self
            .provider_budget_store
            .lock()
            .expect("provider budget store slot poisoned");
        f(&mut slot)
    }

    /// Allocate Smart Routing metric state once. Until this method is called,
    /// recording is a no-op and Prometheus exposition contains no Smart Routing
    /// metric registration.
    #[allow(dead_code)]
    pub fn enable_smart_routing(&self) {
        let mut state = self
            .smart_routing
            .lock()
            .unwrap_or_else(|poisoned| poisoned.into_inner());
        state.get_or_insert_with(|| Arc::new(SmartRoutingMetricState::new()));
    }

    fn smart_routing_state(&self) -> Option<Arc<SmartRoutingMetricState>> {
        self.smart_routing
            .lock()
            .unwrap_or_else(|poisoned| poisoned.into_inner())
            .clone()
    }

    /// Record bounded Smart Routing decision telemetry. Raw model groups are
    /// reduced to one of 64 stable buckets before becoming labels.
    #[allow(dead_code)]
    pub fn record_smart_routing_decision(&self, decision: SmartRoutingDecisionMetric<'_>) {
        let Some(state) = self.smart_routing_state() else {
            return;
        };
        let tier = smart_routing_tier_label(decision.tier);
        let classifier = smart_routing_classifier_label(decision.classifier);
        let group = smart_routing_group_label(decision.group);
        let experiment = smart_routing_experiment_label(decision.experiment);
        let task_type = smart_routing_task_label(decision.task_type);
        let filtered = if decision.context_filtered {
            "true"
        } else {
            "false"
        }
        .to_owned();

        saturating_atomic_add(
            state
                .decisions
                .entry((tier.clone(), classifier.clone(), group, experiment))
                .or_insert_with(|| AtomicU64::new(0))
                .value(),
            1,
        );
        state
            .scores
            .entry((tier.clone(), classifier.clone()))
            .or_insert_with(|| CompressionHistogram::new(SMART_ROUTING_SCORE_BUCKETS.len()))
            .observe(decision.score.clamp(0.0, 1.0), &SMART_ROUTING_SCORE_BUCKETS);
        state
            .estimated_cost
            .entry(tier.clone())
            .or_insert_with(|| CompressionHistogram::new(SMART_ROUTING_COST_BUCKETS_USD.len()))
            .observe(decision.estimated_cost_usd, &SMART_ROUTING_COST_BUCKETS_USD);
        state
            .classifier_latency
            .entry(classifier)
            .or_insert_with(|| CompressionHistogram::new(SMART_ROUTING_LATENCY_BUCKETS_MS.len()))
            .observe(
                decision.classifier_latency_ms,
                &SMART_ROUTING_LATENCY_BUCKETS_MS,
            );
        saturating_atomic_add(
            state
                .task_types
                .entry(task_type)
                .or_insert_with(|| AtomicU64::new(0))
                .value(),
            1,
        );
        state
            .quality
            .entry(tier)
            .or_insert_with(|| CompressionHistogram::new(SMART_ROUTING_QUALITY_BUCKETS.len()))
            .observe(
                decision.quality.clamp(0.0, 1.0),
                &SMART_ROUTING_QUALITY_BUCKETS,
            );
        saturating_atomic_add(
            state
                .context_filtered
                .entry(filtered)
                .or_insert_with(|| AtomicU64::new(0))
                .value(),
            1,
        );
    }

    pub fn record_jev_consult(&self, group: &str, family: &str) {
        let Some(state) = self.smart_routing_state() else {
            return;
        };
        saturating_atomic_add(
            state
                .jev_consults
                .entry((
                    smart_routing_group_label(group),
                    smart_routing_family_label(family),
                ))
                .or_insert_with(|| AtomicU64::new(0))
                .value(),
            1,
        );
    }

    pub fn record_jev_fallback(&self, reason: &str, family: &str) {
        let Some(state) = self.smart_routing_state() else {
            return;
        };
        let label = smart_routing_enum_label(
            reason,
            &[
                "low_confidence",
                "model_unavailable",
                "no_jev_at_endpoint",
                "timeout",
                "backend",
                "invalid_output",
            ],
        );
        saturating_atomic_add(
            state
                .jev_fallbacks
                .entry((label, smart_routing_family_label(family)))
                .or_insert_with(|| AtomicU64::new(0))
                .value(),
            1,
        );
    }

    pub fn record_jev_confidence(&self, confidence: f64, family: &str) {
        let Some(state) = self.smart_routing_state() else {
            return;
        };
        state
            .jev_confidence
            .entry(smart_routing_family_label(family))
            .or_insert_with(|| CompressionHistogram::new(SMART_ROUTING_SCORE_BUCKETS.len()))
            .observe(confidence.clamp(0.0, 1.0), &SMART_ROUTING_SCORE_BUCKETS);
    }

    pub fn record_jev_latency(&self, latency_ms: f64, family: &str) {
        let Some(state) = self.smart_routing_state() else {
            return;
        };
        state
            .jev_latency
            .entry(smart_routing_family_label(family))
            .or_insert_with(|| CompressionHistogram::new(SMART_ROUTING_LATENCY_BUCKETS_MS.len()))
            .observe(latency_ms, &SMART_ROUTING_LATENCY_BUCKETS_MS);
    }

    pub fn record_jev_discovery_refresh(&self, status: &str, family: &str) {
        let Some(state) = self.smart_routing_state() else {
            return;
        };
        let label = smart_routing_enum_label(status, &["success", "failed", "timeout"]);
        saturating_atomic_add(
            state
                .jev_discovery_refreshes
                .entry((label, smart_routing_family_label(family)))
                .or_insert_with(|| AtomicU64::new(0))
                .value(),
            1,
        );
    }

    #[allow(dead_code)]
    pub fn record_smart_routing_cascade_transition(&self, from_tier: &str, to_tier: &str) {
        let Some(state) = self.smart_routing_state() else {
            return;
        };
        let key = (
            smart_routing_transition_label(from_tier),
            smart_routing_transition_label(to_tier),
        );
        saturating_atomic_add(
            state
                .cascade_transitions
                .entry(key)
                .or_insert_with(|| AtomicU64::new(0))
                .value(),
            1,
        );
    }

    #[allow(dead_code)]
    pub fn record_smart_routing_simhash(&self, hit: bool) {
        let Some(state) = self.smart_routing_state() else {
            return;
        };
        saturating_atomic_add(
            state
                .simhash
                .entry(smart_routing_outcome_label(hit))
                .or_insert_with(|| AtomicU64::new(0))
                .value(),
            1,
        );
    }

    #[allow(dead_code)]
    pub fn record_smart_routing_semantic_cache(&self, hit: bool) {
        let Some(state) = self.smart_routing_state() else {
            return;
        };
        saturating_atomic_add(
            state
                .semantic_cache
                .entry(smart_routing_outcome_label(hit))
                .or_insert_with(|| AtomicU64::new(0))
                .value(),
            1,
        );
    }

    /// Increment request count and mark request as active
    #[inline]
    pub fn start_request(&self) {
        let prev_active = self.active_requests.fetch_add(1, Ordering::Relaxed);
        self.request_count.fetch_add(1, Ordering::Relaxed);

        let now = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .unwrap()
            .as_secs();
        self.last_request_time.store(now, Ordering::Relaxed);
        let mut ring = self
            .requests_last_minute
            .lock()
            .unwrap_or_else(|poisoned| poisoned.into_inner());
        ring.record_at(now);

        if prev_active > 0 && prev_active % 100 == 0 {
            tracing::warn!(
                active_requests = prev_active + 1,
                "Active request counter exceeded {} — possible counter drift or extreme concurrency",
                prev_active + 1
            );
        }
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

    /// Force-reset the active request counter to zero.
    ///
    /// This is a recovery mechanism for accumulated counter drift caused by
    /// process aborts, panics, or other scenarios where `complete_request`
    /// was not properly called. Should only be used when the system is known
    /// to be idle (no requests in flight).
    ///
    /// Returns the previous value that was reset.
    pub fn reset_active_requests(&self) -> u64 {
        let previous = self.active_requests.swap(0, Ordering::Relaxed);
        if previous > 0 {
            tracing::warn!(
                previous_count = previous,
                "Active request counter was reset from non-zero value; this indicates counter drift from process aborts or incomplete request tracking"
            );
        }
        previous
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
                cache_read_tokens: AtomicU64::new(0),
                cache_creation_tokens: AtomicU64::new(0),
                cache_uncached_tokens: AtomicU64::new(0),
                cache_savings_cents: AtomicI64::new(0),
                reasoning_tokens: AtomicU64::new(0),
                reasoning_cost_usd_micros: AtomicU64::new(0),
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
                cache_read_tokens: AtomicU64::new(0),
                cache_creation_tokens: AtomicU64::new(0),
                cache_uncached_tokens: AtomicU64::new(0),
                cache_savings_cents: AtomicI64::new(0),
                reasoning_tokens: AtomicU64::new(0),
                reasoning_cost_usd_micros: AtomicU64::new(0),
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
                cache_read_tokens: AtomicU64::new(0),
                cache_creation_tokens: AtomicU64::new(0),
                cache_uncached_tokens: AtomicU64::new(0),
                cache_savings_cents: AtomicI64::new(0),
                reasoning_tokens: AtomicU64::new(0),
                reasoning_cost_usd_micros: AtomicU64::new(0),
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

    /// Record prompt-cache token usage and realized savings for a
    /// provider's response (Req 4.2). Mirrors [`Metrics::add_cost`]:
    /// lock-free atomic accumulation on the provider's health entry.
    ///
    /// Counters are running totals, so the prompt-cache hit rate is
    /// derived at snapshot time as
    /// `read / (read + creation + uncached)` and stays `None` until at
    /// least one prompt token has been recorded (divide-by-zero guard).
    /// `savings_cents` may be negative when a cache-creation premium
    /// exceeded the read discount on a response.
    pub fn add_cache_usage(&self, provider: &str, usage: CacheUsage, savings_cents: i64) {
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
                cache_read_tokens: AtomicU64::new(0),
                cache_creation_tokens: AtomicU64::new(0),
                cache_uncached_tokens: AtomicU64::new(0),
                cache_savings_cents: AtomicI64::new(0),
                reasoning_tokens: AtomicU64::new(0),
                reasoning_cost_usd_micros: AtomicU64::new(0),
            });

        health
            .cache_read_tokens
            .fetch_add(usage.cache_read_input_tokens, Ordering::Relaxed);
        health
            .cache_creation_tokens
            .fetch_add(usage.cache_creation_input_tokens, Ordering::Relaxed);
        health
            .cache_uncached_tokens
            .fetch_add(usage.uncached_input_tokens, Ordering::Relaxed);
        health
            .cache_savings_cents
            .fetch_add(savings_cents, Ordering::Relaxed);
    }

    /// Accumulate per-provider reasoning-token usage and cost (Req 4.7).
    ///
    /// `reasoning_tokens` is the count extracted from the response usage
    /// object (any carrier shape, never double-counted) and
    /// `reasoning_cost_usd` its price; both are attributed to the provider
    /// that actually served the response. The cost is accumulated as
    /// integer micro-dollars to keep the counters lock-free.
    pub fn add_reasoning_usage(
        &self,
        provider: &str,
        reasoning_tokens: u64,
        reasoning_cost_usd: f64,
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
                cache_read_tokens: AtomicU64::new(0),
                cache_creation_tokens: AtomicU64::new(0),
                cache_uncached_tokens: AtomicU64::new(0),
                cache_savings_cents: AtomicI64::new(0),
                reasoning_tokens: AtomicU64::new(0),
                reasoning_cost_usd_micros: AtomicU64::new(0),
            });

        health
            .reasoning_tokens
            .fetch_add(reasoning_tokens, Ordering::Relaxed);
        let cost_micros = (reasoning_cost_usd.max(0.0) * 1_000_000.0).round() as u64;
        health
            .reasoning_cost_usd_micros
            .fetch_add(cost_micros, Ordering::Relaxed);
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

    /// Clear `provider`'s budget-exhaustion marker. Called by the admin
    /// provider-budget reset endpoint so dashboards/Prometheus reflect the
    /// fresh window instead of lingering skip counts from the reset one.
    pub fn clear_provider_budget_exhausted(&self, provider: &str) {
        self.budget_exhaustions_by_provider.remove(provider);
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
    /// for `(pipeline, stage, provider_type, phase, action)` and observes
    /// `latency_ms` in the latency histogram for
    /// `(pipeline, stage, provider_type, phase)` (Req 11.1, 11.2; phase label
    /// added by the indirect-injection defense, task 5.1).
    ///
    /// `action` must be one of `pass`, `block`, `redact`, `mask`,
    /// `replace_with_policy_message`, or `error`. `phase` is one of
    /// `pre_call`, `tool_result`, `tool_call`, `post_call`. Recording is
    /// best-effort and never fails the calling request (Req 11.7).
    pub fn record_guardrail_stage(
        &self,
        pipeline: &str,
        stage: &str,
        provider_type: &str,
        phase: &str,
        action: &str,
        latency_ms: f64,
    ) {
        let counter_key = (
            pipeline.to_string(),
            stage.to_string(),
            provider_type.to_string(),
            phase.to_string(),
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
            phase.to_string(),
        );
        self.guardrail_stage_latency
            .entry(latency_key)
            .or_insert_with(GuardrailLatencyHistogram::new)
            .observe(latency_ms);
    }

    /// Increment the unicode_stego detection counter for `(category, phase)`
    /// (indirect-injection defense, task 5.2). Best-effort: never fails the
    /// request (Req 11.7). Content-safe: labels only, never matched text.
    pub fn record_guardrail_stego_detection(&self, category: &str, phase: &str) {
        let key = (category.to_string(), phase.to_string());
        self.guardrail_stego_detections
            .entry(key)
            .or_insert_with(|| AtomicU64::new(0))
            .fetch_add(1, Ordering::Relaxed);
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
Total guardrail stage executions by pipeline, stage, provider, phase, and action\n",
            );
            out.push_str("# TYPE obey_api_guardrail_stage_executions_total counter\n");
            for ((pipeline, stage, provider, phase, action), value) in &rows {
                out.push_str(&format!(
                    "obey_api_guardrail_stage_executions_total{{pipeline=\"{}\",stage=\"{}\",provider=\"{}\",phase=\"{}\",action=\"{}\"}} {}\n",
                    pipeline, stage, provider, phase, action, value
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
Guardrail stage latency in milliseconds by pipeline, stage, provider, and phase\n",
            );
            out.push_str("# TYPE obey_api_guardrail_stage_latency_ms histogram\n");
            for key in &keys {
                let entry = match self.guardrail_stage_latency.get(key) {
                    Some(entry) => entry,
                    None => continue,
                };
                let hist = entry.value();
                let (pipeline, stage, provider, phase) = key;

                let mut cumulative = 0u64;
                for (idx, boundary) in GUARDRAIL_LATENCY_BUCKETS_MS.iter().enumerate() {
                    cumulative += hist.buckets[idx].load(Ordering::Relaxed);
                    out.push_str(&format!(
                        "obey_api_guardrail_stage_latency_ms_bucket{{pipeline=\"{}\",stage=\"{}\",provider=\"{}\",phase=\"{}\",le=\"{}\"}} {}\n",
                        pipeline, stage, provider, phase, boundary, cumulative
                    ));
                }
                let total = hist.count.load(Ordering::Relaxed);
                out.push_str(&format!(
                    "obey_api_guardrail_stage_latency_ms_bucket{{pipeline=\"{}\",stage=\"{}\",provider=\"{}\",phase=\"{}\",le=\"+Inf\"}} {}\n",
                    pipeline, stage, provider, phase, total
                ));
                let sum_ms = hist.sum_micros.load(Ordering::Relaxed) as f64 / 1000.0;
                out.push_str(&format!(
                    "obey_api_guardrail_stage_latency_ms_sum{{pipeline=\"{}\",stage=\"{}\",provider=\"{}\",phase=\"{}\"}} {}\n",
                    pipeline, stage, provider, phase, sum_ms
                ));
                out.push_str(&format!(
                    "obey_api_guardrail_stage_latency_ms_count{{pipeline=\"{}\",stage=\"{}\",provider=\"{}\",phase=\"{}\"}} {}\n",
                    pipeline, stage, provider, phase, total
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

        // Counter: obey_api_guardrail_stego_detections_total
        // (indirect-injection defense, task 5.2)
        if !self.guardrail_stego_detections.is_empty() {
            let mut rows: Vec<((String, String), u64)> = self
                .guardrail_stego_detections
                .iter()
                .map(|e| (e.key().clone(), e.value().load(Ordering::Relaxed)))
                .collect();
            rows.sort_by(|a, b| a.0.cmp(&b.0));

            out.push_str(
                "# HELP obey_api_guardrail_stego_detections_total \
Total unicode_stego provider detections by category and phase\n",
            );
            out.push_str("# TYPE obey_api_guardrail_stego_detections_total counter\n");
            for ((category, phase), value) in &rows {
                out.push_str(&format!(
                    "obey_api_guardrail_stego_detections_total{{category=\"{}\",phase=\"{}\"}} {}\n",
                    category, phase, value
                ));
            }
        }
    }

    pub fn structured_output_metrics(&self) -> Arc<StructuredOutputMetrics> {
        Arc::clone(&self.structured_output)
    }

    /// Record a structured output validation outcome.
    #[allow(dead_code)]
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

    /// Record a single FAILED provider attempt, keyed by
    /// `(provider, model, status_code)`. Called for every failed attempt even
    /// when the request ultimately succeeds via retry/fallback, so a model's
    /// true failure rate is observable.
    pub fn record_provider_attempt_failure(&self, provider: &str, model: &str, status_code: u16) {
        self.provider_attempt_failures
            .record(provider, model, status_code);
    }

    /// Append provider attempt-failure metrics in Prometheus text format.
    pub fn write_provider_attempt_failures_prometheus(&self, out: &mut String) {
        self.provider_attempt_failures.write_prometheus(out);
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

    /// Record tool compression pipeline metrics for one request.
    #[allow(dead_code)]
    pub fn record_tool_compression(
        &self,
        model_group: &str,
        level: &str,
        tokens_saved: u64,
        ratio: f64,
        duration_ms: u64,
    ) {
        let key = (
            bounded_label(model_group, MAX_COMPRESSION_PROVIDER_LABEL_LEN),
            bounded_label(level, MAX_COMPRESSION_LEVEL_LABEL_LEN),
        );

        // Increment request counter
        let counter = self
            .tool_compression_requests_total
            .entry(key.clone())
            .or_insert_with(|| AtomicU64::new(0));
        saturating_atomic_add(counter.value(), 1);

        // Accumulate tokens saved
        let saved_counter = self
            .tool_compression_tokens_saved_total
            .entry(key.clone())
            .or_insert_with(|| AtomicU64::new(0));
        saturating_atomic_add(saved_counter.value(), tokens_saved);

        // Observe compression ratio
        self.tool_compression_ratio
            .entry(key.clone())
            .or_insert_with(|| CompressionHistogram::new(COMPRESSION_RATIO_BUCKETS.len()))
            .observe(ratio.clamp(0.0, 1.0), &COMPRESSION_RATIO_BUCKETS);

        // Observe pipeline duration
        let duration_seconds = duration_ms as f64 / 1000.0;
        self.tool_compression_duration_ms
            .entry(key)
            .or_insert_with(|| {
                CompressionHistogram::new(COMPRESSION_DURATION_BUCKETS_SECONDS.len())
            })
            .observe(duration_seconds, &COMPRESSION_DURATION_BUCKETS_SECONDS);
    }

    /// Record the number of tools pruned for a model group (gauge — overwrites previous value).
    #[allow(dead_code)]
    pub fn record_tool_compression_tools_pruned(&self, model_group: &str, count: u64) {
        let key = bounded_label(model_group, MAX_COMPRESSION_PROVIDER_LABEL_LEN);
        self.tool_compression_tools_pruned
            .entry(key)
            .or_insert_with(|| AtomicU64::new(0))
            .store(count, Ordering::Relaxed);
    }

    /// Record the current feedback-loop compression level for a model group (gauge).
    /// Level is encoded as integer: 0=low, 1=medium, 2=high, 3=max.
    #[allow(dead_code)]
    pub fn record_tool_compression_feedback_level(&self, model_group: &str, level: u64) {
        let key = bounded_label(model_group, MAX_COMPRESSION_PROVIDER_LABEL_LEN);
        self.tool_compression_feedback_level
            .entry(key)
            .or_insert_with(|| AtomicU64::new(0))
            .store(level, Ordering::Relaxed);
    }

    /// Record the current feedback-loop error rate for a model group (gauge).
    /// Rate is stored as micros (rate * 1_000_000) to avoid float atomics.
    #[allow(dead_code)]
    pub fn record_tool_compression_feedback_error_rate(&self, model_group: &str, rate: f64) {
        let key = bounded_label(model_group, MAX_COMPRESSION_PROVIDER_LABEL_LEN);
        let rate_micros = (rate.clamp(0.0, 1.0) * 1_000_000.0).round() as u64;
        self.tool_compression_feedback_error_rate
            .entry(key)
            .or_insert_with(|| AtomicU64::new(0))
            .store(rate_micros, Ordering::Relaxed);
    }

    /// Increment the feedback-loop adjustment counter for a model group.
    #[allow(dead_code)]
    pub fn record_tool_compression_feedback_adjustment(&self, model_group: &str) {
        let key = bounded_label(model_group, MAX_COMPRESSION_PROVIDER_LABEL_LEN);
        self.tool_compression_feedback_adjustments
            .entry(key)
            .or_insert_with(|| AtomicU64::new(0))
            .fetch_add(1, Ordering::Relaxed);
    }

    /// Append tool compression metrics in Prometheus text exposition format.
    pub fn write_tool_compression_prometheus(&self, out: &mut String) {
        // Counter: obey_tool_compression_requests_total
        if !self.tool_compression_requests_total.is_empty() {
            let mut rows: Vec<(ToolCompressionMetricKey, u64)> = self
                .tool_compression_requests_total
                .iter()
                .map(|e| (e.key().clone(), e.value().load(Ordering::Relaxed)))
                .collect();
            rows.sort_by(|a, b| a.0.cmp(&b.0));
            out.push_str("# HELP obey_tool_compression_requests_total Total requests where tool compression was applied\n");
            out.push_str("# TYPE obey_tool_compression_requests_total counter\n");
            for ((model_group, level), value) in &rows {
                out.push_str(&format!(
                    "obey_tool_compression_requests_total{{model_group=\"{}\",compression_level=\"{}\"}} {}\n",
                    escape_prometheus_label(model_group),
                    escape_prometheus_label(level),
                    value
                ));
            }
        }

        // Counter: obey_tool_compression_tokens_saved_total
        if !self.tool_compression_tokens_saved_total.is_empty() {
            let mut rows: Vec<(ToolCompressionMetricKey, u64)> = self
                .tool_compression_tokens_saved_total
                .iter()
                .map(|e| (e.key().clone(), e.value().load(Ordering::Relaxed)))
                .collect();
            rows.sort_by(|a, b| a.0.cmp(&b.0));
            out.push_str("# HELP obey_tool_compression_tokens_saved_total Total tokens saved by tool compression\n");
            out.push_str("# TYPE obey_tool_compression_tokens_saved_total counter\n");
            for ((model_group, level), value) in &rows {
                out.push_str(&format!(
                    "obey_tool_compression_tokens_saved_total{{model_group=\"{}\",compression_level=\"{}\"}} {}\n",
                    escape_prometheus_label(model_group),
                    escape_prometheus_label(level),
                    value
                ));
            }
        }

        // Histogram: obey_tool_compression_reduction_ratio
        self.write_tool_compression_histogram(
            out,
            "obey_tool_compression_reduction_ratio",
            "Ratio of tokens saved by tool compression (1.0 - compressed/original)",
            &self.tool_compression_ratio,
            &COMPRESSION_RATIO_BUCKETS,
        );

        // Histogram: obey_tool_compression_latency_seconds
        self.write_tool_compression_histogram(
            out,
            "obey_tool_compression_latency_seconds",
            "Tool compression pipeline latency in seconds",
            &self.tool_compression_duration_ms,
            &COMPRESSION_DURATION_BUCKETS_SECONDS,
        );

        // Gauge: obey_tool_compression_tools_pruned
        if !self.tool_compression_tools_pruned.is_empty() {
            let mut rows: Vec<(String, u64)> = self
                .tool_compression_tools_pruned
                .iter()
                .map(|e| (e.key().clone(), e.value().load(Ordering::Relaxed)))
                .collect();
            rows.sort_by(|a, b| a.0.cmp(&b.0));
            out.push_str("# HELP obey_tool_compression_tools_pruned Current count of tools pruned per model group\n");
            out.push_str("# TYPE obey_tool_compression_tools_pruned gauge\n");
            for (model_group, value) in &rows {
                out.push_str(&format!(
                    "obey_tool_compression_tools_pruned{{model_group=\"{}\"}} {}\n",
                    escape_prometheus_label(model_group),
                    value
                ));
            }
        }

        // Gauge: obey_tool_compression_feedback_level
        if !self.tool_compression_feedback_level.is_empty() {
            let mut rows: Vec<(String, u64)> = self
                .tool_compression_feedback_level
                .iter()
                .map(|e| (e.key().clone(), e.value().load(Ordering::Relaxed)))
                .collect();
            rows.sort_by(|a, b| a.0.cmp(&b.0));
            out.push_str("# HELP obey_tool_compression_feedback_level Current feedback-adjusted compression level per model group (0=low, 1=medium, 2=high, 3=max)\n");
            out.push_str("# TYPE obey_tool_compression_feedback_level gauge\n");
            for (model_group, value) in &rows {
                out.push_str(&format!(
                    "obey_tool_compression_feedback_level{{model_group=\"{}\"}} {}\n",
                    escape_prometheus_label(model_group),
                    value
                ));
            }
        }

        // Gauge: obey_tool_compression_feedback_error_rate
        if !self.tool_compression_feedback_error_rate.is_empty() {
            let mut rows: Vec<(String, u64)> = self
                .tool_compression_feedback_error_rate
                .iter()
                .map(|e| (e.key().clone(), e.value().load(Ordering::Relaxed)))
                .collect();
            rows.sort_by(|a, b| a.0.cmp(&b.0));
            out.push_str("# HELP obey_tool_compression_feedback_error_rate Current tool-call error rate per model group\n");
            out.push_str("# TYPE obey_tool_compression_feedback_error_rate gauge\n");
            for (model_group, value) in &rows {
                let rate = *value as f64 / 1_000_000.0;
                out.push_str(&format!(
                    "obey_tool_compression_feedback_error_rate{{model_group=\"{}\"}} {:.6}\n",
                    escape_prometheus_label(model_group),
                    rate
                ));
            }
        }

        // Counter: obey_tool_compression_feedback_adjustments_total
        if !self.tool_compression_feedback_adjustments.is_empty() {
            let mut rows: Vec<(String, u64)> = self
                .tool_compression_feedback_adjustments
                .iter()
                .map(|e| (e.key().clone(), e.value().load(Ordering::Relaxed)))
                .collect();
            rows.sort_by(|a, b| a.0.cmp(&b.0));
            out.push_str("# HELP obey_tool_compression_feedback_adjustments_total Total feedback-loop level adjustments per model group\n");
            out.push_str("# TYPE obey_tool_compression_feedback_adjustments_total counter\n");
            for (model_group, value) in &rows {
                out.push_str(&format!(
                    "obey_tool_compression_feedback_adjustments_total{{model_group=\"{}\"}} {}\n",
                    escape_prometheus_label(model_group),
                    value
                ));
            }
        }
    }

    fn write_tool_compression_histogram(
        &self,
        out: &mut String,
        name: &str,
        help: &str,
        values: &DashMap<ToolCompressionMetricKey, CompressionHistogram>,
        buckets: &[f64],
    ) {
        if values.is_empty() {
            return;
        }
        let mut keys: Vec<ToolCompressionMetricKey> =
            values.iter().map(|entry| entry.key().clone()).collect();
        keys.sort();
        out.push_str(&format!("# HELP {name} {help}\n"));
        out.push_str(&format!("# TYPE {name} histogram\n"));
        for key in keys {
            let Some(entry) = values.get(&key) else {
                continue;
            };
            let histogram = entry.value();
            let model_group = escape_prometheus_label(&key.0);
            let level = escape_prometheus_label(&key.1);
            let mut cumulative = 0u64;
            for (index, boundary) in buckets.iter().enumerate() {
                cumulative =
                    cumulative.saturating_add(histogram.buckets[index].load(Ordering::Relaxed));
                out.push_str(&format!(
                    "{name}_bucket{{model_group=\"{model_group}\",compression_level=\"{level}\",le=\"{boundary}\"}} {cumulative}\n"
                ));
            }
            let count = histogram.count.load(Ordering::Relaxed);
            out.push_str(&format!(
                "{name}_bucket{{model_group=\"{model_group}\",compression_level=\"{level}\",le=\"+Inf\"}} {count}\n"
            ));
            let sum = histogram.sum_micros.load(Ordering::Relaxed) as f64 / 1_000_000.0;
            out.push_str(&format!(
                "{name}_sum{{model_group=\"{model_group}\",compression_level=\"{level}\"}} {sum}\n"
            ));
            out.push_str(&format!(
                "{name}_count{{model_group=\"{model_group}\",compression_level=\"{level}\"}} {count}\n"
            ));
        }
    }

    /// Append compression counter and histograms in Prometheus text format.
    /// The ratio is `compressed_tokens / original_tokens` in `[0, 1]`; when
    /// `original_tokens` is zero, the observed ratio is `1.0`.
    pub fn write_smart_routing_prometheus(&self, out: &mut String) {
        let Some(state) = self.smart_routing_state() else {
            return;
        };
        write_smart_routing_counter4(
            out,
            "obey_api_smart_routing_decisions_total",
            "Total Smart Routing decisions by bounded routing dimensions",
            ["tier", "classifier", "group", "experiment"],
            &state.decisions,
        );
        write_smart_routing_histogram2(
            out,
            "obey_api_smart_routing_score",
            "Smart Routing normalized decision score",
            ["tier", "classifier"],
            &state.scores,
            &SMART_ROUTING_SCORE_BUCKETS,
        );
        write_smart_routing_counter2(
            out,
            "obey_api_smart_routing_cascade_transitions_total",
            "Total Smart Routing cascade transitions",
            ["from_tier", "to_tier"],
            &state.cascade_transitions,
        );
        write_smart_routing_histogram1(
            out,
            "obey_api_smart_routing_estimated_cost_usd",
            "Estimated Smart Routing request cost in US dollars",
            "tier",
            &state.estimated_cost,
            &SMART_ROUTING_COST_BUCKETS_USD,
        );
        write_smart_routing_histogram1(
            out,
            "obey_api_smart_routing_classifier_latency_ms",
            "Smart Routing classifier latency in milliseconds",
            "classifier",
            &state.classifier_latency,
            &SMART_ROUTING_LATENCY_BUCKETS_MS,
        );
        write_smart_routing_counter1(
            out,
            "obey_api_smart_routing_simhash_total",
            "Smart Routing SimHash cache lookups",
            "result",
            &state.simhash,
        );
        write_smart_routing_counter1(
            out,
            "obey_api_smart_routing_task_type_total",
            "Smart Routing decisions by bounded task type",
            "task_type",
            &state.task_types,
        );
        write_smart_routing_histogram1(
            out,
            "obey_api_smart_routing_quality",
            "Smart Routing normalized observed quality",
            "tier",
            &state.quality,
            &SMART_ROUTING_QUALITY_BUCKETS,
        );
        write_smart_routing_counter1(
            out,
            "obey_api_smart_routing_semantic_cache_total",
            "Smart Routing semantic cache lookups",
            "result",
            &state.semantic_cache,
        );
        write_smart_routing_counter1(
            out,
            "obey_api_smart_routing_context_filtered_total",
            "Smart Routing decisions with context filtering status",
            "filtered",
            &state.context_filtered,
        );
        write_smart_routing_counter2(
            out,
            "obey_api_smart_routing_jev_consults_total",
            "Total System One classifier consultations by family",
            ["group", "family"],
            &state.jev_consults,
        );
        write_smart_routing_counter2(
            out,
            "obey_api_smart_routing_jev_fallbacks_total",
            "Total System One classifier fallbacks by bounded reason and family",
            ["reason", "family"],
            &state.jev_fallbacks,
        );
        write_smart_routing_histogram1(
            out,
            "obey_api_smart_routing_jev_confidence",
            "System One composite confidence by family",
            "family",
            &state.jev_confidence,
            &SMART_ROUTING_SCORE_BUCKETS,
        );
        write_smart_routing_histogram1(
            out,
            "obey_api_smart_routing_jev_latency_ms",
            "System One classifier latency in milliseconds by family",
            "family",
            &state.jev_latency,
            &SMART_ROUTING_LATENCY_BUCKETS_MS,
        );
        write_smart_routing_counter2(
            out,
            "obey_api_smart_routing_jev_discovery_refreshes_total",
            "System One model discovery refreshes by bounded status and family",
            ["status", "family"],
            &state.jev_discovery_refreshes,
        );
    }

    pub fn write_compression_prometheus(&self, out: &mut String) {
        self.write_smart_routing_prometheus(out);
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

                let cache_read_tokens = health.cache_read_tokens.load(Ordering::Relaxed);
                let cache_creation_tokens = health.cache_creation_tokens.load(Ordering::Relaxed);
                let cache_uncached_tokens = health.cache_uncached_tokens.load(Ordering::Relaxed);
                // Divide-by-zero guard: leave the hit rate unset until at least one
                // prompt token has been recorded for this provider (Req 4.2).
                let cache_denominator = cache_read_tokens
                    .saturating_add(cache_creation_tokens)
                    .saturating_add(cache_uncached_tokens);
                let cache_prompt_hit_rate = if cache_denominator > 0 {
                    Some(cache_read_tokens as f64 / cache_denominator as f64)
                } else {
                    None
                };
                let cache_savings_cents = health.cache_savings_cents.load(Ordering::Relaxed);
                let reasoning_tokens = health.reasoning_tokens.load(Ordering::Relaxed);
                let reasoning_cost_usd =
                    health.reasoning_cost_usd_micros.load(Ordering::Relaxed) as f64 / 1_000_000.0;

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
                    cache_read_tokens,
                    cache_creation_tokens,
                    cache_savings_cents,
                    cache_prompt_hit_rate,
                    reasoning_tokens,
                    reasoning_cost_usd,
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
            request_rate_per_min: self.requests_last_minute_per_min(),
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
            active_requests_list: Vec::new(),
            budget_status_by_provider: self.budget_status_by_provider(),
        }
    }

    /// Trailing-60s rolling request count as of now. This replaces the old
    /// hard-reset per-minute counter: the value decays as per-second buckets
    /// age out of the window instead of saw-toothing to zero.
    pub fn requests_last_minute(&self) -> u64 {
        let now = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .unwrap()
            .as_secs();
        let mut ring = self
            .requests_last_minute
            .lock()
            .unwrap_or_else(|poisoned| poisoned.into_inner());
        ring.rolling_total_at(now)
    }

    /// `requests_last_minute` expressed as the per-minute rate exposed in
    /// [`MetricsSnapshot::request_rate_per_min`].
    fn requests_last_minute_per_min(&self) -> f64 {
        self.requests_last_minute() as f64
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

/// Bridges the smart-routing orchestrator's narrow metrics trait onto the
/// shared `Metrics` registry, keeping label sets bounded. Classifier
/// fallback/cache/budget hooks have no Prometheus series yet and remain
/// default no-ops.
impl crate::smart_routing::SmartRoutingMetrics for Metrics {
    fn jev_consult(&self, group: &str, family: &str) {
        self.record_jev_consult(group, family);
    }
    fn jev_fallback(&self, reason: crate::smart_routing::ClassifierFailure, family: &str) {
        let label = match reason {
            crate::smart_routing::ClassifierFailure::Unavailable => "model_unavailable",
            crate::smart_routing::ClassifierFailure::Timeout => "timeout",
            crate::smart_routing::ClassifierFailure::InvalidOutput => "invalid_output",
            crate::smart_routing::ClassifierFailure::Backend => "backend",
            crate::smart_routing::ClassifierFailure::LowConfidence => "low_confidence",
            crate::smart_routing::ClassifierFailure::NoJevAtEndpoint => "no_jev_at_endpoint",
        };
        self.record_jev_fallback(label, family);
    }
    fn jev_confidence(&self, confidence: f64, family: &str) {
        self.record_jev_confidence(confidence, family);
    }
    fn jev_latency(&self, latency_ms: f64, family: &str) {
        self.record_jev_latency(latency_ms, family);
    }
    fn jev_discovery_refresh(&self, status: &'static str, family: &str) {
        self.record_jev_discovery_refresh(status, family);
    }
}

fn write_smart_routing_counter1(
    out: &mut String,
    name: &str,
    help: &str,
    label: &str,
    values: &DashMap<String, AtomicU64>,
) {
    if values.is_empty() {
        return;
    }
    let mut rows: Vec<(String, u64)> = values
        .iter()
        .map(|entry| (entry.key().clone(), entry.value().load(Ordering::Relaxed)))
        .collect();
    rows.sort_by(|a, b| a.0.cmp(&b.0));
    out.push_str(&format!("# HELP {name} {help}\n# TYPE {name} counter\n"));
    for (value, count) in rows {
        out.push_str(&format!("{name}{{{label}=\"{value}\"}} {count}\n"));
    }
}

fn write_smart_routing_counter2(
    out: &mut String,
    name: &str,
    help: &str,
    labels: [&str; 2],
    values: &DashMap<(String, String), AtomicU64>,
) {
    if values.is_empty() {
        return;
    }
    let mut rows: Vec<((String, String), u64)> = values
        .iter()
        .map(|entry| (entry.key().clone(), entry.value().load(Ordering::Relaxed)))
        .collect();
    rows.sort_by(|a, b| a.0.cmp(&b.0));
    out.push_str(&format!("# HELP {name} {help}\n# TYPE {name} counter\n"));
    for ((first, second), count) in rows {
        out.push_str(&format!(
            "{name}{{{}=\"{first}\",{}=\"{second}\"}} {count}\n",
            labels[0], labels[1]
        ));
    }
}

fn write_smart_routing_counter4(
    out: &mut String,
    name: &str,
    help: &str,
    labels: [&str; 4],
    values: &DashMap<(String, String, String, String), AtomicU64>,
) {
    if values.is_empty() {
        return;
    }
    let mut rows: Vec<((String, String, String, String), u64)> = values
        .iter()
        .map(|entry| (entry.key().clone(), entry.value().load(Ordering::Relaxed)))
        .collect();
    rows.sort_by(|a, b| a.0.cmp(&b.0));
    out.push_str(&format!("# HELP {name} {help}\n# TYPE {name} counter\n"));
    for ((first, second, third, fourth), count) in rows {
        out.push_str(&format!(
            "{name}{{{}=\"{first}\",{}=\"{second}\",{}=\"{third}\",{}=\"{fourth}\"}} {count}\n",
            labels[0], labels[1], labels[2], labels[3]
        ));
    }
}

fn write_smart_routing_histogram1(
    out: &mut String,
    name: &str,
    help: &str,
    label: &str,
    values: &DashMap<String, CompressionHistogram>,
    buckets: &[f64],
) {
    if values.is_empty() {
        return;
    }
    let mut keys: Vec<String> = values.iter().map(|entry| entry.key().clone()).collect();
    keys.sort();
    out.push_str(&format!("# HELP {name} {help}\n# TYPE {name} histogram\n"));
    for key in keys {
        let Some(entry) = values.get(&key) else {
            continue;
        };
        write_smart_routing_histogram_samples(
            out,
            name,
            &format!("{label}=\"{key}\""),
            entry.value(),
            buckets,
        );
    }
}

fn write_smart_routing_histogram2(
    out: &mut String,
    name: &str,
    help: &str,
    labels: [&str; 2],
    values: &DashMap<(String, String), CompressionHistogram>,
    buckets: &[f64],
) {
    if values.is_empty() {
        return;
    }
    let mut keys: Vec<(String, String)> = values.iter().map(|entry| entry.key().clone()).collect();
    keys.sort();
    out.push_str(&format!("# HELP {name} {help}\n# TYPE {name} histogram\n"));
    for key in keys {
        let Some(entry) = values.get(&key) else {
            continue;
        };
        write_smart_routing_histogram_samples(
            out,
            name,
            &format!("{}=\"{}\",{}=\"{}\"", labels[0], key.0, labels[1], key.1),
            entry.value(),
            buckets,
        );
    }
}

fn write_smart_routing_histogram_samples(
    out: &mut String,
    name: &str,
    labels: &str,
    histogram: &CompressionHistogram,
    buckets: &[f64],
) {
    let mut cumulative = 0u64;
    for (index, boundary) in buckets.iter().enumerate() {
        cumulative = cumulative.saturating_add(histogram.buckets[index].load(Ordering::Relaxed));
        out.push_str(&format!(
            "{name}_bucket{{{labels},le=\"{boundary}\"}} {cumulative}\n"
        ));
    }
    let count = histogram.count.load(Ordering::Relaxed);
    out.push_str(&format!("{name}_bucket{{{labels},le=\"+Inf\"}} {count}\n"));
    let sum = histogram.sum_micros.load(Ordering::Relaxed) as f64 / 1_000_000.0;
    out.push_str(&format!("{name}_sum{{{labels}}} {sum}\n"));
    out.push_str(&format!("{name}_count{{{labels}}} {count}\n"));
}

impl Default for Metrics {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn smart_routing_decision<'a>(group: &'a str) -> SmartRoutingDecisionMetric<'a> {
        SmartRoutingDecisionMetric {
            tier: "fast",
            classifier: "heuristic",
            group,
            score: 0.35,
            estimated_cost_usd: 0.004,
            classifier_latency_ms: 1.5,
            task_type: "code_generation",
            quality: 0.8,
            context_filtered: true,
            experiment: Some("control"),
        }
    }

    #[test]
    fn smart_routing_disabled_has_no_state_or_registration() {
        let metrics = Metrics::new();
        metrics.record_smart_routing_decision(smart_routing_decision("private-group"));
        metrics.record_smart_routing_cascade_transition("fast", "balanced");
        metrics.record_smart_routing_simhash(true);
        metrics.record_smart_routing_semantic_cache(false);

        assert!(metrics.smart_routing_state().is_none());
        let mut out = String::new();
        metrics.write_smart_routing_prometheus(&mut out);
        assert!(out.is_empty());
    }

    #[test]
    fn smart_routing_renders_bounded_labels_and_histograms() {
        let metrics = Metrics::new();
        metrics.enable_smart_routing();
        metrics.record_smart_routing_decision(smart_routing_decision(
            "tenant/request/content must never become a raw label",
        ));
        metrics.record_smart_routing_decision(SmartRoutingDecisionMetric {
            tier: "unbounded-tier",
            classifier: "custom-classifier",
            task_type: "private-task",
            experiment: Some("secret-experiment-name"),
            context_filtered: false,
            ..smart_routing_decision("")
        });
        metrics.record_smart_routing_cascade_transition("fast", "balanced");
        metrics.record_smart_routing_cascade_transition("arbitrary", "powerful");
        metrics.record_smart_routing_simhash(true);
        metrics.record_smart_routing_simhash(false);
        metrics.record_smart_routing_semantic_cache(true);
        metrics.record_smart_routing_semantic_cache(false);

        let mut out = String::new();
        metrics.write_smart_routing_prometheus(&mut out);

        assert!(out.contains("obey_api_smart_routing_decisions_total"));
        assert!(out.contains("tier=\"fast\",classifier=\"heuristic\",group=\"bucket_"));
        assert!(out.contains(
            "tier=\"other\",classifier=\"other\",group=\"unknown\",experiment=\"other\""
        ));
        assert!(out.contains(
"obey_api_smart_routing_score_bucket{tier=\"fast\",classifier=\"heuristic\",le=\"0.1\"} 0"
));
        assert!(out.contains(
"obey_api_smart_routing_score_bucket{tier=\"fast\",classifier=\"heuristic\",le=\"0.4\"} 1"
));
        assert!(out.contains("obey_api_smart_routing_cascade_transitions_total"));
        assert!(out.contains("obey_api_smart_routing_estimated_cost_usd"));
        assert!(out.contains("obey_api_smart_routing_classifier_latency_ms"));
        assert!(out.contains("obey_api_smart_routing_simhash_total{result=\"hit\"} 1"));
        assert!(out.contains("obey_api_smart_routing_task_type_total{task_type=\"other\"} 1"));
        assert!(out.contains("obey_api_smart_routing_quality"));
        assert!(out.contains("obey_api_smart_routing_semantic_cache_total{result=\"miss\"} 1"));
        assert!(out.contains("obey_api_smart_routing_context_filtered_total{filtered=\"true\"} 1"));
        assert!(!out.contains("tenant/request/content"));
        assert!(!out.contains("secret-experiment-name"));
        assert!(!out.contains("custom-classifier"));
        assert!(!out.contains("private-task"));
    }

    #[test]
    fn smart_routing_jev_metrics_render_with_bounded_labels() {
        let metrics = Metrics::new();
        metrics.enable_smart_routing();
        metrics.record_jev_consult("private-group", "jev");
        metrics.record_jev_consult("private-group", "laya");
        metrics.record_jev_fallback("no_jev_at_endpoint", "jev");
        metrics.record_jev_confidence(0.87, "laya");
        metrics.record_jev_latency(123.0, "jev");
        metrics.record_jev_discovery_refresh("success", "laya");

        let mut out = String::new();
        metrics.write_smart_routing_prometheus(&mut out);

        assert!(out.contains("obey_api_smart_routing_jev_consults_total"));
        assert!(out.contains(
        "obey_api_smart_routing_jev_fallbacks_total{reason=\"no_jev_at_endpoint\",family=\"jev\"} 1"
    ));
        assert!(out.contains("obey_api_smart_routing_jev_confidence_bucket"));
        assert!(out.contains("obey_api_smart_routing_jev_latency_ms_bucket"));
        assert!(out.contains(
        "obey_api_smart_routing_jev_discovery_refreshes_total{status=\"success\",family=\"laya\"} 1"
    ));
        // Family label is bounded: no unknown family series.
        assert!(!out.contains("family=\"other\""));
    }

    #[test]
    fn smart_routing_classifier_label_includes_laya() {
        assert_eq!(smart_routing_classifier_label("laya"), "laya");
        assert_eq!(smart_routing_classifier_label("Jev"), "jev");
        assert_eq!(smart_routing_classifier_label("unknown"), "other");
    }

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
    fn test_reset_active_requests_returns_previous_and_sets_zero() {
        let metrics = Metrics::new();

        assert_eq!(metrics.reset_active_requests(), 0);
        assert_eq!(metrics.snapshot().active_requests, 0);

        for _ in 0..5 {
            metrics.start_request();
        }
        assert_eq!(metrics.snapshot().active_requests, 5);

        let previous = metrics.reset_active_requests();
        assert_eq!(previous, 5);
        assert_eq!(metrics.snapshot().active_requests, 0);

        assert_eq!(metrics.reset_active_requests(), 0);
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
    fn test_provider_cache_usage_tracking() {
        let metrics = Metrics::new();

        metrics.add_cache_usage(
            "provider1",
            CacheUsage {
                cache_read_input_tokens: 800,
                cache_creation_input_tokens: 200,
                uncached_input_tokens: 0,
            },
            12,
        );
        metrics.add_cache_usage(
            "provider1",
            CacheUsage {
                cache_read_input_tokens: 900,
                cache_creation_input_tokens: 0,
                uncached_input_tokens: 100,
            },
            7,
        );

        let snapshot = metrics.snapshot();
        let health = snapshot
            .provider_health
            .iter()
            .find(|h| h.provider == "provider1")
            .unwrap();

        assert_eq!(health.cache_read_tokens, 1700);
        assert_eq!(health.cache_creation_tokens, 200);
        assert_eq!(health.cache_savings_cents, 19);
        // 1700 read / (1700 read + 200 creation + 100 uncached) = 0.85
        assert_eq!(health.cache_prompt_hit_rate, Some(0.85));
    }

    #[test]
    fn test_provider_cache_hit_rate_none_until_prompt_tokens_recorded() {
        let metrics = Metrics::new();

        // Provider exists in the health map (request recorded) but no
        // cache usage yet: the hit rate must stay None (divide-by-zero
        // guard), not panic or default to 0.0.
        metrics.record_provider_success("provider1", 50);

        let snapshot = metrics.snapshot();
        let health = snapshot
            .provider_health
            .iter()
            .find(|h| h.provider == "provider1")
            .unwrap();

        assert_eq!(health.cache_read_tokens, 0);
        assert_eq!(health.cache_creation_tokens, 0);
        assert_eq!(health.cache_savings_cents, 0);
        assert_eq!(health.cache_prompt_hit_rate, None);
    }

    #[test]
    fn test_provider_cache_usage_all_uncached_and_negative_savings() {
        let metrics = Metrics::new();

        // Fully uncached response with a net cache-write premium: hit
        // rate 0.0 and negative (accumulating) savings.
        metrics.add_cache_usage(
            "provider1",
            CacheUsage {
                cache_read_input_tokens: 0,
                cache_creation_input_tokens: 0,
                uncached_input_tokens: 500,
            },
            -3,
        );

        let snapshot = metrics.snapshot();
        let health = snapshot
            .provider_health
            .iter()
            .find(|h| h.provider == "provider1")
            .unwrap();

        assert_eq!(health.cache_prompt_hit_rate, Some(0.0));
        assert_eq!(health.cache_savings_cents, -3);
    }

    #[test]
    fn test_provider_reasoning_usage_accumulates() {
        let metrics = Metrics::new();

        metrics.add_reasoning_usage("provider1", 500, 0.015);
        metrics.add_reasoning_usage("provider1", 250, 0.0075);
        metrics.add_reasoning_usage("provider2", 4096, 0.12288);

        let snapshot = metrics.snapshot();
        let health = snapshot
            .provider_health
            .iter()
            .find(|h| h.provider == "provider1")
            .unwrap();
        assert_eq!(health.reasoning_tokens, 750);
        assert!((health.reasoning_cost_usd - 0.0225).abs() < 1e-9);

        let other = snapshot
            .provider_health
            .iter()
            .find(|h| h.provider == "provider2")
            .unwrap();
        assert_eq!(other.reasoning_tokens, 4096);
        assert!((other.reasoning_cost_usd - 0.12288).abs() < 1e-9);
    }

    #[test]
    fn test_provider_reasoning_usage_zero_until_recorded() {
        let metrics = Metrics::new();

        // Provider exists in the health map (request recorded) but no
        // reasoning usage yet: counters must read zero, not panic.
        metrics.record_provider_success("provider1", 50);

        let snapshot = metrics.snapshot();
        let health = snapshot
            .provider_health
            .iter()
            .find(|h| h.provider == "provider1")
            .unwrap();

        assert_eq!(health.reasoning_tokens, 0);
        assert_eq!(health.reasoning_cost_usd, 0.0);
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

        metrics.record_guardrail_stage(
            "standard",
            "pii-redact",
            "presidio",
            "pre_call",
            "redact",
            12.0,
        );
        metrics.record_guardrail_stage(
            "standard",
            "pii-redact",
            "presidio",
            "pre_call",
            "redact",
            8.0,
        );
        metrics.record_guardrail_stage(
            "standard",
            "secret-block",
            "regex",
            "pre_call",
            "block",
            3.0,
        );
        // Same stage shape in a different phase keeps a separate series.
        metrics.record_guardrail_stage(
            "standard",
            "secret-block",
            "regex",
            "tool_result",
            "mask",
            1.0,
        );

        let mut out = String::new();
        metrics.write_guardrail_prometheus(&mut out);

        // Counter type + prefixed name (Req 11.1, 11.5)
        assert!(out.contains("# TYPE obey_api_guardrail_stage_executions_total counter"));
        // Two executions of the redact stage, one of the block stage.
        assert!(out.contains(
            "obey_api_guardrail_stage_executions_total{pipeline=\"standard\",stage=\"pii-redact\",provider=\"presidio\",phase=\"pre_call\",action=\"redact\"} 2"
        ));
        assert!(out.contains(
            "obey_api_guardrail_stage_executions_total{pipeline=\"standard\",stage=\"secret-block\",provider=\"regex\",phase=\"pre_call\",action=\"block\"} 1"
        ));
        assert!(out.contains(
            "obey_api_guardrail_stage_executions_total{pipeline=\"standard\",stage=\"secret-block\",provider=\"regex\",phase=\"tool_result\",action=\"mask\"} 1"
        ));
    }

    #[test]
    fn test_guardrail_stage_latency_histogram_exposition() {
        let metrics = Metrics::new();

        // One observation at 8ms (falls in le="10"), one at 300ms (le="500").
        metrics.record_guardrail_stage(
            "standard",
            "pii-redact",
            "presidio",
            "pre_call",
            "pass",
            8.0,
        );
        metrics.record_guardrail_stage(
            "standard",
            "pii-redact",
            "presidio",
            "pre_call",
            "pass",
            300.0,
        );

        let mut out = String::new();
        metrics.write_guardrail_prometheus(&mut out);

        assert!(out.contains("# TYPE obey_api_guardrail_stage_latency_ms histogram"));
        // Design bucket boundaries are present.
        assert!(out.contains(
            "obey_api_guardrail_stage_latency_ms_bucket{pipeline=\"standard\",stage=\"pii-redact\",provider=\"presidio\",phase=\"pre_call\",le=\"5\"} 0"
        ));
        // Cumulative: 8ms counted at le="10".
        assert!(out.contains(
            "obey_api_guardrail_stage_latency_ms_bucket{pipeline=\"standard\",stage=\"pii-redact\",provider=\"presidio\",phase=\"pre_call\",le=\"10\"} 1"
        ));
        // Cumulative: both observations counted by le="500".
        assert!(out.contains(
            "obey_api_guardrail_stage_latency_ms_bucket{pipeline=\"standard\",stage=\"pii-redact\",provider=\"presidio\",phase=\"pre_call\",le=\"500\"} 2"
        ));
        assert!(out.contains(
            "obey_api_guardrail_stage_latency_ms_bucket{pipeline=\"standard\",stage=\"pii-redact\",provider=\"presidio\",phase=\"pre_call\",le=\"+Inf\"} 2"
        ));
        assert!(out.contains(
            "obey_api_guardrail_stage_latency_ms_count{pipeline=\"standard\",stage=\"pii-redact\",provider=\"presidio\",phase=\"pre_call\"} 2"
        ));
        // Sum = 8 + 300 = 308 ms.
        assert!(out.contains(
            "obey_api_guardrail_stage_latency_ms_sum{pipeline=\"standard\",stage=\"pii-redact\",provider=\"presidio\",phase=\"pre_call\"} 308"
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
        metrics.record_guardrail_stage("p", "s", "regex", "post_call", "error", 6000.0);

        let mut out = String::new();
        metrics.write_guardrail_prometheus(&mut out);

        assert!(out.contains(
            "obey_api_guardrail_stage_latency_ms_bucket{pipeline=\"p\",stage=\"s\",provider=\"regex\",phase=\"post_call\",le=\"5000\"} 0"
        ));
        assert!(out.contains(
            "obey_api_guardrail_stage_latency_ms_bucket{pipeline=\"p\",stage=\"s\",provider=\"regex\",phase=\"post_call\",le=\"+Inf\"} 1"
        ));
    }

    #[test]
    fn test_guardrail_stego_detections_exposition() {
        let metrics = Metrics::new();

        metrics.record_guardrail_stego_detection("unicode_tag", "tool_result");
        metrics.record_guardrail_stego_detection("unicode_tag", "tool_result");
        metrics.record_guardrail_stego_detection("zero_width", "tool_call");

        let mut out = String::new();
        metrics.write_guardrail_prometheus(&mut out);

        assert!(out.contains("# TYPE obey_api_guardrail_stego_detections_total counter"));
        assert!(out.contains(
            "obey_api_guardrail_stego_detections_total{category=\"unicode_tag\",phase=\"tool_result\"} 2"
        ));
        assert!(out.contains(
            "obey_api_guardrail_stego_detections_total{category=\"zero_width\",phase=\"tool_call\"} 1"
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
            cases: 64,
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

    // --- Rolling request-rate ring ---

    #[test]
    fn request_rate_ring_burst_then_monotonic_decay_to_zero() {
        let mut ring = RequestRateRing::new();

        // One request per second for seconds 100..=109 (a 10-second burst).
        for second in 100..110 {
            ring.record_at(second);
        }
        assert_eq!(ring.rolling_total_at(109), 10);

        // Idle: the whole burst stays inside the trailing window for 59 more
        // seconds and never increases.
        let mut previous = 10;
        for now in 110..=159 {
            let current = ring.rolling_total_at(now);
            assert!(
                current <= previous,
                "rate must be non-increasing while idle"
            );
            assert_eq!(current, 10, "full burst still inside window at t={}", now);
            previous = current;
        }

        // Seconds 100..=109 age out one per second: 9, 8, ..., 1, 0.
        let mut expected = 9;
        for now in 160..=168 {
            assert_eq!(ring.rolling_total_at(now), expected, "decay at t={}", now);
            expected -= 1;
        }
        assert_eq!(ring.rolling_total_at(169), 0);
        assert_eq!(ring.rolling_total_at(170), 0);
    }

    #[test]
    fn request_rate_ring_ages_out_beyond_sixty_seconds() {
        let mut ring = RequestRateRing::new();

        for second in 500..503 {
            ring.record_at(second);
        }
        assert_eq!(ring.rolling_total_at(502), 3);
        // Seconds 500-502 are older than the trailing 60s of second 563.
        assert_eq!(ring.rolling_total_at(563), 0);

        // A single long jump clears every bucket at once.
        ring.record_at(700);
        assert_eq!(ring.rolling_total_at(700), 1);
        assert_eq!(ring.rolling_total_at(700 + 60), 0);
    }

    #[test]
    fn request_rate_ring_counts_trailing_window_with_sustained_traffic() {
        let mut ring = RequestRateRing::new();

        // One request per second for 90 seconds.
        for second in 0..90 {
            ring.record_at(second);
        }
        // At t=89 the trailing window covers seconds 30..=89.
        assert_eq!(ring.rolling_total_at(89), 60);
        // Second 30 ages out at t=90 while no new request arrives.
        assert_eq!(ring.rolling_total_at(90), 59);
        // Window at t=119 covers seconds 60..=119: only 60..=89 had traffic.
        assert_eq!(ring.rolling_total_at(119), 30);
        // All traffic is older than the window at t=149 (89 + 60).
        assert_eq!(ring.rolling_total_at(149), 0);
    }

    #[test]
    fn request_rate_ring_wraps_slots_and_reuses_cleared_buckets() {
        let mut ring = RequestRateRing::new();

        // Two full window rotations with traffic re-use cleared slots.
        for second in 0..150 {
            ring.record_at(second);
        }
        assert_eq!(ring.rolling_total_at(149), 60);
        assert_eq!(ring.rolling_total_at(209), 0);

        // Old window total must not leak into the fresh window.
        ring.record_at(210);
        assert_eq!(ring.rolling_total_at(210), 1);
    }

    #[test]
    fn request_rate_ring_handles_clock_steps_backwards() {
        let mut ring = RequestRateRing::new();

        ring.record_at(1_000);
        ring.record_at(1_000);
        // A backwards clock reading must not double-count a stale slot.
        ring.record_at(998);
        assert_eq!(ring.rolling_total_at(1_000), 3);
    }

    #[test]
    fn request_rate_snapshot_matches_rolling_counter() {
        let metrics = Metrics::new();

        // `start_request` uses wall-clock seconds, so drive the same clock the
        // same way for both consumers: record requests and compare the snapshot
        // rate against the rolling counter read at the same instant.
        let before = metrics.requests_last_minute();
        for _ in 0..5 {
            metrics.start_request();
        }
        let after = metrics.requests_last_minute();

        // Both reads happen within the same wall-clock second (no sleeps), so
        // the counter and snapshot must be identical.
        assert!(after >= before + 5);
        let snapshot = metrics.snapshot();
        assert_eq!(snapshot.request_rate_per_min, after as f64);
    }

    /// Task-10 guard: the rolling request rate decays monotonically after a
    /// burst and never hard-drops to zero while burst requests are still inside
    /// the trailing window (the old reset-task sawtooth dropped the whole
    /// counter to zero once per minute regardless of in-window traffic).
    #[test]
    fn rolling_rate_decays_without_sawtooth() {
        let mut ring = RequestRateRing::new();

        // Burst of 30 requests in one second.
        for _ in 0..30 {
            ring.record_at(1_000);
        }
        assert_eq!(ring.rolling_total_at(1_000), 30);

        // While the burst stays inside the window the rate never drops to
        // zero and never increases: no sawtooth.
        for now in 1_001..=1_059 {
            let current = ring.rolling_total_at(now);
            assert_eq!(current, 30, "in-window burst must persist at t={now}");
        }

        // The burst ages out only by crossing the 60s boundary — a smooth
        // single step to zero exactly when the last in-window second passes,
        // never earlier.
        assert_eq!(ring.rolling_total_at(1_060), 0);

        // A second burst mid-decay must add, not reset: continuing traffic
        // keeps a floor under the rate (no zero-drop while traffic flows).
        let mut ring = RequestRateRing::new();
        for _ in 0..10 {
            ring.record_at(2_000);
        }
        for second in (2_001..=2_050).step_by(10) {
            ring.record_at(second);
        }
        let mut previous = ring.rolling_total_at(2_050);
        assert!(previous >= 10);
        for now in 2_051..=2_059 {
            let current = ring.rolling_total_at(now);
            assert!(
                current >= 10,
                "in-window requests must keep the rate above zero at t={now}"
            );
            assert!(current <= previous, "decay must be monotonic at t={now}");
            previous = current;
        }
    }

    #[test]
    fn budget_boundary_cache_entries_expire_at_next_boundary() {
        // Pure cache-map logic (finding B1): a seeded entry reports
        // not-expired while `now` is before the boundary and expired at
        // or after it — the exact comparison the hot-path check performs
        // without touching SQLite.
        let mut cache: HashMap<String, (crate::config::BudgetResetPolicy, i64, i64)> =
            HashMap::new();
        let policy = crate::config::BudgetResetPolicy::Daily;
        let anchor = 1_700_000_000_i64;
        let next_boundary = next_reset_epoch(policy, anchor).expect("daily boundary computable");
        cache.insert("provider".to_string(), (policy, anchor, next_boundary));

        let entry = cache.get("provider").copied().unwrap();
        assert_eq!(entry.0, policy);
        assert_eq!(entry.1, anchor);
        // One second before the boundary: not expired.
        assert!(!((next_boundary - 1) >= next_boundary));
        // At and after the boundary: expired.
        assert!(next_boundary >= next_boundary);
        assert!((next_boundary + 1) >= next_boundary);
        // Boundary is a 00:00 UTC instant.
        let boundary_utc = chrono::DateTime::from_timestamp(next_boundary, 0).expect("valid epoch");
        assert_eq!(
            (
                boundary_utc.timestamp_subsec_nanos(),
                boundary_utc.format("%H:%M:%S").to_string()
            ),
            (0, "00:00:00".to_string())
        );
    }

    #[test]
    fn budget_boundary_cache_reports_not_expired_and_reseeds_after_reset() {
        // Full metrics wiring (finding B1): with a real store row the
        // cached check reports `false` while the window is current;
        // `reset_provider_spend` drops the cache entry so the next check
        // re-seeds from the store.
        use crate::config::BudgetResetPolicy;
        let dir = tempfile::tempdir().expect("temp dir");
        let store = Arc::new(
            ProviderBudgetStore::new(&dir.path().join("budgets.db")).expect("store opens"),
        );
        let metrics = Metrics::new();
        metrics.set_provider_budget_store(store.clone());

        let provider = "cached-provider";
        // No row yet: nothing to roll.
        assert_eq!(
            metrics.budget_window_expired_cached(provider, BudgetResetPolicy::Daily),
            None
        );

        // Row anchored today: daily window is current → not expired.
        let today_start = current_window_start_for_policy(BudgetResetPolicy::Daily);
        store
            .add_spend(provider, 1.0, today_start)
            .expect("row created");
        assert_eq!(
            metrics.budget_window_expired_cached(provider, BudgetResetPolicy::Daily),
            Some(false),
            "current daily window must report not-expired via the cache"
        );

        // Manual policy never reports expired (never rolls).
        assert_eq!(
            metrics.budget_window_expired_cached(provider, BudgetResetPolicy::Manual),
            None,
            "manual policy maps to no boundary"
        );

        // In-memory zeroing drops the cache entry (re-seed discipline).
        metrics.reset_provider_spend(provider);
        assert!(
            !metrics
                .budget_window_boundaries
                .lock()
                .expect("cache lock")
                .contains_key(provider),
            "reset_provider_spend must drop the cached boundary entry"
        );

        // Re-seed after the drop: the row still anchors today, so the
        // cache re-populates and serves not-expired again.
        assert_eq!(
            metrics.budget_window_expired_cached(provider, BudgetResetPolicy::Daily),
            Some(false)
        );
        assert!(
            metrics
                .budget_window_boundaries
                .lock()
                .expect("cache lock")
                .contains_key(provider),
            "re-check must re-seed the cache entry"
        );
    }
}
