//! Performance benchmark tests for the OBEY-API gateway.
//!
//! Validates Requirements 17.1-17.5:
//! 17.1 - Startup within 2 seconds
//! 17.2 - Forwarding overhead < 10ms
//! 17.3 - Memory < 100MB (structural check)
//! 17.4 - 100+ concurrent requests
//! 17.5 - Async I/O (verified by Tokio runtime usage)
//!
//! Router Responsiveness Optimization Spec (Task 1):
//! - Reusable measurement helpers for count, median, p95, p99
//! - Provider-selection fixtures for latency history coverage
//! - 4/16/64 candidate scenarios without upstream network calls

use std::sync::Arc;
use std::time::{Duration, Instant};

use axum::body::Body;
use axum::http::{Request, StatusCode};
use tokio::sync::RwLock;
use tower::ServiceExt;

use ai_gateway::compression::pipeline::{CompressionPipeline, CompressionRequestMetadata};
use ai_gateway::compression::token_counter::TokenCounter;
use ai_gateway::compression::{CompressiblePayload, CompressionContext, CompressionLevel};
use ai_gateway::config::*;
use ai_gateway::gateway::GatewayServer;
use ai_gateway::models::openai::{Choice, Message, OpenAIRequest, OpenAIResponse, Usage};
use ai_gateway::responses::{
    synthesize, translate, ResponsesInput, ResponsesRequest, SynthesisContext, TranslationContext,
};
use ai_gateway::router::circuit_breaker::CircuitState;
use ai_gateway::router::router::Router;
use ai_gateway::router::{CircuitBreaker, LatencyTracker};
use ai_gateway::smart_routing::config::ClassifierMode;
use ai_gateway::smart_routing::heuristic::HeuristicScorer;

mod common;

const TEST_TIMEOUT: Duration = Duration::from_secs(15);

/// Duration sample set for percentile reporting.
/// Records elapsed durations and computes count, median, p95, p99.
pub struct ResponsivenessSampleSet {
    samples: Vec<Duration>,
}

impl ResponsivenessSampleSet {
    pub fn new() -> Self {
        Self {
            samples: Vec::new(),
        }
    }

    pub fn record(&mut self, duration: Duration) {
        self.samples.push(duration);
    }

    pub fn report(&self) -> ResponsivenessReport {
        if self.samples.is_empty() {
            return ResponsivenessReport {
                count: 0,
                median: Duration::ZERO,
                p95: Duration::ZERO,
                p99: Duration::ZERO,
            };
        }

        let mut sorted: Vec<_> = self.samples.iter().collect();
        sorted.sort();

        let count = sorted.len();
        let median = *sorted[count / 2];
        let p95_idx = ((count as f64) * 0.95).floor() as usize;
        let p99_idx = ((count as f64) * 0.99).floor() as usize;
        let p95 = *sorted[p95_idx.min(count - 1)];
        let p99 = *sorted[p99_idx.min(count - 1)];

        ResponsivenessReport {
            count,
            median,
            p95,
            p99,
        }
    }
}

impl Default for ResponsivenessSampleSet {
    fn default() -> Self {
        Self::new()
    }
}

#[derive(Debug, Clone, Copy)]
pub struct ResponsivenessReport {
    pub count: usize,
    pub median: Duration,
    pub p95: Duration,
    pub p99: Duration,
}

impl std::fmt::Display for ResponsivenessReport {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(
            f,
            "count={}, median={:?}, p95={:?}, p99={:?}",
            self.count, self.median, self.p95, self.p99
        )
    }
}

/// Provider selection fixture configuration.
pub struct ProviderSelectionFixture {
    pub candidate_count: usize,
    pub history_coverage: LatencyHistoryCoverage,
    pub priority_spread: PrioritySpread,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum LatencyHistoryCoverage {
    Empty,
    Partial,
    Complete,
}

#[derive(Debug, Clone, Copy)]
pub enum PrioritySpread {
    Uniform,
    Mixed,
}

impl ProviderSelectionFixture {
    pub fn build_model_group(&self) -> ModelGroup {
        let models: Vec<ProviderModel> = (0..self.candidate_count)
            .map(|i| {
                let provider = format!("provider-{}", i);
                let model = format!("model-{}", i);
                let (input_cost, output_cost, priority) = match self.priority_spread {
                    PrioritySpread::Uniform => (10.0, 30.0, 100),
                    PrioritySpread::Mixed => {
                        let base = 10.0 + (i as f64 * 0.5);
                        let priority = 100 + (i as u32 % 10) * 10;
                        (base, base * 3.0, priority)
                    }
                };
                ProviderModel {
                    provider,
                    model,
                    cost_per_million_input_tokens: input_cost,
                    cost_per_million_output_tokens: output_cost,
                    priority,
                    structured_output_passthrough: None,
                    tier: None,
                    context_window: 0,
                    specializations: vec![],
                }
            })
            .collect();

        ModelGroup {
            name: "test-group".to_string(),
            version_fallback_enabled: false,
            compression: None,
            structured_output: None,
            memory: None,
            models,
        }
    }

    pub fn populate_latency_tracker(&self, tracker: &LatencyTracker) {
        match self.history_coverage {
            LatencyHistoryCoverage::Empty => {}
            LatencyHistoryCoverage::Partial => {
                for i in 0..(self.candidate_count / 2) {
                    let provider = format!("provider-{}", i);
                    let latency_ms = 50 + (i as u64 % 10) * 10;
                    tracker.update_latency(&provider, Duration::from_millis(latency_ms));
                }
            }
            LatencyHistoryCoverage::Complete => {
                for i in 0..self.candidate_count {
                    let provider = format!("provider-{}", i);
                    let latency_ms = 50 + (i as u64 % 10) * 10;
                    tracker.update_latency(&provider, Duration::from_millis(latency_ms));
                }
            }
        }
    }

    pub fn build_providers(&self) -> Vec<Provider> {
        (0..self.candidate_count)
            .map(|i| Provider {
                name: format!("provider-{}", i),
                provider_type: "openai".to_string(),
                base_url: Some(format!("http://127.0.0.1:{}", 16000 + i)),
                api_key_env: None,
                api_key_encrypted: None,
                api_secret_env: None,
                api_secret_encrypted: None,
                auth_method: None,
                resolved_api_key: None,
                resolved_api_secret: None,
                region: None,
                timeout_seconds: 30,
                ttfb_timeout_seconds: None,
                total_timeout_seconds: None,
                max_connections: 10,
                rate_limit_per_minute: 0,
                custom_headers: Default::default(),
                connection_pool: ProviderConnectionPoolConfig::default(),
                budget: None,
                manual_models: vec![],
                global_inference_profile: false,
                cross_region_inference: false,
                custom_vpc_endpoint: false,
                prompt_caching: false,
                compression: None,
                memory: None,
                reasoning: true,
                codex_base_url_override: None,
                codex_model_override: None,
                instructions_override: None,
                max_rate_limit_cooldown_seconds: None,
            })
            .collect()
    }
}

fn test_metrics() -> Arc<ai_gateway::metrics::Metrics> {
    Arc::new(ai_gateway::metrics::Metrics::new())
}

fn build_router_from_fixture(fixture: &ProviderSelectionFixture) -> Router {
    let mut config = Config {
        server: ServerConfig {
            host: "127.0.0.1".to_string(),
            port: 0,
            request_timeout_seconds: 30,
            max_request_size_mb: 10,
        },
        tls: None,
        admin: AdminConfig::default(),
        dashboard: DashboardConfig::default(),
        cors: CorsConfig::default(),
        providers: fixture.build_providers(),
        model_groups: vec![fixture.build_model_group()],
        circuit_breaker: CircuitBreakerConfig::default(),
        retry: RetryConfig::default(),
        logging: LoggingConfig::default(),
        semantic_cache: None,
        exact_cache: ExactCacheConfig::default(),
        prometheus: None,
        context: ContextConfig::default(),
        compression: Default::default(),
        memory: None,
        first_launch_completed: false,
        tray: TrayConfig::default(),
        codex_instructions_url: None,
        streaming: None,
        virtual_keys: Default::default(),
        loop_detection: Default::default(),
        guardrails: None,
        tool_compression: Default::default(),
        smart_routing: Default::default(),
        xhigh_models_allowlist: Default::default(),
        reasoning_models_allowlist: Default::default(),
        codex_search: None,
        structured_output: None,
    };
    common::isolate_databases(&mut config);
    Router::new(Arc::new(RwLock::new(config)), test_metrics())
}

/// Measure provider selection overhead in release mode.
/// This test is #[ignore] by default - run with `--ignored` for latency budgets.
#[tokio::test]
#[ignore]
async fn provider_selection_responsiveness_baseline() {
    let warmup_iterations = 10;
    let sample_iterations = 100;

    for &candidate_count in &[4, 16, 64] {
        for &history_coverage in &[
            LatencyHistoryCoverage::Empty,
            LatencyHistoryCoverage::Partial,
            LatencyHistoryCoverage::Complete,
        ] {
            let fixture = ProviderSelectionFixture {
                candidate_count,
                history_coverage,
                priority_spread: PrioritySpread::Mixed,
            };

            let router = build_router_from_fixture(&fixture);
            fixture.populate_latency_tracker(&router.get_latency_tracker());

            let model_group = fixture.build_model_group();

            for _ in 0..warmup_iterations {
                let _ = std::hint::black_box(router.select_provider_order(&model_group).await);
            }

            let mut samples = ResponsivenessSampleSet::new();
            for _ in 0..sample_iterations {
                let start = Instant::now();
                let _ = std::hint::black_box(router.select_provider_order(&model_group).await);
                samples.record(start.elapsed());
            }

            let report = samples.report();
            eprintln!(
                "provider_selection: candidates={} history={:?} {}",
                candidate_count, history_coverage, report
            );

            assert!(
                report.median < Duration::from_millis(10),
                "provider selection median too high for {} candidates: {:?}",
                candidate_count,
                report.median
            );
        }
    }
}

#[test]
fn responsiveness_sample_set_percentile_calculation() {
    let mut samples = ResponsivenessSampleSet::new();

    for i in 1..=100 {
        samples.record(Duration::from_micros(i));
    }

    let report = samples.report();
    assert_eq!(report.count, 100);
    // 1..=100 µs sorted; even count uses upper-middle element (51 µs).
    assert_eq!(report.median, Duration::from_micros(51));
    assert!(report.p95 >= Duration::from_micros(95));
    assert!(report.p99 >= Duration::from_micros(99));
}

#[test]
fn responsiveness_sample_set_empty() {
    let samples = ResponsivenessSampleSet::new();
    let report = samples.report();
    assert_eq!(report.count, 0);
    assert_eq!(report.median, Duration::ZERO);
}

#[test]
fn responsiveness_sample_set_single() {
    let mut samples = ResponsivenessSampleSet::new();
    samples.record(Duration::from_millis(42));
    let report = samples.report();
    assert_eq!(report.count, 1);
    assert_eq!(report.median, Duration::from_millis(42));
    assert_eq!(report.p95, Duration::from_millis(42));
    assert_eq!(report.p99, Duration::from_millis(42));
}

/// Test-only reference provider-ordering policy.
/// Mirrors the documented Router policy (priority asc → 10% cost threshold →
/// latency asc → cost asc; optional version-date sort) using the live
/// `get_latency` path, so the snapshot-based production sort can be checked
/// against it for behavioral equivalence.
fn reference_provider_order(
    models: &[ProviderModel],
    tracker: &LatencyTracker,
    version_fallback_enabled: bool,
) -> Vec<ProviderModel> {
    let mut candidates: Vec<ProviderModel> = models.to_vec();

    candidates.sort_by(|a, b| match a.priority.cmp(&b.priority) {
        std::cmp::Ordering::Equal => {
            let cost_a = a.total_cost();
            let cost_b = b.total_cost();
            let cost_diff = (cost_a - cost_b).abs();
            let cost_threshold = cost_a.min(cost_b) * 0.1;
            if cost_diff <= cost_threshold {
                let latency_a = tracker.get_latency(&a.provider);
                let latency_b = tracker.get_latency(&b.provider);
                latency_a
                    .partial_cmp(&latency_b)
                    .unwrap_or(std::cmp::Ordering::Equal)
            } else {
                cost_a
                    .partial_cmp(&cost_b)
                    .unwrap_or(std::cmp::Ordering::Equal)
            }
        }
        other => other,
    });

    if version_fallback_enabled {
        // Version-date sort (newer first); models without a date are oldest.
        let version_of = |name: &str| -> (u32, u32, u32) {
            let parts: Vec<&str> = name.split('-').collect();
            if parts.len() >= 3 {
                let len = parts.len();
                if let (Ok(y), Ok(m), Ok(d)) = (
                    parts[len - 3].parse::<u32>(),
                    parts[len - 2].parse::<u32>(),
                    parts[len - 1].parse::<u32>(),
                ) {
                    return (y, m, d);
                }
            }
            (0, 0, 0)
        };
        candidates.sort_by(|a, b| version_of(&b.model).cmp(&version_of(&a.model)));
    }

    candidates
}

fn names(order: &[ProviderModel]) -> Vec<String> {
    order.iter().map(|m| m.provider.clone()).collect()
}

/// Fixed-scenario equivalence: optimized select_provider_order must match the
/// reference policy across history coverage, equal priorities, 10% cost
/// boundaries, and version fallback. All fixtures run fully in-process.
#[tokio::test]
async fn provider_order_matches_reference_policy_fixed_scenarios() {
    struct Scenario {
        label: &'static str,
        models: Vec<ProviderModel>,
        known_latencies: Vec<(&'static str, u64)>,
        version_fallback: bool,
    }

    let mk = |provider: &str, model: &str, input: f64, output: f64, priority: u32| ProviderModel {
        provider: provider.to_string(),
        model: model.to_string(),
        cost_per_million_input_tokens: input,
        cost_per_million_output_tokens: output,
        priority,
        structured_output_passthrough: None,
        tier: None,
        context_window: 0,
        specializations: vec![],
    };

    let scenarios = vec![
        // Empty history: all candidates use the 100.0ms default.
        Scenario {
            label: "empty-history",
            models: vec![
                mk("p-a", "model-a", 10.0, 30.0, 100),
                mk("p-b", "model-b", 10.0, 30.0, 100),
                mk("p-c", "model-c", 12.0, 30.0, 100),
            ],
            known_latencies: vec![],
            version_fallback: false,
        },
        // Complete history: known latencies drive ordering.
        Scenario {
            label: "complete-history",
            models: vec![
                mk("p-slow", "model-a", 10.0, 30.0, 100),
                mk("p-fast", "model-b", 10.5, 30.0, 100),
            ],
            known_latencies: vec![("p-slow", 500), ("p-fast", 100)],
            version_fallback: false,
        },
        // Partial history: unknown providers share one coherent fallback.
        Scenario {
            label: "partial-history",
            models: vec![
                mk("p-known-low", "model-a", 10.0, 30.0, 100),
                mk("p-known-high", "model-b", 10.0, 30.0, 100),
                mk("p-unknown-1", "model-c", 10.0, 30.0, 100),
                mk("p-unknown-2", "model-d", 10.0, 30.0, 100),
            ],
            known_latencies: vec![("p-known-low", 100), ("p-known-high", 300)],
            version_fallback: false,
        },
        // Equal priorities, cost difference far beyond 10%: cost wins.
        Scenario {
            label: "equal-priority-cost-dominant",
            models: vec![
                mk("p-expensive", "model-a", 50.0, 150.0, 100),
                mk("p-cheap", "model-b", 5.0, 15.0, 100),
            ],
            known_latencies: vec![("p-expensive", 10), ("p-cheap", 900)],
            version_fallback: false,
        },
        // Exact 10% boundary: 10.0 vs 11.0 → diff 1.0 == threshold → latency path.
        Scenario {
            label: "cost-boundary-inclusive",
            models: vec![
                mk("p-base", "model-a", 10.0, 30.0, 100),
                mk("p-ten-percent", "model-b", 11.0, 30.0, 100),
            ],
            known_latencies: vec![("p-base", 800), ("p-ten-percent", 100)],
            version_fallback: false,
        },
        // Just beyond 10% boundary: cost dominates regardless of latency.
        Scenario {
            label: "cost-boundary-exclusive",
            models: vec![
                mk("p-base", "model-a", 10.0, 30.0, 100),
                mk("p-over-ten-percent", "model-b", 11.1, 30.0, 100),
            ],
            known_latencies: vec![("p-base", 900), ("p-over-ten-percent", 100)],
            version_fallback: false,
        },
        // Version fallback enabled: dated models sort newest-first.
        Scenario {
            label: "version-fallback",
            models: vec![
                mk("p-old", "gpt-4-turbo-2024-04-09", 10.0, 30.0, 100),
                mk("p-new", "gpt-4-turbo-2024-06-13", 10.0, 30.0, 100),
                mk("p-undated", "gpt-4", 10.0, 30.0, 100),
            ],
            known_latencies: vec![("p-old", 100), ("p-new", 100), ("p-undated", 100)],
            version_fallback: true,
        },
    ];

    for scenario in scenarios {
        let fixture = ProviderSelectionFixture {
            candidate_count: 4,
            history_coverage: LatencyHistoryCoverage::Empty,
            priority_spread: PrioritySpread::Uniform,
        };
        let router = build_router_from_fixture(&fixture);
        let tracker = router.get_latency_tracker();
        for (provider, latency_ms) in &scenario.known_latencies {
            tracker.update_latency(provider, Duration::from_millis(*latency_ms));
        }

        let model_group = ModelGroup {
            name: "test-group".to_string(),
            version_fallback_enabled: scenario.version_fallback,
            compression: None,
            structured_output: None,
            memory: None,
            models: scenario.models.clone(),
        };

        let actual = router.select_provider_order(&model_group).await;
        let expected =
            reference_provider_order(&scenario.models, &tracker, scenario.version_fallback);

        assert_eq!(
            names(&actual),
            names(&expected),
            "scenario {} diverged from reference policy",
            scenario.label
        );
    }
}

/// Generated equivalence: deterministic pseudo-random priorities, costs, and
/// latencies across history-coverage mixes must always match the reference.
#[tokio::test]
async fn provider_order_matches_reference_policy_generated_scenarios() {
    let mut seed = 0x5EED_1234u64;
    let mut next = || {
        seed = seed
            .wrapping_mul(6364136223846793005)
            .wrapping_add(1442695040888963407);
        seed >> 33
    };

    let fixture = ProviderSelectionFixture {
        candidate_count: 16,
        history_coverage: LatencyHistoryCoverage::Empty,
        priority_spread: PrioritySpread::Uniform,
    };
    let router = build_router_from_fixture(&fixture);
    let tracker = router.get_latency_tracker();

    for round in 0..25 {
        let model_count = 4 + (next() % 13) as usize; // 4..=16 candidates
        let models: Vec<ProviderModel> = (0..model_count)
            .map(|i| {
                let input = 5.0 + (next() % 100) as f64;
                ProviderModel {
                    provider: format!("gen-p{round}-{i}"),
                    model: format!("gen-model-{i}"),
                    cost_per_million_input_tokens: input,
                    cost_per_million_output_tokens: input * 3.0,
                    priority: 100 + (next() % 5) as u32, // heavy ties on purpose
                    structured_output_passthrough: None,
                    tier: None,
                    context_window: 0,
                    specializations: vec![],
                }
            })
            .collect();

        // Mix of coverage: known, unknown, and repeated providers.
        for (i, m) in models.iter().enumerate() {
            match i % 3 {
                0 | 1 => {
                    tracker.update_latency(&m.provider, Duration::from_millis(10 + next() % 1000))
                }
                _ => {}
            }
        }

        let version_fallback = round % 2 == 0;
        let model_group = ModelGroup {
            name: "test-group".to_string(),
            version_fallback_enabled: version_fallback,
            compression: None,
            structured_output: None,
            memory: None,
            models: models.clone(),
        };

        let actual = router.select_provider_order(&model_group).await;
        let expected = reference_provider_order(&models, &tracker, version_fallback);

        assert_eq!(
            names(&actual),
            names(&expected),
            "generated round {round} diverged from reference policy"
        );
    }
}

/// Build a minimal valid Config for performance tests.
fn test_config() -> Config {
    Config {
        server: ServerConfig {
            host: "127.0.0.1".to_string(),
            port: 0,
            request_timeout_seconds: 30,
            max_request_size_mb: 10,
        },
        tls: None,
        admin: AdminConfig::default(),
        dashboard: DashboardConfig::default(),
        cors: CorsConfig::default(),
        providers: vec![Provider {
            name: "test-provider".to_string(),
            provider_type: "openai".to_string(),
            base_url: Some("http://localhost:11434".to_string()),
            api_key_env: None,
            api_key_encrypted: None,
            api_secret_env: None,
            api_secret_encrypted: None,
            auth_method: None,
            resolved_api_key: None,
            resolved_api_secret: None,
            region: None,
            timeout_seconds: 30,
            ttfb_timeout_seconds: None,
            total_timeout_seconds: None,
            max_connections: 10,
            rate_limit_per_minute: 0,
            custom_headers: Default::default(),
            connection_pool: ProviderConnectionPoolConfig::default(),
            budget: None,
            manual_models: vec![],
            global_inference_profile: false,
            cross_region_inference: false,
            custom_vpc_endpoint: false,
            prompt_caching: false,
            compression: None,
            memory: None,
            reasoning: true,
            codex_base_url_override: None,
            codex_model_override: None,
            instructions_override: None,
            max_rate_limit_cooldown_seconds: None,
        }],
        model_groups: vec![ModelGroup {
            name: "test-group".to_string(),
            version_fallback_enabled: false,
            compression: None,
            memory: None,
            structured_output: None,
            models: vec![ProviderModel {
                provider: "test-provider".to_string(),
                model: "gpt-4".to_string(),
                cost_per_million_input_tokens: 30.0,
                cost_per_million_output_tokens: 60.0,
                priority: 100,
                structured_output_passthrough: None,
                tier: None,
                context_window: 0,
                specializations: vec![],
            }],
        }],
        circuit_breaker: CircuitBreakerConfig::default(),
        retry: RetryConfig::default(),
        logging: LoggingConfig::default(),
        semantic_cache: None,
        exact_cache: ExactCacheConfig::default(),
        prometheus: None,
        context: ai_gateway::config::ContextConfig::default(),
        compression: Default::default(),
        memory: None,
        first_launch_completed: false,
        tray: ai_gateway::config::TrayConfig::default(),
        codex_instructions_url: None,
        streaming: None,
        virtual_keys: Default::default(),
        loop_detection: Default::default(),
        guardrails: None,
        tool_compression: Default::default(),
        smart_routing: Default::default(),
        xhigh_models_allowlist: Default::default(),
        reasoning_models_allowlist: Default::default(),
        codex_search: None,
        structured_output: None,
    }
}

/// Helper: build a router from a config without binding to a port.
async fn build_app(mut config: Config) -> axum::Router {
    common::isolate_databases(&mut config);
    let server = GatewayServer::new(config, None).await.unwrap();
    server.build_router()
}

async fn with_test_timeout<F, T>(name: &str, future: F) -> T
where
    F: std::future::Future<Output = T>,
{
    tokio::time::timeout(TEST_TIMEOUT, future)
        .await
        .unwrap_or_else(|_| panic!("Test '{name}' exceeded {:?}", TEST_TIMEOUT))
}

#[test]
fn smart_routing_disabled_and_enabled_regression_measurement() {
    let request = OpenAIRequest {
        model: "test-group".to_string(),
        messages: (0..50)
            .map(|index| Message {
                role: "user".to_string(),
                content: serde_json::Value::String(format!(
                    "Analyze item {index}: explain the reasoning and provide a concise implementation. {}",
                    "x".repeat(1_900)
                )),
                extra: serde_json::Map::new(),
            })
            .collect(),
        stream: false,
        temperature: Some(0.0),
        max_tokens: Some(512),
        extra: serde_json::Map::new(),
    };
    let iterations = 10u32;

    let baseline_start = Instant::now();
    for _ in 0..iterations {
        std::hint::black_box(&request.model);
    }
    let baseline = baseline_start.elapsed();

    let disabled = ai_gateway::smart_routing::config::SmartRoutingConfig::default();
    let disabled_start = Instant::now();
    for _ in 0..iterations {
        if std::hint::black_box(disabled.enabled) {
            unreachable!();
        }
    }
    let disabled_elapsed = disabled_start.elapsed();

    let scorer = HeuristicScorer::default();
    let enabled_start = Instant::now();
    for _ in 0..iterations {
        std::hint::black_box(scorer.score(&request.messages));
    }
    let enabled_elapsed = enabled_start.elapsed();

    let disabled_per_request = disabled_elapsed / iterations;
    let enabled_per_request = enabled_elapsed / iterations;
    eprintln!(
        "smart routing benchmark: baseline={:?}/op disabled={:?}/op enabled_heuristic={:?}/op",
        baseline / iterations,
        disabled_per_request,
        enabled_per_request
    );
    assert!(!disabled.enabled);
    assert_eq!(disabled.classifier, ClassifierMode::Heuristic);
    assert!(
        enabled_per_request < Duration::from_millis(250),
        "debug-build regression gate exceeded: {enabled_per_request:?} per 50-message/95k-character request"
    );
}

// ---------------------------------------------------------------------------
// 1. Startup time — Req 17.1: < 2 seconds
// ---------------------------------------------------------------------------

#[tokio::test]
#[ignore = "wall-clock startup budget (Req 17.1): run perf suite with --ignored"]
async fn test_startup_time() {
    with_test_timeout("test_startup_time", async {
        let start = Instant::now();
        let mut config = test_config();
        common::isolate_databases(&mut config);
        let server = GatewayServer::new(config, None).await.unwrap();
        let _router = server.build_router();
        let elapsed = start.elapsed();

        assert!(
            elapsed < Duration::from_secs(2),
            "Startup took {:?}, exceeds 2s target (Req 17.1)",
            elapsed
        );
    })
    .await;
}

// ---------------------------------------------------------------------------
// 2. Forwarding overhead — Req 17.2: < 10ms per request
// ---------------------------------------------------------------------------

#[tokio::test]
#[ignore = "wall-clock forwarding budget (Req 17.2): run perf suite with --ignored"]
async fn test_forwarding_overhead() {
    with_test_timeout("test_forwarding_overhead", async {
        let app = build_app(test_config()).await;

        // Warm up: 5 requests to /health
        for _ in 0..5 {
            let warm = app.clone();
            let req = Request::get("/health").body(Body::empty()).unwrap();
            let _ = warm.oneshot(req).await.unwrap();
        }

        let iterations = 100u64;

        // Measure /health
        let start = Instant::now();
        for _ in 0..iterations {
            let svc = app.clone();
            let req = Request::get("/health").body(Body::empty()).unwrap();
            let resp = svc.oneshot(req).await.unwrap();
            assert_eq!(resp.status(), StatusCode::OK);
        }
        let health_avg = start.elapsed() / iterations as u32;

        // Measure /v1/models (exercises router layer)
        let start = Instant::now();
        for _ in 0..iterations {
            let svc = app.clone();
            let req = Request::get("/v1/models").body(Body::empty()).unwrap();
            let resp = svc.oneshot(req).await.unwrap();
            assert_eq!(resp.status(), StatusCode::OK);
        }
        let models_avg = start.elapsed() / iterations as u32;

        assert!(
            health_avg < Duration::from_millis(10),
            "/health avg {:?} exceeds 10ms target (Req 17.2)",
            health_avg
        );
        assert!(
            models_avg < Duration::from_millis(10),
            "/v1/models avg {:?} exceeds 10ms target (Req 17.2)",
            models_avg
        );
    })
    .await;
}

// ---------------------------------------------------------------------------
// 3. Concurrent requests — Req 17.4: 100+ concurrent
// ---------------------------------------------------------------------------

#[tokio::test]
#[ignore = "concurrency stress budget (Req 17.4): run perf suite with --ignored"]
async fn test_concurrent_requests() {
    with_test_timeout("test_concurrent_requests", async {
        let app = build_app(test_config()).await;
        let concurrency = 200usize;

        let mut handles = Vec::with_capacity(concurrency);
        for _ in 0..concurrency {
            let svc = app.clone();
            handles.push(tokio::spawn(async move {
                let req = Request::get("/health").body(Body::empty()).unwrap();
                let resp = svc.oneshot(req).await.unwrap();
                resp.status()
            }));
        }

        let mut ok_count = 0usize;
        for handle in handles {
            let status = tokio::time::timeout(TEST_TIMEOUT, handle)
                .await
                .expect("concurrent request task timed out")
                .expect("task panicked");
            if status == StatusCode::OK {
                ok_count += 1;
            }
        }

        assert_eq!(
            ok_count, concurrency,
            "Only {ok_count}/{concurrency} concurrent requests returned 200 (Req 17.4)"
        );
    })
    .await;
}

// ---------------------------------------------------------------------------
// 4. Release profile configured — Req 17.1-17.3 (build-time)
// ---------------------------------------------------------------------------

#[test]
fn test_release_profile_configured() {
    let cargo_toml =
        std::fs::read_to_string(concat!(env!("CARGO_MANIFEST_DIR"), "/../../Cargo.toml"))
            .expect("workspace Cargo.toml not found");

    assert!(
        cargo_toml.contains("lto = true"),
        "Release profile missing LTO"
    );
    assert!(
        cargo_toml.contains("strip = true"),
        "Release profile missing strip"
    );
    assert!(
        cargo_toml.contains("opt-level = 3"),
        "Release profile missing opt-level = 3"
    );
    assert!(
        cargo_toml.contains("codegen-units = 1"),
        "Release profile missing codegen-units = 1"
    );
}

// ---------------------------------------------------------------------------
// Circuit-breaker concurrent contention measurement (spec task 4, Req 3.1-3.7)
// ---------------------------------------------------------------------------

/// Concurrent contention measurement for `CircuitBreaker::is_available()`.
///
/// The breaker uses a `tokio::sync::RwLock` **write** lock for every
/// `is_available()` call — even the common Closed path — which serializes
/// all concurrent callers. This `#[ignore]` benchmark measures per-call
/// latency under 100-way concurrency across four states to provide evidence
/// for the retain-vs-redesign decision.
///
/// **Decision rule:** if p95 remains sub-millisecond at 100-way concurrency
/// in all states, the critical section is short enough that the current
/// exclusive-lock design is retained (task 5 not authorized). If p95 exceeds
/// 1 ms in any state, task 5 is authorized for a race-safe optimization.
#[tokio::test(flavor = "multi_thread", worker_threads = 8)]
#[ignore]
async fn circuit_breaker_concurrent_contention_measurement() {
    use tokio::sync::Barrier;

    const CONCURRENCY: usize = 100;
    const ITERATIONS: usize = 50;

    async fn measure_state(
        label: &str,
        cb: Arc<CircuitBreaker>,
        concurrency: usize,
        iterations: usize,
    ) {
        let mut all_samples = ResponsivenessSampleSet::new();

        for _ in 0..iterations {
            let barrier = Arc::new(Barrier::new(concurrency + 1));
            let mut handles = Vec::with_capacity(concurrency);

            for _ in 0..concurrency {
                let cb = cb.clone();
                let barrier = barrier.clone();
                handles.push(tokio::spawn(async move {
                    barrier.wait().await;
                    let start = Instant::now();
                    let _ = std::hint::black_box(cb.is_available().await);
                    start.elapsed()
                }));
            }

            barrier.wait().await;

            for handle in handles {
                all_samples.record(handle.await.unwrap());
            }
        }

        let report = all_samples.report();
        eprintln!(
            "cb_contention[{}]: concurrency={} {}",
            label, concurrency, report
        );
    }

    // State 1: Closed — the hot path; every call takes a write lock.
    {
        let cb = Arc::new(CircuitBreaker::new(3));
        measure_state("closed", cb, CONCURRENCY, ITERATIONS).await;
    }

    // State 2: Open not-ready — within the backoff window; all calls
    // acquire the write lock, check elapsed < retry_after, return false.
    {
        let cb = Arc::new(CircuitBreaker::with_backoff_sequence(
            1,
            vec![Duration::from_secs(300)],
        ));
        cb.record_failure().await;
        assert!(!cb.is_available().await);
        measure_state("open-not-ready", cb, CONCURRENCY, ITERATIONS).await;
    }

    // State 3: Open ready-to-transition — backoff elapsed; the first call
    // transitions to HalfOpen, subsequent calls see HalfOpen.
    {
        let cb = Arc::new(CircuitBreaker::with_backoff_sequence(
            1,
            vec![Duration::from_millis(10)],
        ));
        cb.record_failure().await;
        assert!(!cb.is_available().await);
        tokio::time::sleep(Duration::from_millis(20)).await;
        measure_state("open-ready-transition", cb, CONCURRENCY, ITERATIONS).await;
    }

    // State 4: Half-open (stable) — pre-transitioned; all calls see HalfOpen.
    {
        let cb = Arc::new(CircuitBreaker::with_backoff_sequence(
            1,
            vec![Duration::from_millis(10)],
        ));
        cb.record_failure().await;
        tokio::time::sleep(Duration::from_millis(20)).await;
        let _ = cb.is_available().await; // force transition to HalfOpen
        assert_eq!(cb.get_state().await, CircuitState::HalfOpen);
        measure_state("half-open", cb, CONCURRENCY, ITERATIONS).await;
    }
}

// ---------------------------------------------------------------------------
// Compression preparation across failover attempts (spec task 6, Req 4.x)
// ---------------------------------------------------------------------------

/// Heterogeneous provider attempt matrix simulating a failover chain.
///
/// Each entry mirrors what `prepare_compressed_request_with_stats` receives
/// per attempt: a provider/model identity (tokenizer family), a context
/// window, and a prompt-caching flag.
struct AttemptProfile {
    provider: &'static str,
    model: &'static str,
    context_window: u32,
    prompt_caching: bool,
}

impl AttemptProfile {
    fn matrix() -> Vec<Self> {
        vec![
            Self {
                provider: "primary-openai",
                model: "gpt-4o",
                context_window: 128_000,
                prompt_caching: true,
            },
            Self {
                provider: "failover-anthropic",
                model: "claude-sonnet-4",
                context_window: 200_000,
                prompt_caching: true,
            },
            Self {
                provider: "failover-openai-legacy",
                model: "gpt-4-turbo",
                context_window: 8_192,
                prompt_caching: false,
            },
            Self {
                provider: "failover-local",
                model: "llama-3-70b",
                context_window: 16_384,
                prompt_caching: false,
            },
        ]
    }
}

/// Builds a request shaped like the smart-routing regression fixture:
/// 50 user turns of ~1.9 KB plus two tool definitions in `extra`.
fn build_compression_profile_request() -> OpenAIRequest {
    let mut extra = serde_json::Map::new();
    extra.insert(
        "tools".to_string(),
        serde_json::json!([
            {
                "type": "function",
                "function": {
                    "name": "lookup_records",
                    "description": "Look up records by identifier and return matching rows",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "identifier": { "type": "string" },
                            "limit": { "type": "integer" }
                        },
                        "required": ["identifier"]
                    }
                }
            },
            {
                "type": "function",
                "function": {
                    "name": "summarize_document",
                    "description": "Summarize a document into structured sections",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "document_id": { "type": "string" },
                            "sections": { "type": "array", "items": { "type": "string" } }
                        },
                        "required": ["document_id"]
                    }
                }
            }
        ]),
    );

    OpenAIRequest {
        model: "test-group".to_string(),
        messages: (0..50)
            .map(|index| Message {
                role: if index % 10 == 0 { "system" } else { "user" }.to_string(),
                content: serde_json::Value::String(format!(
                    "Analyze item {index}: explain the reasoning and provide a concise implementation. {}",
                    "x".repeat(1_900)
                )),
                extra: serde_json::Map::new(),
            })
            .collect(),
        stream: false,
        temperature: Some(0.0),
        max_tokens: Some(512),
        extra,
    }
}

/// Profiles compression preparation cost per failover attempt without any
/// upstream network call. Breaks the per-attempt cost into:
///
/// 1. `token_count_only` — raw `TokenCounter::count_request` on the shared
///    request (the always-paid, provider-independent floor: every attempt
///    recounts identical content before compression decisions).
/// 2. `pipeline_disabled` — full `compress_auto` with compression disabled
///    (counting + auto-trigger decision, no engines).
/// 3. `pipeline_lite` — full `compress_auto` with Lite triggered (counting +
///    engine chain + candidate re-count + clone overhead).
/// 4. engine wall-clock breakdown from `engine_results` (coarse, ms-resolution).
///
/// **Evidence question (task 6):** if `pipeline_lite` is dominated by work
/// whose inputs are identical across attempts (token counting of unchanged
/// messages, engine transforms of unchanged content), material reusable
/// provider-independent work exists and task 7 reuse is authorized. If engine
/// time is negligible relative to end-to-end routing, retain per-attempt
/// preparation.
#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
#[ignore]
async fn compression_preparation_failover_profile() {
    let attempts = AttemptProfile::matrix();
    let request = build_compression_profile_request();
    let iterations = 30usize;

    // Enabled pipeline: Lite level, auto threshold below request size so the
    // engine chain runs on every attempt.
    let mut enabled_config = ai_gateway::compression::config::CompressionConfig::default();
    enabled_config.enabled = true;
    enabled_config.default_level = CompressionLevel::Lite;
    enabled_config.auto_threshold_tokens = 512;
    let enabled_pipeline = CompressionPipeline::from_config(enabled_config.clone());

    // Disabled pipeline: auto path taken (threshold > 0) but engines skipped.
    let mut disabled_config = ai_gateway::compression::config::CompressionConfig::default();
    disabled_config.enabled = false;
    disabled_config.auto_threshold_tokens = 512;
    let disabled_pipeline = CompressionPipeline::from_config(disabled_config.clone());

    let counter = TokenCounter::new();

    let mut token_count_only = ResponsivenessSampleSet::new();
    let mut pipeline_disabled = ResponsivenessSampleSet::new();
    let mut pipeline_lite = ResponsivenessSampleSet::new();
    let mut engine_wall_ms_total: Vec<u64> = Vec::new();

    let effective_enabled = enabled_config.resolve(None, None);
    let effective_disabled = disabled_config.resolve(None, None);

    // Warmup: one pass per attempt per scenario (tokenizer singletons, allocs).
    for attempt in &attempts {
        let payload = CompressiblePayload::from(&request);
        let context = profile_context(attempt);
        let _ = enabled_pipeline
            .compress_auto(
                payload,
                context,
                effective_enabled,
                CompressionRequestMetadata::default(),
            )
            .await;
    }

    for _ in 0..iterations {
        for attempt in &attempts {
            // 1. Raw token counting floor.
            {
                let mut req_model = request.clone();
                req_model.model = attempt.model.to_string();
                let start = Instant::now();
                let tokens = std::hint::black_box(counter.count_request(&req_model));
                let elapsed = start.elapsed();
                assert!(tokens > 0);
                token_count_only.record(elapsed);
            }

            // 2. Disabled auto path.
            {
                let payload = CompressiblePayload::from(&request);
                let context = profile_context(attempt);
                let start = Instant::now();
                let result = std::hint::black_box(
                    disabled_pipeline
                        .compress_auto(
                            payload,
                            context,
                            effective_disabled,
                            CompressionRequestMetadata::default(),
                        )
                        .await,
                );
                pipeline_disabled.record(start.elapsed());
                assert!(!result.error);
                assert!(result.engine_results.is_empty());
            }

            // 3. Enabled Lite path.
            {
                let payload = CompressiblePayload::from(&request);
                let context = profile_context(attempt);
                let start = Instant::now();
                let result = std::hint::black_box(
                    enabled_pipeline
                        .compress_auto(
                            payload,
                            context,
                            effective_enabled,
                            CompressionRequestMetadata::default(),
                        )
                        .await,
                );
                pipeline_lite.record(start.elapsed());
                assert!(!result.error, "pipeline errored: {:?}", result.errors);
                assert!(!result.timed_out);
                let engine_wall: u64 = result
                    .engine_results
                    .iter()
                    .map(|engine| engine.duration_ms)
                    .sum();
                engine_wall_ms_total.push(engine_wall);
            }
        }
    }

    let count_report = token_count_only.report();
    let disabled_report = pipeline_disabled.report();
    let lite_report = pipeline_lite.report();
    let engine_wall_sum: u64 = engine_wall_ms_total.iter().sum();

    eprintln!(
        "compression_preparation[{} attempts x {} iters]:",
        attempts.len(),
        iterations
    );
    eprintln!("  token_count_only : {}", count_report);
    eprintln!("  pipeline_disabled: {}", disabled_report);
    eprintln!("  pipeline_lite    : {}", lite_report);
    eprintln!(
        "  engine_wall_ms   : sum={}ms over {} attempts (coarse ms-resolution)",
        engine_wall_sum,
        engine_wall_ms_total.len()
    );

    // Failover duplication factor: work repeated when all attempts in the
    // matrix fail over, versus a hypothetical single shared preparation.
    let failover_attempts = attempts.len() as u64;
    let repeat_ms = failover_attempts * (lite_report.median.as_micros() as u64) / 1_000;
    eprintln!(
        "  failover_repetition: ~{}ms median re-spent per full {}-attempt failover chain",
        repeat_ms, failover_attempts
    );

    // Evidence decision: count-only floor vs full Lite pipeline.
    let floor_ratio =
        lite_report.median.as_secs_f64() / count_report.median.as_secs_f64().max(f64::EPSILON);
    eprintln!(
        "  evidence: pipeline_lite/token_count median ratio = {:.1}x; counting is {} of full preparation",
        floor_ratio,
        format_args!(
            "{:.0}%",
            100.0 / floor_ratio.max(1.0)
        )
    );
}

fn profile_context(attempt: &AttemptProfile) -> CompressionContext {
    let mut context = CompressionContext::new(attempt.model, attempt.provider);
    context.context_window = attempt.context_window;
    context.prompt_caching_enabled = attempt.prompt_caching;
    context.tool_compression_applied = false;
    context
}

// ---------------------------------------------------------------------------
// Configuration lookup & budget preparation cardinality (spec task 8, Req 5.x)
// ---------------------------------------------------------------------------

/// Cardinality scenario for configuration-scan benchmarks.
struct CardinalityScenario {
    label: &'static str,
    groups: usize,
    models_per_group: usize,
    providers: usize,
}

/// Builds a Config with `groups × models_per_group` provider-model entries
/// and `providers` providers (every even-indexed provider carries a budget),
/// plus the Router over that config. No ports are bound; Router construction
/// is in-process with isolated temp databases.
fn build_cardinality_scenario(scenario: &CardinalityScenario) -> (Arc<RwLock<Config>>, Router) {
    let providers = (0..scenario.providers)
        .map(|i| Provider {
            name: format!("provider-{}", i),
            provider_type: "openai".to_string(),
            base_url: Some(format!("http://127.0.0.1:{}", 16000 + i)),
            api_key_env: None,
            api_key_encrypted: None,
            api_secret_env: None,
            api_secret_encrypted: None,
            auth_method: None,
            resolved_api_key: None,
            resolved_api_secret: None,
            region: None,
            timeout_seconds: 30,
            ttfb_timeout_seconds: None,
            total_timeout_seconds: None,
            max_connections: 10,
            rate_limit_per_minute: 0,
            custom_headers: Default::default(),
            connection_pool: ProviderConnectionPoolConfig::default(),
            budget: (i % 2 == 0).then(|| ProviderBudgetConfig {
                limit_usd: 10.0,
                reset_policy: BudgetResetPolicy::Manual,
            }),
            manual_models: vec![],
            global_inference_profile: false,
            cross_region_inference: false,
            custom_vpc_endpoint: false,
            prompt_caching: i % 3 == 0,
            compression: None,
            memory: None,
            reasoning: true,
            codex_base_url_override: None,
            codex_model_override: None,
            instructions_override: None,
            max_rate_limit_cooldown_seconds: None,
        })
        .collect();

    let model_groups = (0..scenario.groups)
        .map(|g| {
            let models = (0..scenario.models_per_group)
                .map(|m| ProviderModel {
                    provider: format!(
                        "provider-{}",
                        (g * scenario.models_per_group + m) % scenario.providers
                    ),
                    model: format!("model-{}-{}", g, m),
                    cost_per_million_input_tokens: 10.0 + m as f64,
                    cost_per_million_output_tokens: 30.0 + m as f64,
                    priority: 100 + (((g + m) % 10) * 10) as u32,
                    structured_output_passthrough: None,
                    tier: None,
                    context_window: 0,
                    specializations: vec![],
                })
                .collect();
            ModelGroup {
                name: format!("group-{}", g),
                version_fallback_enabled: false,
                compression: None,
                structured_output: None,
                memory: None,
                models,
            }
        })
        .collect();

    let mut config = Config {
        server: ServerConfig {
            host: "127.0.0.1".to_string(),
            port: 0,
            request_timeout_seconds: 30,
            max_request_size_mb: 10,
        },
        tls: None,
        admin: AdminConfig::default(),
        dashboard: DashboardConfig::default(),
        cors: CorsConfig::default(),
        providers,
        model_groups,
        circuit_breaker: CircuitBreakerConfig::default(),
        retry: RetryConfig::default(),
        logging: LoggingConfig::default(),
        semantic_cache: None,
        exact_cache: ExactCacheConfig::default(),
        prometheus: None,
        context: ContextConfig::default(),
        compression: Default::default(),
        memory: None,
        first_launch_completed: false,
        tray: TrayConfig::default(),
        codex_instructions_url: None,
        streaming: None,
        virtual_keys: Default::default(),
        loop_detection: Default::default(),
        guardrails: None,
        tool_compression: Default::default(),
        smart_routing: Default::default(),
        xhigh_models_allowlist: Default::default(),
        reasoning_models_allowlist: Default::default(),
        codex_search: None,
        structured_output: None,
    };
    common::isolate_databases(&mut config);
    let shared = Arc::new(RwLock::new(config));
    let router = Router::new(shared.clone(), test_metrics());
    (shared, router)
}

/// Mirrors the per-request provider-budget map preparation in
/// `route_with_failover_for_group` (router.rs): read lock + linear scan +
/// name clone per budgeted provider.
async fn build_budget_map(config: &RwLock<Config>) -> std::collections::HashMap<String, f64> {
    let config = config.read().await;
    config
        .providers
        .iter()
        .filter_map(|provider| {
            provider
                .budget
                .as_ref()
                .map(|budget| (provider.name.clone(), budget.limit_usd))
        })
        .collect()
}

/// Measures `Router::find_model_group()` (group-name and per-model linear
/// scans under the config read lock, including the `ModelGroup` clone) and
/// the per-request provider-budget map preparation across typical and high
/// cardinality. Includes a cascade case (two sequential lookups) mirroring
/// smart-routing re-resolution. No upstream calls.
///
/// **Decision rule (task 8):** retain direct scans if worst-case p95 stays
/// below 100µs at the documented high cardinality (512 entries, 128
/// providers); authorize the immutable-index task otherwise.
#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
#[ignore]
async fn config_lookup_cardinality_measurement() {
    const WARMUP: usize = 20;
    const ITERATIONS: usize = 300;

    let scenarios = [
        CardinalityScenario {
            label: "typical",
            groups: 8,
            models_per_group: 4,
            providers: 16,
        },
        CardinalityScenario {
            label: "high",
            groups: 64,
            models_per_group: 8,
            providers: 128,
        },
    ];

    for scenario in &scenarios {
        let (shared_config, router) = build_cardinality_scenario(scenario);
        let last_group = format!("group-{}", scenario.groups - 1);
        let last_model = format!(
            "model-{}-{}",
            scenario.groups - 1,
            scenario.models_per_group - 1
        );
        let entries = scenario.groups * scenario.models_per_group;

        let cases: [(&str, String); 4] = [
            ("best_group_name", "group-0".to_string()),
            ("last_group_name", last_group.clone()),
            ("last_model_entry", last_model.clone()),
            ("miss_full_scan", "nonexistent-model".to_string()),
        ];

        for (label, target) in cases {
            for _ in 0..WARMUP {
                let _ = router.find_model_group(&target).await;
            }
            let mut samples = ResponsivenessSampleSet::new();
            for _ in 0..ITERATIONS {
                let start = Instant::now();
                let found = std::hint::black_box(router.find_model_group(&target).await);
                samples.record(start.elapsed());
                drop(found);
            }
            eprintln!(
                "config_lookup[{}][{}] ({} entries): {}",
                scenario.label,
                label,
                entries,
                samples.report()
            );
        }

        // Cascade re-resolution: group lookup then pinned-model lookup,
        // mirroring smart-routing cascade resolution order.
        {
            for _ in 0..WARMUP {
                let _ = router.find_model_group("group-0").await;
                let _ = router.find_model_group(&last_model).await;
            }
            let mut samples = ResponsivenessSampleSet::new();
            for _ in 0..ITERATIONS {
                let start = Instant::now();
                let _ = router.find_model_group("group-0").await;
                let _ = std::hint::black_box(router.find_model_group(&last_model).await);
                samples.record(start.elapsed());
            }
            eprintln!(
                "config_lookup[{}][cascade_x2] ({} entries): {}",
                scenario.label,
                entries,
                samples.report()
            );
        }

        // Per-request budget map preparation.
        {
            for _ in 0..WARMUP {
                drop(build_budget_map(&shared_config).await);
            }
            let mut samples = ResponsivenessSampleSet::new();
            for _ in 0..ITERATIONS {
                let start = Instant::now();
                let budgets = std::hint::black_box(build_budget_map(&shared_config).await);
                samples.record(start.elapsed());
                assert_eq!(
                    budgets.len(),
                    scenario.providers.div_ceil(2),
                    "expected half the providers budgeted"
                );
            }
        eprintln!(
            "config_lookup[{}][budget_map] ({} providers, {} budgeted): {}",
            scenario.label,
            scenario.providers,
            scenario.providers.div_ceil(2),
            samples.report()
        );
        }
    }
}

/// Measure pure Responses-API translation overhead (translate + synthesize)
/// with no network I/O. Asserts per-iteration < 10ms and p95 < 10ms across
/// 100 iterations, matching the existing forwarding-overhead budget.
#[tokio::test]
#[ignore]
async fn responses_translation_overhead_within_budget() {
    const ITERATIONS: usize = 100;
    const BUDGET: Duration = Duration::from_millis(10);

    let req = ResponsesRequest {
        model: "gpt-4o".to_string(),
        input: ResponsesInput::Text("Translate this prompt for latency measurement.".to_string()),
        instructions: None,
        previous_response_id: None,
        store: false,
        metadata: None,
        temperature: None,
        top_p: None,
        max_output_tokens: None,
        truncation: None,
        parallel_tool_calls: None,
        reasoning: None,
        text: None,
        tools: Vec::new(),
        tool_choice: None,
        stream: false,
        stream_options: None,
        extra: serde_json::Map::new(),
    };

    let tx_ctx = TranslationContext {
        resolved_model: "gpt-4o",
        model_supports_reasoning: false,
    };

    let chat_response = OpenAIResponse {
        id: "chatcmpl-perf".to_string(),
        object: "chat.completion".to_string(),
        created: 1_700_000_000,
        model: "gpt-4o".to_string(),
        choices: vec![Choice {
            index: 0,
            message: Message {
                role: "assistant".to_string(),
                content: serde_json::json!("Response from the gateway."),
                extra: serde_json::Map::new(),
            },
            finish_reason: Some("stop".to_string()),
            extra: serde_json::Map::new(),
        }],
        usage: Usage {
            prompt_tokens: 10,
            completion_tokens: 5,
            total_tokens: 15,
            extra: serde_json::Map::new(),
        },
        extra: serde_json::Map::new(),
    };

    let synth_ctx = SynthesisContext {
        request_model: "gpt-4o",
        request_instructions: None,
        request_temperature: None,
        request_top_p: None,
        request_tools: &[],
        request_tool_choice: None,
        request_metadata: None,
        request_store: false,
        request_previous_response_id: None,
        request_truncation: None,
        request_text: None,
        request_parallel_tool_calls: None,
        request_reasoning: None,
    };

    // Warmup
    for _ in 0..10 {
        let translated = translate(&req, None, &tx_ctx).unwrap();
        let _ = synthesize(&chat_response, &synth_ctx);
        drop(translated);
    }

    let mut samples = ResponsivenessSampleSet::new();
    for _ in 0..ITERATIONS {
        let start = Instant::now();
        let translated = std::hint::black_box(translate(&req, None, &tx_ctx).unwrap());
        let _ = std::hint::black_box(synthesize(&chat_response, &synth_ctx));
        drop(translated);
        let elapsed = start.elapsed();
        samples.record(elapsed);
        assert!(
            elapsed < BUDGET,
            "per-iteration overhead {:?} exceeds {:?}",
            elapsed,
            BUDGET
        );
    }

    let report = samples.report();
    eprintln!("responses_translation_overhead: {}", report);
    assert!(
        report.p95 < BUDGET,
        "p95 {:?} exceeds {:?}",
        report.p95,
        BUDGET
    );
}
