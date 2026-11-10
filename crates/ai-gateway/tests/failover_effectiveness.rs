//! Failover-effectiveness and performance-budget suites for the
//! retry-failover-core-overhaul spec (design.md, Testing Strategy A8).
//!
//! - Failover effectiveness: for each retryable/transient primary failure
//!   class (500, 502, 503, 429, 408, connection-refused, TTFB timeout),
//!   a batch of client requests must ultimately succeed through a healthy
//!   fallback at a >= 99% rate (successful client responses / requests
//!   with an eligible healthy fallback, per scenario).
//! - Performance budget: median gateway overhead on the buffered dispatch
//!   path (classification + retry budget + attempt-count header active by
//!   default) stays < 10 ms, excluding upstream latency (subtracted as a
//!   direct loopback round-trip floor against the same wiremock upstream).
//!
//! Harness follows the repo conventions: wiremock upstreams, the full
//! `GatewayServer` router exercised via `tower::ServiceExt::oneshot`
//! (no ports bound, no live server), and `common::isolate_databases`
//! for SQLite isolation.

use std::net::TcpListener;
use std::time::{Duration, Instant};

use axum::body::Body;
use axum::http::{Request, StatusCode};
use tower::ServiceExt;
use wiremock::matchers::{method, path};
use wiremock::{Mock, MockServer, ResponseTemplate};

use ai_gateway::config::Config;
use ai_gateway::gateway::GatewayServer;

mod common;

/// Wall-clock budget per effectiveness scenario (suite target < 60 s).
const SCENARIO_TIMEOUT: Duration = Duration::from_secs(30);
/// Wall-clock budget for the ignored overhead benchmark.
const OVERHEAD_TIMEOUT: Duration = Duration::from_secs(60);
/// Per-request ceiling inside one scenario batch.
const PER_REQUEST_TIMEOUT: Duration = Duration::from_secs(15);
/// Requests per failover-effectiveness scenario (99% => >= 29 of 30).
const SCENARIO_REQUESTS: usize = 30;
/// Warmup and measured iterations for the overhead benchmark.
const OVERHEAD_WARMUP: usize = 10;
const OVERHEAD_ITERATIONS: usize = 100;

const EFFECTIVENESS_GROUP: &str = "effectiveness-group";
const OVERHEAD_GROUP: &str = "overhead-group";

/// An SSE body for a GLM-style turn: reasoning, content, finish_reason and
/// the `include_usage` frame — the shape the buffered dispatch reassembles
/// (mirrors router.rs's STREAMED_TURN_SSE fixture).
const STREAMED_TURN_SSE: &str = concat!(
    ": keep-alive\n\n",
    "data: {\"id\":\"c1\",\"object\":\"chat.completion.chunk\",\"created\":1,\"model\":\"upstream-model\",\"choices\":[{\"index\":0,\"delta\":{\"role\":\"assistant\",\"reasoning_content\":\"thinking\"}}],\"usage\":null}\n\n",
    "data: {\"id\":\"c1\",\"object\":\"chat.completion.chunk\",\"created\":1,\"model\":\"upstream-model\",\"choices\":[{\"index\":0,\"delta\":{\"content\":\"Hello\"}}],\"usage\":null}\n\n",
    "data: {\"id\":\"c1\",\"object\":\"chat.completion.chunk\",\"created\":1,\"model\":\"upstream-model\",\"choices\":[{\"index\":0,\"delta\":{},\"finish_reason\":\"stop\"}],\"usage\":null}\n\n",
    "data: {\"id\":\"c1\",\"object\":\"chat.completion.chunk\",\"created\":1,\"model\":\"upstream-model\",\"choices\":[],\"usage\":{\"prompt_tokens\":100,\"completion_tokens\":20,\"total_tokens\":120,\"prompt_tokens_details\":{\"cached_tokens\":80},\"completion_tokens_details\":{\"reasoning_tokens\":12}}}\n\n",
    "data: [DONE]\n\n",
);

fn chat_body(group: &str) -> String {
    serde_json::json!({
        "model": group,
        "messages": [{ "role": "user", "content": "failover effectiveness probe" }],
        "stream": false
    })
    .to_string()
}

fn chat_request(group: &str) -> Request<Body> {
    Request::post("/v1/chat/completions")
        .header("content-type", "application/json")
        .body(Body::from(chat_body(group)))
        .unwrap()
}

fn provider_json(name: &str, base_url: &str, ttfb_timeout_seconds: Option<u64>) -> serde_json::Value {
    let mut provider = serde_json::json!({
        "name": name,
        "type": "openai",
        "base_url": base_url,
    });
    if let Some(ttfb) = ttfb_timeout_seconds {
        provider["ttfb_timeout_seconds"] = serde_json::json!(ttfb);
    }
    provider
}

/// Two-provider scenario config (index 0 = primary at priority 1, index 1 =
/// fallback at priority 2) with zero backoff and no jitter so batches stay
/// wall-clock bounded. Retry/budget/classification mechanisms are the
/// defaults; only the backoff pacing is neutralized.
fn scenario_config(group: &str, providers: &[serde_json::Value]) -> Config {
    let models: Vec<serde_json::Value> = providers
        .iter()
        .enumerate()
        .map(|(index, provider)| {
            serde_json::json!({
                "provider": provider["name"],
                "model": "gpt-4",
                "priority": index + 1,
            })
        })
        .collect();
    let mut config: Config = serde_json::from_value(serde_json::json!({
        "server": { "host": "127.0.0.1", "port": 0 },
        "providers": providers,
        "model_groups": [{ "name": group, "models": models }],
    }))
    .expect("failover-effectiveness scenario configuration should deserialize");
 config.retry.max_retries_per_provider = 1;
 config.retry.backoff_sequence_seconds = vec![0];
 config.retry.jitter_enabled = false;
 // R6: the 30 scenario requests are byte-identical by design, so two
 // default-on request-affinity mechanisms would bypass the primary
 // after the first failover and leave its mock mostly unexercised,
 // making exact hit expectations impossible:
 // - the tier-1 exact-match cache would serve requests 2..=30 from
 //   cache without routing;
 // - reasoning_compat's conversation-model affinity (default on)
 //   with the default 300 s stickiness TTL would promote the
 //   fallback to candidate 0 for the shared conversation prefix.
 // Every request must traverse the primary failure.
 config.exact_cache.enabled = false;
 config.cache_aware_routing.stickiness_ttl_seconds = 0;
 common::isolate_databases(&mut config);
 config
}

async fn scenario_app(config: Config) -> axum::Router {
    let server = GatewayServer::new(config, None).await.unwrap();
    server.build_router()
}

/// Healthy upstream on the buffered-dispatch path: SSE turn the gateway
/// reassembles. Mounted without an expectation so it serves unlimited hits.
async fn mount_healthy_upstream(server: &MockServer) {
    Mock::given(method("POST"))
        .and(path("/v1/chat/completions"))
        .respond_with(
            ResponseTemplate::new(200)
                .insert_header("content-type", "text/event-stream")
                .set_body_string(STREAMED_TURN_SSE),
        )
        .mount(server)
        .await;
}

/// Failing primary mock with an exact hit expectation (R6): a router
/// that never dispatches to the primary would otherwise pass the 99%
/// effectiveness bound unexercised.
///
/// Expectation math (config: `max_retries_per_provider = 1`, default
/// circuit breaker threshold 3 with a 5 s first backoff, hint-less
/// failures): rate-limit-class failures never erode the breaker and
/// cool the provider down for 30 s on the first hit, so a 429 batch
/// stops after 1 hit; 503 fails over without a same-provider retry,
/// so its batch stops when the breaker opens (3 hits); the in-provider
/// retryable classes (500/502/408) hit twice per request until the
/// breaker opens on the 3rd request (6 hits).
async fn mount_failing_upstream(server: &MockServer, status: u16, expected_hits: u64) {
 Mock::given(method("POST"))
 .and(path("/v1/chat/completions"))
 .respond_with(ResponseTemplate::new(status).set_body_string("primary is failing"))
 .expect(expected_hits)
 .mount(server)
 .await;
}

/// Exact primary-hit counts per failure class for
/// [`mount_failing_upstream`] (R6): requests until the circuit breaker
/// opens (threshold 3, 5 s first backoff outlasting the batch) times
/// hits per request (2 for the in-provider retryable classes, 1 for
/// the immediate-failover classes). 429s additionally take a 30 s
/// rate-limit cooldown after the first hit, so a hint-less batch sees
/// exactly 1.
fn expected_primary_hits(status: u16) -> u64 {
 match status {
 429 => 1,
 500 | 502 | 408 => 2 * 3,
 _ => 3,
 }
}

/// A loopback URI nobody listens on: bind an ephemeral port, then drop the
/// listener so dialing it yields connection-refused.
fn closed_port_uri() -> String {
    let listener = TcpListener::bind("127.0.0.1:0").unwrap();
    let port = listener.local_addr().unwrap().port();
    drop(listener);
    format!("http://127.0.0.1:{port}")
}

/// Run one scenario batch and count client-visible successes (HTTP 200).
async fn run_scenario(label: &str, app: axum::Router) -> usize {
    let mut successes = 0usize;
    for index in 0..SCENARIO_REQUESTS {
        let svc = app.clone();
        let req = chat_request(EFFECTIVENESS_GROUP);
        let resp = match tokio::time::timeout(PER_REQUEST_TIMEOUT, svc.oneshot(req)).await {
            Ok(result) => result.expect("oneshot routing must not panic"),
            Err(_) => {
                eprintln!(
                    "failover_effectiveness[{label}]: request {} exceeded {:?}",
                    index + 1,
                    PER_REQUEST_TIMEOUT
                );
                continue;
            }
        };
        let status = resp.status();
        let bytes = axum::body::to_bytes(resp.into_body(), 1024 * 1024)
            .await
            .unwrap();
        if status == StatusCode::OK {
            successes += 1;
        } else {
            eprintln!(
                "failover_effectiveness[{label}]: request {} failed with {} body={}",
                index + 1,
                status,
                String::from_utf8_lossy(&bytes)
            );
        }
    }
    eprintln!(
        "failover_effectiveness[{label}]: {}/{} client-visible successes",
        successes, SCENARIO_REQUESTS
    );
    successes
}

fn assert_scenario_success(label: &str, successes: usize) {
    let required = (0.99f64 * SCENARIO_REQUESTS as f64).floor() as usize;
    assert!(
        successes >= required,
        "failover_effectiveness[{label}]: {successes}/{SCENARIO_REQUESTS} client-visible successes, below the 99% bound ({required} required)"
    );
}

async fn with_scenario_timeout<F>(label: &str, future: F) -> usize
where
    F: std::future::Future<Output = usize>,
{
    match tokio::time::timeout(SCENARIO_TIMEOUT, future).await {
        Ok(successes) => successes,
        Err(_) => panic!(
            "failover_effectiveness[{label}] exceeded the {:?} scenario budget",
            SCENARIO_TIMEOUT
        ),
    }
}

/// HTTP-status scenarios (500/502/503/429/408): primary always answers with
/// the failure class, fallback serves every re-routed request. R6: the
/// primary mock carries an exact hit expectation and is `verify()`ed, so a
/// router that never dispatches to the primary cannot pass unexercised.
async fn run_http_status_scenario(status: u16) -> usize {
 let primary = MockServer::start().await;
 mount_failing_upstream(&primary, status, expected_primary_hits(status)).await;
 let fallback = MockServer::start().await;
 mount_healthy_upstream(&fallback).await;
 let app = scenario_app(scenario_config(
 EFFECTIVENESS_GROUP,
 &[
 provider_json("primary", &primary.uri(), None),
 provider_json("fallback", &fallback.uri(), None),
 ],
 ))
 .await;
 let successes = run_scenario(&format!("primary_{status}"), app).await;
 primary.verify().await;
 successes
}

fn median(samples: &[Duration]) -> Duration {
    let mut sorted = samples.to_vec();
    sorted.sort();
    sorted[sorted.len() / 2]
}

// ---------------------------------------------------------------------------
// Failover-effectiveness scenarios (design A8: >= 99% per retryable class)
// ---------------------------------------------------------------------------

/// 500 is retryable in-provider, then fails over to the healthy fallback.
#[tokio::test]
async fn failover_effectiveness_primary_500_recovers_via_fallback() {
    let successes =
        with_scenario_timeout("primary_500", run_http_status_scenario(500)).await;
    assert_scenario_success("primary_500", successes);
}

/// 502 is a retryable upstream class; fallback serves the re-routed batch.
#[tokio::test]
async fn failover_effectiveness_primary_502_recovers_via_fallback() {
    let successes =
        with_scenario_timeout("primary_502", run_http_status_scenario(502)).await;
    assert_scenario_success("primary_502", successes);
}

/// 503 is a retryable upstream class; fallback serves the re-routed batch.
#[tokio::test]
async fn failover_effectiveness_primary_503_recovers_via_fallback() {
    let successes =
        with_scenario_timeout("primary_503", run_http_status_scenario(503)).await;
    assert_scenario_success("primary_503", successes);
}

/// 429 with the default `upstream_retry_after_max_wait_ms: 0` fails over
/// immediately to the fallback (DD-C upgrade-preserving default).
#[tokio::test]
async fn failover_effectiveness_primary_429_recovers_via_fallback() {
    let successes =
        with_scenario_timeout("primary_429", run_http_status_scenario(429)).await;
    assert_scenario_success("primary_429", successes);
}

/// 408 is the in-provider retry class; after the retry it fails over.
#[tokio::test]
async fn failover_effectiveness_primary_408_recovers_via_fallback() {
    let successes =
        with_scenario_timeout("primary_408", run_http_status_scenario(408)).await;
    assert_scenario_success("primary_408", successes);
}

/// Network-refused: primary dials a loopback port nobody listens on, so
/// every attempt is a connection error; the fallback serves the batch.
#[tokio::test]
async fn failover_effectiveness_primary_connection_refused_recovers_via_fallback() {
    let successes = with_scenario_timeout("primary_connection_refused", async {
        let dead_upstream = closed_port_uri();
        let fallback = MockServer::start().await;
        mount_healthy_upstream(&fallback).await;
        let app = scenario_app(scenario_config(
            EFFECTIVENESS_GROUP,
            &[
                provider_json("primary", &dead_upstream, None),
                provider_json("fallback", &fallback.uri(), None),
            ],
        ))
        .await;
        run_scenario("primary_connection_refused", app).await
    })
    .await;
    assert_scenario_success("primary_connection_refused", successes);
}

/// Timeout-class (TTFB): primary's response is delayed 3 s against a 1 s
/// TTFB budget, so the first byte never arrives; the fallback serves the
/// re-routed requests without a same-provider retry (ADR 0001). R6: exact
/// primary expectation — 1 hit per request until the breaker opens
/// (threshold 3; its 5 s first backoff outlasts the batch).
#[tokio::test]
async fn failover_effectiveness_primary_ttfb_timeout_recovers_via_fallback() {
 let successes = with_scenario_timeout("primary_ttfb_timeout", async {
 let primary = MockServer::start().await;
 Mock::given(method("POST"))
 .and(path("/v1/chat/completions"))
 .respond_with(
 ResponseTemplate::new(200)
 .insert_header("content-type", "text/event-stream")
 .set_body_string(STREAMED_TURN_SSE)
 .set_delay(Duration::from_secs(3)),
 )
 // Timeout class fails over without a same-provider retry;
 // the breaker opens after 3 requests.
 .expect(3)
 .mount(&primary)
 .await;
 let fallback = MockServer::start().await;
 mount_healthy_upstream(&fallback).await;
 let app = scenario_app(scenario_config(
 EFFECTIVENESS_GROUP,
 &[
 provider_json("primary", &primary.uri(), Some(1)),
 provider_json("fallback", &fallback.uri(), None),
 ],
 ))
 .await;
 let successes = run_scenario("primary_ttfb_timeout", app).await;
 primary.verify().await;
 successes
 })
 .await;
 assert_scenario_success("primary_ttfb_timeout", successes);
}

// ---------------------------------------------------------------------------
// Performance budget (design A8): median gateway overhead < 10 ms on the
// buffered dispatch path, upstream latency excluded. Wall-clock assertion,
// so it is #[ignore]-d like performance.rs's latency budgets.
// ---------------------------------------------------------------------------

#[tokio::test]
#[ignore = "wall-clock overhead budget (A8): run perf suite with --ignored"]
async fn failover_effectiveness_overhead_median_under_10ms() {
    let result = tokio::time::timeout(OVERHEAD_TIMEOUT, async {
        let upstream = MockServer::start().await;
        mount_healthy_upstream(&upstream).await;
        let app = scenario_app(scenario_config(
            OVERHEAD_GROUP,
            &[provider_json("overhead-primary", &upstream.uri(), None)],
        ))
        .await;

        // The measured path carries the A8 mechanisms by default: budget
        // admission, error classification, and the attempt-count header.
        let resp = app
            .clone()
            .oneshot(chat_request(OVERHEAD_GROUP))
            .await
            .unwrap();
        assert_eq!(resp.status(), StatusCode::OK);
        assert_eq!(
            resp.headers()
                .get("x-obey-attempt-count")
                .expect("measured path emits x-obey-attempt-count")
                .to_str()
                .unwrap(),
            "1"
        );
        axum::body::to_bytes(resp.into_body(), 1024 * 1024)
            .await
            .unwrap();

        for _ in 0..OVERHEAD_WARMUP {
            let svc = app.clone();
            let resp = svc.oneshot(chat_request(OVERHEAD_GROUP)).await.unwrap();
            assert_eq!(resp.status(), StatusCode::OK);
            axum::body::to_bytes(resp.into_body(), 1024 * 1024)
                .await
                .unwrap();
        }

        let mut gateway_samples = Vec::with_capacity(OVERHEAD_ITERATIONS);
        for _ in 0..OVERHEAD_ITERATIONS {
            let svc = app.clone();
            let req = chat_request(OVERHEAD_GROUP);
            let start = Instant::now();
            let resp = svc.oneshot(req).await.unwrap();
            assert_eq!(resp.status(), StatusCode::OK);
            axum::body::to_bytes(resp.into_body(), 1024 * 1024)
                .await
                .unwrap();
            gateway_samples.push(start.elapsed());
        }

        // Upstream floor: direct loopback round trips to the same mock with
        // the same request body, reusing one client like the gateway's pool.
        // Subtracting the floor median from the gateway median isolates
        // gateway processing (routing, budget, reassembly, header, logging).
        let client = reqwest::Client::builder().no_proxy().build().unwrap();
        let url = format!("{}/v1/chat/completions", upstream.uri());
        let body = chat_body(OVERHEAD_GROUP);
        for _ in 0..OVERHEAD_WARMUP {
            let resp = client
                .post(&url)
                .header("content-type", "application/json")
                .body(body.clone())
                .send()
                .await
                .unwrap();
            let _ = resp.text().await.unwrap();
        }
        let mut floor_samples = Vec::with_capacity(OVERHEAD_ITERATIONS);
        for _ in 0..OVERHEAD_ITERATIONS {
            let start = Instant::now();
            let resp = client
                .post(&url)
                .header("content-type", "application/json")
                .body(body.clone())
                .send()
                .await
                .unwrap();
            let _ = resp.text().await.unwrap();
            floor_samples.push(start.elapsed());
        }

        let gateway_median = median(&gateway_samples);
        let floor_median = median(&floor_samples);
        let overhead = gateway_median.saturating_sub(floor_median);
        eprintln!(
            "failover_effectiveness_overhead: gateway median={:?}, upstream floor median={:?}, overhead={:?}",
            gateway_median, floor_median, overhead
        );
        assert!(
            overhead < Duration::from_millis(10),
            "median gateway overhead {overhead:?} exceeds the 10 ms budget (A8), gateway median={gateway_median:?} floor={floor_median:?}"
        );
    })
    .await;
    assert!(result.is_ok(), "overhead benchmark exceeded its budget");
}
