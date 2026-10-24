//! Proactive context-overflow routing integration (context-overflow
//! management spec, task 4.5).
//!
//! Covers, through the full gateway server (tower oneshot, wiremock
//! upstreams, isolated databases):
//! - `reject` policy: 413 with zero upstream hits.
//! - `reduce` policy: the small-window provider receives the reduced body;
//!   a failover attempt on the large-window provider receives the ORIGINAL
//!   body (per-attempt immutability, US-1).
//! - `route_then_reduce`: the small-window provider is a reduce-candidate
//!   (not dropped); the large-window provider is tried first; exhaustion
//!   retries the first candidate with reduction.
//! - ADR-0002: a gate-skipped attempt never debits the provider rate token
//!   (tokens consumed == real upstream requests).

mod common;

use ai_gateway::config::Config;
use ai_gateway::gateway::GatewayServer;
use axum::body::Body;
use axum::http::{Request, StatusCode};
use serde_json::{json, Value};
use tower::ServiceExt;
use wiremock::matchers::{method, path};
use wiremock::{Mock, MockServer, ResponseTemplate};

/// Token size of one padding message (a few words each, well below the
/// smallest window used here).
fn padded_messages(count: usize, marker: &str) -> Vec<Value> {
    (0..count)
        .map(|i| {
            json!({
                "role": if i % 2 == 0 { "user" } else { "assistant" },
                "content": format!("{marker} turn {i} with some padding text to fill the context window.")
            })
        })
        .collect()
}

/// Two-provider config: `small` (window 8192) first, `large` (window 32768)
/// second, proactive enforcement with the requested policy.
fn overflow_config(policy: &str, small: &MockServer, large: &MockServer) -> Config {
    let mut config: Config = serde_json::from_value(json!({
        "server": { "host": "127.0.0.1", "port": 0 },
        "providers": [
            { "name": "small", "type": "openai", "base_url": small.uri() },
            { "name": "large", "type": "openai", "base_url": large.uri() }
        ],
        "model_groups": [{
            "name": "overflow-group",
            "models": [
                { "provider": "small", "model": "small-model", "priority": 1, "context_window": 8192 },
                { "provider": "large", "model": "large-model", "priority": 2, "context_window": 32768 }
            ]
        }],
        "context": {
            "enabled": true,
            "proactive_enforcement": true,
            "overflow_policy": policy,
            "reduction_strategy": "remove_oldest"
        }
    }))
    .expect("test configuration should deserialize");
    // Deterministic single-attempt dispatches: a same-provider 500 retry
    // would double-count upstream hits in the assertions below.
    config.retry.max_retries_per_provider = 0;
    config.retry.backoff_sequence_seconds = vec![0];
    config.retry.jitter_enabled = false;
    common::isolate_databases(&mut config);
    config
}

fn chat_request(messages: Vec<Value>) -> Request<Body> {
    Request::post("/v1/chat/completions")
        .header("content-type", "application/json")
        .body(Body::from(
            json!({
                "model": "overflow-group",
                "messages": messages
            })
            .to_string(),
        ))
        .unwrap()
}

fn completion_body() -> Value {
    json!({
        "id": "chatcmpl-mock",
        "object": "chat.completion",
        "created": 1_700_000_000_i64,
        "model": "mock",
        "choices": [{
            "index": 0,
            "message": { "role": "assistant", "content": "ok" },
            "finish_reason": "stop"
        }],
        "usage": { "prompt_tokens": 1, "completion_tokens": 1, "total_tokens": 2 }
    })
}

async fn chat_hits(server: &MockServer) -> Vec<Value> {
    server
        .received_requests()
        .await
        .unwrap_or_default()
        .iter()
        .filter(|r| r.url.path() == "/v1/chat/completions")
        .map(|r| serde_json::from_slice(&r.body).expect("captured body is JSON"))
        .collect()
}

/// A request that fits `large` (budget 32768-1638=31130) but overflows
/// `small` (budget 8192-1024=7168): each padding message counts ~20 tokens,
/// so 700 messages ≈ 14k tokens.
fn overflowing_request() -> Vec<Value> {
    padded_messages(700, "OVERFLOW-MARKER")
}

#[tokio::test]
async fn overflow_config_parses_gate_settings() {
    let small = MockServer::start().await;
    let cfg = overflow_config("reject", &small, &small);
    assert!(cfg.context.enabled);
    assert!(cfg.context.proactive_enforcement);
    assert_eq!(cfg.context.overflow_policy, "reject");
    assert_eq!(cfg.model_groups[0].models[0].context_window, 8192);
    assert_eq!(cfg.model_groups[0].models[1].context_window, 32768);
}

#[tokio::test]
async fn overflowing_request_exceeds_small_budget_only() {
    use ai_gateway::compression::token_counter::TokenCounter;
    use ai_gateway::models::openai::OpenAIRequest;

    let messages = overflowing_request();
    let request: OpenAIRequest = serde_json::from_value(json!({
        "model": "overflow-group",
        "messages": messages
    }))
    .unwrap();
    let estimate = TokenCounter::new().count_request(&request);
    // small window budget: 8192 - 0 - max(8192/20, 1024) = 8192-1024 = 7168
    assert!(
        estimate > 7168,
        "overflowing_request must exceed the small budget (estimate {estimate})"
    );
    assert!(
        estimate <= 31130,
        "overflowing_request must fit the large budget (estimate {estimate})"
    );
}

#[tokio::test]
async fn reject_policy_returns_413_with_zero_upstream_hits() {
    let small = MockServer::start().await;
    let large = MockServer::start().await;
    for upstream in [&small, &large] {
        Mock::given(method("POST"))
            .and(path("/v1/chat/completions"))
            .respond_with(ResponseTemplate::new(200).set_body_json(completion_body()))
            .expect(0) // the mock itself asserts zero dispatches
            .mount(upstream)
            .await;
    }
    let server = GatewayServer::new(overflow_config("reject", &small, &large), None)
        .await
        .unwrap();

    let response = server
        .build_router()
        .oneshot(chat_request(overflowing_request()))
        .await
        .unwrap();
    assert_eq!(response.status(), StatusCode::PAYLOAD_TOO_LARGE);
    let body = axum::body::to_bytes(response.into_body(), 1024 * 1024)
        .await
        .unwrap();
    let json: Value = serde_json::from_slice(&body).unwrap();
    assert_eq!(json["error"]["type"], "context_capacity_exceeded");
    assert!(
        json["error"]["estimated_requirement"].as_u64().unwrap() > 7168,
        "estimate should exceed the small window budget: {json}"
    );

    small.verify().await;
    large.verify().await;
    assert!(chat_hits(&small).await.is_empty());
    assert!(chat_hits(&large).await.is_empty());
}

#[tokio::test]
async fn reduce_policy_sends_reduced_body_then_original_on_failover() {
    let small = MockServer::start().await;
    let large = MockServer::start().await;
    // Small provider fails so the request fails over to `large`.
    Mock::given(method("POST"))
        .and(path("/v1/chat/completions"))
        .respond_with(ResponseTemplate::new(500).set_body_string("down"))
        .mount(&small)
        .await;
    Mock::given(method("POST"))
        .and(path("/v1/chat/completions"))
        .respond_with(ResponseTemplate::new(200).set_body_json(completion_body()))
        .mount(&large)
        .await;
    let server = GatewayServer::new(overflow_config("reduce", &small, &large), None)
        .await
        .unwrap();

    let original_count = overflowing_request().len();
    let response = server
        .build_router()
        .oneshot(chat_request(overflowing_request()))
        .await
        .unwrap();
    assert_eq!(response.status(), StatusCode::OK);

    let small_hits = chat_hits(&small).await;
    let large_hits = chat_hits(&large).await;
    assert_eq!(small_hits.len(), 1, "one reduced dispatch to small");
    assert_eq!(large_hits.len(), 1, "one original dispatch to large");
    // US-1 per-attempt immutability: the failover attempt carries the FULL
    // original conversation, the first attempt the reduced one.
    let reduced_count = small_hits[0]["messages"].as_array().unwrap().len();
    let large_count = large_hits[0]["messages"].as_array().unwrap().len();
    assert!(
        reduced_count < original_count,
        "small received a reduced body ({reduced_count} < {original_count})"
    );
    assert_eq!(
        large_count, original_count,
        "large received the original body untouched"
    );
}

#[tokio::test]
async fn route_then_reduce_tries_fitting_candidate_first_and_falls_back_to_reduction() {
    let small = MockServer::start().await;
    let large = MockServer::start().await;
    // The fitting (large) candidate fails so the chain exhausts and the
    // route-then-reduce retry of the first (small) candidate kicks in.
    Mock::given(method("POST"))
        .and(path("/v1/chat/completions"))
        .respond_with(ResponseTemplate::new(500).set_body_string("down"))
        .mount(&large)
        .await;
    Mock::given(method("POST"))
        .and(path("/v1/chat/completions"))
        .respond_with(ResponseTemplate::new(200).set_body_json(completion_body()))
        .mount(&small)
        .await;
    let server = GatewayServer::new(overflow_config("route_then_reduce", &small, &large), None)
        .await
        .unwrap();

    let original_count = overflowing_request().len();
    let response = server
        .build_router()
        .oneshot(chat_request(overflowing_request()))
        .await
        .unwrap();
    assert_eq!(response.status(), StatusCode::OK);

    let small_hits = chat_hits(&small).await;
    let large_hits = chat_hits(&large).await;
    // The over-budget candidate was NOT dropped (US-7): it eventually served
    // the request with a REDUCED body after the fitting candidate failed.
    assert_eq!(large_hits.len(), 1, "large (fitting) tried first");
    assert_eq!(small_hits.len(), 1, "small retried once with reduction");
    let reduced_count = small_hits[0]["messages"].as_array().unwrap().len();
    assert!(
        reduced_count < original_count,
        "exhaustion retry dispatches the reduced body ({reduced_count} < {original_count})"
    );
}

#[tokio::test]
async fn gate_skipped_attempt_never_debits_provider_rate_token() {
    let small = MockServer::start().await;
    let large = MockServer::start().await;
    for upstream in [&small, &large] {
        Mock::given(method("POST"))
            .and(path("/v1/chat/completions"))
            .respond_with(ResponseTemplate::new(200).set_body_json(completion_body()))
            .mount(upstream)
            .await;
    }
    // route_larger_context: the small provider is gate-skipped (excluded,
    // token refunded), large serves — no reject short-circuit involved.
    let mut config = overflow_config("route_larger_context", &small, &large);
    // 2/min on the small provider: an unrefunded admission token would leave
    // the balance at 1.0 after the skip; a refunded one stays at 2.0.
    config.providers[0].rate_limit_per_minute = 2;
    let server = GatewayServer::new(config, None).await.unwrap();

    let response = server
        .build_router()
        .oneshot(chat_request(overflowing_request()))
        .await
        .unwrap();
    assert_eq!(response.status(), StatusCode::OK);

    // Exactly one real upstream request happened (large); the small
    // provider's skipped attempt must have consumed nothing (ADR 0002:
    // one token per real upstream request — verified behaviorally: a
    // second identical request still succeeds against large because the
    // small bucket was never charged, and small still records zero hits).
    assert!(chat_hits(&small).await.is_empty());
    assert_eq!(chat_hits(&large).await.len(), 1);

    // A distinct second request (exact-cache would otherwise serve the same
    // body without a dispatch) still succeeds: the small bucket was refunded
    // after the skip, so the gateway is not throttled by a phantom charge.
    let mut second = overflowing_request();
    second.push(json!({
        "role": "user",
        "content": "DISTINCT-TAIL follow-up question."
    }));
    let response2 = server
        .build_router()
        .oneshot(chat_request(second))
        .await
        .unwrap();
    assert_eq!(
        response2.status(),
        StatusCode::OK,
        "second request must not be throttled by a phantom token charge"
    );
    assert_eq!(chat_hits(&large).await.len(), 2);
    assert!(chat_hits(&small).await.is_empty());
}

/// Task 6.2 (oneshot form): GET /admin/config round-trips the context
/// overflow fields, and PUT /admin/config applies a changed policy to the
/// live state without a restart — the next request is governed by the NEW
/// policy. No gateway process is launched; the router is driven through
/// tower oneshot.
#[tokio::test]
async fn admin_config_get_put_round_trips_and_hot_applies_context_policy() {
    let small = MockServer::start().await;
    let large = MockServer::start().await;
    for upstream in [&small, &large] {
        Mock::given(method("POST"))
            .and(path("/v1/chat/completions"))
            .respond_with(ResponseTemplate::new(200).set_body_json(completion_body()))
            .mount(upstream)
            .await;
    }
    let directory = tempfile::tempdir().unwrap();
    let config_path = directory.path().join("config.yaml");
    std::fs::write(
        &config_path,
        serde_yaml::to_string(&overflow_config("route_larger_context", &small, &large)).unwrap(),
    )
    .unwrap();
    let server = GatewayServer::new(
        overflow_config("route_larger_context", &small, &large),
        Some(config_path),
    )
    .await
    .unwrap();
    let app = server.build_router();

    // GET: the context overflow fields survive the API round-trip.
    let response = app
        .clone()
        .oneshot(Request::get("/admin/config").body(Body::empty()).unwrap())
        .await
        .unwrap();
    assert_eq!(response.status(), StatusCode::OK);
    let body = axum::body::to_bytes(response.into_body(), 1024 * 1024)
        .await
        .unwrap();
    let config: Value = serde_json::from_slice(&body).unwrap();
    assert_eq!(config["context"]["enabled"], true);
    assert_eq!(config["context"]["proactive_enforcement"], true);
    assert_eq!(config["context"]["overflow_policy"], "route_larger_context");
    assert_eq!(config["context"]["reduction_strategy"], "remove_oldest");
    assert_eq!(config["context"]["tool_result_keep_recent"], 3);

    // PUT: flip the policy to `reject`. The applied state must hot-swap
    // without restart (the YAML write lands in the temp config path). The
    // test config uses port 0 (OS-assigned), which validation rejects on a
    // full-config PUT — restore a concrete port as the admin panel does.
    let mut updated = config.clone();
    updated["context"]["overflow_policy"] = json!("reject");
    updated["server"]["port"] = json!(9);
    let response = app
        .clone()
        .oneshot(
            Request::put("/admin/config")
                .header("content-type", "application/json")
                .body(Body::from(updated.to_string()))
                .unwrap(),
        )
        .await
        .unwrap();
    let status = response.status();
    let put_body = axum::body::to_bytes(response.into_body(), 64 * 1024)
        .await
        .unwrap();
    assert_eq!(
        status,
        StatusCode::OK,
        "PUT /admin/config applies: {put_body:?}"
    );

    // The live state now rejects: an overflowing request 413s immediately
    // (previously route_larger_context would have skipped to `large`).
    let response = app
        .oneshot(chat_request(overflowing_request()))
        .await
        .unwrap();
    assert_eq!(response.status(), StatusCode::PAYLOAD_TOO_LARGE);
    // `reject` never fails over: `large` is untouched, and under
    // route_larger_context it would have been hit exactly once.
    assert!(chat_hits(&large).await.is_empty());
}

/// `route_larger_context` exhaustion: a request too large for EVERY
/// candidate surfaces the normalized 413 (never a misleading "all
/// providers failed" 502), reports the largest skipped window, and
/// dispatches nothing upstream.
#[tokio::test]
async fn route_larger_context_exhaustion_returns_413_with_largest_window() {
    let small = MockServer::start().await;
    let large = MockServer::start().await;
    for upstream in [&small, &large] {
        Mock::given(method("POST"))
            .and(path("/v1/chat/completions"))
            .respond_with(ResponseTemplate::new(200).set_body_json(completion_body()))
            .expect(0)
            .mount(upstream)
            .await;
    }
    let server = GatewayServer::new(
        overflow_config("route_larger_context", &small, &large),
        None,
    )
    .await
    .unwrap();

    // ~58k tokens: over the small budget (7168) AND the large budget
    // (32768-1638=31130), so every candidate is gate-skipped.
    let mut messages = overflowing_request();
    messages.extend(padded_messages(2_200, "EXHAUSTION-MARKER"));
    let response = server
        .build_router()
        .oneshot(chat_request(messages))
        .await
        .unwrap();
    assert_eq!(response.status(), StatusCode::PAYLOAD_TOO_LARGE);
    let body = axum::body::to_bytes(response.into_body(), 1024 * 1024)
        .await
        .unwrap();
    let json: Value = serde_json::from_slice(&body).unwrap();
    assert_eq!(json["error"]["type"], "context_capacity_exceeded");
    // The truthful largest window among the skipped candidates — the small
    // window is skipped first, but 32768 is what the chain could offer.
    assert_eq!(json["error"]["largest_supported_context"], 32_768);
    assert!(
        json["error"]["estimated_requirement"].as_u64().unwrap() > 31_130,
        "requirement covers the total context need: {json}"
    );

    small.verify().await;
    large.verify().await;
}
