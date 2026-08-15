//! Integration tests for refusal-triggered failover re-dispatch
//! (spec `guardrail-pipelines`, task 17.8).
//!
//! These tests drive the full Axum router via `tower::ServiceExt::oneshot()`
//! (no port binding) to assert:
//!   * Real re-dispatch across the fallback ordering: first target refuses,
//!     second target responds normally (Req 12.5).
//!   * Circuit-breaker skipping: target with open CB is never dispatched to
//!     (Req 12.10).
//!   * Exhaustion: all targets refuse, last response is returned (Req 12.8).
//!   * Bounded attempts: each target attempted at most once (Req 12.7).

use axum::body::Body;
use axum::http::{Request, StatusCode};
use tower::ServiceExt;

use ai_gateway::config::Config;
use ai_gateway::gateway::GatewayServer;

use wiremock::matchers::{method as wm_method, path as wm_path};
use wiremock::{Mock, MockServer, ResponseTemplate};

mod common;

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

/// A refusal response: the assistant content matches the default refusal phrase
/// list ("I'm sorry" / "i cannot help with").
fn refusal_body(model: &str) -> serde_json::Value {
    serde_json::json!({
        "id": "chatcmpl-refusal",
        "object": "chat.completion",
        "created": 1_700_000_000_i64,
        "model": model,
        "choices": [{
            "index": 0,
            "message": { "role": "assistant", "content": "I'm sorry, I cannot help with that request." },
            "finish_reason": "stop"
        }],
        "usage": { "prompt_tokens": 10, "completion_tokens": 12, "total_tokens": 22 }
    })
}

/// A normal (non-refusal) completion response.
fn normal_body(model: &str, content: &str) -> serde_json::Value {
    serde_json::json!({
        "id": "chatcmpl-normal",
        "object": "chat.completion",
        "created": 1_700_000_000_i64,
        "model": model,
        "choices": [{
            "index": 0,
            "message": { "role": "assistant", "content": content },
            "finish_reason": "stop"
        }],
        "usage": { "prompt_tokens": 10, "completion_tokens": 8, "total_tokens": 18 }
    })
}

/// Start a mock OpenAI-compatible provider that always returns a refusal.
async fn start_refusing_provider() -> MockServer {
    let server = MockServer::start().await;
    Mock::given(wm_method("POST"))
        .and(wm_path("/v1/chat/completions"))
        .respond_with(ResponseTemplate::new(200).set_body_json(refusal_body("gpt-4")))
        .mount(&server)
        .await;
    server
}

/// Start a mock OpenAI-compatible provider that returns a normal response.
async fn start_normal_provider(content: &str) -> MockServer {
    let server = MockServer::start().await;
    Mock::given(wm_method("POST"))
        .and(wm_path("/v1/chat/completions"))
        .respond_with(ResponseTemplate::new(200).set_body_json(normal_body("gpt-4", content)))
        .mount(&server)
        .await;
    server
}

/// Build a Config (via YAML) with TWO providers in the same model group
/// (establishing the fallback ordering by priority), plus a guardrails section
/// that enables `failover_on_refusal` on the global default pipeline.
///
/// `provider_a_uri` gets priority 100 (tried first).
/// `provider_b_uri` gets priority 200 (fallback).
fn config_two_providers(provider_a_uri: &str, provider_b_uri: &str) -> Config {
    let yaml = format!(
        r#"
server:
  host: "127.0.0.1"
  port: 8080
  request_timeout_seconds: 30
  max_request_size_mb: 10
providers:
  - name: "provider-a"
    type: "openai"
    base_url: "{provider_a_uri}"
    timeout_seconds: 30
  - name: "provider-b"
    type: "openai"
    base_url: "{provider_b_uri}"
    timeout_seconds: 30
model_groups:
  - name: "test-group"
    version_fallback_enabled: false
    models:
      - provider: "provider-a"
        model: "gpt-4"
        priority: 100
      - provider: "provider-b"
        model: "gpt-4"
        priority: 200
retry:
  max_retries_per_provider: 0
  backoff_sequence_seconds: [1, 2, 4]
circuit_breaker:
  failure_threshold: 3
  backoff_sequence_seconds: [60, 120, 300]
guardrails:
  providers:
    - name: "noop"
      type: "regex"
      failure_policy: "fail_open"
      patterns:
        - name: "never-match"
          regex: "XYZZY_NEVER_MATCH_12345"
          entity: "NOOP"
          mode: "deny"
  pipelines:
    - name: "refusal-pipeline"
      failover_on_refusal: true
      stages:
        - name: "noop-scan"
          provider: "noop"
          phase: "post_call"
          action: "allow"
  global_default_pipeline: "refusal-pipeline"
"#
    );
    serde_yaml::from_str::<Config>(&yaml).expect("test config YAML should deserialize")
}

/// Build a Config with THREE providers in the same model group.
fn config_three_providers(uri_a: &str, uri_b: &str, uri_c: &str) -> Config {
    let yaml = format!(
        r#"
server:
  host: "127.0.0.1"
  port: 8080
  request_timeout_seconds: 30
  max_request_size_mb: 10
providers:
  - name: "provider-a"
    type: "openai"
    base_url: "{uri_a}"
    timeout_seconds: 30
  - name: "provider-b"
    type: "openai"
    base_url: "{uri_b}"
    timeout_seconds: 30
  - name: "provider-c"
    type: "openai"
    base_url: "{uri_c}"
    timeout_seconds: 30
model_groups:
  - name: "test-group"
    version_fallback_enabled: false
    models:
      - provider: "provider-a"
        model: "gpt-4"
        priority: 100
      - provider: "provider-b"
        model: "gpt-4"
        priority: 200
      - provider: "provider-c"
        model: "gpt-4"
        priority: 300
retry:
  max_retries_per_provider: 0
  backoff_sequence_seconds: [1, 2, 4]
circuit_breaker:
  failure_threshold: 3
  backoff_sequence_seconds: [60, 120, 300]
guardrails:
  providers:
    - name: "noop"
      type: "regex"
      failure_policy: "fail_open"
      patterns:
        - name: "never-match"
          regex: "XYZZY_NEVER_MATCH_12345"
          entity: "NOOP"
          mode: "deny"
  pipelines:
    - name: "refusal-pipeline"
      failover_on_refusal: true
      stages:
        - name: "noop-scan"
          provider: "noop"
          phase: "post_call"
          action: "allow"
  global_default_pipeline: "refusal-pipeline"
"#
    );
    serde_yaml::from_str::<Config>(&yaml).expect("test config YAML should deserialize")
}

async fn build_app(mut config: Config) -> (axum::Router, ai_gateway::gateway::AppState) {
    common::isolate_databases(&mut config);
    let server = GatewayServer::new(config, None).await.unwrap();
    let state = server.state.clone();
    (server.build_router(), state)
}

async fn send(app: axum::Router, req: Request<Body>) -> (StatusCode, Vec<u8>) {
    let resp = app.oneshot(req).await.unwrap();
    let status = resp.status();
    let body = axum::body::to_bytes(resp.into_body(), 4 * 1024 * 1024)
        .await
        .unwrap();
    (status, body.to_vec())
}

/// POST a non-streaming chat completion.
fn chat_request(user_content: &str) -> Request<Body> {
    let body = serde_json::json!({
        "model": "gpt-4",
        "messages": [{ "role": "user", "content": user_content }],
        "stream": false
    });
    Request::post("/v1/chat/completions")
        .header("content-type", "application/json")
        .body(Body::from(serde_json::to_string(&body).unwrap()))
        .unwrap()
}

fn assistant_content(body: &[u8]) -> String {
    let json: serde_json::Value = serde_json::from_slice(body).unwrap();
    json["choices"][0]["message"]["content"]
        .as_str()
        .unwrap_or_default()
        .to_string()
}

// ---------------------------------------------------------------------------
// 1. Real re-dispatch across the fallback ordering (Req 12.5)
// ---------------------------------------------------------------------------

/// Req 12.5: first target (provider-a) refuses, the gateway re-dispatches to
/// the second target (provider-b) which responds normally. The caller receives
/// provider-b's content.
#[tokio::test]
async fn failover_redispatches_to_next_provider_on_refusal() {
    let refusing = start_refusing_provider().await;
    let normal = start_normal_provider("Here is your answer from provider-b.").await;

    let cfg = config_two_providers(&refusing.uri(), &normal.uri());
    let (app, _state) = build_app(cfg).await;

    let (status, body) = send(app, chat_request("help me with something")).await;

    assert_eq!(
        status,
        StatusCode::OK,
        "should receive 200 from the fallback provider"
    );
    let content = assistant_content(&body);
    assert_eq!(
        content, "Here is your answer from provider-b.",
        "response must come from the fallback (non-refusing) provider"
    );

    // Verify both providers were called.
    let refusing_reqs = refusing.received_requests().await.unwrap();
    let normal_reqs = normal.received_requests().await.unwrap();
    assert_eq!(
        refusing_reqs.len(),
        1,
        "refusing provider should be called once (initial dispatch)"
    );
    assert_eq!(
        normal_reqs.len(),
        1,
        "normal provider should be called once (failover)"
    );
}

// ---------------------------------------------------------------------------
// 2. Circuit-breaker skipping (Req 12.10)
// ---------------------------------------------------------------------------

/// Req 12.10: provider-b's circuit breaker is open. The failover loop skips it
/// and goes directly to provider-c. Setup: three providers, provider-a refuses,
/// provider-b has its CB tripped open, provider-c responds normally.
#[tokio::test]
async fn failover_skips_provider_with_open_circuit_breaker() {
    let refusing = start_refusing_provider().await;
    let skipped = start_normal_provider("You should never see this.").await;
    let fallback = start_normal_provider("Answer from provider-c.").await;

    let cfg = config_three_providers(&refusing.uri(), &skipped.uri(), &fallback.uri());
    let (app, state) = build_app(cfg).await;

    // Trip provider-b's circuit breaker open (3 failures = threshold).
    let cb = state.router.get_circuit_breaker("provider-b:gpt-4").await;
    cb.record_failure().await;
    cb.record_failure().await;
    cb.record_failure().await;
    assert!(
        !cb.is_available().await,
        "provider-b CB should be open after 3 failures"
    );

    let (status, body) = send(app, chat_request("help me")).await;

    assert_eq!(status, StatusCode::OK);
    let content = assistant_content(&body);
    assert_eq!(
        content, "Answer from provider-c.",
        "failover must skip the open-CB provider-b and reach provider-c"
    );

    // Provider-b (skipped) must NOT have received any request.
    let skipped_reqs = skipped.received_requests().await.unwrap();
    assert_eq!(
        skipped_reqs.len(),
        0,
        "provider-b must be skipped entirely when its circuit breaker is open"
    );

    // Provider-a (initial refusal) and provider-c (successful fallback) each called once.
    let refusing_reqs = refusing.received_requests().await.unwrap();
    let fallback_reqs = fallback.received_requests().await.unwrap();
    assert_eq!(refusing_reqs.len(), 1);
    assert_eq!(fallback_reqs.len(), 1);
}

// ---------------------------------------------------------------------------
// 3. Exhaustion: all targets refuse, last response is returned (Req 12.8)
// ---------------------------------------------------------------------------

/// Req 12.8: when all providers in the fallback ordering refuse, the gateway
/// returns the LAST received response to the caller.
#[tokio::test]
async fn failover_exhaustion_returns_last_response() {
    let refusing_a = start_refusing_provider().await;
    let refusing_b = start_refusing_provider().await;

    let cfg = config_two_providers(&refusing_a.uri(), &refusing_b.uri());
    let (app, _state) = build_app(cfg).await;

    let (status, body) = send(app, chat_request("help me")).await;

    // Both providers refuse; the gateway returns the last response (200 with
    // refusal content from provider-b).
    assert_eq!(
        status,
        StatusCode::OK,
        "exhaustion returns HTTP 200 (the last response)"
    );
    let content = assistant_content(&body);
    assert!(
        content.contains("sorry") || content.contains("cannot"),
        "exhausted response should contain the refusal content; got: {content}"
    );

    // Both providers should have been called exactly once.
    let reqs_a = refusing_a.received_requests().await.unwrap();
    let reqs_b = refusing_b.received_requests().await.unwrap();
    assert_eq!(reqs_a.len(), 1, "provider-a called once (initial)");
    assert_eq!(reqs_b.len(), 1, "provider-b called once (failover)");
}

// ---------------------------------------------------------------------------
// 4. Bounded attempts: each target attempted at most once (Req 12.7)
// ---------------------------------------------------------------------------

/// Req 12.7: with three providers that all refuse, each is attempted exactly
/// once. The total number of upstream requests equals the number of configured
/// providers (3).
#[tokio::test]
async fn failover_attempts_each_target_at_most_once() {
    let refusing_a = start_refusing_provider().await;
    let refusing_b = start_refusing_provider().await;
    let refusing_c = start_refusing_provider().await;

    let cfg = config_three_providers(&refusing_a.uri(), &refusing_b.uri(), &refusing_c.uri());
    let (app, _state) = build_app(cfg).await;

    let (status, _body) = send(app, chat_request("help me")).await;

    assert_eq!(
        status,
        StatusCode::OK,
        "exhaustion returns the last response"
    );

    // Each provider called exactly once — proving the bound and no-retry guarantee.
    let reqs_a = refusing_a.received_requests().await.unwrap();
    let reqs_b = refusing_b.received_requests().await.unwrap();
    let reqs_c = refusing_c.received_requests().await.unwrap();
    assert_eq!(reqs_a.len(), 1, "provider-a called exactly once");
    assert_eq!(reqs_b.len(), 1, "provider-b called exactly once");
    assert_eq!(reqs_c.len(), 1, "provider-c called exactly once");
}

// ---------------------------------------------------------------------------
// 5. Disabled toggle: no failover when failover_on_refusal is false (Req 12.6)
// ---------------------------------------------------------------------------

/// Req 12.6: when `failover_on_refusal` is disabled (default), a refusal from
/// provider-a is returned directly without dispatching to provider-b.
#[tokio::test]
async fn no_failover_when_toggle_disabled() {
    let refusing = start_refusing_provider().await;
    let normal = start_normal_provider("Never reached.").await;

    // Build config WITHOUT failover_on_refusal enabled.
    let yaml = format!(
        r#"
server:
  host: "127.0.0.1"
  port: 8080
  request_timeout_seconds: 30
  max_request_size_mb: 10
providers:
  - name: "provider-a"
    type: "openai"
    base_url: "{}"
    timeout_seconds: 30
  - name: "provider-b"
    type: "openai"
    base_url: "{}"
    timeout_seconds: 30
model_groups:
  - name: "test-group"
    version_fallback_enabled: false
    models:
      - provider: "provider-a"
        model: "gpt-4"
        priority: 100
      - provider: "provider-b"
        model: "gpt-4"
        priority: 200
retry:
  max_retries_per_provider: 0
  backoff_sequence_seconds: [1, 2, 4]
guardrails:
  providers:
    - name: "noop"
      type: "regex"
      failure_policy: "fail_open"
      patterns:
        - name: "never-match"
          regex: "XYZZY_NEVER_MATCH_12345"
          entity: "NOOP"
          mode: "deny"
  pipelines:
    - name: "no-failover-pipeline"
      failover_on_refusal: false
      stages:
        - name: "noop-scan"
          provider: "noop"
          phase: "post_call"
          action: "allow"
  global_default_pipeline: "no-failover-pipeline"
"#,
        refusing.uri(),
        normal.uri()
    );
    let cfg: Config = serde_yaml::from_str(&yaml).expect("test config YAML should deserialize");
    let (app, _state) = build_app(cfg).await;

    let (status, body) = send(app, chat_request("help me")).await;

    assert_eq!(status, StatusCode::OK);
    let content = assistant_content(&body);
    assert!(
        content.contains("sorry") || content.contains("cannot"),
        "disabled toggle should return the refusal directly; got: {content}"
    );

    // Provider-b must NOT have been called (no failover).
    let normal_reqs = normal.received_requests().await.unwrap();
    assert_eq!(
        normal_reqs.len(),
        0,
        "provider-b must not be called when failover is disabled"
    );
}
