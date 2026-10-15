//! Request-outcome observability: every failed upstream try becomes an
//! `#attempt` log row, and a request the client abandons still leaves one
//! outcome row (499 `client_disconnect`) instead of vanishing from the log.

mod common;

use std::time::Duration;

use ai_gateway::config::{Config, StreamingConfig};
use ai_gateway::gateway::GatewayServer;
use ai_gateway::logger::LogEntry;
use axum::body::Body;
use axum::http::{Request, StatusCode};
use futures::StreamExt;
use serde_json::json;
use tower::ServiceExt;
use wiremock::matchers::{method, path};
use wiremock::{Mock, MockServer, ResponseTemplate};

const ATTEMPT_PATH: &str = "/v1/chat/completions#attempt";
const OUTCOME_PATH: &str = "/v1/chat/completions";

fn completion_body() -> serde_json::Value {
    json!({
        "id": "chatcmpl-mock",
        "object": "chat.completion",
        "created": 1_700_000_000_i64,
        "model": "gpt-4",
        "choices": [{
            "index": 0,
            "message": { "role": "assistant", "content": "Hello from mock provider" },
            "finish_reason": "stop"
        }],
        "usage": { "prompt_tokens": 10, "completion_tokens": 5, "total_tokens": 15 }
    })
}

/// One model group whose members are `(provider_name, base_url)` in priority order.
fn outcome_config(providers: &[(&str, String)], max_retries: u32) -> Config {
    let provider_values: Vec<serde_json::Value> = providers
        .iter()
        .map(|(name, url)| json!({ "name": name, "type": "openai", "base_url": url }))
        .collect();
    let models: Vec<serde_json::Value> = providers
        .iter()
        .enumerate()
        .map(|(index, (name, _))| json!({ "provider": name, "model": "gpt-4", "priority": index + 1 }))
        .collect();
    let mut config: Config = serde_json::from_value(json!({
        "server": { "host": "127.0.0.1", "port": 0 },
        "providers": provider_values,
        "model_groups": [{ "name": "outcome-group", "models": models }]
    }))
    .expect("test configuration should deserialize");
    config.retry.max_retries_per_provider = max_retries;
    config.retry.backoff_sequence_seconds = vec![0];
    config.retry.jitter_enabled = false;
    // Deterministic buffer-and-replay behind the early event.
    config.streaming = Some(StreamingConfig {
        emit_early_event: true,
        passthrough_enabled: false,
        ..StreamingConfig::default()
    });
    common::isolate_databases(&mut config);
    config
}

fn chat_request(stream: bool) -> Request<Body> {
    Request::post("/v1/chat/completions")
        .header("content-type", "application/json")
        .body(Body::from(
            json!({
                "model": "outcome-group",
                "messages": [{ "role": "user", "content": "hello" }],
                "stream": stream
            })
            .to_string(),
        ))
        .unwrap()
}

async fn logs(server: &GatewayServer) -> Vec<LogEntry> {
    let response = server
        .build_router()
        .oneshot(Request::get("/dashboard/logs").body(Body::empty()).unwrap())
        .await
        .unwrap();
    assert_eq!(response.status(), StatusCode::OK);
    let body = axum::body::to_bytes(response.into_body(), 4 * 1024 * 1024)
        .await
        .unwrap();
    serde_json::from_slice(&body).expect("dashboard logs should be LogEntry JSON")
}

/// Poll the log until `done` holds or `timeout` passes; returns the last read.
async fn poll_logs(
    server: &GatewayServer,
    timeout: Duration,
    done: impl Fn(&[LogEntry]) -> bool,
) -> Vec<LogEntry> {
    let deadline = tokio::time::Instant::now() + timeout;
    loop {
        let rows = logs(server).await;
        if done(&rows) || tokio::time::Instant::now() >= deadline {
            return rows;
        }
        tokio::time::sleep(Duration::from_millis(50)).await;
    }
}

#[tokio::test]
async fn abandoned_stream_writes_client_disconnect_outcome_row() {
    let upstream = MockServer::start().await;
    Mock::given(method("POST"))
        .and(path("/v1/chat/completions"))
        .respond_with(
            ResponseTemplate::new(200)
                .set_body_json(completion_body())
                .set_delay(Duration::from_secs(20)),
        )
        .mount(&upstream)
        .await;
    let server = GatewayServer::new(outcome_config(&[("slow-provider", upstream.uri())], 0), None)
        .await
        .unwrap();

    let response = server.build_router().oneshot(chat_request(true)).await.unwrap();
    assert_eq!(response.status(), StatusCode::OK);
    let mut body = response.into_body().into_data_stream();
    let first = tokio::time::timeout(Duration::from_secs(5), body.next())
        .await
        .expect("early SSE frame should arrive")
        .expect("body should yield a frame")
        .expect("frame should be readable");
    assert!(String::from_utf8_lossy(&first).contains("data:"));
    // Keep polling (the SSE body is lazy) so the request reaches the slow
    // upstream, then hang up by aborting the reader, which drops the body.
    let reader = tokio::spawn(async move { while body.next().await.is_some() {} });
    tokio::time::sleep(Duration::from_millis(500)).await;
    reader.abort();
    let _ = reader.await;

    let rows = poll_logs(&server, Duration::from_secs(5), |rows| {
        rows.iter().any(|row| row.status_code == 499)
    })
    .await;
    let abandoned: Vec<&LogEntry> = rows
        .iter()
        .filter(|row| row.path == OUTCOME_PATH)
        .collect();
    assert_eq!(abandoned.len(), 1, "exactly one outcome row: {rows:?}");
    let row = abandoned[0];
    assert_eq!(row.status_code, 499);
    assert_eq!(row.error_class.as_deref(), Some("client_disconnect"));
    assert_eq!(row.provider, "slow-provider");
    assert!(row.request_body.is_none());
}

#[tokio::test]
async fn completed_stream_does_not_write_abandon_row() {
    let upstream = MockServer::start().await;
    Mock::given(method("POST"))
        .and(path("/v1/chat/completions"))
        .respond_with(ResponseTemplate::new(200).set_body_json(completion_body()))
        .mount(&upstream)
        .await;
    let server = GatewayServer::new(outcome_config(&[("fast-provider", upstream.uri())], 0), None)
        .await
        .unwrap();

    let response = server.build_router().oneshot(chat_request(true)).await.unwrap();
    assert_eq!(response.status(), StatusCode::OK);
    let body = axum::body::to_bytes(response.into_body(), 1024 * 1024)
        .await
        .unwrap();
    assert!(String::from_utf8_lossy(&body).contains("[DONE]"));

    poll_logs(&server, Duration::from_secs(5), |rows| {
        rows.iter().any(|row| row.status_code == 200)
    })
    .await;
    // Let a (wrong) drop-time row land before asserting its absence.
    tokio::time::sleep(Duration::from_millis(300)).await;
    let rows = logs(&server).await;
    let outcomes: Vec<&LogEntry> = rows.iter().filter(|row| row.path == OUTCOME_PATH).collect();
    assert_eq!(outcomes.len(), 1, "one outcome row: {rows:?}");
    assert_eq!(outcomes[0].status_code, 200);
    assert!(rows.iter().all(|row| row.status_code != 499 && row.status_code != 504));
}

#[tokio::test]
async fn same_provider_retry_failure_is_logged_as_attempt_row() {
    let upstream = MockServer::start().await;
    Mock::given(method("POST"))
        .and(path("/v1/chat/completions"))
        .respond_with(
            ResponseTemplate::new(500)
                .set_body_string("upstream exploded")
                .set_delay(Duration::from_millis(50)),
        )
        .up_to_n_times(1)
        .mount(&upstream)
        .await;
    Mock::given(method("POST"))
        .and(path("/v1/chat/completions"))
        .respond_with(ResponseTemplate::new(200).set_body_json(completion_body()))
        .mount(&upstream)
        .await;
    let server = GatewayServer::new(outcome_config(&[("retry-provider", upstream.uri())], 1), None)
        .await
        .unwrap();

    let response = server.build_router().oneshot(chat_request(false)).await.unwrap();
    assert_eq!(response.status(), StatusCode::OK);
    let body = axum::body::to_bytes(response.into_body(), 1024 * 1024)
        .await
        .unwrap();
    let json: serde_json::Value = serde_json::from_slice(&body).unwrap();
    assert!(json.get("gateway_failed_attempts").is_none());

    let rows = poll_logs(&server, Duration::from_secs(5), |rows| rows.len() >= 2).await;
    let outcomes: Vec<&LogEntry> = rows.iter().filter(|row| row.path == OUTCOME_PATH).collect();
    let attempts: Vec<&LogEntry> = rows.iter().filter(|row| row.path == ATTEMPT_PATH).collect();
    assert_eq!(outcomes.len(), 1, "{rows:?}");
    assert_eq!(outcomes[0].status_code, 200);
    assert_eq!(attempts.len(), 1, "{rows:?}");
    assert_eq!(attempts[0].status_code, 500);
    assert_eq!(attempts[0].provider, "retry-provider");
    assert_eq!(attempts[0].error_class.as_deref(), Some("upstream_http_500"));
    assert!(attempts[0].duration_ms > 0, "{:?}", attempts[0]);
    assert_eq!(attempts[0].trace_id, outcomes[0].trace_id);
}

#[tokio::test]
async fn all_providers_failed_logs_attempt_rows() {
    let first = MockServer::start().await;
    let second = MockServer::start().await;
    for upstream in [&first, &second] {
        Mock::given(method("POST"))
            .and(path("/v1/chat/completions"))
            .respond_with(ResponseTemplate::new(500).set_body_string("down"))
            .mount(upstream)
            .await;
    }
    let server = GatewayServer::new(
        outcome_config(&[("first", first.uri()), ("second", second.uri())], 0),
        None,
    )
    .await
    .unwrap();

    let response = server.build_router().oneshot(chat_request(false)).await.unwrap();
    assert_eq!(response.status(), StatusCode::BAD_GATEWAY);
    let body = axum::body::to_bytes(response.into_body(), 1024 * 1024)
        .await
        .unwrap();
    // The client-visible attempt list keeps its original shape.
    let json: serde_json::Value = serde_json::from_slice(&body).unwrap();
    let attempts = json["error"]["attempts"].as_array().expect("attempts array");
    assert_eq!(attempts.len(), 2);
    for attempt in attempts {
        assert!(attempt.get("duration_ms").is_none());
        assert!(attempt.get("error_class").is_none());
    }

    let rows = poll_logs(&server, Duration::from_secs(5), |rows| rows.len() >= 3).await;
    let outcomes: Vec<&LogEntry> = rows.iter().filter(|row| row.path == OUTCOME_PATH).collect();
    let attempt_rows: Vec<&LogEntry> = rows.iter().filter(|row| row.path == ATTEMPT_PATH).collect();
    assert_eq!(outcomes.len(), 1, "{rows:?}");
    assert_eq!(outcomes[0].status_code, 502);
    assert_eq!(outcomes[0].error_class.as_deref(), Some("upstream_http_500"));
    assert_eq!(attempt_rows.len(), 2, "{rows:?}");
    let mut providers: Vec<&str> = attempt_rows.iter().map(|row| row.provider.as_str()).collect();
    providers.sort_unstable();
    assert_eq!(providers, vec!["first", "second"]);
    assert!(attempt_rows
        .iter()
        .all(|row| row.status_code == 500 && row.error_class.as_deref() == Some("upstream_http_500")));
}

// ─── Context-overflow observability (context-overflow spec, task 5.2) ──────

/// ~20 tokens per message: 700 messages ≈ 14k tokens — over the small
/// budget (8192-1024=7168), under the large budget (32768-1638=31130).
fn overflowing_messages() -> Vec<serde_json::Value> {
    (0..700)
        .map(|i| {
            serde_json::json!({
                "role": if i % 2 == 0 { "user" } else { "assistant" },
                "content": format!("OVERFLOW-MARKER turn {i} with padding text to fill the window.")
            })
        })
        .collect()
}

/// Two-provider config with proactive enforcement: `small` (window 8192,
/// priority 1) and `large` (window 32768, priority 2), reduction strategy
/// `remove_oldest`, and the given overflow policy.
fn overflow_outcome_config(policy: &str, small: &str, large: &str) -> Config {
    let mut config: Config = serde_json::from_value(serde_json::json!({
        "server": { "host": "127.0.0.1", "port": 0 },
        "providers": [
            { "name": "small", "type": "openai", "base_url": small },
            { "name": "large", "type": "openai", "base_url": large }
        ],
        "model_groups": [{
            "name": "outcome-group",
            "models": [
                { "provider": "small", "model": "gpt-4", "priority": 1, "context_window": 8192 },
                { "provider": "large", "model": "gpt-4", "priority": 2, "context_window": 32768 }
            ]
        }],
        "context": {
            "enabled": true,
            "proactive_enforcement": true,
            "overflow_policy": policy,
            "reduction_strategy": "remove_oldest"
        }
    }))
    .expect("overflow outcome config should deserialize");
    config.retry.max_retries_per_provider = 0;
    config.retry.backoff_sequence_seconds = vec![0];
    config.retry.jitter_enabled = false;
    config
}

#[tokio::test]
async fn reduce_policy_outcome_row_carries_context_reduction() {
    let small = MockServer::start().await;
    let large = MockServer::start().await;
    // The over-budget small provider serves the REDUCED copy successfully;
    // the outcome row must carry the reduction metadata of that dispatch.
    Mock::given(method("POST"))
        .and(path("/v1/chat/completions"))
        .respond_with(ResponseTemplate::new(200).set_body_json(completion_body()))
        .mount(&small)
        .await;
    Mock::given(method("POST"))
        .and(path("/v1/chat/completions"))
        .respond_with(ResponseTemplate::new(200).set_body_json(completion_body()))
        .mount(&large)
        .await;
    let mut config = overflow_outcome_config("reduce", &small.uri(), &large.uri());
    common::isolate_databases(&mut config);
    let server = GatewayServer::new(config, None).await.unwrap();

    let response = server
        .build_router()
        .oneshot(
            Request::post("/v1/chat/completions")
                .header("content-type", "application/json")
                .body(Body::from(
                    serde_json::json!({
                        "model": "outcome-group",
                        "messages": overflowing_messages()
                    })
                    .to_string(),
                ))
                .unwrap(),
        )
        .await
        .unwrap();
    assert_eq!(response.status(), StatusCode::OK);

    let rows = poll_logs(&server, Duration::from_secs(5), |rows| {
        rows.iter().any(|row| row.path == OUTCOME_PATH && row.context_reduction.is_some())
    })
    .await;

    let outcomes: Vec<&LogEntry> = rows
        .iter()
        .filter(|row| row.path == OUTCOME_PATH && row.status_code == 200)
        .collect();
    assert_eq!(outcomes.len(), 1, "{rows:?}");
    let outcome = outcomes[0];
    assert_eq!(outcome.provider, "small", "reduced small attempt served it");
    let reduction = outcome
        .context_reduction
        .as_ref()
        .expect("outcome row carries the reduction metadata");
    assert_eq!(reduction.strategy, "remove_oldest");
    assert!(!reduction.summarized);
    assert!(!reduction.fallback_from_summarization);
    // ~14k-token request reduced into the 7168-token small budget.
    assert!(reduction.original_tokens > 7_168, "{reduction:?}");
    assert!(reduction.final_tokens <= 7_168, "{reduction:?}");
    assert!(reduction.final_tokens < reduction.original_tokens);
    assert!(reduction.groups_removed > 0);
    assert_eq!(reduction.tool_results_cleared, 0);

    // Nothing failed: no #attempt rows for this request.
    let attempt_rows: Vec<&LogEntry> = rows
        .iter()
        .filter(|row| row.path == ATTEMPT_PATH && row.trace_id == outcome.trace_id)
        .collect();
    assert!(attempt_rows.is_empty(), "{rows:?}");
}

#[tokio::test]
async fn route_larger_context_skip_attempt_row_carries_context_window_exceeded() {
    let small = MockServer::start().await;
    let large = MockServer::start().await;
    for upstream in [&small, &large] {
        Mock::given(method("POST"))
            .and(path("/v1/chat/completions"))
            .respond_with(ResponseTemplate::new(200).set_body_json(completion_body()))
            .mount(upstream)
            .await;
    }
    let mut config = overflow_outcome_config(
        "route_larger_context",
        &small.uri(),
        &large.uri(),
    );
    common::isolate_databases(&mut config);
    let server = GatewayServer::new(config, None).await.unwrap();

    let response = server
        .build_router()
        .oneshot(
            Request::post("/v1/chat/completions")
                .header("content-type", "application/json")
                .body(Body::from(
                    serde_json::json!({
                        "model": "outcome-group",
                        "messages": overflowing_messages()
                    })
                    .to_string(),
                ))
                .unwrap(),
        )
        .await
        .unwrap();
    assert_eq!(response.status(), StatusCode::OK);

    let rows = poll_logs(&server, Duration::from_secs(5), |rows| {
        rows.iter().any(|row| row.path == ATTEMPT_PATH
            && row.error_class.as_deref() == Some("context_window_exceeded"))
            && rows.iter().any(|row| row.path == OUTCOME_PATH && row.status_code == 200)
    })
    .await;

    // Gate-excluded candidate: #attempt row with the normalized overflow
    // class and a 413 status, sharing the outcome's trace_id.
    let outcomes: Vec<&LogEntry> = rows
        .iter()
        .filter(|row| row.path == OUTCOME_PATH && row.status_code == 200)
        .collect();
    assert_eq!(outcomes.len(), 1, "{rows:?}");
    let skip_rows: Vec<&LogEntry> = rows
        .iter()
        .filter(|row| row.path == ATTEMPT_PATH && row.error_class.as_deref() == Some("context_window_exceeded"))
        .collect();
    assert_eq!(skip_rows.len(), 1, "{rows:?}");
    assert_eq!(skip_rows[0].provider, "small");
    assert_eq!(skip_rows[0].status_code, 413);
    assert_eq!(skip_rows[0].trace_id, outcomes[0].trace_id);
    // A gate skip is a routing decision, not a reduction: no metadata.
    assert!(skip_rows[0].context_reduction.is_none());
}
