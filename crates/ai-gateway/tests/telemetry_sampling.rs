mod common;

use std::time::Duration;

use ai_gateway::config::Config;
use ai_gateway::gateway::GatewayServer;
use axum::body::Body;
use axum::http::{Request, StatusCode};
use serde_json::json;
use tower::ServiceExt;
use wiremock::matchers::{method, path};
use wiremock::{Mock, MockServer, ResponseTemplate};

const REQUEST_ID: &str = "123e4567-e89b-42d3-a456-426614174000";

fn completion_body() -> serde_json::Value {
    json!({
        "id": "chatcmpl-mock", "object": "chat.completion", "created": 1_700_000_000_i64,
        "model": "gpt-4", "choices": [{"index": 0, "message": {"role": "assistant", "content": "ok"}, "finish_reason": "stop"}],
        "usage": {"prompt_tokens": 1, "completion_tokens": 1, "total_tokens": 2}
    })
}

async fn mocks() -> (MockServer, MockServer) {
    let collector = MockServer::start().await;
    Mock::given(method("POST")).and(path("/v1/traces"))
        .respond_with(ResponseTemplate::new(200)).mount(&collector).await;
    let upstream = MockServer::start().await;
    Mock::given(method("POST")).and(path("/v1/chat/completions"))
        .respond_with(ResponseTemplate::new(200).set_body_json(completion_body()))
        .mount(&upstream).await;
    (collector, upstream)
}

fn config(collector: &MockServer, upstream: &MockServer, ratio: f64, port: u16) -> Config {
 let mut config: Config = serde_json::from_value(json!({
 "server": {"host": "127.0.0.1", "port": port},
        "providers": [{"name": "provider", "type": "openai", "base_url": upstream.uri()}],
        "model_groups": [{"name": "telemetry-group", "models": [{"provider": "provider", "model": "gpt-4", "priority": 1}]}],
        "opentelemetry": {"enabled": true, "endpoint": format!("{}/v1/traces", collector.uri()), "sampling_ratio": ratio}
    })).expect("test configuration should deserialize");
    config.retry.max_retries_per_provider = 0;
    config.retry.backoff_sequence_seconds = vec![0];
    config.retry.jitter_enabled = false;
    common::isolate_databases(&mut config);
    config
}

fn chat_request(request_id: &str) -> Request<Body> {
 // Unique content per request: identical bodies would be served by the
 // tier-1 exact cache, which bypasses the provider-dispatch telemetry seam.
 Request::post("/v1/chat/completions").header("content-type", "application/json")
 .header("x-request-id", request_id)
 .body(Body::from(json!({"model": "telemetry-group", "messages": [{"role": "user", "content": format!("hello {request_id}")}], "stream": false}).to_string()))
 .unwrap()
}

async fn send(server: &GatewayServer, request_id: &str) {
    let response = server.build_router().oneshot(chat_request(request_id)).await.unwrap();
    assert_eq!(response.status(), StatusCode::OK, "mock-backed chat request should succeed");
}

async fn wait_for_count(collector: &MockServer, expected: usize) -> usize {
    let deadline = tokio::time::Instant::now() + Duration::from_secs(10);
    loop {
        let count = collector.received_requests().await.unwrap().len();
        if count >= expected || tokio::time::Instant::now() >= deadline { return count; }
        tokio::time::sleep(Duration::from_millis(50)).await;
    }
}

#[tokio::test]
async fn ratio_0_5_deterministic_for_repeated_uuid_request_id() {
    let (collector, upstream) = mocks().await;
    let server = GatewayServer::new(config(&collector, &upstream, 0.5, 0), None).await.unwrap();

    for _ in 0..4 { send(&server, REQUEST_ID).await; }
    tokio::time::sleep(Duration::from_millis(1500)).await;

    let count = collector.received_requests().await.unwrap().len();
    assert!(count == 0 || count == 4, "the same UUID must make one deterministic sampling decision; got {count} exports");
}

#[tokio::test]
async fn ratio_0_then_hot_reload_to_1_exports() {
 let (collector, upstream) = mocks().await;
 let initial = config(&collector, &upstream, 0.0, 8080);
 let config_dir = tempfile::tempdir().expect("temp config dir").keep();
 let config_path = config_dir.join("config.yaml");
 let server = GatewayServer::new(initial, Some(config_path)).await.unwrap();
 let app = server.build_router();

 send(&server, "123e4567-e89b-42d3-a456-426614174001").await;
 send(&server, "123e4567-e89b-42d3-a456-426614174002").await;
 tokio::time::sleep(Duration::from_millis(1500)).await;
 assert_eq!(collector.received_requests().await.unwrap().len(), 0, "ratio 0 must suppress exports before reload");

 // Hot-reload via the standard admin config flow: GET the live body,
 // flip the ratio, PUT it back (Req 9.1 path).
 let get_response = app
 .clone()
 .oneshot(Request::get("/admin/config").body(Body::empty()).unwrap())
 .await
 .unwrap();
 assert_eq!(get_response.status(), StatusCode::OK);
 let mut body: serde_json::Value =
 serde_json::from_slice(&axum::body::to_bytes(get_response.into_body(), 1024 * 1024).await.unwrap()).unwrap();
 body["opentelemetry"]["sampling_ratio"] = json!(1.0);
 let put_response = app
 .clone()
 .oneshot(
 Request::put("/admin/config")
 .header("content-type", "application/json")
 .body(Body::from(body.to_string()))
 .unwrap(),
 )
 .await
 .unwrap();
 assert_eq!(put_response.status(), StatusCode::OK, "config hot reload should accept ratio 1.0");

 send(&server, "123e4567-e89b-42d3-a456-426614174003").await;
 assert!(wait_for_count(&collector, 1).await >= 1, "a request after hot reload to ratio 1 must export");
}

#[tokio::test]
async fn ratio_1_all_requests_exported() {
    let (collector, upstream) = mocks().await;
    let server = GatewayServer::new(config(&collector, &upstream, 1.0, 0), None).await.unwrap();

 for request_id in [
 "123e4567-e89b-42d3-a456-426614174011",
 "123e4567-e89b-42d3-a456-426614174012",
 "123e4567-e89b-42d3-a456-426614174013",
 ] { send(&server, request_id).await; }
 
 // Exporter batches spans (≤64/batch, 5 s idle flush); poll until the
 // collector holds at least 3 spans across all OTLP posts.
 use flate2::read::GzDecoder;
 use std::io::Read;
 let deadline = tokio::time::Instant::now() + Duration::from_secs(10);
 let mut total_spans;
 loop {
 let requests = collector.received_requests().await.unwrap();
 let mut count = 0usize;
 for request in &requests {
 if request.url.path() != "/v1/traces" { continue; }
 let mut decoder = GzDecoder::new(request.body.as_slice());
 let mut decoded = Vec::new();
 if decoder.read_to_end(&mut decoded).is_ok() {
 if let Ok(payload) = serde_json::from_slice::<serde_json::Value>(&decoded) {
 count += payload["resourceSpans"][0]["scopeSpans"][0]["spans"]
 .as_array().map(|a| a.len()).unwrap_or(0);
 }
 }
 }
 total_spans = count;
 if total_spans >= 3 || tokio::time::Instant::now() >= deadline { break; }
 tokio::time::sleep(Duration::from_millis(100)).await;
 }
 assert!(total_spans >= 3, "ratio 1 must export spans for every request; got {total_spans}");
}
