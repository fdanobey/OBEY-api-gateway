mod common;

use std::time::Duration;

use ai_gateway::config::{Config, StreamingConfig};
use ai_gateway::gateway::GatewayServer;
use axum::body::Body;
use axum::http::{Request, StatusCode};
use flate2::read::GzDecoder;
use futures::StreamExt;
use serde_json::{json, Value};
use std::io::Read as _;
use tower::ServiceExt;
use wiremock::matchers::{method, path};
use wiremock::{Mock, MockServer, ResponseTemplate};

fn completion_body() -> Value {
    json!({"id":"chatcmpl-mock","object":"chat.completion","created":1700000000_i64,"model":"gpt-4","choices":[{"index":0,"message":{"role":"assistant","content":"ok"},"finish_reason":"stop"}],"usage":{"prompt_tokens":1,"completion_tokens":1,"total_tokens":2}})
}

fn config(upstream: &str, endpoint: &str) -> Config {
    let mut config: Config = serde_json::from_value(json!({
        "server":{"host":"127.0.0.1","port":0},
        "providers":[{"name":"provider","type":"openai","base_url":upstream}],
        "model_groups":[{"name":"telemetry-group","models":[{"provider":"provider","model":"gpt-4","priority":1}]}],
 "opentelemetry":{"enabled":true,"endpoint":endpoint,"sampling_ratio":1.0},
 "prometheus":{"enabled":true,"path":"/metrics"}
 })).expect("test configuration should deserialize");
    config.retry.max_retries_per_provider = 0;
    config.retry.backoff_sequence_seconds = vec![0];
    config.retry.jitter_enabled = false;
    common::isolate_databases(&mut config);
    config
}

fn chat_request(stream: bool) -> Request<Body> {
    Request::post("/v1/chat/completions").header("content-type","application/json")
        .body(Body::from(json!({"model":"telemetry-group","messages":[{"role":"user","content":"hello"}],"stream":stream}).to_string())).unwrap()
}

async fn upstream(delay: Option<Duration>) -> MockServer {
    let upstream = MockServer::start().await;
    let mut response = ResponseTemplate::new(200).set_body_json(completion_body());
    if let Some(delay) = delay { response = response.set_delay(delay); }
    Mock::given(method("POST")).and(path("/v1/chat/completions")).respond_with(response).mount(&upstream).await;
    upstream
}

async fn collector() -> MockServer {
    let collector = MockServer::start().await;
    Mock::given(method("POST")).and(path("/v1/traces")).respond_with(ResponseTemplate::new(200)).mount(&collector).await;
    collector
}

async fn metrics(server: &GatewayServer) -> String {
 let response = server.build_router().oneshot(Request::get("/metrics").body(Body::empty()).unwrap()).await.unwrap();
 assert_eq!(response.status(), StatusCode::OK, "metrics endpoint should remain available");
 String::from_utf8(axum::body::to_bytes(response.into_body(), 4 * 1024 * 1024).await.unwrap().to_vec()).unwrap()
}

async fn wait_for_metric(server: &GatewayServer, name: &str) -> String {
    let deadline = tokio::time::Instant::now() + Duration::from_secs(10);
    loop {
        let text = metrics(server).await;
        if text.contains(name) || tokio::time::Instant::now() >= deadline { return text; }
        tokio::time::sleep(Duration::from_millis(50)).await;
    }
}

fn spans(body: &[u8]) -> Vec<Value> {
    let mut decoder = GzDecoder::new(body);
    let mut out = String::new();
    decoder.read_to_string(&mut out).expect("collector body should be gzip");
    let otlp: Value = serde_json::from_str(&out).expect("collector body should be OTLP JSON");
    otlp["resourceSpans"].as_array().unwrap().iter()
        .flat_map(|r| r["scopeSpans"].as_array().unwrap())
        .flat_map(|s| s["spans"].as_array().unwrap().iter().cloned()).collect()
}

fn error_type(span: &Value) -> Option<&str> {
    span["attributes"].as_array()?.iter().find(|a| a["key"] == "error.type")?["value"]["stringValue"].as_str()
}

#[tokio::test]
async fn metrics_counter_increments_after_export() {
    let collector = collector().await;
    let upstream = upstream(None).await;
    let server = GatewayServer::new(config(&upstream.uri(), &format!("{}/v1/traces", collector.uri())), None).await.unwrap();

    let response = server.build_router().oneshot(chat_request(false)).await.unwrap();
    assert_eq!(response.status(), StatusCode::OK);

    let text = wait_for_metric(&server, "otlp_spans_exported_total").await;
    assert!(text.contains("otlp_spans_exported_total"), "successful export counter should be exposed: {text}");
}

#[tokio::test]
async fn unreachable_collector_fail_open() {
    let upstream = upstream(None).await;
    let server = GatewayServer::new(config(&upstream.uri(), "http://127.0.0.1:9/v1/traces"), None).await.unwrap();

    let response = server.build_router().oneshot(chat_request(false)).await.unwrap();
    assert_eq!(response.status(), StatusCode::OK, "collector failure must not fail the chat request");

    let text = wait_for_metric(&server, "otlp_export_failures_total").await;
    assert!(text.contains("otlp_export_failures_total"), "export failure counter should be exposed: {text}");
}

#[tokio::test]
async fn abandoned_stream_emits_terminal_root() {
    let collector = collector().await;
    let upstream = upstream(Some(Duration::from_secs(20))).await;
    let mut cfg = config(&upstream.uri(), &format!("{}/v1/traces", collector.uri()));
    cfg.streaming = Some(StreamingConfig { emit_early_event: true, passthrough_enabled: false, ..StreamingConfig::default() });
    let server = GatewayServer::new(cfg, None).await.unwrap();

    let response = server.build_router().oneshot(chat_request(true)).await.unwrap();
    assert_eq!(response.status(), StatusCode::OK);
    let mut body = response.into_body().into_data_stream();
    let first = tokio::time::timeout(Duration::from_secs(5), body.next()).await.expect("early SSE frame should arrive").expect("body should yield a frame").expect("frame should be readable");
    assert!(String::from_utf8_lossy(&first).contains("data:"));
    let reader = tokio::spawn(async move { while body.next().await.is_some() {} });
    tokio::time::sleep(Duration::from_millis(500)).await;
    reader.abort();
    let _ = reader.await;

    let deadline = tokio::time::Instant::now() + Duration::from_secs(10);
    loop {
        let requests = collector.received_requests().await.unwrap();
        let exported_spans: Vec<Value> = requests.iter().flat_map(|request| spans(&request.body)).collect();
        let found = exported_spans.iter().any(|span| {
            span["kind"].as_i64() == Some(2)
                && span["status"]["code"].as_i64() == Some(2)
                && span["attributes"].as_array().is_some_and(|attributes| attributes.iter().any(|attribute| {
                    attribute["key"] == "http.response.status_code"
                        && attribute["value"]["intValue"] == "499"
                }))
        }) && exported_spans.iter().any(|span| {
            error_type(span).is_some_and(|value| value.contains("client_disconnect"))
        });
        if found { break; }
        assert!(tokio::time::Instant::now() < deadline, "terminal root span with client_disconnect error.type was not exported; spans={exported_spans:#?}");
        tokio::time::sleep(Duration::from_millis(50)).await;
    }
}
