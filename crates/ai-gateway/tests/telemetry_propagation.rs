//! Inbound W3C trace-context propagation tests (opentelemetry-trace-export
//! spec, task 8a; requirements 4 and 5.5, design "Integration Testing").
//!
//! Requests carry a `traceparent` header and the tests observe the OTLP
//! payload a wiremock mock collector actually receives: inbound trace ids,
//! parent span ids, sampled-flag override in both directions, invalid-header
//! fallback, and the disabled zero-export path.

mod common;

use std::time::Duration;

use ai_gateway::config::Config;
use ai_gateway::gateway::GatewayServer;
use axum::body::Body;
use axum::http::{Request, StatusCode};
use flate2::read::GzDecoder;
use serde_json::{json, Value};
use std::io::Read as _;
use tower::ServiceExt;
use wiremock::matchers::{method, path};
use wiremock::{Mock, MockServer, ResponseTemplate};

const TRACES_PATH: &str = "/v1/traces";

/// Fixed W3C context reused across tests: trace id + parent span id as hex.
const INBOUND_TRACE_ID: &str = "4bf92f3577b34da6a3ce929d0e0e4736";
const INBOUND_PARENT_SPAN_ID: &str = "00f067aa0ba902b7";

/// UUID request id whose 16 bytes become the OTLP trace id (Req 2.6).
const UUID_REQUEST_ID: &str = "123e4567-e89b-42d3-a456-426614174000";
const UUID_REQUEST_ID_HEX: &str = "123e4567e89b42d3a456426614174000";

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

async fn mount_collector() -> MockServer {
    let collector = MockServer::start().await;
    Mock::given(method("POST"))
        .and(path(TRACES_PATH))
        .respond_with(ResponseTemplate::new(200))
        .mount(&collector)
        .await;
    collector
}

async fn mount_upstream() -> MockServer {
    let upstream = MockServer::start().await;
    Mock::given(method("POST"))
        .and(path("/v1/chat/completions"))
        .respond_with(ResponseTemplate::new(200).set_body_json(completion_body()))
        .mount(&upstream)
        .await;
    upstream
}

/// Enabled telemetry config with the provider pointed at the live mock
/// upstream, exporter at the mock collector, and the given section overlay
/// (`sampling_ratio` is the only knob these tests vary).
async fn server_for(
    collector: &MockServer,
    upstream: &MockServer,
    sampling_ratio: Option<f64>,
    enabled: bool,
) -> GatewayServer {
    let mut opentelemetry = json!({
        "enabled": enabled,
        "endpoint": format!("{}/v1/traces", collector.uri()),
    });
    if let Some(ratio) = sampling_ratio {
        opentelemetry["sampling_ratio"] = json!(ratio);
    }
    let mut config: Config = serde_json::from_value(json!({
        "server": { "host": "127.0.0.1", "port": 0 },
        "providers": [
            { "name": "openai-provider", "type": "openai", "base_url": upstream.uri() }
        ],
        "model_groups": [{
            "name": "telemetry-group",
            "models": [
                { "provider": "openai-provider", "model": "gpt-4", "priority": 1 }
            ]
        }],
        "opentelemetry": opentelemetry,
    }))
    .expect("test configuration should deserialize");
    config.retry.max_retries_per_provider = 0;
    config.retry.backoff_sequence_seconds = vec![0];
    config.retry.jitter_enabled = false;
    common::isolate_databases(&mut config);
    GatewayServer::new(config, None).await.unwrap()
}

fn chat_request_with(traceparent: Option<&str>, x_request_id: Option<&str>) -> Request<Body> {
    let mut builder = Request::post("/v1/chat/completions")
        .header("content-type", "application/json");
    if let Some(traceparent) = traceparent {
        builder = builder.header("traceparent", traceparent);
    }
    if let Some(request_id) = x_request_id {
        builder = builder.header("x-request-id", request_id);
    }
    builder
        .body(Body::from(
            json!({
                "model": "telemetry-group",
                "messages": [{ "role": "user", "content": "hello" }],
                "stream": false
            })
            .to_string(),
        ))
        .unwrap()
}

fn gunzip_json(body: &[u8]) -> Value {
    let mut decoder = GzDecoder::new(body);
    let mut out = String::new();
    decoder
        .read_to_string(&mut out)
        .expect("collector body should be a valid gzip stream");
    serde_json::from_str(&out).expect("decompressed body should be valid OTLP JSON")
}

fn spans_of(otlp: &Value) -> Vec<&Value> {
    otlp["resourceSpans"]
        .as_array()
        .expect("resourceSpans array")
        .iter()
        .flat_map(|resource_spans| {
            resource_spans["scopeSpans"]
                .as_array()
                .expect("scopeSpans array")
                .iter()
                .flat_map(|scope_spans| {
                    scope_spans["spans"]
                        .as_array()
                        .expect("spans array")
                        .iter()
                })
        })
        .collect()
}

fn root_span<'a>(spans: &[&'a Value]) -> &'a Value {
    // The gateway root span is the only SERVER-kind span (OTLP kind 2);
    // with a valid inbound traceparent it also carries a remote
    // parentSpanId, so parent-absence cannot identify it here.
    spans
        .iter()
        .copied()
        .find(|span| span["kind"].as_i64() == Some(2))
        .expect("exactly one SERVER root span")
}

fn attr<'a>(span: &'a Value, key: &str) -> Option<&'a Value> {
    span["attributes"].as_array().and_then(|attrs| {
        attrs
            .iter()
            .find(|kv| kv["key"].as_str() == Some(key))
            .map(|kv| &kv["value"])
    })
}

fn attr_str(span: &Value, key: &str) -> Option<String> {
    attr(span, key)
        .and_then(|value| value["stringValue"].as_str())
        .map(str::to_string)
}

async fn wait_for_collector_requests(
    collector: &MockServer,
    count: usize,
) -> Vec<wiremock::Request> {
    let deadline = tokio::time::Instant::now() + Duration::from_secs(10);
    loop {
        let received = collector.received_requests().await.unwrap();
        if received.len() >= count || tokio::time::Instant::now() >= deadline {
            return received;
        }
        tokio::time::sleep(Duration::from_millis(50)).await;
    }
}

fn assert_gzip_post(request: &wiremock::Request) -> Value {
    let content_encoding = request
        .headers
        .get("content-encoding")
        .and_then(|value| value.to_str().ok())
        .unwrap_or_default();
    assert_eq!(content_encoding, "gzip", "OTLP POST must be gzip-encoded");
    gunzip_json(&request.body)
}

#[tokio::test]
async fn inbound_traceparent_ids_reach_the_otlp_root_span() {
    let collector = mount_collector().await;
    let upstream = mount_upstream().await;
    let server = server_for(&collector, &upstream, None, true).await;

    let response = server
        .build_router()
        .oneshot(chat_request_with(
            Some(&format!("00-{INBOUND_TRACE_ID}-{INBOUND_PARENT_SPAN_ID}-01")),
            None,
        ))
        .await
        .unwrap();
    assert_eq!(response.status(), StatusCode::OK);

    let received = wait_for_collector_requests(&collector, 1).await;
    let otlp = assert_gzip_post(&received[0]);
    let spans = spans_of(&otlp);
    let root = root_span(&spans);

    // Req 4.1: the root span uses the inbound trace id and parent span id.
    assert_eq!(
        root["traceId"].as_str(),
        Some(INBOUND_TRACE_ID),
        "root span continues the inbound trace"
    );
    assert_eq!(
        root["parentSpanId"].as_str(),
        Some(INBOUND_PARENT_SPAN_ID),
        "root span parented under the inbound parent span id"
    );
    // Correlation: the gateway request id rides as obey.request_id.
    assert!(
        attr_str(root, "obey.request_id").is_some(),
        "obey.request_id present alongside the inbound trace id"
    );
}

#[tokio::test]
async fn invalid_traceparent_falls_back_to_request_trace_id() {
    let collector = mount_collector().await;
    let upstream = mount_upstream().await;
    let server = server_for(&collector, &upstream, None, true).await;

    let response = server
        .build_router()
        .oneshot(chat_request_with(Some("not-a-traceparent"), Some(UUID_REQUEST_ID)))
        .await
        .unwrap();
    assert_eq!(response.status(), StatusCode::OK);

    let received = wait_for_collector_requests(&collector, 1).await;
    let otlp = assert_gzip_post(&received[0]);
    let root = root_span(&spans_of(&otlp));

    // Req 4.2/2.6: invalid header ignored; trace id reuses the UUID
    // request id; no remote parent.
    assert_eq!(
        root["traceId"].as_str(),
        Some(UUID_REQUEST_ID_HEX),
        "trace id derived from the UUID request id bytes"
    );
    assert!(
        root.get("parentSpanId").is_none(),
        "no remote parent for invalid inbound traceparent"
    );
    assert_eq!(
        attr_str(root, "obey.request_id"),
        Some(UUID_REQUEST_ID.to_string())
    );
}

#[tokio::test]
async fn absent_traceparent_uses_uuid_request_id_without_remote_parent() {
    let collector = mount_collector().await;
    let upstream = mount_upstream().await;
    let server = server_for(&collector, &upstream, None, true).await;

    let response = server
        .build_router()
        .oneshot(chat_request_with(None, Some(UUID_REQUEST_ID)))
        .await
        .unwrap();
    assert_eq!(response.status(), StatusCode::OK);

    let received = wait_for_collector_requests(&collector, 1).await;
    let otlp = assert_gzip_post(&received[0]);
    let root = root_span(&spans_of(&otlp));

    assert_eq!(
        root["traceId"].as_str(),
        Some(UUID_REQUEST_ID_HEX),
        "UUID request id becomes the OTLP trace id"
    );
    assert!(root.get("parentSpanId").is_none());
}

#[tokio::test]
async fn inbound_sampled_flag_one_overrides_ratio_zero() {
    // Req 5.5: sampled=1 in the traceparent forces sampling even at
    // sampling_ratio 0.0.
    let collector = mount_collector().await;
    let upstream = mount_upstream().await;
    let server = server_for(&collector, &upstream, Some(0.0), true).await;

    let response = server
        .build_router()
        .oneshot(chat_request_with(
            Some(&format!("00-{INBOUND_TRACE_ID}-{INBOUND_PARENT_SPAN_ID}-01")),
            None,
        ))
        .await
        .unwrap();
    assert_eq!(response.status(), StatusCode::OK);

    let received = wait_for_collector_requests(&collector, 1).await;
    let otlp = assert_gzip_post(&received[0]);
    let root = root_span(&spans_of(&otlp));
    assert_eq!(
        root["traceId"].as_str(),
        Some(INBOUND_TRACE_ID),
        "parent-based override samples the request despite ratio 0.0"
    );
}

#[tokio::test]
async fn inbound_sampled_flag_zero_suppresses_even_at_ratio_one() {
    // Req 5.5: sampled=0 in the traceparent makes the request unsampled
    // even at sampling_ratio 1.0 — no export POST at all.
    let collector = mount_collector().await;
    let upstream = mount_upstream().await;
    let server = server_for(&collector, &upstream, Some(1.0), true).await;

    let response = server
        .build_router()
        .oneshot(chat_request_with(
            Some(&format!("00-{INBOUND_TRACE_ID}-{INBOUND_PARENT_SPAN_ID}-00")),
            None,
        ))
        .await
        .unwrap();
    assert_eq!(response.status(), StatusCode::OK);

    tokio::time::sleep(Duration::from_millis(1500)).await;
    let received = collector.received_requests().await.unwrap();
    assert!(
        received.is_empty(),
        "sampled=0 inbound flag must suppress export regardless of ratio: {received:?}"
    );
}

#[tokio::test]
async fn unsampled_ratio_zero_without_inbound_context_exports_nothing() {
    // Req 5.3/5.4: ratio 0.0 without a sampled inbound flag = no POSTs,
    // while the request itself still succeeds.
    let collector = mount_collector().await;
    let upstream = mount_upstream().await;
    let server = server_for(&collector, &upstream, Some(0.0), true).await;

    let response = server
        .build_router()
        .oneshot(chat_request_with(None, None))
        .await
        .unwrap();
    assert_eq!(response.status(), StatusCode::OK);
    let body = axum::body::to_bytes(response.into_body(), 4 * 1024 * 1024)
        .await
        .unwrap();
    let completion: Value = serde_json::from_slice(&body).unwrap();
    assert_eq!(completion["choices"][0]["finish_reason"], "stop");

    tokio::time::sleep(Duration::from_millis(1500)).await;
    let received = collector.received_requests().await.unwrap();
    assert!(
        received.is_empty(),
        "ratio 0.0 must produce zero OTLP POSTs (Req 5.3/5.4): {received:?}"
    );
}

#[tokio::test]
async fn disabled_export_ignores_traceparent_headers() {
    // Req 4.4: with export disabled the traceparent has no observable
    // effect beyond today's behavior — and no POSTs happen at all.
    let collector = mount_collector().await;
    let upstream = mount_upstream().await;
    let server = server_for(&collector, &upstream, None, false).await;

    let response = server
        .build_router()
        .oneshot(chat_request_with(
            Some(&format!("00-{INBOUND_TRACE_ID}-{INBOUND_PARENT_SPAN_ID}-01")),
            None,
        ))
        .await
        .unwrap();
    assert_eq!(response.status(), StatusCode::OK);

    tokio::time::sleep(Duration::from_millis(1500)).await;
    let received = collector.received_requests().await.unwrap();
    assert!(
        received.is_empty(),
        "disabled telemetry must ignore traceparent and export nothing: {received:?}"
    );
}
