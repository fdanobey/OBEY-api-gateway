//! Integration tests for OpenTelemetry trace export (opentelemetry-trace-export
//! spec, tasks 8a; requirements 2-5/10, design "Integration Testing").
//!
//! Every test drives a real chat request through the gateway router with
//! `tower::ServiceExt::oneshot` (no bound ports) and observes the actual
//! gzip-compressed OTLP/HTTP JSON payloads received by a wiremock mock
//! collector — the same wire format a real OpenTelemetry Collector parses.

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

/// One model group whose members are `(provider_name, base_url)` in priority
/// order, with the OTLP exporter pointed at the mock collector. Models carry
/// nonzero token prices so the cost attributes are exercised.
fn telemetry_config(
    providers: &[(&str, String)],
    opentelemetry: serde_json::Value,
    max_retries: u32,
) -> Config {
    let provider_values: Vec<serde_json::Value> = providers
        .iter()
        .map(|(name, url)| json!({ "name": name, "type": "openai", "base_url": url }))
        .collect();
    let models: Vec<serde_json::Value> = providers
        .iter()
        .enumerate()
        .map(|(index, (name, _))| {
            json!({
                "provider": name,
                "model": "gpt-4",
                "priority": index + 1,
                "cost_per_million_input_tokens": 3.0,
                "cost_per_million_output_tokens": 15.0,
            })
        })
        .collect();
    let mut config: Config = serde_json::from_value(json!({
        "server": { "host": "127.0.0.1", "port": 0 },
        "providers": provider_values,
        "model_groups": [{ "name": "telemetry-group", "models": models }],
        "opentelemetry": opentelemetry,
    }))
    .expect("test configuration should deserialize");
    config.retry.max_retries_per_provider = max_retries;
    config.retry.backoff_sequence_seconds = vec![0];
    config.retry.jitter_enabled = false;
    common::isolate_databases(&mut config);
    config
}

fn enabled_section(endpoint: &str) -> serde_json::Value {
    json!({
        "enabled": true,
        "endpoint": format!("{endpoint}/v1/traces"),
        "service_name": "test-gateway",
    })
}

fn disabled_section(endpoint: &str) -> serde_json::Value {
    json!({
        "enabled": false,
        "endpoint": format!("{endpoint}/v1/traces"),
    })
}

fn chat_request() -> Request<Body> {
    Request::post("/v1/chat/completions")
        .header("content-type", "application/json")
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

/// Extract the flat span list from an OTLP ExportTraceServiceRequest.
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

fn attr_int(span: &Value, key: &str) -> Option<i64> {
    // 64-bit ints are decimal strings in the OTLP JSON protobuf mapping.
    attr(span, key)
        .and_then(|value| value["intValue"].as_str())
        .and_then(|raw| raw.parse::<i64>().ok())
        .or_else(|| attr(span, key).and_then(|value| value["intValue"].as_i64()))
}

fn attr_double(span: &Value, key: &str) -> Option<f64> {
    attr(span, key).and_then(|value| value["doubleValue"].as_f64())
}

fn resource_attr(otlp: &Value, key: &str) -> Option<String> {
    otlp["resourceSpans"]
        .as_array()?
        .get(0)?
        ["resource"]["attributes"]
        .as_array()?
        .iter()
        .find(|kv| kv["key"].as_str() == Some(key))
        .and_then(|kv| kv["value"]["stringValue"].as_str())
        .map(str::to_string)
}

/// The gateway root span is the only SERVER-kind span (OTLP kind 2).
fn root_span<'a>(spans: &[&'a Value]) -> &'a Value {
    spans
        .iter()
        .copied()
        .find(|span| span["kind"].as_i64() == Some(2))
        .expect("exactly one SERVER root span")
}

fn attempt_spans<'a>(spans: &[&'a Value]) -> Vec<&'a Value> {
    spans
        .iter()
        .copied()
        .filter(|span| span.get("parentSpanId").is_some())
        .collect()
}

/// Wait until the collector has received at least `count` requests (the
/// exporter batches asynchronously on a background task).
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

/// Assert the structural OTLP contract (Req 10.1) on one captured POST:
/// gzip Content-Encoding, application/json Content-Type, and a body that
/// decompresses to `resourceSpans`/`scopeSpans`/`spans` OTLP JSON.
fn assert_otlp_post(request: &wiremock::Request) -> Value {
    let content_encoding = request
        .headers
        .get("content-encoding")
        .and_then(|value| value.to_str().ok())
        .unwrap_or_default();
    assert_eq!(
        content_encoding, "gzip",
        "OTLP POST must carry Content-Encoding: gzip (Req 6.1)"
    );
    let content_type = request
        .headers
        .get("content-type")
        .and_then(|value| value.to_str().ok())
        .unwrap_or_default();
    assert!(
        content_type.starts_with("application/json"),
        "OTLP POST must carry Content-Type: application/json (Req 6.1), got {content_type}"
    );
    gunzip_json(&request.body)
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

async fn mount_upstream(status: u16) -> MockServer {
    let upstream = MockServer::start().await;
    let response = if status == 200 {
        ResponseTemplate::new(200).set_body_json(completion_body())
    } else {
        ResponseTemplate::new(status).set_body_string("upstream exploded")
    };
    Mock::given(method("POST"))
        .and(path("/v1/chat/completions"))
        .respond_with(response)
        .mount(&upstream)
        .await;
    upstream
}

async fn server_for(config: Config) -> GatewayServer {
    GatewayServer::new(config, None).await.unwrap()
}

#[tokio::test]
async fn chat_success_exports_root_and_attempt_span_with_usage_and_resource() {
    let collector = mount_collector().await;
    let upstream = mount_upstream(200).await;

    let server = server_for(telemetry_config(
        &[("openai-provider", upstream.uri())],
        enabled_section(&collector.uri()),
        0,
    ))
    .await;

    let response = server
        .build_router()
        .oneshot(chat_request())
        .await
        .unwrap();
    assert_eq!(response.status(), StatusCode::OK);

    let received = wait_for_collector_requests(&collector, 1).await;
    assert_eq!(
        received.len(),
        1,
        "exactly one OTLP POST for one request: {received:?}"
    );
    let otlp = assert_otlp_post(&received[0]);

    // Resource (Req 10.2): configurable service.name + service.version.
    assert_eq!(
        resource_attr(&otlp, "service.name"),
        Some("test-gateway".to_string())
    );
    assert!(
        resource_attr(&otlp, "service.version").is_some(),
        "service.version must be present on the resource"
    );

    let spans = spans_of(&otlp);
    assert_eq!(
        spans.len(),
        2,
        "root + one serving attempt span expected, got {}",
        spans.len()
    );

    let root = root_span(&spans);
    let attempts = attempt_spans(&spans);
    assert_eq!(attempts.len(), 1);

    // Root span (Req 2.1): SERVER kind, name `{method} {route}`, HTTP +
    // obey.* attributes.
    assert_eq!(root["kind"].as_i64(), Some(2), "root span kind SERVER");
    assert_eq!(
        root["name"].as_str(),
        Some("POST /v1/chat/completions"),
        "root span name {{method}} {{route}}"
    );
    assert_eq!(
        attr_str(root, "http.request.method"),
        Some("POST".to_string())
    );
    assert_eq!(
        attr_str(root, "http.route"),
        Some("/v1/chat/completions".to_string())
    );
    assert_eq!(attr_int(root, "http.response.status_code"), Some(200));
    assert!(
        attr_str(root, "obey.request_id").is_some(),
        "root span carries obey.request_id"
    );
    assert_eq!(
        attr_str(root, "obey.model.group"),
        Some("telemetry-group".to_string())
    );

    // Request-aggregated usage rides on the root span (Req 3.2).
    assert_eq!(attr_int(root, "gen_ai.usage.input_tokens"), Some(10));
    assert_eq!(attr_int(root, "gen_ai.usage.output_tokens"), Some(5));
    // Cost estimate (Req 3.3): decimal total + ISO currency default USD.
    // (10 prompt * $3 + 5 completion * $15) / 1e6 = 0.000105.
    let cost = attr_double(root, "gen_ai.usage.cost")
        .expect("root carries gen_ai.usage.cost");
    assert!(
        (cost - 0.000_105).abs() < 1e-9,
        "cost derived from usage and prices, got {cost}"
    );
    assert_eq!(
        attr_str(root, "gen_ai.usage.currency"),
        Some("USD".to_string())
    );

    // Attempt span (Req 3.1): CLIENT kind, `chat {model}` name, GenAI semconv.
    let attempt = attempts[0];
    assert_eq!(attempt["kind"].as_i64(), Some(3), "attempt span kind CLIENT");
    assert_eq!(
        attempt["name"].as_str(),
        Some("chat gpt-4"),
        "attempt span name {{operation}} {{model}}"
    );
    // Parent linkage (Req 2.3): attempt parented under the root span id.
    assert_eq!(
        attempt["parentSpanId"].as_str(),
        root["spanId"].as_str(),
        "attempt span parented under the root span"
    );
    assert_eq!(
        attr_str(attempt, "gen_ai.operation.name"),
        Some("chat".to_string())
    );
    assert_eq!(
        attr_str(attempt, "gen_ai.provider.name"),
        Some("openai-provider".to_string())
    );
    assert_eq!(
        attr_str(attempt, "gen_ai.system"),
        Some("openai-provider".to_string()),
        "gen_ai.system compatibility alias mirrors provider name lowercased"
    );
    assert_eq!(
        attr_str(attempt, "gen_ai.request.model"),
        Some("gpt-4".to_string())
    );
    // Responded model equals the requested provider model here, so no
    // `gen_ai.response.model` (Req 3.1 "when it differs").
    assert!(attr_str(attempt, "gen_ai.response.model").is_none());
    // Per-attempt usage (Req 3.2).
    assert_eq!(attr_int(attempt, "gen_ai.usage.input_tokens"), Some(10));
    assert_eq!(attr_int(attempt, "gen_ai.usage.output_tokens"), Some(5));
    assert_eq!(attr_int(attempt, "obey.attempt.index"), Some(0));
    // Successful attempt: no error.type (Req 3.4).
    assert!(attr(attempt, "error.type").is_none());
    // Trace ids agree between root and attempt (Req 2.3).
    assert_eq!(attempt["traceId"].as_str(), root["traceId"].as_str());
    // Span statuses (OTLP enum: OK=1, ERROR=2).
    assert_eq!(root["status"]["code"].as_i64(), Some(1));
    assert_eq!(attempt["status"]["code"].as_i64(), Some(1));
}

#[tokio::test]
async fn chat_success_exports_no_content_attributes_by_default() {
    let collector = mount_collector().await;
    let upstream = mount_upstream(200).await;

    let server = server_for(telemetry_config(
        &[("openai-provider", upstream.uri())],
        // Default section: capture_content unset (Req 3.5).
        json!({ "enabled": true, "endpoint": format!("{}/v1/traces", collector.uri()) }),
        0,
    ))
    .await;

    let response = server
        .build_router()
        .oneshot(chat_request())
        .await
        .unwrap();
    assert_eq!(response.status(), StatusCode::OK);

    let received = wait_for_collector_requests(&collector, 1).await;
    let otlp = assert_otlp_post(&received[0]);
    let spans = spans_of(&otlp);
    assert!(!spans.is_empty());

    for span in &spans {
        for key in ["gen_ai.prompt", "gen_ai.completion"] {
            assert!(
                attr(span, key).is_none(),
                "span attribute {key} must be absent when capture_content is off (Req 3.5): {}",
                span["name"]
            );
        }
    }
    // The provider's response content must not leak anywhere in the payload.
    let raw = serde_json::to_string(&otlp).unwrap();
    assert!(
        !raw.contains("Hello from mock provider"),
        "provider completion text must not appear in the exported payload"
    );
}

#[tokio::test]
async fn failover_exports_all_sibling_attempts_under_one_root() {
    let collector = mount_collector().await;
    let failing = mount_upstream(500).await;
    let serving = mount_upstream(200).await;

    let server = server_for(telemetry_config(
        &[
            ("failing-provider", failing.uri()),
            ("serving-provider", serving.uri()),
        ],
        enabled_section(&collector.uri()),
        0,
    ))
    .await;

    let response = server
        .build_router()
        .oneshot(chat_request())
        .await
        .unwrap();
    assert_eq!(response.status(), StatusCode::OK);

    let received = wait_for_collector_requests(&collector, 1).await;
    let otlp = assert_otlp_post(&received[0]);
    let spans = spans_of(&otlp);

    let root = root_span(&spans);
    let attempts = attempt_spans(&spans);
    assert_eq!(
        attempts.len(),
        2,
        "failed attempt + serving attempt expected: {}",
        serde_json::to_string(&otlp).unwrap()
    );

    // Both siblings share the root as parent (Req 2.3) and the trace id.
    for attempt in &attempts {
        assert_eq!(attempt["traceId"].as_str(), root["traceId"].as_str());
        assert_eq!(attempt["parentSpanId"].as_str(), root["spanId"].as_str());
    }

    // Sibling order follows attempt order: failed first, serving second.
    assert_eq!(
        attr_str(attempts[0], "gen_ai.provider.name"),
        Some("failing-provider".to_string())
    );
    assert_eq!(
        attr_str(attempts[1], "gen_ai.provider.name"),
        Some("serving-provider".to_string())
    );
    assert_eq!(attr_int(attempts[0], "obey.attempt.index"), Some(0));
    assert_eq!(attr_int(attempts[1], "obey.attempt.index"), Some(1));

    // Failed attempt (Req 3.4): bounded error.type class + ERROR status;
    // usage absent (the provider never answered with usage).
    assert_eq!(attempts[0]["status"]["code"].as_i64(), Some(2));
    let error_type = attr_str(attempts[0], "error.type").expect("error.type on failed attempt");
    assert_eq!(error_type, "upstream_http_500");
    assert!(
        !error_type.contains("upstream exploded"),
        "error.type is the content-free class, never the provider body"
    );
    assert!(attr_int(attempts[0], "gen_ai.usage.input_tokens").is_none());

    // Serving attempt (Req 3.2): usage + OK status; the root carries the
    // request totals.
    assert_eq!(attempts[1]["status"]["code"].as_i64(), Some(1));
    assert_eq!(attr_int(attempts[1], "gen_ai.usage.input_tokens"), Some(10));
    assert_eq!(attr_int(attempts[1], "gen_ai.usage.output_tokens"), Some(5));
    assert_eq!(
        attr_int(root_span(&spans), "gen_ai.usage.input_tokens"),
        Some(10)
    );
}

#[tokio::test]
async fn capture_on_bounded_content_attributes() {
    let collector = mount_collector().await;
    let upstream = mount_upstream(200).await;

    let mut config = telemetry_config(
        &[("openai-provider", upstream.uri())],
        json!({
            "enabled": true,
            "endpoint": format!("{}/v1/traces", collector.uri()),
            "capture_content": true,
        }),
        0,
    );
    // Tiny body bound (Req 3.6: content attributes are size-bounded by
    // logging.max_body_size_bytes) — the serialized request far exceeds it.
    config.logging.max_body_size_bytes = 32;
    let server = server_for(config).await;

    let response = server
        .build_router()
        .oneshot(chat_request())
        .await
        .unwrap();
    assert_eq!(response.status(), StatusCode::OK);

    let received = wait_for_collector_requests(&collector, 1).await;
    let otlp = assert_otlp_post(&received[0]);
    let spans = spans_of(&otlp);

    // Content rides on the (last) serving attempt span (Req 3.6).
    let attempt = attempt_spans(&spans)
        .last()
        .copied()
        .expect("serving attempt span present");

    let prompt = attr_str(attempt, "gen_ai.prompt").expect("gen_ai.prompt when capture_content=true");
    let completion = attr_str(attempt, "gen_ai.completion")
        .expect("gen_ai.completion when capture_content=true");
    // Truncation marker (Req 3.6): prompts larger than the bound are
    // truncated by spans.rs with an explicit "... [truncated by N bytes]"
    // marker. The marker MUST appear when truncation occurred — the old
    // entry-time silent truncation is gone.
    assert!(
        prompt.contains("[truncated by"),
        "truncation marker must be present for over-limit content: {prompt}"
    );
    assert!(
        prompt.starts_with("{\"model\":\"telemetry-group\""),
        "captured prompt is the bounded serialized request: {prompt}"
    );
    // The 24-byte completion fits the 32-byte bound unchanged.
    assert_eq!(completion, "Hello from mock provider");

    // No other span carries content attributes.
    for span in spans.iter().rev().skip(1) {
        assert!(attr(span, "gen_ai.prompt").is_none());
        assert!(attr(span, "gen_ai.completion").is_none());
    }
}

#[tokio::test]
async fn disabled_section_exports_zero_post_requests() {
    let collector = mount_collector().await;
    let upstream = mount_upstream(200).await;

    // Disabled but fully configured (Req 1.5/5.4): valid section, no export.
    let server = server_for(telemetry_config(
        &[("openai-provider", upstream.uri())],
        disabled_section(&collector.uri()),
        0,
    ))
    .await;

    let response = server
        .build_router()
        .oneshot(chat_request())
        .await
        .unwrap();
    assert_eq!(response.status(), StatusCode::OK);
    let body = axum::body::to_bytes(response.into_body(), 4 * 1024 * 1024)
        .await
        .unwrap();
    let completion: Value = serde_json::from_slice(&body).unwrap();
    assert_eq!(completion["choices"][0]["finish_reason"], "stop");

    // Give the (dormant) exporter ample time — nothing may arrive.
    tokio::time::sleep(Duration::from_millis(1500)).await;
    let received = collector.received_requests().await.unwrap();
    assert!(
        received.is_empty(),
        "disabled telemetry must produce zero OTLP POSTs (Req 5.4): {received:?}"
    );
}
