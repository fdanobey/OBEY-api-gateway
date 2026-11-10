//! Admin OpenTelemetry integration tests (opentelemetry-trace-export
//! spec, Task 9; Reqs 1, 7.3, 8, 9).
//!
//! Exercises the standard admin config API (`GET/PUT /admin/config`)
//! and `GET /admin/telemetry/status` through the full admin router via
//! `tower::ServiceExt::oneshot()` — no live server. OTLP collectors and
//! the chat provider are wiremock loopback mock servers; real span
//! export is observed by decoding the gzip OTLP/HTTP JSON POSTs the
//! background exporter delivers.
//!
//! Covers:
//! - Config round-trip: header values masked with the `[configured]`
//! sentinel in GET (Req 8.2); an unchanged mask PUT preserves the
//! stored value; an edited value replaces it — verified against the
//! actual collector request headers (Req 8.2/8.3)
//! - Validation errors: enabled + missing endpoint → structured error
//! naming `opentelemetry.endpoint`; sampling_ratio 1.5 → error naming
//! the field and the `[0.0, 1.0]` range (Req 1.2/1.3)
//! - Hot-reload retarget: spans arrive at collector A, then a config
//! PUT switching the endpoint routes later spans to B with nothing
//! further at A (Req 9.1)
//! - Status endpoint: enabled, endpoint host, spans_exported > 0,
//! healthy, and header values never appear in the status JSON
//! (Req 7.3)

use std::io::Read;
use std::time::Duration;

use axum::body::Body;
use axum::http::{Request, StatusCode};
use flate2::read::GzDecoder;
use tower::ServiceExt;
use wiremock::matchers::{method as wm_method, path as wm_path};
use wiremock::{Mock, MockServer, ResponseTemplate};

use ai_gateway::config::Config;
use ai_gateway::gateway::GatewayServer;
use serde_json::{json, Value};

mod common;

// Unique raw secrets per concern so cross-test leakage in an assertion
// message is unambiguous.
const RAW_SECRET: &str = "otel-secret-raw-4A71";
const REPLACEMENT_SECRET: &str = "otel-secret-rotated-9B22";
const STATUS_SECRET: &str = "otel-secret-status-6C55";
const MASK: &str = "[configured]";

/// Poll budget for off-path OTLP delivery (batch interval is 5 s).
const POLL_BUDGET: Duration = Duration::from_secs(10);

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

/// Minimal valid config: one wiremock openai provider + one model group,
/// plus the given `opentelemetry` section (a JSON value so each test
/// shapes its own). PUT validation rejects port 0, hence 8080.
fn gateway_config(provider_base_url: &str, telemetry: Value) -> Config {
    let mut config: Config = serde_json::from_value(json!({
        "server": {
            "host": "127.0.0.1",
            "port": 8080,
            "request_timeout_seconds": 30,
            "max_request_size_mb": 10
        },
        "providers": [{
            "name": "otel-provider",
            "type": "openai",
            "base_url": provider_base_url
        }],
        "model_groups": [{
            "name": "otel-group",
            "models": [{ "provider": "otel-provider", "model": "gpt-4", "priority": 100 }]
        }],
        "opentelemetry": telemetry
    }))
    .expect("test configuration should deserialize");
    common::isolate_databases(&mut config);
    config
}

/// Build the app with isolated databases and a temp config path (PUT
/// persists YAML through `save_config` — never into the repo tree).
async fn build_app(mut config: Config) -> (axum::Router, GatewayServer) {
    common::isolate_databases(&mut config);
    let config_dir = tempfile::tempdir().expect("temp config dir");
    let config_path = config_dir.path().join("config.yaml");
    // The dir must outlive the server's config_path handle; the gateway
    // holds no open handle to the YAML, so keep() matches the sibling
    // tests' leak-the-tiny-file discipline.
    let config_dir = config_dir.keep();
    let _ = &config_dir;
    let server = GatewayServer::new(config, Some(config_path)).await.unwrap();
    let router = server.build_router();
    (router, server)
}

async fn send(app: axum::Router, req: Request<Body>) -> (StatusCode, Value) {
    let resp = app.oneshot(req).await.unwrap();
    let status = resp.status();
    let bytes = axum::body::to_bytes(resp.into_body(), 1024 * 1024)
        .await
        .unwrap();
    let json = if bytes.is_empty() {
        Value::Null
    } else {
        serde_json::from_slice(&bytes).unwrap()
    };
    (status, json)
}

async fn get_config(app: axum::Router) -> Value {
    let (status, json) = send(app, Request::get("/admin/config").body(Body::empty()).unwrap()).await;
    assert_eq!(status, StatusCode::OK, "config GET succeeds: {json}");
    json
}

async fn put_config(app: axum::Router, body: &Value) -> (StatusCode, Value) {
    send(
        app,
        Request::put("/admin/config")
            .header("content-type", "application/json")
            .body(Body::from(body.to_string()))
            .unwrap(),
    )
    .await
}

async fn get_status(app: axum::Router) -> Value {
    let (status, json) = send(
        app,
        Request::get("/admin/telemetry/status")
            .body(Body::empty())
            .unwrap(),
    )
    .await;
    assert_eq!(status, StatusCode::OK, "status GET succeeds: {json}");
    json
}

/// Mount the mock chat provider (plain 200 completion with usage).
async fn mount_upstream(server: &MockServer) {
    Mock::given(wm_method("POST"))
        .and(wm_path("/v1/chat/completions"))
        .respond_with(ResponseTemplate::new(200).set_body_json(json!({
            "id": "chatcmpl-otel-admin",
            "object": "chat.completion",
            "created": 1700000000_i64,
            "model": "gpt-4",
            "choices": [{
                "index": 0,
                "message": { "role": "assistant", "content": "ok" },
                "finish_reason": "stop"
            }],
            "usage": { "prompt_tokens": 10, "completion_tokens": 5, "total_tokens": 15 }
        })))
        .mount(server)
        .await;
}

/// Mount a 200-OK OTLP collector at `/v1/traces`.
async fn mount_collector(server: &MockServer) {
    Mock::given(wm_method("POST"))
        .and(wm_path("/v1/traces"))
        .respond_with(ResponseTemplate::new(200))
        .mount(server)
        .await;
}

/// Drive one non-stream chat completion through the gateway.
async fn drive_completion(app: axum::Router, turn: &str) -> StatusCode {
    let req = Request::post("/v1/chat/completions")
        .header("content-type", "application/json")
        .body(Body::from(
            json!({
                "model": "otel-group",
                "messages": [{ "role": "user", "content": turn }]
            })
            .to_string(),
        ))
        .unwrap();
    let resp = app.oneshot(req).await.unwrap();
    resp.status()
}

async fn otlp_posts(collector: &MockServer) -> Vec<wiremock::Request> {
    collector
        .received_requests()
        .await
        .unwrap_or_default()
        .into_iter()
        .filter(|request| request.url.path() == "/v1/traces")
        .collect()
}

/// Poll until the collector holds at least `min` OTLP POSTs (bounded by
/// [`POLL_BUDGET`]); panics with the observed count otherwise.
async fn wait_for_otlp(collector: &MockServer, min: usize) -> Vec<wiremock::Request> {
    let deadline = tokio::time::Instant::now() + POLL_BUDGET;
    loop {
        let posts = otlp_posts(collector).await;
        if posts.len() >= min {
            return posts;
        }
        if tokio::time::Instant::now() >= deadline {
            panic!(
                "collector at {} never received {min} OTLP posts (saw {})",
                collector.uri(),
                posts.len()
            );
        }
        tokio::time::sleep(Duration::from_millis(50)).await;
    }
}

/// Decode a delivered OTLP/HTTP JSON body (gzip per Req 6.1, with a
/// plain-JSON fallback for robustness).
fn decode_otlp_json(request: &wiremock::Request) -> Value {
    if let Ok(value) = serde_json::from_slice(&request.body) {
        return value;
    }
    let mut decoder = GzDecoder::new(request.body.as_slice());
    let mut decoded = Vec::new();
    decoder
        .read_to_end(&mut decoded)
        .expect("gzip OTLP body decodes");
    serde_json::from_slice(&decoded).expect("OTLP payload is JSON")
}

/// Assert the payload is structurally valid OTLP JSON with at least one
/// span and the `ai-gateway` service resource (Req 10.1/10.2) — the
/// collector really received spans, not just any POST.
fn assert_otlp_spans(payload: &Value) {
    let spans = payload["resourceSpans"][0]["scopeSpans"][0]["spans"]
        .as_array()
        .unwrap_or_else(|| panic!("OTLP payload has spans array: {payload}"));
    assert!(!spans.is_empty(), "OTLP batch must contain spans: {payload}");
    let attrs = payload["resourceSpans"][0]["resource"]["attributes"]
        .as_array()
        .unwrap_or_else(|| panic!("OTLP payload has resource attributes: {payload}"));
    let service_name = attrs
        .iter()
        .find(|attr| attr["key"] == "service.name")
        .map(|attr| attr["value"]["stringValue"].clone())
        .unwrap_or_else(|| panic!("resource carries service.name: {payload}"));
    assert_eq!(service_name, "ai-gateway", "default service.name");
}

fn header_value<'r>(request: &'r wiremock::Request, name: &str) -> &'r str {
    request
        .headers
        .get(name)
        .unwrap_or_else(|| panic!("{name} header present on OTLP post"))
        .to_str()
        .expect("header value is ASCII")
}

fn endpoint_url(collector: &MockServer) -> String {
    format!("{}/v1/traces", collector.uri())
}

/// Read the live stored `opentelemetry.headers.authorization` value.
async fn stored_authorization(server: &GatewayServer) -> Option<String> {
    let live = server.state.config.read().await;
    live.opentelemetry
        .headers
        .get("authorization")
        .cloned()
}

// ---------------------------------------------------------------------------
// Config round-trip: mask, preserve, replace (Reqs 8.2, 8.3)
// ---------------------------------------------------------------------------

/// GET masks the header value; PUT of the unchanged mask preserves the
/// stored secret — proven by a real export carrying the ORIGINAL value —
/// and PUT of an edited value replaces it on the wire.
#[tokio::test]
async fn config_round_trip_masks_preserves_and_replaces_header() {
    let upstream = MockServer::start().await;
    mount_upstream(&upstream).await;
    let collector = MockServer::start().await;
    mount_collector(&collector).await;

    let telemetry = json!({
        "enabled": true,
        "endpoint": endpoint_url(&collector),
        "headers": { "authorization": RAW_SECRET }
    });
    let (app, server) = build_app(gateway_config(&upstream.uri(), telemetry)).await;

    // GET: section present, header value masked, raw secret absent.
    let body = get_config(app.clone()).await;
    assert_eq!(body["opentelemetry"]["enabled"], true);
    assert_eq!(body["opentelemetry"]["endpoint"], endpoint_url(&collector));
    assert_eq!(
        body["opentelemetry"]["headers"]["authorization"], MASK,
        "header value must be masked in config GET: {body}"
    );
    let serialized = body.to_string();
    assert!(
        !serialized.contains(RAW_SECRET),
        "config GET must never leak the raw header value: {serialized}"
    );

    // PUT the masked body back unchanged: the stored secret survives.
    let (status, error) = put_config(app.clone(), &body).await;
    assert_eq!(status, StatusCode::OK, "mask-unchanged PUT applies: {error}");
    assert_eq!(
        stored_authorization(&server).await.as_deref(),
        Some(RAW_SECRET),
        "unchanged mask must preserve the stored header value"
    );

    // GET again: still masked, still no leak.
    let body = get_config(app.clone()).await;
    assert_eq!(body["opentelemetry"]["headers"]["authorization"], MASK);
    assert!(!body.to_string().contains(RAW_SECRET));

    // Real export round-trip: the preserved ORIGINAL value reaches the
    // collector on the OTLP POST (Req 8.2 preservation is observable).
    let status = drive_completion(app.clone(), "preserve-check").await;
    assert_eq!(status, StatusCode::OK, "chat completion succeeds");
    let posts = wait_for_otlp(&collector, 1).await;
    assert_eq!(
        header_value(&posts[0], "authorization"),
        RAW_SECRET,
        "preserved header value must be sent to the collector"
    );
    assert_otlp_spans(&decode_otlp_json(&posts[0]));

    // PUT with an edited value replaces the stored secret.
    let mut edited = get_config(app.clone()).await;
    edited["opentelemetry"]["headers"]["authorization"] =
        Value::String(REPLACEMENT_SECRET.to_string());
    let (status, error) = put_config(app.clone(), &edited).await;
    assert_eq!(status, StatusCode::OK, "edited-value PUT applies: {error}");
    assert_eq!(
        stored_authorization(&server).await.as_deref(),
        Some(REPLACEMENT_SECRET),
        "edited header value must replace the stored one"
    );

    // The replacement is observable on the wire.
    let status = drive_completion(app.clone(), "rotate-check").await;
    assert_eq!(status, StatusCode::OK, "second chat completion succeeds");
    let posts = wait_for_otlp(&collector, 2).await;
    let last = posts.last().expect("two OTLP posts by now");
    assert_eq!(
        header_value(last, "authorization"),
        REPLACEMENT_SECRET,
        "rotated header value must be sent to the collector"
    );
    assert!(
        !posts
            .iter()
            .any(|post| header_value(post, "authorization") == MASK),
        "the mask literal must never be sent to a collector"
    );
}

// ---------------------------------------------------------------------------
// Validation errors (Reqs 1.2, 1.3)
// ---------------------------------------------------------------------------

/// PUT with `enabled: true` and a missing endpoint, then an
/// out-of-range sampling_ratio, yields structured validation errors
/// naming the offending field (and the allowed range).
#[tokio::test]
async fn config_put_rejects_invalid_telemetry_section() {
    let upstream = MockServer::start().await;
    let collector = MockServer::start().await;
    mount_collector(&collector).await;

    let (app, _server) = build_app(gateway_config(&upstream.uri(), json!({}))).await;

    // Enabled + missing endpoint → error naming opentelemetry.endpoint.
    let mut body = get_config(app.clone()).await;
    body["opentelemetry"]["enabled"] = Value::Bool(true);
    body["opentelemetry"]["endpoint"] = Value::Null;
    let (status, error) = put_config(app.clone(), &body).await;
    assert_eq!(
        status,
        StatusCode::BAD_REQUEST,
        "enabled without endpoint must be rejected: {error}"
    );
    assert_eq!(error["error"]["type"], "validation_error", "{error}");
    let details = serde_json::to_string(
        error["error"]["details"]
            .as_array()
            .expect("structured details array"),
    )
    .unwrap();
    assert!(
        details.contains("opentelemetry.endpoint"),
        "error must name opentelemetry.endpoint: {details}"
    );

    // sampling_ratio 1.5 → error naming the field and the range.
    let mut body = get_config(app.clone()).await;
    body["opentelemetry"]["enabled"] = Value::Bool(true);
    body["opentelemetry"]["endpoint"] = Value::String(endpoint_url(&collector));
    body["opentelemetry"]["sampling_ratio"] = json!(1.5);
    let (status, error) = put_config(app.clone(), &body).await;
    assert_eq!(
        status,
        StatusCode::BAD_REQUEST,
        "sampling_ratio 1.5 must be rejected: {error}"
    );
    assert_eq!(error["error"]["type"], "validation_error", "{error}");
    let details = serde_json::to_string(
        error["error"]["details"]
            .as_array()
            .expect("structured details array"),
    )
    .unwrap();
    assert!(
        details.contains("opentelemetry.sampling_ratio"),
        "error must name opentelemetry.sampling_ratio: {details}"
    );
    assert!(
        details.contains("[0.0, 1.0]"),
        "error must state the allowed range: {details}"
    );
}

// ---------------------------------------------------------------------------
// Hot-reload retarget (Req 9.1)
// ---------------------------------------------------------------------------

/// Spans flow to collector A; a config PUT switching the endpoint to B
/// routes later spans to B with nothing further delivered to A.
#[tokio::test]
async fn hot_reload_retargets_collector_a_to_b() {
    let upstream = MockServer::start().await;
    mount_upstream(&upstream).await;
    let collector_a = MockServer::start().await;
    let collector_b = MockServer::start().await;
    mount_collector(&collector_a).await;
    mount_collector(&collector_b).await;

    let telemetry = json!({
        "enabled": true,
        "endpoint": endpoint_url(&collector_a)
    });
    let (app, _server) = build_app(gateway_config(&upstream.uri(), telemetry)).await;

    // Before A: no OTLP traffic at either collector.
    assert!(
        otlp_posts(&collector_a).await.is_empty(),
        "no spans before any request"
    );

    // One completion → spans arrive at A.
    let status = drive_completion(app.clone(), "target-a").await;
    assert_eq!(status, StatusCode::OK);
    let posts = wait_for_otlp(&collector_a, 1).await;
    assert_eq!(posts.len(), 1, "exactly one export at A so far");
    assert_otlp_spans(&decode_otlp_json(&posts[0]));

    // Hot-reload retarget via the standard config PUT (no restart).
    let mut body = get_config(app.clone()).await;
    assert_eq!(
        body["opentelemetry"]["endpoint"],
        endpoint_url(&collector_a),
        "GET reflects the pre-retarget endpoint"
    );
    body["opentelemetry"]["endpoint"] = Value::String(endpoint_url(&collector_b));
    let (status, error) = put_config(app.clone(), &body).await;
    assert_eq!(status, StatusCode::OK, "retarget PUT applies: {error}");

    // Another completion → spans arrive at B, none further at A.
    let status = drive_completion(app.clone(), "target-b").await;
    assert_eq!(status, StatusCode::OK);
    let _ = wait_for_otlp(&collector_b, 1).await;
    // Grace period so a stray late A delivery would be caught.
    tokio::time::sleep(Duration::from_millis(300)).await;
    assert_eq!(
        otlp_posts(&collector_a).await.len(),
        1,
        "no further OTLP posts at A after retarget to B"
    );
    let b_posts = otlp_posts(&collector_b).await;
    assert!(!b_posts.is_empty(), "spans arrive at B after retarget");
    assert_otlp_spans(&decode_otlp_json(&b_posts[0]));
}

// ---------------------------------------------------------------------------
// Status endpoint (Req 7.3)
// ---------------------------------------------------------------------------

/// After real exports, the status card reports enabled, the endpoint
/// HOST, exported-span counters, health — and never the header value.
#[tokio::test]
async fn status_endpoint_reports_counters_without_header_leak() {
    let upstream = MockServer::start().await;
    mount_upstream(&upstream).await;
    let collector = MockServer::start().await;
    mount_collector(&collector).await;

    let telemetry = json!({
        "enabled": true,
        "endpoint": endpoint_url(&collector),
        "headers": { "authorization": STATUS_SECRET }
    });
    let (app, _server) = build_app(gateway_config(&upstream.uri(), telemetry)).await;

    let status = drive_completion(app.clone(), "status-check").await;
    assert_eq!(status, StatusCode::OK, "chat completion succeeds");

    // Poll the status endpoint until the export lands (off-path batch).
    let deadline = tokio::time::Instant::now() + POLL_BUDGET;
    let status_json = loop {
        let status_json = get_status(app.clone()).await;
        if status_json["spans_exported"].as_u64().unwrap_or(0) > 0 {
            break status_json;
        }
        if tokio::time::Instant::now() >= deadline {
            panic!("spans_exported never went positive: {status_json}");
        }
        tokio::time::sleep(Duration::from_millis(50)).await;
    };

    assert_eq!(status_json["enabled"], true, "exporter enabled: {status_json}");
    assert_eq!(
        status_json["endpoint_host"], "127.0.0.1",
        "endpoint host only (Req 7.3): {status_json}"
    );
    assert!(
        status_json["spans_exported"].as_u64().unwrap_or(0) > 0,
        "spans_exported > 0: {status_json}"
    );
    assert_eq!(
        status_json["export_failures"], 0,
        "no failures against the 200 collector: {status_json}"
    );
    assert_eq!(status_json["healthy"], true, "healthy: {status_json}");

    let serialized = status_json.to_string();
    assert!(
        !serialized.contains(STATUS_SECRET),
        "header values must never appear in the status JSON: {serialized}"
    );
    assert!(
        !serialized.contains(MASK),
        "status is not a config surface; the mask sentinel has no business there: {serialized}"
    );
}
