//! Integration tests for the decisions front door (`POST /v1/decisions`,
//! alias `POST /v1/systemone`) — spec decisions-api, Task 5.
//!
//! Drives the full Axum router via `tower::ServiceExt::oneshot()` (no port
//! binding for the gateway) with `wiremock` upstreams standing in for the
//! System One providers, and per-test isolated SQLite databases via
//! `common::isolate_databases`.
//!
//! Coverage map (Task 5 list, numbered):
//! 1. happy path: verbatim answers under caller ids + usage + model echo +
//!    content-free log row (key attribution, no state/questions even with
//!    body logging enabled)
//! 2. `auto`/alias resolution via `/v1/models` discovery, resolved id echoed
//! 3. generic `systemone` provider: pinned explicit id; `auto` -> 400
//! 4. validation 400s naming the field, zero upstream contact
//! 5. 404 unknown model (with and without decision providers)
//! 6. failover: primary 429 (with Retry-After) -> secondary 200
//! 7. hung primary -> timeout-class failover within the provider budget
//! 8. both candidates fail -> aggregated 502 with per-attempt details
//! 9. upstream status mapping: 401->502, 413->413, 422->422, 400->400
//! 10. vk `required` 401; `model_access` denial 403
//! 11. `/v1/systemone` alias behaves identically (canonical log path)
//! 12. chat completions for a decision model id -> existing not-found shape
//! 13. `/v1/models`: manual_models listed (systemone + family); provider
//!     without manual_models contributes no entry (Req 5.2 MAY-omit rule)
//! 14. hot-reload adds a decision provider without restart
//! 15. decision-provider outage (connection refused) leaves chat unaffected
//! 16. NFR Performance: mean gateway overhead per request under budget
//! against an instant-answering mock (`#[ignore]`, run with `--ignored`)
//! 17. handler rejections: non-JSON Content-Type + malformed JSON body
//! (400, zero upstream contact, logged rows)
//! 18. admission: local rate-limit skip fails over with a 429
//! `#attempt` row (slot saturation covered by router unit tests)
//! 19. discovery failure (listing 500) fails over to a healthy
//! secondary auto candidate
//! 20. whitespace-padded model ids trim to their provider
//!
//! Note on the attempt ledger: failed decision tries surface BOTH ways
//! (Req 2.7) — per-attempt log rows under `/v1/decisions#attempt`
//! (shared parent trace_id; provider/model/status/duration/error_class;
//! error string in response_body) plus the aggregated error body's
//! `attempts` array (scenario 8).

use std::time::{Duration, Instant};

use ai_gateway::config::{Config, EnforcementMode};
use ai_gateway::gateway::{apply_runtime_config_update, GatewayServer};
use ai_gateway::logger::LogFilter;
use ai_gateway::virtual_keys::models::CreateKeyParams;
use axum::body::Body;
use axum::http::{Request, StatusCode};
use serde_json::{json, Value};
use tower::ServiceExt;
use wiremock::matchers::{method, path};
use wiremock::{Mock, MockServer, ResponseTemplate};

use crate::common;

/// Canonical decisions route (the alias logs under this path too, Req 4.1).
const CANONICAL: &str = "/v1/decisions";
/// Alias route served by the same handler (Req 1.2).
const ALIAS: &str = "/v1/systemone";

/// NFR Performance budget: mean per-request gateway overhead measured over
/// 20 sequential oneshot round-trips (after warm-up) against an
/// instant-answering mock — upstream time excluded by construction.
/// 10 ms is the spec target; see `mean_gateway_overhead_under_budget`.
const OVERHEAD_BUDGET_MS: u64 = 10;

// ---------------------------------------------------------------------------
// 8c. Admission: local rate-limit skip fails over with a 429 attempt row
// ---------------------------------------------------------------------------

#[tokio::test]
async fn local_rate_limit_skip_fails_over_with_429_attempt_row() {
 let primary = MockServer::start().await;
 mount_evaluate(&primary, 200, answers_envelope("jev-1.13.0"), None).await;
 let secondary = MockServer::start().await;
 mount_evaluate(&secondary, 200, answers_envelope("jev-1.13.0"), None).await;
 let mut config = base_config(
 &[
 decision_provider("jev-throttled", "jev", &primary.uri(), &["jev-1.13.0"]),
 decision_provider("jev-relief", "jev", &secondary.uri(), &["jev-1.13.0"]),
 ],
 &[],
 );
 // One token per minute: the first request consumes it, the second is
 // skipped locally (non-blocking) and fails over — without waiting for
 // the primary (ADR 0002: admission never blocks on a token).
 config.providers[0].rate_limit_per_minute = 1;
 let server = GatewayServer::new(config, None).await.expect("gateway starts");
 let app = server.build_router();

 // First request: the primary's single token is consumed.
 let (status, body) = post_decisions(app.clone(), CANONICAL, None, decision_request("jev-1.13.0")).await;
 assert_eq!(status, StatusCode::OK, "body: {body}");

 // Second request: the exhausted local limiter skips the primary
 // pre-dispatch (429 attempt row) and the healthy secondary answers.
 let started = Instant::now();
 let (status, body) = post_decisions(app, CANONICAL, None, decision_request("jev-1.13.0")).await;
 assert_eq!(status, StatusCode::OK, "body: {body}");
 assert!(
 started.elapsed() < Duration::from_secs(5),
 "local skip is non-blocking (took {:?})",
 started.elapsed()
 );
 assert_eq!(
 upstream_posts(&primary).await.len(),
 1,
 "the skipped try never dispatched to the primary"
 );
 assert_eq!(upstream_posts(&secondary).await.len(), 1);

 let attempts = decision_attempt_rows(&server);
 let row = attempts
 .iter()
 .find(|row| row.provider == "jev-throttled")
 .expect("429 #attempt row for the locally skipped primary");
 assert_eq!(row.status_code, 429, "row: {row:?}");
 assert_eq!(row.error_class.as_deref(), Some("rate_limit_skip"));
 assert_eq!(row.model, "jev-1.13.0");
}

#[tokio::test]
async fn discovery_failure_on_primary_auto_fails_over_to_secondary() {
 let primary = MockServer::start().await;
 // Listing 500s; discovery fails as a transient-class attempt (502 row)
 // and the healthy secondary auto candidate takes over.
 Mock::given(method("GET"))
 .and(path("/v1/models"))
 .respond_with(ResponseTemplate::new(500).set_body_json(json!({ "error": "listing down" })))
 .mount(&primary)
 .await;
 let secondary = MockServer::start().await;
 mount_models_listing(&secondary, &["jev-1.14.0"]).await;
 mount_evaluate(&secondary, 200, answers_envelope("jev-1.14.0"), None).await;
 let config = base_config(
 &[
 decision_provider("jev-listing-dead", "jev", &primary.uri(), &[]),
 decision_provider("jev-listing-alive", "jev", &secondary.uri(), &[]),
 ],
 &[],
 );
 let server = GatewayServer::new(config, None).await.expect("gateway starts");

 let (status, body) = post_decisions(
 server.build_router(),
 CANONICAL,
 None,
 decision_request("auto"),
 )
 .await;
 assert_eq!(status, StatusCode::OK, "body: {body}");
 assert_eq!(body["model"], json!("jev-1.14.0"));
 assert!(
 upstream_posts(&primary).await.is_empty(),
 "primary never reaches evaluate — only its listing was consulted"
 );

 let attempts = decision_attempt_rows(&server);
 let row = attempts
 .iter()
 .find(|row| row.provider == "jev-listing-dead")
 .expect("502-ish attempt row for the primary");
 assert_eq!(row.status_code, 502, "listing failure maps to 502");
 assert_eq!(row.model, "auto");
 assert!(row.error_class.is_some(), "attempt rows carry a class");
}

// ---------------------------------------------------------------------------
// 9. Upstream status mapping: 401->502, 413->413, 422->422, 400->400
// ---------------------------------------------------------------------------


/// Minimal config with the given providers/groups, retry ladder collapsed
/// (the failover behavior under test is the router's candidate loop, not
/// the family retry ladder) and isolated databases.
fn base_config(providers: &[Value], model_groups: &[Value]) -> Config {
    let mut config: Config = serde_json::from_value(json!({
        "server": { "host": "127.0.0.1", "port": 0 },
        "providers": providers,
        "model_groups": model_groups,
    }))
    .expect("decisions test config deserializes");
    config.retry.max_retries_per_provider = 0;
    config.retry.backoff_sequence_seconds = vec![0];
    for provider in &mut config.providers {
        provider.resolved_api_key = Some("test-decision-key".to_string());
    }
    common::isolate_databases(&mut config);
    config
}

/// A decision provider entry (`jev` / `laya` / `systemone`).
fn decision_provider(
    name: &str,
    provider_type: &str,
    base_url: &str,
    manual_models: &[&str],
) -> Value {
    json!({
        "name": name,
        "type": provider_type,
        "base_url": base_url,
        "manual_models": manual_models,
    })
}

/// Provider-reported answer envelope with one answer per question type.
fn answers_envelope(model: &str) -> Value {
    json!({
        "model": model,
        "answers": {
            "is_urgent": { "type": "noul", "noul": 0.95 },
            "department": {
                "type": "choice",
                "choice": "technical",
                "probabilities": { "billing": 0.12, "technical": 0.88 },
                "confidence": 0.9
            },
            "sentiment": {
                "type": "score",
                "score": 1.95,
                "confidence": 0.85,
                "probabilities": { "0": 0.05, "1": 0.95 },
                "legend": { "0": "Calm", "1": "Frustrated" }
            }
        },
 "usage": { "input_tokens": 307, "output_tokens": 20, "latency_ms": 12 },
 "provider_extras": { "serving_region": "eu-west-1", "degraded": false }
 })
}

/// A valid decision request body for the given model.
fn decision_request(model: &str) -> Value {
    json!({
        "model": model,
        "state": { "messages": [{ "role": "user", "content": "the printer is on fire" }] },
        "questions": {
            "is_urgent": { "type": "noul" },
            "department": { "type": "choice", "criteria": { "options": ["billing", "technical"] } },
            "sentiment": { "type": "score", "criteria": { "levels": ["calm", "angry"] } }
        }
    })
}

/// Mount `POST /v1/systemone` on the mock: given status/body, optional delay.
async fn mount_evaluate(mock: &MockServer, status: u16, body: Value, delay: Option<Duration>) {
    let mut template = ResponseTemplate::new(status).set_body_json(body);
    if let Some(delay) = delay {
        template = template.set_delay(delay);
    }
    Mock::given(method("POST"))
        .and(path("/v1/systemone"))
        .respond_with(template)
        .mount(mock)
        .await;
}

/// Mount a transient rate-limit response honoring Retry-After semantics.
/// The 60 s hint must NOT block: failover is immediate (asserted in
/// `rate_limited_primary_fails_over_to_secondary`).
async fn mount_rate_limited(mock: &MockServer) {
 Mock::given(method("POST"))
 .and(path("/v1/systemone"))
 .respond_with(
 ResponseTemplate::new(429)
 .set_body_json(json!({ "error": "rate limited" }))
 .append_header("retry-after", "60"),
 )
 .mount(mock)
 .await;
}

/// Mount `GET /v1/models` returning the given catalog ids.
async fn mount_models_listing(mock: &MockServer, ids: &[&str]) {
    let data: Vec<Value> = ids.iter().map(|id| json!({ "id": id })).collect();
    Mock::given(method("GET"))
        .and(path("/v1/models"))
        .respond_with(ResponseTemplate::new(200).set_body_json(json!({ "data": data })))
        .mount(mock)
        .await;
}

/// Mount `POST /v1/chat/completions` returning a valid chat completion.
async fn mount_chat_completion(mock: &MockServer) {
    Mock::given(method("POST"))
        .and(path("/v1/chat/completions"))
        .respond_with(ResponseTemplate::new(200).set_body_json(json!({
            "id": "chatcmpl-decisions-isolation",
            "object": "chat.completion",
            "created": 1_700_000_000_i64,
            "model": "gpt-4",
            "choices": [{
                "index": 0,
                "message": { "role": "assistant", "content": "still chatting" },
                "finish_reason": "stop"
            }],
            "usage": { "prompt_tokens": 4, "completion_tokens": 3, "total_tokens": 7 }
        })))
        .mount(mock)
        .await;
}

/// POST a decision body to `route` through the gateway router.
async fn post_decisions(
    app: axum::Router,
    route: &str,
    bearer: Option<&str>,
    body: Value,
) -> (StatusCode, Value) {
    let mut builder = Request::post(route).header("content-type", "application/json");
    if let Some(key) = bearer {
        builder = builder.header("authorization", format!("Bearer {key}"));
    }
    let response = app
        .oneshot(builder.body(Body::from(body.to_string())).unwrap())
        .await
        .expect("gateway responded");
    let status = response.status();
    let bytes = axum::body::to_bytes(response.into_body(), 1024 * 1024)
        .await
        .expect("bounded response body");
    (status, serde_json::from_slice(&bytes).expect("JSON response body"))
}

/// Decision outcome rows written under the canonical path.
fn decision_rows(server: &GatewayServer) -> Vec<ai_gateway::logger::LogEntry> {
 server
 .state
 .logger
 .query(LogFilter::default())
 .unwrap_or_default()
 .into_iter()
 .filter(|row| row.path == CANONICAL)
 .collect()
}

/// Per-attempt ledger rows for failed decision tries (`#attempt` path,
/// Req 2.7) — same trace id family as the outcome row.
fn decision_attempt_rows(server: &GatewayServer) -> Vec<ai_gateway::logger::LogEntry> {
 server
 .state
 .logger
 .query(LogFilter::default())
 .unwrap_or_default()
 .into_iter()
 .filter(|row| row.path == "/v1/decisions#attempt")
 .collect()
}

/// Evaluate (POST /v1/systemone) bodies captured by a mock upstream.
async fn upstream_posts(mock: &MockServer) -> Vec<Value> {
    mock.received_requests()
        .await
        .expect("wiremock request recording")
        .into_iter()
        .filter(|request| request.url.path().contains("/v1/systemone"))
        .map(|request| serde_json::from_slice(&request.body).expect("evaluate body JSON"))
        .collect()
}

/// Provision an unrestricted virtual key on the server's key manager.
async fn create_virtual_key(
    server: &GatewayServer,
    model_access: Option<Vec<String>>,
) -> String {
    let created = server
        .state
        .virtual_key_manager
        .create_key(CreateKeyParams {
            name: None,
            budget_limit_usd: None,
            token_budget: None,
            budget_window: None,
            requests_per_minute: None,
            tokens_per_minute: None,
            model_access,
            expires_in: None,
            loop_detection: None,
        })
        .await
        .expect("virtual key created");
    created.key
}

// ---------------------------------------------------------------------------
// 1. Happy path + content-free log row
// ---------------------------------------------------------------------------

#[tokio::test]
async fn happy_path_serves_verbatim_answers_and_writes_content_free_log_row() {
    let mock = MockServer::start().await;
    mount_evaluate(&mock, 200, answers_envelope("jev-1.13.0"), None).await;
    let mut config = base_config(
        &[decision_provider("jev-primary", "jev", &mock.uri(), &["jev-1.13.0"])],
        &[],
    );
    config.logging.request_body_logging = true;
    config.virtual_keys.enforcement = EnforcementMode::Required;
    let server = GatewayServer::new(config, None).await.expect("gateway starts");
    let key = create_virtual_key(&server, None).await;

    let (status, body) = post_decisions(
        server.build_router(),
        CANONICAL,
        Some(&key),
        decision_request("jev-1.13.0"),
    )
    .await;
    assert_eq!(status, StatusCode::OK, "body: {body}");
    assert_eq!(body["model"], json!("jev-1.13.0"), "model echo (Req 1.5)");
    // Verbatim answers under the caller's original question ids (Req 1.3).
    assert_eq!(
        body["answers"]["is_urgent"],
        json!({ "type": "noul", "noul": 0.95 })
    );
    assert_eq!(body["answers"]["department"]["type"], json!("choice"));
    assert_eq!(body["answers"]["department"]["choice"], json!("technical"));
    assert_eq!(
        body["answers"]["department"]["probabilities"]["technical"],
        json!(0.88)
    );
    assert_eq!(
 body["answers"]["sentiment"]["score"],
 json!(1.95)
 );
 // confidence/legend and unknown provider-added fields round-trip
 // verbatim (Req 1.3/1.5 unmodified passthrough).
 assert_eq!(body["answers"]["sentiment"]["confidence"], json!(0.85));
 assert_eq!(
 body["answers"]["sentiment"]["legend"],
 json!({ "0": "Calm", "1": "Frustrated" })
 );
 assert_eq!(
 body["answers"]["department"]["confidence"],
 json!(0.9)
 );
 assert_eq!(
 body["provider_extras"],
 json!({ "serving_region": "eu-west-1", "degraded": false }),
 "unknown provider-added fields survive (Req 1.5)"
 );
 assert_eq!(
 body["usage"]["latency_ms"],
 json!(12),
 "unknown usage fields survive (Req 1.5)"
 );
 assert_eq!(
 body["usage"],
 json!({ "input_tokens": 307, "output_tokens": 20, "latency_ms": 12 }),
 "usage unmodified (Req 4.4)"
 );

    // Upstream received the pinned id plus the caller's state verbatim.
    let posts = upstream_posts(&mock).await;
    assert_eq!(posts.len(), 1, "exactly one evaluate call");
    assert_eq!(posts[0]["model"], json!("jev-1.13.0"));
    assert_eq!(
        posts[0]["state"],
        json!({ "messages": [{ "role": "user", "content": "the printer is on fire" }] })
    );
    assert_eq!(posts[0]["questions"]["is_urgent"]["type"], json!("noul"));

    // Req 4.1-4.3: content-free log row with models, provider, usage, and
    // key attribution — even with body logging enabled.
    let rows = decision_rows(&server);
    let row = rows
        .iter()
        .find(|row| row.status_code == 200)
        .expect("200 decision outcome row");
    assert_eq!(row.requested_model.as_deref(), Some("jev-1.13.0"));
    assert_eq!(row.responded_model.as_deref(), Some("jev-1.13.0"));
    assert_eq!(row.provider, "jev-primary");
    assert_eq!(row.input_tokens, Some(307));
    assert_eq!(row.output_tokens, Some(20));
    assert!(row.virtual_key_id.is_some(), "key attribution (Req 4.3)");
    assert!(
        row.request_body.is_none(),
        "state/questions must never be logged (Req 4.2)"
    );
    assert!(row.response_body.is_none(), "answers must never be logged");
}

// ---------------------------------------------------------------------------
// 2. auto/alias resolution through discovery
// ---------------------------------------------------------------------------

#[tokio::test]
async fn auto_alias_resolves_latest_family_model_via_discovery() {
    let mock = MockServer::start().await;
    mount_models_listing(&mock, &["gpt-4o", "jev-1.12.0", "jev-1.14.0"]).await;
    mount_evaluate(&mock, 200, answers_envelope("jev-1.14.0"), None).await;
    let config = base_config(&[decision_provider("jev-auto", "jev", &mock.uri(), &[])], &[]);
    let server = GatewayServer::new(config, None).await.expect("gateway starts");

    let (status, body) = post_decisions(
        server.build_router(),
        CANONICAL,
        None,
        decision_request("auto"),
    )
    .await;
    assert_eq!(status, StatusCode::OK, "body: {body}");
    assert_eq!(
        body["model"],
        json!("jev-1.14.0"),
        "resolved concrete id echoed — alias-drift visibility (Req 2.4)"
    );

 let listing_calls = mock
 .received_requests()
 .await
 .expect("wiremock request recording")
 .iter()
 .filter(|request| request.url.path().contains("/v1/models"))
 .count();
 assert_eq!(listing_calls, 1, "discovery consulted the listing exactly once");

 // TTL-cached discovery (Req 2.3, 300 s): a second auto request reuses
 // the cached listing — no second `/v1/models` fetch, same mock.
 let (status, body) = post_decisions(
 server.build_router(),
 CANONICAL,
 None,
 decision_request("auto"),
 )
 .await;
 assert_eq!(status, StatusCode::OK, "body: {body}");
 assert_eq!(body["model"], json!("jev-1.14.0"), "same resolved id");
 let listing_calls = mock
 .received_requests()
 .await
 .expect("wiremock request recording")
 .iter()
 .filter(|request| request.url.path().contains("/v1/models"))
 .count();
 assert_eq!(
 listing_calls, 1,
 "cached discovery serves the second auto request — no re-fetch"
 );
}

// ---------------------------------------------------------------------------
// 3. Generic systemone provider: pinned id + auto rejection
// ---------------------------------------------------------------------------

#[tokio::test]
async fn generic_systemone_serves_pinned_id_and_rejects_auto() {
    let mock = MockServer::start().await;
    mount_evaluate(&mock, 200, answers_envelope("drex-2.1"), None).await;
    let config = base_config(
        &[decision_provider("drex-endpoint", "systemone", &mock.uri(), &["drex-2.1"])],
        &[],
    );
    let server = GatewayServer::new(config, None).await.expect("gateway starts");
    let app = server.build_router();

    // Explicit id served pinned (Req 2.6).
    let (status, body) = post_decisions(
        app.clone(),
        CANONICAL,
        None,
        decision_request("drex-2.1"),
    )
    .await;
    assert_eq!(status, StatusCode::OK, "body: {body}");
    assert_eq!(body["model"], json!("drex-2.1"));
    let posts = upstream_posts(&mock).await;
    assert_eq!(posts.len(), 1);
    assert_eq!(posts[0]["model"], json!("drex-2.1"), "id forwarded pinned");

    // `auto` with only generic systemone providers -> 400 naming the
    // restriction (Req 2.5), without contacting the provider again.
    let (status, body) = post_decisions(app, CANONICAL, None, decision_request("auto")).await;
    assert_eq!(status, StatusCode::BAD_REQUEST, "body: {body}");
    assert_eq!(body["error"]["type"], json!("invalid_request_error"));
    assert!(
        body["error"]["message"]
            .as_str()
            .unwrap_or_default()
            .contains("auto"),
        "names the auto restriction: {body}"
    );
    assert_eq!(
        upstream_posts(&mock).await.len(),
        1,
        "auto rejection never contacts the provider"
    );
}

// ---------------------------------------------------------------------------
// 4. Validation 400s name the field, no upstream contact
// ---------------------------------------------------------------------------

#[tokio::test]
async fn validation_failures_name_the_field_and_never_contact_upstream() {
    let mock = MockServer::start().await;
    mount_evaluate(&mock, 200, answers_envelope("jev-1.13.0"), None).await;
    let config = base_config(
        &[decision_provider("jev-primary", "jev", &mock.uri(), &["jev-1.13.0"])],
        &[],
    );
    let server = GatewayServer::new(config, None).await.expect("gateway starts");
    let app = server.build_router();

    let template = decision_request("jev-1.13.0");
    let mut missing_model = template.clone();
    missing_model.as_object_mut().unwrap().remove("model");
    let mut missing_state = template.clone();
    missing_state.as_object_mut().unwrap().remove("state");
    let mut missing_questions = template.clone();
    missing_questions.as_object_mut().unwrap().remove("questions");
    let mut empty_questions = template.clone();
    empty_questions["questions"] = json!({});
    let mut array_questions = template.clone();
    array_questions["questions"] = json!([]);

    for (body, field) in [
        (missing_model, "model"),
        (missing_state, "state"),
        (missing_questions, "questions"),
        (empty_questions, "questions"),
        (array_questions, "questions"),
    ] {
        let (status, response) = post_decisions(app.clone(), CANONICAL, None, body).await;
        assert_eq!(status, StatusCode::BAD_REQUEST, "field {field}: {response}");
        assert_eq!(
            response["error"]["type"],
            json!("invalid_request_error"),
            "{response}"
        );
        let message = response["error"]["message"].as_str().unwrap_or_default();
        assert!(message.contains(field), "must name the field {field}: {response}");
    }
 assert!(
 mock.received_requests()
 .await
 .expect("wiremock request recording")
 .is_empty(),
 "validation failures never contact the provider (Req 1.4)"
 );
}

// ---------------------------------------------------------------------------
// 4b. Handler rejections: Content-Type + malformed JSON (Req 1.4)
// ---------------------------------------------------------------------------

#[tokio::test]
async fn handler_rejects_non_json_content_type_and_malformed_json() {
 let mock = MockServer::start().await;
 mount_evaluate(&mock, 200, answers_envelope("jev-1.13.0"), None).await;
 let config = base_config(
 &[decision_provider("jev-primary", "jev", &mock.uri(), &["jev-1.13.0"])],
 &[],
 );
 let server = GatewayServer::new(config, None).await.expect("gateway starts");
 let app = server.build_router();

 // Non-JSON Content-Type: rejected before parsing.
 let response = app
 .clone()
 .oneshot(
 Request::post(CANONICAL)
 .header("content-type", "text/plain")
 .body(Body::from(decision_request("jev-1.13.0").to_string()))
 .unwrap(),
 )
 .await
 .expect("gateway responded");
 assert_eq!(response.status(), StatusCode::BAD_REQUEST);
 let bytes = axum::body::to_bytes(response.into_body(), 64 * 1024)
 .await
 .expect("bounded body");
 let body: Value = serde_json::from_slice(&bytes).expect("JSON body");
 assert_eq!(body["error"]["type"], json!("invalid_request_error"));
 assert!(
 body["error"]["message"]
 .as_str()
 .unwrap_or_default()
 .contains("Content-Type"),
 "names the Content-Type rule: {body}"
 );

 // Malformed JSON body: rejected as invalid JSON, not a field error.
 let response = app
 .oneshot(
 Request::post(CANONICAL)
 .header("content-type", "application/json")
 .body(Body::from("{not json"))
 .unwrap(),
 )
 .await
 .expect("gateway responded");
 assert_eq!(response.status(), StatusCode::BAD_REQUEST);
 let bytes = axum::body::to_bytes(response.into_body(), 64 * 1024)
 .await
 .expect("bounded body");
 let body: Value = serde_json::from_slice(&bytes).expect("JSON body");
 assert_eq!(body["error"]["type"], json!("invalid_request_error"));
 assert!(
 body["error"]["message"]
 .as_str()
 .unwrap_or_default()
 .contains("JSON"),
 "names the JSON parse failure: {body}"
 );

 // Both rejections stayed local: zero upstream contact.
 assert!(
 mock.received_requests()
 .await
 .expect("wiremock request recording")
 .is_empty(),
 "handler rejections never contact the provider"
 );
 // Each rejection wrote a 400 outcome row with a class (Req 4.3).
 let rows = decision_rows(&server);
 assert_eq!(
 rows.iter().filter(|row| row.status_code == 400).count(),
 2,
 "one 400 outcome row per rejection"
 );
 assert!(rows.iter().all(|row| row.request_body.is_none()));
}

// ---------------------------------------------------------------------------
// 5. 404 unknown model, with and without decision providers
// ---------------------------------------------------------------------------

#[tokio::test]
async fn unknown_model_is_404_with_and_without_decision_providers() {
    let mock = MockServer::start().await;
    mount_evaluate(&mock, 200, answers_envelope("jev-1.13.0"), None).await;
    let config = base_config(
        &[decision_provider("jev-primary", "jev", &mock.uri(), &["jev-1.13.0"])],
        &[],
    );
    let with_provider = GatewayServer::new(config, None)
        .await
        .expect("gateway starts");

    // A configured family does not serve other families' models (D4).
    let (status, body) = post_decisions(
        with_provider.build_router(),
        CANONICAL,
        None,
        decision_request("laya-9.9.9"),
    )
    .await;
 assert_eq!(status, StatusCode::NOT_FOUND, "body: {body}");
 assert_eq!(body["error"]["type"], json!("invalid_request_error"));
 assert!(
 body["error"]["message"]
 .as_str()
 .unwrap_or_default()
 .contains("laya-9.9.9"),
 "names the model: {body}"
 );
 // The 404 outcome row carries status, class, and no bodies (Req 4.3).
 let not_found = decision_rows(&with_provider)
 .into_iter()
 .find(|row| row.status_code == 404)
 .expect("404 decision outcome row");
 assert_eq!(not_found.error_class.as_deref(), Some("upstream_http_404"));
 assert_eq!(not_found.requested_model.as_deref(), Some("laya-9.9.9"));
 assert!(not_found.request_body.is_none());
 assert!(not_found.response_body.is_none());

    // No decision providers at all: same deterministic 404 (Req 3.5).
    let chat_only = base_config(
        &[json!({ "name": "chat-only", "type": "openai", "base_url": mock.uri() })],
        &[],
    );
    let without_provider = GatewayServer::new(chat_only, None)
        .await
        .expect("gateway starts");
    let (status, body) = post_decisions(
        without_provider.build_router(),
        CANONICAL,
        None,
        decision_request("jev-1.13.0"),
    )
    .await;
 assert_eq!(status, StatusCode::NOT_FOUND, "body: {body}");
 assert_eq!(body["error"]["type"], json!("invalid_request_error"));
 // Validation-400 and no-provider-404 rows carry status + class, and
 // never persist the request body (Req 4.2/4.3).
 let rejected = decision_rows(&without_provider)
 .into_iter()
 .find(|row| row.status_code == 404)
 .expect("404 outcome row without decision providers");
 assert!(rejected.error_class.is_some(), "class always set");
 assert!(rejected.request_body.is_none());
}

// ---------------------------------------------------------------------------
// 6. Failover: primary 429 (Retry-After) -> secondary 200
// ---------------------------------------------------------------------------

#[tokio::test]
async fn rate_limited_primary_fails_over_to_secondary() {
    let primary = MockServer::start().await;
    mount_rate_limited(&primary).await;
    let secondary = MockServer::start().await;
    mount_evaluate(&secondary, 200, answers_envelope("jev-1.13.0"), None).await;
    let config = base_config(
        &[
            decision_provider("jev-primary", "jev", &primary.uri(), &["jev-1.13.0"]),
            decision_provider("jev-secondary", "jev", &secondary.uri(), &["jev-1.13.0"]),
        ],
        &[],
    );
 let server = GatewayServer::new(config, None).await.expect("gateway starts");

 let started = Instant::now();
 let (status, body) = post_decisions(
 server.build_router(),
 CANONICAL,
 None,
 decision_request("jev-1.13.0"),
 )
 .await;
 assert_eq!(status, StatusCode::OK, "body: {body}");
 assert_eq!(body["model"], json!("jev-1.13.0"));
 assert_eq!(
 upstream_posts(&primary).await.len(),
 1,
 "rate-limited primary tried exactly once"
 );
 assert!(
 started.elapsed() < Duration::from_secs(5),
 "429 failover is non-blocking despite Retry-After: 60 (took {:?})",
 started.elapsed()
 );

    // The outcome row attributes the answer to the secondary (Req 2.7).
    let rows = decision_rows(&server);
    let row = rows
        .iter()
        .find(|row| row.status_code == 200)
        .expect("200 decision outcome row");
    assert_eq!(row.provider, "jev-secondary");
}

// ---------------------------------------------------------------------------
// 7. Hung primary -> timeout-class failover within the per-try budget
// ---------------------------------------------------------------------------

#[tokio::test]
async fn hung_primary_times_out_and_fails_over() {
    let hung = MockServer::start().await;
    mount_evaluate(&hung, 200, answers_envelope("jev-1.13.0"), Some(Duration::from_secs(3))).await;
    let secondary = MockServer::start().await;
    mount_evaluate(&secondary, 200, answers_envelope("jev-1.13.0"), None).await;
    let mut config = base_config(
        &[
            decision_provider("jev-hung", "jev", &hung.uri(), &["jev-1.13.0"]),
            decision_provider("jev-fast", "jev", &secondary.uri(), &["jev-1.13.0"]),
        ],
        &[],
    );
    // Whole-try budget of 1s (Req 3.1): the 3s hang must not be awaited.
    config.providers[0].timeout_seconds = 1;
    let server = GatewayServer::new(config, None).await.expect("gateway starts");

    let started = Instant::now();
    let (status, body) = post_decisions(
        server.build_router(),
        CANONICAL,
        None,
        decision_request("jev-1.13.0"),
    )
    .await;
    assert_eq!(status, StatusCode::OK, "body: {body}");
    assert!(
        started.elapsed() < Duration::from_millis(2500),
        "timeout budget fired before the 3s hang; took {:?}",
        started.elapsed()
    );

    let rows = decision_rows(&server);
    let row = rows
        .iter()
        .find(|row| row.status_code == 200)
        .expect("200 decision outcome row");
    assert_eq!(row.provider, "jev-fast");
}

// ---------------------------------------------------------------------------
// 8. Both candidates fail -> aggregated 502 with per-attempt details
// ---------------------------------------------------------------------------

#[tokio::test]
async fn exhausted_failover_returns_aggregated_attempts() {
    let first = MockServer::start().await;
    mount_rate_limited(&first).await;
    let second = MockServer::start().await;
    mount_rate_limited(&second).await;
    let config = base_config(
        &[
            decision_provider("jev-first", "jev", &first.uri(), &["jev-1.13.0"]),
            decision_provider("jev-second", "jev", &second.uri(), &["jev-1.13.0"]),
        ],
        &[],
    );
    let server = GatewayServer::new(config, None).await.expect("gateway starts");

    let (status, body) = post_decisions(
        server.build_router(),
        CANONICAL,
        None,
        decision_request("jev-1.13.0"),
    )
    .await;
    assert_eq!(status, StatusCode::BAD_GATEWAY, "body: {body}");
    assert_eq!(body["error"]["type"], json!("all_providers_failed"));
    let attempts = body["error"]["attempts"].as_array().expect("attempts array");
    assert_eq!(attempts.len(), 2, "every failed try listed: {body}");
    assert_eq!(attempts[0]["provider"], json!("jev-first"));
    assert_eq!(attempts[0]["status_code"], json!(429));
    assert_eq!(attempts[1]["provider"], json!("jev-second"));
    assert_eq!(attempts[1]["status_code"], json!(429));

    // The outcome row carries the aggregated failure class (Req 4.3).
    let rows = decision_rows(&server);
    let row = rows
        .iter()
        .find(|row| row.status_code == 502)
        .expect("502 decision outcome row");
    assert_eq!(row.error_class.as_deref(), Some("transient"));
}

// ---------------------------------------------------------------------------
// 8b. Terminal errors must NOT fail over (Req 3.3)
// ---------------------------------------------------------------------------

#[tokio::test]
async fn terminal_primary_error_returns_without_contacting_secondary() {
 let primary = MockServer::start().await;
 // 413 (PayloadTooLarge) is a deterministic class: exact status, no
 // retry, no failover — even with a healthy secondary configured.
 mount_evaluate(&primary, 413, json!({ "error": "too big" }), None).await;
 let secondary = MockServer::start().await;
 mount_evaluate(&secondary, 200, answers_envelope("jev-1.13.0"), None).await;
 let config = base_config(
 &[
 decision_provider("jev-terminal", "jev", &primary.uri(), &["jev-1.13.0"]),
 decision_provider("jev-backup", "jev", &secondary.uri(), &["jev-1.13.0"]),
 ],
 &[],
 );
 let server = GatewayServer::new(config, None).await.expect("gateway starts");

 let (status, body) = post_decisions(
 server.build_router(),
 CANONICAL,
 None,
 decision_request("jev-1.13.0"),
 )
 .await;
 assert_eq!(status, StatusCode::PAYLOAD_TOO_LARGE, "body: {body}");
 assert_eq!(
 body["error"]["message"]
 .as_str()
 .unwrap_or_default()
 .contains("413"),
 true,
 "terminal 413 shape reaches the client: {body}"
 );
 assert!(
 upstream_posts(&secondary).await.is_empty(),
 "terminal errors never fail over to the secondary (Req 3.3)"
 );
 assert_eq!(
 upstream_posts(&primary).await.len(),
 1,
 "terminal 413 is not retried in-provider either"
 );
}

#[tokio::test]
async fn upstream_status_classes_map_to_spec_statuses() {
    for (upstream_status, expected) in [(401_u16, 502_u16), (413, 413), (422, 422), (400, 400)] {
        let mock = MockServer::start().await;
        mount_evaluate(&mock, upstream_status, json!({ "error": "upstream said no" }), None).await;
        let config = base_config(
            &[decision_provider("jev-primary", "jev", &mock.uri(), &["jev-1.13.0"])],
            &[],
        );
        let server = GatewayServer::new(config, None).await.expect("gateway starts");

        let (status, body) = post_decisions(
            server.build_router(),
            CANONICAL,
            None,
            decision_request("jev-1.13.0"),
        )
        .await;
        assert_eq!(
            status,
            StatusCode::from_u16(expected).unwrap(),
            "upstream {upstream_status}: {body}"
        );
        assert_eq!(
            body["error"]["type"],
            json!("provider_error"),
            "deterministic class surfaces directly (no all_providers_failed shape): {body}"
        );
        let message = body["error"]["message"].as_str().unwrap_or_default();
        assert!(
            message.contains(&format!("HTTP {upstream_status}")),
            "identifies the upstream status: {body}"
        );
        assert_eq!(
            upstream_posts(&mock).await.len(),
            1,
            "deterministic classes never fail over (Req 3.3)"
        );
    }
}

// ---------------------------------------------------------------------------
// 10. Virtual-key enforcement: required 401 + model_access denial
// ---------------------------------------------------------------------------

#[tokio::test]
async fn virtual_key_enforcement_401_and_model_access_denial() {
    let mock = MockServer::start().await;
    mount_evaluate(&mock, 200, answers_envelope("jev-1.13.0"), None).await;
    let mut config = base_config(
        &[decision_provider("jev-primary", "jev", &mock.uri(), &["jev-1.13.0"])],
        &[],
    );
    config.virtual_keys.enforcement = EnforcementMode::Required;
    let server = GatewayServer::new(config, None).await.expect("gateway starts");
    let app = server.build_router();

    // required + no key -> 401 legacy envelope (non-messages route shape).
    let (status, body) = post_decisions(
        app.clone(),
        CANONICAL,
        None,
        decision_request("jev-1.13.0"),
    )
    .await;
    assert_eq!(status, StatusCode::UNAUTHORIZED, "body: {body}");
    assert!(body["error"].is_string(), "legacy envelope: {body}");
    assert!(upstream_posts(&mock).await.is_empty());

    // model_access denial -> 403 naming the model and the permitted scope.
    let key = create_virtual_key(&server, Some(vec!["unrelated-model".to_string()])).await;
    let (status, body) = post_decisions(
        app,
        CANONICAL,
        Some(&key),
        decision_request("jev-1.13.0"),
    )
    .await;
    assert_eq!(status, StatusCode::FORBIDDEN, "body: {body}");
    assert_eq!(body["model"], json!("jev-1.13.0"));
    assert_eq!(body["allowed"], json!(["unrelated-model"]));
    assert!(
        upstream_posts(&mock).await.is_empty(),
        "denied request never reaches the provider"
    );
}

// ---------------------------------------------------------------------------
// 11. /v1/systemone alias behaves identically
// ---------------------------------------------------------------------------

#[tokio::test]
async fn systemone_alias_route_matches_canonical_route() {
    let mock = MockServer::start().await;
    mount_evaluate(&mock, 200, answers_envelope("jev-1.13.0"), None).await;
    let config = base_config(
        &[decision_provider("jev-primary", "jev", &mock.uri(), &["jev-1.13.0"])],
        &[],
    );
    let server = GatewayServer::new(config, None).await.expect("gateway starts");

    let (status, body) = post_decisions(
        server.build_router(),
        ALIAS,
        None,
        decision_request("jev-1.13.0"),
    )
    .await;
    assert_eq!(status, StatusCode::OK, "body: {body}");
    assert_eq!(body["model"], json!("jev-1.13.0"));
    assert_eq!(body["usage"]["input_tokens"], json!(307));

    // Req 4.1: the alias logs under the single canonical path.
    let rows = decision_rows(&server);
    assert!(
        rows.iter().any(|row| row.status_code == 200),
        "alias outcome logged under {CANONICAL}"
    );
}

// ---------------------------------------------------------------------------
// 12. Chat completions for a decision model id -> existing not-found shape
// ---------------------------------------------------------------------------

#[tokio::test]
async fn chat_completions_rejects_decision_model_id() {
    let mock = MockServer::start().await;
    let config = base_config(
        &[decision_provider("jev-primary", "jev", &mock.uri(), &["jev-1.13.0"])],
        &[],
    );
    let server = GatewayServer::new(config, None).await.expect("gateway starts");

    let request = Request::post("/v1/chat/completions")
        .header("content-type", "application/json")
        .body(Body::from(
            json!({ "model": "jev-1.13.0", "messages": [{ "role": "user", "content": "hi" }] })
                .to_string(),
        ))
        .unwrap();
    let response = server
        .build_router()
        .oneshot(request)
        .await
        .expect("gateway responded");
    assert_eq!(response.status(), StatusCode::BAD_REQUEST);
    let bytes = axum::body::to_bytes(response.into_body(), 64 * 1024)
        .await
        .expect("bounded body");
    let body: Value = serde_json::from_slice(&bytes).expect("JSON body");
    assert_eq!(body["error"]["type"], json!("invalid_request_error"));
    assert!(
        body["error"]["message"]
            .as_str()
            .unwrap_or_default()
            .contains("jev-1.13.0"),
        "existing chat not-found shape names the model (Req 5.3): {body}"
    );
    assert!(
        mock.received_requests()
            .await
            .expect("wiremock request recording")
            .is_empty(),
        "decision models never answer through the chat front door"
    );
}

// ---------------------------------------------------------------------------
// 13. /v1/models: manual_models listed (Req 5.2)
// ---------------------------------------------------------------------------

#[tokio::test]
async fn models_listing_includes_decision_manual_models() {
    let mock = MockServer::start().await;
    let config = base_config(
        &[
            decision_provider("jev-manual", "jev", &mock.uri(), &["jev-1.13.0"]),
            decision_provider("drex-endpoint", "systemone", &mock.uri(), &["drex-2.1"]),
            // Family provider WITHOUT manual_models contributes no entry
            // (Req 5.2 MAY-omit rule): discovery-only providers advertise
            // nothing static.
            decision_provider("jev-discovery-only", "jev", &mock.uri(), &[]),
        ],
        &[],
    );
    let server = GatewayServer::new(config, None).await.expect("gateway starts");

    let response = server
        .build_router()
        .oneshot(Request::get("/v1/models").body(Body::empty()).unwrap())
        .await
        .expect("gateway responded");
    assert_eq!(response.status(), StatusCode::OK);
    let bytes = axum::body::to_bytes(response.into_body(), 1024 * 1024)
        .await
        .expect("bounded body");
    let body: Value = serde_json::from_slice(&bytes).expect("JSON body");
    let data = body["data"].as_array().expect("data array");

    let ids: Vec<&str> = data
        .iter()
        .map(|entry| entry["id"].as_str().unwrap_or_default())
        .collect();
    assert!(
        ids.contains(&"jev-1.13.0"),
        "family manual_models listed: {ids:?}"
    );
    assert!(
        ids.contains(&"drex-2.1"),
        "systemone manual_models listed: {ids:?}"
    );
    assert_eq!(
        ids.iter().filter(|id| **id == "jev-1.13.0").count(),
        1,
        "each manual id appears exactly once even with two jev providers"
    );
    // The discovery-only provider contributes no entry: its ids are only
    // known after consulting its (mock-absent) listing.
    assert!(
        mock.received_requests()
            .await
            .expect("wiremock request recording")
            .is_empty(),
        "models listing is a config projection — no upstream contact"
    );
}

// ---------------------------------------------------------------------------
// 14. Hot-reload adds a decision provider without restart
// ---------------------------------------------------------------------------

#[tokio::test]
async fn hot_reload_adds_decision_provider_without_restart() {
    let mock = MockServer::start().await;
    mount_evaluate(&mock, 200, answers_envelope("jev-1.13.0"), None).await;
    // Start with a jev provider only.
    let mut config = base_config(
        &[decision_provider("jev-primary", "jev", &mock.uri(), &["jev-1.13.0"])],
        &[],
    );
    let server = GatewayServer::new(config.clone(), None)
        .await
        .expect("gateway starts");

    // Unknown before reload.
    let (status, _) = post_decisions(
        server.build_router(),
        CANONICAL,
        None,
        decision_request("laya-2.0.0"),
    )
    .await;
    assert_eq!(status, StatusCode::NOT_FOUND);

    // Reload with a laya provider added (same mock serves both).
    let laya_mock = MockServer::start().await;
    mount_evaluate(&laya_mock, 200, answers_envelope("laya-2.0.0"), None).await;
    config.providers.push({
        let mut provider = config.providers[0].clone();
        provider.name = "laya-primary".to_string();
        provider.provider_type = "laya".to_string();
        provider.base_url = Some(laya_mock.uri());
        provider.manual_models = vec!["laya-2.0.0".to_string()];
        provider
    });
 apply_runtime_config_update(&server.state, config.clone()).await;

 let (status, body) = post_decisions(
 server.build_router(),
 CANONICAL,
 None,
 decision_request("laya-2.0.0"),
 )
 .await;
 assert_eq!(status, StatusCode::OK, "body: {body}");
 assert_eq!(body["model"], json!("laya-2.0.0"));

 // Stale-client regression (Req 2.8): reload the SAME provider name with
 // a changed base_url and key — the cached transport client must not keep
 // serving the old endpoint or credential.
 let point_b = MockServer::start().await;
 mount_evaluate(&point_b, 200, answers_envelope("jev-1.13.0"), None).await;
 let first_post_count = upstream_posts(&mock).await.len();
 config.providers[0].base_url = Some(point_b.uri());
 config.providers[0].resolved_api_key = Some("rotated-decision-key".to_string());
 apply_runtime_config_update(&server.state, config).await;

 let (status, body) = post_decisions(
 server.build_router(),
 CANONICAL,
 None,
 decision_request("jev-1.13.0"),
 )
 .await;
 assert_eq!(status, StatusCode::OK, "body: {body}");
 assert_eq!(body["model"], json!("jev-1.13.0"));
 assert_eq!(
 upstream_posts(&point_b).await.len(),
 1,
 "reload re-pointed the provider at the new endpoint"
 );
 assert_eq!(
 upstream_posts(&mock).await.len(),
 first_post_count,
 "zero further posts to the old endpoint after reload"
 );
 // The per-call key (F5) travels as the Bearer credential — the rotated
 // value must reach the new endpoint, proving the stale cached client
 // (old base_url + construction-time key) was dropped.
 let received = point_b
 .received_requests()
 .await
 .expect("wiremock request recording");
 let auth = received
 .last()
 .expect("one request to the new endpoint")
 .headers
 .get("authorization")
 .and_then(|value| value.to_str().ok())
 .unwrap_or_default();
 assert_eq!(auth, "Bearer rotated-decision-key");
}

// ---------------------------------------------------------------------------
// 15. Decision-provider outage leaves chat completions unaffected (NFR)
// ---------------------------------------------------------------------------

#[tokio::test]
async fn decision_provider_outage_does_not_affect_chat_completions() {
    // Decision provider points at a dead port (connection refused).
    let config = base_config(
        &[
            decision_provider("jev-dead", "jev", "http://127.0.0.1:1", &["jev-1.13.0"]),
            json!({
                "name": "chat-alive",
                "type": "openai",
                "base_url": "http://127.0.0.1:1", // replaced below with live mock
            }),
        ],
        &[json!({ "name": "chat-group", "models": [{ "provider": "chat-alive", "model": "gpt-4" }] })],
    );
    // Re-point the chat provider at a live mock via a fresh build.
    let chat_mock = MockServer::start().await;
    mount_chat_completion(&chat_mock).await;
    let mut config = config;
    config.providers[1].base_url = Some(chat_mock.uri());
    config.providers[1].resolved_api_key = Some("test-decision-key".to_string());
    let server = GatewayServer::new(config, None).await.expect("gateway starts");

    // Decision request fails against the dead provider...
    let (status, body) = post_decisions(
        server.build_router(),
        CANONICAL,
        None,
        decision_request("jev-1.13.0"),
    )
    .await;
 assert_eq!(
 status,
 StatusCode::BAD_GATEWAY,
 "dead decision provider surfaces a gateway error: {status} {body}"
 );

    // ...while chat completions through the healthy provider still succeed.
    let request = Request::post("/v1/chat/completions")
        .header("content-type", "application/json")
        .body(Body::from(
            json!({
                "model": "chat-group",
                "messages": [{ "role": "user", "content": "still there?" }]
            })
            .to_string(),
        ))
        .unwrap();
    let response = server
        .build_router()
        .oneshot(request)
        .await
        .expect("gateway responded");
    assert_eq!(
        response.status(),
        StatusCode::OK,
        "chat path isolated from the decision-provider outage (NFR Reliability)"
    );
}

// ---------------------------------------------------------------------------
// 16. NFR Performance: mean gateway overhead under budget
// ---------------------------------------------------------------------------

/// Wall-clock budget test (repo convention: performance.rs pattern) —
/// `#[ignore]` in the default profile; run with `--ignored`.
#[tokio::test]
#[ignore = "wall-clock overhead budget: run perf suite with --ignored"]
async fn mean_gateway_overhead_under_budget() {
    let mock = MockServer::start().await;
    mount_evaluate(&mock, 200, answers_envelope("jev-1.13.0"), None).await;
    let config = base_config(
        &[decision_provider("jev-primary", "jev", &mock.uri(), &["jev-1.13.0"])],
        &[],
    );
    let server = GatewayServer::new(config, None).await.expect("gateway starts");
    let app = server.build_router();
    let body = decision_request("jev-1.13.0");

    // Warm-up (JIT-free but pays first-connection pool setup, DNS-ish
    // loopback resolution, logger schema init).
    for _ in 0..5 {
        let (status, _) = post_decisions(app.clone(), CANONICAL, None, body.clone()).await;
        assert_eq!(status, StatusCode::OK);
    }

    // 20 sequential timed round-trips. The mock answers instantly, so the
    // measured time is gateway overhead (parse/route/admit/relay/log) with
    // only residual loopback I/O.
    let iterations = 20;
    let mut total = Duration::ZERO;
    for _ in 0..iterations {
        let started = Instant::now();
        let (status, response) = post_decisions(app.clone(), CANONICAL, None, body.clone()).await;
        let elapsed = started.elapsed();
        assert_eq!(status, StatusCode::OK, "body: {response}");
        total += elapsed;
    }
 let mean = total / iterations;
 assert!(
 mean < Duration::from_millis(OVERHEAD_BUDGET_MS),
 "mean gateway overhead {mean:?} exceeded budget {OVERHEAD_BUDGET_MS} ms \
 (spec NFR Performance p95; sequential oneshot mean over {iterations} tries)"
 );
}

// ---------------------------------------------------------------------------
// 17. Whitespace-padded model ids trim to their provider
// ---------------------------------------------------------------------------

#[tokio::test]
async fn whitespace_padded_model_ids_route_after_trim() {
 // Intended contract: the handler trims the requested model at parse
 // (decisions.rs `parse_decision_request`), so padded ids pass
 // validation AND route — `" jev-1.13.0"` serves the pinned jev model,
 // `" auto"` resolves like `auto`. `"  "` (trim-empty) is a 400 naming
 // the model field.
 let jev = MockServer::start().await;
 mount_models_listing(&jev, &["jev-1.13.0"]).await;
 mount_evaluate(&jev, 200, answers_envelope("jev-1.13.0"), None).await;
 let laya = MockServer::start().await;
 mount_models_listing(&laya, &["laya-2.4.0"]).await;
 mount_evaluate(&laya, 200, answers_envelope("laya-2.4.0"), None).await;
 // `auto` binds to the FIRST configured family (D4), so laya leads.
 let config = base_config(
 &[
 decision_provider("laya-primary", "laya", &laya.uri(), &[]),
 decision_provider("jev-primary", "jev", &jev.uri(), &["jev-1.13.0"]),
 ],
 &[],
 );
 let server = GatewayServer::new(config, None).await.expect("gateway starts");
 let app = server.build_router();

 // Pinned id with leading/trailing whitespace routes to the jev provider.
 let (status, body) = post_decisions(app.clone(), CANONICAL, None, decision_request(" jev-1.13.0 ")).await;
 assert_eq!(status, StatusCode::OK, "body: {body}");
 assert_eq!(body["model"], json!("jev-1.13.0"));
 let posts = upstream_posts(&jev).await;
 assert_eq!(posts.len(), 1);
 assert_eq!(
 posts[0]["model"],
 json!("jev-1.13.0"),
 "trimmed id forwarded upstream"
 );

 // Padded `auto` resolves through discovery like its unpadded form.
 let (status, body) = post_decisions(app.clone(), CANONICAL, None, decision_request(" auto ")).await;
 assert_eq!(status, StatusCode::OK, "body: {body}");
 assert_eq!(body["model"], json!("laya-2.4.0"));

 // Padded bare family alias routes through discovery too.
 let (status, body) = post_decisions(app.clone(), CANONICAL, None, decision_request(" jev ")).await;
 assert_eq!(status, StatusCode::OK, "body: {body}");
 assert_eq!(body["model"], json!("jev-1.13.0"));

 // Whitespace-only model: rejected after trim, naming the field.
 let (status, body) = post_decisions(app, CANONICAL, None, decision_request("   ")).await;
 assert_eq!(status, StatusCode::BAD_REQUEST, "body: {body}");
 assert!(
 body["error"]["message"]
 .as_str()
 .unwrap_or_default()
 .contains("model"),
 "trim-empty model still names the field: {body}"
 );
}
