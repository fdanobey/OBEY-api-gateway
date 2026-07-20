//! Integration tests for the Presidio guardrail provider's HTTP behavior
//! (task 7.4).
//!
//! These tests stand up a mock Presidio-compatible `/analyze` endpoint with
//! `wiremock` and exercise [`PresidioProvider::analyze`] against it end to end,
//! following the repo's existing wiremock conventions. They cover:
//!
//! - the request payload carries the analyzed `text` and the configured
//!   `entities` list (Req 6.1);
//! - a well-formed response is parsed and mapped to findings (Req 6.1);
//! - every failure mode — server unreachable, non-2xx status, malformed /
//!   bad-schema response body, and timeout — surfaces the appropriate
//!   [`GuardrailProviderError`], and the engine's failure-policy wrapper then
//!   maps that error onto `fail_open` (skip) / `fail_close` (halt) (Req 6.4).

use std::time::Duration;

use ai_gateway::guardrail::provider::{
    analyze_with_policy, GuardrailProvider, GuardrailProviderError, StageDisposition,
};
use ai_gateway::guardrail::providers::presidio::PresidioProvider;
use ai_gateway::guardrail::FailurePolicy;

use serde_json::{json, Value};
use wiremock::matchers::{method, path};
use wiremock::{Mock, MockServer, ResponseTemplate};

/// Entity types configured for detection in these tests.
fn configured_entities() -> Vec<String> {
    vec!["EMAIL_ADDRESS".to_string(), "US_SSN".to_string()]
}

/// Build a provider pointed at `endpoint` with the standard test config.
fn provider_for(endpoint: String, timeout_secs: Option<u64>) -> PresidioProvider {
    provider_for_language(endpoint, None, timeout_secs)
}

fn provider_for_language(
    endpoint: String,
    language: Option<&str>,
    timeout_secs: Option<u64>,
) -> PresidioProvider {
    PresidioProvider::new(
        reqwest::Client::new(),
        endpoint,
        configured_entities(),
        language.map(str::to_string),
        Some(0.5),
        timeout_secs,
    )
}

// ---------------------------------------------------------------------------
// Req 6.1 — request payload + successful parsing
// ---------------------------------------------------------------------------

#[tokio::test]
async fn request_payload_contains_text_and_configured_entities() {
    let server = MockServer::start().await;

    // Empty findings response; we only care about the outgoing request here.
    Mock::given(method("POST"))
        .and(path("/analyze"))
        .respond_with(ResponseTemplate::new(200).set_body_json(json!([])))
        .mount(&server)
        .await;

    let provider = provider_for(format!("{}/analyze", server.uri()), None);
    let content = "contact me at jane@example.com";
    let findings = provider.analyze(content).await.expect("analyze succeeds");
    assert!(findings.is_empty(), "empty upstream response → no findings");

    // Inspect the request wiremock actually received (Req 6.1).
    let requests = server.received_requests().await.expect("recording enabled");
    assert_eq!(requests.len(), 1, "exactly one analyze request sent");

    let body: Value = serde_json::from_slice(&requests[0].body).expect("payload is JSON");
    assert_eq!(
        body["text"], content,
        "payload includes the analyzed text (Req 6.1)"
    );
    assert_eq!(
        body["entities"],
        json!(["EMAIL_ADDRESS", "US_SSN"]),
        "payload includes the configured entity list (Req 6.1)"
    );
    assert_eq!(body["language"], "en", "payload includes Presidio language");
}

#[tokio::test]
async fn request_payload_uses_configured_language() {
    let server = MockServer::start().await;
    Mock::given(method("POST"))
        .and(path("/analyze"))
        .respond_with(ResponseTemplate::new(200).set_body_json(json!([])))
        .mount(&server)
        .await;

    let provider = provider_for_language(format!("{}/analyze", server.uri()), Some("es"), None);
    provider.analyze("correo").await.expect("analyze succeeds");

    let requests = server.received_requests().await.expect("recording enabled");
    let body: Value = serde_json::from_slice(&requests[0].body).expect("payload is JSON");
    assert_eq!(body["language"], "es");
}

#[tokio::test]
async fn successful_response_maps_entities_to_findings() {
    let server = MockServer::start().await;

    // Two configured hits (kept) plus one unconfigured type and one
    // below-threshold hit (both dropped by the provider's filter).
    let response = json!([
        { "entity_type": "EMAIL_ADDRESS", "start": 14, "end": 30, "score": 0.99 },
        { "entity_type": "US_SSN",        "start": 40, "end": 51, "score": 0.5  },
        { "entity_type": "PERSON",        "start": 0,  "end": 4,  "score": 0.97 },
        { "entity_type": "EMAIL_ADDRESS", "start": 60, "end": 70, "score": 0.10 }
    ]);
    Mock::given(method("POST"))
        .and(path("/analyze"))
        .respond_with(ResponseTemplate::new(200).set_body_json(response))
        .mount(&server)
        .await;

    let provider = provider_for(format!("{}/analyze", server.uri()), None);
    let findings = provider
        .analyze("some content with pii inside of it here")
        .await
        .expect("analyze succeeds");

    assert_eq!(
        findings.len(),
        2,
        "only configured, at/above-threshold hits"
    );
    assert_eq!(findings[0].entity_label, "EMAIL_ADDRESS");
    assert_eq!((findings[0].start, findings[0].end), (14, 30));
    assert_eq!(findings[0].score, Some(0.99));
    assert_eq!(findings[1].entity_label, "US_SSN");
    assert_eq!(findings[1].score, Some(0.5));
}

// ---------------------------------------------------------------------------
// Req 6.4 — failure handling and failure-policy application
// ---------------------------------------------------------------------------

/// Assert the failure-policy wrapper maps a provider failure to the expected
/// disposition for each policy (Req 6.4): `fail_open` skips, `fail_close` halts.
async fn assert_failure_policy_applies(provider: &PresidioProvider) {
    let open = analyze_with_policy(
        provider,
        "content",
        Duration::from_secs(30),
        FailurePolicy::FailOpen,
    )
    .await;
    assert_eq!(
        open,
        StageDisposition::SkipOpen,
        "fail_open → skip the stage (Req 6.4)"
    );

    let close = analyze_with_policy(
        provider,
        "content",
        Duration::from_secs(30),
        FailurePolicy::FailClose,
    )
    .await;
    assert_eq!(
        close,
        StageDisposition::HaltClosed,
        "fail_close → halt the pipeline (Req 6.4)"
    );
}

#[tokio::test]
async fn unreachable_endpoint_surfaces_error_and_applies_policy() {
    // Port 1 refuses connections → transport failure surfaces as `Unreachable`.
    let provider = provider_for("http://127.0.0.1:1/analyze".to_string(), Some(2));

    let err = provider
        .analyze("content")
        .await
        .expect_err("unreachable endpoint must error");
    assert!(
        matches!(err, GuardrailProviderError::Unreachable(_)),
        "expected Unreachable, got {err:?}"
    );

    assert_failure_policy_applies(&provider).await;
}

#[tokio::test]
async fn non_2xx_status_surfaces_upstream_status_and_applies_policy() {
    let server = MockServer::start().await;
    Mock::given(method("POST"))
        .and(path("/analyze"))
        .respond_with(ResponseTemplate::new(500).set_body_string("internal error"))
        .mount(&server)
        .await;

    let provider = provider_for(format!("{}/analyze", server.uri()), None);

    let err = provider
        .analyze("content")
        .await
        .expect_err("non-2xx must error");
    match err {
        GuardrailProviderError::UpstreamStatus { status, .. } => assert_eq!(status, 500),
        other => panic!("expected UpstreamStatus, got {other:?}"),
    }

    assert_failure_policy_applies(&provider).await;
}

#[tokio::test]
async fn malformed_response_surfaces_malformed_and_applies_policy() {
    let server = MockServer::start().await;
    // 200 OK but the body does not conform to the expected `[PresidioEntity]`
    // schema (an object, not an array) → parse failure (Req 6.4).
    Mock::given(method("POST"))
        .and(path("/analyze"))
        .respond_with(ResponseTemplate::new(200).set_body_json(json!({ "unexpected": "shape" })))
        .mount(&server)
        .await;

    let provider = provider_for(format!("{}/analyze", server.uri()), None);

    let err = provider
        .analyze("content")
        .await
        .expect_err("bad schema must error");
    assert!(
        matches!(err, GuardrailProviderError::MalformedResponse(_)),
        "expected MalformedResponse, got {err:?}"
    );

    assert_failure_policy_applies(&provider).await;
}

#[tokio::test]
async fn timeout_surfaces_error_and_applies_policy() {
    let server = MockServer::start().await;
    // Delay well beyond the clamped 1 s request timeout so the request-level
    // timeout fires before a response is produced (Req 6.4).
    Mock::given(method("POST"))
        .and(path("/analyze"))
        .respond_with(
            ResponseTemplate::new(200)
                .set_body_json(json!([]))
                .set_delay(Duration::from_secs(10)),
        )
        .mount(&server)
        .await;

    // timeout_secs = 1 is the minimum after clamping (Req 6.5).
    let provider = provider_for(format!("{}/analyze", server.uri()), Some(1));

    let err = provider
        .analyze("content")
        .await
        .expect_err("timed-out request must error");
    // A reqwest request timeout surfaces through the transport as `Unreachable`.
    assert!(
        matches!(err, GuardrailProviderError::Unreachable(_)),
        "expected Unreachable (timeout), got {err:?}"
    );

    assert_failure_policy_applies(&provider).await;
}
