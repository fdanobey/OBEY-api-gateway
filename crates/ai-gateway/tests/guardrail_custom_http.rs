//! Integration tests for the `custom_http` guardrail provider (Req 8.4, task 7.7).
//!
//! Exercises the failure-policy paths and happy path of
//! [`CustomHttpProvider::analyze`] against a mock HTTP endpoint (wiremock):
//!
//! - non-2xx HTTP status  → [`GuardrailProviderError::UpstreamStatus`]
//! - malformed / schema-mismatch body → [`GuardrailProviderError::MalformedResponse`]
//! - valid documented-schema body → parses into the expected findings
//!
//! `CustomHttpProvider::new` is public, so these tests drive the provider
//! directly without standing up the full gateway.

use ai_gateway::guardrail::provider::GuardrailProviderError;
use ai_gateway::guardrail::providers::custom_http::CustomHttpProvider;
use ai_gateway::guardrail::GuardrailProvider;

use reqwest::Client;
use wiremock::matchers::{method, path};
use wiremock::{Mock, MockServer, ResponseTemplate};

/// Build a provider pointed at the mock server's `/scan` endpoint.
fn provider_for(server: &MockServer) -> CustomHttpProvider {
    CustomHttpProvider::new(Client::new(), format!("{}/scan", server.uri()), Some(5))
}

// ---------------------------------------------------------------------------
// Happy path: valid documented-schema response parses into findings (Req 8.3).
// ---------------------------------------------------------------------------

#[tokio::test]
async fn custom_http_valid_response_parses_findings() {
    let server = MockServer::start().await;

    let body = serde_json::json!({
        "findings": [
            { "entity_label": "API_KEY", "start": 0, "end": 5, "score": 0.97 },
            { "entity_label": "EMAIL", "start": 10, "end": 20 }
        ]
    });

    Mock::given(method("POST"))
        .and(path("/scan"))
        .respond_with(ResponseTemplate::new(200).set_body_json(body))
        .expect(1)
        .mount(&server)
        .await;

    let provider = provider_for(&server);
    let findings = provider
        .analyze("some content to scan")
        .await
        .expect("valid documented-schema response should parse");

    assert_eq!(findings.len(), 2);
    assert_eq!(findings[0].entity_label, "API_KEY");
    assert_eq!(findings[0].start, 0);
    assert_eq!(findings[0].end, 5);
    assert_eq!(findings[0].score, Some(0.97));
    assert_eq!(findings[1].entity_label, "EMAIL");
    assert_eq!(findings[1].score, None);
}

// ---------------------------------------------------------------------------
// Failure path: non-2xx HTTP status → UpstreamStatus (Req 8.4).
// ---------------------------------------------------------------------------

#[tokio::test]
async fn custom_http_non_2xx_status_is_upstream_status_error() {
    let server = MockServer::start().await;

    Mock::given(method("POST"))
        .and(path("/scan"))
        .respond_with(ResponseTemplate::new(500).set_body_string("scanner exploded"))
        .expect(1)
        .mount(&server)
        .await;

    let provider = provider_for(&server);
    let err = provider
        .analyze("content")
        .await
        .expect_err("non-2xx status should be a provider failure");

    match err {
        GuardrailProviderError::UpstreamStatus { status, message } => {
            assert_eq!(status, 500);
            assert!(
                message.contains("scanner exploded"),
                "message should carry the upstream body, got: {message}"
            );
        }
        other => panic!("expected UpstreamStatus, got {other:?}"),
    }
}

// ---------------------------------------------------------------------------
// Failure path: 200 but schema-mismatch body → MalformedResponse (Req 8.4).
// ---------------------------------------------------------------------------

#[tokio::test]
async fn custom_http_schema_mismatch_is_malformed_response() {
    let server = MockServer::start().await;

    // `start` must be an integer per the documented schema; a string is a
    // schema mismatch that must surface as MalformedResponse.
    let body = r#"{ "findings": [ { "entity_label": "X", "start": "nope", "end": 5 } ] }"#;

    Mock::given(method("POST"))
        .and(path("/scan"))
        .respond_with(ResponseTemplate::new(200).set_body_string(body))
        .expect(1)
        .mount(&server)
        .await;

    let provider = provider_for(&server);
    let err = provider
        .analyze("content")
        .await
        .expect_err("schema mismatch should be a provider failure");

    assert!(
        matches!(err, GuardrailProviderError::MalformedResponse(_)),
        "expected MalformedResponse, got {err:?}"
    );
}

// ---------------------------------------------------------------------------
// Failure path: 200 with non-JSON body → MalformedResponse (Req 8.4).
// ---------------------------------------------------------------------------

#[tokio::test]
async fn custom_http_non_json_body_is_malformed_response() {
    let server = MockServer::start().await;

    Mock::given(method("POST"))
        .and(path("/scan"))
        .respond_with(ResponseTemplate::new(200).set_body_string("this is not json"))
        .expect(1)
        .mount(&server)
        .await;

    let provider = provider_for(&server);
    let err = provider
        .analyze("content")
        .await
        .expect_err("non-JSON body should be a provider failure");

    assert!(
        matches!(err, GuardrailProviderError::MalformedResponse(_)),
        "expected MalformedResponse, got {err:?}"
    );
}
