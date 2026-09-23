//! Wiremock integration tests for the Jev HTTP client and model discovery.
//!
//! No live network calls: every provider endpoint is local and deterministic.

use std::time::Duration;

use ai_gateway::smart_routing::jev::client::{JevCallError, JevClient};
use ai_gateway::smart_routing::jev::models::{Question, SystemOneRequest};
use serde_json::json;
use wiremock::matchers::{header, method, path};
use wiremock::{Mock, MockServer, Request, Respond, ResponseTemplate};

fn client(server: &MockServer, retries: u8) -> JevClient {
    JevClient::new(
        &server.uri(),
        "test-api-key",
        Duration::from_millis(2_000),
        retries,
        Duration::from_millis(1),
    )
    .unwrap()
}

fn systemone_request(model: &str) -> SystemOneRequest {
    let mut questions = std::collections::HashMap::new();
    questions.insert(
        "complexity".to_string(),
        Question {
            question_type: "score",
            instructions: json!("complexity"),
            criteria: json!(["low", "high"]),
        },
    );
    SystemOneRequest {
        state: json!({"messages": [{"role": "user", "content": "hello"}]}),
        model: model.to_string(),
        questions,
    }
}

#[tokio::test]
async fn systemone_happy_path_sends_auth_and_parses_response() {
    let server = MockServer::start().await;
    Mock::given(method("POST"))
        .and(path("/v1/systemone"))
        .and(header("authorization", "Bearer test-api-key"))
        .respond_with(ResponseTemplate::new(200).set_body_json(json!({
            "model": "jev-1.13.0",
            "answers": {
                "complexity": {
                    "type": "score",
                    "score": 1.0,
                    "confidence": 0.91,
                    "probabilities": {"0": 0.0, "1": 1.0},
                    "legend": {"0": "low", "1": "high"}
                }
            },
            "usage": {"input_tokens": 25, "output_tokens": 2}
        })))
        .expect(1)
        .mount(&server)
        .await;

    let response = client(&server, 0)
        .evaluate(&systemone_request("jev-1.13.0"))
        .await
        .unwrap();
    assert_eq!(response.model, "jev-1.13.0");
    assert_eq!(response.usage.input_tokens, 25);
}

#[tokio::test]
async fn model_listing_happy_path_parses_available_models() {
    let server = MockServer::start().await;
    Mock::given(method("GET"))
        .and(path("/v1/models"))
        .and(header("authorization", "Bearer test-api-key"))
        .respond_with(ResponseTemplate::new(200).set_body_json(json!({
            "data": [
                {"id": "gpt-4o-mini", "owned_by": "openai"},
                {"id": "typesafe/jev-1.12.0"},
                {"id": "typesafe/jev-1.13.0"}
            ]
        })))
        .expect(1)
        .mount(&server)
        .await;

    let listing = client(&server, 0).list_models().await.unwrap();
    assert_eq!(listing.data.len(), 3);
    assert_eq!(listing.data[2].id, "typesafe/jev-1.13.0");
}

/// Stateful responder: first request is 429, second succeeds.
struct RateLimitThenSuccess(std::sync::atomic::AtomicUsize);

impl Respond for RateLimitThenSuccess {
    fn respond(&self, _request: &Request) -> ResponseTemplate {
        let call = self.0.fetch_add(1, std::sync::atomic::Ordering::SeqCst);
        if call == 0 {
            ResponseTemplate::new(429).insert_header("Retry-After", "0")
        } else {
            ResponseTemplate::new(200).set_body_json(json!({
                "model": "jev-1.13.0",
                "answers": {},
                "usage": {"input_tokens": 1, "output_tokens": 0}
            }))
        }
    }
}

#[tokio::test]
async fn rate_limit_retries_then_succeeds() {
    let server = MockServer::start().await;
    Mock::given(method("POST"))
        .and(path("/v1/systemone"))
        .respond_with(RateLimitThenSuccess(std::sync::atomic::AtomicUsize::new(0)))
        .expect(2)
        .mount(&server)
        .await;

    let response = client(&server, 1)
        .evaluate(&systemone_request("jev-1.13.0"))
        .await
        .unwrap();
    assert_eq!(response.model, "jev-1.13.0");
}

#[tokio::test]
async fn persistent_529_retries_are_bounded() {
    let server = MockServer::start().await;
    Mock::given(method("POST"))
        .and(path("/v1/systemone"))
        .respond_with(ResponseTemplate::new(529))
        .expect(3)
        .mount(&server)
        .await;

    let error = client(&server, 2)
        .evaluate(&systemone_request("jev-1.13.0"))
        .await
        .unwrap_err();
    assert_eq!(error, JevCallError::Transient { status: 529 });
}

#[tokio::test]
async fn auth_and_validation_errors_do_not_retry() {
    for (status, expected) in [
        (401, JevCallError::Auth { status: 401 }),
        (403, JevCallError::Auth { status: 403 }),
        (422, JevCallError::BadRequest { status: 422 }),
    ] {
        let server = MockServer::start().await;
        Mock::given(method("POST"))
            .and(path("/v1/systemone"))
            .respond_with(ResponseTemplate::new(status))
            .expect(1)
            .mount(&server)
            .await;

        let error = client(&server, 2)
            .evaluate(&systemone_request("jev-1.13.0"))
            .await
            .unwrap_err();
        assert_eq!(error, expected);
    }
}

#[tokio::test]
async fn oversized_response_body_is_rejected() {
    let server = MockServer::start().await;
    Mock::given(method("GET"))
        .and(path("/v1/models"))
        .respond_with(ResponseTemplate::new(200).set_body_string("x".repeat(600 * 1024)))
        .expect(1)
        .mount(&server)
        .await;

    let error = client(&server, 0).list_models().await.unwrap_err();
    assert!(matches!(error, JevCallError::Unavailable { reason } if reason.contains("size cap")));
}

#[tokio::test]
async fn large_model_listing_within_cap_is_accepted() {
    // Real model catalogs commonly exceed 16 KiB; listings are capped at
    // 512 KiB instead of the 16 KiB evaluation-answer cap.
    let server = MockServer::start().await;
    Mock::given(method("GET"))
        .and(path("/v1/models"))
        .respond_with(ResponseTemplate::new(200).set_body_string("x".repeat(20 * 1024)))
        .expect(1)
        .mount(&server)
        .await;

    // Body exceeds the answer cap but stays under the listing cap; the
    // request succeeds and only JSON decoding fails.
    let error = client(&server, 0).list_models().await.unwrap_err();
    assert!(
        matches!(error, JevCallError::Unavailable { ref reason } if reason.contains("decode listing")),
        "20 KiB listing must not hit the size cap: {error:?}"
    );
}

#[tokio::test]
async fn timeout_maps_to_typed_error() {
    let server = MockServer::start().await;
    Mock::given(method("GET"))
        .and(path("/v1/models"))
        .respond_with(
            ResponseTemplate::new(200)
                .set_delay(Duration::from_millis(100))
                .set_body_json(json!({"data": []})),
        )
        .expect(1)
        .mount(&server)
        .await;

    let client = JevClient::new(
        &server.uri(),
        "key",
        Duration::from_millis(10),
        0,
        Duration::from_millis(1),
    )
    .unwrap();
    assert_eq!(
        client.list_models().await.unwrap_err(),
        JevCallError::Timeout
    );
}
