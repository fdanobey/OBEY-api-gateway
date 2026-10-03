//! Wiremock integration tests for the Laya System One family: HTTP client,
//! model discovery, and the confidence gate.

//! No live network calls: every provider endpoint is local and deterministic.

use std::time::Duration;

use ai_gateway::smart_routing::config::{DimensionWeights, JevFallbackPolicy, LayaConfig};
use ai_gateway::smart_routing::laya::{LayaClassifier, FAMILY};
use ai_gateway::smart_routing::systemone::client::SystemOneClient;
use ai_gateway::smart_routing::systemone::models::{Question, SystemOneRequest};
use ai_gateway::smart_routing::OptionalClassifier;
use serde_json::json;
use wiremock::matchers::{header, method, path};
use wiremock::{Mock, MockServer, Request, Respond, ResponseTemplate};

fn client(server: &MockServer, retries: u8) -> SystemOneClient {
    SystemOneClient::new(
        FAMILY,
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
async fn systemone_happy_path_sends_auth_and_parses_answer_confidence() {
    let server = MockServer::start().await;
    Mock::given(method("POST"))
        .and(path("/v1/systemone"))
        .and(header("authorization", "Bearer test-api-key"))
        .respond_with(ResponseTemplate::new(200).set_body_json(json!({
            "model": "laya-typed-decisions",
            "answers": {
                "complexity": {
                    "type": "score",
                    "score": 1.0,
                    "confidence": 0.40,
                    "answer_confidence": 0.91,
                    "probabilities": {"0": 0.0, "1": 1.0}
                }
            },
            "usage": {"input_tokens": 25, "output_tokens": 2}
        })))
        .expect(1)
        .mount(&server)
        .await;

    let response = client(&server, 0)
        .evaluate(&systemone_request("laya-typed-decisions"))
        .await
        .unwrap();
    assert_eq!(response.model, "laya-typed-decisions");
    assert_eq!(response.usage.input_tokens, 25);
    let answer = response.answers.get("complexity").unwrap();
    assert_eq!(answer.gate_confidence(&FAMILY), Some(0.91));
}

#[tokio::test]
async fn model_listing_ranks_laya_models_by_version() {
    let server = MockServer::start().await;
    Mock::given(method("GET"))
        .and(path("/v1/models"))
        .and(header("authorization", "Bearer test-api-key"))
        .respond_with(ResponseTemplate::new(200).set_body_json(json!({
            "data": [
                {"id": "gpt-4o-mini"},
                {"id": "convai/laya-1.9.0"},
                {"id": "convai/laya-2.0.1"},
                {"id": "layabout"},
                {"id": "jev-1.13.0"}
            ]
        })))
        .expect(1)
        .mount(&server)
        .await;

    let listing = client(&server, 0).list_models().await.unwrap();
    let ids: Vec<String> = listing.data.into_iter().map(|entry| entry.id).collect();
    let resolved =
        ai_gateway::smart_routing::systemone::discovery::latest_family_model(&FAMILY, &ids);
    assert_eq!(resolved.as_deref(), Some("convai/laya-2.0.1"));
}

/// Stateful responder: first request is 503 with Retry-After, second succeeds.
struct BusyThenSuccess(std::sync::atomic::AtomicUsize);

impl Respond for BusyThenSuccess {
    fn respond(&self, _request: &Request) -> ResponseTemplate {
        let call = self.0.fetch_add(1, std::sync::atomic::Ordering::SeqCst);
        if call == 0 {
            ResponseTemplate::new(503).insert_header("Retry-After", "0")
        } else {
            ResponseTemplate::new(200).set_body_json(json!({
                "model": "laya-typed-decisions",
                "answers": {},
                "usage": {"input_tokens": 1, "output_tokens": 0}
            }))
        }
    }
}

#[tokio::test]
async fn laya_503_is_retryable_and_honors_retry_after() {
    let server = MockServer::start().await;
    Mock::given(method("POST"))
        .and(path("/v1/systemone"))
        .respond_with(BusyThenSuccess(std::sync::atomic::AtomicUsize::new(0)))
        .expect(2)
        .mount(&server)
        .await;

    let response = client(&server, 1)
        .evaluate(&systemone_request("laya-typed-decisions"))
        .await
        .unwrap();
    assert_eq!(response.model, "laya-typed-decisions");
}

#[tokio::test]
async fn persistent_503_retries_are_bounded() {
    let server = MockServer::start().await;
    Mock::given(method("POST"))
        .and(path("/v1/systemone"))
        .respond_with(ResponseTemplate::new(503))
        .expect(3)
        .mount(&server)
        .await;

    let error = client(&server, 2)
        .evaluate(&systemone_request("laya-typed-decisions"))
        .await
        .unwrap_err();
    assert_eq!(
        error,
        ai_gateway::smart_routing::systemone::client::SystemOneCallError::Transient { status: 503 }
    );
}

#[tokio::test]
async fn auth_errors_do_not_retry() {
    let server = MockServer::start().await;
    Mock::given(method("POST"))
        .and(path("/v1/systemone"))
        .respond_with(ResponseTemplate::new(401))
        .expect(1)
        .mount(&server)
        .await;

    let error = client(&server, 2)
        .evaluate(&systemone_request("laya-typed-decisions"))
        .await
        .unwrap_err();
    assert_eq!(
        error,
        ai_gateway::smart_routing::systemone::client::SystemOneCallError::Auth { status: 401 }
    );
}

#[tokio::test]
async fn payload_too_large_is_a_typed_non_retryable_error() {
    let server = MockServer::start().await;
    Mock::given(method("POST"))
        .and(path("/v1/systemone"))
        .respond_with(ResponseTemplate::new(413))
        .expect(1)
        .mount(&server)
        .await;

    let error = client(&server, 2)
        .evaluate(&systemone_request("laya-typed-decisions"))
        .await
        .unwrap_err();
    assert_eq!(
        error,
        ai_gateway::smart_routing::systemone::client::SystemOneCallError::PayloadTooLarge
    );
}

#[tokio::test(flavor = "multi_thread")]
async fn laya_classifier_pins_default_when_listing_has_no_laya_models() {
    // Missing listing scenario: the endpoint serves a listing without any
    // Laya-capable model; discovery must pin `typed-decisions`.
    let server = MockServer::start().await;
    Mock::given(method("GET"))
        .and(path("/v1/models"))
        .respond_with(ResponseTemplate::new(200).set_body_json(json!({
            "data": [{"id": "gpt-4o-mini"}]
        })))
        .mount(&server)
        .await;
    Mock::given(method("POST"))
        .and(path("/v1/systemone"))
        .respond_with(
            ResponseTemplate::new(200).set_body_json(json!({
                "model": "typed-decisions",
                "answers": {},
                "usage": {"input_tokens": 1, "output_tokens": 0}
            })),
        )
        .mount(&server)
        .await;

    let config = LayaConfig {
        api_key: Some("test-key".to_string()),
        base_url: server.uri(),
        model: "auto".to_string(),
        ..LayaConfig::default()
    };
    let classifier = LayaClassifier::new(config).unwrap();
    classifier.prime_discovery();
    assert_eq!(classifier.resolved_model(), None);
}

#[tokio::test(flavor = "multi_thread")]
async fn laya_classifier_pins_configured_checkpoint_without_discovery() {
    let server = MockServer::start().await;
    // A pinned checkpoint performs no listing call and no evaluation during
    // construction or priming; the server verifies zero requests.
    let config = LayaConfig {
        api_key: Some("test-key".to_string()),
        base_url: server.uri(),
        model: "typed-decisions".to_string(),
        ..LayaConfig::default()
    };
    let classifier = LayaClassifier::new(config).unwrap();
    classifier.prime_discovery();
    // Pinned models are never cached as a discovery resolution.
    assert_eq!(classifier.resolved_model(), None);
    // No endpoint traffic: neither models listing nor evaluation.
    let server_hits = server.received_requests().await.unwrap().len();
    assert_eq!(server_hits, 0);
}

fn laya_config(server: &MockServer, model: &str, min_confidence: f64) -> LayaConfig {
    LayaConfig {
        api_key: Some("test-key".to_string()),
        base_url: server.uri(),
        model: model.to_string(),
        min_confidence,
        min_task_confidence: 0.50,
        timeout_ms: 2_000,
        char_budget: 1_024,
        fallback_policy: JevFallbackPolicy::Fallback,
        dimension_weights: DimensionWeights::default(),
        retry: ai_gateway::smart_routing::config::JevRetryConfig {
            max_attempts: 2,
            backoff_ms: 1,
        },
        ..LayaConfig::default()
    }
}

/// A full Laya-shaped evaluation answering every rubric question with the
/// given score and `answer_confidence`.
fn full_laya_response(score: f64, answer_confidence: f64) -> serde_json::Value {
    let mut answers = serde_json::Map::new();
    for id in [
        "reasoning_depth",
        "tool_coupling",
        "context_synthesis",
        "output_precision",
        "domain_load",
        "ambiguity",
    ] {
        answers.insert(
            id.to_string(),
            json!({
                "type": "score",
                "score": score,
                "confidence": 0.30,
                "answer_confidence": answer_confidence,
                "probabilities": {}
            }),
        );
    }
    answers.insert(
        "task_type".to_string(),
        json!({
            "type": "choice",
            "choice": "code_generation",
            "probabilities": {},
            "confidence": 0.30,
            "answer_confidence": answer_confidence
        }),
    );
    json!({
        "model": "laya-typed-decisions",
        "answers": serde_json::Value::Object(answers),
        "usage": {"input_tokens": 120, "output_tokens": 8}
    })
}

#[tokio::test(flavor = "multi_thread")]
async fn laya_classifier_high_answer_confidence_produces_decision() {
    let server = MockServer::start().await;
    Mock::given(method("POST"))
        .and(path("/v1/systemone"))
        .respond_with(ResponseTemplate::new(200).set_body_json(full_laya_response(3.0, 0.95)))
        .expect(1)
        .mount(&server)
        .await;

    let classifier = LayaClassifier::new(laya_config(&server, "typed-decisions", 0.55)).unwrap();
    let request = ai_gateway::models::openai::OpenAIRequest {
        model: "g".to_string(),
        messages: vec![ai_gateway::models::openai::Message {
            role: "user".to_string(),
            content: serde_json::json!("write a compiler pass"),
            extra: Default::default(),
        }],
        stream: false,
        temperature: None,
        max_tokens: None,
        extra: Default::default(),
    };
    let model_group = ai_gateway::config::ModelGroup {
        name: "group".to_string(),
        version_fallback_enabled: false,
        compression: None,
        memory: None,
        structured_output: None,
        models: Vec::new(),
    };
    let pinned_context = ai_gateway::smart_routing::PinnedRoutingContext::default();

    let output = classifier
        .classify(ai_gateway::smart_routing::ClassifierInput {
            request: &request,
            model_group: &model_group,
            pinned_context: &pinned_context,
            heuristic_score: ai_gateway::smart_routing::tier::ComplexityScore::new(0.5),
            heuristic_task_type: ai_gateway::smart_routing::tier::TaskType::General,
            jev_trust: None,
        })
        .await
        .unwrap();
    assert_eq!(output.task_type, Some(ai_gateway::smart_routing::tier::TaskType::CodeGeneration));
    assert_eq!(output.confidence, Some(0.95));
    assert_eq!(output.resolved_model.as_deref(), Some("typed-decisions"));
}

#[tokio::test(flavor = "multi_thread")]
async fn laya_classifier_low_answer_confidence_falls_back() {
    let server = MockServer::start().await;
    Mock::given(method("POST"))
        .and(path("/v1/systemone"))
        .respond_with(ResponseTemplate::new(200).set_body_json(full_laya_response(3.0, 0.10)))
        .expect(1)
        .mount(&server)
        .await;

    let classifier = LayaClassifier::new(laya_config(&server, "typed-decisions", 0.55)).unwrap();
    let request = ai_gateway::models::openai::OpenAIRequest {
        model: "g".to_string(),
        messages: vec![ai_gateway::models::openai::Message {
            role: "user".to_string(),
            content: serde_json::json!("hi"),
            extra: Default::default(),
        }],
        stream: false,
        temperature: None,
        max_tokens: None,
        extra: Default::default(),
    };
    let model_group = ai_gateway::config::ModelGroup {
        name: "group".to_string(),
        version_fallback_enabled: false,
        compression: None,
        memory: None,
        structured_output: None,
        models: Vec::new(),
    };
    let pinned_context = ai_gateway::smart_routing::PinnedRoutingContext::default();

    let outcome = classifier
        .classify(ai_gateway::smart_routing::ClassifierInput {
            request: &request,
            model_group: &model_group,
            pinned_context: &pinned_context,
            heuristic_score: ai_gateway::smart_routing::tier::ComplexityScore::new(0.5),
            heuristic_task_type: ai_gateway::smart_routing::tier::TaskType::General,
            jev_trust: None,
        })
        .await;
    assert!(matches!(
        outcome,
        Err(ai_gateway::smart_routing::ClassifierFailure::LowConfidence)
    ));
}
