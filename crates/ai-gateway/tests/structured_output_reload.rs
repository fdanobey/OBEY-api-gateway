use std::{sync::Arc, time::Duration};

use ai_gateway::{
    config::{load_and_validate_config, Config},
    gateway::{apply_runtime_config_update, GatewayServer},
    structured_output::StructuredOutputEngine,
};
use axum::{
    body::Body,
    http::{Request, Response, StatusCode},
};
use serde_json::{json, Value};
use tempfile::TempDir;
use tokio::sync::Notify;
use tower::ServiceExt;
use wiremock::{
    matchers::{method, path},
    Mock, MockServer, ResponseTemplate,
};

const PROVIDER_NAME: &str = "reload-provider";
const PROVIDER_MODEL: &str = "reload-model";
const MODEL_GROUP: &str = "reload-group";
const VALIDATION_HEADER: &str = "x-obey-validation-status";

fn config_with_structured_output(
    provider_uri: &str,
    temp: &TempDir,
    max_retries: u8,
    retry_temperature: f32,
    passthrough: bool,
) -> Config {
    let passthrough_providers = if passthrough {
        format!("[{PROVIDER_NAME}]")
    } else {
        "[]".to_string()
    };
    let yaml = format!(
        r#"
server:
  host: 127.0.0.1
  port: 18080
  request_timeout_seconds: 30
providers:
  - name: {PROVIDER_NAME}
    type: openai
    base_url: {provider_uri}
    timeout_seconds: 30
model_groups:
  - name: {MODEL_GROUP}
    models:
      - provider: {PROVIDER_NAME}
        model: {PROVIDER_MODEL}
retry:
  max_retries_per_provider: 0
  backoff_sequence_seconds: [0]
  jitter_enabled: false
structured_output:
  enabled: true
  max_retries: {max_retries}
  retry_temperature: {retry_temperature}
  passthrough_providers: {passthrough_providers}
"#
    );

    let mut config: Config = serde_yaml::from_str(&yaml).expect("structured reload test config");
    config.logging.database_path = temp
        .path()
        .join("request-logs.db")
        .to_string_lossy()
        .into_owned();
    config.virtual_keys.database_path = temp
        .path()
        .join("virtual-keys.db")
        .to_string_lossy()
        .into_owned();
    config.exact_cache.enabled = false;
    config
}

fn effective(
    engine: &StructuredOutputEngine,
) -> ai_gateway::structured_output::config::EffectiveConfig {
    engine.effective_config(MODEL_GROUP, PROVIDER_NAME, PROVIDER_MODEL)
}

fn structured_request(prompt: &str) -> Value {
    json!({
        "model": MODEL_GROUP,
        "messages": [{"role": "user", "content": prompt}],
        "temperature": 0.0,
        "response_format": {
            "type": "json_schema",
            "json_schema": {
                "name": "answer",
                "strict": true,
                "schema": {
                    "type": "object",
                    "properties": {"answer": {"type": "string"}},
                    "required": ["answer"],
                    "additionalProperties": false
                }
            }
        }
    })
}

fn completion(id: &str, content: &str) -> Value {
    json!({
        "id": id,
        "object": "chat.completion",
        "created": 1_700_000_000_i64,
        "model": PROVIDER_MODEL,
        "choices": [{
            "index": 0,
            "message": {"role": "assistant", "content": content},
            "finish_reason": "stop"
        }],
        "usage": {"prompt_tokens": 5, "completion_tokens": 3, "total_tokens": 8}
    })
}

fn post_chat(app: axum::Router, prompt: &str) -> impl std::future::Future<Output = Response<Body>> {
    let request = Request::post("/v1/chat/completions")
        .header("content-type", "application/json")
        .body(Body::from(
            serde_json::to_vec(&structured_request(prompt)).unwrap(),
        ))
        .unwrap();
    async move { app.oneshot(request).await.expect("gateway response") }
}

async fn response_json(response: Response<Body>) -> Value {
    let bytes = axum::body::to_bytes(response.into_body(), 1024 * 1024)
        .await
        .expect("bounded response body");
    serde_json::from_slice(&bytes).expect("JSON response body")
}

async fn recorded_provider_requests(provider: &MockServer) -> Vec<Value> {
    provider
        .received_requests()
        .await
        .expect("wiremock request recording")
        .into_iter()
        .map(|request| serde_json::from_slice(&request.body).expect("provider request JSON"))
        .collect()
}

#[tokio::test]
async fn reload_swaps_structured_output_snapshots_and_rejects_invalid_candidate() {
    let provider = MockServer::start().await;
    let temp = tempfile::tempdir().expect("temporary reload directory");
    let config_path = temp.path().join("config.yaml");
    let config_a = config_with_structured_output(&provider.uri(), &temp, 1, 0.25, false);
    std::fs::write(
        &config_path,
        serde_yaml::to_string(&config_a).expect("serialize config A"),
    )
    .expect("write config A");

    let server = GatewayServer::new(config_a, Some(config_path.clone()))
        .await
        .expect("gateway starts with config A");
    let app = server.build_router();
    let engine_a = server
        .state
        .structured_output_engine
        .read()
        .await
        .clone()
        .expect("config A engine");
    let effective_a = effective(&engine_a);
    assert_eq!(effective_a.max_retries, 1);
    assert_eq!(effective_a.retry_temperature, 0.25);
    assert!(!effective_a.passthrough);

    let config_b = config_with_structured_output(&provider.uri(), &temp, 4, 0.75, true);
    std::fs::write(
        &config_path,
        serde_yaml::to_string(&config_b).expect("serialize config B"),
    )
    .expect("write config B");
    let reload_response = app
        .oneshot(
            Request::post("/admin/config/reload")
                .body(Body::empty())
                .unwrap(),
        )
        .await
        .expect("reload endpoint response");
    assert_eq!(reload_response.status(), StatusCode::OK);

    let engine_b = server
        .state
        .structured_output_engine
        .read()
        .await
        .clone()
        .expect("config B engine");
    assert!(!Arc::ptr_eq(&engine_a, &engine_b));
    let effective_b = effective(&engine_b);
    assert_eq!(effective_b.max_retries, 4);
    assert_eq!(effective_b.retry_temperature, 0.75);
    assert!(effective_b.passthrough);
    assert_eq!(
        effective(&engine_a),
        effective_a,
        "old Arc retains config A"
    );

    let mut invalid = config_b.clone();
    invalid
        .structured_output
        .as_mut()
        .expect("structured output section")
        .max_retries = 6;
    assert!(
        invalid.validate().is_err(),
        "invalid candidate fails validation"
    );
    std::fs::write(
        &config_path,
        serde_yaml::to_string(&invalid).expect("serialize invalid config"),
    )
    .expect("write invalid config");
    assert!(load_and_validate_config(&config_path).is_err());
    let engine_after_invalid = server
        .state
        .structured_output_engine
        .read()
        .await
        .clone()
        .expect("active engine survives invalid candidate");
    assert!(Arc::ptr_eq(&engine_b, &engine_after_invalid));

    let mut without_section = config_b;
    without_section.structured_output = None;
    apply_runtime_config_update(&server.state, without_section).await;
    assert!(server.state.structured_output_engine.read().await.is_none());
    assert_eq!(effective(&engine_a), effective_a);
    assert_eq!(effective(&engine_b), effective_b);
}

#[tokio::test]
async fn in_flight_request_keeps_config_a_while_subsequent_request_uses_passthrough_b() {
    let provider = MockServer::start().await;
    let initial_request_seen = Arc::new(Notify::new());
    Mock::given(method("POST"))
        .and(path("/v1/chat/completions"))
        .respond_with({
            let initial_request_seen = Arc::clone(&initial_request_seen);
            move |request: &wiremock::Request| {
                let body: Value =
                    serde_json::from_slice(&request.body).expect("provider request JSON");
                let is_retry = body["messages"]
                    .as_array()
                    .is_some_and(|messages| messages.len() > 1);
                if is_retry {
                    ResponseTemplate::new(200)
                        .set_body_json(completion("retry", r#"{"answer":"from-a"}"#))
                } else {
                    initial_request_seen.notify_one();
                    ResponseTemplate::new(200)
                        .set_delay(Duration::from_millis(300))
                        .set_body_json(completion("initial", "not valid json"))
                }
            }
        })
        .mount(&provider)
        .await;

    let temp = tempfile::tempdir().expect("temporary request-snapshot directory");
    let config_a = config_with_structured_output(&provider.uri(), &temp, 1, 0.25, false);
    let server = GatewayServer::new(config_a, None)
        .await
        .expect("gateway starts with config A");
    let app = server.build_router();

    let in_flight = tokio::spawn(post_chat(app.clone(), "request-under-a"));
    tokio::time::timeout(Duration::from_secs(2), initial_request_seen.notified())
        .await
        .expect("provider observes request before config swap");

    let config_b = config_with_structured_output(&provider.uri(), &temp, 5, 1.5, true);
    apply_runtime_config_update(&server.state, config_b).await;
    let current = server
        .state
        .structured_output_engine
        .read()
        .await
        .clone()
        .expect("config B engine");
    let effective_b = effective(&current);
    assert_eq!(effective_b.max_retries, 5);
    assert_eq!(effective_b.retry_temperature, 1.5);
    assert!(effective_b.passthrough);

    let in_flight_response = in_flight.await.expect("in-flight request task");
    assert_eq!(in_flight_response.status(), StatusCode::OK);
    assert_eq!(
        in_flight_response
            .headers()
            .get(VALIDATION_HEADER)
            .and_then(|value| value.to_str().ok()),
        Some("passed")
    );
    let in_flight_body = response_json(in_flight_response).await;
    assert_eq!(
        in_flight_body["choices"][0]["message"]["content"],
        r#"{"answer":"from-a"}"#
    );

    let subsequent_response = post_chat(app, "request-under-b").await;
    assert_eq!(subsequent_response.status(), StatusCode::OK);
    assert_eq!(
        subsequent_response
            .headers()
            .get(VALIDATION_HEADER)
            .and_then(|value| value.to_str().ok()),
        Some("skipped")
    );
    let subsequent_body = response_json(subsequent_response).await;
    assert_eq!(
        subsequent_body["choices"][0]["message"]["content"],
        "not valid json"
    );

    let requests = recorded_provider_requests(&provider).await;
    assert_eq!(
        requests.len(),
        3,
        "A retries once; passthrough B does not retry"
    );
    assert_eq!(requests[0]["messages"].as_array().unwrap().len(), 1);
    assert_eq!(requests[1]["messages"].as_array().unwrap().len(), 3);
    assert_eq!(requests[1]["temperature"].as_f64(), Some(0.25));
    assert_eq!(requests[2]["messages"].as_array().unwrap().len(), 1);
}
