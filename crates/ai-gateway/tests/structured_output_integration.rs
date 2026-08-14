//! End-to-end structured-output validation scenarios for Wave 10 tasks 13.1-13.4.

use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::Arc;

use ai_gateway::config::*;
use ai_gateway::gateway::GatewayServer;
use ai_gateway::guardrail::{
    FailurePolicy, GuardrailConfig, GuardrailProviderConfig, GuardrailProviderType,
    InstructionInsertionMode, PipelineConfig, PolicyAction, ProviderSettings, RegexPatternConfig,
    RegexRuleMode, StageConfig, StagePhase,
};
use ai_gateway::structured_output::config::{StructuredOutputConfig, StructuredOutputOverride};
use axum::body::Body;
use axum::http::{Request, Response, StatusCode};
use serde_json::{json, Value};
use tempfile::TempDir;
use tower::ServiceExt;
use wiremock::matchers::{method, path};
use wiremock::{Mock, MockServer, ResponseTemplate};

const VALIDATION_HEADER: &str = "x-obey-validation-status";
const PROVIDER_PATH: &str = "/v1/chat/completions";
const RETRY_TEMPERATURE: f64 = 0.37;

fn chat_completion(id: &str, model: &str, content: &str) -> Value {
    json!({
        "id": id,
        "object": "chat.completion",
        "created": 1_700_000_000_i64,
        "model": model,
        "choices": [{
            "index": 0,
            "message": {"role": "assistant", "content": content},
            "finish_reason": "stop"
        }],
        "usage": {"prompt_tokens": 10, "completion_tokens": 8, "total_tokens": 18}
    })
}

fn sse_completion(content: &str) -> String {
    let chunk = json!({
        "id": "chatcmpl-initial-stream",
        "object": "chat.completion.chunk",
        "created": 1_700_000_000_i64,
        "model": "provider-model",
        "choices": [{
            "index": 0,
            "delta": {"role": "assistant", "content": content},
            "finish_reason": "stop"
        }]
    });
    format!("data: {chunk}\n\ndata: [DONE]\n\n")
}

fn structured_request(stream: bool, prompt: &str) -> Value {
    json!({
        "model": "structured-group",
        "messages": [{"role": "user", "content": prompt}],
        "stream": stream,
        "temperature": 0.0,
        "response_format": {
            "type": "json_schema",
            "json_schema": {
                "name": "answer",
                "strict": true,
                "schema": {
                    "type": "object",
                    "properties": {
                        "answer": {"type": "string"},
                        "count": {"type": "integer"}
                    },
                    "required": ["answer", "count"],
                    "additionalProperties": false
                }
            }
        }
    })
}

fn compile_invalid_request(prompt: &str) -> Value {
    let mut request = structured_request(false, prompt);
    request["response_format"]["json_schema"]["schema"] = json!({"type": 42});
    request
}

struct TestApp {
    app: axum::Router,
    _temp: TempDir,
}

async fn build_app(
    provider_uri: &str,
    structured_output: StructuredOutputConfig,
    group_override: Option<StructuredOutputOverride>,
    provider_model_passthrough: Option<bool>,
    guardrails: Option<GuardrailConfig>,
) -> TestApp {
    let temp = tempfile::tempdir().expect("temporary integration-test directory");
    let mut config = base_config(provider_uri);
    config.logging.database_path = temp.path().join("logs.db").to_string_lossy().into_owned();
    config.virtual_keys.database_path = temp.path().join("keys.db").to_string_lossy().into_owned();
    config.structured_output = Some(structured_output);
    config.prometheus = Some(PrometheusConfig {
        enabled: true,
        path: "/metrics".to_string(),
    });
    config.model_groups[0].structured_output = group_override;
    config.model_groups[0].models[0].structured_output_passthrough = provider_model_passthrough;
    config.guardrails = guardrails;

    let server = GatewayServer::new(config, None)
        .await
        .expect("gateway server builds for structured-output integration test");
    TestApp {
        app: server.build_router(),
        _temp: temp,
    }
}

fn base_config(provider_uri: &str) -> Config {
    Config {
        server: ServerConfig {
            host: "127.0.0.1".to_string(),
            port: 0,
            request_timeout_seconds: 30,
            max_request_size_mb: 10,
        },
        tls: None,
        admin: AdminConfig::default(),
        dashboard: DashboardConfig::default(),
        cors: CorsConfig::default(),
        providers: vec![Provider {
            name: "test-provider".to_string(),
            provider_type: "openai".to_string(),
            base_url: Some(provider_uri.to_string()),
            api_key_env: None,
            api_key_encrypted: None,
            api_secret_env: None,
            api_secret_encrypted: None,
            auth_method: None,
            resolved_api_key: None,
            resolved_api_secret: None,
            region: None,
            timeout_seconds: 30,
            ttfb_timeout_seconds: None,
            total_timeout_seconds: None,
            max_connections: 10,
            rate_limit_per_minute: 0,
            custom_headers: Default::default(),
            connection_pool: ProviderConnectionPoolConfig::default(),
            budget: None,
            manual_models: vec![],
            global_inference_profile: false,
            cross_region_inference: false,
            custom_vpc_endpoint: false,
            prompt_caching: false,
            compression: None,
            memory: None,
            reasoning: true,
            codex_base_url_override: None,
            codex_model_override: None,
            instructions_override: None,
            max_rate_limit_cooldown_seconds: None,
        }],
        model_groups: vec![ModelGroup {
            name: "structured-group".to_string(),
            version_fallback_enabled: false,
            compression: None,
            memory: None,
            structured_output: None,
            models: vec![ProviderModel {
                provider: "test-provider".to_string(),
                model: "provider-model".to_string(),
                cost_per_million_input_tokens: 0.0,
                cost_per_million_output_tokens: 0.0,
                priority: 100,
                structured_output_passthrough: None,
                tier: None,
                context_window: 0,
                specializations: vec![],
            }],
        }],
        circuit_breaker: CircuitBreakerConfig::default(),
        retry: RetryConfig {
            max_retries_per_provider: 0,
            backoff_sequence_seconds: vec![0],
            jitter_enabled: false,
            ..RetryConfig::default()
        },
        logging: LoggingConfig::default(),
        semantic_cache: None,
        exact_cache: ExactCacheConfig {
            enabled: true,
            max_entries: 128,
            ttl_seconds: 300,
            temperature_threshold: 0.15,
        },
        prometheus: None,
        context: ContextConfig::default(),
        compression: Default::default(),
        memory: None,
        first_launch_completed: false,
        tray: TrayConfig::default(),
        codex_instructions_url: None,
        streaming: Some(StreamingConfig {
            emit_early_event: false,
            passthrough_enabled: true,
            ..StreamingConfig::default()
        }),
        virtual_keys: VirtualKeysConfig::default(),
        loop_detection: Default::default(),
        structured_output: None,
        guardrails: None,
        tool_compression: Default::default(),
        smart_routing: Default::default(),
        xhigh_models_allowlist: Default::default(),
        reasoning_models_allowlist: Default::default(),
    }
}

fn validating_config(max_retries: u8) -> StructuredOutputConfig {
    StructuredOutputConfig {
        enabled: true,
        max_retries,
        retry_temperature: RETRY_TEMPERATURE as f32,
        passthrough_providers: vec![],
    }
}

fn post_call_redaction_guardrail() -> GuardrailConfig {
    GuardrailConfig {
        providers: vec![GuardrailProviderConfig {
            name: "retry-redactor".to_string(),
            provider_type: GuardrailProviderType::Regex,
            failure_policy: FailurePolicy::FailClose,
            timeout_seconds: 5,
            settings: ProviderSettings {
                patterns: vec![RegexPatternConfig {
                    name: "secret".to_string(),
                    regex: "SECRET_VALUE".to_string(),
                    entity: "SECRET".to_string(),
                    mode: RegexRuleMode::Deny,
                }],
                ..ProviderSettings::default()
            },
        }],
        pipelines: vec![PipelineConfig {
            name: "post-call-redaction".to_string(),
            stages: vec![StageConfig {
                name: "redact-secret".to_string(),
                provider: "retry-redactor".to_string(),
                phase: StagePhase::PostCall,
                action: PolicyAction::Redact,
            }],
            redaction_notice_instruction: None,
            instruction_insertion_mode: InstructionInsertionMode::default(),
            failover_on_refusal: false,
            refusal_phrase_list: None,
        }],
        global_default_pipeline: Some("post-call-redaction".to_string()),
        bindings: Default::default(),
        ..Default::default()
    }
}

async fn post(app: axum::Router, body: &Value) -> Response<Body> {
    let request = Request::post("/v1/chat/completions")
        .header("content-type", "application/json")
        .body(Body::from(serde_json::to_vec(body).unwrap()))
        .unwrap();
    app.oneshot(request)
        .await
        .expect("gateway oneshot response")
}

async fn response_json(response: Response<Body>) -> Value {
    let bytes = axum::body::to_bytes(response.into_body(), 4 * 1024 * 1024)
        .await
        .expect("bounded response body");
    serde_json::from_slice(&bytes).expect("JSON response body")
}

async fn response_text(response: Response<Body>) -> String {
    let bytes = axum::body::to_bytes(response.into_body(), 4 * 1024 * 1024)
        .await
        .expect("bounded response body");
    String::from_utf8(bytes.to_vec()).expect("UTF-8 response body")
}

async fn provider_requests(server: &MockServer) -> Vec<Value> {
    server
        .received_requests()
        .await
        .expect("wiremock request recording")
        .into_iter()
        .map(|request| serde_json::from_slice(&request.body).expect("provider request JSON"))
        .collect()
}

fn assert_corrective_retry(initial: &Value, retry: &Value, expected_stream: bool) {
    assert_eq!(initial["messages"].as_array().unwrap().len(), 1);
    let messages = retry["messages"].as_array().expect("retry messages array");
    assert_eq!(messages.len(), 3, "retry appends exactly two messages");
    assert_eq!(messages[0], initial["messages"][0]);
    assert_eq!(messages[1]["role"], "assistant");
    assert_eq!(messages[1]["content"], "not valid json");
    assert_eq!(messages[2]["role"], "user");
    let correction = messages[2]["content"]
        .as_str()
        .expect("corrective user message text");
    assert!(correction.contains("previous output was not valid JSON"));
    assert!(correction.contains("Output ONLY valid JSON"));
    assert_eq!(retry["temperature"].as_f64(), Some(RETRY_TEMPERATURE));
    assert_eq!(retry["stream"], expected_stream);
}

fn validation_header(response: &Response<Body>) -> Option<&str> {
    response
        .headers()
        .get(VALIDATION_HEADER)
        .and_then(|value| value.to_str().ok())
}

fn sse_data_lines(body: &str) -> Vec<&str> {
    body.lines()
        .filter_map(|line| line.strip_prefix("data: "))
        .collect()
}

fn sse_content(body: &str) -> String {
    sse_data_lines(body)
        .into_iter()
        .filter(|line| *line != "[DONE]")
        .filter_map(|line| serde_json::from_str::<Value>(line).ok())
        .filter_map(|chunk| {
            chunk["choices"][0]["delta"]["content"]
                .as_str()
                .map(str::to_owned)
        })
        .collect()
}

async fn start_sequenced_json_provider(first_content: &str, second_content: &str) -> MockServer {
    let server = MockServer::start().await;
    let calls = Arc::new(AtomicUsize::new(0));
    Mock::given(method("POST"))
        .and(path(PROVIDER_PATH))
        .respond_with({
            let calls = Arc::clone(&calls);
            let first = chat_completion("chatcmpl-first", "provider-model", first_content);
            let second = chat_completion("chatcmpl-second", "provider-model", second_content);
            move |_request: &wiremock::Request| {
                let body = if calls.fetch_add(1, Ordering::SeqCst) == 0 {
                    first.clone()
                } else {
                    second.clone()
                };
                ResponseTemplate::new(200).set_body_json(body)
            }
        })
        .mount(&server)
        .await;
    server
}

#[tokio::test]
async fn task_13_1_non_stream_corrective_retry_and_post_call_guardrail() {
    let provider =
        start_sequenced_json_provider("not valid json", r#"{"answer":"SECRET_VALUE","count":2}"#)
            .await;
    let test_app = build_app(
        &provider.uri(),
        validating_config(1),
        None,
        None,
        Some(post_call_redaction_guardrail()),
    )
    .await;
    let request = structured_request(false, "task-13.1-corrective-retry");

    let response = post(test_app.app, &request).await;
    assert_eq!(response.status(), StatusCode::OK);
    assert_eq!(validation_header(&response), Some("passed"));
    let body = response_json(response).await;
    assert_eq!(
        body["choices"][0]["message"]["content"], r#"{"answer":"[REDACTED]","count":2}"#,
        "post-call guardrails rerun on the accepted retry before it reaches the client"
    );
    let serialized = body.to_string();
    assert!(!serialized.contains("not valid json"));
    assert!(!serialized.contains("previous output was not valid JSON"));
    assert!(!serialized.contains("Output ONLY valid JSON"));

    let requests = provider_requests(&provider).await;
    assert_eq!(
        requests.len(),
        2,
        "one initial call plus one corrective retry"
    );
    assert_corrective_retry(&requests[0], &requests[1], false);
}

#[tokio::test]
async fn task_13_2_streaming_invalid_sse_retries_non_stream_and_finishes_sse() {
    let provider = MockServer::start().await;
    let calls = Arc::new(AtomicUsize::new(0));
    Mock::given(method("POST"))
        .and(path(PROVIDER_PATH))
        .respond_with({
            let calls = Arc::clone(&calls);
            move |_request: &wiremock::Request| {
                if calls.fetch_add(1, Ordering::SeqCst) == 0 {
                    ResponseTemplate::new(200)
                        .insert_header("content-type", "text/event-stream")
                        .set_body_string(sse_completion("not valid json"))
                } else {
                    ResponseTemplate::new(200).set_body_json(chat_completion(
                        "chatcmpl-stream-retry",
                        "provider-model",
                        r#"{"answer":"stream recovered","count":3}"#,
                    ))
                }
            }
        })
        .mount(&provider)
        .await;
    let test_app = build_app(&provider.uri(), validating_config(1), None, None, None).await;
    let request = structured_request(true, "task-13.2-streaming-retry");

    let response = post(test_app.app, &request).await;
    assert_eq!(response.status(), StatusCode::OK);
    assert_eq!(
        validation_header(&response),
        Some("passed"),
        "final validation status is available before the streaming body is consumed"
    );
    let body = response_text(response).await;
    assert_eq!(
        sse_content(&body),
        r#"{"answer":"stream recovered","count":3}"#
    );
    let data_lines = sse_data_lines(&body);
    assert_eq!(data_lines.last().copied(), Some("[DONE]"));
    assert_eq!(
        data_lines.iter().filter(|line| **line == "[DONE]").count(),
        1
    );
    assert!(!body.contains("not valid json"));
    assert!(!body.contains("previous output was not valid JSON"));

    let requests = provider_requests(&provider).await;
    assert_eq!(requests.len(), 2);
    assert_eq!(requests[0]["stream"], true);
    assert_corrective_retry(&requests[0], &requests[1], false);
}

#[tokio::test]
async fn task_13_3_passing_response_is_cached_and_replayed_without_provider_call() {
    let provider = MockServer::start().await;
    Mock::given(method("POST"))
        .and(path(PROVIDER_PATH))
        .respond_with(ResponseTemplate::new(200).set_body_json(chat_completion(
            "chatcmpl-cache-pass",
            "provider-model",
            r#"{"answer":"cached","count":1}"#,
        )))
        .mount(&provider)
        .await;
    let test_app = build_app(&provider.uri(), validating_config(1), None, None, None).await;
    let request = structured_request(false, "task-13.3-cache-pass");

    let first = post(test_app.app.clone(), &request).await;
    assert_eq!(validation_header(&first), Some("passed"));
    let first_body = response_json(first).await;
    let second = post(test_app.app, &request).await;
    let second_body = response_json(second).await;

    assert_eq!(
        second_body, first_body,
        "cache replay preserves canonical response"
    );
    assert_eq!(provider_requests(&provider).await.len(), 1);
}

#[tokio::test]
async fn task_13_3_exhausted_failure_is_not_cached() {
    let provider = MockServer::start().await;
    Mock::given(method("POST"))
        .and(path(PROVIDER_PATH))
        .respond_with(ResponseTemplate::new(200).set_body_json(chat_completion(
            "chatcmpl-failed",
            "provider-model",
            "not valid json",
        )))
        .mount(&provider)
        .await;
    let test_app = build_app(&provider.uri(), validating_config(1), None, None, None).await;
    let request = structured_request(false, "task-13.3-failed-not-cached");

    for _ in 0..2 {
        let response = post(test_app.app.clone(), &request).await;
        assert_eq!(response.status(), StatusCode::OK);
        assert_eq!(validation_header(&response), Some("failed"));
        let body = response_json(response).await;
        assert_eq!(body["choices"][0]["message"]["content"], "not valid json");
    }

    assert_eq!(
        provider_requests(&provider).await.len(),
        4,
        "each client call performs its own initial attempt and exhausted retry"
    );
}

#[tokio::test]
async fn task_13_3_zero_retry_failure_does_not_record_retry_outcome() {
    let provider = MockServer::start().await;
    Mock::given(method("POST"))
        .and(path(PROVIDER_PATH))
        .respond_with(ResponseTemplate::new(200).set_body_json(chat_completion(
            "chatcmpl-no-retry",
            "provider-model",
            "not valid json",
        )))
        .mount(&provider)
        .await;
    let test_app = build_app(&provider.uri(), validating_config(0), None, None, None).await;
    let request = structured_request(false, "task-13.3-zero-retry-metrics");

    let response = post(test_app.app.clone(), &request).await;
    assert_eq!(validation_header(&response), Some("failed"));
    assert_eq!(provider_requests(&provider).await.len(), 1);

    let metrics = test_app
        .app
        .oneshot(Request::get("/metrics").body(Body::empty()).unwrap())
        .await
        .expect("metrics response");
    let metrics = response_text(metrics).await;
    assert!(!metrics.contains("obey_api_structured_output_retries_total{"));
}

#[tokio::test]
async fn task_13_3_compile_invalid_skip_is_not_cached() {
    let provider = MockServer::start().await;
    Mock::given(method("POST"))
        .and(path(PROVIDER_PATH))
        .respond_with(ResponseTemplate::new(200).set_body_json(chat_completion(
            "chatcmpl-compile-skip",
            "provider-model",
            r#"{"answer":"identity","count":4}"#,
        )))
        .mount(&provider)
        .await;
    let test_app = build_app(&provider.uri(), validating_config(1), None, None, None).await;
    let request = compile_invalid_request("task-13.3-compile-skip-not-cached");

    for _ in 0..2 {
        let response = post(test_app.app.clone(), &request).await;
        assert_eq!(validation_header(&response), Some("skipped"));
        let body = response_json(response).await;
        assert_eq!(
            body["choices"][0]["message"]["content"],
            r#"{"answer":"identity","count":4}"#
        );
    }

    assert_eq!(provider_requests(&provider).await.len(), 2);
}

#[tokio::test]
async fn task_13_3_skipped_request_records_one_latency_sample() {
    let provider = MockServer::start().await;
    Mock::given(method("POST"))
        .and(path(PROVIDER_PATH))
        .respond_with(ResponseTemplate::new(200).set_body_json(chat_completion(
            "chatcmpl-skip-latency",
            "provider-model",
            r#"{"answer":"identity","count":4}"#,
        )))
        .mount(&provider)
        .await;
    let test_app = build_app(&provider.uri(), validating_config(1), None, None, None).await;
    let request = compile_invalid_request("task-13.3-skip-latency");

    let response = post(test_app.app.clone(), &request).await;
    assert_eq!(validation_header(&response), Some("skipped"));

    let metrics = test_app
        .app
        .oneshot(Request::get("/metrics").body(Body::empty()).unwrap())
        .await
        .expect("metrics response");
    let metrics = response_text(metrics).await;
    assert!(metrics.contains("status=\"skip\"} 1"));
    assert!(metrics.contains("obey_api_structured_output_latency_ms_count{provider=\"test-provider\",model=\"provider-model\"} 1"));
}

#[tokio::test]
async fn task_13_3_invalid_stream_is_not_cached() {
    let provider = MockServer::start().await;
    Mock::given(method("POST"))
        .and(path(PROVIDER_PATH))
        .respond_with(
            ResponseTemplate::new(200)
                .insert_header("content-type", "text/event-stream")
                .set_body_string(sse_completion("not valid json")),
        )
        .mount(&provider)
        .await;
    let test_app = build_app(&provider.uri(), validating_config(0), None, None, None).await;
    let request = structured_request(true, "task-13.3-invalid-stream-not-cached");

    for _ in 0..2 {
        let response = post(test_app.app.clone(), &request).await;
        assert_eq!(validation_header(&response), Some("failed"));
        let body = response_text(response).await;
        assert_eq!(sse_data_lines(&body).last().copied(), Some("[DONE]"));
        assert_eq!(sse_content(&body), "not valid json");
    }

    assert_eq!(provider_requests(&provider).await.len(), 2);
}

#[tokio::test]
async fn task_13_4_provider_model_passthrough_overrides_group_and_global() {
    let canonical = chat_completion(
        "chatcmpl-passthrough",
        "provider-model",
        "provider-native-non-json-output",
    );
    let provider = MockServer::start().await;
    Mock::given(method("POST"))
        .and(path(PROVIDER_PATH))
        .respond_with(ResponseTemplate::new(200).set_body_json(canonical.clone()))
        .mount(&provider)
        .await;
    let global = StructuredOutputConfig {
        enabled: true,
        max_retries: 2,
        retry_temperature: 0.9,
        passthrough_providers: vec!["other-provider".to_string()],
    };
    let group = StructuredOutputOverride {
        enabled: Some(true),
        max_retries: Some(1),
        retry_temperature: Some(0.6),
        passthrough_providers: Some(vec![]),
    };
    let test_app = build_app(&provider.uri(), global, Some(group), Some(true), None).await;
    let request = structured_request(false, "task-13.4-provider-model-passthrough");

    for _ in 0..2 {
        let response = post(test_app.app.clone(), &request).await;
        assert_eq!(response.status(), StatusCode::OK);
        assert_eq!(validation_header(&response), Some("skipped"));
        let body = response_json(response).await;
        assert_eq!(
            body, canonical,
            "passthrough returns the provider's canonical OpenAI response unchanged"
        );
    }

    let requests = provider_requests(&provider).await;
    assert_eq!(
        requests.len(),
        2,
        "skipped passthrough responses are not cached"
    );
    for request in requests {
        assert_eq!(request["messages"].as_array().unwrap().len(), 1);
        assert_eq!(request["temperature"], 0.0);
        assert_eq!(request["stream"], false);
    }
}
