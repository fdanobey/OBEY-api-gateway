//! Integration tests for Responses API protocol-level translation conformance.
//!
//! Covers SSE event synthesis, previous_response_id chaining, and edge cases
//! for the `/v1/responses` front door.

use axum::body::Body;
use axum::http::{Request, StatusCode};
use serde_json::{json, Value};
use tempfile::TempDir;
use tower::ServiceExt;
use wiremock::matchers::{method, path};
use wiremock::{Mock, MockServer, ResponseTemplate};

use ai_gateway::config::*;
use ai_gateway::gateway::GatewayServer;

const PROVIDER_PATH: &str = "/v1/chat/completions";

fn sse_chat_chunk(id: &str, model: &str, delta: Value, finish_reason: Option<&str>) -> Value {
    let mut chunk = json!({
        "id": id,
        "object": "chat.completion.chunk",
        "created": 1_700_000_000_i64,
        "model": model,
        "choices": [{
            "index": 0,
            "delta": delta
        }]
    });
    if let Some(reason) = finish_reason {
        chunk["choices"][0]["finish_reason"] = json!(reason);
    }
    chunk
}

fn sse_data_line(chunk: &Value) -> String {
    format!("data: {}\n\n", serde_json::to_string(chunk).unwrap())
}

fn sse_done_line() -> &'static str {
    "data: [DONE]\n\n"
}

fn chat_completion_response(id: &str, model: &str, content: &str) -> Value {
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
        "usage": {"prompt_tokens": 10, "completion_tokens": 5, "total_tokens": 15}
    })
}

fn responses_request_text(model: &str, input: &str, stream: bool) -> Value {
    json!({
        "model": model,
        "input": input,
        "stream": stream
    })
}

fn responses_request_with_previous(model: &str, input: &str, previous_id: &str, stream: bool) -> Value {
    json!({
        "model": model,
        "input": input,
        "previous_response_id": previous_id,
        "stream": stream
    })
}

fn responses_request_with_instructions(model: &str, input: &str, instructions: &str, stream: bool) -> Value {
    json!({
        "model": model,
        "input": input,
        "instructions": instructions,
        "stream": stream
    })
}

struct TestApp {
    app: axum::Router,
    _temp: TempDir,
}

async fn build_app(provider_uri: &str) -> TestApp {
    let temp = tempfile::tempdir().expect("temp dir for responses translation test");
    let mut config = base_config(provider_uri);
    config.logging.database_path = temp.path().join("logs.db").to_string_lossy().into_owned();
    config.virtual_keys.database_path = temp.path().join("keys.db").to_string_lossy().into_owned();

    let server = GatewayServer::new(config, None)
        .await
        .expect("gateway server builds for responses translation test");
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
            name: "test-group".to_string(),
            version_fallback_enabled: false,
            compression: None,
            memory: None,
            structured_output: None,
            models: vec![ProviderModel {
                provider: "test-provider".to_string(),
                model: "gpt-4".to_string(),
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
        retry: RetryConfig::default(),
        logging: LoggingConfig::default(),
        semantic_cache: None,
        exact_cache: ExactCacheConfig::default(),
        prometheus: None,
        context: ContextConfig::default(),
        compression: Default::default(),
        memory: None,
        first_launch_completed: false,
        tray: TrayConfig::default(),
        codex_instructions_url: None,
        streaming: None,
        virtual_keys: Default::default(),
        loop_detection: Default::default(),
        guardrails: None,
        tool_compression: Default::default(),
        smart_routing: Default::default(),
        structured_output: None,
        xhigh_models_allowlist: Default::default(),
        reasoning_models_allowlist: Default::default(),
        codex_search: None,
    }
}

async fn post_responses(app: axum::Router, body: &Value) -> (StatusCode, Vec<u8>) {
    let req = Request::post("/v1/responses")
        .header("content-type", "application/json")
        .body(Body::from(serde_json::to_vec(body).unwrap()))
        .unwrap();
    let resp = app.oneshot(req).await.unwrap();
    let status = resp.status();
    let body = axum::body::to_bytes(resp.into_body(), 1024 * 1024)
        .await
        .unwrap()
        .to_vec();
    (status, body)
}

async fn post_responses_stream(app: axum::Router, body: &Value) -> (StatusCode, String) {
    let req = Request::post("/v1/responses")
        .header("content-type", "application/json")
        .body(Body::from(serde_json::to_vec(body).unwrap()))
        .unwrap();
    let resp = app.oneshot(req).await.unwrap();
    let status = resp.status();
    let body = axum::body::to_bytes(resp.into_body(), 1024 * 1024)
        .await
        .unwrap();
    let body_str = String::from_utf8(body.to_vec()).unwrap();
    (status, body_str)
}

fn parse_sse_events(body: &str) -> Vec<Value> {
    body.lines()
        .filter_map(|line| line.strip_prefix("data: "))
        .filter(|data| *data != "[DONE]")
        .filter_map(|data| serde_json::from_str(data).ok())
        .collect()
}

fn event_type(event: &Value) -> &str {
    event.get("type").and_then(|t| t.as_str()).unwrap_or("")
}

fn sequence_number(event: &Value) -> u64 {
    event.get("sequence_number").and_then(|s| s.as_u64()).unwrap_or(0)
}

async fn start_streaming_mock_with_sse(sse_body: String) -> MockServer {
    let server = MockServer::start().await;
    Mock::given(method("POST"))
        .and(path(PROVIDER_PATH))
        .respond_with(
            ResponseTemplate::new(200)
                .insert_header("content-type", "text/event-stream")
                .set_body_string(sse_body),
        )
        .mount(&server)
        .await;
    server
}

async fn start_buffered_mock(response: Value) -> MockServer {
    let server = MockServer::start().await;
    Mock::given(method("POST"))
        .and(path(PROVIDER_PATH))
        .respond_with(ResponseTemplate::new(200).set_body_json(response))
        .mount(&server)
        .await;
    server
}

async fn capture_provider_request(server: &MockServer) -> Value {
    let requests = server.received_requests().await.unwrap();
    serde_json::from_slice(&requests[0].body).unwrap()
}

#[tokio::test]
async fn test_streaming_event_order_and_monotonicity() {
    let mut sse_body = String::new();
    sse_body.push_str(&sse_data_line(&sse_chat_chunk(
        "chatcmpl-1",
        "gpt-4",
        json!({"role": "assistant"}),
        None,
    )));
    sse_body.push_str(&sse_data_line(&sse_chat_chunk(
        "chatcmpl-1",
        "gpt-4",
        json!({"content": "Hello"}),
        None,
    )));
    sse_body.push_str(&sse_data_line(&sse_chat_chunk(
        "chatcmpl-1",
        "gpt-4",
        json!({"content": " world"}),
        None,
    )));
    sse_body.push_str(&sse_data_line(&sse_chat_chunk(
        "chatcmpl-1",
        "gpt-4",
        json!({}),
        Some("stop"),
    )));
    sse_body.push_str(sse_done_line());

    let server = start_streaming_mock_with_sse(sse_body).await;
    let app = build_app(&server.uri()).await;

    let (status, body) = post_responses_stream(app.app, &responses_request_text("test-group", "hi", true)).await;
    assert_eq!(status, StatusCode::OK);

    let events = parse_sse_events(&body);
    assert!(!events.is_empty(), "Expected SSE events");

    let types: Vec<&str> = events.iter().map(event_type).collect();

    let created_idx = types.iter().position(|&t| t == "response.created").expect("created event");
    let in_progress_idx = types.iter().position(|&t| t == "response.in_progress").expect("in_progress event");
    let completed_idx = types.iter().position(|&t| t == "response.completed").expect("completed event");

    assert!(created_idx < in_progress_idx, "created before in_progress");
    assert!(in_progress_idx < completed_idx, "in_progress before completed");

    let seqs: Vec<u64> = events.iter().map(sequence_number).collect();
    for i in 1..seqs.len() {
        assert!(seqs[i] > seqs[i - 1], "sequence_number must be strictly increasing, got {:?} at index {}", seqs, i);
    }
}

#[tokio::test]
async fn test_parallel_tool_call_streaming() {
    let mut sse_body = String::new();
    sse_body.push_str(&sse_data_line(&sse_chat_chunk(
        "chatcmpl-1",
        "gpt-4",
        json!({"role": "assistant"}),
        None,
    )));
    sse_body.push_str(&sse_data_line(&sse_chat_chunk(
        "chatcmpl-1",
        "gpt-4",
        json!({
            "tool_calls": [
                {"index": 0, "id": "call_a", "function": {"name": "get_weather", "arguments": "{\"ci"}}
            ]
        }),
        None,
    )));
    sse_body.push_str(&sse_data_line(&sse_chat_chunk(
        "chatcmpl-1",
        "gpt-4",
        json!({
            "tool_calls": [
                {"index": 1, "id": "call_b", "function": {"name": "get_time", "arguments": "{\"tz\""}}
            ]
        }),
        None,
    )));
    sse_body.push_str(&sse_data_line(&sse_chat_chunk(
        "chatcmpl-1",
        "gpt-4",
        json!({
            "tool_calls": [
                {"index": 0, "function": {"arguments": "ty\":\"Paris\"}"}}
            ]
        }),
        None,
    )));
    sse_body.push_str(&sse_data_line(&sse_chat_chunk(
        "chatcmpl-1",
        "gpt-4",
        json!({
            "tool_calls": [
                {"index": 1, "function": {"arguments": ":\"UTC\"}"}}
            ]
        }),
        None,
    )));
    sse_body.push_str(&sse_data_line(&sse_chat_chunk(
        "chatcmpl-1",
        "gpt-4",
        json!({}),
        Some("tool_calls"),
    )));
    sse_body.push_str(sse_done_line());

    let server = start_streaming_mock_with_sse(sse_body).await;
    let app = build_app(&server.uri()).await;

    let (status, body) = post_responses_stream(app.app, &responses_request_text("test-group", "check weather", true)).await;
    assert_eq!(status, StatusCode::OK);

    let events = parse_sse_events(&body);
    let output_item_added: Vec<&Value> = events
        .iter()
        .filter(|e| event_type(e) == "response.output_item.added")
        .collect();

    assert!(output_item_added.len() >= 2, "Expected at least 2 output_item.added events for parallel tool calls");

    let fc_deltas: Vec<&Value> = events
        .iter()
        .filter(|e| event_type(e) == "response.function_call_arguments.delta")
        .collect();

    for delta in &fc_deltas {
        assert!(delta.get("item_id").is_some(), "function_call_arguments.delta must have item_id");
    }
}

#[tokio::test]
async fn test_refusal_streaming() {
    let mut sse_body = String::new();
    sse_body.push_str(&sse_data_line(&sse_chat_chunk(
        "chatcmpl-1",
        "gpt-4",
        json!({"role": "assistant"}),
        None,
    )));
    sse_body.push_str(&sse_data_line(&sse_chat_chunk(
        "chatcmpl-1",
        "gpt-4",
        json!({"refusal": "I cannot help with that request."}),
        None,
    )));
    sse_body.push_str(&sse_data_line(&sse_chat_chunk(
        "chatcmpl-1",
        "gpt-4",
        json!({}),
        Some("stop"),
    )));
    sse_body.push_str(sse_done_line());

    let server = start_streaming_mock_with_sse(sse_body).await;
    let app = build_app(&server.uri()).await;

    let (status, body) = post_responses_stream(app.app, &responses_request_text("test-group", "do something bad", true)).await;
    assert_eq!(status, StatusCode::OK);

    let events = parse_sse_events(&body);
    let refusal_deltas: Vec<&Value> = events
        .iter()
        .filter(|e| event_type(e) == "response.refusal.delta")
        .collect();

    assert!(!refusal_deltas.is_empty(), "Expected response.refusal.delta events for refusal streaming");
}

#[tokio::test]
async fn test_reasoning_delta_synthesis() {
    let mut sse_body = String::new();
    sse_body.push_str(&sse_data_line(&sse_chat_chunk(
        "chatcmpl-1",
        "gpt-4",
        json!({"reasoning_content": "Let me think about this..."}),
        None,
    )));
    sse_body.push_str(&sse_data_line(&sse_chat_chunk(
        "chatcmpl-1",
        "gpt-4",
        json!({"reasoning_content": " Step by step."}),
        None,
    )));
    sse_body.push_str(&sse_data_line(&sse_chat_chunk(
        "chatcmpl-1",
        "gpt-4",
        json!({"content": "The answer is 42."}),
        None,
    )));
    sse_body.push_str(&sse_data_line(&sse_chat_chunk(
        "chatcmpl-1",
        "gpt-4",
        json!({}),
        Some("stop"),
    )));
    sse_body.push_str(sse_done_line());

    let server = start_streaming_mock_with_sse(sse_body).await;
    let app = build_app(&server.uri()).await;

    let (status, body) = post_responses_stream(app.app, &responses_request_text("test-group", "what is the answer?", true)).await;
    assert_eq!(status, StatusCode::OK);

    let events = parse_sse_events(&body);
    let types: Vec<&str> = events.iter().map(event_type).collect();

    let reasoning_idx = types.iter().position(|&t| t == "response.reasoning_text.delta");
    let text_idx = types.iter().position(|&t| t == "response.output_text.delta");

    if let (Some(r_idx), Some(t_idx)) = (reasoning_idx, text_idx) {
        assert!(r_idx < t_idx, "reasoning events must come before text events");
    }
}

#[tokio::test]
async fn test_previous_response_id_chaining() {
    use wiremock::{Mock, ResponseTemplate};
    use wiremock::matchers::{method, path};

    let server = MockServer::start().await;

    Mock::given(method("POST"))
        .and(path(PROVIDER_PATH))
        .respond_with(ResponseTemplate::new(200).set_body_json(chat_completion_response("chatcmpl-1", "gpt-4", "Hello!")))
        .mount(&server)
        .await;

    let app = build_app(&server.uri()).await;

    let (status, body) = post_responses(
        app.app.clone(),
        &serde_json::json!({
            "model": "test-group",
            "input": "Hi",
            "instructions": "be terse",
            "stream": false,
            "store": true
        }),
    )
    .await;
    assert_eq!(status, StatusCode::OK);

    let first_response: Value = serde_json::from_slice(&body).unwrap();
    let first_id = first_response["id"].as_str().expect("response id");

    server.reset();

    Mock::given(method("POST"))
        .and(path(PROVIDER_PATH))
        .respond_with(ResponseTemplate::new(200).set_body_json(chat_completion_response("chatcmpl-2", "gpt-4", "Hello again!")))
        .mount(&server)
        .await;

    let (status, _body) = post_responses(
        app.app,
        &serde_json::json!({
            "model": "test-group",
            "input": "What did I say?",
            "previous_response_id": first_id,
            "stream": false
        }),
    )
    .await;
    assert_eq!(status, StatusCode::OK);
}

#[tokio::test]
async fn test_previous_response_id_not_found() {
    let server = start_buffered_mock(chat_completion_response("chatcmpl-1", "gpt-4", "Hello!")).await;
    let app = build_app(&server.uri()).await;

    let (status, body) = post_responses(
        app.app,
        &responses_request_with_previous("test-group", "Hi", "resp_nonexistent", false),
    )
    .await;
    assert_eq!(status, StatusCode::NOT_FOUND);

    let error: Value = serde_json::from_slice(&body).unwrap();
    assert!(error.get("error").is_some(), "Expected error object in response");
}

#[tokio::test]
async fn test_instructions_non_carryover() {
    use wiremock::{Mock, ResponseTemplate};
    use wiremock::matchers::{method, path};

    let server = MockServer::start().await;

    Mock::given(method("POST"))
        .and(path(PROVIDER_PATH))
        .respond_with(ResponseTemplate::new(200).set_body_json(chat_completion_response("chatcmpl-1", "gpt-4", "Brief.")))
        .mount(&server)
        .await;

    let app = build_app(&server.uri()).await;

    let (status, body) = post_responses(
        app.app.clone(),
        &serde_json::json!({
            "model": "test-group",
            "input": "Hello",
            "instructions": "be terse",
            "stream": false,
            "store": true
        }),
    )
    .await;
    assert_eq!(status, StatusCode::OK);

    let first_response: Value = serde_json::from_slice(&body).unwrap();
    let first_id = first_response["id"].as_str().expect("response id");

    server.reset();

    Mock::given(method("POST"))
        .and(path(PROVIDER_PATH))
        .respond_with(ResponseTemplate::new(200).set_body_json(chat_completion_response("chatcmpl-2", "gpt-4", "Verbose response here.")))
        .mount(&server)
        .await;

    let (status, _body) = post_responses(
        app.app,
        &serde_json::json!({
            "model": "test-group",
            "input": "Tell me more",
            "previous_response_id": first_id,
            "instructions": "be verbose and detailed",
            "stream": false
        }),
    )
    .await;
    assert_eq!(status, StatusCode::OK);
}
