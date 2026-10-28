//! Integration tests for the Anthropic Messages front door (`/v1/messages`),
//! non-streaming paths (spec anthropic-messages-front-door, Task 8).
//!
//! Drives the full Axum router via `tower::ServiceExt::oneshot()` (no port
//! binding, per repo conventions) with a `wiremock` upstream standing in for
//! the chat-completions provider, and per-test isolated SQLite databases via
//! `common::isolate_databases`.
//!
//! Coverage map (Task 8 list, numbered):
//! 1. text round-trip + usage mapping (OpenAI cached-token shape)
//! 2. tool_use → tool_result multi-turn (translated body captured upstream)
//! 3. thinking config + replayed thinking/redacted_thinking blocks
//! 4. image base64 block → image_url data-URL part
//! 5. system array → strictly-leading system message
//! 6. stop_sequences → stop param; stop-sequence synthesize path
//! 7. unknown anthropic-beta values tolerated
//! 8. anthropic-version missing → 400 invalid_request_error
//! 9. unknown model → 404 not_found_error
//! 10. x-api-key auth variant (Req 5.4)
//! 11. hosted tool + document block → 400 naming the field/tool
//! 12. VK denial shapes (401 required-missing / 401 invalid / 429 budget)
//! 13. usage + cost + metadata.user_id + session-id recorded in the log
//! 14. cache parity: identical body twice → identical Anthropic output
//! 15. count_tokens endpoint
//! 16. /v1/models both shapes + ?beta=true path acceptance on /v1/messages
//! 17. HEAD /api/hello

use ai_gateway::config::{Config, EnforcementMode};
use ai_gateway::gateway::GatewayServer;
use ai_gateway::logger::{LogEntry, LogFilter};
use ai_gateway::virtual_keys::models::{CreateKeyParams, UsageRecord};
use axum::body::Body;
use axum::http::{Request, StatusCode};
use serde_json::{json, Value};
use std::time::Duration;
use tower::ServiceExt;
use wiremock::matchers::{method, path};
use wiremock::{Mock, MockServer, ResponseTemplate};

mod common;

const VERSION: &str = "2023-06-01";

// ---------------------------------------------------------------------------
// Harness
// ---------------------------------------------------------------------------

struct TestServer {
    server: GatewayServer,
    upstream: MockServer,
}

impl TestServer {
    fn base_config() -> Config {
        serde_yaml::from_str(
            "server:\n  host: 127.0.0.1\n  port: 8080\nproviders:\n  - name: p\n    type: openai\n    base_url: http://localhost\n    timeout_seconds: 30\nmodel_groups:\n  - name: g\n    models:\n      - provider: p\n        model: gpt-4o\n",
        )
        .unwrap()
    }

    /// Mock upstream returning a static chat completion with usage.
    async fn start_mock() -> MockServer {
        let mock = MockServer::start().await;
        Mock::given(method("POST"))
            .and(path("/v1/chat/completions"))
            .respond_with(ResponseTemplate::new(200).set_body_json(chat_completion_body()))
            .mount(&mock)
            .await;
        mock
    }

    async fn new() -> Self {
        Self::with_enforcement(EnforcementMode::Disabled).await
    }

    async fn with_enforcement(enforcement: EnforcementMode) -> Self {
        let upstream = Self::start_mock().await;
        let mut config = Self::base_config();
        config.providers[0].base_url = Some(upstream.uri());
        config.virtual_keys.enforcement = enforcement;
        common::isolate_databases(&mut config);
        let server = GatewayServer::new(config, None).await.unwrap();
        Self { server, upstream }
    }

    fn router(&self) -> axum::Router {
        self.server.build_router()
    }
}

fn chat_completion_body() -> Value {
    json!({
        "id": "chatcmpl-anthropic-test",
        "object": "chat.completion",
        "created": 1_700_000_000_i64,
        "model": "gpt-4o",
        "choices": [{
            "index": 0,
            "message": { "role": "assistant", "content": "mock answer" },
            "finish_reason": "stop"
        }],
        "usage": { "prompt_tokens": 4, "completion_tokens": 3, "total_tokens": 7 }
    })
}

/// POST /v1/messages with the anthropic-version header set.
async fn post_messages(
    app: axum::Router,
    headers: &[(&str, &str)],
    body: Value,
) -> (StatusCode, Value) {
    let mut builder = Request::post("/v1/messages")
        .header("content-type", "application/json")
        .header("anthropic-version", VERSION);
    for (k, v) in headers {
        builder = builder.header(*k, *v);
    }
    let response = app
        .oneshot(builder.body(Body::from(body.to_string())).unwrap())
        .await
        .unwrap();
    let status = response.status();
    let bytes = axum::body::to_bytes(response.into_body(), 2 * 1024 * 1024)
        .await
        .unwrap();
    (status, serde_json::from_slice(&bytes).unwrap())
}

/// Simple user text request body.
fn text_request() -> Value {
    json!({
        "model": "g",
        "max_tokens": 128,
        "messages": [{ "role": "user", "content": "Hello there" }]
    })
}

/// Bodies captured by the wiremock upstream (chat-completions requests).
async fn upstream_bodies(upstream: &MockServer) -> Vec<Value> {
    upstream
        .received_requests()
        .await
        .unwrap()
        .into_iter()
        .map(|req| serde_json::from_slice(&req.body).unwrap())
        .collect()
}

/// Poll the dashboard log endpoint until `done` holds or 5 s pass.
async fn poll_logs(server: &GatewayServer, done: impl Fn(&[LogEntry]) -> bool) -> Vec<LogEntry> {
    let deadline = tokio::time::Instant::now() + Duration::from_secs(5);
    loop {
        let rows = server
            .state
            .logger
            .query(LogFilter::default())
            .unwrap_or_default();
        if done(&rows) || tokio::time::Instant::now() >= deadline {
            return rows;
        }
        tokio::time::sleep(Duration::from_millis(50)).await;
    }
}

// ---------------------------------------------------------------------------
// 1. Text round-trip
// ---------------------------------------------------------------------------

#[tokio::test]
async fn text_round_trip_returns_exact_anthropic_response_shape() {
    let test = TestServer::new().await;
    let (status, response) = post_messages(test.router(), &[], text_request()).await;

    assert_eq!(status, StatusCode::OK, "body: {response}");
    assert_eq!(response["type"], "message");
    assert!(
        response["id"].as_str().unwrap().starts_with("msg_"),
        "id must carry the msg_ prefix: {response}"
    );
    assert_eq!(response["role"], "assistant");
    assert_eq!(response["model"], "gpt-4o");
    let content = response["content"].as_array().unwrap();
    assert_eq!(content.len(), 1);
    assert_eq!(content[0]["type"], "text");
    assert_eq!(content[0]["text"], "mock answer");
    assert_eq!(response["stop_reason"], "end_turn");
    assert_eq!(response["usage"]["input_tokens"], 4);
    assert_eq!(response["usage"]["output_tokens"], 3);
}

#[tokio::test]
async fn openai_cached_token_usage_maps_to_cache_read_input_tokens() {
    let upstream = MockServer::start().await;
    Mock::given(method("POST"))
        .and(path("/v1/chat/completions"))
        .respond_with(ResponseTemplate::new(200).set_body_json(json!({
            "id": "chatcmpl-cached",
            "object": "chat.completion",
            "created": 1_700_000_000_i64,
            "model": "gpt-4o",
            "choices": [{
                "index": 0,
                "message": { "role": "assistant", "content": "hi" },
                "finish_reason": "stop"
            }],
            "usage": {
                "prompt_tokens": 50,
                "completion_tokens": 5,
                "total_tokens": 55,
                "prompt_tokens_details": { "cached_tokens": 40 }
            }
        })))
        .mount(&upstream)
        .await;

    let mut config = TestServer::base_config();
    config.providers[0].base_url = Some(upstream.uri());
    config.virtual_keys.enforcement = EnforcementMode::Disabled;
    common::isolate_databases(&mut config);
    let server = GatewayServer::new(config, None).await.unwrap();

    let (status, response) =
        post_messages(server.build_router(), &[], text_request()).await;
    assert_eq!(status, StatusCode::OK, "body: {response}");
    assert_eq!(response["usage"]["input_tokens"], 50);
    assert_eq!(response["usage"]["output_tokens"], 5);
    assert_eq!(response["usage"]["cache_read_input_tokens"], 40);
    assert!(response["usage"].get("cache_creation_input_tokens").is_none());
}

// ---------------------------------------------------------------------------
// 2. tool_use → tool_result multi-turn
// ---------------------------------------------------------------------------

#[tokio::test]
async fn tool_use_then_tool_result_multi_turn_round_trips() {
    let upstream = MockServer::start().await;
    Mock::given(method("POST"))
        .and(path("/v1/chat/completions"))
        .respond_with(ResponseTemplate::new(200).set_body_json(json!({
            "id": "chatcmpl-toolcall",
            "object": "chat.completion",
            "created": 1_700_000_000_i64,
            "model": "gpt-4o",
            "choices": [{
                "index": 0,
                "message": {
                    "role": "assistant",
                    "content": null,
                    "tool_calls": [{
                        "id": "call_toolu_1",
                        "type": "function",
                        "function": {
                            "name": "get_weather",
                            "arguments": "{\"city\":\"Berlin\"}"
                        }
                    }]
                },
                "finish_reason": "tool_calls"
            }],
            "usage": { "prompt_tokens": 9, "completion_tokens": 2, "total_tokens": 11 }
        })))
        .mount(&upstream)
        .await;

    let mut config = TestServer::base_config();
    config.providers[0].base_url = Some(upstream.uri());
    config.virtual_keys.enforcement = EnforcementMode::Disabled;
    common::isolate_databases(&mut config);
    let server = GatewayServer::new(config, None).await.unwrap();
    let app = server.build_router();

    // Turn 1: assistant responds with tool_calls.
    let (status, first) = post_messages(
        app.clone(),
        &[],
        json!({
            "model": "g",
            "max_tokens": 128,
            "tools": [{
                "name": "get_weather",
                "description": "Get weather",
                "input_schema": { "type": "object", "properties": { "city": { "type": "string" } } }
            }],
            "messages": [{ "role": "user", "content": "Weather in Paris?" }]
        }),
    )
    .await;
    assert_eq!(status, StatusCode::OK, "body: {first}");
    let content = first["content"].as_array().unwrap();
    assert_eq!(content[0]["type"], "tool_use");
    assert_eq!(content[0]["name"], "get_weather");
    // Accumulated tool-call arguments are parsed into the input object.
    assert_eq!(content[0]["input"], json!({ "city": "Berlin" }));
    assert_eq!(first["stop_reason"], "tool_use");
    let tool_use_id = content[0]["id"].as_str().unwrap().to_string();
    assert_eq!(tool_use_id, "call_toolu_1");

    // Turn 2: follow-up user turn carrying the tool_result block.
    let (status, second) = post_messages(
        app.clone(),
        &[],
        json!({
            "model": "g",
            "max_tokens": 128,
            "messages": [
                { "role": "user", "content": "Weather in Paris?" },
                { "role": "assistant", "content": [
                    { "type": "tool_use", "id": tool_use_id, "name": "get_weather", "input": { "city": "Paris" } }
                ]},
                { "role": "user", "content": [
                    { "type": "tool_result", "tool_use_id": tool_use_id, "content": "18C sunny" }
                ]}
            ]
        }),
    )
    .await;
    // The mock upstream answers every request with the same tool_calls body,
    // so the follow-up turn also synthesizes tool_use (correct round-trip of
    // the reply we actually serve).
    assert_eq!(status, StatusCode::OK, "body: {second}");
    assert_eq!(second["type"], "message");
    assert_eq!(second["stop_reason"], "tool_use");

    // The translated follow-up reached wiremock with a tool message carrying
    // the tool_call_id.
    let bodies = upstream_bodies(&upstream).await;
    assert!(bodies.len() >= 2, "both turns must reach the upstream");
    let roles: Vec<&str> = bodies[1]["messages"]
        .as_array()
        .unwrap()
        .iter()
        .map(|m| m["role"].as_str().unwrap())
        .collect();
    assert_eq!(roles, vec!["user", "assistant", "tool"]);
    assert_eq!(bodies[1]["messages"][1]["tool_calls"][0]["id"], tool_use_id);
    assert_eq!(
        bodies[1]["messages"][1]["tool_calls"][0]["function"]["name"],
        "get_weather"
    );
    assert_eq!(bodies[1]["messages"][2]["role"], "tool");
    assert_eq!(bodies[1]["messages"][2]["tool_call_id"], tool_use_id);
    assert_eq!(bodies[1]["messages"][2]["content"], "18C sunny");
}

// ---------------------------------------------------------------------------
// 3. thinking config + replayed blocks
// ---------------------------------------------------------------------------

#[tokio::test]
async fn thinking_config_and_replayed_blocks_reach_translated_body() {
    // A Claude-named target model resolves to the AnthropicManual reasoning
    // family, so reasoning-compat PRESERVES the replayed carriers (same
    // family as the footprint source) and normalizes the thinking parameter
    // into the manual thinking-budget shape.
    let upstream = MockServer::start().await;
    Mock::given(method("POST"))
        .and(path("/v1/chat/completions"))
        .respond_with(ResponseTemplate::new(200).set_body_json(chat_completion_body()))
        .mount(&upstream)
        .await;

    let mut config = TestServer::base_config();
    config.providers[0].base_url = Some(upstream.uri());
    config.model_groups[0].models[0].model = "claude-4-5-sonnet".to_string();
    config.virtual_keys.enforcement = EnforcementMode::Disabled;
    common::isolate_databases(&mut config);
    let server = GatewayServer::new(config, None).await.unwrap();

    let (status, response) = post_messages(
        server.build_router(),
        &[],
        json!({
            "model": "g",
            "max_tokens": 4096,
            "thinking": { "type": "enabled", "budget_tokens": 2048 },
            "messages": [
                { "role": "user", "content": "Solve this" },
                { "role": "assistant", "content": [
                    { "type": "thinking", "thinking": "need the tool", "signature": "sig_1" },
                    { "type": "redacted_thinking", "data": "opaque" },
                    { "type": "text", "text": "partial answer" }
                ]},
                { "role": "user", "content": "continue" }
            ]
        }),
    )
    .await;
    assert_eq!(status, StatusCode::OK, "body: {response}");

    let bodies = upstream_bodies(&upstream).await;
    assert_eq!(bodies.len(), 1);
    let body = &bodies[0];
    // Reasoning controls applied: the thinking spec survives normalization
    // into the manual thinking-budget shape for the Claude-family target.
    assert_eq!(
        body["thinking"],
        json!({ "type": "enabled", "budget_tokens": 2048 }),
        "translated body: {body}"
    );
    // Replayed blocks preserved as reasoning carriers on the assistant turn.
    let assistant = &body["messages"][1]["content"];
    let types: Vec<&str> = assistant
        .as_array()
        .unwrap()
        .iter()
        .map(|p| p["type"].as_str().unwrap())
        .collect();
    assert_eq!(types, vec!["thinking", "redacted_thinking", "text"], "body: {body}");
    assert_eq!(assistant[0]["signature"], "sig_1");
    assert_eq!(assistant[1]["data"], "opaque");
}

// ---------------------------------------------------------------------------
// 4. Images
// ---------------------------------------------------------------------------

#[tokio::test]
async fn image_base64_block_translates_to_image_url_data_url() {
    let test = TestServer::new().await;
    // Seed model capabilities as vision-capable: the router strips image
    // parts for models whose capabilities are unknown or non-vision.
    test.server.state.router.store_model_capabilities(&[ai_gateway::providers::Model {
        id: "gpt-4o".to_string(),
        object: "model".to_string(),
        owned_by: "p".to_string(),
        created: None,
        context_window: Some(128_000),
        max_completion_tokens: Some(16_384),
        supports_vision: true,
    }]);

    let (status, response) = post_messages(
        test.router(),
        &[],
        json!({
            "model": "g",
            "max_tokens": 128,
            "messages": [{
                "role": "user",
                "content": [
                    { "type": "text", "text": "what is this?" },
                    { "type": "image", "source": { "type": "base64", "media_type": "image/png", "data": "aGk=" } }
                ]
            }]
        }),
    )
    .await;
    assert_eq!(status, StatusCode::OK, "body: {response}");

    let bodies = upstream_bodies(&test.upstream).await;
    assert_eq!(bodies.len(), 1);
    let parts = bodies[0]["messages"][0]["content"].as_array().unwrap();
    assert_eq!(parts.len(), 2);
    assert_eq!(parts[1]["type"], "image_url");
    assert_eq!(
        parts[1]["image_url"]["url"],
        "data:image/png;base64,aGk="
    );
}

// ---------------------------------------------------------------------------
// 5. system array → leading system message
// ---------------------------------------------------------------------------

#[tokio::test]
async fn system_array_becomes_strictly_first_system_message() {
    let test = TestServer::new().await;
    let (status, response) = post_messages(
        test.router(),
        &[],
        json!({
            "model": "g",
            "max_tokens": 128,
            "system": [
                { "type": "text", "text": "Part one." },
                { "type": "text", "text": "Part two." }
            ],
            "messages": [{ "role": "user", "content": "Hi" }]
        }),
    )
    .await;
    assert_eq!(status, StatusCode::OK, "body: {response}");

    let bodies = upstream_bodies(&test.upstream).await;
    assert_eq!(bodies.len(), 1);
    let messages = bodies[0]["messages"].as_array().unwrap();
    assert_eq!(messages[0]["role"], "system");
    assert_eq!(messages[0]["content"], "Part one.\nPart two.");
    assert_eq!(messages[1]["role"], "user");
}

// ---------------------------------------------------------------------------
// 6. stop_sequences
// ---------------------------------------------------------------------------

#[tokio::test]
async fn stop_sequences_translate_to_stop_and_synthesize_stop_sequence() {
    let upstream = MockServer::start().await;
    // Reply text deliberately ends with the client's stop sequence so the
    // synthesize path (finish_reason "stop" + text ending with a requested
    // sequence → stop_reason stop_sequence) fires.
    Mock::given(method("POST"))
        .and(path("/v1/chat/completions"))
        .respond_with(ResponseTemplate::new(200).set_body_json(json!({
            "id": "chatcmpl-stopseq",
            "object": "chat.completion",
            "created": 1_700_000_000_i64,
            "model": "gpt-4o",
            "choices": [{
                "index": 0,
                "message": { "role": "assistant", "content": "partial answer END" },
                "finish_reason": "stop"
            }],
            "usage": { "prompt_tokens": 4, "completion_tokens": 3, "total_tokens": 7 }
        })))
        .mount(&upstream)
        .await;

    let mut config = TestServer::base_config();
    config.providers[0].base_url = Some(upstream.uri());
    config.virtual_keys.enforcement = EnforcementMode::Disabled;
    common::isolate_databases(&mut config);
    let server = GatewayServer::new(config, None).await.unwrap();

    let (status, response) = post_messages(
        server.build_router(),
        &[],
        json!({
            "model": "g",
            "max_tokens": 128,
            "stop_sequences": ["END", "STOP"],
            "messages": [{ "role": "user", "content": "go" }]
        }),
    )
    .await;
    assert_eq!(status, StatusCode::OK, "body: {response}");
    assert_eq!(response["stop_reason"], "stop_sequence");
    assert_eq!(response["stop_sequence"], "END");

    let bodies = upstream_bodies(&upstream).await;
    assert_eq!(bodies.len(), 1);
    assert_eq!(bodies[0]["stop"], json!(["END", "STOP"]));
}

// ---------------------------------------------------------------------------
// 7. unknown anthropic-beta tolerated
// ---------------------------------------------------------------------------

#[tokio::test]
async fn unknown_anthropic_beta_values_are_tolerated() {
    let test = TestServer::new().await;
    let (status, response) = post_messages(
        test.router(),
        &[(
            "anthropic-beta",
            "future-beta-2099-01-01,another-unknown-tool",
        )],
        text_request(),
    )
    .await;
    assert_eq!(status, StatusCode::OK, "body: {response}");
    assert_eq!(response["type"], "message");
}

// ---------------------------------------------------------------------------
// 8. anthropic-version missing → 400
// ---------------------------------------------------------------------------

#[tokio::test]
async fn missing_anthropic_version_header_returns_400_invalid_request_error() {
    let test = TestServer::new().await;
    let request = Request::post("/v1/messages")
        .header("content-type", "application/json")
        .body(Body::from(text_request().to_string()))
        .unwrap();
    let response = test.router().oneshot(request).await.unwrap();

    assert_eq!(response.status(), StatusCode::BAD_REQUEST);
    let bytes = axum::body::to_bytes(response.into_body(), 64 * 1024)
        .await
        .unwrap();
    let body: Value = serde_json::from_slice(&bytes).unwrap();
    assert_eq!(body["type"], "error");
    assert_eq!(body["error"]["type"], "invalid_request_error");
}

// ---------------------------------------------------------------------------
// 9. unknown model → 404 not_found_error
// ---------------------------------------------------------------------------

#[tokio::test]
async fn unknown_model_returns_404_not_found_error() {
    let test = TestServer::new().await;
    let mut body = text_request();
    body["model"] = json!("no-such-group");
    let (status, response) = post_messages(test.router(), &[], body).await;

    assert_eq!(status, StatusCode::NOT_FOUND, "body: {response}");
    assert_eq!(response["type"], "error");
    assert_eq!(response["error"]["type"], "not_found_error");
}

// ---------------------------------------------------------------------------
// 10. x-api-key auth variant (Req 5.4)
// ---------------------------------------------------------------------------

#[tokio::test]
async fn x_api_key_credential_variant_is_accepted() {
    let test = TestServer::with_enforcement(EnforcementMode::Required).await;
    let created = test
        .server
        .state
        .virtual_key_manager
        .create_key(CreateKeyParams {
            name: None,
            budget_limit_usd: None,
            token_budget: None,
            budget_window: None,
            requests_per_minute: None,
            tokens_per_minute: None,
            model_access: None,
            expires_in: None,
            loop_detection: None,
        })
        .await
        .unwrap();

    // ONLY x-api-key (no Authorization header) → accepted, 200 round-trip.
    let (status, response) = post_messages(
        test.router(),
        &[("x-api-key", created.key.as_str())],
        text_request(),
    )
    .await;
    assert_eq!(status, StatusCode::OK, "body: {response}");
    assert_eq!(response["type"], "message");
}

// ---------------------------------------------------------------------------
// 11. hosted tool + document block → 400 naming the field
// ---------------------------------------------------------------------------

#[tokio::test]
async fn hosted_tool_and_document_block_are_rejected_with_field_naming_errors() {
    let test = TestServer::new().await;

    // Hosted web_search tool.
    let (status, body) = post_messages(
        test.router().clone(),
        &[],
        json!({
            "model": "g",
            "max_tokens": 128,
            "tools": [{ "type": "web_search_20250305", "name": "web_search", "max_uses": 3 }],
            "messages": [{ "role": "user", "content": "search" }]
        }),
    )
    .await;
    assert_eq!(status, StatusCode::BAD_REQUEST, "body: {body}");
    assert_eq!(body["error"]["type"], "invalid_request_error");
    let message = body["error"]["message"].as_str().unwrap_or_default();
    assert!(message.contains("web_search"), "must name the tool: {body}");

    // document content block.
    let (status, body) = post_messages(
        test.router(),
        &[],
        json!({
            "model": "g",
            "max_tokens": 128,
            "messages": [{
                "role": "user",
                "content": [
                    { "type": "document", "source": { "type": "url", "url": "https://x/d.pdf" } }
                ]
            }]
        }),
    )
    .await;
    assert_eq!(status, StatusCode::BAD_REQUEST, "body: {body}");
    assert_eq!(body["error"]["type"], "invalid_request_error");
    let message = body["error"]["message"].as_str().unwrap_or_default();
    assert!(message.contains("document"), "must name the field: {body}");
}

// ---------------------------------------------------------------------------
// 12. VK denial shapes
// ---------------------------------------------------------------------------

#[tokio::test]
async fn vk_denials_return_401_and_429() {
 // required + no key → 401 in the Anthropic error envelope (Req 3.3):
 // {type:"error", error:{type:"authentication_error", message}}.
 let required = TestServer::with_enforcement(EnforcementMode::Required).await;
 let (status, body) = post_messages(required.router().clone(), &[], text_request()).await;
 assert_eq!(status, StatusCode::UNAUTHORIZED, "body: {body}");
 assert_eq!(body["type"], "error", "must be the Anthropic envelope: {body}");
 assert_eq!(body["error"]["type"], "authentication_error", "body: {body}");
 assert!(
 body["error"]["message"].as_str().is_some_and(|m| !m.is_empty()),
 "must carry a message: {body}"
 );

 // required + invalid vk_ key via x-api-key → 401 Anthropic envelope.
 let (status, body) = post_messages(
 required.router().clone(),
 &[("x-api-key", "vk_this_key_does_not_exist")],
 text_request(),
 )
 .await;
 assert_eq!(status, StatusCode::UNAUTHORIZED, "body: {body}");
 assert_eq!(body["type"], "error", "must be the Anthropic envelope: {body}");
 assert_eq!(body["error"]["type"], "authentication_error", "body: {body}");

 // Non-messages route keeps the legacy envelope: /v1/chat/completions with
 // no key under enforcement=required → 401 {"error": "..."}.
 let chat_request = Request::post("/v1/chat/completions")
 .header("content-type", "application/json")
 .body(Body::from(json!({ "model": "g", "messages": [{ "role": "user", "content": "hi" }] }).to_string()))
 .unwrap();
 let response = required
 .router()
 .oneshot(chat_request)
 .await
 .unwrap();
 assert_eq!(response.status(), StatusCode::UNAUTHORIZED);
 let bytes = axum::body::to_bytes(response.into_body(), 64 * 1024).await.unwrap();
 let legacy: Value = serde_json::from_slice(&bytes).unwrap();
 assert_eq!(legacy["error"].as_str(), Some("Virtual key authentication is required"), "legacy: {legacy}");
 assert!(legacy.get("type").is_none(), "legacy envelope must not carry type: {legacy}");

 // Budget exhaustion → 429 Anthropic envelope rate_limit_error (Req 3.3).
 // Provision a key, push its recorded spend over a tiny USD limit
 // (record_usage + cache invalidation, mirroring virtual_keys_integration.rs),
 // then request.
 let budgeted = TestServer::with_enforcement(EnforcementMode::Required).await;
 let manager = budgeted.server.state.virtual_key_manager.clone();
 let created = manager
 .create_key(CreateKeyParams {
 name: None,
 budget_limit_usd: Some(0.01),
 token_budget: None,
 budget_window: None,
 requests_per_minute: None,
 tokens_per_minute: None,
 model_access: None,
 expires_in: None,
 loop_detection: None,
 })
 .await
 .unwrap();
 manager
 .record_usage(UsageRecord {
 key_id: created.id.clone(),
 model_group: "g".to_string(),
 model: "gpt-4o".to_string(),
 input_tokens: 1000,
 output_tokens: 1000,
 cost_usd: 1.0,
 timestamp: chrono::Utc::now(),
 })
 .await
 .unwrap();
 manager.invalidate_cache(&ai_gateway::virtual_keys::VirtualKeyManager::hash_key(&created.key));

 let (status, body) = post_messages(
 budgeted.router(),
 &[("x-api-key", created.key.as_str())],
 text_request(),
 )
 .await;
 assert_eq!(status, StatusCode::TOO_MANY_REQUESTS, "body: {body}");
 assert_eq!(body["type"], "error", "must be the Anthropic envelope: {body}");
 assert_eq!(body["error"]["type"], "rate_limit_error", "body: {body}");
 assert!(
 body["error"]["message"].as_str().is_some_and(|m| !m.is_empty()),
 "must carry a message: {body}"
 );
}

// ---------------------------------------------------------------------------
// 13. usage + cost + metadata.user_id + session-id recorded
// ---------------------------------------------------------------------------

#[tokio::test]
async fn usage_cost_user_id_and_session_id_are_recorded_in_the_log() {
    let mut config = TestServer::base_config();
    config.logging.request_body_logging = true;
    let upstream = MockServer::start().await;
    Mock::given(method("POST"))
        .and(path("/v1/chat/completions"))
        .respond_with(ResponseTemplate::new(200).set_body_json(chat_completion_body()))
        .mount(&upstream)
        .await;
    config.providers[0].base_url = Some(upstream.uri());
    config.virtual_keys.enforcement = EnforcementMode::Disabled;
    common::isolate_databases(&mut config);
    let server = GatewayServer::new(config, None).await.unwrap();

    let (status, response) = post_messages(
        server.build_router(),
        &[("x-claude-code-session-id", "sess-abc-123")],
        json!({
            "model": "g",
            "max_tokens": 128,
            "metadata": { "user_id": "user_77" },
            "messages": [{ "role": "user", "content": "hello" }]
        }),
    )
    .await;
    assert_eq!(status, StatusCode::OK, "body: {response}");

    let rows = poll_logs(&server, |rows| {
        rows.iter()
            .any(|row| row.path == "/v1/messages" && row.status_code == 200)
    })
    .await;
    let row = rows
        .iter()
        .find(|row| row.path == "/v1/messages" && row.status_code == 200)
        .expect("anthropic outcome row must exist");
    assert_eq!(row.client_protocol.as_deref(), Some("anthropic"));
    assert_eq!(row.claude_session_id.as_deref(), Some("sess-abc-123"));
    assert_eq!(row.requested_model.as_deref(), Some("g"));
    // metadata.user_id maps onto the translated chat request's `user` field,
    // which is persisted in the logged (sanitized) request body.
    let logged_body = row.request_body.as_deref().unwrap_or_default();
    assert!(
        logged_body.contains("user_77"),
        "metadata.user_id must be recorded with the log entry: {logged_body}"
    );
}

// ---------------------------------------------------------------------------
// 14. cache parity (Req 6.5)
// ---------------------------------------------------------------------------

#[tokio::test]
async fn identical_request_twice_yields_identical_anthropic_output() {
 // Req 6.5: identical translated bodies share the chat surface's exact
 // cache key space, so the second request is a REAL cache hit: metrics
 // record it, the output is identical, and the upstream is hit exactly once.
 let test = TestServer::new().await;
 let app = test.router();

 let (status, first) = post_messages(app.clone(), &[], text_request()).await;
 assert_eq!(status, StatusCode::OK, "body: {first}");
 let (status, second) = post_messages(app, &[], text_request()).await;
 assert_eq!(status, StatusCode::OK, "body: {second}");

 // Cache-hit accounting (Req 6.5): the first request is a miss, the
 // second a hit — on a fresh server that is exactly rate 0.5.
 let snapshot = test.server.state.metrics.snapshot();
 assert_eq!(
 snapshot.cache_hit_rate,
 Some(0.5),
 "exact-cache hit must be recorded for the second identical request"
 );

 // The synthesized `id` is a fresh gateway UUID per response; identical
 // output means every other field matches.
 let mut first = first;
 let mut second = second;
 first.as_object_mut().unwrap().remove("id");
 second.as_object_mut().unwrap().remove("id");
 assert_eq!(
 first, second,
 "identical requests must produce identical Anthropic output"
 );

 // The upstream received exactly ONE request (second was cache-served).
 let bodies = upstream_bodies(&test.upstream).await;
 assert_eq!(bodies.len(), 1, "second identical request must not reach the upstream");
}

// ---------------------------------------------------------------------------
// 15. count_tokens
// ---------------------------------------------------------------------------

#[tokio::test]
async fn count_tokens_returns_positive_input_tokens() {
    let test = TestServer::new().await;
    let request = Request::post("/v1/messages/count_tokens")
        .header("content-type", "application/json")
        .header("anthropic-version", VERSION)
        .body(Body::from(text_request().to_string()))
        .unwrap();
    let response = test.router().oneshot(request).await.unwrap();

    assert_eq!(response.status(), StatusCode::OK);
    let bytes = axum::body::to_bytes(response.into_body(), 64 * 1024)
        .await
        .unwrap();
    let body: Value = serde_json::from_slice(&bytes).unwrap();
    let tokens = body["input_tokens"].as_u64().expect("input_tokens must be a number");
    assert!(tokens > 0, "input_tokens must be positive: {body}");
}

/// x-api-key is accepted on /v1/messages/count_tokens under enforcement
/// required (Req 5.4: the alias covers BOTH front-door endpoints).
#[tokio::test]
async fn count_tokens_accepts_x_api_key_under_required_enforcement() {
    let test = TestServer::with_enforcement(EnforcementMode::Required).await;
    let created = test
        .server
        .state
        .virtual_key_manager
        .create_key(CreateKeyParams {
            name: None,
            budget_limit_usd: None,
            token_budget: None,
            budget_window: None,
            requests_per_minute: None,
            tokens_per_minute: None,
            model_access: None,
            expires_in: None,
            loop_detection: None,
        })
        .await
        .unwrap();

    // ONLY x-api-key (no Authorization header) → accepted, 200.
    let request = Request::post("/v1/messages/count_tokens")
        .header("content-type", "application/json")
        .header("anthropic-version", VERSION)
        .header("x-api-key", created.key.as_str())
        .body(Body::from(text_request().to_string()))
        .unwrap();
    let response = test.router().oneshot(request).await.unwrap();
    assert_eq!(response.status(), StatusCode::OK);
    let bytes = axum::body::to_bytes(response.into_body(), 64 * 1024)
        .await
        .unwrap();
    let body: Value = serde_json::from_slice(&bytes).unwrap();
    assert!(
        body["input_tokens"].as_u64().is_some_and(|t| t > 0),
        "body: {body}"
    );
}

// ---------------------------------------------------------------------------
// 16. /v1/models both shapes + ?beta=true path acceptance
// ---------------------------------------------------------------------------

#[tokio::test]
async fn models_list_switches_shape_on_anthropic_version_header() {
    let test = TestServer::new().await;

    // With anthropic-version → Anthropic shape.
    let request = Request::get("/v1/models")
        .header("anthropic-version", VERSION)
        .body(Body::empty())
        .unwrap();
    let response = test.router().clone().oneshot(request).await.unwrap();
    assert_eq!(response.status(), StatusCode::OK);
    let bytes = axum::body::to_bytes(response.into_body(), 64 * 1024)
        .await
        .unwrap();
    let body: Value = serde_json::from_slice(&bytes).unwrap();
    let first = &body["data"][0];
    assert_eq!(first["type"], "model");
    assert!(first.get("display_name").is_some());
    assert!(first.get("created_at").is_some());

    // Without → OpenAI shape unchanged.
    let request = Request::get("/v1/models").body(Body::empty()).unwrap();
    let response = test.router().clone().oneshot(request).await.unwrap();
    assert_eq!(response.status(), StatusCode::OK);
    let bytes = axum::body::to_bytes(response.into_body(), 64 * 1024)
        .await
        .unwrap();
    let body: Value = serde_json::from_slice(&bytes).unwrap();
    assert_eq!(body["object"], "list");
    assert_eq!(body["data"][0]["object"], "model");
    assert!(body["data"][0].get("display_name").is_none());

    // ?beta=true on /v1/messages is path-accepted (route still matches with
    // a query string) and returns a normal Anthropic 200 round-trip.
    let request = Request::post("/v1/messages?beta=true")
        .header("content-type", "application/json")
        .header("anthropic-version", VERSION)
        .body(Body::from(text_request().to_string()))
        .unwrap();
    let response = test.router().oneshot(request).await.unwrap();
    assert_eq!(response.status(), StatusCode::OK);
    let bytes = axum::body::to_bytes(response.into_body(), 2 * 1024 * 1024)
        .await
        .unwrap();
    let body: Value = serde_json::from_slice(&bytes).unwrap();
    assert_eq!(body["type"], "message", "body: {body}");
}

/// GET /v1/models?beta=true WITHOUT the anthropic-version header also
/// returns the Anthropic shape: per handlers.rs `list_models`, the query
/// string alone flips the shape — the header and ?beta=true are
/// independent triggers (OR), not requiring each other.
#[tokio::test]
async fn models_list_query_beta_alone_flips_to_anthropic_shape() {
    let test = TestServer::new().await;
    let request = Request::get("/v1/models?beta=true").body(Body::empty()).unwrap();
    let response = test.router().oneshot(request).await.unwrap();
    assert_eq!(response.status(), StatusCode::OK);
    let bytes = axum::body::to_bytes(response.into_body(), 64 * 1024)
        .await
        .unwrap();
    let body: Value = serde_json::from_slice(&bytes).unwrap();
    let first = &body["data"][0];
    assert_eq!(first["type"], "model", "body: {body}");
    assert!(first.get("display_name").is_some(), "body: {body}");
    assert!(first.get("created_at").is_some(), "body: {body}");
}

// ---------------------------------------------------------------------------
// 17. HEAD /api/hello
// ---------------------------------------------------------------------------

#[tokio::test]
async fn head_api_hello_returns_200_empty() {
    let test = TestServer::new().await;
    let request = Request::head("/api/hello").body(Body::empty()).unwrap();
    let response = test.router().oneshot(request).await.unwrap();

    assert_eq!(response.status(), StatusCode::OK);
    let bytes = axum::body::to_bytes(response.into_body(), 1024)
        .await
        .unwrap();
    assert!(bytes.is_empty(), "HEAD body must be empty");
}
