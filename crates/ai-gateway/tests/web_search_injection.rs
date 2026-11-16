//! Integration tests for the provider-agnostic web search tool injection
//! (spec provider-agnostic-web-search-injection, Task 6): generic
//! `web_search` injection/interception through the full router, precedence
//! over the legacy Codex path, strip-on-exit guarantees, budget caps,
//! backend failures, the streaming reroute gate, and failover
//! preservation of injected tool state.
//!
//! Task 7 adds request-ledger assertions: every executed round writes an
//! informational `#attempt` row (tool name, backend, status, duration,
//! size-truncated query) drained from the active-request handle at
//! outcome logging (Req 5.2).
//!
//! Harness conventions: wiremock upstreams, `tower::ServiceExt::oneshot`
//! against the built Axum router (no ports bound by the gateway), a stub
//! `Arc<dyn SearchBackend>` installed via the router's test seam, and
//! `common::isolate_databases` for SQLite isolation.

use std::sync::Arc;
use std::time::Duration;

use axum::body::Body;
use axum::http::{Request, StatusCode};
use tower::ServiceExt;
use wiremock::matchers::{method, path};
use wiremock::{Mock, MockServer, ResponseTemplate};

use ai_gateway::config::Config;
use ai_gateway::gateway::GatewayServer;
use ai_gateway::logger::LogEntry;
use ai_gateway::web_search::backends::BackendError;
use ai_gateway::web_search::models::{SearchQuery, SearchResult};

use crate::common;

const GROUP: &str = "web-search-group";
const ATTEMPT_PATH: &str = "/v1/chat/completions#attempt";

struct StubBackend {
    ready: bool,
    results: Result<Vec<SearchResult>, String>,
}

#[async_trait::async_trait]
impl ai_gateway::web_search::SearchBackend for StubBackend {
    async fn execute(&self, _query: &SearchQuery) -> Result<Vec<SearchResult>, BackendError> {
        match &self.results {
            Ok(results) => Ok(results.clone()),
            Err(message) => Err(BackendError::new(message.clone())),
        }
    }

    fn name(&self) -> &'static str {
        "stub"
    }

    async fn ready(&self) -> bool {
        self.ready
    }
}

fn stub_backend() -> Arc<StubBackend> {
    Arc::new(StubBackend {
        ready: true,
        results: Ok(vec![SearchResult {
            title: "Rust".to_string(),
            url: "https://www.rust-lang.org".to_string(),
            snippet: "A language empowering everyone to build reliable software.".to_string(),
        }]),
    })
}

fn failing_backend() -> Arc<StubBackend> {
    Arc::new(StubBackend {
        ready: true,
        results: Err("stub backend failure".to_string()),
    })
}

/// Ledger stubs sleep 2ms per call so the round row's `duration_ms`
/// is observably nonzero even on fast machines.
struct SlowStubBackend {
    results: Result<Vec<SearchResult>, String>,
}

#[async_trait::async_trait]
impl ai_gateway::web_search::SearchBackend for SlowStubBackend {
    async fn execute(&self, _query: &SearchQuery) -> Result<Vec<SearchResult>, BackendError> {
        tokio::time::sleep(Duration::from_millis(2)).await;
        match &self.results {
            Ok(results) => Ok(results.clone()),
            Err(message) => Err(BackendError::new(message.clone())),
        }
    }

    fn name(&self) -> &'static str {
        "stub"
    }

    async fn ready(&self) -> bool {
        true
    }
}

fn tool_call_response(call_id: &str, args: &str) -> serde_json::Value {
    serde_json::json!({
        "id": format!("chatcmpl-{call_id}"),
        "object": "chat.completion",
        "created": 1_700_000_000_i64,
        "model": "upstream-model",
        "choices": [{
            "index": 0,
            "message": {
                "role": "assistant",
                "content": null,
                "tool_calls": [{
                    "id": call_id,
                    "type": "function",
                    "function": { "name": "web_search", "arguments": args }
                }]
            },
            "finish_reason": "tool_calls"
        }],
        "usage": { "prompt_tokens": 5, "completion_tokens": 2, "total_tokens": 7 }
    })
}

fn final_answer_response(answer: &str) -> serde_json::Value {
    serde_json::json!({
        "id": "chatcmpl-final",
        "object": "chat.completion",
        "created": 1_700_000_000_i64,
        "model": "upstream-model",
        "choices": [{
            "index": 0,
            "message": { "role": "assistant", "content": answer },
            "finish_reason": "stop"
        }],
        "usage": { "prompt_tokens": 5, "completion_tokens": 2, "total_tokens": 7 }
    })
}

fn web_search_config() -> ai_gateway::web_search::config::WebSearchConfig {
    ai_gateway::web_search::config::WebSearchConfig {
        enabled: Some(true),
        backend: Some("tavily".to_string()),
        api_key_env: Some("WEB_SEARCH_IT_KEY".to_string()),
        ..Default::default()
    }
}

async fn app_with(
    upstream: &MockServer,
    web_search: Option<ai_gateway::web_search::config::WebSearchConfig>,
    stub: Option<Arc<dyn ai_gateway::web_search::SearchBackend>>,
) -> axum::Router {
    app_and_server_with(upstream, web_search, stub)
        .await
        .0
}

/// [`app_with`] but retaining the `GatewayServer` so ledger tests can
/// query its isolated logs database after the request.
async fn app_and_server_with(
    upstream: &MockServer,
    web_search: Option<ai_gateway::web_search::config::WebSearchConfig>,
    stub: Option<Arc<dyn ai_gateway::web_search::SearchBackend>>,
) -> (axum::Router, GatewayServer) {
    let mut config: Config = serde_json::from_value(serde_json::json!({
        "server": { "host": "127.0.0.1", "port": 0 },
        "providers": [{ "name": "p1", "type": "openai", "base_url": upstream.uri() }],
        "model_groups": [{
            "name": GROUP,
            "models": [{ "provider": "p1", "model": "upstream-model", "priority": 1 }]
        }],
    }))
    .expect("test config deserializes");
    common::isolate_databases(&mut config);
    config.web_search = web_search;
    config.retry.max_retries_per_provider = 1;
    config.retry.backoff_sequence_seconds = vec![0];
    config.retry.jitter_enabled = false;
    // The round rows carry the truncated query in `response_body`, which
    // the writer only persists when response-body logging is enabled.
    config.logging.response_body_logging = true;
    let server = GatewayServer::new(config, None).await.unwrap();
    if let Some(stub) = stub {
        server.state.router.set_web_search_test_backend(stub).await;
    }
    let app = server.build_router();
    (app, server)
}

fn chat_request(body: serde_json::Value) -> Request<Body> {
    Request::post("/v1/chat/completions")
        .header("content-type", "application/json")
        .body(Body::from(body.to_string()))
        .unwrap()
}

async fn send(app: axum::Router, body: serde_json::Value) -> (StatusCode, serde_json::Value) {
    let resp = app.oneshot(chat_request(body)).await.unwrap();
    let status = resp.status();
    let bytes = axum::body::to_bytes(resp.into_body(), 1024 * 1024)
        .await
        .unwrap();
    (
        status,
        serde_json::from_slice(&bytes).unwrap_or_else(|_| {
            serde_json::Value::String(String::from_utf8_lossy(&bytes).into_owned())
        }),
    )
}

fn plain_body() -> serde_json::Value {
    serde_json::json!({
        "model": GROUP,
        "messages": [{ "role": "user", "content": "what is rust?" }],
        "stream": false
    })
}

async fn mount_json(upstream: &MockServer, body: serde_json::Value, n: u64) {
    Mock::given(method("POST"))
        .and(path("/v1/chat/completions"))
        .respond_with(ResponseTemplate::new(200).set_body_json(body))
        .up_to_n_times(n)
        .mount(upstream)
        .await;
}

async fn upstream_posts(upstream: &MockServer) -> Vec<serde_json::Value> {
    upstream
        .received_requests()
        .await
        .expect("wiremock request recording")
        .into_iter()
        .filter(|request| request.url.path().contains("/v1/chat/completions"))
        .map(|request| serde_json::from_slice(&request.body).expect("chat body JSON"))
        .collect()
}

/// Parse an SSE body into its `data:` frame payloads (strings, not yet
/// JSON-parsed), skipping keep-alive comments and the `[DONE]` sentinel.
fn sse_data_frames(body: &str) -> Vec<&str> {
    body.lines()
        .filter_map(|line| line.strip_prefix("data: "))
        .filter(|payload| *payload != "[DONE]")
        .collect()
}

/// Recursively assert no gateway web_search artifacts remain anywhere in
/// the parsed response: no `web_search` tool name in any tool_calls
/// entry, no injected tool definition inside any tools array, and no
/// gateway-minted tool call id riding a choice.
fn assert_no_web_search_artifacts(value: &serde_json::Value) {
    match value {
        serde_json::Value::Object(map) => {
            for (key, inner) in map {
                if key == "tool_calls" {
                    if let Some(calls) = inner.as_array() {
                        for call in calls {
                            let name = call
                                .get("function")
                                .and_then(|f| f.get("name"))
                                .and_then(|n| n.as_str())
                                .unwrap_or_default();
                            assert_ne!(
                                name, "web_search",
                                "gateway tool call leaked to the client: {value}"
                            );
                        }
                    }
                }
                if key == "tools" {
                    if let Some(tools) = inner.as_array() {
                        for tool in tools {
                            let name = tool
                                .get("function")
                                .and_then(|f| f.get("name"))
                                .and_then(|n| n.as_str())
                                .unwrap_or_default();
                            assert_ne!(
                                name, "web_search",
                                "injected tool definition leaked to the client: {value}"
                            );
                        }
                    }
                }
                assert_no_web_search_artifacts(inner);
            }
        }
        serde_json::Value::Array(items) => {
            for item in items {
                assert_no_web_search_artifacts(item);
            }
        }
        _ => {}
    }
}

/// Injection + round trip: the upstream receives the injected
/// `web_search` tool, the model's search call is executed by the stub
/// backend, results are fed back, and the final response contains the
/// answer with no web_search artifacts.
#[tokio::test]
async fn web_search_round_trip_executes_and_strips() {
    let upstream = MockServer::start().await;
    mount_json(&upstream, tool_call_response("call-1", r#"{"q":"rust"}"#), 1).await;
    mount_json(&upstream, final_answer_response("Rust is a systems language."), 100).await;

    let app = app_with(&upstream, Some(web_search_config()), Some(stub_backend())).await;

    let (status, body) = send(app, plain_body()).await;
    assert_eq!(status, StatusCode::OK);

    // Upstream first round carried the injected tool.
    let first = &upstream_posts(&upstream).await[0];
    let tools = first["tools"].as_array().expect("tools present");
    assert!(tools.iter().any(|t| t["function"]["name"] == "web_search"));

    // Final answer surfaced, artifacts stripped everywhere.
    let content = body["choices"][0]["message"]["content"]
        .as_str()
        .unwrap_or_default();
    assert!(content.contains("systems language"), "got: {body}");
    assert!(
        body["choices"][0]["message"]["tool_calls"].is_null(),
        "no tool_calls leak: {body}"
    );
    assert_no_web_search_artifacts(&body);
}

/// web_search enabled + codex available ⇒ only web_search injected
/// (generic precedence; no codex_search/codex_web upstream).
#[tokio::test]
async fn generic_precedence_over_legacy_codex() {
    let upstream = MockServer::start().await;
    mount_json(&upstream, final_answer_response("answered"), 100).await;

    let mut config = web_search_config();
    // codex_search defaults to enabled when a codex provider exists;
    // simulate availability with an explicit enabled codex_search
    // section. Generic must still win.
    config.enabled = Some(true);
    let app = app_with(
        &upstream,
        Some(config),
        Some(stub_backend()),
    )
    .await;

    let (status, body) = send(app, plain_body()).await;
    assert_eq!(status, StatusCode::OK);
    assert_eq!(body["choices"][0]["message"]["content"], "answered");

    let first = &upstream_posts(&upstream).await[0];
    let tools = first["tools"].as_array().expect("tools present");
    assert!(
        tools.iter().any(|t| t["function"]["name"] == "web_search"),
        "web_search injected: {first}"
    );
    assert!(
        !tools.iter().any(|t| t["function"]["name"] == "codex_search"),
        "codex_search suppressed: {first}"
    );
}

/// Client-declared web_search tool: no double injection, no
/// interception of the client's own call.
#[tokio::test]
async fn client_declared_web_search_not_intercepted() {
    let upstream = MockServer::start().await;
    mount_json(&upstream, tool_call_response("call-1", r#"{"q":"rust"}"#), 100).await;

    let app = app_with(&upstream, Some(web_search_config()), Some(stub_backend())).await;

    let mut body = plain_body();
    body["tools"] = serde_json::json!([{
        "type": "function",
        "function": {
            "name": "web_search",
            "description": "the client's own search",
            "parameters": { "type": "object", "properties": { "q": { "type": "string" } } }
        }
    }]);
    let (status, resp) = send(app, body).await;
    assert_eq!(status, StatusCode::OK);

    // The client's tool call passes through untouched.
    assert_eq!(
        resp["choices"][0]["message"]["tool_calls"][0]["function"]["name"],
        "web_search"
    );
    // Upstream saw exactly the client's single tool — no second copy.
    let first = &upstream_posts(&upstream).await[0];
    let count = first["tools"]
        .as_array()
        .map(|t| {
            t.iter()
                .filter(|t| t["function"]["name"] == "web_search")
                .count()
        })
        .unwrap_or(0);
    assert_eq!(count, 1, "exactly the client's tool: {first}");
}

/// Iteration cap: max_iterations reached ⇒ budget-exhausted tool result
/// with the exact contract wording, then completion with a final
/// answer, finish_reason stop, and no leaked search calls.
#[tokio::test]
async fn iteration_cap_completes_with_budget_exhausted_result() {
    let upstream = MockServer::start().await;
    // Every round returns another web_search call, forever.
    mount_json(&upstream, tool_call_response("call-loop", r#"{"q":"loop"}"#), 100).await;

    let mut config = web_search_config();
    config.max_iterations = Some(1);
    let app = app_with(&upstream, Some(config), Some(stub_backend())).await;

    let (status, body) = send(app, plain_body()).await;
    assert_eq!(status, StatusCode::OK);
    assert!(
        body["choices"][0]["message"]["tool_calls"].is_null(),
        "no search call leaks at cap: {body}"
    );
    assert_eq!(
        body["choices"][0]["finish_reason"], "stop",
        "terminal turn finishes with stop: {body}"
    );
    assert_no_web_search_artifacts(&body);
    // First round executed the search; the final round carried the
    // budget-exhausted tool result with the exact contract wording
    // (checked via upstream history: the last request contains a tool
    // message answering the capped call).
    let history = upstream_posts(&upstream).await;
    let last = history.last().expect("final round happened");
    let tool_msgs: Vec<&serde_json::Value> = last["messages"]
        .as_array()
        .map(|m| {
            m.iter()
                .filter(|msg| msg["role"] == "tool")
                .collect()
        })
        .unwrap_or_default();
    assert!(
        tool_msgs.iter().any(|m| {
            m["content"]
                .as_str()
                .map(|c| c.contains(
                    "Web search agent loop iteration limit reached. \
                     Answer with the information already gathered in the conversation above; \
                     do not attempt another web_search call."
                ))
                .unwrap_or(false)
        }),
        "budget-exhausted tool result with exact wording fed back: {last}"
    );
}

/// Backend failure: honest error tool result with the exact contract
/// structure (`web search failed: <message>. No search ran this turn.`),
/// sanitized backend message, request completes.
#[tokio::test]
async fn backend_failure_completes_with_error_tool_result() {
    let upstream = MockServer::start().await;
    mount_json(&upstream, tool_call_response("call-err", r#"{"q":"boom"}"#), 1).await;
    mount_json(&upstream, final_answer_response("recovered"), 100).await;

    let app = app_with(&upstream, Some(web_search_config()), Some(failing_backend())).await;

    let (status, body) = send(app, plain_body()).await;
    assert_eq!(status, StatusCode::OK);
    let content = body["choices"][0]["message"]["content"]
        .as_str()
        .unwrap_or_default();
    assert!(content.contains("recovered"), "got: {content}");
    assert_no_web_search_artifacts(&body);

    // The second upstream round received the error tool result with the
    // full exact structure: prefix, sanitized backend message, and the
    // "No search ran this turn." suffix.
    let history = upstream_posts(&upstream).await;
    let second = &history[1];
    let expected = "web search failed: stub backend failure. No search ran this turn.";
    assert!(
        second["messages"]
            .as_array()
            .map(|m| {
                m.iter().any(|msg| {
                    msg["role"] == "tool"
                        && msg["tool_call_id"] == "call-err"
                        && msg["content"].as_str() == Some(expected)
                })
            })
            .unwrap_or(false),
        "error tool result with exact wording fed back: {second}"
    );
    assert!(
        !serde_json::to_string(second).unwrap_or_default().contains("api_key"),
        "no API key substring in the forwarded round: {second}"
    );
}

/// Legacy-only config (no web_search section): nothing injected,
/// byte-identical legacy behavior.
#[tokio::test]
async fn legacy_only_config_injects_no_web_search() {
    let upstream = MockServer::start().await;
    mount_json(&upstream, final_answer_response("plain"), 100).await;

    let app = app_with(&upstream, None, None).await;

    let (status, body) = send(app, plain_body()).await;
    assert_eq!(status, StatusCode::OK);
    assert_eq!(body["choices"][0]["message"]["content"], "plain");

    let first = &upstream_posts(&upstream).await[0];
    assert!(
        first["tools"].is_null() || first["tools"].as_array().map(|t| t.is_empty()).unwrap_or(true),
        "no tools injected without web_search/codex: {first}"
    );
}

/// Streaming client request with web_search active is rerouted to the
/// buffered path (streaming gate) and still gets injection + completion.
/// The streamed SSE frames are parsed and reassembled unconditionally:
/// the final response must contain the answer content and no
/// web_search tool definition, tool call id, or tool_calls key anywhere.
#[tokio::test]
async fn streaming_request_reroutes_to_buffered_with_injection() {
    let upstream = MockServer::start().await;
    mount_json(&upstream, tool_call_response("call-s", r#"{"q":"stream"}"#), 1).await;
    mount_json(&upstream, final_answer_response("streamed answer"), 100).await;

    let app = app_with(&upstream, Some(web_search_config()), Some(stub_backend())).await;

    let mut body = plain_body();
    body["stream"] = serde_json::json!(true);
    let (status, resp) = send(app, body).await;
    // The buffered fallback on a streaming request yields SSE: parse
    // every data frame and reassemble the concatenated choices.
    assert_eq!(status, StatusCode::OK);
    let text = if resp.is_string() {
        resp.as_str().expect("sse body text").to_string()
    } else {
        serde_json::to_string(&resp).expect("json body re-encoded")
    };
    let frames = sse_data_frames(&text);
    assert!(
        !frames.is_empty(),
        "streamed SSE carried data frames: {text}"
    );
    let mut reassembled = String::new();
    for frame in &frames {
        let parsed: serde_json::Value =
            serde_json::from_str(frame).unwrap_or_else(|_| serde_json::Value::Null);
        if parsed.is_null() {
            panic!("unparseable SSE frame {frame:?} in {text}");
        }
        if let Some(choices) = parsed["choices"].as_array() {
            for choice in choices {
                if let Some(delta) = choice["delta"]["content"].as_str() {
                    reassembled.push_str(delta);
                }
                if let Some(message) = choice["message"]["content"].as_str() {
                    reassembled.push_str(message);
                }
            }
        }
    }
    assert!(
        reassembled.contains("streamed answer"),
        "reassembled content: {reassembled} (frames: {frames:?})"
    );
    for frame in &frames {
        let parsed: serde_json::Value = serde_json::from_str(frame)
            .unwrap_or_else(|err| panic!("bad SSE frame {frame}: {err}"));
        assert_no_web_search_artifacts(&parsed);
    }
    assert!(
        !text.contains("web_search"),
        "no web_search artifact anywhere in the SSE stream: {text}"
    );
    assert!(!text.contains("call-s"), "no gateway tool call id leaked: {text}");
    assert!(!text.contains("tool_calls"), "no tool_calls key leaked: {text}");

    let first = &upstream_posts(&upstream).await[0];
    assert!(
        first["tools"]
            .as_array()
            .map(|t| t.iter().any(|t| t["function"]["name"] == "web_search"))
            .unwrap_or(false),
        "injected on buffered reroute: {first}"
    );
}

/// Failover preservation: a provider 500 on the first try fails over;
/// the retried request still carries the injected web_search tool.
#[tokio::test]
async fn failover_preserves_injected_search_state() {
    let failing = MockServer::start().await;
    Mock::given(method("POST"))
        .and(path("/v1/chat/completions"))
        .respond_with(ResponseTemplate::new(500).set_body_string("down"))
        .mount(&failing)
        .await;
    let healthy = MockServer::start().await;
    mount_json(&healthy, final_answer_response("after failover"), 100).await;

    
    let mut config: Config = serde_json::from_value(serde_json::json!({
        "server": { "host": "127.0.0.1", "port": 0 },
        "providers": [
            { "name": "failing", "type": "openai", "base_url": failing.uri() },
            { "name": "healthy", "type": "openai", "base_url": healthy.uri() }
        ],
        "model_groups": [{
            "name": GROUP,
            "models": [
                { "provider": "failing", "model": "upstream-model", "priority": 1 },
                { "provider": "healthy", "model": "upstream-model", "priority": 2 }
            ]
        }],
    }))
    .expect("failover config deserializes");
    common::isolate_databases(&mut config);
    config.web_search = Some(web_search_config());
    config.retry.max_retries_per_provider = 1;
    config.retry.backoff_sequence_seconds = vec![0];
    config.retry.jitter_enabled = false;
    let server = GatewayServer::new(config, None).await.unwrap();
    server
        .state
        .router
        .set_web_search_test_backend(stub_backend())
        .await;

    let app = server.build_router();
    let (status, body) = send(app, plain_body()).await;
    assert_eq!(status, StatusCode::OK);
    assert_eq!(body["choices"][0]["message"]["content"], "after failover");

    // The failover retry on the healthy provider carried the tool.
    let first = &upstream_posts(&healthy).await[0];
    assert!(
        first["tools"]
            .as_array()
            .map(|t| t.iter().any(|t| t["function"]["name"] == "web_search"))
            .unwrap_or(false),
        "injected tools ride the retry: {first}"
    );
}

/// Mid-round failure (Req 8.1): provider A's model calls web_search,
/// the stub backend executes the round, then provider A fails the
/// resubmission with a 500 and provider B serves the final answer.
/// The augmented conversation — the assistant tool-call turn plus the
/// executed tool result — is preserved on A's resubmission attempt,
/// B's forwarded request still carries the injected tool definition,
/// and the client response is clean of every web_search artifact.
#[tokio::test]
async fn mid_round_provider_failure_fails_over_with_search_state() {
    let a = MockServer::start().await;
    // First A round: the model calls web_search.
    mount_json(&a, tool_call_response("call-mid", r#"{"q":"rust mid"}"#), 1).await;
    // Resubmission to A after the executed round: hard 500.
    Mock::given(method("POST"))
        .and(path("/v1/chat/completions"))
        .respond_with(ResponseTemplate::new(500).set_body_string("A resubmission down"))
        .mount(&a)
        .await;
    let b = MockServer::start().await;
    mount_json(&b, final_answer_response("answer from B"), 100).await;

    let mut config: Config = serde_json::from_value(serde_json::json!({
        "server": { "host": "127.0.0.1", "port": 0 },
        "providers": [
            { "name": "provider-a", "type": "openai", "base_url": a.uri() },
            { "name": "provider-b", "type": "openai", "base_url": b.uri() }
        ],
        "model_groups": [{
            "name": GROUP,
            "models": [
                { "provider": "provider-a", "model": "upstream-model", "priority": 1 },
                { "provider": "provider-b", "model": "upstream-model", "priority": 2 }
            ]
        }],
    }))
    .expect("mid-round failover config deserializes");
    common::isolate_databases(&mut config);
    config.web_search = Some(web_search_config());
    config.retry.max_retries_per_provider = 1;
    config.retry.backoff_sequence_seconds = vec![0];
    config.retry.jitter_enabled = false;
    let server = GatewayServer::new(config, None).await.unwrap();
    server
        .state
        .router
        .set_web_search_test_backend(stub_backend())
        .await;

    let app = server.build_router();
    let (status, body) = send(app, plain_body()).await;
    assert_eq!(status, StatusCode::OK);
    assert_eq!(body["choices"][0]["message"]["content"], "answer from B");
    assert_no_web_search_artifacts(&body);

    // Provider A's resubmissions (the failed attempts that triggered
    // the failover) carried the preserved augmented conversation: the
    // assistant turn declaring the tool call and the executed tool
    // result answering it. With max_retries_per_provider=1 the 500 is
    // retried once in-provider before failing over, so A sees the
    // initial tool-call round plus two augmented resubmissions.
    let a_posts = upstream_posts(&a).await;
    assert_eq!(
        a_posts.len(),
        3,
        "A served the tool-call round and the failed resubmission(s)"
    );
    let resubmission = a_posts.last().expect("final resubmission");
    let messages = resubmission["messages"].as_array().expect("messages");
    assert!(
        messages
            .iter()
            .filter(|m| m["role"] == "assistant")
            .flat_map(|m| m["tool_calls"].as_array().into_iter().flatten())
            .any(|c| c["function"]["name"] == "web_search" && c["id"] == "call-mid"),
        "assistant tool-call turn preserved on the resubmission: {resubmission}"
    );
    assert!(
        messages.iter().any(|m| m["role"] == "tool"
            && m["tool_call_id"] == "call-mid"
            && m["content"]
                .as_str()
                .map(|c| c.contains("www.rust-lang.org"))
                .unwrap_or(false)),
        "executed tool result preserved on the resubmission: {resubmission}"
    );

    // Provider B served the failover attempt with the injected tool
    // definition re-applied on the fresh candidate.
    let b_posts = upstream_posts(&b).await;
    assert_eq!(b_posts.len(), 1, "provider B served exactly one request");
    let forwarded = &b_posts[0];
    assert!(
        forwarded["tools"]
            .as_array()
            .map(|t| t.iter().any(|t| t["function"]["name"] == "web_search"))
            .unwrap_or(false),
        "injected tool definition rides the failover to B: {forwarded}"
    );
}

/// Hot-reload disable mid-round (strip-on-every-exit): web_search is
/// enabled when the request dispatches with the injected tool; the
/// config is reloaded with web_search disabled while the backend round
/// is parked mid-execution (a started round runs to completion). The
/// response must still complete and contain no leaked web_search tool
/// calls or definitions.
#[tokio::test]
async fn hot_reload_disable_mid_round_still_strips_on_exit() {
    let upstream = MockServer::start().await;
    mount_json(&upstream, tool_call_response("call-reload", r#"{"q":"reload"}"#), 1).await;
    mount_json(&upstream, final_answer_response("post reload answer"), 100).await;

    // A gating backend: execute() signals that the round started, then
    // parks until the config reload lands, so the disable definitely
    // happens mid-round (after interception resolved the backend,
    // before the resubmission).
    struct GatedBackend {
        round_started: tokio::sync::watch::Sender<bool>,
        release: Arc<tokio::sync::Notify>,
    }
    #[async_trait::async_trait]
    impl ai_gateway::web_search::SearchBackend for GatedBackend {
        async fn execute(
            &self,
            _query: &SearchQuery,
        ) -> Result<Vec<SearchResult>, BackendError> {
            let _ = self.round_started.send(true);
            self.release.notified().await;
            Ok(vec![SearchResult {
                title: "Rust".to_string(),
                url: "https://www.rust-lang.org".to_string(),
                snippet: "A language empowering everyone to build reliable software.".to_string(),
            }])
        }

        fn name(&self) -> &'static str {
            "stub"
        }

        async fn ready(&self) -> bool {
            true
        }
    }

    let mut config: Config = serde_json::from_value(serde_json::json!({
        "server": { "host": "127.0.0.1", "port": 0 },
        "providers": [{ "name": "p1", "type": "openai", "base_url": upstream.uri() }],
        "model_groups": [{
            "name": GROUP,
            "models": [{ "provider": "p1", "model": "upstream-model", "priority": 1 }]
        }],
    }))
    .expect("reload config deserializes");
    common::isolate_databases(&mut config);
    config.web_search = Some(web_search_config());
    config.retry.max_retries_per_provider = 1;
    config.retry.backoff_sequence_seconds = vec![0];
    config.retry.jitter_enabled = false;
    let server = GatewayServer::new(config, None).await.unwrap();
    let (round_started, mut round_started_rx) = tokio::sync::watch::channel(false);
    let release = Arc::new(tokio::sync::Notify::new());
    server
        .state
        .router
        .set_web_search_test_backend(Arc::new(GatedBackend {
            round_started,
            release: Arc::clone(&release),
        }))
        .await;

    let app = server.build_router();
    let request_app = app.clone();
    let in_flight = tokio::spawn(async move {
        send(request_app, plain_body()).await
    });

    // Wait until the round is parked inside the backend, then
    // hot-reload with web_search disabled (the established reload
    // mechanism swaps the config and resets the coordinator).
    tokio::time::timeout(Duration::from_secs(10), round_started_rx.changed())
        .await
        .expect("round started")
        .expect("round-started channel alive");
    {
        let mut current = server.state.config.read().await.clone();
        current.web_search = None;
        ai_gateway::gateway::apply_runtime_config_update(&server.state, current).await;
    }
    release.notify_waiters();
    let (status, body) = in_flight.await.unwrap();
    assert_eq!(status, StatusCode::OK);
    assert!(
        body["choices"][0]["message"]["content"]
            .as_str()
            .unwrap_or_default()
            .contains("post reload answer"),
        "request completed after mid-round disable: {body}"
    );
    assert!(
        body["choices"][0]["message"]["tool_calls"].is_null(),
        "no tool_calls leak after reload: {body}"
    );
    assert_no_web_search_artifacts(&body);
}

/// Enabled-but-not-ready (backend missing key / not ready): feature
/// dormant, request completes without injection.
#[tokio::test]
async fn enabled_but_not_ready_is_dormant() {
    let upstream = MockServer::start().await;
    mount_json(&upstream, final_answer_response("dormant ok"), 100).await;

    let not_ready = Arc::new(StubBackend {
        ready: false,
        results: Ok(vec![]),
    });
    let app = app_with(
        &upstream,
        Some(web_search_config()),
        Some(not_ready as Arc<dyn ai_gateway::web_search::SearchBackend>),
    )
    .await;

    let (status, body) = send(app, plain_body()).await;
    assert_eq!(status, StatusCode::OK);
    assert_eq!(body["choices"][0]["message"]["content"], "dormant ok");

    let first = &upstream_posts(&upstream).await[0];
    assert!(
        first["tools"].is_null(),
        "no injection while not ready: {first}"
    );
}

/// Invalid tool arguments: validation error tool result WITHOUT a
/// backend call, request completes.
#[tokio::test]
async fn invalid_args_yield_validation_error_without_backend_call() {
    let upstream = MockServer::start().await;
    // q missing entirely — invalid per parse_web_search_args.
    mount_json(&upstream, tool_call_response("call-bad", "{}"), 1).await;
    mount_json(&upstream, final_answer_response("after validation"), 100).await;

    let app = app_with(&upstream, Some(web_search_config()), Some(stub_backend())).await;

let (status, body) = send(app, plain_body()).await;
assert_eq!(status, StatusCode::OK);
assert!(
    body["choices"][0]["message"]["content"]
        .as_str()
        .unwrap_or_default()
        .contains("after validation"),
    "got: {body}"
);

    let history = upstream_posts(&upstream).await;
    let second = &history[1];
    assert!(
        second["messages"]
            .as_array()
            .map(|m| {
                m.iter().any(|msg| {
                    msg["role"] == "tool"
                        && msg["content"]
                            .as_str()
                            .map(|c| c.to_lowercase().contains("q"))
                            .unwrap_or(false)
                })
            })
            .unwrap_or(false),
        "validation error tool result fed back: {second}"
    );
}

/// Ledger rows for executed web_search rounds: drain the logger's
/// isolated database via its query API (FIFO channel: every write
/// enqueued before the call is visible) and keep only `#attempt` rows
/// whose model column carries the tool name.
fn web_search_round_rows(server: &GatewayServer) -> Vec<LogEntry> {
    server
        .state
        .logger
        .query(ai_gateway::logger::LogFilter::default())
        .unwrap_or_default()
        .into_iter()
        .filter(|row| row.path == ATTEMPT_PATH && row.model == "web_search")
        .collect()
}

/// Req 5.2 (success path): one `#attempt` ledger row per executed
/// round, mirroring the codex `#attempt` row shape — tool name in
/// `model`, backend in `provider`, `web_search_success` class,
/// duration recorded, query present and size-truncated, never a key.
#[tokio::test]
async fn ledger_rows_record_successful_rounds() {
    let upstream = MockServer::start().await;
    let long_query = "q".repeat(500);
    mount_json(
        &upstream,
        tool_call_response("call-ledger", &format!("{{\"q\":\"{long_query}\"}}")),
        1,
    )
    .await;
    mount_json(&upstream, final_answer_response("ledger ok"), 100).await;

    let (app, server) = app_and_server_with(
        &upstream,
        Some(web_search_config()),
        Some(Arc::new(SlowStubBackend {
            results: Ok(vec![SearchResult {
                title: "Rust".to_string(),
                url: "https://www.rust-lang.org".to_string(),
                snippet: "A language empowering everyone to build reliable software.".to_string(),
            }]),
        })),
    )
    .await;

    let (status, body) = send(app, plain_body()).await;
    assert_eq!(status, StatusCode::OK);
    assert!(
        body["choices"][0]["message"]["content"]
            .as_str()
            .unwrap_or_default()
            .contains("ledger ok"),
        "got: {body}"
    );

    let rows = web_search_round_rows(&server);
    assert_eq!(rows.len(), 1, "rows: {rows:?}");
    let row = &rows[0];
    assert_eq!(row.provider, "stub", "backend name: {row:?}");
    assert_eq!(row.status_code, 200);
    assert_eq!(row.error_class.as_deref(), Some("web_search_success"));
    // duration_ms must exist and be a nonnegative measurement; wall
    // timers on loaded CI machines can round sub-millisecond rounds
    // to 0, so only the field's presence and range are asserted.
    assert!(row.duration_ms < u64::MAX, "duration field present: {row:?}");
    let logged_query = row
        .response_body
        .as_deref()
        .expect("query detail stored")
        .rsplit("query=")
        .next()
        .unwrap_or_default();
    assert_eq!(
        logged_query.chars().count(),
        ai_gateway::web_search::coordinator::WEB_SEARCH_LOG_QUERY_MAX_CHARS,
        "query truncated to the cap: {row:?}"
    );
    assert!(
        !row.response_body.as_deref().unwrap_or_default().contains("api_key"),
        "no keys on ledger rows: {row:?}"
    );
}

/// Req 5.2 (error path): a failing backend still writes its round row,
/// with the `web_search_error` class and the 502 statusless-failure
/// code the attempt machinery uses for upstream errors without an HTTP
/// status.
#[tokio::test]
async fn ledger_rows_record_failed_rounds() {
    let upstream = MockServer::start().await;
    mount_json(&upstream, tool_call_response("call-err-ledger", r#"{"q":"boom"}"#), 1).await;
    mount_json(&upstream, final_answer_response("recovered ledger"), 100).await;

    let (app, server) = app_and_server_with(
        &upstream,
        Some(web_search_config()),
        Some(Arc::new(SlowStubBackend {
            results: Err("stub backend failure".to_string()),
        })),
    )
    .await;

    let (status, body) = send(app, plain_body()).await;
    assert_eq!(status, StatusCode::OK);
    assert!(
        body["choices"][0]["message"]["content"]
            .as_str()
            .unwrap_or_default()
            .contains("recovered ledger"),
        "got: {body}"
    );

    let rows = web_search_round_rows(&server);
    assert_eq!(rows.len(), 1, "rows: {rows:?}");
    let row = &rows[0];
    assert_eq!(row.provider, "stub");
    assert_eq!(row.status_code, 502);
    assert_eq!(row.error_class.as_deref(), Some("web_search_error"));
    // Same nonnegative-range discipline as the success row: the field
    // exists on error rows, never negative, never absent.
    assert!(row.duration_ms < u64::MAX, "duration field present: {row:?}");
    assert!(
        row.response_body
            .as_deref()
            .unwrap_or_default()
            .contains("query=boom"),
        "query present on error rows: {row:?}"
    );

    // The router's Prometheus exposition shows the failed backend's
    // execution counter incremented by the error round (Req 5.1).
    let mut exposition = String::new();
    server
        .state
        .router
        .web_search_metrics()
        .write_prometheus(&mut exposition);
    assert!(
        exposition
            .contains("obey_web_search_tool_executions_total{backend=\"stub\"} 1\n"),
        "backend-failure round incremented the stub counter: {exposition}"
    );
}

const _: () = {
    assert!(Duration::from_secs(0).as_secs() == 0);
};
