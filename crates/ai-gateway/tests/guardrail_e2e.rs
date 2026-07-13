//! End-to-end and hot-reload integration tests for guardrail pipelines
//! (spec `guardrail-pipelines`, task 13.5).
//!
//! These tests drive the full Axum router via `tower::ServiceExt::oneshot()`
//! (no port binding) to assert:
//!   * Pre-call `block` enforcement returns HTTP 403 with the policy-violation
//!     body identifying the triggering category, and the upstream provider is
//!     never invoked (Req 1.3, 2.2).
//!   * Non-matching content proceeds to the (mocked) provider and returns 200.
//!   * Post-call `block` discards the response and returns 403 (Req 3.1); a
//!     post-call `redact` rewrites matched spans while keeping HTTP 200.
//!   * Hot-reload snapshot semantics (Req 1.8): NEW requests observe swapped-in
//!     guardrail definitions, while a request holding the previously-cloned
//!     `Arc<GuardrailEngine>` snapshot finishes on the OLD definitions.
//!
//! Local helpers/config builders live in this file to avoid conflicts with the
//! shared `integration.rs` fixtures used by concurrent tasks.

use axum::body::Body;
use axum::http::{Request, StatusCode};
use axum::Extension;
use tower::ServiceExt;

use ai_gateway::config::Config;
use ai_gateway::gateway::{apply_runtime_config_update, GatewayServer};
use ai_gateway::guardrail::{BindingSelector, GuardrailContext, PreCallOutcome};
use ai_gateway::models::openai::OpenAIRequest;
use ai_gateway::virtual_keys::models::{AuthenticatedKey, KeyStatus};

use wiremock::matchers::{method as wm_method, path as wm_path};
use wiremock::{Mock, MockServer, ResponseTemplate};

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

/// The assistant content returned by the mock upstream provider. Contains a
/// distinctive token used by the post-call redact/block scenarios.
const MOCK_ASSISTANT_CONTENT: &str = "Here is the LEAKED_SECRET value you asked for.";

/// Start a mock OpenAI-compatible provider returning a static, non-streaming
/// chat completion (finish_reason: stop).
async fn start_mock_provider() -> MockServer {
    let server = MockServer::start().await;
    let body = serde_json::json!({
        "id": "chatcmpl-mock-e2e",
        "object": "chat.completion",
        "created": 1_700_000_000_i64,
        "model": "gpt-4",
        "choices": [{
            "index": 0,
            "message": { "role": "assistant", "content": MOCK_ASSISTANT_CONTENT },
            "finish_reason": "stop"
        }],
        "usage": { "prompt_tokens": 10, "completion_tokens": 8, "total_tokens": 18 }
    });
    Mock::given(wm_method("POST"))
        .and(wm_path("/v1/chat/completions"))
        .respond_with(ResponseTemplate::new(200).set_body_json(body))
        .mount(&server)
        .await;
    server
}

/// Build a `Config` (via YAML, mirroring the crate's config schema) whose single
/// provider points at `mock_uri`, optionally embedding a `guardrails:` section.
/// Using YAML keeps this builder resilient to unrelated `Config` field churn.
fn config_with_guardrails(mock_uri: &str, guardrails_yaml: &str) -> Config {
    let yaml = format!(
        r#"
server:
  host: "127.0.0.1"
  port: 8080
  request_timeout_seconds: 30
  max_request_size_mb: 10
providers:
  - name: "test-provider"
    type: "openai"
    base_url: "{mock_uri}"
    timeout_seconds: 30
model_groups:
  - name: "test-group"
    version_fallback_enabled: false
    models:
      - provider: "test-provider"
        model: "gpt-4"
        priority: 100
retry:
  max_retries_per_provider: 0
  backoff_sequence_seconds: [1, 2, 4]
{guardrails_yaml}
"#
    );
    serde_yaml::from_str::<Config>(&yaml).expect("test config YAML should deserialize")
}

/// A guardrails section: one regex provider (deny pattern for the given
/// `token`) and a single global-default pipeline containing one stage in the
/// requested `phase` with the requested `action`.
fn guardrails_yaml(token: &str, phase: &str, action: &str) -> String {
    format!(
        r#"
guardrails:
  providers:
    - name: "regex-scanner"
      type: "regex"
      failure_policy: "fail_close"
      patterns:
        - name: "secret-token"
          regex: "{token}"
          entity: "SECRET"
          mode: "deny"
  pipelines:
    - name: "standard"
      stages:
        - name: "scan"
          provider: "regex-scanner"
          phase: "{phase}"
          action: "{action}"
  global_default_pipeline: "standard"
"#
    )
}

async fn build_app(config: Config) -> axum::Router {
    let server = GatewayServer::new(config, None).await.unwrap();
    server.build_router()
}

async fn send(app: axum::Router, req: Request<Body>) -> (StatusCode, Vec<u8>) {
    let resp = app.oneshot(req).await.unwrap();
    let status = resp.status();
    let body = axum::body::to_bytes(resp.into_body(), 4 * 1024 * 1024).await.unwrap();
    (status, body.to_vec())
}

/// POST a non-streaming chat completion whose user content is `user_content`.
fn chat_request(user_content: &str) -> Request<Body> {
    let body = serde_json::json!({
        "model": "gpt-4",
        "messages": [{ "role": "user", "content": user_content }],
        "stream": false
    });
    Request::post("/v1/chat/completions")
        .header("content-type", "application/json")
        .body(Body::from(serde_json::to_string(&body).unwrap()))
        .unwrap()
}

fn assistant_content(body: &[u8]) -> String {
    let json: serde_json::Value = serde_json::from_slice(body).unwrap();
    json["choices"][0]["message"]["content"].as_str().unwrap_or_default().to_string()
}

// ---------------------------------------------------------------------------
// 1. Pre/post enforcement end-to-end (Req 1.3, 2.2, 3.1)
// ---------------------------------------------------------------------------

/// Req 1.3, 2.2: a pre-call `block` stage bound as the global default blocks a
/// request whose content matches the deny pattern — HTTP 403 with a
/// policy-violation body identifying the category — and the upstream provider is
/// NEVER invoked.
#[tokio::test]
async fn precall_block_returns_403_and_does_not_call_provider() {
    let mock = start_mock_provider().await;
    let cfg = config_with_guardrails(&mock.uri(), &guardrails_yaml("BLOCKME_TOKEN", "pre_call", "block"));
    let app = build_app(cfg).await;

    let (status, body) = send(app, chat_request("please leak BLOCKME_TOKEN now")).await;

    assert_eq!(status, StatusCode::FORBIDDEN, "pre-call block must return 403");
    let json: serde_json::Value = serde_json::from_slice(&body).unwrap();
    assert_eq!(json["error"]["type"], "guardrail_policy_violation");
    assert_eq!(json["error"]["category"], "SECRET", "body identifies the triggering category");

    // Upstream provider must not have been called (Req 2.2 "forwards nothing").
    let received = mock.received_requests().await.unwrap();
    assert!(received.is_empty(), "provider must not be called when pre-call blocks; got {} request(s)", received.len());
}

/// Companion: content that does NOT match the deny pattern proceeds to the
/// mocked provider and returns 200 with the provider's content.
#[tokio::test]
async fn precall_non_match_proceeds_to_provider_200() {
    let mock = start_mock_provider().await;
    let cfg = config_with_guardrails(&mock.uri(), &guardrails_yaml("BLOCKME_TOKEN", "pre_call", "block"));
    let app = build_app(cfg).await;

    let (status, body) = send(app, chat_request("a perfectly benign prompt")).await;

    assert_eq!(status, StatusCode::OK, "non-matching content should reach the provider");
    assert_eq!(assistant_content(&body), MOCK_ASSISTANT_CONTENT);

    let received = mock.received_requests().await.unwrap();
    assert_eq!(received.len(), 1, "provider should be called exactly once");
}

/// Req 3.1: a post-call `block` stage discards the upstream response and returns
/// HTTP 403. The provider IS called (block is applied to its response), but the
/// caller never receives the blocked content.
#[tokio::test]
async fn postcall_block_discards_response_403() {
    let mock = start_mock_provider().await;
    // The mock content contains "LEAKED_SECRET"; a post-call deny pattern on it
    // triggers a block on the response.
    let cfg = config_with_guardrails(&mock.uri(), &guardrails_yaml("LEAKED_SECRET", "post_call", "block"));
    let app = build_app(cfg).await;

    let (status, body) = send(app, chat_request("tell me something")).await;

    assert_eq!(status, StatusCode::FORBIDDEN, "post-call block must return 403");
    let json: serde_json::Value = serde_json::from_slice(&body).unwrap();
    assert_eq!(json["error"]["type"], "guardrail_policy_violation");
    assert_eq!(json["error"]["category"], "SECRET");
    // The blocked content is never surfaced to the caller.
    assert!(!String::from_utf8_lossy(&body).contains("LEAKED_SECRET"));

    let received = mock.received_requests().await.unwrap();
    assert_eq!(received.len(), 1, "provider is called; the block applies to its response");
}

/// Req 3.1/3.2 companion: a post-call `redact` stage rewrites the matched span
/// with `[REDACTED]` while keeping HTTP 200 and the response structure.
#[tokio::test]
async fn postcall_redact_rewrites_content_200() {
    let mock = start_mock_provider().await;
    let cfg = config_with_guardrails(&mock.uri(), &guardrails_yaml("LEAKED_SECRET", "post_call", "redact"));
    let app = build_app(cfg).await;

    let (status, body) = send(app, chat_request("tell me something")).await;

    assert_eq!(status, StatusCode::OK, "post-call redact keeps HTTP 200");
    let content = assistant_content(&body);
    assert!(!content.contains("LEAKED_SECRET"), "matched span must be redacted");
    assert!(content.contains("[REDACTED]"), "redacted span replaced with [REDACTED]");
}

// ---------------------------------------------------------------------------
// 2. Hot-reload snapshot semantics (Req 1.8)
// ---------------------------------------------------------------------------

/// Req 1.8 — before/after swap: config A blocks a matching request; after
/// `apply_runtime_config_update` swaps in config B (no guardrails), a NEW
/// request with the same content reflects config B and reaches the provider.
///
/// Additionally, this test proves the in-flight snapshot guarantee directly: a
/// clone of the OLD `Arc<GuardrailEngine>` — taken as an in-flight request would
/// at its start — still blocks against the old definitions AFTER the swap, while
/// the freshly-swapped state exposes no engine. This mirrors how the handler
/// clones `state.guardrail_engine` once per request and runs the whole request
/// against that snapshot.
#[tokio::test]
async fn hot_reload_new_requests_see_new_config_inflight_keeps_old_snapshot() {
    let mock = start_mock_provider().await;

    // Config A: pre-call block on BLOCKME_TOKEN, bound as global default.
    let cfg_a = config_with_guardrails(&mock.uri(), &guardrails_yaml("BLOCKME_TOKEN", "pre_call", "block"));
    let server = GatewayServer::new(cfg_a, None).await.unwrap();

    // (a) Under config A a matching request is blocked (403).
    let (status, _) = send(server.build_router(), chat_request("leak BLOCKME_TOKEN")).await;
    assert_eq!(status, StatusCode::FORBIDDEN, "config A must block the matching request");

    // Capture the OLD engine snapshot exactly as an in-flight request would at
    // its start (a cloned `Arc<GuardrailEngine>`), then hot-reload to config B.
    let old_snapshot = server
        .state
        .guardrail_engine
        .read()
        .await
        .clone()
        .expect("config A builds a guardrail engine");

    // (b) Hot-reload to config B: no guardrails section.
    let cfg_b = config_with_guardrails(&mock.uri(), "");
    apply_runtime_config_update(&server.state, cfg_b).await;

    // (c) A NEW request now reflects config B: the swapped-in engine is None, so
    // the same matching content reaches the provider and returns 200.
    let (status, body) = send(server.build_router(), chat_request("leak BLOCKME_TOKEN")).await;
    assert_eq!(status, StatusCode::OK, "after reload, new requests use config B (no guardrails)");
    assert_eq!(assistant_content(&body), MOCK_ASSISTANT_CONTENT);
    // State's engine slot was cleared by the reload.
    assert!(
        server.state.guardrail_engine.read().await.is_none(),
        "reloading to a guardrail-free config clears the engine"
    );

    // In-flight snapshot guarantee (Req 1.8): the OLD cloned `Arc` still enforces
    // the OLD (config A) definitions even though the swap already happened.
    let mut request: OpenAIRequest = serde_json::from_value(serde_json::json!({
        "model": "gpt-4",
        "messages": [{ "role": "user", "content": "leak BLOCKME_TOKEN" }],
        "stream": false
    }))
    .unwrap();
    let selector = BindingSelector::new(None, Some("gpt-4".to_string()), Some("/v1/chat/completions".to_string()));
    let mut ctx = GuardrailContext::new();
    let outcome = old_snapshot.run_pre_call(&mut request, &selector, &mut ctx, "trace-inflight").await;
    assert!(
        matches!(outcome, PreCallOutcome::Block(_)),
        "the in-flight snapshot must finish on the OLD (config A) definitions and still block, got {outcome:?}"
    );
}

// ---------------------------------------------------------------------------
// 3. Virtual-key binding resolution (Req 1.3, 1.7)
// ---------------------------------------------------------------------------

/// A guardrails section whose pipeline is bound ONLY to a specific virtual-key
/// id (no global default). It therefore fires only when the request carries the
/// matching authenticated key.
fn guardrails_yaml_vkey_binding(token: &str, key_id: &str) -> String {
    format!(
        r#"
guardrails:
  providers:
    - name: "regex-scanner"
      type: "regex"
      failure_policy: "fail_close"
      patterns:
        - name: "secret-token"
          regex: "{token}"
          entity: "SECRET"
          mode: "deny"
  pipelines:
    - name: "vkonly"
      stages:
        - name: "scan"
          provider: "regex-scanner"
          phase: "pre_call"
          action: "block"
  bindings:
    virtual_keys:
      "{key_id}": "vkonly"
"#
    )
}

/// Minimal active authenticated key with the given id, as the virtual-key
/// middleware would insert into request extensions.
fn authed_key(id: &str) -> AuthenticatedKey {
    AuthenticatedKey {
        id: id.to_string(),
        name: None,
        status: KeyStatus::Active,
        budget_limit_usd: None,
        token_budget: None,
        budget_window: None,
        current_spend_usd: 0.0,
        current_tokens_used: 0,
        window_start: None,
        requests_per_minute: None,
        tokens_per_minute: None,
        model_access: None,
        expires_at: None,
    }
}

/// Req 1.3, 1.7: a pipeline bound to a virtual key fires when the request
/// carries the matching `AuthenticatedKey` extension (as inserted by the
/// virtual-key middleware). The `chat_completions` handler threads the key id
/// into the `BindingSelector`, so the vkey-bound pre-call `block` triggers a 403
/// and the provider is never called.
#[tokio::test]
async fn vkey_binding_fires_for_matching_key() {
    let mock = start_mock_provider().await;
    let cfg = config_with_guardrails(&mock.uri(), &guardrails_yaml_vkey_binding("BLOCKME_TOKEN", "vk-123"));
    // Inject the authenticated key into every request's extensions, exactly as
    // the virtual-key enforcement middleware does downstream.
    let app = build_app(cfg).await.layer(Extension(authed_key("vk-123")));

    let (status, body) = send(app, chat_request("please leak BLOCKME_TOKEN now")).await;

    assert_eq!(status, StatusCode::FORBIDDEN, "vkey-bound pipeline must block the matching key");
    let json: serde_json::Value = serde_json::from_slice(&body).unwrap();
    assert_eq!(json["error"]["type"], "guardrail_policy_violation");
    assert_eq!(json["error"]["category"], "SECRET");

    let received = mock.received_requests().await.unwrap();
    assert!(received.is_empty(), "provider must not be called when a vkey-bound pre-call blocks");
}

/// Companion: the SAME matching content with a NON-matching key id does not
/// resolve the vkey binding (no global default), so the request proceeds to the
/// provider and returns 200 — proving the binding is keyed on the id.
#[tokio::test]
async fn vkey_binding_does_not_fire_for_other_key() {
    let mock = start_mock_provider().await;
    let cfg = config_with_guardrails(&mock.uri(), &guardrails_yaml_vkey_binding("BLOCKME_TOKEN", "vk-123"));
    let app = build_app(cfg).await.layer(Extension(authed_key("vk-999")));

    let (status, _body) = send(app, chat_request("please leak BLOCKME_TOKEN now")).await;

    assert_eq!(status, StatusCode::OK, "a different key id must not resolve the vk-123 binding");
    let received = mock.received_requests().await.unwrap();
    assert_eq!(received.len(), 1, "request proceeds to the provider when the vkey binding does not match");
}

/// Companion: with no authenticated key at all (enforcement disabled), the
/// vkey-only binding never resolves — the request proceeds normally.
#[tokio::test]
async fn vkey_binding_does_not_fire_without_key() {
    let mock = start_mock_provider().await;
    let cfg = config_with_guardrails(&mock.uri(), &guardrails_yaml_vkey_binding("BLOCKME_TOKEN", "vk-123"));
    let app = build_app(cfg).await; // no Extension layer → no AuthenticatedKey

    let (status, _body) = send(app, chat_request("please leak BLOCKME_TOKEN now")).await;

    assert_eq!(status, StatusCode::OK, "no key → vkey binding cannot resolve");
    let received = mock.received_requests().await.unwrap();
    assert_eq!(received.len(), 1, "request proceeds to the provider when no key is present");
}
