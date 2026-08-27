//! Integration tests for the Responses API front door (`/v1/responses`).
//!
//! Drives the full Axum router via `tower::ServiceExt::oneshot()` (no port
//! binding, per repo conventions) with a `wiremock` upstream standing in for
//! the LLM provider, and per-test isolated SQLite databases via
//! `common::isolate_databases`.
//!
//! Endpoint coverage:
//! - POST /v1/responses (buffered) — round-trip through the mock provider,
//!   ResponseObject shape, and store persistence
//! - GET /v1/responses/{id} — retrieval + owner scoping
//! - GET /v1/responses — list with limit + cursor pagination, has_more
//! - DELETE /v1/responses/{id} — deletion + subsequent 404
//! - GET /v1/responses/{id}/input_items — input item listing with pagination
//! - Unsupported field rejection (`background: true` → 400)
//! - Virtual-key authentication enforcement (required → 401 / valid key 2xx)

use ai_gateway::config::{Config, EnforcementMode};
use ai_gateway::gateway::GatewayServer;
use ai_gateway::virtual_keys::models::{AuthenticatedKey, CreateKeyParams, KeyStatus};
use axum::body::Body;
use axum::http::{Request, StatusCode};
use axum::Extension;
use serde_json::{json, Value};
use tower::ServiceExt;
use wiremock::matchers::{method, path};
use wiremock::{Mock, MockServer, ResponseTemplate};

mod common;

struct TestServer {
    server: GatewayServer,
    _mock: MockServer,
}

impl TestServer {
    fn base_config() -> Config {
        serde_yaml::from_str(
            "server:\n  host: 127.0.0.1\n  port: 8080\nproviders:\n  - name: p\n    type: openai\n    base_url: http://localhost\n    timeout_seconds: 30\nmodel_groups:\n  - name: g\n    models:\n      - provider: p\n        model: gpt-4o\n",
        )
        .unwrap()
    }

    /// Mock provider returning a static chat completion with usage.
    async fn start_mock() -> MockServer {
        let mock = MockServer::start().await;
        Mock::given(method("POST"))
            .and(path("/v1/chat/completions"))
            .respond_with(ResponseTemplate::new(200).set_body_json(json!({
                "id": "chatcmpl-resp-test",
                "object": "chat.completion",
                "created": 1_700_000_000_i64,
                "model": "gpt-4o",
                "choices": [{
                    "index": 0,
                    "message": {"role": "assistant", "content": "mock answer"},
                    "finish_reason": "stop"
                }],
                "usage": {"prompt_tokens": 4, "completion_tokens": 3, "total_tokens": 7}
            })))
            .mount(&mock)
            .await;
        mock
    }

    /// Server with a mock provider, enforcement disabled (identity injected
    /// directly via the `AuthenticatedKey` extension layer).
    async fn new() -> Self {
        let mock = Self::start_mock().await;
        let mut config = Self::base_config();
        config.providers[0].base_url = Some(mock.uri());
        config.virtual_keys.enforcement = EnforcementMode::Disabled;
        common::isolate_databases(&mut config);
        let server = GatewayServer::new(config, None).await.unwrap();
        Self { server, _mock: mock }
    }

    /// Server with virtual-key enforcement set to `required`.
    async fn with_required_auth() -> Self {
        let mock = Self::start_mock().await;
        let mut config = Self::base_config();
        config.providers[0].base_url = Some(mock.uri());
        config.virtual_keys.enforcement = EnforcementMode::Required;
        common::isolate_databases(&mut config);
        let server = GatewayServer::new(config, None).await.unwrap();
        Self { server, _mock: mock }
    }

    fn router(&self, identity: Option<&str>) -> axum::Router {
        let router = self.server.build_router();
        match identity {
            Some(identity) => router.layer(Extension(authenticated_key(identity))),
            None => router,
        }
    }
}

fn authenticated_key(id: &str) -> AuthenticatedKey {
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
        loop_detection: None,
    }
}

async fn request(
    app: axum::Router,
    method: &str,
    uri: &str,
    body: Option<Value>,
) -> (StatusCode, Value) {
    let mut builder = Request::builder().method(method).uri(uri);
    let body = match body {
        Some(body) => {
            builder = builder.header("content-type", "application/json");
            Body::from(body.to_string())
        }
        None => Body::empty(),
    };
    let response = app.oneshot(builder.body(body).unwrap()).await.unwrap();
    let status = response.status();
    let bytes = axum::body::to_bytes(response.into_body(), 2 * 1024 * 1024)
        .await
        .unwrap();
    (status, serde_json::from_slice(&bytes).unwrap())
}

/// POST a stored response as `owner`, returning its id.
async fn create_stored(test: &TestServer, owner: &str, input: Value) -> String {
    let (status, created) = request(
        test.router(Some(owner)),
        "POST",
        "/v1/responses",
        Some(json!({
            "model": "gpt-4o",
            "input": input,
            "store": true
        })),
    )
    .await;
    assert_eq!(status, StatusCode::OK, "create failed: {created}");
    created["id"].as_str().unwrap().to_string()
}

// ---------------------------------------------------------------------------
// 1. Buffered create round-trip
// ---------------------------------------------------------------------------

#[tokio::test]
async fn buffered_create_round_trips_and_persists() {
    let test = TestServer::new().await;
    let (status, response) = request(
        test.router(Some("key-a")),
        "POST",
        "/v1/responses",
        Some(json!({
            "model": "gpt-4o",
            "input": "hello there",
            "store": true
        })),
    )
    .await;

    assert_eq!(status, StatusCode::OK, "body: {response}");
    assert_eq!(response["object"], "response");
    assert_eq!(response["status"], "completed");
    assert_eq!(response["model"], "gpt-4o");
    let id = response["id"].as_str().unwrap().to_string();

    // Output contains the synthesized assistant message with mock content.
    let output_text = response["output"][0]["content"][0]["text"]
        .as_str()
        .unwrap_or_default();
    assert!(output_text.contains("mock answer"), "output: {response}");
    assert_eq!(response["usage"]["total_tokens"], 7);

    // Persisted in the store, scoped to the owning key.
    let stored = test
        .server
        .state
        .responses_store
        .get_response("key-a", &id)
        .expect("store read")
        .expect("response persisted");
    assert_eq!(stored.id, id);
    assert_eq!(stored.owner_id, "key-a");
    assert!(stored.store);
    assert_eq!(stored.usage.as_ref().unwrap().total_tokens, 7);
}

// ---------------------------------------------------------------------------
// 2. GET /v1/responses/{id} + owner scoping
// ---------------------------------------------------------------------------

#[tokio::test]
async fn get_response_retrieves_and_is_owner_scoped() {
    let test = TestServer::new().await;
    let id = create_stored(&test, "key-a", json!("hello")).await;

    let (status, fetched) = request(
        test.router(Some("key-a")),
        "GET",
        &format!("/v1/responses/{id}"),
        None,
    )
    .await;
    assert_eq!(status, StatusCode::OK);
    assert_eq!(fetched["id"], id);
    assert_eq!(fetched["object"], "response");
    assert_eq!(fetched["status"], "completed");

    // A different key cannot see key-a's response.
    let (status, hidden) = request(
        test.router(Some("key-b")),
        "GET",
        &format!("/v1/responses/{id}"),
        None,
    )
    .await;
    assert_eq!(status, StatusCode::NOT_FOUND);
    assert_eq!(hidden["error"]["type"], "invalid_request_error");
}

// ---------------------------------------------------------------------------
// 3. GET /v1/responses (list + pagination)
// ---------------------------------------------------------------------------

#[tokio::test]
async fn list_responses_paginates_with_limit_and_cursor() {
    let test = TestServer::new().await;
    for i in 0..3 {
        create_stored(&test, "key-a", json!(format!("message {i}"))).await;
    }
    create_stored(&test, "key-b", json!("other owner")).await;

    // Small limit: has_more is true, only `limit` items returned.
    let (status, page) = request(
        test.router(Some("key-a")),
        "GET",
        "/v1/responses?limit=2&order=asc",
        None,
    )
    .await;
    assert_eq!(status, StatusCode::OK);
    assert_eq!(page["object"], "list");
    assert_eq!(page["data"].as_array().unwrap().len(), 2);
    assert_eq!(page["has_more"], true);
    let last_id = page["last_id"].as_str().unwrap().to_string();
    assert_eq!(page["data"][1]["id"].as_str().unwrap(), last_id);

    // Cursor after last_id returns the remaining item, has_more false.
    let (status, next_page) = request(
        test.router(Some("key-a")),
        "GET",
        &format!("/v1/responses?limit=2&order=asc&after={last_id}"),
        None,
    )
    .await;
    assert_eq!(status, StatusCode::OK);
    assert_eq!(next_page["data"].as_array().unwrap().len(), 1);
    assert_eq!(next_page["has_more"], false);

    // Owner scoping: key-b only sees its own single response.
    let (status, other) = request(
        test.router(Some("key-b")),
        "GET",
        "/v1/responses?limit=20&order=asc",
        None,
    )
    .await;
    assert_eq!(status, StatusCode::OK);
    assert_eq!(other["data"].as_array().unwrap().len(), 1);
    assert_eq!(other["has_more"], false);
}

// ---------------------------------------------------------------------------
// 4. DELETE /v1/responses/{id}
// ---------------------------------------------------------------------------

#[tokio::test]
async fn delete_response_then_get_returns_404() {
    let test = TestServer::new().await;
    let id = create_stored(&test, "key-a", json!("to be deleted")).await;

    let (status, deleted) = request(
        test.router(Some("key-a")),
        "DELETE",
        &format!("/v1/responses/{id}"),
        None,
    )
    .await;
    assert_eq!(status, StatusCode::OK);
    assert_eq!(deleted["id"], id);
    assert_eq!(deleted["object"], "response");
    assert_eq!(deleted["deleted"], true);

    let (status, gone) = request(
        test.router(Some("key-a")),
        "GET",
        &format!("/v1/responses/{id}"),
        None,
    )
    .await;
    assert_eq!(status, StatusCode::NOT_FOUND);
    assert_eq!(gone["error"]["type"], "invalid_request_error");
}

// ---------------------------------------------------------------------------
// 5. GET /v1/responses/{id}/input_items
// ---------------------------------------------------------------------------

#[tokio::test]
async fn input_items_list_paginates_by_cursor() {
    let test = TestServer::new().await;
    let id = create_stored(
        &test,
        "key-a",
        json!([
            {"role": "user", "content": "first question", "id": "msg_1"},
            {"role": "user", "content": "second question", "id": "msg_2"},
            {"role": "user", "content": "third question", "id": "msg_3"}
        ]),
    )
    .await;

    let (status, page) = request(
        test.router(Some("key-a")),
        "GET",
        &format!("/v1/responses/{id}/input_items?limit=2&order=asc"),
        None,
    )
    .await;
    assert_eq!(status, StatusCode::OK);
    assert_eq!(page["object"], "list");
    assert_eq!(page["data"].as_array().unwrap().len(), 2);
    assert_eq!(page["has_more"], true);
    let first = &page["data"][0];
    assert_eq!(first["role"], "user");
    assert_eq!(first["content"], "first question");
    let last_id = page["last_id"].as_str().unwrap().to_string();

    let (status, next_page) = request(
        test.router(Some("key-a")),
        "GET",
        &format!("/v1/responses/{id}/input_items?limit=2&order=asc&after={last_id}"),
        None,
    )
    .await;
    assert_eq!(status, StatusCode::OK);
    assert_eq!(next_page["data"].as_array().unwrap().len(), 1);
    assert_eq!(next_page["data"][0]["content"], "third question");
    assert_eq!(next_page["has_more"], false);

    // Unknown response id → 404.
    let (status, missing) = request(
        test.router(Some("key-a")),
        "GET",
        "/v1/responses/resp_missing/input_items",
        None,
    )
    .await;
    assert_eq!(status, StatusCode::NOT_FOUND);
    assert_eq!(missing["error"]["type"], "invalid_request_error");
}

// ---------------------------------------------------------------------------
// 6. Unsupported field → 400
// ---------------------------------------------------------------------------

#[tokio::test]
async fn background_field_is_rejected_with_400() {
    let test = TestServer::new().await;
    let (status, error) = request(
        test.router(Some("key-a")),
        "POST",
        "/v1/responses",
        Some(json!({
            "model": "gpt-4o",
            "input": "hello",
            "background": true
        })),
    )
    .await;
    assert_eq!(status, StatusCode::BAD_REQUEST);
    assert_eq!(error["error"]["type"], "invalid_request_error");
    let message = error["error"]["message"].as_str().unwrap_or_default();
    assert!(
        message.contains("background"),
        "error message should mention 'background': {error}"
    );
}

// ---------------------------------------------------------------------------
// 7. Authentication enforcement
// ---------------------------------------------------------------------------

#[tokio::test]
async fn required_auth_rejects_missing_key_and_accepts_valid_key() {
    let test = TestServer::with_required_auth().await;
    let app = test.server.build_router();

    // No key → 401.
    let (status, _body) = request(
        app.clone(),
        "POST",
        "/v1/responses",
        Some(json!({"model": "gpt-4o", "input": "hello"})),
    )
    .await;
    assert_eq!(status, StatusCode::UNAUTHORIZED);

    // Valid virtual key → reaches the mock provider and completes.
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

    let builder = Request::post("/v1/responses")
        .header("content-type", "application/json")
        .header("authorization", format!("Bearer {}", created.key));
    let response = app
        .oneshot(
            builder
                .body(Body::from(
                    json!({"model": "gpt-4o", "input": "hello", "store": true}).to_string(),
                ))
                .unwrap(),
        )
        .await
        .unwrap();
    assert_eq!(response.status(), StatusCode::OK);
    let bytes = axum::body::to_bytes(response.into_body(), 2 * 1024 * 1024)
        .await
        .unwrap();
    let body: Value = serde_json::from_slice(&bytes).unwrap();
    assert_eq!(body["object"], "response");
    assert_eq!(body["status"], "completed");

    // Stored under the virtual key's id.
    let id = body["id"].as_str().unwrap();
    assert!(test
        .server
        .state
        .responses_store
        .get_response(&created.id, id)
        .expect("store read")
        .is_some());
}
