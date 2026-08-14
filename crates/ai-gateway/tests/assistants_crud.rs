use ai_gateway::config::{Config, EnforcementMode};
use ai_gateway::gateway::GatewayServer;
use ai_gateway::virtual_keys::models::{AuthenticatedKey, KeyStatus};
use axum::body::Body;
use axum::http::{Request, StatusCode};
use axum::Extension;
use serde_json::{json, Value};
use tower::ServiceExt;
use wiremock::matchers::{method, path};
use wiremock::{Mock, MockServer, ResponseTemplate};

struct TestServer {
    server: GatewayServer,
    _mock: Option<MockServer>,
    _temp: tempfile::TempDir,
}

impl TestServer {
    fn test_config(temp: &tempfile::TempDir) -> Config {
        let mut config: Config = serde_yaml::from_str(
"server:\n  host: 127.0.0.1\n  port: 8080\nproviders:\n  - name: p\n    type: openai\n    base_url: http://localhost\n    timeout_seconds: 30\nmodel_groups:\n  - name: g\n    models:\n      - provider: p\n        model: gpt-4o\n",
)
.unwrap();
        config.logging.database_path = temp.path().join("logs.db").to_string_lossy().into_owned();
        config.virtual_keys.database_path =
            temp.path().join("keys.db").to_string_lossy().into_owned();
        config.virtual_keys.enforcement = EnforcementMode::Disabled;
        config
    }

    async fn new() -> Self {
        let temp = tempfile::tempdir().unwrap();
        let config = Self::test_config(&temp);
        let server = GatewayServer::new(config, None).await.unwrap();
        assert_eq!(
            ai_gateway::assistants::AssistantsStore::sibling_database_path(
                &server.state.config.read().await.logging.database_path
            ),
            temp.path().join("assistants.db")
        );
        assert!(temp.path().join("assistants.db").exists());
        Self {
            server,
            _mock: None,
            _temp: temp,
        }
    }

    async fn with_mock_provider() -> Self {
        let mock = MockServer::start().await;
        Mock::given(method("POST"))
            .and(path("/v1/chat/completions"))
            .respond_with(ResponseTemplate::new(200).set_body_json(json!({
            "id": "chatcmpl-run",
            "object": "chat.completion",
            "created": 1_700_000_000_i64,
            "model": "gpt-4o",
            "choices": [{
            "index": 0,
            "message": {"role": "assistant", "content": "completed response"},
            "finish_reason": "stop"
            }],
            "usage": {"prompt_tokens": 3, "completion_tokens": 2, "total_tokens": 5}
            })))
            .mount(&mock)
            .await;
        let temp = tempfile::tempdir().unwrap();
        let mut config = TestServer::test_config(&temp);
        config.providers[0].base_url = Some(mock.uri());
        config.logging.database_path = temp.path().join("logs.db").to_string_lossy().into_owned();
        config.virtual_keys.database_path =
            temp.path().join("keys.db").to_string_lossy().into_owned();
        config.virtual_keys.enforcement = EnforcementMode::Disabled;
        let server = GatewayServer::new(config, None).await.unwrap();
        Self {
            server,
            _mock: Some(mock),
            _temp: temp,
        }
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

fn restricted_key(id: &str, allowed: &[&str]) -> AuthenticatedKey {
    AuthenticatedKey {
        model_access: Some(allowed.iter().map(|s| s.to_string()).collect()),
        ..authenticated_key(id)
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

#[tokio::test]
async fn assistants_crud_is_openai_compatible_and_identity_scoped() {
    let test = TestServer::new().await;
    let (status, assistant) = request(
        test.router(Some("key-a")),
        "POST",
        "/v1/assistants",
        Some(json!({
            "model": "gpt-4o",
            "name": "initial",
            "instructions": "help"
        })),
    )
    .await;
    assert_eq!(status, StatusCode::OK);
    assert_eq!(assistant["object"], "assistant");
    let assistant_id = assistant["id"].as_str().unwrap();

    let (status, list) = request(
        test.router(Some("key-a")),
        "GET",
        "/v1/assistants?limit=10&order=asc",
        None,
    )
    .await;
    assert_eq!(status, StatusCode::OK);
    assert_eq!(list["object"], "list");
    assert_eq!(list["data"].as_array().unwrap().len(), 1);

    let (status, hidden) = request(
        test.router(Some("key-b")),
        "GET",
        &format!("/v1/assistants/{assistant_id}"),
        None,
    )
    .await;
    assert_eq!(status, StatusCode::NOT_FOUND);
    assert_eq!(hidden["error"]["type"], "invalid_request_error");

    let (status, modified) = request(
        test.router(Some("key-a")),
        "POST",
        &format!("/v1/assistants/{assistant_id}"),
        Some(json!({"name": "modified"})),
    )
    .await;
    assert_eq!(status, StatusCode::OK);
    assert_eq!(modified["name"], "modified");

    let (status, deleted) = request(
        test.router(Some("key-a")),
        "DELETE",
        &format!("/v1/assistants/{assistant_id}"),
        None,
    )
    .await;
    assert_eq!(status, StatusCode::OK);
    assert_eq!(deleted["object"], "assistant.deleted");
    assert_eq!(deleted["deleted"], true);
}

#[tokio::test]
async fn stateful_assistants_endpoints_reject_unauthenticated_requests() {
    let test = TestServer::new().await;
    for (method, uri, body) in [
        ("POST", "/v1/assistants", Some(json!({"model": "gpt-4o"}))),
        ("POST", "/v1/threads", Some(json!({}))),
        ("GET", "/v1/files", None),
    ] {
        let (status, error) = request(test.router(None), method, uri, body).await;
        assert_eq!(status, StatusCode::UNAUTHORIZED, "{method} {uri}");
        assert_eq!(error["error"]["type"], "authentication_error");
        assert_eq!(error["error"]["code"], "authentication_required");
    }
}

#[tokio::test]
async fn threads_and_messages_support_authenticated_crud() {
    let test = TestServer::new().await;
    let (status, thread) = request(
        test.router(Some("key-a")),
        "POST",
        "/v1/threads",
        Some(json!({
            "metadata": {"topic": "test"},
            "messages": [{"role": "user", "content": "initial"}]
        })),
    )
    .await;
    assert_eq!(status, StatusCode::OK);
    assert_eq!(thread["object"], "thread");
    let thread_id = thread["id"].as_str().unwrap();

    let (status, threads) = request(test.router(Some("key-a")), "GET", "/v1/threads", None).await;
    assert_eq!(status, StatusCode::OK);
    assert_eq!(threads["data"].as_array().unwrap().len(), 1);

    let (status, messages) = request(
        test.router(Some("key-a")),
        "GET",
        &format!("/v1/threads/{thread_id}/messages?order=asc"),
        None,
    )
    .await;
    assert_eq!(status, StatusCode::OK);
    assert_eq!(messages["data"].as_array().unwrap().len(), 1);
    assert_eq!(messages["data"][0]["content"][0]["type"], "text");

    let (status, created) = request(
        test.router(Some("key-a")),
        "POST",
        &format!("/v1/threads/{thread_id}/messages"),
        Some(json!({"role": "user", "content": "second"})),
    )
    .await;
    assert_eq!(status, StatusCode::OK);
    assert_eq!(created["object"], "thread.message");
    let message_id = created["id"].as_str().unwrap();

    let (status, fetched) = request(
        test.router(Some("key-a")),
        "GET",
        &format!("/v1/threads/{thread_id}/messages/{message_id}"),
        None,
    )
    .await;
    assert_eq!(status, StatusCode::OK);
    assert_eq!(fetched["id"], message_id);

    let (status, modified) = request(
        test.router(Some("key-a")),
        "POST",
        &format!("/v1/threads/{thread_id}/messages/{message_id}"),
        Some(json!({"metadata": {"edited": "true"}})),
    )
    .await;
    assert_eq!(status, StatusCode::OK);
    assert_eq!(modified["metadata"]["edited"], "true");

    let (status, _hidden) = request(
        test.router(Some("key-b")),
        "GET",
        &format!("/v1/threads/{thread_id}"),
        None,
    )
    .await;
    assert_eq!(status, StatusCode::NOT_FOUND);

    let (status, deleted_message) = request(
        test.router(Some("key-a")),
        "DELETE",
        &format!("/v1/threads/{thread_id}/messages/{message_id}"),
        None,
    )
    .await;
    assert_eq!(status, StatusCode::OK);
    assert_eq!(deleted_message["object"], "thread.message.deleted");

    let (status, modified_thread) = request(
        test.router(Some("key-a")),
        "POST",
        &format!("/v1/threads/{thread_id}"),
        Some(json!({"metadata": {"topic": "updated"}})),
    )
    .await;
    assert_eq!(status, StatusCode::OK);
    assert_eq!(modified_thread["metadata"]["topic"], "updated");

    let (status, deleted_thread) = request(
        test.router(Some("key-a")),
        "DELETE",
        &format!("/v1/threads/{thread_id}"),
        None,
    )
    .await;
    assert_eq!(status, StatusCode::OK);
    assert_eq!(deleted_thread["object"], "thread.deleted");
}

#[tokio::test]
async fn run_executes_and_appends_completed_assistant_message() {
    let test = TestServer::with_mock_provider().await;
    let (status, assistant) = request(
        test.router(Some("key-a")),
        "POST",
        "/v1/assistants",
        Some(json!({"model": "gpt-4o", "instructions": "be concise"})),
    )
    .await;
    assert_eq!(status, StatusCode::OK);
    let (status, thread) = request(
        test.router(Some("key-a")),
        "POST",
        "/v1/threads",
        Some(json!({})),
    )
    .await;
    assert_eq!(status, StatusCode::OK);
    let thread_id = thread["id"].as_str().unwrap();
    let (status, _) = request(
        test.router(Some("key-a")),
        "POST",
        &format!("/v1/threads/{thread_id}/messages"),
        Some(json!({"role": "user", "content": "hello"})),
    )
    .await;
    assert_eq!(status, StatusCode::OK);
    let (status, run) = request(
        test.router(Some("key-a")),
        "POST",
        &format!("/v1/threads/{thread_id}/runs"),
        Some(json!({"assistant_id": assistant["id"]})),
    )
    .await;
    assert_eq!(status, StatusCode::OK);
    assert_eq!(run["status"], "completed");
    let run_id = run["id"].as_str().unwrap();
    let (status, steps) = request(
        test.router(Some("key-a")),
        "GET",
        &format!("/v1/threads/{thread_id}/runs/{run_id}/steps"),
        None,
    )
    .await;
    assert_eq!(status, StatusCode::OK);
    assert_eq!(steps["data"][0]["type"], "message_creation");
    let (status, messages) = request(
        test.router(Some("key-a")),
        "GET",
        &format!("/v1/threads/{thread_id}/messages?order=asc"),
        None,
    )
    .await;
    assert_eq!(status, StatusCode::OK);
    assert_eq!(messages["data"].as_array().unwrap().len(), 2);
    let assistant_message = messages["data"]
        .as_array()
        .unwrap()
        .iter()
        .find(|message| message["role"] == "assistant")
        .unwrap();
    assert_eq!(assistant_message["run_id"], run_id);
    assert_eq!(
        assistant_message["content"][0]["text"]["value"],
        "completed response"
    );
}

async fn multipart_request(
    app: axum::Router,
    identity: &str,
    file_name: &str,
    content: Vec<u8>,
) -> (StatusCode, Value) {
    let boundary = "obey-test-boundary";
    let mut body = format!(
"--{boundary}\r\nContent-Disposition: form-data; name=\"purpose\"\r\n\r\nassistants\r\n--{boundary}\r\nContent-Disposition: form-data; name=\"file\"; filename=\"{file_name}\"\r\nContent-Type: application/octet-stream\r\n\r\n"
)
.into_bytes();
    body.extend(content);
    body.extend(format!("\r\n--{boundary}--\r\n").into_bytes());
    let response = app
        .layer(Extension(authenticated_key(identity)))
        .oneshot(
            Request::post("/v1/files")
                .header(
                    "content-type",
                    format!("multipart/form-data; boundary={boundary}"),
                )
                .body(Body::from(body))
                .unwrap(),
        )
        .await
        .unwrap();
    let status = response.status();
    let bytes = axum::body::to_bytes(response.into_body(), 6 * 1024 * 1024)
        .await
        .unwrap();
    (status, serde_json::from_slice(&bytes).unwrap())
}

#[tokio::test]
async fn file_lifecycle_is_scoped_and_size_bounded() {
    let test = TestServer::new().await;
    let (status, file) = multipart_request(
        test.server.build_router(),
        "key-a",
        "notes.txt",
        b"private contents".to_vec(),
    )
    .await;
    assert_eq!(status, StatusCode::OK);
    let file_id = file["id"].as_str().unwrap();
    assert_eq!(file["bytes"], 16);
    let (status, list) = request(test.router(Some("key-a")), "GET", "/v1/files", None).await;
    assert_eq!(status, StatusCode::OK);
    assert_eq!(list["data"].as_array().unwrap().len(), 1);
    let (status, hidden) = request(
        test.router(Some("key-b")),
        "GET",
        &format!("/v1/files/{file_id}"),
        None,
    )
    .await;
    assert_eq!(status, StatusCode::NOT_FOUND);
    assert_eq!(hidden["error"]["type"], "invalid_request_error");
    let response = test
        .router(Some("key-a"))
        .oneshot(
            Request::get(format!("/v1/files/{file_id}/content"))
                .body(Body::empty())
                .unwrap(),
        )
        .await
        .unwrap();
    assert_eq!(response.status(), StatusCode::OK);
    let content = axum::body::to_bytes(response.into_body(), 1024)
        .await
        .unwrap();
    assert_eq!(&content[..], b"private contents");
    let (status, deleted) = request(
        test.router(Some("key-a")),
        "DELETE",
        &format!("/v1/files/{file_id}"),
        None,
    )
    .await;
    assert_eq!(status, StatusCode::OK);
    assert_eq!(deleted["deleted"], true);
    let (status, error) = multipart_request(
        test.server.build_router(),
        "key-a",
        "large.bin",
        vec![b'x'; ai_gateway::assistants::MAX_FILE_BYTES + 1],
    )
    .await;
    assert_eq!(status, StatusCode::PAYLOAD_TOO_LARGE);
    assert_eq!(error["error"]["type"], "invalid_request_error");
}

#[tokio::test]
async fn fine_tuning_proxies_to_openai_compatible_provider() {
    let mock = MockServer::start().await;
    Mock::given(method("POST"))
        .and(path("/v1/fine_tuning/jobs"))
        .respond_with(ResponseTemplate::new(200).set_body_json(json!({
            "id": "ftjob-proxy-1",
            "object": "fine_tuning.job",
            "status": "queued",
            "model": "gpt-4o"
        })))
        .mount(&mock)
        .await;
    Mock::given(method("GET"))
        .and(path("/v1/fine_tuning/jobs/ftjob-proxy-1"))
        .respond_with(ResponseTemplate::new(200).set_body_json(json!({
            "id": "ftjob-proxy-1",
            "object": "fine_tuning.job",
            "status": "succeeded"
        })))
        .mount(&mock)
        .await;

    let temp = tempfile::tempdir().unwrap();
    let mut config = TestServer::test_config(&temp);
    config.providers[0].base_url = Some(mock.uri());
    config.logging.database_path = temp.path().join("logs.db").to_string_lossy().into_owned();
    config.virtual_keys.database_path =
        temp.path().join("keys.db").to_string_lossy().into_owned();
    let server = GatewayServer::new(config, None).await.unwrap();

    let (status, created) = request(
        server.build_router(),
        "POST",
        "/v1/fine_tuning/jobs",
        Some(json!({"model": "gpt-4o", "training_file": "file-1"})),
    )
    .await;
    assert_eq!(status, StatusCode::OK);
    assert_eq!(created["object"], "fine_tuning.job");
    assert_eq!(created["id"], "ftjob-proxy-1");

    let (status, fetched) = request(
        server.build_router(),
        "GET",
        "/v1/fine_tuning/jobs/ftjob-proxy-1",
        None,
    )
    .await;
    assert_eq!(status, StatusCode::OK);
    assert_eq!(fetched["status"], "succeeded");
}

#[tokio::test]
async fn fine_tuning_returns_structured_unsupported_feature() {
    let temp = tempfile::tempdir().unwrap();
    let mut config = TestServer::test_config(&temp);
    // Bedrock providers have no OpenAI-compatible HTTP API for fine-tuning.
    config.providers[0].provider_type = "bedrock".to_string();
    config.logging.database_path = temp.path().join("logs.db").to_string_lossy().into_owned();
    config.virtual_keys.database_path =
        temp.path().join("keys.db").to_string_lossy().into_owned();
    let server = GatewayServer::new(config, None).await.unwrap();

    let (status, error) = request(
        server.build_router(),
        "POST",
        "/v1/fine_tuning/jobs",
        Some(json!({"model": "gpt-4o", "training_file": "file_missing"})),
    )
    .await;
    assert_eq!(status, StatusCode::NOT_IMPLEMENTED);
    assert_eq!(error["error"]["type"], "unsupported_feature");
    assert_eq!(error["error"]["code"], "unsupported_feature");
}

#[tokio::test]
async fn run_denies_model_not_permitted_for_virtual_key() {
    let test = TestServer::with_mock_provider().await;
    let router = test
        .server
        .build_router()
        .layer(Extension(restricted_key("key-a", &["other-model"])));

    let (status, assistant) = request(
        router.clone(),
        "POST",
        "/v1/assistants",
        Some(json!({"model": "gpt-4o", "name": "helper"})),
    )
    .await;
    assert_eq!(status, StatusCode::OK);
    let assistant_id = assistant["id"].as_str().unwrap();

    let (status, thread) = request(router.clone(), "POST", "/v1/threads", Some(json!({}))).await;
    assert_eq!(status, StatusCode::OK);
    let thread_id = thread["id"].as_str().unwrap();

    let (status, _) = request(
        router.clone(),
        "POST",
        &format!("/v1/threads/{thread_id}/messages"),
        Some(json!({"role": "user", "content": "hello"})),
    )
    .await;
    assert_eq!(status, StatusCode::OK);

    // The request body has no `model`, so the effective model comes from the
    // assistant — authorization must still reject it.
    let (status, error) = request(
        router.clone(),
        "POST",
        &format!("/v1/threads/{thread_id}/runs"),
        Some(json!({"assistant_id": assistant_id})),
    )
    .await;
    assert_eq!(status, StatusCode::FORBIDDEN);
    assert_eq!(error["error"]["code"], "model_access_denied");
    assert_eq!(error["error"]["model"], "gpt-4o");

    // The stranded run must be persisted as failed, not in_progress.
    let (status, runs) = request(
        router.clone(),
        "GET",
        &format!("/v1/threads/{thread_id}/runs"),
        None,
    )
    .await;
    assert_eq!(status, StatusCode::OK);
    let runs = runs["data"].as_array().unwrap();
    assert_eq!(runs.len(), 1);
    assert_eq!(runs[0]["status"], "failed");
}

#[tokio::test]
async fn run_cancellation_aborts_inflight_execution() {
    let mock = MockServer::start().await;
    Mock::given(method("POST"))
        .and(path("/v1/chat/completions"))
        .respond_with(
            ResponseTemplate::new(200)
                .set_delay(std::time::Duration::from_millis(1500))
                .set_body_json(json!({
                    "id": "chatcmpl-slow",
                    "object": "chat.completion",
                    "created": 1_700_000_000_i64,
                    "model": "gpt-4o",
                    "choices": [{
                        "index": 0,
                        "message": {"role": "assistant", "content": "slow"},
                        "finish_reason": "stop"
                    }],
                    "usage": {"prompt_tokens": 1, "completion_tokens": 1, "total_tokens": 2}
                })),
        )
        .mount(&mock)
        .await;

    let temp = tempfile::tempdir().unwrap();
    let mut config = TestServer::test_config(&temp);
    config.providers[0].base_url = Some(mock.uri());
    config.logging.database_path = temp.path().join("logs.db").to_string_lossy().into_owned();
    config.virtual_keys.database_path =
        temp.path().join("keys.db").to_string_lossy().into_owned();
    let server = GatewayServer::new(config, None).await.unwrap();
    let router = server.build_router().layer(Extension(authenticated_key("key-a")));

    let (status, assistant) = request(
        router.clone(),
        "POST",
        "/v1/assistants",
        Some(json!({"model": "gpt-4o"})),
    )
    .await;
    assert_eq!(status, StatusCode::OK);
let assistant_id = assistant["id"].as_str().unwrap().to_string();
let (status, thread) = request(router.clone(), "POST", "/v1/threads", Some(json!({}))).await;
assert_eq!(status, StatusCode::OK);
let thread_id = thread["id"].as_str().unwrap();
let (status, _) = request(
router.clone(),
"POST",
&format!("/v1/threads/{thread_id}/messages"),
Some(json!({"role": "user", "content": "hello"})),
)
.await;
assert_eq!(status, StatusCode::OK);

let run_router = router.clone();
let run_thread = thread_id.to_string();
let run_assistant_id = assistant_id.clone();
let run_handle = tokio::spawn(async move {
let response = run_router
.oneshot(
Request::post(format!("/v1/threads/{run_thread}/runs"))
.header("content-type", "application/json")
.body(Body::from(
json!({"assistant_id": run_assistant_id}).to_string(),
))
.unwrap(),
)
.await
.unwrap();
let status = response.status();
let bytes = axum::body::to_bytes(response.into_body(), 1024 * 1024)
.await
.unwrap();
(status, serde_json::from_slice::<Value>(&bytes).unwrap())
});

    // Wait for the run to register, then cancel it mid-flight.
    tokio::time::sleep(std::time::Duration::from_millis(300)).await;
    let (status, runs) = request(
        router.clone(),
        "GET",
        &format!("/v1/threads/{thread_id}/runs"),
        None,
    )
    .await;
    assert_eq!(status, StatusCode::OK);
    let run_id = runs["data"][0]["id"].as_str().unwrap().to_string();

    let (status, cancelled) = request(
        router.clone(),
        "POST",
        &format!("/v1/threads/{thread_id}/runs/{run_id}/cancel"),
        None,
    )
    .await;
    assert_eq!(status, StatusCode::OK);
    assert_eq!(cancelled["status"], "cancelled");

    let (run_status, run_body) = run_handle.await.unwrap();
    assert_eq!(run_status, StatusCode::OK);
    assert_eq!(run_body["status"], "cancelled");
}
