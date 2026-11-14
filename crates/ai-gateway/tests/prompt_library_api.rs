//! `/v1/chat/completions` prompt-field consumption and logging
//! integration tests (spec: prompt-library-versioning, Task 10, Req 4).
//!
//! Drives the full gateway HTTP surface via `tower::ServiceExt::oneshot()`
//! with a wiremock OpenAI-compatible provider - no ports, no servers.
//! Databases are isolated per test via `common::isolate_databases`, and the
//! prompt store is redirected to a sibling of the redirected logs.db
//! (same placement rule as tests/prompt_library_admin.rs), so parallel
//! tests never share `./prompts.db`.
//!
//! Covers (Req 4.1-4.6):
//! - prompt-by-name / name@label / uuid resolution and routing; label
//!   omitted -> production (4.1)
//! - pinned-version stability across a production label move (Req 3.5)
//! - merge order: template messages first, client messages appended (4.2)
//! - variable rendering incl. missing variable -> empty string (4.1)
//! - version + label-bearing id -> 400 naming the field, no provider
//!   contact (4.6)
//! - unknown name/label -> 404 invalid_request_error naming prompt and
//!   label, no provider contact (4.3)
//! - malformed prompt field (non-object, missing id, variables
//!   non-object) -> 400 naming the field (4.3)
//! - no-prompt request routes identically to before (4.1 zero-overhead
//!   skip)
//! - outcome log row carries prompt_id/prompt_version/prompt_label and
//!   no template content or variable values (4.4)
//! - virtual-key model restriction evaluated against the merged request
//!   exactly as for a plain request (4.5)

use crate::common;

use std::time::Duration;

use ai_gateway::config::{Config, EnforcementMode};
use ai_gateway::gateway::GatewayServer;
use ai_gateway::logger::LogEntry;
use ai_gateway::virtual_keys::models::CreateKeyParams;
use axum::body::Body;
use axum::http::{Request, StatusCode};
use serde_json::{json, Value};
use tower::ServiceExt;
use wiremock::matchers::{method, path};
use wiremock::{Mock, MockServer, ResponseTemplate};

const ADMIN_BASE: &str = "/admin/prompt-library";
const MODEL_GROUP: &str = "prompt-group";

/// Two-message JSON template (system + user) with distinct content markers.
const TEMPLATE: &str = r#"[{"role":"system","content":"TEMPLATE SYSTEM VOICE"},{"role":"user","content":"TEMPLATE TASK"}]"#;

/// Build a minimal valid Config with one provider behind one model group,
/// mirroring the harness in tests/request_outcome_logging.rs.
fn prompt_api_config(mock_uri: &str) -> Config {
    let mut config: Config = serde_json::from_value(json!({
        "server": { "host": "127.0.0.1", "port": 0 },
        "providers": [{ "name": "prompt-provider", "type": "openai", "base_url": mock_uri }],
        "model_groups": [{ "name": MODEL_GROUP, "models": [{ "provider": "prompt-provider", "model": "gpt-4", "priority": 1 }] }]
    }))
    .expect("test configuration should deserialize");
    config.retry.max_retries_per_provider = 0;
    config.retry.backoff_sequence_seconds = vec![0];
    config.retry.jitter_enabled = false;
    // Identical merged bodies across resolution-form calls must still hit
    // the provider (hit-count assertions), never the exact cache.
    config.exact_cache.enabled = false;
    common::isolate_databases(&mut config);
    config
}

/// Build the app with the prompt store placed as a sibling of the
/// redirected logs.db so each test gets its own prompts.db.
async fn build_app(mut config: Config) -> (axum::Router, GatewayServer) {
    config.prompt_library.database_path = std::path::Path::new(&config.logging.database_path)
        .parent()
        .expect("isolated logs.db parent")
        .join("prompts.db")
        .to_string_lossy()
        .into_owned();
    let server = GatewayServer::new(config, None).await.unwrap();
    let router = server.build_router();
    (router, server)
}

async fn start_mock_provider() -> MockServer {
    let server = MockServer::start().await;
    let body = json!({
        "id": "chatcmpl-prompt-mock",
        "object": "chat.completion",
        "created": 1_700_000_000_i64,
        "model": "gpt-4",
        "choices": [{
            "index": 0,
            "message": { "role": "assistant", "content": "Hello from mock provider!" },
            "finish_reason": "stop"
        }],
        "usage": { "prompt_tokens": 10, "completion_tokens": 8, "total_tokens": 18 }
    });
    Mock::given(method("POST"))
        .and(path("/v1/chat/completions"))
        .respond_with(ResponseTemplate::new(200).set_body_json(body))
        .mount(&server)
        .await;
    server
}

async fn send(app: axum::Router, request: Request<Body>) -> (StatusCode, Value) {
    let response = app.oneshot(request).await.unwrap();
    let status = response.status();
    let bytes = axum::body::to_bytes(response.into_body(), 1024 * 1024)
        .await
        .unwrap();
    let json = if bytes.is_empty() {
        Value::Null
    } else {
        serde_json::from_slice(&bytes).unwrap_or(Value::Null)
    };
    (status, json)
}

async fn admin_post(app: axum::Router, uri: &str, body: Value) -> (StatusCode, Value) {
    send(
        app,
        Request::post(uri)
            .header("content-type", "application/json")
            .body(Body::from(body.to_string()))
            .unwrap(),
    )
    .await
}

async fn chat(app: axum::Router, body: Value) -> (StatusCode, Value) {
    send(
        app,
        Request::post("/v1/chat/completions")
            .header("content-type", "application/json")
            .body(Body::from(body.to_string()))
            .unwrap(),
    )
    .await
}

async fn chat_with_key(app: axum::Router, bearer: &str, body: Value) -> (StatusCode, Value) {
    send(
        app,
        Request::post("/v1/chat/completions")
            .header("content-type", "application/json")
            .header("authorization", format!("Bearer {bearer}"))
            .body(Body::from(body.to_string()))
            .unwrap(),
    )
    .await
}

/// A chat request body carrying the given `prompt` reference plus a
/// one-message client tail.
fn chat_body(prompt: Value) -> Value {
    json!({
        "model": MODEL_GROUP,
        "messages": [{ "role": "user", "content": "client tail" }],
        "prompt": prompt
    })
}

async fn create_prompt(app: &axum::Router, name: &str, content: &str) -> Value {
    let (status, body) = admin_post(
        app.clone(),
        &format!("{ADMIN_BASE}/prompts"),
        json!({ "name": name, "content": content }),
    )
    .await;
    assert_eq!(status, StatusCode::OK, "seed create failed: {body}");
    body
}

async fn create_version(app: &axum::Router, id: &str, content: &str, change_note: &str) {
    let (status, body) = admin_post(
        app.clone(),
        &format!("{ADMIN_BASE}/prompts/{id}/versions"),
        json!({ "content": content, "change_note": change_note }),
    )
    .await;
    assert_eq!(status, StatusCode::OK, "seed version failed: {body}");
}

async fn promote(app: &axum::Router, id: &str, label: &str, to_version: i64) {
    let (status, body) = admin_post(
        app.clone(),
        &format!("{ADMIN_BASE}/prompts/{id}/promote"),
        json!({ "label": label, "to_version": to_version }),
    )
    .await;
    assert_eq!(status, StatusCode::OK, "seed promote failed: {body}");
}

/// Bodies the mock provider received for chat completions, in order.
async fn upstream_bodies(mock: &MockServer) -> Vec<Value> {
    mock.received_requests()
        .await
        .expect("wiremock request recording")
        .into_iter()
        .filter(|request| request.url.path() == "/v1/chat/completions")
        .map(|request| serde_json::from_slice(&request.body).expect("provider body is JSON"))
        .collect()
}

/// Poll `/dashboard/logs` until `done` holds or `timeout` passes; returns
/// the last read (pattern from tests/request_outcome_logging.rs).
async fn poll_logs(
    server: &GatewayServer,
    timeout: Duration,
    done: impl Fn(&[LogEntry]) -> bool,
) -> Vec<LogEntry> {
    let deadline = tokio::time::Instant::now() + timeout;
    loop {
        let response = server
            .build_router()
            .oneshot(Request::get("/dashboard/logs").body(Body::empty()).unwrap())
            .await
            .unwrap();
        assert_eq!(response.status(), StatusCode::OK);
        let bytes = axum::body::to_bytes(response.into_body(), 4 * 1024 * 1024)
            .await
            .unwrap();
        let rows: Vec<LogEntry> =
            serde_json::from_slice(&bytes).expect("dashboard logs should be LogEntry JSON");
        if done(&rows) || tokio::time::Instant::now() >= deadline {
            return rows;
        }
        tokio::time::sleep(Duration::from_millis(50)).await;
    }
}

/// Default (all-unlimited) create params (pattern from
/// tests/virtual_keys_integration.rs).
fn key_defaults() -> CreateKeyParams {
    CreateKeyParams {
        name: None,
        budget_limit_usd: None,
        token_budget: None,
        budget_window: None,
        requests_per_minute: None,
        tokens_per_minute: None,
        model_access: None,
        expires_in: None,
        loop_detection: None,
    }
}

// ---------------------------------------------------------------------------
// Resolution forms (Req 4.1)
// ---------------------------------------------------------------------------

/// Req 4.1: prompt-by-name, name@label, and uuid id references all
/// resolve, merge, and route; the mock provider's 200 reaches the client.
/// Label omitted defaults to `production`.
#[tokio::test]
async fn prompt_by_name_label_and_uuid_all_resolve_and_route() {
    let mock = start_mock_provider().await;
    let (app, _server) = build_app(prompt_api_config(&mock.uri())).await;
    let created = create_prompt(&app, "greeting", TEMPLATE).await;
    let id = created["id"].as_str().expect("created id").to_owned();

    for reference in ["greeting", "greeting@production", id.as_str()] {
        let (status, body) = chat(app.clone(), chat_body(json!({ "id": reference }))).await;
        assert_eq!(status, StatusCode::OK, "reference {reference}: {body}");
        assert_eq!(body["choices"][0]["message"]["content"], "Hello from mock provider!");
    }

    let hits = upstream_bodies(&mock).await;
    assert_eq!(hits.len(), 3, "each resolved form dispatches once");
    for hit in &hits {
        assert_eq!(
            hit["messages"].as_array().map(Vec::len),
            Some(3),
            "template + client messages merged: {hit}"
        );
        assert_eq!(hit["messages"][0]["content"], "TEMPLATE SYSTEM VOICE");
        assert_eq!(hit["messages"][1]["content"], "TEMPLATE TASK");
        assert_eq!(hit["messages"][2]["content"], "client tail");
        assert!(
            hit.get("prompt").is_none(),
            "the prompt field is stripped from the outgoing body: {hit}"
        );
    }
}

/// Req 3.5: after production is promoted to v2, an explicit version=1
/// call still resolves immutable v1 content, while a call by name
/// follows the moved label and serves v2.
#[tokio::test]
async fn pinned_version_remains_stable_across_label_promotion() {
    let mock = start_mock_provider().await;
    let (app, _server) = build_app(prompt_api_config(&mock.uri())).await;
    let created = create_prompt(&app, "movable", r#"[{"role":"user","content":"TEMPLATE ONE"}]"#).await;
    let id = created["id"].as_str().expect("created id").to_owned();
    create_version(&app, &id, r#"[{"role":"user","content":"TEMPLATE TWO"}]"#, "second version").await;
    promote(&app, &id, "production", 2).await;

    let (status, body) = chat(app.clone(), chat_body(json!({ "id": "movable", "version": 1 }))).await;
    assert_eq!(status, StatusCode::OK, "{body}");

    let (status, body) = chat(app.clone(), chat_body(json!({ "id": "movable" }))).await;
    assert_eq!(status, StatusCode::OK, "{body}");

    let hits = upstream_bodies(&mock).await;
    assert_eq!(hits.len(), 2);
    let pinned = hits[0]["messages"][0]["content"].as_str().unwrap();
    assert!(
        pinned.contains("TEMPLATE ONE") && !pinned.contains("TEMPLATE TWO"),
        "explicit version=1 pins v1: {pinned}"
    );
    let by_label = hits[1]["messages"][0]["content"].as_str().unwrap();
    assert!(
        by_label.contains("TEMPLATE TWO") && !by_label.contains("TEMPLATE ONE"),
        "bare name follows the promoted production label: {by_label}"
    );
}

// ---------------------------------------------------------------------------
// Merge order and rendering (Req 4.2, 4.1)
// ---------------------------------------------------------------------------

/// Req 4.2: template messages come first, client-provided messages append
/// after, and other client fields (model, temperature) pass through
/// untouched - asserted on the body the mock provider received.
#[tokio::test]
async fn template_messages_merge_before_client_messages() {
    let mock = start_mock_provider().await;
    let (app, _server) = build_app(prompt_api_config(&mock.uri())).await;
    create_prompt(
        &app,
        "ordered",
        r#"[{"role":"system","content":"TEMPLATE FIRST"},{"role":"user","content":"TEMPLATE SECOND"}]"#,
    )
    .await;

    let (status, body) = chat(
        app,
        json!({
            "model": MODEL_GROUP,
            "messages": [{ "role": "user", "content": "CLIENT AFTER TEMPLATE" }],
            "temperature": 0.25,
            "prompt": { "id": "ordered" }
        }),
    )
    .await;
    assert_eq!(status, StatusCode::OK, "{body}");

    let hits = upstream_bodies(&mock).await;
    assert_eq!(hits.len(), 1);
    let messages = hits[0]["messages"].as_array().expect("merged messages");
    assert_eq!(messages.len(), 3);
    assert_eq!(messages[0]["content"], "TEMPLATE FIRST");
    assert_eq!(messages[1]["content"], "TEMPLATE SECOND");
    assert_eq!(messages[2]["content"], "CLIENT AFTER TEMPLATE");
    assert_eq!(hits[0]["model"], "gpt-4", "client fields route through untouched");
    assert_eq!(hits[0]["temperature"], 0.25);
}

/// Req 4.1: provided variables are substituted into the rendered template
/// (non-strings serialized compactly); a missing variable renders as the
/// empty string.
#[tokio::test]
async fn variables_render_and_missing_variables_become_empty() {
    let mock = start_mock_provider().await;
    let (app, _server) = build_app(prompt_api_config(&mock.uri())).await;
    create_prompt(
        &app,
        "templated",
        r#"[{"role":"user","content":"Hello {{name}}, age {{age}}, missing {{absent}}"}]"#,
    )
    .await;

    let (status, body) = chat(
        app,
        chat_body(json!({
            "id": "templated",
            "variables": { "name": "Ada", "age": 36 }
        })),
    )
    .await;
    assert_eq!(status, StatusCode::OK, "{body}");

    let hits = upstream_bodies(&mock).await;
    assert_eq!(hits.len(), 1);
    assert_eq!(hits[0]["messages"][0]["content"], "Hello Ada, age 36, missing ");
}

// ---------------------------------------------------------------------------
// Reference errors (Req 4.6, 4.3)
// ---------------------------------------------------------------------------

/// Req 4.6: `version` combined with a label-bearing id is an ambiguous
/// reference - 400 naming the `version` field, with no provider contact.
#[tokio::test]
async fn version_combined_with_label_reference_is_rejected() {
    let mock = start_mock_provider().await;
    let (app, _server) = build_app(prompt_api_config(&mock.uri())).await;
    create_prompt(&app, "ambiguous", TEMPLATE).await;

    let (status, body) = chat(
        app,
        chat_body(json!({ "id": "ambiguous@production", "version": 2 })),
    )
    .await;
    assert_eq!(status, StatusCode::BAD_REQUEST, "{body}");
    assert_eq!(body["error"]["type"], "invalid_request_error");
    assert!(
        body["error"]["message"].as_str().unwrap().contains("version"),
        "error must name the field: {body}"
    );
    assert!(
        mock.received_requests().await.unwrap().is_empty(),
        "rejected before dispatch: no provider contact"
    );
}

/// Req 4.3: unknown name and unresolvable label both fail with a
/// 404-class invalid_request_error naming the prompt and the label, with
/// no provider contact.
#[tokio::test]
async fn unknown_prompt_or_label_returns_404_without_provider_contact() {
    let mock = start_mock_provider().await;
    let (app, _server) = build_app(prompt_api_config(&mock.uri())).await;
    create_prompt(&app, "known", TEMPLATE).await;

    let (status, body) = chat(app.clone(), chat_body(json!({ "id": "ghost-prompt" }))).await;
    assert_eq!(status, StatusCode::NOT_FOUND, "{body}");
    assert_eq!(body["error"]["type"], "invalid_request_error");
    let message = body["error"]["message"].as_str().unwrap();
    assert!(
        message.contains("ghost-prompt") && message.contains("production"),
        "error names prompt and label: {message}"
    );

    let (status, body) = chat(app, chat_body(json!({ "id": "known@staging" }))).await;
    assert_eq!(status, StatusCode::NOT_FOUND, "{body}");
    assert_eq!(body["error"]["type"], "invalid_request_error");
    let message = body["error"]["message"].as_str().unwrap();
    assert!(
        message.contains("known") && message.contains("staging"),
        "error names prompt and label: {message}"
    );

    assert!(
        mock.received_requests().await.unwrap().is_empty(),
        "unresolvable references never contact a provider"
    );
}

/// Req 4.3: a malformed prompt field (non-object, missing id, non-object
/// variables) is a 400 naming the offending sub-field, with no provider
/// contact.
#[tokio::test]
async fn malformed_prompt_field_returns_400_naming_the_field() {
    let mock = start_mock_provider().await;
    let (app, _server) = build_app(prompt_api_config(&mock.uri())).await;

    let cases = [
        (json!(["not", "an", "object"]), "prompt"),
        (json!({}), "id"),
        (json!({ "id": "greeting", "variables": "map" }), "variables"),
    ];
    for (prompt, field) in cases {
        let (status, body) = chat(app.clone(), chat_body(prompt)).await;
        assert_eq!(status, StatusCode::BAD_REQUEST, "field {field}: {body}");
        assert_eq!(body["error"]["type"], "invalid_request_error");
        assert!(
            body["error"]["message"].as_str().unwrap().contains(field),
            "error must name {field}: {body}"
        );
    }
    assert!(
        mock.received_requests().await.unwrap().is_empty(),
        "malformed prompt fields never contact a provider"
    );
}

// ---------------------------------------------------------------------------
// Baseline and observability (Req 4.1, 4.4)
// ---------------------------------------------------------------------------

/// Req 4.1: a request without the `prompt` field routes exactly as
/// before - same dispatch, same body, and an unattributed log row.
#[tokio::test]
async fn request_without_prompt_field_routes_normally() {
    let mock = start_mock_provider().await;
    let (app, server) = build_app(prompt_api_config(&mock.uri())).await;

    let (status, body) = chat(
        app,
        json!({
            "model": MODEL_GROUP,
            "messages": [{ "role": "user", "content": "plain request" }]
        }),
    )
    .await;
    assert_eq!(status, StatusCode::OK, "{body}");
    assert_eq!(body["choices"][0]["message"]["content"], "Hello from mock provider!");

    let hits = upstream_bodies(&mock).await;
    assert_eq!(hits.len(), 1);
    assert_eq!(hits[0]["messages"].as_array().map(Vec::len), Some(1));
    assert_eq!(hits[0]["messages"][0]["content"], "plain request");

    let rows = poll_logs(&server, Duration::from_secs(5), |rows| {
        rows.iter()
            .any(|row| row.path == "/v1/chat/completions" && row.status_code == 200)
    })
    .await;
    let row = rows
        .iter()
        .find(|row| row.path == "/v1/chat/completions" && row.status_code == 200)
        .expect("outcome row");
    assert!(
        row.prompt_id.is_none() && row.prompt_version.is_none() && row.prompt_label.is_none(),
        "no-prompt rows stay unattributed: {row:?}"
    );
}

/// Req 4.4: the outcome log row carries prompt_id/prompt_version/
/// prompt_label, and neither template content nor variable values are
/// stored anywhere in the row.
#[tokio::test]
async fn outcome_log_row_carries_prompt_attribution_without_template_content() {
    let mock = start_mock_provider().await;
    let (app, server) = build_app(prompt_api_config(&mock.uri())).await;
    let created = create_prompt(
        &app,
        "audited",
        r#"[{"role":"user","content":"AUDIT-SENTINEL {{secret}}"}]"#,
    )
    .await;
    let id = created["id"].as_str().expect("created id").to_owned();

    let (status, body) = chat(
        app,
        chat_body(json!({
            "id": "audited",
            "variables": { "secret": "variable-value-sentinel" }
        })),
    )
    .await;
    assert_eq!(status, StatusCode::OK, "{body}");

    let rows = poll_logs(&server, Duration::from_secs(5), |rows| {
        rows.iter().any(|row| row.prompt_id.as_deref() == Some(id.as_str()))
    })
    .await;
    let row = rows
        .iter()
        .find(|row| row.prompt_id.as_deref() == Some(id.as_str()))
        .expect("prompt-attributed outcome row");
    assert_eq!(row.status_code, 200);
    assert_eq!(row.prompt_id.as_deref(), Some(id.as_str()));
    assert_eq!(row.prompt_version, Some(1));
    assert_eq!(row.prompt_label.as_deref(), Some("production"));
    assert!(
        row.request_body.is_none(),
        "request snapshots never carry the template"
    );
    let serialized = serde_json::to_string(row).unwrap();
    assert!(
        !serialized.contains("AUDIT-SENTINEL") && !serialized.contains("variable-value-sentinel"),
        "template content and variable values stay out of the log row: {serialized}"
    );
}

/// Req 4.4 with request-body logging ON: even then the prompt-resolved
/// row stores no request body - the rendered template and variable
/// values must not land in the log store - while a plain request
/// logged through the same logger keeps its body.
#[tokio::test]
async fn prompt_row_stores_no_template_content_even_with_body_logging_enabled() {
    let mock = start_mock_provider().await;
    let mut config = prompt_api_config(&mock.uri());
    config.logging.request_body_logging = true;
    let (app, server) = build_app(config).await;
    let created = create_prompt(
        &app,
        "body-logged",
        r#"[{"role":"user","content":"BODY-LOG-SENTINEL {{secret}}"}]"#,
    )
    .await;
    let id = created["id"].as_str().expect("created id").to_owned();

    let (status, body) = chat(
        app.clone(),
        chat_body(json!({
            "id": "body-logged",
            "variables": { "secret": "body-log-variable-value" }
        })),
    )
    .await;
    assert_eq!(status, StatusCode::OK, "{body}");

    let (status, plain_body) = chat(
        app,
        json!({
            "model": MODEL_GROUP,
            "messages": [{ "role": "user", "content": "plain logged request" }]
        }),
    )
    .await;
    assert_eq!(status, StatusCode::OK, "{plain_body}");

    let rows = poll_logs(&server, Duration::from_secs(5), |rows| {
        rows.iter().any(|row| row.prompt_id.as_deref() == Some(id.as_str()))
            && rows
                .iter()
                .any(|row| row.path == "/v1/chat/completions" && row.prompt_id.is_none())
    })
    .await;
    let prompt_row = rows
        .iter()
        .find(|row| row.prompt_id.as_deref() == Some(id.as_str()))
        .expect("prompt-attributed outcome row");
    assert!(
        prompt_row.request_body.is_none(),
        "prompt-resolved rows never store the merged body, even with body logging on"
    );
    let plain_row = rows
        .iter()
        .find(|row| row.path == "/v1/chat/completions" && row.prompt_id.is_none())
        .expect("plain outcome row");
    assert!(
        plain_row
            .request_body
            .as_ref()
            .is_some_and(|body| body.contains("plain logged request")),
        "plain rows keep their logged body: {:?}",
        plain_row.request_body
    );
}

// ---------------------------------------------------------------------------
// Virtual-key enforcement (Req 4.5)
// ---------------------------------------------------------------------------

/// Req 4.5: a virtual key restricted to a model the merged request does
/// not use is blocked with 403 exactly as for a plain request; a key
/// allowed the model dispatches the merged request normally.
#[tokio::test]
async fn virtual_key_model_restriction_applies_to_merged_request() {
    let mock = start_mock_provider().await;
    let mut config = prompt_api_config(&mock.uri());
    config.virtual_keys.enforcement = EnforcementMode::Required;
    let (app, server) = build_app(config).await;
    create_prompt(&app, "guarded", r#"[{"role":"user","content":"TEMPLATE GUARDED"}]"#).await;

    let denied = server
        .state
        .virtual_key_manager
        .create_key(CreateKeyParams {
            model_access: Some(vec!["allowed-group".to_owned()]),
            ..key_defaults()
        })
        .await
        .unwrap();
    let allowed = server
        .state
        .virtual_key_manager
        .create_key(CreateKeyParams {
            model_access: Some(vec![MODEL_GROUP.to_owned()]),
            ..key_defaults()
        })
        .await
        .unwrap();

    let (status, body) = chat_with_key(app.clone(), &denied.key, chat_body(json!({ "id": "guarded" }))).await;
    assert_eq!(
        status,
        StatusCode::FORBIDDEN,
        "merged requests face the same model restriction as plain ones: {body}"
    );

    let (status, body) = chat_with_key(app, &allowed.key, chat_body(json!({ "id": "guarded" }))).await;
    assert_eq!(status, StatusCode::OK, "{body}");

    let hits = upstream_bodies(&mock).await;
    assert_eq!(hits.len(), 1, "only the permitted merged request reaches the provider");
    assert_eq!(hits[0]["messages"][0]["content"], "TEMPLATE GUARDED");
}
