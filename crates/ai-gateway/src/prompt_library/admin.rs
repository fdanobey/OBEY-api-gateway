//! Administrative HTTP API, nested under `/admin/prompt-library`.
//! Authentication belongs to the parent admin router.

use std::sync::Arc;

use axum::{
    extract::{rejection::{JsonRejection, QueryRejection}, Path, Query, State},
    http::StatusCode,
    response::{IntoResponse, Response},
    routing::{get, post},
    Json, Router,
};
use serde::{Deserialize, Serialize};
use serde_json::json;
use tokio::sync::RwLock;

use super::{ListPage, ListParams, PromptLabel, PromptLibrary, PromptLibraryConfig, PromptStore, StoreError};
use crate::config::save_config;
use crate::gateway::{apply_runtime_config_update, AppState};

type OptionalPromptLibrary = Arc<RwLock<Option<Arc<PromptLibrary>>>>;

#[derive(Clone)]
pub enum PromptLibraryAdminState {
    Gateway(AppState),
    Optional(OptionalPromptLibrary),
}

impl From<AppState> for PromptLibraryAdminState {
    fn from(state: AppState) -> Self {
        Self::Gateway(state)
    }
}

impl From<OptionalPromptLibrary> for PromptLibraryAdminState {
    fn from(library: OptionalPromptLibrary) -> Self {
        Self::Optional(library)
    }
}

impl PromptLibraryAdminState {
    async fn snapshot(&self) -> Option<Arc<PromptLibrary>> {
        match self {
            Self::Gateway(state) => state.prompt_library.read().await.clone(),
            Self::Optional(library) => library.read().await.clone(),
        }
    }
}

pub fn routes<S>(state: impl Into<PromptLibraryAdminState>) -> Router<S>
where
    S: Clone + Send + Sync + 'static,
{
    Router::new()
        .route("/prompts", get(list_prompts).post(create_prompt))
        .route("/prompts/{id}", get(get_prompt).delete(delete_prompt))
        .route("/prompts/{id}/versions", get(list_versions).post(create_version))
        .route("/prompts/{id}/diff", post(diff_versions))
        .route("/prompts/{id}/promote", post(promote_label))
        .route("/prompts/{id}/promotions", get(list_promotions))
        .route("/prompts/{id}/labels", get(list_labels))
        .route("/config", get(get_config).put(update_config))
        .with_state(state.into())
}

#[derive(Debug, Deserialize)]
#[serde(deny_unknown_fields)]
struct CreatePrompt {
    name: String,
    content: String,
    created_by: Option<String>,
}

#[derive(Debug, Deserialize)]
#[serde(deny_unknown_fields)]
struct CreateVersion {
content: String,
change_note: String,
name: Option<String>,
created_by: Option<String>,
}

#[derive(Debug, Deserialize)]
#[serde(deny_unknown_fields)]
struct DiffVersions {
    from_version: i64,
    to_version: i64,
}

#[derive(Debug, Deserialize)]
#[serde(deny_unknown_fields)]
struct PromoteLabel {
    label: String,
    to_version: i64,
    actor: Option<String>,
}

#[derive(Debug, Deserialize)]
#[serde(deny_unknown_fields)]
struct UpdateConfig {
    database_path: String,
}

async fn resolve_library(state: &PromptLibraryAdminState) -> Result<Arc<PromptLibrary>, Response> {
    state.snapshot().await.ok_or_else(unavailable)
}

/// Clone the snapshot before dispatching synchronous SQLite work to the blocking pool.
async fn spawn_store_task<T, F>(state: &PromptLibraryAdminState, task: F) -> Result<T, Response>
where
    F: FnOnce(&PromptStore) -> Result<T, StoreError> + Send + 'static,
    T: Send + 'static,
{
    let library = resolve_library(state).await?;
    match tokio::task::spawn_blocking(move || task(library.store())).await {
        Ok(result) => result.map_err(store_error_response),
        Err(_) => Err(unavailable()),
    }
}

fn json_response<T: Serialize>(result: Result<T, Response>) -> Response {
    match result {
        Ok(value) => Json(value).into_response(),
        Err(response) => response,
    }
}

async fn list_prompts(
    State(state): State<PromptLibraryAdminState>,
    query: Result<Query<ListParams>, QueryRejection>,
) -> Response {
    let Query(query) = match query {
        Ok(query) => query,
        Err(error) => return bad_request(&error.body_text()),
    };
    if state.snapshot().await.is_none() {
        return Json(ListPage { data: Vec::new(), has_more: false }.into_openai_response())
            .into_response();
    }
    // Cursor errors in the store call name "cursor"; include the concrete query field too.
    let cursor_field = if query.after.is_some() { "after" } else { "before" };
    json_response(spawn_store_task(&state, move |store| {
        store.list_prompts(&query).map(ListPage::into_openai_response).map_err(|error| {
            match error {
                StoreError::InvalidRequest(message) if message.starts_with("cursor ") => {
                    StoreError::InvalidRequest(format!("{cursor_field}: {message}"))
                }
                error => error,
            }
        })
    }).await)
}

async fn create_prompt(
    State(state): State<PromptLibraryAdminState>,
    payload: Result<Json<CreatePrompt>, JsonRejection>,
) -> Response {
    let Json(payload) = match payload {
        Ok(payload) => payload,
        Err(error) => return bad_request(&error.body_text()),
    };
    match spawn_store_task(&state, move |store| {
        store.create_prompt(&payload.name, &payload.content, payload.created_by.as_deref().unwrap_or("admin"))
    }).await {
        Ok(prompt) => (StatusCode::OK, Json(prompt)).into_response(),
        Err(response) => response,
    }
}

async fn get_prompt(State(state): State<PromptLibraryAdminState>, Path(id): Path<String>) -> Response {
    json_response(spawn_store_task(&state, move |store| store.get_prompt(&id)).await)
}

async fn create_version(
    State(state): State<PromptLibraryAdminState>,
    Path(id): Path<String>,
    payload: Result<Json<CreateVersion>, JsonRejection>,
) -> Response {
    let Json(payload) = match payload {
        Ok(payload) => payload,
        Err(error) => return bad_request(&error.body_text()),
    };
 match spawn_store_task(&state, move |store| {
 store.create_version(&id, payload.name.as_deref(), &payload.content, &payload.change_note, payload.created_by.as_deref().unwrap_or("admin"))
 }).await {
        Ok(version) => (StatusCode::OK, Json(version)).into_response(),
        Err(response) => response,
    }
}

async fn list_versions(State(state): State<PromptLibraryAdminState>, Path(id): Path<String>) -> Response {
    json_response(spawn_store_task(&state, move |store| store.list_versions(&id)).await)
}

async fn diff_versions(
    State(state): State<PromptLibraryAdminState>,
    Path(id): Path<String>,
    payload: Result<Json<DiffVersions>, JsonRejection>,
) -> Response {
    let Json(payload) = match payload {
        Ok(payload) => payload,
        Err(error) => return bad_request(&error.body_text()),
    };
    json_response(spawn_store_task(&state, move |store| {
        store.diff_versions(&id, payload.from_version, payload.to_version)
    }).await)
}

async fn promote_label(
    State(state): State<PromptLibraryAdminState>,
    Path(id): Path<String>,
    payload: Result<Json<PromoteLabel>, JsonRejection>,
) -> Response {
    let Json(payload) = match payload {
        Ok(payload) => payload,
        Err(error) => return bad_request(&error.body_text()),
    };
    json_response(spawn_store_task(&state, move |store| {
        store.promote_label(&id, &payload.label, payload.to_version, payload.actor.as_deref().unwrap_or("admin"))?;
        Ok(PromptLabel {
            prompt_id: id,
            object: "prompt.label",
            label: payload.label,
            version: Some(payload.to_version),
            is_virtual: false,
        })
    }).await)
}

async fn list_promotions(State(state): State<PromptLibraryAdminState>, Path(id): Path<String>) -> Response {
    json_response(spawn_store_task(&state, move |store| {
        store.list_promotions(&id).map(ListPage::into_openai_response)
    }).await)
}

async fn list_labels(State(state): State<PromptLibraryAdminState>, Path(id): Path<String>) -> Response {
    json_response(spawn_store_task(&state, move |store| store.list_labels(&id)).await)
}

async fn delete_prompt(State(state): State<PromptLibraryAdminState>, Path(id): Path<String>) -> Response {
    match spawn_store_task(&state, move |store| store.delete_prompt(&id)).await {
        Ok(()) => StatusCode::NO_CONTENT.into_response(),
        Err(response) => response,
    }
}

async fn get_config(State(state): State<PromptLibraryAdminState>) -> Response {
    let config = match state.snapshot().await {
        Some(library) => PromptLibraryConfig { database_path: library.database_path().to_owned() },
        None => PromptLibraryConfig::default(),
    };
    Json(config).into_response()
}

async fn update_config(
    State(state): State<PromptLibraryAdminState>,
    payload: Result<Json<UpdateConfig>, JsonRejection>,
) -> Response {
    let Json(payload) = match payload {
        Ok(payload) => payload,
        Err(error) => return bad_request(&error.body_text()),
    };
    if let Err(response) = resolve_library(&state).await {
        return response;
    }
    let config = PromptLibraryConfig { database_path: payload.database_path.clone() };
    if let Err(errors) = config.validate() {
        return bad_request(&errors.iter().map(ToString::to_string).collect::<Vec<_>>().join("; "));
    }
    let PromptLibraryAdminState::Gateway(state) = state else {
        return unavailable();
    };
    let mut candidate = state.config.read().await.clone();
    candidate.prompt_library = config;
    if let Err(errors) = candidate.validate() {
        return bad_request(&errors.iter().map(ToString::to_string).collect::<Vec<_>>().join("; "));
    }
    // Mirror the sibling smart-routing config endpoint: validate, persist, then apply.
    candidate.admin.security_mode = Some(crate::admin::auth_policy::resolve_admin_security_mode(&candidate));
    if let Err(error) = save_config(state.config_path.as_ref(), &candidate) {
        return error_response(StatusCode::INTERNAL_SERVER_ERROR, &error, "io_error");
    }
    apply_runtime_config_update(&state, candidate).await;
    // The apply step only warns when the new store fails to open, so the
    // response must verify the switch actually happened: read the live
    // snapshot's database_path back and compare it with the request. The
    // persisted config file is deliberately not rolled back on mismatch.
    let active_path = state.prompt_library.read().await.as_ref().map(|library| {
        library.database_path().to_owned()
    });
    match active_path {
        Some(path) if path == payload.database_path => Json(json!({
            "status": "ok",
            "database_path": path
        }))
        .into_response(),
        _ => error_response(
            StatusCode::SERVICE_UNAVAILABLE,
            "prompt library did not switch to the requested database path; previous snapshot retained",
            "api_error",
        ),
    }
}

fn bad_request(message: &str) -> Response {
    error_response(StatusCode::BAD_REQUEST, message, "invalid_request")
}

fn unavailable() -> Response {
    error_response(StatusCode::SERVICE_UNAVAILABLE, "prompt library unavailable", "service_unavailable")
}

fn store_error_response(error: StoreError) -> Response {
    match error {
        StoreError::NotFound { .. } => error_response(StatusCode::NOT_FOUND, &error.to_string(), "not_found"),
        StoreError::InvalidRequest(_) | StoreError::TooLarge { .. } | StoreError::LimitExceeded { .. } => {
            bad_request(&error.to_string())
        }
        error => error_response(StatusCode::SERVICE_UNAVAILABLE, &error.to_string(), "service_unavailable"),
    }
}

fn error_response(status: StatusCode, message: &str, error_type: &str) -> Response {
    (status, Json(json!({ "error": { "message": message, "type": error_type } }))).into_response()
}

#[cfg(test)]
mod tests {
    use axum::{body::{to_bytes, Body}, http::Request};
    use serde_json::Value;
    use tower::ServiceExt;

    use super::*;

    fn app() -> (tempfile::TempDir, Router) {
        let temp = tempfile::tempdir().unwrap();
        let config = PromptLibraryConfig {
            database_path: temp.path().join("prompts.db").to_string_lossy().into_owned(),
        };
        let library = super::super::build_prompt_library(&config).unwrap();
        let handle: OptionalPromptLibrary = Arc::new(RwLock::new(Some(library)));
        (temp, routes(handle))
    }

    async fn request(app: Router, method: &str, uri: &str, body: Option<Value>) -> (StatusCode, Value) {
        let mut builder = Request::builder().method(method).uri(uri);
        let body = match body {
            Some(body) => {
                builder = builder.header("content-type", "application/json");
                Body::from(serde_json::to_vec(&body).unwrap())
            }
            None => Body::empty(),
        };
        let response = app.oneshot(builder.body(body).unwrap()).await.unwrap();
        let status = response.status();
        let bytes = to_bytes(response.into_body(), 1024 * 1024).await.unwrap();
        (status, if bytes.is_empty() { Value::Null } else { serde_json::from_slice(&bytes).unwrap() })
    }

    #[tokio::test]
    async fn prompt_round_trip() {
        let (_temp, app) = app();
        let (status, created) = request(app.clone(), "POST", "/prompts", Some(json!({
            "name": "greeting", "content": "Hello {{name}}!"
        }))).await;
        assert_eq!(status, StatusCode::OK);
        assert_eq!(created["labels"]["production"], 1);
        let uri = format!("/prompts/{}", created["id"].as_str().unwrap());
        let (status, fetched) = request(app.clone(), "GET", &uri, None).await;
        assert_eq!(status, StatusCode::OK);
        assert_eq!(fetched, created);
        let (status, version) = request(app.clone(), "POST", &format!("{uri}/versions"), Some(json!({
            "content": "Welcome {{name}}!", "change_note": "new greeting", "created_by": "editor"
        }))).await;
        assert_eq!(status, StatusCode::OK);
        assert_eq!(version["version"], 2);
        assert_eq!(version["created_by"], "editor");
        let (status, versions) = request(app.clone(), "GET", &format!("{uri}/versions"), None).await;
        assert_eq!(status, StatusCode::OK);
        assert_eq!(versions[0]["version"], 2);
        assert_eq!(versions[1]["version"], 1);
        assert_eq!(versions[1]["created_by"], "admin");
        let (status, diff) = request(app.clone(), "POST", &format!("{uri}/diff"), Some(json!({
            "from_version": 1, "to_version": 2
        }))).await;
        assert_eq!(status, StatusCode::OK);
        assert_eq!(diff, json!([
            {"kind": "removed", "text": "Hello {{name}}!"},
            {"kind": "added", "text": "Welcome {{name}}!"}
        ]));
        let (status, moved) = request(app.clone(), "POST", &format!("{uri}/promote"), Some(json!({
            "label": "production", "to_version": 2, "actor": "operator"
        }))).await;
        assert_eq!(status, StatusCode::OK);
        assert_eq!(moved["label"], "production");
        assert_eq!(moved["version"], 2);
        let (status, history) = request(app.clone(), "GET", &format!("{uri}/promotions"), None).await;
        assert_eq!(status, StatusCode::OK);
        let rows = history["data"].as_array().unwrap();
        assert_eq!(rows.len(), 2);
        // Same-second promotions use the store's id tie-breaker.
        assert!(rows.windows(2).all(|pair| {
            (pair[0]["created_at"].as_i64(), pair[0]["id"].as_str()) >=
                (pair[1]["created_at"].as_i64(), pair[1]["id"].as_str())
        }));
        let promotion = rows.iter().find(|row| row["to_version"] == 2).unwrap();
        assert_eq!(promotion["from_version"], 1);
        assert_eq!(promotion["actor"], "operator");
        let (status, labels) = request(app.clone(), "GET", &format!("{uri}/labels"), None).await;
        assert_eq!(status, StatusCode::OK);
        let labels = labels.as_array().unwrap();
        let latest = labels.iter().find(|label| label["label"] == "latest").unwrap();
        assert_eq!(latest["virtual"], true);
        assert_eq!(latest["version"], 2);
        assert_eq!(labels.iter().find(|label| label["label"] == "production").unwrap()["version"], 2);
        let (_, fetched) = request(app.clone(), "GET", &uri, None).await;
        assert_eq!(fetched["version_count"], 2);
        assert_eq!(fetched["labels"]["production"], 2);
        let (status, _) = request(app.clone(), "DELETE", &uri, None).await;
        assert_eq!(status, StatusCode::NO_CONTENT);
        for path in [&uri, &format!("{uri}/versions"), &format!("{uri}/labels"), &format!("{uri}/promotions")] {
            let (status, body) = request(app.clone(), "GET", path, None).await;
            assert_eq!(status, StatusCode::NOT_FOUND);
            assert_eq!(body["error"]["type"], "not_found");
        }
    }

    #[tokio::test]
    async fn invalid_create_names_the_field() {
        let (_temp, app) = app();
        for (payload, field) in [
            (json!({"name": "Invalid", "content": "hello"}), "name"),
            (json!({"name": "valid", "content": ""}), "content"),
            (json!({"name": "valid"}), "content"),
            (json!({"name": 123, "content": "hello"}), "name"),
            (json!({"name": "valid", "content": "x".repeat(100_001)}), "content"),
        ] {
            let (status, body) = request(app.clone(), "POST", "/prompts", Some(payload)).await;
            assert_eq!(status, StatusCode::BAD_REQUEST);
            assert_eq!(body["error"]["type"], "invalid_request");
            assert!(body["error"]["message"].as_str().unwrap().contains(field), "{body}");
        }
        let response = app.oneshot(Request::post("/prompts")
            .header("content-type", "application/json")
            .body(Body::from(r#"{"name":"valid","content": }"#)).unwrap()).await.unwrap();
        assert_eq!(response.status(), StatusCode::BAD_REQUEST);
        let bytes = to_bytes(response.into_body(), 64 * 1024).await.unwrap();
        let body: Value = serde_json::from_slice(&bytes).unwrap();
        assert_eq!(body["error"]["type"], "invalid_request");
    }

    #[tokio::test]
    async fn dormant_list_config_and_mutations() {
        let handle: OptionalPromptLibrary = Arc::new(RwLock::new(None));
        let app = routes(handle);
        let (status, page) = request(app.clone(), "GET", "/prompts", None).await;
        assert_eq!(status, StatusCode::OK);
        assert_eq!(page, json!({"object": "list", "data": [], "first_id": null, "last_id": null, "has_more": false}));
        let (status, config) = request(app.clone(), "GET", "/config", None).await;
        assert_eq!(status, StatusCode::OK);
        assert_eq!(config, json!({"database_path": PromptLibraryConfig::default().database_path}));
        for (method, uri, payload) in [
            ("POST", "/prompts", Some(json!({"name": "test", "content": "hello"}))),
            ("POST", "/prompts/missing/versions", Some(json!({"content": "hello", "change_note": "new"}))),
            ("POST", "/prompts/missing/promote", Some(json!({"label": "production", "to_version": 1}))),
            ("DELETE", "/prompts/missing", None),
            ("PUT", "/config", Some(json!({"database_path": "unused.db"}))),
        ] {
            let (status, body) = request(app.clone(), method, uri, payload).await;
            assert_eq!(status, StatusCode::SERVICE_UNAVAILABLE);
            assert_eq!(body["error"]["type"], "service_unavailable");
            assert_eq!(body["error"]["message"], "prompt library unavailable");
        }
    }

    #[tokio::test]
    async fn list_pagination_and_config_snapshot() {
        let (temp, app) = app();
        for name in ["first", "second", "third"] {
            let (status, _) = request(app.clone(), "POST", "/prompts", Some(json!({"name": name, "content": "text"}))).await;
            assert_eq!(status, StatusCode::OK);
        }
        let (status, page) = request(app.clone(), "GET", "/prompts?limit=2", None).await;
        assert_eq!(status, StatusCode::OK);
        assert_eq!(page["data"].as_array().unwrap().len(), 2);
        assert_eq!(page["has_more"], true);
        let (_, next) = request(app.clone(), "GET", &format!("/prompts?after={}", page["last_id"].as_str().unwrap()), None).await;
        assert_eq!(next["data"].as_array().unwrap().len(), 1);
        let (_, previous) = request(app.clone(), "GET", &format!("/prompts?before={}", next["last_id"].as_str().unwrap()), None).await;
        assert_eq!(previous["data"].as_array().unwrap().len(), 2);
        for (uri, field) in [("/prompts?limit=0", "limit"), ("/prompts?limit=bad", "limit"), ("/prompts?after=missing", "after")] {
            let (status, body) = request(app.clone(), "GET", uri, None).await;
            assert_eq!(status, StatusCode::BAD_REQUEST);
            assert!(body["error"]["message"].as_str().unwrap().contains(field));
        }
        let (_, config) = request(app.clone(), "GET", "/config", None).await;
        assert_eq!(config["database_path"], temp.path().join("prompts.db").to_string_lossy().as_ref());
        let (status, body) = request(app, "PUT", "/config", Some(json!({"database_path": ""}))).await;
        assert_eq!(status, StatusCode::BAD_REQUEST);
        assert!(body["error"]["message"].as_str().unwrap().contains("database_path"));
    }

    #[tokio::test]
    async fn store_infrastructure_errors_are_503() {
        let response = store_error_response(StoreError::Lock);
        assert_eq!(response.status(), StatusCode::SERVICE_UNAVAILABLE);
        let bytes = to_bytes(response.into_body(), 64 * 1024).await.unwrap();
        let body: Value = serde_json::from_slice(&bytes).unwrap();
        assert_eq!(body["error"]["type"], "service_unavailable");
    }
}
