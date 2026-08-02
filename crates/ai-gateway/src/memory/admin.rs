//! Administrative HTTP API for persistent memory.
//!
//! Paths are relative to the parent `/admin/memory` nest point. Authentication
//! is intentionally left to that parent router.

use std::sync::Arc;

use axum::{
    extract::{rejection::JsonRejection, Path, Query, State},
    http::StatusCode,
    response::{IntoResponse, Response},
    routing::get,
    Json, Router,
};
use serde::Deserialize;
use serde_json::{json, Value};
use tokio::sync::RwLock;
use uuid::Uuid;

use super::{validate_namespace, AdminCreateError, MemoryError, MemorySystem, MemoryType};

const DEFAULT_LIMIT: usize = 50;
const MAX_LIMIT: usize = 200;

type OptionalMemorySystem = Arc<RwLock<Option<Arc<MemorySystem>>>>;

#[derive(Clone)]
pub enum MemoryAdminState {
    Enabled(Arc<MemorySystem>),
    Optional(OptionalMemorySystem),
}

impl From<Arc<MemorySystem>> for MemoryAdminState {
    fn from(system: Arc<MemorySystem>) -> Self {
        Self::Enabled(system)
    }
}

impl From<OptionalMemorySystem> for MemoryAdminState {
    fn from(system: OptionalMemorySystem) -> Self {
        Self::Optional(system)
    }
}

pub fn routes<S>(state: impl Into<MemoryAdminState>) -> Router<S>
where
    S: Clone + Send + Sync + 'static,
{
    Router::new()
        .route("/entries", get(list_entries).post(create_entry))
        .route("/entries/{id}", axum::routing::delete(delete_entry))
        .route(
            "/namespaces/{namespace}",
            axum::routing::delete(clear_namespace),
        )
        .route("/stats", get(stats))
        .route("/projects", get(projects))
        .with_state(state.into())
}

#[derive(Debug, Deserialize)]
struct ListQuery {
    namespace: Option<String>,
    #[serde(default = "default_limit")]
    limit: usize,
    #[serde(default)]
    offset: usize,
}

fn default_limit() -> usize {
    DEFAULT_LIMIT
}

#[derive(Debug, Deserialize)]
#[serde(deny_unknown_fields)]
struct CreateEntry {
    namespace: String,
    content: String,
    memory_type: MemoryType,
}

async fn resolve_system(state: &MemoryAdminState) -> Result<Arc<MemorySystem>, Response> {
    match state {
        MemoryAdminState::Enabled(system) => Ok(system.clone()),
        MemoryAdminState::Optional(handle) => handle.read().await.clone().ok_or_else(|| {
            error_response(
                StatusCode::SERVICE_UNAVAILABLE,
                "memory system is disabled",
                "service_unavailable",
            )
        }),
    }
}

async fn list_entries(
    State(state): State<MemoryAdminState>,
    Query(query): Query<ListQuery>,
) -> Response {
    let Some(namespace) = query.namespace else {
        return bad_request("namespace is required");
    };
    if !validate_namespace(&namespace) {
        return bad_request("namespace is invalid");
    }
    let system = match resolve_system(&state).await {
        Ok(system) => system,
        Err(response) => return response,
    };
    match system
        .store
        .list_entries(&namespace, query.limit.min(MAX_LIMIT), query.offset)
    {
        Ok(page) => Json(json!({
            "entries": page.entries,
            "total_count": page.total_count,
            "limit": page.limit,
            "offset": page.offset,
        }))
        .into_response(),
        Err(error) => memory_error(error),
    }
}

async fn create_entry(
    State(state): State<MemoryAdminState>,
    payload: Result<Json<CreateEntry>, JsonRejection>,
) -> Response {
    let Json(payload) = match payload {
        Ok(payload) => payload,
        Err(error) => return bad_request(&error.body_text()),
    };
    let system = match resolve_system(&state).await {
        Ok(system) => system,
        Err(response) => return response,
    };
    match system
        .admin_create(payload.namespace, payload.content, payload.memory_type)
        .await
    {
        Ok(entry) => (StatusCode::CREATED, Json(entry)).into_response(),
        Err(AdminCreateError::InvalidNamespace) => bad_request("namespace is invalid"),
        Err(AdminCreateError::InvalidContent(message)) => bad_request(&message),
        Err(AdminCreateError::SensitiveContent) => error_response(
            StatusCode::BAD_REQUEST,
            "content contains sensitive information",
            "sensitive_content",
        ),
        Err(AdminCreateError::Scan(message)) => {
            error_response(StatusCode::INTERNAL_SERVER_ERROR, &message, "server_error")
        }
        Err(AdminCreateError::Memory(error)) => memory_error(error),
    }
}

async fn delete_entry(State(state): State<MemoryAdminState>, Path(id): Path<String>) -> Response {
    let id = match Uuid::parse_str(&id) {
        Ok(id) => id,
        Err(_) => return bad_request("entry id must be a UUID"),
    };
    let system = match resolve_system(&state).await {
        Ok(system) => system,
        Err(response) => return response,
    };
    match system.store.delete_entry(id) {
        Ok(true) => StatusCode::NO_CONTENT.into_response(),
        Ok(false) => error_response(StatusCode::NOT_FOUND, "memory entry not found", "not_found"),
        Err(error) => memory_error(error),
    }
}

async fn clear_namespace(
    State(state): State<MemoryAdminState>,
    Path(namespace): Path<String>,
) -> Response {
    if !validate_namespace(&namespace) {
        return bad_request("namespace is invalid");
    }
    let system = match resolve_system(&state).await {
        Ok(system) => system,
        Err(response) => return response,
    };
    match system.store.delete_namespace(&namespace) {
        Ok(deleted_count) => Json(json!({ "deleted_count": deleted_count })).into_response(),
        Err(error) => memory_error(error),
    }
}

async fn stats(State(state): State<MemoryAdminState>) -> Response {
    let system = match resolve_system(&state).await {
        Ok(system) => system,
        Err(response) => return response,
    };
    match system.store.stats() {
        Ok(stats) => Json(json!({
            "total_count": stats.total_count,
            "memories_per_namespace": stats.memories_per_namespace,
            "average_relevance_score": stats.average_relevance_score,
            "storage_size_bytes": stats.storage_size_bytes,
            "last_decay_cycle": stats.last_decay_cycle,
        }))
        .into_response(),
        Err(error) => memory_error(error),
    }
}

async fn projects(State(state): State<MemoryAdminState>) -> Response {
    let system = match resolve_system(&state).await {
        Ok(system) => system,
        Err(response) => return response,
    };
    match system.store.list_project_namespaces() {
        Ok(projects) => Json(Value::Array(
            projects
                .into_iter()
                .map(|project| {
                    json!({
                        "namespace": project.namespace,
                        "entry_count": project.entry_count,
                        "last_activity": project.last_activity,
                    })
                })
                .collect(),
        ))
        .into_response(),
        Err(error) => memory_error(error),
    }
}

fn bad_request(message: &str) -> Response {
    error_response(StatusCode::BAD_REQUEST, message, "invalid_request")
}

fn memory_error(error: MemoryError) -> Response {
    error_response(
        StatusCode::INTERNAL_SERVER_ERROR,
        &error.to_string(),
        "server_error",
    )
}

fn error_response(status: StatusCode, message: &str, error_type: &str) -> Response {
    (
        status,
        Json(json!({ "error": { "message": message, "type": error_type } })),
    )
        .into_response()
}

#[cfg(test)]
mod tests {
    use axum::{
        body::{to_bytes, Body},
        http::Request,
    };
    use serde_json::Value;
    use tower::ServiceExt;

    use super::*;
    use crate::memory::MemoryConfig;

    async fn system() -> Arc<MemorySystem> {
        let config = MemoryConfig {
            database_path: ":memory:".to_owned(),
            ..MemoryConfig::default()
        };
        Arc::new(MemorySystem::new(config, None, None).await.unwrap())
    }

    async fn request(
        app: Router,
        method: &str,
        uri: &str,
        body: Option<Value>,
    ) -> (StatusCode, Value) {
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
        let bytes = to_bytes(response.into_body(), 64 * 1024).await.unwrap();
        let value = if bytes.is_empty() {
            Value::Null
        } else {
            serde_json::from_slice(&bytes).unwrap()
        };
        (status, value)
    }

    fn entry(namespace: &str, content: &str, memory_type: &str) -> Value {
        json!({
            "namespace": namespace,
            "content": content,
            "memory_type": memory_type,
        })
    }

    #[tokio::test]
    async fn create_list_and_page_entries() {
        let app = routes(system().await);
        let (status, created) = request(
            app.clone(),
            "POST",
            "/entries",
            Some(entry("user::alpha", "first useful fact", "fact")),
        )
        .await;
        assert_eq!(status, StatusCode::CREATED);
        assert_eq!(created["relevance_score"], 1.0);

        request(
            app.clone(),
            "POST",
            "/entries",
            Some(entry("user::alpha", "second useful fact", "context")),
        )
        .await;
        let (status, page) = request(
            app,
            "GET",
            "/entries?namespace=user%3A%3Aalpha&limit=1&offset=1",
            None,
        )
        .await;
        assert_eq!(status, StatusCode::OK);
        assert_eq!(page["total_count"], 2);
        assert_eq!(page["limit"], 1);
        assert_eq!(page["offset"], 1);
        assert_eq!(page["entries"].as_array().unwrap().len(), 1);
    }

    #[tokio::test]
    async fn delete_is_exact_and_reports_not_found() {
        let app = routes(system().await);
        let (_, created) = request(
            app.clone(),
            "POST",
            "/entries",
            Some(entry("user::delete", "delete this fact", "fact")),
        )
        .await;
        let uri = format!("/entries/{}", created["id"].as_str().unwrap());
        let (status, body) = request(app.clone(), "DELETE", &uri, None).await;
        assert_eq!(status, StatusCode::NO_CONTENT);
        assert!(body.is_null());
        let (status, body) = request(app, "DELETE", &uri, None).await;
        assert_eq!(status, StatusCode::NOT_FOUND);
        assert_eq!(body["error"]["type"], "not_found");
    }

    #[tokio::test]
    async fn clear_uses_percent_decoded_exact_namespace_and_rejects_wildcards() {
        let app = routes(system().await);
        for namespace in ["user::one", "user::one-extra"] {
            request(
                app.clone(),
                "POST",
                "/entries",
                Some(entry(namespace, "namespace test fact", "fact")),
            )
            .await;
        }
        let (status, body) =
            request(app.clone(), "DELETE", "/namespaces/user%3A%3Aone", None).await;
        assert_eq!(status, StatusCode::OK);
        assert_eq!(body["deleted_count"], 1);
        let (_, page) = request(
            app.clone(),
            "GET",
            "/entries?namespace=user%3A%3Aone-extra",
            None,
        )
        .await;
        assert_eq!(page["total_count"], 1);
        let (status, _) = request(app, "DELETE", "/namespaces/user%3A%3A%2A", None).await;
        assert_eq!(status, StatusCode::BAD_REQUEST);
    }

    #[tokio::test]
    async fn stats_and_projects_reflect_stored_entries() {
        let app = routes(system().await);
        request(
            app.clone(),
            "POST",
            "/entries",
            Some(entry(
                "user::alpha::project::abc123",
                "project decision retained",
                "decision",
            )),
        )
        .await;
        let (status, stats) = request(app.clone(), "GET", "/stats", None).await;
        assert_eq!(status, StatusCode::OK);
        assert_eq!(stats["total_count"], 1);
        assert_eq!(
            stats["memories_per_namespace"]["user::alpha::project::abc123"],
            1
        );
        let (status, projects) = request(app, "GET", "/projects", None).await;
        assert_eq!(status, StatusCode::OK);
        assert_eq!(projects.as_array().unwrap().len(), 1);
        assert_eq!(projects[0]["entry_count"], 1);
    }

    #[tokio::test]
    async fn invalid_requests_are_json_errors() {
        let app = routes(system().await);
        let (status, body) = request(app.clone(), "GET", "/entries", None).await;
        assert_eq!(status, StatusCode::BAD_REQUEST);
        assert_eq!(body["error"]["type"], "invalid_request");
        for invalid in [
            entry("user::*", "valid content", "fact"),
            entry("user::alpha", "tiny", "fact"),
            entry("user::alpha", "valid content", "unknown"),
        ] {
            let (status, body) = request(app.clone(), "POST", "/entries", Some(invalid)).await;
            assert_eq!(status, StatusCode::BAD_REQUEST);
            assert!(body["error"]["message"].is_string());
        }
    }

    #[tokio::test]
    async fn sensitive_content_is_rejected_before_storage() {
        let app = routes(system().await);
        let (status, body) = request(
            app.clone(),
            "POST",
            "/entries",
            Some(entry(
                "user::secure",
                "secret sk-abcdefghijklmnopqrstuvwxyz123456",
                "fact",
            )),
        )
        .await;
        assert_eq!(status, StatusCode::BAD_REQUEST);
        assert_eq!(body["error"]["type"], "sensitive_content");
        let (_, page) = request(app, "GET", "/entries?namespace=user%3A%3Asecure", None).await;
        assert_eq!(page["total_count"], 0);
    }

    #[tokio::test]
    async fn optional_disabled_state_returns_json_503() {
        let handle: OptionalMemorySystem = Arc::new(RwLock::new(None));
        let app = routes(handle);
        for (method, uri) in [
            ("GET", "/entries?namespace=user%3A%3Aalpha"),
            ("GET", "/stats"),
            ("GET", "/projects"),
        ] {
            let (status, body) = request(app.clone(), method, uri, None).await;
            assert_eq!(status, StatusCode::SERVICE_UNAVAILABLE);
            assert_eq!(body["error"]["type"], "service_unavailable");
        }
    }
}
