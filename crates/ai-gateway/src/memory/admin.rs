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
        .route(
            "/entries/{id}",
            axum::routing::delete(delete_entry).patch(edit_entry),
        )
        .route(
            "/entries/{id}/supersede",
            axum::routing::post(supersede_entry),
        )
        .route("/search", get(search))
        .route("/export", get(export))
        .route("/consolidate", axum::routing::post(consolidate))
        .route(
            "/namespaces/{namespace}",
            axum::routing::delete(clear_namespace),
        )
        .route("/stats", get(stats))
        .route("/namespaces", get(namespaces))
        .route("/projects", get(projects))
        .route(
            "/qdrant/collection",
            axum::routing::delete(recreate_qdrant_collection),
        )
        .with_state(state.into())
}

#[derive(Debug, Deserialize)]
struct ListQuery {
    namespace: Option<String>,
    #[serde(default)]
    include_superseded: bool,
    #[serde(default = "default_limit")]
    limit: usize,
    #[serde(default)]
    offset: usize,
}

#[derive(Debug, Deserialize)]
struct SearchQuery {
    q: Option<String>,
    namespace: Option<String>,
    memory_type: Option<String>,
    #[serde(default)]
    include_superseded: bool,
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

#[derive(Debug, Deserialize)]
#[serde(deny_unknown_fields)]
struct EditEntry {
    content: String,
    memory_type: MemoryType,
}

#[derive(Debug, Deserialize)]
#[serde(deny_unknown_fields)]
struct SupersedeEntry {
    superseded_by: String,
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
    match system.store.list_entries_with_superseded(
        &namespace,
        query.limit.min(MAX_LIMIT),
        query.offset,
        query.include_superseded,
    ) {
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

/// Search memory content across namespaces through the FTS5 index (Req 4.1).
///
/// An empty or whitespace-only `q` falls back to the default filtered listing
/// inside [`MemoryStore::search_entries`], so it always answers 200. Filters
/// are validated here for actionable 400 responses before reaching the store,
/// where they would otherwise surface as configuration errors.
async fn search(State(state): State<MemoryAdminState>, Query(query): Query<SearchQuery>) -> Response {
    if let Some(namespace) = &query.namespace {
        if !validate_namespace(namespace) {
            return bad_request("namespace is invalid");
        }
    }
    if let Some(memory_type) = &query.memory_type {
        if !is_valid_memory_type(memory_type) {
            return bad_request("memory_type must be one of preference, fact, context, decision");
        }
    }
    let system = match resolve_system(&state).await {
        Ok(system) => system,
        Err(response) => return response,
    };
    let query_text = query.q.as_deref().map(str::trim).unwrap_or("");
    match system.store.search_entries(
        query_text,
        query.namespace.as_deref(),
        query.memory_type.as_deref(),
        query.include_superseded,
        query.limit.min(MAX_LIMIT),
        query.offset,
    ) {
        Ok(page) => Json(json!({
            "entries": page.entries,
            "total_count": page.total_count,
            "limit": page.limit,
            "offset": page.offset,
        }))
        .into_response(),
        // Sanitized filter failures (unreachable behind the pre-validation
        // above) still answer 400 actionable rather than 500.
        Err(MemoryError::Config(message)) => bad_request(&message),
        Err(error) => memory_error(error),
    }
}

fn is_valid_memory_type(memory_type: &str) -> bool {
    matches!(
        memory_type,
        "preference" | "fact" | "context" | "decision"
    )
}

fn memory_type_label(memory_type: MemoryType) -> &'static str {
    match memory_type {
        MemoryType::Preference => "preference",
        MemoryType::Fact => "fact",
        MemoryType::Context => "context",
        MemoryType::Decision => "decision",
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

/// Edit one stored entry in place (Req 1.1-1.4).
///
/// Validation and sensitive screening run inside [`MemorySystem::admin_edit`];
/// a missing id maps to 404 and actionable rejections to 400, matching the
/// create-entry error mapping.
async fn edit_entry(
    State(state): State<MemoryAdminState>,
    Path(id): Path<String>,
    payload: Result<Json<EditEntry>, JsonRejection>,
) -> Response {
    let id = match Uuid::parse_str(&id) {
        Ok(id) => id,
        Err(_) => return bad_request("entry id must be a UUID"),
    };
    let Json(payload) = match payload {
        Ok(payload) => payload,
        Err(error) => return bad_request(&error.body_text()),
    };
    let system = match resolve_system(&state).await {
        Ok(system) => system,
        Err(response) => return response,
    };
    match system
        .admin_edit(id, &payload.content, memory_type_label(payload.memory_type))
        .await
    {
        Ok(Some(entry)) => Json(entry).into_response(),
        Ok(None) => error_response(StatusCode::NOT_FOUND, "memory entry not found", "not_found"),
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

/// Mark one entry as superseded by another (Req 2.1, 2.3).
///
/// Both ids must exist and the target must not already be superseded; a
/// missing id maps to 404 and an already-superseded target to 409, following
/// the conflict-status convention used by the virtual-keys admin API.
async fn supersede_entry(
    State(state): State<MemoryAdminState>,
    Path(id): Path<String>,
    payload: Result<Json<SupersedeEntry>, JsonRejection>,
) -> Response {
    let id = match Uuid::parse_str(&id) {
        Ok(id) => id,
        Err(_) => return bad_request("entry id must be a UUID"),
    };
    let Json(payload) = match payload {
        Ok(payload) => payload,
        Err(error) => return bad_request(&error.body_text()),
    };
    let superseded_by = match Uuid::parse_str(&payload.superseded_by) {
        Ok(id) => id,
        Err(_) => return bad_request("superseded_by must be a UUID"),
    };
    if superseded_by == id {
        return bad_request("an entry cannot supersede itself");
    }
    let system = match resolve_system(&state).await {
        Ok(system) => system,
        Err(response) => return response,
    };
    let target = match system.store.get_entry_by_id(id) {
        Ok(Some(entry)) => entry,
        Ok(None) => {
            return error_response(StatusCode::NOT_FOUND, "memory entry not found", "not_found")
        }
        Err(error) => return memory_error(error),
    };
    if target.superseded_by.is_some() {
        return error_response(
            StatusCode::CONFLICT,
            "memory entry is already superseded",
            "already_superseded",
        );
    }
    match system.store.get_entry_by_id(superseded_by) {
        Ok(Some(_)) => {}
        Ok(None) => {
            return error_response(
                StatusCode::NOT_FOUND,
                "superseded_by entry not found",
                "not_found",
            )
        }
        Err(error) => return memory_error(error),
    }
    match system.store.mark_superseded(&[id], superseded_by) {
        Ok(marked) => match system.store.get_entry_by_id(id) {
            Ok(entry) => Json(json!({
                "marked_count": marked,
                "entry": entry,
            }))
            .into_response(),
            Err(error) => memory_error(error),
        },
        Err(error) => memory_error(error),
    }
}

/// Export every stored entry, superseded ones included, as one JSON document
/// (Req 5.1-5.2). Annotations travel on each entry through `updated_at` and
/// `superseded_by`.
async fn export(State(state): State<MemoryAdminState>) -> Response {
    let system = match resolve_system(&state).await {
        Ok(system) => system,
        Err(response) => return response,
    };
    match system.store.export_all() {
        Ok(entries) => Json(json!({
            "total": entries.len(),
            "entries": entries,
        }))
        .into_response(),
        Err(error) => memory_error(error),
    }
}

/// Run one consolidation pass immediately and return its outcome counts
/// (Req 3.8).
///
/// Seam: a live pass needs the extraction provider and sensitive scanner
/// handles, which `MemorySystem` does not expose yet, and the dedicated
/// consolidation config fields are landing in a concurrent wave. Until the
/// provider is reachable, manual passes answer 503 `consolidation_unavailable`
/// — never a half-configured 500. The provider name resolves through the
/// Req 3.7 fallback chain (`consolidation_provider` once it exists, then
/// `auto_extract_provider`).
async fn consolidate(State(state): State<MemoryAdminState>) -> Response {
    let system = match resolve_system(&state).await {
        Ok(system) => system,
        Err(response) => return response,
    };
    let config = system.config.read().await;
    if consolidation_provider_name(&config).is_empty() {
        return error_response(
            StatusCode::SERVICE_UNAVAILABLE,
            "consolidation provider is not configured",
            "consolidation_unavailable",
        );
    }
    error_response(
        StatusCode::SERVICE_UNAVAILABLE,
        "consolidation is not available until the provider is wired into the memory system",
        "consolidation_unavailable",
    )
}

/// Resolve the consolidation provider name with the Req 3.7 fallback chain.
///
/// The dedicated `consolidation_provider` field does not exist yet in this
/// tree; swap it in here when the concurrent config wave lands.
fn consolidation_provider_name(config: &super::MemoryConfig) -> &str {
    &config.auto_extract_provider
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

async fn namespaces(State(state): State<MemoryAdminState>) -> Response {
    let system = match resolve_system(&state).await {
        Ok(system) => system,
        Err(response) => return response,
    };
    match system.store.list_namespaces() {
        Ok(namespaces) => Json(
            namespaces
                .into_iter()
                .map(|item| {
                    json!({
                        "namespace": item.namespace,
                        "context_kind": item.context_kind,
                        "display_name": item.display_name,
                        "client_name": item.client_name,
                        "entry_count": item.entry_count,
                        "last_activity": item.last_activity,
                    })
                })
                .collect::<Vec<_>>(),
        )
        .into_response(),
        Err(error) => memory_error(error),
    }
}

async fn projects(State(state): State<MemoryAdminState>) -> Response {
    let system = match resolve_system(&state).await {
        Ok(system) => system,
        Err(response) => return response,
    };
    match system.store.list_namespaces() {
        Ok(projects) => Json(Value::Array(
            projects
                .into_iter()
                .map(|project| {
                    json!({
                        "namespace": project.namespace,
                        "context_kind": project.context_kind,
                        "display_name": project.display_name,
                        "client_name": project.client_name,
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

async fn recreate_qdrant_collection(State(state): State<MemoryAdminState>) -> Response {
    let system = match resolve_system(&state).await {
        Ok(system) => system,
        Err(response) => return response,
    };
    let qdrant_config = match system.config.read().await.qdrant.clone() {
        Some(config) => config,
        None => {
            return error_response(
                StatusCode::BAD_REQUEST,
                "Qdrant is not configured in memory settings",
                "not_configured",
            )
        }
    };
    let qdrant_url = super::vector::normalize_qdrant_url(&qdrant_config.qdrant_url);
    let qdrant = match qdrant_client::Qdrant::from_url(&qdrant_url).build() {
        Ok(client) => client,
        Err(error) => {
            return error_response(
                StatusCode::INTERNAL_SERVER_ERROR,
                &format!("Failed to create Qdrant client: {error}"),
                "qdrant_client_error",
            );
        }
    };
    let collection = &qdrant_config.qdrant_collection;
    match qdrant.collection_exists(collection).await {
        Ok(exists) => {
            if exists {
                if let Err(error) = qdrant.delete_collection(collection).await {
                    return error_response(
                        StatusCode::INTERNAL_SERVER_ERROR,
                        &format!("Failed to delete Qdrant collection: {error}"),
                        "qdrant_delete_error",
                    );
                }
            }
        }
        Err(error) => {
            return error_response(
                StatusCode::INTERNAL_SERVER_ERROR,
                &format!("Failed to check Qdrant collection: {error}"),
                "qdrant_check_error",
            );
        }
    }
    let providers = system.config.read().await;
    let provider_config = system.config.read().await;
    drop(providers);
    let _ = provider_config;
    system.set_vector_tier(None).await;
    if let Err(error) = system.store.mark_all_vector_entries_pending() {
        tracing::warn!(error = %error, "Failed to mark vector entries pending after collection recreation");
    }
    (
StatusCode::OK,
Json(json!({
"message": format!("Qdrant collection '{}' recreated successfully. Reload config or restart gateway to reinitialize vector tier.", collection)
})),
)
.into_response()
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
            ("GET", "/search?q=anything"),
            ("GET", "/export"),
            ("POST", "/consolidate"),
        ] {
            let (status, body) = request(app.clone(), method, uri, None).await;
            assert_eq!(status, StatusCode::SERVICE_UNAVAILABLE);
            assert_eq!(body["error"]["type"], "service_unavailable");
        }
    }

    #[tokio::test]
    async fn edit_updates_entry_and_reports_errors() {
        let app = routes(system().await);
        let (_, created) = request(
            app.clone(),
            "POST",
            "/entries",
            Some(entry("user::edit", "original useful fact", "fact")),
        )
        .await;
        let uri = format!("/entries/{}", created["id"].as_str().unwrap());
        let (status, updated) = request(
            app.clone(),
            "PATCH",
            &uri,
            Some(json!({"content": "revised useful fact", "memory_type": "decision"})),
        )
        .await;
        assert_eq!(status, StatusCode::OK);
        assert_eq!(updated["content"], "revised useful fact");
        assert_eq!(updated["memory_type"], "decision");
        assert!(updated["updated_at"].is_i64());

        let (status, body) = request(
            app.clone(),
            "PATCH",
            &uri,
            Some(json!({"content": "tiny", "memory_type": "fact"})),
        )
        .await;
        assert_eq!(status, StatusCode::BAD_REQUEST);
        assert!(body["error"]["message"].is_string());

        let missing = format!("/entries/{}", Uuid::new_v4());
        let (status, body) = request(
            app,
            "PATCH",
            &missing,
            Some(json!({"content": "valid content", "memory_type": "fact"})),
        )
        .await;
        assert_eq!(status, StatusCode::NOT_FOUND);
        assert_eq!(body["error"]["type"], "not_found");
    }

    #[tokio::test]
    async fn search_matches_filters_and_falls_back_on_empty_query() {
        let app = routes(system().await);
        request(
            app.clone(),
            "POST",
            "/entries",
            Some(entry("user::one", "kubernetes cluster fact", "fact")),
        )
        .await;
        request(
            app.clone(),
            "POST",
            "/entries",
            Some(entry("user::two", "unrelated preference text", "preference")),
        )
        .await;
        let (status, page) = request(app.clone(), "GET", "/search?q=kubernetes", None).await;
        assert_eq!(status, StatusCode::OK);
        assert_eq!(page["total_count"], 1);
        assert_eq!(page["entries"].as_array().unwrap().len(), 1);
        assert!(page["entries"][0]["content"]
            .as_str()
            .unwrap()
            .contains("kubernetes"));

        let (status, page) = request(app.clone(), "GET", "/search?q=", None).await;
        assert_eq!(status, StatusCode::OK);
        assert_eq!(page["total_count"], 2);

        let (status, page) = request(
            app.clone(),
            "GET",
            "/search?q=&memory_type=preference",
            None,
        )
        .await;
        assert_eq!(status, StatusCode::OK);
        assert_eq!(page["total_count"], 1);

        let (status, body) = request(app.clone(), "GET", "/search?q=x&memory_type=bogus", None).await;
        assert_eq!(status, StatusCode::BAD_REQUEST);
        assert_eq!(body["error"]["type"], "invalid_request");

        let (status, _) = request(app.clone(), "GET", "/search?q=x&namespace=user%3A%3A%2A", None).await;
        assert_eq!(status, StatusCode::BAD_REQUEST);
    }

    #[tokio::test]
    async fn export_returns_single_document_with_total() {
        let app = routes(system().await);
        request(
            app.clone(),
            "POST",
            "/entries",
            Some(entry("user::beta", "beta export content", "fact")),
        )
        .await;
        request(
            app.clone(),
            "POST",
            "/entries",
            Some(entry("user::alpha", "alpha export content", "fact")),
        )
        .await;
        let (status, document) = request(app, "GET", "/export", None).await;
        assert_eq!(status, StatusCode::OK);
        assert_eq!(document["total"], 2);
        assert_eq!(document["entries"].as_array().unwrap().len(), 2);
    }

    #[tokio::test]
    async fn supersede_marks_entry_and_validates_ids() {
        let app = routes(system().await);
        let (_, target) = request(
            app.clone(),
            "POST",
            "/entries",
            Some(entry("user::supersede", "outdated fact content", "fact")),
        )
        .await;
        let (_, replacement) = request(
            app.clone(),
            "POST",
            "/entries",
            Some(entry("user::supersede", "replacement fact content", "fact")),
        )
        .await;
        let uri = format!("/entries/{}/supersede", target["id"].as_str().unwrap());
        let (status, body) = request(
            app.clone(),
            "POST",
            &uri,
            Some(json!({"superseded_by": replacement["id"].as_str().unwrap()})),
        )
        .await;
        assert_eq!(status, StatusCode::OK);
        assert_eq!(body["marked_count"], 1);
        assert_eq!(body["entry"]["superseded_by"], replacement["id"]);

        let (status, body) = request(
            app.clone(),
            "POST",
            &uri,
            Some(json!({"superseded_by": replacement["id"].as_str().unwrap()})),
        )
        .await;
        assert_eq!(status, StatusCode::CONFLICT);
        assert_eq!(body["error"]["type"], "already_superseded");

        let missing = format!("/entries/{}/supersede", Uuid::new_v4());
        let (status, body) = request(
            app.clone(),
            "POST",
            &missing,
            Some(json!({"superseded_by": replacement["id"].as_str().unwrap()})),
        )
        .await;
        assert_eq!(status, StatusCode::NOT_FOUND);
        assert_eq!(body["error"]["type"], "not_found");

        let missing_by = format!(
            "/entries/{}/supersede",
            replacement["id"].as_str().unwrap()
        );
        let (status, _) = request(
            app,
            "POST",
            &missing_by,
            Some(json!({"superseded_by": Uuid::new_v4().to_string()})),
        )
        .await;
        assert_eq!(status, StatusCode::NOT_FOUND);
    }

    #[tokio::test]
    async fn list_includes_superseded_only_when_requested() {
        let app = routes(system().await);
        let (_, target) = request(
            app.clone(),
            "POST",
            "/entries",
            Some(entry("user::history", "first history fact", "fact")),
        )
        .await;
        let (_, replacement) = request(
            app.clone(),
            "POST",
            "/entries",
            Some(entry("user::history", "second history fact", "fact")),
        )
        .await;
        request(
            app.clone(),
            "POST",
            &format!("/entries/{}/supersede", target["id"].as_str().unwrap()),
            Some(json!({"superseded_by": replacement["id"].as_str().unwrap()})),
        )
        .await;
        let (status, page) = request(
            app.clone(),
            "GET",
            "/entries?namespace=user%3A%3Ahistory",
            None,
        )
        .await;
        assert_eq!(status, StatusCode::OK);
        assert_eq!(page["total_count"], 1);

        let (status, page) = request(
            app,
            "GET",
            "/entries?namespace=user%3A%3Ahistory&include_superseded=true",
            None,
        )
        .await;
        assert_eq!(status, StatusCode::OK);
        assert_eq!(page["total_count"], 2);
        assert!(page["entries"].as_array().unwrap().iter().any(|entry| {
            entry["superseded_by"].is_string()
        }));
    }

    #[tokio::test]
    async fn consolidate_reports_unavailable_until_provider_wired() {
        let app = routes(system().await);
        let (status, body) = request(app.clone(), "POST", "/consolidate", None).await;
        assert_eq!(status, StatusCode::SERVICE_UNAVAILABLE);
        assert_eq!(body["error"]["type"], "consolidation_unavailable");

        let config = MemoryConfig {
            database_path: ":memory:".to_owned(),
            auto_extract_provider: "test-provider".to_owned(),
            ..MemoryConfig::default()
        };
        let configured =
            Arc::new(MemorySystem::new(config, None, None).await.unwrap());
        let app = routes(configured);
        let (status, body) = request(app, "POST", "/consolidate", None).await;
        assert_eq!(status, StatusCode::SERVICE_UNAVAILABLE);
        assert_eq!(body["error"]["type"], "consolidation_unavailable");
    }
}
