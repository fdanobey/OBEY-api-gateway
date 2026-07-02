//! Admin API route handlers for virtual key management (Req 10.1-10.6).
//!
//! This module exposes the RESTful CRUD surface for virtual keys as an Axum
//! [`Router`] built by [`routes`]. Handlers are thin: they extract JSON bodies
//! and query parameters, delegate to [`VirtualKeyManager`], and return the
//! manager's result as JSON. All error mapping is centralized in the
//! `IntoResponse` implementation for [`KeyError`] (see `errors.rs`), which
//! renders the design's "HTTP Error Mapping" table (404/409/400/500).
//!
//! ## Endpoints (paths shown relative to the nest point)
//!
//! | Method | Path            | Handler              | Success |
//! |--------|-----------------|----------------------|---------|
//! | POST   | `/`             | [`create_key`]       | 201     |
//! | GET    | `/`             | [`list_keys`]        | 200     |
//! | GET    | `/{id}`         | [`get_key`]          | 200     |
//! | PATCH  | `/{id}`         | [`update_key`]       | 200     |
//! | DELETE | `/{id}`         | [`delete_key`]       | 204     |
//! | POST   | `/{id}/revoke`  | [`revoke_key`]       | 200     |
//! | GET    | `/{id}/usage`   | [`query_usage`]      | 200     |
//!
//! The router is state-complete (`Router` with no outstanding state): it owns an
//! `Arc<VirtualKeyManager>` via [`Router::with_state`]. Task 13.1 nests it under
//! the admin panel, e.g. `.nest("/admin/keys", virtual_keys::admin::routes(mgr))`,
//! which yields the design's public paths (`/admin/keys`, `/admin/keys/{id}`,
//! `/admin/keys/{id}/revoke`, `/admin/keys/{id}/usage`).
//!
//! ## Authentication
//!
//! These handlers do NOT apply admin authentication themselves. The existing
//! admin auth mechanism (HTTP Basic Auth with a `WWW-Authenticate` challenge on
//! 401, per Req 10.2) is applied by the nesting task (13.1) via the existing
//! admin middleware layer, so this module stays decoupled from `AppState`.

use std::sync::Arc;

use axum::{
    extract::{Path, Query, State},
    http::StatusCode,
    response::{IntoResponse, Response},
    routing::{get, post},
    Json, Router,
};
use serde::Deserialize;

use super::models::{
    CreateKeyParams, ListKeysParams, UpdateKeyParams, UsageQueryParams,
};
use super::{KeyError, VirtualKeyManager};

/// Shared handler state: the virtual key manager behind an `Arc`.
type ManagerState = State<Arc<VirtualKeyManager>>;

/// Build the virtual key admin API router with its state pre-applied.
///
/// The returned router is self-contained (no outstanding state parameter) so
/// the wiring task (13.1) can nest it directly under the admin panel without
/// threading `AppState`:
///
/// ```ignore
/// admin_router.nest("/admin/keys", virtual_keys::admin::routes(manager));
/// ```
///
/// Admin authentication (Req 10.2) is layered on by the nesting task via the
/// existing admin middleware, not here.
///
/// _Requirements: 10.1, 10.5, 10.6_
pub fn routes(manager: Arc<VirtualKeyManager>) -> Router {
    Router::new()
        .route("/", post(create_key).get(list_keys))
        .route(
            "/{id}",
            get(get_key).patch(update_key).delete(delete_key),
        )
        .route("/{id}/revoke", post(revoke_key))
        .route("/{id}/usage", get(query_usage))
        .with_state(manager)
}

/// Query parameters for `GET /` (list keys).
///
/// `limit` defaults to 50 and is capped at 200 (Req 10.5); `offset` defaults
/// to 0. Values are clamped at the API boundary before delegating to the
/// manager.
#[derive(Debug, Deserialize)]
struct ListQuery {
    #[serde(default = "default_list_limit")]
    limit: u32,
    #[serde(default)]
    offset: u32,
}

/// Default page size for the list endpoint (Req 10.5).
fn default_list_limit() -> u32 {
    50
}

/// Maximum page size accepted by the list endpoint (Req 10.5).
const LIST_MAX_LIMIT: u32 = 200;

/// `POST /` — generate a new virtual key.
///
/// Returns HTTP 201 with the [`CreateKeyResponse`](super::models::CreateKeyResponse),
/// which includes the plaintext key value exactly once (Req 10.1, 10.6).
async fn create_key(
    State(manager): ManagerState,
    Json(params): Json<CreateKeyParams>,
) -> Result<Response, KeyError> {
    let created = manager.create_key(params).await?;
    Ok((StatusCode::CREATED, Json(created)).into_response())
}

/// `GET /` — list keys with pagination.
///
/// Supports `limit` (default 50, max 200) and `offset` (default 0) query
/// parameters and returns HTTP 200 with a
/// [`PaginatedKeys`](super::models::PaginatedKeys) payload including the total
/// count (Req 10.5).
async fn list_keys(
    State(manager): ManagerState,
    Query(query): Query<ListQuery>,
) -> Result<Response, KeyError> {
    let params = ListKeysParams {
        limit: query.limit.min(LIST_MAX_LIMIT),
        offset: query.offset,
    };
    let page = manager.list_keys(params).await?;
    Ok((StatusCode::OK, Json(page)).into_response())
}

/// `GET /{id}` — fetch a single key's masked info (Req 10.1).
async fn get_key(
    State(manager): ManagerState,
    Path(id): Path<String>,
) -> Result<Response, KeyError> {
    let info = manager.get_key(&id).await?;
    Ok((StatusCode::OK, Json(info)).into_response())
}

/// `PATCH /{id}` — apply a partial update and return the updated resource.
///
/// Returns HTTP 200 with the updated key info (Req 10.6). Revoked keys yield
/// 409 and out-of-range fields yield a structured 400 via [`KeyError`].
async fn update_key(
    State(manager): ManagerState,
    Path(id): Path<String>,
    Json(params): Json<UpdateKeyParams>,
) -> Result<Response, KeyError> {
    let info = manager.update_key(&id, params).await?;
    Ok((StatusCode::OK, Json(info)).into_response())
}

/// `DELETE /{id}` — delete a key and its usage history; returns HTTP 204.
async fn delete_key(
    State(manager): ManagerState,
    Path(id): Path<String>,
) -> Result<Response, KeyError> {
    manager.delete_key(&id).await?;
    Ok(StatusCode::NO_CONTENT.into_response())
}

/// `POST /{id}/revoke` — revoke a key and return its updated info (Req 10.6).
async fn revoke_key(
    State(manager): ManagerState,
    Path(id): Path<String>,
) -> Result<Response, KeyError> {
    let info = manager.revoke_key(&id).await?;
    Ok((StatusCode::OK, Json(info)).into_response())
}

/// `GET /{id}/usage` — aggregate a key's usage over an inclusive time range.
///
/// `start` and `end` are RFC 3339 timestamps supplied as query parameters
/// (e.g. `?start=2024-01-01T00:00:00Z&end=2024-01-31T23:59:59Z`) and are parsed
/// into [`DateTime<Utc>`](chrono::DateTime) via the
/// [`UsageQueryParams`](super::models::UsageQueryParams) deserialization.
/// Returns HTTP 200 with a [`UsageAggregate`](super::models::UsageAggregate);
/// an unknown key id yields 404 (Req 9.3).
async fn query_usage(
    State(manager): ManagerState,
    Path(id): Path<String>,
    Query(params): Query<UsageQueryParams>,
) -> Result<Response, KeyError> {
    let aggregate = manager.query_usage(&id, params).await?;
    Ok((StatusCode::OK, Json(aggregate)).into_response())
}

#[cfg(test)]
mod tests {
    //! Unit tests for the admin API endpoints (Req 10.1-10.5).
    //!
    //! These exercise the router built by [`routes`] end-to-end via
    //! `tower::ServiceExt::oneshot` (no port binding), backed by a real
    //! [`VirtualKeyManager`] over a temporary SQLite database. Response bodies
    //! are read with [`axum::body::to_bytes`] and parsed as JSON.

    use super::*;

    use axum::body::Body;
    use axum::http::{Request, StatusCode};
    use serde_json::{json, Value};
    use tempfile::NamedTempFile;
    use tower::ServiceExt;

    /// Build a manager over a fresh temp DB. The `NamedTempFile` is returned so
    /// the caller keeps it alive for the test's duration (dropping it removes
    /// the backing file).
    fn manager() -> (Arc<VirtualKeyManager>, NamedTempFile) {
        let tmp = NamedTempFile::new().unwrap();
        let mgr = Arc::new(VirtualKeyManager::new(tmp.path()).unwrap());
        (mgr, tmp)
    }

    /// Construct a fresh router sharing the given manager. `oneshot` consumes a
    /// router per request, so each call rebuilds one over the same `Arc`.
    fn app(mgr: &Arc<VirtualKeyManager>) -> Router {
        routes(Arc::clone(mgr))
    }

    /// Send one request through the router and return the status plus the JSON
    /// body (or `Value::Null` for empty bodies such as 204 responses).
    async fn send(
        router: Router,
        method: &str,
        uri: &str,
        body: Option<Value>,
    ) -> (StatusCode, Value) {
        let builder = Request::builder().method(method).uri(uri);
        let request = match body {
            Some(b) => builder
                .header("content-type", "application/json")
                .body(Body::from(serde_json::to_vec(&b).unwrap()))
                .unwrap(),
            None => builder.body(Body::empty()).unwrap(),
        };

        let resp = router.oneshot(request).await.unwrap();
        let status = resp.status();
        let bytes = axum::body::to_bytes(resp.into_body(), usize::MAX)
            .await
            .unwrap();
        let value = if bytes.is_empty() {
            Value::Null
        } else {
            serde_json::from_slice(&bytes).unwrap()
        };
        (status, value)
    }

    /// Create a key via the API and return its `(id, key)` pair.
    async fn create_key(mgr: &Arc<VirtualKeyManager>, body: Value) -> (String, String) {
        let (status, resp) = send(app(mgr), "POST", "/", Some(body)).await;
        assert_eq!(status, StatusCode::CREATED);
        (
            resp["id"].as_str().unwrap().to_string(),
            resp["key"].as_str().unwrap().to_string(),
        )
    }

    /// Req 10.1/10.6: POST / returns 201 with a body carrying the plaintext key
    /// (shown once) and the new key's id.
    #[tokio::test]
    async fn post_create_returns_201_with_key_and_id() {
        let (mgr, _tmp) = manager();

        let (status, body) = send(app(&mgr), "POST", "/", Some(json!({ "name": "alpha" }))).await;

        assert_eq!(status, StatusCode::CREATED);
        let key = body["key"].as_str().expect("key present");
        assert!(key.starts_with("vk_"), "expected vk_ prefix, got {key:?}");
        assert!(body["id"].as_str().is_some(), "id present");
        assert_eq!(body["name"], json!("alpha"));
        assert_eq!(body["status"], json!("active"));
    }

    /// Req 10.5: GET / returns 200 with `total` and `keys`, and honours the
    /// `limit`/`offset` query parameters. A `limit=1` page returns a single key
    /// while `total` still reflects the full store count.
    #[tokio::test]
    async fn get_list_returns_200_with_pagination_metadata() {
        let (mgr, _tmp) = manager();
        create_key(&mgr, json!({ "name": "first" })).await;
        create_key(&mgr, json!({ "name": "second" })).await;

        // Full list: total and keys both reflect the two created keys.
        let (status, body) = send(app(&mgr), "GET", "/", None).await;
        assert_eq!(status, StatusCode::OK);
        assert_eq!(body["total"], json!(2));
        assert_eq!(body["keys"].as_array().unwrap().len(), 2);

        // limit=1 returns a single key; total still reports the full count.
        let (status, page) = send(app(&mgr), "GET", "/?limit=1&offset=0", None).await;
        assert_eq!(status, StatusCode::OK);
        assert_eq!(page["total"], json!(2));
        assert_eq!(page["keys"].as_array().unwrap().len(), 1);

        // offset past the first key returns the remaining key.
        let (status, page2) = send(app(&mgr), "GET", "/?limit=1&offset=1", None).await;
        assert_eq!(status, StatusCode::OK);
        assert_eq!(page2["total"], json!(2));
        assert_eq!(page2["keys"].as_array().unwrap().len(), 1);
    }

    /// Req 10.1: GET /{id} returns 200 with the masked info for an existing key.
    #[tokio::test]
    async fn get_by_id_returns_200_for_existing() {
        let (mgr, _tmp) = manager();
        let (id, key) = create_key(&mgr, json!({ "name": "lookup" })).await;

        let (status, body) = send(app(&mgr), "GET", &format!("/{id}"), None).await;

        assert_eq!(status, StatusCode::OK);
        assert_eq!(body["id"], json!(id));
        assert_eq!(body["name"], json!("lookup"));
        // Only the masked prefix is exposed; the full key never appears.
        assert_eq!(body["key_prefix"], json!(&key[..8]));
        assert!(body.get("key").is_none());
    }

    /// Req 10.4: GET /{id} for an unknown id returns 404 with an error message.
    #[tokio::test]
    async fn get_by_id_returns_404_for_unknown() {
        let (mgr, _tmp) = manager();

        let (status, body) = send(app(&mgr), "GET", "/does-not-exist", None).await;

        assert_eq!(status, StatusCode::NOT_FOUND);
        assert!(body["error"].as_str().is_some());
    }

    /// Req 10.1/10.6: PATCH /{id} returns 200 and the response reflects the
    /// updated field while preserving omitted ones.
    #[tokio::test]
    async fn patch_updates_and_reflects_change() {
        let (mgr, _tmp) = manager();
        let (id, _key) = create_key(
            &mgr,
            json!({ "name": "before", "budget_limit_usd": 100.0 }),
        )
        .await;

        let (status, body) = send(
            app(&mgr),
            "PATCH",
            &format!("/{id}"),
            Some(json!({ "name": "after" })),
        )
        .await;

        assert_eq!(status, StatusCode::OK);
        assert_eq!(body["name"], json!("after"));
        // Omitted budget field is preserved.
        assert_eq!(body["budget_limit_usd"], json!(100.0));
    }

    /// Req 10.3: PATCH /{id} with an out-of-range field returns 400 with a
    /// top-level `errors` array identifying the offending field.
    #[tokio::test]
    async fn patch_invalid_input_returns_400_with_errors() {
        let (mgr, _tmp) = manager();
        let (id, _key) = create_key(&mgr, json!({ "name": "victim" })).await;

        let (status, body) = send(
            app(&mgr),
            "PATCH",
            &format!("/{id}"),
            // budget_limit_usd = 0 is below the 0.01 minimum.
            Some(json!({ "budget_limit_usd": 0.0 })),
        )
        .await;

        assert_eq!(status, StatusCode::BAD_REQUEST);
        let errors = body["errors"].as_array().expect("errors array present");
        assert!(errors
            .iter()
            .any(|e| e["field"] == json!("budget_limit_usd")));
    }

    /// Req 7.8 via Req 10.x: PATCH /{id} on a revoked key returns 409.
    #[tokio::test]
    async fn patch_revoked_key_returns_409() {
        let (mgr, _tmp) = manager();
        let (id, _key) = create_key(&mgr, json!({ "name": "to-revoke" })).await;

        // Revoke via the revoke endpoint first.
        let (status, _) = send(app(&mgr), "POST", &format!("/{id}/revoke"), None).await;
        assert_eq!(status, StatusCode::OK);

        let (status, body) = send(
            app(&mgr),
            "PATCH",
            &format!("/{id}"),
            Some(json!({ "name": "nope" })),
        )
        .await;

        assert_eq!(status, StatusCode::CONFLICT);
        assert!(body["error"].as_str().is_some());
    }

    /// Req 10.1: DELETE /{id} returns 204 and a subsequent GET returns 404.
    #[tokio::test]
    async fn delete_returns_204_then_get_404() {
        let (mgr, _tmp) = manager();
        let (id, _key) = create_key(&mgr, json!({ "name": "temp" })).await;

        let (status, body) = send(app(&mgr), "DELETE", &format!("/{id}"), None).await;
        assert_eq!(status, StatusCode::NO_CONTENT);
        assert_eq!(body, Value::Null);

        let (status, _) = send(app(&mgr), "GET", &format!("/{id}"), None).await;
        assert_eq!(status, StatusCode::NOT_FOUND);
    }

    /// Req 10.6: POST /{id}/revoke returns 200 and the key's status becomes
    /// `revoked`.
    #[tokio::test]
    async fn revoke_returns_200_and_status_revoked() {
        let (mgr, _tmp) = manager();
        let (id, _key) = create_key(&mgr, json!({ "name": "active-key" })).await;

        let (status, body) = send(app(&mgr), "POST", &format!("/{id}/revoke"), None).await;

        assert_eq!(status, StatusCode::OK);
        assert_eq!(body["status"], json!("revoked"));

        // Confirmed via a follow-up info fetch.
        let (status, info) = send(app(&mgr), "GET", &format!("/{id}"), None).await;
        assert_eq!(status, StatusCode::OK);
        assert_eq!(info["status"], json!("revoked"));
    }

    /// Req 10.3: POST / with an empty name returns 400 with an `errors` array.
    #[tokio::test]
    async fn create_empty_name_returns_400_with_errors() {
        let (mgr, _tmp) = manager();

        let (status, body) = send(app(&mgr), "POST", "/", Some(json!({ "name": "" }))).await;

        assert_eq!(status, StatusCode::BAD_REQUEST);
        let errors = body["errors"].as_array().expect("errors array present");
        assert!(errors.iter().any(|e| e["field"] == json!("name")));
    }

    /// Req 10.3: POST / with an out-of-range `tokens_per_minute` returns 400
    /// with an `errors` array identifying the field.
    #[tokio::test]
    async fn create_tpm_too_high_returns_400_with_errors() {
        let (mgr, _tmp) = manager();

        let (status, body) = send(
            app(&mgr),
            "POST",
            "/",
            // 20,000,000 exceeds the 10,000,000 creation ceiling.
            Some(json!({ "tokens_per_minute": 20_000_000 })),
        )
        .await;

        assert_eq!(status, StatusCode::BAD_REQUEST);
        let errors = body["errors"].as_array().expect("errors array present");
        assert!(errors
            .iter()
            .any(|e| e["field"] == json!("tokens_per_minute")));
    }
}
