//! Admin prompt-library API integration tests (spec:
//! prompt-library-versioning, Task 9).
//!
//! Exercises the `/admin/prompt-library` routes through the full admin
//! router via `tower::ServiceExt::oneshot()` - no live server. Databases
//! are isolated per test via `common::isolate_databases`, and the prompt
//! store is redirected into the same unique temp dir (sibling of the
//! redirected logs.db), so RUST_TEST_THREADS-parallel tests never share
//! `./prompts.db`.
//!
//! Covers (Reqs 1, 2, 3, 5):
//! - create/list/get/delete round trip with validation errors naming the
//!   field: bad name, empty content, oversize content, duplicate name
//!   (Req 1.1-1.3)
//! - edit via POST /prompts/{id}/versions -> new immutable version;
//!   version list newest-first (Req 1.3, 2.2)
//! - diff: added/removed/unchanged; equal versions -> field-naming error;
//!   unknown version -> 404 (Req 2.3)
//! - promote + promotion history (from_version recorded); promote to the
//!   virtual `latest` label rejected naming the label (Req 3)
//! - reserved label set on the labels endpoint incl. virtual latest =
//!   max version, never stored (Req 3)
//! - bounds only at the admin layer (name/content/change_note); the
//! MAX_PROMPTS=1000 and version-cap=100 limits are enforced by the
//! store unit tests in `src/prompt_library/store.rs` (the prompt cap
//! via direct-SQL seeding, the version cap by looping) - driving 1000
//! HTTP creates here would be too slow, so the full-cap path is
//! deliberately not re-tested
//! - dormant library: fresh isolated state lists empty; the config
//! endpoint reports the active store path (Req 5)
//! - hot-reload: PUT /config switches to a new database_path (ok
//! readback, new store serves ops, old file kept); an unopenable
//! path 503s and retains the previous snapshot (Req 6.4)
//! - 401 without credentials (existing admin auth layer)
//! - label move does not alter version rows: v1 stays retrievable and
//!   unchanged after production is promoted to v2 (Req 3.5)

use crate::common;

use ai_gateway::config::{AdminAuthConfig, Config};
use ai_gateway::gateway::GatewayServer;
use axum::body::Body;
use axum::http::{Request, StatusCode};
use serde_json::{json, Value};
use tower::ServiceExt;

const BASE: &str = "/admin/prompt-library";

fn prompt_library_test_config() -> Config {
    let mut config = common::budget_test_config(None, None);
    config.prompt_library.database_path = String::new(); // set by isolate
    config
}

/// Build the app with isolated databases; the prompt store is placed as a
/// sibling of the redirected logs.db so each test gets its own prompts.db.
/// Returns the app, the server, and the isolated prompts.db path.
async fn build_app(config: Config) -> (axum::Router, GatewayServer, String) {
    let (router, server, prompts_db) = build_app_with_config_path(config, None).await;
    (router, server, prompts_db)
}

/// [`build_app`] with an explicit config file path: `Some(temp path)`
/// keeps PUT /config persistence (save_config) out of the repo tree
/// (webhooks_admin.rs discipline), while `None` keeps the legacy
/// default for tests that never persist.
async fn build_app_with_config_path(
    mut config: Config,
    config_path: Option<std::path::PathBuf>,
) -> (axum::Router, GatewayServer, String) {
    common::isolate_databases(&mut config);
    let prompts_db = std::path::Path::new(&config.logging.database_path)
        .parent()
        .expect("isolated logs.db parent")
        .join("prompts.db")
        .to_string_lossy()
        .into_owned();
    config.prompt_library.database_path = prompts_db.clone();
    let server = GatewayServer::new(config, config_path).await.unwrap();
    let router = server.build_router();
    (router, server, prompts_db)
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

async fn get(app: axum::Router, uri: &str) -> (StatusCode, Value) {
    send(app, Request::get(uri).body(Body::empty()).unwrap()).await
}

async fn post(app: axum::Router, uri: &str, body: Value) -> (StatusCode, Value) {
    send(
        app,
        Request::post(uri)
            .header("content-type", "application/json")
            .body(Body::from(serde_json::to_string(&body).unwrap()))
            .unwrap(),
    )
    .await
}

async fn create_prompt(app: axum::Router, name: &str, content: &str) -> Value {
    let (status, body) = post(app, &format!("{BASE}/prompts"), json!({ "name": name, "content": content })).await;
    assert_eq!(status, StatusCode::OK, "create failed: {body}");
    body
}

fn assert_field_error(body: &Value, status: StatusCode, field: &str) {
    assert_eq!(status, StatusCode::BAD_REQUEST, "expected 400 for {field}: {body}");
    assert_eq!(body["error"]["type"], "invalid_request", "{body}");
    assert!(
        body["error"]["message"].as_str().unwrap().contains(field),
        "message must name {field}: {body}"
    );
}

// ---------------------------------------------------------------------------
// Create/list/get/delete round trip + validation errors (Req 1)
// ---------------------------------------------------------------------------

#[tokio::test]
async fn create_get_list_delete_round_trip_with_validation_errors() {
    let (app, _server, _prompts_db) = build_app(prompt_library_test_config()).await;

    let created = create_prompt(app.clone(), "greeting", "Hello {{name}}!").await;
    let id = created["id"].as_str().expect("created id").to_owned();
    assert_eq!(created["object"], "prompt");
    assert_eq!(created["name"], "greeting");
    assert_eq!(created["labels"]["production"], 1);
    assert_eq!(created["latest_version"], 1);
    assert_eq!(created["version_count"], 1);

    // Validation errors: each names the invalid field and stores nothing.
    let (status, body) = post(app.clone(), &format!("{BASE}/prompts"), json!({
        "name": "Not-Kebab!", "content": "hello"
    })).await;
    assert_field_error(&body, status, "name");

    let (status, body) = post(app.clone(), &format!("{BASE}/prompts"), json!({
        "name": "valid-name", "content": ""
    })).await;
    assert_field_error(&body, status, "content");

    let (status, body) = post(app.clone(), &format!("{BASE}/prompts"), json!({
        "name": "valid-name", "content": "x".repeat(100_001)
    })).await;
    assert_field_error(&body, status, "content");

    let (status, body) = post(app.clone(), &format!("{BASE}/prompts"), json!({
        "name": "greeting", "content": "duplicate"
    })).await;
    assert_field_error(&body, status, "name");
    assert!(body["error"]["message"].as_str().unwrap().contains("already exists"), "{body}");

    // Nothing was stored by the rejected creates.
    let (status, page) = get(app.clone(), &format!("{BASE}/prompts")).await;
    assert_eq!(status, StatusCode::OK);
    assert_eq!(page["object"], "list");
    assert_eq!(page["data"].as_array().unwrap().len(), 1);
    assert_eq!(page["has_more"], false);

    let (status, fetched) = get(app.clone(), &format!("{BASE}/prompts/{id}")).await;
    assert_eq!(status, StatusCode::OK);
    assert_eq!(fetched, created);

    let (status, _) = send(app.clone(), Request::delete(format!("{BASE}/prompts/{id}")).body(Body::empty()).unwrap()).await;
    assert_eq!(status, StatusCode::NO_CONTENT);
    let (status, body) = get(app.clone(), &format!("{BASE}/prompts/{id}")).await;
    assert_eq!(status, StatusCode::NOT_FOUND);
    assert_eq!(body["error"]["type"], "not_found");
}

// ---------------------------------------------------------------------------
// Edit creates a new immutable version; list is newest-first (Req 1.3, 2.2)
// ---------------------------------------------------------------------------

#[tokio::test]
async fn edit_creates_new_immutable_version_listed_newest_first() {
    let (app, _server, _prompts_db) = build_app(prompt_library_test_config()).await;
    let created = create_prompt(app.clone(), "summarizer", "Summarize:\n{{text}}").await;
    let id = created["id"].as_str().unwrap().to_owned();
    let base = format!("{BASE}/prompts/{id}");

    let (status, version) = post(app.clone(), &format!("{base}/versions"), json!({
        "content": "Summarize for {{audience}}:\n{{text}}",
        "change_note": "add audience variable",
        "created_by": "editor"
    })).await;
    assert_eq!(status, StatusCode::OK, "{version}");
    assert_eq!(version["version"], 2);
    assert_eq!(version["change_note"], "add audience variable");
    assert_eq!(version["created_by"], "editor");
    assert_eq!(version["variables"], json!(["audience", "text"]));

    // v1 remains retrievable and unchanged.
    let (status, versions) = get(app.clone(), &format!("{base}/versions")).await;
    assert_eq!(status, StatusCode::OK);
    let rows = versions.as_array().expect("version array");
    assert_eq!(rows.len(), 2);
    assert_eq!(rows[0]["version"], 2, "newest first: {versions}");
    assert_eq!(rows[1]["version"], 1);
    assert_eq!(rows[1]["content"], "Summarize:\n{{text}}");

 // Bounds on the edit payload name their fields too.
 let (status, body) = post(app.clone(), &format!("{base}/versions"), json!({
 "content": "new", "change_note": ""
 })).await;
 assert_field_error(&body, status, "change_note");
 let (status, body) = post(app.clone(), &format!("{base}/versions"), json!({
 "content": "x".repeat(100_001), "change_note": "ok"
 })).await;
 assert_field_error(&body, status, "content");
}

// ---------------------------------------------------------------------------
// Edit with a changed name renames in the same version transaction (Req 1.3)
// ---------------------------------------------------------------------------

#[tokio::test]
async fn edit_with_changed_name_renames_and_frees_old_name() {
 let (app, _server, _prompts_db) = build_app(prompt_library_test_config()).await;
 let created = create_prompt(app.clone(), "greeter", "Hello!").await;
 let id = created["id"].as_str().unwrap().to_owned();
 let base = format!("{BASE}/prompts/{id}");

 let (status, version) = post(app.clone(), &format!("{base}/versions"), json!({
 "content": "Hello!",
 "change_note": "rename to welcomer",
 "name": "welcomer"
 })).await;
 assert_eq!(status, StatusCode::OK, "{version}");
 assert_eq!(version["version"], 2);
 assert_eq!(version["prompt_id"], id, "versions stay under the same id");

 // get/list show the new name under the same id.
 let (status, fetched) = get(app.clone(), &format!("{BASE}/prompts/{id}")).await;
 assert_eq!(status, StatusCode::OK, "{fetched}");
 assert_eq!(fetched["name"], "welcomer");
 assert_eq!(fetched["id"], id);
 assert_eq!(fetched["version_count"], 2);
 let (status, page) = get(app.clone(), &format!("{BASE}/prompts")).await;
 assert_eq!(status, StatusCode::OK);
 assert_eq!(page["data"].as_array().unwrap().len(), 1);
 assert_eq!(page["data"][0]["name"], "welcomer");
 assert_eq!(page["data"][0]["id"], id);

 // The old name is free again: creating a new prompt with it succeeds.
 let (status, recreated) = post(app.clone(), &format!("{BASE}/prompts"), json!({
 "name": "greeter", "content": "fresh"
 })).await;
 assert_eq!(status, StatusCode::OK, "{recreated}");
 assert_ne!(recreated["id"].as_str().unwrap(), id);

 // Renaming to the other prompt's name is rejected naming the field.
 let (status, body) = post(app.clone(), &format!("{base}/versions"), json!({
 "content": "Hello!", "change_note": "clash", "name": "greeter"
 })).await;
 assert_field_error(&body, status, "name");

 // Name-only edit (same content, new name) also creates a new version.
 let (status, version) = post(app.clone(), &format!("{base}/versions"), json!({
 "content": "Hello!",
 "change_note": "name only",
 "name": "welcomer-v2"
 })).await;
 assert_eq!(status, StatusCode::OK, "{version}");
 assert_eq!(version["version"], 3);
 let (status, fetched) = get(app.clone(), &format!("{BASE}/prompts/{id}")).await;
 assert_eq!(status, StatusCode::OK);
 assert_eq!(fetched["name"], "welcomer-v2");
 assert_eq!(fetched["version_count"], 3);

 // An invalid rename is rejected without storing a version.
 let (status, body) = post(app.clone(), &format!("{base}/versions"), json!({
 "content": "Hello!", "change_note": "bad name", "name": "Not Kebab"
 })).await;
 assert_field_error(&body, status, "name");
 let (status, fetched) = get(app.clone(), &format!("{BASE}/prompts/{id}")).await;
 assert_eq!(status, StatusCode::OK);
 assert_eq!(fetched["name"], "welcomer-v2");
 assert_eq!(fetched["version_count"], 3);
}

// ---------------------------------------------------------------------------
// Diff: added/removed/unchanged; equal ? 400 naming versions; unknown ? 404
// (Req 2.3)
// ---------------------------------------------------------------------------

#[tokio::test]
async fn diff_reports_lines_and_rejects_equal_or_unknown_versions() {
    let (app, _server, _prompts_db) = build_app(prompt_library_test_config()).await;
    let created = create_prompt(app.clone(), "diffed", "same line\nremoved line").await;
    let id = created["id"].as_str().unwrap().to_owned();
    let base = format!("{BASE}/prompts/{id}");

    let (status, _) = post(app.clone(), &format!("{base}/versions"), json!({
        "content": "same line\nadded line", "change_note": "swap lines"
    })).await;
    assert_eq!(status, StatusCode::OK);

    let (status, diff) = post(app.clone(), &format!("{base}/diff"), json!({
        "from_version": 1, "to_version": 2
    })).await;
    assert_eq!(status, StatusCode::OK, "{diff}");
    assert_eq!(diff, json!([
        { "kind": "unchanged", "text": "same line" },
        { "kind": "removed", "text": "removed line" },
        { "kind": "added", "text": "added line" }
    ]));

    let (status, body) = post(app.clone(), &format!("{base}/diff"), json!({
        "from_version": 1, "to_version": 1
    })).await;
    assert_field_error(&body, status, "from_version");

    let (status, body) = post(app.clone(), &format!("{base}/diff"), json!({
        "from_version": 1, "to_version": 99
    })).await;
    assert_eq!(status, StatusCode::NOT_FOUND, "{body}");
    assert_eq!(body["error"]["type"], "not_found");
}

// ---------------------------------------------------------------------------
// Promote + promotion history; latest label is virtual (Req 3)
// ---------------------------------------------------------------------------

#[tokio::test]
async fn promote_records_history_and_rejects_virtual_latest_label() {
    let (app, _server, _prompts_db) = build_app(prompt_library_test_config()).await;
    let created = create_prompt(app.clone(), "classifier", "v1 text").await;
    let id = created["id"].as_str().unwrap().to_owned();
    let base = format!("{BASE}/prompts/{id}");

    let (status, version) = post(app.clone(), &format!("{base}/versions"), json!({
        "content": "v2 text", "change_note": "second"
    })).await;
    assert_eq!(status, StatusCode::OK);
    assert_eq!(version["version"], 2);

    let (status, moved) = post(app.clone(), &format!("{base}/promote"), json!({
        "label": "production", "to_version": 2, "actor": "operator"
    })).await;
    assert_eq!(status, StatusCode::OK, "{moved}");
    assert_eq!(moved["label"], "production");
    assert_eq!(moved["version"], 2);
    assert_eq!(moved["virtual"], false);

    // Promotion history: initial create row plus the explicit move, with
    // from_version recorded for the latter.
    let (status, history) = get(app.clone(), &format!("{base}/promotions")).await;
    assert_eq!(status, StatusCode::OK, "{history}");
    let rows = history["data"].as_array().unwrap();
    assert_eq!(rows.len(), 2);
    let promotion = rows.iter().find(|row| row["to_version"] == 2).expect("promotion row");
    assert_eq!(promotion["from_version"], 1);
    assert_eq!(promotion["actor"], "operator");

    // Promoting the virtual latest label is rejected naming the label.
    let (status, body) = post(app.clone(), &format!("{base}/promote"), json!({
        "label": "latest", "to_version": 1
    })).await;
    assert_field_error(&body, status, "label");
}

// ---------------------------------------------------------------------------
// Reserved labels on the labels endpoint; virtual latest = max version
// (Req 3)
// ---------------------------------------------------------------------------

#[tokio::test]
async fn labels_endpoint_lists_reserved_labels_with_virtual_latest() {
    let (app, _server, _prompts_db) = build_app(prompt_library_test_config()).await;
    let created = create_prompt(app.clone(), "labeled", "one").await;
    let id = created["id"].as_str().unwrap().to_owned();
    let base = format!("{BASE}/prompts/{id}");

    let (status, _) = post(app.clone(), &format!("{base}/versions"), json!({
        "content": "two", "change_note": "second"
    })).await;
    assert_eq!(status, StatusCode::OK);

    let (status, labels) = get(app.clone(), &format!("{base}/labels")).await;
    assert_eq!(status, StatusCode::OK, "{labels}");
    let labels = labels.as_array().unwrap();
    let by_label = |name: &str| labels.iter().find(|entry| entry["label"] == name).unwrap();
    assert_eq!(labels.len(), 4, "three movable reserved labels + latest: {labels:?}");
    assert_eq!(by_label("production")["version"], 1);
    assert_eq!(by_label("production")["virtual"], false);
    assert_eq!(by_label("staging")["version"], Value::Null);
    assert_eq!(by_label("development")["version"], Value::Null);
    let latest = by_label("latest");
    assert_eq!(latest["version"], 2, "virtual latest = max version");
    assert_eq!(latest["virtual"], true);
}

// ---------------------------------------------------------------------------
// Dormant library: empty list envelope + default config path (Req 5)
// ---------------------------------------------------------------------------

#[tokio::test]
async fn dormant_library_lists_empty_and_reports_isolated_path() {
    let (app, _server, prompts_db) = build_app(prompt_library_test_config()).await;
    let (status, page) = get(app.clone(), &format!("{BASE}/prompts")).await;
    assert_eq!(status, StatusCode::OK);
    assert_eq!(page, json!({
        "object": "list", "data": [], "first_id": null, "last_id": null, "has_more": false
    }));
    // Dormant library with an opened (schema-initialized) store: the config
    // endpoint reports the active store's path  the default `./prompts.db`
    // shape only appears when no library was opened at all (unit-tested in
    // src/prompt_library/admin.rs).
    let (status, config) = get(app, &format!("{BASE}/config")).await;
    assert_eq!(status, StatusCode::OK);
    assert_eq!(config["database_path"], prompts_db);
}

// ---------------------------------------------------------------------------
// Auth: 401 without credentials (existing admin auth layer)
// ---------------------------------------------------------------------------

/// Restore a process-global env var on scope exit, including
/// assertion-failure unwinds, so one failing test never leaks
/// credentials into the shared test process.
struct EnvVarGuard {
    key: &'static str,
    prior: Option<String>,
}

impl EnvVarGuard {
    fn set(key: &'static str, value: &str) -> Self {
        let prior = std::env::var(key).ok();
        std::env::set_var(key, value);
        Self { key, prior }
    }
}

impl Drop for EnvVarGuard {
    fn drop(&mut self) {
        match self.prior.take() {
            Some(value) => std::env::set_var(self.key, value),
            None => std::env::remove_var(self.key),
        }
    }
}

#[tokio::test]
async fn prompt_library_routes_require_credentials_when_auth_enabled() {
    let user_env = "PROMPT_LIBRARY_ADMIN_TEST_USER";
    let pass_env = "PROMPT_LIBRARY_ADMIN_TEST_PASS";
    // Unique env-var names per test binary: env is process-global
    // (webhooks_admin.rs / cost_reporting_admin.rs precedent). The
    // guards remove both vars on every exit path, including a failed
    // assertion mid-test.
    let _user = EnvVarGuard::set(user_env, "admin-user");
    let _pass = EnvVarGuard::set(pass_env, "admin-pass");

    let mut config = prompt_library_test_config();
    config.admin.auth = AdminAuthConfig {
        enabled: true,
        username_env: Some(user_env.to_string()),
        password_env: Some(pass_env.to_string()),
        allow_unauthenticated_non_loopback: false,
    };
    let (app, _server, _prompts_db) = build_app(config).await;

    let (status, _) = get(app.clone(), &format!("{BASE}/prompts")).await;
    assert_eq!(status, StatusCode::UNAUTHORIZED);
}

// ---------------------------------------------------------------------------
// Label moves never alter version rows (Req 3.5 admin-level stability)
// ---------------------------------------------------------------------------

#[tokio::test]
async fn label_move_does_not_alter_version_rows() {
    let (app, _server, _prompts_db) = build_app(prompt_library_test_config()).await;
    let created = create_prompt(app.clone(), "pinned", "stable v1").await;
    let id = created["id"].as_str().unwrap().to_owned();
    let base = format!("{BASE}/prompts/{id}");

    let (status, _) = post(app.clone(), &format!("{base}/versions"), json!({
        "content": "second version", "change_note": "second"
    })).await;
    assert_eq!(status, StatusCode::OK);

    let (status, _) = post(app.clone(), &format!("{base}/promote"), json!({
        "label": "production", "to_version": 2
    })).await;
    assert_eq!(status, StatusCode::OK);

    // Both versions survive the label move unchanged.
    let (status, versions) = get(app.clone(), &format!("{base}/versions")).await;
    assert_eq!(status, StatusCode::OK);
    let rows = versions.as_array().unwrap();
    assert_eq!(rows.len(), 2);
    assert_eq!(rows[0]["version"], 2);
    assert_eq!(rows[0]["content"], "second version");
    assert_eq!(rows[1]["version"], 1);
    assert_eq!(rows[1]["content"], "stable v1");

// The prompt summary reflects the move without losing either version.
let (status, fetched) = get(app.clone(), &format!("{base}")).await;
assert_eq!(status, StatusCode::OK);
assert_eq!(fetched["labels"]["production"], 2);
assert_eq!(fetched["latest_version"], 2);
assert_eq!(fetched["version_count"], 2);
}

// ---------------------------------------------------------------------------
// Hot-reload: PUT /config switches stores; failures retain the snapshot (Req 6.4)
// ---------------------------------------------------------------------------

async fn put(app: axum::Router, uri: &str, body: Value) -> (StatusCode, Value) {
send(
app,
Request::put(uri)
.header("content-type", "application/json")
.body(Body::from(serde_json::to_string(&body).unwrap()))
.unwrap(),
)
.await
}

/// PUT /config with a new valid temp database_path: ok response naming
/// the new path, subsequent prompt operations run against the new
/// store, and the old store file is untouched on disk.
#[tokio::test]
async fn put_config_switches_to_new_store_and_keeps_old_file() {
let mut config = prompt_library_test_config();
config.server.port = 8080; // full-config validation (PUT persists) rejects port 0
let config_dir = tempfile::tempdir().expect("temp config dir").keep();
let config_path = config_dir.join("config.yaml");
let (app, _server, old_prompts_db) =
build_app_with_config_path(config, Some(config_path)).await;

// Seed the old store so the switch is observable through the API.
let seeded = create_prompt(app.clone(), "old-store-prompt", "v1").await;
let old_id = seeded["id"].as_str().expect("seeded id").to_owned();

let new_dir = tempfile::tempdir().expect("new prompts dir").keep();
let new_prompts_db = new_dir.join("prompts.db");
let new_prompts_db = new_prompts_db.to_string_lossy().into_owned();
let (status, body) = put(
app.clone(),
&format!("{BASE}/config"),
json!({ "database_path": new_prompts_db }),
)
.await;
assert_eq!(status, StatusCode::OK, "switch should apply: {body}");
assert_eq!(body["status"], "ok");
assert_eq!(body["database_path"], new_prompts_db);

// The live snapshot serves the new store: the seeded prompt is gone,
// a new create lands, and it is retrievable via GET.
let (status, page) = get(app.clone(), &format!("{BASE}/prompts")).await;
assert_eq!(status, StatusCode::OK, "{page}");
let ids: Vec<&str> = page["data"]
.as_array()
.expect("list envelope")
.iter()
.filter_map(|entry| entry["id"].as_str())
.collect();
assert!(!ids.contains(&old_id.as_str()), "new store starts empty: {page}");

let created = create_prompt(app.clone(), "new-store-prompt", "fresh").await;
let new_id = created["id"].as_str().expect("created id").to_owned();
let (status, fetched) = get(app.clone(), &format!("{BASE}/prompts/{new_id}")).await;
assert_eq!(status, StatusCode::OK, "created prompt readable from new store: {fetched}");
assert_eq!(fetched["name"], "new-store-prompt");

// The old database file still exists (neither dropped nor deleted).
assert!(
std::path::Path::new(&old_prompts_db).exists(),
"old store file must survive the switch"
);
}

/// PUT /config with an invalid path (parent is a regular file): 503,
/// previous snapshot retained - listing still works against it.
#[tokio::test]
async fn put_config_with_unopenable_path_returns_503_and_retains_snapshot() {
let mut config = prompt_library_test_config();
config.server.port = 8080; // full-config validation (PUT persists) rejects port 0
let config_dir = tempfile::tempdir().expect("temp config dir").keep();
let config_path = config_dir.join("config.yaml");
let (app, _server, _prompts_db) =
build_app_with_config_path(config, Some(config_path)).await;

let seeded = create_prompt(app.clone(), "survivor", "still here").await;
let survivor_id = seeded["id"].as_str().expect("seeded id").to_owned();

// The parent of the database path is a regular file: SQLite cannot
// open `<file>/prompts.db`.
let blocker = tempfile::NamedTempFile::new().expect("blocker file");
let blocked_path = blocker.path().join("prompts.db");
let blocked_path = blocked_path.to_string_lossy().into_owned();
let (status, body) = put(
app.clone(),
&format!("{BASE}/config"),
json!({ "database_path": blocked_path }),
)
.await;
assert_eq!(
status,
StatusCode::SERVICE_UNAVAILABLE,
"unopenable path must 503: {body}"
);
assert_eq!(body["error"]["type"], "api_error");

// Previous snapshot retained: the list still serves the seeded prompt.
let (status, page) = get(app.clone(), &format!("{BASE}/prompts")).await;
assert_eq!(status, StatusCode::OK, "list must keep working: {page}");
let ids: Vec<&str> = page["data"]
.as_array()
.expect("list envelope")
.iter()
.filter_map(|entry| entry["id"].as_str())
.collect();
assert!(
ids.contains(&survivor_id.as_str()),
"previous snapshot must survive the failed switch: {page}"
);
}
