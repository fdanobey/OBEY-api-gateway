//! Admin webhooks API integration tests (budget-alerts-unified-webhooks,
//! Task 7).
//!
//! Exercises the four `/admin/webhooks` routes through the full admin
//! router via `tower::ServiceExt::oneshot()` — no live server. Databases
//! are isolated per test via `common::isolate_databases`
//! (`webhooks.db` is a sibling of the redirected logs.db).
//!
//! Covers:
//! - GET returns masked secrets — sentinel present, raw secret absent
//! (Req 3.6)
//! - PUT round-trip: new endpoint config reflected by GET; a sentinel
//! secret preserves the stored secret for the same URL (Req 3.6)
//! - PUT with an http URL for a non-allowlisted host → 400 with a
//! host/scheme-scoped reason and no secret in the error body (Req 4.5,
//! design Error Handling 2)
//! - POST test against a wiremock receiver echoes the observed status
//! (Req 4.4); unknown URL → 404
//! - GET deliveries lists ledger rows (Req 4.8)
//! - failing rollup: 3 failed attempts → endpoint marked failing
//! (Req 5.5)
//! - 401 without credentials on all four routes (NFR Security: routes
//! sit behind the existing admin auth layer)

use axum::body::Body;
use axum::http::{Request, StatusCode};
use tower::ServiceExt;
use wiremock::matchers::{method as wm_method, path as wm_path};
use wiremock::{Mock, MockServer, ResponseTemplate};

use ai_gateway::config::*;
use ai_gateway::gateway::GatewayServer;

mod common;

const RAW_SECRET: &str = "whsec_raw_admin_secret";

fn webhooks_admin_test_config() -> Config {
    let mut config = common::budget_test_config(None, None);
    // Full-config validation (run by PUT before persist) rejects port 0.
    config.server.port = 8080;
    config.webhooks = WebhooksConfig {
        endpoints: vec![WebhookEndpoint {
            url: "https://hooks.example.com/alerts".to_string(),
            secret: Some(RAW_SECRET.to_string()),
            events: None,
        }],
        allowed_private_hosts: vec!["hooks.internal".to_string()],
    };
    config
}

/// Build the app with isolated databases and a temp config path (PUT
/// persists YAML through `save_config` — never into the repo tree).
async fn build_app(mut config: Config) -> (axum::Router, GatewayServer) {
    common::isolate_databases(&mut config);
    let config_dir = tempfile::tempdir().expect("temp config dir");
    let config_path = config_dir.path().join("config.yaml");
    // The dir must outlive the server's config_path handle; the gateway
    // holds no open handle to the YAML, so keep() matches the sibling
    // tests' leak-the-tiny-file discipline.
    let config_dir = config_dir.keep();
    let _ = &config_dir;
    let server = GatewayServer::new(config, Some(config_path)).await.unwrap();
    let router = server.build_router();
    (router, server)
}

async fn send(app: axum::Router, req: Request<Body>) -> (StatusCode, serde_json::Value) {
    let resp = app.oneshot(req).await.unwrap();
    let status = resp.status();
    let bytes = axum::body::to_bytes(resp.into_body(), 1024 * 1024)
        .await
        .unwrap();
    let json = if bytes.is_empty() {
        serde_json::Value::Null
    } else {
        serde_json::from_slice(&bytes).unwrap()
    };
    (status, json)
}

fn get_webhooks() -> Request<Body> {
    Request::get("/admin/webhooks").body(Body::empty()).unwrap()
}

fn put_webhooks(config: &WebhooksConfig) -> Request<Body> {
    Request::put("/admin/webhooks")
        .header("content-type", "application/json")
        .body(Body::from(serde_json::to_string(config).unwrap()))
        .unwrap()
}

/// Minimal percent-encoding for a URL carried in a query parameter
/// (scheme separator + path slashes).
fn encode_query(value: &str) -> String {
    value
        .replace('%', "%25")
        .replace(':', "%3A")
        .replace('/', "%2F")
}

// ---------------------------------------------------------------------------
// GET: masked secrets (Req 3.6)
// ---------------------------------------------------------------------------

#[tokio::test]
async fn get_webhooks_masks_secrets_and_lists_config() {
    let (app, _server) = build_app(webhooks_admin_test_config()).await;
    let (status, json) = send(app, get_webhooks()).await;

    assert_eq!(status, StatusCode::OK);
    let endpoints = json["endpoints"].as_array().expect("endpoints array");
    assert_eq!(endpoints.len(), 1);
    assert_eq!(endpoints[0]["url"], "https://hooks.example.com/alerts");
    assert_eq!(endpoints[0]["secret"], "[configured]");
    assert_eq!(endpoints[0]["signed"], true);
    assert_eq!(endpoints[0]["failing"], false);
    assert_eq!(json["allowed_private_hosts"][0], "hooks.internal");

    let serialized = serde_json::to_string(&json).unwrap();
    assert!(
        !serialized.contains(RAW_SECRET),
        "raw secret must never appear in the GET response: {serialized}"
    );
}

// ---------------------------------------------------------------------------
// PUT: round-trip + sentinel preserve (Req 3.6)
// ---------------------------------------------------------------------------

#[tokio::test]
async fn put_webhooks_round_trip_and_sentinel_preserves_stored_secret() {
    let (app, server) = build_app(webhooks_admin_test_config()).await;

    // Sentinel preserve FIRST, while the stored config still holds the
    // original endpoint with the raw secret: PUT the same endpoint set
    // with the mask sentinel as the secret — the stored secret for that
    // URL must survive (asserted via the live config the notifier signs
    // with).
    let preserved = WebhooksConfig {
        endpoints: vec![WebhookEndpoint {
            url: "https://hooks.example.com/alerts".to_string(),
            secret: Some("[configured]".to_string()),
            events: None,
        }],
        allowed_private_hosts: vec![],
    };
    let (status, body) = send(app.clone(), put_webhooks(&preserved)).await;
    assert_eq!(status, StatusCode::OK, "sentinel PUT applies: {body}");

    let stored_secret = {
        let live = server.state.config.read().await;
        live.webhooks.endpoints[0].secret.clone()
    };
    assert_eq!(
        stored_secret.as_deref(),
        Some(RAW_SECRET),
        "mask sentinel returned unchanged must preserve the stored secret"
    );

    // GET still masks it.
    let (status, json) = send(app.clone(), get_webhooks()).await;
    assert_eq!(status, StatusCode::OK);
    assert_eq!(json["endpoints"][0]["secret"], "[configured]");
    assert!(
        !serde_json::to_string(&json).unwrap().contains(RAW_SECRET),
        "GET must never echo the raw secret"
    );

    // Round-trip a new endpoint set: drop the stored endpoint, add a new
    // unsigned one plus a new allowlist entry.
    let updated = WebhooksConfig {
        endpoints: vec![WebhookEndpoint {
            url: "https://hooks.example.com/new".to_string(),
            secret: None,
            events: Some(vec![BudgetEventType::BudgetWarning]),
        }],
        allowed_private_hosts: vec!["hooks.other.internal".to_string()],
    };
    let (status, body) = send(app.clone(), put_webhooks(&updated)).await;
    assert_eq!(status, StatusCode::OK, "PUT applies: {body}");

    let (status, json) = send(app, get_webhooks()).await;
    assert_eq!(status, StatusCode::OK);
    assert_eq!(json["endpoints"][0]["url"], "https://hooks.example.com/new");
    assert_eq!(json["endpoints"][0]["secret"], serde_json::Value::Null);
    assert_eq!(json["endpoints"][0]["signed"], false);
    assert_eq!(json["endpoints"][0]["events"][0], "budget.warning");
    assert_eq!(json["allowed_private_hosts"][0], "hooks.other.internal");
}

// ---------------------------------------------------------------------------
// PUT: invalid URL → 400, host-scoped reason, no secret leak (Req 4.5)
// ---------------------------------------------------------------------------

#[tokio::test]
async fn put_webhooks_rejects_non_allowlisted_http_url_without_leaking_secret() {
    let (app, _server) = build_app(webhooks_admin_test_config()).await;

    let invalid = WebhooksConfig {
        endpoints: vec![WebhookEndpoint {
            url: "http://hooks.example.com/alerts".to_string(),
            secret: Some("whsec_should_not_leak".to_string()),
            events: None,
        }],
        // Allowlist names a DIFFERENT host — exact-match semantics.
        allowed_private_hosts: vec!["other.internal".to_string()],
    };
    let (status, json) = send(app, put_webhooks(&invalid)).await;

    assert_eq!(status, StatusCode::BAD_REQUEST, "body: {json}");
    let serialized = serde_json::to_string(&json).unwrap();
    assert!(
        !serialized.contains("whsec_should_not_leak"),
        "error body must never include the secret: {serialized}"
    );
    let details = json["error"]["details"].as_array().expect("details array");
    assert!(
        details.iter().any(|d| d.as_str().unwrap_or_default().contains("webhooks.endpoints[0].url")),
        "reason must be scoped to the endpoint URL field: {json}"
    );
    assert!(
        serialized.contains("https"),
        "reason must explain the https-only rule: {json}"
    );
}

// ---------------------------------------------------------------------------
// POST test: echoes observed status; unknown URL 404 (Req 4.4)
// ---------------------------------------------------------------------------

#[tokio::test]
async fn post_test_webhook_echoes_observed_status() {
    let receiver = MockServer::start().await;
    Mock::given(wm_method("POST"))
        .and(wm_path("/hook"))
        .respond_with(ResponseTemplate::new(500))
        .mount(&receiver)
        .await;

    let mut config = webhooks_admin_test_config();
    let receiver_url = format!("{}/hook", receiver.uri());
    let receiver_host = reqwest::Url::parse(&receiver.uri())
        .unwrap()
        .host_str()
        .unwrap()
        .to_string();
    config.webhooks = WebhooksConfig {
        endpoints: vec![WebhookEndpoint {
            url: receiver_url,
            secret: Some(RAW_SECRET.to_string()),
            events: None,
        }],
        allowed_private_hosts: vec![receiver_host],
    };
    let (app, server) = build_app(config).await;

    // Unknown URL → 404 (sibling not-found convention).
    let req = Request::post("/admin/webhooks/test")
        .header("content-type", "application/json")
        .body(Body::from(
            serde_json::to_string(&serde_json::json!({"url": "https://no-such-endpoint.example/"}))
                .unwrap(),
        ))
        .unwrap();
    let (status, json) = send(app.clone(), req).await;
    assert_eq!(status, StatusCode::NOT_FOUND, "body: {json}");
    assert_eq!(json["error"]["type"], "endpoint_not_found");

    // Known URL: single attempt echoes the observed 500 (Req 4.4).
    let endpoint_url = server
        .state
        .config
        .read()
        .await
        .webhooks
        .endpoints[0]
        .url
        .clone();
    let req = Request::post("/admin/webhooks/test")
        .header("content-type", "application/json")
        .body(Body::from(
            serde_json::to_string(&serde_json::json!({ "url": endpoint_url })).unwrap(),
        ))
        .unwrap();
    let (status, json) = send(app, req).await;
    assert_eq!(status, StatusCode::OK, "test-send reports, not fails: {json}");
    assert_eq!(json["status_code"], 500);
    assert_eq!(json["success"], false);
    assert_eq!(json["error"], serde_json::Value::Null);
}

#[tokio::test]
async fn post_test_webhook_success_echoes_200() {
    let receiver = MockServer::start().await;
    Mock::given(wm_method("POST"))
        .respond_with(ResponseTemplate::new(200))
        .mount(&receiver)
        .await;

    let mut config = webhooks_admin_test_config();
    let receiver_url = format!("{}/hook", receiver.uri());
    let receiver_host = reqwest::Url::parse(&receiver.uri())
        .unwrap()
        .host_str()
        .unwrap()
        .to_string();
    config.webhooks = WebhooksConfig {
        endpoints: vec![WebhookEndpoint {
            url: receiver_url.clone(),
            secret: None,
            events: None,
        }],
        allowed_private_hosts: vec![receiver_host],
    };
    let (app, _server) = build_app(config).await;

    let req = Request::post("/admin/webhooks/test")
        .header("content-type", "application/json")
        .body(Body::from(
            serde_json::to_string(&serde_json::json!({ "url": receiver_url })).unwrap(),
        ))
        .unwrap();
    let (status, json) = send(app, req).await;
    assert_eq!(status, StatusCode::OK);
    assert_eq!(json["status_code"], 200);
    assert_eq!(json["success"], true);
}

// ---------------------------------------------------------------------------
// GET deliveries: ledger rows listed (Req 4.8)
// ---------------------------------------------------------------------------

/// Seed the ledger directly (the simplest path — no spend event needed):
/// one event row plus `failed` attempts, mirroring what the delivery
/// engine writes.
fn seed_ledger_failures(server: &GatewayServer, endpoint_url: &str, attempts: i64) {
    let ledger = &server.state.webhook_ledger;
    let event_id = format!("evt-{endpoint_url}");
    ledger
        .insert_event(&event_id, "budget.warning", "provider", "budget-provider", "{}", 1)
        .unwrap();
    for attempt in 1..=attempts {
        ledger
            .record_attempt(&event_id, endpoint_url, attempt, "failed", Some(500), None, attempt)
            .unwrap();
    }
}

#[tokio::test]
async fn get_deliveries_lists_ledger_rows() {
    let (app, server) = build_app(webhooks_admin_test_config()).await;
    let endpoint_url = "https://hooks.example.com/alerts";
    seed_ledger_failures(&server, endpoint_url, 2);

    let req = Request::get(format!(
        "/admin/webhooks/deliveries?endpoint={}&limit=1",
        encode_query(endpoint_url)
    ))
    .body(Body::empty())
    .unwrap();
    let (status, json) = send(app, req).await;

    assert_eq!(status, StatusCode::OK);
    assert_eq!(json["endpoint"], endpoint_url);
    assert_eq!(json["limit"], 1);
    let rows = json["deliveries"].as_array().expect("deliveries array");
    assert_eq!(rows.len(), 1, "limit is honored: {json}");
    // Newest first.
    assert_eq!(rows[0]["attempt"], 2);
    assert_eq!(rows[0]["outcome"], "failed");
    assert_eq!(rows[0]["status_code"], 500);
    assert_eq!(rows[0]["endpoint_url"], endpoint_url);
}

/// "All endpoints" mode: no endpoint parameter → newest rows across
/// every endpoint (the admin UI sends no endpoint param for the All
/// view).
#[tokio::test]
async fn get_deliveries_without_endpoint_lists_all_endpoints() {
    let (app, server) = build_app(webhooks_admin_test_config()).await;
    seed_ledger_failures(&server, "https://hooks.example.com/alerts", 2);

    let req = Request::get("/admin/webhooks/deliveries?limit=10")
        .body(Body::empty())
        .unwrap();
    let (status, json) = send(app, req).await;

    assert_eq!(status, StatusCode::OK, "body: {json}");
    assert_eq!(json["endpoint"], serde_json::Value::Null);
    let rows = json["deliveries"].as_array().expect("deliveries array");
    assert_eq!(rows.len(), 2, "all-endpoint mode returns every row: {json}");
    assert_eq!(rows[0]["attempt"], 2, "newest first");
    assert_eq!(rows[1]["attempt"], 1);
}

// ---------------------------------------------------------------------------
// Failing rollup: last 3 attempts failed/parked → failing (Req 5.5)
// ---------------------------------------------------------------------------

#[tokio::test]
async fn failing_rollup_marks_endpoint_failing_after_three_failures() {
    let (app, server) = build_app(webhooks_admin_test_config()).await;
    let endpoint_url = "https://hooks.example.com/alerts";

    // Below the window: not failing.
    let (status, json) = send(app.clone(), get_webhooks()).await;
    assert_eq!(status, StatusCode::OK);
    assert_eq!(json["endpoints"][0]["failing"], false);

    seed_ledger_failures(&server, endpoint_url, 3);

    let (status, json) = send(app.clone(), get_webhooks()).await;
    assert_eq!(status, StatusCode::OK);
    assert_eq!(
        json["endpoints"][0]["failing"], true,
        "last 3 failed attempts must mark the endpoint failing: {json}"
    );
    assert_eq!(json["endpoints"][0]["last_outcome"], "failed");
}

#[tokio::test]
async fn failing_rollup_requires_all_three_recent_attempts_failed() {
    let (app, server) = build_app(webhooks_admin_test_config()).await;
    let endpoint_url = "https://hooks.example.com/alerts";

    // Two failures then a SUCCESS (newest) — the most recent outcome is
    // not a failure, so the endpoint is not failing.
    seed_ledger_failures(&server, endpoint_url, 2);
    let ledger = &server.state.webhook_ledger;
    ledger
        .record_attempt("evt-https://hooks.example.com/alerts", endpoint_url, 3, "success", Some(200), None, 3)
        .unwrap();

    let (status, json) = send(app, get_webhooks()).await;
    assert_eq!(status, StatusCode::OK);
    assert_eq!(json["endpoints"][0]["failing"], false);
    assert_eq!(json["endpoints"][0]["last_outcome"], "success");
}

// ---------------------------------------------------------------------------
// Auth: 401 without credentials on all four routes (NFR Security)
// ---------------------------------------------------------------------------

#[tokio::test]
async fn webhooks_admin_routes_require_credentials_when_auth_enabled() {
    let user_env = "WEBHOOKS_ADMIN_TEST_USER";
    let pass_env = "WEBHOOKS_ADMIN_TEST_PASS";
    std::env::set_var(user_env, "admin-user");
    std::env::set_var(pass_env, "admin-pass");

    let mut config = webhooks_admin_test_config();
    config.admin.auth = AdminAuthConfig {
        enabled: true,
        username_env: Some(user_env.to_string()),
        password_env: Some(pass_env.to_string()),
        allow_unauthenticated_non_loopback: false,
    };
    let (app, _server) = build_app(config).await;

    let requests = vec![
        Request::get("/admin/webhooks").body(Body::empty()).unwrap(),
        Request::put("/admin/webhooks")
            .header("content-type", "application/json")
            .body(Body::from("{}"))
            .unwrap(),
        Request::post("/admin/webhooks/test")
            .header("content-type", "application/json")
            .body(Body::from("{}"))
            .unwrap(),
        Request::get("/admin/webhooks/deliveries?endpoint=https%3A%2F%2Fhooks.example.com")
            .body(Body::empty())
            .unwrap(),
    ];
    for req in requests {
        let (status, _) = send(app.clone(), req).await;
        assert_eq!(status, StatusCode::UNAUTHORIZED);
    }

    std::env::remove_var(user_env);
    std::env::remove_var(pass_env);
}
