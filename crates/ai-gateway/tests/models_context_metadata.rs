//! Integration tests for `/v1/models` context-capacity metadata and the
//! `GET /v1/models/{model_id}` retrieve endpoint (spec
//! models-list-context-metadata, Task 6).
//!
//! Drives the full Axum router via `tower::ServiceExt::oneshot()` (no port
//! binding, per repo conventions) with per-test isolated SQLite databases
//! via `common::isolate_databases`. No upstream provider is contacted —
//! the models endpoints are pure config projections.
//!
//! Coverage map (Task 6 list, numbered):
//! 1. list carries declared per-model `context_window` and the group MAX
//!    aggregation (128000 + 200000 candidates → group advertises 200000)
//! 2. unknown-capacity entries keep today's exact shape (no
//!    context_window / max_completion_tokens keys; id/object/owned_by
//!    unchanged)
//! 3. GET /v1/models/{id}: 200 + capacity for a GROUP id, 200 for a
//! per-MODEL id, 404 + OpenAI-style error JSON for an unknown id;
//! 200 for a slash-containing NVIDIA NIM fallback-catalog id
//! 4. 404 under virtual-key `model_access` restriction — status and body
//! shape identical to an unknown id
//! 5. Anthropic-shape regression: `anthropic-version` header / `?beta=true`
//! keep the pre-existing shape (no capacity keys added); the beta=true
//! response equals the header-triggered response for the same config
//! 6. hot-reload: after POST /admin/config/reload the list reflects the
//! newly configured window

use ai_gateway::config::*;
use ai_gateway::gateway::GatewayServer;
use ai_gateway::virtual_keys::models::CreateKeyParams;
use axum::body::Body;
use axum::http::{Request, StatusCode};
use serde_json::{json, Value};
use tower::ServiceExt;

use crate::common;

// ---------------------------------------------------------------------------
// Harness
// ---------------------------------------------------------------------------

/// Group with two declared-capacity candidates (MAX aggregation target).
const CAP_GROUP: &str = "cap-group";
/// Group with no declared capacity anywhere (shape-regression target).
const PLAIN_GROUP: &str = "plain-group";
const MODEL_A: &str = "model-a";
const MODEL_B: &str = "model-b";

fn provider() -> Provider {
    Provider {
        name: "p".to_string(),
        provider_type: "openai".to_string(),
        base_url: Some("http://localhost:1234".to_string()),
        api_key_env: None,
        api_key_encrypted: None,
        api_secret_env: None,
        api_secret_encrypted: None,
        auth_method: None,
        resolved_api_key: None,
        resolved_api_secret: None,
        region: None,
        timeout_seconds: 30,
        ttfb_timeout_seconds: None,
        total_timeout_seconds: None,
        buffered_upstream_streaming: None,
        rate_limit_max_wait_ms: None,
        max_connections: 10,
        rate_limit_per_minute: 0,
        custom_headers: Default::default(),
        user_agent: None,
        connection_pool: ProviderConnectionPoolConfig::default(),
        budget: None,
        manual_models: vec![],
        global_inference_profile: false,
        cross_region_inference: false,
        custom_vpc_endpoint: false,
        prompt_caching: false,
        compression: None,
        reasoning: true,
        codex_base_url_override: None,
        codex_model_override: None,
        instructions_override: None,
        max_rate_limit_cooldown_seconds: None,
        memory: None,
    }
}

fn group_model(model: &str, context_window: u32) -> ProviderModel {
    ProviderModel {
        provider: "p".to_string(),
        model: model.to_string(),
        cost_per_million_input_tokens: 30.0,
        cost_per_million_output_tokens: 60.0,
        cache_support: None,
        cache_min_tokens: None,
        cost_per_million_cache_read_input_tokens: None,
        cost_per_million_cache_creation_input_tokens: None,
        priority: 100,
        structured_output_passthrough: None,
        tier: None,
        context_window,
        specializations: vec![],
        cost_per_million_reasoning_tokens: None,
        reasoning_family: None,
        reasoning_parameter: None,
    }
}

fn base_config(enforcement: EnforcementMode) -> Config {
    let mut config = Config {
        prompt_library: Default::default(),
        security: Default::default(),
        cache_aware_routing: Default::default(),
        server: ServerConfig {
            host: "127.0.0.1".to_string(),
            port: 0,
            request_timeout_seconds: 30,
            max_request_size_mb: 10,
        },
        tls: None,
        admin: AdminConfig::default(),
        dashboard: DashboardConfig::default(),
        cors: CorsConfig::default(),
        // An `nvidia_nim` provider triggers the built-in fallback catalog in
        // `build_models_entries` (slash-containing ids, no capacity keys).
        providers: vec![provider(), {
            let mut nim = provider();
            nim.name = "nim".to_string();
            nim.provider_type = "nvidia_nim".to_string();
            nim
        }],
        model_groups: vec![
            ModelGroup {
                name: CAP_GROUP.to_string(),
                version_fallback_enabled: false,
                compression: None,
                structured_output: None,
                memory: None,
                models: vec![
                    group_model(MODEL_A, 128_000),
                    group_model(MODEL_B, 200_000),
                ],
            },
            ModelGroup {
                name: PLAIN_GROUP.to_string(),
                version_fallback_enabled: false,
                compression: None,
                structured_output: None,
                memory: None,
                models: vec![group_model("gpt-4o", 0)],
            },
        ],
        circuit_breaker: CircuitBreakerConfig::default(),
        retry: RetryConfig::default(),
        logging: LoggingConfig::default(),
        semantic_cache: None,
        exact_cache: ExactCacheConfig::default(),
        prometheus: None,
        context: ContextConfig::default(),
        compression: Default::default(),
        first_launch_completed: false,
        tray: TrayConfig::default(),
        codex_instructions_url: None,
        streaming: None,
        virtual_keys: VirtualKeysConfig {
            enforcement,
            database_path: String::new(),
        },
        loop_detection: Default::default(),
        guardrails: None,
        tool_compression: Default::default(),
        smart_routing: Default::default(),
        xhigh_models_allowlist: Default::default(),
        reasoning_models_allowlist: Default::default(),
        codex_search: None,
        structured_output: None,
        memory: None,
        reasoning_compat: Default::default(),
        webhooks: Default::default(),
        opentelemetry: Default::default(),
    };
    common::isolate_databases(&mut config);
    config
}

struct TestServer {
    server: GatewayServer,
}

impl TestServer {
    async fn new() -> Self {
        Self::with_enforcement(EnforcementMode::Disabled).await
    }

    async fn with_enforcement(enforcement: EnforcementMode) -> Self {
        let server = GatewayServer::new(base_config(enforcement), None)
            .await
            .unwrap();
        Self { server }
    }

    fn router(&self) -> axum::Router {
        self.server.build_router()
    }
}

/// GET a URI through the router, returning (status, parsed JSON body).
async fn get(app: axum::Router, uri: &str, headers: &[(&str, &str)]) -> (StatusCode, Value) {
    let mut builder = Request::get(uri);
    for (k, v) in headers {
        builder = builder.header(*k, *v);
    }
    let response = app
        .oneshot(builder.body(Body::empty()).unwrap())
        .await
        .unwrap();
    let status = response.status();
    let bytes = axum::body::to_bytes(response.into_body(), 1024 * 1024)
        .await
        .unwrap();
    (status, serde_json::from_slice(&bytes).unwrap())
}

/// Find a model entry by id inside a /v1/models `data` array.
fn entry<'a>(data: &'a [Value], id: &str) -> &'a Value {
    data.iter()
        .find(|m| m["id"] == json!(id))
        .unwrap_or_else(|| panic!("no entry '{id}' in data: {data:?}"))
}

// ---------------------------------------------------------------------------
// 1. list carries per-model + group-MAX context_window
// ---------------------------------------------------------------------------

#[tokio::test]
async fn models_list_carries_declared_and_group_max_context_window() {
    let test = TestServer::new().await;
    let (status, body) = get(test.router(), "/v1/models", &[]).await;

    assert_eq!(status, StatusCode::OK);
    assert_eq!(body["object"], "list");
    let data = body["data"].as_array().expect("data array");

    // Per-model entries carry their configured window.
    assert_eq!(entry(data, MODEL_A)["context_window"], json!(128_000));
    assert_eq!(entry(data, MODEL_B)["context_window"], json!(200_000));

    // Group entry aggregates MAX over positive candidate windows.
    assert_eq!(entry(data, CAP_GROUP)["context_window"], json!(200_000));
}

// ---------------------------------------------------------------------------
// 2. unknown-capacity entries keep today's exact shape
// ---------------------------------------------------------------------------

#[tokio::test]
async fn unknown_capacity_entries_serialize_without_capacity_keys() {
    let test = TestServer::new().await;
    let (status, body) = get(test.router(), "/v1/models", &[]).await;

    assert_eq!(status, StatusCode::OK);
    let data = body["data"].as_array().expect("data array");

    let group_entry = entry(data, PLAIN_GROUP);
    assert!(group_entry.get("context_window").is_none(), "{group_entry}");
    assert!(
        group_entry.get("max_completion_tokens").is_none(),
        "{group_entry}"
    );
    assert_eq!(group_entry["id"], json!(PLAIN_GROUP));
    assert_eq!(group_entry["object"], "model");
    assert_eq!(group_entry["owned_by"], "gateway");

    let model_entry = entry(data, "gpt-4o");
    assert!(model_entry.get("context_window").is_none(), "{model_entry}");
    assert!(
        model_entry.get("max_completion_tokens").is_none(),
        "{model_entry}"
    );
    assert_eq!(model_entry["id"], "gpt-4o");
    assert_eq!(model_entry["object"], "model");
    assert_eq!(model_entry["owned_by"], "p");
}

// ---------------------------------------------------------------------------
// 3. GET /v1/models/{id}: group id, model id, unknown id
// ---------------------------------------------------------------------------

#[tokio::test]
async fn get_model_returns_group_and_model_entries_and_404_for_unknown() {
    let test = TestServer::new().await;

    // GROUP id → 200 + aggregated capacity.
    let (status, body) = get(test.router(), &format!("/v1/models/{CAP_GROUP}"), &[]).await;
    assert_eq!(status, StatusCode::OK, "body: {body}");
    assert_eq!(body["id"], json!(CAP_GROUP));
    assert_eq!(body["object"], "model");
    assert_eq!(body["owned_by"], "gateway");
    assert_eq!(body["context_window"], json!(200_000));
    assert!(body.get("data").is_none(), "must be a bare model object");

    // Per-MODEL id → 200 + its own capacity.
    let (status, body) = get(test.router(), &format!("/v1/models/{MODEL_A}"), &[]).await;
    assert_eq!(status, StatusCode::OK, "body: {body}");
    assert_eq!(body["id"], json!(MODEL_A));
    assert_eq!(body["owned_by"], "p");
    assert_eq!(body["context_window"], json!(128_000));

    // Unknown id → 404 with an OpenAI-style error object.
    let (status, body) = get(test.router(), "/v1/models/no-such-model", &[]).await;
    assert_eq!(status, StatusCode::NOT_FOUND, "body: {body}");
    assert_eq!(body["error"]["type"], "invalid_request_error");
    assert!(
        body["error"]["message"]
            .as_str()
            .is_some_and(|m| m.contains("no-such-model")),
        "must name the model: {body}"
    );
}

// ---------------------------------------------------------------------------
// 3b. slash-containing ids (NVIDIA NIM catalog) resolve via the catch-all
// ---------------------------------------------------------------------------

/// A `nvidia_nim` provider triggers the built-in fallback catalog in
/// `build_models_entries`, whose ids contain `/` (e.g. `meta/…`).
#[tokio::test]
async fn get_model_resolves_slash_containing_nim_catalog_ids() {
    let test = TestServer::new().await;
    // Pick a real id straight from the built-in NIM fallback catalog.
    let slash_id =
        ai_gateway::providers::nvidia_nim::NVIDIA_NIM_FALLBACK_MODELS[0]
            .id
            .to_string();
    assert!(slash_id.contains('/'), "catalog id must contain '/': {slash_id}");

    // Slash-containing id → 200 with exactly that id (no 404, no split path).
    let (status, body) = get(test.router(), &format!("/v1/models/{slash_id}"), &[]).await;
    assert_eq!(status, StatusCode::OK, "body: {body}");
    assert_eq!(body["id"], json!(slash_id));
    assert_eq!(body["object"], "model");
    assert!(body.get("data").is_none(), "must be a bare model object");

    // The list also carries the entry, so the retrieve result is consistent.
    let (list_status, list) = get(test.router(), "/v1/models", &[]).await;
    assert_eq!(list_status, StatusCode::OK);
    let data = list["data"].as_array().expect("data array");
    assert!(
        data.iter().any(|m| m["id"] == json!(slash_id)),
        "list must contain '{slash_id}'"
    );
}

// ---------------------------------------------------------------------------
// 4. vk model_access restriction → 404 indistinguishable from unknown id
// ---------------------------------------------------------------------------

#[tokio::test]
async fn get_model_returns_404_under_vk_model_access_restriction() {
    let test = TestServer::with_enforcement(EnforcementMode::Required).await;
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
            model_access: Some(vec![MODEL_B.to_string()]),
            expires_in: None,
            loop_detection: None,
        })
        .await
        .unwrap();
    let auth_header = format!("Bearer {}", created.key);
    let auth = &[("authorization", auth_header.as_str())];

    // The restricted id resolves to 404 …
    let (hidden_status, hidden_body) =
        get(test.router(), &format!("/v1/models/{CAP_GROUP}"), auth).await;
    assert_eq!(hidden_status, StatusCode::NOT_FOUND, "body: {hidden_body}");
    assert_eq!(hidden_body["error"]["type"], "invalid_request_error");

    // … identical status and body shape as a truly unknown id.
    let (unknown_status, unknown_body) =
        get(test.router(), "/v1/models/never-configured", auth).await;
    assert_eq!(unknown_status, hidden_status);
    assert_eq!(unknown_body["error"]["type"], hidden_body["error"]["type"]);
    assert!(
        unknown_body["error"]["message"].as_str().is_some()
            && hidden_body["error"]["message"].as_str().is_some()
    );

    // Sanity: an allowed id still resolves to 200 under the same key.
    let (allowed_status, allowed_body) =
        get(test.router(), &format!("/v1/models/{MODEL_B}"), auth).await;
    assert_eq!(allowed_status, StatusCode::OK, "body: {allowed_body}");
    assert_eq!(allowed_body["context_window"], json!(200_000));
}

// ---------------------------------------------------------------------------
// 5. Anthropic-shape regression: no capacity keys leak into that shape
// ---------------------------------------------------------------------------

#[tokio::test]
async fn anthropic_shape_models_list_has_no_capacity_keys() {
    let test = TestServer::new().await;

    // anthropic-version header → Anthropic shape, unchanged.
    let (status, header_body) = get(
        test.router(),
        "/v1/models",
        &[("anthropic-version", "2023-06-01")],
    )
    .await;
    assert_eq!(status, StatusCode::OK, "body: {header_body}");
    assert_eq!(header_body["has_more"], json!(false));
    assert!(header_body.get("object").is_none(), "no OpenAI 'object': {header_body}");
    let data = header_body["data"].as_array().expect("data array");
    assert!(!data.is_empty());
    for m in data {
        assert_eq!(m["type"], "model");
        assert!(m["id"].as_str().is_some());
        assert!(m["display_name"].as_str().is_some());
        assert!(m["created_at"].as_str().is_some());
        assert!(m.get("context_window").is_none(), "capacity leaked: {m}");
        assert!(
            m.get("max_completion_tokens").is_none(),
            "capacity leaked: {m}"
        );
    }
    // Known config ids survive the shape switch.
    assert!(data.iter().any(|m| m["id"] == json!(CAP_GROUP)));

    // ?beta=true query → the same Anthropic response as the header trigger
    // for the same config. `created_at` is a per-request wall-clock stamp,
    // so normalize it out before full Value equality; the comparison still
    // proves identical ids, shape, and order (and the header case above
    // proved data non-empty with known ids, so this cannot pass vacuously).
    let (status, beta_body) = get(test.router(), "/v1/models?beta=true", &[]).await;
    assert_eq!(status, StatusCode::OK, "body: {beta_body}");
    let beta_normalized = normalize_created_at(beta_body);
    let header_normalized = normalize_created_at(header_body);
    assert_eq!(beta_normalized, header_normalized);
    assert!(
        !beta_normalized["data"]
            .as_array()
            .expect("data array")
            .is_empty()
    );
}

/// Replace every data entry's `created_at` with a fixed placeholder so two
/// Anthropic-shape responses captured at different times compare equal.
fn normalize_created_at(mut body: Value) -> Value {
    if let Some(data) = body["data"].as_array_mut() {
        for m in data {
            if m.get("created_at").is_some() {
                m["created_at"] = json!("<normalized>");
            }
        }
    }
    body
}

// ---------------------------------------------------------------------------
// 6. hot-reload updates the advertised window
// ---------------------------------------------------------------------------

#[tokio::test]
async fn config_reload_updates_advertised_context_window() {
    let temp = tempfile::tempdir().expect("temp dir for reload config");
    let config_path = temp.path().join("config.yaml");

    // Config v1: model-a advertises 128000. The reload handler validates
    // the on-disk YAML, so the port must be non-zero (unlike the in-memory
    // oneshot tests, which never bind).
    let mut config_v1 = base_config(EnforcementMode::Disabled);
    config_v1.server.port = 18099;
    std::fs::write(
        &config_path,
        serde_yaml::to_string(&config_v1).expect("serialize config v1"),
    )
    .expect("write config v1");
    let test = TestServer {
        server: GatewayServer::new(config_v1.clone(), Some(config_path.clone()))
            .await
            .expect("gateway starts with config v1"),
    };

    let (status, body) = get(test.router(), &format!("/v1/models/{MODEL_A}"), &[]).await;
    assert_eq!(status, StatusCode::OK);
    assert_eq!(body["context_window"], json!(128_000));

    // Config v2: model-a advertises 256000.
    let mut config_v2 = config_v1.clone();
    config_v2.model_groups[0].models[0].context_window = 256_000;
    std::fs::write(
        &config_path,
        serde_yaml::to_string(&config_v2).expect("serialize config v2"),
    )
    .expect("write config v2");

    let reload = test
        .router()
        .oneshot(
            Request::post("/admin/config/reload")
                .body(Body::empty())
                .unwrap(),
        )
        .await
        .expect("reload endpoint response");
    assert_eq!(reload.status(), StatusCode::OK);

    // Post-reload list + retrieve reflect the new window (group MAX too).
    let (status, body) = get(test.router(), &format!("/v1/models/{MODEL_A}"), &[]).await;
    assert_eq!(status, StatusCode::OK);
    assert_eq!(body["context_window"], json!(256_000));

    let (status, list) = get(test.router(), "/v1/models", &[]).await;
    assert_eq!(status, StatusCode::OK);
    let data = list["data"].as_array().expect("data array");
    assert_eq!(entry(data, MODEL_A)["context_window"], json!(256_000));
    assert_eq!(entry(data, CAP_GROUP)["context_window"], json!(256_000));
}
