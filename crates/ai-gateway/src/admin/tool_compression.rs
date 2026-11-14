//! Admin API endpoints for tool compression pipeline management.
//!
//! Provides feedback loop control and description compressor management.

use axum::{
    extract::{Json, State},
    http::StatusCode,
    response::{IntoResponse, Response},
    routing::{get, post},
    Router,
};
use serde::{Deserialize, Serialize};
use serde_json::json;

use crate::gateway::AppState;
use crate::tool_compression::config::CompressionLevel;

// ─── Routes ───────────────────────────────────────────────────────────────────

/// Build tool compression admin routes.
pub fn routes() -> Router<AppState> {
    Router::new()
        .route("/feedback", get(list_feedback_states))
        .route("/feedback/reset", post(reset_feedback))
        .route("/feedback/lock", post(lock_feedback))
        .route("/descriptions/recompute", post(recompute_descriptions))
        .route("/descriptions", get(list_descriptions))
}

// ─── Request/Response types ───────────────────────────────────────────────────

#[derive(Serialize)]
struct FeedbackGroupState {
    group: String,
    error_rate: f32,
    current_level: CompressionLevel,
    locked: bool,
    window_size: usize,
    baseline_rate: Option<f32>,
    recovery_counter: u32,
}

#[derive(Deserialize)]
struct ResetFeedbackRequest {
    /// Model group to reset. If absent, resets all groups.
    #[serde(default)]
    group: Option<String>,
}

#[derive(Deserialize)]
struct LockFeedbackRequest {
    /// Model group to lock (required).
    group: String,
    /// Compression level to lock to (required).
    level: CompressionLevel,
}

#[derive(Serialize)]
struct DescriptionEntry {
    tool_name: String,
    compressed_description: String,
}

#[derive(Deserialize)]
struct RecomputeRequest {
    #[serde(default)]
    tools: Vec<String>,
}

// ─── Handlers ─────────────────────────────────────────────────────────────────

/// GET /admin/tool-compression/feedback — list all group states.
async fn list_feedback_states(State(state): State<AppState>) -> Response {
    let tc_state = &state.tool_compression_state;
    let feedback_loop = &tc_state.feedback_loop;

    let groups: Vec<FeedbackGroupState> = feedback_loop
        .group_names()
        .into_iter()
        .filter_map(|group_name| {
            feedback_loop
                .get_state(&group_name)
                .map(|fs| FeedbackGroupState {
                    group: group_name,
                    error_rate: fs.current_error_rate(),
                    current_level: fs.current_level,
                    locked: fs.locked,
                    window_size: fs.window.len(),
                    baseline_rate: fs.baseline_rate,
                    recovery_counter: fs.recovery_counter,
                })
        })
        .collect();

    (StatusCode::OK, Json(json!({ "groups": groups }))).into_response()
}

/// POST /admin/tool-compression/feedback/reset — Reset feedback state.
///
/// Accepts an optional `group` field in JSON body. If present, resets that
/// group only (404 if invalid). If absent, resets all groups.
async fn reset_feedback(
    State(state): State<AppState>,
    Json(body): Json<ResetFeedbackRequest>,
) -> Response {
    let tc_state = &state.tool_compression_state;
    let feedback_loop = &tc_state.feedback_loop;

    match body.group {
        Some(group) => {
            if feedback_loop.has_group(&group) {
                feedback_loop.reset_group(&group);
                (
                    StatusCode::OK,
                    Json(json!({
                        "status": "reset",
                        "group": group
                    })),
                )
                    .into_response()
            } else {
                let available = feedback_loop.group_names();
                (
                    StatusCode::NOT_FOUND,
                    Json(json!({
                        "error": {
                            "message": format!("Model group '{}' not found", group),
                            "type": "not_found",
                            "available_groups": available
                        }
                    })),
                )
                    .into_response()
            }
        }
        None => {
            let count = feedback_loop.group_names().len();
            feedback_loop.reset_all();
            (
                StatusCode::OK,
                Json(json!({
                    "status": "reset_all",
                    "groups_cleared": count
                })),
            )
                .into_response()
        }
    }
}

/// POST /admin/tool-compression/feedback/lock — Lock a model group to a specific level.
///
/// Accepts required `group` and `level` fields. Returns 404 for invalid group.
async fn lock_feedback(
    State(state): State<AppState>,
    Json(body): Json<LockFeedbackRequest>,
) -> Response {
    let tc_state = &state.tool_compression_state;
    let feedback_loop = &tc_state.feedback_loop;

    if !feedback_loop.has_group(&body.group) {
        let available = feedback_loop.group_names();
        return (
            StatusCode::NOT_FOUND,
            Json(json!({
                "error": {
                    "message": format!("Model group '{}' not found", body.group),
                    "type": "not_found",
                    "available_groups": available
                }
            })),
        )
            .into_response();
    }

    feedback_loop.lock_group_at_level(&body.group, body.level);

    (
        StatusCode::OK,
        Json(json!({
            "status": "locked",
            "group": body.group,
            "level": body.level
        })),
    )
        .into_response()
}

/// POST /admin/tool-compression/descriptions/recompute — trigger recomputation.
async fn recompute_descriptions(
    State(state): State<AppState>,
    Json(body): Json<RecomputeRequest>,
) -> Response {
    let tc_state = &state.tool_compression_state;

    if body.tools.is_empty() {
        // Recompute all
        tc_state.description_compressor.clear();
        (
            StatusCode::OK,
            Json(json!({
                "status": "recomputing_all",
                "message": "All compressed descriptions cleared; will be recomputed on next request"
            })),
        )
            .into_response()
    } else {
        // Recompute specific tools
        for tool_name in &body.tools {
            tc_state.description_compressor.remove(tool_name);
        }
        (
            StatusCode::OK,
            Json(json!({
                "status": "recomputing_subset",
                "tools_cleared": body.tools.len()
            })),
        )
            .into_response()
    }
}

/// GET /admin/tool-compression/descriptions — list current compressed descriptions.
async fn list_descriptions(State(state): State<AppState>) -> Response {
    let tc_state = &state.tool_compression_state;
    let entries: Vec<DescriptionEntry> = tc_state
        .description_compressor
        .iter()
        .map(|entry| DescriptionEntry {
            tool_name: entry.key().clone(),
            compressed_description: entry.value().clone(),
        })
        .collect();

    (StatusCode::OK, Json(json!({ "descriptions": entries }))).into_response()
}

#[cfg(test)]
mod disabled_providers_config_tests {
    use crate::config::{
        AdminConfig, Config, ContextConfig, CorsConfig, DashboardConfig, LoggingConfig,
        ModelGroup, Provider, ProviderModel, RetryConfig, ServerConfig,
    };
    use crate::gateway::GatewayServer;
    use axum::body::Body;
    use axum::http::Request;
    use tower::ServiceExt;

    fn test_config() -> Config {
        Config {
            prompt_library: Default::default(),
            security: Default::default(),
            cache_aware_routing: Default::default(),
            server: ServerConfig {
                host: "127.0.0.1".to_string(),
                port: 8080,
                request_timeout_seconds: 30,
                max_request_size_mb: 10,
            },
            tls: None,
            admin: AdminConfig::default(),
            dashboard: DashboardConfig::default(),
            cors: CorsConfig::default(),
            providers: vec![Provider {
                name: "test-provider".to_string(),
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
                connection_pool: crate::config::ProviderConnectionPoolConfig::default(),
                budget: None,
                manual_models: vec![],
                global_inference_profile: false,
                cross_region_inference: false,
                custom_vpc_endpoint: false,
                prompt_caching: false,
                compression: None,
                memory: None,
                reasoning: true,
                codex_base_url_override: None,
                codex_model_override: None,
                instructions_override: None,
                max_rate_limit_cooldown_seconds: None,
            }],
            model_groups: vec![ModelGroup {
                name: "default".to_string(),
                version_fallback_enabled: false,
                compression: None,
                structured_output: None,
                memory: None,
                models: vec![ProviderModel {
                    cache_support: None,
                    cache_min_tokens: None,
                    cost_per_million_cache_read_input_tokens: None,
                    cost_per_million_cache_creation_input_tokens: None,
                    provider: "test-provider".to_string(),
                    model: "gpt-4".to_string(),
                    cost_per_million_input_tokens: 0.0,
                    cost_per_million_output_tokens: 0.0,
                    priority: 100,
                    structured_output_passthrough: None,
                    tier: None,
                    context_window: 0,
                    specializations: vec![],
                    cost_per_million_reasoning_tokens: None,
                    reasoning_family: None,
                    reasoning_parameter: None,
                }],
            }],
            circuit_breaker: Default::default(),
            retry: RetryConfig::default(),
            logging: LoggingConfig::default(),
            semantic_cache: None,
            exact_cache: Default::default(),
            prometheus: None,
            context: ContextConfig::default(),
            compression: Default::default(),
            structured_output: None,
            first_launch_completed: false,
            tray: Default::default(),
            codex_instructions_url: None,
            streaming: None,
            virtual_keys: Default::default(),
            loop_detection: Default::default(),
            guardrails: None,
            webhooks: Default::default(),
            opentelemetry: Default::default(),
            tool_compression: Default::default(),
            smart_routing: Default::default(),
            memory: None,
            xhigh_models_allowlist: Default::default(),
            reasoning_models_allowlist: Default::default(),
            codex_search: None,
            reasoning_compat: Default::default(),
        }
    }

    async fn response_json(response: axum::response::Response) -> serde_json::Value {
        let body = axum::body::to_bytes(response.into_body(), 64 * 1024)
            .await
            .unwrap();
        serde_json::from_slice(&body).unwrap()
    }

    /// The dashboard config GET serializes `ToolCompressionConfig` wholesale,
    /// so `disabled_providers` must appear (absent config → `[]`).
    #[tokio::test]
    async fn tool_compression_config_get_includes_disabled_providers_default_empty() {
        let config_dir = std::env::temp_dir().join(format!(
            "obey-tc-disabled-providers-{}",
            uuid::Uuid::new_v4().simple()
        ));
        std::fs::create_dir_all(&config_dir).unwrap();
        let server = GatewayServer::new(test_config(), Some(config_dir.join("config.yaml")))
            .await
            .unwrap();

        let response = server
            .build_router()
            .oneshot(
                Request::get("/dashboard/tool-compression/config")
                    .body(Body::empty())
                    .unwrap(),
            )
            .await
            .unwrap();
        assert_eq!(response.status(), axum::http::StatusCode::OK);
        let body = response_json(response).await;
        assert!(
            body.get("disabled_providers").is_some_and(serde_json::Value::is_array),
            "disabled_providers key must be present even when empty (never omitted/null): {body}"
        );
        assert_eq!(
            body["disabled_providers"],
            serde_json::json!([]),
            "GET must include disabled_providers, defaulting to []: {body}"
        );

        let _ = std::fs::remove_dir_all(&config_dir);
    }

    /// Dashboard GET with a seeded list: the handler serializes
    /// `ToolCompressionConfig` wholesale, so a config created with
    /// `disabled_providers: ["alpha", "beta"]` must surface it verbatim.
    /// Task 8 of spec per-provider-tool-compression-toggle.
    #[tokio::test]
    async fn dashboard_tool_compression_config_get_returns_seeded_disabled_providers() {
        let config_dir = std::env::temp_dir().join(format!(
            "obey-tc-disabled-providers-seeded-{}",
            uuid::Uuid::new_v4().simple()
        ));
        std::fs::create_dir_all(&config_dir).unwrap();
        let mut seeded = test_config();
        seeded.tool_compression.disabled_providers =
            vec!["alpha".to_string(), "beta".to_string()];
        let server = GatewayServer::new(seeded, Some(config_dir.join("config.yaml")))
            .await
            .unwrap();

        let response = server
            .build_router()
            .oneshot(
                Request::get("/dashboard/tool-compression/config")
                    .body(Body::empty())
                    .unwrap(),
            )
            .await
            .unwrap();
        assert_eq!(response.status(), axum::http::StatusCode::OK);
        let body = response_json(response).await;
        assert!(
            body.get("disabled_providers").is_some_and(serde_json::Value::is_array),
            "disabled_providers key must be present (never omitted): {body}"
        );
        assert_eq!(body["disabled_providers"], serde_json::json!(["alpha", "beta"]));

        let _ = std::fs::remove_dir_all(&config_dir);
    }

    /// Admin-panel save path round trip: PUT /admin/config with
    /// `tool_compression.disabled_providers` must persist the list to the
    /// live config and the on-disk YAML, and a follow-up dashboard GET must
    /// return it.
    #[tokio::test]
    async fn admin_config_put_round_trips_disabled_providers() {
        let config_dir = std::env::temp_dir().join(format!(
            "obey-tc-disabled-providers-put-{}",
            uuid::Uuid::new_v4().simple()
        ));
        std::fs::create_dir_all(&config_dir).unwrap();
        let config_path = config_dir.join("config.yaml");
        let mut initial = test_config();
        initial.server.port = 8080;
        let server = GatewayServer::new(initial, Some(config_path.clone()))
            .await
            .unwrap();

        // GET the config as the panel does, set the list, PUT it back.
        let response = server
            .build_router()
            .oneshot(
                Request::get("/admin/config")
                    .body(Body::empty())
                    .unwrap(),
            )
            .await
            .unwrap();
        assert_eq!(response.status(), axum::http::StatusCode::OK);
        let mut config = response_json(response).await;

        let expected = serde_json::json!(["test-provider", "other-provider"]);
        config["tool_compression"]["disabled_providers"] = expected.clone();

        let put = axum::http::Request::builder()
            .method("PUT")
            .uri("/admin/config")
            .header("content-type", "application/json")
            .body(Body::from(config.to_string()))
            .unwrap();
        let response = server.build_router().oneshot(put).await.unwrap();
        let status = response.status();
        let body = response_json(response).await;
        assert_eq!(
            status,
            axum::http::StatusCode::OK,
            "PUT /admin/config failed: {body}"
        );

        // Live config carries the list.
        {
            let live = server.state.config.read().await;
            assert_eq!(
                live.tool_compression.disabled_providers,
                vec!["test-provider".to_string(), "other-provider".to_string()],
                "live config lost disabled_providers"
            );
        }

        // On-disk YAML persists it.
        let saved: Config =
            serde_yaml::from_str(&std::fs::read_to_string(&config_path).unwrap()).unwrap();
        assert_eq!(
            saved.tool_compression.disabled_providers,
            vec!["test-provider".to_string(), "other-provider".to_string()],
            "persisted YAML lost disabled_providers"
        );

        // Follow-up dashboard GET returns it.
        let response = server
            .build_router()
            .oneshot(
                Request::get("/dashboard/tool-compression/config")
                    .body(Body::empty())
                    .unwrap(),
            )
            .await
            .unwrap();
        assert_eq!(response.status(), axum::http::StatusCode::OK);
        let body = response_json(response).await;
        assert_eq!(body["disabled_providers"], expected);

        let _ = std::fs::remove_dir_all(&config_dir);
    }
}
