use axum::{
    extract::ws::{Message, WebSocket, WebSocketUpgrade},
    extract::{Path, Query, State},
    http::{header, StatusCode},
    response::{IntoResponse, Response},
    routing::get,
    Json, Router,
};
use chrono::DateTime;
use rust_embed::Embed;
use serde::Deserialize;

use crate::gateway::AppState;
use crate::logger::LogFilter;

#[derive(Embed)]
#[folder = "src/dashboard/static/"]
struct DashboardAssets;

/// Query parameters for the GET /logs endpoint (Req 16.13, 33.6).
#[derive(Debug, Deserialize)]
struct LogQueryParams {
    from: Option<String>,
    to: Option<String>,
    provider: Option<String>,
    model: Option<String>,
    status_code: Option<u16>,
    trace_id: Option<String>,
    limit: Option<usize>,
}

pub fn dashboard_routes(state: AppState) -> Router<AppState> {
    let _ = state;
    Router::new()
        .route("/ws", get(ws_handler))
        .route("/metrics", get(metrics_handler))
        .route("/errors", get(errors_handler))
        .route("/logs", get(logs_handler))
        .route("/", get(index_handler))
        .route("/{*path}", get(static_handler))
}

async fn ws_handler(ws: WebSocketUpgrade, State(state): State<AppState>) -> impl IntoResponse {
    ws.on_upgrade(move |socket| handle_ws(socket, state))
}
async fn handle_ws(mut socket: WebSocket, state: AppState) {
    let mut subscription = state.loop_detector.events.subscribe();
    for event in subscription.replay {
        let message = serde_json::json!({"type": "loop_detection", "data": event});
        if socket
            .send(Message::Text(message.to_string().into()))
            .await
            .is_err()
        {
            return;
        }
    }

    let mut interval = tokio::time::interval(std::time::Duration::from_secs(1));
    loop {
        tokio::select! {
            event = subscription.receiver.recv() => {
                match event {
                    Ok(event) => {
                        let message = serde_json::json!({"type": "loop_detection", "data": event});
                        if socket.send(Message::Text(message.to_string().into())).await.is_err() {
                            break;
                        }
                    }
                    Err(tokio::sync::broadcast::error::RecvError::Lagged(_)) => continue,
                    Err(tokio::sync::broadcast::error::RecvError::Closed) => break,
                }
            }
            _ = interval.tick() => {
                let snapshot = build_dashboard_snapshot(&state).await;
                let msg = serde_json::json!({"type": "metrics", "data": snapshot});
                if socket.send(Message::Text(msg.to_string().into())).await.is_err() {
                    break;
                }

                let errors = recent_errors(&state, 25);
                let errors_msg = serde_json::json!({"type": "errors", "data": errors});
                if socket.send(Message::Text(errors_msg.to_string().into())).await.is_err() {
                    break;
                }
            }
        }
    }
}

async fn metrics_handler(State(state): State<AppState>) -> impl IntoResponse {
    Json(build_dashboard_snapshot(&state).await)
}

async fn logs_handler(
    State(state): State<AppState>,
    Query(params): Query<LogQueryParams>,
) -> Response {
    let filter = LogFilter {
        trace_id: params.trace_id,
        start_time: params.from.as_deref().and_then(parse_datetime),
        end_time: params.to.as_deref().and_then(parse_datetime),
        model: params.model,
        provider: params.provider,
        status_code: params.status_code,
        limit: params.limit,
    };

    match state.logger.query(filter) {
        Ok(entries) => Json(entries).into_response(),
        Err(e) => {
            tracing::error!("Log query failed: {}", e);
            (
                StatusCode::INTERNAL_SERVER_ERROR,
                Json(serde_json::json!({
                    "error": {
                        "message": "Failed to query logs",
                        "type": "internal_error"
                    }
                })),
            )
                .into_response()
        }
    }
}

async fn errors_handler(State(state): State<AppState>) -> Response {
    Json(recent_errors(&state, 25)).into_response()
}

fn parse_datetime(s: &str) -> Option<DateTime<chrono::Utc>> {
    if let Ok(dt) = DateTime::parse_from_rfc3339(s) {
        return Some(dt.with_timezone(&chrono::Utc));
    }
    if let Ok(nd) = chrono::NaiveDate::parse_from_str(s, "%Y-%m-%d") {
        let ndt = nd.and_hms_opt(0, 0, 0)?;
        return Some(DateTime::from_naive_utc_and_offset(ndt, chrono::Utc));
    }
    None
}
async fn index_handler(State(state): State<AppState>) -> impl IntoResponse {
    serve_index_html(&state)
}

async fn static_handler(Path(path): Path<String>) -> impl IntoResponse {
    serve_embedded(&path)
}

fn serve_embedded(path: &str) -> Response {
    match DashboardAssets::get(path) {
        Some(content) => {
            let mime = mime_from_path(path);
            (
                StatusCode::OK,
                [(header::CONTENT_TYPE, mime)],
                content.data.to_vec(),
            )
                .into_response()
        }
        None => (StatusCode::NOT_FOUND, "Not Found").into_response(),
    }
}

async fn build_dashboard_snapshot(state: &AppState) -> crate::metrics::MetricsSnapshot {
    let mut snapshot = state.metrics.snapshot();
    let cb_states = state.router.get_circuit_breaker_states().await;
    snapshot.circuit_breaker_states = cb_states.clone();
    snapshot.enrich_circuit_breaker_states(&cb_states);
    snapshot
}

fn recent_errors(state: &AppState, limit: usize) -> Vec<crate::logger::LogEntry> {
    match state.logger.query(LogFilter {
        limit: Some(limit * 4),
        ..Default::default()
    }) {
        Ok(entries) => entries
            .into_iter()
            .filter(|entry| entry.status_code >= 400)
            .take(limit)
            .collect(),
        Err(error) => {
            tracing::error!(%error, "Failed to load dashboard error entries");
            Vec::new()
        }
    }
}

fn serve_index_html(state: &AppState) -> Response {
    match DashboardAssets::get("index.html") {
        Some(content) => {
            let mut html = String::from_utf8_lossy(&content.data).into_owned();
            let config = state.config.try_read().expect("config lock poisoned");
            // Inject <base> so relative asset URLs (logo, favicon) resolve under the dashboard path
            let dashboard_base = format!("{}/", config.dashboard.path.trim_end_matches('/'));
            html = html.replace(
                "<head>",
                &format!("<head><base href=\"{}\">", dashboard_base),
            );
            let bootstrap = format!(
                "<script>window.__dashboardBasePath={:?};window.__adminBasePath={:?};window.__dashboardPollIntervalMs={};</script>",
                config.dashboard.path,
                config.admin.path,
                config.dashboard.metrics_update_interval_seconds.saturating_mul(1000)
            );
            html = html.replace("</head>", &(bootstrap + "</head>"));
            (
                StatusCode::OK,
                [(header::CONTENT_TYPE, "text/html; charset=utf-8")],
                html,
            )
                .into_response()
        }
        None => (StatusCode::NOT_FOUND, "Not Found").into_response(),
    }
}

fn mime_from_path(path: &str) -> &'static str {
    match path.rsplit('.').next() {
        Some("html") => "text/html; charset=utf-8",
        Some("css") => "text/css; charset=utf-8",
        Some("js") => "application/javascript; charset=utf-8",
        Some("json") => "application/json",
        Some("png") => "image/png",
        Some("jpg") | Some("jpeg") => "image/jpeg",
        Some("svg") => "image/svg+xml",
        Some("ico") => "image/x-icon",
        _ => "application/octet-stream",
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::config::{
        AdminConfig, CircuitBreakerConfig, Config, ContextConfig, CorsConfig, DashboardConfig,
        ExactCacheConfig, LoggingConfig, ModelGroup, Provider, ProviderModel, RetryConfig,
        ServerConfig, TrayConfig,
    };
    use crate::gateway::GatewayServer;
    use crate::logger::LogEntry;
    use axum::body::Body;
    use axum::http::Request;
    use chrono::{Datelike, Timelike};
    use tower::ServiceExt;

    fn test_config() -> Config {
        Config {
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
                max_connections: 10,
                rate_limit_per_minute: 0,
                custom_headers: Default::default(),
                connection_pool: crate::config::ProviderConnectionPoolConfig::default(),
                budget: None,
                manual_models: vec![],
                global_inference_profile: false,
                cross_region_inference: false,
                custom_vpc_endpoint: false,
                prompt_caching: false,
                reasoning: true,
                codex_base_url_override: None,
                codex_model_override: None,
                instructions_override: None,
                max_rate_limit_cooldown_seconds: None,
            }],
            model_groups: vec![ModelGroup {
                name: "default".to_string(),
                version_fallback_enabled: false,
                models: vec![ProviderModel {
                    provider: "test-provider".to_string(),
                    model: "gpt-4".to_string(),
                    cost_per_million_input_tokens: 0.0,
                    cost_per_million_output_tokens: 0.0,
                    priority: 100,
                }],
            }],
            circuit_breaker: CircuitBreakerConfig::default(),
            retry: RetryConfig::default(),
            logging: LoggingConfig::default(),
            semantic_cache: None,
            exact_cache: ExactCacheConfig::default(),
            prometheus: None,
            context: ContextConfig::default(),
            first_launch_completed: false,
            tray: TrayConfig::default(),
            codex_instructions_url: None,
            streaming: None,
            virtual_keys: Default::default(),
            loop_detection: Default::default(),
            guardrails: None,
        }
    }

    #[test]
    fn test_dashboard_index_embedded() {
        let asset = DashboardAssets::get("index.html");
        assert!(asset.is_some(), "index.html should be embedded");
        let data = asset.unwrap();
        let html = std::str::from_utf8(&data.data).unwrap();
        assert!(html.contains("OBEY-API Dashboard"));
    }

    #[test]
    fn test_dashboard_serve_not_found() {
        let resp = serve_embedded("nonexistent.file");
        assert_eq!(resp.status(), StatusCode::NOT_FOUND);
    }

    #[test]
    fn test_dashboard_serve_index() {
        let resp = serve_embedded("index.html");
        assert_eq!(resp.status(), StatusCode::OK);
    }

    #[test]
    fn test_dashboard_mime_types() {
        assert_eq!(mime_from_path("index.html"), "text/html; charset=utf-8");
        assert_eq!(
            mime_from_path("app.js"),
            "application/javascript; charset=utf-8"
        );
        assert_eq!(mime_from_path("style.css"), "text/css; charset=utf-8");
        assert_eq!(mime_from_path("unknown.xyz"), "application/octet-stream");
    }

    #[test]
    fn test_dashboard_html_has_required_sections() {
        let asset = DashboardAssets::get("index.html").unwrap();
        let html = std::str::from_utf8(&asset.data).unwrap();
        assert!(html.contains("Total Requests"));
        assert!(html.contains("Avg Response Time"));
        assert!(html.contains("Request Rate"));
        assert!(html.contains("Active Requests"));
        assert!(html.contains("Cumulative Cost"));
        assert!(html.contains("Cache Hit Rate"));
        assert!(html.contains("Provider Health"));
        assert!(html.contains("Circuit Breaker"));
        assert!(html.contains("Recent Errors"));
        assert!(html.contains("Log Viewer"));
        assert!(html.contains("WebSocket"));
        assert!(html.contains("conn-dot"));
        assert!(html.contains("Chart"));
    }

    #[test]
    fn test_parse_datetime_rfc3339() {
        let dt = parse_datetime("2024-01-15T10:30:00Z");
        assert!(dt.is_some());
        let dt = dt.unwrap();
        assert_eq!(dt.year(), 2024);
        assert_eq!(dt.month(), 1);
        assert_eq!(dt.day(), 15);
    }

    #[test]
    fn test_parse_datetime_bare_date() {
        let dt = parse_datetime("2024-06-01");
        assert!(dt.is_some());
        let dt = dt.unwrap();
        assert_eq!(dt.year(), 2024);
        assert_eq!(dt.month(), 6);
        assert_eq!(dt.hour(), 0);
    }

    #[test]
    fn test_parse_datetime_invalid() {
        assert!(parse_datetime("not-a-date").is_none());
        assert!(parse_datetime("").is_none());
    }

    #[tokio::test(flavor = "multi_thread", worker_threads = 2)]
    async fn test_dashboard_metrics_enriches_circuit_breaker_state() {
        let server = GatewayServer::new(test_config(), None).await.unwrap();
        let cb = server
            .state
            .router
            .get_circuit_breaker("test-provider:gpt-4")
            .await;
        cb.record_failure().await;
        cb.record_failure().await;
        cb.record_failure().await;

        let app = server.build_router();
        let response = app
            .oneshot(
                Request::get("/dashboard/metrics")
                    .body(Body::empty())
                    .unwrap(),
            )
            .await
            .unwrap();
        assert_eq!(response.status(), StatusCode::OK);

        let body = axum::body::to_bytes(response.into_body(), 64 * 1024)
            .await
            .unwrap();
        let json: serde_json::Value = serde_json::from_slice(&body).unwrap();
        assert!(json["circuit_breaker_states"]
            .as_array()
            .is_some_and(|arr| !arr.is_empty()));
    }

    #[tokio::test]
    async fn test_dashboard_errors_endpoint_returns_failed_logs() {
        let server = GatewayServer::new(test_config(), None).await.unwrap();
        server
            .state
            .logger
            .log(LogEntry {
                trace_id: "trace-1".to_string(),
                timestamp: chrono::Utc::now(),
                method: "POST".to_string(),
                path: "/v1/chat/completions".to_string(),
                model: "gpt-4".to_string(),
                provider: "test-provider".to_string(),
                status_code: 502,
                duration_ms: 120,
                cost: 0.0,
                request_body: None,
                response_body: None,
                requested_model: Some("gpt-4".to_string()),
                responded_model: None,
            })
            .unwrap();

        let app = server.build_router();
        let response = app
            .oneshot(
                Request::get("/dashboard/errors")
                    .body(Body::empty())
                    .unwrap(),
            )
            .await
            .unwrap();
        assert_eq!(response.status(), StatusCode::OK);

        let body = axum::body::to_bytes(response.into_body(), 64 * 1024)
            .await
            .unwrap();
        let json: serde_json::Value = serde_json::from_slice(&body).unwrap();
        assert!(json.as_array().is_some_and(|arr| !arr.is_empty()));
        assert_eq!(json[0]["status_code"], 502);
    }
}
