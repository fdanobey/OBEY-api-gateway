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
use std::{collections::VecDeque, sync::Mutex};
use tokio::sync::broadcast;

use crate::compression::stats::{
    sanitize_operational_metadata, CompressionStats, MAX_ENGINE_LABEL_LEN, MAX_MODEL_LEN,
    MAX_PROVIDER_LEN, MAX_REQUEST_ID_LEN,
};
use crate::gateway::AppState;
use crate::logger::LogFilter;

const COMPRESSION_EVENT_CAPACITY: usize = 100;
const COMPRESSION_REPLAY_CAPACITY: usize = 100;

/// Bounded live and replay delivery for content-free compression statistics.
#[derive(Debug)]
pub struct CompressionEventHub {
    sender: broadcast::Sender<CompressionStats>,
    replay: Mutex<VecDeque<CompressionStats>>,
}

impl CompressionEventHub {
    pub fn new() -> Self {
        let (sender, _) = broadcast::channel(COMPRESSION_EVENT_CAPACITY);
        Self {
            sender,
            replay: Mutex::new(VecDeque::with_capacity(COMPRESSION_REPLAY_CAPACITY)),
        }
    }

    pub fn publish(&self, stats: CompressionStats) {
        let stats = sanitize_compression_stats(stats);
        let mut replay = self
            .replay
            .lock()
            .expect("compression event replay mutex poisoned");
        if replay.len() == COMPRESSION_REPLAY_CAPACITY {
            replay.pop_front();
        }
        replay.push_back(stats.clone());
        let _ = self.sender.send(stats);
    }

    pub fn subscribe(&self) -> CompressionEventSubscription {
        let replay = self
            .replay
            .lock()
            .expect("compression event replay mutex poisoned");
        let receiver = self.sender.subscribe();
        CompressionEventSubscription {
            replay: replay.iter().cloned().collect(),
            receiver,
        }
    }
}

impl Default for CompressionEventHub {
    fn default() -> Self {
        Self::new()
    }
}

pub struct CompressionEventSubscription {
    pub replay: Vec<CompressionStats>,
    pub receiver: broadcast::Receiver<CompressionStats>,
}

fn sanitize_compression_stats(mut stats: CompressionStats) -> CompressionStats {
    stats.request_id = sanitize_operational_metadata(&stats.request_id, MAX_REQUEST_ID_LEN);
    stats.provider = sanitize_operational_metadata(&stats.provider, MAX_PROVIDER_LEN);
    stats.model = sanitize_operational_metadata(&stats.model, MAX_MODEL_LEN);
    stats.engines_applied = stats
        .engines_applied
        .into_iter()
        .map(|engine| sanitize_operational_metadata(&engine, MAX_ENGINE_LABEL_LEN))
        .collect();
    for result in &mut stats.engine_results {
        result.engine_name =
            sanitize_operational_metadata(&result.engine_name, MAX_ENGINE_LABEL_LEN);
    }
    stats
}

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
    compression_level: Option<String>,
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
    let mut compression_subscription = state.compression_events.subscribe();
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
    for stats in compression_subscription.replay {
        if send_compression_event(&mut socket, stats).await.is_err() {
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
            event = compression_subscription.receiver.recv() => {
                match event {
                    Ok(stats) => {
                        if send_compression_event(&mut socket, stats).await.is_err() {
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

async fn send_compression_event(
    socket: &mut WebSocket,
    stats: CompressionStats,
) -> Result<(), axum::Error> {
    let message = compression_message(stats);
    socket.send(Message::Text(message.to_string().into())).await
}

fn compression_message(stats: CompressionStats) -> serde_json::Value {
    serde_json::json!({"type": "compression", "data": stats})
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
        compression_level: params.compression_level,
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
    use crate::compression::{stats::CompressionEngineStats, CompressionLevel};
    use crate::config::{
        AdminConfig, CircuitBreakerConfig, Config, ContextConfig, CorsConfig, DashboardConfig,
        ExactCacheConfig, LoggingConfig, ModelGroup, Provider, ProviderModel, RetryConfig,
        ServerConfig, TrayConfig,
    };
    use crate::gateway::GatewayServer;
    use crate::logger::{CompressionLogMetadata, LogEntry};
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
                compression: None,
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
            compression: Default::default(),
            first_launch_completed: false,
            tray: TrayConfig::default(),
            codex_instructions_url: None,
            streaming: None,
            virtual_keys: Default::default(),
            loop_detection: Default::default(),
            guardrails: None,
        }
    }

    fn compression_stats(request_id: impl Into<String>) -> CompressionStats {
        CompressionStats {
            request_id: request_id.into(),
            level: CompressionLevel::Standard,
            engines_applied: vec!["standard".to_owned()],
            original_tokens: 1_000,
            compressed_tokens: 600,
            savings_percent: 40.0,
            compression_time_ms: 12,
            auto_triggered: true,
            cache_downgrade_applied: false,
            tool_definitions_tokens_saved: 25,
            caveman_applied: false,
            timed_out: false,
            error: false,
            provider: "test-provider".to_owned(),
            model: "gpt-4".to_owned(),
            engine_results: vec![CompressionEngineStats {
                engine_name: "standard".to_owned(),
                tokens_before: 1_000,
                tokens_after: 600,
                tokens_saved: 400,
                savings_percent: 40.0,
                duration_ms: 12,
                applied: true,
            }],
        }
    }

    #[tokio::test]
    async fn compression_event_hub_replays_bounded_ring_and_live_events() {
        let hub = CompressionEventHub::new();
        for index in 0..(COMPRESSION_REPLAY_CAPACITY + 5) {
            hub.publish(compression_stats(format!("request-{index}")));
        }

        let mut subscription = hub.subscribe();
        assert_eq!(subscription.replay.len(), COMPRESSION_REPLAY_CAPACITY);
        assert_eq!(subscription.replay[0].request_id, "request-5");
        assert_eq!(
            subscription.replay.last().unwrap().request_id,
            format!("request-{}", COMPRESSION_REPLAY_CAPACITY + 4)
        );

        hub.publish(compression_stats("request-live"));
        let live = subscription.receiver.recv().await.unwrap();
        assert_eq!(live.request_id, "request-live");
    }

    #[test]
    fn compression_event_hub_replay_is_reusable_and_sanitized() {
        let hub = CompressionEventHub::default();
        let mut stats = compression_stats("Bearer replay-secret");
        stats.provider = "https://user:provider-secret@example.com".to_owned();
        stats.model = "sk-model-secret".to_owned();
        stats.engines_applied = vec!["Bearer engine-secret".to_owned()];
        stats.engine_results[0].engine_name = "AKIAENGINESECRET".to_owned();
        hub.publish(stats);

        let first = hub.subscribe();
        let second = hub.subscribe();
        assert_eq!(first.replay, second.replay);
        assert_eq!(first.replay.len(), 1);

        let json = serde_json::to_string(&first.replay[0]).unwrap();
        for secret in [
            "replay-secret",
            "provider-secret",
            "model-secret",
            "engine-secret",
            "AKIAENGINESECRET",
        ] {
            assert!(!json.contains(secret));
        }
        for content_field in ["payload", "content", "messages", "prompt", "request_body"] {
            assert!(first.replay[0]
                .to_json_value()
                .unwrap()
                .get(content_field)
                .is_none());
        }
    }

    #[test]
    fn compression_websocket_message_has_expected_safe_shape() {
        let mut stats = compression_stats("request-123");
        stats.model = "sk-websocket-secret".to_owned();

        let message = compression_message(stats);

        assert_eq!(message["type"], "compression");
        assert_eq!(message["data"]["request_id"], "request-123");
        assert_eq!(message["data"]["original_tokens"], 1_000);
        assert_eq!(message["data"]["compressed_tokens"], 600);
        assert!(message.get("payload").is_none());
        assert!(message["data"].get("content").is_none());
        assert!(!message.to_string().contains("websocket-secret"));
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
    async fn dashboard_logs_filters_by_compression_level_and_provider() {
        let mut config = test_config();
        let database = tempfile::NamedTempFile::new().unwrap();
        config.logging.database_path = database.path().to_string_lossy().into_owned();
        let server = GatewayServer::new(config, None).await.unwrap();
        for (trace_id, provider, level) in [
            ("trace-standard", "test-provider", "standard"),
            ("trace-lite", "test-provider", "lite"),
            ("trace-other", "other-provider", "standard"),
        ] {
            server
                .state
                .logger
                .log(LogEntry {
                    trace_id: trace_id.to_owned(),
                    timestamp: chrono::Utc::now(),
                    method: "POST".to_owned(),
                    path: "/v1/chat/completions".to_owned(),
                    model: "gpt-4".to_owned(),
                    provider: provider.to_owned(),
                    status_code: 200,
                    duration_ms: 12,
                    cost: 0.0,
                    request_body: None,
                    response_body: None,
                    requested_model: Some("gpt-4".to_owned()),
                    responded_model: Some("gpt-4".to_owned()),
                    compression: Some(CompressionLogMetadata {
                        compression_level: level.to_owned(),
                        original_tokens: 100,
                        compressed_tokens: 80,
                        savings_percent: 20.0,
                        engines_applied: vec![level.to_owned()],
                        duration_ms: 3,
                        auto_triggered: false,
                        cache_downgrade_applied: false,
                        tool_definitions_tokens_saved: 0,
                        caveman_applied: false,
                        timed_out: false,
                        error: false,
                    }),
                })
                .unwrap();
        }

        let response = server
            .build_router()
            .oneshot(
                Request::get("/dashboard/logs?provider=test-provider&compression_level=standard")
                    .body(Body::empty())
                    .unwrap(),
            )
            .await
            .unwrap();
        assert_eq!(response.status(), StatusCode::OK);
        let body = axum::body::to_bytes(response.into_body(), 64 * 1024)
            .await
            .unwrap();
        let entries: Vec<LogEntry> = serde_json::from_slice(&body).unwrap();
        assert_eq!(entries.len(), 1);
        assert_eq!(entries[0].trace_id, "trace-standard");
        assert_eq!(
            entries[0]
                .compression
                .as_ref()
                .map(|metadata| metadata.compression_level.as_str()),
            Some("standard")
        );
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
                compression: None,
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
