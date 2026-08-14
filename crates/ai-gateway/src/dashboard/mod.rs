use axum::{
    extract::ws::{Message, WebSocket, WebSocketUpgrade},
    extract::{Path, Query, State},
    http::{header, StatusCode},
    response::{IntoResponse, Response},
    routing::{get, post},
    Json, Router,
};
use chrono::{DateTime, Utc};
use rust_embed::Embed;
use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};
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
const MEMORY_EVENT_CAPACITY: usize = 100;
const MEMORY_REPLAY_CAPACITY: usize = 100;

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum MemoryEventType {
    Injection,
    Extraction,
    Eviction,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct MemoryDashboardEvent {
    pub event_type: MemoryEventType,
    pub namespace: String,
    pub count: u32,
    pub timestamp: String,
}

impl MemoryDashboardEvent {
    pub fn new(event_type: MemoryEventType, namespace: &str, count: u32) -> Self {
        Self {
            event_type,
            namespace: hashed_namespace(namespace),
            count,
            timestamp: Utc::now().to_rfc3339(),
        }
    }
}

pub fn hashed_namespace(namespace: &str) -> String {
    let digest = Sha256::digest(namespace.as_bytes());
    digest[..8]
        .iter()
        .map(|byte| format!("{byte:02x}"))
        .collect()
}

#[derive(Debug)]
pub struct MemoryEventHub {
    sender: broadcast::Sender<MemoryDashboardEvent>,
    replay: Mutex<VecDeque<MemoryDashboardEvent>>,
}

impl MemoryEventHub {
    pub fn new() -> Self {
        let (sender, _) = broadcast::channel(MEMORY_EVENT_CAPACITY);
        Self {
            sender,
            replay: Mutex::new(VecDeque::with_capacity(MEMORY_REPLAY_CAPACITY)),
        }
    }

    pub fn publish(&self, event: MemoryDashboardEvent) {
        let mut replay = self
            .replay
            .lock()
            .expect("memory event replay mutex poisoned");
        if replay.len() == MEMORY_REPLAY_CAPACITY {
            replay.pop_front();
        }
        replay.push_back(event.clone());
        let _ = self.sender.send(event);
    }

    pub fn subscribe(&self) -> MemoryEventSubscription {
        let replay = self
            .replay
            .lock()
            .expect("memory event replay mutex poisoned");
        let receiver = self.sender.subscribe();
        MemoryEventSubscription {
            replay: replay.iter().cloned().collect(),
            receiver,
        }
    }
}

impl Default for MemoryEventHub {
    fn default() -> Self {
        Self::new()
    }
}

impl crate::memory::EvictionEventPublisher for MemoryEventHub {
    fn publish_eviction(&self, namespace: &str, count: u64) {
        self.publish(MemoryDashboardEvent::new(
            MemoryEventType::Eviction,
            namespace,
            u32::try_from(count).unwrap_or(u32::MAX),
        ));
    }
}

pub struct MemoryEventSubscription {
    pub replay: Vec<MemoryDashboardEvent>,
    pub receiver: broadcast::Receiver<MemoryDashboardEvent>,
}

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

/// Event emitted by the tool compression middleware after each compression.
///
/// Published to the compression events hub for dashboard WebSocket delivery.
/// Contains all fields required for real-time monitoring of compression activity.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ToolCompressionEvent {
    /// Unique request identifier for correlation.
    pub request_id: String,
    /// Model group the request targeted.
    pub model_group: String,
    /// Effective compression level applied.
    pub level: String,
    /// Estimated token count of the original tools array.
    pub original_tokens: u64,
    /// Estimated token count of the compressed tools array.
    pub compressed_tokens: u64,
    /// Names of compression strategies that were applied.
    pub strategies_applied: Vec<String>,
    /// Number of tools pruned by the Tool Pruner stage.
    pub tools_pruned_count: usize,
    /// Whether the semantic retrieval stage was active for this request.
    pub semantic_retrieval_active: bool,
    /// Names of tools deferred by semantic retrieval (below similarity threshold).
    pub tools_deferred: Vec<String>,
    /// Whether the feedback loop adjusted the compression level for this request.
    pub feedback_adjusted: bool,
}

impl CompressionEventHub {
    /// Publish a tool compression event as a synthetic `CompressionStats` entry
    /// so existing dashboard subscribers receive it alongside normal compression
    /// events.
    pub fn publish_tool_compression(&self, event: ToolCompressionEvent) {
        let tokens_saved = event
            .original_tokens
            .saturating_sub(event.compressed_tokens);
        let savings_percent = if event.original_tokens > 0 {
            (tokens_saved as f64 / event.original_tokens as f64) * 100.0
        } else {
            0.0
        };
        let stats = CompressionStats {
            request_id: event.request_id.clone(),
            provider: event.model_group.clone(),
            model: event.level.clone(),
            original_tokens: event.original_tokens as u32,
            compressed_tokens: event.compressed_tokens as u32,
            savings_percent,
            compression_time_ms: 0,
            level: crate::compression::CompressionLevel::None,
            engines_applied: event.strategies_applied,
            engine_results: Vec::new(),
            timed_out: false,
            auto_triggered: false,
            cache_downgrade_applied: false,
            tool_definitions_tokens_saved: tokens_saved as u32,
            caveman_applied: false,
            error: false,
        };
        self.publish(stats);
    }
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
        .route("/memory", get(memory_snapshot_handler))
        .route("/memory/snapshot", get(memory_snapshot_handler))
        .route("/errors", get(errors_handler))
        .route("/logs", get(logs_handler))
        .route(
            "/tool-compression/config",
            get(tool_compression_config_handler),
        )
        .route(
            "/tool-compression/overrides",
            get(tool_compression_overrides_handler),
        )
        .route(
            "/tool-compression/stats",
            get(tool_compression_stats_handler),
        )
        .route(
            "/tool-compression/test",
            post(tool_compression_test_handler),
        )
        .route(
            "/tool-compression/activity",
            get(tool_compression_activity_handler),
        )
        .route("/", get(index_handler))
        .route("/{*path}", get(static_handler))
}

async fn ws_handler(ws: WebSocketUpgrade, State(state): State<AppState>) -> impl IntoResponse {
    ws.on_upgrade(move |socket| handle_ws(socket, state))
}
async fn handle_ws(mut socket: WebSocket, state: AppState) {
    let mut subscription = state.loop_detector.events.subscribe();
    let mut compression_subscription = state.compression_events.subscribe();
    let mut memory_subscription = state.memory_events.subscribe();
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
    for event in memory_subscription.replay {
        if send_memory_event(&mut socket, event).await.is_err() {
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
            event = memory_subscription.receiver.recv() => {
                match event {
                    Ok(event) => {
                        if send_memory_event(&mut socket, event).await.is_err() {
                            break;
                        }
                    }
                    Err(tokio::sync::broadcast::error::RecvError::Lagged(_)) => continue,
                    Err(tokio::sync::broadcast::error::RecvError::Closed) => break,
                }
            }
            _ = interval.tick() => {
                // Safety net: drop any in-flight entries that outlived their
                // request (e.g. after a panic). Normal completion deregisters
                // via the request guard's Drop.
                let timeout_secs = state
                    .config
                    .try_read()
                    .map(|c| c.server.request_timeout_seconds)
                    .unwrap_or(30);
                state
                    .active_requests
                    .sweep_stale(std::time::Duration::from_secs(timeout_secs.saturating_mul(2)));

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

async fn send_memory_event(
    socket: &mut WebSocket,
    event: MemoryDashboardEvent,
) -> Result<(), axum::Error> {
    let message = memory_message(event);
    socket.send(Message::Text(message.to_string().into())).await
}

fn memory_message(event: MemoryDashboardEvent) -> serde_json::Value {
    serde_json::json!({"type": "memory_event", "data": event})
}

#[derive(Debug, Serialize)]
struct MemoryDashboardSnapshot {
    enabled: bool,
    total_count: u64,
    namespace_counts: std::collections::BTreeMap<String, u64>,
    average_relevance_score: f64,
    storage_size_bytes: Option<u64>,
    last_decay_cycle: Option<DateTime<Utc>>,
    events: MemoryEventAggregates,
}

#[derive(Debug, Default, Serialize)]
struct MemoryEventAggregates {
    injections: u64,
    extractions: u64,
    evictions: u64,
}

async fn memory_snapshot_handler(State(state): State<AppState>) -> impl IntoResponse {
    let events = state.memory_events.subscribe().replay.into_iter().fold(
        MemoryEventAggregates::default(),
        |mut totals, event| {
            match event.event_type {
                MemoryEventType::Injection => totals.injections += u64::from(event.count),
                MemoryEventType::Extraction => totals.extractions += u64::from(event.count),
                MemoryEventType::Eviction => totals.evictions += u64::from(event.count),
            }
            totals
        },
    );
    let Some(system) = state.memory_system.read().await.clone() else {
        return Json(MemoryDashboardSnapshot {
            enabled: false,
            total_count: 0,
            namespace_counts: Default::default(),
            average_relevance_score: 0.0,
            storage_size_bytes: None,
            last_decay_cycle: None,
            events,
        });
    };
    match system.store.stats() {
        Ok(stats) => {
            let mut namespace_counts = std::collections::BTreeMap::new();
            for (namespace, count) in stats.memories_per_namespace {
                let kind = if namespace.contains("::project::") {
                    "project"
                } else if namespace.contains("::agent::") {
                    "agent"
                } else {
                    "user"
                };
                *namespace_counts.entry(kind.to_owned()).or_default() += count;
            }
            Json(MemoryDashboardSnapshot {
                enabled: true,
                total_count: stats.total_count,
                namespace_counts,
                average_relevance_score: stats.average_relevance_score,
                storage_size_bytes: stats.storage_size_bytes,
                last_decay_cycle: stats.last_decay_cycle,
                events,
            })
        }
        Err(error) => {
            tracing::warn!(error = %error, "Failed to build memory dashboard snapshot");
            Json(MemoryDashboardSnapshot {
                enabled: true,
                total_count: 0,
                namespace_counts: Default::default(),
                average_relevance_score: 0.0,
                storage_size_bytes: None,
                last_decay_cycle: None,
                events,
            })
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
    snapshot.active_requests_list = state.active_requests.snapshot();
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

// ─── Tool Compression Dashboard Handlers ──────────────────────────────────────

/// GET /dashboard/tool-compression/config — serve current tool compression config as JSON.
async fn tool_compression_config_handler(State(state): State<AppState>) -> Response {
    let config = state.config.read().await;
    let tc_config = &config.tool_compression;
    (
        StatusCode::OK,
        Json(serde_json::to_value(tc_config).unwrap_or_default()),
    )
        .into_response()
}

/// GET /dashboard/tool-compression/overrides — serve per-model-group overrides.
async fn tool_compression_overrides_handler(State(state): State<AppState>) -> Response {
    let config = state.config.read().await;
    let overrides = &config.tool_compression.model_group_overrides;
    (
        StatusCode::OK,
        Json(serde_json::to_value(overrides).unwrap_or_default()),
    )
        .into_response()
}

/// GET /dashboard/tool-compression/stats — serve real-time compression statistics.
async fn tool_compression_stats_handler(State(state): State<AppState>) -> Response {
    let tc_state = &state.tool_compression_state;

    // Build per-group feedback summary
    let feedback_groups: Vec<serde_json::Value> = tc_state
        .feedback_loop
        .group_names()
        .into_iter()
        .filter_map(|name| {
            tc_state.feedback_loop.get_state(&name).map(|fs| {
                serde_json::json!({
                    "group": name,
                    "level": format!("{:?}", fs.current_level),
                    "error_rate": fs.current_error_rate(),
                    "baseline_rate": fs.baseline_rate,
                    "locked": fs.locked,
                    "recovery_counter": fs.recovery_counter,
                    "window_size": fs.window.len(),
                })
            })
        })
        .collect();

    let stats = serde_json::json!({
        "feedback_groups": feedback_groups,
        "feedback_group_count": tc_state.feedback_loop.group_names().len(),
        "cached_descriptions": tc_state.description_compressor.len(),
        "active_sessions": tc_state.disclosure_state.len(),
        "semantic_embeddings": tc_state.semantic_state.len(),
        "namespaces": tc_state.namespace_state.len(),
    });
    (StatusCode::OK, Json(stats)).into_response()
}

/// POST /dashboard/tool-compression/test — apply compression to sample input.
///
/// Accepts `{"tools": [...]}`, runs the configured compression pipeline stages,
/// and returns a side-by-side comparison of original vs compressed output with
/// token estimates and reduction percentage.
async fn tool_compression_test_handler(
    State(state): State<AppState>,
    Json(body): Json<serde_json::Value>,
) -> Response {
    use crate::tool_compression::config::CompressionLevel as TcLevel;
    use crate::tool_compression::stage::CompressionStage;
    use crate::tool_compression::stages::deduplicator::SchemaDeduplicator;
    use crate::tool_compression::stages::minifier::SchemaMinifier;
    use crate::tool_compression::stages::truncator::DescriptionTruncator;
    use crate::tool_compression::types::{CompressionContext, ProviderCaps, ToolDefinition};
    use std::time::Instant;

    let tools = body.get("tools").and_then(|v| v.as_array());
    let Some(tools_arr) = tools else {
        return (
            StatusCode::BAD_REQUEST,
            Json(serde_json::json!({
                "error": { "message": "Request body must contain a 'tools' array" }
            })),
        )
            .into_response();
    };

    if tools_arr.is_empty() {
        return (
            StatusCode::OK,
            Json(serde_json::json!({
                "original": [],
                "compressed": [],
                "original_tokens": 0,
                "compressed_tokens": 0,
                "reduction_percent": 0.0,
                "strategies_applied": []
            })),
        )
            .into_response();
    }

    // Compute original token estimate (chars / 4).
    let original_json = serde_json::to_string_pretty(tools_arr).unwrap_or_default();
    let original_tokens = original_json.len() as u64 / 4;

    // Read current config for compression settings.
    let config = state.config.read().await;
    let tc_config = &config.tool_compression;
    let level = tc_config.level;

    // Build ToolDefinition vec from the input.
    let mut tools_vec: Vec<ToolDefinition> = tools_arr
        .iter()
        .map(|raw| {
            let name = raw
                .pointer("/function/name")
                .and_then(|n| n.as_str())
                .unwrap_or("")
                .to_string();
            ToolDefinition {
                raw: raw.clone(),
                name,
                content_hash: 0,
            }
        })
        .collect();

    // Build a compression context with current config level.
    let mut ctx = CompressionContext {
        level,
        provider_caps: ProviderCaps::conservative(),
        ..Default::default()
    };
    // For deduplicator to work, enable $ref support in test context.
    ctx.provider_caps.supports_ref = true;
    ctx.provider_caps.supports_nullable = true;

    // Instantiate pipeline stages that don't require session state.
    let stages: Vec<Box<dyn CompressionStage>> = vec![
        Box::new(SchemaMinifier),
        Box::new(DescriptionTruncator::with_state(std::sync::Arc::clone(
            &state.tool_compression_state,
        ))),
        Box::new(SchemaDeduplicator),
    ];

    let start = Instant::now();

    // Run enabled stages.
    for stage in &stages {
        if stage.is_enabled(tc_config, level) {
            stage.apply(&mut tools_vec, &mut ctx);
        }
    }

    let elapsed_ms = start.elapsed().as_secs_f64() * 1000.0;

    // Build compressed output.
    let compressed_arr: Vec<serde_json::Value> = tools_vec.iter().map(|t| t.raw.clone()).collect();
    let compressed_json = serde_json::to_string_pretty(&compressed_arr).unwrap_or_default();
    let compressed_tokens = compressed_json.len() as u64 / 4;

    let reduction_percent = if original_tokens > 0 {
        ((original_tokens as f64 - compressed_tokens as f64) / original_tokens as f64) * 100.0
    } else {
        0.0
    };

    let level_str = match level {
        TcLevel::Low => "low",
        TcLevel::Medium => "medium",
        TcLevel::High => "high",
        TcLevel::Max => "max",
    };

    (
        StatusCode::OK,
        Json(serde_json::json!({
            "original": tools_arr,
            "compressed": compressed_arr,
            "original_tokens": original_tokens,
            "compressed_tokens": compressed_tokens,
            "reduction_percent": (reduction_percent * 10.0).round() / 10.0,
            "strategies_applied": ctx.strategies_applied,
            "level": level_str,
            "compression_time_ms": (elapsed_ms * 100.0).round() / 100.0,
            "tool_count": tools_arr.len(),
        })),
    )
        .into_response()
}

/// GET /dashboard/tool-compression/activity — pruning and disclosure activity.
async fn tool_compression_activity_handler(State(state): State<AppState>) -> Response {
    let tc_state = &state.tool_compression_state;
    let config = state.config.read().await;

    // Pruning activity: per-key usage frequencies with top tools
    let key_usage: Vec<serde_json::Value> = tc_state
        .key_usage
        .iter()
        .take(50) // Limit response size
        .map(|entry| {
            let mut top_tools: Vec<serde_json::Value> = entry
                .value()
                .entries
                .iter()
                .take(20)
                .map(|(tool_name, (call_count, _tick))| {
                    serde_json::json!({
                        "tool_name": tool_name.clone(),
                        "call_count": *call_count,
                    })
                })
                .collect();
            top_tools.sort_by(|a, b| {
                let ac = a["call_count"].as_u64().unwrap_or(0);
                let bc = b["call_count"].as_u64().unwrap_or(0);
                bc.cmp(&ac)
            });
            serde_json::json!({
                "api_key": entry.key().clone(),
                "tool_count": entry.value().entries.len(),
                "top_tools": top_tools,
            })
        })
        .collect();

    // Currently pruned tools per session (tools with zero calls past threshold)
    let pruned_sessions: Vec<serde_json::Value> = tc_state
        .session_usage
        .iter()
        .take(50)
        .filter_map(|entry| {
            let session_id = entry.key().clone();
            let usage_map = entry.value();
            let request_count = tc_state
                .session_request_count
                .get(&session_id)
                .map(|r| *r)
                .unwrap_or(0);
            let min_requests = config.tool_compression.pruning.min_requests;
            if request_count < min_requests as u64 {
                return None;
            }
            let pruned: Vec<String> = usage_map
                .iter()
                .filter(|(_name, count)| **count == 0)
                .map(|(name, _count)| name.clone())
                .collect();
            if pruned.is_empty() {
                return None;
            }
            Some(serde_json::json!({
                "session_id": session_id,
                "request_count": request_count,
                "pruned_tools": pruned,
            }))
        })
        .collect();

    // always_include allowlist from config
    let always_include = &config.tool_compression.pruning.always_include;

    // Disclosure activity: active sessions with details
    let disclosure_sessions: Vec<serde_json::Value> = tc_state
        .disclosure_state
        .iter()
        .take(50)
        .map(|entry| {
            let tool_names: Vec<String> = entry.value().iter().take(20).cloned().collect();
            serde_json::json!({
                "session_id": entry.key().clone(),
                "disclosed_tools": entry.value().len(),
                "tool_names": tool_names,
            })
        })
        .collect();

    // get_tool_schema call frequency from disclosure state
    let total_disclosed: usize = tc_state
        .disclosure_state
        .iter()
        .map(|entry| entry.value().len())
        .sum();

    (
        StatusCode::OK,
        Json(serde_json::json!({
            "pruning": {
                "key_usage_entries": key_usage,
                "pruned_sessions": pruned_sessions,
                "always_include": always_include,
            },
            "disclosure": {
                "active_sessions": disclosure_sessions,
                "total_active_sessions": tc_state.disclosure_state.len(),
                "total_disclosed_tools": total_disclosed,
            }
        })),
    )
        .into_response()
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
                    provider: "test-provider".to_string(),
                    model: "gpt-4".to_string(),
                    cost_per_million_input_tokens: 0.0,
                    cost_per_million_output_tokens: 0.0,
                    priority: 100,
                    structured_output_passthrough: None,
                    tier: None,
                    context_window: 0,
                    specializations: vec![],
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
            structured_output: None,
            first_launch_completed: false,
            tray: TrayConfig::default(),
            codex_instructions_url: None,
            streaming: None,
            virtual_keys: Default::default(),
            loop_detection: Default::default(),
            guardrails: None,
            tool_compression: Default::default(),
            smart_routing: Default::default(),
            memory: None,
            xhigh_models_allowlist: Default::default(),
            reasoning_models_allowlist: Default::default(),
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
    fn memory_event_hashes_namespace_and_contains_no_content() {
        let namespace = "user::vk_secret_value::project::project-hash";
        let event = MemoryDashboardEvent::new(MemoryEventType::Injection, namespace, 4);
        let message = memory_message(event.clone());

        assert_eq!(event.namespace.len(), 16);
        assert_eq!(event.namespace, hashed_namespace(namespace));
        assert_eq!(event.namespace, hashed_namespace(namespace));
        assert!(!message.to_string().contains("vk_secret_value"));
        assert_eq!(message["type"], "memory_event");
        assert_eq!(message["data"]["event_type"], "injection");
        assert_eq!(message["data"]["count"], 4);
        for field in ["content", "messages", "prompt", "request_body", "payload"] {
            assert!(message["data"].get(field).is_none());
        }
    }

    #[tokio::test]
    async fn memory_event_hub_replay_is_bounded() {
        let hub = MemoryEventHub::new();
        for index in 0..(MEMORY_REPLAY_CAPACITY + 5) {
            hub.publish(MemoryDashboardEvent::new(
                MemoryEventType::Extraction,
                &format!("user::vk-{index}"),
                index as u32,
            ));
        }
        let subscription = hub.subscribe();
        assert_eq!(subscription.replay.len(), MEMORY_REPLAY_CAPACITY);
        assert_eq!(subscription.replay[0].count, 5);
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
                    memories_injected: 0,
                    memories_stored: 0,
                    injection_tokens: 0,
                    detected_project: None,
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
                memories_injected: 0,
                memories_stored: 0,
                injection_tokens: 0,
                detected_project: None,
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
