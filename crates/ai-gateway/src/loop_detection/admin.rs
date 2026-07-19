use crate::loop_detection::{EnforcementLevel, SessionState};
use axum::{
    extract::{Path, Query, State},
    http::StatusCode,
    response::{IntoResponse, Response},
    routing::{get, post},
    Json, Router,
};
use serde::{Deserialize, Serialize};
use serde_json::json;
use std::{collections::BTreeMap, mem::size_of, time::SystemTime};

#[derive(Debug, Deserialize)]
pub struct SessionQuery {
    #[serde(default = "default_limit")]
    limit: usize,
    #[serde(default)]
    offset: usize,
}

fn default_limit() -> usize {
    50
}

#[derive(Debug, Serialize)]
struct SessionSummary {
    session_id: String,
    virtual_key: Option<String>,
    request_count: u32,
    current_confidence: f32,
    current_enforcement_level: &'static str,
    last_activity_timestamp: chrono::DateTime<chrono::Utc>,
    dominant_signal: &'static str,
}

pub fn routes() -> Router<crate::gateway::AppState> {
    Router::new()
        .route("/sessions", get(list_sessions))
        .route("/sessions/{id}", get(session_detail))
        .route("/sessions/{id}/reset", post(reset_session))
        .route("/stats", get(stats))
}

async fn list_sessions(
    State(state): State<crate::gateway::AppState>,
    Query(query): Query<SessionQuery>,
) -> Json<serde_json::Value> {
    let total = state.loop_detector.sessions.len();
    let mut sessions = state
        .loop_detector
        .sessions
        .iter()
        .map(|entry| summary(entry.key().clone(), entry.value()))
        .collect::<Vec<_>>();
    sessions.sort_by(|left, right| left.session_id.cmp(&right.session_id));
    let limit = query.limit.clamp(1, 200);
    let sessions = sessions
        .into_iter()
        .skip(query.offset)
        .take(limit)
        .collect::<Vec<_>>();
    Json(json!({"sessions": sessions, "total": total, "limit": limit, "offset": query.offset}))
}

async fn session_detail(
    State(state): State<crate::gateway::AppState>,
    Path(id): Path<String>,
) -> Response {
    let Some(session) = state.loop_detector.sessions.get(&id) else {
        return (
            StatusCode::NOT_FOUND,
            Json(json!({"error":{"message":"Session not found","type":"not_found"}})),
        )
            .into_response();
    };
    Json(json!({
        "session_id": id,
        "virtual_key": session.vk_id,
        "request_count": session.request_count,
        "confidence": session.smoothed_confidence,
        "peak_confidence": session.peak_confidence,
        "enforcement_level": level_label(session.enforcement_level),
        "dominant_signal": session.dominant_signal,
        "total_tokens": session.total_tokens,
        "total_cost": session.total_cost,
        "error_count": session.error_count,
        "recent_request_hashes": session.request_hashes.iter().rev().copied().collect::<Vec<_>>(),
        "recent_tool_fingerprints": session.tool_fingerprints.iter().rev().copied().collect::<Vec<_>>(),
        "signal_history": session.signal_history.iter().rev().map(|signals| json!({
            "content_similarity": signals.content_similarity,
            "tool_call_repetition": signals.tool_call_repetition,
            "response_stagnation": signals.response_stagnation,
            "token_velocity": signals.token_velocity,
            "error_cycling": signals.error_cycling,
            "context_growth": signals.context_growth,
            "cost_velocity": signals.cost_velocity,
        })).collect::<Vec<_>>(),
        "recent_response_descriptors": session.response_descriptors.iter().rev().map(|response| json!({
            "token_count": response.token_count,
            "block_type_hash": response.block_type_hash,
            "is_error": response.is_error,
        })).collect::<Vec<_>>(),
        "escalation_timeline": session.escalation_history.iter().map(|event| json!({
            "timestamp": event.timestamp,
            "from": level_label(event.from_level),
            "to": level_label(event.to_level),
            "confidence": event.confidence,
        })).collect::<Vec<_>>(),
    }))
    .into_response()
}

async fn reset_session(
    State(state): State<crate::gateway::AppState>,
    Path(id): Path<String>,
) -> Response {
    let Some(mut session) = state.loop_detector.sessions.get_mut(&id) else {
        return (
            StatusCode::NOT_FOUND,
            Json(json!({"error":{"message":"Session not found","type":"not_found"}})),
        )
            .into_response();
    };
    reset_session_state(&mut session);
    Json(json!({
        "session_id": id,
        "request_count": session.request_count,
        "confidence": session.smoothed_confidence,
        "enforcement_level": level_label(session.enforcement_level),
        "dominant_signal": session.dominant_signal,
    }))
    .into_response()
}

pub fn reset_session_state(session: &mut SessionState) {
    let vk_id = session.vk_id.clone();
    let history_depth = session.history_depth();
    *session = SessionState::new(vk_id, history_depth);
}

async fn stats(State(state): State<crate::gateway::AppState>) -> Json<serde_json::Value> {
    let mut enforcement_counts = BTreeMap::<&'static str, usize>::new();
    let mut signal_distribution = BTreeMap::<&'static str, usize>::new();
    let mut total_confidence = 0.0f64;
    let mut top = Vec::new();
    for entry in state.loop_detector.sessions.iter() {
        *enforcement_counts
            .entry(level_label(entry.enforcement_level))
            .or_default() += 1;
        *signal_distribution
            .entry(entry.dominant_signal)
            .or_default() += 1;
        total_confidence += f64::from(entry.smoothed_confidence);
        top.push(summary(entry.key().clone(), entry.value()));
    }
    top.sort_by(|left, right| right.current_confidence.total_cmp(&left.current_confidence));
    top.truncate(10);
    let total = state.loop_detector.sessions.len();
    Json(json!({
        "total_sessions": total,
        "enforcement_counts": enforcement_counts,
        "average_confidence": if total == 0 { 0.0 } else { total_confidence / total as f64 },
        "top_sessions": top,
        "signal_distribution": signal_distribution,
        "estimated_memory_bytes": total.saturating_mul(size_of::<SessionState>()),
        "eviction_count_since_reset": state.loop_detector.metrics.evicted_total(),
        "evictions_per_minute": state.loop_detector.metrics.evictions_per_minute(),
    }))
}

fn summary(session_id: String, session: &SessionState) -> SessionSummary {
    SessionSummary {
        session_id,
        virtual_key: session.vk_id.clone(),
        request_count: session.request_count,
        current_confidence: session.smoothed_confidence,
        current_enforcement_level: level_label(session.enforcement_level),
        last_activity_timestamp: chrono::DateTime::<chrono::Utc>::from(
            SystemTime::now()
                .checked_sub(session.last_active.elapsed())
                .unwrap_or(SystemTime::UNIX_EPOCH),
        ),
        dominant_signal: session.dominant_signal,
    }
}

fn level_label(level: EnforcementLevel) -> &'static str {
    match level {
        EnforcementLevel::None => "none",
        EnforcementLevel::Warn => "warn",
        EnforcementLevel::Throttle => "throttle",
        EnforcementLevel::Inject => "inject",
        EnforcementLevel::HardStop => "hard_stop",
    }
}
