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
            feedback_loop.get_state(&group_name).map(|fs| FeedbackGroupState {
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

    (
        StatusCode::OK,
        Json(json!({ "descriptions": entries })),
    )
        .into_response()
}
