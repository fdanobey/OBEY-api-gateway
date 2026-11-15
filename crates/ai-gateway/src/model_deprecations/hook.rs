//! Runtime model-not-found detection hook (model-deprecation-alerts
//! spec, Req 5.1/6.1).
//!
//! Pure signature classification ([`is_model_not_found`]) plus a thin
//! reporting seam ([`note_runtime_gone`]) the router calls at its
//! existing error-classification sites. The seam records the
//! `(provider, model)` as runtime-detected gone in the shared
//! [`DeprecationState`] — a DashMap-cheap insert — and never blocks
//! the response path: the call is synchronous and off the hot path
//! (no spawn needed; `record_runtime_gone` is lock-shard work only).
//!
//! Episode evaluation and event emission live in the state core +
//! webhooks tasks (6/7); this module never touches them.
//!
//! _Requirements: 5.1 (signature), 6.1 (failover classification
//! consumed by the router), 5.2 (state recorded for candidate skip)_

use std::sync::Arc;

use chrono::Utc;

use crate::model_deprecations::state::DeprecationState;

/// Ledger `error_class` for a model-not-found-signature failure
/// (Req 6.3: existing row shape, model-not-found class).
pub(crate) const MODEL_NOT_FOUND_CLASS: &str = "model_not_found";

/// Whether an upstream failure carries the model-not-found /
/// deprecation signature (Req 5.1): HTTP 404 or 410 AND a body that
/// mentions `model_not_found`, "does not exist", or "deprecat"
/// (case-insensitive substring — "deprecated", "deprecation", and
/// "DeprecatED" all match).
pub fn is_model_not_found(status: u16, body: &str) -> bool {
    if status != 404 && status != 410 {
        return false;
    }
    let lower = body.to_ascii_lowercase();
    lower.contains("model_not_found")
        || lower.contains("does not exist")
        || lower.contains("deprecat")
}

/// Record a runtime-gone detection for `(provider, model)` (Req
/// 5.1/5.2). Synchronous DashMap insert — cheap enough for the
/// router's post-failure path, no spawn, no I/O, never blocks the
/// response. The 24-hour unconfirmed bound, reload clearing, and
/// scan re-arming are enforced by [`DeprecationState`].
///
/// Task 9 emission: the ROUTER call sites fire
/// [`crate::model_deprecations::emission::spawn_deprecation_emission`]
/// right after this when the router carries the emission deps
/// (per-instance, set once in `GatewayServer::new`) — the hook
/// itself stays pure so bare-router tests stay dormant.
pub fn note_runtime_gone(state: &Arc<DeprecationState>, provider: &str, model: &str) {
    state.record_runtime_gone(provider, model, Utc::now());
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Signature matrix: status × body-string. Positives cover every
    /// marker phrase on both qualifying statuses (incl. "deprecat"
    /// substring variants and case insensitivity); negatives cover
    /// non-qualifying statuses, qualifying statuses without a marker,
    /// and empty bodies (Req 5.1, design "Testing Strategy").
    // Validates: Requirements 5.1, 6.1
    #[test]
    fn signature_matrix_status_x_body() {
        let cases: &[(u16, &str, bool)] = &[
            // --- qualifying statuses × matching bodies -------------
            (404, r#"{"error":{"code":"model_not_found"}}"#, true),
            (404, r#"{"error":{"message":"The model 'x' does not exist"}"#, true),
            (404, "this model is deprecated", true),
            (404, "deprecation scheduled", true),
            (404, "DeprecatED", true),
            (404, "MODEL_NOT_FOUND", true),
            (404, "Model_Not_Found", true),
            (404, r#"{"error":{"type":"model_not_found"}}"#, true),
            (410, r#"{"error":{"code":"model_not_found"}}"#, true),
            (410, "model does not exist anymore", true),
            (410, "gone: DEPRECATED", true),
            // --- qualifying statuses × non-matching bodies ---------
            (404, r#"{"error":{"message":"not found"}}"#, false),
            (404, "no such endpoint", false),
            (404, "unauthorized", false),
            (410, "", false),
            (404, "", false),
            // --- non-qualifying statuses (even with markers) -------
            (400, "model_not_found", false),
            (400, "this model is deprecated", false),
            (500, r#"{"error":{"code":"model_not_found"}}"#, false),
            (429, "model_not_found deprecat does not exist", false),
            (200, "model_not_found", false),
            (503, "deprecated", false),
            (408, "does not exist", false),
            // near-miss markers must not match
            (404, "model not found", false),
            (404, "modell_not_found", false),
            (404, "does exists", false),
        ];
        for (status, body, expected) in cases {
            assert_eq!(
                is_model_not_found(*status, body),
                *expected,
                "status {status} body {body:?}"
            );
        }
    }

    /// The reporting seam stamps a live detection time and arms the
    /// 24-hour bound (Req 5.1/5.2): `is_gone` flips true immediately.
    #[test]
    fn note_runtime_gone_marks_state() {
        let state = Arc::new(DeprecationState::default());
        assert!(!state.is_gone("prov", "m1"));
        note_runtime_gone(&state, "prov", "m1");
        assert!(state.is_gone("prov", "m1"));
        assert!(state.gone_since("prov", "m1").is_some());
        // Other pairs untouched.
        assert!(!state.is_gone("prov", "m2"));
        assert!(!state.is_gone("other", "m1"));
    }
}
