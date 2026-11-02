//! Admin API routes for the budget-alert webhook subsystem
//! (budget-alerts-unified-webhooks spec, Task 7; design "Component 4").
//!
//! Surface (every route sits behind the existing admin auth
//! `route_layer` applied by `admin::admin_routes` — no new auth
//! surface, NFR Security):
//! - `GET /admin/webhooks` — the `webhooks` config section with MASKED
//! secrets (Req 3.6: secret values are never echoed; the
//! `[configured]` sentinel matches the credential-redaction convention
//! the provider config path uses) plus the per-endpoint failing-state
//! rollup (Req 5.5: failing iff the last 3 delivery attempts are all
//! `failed`/`parked`) and `allowed_private_hosts` (Req 5.2).
//! - `PUT /admin/webhooks` — saves the section through the same
//! validate → persist → apply pipeline every other config section uses
//! (design: "Reuses: existing config put/validate flow"), so
//! https-only/allowlist/event-name validation runs before persist
//! (Req 4.5). A secret returned unchanged as the mask sentinel
//! preserves the stored secret for that URL (Req 3.6 round-trip rule,
//! same as the provider-key path).
//! - `POST /admin/webhooks/test` — single-attempt test send addressed
//! by endpoint URL through the same signing pipeline (Req 4.4); echoes
//! the observed `TestResult` (status code or error class, never a
//! body, never a secret).
//! - `GET /admin/webhooks/deliveries?endpoint=<url>&limit=` — ledger
//! delivery rows for one endpoint (Req 4.8).
//!
//! Error bodies never include secrets (design, Error Handling 2).
//!
//! _Requirements: 3.6, 4.4, 4.5, 4.8, 5.1, 5.2, 5.5_

use std::sync::Arc;

use axum::{
    extract::{Json, Query, State},
    http::StatusCode,
    response::{IntoResponse, Response},
    routing::{get, post},
    Router,
};
use serde::Deserialize;
use serde_json::json;

use crate::config::{save_config, WebhookEndpoint, WebhooksConfig};
use crate::gateway::AppState;
use crate::webhooks::ledger::DeliveryRow;

/// Sentinel replacing a configured webhook secret in GET responses
/// (Req 3.6). Same `[configured]` style the provider custom-header
/// redaction uses; the UI round-trips it unchanged to preserve the
/// stored secret on save. Shared with the full-config redaction path
/// (`admin::redact_config_for_response`) via
/// [`crate::webhooks::admin::SECRET_MASK`] re-export — one constant,
/// one masking convention across every GET surface.
pub const SECRET_MASK: &str = "[configured]";

/// Failing-rollup window (Req 5.5): an endpoint is failing iff its
/// last 3 delivery attempts are all `failed` or `parked`.
const FAILING_ATTEMPT_WINDOW: u32 = 3;

/// Default page size for the deliveries listing (Req 4.8).
const DEFAULT_DELIVERIES_LIMIT: u32 = 50;

/// Upper bound for the deliveries `limit` query parameter.
const MAX_DELIVERIES_LIMIT: u32 = 200;

/// Build the webhooks admin router. Nested under the configured admin
/// path by `admin::admin_routes`, so the admin auth `route_layer`
/// protects every route here.
pub fn routes() -> Router<AppState> {
    Router::new()
        .route("/", get(get_webhooks).put(update_webhooks))
        .route("/test", post(test_webhook))
        .route("/deliveries", get(list_deliveries))
}

/// GET /admin/webhooks — the webhooks config section with masked
/// secrets plus the per-endpoint failing-state rollup (Reqs 3.6, 5.1,
/// 5.2, 5.5). Never echoes secret values.
async fn get_webhooks(State(state): State<AppState>) -> Response {
    let webhooks = {
        let config = state.config.read().await;
        config.webhooks.clone()
    };

    // One batched query for every configured endpoint (Req 5.5):
    // newest 3 attempts per endpoint via a window function — no
    // per-endpoint ledger round-trips.
    let rollup = recent_attempts_per_endpoint(&state, &webhooks, FAILING_ATTEMPT_WINDOW).await;

    let mut endpoints = Vec::with_capacity(webhooks.endpoints.len());
    for endpoint in &webhooks.endpoints {
        let attempts = rollup
            .iter()
            .filter(|(url, _)| url == &endpoint.url)
            .map(|(_, rows)| rows.as_slice())
            .next()
            .unwrap_or(&[]);
        let failing = attempts.len() == FAILING_ATTEMPT_WINDOW as usize
            && attempts.iter().all(|row| is_failure_outcome(&row.outcome));
        endpoints.push(json!({
            "url": endpoint.url,
            "secret": masked_secret(endpoint),
            "events": endpoint.events,
            "signed": is_signed(endpoint),
            "failing": failing,
            "last_outcome": attempts.first().map(|row| row.outcome.clone()),
        }));
    }

    (
        StatusCode::OK,
        Json(json!({
            "endpoints": endpoints,
            "allowed_private_hosts": webhooks.allowed_private_hosts,
        })),
    )
    .into_response()
}

/// PUT /admin/webhooks — validate, persist, and apply the webhooks
/// section through the existing config save pipeline (Req 4.5:
/// https-only/allowlist/event-name validation runs before persist).
/// Mask-preserve rule (Req 3.6): an incoming secret equal to the mask
/// sentinel keeps the stored secret for that URL.
async fn update_webhooks(
    State(state): State<AppState>,
    Json(mut incoming): Json<WebhooksConfig>,
) -> Response {
    let mut candidate = state.config.read().await.clone();
    preserve_masked_secrets(&mut incoming, &candidate.webhooks);
    candidate.webhooks = incoming;

    // Validate first (same ordering as the full-config PUT).
    if let Err(errors) = candidate.validate() {
        let msgs: Vec<String> = errors.iter().map(|e| e.to_string()).collect();
        return (
            StatusCode::BAD_REQUEST,
            Json(json!({
                "error": {
                    "message": "Configuration validation failed",
                    "type": "validation_error",
                    "details": msgs
                }
            })),
        )
            .into_response();
    }

    // Stamp the resolved security mode so the on-disk YAML no longer
    // reads as legacy (mirrors the full-config PUT path).
    candidate.admin.security_mode =
        Some(crate::admin::auth_policy::resolve_admin_security_mode(&candidate));

    if let Err(error) = save_config(state.config_path.as_ref(), &candidate) {
        tracing::error!("Failed to write config file: {}", error);
        return (
            StatusCode::INTERNAL_SERVER_ERROR,
            Json(json!({
                "error": {
                    "message": "Failed to write configuration file",
                    "type": "io_error"
                }
            })),
        )
            .into_response();
    }

    crate::gateway::apply_runtime_config_update(&state, candidate).await;

    tracing::info!("Webhooks configuration updated and saved");

    (
        StatusCode::OK,
        Json(json!({
            "status": "ok",
            "message": "Webhooks configuration updated successfully"
        })),
    )
        .into_response()
}

#[derive(Debug, Deserialize)]
struct TestWebhookRequest {
    url: String,
}

/// POST /admin/webhooks/test — single-attempt test send to the
/// configured endpoint with that URL (Req 4.4: endpoint identity is
/// the URL). Unknown URL → 404 (sibling `*_not_found` shape). Reuses
/// the shared `AppState` notifier (built once at startup) —
/// `send_test` itself owns the signing pipeline, SSRF re-validation,
/// and the single ledger event/attempt rows.
async fn test_webhook(
    State(state): State<AppState>,
    Json(body): Json<TestWebhookRequest>,
) -> Response {
    let configured = {
        let config = state.config.read().await;
        config
            .webhooks
            .endpoints
            .iter()
            .any(|endpoint| endpoint.url == body.url)
    };
    if !configured {
        return (
            StatusCode::NOT_FOUND,
            Json(json!({
                "error": {
                    "type": "endpoint_not_found",
                    "message": "No configured webhook endpoint matches that URL"
                }
            })),
        )
        .into_response();
    }

    let result = state.webhook_notifier.send_test(&body.url).await;

    (
        StatusCode::OK,
        Json(json!({
            "status_code": result.status_code,
            "error": result.error,
            "success": result.success,
        })),
    )
        .into_response()
}

#[derive(Debug, Deserialize)]
struct DeliveriesQuery {
    /// Optional endpoint filter; absent/empty lists deliveries for
    /// ALL endpoints (newest first). The admin UI's "All endpoints"
    /// mode sends no endpoint parameter.
    endpoint: Option<String>,
    limit: Option<u32>,
}

/// GET /admin/webhooks/deliveries?endpoint=<url>&limit= — the newest
/// ledger delivery rows for one endpoint, or for all endpoints when
/// `endpoint` is absent/empty (Req 4.8). Default limit 50, capped at
/// 200. Newest first.
async fn list_deliveries(
    State(state): State<AppState>,
    Query(query): Query<DeliveriesQuery>,
) -> Response {
    let limit = query
        .limit
        .filter(|limit| *limit > 0)
        .unwrap_or(DEFAULT_DELIVERIES_LIMIT)
        .min(MAX_DELIVERIES_LIMIT);
    let endpoint = query
        .endpoint
        .as_deref()
        .map(str::trim)
        .filter(|endpoint| !endpoint.is_empty());
    let rows = recent_attempts(&state, endpoint, limit).await;
    let deliveries: Vec<serde_json::Value> = rows.iter().map(delivery_row_json).collect();

    (
        StatusCode::OK,
        Json(json!({
            "endpoint": endpoint,
            "limit": limit,
            "deliveries": deliveries,
        })),
    )
    .into_response()
}

/// Replace a mask-sentinel secret with the stored secret for the same
/// URL (Req 3.6 round-trip rule). A sentinel for a URL with no stored
/// endpoint preserves nothing and must not persist the mask itself —
/// it becomes an unsigned endpoint.
fn preserve_masked_secrets(incoming: &mut WebhooksConfig, stored: &WebhooksConfig) {
    for endpoint in &mut incoming.endpoints {
        if endpoint.secret.as_deref() == Some(SECRET_MASK) {
            endpoint.secret = stored
                .endpoints
                .iter()
                .find(|stored_endpoint| stored_endpoint.url == endpoint.url)
                .and_then(|stored_endpoint| stored_endpoint.secret.clone())
                .filter(|secret| !secret.is_empty());
        }
    }
}

/// Ledger read off the async runtime (NFR Performance: SQLite reads go
/// through the blocking pool, mirroring the delivery write path).
/// A JoinError (panic or cancellation) logs a warn and degrades to an
/// empty list — `failing` then reads as its conservative `false`
/// default (UI contract: the panel checks `failing === true`), never
/// a 500.
async fn recent_attempts(
    state: &AppState,
    endpoint_url: Option<&str>,
    limit: u32,
) -> Vec<DeliveryRow> {
    let ledger = Arc::clone(&state.webhook_ledger);
    let endpoint_url = endpoint_url.map(str::to_string);
    match tokio::task::spawn_blocking(move || {
        ledger.recent_deliveries(endpoint_url.as_deref(), limit)
    })
    .await
    {
        Ok(rows) => rows,
        Err(error) => {
            tracing::warn!(
                error = %error,
                "webhooks ledger read task failed; deliveries list degraded to empty"
            );
            Vec::new()
        }
    }
}

/// Batched failing-rollup query (Req 5.5): the newest
/// `per_endpoint_limit` attempts for each of the configured endpoint
/// URLs, fetched in ONE blocking ledger round-trip (the store runs a
/// window-function query). JoinError degrades to an empty rollup
/// with a warn — endpoints then show their conservative defaults
/// (`failing: false`, `last_outcome: null`), same as a ledger read
/// failure.
async fn recent_attempts_per_endpoint(
    state: &AppState,
    webhooks: &WebhooksConfig,
    per_endpoint_limit: u32,
) -> Vec<(String, Vec<DeliveryRow>)> {
    let ledger = Arc::clone(&state.webhook_ledger);
    let urls: Vec<String> = webhooks.endpoints.iter().map(|e| e.url.clone()).collect();
    match tokio::task::spawn_blocking(move || {
        ledger.recent_deliveries_per_endpoint(&urls, per_endpoint_limit)
    })
    .await
    {
        Ok(rows) => rows,
        Err(error) => {
            tracing::warn!(
                error = %error,
                "webhooks ledger rollup task failed; failing states degrade to defaults"
            );
            Vec::new()
        }
    }
}

/// Masked secret for GET responses: the sentinel when a non-empty
/// secret is configured, null otherwise (Req 3.6).
fn masked_secret(endpoint: &WebhookEndpoint) -> serde_json::Value {
    match endpoint.secret.as_deref() {
        Some(secret) if !secret.is_empty() => serde_json::Value::String(SECRET_MASK.to_string()),
        _ => serde_json::Value::Null,
    }
}

/// Whether the endpoint delivers signed webhooks (non-empty secret).
fn is_signed(endpoint: &WebhookEndpoint) -> bool {
    endpoint.secret.as_deref().is_some_and(|secret| !secret.is_empty())
}

/// Terminal failure outcomes counted by the failing rollup (Req 5.5).
fn is_failure_outcome(outcome: &str) -> bool {
    outcome == "failed" || outcome == "parked"
}

fn delivery_row_json(row: &DeliveryRow) -> serde_json::Value {
    json!({
        "id": row.id,
        "event_id": row.event_id,
        "endpoint_url": row.endpoint_url,
        "attempt": row.attempt,
        "outcome": row.outcome,
        "status_code": row.status_code,
        "error": row.error,
        "attempted_at": row.attempted_at,
    })
}
