//! Virtual key authentication: manager method + Axum enforcement middleware.
//!
//! This module implements [`VirtualKeyManager::authenticate`], which resolves a
//! presented raw key to an [`AuthenticatedKey`] via a DashMap cache (fast path)
//! backed by the SQLite [`KeyStore`] (slow path), and validates the key's
//! status and expiration. It also provides [`virtual_key_auth_middleware`], the
//! Axum layer that enforces the configured `virtual_keys.enforcement` mode.
//!
//! Enforcement modes (design "Authentication Middleware"):
//! - `disabled`: skip entirely, ignore any `vk_` key (Req 11.1, 11.5, 2.4)
//! - `optional`: validate a presented `vk_` key, otherwise pass through
//!   (Req 11.4, 2.4)
//! - `required`: reject with 401 unless a valid `vk_` key is presented
//!   (Req 11.2)
//!
//! HTTP error mapping (design "HTTP Error Mapping"):
//! - [`AuthError::InvalidKey`] → 401 `{"error": "Invalid or unrecognized API key"}`
//! - [`AuthError::Expired`]    → 403 `{"error": "API key has expired"}`
//! - [`AuthError::Revoked`]    → 403 `{"error": "API key has been revoked"}`
//!
//! On the Anthropic Messages front-door paths (`/v1/messages`,
//! `/v1/messages/count_tokens`) middleware rejections are instead emitted in
//! the Anthropic error envelope `{"type":"error","error":{"type","message"}}`
//! (anthropic-messages-front-door Req 3.3): missing/invalid key → 401
//! `authentication_error`, expired/revoked key or model-access denial → 403
//! `permission_error`, budget exhaustion and per-key rate limit → 429
//! `rate_limit_error`.
//!
//! Credential shapes: `Authorization: Bearer <vk_...>` everywhere, plus
//! `x-api-key: <vk_...>` as an alias accepted ONLY on the Anthropic Messages
//! front-door paths `/v1/messages` and `/v1/messages/count_tokens`
//! (anthropic-messages-front-door Req 5.4, design D3); both flow through the
//! identical `authenticate` + enforcement pipeline.`

use std::sync::Arc;
use std::time::Instant;

use axum::{
    body::{to_bytes, Body},
    extract::{Request, State},
    http::{header, StatusCode},
    middleware::Next,
    response::{IntoResponse, Response},
    Json,
};
use chrono::Utc;
use serde_json::json;

use crate::anthropic::errors::{
    error_response as anthropic_error_response, http_status as anthropic_http_status,
};
use crate::anthropic::models::error_types as anthropic_error_types;
use crate::config::{Config, EnforcementMode};
use crate::gateway::handlers::multipart_model;
use crate::gateway::AppState;

use super::models::{AuthenticatedKey, CachedKey, KeyStatus, UsageRecord};
use super::store::StoredVirtualKey;
use super::{
    compute_cost, sha256_hex, AccessError, BudgetError, RateLimitError, VirtualKeyManager,
};

/// Authentication failure for a presented virtual key.
///
/// Variants map to HTTP status codes per the design's error mapping table.
#[derive(Debug, Clone, PartialEq, Eq, thiserror::Error)]
pub enum AuthError {
    /// Key is unknown, unrecognized, or could not be validated (HTTP 401).
    #[error("Invalid or unrecognized key")]
    InvalidKey,
    /// Key's expiration timestamp is in the past (HTTP 403).
    #[error("Key has expired")]
    Expired,
    /// Key has been revoked (HTTP 403).
    #[error("Key has been revoked")]
    Revoked,
}

impl AuthError {
    /// HTTP status code for this error per the design mapping.
    pub fn status_code(&self) -> StatusCode {
        match self {
            AuthError::InvalidKey => StatusCode::UNAUTHORIZED,
            AuthError::Expired | AuthError::Revoked => StatusCode::FORBIDDEN,
        }
    }

    /// User-facing message for the JSON error body.
    fn client_message(&self) -> &'static str {
        match self {
            AuthError::InvalidKey => "Invalid or unrecognized API key",
            AuthError::Expired => "API key has expired",
            AuthError::Revoked => "API key has been revoked",
        }
    }
}

impl IntoResponse for AuthError {
    fn into_response(self) -> Response {
        (
            self.status_code(),
            Json(json!({ "error": self.client_message() })),
        )
            .into_response()
    }
}

impl AuthError {
    /// Anthropic-envelope response for this error (front-door Req 3.3):
    /// `InvalidKey` → 401 `authentication_error`; `Expired`/`Revoked` → 403
    /// `permission_error` per Anthropic semantics.
    fn into_anthropic_response(&self) -> Response {
        let kind = match self {
            AuthError::InvalidKey => anthropic_error_types::AUTHENTICATION_ERROR,
            AuthError::Expired | AuthError::Revoked => anthropic_error_types::PERMISSION_ERROR,
        };
        anthropic_shaped_error(kind, self.client_message())
    }
}

impl VirtualKeyManager {
    /// Authenticate a presented raw key value.
    ///
    /// Computes the SHA-256 hash of `raw_key`, resolves it via the DashMap
    /// cache (populating from the [`KeyStore`] on a miss), converts the stored
    /// record into an [`AuthenticatedKey`], and validates status/expiration:
    /// revoked → [`AuthError::Revoked`], expired-by-timestamp (or `Expired`
    /// status) → [`AuthError::Expired`], unknown hash → [`AuthError::InvalidKey`].
    ///
    /// On success the loaded key constraints are returned for downstream
    /// enforcement (budget, rate limits, model access).
    ///
    /// _Requirements: 2.1, 2.2, 2.3, 2.5, 2.6_
    pub async fn authenticate(&self, raw_key: &str) -> Result<AuthenticatedKey, AuthError> {
        let key_hash = sha256_hex(raw_key);

        // Fast path: cache hit. Re-validate status/expiry on every request so
        // that time-based expiry is honored even for cached entries. Cache is
        // invalidated on revoke/update/delete (see `invalidate_cache`).
        if let Some(cached) = self.cache.get(&key_hash) {
            return validate_status(&cached.key);
        }

        // Slow path: look up in the store. Treat store errors as an invalid key
        // (reject the request) while logging the underlying cause; the plaintext
        // key is never logged.
        let stored = match self.store.get_key_by_hash(&key_hash) {
            Ok(Some(stored)) => stored,
            Ok(None) => return Err(AuthError::InvalidKey),
            Err(e) => {
                tracing::warn!("virtual key store lookup failed: {e}");
                return Err(AuthError::InvalidKey);
            }
        };

        let authenticated = stored_to_authenticated(&stored);

        // Populate the cache with the loaded constraints for sub-5ms subsequent
        // lookups (Req 2.5).
        self.cache.insert(
            key_hash,
            CachedKey {
                key: authenticated.clone(),
                cached_at: Instant::now(),
            },
        );

        validate_status(&authenticated)
    }
}

/// Validate a loaded key's status and expiration, returning a clone on success.
///
/// Order matters: a revoked key yields [`AuthError::Revoked`] (403); otherwise a
/// key whose `expires_at` is in the past — or whose stored status is already
/// `Expired` — yields [`AuthError::Expired`] (403), even if the stored status is
/// still `Active`.
fn validate_status(key: &AuthenticatedKey) -> Result<AuthenticatedKey, AuthError> {
    if key.status == KeyStatus::Revoked {
        return Err(AuthError::Revoked);
    }
    if key.status == KeyStatus::Expired {
        return Err(AuthError::Expired);
    }
    if let Some(expires_at) = key.expires_at {
        if expires_at <= Utc::now() {
            return Err(AuthError::Expired);
        }
    }
    Ok(key.clone())
}

/// Convert a persisted key row into the in-memory [`AuthenticatedKey`] carried
/// through the request pipeline.
fn stored_to_authenticated(stored: &StoredVirtualKey) -> AuthenticatedKey {
    AuthenticatedKey {
        id: stored.id.clone(),
        name: stored.name.clone(),
        status: stored.status.clone(),
        budget_limit_usd: stored.budget_limit_usd,
        token_budget: stored.token_budget,
        budget_window: stored.budget_window.clone(),
        current_spend_usd: stored.current_spend_usd,
        current_tokens_used: stored.current_tokens_used,
        window_start: stored.window_start,
        requests_per_minute: stored.requests_per_minute,
        tokens_per_minute: stored.tokens_per_minute,
        model_access: stored.model_access_list.clone(),
        expires_at: stored.expires_at,
        loop_detection: stored.loop_detection.clone(),
    }
}

/// Extract a Bearer token value from the `Authorization` header, if present.
fn extract_bearer(request: &Request) -> Option<String> {
    request
        .headers()
        .get(header::AUTHORIZATION)
        .and_then(|v| v.to_str().ok())
        .and_then(|v| v.strip_prefix("Bearer "))
        .map(|t| t.trim().to_string())
        .filter(|t| !t.is_empty())
}

/// Whether a request path is one of the Anthropic Messages front-door
/// endpoints that additionally accept `x-api-key` as a credential alias
/// (anthropic-messages-front-door Req 5.4; design D3).
///
/// Exact, path-only matches: query strings (e.g. `/v1/messages?beta=true`)
/// are excluded by callers passing `uri().path()`. Any other `/v1/*` route —
/// notably `/v1/chat/completions` — is structurally outside the alias, so
/// Bearer remains the only accepted credential shape there.
fn is_messages_front_door_path(path: &str) -> bool {
    path == "/v1/messages" || path == "/v1/messages/count_tokens"
}

/// Extract the credential alias from the `x-api-key` header, applicable ONLY
/// on the Messages front-door paths (Req 5.4; design D3).
///
/// Returns `None` on every other path regardless of header presence, making
/// the alias structurally unable to apply to non-messages routes.
fn extract_x_api_key_alias(request: &Request) -> Option<String> {
    if !is_messages_front_door_path(request.uri().path()) {
        return None;
    }
    request
        .headers()
        .get("x-api-key")
        .and_then(|v| v.to_str().ok())
        .map(|t| t.trim().to_string())
        .filter(|t| !t.is_empty())
}

/// Resolve the virtual-key credential for a request (Req 5.4).
///
/// `Authorization: Bearer <vk_...>` wins when present and valid-shaped (i.e.
/// `vk_`-prefixed). Otherwise, on the Messages front-door paths only, a
/// `vk_`-prefixed `x-api-key` value is tried through the exact same
/// `authenticate` + enforcement pipeline. Non-`vk_` values are ignored for
/// both shapes, and non-messages paths never consult `x-api-key`.
fn extract_virtual_key_credential(request: &Request) -> Option<String> {
    extract_bearer(request)
        .filter(|t| t.starts_with("vk_"))
        .or_else(|| extract_x_api_key_alias(request))
        .filter(|t| t.starts_with("vk_"))
}

/// JSON 401 response used when `enforcement=required` and no valid virtual key
/// is presented (Req 11.2). Anthropic-shaped on the Messages front-door paths
/// (401 `authentication_error`, front-door Req 3.3); legacy envelope elsewhere.
fn required_auth_missing_response(messages_path: bool) -> Response {
    if messages_path {
        let body = anthropic_error_response(
            anthropic_error_types::AUTHENTICATION_ERROR,
            "Virtual key authentication is required",
        );
        let status = StatusCode::from_u16(anthropic_http_status(
            anthropic_error_types::AUTHENTICATION_ERROR,
        ))
        .unwrap_or(StatusCode::UNAUTHORIZED);
        return (status, Json(body)).into_response();
    }
    (
        StatusCode::UNAUTHORIZED,
        Json(json!({ "error": "Virtual key authentication is required" })),
    )
        .into_response()
}

/// Build an Anthropic-envelope error response (front-door Req 3.3) with the
/// status derived from the error type.
fn anthropic_shaped_error(kind: &str, message: impl Into<String>) -> Response {
    let body = anthropic_error_response(kind, message);
    let status =
        StatusCode::from_u16(anthropic_http_status(kind)).unwrap_or(StatusCode::BAD_REQUEST);
    (status, Json(body)).into_response()
}

/// Axum middleware enforcing the configured virtual key policy.
///
/// The [`VirtualKeyManager`] is resolved from [`AppState`]
/// (`state.virtual_key_manager`). On successful authentication the middleware
/// runs the full enforcement pipeline in order — model access → budget → rate
/// limit (Req 6.4, 5.5) — inserts the [`AuthenticatedKey`] into the request
/// extensions for downstream use, forwards the request, and records usage from
/// the provider response (Req 3.5, 3.6, 5.3, 9.1).
///
/// Enforcement modes:
/// - `disabled`: skip entirely, ignore any `vk_` key (Req 11.1, 11.5, 2.4)
/// - `optional`: validate a presented `vk_` key, else pass through (Req 11.4, 2.4)
/// - `required`: reject with 401 unless a valid `vk_` key is presented (Req 11.2)
///
/// _Requirements: 2.1, 2.4, 5.5, 6.4, 11.1, 11.2, 11.3, 11.4, 11.5_
pub async fn virtual_key_auth_middleware(
    State(state): State<AppState>,
    request: Request,
    next: Next,
) -> Result<Response, Response> {
    let enforcement = { state.config.read().await.virtual_keys.enforcement.clone() };

    // `disabled`: skip entirely and ignore any presented `vk_` key
    // (Req 11.1, 11.5, 2.4).
    if enforcement == EnforcementMode::Disabled {
        return Ok(next.run(request).await);
    }

    // Manager is always present on AppState (constructed at startup).
    let manager = Arc::clone(&state.virtual_key_manager);

    // Only `vk_`-prefixed credentials are treated as virtual keys. Bearer is
    // always considered; `x-api-key` is an alias accepted ONLY on the Messages
    // front-door paths (Req 5.4, design D3) and is structurally unreachable on
    // any other route. Bearer wins when present and valid-shaped.
    let vk_token = extract_virtual_key_credential(&request);
    // Anthropic Messages front-door paths get Anthropic-shaped middleware
    // rejections (front-door Req 3.3); every other route keeps the legacy
    // `{"error": "..."}` envelope.
    let messages_path = is_messages_front_door_path(request.uri().path());

    match enforcement {
        // Handled above; retained for exhaustiveness.
        EnforcementMode::Disabled => Ok(next.run(request).await),

        // `optional`: validate a presented `vk_` key; otherwise pass through
        // using provider keys directly (Req 11.4, 2.4).
        EnforcementMode::Optional => match vk_token {
            Some(token) => match manager.authenticate(&token).await {
                Ok(key) => {
                    Ok(enforce_authenticated(&state, &manager, key, request, next).await)
                }
                Err(err) => Err(auth_error_response(&err, messages_path)),
            },
            None => Ok(next.run(request).await),
        },

        // `required`: a valid `vk_` key is mandatory (Req 11.2).
        EnforcementMode::Required => {
            let Some(token) = vk_token else {
                return Err(required_auth_missing_response(messages_path));
            };
            match manager.authenticate(&token).await {
                Ok(key) => {
                    Ok(enforce_authenticated(&state, &manager, key, request, next).await)
                }
                Err(err) => Err(auth_error_response(&err, messages_path)),
            }
        }
    }
}

/// Render an [`AuthError`] for the client: Anthropic envelope on the Messages
/// front-door paths (front-door Req 3.3), legacy `{"error": "..."}` envelope
/// elsewhere.
fn auth_error_response(err: &AuthError, messages_path: bool) -> Response {
    if messages_path {
        err.into_anthropic_response()
    } else {
        err.clone().into_response()
    }
}

/// Maximum bytes buffered from a JSON response body when extracting token usage
/// for post-response recording. Streaming (SSE) responses are never buffered.
const RESPONSE_USAGE_BUFFER_LIMIT: usize = 64 * 1024 * 1024;

/// Whether the given headers declare a JSON content type.
fn is_json_content_type(headers: &axum::http::HeaderMap) -> bool {
    headers
        .get(header::CONTENT_TYPE)
        .and_then(|v| v.to_str().ok())
        .map(|v| v.to_ascii_lowercase().contains("application/json"))
        .unwrap_or(false)
}

/// Run the post-authentication enforcement pipeline for an authenticated key,
/// forward the request, and record usage from the response.
///
/// Order (design "Request Flow", Req 6.4): model access → budget → rate limit
/// → forward. Model access requires the requested model, extracted from JSON
/// request bodies; non-JSON requests (e.g. multipart audio/file uploads) skip
/// the model-access check but still enforce budget and rate limits. Any
/// enforcement rejection short-circuits with the mapped 403/429 response.
async fn enforce_authenticated(
    state: &AppState,
    manager: &Arc<VirtualKeyManager>,
    key: AuthenticatedKey,
    request: Request,
    next: Next,
) -> Response {
    let messages_path = is_messages_front_door_path(request.uri().path());
    let content_type = request
        .headers()
        .get(header::CONTENT_TYPE)
        .and_then(|v| v.to_str().ok())
        .unwrap_or("")
        .to_ascii_lowercase();
    let json_body = content_type.contains("application/json");
    let multipart_body = content_type.starts_with("multipart/form-data");

    // Buffer + inspect request bodies to extract the requested model group.
    let (request, requested_model) = if json_body || multipart_body {
        // Read from the lock-free snapshot rather than `config.read().await`:
        // this middleware wraps every API route, and the config lock is
        // write-preferring.
        let max_body_bytes = state.runtime_limits.max_request_size_bytes();
        let (parts, body) = request.into_parts();
        let bytes = match to_bytes(body, max_body_bytes).await {
            Ok(b) => b,
            Err(_) => {
                return (
                    StatusCode::BAD_REQUEST,
                    Json(json!({ "error": "Invalid or oversized request body" })),
                )
                    .into_response();
            }
        };
        let model = if json_body {
            serde_json::from_slice::<serde_json::Value>(&bytes)
                .ok()
                .and_then(|v| v.get("model").and_then(|m| m.as_str()).map(str::to_string))
        } else {
            let ct = parts
                .headers
                .get(header::CONTENT_TYPE)
                .and_then(|v| v.to_str().ok())
                .unwrap_or("");
            multipart_model(ct, &bytes).ok()
        };

        // Req 6.4: model access is checked BEFORE budget / rate limit so denied
        // requests consume no budget or rate-limit capacity.
 if let Some(model) = &model {
 if let Err(err) = manager.check_model_access(&key, model) {
 return access_denied_response(&err, messages_path);
 }
 }

 (Request::from_parts(parts, Body::from(bytes)), model)
 } else {
 (request, None)
 };

 // Budget (Req 3.2 / 4.2) then per-key rate limit (Req 5.1-5.5).
 if let Err(err) = manager.check_budget(&key) {
 return budget_exhausted_response(&err, messages_path);
 }
 if let Err(err) = manager.check_rate_limit(&key) {
 return rate_limited_response(&err, messages_path);
 }

    // Carry the authenticated key downstream (Req 2.1) and forward.
    let key_id = key.id.clone();
    let mut request = request;
    request.extensions_mut().insert(key);
    let response = next.run(request).await;

    record_usage_from_response(state, manager, &key_id, requested_model, response).await
}

/// Map an [`AccessError`] to its HTTP 403 response (design HTTP Error
/// Mapping). Anthropic `permission_error` envelope on the Messages front-door
/// paths (front-door Req 3.3).
fn access_denied_response(err: &AccessError, messages_path: bool) -> Response {
    if messages_path {
        return anthropic_shaped_error(anthropic_error_types::PERMISSION_ERROR, err.to_string());
    }
    let AccessError::ModelDenied { model, allowed } = err;
    (
        StatusCode::FORBIDDEN,
        Json(json!({
            "error": "Model not permitted",
            "model": model,
            "allowed": allowed,
        })),
    )
        .into_response()
}

/// Map a [`BudgetError`] to HTTP 429 (design HTTP Error Mapping).
/// Anthropic `rate_limit_error` envelope on the Messages front-door paths
/// (front-door Req 3.3).
fn budget_exhausted_response(err: &BudgetError, messages_path: bool) -> Response {
    if messages_path {
        return anthropic_shaped_error(anthropic_error_types::RATE_LIMIT_ERROR, err.to_string());
    }
    let message = match err {
        BudgetError::BudgetExhausted => "USD budget limit reached",
        BudgetError::TokenBudgetExhausted => "Token budget limit reached",
    };
    (
        StatusCode::TOO_MANY_REQUESTS,
        Json(json!({ "error": message })),
    )
        .into_response()
}

/// Map a [`RateLimitError`] to HTTP 429 + `Retry-After` (design HTTP Error
/// Mapping, Req 5.2 / 5.4). Anthropic `rate_limit_error` envelope on the
/// Messages front-door paths (front-door Req 3.3); the `Retry-After` header
/// is preserved in both shapes.
fn rate_limited_response(err: &RateLimitError, messages_path: bool) -> Response {
    let (message, retry_after) = match err {
        RateLimitError::RpmExceeded {
            retry_after_seconds,
        } => ("Rate limit exceeded", *retry_after_seconds),
        RateLimitError::TpmExceeded {
            retry_after_seconds,
        } => ("Token rate limit exceeded", *retry_after_seconds),
    };
    let headers = [(header::RETRY_AFTER, retry_after.to_string())];
    if messages_path {
        let body = anthropic_error_response(anthropic_error_types::RATE_LIMIT_ERROR, message);
        let status = StatusCode::from_u16(anthropic_http_status(
            anthropic_error_types::RATE_LIMIT_ERROR,
        ))
        .unwrap_or(StatusCode::TOO_MANY_REQUESTS);
        return (status, headers, Json(body)).into_response();
    }
    (StatusCode::TOO_MANY_REQUESTS, headers, Json(json!({ "error": message }))).into_response()
}

/// Record usage from a completed provider response, when possible.
///
/// Only JSON (non-streaming) responses carry a parseable `usage` object; SSE
/// streaming responses are passed through untouched and usage is skipped with a
/// warning (Req 3.6). Recording runs off the response path (spawned) so it does
/// not add latency. The TPM rolling window is updated inline (Req 5.3).
async fn record_usage_from_response(
    state: &AppState,
    manager: &Arc<VirtualKeyManager>,
    key_id: &str,
    requested_model: Option<String>,
    response: Response,
) -> Response {
    // Non-JSON (streaming/multipart) responses lack a parseable usage object.
    if !is_json_content_type(response.headers()) {
        tracing::warn!(
            key_id = %key_id,
            "virtual key usage not recorded: response is not JSON (streaming responses omit token usage)"
        );
        return response;
    }
    // Without the requested model group we cannot attribute usage.
    let Some(model_group) = requested_model else {
        return response;
    };

    let (parts, body) = response.into_parts();
    let bytes = match to_bytes(body, RESPONSE_USAGE_BUFFER_LIMIT).await {
        Ok(b) => b,
        Err(e) => {
            tracing::warn!(error = %e, "failed to buffer response body for virtual key usage recording");
            return Response::from_parts(parts, Body::empty());
        }
    };

    if let Some((record, tpm_tokens)) =
        build_usage_record(state, key_id, &model_group, &bytes).await
    {
        let manager = Arc::clone(manager);
        // TPM rolling window reflects real consumption (Req 5.3).
        manager.record_tpm_usage(&record.key_id, tpm_tokens);
        // Persist spend/token/request counters off the response path (Req 3.5).
        tokio::spawn(async move {
            if let Err(e) = manager.record_usage(record).await {
                tracing::warn!(error = %e, "failed to record virtual key usage");
            }
        });
    }

    Response::from_parts(parts, Body::from(bytes))
}

/// Build a [`UsageRecord`] from a buffered JSON response body, or `None` when
/// token usage is absent (Req 3.6 — skip recording + warn).
async fn build_usage_record(
    state: &AppState,
    key_id: &str,
    model_group: &str,
    body: &[u8],
) -> Option<(UsageRecord, u64)> {
    let value: serde_json::Value = serde_json::from_slice(body).ok()?;
    let usage = value.get("usage");
    // OpenAI-shaped counters first; Anthropic Messages bodies carry
    // `input_tokens`/`output_tokens` instead, so they serve as fallbacks when
    // the OpenAI spellings are absent or zero (anthropic-messages-front-door:
    // without this, VK spend/token budgets/TPM were never recorded on the
    // /v1/messages front door).
    let input_tokens = usage_tokens(usage, "prompt_tokens", "input_tokens");
    let output_tokens = usage_tokens(usage, "completion_tokens", "output_tokens");

    // Req 3.6: a response without usage counts is not recorded.
    // Embedding responses commonly omit completion_tokens; accept input-only usage.
    let (input_tokens, output_tokens) = match (input_tokens, output_tokens) {
        (Some(i), Some(o)) if i.saturating_add(o) > 0 => (i, o),
        (Some(i), None) if i > 0 => (i, 0),
        (Some(i), Some(0)) if i > 0 => (i, 0),
        _ => {
            tracing::warn!(
                key_id = %key_id,
                "virtual key usage not recorded: provider response missing token usage"
            );
            return None;
        }
    };

    // Prefer the gateway-annotated responded model; fall back to the envelope
    // model, then to the requested model group.
    let responded_model = value
        .get("gateway_responded_model")
        .and_then(serde_json::Value::as_str)
        .or_else(|| value.get("model").and_then(serde_json::Value::as_str))
        .unwrap_or(model_group)
        .to_string();

    // Cost via compute_cost using the model group's configured rates for the
    // responded model; fall back to the router's precomputed `gateway_cost`.
    let cost_usd = {
        let cfg = state.config.read().await;
        match lookup_model_rates(&cfg, model_group, &responded_model) {
            Some((input_rate, output_rate)) => {
                compute_cost(input_tokens, output_tokens, input_rate, output_rate)
            }
            None => value
                .get("gateway_cost")
                .and_then(serde_json::Value::as_f64)
                .unwrap_or(0.0),
        }
    };

    let record = UsageRecord {
        key_id: key_id.to_string(),
        model_group: model_group.to_string(),
        model: responded_model,
        input_tokens,
        output_tokens,
        cost_usd,
        timestamp: Utc::now(),
    };
    Some((record, input_tokens.saturating_add(output_tokens)))
}

/// Read a token counter from a `usage` object: the primary (OpenAI) key wins
/// when present and non-zero; otherwise the Anthropic fallback key is
/// consulted. Returns `None` when neither spelling carries a non-zero count.
fn usage_tokens(
    usage: Option<&serde_json::Value>,
    primary: &str,
    fallback: &str,
) -> Option<u64> {
    let primary = usage
        .and_then(|u| u.get(primary))
        .and_then(serde_json::Value::as_u64);
    if primary.is_some_and(|tokens| tokens > 0) {
        return primary;
    }
    usage
        .and_then(|u| u.get(fallback))
        .and_then(serde_json::Value::as_u64)
        .filter(|tokens| *tokens > 0)
}

/// Resolve `(input_rate, output_rate)` per-million-token cost rates for the
/// responded model within a model group, falling back to the group's first
/// model when the exact model is not found.
pub fn lookup_model_rates(
    config: &Config,
    model_group: &str,
    responded_model: &str,
) -> Option<(f64, f64)> {
    let group = config.model_groups.iter().find(|g| g.name == model_group)?;
    let model = group
        .models
        .iter()
        .find(|m| m.model == responded_model)
        .or_else(|| group.models.first())?;
    Some((
        model.cost_per_million_input_tokens,
        model.cost_per_million_output_tokens,
    ))
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::secrets;
    use crate::virtual_keys::models::{BudgetWindow, CreateKeyParams, ExpiresIn};
    use crate::virtual_keys::store::KeyUpdates;
    use chrono::Duration;
    use tempfile::NamedTempFile;

    fn temp_manager() -> (VirtualKeyManager, NamedTempFile) {
        // Ensure encryption has a key available for create_key round-trips.
        let temp = NamedTempFile::new().unwrap();
        let mgr = VirtualKeyManager::new(temp.path()).unwrap();
        (mgr, temp)
    }

    fn defaults() -> CreateKeyParams {
        CreateKeyParams {
            name: None,
            budget_limit_usd: None,
            token_budget: None,
            budget_window: None,
            requests_per_minute: None,
            tokens_per_minute: None,
            model_access: None,
            expires_in: None,
            loop_detection: None,
        }
    }

    /// A stored key authenticates successfully and its constraints are loaded.
    #[tokio::test]
    async fn stored_key_authenticates() {
        let (mgr, _tmp) = temp_manager();
        let params = CreateKeyParams {
            name: Some("svc".to_string()),
            budget_limit_usd: Some(10.0),
            token_budget: Some(1_000),
            budget_window: Some(BudgetWindow::Daily),
            model_access: Some(vec!["gpt-4".to_string()]),
            ..defaults()
        };
        let created = mgr.create_key(params).await.unwrap();

        let authed = mgr.authenticate(&created.key).await.unwrap();
        assert_eq!(authed.id, created.id);
        assert_eq!(authed.status, KeyStatus::Active);
        assert_eq!(authed.budget_limit_usd, Some(10.0));
        assert_eq!(authed.token_budget, Some(1_000));
        assert_eq!(
            authed.model_access.as_deref(),
            Some(&["gpt-4".to_string()][..])
        );
    }

    /// An unknown key hash fails with `InvalidKey` (→ 401).
    #[tokio::test]
    async fn unknown_key_is_invalid() {
        let (mgr, _tmp) = temp_manager();
        let err = mgr.authenticate("vk_does_not_exist").await.unwrap_err();
        assert_eq!(err, AuthError::InvalidKey);
        assert_eq!(err.status_code(), StatusCode::UNAUTHORIZED);
    }

    /// A key whose `expires_at` is in the past fails with `Expired` (→ 403),
    /// even though the stored status remains `Active`.
    #[tokio::test]
    async fn expired_key_by_timestamp() {
        let (mgr, _tmp) = temp_manager();
        let created = mgr.create_key(defaults()).await.unwrap();

        // Backdate the expiration to the past directly in the store.
        let past = Utc::now() - Duration::days(1);
        mgr.store
            .update_key(
                &created.id,
                &KeyUpdates {
                    expires_at: Some(Some(past)),
                    ..Default::default()
                },
            )
            .unwrap();

        let err = mgr.authenticate(&created.key).await.unwrap_err();
        assert_eq!(err, AuthError::Expired);
        assert_eq!(err.status_code(), StatusCode::FORBIDDEN);
    }

    /// A revoked key fails with `Revoked` (→ 403).
    #[tokio::test]
    async fn revoked_key_rejected() {
        let (mgr, _tmp) = temp_manager();
        let created = mgr.create_key(defaults()).await.unwrap();

        mgr.store
            .update_key(
                &created.id,
                &KeyUpdates {
                    status: Some(KeyStatus::Revoked),
                    ..Default::default()
                },
            )
            .unwrap();

        let err = mgr.authenticate(&created.key).await.unwrap_err();
        assert_eq!(err, AuthError::Revoked);
        assert_eq!(err.status_code(), StatusCode::FORBIDDEN);
    }

    /// A future expiration authenticates normally.
    #[tokio::test]
    async fn future_expiry_authenticates() {
        let (mgr, _tmp) = temp_manager();
        let created = mgr
            .create_key(CreateKeyParams {
                expires_in: Some(ExpiresIn::OneYear),
                ..defaults()
            })
            .await
            .unwrap();

        let authed = mgr.authenticate(&created.key).await.unwrap();
        assert_eq!(authed.id, created.id);
    }

    /// The cache-hit path returns the same result as the initial store lookup.
    #[tokio::test]
    async fn cache_hit_path_returns_key() {
        let (mgr, _tmp) = temp_manager();
        let created = mgr.create_key(defaults()).await.unwrap();

        // First call populates the cache from the store.
        let first = mgr.authenticate(&created.key).await.unwrap();
        // Delete the underlying store row; a cache hit must still resolve.
        mgr.store.delete_key(&created.id).unwrap();
        let second = mgr.authenticate(&created.key).await.unwrap();
        assert_eq!(first.id, second.id);

        // After invalidation the (now-deleted) key is no longer resolvable.
        mgr.invalidate_cache(&VirtualKeyManager::hash_key(&created.key));
        let err = mgr.authenticate(&created.key).await.unwrap_err();
        assert_eq!(err, AuthError::InvalidKey);
    }

    /// Encryption round-trip smoke check so `secrets` linkage is exercised.
    #[test]
    fn secrets_available_for_tests() {
        let enc = secrets::encrypt_provider_secret("vk_probe").unwrap();
        assert_eq!(secrets::decrypt_provider_secret(&enc).unwrap(), "vk_probe");
    }

    // ---------------------------------------------------------------------
    // x-api-key credential alias (anthropic-messages-front-door Req 5.4, D3)
    // ---------------------------------------------------------------------

    use axum::http::Request as HttpRequest;

    fn request_for(path_and_query: &str, headers: &[(&str, &str)]) -> Request {
        let builder = HttpRequest::builder()
            .method("POST")
            .uri(path_and_query);
        let builder = headers
            .iter()
            .fold(builder, |b, (k, v)| b.header(*k, *v));
        builder.body(Body::empty()).unwrap()
    }

    /// Path predicate admits exactly the two Messages front-door endpoints,
    /// path-only (query strings ignored).
    #[test]
    fn front_door_path_predicate_is_exact_and_path_only() {
        assert!(is_messages_front_door_path("/v1/messages"));
        assert!(is_messages_front_door_path("/v1/messages/count_tokens"));
        // Everything else — including near-misses — is excluded.
        for path in [
            "/v1/chat/completions",
            "/v1/messages/",
            "/v1/messages/count_tokens/",
            "/v1/v1/messages",
            "/v1/models",
            "/v1/embeddings",
            "/messages",
        ] {
            assert!(!is_messages_front_door_path(path), "{path} must not match");
        }
    }

    /// `x-api-key` resolves on both front-door paths, with query strings
    /// ignored (Req 5.4).
    #[test]
    fn x_api_key_alias_extracted_on_front_door_paths() {
        for uri in ["/v1/messages", "/v1/messages?beta=true", "/v1/messages/count_tokens"] {
            let request = request_for(uri, &[("x-api-key", "vk_alias")]);
            assert_eq!(
                extract_virtual_key_credential(&request).as_deref(),
                Some("vk_alias"),
                "alias must resolve on {uri}"
            );
        }
    }

    /// On non-messages paths the alias is structurally ignored: a request
    /// carrying ONLY `x-api-key` resolves no credential (Req 5.4, D3), so in
    /// `required` mode it still 401s and in `optional` mode it passes through
    /// unauthenticated — exactly the pre-alias Bearer-only behavior.
    #[test]
    fn x_api_key_alias_ignored_on_non_front_door_paths() {
        let request = request_for("/v1/chat/completions", &[("x-api-key", "vk_alias")]);
        assert_eq!(extract_virtual_key_credential(&request), None);
    }

    /// Bearer wins over `x-api-key` when both are present and Bearer is
    /// valid-shaped; a valid-shaped alias also wins over a non-`vk_` Bearer.
    #[test]
    fn bearer_takes_precedence_over_alias() {
        let request = request_for(
            "/v1/messages",
            &[("authorization", "Bearer vk_bearer"), ("x-api-key", "vk_alias")],
        );
        assert_eq!(extract_virtual_key_credential(&request).as_deref(), Some("vk_bearer"));

        // Non-`vk_` Bearer is not a virtual key; the front-door alias applies.
        let request = request_for(
            "/v1/messages",
            &[("authorization", "Bearer sk-upstream"), ("x-api-key", "vk_alias")],
        );
        assert_eq!(extract_virtual_key_credential(&request).as_deref(), Some("vk_alias"));
    }

    /// Bearer continues to resolve everywhere, including the messages routes,
    /// with or without a concurrent `x-api-key` header.
    #[test]
    fn bearer_resolves_everywhere() {
        for uri in ["/v1/chat/completions", "/v1/messages", "/v1/messages/count_tokens"] {
            let request = request_for(uri, &[("authorization", "Bearer vk_bearer")]);
            assert_eq!(
                extract_virtual_key_credential(&request).as_deref(),
                Some("vk_bearer"),
                "Bearer must resolve on {uri}"
            );
        }
    }

/// Non-`vk_` `x-api-key` values are ignored on front-door paths too — the
/// same `vk_` prefix filtering the Bearer path applies (Req 5.4).
#[test]
fn non_vk_alias_values_are_ignored() {
    let request = request_for("/v1/messages", &[("x-api-key", "sk-ant-upstream")]);
    assert_eq!(extract_virtual_key_credential(&request), None);
}

// ---------------------------------------------------------------------
// Anthropic-shaped usage recording (budget bypass fix)
// ---------------------------------------------------------------------

/// Minimal YAML config (same shape as the front-door integration harness)
/// whose `anthropic-group` carries rates `lookup_model_rates` can resolve.
fn anthropic_usage_test_config() -> crate::config::Config {
    let mut config: crate::config::Config = serde_yaml::from_str(
        "server:\n  host: 127.0.0.1\n  port: 0\nproviders:\n  - name: p\n    type: openai\n    base_url: http://localhost\n    timeout_seconds: 30\nmodel_groups:\n  - name: anthropic-group\n    models:\n      - provider: p\n        model: claude-sonnet-4-6\n        cost_per_million_input_tokens: 3.0\n        cost_per_million_output_tokens: 6.0\n",
    )
    .unwrap();
    // Isolated SQLite databases so parallel tests never share ./logs.db.
    let dir = tempfile::tempdir().unwrap().keep();
    config.logging.database_path = dir.join("logs.db").to_string_lossy().into_owned();
    config.virtual_keys.database_path = dir.join("keys.db").to_string_lossy().into_owned();
    config
}

/// `build_usage_record` accepts Anthropic Messages `input_tokens`/
/// `output_tokens` when the OpenAI spellings are absent, so /v1/messages
/// responses record VK spend/token budgets/TPM instead of silently
/// skipping (the budget bypass this fixes).
#[tokio::test]
async fn anthropic_shaped_usage_is_recorded() {
    let server = crate::gateway::GatewayServer::new(anthropic_usage_test_config(), None)
        .await
        .expect("test server");
    let body = serde_json::json!({
        "id": "msg_usage",
        "type": "message",
        "role": "assistant",
        "model": "claude-sonnet-4-6",
        "content": [{ "type": "text", "text": "hi" }],
        "usage": { "input_tokens": 21, "output_tokens": 4 }
    });
    let bytes = serde_json::to_vec(&body).unwrap();
    let (record, tpm) = build_usage_record(
        &server.state,
        "key_anthropic",
        "anthropic-group",
        &bytes,
    )
    .await
    .expect("Anthropic-shaped usage must be recorded");
    assert_eq!(record.input_tokens, 21);
    assert_eq!(record.output_tokens, 4);
    assert_eq!(record.model, "claude-sonnet-4-6");
    assert_eq!(tpm, 21 + 4);
    // Rates resolve from the group config: $3/M in, $6/M out.
    assert!((record.cost_usd - compute_cost(21, 4, 3.0, 6.0)).abs() < 1e-12);
}

/// A body carrying BOTH spellings prefers the non-zero OpenAI counters.
#[tokio::test]
async fn openai_spelling_wins_when_present() {
    let server = crate::gateway::GatewayServer::new(anthropic_usage_test_config(), None)
        .await
        .expect("test server");
    let body = serde_json::json!({
        "model": "claude-sonnet-4-6",
        "usage": {
            "prompt_tokens": 10,
            "completion_tokens": 5,
            "input_tokens": 999,
            "output_tokens": 999
        }
    });
    let bytes = serde_json::to_vec(&body).unwrap();
    let (record, _) = build_usage_record(
        &server.state,
        "key_mixed",
        "anthropic-group",
        &bytes,
    )
    .await
    .expect("mixed-shape usage must be recorded");
    assert_eq!(record.input_tokens, 10);
    assert_eq!(record.output_tokens, 5);
}

    // ---------------------------------------------------------------------
    // Property-based tests
    // ---------------------------------------------------------------------

    use proptest::prelude::*;

    /// Build a fresh tokio runtime for driving the async `authenticate` API
    /// inside a (synchronous) proptest case body.
    fn rt() -> tokio::runtime::Runtime {
        tokio::runtime::Runtime::new().unwrap()
    }

    proptest! {
        #![proptest_config(ProptestConfig {
            cases: 64,
            .. ProptestConfig::default()
        })]

        // Feature: virtual-key-management, Property 5: Authentication Correctness
        //
        // For any virtual key stored in the Key_Store, presenting that key in an
        // Authorization Bearer header SHALL result in successful authentication
        // returning the correct key constraints.
        //
        // **Validates: Requirements 2.1**

        /// Create a key with arbitrary (valid) constraints, then authenticate
        /// the returned plaintext: it SHALL succeed and echo back the correct
        /// key id and constraints. Each case uses a fresh `temp_manager` for
        /// isolation.
        #[test]
        fn prop_stored_key_authenticates(
            name in proptest::option::of("[a-zA-Z0-9 ]{1,128}"),
            budget_cents in proptest::option::of(1u64..=100_000_000u64),
            token_budget in proptest::option::of(1u64..=999_999_999u64),
        ) {
            let budget_limit_usd = budget_cents.map(|c| c as f64 / 100.0);
            let expected_name = name.clone();

            let (created, authed) = rt().block_on(async {
                let (mgr, _tmp) = temp_manager();
                let params = CreateKeyParams {
                    name,
                    budget_limit_usd,
                    token_budget,
                    ..defaults()
                };
                let created = mgr.create_key(params).await.unwrap();
                let authed = mgr.authenticate(&created.key).await;
                (created, authed)
            });

            let authed = authed.expect("stored key must authenticate");
            prop_assert_eq!(&authed.id, &created.id);
            prop_assert_eq!(authed.status, KeyStatus::Active);
            prop_assert_eq!(authed.name, expected_name);
            prop_assert_eq!(authed.budget_limit_usd, budget_limit_usd);
            prop_assert_eq!(authed.token_budget, token_budget);
        }
    }

    proptest! {
        #![proptest_config(ProptestConfig {
            cases: 64,
            .. ProptestConfig::default()
        })]

        // Feature: virtual-key-management, Property 5: Authentication Correctness
        //
        // For any string not matching a stored key's plaintext, authentication
        // SHALL fail with HTTP 401 (AuthError::InvalidKey).
        //
        // **Validates: Requirements 2.2**

        /// With exactly one key stored, authenticating an arbitrary string that
        /// is not that key's plaintext SHALL fail with `InvalidKey` (401). The
        /// `prop_assume!` guards against the astronomically unlikely case where
        /// the random string equals the generated key.
        #[test]
        fn prop_non_stored_string_is_invalid(random in any::<String>()) {
            let (created_key, result) = rt().block_on(async {
                let (mgr, _tmp) = temp_manager();
                let created = mgr.create_key(defaults()).await.unwrap();
                let result = mgr.authenticate(&random).await;
                (created.key, result)
            });

            // Ensure the random string does not collide with the stored key.
            prop_assume!(random != created_key);

            let err = result.expect_err("non-stored string must not authenticate");
            prop_assert_eq!(err.clone(), AuthError::InvalidKey);
            prop_assert_eq!(err.status_code(), StatusCode::UNAUTHORIZED);
        }
    }

    proptest! {
        #![proptest_config(ProptestConfig {
            cases: 64,
            .. ProptestConfig::default()
        })]

        // Feature: virtual-key-management, Property 5: Authentication Correctness
        //
        // For any key whose `expires_at` is in the past, authentication SHALL
        // fail with HTTP 403 (AuthError::Expired).
        //
        // **Validates: Requirements 2.3**

        /// Create a key, backdate its `expires_at` by an arbitrary positive
        /// duration, then authenticate: it SHALL fail with `Expired` (403).
        #[test]
        fn prop_expired_key_fails(days_in_past in 1i64..=3_650i64) {
            let result = rt().block_on(async {
                let (mgr, _tmp) = temp_manager();
                let created = mgr.create_key(defaults()).await.unwrap();
                let past = Utc::now() - Duration::days(days_in_past);
                mgr.store
                    .update_key(
                        &created.id,
                        &KeyUpdates {
                            expires_at: Some(Some(past)),
                            ..Default::default()
                        },
                    )
                    .unwrap();
                mgr.authenticate(&created.key).await
            });

            let err = result.expect_err("expired key must not authenticate");
            prop_assert_eq!(err.clone(), AuthError::Expired);
            prop_assert_eq!(err.status_code(), StatusCode::FORBIDDEN);
        }
    }
}
