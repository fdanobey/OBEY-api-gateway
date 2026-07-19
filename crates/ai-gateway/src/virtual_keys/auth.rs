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

use crate::config::{Config, EnforcementMode};
use crate::gateway::AppState;

use super::models::{AuthenticatedKey, CachedKey, KeyStatus, UsageRecord};
use super::store::StoredVirtualKey;
use super::{compute_cost, sha256_hex, AccessError, BudgetError, RateLimitError, VirtualKeyManager};

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

/// JSON 401 response used when `enforcement=required` and no valid virtual key
/// is presented (Req 11.2).
fn required_auth_missing_response() -> Response {
    (
        StatusCode::UNAUTHORIZED,
        Json(json!({ "error": "Virtual key authentication is required" })),
    )
        .into_response()
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

    // Only Bearer tokens with the `vk_` prefix are treated as virtual keys.
    let vk_token = extract_bearer(&request).filter(|t| t.starts_with("vk_"));

    match enforcement {
        // Handled above; retained for exhaustiveness.
        EnforcementMode::Disabled => Ok(next.run(request).await),

        // `optional`: validate a presented `vk_` key; otherwise pass through
        // using provider keys directly (Req 11.4, 2.4).
        EnforcementMode::Optional => match vk_token {
            Some(token) => match manager.authenticate(&token).await {
                Ok(key) => Ok(enforce_authenticated(&state, &manager, key, request, next).await),
                Err(err) => Err(err.into_response()),
            },
            None => Ok(next.run(request).await),
        },

        // `required`: a valid `vk_` key is mandatory (Req 11.2).
        EnforcementMode::Required => {
            let Some(token) = vk_token else {
                return Err(required_auth_missing_response());
            };
            match manager.authenticate(&token).await {
                Ok(key) => Ok(enforce_authenticated(&state, &manager, key, request, next).await),
                Err(err) => Err(err.into_response()),
            }
        }
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
    let json_body = is_json_content_type(request.headers());

    // Buffer + inspect JSON bodies to extract the requested model group.
    let (request, requested_model) = if json_body {
        let max_body_bytes = {
            let cfg = state.config.read().await;
            (cfg.server.max_request_size_mb as usize).saturating_mul(1024 * 1024)
        };
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
        let model = serde_json::from_slice::<serde_json::Value>(&bytes)
            .ok()
            .and_then(|v| {
                v.get("model")
                    .and_then(|m| m.as_str())
                    .map(str::to_string)
            });

        // Req 6.4: model access is checked BEFORE budget / rate limit so denied
        // requests consume no budget or rate-limit capacity.
        if let Some(model) = &model {
            if let Err(err) = manager.check_model_access(&key, model) {
                return access_denied_response(&err);
            }
        }

        (Request::from_parts(parts, Body::from(bytes)), model)
    } else {
        (request, None)
    };

    // Budget (Req 3.2 / 4.2) then per-key rate limit (Req 5.1-5.5).
    if let Err(err) = manager.check_budget(&key) {
        return budget_exhausted_response(&err);
    }
    if let Err(err) = manager.check_rate_limit(&key) {
        return rate_limited_response(&err);
    }

    // Carry the authenticated key downstream (Req 2.1) and forward.
    let key_id = key.id.clone();
    let mut request = request;
    request.extensions_mut().insert(key);
    let response = next.run(request).await;

    record_usage_from_response(state, manager, &key_id, requested_model, response).await
}

/// Map an [`AccessError`] to its HTTP 403 response (design HTTP Error Mapping).
fn access_denied_response(err: &AccessError) -> Response {
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
fn budget_exhausted_response(err: &BudgetError) -> Response {
    let message = match err {
        BudgetError::BudgetExhausted => "USD budget limit reached",
        BudgetError::TokenBudgetExhausted => "Token budget limit reached",
    };
    (StatusCode::TOO_MANY_REQUESTS, Json(json!({ "error": message }))).into_response()
}

/// Map a [`RateLimitError`] to HTTP 429 + `Retry-After` (design HTTP Error
/// Mapping, Req 5.2 / 5.4).
fn rate_limited_response(err: &RateLimitError) -> Response {
    let (message, retry_after) = match err {
        RateLimitError::RpmExceeded {
            retry_after_seconds,
        } => ("Rate limit exceeded", *retry_after_seconds),
        RateLimitError::TpmExceeded {
            retry_after_seconds,
        } => ("Token rate limit exceeded", *retry_after_seconds),
    };
    (
        StatusCode::TOO_MANY_REQUESTS,
        [(header::RETRY_AFTER, retry_after.to_string())],
        Json(json!({ "error": message })),
    )
        .into_response()
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
    let input_tokens = usage
        .and_then(|u| u.get("prompt_tokens"))
        .and_then(serde_json::Value::as_u64);
    let output_tokens = usage
        .and_then(|u| u.get("completion_tokens"))
        .and_then(serde_json::Value::as_u64);

    // Req 3.6: a response without usage counts is not recorded.
    let (input_tokens, output_tokens) = match (input_tokens, output_tokens) {
        (Some(i), Some(o)) if i.saturating_add(o) > 0 => (i, o),
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

/// Resolve `(input_rate, output_rate)` per-million-token cost rates for the
/// responded model within a model group, falling back to the group's first
/// model when the exact model is not found.
fn lookup_model_rates(
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
        assert_eq!(authed.model_access.as_deref(), Some(&["gpt-4".to_string()][..]));
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
            cases: 100,
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
            cases: 100,
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
            cases: 100,
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
