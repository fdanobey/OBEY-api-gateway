use axum::{
    extract::{Json, Path, Query, Request, State},
    http::{header, StatusCode},
    middleware::{self, Next},
    response::{IntoResponse, Response},
    routing::{get, post},
    Router,
};
use base64::Engine;
use rust_embed::Embed;
use serde_json::json;

use crate::config::{load_and_validate_config, save_config, Config};
use crate::gateway::apply_runtime_config_update;
use crate::gateway::AppState;
use crate::secrets;

/// Embedded admin panel static assets (Req 13.17, 13.18, 1.3).
#[derive(Embed)]
#[folder = "src/admin/static/"]
struct AdminAssets;

/// Build the admin panel router.
/// All routes are relative — the caller nests them under the configured admin path.
///
/// Config API endpoints (Req 13.11-13.15, 32.1-32.7):
///   GET    /config          — return current configuration
///   PUT    /config          — update configuration (validate, write YAML, apply)
///   POST   /config/validate — validate configuration without applying
///   POST   /config/reload   — hot-reload from disk
///   GET    /config/export   — download YAML
///   POST   /config/import   — upload YAML, validate, return parsed config
///
/// Authentication (Req 35.1-35.7):
///   When admin.auth.enabled is true, all endpoints require HTTP Basic Auth.
///   Credentials are resolved from environment variables at runtime.
pub fn admin_routes(state: AppState) -> Router<AppState> {
    let config_api = Router::new()
        .route("/", get(get_config).put(update_config))
        .route("/validate", post(validate_config))
        .route("/reload", post(reload_config))
        .route("/export", get(export_config))
        .route("/import", post(import_config));

    // Virtual key admin API (Req 10.1-10.6). Self-contained router owning an
    // `Arc<VirtualKeyManager>`; nested here so the admin auth `route_layer`
    // below protects every `/keys` endpoint (Req 10.2).
    let virtual_keys_api =
        crate::virtual_keys::admin::routes(std::sync::Arc::clone(&state.virtual_key_manager));

    let loop_detection_api = crate::loop_detection::admin::routes();

    Router::new()
        .nest("/config", config_api)
        .nest("/loop-detection", loop_detection_api)
        .nest_service("/keys", virtual_keys_api)
        .route("/providers/models", get(proxy_provider_models))
        .route("/test-connection", post(test_connection))
        .route("/oauth/openai/login", post(oauth_login))
        .route("/oauth/openai/complete", post(oauth_complete))
        .route("/oauth/openai/status", get(oauth_status))
        .route("/oauth/openai/usage", get(oauth_usage))
        .route("/oauth/openai/logout", post(oauth_logout))
        .route("/", get(index_handler))
        .route("/{*path}", get(static_handler))
        .route_layer(middleware::from_fn_with_state(state, admin_auth_middleware))
}

/// HTTP Basic Authentication middleware for admin panel (Req 35.1-35.7).
///
/// When `admin.auth.enabled` is true, requires valid Basic credentials.
/// When disabled (default, Req 35.7), passes all requests through.
/// Returns 401 with WWW-Authenticate header on invalid credentials (Req 35.4).
/// Uses constant-time comparison to prevent timing attacks.
async fn admin_auth_middleware(
    State(state): State<AppState>,
    request: Request,
    next: Next,
) -> Response {
    let config = state.config.read().await;
    let auth_config = &config.admin.auth;

    // Req 35.5, 35.7: when auth is disabled, allow unrestricted access
    if !auth_config.enabled {
        drop(config);
        return next.run(request).await;
    }

    // Resolve expected credentials from env vars (Req 35.6)
    let expected_username = match auth_config.username_env.as_ref() {
        Some(env_var) => match std::env::var(env_var) {
            Ok(val) => val,
            Err(_) => {
                tracing::error!("Admin auth enabled but env var '{}' not set", env_var);
                drop(config);
                return unauthorized_response();
            }
        },
        None => {
            tracing::error!("Admin auth enabled but no username_env configured");
            drop(config);
            return unauthorized_response();
        }
    };
    let expected_password = match auth_config.password_env.as_ref() {
        Some(env_var) => match std::env::var(env_var) {
            Ok(val) => val,
            Err(_) => {
                tracing::error!("Admin auth enabled but env var '{}' not set", env_var);
                drop(config);
                return unauthorized_response();
            }
        },
        None => {
            tracing::error!("Admin auth enabled but no password_env configured");
            drop(config);
            return unauthorized_response();
        }
    };

    drop(config);

    // Extract and validate Authorization header (Req 35.3)
    let auth_header = request
        .headers()
        .get(header::AUTHORIZATION)
        .and_then(|v| v.to_str().ok());

    let authorized = match auth_header {
        Some(value) if value.starts_with("Basic ") => {
            match base64::engine::general_purpose::STANDARD.decode(&value[6..]) {
                Ok(decoded) => match String::from_utf8(decoded) {
                    Ok(credentials) => {
                        if let Some((user, pass)) = credentials.split_once(':') {
                            // Constant-time comparison to prevent timing attacks
                            constant_time_eq(user.as_bytes(), expected_username.as_bytes())
                                && constant_time_eq(pass.as_bytes(), expected_password.as_bytes())
                        } else {
                            false
                        }
                    }
                    Err(_) => false,
                },
                Err(_) => false,
            }
        }
        _ => false,
    };

    if authorized {
        next.run(request).await
    } else {
        // Req 35.4: return 401 with generic message (don't reveal which part was wrong)
        unauthorized_response()
    }
}

/// Build a 401 Unauthorized response with WWW-Authenticate header.
fn unauthorized_response() -> Response {
    (
        StatusCode::UNAUTHORIZED,
        [
            (header::WWW_AUTHENTICATE, "Basic realm=\"OBEY-API Admin\""),
            (header::CONTENT_TYPE, "application/json"),
        ],
        serde_json::to_string(&json!({
            "error": {
                "message": "Authentication required",
                "type": "authentication_error"
            }
        }))
        .unwrap_or_default(),
    )
        .into_response()
}

/// Constant-time byte comparison to prevent timing attacks on credential checks.
fn constant_time_eq(a: &[u8], b: &[u8]) -> bool {
    if a.len() != b.len() {
        // Still do work proportional to max length to avoid length-based timing leak
        let max_len = a.len().max(b.len());
        let mut _acc: u8 = 1;
        for i in 0..max_len {
            let x = a.get(i).copied().unwrap_or(0);
            let y = b.get(i).copied().unwrap_or(0);
            _acc |= x ^ y;
        }
        return false;
    }
    let mut acc: u8 = 0;
    for (x, y) in a.iter().zip(b.iter()) {
        acc |= x ^ y;
    }
    acc == 0
}

/// Serve the main SPA page.
async fn index_handler(State(state): State<AppState>) -> impl IntoResponse {
    serve_index_html(&state)
}

/// Serve any static asset by path.
async fn static_handler(Path(path): Path<String>) -> impl IntoResponse {
    serve_embedded(&path)
}

/// Look up an embedded file and return it with the correct content-type.
fn serve_embedded(path: &str) -> Response {
    match AdminAssets::get(path) {
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

fn serve_index_html(state: &AppState) -> Response {
    match AdminAssets::get("index.html") {
        Some(content) => {
            let mut html = String::from_utf8_lossy(&content.data).into_owned();
            let config = state.config.try_read().expect("config lock poisoned");
            let dashboard_href = if config.dashboard.path == "/" {
                "/".to_string()
            } else {
                config.dashboard.path.trim_end_matches('/').to_string()
            };
            // Inject <base> so relative asset URLs (logo, favicon) resolve under the admin path
            let admin_base = format!("{}/", config.admin.path.trim_end_matches('/'));
            html = html.replace("<head>", &format!("<head><base href=\"{}\">", admin_base));
            html = html.replace(
                "href=\"/dashboard\" id=\"dashboard-link\"",
                &format!("href=\"{}\" id=\"dashboard-link\"", dashboard_href),
            );
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

/// Derive MIME type from file extension without adding a dependency.
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
        Some("woff2") => "font/woff2",
        Some("woff") => "font/woff",
        _ => "application/octet-stream",
    }
}

// ---------------------------------------------------------------------------
// Configuration API handlers (Req 13.11-13.15, 32.1-32.7)
// ---------------------------------------------------------------------------

/// Redact sensitive fields from a Config before returning it to the client.
///
/// For encrypted keys: Returns null in api_key_env (frontend shows "encrypted" status)
/// For env var references: Resolves the actual value from the environment and returns it
///                         in api_key_env so the frontend can display and re-save it
/// For plaintext keys: Returns the plaintext value directly
fn redact_config_for_response(config: &Config) -> serde_json::Value {
    let mut body = serde_json::to_value(config).unwrap_or(json!({"error": "serialization failed"}));

    if let Some(providers) = body.get_mut("providers").and_then(|v| v.as_array_mut()) {
        for (provider_json, provider) in providers.iter_mut().zip(config.providers.iter()) {
            if let Some(obj) = provider_json.as_object_mut() {
                // Handle api_key_env - resolve env var references to their actual values
                // so the frontend can display and re-save them (which triggers encryption)
                let resolved_key_value = if provider.has_encrypted_api_key() {
                    // Already encrypted - don't show anything in the textbox
                    // Frontend will show "encrypted" status
                    None
                } else if let Some(value) = provider.api_key_env.as_deref() {
                    if secrets::is_env_var_reference(value) {
                        // This is an env var name like OPENAI_API_KEY - resolve it
                        std::env::var(value).ok().or_else(|| {
                            // If env var not found, check if there's a resolved_api_key from startup
                            provider.resolved_api_key.clone()
                        })
                    } else if secrets::looks_like_plaintext_secret(value) {
                        // Already a plaintext secret - return as-is
                        Some(value.to_string())
                    } else {
                        // Neither - treat as potentially invalid, return as-is
                        Some(value.to_string())
                    }
                } else {
                    None
                };

                obj.insert(
                    "api_key_env".to_string(),
                    resolved_key_value
                        .map(serde_json::Value::String)
                        .unwrap_or(serde_json::Value::Null),
                );
                obj.remove("api_key_encrypted");
                obj.insert(
                    "api_key_status".to_string(),
                    serde_json::Value::String(provider_key_status(provider).to_string()),
                );
                obj.insert(
                    "api_key_configured".to_string(),
                    serde_json::Value::Bool(provider.has_api_key_configured()),
                );

                // Handle api_secret_env - resolve env var references to their actual values
                let resolved_secret_value = if provider.has_encrypted_api_secret() {
                    // Already encrypted - don't show anything in the textbox
                    None
                } else if let Some(value) = provider.api_secret_env.as_deref() {
                    if secrets::is_env_var_reference(value) {
                        // This is an env var name - resolve it
                        std::env::var(value)
                            .ok()
                            .or_else(|| provider.resolved_api_secret.clone())
                    } else if secrets::looks_like_plaintext_secret(value) {
                        // Already a plaintext secret - return as-is
                        Some(value.to_string())
                    } else {
                        // Neither - treat as potentially invalid, return as-is
                        Some(value.to_string())
                    }
                } else {
                    None
                };

                obj.insert(
                    "api_secret_env".to_string(),
                    resolved_secret_value
                        .map(serde_json::Value::String)
                        .unwrap_or(serde_json::Value::Null),
                );
                obj.remove("api_secret_encrypted");
                obj.insert(
                    "api_secret_status".to_string(),
                    serde_json::Value::String(provider_secret_status(provider).to_string()),
                );
                obj.insert(
                    "api_secret_configured".to_string(),
                    serde_json::Value::Bool(provider.has_api_secret_configured()),
                );
            }
        }
    }

    body
}

fn provider_key_status(provider: &crate::config::Provider) -> &'static str {
    if provider.has_plaintext_api_key_input() {
        "plaintext"
    } else if provider.has_encrypted_api_key() {
        "encrypted"
    } else if provider
        .api_key_env
        .as_deref()
        .is_some_and(secrets::is_env_var_reference)
    {
        "environment"
    } else {
        "missing"
    }
}

fn provider_secret_status(provider: &crate::config::Provider) -> &'static str {
    if provider.has_plaintext_api_secret_input() {
        "plaintext"
    } else if provider.has_encrypted_api_secret() {
        "encrypted"
    } else if provider
        .api_secret_env
        .as_deref()
        .is_some_and(secrets::is_env_var_reference)
    {
        "environment"
    } else {
        "missing"
    }
}

fn hydrate_provider_runtime_secrets(config: &mut Config) {
    for provider in &mut config.providers {
        // Handle api_key
        provider.resolved_api_key = None;

        if let Some(encrypted) = provider.api_key_encrypted.as_deref() {
            match secrets::decrypt_provider_secret(encrypted) {
                Ok(value) => provider.resolved_api_key = Some(value),
                Err(error) => {
                    tracing::warn!(provider = %provider.name, error = %error, "Failed to decrypt provider api_key in admin flow")
                }
            }
        } else if let Some(value) = provider.api_key_env.as_deref() {
            if secrets::looks_like_plaintext_secret(value) {
                provider.resolved_api_key = Some(value.to_string());
            }
        }

        // Handle api_secret
        provider.resolved_api_secret = None;

        if let Some(encrypted) = provider.api_secret_encrypted.as_deref() {
            match secrets::decrypt_provider_secret(encrypted) {
                Ok(value) => provider.resolved_api_secret = Some(value),
                Err(error) => {
                    tracing::warn!(provider = %provider.name, error = %error, "Failed to decrypt provider api_secret in admin flow")
                }
            }
        } else if let Some(value) = provider.api_secret_env.as_deref() {
            if secrets::looks_like_plaintext_secret(value) {
                provider.resolved_api_secret = Some(value.to_string());
            }
        }
    }
}

fn normalize_config_for_storage(
    mut config: Config,
    existing_config: Option<&Config>,
) -> Result<Config, String> {
    for provider in &mut config.providers {
        let existing_provider = existing_config.and_then(|cfg| {
            cfg.providers
                .iter()
                .find(|candidate| candidate.name == provider.name)
        });

        // Handle api_key_env encryption
        let key_input = provider
            .api_key_env
            .as_deref()
            .map(str::trim)
            .filter(|value| !value.is_empty());

        if let Some(value) = key_input {
            if secrets::looks_like_plaintext_secret(value) {
                let encrypted = secrets::encrypt_provider_secret(value).map_err(|error| {
                    format!(
                        "Failed to encrypt API key for provider '{}': {}",
                        provider.name, error
                    )
                })?;
                provider.api_key_encrypted = Some(encrypted);
                provider.resolved_api_key = Some(value.to_string());
                provider.api_key_env = None;
            } else {
                provider.api_key_encrypted = None;
                provider.resolved_api_key = None;

                if !secrets::is_env_var_reference(value) {
                    provider.api_key_env = None;
                }
            }
        } else if provider.has_encrypted_api_key() {
            if provider.resolved_api_key.is_none() {
                provider.resolved_api_key = provider
                    .api_key_encrypted
                    .as_deref()
                    .and_then(|encrypted| secrets::decrypt_provider_secret(encrypted).ok());
            }

            if provider
                .api_key_env
                .as_deref()
                .is_some_and(|value| value.trim().is_empty())
            {
                provider.api_key_env = None;
            }
        } else if let Some(existing) = existing_provider {
            if existing.has_encrypted_api_key() {
                provider.api_key_encrypted = existing.api_key_encrypted.clone();
                provider.resolved_api_key = existing.resolved_api_key.clone();
                provider.api_key_env = None;
            }
        }

        // Handle api_secret_env encryption
        let secret_input = provider
            .api_secret_env
            .as_deref()
            .map(str::trim)
            .filter(|value| !value.is_empty());

        if let Some(value) = secret_input {
            if secrets::looks_like_plaintext_secret(value) {
                let encrypted = secrets::encrypt_provider_secret(value).map_err(|error| {
                    format!(
                        "Failed to encrypt API secret for provider '{}': {}",
                        provider.name, error
                    )
                })?;
                provider.api_secret_encrypted = Some(encrypted);
                provider.resolved_api_secret = Some(value.to_string());
                provider.api_secret_env = None;
            } else {
                provider.api_secret_encrypted = None;
                provider.resolved_api_secret = None;

                if !secrets::is_env_var_reference(value) {
                    provider.api_secret_env = None;
                }
            }
        } else if provider.has_encrypted_api_secret() {
            if provider.resolved_api_secret.is_none() {
                provider.resolved_api_secret = provider
                    .api_secret_encrypted
                    .as_deref()
                    .and_then(|encrypted| secrets::decrypt_provider_secret(encrypted).ok());
            }

            if provider
                .api_secret_env
                .as_deref()
                .is_some_and(|value| value.trim().is_empty())
            {
                provider.api_secret_env = None;
            }
        } else if let Some(existing) = existing_provider {
            if existing.has_encrypted_api_secret() {
                provider.api_secret_encrypted = existing.api_secret_encrypted.clone();
                provider.resolved_api_secret = existing.resolved_api_secret.clone();
                provider.api_secret_env = None;
            }
        }
    }

    Ok(config)
}

/// GET /admin/config — return current configuration (Req 13.12)
///
/// Returns the live config as JSON with API key env var names only (never
/// resolved values).
async fn get_config(State(state): State<AppState>) -> Response {
    let config = state.config.read().await;
    let body = redact_config_for_response(&config);
    (StatusCode::OK, Json(body)).into_response()
}

/// PUT /admin/config — validate, persist to YAML, and apply (Req 13.11)
///
/// Accepts a full Config JSON body. On success the new config is written to
/// the YAML file and swapped into the live state.
async fn update_config(State(state): State<AppState>, Json(new_config): Json<Config>) -> Response {
    let current_config = state.config.read().await.clone();

    let normalized_config = match normalize_config_for_storage(new_config, Some(&current_config)) {
        Ok(config) => config,
        Err(error) => {
            return (
                StatusCode::BAD_REQUEST,
                Json(json!({
                    "error": {
                        "message": error,
                        "type": "encryption_error"
                    }
                })),
            )
                .into_response();
        }
    };

    // Validate first
    if let Err(errors) = normalized_config.validate() {
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

    let config_path = state.config_path.as_ref();
    if let Err(e) = save_config(config_path, &normalized_config) {
        tracing::error!("Failed to write config file: {}", e);
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

    // Apply to live state
    apply_runtime_config_update(&state, normalized_config).await;

    tracing::info!(
        "Configuration updated and saved to {}",
        config_path.display()
    );

    (
        StatusCode::OK,
        Json(json!({
            "status": "ok",
            "message": "Configuration updated successfully"
        })),
    )
        .into_response()
}

/// POST /admin/config/validate — dry-run validation (Req 13.11, 13.16)
///
/// Accepts a Config JSON body, validates it, and returns success or a list of
/// validation errors without applying any changes.
async fn validate_config(Json(config): Json<Config>) -> Response {
    match config.validate() {
        Ok(()) => (
            StatusCode::OK,
            Json(json!({
                "valid": true,
                "message": "Configuration is valid"
            })),
        )
            .into_response(),
        Err(errors) => {
            let msgs: Vec<String> = errors.iter().map(|e| e.to_string()).collect();
            (
                StatusCode::BAD_REQUEST,
                Json(json!({
                    "valid": false,
                    "errors": msgs
                })),
            )
                .into_response()
        }
    }
}

/// POST /admin/config/reload — hot-reload from disk (Req 26.1-26.7)
///
/// Delegates to the same logic as the existing reload handler in
/// gateway/handlers.rs.
async fn reload_config(State(state): State<AppState>) -> Response {
    let config_path = state.config_path.as_ref();

    let new_config = match load_and_validate_config(config_path) {
        Ok(cfg) => cfg,
        Err(e) => {
            tracing::warn!("Config reload validation failed: {}", e);
            return (
                StatusCode::BAD_REQUEST,
                Json(json!({
                    "error": {
                        "message": format!("Configuration validation failed: {}", e),
                        "type": "configuration_error"
                    }
                })),
            )
                .into_response();
        }
    };

    apply_runtime_config_update(&state, new_config).await;

    tracing::info!("Configuration reloaded from {}", config_path.display());

    (
        StatusCode::OK,
        Json(json!({
            "status": "ok",
            "message": "Configuration reloaded successfully"
        })),
    )
        .into_response()
}

/// GET /admin/config/export — download current config as YAML (Req 32.1, 32.2, 32.7)
///
/// Returns the current configuration serialized as YAML with a
/// Content-Disposition header to trigger a browser download.
async fn export_config(State(state): State<AppState>) -> Response {
    let config = state.config.read().await;

    let yaml = match serde_yaml::to_string(&*config) {
        Ok(y) => y,
        Err(_) => {
            return (
                StatusCode::INTERNAL_SERVER_ERROR,
                Json(json!({
                    "error": {
                        "message": "Failed to serialize configuration",
                        "type": "internal_error"
                    }
                })),
            )
                .into_response();
        }
    };

    (
        StatusCode::OK,
        [
            (header::CONTENT_TYPE, "application/x-yaml"),
            (
                header::CONTENT_DISPOSITION,
                "attachment; filename=\"config.yaml\"",
            ),
        ],
        yaml,
    )
        .into_response()
}

/// POST /admin/config/import — upload YAML, validate, return parsed config
/// (Req 32.3-32.7)
///
/// Accepts a raw YAML body, deserializes and validates it, then returns the
/// parsed config as JSON. Does NOT apply the config — the caller can review
/// and then PUT /admin/config to apply.
async fn import_config(body: String) -> Response {
    if body.trim().is_empty() {
        return (
            StatusCode::BAD_REQUEST,
            Json(json!({
                "error": {
                    "message": "Import body is empty",
                    "type": "parse_error"
                }
            })),
        )
            .into_response();
    }

    // Parse YAML
    let mut config: Config = match serde_yaml::from_str(&body) {
        Ok(c) => c,
        Err(e) => {
            return (
                StatusCode::BAD_REQUEST,
                Json(json!({
                    "error": {
                        "message": format!("Invalid YAML: {}", e),
                        "type": "parse_error"
                    }
                })),
            )
                .into_response();
        }
    };

    hydrate_provider_runtime_secrets(&mut config);

    // Validate
    if let Err(errors) = config.validate() {
        let msgs: Vec<String> = errors.iter().map(|e| e.to_string()).collect();
        return (
            StatusCode::BAD_REQUEST,
            Json(json!({
                "valid": false,
                "errors": msgs
            })),
        )
            .into_response();
    }

    // Return parsed config as JSON for review (Req 32.5)
    let body = redact_config_for_response(&config);
    (
        StatusCode::OK,
        Json(json!({
            "valid": true,
            "config": body
        })),
    )
        .into_response()
}

// ---------------------------------------------------------------------------
// Test Connection endpoint (Req 11.3, 11.6, 12.2)
// ---------------------------------------------------------------------------

/// POST /admin/test-connection
///
/// Body: `{"provider_name": "..."}`
/// Tests end-to-end connectivity for a Codex provider by issuing a minimal
/// chat completion request. Response bodies never include access tokens,
/// refresh tokens, or chatgpt-account-id (Req 11.6, 12.2).
async fn test_connection(
    State(state): State<AppState>,
    Json(body): Json<serde_json::Value>,
) -> Response {
    // Extract provider_name from body
    let provider_name = match body.get("provider_name").and_then(|v| v.as_str()) {
        Some(name) => name.to_string(),
        None => {
            return (
                StatusCode::BAD_REQUEST,
                Json(json!({"status": "error", "message": "missing provider_name"})),
            )
                .into_response();
        }
    };

    // Look up provider in config
    let config = state.config.read().await;
    let provider_cfg = match config.providers.iter().find(|p| p.name == provider_name) {
        Some(p) => p.clone(),
        None => {
            return (
                StatusCode::BAD_REQUEST,
                Json(json!({"status": "not_found"})),
            )
                .into_response();
        }
    };
    drop(config);

    // Check if this is a Codex provider (oauth + openai)
    let is_codex = provider_cfg.auth_method.as_deref() == Some("oauth")
        && provider_cfg.provider_type == "openai";

    if !is_codex {
        return (
            StatusCode::BAD_REQUEST,
            Json(json!({"status": "error", "message": "not a Codex provider"})),
        )
            .into_response();
    }

    // Check OAuth session
    let oauth = match &state.oauth_manager {
        Some(m) => m.clone(),
        None => {
            return (
                StatusCode::BAD_REQUEST,
                Json(
                    json!({"status": "unauthenticated", "message": "OpenAI OAuth login required"}),
                ),
            )
                .into_response();
        }
    };

    // Check if we have a valid token (Req 11.3)
    let _access_token = match oauth.get_access_token().await {
        Some(t) => t,
        None => {
            return (
                StatusCode::BAD_REQUEST,
                Json(
                    json!({"status": "unauthenticated", "message": "OpenAI OAuth login required"}),
                ),
            )
                .into_response();
        }
    };

    // Build a minimal test request
    let test_model = provider_cfg
        .codex_model_override
        .clone()
        .unwrap_or_else(|| "gpt-4.1-nano".to_string());
    let test_request = crate::models::openai::OpenAIRequest {
        model: test_model.clone(),
        messages: vec![crate::models::openai::Message {
            role: "user".to_string(),
            content: serde_json::json!("Say hi"),
            extra: serde_json::Map::new(),
        }],
        stream: false,
        temperature: None,
        max_tokens: Some(16),
        extra: serde_json::Map::new(),
    };

    // Get instructions store
    let instructions = match state.router.instructions_store() {
        Some(s) => s,
        None => {
            return (
                StatusCode::INTERNAL_SERVER_ERROR,
                Json(json!({"status": "error", "message": "instructions store not initialized"})),
            )
                .into_response();
        }
    };

    // Build CodexProviderClient and dispatch
    let http = reqwest::Client::new();
    let codex_client = crate::codex::client::CodexProviderClient::new(
        provider_name.clone(),
        oauth,
        instructions,
        http,
        state.metrics.clone(),
        state.oauth_usage_tracker.clone(),
        provider_cfg.codex_base_url_override.clone(),
        provider_cfg.codex_model_override.clone(),
        provider_cfg.instructions_override.clone(),
        vec![],
        vec![],
    );

    let start = std::time::Instant::now();
    use crate::providers::ProviderClient;
    match codex_client.chat_completion(test_request).await {
        Ok(_response) => {
            let latency_ms = start.elapsed().as_millis() as u64;
            (
                StatusCode::OK,
                Json(json!({
                    "status": "ok",
                    "model": test_model,
                    "latency_ms": latency_ms,
                })),
            )
                .into_response()
        }
        Err(crate::error::GatewayError::Provider {
            status_code: Some(code),
            ..
        }) if code == 401 || code == 403 => (
            StatusCode::BAD_GATEWAY,
            Json(json!({
                "status": "auth_error",
                "upstream_status": code,
            })),
        )
            .into_response(),
        Err(crate::error::GatewayError::Provider {
            status_code: Some(code),
            ..
        }) => (
            StatusCode::BAD_GATEWAY,
            Json(json!({
                "status": "upstream_error",
                "upstream_status": code,
            })),
        )
            .into_response(),
        Err(_) => (
            StatusCode::BAD_GATEWAY,
            Json(json!({
                "status": "upstream_error",
                "upstream_status": 502,
            })),
        )
            .into_response(),
    }
}

// ---------------------------------------------------------------------------
// OAuth admin endpoints (Req 1.1, 1.6, 7.1, 7.2, 7.3, 8.5)
// ---------------------------------------------------------------------------

/// POST /admin/oauth/openai/login — Initiate OAuth login flow (Req 1.1, 1.6)
///
/// On success (browser opened): `{ "status": "initiated", "message": "Browser opened for authentication" }`
/// On browser-open failure: `{ "status": "manual_required", "auth_url": "…" }`
/// If OAuth is not configured: 500 with "OAuth not configured"
async fn oauth_login(State(state): State<AppState>) -> Response {
    let manager = match &state.oauth_manager {
        Some(m) => m.clone(),
        None => {
            return (
                StatusCode::INTERNAL_SERVER_ERROR,
                Json(json!({
                    "error": {
                        "message": "OAuth not configured",
                        "type": "configuration_error"
                    }
                })),
            )
                .into_response();
        }
    };

    match manager.initiate_login().await {
        Ok(outcome) => {
            use crate::oauth::flow::InitiationOutcome;
            match outcome {
                InitiationOutcome::BrowserOpened { .. } => (
                    StatusCode::OK,
                    Json(json!({
                        "status": "initiated",
                        "message": "Browser opened for authentication"
                    })),
                )
                    .into_response(),
                InitiationOutcome::ManualNavigationRequired { auth_url } => (
                    StatusCode::OK,
                    Json(json!({
                        "status": "manual_required",
                        "auth_url": auth_url
                    })),
                )
                    .into_response(),
            }
        }
        Err(e) => (
            StatusCode::INTERNAL_SERVER_ERROR,
            Json(json!({
                "error": {
                    "message": format!("Failed to initiate OAuth login: {}", e),
                    "type": "oauth_error"
                }
            })),
        )
            .into_response(),
    }
}

/// POST /admin/oauth/openai/complete — Complete login with a manually-pasted
/// authorization code (headless / remote Docker case).
///
/// Body: `{ "code": "<authorization code or full redirect URL>" }`
///
/// When the gateway runs in a container on a different host than the user's
/// browser, OpenAI's redirect to `http://localhost:1455/...` cannot reach the
/// container. The user copies the resulting URL (or just the `code`) and
/// submits it here. Requires a prior `POST /admin/oauth/openai/login` so the
/// PKCE verifier is available.
async fn oauth_complete(
    State(state): State<AppState>,
    Json(body): Json<serde_json::Value>,
) -> Response {
    let manager = match &state.oauth_manager {
        Some(m) => m.clone(),
        None => {
            return (
                StatusCode::INTERNAL_SERVER_ERROR,
                Json(json!({
                    "error": {
                        "message": "OAuth not configured",
                        "type": "configuration_error"
                    }
                })),
            )
                .into_response();
        }
    };

    let code = body
        .get("code")
        .and_then(|v| v.as_str())
        .unwrap_or("")
        .trim()
        .to_string();

    if code.is_empty() {
        return (
            StatusCode::BAD_REQUEST,
            Json(json!({
                "error": {
                    "message": "Missing 'code' — paste the authorization code or the full redirect URL",
                    "type": "invalid_request"
                }
            })),
        )
            .into_response();
    }

    match manager.complete_login_with_code(&code).await {
        Ok(()) => (
            StatusCode::OK,
            Json(json!({
                "status": "authenticated",
                "message": "Sign-in completed"
            })),
        )
            .into_response(),
        Err(e) => (
            StatusCode::BAD_REQUEST,
            Json(json!({
                "error": {
                    "message": format!("Failed to complete OAuth login: {}", e),
                    "type": "oauth_error"
                }
            })),
        )
            .into_response(),
    }
}

/// GET /admin/oauth/openai/status — Return OAuth session status (Req 7.1, 7.2, 8.4)
///
/// Returns `{ "state": "...", "expires_at": ..., "scopes": "..." }`.
/// NEVER includes access_token or refresh_token values.
async fn oauth_status(State(state): State<AppState>) -> Response {
    let manager = match &state.oauth_manager {
        Some(m) => m.clone(),
        None => {
            return (
                StatusCode::OK,
                Json(json!({
                    "state": "unauthenticated",
                    "expires_at": null,
                    "scopes": null
                })),
            )
                .into_response();
        }
    };

    let session = manager.session_state().await;

    let (state_str, expires_at, scopes) = match &session {
        crate::oauth::flow::OAuthSessionState::Unauthenticated => ("unauthenticated", None, None),
        crate::oauth::flow::OAuthSessionState::Authenticated { expires_at, scopes } => {
            ("authenticated", Some(*expires_at), Some(scopes.clone()))
        }
        crate::oauth::flow::OAuthSessionState::Expired => ("expired", None, None),
        crate::oauth::flow::OAuthSessionState::Refreshing => ("refreshing", None, None),
    };

    (
        StatusCode::OK,
        Json(json!({
            "state": state_str,
            "expires_at": expires_at,
            "scopes": scopes
        })),
    )
        .into_response()
}

/// GET /admin/oauth/openai/usage — Return current rate-limit usage snapshot.
///
/// Returns the captured `x-ratelimit-*` header data so the admin UI can
/// display the 5h / weekly usage windows for browser-login accounts.
async fn oauth_usage(State(state): State<AppState>) -> Response {
    let snapshot = state.oauth_usage_tracker.snapshot().await;
    (StatusCode::OK, Json(json!(snapshot))).into_response()
}

/// POST /admin/oauth/openai/logout — Clear stored tokens (Req 7.3)
///
/// Deletes tokens via `OAuthTokenStore::delete`, sets session to
/// `Unauthenticated`, and returns `{ "status": "logged_out" }`.
async fn oauth_logout(State(state): State<AppState>) -> Response {
    let manager = match &state.oauth_manager {
        Some(m) => m.clone(),
        None => {
            // No OAuth configured — already effectively logged out.
            return (StatusCode::OK, Json(json!({ "status": "logged_out" }))).into_response();
        }
    };

    if let Err(e) = manager.logout().await {
        return (
            StatusCode::INTERNAL_SERVER_ERROR,
            Json(json!({
                "error": {
                    "message": format!("Failed to logout: {}", e),
                    "type": "oauth_error"
                }
            })),
        )
            .into_response();
    }

    (StatusCode::OK, Json(json!({ "status": "logged_out" }))).into_response()
}

/// Query parameters for the provider models proxy endpoint.
#[derive(serde::Deserialize)]
struct ProxyModelsParams {
    /// The provider's base URL (e.g. `https://api.openai.com/v1` or a Mantle endpoint).
    base_url: String,
    /// Optional Bearer token / API key for the upstream provider.
    #[serde(default)]
    api_key: String,
    /// Optional provider name — when api_key is empty, the backend will look up
    /// the resolved (decrypted) key from the live config for this provider.
    #[serde(default)]
    provider_name: Option<String>,
    /// Optional provider type. NVIDIA NIM uses its built-in fallback when live
    /// discovery fails or returns an empty catalog.
    #[serde(default)]
    provider_type: Option<String>,
}

/// Proxy endpoint that fetches `/v1/models` from an upstream provider server-side,
/// avoiding browser CORS restrictions.
///
/// `GET /admin/providers/models?base_url=...&api_key=...&provider_name=...`
async fn proxy_provider_models(
    State(state): State<AppState>,
    Query(params): Query<ProxyModelsParams>,
) -> Response {
    let base = params.base_url.trim_end_matches('/');
    let models_url = if base.ends_with("/v1") || base.ends_with("/v1/") {
        format!("{}/models", base.trim_end_matches('/'))
    } else {
        format!("{}/v1/models", base)
    };

    // Resolve the effective API key: use the provided key, or fall back to the
    // decrypted key stored in the live config for the named provider. For
    // OAuth-based providers (auth_method: oauth), return the static Codex model
    // hints directly — the Codex OAuth token is not scoped for api.openai.com/v1/models.
    let effective_api_key = if !params.api_key.is_empty() {
        params.api_key.clone()
    } else if let Some(ref name) = params.provider_name {
        let config = state.config.read().await;
        let provider = config.providers.iter().find(|p| &p.name == name);
        match provider {
            Some(p) if p.auth_method.as_deref() == Some("oauth") => {
                // OAuth/Codex provider — use dynamic model discovery with
                // caching and staleness fallback.
                drop(config);
                let (mut models_response, is_stale) =
                    state.codex_models_discovery.get_models().await;
                if is_stale {
                    if let Some(obj) = models_response.as_object_mut() {
                        obj.insert("stale".to_string(), serde_json::json!(true));
                        obj.insert(
                            "stale_reason".to_string(),
                            serde_json::json!(
                                "Could not fetch latest models from OpenAI. List may be outdated."
                            ),
                        );
                    }
                }
                return (StatusCode::OK, Json(models_response)).into_response();
            }
            Some(p) => p.resolve_api_key().unwrap_or_default(),
            None => String::new(),
        }
    } else {
        String::new()
    };

    let client = reqwest::Client::builder()
        .timeout(std::time::Duration::from_secs(10))
        .build()
        .unwrap_or_default();

    let mut req = client.get(&models_url);
    if !effective_api_key.is_empty() {
        req = req.header("Authorization", format!("Bearer {}", effective_api_key));
    }

    match req.send().await {
        Ok(resp) => {
            let status = resp.status().as_u16();
            match resp.text().await {
                Ok(body) => {
                    if params.provider_type.as_deref() == Some("nvidia_nim") {
                        let parsed = serde_json::from_str::<serde_json::Value>(&body).ok();
                        let has_models = parsed
                            .as_ref()
                            .and_then(|value| value.get("data"))
                            .and_then(|value| value.as_array())
                            .is_some_and(|models| !models.is_empty());
                        if status < 400 && has_models {
                            return (
                                StatusCode::OK,
                                [(header::CONTENT_TYPE, "application/json")],
                                body,
                            )
                                .into_response();
                        }

                        tracing::warn!(
                            upstream_status = status,
                            "NVIDIA NIM admin model discovery failed or returned no models; using built-in fallback catalog"
                        );
                        return (
                            StatusCode::OK,
                            Json(json!({
                                "object": "list",
                                "data": crate::providers::nvidia_nim::fallback_models(),
                                "fallback": true
                            })),
                        )
                            .into_response();
                    }

                    let axum_status =
                        StatusCode::from_u16(status).unwrap_or(StatusCode::BAD_GATEWAY);
                    (
                        axum_status,
                        [(header::CONTENT_TYPE, "application/json")],
                        body,
                    )
                        .into_response()
                }
                Err(error) if params.provider_type.as_deref() == Some("nvidia_nim") => {
                    tracing::warn!(
                        error = %error,
                        "Failed to read NVIDIA NIM model response; using built-in fallback catalog"
                    );
                    (
                        StatusCode::OK,
                        Json(json!({
                            "object": "list",
                            "data": crate::providers::nvidia_nim::fallback_models(),
                            "fallback": true
                        })),
                    )
                        .into_response()
                }
                Err(error) => (
                    StatusCode::BAD_GATEWAY,
                    Json(
                        json!({ "error": format!("Failed to read upstream response: {}", error) }),
                    ),
                )
                    .into_response(),
            }
        }
        Err(error) if params.provider_type.as_deref() == Some("nvidia_nim") => {
            tracing::warn!(
                error = %error,
                "Failed to reach NVIDIA NIM model endpoint; using built-in fallback catalog"
            );
            (
                StatusCode::OK,
                Json(json!({
                    "object": "list",
                    "data": crate::providers::nvidia_nim::fallback_models(),
                    "fallback": true
                })),
            )
                .into_response()
        }
        Err(error) => (
            StatusCode::BAD_GATEWAY,
            Json(json!({ "error": format!("Failed to reach provider: {}", error) })),
        )
            .into_response(),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::config::*;
    use axum::body::to_bytes;
    use proptest::prelude::*;
    use proptest::strategy::ValueTree;
    use std::collections::HashMap;

    // --- Proptest strategies for Config generation ---

    fn arb_server_config() -> impl Strategy<Value = ServerConfig> {
        (
            prop::sample::select(vec![
                "0.0.0.0".to_string(),
                "127.0.0.1".to_string(),
                "localhost".to_string(),
            ]),
            1u16..=65535u16,
            1u64..=300u64,
            1u64..=100u64,
        )
            .prop_map(|(host, port, timeout, max_size)| ServerConfig {
                host,
                port,
                request_timeout_seconds: timeout,
                max_request_size_mb: max_size,
            })
    }

    fn arb_provider() -> impl Strategy<Value = Provider> {
        (
            "[a-z][a-z0-9_]{1,10}",
            prop::sample::select(vec![
                "openai".to_string(),
                "ollama".to_string(),
                "bedrock".to_string(),
                "groq".to_string(),
            ]),
            prop::option::of(Just("https://api.example.com/v1".to_string())),
            prop::option::of("[A-Z_][A-Z0-9_]{2,10}"),
            prop::option::of("[A-Z_][A-Z0-9_]{2,10}"),
            prop::option::of(Just("us-east-1".to_string())),
            1u64..=120u64,
            1u32..=500u32,
            0u32..=1000u32,
        )
            .prop_map(
                |(
                    name,
                    ptype,
                    base_url,
                    api_key_env,
                    api_secret_env,
                    region,
                    timeout,
                    max_conn,
                    rate_limit,
                )| Provider {
                    name,
                    provider_type: ptype,
                    base_url,
                    api_key_env,
                    api_key_encrypted: None,
                    api_secret_env,
                    api_secret_encrypted: None,
                    auth_method: None,
                    resolved_api_key: None,
                    resolved_api_secret: None,
                    region,
                    timeout_seconds: timeout,
                    ttfb_timeout_seconds: None,
                    total_timeout_seconds: None,
                    max_connections: max_conn,
                    rate_limit_per_minute: rate_limit,
                    custom_headers: HashMap::new(),
                    connection_pool: ProviderConnectionPoolConfig::default(),
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
                },
            )
    }

    fn arb_provider_model(provider_name: String) -> impl Strategy<Value = ProviderModel> {
        (
            "[a-z][a-z0-9-]{2,15}",
            0.0f64..100.0f64,
            0.0f64..100.0f64,
            1u32..=1000u32,
        )
            .prop_map(move |(model, cost_in, cost_out, priority)| ProviderModel {
                provider: provider_name.clone(),
                model,
                cost_per_million_input_tokens: cost_in,
                cost_per_million_output_tokens: cost_out,
                priority,
            })
    }

    fn arb_model_group(provider_name: String) -> impl Strategy<Value = ModelGroup> {
        (
            "[a-z][a-z0-9-]{2,10}",
            prop::bool::ANY,
            prop::collection::vec(arb_provider_model(provider_name), 1..=3),
        )
            .prop_map(|(name, vf, models)| ModelGroup {
                name,
                version_fallback_enabled: vf,
                compression: None,
                models,
            })
    }

    fn arb_config() -> impl Strategy<Value = Config> {
        arb_server_config().prop_flat_map(|server| {
            arb_provider().prop_flat_map(move |provider| {
                let pname = provider.name.clone();
                let server = server.clone();
                let provider = provider.clone();
                arb_model_group(pname).prop_map(move |group| Config {
                    server: server.clone(),
                    tls: None,
                    admin: AdminConfig::default(),
                    dashboard: DashboardConfig::default(),
                    cors: CorsConfig::default(),
                    providers: vec![provider.clone()],
                    model_groups: vec![group],
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
                })
            })
        })
    }

    // Feature: ai-gateway, Property 35: Configuration Export-Import Round-Trip
    // **Validates: Requirements 32.1, 32.2, 32.4, 32.5, 32.7**
    proptest! {
        #![proptest_config(ProptestConfig::with_cases(64))]

        #[test]
        fn prop_config_export_import_round_trip(config in arb_config()) {
            // Export: serialize to YAML (Req 32.1, 32.2)
            let yaml = serde_yaml::to_string(&config)
                .expect("Config should serialize to YAML");

            // Import: deserialize from YAML (Req 32.4, 32.5)
            let imported: Config = serde_yaml::from_str(&yaml)
                .expect("Exported YAML should deserialize back");

            // Round-trip equivalence
            prop_assert_eq!(&config.server, &imported.server, "server mismatch");
            prop_assert_eq!(config.providers.len(), imported.providers.len(), "providers count mismatch");
            prop_assert_eq!(&config.providers[0].name, &imported.providers[0].name);
            prop_assert_eq!(&config.providers[0].api_key_env, &imported.providers[0].api_key_env,
                "Env var references must be preserved (Req 32.7)");
            prop_assert_eq!(&config.providers[0].api_secret_env, &imported.providers[0].api_secret_env,
                "Env var references must be preserved (Req 32.7)");
            prop_assert_eq!(config.model_groups.len(), imported.model_groups.len(), "model_groups count mismatch");
            prop_assert_eq!(&config.model_groups[0].name, &imported.model_groups[0].name);
            prop_assert_eq!(config.model_groups[0].version_fallback_enabled, imported.model_groups[0].version_fallback_enabled);
            prop_assert_eq!(config.model_groups[0].models.len(), imported.model_groups[0].models.len());
            prop_assert_eq!(&config.circuit_breaker, &imported.circuit_breaker);
            prop_assert_eq!(&config.retry, &imported.retry);
            prop_assert_eq!(&config.admin, &imported.admin);
            prop_assert_eq!(&config.cors, &imported.cors);
        }
    }

    #[test]
    fn test_mime_from_path() {
        assert_eq!(mime_from_path("index.html"), "text/html; charset=utf-8");
        assert_eq!(mime_from_path("style.css"), "text/css; charset=utf-8");
        assert_eq!(
            mime_from_path("app.js"),
            "application/javascript; charset=utf-8"
        );
        assert_eq!(mime_from_path("data.json"), "application/json");
        assert_eq!(mime_from_path("unknown.xyz"), "application/octet-stream");
    }

    #[test]
    fn test_index_html_embedded() {
        // Verify the index.html is actually embedded
        let asset = AdminAssets::get("index.html");
        assert!(asset.is_some(), "index.html should be embedded");
        let data = asset.unwrap();
        let html = std::str::from_utf8(&data.data).unwrap();
        assert!(
            html.contains("OBEY-API Admin"),
            "Should contain admin panel title"
        );
    }

    #[test]
    fn config_save_ui_times_out_and_reports_server_errors() {
        let asset = AdminAssets::get("index.html").expect("index embedded");
        let html = std::str::from_utf8(&asset.data).unwrap();

        assert!(html.contains("new AbortController()"));
        assert!(html.contains("request timed out after 15 seconds"));
        assert!(html.contains("data.error.details"));
        assert!(html.contains("config=cfg;setStatus('Configuration saved')"));
    }

    #[test]
    fn token_compression_admin_ui_round_trips_global_and_overrides() {
        let asset = AdminAssets::get("index.html").expect("index embedded");
        let html = std::str::from_utf8(&asset.data).unwrap();

        for marker in [
            "tab-compression-config",
            "page-compression",
            "compression-enabled",
            "compression-default-level",
            "compression-threshold",
            "compression-budget-stacked",
            "compression-rule-code-blocks",
            "compression-perplexity-enabled",
            "compression-precompressed-list",
            "compression-pipelines-list",
            "prov-compression-override",
            "grp-compression-override",
        ] {
            assert!(
                html.contains(marker),
                "missing compression UI marker {marker}"
            );
        }

        assert!(html.contains("function populateCompression(value)"));
        assert!(html.contains("function collectCompression()"));
        assert!(html.contains("function collectCompressionOverride(card, prefix)"));
        assert!(html.contains("'page-compression': validateCompressionPage"));
        assert!(html.contains("cfg.compression=collectCompression()"));
        assert!(html.contains("collectCompressionOverride(card,'prov')"));
        assert!(html.contains("collectCompressionOverride(card,'grp')"));
    }

    #[test]
    fn loop_detection_ui_has_valid_presets_and_put_save_path() {
        let asset = AdminAssets::get("index.html").expect("index embedded");
        let html = std::str::from_utf8(&asset.data).unwrap();
        for id in [
            "page-loop-detection",
            "loop-enabled",
            "loop-session-timeout",
            "loop-max-sessions",
            "loop-history-depth",
            "loop-threshold-warn",
            "loop-count-warn",
            "loop-weight-sum",
            "loop-throttle-delay",
            "loop-injection-strategy",
            "loop-break-template",
        ] {
            assert!(html.contains(id), "missing loop UI field {id}");
        }
        assert!(html.contains("balanced:{content_similarity:0.25"));
        assert!(html.contains("'tool-heavy':{content_similarity:0.15"));
        assert!(html.contains("'cost-sensitive':{content_similarity:0.18"));
        assert!(html.contains("fetch(apiUrl('config'),{method:'PUT'"));
        assert!(html.contains("cfg.loop_detection=collectLoopDetection()"));
        assert!(html.contains("Reset all Loop Detection settings to defaults?"));
    }

    #[test]
    fn virtual_key_ui_collects_and_clears_loop_overrides() {
        let asset = AdminAssets::get("index.html").expect("index embedded");
        let html = std::str::from_utf8(&asset.data).unwrap();
        for id in [
            "vk-f-loop-enabled",
            "vk-loop-thresholds",
            "vk-loop-counts",
            "vk-loop-weights",
            "vk-loop-delay",
            "vk-loop-strategy",
            "vk-loop-template",
            "vk-loop-clear",
        ] {
            assert!(html.contains(id), "missing VK loop control {id}");
        }
        assert!(html.contains("body.loop_detection = vkCollectLoopOverrides()"));
        assert!(html.contains(
            "body.loop_detection = f.loopEnabled.checked ? vkCollectLoopOverrides() : null"
        ));
        assert!(html.contains("vkPopulateLoopOverrides(d.loop_detection || null)"));
    }

    #[test]
    fn test_serve_embedded_not_found() {
        let resp = serve_embedded("nonexistent.file");
        assert_eq!(resp.status(), StatusCode::NOT_FOUND);
    }

    #[test]
    fn test_serve_embedded_index() {
        let resp = serve_embedded("index.html");
        assert_eq!(resp.status(), StatusCode::OK);
    }

    #[tokio::test]
    async fn test_admin_index_links_to_canonical_dashboard_path() {
        let config = test_config_with_auth(false);
        let server = crate::gateway::GatewayServer::new(config, None)
            .await
            .unwrap();
        let resp = serve_index_html(&server.state);
        assert_eq!(resp.status(), StatusCode::OK);

        let body = to_bytes(resp.into_body(), 512 * 1024).await.unwrap();
        let html = std::str::from_utf8(&body).unwrap();
        assert!(html.contains("href=\"/dashboard\" id=\"dashboard-link\""));
        assert!(!html.contains("href=\"/dashboard/\" id=\"dashboard-link\""));
    }

    #[tokio::test]
    async fn test_import_config_empty_body_rejected() {
        let response = import_config(String::new()).await;
        assert_eq!(response.status(), StatusCode::BAD_REQUEST);

        let body = to_bytes(response.into_body(), 64 * 1024).await.unwrap();
        let json: serde_json::Value = serde_json::from_slice(&body).unwrap();
        assert_eq!(json["error"]["type"], "parse_error");
    }

    #[tokio::test]
    async fn test_import_config_valid_yaml_returns_config() {
        let yaml = r#"
server:
  host: "127.0.0.1"
  port: 8080
  request_timeout_seconds: 30
  max_request_size_mb: 10
providers:
  - name: "openai"
    type: "openai"
    base_url: "https://api.openai.com/v1"
    timeout_seconds: 30
model_groups:
  - name: "default"
    version_fallback_enabled: false
    models:
      - provider: "openai"
        model: "gpt-4"
        priority: 100
retry:
  max_retries_per_provider: 1
  backoff_sequence_seconds: [1, 2, 4]
"#;

        let response = import_config(yaml.to_string()).await;
        assert_eq!(response.status(), StatusCode::OK);

        let body = to_bytes(response.into_body(), 64 * 1024).await.unwrap();
        let json: serde_json::Value = serde_json::from_slice(&body).unwrap();
        assert_eq!(json["valid"], true);
        assert_eq!(json["config"]["providers"][0]["name"], "openai");
        assert_eq!(json["config"]["retry"]["max_retries_per_provider"], 1);
    }

    #[test]
    fn test_redact_config_for_response_hides_encrypted_provider_key() {
        let mut config = arb_config()
            .new_tree(&mut proptest::test_runner::TestRunner::default())
            .unwrap()
            .current();
        config.providers[0].api_key_env = None;
        config.providers[0].api_key_encrypted = Some("enc-v1:test:test".to_string());
        config.providers[0].resolved_api_key = Some("sk-secret-value".to_string());

        let redacted = redact_config_for_response(&config);
        assert_eq!(redacted["providers"][0]["api_key_status"], "encrypted");
        assert_eq!(redacted["providers"][0]["api_key_configured"], true);
        assert!(redacted["providers"][0].get("api_key_encrypted").is_none());
        assert!(redacted["providers"][0]["api_key_env"].is_null());
    }

    #[test]
    fn test_normalize_config_for_storage_encrypts_plaintext_key() {
        let mut config = arb_config()
            .new_tree(&mut proptest::test_runner::TestRunner::default())
            .unwrap()
            .current();
        config.providers[0].name = "plain-provider".to_string();
        config.providers[0].api_key_env = Some("sk-test-secret-12345678901234567890".to_string());
        config.providers[0].api_key_encrypted = None;
        config.providers[0].resolved_api_key = None;

        let normalized = normalize_config_for_storage(config, None).unwrap();
        assert!(normalized.providers[0].api_key_env.is_none());
        assert!(normalized.providers[0]
            .api_key_encrypted
            .as_deref()
            .is_some_and(secrets::is_encrypted_secret));
        assert_eq!(
            normalized.providers[0].resolved_api_key.as_deref(),
            Some("sk-test-secret-12345678901234567890")
        );
    }

    #[test]
    fn test_normalize_config_for_storage_preserves_existing_encrypted_key_when_blank() {
        let mut existing = arb_config()
            .new_tree(&mut proptest::test_runner::TestRunner::default())
            .unwrap()
            .current();
        existing.providers[0].name = "persisted-provider".to_string();
        existing.providers[0].api_key_encrypted = Some("enc-v1:nonce:data".to_string());
        existing.providers[0].resolved_api_key = Some("sk-existing-secret".to_string());
        existing.providers[0].api_key_env = None;

        let mut incoming = existing.clone();
        incoming.providers[0].api_key_encrypted = None;
        incoming.providers[0].resolved_api_key = None;
        incoming.providers[0].api_key_env = None;

        let normalized = normalize_config_for_storage(incoming, Some(&existing)).unwrap();
        assert_eq!(
            normalized.providers[0].api_key_encrypted.as_deref(),
            Some("enc-v1:nonce:data")
        );
        assert_eq!(
            normalized.providers[0].resolved_api_key.as_deref(),
            Some("sk-existing-secret")
        );
    }

    // --- Admin authentication tests (Req 35.1-35.7) ---

    #[test]
    fn test_constant_time_eq_equal() {
        assert!(constant_time_eq(b"admin", b"admin"));
        assert!(constant_time_eq(b"", b""));
    }

    #[test]
    fn test_constant_time_eq_not_equal() {
        assert!(!constant_time_eq(b"admin", b"Admin"));
        assert!(!constant_time_eq(b"admin", b"admi"));
        assert!(!constant_time_eq(b"a", b"ab"));
    }

    #[test]
    fn test_unauthorized_response_format() {
        let resp = unauthorized_response();
        assert_eq!(resp.status(), StatusCode::UNAUTHORIZED);
        assert_eq!(
            resp.headers()
                .get("www-authenticate")
                .unwrap()
                .to_str()
                .unwrap(),
            "Basic realm=\"OBEY-API Admin\""
        );
    }

    fn test_config_with_auth_env(enabled: bool, user_env: &str, pass_env: &str) -> Config {
        Config {
            server: ServerConfig {
                host: "127.0.0.1".to_string(),
                port: 0,
                request_timeout_seconds: 30,
                max_request_size_mb: 10,
            },
            tls: None,
            admin: AdminConfig {
                enabled: true,
                path: "/admin".to_string(),
                auth: AdminAuthConfig {
                    enabled,
                    username_env: Some(user_env.to_string()),
                    password_env: Some(pass_env.to_string()),
                },
            },
            dashboard: DashboardConfig::default(),
            cors: CorsConfig::default(),
            providers: vec![Provider {
                name: "test".to_string(),
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
                connection_pool: ProviderConnectionPoolConfig::default(),
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
                name: "test-group".to_string(),
                version_fallback_enabled: false,
                compression: None,
                models: vec![ProviderModel {
                    provider: "test".to_string(),
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

    fn test_config_with_auth(enabled: bool) -> Config {
        test_config_with_auth_env(enabled, "TEST_ADMIN_USER", "TEST_ADMIN_PASS")
    }

    fn basic_auth_header(user: &str, pass: &str) -> String {
        use base64::Engine;
        let encoded =
            base64::engine::general_purpose::STANDARD.encode(format!("{}:{}", user, pass));
        format!("Basic {}", encoded)
    }

    #[tokio::test]
    async fn loop_detection_admin_requires_auth_when_enabled() {
        let user_env = "LOOP_ADMIN_TEST_USER";
        let pass_env = "LOOP_ADMIN_TEST_PASS";
        std::env::set_var(user_env, "admin");
        std::env::set_var(pass_env, "secret");
        let cfg = test_config_with_auth_env(true, user_env, pass_env);
        let server = crate::gateway::GatewayServer::new(cfg, None).await.unwrap();
        let app = server.build_router();

        let unauthorized = axum::http::Request::get("/admin/loop-detection/sessions")
            .body(axum::body::Body::empty())
            .unwrap();
        let response = tower::ServiceExt::oneshot(app.clone(), unauthorized)
            .await
            .unwrap();
        assert_eq!(response.status(), StatusCode::UNAUTHORIZED);

        let authorized = axum::http::Request::get("/admin/loop-detection/sessions")
            .header("Authorization", basic_auth_header("admin", "secret"))
            .body(axum::body::Body::empty())
            .unwrap();
        let response = tower::ServiceExt::oneshot(app, authorized).await.unwrap();
        assert_eq!(response.status(), StatusCode::OK);
        std::env::remove_var(user_env);
        std::env::remove_var(pass_env);
    }

    #[tokio::test]
    async fn test_auth_disabled_allows_all_requests() {
        // Req 35.5, 35.7: auth disabled → unrestricted access
        let cfg = test_config_with_auth(false);
        let server = crate::gateway::GatewayServer::new(cfg, None).await.unwrap();
        let app = server.build_router();

        let req = axum::http::Request::builder()
            .method("GET")
            .uri("/admin/config")
            .body(axum::body::Body::empty())
            .unwrap();

        let resp = tower::ServiceExt::oneshot(app, req).await.unwrap();
        assert_eq!(resp.status(), StatusCode::OK);
    }

    #[tokio::test]
    async fn test_auth_enabled_rejects_no_credentials() {
        // Req 35.1, 35.4: auth enabled, no credentials → 401
        let user_env = "AUTH_TEST_NOCRED_USER";
        let pass_env = "AUTH_TEST_NOCRED_PASS";
        std::env::set_var(user_env, "admin");
        std::env::set_var(pass_env, "secret");

        let cfg = test_config_with_auth_env(true, user_env, pass_env);
        let server = crate::gateway::GatewayServer::new(cfg, None).await.unwrap();
        let app = server.build_router();

        let req = axum::http::Request::builder()
            .method("GET")
            .uri("/admin/config")
            .body(axum::body::Body::empty())
            .unwrap();

        let resp = tower::ServiceExt::oneshot(app, req).await.unwrap();
        assert_eq!(resp.status(), StatusCode::UNAUTHORIZED);
        assert!(resp.headers().get("www-authenticate").is_some());

        std::env::remove_var(user_env);
        std::env::remove_var(pass_env);
    }

    #[tokio::test]
    async fn test_auth_enabled_accepts_valid_credentials() {
        // Req 35.1, 35.3: auth enabled, valid Basic credentials → pass through
        let user_env = "AUTH_TEST_VALID_USER";
        let pass_env = "AUTH_TEST_VALID_PASS";
        std::env::set_var(user_env, "admin");
        std::env::set_var(pass_env, "secret");

        let cfg = test_config_with_auth_env(true, user_env, pass_env);
        let server = crate::gateway::GatewayServer::new(cfg, None).await.unwrap();
        let app = server.build_router();

        let req = axum::http::Request::builder()
            .method("GET")
            .uri("/admin/config")
            .header("Authorization", basic_auth_header("admin", "secret"))
            .body(axum::body::Body::empty())
            .unwrap();

        let resp = tower::ServiceExt::oneshot(app, req).await.unwrap();
        assert_eq!(resp.status(), StatusCode::OK);

        std::env::remove_var(user_env);
        std::env::remove_var(pass_env);
    }

    #[tokio::test]
    async fn test_auth_enabled_rejects_wrong_password() {
        // Req 35.4: invalid credentials → 401
        let user_env = "AUTH_TEST_WRONGPW_USER";
        let pass_env = "AUTH_TEST_WRONGPW_PASS";
        std::env::set_var(user_env, "admin");
        std::env::set_var(pass_env, "secret");

        let cfg = test_config_with_auth_env(true, user_env, pass_env);
        let server = crate::gateway::GatewayServer::new(cfg, None).await.unwrap();
        let app = server.build_router();

        let req = axum::http::Request::builder()
            .method("GET")
            .uri("/admin/config")
            .header("Authorization", basic_auth_header("admin", "wrong"))
            .body(axum::body::Body::empty())
            .unwrap();

        let resp = tower::ServiceExt::oneshot(app, req).await.unwrap();
        assert_eq!(resp.status(), StatusCode::UNAUTHORIZED);

        std::env::remove_var(user_env);
        std::env::remove_var(pass_env);
    }

    // Feature: ai-gateway, Property 36: Admin Authentication Enforcement
    // **Validates: Requirements 35.1, 35.3, 35.4**
    //
    // For any admin panel endpoint when admin authentication is enabled,
    // requests without valid credentials shall receive HTTP 401 status.
    // When auth is disabled, all requests pass through without credentials.
    proptest! {
        #![proptest_config(ProptestConfig::with_cases(32))]

        #[test]
        fn prop_admin_auth_enforcement(
            username in "[a-zA-Z0-9_]{1,20}",
            password in "[a-zA-Z0-9!@#$%^&*]{1,20}",
            bad_user in "[a-zA-Z0-9_]{1,20}",
            bad_pass in "[a-zA-Z0-9!@#$%^&*]{1,20}",
        ) {
            // Use a dedicated tokio runtime for each proptest iteration
            let rt = tokio::runtime::Runtime::new().unwrap();
            rt.block_on(async {
                // Use unique env var names to avoid collisions with parallel tests
                let id = format!("{:x}", std::time::SystemTime::now()
                    .duration_since(std::time::UNIX_EPOCH).unwrap().as_nanos());
                let user_env = format!("PROP36_USER_{}", id);
                let pass_env = format!("PROP36_PASS_{}", id);

                std::env::set_var(&user_env, &username);
                std::env::set_var(&pass_env, &password);

                // --- Auth ENABLED ---
                let cfg_enabled = test_config_with_auth_env(true, &user_env, &pass_env);
                let server = crate::gateway::GatewayServer::new(cfg_enabled, None).await.unwrap();
                let app = server.build_router();

                // 1) No credentials → 401 (Req 35.1, 35.4)
                let req = axum::http::Request::builder()
                    .method("GET")
                    .uri("/admin/config")
                    .body(axum::body::Body::empty())
                    .unwrap();
                let resp = tower::ServiceExt::oneshot(app, req).await.unwrap();
                assert_eq!(resp.status(), StatusCode::UNAUTHORIZED,
                    "Auth enabled + no creds must be 401");

                // 2) Valid credentials → 200 (Req 35.1, 35.3)
                let server2 = crate::gateway::GatewayServer::new(
                    test_config_with_auth_env(true, &user_env, &pass_env), None
                ).await.unwrap();
                let app2 = server2.build_router();
                let req = axum::http::Request::builder()
                    .method("GET")
                    .uri("/admin/config")
                    .header("Authorization", basic_auth_header(&username, &password))
                    .body(axum::body::Body::empty())
                    .unwrap();
                let resp = tower::ServiceExt::oneshot(app2, req).await.unwrap();
                assert_eq!(resp.status(), StatusCode::OK,
                    "Auth enabled + valid creds must be 200");

                // 3) Invalid credentials → 401 (Req 35.4)
                // Ensure bad creds differ from good ones
                let wrong_user = if bad_user == username {
                    format!("x{}", bad_user)
                } else {
                    bad_user.clone()
                };
                let wrong_pass = if bad_pass == password {
                    format!("x{}", bad_pass)
                } else {
                    bad_pass.clone()
                };

                let server3 = crate::gateway::GatewayServer::new(
                    test_config_with_auth_env(true, &user_env, &pass_env), None
                ).await.unwrap();
                let app3 = server3.build_router();
                let req = axum::http::Request::builder()
                    .method("GET")
                    .uri("/admin/config")
                    .header("Authorization", basic_auth_header(&wrong_user, &wrong_pass))
                    .body(axum::body::Body::empty())
                    .unwrap();
                let resp = tower::ServiceExt::oneshot(app3, req).await.unwrap();
                assert_eq!(resp.status(), StatusCode::UNAUTHORIZED,
                    "Auth enabled + wrong creds must be 401");

                // --- Auth DISABLED → pass through (Req 35.5, 35.7) ---
                let cfg_disabled = test_config_with_auth_env(false, &user_env, &pass_env);
                let server4 = crate::gateway::GatewayServer::new(cfg_disabled, None).await.unwrap();
                let app4 = server4.build_router();
                let req = axum::http::Request::builder()
                    .method("GET")
                    .uri("/admin/config")
                    .body(axum::body::Body::empty())
                    .unwrap();
                let resp = tower::ServiceExt::oneshot(app4, req).await.unwrap();
                assert_eq!(resp.status(), StatusCode::OK,
                    "Auth disabled + no creds must be 200");

                // Cleanup env vars
                std::env::remove_var(&user_env);
                std::env::remove_var(&pass_env);
            });
        }
    }

    #[tokio::test]
    async fn test_auth_enabled_rejects_wrong_username() {
        // Req 35.4: invalid credentials → 401
        let user_env = "AUTH_TEST_WRONGUN_USER";
        let pass_env = "AUTH_TEST_WRONGUN_PASS";
        std::env::set_var(user_env, "admin");
        std::env::set_var(pass_env, "secret");

        let cfg = test_config_with_auth_env(true, user_env, pass_env);
        let server = crate::gateway::GatewayServer::new(cfg, None).await.unwrap();
        let app = server.build_router();

        let req = axum::http::Request::builder()
            .method("GET")
            .uri("/admin/config")
            .header("Authorization", basic_auth_header("hacker", "secret"))
            .body(axum::body::Body::empty())
            .unwrap();

        let resp = tower::ServiceExt::oneshot(app, req).await.unwrap();
        assert_eq!(resp.status(), StatusCode::UNAUTHORIZED);

        std::env::remove_var(user_env);
        std::env::remove_var(pass_env);
    }

    // -----------------------------------------------------------------------
    // Property 7 — No Tokens in Status Response (Req 7.2, 8.4)
    // -----------------------------------------------------------------------
    //
    // For all `OAuthSessionState` values, the serialized status JSON response
    // (as built by the `oauth_status` handler) does NOT contain the
    // access_token or refresh_token string values from the token store.
    //
    // **Validates: Requirements 7.2, 8.4**

    /// Replicate the status-response serialization logic from `oauth_status`.
    fn build_status_json(session: &crate::oauth::flow::OAuthSessionState) -> String {
        let (state_str, expires_at, scopes) = match session {
            crate::oauth::flow::OAuthSessionState::Unauthenticated => {
                ("unauthenticated", None, None)
            }
            crate::oauth::flow::OAuthSessionState::Authenticated { expires_at, scopes } => {
                ("authenticated", Some(*expires_at), Some(scopes.clone()))
            }
            crate::oauth::flow::OAuthSessionState::Expired => ("expired", None, None),
            crate::oauth::flow::OAuthSessionState::Refreshing => ("refreshing", None, None),
        };

        serde_json::to_string(&json!({
            "state": state_str,
            "expires_at": expires_at,
            "scopes": scopes
        }))
        .expect("status response should serialize")
    }

    /// Strategy that generates arbitrary `OAuthSessionState` values.
    fn arb_oauth_session_state() -> impl Strategy<Value = crate::oauth::flow::OAuthSessionState> {
        prop_oneof![
            Just(crate::oauth::flow::OAuthSessionState::Unauthenticated),
            (any::<u64>(), "[a-zA-Z0-9_ ]{0,50}").prop_map(|(exp, scopes)| {
                crate::oauth::flow::OAuthSessionState::Authenticated {
                    expires_at: exp,
                    scopes,
                }
            }),
            Just(crate::oauth::flow::OAuthSessionState::Expired),
            Just(crate::oauth::flow::OAuthSessionState::Refreshing),
        ]
    }

    proptest! {
        #![proptest_config(ProptestConfig::with_cases(256))]

        #[test]
        fn prop_status_response_never_contains_tokens(
            session in arb_oauth_session_state(),
            access_token in ".{1,64}",
            refresh_token in ".{1,64}",
        ) {
            let json_output = build_status_json(&session);

            // The status response must never contain the raw token values.
            // Only check non-trivial tokens (single-char tokens like "e" would
            // trivially match inside JSON keys like "state" or "expires_at").
            if access_token.len() > 4 {
                prop_assert!(
                    !json_output.contains(&access_token),
                    "Status response must not contain access_token value.\n\
                     access_token: {:?}\njson: {:?}",
                    access_token,
                    json_output
                );
            }
            if refresh_token.len() > 4 {
                prop_assert!(
                    !json_output.contains(&refresh_token),
                    "Status response must not contain refresh_token value.\n\
                     refresh_token: {:?}\njson: {:?}",
                    refresh_token,
                    json_output
                );
            }
        }
    }

    // -----------------------------------------------------------------------
    // Task 14.7 — Integration tests for admin OAuth endpoints (Req 7.1, 7.3)
    // -----------------------------------------------------------------------

    /// Helper: build a GatewayServer with OAuth enabled (using a temp dir for
    /// the token store so tests don't touch real storage).
    async fn build_oauth_test_app() -> Router {
        let cfg = test_config_with_auth(false);
        let server = crate::gateway::GatewayServer::new(cfg, None).await.unwrap();
        server.build_router()
    }

    /// POST /admin/oauth/openai/login — in a headless test environment the
    /// browser cannot open, so we expect the `manual_required` variant with
    /// an `auth_url` field (Req 1.6).
    #[tokio::test]
    async fn test_oauth_login_returns_manual_required_when_headless() {
        let app = build_oauth_test_app().await;

        let req = axum::http::Request::builder()
            .method("POST")
            .uri("/admin/oauth/openai/login")
            .body(axum::body::Body::empty())
            .unwrap();

        let resp = tower::ServiceExt::oneshot(app, req).await.unwrap();
        let status = resp.status();

        let body = to_bytes(resp.into_body(), 64 * 1024).await.unwrap();
        let json: serde_json::Value = serde_json::from_slice(&body).unwrap();

        // The endpoint should return 200 with either "initiated" or
        // "manual_required". In a headless CI/test environment without a
        // display server, `open::that()` fails and we get `manual_required`.
        assert_eq!(status, StatusCode::OK);
        let resp_status = json["status"].as_str().unwrap();
        assert!(
            resp_status == "manual_required" || resp_status == "initiated",
            "Expected 'manual_required' or 'initiated', got: {}",
            resp_status
        );

        // When manual_required, auth_url must be present and point to OpenAI.
        if resp_status == "manual_required" {
            let auth_url = json["auth_url"].as_str().unwrap();
            assert!(
                auth_url.starts_with("https://auth.openai.com/"),
                "auth_url should point to OpenAI, got: {}",
                auth_url
            );
        }
    }

    /// GET /admin/oauth/openai/status — verify response shape has `state`,
    /// `expires_at`, `scopes` fields. When no login has occurred, state
    /// should be `unauthenticated` (Req 7.1).
    #[tokio::test]
    async fn test_oauth_status_returns_expected_shape() {
        let app = build_oauth_test_app().await;

        let req = axum::http::Request::builder()
            .method("GET")
            .uri("/admin/oauth/openai/status")
            .body(axum::body::Body::empty())
            .unwrap();

        let resp = tower::ServiceExt::oneshot(app, req).await.unwrap();
        assert_eq!(resp.status(), StatusCode::OK);

        let body = to_bytes(resp.into_body(), 64 * 1024).await.unwrap();
        let json: serde_json::Value = serde_json::from_slice(&body).unwrap();

        // Must have all three fields
        assert!(
            json.get("state").is_some(),
            "response must have 'state' field"
        );
        assert!(
            json.get("expires_at").is_some(),
            "response must have 'expires_at' field"
        );
        assert!(
            json.get("scopes").is_some(),
            "response must have 'scopes' field"
        );

        // Fresh server with no prior login → unauthenticated
        assert_eq!(json["state"], "unauthenticated");
        assert!(json["expires_at"].is_null());
        assert!(json["scopes"].is_null());
    }

    /// POST /admin/oauth/openai/logout — verify returns `{ "status": "logged_out" }`;
    /// subsequent GET to status returns `unauthenticated` (Req 7.3).
    #[tokio::test]
    async fn test_oauth_logout_clears_tokens_and_status_becomes_unauthenticated() {
        let cfg = test_config_with_auth(false);
        let server = crate::gateway::GatewayServer::new(cfg, None).await.unwrap();

        // --- Logout ---
        let app = server.build_router();
        let req = axum::http::Request::builder()
            .method("POST")
            .uri("/admin/oauth/openai/logout")
            .body(axum::body::Body::empty())
            .unwrap();

        let resp = tower::ServiceExt::oneshot(app, req).await.unwrap();
        assert_eq!(resp.status(), StatusCode::OK);

        let body = to_bytes(resp.into_body(), 64 * 1024).await.unwrap();
        let json: serde_json::Value = serde_json::from_slice(&body).unwrap();
        assert_eq!(json["status"], "logged_out");

        // --- Subsequent status check must be unauthenticated ---
        let app = server.build_router();
        let req = axum::http::Request::builder()
            .method("GET")
            .uri("/admin/oauth/openai/status")
            .body(axum::body::Body::empty())
            .unwrap();

        let resp = tower::ServiceExt::oneshot(app, req).await.unwrap();
        assert_eq!(resp.status(), StatusCode::OK);

        let body = to_bytes(resp.into_body(), 64 * 1024).await.unwrap();
        let json: serde_json::Value = serde_json::from_slice(&body).unwrap();
        assert_eq!(json["state"], "unauthenticated");
    }

    // -----------------------------------------------------------------------
    // Task 16.3 — Admin-auth protection test (Req 8.5)
    // -----------------------------------------------------------------------
    //
    // With admin auth enabled and no credentials, all three OAuth endpoints
    // return 401; with valid credentials they return 2xx.

    #[tokio::test]
    async fn test_oauth_endpoints_require_admin_auth_when_enabled() {
        let user_env = "AUTH_TEST_OAUTH_PROT_USER";
        let pass_env = "AUTH_TEST_OAUTH_PROT_PASS";
        let username = "admin";
        let password = "supersecret";
        std::env::set_var(user_env, username);
        std::env::set_var(pass_env, password);

        let cfg = test_config_with_auth_env(true, user_env, pass_env);

        // --- Without credentials: all three OAuth endpoints must return 401 ---

        let endpoints: &[(&str, &str)] = &[
            ("POST", "/admin/oauth/openai/login"),
            ("GET", "/admin/oauth/openai/status"),
            ("GET", "/admin/oauth/openai/usage"),
            ("POST", "/admin/oauth/openai/logout"),
        ];

        for (method, uri) in endpoints {
            let server = crate::gateway::GatewayServer::new(cfg.clone(), None)
                .await
                .unwrap();
            let app = server.build_router();

            let req = axum::http::Request::builder()
                .method(*method)
                .uri(*uri)
                .body(axum::body::Body::empty())
                .unwrap();

            let resp = tower::ServiceExt::oneshot(app, req).await.unwrap();
            assert_eq!(
                resp.status(),
                StatusCode::UNAUTHORIZED,
                "{} {} without credentials must return 401",
                method,
                uri
            );
        }

        // --- With valid credentials: all three OAuth endpoints must return 2xx ---

        for (method, uri) in endpoints {
            let server = crate::gateway::GatewayServer::new(cfg.clone(), None)
                .await
                .unwrap();
            let app = server.build_router();

            let req = axum::http::Request::builder()
                .method(*method)
                .uri(*uri)
                .header("Authorization", basic_auth_header(username, password))
                .body(axum::body::Body::empty())
                .unwrap();

            let resp = tower::ServiceExt::oneshot(app, req).await.unwrap();
            assert!(
                resp.status().is_success(),
                "{} {} with valid credentials must return 2xx, got {}",
                method,
                uri,
                resp.status()
            );
        }

        // Cleanup
        std::env::remove_var(user_env);
        std::env::remove_var(pass_env);
    }
}
