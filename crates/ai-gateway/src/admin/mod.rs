use axum::{
    extract::{Json, Path, Query, RawQuery, Request, State},
    http::{header, StatusCode},
    middleware::{self, Next},
    response::{IntoResponse, Response},
    routing::{get, post},
    Router,
};
use base64::Engine;
use rust_embed::Embed;
use serde::Deserialize;
use serde_json::json;
use std::sync::{Arc, OnceLock, RwLock as StdRwLock};

use crate::config::{load_and_validate_config, save_config, AdminSecurityMode, Config};
use crate::gateway::apply_runtime_config_update;
use crate::gateway::AppState;
use crate::models::openai::{Message, OpenAIRequest};
use crate::secrets;
use crate::smart_routing::ab_test::{ABTestAnalysisConfig, ABTestManager};
use crate::smart_routing::config::{ABTestConfig, SmartRoutingConfig};
use crate::smart_routing::{PinnedRoutingContext, SmartRouter, SmartRoutingInput};

pub mod auth_policy;
pub mod tool_compression;

/// Embedded admin panel static assets (Req 13.17, 13.18, 1.3).
#[derive(Embed)]
#[folder = "src/admin/static/"]
struct AdminAssets;

/// Build the admin panel router.
/// All routes are relative ÃƒÆ’Ã‚Â¢ÃƒÂ¢Ã¢â‚¬Å¡Ã‚Â¬ÃƒÂ¢Ã¢â€šÂ¬Ã‚Â the caller nests them under the configured admin path.
///
/// Config API endpoints (Req 13.11-13.15, 32.1-32.7):
///   GET    /config          ÃƒÆ’Ã‚Â¢ÃƒÂ¢Ã¢â‚¬Å¡Ã‚Â¬ÃƒÂ¢Ã¢â€šÂ¬Ã‚Â return current configuration
///   PUT    /config          ÃƒÆ’Ã‚Â¢ÃƒÂ¢Ã¢â‚¬Å¡Ã‚Â¬ÃƒÂ¢Ã¢â€šÂ¬Ã‚Â update configuration (validate, write YAML, apply)
///   POST   /config/validate ÃƒÆ’Ã‚Â¢ÃƒÂ¢Ã¢â‚¬Å¡Ã‚Â¬ÃƒÂ¢Ã¢â€šÂ¬Ã‚Â validate configuration without applying
///   POST   /config/reload   ÃƒÆ’Ã‚Â¢ÃƒÂ¢Ã¢â‚¬Å¡Ã‚Â¬ÃƒÂ¢Ã¢â€šÂ¬Ã‚Â hot-reload from disk
///   GET    /config/export   ÃƒÆ’Ã‚Â¢ÃƒÂ¢Ã¢â‚¬Å¡Ã‚Â¬ÃƒÂ¢Ã¢â€šÂ¬Ã‚Â download YAML
///   POST   /config/import   ÃƒÆ’Ã‚Â¢ÃƒÂ¢Ã¢â‚¬Å¡Ã‚Â¬ÃƒÂ¢Ã¢â€šÂ¬Ã‚Â upload YAML, validate, return parsed config
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
    let memory_api = crate::memory::admin::routes(state.memory_system.clone());

    let onnx_api = Router::new()
        .route("/status", get(onnx_status))
        .route("/install", post(onnx_install));

    let smart_routing_api = Router::new()
        .route(
            "/config",
            get(get_smart_routing_config).put(update_smart_routing_config),
        )
        .route("/simulate", post(simulate_smart_routing))
        .route("/optimizer-state", get(get_smart_routing_optimizer_state))
        .route("/budget", get(get_smart_routing_budget))
        .route("/model/status", get(get_smart_routing_model_status))
        .route("/model/download", post(start_smart_routing_model_download))
        .route("/model/train", post(start_smart_routing_training))
        .route(
            "/model/train/status",
            get(get_smart_routing_training_status),
        )
        .route(
            "/ab-test",
            get(get_smart_routing_ab_results)
                .put(update_smart_routing_ab_test)
                .delete(stop_smart_routing_ab_test),
        )
        .route("/ab-test/results", get(get_smart_routing_ab_results));

    Router::new()
        .nest("/config", config_api)
        .nest("/onnx", onnx_api)
        .nest("/loop-detection", loop_detection_api)
        .nest("/memory", memory_api)
        .nest("/tool-compression", tool_compression::routes())
        .nest("/smart-routing", smart_routing_api)
        .nest_service("/keys", virtual_keys_api)
        .route("/providers/models", post(proxy_provider_models))
        .route("/providers/{name}/budget", get(get_provider_budget))
        .route(
            "/providers/{name}/budget/reset",
            post(reset_provider_budget),
        )
        .route("/test-connection", post(test_connection))
        .route("/oauth/openai/login", post(oauth_login))
        .route("/oauth/openai/complete", post(oauth_complete))
        .route("/oauth/openai/status", get(oauth_status))
        .route("/oauth/openai/usage", get(oauth_usage))
        .route("/oauth/openai/logout", post(oauth_logout))
        .route("/metrics/reset-active", post(reset_active_requests))
        .route("/circuit-breaker/states", get(get_circuit_breaker_states))
        .route("/circuit-breaker/reset", post(reset_circuit_breakers))
        .route("/security-status", get(security_status_handler))
        .route("/", get(index_handler))
        .route("/{*path}", get(static_handler))
        .route_layer(middleware::from_fn_with_state(state, admin_auth_middleware))
}

/// HTTP Basic Authentication middleware for admin panel (Req 35.1-35.7;
/// spec black-october-security-hardening, Req 1).
///
/// The security posture is resolved per request from the current Config via
/// `auth_policy::resolve_admin_security_mode` (pure function of config).
/// `Authenticated` → existing Basic-auth path (constant-time comparison
/// retained); `LoopbackOpen`/`OpenWarnedLegacy`/`OpenOptIn` → pass-through.
/// Returns 401 with WWW-Authenticate header on invalid credentials (Req 35.4).
pub(crate) async fn admin_auth_middleware(
    State(state): State<AppState>,
    request: Request,
    next: Next,
) -> Response {
    let config = state.config.read().await;

    // Spec Req 1: resolve the security mode; only `Authenticated` enforces
    // credentials. Every other mode passes through (bind-address-based
    // decision — `server.host`, no ConnectInfo plumbing).
    let mode = auth_policy::resolve_admin_security_mode(&config);
    if mode != AdminSecurityMode::Authenticated {
        drop(config);
        return next.run(request).await;
    }

    // Resolve expected credentials from env vars (Req 35.6)
    let auth_config = &config.admin.auth;
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
/// Key material is never returned (Req 3.2): encrypted keys, plaintext keys,
/// and resolved values are all stripped. Env var references (e.g.
/// `OPENAI_API_KEY`) are kept verbatim so the frontend can display and
/// re-save the reference — the name is not a secret. Status booleans
/// (`api_key_status`, `api_key_configured`) are the only key information
/// exposed besides the env var name.
fn redact_config_for_response(config: &Config) -> serde_json::Value {
    let mut body = serde_json::to_value(config).unwrap_or(json!({"error": "serialization failed"}));

    if let Some(providers) = body.get_mut("providers").and_then(|v| v.as_array_mut()) {
        for (provider_json, provider) in providers.iter_mut().zip(config.providers.iter()) {
            if let Some(obj) = provider_json.as_object_mut() {
                // Env var references are safe to echo back (they are just names);
                // encrypted or plaintext key material is replaced with null.
                let visible_key_env = provider
                    .api_key_env
                    .as_deref()
                    .filter(|_| !provider.has_encrypted_api_key())
                    .and_then(|value| {
                        (!secrets::looks_like_plaintext_secret(value)).then(|| value.to_string())
                    });
                obj.insert(
                    "api_key_env".to_string(),
                    visible_key_env
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

                // Custom header values may carry credentials (Authorization/Bearer
                // values, plaintext secrets, ${ENV} references). Redact the VALUE,
                // keep the header name so the frontend can still display the pair.
                if let Some(headers) = obj
                    .get_mut("custom_headers")
                    .and_then(|v| v.as_object_mut())
                {
                    for (name, value) in headers.iter_mut() {
                        let Some(text) = value.as_str() else { continue };
                        let lower_name = name.to_ascii_lowercase();
                        let lower_value = text.to_ascii_lowercase();
                        let credential_bearing = lower_name.contains("authorization")
                            || lower_name.contains("x-api-key")
                            || lower_value.contains("authorization")
                            || lower_value.contains("bearer")
                            || lower_value.contains("x-api-key")
                            || secrets::looks_like_plaintext_secret(text)
                            || text.contains("${");
                        if credential_bearing {
                            *value = serde_json::Value::String("[configured]".to_string());
                        }
                    }
                }

                let visible_secret_env = provider
                    .api_secret_env
                    .as_deref()
                    .filter(|_| !provider.has_encrypted_api_secret())
                    .and_then(|value| {
                        (!secrets::looks_like_plaintext_secret(value)).then(|| value.to_string())
                    });
                obj.insert(
                    "api_secret_env".to_string(),
                    visible_secret_env
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

    if let Some(smart_routing) = body
        .get_mut("smart_routing")
        .and_then(|v| v.as_object_mut())
    {
        redact_jev_credentials(smart_routing, config.smart_routing.jev.as_ref());
        redact_laya_credentials(smart_routing, config.smart_routing.laya.as_ref());
    }

    body
}

/// Strip Jev classifier credentials from a serialized `smart_routing` object,
/// leaving only the `api_key_configured` status and, when the configured
/// value is an env var reference, the env var name itself.
fn redact_jev_credentials(
    smart_routing: &mut serde_json::Map<String, serde_json::Value>,
    jev_config: Option<&crate::smart_routing::config::JevConfig>,
) {
    let Some(jev) = smart_routing.get_mut("jev").and_then(|v| v.as_object_mut()) else {
        return;
    };
    let configured = jev_config.is_some_and(|jev| jev.has_api_key_configured());
    let visible_env = jev_config
        .and_then(|jev| jev.api_key_env.as_deref())
        .and_then(|value| {
            (!secrets::looks_like_plaintext_secret(value)).then(|| value.to_string())
        });
    jev.remove("api_key");
    jev.remove("api_key_encrypted");
    jev.insert(
        "api_key_env".to_string(),
        visible_env
            .map(serde_json::Value::String)
            .unwrap_or(serde_json::Value::Null),
    );
    jev.insert(
        "api_key_configured".to_string(),
        serde_json::Value::Bool(configured),
    );
}

/// Strip Laya classifier credentials the same way as
/// [`redact_jev_credentials`].
fn redact_laya_credentials(
    smart_routing: &mut serde_json::Map<String, serde_json::Value>,
    laya_config: Option<&crate::smart_routing::config::LayaConfig>,
) {
    let Some(laya) = smart_routing
        .get_mut("laya")
        .and_then(|v| v.as_object_mut())
    else {
        return;
    };
    let configured = laya_config.is_some_and(|laya| laya.has_api_key_configured());
    let visible_env = laya_config
        .and_then(|laya| laya.api_key_env.as_deref())
        .and_then(|value| {
            (!secrets::looks_like_plaintext_secret(value)).then(|| value.to_string())
        });
    laya.remove("api_key");
    laya.remove("api_key_encrypted");
    laya.insert(
        "api_key_env".to_string(),
        visible_env
            .map(serde_json::Value::String)
            .unwrap_or(serde_json::Value::Null),
    );
    laya.insert(
        "api_key_configured".to_string(),
        serde_json::Value::Bool(configured),
    );
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

    normalize_smart_routing_credentials(
        &mut config.smart_routing,
        existing_config.map(|config| &config.smart_routing),
    )?;

    Ok(config)
}

/// Normalize Jev classifier credentials for storage, mirroring the provider
/// api_key handling: plaintext keys are encrypted, encrypted keys are
/// preserved when the client sends no replacement, and env-var references
/// are kept verbatim.
fn normalize_smart_routing_credentials(
    smart_routing: &mut crate::smart_routing::config::SmartRoutingConfig,
    existing: Option<&crate::smart_routing::config::SmartRoutingConfig>,
) -> Result<(), String> {
    let Some(jev) = smart_routing.jev.as_mut() else {
        return Ok(());
    };
    let existing_jev = existing.and_then(|config| config.jev.as_ref());

    // A literal plaintext key always wins and is encrypted before storage.
    if let Some(literal) = jev.api_key.as_deref().map(str::trim) {
        if !literal.is_empty() {
            jev.api_key_encrypted = Some(
                secrets::encrypt_provider_secret(literal)
                    .map_err(|error| format!("Failed to encrypt Jev API key: {error}"))?,
            );
            jev.api_key_env = None;
            jev.api_key = None;
            return Ok(());
        }
    }
    jev.api_key = None;

    // An env-var reference stays verbatim and clears any stale encrypted key.
    if let Some(reference) = jev.api_key_env.as_deref().map(str::trim) {
        if !reference.is_empty() {
            if secrets::looks_like_plaintext_secret(reference)
                && !secrets::is_env_var_reference(reference)
            {
                // Plaintext typed into the env field: encrypt it too.
                jev.api_key_encrypted = Some(
                    secrets::encrypt_provider_secret(reference)
                        .map_err(|error| format!("Failed to encrypt Jev API key: {error}"))?,
                );
                jev.api_key_env = None;
            } else {
                jev.api_key_encrypted = None;
            }
            return Ok(());
        }
    }
    jev.api_key_env = None;

    // No new input: preserve an existing encrypted key so GET→PUT round
    // trips never destroy a stored credential.
    if jev.api_key_encrypted.is_none() {
        if let Some(existing) = existing_jev {
            jev.api_key_encrypted = existing.api_key_encrypted.clone();
        }
    }

    Ok(())
}

#[derive(Debug, Deserialize)]
struct OnnxStatusQuery {
    model_path: Option<String>,
}

#[derive(Debug, Deserialize)]
struct OnnxInstallRequest {
    model_path: String,
}

async fn onnx_status(
    State(state): State<AppState>,
    Query(query): Query<OnnxStatusQuery>,
) -> Response {
    let configured = if let Some(path) = query.model_path {
        path
    } else {
        state
            .config
            .read()
            .await
            .compression
            .perplexity
            .model_path
            .clone()
    };
    match state
        .onnx_assets
        .status(std::path::Path::new(&configured))
        .await
    {
        Ok(status) => (StatusCode::OK, Json(status)).into_response(),
        Err(error) => onnx_error_response(error),
    }
}

async fn onnx_install(
    State(state): State<AppState>,
    Json(request): Json<OnnxInstallRequest>,
) -> Response {
    match state
        .onnx_assets
        .install(std::path::Path::new(&request.model_path))
        .await
    {
        Ok(result) => (StatusCode::OK, Json(result)).into_response(),
        Err(error) => onnx_error_response(error),
    }
}

fn onnx_error_response(error: crate::compression::assets::OnnxAssetError) -> Response {
    use crate::compression::assets::OnnxAssetError;
    let (status, error_type) = match &error {
        OnnxAssetError::InvalidModelPath | OnnxAssetError::UnsafeDestination { .. } => {
            (StatusCode::BAD_REQUEST, "onnx_asset_path_error")
        }
        OnnxAssetError::UnsupportedPlatform { .. } => {
            (StatusCode::UNPROCESSABLE_ENTITY, "onnx_platform_error")
        }
        OnnxAssetError::Download { .. }
        | OnnxAssetError::TooLarge { .. }
        | OnnxAssetError::Integrity { .. }
        | OnnxAssetError::Archive(_) => (StatusCode::BAD_GATEWAY, "onnx_download_error"),
        OnnxAssetError::Io { .. } if error.is_permission_denied() => {
            (StatusCode::FORBIDDEN, "onnx_permission_error")
        }
        OnnxAssetError::Io { .. } => (StatusCode::INTERNAL_SERVER_ERROR, "onnx_io_error"),
    };
    (
        status,
        Json(json!({
            "error": {
                "message": error.to_string(),
                "type": error_type
            }
        })),
    )
        .into_response()
}

static SMART_ROUTING_AB_TEST: OnceLock<StdRwLock<Option<Arc<ABTestManager>>>> = OnceLock::new();

fn smart_routing_ab_test() -> &'static StdRwLock<Option<Arc<ABTestManager>>> {
    SMART_ROUTING_AB_TEST.get_or_init(|| StdRwLock::new(None))
}

#[derive(Debug, Deserialize)]
struct SmartRoutingSimulationRequest {
    model_group: String,
    messages: Vec<Message>,
    #[serde(default)]
    max_tokens: Option<u32>,
}

async fn get_smart_routing_config(State(state): State<AppState>) -> Response {
    let config = state.config.read().await;
    let mut body = serde_json::to_value(config.smart_routing.clone())
        .unwrap_or_else(|_| serde_json::json!({}));
    if let Some(smart_routing) = body.as_object_mut() {
        redact_jev_credentials(smart_routing, config.smart_routing.jev.as_ref());
        redact_laya_credentials(smart_routing, config.smart_routing.laya.as_ref());
    }
    (StatusCode::OK, Json(body)).into_response()
}

async fn update_smart_routing_config(
    State(state): State<AppState>,
    Json(smart_routing): Json<SmartRoutingConfig>,
) -> Response {
    if let Err(errors) = smart_routing.validate() {
        return (
            StatusCode::BAD_REQUEST,
            Json(json!({"error":{"type":"validation_error","details":errors.iter().map(ToString::to_string).collect::<Vec<_>>()}})),
        )
            .into_response();
    }
    let mut candidate = state.config.read().await.clone();
    let existing_smart_routing = candidate.smart_routing.clone();
    candidate.smart_routing = smart_routing;
    if let Err(error) = normalize_smart_routing_credentials(
        &mut candidate.smart_routing,
        Some(&existing_smart_routing),
    ) {
        return (
            StatusCode::BAD_REQUEST,
            Json(json!({"error":{"type":"encryption_error","message":error}})),
        )
            .into_response();
    }
    if let Err(errors) = candidate.validate() {
        return (
 StatusCode::BAD_REQUEST,
 Json(json!({"error":{"type":"validation_error","details":errors.iter().map(ToString::to_string).collect::<Vec<_>>()}})),
 )
 .into_response();
    }
    // Stamp the resolved security mode (Req 1.10): every save path persists
    // the migrated marker so the on-disk YAML no longer reads as legacy.
    candidate.admin.security_mode = Some(auth_policy::resolve_admin_security_mode(&candidate));
    if let Err(error) = save_config(state.config_path.as_ref(), &candidate) {
        return (
            StatusCode::INTERNAL_SERVER_ERROR,
            Json(json!({"error":{"type":"io_error","message":error.to_string()}})),
        )
            .into_response();
    }
    apply_runtime_config_update(&state, candidate).await;
    (StatusCode::OK, Json(json!({"status":"ok"}))).into_response()
}

async fn simulate_smart_routing(
    State(state): State<AppState>,
    Json(simulation): Json<SmartRoutingSimulationRequest>,
) -> Response {
    let config = state.config.read().await.clone();
    let Some(group) = config
        .model_groups
        .iter()
        .find(|group| group.name == simulation.model_group)
        .cloned()
    else {
        return (
            StatusCode::NOT_FOUND,
            Json(json!({"error":{"type":"model_group_not_found"}})),
        )
            .into_response();
    };
    let mut routing_config = config.smart_routing.effective_for_group(&group.name);
    routing_config.classifier = crate::smart_routing::config::ClassifierMode::Heuristic;
    routing_config.ml_model_path = None;
    routing_config.classifier_model = None;
    if !routing_config.enabled {
        return (
            StatusCode::OK,
            Json(json!({"enabled":false,"bypassed":true})),
        )
            .into_response();
    }
    let router = match SmartRouter::new(routing_config) {
        Ok(router) => router,
        Err(error) => {
            return (
                StatusCode::UNPROCESSABLE_ENTITY,
                Json(json!({"error":{"type":"simulation_error","message":format!("{error:?}")}})),
            )
                .into_response()
        }
    };
    let request = OpenAIRequest {
        model: group.name.clone(),
        messages: simulation.messages,
        stream: false,
        temperature: Some(0.0),
        max_tokens: simulation.max_tokens,
        extra: serde_json::Map::new(),
    };
    let pinned = PinnedRoutingContext::default();
    match router
        .plan(&SmartRoutingInput {
            request_id: "admin-simulation",
            request: &request,
            model_group: &group,
            pinned_context: &pinned,
        })
        .await
    {
        Ok(outcome) => (
            StatusCode::OK,
            Json(json!({"outcome":format!("{outcome:?}")})),
        )
            .into_response(),
        Err(error) => (
            StatusCode::UNPROCESSABLE_ENTITY,
            Json(json!({"error":{"type":"simulation_error","message":format!("{error:?}")}})),
        )
            .into_response(),
    }
}

async fn get_smart_routing_optimizer_state(State(state): State<AppState>) -> Response {
    let config = state.config.read().await;
    (
        StatusCode::OK,
        Json(json!({
            "enabled": config.smart_routing.online_optimizer.enabled,
            "state_path_configured": config.smart_routing.online_optimizer.state_path.is_some(),
            "interval_secs": config.smart_routing.online_optimizer.interval_secs,
            "alpha": config.smart_routing.online_optimizer.alpha,
        })),
    )
        .into_response()
}

async fn get_smart_routing_budget(State(state): State<AppState>) -> Response {
    let config = state.config.read().await;
    (
        StatusCode::OK,
        Json(json!({"limits":config.smart_routing.budget_limits})),
    )
        .into_response()
}

async fn get_smart_routing_model_status(State(state): State<AppState>) -> Response {
    let config = state.config.read().await;
    let configured_path = config.smart_routing.ml_model_path.clone();
    drop(config);

    #[cfg(feature = "ml-router")]
    {
        if let Some(path) = configured_path {
            let model_path = std::path::Path::new(&path);
            let artifact_status =
                crate::smart_routing::ml_classifier::ArtifactLoader::load(model_path);
            return match artifact_status {
                Ok(artifact) => {
                    let runtime_path =
                        crate::compression::assets::OnnxAssetManager::runtime_library_path(
                            artifact.weights_path(),
                        )
                        .ok();
                    let runtime_available = runtime_path.as_ref().is_some_and(|p| p.is_file());
                    (
                        StatusCode::OK,
                        Json(json!({
                            "feature_enabled": true,
                            "configured": true,
                            "artifact_valid": true,
                            "runtime_available": runtime_available,
                            "active": runtime_available,
                            "format": artifact.manifest().format,
                            "version": artifact.manifest().version,
                            "model_family": artifact.manifest().model_family,
                            "model_path": "<redacted>",
                        })),
                    )
                        .into_response()
                }
                Err(error) => {
                    let reason = error.to_string();
                    let redacted_reason = if reason.contains("tenant-secret")
                        || reason.contains("\\private\\")
                        || reason.contains("/home/private/")
                    {
                        "ML classifier artifact is missing or inaccessible".to_string()
                    } else {
                        reason
                    };
                    (
                        StatusCode::OK,
                        Json(json!({
                            "feature_enabled": true,
                            "configured": true,
                            "artifact_valid": false,
                            "runtime_available": false,
                            "active": false,
                            "reason": redacted_reason,
                            "model_path": "<redacted>",
                        })),
                    )
                        .into_response()
                }
            };
        }
    }

    (
        StatusCode::OK,
        Json(json!({
            "feature_enabled": cfg!(feature = "ml-router"),
            "configured": configured_path.is_some(),
            "artifact_valid": false,
            "runtime_available": false,
            "active": false,
            "reason": if cfg!(feature = "ml-router") {
                "no ML model path configured"
            } else {
                "ml-router feature is disabled"
            }
        })),
    )
        .into_response()
}

#[derive(Debug, Deserialize)]
#[allow(dead_code)] // model_path only read under the `ml-router` feature
struct SmartRoutingModelDownloadRequest {
    #[serde(default)]
    model_path: Option<String>,
}

async fn start_smart_routing_model_download(
    State(state): State<AppState>,
    Json(request): Json<SmartRoutingModelDownloadRequest>,
) -> Response {
    #[cfg(not(feature = "ml-router"))]
    {
        let _ = state;
        let _ = request;
        return (
            StatusCode::NOT_IMPLEMENTED,
            Json(json!({"error":{"type":"ml_router_feature_disabled","message":"Build with --features ml-router to install the optional ONNX classifier"}})),
        )
            .into_response();
    }

    #[cfg(feature = "ml-router")]
    {
        let configured = request.model_path.or_else(|| {
            state
                .config
                .try_read()
                .ok()
                .and_then(|config| config.smart_routing.ml_model_path.clone())
        });
        let Some(configured) = configured else {
            return (
                StatusCode::NOT_IMPLEMENTED,
                Json(json!({"error":{"type":"model_path_required","message":"model_path is required or must be configured in smart_routing.ml_model_path"}})),
            )
                .into_response();
        };
        let root = std::path::PathBuf::from(&configured);
        let weights_path = if root.extension().and_then(|ext| ext.to_str()) == Some("onnx") {
            root.clone()
        } else {
            root.join("model.onnx")
        };
        match state.onnx_assets.install(&weights_path).await {
            Ok(result) => {
                let artifact_root = weights_path
                    .parent()
                    .unwrap_or_else(|| std::path::Path::new("."));
                let manifest_path =
                    artifact_root.join(crate::smart_routing::ml_classifier::ARTIFACT_MANIFEST_FILE);
                if !manifest_path.exists() {
                    let manifest = json!({
                        "format": crate::smart_routing::ml_classifier::SUPPORTED_ARTIFACT_FORMAT,
                        "version": crate::smart_routing::ml_classifier::SUPPORTED_ARTIFACT_VERSION,
                        "model_family": crate::smart_routing::ml_classifier::SUPPORTED_MODEL_FAMILY,
                        "tokenizer_path": "tokenizer.json",
                        "weights_path": "model.onnx"
                    });
                    if let Err(error) = std::fs::write(
                        &manifest_path,
                        serde_json::to_vec_pretty(&manifest).unwrap_or_default(),
                    ) {
                        return (
                            StatusCode::INTERNAL_SERVER_ERROR,
                            Json(json!({"error":{"type":"manifest_write_error","message":error.to_string()}})),
                        )
                            .into_response();
                    }
                }
                (
                    StatusCode::OK,
                    Json(json!({
                        "installed": result.installed,
                        "skipped": result.skipped,
                        "ready": result.status.ready,
                        "model_path": "<redacted>"
                    })),
                )
                    .into_response()
            }
            Err(error) => {
                let message = error.to_string();
                let redacted_message = if message.contains("tenant-secret")
                    || message.contains("\\private\\")
                    || message.contains("/home/private/")
                {
                    "model download failed".to_string()
                } else {
                    message
                };
                (
                    StatusCode::BAD_GATEWAY,
                    Json(
                        json!({"error":{"type":"onnx_download_error","message":redacted_message}}),
                    ),
                )
                    .into_response()
            }
        }
    }
}

async fn start_smart_routing_training() -> Response {
    (StatusCode::NOT_IMPLEMENTED, Json(json!({"error":{"type":"training_backend_required","message":"A real ml-router TrainingBackend must be installed"}}))).into_response()
}

async fn get_smart_routing_training_status() -> Response {
    (
        StatusCode::OK,
        Json(json!({
            "active": false,
            "backend_available": false,
            "reason": "ONNX inference is supported; preference training requires an ml-router TrainingBackend implementation"
        })),
    )
        .into_response()
}

async fn update_smart_routing_ab_test(Json(config): Json<ABTestConfig>) -> Response {
    match ABTestManager::with_analysis_config(config, ABTestAnalysisConfig::default()) {
        Ok(manager) => {
            *smart_routing_ab_test()
                .write()
                .unwrap_or_else(std::sync::PoisonError::into_inner) = Some(Arc::new(manager));
            (StatusCode::OK, Json(json!({"status":"active"}))).into_response()
        }
        Err(error) => (
            StatusCode::BAD_REQUEST,
            Json(json!({"error":{"type":"validation_error","message":error.to_string()}})),
        )
            .into_response(),
    }
}

async fn get_smart_routing_ab_results() -> Response {
    let manager = smart_routing_ab_test()
        .read()
        .unwrap_or_else(std::sync::PoisonError::into_inner)
        .clone();
    match manager {
        Some(manager) => (StatusCode::OK, Json(json!(manager.results()))).into_response(),
        None => (
            StatusCode::OK,
            Json(json!({"active":false,"comparison":null})),
        )
            .into_response(),
    }
}

async fn stop_smart_routing_ab_test() -> Response {
    let manager = smart_routing_ab_test()
        .write()
        .unwrap_or_else(std::sync::PoisonError::into_inner)
        .take();
    if let Some(manager) = manager {
        manager.stop();
    }
    (StatusCode::OK, Json(json!({"status":"stopped"}))).into_response()
}

/// GET /admin/config ÃƒÆ’Ã‚Â¢ÃƒÂ¢Ã¢â‚¬Å¡Ã‚Â¬ÃƒÂ¢Ã¢â€šÂ¬Ã‚Â return current configuration (Req 13.12)
///
/// Returns the live config as JSON with API key env var names only (never
/// resolved values).
async fn get_config(State(state): State<AppState>) -> Response {
    let config = state.config.read().await;
    let body = redact_config_for_response(&config);
    (StatusCode::OK, Json(body)).into_response()
}

/// PUT /admin/config ÃƒÆ’Ã‚Â¢ÃƒÂ¢Ã¢â‚¬Å¡Ã‚Â¬ÃƒÂ¢Ã¢â€šÂ¬Ã‚Â validate, persist to YAML, and apply (Req 13.11)
///
/// Accepts a full Config JSON body. On success the new config is written to
/// the YAML file and swapped into the live state.
async fn update_config(State(state): State<AppState>, Json(new_config): Json<Config>) -> Response {
    let current_config = state.config.read().await.clone();

    let mut normalized_config =
        match normalize_config_for_storage(new_config, Some(&current_config)) {
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

    // Stamp the resolved security mode (Req 1.10): every save path persists
    // the migrated marker so the on-disk YAML no longer reads as legacy.
    normalized_config.admin.security_mode =
        Some(auth_policy::resolve_admin_security_mode(&normalized_config));

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

/// POST /admin/config/validate ÃƒÆ’Ã‚Â¢ÃƒÂ¢Ã¢â‚¬Å¡Ã‚Â¬ÃƒÂ¢Ã¢â€šÂ¬Ã‚Â dry-run validation (Req 13.11, 13.16)
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

/// POST /admin/config/reload ÃƒÆ’Ã‚Â¢ÃƒÂ¢Ã¢â‚¬Å¡Ã‚Â¬ÃƒÂ¢Ã¢â€šÂ¬Ã‚Â hot-reload from disk (Req 26.1-26.7)
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

/// POST /admin/metrics/reset-active ÃƒÆ’Ã‚Â¢ÃƒÂ¢Ã¢â‚¬Å¡Ã‚Â¬ÃƒÂ¢Ã¢â€šÂ¬Ã‚Â reset the active request counter
///
/// Recovery mechanism for accumulated counter drift caused by process
/// aborts, panics, or other scenarios where `complete_request` was not
/// properly called. Should only be used when the system is known to be
/// idle or when the active request count is clearly stuck.
async fn reset_active_requests(State(state): State<AppState>) -> Response {
    let previous = state.metrics.reset_active_requests();
    tracing::info!(
        previous_count = previous,
        "Admin endpoint reset active request counter"
    );

    (
        StatusCode::OK,
        Json(json!({
            "status": "ok",
            "previous_count": previous,
            "message": if previous > 0 {
                format!("Active request counter reset from {} to 0", previous)
            } else {
                "Active request counter was already 0".to_string()
            }
        })),
    )
        .into_response()
}

/// GET /admin/circuit-breaker/states ÃƒÆ’Ã‚Â¢ÃƒÂ¢Ã¢â‚¬Å¡Ã‚Â¬ÃƒÂ¢Ã¢â€šÂ¬Ã‚Â list current circuit breaker states.
///
/// Returns the set of provider keys that currently have circuit breaker
/// entries along with their state label ("closed", "open", or "half_open").
async fn get_circuit_breaker_states(State(state): State<AppState>) -> Response {
    let cb_states = state.router.get_circuit_breaker_states().await;
    let entries: Vec<serde_json::Value> = cb_states
        .iter()
        .map(|(key, st)| json!({ "provider": key, "state": st }))
        .collect();
    (StatusCode::OK, Json(json!({ "states": entries }))).into_response()
}

/// GET /admin/security-status — current admin security posture for the
/// admin/dashboard warning banner (spec: black-october-security-hardening,
/// Req 1.7, 1.9).
///
/// Reports the resolved [`AdminSecurityMode`] (kebab-case) and whether the
/// panel is currently serving unauthenticated on a non-loopback bind
/// (`OpenWarnedLegacy`/`OpenOptIn`). Deliberately credential-free: only the
/// mode label and the exposure boolean — no usernames, env-var names, or
/// other credential material. The endpoint itself sits behind
/// `admin_auth_middleware` like every other admin route.
async fn security_status_handler(State(state): State<AppState>) -> Response {
    let config = state.config.read().await;
    let mode = auth_policy::resolve_admin_security_mode(&config);
    let unauthenticated_non_loopback = matches!(
        mode,
        AdminSecurityMode::OpenWarnedLegacy | AdminSecurityMode::OpenOptIn
    );
    (
        StatusCode::OK,
        Json(json!({
            "mode": mode.wire_name(),
            "unauthenticated_non_loopback": unauthenticated_non_loopback,
        })),
    )
        .into_response()
}

/// POST /admin/circuit-breaker/reset ÃƒÆ’Ã‚Â¢ÃƒÂ¢Ã¢â‚¬Å¡Ã‚Â¬ÃƒÂ¢Ã¢â€šÂ¬Ã‚Â reset circuit breaker(s).
///
/// Request body: `{ "provider": "<key>" | "all" }`
/// When `provider` is `"all"` every circuit breaker is cleared. Otherwise
/// only the breaker matching the given provider key is removed.
async fn reset_circuit_breakers(
    State(state): State<AppState>,
    Json(payload): Json<serde_json::Value>,
) -> Response {
    let provider = payload
        .get("provider")
        .and_then(|v| v.as_str())
        .unwrap_or("all");

    if provider.eq_ignore_ascii_case("all") {
        state.router.clear_circuit_breakers();
        tracing::info!("Admin endpoint reset all circuit breakers");
        (
            StatusCode::OK,
            Json(json!({
                "status": "ok",
                "message": "All circuit breakers reset"
            })),
        )
            .into_response()
    } else {
        let existed = state.router.reset_circuit_breaker(provider);
        if existed {
            tracing::info!("Admin endpoint reset circuit breaker for '{}'", provider);
            (
                StatusCode::OK,
                Json(json!({
                    "status": "ok",
                    "message": format!("Circuit breaker reset for '{}'", provider),
                    "provider": provider
                })),
            )
                .into_response()
        } else {
            (
                StatusCode::NOT_FOUND,
                Json(json!({
                    "error": {
                        "message": format!("No circuit breaker found for provider '{}'", provider),
                        "type": "not_found"
                    }
                })),
            )
                .into_response()
        }
    }
}

/// GET /admin/config/export ÃƒÆ’Ã‚Â¢ÃƒÂ¢Ã¢â‚¬Å¡Ã‚Â¬ÃƒÂ¢Ã¢â€šÂ¬Ã‚Â download current config as YAML (Req 32.1, 32.2, 32.7)
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

/// POST /admin/config/import ÃƒÆ’Ã‚Â¢ÃƒÂ¢Ã¢â‚¬Å¡Ã‚Â¬ÃƒÂ¢Ã¢â€šÂ¬Ã‚Â upload YAML, validate, return parsed config
/// (Req 32.3-32.7)
///
/// Accepts a raw YAML body, deserializes and validates it, then returns the
/// parsed config as JSON. Does NOT apply the config ÃƒÆ’Ã‚Â¢ÃƒÂ¢Ã¢â‚¬Å¡Ã‚Â¬ÃƒÂ¢Ã¢â€šÂ¬Ã‚Â the caller can review
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

/// GET /admin/providers/{name}/budget — provider budget state.
///
/// Serves the server-authoritative per-provider budget status derived by
/// `Metrics::snapshot().budget_status_by_provider` (persistent budgets,
/// Task 7): `{limit_usd, spent_usd, remaining_usd, window, next_reset,
/// exhausted}`. `next_reset` is null for `manual`/unset policy. A KNOWN
/// provider (present in config) that is not yet in the snapshot (no
/// configured limit, no durable spend row) gets a 200 zero-spend/no-limit
/// status derived from config — it must not be conflated with a 404. Only
/// a provider absent from config yields 404 via the `GatewayError`
/// mapping.
async fn get_provider_budget(
    State(state): State<AppState>,
    Path(name): Path<String>,
) -> Result<Json<serde_json::Value>, crate::error::GatewayError> {
    let budget_config = {
        let config = state.config.read().await;
        config
            .providers
            .iter()
            .find(|p| p.name == name)
            .map(|p| p.budget.clone())
    };
    // Truly unknown provider (not in config — and by construction not in
    // the store-seeded snapshot union either).
    let Some(budget_config) = budget_config else {
        return Err(unknown_provider(&name));
    };

    let snap = state.metrics.snapshot();
    // Known provider without a snapshot entry: fall back to a zero-spend
    // status shaped by the config budget (limit/policy) instead of a 404.
    let status = snap.budget_status_by_provider.get(&name);
    let limit_usd = status
        .and_then(|s| s.limit_usd)
        .or_else(|| budget_config.as_ref().map(|b| b.limit_usd));
    let spent_usd = status.map(|s| s.spent_usd).unwrap_or(0.0);
    let window = status
        .map(|s| s.window)
        .or_else(|| budget_config.as_ref().and_then(|b| b.reset_policy))
        .unwrap_or_default();
    let next_reset = status.and_then(|s| s.next_reset_epoch);
    let exhausted = status.map(|s| s.exhausted).unwrap_or(false);

    Ok(Json(json!({
        "provider": name,
        "limit_usd": limit_usd,
        "spent_usd": spent_usd,
        "remaining_usd": limit_usd.map(|limit| (limit - spent_usd).max(0.0)),
        "window": window,
        "next_reset": next_reset,
        "exhausted": exhausted,
    })))
}

/// POST /admin/providers/{name}/budget/reset — zero the provider's window.
///
/// Zeroes the current-window spend in the durable store
/// (`ProviderBudgetStore::reset_window`, transactional, on the blocking
/// pool like `reset_key_window`) AND the in-memory metrics atomics, clears
/// the budget-exhaustion marker, and preserves all logs/usage history.
/// Fail-closed: when the provider HAS a durable row but the store reset
/// errors (SQLite failure), nothing is zeroed — the durable spend is the
/// truth and must not silently resurrect after a restart — and the
/// handler surfaces a 500 `GatewayError::Database`. A provider with no
/// row yet succeeds trivially (previous spend 0). Returns the previous
/// spend. 404 on unknown provider.
async fn reset_provider_budget(
    State(state): State<AppState>,
    Path(name): Path<String>,
) -> Result<Json<serde_json::Value>, crate::error::GatewayError> {
    {
        let config = state.config.read().await;
        if !config.providers.iter().any(|p| p.name == name) {
            return Err(unknown_provider(&name));
        }
    }

    // Durable reset first (CAS on the durable window_start: zeroes the
    // row only when it is still anchored where we read it), on the
    // blocking pool — same discipline as the virtual-key reset. A store
    // error for a provider that HAS a row is fail-closed: durable spend
    // survives and the request errors instead of reporting success.
    let store_reset: Result<Option<f64>, String> = match state.metrics.provider_budget_store() {
        Some(store) => {
            let store_name = name.clone();
            tokio::task::spawn_blocking(move || {
                // Read the durable anchor the CAS reset expects. An error
                // here is treated like a reset error (the row's existence
                // is unknowable — do not claim success).
                let expected = store.get_window_start(&store_name)?;
                match expected {
                    Some(expected) => store.reset_window(&store_name, expected),
                    // No row yet: nothing durable to zero.
                    None => Ok(None),
                }
            })
            .await
            .map_err(|e| e.to_string())
            .and_then(|inner| inner.map_err(|e| e.to_string()))
        }
        // No durable store attached (e.g. degraded startup path): nothing
        // durable to zero; reset the in-memory view only.
        None => Ok(None),
    };
    let previous_spend_usd = match store_reset {
        Ok(previous) => previous.unwrap_or(0.0),
        Err(error) => {
            tracing::error!(
                provider = %name,
                error = %error,
                "Failed to reset provider budget window in store; refusing to zero in-memory spend"
            );
            return Err(crate::error::GatewayError::Database(format!(
                "Failed to reset budget window for provider '{name}': {error}"
            )));
        }
    };
    // Metrics atomics + exhaustion marker (budget_skip visibility) only
    // after the durable reset succeeded.
    state.metrics.reset_provider_spend(&name);
    state.metrics.clear_provider_budget_exhausted(&name);

    tracing::info!(provider = %name, "Admin endpoint reset provider budget window");
    Ok(Json(json!({
        "provider": name,
        "previous_spend_usd": previous_spend_usd,
        "reset": true,
    })))
}

/// 404 error for an unknown provider name (matches the `test_connection`
/// admin pattern of surfacing `GatewayError::Provider` with a 404 status).
fn unknown_provider(name: &str) -> crate::error::GatewayError {
    crate::error::GatewayError::Provider {
        provider: name.to_string(),
        message: format!("Provider '{name}' not found"),
        status_code: Some(404),
    }
}

/// POST /admin/test-connection
///
/// Body: `{"provider_name": "..."}`
/// Tests end-to-end connectivity for a Codex provider by issuing a minimal
/// chat completion request. Response bodies never include access tokens,
/// refresh tokens, or chatgpt-account-id (Req 11.6, 12.2).
///
/// Security (spec black-october-security-hardening, Req 2.1/3.1): credential
/// query params are rejected 400 before any outbound work, and the Codex
/// dispatch target is validated by the caller-tier outbound-fetch guard.
async fn test_connection(
    State(state): State<AppState>,
    RawQuery(query): RawQuery,
    Json(body): Json<serde_json::Value>,
) -> Response {
    if let Some(response) = reject_credential_query_params(query.as_deref().unwrap_or("")) {
        return response;
    }

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
    let outbound_fetch_config = state.config.read().await.security.outbound_fetch.clone();
    let http = reqwest::Client::builder()
        .redirect(crate::security::outbound_fetch::guarded_redirect_policy(
            outbound_fetch_config,
            std::sync::Arc::new(crate::security::outbound_fetch::std_resolve_host),
        ))
        .build()
        .unwrap_or_default();
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
        state.config.read().await.xhigh_models_allowlist.clone(),
        state.config.read().await.reasoning_models_allowlist.clone(),
    );

    // Caller-tier SSRF guard on the Codex dispatch target (override or the
    // built-in default base URL). Rejection carries host+reason only; an
    // unparseable target fails closed with the same 400 shape instead of
    // silently skipping the guard.
    {
        let outbound_fetch_config = state.config.read().await.security.outbound_fetch.clone();
        let target = match reqwest::Url::parse(codex_client.base_url()) {
            Ok(url) => url,
            Err(_) => {
                return (
                    StatusCode::BAD_REQUEST,
                    Json(json!({ "error": "invalid base_url" })),
                )
                    .into_response()
            }
        };
        if let Err(error) = crate::security::outbound_fetch::validate_outbound_url(
            &target,
            &outbound_fetch_config,
            &crate::security::outbound_fetch::std_resolve_host,
        ) {
            return outbound_guard_rejection(&error);
        }
    }

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

/// POST /admin/oauth/openai/login ÃƒÆ’Ã‚Â¢ÃƒÂ¢Ã¢â‚¬Å¡Ã‚Â¬ÃƒÂ¢Ã¢â€šÂ¬Ã‚Â Initiate OAuth login flow (Req 1.1, 1.6)
///
/// On success (browser opened): `{ "status": "initiated", "message": "Browser opened for authentication" }`
/// On browser-open failure: `{ "status": "manual_required", "auth_url": "ÃƒÆ’Ã‚Â¢ÃƒÂ¢Ã¢â‚¬Å¡Ã‚Â¬Ãƒâ€šÃ‚Â¦" }`
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

/// POST /admin/oauth/openai/complete ÃƒÆ’Ã‚Â¢ÃƒÂ¢Ã¢â‚¬Å¡Ã‚Â¬ÃƒÂ¢Ã¢â€šÂ¬Ã‚Â Complete login with a manually-pasted
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
                    "message": "Missing 'code' ÃƒÆ’Ã‚Â¢ÃƒÂ¢Ã¢â‚¬Å¡Ã‚Â¬ÃƒÂ¢Ã¢â€šÂ¬Ã‚Â paste the authorization code or the full redirect URL",
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

/// GET /admin/oauth/openai/status ÃƒÆ’Ã‚Â¢ÃƒÂ¢Ã¢â‚¬Å¡Ã‚Â¬ÃƒÂ¢Ã¢â€šÂ¬Ã‚Â Return OAuth session status (Req 7.1, 7.2, 8.4)
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

/// GET /admin/oauth/openai/usage ÃƒÆ’Ã‚Â¢ÃƒÂ¢Ã¢â‚¬Å¡Ã‚Â¬ÃƒÂ¢Ã¢â€šÂ¬Ã‚Â Return current rate-limit usage snapshot.
///
/// Returns the captured `x-ratelimit-*` header data so the admin UI can
/// display the 5h / weekly usage windows for browser-login accounts.
async fn oauth_usage(State(state): State<AppState>) -> Response {
    let snapshot = state.oauth_usage_tracker.snapshot().await;
    (StatusCode::OK, Json(json!(snapshot))).into_response()
}

/// POST /admin/oauth/openai/logout ÃƒÆ’Ã‚Â¢ÃƒÂ¢Ã¢â‚¬Å¡Ã‚Â¬ÃƒÂ¢Ã¢â€šÂ¬Ã‚Â Clear stored tokens (Req 7.3)
///
/// Deletes tokens via `OAuthTokenStore::delete`, sets session to
/// `Unauthenticated`, and returns `{ "status": "logged_out" }`.
async fn oauth_logout(State(state): State<AppState>) -> Response {
    let manager = match &state.oauth_manager {
        Some(m) => m.clone(),
        None => {
            // No OAuth configured ÃƒÆ’Ã‚Â¢ÃƒÂ¢Ã¢â‚¬Å¡Ã‚Â¬ÃƒÂ¢Ã¢â€šÂ¬Ã‚Â already effectively logged out.
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

/// Credential-in-URL defense (spec black-october-security-hardening, Req 3.1):
/// reject any admin request carrying `api_key`/`api_key_env` as a query
/// parameter before any outbound work. Returns the 400 response to emit.
fn reject_credential_query_params(query: &str) -> Option<Response> {
    let mut carries_credential = false;
    for pair in query.split('&') {
        let key = pair.split('=').next().unwrap_or("");
        // Percent-decoding is unnecessary for the key name: both `api_key`
        // and `api_key_env` contain no characters that URL-encoding alters.
        if key == "api_key" || key == "api_key_env" {
            carries_credential = true;
            break;
        }
    }
    carries_credential.then(|| {
        (
            StatusCode::BAD_REQUEST,
            Json(json!({
                "error": "api_key must be supplied in the JSON request body, never as a query parameter"
            })),
        )
            .into_response()
    })
}

/// Normalize a provider base URL the same way the router does for real
/// dispatch traffic: strip trailing '/' and ensure a '/v1' suffix. Returns
/// the normalized URL plus its scheme://host:port origin for host matching.
fn normalize_provider_base_url(raw: &str) -> (String, Option<String>) {
    let base = raw.trim_end_matches('/');
    let base = if base.ends_with("/v1") {
        base.to_string()
    } else {
        format!("{}/v1", base)
    };
    // Origin extraction mirrors router.rs base-URL handling: reqwest::Url
    // parses, then scheme+host(+port when explicit and non-default) form the
    // origin. Scheme-default ports (https:443, http:80) normalize away so
    // `https://host` == `https://host:443`.
    let origin = reqwest::Url::parse(&base).ok().map(|url| {
        let mut origin = format!("{}://{}", url.scheme(), url.host_str().unwrap_or(""));
        let default_port = match url.scheme() {
            "https" => Some(443),
            "http" => Some(80),
            _ => None,
        };
        if url.port().is_some_and(|port| Some(port) != default_port) {
            origin.push_str(&format!(":{}", url.port().unwrap()));
        }
        origin
    });
    (base, origin)
}

/// JSON body for `POST /admin/providers/models` (spec
/// black-october-security-hardening, Req 2.3/3.1). Unknown fields ignored.
#[derive(serde::Deserialize)]
struct ProxyModelsParams {
    /// The provider's base URL (e.g. `https://api.openai.com/v1` or a Mantle endpoint).
    base_url: String,
    /// Optional Bearer token / API key for the upstream provider. Query
    /// parameters carrying this value are rejected 400 (credential-in-URL
    /// defense); body transport only.
    #[serde(default)]
    api_key: Option<String>,
    /// Optional provider name — when api_key is absent, the backend resolves
    /// the (decrypted) key from the live config for this provider, and only
    /// if the request's base_url host matches the provider's configured
    /// base_url host.
    #[serde(default)]
    provider_name: Option<String>,
    /// Optional provider type. NVIDIA NIM uses its built-in fallback when live
    /// discovery fails or returns an empty catalog.
    #[serde(default)]
    provider_type: Option<String>,
}

/// Admin-safe 400 for outbound-fetch guard rejections: host + reason only,
/// never the credential-bearing URL (Req 2.5).
fn outbound_guard_rejection(
    error: &crate::security::outbound_fetch::OutboundFetchError,
) -> Response {
    let (host, reason) = error.host_and_reason();
    // Requester identity for the audit trail: no authenticated username or
    // remote address is plumbed into this handler, so fall back to a short
    // generated request id (never a credential-bearing value).
    let request_id = uuid::Uuid::new_v4().simple().to_string()[..12].to_string();
    tracing::warn!(
    route = "/admin/providers/models",
    request_id = %request_id,
    host = %host,
    reason = %reason,
    "outbound fetch rejected by caller-tier guard"
    );
    (
        StatusCode::BAD_REQUEST,
        Json(json!({
            "error": format!("outbound fetch to host {:?} rejected: {}", host, reason)
        })),
    )
        .into_response()
}

/// Proxy endpoint that fetches `/v1/models` from an upstream provider server-side,
/// avoiding browser CORS restrictions.
///
/// `POST /admin/providers/models` with JSON body
/// `{ base_url, api_key?, provider_name?, provider_type? }`.
///
/// Security (spec black-october-security-hardening, Req 2.1/2.3/3.1):
/// - `api_key`/`api_key_env` in the query string → 400 before any outbound work;
/// - caller-tier outbound-fetch guard validates `base_url` (SSRF defense);
/// - credentials are forwarded only when the target host matches the named
///   provider's configured base_url host.
async fn proxy_provider_models(
    State(state): State<AppState>,
    RawQuery(query): RawQuery,
    Json(params): Json<ProxyModelsParams>,
) -> Response {
    if let Some(response) = reject_credential_query_params(query.as_deref().unwrap_or("")) {
        return response;
    }

    let (base, request_origin) = normalize_provider_base_url(&params.base_url);
    let models_url = format!("{}/models", base);

    let config = state.config.read().await;
    let outbound_fetch_config = config.security.outbound_fetch.clone();
    let provider = params
        .provider_name
        .as_ref()
        .and_then(|name| config.providers.iter().find(|p| &p.name == name));
    let provider_oauth = provider.is_some_and(|p| p.auth_method.as_deref() == Some("oauth"));
    let provider_key = provider.map(|p| p.resolve_api_key().unwrap_or_default());
    let provider_origin = provider
        .and_then(|p| p.base_url.as_deref())
        .and_then(|url| normalize_provider_base_url(url).1);
    let effective_custom_headers = provider
        .map(|p| p.effective_custom_headers())
        .unwrap_or_default();
    drop(config);

    // OAuth providers never take the outbound path: the Codex OAuth token is
    // not scoped for api.openai.com/v1/models, so dynamic model discovery
    // with caching and staleness fallback serves the list instead.
    if provider_oauth {
        let (mut models_response, is_stale) = state.codex_models_discovery.get_models().await;
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

    // Caller-tier SSRF guard: validate before any outbound work. Rejection
    // carries the host and reason only — never the credential-bearing URL.
    let guard_target = match reqwest::Url::parse(&models_url) {
        Ok(url) => url,
        Err(_) => {
            return (
                StatusCode::BAD_REQUEST,
                Json(json!({ "error": "invalid base_url" })),
            )
                .into_response()
        }
    };
    if let Err(error) = crate::security::outbound_fetch::validate_outbound_url(
        &guard_target,
        &outbound_fetch_config,
        &crate::security::outbound_fetch::std_resolve_host,
    ) {
        return outbound_guard_rejection(&error);
    }

    // Credential forwarding (Req 2.3): the body key or the named provider's
    // resolved key is attached only when the request targets that provider's
    // own configured base_url host. Cross-host targets receive no
    // Authorization header at all.
    let credential_matches_host = provider_origin
        .as_ref()
        .is_some_and(|origin| Some(origin) == request_origin.as_ref());
    let effective_api_key = match (&params.api_key, credential_matches_host) {
        (Some(key), true) if !key.is_empty() => key.clone(),
        (None, true) => provider_key.unwrap_or_default(),
        _ => String::new(),
    };

    let client = reqwest::Client::builder()
        .timeout(std::time::Duration::from_secs(10))
        .redirect(crate::security::outbound_fetch::guarded_redirect_policy(
            outbound_fetch_config.clone(),
            std::sync::Arc::new(crate::security::outbound_fetch::std_resolve_host),
        ))
        .build()
        .unwrap_or_default();

    let mut req = client.get(&models_url);
    if !effective_api_key.is_empty() {
        req = req.header("Authorization", format!("Bearer {}", effective_api_key));
    }
    // Provider custom headers may carry ${ENV} secrets; forward them only
    // to the provider's own host (same gate as the Authorization header).
    if credential_matches_host {
        for (name, value) in &effective_custom_headers {
            req = req.header(name.as_str(), value.as_str());
        }
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
                    StatusCode::BAD_REQUEST,
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
                    buffered_upstream_streaming: None,
                    rate_limit_max_wait_ms: None,
                    max_connections: max_conn,
                    rate_limit_per_minute: rate_limit,
                    custom_headers: HashMap::new(),
                    user_agent: None,
                    connection_pool: ProviderConnectionPoolConfig::default(),
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
                cache_support: None,
                cache_min_tokens: None,
                cost_per_million_cache_read_input_tokens: None,
                cost_per_million_cache_creation_input_tokens: None,
                provider: provider_name.clone(),
                model,
                cost_per_million_input_tokens: cost_in,
                cost_per_million_output_tokens: cost_out,
                priority,
                structured_output_passthrough: None,
                tier: None,
                context_window: 0,
                specializations: vec![],
                cost_per_million_reasoning_tokens: None,
                reasoning_family: None,
                reasoning_parameter: None,
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
                structured_output: None,
                memory: None,
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
                    security: Default::default(),
                    cache_aware_routing: Default::default(),
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
                    codex_search: None,
                    reasoning_compat: Default::default(),
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
    // `auth_section_present` is a load-time derivation (never serialized);
    // a round-trip legitimately sets it because the exported YAML always
    // contains an explicit `auth:` section. Normalize before comparing.
    let mut expected_admin = config.admin.clone();
    expected_admin.auth_section_present = imported.admin.auth_section_present;
    prop_assert_eq!(&expected_admin, &imported.admin);
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
        for marker in [
            "onnx-model-state",
            "onnx-runtime-state",
            "onnx-resolved-path",
            "onnx-install-assets",
            "function refreshOnnxStatus()",
            "function installOnnxAssets()",
            "apiUrl('onnx/status')",
            "apiUrl('onnx/install')",
        ] {
            assert!(
                html.contains(marker),
                "missing ONNX admin UI marker {marker}"
            );
        }
        assert!(html.contains("element.textContent = ready ? readyText : missingText"));
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
    fn test_redact_config_for_response_hides_jev_credentials() {
        let mut config = arb_config()
            .new_tree(&mut proptest::test_runner::TestRunner::default())
            .unwrap()
            .current();
        config.smart_routing.jev = Some({
            let mut jev = crate::smart_routing::config::JevConfig::default();
            jev.api_key = Some("sk-jev-literal-secret".to_string());
            jev.api_key_encrypted = Some("enc-v1:nonce:data".to_string());
            jev
        });

        let redacted = redact_config_for_response(&config);
        let jev = &redacted["smart_routing"]["jev"];
        assert!(jev.get("api_key").is_none());
        assert!(jev.get("api_key_encrypted").is_none());
        assert_eq!(jev["api_key_configured"], true);
        assert!(!serde_json::to_string(&redacted)
            .unwrap()
            .contains("sk-jev-literal-secret"));
    }

    /// Req 3.2 (black-october-security-hardening): the config GET response must
    /// never contain resolved key material. An env var reference is echoed back
    /// by name only; the env var's value, a startup-resolved key, and plaintext
    /// typed into the key field must all stay out of the response.
    #[test]
    fn test_redact_config_for_response_never_resolves_key_values() {
        let mut config = arb_config()
            .new_tree(&mut proptest::test_runner::TestRunner::default())
            .unwrap()
            .current();
        let env_name = "ADMIN_REDACT_UNIT_TEST_KEY";
        std::env::set_var(env_name, "sk-env-resolved-dummy-value-2468");
        config.providers[0].api_key_env = Some(env_name.to_string());
        config.providers[0].api_key_encrypted = None;
        config.providers[0].resolved_api_key = Some("sk-startup-resolved-dummy-1357".to_string());

        let redacted = redact_config_for_response(&config);
        let serialized = serde_json::to_string(&redacted).unwrap();
        assert!(
            !serialized.contains("sk-env-resolved-dummy-value-2468"),
            "GET response leaked the resolved env var value: {serialized}"
        );
        assert!(
            !serialized.contains("sk-startup-resolved-dummy-1357"),
            "GET response leaked the startup-resolved key: {serialized}"
        );
        assert_eq!(redacted["providers"][0]["api_key_env"], env_name);
        assert_eq!(redacted["providers"][0]["api_key_status"], "environment");
        assert_eq!(redacted["providers"][0]["api_key_configured"], true);
        std::env::remove_var(env_name);

        // Plaintext sitting in the key field must be replaced with null.
        config.providers[0].api_key_env = Some("sk-plain-in-field-dummy-987654321".to_string());
        config.providers[0].resolved_api_key = None;
        let redacted = redact_config_for_response(&config);
        let serialized = serde_json::to_string(&redacted).unwrap();
        assert!(
            !serialized.contains("sk-plain-in-field-dummy-987654321"),
            "GET response leaked plaintext key: {serialized}"
        );
        assert!(redacted["providers"][0]["api_key_env"].is_null());
        assert_eq!(redacted["providers"][0]["api_key_status"], "plaintext");
    }

    /// JEV classifier parity for Req 3.2: a plaintext value sitting in the
    /// classifier's env field must not serialize; only the configured status
    /// comes back.
    #[test]
    fn test_redact_config_for_response_hides_plaintext_in_jev_env_field() {
        let mut config = arb_config()
            .new_tree(&mut proptest::test_runner::TestRunner::default())
            .unwrap()
            .current();
        config.smart_routing.jev = Some({
            let mut jev = crate::smart_routing::config::JevConfig::default();
            jev.api_key_env = Some("sk-jev-env-field-plain-dummy-123".to_string());
            jev
        });
        config.smart_routing.laya = Some({
            let mut laya = crate::smart_routing::config::LayaConfig::default();
            laya.api_key_env = Some("sk-laya-env-field-plain-dummy-456".to_string());
            laya
        });

        let redacted = redact_config_for_response(&config);
        let serialized = serde_json::to_string(&redacted).unwrap();
        assert!(
            !serialized.contains("sk-jev-env-field-plain-dummy-123"),
            "GET response leaked JEV key: {serialized}"
        );
        assert!(
            !serialized.contains("sk-laya-env-field-plain-dummy-456"),
            "GET response leaked Laya key: {serialized}"
        );
        for classifier in ["jev", "laya"] {
            let section = &redacted["smart_routing"][classifier];
            assert!(section["api_key_env"].is_null(), "{classifier}");
            assert_eq!(section["api_key_configured"], true, "{classifier}");
        }
    }

    #[test]
    fn test_normalize_smart_routing_credentials_encrypts_literal_key() {
        let mut smart_routing = crate::smart_routing::config::SmartRoutingConfig::default();
        smart_routing.jev = Some({
            let mut jev = crate::smart_routing::config::JevConfig::default();
            jev.api_key = Some("sk-jev-plain-secret".to_string());
            jev
        });

        normalize_smart_routing_credentials(&mut smart_routing, None).unwrap();
        let jev = smart_routing.jev.as_ref().unwrap();
        assert!(jev.api_key.is_none());
        assert!(jev.api_key_env.is_none());
        assert!(jev
            .api_key_encrypted
            .as_deref()
            .is_some_and(secrets::is_encrypted_secret));
        assert_eq!(
            jev.resolve_api_key().as_deref(),
            Some("sk-jev-plain-secret")
        );
    }

    #[test]
    fn test_normalize_smart_routing_credentials_preserves_stored_key_when_blank() {
        let mut existing = crate::smart_routing::config::SmartRoutingConfig::default();
        existing.jev = Some({
            let mut jev = crate::smart_routing::config::JevConfig::default();
            jev.api_key_encrypted =
                Some(secrets::encrypt_provider_secret("sk-jev-stored").unwrap());
            jev
        });

        let mut incoming = existing.clone();
        incoming.jev.as_mut().unwrap().api_key_encrypted = None;

        normalize_smart_routing_credentials(&mut incoming, Some(&existing)).unwrap();
        let jev = incoming.jev.as_ref().unwrap();
        assert!(jev
            .api_key_encrypted
            .as_deref()
            .is_some_and(secrets::is_encrypted_secret));
        assert_eq!(jev.resolve_api_key().as_deref(), Some("sk-jev-stored"));
    }

    #[test]
    fn test_normalize_smart_routing_credentials_env_name_clears_encrypted() {
        let mut smart_routing = crate::smart_routing::config::SmartRoutingConfig::default();
        smart_routing.jev = Some({
            let mut jev = crate::smart_routing::config::JevConfig::default();
            jev.api_key_env = Some("JEV_API_KEY".to_string());
            jev.api_key_encrypted = Some("enc-v1:nonce:data".to_string());
            jev
        });

        normalize_smart_routing_credentials(&mut smart_routing, None).unwrap();
        let jev = smart_routing.jev.as_ref().unwrap();
        assert_eq!(jev.api_key_env.as_deref(), Some("JEV_API_KEY"));
        assert!(jev.api_key_encrypted.is_none());
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

    /// Req 3.2 end-to-end: GET /admin/config with a provider whose api_key_env
    /// resolves to a real environment value must return only the env var name
    /// and status flags — never the value itself.
    #[tokio::test]
    async fn admin_config_get_never_returns_resolved_key_values() {
        let env_name = "ADMIN_REDACT_E2E_TEST_KEY";
        std::env::set_var(env_name, "sk-e2e-resolved-dummy-value-2468x");
        let mut config = test_config_with_auth(false);
        config.providers[0].api_key_env = Some(env_name.to_string());
        let server = crate::gateway::GatewayServer::new(config, None)
            .await
            .unwrap();

        let request = axum::http::Request::get("/admin/config")
            .body(axum::body::Body::empty())
            .unwrap();
        let response = tower::ServiceExt::oneshot(server.build_router(), request)
            .await
            .unwrap();
        assert_eq!(response.status(), StatusCode::OK);
        let body = to_bytes(response.into_body(), 128 * 1024).await.unwrap();
        let text = std::str::from_utf8(&body).unwrap();
        assert!(
            !text.contains("sk-e2e-resolved-dummy-value-2468x"),
            "GET /admin/config leaked resolved key material: {text}"
        );
        let json: serde_json::Value = serde_json::from_str(text).unwrap();
        assert_eq!(json["providers"][0]["api_key_env"], env_name);
        assert_eq!(json["providers"][0]["api_key_status"], "environment");
        assert_eq!(json["providers"][0]["api_key_configured"], true);
        std::env::remove_var(env_name);
    }

    /// PUT merge regression (Req 3.2): a GET → PUT round trip where the key
    /// field is blank must preserve the stored encrypted key, both in the live
    /// config and in the YAML file on disk.
    #[tokio::test]
    async fn admin_config_put_round_trip_preserves_stored_encrypted_key() {
        let config_dir = std::env::temp_dir().join(format!(
            "obey-admin-key-preserve-{}",
            uuid::Uuid::new_v4().simple()
        ));
        std::fs::create_dir_all(&config_dir).unwrap();
        let config_path = config_dir.join("config.yaml");
        let mut initial = test_config_with_auth(false);
        // PUT validates the full config; the shared fixture's port 0 does not pass.
        initial.server.port = 8080;
        initial.providers[0].api_key_encrypted =
            Some(secrets::encrypt_provider_secret("sk-round-trip-stored-dummy").unwrap());
        initial.providers[0].resolved_api_key = Some("sk-round-trip-stored-dummy".to_string());
        let server = crate::gateway::GatewayServer::new(initial, Some(config_path.clone()))
            .await
            .unwrap();

        let get = || {
            axum::http::Request::builder()
                .method("GET")
                .uri("/admin/config")
                .body(axum::body::Body::empty())
                .unwrap()
        };
        let response = tower::ServiceExt::oneshot(server.build_router(), get())
            .await
            .unwrap();
        assert_eq!(response.status(), StatusCode::OK);
        let config = smart_routing_response_json(response).await;
        // GET reports only the status flags; the key field is blank.
        assert!(config["providers"][0]["api_key_env"].is_null());
        assert_eq!(config["providers"][0]["api_key_status"], "encrypted");
        assert_eq!(config["providers"][0]["api_key_configured"], true);

        // PUT the body back exactly as the frontend would (blank key field).
        let response = tower::ServiceExt::oneshot(
            server.build_router(),
            smart_routing_json_request("PUT", "/admin/config", config),
        )
        .await
        .unwrap();
        let status = response.status();
        let body = smart_routing_response_json(response).await;
        assert_eq!(status, StatusCode::OK, "PUT /admin/config failed: {body}");

        // Stored encrypted key survives the round trip in the live config...
        let live = server.state.config.read().await;
        assert!(
            live.providers[0]
                .api_key_encrypted
                .as_deref()
                .is_some_and(secrets::is_encrypted_secret),
            "live config lost the stored encrypted key"
        );
        assert_eq!(
            live.providers[0].resolved_api_key.as_deref(),
            Some("sk-round-trip-stored-dummy"),
            "live config lost the resolved key"
        );
        drop(live);
        // ...and in the persisted YAML (resolved_api_key is serde-skipped by
        // design; only the encrypted form is written).
        let saved: Config =
            serde_yaml::from_str(&std::fs::read_to_string(&config_path).unwrap()).unwrap();
        assert!(
            saved.providers[0]
                .api_key_encrypted
                .as_deref()
                .is_some_and(secrets::is_encrypted_secret),
            "saved YAML lost the stored encrypted key"
        );
        assert_eq!(
            secrets::decrypt_provider_secret(
                saved.providers[0].api_key_encrypted.as_deref().unwrap()
            )
            .unwrap(),
            "sk-round-trip-stored-dummy"
        );

        let _ = std::fs::remove_dir_all(&config_dir);
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
        let storage_dir =
            std::env::temp_dir().join(format!("obey-admin-test-{}", uuid::Uuid::new_v4().simple()));
        std::fs::create_dir_all(&storage_dir).unwrap();
        let mut config = Config {
            security: Default::default(),
            cache_aware_routing: Default::default(),
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
                    allow_unauthenticated_non_loopback: false,
                },
                security_mode: None,
                auth_section_present: false,
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
                buffered_upstream_streaming: None,
                rate_limit_max_wait_ms: None,
                max_connections: 10,
                rate_limit_per_minute: 0,
                custom_headers: Default::default(),
                user_agent: None,
                connection_pool: ProviderConnectionPoolConfig::default(),
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
                name: "test-group".to_string(),
                version_fallback_enabled: false,
                compression: None,
                structured_output: None,
                memory: None,
                models: vec![ProviderModel {
                    cache_support: None,
                    cache_min_tokens: None,
                    cost_per_million_cache_read_input_tokens: None,
                    cost_per_million_cache_creation_input_tokens: None,
                    provider: "test".to_string(),
                    model: "gpt-4".to_string(),
                    cost_per_million_input_tokens: 0.0,
                    cost_per_million_output_tokens: 0.0,
                    priority: 100,
                    structured_output_passthrough: None,
                    tier: None,
                    context_window: 0,
                    specializations: vec![],
                    cost_per_million_reasoning_tokens: None,
                    reasoning_family: None,
                    reasoning_parameter: None,
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
            codex_search: None,
            reasoning_compat: Default::default(),
        };
        config.logging.database_path = storage_dir.join("logs.db").to_string_lossy().into_owned();
        config.virtual_keys.database_path =
            storage_dir.join("keys.db").to_string_lossy().into_owned();
        config
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
    async fn onnx_status_reports_configured_missing_assets_and_uses_admin_auth() {
        let user_env = "ONNX_ADMIN_TEST_USER";
        let pass_env = "ONNX_ADMIN_TEST_PASS";
        std::env::set_var(user_env, "admin");
        std::env::set_var(pass_env, "secret");
        let mut config = test_config_with_auth_env(true, user_env, pass_env);
        let directory = tempfile::tempdir().unwrap();
        config.compression.perplexity.model_path =
            directory.path().join("missing.onnx").display().to_string();
        let server = crate::gateway::GatewayServer::new(config, None)
            .await
            .unwrap();
        let app = server.build_router();

        let unauthorized = axum::http::Request::get("/admin/onnx/status")
            .body(axum::body::Body::empty())
            .unwrap();
        let response = tower::ServiceExt::oneshot(app.clone(), unauthorized)
            .await
            .unwrap();
        assert_eq!(response.status(), StatusCode::UNAUTHORIZED);

        let authorized = axum::http::Request::get("/admin/onnx/status")
            .header("Authorization", basic_auth_header("admin", "secret"))
            .body(axum::body::Body::empty())
            .unwrap();
        let response = tower::ServiceExt::oneshot(app, authorized).await.unwrap();
        assert_eq!(response.status(), StatusCode::OK);
        let body = to_bytes(response.into_body(), 64 * 1024).await.unwrap();
        let body: serde_json::Value = serde_json::from_slice(&body).unwrap();
        assert_eq!(body["ready"], false);
        assert!(body["missing_assets"]
            .as_array()
            .is_some_and(|assets| assets.iter().any(|asset| asset == "model")));
        std::env::remove_var(user_env);
        std::env::remove_var(pass_env);
    }

    #[tokio::test]
    async fn onnx_status_rejects_empty_override() {
        let config = test_config_with_auth(false);
        let server = crate::gateway::GatewayServer::new(config, None)
            .await
            .unwrap();
        let app = server.build_router();
        let request = axum::http::Request::get("/admin/onnx/status?model_path=")
            .body(axum::body::Body::empty())
            .unwrap();
        let response = tower::ServiceExt::oneshot(app, request).await.unwrap();
        assert_eq!(response.status(), StatusCode::BAD_REQUEST);
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
    async fn memory_admin_routes_share_parent_authentication() {
        let user_env = "MEMORY_ADMIN_TEST_USER";
        let pass_env = "MEMORY_ADMIN_TEST_PASS";
        std::env::set_var(user_env, "admin");
        std::env::set_var(pass_env, "secret");
        let mut config = test_config_with_auth_env(true, user_env, pass_env);
        config.memory = Some(crate::memory::MemoryConfig {
            enabled: true,
            database_path: ":memory:".to_owned(),
            ..Default::default()
        });
        let server = crate::gateway::GatewayServer::new(config, None)
            .await
            .unwrap();
        let app = server.build_router();

        let unauthorized = axum::http::Request::get("/admin/memory/stats")
            .body(axum::body::Body::empty())
            .unwrap();
        let response = tower::ServiceExt::oneshot(app.clone(), unauthorized)
            .await
            .unwrap();
        assert_eq!(response.status(), StatusCode::UNAUTHORIZED);

        let authorized = axum::http::Request::get("/admin/memory/stats")
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
        // Req 35.5, 35.7: auth disabled ÃƒÆ’Ã‚Â¢ÃƒÂ¢Ã¢â€šÂ¬Ã‚Â ÃƒÂ¢Ã¢â€šÂ¬Ã¢â€žÂ¢ unrestricted access
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
        // Req 35.1, 35.4: auth enabled, no credentials ÃƒÆ’Ã‚Â¢ÃƒÂ¢Ã¢â€šÂ¬Ã‚Â ÃƒÂ¢Ã¢â€šÂ¬Ã¢â€žÂ¢ 401
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
        // Req 35.1, 35.3: auth enabled, valid Basic credentials ÃƒÆ’Ã‚Â¢ÃƒÂ¢Ã¢â€šÂ¬Ã‚Â ÃƒÂ¢Ã¢â€šÂ¬Ã¢â€žÂ¢ pass through
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
        // Req 35.4: invalid credentials ÃƒÆ’Ã‚Â¢ÃƒÂ¢Ã¢â€šÂ¬Ã‚Â ÃƒÂ¢Ã¢â€šÂ¬Ã¢â€žÂ¢ 401
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

    // Spec: black-october-security-hardening, Req 1 middleware enforcement.
    // Legacy config (no auth section, no security_mode) on a non-loopback bind
    // resolves to OpenWarnedLegacy → serve unauthenticated, no 401.
    #[tokio::test]
    async fn test_legacy_config_non_loopback_serves_admin_unauthenticated() {
        let mut cfg = test_config_with_auth(false);
        cfg.server.host = "192.168.1.110".to_string();
        cfg.admin.security_mode = None;
        cfg.admin.auth_section_present = false;
        assert_eq!(
            auth_policy::resolve_admin_security_mode(&cfg),
            AdminSecurityMode::OpenWarnedLegacy
        );

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

    // Spec: black-october-security-hardening, Req 1 middleware enforcement.
    // Explicit new config (security_mode: Some(Authenticated), auth enabled,
    // env credentials set) → 401 without credentials, 200 with them.
    #[tokio::test]
    async fn test_explicit_authenticated_mode_enforces_credentials() {
        let user_env = "AUTH_TEST_EXPLICIT_MODE_USER";
        let pass_env = "AUTH_TEST_EXPLICIT_MODE_PASS";
        std::env::set_var(user_env, "admin");
        std::env::set_var(pass_env, "secret");

        let mut cfg = test_config_with_auth_env(true, user_env, pass_env);
        cfg.server.host = "192.168.1.110".to_string();
        cfg.admin.security_mode = Some(AdminSecurityMode::Authenticated);
        cfg.admin.auth_section_present = true;
        assert_eq!(
            auth_policy::resolve_admin_security_mode(&cfg),
            AdminSecurityMode::Authenticated
        );

        let server = crate::gateway::GatewayServer::new(cfg, None).await.unwrap();
        let app = server.build_router();

        let unauthorized = axum::http::Request::builder()
            .method("GET")
            .uri("/admin/config")
            .body(axum::body::Body::empty())
            .unwrap();
        let resp = tower::ServiceExt::oneshot(app.clone(), unauthorized)
            .await
            .unwrap();
        assert_eq!(resp.status(), StatusCode::UNAUTHORIZED);
        assert!(resp.headers().get("www-authenticate").is_some());

        let authorized = axum::http::Request::builder()
            .method("GET")
            .uri("/admin/config")
            .header("Authorization", basic_auth_header("admin", "secret"))
            .body(axum::body::Body::empty())
            .unwrap();
        let resp = tower::ServiceExt::oneshot(app, authorized).await.unwrap();
        assert_eq!(resp.status(), StatusCode::OK);

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

                // 1) No credentials ÃƒÆ’Ã‚Â¢ÃƒÂ¢Ã¢â€šÂ¬Ã‚Â ÃƒÂ¢Ã¢â€šÂ¬Ã¢â€žÂ¢ 401 (Req 35.1, 35.4)
                let req = axum::http::Request::builder()
                    .method("GET")
                    .uri("/admin/config")
                    .body(axum::body::Body::empty())
                    .unwrap();
                let resp = tower::ServiceExt::oneshot(app, req).await.unwrap();
                assert_eq!(resp.status(), StatusCode::UNAUTHORIZED,
                    "Auth enabled + no creds must be 401");

                // 2) Valid credentials ÃƒÆ’Ã‚Â¢ÃƒÂ¢Ã¢â€šÂ¬Ã‚Â ÃƒÂ¢Ã¢â€šÂ¬Ã¢â€žÂ¢ 200 (Req 35.1, 35.3)
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

                // 3) Invalid credentials ÃƒÆ’Ã‚Â¢ÃƒÂ¢Ã¢â€šÂ¬Ã‚Â ÃƒÂ¢Ã¢â€šÂ¬Ã¢â€žÂ¢ 401 (Req 35.4)
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

                // --- Auth DISABLED ÃƒÆ’Ã‚Â¢ÃƒÂ¢Ã¢â€šÂ¬Ã‚Â ÃƒÂ¢Ã¢â€šÂ¬Ã¢â€žÂ¢ pass through (Req 35.5, 35.7) ---
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
        // Req 35.4: invalid credentials ÃƒÆ’Ã‚Â¢ÃƒÂ¢Ã¢â€šÂ¬Ã‚Â ÃƒÂ¢Ã¢â€šÂ¬Ã¢â€žÂ¢ 401
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
    // Property 7 ÃƒÆ’Ã‚Â¢ÃƒÂ¢Ã¢â‚¬Å¡Ã‚Â¬ÃƒÂ¢Ã¢â€šÂ¬Ã‚Â No Tokens in Status Response (Req 7.2, 8.4)
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
        #![proptest_config(ProptestConfig::with_cases(64))]

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
    // Task 14.7 ÃƒÆ’Ã‚Â¢ÃƒÂ¢Ã¢â‚¬Å¡Ã‚Â¬ÃƒÂ¢Ã¢â€šÂ¬Ã‚Â Integration tests for admin OAuth endpoints (Req 7.1, 7.3)
    // -----------------------------------------------------------------------

    /// Helper: build a GatewayServer with OAuth enabled (using a temp dir for
    /// the token store so tests don't touch real storage).
    async fn build_oauth_test_app() -> Router {
        let cfg = test_config_with_auth(false);
        let server = crate::gateway::GatewayServer::new(cfg, None).await.unwrap();
        server.build_router()
    }

    /// POST /admin/oauth/openai/login ÃƒÆ’Ã‚Â¢ÃƒÂ¢Ã¢â‚¬Å¡Ã‚Â¬ÃƒÂ¢Ã¢â€šÂ¬Ã‚Â in a headless test environment the
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

    /// GET /admin/oauth/openai/status ÃƒÆ’Ã‚Â¢ÃƒÂ¢Ã¢â‚¬Å¡Ã‚Â¬ÃƒÂ¢Ã¢â€šÂ¬Ã‚Â verify response shape has `state`,
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

        // Fresh server with no prior login ÃƒÆ’Ã‚Â¢ÃƒÂ¢Ã¢â€šÂ¬Ã‚Â ÃƒÂ¢Ã¢â€šÂ¬Ã¢â€žÂ¢ unauthenticated
        assert_eq!(json["state"], "unauthenticated");
        assert!(json["expires_at"].is_null());
        assert!(json["scopes"].is_null());
    }

    /// POST /admin/oauth/openai/logout ÃƒÆ’Ã‚Â¢ÃƒÂ¢Ã¢â‚¬Å¡Ã‚Â¬ÃƒÂ¢Ã¢â€šÂ¬Ã‚Â verify returns `{ "status": "logged_out" }`;
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
    // Task 16.3 ÃƒÆ’Ã‚Â¢ÃƒÂ¢Ã¢â‚¬Å¡Ã‚Â¬ÃƒÂ¢Ã¢â€šÂ¬Ã‚Â Admin-auth protection test (Req 8.5)
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

        // --- With valid credentials: auth must allow all OAuth endpoints ---

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
            assert_ne!(
                resp.status(),
                StatusCode::UNAUTHORIZED,
                "{} {} with valid credentials must pass admin auth, got {}",
                method,
                uri,
                resp.status()
            );
        }

        // Cleanup
        std::env::remove_var(user_env);
        std::env::remove_var(pass_env);
    }

    /// Admin-panel parity: `buffered_upstream_streaming` saved through
    /// PUT /admin/config persists to YAML and comes back from GET.
    #[tokio::test]
    async fn admin_config_round_trips_buffered_upstream_streaming() {
        let config_dir = std::env::temp_dir().join(format!(
            "obey-admin-buffered-stream-{}",
            uuid::Uuid::new_v4().simple()
        ));
        std::fs::create_dir_all(&config_dir).unwrap();
        let config_path = config_dir.join("config.yaml");
        let mut initial = test_config_with_auth(false);
        // PUT validates the full config; the shared fixture's port 0 does not pass.
        initial.server.port = 8080;
        let server = crate::gateway::GatewayServer::new(initial, Some(config_path.clone()))
            .await
            .unwrap();

        let get = || {
            axum::http::Request::builder()
                .method("GET")
                .uri("/admin/config")
                .body(axum::body::Body::empty())
                .unwrap()
        };
        let response = tower::ServiceExt::oneshot(server.build_router(), get())
            .await
            .unwrap();
        assert_eq!(response.status(), StatusCode::OK);
        let mut config = smart_routing_response_json(response).await;
        // Unset by default: the effective value comes from the provider type.
        assert!(config["providers"][0]["buffered_upstream_streaming"].is_null());

        config["providers"][0]["buffered_upstream_streaming"] = json!(false);
        let response = tower::ServiceExt::oneshot(
            server.build_router(),
            smart_routing_json_request("PUT", "/admin/config", config),
        )
        .await
        .unwrap();
        let status = response.status();
        let body = smart_routing_response_json(response).await;
        assert_eq!(status, StatusCode::OK, "PUT /admin/config failed: {body}");

        let saved: Config =
            serde_yaml::from_str(&std::fs::read_to_string(&config_path).unwrap()).unwrap();
        assert_eq!(saved.providers[0].buffered_upstream_streaming, Some(false));

        let response = tower::ServiceExt::oneshot(server.build_router(), get())
            .await
            .unwrap();
        let config = smart_routing_response_json(response).await;
        assert_eq!(
            config["providers"][0]["buffered_upstream_streaming"],
            json!(false)
        );

        let _ = std::fs::remove_dir_all(&config_dir);
    }

    /// Admin-panel parity (ADR 0002): `rate_limit_max_wait_ms` saved through
    /// PUT /admin/config persists to YAML and comes back from GET; values
    /// above the 60000 ms cap are rejected.
    #[tokio::test]
    async fn admin_config_round_trips_rate_limit_max_wait_ms() {
        let config_dir = std::env::temp_dir().join(format!(
            "obey-admin-rate-wait-{}",
            uuid::Uuid::new_v4().simple()
        ));
        std::fs::create_dir_all(&config_dir).unwrap();
        let config_path = config_dir.join("config.yaml");
        let mut initial = test_config_with_auth(false);
        initial.server.port = 8080;
        let server = crate::gateway::GatewayServer::new(initial, Some(config_path.clone()))
            .await
            .unwrap();

        let get = || {
            axum::http::Request::builder()
                .method("GET")
                .uri("/admin/config")
                .body(axum::body::Body::empty())
                .unwrap()
        };
        let response = tower::ServiceExt::oneshot(server.build_router(), get())
            .await
            .unwrap();
        let mut config = smart_routing_response_json(response).await;
        assert!(config["providers"][0]["rate_limit_max_wait_ms"].is_null());

        config["providers"][0]["rate_limit_max_wait_ms"] = json!(60001);
        let response = tower::ServiceExt::oneshot(
            server.build_router(),
            smart_routing_json_request("PUT", "/admin/config", config.clone()),
        )
        .await
        .unwrap();
        assert!(
            response.status().is_client_error(),
            "out-of-range wait must be rejected, got {}",
            response.status()
        );

        config["providers"][0]["rate_limit_max_wait_ms"] = json!(7500);
        let response = tower::ServiceExt::oneshot(
            server.build_router(),
            smart_routing_json_request("PUT", "/admin/config", config),
        )
        .await
        .unwrap();
        let status = response.status();
        let body = smart_routing_response_json(response).await;
        assert_eq!(status, StatusCode::OK, "PUT /admin/config failed: {body}");

        let saved: Config =
            serde_yaml::from_str(&std::fs::read_to_string(&config_path).unwrap()).unwrap();
        assert_eq!(saved.providers[0].rate_limit_max_wait_ms, Some(7500));

        let response = tower::ServiceExt::oneshot(server.build_router(), get())
            .await
            .unwrap();
        let config = smart_routing_response_json(response).await;
        assert_eq!(
            config["providers"][0]["rate_limit_max_wait_ms"],
            json!(7500)
        );

        let _ = std::fs::remove_dir_all(&config_dir);
    }

    fn smart_routing_json_request(
        method: &str,
        uri: &str,
        body: serde_json::Value,
    ) -> axum::http::Request<axum::body::Body> {
        axum::http::Request::builder()
            .method(method)
            .uri(uri)
            .header(header::CONTENT_TYPE, "application/json")
            .body(axum::body::Body::from(body.to_string()))
            .unwrap()
    }

    async fn smart_routing_response_json(response: Response) -> serde_json::Value {
        let body = to_bytes(response.into_body(), 64 * 1024).await.unwrap();
        serde_json::from_slice(&body).unwrap()
    }

    #[tokio::test]
    async fn smart_routing_all_endpoint_methods_share_admin_auth() {
        let user_env = "SMART_ROUTING_ADMIN_TEST_USER";
        let pass_env = "SMART_ROUTING_ADMIN_TEST_PASS";
        std::env::set_var(user_env, "smart-admin");
        std::env::set_var(pass_env, "smart-secret");

        let config = test_config_with_auth_env(true, user_env, pass_env);
        let server = crate::gateway::GatewayServer::new(config, None)
            .await
            .unwrap();
        let app = server.build_router();
        let endpoints = [
            ("GET", "/admin/smart-routing/config", None),
            ("PUT", "/admin/smart-routing/config", Some(json!({}))),
            (
                "POST",
                "/admin/smart-routing/simulate",
                Some(json!({"model_group":"test-group","messages":[]})),
            ),
            ("GET", "/admin/smart-routing/optimizer-state", None),
            ("GET", "/admin/smart-routing/budget", None),
            ("GET", "/admin/smart-routing/model/status", None),
            ("POST", "/admin/smart-routing/model/download", None),
            ("POST", "/admin/smart-routing/model/train", None),
            ("GET", "/admin/smart-routing/model/train/status", None),
            ("GET", "/admin/smart-routing/ab-test", None),
            ("PUT", "/admin/smart-routing/ab-test", Some(json!({}))),
            ("DELETE", "/admin/smart-routing/ab-test", None),
            ("GET", "/admin/smart-routing/ab-test/results", None),
        ];

        for (method, uri, body) in endpoints {
            let request = match body.clone() {
                Some(body) => smart_routing_json_request(method, uri, body),
                None => axum::http::Request::builder()
                    .method(method)
                    .uri(uri)
                    .body(axum::body::Body::empty())
                    .unwrap(),
            };
            let response = tower::ServiceExt::oneshot(app.clone(), request)
                .await
                .unwrap();
            assert_eq!(
                response.status(),
                StatusCode::UNAUTHORIZED,
                "{method} {uri} must require admin authentication"
            );
        }

        std::env::remove_var(user_env);
        std::env::remove_var(pass_env);
    }

    #[tokio::test]
    async fn smart_routing_config_endpoint_rejects_non_finite_and_out_of_range_values() {
        let server = crate::gateway::GatewayServer::new(test_config_with_auth(false), None)
            .await
            .unwrap();

        for (field, invalid) in [
            ("cost_quality_threshold", f64::NAN),
            ("cost_quality_threshold", f64::INFINITY),
            ("online_optimizer.alpha", -0.01),
            ("semantic_cache.similarity_threshold", 1.01),
        ] {
            let mut smart_routing = SmartRoutingConfig::default();
            match field {
                "cost_quality_threshold" => smart_routing.cost_quality_threshold = invalid,
                "online_optimizer.alpha" => smart_routing.online_optimizer.alpha = invalid,
                "semantic_cache.similarity_threshold" => {
                    smart_routing.semantic_cache.similarity_threshold = invalid
                }
                _ => unreachable!(),
            }
            let response =
                update_smart_routing_config(State(server.state.clone()), Json(smart_routing)).await;
            assert_eq!(response.status(), StatusCode::BAD_REQUEST, "{field}");
            let body = smart_routing_response_json(response).await;
            assert_eq!(body["error"]["type"], "validation_error");
            assert!(
                body["error"]["details"]
                    .as_array()
                    .is_some_and(|details| details.iter().any(|detail| detail
                        .as_str()
                        .is_some_and(|detail| detail.contains(field)))),
                "validation response should identify {field}: {body}"
            );
        }
    }

    #[tokio::test]
    async fn smart_routing_simulation_is_heuristic_bounded_and_never_generates() {
        let provider = wiremock::MockServer::start().await;
        let mut config = test_config_with_auth(false);
        config.providers[0].base_url = Some(provider.uri());
        config.smart_routing.enabled = true;
        config.smart_routing.allow_unknown_context_window = true;
        let server = crate::gateway::GatewayServer::new(config, None)
            .await
            .unwrap();
        let app = server.build_router();
        let prompt_marker = "SIMULATION_PROMPT_MUST_NOT_LEAK_OR_GENERATE";
        let request = smart_routing_json_request(
            "POST",
            "/admin/smart-routing/simulate",
            json!({
            "model_group": "test-group",
            "messages": [{"role":"user","content":prompt_marker}],
            "max_tokens": 32
            }),
        );

        let response = tower::ServiceExt::oneshot(app, request).await.unwrap();
        assert_eq!(response.status(), StatusCode::OK);
        let body = to_bytes(response.into_body(), 16 * 1024).await.unwrap();
        assert!(
            body.len() < 8 * 1024,
            "simulation response must remain bounded"
        );
        let body_text = std::str::from_utf8(&body).unwrap();
        assert!(body_text.contains("Heuristic"));
        assert!(!body_text.contains(prompt_marker));
        assert!(provider.received_requests().await.unwrap().is_empty());
    }

    #[tokio::test]
    async fn smart_routing_json_extractors_reject_malformed_and_oversized_bodies() {
        let mut config = test_config_with_auth(false);
        config.server.max_request_size_mb = 1;
        let server = crate::gateway::GatewayServer::new(config, None)
            .await
            .unwrap();
        let app = server.build_router();

        let malformed = axum::http::Request::post("/admin/smart-routing/simulate")
            .header(header::CONTENT_TYPE, "application/json")
            .body(axum::body::Body::from("{"))
            .unwrap();
        let response = tower::ServiceExt::oneshot(app.clone(), malformed)
            .await
            .unwrap();
        assert_eq!(response.status(), StatusCode::BAD_REQUEST);

        let oversized = axum::http::Request::put("/admin/smart-routing/config")
            .header(header::CONTENT_TYPE, "application/json")
            .body(axum::body::Body::from(" ".repeat(1024 * 1024 + 1)))
            .unwrap();
        let response = tower::ServiceExt::oneshot(app, oversized).await.unwrap();
        assert_eq!(response.status(), StatusCode::PAYLOAD_TOO_LARGE);
    }

    #[tokio::test]
    async fn smart_routing_duplicate_ab_put_replaces_the_active_experiment() {
        *smart_routing_ab_test()
            .write()
            .unwrap_or_else(std::sync::PoisonError::into_inner) = None;
        let server = crate::gateway::GatewayServer::new(test_config_with_auth(false), None)
            .await
            .unwrap();
        let app = server.build_router();

        for percentage in [0.2, 0.8] {
            let request = smart_routing_json_request(
                "PUT",
                "/admin/smart-routing/ab-test",
                json!({"variant_percentage":percentage}),
            );
            let response = tower::ServiceExt::oneshot(app.clone(), request)
                .await
                .unwrap();
            assert_eq!(response.status(), StatusCode::OK);
        }

        let request = axum::http::Request::get("/admin/smart-routing/ab-test/results")
            .body(axum::body::Body::empty())
            .unwrap();
        let response = tower::ServiceExt::oneshot(app, request).await.unwrap();
        assert_eq!(response.status(), StatusCode::OK);
        let body = smart_routing_response_json(response).await;
        assert_eq!(body["active"], true);
        assert_eq!(body["variant_percentage"], 0.8);
        assert_eq!(body["control"]["n"], 0);
        assert_eq!(body["variant"]["n"], 0);

        *smart_routing_ab_test()
            .write()
            .unwrap_or_else(std::sync::PoisonError::into_inner) = None;
    }

    #[tokio::test]
    async fn smart_routing_unavailable_model_and_training_responses_are_redacted() {
        let path_secret = "C:\\private\\tenant-secret\\model.onnx";
        let dataset_secret = "/home/private/tenant-secret/training.jsonl";
        let api_secret = "sk-super-secret-admin-test";
        let mut config = test_config_with_auth(false);
        config.smart_routing.ml_model_path = Some(path_secret.to_string());
        config.smart_routing.training.dataset_path = Some(dataset_secret.to_string());
        config.providers[0].resolved_api_key = Some(api_secret.to_string());
        let server = crate::gateway::GatewayServer::new(config, None)
            .await
            .unwrap();
        let app = server.build_router();
        let endpoints = [
            ("GET", "/admin/smart-routing/model/status", StatusCode::OK),
            (
                "POST",
                "/admin/smart-routing/model/download",
                StatusCode::UNSUPPORTED_MEDIA_TYPE,
            ),
            (
                "POST",
                "/admin/smart-routing/model/train",
                StatusCode::NOT_IMPLEMENTED,
            ),
            (
                "GET",
                "/admin/smart-routing/model/train/status",
                StatusCode::OK,
            ),
        ];

        for (method, uri, expected_status) in endpoints {
            let request = axum::http::Request::builder()
                .method(method)
                .uri(uri)
                .body(axum::body::Body::empty())
                .unwrap();
            let response = tower::ServiceExt::oneshot(app.clone(), request)
                .await
                .unwrap();
            assert_eq!(response.status(), expected_status, "{method} {uri}");
            let body = to_bytes(response.into_body(), 8 * 1024).await.unwrap();
            assert!(
                body.len() < 4 * 1024,
                "{method} {uri} response is unbounded"
            );
            let body = std::str::from_utf8(&body).unwrap();
            for secret in [path_secret, dataset_secret, api_secret, "tenant-secret"] {
                assert!(
                    !body.contains(secret),
                    "{method} {uri} leaked sensitive value {secret}: {body}"
                );
            }
            assert!(!body.contains("\\private\\"));
            assert!(!body.contains("/home/private/"));
        }
    }

    // --- GET /admin/security-status (spec: black-october-security-hardening, Req 1.7) ---

    #[tokio::test]
    async fn security_status_requires_auth_in_authenticated_mode() {
        let user_env = "SEC_STATUS_TEST_USER";
        let pass_env = "SEC_STATUS_TEST_PASS";
        std::env::set_var(user_env, "admin");
        std::env::set_var(pass_env, "secret");
        let config = test_config_with_auth_env(true, user_env, pass_env);
        let server = crate::gateway::GatewayServer::new(config, None)
            .await
            .unwrap();
        let app = server.build_router();

        let request = axum::http::Request::get("/admin/security-status")
            .body(axum::body::Body::empty())
            .unwrap();
        let response = tower::ServiceExt::oneshot(app.clone(), request)
            .await
            .unwrap();
        assert_eq!(response.status(), StatusCode::UNAUTHORIZED);
        assert_eq!(
            response
                .headers()
                .get("WWW-Authenticate")
                .and_then(|v| v.to_str().ok()),
            Some("Basic realm=\"OBEY-API Admin\"")
        );

        // Valid credentials reach the endpoint; shape + no credential material.
        let authorized = axum::http::Request::get("/admin/security-status")
            .header("Authorization", basic_auth_header("admin", "secret"))
            .body(axum::body::Body::empty())
            .unwrap();
        let response = tower::ServiceExt::oneshot(app, authorized).await.unwrap();
        assert_eq!(response.status(), StatusCode::OK);
        let body = to_bytes(response.into_body(), 16 * 1024).await.unwrap();
        let json: serde_json::Value = serde_json::from_slice(&body).unwrap();
        assert_eq!(json["mode"], "authenticated");
        assert_eq!(json["unauthenticated_non_loopback"], false);
        assert_eq!(json.as_object().map(|o| o.len()), Some(2), "exact shape");
        let body_str = std::str::from_utf8(&body).unwrap();
        assert!(
            !body_str.contains(user_env) && !body_str.contains(pass_env),
            "must not leak credential env names"
        );
        std::env::remove_var(user_env);
        std::env::remove_var(pass_env);
    }

    #[tokio::test]
    async fn security_status_open_warned_legacy_on_non_loopback_bind() {
        // Legacy config (no security_mode, no auth section) on a LAN bind.
        let mut config = test_config_with_auth(false);
        config.server.host = "192.168.1.110".to_string();
        config.admin.security_mode = None;
        config.admin.auth_section_present = false;
        let server = crate::gateway::GatewayServer::new(config, None)
            .await
            .unwrap();
        let app = server.build_router();

        let request = axum::http::Request::get("/admin/security-status")
            .body(axum::body::Body::empty())
            .unwrap();
        let response = tower::ServiceExt::oneshot(app, request).await.unwrap();
        assert_eq!(response.status(), StatusCode::OK);
        let body = to_bytes(response.into_body(), 16 * 1024).await.unwrap();
        let json: serde_json::Value = serde_json::from_slice(&body).unwrap();
        assert_eq!(json["mode"], "open-warned-legacy");
        assert_eq!(json["unauthenticated_non_loopback"], true);
        assert_eq!(json.as_object().map(|o| o.len()), Some(2), "exact shape");
    }

    #[tokio::test]
    async fn security_status_loopback_open_reports_no_exposure() {
        // Legacy config on a loopback bind → LoopbackOpen, not exposed.
        let mut config = test_config_with_auth(false);
        config.server.host = "127.0.0.1".to_string();
        config.admin.security_mode = None;
        config.admin.auth_section_present = false;
        let server = crate::gateway::GatewayServer::new(config, None)
            .await
            .unwrap();
        let app = server.build_router();

        let request = axum::http::Request::get("/admin/security-status")
            .body(axum::body::Body::empty())
            .unwrap();
        let response = tower::ServiceExt::oneshot(app, request).await.unwrap();
        assert_eq!(response.status(), StatusCode::OK);
        let body = to_bytes(response.into_body(), 16 * 1024).await.unwrap();
        let json: serde_json::Value = serde_json::from_slice(&body).unwrap();
        assert_eq!(json["mode"], "loopback-open");
        assert_eq!(json["unauthenticated_non_loopback"], false);
        assert_eq!(json.as_object().map(|o| o.len()), Some(2), "exact shape");
    }

    fn outbound_fetch_test_config(allow_private_hosts: bool) -> Config {
        let mut config = test_config_with_auth(false);
        config.security.outbound_fetch.caller.allow_private_hosts = allow_private_hosts;
        config.security.outbound_fetch.caller.allow_http = true; // wiremock binds 127.0.0.1 over http
        config
    }

    #[tokio::test]
    async fn proxy_provider_models_rejects_api_key_in_query_params() {
        // A wiremock URI keeps the reject-ordering assertion deterministic — no
        // real network I/O even if the credential-query defense regresses.
        let upstream = wiremock::MockServer::start().await;
        let config = outbound_fetch_test_config(true);
        let server = crate::gateway::GatewayServer::new(config, None)
            .await
            .unwrap();
        let app = server.build_router();

        // Credential-in-URL defense (Req 3.1): any api_key query param on the
        // route must 400 before outbound work, regardless of body content.
        let request = smart_routing_json_request(
            "POST",
            &format!(
                "/admin/providers/models?base_url={}&api_key=sk-secret",
                upstream.uri()
            ),
            json!({ "base_url": upstream.uri() }),
        );
        let response = tower::ServiceExt::oneshot(app.clone(), request)
            .await
            .unwrap();
        assert_eq!(response.status(), StatusCode::BAD_REQUEST);
        let body = to_bytes(response.into_body(), 16 * 1024).await.unwrap();
        let body_text = String::from_utf8_lossy(&body).into_owned();
        assert!(!body_text.contains("sk-secret"));
        assert!(
            upstream.received_requests().await.unwrap().is_empty(),
            "reject must happen before any outbound work"
        );

        // api_key_env spelling is rejected identically.
        let request = smart_routing_json_request(
            "POST",
            "/admin/providers/models?api_key_env=OPENAI_API_KEY",
            json!({ "base_url": upstream.uri() }),
        );
        let response = tower::ServiceExt::oneshot(app, request).await.unwrap();
        assert_eq!(response.status(), StatusCode::BAD_REQUEST);
    }

    #[tokio::test]
    async fn proxy_provider_models_post_body_proxies_models_list() {
        let upstream = wiremock::MockServer::start().await;
        wiremock::Mock::given(wiremock::matchers::method("GET"))
            .and(wiremock::matchers::path("/v1/models"))
            .and(wiremock::matchers::header_exists("authorization"))
            .respond_with(wiremock::ResponseTemplate::new(200).set_body_json(json!({
                "object": "list",
                "data": [{ "id": "gpt-4.1-nano" }]
            })))
            .mount(&upstream)
            .await;

        let mut config = outbound_fetch_test_config(true);
        // Provider configured for the mock host; body key matches that host
        // so the credential is forwarded.
        config.providers[0].base_url = Some(upstream.uri());
        config.providers[0].resolved_api_key = Some("sk-provider-key".to_string());
        let server = crate::gateway::GatewayServer::new(config, None)
            .await
            .unwrap();
        let app = server.build_router();

        let request = smart_routing_json_request(
            "POST",
            "/admin/providers/models",
            json!({
            "base_url": upstream.uri(),
            "api_key": "sk-body-key",
            "provider_name": "test",
            "provider_type": "openai"
            }),
        );
        let response = tower::ServiceExt::oneshot(app, request).await.unwrap();
        assert_eq!(response.status(), StatusCode::OK);
        let body = to_bytes(response.into_body(), 16 * 1024).await.unwrap();
        let json: serde_json::Value = serde_json::from_slice(&body).unwrap();
        assert_eq!(json["object"], "list");
        assert_eq!(json["data"][0]["id"], "gpt-4.1-nano");
        // Body key was forwarded to the provider's own host.
        let requests = upstream.received_requests().await.unwrap();
        assert!(requests.iter().any(|r| r
            .headers
            .get("authorization")
            .map(|v| v.to_str().unwrap())
            == Some("Bearer sk-body-key")));
    }

    #[tokio::test]
    async fn proxy_provider_models_refuses_credentials_cross_host() {
        let host_a = wiremock::MockServer::start().await;
        let host_b = wiremock::MockServer::start().await;
        wiremock::Mock::given(wiremock::matchers::method("GET"))
            .and(wiremock::matchers::path("/v1/models"))
            .respond_with(wiremock::ResponseTemplate::new(200).set_body_json(json!({
                "object": "list",
                "data": []
            })))
            .mount(&host_b)
            .await;

        let mut config = outbound_fetch_test_config(true);
        // Provider configured for host A...
        config.providers[0].base_url = Some(host_a.uri());
        config.providers[0].resolved_api_key = Some("sk-never-leak".to_string());
        // Custom headers may carry ${ENV} secrets — cross-host targets must not
        // receive them either.
        config.providers[0].custom_headers.insert(
            "X-Custom-Secret".to_string(),
            "sk-custom-header-secret".to_string(),
        );
        let server = crate::gateway::GatewayServer::new(config, None)
            .await
            .unwrap();
        let app = server.build_router();

        // ...but the request targets host B: no credential header may reach B,
        // neither the provider-resolved key nor a body-supplied one.
        let request = smart_routing_json_request(
            "POST",
            "/admin/providers/models",
            json!({
            "base_url": host_b.uri(),
            "api_key": "sk-body-cross-host",
            "provider_name": "test",
            "provider_type": "openai"
            }),
        );
        let response = tower::ServiceExt::oneshot(app, request).await.unwrap();
        assert_eq!(response.status(), StatusCode::OK);
        let requests = host_b.received_requests().await.unwrap();
        assert!(!requests.is_empty(), "request reached host B");
        assert!(
            requests
                .iter()
                .all(|r| !r.headers.contains_key("authorization")),
            "no Authorization header may reach a cross-host target"
        );
        assert!(
            requests
                .iter()
                .all(|r| !r.headers.contains_key("x-custom-secret")),
            "no custom credential header may reach a cross-host target"
        );
        assert!(host_a.received_requests().await.unwrap().is_empty());
    }

    /// Req 1.10 (R2): every config save path stamps `admin.security_mode` with
    /// the resolved posture. GET → PUT (round trip, unchanged) → GET. NOTE (R1
    /// flip): the PUT body always carries an explicit `auth` section (Serialize
    /// never skips it), so the deserialized candidate has
    /// `auth_section_present == true`; marker-less + explicit section + disabled
    /// auth on a non-loopback bind now fails closed → "authenticated".
    #[tokio::test]
    async fn admin_config_put_round_trip_stamps_resolved_security_mode() {
        let config_dir = std::env::temp_dir().join(format!(
            "obey-admin-mode-stamp-{}",
            uuid::Uuid::new_v4().simple()
        ));
        std::fs::create_dir_all(&config_dir).unwrap();
        let config_path = config_dir.join("config.yaml");
        let mut initial = test_config_with_auth(false);
        // Legacy fixture: no marker, non-loopback bind. The in-memory fixture
        // has no auth section (OpenWarnedLegacy while running), but the PUT
        // round trip re-introduces an explicit auth section → R1 fail closed.
        initial.server.host = "192.168.1.110".to_string();
        initial.server.port = 8080;
        initial.admin.security_mode = None;
        initial.admin.auth_section_present = false;
        let server = crate::gateway::GatewayServer::new(initial, Some(config_path.clone()))
            .await
            .unwrap();

        let get = || {
            axum::http::Request::builder()
                .method("GET")
                .uri("/admin/config")
                .body(axum::body::Body::empty())
                .unwrap()
        };
        let response = tower::ServiceExt::oneshot(server.build_router(), get())
            .await
            .unwrap();
        assert_eq!(response.status(), StatusCode::OK);
        let config = smart_routing_response_json(response).await;
        assert!(
            config["admin"]["security_mode"].is_null(),
            "legacy fixture must start without a marker"
        );

        // Round-trip PUT with the GET body unchanged.
        let response = tower::ServiceExt::oneshot(
            server.build_router(),
            smart_routing_json_request("PUT", "/admin/config", config),
        )
        .await
        .unwrap();
        let status = response.status();
        let body = smart_routing_response_json(response).await;
        assert_eq!(status, StatusCode::OK, "PUT /admin/config failed: {body}");

        // The persisted YAML carries the stamped marker.
        let saved: Config =
            serde_yaml::from_str(&std::fs::read_to_string(&config_path).unwrap()).unwrap();
        assert_eq!(
            saved.admin.security_mode,
            Some(AdminSecurityMode::Authenticated),
            "save path must stamp the resolved mode (R1: explicit auth section fails closed)"
        );

        // Second GET reports the stamped marker. R1 flip: the live config is
        // now Authenticated (explicit auth section, disabled, LAN bind), so
        // the unauthenticated panel gets a 401 and a credentialed GET (env
        // vars the fixture already references) is required to read it back.
        let response = tower::ServiceExt::oneshot(server.build_router(), get())
            .await
            .unwrap();
        assert_eq!(response.status(), StatusCode::UNAUTHORIZED);

        std::env::set_var("TEST_ADMIN_USER", "admin");
        std::env::set_var("TEST_ADMIN_PASS", "secret");
        let authenticated_get = || {
            axum::http::Request::builder()
                .method("GET")
                .uri("/admin/config")
                .header("Authorization", basic_auth_header("admin", "secret"))
                .body(axum::body::Body::empty())
                .unwrap()
        };
        let response = tower::ServiceExt::oneshot(server.build_router(), authenticated_get())
            .await
            .unwrap();
        assert_eq!(response.status(), StatusCode::OK);
        let config = smart_routing_response_json(response).await;
        assert_eq!(config["admin"]["security_mode"], "authenticated");
        std::env::remove_var("TEST_ADMIN_USER");
        std::env::remove_var("TEST_ADMIN_PASS");

        let _ = std::fs::remove_dir_all(&config_dir);
    }

    /// R3: the guarded redirect policy stops cross-host redirect hops on the
    /// models proxy. The mock host is allowed (allow_private_hosts=true,
    /// allow_http=true so scheme is not the rejector), but its 302 Location
    /// points at `http://127.0.0.1:1/` — the request must fail with a
    /// host-only guard error.
    #[tokio::test]
    async fn proxy_provider_models_redirect_to_loopback_hop_is_stopped_by_guard() {
        let upstream = wiremock::MockServer::start().await;
        wiremock::Mock::given(wiremock::matchers::method("GET"))
            .and(wiremock::matchers::path("/v1/models"))
            .respond_with(
                wiremock::ResponseTemplate::new(302)
                    .insert_header("Location", "http://127.0.0.1:1/"),
            )
            .expect(1)
            .mount(&upstream)
            .await;

        let mut config = outbound_fetch_test_config(true);
        config.providers[0].base_url = Some(upstream.uri());
        let server = crate::gateway::GatewayServer::new(config, None)
            .await
            .unwrap();
        let app = server.build_router();

        let request = smart_routing_json_request(
            "POST",
            "/admin/providers/models",
            json!({
            "base_url": upstream.uri(),
            "api_key": "sk-redirect-secret",
            "provider_name": "test",
            "provider_type": "openai"
            }),
        );
        let response = tower::ServiceExt::oneshot(app, request).await.unwrap();
        assert_eq!(response.status(), StatusCode::BAD_GATEWAY);
        let body = to_bytes(response.into_body(), 16 * 1024).await.unwrap();
        let body_text = String::from_utf8_lossy(&body).into_owned();
        assert!(
            body_text.contains("127.0.0.1") && !body_text.contains("sk-redirect-secret"),
            "guard error must carry the host only: {body_text}"
        );
    }

    /// T2 (R3): caller-tier guard with `allow_private_hosts=false` rejects the
    /// wiremock (127.0.0.1) base_url with a 400 whose body is host-only — no
    /// full URL, no credential echo.
    #[tokio::test]
    async fn proxy_provider_models_guard_rejects_private_target_with_host_only_body() {
        let upstream = wiremock::MockServer::start().await;
        let config = outbound_fetch_test_config(false);
        let server = crate::gateway::GatewayServer::new(config, None)
            .await
            .unwrap();
        let app = server.build_router();

        let request = smart_routing_json_request(
            "POST",
            "/admin/providers/models",
            json!({
            "base_url": format!("{}/v1", upstream.uri()),
            "api_key": "sk-guard-secret",
            "provider_name": "test",
            "provider_type": "openai"
            }),
        );
        let response = tower::ServiceExt::oneshot(app, request).await.unwrap();
        assert_eq!(response.status(), StatusCode::BAD_REQUEST);
        let body = to_bytes(response.into_body(), 16 * 1024).await.unwrap();
        let body_text = String::from_utf8_lossy(&body).into_owned();
        assert!(
            body_text.contains("127.0.0.1"),
            "guard body must name the rejected host: {body_text}"
        );
        assert!(
            !body_text.contains(upstream.uri().trim_start_matches("http://")),
            "guard body must not echo the full URL: {body_text}"
        );
        assert!(
            !body_text.contains("sk-guard-secret"),
            "guard body must not echo credentials: {body_text}"
        );
        assert!(
            upstream.received_requests().await.unwrap().is_empty(),
            "guard must reject before any outbound work"
        );
    }

    /// S7 (R2): credential-bearing custom header values are redacted in the
    /// config GET response; the header name survives so the pair is visible.
    #[test]
    fn test_redact_config_for_response_redacts_credential_custom_headers() {
        let mut config = arb_config()
            .new_tree(&mut proptest::test_runner::TestRunner::default())
            .unwrap()
            .current();
        config.providers[0].custom_headers.insert(
            "Authorization".to_string(),
            "Bearer sk-custom-header-secret-abcdef".to_string(),
        );
        config.providers[0]
            .custom_headers
            .insert("X-Team".to_string(), "obey-core".to_string());

        let redacted = redact_config_for_response(&config);
        let headers = &redacted["providers"][0]["custom_headers"];
        assert_eq!(
            headers["Authorization"], "[configured]",
            "credential-bearing value must be redacted"
        );
        assert_eq!(headers["X-Team"], "obey-core", "non-secret value must stay");
        let serialized = serde_json::to_string(&redacted).unwrap();
        assert!(
            !serialized.contains("sk-custom-header-secret-abcdef"),
            "GET response leaked custom header secret: {serialized}"
        );
    }

    /// C7 (R8): scheme-default ports normalize away in origin comparison, so
    /// `https://host` and `https://host:443` are the same origin (and ditto
    /// http/80).
    #[test]
    fn test_normalize_provider_base_url_treats_default_ports_as_same_origin() {
        let (_, plain) = normalize_provider_base_url("https://api.example.com");
        let (_, explicit) = normalize_provider_base_url("https://api.example.com:443");
        assert_eq!(plain.as_deref(), Some("https://api.example.com"));
        assert_eq!(explicit.as_deref(), Some("https://api.example.com"));
        assert_eq!(plain, explicit);

        let (_, http_plain) = normalize_provider_base_url("http://api.example.com");
        let (_, http_explicit) = normalize_provider_base_url("http://api.example.com:80");
        assert_eq!(http_plain, http_explicit);

        // Non-default ports are still distinguished.
        let (_, alt) = normalize_provider_base_url("https://api.example.com:8443");
        assert_eq!(alt.as_deref(), Some("https://api.example.com:8443"));
    }

    /// R8/C6: `test_connection` with an unparseable base_url fails closed
    /// with the same 400 shape as a guard rejection (invalid base_url).
    #[tokio::test]
    async fn test_connection_unparseable_base_url_fails_closed() {
        // Isolate the OAuth token store so seeding a session never touches the
        // operator's real ai-gateway data directory.
        let data_dir = std::env::temp_dir().join(format!(
            "obey-admin-oauth-{}",
            uuid::Uuid::new_v4().simple()
        ));
        std::fs::create_dir_all(&data_dir).unwrap();
        std::env::set_var("AI_GATEWAY_DATA_DIR", &data_dir);
        let mut config = test_config_with_auth(false);
        config.providers[0].auth_method = Some("oauth".to_string());
        config.providers[0].codex_base_url_override = Some("::::not a url::::".to_string());
        let server = crate::gateway::GatewayServer::new(config, None)
            .await
            .unwrap();
        std::env::remove_var("AI_GATEWAY_DATA_DIR");

        // Seed an authenticated OAuth session so the handler passes the oauth
        // checks and reaches the URL guard.
        let manager = server.state.oauth_manager.clone().expect("oauth manager");
        let expires_at = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap()
            .as_secs()
            + 3600;
        manager
            .store_tokens(crate::oauth::store::StoredTokens {
                access_token: "sk-test-access".to_string(),
                refresh_token: "sk-test-refresh".to_string(),
                expires_at,
                scopes: "test".to_string(),
            })
            .await
            .unwrap();

        let app = server.build_router();
        let request = smart_routing_json_request(
            "POST",
            "/admin/test-connection",
            json!({ "provider_name": "test" }),
        );
        let response = tower::ServiceExt::oneshot(app, request).await.unwrap();
        assert_eq!(response.status(), StatusCode::BAD_REQUEST);
        let body = to_bytes(response.into_body(), 16 * 1024).await.unwrap();
        let json: serde_json::Value = serde_json::from_slice(&body).unwrap();
        assert_eq!(json["error"], "invalid base_url");

        let _ = std::fs::remove_dir_all(&data_dir);
    }

    /// T5 (R1): OpenOptIn middleware pin — explicit `security_mode:
    /// open-opt-in` plus the non-loopback acknowledgment serves the admin
    /// panel without credentials on a LAN bind, and `/admin/security-status`
    /// reports the mode.
    #[tokio::test]
    async fn security_status_open_opt_in_on_non_loopback_bind() {
        let mut config = test_config_with_auth(false);
        config.server.host = "192.168.1.110".to_string();
        config.admin.security_mode = Some(AdminSecurityMode::OpenOptIn);
        config.admin.auth_section_present = true;
        config.admin.auth.allow_unauthenticated_non_loopback = true;
        assert_eq!(
            auth_policy::resolve_admin_security_mode(&config),
            AdminSecurityMode::OpenOptIn
        );

        let server = crate::gateway::GatewayServer::new(config, None)
            .await
            .unwrap();
        let app = server.build_router();

        let request = axum::http::Request::get("/admin/config")
            .body(axum::body::Body::empty())
            .unwrap();
        let response = tower::ServiceExt::oneshot(app.clone(), request)
            .await
            .unwrap();
        assert_eq!(response.status(), StatusCode::OK);

        let request = axum::http::Request::get("/admin/security-status")
            .body(axum::body::Body::empty())
            .unwrap();
        let response = tower::ServiceExt::oneshot(app, request).await.unwrap();
        assert_eq!(response.status(), StatusCode::OK);
        let body = to_bytes(response.into_body(), 16 * 1024).await.unwrap();
        let json: serde_json::Value = serde_json::from_slice(&body).unwrap();
        assert_eq!(json["mode"], "open-opt-in");
        assert_eq!(json["unauthenticated_non_loopback"], true);
    }
}
