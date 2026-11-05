#![allow(dead_code)]
//! GitHub Copilot provider adapter (task 7 of the
//! `github-copilot-oauth-provider` spec, design Component 5).
//!
//! [`GitHubCopilotProvider`] implements the same [`crate::providers::ProviderClient`]
//! surface used by the `openai_compatible` provider family, so the router's
//! existing buffered-dispatch machinery (ADR 0001) drives Copilot requests
//! unchanged — there is NO Copilot-specific streaming code (Req 4.5).
//!
//! # Authentication (Req 8.1, 8.3)
//!
//! The bearer is resolved STRICTLY per the configured `auth_method`; the
//! selected method is authoritative. A missing or invalid credential for the
//! selected method is an actionable [`GatewayError::Provider`] error — the
//! adapter NEVER silently falls back to the other credential source:
//!
//! * `device-oauth` (default) — [`CopilotTokenManager::get_copilot_token`]
//! (task 5; includes the Req 3.4 inline refresh-once for just-expired
//! tokens). When authenticated, the manager's `api_base` (the
//! `endpoints.api` override from the token exchange) WINS over the
//! configured `base_url`.
//! * `api-key` — the PAT resolved through the existing
//! `api_key`/`api_key_env` config mechanism (passed in already resolved,
//! mirroring how the router feeds [`crate::config::Provider::resolve_api_key`]
//! to every other provider). The configured/effective base URL applies.
//!
//! # Streaming
//!
//! `chat_completion` parses a JSON body only: the router's copilot arm
//! always dispatches with `stream: false` (buffered dispatch, Bedrock arm
//! precedent), so an SSE body never reaches this adapter (Req 4.5).
//!
//! # Mandatory headers (Req 4.1, 4.2 / design Component 5)
//!
//! Every dispatched request carries the exact first-party client header set
//! (see [`copilot_mandatory_headers`]) plus the constant
//! `X-Initiator: agent` (design Decision 5a: an honest value for
//! programmatic gateway traffic; a gateway cannot reliably classify
//! end-user intent, and community evidence — copilot-api #68, LiteLLM
//! #18155 — ties initiator mislabeling to premium-quota burn). The constant
//! is applied LAST so operator `custom_headers` can never override the
//! mandatory set.
//!
//! # Machine / session ids (design "Machine/session ids")
//!
//! The stable machine id comes from the manager's persisted encrypted store
//! (task 5) and the session id is generated in-memory per adapter instance.
//! They are NOT request headers (the mandatory set above is closed per
//! design); they are exposed as diagnostics and included — as non-secret
//! fields — in the dispatch tracing context.
//!
//! # Auth-class failures (Req 4.3)
//!
//! Upstream 401/403 responses are mapped to a provider error carrying the
//! status code and the [`AUTH_CLASS_MARKER`] prefix. [`is_copilot_auth_error`]
//! classifies these for the router wiring (task 9), which forces one token
//! refresh and one re-dispatch before failing over.
//!
//! # Error surfacing (Req 4.6, 10.2)
//!
//! Upstream error bodies are truncated (256 bytes, UTF-8-boundary-safe,
//! same policy as `oauth/github_copilot/copilot_token.rs`) and scrubbed of
//! the outgoing bearer before being surfaced, with provider/status
//! diagnostic context. No token material ever reaches an error message.
//!
//! # Usage tracking (Req 4.4, 7.1, 7.2)
//!
//! Every Copilot response — success or error — feeds its headers to the
//! shared [`UsageTracker`] via `update_from_headers`, which parses both the
//! count-based `x-ratelimit-*` windows and the task 6
//! `x-copilot-premium-requests-*` percentage window.

use std::collections::HashMap;
use std::pin::Pin;
use std::sync::Arc;
use std::time::Instant;

use async_trait::async_trait;
use futures::stream::{Stream, StreamExt};
use serde::Deserialize;

use crate::error::GatewayError;
use crate::models::openai::{OpenAIRequest, OpenAIResponse};
use crate::oauth::github_copilot::copilot_token::DEFAULT_COPILOT_API_BASE;
use crate::oauth::github_copilot::manager::CopilotTokenManager;
use crate::oauth::github_copilot::generate_machine_id;
use crate::oauth::UsageTracker;
use crate::providers::{Model, ProviderClient, ProviderResponse, SSEEvent};

// ─── Mandatory header constants (Req 4.1, design Component 5) ─────────────
//
// Values mirror the header set used by the VS Code Copilot Chat client and
// established third-party clients (LiteLLM `copilot/handler.py`,
// CopilotChat.nvim, opencode): the ecosystem's known-good constants.

/// `Copilot-Integration-Id` — the allowlisted integration identifier used
/// by VS Code Copilot Chat. Copilot's edge rejects unknown values.
pub const COPILOT_INTEGRATION_ID: &str = "vscode-chat";

/// `Editor-Version` — editor identity presented to the Copilot edge.
pub const COPILOT_EDITOR_VERSION: &str = "vscode/1.95.0";

/// `Editor-Plugin-Version` — the Copilot Chat extension version.
pub const COPILOT_EDITOR_PLUGIN_VERSION: &str = "copilot-chat/0.23.5";

/// `User-Agent` — `GitHubCopilotChat/<version>` per design Component 5;
/// the version tracks the plugin version above.
pub const COPILOT_USER_AGENT: &str = "GitHubCopilotChat/0.23.5";

/// `Openai-Intent` — the conversation mode the Copilot backend meters
/// against (Req 4.1 pins this exact value).
pub const COPILOT_OPENAI_INTENT: &str = "conversation-edits";

/// `X-GitHub-Api-Version` — the Copilot API version date used by the
/// ecosystem (task brief: "a sensible constant like the ecosystem's").
pub const COPILOT_GITHUB_API_VERSION: &str = "2024-08-01";

/// `X-Initiator` — CONSTANT `agent` on every request (design Decision 5a).
/// No heuristic: per-request user/agent classification is unverifiable from
/// the gateway, and mislabeling agent traffic as user-initiated burns
/// premium-request quota.
pub const COPILOT_X_INITIATOR: &str = "agent";

/// Maximum number of bytes of an upstream error body retained in a
/// [`GatewayError::Provider`] message — identical policy to
/// `oauth/github_copilot/copilot_token.rs` (Req 4.6, 10.2).
const ERROR_BODY_MAX_BYTES: usize = 256;

/// Message prefix marking an upstream 401/403 auth-class failure that the
/// router wiring (task 9) treats as refresh-and-retry-once (Req 4.3).
const AUTH_CLASS_MARKER: &str = "copilot auth-class failure";

/// The mandatory Copilot header set (Req 4.1) plus the constant
/// `X-Initiator: agent` (Req 4.2 / design Decision 5a). Applied after
/// Authorization/Content-Type and any operator custom headers so the
/// mandatory values always win.
pub fn copilot_mandatory_headers() -> Vec<(&'static str, &'static str)> {
    vec![
        ("Copilot-Integration-Id", COPILOT_INTEGRATION_ID),
        ("Editor-Version", COPILOT_EDITOR_VERSION),
        ("Editor-Plugin-Version", COPILOT_EDITOR_PLUGIN_VERSION),
        ("User-Agent", COPILOT_USER_AGENT),
        ("Openai-Intent", COPILOT_OPENAI_INTENT),
        ("X-GitHub-Api-Version", COPILOT_GITHUB_API_VERSION),
        // Req 4.2 / Decision 5a: constant, never a heuristic.
        ("X-Initiator", COPILOT_X_INITIATOR),
    ]
}

/// Classify a [`GatewayError`] as a Copilot auth-class failure
/// (refresh-and-retry-once, Req 4.3) for the router wiring (task 9).
///
/// Matches only errors produced by [`GitHubCopilotProvider`]: an upstream
/// 401/403 mapped through [`AUTH_CLASS_MARKER`]. Missing-credential errors
/// (also 401-shaped) deliberately do NOT match — a forced token refresh
/// cannot remedy an unconfigured credential source.
pub fn is_copilot_auth_error(error: &GatewayError) -> bool {
    match error {
        GatewayError::Provider {
            status_code: Some(status),
            message,
            ..
        } => matches!(status, 401 | 403) && message.contains(AUTH_CLASS_MARKER),
        _ => false,
    }
}

/// Credential source selected by `auth_method` (Req 8.1). The selected
/// method is authoritative (Req 8.3): dispatch never falls back to the
/// other source.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum CopilotAuthMethod {
    /// Device-flow OAuth login → `CopilotTokenManager` (default).
    DeviceOauth,
    /// PAT via the existing `api_key`/`api_key_env` mechanism.
    ApiKey,
}

impl CopilotAuthMethod {
    /// Parse the configured `auth_method` string. `None` maps to the
    /// config-level default `device-oauth` (Req 1.1); unknown values are a
    /// configuration error (config validation rejects them earlier; the
    /// adapter defends in depth).
    pub fn parse(value: Option<&str>) -> Result<Self, GatewayError> {
        match value {
            None | Some("device-oauth") => Ok(Self::DeviceOauth),
            Some("api-key") => Ok(Self::ApiKey),
            Some(other) => Err(GatewayError::Configuration(format!(
                "unknown auth_method '{other}' for github-copilot provider \
                 (expected 'device-oauth' or 'api-key')"
            ))),
        }
    }
}

/// Truncate a response body to at most [`ERROR_BODY_MAX_BYTES`] bytes on a
/// UTF-8 character boundary (identical policy to
/// `oauth/github_copilot/copilot_token.rs`; Req 4.6, 10.2).
fn truncate_error_body(body: String) -> String {
    if body.len() <= ERROR_BODY_MAX_BYTES {
        return body;
    }
    let mut end = ERROR_BODY_MAX_BYTES;
    while end > 0 && !body.is_char_boundary(end) {
        end -= 1;
    }
    let mut truncated = body;
    truncated.truncate(end);
    truncated
}

/// Scrub the outgoing bearer (or PAT) from an upstream error body before it
/// is surfaced (Req 10.2: no token leakage). Upstream error bodies can echo
/// request material; replacing the exact credential string guarantees the
/// surfaced message is token-free.
fn redact_bearer(body: &str, bearer: &str) -> String {
    if bearer.is_empty() {
        body.to_string()
    } else {
        body.replace(bearer, "<redacted>")
    }
}

/// Per-dispatch resolved credential context (Req 8.1 strict resolution).
///
/// `machine_id`/`session_id` are non-secret diagnostics (see module docs);
/// they ride along so the dispatch tracing span carries the full context.
/// `bearer` is secret — the `Debug` impl is skipped by construction (no
/// derived Debug on this type).
pub struct CopilotDispatchContext {
    /// Bearer for `Authorization` (Copilot token or PAT per auth_method).
    pub bearer: String,
    /// Effective API base (manager `endpoints.api` override wins when
    /// authenticated; otherwise the configured/effective base URL).
    pub base_url: String,
    /// Stable machine id from the manager's encrypted store, when known.
    pub machine_id: Option<String>,
}

/// GitHub Copilot provider client (design Component 5).
///
/// Implements the same [`ProviderClient`] trait surface as
/// [`super::openai_compatible::OpenAICompatibleProvider`]; the router's
/// buffered-dispatch path drives it unchanged (Req 4.5 — no
/// Copilot-specific streaming code; [`Self::chat_completion_stream`] is the
/// same generic SSE passthrough the openai_compatible family uses).
pub struct GitHubCopilotProvider {
    /// Provider name for identification.
    name: String,
    /// Authoritative credential source (Req 8.3).
    auth_method: CopilotAuthMethod,
    /// PAT resolved through the existing `api_key`/`api_key_env` mechanism
    /// (`api-key` mode only).
    api_key: Option<String>,
    /// Copilot credential-lifecycle manager (`device-oauth` mode only).
    copilot_manager: Option<Arc<CopilotTokenManager>>,
    /// Usage tracker fed from every Copilot response's rate-limit headers.
    usage_tracker: Arc<UsageTracker>,
    /// Configured effective base URL (config task 1's `effective_base_url`;
    /// defaults to [`DEFAULT_COPILOT_API_BASE`] when `None`).
    base_url: Option<String>,
    /// Shared HTTP client (connection pooling owned by the caller, matching
    /// the `CodexProviderClient` pattern).
    http_client: reqwest::Client,
    /// Operator custom headers; applied before the mandatory set so the
    /// mandatory set always wins.
    custom_headers: HashMap<String, String>,
    /// Per-process session id (in-memory, generated once per adapter).
    session_id: String,
}

impl GitHubCopilotProvider {
    /// Construct a Copilot provider adapter.
    ///
    /// * `auth_method` — `Some("device-oauth")`/`None` (default) or
    /// `Some("api-key")`; authoritative for bearer resolution.
    /// * `api_key` — the already-resolved PAT for `api-key` mode (from
    /// `Provider::resolve_api_key()`); ignored (never a fallback) in
    /// `device-oauth` mode.
    /// * `copilot_manager` — the shared manager for `device-oauth` mode;
    /// ignored (never a fallback) in `api-key` mode.
    /// * `usage_tracker` — shared tracker receiving `x-ratelimit-*` /
    /// `x-copilot-premium-requests-*` headers from every response.
    /// * `base_url` — the provider's effective base URL (config task 1);
    /// `None` falls back to [`DEFAULT_COPILOT_API_BASE`]. When the OAuth
    /// session is authenticated, the manager's `endpoints.api` value wins.
    /// * `http_client` — shared reqwest client (pooling/timeouts owned by
    /// the caller, like `CodexProviderClient`).
    /// * `custom_headers` — operator headers applied before the mandatory
    /// Copilot set (mandatory headers always win collisions).
    pub fn new(
        name: String,
        auth_method: Option<&str>,
        api_key: Option<String>,
        copilot_manager: Option<Arc<CopilotTokenManager>>,
        usage_tracker: Arc<UsageTracker>,
        base_url: Option<String>,
        http_client: reqwest::Client,
        custom_headers: HashMap<String, String>,
    ) -> Result<Self, GatewayError> {
        Ok(Self {
            name,
            auth_method: CopilotAuthMethod::parse(auth_method)?,
            api_key,
            copilot_manager,
            usage_tracker,
            base_url,
            http_client,
            custom_headers,
            session_id: generate_machine_id(),
        })
    }

    /// Resolve the bearer + effective base URL STRICTLY per the configured
    /// `auth_method` (Req 8.1, 8.3). A missing credential for the selected
    /// method is an actionable error naming the remedy — never a silent
    /// fallback to the other credential source.
    pub async fn resolve_dispatch(&self) -> Result<CopilotDispatchContext, GatewayError> {
        let missing = |method: &str, remedy: &str| GatewayError::Provider {
            provider: self.name.clone(),
            message: format!(
                "auth_method '{method}' selected for github-copilot provider but {remedy}; \
                 configure the selected auth method — no fallback to the other credential \
                 source is performed (Req 8.3)"
            ),
            status_code: Some(401),
        };

        match self.auth_method {
            CopilotAuthMethod::DeviceOauth => {
                let manager = self.copilot_manager.as_ref().ok_or_else(|| {
                    missing(
                        "device-oauth",
                        "no Copilot token manager is attached (complete the GitHub \
                         device-flow login from the admin panel)",
                    )
                })?;
                let bearer = manager.get_copilot_token().await.map_err(|e| {
                    missing(
                        "device-oauth",
                        &format!("no usable Copilot token ({e}); re-login from the admin panel"),
                    )
                })?;
                // Manager value wins when authenticated: the manager cache is
                // populated exactly when get_copilot_token succeeded above.
                let base_url = manager
                    .api_base()
                    .await
                    .filter(|base| !base.is_empty())
                    .or_else(|| self.base_url.clone())
                    .unwrap_or_else(|| DEFAULT_COPILOT_API_BASE.to_string());
                let machine_id = manager.machine_id().await;
                Ok(CopilotDispatchContext {
                    bearer,
                    base_url,
                    machine_id,
                })
            }
            CopilotAuthMethod::ApiKey => {
                let pat = self
                    .api_key
                    .clone()
                    .filter(|key| !key.trim().is_empty())
                    .ok_or_else(|| {
                        missing(
                            "api-key",
                            "no api_key/api_key_env credential resolved (set a GitHub PAT \
                             with Copilot access)",
                        )
                    })?;
                let base_url = self
                    .base_url
                    .clone()
                    .filter(|base| !base.trim().is_empty())
                    .unwrap_or_else(|| DEFAULT_COPILOT_API_BASE.to_string());
                // Machine id: from the manager's persisted store when a
                // manager is attached and holds one; PAT-only deployments
                // have none (Option stays None — honest, not fabricated).
                let machine_id = match self.copilot_manager.as_ref() {
                    Some(manager) => manager.machine_id().await,
                    None => None,
                };
                Ok(CopilotDispatchContext {
                    bearer: pat,
                    base_url,
                    machine_id,
                })
            }
        }
    }

    /// Stable machine id from the manager's encrypted store (design
    /// "Machine/session ids"); `None` when no credential exists.
    pub async fn machine_id(&self) -> Option<String> {
        match self.copilot_manager.as_ref() {
            Some(manager) => manager.machine_id().await,
            None => None,
        }
    }

    /// Per-process session id (in-memory, generated once per adapter
    /// instance). Non-secret; safe for diagnostics.
    pub fn session_id(&self) -> &str {
        &self.session_id
    }

    /// Build a request with Authorization, custom headers, and the exact
    /// mandatory Copilot header set. Custom headers colliding with a
    /// mandatory name (case-insensitive) are SKIPPED — reqwest appends
    /// rather than replaces, so the only way to guarantee the exact
    /// mandatory values on the wire is to never add a competing value
    /// (Req 4.1/4.2).
    fn apply_dispatch_headers(
        &self,
        builder: reqwest::RequestBuilder,
        ctx: &CopilotDispatchContext,
    ) -> reqwest::RequestBuilder {
        let mut b = builder
            .header("Authorization", format!("Bearer {}", ctx.bearer))
            .header("Content-Type", "application/json");

        let mandatory = copilot_mandatory_headers();
        for (key, value) in &self.custom_headers {
            let collides = mandatory
                .iter()
                .any(|(m_key, _)| m_key.eq_ignore_ascii_case(key));
            if !collides {
                b = b.header(key.as_str(), value.as_str());
            }
        }

        for (key, value) in mandatory {
            b = b.header(key, value);
        }

        b
    }

    /// Dispatch URL for a path segment against the resolved base URL
    /// (Copilot's OpenAI-compatible surface hangs directly off the base —
    /// no `/v1` segment).
    fn dispatch_url(base_url: &str, path: &str) -> String {
        format!("{}{}", base_url.trim_end_matches('/'), path)
    }

    /// Map an upstream non-success response to a surfaced error (Req 4.3,
    /// 4.6, 10.2): auth-class marker + diagnostic context for 401/403,
    /// truncated + bearer-redacted body, no token material.
    fn upstream_error(
        &self,
        status: u16,
        body: String,
        bearer: &str,
        auth_class: bool,
    ) -> GatewayError {
        let body = truncate_error_body(redact_bearer(&body, bearer));
        let message = if auth_class {
            format!(
                "{AUTH_CLASS_MARKER} (refresh-and-retry-once): upstream HTTP {status} \
                 from Copilot API; body: {body}"
            )
        } else {
            format!("upstream HTTP {status} from Copilot API; body: {body}")
        };
        GatewayError::Provider {
            provider: self.name.clone(),
            message,
            status_code: Some(status),
        }
    }
}

// ─── SSE passthrough (identical generic parsing to openai_compatible) ──────

/// Parse one SSE chunk from a streaming response body — the same generic
/// parser `providers::openai_compatible` uses (kept local because that
/// module's helper is private). Not Copilot-specific logic (Req 4.5).
fn parse_sse_chunk(text: &str) -> Result<Option<SSEEvent>, GatewayError> {
    for line in text.lines() {
        if let Some(data) = line.strip_prefix("data: ") {
            if data.trim() == "[DONE]" {
                return Ok(None);
            }
            return Ok(Some(SSEEvent::new(data.to_string())));
        }
    }
    Ok(None)
}

/// `GET {base}/models` wire shape. Reuses the catalog entry type from task 8
/// (which models the Copilot-specific fields) instead of forking a parser.
#[derive(Debug, Deserialize)]
struct CopilotModelsBody {
    #[serde(default)]
    data: Vec<crate::oauth::github_copilot::catalog::CopilotCatalogEntry>,
}

#[async_trait]
impl ProviderClient for GitHubCopilotProvider {
    async fn chat_completion(
        &self,
        request: OpenAIRequest,
    ) -> Result<ProviderResponse, GatewayError> {
        let start = Instant::now();
        let ctx = self.resolve_dispatch().await?;
        let url = Self::dispatch_url(&ctx.base_url, "/chat/completions");

        tracing::info!(
            provider = %self.name,
            model = %request.model,
            url = %url,
            machine_id = ctx.machine_id.as_deref().unwrap_or("unknown"),
            session_id = %self.session_id,
            auth_method = ?self.auth_method,
            "Dispatching to GitHub Copilot"
        );

        let response = self
            .apply_dispatch_headers(self.http_client.post(&url), &ctx)
            .json(&request)
            .send()
            .await
            .map_err(|e| GatewayError::Network(format!("Copilot request failed: {}", e)))?;

        let status = response.status().as_u16();
        let latency_ms = start.elapsed().as_millis() as u64;

        // Req 4.4 / 7.1 / 7.2: every response — success or error — feeds the
        // usage tracker (update_from_headers parses both the count-based
        // x-ratelimit-* windows and the premium percentage window).
        let headers = response.headers().clone();
        self.usage_tracker.update_from_headers(&headers).await;

        if !reqwest::StatusCode::from_u16(status).is_ok_and(|s| s.is_success()) {
            let body = response
                .text()
                .await
                .unwrap_or_else(|_| "unknown error".to_string());
            let auth_class = status == 401 || status == 403;
            return Err(self.upstream_error(status, body, &ctx.bearer, auth_class));
        }

        let openai_response: OpenAIResponse = response
            .json()
            .await
            .map_err(|e| GatewayError::Network(format!("Failed to parse Copilot response: {}", e)))?;

        Ok(ProviderResponse {
            response: openai_response,
            provider_name: self.name.clone(),
            latency_ms,
        })
    }

    async fn chat_completion_stream(
        &self,
        request: OpenAIRequest,
    ) -> Result<Pin<Box<dyn Stream<Item = Result<SSEEvent, GatewayError>> + Send>>, GatewayError>
    {
        let ctx = self.resolve_dispatch().await?;
        let url = Self::dispatch_url(&ctx.base_url, "/chat/completions");

        let response = self
            .apply_dispatch_headers(self.http_client.post(&url), &ctx)
            .json(&request)
            .send()
            .await
            .map_err(|e| GatewayError::Network(format!("Copilot request failed: {}", e)))?;

        let status = response.status().as_u16();

        let headers = response.headers().clone();
        self.usage_tracker.update_from_headers(&headers).await;

        if !reqwest::StatusCode::from_u16(status).is_ok_and(|s| s.is_success()) {
            let body = response
                .text()
                .await
                .unwrap_or_else(|_| "unknown error".to_string());
            let auth_class = status == 401 || status == 403;
            return Err(self.upstream_error(status, body, &ctx.bearer, auth_class));
        }

        // Generic SSE passthrough — same shape as openai_compatible's
        // stream path; no Copilot-specific streaming logic (Req 4.5).
        let stream = response.bytes_stream();
        let sse_stream = stream
            .map(move |chunk_result| {
                chunk_result
                    .map_err(|e| GatewayError::Network(format!("Copilot stream error: {}", e)))
                    .and_then(|bytes| parse_sse_chunk(&String::from_utf8_lossy(&bytes)))
            })
            .filter_map(|result| async move {
                match result {
                    Ok(Some(event)) => Some(Ok(event)),
                    Ok(None) => None,
                    Err(e) => Some(Err(e)),
                }
            });

        Ok(Box::pin(sse_stream))
    }

    async fn list_models(&self) -> Result<Vec<Model>, GatewayError> {
        let ctx = self.resolve_dispatch().await?;
        let url = Self::dispatch_url(&ctx.base_url, "/models");

        let response = self
            .apply_dispatch_headers(self.http_client.get(&url), &ctx)
            .send()
            .await
            .map_err(|e| GatewayError::Network(format!("Copilot request failed: {}", e)))?;

        let status = response.status().as_u16();

        let headers = response.headers().clone();
        self.usage_tracker.update_from_headers(&headers).await;

        if !reqwest::StatusCode::from_u16(status).is_ok_and(|s| s.is_success()) {
            let body = response
                .text()
                .await
                .unwrap_or_else(|_| "unknown error".to_string());
            let auth_class = status == 401 || status == 403;
            return Err(self.upstream_error(status, body, &ctx.bearer, auth_class));
        }

        let body: CopilotModelsBody = response
            .json()
            .await
            .map_err(|e| GatewayError::Network(format!("Failed to parse Copilot models response: {}", e)))?;

        // Same enabled-only exposure as the task 8 catalog (Req 5.2).
        Ok(body
            .data
            .into_iter()
            .filter(|entry| entry.is_enabled())
            .map(|entry| entry.to_model(&self.name))
            .collect())
    }

    fn provider_name(&self) -> &str {
        &self.name
    }
}

#[cfg(test)]
mod tests {
use super::*;
use crate::models::openai::Message;
use wiremock::matchers::{method, path};
use wiremock::{Mock, MockServer, ResponseTemplate};

use crate::oauth::github_copilot::test_support::{now_plus, test_client};

// ---- test scaffolding -------------------------------------------------
//
// Endpoint/body builders are local; storage/client/time helpers are
// shared via the oauth::github_copilot::test_support module.

/// Unique credential-store path for one test.
fn unique_store() -> crate::oauth::github_copilot::store::CopilotTokenStore {
crate::oauth::github_copilot::test_support::unique_store("adapter_copilot_tokens")
}

use crate::oauth::github_copilot::manager::CopilotEndpoints;
use crate::oauth::github_copilot::device_flow::DeviceFlowEndpoints;

    fn endpoints(server: &MockServer) -> CopilotEndpoints {
        CopilotEndpoints {
            device_flow: DeviceFlowEndpoints {
                device_code_url: format!("{}/login/device/code", server.uri()),
                token_url: format!("{}/login/oauth/access_token", server.uri()),
            },
            copilot_token_url: format!("{}/copilot_internal/v2/token", server.uri()),
        }
    }

    /// Exchange body whose `endpoints.api` override points at the DISPATCH
    /// mock server, so the manager's api_base wins over any configured
    /// base URL and dispatch lands on the mock.
    fn copilot_token_body(dispatch_uri: &str, expires_at: u64) -> serde_json::Value {
        serde_json::json!({
            "token": "tid=test-copilot-bearer",
            "expires_at": expires_at,
            "refresh_in": 1500,
            "endpoints": { "api": dispatch_uri },
            "sku": "copilot_pro"
        })
    }

    /// Drive a full successful device-flow login (device grant + immediate
    /// poll success + exchange against `auth_server`, with the exchange's
    /// `endpoints.api` pointing at `dispatch_server`).
    async fn logged_in_manager(
        auth_server: &MockServer,
        dispatch_server_uri: &str,
    ) -> Arc<CopilotTokenManager> {
        Mock::given(method("POST"))
            .and(path("/login/device/code"))
            .respond_with(ResponseTemplate::new(200).set_body_json(serde_json::json!({
                "device_code": "dc-adapter-1",
                "user_code": "ADPT-1234",
                "verification_uri": "https://github.com/login/device",
                "expires_in": 900,
                "interval": 0
            })))
            .expect(1)
            .mount(auth_server)
            .await;
        Mock::given(method("POST"))
            .and(path("/login/oauth/access_token"))
            .respond_with(ResponseTemplate::new(200).set_body_json(serde_json::json!({
                "access_token": "gho-test-github-token",
                "token_type": "bearer",
                "scope": "read:user"
            })))
            .expect(1)
            .mount(auth_server)
            .await;
        Mock::given(method("GET"))
            .and(path("/copilot_internal/v2/token"))
            .respond_with(ResponseTemplate::new(200).set_body_json(copilot_token_body(
                dispatch_server_uri,
                now_plus(3600),
            )))
            .expect(1)
            .mount(auth_server)
            .await;

        let manager = CopilotTokenManager::with_endpoints(
            unique_store(),
            test_client(),
            endpoints(auth_server),
        );
        manager.start_login().await.expect("login should start");

        // Login flips to Authenticated only after the exchange (Req 2.4).
        for _ in 0..300 {
            if matches!(
                manager.session_state().await,
                crate::oauth::github_copilot::manager::CopilotSessionState::Authenticated { .. }
            ) {
                return Arc::new(manager);
            }
            tokio::time::sleep(std::time::Duration::from_millis(10)).await;
        }
        panic!(
            "session never became authenticated: {:?}",
            manager.session_state().await
        );
    }

    fn chat_request() -> OpenAIRequest {
        OpenAIRequest {
            model: "gpt-5-codex".to_string(),
            messages: vec![Message {
                role: "user".to_string(),
                content: serde_json::json!("hello"),
                extra: serde_json::Map::new(),
            }],
            stream: false,
            temperature: None,
            max_tokens: Some(16),
            extra: serde_json::Map::new(),
        }
    }

    fn chat_response_json() -> serde_json::Value {
        serde_json::json!({
            "id": "chatcmpl-copilot-test",
            "object": "chat.completion",
            "created": 1234567890i64,
            "model": "gpt-5-codex",
            "choices": [{
                "index": 0,
                "message": {"role": "assistant", "content": "hi"},
                "finish_reason": "stop"
            }],
            "usage": {"prompt_tokens": 1, "completion_tokens": 1, "total_tokens": 2}
        })
    }

    async fn mount_chat(dispatch: &MockServer, template: ResponseTemplate) {
        Mock::given(method("POST"))
            .and(path("/chat/completions"))
            .respond_with(template)
            .mount(dispatch)
            .await;
    }

    fn adapter(
        name: &str,
        auth_method: Option<&str>,
        api_key: Option<String>,
        manager: Option<Arc<CopilotTokenManager>>,
        base_url: Option<String>,
    ) -> GitHubCopilotProvider {
        GitHubCopilotProvider::new(
            name.to_string(),
            auth_method,
            api_key,
            manager,
            Arc::new(UsageTracker::new()),
            base_url,
            test_client(),
            HashMap::new(),
        )
        .unwrap()
    }

    /// Header value of a received request as a lossy string (HeaderMap
    /// lookups are case-insensitive).
    fn header_value<'r>(
        request: &'r wiremock::Request,
        name: &str,
    ) -> Option<&'r str> {
        request.headers.get(name).and_then(|v| v.to_str().ok())
    }

    // ---- Req 4.1 / 4.2: header builder ------------------------------------

    /// The mandatory header set is EXACT — every header present with the
    /// exact ecosystem value, including the constant X-Initiator: agent.
    #[test]
    fn header_builder_mandatory_set_is_exact() {
        let headers = copilot_mandatory_headers();
        assert_eq!(
            headers,
            vec![
                ("Copilot-Integration-Id", "vscode-chat"),
                ("Editor-Version", "vscode/1.95.0"),
                ("Editor-Plugin-Version", "copilot-chat/0.23.5"),
                ("User-Agent", "GitHubCopilotChat/0.23.5"),
                ("Openai-Intent", "conversation-edits"),
                ("X-GitHub-Api-Version", "2024-08-01"),
                ("X-Initiator", "agent"),
            ]
        );
    }

    /// Design Decision 5a: X-Initiator is a CONSTANT — no heuristic, no
    /// per-request variation (the builder has no inputs to vary it by).
    #[test]
    fn x_initiator_is_constant_agent_by_design_5a() {
        assert_eq!(COPILOT_X_INITIATOR, "agent");
        for _ in 0..8 {
            let set = copilot_mandatory_headers();
            assert!(
                set.contains(&("X-Initiator", "agent")),
                "X-Initiator: agent must be present on every build"
            );
        }
    }

    // ---- Req 8.1 / 8.3: strict per-auth_method bearer resolution ----------

    /// PAT mode dispatches with the PAT as bearer plus the full mandatory
    /// header set (Req 4.1 verified on the wire).
    #[tokio::test]
    async fn pat_dispatch_uses_pat_bearer_and_mandatory_headers() {
        let dispatch = MockServer::start().await;
        mount_chat(&dispatch, ResponseTemplate::new(200).set_body_json(chat_response_json()))
            .await;

        let provider = adapter(
            "copilot-pat",
            Some("api-key"),
            Some("ghp_test_pat_token".to_string()),
            None,
            Some(dispatch.uri()),
        );

        let response = provider
            .chat_completion(chat_request())
            .await
            .expect("pat dispatch should succeed");
        assert_eq!(response.provider_name, "copilot-pat");

        let requests = dispatch.received_requests().await.unwrap();
        assert_eq!(requests.len(), 1);
        let request = &requests[0];
        assert_eq!(
            header_value(request, "authorization"),
            Some("Bearer ghp_test_pat_token")
        );
        // Full mandatory set on the wire (Req 4.1 + 4.2).
        assert_eq!(
            header_value(request, "copilot-integration-id"),
            Some("vscode-chat")
        );
        assert_eq!(
            header_value(request, "editor-version"),
            Some("vscode/1.95.0")
        );
        assert_eq!(
            header_value(request, "editor-plugin-version"),
            Some("copilot-chat/0.23.5")
        );
        assert_eq!(
            header_value(request, "user-agent"),
            Some("GitHubCopilotChat/0.23.5")
        );
        assert_eq!(
            header_value(request, "openai-intent"),
            Some("conversation-edits")
        );
        assert_eq!(
            header_value(request, "x-github-api-version"),
            Some("2024-08-01")
        );
        assert_eq!(header_value(request, "x-initiator"), Some("agent"));
    }

    /// PAT mode with a logged-in OAuth session available: the selected
    /// method is authoritative — a missing PAT is an actionable error and
    /// dispatch NEVER falls back to the Copilot token (Req 8.3).
    #[tokio::test]
    async fn pat_mode_missing_pat_is_actionable_error_no_oauth_fallback() {
        let auth_server = MockServer::start().await;
        let dispatch = MockServer::start().await;
        let manager = logged_in_manager(&auth_server, &dispatch.uri()).await;

        let provider = adapter(
            "copilot-pat-missing",
            Some("api-key"),
            None,
            Some(manager),
            Some(dispatch.uri()),
        );

        let err = provider
            .chat_completion(chat_request())
            .await
            .expect_err("missing PAT must fail");
        match err {
            GatewayError::Provider {
                message, status_code, ..
            } => {
                assert_eq!(status_code, Some(401));
                assert!(
                    message.contains("api-key")
                        && message.contains("api_key")
                        && message.contains("no fallback"),
                    "error must be actionable and name the remedy: {message}"
                );
            }
            other => panic!("expected Provider error, got {other:?}"),
        }

        // No dispatch happened — the OAuth token was NOT used as fallback.
        assert_eq!(dispatch.received_requests().await.unwrap().len(), 0);
    }

    /// OAuth mode with a PAT configured: an unauthenticated session is an
    /// actionable error and dispatch NEVER falls back to the PAT (Req 8.3).
    #[tokio::test]
    async fn oauth_mode_unauthenticated_is_actionable_error_no_pat_fallback() {
        let dispatch = MockServer::start().await;
        let unauthenticated = Arc::new(CopilotTokenManager::with_endpoints(
            unique_store(),
            test_client(),
            CopilotEndpoints {
                device_flow: DeviceFlowEndpoints {
                    device_code_url: "http://127.0.0.1:9/login/device/code".to_string(),
                    token_url: "http://127.0.0.1:9/login/oauth/access_token".to_string(),
                },
                copilot_token_url: "http://127.0.0.1:9/copilot_internal/v2/token".to_string(),
            },
        ));

        let provider = adapter(
            "copilot-oauth-missing",
            Some("device-oauth"),
            Some("ghp_would_be_fallback_pat".to_string()),
            Some(unauthenticated),
            Some(dispatch.uri()),
        );

        let err = provider
            .chat_completion(chat_request())
            .await
            .expect_err("unauthenticated session must fail");
        match err {
            GatewayError::Provider {
                message, status_code, ..
            } => {
                assert_eq!(status_code, Some(401));
                assert!(
                    message.contains("device-oauth")
                        && message.contains("no usable Copilot token")
                        && message.contains("no fallback"),
                    "error must be actionable and name the remedy: {message}"
                );
            }
            other => panic!("expected Provider error, got {other:?}"),
        }

        // No dispatch happened — the PAT was NOT used as fallback.
        assert_eq!(dispatch.received_requests().await.unwrap().len(), 0);
    }

    /// OAuth mode with no manager attached at all: actionable error
    /// pointing at the admin login (Req 8.3).
    #[tokio::test]
    async fn oauth_mode_missing_manager_is_actionable_error() {
        let provider = adapter(
            "copilot-no-manager",
            None, // default auth_method = device-oauth
            None,
            None,
            Some("https://api.githubcopilot.com".to_string()),
        );

        let err = provider
            .resolve_dispatch()
            .await
            .err()
            .expect("missing manager must fail");
        assert!(
            err.to_string().contains("device-flow login"),
            "error must point at the login remedy: {err}"
        );
    }

    /// OAuth mode dispatches with the Copilot token as bearer, and the
    /// manager's `endpoints.api` override WINS over the configured base
    /// URL (dispatch lands on the override host).
    #[tokio::test]
    async fn oauth_dispatch_uses_copilot_token_and_manager_base_url_override() {
        let auth_server = MockServer::start().await;
        let dispatch = MockServer::start().await;
        let manager = logged_in_manager(&auth_server, &dispatch.uri()).await;
        mount_chat(&dispatch, ResponseTemplate::new(200).set_body_json(chat_response_json()))
            .await;

        let provider = adapter(
            "copilot-oauth",
            Some("device-oauth"),
            None,
            Some(manager),
            // Configured base would send traffic elsewhere — must be ignored.
            Some("https://configured-but-overridden.example".to_string()),
        );

        provider
            .chat_completion(chat_request())
            .await
            .expect("oauth dispatch should succeed");

        let requests = dispatch.received_requests().await.unwrap();
        assert_eq!(requests.len(), 1, "dispatch must use the manager api_base");
        assert_eq!(
            header_value(&requests[0], "authorization"),
            Some("Bearer tid=test-copilot-bearer")
        );
        assert_eq!(header_value(&requests[0], "x-initiator"), Some("agent"));
    }

    /// Base URL resolution matrix: PAT mode uses the configured base and
    /// falls back to the production default; OAuth mode lets the manager
    /// value win.
    #[tokio::test]
    async fn base_url_resolution_pat_configured_and_default() {
        let pat_configured = adapter(
            "copilot-url-1",
            Some("api-key"),
            Some("ghp_x".to_string()),
            None,
            Some("https://copilot.mirror.example".to_string()),
        );
        let ctx = pat_configured.resolve_dispatch().await.unwrap();
        assert_eq!(ctx.base_url, "https://copilot.mirror.example");

        let pat_default = adapter(
            "copilot-url-2",
            Some("api-key"),
            Some("ghp_x".to_string()),
            None,
            None,
        );
        let ctx = pat_default.resolve_dispatch().await.unwrap();
        assert_eq!(ctx.base_url, "https://api.githubcopilot.com");
    }

    // ---- Req 4.3: 401/403 → auth-class retryable-once ----------------------

    #[tokio::test]
    async fn upstream_401_maps_to_auth_class_retryable_once() {
        let dispatch = MockServer::start().await;
        mount_chat(
            &dispatch,
            ResponseTemplate::new(401).set_body_string("{\"error\": \"token expired\"}"),
        )
        .await;

        let provider = adapter(
            "copilot-401",
            Some("api-key"),
            Some("ghp_test_pat".to_string()),
            None,
            Some(dispatch.uri()),
        );

        let err = provider
            .chat_completion(chat_request())
            .await
            .expect_err("401 must fail");
        assert!(
            is_copilot_auth_error(&err),
            "401 must classify as auth-class: {err:?}"
        );
        match &err {
            GatewayError::Provider {
                status_code,
                message,
                ..
            } => {
                assert_eq!(status_code, &Some(401));
                assert!(message.contains("refresh-and-retry-once"));
                assert!(message.contains("token expired"), "diagnostic body retained");
            }
            other => panic!("expected Provider error, got {other:?}"),
        }
    }

    #[tokio::test]
    async fn upstream_403_maps_to_auth_class_retryable_once() {
        let dispatch = MockServer::start().await;
        mount_chat(
            &dispatch,
            ResponseTemplate::new(403).set_body_string("{\"error\": \"denied\"}"),
        )
        .await;

        let provider = adapter(
            "copilot-403",
            Some("api-key"),
            Some("ghp_test_pat".to_string()),
            None,
            Some(dispatch.uri()),
        );

        let err = provider
            .chat_completion(chat_request())
            .await
            .expect_err("403 must fail");
        assert!(is_copilot_auth_error(&err));
        assert!(err.to_string().contains("denied"));
    }

    /// Non-auth upstream errors are NOT auth-class, and missing-credential
    /// errors (401-shaped) do not classify either — a refresh cannot fix
    /// an unconfigured credential.
    #[tokio::test]
    async fn non_auth_errors_do_not_classify_as_auth_class() {
        let dispatch = MockServer::start().await;
        mount_chat(
            &dispatch,
            ResponseTemplate::new(500).set_body_string("upstream exploded"),
        )
        .await;

        let provider = adapter(
            "copilot-500",
            Some("api-key"),
            Some("ghp_test_pat".to_string()),
            None,
            Some(dispatch.uri()),
        );
        let err = provider
            .chat_completion(chat_request())
            .await
            .expect_err("500 must fail");
        assert!(!is_copilot_auth_error(&err));
        assert!(err.to_string().contains("upstream exploded"));

        let missing = adapter(
            "copilot-missing-cred",
            Some("api-key"),
            None,
            None,
            Some(dispatch.uri()),
        );
        let err = missing
            .chat_completion(chat_request())
            .await
            .expect_err("missing credential must fail");
        assert!(
            !is_copilot_auth_error(&err),
            "missing-credential 401 must not trigger refresh-and-retry"
        );
    }

    // ---- Req 4.6 / 10.2: truncated, redacted error bodies -------------------

    /// Error bodies are truncated to the 256-byte cap and the outgoing
    /// bearer is scrubbed before surfacing.
    #[tokio::test]
    async fn error_body_truncated_and_bearer_redacted() {
        let dispatch = MockServer::start().await;
        let bearer = "ghp_secret_pat_do_not_leak";
        // Body > 256 bytes that embeds the bearer.
        let mut body = format!("token={bearer} ").repeat(60);
        body.push_str("tail_marker_xyz");
        mount_chat(&dispatch, ResponseTemplate::new(502).set_body_string(body.clone()))
            .await;

        let provider = adapter(
            "copilot-redact",
            Some("api-key"),
            Some(bearer.to_string()),
            None,
            Some(dispatch.uri()),
        );

        let err = provider
            .chat_completion(chat_request())
            .await
            .expect_err("502 must fail");
        let message = err.to_string();
        assert!(!message.contains(bearer), "bearer must never leak: {message}");
        assert!(message.contains("<redacted>"));
        // Truncation: the surfaced body portion stays within the cap
        // (message overhead varies; assert the body did not pass through
        // whole — the original is ~1.6 KB).
        assert!(message.len() < body.len(), "body must be truncated");
        assert!(
            !message.contains("tail_marker_xyz"),
            "truncation must cut the tail"
        );
    }

    // ---- Req 4.4 / 7.1 / 7.2: usage tracker receives parsed headers ---------

    /// Response rate-limit headers (count-based + premium percentage) reach
    /// the usage tracker on successful dispatches.
    #[tokio::test]
    async fn usage_tracker_receives_parsed_rate_limit_headers() {
        let dispatch = MockServer::start().await;
        let now = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap()
            .as_secs();
        mount_chat(
            &dispatch,
            ResponseTemplate::new(200)
                .set_body_json(chat_response_json())
                .insert_header("x-ratelimit-limit-requests", "100")
                .insert_header("x-ratelimit-remaining-requests", "37")
                .insert_header("x-ratelimit-reset-requests", "3m")
                .insert_header("x-copilot-premium-requests-used", "42")
                .insert_header("x-copilot-premium-requests-limit", "100")
                .insert_header("x-copilot-premium-requests-reset", "1h"),
        )
        .await;
        let _ = now;

        let tracker = Arc::new(UsageTracker::new());
        let provider = GitHubCopilotProvider::new(
            "copilot-usage".to_string(),
            Some("api-key"),
            Some("ghp_usage_pat".to_string()),
            None,
            tracker.clone(),
            Some(dispatch.uri()),
            test_client(),
            HashMap::new(),
        )
        .unwrap();

        provider
            .chat_completion(chat_request())
            .await
            .expect("dispatch should succeed");

        let snapshot = tracker.snapshot().await;
        // Count-based short window populated.
        assert_eq!(snapshot.short.requests.limit, Some(100));
        assert_eq!(snapshot.short.requests.remaining, Some(37));
        assert!(snapshot.short.requests.resets_at.is_some());
        // Premium percentage window populated alongside (task 6 surface).
        let premium = snapshot.premium.expect("premium window recorded");
        assert!(premium.is_percentage);
        assert_eq!(premium.limit, Some(100));
        assert_eq!(premium.remaining, Some(58));
        assert!(premium.resets_at.is_some());
    }

    /// Headers are fed to the tracker even when the upstream errors —
    /// 429/401 responses still carry rate-limit context.
    #[tokio::test]
    async fn usage_tracker_fed_on_error_responses_too() {
        let dispatch = MockServer::start().await;
        mount_chat(
            &dispatch,
            ResponseTemplate::new(401)
                .set_body_string("denied")
                .insert_header("x-ratelimit-limit-requests", "10")
                .insert_header("x-ratelimit-remaining-requests", "0")
                .insert_header("x-ratelimit-reset-requests", "30m"),
        )
        .await;

        let tracker = Arc::new(UsageTracker::new());
        let provider = GitHubCopilotProvider::new(
            "copilot-usage-err".to_string(),
            Some("api-key"),
            Some("ghp_usage_pat".to_string()),
            None,
            tracker.clone(),
            Some(dispatch.uri()),
            test_client(),
            HashMap::new(),
        )
        .unwrap();

        assert!(provider.chat_completion(chat_request()).await.is_err());

        let snapshot = tracker.snapshot().await;
        assert_eq!(snapshot.short.requests.limit, Some(10));
        assert_eq!(snapshot.short.requests.remaining, Some(0));
    }

    // ---- Machine / session ids ----------------------------------------------

    /// Machine id comes from the manager's persisted encrypted store;
    /// session id is per-adapter in-memory, UUID-shaped, and stable.
    #[tokio::test]
    async fn machine_id_from_manager_store_and_session_id_stable() {
        let auth_server = MockServer::start().await;
        let dispatch = MockServer::start().await;
        let manager = logged_in_manager(&auth_server, &dispatch.uri()).await;

        let provider = adapter(
            "copilot-ids",
            Some("device-oauth"),
            None,
            Some(manager.clone()),
            None,
        );

        let machine_id = provider.machine_id().await.expect("machine id present");
        assert_eq!(machine_id.len(), 36, "UUID-shaped machine id");
        let groups: Vec<usize> = machine_id.split('-').map(str::len).collect();
        assert_eq!(groups, vec![8, 4, 4, 4, 12]);
        // Matches the store's persisted value.
        // (manager.store is private; equivalently, resolve_dispatch exposes it)
        let ctx = provider.resolve_dispatch().await.unwrap();
        assert_eq!(ctx.machine_id.as_deref(), Some(machine_id.as_str()));

        // Session id: stable across calls, UUID-shaped, distinct per adapter.
        let s1 = provider.session_id();
        let s2 = provider.session_id();
        assert_eq!(s1, s2);
        assert_eq!(s1.split('-').map(str::len).collect::<Vec<_>>(), vec![8, 4, 4, 4, 12]);

        let other = adapter(
            "copilot-ids-2",
            Some("device-oauth"),
            None,
            Some(manager),
            None,
        );
        assert_ne!(other.session_id(), provider.session_id());
    }

    // ---- Req 4.5: streaming rides the generic SSE passthrough ---------------

    /// Streaming reuses the generic SSE passthrough (same shape as
    /// openai_compatible) — no Copilot-specific streaming logic.
    #[tokio::test]
    async fn stream_passthrough_uses_generic_sse_parsing() {
        let dispatch = MockServer::start().await;
        mount_chat(
            &dispatch,
            ResponseTemplate::new(200)
                .set_body_string("data: {\"choices\":[{\"delta\":{\"content\":\"hi\"}}]}\n\n")
                .insert_header("content-type", "text/event-stream"),
        )
        .await;

        let provider = adapter(
            "copilot-stream",
            Some("api-key"),
            Some("ghp_stream_pat".to_string()),
            None,
            Some(dispatch.uri()),
        );

        let mut stream = provider.chat_completion_stream(chat_request()).await.unwrap();
        let mut events = 0;
        while let Some(item) = stream.next().await {
            let event = item.expect("sse event");
            assert!(event.data.contains("\"delta\""));
            events += 1;
        }
        assert_eq!(events, 1);

        // The streaming dispatch carried the same mandatory headers.
        let requests = dispatch.received_requests().await.unwrap();
        assert_eq!(requests.len(), 1);
        assert_eq!(header_value(&requests[0], "x-initiator"), Some("agent"));
        assert_eq!(
            header_value(&requests[0], "copilot-integration-id"),
            Some("vscode-chat")
        );
    }

    // ---- auth_method parsing -------------------------------------------------

    #[test]
    fn auth_method_parse_matrix() {
        assert_eq!(
            CopilotAuthMethod::parse(None).unwrap(),
            CopilotAuthMethod::DeviceOauth,
            "None = config default device-oauth"
        );
        assert_eq!(
            CopilotAuthMethod::parse(Some("device-oauth")).unwrap(),
            CopilotAuthMethod::DeviceOauth
        );
        assert_eq!(
            CopilotAuthMethod::parse(Some("api-key")).unwrap(),
            CopilotAuthMethod::ApiKey
        );
        for bad in ["oauth", "pat", "device", ""] {
            assert!(
                CopilotAuthMethod::parse(Some(bad)).is_err(),
                "'{bad}' must be rejected"
            );
        }
    }

    /// Unknown auth_method at construction is a configuration error
    /// (defense in depth; config validation rejects it earlier).
    #[test]
    fn unknown_auth_method_at_construction_is_configuration_error() {
        let result = GitHubCopilotProvider::new(
            "copilot-bad".to_string(),
            Some("pat"),
            None,
            None,
            Arc::new(UsageTracker::new()),
            None,
            test_client(),
            HashMap::new(),
        );
        assert!(matches!(result, Err(GatewayError::Configuration(_))));
    }

    /// Custom headers ride along but the mandatory set wins collisions.
    #[tokio::test]
    async fn custom_headers_applied_but_mandatory_set_wins() {
        let dispatch = MockServer::start().await;
        mount_chat(&dispatch, ResponseTemplate::new(200).set_body_json(chat_response_json()))
            .await;

        let mut custom = HashMap::new();
        custom.insert("X-Custom-Tag".to_string(), "custom-value".to_string());
        // Attempt to override two mandatory headers — must be ignored.
        custom.insert("X-Initiator".to_string(), "user".to_string());
        custom.insert("Copilot-Integration-Id".to_string(), "evil-id".to_string());

        let provider = GitHubCopilotProvider::new(
            "copilot-custom".to_string(),
            Some("api-key"),
            Some("ghp_custom".to_string()),
            None,
            Arc::new(UsageTracker::new()),
            Some(dispatch.uri()),
            test_client(),
            custom,
        )
        .unwrap();

        provider
            .chat_completion(chat_request())
            .await
            .expect("dispatch should succeed");

        let requests = dispatch.received_requests().await.unwrap();
        assert_eq!(header_value(&requests[0], "x-custom-tag"), Some("custom-value"));
        assert_eq!(header_value(&requests[0], "x-initiator"), Some("agent"));
        assert_eq!(
            header_value(&requests[0], "copilot-integration-id"),
            Some("vscode-chat")
        );
    }
}
