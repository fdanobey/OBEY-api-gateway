#![allow(dead_code)]
//! Model-catalog sync for the `github-copilot` provider (task 8 of the
//! `github-copilot-oauth-provider` spec, design Component 6; Reqs
//! 5.1-5.4).
//!
//! Once the Copilot session is authenticated, [`CopilotCatalogSync`]
//! fetches `GET {api_base}/models` with the Copilot bearer token from
//! [`super::manager::CopilotTokenManager`] (Req 5.1) and caches the
//! result on the EXISTING provider-catalog cadence: the same 24 h TTL
//! cache pattern as [`crate::codex::models_discovery::ModelsDiscovery`],
//! checked on the shared [`crate::oauth::lifecycle::REFRESH_TICK_INTERVAL`]
//! tick used by both OAuth managers. No new scheduler is introduced.
//!
//! # Filtering (Req 5.2)
//!
//! Response entries carry `model_picker_enabled` plus optional
//! policy/entitlement flags; only enabled entries are exposed through
//! [`CopilotCatalogSync::enabled_models`], which feeds the existing
//! models aggregation behind `/v1/models` (gateway wiring: task 9).
//! Entries missing `model_picker_enabled` default to disabled — only
//! entries explicitly enabled for the account are exposed. There is NO
//! hard-coded fallback model list (spec task rules): with no
//! successful fetch yet, the catalog is simply empty.
//!
//! # Warnings
//!
//! * **Configured-model-absent (Req 5.3)** — when the provider config
//! lists explicit `models` absent from the fetched catalog, the result
//! of [`crate::config::validation::unknown_copilot_models`] is
//! surfaced via `tracing::warn!` and returned in
//! [`CatalogStatus::unknown_configured_models`]. The configured list is
//! supplied by the gateway wiring (task 9) via
//! [`CopilotCatalogSync::set_configured_models`].
//! * **Fetch failure (Req 5.4)** — on failure the LAST known catalog
//! keeps being served ([`CopilotCatalogSync::enabled_models`] is never
//! cleared by a failure) and [`CatalogStatus::fetch_failed`] is set
//! for the admin status card (tasks 10/11).
//!
//! # Never blocks startup or dispatch
//!
//! All network I/O lives in [`CopilotCatalogSync::refresh_now`] /
//! [`CopilotCatalogSync::refresh_if_due`], which only the spawned
//! background task (or an explicit admin-triggered call) awaits.
//! `enabled_models`/`status` are lock-only snapshot reads with no
//! network I/O, so catalog sync can never block startup or dispatch.

use std::sync::Arc;
use std::time::SystemTime;

use serde::{Deserialize, Serialize};
use tokio::sync::RwLock;
use tokio::task::JoinHandle;

use super::manager::{CopilotSessionState, CopilotTokenManager};
use crate::oauth::lifecycle::REFRESH_TICK_INTERVAL;
use crate::providers::Model;

/// Catalog refresh TTL — the existing provider-catalog cadence
/// (`codex::models_discovery::CACHE_TTL_SECS`, 24 h). A cached catalog
/// younger than this is served without a refetch; the background tick
/// retries after expiry (and after failures, since a failed fetch does
/// not bump the cached timestamp).
pub(crate) const CATALOG_TTL_SECS: u64 = 86_400;

/// `policy` object of a catalog entry (Req 5.2: "policy/entitlement
/// flags permitting"). Only `state` is modeled; unknown fields are
/// ignored by serde. A present-but-unrecognized state excludes the
/// model — the conservative direction for entitlement gating.
#[derive(Debug, Clone, Deserialize)]
pub struct CopilotModelPolicy {
    /// Policy state, e.g. `"enabled"`, `"off"`, `"disabled"`.
    #[serde(default)]
    pub state: Option<String>,
}

/// One entry of the `GET {api_base}/models` response (Req 5.2).
#[derive(Debug, Clone, Deserialize)]
pub struct CopilotCatalogEntry {
    /// Model identifier as routed through the Copilot API.
    pub id: String,
    /// Whether the model is enabled for the account's model picker.
    /// Missing defaults to `false` (disabled) — only entries
    /// explicitly enabled for the account are exposed.
    #[serde(default)]
    pub model_picker_enabled: bool,
    /// Optional creation timestamp (Unix seconds).
    #[serde(default)]
    pub created: Option<i64>,
    /// Optional policy/entitlement gate.
    #[serde(default)]
    pub policy: Option<CopilotModelPolicy>,
}

impl CopilotCatalogEntry {
    /// Enabled for this account: `model_picker_enabled` AND, when a
    /// policy object is present, its `state` must be `"enabled"`
    /// (case-insensitive; a missing state inside a present policy
    /// object does not gate).
    pub fn is_enabled(&self) -> bool {
        if !self.model_picker_enabled {
            return false;
        }
        match &self.policy {
            Some(policy) => match &policy.state {
                Some(state) => state.eq_ignore_ascii_case("enabled"),
                None => true,
            },
            None => true,
        }
    }

    /// Map to the gateway's aggregated [`Model`] shape behind
    /// `/v1/models` (Req 5.2 feed).
    pub fn to_model(&self, owner: &str) -> Model {
        Model {
            id: self.id.clone(),
            object: "model".to_string(),
            owned_by: owner.to_string(),
            created: self.created,
            context_window: None,
            max_completion_tokens: None,
            supports_vision: false,
        }
    }
}

/// Wire shape of `GET {api_base}/models` (OpenAI-compatible `data`
/// array).
#[derive(Debug, Deserialize)]
struct CopilotModelsResponse {
    #[serde(default)]
    data: Vec<CopilotCatalogEntry>,
}

/// Cached catalog state (last known good fetch).
#[derive(Debug, Clone)]
struct CachedCatalog {
    /// All fetched entry ids (Req 5.3 comparison domain: the fetched
    /// catalog, not only the enabled subset).
    all_ids: Vec<String>,
    /// Enabled-only entries mapped to the aggregation [`Model`] shape.
    enabled_models: Vec<Model>,
    /// Unix seconds of the successful fetch.
    fetched_at: u64,
}

/// Non-secret snapshot of the catalog state for the admin status card
/// (Req 5.4 warning flag consumed by tasks 10/11). Carries no token
/// material.
#[derive(Debug, Clone, PartialEq, Serialize)]
pub struct CatalogStatus {
    /// Unix seconds of the last successful fetch; `None` when the
    /// catalog has never been fetched successfully.
    pub last_success_at: Option<u64>,
    /// Number of enabled models currently served.
    pub model_count: usize,
    /// Req 5.4 warning flag: the most recent refresh attempt failed
    /// and the last known catalog is being served.
    pub fetch_failed: bool,
    /// Truncated, non-secret description of the last failure, when
    /// [`Self::fetch_failed`] is set.
    pub last_error: Option<String>,
    /// Req 5.3 result: configured `models` absent from the fetched
    /// catalog (empty when all configured models are known or no
    /// explicit list is configured).
    pub unknown_configured_models: Vec<String>,
}

impl Default for CatalogStatus {
    fn default() -> Self {
        Self {
            last_success_at: None,
            model_count: 0,
            fetch_failed: false,
            last_error: None,
            unknown_configured_models: Vec::new(),
        }
    }
}

/// Fetch `GET {api_base}/models` with the Copilot bearer (Req 5.1).
///
/// Pure fetch — no caching, no state. Error strings carry only the
/// HTTP status / transport description, never the bearer token
/// (Req 8.4 / 10.2: no token material in diagnostics).
async fn fetch_catalog_entries(
    http: &reqwest::Client,
    bearer: &str,
    api_base: &str,
) -> Result<Vec<CopilotCatalogEntry>, String> {
    let url = format!("{}/models", api_base.trim_end_matches('/'));
    let response = http
        .get(&url)
        .bearer_auth(bearer)
        .send()
        .await
        .map_err(|e| format!("catalog request failed: {e}"))?;
    let status = response.status();
    if !status.is_success() {
        return Err(format!("catalog endpoint returned HTTP {status}"));
    }
    let body: CopilotModelsResponse = response
        .json()
        .await
        .map_err(|e| format!("catalog body parse failed: {e}"))?;
    Ok(body.data)
}

/// Model-catalog sync component (design Component 6).
///
/// Holds the last-known catalog and a non-secret status snapshot; all
/// mutable state sits behind `RwLock`s so snapshot reads never block
/// on network I/O (never blocks startup or dispatch).
pub struct CopilotCatalogSync {
    manager: Arc<CopilotTokenManager>,
    http: reqwest::Client,
    /// `owned_by` tag for aggregated models (the provider's name).
    owner: String,
    cache: RwLock<Option<CachedCatalog>>,
    status: RwLock<CatalogStatus>,
    /// Explicit `models` list configured on the `github-copilot`
    /// provider (Req 5.3 input), kept current by the gateway wiring.
    configured_models: RwLock<Vec<String>>,
}

impl CopilotCatalogSync {
    /// Construct the catalog sync over an authenticated-capable
    /// manager. Nothing is fetched until the background task (or an
    /// explicit refresh call) runs.
    pub fn new(manager: Arc<CopilotTokenManager>, http: reqwest::Client) -> Self {
        Self::with_owner(manager, http, "github-copilot".to_string())
    }

    /// Construct with an explicit `owned_by` tag (the provider's
    /// configured name) for aggregated model entries.
    pub fn with_owner(
        manager: Arc<CopilotTokenManager>,
        http: reqwest::Client,
        owner: String,
    ) -> Self {
        Self {
            manager,
            http,
            owner,
            cache: RwLock::new(None),
            status: RwLock::new(CatalogStatus::default()),
            configured_models: RwLock::new(Vec::new()),
        }
    }

    /// Update the configured explicit `models` list (Req 5.3 input;
    /// set by the gateway/config wiring and on hot-reload).
    pub async fn set_configured_models(&self, models: Vec<String>) {
        *self.configured_models.write().await = models;
    }

    /// TTL-gated refresh on the existing provider-catalog cadence:
    /// a no-op while the cached catalog is younger than
    /// [`CATALOG_TTL_SECS`]; delegates to [`Self::refresh_now`]
    /// otherwise. Skipped (not a failure) while the session is not
    /// authenticated.
    pub async fn refresh_if_due(&self) {
        let fresh = {
            let cache = self.cache.read().await;
            cache
                .as_ref()
                .is_some_and(|cached| now_unix_secs().saturating_sub(cached.fetched_at) < CATALOG_TTL_SECS)
        };
        if fresh {
            return;
        }
        self.refresh_now().await;
    }

    /// Fetch the catalog now (bypasses the TTL; still a no-op while
    /// unauthenticated — Req 5.1 keys the fetch on an authenticated
    /// session).
    ///
    /// * Success → cache replaced, [`CatalogStatus`] updated, Req 5.3
    /// check re-evaluated against the fresh catalog.
    /// * Failure → last known catalog retained, `fetch_failed`
    /// warning set (Req 5.4).
    pub async fn refresh_now(&self) {
        // Req 5.1: only fetch once the Copilot session is
        // authenticated. An unauthenticated/expired session is a skip,
        // not a fetch failure — no warning flag, no request.
        match self.manager.session_state().await {
            CopilotSessionState::Authenticated { .. } => {}
            _ => return,
        }

        let bearer = match self.manager.get_copilot_token().await {
            Ok(token) => token,
            Err(e) => {
                self.record_failure(format!("copilot token unavailable: {e}"))
                    .await;
                return;
            }
        };
        let api_base = self.manager.api_base().await.unwrap_or_default();
        if api_base.is_empty() {
            self.record_failure("no copilot api base resolved".to_string())
                .await;
            return;
        }

        match fetch_catalog_entries(&self.http, &bearer, &api_base).await {
            Ok(entries) => self.apply_fresh_catalog(entries).await,
            Err(error) => self.record_failure(error).await,
        }
    }

    /// Install a successfully fetched catalog: filter to enabled
    /// entries (Req 5.2), cache, and re-run the configured-model-absent
    /// check (Req 5.3) against the full fetched id set.
    async fn apply_fresh_catalog(&self, entries: Vec<CopilotCatalogEntry>) {
        let now = now_unix_secs();
        let all_ids: Vec<String> = entries
            .iter()
            .filter(|entry| !entry.id.is_empty())
            .map(|entry| entry.id.clone())
            .collect();
        let enabled_models: Vec<Model> = entries
            .iter()
            .filter(|entry| !entry.id.is_empty() && entry.is_enabled())
            .map(|entry| entry.to_model(&self.owner))
            .collect();

        // Req 5.3: warn through the config-validation warning channel
        // (task 1 helper) when explicit configured models are absent
        // from the fetched catalog.
        let configured = self.configured_models.read().await.clone();
        let catalog_refs: Vec<&str> = all_ids.iter().map(String::as_str).collect();
        let unknown =
            crate::config::validation::unknown_copilot_models(&configured, &catalog_refs);
        if !unknown.is_empty() {
            tracing::warn!(
                provider = %self.owner,
                unknown_models = ?unknown,
                "copilot: configured models are absent from the fetched catalog"
            );
        }

        let model_count = enabled_models.len();
        *self.cache.write().await = Some(CachedCatalog {
            all_ids,
            enabled_models,
            fetched_at: now,
        });
        let mut status = self.status.write().await;
        status.last_success_at = Some(now);
        status.model_count = model_count;
        status.fetch_failed = false;
        status.last_error = None;
        status.unknown_configured_models = unknown;
    }

    /// Record a fetch failure (Req 5.4): the last known catalog stays
    /// cached and the warning flag lands in the status snapshot for
    /// the admin card.
    async fn record_failure(&self, error: String) {
        tracing::warn!(
            provider = %self.owner,
            error = %error,
            "copilot: catalog fetch failed; serving last known catalog"
        );
        let mut status = self.status.write().await;
        status.fetch_failed = true;
        status.last_error = Some(error);
    }

    /// Enabled models from the last known catalog (Req 5.2 feed for
    /// the `/v1/models` aggregation). Empty until a first successful
    /// fetch; lock-only — never performs network I/O and never blocks
    /// dispatch.
    pub async fn enabled_models(&self) -> Vec<Model> {
        self.cache
            .read()
            .await
            .as_ref()
            .map(|cached| cached.enabled_models.clone())
            .unwrap_or_default()
    }

/// Non-secret status snapshot for the admin status card (Req 5.4
/// warning flag; tasks 10/11 consume this).
pub async fn status(&self) -> CatalogStatus {
    self.status.read().await.clone()
}

/// Save-time Req 5.3 warning helper: given the explicit `models` a
/// saved config points at this provider, return those absent from the
/// last KNOWN catalog cache. `None` when no catalog has been fetched
/// (nothing to compare against — never warn on an empty cache) or when
/// every configured model is known. The admin save path appends the
/// result to its save response warnings; it must never block a save.
pub async fn warning_for_saved_config(&self, models: &[String]) -> Option<Vec<String>> {
    let cache = self.cache.read().await;
    let cached = cache.as_ref()?;
    let catalog_refs: Vec<&str> = cached.all_ids.iter().map(String::as_str).collect();
    let unknown =
        crate::config::validation::unknown_copilot_models(models, &catalog_refs);
    if unknown.is_empty() {
        None
    } else {
        Some(unknown)
    }
}

    /// Spawn the background sync task on the existing cadence
    /// machinery: an immediate post-start attempt (covers "fetch
    /// after the session is authenticated", Req 5.1 — cheap because
    /// TTL-gated and skipped while unauthenticated), then the shared
    /// [`REFRESH_TICK_INTERVAL`] tick calls [`Self::refresh_if_due`],
    /// mirroring the OAuth managers' `start_refresh_loop` pattern.
    /// The returned handle lets the caller abort the task (logout /
    /// shutdown wiring in tasks 9/10).
    pub fn spawn_background_sync(self: &Arc<Self>) -> JoinHandle<()> {
        let sync = Arc::clone(self);
        tokio::spawn(async move {
            // Prompt first attempt (no-op while unauthenticated).
            sync.refresh_if_due().await;
            let mut interval = tokio::time::interval(REFRESH_TICK_INTERVAL);
            // Consume the immediate first tick, mirroring the manager
            // refresh loops (full cadence before the next check).
            interval.tick().await;
            loop {
                interval.tick().await;
                sync.refresh_if_due().await;
            }
        })
    }
}

/// Current wall-clock time as seconds since the Unix epoch, clamped
/// at 0 (same helper as the managers).
fn now_unix_secs() -> u64 {
    SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .map(|d| d.as_secs())
        .unwrap_or(0)
}

#[cfg(test)]
mod tests {
use super::*;
use crate::oauth::github_copilot::test_support::{
now_plus, test_client,
};
use wiremock::matchers::{method, path};
use wiremock::{Mock, MockServer, ResponseTemplate};

use crate::oauth::github_copilot::copilot_token::COPILOT_TOKEN_URL;
use crate::oauth::github_copilot::device_flow::{
DeviceFlowEndpoints, COPILOT_OAUTH_SCOPE,
};
use crate::oauth::github_copilot::manager::{CopilotEndpoints, CopilotTokenManager};
use crate::oauth::github_copilot::store::CopilotTokenStore;

// ---- test scaffolding (endpoint/body builders local to this file;
// storage/client/time helpers shared via test_support) -------------

fn unique_store() -> CopilotTokenStore {
crate::oauth::github_copilot::test_support::unique_store("catalog_tokens")
}

fn endpoints(server: &MockServer) -> CopilotEndpoints {
        CopilotEndpoints {
            device_flow: DeviceFlowEndpoints {
                device_code_url: format!("{}/login/device/code", server.uri()),
                token_url: format!("{}/login/oauth/access_token", server.uri()),
            },
            copilot_token_url: COPILOT_TOKEN_URL
                .replace("https://api.github.com", server.uri().trim_end_matches('/')),
        }
    }

    /// Exchange body whose `endpoints.api` points at the mock server so
    /// the catalog fetch (`{api_base}/models`) is drivable by wiremock.
    fn copilot_token_body(server_uri: &str, expires_at: u64) -> serde_json::Value {
        serde_json::json!({
            "token": "tid=test-copilot-bearer",
            "expires_at": expires_at,
            "refresh_in": 1500,
            "endpoints": { "api": server_uri },
            "sku": "copilot_pro"
        })
    }

    async fn mount_login(server: &MockServer, copilot_expires_at: u64) {
        Mock::given(method("POST"))
            .and(path("/login/device/code"))
            .respond_with(ResponseTemplate::new(200).set_body_json(
                serde_json::json!({
                    "device_code": "dc-cat-1",
                    "user_code": "ABCD-1234",
                    "verification_uri": "https://github.com/login/device",
                    "expires_in": 900,
                    "interval": 0
                }),
            ))
            .expect(1)
            .mount(server)
            .await;
        Mock::given(method("POST"))
            .and(path("/login/oauth/access_token"))
            .respond_with(ResponseTemplate::new(200).set_body_json(
                serde_json::json!({
                    "access_token": "gho-test-github-token",
                    "token_type": "bearer",
                    "scope": COPILOT_OAUTH_SCOPE
                }),
            ))
            .expect(1)
            .mount(server)
            .await;
        Mock::given(method("GET"))
            .and(path("/copilot_internal/v2/token"))
            .respond_with(ResponseTemplate::new(200).set_body_json(
                copilot_token_body(&server.uri(), copilot_expires_at),
            ))
            .expect(1)
            .mount(server)
            .await;
    }

    /// Catalog response mixing enabled, picker-gated, and policy-gated
    /// entries (Req 5.2 filter domain).
    fn catalog_body() -> serde_json::Value {
        serde_json::json!({
            "data": [
                {
                    "id": "gpt-5-codex",
                    "model_picker_enabled": true,
                    "created": 1_700_000_000
                },
                {
                    "id": "gated-picker-model",
                    "model_picker_enabled": false
                },
                {
                    "id": "gated-policy-model",
                    "model_picker_enabled": true,
                    "policy": { "state": "off" }
                },
                {
                    "id": "missing-flag-model"
                }
            ]
        })
    }

    fn models_mock(body: serde_json::Value) -> Mock {
        Mock::given(method("GET"))
            .and(path("/models"))
            .respond_with(ResponseTemplate::new(200).set_body_json(body))
    }

    async fn logged_in_manager(
        server: &MockServer,
        copilot_expires_at: u64,
    ) -> Arc<CopilotTokenManager> {
        mount_login(server, copilot_expires_at).await;
        let manager = CopilotTokenManager::with_endpoints(
            unique_store(),
            test_client(),
            endpoints(server),
        );
        manager.start_login().await.expect("login should start");
        for _ in 0..300 {
            if matches!(
                manager.session_state().await,
                CopilotSessionState::Authenticated { .. }
            ) {
                return Arc::new(manager);
            }
            tokio::time::sleep(std::time::Duration::from_millis(10)).await;
        }
panic!("session never became authenticated");
}

async fn count_models_requests(server: &MockServer) -> usize {
        server
            .received_requests()
            .await
            .unwrap()
            .into_iter()
            .filter(|r| r.url.path() == "/models")
            .count()
    }

    // ---- Req 5.1 + 5.2: authenticated → fetched + filtered --------

    #[tokio::test]
    async fn authenticated_catalog_fetched_with_bearer_and_filtered() {
        let server = MockServer::start().await;
        let manager = logged_in_manager(&server, now_plus(3600)).await;
        models_mock(catalog_body()).expect(1).mount(&server).await;

        let sync = Arc::new(CopilotCatalogSync::with_owner(
            manager,
            test_client(),
            "copilot-main".to_string(),
        ));
        sync.refresh_if_due().await;

        // Req 5.2: only the enabled entry survives the filter; the
        // picker-gated, policy-gated, and flag-missing entries are
        // excluded.
        let models = sync.enabled_models().await;
        assert_eq!(
            models.iter().map(|m| m.id.as_str()).collect::<Vec<_>>(),
            vec!["gpt-5-codex"],
            "only model_picker_enabled (+policy-permitting) entries are exposed"
        );
        assert_eq!(models[0].owned_by, "copilot-main");
        assert_eq!(models[0].object, "model");
        assert_eq!(models[0].created, Some(1_700_000_000));

        let status = sync.status().await;
        assert!(!status.fetch_failed);
        assert_eq!(status.model_count, 1);
        assert!(status.last_success_at.is_some());

        // Req 5.1: the fetch used the Copilot bearer from the manager.
        let models_requests: Vec<_> = server
            .received_requests()
            .await
            .unwrap()
            .into_iter()
            .filter(|r| r.url.path() == "/models")
            .collect();
        assert_eq!(models_requests.len(), 1);
        let auth = models_requests[0]
            .headers
            .get("authorization")
            .and_then(|v| v.to_str().ok())
            .expect("authorization header present");
        assert_eq!(auth, "Bearer tid=test-copilot-bearer");

        // Existing cadence: a second TTL-gated refresh is a no-op.
        sync.refresh_if_due().await;
        assert_eq!(count_models_requests(&server).await, 1);
    }

    // ---- Req 5.1: unauthenticated → no fetch ----------------------

    #[tokio::test]
    async fn unauthenticated_session_performs_no_catalog_fetch() {
        let server = MockServer::start().await;
        // No login mocks: the manager stays Unauthenticated.
        let manager = Arc::new(CopilotTokenManager::with_endpoints(
            unique_store(),
            test_client(),
            endpoints(&server),
        ));
        // Mount the catalog endpoint so ANY unexpected fetch would be
        // observable in received_requests (and would fail the count
        // assertion below).
        models_mock(catalog_body()).mount(&server).await;

        let sync = CopilotCatalogSync::new(manager, test_client());
        sync.refresh_now().await;
        sync.refresh_if_due().await;

        assert_eq!(
            count_models_requests(&server).await,
            0,
            "unauthenticated session must not fetch the catalog"
        );
        assert!(sync.enabled_models().await.is_empty());
        let status = sync.status().await;
        assert!(
            !status.fetch_failed,
            "unauthenticated skip is not a fetch failure"
        );
        assert!(status.last_success_at.is_none());
    }

    // ---- Req 5.4: failure → last known retained + warning flag ----

    #[tokio::test]
    async fn fetch_failure_retains_last_known_catalog_and_sets_warning() {
        let server = MockServer::start().await;
        let manager = logged_in_manager(&server, now_plus(3600)).await;

        // wiremock matches the earliest mounted mock first: the success
        // mock serves exactly one request, then the 500 takes over for
        // the failure phase.
        models_mock(catalog_body())
            .up_to_n_times(1)
            .mount(&server)
            .await;
        Mock::given(method("GET"))
            .and(path("/models"))
            .respond_with(ResponseTemplate::new(500).set_body_string("boom"))
            .mount(&server)
            .await;

        let sync = CopilotCatalogSync::new(manager, test_client());

        // Phase 1: successful fetch populates the cache.
        sync.refresh_now().await;
        assert_eq!(sync.enabled_models().await.len(), 1);
        assert!(!sync.status().await.fetch_failed);

        // Phase 2: failing fetch keeps the last known catalog and sets
        // the Req 5.4 warning flag.
        sync.refresh_now().await;
        assert_eq!(
            sync.enabled_models()
                .await
                .iter()
                .map(|m| m.id.as_str())
                .collect::<Vec<_>>(),
            vec!["gpt-5-codex"],
            "last known catalog must keep being served on failure"
        );
        let status = sync.status().await;
        assert!(status.fetch_failed, "warning flag must be set on failure");
        assert!(status
            .last_error
            .as_deref()
            .is_some_and(|e| e.contains("500")));
        assert_eq!(status.model_count, 1);
        assert!(status.last_success_at.is_some());
    }

    // ---- Req 5.3: configured-model-absent warning -----------------

    #[tokio::test]
    async fn configured_models_absent_from_catalog_surface_warning() {
        let server = MockServer::start().await;
        let manager = logged_in_manager(&server, now_plus(3600)).await;
        models_mock(serde_json::json!({
            "data": [
                { "id": "known-a", "model_picker_enabled": true },
                { "id": "known-b", "model_picker_enabled": true },
                { "id": "known-but-gated", "model_picker_enabled": false }
            ]
        }))
        .expect(1)
        .mount(&server)
        .await;

        let sync = CopilotCatalogSync::new(manager, test_client());
        sync.set_configured_models(vec![
            "known-a".to_string(),
            "ghost-model".to_string(),
            "known-but-gated".to_string(),
        ])
        .await;
        sync.refresh_now().await;

        // The absent check runs against the FULL fetched catalog (the
        // gated-but-present entry is not "absent"); only the truly
        // unknown configured model warns.
        let status = sync.status().await;
        assert_eq!(
            status.unknown_configured_models,
            vec!["ghost-model".to_string()],
            "only configured models absent from the fetched catalog warn"
        );

        // Enabled list stays the Req 5.2 filtered subset.
        assert_eq!(
            sync.enabled_models()
                .await
                .iter()
                .map(|m| m.id.as_str())
                .collect::<Vec<_>>(),
            vec!["known-a", "known-b"]
        );
    }

    // ---- filter unit tests (no network) ---------------------------

    #[test]
    fn is_enabled_matrix() {
        let mut entry = CopilotCatalogEntry {
            id: "m".to_string(),
            model_picker_enabled: true,
            created: None,
            policy: None,
        };
        assert!(entry.is_enabled());

        entry.model_picker_enabled = false;
        assert!(!entry.is_enabled());

        entry.model_picker_enabled = true;
        entry.policy = Some(CopilotModelPolicy { state: None });
        assert!(entry.is_enabled(), "policy without state does not gate");

        entry.policy = Some(CopilotModelPolicy {
            state: Some("enabled".to_string()),
        });
        assert!(entry.is_enabled());

        entry.policy = Some(CopilotModelPolicy {
            state: Some("off".to_string()),
        });
        assert!(!entry.is_enabled());
    }

    #[test]
    fn entries_missing_model_picker_flag_default_disabled() {
        let entry: CopilotCatalogEntry = serde_json::from_str(r#"{"id":"x"}"#).unwrap();
        assert!(!entry.model_picker_enabled);
        assert!(!entry.is_enabled());
    }
}
