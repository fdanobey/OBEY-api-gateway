//! Provider metadata scanner (model-deprecation-alerts spec,
//! design "Component 1").
//!
//! [`ModelMetadataScanner`] fetches each configured OpenAI-compatible
//! provider's `GET {normalized_base}/models` listing on a 24-hour
//! cadence (plus one scan shortly after startup) and on manual admin
//! trigger, parses per-model deprecation fields, and hands the
//! results to [`DeprecationState::record_scan`]. Providers whose
//! type has no models endpoint (Bedrock, the SystemOne decision
//! family) or whose auth is OAuth-scoped away from `/models` (Codex
//! OAuth OpenAI, GitHub Copilot) are skipped with a `debug` log.
//!
//! Failure discipline (Req 1.3): any per-provider failure — network,
//! timeout, non-200, body/parse error, SSRF-guard rejection — keeps
//! the provider's previous scan data (the scanner simply does not
//! call `record_scan`), logs exactly ONE `warn` carrying the
//! provider name and an error class (never the URL, never
//! credentials, never bodies), and never fails the scan loop.
//!
//! Config-read discipline (delivery.rs pattern): the provider
//! snapshot (names, base URLs, resolved credentials, provider type)
//! and the outbound-fetch guard config are cloned under the config
//! read lock, and the lock is DROPPED before any network I/O — it is
//! never held across an await.
//!
//! Single-flight guard (Req 1.5): an in-flight [`AtomicBool`] claim
//! makes a manual `scan_all` while another scan is running return a
//! [`ScanSummary`] marked `already_running` instead of starting a
//! second concurrent scan.
//!
//! _Requirements: 1.1, 1.1a, 1.2, 1.3, 1.5_

use std::collections::HashMap;
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::Arc;
use std::time::Duration;

use chrono::{DateTime, NaiveDate, Utc};
use dashmap::DashMap;
use tokio::sync::{RwLock, watch};

use crate::config::Config;
use crate::logger::RequestLogger;
use crate::model_deprecations::emission::spawn_deprecation_emission;
use crate::model_deprecations::state::{DeprecationState, ScanModel};
use crate::security::outbound_fetch::{
    self, OutboundFetchConfig, std_resolve_host, validate_outbound_url,
};
use crate::webhooks::delivery::WebhookNotifier;

/// Per-provider fetch timeout (connect + headers + body), matching
/// the webhook delivery attempt budget (Req 1.1).
const SCAN_TIMEOUT: Duration = Duration::from_secs(10);

/// Background cadence between full scan passes (Req 1.1: 24 hours).
const SCAN_CADENCE: Duration = Duration::from_secs(86_400);

/// Delay before the startup scan (Req 1.1a): shortly after boot, not
/// 24 hours — past the busiest startup moment, well before the first
/// cadence tick.
const STARTUP_SCAN_DELAY: Duration = Duration::from_secs(15);

/// Models-listing body size ceiling: a Content-Length over the cap
/// fails the scan before reading, and the streamed read aborts past
/// it (`body_too_large` → the stale-keep/failed path, Req 1.3) —
/// the scanner never reads an unbounded body.
const MODELS_BODY_CAP_BYTES: usize = 2 * 1024 * 1024;

/// Static provider-type map (design D11): provider types whose
/// `{base}/models` endpoint serves an OpenAI-compatible listing
/// behind plain Bearer auth. Types with different auth shapes
/// (anthropic: `x-api-key` + `anthropic-version` headers; azure:
/// `api-key` header + `api-version` query parameter; google:
/// key-as-query-parameter), OAuth-scoped types (Codex via `openai` +
/// `auth_method: oauth`, `github-copilot`), Bedrock (no compatible
/// listing), and the SystemOne decision family (`jev`/`laya`/
/// `systemone`, non-OpenAI wire protocol) are NOT scanned.
const SCANNED_PROVIDER_TYPES: &[&str] = &[
    "openai",
    "custom",
    "groq",
    "together",
    "nvidia_nim",
    "vllm",
    "lmstudio",
    "ollama",
];

/// Why a configured provider is not scanned (static map outcome).
#[derive(Debug, Clone, Copy, PartialEq, Eq, serde::Serialize)]
#[serde(rename_all = "snake_case")]
pub enum ScanSkipReason {
    /// Provider type has no OpenAI-compatible models endpoint
    /// (bedrock, jev/laya/systemone).
    NoModelsEndpoint,
    /// Credentials are OAuth-scoped away from `/models` (Codex OAuth
    /// OpenAI, GitHub Copilot).
    OauthScoped,
    /// Provider type is not in the static scan map.
    UnsupportedType,
}

/// One provider's outcome in a [`ScanSummary`] pass.
#[derive(Debug, Clone, PartialEq, serde::Serialize)]
#[serde(tag = "status", rename_all = "snake_case")]
pub enum ScanOutcome {
    /// Listing fetched, parsed, and recorded into the state core.
    Scanned {
        /// Number of models in the parsed listing.
        models: usize,
    },
    /// Provider type not scanned (static map exclusion).
    Skipped {
        reason: ScanSkipReason,
    },
    /// Scan attempt failed and no previous scan data existed for the
    /// provider (nothing to keep).
    Failed {
        error_class: String,
    },
    /// Scan attempt failed; the provider's previous scan data was
    /// kept untouched (Req 1.3 stale-keep).
    StaleKept {
        error_class: String,
    },
}

/// Per-provider row of the admin scan view.
#[derive(Debug, Clone, PartialEq, serde::Serialize)]
pub struct ProviderScanResult {
    pub provider: String,
    #[serde(flatten)]
    pub outcome: ScanOutcome,
}

/// Result of one `scan_all` pass (manual trigger return value and
/// background-loop log input).
#[derive(Debug, Clone, PartialEq, serde::Serialize)]
pub struct ScanSummary {
    /// `true` when another scan was in flight and this call did NOT
    /// start a second one (Req 1.5).
    pub already_running: bool,
    /// Pass start time.
    pub started_at: DateTime<Utc>,
    /// Per-provider outcomes, in configured provider order.
    pub providers: Vec<ProviderScanResult>,
}

/// Scan eligibility from the static provider-type map (design D11).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum ScanEligibility {
    Scan,
    NoModelsEndpoint,
    OauthScoped,
    UnsupportedType,
}

/// Config snapshot for one provider, cloned under the config read
/// lock so no lock is held across the fetch awaits.
struct ProviderSnapshot {
    name: String,
    base_url: Option<String>,
    api_key: Option<String>,
    eligibility: ScanEligibility,
}

/// Releases the in-flight claim on drop (panic-safe; also covers
/// early returns inside the scan loop).
struct InFlightGuard<'a>(&'a AtomicBool);

impl Drop for InFlightGuard<'_> {
    fn drop(&mut self) {
        self.0.store(false, Ordering::Release);
    }
}

/// Provider metadata scanner (design "Component 1"). Clones share
/// the once-built HTTP client (10 s timeout, redirects disabled),
/// the state core, and the in-flight claim, so `AppState` can hold
/// one `Arc` for both the background loop and the manual admin
/// trigger.
pub struct ModelMetadataScanner {
    config: Arc<RwLock<Config>>,
    state: Arc<DeprecationState>,
    /// Emission dependencies (Task 9): the shared logger (usage
    /// evidence) and notifier (event fan-out) Arcs from
    /// `GatewayServer::new`, so every completed scan pass fires
    /// `spawn_deprecation_emission` (fire-and-forget — the fn
    /// spawns internally). `None` leaves the scanner buildable in
    /// isolation tests; a pass then skips emission.
    emission: Option<(
        Arc<RequestLogger>,
        Arc<WebhookNotifier>,
    )>,
    /// Built once: per-request 10 s timeout, redirects disabled (a
    /// 3xx surfaces as a failed attempt — no redirect hops to
    /// re-validate, so the pre-fetch SSRF check covers the exchange).
    client: reqwest::Client,
    /// `true` when client construction failed at startup: every
    /// per-provider fetch fails fast locally with the
    /// `scanner_unavailable` error class (no unguarded default
    /// client — the delivery.rs degraded-mode discipline).
    degraded: bool,
    /// Single-flight claim (Req 1.5): compare_exchange claims a scan
    /// pass; a second `scan_all` while held returns `already_running`.
    scan_in_flight: AtomicBool,
    /// Per-provider last successful scan timestamp (admin
    /// `last_scan_info` view + stale-kept-vs-failed distinction).
    last_success: DashMap<String, DateTime<Utc>>,
}

impl ModelMetadataScanner {
    /// Build the scanner around the live config and the shared state
    /// core. Client construction failure degrades the scanner (every
    /// scan fails fast locally) instead of falling back to an
    /// unguarded default client.
    pub fn new(config: Arc<RwLock<Config>>, state: Arc<DeprecationState>) -> Self {
        Self::with_emission(config, state, None)
    }

    /// [`Self::new`] plus the emission dependencies (Task 9): every
    /// completed `scan_all` pass (background cadence and manual admin
    /// trigger alike) fires `spawn_deprecation_emission` with these
    /// Arcs — the production wiring from `GatewayServer::new`.
    pub fn with_emission(
        config: Arc<RwLock<Config>>,
        state: Arc<DeprecationState>,
        emission: Option<(Arc<RequestLogger>, Arc<WebhookNotifier>)>,
    ) -> Self {
        let client = reqwest::Client::builder()
            .timeout(SCAN_TIMEOUT)
            .redirect(reqwest::redirect::Policy::none())
            .build();
        let (client, degraded) = match client {
            Ok(client) => (client, false),
            Err(error) => {
                tracing::warn!(
                    error = %error,
                    "model deprecation scanner client construction failed; scans disabled until restart"
                );
                (reqwest::Client::new(), true)
            }
        };
        Self {
            config,
            state,
            emission,
            client,
            degraded,
            scan_in_flight: AtomicBool::new(false),
            last_success: DashMap::new(),
        }
    }

    /// One full scan pass over every configured provider (Req 1.1).
    /// The background loop and the manual admin trigger call the
    /// same method. Never fails: per-provider failures degrade to
    /// `Failed`/`StaleKept` rows. While another pass is in flight,
    /// returns `already_running` without starting a second scan
    /// (Req 1.5).
    pub async fn scan_all(&self) -> ScanSummary {
        let started_at = Utc::now();

        // Single-flight claim: a manual trigger during a running scan
        // reports already_running without touching state or network.
        if self
            .scan_in_flight
            .compare_exchange(false, true, Ordering::AcqRel, Ordering::Relaxed)
            .is_err()
        {
            return ScanSummary {
                already_running: true,
                started_at,
                providers: Vec::new(),
            };
        }
        let _in_flight = InFlightGuard(&self.scan_in_flight);

        // Config snapshot under the read lock; the guard drops at the
        // end of this block, before any network I/O (delivery.rs
        // discipline — the lock is never held across an await).
        let (outbound_config, snapshots) = {
            let config = self.config.read().await;
            let outbound_config = config.security.outbound_fetch.clone();
            let snapshots = config
                .providers
                .iter()
                .map(|provider| ProviderSnapshot {
                    name: provider.name.clone(),
                    base_url: provider.base_url.clone(),
                    api_key: provider.resolve_api_key(),
                    eligibility: scan_eligibility(
                        &provider.provider_type,
                        provider.auth_method.as_deref(),
                    ),
                })
                .collect::<Vec<_>>();
            (outbound_config, snapshots)
        };

        let mut providers = Vec::with_capacity(snapshots.len());
        for snapshot in &snapshots {
            let outcome = self.scan_provider(&outbound_config, snapshot).await;
            providers.push(ProviderScanResult {
                provider: snapshot.name.clone(),
                outcome,
            });
        }

        // Scanner-cadence full sweep: evict runtime-gone marks past
        // their 24-hour bound so the registry stays bounded (state.rs
        // eviction contract).
        self.state.evict_expired(Utc::now());

        // Task 9 production emission wiring: every completed pass
        // (background cadence AND manual admin trigger — both call
        // this method) evaluates episodes and emits resulting
        // model.deprecated events. Fire-and-forget: the fn spawns
        // internally, never blocks the pass; `None` emission deps
        // (isolation-built scanner) skip it.
        if let Some((logger, notifier)) = &self.emission {
            spawn_deprecation_emission(
                Arc::clone(&self.config),
                Arc::clone(&self.state),
                Arc::clone(logger),
                Arc::clone(notifier),
            );
        }

        ScanSummary {
            already_running: false,
            started_at,
            providers,
        }
    }

    /// Per-provider scan (Req 1.2/1.3): fetch `{normalized_base}/models`,
    /// parse the deprecation fields, and record the listing into the
    /// state core on success. Any failure keeps the provider's
    /// previous state and logs exactly one `warn` with the provider
    /// name and error class only.
    async fn scan_provider(
        &self,
        outbound_config: &OutboundFetchConfig,
        snapshot: &ProviderSnapshot,
    ) -> ScanOutcome {
        match snapshot.eligibility {
            ScanEligibility::Scan => {}
            ScanEligibility::OauthScoped => {
                tracing::debug!(
                    provider = %snapshot.name,
                    "model deprecation scan skipped: OAuth-scoped provider type has no models endpoint"
                );
                return ScanOutcome::Skipped {
                    reason: ScanSkipReason::OauthScoped,
                };
            }
            ScanEligibility::NoModelsEndpoint => {
                tracing::debug!(
                    provider = %snapshot.name,
                    "model deprecation scan skipped: provider type has no models endpoint"
                );
                return ScanOutcome::Skipped {
                    reason: ScanSkipReason::NoModelsEndpoint,
                };
            }
            ScanEligibility::UnsupportedType => {
                tracing::debug!(
                    provider = %snapshot.name,
                    "model deprecation scan skipped: provider type not in the scan map"
                );
                return ScanOutcome::Skipped {
                    reason: ScanSkipReason::UnsupportedType,
                };
            }
        }

        let models = match self.fetch_models(outbound_config, snapshot).await {
            Ok(models) => models,
            Err(error_class) => return self.apply_failed_scan(&snapshot.name, error_class),
        };

        let model_count = models.len();
        self.state.record_scan(&snapshot.name, models);
        self.last_success.insert(snapshot.name.clone(), Utc::now());
        ScanOutcome::Scanned {
            models: model_count,
        }
    }

    /// Failure path (Req 1.3): one `warn` (provider name + error
    /// class only — never URLs with credentials, never bodies), previous
    /// scan data kept by simply not calling `record_scan`. The
    /// outcome distinguishes `stale_kept` (previous data exists) from
    /// `failed` (nothing to keep).
    fn apply_failed_scan(&self, provider: &str, error_class: String) -> ScanOutcome {
        tracing::warn!(
            provider = %provider,
            error_class = %error_class,
            "model deprecation scan failed; keeping previous scan data"
        );
        if self.last_success.contains_key(provider) {
            ScanOutcome::StaleKept { error_class }
        } else {
            ScanOutcome::Failed { error_class }
        }
    }

    /// Fetch and parse one provider's models listing. Every error is
    /// returned as a compact error class string (no URLs, no bodies,
    /// no credentials — log-safe by construction).
    async fn fetch_models(
        &self,
        outbound_config: &OutboundFetchConfig,
        snapshot: &ProviderSnapshot,
    ) -> Result<Vec<ScanModel>, String> {
        if self.degraded {
            return Err("scanner_unavailable".to_string());
        }

        // Normalize the base URL exactly as the router does for
        // dispatch traffic: strip trailing '/', append '/v1' when
        // missing, then target '{base}/models'.
        let Some(raw_base) = snapshot
            .base_url
            .as_deref()
            .map(str::trim)
            .filter(|base| !base.is_empty())
        else {
            return Err("invalid_base_url".to_string());
        };
        let mut base = raw_base.trim_end_matches('/').to_string();
        if !base.ends_with("/v1") {
            base.push_str("/v1");
        }
        let url = reqwest::Url::parse(&format!("{base}/models"))
            .map_err(|_| "invalid_base_url".to_string())?;

 // Caller-tier SSRF guard (Req 1.1) before any bytes leave:
    // rejection carries host + reason only. Non-literal-IP hosts
    // resolve DNS on the blocking pool (`std_resolve_host` is a
    // blocking `ToSocketAddrs` call — never on a tokio worker). The
    // single per-provider `warn` for this failure comes from
    // `apply_failed_scan` (Req 1.3 one-warn discipline).
    if validate_url_guarded(&url, outbound_config.clone())
        .await
        .is_err()
    {
        return Err("ssrf_guard_rejected".to_string());
    }

    // Same credential resolution the router uses (env-var-first
    // via `Provider::resolve_api_key`); an unresolvable key
    // simply sends no Authorization header (public listings like
    // Ollama still scan).
    let mut request = self.client.get(url);
    if let Some(key) = snapshot.api_key.as_deref().filter(|key| !key.is_empty()) {
        request = request.header("authorization", format!("Bearer {key}"));
    }

    let response = request
        .send()
        .await
        .map_err(|error| reqwest_error_class(&error))?;
    let status = response.status();
    if !status.is_success() {
        return Err(format!("http_{}", status.as_u16()));
    }
    // Body size ceiling: a declared Content-Length over the cap
    // fails the scan before reading anything.
    if response
        .headers()
        .get(reqwest::header::CONTENT_LENGTH)
        .and_then(|value| value.to_str().ok())
        .and_then(|value| value.trim().parse::<u64>().ok())
        .is_some_and(|length| length > MODELS_BODY_CAP_BYTES as u64)
    {
        return Err("body_too_large".to_string());
    }
    // Chunked read with the same cap: aborts (and fails the scan)
    // once the body exceeds it — no unbounded `text()` read.
    let mut response = response;
    let mut bytes = Vec::new();
    loop {
        let chunk = response
            .chunk()
            .await
            .map_err(|_| "body_read".to_string())?
            .unwrap_or_default();
        if chunk.is_empty() {
            break;
        }
        if bytes.len() + chunk.len() > MODELS_BODY_CAP_BYTES {
            return Err("body_too_large".to_string());
        }
        bytes.extend_from_slice(&chunk);
    }
    let body = String::from_utf8_lossy(&bytes);
    let value: serde_json::Value =
        serde_json::from_str(&body).map_err(|_| "parse".to_string())?;
    parse_models_listing(&value).ok_or_else(|| "parse".to_string())
    }

    /// Per-provider last successful scan timestamps (admin status
    /// view input); providers never scanned are absent.
    pub fn last_scan_info(&self) -> HashMap<String, DateTime<Utc>> {
        self.last_success
            .iter()
            .map(|entry| (entry.key().clone(), *entry.value()))
            .collect()
    }

    /// `true` while a scan pass holds the single-flight claim (Req
    /// 1.5): the admin manual trigger checks this BEFORE spawning so
    /// a BACKGROUND scan in flight answers `already_running`
    /// immediately instead of starting a doomed-to-lose task.
    pub fn scan_in_flight(&self) -> bool {
        self.scan_in_flight.load(Ordering::Acquire)
    }

    /// Spawn the background cadence loop (Req 1.1/1.1a): one scan
    /// shortly after startup, then every 24 hours, honoring the
    /// shutdown token (the memory decay-scheduler `watch` pattern).
    /// The returned handle lets the caller abort the task; the loop
    /// itself exits when the token fires or its sender drops. Not
    /// wired into `GatewayServer` here — later spec task.
    pub fn spawn_background(
        self: &Arc<Self>,
        mut shutdown: watch::Receiver<bool>,
    ) -> tokio::task::JoinHandle<()> {
        let scanner = Arc::clone(self);
        tokio::spawn(async move {
            // First tick fires at STARTUP_SCAN_DELAY (the startup
            // scan), subsequent ticks at the 24-hour cadence.
            let mut interval = tokio::time::interval_at(
                tokio::time::Instant::now() + STARTUP_SCAN_DELAY,
                SCAN_CADENCE,
            );
            interval.set_missed_tick_behavior(tokio::time::MissedTickBehavior::Delay);
            loop {
                tokio::select! {
                    changed = shutdown.changed() => {
                        if changed.is_err() || *shutdown.borrow() {
                            break;
                        }
                    }
                    _ = interval.tick() => {
                        let summary = scanner.scan_all().await;
                        tracing::info!(
                            already_running = summary.already_running,
                            providers = summary.providers.len(),
                            "model deprecation scan pass complete"
                        );
                    }
                }
            }
            tracing::debug!("model deprecation scanner background loop stopped");
        })
    }
}

/// Static provider-type map lookup (design D11). Codex OAuth is
/// `openai` + `auth_method: oauth` (`is_codex_oauth_provider`
/// discriminator).
fn scan_eligibility(provider_type: &str, auth_method: Option<&str>) -> ScanEligibility {
    match provider_type {
        "github-copilot" => ScanEligibility::OauthScoped,
        "bedrock" | "jev" | "laya" | "systemone" => ScanEligibility::NoModelsEndpoint,
        provider_type if SCANNED_PROVIDER_TYPES.contains(&provider_type) => {
            if auth_method == Some("oauth") {
                ScanEligibility::OauthScoped
            } else {
                ScanEligibility::Scan
            }
        }
        _ => ScanEligibility::UnsupportedType,
    }
}

/// Caller-tier SSRF validation with blocking-pool DNS for hostname
/// targets (webhooks delivery.rs discipline): literal-IP hosts skip
/// DNS (cheap, no I/O); everything else resolves on the blocking
/// pool.
async fn validate_url_guarded(
    url: &reqwest::Url,
    outbound_config: OutboundFetchConfig,
) -> Result<(), outbound_fetch::OutboundFetchError> {
    let host = url.host_str().unwrap_or_default();
    if host.parse::<std::net::IpAddr>().is_ok() {
        return validate_outbound_url(url, &outbound_config, &std_resolve_host);
    }
    let host = host.to_string();
    let url = url.clone();
    tokio::task::spawn_blocking(move || {
        validate_outbound_url(&url, &outbound_config, &std_resolve_host)
    })
    .await
    .map_err(|error| {
        tracing::warn!(error = %error, "model deprecation scan SSRF validation task failed");
        outbound_fetch::OutboundFetchError::UnresolvableHost { host }
    })?
}

/// Compact transport error class for the failure warn log and the
/// scan summary (never the error display — reqwest messages can
/// carry URL fragments).
fn reqwest_error_class(error: &reqwest::Error) -> String {
    if error.is_timeout() {
        "timeout".to_string()
    } else if error.is_connect() {
        "connect".to_string()
    } else {
        "network".to_string()
    }
}

/// Parse an OpenAI-shaped models listing (Req 1.2): `{"data":[...]}`or a bare top-level array. Per entry: `id` (string; entries
/// without one are silently skipped — unknown shapes are not
/// deprecated by default), retirement date with precedence
/// `shutdown_date` > `expiration_date` > `deprecation_date` (first
/// present AND parseable ISO 8601 date or RFC 3339 datetime wins),
/// `deprecated` (bool), and `successor_model` (string). Returns
/// `None` on a structurally invalid listing (missing/non-array
/// `data` on an object body, or a non-object/non-array body).
fn parse_models_listing(value: &serde_json::Value) -> Option<Vec<ScanModel>> {
    let entries = match value {
        serde_json::Value::Array(entries) => entries.as_slice(),
        serde_json::Value::Object(map) => match map.get("data") {
            Some(serde_json::Value::Array(entries)) => entries.as_slice(),
            _ => return None,
        },
        _ => return None,
    };

    let mut models = Vec::with_capacity(entries.len());
    for entry in entries {
        let Some(id) = entry.get("id").and_then(|value| value.as_str()) else {
            continue;
        };
        if id.is_empty() {
            continue;
        }
        // First PARSEABLE date wins: a present-but-unparseable
        // higher-precedence field (e.g. garbage `shutdown_date`)
        // falls through so `expiration_date`/`deprecation_date`
        // can still supply the retirement date.
        let retirement_date = ["shutdown_date", "expiration_date", "deprecation_date"]
            .iter()
            .find_map(|field| {
                entry
                    .get(*field)
                    .and_then(|value| value.as_str())
                    .and_then(parse_model_date)
            });
        let replacement = entry
            .get("successor_model")
            .and_then(|value| value.as_str())
            .filter(|value| !value.is_empty())
            .map(str::to_string);
        let deprecated = entry
            .get("deprecated")
            .and_then(|value| value.as_bool())
            .unwrap_or(false);
        models.push(ScanModel {
            model: id.to_string(),
            retirement_date,
            replacement,
            deprecated,
        });
    }
    Some(models)
}

/// ISO 8601 calendar date (`YYYY-MM-DD`) or RFC 3339 datetime
/// (datetime contributes its calendar date), mirroring the state
/// core's accepted shapes.
fn parse_model_date(value: &str) -> Option<NaiveDate> {
    if let Ok(date) = NaiveDate::parse_from_str(value, "%Y-%m-%d") {
        return Some(date);
    }
    chrono::DateTime::parse_from_rfc3339(value)
        .ok()
        .map(|datetime| datetime.date_naive())
}

#[cfg(test)]
mod tests {
    use super::*;

    fn test_config() -> Config {
        serde_yaml::from_str(
            "server:\n  host: 127.0.0.1\n  port: 8080\nproviders: []\nmodel_groups: []\n",
        )
        .expect("minimal test config must parse")
    }

    fn scanner() -> ModelMetadataScanner {
        ModelMetadataScanner::new(
            Arc::new(RwLock::new(test_config())),
            Arc::new(DeprecationState::default()),
        )
    }

    fn utc(instant: &str) -> DateTime<Utc> {
        chrono::DateTime::parse_from_rfc3339(instant)
            .expect("test timestamp must parse")
            .with_timezone(&Utc)
    }

    // ---- fixture parsing (Req 1.2) ------------------------------------

    #[test]
    fn parse_openai_shape_with_shutdown_date() {
        let value: serde_json::Value = serde_json::from_str(
            r#"{"object":"list","data":[
                {"id":"gpt-x","object":"model","owned_by":"openai","shutdown_date":"2026-12-01"},
                {"id":"gpt-clean","object":"model","owned_by":"openai"}
            ]}"#,
        )
        .expect("fixture must parse");
        let models = parse_models_listing(&value).expect("openai shape must parse");
        assert_eq!(models.len(), 2);
        assert_eq!(
            models[0],
            ScanModel {
                model: "gpt-x".to_string(),
                retirement_date: NaiveDate::from_ymd_opt(2026, 12, 1),
                replacement: None,
                deprecated: false,
            }
        );
        // Clean listing entry: present (re-arms dropped signals) but signal-free.
        assert_eq!(
            models[1],
            ScanModel {
                model: "gpt-clean".to_string(),
                retirement_date: None,
                replacement: None,
                deprecated: false,
            }
        );
    }

    #[test]
    fn parse_openrouter_shape_expiration_and_successor() {
        let value: serde_json::Value = serde_json::from_str(
            r#"{"data":[{"id":"vendor/old-model","expiration_date":"2026-11-20","successor_model":"vendor/new-model"}]}"#,
        )
        .expect("fixture must parse");
        let models = parse_models_listing(&value).expect("openrouter shape must parse");
        assert_eq!(
            models,
            vec![ScanModel {
                model: "vendor/old-model".to_string(),
                retirement_date: NaiveDate::from_ymd_opt(2026, 11, 20),
                replacement: Some("vendor/new-model".to_string()),
                deprecated: false,
            }]
        );
    }

    #[test]
    fn parse_cloudflare_shape_deprecated_flag_only_is_date_less() {
        let value: serde_json::Value = serde_json::from_str(
            r#"{"data":[{"id":"@cf/mistral-7b","deprecated":true}]}"#,
        )
        .expect("fixture must parse");
        let models = parse_models_listing(&value).expect("cloudflare shape must parse");
        assert_eq!(
            models,
            vec![ScanModel {
                model: "@cf/mistral-7b".to_string(),
                retirement_date: None,
                replacement: None,
                deprecated: true,
            }]
        );
    }

    #[test]
    fn parse_bare_top_level_array() {
        let value: serde_json::Value =
            serde_json::from_str(r#"[{"id":"m1","deprecation_date":"2026-11-15"}]"#)
                .expect("fixture must parse");
        let models = parse_models_listing(&value).expect("bare array must parse");
        assert_eq!(models.len(), 1);
        assert_eq!(models[0].retirement_date, NaiveDate::from_ymd_opt(2026, 11, 15));
    }

    #[test]
    fn parse_accepts_date_and_datetime_forms() {
        assert_eq!(
            parse_model_date("2026-08-15"),
            NaiveDate::from_ymd_opt(2026, 8, 15)
        );
        assert_eq!(
            parse_model_date("2026-08-15T14:30:00Z"),
            NaiveDate::from_ymd_opt(2026, 8, 15)
        );
        assert_eq!(parse_model_date("not-a-date"), None);
    }

    #[test]
    fn parse_date_precedence_shutdown_expiration_deprecation() {
        let value: serde_json::Value = serde_json::from_str(
            r#"{"data":[{"id":"m","shutdown_date":"2027-01-01","expiration_date":"2027-02-01","deprecation_date":"2027-03-01"}]}"#,
        )
        .expect("fixture must parse");
        let models = parse_models_listing(&value).expect("fixture must parse");
        assert_eq!(models[0].retirement_date, NaiveDate::from_ymd_opt(2027, 1, 1));

        // shutdown_date absent → expiration_date wins.
        let value: serde_json::Value = serde_json::from_str(
            r#"{"data":[{"id":"m","expiration_date":"2027-02-01","deprecation_date":"2027-03-01"}]}"#,
        )
        .expect("fixture must parse");
        let models = parse_models_listing(&value).expect("fixture must parse");
        assert_eq!(models[0].retirement_date, NaiveDate::from_ymd_opt(2027, 2, 1));
    }

    #[test]
    fn parse_structurally_invalid_listings_fail() {
        // Object without a data array.
        let value: serde_json::Value =
            serde_json::from_str(r#"{"object":"list"}"#).expect("fixture must parse");
        assert!(parse_models_listing(&value).is_none());
        // data present but not an array.
        let value: serde_json::Value =
            serde_json::from_str(r#"{"data":{"id":"m"}}"#).expect("fixture must parse");
        assert!(parse_models_listing(&value).is_none());
        // Scalar body.
        let value: serde_json::Value = serde_json::Value::String("nope".to_string());
        assert!(parse_models_listing(&value).is_none());
    }

    #[test]
    fn parse_skips_entries_without_id() {
        let value: serde_json::Value = serde_json::from_str(
            r#"[{"owned_by":"x"},{"id":""},{"id":"ok"}]"#,
        )
        .expect("fixture must parse");
        let models = parse_models_listing(&value).expect("bare array must parse");
        assert_eq!(models.len(), 1);
        assert_eq!(models[0].model, "ok");
    }

    // ---- static provider-type map (Req 1.2, design D11) ---------------

#[test]
fn eligibility_scans_openai_compatible_types() {
    for provider_type in [
        "openai",
        "custom",
        "groq",
        "together",
        "nvidia_nim",
        "vllm",
        "lmstudio",
        "ollama",
    ] {
        assert_eq!(
            scan_eligibility(provider_type, None),
            ScanEligibility::Scan,
            "{provider_type} must be scanned"
        );
    }
}

#[test]
fn eligibility_excludes_oauth_scoped_and_endpointless_types() {
    // Codex OAuth: openai + auth_method oauth.
    assert_eq!(
        scan_eligibility("openai", Some("oauth")),
        ScanEligibility::OauthScoped
    );
    assert_eq!(
        scan_eligibility("github-copilot", None),
        ScanEligibility::OauthScoped
    );
    // No models endpoint / non-OpenAI wire protocol.
    for provider_type in ["bedrock", "jev", "laya", "systemone"] {
        assert_eq!(
            scan_eligibility(provider_type, None),
            ScanEligibility::NoModelsEndpoint,
            "{provider_type} must not be scanned"
        );
    }
    // Non-Bearer auth shapes: the plain Bearer fetch cannot talk to
    // their `/models` endpoints (x-api-key / api-key header /
    // api-version or key query parameters) — excluded rather than
    // failing every pass.
    for provider_type in ["anthropic", "azure", "google"] {
        assert_eq!(
            scan_eligibility(provider_type, None),
            ScanEligibility::UnsupportedType,
            "{provider_type} uses a non-Bearer auth shape and must not be scanned"
        );
    }
    // Unknown types default to not scanned.
    assert_eq!(
        scan_eligibility("martian", None),
        ScanEligibility::UnsupportedType
    );
}

    // ---- failure discipline (Req 1.3) ----------------------------------

    #[test]
    fn non_200_and_parse_failures_keep_previous_state() {
        let scanner = scanner();
        let state = DeprecationState::default();
        // Seed prior scan data: a model retiring tomorrow.
        state.record_scan(
            "openai",
            vec![ScanModel {
                model: "gpt-x".to_string(),
                retirement_date: NaiveDate::from_ymd_opt(2026, 11, 15),
                replacement: None,
                deprecated: false,
            }],
        );
        scanner
            .last_success
            .insert("openai".to_string(), utc("2026-11-13T00:00:00Z"));

        let now = utc("2026-11-14T00:00:00Z");
        let usage = |_: &str, _: &str| Some((5u64, None));

        // The seeded signal qualifies pre-failure.
        assert_eq!(state.status_rows(&[], usage, 30, now).len(), 1);

        // Non-200 failure: StaleKept, previous state untouched.
        let outcome = scanner.apply_failed_scan("openai", "http_503".to_string());
        assert_eq!(
            outcome,
            ScanOutcome::StaleKept {
                error_class: "http_503".to_string()
            }
        );
        assert_eq!(state.status_rows(&[], usage, 30, now).len(), 1);

        // Parse failure: same stale-keep discipline.
        let outcome = scanner.apply_failed_scan("openai", "parse".to_string());
        assert_eq!(
            outcome,
            ScanOutcome::StaleKept {
                error_class: "parse".to_string()
            }
        );
        assert_eq!(state.status_rows(&[], usage, 30, now).len(), 1);
    }

    #[test]
    fn failure_without_prior_success_reports_failed_not_stale_kept() {
        let scanner = scanner();
        let outcome = scanner.apply_failed_scan("fresh", "connect".to_string());
        assert_eq!(
            outcome,
            ScanOutcome::Failed {
                error_class: "connect".to_string()
            }
        );
    }

    // ---- single-flight guard (Req 1.5) ---------------------------------

    #[tokio::test]
    async fn scan_all_returns_already_running_while_in_flight() {
        let scanner = scanner();
        scanner.scan_in_flight.store(true, Ordering::SeqCst);
        let summary = scanner.scan_all().await;
        assert!(summary.already_running);
        assert!(summary.providers.is_empty());
        // The claim was not stolen: the running scan still owns it.
        assert!(scanner.scan_in_flight.load(Ordering::SeqCst));
    }

    // ---- background loop shutdown token ---------------------------------

    #[tokio::test]
    async fn background_loop_stops_on_shutdown_token() {
        let scanner = Arc::new(scanner());
        let (shutdown_tx, shutdown_rx) = watch::channel(false);
        let handle = scanner.spawn_background(shutdown_rx);
        shutdown_tx.send(true).expect("shutdown send must succeed");
        handle.await.expect("background loop must stop cleanly");
    }

    // ---- parse → record_scan integration (pure, no network) ------------

    #[test]
    fn parsed_fixture_feeds_record_scan_and_status_rows() {
        let value: serde_json::Value = serde_json::from_str(
            r#"{"data":[
                {"id":"gpt-x","shutdown_date":"2026-12-01"},
                {"id":"gpt-clean"}
            ]}"#,
        )
        .expect("fixture must parse");
        let models = parse_models_listing(&value).expect("fixture must parse");
        let state = DeprecationState::default();
        state.record_scan("openai", models);

        let now = utc("2026-11-14T00:00:00Z");
        let rows = state.status_rows(&[], |_, _| Some((5, None)), 30, now);
        // Only the retiring model carries a signal; the clean entry
        // was recorded signal-free (episode re-arm input).
        assert_eq!(rows.len(), 1);
        assert_eq!(rows[0].model, "gpt-x");
        assert_eq!(rows[0].retirement_date.as_deref(), Some("2026-12-01"));
    }
}
