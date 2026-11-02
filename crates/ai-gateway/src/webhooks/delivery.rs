//! Webhook delivery engine (budget-alerts-unified-webhooks spec,
//! design "Component 2").
//!
//! [`WebhookNotifier`] fans each [`BudgetAlertEvent`] out to the
//! endpoints subscribed to its type: Standard Webhooks HMAC signing
//! for endpoints with a secret (Req 3.4), unsigned `"signed": false`
//! envelopes for the rest (Req 3.5), bounded retries at 0 s/30 s/2
//! m/10 m (Req 4.2), per-attempt SSRF re-validation with fresh DNS
//! (Req 3.5/4.5 NFR Security), and one ledger row per attempt through
//! the blocking pool (Req 4.3). Redirects are disabled: receivers
//! must answer directly with 200-299 (a 3xx fails the attempt).
//!
//! Config-read discipline (design "Component 2"): every delivery path
//! clones its endpoint/allowlist snapshot under the config read lock
//! and DROPS the lock before any signing or network I/O — the lock is
//! never held across an await.
//!
//! Tracing discipline: outcomes are logged at info/warn with the
//! event id, endpoint URL, attempt number and status/error class
//! only — never bodies, never secrets.
//!
//! _Requirements: 3.2, 3.4, 3.5, 4.1, 4.2, 4.4, 4.5, 4.6, 4.7_

use std::sync::Arc;
use std::time::Duration;

use base64::Engine as _;
use futures::future::BoxFuture;
use tokio::sync::{RwLock, Semaphore};

use crate::config::{BudgetEventType, Config, WebhookEndpoint};
use crate::security::outbound_fetch::{
    self, CallerFetchConfig, OutboundFetchConfig, std_resolve_host, validate_outbound_url,
};
use crate::webhooks::events::{BudgetAlertEvent, Scope, WebhookTestEvent};
use crate::webhooks::ledger::WebhookLedgerStore;

/// Per-attempt total timeout (connect + headers + body). Webhook
/// payloads are tiny JSON envelopes; 10 s matches the admin
/// connection-test client default (`admin/mod.rs` client construction).
const ATTEMPT_TIMEOUT: Duration = Duration::from_secs(10);

/// Retry schedule (Req 4.2, design D6): attempt 1 fires immediately,
/// attempts 2-4 follow after 30 s / 2 m / 10 m. Injected sleeper lets
/// tests compress the schedule to zero.
const RETRY_DELAYS: [Duration; 3] = [
    Duration::from_secs(30),
    Duration::from_secs(120),
    Duration::from_secs(600),
];

/// Maximum delivery attempts per (event, endpoint) chain (Req 4.2:
/// "at most 4 attempts total").
const MAX_ATTEMPTS: i64 = 4;

/// Bound on concurrently running delivery chains (design NFR
/// Performance). Acquired non-blockingly before spawning a chain: a
/// full semaphore logs a warn and records a failed ledger row — emit
/// never blocks (Req 4.1) and never silently drops.
const MAX_CONCURRENT_DELIVERY_CHAINS: usize = 64;

/// Injectable delay seam: production sleeps via `tokio::time::sleep`;
/// tests inject an instant sleeper to compress the 30 s/2 m/10 m
/// schedule (design Testing Strategy).
pub type DeliverySleeper = Arc<dyn Fn(Duration) -> BoxFuture<'static, ()> + Send + Sync>;

/// Injectable clock seam returning UTC epoch seconds (signature
/// timestamps + ledger `attempted_at`); tests can pin it.
pub type DeliveryClock = Arc<dyn Fn() -> i64 + Send + Sync>;

/// Observed outcome of a single-attempt admin test send (Req 4.4):
/// echoes the HTTP status when a response arrived, or the error class
/// when it did not. Carries no body and no secrets.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct TestResult {
    pub status_code: Option<u16>,
    pub error: Option<String>,
    pub success: bool,
}

/// Fan-out webhook notifier (design "Component 2"). Clones share the
/// once-built HTTP client, the ledger handle and the config Arc, so
/// `AppState` can hold it next to the other shared managers.
#[derive(Clone)]
pub struct WebhookNotifier {
    config: Arc<RwLock<Config>>,
    ledger: Arc<WebhookLedgerStore>,
    /// Built once with redirects disabled: webhook receivers must
    /// answer directly with 200-299 — a 3xx fails the attempt (no
    /// redirect hops to re-validate, so the per-attempt SSRF check
    /// covers the whole exchange). The per-attempt timeout and
    /// redirect policy cannot silently regress to a default client:
    /// construction failure propagates (see [`Self::new`]).
    client: reqwest::Client,
    sleeper: DeliverySleeper,
    clock: DeliveryClock,
    /// Bounds concurrently running delivery chains (see
    /// [`MAX_CONCURRENT_DELIVERY_CHAINS`]). Shared by clones so the
    /// cap is process-wide, not per-emit.
    chain_permits: Arc<Semaphore>,
    /// Degraded-mode flag: `true` when the guarded client could not
    /// be constructed at startup — every attempt then fails fast
    /// locally (never falls back to an unguarded default client).
    delivery_disabled: bool,
}

fn default_sleeper() -> DeliverySleeper {
    Arc::new(|delay: Duration| Box::pin(tokio::time::sleep(delay)))
}

fn default_clock() -> DeliveryClock {
    Arc::new(|| chrono::Utc::now().timestamp())
}

impl WebhookNotifier {
    /// Build the notifier. Client construction failure propagates
    /// (no silent fallback to a default client with no timeout and
    /// default redirect handling — the caller decides how to degrade,
    /// mirroring the ledger-open failure handling in
    /// `GatewayServer::new`).
    pub async fn new(
        config: Arc<RwLock<Config>>,
        ledger: Arc<WebhookLedgerStore>,
    ) -> Result<Self, String> {
        let client = reqwest::Client::builder()
            .timeout(ATTEMPT_TIMEOUT)
            // Webhook deliveries never follow redirects: the receiver
            // must answer directly. A 3xx surfaces as a failed
            // attempt (non-2xx status), which the retry schedule
            // handles like any other failure.
            .redirect(reqwest::redirect::Policy::none())
            .build()
            .map_err(|error| format!("webhook delivery client construction failed: {error}"))?;
        Ok(Self {
            config,
            ledger,
            client,
            sleeper: default_sleeper(),
            clock: default_clock(),
            chain_permits: Arc::new(Semaphore::new(MAX_CONCURRENT_DELIVERY_CHAINS)),
            delivery_disabled: false,
        })
    }

    /// Test seam: replace the retry-delay sleeper (compresses the
    /// 30 s/2 m/10 m schedule in tests).
    #[cfg(test)]
    pub fn with_sleeper(mut self, sleeper: DeliverySleeper) -> Self {
        self.sleeper = sleeper;
        self
    }

    /// Test seam: replace the unix-seconds clock.
    #[cfg(test)]
    pub fn with_clock(mut self, clock: DeliveryClock) -> Self {
        self.clock = clock;
        self
    }

    /// Degraded-mode sentinel (startup fallback when client
    /// construction failed): holds the live config and an in-memory
    /// ledger, and every send fails fast locally — no network, no
    /// default-client escape hatch. Ledger rows still record the
    /// failed attempts so delivery state stays observable.
    pub fn degraded(config: Arc<RwLock<Config>>) -> Self {
        Self {
            config,
            ledger: Arc::new(WebhookLedgerStore::new_in_memory()),
            client: reqwest::Client::new(),
            sleeper: default_sleeper(),
            clock: default_clock(),
            chain_permits: Arc::new(Semaphore::new(MAX_CONCURRENT_DELIVERY_CHAINS)),
            delivery_disabled: true,
        }
    }

    /// Emit an event: returns immediately — the fan-out (ledger row,
    /// endpoint fan-out, deliveries) runs inside a spawned task
    /// (Req 4.1: asynchronous, never blocking request handling).
    /// Callers only invoke this while the subsystem is active (at
    /// least one endpoint configured; dormancy contract).
    pub async fn emit(&self, event: BudgetAlertEvent) {
        let notifier = self.clone();
        tokio::spawn(async move {
            notifier.fan_out(event).await;
        });
    }

    /// Spawned fan-out body: snapshot endpoints under the read lock
    /// (lock dropped before any I/O), record the event row, then either
    /// record `skipped_no_endpoints` (Req 4.7) or spawn one
    /// delivery chain per subscribed endpoint.
    async fn fan_out(&self, event: BudgetAlertEvent) {
        // Endpoint snapshot under the read lock; the guard drops at
        // the end of this block, before any signing or network I/O
        // (design "Config-read discipline").
        let (endpoints, allowed_private_hosts) = {
            let config = self.config.read().await;
            (
                config.webhooks.endpoints.clone(),
                config.webhooks.allowed_private_hosts.clone(),
            )
        };

        self.insert_alert_event_row(&event).await;

        let subscribed: Vec<WebhookEndpoint> = endpoints
            .into_iter()
            .filter(|endpoint| endpoint_subscribes(endpoint, &event.event_type))
            .collect();

        if subscribed.is_empty() {
            // Active subsystem, no subscriber for this type (Req 4.7):
            // one `skipped_no_endpoints` row, no delivery attempts.
            let ledger = Arc::clone(&self.ledger);
            let event_id = event.id.to_string();
            let _ = tokio::task::spawn_blocking(move || {
                ledger.record_undelivered(&event_id)
            })
            .await;
            tracing::info!(
                event_id = %event.id,
                "webhook event recorded with no subscribing endpoint"
            );
            return;
        }

        for endpoint in subscribed {
            // Serialize-once note: the design's "serialize once"
            // refers to the payload template. `signed` is a field of
            // the body itself (Req 3.5), so each endpoint gets its
            // own serialization with the flag set from its secret.
            let mut variant = event.clone();
            variant.signed = effective_secret(&endpoint).is_some();
            let body = match serde_json::to_vec(&variant) {
                Ok(body) => body,
                Err(error) => {
                    tracing::warn!(
                        event_id = %event.id,
                        error = %error,
                        "webhook envelope serialization failed; endpoint skipped"
                    );
                    continue;
                }
            };
            // Bounded concurrency (NFR Performance): acquire a chain
            // permit WITHOUT blocking — a full semaphore records a
            // failed ledger row instead of queueing work emit can
            // never await (Req 4.1) and never silently dropping.
            let event_id = event.id.to_string();
            match Arc::clone(&self.chain_permits).try_acquire_owned() {
                Ok(permit) => {
                    let notifier = self.clone();
                    let allowed = allowed_private_hosts.clone();
                    tokio::spawn(async move {
                        let _permit = permit;
                        notifier
                            .deliver_with_retries(endpoint, body, event_id, allowed)
                            .await;
                    });
                }
                Err(_) => {
                    tracing::warn!(
                        event_id = %event_id,
                        endpoint_url = %endpoint.url,
                        "webhook delivery-chain concurrency cap reached; attempt recorded as failed"
                    );
                    self.record_attempt_row(
                        &event_id,
                        &endpoint.url,
                        1,
                        "failed",
                        None,
                        Some("delivery_concurrency_cap".to_string()),
                    )
                    .await;
                }
            }
        }
    }

/// One endpoint's delivery chain: at most [`MAX_ATTEMPTS`]
/// attempts at the [`RETRY_DELAYS`] schedule. SSRF validation
/// re-runs before EVERY attempt with fresh DNS (Req 4.5 NFR
/// Security — closes the rebinding window across the retry span).
/// Every attempt writes its ledger row BEFORE the network I/O as
/// `inflight` and overwrites it with the final outcome when the
/// attempt completes (Req 4.3 + crash-visibility: a restart mid-
/// attempt leaves the `inflight` marker that the startup park pass
/// converts to `parked`). Pre-I/O terminal failures (invalid URL,
/// degraded mode, SSRF rejection) never observe network contact, so
/// they write their terminal row directly. The final failed attempt
/// writes the terminal `parked` row (Req 4.2).
async fn deliver_with_retries(
&self,
endpoint: WebhookEndpoint,
body: Vec<u8>,
event_id: String,
allowed_private_hosts: Vec<String>,
) {
for attempt in 1..=MAX_ATTEMPTS {
if attempt > 1 {
(self.sleeper)(RETRY_DELAYS[(attempt - 2) as usize]).await;
}

let url = match reqwest::Url::parse(&endpoint.url) {
Ok(url) => url,
Err(error) => {
tracing::warn!(
event_id = %event_id,
endpoint_url = %endpoint.url,
error = %error,
"webhook endpoint URL failed to parse; chain failed"
);
self.record_attempt_row(
&event_id,
&endpoint.url,
attempt,
"failed",
None,
Some("invalid_url".to_string()),
)
.await;
return;
}
};

// Degraded mode (startup client-construction failure):
// fail fast locally — never an unguarded default client.
if self.delivery_disabled {
tracing::warn!(
event_id = %event_id,
endpoint_url = %endpoint.url,
attempt = attempt,
"webhook delivery disabled (degraded mode); attempt failed"
);
let outcome = if attempt == MAX_ATTEMPTS {
"parked"
} else {
"failed"
};
self.record_attempt_row(
&event_id,
&endpoint.url,
attempt,
outcome,
None,
Some("delivery_disabled".to_string()),
)
.await;
return;
}

// SSRF guard before every attempt, first and retries
// alike (Req 4.5): first-attempt failure ends the chain
// with a failed row and no network call; a retry-time
// failure parks it (the receiver's DNS turned hostile
// mid-chain — never silently resumed).
if let Err(error) = self.validate_delivery_url(&url, &allowed_private_hosts).await {
let (host, reason) = error.host_and_reason();
tracing::warn!(
event_id = %event_id,
host = %host,
reason = %reason,
attempt = attempt,
"webhook delivery blocked by outbound-fetch guard"
);
let outcome = if attempt == 1 { "failed" } else { "parked" };
self.record_attempt_row(
&event_id,
&endpoint.url,
attempt,
outcome,
None,
Some(format!("ssrf: {reason}")),
)
.await;
return;
}

// Crash-visibility (design Error Handling 5): mark the attempt
// inflight BEFORE any network I/O — a restart between here and
// the outcome write leaves the marker the startup park pass
// converts to `parked`.
self.record_inflight_row(&event_id, &endpoint.url, attempt)
.await;

// Fresh timestamp + signature per attempt (Standard
// Webhooks receivers reject stale timestamps); the HMAC
// covers the exact body bytes sent.
let timestamp = (self.clock)();
let mut request = self
.client
.post(url)
.header(reqwest::header::CONTENT_TYPE, "application/json");
if let Some(secret) = effective_secret(&endpoint) {
request = request
.header("webhook-id", &event_id)
.header("webhook-timestamp", timestamp.to_string())
.header(
"webhook-signature",
webhook_signature(secret, &event_id, timestamp, &body),
);
}

match request.body(body.clone()).send().await {
Ok(response) if response.status().is_success() => {
let status_code = response.status().as_u16();
tracing::info!(
event_id = %event_id,
endpoint_url = %endpoint.url,
attempt = attempt,
status_code = status_code,
"webhook delivered"
);
self.finish_inflight_row(
&event_id,
&endpoint.url,
attempt,
"success",
Some(i64::from(status_code)),
None,
)
.await;
return;
}
Ok(response) => {
let status_code = response.status().as_u16();
let final_attempt = attempt == MAX_ATTEMPTS;
tracing::warn!(
event_id = %event_id,
endpoint_url = %endpoint.url,
attempt = attempt,
status_code = status_code,
parked = final_attempt,
"webhook delivery attempt failed"
);
if final_attempt {
// Terminal parked row after the final attempt
// (Req 4.2): visible in the panel, never
// silently resumed or dropped.
self.finish_inflight_row(
&event_id,
&endpoint.url,
attempt,
"parked",
Some(i64::from(status_code)),
None,
)
.await;
return;
}
self.finish_inflight_row(
&event_id,
&endpoint.url,
attempt,
"failed",
Some(i64::from(status_code)),
None,
)
.await;
}
Err(error) => {
let class = transport_error_class(&error);
let final_attempt = attempt == MAX_ATTEMPTS;
tracing::warn!(
event_id = %event_id,
endpoint_url = %endpoint.url,
attempt = attempt,
error_class = class,
parked = final_attempt,
"webhook delivery attempt failed"
);
if final_attempt {
self.finish_inflight_row(
&event_id,
&endpoint.url,
attempt,
"parked",
None,
Some(class.to_string()),
)
.await;
return;
}
self.finish_inflight_row(
&event_id,
&endpoint.url,
attempt,
"failed",
None,
Some(class.to_string()),
)
.await;
}
}
}
}

    /// Admin test send (Req 4.4): single attempt — no retries — to the
    /// endpoint in `webhooks.endpoints` whose URL matches
    /// `endpoint_url` (endpoint identity is the URL, design Component
    /// 2). Same envelope shape with `type: "webhook.test"` and
    /// synthetic data, same signing pipeline, same SSRF validation.
    /// Writes a ledger event row + attempt row like a real send unless
    /// validation fails before any network call (then just the error).
    pub async fn send_test(&self, endpoint_url: &str) -> TestResult {
        // Endpoint + allowlist snapshot under the read lock, dropped
        // before any I/O.
        let (endpoint, allowed_private_hosts) = {
            let config = self.config.read().await;
            let endpoint = config
                .webhooks
                .endpoints
                .iter()
                .find(|endpoint| endpoint.url == endpoint_url)
                .cloned();
            (endpoint, config.webhooks.allowed_private_hosts.clone())
        };
        let Some(endpoint) = endpoint else {
            return TestResult {
                status_code: None,
                error: Some("endpoint not found in configuration".to_string()),
                success: false,
            };
        };

        let mut event = WebhookTestEvent::new(endpoint_url);
        event.signed = effective_secret(&endpoint).is_some();
        let body = match serde_json::to_vec(&event) {
            Ok(body) => body,
            Err(error) => {
                return TestResult {
                    status_code: None,
                    error: Some(error.to_string()),
                    success: false,
                };
            }
        };

        let url = match reqwest::Url::parse(endpoint_url) {
            Ok(url) => url,
            Err(error) => {
                return TestResult {
                    status_code: None,
                    error: Some(error.to_string()),
                    success: false,
                };
            }
        };
        // Degraded mode: fail fast locally, mirror the chain behavior.
        if self.delivery_disabled {
            return TestResult {
                status_code: None,
                error: Some("delivery_disabled".to_string()),
                success: false,
            };
        }
        if let Err(error) = self.validate_delivery_url(&url, &allowed_private_hosts).await {
            let (host, reason) = error.host_and_reason();
            tracing::warn!(
                host = %host,
                reason = %reason,
                "webhook test send blocked by outbound-fetch guard"
            );
            // Validation failed before any send: no ledger rows.
            return TestResult {
                status_code: None,
                error: Some(format!("ssrf: {reason}")),
                success: false,
            };
        }

        // Ledger event row first (deliveries FK-reference it).
        let event_id = event.id.to_string();
        let payload_json = String::from_utf8_lossy(&body).into_owned();
        let created_at = (self.clock)();
        {
            let ledger = Arc::clone(&self.ledger);
            let scope_label = endpoint_url.to_string();
            let _ = tokio::task::spawn_blocking(move || {
                ledger.insert_event(
                    &event_id,
                    crate::webhooks::events::WEBHOOK_TEST_EVENT_TYPE,
                    "test",
                    &scope_label,
                    &payload_json,
                    created_at,
                )
            })
            .await;
        }

        let timestamp = (self.clock)();
        let mut request = self
            .client
            .post(url)
            .header(reqwest::header::CONTENT_TYPE, "application/json");
        if let Some(secret) = effective_secret(&endpoint) {
            request = request
                .header("webhook-id", event.id.to_string())
                .header("webhook-timestamp", timestamp.to_string())
                .header(
                    "webhook-signature",
                    webhook_signature(secret, &event.id.to_string(), timestamp, &body),
                );
        }

        let outcome: (&'static str, Option<u16>, Option<String>) = match request
            .body(body)
            .send()
            .await
        {
            Ok(response) => {
                let status_code = response.status().as_u16();
                if response.status().is_success() {
                    ("success", Some(status_code), None)
                } else {
                    ("failed", Some(status_code), None)
                }
            }
            Err(error) => {
                let class = transport_error_class(&error).to_string();
                ("failed", None, Some(class))
            }
        };
        let (outcome_name, status_code, error) = outcome;
        self.record_attempt_row(
            &event.id.to_string(),
            endpoint_url,
            1,
            outcome_name,
            status_code.map(i64::from),
            error.clone(),
        )
        .await;
        if outcome_name == "success" {
            tracing::info!(
                endpoint_url = %endpoint_url,
                status_code = status_code.unwrap_or(0),
                "webhook test send delivered"
            );
        } else {
            tracing::warn!(
                endpoint_url = %endpoint_url,
                status_code = status_code.unwrap_or(0),
                "webhook test send failed"
            );
        }
        TestResult {
            status_code,
            error,
            success: outcome_name == "success",
        }
    }

    /// Record the alert-event row (unsigned template payload — the
    /// per-endpoint `signed` variants are delivery-time only) through
    /// the blocking pool.
    async fn insert_alert_event_row(&self, event: &BudgetAlertEvent) {
        let id = event.id.to_string();
        let event_type = budget_event_type_name(&event.event_type).to_string();
        let scope = match event.data.scope {
            Scope::Provider => "provider".to_string(),
            Scope::VirtualKey => "virtual_key".to_string(),
        };
        let scope_label = alert_scope_label(event);
        let payload_json = serde_json::to_string(event).unwrap_or_else(|_| "{}".to_string());
        let created_at = event.created_at.timestamp();
        let ledger = Arc::clone(&self.ledger);
        let _ = tokio::task::spawn_blocking(move || {
            ledger.insert_event(&id, &event_type, &scope, &scope_label, &payload_json, created_at)
        })
        .await;
    }

/// One ledger attempt row via spawn_blocking (NFR Performance:
/// ledger writes never block the async runtime; the store logs its
/// own warn on failure — observability, not control).
async fn record_attempt_row(
&self,
event_id: &str,
endpoint_url: &str,
attempt: i64,
outcome: &str,
status_code: Option<i64>,
error: Option<String>,
) {
let ledger = Arc::clone(&self.ledger);
let event_id = event_id.to_string();
let endpoint_url = endpoint_url.to_string();
let outcome = outcome.to_string();
let attempted_at = (self.clock)();
let _ = tokio::task::spawn_blocking(move || {
ledger.record_attempt(
&event_id,
&endpoint_url,
attempt,
&outcome,
status_code,
error.as_deref(),
attempted_at,
)
})
.await;
}

/// Inflight marker row via spawn_blocking: written before network
/// I/O, finished by [`Self::finish_inflight_row`].
async fn record_inflight_row(&self, event_id: &str, endpoint_url: &str, attempt: i64) {
let ledger = Arc::clone(&self.ledger);
let event_id = event_id.to_string();
let endpoint_url = endpoint_url.to_string();
let attempted_at = (self.clock)();
let _ = tokio::task::spawn_blocking(move || {
ledger.record_inflight_attempt(&event_id, &endpoint_url, attempt, attempted_at)
})
.await;
}

/// Terminal overwrite of the inflight marker row via spawn_blocking.
async fn finish_inflight_row(
&self,
event_id: &str,
endpoint_url: &str,
attempt: i64,
outcome: &str,
status_code: Option<i64>,
error: Option<String>,
) {
let ledger = Arc::clone(&self.ledger);
let event_id = event_id.to_string();
let endpoint_url = endpoint_url.to_string();
let outcome = outcome.to_string();
let attempted_at = (self.clock)();
let _ = tokio::task::spawn_blocking(move || {
ledger.finish_inflight_attempt(
&event_id,
&endpoint_url,
attempt,
&outcome,
status_code,
error.as_deref(),
attempted_at,
)
})
.await;
}

/// Delivery-time SSRF validation (Req 4.5, fresh DNS per attempt).
/// See the free [`validate_delivery_url`] for the tier rules and
/// the blocking-pool DNS discipline.
async fn validate_delivery_url(
&self,
url: &reqwest::Url,
allowed_private_hosts: &[String],
) -> Result<(), outbound_fetch::OutboundFetchError> {
validate_delivery_url(url, allowed_private_hosts).await
}
}

/// Whether `endpoint` subscribes to `event_type` (Req 3.2): `events`
/// absent = all types; otherwise the type must be listed.
fn endpoint_subscribes(endpoint: &WebhookEndpoint, event_type: &BudgetEventType) -> bool {
match &endpoint.events {
None => true,
Some(events) => events.contains(event_type),
}
}

/// Wire name of a budget event type (ledger rows + tracing).
fn budget_event_type_name(event_type: &BudgetEventType) -> &'static str {
    match event_type {
        BudgetEventType::BudgetWarning => "budget.warning",
        BudgetEventType::BudgetExhausted => "budget.exhausted",
        BudgetEventType::BudgetReset => "budget.reset",
    }
}

/// Ledger `scope_label` (design ledger schema): provider name, or
/// `key:<id>:<name>` for virtual keys.
fn alert_scope_label(event: &BudgetAlertEvent) -> String {
    match event.data.scope {
        Scope::Provider => event
            .data
            .name
            .clone()
            .unwrap_or_else(|| "provider".to_string()),
        Scope::VirtualKey => match (&event.data.key_id, &event.data.name) {
            (Some(key_id), Some(name)) => format!("key:{key_id}:{name}"),
            (Some(key_id), None) => format!("key:{key_id}"),
            (None, Some(name)) => name.clone(),
            (None, None) => "virtual_key".to_string(),
        },
    }
}

/// Effective signing secret: `Some` only when configured and
/// non-empty (an empty string cannot produce a meaningful HMAC).
fn effective_secret(endpoint: &WebhookEndpoint) -> Option<&str> {
    endpoint.secret.as_deref().filter(|secret| !secret.is_empty())
}

/// Delivery-time SSRF validation (Req 4.5): the same
/// [`validate_outbound_url`] rules with fresh DNS, plus the webhooks
/// escape hatch — plain `http` permitted only for hosts that
/// exactly match `webhooks.allowed_private_hosts` (mirrors the
/// config-save validation rule; `validate_outbound_url` itself has no
/// per-host http tier). Private-address permission for listed hosts
/// comes from passing them as the outbound-fetch allowlist.
///
/// DNS resolution runs on the blocking pool (`spawn_blocking`) —
/// `std_resolve_host` is a blocking `ToSocketAddrs` call and must
/// never run on a tokio worker (NFR Performance). The whole check is
/// bounded by the per-attempt timeout applied around the send.
async fn validate_delivery_url(
    url: &reqwest::Url,
    allowed_private_hosts: &[String],
) -> Result<(), outbound_fetch::OutboundFetchError> {
    let host = url.host_str().unwrap_or_default();
    let allowlisted = allowed_private_hosts
        .iter()
        .any(|allowed| allowed.eq_ignore_ascii_case(host));
    if url.scheme() == "http" && !allowlisted {
        return Err(outbound_fetch::OutboundFetchError::SchemeNotAllowed {
            host: host.to_string(),
            scheme: "http".to_string(),
        });
    }
    // The scheme gate above already enforced per-host http; the inner
    // config keeps allow_http on so listed http hosts reach the
    // address-class checks, which honor the same allowlist.
    let config = OutboundFetchConfig {
        caller: CallerFetchConfig {
            allow_private_hosts: false,
            allow_http: true,
            allowed_private_hosts: allowed_private_hosts.to_vec(),
        },
    };
    // Non-IP hosts need blocking DNS: run the resolver off the async
    // runtime. Literal-IP hosts skip DNS entirely, so the sync path
    // stays for them (cheap, no I/O).
    let host = host.to_string();
    let is_literal_ip = host.parse::<std::net::IpAddr>().is_ok();
    if is_literal_ip {
        return validate_outbound_url(url, &config, &std_resolve_host);
    }
    let url = url.clone();
    tokio::task::spawn_blocking(move || {
        validate_outbound_url(&url, &config, &std_resolve_host)
    })
    .await
    .map_err(|error| {
        tracing::warn!(error = %error, "webhook SSRF validation task failed");
        outbound_fetch::OutboundFetchError::UnresolvableHost { host }
    })?
}

/// Standard Webhooks signature (Req 3.4), per the official spec
/// (github.com/standard-webhooks/standard-webhooks):
/// `v1,base64(HMAC-SHA256(key, "{id}.{timestamp}.{body}"))` over the
/// exact body bytes sent. The header value uses `v1,` (comma), and the
/// HMAC key is the base64-DECODED configured secret (after stripping
/// an optional `whsec_` prefix). If the secret is not valid base64,
/// the raw bytes are used as the key (logged once at delivery) so
/// operators who paste a non-base64 secret still get signed
/// deliveries instead of unsigned ones.
fn webhook_signature(secret: &str, event_id: &str, timestamp: i64, body: &[u8]) -> String {
    let key = ring::hmac::Key::new(ring::hmac::HMAC_SHA256, &signing_key(secret));
    let timestamp = timestamp.to_string();
    let mut payload = Vec::with_capacity(
        event_id.len() + timestamp.len() + body.len() + 2,
    );
    payload.extend_from_slice(event_id.as_bytes());
    payload.push(b'.');
    payload.extend_from_slice(timestamp.as_bytes());
    payload.push(b'.');
    payload.extend_from_slice(body);
    let tag = ring::hmac::sign(&key, &payload);
    format!(
        "v1,{}",
        base64::engine::general_purpose::STANDARD.encode(tag.as_ref())
    )
}

/// Decode a configured Standard Webhooks secret into HMAC key bytes:
/// strip an optional `whsec_` prefix, then base64-decode. A secret
/// that fails base64 decoding falls back to its raw bytes (warned
/// once) so misconfigured secrets still sign rather than silently
/// going unsigned.
pub fn signing_key(secret: &str) -> Vec<u8> {
    let trimmed = secret.strip_prefix("whsec_").unwrap_or(secret);
    match base64::engine::general_purpose::STANDARD.decode(trimmed) {
        Ok(key) => key,
        Err(_) => {
            tracing::warn!(
                "webhook secret is not valid base64 (after 'whsec_' strip); using raw bytes as HMAC key"
            );
            trimmed.as_bytes().to_vec()
        }
    }
}

/// Short transport error class for ledger rows — never the full
/// reqwest display (which embeds the URL) and never a body.
fn transport_error_class(error: &reqwest::Error) -> &'static str {
    if error.is_timeout() {
        "timeout"
    } else if error.is_connect() {
        "connect_error"
    } else {
        "transport_error"
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::config::WebhooksConfig;
    use crate::webhooks::events::BudgetEventData;
    use std::time::Instant;
    use tempfile::NamedTempFile;
    use wiremock::matchers::{method, path};
    use wiremock::{Mock, MockServer, ResponseTemplate};

    const SECRET: &str = "whsec_test_secret";

    fn instant_sleeper() -> DeliverySleeper {
        Arc::new(|_delay: Duration| Box::pin(futures::future::ready(())))
    }

    fn temp_ledger() -> (Arc<WebhookLedgerStore>, NamedTempFile) {
        let temp = NamedTempFile::new().unwrap();
        let store = Arc::new(WebhookLedgerStore::new(temp.path()).unwrap());
        (store, temp)
    }

    /// Minimal config carrying only the webhooks section under test
    /// (same serde_yaml construction style as the summarizer tests).
    /// The allowlist opt-in mirrors the operator LAN escape hatch —
    /// wiremock binds 127.0.0.1, which the strict tier would reject.
    fn test_config(endpoints: Vec<WebhookEndpoint>, allowed: Vec<String>) -> Arc<RwLock<Config>> {
        let mut config: Config =
            serde_yaml::from_str("server:\n  host: 127.0.0.1\n  port: 0\nproviders: []\nmodel_groups: []\n")
                .expect("test config must parse");
        config.webhooks = WebhooksConfig {
            endpoints,
            allowed_private_hosts: allowed,
        };
        Arc::new(RwLock::new(config))
    }

    fn signed_endpoint(url: String) -> WebhookEndpoint {
        WebhookEndpoint {
            url,
            secret: Some(SECRET.to_string()),
            events: None,
        }
    }

    fn warning_event() -> BudgetAlertEvent {
        BudgetAlertEvent::warning(BudgetEventData {
            scope: Scope::Provider,
            name: Some("openai".to_string()),
            key_id: None,
            limit_usd: Some(10.0),
            spent_usd: Some(8.0),
            percent: Some(80),
            window: Some("daily".to_string()),
            window_start: Some(1_735_689_600),
            reset_source: None,
        })
    }

async fn notifier(
    endpoints: Vec<WebhookEndpoint>,
    allowed: Vec<String>,
    ledger: Arc<WebhookLedgerStore>,
) -> WebhookNotifier {
    WebhookNotifier::new(test_config(endpoints, allowed), ledger)
        .await
        .expect("notifier construction must succeed in tests")
        .with_sleeper(instant_sleeper())
}

    /// Poll until `predicate` holds (200 x 25 ms = 5 s budget) —
    /// spawned delivery chains finish asynchronously.
    async fn wait_until<F, Fut>(predicate: F)
    where
        F: Fn() -> Fut,
        Fut: std::future::Future<Output = bool>,
    {
        for _ in 0..200 {
            if predicate().await {
                return;
            }
            tokio::time::sleep(Duration::from_millis(25)).await;
        }
        panic!("condition not met within 5 s");
    }

    /// Wait until the mock server has received at least one request.
    async fn wait_for_first_request(server: &MockServer) {
        wait_until(|| async {
            server
                .received_requests()
                .await
                .map(|requests| !requests.is_empty())
                .unwrap_or(false)
        })
        .await
    }

    /// 1. Signed deliveries carry the Standard Webhooks headers over
    /// the exact body bytes (Req 3.4).
    // Validates: Requirement 3.4
    #[tokio::test]
    async fn signature_headers_match_recomputed_hmac() {
        let (ledger, _temp) = temp_ledger();
        let server = MockServer::start().await;
        Mock::given(method("POST"))
            .and(path("/hook"))
            .respond_with(ResponseTemplate::new(200))
            .expect(1)
            .mount(&server)
            .await;

        let url = format!("{}/hook", server.uri());
        let notifier = notifier(vec![signed_endpoint(url.clone())], vec!["127.0.0.1".to_string()], ledger.clone()).await;
        let event = warning_event();
        let event_id = event.id.to_string();
        notifier.emit(event).await;

        wait_for_first_request(&server).await;
        let request = &server.received_requests().await.unwrap()[0];

    // Receiver-side recompute over {id}.{timestamp}.{body} with the
    // base64-decoded secret (Standard Webhooks: key = decoded bytes
    // after the optional whsec_ prefix, header prefixed "v1,").
    let header_id = request
        .headers
        .get("webhook-id")
        .and_then(|value| value.to_str().ok())
        .unwrap()
        .to_string();
    let header_timestamp = request
        .headers
        .get("webhook-timestamp")
        .and_then(|value| value.to_str().ok())
        .unwrap()
        .to_string();
    let header_signature = request
        .headers
        .get("webhook-signature")
        .and_then(|value| value.to_str().ok())
        .unwrap()
        .to_string();
    assert_eq!(header_id, event_id);

        let key = ring::hmac::Key::new(ring::hmac::HMAC_SHA256, &signing_key(SECRET));
    let mut base = format!("{header_id}.{header_timestamp}.").into_bytes();
    base.extend_from_slice(&request.body);
    let expected = format!(
        "v1,{}",
        base64::engine::general_purpose::STANDARD
            .encode(ring::hmac::sign(&key, &base).as_ref())
    );
    assert_eq!(header_signature, expected);

        // Signed endpoint: envelope marks the delivery signed.
        let body: serde_json::Value = serde_json::from_slice(&request.body).unwrap();
        assert_eq!(body["signed"], serde_json::json!(true));

        // Success row landed.
        wait_until(|| {
            let rows = ledger.recent_deliveries(Some(&url), 10);
            async move { !rows.is_empty() }
        })
        .await;
        let rows = ledger.recent_deliveries(Some(&url), 10);
        assert_eq!(rows[0].outcome, "success");
    }

    /// 2. One 500 then 200: the retry succeeds and the ledger carries
    /// both attempt rows (Req 4.2, 4.3).
    // Validates: Requirements 4.2, 4.3
    #[tokio::test]
    async fn retry_after_500_succeeds_on_second_attempt() {
        struct FailOnceThen(std::sync::atomic::AtomicUsize);
        impl wiremock::Respond for FailOnceThen {
            fn respond(&self, _request: &wiremock::Request) -> wiremock::ResponseTemplate {
                let call = self.0.fetch_add(1, std::sync::atomic::Ordering::SeqCst);
                if call == 0 {
                    ResponseTemplate::new(500)
                } else {
                    ResponseTemplate::new(200)
                }
            }
        }

        let (ledger, _temp) = temp_ledger();
        let server = MockServer::start().await;
        Mock::given(method("POST"))
            .and(path("/hook"))
            .respond_with(FailOnceThen(std::sync::atomic::AtomicUsize::new(0)))
            .expect(2)
            .mount(&server)
            .await;

        let url = format!("{}/hook", server.uri());
        let notifier = notifier(vec![signed_endpoint(url.clone())], vec!["127.0.0.1".to_string()], ledger.clone()).await;
        let event = warning_event();
        let event_id = event.id.to_string();
        notifier.emit(event).await;

    wait_until(|| {
        let rows = ledger.recent_deliveries(Some(&url), 10);
        async move { rows.iter().any(|row| row.outcome == "success") }
    })
    .await;
    assert_eq!(server.received_requests().await.unwrap().len(), 2);

        let rows = ledger.recent_deliveries(Some(&url), 10);
        assert_eq!(rows.len(), 2, "exactly two attempt rows");
        // Newest first: success (attempt 2), failed 500 (attempt 1).
        assert_eq!(rows[0].outcome, "success");
        assert_eq!(rows[0].attempt, 2);
        assert_eq!(rows[1].outcome, "failed");
        assert_eq!(rows[1].status_code, Some(500));
        assert_eq!(rows[1].attempt, 1);
        assert!(rows.iter().all(|row| row.event_id == event_id));

        // Event row recorded before any delivery (FK target).
        let events = ledger.recent_events(5);
        assert_eq!(events[0].id, event_id);
        assert_eq!(events[0].event_type, "budget.warning");
    }

    /// 3. Always-500 parks after exactly 4 attempts (Req 4.2),
    ///    compressed to zero delay via the injectable sleeper.
    // Validates: Requirement 4.2
    #[tokio::test]
    async fn always_500_parks_after_four_attempts() {
        let (ledger, _temp) = temp_ledger();
        let server = MockServer::start().await;
        Mock::given(method("POST"))
            .and(path("/hook"))
            .respond_with(ResponseTemplate::new(500))
            .expect(4)
            .mount(&server)
            .await;

        let url = format!("{}/hook", server.uri());
        let notifier = notifier(vec![signed_endpoint(url.clone())], vec!["127.0.0.1".to_string()], ledger.clone()).await;
        notifier.emit(warning_event()).await;

        wait_until(|| {
            let rows = ledger.recent_deliveries(Some(&url), 10);
            async move { rows.iter().any(|row| row.outcome == "parked") }
        })
        .await;
        assert_eq!(server.received_requests().await.unwrap().len(), 4);

        let rows = ledger.recent_deliveries(Some(&url), 10);
        assert_eq!(rows.len(), 4, "exactly four attempt rows");
        // Newest first: parked terminal row (attempt 4) over failed 1-3.
        assert_eq!(rows[0].outcome, "parked");
        assert_eq!(rows[0].attempt, 4);
        assert_eq!(rows[0].status_code, Some(500));
        assert_eq!(rows[1].outcome, "failed");
        assert_eq!(rows[2].outcome, "failed");
        assert_eq!(rows[3].outcome, "failed");
    }

    /// 4. Unsigned endpoints: `"signed": false` in the body and no
    ///    webhook-* headers at all (Req 3.5).
    // Validates: Requirement 3.5
    #[tokio::test]
    async fn unsigned_endpoint_gets_signed_false_and_no_webhook_headers() {
        let (ledger, _temp) = temp_ledger();
        let server = MockServer::start().await;
        Mock::given(method("POST"))
            .and(path("/hook"))
            .respond_with(ResponseTemplate::new(200))
            .expect(1)
            .mount(&server)
            .await;

        let url = format!("{}/hook", server.uri());
        let endpoint = WebhookEndpoint {
            url: url.clone(),
            secret: None,
            events: None,
        };
        let notifier = notifier(vec![endpoint], vec!["127.0.0.1".to_string()], ledger).await;
        notifier.emit(warning_event()).await;

        wait_for_first_request(&server).await;
        let request = &server.received_requests().await.unwrap()[0];

        let body: serde_json::Value = serde_json::from_slice(&request.body).unwrap();
        assert_eq!(body["signed"], serde_json::json!(false));
        for header in ["webhook-id", "webhook-timestamp", "webhook-signature"] {
            assert!(
                request.headers.get(header).is_none(),
                "unsigned delivery must not carry {header}"
            );
        }
    }

    /// 5. send_test echoes the observed status on success and the
    ///    error class on a dead URL, with ledger rows like a real
    ///    send (Req 4.4).
    // Validates: Requirement 4.4
    #[tokio::test]
    async fn send_test_echoes_status_and_dead_url_error() {
        let (ledger, _temp) = temp_ledger();
        let server = MockServer::start().await;
        Mock::given(method("POST"))
            .and(path("/hook"))
            .respond_with(ResponseTemplate::new(200))
            .expect(1)
            .mount(&server)
            .await;

        let live = format!("{}/hook", server.uri());
        // Port 1 on loopback: nothing listens — connection refused.
        let dead = "http://127.0.0.1:1/hook".to_string();
        let notifier = notifier(
            vec![signed_endpoint(live.clone()), signed_endpoint(dead.clone())],
            vec!["127.0.0.1".to_string()],
            ledger.clone(),
        )
        .await;

        let result = notifier.send_test(&live).await;
        assert!(result.success);
        assert_eq!(result.status_code, Some(200));
        assert!(result.error.is_none());
        assert_eq!(server.received_requests().await.unwrap().len(), 1);
        let request = &server.received_requests().await.unwrap()[0];
        let body: serde_json::Value = serde_json::from_slice(&request.body).unwrap();
        assert_eq!(body["type"], serde_json::json!("webhook.test"));
        assert_eq!(body["data"]["scope"], serde_json::json!("test"));
        assert_eq!(body["data"]["url"], serde_json::json!(live));
        assert_eq!(
            request.headers.get("webhook-id").is_some(),
            true,
            "test send uses the signing pipeline"
        );
        // Ledger rows like a real send: event + attempt.
        wait_until(|| {
            let rows = ledger.recent_deliveries(Some(&live), 10);
            async move { !rows.is_empty() }
        })
        .await;
        let rows = ledger.recent_deliveries(Some(&live), 10);
        assert_eq!(rows[0].outcome, "success");
        assert_eq!(ledger.recent_events(5)[0].event_type, "webhook.test");

        let result = notifier.send_test(&dead).await;
        assert!(!result.success);
        assert_eq!(result.status_code, None);
        let error = result.error.expect("dead URL must echo an error");
        assert!(error.contains("connect_error"), "echoed error class: {error}");
    }

    /// 6. Event filtering: an endpoint subscribed to
    ///    `budget.exhausted` only receives no `budget.warning`
    ///    delivery, and the skipped path records
    ///    `skipped_no_endpoints` (Reqs 3.2, 4.7).
    // Validates: Requirements 3.2, 4.7
    #[tokio::test]
    async fn event_filter_skips_unsubscribed_endpoint_and_records_skipped() {
        let (ledger, _temp) = temp_ledger();
        let server = MockServer::start().await;
        Mock::given(method("POST"))
            .and(path("/hook"))
            .respond_with(ResponseTemplate::new(200))
            .expect(0)
            .mount(&server)
            .await;

        let url = format!("{}/hook", server.uri());
        let endpoint = WebhookEndpoint {
            url,
            secret: None,
            events: Some(vec![BudgetEventType::BudgetExhausted]),
        };
        let notifier = notifier(vec![endpoint], vec!["127.0.0.1".to_string()], ledger.clone()).await;
        let event = warning_event();
        let event_id = event.id.to_string();
        notifier.emit(event).await;

        // The single endpoint does not subscribe to budget.warning:
        // one skipped_no_endpoints row (empty endpoint_url), zero
        // delivery attempts anywhere.
        wait_until(|| {
            let rows = ledger.recent_deliveries(Some(""), 10);
            let event_id = event_id.clone();
            async move {
                rows.iter()
                    .any(|row| row.outcome == "skipped_no_endpoints" && row.event_id == event_id)
            }
        })
        .await;
        assert!(
            server.received_requests().await.unwrap().is_empty(),
            "unsubscribed endpoint must receive no delivery"
        );
    }

    /// 7. SSRF: a loopback URL NOT in the allowlist fails validation
    ///    before the first attempt — no HTTP call, one failed row
    ///    (Req 4.5).
    // Validates: Requirement 4.5
    #[tokio::test]
    async fn ssrf_rejection_makes_no_http_call() {
        let (ledger, _temp) = temp_ledger();
        let server = MockServer::start().await;
        Mock::given(method("POST"))
            .and(path("/hook"))
            .respond_with(ResponseTemplate::new(200))
            .expect(0)
            .mount(&server)
            .await;

        // Loopback http URL with an EMPTY allowlist: strict tier
        // rejects the scheme before any network I/O.
        let url = format!("{}/hook", server.uri());
        let notifier = notifier(vec![signed_endpoint(url.clone())], Vec::new(), ledger.clone()).await;
        let event = warning_event();
        let event_id = event.id.to_string();
        notifier.emit(event).await;

        wait_until(|| {
            let rows = ledger.recent_deliveries(Some(&url), 10);
            async move { rows.iter().any(|row| row.outcome == "failed") }
        })
        .await;
        let rows = ledger.recent_deliveries(Some(&url), 10);
        assert_eq!(rows.len(), 1, "first-attempt rejection ends the chain");
        assert_eq!(rows[0].outcome, "failed");
        assert_eq!(rows[0].event_id, event_id);
        assert!(rows[0].error.as_deref().unwrap_or("").starts_with("ssrf:"));
        assert!(
            server.received_requests().await.unwrap().is_empty(),
            "rejected delivery must never reach the network"
        );
    }

    /// 7b. SSRF bypass vectors against the delivery URL validator: a
    /// trailing-dot host, userinfo, and an IPv6 zone-id must each fail
    /// closed when the host is NOT allowlisted — and an exact-match
    /// allowlist entry admits ONLY the plain hostname, never these
    /// variants (Req 4.5).
    // Validates: Requirement 4.5
    #[tokio::test]
    async fn ssrf_bypass_variants_fail_closed_and_allowlist_matches_plain_host_only() {
        // NOT allowlisted: every variant fails closed before any
        // network I/O (scheme gate or address-class rejection).
        // Trailing-dot host and userinfo: parseable URLs that must be
        // rejected by the scheme gate / allowlist exact-match rules.
        let vectors = [
            "http://hooks.internal./x",
            "https://hooks.internal./x",
            "http://hooks.internal@evil/x",
            "https://hooks.internal@evil/x",
        ];
        for url in vectors {
            let parsed: reqwest::Url = url.parse().expect("vector URL parses");
            let err = validate_delivery_url(&parsed, &[]).await.expect_err(url);
            assert!(
                matches!(
                    err,
                    outbound_fetch::OutboundFetchError::SchemeNotAllowed { .. }
                        | outbound_fetch::OutboundFetchError::ForbiddenAddress { .. }
                        | outbound_fetch::OutboundFetchError::UnresolvableHost { .. }
                ),
                "{url} must fail closed, got {err:?}"
            );
        }

        // IPv6 zone-id: the URL parser itself rejects scoped literals
        // today (InvalidIpv6Address) — parse failure IS failing closed.
        // Pin both halves: if a future url upgrade ever accepts zone
        // ids, the validator must still reject the loopback literal.
        for url in ["http://[::1%25eth0]/x", "https://[fe80::1%25eth0]:8443/x"] {
            match reqwest::Url::parse(url) {
                Err(parse_error) if parse_error.to_string().contains("IPv6") => {}
                Err(other) => panic!("{url} rejected for the wrong reason: {other}"),
                Ok(parsed) => {
                    let err = validate_delivery_url(&parsed, &[])
                        .await
                        .err()
                        .unwrap_or_else(|| panic!("{url} parsed and passed — zone-id must fail closed"));
                    assert!(matches!(
                        err,
                        outbound_fetch::OutboundFetchError::ForbiddenAddress { .. }
                    ));
                }
            }
        }
        // Exact-match allowlist entry admits the PLAIN hostname only:
        // with "hooks.internal" allowlisted, the plain http host passes
        // the scheme gate, while the trailing-dot, userinfo, and
        // zone-id variants stay rejected — different host strings
        // never inherit the entry.
        let allowed = vec!["hooks.internal".to_string()];
        let plain: reqwest::Url = "http://hooks.internal/x".parse().unwrap();
        let plain_err = validate_delivery_url(&plain, &allowed).await.err();
        assert!(
            !matches!(
                plain_err,
                Some(outbound_fetch::OutboundFetchError::SchemeNotAllowed { .. })
            ),
            "exact-match allowlist must admit the plain http hostname (got {plain_err:?})"
        );
        for url in [
            "http://hooks.internal./x",
            "http://hooks.internal@evil/x",
        ] {
            let parsed: reqwest::Url = url.parse().unwrap();
            let err = validate_delivery_url(&parsed, &allowed)
                .await
                .err()
                .unwrap_or_else(|| {
                    panic!("{url} must stay rejected even with the plain host allowlisted")
                });
            assert!(
                matches!(
                    err,
                    outbound_fetch::OutboundFetchError::SchemeNotAllowed { .. }
                        | outbound_fetch::OutboundFetchError::ForbiddenAddress { .. }
                        | outbound_fetch::OutboundFetchError::UnresolvableHost { .. }
                ),
                "{url} must fail closed against the allowlist, got {err:?}"
            );
        }
    }

    /// 8. Config-lock discipline: emit returns immediately while the
    ///    receiver is still delaying its response — no await on
    ///    network anywhere on the emit path (Req 4.1).
    // Validates: Requirement 4.1
    #[tokio::test]
    async fn emit_returns_before_network_completes() {
        let (ledger, _temp) = temp_ledger();
        let server = MockServer::start().await;
        Mock::given(method("POST"))
            .and(path("/hook"))
            .respond_with(
                ResponseTemplate::new(200).set_delay(Duration::from_millis(1500)),
            )
            .expect(1)
            .mount(&server)
            .await;

        let url = format!("{}/hook", server.uri());
        let notifier = notifier(vec![signed_endpoint(url)], vec!["127.0.0.1".to_string()], ledger).await;

        let start = Instant::now();
        notifier.emit(warning_event()).await;
        let emit_elapsed = start.elapsed();
        assert!(
            emit_elapsed < Duration::from_millis(900),
            "emit must not await network I/O (took {emit_elapsed:?})"
        );

        // The spawned chain still completes in the background.
        wait_for_first_request(&server).await;
    }
}
