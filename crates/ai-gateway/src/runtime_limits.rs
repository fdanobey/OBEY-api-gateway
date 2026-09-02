//! Lock-free snapshot of hot-reloadable per-request limits.
//!
//! Two limits are consulted on the path of *every* request: the maximum body
//! size and the global request deadline. Reading them from
//! `Arc<tokio::sync::RwLock<Config>>` per request is a latent availability bug:
//! that lock is write-preferring, so a single queued writer (config hot-reload,
//! tray update, admin save) blocks every subsequent reader — including cheap
//! liveness endpoints like `/health`. Mirroring both values into atomics keeps
//! the request path lock-free while still honouring hot-reload, which republishes
//! the snapshot through [`RuntimeLimits::apply`].

use std::sync::atomic::{AtomicU64, AtomicUsize, Ordering};
use std::time::Duration;

use axum::extract::{Request, State};
use axum::http::StatusCode;
use axum::middleware::Next;
use axum::response::{IntoResponse, Response};
use axum::Json;
use serde_json::json;

use crate::config::{Config, Provider};
use crate::gateway::AppState;

/// Headroom added on top of the longest configured provider timeout when
/// deriving the global request ceiling.
///
/// The ceiling must never sever a request that provider configuration
/// explicitly permits, otherwise enabling enforcement would truncate working
/// long-running deployments. The grace covers gateway-side overhead outside the
/// upstream call (guardrails, compression, memory retrieval, retry backoff).
const PROVIDER_TIMEOUT_GRACE_SECONDS: u64 = 60;

/// A model name guaranteed to satisfy [`crate::config::is_thinking_model`].
///
/// Used to resolve each provider's *worst-case* total timeout through the
/// existing [`Provider::effective_total_timeout`] logic rather than duplicating
/// its default-resolution rules here.
const THINKING_MODEL_PROBE: &str = "o1";

/// Atomically published view of the limits enforced on every request.
#[derive(Debug)]
pub struct RuntimeLimits {
    max_request_size_bytes: AtomicUsize,
    request_timeout_seconds: AtomicU64,
}

impl RuntimeLimits {
    /// Build a snapshot from `config`.
    pub fn from_config(config: &Config) -> Self {
        let limits = Self {
            max_request_size_bytes: AtomicUsize::new(0),
            request_timeout_seconds: AtomicU64::new(0),
        };
        limits.apply(config);
        limits
    }

    /// Republish the snapshot after a config change (hot-reload).
    pub fn apply(&self, config: &Config) {
        self.apply_parts(
            (config.server.max_request_size_mb as usize).saturating_mul(1024 * 1024),
            effective_request_timeout_seconds(config),
        );
    }

    fn apply_parts(&self, max_request_size_bytes: usize, request_timeout_seconds: u64) {
        self.max_request_size_bytes
            .store(max_request_size_bytes, Ordering::Relaxed);
        self.request_timeout_seconds
            .store(request_timeout_seconds, Ordering::Relaxed);
    }

    /// Current request body ceiling in bytes.
    pub fn max_request_size_bytes(&self) -> usize {
        self.max_request_size_bytes.load(Ordering::Relaxed)
    }

    /// Current global request deadline; `None` disables enforcement.
    pub fn request_timeout(&self) -> Option<Duration> {
        let seconds = self.request_timeout_seconds.load(Ordering::Relaxed);
        (seconds > 0).then(|| Duration::from_secs(seconds))
    }

    /// Current global request deadline in seconds (0 when disabled).
    pub fn request_timeout_seconds(&self) -> u64 {
        self.request_timeout_seconds.load(Ordering::Relaxed)
    }
}

/// Resolve the enforced global request ceiling for `config`.
///
/// Returns the larger of `server.request_timeout_seconds` and the longest
/// worst-case provider total timeout plus [`PROVIDER_TIMEOUT_GRACE_SECONDS`].
/// Taking the maximum is deliberate: `server.request_timeout_seconds` defaults
/// to 30s while provider `total_timeout_seconds` values are routinely in the
/// hundreds, so enforcing the server value literally would abort requests the
/// provider configuration allows. The derived value still bounds work that no
/// provider timeout covers (slot queue waits, guardrail and OAuth HTTP calls
/// built without their own timeout), which is what turns a stall into a
/// permanent hang.
pub fn effective_request_timeout_seconds(config: &Config) -> u64 {
    let longest_provider = config
        .providers
        .iter()
        .map(worst_case_total_timeout)
        .max()
        .unwrap_or(0);

    ceiling_seconds(config.server.request_timeout_seconds, longest_provider)
}

/// Pure ceiling arithmetic, split out so it is testable without a full [`Config`].
fn ceiling_seconds(server_timeout_seconds: u64, longest_provider_timeout_seconds: u64) -> u64 {
    server_timeout_seconds
        .max(longest_provider_timeout_seconds.saturating_add(PROVIDER_TIMEOUT_GRACE_SECONDS))
}

/// The largest total timeout `provider` can resolve to across any model.
fn worst_case_total_timeout(provider: &Provider) -> u64 {
    // `effective_total_timeout` returns the model-aware default only when no
    // explicit value is configured, and the thinking-model default is the larger
    // of the two branches, so probing with a thinking model yields the maximum.
    provider.effective_total_timeout(THINKING_MODEL_PROBE)
}

/// Axum middleware enforcing the global request deadline.
///
/// Bounds the time the handler may take to *produce a response*, not the
/// lifetime of a streaming body: `next.run` resolves once the handler returns
/// headers and a body handle, so SSE relays keep streaming afterwards under
/// their own inter-chunk and total limits. Buffered requests, which await the
/// full upstream round-trip inside the handler, are bounded end to end.
pub async fn request_timeout_middleware(
    State(state): State<AppState>,
    request: Request,
    next: Next,
) -> Response {
    let Some(timeout) = state.runtime_limits.request_timeout() else {
        return next.run(request).await;
    };

    let method = request.method().clone();
    let uri = request.uri().clone();

    match tokio::time::timeout(timeout, next.run(request)).await {
        Ok(response) => response,
        Err(_) => {
            tracing::error!(
                method = %method,
                uri = %uri,
                timeout_seconds = timeout.as_secs(),
                "Request exceeded the global gateway deadline and was aborted"
            );
            gateway_timeout(timeout)
        }
    }
}

fn gateway_timeout(timeout: Duration) -> Response {
    (
        StatusCode::GATEWAY_TIMEOUT,
        Json(json!({
            "error": {
                "message": format!(
                    "Request exceeded the gateway deadline of {} seconds. This ceiling is derived from `server.request_timeout_seconds` and the longest configured provider `total_timeout_seconds`; raise whichever applies.",
                    timeout.as_secs()
                ),
                "type": "timeout_error",
                "code": "gateway_timeout"
            }
        })),
    )
        .into_response()
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::config::is_thinking_model;

    #[test]
    fn thinking_model_probe_actually_selects_the_thinking_defaults() {
        // `worst_case_total_timeout` is only correct if the probe is classified
        // as a thinking model; assert the invariant rather than assuming it.
        assert!(is_thinking_model(THINKING_MODEL_PROBE));
    }

    #[test]
    fn ceiling_never_truncates_a_configured_provider_timeout() {
        // Live-config shape: 30s server value, 800s provider total.
        assert_eq!(
            ceiling_seconds(30, 800),
            800 + PROVIDER_TIMEOUT_GRACE_SECONDS
        );
    }

    #[test]
    fn ceiling_honours_a_larger_server_value() {
        assert_eq!(ceiling_seconds(5_000, 60), 5_000);
    }

    #[test]
    fn ceiling_saturates_instead_of_overflowing() {
        assert_eq!(ceiling_seconds(0, u64::MAX), u64::MAX);
    }

    #[test]
    fn ceiling_with_no_providers_falls_back_to_the_server_value() {
        // `effective_request_timeout_seconds` passes 0 when no providers exist.
        assert_eq!(
            ceiling_seconds(30, 0),
            PROVIDER_TIMEOUT_GRACE_SECONDS.max(30)
        );
    }

    #[test]
    fn snapshot_publishes_and_republishes_values() {
        let limits = RuntimeLimits {
            max_request_size_bytes: AtomicUsize::new(0),
            request_timeout_seconds: AtomicU64::new(0),
        };

        limits.apply_parts(10 * 1024 * 1024, 860);
        assert_eq!(limits.max_request_size_bytes(), 10 * 1024 * 1024);
        assert_eq!(limits.request_timeout_seconds(), 860);
        assert_eq!(limits.request_timeout(), Some(Duration::from_secs(860)));

        limits.apply_parts(50 * 1024 * 1024, 4_000);
        assert_eq!(limits.max_request_size_bytes(), 50 * 1024 * 1024);
        assert_eq!(limits.request_timeout(), Some(Duration::from_secs(4_000)));
    }

    #[test]
    fn zero_timeout_disables_enforcement() {
        let limits = RuntimeLimits {
            max_request_size_bytes: AtomicUsize::new(0),
            request_timeout_seconds: AtomicU64::new(0),
        };

        assert_eq!(limits.request_timeout(), None);
    }
}
