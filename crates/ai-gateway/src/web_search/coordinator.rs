//! Web search coordinator — resolves which search feature (generic
//! `web_search` vs legacy Codex injection) is active for a request and
//! owns the generic backend lifecycle.
//!
//! Precedence (spec provider-agnostic-web-search-injection): when the
//! `web_search` config section is effective-enabled AND its backend is
//! ready, the generic feature wins and the legacy Codex injection is
//! suppressed. Otherwise the legacy Codex path behaves exactly as today.
//! Both absent ⇒ [`ActiveSearchConfig::None`].
//!
//! The router owns the config lock and passes the current
//! [`WebSearchConfig`] in; the coordinator never reads config itself.
//! Backends are built once per config epoch (compared by value) and
//! reused across requests — a hot-reload that changes any field rebuilds
//! the backend via [`WebSearchCoordinator::reset`], mirroring how
//! `clear_http_clients` pools HTTP clients.

use std::sync::Arc;

use tokio::sync::RwLock;
use tracing::warn;

use crate::web_search::backends::{
    resolve_api_key, BraveBackend, CodexSearchBackend, ExaBackend, SearchBackend,
    SearxngBackend, TavilyBackend,
};
use crate::web_search::config::WebSearchConfig;

/// Which search feature is active for the current request.
pub enum ActiveSearchConfig {
    /// Generic `web_search` tool: the resolved, ready backend plus the
    /// effective config knobs the round loop needs.
    Generic {
        backend: Arc<dyn SearchBackend>,
        config: WebSearchConfig,
    },
    /// The generic `web_search` feature is enabled but its backend is not
    /// ready (missing key, no OAuth session). Nothing is injected, and the
    /// legacy Codex injection is suppressed too: the operator explicitly
    /// chose the generic feature (Req 6.2, operator intent wins), so
    /// silently swapping in the Codex backend would surprise both model
    /// and billing expectations.
    GenericNotReady,
    /// Legacy Codex search config, exactly what `codex_search_ready`
    /// returns today.
    LegacyCodex(crate::codex::search::CodexSearchConfig),
    /// No search feature active: nothing is injected or intercepted.
    None,
}

/// A backend plus the config epoch it was built from.
struct CachedBackend {
    epoch: WebSearchConfig,
    backend: Arc<dyn SearchBackend>,
    /// Warn-once-per-epoch guard for the "enabled but not ready" log.
    warned_not_ready: bool,
}

/// Builds and caches the generic web-search backend, one instance per
/// config epoch. Never rebuilt per request.
#[derive(Default)]
pub struct WebSearchCoordinator {
    cached: RwLock<Option<CachedBackend>>,
    /// Test seam: when set, used instead of `build_backend` for the next
    /// miss on the current epoch. Lets integration tests construct the
    /// router with a stub `Arc<dyn SearchBackend>`.
    test_backend: RwLock<Option<Arc<dyn SearchBackend>>>,
}

impl WebSearchCoordinator {
    pub fn new() -> Self {
        Self::default()
    }

    /// Drop the cached backend so the next resolution rebuilds from the
    /// live config. Called on config hot-reload alongside
    /// `clear_http_clients`.
    pub async fn reset(&self) {
        *self.cached.write().await = None;
    }

    fn build_backend(
        config: &WebSearchConfig,
        oauth_manager: Option<&Arc<crate::oauth::OAuthManager>>,
    ) -> Arc<dyn SearchBackend> {
        let backend = config.backend.as_deref().unwrap_or_default();
        let timeout = config.effective_timeout_seconds();
        let api_key = config
            .api_key_env
            .as_deref()
            .map(resolve_api_key)
            .unwrap_or_default();
        match backend {
            "tavily" => match TavilyBackend::new(api_key, config.base_url.clone(), timeout) {
                Ok(b) => Arc::new(b),
                Err(_) => Arc::new(NeverReady),
            },
            "brave" => match BraveBackend::new(api_key, config.base_url.clone(), timeout) {
                Ok(b) => Arc::new(b),
                Err(_) => Arc::new(NeverReady),
            },
            "exa" => match ExaBackend::new(api_key, config.base_url.clone(), timeout) {
                Ok(b) => Arc::new(b),
                Err(_) => Arc::new(NeverReady),
            },
            "searxng" => match SearxngBackend::new(
                config.base_url.clone().unwrap_or_default(),
                timeout,
            ) {
                Ok(b) => Arc::new(b),
                Err(_) => Arc::new(NeverReady),
            },
            "codex" => match oauth_manager {
                Some(oauth) => {
                    let executor = Arc::new(
                        crate::codex::search::executor::SearchExecutor::new(
                            reqwest::Client::new(),
                            oauth.clone(),
                            Arc::new(crate::oauth::UsageTracker::new()),
                            Arc::new(crate::codex::search::metrics::SearchMetrics::new()),
                            crate::codex::search::config::CodexSearchConfig::default()
                                .effective_base_url(),
                            std::time::Duration::from_secs(timeout),
                        ),
                    );
                    let readiness = Arc::new(
                        crate::web_search::backends::codex::OAuthSessionReadiness::new(
                            oauth.clone(),
                        ),
                    );
                    Arc::new(CodexSearchBackend::new(executor, readiness))
                }
                None => Arc::new(NeverReady),
            },
            _ => Arc::new(NeverReady),
        }
    }

    /// Resolve the cached-or-built backend for `config`, probing
    /// readiness. Returns `None` when the backend is not ready (missing
    /// key, no OAuth session) with a warn-once-per-epoch log, or when the
    /// config is not effective-enabled.
    pub async fn resolve_backend(
        &self,
        config: &WebSearchConfig,
        oauth_manager: Option<&Arc<crate::oauth::OAuthManager>>,
    ) -> Option<Arc<dyn SearchBackend>> {
        if !config.effective_enabled() {
            return None;
        }
        let backend = {
            let mut cached = self.cached.write().await;
            let needs_build = cached.as_ref().map_or(true, |c| c.epoch != *config);
            if needs_build {
                let backend = self
                    .test_backend
                    .write()
                    .await
                    .take()
                    .unwrap_or_else(|| Self::build_backend(config, oauth_manager));
                *cached = Some(CachedBackend {
                    epoch: config.clone(),
                    backend,
                    warned_not_ready: false,
                });
            }
            cached.as_ref().expect("just populated").backend.clone()
        };
        if !backend.ready().await {
            let mut cached = self.cached.write().await;
            if let Some(slot) = cached.as_mut() {
                if slot.epoch == *config && !slot.warned_not_ready {
                    warn!(
                        backend = %config.backend.as_deref().unwrap_or_default(),
                        "web_search enabled but backend not ready (missing API key or OAuth session); feature dormant until config reload"
                    );
                    slot.warned_not_ready = true;
                }
            }
            return None;
        }
        Some(backend)
    }

    /// Readiness snapshot for the admin surface: `None` when the feature
    /// is disabled, `Some((backend_name, ready))` otherwise. Does not
    /// log and does not mutate the warn-once flag.
    pub async fn readiness(
        &self,
        config: &WebSearchConfig,
        oauth_manager: Option<&Arc<crate::oauth::OAuthManager>>,
    ) -> Option<(&'static str, bool)> {
        if !config.effective_enabled() {
            return None;
        }
        let backend = {
            let cached = self.cached.read().await;
            match cached.as_ref() {
                Some(slot) if slot.epoch == *config => Some(slot.backend.clone()),
                _ => None,
            }
        };
        match backend {
            Some(backend) => {
                let ready = backend.ready().await;
                Some((backend.name(), ready))
            }
            None => {
                let backend = Self::build_backend(config, oauth_manager);
                let name = backend.name();
                let ready = backend.ready().await;
                Some((name, ready))
            }
        }
    }

    /// Test seam: queue a stub backend used for the next epoch miss in
    /// [`Self::resolve_backend`] instead of constructing from config.
    pub async fn set_test_backend(&self, backend: Arc<dyn SearchBackend>) {
        *self.test_backend.write().await = Some(backend);
        *self.cached.write().await = None;
    }
}

/// Char cap for the query string carried on web_search round ledger
/// rows (`error_class` detail). The query is model-authored text, so
/// it is size-truncated like the Codex executor's log previews rather
/// than persisted whole (Req 5.2: never keys, never full results).
pub const WEB_SEARCH_LOG_QUERY_MAX_CHARS: usize = 200;

/// Status word for a web_search round's attempt-ledger row: one of
/// `success`, `error`, `validation-error`, `budget-exhausted`.
///
/// The row mirrors the `#attempt` pattern (`ProviderAttempt` through
/// `ActiveRequestHandle::record_failed_attempt`): `model` carries the
/// tool name (`web_search`), `provider` carries the backend name,
/// `duration_ms` the backend-call time, the content-free class is
/// `web_search_<status>` (snake_case, ≤ 32 chars, like every other
/// class), and the size-truncated query rides the error string. The
/// status code keeps the row out of the dashboard Errors tab unless
/// something went wrong: 200 success, 502 backend failure (matches
/// `ATTEMPT_FALLBACK_STATUS` for statusless upstream errors), 400
/// invalid tool arguments, 429 iteration budget spent (the rate-limit
/// family convention).
pub fn web_search_round_attempt(
    backend_name: &str,
    status: &str,
    query: Option<&str>,
    elapsed: std::time::Duration,
) -> crate::error::ProviderAttempt {
    let status_code = match status {
        "success" => 200,
        "validation-error" => 400,
        "budget-exhausted" => 429,
        _ => 502,
    };
    let detail = match query {
        Some(q) => format!(
            "web_search round: {status} query={}",
            truncate_query_for_log(q)
        ),
        None => format!("web_search round: {status}"),
    };
    crate::error::ProviderAttempt::new(
        backend_name.to_string(),
        crate::web_search::injector::WEB_SEARCH_TOOL_NAME.to_string(),
        detail,
        Some(status_code),
    )
    .with_duration(elapsed)
    .with_error_class(&format!("web_search_{}", status.replace('-', "_")))
}

/// Size-truncate `q` to [`WEB_SEARCH_LOG_QUERY_MAX_CHARS`] chars on a
/// UTF-8 boundary (multibyte-safe, same approach as the Codex
/// executor's previews) and strip control characters except space
/// before truncation. Never receives keys — only the model's query
/// text.
pub fn truncate_query_for_log(q: &str) -> String {
    let sanitized: String = q
        .chars()
        .filter(|c| !c.is_control() || *c == ' ')
        .collect();
    if sanitized.chars().count() <= WEB_SEARCH_LOG_QUERY_MAX_CHARS {
        return sanitized;
    }
    sanitized
        .chars()
        .take(WEB_SEARCH_LOG_QUERY_MAX_CHARS)
        .collect()
}

/// Placeholder backend for unknown names and construction failures; its
/// `ready` is `false` so the feature stays dormant rather than erroring
/// per request.
struct NeverReady;

#[async_trait::async_trait]
impl SearchBackend for NeverReady {
    async fn execute(
        &self,
        _query: &crate::web_search::models::SearchQuery,
    ) -> Result<Vec<crate::web_search::models::SearchResult>, crate::web_search::backends::BackendError>
    {
        Err(crate::web_search::backends::BackendError::new(
            "web_search backend unavailable",
        ))
    }

    fn name(&self) -> &'static str {
        "none"
    }

    async fn ready(&self) -> bool {
        false
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::web_search::backends::BackendError;
    use crate::web_search::models::{SearchQuery, SearchResult};
    use std::time::Duration;

    struct StubBackend {
        ready: bool,
    }

    #[async_trait::async_trait]
    impl SearchBackend for StubBackend {
        async fn execute(&self, _query: &SearchQuery) -> Result<Vec<SearchResult>, BackendError> {
            Ok(vec![SearchResult {
                title: "t".to_string(),
                url: "https://example.com".to_string(),
                snippet: "s".to_string(),
            }])
        }

        fn name(&self) -> &'static str {
            "stub"
        }

        async fn ready(&self) -> bool {
            self.ready
        }
    }

    fn enabled_config(backend: &str) -> WebSearchConfig {
        WebSearchConfig {
            enabled: Some(true),
            backend: Some(backend.to_string()),
            api_key_env: Some("WEB_SEARCH_TEST_KEY_UNSET".to_string()),
            ..Default::default()
        }
    }

    #[tokio::test]
    async fn disabled_config_yields_none() {
        let coordinator = WebSearchCoordinator::new();
        let config = WebSearchConfig::default();
        assert!(coordinator.resolve_backend(&config, None).await.is_none());
    }

    #[tokio::test]
    async fn stub_backend_resolves_when_ready() {
        let coordinator = WebSearchCoordinator::new();
        let config = enabled_config("tavily");
        coordinator
            .set_test_backend(Arc::new(StubBackend { ready: true }))
            .await;
        let backend = coordinator
            .resolve_backend(&config, None)
            .await
            .expect("ready stub resolves");
        assert_eq!(backend.name(), "stub");
    }

    #[tokio::test]
    async fn not_ready_backend_yields_none_but_readiness_reports() {
        let coordinator = WebSearchCoordinator::new();
        let config = enabled_config("tavily");
        coordinator
            .set_test_backend(Arc::new(StubBackend { ready: false }))
            .await;
        assert!(coordinator.resolve_backend(&config, None).await.is_none());
        let (name, ready) = coordinator
            .readiness(&config, None)
            .await
            .expect("readiness still reports");
        assert_eq!(name, "stub");
        assert!(!ready);
    }

    #[tokio::test]
    async fn epoch_change_rebuilds_backend() {
        let coordinator = WebSearchCoordinator::new();
        let config = enabled_config("tavily");
        coordinator
            .set_test_backend(Arc::new(StubBackend { ready: true }))
            .await;
        let first = coordinator
            .resolve_backend(&config, None)
            .await
            .expect("stub resolves");
        assert_eq!(first.name(), "stub");
        // Same epoch: cached, identical instance, no rebuild.
        let again = coordinator
            .resolve_backend(&config, None)
            .await
            .expect("cached stub resolves");
        assert_eq!(again.name(), "stub");
        // Different epoch (no stub queued): a real Tavily backend is
        // built. With the test's env unset, resolve_api_key falls back
        // to the literal env-var name, which counts as a key, so the
        // rebuilt backend IS ready — assert the rebuild happened by
        // name change, not by dormancy.
        let mut changed = config.clone();
        changed.timeout_seconds = Some(30);
        let rebuilt = coordinator
            .resolve_backend(&changed, None)
            .await
            .expect("rebuilt tavily resolves");
        assert_eq!(rebuilt.name(), "tavily");
    }

    #[tokio::test]
    async fn reset_clears_cache() {
        let coordinator = WebSearchCoordinator::new();
        let config = enabled_config("tavily");
        coordinator
            .set_test_backend(Arc::new(StubBackend { ready: true }))
            .await;
        let first = coordinator
            .resolve_backend(&config, None)
            .await
            .expect("stub resolves");
        assert_eq!(first.name(), "stub");
        coordinator.reset().await;
        // After reset with no stub queued, a real backend is built
        // (ready per env-fallback key semantics) — the name change
        // proves the cache was cleared.
        let rebuilt = coordinator
            .resolve_backend(&config, None)
            .await
            .expect("rebuilt tavily resolves");
        assert_eq!(rebuilt.name(), "tavily");
    }

    #[tokio::test]
    async fn missing_key_backends_not_ready() {
        let coordinator = WebSearchCoordinator::new();
        for backend in ["tavily", "brave", "exa"] {
            let config = enabled_config(backend);
            // resolve_api_key falls back to the literal env name; a key
            // that IS the env-var name is treated as missing by the
            // backend's ready() check only when empty — so assert via a
            // genuinely empty key env instead.
            let mut empty = config.clone();
            empty.api_key_env = None;
            assert!(
                coordinator.resolve_backend(&empty, None).await.is_none(),
                "{backend} without key must be dormant"
            );
        }
    }

    #[tokio::test]
    async fn codex_backend_without_oauth_not_ready() {
        let coordinator = WebSearchCoordinator::new();
        let config = WebSearchConfig {
            enabled: Some(true),
            backend: Some("codex".to_string()),
            ..Default::default()
        };
        assert!(coordinator.resolve_backend(&config, None).await.is_none());
    }

    #[tokio::test]
    async fn codex_backend_readiness_reports_name() {
        let coordinator = WebSearchCoordinator::new();
        let config = WebSearchConfig {
            enabled: Some(true),
            backend: Some("codex".to_string()),
            ..Default::default()
        };
        let (name, ready) = coordinator
            .readiness(&config, None)
            .await
            .expect("readiness reports");
        assert_eq!(name, "none");
        assert!(!ready);
    }

    #[tokio::test]
    async fn unknown_backend_name_is_never_ready() {
        let coordinator = WebSearchCoordinator::new();
        let config = WebSearchConfig {
            enabled: Some(true),
            backend: Some("does-not-exist".to_string()),
            ..Default::default()
        };
        assert!(coordinator.resolve_backend(&config, None).await.is_none());
    }

    #[tokio::test]
    async fn searxng_backend_builds_and_is_ready() {
        let coordinator = WebSearchCoordinator::new();
        let config = WebSearchConfig {
            enabled: Some(true),
            backend: Some("searxng".to_string()),
            base_url: Some("http://127.0.0.1:1/search".to_string()),
            ..Default::default()
        };
    let backend = coordinator
        .resolve_backend(&config, None)
        .await
        .expect("searxng needs no key");
    assert_eq!(backend.name(), "searxng");
    let _ = Duration::from_secs(0);
}

#[test]
fn round_attempt_row_shape_mirrors_attempt_pattern() {
    let attempt = web_search_round_attempt(
        "tavily",
        "success",
        Some("rust async"),
        Duration::from_millis(42),
    );
    assert_eq!(attempt.provider, "tavily");
    assert_eq!(attempt.model, "web_search");
    assert_eq!(attempt.status_code, Some(200));
    assert_eq!(attempt.error_class.as_deref(), Some("web_search_success"));
    assert_eq!(attempt.duration_ms, Some(42));
    assert!(attempt.error.contains("query=rust async"));

    let failed = web_search_round_attempt(
        "stub",
        "error",
        Some("q"),
        Duration::from_millis(1),
    );
    assert_eq!(failed.status_code, Some(502));
    assert_eq!(failed.error_class.as_deref(), Some("web_search_error"));
}

#[test]
fn round_attempt_row_without_query_is_key_free() {
    let attempt = web_search_round_attempt(
        "searxng",
        "budget-exhausted",
        None,
        Duration::ZERO,
    );
    assert!(!attempt.error.contains("api_key"));
    assert!(!attempt.error.contains("key="));
    assert_eq!(
        attempt.error_class.as_deref(),
        Some("web_search_budget_exhausted")
    );
    assert_eq!(attempt.status_code, Some(429));
}

    #[test]
    fn query_truncation_caps_at_200_chars_and_preserves_utf8() {
        let long = "x".repeat(WEB_SEARCH_LOG_QUERY_MAX_CHARS + 50);
        let truncated = truncate_query_for_log(&long);
        assert_eq!(truncated.chars().count(), WEB_SEARCH_LOG_QUERY_MAX_CHARS);

        let short = "short";
        assert_eq!(truncate_query_for_log(short), short);

        let multibyte = "\u{1F600}".repeat(300);
        let truncated = truncate_query_for_log(&multibyte);
        assert_eq!(truncated.chars().count(), WEB_SEARCH_LOG_QUERY_MAX_CHARS);
    }

    #[test]
    fn query_sanitizer_strips_control_characters_but_keeps_space() {
        let dirty = "ru\u{0000}st \u{0007}async\u{001B}[0m\ttokio";
        let sanitized = truncate_query_for_log(dirty);
        assert_eq!(sanitized, "rust async[0mtokio");
        assert!(!sanitized.chars().any(|c| c.is_control() && c != ' '));

        let plain = "a b c";
        assert_eq!(truncate_query_for_log(plain), plain);
    }

    #[test]
    fn query_sanitizer_applies_before_truncation() {
        let dirty = format!("a\u{0000}b{}", "x".repeat(WEB_SEARCH_LOG_QUERY_MAX_CHARS));
        let truncated = truncate_query_for_log(&dirty);
        assert_eq!(truncated.chars().count(), WEB_SEARCH_LOG_QUERY_MAX_CHARS);
        assert!(!truncated.contains('\u{0000}'));
        assert!(truncated.starts_with("ab"));
    }

    #[test]
    fn round_attempt_query_detail_is_sanitized() {
        let attempt = web_search_round_attempt(
            "tavily",
            "success",
            Some("bad\u{0007}query"),
            Duration::from_millis(1),
        );
        assert!(attempt.error.contains("query=badquery"));
        assert!(!attempt.error.contains('\u{0007}'));
    }

#[test]
fn validation_error_row_uses_validation_status() {
    let attempt = web_search_round_attempt("tavily", "validation-error", None, Duration::ZERO);
    assert_eq!(attempt.status_code, Some(400));
    assert_eq!(
        attempt.error_class.as_deref(),
        Some("web_search_validation_error")
    );
}
}
