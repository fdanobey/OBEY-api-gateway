//! Model discovery for the Jev classifier: resolve which Jev model to use at
//! the configured base URL.
//!
//! `auto` lists models via `GET {base_url}/v1/models`, filters Jev-capable
//! ids, and selects the latest by version ordering. A pinned model is used
//! verbatim. Results are cached for a configurable TTL; discovery failure
//! never blocks classification — a cached selection survives, and without one
//! the caller falls back with a rate-limited warning.

use std::sync::Mutex;
use std::time::{Duration, Instant};

use crate::smart_routing::config::JevConfig;

use super::models::ModelsListing;

/// Resolved model identifier plus how it was obtained.
#[derive(Debug, Clone, PartialEq)]
pub struct ResolvedModel {
    pub model: String,
    pub origin: ResolutionOrigin,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ResolutionOrigin {
    /// Selected by `latest_jev` from a fresh listing.
    Auto,
    /// Served from the TTL cache (same model as a prior resolution).
    Cached,
    /// Pinned explicitly by configuration.
    Pinned,
}

/// Discovery failures mapped for the caller's fallback decision.
#[derive(Debug, Clone, PartialEq)]
pub enum DiscoveryError {
    /// The endpoint listing succeeded but contains no Jev-capable model.
    NoJevAtEndpoint,
    /// A pinned model was requested but is absent from the listing.
    ModelUnavailable { model: String },
    /// The listing request failed (transport, auth, status, body).
    Listing(String),
}

/// True when a model id is Jev-capable: a word-delimited, case-insensitive
/// `jev` token in the id (e.g. `jev-1.13.0`, `typesafe/jev-1.13`, `Jev`).
/// `jealous-model` and `notjev` do not qualify.
pub fn is_jev_model(id: &str) -> bool {
    id.split(|character: char| !character.is_ascii_alphanumeric())
        .any(|token| token.eq_ignore_ascii_case("jev"))
}

/// Select the latest Jev-capable model from a listing of ids.
///
/// Version ordering: trailing semantic-version segments (`jev-1.13.0`,
/// `typesafe/jev-2.0.1`) compare numerically field-by-field; ids without a
/// parsable version compare lexicographically; versioned ids rank above
/// unversioned ones.
pub fn latest_jev(ids: &[String]) -> Option<String> {
    ids.iter()
        .filter(|id| is_jev_model(id))
        .max_by(|left, right| compare_jev_ids(left, right))
        .cloned()
}

/// Compare two Jev model ids for recency.
fn compare_jev_ids(left: &str, right: &str) -> std::cmp::Ordering {
    let left_version = parse_trailing_version(left);
    let right_version = parse_trailing_version(right);
    match (left_version, right_version) {
        (Some(left_version), Some(right_version)) => left_version.cmp(&right_version),
        (Some(_), None) => std::cmp::Ordering::Greater,
        (None, Some(_)) => std::cmp::Ordering::Less,
        (None, None) => left.cmp(right),
    }
}

/// Parse a trailing semantic-version suffix from a model id: the final
/// `-`/`/`-separated segment consisting of dot-separated numbers.
fn parse_trailing_version(id: &str) -> Option<Vec<u64>> {
    let tail = id.rsplit(['-', '/']).next()?;
    if tail.is_empty() || !tail.chars().all(|c| c.is_ascii_digit() || c == '.') {
        return None;
    }
    let fields: Vec<u64> = tail
        .split('.')
        .map(|field| field.parse::<u64>().ok())
        .collect::<Option<Vec<_>>>()?;
    if fields.is_empty() {
        return None;
    }
    Some(fields)
}

/// Fetcher boundary so discovery logic is testable without network.
pub trait ModelsFetcher: Send + Sync {
    fn fetch(&self) -> Result<ModelsListing, String>;
}

/// TTL-cached model resolution.
pub struct ModelDiscovery {
    config_model: String,
    ttl: Duration,
    state: Mutex<DiscoveryState>,
    warned_no_jev_at: Mutex<Option<Instant>>,
}

/// Internal cache. Guarded by a short-lived mutex; never held across awaits
/// because resolution is fully synchronous once the fetcher returns.
#[derive(Debug, Default)]
pub enum DiscoveryState {
    #[default]
    Empty,
    Resolved {
        model: String,
        resolved_at: Instant,
    },
    Failed {
        error: DiscoveryError,
        failed_at: Instant,
    },
}

impl ModelDiscovery {
    pub fn new(config: &JevConfig) -> Self {
        Self {
            config_model: config.model.trim().to_string(),
            ttl: Duration::from_secs(config.discovery_ttl_secs),
            state: Mutex::new(DiscoveryState::Empty),
            warned_no_jev_at: Mutex::new(None),
        }
    }

    /// Resolve the model to use, consulting the TTL cache first.
    ///
    /// Pinned models return immediately. `auto` uses a cached resolution when
    /// fresh; otherwise it fetches, selects, and caches. Warnings for a
    /// Jev-less endpoint fire at most once per TTL window.
    pub fn resolve(&self, fetcher: &dyn ModelsFetcher) -> Result<ResolvedModel, DiscoveryError> {
        if !self.config_model.is_empty() && self.config_model != "auto" {
            return Ok(ResolvedModel {
                model: self.config_model.clone(),
                origin: ResolutionOrigin::Pinned,
            });
        }

        if let Some(cached) = self.cached() {
            return cached;
        }

        let listing = match fetcher.fetch() {
            Ok(listing) => listing,
            Err(message) => {
                let error = DiscoveryError::Listing(message);
                self.cache_failure(error.clone());
                return Err(error);
            }
        };
        let ids: Vec<String> = listing.data.into_iter().map(|entry| entry.id).collect();
        let Some(model) = latest_jev(&ids) else {
            self.warn_no_jev_once();
            let error = DiscoveryError::NoJevAtEndpoint;
            self.cache_failure(error.clone());
            return Err(error);
        };

        *self
            .state
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner) = DiscoveryState::Resolved {
            model: model.clone(),
            resolved_at: Instant::now(),
        };
        Ok(ResolvedModel {
            model,
            origin: ResolutionOrigin::Auto,
        })
    }

    fn cached(&self) -> Option<Result<ResolvedModel, DiscoveryError>> {
        let state = self
            .state
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner);
        match &*state {
            DiscoveryState::Resolved { model, resolved_at } if resolved_at.elapsed() < self.ttl => {
                Some(Ok(ResolvedModel {
                    model: model.clone(),
                    origin: ResolutionOrigin::Cached,
                }))
            }
            DiscoveryState::Failed { error, failed_at } if failed_at.elapsed() < self.ttl => {
                Some(Err(error.clone()))
            }
            DiscoveryState::Empty
            | DiscoveryState::Resolved { .. }
            | DiscoveryState::Failed { .. } => None,
        }
    }

    fn cache_failure(&self, error: DiscoveryError) {
        *self
            .state
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner) = DiscoveryState::Failed {
            error,
            failed_at: Instant::now(),
        };
    }

    /// Rate-limited warning: at most once per TTL window. The configured URL
    /// is named so operators know which endpoint lacks Jev.
    fn warn_no_jev_once(&self) {
        let mut warned_at = self
            .warned_no_jev_at
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner);
        let should_warn = warned_at.map(|at| at.elapsed() >= self.ttl).unwrap_or(true);
        if should_warn {
            tracing::warn!(
                ttl_secs = self.ttl.as_secs(),
                "Jev classifier: no Jev-capable model at the configured endpoint; \
                 falling back to the existing classifier chain"
            );
            *warned_at = Some(Instant::now());
        }
    }

    /// Clear cached resolution (used on config hot-reload).
    pub fn invalidate(&self) {
        *self
            .state
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner) = DiscoveryState::Empty;
    }

    /// Read-only view of the cached resolution; never fetches, so it cannot
    /// block on network or poison the cache.
    pub fn resolved_model(&self) -> Option<String> {
        let state = self
            .state
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner);
        match &*state {
            DiscoveryState::Resolved { model, .. } => Some(model.clone()),
            DiscoveryState::Failed { .. } | DiscoveryState::Empty => None,
        }
    }
}

#[cfg(test)]
mod tests {
    use std::sync::atomic::{AtomicUsize, Ordering};

    use super::*;
    use proptest::prelude::*;

    struct CountingFetcher {
        calls: AtomicUsize,
        result: Result<ModelsListing, String>,
    }

    impl CountingFetcher {
        fn listing(ids: &[&str]) -> Self {
            Self {
                calls: AtomicUsize::new(0),
                result: Ok(ModelsListing {
                    data: ids
                        .iter()
                        .map(|id| super::super::models::ModelEntry {
                            id: id.to_string(),
                            owned_by: None,
                        })
                        .collect(),
                }),
            }
        }

        fn failing() -> Self {
            Self {
                calls: AtomicUsize::new(0),
                result: Err("transport error".to_string()),
            }
        }

        fn calls(&self) -> usize {
            self.calls.load(Ordering::SeqCst)
        }
    }

    impl ModelsFetcher for CountingFetcher {
        fn fetch(&self) -> Result<ModelsListing, String> {
            self.calls.fetch_add(1, Ordering::SeqCst);
            self.result.clone()
        }
    }

    fn config_with(model: &str, ttl_secs: u64) -> JevConfig {
        JevConfig {
            model: model.to_string(),
            discovery_ttl_secs: ttl_secs,
            ..JevConfig::default()
        }
    }

    #[test]
    fn jev_id_detection_is_word_delimited() {
        for id in [
            "jev",
            "jev-1.13.0",
            "typesafe/jev-1.13",
            "Jev-2.0",
            "JEV",
            "vendor_jev_pro",
        ] {
            assert!(is_jev_model(id), "{id} should be Jev-capable");
        }
        for id in ["jealous-model", "notjev", "gpt-4o-mini", "jevvia"] {
            assert!(!is_jev_model(id), "{id} should not be Jev-capable");
        }
    }

    #[test]
    fn latest_jev_prefers_highest_version() {
        let ids: Vec<String> = ["gpt-4o", "jev-1.9.0", "jev-1.13.0", "jev-1.13.0-rc1"]
            .iter()
            .map(|id| id.to_string())
            .collect();
        assert_eq!(latest_jev(&ids).as_deref(), Some("jev-1.13.0"));

        let ids: Vec<String> = ["jev-1.9", "jev-1.13", "jev-2.0"]
            .iter()
            .map(|id| id.to_string())
            .collect();
        assert_eq!(latest_jev(&ids).as_deref(), Some("jev-2.0"));

        let ids: Vec<String> = ["typesafe/jev-1.12.2", "typesafe/jev-1.13.0"]
            .iter()
            .map(|id| id.to_string())
            .collect();
        assert_eq!(latest_jev(&ids).as_deref(), Some("typesafe/jev-1.13.0"));
    }

    #[test]
    fn latest_jev_numeric_fields_compare_not_lexicographically() {
        let ids: Vec<String> = ["jev-1.9.0", "jev-1.13.0"]
            .iter()
            .map(|id| id.to_string())
            .collect();
        assert_eq!(latest_jev(&ids).as_deref(), Some("jev-1.13.0"));
    }

    #[test]
    fn latest_jev_falls_back_to_lexicographic_for_unversioned() {
        let ids: Vec<String> = ["jev-alpha", "jev-beta"]
            .iter()
            .map(|id| id.to_string())
            .collect();
        assert_eq!(latest_jev(&ids).as_deref(), Some("jev-beta"));

        // Versioned outranks unversioned.
        let ids: Vec<String> = ["jev-zeta", "jev-0.1.0"]
            .iter()
            .map(|id| id.to_string())
            .collect();
        assert_eq!(latest_jev(&ids).as_deref(), Some("jev-0.1.0"));
    }

    #[test]
    fn latest_jev_returns_none_without_jev_models() {
        let ids: Vec<String> = ["gpt-4o", "claude-3"]
            .iter()
            .map(|id| id.to_string())
            .collect();
        assert_eq!(latest_jev(&ids), None);
    }

    #[test]
    fn pinned_model_bypasses_fetch_and_cache() {
        let discovery = ModelDiscovery::new(&config_with("jev-1.13.0", 600));
        let fetcher = CountingFetcher::listing(&["jev-1.13.0"]);

        let resolved = discovery.resolve(&fetcher).unwrap();
        assert_eq!(
            resolved,
            ResolvedModel {
                model: "jev-1.13.0".to_string(),
                origin: ResolutionOrigin::Pinned,
            }
        );
        assert_eq!(fetcher.calls(), 0);

        // Pinned resolution is not cached: a later listing change does not
        // matter and no fetch happens either.
        assert!(discovery.resolve(&fetcher).is_ok());
        assert_eq!(fetcher.calls(), 0);
    }

    #[test]
    fn auto_resolution_caches_within_ttl() {
        let discovery = ModelDiscovery::new(&config_with("auto", 600));
        let fetcher = CountingFetcher::listing(&["jev-1.12.0", "jev-1.13.0"]);

        let first = discovery.resolve(&fetcher).unwrap();
        assert_eq!(first.origin, ResolutionOrigin::Auto);
        assert_eq!(first.model, "jev-1.13.0");
        assert_eq!(fetcher.calls(), 1);

        let second = discovery.resolve(&fetcher).unwrap();
        assert_eq!(second.origin, ResolutionOrigin::Cached);
        assert_eq!(second.model, "jev-1.13.0");
        assert_eq!(fetcher.calls(), 1);
    }

    #[test]
    fn ttl_expiry_refetches() {
        let discovery = ModelDiscovery::new(&config_with("auto", 0));
        let fetcher = CountingFetcher::listing(&["jev-1.13.0"]);

        // TTL 0: every resolution is a fresh fetch.
        assert_eq!(
            discovery.resolve(&fetcher).unwrap().origin,
            ResolutionOrigin::Auto
        );
        assert_eq!(
            discovery.resolve(&fetcher).unwrap().origin,
            ResolutionOrigin::Auto
        );
        assert_eq!(fetcher.calls(), 2);
    }

    #[test]
    fn no_jev_at_endpoint_is_a_typed_error() {
        let discovery = ModelDiscovery::new(&config_with("auto", 600));
        let fetcher = CountingFetcher::listing(&["gpt-4o", "claude-3"]);

        assert_eq!(
            discovery.resolve(&fetcher),
            Err(DiscoveryError::NoJevAtEndpoint)
        );
    }

    #[test]
    fn listing_failure_is_a_typed_error() {
        let discovery = ModelDiscovery::new(&config_with("auto", 600));
        let fetcher = CountingFetcher::failing();

        assert!(matches!(
            discovery.resolve(&fetcher),
            Err(DiscoveryError::Listing(_))
        ));
    }

    #[test]
    fn listing_failures_are_negatively_cached_within_ttl() {
        let discovery = ModelDiscovery::new(&config_with("auto", 600));
        let fetcher = CountingFetcher::failing();

        assert!(discovery.resolve(&fetcher).is_err());
        assert!(discovery.resolve(&fetcher).is_err());
        assert_eq!(fetcher.calls(), 1, "failure must be cached, not refetched");

        discovery.invalidate();
        assert!(discovery.resolve(&fetcher).is_err());
        assert_eq!(
            fetcher.calls(),
            2,
            "invalidate must clear the failure cache"
        );
    }

    #[test]
    fn no_jev_at_endpoint_is_negatively_cached_within_ttl() {
        let discovery = ModelDiscovery::new(&config_with("auto", 600));
        let fetcher = CountingFetcher::listing(&["gpt-4o"]);

        assert!(matches!(
            discovery.resolve(&fetcher),
            Err(DiscoveryError::NoJevAtEndpoint)
        ));
        assert!(matches!(
            discovery.resolve(&fetcher),
            Err(DiscoveryError::NoJevAtEndpoint)
        ));
        assert_eq!(
            fetcher.calls(),
            1,
            "missing-Jev must not refetch per request"
        );
    }

    #[test]
    fn resolved_model_reads_cache_without_fetching() {
        let discovery = ModelDiscovery::new(&config_with("auto", 600));
        let fetcher = CountingFetcher::listing(&["jev-1.13.0"]);

        assert!(discovery.resolved_model().is_none());
        discovery.resolve(&fetcher).unwrap();
        assert_eq!(discovery.resolved_model().as_deref(), Some("jev-1.13.0"));
        assert_eq!(fetcher.calls(), 1, "status probe must never fetch");
    }

    #[test]
    fn invalidate_forces_refetch() {
        let discovery = ModelDiscovery::new(&config_with("auto", 600));
        let fetcher = CountingFetcher::listing(&["jev-1.13.0"]);

        discovery.resolve(&fetcher).unwrap();
        discovery.invalidate();
        let resolved = discovery.resolve(&fetcher).unwrap();
        assert_eq!(resolved.origin, ResolutionOrigin::Auto);
        assert_eq!(fetcher.calls(), 2);
    }

    proptest! {
        #![proptest_config(ProptestConfig::with_cases(64))]

        #[test]
        fn property_latest_jev_always_returns_a_jev_model(
            ids in prop::collection::vec(
                prop_oneof![
                    Just("jev-1.0.0".to_string()),
                    Just("jev-2.5.3".to_string()),
                    Just("typesafe/jev-0.9.1".to_string()),
                    Just("gpt-4o-mini".to_string()),
                    Just("claude-3.5".to_string()),
                    Just("jealous".to_string()),
                ],
                0..=32
            ),
        ) {
            let selection = latest_jev(&ids);
            match selection {
                Some(model) => prop_assert!(is_jev_model(&model)),
                None => prop_assert!(ids.iter().all(|id| !is_jev_model(id))),
            }
        }

        #[test]
        fn property_version_ordering_is_transitive_on_examples(
            major in 0u32..=20,
            minor in 0u32..=20,
            patch in 0u32..=20,
        ) {
            let lower = format!("jev-{major}.{minor}.{patch}");
            let higher = format!("jev-{}.{}.{}", major + 1, minor, patch);
            let ids = vec![lower.clone(), higher.clone()];
            prop_assert_eq!(latest_jev(&ids), Some(higher.clone()));

            let higher_minor = format!("jev-{major}.{}.{}", minor + 1, patch);
            let ids = vec![lower.clone(), higher_minor.clone()];
            prop_assert_eq!(latest_jev(&ids), Some(higher_minor.clone()));

            let higher_patch = format!("jev-{major}.{minor}.{}", patch + 1);
            let ids = vec![lower, higher_patch.clone()];
            prop_assert_eq!(latest_jev(&ids), Some(higher_patch.clone()));
        }
    }
}
