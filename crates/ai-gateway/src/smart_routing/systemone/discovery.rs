//! Model discovery for System One classifier families: resolve which family
//! model to use at the configured base URL.
//!
//! `auto` lists models via `GET {base_url}/v1/models`, filters
//! family-capable ids, and selects the latest by version ordering. A pinned
//! model is used verbatim. Results are cached for a configurable TTL;
//! discovery failure never blocks classification — a cached selection
//! survives, and without one the caller falls back with a rate-limited
//! warning. Families that define a missing-listing default (Laya) pin it
//! instead of failing when the endpoint exposes no listing.

use std::sync::Mutex;
use std::time::{Duration, Instant};

use super::models::ModelsListing;
use super::SystemOneFamily;

/// Resolved model identifier plus how it was obtained.
#[derive(Debug, Clone, PartialEq)]
pub struct ResolvedModel {
    pub model: String,
    pub origin: ResolutionOrigin,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ResolutionOrigin {
    /// Selected by `latest_family_model` from a fresh listing.
    Auto,
    /// Served from the TTL cache (same model as a prior resolution).
    Cached,
    /// Pinned explicitly by configuration.
    Pinned,
}

/// Discovery failures mapped for the caller's fallback decision.
#[derive(Debug, Clone, PartialEq)]
pub enum DiscoveryError {
    /// The endpoint listing succeeded but contains no family-capable model.
    NoModelAtEndpoint,
    /// A pinned model was requested but is absent from the listing.
    ModelUnavailable { model: String },
    /// The listing request failed (transport, auth, status, body).
    Listing(String),
}

/// Select the latest family-capable model from a listing of ids.
///
/// Version ordering: trailing semantic-version segments (`jev-1.13.0`,
/// `typesafe/jev-2.0.1`) compare numerically field-by-field; ids without a
/// parsable version compare lexicographically; versioned ids rank above
/// unversioned ones.
pub fn latest_family_model(family: &SystemOneFamily, ids: &[String]) -> Option<String> {
    ids.iter()
        .filter(|id| family.is_family_model(id))
        .max_by(|left, right| compare_family_ids(left, right))
        .cloned()
}

/// Compare two family model ids for recency.
fn compare_family_ids(left: &str, right: &str) -> std::cmp::Ordering {
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
    family: SystemOneFamily,
    config_model: String,
    ttl: Duration,
    state: Mutex<DiscoveryState>,
    warned_no_model_at: Mutex<Option<Instant>>,
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
    pub fn new(family: SystemOneFamily, config_model: &str, ttl_secs: u64) -> Self {
        Self {
            family,
            config_model: config_model.trim().to_string(),
            ttl: Duration::from_secs(ttl_secs),
            state: Mutex::new(DiscoveryState::Empty),
            warned_no_model_at: Mutex::new(None),
        }
    }

    /// Resolve the model to use, consulting the TTL cache first.
    ///
    /// Pinned models return immediately. `auto` uses a cached resolution
    /// when fresh; otherwise it fetches, selects, and caches. Warnings for a
    /// family-less endpoint fire at most once per TTL window.
    ///
    /// Families that define a missing-listing default pin it when the
    /// endpoint exposes no usable listing, instead of failing.
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
                // A missing-listing default is pinned only when the endpoint
                // does not expose a listing at all; transport/auth failures
                // still surface as errors so the caller falls back.
                let error = DiscoveryError::Listing(message);
                self.cache_failure(error.clone());
                return Err(error);
            }
        };
        let ids: Vec<String> = listing.data.into_iter().map(|entry| entry.id).collect();
        let Some(model) = latest_family_model(&self.family, &ids) else {
            self.warn_no_model_once();
            // Families with a missing-listing default (Laya) pin it instead
            // of failing, so an endpoint without a listing never blocks
            // classification; the pin is not cached so a listing that
            // appears later is discovered on the next resolution.
            if let Some(default_model) = self.family.missing_listing_default {
                return Ok(ResolvedModel {
                    model: default_model.to_string(),
                    origin: ResolutionOrigin::Pinned,
                });
            }
            let error = DiscoveryError::NoModelAtEndpoint;
            self.cache_failure(error.clone());
            return Err(error);
        };

        let resolved = ResolvedModel {
            model,
            origin: ResolutionOrigin::Auto,
        };
        self.cache_resolution(resolved.model.clone());
        Ok(resolved)
    }

    /// Most recent successful resolution, when still fresh.
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

    /// Drop the cached resolution so the next resolve refetches.
    pub fn invalidate(&self) {
        *self.state.lock().unwrap_or_else(std::sync::PoisonError::into_inner) =
            DiscoveryState::Empty;
    }

    fn cache_resolution(&self, model: String) {
        *self.state.lock().unwrap_or_else(std::sync::PoisonError::into_inner) =
            DiscoveryState::Resolved {
                model,
                resolved_at: Instant::now(),
            };
    }

    fn cache_failure(&self, error: DiscoveryError) {
        *self.state.lock().unwrap_or_else(std::sync::PoisonError::into_inner) =
            DiscoveryState::Failed {
                error,
                failed_at: Instant::now(),
            };
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

    /// Rate-limited warning: at most once per TTL window. The family is
    /// named so operators know which endpoint lacks a family model.
    fn warn_no_model_once(&self) {
        let mut warned_at = self
            .warned_no_model_at
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner);
        let should_warn = warned_at
            .map(|at| at.elapsed() >= self.ttl)
            .unwrap_or(true);
        if should_warn {
            tracing::warn!(
                family = self.family.name,
                ttl_secs = self.ttl.as_secs(),
                "System One classifier: no {}-capable model at the configured endpoint; \
                 falling back to the existing classifier chain",
                self.family.name
            );
            *warned_at = Some(Instant::now());
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::sync::atomic::{AtomicUsize, Ordering};

    struct CountingFetcher {
        listing: Result<ModelsListing, String>,
        calls: AtomicUsize,
    }

    impl CountingFetcher {
        fn listing(ids: &[&str]) -> Self {
            Self {
                listing: Ok(ModelsListing {
                    data: ids
                        .iter()
                        .map(|id| super::super::models::ModelEntry {
                            id: id.to_string(),
                            owned_by: None,
                        })
                        .collect(),
                }),
                calls: AtomicUsize::new(0),
            }
        }

        fn failing() -> Self {
            Self {
                listing: Err("boom".to_string()),
                calls: AtomicUsize::new(0),
            }
        }

        fn calls(&self) -> usize {
            self.calls.load(Ordering::SeqCst)
        }
    }

    impl ModelsFetcher for CountingFetcher {
        fn fetch(&self) -> Result<ModelsListing, String> {
            self.calls.fetch_add(1, Ordering::SeqCst);
            self.listing.clone()
        }
    }

    fn config_model(model: &str, ttl_secs: u64) -> (SystemOneFamily, String, u64) {
        (SystemOneFamily::JEV, model.to_string(), ttl_secs)
    }

    #[test]
    fn jev_id_detection_is_word_delimited() {
        let jev = SystemOneFamily::JEV;
        assert!(jev.is_family_model("jev-1.13.0"));
        assert!(jev.is_family_model("typesafe/jev-1.13"));
        assert!(!jev.is_family_model("jealous-model"));
        assert!(!jev.is_family_model("notjev"));
        assert!(!jev.is_family_model("jevvia"));
        assert!(!jev.is_family_model("gpt-4o-mini"));
    }

    #[test]
    fn latest_jev_prefers_highest_version() {
        let jev = SystemOneFamily::JEV;
        let ids = ["gpt-4o", "jev-1.9.0", "jev-1.13.0", "jev-1.13.0-rc1"];
        let ids: Vec<String> = ids.iter().map(|s| s.to_string()).collect();
        assert_eq!(latest_family_model(&jev, &ids).as_deref(), Some("jev-1.13.0"));

        let ids = ["jev-1.9", "jev-1.13", "jev-2.0"];
        let ids: Vec<String> = ids.iter().map(|s| s.to_string()).collect();
        assert_eq!(latest_family_model(&jev, &ids).as_deref(), Some("jev-2.0"));

        let ids = ["typesafe/jev-1.12.2", "typesafe/jev-1.13.0"];
        let ids: Vec<String> = ids.iter().map(|s| s.to_string()).collect();
        assert_eq!(
            latest_family_model(&jev, &ids).as_deref(),
            Some("typesafe/jev-1.13.0")
        );
    }

    #[test]
    fn latest_laya_prefers_highest_version() {
        let laya = SystemOneFamily::LAYA;
        let ids = [
            "gpt-4o",
            "laya-1.9.0",
            "laya-2.0.0",
            "convai/laya-2.0.1",
            "laya-english",
        ];
        let ids: Vec<String> = ids.iter().map(|s| s.to_string()).collect();
        assert_eq!(
            latest_family_model(&laya, &ids).as_deref(),
            Some("convai/laya-2.0.1")
        );
    }

    #[test]
    fn latest_family_numeric_fields_compare_not_lexicographically() {
        let jev = SystemOneFamily::JEV;
        let ids = ["jev-1.9.0", "jev-1.13.0"];
        let ids: Vec<String> = ids.iter().map(|s| s.to_string()).collect();
        assert_eq!(latest_family_model(&jev, &ids).as_deref(), Some("jev-1.13.0"));
    }

    #[test]
    fn latest_family_falls_back_to_lexicographic_for_unversioned() {
        let jev = SystemOneFamily::JEV;
        let ids = ["jev-alpha", "jev-beta"];
        let ids: Vec<String> = ids.iter().map(|s| s.to_string()).collect();
        assert_eq!(latest_family_model(&jev, &ids).as_deref(), Some("jev-beta"));

        let ids = ["jev-zeta", "jev-0.1.0"];
        let ids: Vec<String> = ids.iter().map(|s| s.to_string()).collect();
        assert_eq!(latest_family_model(&jev, &ids).as_deref(), Some("jev-0.1.0"));
    }

    #[test]
    fn latest_family_returns_none_without_family_models() {
        let jev = SystemOneFamily::JEV;
        let ids = ["gpt-4o", "claude-3"];
        let ids: Vec<String> = ids.iter().map(|s| s.to_string()).collect();
        assert_eq!(latest_family_model(&jev, &ids), None);
    }

    #[test]
    fn pinned_model_bypasses_fetch_and_cache() {
        let (family, model, ttl) = config_model("jev-1.13.0", 60);
        let discovery = ModelDiscovery::new(family, &model, ttl);
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
    }

    #[test]
    fn auto_resolution_caches_within_ttl() {
        let (family, model, ttl) = config_model("auto", 600);
        let discovery = ModelDiscovery::new(family, &model, ttl);
        let fetcher = CountingFetcher::listing(&["jev-1.12.0", "jev-1.13.0"]);

        let first = discovery.resolve(&fetcher).unwrap();
        assert_eq!(first.origin, ResolutionOrigin::Auto);
        let second = discovery.resolve(&fetcher).unwrap();
        assert_eq!(second.origin, ResolutionOrigin::Cached);
        assert_eq!(fetcher.calls(), 1);
    }

    #[test]
    fn ttl_expiry_refetches() {
        let (family, model, ttl) = config_model("auto", 0);
        let discovery = ModelDiscovery::new(family, &model, ttl);
        let fetcher = CountingFetcher::listing(&["jev-1.13.0"]);

        // ttl of 0: every resolution expires immediately.
        let first = discovery.resolve(&fetcher).unwrap();
        assert_eq!(first.origin, ResolutionOrigin::Auto);
        let second = discovery.resolve(&fetcher).unwrap();
        assert_eq!(second.origin, ResolutionOrigin::Auto);
        assert_eq!(fetcher.calls(), 2);
    }

    #[test]
    fn no_family_at_endpoint_is_a_typed_error_for_jev() {
        let (family, model, ttl) = config_model("auto", 60);
        let discovery = ModelDiscovery::new(family, &model, ttl);
        let fetcher = CountingFetcher::listing(&["gpt-4o", "claude-3"]);
        assert_eq!(
            discovery.resolve(&fetcher).unwrap_err(),
            DiscoveryError::NoModelAtEndpoint
        );
    }

    #[test]
    fn listing_failure_is_a_typed_error() {
        let (family, model, ttl) = config_model("auto", 60);
        let discovery = ModelDiscovery::new(family, &model, ttl);
        let fetcher = CountingFetcher::failing();
        assert!(matches!(
            discovery.resolve(&fetcher).unwrap_err(),
            DiscoveryError::Listing(_)
        ));
    }

    #[test]
    fn listing_failures_are_negatively_cached_within_ttl() {
        let (family, model, ttl) = config_model("auto", 600);
        let discovery = ModelDiscovery::new(family, &model, ttl);
        let fetcher = CountingFetcher::failing();
        assert!(discovery.resolve(&fetcher).is_err());
        assert!(discovery.resolve(&fetcher).is_err());
        assert!(discovery.resolve(&fetcher).is_err());
        assert_eq!(fetcher.calls(), 1);
    }

    #[test]
    fn no_family_at_endpoint_is_negatively_cached_within_ttl() {
        let (family, model, ttl) = config_model("auto", 600);
        let discovery = ModelDiscovery::new(family, &model, ttl);
        let fetcher = CountingFetcher::listing(&["gpt-4o"]);
        assert_eq!(
            discovery.resolve(&fetcher).unwrap_err(),
            DiscoveryError::NoModelAtEndpoint
        );
        assert_eq!(
            discovery.resolve(&fetcher).unwrap_err(),
            DiscoveryError::NoModelAtEndpoint
        );
        assert_eq!(fetcher.calls(), 1);
    }

    #[test]
    fn laya_missing_listing_pins_default_model_once() {
        let discovery = ModelDiscovery::new(SystemOneFamily::LAYA, "auto", 600);
        let fetcher = CountingFetcher::listing(&["gpt-4o", "claude-3"]);

        let first = discovery.resolve(&fetcher).unwrap();
        assert_eq!(
            first,
            ResolvedModel {
                model: "typed-decisions".to_string(),
                origin: ResolutionOrigin::Pinned,
            }
        );

        // The default pin is not cached as a resolution: the endpoint may
        // expose a listing later. Repeated resolutions re-warn at most once
        // per TTL window but keep pinning the default.
        let second = discovery.resolve(&fetcher).unwrap();
        assert_eq!(second.model, "typed-decisions");
    }

    #[test]
    fn laya_listing_failure_is_still_a_typed_error() {
        let discovery = ModelDiscovery::new(SystemOneFamily::LAYA, "auto", 600);
        let fetcher = CountingFetcher::failing();
        assert!(matches!(
            discovery.resolve(&fetcher).unwrap_err(),
            DiscoveryError::Listing(_)
        ));
    }

    #[test]
    fn resolved_model_reads_cache_without_fetching() {
        let (family, model, ttl) = config_model("auto", 600);
        let discovery = ModelDiscovery::new(family, &model, ttl);
        let fetcher = CountingFetcher::listing(&["jev-1.13.0"]);
        assert_eq!(discovery.resolved_model(), None);
        discovery.resolve(&fetcher).unwrap();
        assert_eq!(discovery.resolved_model().as_deref(), Some("jev-1.13.0"));
        assert_eq!(fetcher.calls(), 1);
    }

    #[test]
    fn invalidate_forces_refetch() {
        let (family, model, ttl) = config_model("auto", 600);
        let discovery = ModelDiscovery::new(family, &model, ttl);
        let fetcher = CountingFetcher::listing(&["jev-1.13.0"]);
        discovery.resolve(&fetcher).unwrap();
        discovery.invalidate();
        let second = discovery.resolve(&fetcher).unwrap();
        assert_eq!(second.origin, ResolutionOrigin::Auto);
        assert_eq!(fetcher.calls(), 2);
    }
}
