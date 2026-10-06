//! Model-labeled provider attempt-failure metrics.
//!
//! The gateway's existing provider counters (`record_provider_failure_*`) are
//! keyed by provider only, so a model like `glm-5.3:dev` returning 504s is
//! invisible when the request ultimately succeeds via retry/fallback. This
//! recorder captures every FAILED provider attempt keyed by
//! `(provider, model, status_code)` and exposes it in Prometheus with a
//! `model` label, mirroring the per-model structured-output metrics pattern.
//!
//! Recording is best-effort: invalid labels and new series beyond the
//! cardinality cap are silently discarded so metrics can never fail or slow a
//! request.

use std::collections::BTreeMap;
use std::fmt::Write;
use std::sync::{Mutex, MutexGuard};

const ATTEMPT_FAILURES_METRIC: &str = "obey_api_provider_attempt_failures_total";
const MAX_PROVIDER_LABEL_BYTES: usize = 64;
const MAX_MODEL_LABEL_BYTES: usize = 128;
const MAX_SERIES_PER_METRIC: usize = 4096;

/// Bounded label set: (provider, model, status_code).
type CounterKey = (String, String, u16);

#[derive(Debug, Default)]
struct ProviderAttemptFailureState {
    failures: BTreeMap<CounterKey, u64>,
}

/// Thread-safe, best-effort recorder for failed provider attempts, keyed by
/// `(provider, model, status_code)`.
#[derive(Debug, Default)]
pub struct ProviderAttemptFailureMetrics {
    state: Mutex<ProviderAttemptFailureState>,
}

impl ProviderAttemptFailureMetrics {
    pub fn new() -> Self {
        Self::default()
    }

    /// Record a single failed provider attempt. Invalid labels are discarded.
    pub fn record(&self, provider: &str, model: &str, status_code: u16) {
        let Some((provider, model)) = validated_labels(provider, model) else {
            return;
        };

        let mut state = self.lock_state();
        let key = (provider, model, status_code);
        if let Some(count) = state.failures.get_mut(&key) {
            *count = count.saturating_add(1);
        } else if state.failures.len() < MAX_SERIES_PER_METRIC {
            state.failures.insert(key, 1);
        }
    }

    /// Append deterministic Prometheus text exposition with HELP and TYPE
    /// metadata.
    pub fn write_prometheus(&self, out: &mut String) {
        let state = self.lock_state();

        out.push_str("# HELP obey_api_provider_attempt_failures_total Failed upstream provider attempts by provider, model, and status_code (includes attempts masked by a successful retry/fallback)\n");
        out.push_str("# TYPE obey_api_provider_attempt_failures_total counter\n");
        for ((provider, model, status_code), count) in &state.failures {
            let _ = writeln!(
                out,
                "{ATTEMPT_FAILURES_METRIC}{{provider=\"{}\",model=\"{}\",status_code=\"{status_code}\"}} {count}",
                escape_prometheus_label(provider),
                escape_prometheus_label(model),
            );
        }
    }

    fn lock_state(&self) -> MutexGuard<'_, ProviderAttemptFailureState> {
        self.state
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner)
    }
}

fn validated_labels(provider: &str, model: &str) -> Option<(String, String)> {
    if !valid_label(provider, MAX_PROVIDER_LABEL_BYTES)
        || !valid_label(model, MAX_MODEL_LABEL_BYTES)
    {
        return None;
    }

    Some((provider.to_owned(), model.to_owned()))
}

fn valid_label(value: &str, max_bytes: usize) -> bool {
    !value.is_empty() && value.len() <= max_bytes && !value.chars().any(char::is_control)
}

fn escape_prometheus_label(value: &str) -> String {
    let mut escaped = String::with_capacity(value.len());
    for character in value.chars() {
        match character {
            '\\' => escaped.push_str("\\\\"),
            '"' => escaped.push_str("\\\""),
            _ => escaped.push(character),
        }
    }
    escaped
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn counters_increment_exact_matching_series() {
        let metrics = ProviderAttemptFailureMetrics::new();
        metrics.record("electronhub", "glm-5.3:dev", 504);
        metrics.record("electronhub", "glm-5.3:dev", 504);
        metrics.record("electronhub", "glm-5.3:dev", 500);

        let mut out = String::new();
        metrics.write_prometheus(&mut out);

        assert!(out.contains(
            "obey_api_provider_attempt_failures_total{provider=\"electronhub\",model=\"glm-5.3:dev\",status_code=\"504\"} 2\n"
        ));
        assert!(out.contains(
            "obey_api_provider_attempt_failures_total{provider=\"electronhub\",model=\"glm-5.3:dev\",status_code=\"500\"} 1\n"
        ));
    }

    #[test]
    fn exposition_has_help_and_type_lines() {
        let metrics = ProviderAttemptFailureMetrics::new();
        metrics.record("openai", "gpt-4o", 503);

        let mut out = String::new();
        metrics.write_prometheus(&mut out);

        assert!(out.contains("# HELP obey_api_provider_attempt_failures_total"));
        assert!(out.contains("# TYPE obey_api_provider_attempt_failures_total counter"));
    }

    #[test]
    fn invalid_labels_are_discarded() {
        let metrics = ProviderAttemptFailureMetrics::new();
        metrics.record("", "gpt-4o", 504);
        metrics.record("open\nai", "gpt-4o", 504);
        metrics.record("openai", &"m".repeat(MAX_MODEL_LABEL_BYTES + 1), 504);
        metrics.record("openai", "gpt-4o", 500);

        let mut out = String::new();
        metrics.write_prometheus(&mut out);

        assert_eq!(out.matches(&format!("{ATTEMPT_FAILURES_METRIC}{{")).count(), 1);
        assert!(out.contains("status_code=\"500\"} 1\n"));
    }

    #[test]
    fn exposition_is_escaped_and_sorted() {
        let metrics = ProviderAttemptFailureMetrics::new();
        metrics.record("zeta", "model", 502);
        metrics.record("acme\"cloud", "path\\model", 502);

        let mut out = String::new();
        metrics.write_prometheus(&mut out);

        let escaped = "provider=\"acme\\\"cloud\",model=\"path\\\\model\",status_code=\"502\"";
        assert!(out.contains(escaped));
        assert!(out.find(escaped).unwrap() < out.find("provider=\"zeta\"").unwrap());
    }
}
