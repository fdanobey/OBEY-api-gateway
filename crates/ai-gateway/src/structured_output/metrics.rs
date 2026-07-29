use std::collections::BTreeMap;
use std::fmt::Write;
use std::sync::{Mutex, MutexGuard};

const VALIDATIONS_METRIC: &str = "obey_api_structured_output_validations_total";
const RETRIES_METRIC: &str = "obey_api_structured_output_retries_total";
const LATENCY_METRIC: &str = "obey_api_structured_output_latency_ms";
const LATENCY_BUCKETS_MS: [u64; 10] = [1, 5, 10, 25, 50, 100, 250, 500, 1000, 5000];
const MAX_PROVIDER_LABEL_BYTES: usize = 64;
const MAX_MODEL_LABEL_BYTES: usize = 128;
const MAX_SERIES_PER_METRIC: usize = 4096;

type CounterKey = (String, String, &'static str);
type LatencyKey = (String, String);

#[derive(Debug, Default)]
struct LatencyHistogram {
    buckets: [u64; LATENCY_BUCKETS_MS.len()],
    count: u64,
    sum_ms: f64,
}

impl LatencyHistogram {
    fn observe(&mut self, latency_ms: f64) {
        self.count = self.count.saturating_add(1);
        self.sum_ms = (self.sum_ms + latency_ms).min(f64::MAX);

        if let Some(index) = LATENCY_BUCKETS_MS
            .iter()
            .position(|boundary| latency_ms <= *boundary as f64)
        {
            self.buckets[index] = self.buckets[index].saturating_add(1);
        }
    }
}

#[derive(Debug, Default)]
struct StructuredOutputMetricState {
    validations: BTreeMap<CounterKey, u64>,
    retries: BTreeMap<CounterKey, u64>,
    latency: BTreeMap<LatencyKey, LatencyHistogram>,
}

/// Thread-safe, best-effort metrics recorder for structured output validation.
///
/// Recording methods intentionally return no result. Invalid observations and
/// new series beyond the per-metric cardinality cap are silently discarded so
/// metrics cannot fail a request.
#[derive(Debug, Default)]
pub struct StructuredOutputMetrics {
    state: Mutex<StructuredOutputMetricState>,
}

impl StructuredOutputMetrics {
    pub fn new() -> Self {
        Self::default()
    }

    /// Record a validation outcome with status `pass`, `fail`, or `skip`.
    pub fn record_structured_output_validation(&self, provider: &str, model: &str, status: &str) {
        let Some((provider, model)) = validated_labels(provider, model) else {
            return;
        };
        let Some(status) = validation_status(status) else {
            return;
        };

        increment_counter(
            &mut self.lock_state().validations,
            (provider, model, status),
        );
    }

    /// Record a retry outcome with outcome `recovered` or `exhausted`.
    pub fn record_structured_output_retry(&self, provider: &str, model: &str, outcome: &str) {
        let Some((provider, model)) = validated_labels(provider, model) else {
            return;
        };
        let Some(outcome) = retry_outcome(outcome) else {
            return;
        };

        increment_counter(&mut self.lock_state().retries, (provider, model, outcome));
    }

    /// Observe validation and retry processing latency in milliseconds.
    pub fn observe_structured_output_latency(&self, provider: &str, model: &str, latency_ms: f64) {
        if !latency_ms.is_finite() || latency_ms < 0.0 {
            return;
        }
        let Some(key) = validated_labels(provider, model) else {
            return;
        };

        let mut state = self.lock_state();
        if let Some(histogram) = state.latency.get_mut(&key) {
            histogram.observe(latency_ms);
        } else if state.latency.len() < MAX_SERIES_PER_METRIC {
            let mut histogram = LatencyHistogram::default();
            histogram.observe(latency_ms);
            state.latency.insert(key, histogram);
        }
    }

    /// Append deterministic Prometheus text exposition with HELP and TYPE metadata.
    pub fn write_structured_output_prometheus(&self, out: &mut String) {
        let state = self.lock_state();

        out.push_str("# HELP obey_api_structured_output_validations_total Structured output validation outcomes by provider, model, and status\n");
        out.push_str("# TYPE obey_api_structured_output_validations_total counter\n");
        for ((provider, model, status), count) in &state.validations {
            let _ = writeln!(
                out,
                "{VALIDATIONS_METRIC}{{provider=\"{}\",model=\"{}\",status=\"{status}\"}} {count}",
                escape_prometheus_label(provider),
                escape_prometheus_label(model),
            );
        }

        out.push_str("# HELP obey_api_structured_output_retries_total Structured output retry outcomes by provider, model, and outcome\n");
        out.push_str("# TYPE obey_api_structured_output_retries_total counter\n");
        for ((provider, model, outcome), count) in &state.retries {
            let _ = writeln!(
                out,
                "{RETRIES_METRIC}{{provider=\"{}\",model=\"{}\",outcome=\"{outcome}\"}} {count}",
                escape_prometheus_label(provider),
                escape_prometheus_label(model),
            );
        }

        out.push_str("# HELP obey_api_structured_output_latency_ms Additional structured output validation and retry latency in milliseconds\n");
        out.push_str("# TYPE obey_api_structured_output_latency_ms histogram\n");
        for ((provider, model), histogram) in &state.latency {
            let provider = escape_prometheus_label(provider);
            let model = escape_prometheus_label(model);
            let mut cumulative = 0u64;

            for (index, boundary) in LATENCY_BUCKETS_MS.iter().enumerate() {
                cumulative = cumulative.saturating_add(histogram.buckets[index]);
                let _ = writeln!(
                    out,
                    "{LATENCY_METRIC}_bucket{{provider=\"{provider}\",model=\"{model}\",le=\"{boundary}\"}} {cumulative}"
                );
            }
            let _ = writeln!(
                out,
                "{LATENCY_METRIC}_bucket{{provider=\"{provider}\",model=\"{model}\",le=\"+Inf\"}} {}",
                histogram.count
            );
            let _ = writeln!(
                out,
                "{LATENCY_METRIC}_sum{{provider=\"{provider}\",model=\"{model}\"}} {}",
                histogram.sum_ms
            );
            let _ = writeln!(
                out,
                "{LATENCY_METRIC}_count{{provider=\"{provider}\",model=\"{model}\"}} {}",
                histogram.count
            );
        }
    }

    fn lock_state(&self) -> MutexGuard<'_, StructuredOutputMetricState> {
        self.state
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner)
    }
}

fn increment_counter(values: &mut BTreeMap<CounterKey, u64>, key: CounterKey) {
    if let Some(count) = values.get_mut(&key) {
        *count = count.saturating_add(1);
    } else if values.len() < MAX_SERIES_PER_METRIC {
        values.insert(key, 1);
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

fn validation_status(status: &str) -> Option<&'static str> {
    match status {
        "pass" => Some("pass"),
        "fail" => Some("fail"),
        "skip" => Some("skip"),
        _ => None,
    }
}

fn retry_outcome(outcome: &str) -> Option<&'static str> {
    match outcome {
        "recovered" => Some("recovered"),
        "exhausted" => Some("exhausted"),
        _ => None,
    }
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
    use proptest::prelude::*;

    fn bounded_label(max_bytes: usize) -> impl Strategy<Value = String> {
        prop::collection::vec(
            prop_oneof![
                Just('a'),
                Just('z'),
                Just('0'),
                Just('-'),
                Just('_'),
                Just('.'),
                Just('"'),
                Just('\\'),
            ],
            1..=max_bytes,
        )
        .prop_map(|characters| characters.into_iter().collect())
    }

    fn validation_sample_lines(output: &str) -> Vec<&str> {
        output
            .lines()
            .filter(|line| line.starts_with(&format!("{VALIDATIONS_METRIC}{{")))
            .collect()
    }

    fn retry_sample_lines(output: &str) -> Vec<&str> {
        output
            .lines()
            .filter(|line| line.starts_with(&format!("{RETRIES_METRIC}{{")))
            .collect()
    }

    #[test]
    fn counters_increment_exact_matching_series() {
        let metrics = StructuredOutputMetrics::new();

        for status in ["pass", "fail", "skip"] {
            metrics.record_structured_output_validation("openai", "gpt-4o", status);
        }
        metrics.record_structured_output_validation("openai", "gpt-4o", "pass");
        metrics.record_structured_output_retry("openai", "gpt-4o", "recovered");
        metrics.record_structured_output_retry("openai", "gpt-4o", "recovered");
        metrics.record_structured_output_retry("openai", "gpt-4o", "exhausted");

        let mut output = String::new();
        metrics.write_structured_output_prometheus(&mut output);

        assert!(output.contains("obey_api_structured_output_validations_total{provider=\"openai\",model=\"gpt-4o\",status=\"pass\"} 2\n"));
        assert!(output.contains("obey_api_structured_output_validations_total{provider=\"openai\",model=\"gpt-4o\",status=\"fail\"} 1\n"));
        assert!(output.contains("obey_api_structured_output_validations_total{provider=\"openai\",model=\"gpt-4o\",status=\"skip\"} 1\n"));
        assert!(output.contains("obey_api_structured_output_retries_total{provider=\"openai\",model=\"gpt-4o\",outcome=\"recovered\"} 2\n"));
        assert!(output.contains("obey_api_structured_output_retries_total{provider=\"openai\",model=\"gpt-4o\",outcome=\"exhausted\"} 1\n"));
    }

    #[test]
    fn invalid_labels_statuses_outcomes_and_latencies_are_discarded() {
        let metrics = StructuredOutputMetrics::new();
        metrics.record_structured_output_validation("openai", "gpt-4o", "pass");

        metrics.record_structured_output_validation("open\nai", "gpt-4o", "fail");
        metrics.record_structured_output_validation("", "gpt-4o", "fail");
        metrics.record_structured_output_validation(
            &"p".repeat(MAX_PROVIDER_LABEL_BYTES + 1),
            "gpt-4o",
            "fail",
        );
        metrics.record_structured_output_validation("openai", "gpt-4o", "unknown");
        metrics.record_structured_output_retry("openai", "gpt-4o", "retried");
        metrics.observe_structured_output_latency("openai", "bad\tmodel", 10.0);
        metrics.observe_structured_output_latency("openai", "gpt-4o", -1.0);
        metrics.observe_structured_output_latency("openai", "gpt-4o", f64::NAN);

        let mut output = String::new();
        metrics.write_structured_output_prometheus(&mut output);

        assert_eq!(output.matches("validations_total{").count(), 1);
        assert!(output.contains("status=\"pass\"} 1\n"));
        assert!(!output.contains("status=\"fail\"}"));
        assert!(!output.contains("retries_total{"));
        assert!(!output.contains("latency_ms_bucket{"));
    }

    #[test]
    fn histogram_buckets_are_rendered_cumulatively() {
        let metrics = StructuredOutputMetrics::new();
        for latency_ms in [0.5, 5.0, 7.0, 250.0, 5001.0] {
            metrics.observe_structured_output_latency("anthropic", "claude", latency_ms);
        }

        let mut output = String::new();
        metrics.write_structured_output_prometheus(&mut output);

        let bucket = |boundary: &str| {
            format!(
                "obey_api_structured_output_latency_ms_bucket{{provider=\"anthropic\",model=\"claude\",le=\"{boundary}\"}}"
            )
        };
        assert!(output.contains(&format!("{} 1\n", bucket("1"))));
        assert!(output.contains(&format!("{} 2\n", bucket("5"))));
        assert!(output.contains(&format!("{} 3\n", bucket("10"))));
        assert!(output.contains(&format!("{} 4\n", bucket("250"))));
        assert!(output.contains(&format!("{} 4\n", bucket("5000"))));
        assert!(output.contains(&format!("{} 5\n", bucket("+Inf"))));
        assert!(output.contains("obey_api_structured_output_latency_ms_count{provider=\"anthropic\",model=\"claude\"} 5\n"));
        assert!(output.contains("obey_api_structured_output_latency_ms_sum{provider=\"anthropic\",model=\"claude\"} 5263.5\n"));
    }

    #[test]
    fn exposition_is_escaped_and_deterministically_sorted() {
        let metrics = StructuredOutputMetrics::new();
        metrics.record_structured_output_validation("zeta", "model", "pass");
        metrics.record_structured_output_validation("acme\"cloud", "path\\model", "pass");

        let mut output = String::new();
        metrics.write_structured_output_prometheus(&mut output);

        let escaped = "provider=\"acme\\\"cloud\",model=\"path\\\\model\",status=\"pass\"";
        assert!(output.contains(escaped));
        assert!(output.find(escaped).unwrap() < output.find("provider=\"zeta\"").unwrap());
        assert!(output.contains("# HELP obey_api_structured_output_validations_total"));
        assert!(output.contains("# TYPE obey_api_structured_output_validations_total counter"));
        assert!(output.contains("# HELP obey_api_structured_output_retries_total"));
        assert!(output.contains("# TYPE obey_api_structured_output_retries_total counter"));
        assert!(output.contains("# HELP obey_api_structured_output_latency_ms"));
        assert!(output.contains("# TYPE obey_api_structured_output_latency_ms histogram"));
    }

    // Property 14: Metrics Label Correctness
    proptest! {
    #![proptest_config(ProptestConfig {
    cases: 128,
    .. ProptestConfig::default()
    })]

    #[test]
    fn prop_validation_metrics_have_exact_allowed_bounded_labels(
    provider in bounded_label(MAX_PROVIDER_LABEL_BYTES),
    model in bounded_label(MAX_MODEL_LABEL_BYTES),
    status in prop::sample::select(vec!["pass", "fail", "skip"]),
    ) {
    let metrics = StructuredOutputMetrics::new();
    metrics.record_structured_output_validation(&provider, &model, status);

    let mut output = String::new();
    metrics.write_structured_output_prometheus(&mut output);
    let samples = validation_sample_lines(&output);
    let expected = format!(
    "{VALIDATIONS_METRIC}{{provider=\"{}\",model=\"{}\",status=\"{status}\"}} 1",
    escape_prometheus_label(&provider),
    escape_prometheus_label(&model),
    );

    prop_assert!(provider.len() <= MAX_PROVIDER_LABEL_BYTES);
    prop_assert!(model.len() <= MAX_MODEL_LABEL_BYTES);
    prop_assert_eq!(samples.len(), 1);
    prop_assert_eq!(samples[0], expected);
    }

    #[test]
    fn prop_retry_metrics_have_exact_allowed_bounded_labels(
    provider in bounded_label(MAX_PROVIDER_LABEL_BYTES),
    model in bounded_label(MAX_MODEL_LABEL_BYTES),
    outcome in prop::sample::select(vec!["recovered", "exhausted"]),
    ) {
    let metrics = StructuredOutputMetrics::new();
    metrics.record_structured_output_retry(&provider, &model, outcome);

    let mut output = String::new();
    metrics.write_structured_output_prometheus(&mut output);
    let samples = retry_sample_lines(&output);
    let expected = format!(
    "{RETRIES_METRIC}{{provider=\"{}\",model=\"{}\",outcome=\"{outcome}\"}} 1",
    escape_prometheus_label(&provider),
    escape_prometheus_label(&model),
    );

    prop_assert!(provider.len() <= MAX_PROVIDER_LABEL_BYTES);
    prop_assert!(model.len() <= MAX_MODEL_LABEL_BYTES);
    prop_assert_eq!(samples.len(), 1);
    prop_assert_eq!(samples[0], expected);
    }

    #[test]
    fn prop_invalid_enum_labels_produce_no_samples(
    provider in bounded_label(MAX_PROVIDER_LABEL_BYTES),
    model in bounded_label(MAX_MODEL_LABEL_BYTES),
    invalid_status in prop::sample::select(vec!["", "PASS", "passed", "recovered", "unknown"]),
    invalid_outcome in prop::sample::select(vec!["", "RECOVERED", "retry", "pass", "unknown"]),
    ) {
    let metrics = StructuredOutputMetrics::new();
    metrics.record_structured_output_validation(&provider, &model, invalid_status);
    metrics.record_structured_output_retry(&provider, &model, invalid_outcome);

    let mut output = String::new();
    metrics.write_structured_output_prometheus(&mut output);

    prop_assert!(validation_sample_lines(&output).is_empty());
    prop_assert!(retry_sample_lines(&output).is_empty());
    }
    }
}
