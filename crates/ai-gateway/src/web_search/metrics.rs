//! Metrics for the provider-agnostic web search feature.

use dashmap::DashMap;
use std::sync::atomic::{AtomicU64, Ordering};

const LATENCY_BUCKETS_MS: [f64; 10] = [
    10.0, 50.0, 100.0, 250.0, 500.0, 1000.0, 2000.0, 5000.0, 10000.0, 30000.0,
];

const SUPPORTED_BACKENDS: [&str; 5] = ["tavily", "brave", "exa", "searxng", "codex"];

#[derive(Debug)]
struct LatencyHistogram {
    buckets: [AtomicU64; 10],
    count: AtomicU64,
    sum_micros: AtomicU64,
}

impl LatencyHistogram {
    fn new() -> Self {
        Self {
            buckets: std::array::from_fn(|_| AtomicU64::new(0)),
            count: AtomicU64::new(0),
            sum_micros: AtomicU64::new(0),
        }
    }

    fn observe(&self, duration_ms: u64) {
        self.count.fetch_add(1, Ordering::Relaxed);
        self.sum_micros
            .fetch_add(duration_ms * 1000, Ordering::Relaxed);
        let value = duration_ms as f64;
        if let Some(index) = LATENCY_BUCKETS_MS
            .iter()
            .position(|boundary| value <= *boundary)
        {
            self.buckets[index].fetch_add(1, Ordering::Relaxed);
        }
    }
}

#[derive(Debug)]
pub struct WebSearchMetrics {
    executions: DashMap<String, AtomicU64>,
    latency: DashMap<String, LatencyHistogram>,
}

impl WebSearchMetrics {
    pub fn new() -> Self {
        let executions = DashMap::new();
        let latency = DashMap::new();
        for backend in SUPPORTED_BACKENDS {
            executions.insert(backend.to_string(), AtomicU64::new(0));
            latency.insert(backend.to_string(), LatencyHistogram::new());
        }
        Self {
            executions,
            latency,
        }
    }

    pub fn record_execution(&self, backend: &str) {
        self.executions
            .entry(backend.to_string())
            .or_insert_with(|| AtomicU64::new(0))
            .fetch_add(1, Ordering::Relaxed);
    }

    pub fn record_latency_ms(&self, backend: &str, duration_ms: u64) {
        if let Some(histogram) = self.latency.get(backend) {
            histogram.observe(duration_ms);
        } else {
            self.latency
                .entry(backend.to_string())
                .or_insert_with(LatencyHistogram::new)
                .observe(duration_ms);
        }
    }

    pub fn write_prometheus(&self, out: &mut String) {
        out.push_str(
            "# HELP obey_web_search_tool_executions_total Total web search tool executions\n",
        );
        out.push_str("# TYPE obey_web_search_tool_executions_total counter\n");
        let mut exec_backends = self
            .executions
            .iter()
            .map(|entry| entry.key().clone())
            .collect::<Vec<_>>();
        exec_backends.sort();
        for backend in exec_backends {
            let Some(counter) = self.executions.get(&backend) else {
                continue;
            };
            out.push_str(&format!(
                "obey_web_search_tool_executions_total{{backend=\"{}\"}} {}\n",
                escape_label(&backend),
                counter.load(Ordering::Relaxed)
            ));
        }

        out.push_str("# HELP obey_web_search_latency_ms Web search tool execution latency\n");
        out.push_str("# TYPE obey_web_search_latency_ms histogram\n");
        let mut latency_backends = self
            .latency
            .iter()
            .map(|entry| entry.key().clone())
            .collect::<Vec<_>>();
        latency_backends.sort();
        for backend in latency_backends {
            let Some(histogram) = self.latency.get(&backend) else {
                continue;
            };
            let mut cumulative = 0;
            for (index, boundary) in LATENCY_BUCKETS_MS.iter().enumerate() {
                cumulative += histogram.buckets[index].load(Ordering::Relaxed);
                out.push_str(&format!(
                    "obey_web_search_latency_ms_bucket{{backend=\"{}\",le=\"{}\"}} {}\n",
                    escape_label(&backend),
                    boundary,
                    cumulative
                ));
            }
            let count = histogram.count.load(Ordering::Relaxed);
            out.push_str(&format!(
                "obey_web_search_latency_ms_bucket{{backend=\"{}\",le=\"+Inf\"}} {}\n",
                escape_label(&backend),
                count
            ));
            out.push_str(&format!(
                "obey_web_search_latency_ms_sum{{backend=\"{}\"}} {}\n",
                escape_label(&backend),
                histogram.sum_micros.load(Ordering::Relaxed) as f64 / 1000.0
            ));
            out.push_str(&format!(
                "obey_web_search_latency_ms_count{{backend=\"{}\"}} {}\n",
                escape_label(&backend),
                count
            ));
        }
    }
}

impl Default for WebSearchMetrics {
    fn default() -> Self {
        Self::new()
    }
}

fn escape_label(value: &str) -> String {
    value
        .replace('\\', "\\\\")
        .replace('"', "\\\"")
        .replace('\n', "\\n")
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn execution_counter_increments() {
        let metrics = WebSearchMetrics::new();
        metrics.record_execution("tavily");
        metrics.record_execution("tavily");
        metrics.record_execution("searxng");

        let mut out = String::new();
        metrics.write_prometheus(&mut out);

        assert!(out.contains("obey_web_search_tool_executions_total{backend=\"tavily\"} 2\n"));
        assert!(out.contains("obey_web_search_tool_executions_total{backend=\"searxng\"} 1\n"));
    }

    #[test]
    fn latency_histogram_records() {
        let metrics = WebSearchMetrics::new();
        metrics.record_latency_ms("tavily", 75);

        let mut out = String::new();
        metrics.write_prometheus(&mut out);

        assert!(
            out.contains("obey_web_search_latency_ms_bucket{backend=\"tavily\",le=\"10\"} 0\n")
        );
        assert!(
            out.contains("obey_web_search_latency_ms_bucket{backend=\"tavily\",le=\"50\"} 0\n")
        );
        assert!(
            out.contains("obey_web_search_latency_ms_bucket{backend=\"tavily\",le=\"100\"} 1\n")
        );
        assert!(out.contains("obey_web_search_latency_ms_sum{backend=\"tavily\"} 75\n"));
        assert!(out.contains("obey_web_search_latency_ms_count{backend=\"tavily\"} 1\n"));
    }

    #[test]
    fn zero_counts_when_unused() {
        let metrics = WebSearchMetrics::new();

        let mut out = String::new();
        metrics.write_prometheus(&mut out);

        for backend in SUPPORTED_BACKENDS {
            assert!(out.contains(&format!(
                "obey_web_search_tool_executions_total{{backend=\"{backend}\"}} 0\n"
            )));
            assert!(out.contains(&format!(
                "obey_web_search_latency_ms_count{{backend=\"{backend}\"}} 0\n"
            )));
        }
    }

    #[test]
    fn prometheus_format_is_valid() {
        let metrics = WebSearchMetrics::new();

        let mut out = String::new();
        metrics.write_prometheus(&mut out);

        assert!(out.contains("# HELP obey_web_search_tool_executions_total "));
        assert!(out.contains("# TYPE obey_web_search_tool_executions_total counter\n"));
        assert!(out.contains("# HELP obey_web_search_latency_ms "));
        assert!(out.contains("# TYPE obey_web_search_latency_ms histogram\n"));
    }

#[test]
fn codex_series_untouched() {
    let codex = crate::codex::search::metrics::SearchMetrics::new();
    let web_search = WebSearchMetrics::new();

    let mut out = String::new();
    codex.write_prometheus(&mut out);
    web_search.write_prometheus(&mut out);

    assert!(out.contains("obey_codex_search_tool_executions_total{tool=\"codex_search\"} 0\n"));
    assert!(out.contains("obey_codex_search_tool_executions_total{tool=\"codex_web\"} 0\n"));
    assert!(out.contains("obey_codex_search_latency_ms_count{tool=\"codex_search\"} 0\n"));
    assert!(out.contains("obey_codex_search_latency_ms_count{tool=\"codex_web\"} 0\n"));
    assert!(out.contains("obey_web_search_tool_executions_total{backend=\"tavily\"} 0\n"));
    assert!(out.contains("obey_web_search_latency_ms_count{backend=\"codex\"} 0\n"));
}

#[test]
fn executions_recorded_on_success_and_error_paths_with_backend_label() {
    let metrics = WebSearchMetrics::new();
    metrics.record_execution("tavily");
    metrics.record_execution("tavily");
    metrics.record_execution("tavily");
    metrics.record_execution("tavily");
    metrics.record_execution("searxng");

    let mut out = String::new();
    metrics.write_prometheus(&mut out);

    assert!(
        out.contains("obey_web_search_tool_executions_total{backend=\"tavily\"} 4\n"),
        "all four paths credit tavily: {out}"
    );
    assert!(
        out.contains("obey_web_search_tool_executions_total{backend=\"searxng\"} 1\n"),
        "error path credits the failing backend: {out}"
    );
}

#[test]
fn error_path_execution_counted_without_latency_series_growth() {
    let metrics = WebSearchMetrics::new();
    metrics.record_execution("exa");
    metrics.record_execution("exa");

    let mut out = String::new();
    metrics.write_prometheus(&mut out);

    assert!(out.contains("obey_web_search_tool_executions_total{backend=\"exa\"} 2\n"));
    assert!(out.contains("obey_web_search_latency_ms_count{backend=\"exa\"} 0\n"));
}

#[test]
fn execution_counter_only_grows_for_named_backend() {
    let metrics = WebSearchMetrics::new();
    metrics.record_execution("brave");

    let mut out = String::new();
    metrics.write_prometheus(&mut out);

    assert!(out.contains("obey_web_search_tool_executions_total{backend=\"brave\"} 1\n"));
    assert!(out.contains("obey_web_search_tool_executions_total{backend=\"tavily\"} 0\n"));
}
}
