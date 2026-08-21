//! Metrics for the Codex Search feature.

use dashmap::DashMap;
use std::sync::atomic::{AtomicU64, Ordering};

const LATENCY_BUCKETS_MS: [f64; 10] = [
    10.0, 50.0, 100.0, 250.0, 500.0, 1000.0, 2000.0, 5000.0, 10000.0, 30000.0,
];

const SUPPORTED_TOOLS: [&str; 2] = ["codex_search", "codex_web"];

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
pub struct SearchMetrics {
    executions: DashMap<String, AtomicU64>,
    latency: DashMap<String, LatencyHistogram>,
}

impl SearchMetrics {
    pub fn new() -> Self {
        let executions = DashMap::new();
        let latency = DashMap::new();
        for tool in SUPPORTED_TOOLS {
            executions.insert(tool.to_string(), AtomicU64::new(0));
            latency.insert(tool.to_string(), LatencyHistogram::new());
        }
        Self {
            executions,
            latency,
        }
    }

    pub fn record_execution(&self, tool: &str) {
        self.executions
            .entry(tool.to_string())
            .or_insert_with(|| AtomicU64::new(0))
            .fetch_add(1, Ordering::Relaxed);
    }

    pub fn record_latency(&self, tool: &str, duration_ms: u64) {
        if let Some(histogram) = self.latency.get(tool) {
            histogram.observe(duration_ms);
        } else {
            self.latency
                .entry(tool.to_string())
                .or_insert_with(LatencyHistogram::new)
                .observe(duration_ms);
        }
}

pub fn write_prometheus(&self, out: &mut String) {
        out.push_str(
            "# HELP obey_codex_search_tool_executions_total Total codex search tool executions\n",
        );
        out.push_str("# TYPE obey_codex_search_tool_executions_total counter\n");
        let mut exec_tools = self
            .executions
            .iter()
            .map(|entry| entry.key().clone())
            .collect::<Vec<_>>();
        exec_tools.sort();
        for tool in exec_tools {
            let Some(counter) = self.executions.get(&tool) else {
                continue;
            };
            out.push_str(&format!(
                "obey_codex_search_tool_executions_total{{tool=\"{}\"}} {}\n",
                escape_label(&tool),
                counter.load(Ordering::Relaxed)
            ));
        }

        out.push_str("# HELP obey_codex_search_latency_ms Codex search tool execution latency\n");
        out.push_str("# TYPE obey_codex_search_latency_ms histogram\n");
        let mut latency_tools = self
            .latency
            .iter()
            .map(|entry| entry.key().clone())
            .collect::<Vec<_>>();
        latency_tools.sort();
        for tool in latency_tools {
            let Some(histogram) = self.latency.get(&tool) else {
                continue;
            };
            let mut cumulative = 0;
            for (index, boundary) in LATENCY_BUCKETS_MS.iter().enumerate() {
                cumulative += histogram.buckets[index].load(Ordering::Relaxed);
                out.push_str(&format!(
                    "obey_codex_search_latency_ms_bucket{{tool=\"{}\",le=\"{}\"}} {}\n",
                    escape_label(&tool),
                    boundary,
                    cumulative
                ));
            }
            let count = histogram.count.load(Ordering::Relaxed);
            out.push_str(&format!(
                "obey_codex_search_latency_ms_bucket{{tool=\"{}\",le=\"+Inf\"}} {}\n",
                escape_label(&tool),
                count
            ));
            out.push_str(&format!(
                "obey_codex_search_latency_ms_sum{{tool=\"{}\"}} {}\n",
                escape_label(&tool),
                histogram.sum_micros.load(Ordering::Relaxed) as f64 / 1000.0
            ));
            out.push_str(&format!(
                "obey_codex_search_latency_ms_count{{tool=\"{}\"}} {}\n",
                escape_label(&tool),
                count
            ));
        }
    }
}

impl Default for SearchMetrics {
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
        let metrics = SearchMetrics::new();
        metrics.record_execution("codex_search");
        metrics.record_execution("codex_search");
        metrics.record_execution("codex_web");

        let mut out = String::new();
        metrics.write_prometheus(&mut out);

        assert!(out.contains("obey_codex_search_tool_executions_total{tool=\"codex_search\"} 2\n"));
        assert!(out.contains("obey_codex_search_tool_executions_total{tool=\"codex_web\"} 1\n"));
    }

    #[test]
    fn latency_histogram_records() {
        let metrics = SearchMetrics::new();
        metrics.record_latency("codex_search", 75);

        let mut out = String::new();
        metrics.write_prometheus(&mut out);

        assert!(out
            .contains("obey_codex_search_latency_ms_bucket{tool=\"codex_search\",le=\"10\"} 0\n"));
        assert!(out
            .contains("obey_codex_search_latency_ms_bucket{tool=\"codex_search\",le=\"50\"} 0\n"));
        assert!(out
            .contains("obey_codex_search_latency_ms_bucket{tool=\"codex_search\",le=\"100\"} 1\n"));
        assert!(out.contains("obey_codex_search_latency_ms_sum{tool=\"codex_search\"} 75\n"));
        assert!(out.contains("obey_codex_search_latency_ms_count{tool=\"codex_search\"} 1\n"));
    }

    #[test]
    fn latency_not_recorded_for_unsupported_tools_implicitly() {
        let metrics = SearchMetrics::new();

        let mut out = String::new();
        metrics.write_prometheus(&mut out);

        assert!(out.contains("obey_codex_search_tool_executions_total{tool=\"codex_search\"} 0\n"));
        assert!(out.contains("obey_codex_search_tool_executions_total{tool=\"codex_web\"} 0\n"));
        assert!(out.contains("obey_codex_search_latency_ms_count{tool=\"codex_search\"} 0\n"));
        assert!(out.contains("obey_codex_search_latency_ms_count{tool=\"codex_web\"} 0\n"));
    }

    #[test]
    fn prometheus_format_is_valid() {
        let metrics = SearchMetrics::new();

        let mut out = String::new();
        metrics.write_prometheus(&mut out);

        assert!(out.contains("# HELP obey_codex_search_tool_executions_total "));
        assert!(out.contains("# TYPE obey_codex_search_tool_executions_total counter\n"));
        assert!(out.contains("# HELP obey_codex_search_latency_ms "));
        assert!(out.contains("# TYPE obey_codex_search_latency_ms histogram\n"));
    }
}
