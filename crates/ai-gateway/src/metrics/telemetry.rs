//! OTLP trace-export health counters (opentelemetry-trace-export spec,
//! Req 7.1/7.2).
//!
//! The telemetry exporter's background task increments these cumulative
//! counters through a shared [`TelemetryMetrics`] handle (obtained via
//! `Metrics::telemetry_metrics()`), so the exporter internals and the
//! `/metrics` Prometheus exposition always observe the same series:
//!
//! - `otlp_spans_exported_total` — spans delivered with an HTTP 2xx
//! - `otlp_spans_dropped_total` — spans dropped (queue full, retry
//!   exhaustion, shutdown overflow)
//! - `otlp_export_failures_total` — batches that exhausted retries
//!
//! Counter updates are lock-free atomic adds; exposition is deterministic
//! and always renders all three series (even at zero) so dashboards can
//! alert on absence of traffic.

use std::fmt::Write;
use std::sync::atomic::{AtomicU64, Ordering};

const SPANS_EXPORTED_METRIC: &str = "otlp_spans_exported_total";
const SPANS_DROPPED_METRIC: &str = "otlp_spans_dropped_total";
const EXPORT_FAILURES_METRIC: &str = "otlp_export_failures_total";

/// Cumulative OTLP export counters shared between the exporter's delivery
/// task and Prometheus rendering.
#[derive(Debug, Default)]
pub struct TelemetryMetrics {
    spans_exported: AtomicU64,
    spans_dropped: AtomicU64,
    export_failures: AtomicU64,
}

impl TelemetryMetrics {
    pub fn new() -> Self {
        Self::default()
    }

    /// Count spans whose batch was delivered (HTTP 2xx).
    pub fn record_exported(&self, span_count: u64) {
        self.spans_exported
            .fetch_add(span_count, Ordering::Relaxed);
    }

    /// Count spans dropped whole-batch (queue full, retry exhaustion,
    /// shutdown overflow). Never a partial count: batches are accepted or
    /// dropped whole.
    pub fn record_dropped(&self, span_count: u64) {
        self.spans_dropped.fetch_add(span_count, Ordering::Relaxed);
    }

    /// Count one retry-exhausted (or otherwise failed) export batch.
    pub fn record_failure(&self) {
        self.export_failures.fetch_add(1, Ordering::Relaxed);
    }

    pub fn exported(&self) -> u64 {
        self.spans_exported.load(Ordering::Relaxed)
    }

    pub fn dropped(&self) -> u64 {
        self.spans_dropped.load(Ordering::Relaxed)
    }

    pub fn failures(&self) -> u64 {
        self.export_failures.load(Ordering::Relaxed)
    }

    /// Append the three counter series with HELP/TYPE metadata in
    /// Prometheus text exposition format.
    pub fn write_prometheus(&self, out: &mut String) {
        out.push_str("# HELP otlp_spans_exported_total OTLP spans successfully exported to the trace endpoint\n");
        out.push_str("# TYPE otlp_spans_exported_total counter\n");
        let _ = writeln!(
            out,
            "{SPANS_EXPORTED_METRIC} {}",
            self.spans_exported.load(Ordering::Relaxed)
        );

        out.push_str("# HELP otlp_spans_dropped_total OTLP spans dropped whole-batch (queue full, retry exhaustion, shutdown overflow)\n");
        out.push_str("# TYPE otlp_spans_dropped_total counter\n");
        let _ = writeln!(
            out,
            "{SPANS_DROPPED_METRIC} {}",
            self.spans_dropped.load(Ordering::Relaxed)
        );

        out.push_str("# HELP otlp_export_failures_total OTLP export batches that exhausted retries and were dropped\n");
        out.push_str("# TYPE otlp_export_failures_total counter\n");
        let _ = writeln!(
            out,
            "{EXPORT_FAILURES_METRIC} {}",
            self.export_failures.load(Ordering::Relaxed)
        );
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn counters_increment_and_render() {
        let metrics = TelemetryMetrics::new();
        metrics.record_exported(7);
        metrics.record_exported(2);
        metrics.record_dropped(3);
        metrics.record_failure();
        metrics.record_failure();
        metrics.record_failure();

        let mut out = String::new();
        metrics.write_prometheus(&mut out);

        assert!(out.contains("otlp_spans_exported_total 9\n"));
        assert!(out.contains("otlp_spans_dropped_total 3\n"));
        assert!(out.contains("otlp_export_failures_total 3\n"));
    }

    #[test]
    fn exposition_has_help_and_type_lines() {
        let metrics = TelemetryMetrics::new();

        let mut out = String::new();
        metrics.write_prometheus(&mut out);

        assert!(out.contains("# HELP otlp_spans_exported_total"));
        assert!(out.contains("# TYPE otlp_spans_exported_total counter"));
        assert!(out.contains("# HELP otlp_spans_dropped_total"));
        assert!(out.contains("# TYPE otlp_spans_dropped_total counter"));
        assert!(out.contains("# HELP otlp_export_failures_total"));
        assert!(out.contains("# TYPE otlp_export_failures_total counter"));
    }

    #[test]
    fn counters_render_zero_before_activity() {
        let metrics = TelemetryMetrics::new();

        let mut out = String::new();
        metrics.write_prometheus(&mut out);

        assert!(out.contains("otlp_spans_exported_total 0\n"));
        assert!(out.contains("otlp_spans_dropped_total 0\n"));
        assert!(out.contains("otlp_export_failures_total 0\n"));
    }

    #[test]
    fn getter_snapshot_matches_recorded_values() {
        let metrics = TelemetryMetrics::new();
        metrics.record_exported(4);
        metrics.record_dropped(1);
        metrics.record_failure();

        assert_eq!(metrics.exported(), 4);
        assert_eq!(metrics.dropped(), 1);
        assert_eq!(metrics.failures(), 1);
    }
}
