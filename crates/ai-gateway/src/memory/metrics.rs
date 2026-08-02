//! Best-effort Prometheus counters for persistent-memory operations.

use std::fmt::Write;
use std::sync::atomic::{AtomicU64, Ordering};

const RETRIEVALS_METRIC: &str = "obey_memory_retrievals_total";
const STORES_METRIC: &str = "obey_memory_stores_total";
const INJECTION_TOKENS_METRIC: &str = "obey_memory_injection_tokens_total";
const PROJECT_DETECTIONS_METRIC: &str = "obey_memory_project_detections_total";
const DECAY_EVICTIONS_METRIC: &str = "obey_memory_decay_evictions_total";

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum NamespaceType {
    Project,
    Agent,
    User,
}

impl NamespaceType {
    pub(crate) fn from_namespace(namespace: &str) -> Self {
        if namespace.contains("::project::") {
            Self::Project
        } else if namespace.contains("::agent::") {
            Self::Agent
        } else {
            Self::User
        }
    }

    fn label(self) -> &'static str {
        match self {
            Self::Project => "project",
            Self::Agent => "agent",
            Self::User => "user",
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum StoreMethod {
    Explicit,
    AsyncLlm,
    Heuristic,
}

impl StoreMethod {
    fn label(self) -> &'static str {
        match self {
            Self::Explicit => "explicit",
            Self::AsyncLlm => "async_llm",
            Self::Heuristic => "heuristic",
        }
    }
}

/// Fixed-cardinality, infallible counters owned by one memory system.
#[derive(Debug, Default)]
pub struct MemoryMetrics {
    retrievals: [AtomicU64; 3],
    stores: [AtomicU64; 3],
    injection_tokens: AtomicU64,
    project_detections: [AtomicU64; 3],
    decay_evictions: AtomicU64,
}

impl MemoryMetrics {
    pub fn new() -> Self {
        Self::default()
    }

    pub(crate) fn record_retrievals(&self, namespace_type: NamespaceType, count: u64) {
        saturating_add(&self.retrievals[namespace_type.index()], count);
    }

    pub(crate) fn record_store(&self, method: StoreMethod) {
        saturating_add(&self.stores[method.index()], 1);
    }

    pub(crate) fn record_injection_tokens(&self, count: u64) {
        saturating_add(&self.injection_tokens, count);
    }

    pub(crate) fn record_project_detection(&self, context_type: NamespaceType) {
        saturating_add(&self.project_detections[context_type.index()], 1);
    }

    pub(crate) fn record_decay_evictions(&self, count: u64) {
        saturating_add(&self.decay_evictions, count);
    }

    /// Gather this memory system's complete Prometheus text exposition.
    pub fn gather(&self) -> String {
        let mut output = String::new();
        self.write_prometheus(&mut output);
        output
    }

    /// Append this memory system's complete Prometheus text exposition.
    pub fn write_prometheus(&self, output: &mut String) {
        output.push_str("# HELP obey_memory_retrievals_total Total memory entries retrieved\n");
        output.push_str("# TYPE obey_memory_retrievals_total counter\n");
        for namespace_type in NamespaceType::ALL {
            write_labeled_counter(
                output,
                RETRIEVALS_METRIC,
                "namespace_type",
                namespace_type.label(),
                load(&self.retrievals[namespace_type.index()]),
            );
        }

        output.push_str("# HELP obey_memory_stores_total Total memory entries stored\n");
        output.push_str("# TYPE obey_memory_stores_total counter\n");
        for method in StoreMethod::ALL {
            write_labeled_counter(
                output,
                STORES_METRIC,
                "extraction_method",
                method.label(),
                load(&self.stores[method.index()]),
            );
        }

        output.push_str(
            "# HELP obey_memory_injection_tokens_total Total tokens injected from memory\n",
        );
        output.push_str("# TYPE obey_memory_injection_tokens_total counter\n");
        write_counter(
            output,
            INJECTION_TOKENS_METRIC,
            load(&self.injection_tokens),
        );

        output.push_str(
            "# HELP obey_memory_project_detections_total Context detection completions\n",
        );
        output.push_str("# TYPE obey_memory_project_detections_total counter\n");
        for context_type in NamespaceType::ALL {
            write_labeled_counter(
                output,
                PROJECT_DETECTIONS_METRIC,
                "context_type",
                context_type.label(),
                load(&self.project_detections[context_type.index()]),
            );
        }

        output.push_str(
            "# HELP obey_memory_decay_evictions_total Memories evicted during decay cycles\n",
        );
        output.push_str("# TYPE obey_memory_decay_evictions_total counter\n");
        write_counter(output, DECAY_EVICTIONS_METRIC, load(&self.decay_evictions));
    }
}

impl NamespaceType {
    const ALL: [Self; 3] = [Self::Project, Self::Agent, Self::User];

    fn index(self) -> usize {
        match self {
            Self::Project => 0,
            Self::Agent => 1,
            Self::User => 2,
        }
    }
}

impl StoreMethod {
    const ALL: [Self; 3] = [Self::Explicit, Self::AsyncLlm, Self::Heuristic];

    fn index(self) -> usize {
        match self {
            Self::Explicit => 0,
            Self::AsyncLlm => 1,
            Self::Heuristic => 2,
        }
    }
}

fn saturating_add(counter: &AtomicU64, value: u64) {
    let _ = counter.fetch_update(Ordering::Relaxed, Ordering::Relaxed, |current| {
        Some(current.saturating_add(value))
    });
}

fn load(counter: &AtomicU64) -> u64 {
    counter.load(Ordering::Relaxed)
}

fn write_labeled_counter(
    output: &mut String,
    metric: &str,
    label_name: &str,
    label_value: &str,
    value: u64,
) {
    let _ = writeln!(output, "{metric}{{{label_name}=\"{label_value}\"}} {value}");
}

fn write_counter(output: &mut String, metric: &str, value: u64) {
    let _ = writeln!(output, "{metric} {value}");
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn prometheus_names_and_exact_bounded_labels_are_stable() {
        let metrics = MemoryMetrics::new();
        metrics.record_retrievals(NamespaceType::Project, 2);
        metrics.record_retrievals(NamespaceType::Agent, 3);
        metrics.record_retrievals(NamespaceType::User, 4);
        metrics.record_store(StoreMethod::Explicit);
        metrics.record_store(StoreMethod::AsyncLlm);
        metrics.record_store(StoreMethod::Heuristic);
        metrics.record_injection_tokens(17);
        metrics.record_project_detection(NamespaceType::Project);
        metrics.record_project_detection(NamespaceType::Agent);
        metrics.record_project_detection(NamespaceType::User);
        metrics.record_decay_evictions(5);

        let output = metrics.gather();

        assert!(output.contains("obey_memory_retrievals_total{namespace_type=\"project\"} 2\n"));
        assert!(output.contains("obey_memory_retrievals_total{namespace_type=\"agent\"} 3\n"));
        assert!(output.contains("obey_memory_retrievals_total{namespace_type=\"user\"} 4\n"));
        assert!(output.contains("obey_memory_stores_total{extraction_method=\"explicit\"} 1\n"));
        assert!(output.contains("obey_memory_stores_total{extraction_method=\"async_llm\"} 1\n"));
        assert!(output.contains("obey_memory_stores_total{extraction_method=\"heuristic\"} 1\n"));
        assert!(output.contains("obey_memory_injection_tokens_total 17\n"));
        assert!(
            output.contains("obey_memory_project_detections_total{context_type=\"project\"} 1\n")
        );
        assert!(output.contains("obey_memory_project_detections_total{context_type=\"agent\"} 1\n"));
        assert!(output.contains("obey_memory_project_detections_total{context_type=\"user\"} 1\n"));
        assert!(output.contains("obey_memory_decay_evictions_total 5\n"));
        assert_eq!(output.matches("namespace_type=").count(), 3);
        assert_eq!(output.matches("extraction_method=").count(), 3);
        assert_eq!(output.matches("context_type=").count(), 3);
    }

    #[test]
    fn exposition_never_contains_raw_namespaces() {
        let raw_namespace = "user::private-key::project::0123456789abcdef";
        let metrics = MemoryMetrics::new();
        metrics.record_retrievals(NamespaceType::from_namespace(raw_namespace), 1);

        let output = metrics.gather();

        assert!(!output.contains(raw_namespace));
        assert!(!output.contains("private-key"));
        assert!(output.contains("namespace_type=\"project\""));
    }

    #[test]
    fn counters_saturate_without_returning_failures() {
        let metrics = MemoryMetrics::new();
        metrics.record_injection_tokens(u64::MAX);
        metrics.record_injection_tokens(1);

        let output = metrics.gather();

        assert!(output.contains(&format!(
            "obey_memory_injection_tokens_total {}\n",
            u64::MAX
        )));
    }
}
