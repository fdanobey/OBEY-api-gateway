use dashmap::DashMap;
use std::{
    collections::VecDeque,
    sync::{
        atomic::{AtomicU64, Ordering},
        Mutex,
    },
    time::Instant,
};

const CONFIDENCE_BUCKETS: [f32; 10] = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0];

#[derive(Debug)]
struct ConfidenceHistogram {
    buckets: [AtomicU64; 10],
    count: AtomicU64,
    sum_micros: AtomicU64,
}

impl ConfidenceHistogram {
    fn new() -> Self {
        Self {
            buckets: std::array::from_fn(|_| AtomicU64::new(0)),
            count: AtomicU64::new(0),
            sum_micros: AtomicU64::new(0),
        }
    }

    fn observe(&self, value: f32) {
        let value = if value.is_finite() {
            value.clamp(0.0, 1.0)
        } else {
            0.0
        };
        self.count.fetch_add(1, Ordering::Relaxed);
        self.sum_micros.fetch_add(
            (value as f64 * 1_000_000.0).round() as u64,
            Ordering::Relaxed,
        );
        if let Some(index) = CONFIDENCE_BUCKETS
            .iter()
            .position(|boundary| value <= *boundary)
        {
            self.buckets[index].fetch_add(1, Ordering::Relaxed);
        }
    }
}

#[derive(Debug)]
pub struct LoopDetectionMetrics {
    confidence: DashMap<String, ConfidenceHistogram>,
    enforcement: DashMap<(String, String), AtomicU64>,
    evicted_total: AtomicU64,
    eviction_times: Mutex<VecDeque<Instant>>,
}

impl LoopDetectionMetrics {
    pub fn new() -> Self {
        Self {
            confidence: DashMap::new(),
            enforcement: DashMap::new(),
            evicted_total: AtomicU64::new(0),
            eviction_times: Mutex::new(VecDeque::new()),
        }
    }

    pub fn record_confidence(&self, virtual_key: Option<&str>, confidence: f32) {
        self.confidence
            .entry(private_virtual_key_label(virtual_key))
            .or_insert_with(ConfidenceHistogram::new)
            .observe(confidence);
    }

    pub fn record_enforcement(&self, level: &str, virtual_key: Option<&str>) {
        self.enforcement
            .entry((level.to_string(), private_virtual_key_label(virtual_key)))
            .or_insert_with(|| AtomicU64::new(0))
            .fetch_add(1, Ordering::Relaxed);
    }

    pub fn record_eviction(&self) {
        self.evicted_total.fetch_add(1, Ordering::Relaxed);
        let now = Instant::now();
        let mut times = self
            .eviction_times
            .lock()
            .expect("eviction metrics mutex poisoned");
        times.push_back(now);
        while times
            .front()
            .is_some_and(|timestamp| now.duration_since(*timestamp).as_secs() > 300)
        {
            times.pop_front();
        }
    }

    pub fn evicted_total(&self) -> u64 {
        self.evicted_total.load(Ordering::Relaxed)
    }

    pub fn evictions_per_minute(&self) -> f64 {
        let now = Instant::now();
        let mut times = self
            .eviction_times
            .lock()
            .expect("eviction metrics mutex poisoned");
        while times
            .front()
            .is_some_and(|timestamp| now.duration_since(*timestamp).as_secs() > 300)
        {
            times.pop_front();
        }
        times.len() as f64 / 5.0
    }

    pub fn write_prometheus(&self, out: &mut String, active_sessions: usize) {
        out.push_str("# HELP obey_loop_confidence_score Agent-loop confidence by virtual key\n");
        out.push_str("# TYPE obey_loop_confidence_score histogram\n");
        let mut labels = self
            .confidence
            .iter()
            .map(|entry| entry.key().clone())
            .collect::<Vec<_>>();
        labels.sort();
        for label in labels {
            let Some(histogram) = self.confidence.get(&label) else {
                continue;
            };
            let mut cumulative = 0;
            for (index, boundary) in CONFIDENCE_BUCKETS.iter().enumerate() {
                cumulative += histogram.buckets[index].load(Ordering::Relaxed);
                out.push_str(&format!(
                    "obey_loop_confidence_score_bucket{{virtual_key=\"{}\",le=\"{}\"}} {}\n",
                    escape_label(&label),
                    boundary,
                    cumulative
                ));
            }
            let count = histogram.count.load(Ordering::Relaxed);
            out.push_str(&format!(
                "obey_loop_confidence_score_bucket{{virtual_key=\"{}\",le=\"+Inf\"}} {}\n",
                escape_label(&label),
                count
            ));
            out.push_str(&format!(
                "obey_loop_confidence_score_sum{{virtual_key=\"{}\"}} {}\n",
                escape_label(&label),
                histogram.sum_micros.load(Ordering::Relaxed) as f64 / 1_000_000.0
            ));
            out.push_str(&format!(
                "obey_loop_confidence_score_count{{virtual_key=\"{}\"}} {}\n",
                escape_label(&label),
                count
            ));
        }

        out.push_str("# HELP obey_loop_enforcement_total Agent-loop enforcement transitions\n");
        out.push_str("# TYPE obey_loop_enforcement_total counter\n");
        let mut enforcement = self
            .enforcement
            .iter()
            .map(|entry| (entry.key().clone(), entry.value().load(Ordering::Relaxed)))
            .collect::<Vec<_>>();
        enforcement.sort_by(|left, right| left.0.cmp(&right.0));
        for ((level, virtual_key), count) in enforcement {
            out.push_str(&format!(
                "obey_loop_enforcement_total{{level=\"{}\",virtual_key=\"{}\"}} {}\n",
                escape_label(&level),
                escape_label(&virtual_key),
                count
            ));
        }
        out.push_str("# HELP obey_loop_sessions_active Active agent-loop sessions\n# TYPE obey_loop_sessions_active gauge\n");
        out.push_str(&format!("obey_loop_sessions_active {}\n", active_sessions));
        out.push_str("# HELP obey_loop_sessions_evicted_total Evicted agent-loop sessions\n# TYPE obey_loop_sessions_evicted_total counter\n");
        out.push_str(&format!(
            "obey_loop_sessions_evicted_total {}\n",
            self.evicted_total()
        ));
    }
}

impl Default for LoopDetectionMetrics {
    fn default() -> Self {
        Self::new()
    }
}

fn private_virtual_key_label(virtual_key: Option<&str>) -> String {
    virtual_key
        .map(|value| format!("id:{value}"))
        .unwrap_or_else(|| "none".to_string())
}

fn escape_label(value: &str) -> String {
    value
        .replace('\\', "\\\\")
        .replace('"', "\\\"")
        .replace('\n', "\\n")
}
