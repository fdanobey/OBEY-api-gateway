use crate::loop_detection::session::{SessionId, SessionState};
use dashmap::DashMap;
use std::{sync::Arc, time::Duration};
use tokio::time::{interval, timeout};

#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub struct EvictionStats {
    pub scanned: usize,
    pub evicted: usize,
    pub timed_out: bool,
}

pub async fn eviction_loop(
    sessions: Arc<DashMap<SessionId, SessionState>>,
    metrics: Arc<crate::loop_detection::metrics::LoopDetectionMetrics>,
    interval_duration: Duration,
    ttl: Duration,
    max_per_cycle: usize,
) {
    let mut ticker = interval(interval_duration);
    ticker.tick().await;
    loop {
        ticker.tick().await;
        let stats = evict_expired(&sessions, ttl, max_per_cycle).await;
        for _ in 0..stats.evicted {
            metrics.record_eviction();
        }
        if stats.timed_out {
            tracing::warn!(
                scanned = stats.scanned,
                total = sessions.len(),
                "Loop detection eviction cycle timed out"
            );
        }
    }
}

pub async fn evict_expired(
    sessions: &DashMap<SessionId, SessionState>,
    ttl: Duration,
    max_per_cycle: usize,
) -> EvictionStats {
    let total = sessions.len();
    let work = async {
        let now = std::time::Instant::now();
        let mut stats = EvictionStats::default();
        let expired = sessions
            .iter()
            .filter_map(|entry| {
                stats.scanned += 1;
                (now.saturating_duration_since(entry.value().last_active) > ttl)
                    .then(|| entry.key().clone())
            })
            .take(max_per_cycle)
            .collect::<Vec<_>>();
        for session_id in expired {
            if let Some((session_id, state)) = sessions.remove(&session_id) {
                stats.evicted += 1;
                log_eviction(&session_id, &state, "ttl");
            }
        }
        stats
    };

    match timeout(Duration::from_secs(5), work).await {
        Ok(stats) => stats,
        Err(_) => EvictionStats {
            scanned: total,
            evicted: 0,
            timed_out: true,
        },
    }
}

pub fn ensure_capacity(
    sessions: &DashMap<SessionId, SessionState>,
    max_sessions: usize,
) -> Option<SessionId> {
    if sessions.len() < max_sessions {
        return None;
    }
    let lru_id = sessions
        .iter()
        .min_by_key(|entry| entry.value().last_active)
        .map(|entry| entry.key().clone())?;
    sessions.remove(&lru_id).map(|(session_id, state)| {
        log_eviction(&session_id, &state, "capacity");
        session_id
    })
}

pub fn insert_bounded(
    sessions: &DashMap<SessionId, SessionState>,
    session_id: SessionId,
    state: SessionState,
    max_sessions: usize,
    metrics: Option<&crate::loop_detection::metrics::LoopDetectionMetrics>,
) -> Option<SessionId> {
    if sessions.contains_key(&session_id) {
        sessions.insert(session_id, state);
        return None;
    }
    let evicted = ensure_capacity(sessions, max_sessions.max(1));
    if evicted.is_some() {
        if let Some(metrics) = metrics {
            metrics.record_eviction();
        }
    }
    sessions.insert(session_id, state);
    evicted
}

fn log_eviction(session_id: &str, state: &SessionState, reason: &'static str) {
    tracing::info!(
        session_id,
        reason,
        total_requests = state.request_count,
        peak_confidence = state.peak_confidence,
        enforcement_level = ?state.enforcement_level,
        enforcement_actions = state.escalation_history.len(),
        last_active = ?state.last_active,
        "Loop detection session evicted"
    );
}
