//! Background lifecycle scheduling for memory relevance decay.

use std::sync::Arc;
use std::time::Duration;

use async_trait::async_trait;
use chrono::Utc;
use tokio::sync::watch;
use tokio::task::JoinHandle;
use tokio::time::{self, Instant, MissedTickBehavior};
use tracing::{error, info};

use super::metrics::MemoryMetrics;
use super::store::{DecayCycleResult, MemoryStore};
use super::MemoryError;

const MIN_SCHEDULE_HOURS: u32 = 1;

#[async_trait]
pub trait VectorRetryCallback: Send + Sync {
    async fn retry_pending(&self) -> Result<u64, MemoryError>;
}

pub trait EvictionEventPublisher: Send + Sync {
    fn publish_eviction(&self, namespace: &str, count: u64);
}

/// Owns exactly one decay task and cancels it when replaced or dropped.
pub struct DecayScheduler {
    store: MemoryStore,
    metrics: Arc<MemoryMetrics>,
    max_memories_per_namespace: usize,
    vector_retry: Option<Arc<dyn VectorRetryCallback>>,
    eviction_publisher: Option<Arc<dyn EvictionEventPublisher>>,
    cancel: Option<watch::Sender<bool>>,
    handle: Option<JoinHandle<()>>,
}

impl DecayScheduler {
    pub fn start(
        store: MemoryStore,
        schedule_hours: u32,
        max_memories_per_namespace: usize,
    ) -> Result<Self, MemoryError> {
        Self::with_metrics(
            store,
            schedule_hours,
            max_memories_per_namespace,
            Arc::new(MemoryMetrics::new()),
        )
    }

    pub(crate) fn with_metrics(
        store: MemoryStore,
        schedule_hours: u32,
        max_memories_per_namespace: usize,
        metrics: Arc<MemoryMetrics>,
    ) -> Result<Self, MemoryError> {
        validate_cap(max_memories_per_namespace)?;
        let mut scheduler = Self {
            store,
            metrics,
            max_memories_per_namespace,
            vector_retry: None,
            eviction_publisher: None,
            cancel: None,
            handle: None,
        };
        scheduler.restart_decay_scheduler(schedule_hours, max_memories_per_namespace)?;
        Ok(scheduler)
    }

    pub fn set_vector_retry_callback(&mut self, callback: Arc<dyn VectorRetryCallback>) {
        self.vector_retry = Some(callback);
    }

    pub fn set_eviction_publisher(&mut self, publisher: Arc<dyn EvictionEventPublisher>) {
        self.eviction_publisher = Some(publisher);
    }

    pub fn restart_decay_scheduler(
        &mut self,
        schedule_hours: u32,
        max_memories_per_namespace: usize,
    ) -> Result<(), MemoryError> {
        validate_cap(max_memories_per_namespace)?;
        self.cancel_current();
        self.max_memories_per_namespace = max_memories_per_namespace;
        let period = Duration::from_secs(u64::from(schedule_hours.max(MIN_SCHEDULE_HOURS)) * 3_600);
        let (cancel, vector_retry_receiver) = watch::channel(false);
        self.cancel = Some(cancel);
        self.handle = Some(spawn_scheduler(
            self.store.clone(),
            self.metrics.clone(),
            period,
            max_memories_per_namespace,
            self.vector_retry.clone(),
            self.eviction_publisher.clone(),
            vector_retry_receiver,
        ));

        Ok(())
    }

    pub fn cancel(&mut self) {
        self.cancel_current();
    }

    pub async fn run_once(&self) -> Result<DecayCycleResult, MemoryError> {
        let result = run_cycle(self.store.clone(), self.max_memories_per_namespace).await?;
        self.metrics.record_decay_evictions(result.evicted_count);
        publish_evictions(self.eviction_publisher.as_deref(), &result);
        Ok(result)
    }

    fn cancel_current(&mut self) {
        if let Some(cancel) = self.cancel.take() {
            let _ = cancel.send(true);
        }
        if let Some(handle) = self.handle.take() {
            handle.abort();
        }
    }
}

impl Drop for DecayScheduler {
    fn drop(&mut self) {
        self.cancel_current();
    }
}

fn validate_cap(max_memories_per_namespace: usize) -> Result<(), MemoryError> {
    if max_memories_per_namespace == 0 {
        return Err(MemoryError::Config(
            "max_memories_per_namespace must be at least 1".to_string(),
        ));
    }
    Ok(())
}

fn spawn_scheduler(
    store: MemoryStore,
    metrics: Arc<MemoryMetrics>,
    period: Duration,
    max_memories_per_namespace: usize,
    vector_retry: Option<Arc<dyn VectorRetryCallback>>,
    eviction_publisher: Option<Arc<dyn EvictionEventPublisher>>,
    mut cancel: watch::Receiver<bool>,
) -> JoinHandle<()> {
    tokio::spawn(async move {
        let mut interval = time::interval_at(Instant::now() + period, period);
        interval.set_missed_tick_behavior(MissedTickBehavior::Skip);
        loop {
            tokio::select! {
            changed = cancel.changed() => {
            if changed.is_err() || *cancel.borrow() {
            break;
            }
            }
            _ = interval.tick() => {
            match run_cycle(store.clone(), max_memories_per_namespace).await {
            Ok(result) => {
            metrics.record_decay_evictions(result.evicted_count);
            if let Some(callback) = &vector_retry {
            if let Err(error) = callback.retry_pending().await {
            error!(error = %error, "memory vector retry cycle failed");
            }
            }
                    publish_evictions(eviction_publisher.as_deref(), &result);
                    for eviction in &result.namespace_evictions {
                        info!(
            namespace = %eviction.namespace,
            evicted_count = eviction.evicted_count,
            lowest_evicted_score = eviction.lowest_evicted_score,
            "memory decay evicted namespace entries"
            );
            }
            }
            Err(error) => error!(error = %error, "memory decay cycle failed"),
            }
            }
            }
        }
    })
}

fn publish_evictions(publisher: Option<&dyn EvictionEventPublisher>, result: &DecayCycleResult) {
    let Some(publisher) = publisher else {
        return;
    };
    for eviction in &result.namespace_evictions {
        publisher.publish_eviction(&eviction.namespace, eviction.evicted_count);
    }
}

async fn run_cycle(
    store: MemoryStore,
    max_memories_per_namespace: usize,
) -> Result<DecayCycleResult, MemoryError> {
    tokio::task::spawn_blocking(move || {
        store.run_decay_cycle(max_memories_per_namespace, Utc::now())
    })
    .await
    .map_err(|error| MemoryError::TaskFailed(format!("memory decay task failed: {error}")))?
}

#[cfg(test)]
mod tests {
    use std::path::Path;

    use super::*;

    #[tokio::test]
    async fn direct_cycle_runs_on_blocking_pool() {
        let store = MemoryStore::new(Path::new(":memory:")).unwrap();
        let scheduler = DecayScheduler {
            store: store.clone(),
            metrics: Arc::new(MemoryMetrics::new()),
            max_memories_per_namespace: 10,
            vector_retry: None,
            eviction_publisher: None,
            cancel: None,
            handle: None,
        };

        let result = scheduler.run_once().await.unwrap();
        assert_eq!(result.decayed_count, 0);
        assert_eq!(result.evicted_count, 0);
        assert_eq!(result.vector_retry_pending_count, 0);
        assert!(store.stats().unwrap().last_decay_cycle.is_some());
    }

    #[tokio::test]
    async fn cancellation_and_restart_replace_the_owned_task() {
        let store = MemoryStore::new(Path::new(":memory:")).unwrap();
        let mut scheduler = DecayScheduler::start(store, 1, 10).unwrap();
        assert!(scheduler
            .handle
            .as_ref()
            .is_some_and(|handle| !handle.is_finished()));

        scheduler.restart_decay_scheduler(2, 10).unwrap();
        assert!(scheduler
            .handle
            .as_ref()
            .is_some_and(|handle| !handle.is_finished()));

        scheduler.cancel();
        assert!(scheduler.handle.is_none());
        assert!(scheduler.cancel.is_none());
    }
}
