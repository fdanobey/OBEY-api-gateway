//! Durable, content-free online adaptation for Smart Routing.

use std::fmt;
#[cfg(unix)]
use std::fs::File;
use std::fs::{self, OpenOptions};
use std::io::{self, Write};
use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::{mpsc, Arc, Mutex, RwLock, Weak};
use std::thread::{self, JoinHandle};
use std::time::Duration;

use chrono::{DateTime, Utc};
use serde::{Deserialize, Serialize};

use super::config::{OnlineOptimizerConfig, TierBoundaries};
use super::tier::{SmartRoutingTier, TaskType};
use super::RoutingOptimizerHook;

const STATE_VERSION: u32 = 1;
const MAX_STATE_BYTES: u64 = 1_048_576;
const MAX_BOUNDARY_MOVE: f64 = 0.05;
const MIN_BOUNDARY_GAP: f64 = 1.0e-6;
const HIGHER_TIER_QUALITY: f64 = 0.7;
const DEFAULT_STATE_PATH: &str = "smart_routing_state.json";

/// Largest single-request cost accepted by the aggregate recorder.
pub const MAX_OUTCOME_COST_USD: f64 = 1_000_000.0;
/// Largest single-request latency accepted by the aggregate recorder (24 hours).
pub const MAX_OUTCOME_LATENCY_MS: f64 = 86_400_000.0;

static TEMP_FILE_SEQUENCE: AtomicU64 = AtomicU64::new(0);
static QUARANTINE_SEQUENCE: AtomicU64 = AtomicU64::new(0);

const ALL_TIERS: [SmartRoutingTier; 3] = [
    SmartRoutingTier::Fast,
    SmartRoutingTier::Balanced,
    SmartRoutingTier::Powerful,
];
const ALL_TASK_TYPES: [TaskType; 7] = [
    TaskType::CodeGeneration,
    TaskType::MathReasoning,
    TaskType::CreativeWriting,
    TaskType::FactualQA,
    TaskType::ToolUse,
    TaskType::Summarization,
    TaskType::General,
];

/// A content-free, bounded request outcome accepted by the optimizer.
#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
pub struct RequestOutcome {
    pub complexity_score: f64,
    pub task_type: TaskType,
    pub tier: SmartRoutingTier,
    pub quality_score: f64,
    pub cost_usd: f64,
    pub latency_ms: f64,
}

impl RequestOutcome {
    fn validate(self) -> Result<Self, OutcomeRejection> {
        validate_closed_unit(self.complexity_score)
            .map_err(|_| OutcomeRejection::ComplexityScore)?;
        validate_closed_unit(self.quality_score).map_err(|_| OutcomeRejection::QualityScore)?;
        validate_bounded_non_negative(self.cost_usd, MAX_OUTCOME_COST_USD)
            .map_err(|_| OutcomeRejection::Cost)?;
        validate_bounded_non_negative(self.latency_ms, MAX_OUTCOME_LATENCY_MS)
            .map_err(|_| OutcomeRejection::Latency)?;
        Ok(self)
    }
}

/// Why an outcome was rejected before it could affect aggregate state.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum OutcomeRejection {
    ComplexityScore,
    QualityScore,
    Cost,
    Latency,
}

/// Result of a bounded outcome-recording attempt.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum RecordResult {
    Disabled,
    Recorded,
    Rejected(OutcomeRejection),
}

/// EMA values for one bounded aggregate bucket.
#[derive(Debug, Clone, Copy, Default, PartialEq, Serialize, Deserialize)]
pub struct AggregateSnapshot {
    pub samples: u64,
    pub complexity_score_ema: Option<f64>,
    pub quality_ema: Option<f64>,
    pub cost_usd_ema: Option<f64>,
    pub latency_ms_ema: Option<f64>,
}

impl AggregateSnapshot {
    fn observe(&mut self, outcome: RequestOutcome, alpha: f64) {
        self.samples = self.samples.saturating_add(1);
        update_ema(
            &mut self.complexity_score_ema,
            outcome.complexity_score,
            alpha,
        );
        update_ema(&mut self.quality_ema, outcome.quality_score, alpha);
        update_ema(&mut self.cost_usd_ema, outcome.cost_usd, alpha);
        update_ema(&mut self.latency_ms_ema, outcome.latency_ms, alpha);
    }

    fn validate(self) -> Result<(), OptimizerStateError> {
        validate_optional_unit("complexity_score_ema", self.complexity_score_ema)?;
        validate_optional_unit("quality_ema", self.quality_ema)?;
        validate_optional_bound("cost_usd_ema", self.cost_usd_ema, MAX_OUTCOME_COST_USD)?;
        validate_optional_bound(
            "latency_ms_ema",
            self.latency_ms_ema,
            MAX_OUTCOME_LATENCY_MS,
        )?;
        if self.samples == 0
            && (self.complexity_score_ema.is_some()
                || self.quality_ema.is_some()
                || self.cost_usd_ema.is_some()
                || self.latency_ms_ema.is_some())
        {
            return Err(OptimizerStateError::InvalidState(
                "aggregate with zero samples contains EMA values".to_owned(),
            ));
        }
        if self.samples > 0
            && (self.complexity_score_ema.is_none()
                || self.quality_ema.is_none()
                || self.cost_usd_ema.is_none()
                || self.latency_ms_ema.is_none())
        {
            return Err(OptimizerStateError::InvalidState(
                "aggregate with samples is missing EMA values".to_owned(),
            ));
        }
        Ok(())
    }
}

/// Aggregate state for one capability tier.
#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
pub struct TierAggregateSnapshot {
    pub tier: SmartRoutingTier,
    pub aggregate: AggregateSnapshot,
}

/// Aggregate state for one task-type and tier pair.
#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
pub struct TaskTierAggregateSnapshot {
    pub task_type: TaskType,
    pub tier: SmartRoutingTier,
    pub aggregate: AggregateSnapshot,
}

/// Adaptive boundaries for one task type.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct TaskBoundarySnapshot {
    pub task_type: TaskType,
    pub boundaries: TierBoundaries,
}

/// Immutable optimizer state published to readers.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct OptimizerSnapshot {
    pub version: u32,
    pub enabled: bool,
    pub configured_boundaries: TierBoundaries,
    pub adjusted_boundaries: Vec<TaskBoundarySnapshot>,
    pub tier_aggregates: Vec<TierAggregateSnapshot>,
    pub task_tier_aggregates: Vec<TaskTierAggregateSnapshot>,
    pub last_update: DateTime<Utc>,
}

impl OptimizerSnapshot {
    /// Return adaptive boundaries for one task type, falling back to configured values.
    pub fn boundaries_for(&self, task_type: TaskType) -> &TierBoundaries {
        self.adjusted_boundaries
            .iter()
            .find(|entry| entry.task_type == task_type)
            .map(|entry| &entry.boundaries)
            .unwrap_or(&self.configured_boundaries)
    }

    /// Return one task/tier aggregate without exposing raw observations.
    pub fn aggregate_for(
        &self,
        task_type: TaskType,
        tier: SmartRoutingTier,
    ) -> Option<AggregateSnapshot> {
        self.task_tier_aggregates
            .iter()
            .find(|entry| entry.task_type == task_type && entry.tier == tier)
            .map(|entry| entry.aggregate)
    }
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
struct PersistedOptimizerState {
    version: u32,
    configured_boundaries: TierBoundaries,
    adjusted_boundaries: Vec<TaskBoundarySnapshot>,
    tier_aggregates: Vec<TierAggregateSnapshot>,
    task_tier_aggregates: Vec<TaskTierAggregateSnapshot>,
    last_update: DateTime<Utc>,
}

impl PersistedOptimizerState {
    fn defaults(boundaries: TierBoundaries, now: DateTime<Utc>) -> Self {
        Self {
            version: STATE_VERSION,
            configured_boundaries: boundaries.clone(),
            adjusted_boundaries: ALL_TASK_TYPES
                .into_iter()
                .map(|task_type| TaskBoundarySnapshot {
                    task_type,
                    boundaries: boundaries.clone(),
                })
                .collect(),
            tier_aggregates: ALL_TIERS
                .into_iter()
                .map(|tier| TierAggregateSnapshot {
                    tier,
                    aggregate: AggregateSnapshot::default(),
                })
                .collect(),
            task_tier_aggregates: ALL_TASK_TYPES
                .into_iter()
                .flat_map(|task_type| {
                    ALL_TIERS
                        .into_iter()
                        .map(move |tier| TaskTierAggregateSnapshot {
                            task_type,
                            tier,
                            aggregate: AggregateSnapshot::default(),
                        })
                })
                .collect(),
            last_update: now,
        }
    }

    fn validate(&self) -> Result<(), OptimizerStateError> {
        if self.version != STATE_VERSION {
            return Err(OptimizerStateError::UnsupportedVersion {
                found: self.version,
                expected: STATE_VERSION,
            });
        }
        validate_boundaries(&self.configured_boundaries)?;
        validate_task_boundaries(&self.adjusted_boundaries)?;
        validate_tier_aggregates(&self.tier_aggregates)?;
        validate_task_tier_aggregates(&self.task_tier_aggregates)?;
        Ok(())
    }

    fn snapshot(&self, enabled: bool) -> OptimizerSnapshot {
        OptimizerSnapshot {
            version: self.version,
            enabled,
            configured_boundaries: self.configured_boundaries.clone(),
            adjusted_boundaries: self.adjusted_boundaries.clone(),
            tier_aggregates: self.tier_aggregates.clone(),
            task_tier_aggregates: self.task_tier_aggregates.clone(),
            last_update: self.last_update,
        }
    }

    fn tier_aggregate_mut(&mut self, tier: SmartRoutingTier) -> &mut AggregateSnapshot {
        &mut self
            .tier_aggregates
            .iter_mut()
            .find(|entry| entry.tier == tier)
            .expect("fixed tier aggregate exists")
            .aggregate
    }

    fn task_tier_aggregate_mut(
        &mut self,
        task_type: TaskType,
        tier: SmartRoutingTier,
    ) -> &mut AggregateSnapshot {
        &mut self
            .task_tier_aggregates
            .iter_mut()
            .find(|entry| entry.task_type == task_type && entry.tier == tier)
            .expect("fixed task/tier aggregate exists")
            .aggregate
    }

    fn task_quality(&self, task_type: TaskType, tier: SmartRoutingTier) -> Option<f64> {
        self.task_tier_aggregates
            .iter()
            .find(|entry| entry.task_type == task_type && entry.tier == tier)
            .and_then(|entry| entry.aggregate.quality_ema)
    }
}

/// Source selected while recovering persisted optimizer state.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum OptimizerStateSource {
    DisabledDefaults,
    Primary,
    LastKnownGood,
    Defaults,
}

/// Bounded startup recovery report suitable for operational telemetry.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct OptimizerLoadReport {
    pub source: OptimizerStateSource,
    pub quarantined_paths: Vec<PathBuf>,
    pub primary_restored: bool,
}

/// State validation failures. No request or response content can enter these variants.
#[derive(Debug)]
pub enum OptimizerStateError {
    Io {
        path: PathBuf,
        source: io::Error,
    },
    CorruptJson {
        path: PathBuf,
        source: serde_json::Error,
    },
    UnsupportedVersion {
        found: u32,
        expected: u32,
    },
    InvalidState(String),
    StateTooLarge {
        path: PathBuf,
        bytes: u64,
    },
}

impl fmt::Display for OptimizerStateError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Io { path, source } => {
                write!(
                    formatter,
                    "failed to read optimizer state `{}`: {source}",
                    path.display()
                )
            }
            Self::CorruptJson { path, source } => write!(
                formatter,
                "optimizer state `{}` contains corrupt JSON: {source}",
                path.display()
            ),
            Self::UnsupportedVersion { found, expected } => write!(
                formatter,
                "unsupported optimizer state version {found}; expected {expected}"
            ),
            Self::InvalidState(detail) => write!(formatter, "invalid optimizer state: {detail}"),
            Self::StateTooLarge { path, bytes } => write!(
                formatter,
                "optimizer state `{}` is {bytes} bytes; maximum is {MAX_STATE_BYTES}",
                path.display()
            ),
        }
    }
}

impl std::error::Error for OptimizerStateError {
    fn source(&self) -> Option<&(dyn std::error::Error + 'static)> {
        match self {
            Self::Io { source, .. } => Some(source),
            Self::CorruptJson { source, .. } => Some(source),
            Self::UnsupportedVersion { .. }
            | Self::InvalidState(_)
            | Self::StateTooLarge { .. } => None,
        }
    }
}

/// Categorized persistence failure reported without exposing observations.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum OptimizerPersistenceFailure {
    Serialize,
    Primary,
    LastKnownGood,
}

/// One validated adaptive boundary movement.
#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
pub struct BoundaryMove {
    pub task_type: TaskType,
    pub boundary: BoundaryKind,
    pub previous: f64,
    pub current: f64,
}

/// The boundary changed by an optimization interval.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum BoundaryKind {
    FastMax,
    BalancedMax,
}

/// Why an optimization call did or did not run.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum OptimizationStatus {
    Disabled,
    IntervalPending,
    Completed,
}

/// Bounded result of one optimization attempt.
#[derive(Debug, Clone)]
pub struct OptimizationReport {
    pub status: OptimizationStatus,
    pub attempted_at: DateTime<Utc>,
    pub next_eligible_at: DateTime<Utc>,
    pub boundary_moves: Vec<BoundaryMove>,
    pub primary_persisted: bool,
    pub last_known_good_persisted: bool,
    pub persistence_failure: Option<OptimizerPersistenceFailure>,
    pub snapshot: Arc<OptimizerSnapshot>,
}

/// Durable, concurrency-safe optimizer with immutable published snapshots.
#[derive(Debug)]
pub struct OnlineOptimizer {
    enabled: bool,
    alpha: f64,
    interval: Duration,
    quality_threshold: f64,
    state_path: PathBuf,
    lkg_path: PathBuf,
    state: Mutex<PersistedOptimizerState>,
    published: RwLock<Arc<OptimizerSnapshot>>,
}

impl OnlineOptimizer {
    /// Load optimizer state, quarantining invalid primary/LKG files and continuing safely.
    pub fn load(
        config: &OnlineOptimizerConfig,
        configured_boundaries: TierBoundaries,
    ) -> (Self, OptimizerLoadReport) {
        Self::load_at(config, configured_boundaries, Utc::now())
    }

    /// Deterministic loading entry point for timers, tests, and controlled startup.
    pub fn load_at(
        config: &OnlineOptimizerConfig,
        configured_boundaries: TierBoundaries,
        now: DateTime<Utc>,
    ) -> (Self, OptimizerLoadReport) {
        let configured_boundaries = normalized_boundaries(configured_boundaries);
        let state_path = config
            .state_path
            .as_deref()
            .map(PathBuf::from)
            .unwrap_or_else(|| PathBuf::from(DEFAULT_STATE_PATH));
        let lkg_path = lkg_path_for(&state_path);
        let alpha = normalized_alpha(config.alpha);
        let interval = Duration::from_secs(config.interval_secs.max(1));
        let quality_threshold = normalized_quality_threshold(config.quality_threshold);

        if !config.enabled {
            let state = PersistedOptimizerState::defaults(configured_boundaries, now);
            let published = Arc::new(state.snapshot(false));
            return (
                Self {
                    enabled: false,
                    alpha,
                    interval,
                    quality_threshold,
                    state_path,
                    lkg_path,
                    state: Mutex::new(state),
                    published: RwLock::new(published),
                },
                OptimizerLoadReport {
                    source: OptimizerStateSource::DisabledDefaults,
                    quarantined_paths: Vec::new(),
                    primary_restored: false,
                },
            );
        }

        let mut quarantined_paths = Vec::new();
        let primary = load_candidate(&state_path);
        let (mut state, source, primary_restored) = match primary {
            CandidateState::Valid(state) => (state, OptimizerStateSource::Primary, false),
            CandidateState::Missing => recover_lkg_or_defaults(
                &state_path,
                &lkg_path,
                configured_boundaries.clone(),
                now,
                &mut quarantined_paths,
            ),
            CandidateState::Invalid => {
                quarantine_if_present(&state_path, now, &mut quarantined_paths);
                recover_lkg_or_defaults(
                    &state_path,
                    &lkg_path,
                    configured_boundaries.clone(),
                    now,
                    &mut quarantined_paths,
                )
            }
        };

        state.configured_boundaries = configured_boundaries.clone();
        if state.last_update > now {
            state.last_update = now;
        }
        if validate_task_boundaries(&state.adjusted_boundaries).is_err() {
            state.adjusted_boundaries =
                PersistedOptimizerState::defaults(configured_boundaries.clone(), now)
                    .adjusted_boundaries;
        }
        let published = Arc::new(state.snapshot(true));
        (
            Self {
                enabled: true,
                alpha,
                interval,
                quality_threshold,
                state_path,
                lkg_path,
                state: Mutex::new(state),
                published: RwLock::new(published),
            },
            OptimizerLoadReport {
                source,
                quarantined_paths,
                primary_restored,
            },
        )
    }

    /// Record only validated numeric and enum fields; no content-bearing type is accepted.
    pub fn record(&self, outcome: RequestOutcome) -> RecordResult {
        if !self.enabled {
            return RecordResult::Disabled;
        }
        let outcome = match outcome.validate() {
            Ok(outcome) => outcome,
            Err(reason) => return RecordResult::Rejected(reason),
        };
        let mut state = lock_unpoisoned(&self.state);
        state
            .tier_aggregate_mut(outcome.tier)
            .observe(outcome, self.alpha);
        state
            .task_tier_aggregate_mut(outcome.task_type, outcome.tier)
            .observe(outcome, self.alpha);
        RecordResult::Recorded
    }

    /// Run one interval-gated optimization and atomically persist the resulting state.
    pub fn optimize(&self) -> OptimizationReport {
        self.optimize_at(Utc::now())
    }

    /// Run one deterministic interval-gated optimization step.
    pub fn optimize_at(&self, now: DateTime<Utc>) -> OptimizationReport {
        if !self.enabled {
            let current = self.snapshot();
            return OptimizationReport {
                status: OptimizationStatus::Disabled,
                attempted_at: now,
                next_eligible_at: current.last_update,
                boundary_moves: Vec::new(),
                primary_persisted: false,
                last_known_good_persisted: false,
                persistence_failure: None,
                snapshot: current,
            };
        }

        let mut state = lock_unpoisoned(&self.state);
        let next_eligible_at = add_std_duration(state.last_update, self.interval);
        if now < next_eligible_at {
            let snapshot = Arc::new(state.snapshot(true));
            self.publish(snapshot.clone());
            return OptimizationReport {
                status: OptimizationStatus::IntervalPending,
                attempted_at: now,
                next_eligible_at,
                boundary_moves: Vec::new(),
                primary_persisted: false,
                last_known_good_persisted: false,
                persistence_failure: None,
                snapshot,
            };
        }

        let mut next = state.clone();
        let boundary_moves = adjust_boundaries(&mut next, self.quality_threshold);
        next.last_update = now;
        let bytes = match serde_json::to_vec_pretty(&next) {
            Ok(bytes) => bytes,
            Err(_) => {
                let snapshot = Arc::new(next.snapshot(true));
                *state = next;
                self.publish(snapshot.clone());
                return OptimizationReport {
                    status: OptimizationStatus::Completed,
                    attempted_at: now,
                    next_eligible_at: add_std_duration(now, self.interval),
                    boundary_moves,
                    primary_persisted: false,
                    last_known_good_persisted: false,
                    persistence_failure: Some(OptimizerPersistenceFailure::Serialize),
                    snapshot,
                };
            }
        };

        let primary_persisted = atomic_replace(&self.state_path, &bytes).is_ok();
        let last_known_good_persisted = if primary_persisted {
            atomic_replace(&self.lkg_path, &bytes).is_ok()
        } else {
            false
        };
        let persistence_failure = if !primary_persisted {
            Some(OptimizerPersistenceFailure::Primary)
        } else if !last_known_good_persisted {
            Some(OptimizerPersistenceFailure::LastKnownGood)
        } else {
            None
        };
        let snapshot = Arc::new(next.snapshot(true));
        *state = next;
        self.publish(snapshot.clone());

        OptimizationReport {
            status: OptimizationStatus::Completed,
            attempted_at: now,
            next_eligible_at: add_std_duration(now, self.interval),
            boundary_moves,
            primary_persisted,
            last_known_good_persisted,
            persistence_failure,
            snapshot,
        }
    }

    /// Clone the current immutable snapshot for lock-free use after this call returns.
    pub fn snapshot(&self) -> Arc<OptimizerSnapshot> {
        read_unpoisoned(&self.published).clone()
    }

    /// Start the interval timer. Dropping the returned worker wakes and joins it.
    pub fn start(self: &Arc<Self>) -> Option<OptimizerWorker> {
        if !self.enabled {
            return None;
        }
        let optimizer = Arc::downgrade(self);
        let interval = self.interval;
        let (shutdown, receiver) = mpsc::channel();
        let handle = thread::Builder::new()
            .name("smart-routing-optimizer".to_owned())
            .spawn(move || run_worker(optimizer, interval, receiver))
            .ok()?;
        Some(OptimizerWorker {
            shutdown: Some(shutdown),
            handle: Some(handle),
        })
    }

    fn publish(&self, snapshot: Arc<OptimizerSnapshot>) {
        *write_unpoisoned(&self.published) = snapshot;
    }
}

impl RoutingOptimizerHook for OnlineOptimizer {
    fn cost_quality_threshold(
        &self,
        _model_group: &str,
        task_type: TaskType,
        configured: f64,
    ) -> f64 {
        if !self.enabled {
            return normalize_hook_threshold(configured);
        }
        let snapshot = self.snapshot();
        let adaptive = snapshot.boundaries_for(task_type);
        let baseline = &snapshot.configured_boundaries;
        let fast_shift = baseline.fast_max - adaptive.fast_max;
        let balanced_shift = baseline.balanced_max - adaptive.balanced_max;
        (normalize_hook_threshold(configured) + (fast_shift + balanced_shift) / 2.0).clamp(0.0, 1.0)
    }
}

/// Owns and promptly stops the optimizer's background interval timer.
#[derive(Debug)]
pub struct OptimizerWorker {
    shutdown: Option<mpsc::Sender<()>>,
    handle: Option<JoinHandle<()>>,
}

impl OptimizerWorker {
    pub fn shutdown(mut self) {
        self.stop();
    }

    fn stop(&mut self) {
        if let Some(shutdown) = self.shutdown.take() {
            let _ = shutdown.send(());
        }
        if let Some(handle) = self.handle.take() {
            let _ = handle.join();
        }
    }
}

impl Drop for OptimizerWorker {
    fn drop(&mut self) {
        self.stop();
    }
}

fn run_worker(optimizer: Weak<OnlineOptimizer>, interval: Duration, receiver: mpsc::Receiver<()>) {
    loop {
        match receiver.recv_timeout(interval) {
            Ok(()) | Err(mpsc::RecvTimeoutError::Disconnected) => return,
            Err(mpsc::RecvTimeoutError::Timeout) => {
                let Some(optimizer) = optimizer.upgrade() else {
                    return;
                };
                let _ = optimizer.optimize();
            }
        }
    }
}

fn adjust_boundaries(
    state: &mut PersistedOptimizerState,
    quality_threshold: f64,
) -> Vec<BoundaryMove> {
    let mut moves = Vec::new();
    for task_type in ALL_TASK_TYPES {
        let fast_quality = state.task_quality(task_type, SmartRoutingTier::Fast);
        let balanced_quality = state.task_quality(task_type, SmartRoutingTier::Balanced);
        let powerful_quality = state.task_quality(task_type, SmartRoutingTier::Powerful);
        let boundaries = &mut state
            .adjusted_boundaries
            .iter_mut()
            .find(|entry| entry.task_type == task_type)
            .expect("fixed task boundary exists")
            .boundaries;

        if fast_quality.is_some_and(|quality| quality < quality_threshold)
            && balanced_quality.is_some_and(|quality| quality > HIGHER_TIER_QUALITY)
        {
            let previous = boundaries.fast_max;
            let current = (previous - MAX_BOUNDARY_MOVE).max(MIN_BOUNDARY_GAP);
            if current < previous && current + MIN_BOUNDARY_GAP < boundaries.balanced_max {
                boundaries.fast_max = current;
                moves.push(BoundaryMove {
                    task_type,
                    boundary: BoundaryKind::FastMax,
                    previous,
                    current,
                });
            }
        }

        if balanced_quality.is_some_and(|quality| quality < quality_threshold)
            && powerful_quality.is_some_and(|quality| quality > HIGHER_TIER_QUALITY)
        {
            let previous = boundaries.balanced_max;
            let floor = boundaries.fast_max + MIN_BOUNDARY_GAP;
            let current = (previous - MAX_BOUNDARY_MOVE).max(floor);
            if current < previous && current < 1.0 {
                boundaries.balanced_max = current;
                moves.push(BoundaryMove {
                    task_type,
                    boundary: BoundaryKind::BalancedMax,
                    previous,
                    current,
                });
            }
        }
    }
    debug_assert!(validate_task_boundaries(&state.adjusted_boundaries).is_ok());
    moves
}

fn recover_lkg_or_defaults(
    state_path: &Path,
    lkg_path: &Path,
    configured_boundaries: TierBoundaries,
    now: DateTime<Utc>,
    quarantined_paths: &mut Vec<PathBuf>,
) -> (PersistedOptimizerState, OptimizerStateSource, bool) {
    match load_candidate(lkg_path) {
        CandidateState::Valid(state) => {
            let restored = serde_json::to_vec_pretty(&state)
                .ok()
                .is_some_and(|bytes| atomic_replace(state_path, &bytes).is_ok());
            (state, OptimizerStateSource::LastKnownGood, restored)
        }
        CandidateState::Invalid => {
            quarantine_if_present(lkg_path, now, quarantined_paths);
            (
                PersistedOptimizerState::defaults(configured_boundaries, now),
                OptimizerStateSource::Defaults,
                false,
            )
        }
        CandidateState::Missing => (
            PersistedOptimizerState::defaults(configured_boundaries, now),
            OptimizerStateSource::Defaults,
            false,
        ),
    }
}

enum CandidateState {
    Missing,
    Valid(PersistedOptimizerState),
    Invalid,
}

fn load_candidate(path: &Path) -> CandidateState {
    let metadata = match fs::metadata(path) {
        Ok(metadata) => metadata,
        Err(source) if source.kind() == io::ErrorKind::NotFound => return CandidateState::Missing,
        Err(_) => return CandidateState::Invalid,
    };
    if metadata.len() > MAX_STATE_BYTES {
        return CandidateState::Invalid;
    }
    let bytes = match fs::read(path) {
        Ok(bytes) => bytes,
        Err(source) if source.kind() == io::ErrorKind::NotFound => return CandidateState::Missing,
        Err(_) => return CandidateState::Invalid,
    };
    let state: PersistedOptimizerState = match serde_json::from_slice(&bytes) {
        Ok(state) => state,
        Err(_) => return CandidateState::Invalid,
    };
    match state.validate() {
        Ok(()) => CandidateState::Valid(state),
        Err(_) => CandidateState::Invalid,
    }
}

fn quarantine_if_present(path: &Path, now: DateTime<Utc>, quarantined: &mut Vec<PathBuf>) {
    if !path.exists() {
        return;
    }
    let sequence = QUARANTINE_SEQUENCE.fetch_add(1, Ordering::Relaxed);
    let file_name = path
        .file_name()
        .and_then(|name| name.to_str())
        .unwrap_or("optimizer-state.json");
    let quarantine = path.with_file_name(format!(
        "{file_name}.corrupt.{}.{}",
        now.timestamp_millis(),
        sequence
    ));
    if fs::rename(path, &quarantine).is_ok() {
        quarantined.push(quarantine);
    }
}

fn validate_task_boundaries(entries: &[TaskBoundarySnapshot]) -> Result<(), OptimizerStateError> {
    if entries.len() != ALL_TASK_TYPES.len() {
        return Err(OptimizerStateError::InvalidState(
            "task boundary cardinality is invalid".to_owned(),
        ));
    }
    for task_type in ALL_TASK_TYPES {
        let matching: Vec<_> = entries
            .iter()
            .filter(|entry| entry.task_type == task_type)
            .collect();
        if matching.len() != 1 {
            return Err(OptimizerStateError::InvalidState(
                "task boundary keys are missing or duplicated".to_owned(),
            ));
        }
        validate_boundaries(&matching[0].boundaries)?;
    }
    Ok(())
}

fn validate_tier_aggregates(entries: &[TierAggregateSnapshot]) -> Result<(), OptimizerStateError> {
    if entries.len() != ALL_TIERS.len() {
        return Err(OptimizerStateError::InvalidState(
            "tier aggregate cardinality is invalid".to_owned(),
        ));
    }
    for tier in ALL_TIERS {
        let matching: Vec<_> = entries.iter().filter(|entry| entry.tier == tier).collect();
        if matching.len() != 1 {
            return Err(OptimizerStateError::InvalidState(
                "tier aggregate keys are missing or duplicated".to_owned(),
            ));
        }
        matching[0].aggregate.validate()?;
    }
    Ok(())
}

fn validate_task_tier_aggregates(
    entries: &[TaskTierAggregateSnapshot],
) -> Result<(), OptimizerStateError> {
    if entries.len() != ALL_TASK_TYPES.len() * ALL_TIERS.len() {
        return Err(OptimizerStateError::InvalidState(
            "task/tier aggregate cardinality is invalid".to_owned(),
        ));
    }
    for task_type in ALL_TASK_TYPES {
        for tier in ALL_TIERS {
            let matching: Vec<_> = entries
                .iter()
                .filter(|entry| entry.task_type == task_type && entry.tier == tier)
                .collect();
            if matching.len() != 1 {
                return Err(OptimizerStateError::InvalidState(
                    "task/tier aggregate keys are missing or duplicated".to_owned(),
                ));
            }
            matching[0].aggregate.validate()?;
        }
    }
    Ok(())
}

fn validate_boundaries(boundaries: &TierBoundaries) -> Result<(), OptimizerStateError> {
    if boundaries.fast_max.is_finite()
        && boundaries.balanced_max.is_finite()
        && boundaries.fast_max > 0.0
        && boundaries.fast_max + MIN_BOUNDARY_GAP <= boundaries.balanced_max
        && boundaries.balanced_max < 1.0
    {
        Ok(())
    } else {
        Err(OptimizerStateError::InvalidState(
            "tier boundaries must satisfy 0 < fast_max < balanced_max < 1".to_owned(),
        ))
    }
}

fn normalized_boundaries(boundaries: TierBoundaries) -> TierBoundaries {
    if validate_boundaries(&boundaries).is_ok() {
        boundaries
    } else {
        TierBoundaries::default()
    }
}

fn normalized_alpha(alpha: f64) -> f64 {
    if alpha.is_finite() && alpha > 0.0 {
        alpha.min(1.0)
    } else {
        0.01
    }
}

fn normalized_quality_threshold(threshold: f64) -> f64 {
    if threshold.is_finite() {
        threshold.clamp(0.0, 1.0)
    } else {
        0.5
    }
}

fn normalize_hook_threshold(threshold: f64) -> f64 {
    if threshold.is_finite() {
        threshold.clamp(0.0, 1.0)
    } else {
        0.5
    }
}

fn update_ema(current: &mut Option<f64>, observed: f64, alpha: f64) {
    *current = Some(match *current {
        Some(previous) => alpha * observed + (1.0 - alpha) * previous,
        None => observed,
    });
}

fn validate_closed_unit(value: f64) -> Result<(), ()> {
    if value.is_finite() && (0.0..=1.0).contains(&value) {
        Ok(())
    } else {
        Err(())
    }
}

fn validate_bounded_non_negative(value: f64, maximum: f64) -> Result<(), ()> {
    if value.is_finite() && (0.0..=maximum).contains(&value) {
        Ok(())
    } else {
        Err(())
    }
}

fn validate_optional_unit(field: &str, value: Option<f64>) -> Result<(), OptimizerStateError> {
    if value.is_none_or(|value| validate_closed_unit(value).is_ok()) {
        Ok(())
    } else {
        Err(OptimizerStateError::InvalidState(format!(
            "{field} must be finite and in 0..=1"
        )))
    }
}

fn validate_optional_bound(
    field: &str,
    value: Option<f64>,
    maximum: f64,
) -> Result<(), OptimizerStateError> {
    if value.is_none_or(|value| validate_bounded_non_negative(value, maximum).is_ok()) {
        Ok(())
    } else {
        Err(OptimizerStateError::InvalidState(format!(
            "{field} must be finite and in 0..={maximum}"
        )))
    }
}

fn add_std_duration(time: DateTime<Utc>, duration: Duration) -> DateTime<Utc> {
    chrono::Duration::from_std(duration)
        .ok()
        .and_then(|duration| time.checked_add_signed(duration))
        .unwrap_or(DateTime::<Utc>::MAX_UTC)
}

fn lkg_path_for(path: &Path) -> PathBuf {
    let mut value = path.as_os_str().to_os_string();
    value.push(".lkg");
    PathBuf::from(value)
}

fn atomic_replace(path: &Path, bytes: &[u8]) -> io::Result<()> {
    let parent = path
        .parent()
        .filter(|parent| !parent.as_os_str().is_empty())
        .unwrap_or_else(|| Path::new("."));
    fs::create_dir_all(parent)?;
    let file_name = path
        .file_name()
        .and_then(|name| name.to_str())
        .unwrap_or("optimizer-state.json");
    let sequence = TEMP_FILE_SEQUENCE.fetch_add(1, Ordering::Relaxed);
    let temporary = parent.join(format!(
        ".{file_name}.{}.{}.tmp",
        std::process::id(),
        sequence
    ));

    let result = (|| {
        let mut file = OpenOptions::new()
            .write(true)
            .create_new(true)
            .open(&temporary)?;
        file.write_all(bytes)?;
        file.sync_all()?;
        drop(file);
        replace_file(&temporary, path)?;
        sync_parent_directory(parent)?;
        Ok(())
    })();
    if result.is_err() {
        let _ = fs::remove_file(&temporary);
    }
    result
}

#[cfg(not(windows))]
fn replace_file(temporary: &Path, destination: &Path) -> io::Result<()> {
    fs::rename(temporary, destination)
}

#[cfg(windows)]
fn replace_file(temporary: &Path, destination: &Path) -> io::Result<()> {
    use std::os::windows::ffi::OsStrExt;

    const MOVEFILE_REPLACE_EXISTING: u32 = 0x1;
    const MOVEFILE_WRITE_THROUGH: u32 = 0x8;

    #[link(name = "Kernel32")]
    extern "system" {
        fn MoveFileExW(existing: *const u16, replacement: *const u16, flags: u32) -> i32;
    }

    let existing: Vec<u16> = temporary.as_os_str().encode_wide().chain(Some(0)).collect();
    let replacement: Vec<u16> = destination
        .as_os_str()
        .encode_wide()
        .chain(Some(0))
        .collect();
    let result = unsafe {
        MoveFileExW(
            existing.as_ptr(),
            replacement.as_ptr(),
            MOVEFILE_REPLACE_EXISTING | MOVEFILE_WRITE_THROUGH,
        )
    };
    if result == 0 {
        Err(io::Error::last_os_error())
    } else {
        Ok(())
    }
}

#[cfg(unix)]
fn sync_parent_directory(parent: &Path) -> io::Result<()> {
    File::open(parent)?.sync_all()
}

#[cfg(not(unix))]
fn sync_parent_directory(_parent: &Path) -> io::Result<()> {
    Ok(())
}

fn lock_unpoisoned<T>(mutex: &Mutex<T>) -> std::sync::MutexGuard<'_, T> {
    mutex
        .lock()
        .unwrap_or_else(|poisoned| poisoned.into_inner())
}

fn read_unpoisoned<T>(lock: &RwLock<T>) -> std::sync::RwLockReadGuard<'_, T> {
    lock.read().unwrap_or_else(|poisoned| poisoned.into_inner())
}

fn write_unpoisoned<T>(lock: &RwLock<T>) -> std::sync::RwLockWriteGuard<'_, T> {
    lock.write()
        .unwrap_or_else(|poisoned| poisoned.into_inner())
}

#[cfg(test)]
mod tests {
    use super::*;
    use chrono::Duration as ChronoDuration;
    use proptest::prelude::*;
    use tempfile::TempDir;

    fn config(path: &Path) -> OnlineOptimizerConfig {
        OnlineOptimizerConfig {
            enabled: true,
            alpha: 0.5,
            interval_secs: 600,
            state_path: Some(path.to_string_lossy().into_owned()),
            quality_threshold: 0.5,
        }
    }

    fn boundaries() -> TierBoundaries {
        TierBoundaries {
            fast_max: 0.33,
            balanced_max: 0.66,
        }
    }

    fn outcome_for(
        task_type: TaskType,
        tier: SmartRoutingTier,
        complexity_score: f64,
        quality_score: f64,
        cost_usd: f64,
        latency_ms: f64,
    ) -> RequestOutcome {
        RequestOutcome {
            complexity_score,
            task_type,
            tier,
            quality_score,
            cost_usd,
            latency_ms,
        }
    }

    fn outcome(tier: SmartRoutingTier, quality_score: f64) -> RequestOutcome {
        outcome_for(
            TaskType::CodeGeneration,
            tier,
            0.4,
            quality_score,
            0.01,
            100.0,
        )
    }

    fn assert_close(actual: f64, expected: f64) {
        assert!(
            (actual - expected).abs() <= 1.0e-12,
            "expected {expected}, got {actual}"
        );
    }

    proptest! {
    #![proptest_config(ProptestConfig::with_cases(256))]

    // Feature: smart-routing, Property 26: optimizer boundary adjustment per interval is bounded.
    #[test]
    fn property_26_boundary_adjustment_is_at_most_five_percent_per_interval(
        fast_max in 0.05f64..0.90,
        balanced_gap in 0.05f64..0.95,
        low_quality in 0.0f64..0.5,
        high_quality in 0.700_000_000_001f64..=1.0,
        adjust_fast in any::<bool>(),
    ) {
        let balanced_max = (fast_max + balanced_gap).min(0.999_999);
        prop_assume!(fast_max < balanced_max);
        let configured = TierBoundaries {
            fast_max,
            balanced_max,
        };
        let directory = TempDir::new().unwrap();
        let now = Utc::now();
        let (optimizer, _) = OnlineOptimizer::load_at(
            &config(&directory.path().join("state.json")),
            configured.clone(),
            now,
        );
        let (lower_tier, higher_tier, expected_boundary) = if adjust_fast {
            (
                SmartRoutingTier::Fast,
                SmartRoutingTier::Balanced,
                BoundaryKind::FastMax,
            )
        } else {
            (
                SmartRoutingTier::Balanced,
                SmartRoutingTier::Powerful,
                BoundaryKind::BalancedMax,
            )
        };
        optimizer.record(outcome(lower_tier, low_quality));
        optimizer.record(outcome(higher_tier, high_quality));

        let completed = optimizer.optimize_at(now + ChronoDuration::seconds(600));

        prop_assert_eq!(completed.status, OptimizationStatus::Completed);
        prop_assert_eq!(completed.boundary_moves.len(), 1);
        let movement = completed.boundary_moves[0];
        prop_assert_eq!(movement.task_type, TaskType::CodeGeneration);
        prop_assert_eq!(movement.boundary, expected_boundary);
        prop_assert!(movement.current <= movement.previous);
        prop_assert!(movement.previous - movement.current <= MAX_BOUNDARY_MOVE + f64::EPSILON);
        let adjusted = completed
            .snapshot
            .boundaries_for(TaskType::CodeGeneration)
            .clone();
        prop_assert!(adjusted.fast_max > 0.0);
        prop_assert!(adjusted.fast_max < adjusted.balanced_max);
        prop_assert!(adjusted.balanced_max < 1.0);

        let pending = optimizer.optimize_at(now + ChronoDuration::seconds(1_199));
        prop_assert_eq!(pending.status, OptimizationStatus::IntervalPending);
        prop_assert!(pending.boundary_moves.is_empty());
        prop_assert_eq!(
            pending.snapshot.boundaries_for(TaskType::CodeGeneration),
            &adjusted
        );
    }
    }

    #[test]
    fn ema_updates_all_numeric_aggregates() {
        let directory = TempDir::new().unwrap();
        let path = directory.path().join("state.json");
        let mut optimizer_config = config(&path);
        optimizer_config.alpha = 0.25;
        let (optimizer, _) = OnlineOptimizer::load(&optimizer_config, boundaries());
        let first = outcome_for(
            TaskType::CodeGeneration,
            SmartRoutingTier::Fast,
            0.2,
            0.4,
            2.0,
            200.0,
        );
        let second = outcome_for(
            TaskType::CodeGeneration,
            SmartRoutingTier::Fast,
            0.8,
            1.0,
            6.0,
            1_000.0,
        );

        assert_eq!(optimizer.record(first), RecordResult::Recorded);
        assert_eq!(optimizer.record(second), RecordResult::Recorded);
        let aggregate = lock_unpoisoned(&optimizer.state)
            .task_tier_aggregates
            .iter()
            .find(|entry| {
                entry.task_type == TaskType::CodeGeneration && entry.tier == SmartRoutingTier::Fast
            })
            .unwrap()
            .aggregate;

        assert_eq!(aggregate.samples, 2);
        assert_close(aggregate.complexity_score_ema.unwrap(), 0.35);
        assert_close(aggregate.quality_ema.unwrap(), 0.55);
        assert_close(aggregate.cost_usd_ema.unwrap(), 3.0);
        assert_close(aggregate.latency_ms_ema.unwrap(), 400.0);
    }

    #[test]
    fn task_type_aggregates_and_boundaries_are_isolated() {
        let directory = TempDir::new().unwrap();
        let now = Utc::now();
        let (optimizer, _) = OnlineOptimizer::load_at(
            &config(&directory.path().join("state.json")),
            boundaries(),
            now,
        );
        for tier in [SmartRoutingTier::Fast, SmartRoutingTier::Balanced] {
            let quality = if tier == SmartRoutingTier::Fast {
                0.1
            } else {
                0.9
            };
            assert_eq!(
                optimizer.record(outcome_for(
                    TaskType::MathReasoning,
                    tier,
                    0.6,
                    quality,
                    0.02,
                    120.0,
                )),
                RecordResult::Recorded
            );
        }

        let report = optimizer.optimize_at(now + ChronoDuration::seconds(600));
        let math = report.snapshot.boundaries_for(TaskType::MathReasoning);
        let code = report.snapshot.boundaries_for(TaskType::CodeGeneration);

        assert_close(math.fast_max, boundaries().fast_max - MAX_BOUNDARY_MOVE);
        assert_eq!(math.balanced_max, boundaries().balanced_max);
        assert_eq!(code, &boundaries());
        assert_eq!(
            report
                .snapshot
                .aggregate_for(TaskType::MathReasoning, SmartRoutingTier::Fast)
                .unwrap()
                .samples,
            1
        );
        assert_eq!(
            report
                .snapshot
                .aggregate_for(TaskType::CodeGeneration, SmartRoutingTier::Fast)
                .unwrap(),
            AggregateSnapshot::default()
        );
        assert!(report
            .boundary_moves
            .iter()
            .all(|movement| { movement.task_type == TaskType::MathReasoning }));
    }

    #[test]
    fn persisted_state_round_trips_aggregates_and_adjusted_boundaries() {
        let directory = TempDir::new().unwrap();
        let path = directory.path().join("state.json");
        let now = Utc::now();
        let (optimizer, _) = OnlineOptimizer::load_at(&config(&path), boundaries(), now);
        optimizer.record(outcome_for(
            TaskType::CreativeWriting,
            SmartRoutingTier::Fast,
            0.3,
            0.1,
            0.04,
            80.0,
        ));
        optimizer.record(outcome_for(
            TaskType::CreativeWriting,
            SmartRoutingTier::Balanced,
            0.7,
            0.9,
            0.08,
            160.0,
        ));
        let persisted = optimizer.optimize_at(now + ChronoDuration::seconds(600));
        assert!(persisted.primary_persisted);
        assert!(persisted.last_known_good_persisted);

        let (reloaded, report) = OnlineOptimizer::load_at(
            &config(&path),
            boundaries(),
            now + ChronoDuration::seconds(601),
        );

        assert_eq!(report.source, OptimizerStateSource::Primary);
        assert!(!report.primary_restored);
        let restored = reloaded.snapshot();
        assert_eq!(
            restored.boundaries_for(TaskType::CreativeWriting),
            persisted.snapshot.boundaries_for(TaskType::CreativeWriting)
        );
        assert_eq!(
            restored.aggregate_for(TaskType::CreativeWriting, SmartRoutingTier::Fast),
            persisted
                .snapshot
                .aggregate_for(TaskType::CreativeWriting, SmartRoutingTier::Fast)
        );
    }

    #[test]
    fn corrupt_primary_and_lkg_are_quarantined_before_safe_defaults() {
        let directory = TempDir::new().unwrap();
        let path = directory.path().join("state.json");
        let lkg_path = lkg_path_for(&path);
        fs::write(&path, b"not-json").unwrap();
        fs::write(&lkg_path, br#"{"version":999}"#).unwrap();
        let now = Utc::now();

        let (optimizer, report) = OnlineOptimizer::load_at(&config(&path), boundaries(), now);

        assert_eq!(report.source, OptimizerStateSource::Defaults);
        assert!(!report.primary_restored);
        assert_eq!(report.quarantined_paths.len(), 2);
        assert!(report
            .quarantined_paths
            .iter()
            .all(|quarantine| quarantine.exists()));
        assert!(!path.exists());
        assert!(!lkg_path.exists());
        assert_eq!(optimizer.snapshot().configured_boundaries, boundaries());
        assert!(optimizer
            .snapshot()
            .task_tier_aggregates
            .iter()
            .all(|entry| entry.aggregate == AggregateSnapshot::default()));
    }

    #[test]
    fn records_bounded_ema_without_raw_content_fields() {
        let directory = TempDir::new().unwrap();
        let now = Utc::now();
        let (optimizer, _) = OnlineOptimizer::load_at(
            &config(&directory.path().join("state.json")),
            boundaries(),
            now,
        );
        assert_eq!(
            optimizer.record(outcome(SmartRoutingTier::Fast, 0.2)),
            RecordResult::Recorded
        );
        assert_eq!(
            optimizer.record(outcome(SmartRoutingTier::Fast, 0.8)),
            RecordResult::Recorded
        );
        let report = optimizer.optimize_at(now + ChronoDuration::minutes(10));
        let aggregate = report
            .snapshot
            .aggregate_for(TaskType::CodeGeneration, SmartRoutingTier::Fast)
            .unwrap();
        assert_eq!(aggregate.samples, 2);
        assert_eq!(aggregate.quality_ema, Some(0.5));
        let json = serde_json::to_string(report.snapshot.as_ref()).unwrap();
        assert!(!json.contains("request"));
        assert!(!json.contains("response"));
    }

    #[test]
    fn interval_gate_caps_moves_and_preserves_ordering() {
        let directory = TempDir::new().unwrap();
        let now = Utc::now();
        let (optimizer, _) = OnlineOptimizer::load_at(
            &config(&directory.path().join("state.json")),
            boundaries(),
            now,
        );
        optimizer.record(outcome(SmartRoutingTier::Fast, 0.2));
        optimizer.record(outcome(SmartRoutingTier::Balanced, 0.8));

        let pending = optimizer.optimize_at(now + ChronoDuration::seconds(599));
        assert_eq!(pending.status, OptimizationStatus::IntervalPending);
        let completed = optimizer.optimize_at(now + ChronoDuration::seconds(600));
        assert_eq!(completed.status, OptimizationStatus::Completed);
        assert_eq!(completed.boundary_moves.len(), 1);
        let movement = completed.boundary_moves[0];
        assert!(movement.previous - movement.current <= MAX_BOUNDARY_MOVE);
        let adjusted = completed.snapshot.boundaries_for(TaskType::CodeGeneration);
        assert!(adjusted.fast_max > 0.0);
        assert!(adjusted.fast_max < adjusted.balanced_max);
        assert!(adjusted.balanced_max < 1.0);
        assert_eq!(
            optimizer
                .optimize_at(now + ChronoDuration::seconds(601))
                .status,
            OptimizationStatus::IntervalPending
        );
    }

    #[test]
    fn persists_and_recovers_last_known_good_after_corruption() {
        let directory = TempDir::new().unwrap();
        let path = directory.path().join("state.json");
        let now = Utc::now();
        let (optimizer, _) = OnlineOptimizer::load_at(&config(&path), boundaries(), now);
        optimizer.record(outcome(SmartRoutingTier::Fast, 0.2));
        let report = optimizer.optimize_at(now + ChronoDuration::minutes(10));
        assert!(report.primary_persisted);
        assert!(report.last_known_good_persisted);
        fs::write(&path, b"not-json").unwrap();

        let (reloaded, load_report) = OnlineOptimizer::load_at(
            &config(&path),
            boundaries(),
            now + ChronoDuration::minutes(11),
        );
        assert_eq!(load_report.source, OptimizerStateSource::LastKnownGood);
        assert!(load_report.primary_restored);
        assert_eq!(load_report.quarantined_paths.len(), 1);
        assert_eq!(
            reloaded
                .snapshot()
                .aggregate_for(TaskType::CodeGeneration, SmartRoutingTier::Fast)
                .unwrap()
                .samples,
            1
        );
    }

    #[test]
    fn disabled_path_does_not_read_or_write_state() {
        let directory = TempDir::new().unwrap();
        let path = directory.path().join("state.json");
        fs::write(&path, b"corrupt").unwrap();
        let mut disabled = config(&path);
        disabled.enabled = false;
        let (optimizer, report) = OnlineOptimizer::load(&disabled, boundaries());
        assert_eq!(report.source, OptimizerStateSource::DisabledDefaults);
        assert!(report.quarantined_paths.is_empty());
        assert_eq!(
            optimizer.record(outcome(SmartRoutingTier::Fast, 0.0)),
            RecordResult::Disabled
        );
        assert_eq!(optimizer.optimize().status, OptimizationStatus::Disabled);
        assert_eq!(fs::read(&path).unwrap(), b"corrupt");
        assert_eq!(
            optimizer.cost_quality_threshold("ignored", TaskType::General, 0.4),
            0.4
        );
    }

    #[test]
    fn rejects_non_finite_and_unbounded_outcomes() {
        let directory = TempDir::new().unwrap();
        let (optimizer, _) =
            OnlineOptimizer::load(&config(&directory.path().join("state.json")), boundaries());
        let mut invalid = outcome(SmartRoutingTier::Fast, 0.5);
        invalid.cost_usd = f64::INFINITY;
        assert_eq!(
            optimizer.record(invalid),
            RecordResult::Rejected(OutcomeRejection::Cost)
        );
        invalid = outcome(SmartRoutingTier::Fast, 0.5);
        invalid.latency_ms = MAX_OUTCOME_LATENCY_MS + 1.0;
        assert_eq!(
            optimizer.record(invalid),
            RecordResult::Rejected(OutcomeRejection::Latency)
        );
    }
}
