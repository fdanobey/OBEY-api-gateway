//! Durable, content-free budget accounting for smart-routing model groups.

use std::collections::HashMap;
use std::fmt;
#[cfg(unix)]
use std::fs::File;
use std::fs::{self, OpenOptions};
use std::io::{self, Write};
use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::Mutex;

use async_trait::async_trait;
use chrono::{DateTime, Datelike, TimeZone, Timelike, Utc};
use serde::{Deserialize, Serialize};

use crate::config::ProviderModel;

use super::config::BudgetLimits;
use super::tier::SmartRoutingTier;
use super::{BudgetCheckInput, BudgetDecision, BudgetPolicy, BudgetRejectionReason};

const STATE_VERSION: u32 = 1;
const DOWNGRADE_THRESHOLD: f64 = 0.80;
const EXHAUSTED_THRESHOLD: f64 = 1.0;
const TOKENS_PER_MILLION: f64 = 1_000_000.0;

static TEMP_FILE_SEQUENCE: AtomicU64 = AtomicU64::new(0);

/// Identifies whether recorded spend came from provider usage or a projection.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum SpendAccuracy {
    Exact,
    Estimated,
}

/// Token counts used to calculate a model charge.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct TokenUsage {
    pub input_tokens: u64,
    pub output_tokens: u64,
}

/// A finite, non-negative model charge and its accounting label.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct SpendCharge {
    pub usd: f64,
    pub accuracy: SpendAccuracy,
}

impl SpendCharge {
    /// Build a validated charge that is safe to persist as JSON.
    pub fn new(usd: f64, accuracy: SpendAccuracy) -> Result<Self, BudgetPricingError> {
        validate_money("charge_usd", usd)?;
        Ok(Self { usd, accuracy })
    }
}

/// Pricing failures are explicit so non-finite values never enter budget state.
#[derive(Debug, Clone, PartialEq)]
pub enum BudgetPricingError {
    InvalidRate { field: &'static str, value: f64 },
    NonFiniteCost,
    NoMatchingModel,
}

impl fmt::Display for BudgetPricingError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::InvalidRate { field, value } => {
                write!(formatter, "invalid {field} model price: {value}")
            }
            Self::NonFiniteCost => formatter.write_str("calculated model cost is not finite"),
            Self::NoMatchingModel => {
                formatter.write_str("no model matched the pinned routing context")
            }
        }
    }
}

impl std::error::Error for BudgetPricingError {}

/// Calculate a finite model charge from per-million-token prices.
pub fn calculate_model_charge(
    model: &ProviderModel,
    usage: TokenUsage,
    accuracy: SpendAccuracy,
) -> Result<SpendCharge, BudgetPricingError> {
    validate_rate(
        "cost_per_million_input_tokens",
        model.cost_per_million_input_tokens,
    )?;
    validate_rate(
        "cost_per_million_output_tokens",
        model.cost_per_million_output_tokens,
    )?;

    let input_cost =
        usage.input_tokens as f64 * model.cost_per_million_input_tokens / TOKENS_PER_MILLION;
    let output_cost =
        usage.output_tokens as f64 * model.cost_per_million_output_tokens / TOKENS_PER_MILLION;
    let usd = input_cost + output_cost;
    if !usd.is_finite() {
        return Err(BudgetPricingError::NonFiniteCost);
    }

    SpendCharge::new(usd, accuracy)
}

/// Categorized failures while loading persisted budget state.
#[derive(Debug)]
pub enum BudgetLoadError {
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
    InvalidState {
        detail: String,
    },
}

impl fmt::Display for BudgetLoadError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Io { path, source } => {
                write!(
                    formatter,
                    "failed to load budget state `{}`: {source}",
                    path.display()
                )
            }
            Self::CorruptJson { path, source } => write!(
                formatter,
                "budget state `{}` contains corrupt JSON: {source}",
                path.display()
            ),
            Self::UnsupportedVersion { found, expected } => write!(
                formatter,
                "unsupported budget state version {found}; expected {expected}"
            ),
            Self::InvalidState { detail } => write!(formatter, "invalid budget state: {detail}"),
        }
    }
}

impl std::error::Error for BudgetLoadError {
    fn source(&self) -> Option<&(dyn std::error::Error + 'static)> {
        match self {
            Self::Io { source, .. } => Some(source),
            Self::CorruptJson { source, .. } => Some(source),
            Self::UnsupportedVersion { .. } | Self::InvalidState { .. } => None,
        }
    }
}

/// Runtime accounting failures. `BudgetPolicy` maps these conservatively to policy rejection.
#[derive(Debug)]
pub enum BudgetControllerError {
    Pricing(BudgetPricingError),
    LockPoisoned,
    Serialize(serde_json::Error),
    Persist { path: PathBuf, source: io::Error },
}

impl fmt::Display for BudgetControllerError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Pricing(source) => write!(formatter, "budget pricing failed: {source}"),
            Self::LockPoisoned => formatter.write_str("budget state lock is poisoned"),
            Self::Serialize(source) => {
                write!(formatter, "budget state serialization failed: {source}")
            }
            Self::Persist { path, source } => write!(
                formatter,
                "failed to persist budget state `{}`: {source}",
                path.display()
            ),
        }
    }
}

impl std::error::Error for BudgetControllerError {
    fn source(&self) -> Option<&(dyn std::error::Error + 'static)> {
        match self {
            Self::Pricing(source) => Some(source),
            Self::Serialize(source) => Some(source),
            Self::Persist { source, .. } => Some(source),
            Self::LockPoisoned => None,
        }
    }
}

impl From<BudgetPricingError> for BudgetControllerError {
    fn from(source: BudgetPricingError) -> Self {
        Self::Pricing(source)
    }
}

/// Exact and estimated spend within one UTC window.
#[derive(Debug, Clone, Copy, Default, PartialEq, Serialize, Deserialize)]
pub struct SpendBreakdown {
    pub exact_usd: f64,
    pub estimated_usd: f64,
}

impl SpendBreakdown {
    pub fn total_usd(self) -> f64 {
        self.exact_usd + self.estimated_usd
    }

    fn add(&mut self, charge: SpendCharge) -> Result<(), BudgetPricingError> {
        let target = match charge.accuracy {
            SpendAccuracy::Exact => &mut self.exact_usd,
            SpendAccuracy::Estimated => &mut self.estimated_usd,
        };
        let updated = *target + charge.usd;
        validate_money("accumulated_spend_usd", updated)?;
        *target = updated;
        Ok(())
    }

    fn validate(self, field: &str) -> Result<(), BudgetLoadError> {
        for (label, value) in [
            ("exact_usd", self.exact_usd),
            ("estimated_usd", self.estimated_usd),
        ] {
            if !value.is_finite() || value < 0.0 {
                return Err(BudgetLoadError::InvalidState {
                    detail: format!("{field}.{label} must be finite and non-negative"),
                });
            }
        }
        Ok(())
    }
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
struct WindowSpend {
    starts_at: DateTime<Utc>,
    spend: SpendBreakdown,
}

impl WindowSpend {
    fn new(starts_at: DateTime<Utc>) -> Self {
        Self {
            starts_at,
            spend: SpendBreakdown::default(),
        }
    }

    fn reset_to(&mut self, starts_at: DateTime<Utc>) -> bool {
        if self.starts_at == starts_at {
            return false;
        }
        *self = Self::new(starts_at);
        true
    }
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
struct GroupSpend {
    hourly: WindowSpend,
    daily: WindowSpend,
    monthly: WindowSpend,
}

impl GroupSpend {
    fn new(now: DateTime<Utc>) -> Self {
        Self {
            hourly: WindowSpend::new(hour_start(now)),
            daily: WindowSpend::new(day_start(now)),
            monthly: WindowSpend::new(month_start(now)),
        }
    }

    fn reset_expired(&mut self, now: DateTime<Utc>) -> bool {
        let hourly = self.hourly.reset_to(hour_start(now));
        let daily = self.daily.reset_to(day_start(now));
        let monthly = self.monthly.reset_to(month_start(now));
        hourly || daily || monthly
    }

    fn add(&mut self, charge: SpendCharge) -> Result<(), BudgetPricingError> {
        self.hourly.spend.add(charge)?;
        self.daily.spend.add(charge)?;
        self.monthly.spend.add(charge)?;
        Ok(())
    }
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
struct PersistedBudgetState {
    version: u32,
    groups: HashMap<String, GroupSpend>,
}

impl Default for PersistedBudgetState {
    fn default() -> Self {
        Self {
            version: STATE_VERSION,
            groups: HashMap::new(),
        }
    }
}

impl PersistedBudgetState {
    fn validate(&self) -> Result<(), BudgetLoadError> {
        if self.version != STATE_VERSION {
            return Err(BudgetLoadError::UnsupportedVersion {
                found: self.version,
                expected: STATE_VERSION,
            });
        }
        for (group, spend) in &self.groups {
            spend
                .hourly
                .spend
                .validate(&format!("groups.{group}.hourly"))?;
            spend
                .daily
                .spend
                .validate(&format!("groups.{group}.daily"))?;
            spend
                .monthly
                .spend
                .validate(&format!("groups.{group}.monthly"))?;
            if spend.hourly.starts_at != hour_start(spend.hourly.starts_at)
                || spend.daily.starts_at != day_start(spend.daily.starts_at)
                || spend.monthly.starts_at != month_start(spend.monthly.starts_at)
            {
                return Err(BudgetLoadError::InvalidState {
                    detail: format!("groups.{group} contains a non-UTC-aligned window"),
                });
            }
        }
        Ok(())
    }
}

/// One period's current spend, configured limit, and non-negative remainder.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct BudgetPeriodSnapshot {
    pub starts_at: DateTime<Utc>,
    pub spend: SpendBreakdown,
    pub limit_usd: Option<f64>,
    pub remaining_usd: Option<f64>,
}

/// Content-free budget state for one model group.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct BudgetSnapshot {
    pub hourly: BudgetPeriodSnapshot,
    pub daily: BudgetPeriodSnapshot,
    pub monthly: BudgetPeriodSnapshot,
}

/// Durable, concurrency-safe per-model-group budget controller.
#[derive(Debug)]
pub struct BudgetController {
    state_path: PathBuf,
    state: Mutex<PersistedBudgetState>,
}

impl BudgetController {
    /// Load state from disk, or start empty when the path does not exist.
    ///
    /// Malformed JSON is returned as [`BudgetLoadError::CorruptJson`] and is
    /// never silently replaced.
    pub fn load(path: impl Into<PathBuf>) -> Result<Self, BudgetLoadError> {
        let state_path = path.into();
        let state = match fs::read(&state_path) {
            Ok(bytes) => {
                let state: PersistedBudgetState =
                    serde_json::from_slice(&bytes).map_err(|source| {
                        BudgetLoadError::CorruptJson {
                            path: state_path.clone(),
                            source,
                        }
                    })?;
                state.validate()?;
                state
            }
            Err(source) if source.kind() == io::ErrorKind::NotFound => {
                PersistedBudgetState::default()
            }
            Err(source) => {
                return Err(BudgetLoadError::Io {
                    path: state_path,
                    source,
                })
            }
        };
        Ok(Self {
            state_path,
            state: Mutex::new(state),
        })
    }

    /// Record an already calculated exact or estimated charge at the current UTC time.
    pub fn record_spend(
        &self,
        model_group: &str,
        charge: SpendCharge,
    ) -> Result<BudgetSnapshot, BudgetControllerError> {
        self.record_spend_at(model_group, charge, Utc::now(), None)
    }

    /// Calculate and record token usage against one model's configured prices.
    pub fn record_usage(
        &self,
        model_group: &str,
        model: &ProviderModel,
        usage: TokenUsage,
        accuracy: SpendAccuracy,
    ) -> Result<BudgetSnapshot, BudgetControllerError> {
        let charge = calculate_model_charge(model, usage, accuracy)?;
        self.record_spend(model_group, charge)
    }

    /// Return current UTC-window spend and remaining amounts without storing content.
    pub fn snapshot(
        &self,
        model_group: &str,
        limits: Option<&BudgetLimits>,
    ) -> Result<BudgetSnapshot, BudgetControllerError> {
        self.snapshot_at(model_group, limits, Utc::now())
    }

    /// Perform the policy check with an explicit time for deterministic callers and tests.
    pub fn check_at(
        &self,
        input: BudgetCheckInput<'_>,
        now: DateTime<Utc>,
    ) -> Result<BudgetDecision, BudgetControllerError> {
        let Some(limits) = input.configured_limits else {
            return Ok(BudgetDecision::Allow);
        };
        let projected_charge = self.estimate_request_charge(&input)?;
        let mut state = self
            .state
            .lock()
            .map_err(|_| BudgetControllerError::LockPoisoned)?;
        let mut next = state.clone();
        let group = next
            .groups
            .entry(input.model_group.name.clone())
            .or_insert_with(|| GroupSpend::new(now));
        let changed = group.reset_expired(now);
        let decision = decide(group, limits, projected_charge.usd);
        if changed {
            self.persist(&next)?;
            *state = next;
        }
        Ok(decision)
    }

    fn estimate_request_charge(
        &self,
        input: &BudgetCheckInput<'_>,
    ) -> Result<SpendCharge, BudgetPricingError> {
        let usage = TokenUsage {
            input_tokens: input
                .classification
                .token_estimate
                .saturating_add(input.pinned_context.additional_input_tokens),
            output_tokens: u64::from(input.request.max_tokens.unwrap_or(0))
                .max(input.pinned_context.reserved_output_tokens),
        };
        let candidates: Vec<&ProviderModel> = input
            .model_group
            .models
            .iter()
            .filter(|model| {
                input
                    .pinned_context
                    .provider
                    .as_deref()
                    .is_none_or(|provider| provider == model.provider)
                    && input
                        .pinned_context
                        .model
                        .as_deref()
                        .is_none_or(|name| name == model.model)
            })
            .collect();
        if candidates.is_empty() {
            return Err(BudgetPricingError::NoMatchingModel);
        }

        let mut maximum = 0.0_f64;
        for model in candidates {
            let charge = calculate_model_charge(model, usage, SpendAccuracy::Estimated)?;
            maximum = maximum.max(charge.usd);
        }
        SpendCharge::new(maximum, SpendAccuracy::Estimated)
    }

    fn record_spend_at(
        &self,
        model_group: &str,
        charge: SpendCharge,
        now: DateTime<Utc>,
        limits: Option<&BudgetLimits>,
    ) -> Result<BudgetSnapshot, BudgetControllerError> {
        let charge = SpendCharge::new(charge.usd, charge.accuracy)?;
        let mut state = self
            .state
            .lock()
            .map_err(|_| BudgetControllerError::LockPoisoned)?;
        let mut next = state.clone();
        let group = next
            .groups
            .entry(model_group.to_owned())
            .or_insert_with(|| GroupSpend::new(now));
        group.reset_expired(now);
        group.add(charge)?;
        let snapshot = snapshot_for(group, limits);
        self.persist(&next)?;
        *state = next;
        Ok(snapshot)
    }

    fn snapshot_at(
        &self,
        model_group: &str,
        limits: Option<&BudgetLimits>,
        now: DateTime<Utc>,
    ) -> Result<BudgetSnapshot, BudgetControllerError> {
        let mut state = self
            .state
            .lock()
            .map_err(|_| BudgetControllerError::LockPoisoned)?;
        let mut next = state.clone();
        let group = next
            .groups
            .entry(model_group.to_owned())
            .or_insert_with(|| GroupSpend::new(now));
        let changed = group.reset_expired(now);
        let snapshot = snapshot_for(group, limits);
        if changed {
            self.persist(&next)?;
            *state = next;
        }
        Ok(snapshot)
    }

    fn persist(&self, state: &PersistedBudgetState) -> Result<(), BudgetControllerError> {
        let bytes = serde_json::to_vec_pretty(state).map_err(BudgetControllerError::Serialize)?;
        atomic_replace(&self.state_path, &bytes).map_err(|source| BudgetControllerError::Persist {
            path: self.state_path.clone(),
            source,
        })
    }
}

#[async_trait]
impl BudgetPolicy for BudgetController {
    async fn check(&self, input: BudgetCheckInput<'_>) -> BudgetDecision {
        self.check_at(input, Utc::now())
            .unwrap_or(BudgetDecision::Reject {
                reason: BudgetRejectionReason::Policy,
            })
    }
}

fn decide(group: &GroupSpend, limits: &BudgetLimits, projected_usd: f64) -> BudgetDecision {
    for (limit, spent, reason) in [
        (
            limits.hourly_limit_usd,
            group.hourly.spend.total_usd(),
            BudgetRejectionReason::HourlyLimit,
        ),
        (
            limits.daily_limit_usd,
            group.daily.spend.total_usd(),
            BudgetRejectionReason::DailyLimit,
        ),
        (
            limits.monthly_limit_usd,
            group.monthly.spend.total_usd(),
            BudgetRejectionReason::MonthlyLimit,
        ),
    ] {
        if limit.is_some_and(|limit| (spent + projected_usd) / limit >= EXHAUSTED_THRESHOLD) {
            return BudgetDecision::Reject { reason };
        }
    }

    let should_downgrade = [
        (limits.hourly_limit_usd, group.hourly.spend.total_usd()),
        (limits.daily_limit_usd, group.daily.spend.total_usd()),
        (limits.monthly_limit_usd, group.monthly.spend.total_usd()),
    ]
    .into_iter()
    .any(|(limit, spent)| {
        limit.is_some_and(|limit| (spent + projected_usd) / limit >= DOWNGRADE_THRESHOLD)
    });
    if should_downgrade {
        BudgetDecision::Downgrade {
            maximum_tier: SmartRoutingTier::Balanced,
        }
    } else {
        BudgetDecision::Allow
    }
}

fn snapshot_for(group: &GroupSpend, limits: Option<&BudgetLimits>) -> BudgetSnapshot {
    let limits = limits.cloned().unwrap_or_default();
    BudgetSnapshot {
        hourly: period_snapshot(&group.hourly, limits.hourly_limit_usd),
        daily: period_snapshot(&group.daily, limits.daily_limit_usd),
        monthly: period_snapshot(&group.monthly, limits.monthly_limit_usd),
    }
}

fn period_snapshot(window: &WindowSpend, limit_usd: Option<f64>) -> BudgetPeriodSnapshot {
    BudgetPeriodSnapshot {
        starts_at: window.starts_at,
        spend: window.spend,
        limit_usd,
        remaining_usd: limit_usd.map(|limit| (limit - window.spend.total_usd()).max(0.0)),
    }
}

fn validate_rate(field: &'static str, value: f64) -> Result<(), BudgetPricingError> {
    if !value.is_finite() || value < 0.0 {
        Err(BudgetPricingError::InvalidRate { field, value })
    } else {
        Ok(())
    }
}

fn validate_money(field: &'static str, value: f64) -> Result<(), BudgetPricingError> {
    if !value.is_finite() || value < 0.0 {
        Err(BudgetPricingError::InvalidRate { field, value })
    } else {
        Ok(())
    }
}

fn hour_start(now: DateTime<Utc>) -> DateTime<Utc> {
    Utc.with_ymd_and_hms(now.year(), now.month(), now.day(), now.hour(), 0, 0)
        .single()
        .expect("UTC calendar components are valid")
}

fn day_start(now: DateTime<Utc>) -> DateTime<Utc> {
    Utc.with_ymd_and_hms(now.year(), now.month(), now.day(), 0, 0, 0)
        .single()
        .expect("UTC calendar components are valid")
}

fn month_start(now: DateTime<Utc>) -> DateTime<Utc> {
    Utc.with_ymd_and_hms(now.year(), now.month(), 1, 0, 0, 0)
        .single()
        .expect("UTC calendar components are valid")
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
        .unwrap_or("budget-state.json");
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

#[cfg(test)]
mod tests {
    use std::sync::Arc;

    use chrono::Duration;
    use proptest::prelude::*;
    use serde_json::{Map, Value};
    use tempfile::tempdir;

    use crate::config::ModelGroup;
    use crate::models::openai::OpenAIRequest;

    use super::*;
    use crate::smart_routing::tier::{ClassifierUsed, ComplexityScore, TaskType};
    use crate::smart_routing::{Classification, PinnedRoutingContext};

    fn at(year: i32, month: u32, day: u32, hour: u32, minute: u32) -> DateTime<Utc> {
        Utc.with_ymd_and_hms(year, month, day, hour, minute, 0)
            .single()
            .unwrap()
    }

    fn model(input_price: f64, output_price: f64) -> ProviderModel {
        ProviderModel {
            provider: "provider".to_owned(),
            model: "model".to_owned(),
            cost_per_million_input_tokens: input_price,
            cost_per_million_output_tokens: output_price,
            priority: 1,
            structured_output_passthrough: None,
            tier: Some(SmartRoutingTier::Powerful),
            context_window: 100_000,
            specializations: vec![],
        }
    }

    fn limits(value: f64) -> BudgetLimits {
        BudgetLimits {
            hourly_limit_usd: Some(value),
            daily_limit_usd: Some(value),
            monthly_limit_usd: Some(value),
        }
    }

    fn request() -> OpenAIRequest {
        OpenAIRequest {
            model: "group".to_owned(),
            messages: vec![],
            stream: false,
            temperature: None,
            max_tokens: Some(0),
            extra: Map::new(),
        }
    }

    fn classification(tokens: u64) -> Classification {
        Classification {
            score: ComplexityScore::new(0.9),
            task_type: TaskType::General,
            classifier: ClassifierUsed::Heuristic,
            token_estimate: tokens,
        }
    }

    #[test]
    fn pricing_is_finite_and_labeled() {
        let charge = calculate_model_charge(
            &model(2.0, 8.0),
            TokenUsage {
                input_tokens: 500_000,
                output_tokens: 250_000,
            },
            SpendAccuracy::Exact,
        )
        .unwrap();
        assert_eq!(
            charge,
            SpendCharge {
                usd: 3.0,
                accuracy: SpendAccuracy::Exact
            }
        );
        assert!(matches!(
            calculate_model_charge(
                &model(f64::NAN, 1.0),
                TokenUsage {
                    input_tokens: 1,
                    output_tokens: 1
                },
                SpendAccuracy::Estimated,
            ),
            Err(BudgetPricingError::InvalidRate { .. })
        ));
    }

    #[test]
    fn exact_and_estimated_spend_are_separate_and_durable() {
        let directory = tempdir().unwrap();
        let path = directory.path().join("nested").join("budget.json");
        let controller = BudgetController::load(&path).unwrap();
        let now = at(2026, 8, 5, 12, 30);
        controller
            .record_spend_at(
                "group",
                SpendCharge::new(2.0, SpendAccuracy::Exact).unwrap(),
                now,
                Some(&limits(10.0)),
            )
            .unwrap();
        let snapshot = controller
            .record_spend_at(
                "group",
                SpendCharge::new(1.5, SpendAccuracy::Estimated).unwrap(),
                now,
                Some(&limits(10.0)),
            )
            .unwrap();
        assert_eq!(snapshot.hourly.spend.exact_usd, 2.0);
        assert_eq!(snapshot.hourly.spend.estimated_usd, 1.5);
        assert_eq!(snapshot.hourly.remaining_usd, Some(6.5));

        let reloaded = BudgetController::load(path).unwrap();
        let snapshot = reloaded
            .snapshot_at("group", Some(&limits(10.0)), now)
            .unwrap();
        assert_eq!(snapshot.daily.spend.total_usd(), 3.5);
    }

    #[test]
    fn thresholds_and_exhausted_precedence_are_deterministic() {
        let group = GroupSpend::new(at(2026, 8, 5, 12, 0));
        assert_eq!(decide(&group, &limits(10.0), 7.99), BudgetDecision::Allow);
        assert_eq!(
            decide(&group, &limits(10.0), 8.0),
            BudgetDecision::Downgrade {
                maximum_tier: SmartRoutingTier::Balanced
            }
        );
        assert_eq!(
            decide(&group, &limits(10.0), 10.0),
            BudgetDecision::Reject {
                reason: BudgetRejectionReason::HourlyLimit
            }
        );
    }

    proptest! {
    #![proptest_config(ProptestConfig::with_cases(256))]

    // Feature: smart-routing, Property 27: Budget pressure permits at most a one-tier downgrade.
    #[test]
    fn property_27_budget_downgrade_is_capped_at_one_tier(
    limit_units in 1_000u32..=1_000_000,
    utilization_millis in 801u32..1_000,
    period in 0u8..3,
    ) {
    let limit = f64::from(limit_units);
    let projected = limit * f64::from(utilization_millis) / 1_000.0;
    let mut configured_limits = BudgetLimits::default();
    match period {
    0 => configured_limits.hourly_limit_usd = Some(limit),
    1 => configured_limits.daily_limit_usd = Some(limit),
    _ => configured_limits.monthly_limit_usd = Some(limit),
    }
    let group = GroupSpend::new(at(2026, 8, 5, 12, 0));

    let decision = decide(&group, &configured_limits, projected);
    prop_assert_eq!(
    decision,
    BudgetDecision::Downgrade {
    maximum_tier: SmartRoutingTier::Balanced,
    }
    );
    if let BudgetDecision::Downgrade { maximum_tier } = decision {
    prop_assert_ne!(maximum_tier, SmartRoutingTier::Fast);
    }
    }

    // Feature: smart-routing, Property 28: Reaching 100% rejects for the deterministic period.
    #[test]
    fn property_28_full_budget_rejects_for_deterministic_period(
    limit_units in 1u32..=1_000_000,
    spent_units in 0u32..=1_000_000,
    period in 0u8..3,
    ) {
    let limit = f64::from(limit_units);
    let spent = f64::from(spent_units.min(limit_units));
    let projected = limit - spent;
    let now = at(2026, 8, 5, 12, 0);
    let mut group = GroupSpend::new(now);
    group.add(SpendCharge::new(spent, SpendAccuracy::Exact).unwrap()).unwrap();
    let mut configured_limits = BudgetLimits::default();
    let expected_reason = match period {
    0 => {
    configured_limits.hourly_limit_usd = Some(limit);
    BudgetRejectionReason::HourlyLimit
    }
    1 => {
    configured_limits.daily_limit_usd = Some(limit);
    BudgetRejectionReason::DailyLimit
    }
    _ => {
    configured_limits.monthly_limit_usd = Some(limit);
    BudgetRejectionReason::MonthlyLimit
    }
    };

    let first = decide(&group, &configured_limits, projected);
    let second = decide(&group, &configured_limits, projected);
    prop_assert_eq!(
    first,
    BudgetDecision::Reject {
    reason: expected_reason,
    }
    );
    prop_assert_eq!(second, first);
    }
    }

    #[test]
    fn utc_boundaries_reset_only_expired_windows() {
        let directory = tempdir().unwrap();
        let controller = BudgetController::load(directory.path().join("budget.json")).unwrap();
        let before = at(2026, 8, 31, 23, 30);
        controller
            .record_spend_at(
                "group",
                SpendCharge::new(4.0, SpendAccuracy::Exact).unwrap(),
                before,
                None,
            )
            .unwrap();
        let next_hour = before + Duration::hours(1);
        let snapshot = controller.snapshot_at("group", None, next_hour).unwrap();
        assert_eq!(snapshot.hourly.spend.total_usd(), 0.0);
        assert_eq!(snapshot.daily.spend.total_usd(), 0.0);
        assert_eq!(snapshot.monthly.spend.total_usd(), 0.0);
        assert_eq!(snapshot.monthly.starts_at, at(2026, 9, 1, 0, 0));
    }

    #[test]
    fn concurrent_updates_do_not_lose_spend() {
        let directory = tempdir().unwrap();
        let controller =
            Arc::new(BudgetController::load(directory.path().join("budget.json")).unwrap());
        let now = at(2026, 8, 5, 12, 30);
        let handles: Vec<_> = (0..16)
            .map(|_| {
                let controller = Arc::clone(&controller);
                std::thread::spawn(move || {
                    controller
                        .record_spend_at(
                            "group",
                            SpendCharge::new(0.25, SpendAccuracy::Exact).unwrap(),
                            now,
                            None,
                        )
                        .unwrap();
                })
            })
            .collect();
        for handle in handles {
            handle.join().unwrap();
        }
        let snapshot = controller.snapshot_at("group", None, now).unwrap();
        assert_eq!(snapshot.hourly.spend.exact_usd, 4.0);
    }

    #[test]
    fn corrupt_json_returns_categorized_load_error() {
        let directory = tempdir().unwrap();
        let path = directory.path().join("budget.json");
        fs::write(&path, b"{not-json").unwrap();
        assert!(matches!(
            BudgetController::load(path),
            Err(BudgetLoadError::CorruptJson { .. })
        ));
    }

    #[tokio::test]
    async fn budget_policy_uses_conservative_projected_model_cost() {
        let directory = tempdir().unwrap();
        let controller = BudgetController::load(directory.path().join("budget.json")).unwrap();
        let request = request();
        let group = ModelGroup {
            name: "group".to_owned(),
            version_fallback_enabled: false,
            compression: None,
            memory: None,
            structured_output: None,
            models: vec![model(1.0, 1.0), model(10.0, 10.0)],
        };
        let pinned = PinnedRoutingContext::default();
        let configured_limits = limits(10.0);
        let decision = controller
            .check_at(
                BudgetCheckInput {
                    request: &request,
                    model_group: &group,
                    pinned_context: &pinned,
                    classification: classification(800_000),
                    configured_limits: Some(&configured_limits),
                },
                at(2026, 8, 5, 12, 30),
            )
            .unwrap();
        assert_eq!(
            decision,
            BudgetDecision::Downgrade {
                maximum_tier: SmartRoutingTier::Balanced
            }
        );
    }

    #[test]
    fn persisted_json_contains_no_request_content() {
        let directory = tempdir().unwrap();
        let path = directory.path().join("budget.json");
        let controller = BudgetController::load(&path).unwrap();
        controller
            .record_spend_at(
                "group",
                SpendCharge::new(1.0, SpendAccuracy::Exact).unwrap(),
                at(2026, 8, 5, 12, 30),
                None,
            )
            .unwrap();
        let json: Value = serde_json::from_slice(&fs::read(path).unwrap()).unwrap();
        assert!(json.get("groups").is_some());
        assert!(json.to_string().contains("exact_usd"));
        assert!(!json.to_string().contains("messages"));
    }
}
