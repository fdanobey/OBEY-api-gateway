//! Concurrency-safe A/B experiments for smart-routing policies.
//!
//! This module retains only fixed-cardinality aggregate statistics. Request IDs
//! are used transiently for assignment and are never stored; request/response
//! content and tenant identifiers are not accepted by the recording API.

use std::sync::{RwLock, RwLockReadGuard, RwLockWriteGuard};

use serde::Serialize;
use sha2::{Digest, Sha256};

use super::config::{ABTestConfig, RoutingPolicySnapshot, SmartRoutingConfigError};
use super::tier::SmartRoutingTier;
use super::AbRoutingHook;

const ASSIGNMENT_BUCKETS: u64 = 1_000_000;
const Z_95: f64 = 1.959_963_984_540_054;
const SIGNIFICANCE_LEVEL: f64 = 0.05;

pub const DEFAULT_MIN_SAMPLE_SIZE: u64 = 30;
pub const DEFAULT_MAX_SAMPLES_PER_ARM: u64 = 1_000_000_000;
pub const DEFAULT_QUALITY_SUCCESS_THRESHOLD: f64 = 0.5;
pub const MAX_OBSERVED_COST_USD: f64 = 1_000_000.0;
pub const MAX_OBSERVED_LATENCY_MS: f64 = 86_400_000.0;

/// The fixed experiment arm assigned to a request.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize)]
#[serde(rename_all = "lowercase")]
pub enum ExperimentArm {
    Control,
    Variant,
}

/// Statistical-analysis limits applied to one experiment.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct ABTestAnalysisConfig {
    pub min_sample_size: u64,
    pub max_samples_per_arm: u64,
    pub quality_success_threshold: f64,
}

impl Default for ABTestAnalysisConfig {
    fn default() -> Self {
        Self {
            min_sample_size: DEFAULT_MIN_SAMPLE_SIZE,
            max_samples_per_arm: DEFAULT_MAX_SAMPLES_PER_ARM,
            quality_success_threshold: DEFAULT_QUALITY_SUCCESS_THRESHOLD,
        }
    }
}

/// Construction failure for an invalid policy, split, or analysis limit.
#[derive(Debug, Clone, PartialEq)]
pub enum ABTestBuildError {
    InvalidControl(Vec<SmartRoutingConfigError>),
    InvalidVariant(Vec<SmartRoutingConfigError>),
    InvalidVariantPercentage,
    InvalidMinSampleSize,
    InvalidMaxSamplesPerArm,
    InvalidQualitySuccessThreshold,
}

impl std::fmt::Display for ABTestBuildError {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::InvalidControl(_) => formatter.write_str("invalid A/B control policy"),
            Self::InvalidVariant(_) => formatter.write_str("invalid A/B variant policy"),
            Self::InvalidVariantPercentage => {
                formatter.write_str("A/B variant percentage must be finite and in (0, 1]")
            }
            Self::InvalidMinSampleSize => {
                formatter.write_str("A/B minimum sample size must be greater than zero")
            }
            Self::InvalidMaxSamplesPerArm => formatter
                .write_str("A/B maximum samples per arm must cover the minimum sample size"),
            Self::InvalidQualitySuccessThreshold => {
                formatter.write_str("A/B quality success threshold must be finite and in [0, 1]")
            }
        }
    }
}

impl std::error::Error for ABTestBuildError {}

/// A bounded, content-free outcome recorded for one arm.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct ABTestObservation {
    pub tier: SmartRoutingTier,
    pub quality_score: f64,
    pub cost_usd: f64,
    pub latency_ms: f64,
}

impl ABTestObservation {
    fn validate(self) -> Result<Self, ObservationRejection> {
        validate_closed_unit(self.quality_score).map_err(|_| ObservationRejection::Quality)?;
        validate_bounded_non_negative(self.cost_usd, MAX_OBSERVED_COST_USD)
            .map_err(|_| ObservationRejection::Cost)?;
        validate_bounded_non_negative(self.latency_ms, MAX_OBSERVED_LATENCY_MS)
            .map_err(|_| ObservationRejection::Latency)?;
        Ok(self)
    }
}

/// Why an observation did not affect experiment aggregates.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ObservationRejection {
    Quality,
    Cost,
    Latency,
}

/// Result of attempting to record an experiment observation.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ABTestRecordResult {
    Inactive,
    Recorded,
    CapacityReached,
    Rejected(ObservationRejection),
}

/// Fixed-cardinality routing counts for one experiment arm.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq, Serialize)]
pub struct RoutingDistribution {
    pub fast: u64,
    pub balanced: u64,
    pub powerful: u64,
}

impl RoutingDistribution {
    fn increment(&mut self, tier: SmartRoutingTier) {
        let count = match tier {
            SmartRoutingTier::Fast => &mut self.fast,
            SmartRoutingTier::Balanced => &mut self.balanced,
            SmartRoutingTier::Powerful => &mut self.powerful,
        };
        *count = count.saturating_add(1);
    }
}

/// Public aggregate report for one experiment arm.
#[derive(Debug, Clone, Copy, PartialEq, Serialize)]
pub struct ArmStatistics {
    pub n: u64,
    pub routing_distribution: RoutingDistribution,
    pub quality_successes: u64,
    pub average_quality_score: Option<f64>,
    pub average_cost_usd: Option<f64>,
    pub average_latency_ms: Option<f64>,
}

/// A closed 95% confidence interval for the variant-minus-control effect.
#[derive(Debug, Clone, Copy, PartialEq, Serialize)]
pub struct ConfidenceInterval95 {
    pub lower: f64,
    pub upper: f64,
}

/// Two-proportion comparison, emitted only after both arms meet the minimum.
#[derive(Debug, Clone, Copy, PartialEq, Serialize)]
pub struct ABTestComparison {
    pub control_n: u64,
    pub variant_n: u64,
    pub effect: f64,
    pub confidence_interval_95: ConfidenceInterval95,
    pub p_value_two_sided: f64,
    pub significant: bool,
}

/// Content-free experiment results suitable for an administrative response.
#[derive(Debug, Clone, Copy, PartialEq, Serialize)]
pub struct ABTestResults {
    pub active: bool,
    pub variant_percentage: f64,
    pub min_sample_size: u64,
    pub max_samples_per_arm: u64,
    pub quality_success_threshold: f64,
    pub control: ArmStatistics,
    pub variant: ArmStatistics,
    pub comparison: Option<ABTestComparison>,
}

#[derive(Debug, Clone, Copy, Default)]
struct ArmAggregate {
    n: u64,
    routing_distribution: RoutingDistribution,
    quality_successes: u64,
    average_quality_score: f64,
    average_cost_usd: f64,
    average_latency_ms: f64,
}

impl ArmAggregate {
    fn observe(&mut self, observation: ABTestObservation, quality_success_threshold: f64) {
        self.n = self.n.saturating_add(1);
        self.routing_distribution.increment(observation.tier);
        if observation.quality_score >= quality_success_threshold {
            self.quality_successes = self.quality_successes.saturating_add(1);
        }
        update_mean(
            &mut self.average_quality_score,
            observation.quality_score,
            self.n,
        );
        update_mean(&mut self.average_cost_usd, observation.cost_usd, self.n);
        update_mean(&mut self.average_latency_ms, observation.latency_ms, self.n);
    }

    fn snapshot(self) -> ArmStatistics {
        let has_samples = self.n > 0;
        ArmStatistics {
            n: self.n,
            routing_distribution: self.routing_distribution,
            quality_successes: self.quality_successes,
            average_quality_score: has_samples.then_some(self.average_quality_score),
            average_cost_usd: has_samples.then_some(self.average_cost_usd),
            average_latency_ms: has_samples.then_some(self.average_latency_ms),
        }
    }
}

#[derive(Debug, Default)]
struct ExperimentState {
    active: bool,
    control: ArmAggregate,
    variant: ArmAggregate,
}

/// Stable traffic assignment and bounded aggregate analysis for two policies.
#[derive(Debug)]
pub struct ABTestManager {
    control: RoutingPolicySnapshot,
    variant: RoutingPolicySnapshot,
    variant_percentage: f64,
    variant_buckets: u64,
    analysis: ABTestAnalysisConfig,
    state: RwLock<ExperimentState>,
}

impl ABTestManager {
    /// Construct and activate an experiment with default analysis limits.
    pub fn new(config: ABTestConfig) -> Result<Self, ABTestBuildError> {
        Self::with_analysis_config(config, ABTestAnalysisConfig::default())
    }

    /// Construct and activate an experiment with explicit bounded analysis limits.
    pub fn with_analysis_config(
        config: ABTestConfig,
        analysis: ABTestAnalysisConfig,
    ) -> Result<Self, ABTestBuildError> {
        config
            .control
            .validate()
            .map_err(ABTestBuildError::InvalidControl)?;
        config
            .variant
            .validate()
            .map_err(ABTestBuildError::InvalidVariant)?;
        if !config.variant_percentage.is_finite()
            || config.variant_percentage <= 0.0
            || config.variant_percentage > 1.0
        {
            return Err(ABTestBuildError::InvalidVariantPercentage);
        }
        if analysis.min_sample_size == 0 {
            return Err(ABTestBuildError::InvalidMinSampleSize);
        }
        if analysis.max_samples_per_arm < analysis.min_sample_size
            || analysis.max_samples_per_arm > DEFAULT_MAX_SAMPLES_PER_ARM
        {
            return Err(ABTestBuildError::InvalidMaxSamplesPerArm);
        }
        if validate_closed_unit(analysis.quality_success_threshold).is_err() {
            return Err(ABTestBuildError::InvalidQualitySuccessThreshold);
        }

        let variant_buckets = percentage_to_buckets(config.variant_percentage);
        Ok(Self {
            control: config.control,
            variant: config.variant,
            variant_percentage: config.variant_percentage,
            variant_buckets,
            analysis,
            state: RwLock::new(ExperimentState {
                active: true,
                ..ExperimentState::default()
            }),
        })
    }

    /// Assign a request using SHA-256. Inactive experiments always use control.
    pub fn assign(&self, request_id: &str) -> ExperimentArm {
        let state = read_unpoisoned(&self.state);
        self.assign_with_active(request_id, state.active)
    }

    /// Return the assigned immutable policy without retaining either identifier.
    pub fn policy_for_request(&self, request_id: &str) -> RoutingPolicySnapshot {
        let state = read_unpoisoned(&self.state);
        match self.assign_with_active(request_id, state.active) {
            ExperimentArm::Control => self.control.clone(),
            ExperimentArm::Variant => self.variant.clone(),
        }
    }

    /// Record one validated observation while the experiment remains active.
    pub fn record(&self, arm: ExperimentArm, observation: ABTestObservation) -> ABTestRecordResult {
        let observation = match observation.validate() {
            Ok(observation) => observation,
            Err(reason) => return ABTestRecordResult::Rejected(reason),
        };
        let mut state = write_unpoisoned(&self.state);
        if !state.active {
            return ABTestRecordResult::Inactive;
        }
        let aggregate = match arm {
            ExperimentArm::Control => &mut state.control,
            ExperimentArm::Variant => &mut state.variant,
        };
        if aggregate.n >= self.analysis.max_samples_per_arm {
            return ABTestRecordResult::CapacityReached;
        }
        aggregate.observe(observation, self.analysis.quality_success_threshold);
        ABTestRecordResult::Recorded
    }

    /// Stop assignment and atomically make control the policy for every caller.
    ///
    /// Existing aggregates remain available for reporting. This method never
    /// promotes the variant or mutates either policy.
    pub fn stop(&self) -> RoutingPolicySnapshot {
        write_unpoisoned(&self.state).active = false;
        self.control.clone()
    }

    pub fn is_active(&self) -> bool {
        read_unpoisoned(&self.state).active
    }

    /// Snapshot bounded aggregates and, when adequately sampled, significance.
    pub fn results(&self) -> ABTestResults {
        let state = read_unpoisoned(&self.state);
        let control = state.control.snapshot();
        let variant = state.variant.snapshot();
        let comparison = comparison(control, variant, self.analysis.min_sample_size);
        ABTestResults {
            active: state.active,
            variant_percentage: self.variant_percentage,
            min_sample_size: self.analysis.min_sample_size,
            max_samples_per_arm: self.analysis.max_samples_per_arm,
            quality_success_threshold: self.analysis.quality_success_threshold,
            control,
            variant,
            comparison,
        }
    }

    fn assign_with_active(&self, request_id: &str, active: bool) -> ExperimentArm {
        if !active {
            return ExperimentArm::Control;
        }
        let digest = Sha256::digest(request_id.as_bytes());
        let hash_prefix = u64::from_be_bytes(digest[..8].try_into().expect("SHA-256 prefix"));
        if hash_prefix % ASSIGNMENT_BUCKETS < self.variant_buckets {
            ExperimentArm::Variant
        } else {
            ExperimentArm::Control
        }
    }
}

impl AbRoutingHook for ABTestManager {
    fn policy(&self, request_id: &str, _model_group: &str) -> Option<RoutingPolicySnapshot> {
        Some(self.policy_for_request(request_id))
    }
}

fn percentage_to_buckets(percentage: f64) -> u64 {
    if percentage >= 1.0 {
        ASSIGNMENT_BUCKETS
    } else {
        (percentage * ASSIGNMENT_BUCKETS as f64).round() as u64
    }
}

fn update_mean(mean: &mut f64, observation: f64, n: u64) {
    *mean += (observation - *mean) / n as f64;
}

fn comparison(
    control: ArmStatistics,
    variant: ArmStatistics,
    min_sample_size: u64,
) -> Option<ABTestComparison> {
    if control.n < min_sample_size || variant.n < min_sample_size {
        return None;
    }
    let control_rate = control.quality_successes as f64 / control.n as f64;
    let variant_rate = variant.quality_successes as f64 / variant.n as f64;
    let effect = variant_rate - control_rate;
    let unpooled_standard_error = (control_rate * (1.0 - control_rate) / control.n as f64
        + variant_rate * (1.0 - variant_rate) / variant.n as f64)
        .sqrt();
    let confidence_interval_95 = ConfidenceInterval95 {
        lower: (effect - Z_95 * unpooled_standard_error).clamp(-1.0, 1.0),
        upper: (effect + Z_95 * unpooled_standard_error).clamp(-1.0, 1.0),
    };

    let total_n = control.n.saturating_add(variant.n);
    let pooled_rate = control
        .quality_successes
        .saturating_add(variant.quality_successes) as f64
        / total_n as f64;
    let pooled_standard_error =
        (pooled_rate * (1.0 - pooled_rate) * (1.0 / control.n as f64 + 1.0 / variant.n as f64))
            .sqrt();
    let p_value_two_sided = if pooled_standard_error == 0.0 {
        1.0
    } else {
        let z = effect.abs() / pooled_standard_error;
        (2.0 * (1.0 - standard_normal_cdf(z))).clamp(0.0, 1.0)
    };

    Some(ABTestComparison {
        control_n: control.n,
        variant_n: variant.n,
        effect,
        confidence_interval_95,
        p_value_two_sided,
        significant: p_value_two_sided < SIGNIFICANCE_LEVEL,
    })
}

fn standard_normal_cdf(value: f64) -> f64 {
    let x = value.abs();
    let t = 1.0 / (1.0 + 0.231_641_9 * x);
    let density = (-0.5 * x * x).exp() / (2.0 * std::f64::consts::PI).sqrt();
    let polynomial = t
        * (0.319_381_530
            + t * (-0.356_563_782
                + t * (1.781_477_937 + t * (-1.821_255_978 + t * 1.330_274_429))));
    let positive_cdf = 1.0 - density * polynomial;
    if value >= 0.0 {
        positive_cdf
    } else {
        1.0 - positive_cdf
    }
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

fn read_unpoisoned<T>(lock: &RwLock<T>) -> RwLockReadGuard<'_, T> {
    lock.read().unwrap_or_else(|poisoned| poisoned.into_inner())
}

fn write_unpoisoned<T>(lock: &RwLock<T>) -> RwLockWriteGuard<'_, T> {
    lock.write()
        .unwrap_or_else(|poisoned| poisoned.into_inner())
}

#[cfg(test)]
mod tests {
    use std::collections::BTreeSet;
    use std::sync::Arc;
    use std::thread;

    use proptest::prelude::*;
    use serde_json::Value;

    use super::*;

    fn policies(variant_percentage: f64) -> ABTestConfig {
        let mut control = RoutingPolicySnapshot::default();
        control.enabled = true;
        control.cost_quality_threshold = 0.2;
        let mut variant = control.clone();
        variant.cost_quality_threshold = 0.8;
        ABTestConfig {
            control,
            variant,
            variant_percentage,
        }
    }

    fn observation(quality_score: f64) -> ABTestObservation {
        observation_with(quality_score, SmartRoutingTier::Balanced, 0.02, 120.0)
    }

    fn observation_with(
        quality_score: f64,
        tier: SmartRoutingTier,
        cost_usd: f64,
        latency_ms: f64,
    ) -> ABTestObservation {
        ABTestObservation {
            tier,
            quality_score,
            cost_usd,
            latency_ms,
        }
    }

    fn assert_close(actual: f64, expected: f64) {
        assert!(
            (actual - expected).abs() < 1.0e-12,
            "expected {expected}, got {actual}"
        );
    }

    fn deterministic_request_ids(count: usize) -> Vec<String> {
        (0..count)
            .map(|index| format!("property-34-request-{index:04}"))
            .collect()
    }

    // Feature: smart-routing, Property 34: A/B assignment is stable across manager instances.
    #[test]
    fn property_34_stable_assignment_across_manager_instances_and_restarts_has_256_cases() {
        let request_ids = deterministic_request_ids(256);
        let original = ABTestManager::new(policies(0.37)).unwrap();
        let assignments: Vec<_> = request_ids
            .iter()
            .map(|request_id| original.assign(request_id))
            .collect();

        for _restart in 0..3 {
            let restarted = ABTestManager::new(policies(0.37)).unwrap();
            for (request_id, expected) in request_ids.iter().zip(&assignments) {
                assert_eq!(restarted.assign(request_id), *expected, "{request_id}");
                assert_eq!(restarted.assign(request_id), *expected, "{request_id}");
            }
        }
    }

    proptest! {
    #![proptest_config(ProptestConfig::with_cases(64))]

    // Feature: smart-routing, Property 34: A/B assignment is stable across manager instances.
    #[test]
    fn property_34_stable_assignment_for_arbitrary_request_ids(
    request_id in prop::collection::vec(any::<char>(), 0..=256)
    .prop_map(|characters| characters.into_iter().collect::<String>()),
    variant_percentage in 0.000_001f64..=1.0,
    ) {
    let first = ABTestManager::new(policies(variant_percentage)).unwrap();
    let expected = first.assign(&request_id);
    let restarted = ABTestManager::new(policies(variant_percentage)).unwrap();

    prop_assert_eq!(first.assign(&request_id), expected);
    prop_assert_eq!(restarted.assign(&request_id), expected);
    }
    }

    #[test]
    fn deterministic_split_tracks_configured_percentage_within_tolerance() {
        const SAMPLE_SIZE: usize = 100_000;
        for percentage in [0.1, 0.25, 0.5, 0.75, 0.9] {
            let manager = ABTestManager::new(policies(percentage)).unwrap();
            let variant_count = deterministic_request_ids(SAMPLE_SIZE)
                .iter()
                .filter(|request_id| manager.assign(request_id) == ExperimentArm::Variant)
                .count();
            let observed = variant_count as f64 / SAMPLE_SIZE as f64;
            let standard_deviation = (percentage * (1.0 - percentage) / SAMPLE_SIZE as f64).sqrt();
            let tolerance = 5.0 * standard_deviation + 1.0 / SAMPLE_SIZE as f64;
            assert!(
                (observed - percentage).abs() <= tolerance,
                "configured={percentage}, observed={observed}, tolerance={tolerance}"
            );
        }
    }

    #[test]
    fn inactive_experiment_routes_and_records_only_control_behavior() {
        let manager = ABTestManager::new(policies(1.0)).unwrap();
        assert_eq!(manager.assign("before-stop"), ExperimentArm::Variant);
        manager.stop();

        for request_id in deterministic_request_ids(256) {
            assert_eq!(manager.assign(&request_id), ExperimentArm::Control);
            assert_eq!(
                manager
                    .policy_for_request(&request_id)
                    .cost_quality_threshold,
                0.2
            );
        }
        assert_eq!(
            manager.record(ExperimentArm::Control, observation(1.0)),
            ABTestRecordResult::Inactive
        );
        let results = manager.results();
        assert!(!results.active);
        assert_eq!(results.control.n, 0);
        assert_eq!(results.variant.n, 0);
    }

    #[test]
    fn aggregate_results_expose_only_bounded_fixed_labels() {
        let analysis = ABTestAnalysisConfig {
            min_sample_size: 1,
            max_samples_per_arm: 10,
            quality_success_threshold: 0.5,
        };
        let manager = ABTestManager::with_analysis_config(policies(0.5), analysis).unwrap();
        for (tier, quality, cost, latency) in [
            (SmartRoutingTier::Fast, 0.25, 0.01, 10.0),
            (SmartRoutingTier::Balanced, 0.5, 0.02, 20.0),
            (SmartRoutingTier::Powerful, 1.0, 0.03, 30.0),
        ] {
            assert_eq!(
                manager.record(
                    ExperimentArm::Control,
                    observation_with(quality, tier, cost, latency),
                ),
                ABTestRecordResult::Recorded
            );
        }

        let results = manager.results();
        assert_eq!(results.control.n, 3);
        assert_eq!(results.control.quality_successes, 2);
        assert_eq!(
            results.control.routing_distribution,
            RoutingDistribution {
                fast: 1,
                balanced: 1,
                powerful: 1,
            }
        );
        assert_close(
            results.control.average_quality_score.unwrap(),
            0.583_333_333_333_333_4,
        );
        assert_close(results.control.average_cost_usd.unwrap(), 0.02);
        assert_close(results.control.average_latency_ms.unwrap(), 20.0);

        let serialized = serde_json::to_value(results).unwrap();
        let Value::Object(top_level) = serialized else {
            panic!("A/B results must serialize as an object");
        };
        let expected_top_level = BTreeSet::from([
            "active",
            "comparison",
            "control",
            "max_samples_per_arm",
            "min_sample_size",
            "quality_success_threshold",
            "variant",
            "variant_percentage",
        ]);
        assert_eq!(
            top_level
                .keys()
                .map(String::as_str)
                .collect::<BTreeSet<_>>(),
            expected_top_level
        );
        let control = top_level["control"].as_object().unwrap();
        let expected_arm_labels = BTreeSet::from([
            "average_cost_usd",
            "average_latency_ms",
            "average_quality_score",
            "n",
            "quality_successes",
            "routing_distribution",
        ]);
        assert_eq!(
            control.keys().map(String::as_str).collect::<BTreeSet<_>>(),
            expected_arm_labels
        );
        let routing = control["routing_distribution"].as_object().unwrap();
        assert_eq!(
            routing.keys().map(String::as_str).collect::<BTreeSet<_>>(),
            BTreeSet::from(["balanced", "fast", "powerful"]),
        );
    }

    #[test]
    fn comparison_requires_minimum_samples_in_each_arm() {
        let analysis = ABTestAnalysisConfig {
            min_sample_size: 4,
            max_samples_per_arm: 10,
            quality_success_threshold: 0.5,
        };
        let manager = ABTestManager::with_analysis_config(policies(0.5), analysis).unwrap();
        for _ in 0..4 {
            manager.record(ExperimentArm::Control, observation(1.0));
        }
        for _ in 0..3 {
            manager.record(ExperimentArm::Variant, observation(0.0));
        }
        assert!(manager.results().comparison.is_none());

        manager.record(ExperimentArm::Variant, observation(0.0));
        assert!(manager.results().comparison.is_some());
    }

    #[test]
    fn result_calculations_match_known_rates_intervals_and_significance() {
        let analysis = ABTestAnalysisConfig {
            min_sample_size: 100,
            max_samples_per_arm: 100,
            quality_success_threshold: 0.5,
        };
        let manager = ABTestManager::with_analysis_config(policies(0.5), analysis).unwrap();
        for index in 0..100 {
            manager.record(
                ExperimentArm::Control,
                observation_with(
                    if index < 40 { 1.0 } else { 0.0 },
                    SmartRoutingTier::Fast,
                    index as f64 / 100.0,
                    index as f64,
                ),
            );
            manager.record(
                ExperimentArm::Variant,
                observation_with(
                    if index < 60 { 1.0 } else { 0.0 },
                    SmartRoutingTier::Powerful,
                    1.0 + index as f64 / 100.0,
                    100.0 + index as f64,
                ),
            );
        }

        let results = manager.results();
        assert_eq!(results.control.quality_successes, 40);
        assert_eq!(results.variant.quality_successes, 60);
        assert_close(results.control.average_cost_usd.unwrap(), 0.495);
        assert_close(results.variant.average_cost_usd.unwrap(), 1.495);
        assert_close(results.control.average_latency_ms.unwrap(), 49.5);
        assert_close(results.variant.average_latency_ms.unwrap(), 149.5);
        let comparison = results.comparison.unwrap();
        assert_eq!(comparison.control_n, 100);
        assert_eq!(comparison.variant_n, 100);
        assert_close(comparison.effect, 0.2);
        assert_close(
            comparison.confidence_interval_95.lower,
            0.064_209_711_910_859_34,
        );
        assert_close(
            comparison.confidence_interval_95.upper,
            0.335_790_288_089_140_7,
        );
        assert_close(comparison.p_value_two_sided, 0.004_677_860_217_238_15);
        assert!(comparison.significant);
    }

    #[test]
    fn sha256_assignment_is_stable_and_split_selects_both_arms() {
        let manager = ABTestManager::new(policies(0.5)).unwrap();
        let first = manager.assign("stable-request-id");
        assert_eq!(manager.assign("stable-request-id"), first);

        let arms: Vec<_> = (0..256)
            .map(|index| manager.assign(&format!("request-{index}")))
            .collect();
        assert!(arms.contains(&ExperimentArm::Control));
        assert!(arms.contains(&ExperimentArm::Variant));
    }

    #[test]
    fn stop_returns_and_permanently_routes_to_control() {
        let manager = ABTestManager::new(policies(1.0)).unwrap();
        assert_eq!(manager.assign("request"), ExperimentArm::Variant);
        let control = manager.stop();
        assert_eq!(control.cost_quality_threshold, 0.2);
        assert_eq!(manager.assign("request"), ExperimentArm::Control);
        assert_eq!(
            AbRoutingHook::policy(&manager, "request", "ignored-group")
                .unwrap()
                .cost_quality_threshold,
            0.2
        );
        assert_eq!(
            manager.record(ExperimentArm::Control, observation(1.0)),
            ABTestRecordResult::Inactive
        );
    }

    #[test]
    fn results_gate_significance_until_both_arms_meet_minimum() {
        let analysis = ABTestAnalysisConfig {
            min_sample_size: 20,
            max_samples_per_arm: 100,
            quality_success_threshold: 0.5,
        };
        let manager = ABTestManager::with_analysis_config(policies(0.5), analysis).unwrap();
        for index in 0..20 {
            let control_quality = if index < 5 { 1.0 } else { 0.0 };
            let variant_quality = if index < 15 { 1.0 } else { 0.0 };
            assert_eq!(
                manager.record(ExperimentArm::Control, observation(control_quality)),
                ABTestRecordResult::Recorded
            );
            if index < 19 {
                manager.record(ExperimentArm::Variant, observation(variant_quality));
            }
        }
        assert!(manager.results().comparison.is_none());
        manager.record(ExperimentArm::Variant, observation(0.0));

        let results = manager.results();
        assert_eq!(results.control.n, 20);
        assert_eq!(results.variant.n, 20);
        assert_eq!(results.control.routing_distribution.balanced, 20);
        let comparison = results.comparison.unwrap();
        assert!((comparison.effect - 0.5).abs() < 1.0e-12);
        assert!(comparison.p_value_two_sided < 0.05);
        assert!(comparison.significant);
    }

    #[test]
    fn recording_is_bounded_and_concurrency_safe() {
        let analysis = ABTestAnalysisConfig {
            min_sample_size: 1,
            max_samples_per_arm: 200,
            quality_success_threshold: 0.5,
        };
        let manager =
            Arc::new(ABTestManager::with_analysis_config(policies(0.5), analysis).unwrap());
        let workers: Vec<_> = (0..8)
            .map(|_| {
                let manager = Arc::clone(&manager);
                thread::spawn(move || {
                    for _ in 0..100 {
                        manager.record(ExperimentArm::Control, observation(0.75));
                    }
                })
            })
            .collect();
        for worker in workers {
            worker.join().unwrap();
        }

        let control = manager.results().control;
        assert_eq!(control.n, 200);
        assert_eq!(control.routing_distribution.balanced, 200);
        assert_eq!(control.average_quality_score, Some(0.75));
        assert_eq!(
            manager.record(ExperimentArm::Control, observation(0.75)),
            ABTestRecordResult::CapacityReached
        );
    }

    #[test]
    fn non_finite_or_out_of_range_observations_are_rejected() {
        let manager = ABTestManager::new(policies(0.5)).unwrap();
        let mut invalid = observation(0.5);
        invalid.cost_usd = f64::INFINITY;
        assert_eq!(
            manager.record(ExperimentArm::Control, invalid),
            ABTestRecordResult::Rejected(ObservationRejection::Cost)
        );
        assert_eq!(manager.results().control.n, 0);
    }
}
