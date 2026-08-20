//! Production orchestration for complexity-aware model routing.

pub mod ab_test;
pub mod budget_controller;
pub mod cascade;
pub mod config;
pub mod context_filter;
pub mod decision_engine;
pub mod evaluation;
pub mod heuristic;
pub mod llm_classifier;
#[cfg(feature = "ml-router")]
pub mod ml_classifier;
pub mod online_optimizer;
pub mod quality_evaluator;
pub mod semantic_cache;
pub mod tier;
#[cfg(feature = "ml-router")]
pub mod training;

use std::fmt;
use std::sync::Arc;

use async_trait::async_trait;

use crate::config::{ModelGroup, ProviderModel};
use crate::models::openai::OpenAIRequest;

use self::cascade::CascadeEvaluator;
use self::config::{
    BudgetLimits, ClassifierMode, CompositeWeights, RoutingPolicySnapshot, SmartRoutingConfig,
    SmartRoutingConfigError,
};
use self::context_filter::{
    filter_by_context_capacity, ContextFilterResult, ContextRequirement, NoSafeCandidate,
};
use self::decision_engine::DecisionEngine;
use self::heuristic::{HeuristicAssessment, HeuristicScorer};
use self::tier::{ClassifierUsed, ComplexityScore, RoutingDecision, SmartRoutingTier, TaskType};

const DEFAULT_CLASSIFICATION_SCORE: f64 = 0.5;

/// Immutable request inputs needed by smart-routing planning.
pub struct SmartRoutingInput<'a> {
    pub request_id: &'a str,
    pub request: &'a OpenAIRequest,
    pub model_group: &'a ModelGroup,
    pub pinned_context: &'a PinnedRoutingContext,
}

/// Routing and token-capacity facts established before smart routing begins.
#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub struct PinnedRoutingContext {
    pub provider: Option<String>,
    pub model: Option<String>,
    pub additional_input_tokens: u64,
    pub reserved_output_tokens: u64,
    pub provider_overhead_tokens: u64,
    pub safety_margin_tokens: u64,
    pub allow_unknown_context_window: bool,
}

/// Successful classification, including context-planning metadata.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Classification {
    pub score: ComplexityScore,
    pub task_type: TaskType,
    pub classifier: ClassifierUsed,
    pub token_estimate: u64,
}

/// Validated output returned by an optional classifier implementation.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct ClassifierOutput {
    pub score: f64,
}

/// Bounded failure categories; classifier implementations cannot attach prompts or responses.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ClassifierFailure {
    Unavailable,
    Timeout,
    InvalidOutput,
    Backend,
}

/// Input available to an optional ML or LLM classifier.
pub struct ClassifierInput<'a> {
    pub request: &'a OpenAIRequest,
    pub model_group: &'a ModelGroup,
    pub pinned_context: &'a PinnedRoutingContext,
    pub heuristic_score: ComplexityScore,
    pub heuristic_task_type: TaskType,
}

/// Object-safe boundary for future ML and LLM classifier implementations.
#[async_trait]
pub trait OptionalClassifier: Send + Sync {
    async fn classify(
        &self,
        input: ClassifierInput<'_>,
    ) -> Result<ClassifierOutput, ClassifierFailure>;
}

/// Result of a budget policy check performed before tier selection.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum BudgetDecision {
    Allow,
    Downgrade { maximum_tier: SmartRoutingTier },
    Reject { reason: BudgetRejectionReason },
}

/// Content-free reason a caller can map to its public rejection response.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum BudgetRejectionReason {
    HourlyLimit,
    DailyLimit,
    MonthlyLimit,
    Policy,
}

/// Typed budget rejection intentionally left for the parent caller to map.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct BudgetRejection {
    pub reason: BudgetRejectionReason,
}

/// Inputs available to a future budget accounting implementation.
pub struct BudgetCheckInput<'a> {
    pub request: &'a OpenAIRequest,
    pub model_group: &'a ModelGroup,
    pub pinned_context: &'a PinnedRoutingContext,
    pub classification: Classification,
    pub configured_limits: Option<&'a BudgetLimits>,
}

#[async_trait]
pub trait BudgetPolicy: Send + Sync {
    async fn check(&self, input: BudgetCheckInput<'_>) -> BudgetDecision;
}

/// Opaque, content-free metadata for a safe semantic-cache hit.
#[derive(Debug, Clone, PartialEq)]
pub struct SemanticCacheHit {
    pub entry_id: u64,
    pub similarity: ComplexityScore,
    pub quality_score: ComplexityScore,
    pub decision: RoutingDecision,
}

pub struct SemanticCacheLookup<'a> {
    pub request: &'a OpenAIRequest,
    pub model_group: &'a ModelGroup,
    pub pinned_context: &'a PinnedRoutingContext,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum SemanticCacheFailure {
    Unavailable,
    Backend,
}

/// Safe cache boundary. Implementations return metadata, never plaintext cache content.
#[async_trait]
pub trait SemanticRoutingCache: Send + Sync {
    async fn lookup(
        &self,
        input: SemanticCacheLookup<'_>,
    ) -> Result<Option<SemanticCacheHit>, SemanticCacheFailure>;
}

/// Content-free classifier fallback telemetry.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct ClassifierFallbackEvent {
    pub configured: ClassifierMode,
    pub reason: ClassifierFailure,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum CacheLookupEvent {
    Hit,
    Miss,
    Error,
}

/// Narrow metrics surface that cannot receive request or response content.
pub trait SmartRoutingMetrics: Send + Sync {
    fn classifier_fallback(&self, _event: ClassifierFallbackEvent) {}
    fn cache_lookup(&self, _event: CacheLookupEvent) {}
    fn budget_decision(&self, _decision: BudgetDecision) {}
}

#[derive(Debug, Default)]
struct NoopSmartRoutingMetrics;

impl SmartRoutingMetrics for NoopSmartRoutingMetrics {}

/// Optional post-plan observation point for future quality evaluators.
pub trait QualityEvaluatorHook: Send + Sync {
    fn observe_plan(&self, decision: &RoutingDecision);
}

/// Optional online optimizer input applied to the cost/quality threshold.
pub trait RoutingOptimizerHook: Send + Sync {
    fn cost_quality_threshold(
        &self,
        model_group: &str,
        task_type: TaskType,
        configured: f64,
    ) -> f64;
}

/// Optional A/B assignment hook. It selects only from already validated policies.
pub trait AbRoutingHook: Send + Sync {
    fn policy(&self, request_id: &str, model_group: &str) -> Option<RoutingPolicySnapshot>;
}

/// Initial candidate plan. Full adjacent-tier filtering belongs to Task 5.2.
#[derive(Debug, Clone, PartialEq)]
pub struct CandidatePlan {
    pub decision: RoutingDecision,
    pub candidates: Vec<ProviderModel>,
    pub excluded_for_context: usize,
    pub estimated_context_tokens: u64,
    pub bypassed: bool,
}

/// A cache hit short-circuits routing; a budget rejection is mapped by the caller.
#[derive(Debug, Clone, PartialEq)]
pub enum RoutingPlanOutcome {
    CacheHit(SemanticCacheHit),
    Route(CandidatePlan),
    BudgetRejected(BudgetRejection),
}

/// Planning errors that are safe to map at the HTTP boundary.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum RoutingPlanningError {
    DisabledForModelGroup,
    NoCandidates,
    ContextCapacity(ContextPlanningError),
}

/// Typed oversized-request error that maps to HTTP 413.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct ContextPlanningError {
    pub excluded_count: usize,
    pub largest_known_context: Option<u32>,
    pub estimated_requirement: u64,
}

impl ContextPlanningError {
    pub const fn status_code(self) -> u16 {
        413
    }
}

impl From<NoSafeCandidate> for ContextPlanningError {
    fn from(value: NoSafeCandidate) -> Self {
        Self {
            excluded_count: value.excluded_count,
            largest_known_context: value.largest_known_context,
            estimated_requirement: value.estimated_requirement,
        }
    }
}

/// Construction failure. Disabled routing is intentionally parent-owned.
#[derive(Debug, Clone, PartialEq)]
pub enum SmartRouterBuildError {
    Disabled,
    InvalidConfig(Vec<SmartRoutingConfigError>),
}

impl fmt::Display for SmartRouterBuildError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Disabled => formatter.write_str("smart routing is disabled"),
            Self::InvalidConfig(errors) => {
                write!(
                    formatter,
                    "invalid smart-routing config ({} errors)",
                    errors.len()
                )
            }
        }
    }
}

impl std::error::Error for SmartRouterBuildError {}

/// Smart-routing orchestrator. Optional components are injected, never fabricated.
pub struct SmartRouter {
    config: SmartRoutingConfig,
    heuristic: HeuristicScorer,
    ml: Option<Arc<dyn OptionalClassifier>>,
    llm: Option<Arc<dyn OptionalClassifier>>,
    decision_engine: DecisionEngine,
    cascade_evaluator: CascadeEvaluator,
    quality_evaluator: Option<Arc<dyn QualityEvaluatorHook>>,
    optimizer: Option<Arc<dyn RoutingOptimizerHook>>,
    budget: Option<Arc<dyn BudgetPolicy>>,
    cache: Option<Arc<dyn SemanticRoutingCache>>,
    ab_test: Option<Arc<dyn AbRoutingHook>>,
    metrics: Arc<dyn SmartRoutingMetrics>,
}

impl SmartRouter {
    /// Validate configuration without loading models, providers, caches, or budgets.
    pub fn new(config: SmartRoutingConfig) -> Result<Self, SmartRouterBuildError> {
        if !config.enabled {
            return Err(SmartRouterBuildError::Disabled);
        }
        config
            .validate()
            .map_err(SmartRouterBuildError::InvalidConfig)?;

        Ok(Self {
            heuristic: HeuristicScorer::new(config.heuristic_weights.clone()),
            decision_engine: DecisionEngine,
            cascade_evaluator: CascadeEvaluator,
            config,
            ml: None,
            llm: None,
            quality_evaluator: None,
            optimizer: None,
            budget: None,
            cache: None,
            ab_test: None,
            metrics: Arc::new(NoopSmartRoutingMetrics),
        })
    }

    pub fn cascade_evaluator(&self) -> &CascadeEvaluator {
        &self.cascade_evaluator
    }

    pub fn cascade_config(&self) -> &config::CascadeConfig {
        &self.config.cascade
    }

    pub fn with_ml_classifier(mut self, classifier: Arc<dyn OptionalClassifier>) -> Self {
        self.ml = Some(classifier);
        self
    }

    pub fn with_llm_classifier(mut self, classifier: Arc<dyn OptionalClassifier>) -> Self {
        self.llm = Some(classifier);
        self
    }

    pub fn with_budget(mut self, budget: Arc<dyn BudgetPolicy>) -> Self {
        self.budget = Some(budget);
        self
    }

    pub fn with_cache(mut self, cache: Arc<dyn SemanticRoutingCache>) -> Self {
        self.cache = Some(cache);
        self
    }

    pub fn with_metrics(mut self, metrics: Arc<dyn SmartRoutingMetrics>) -> Self {
        self.metrics = metrics;
        self
    }

    pub fn with_quality_evaluator(mut self, evaluator: Arc<dyn QualityEvaluatorHook>) -> Self {
        self.quality_evaluator = Some(evaluator);
        self
    }

    pub fn with_optimizer(mut self, optimizer: Arc<dyn RoutingOptimizerHook>) -> Self {
        self.optimizer = Some(optimizer);
        self
    }

    pub fn with_ab_test(mut self, ab_test: Arc<dyn AbRoutingHook>) -> Self {
        self.ab_test = Some(ab_test);
        self
    }

    /// Classify without surfacing optional-backend failures.
    pub async fn classify(&self, input: &SmartRoutingInput<'_>) -> Classification {
        let policy = self.policy_for(input);
        self.classify_with_policy(input, &policy).await
    }

    /// Run cache, classification, budget, context, and tier planning in order.
    pub async fn plan(
        &self,
        input: &SmartRoutingInput<'_>,
    ) -> Result<RoutingPlanOutcome, RoutingPlanningError> {
        let policy = self.policy_for(input);
        if !policy.enabled {
            return Err(RoutingPlanningError::DisabledForModelGroup);
        }

        if policy.semantic_cache.enabled && input.pinned_context.model.is_none() {
            if let Some(cache) = &self.cache {
                match cache
                    .lookup(SemanticCacheLookup {
                        request: input.request,
                        model_group: input.model_group,
                        pinned_context: input.pinned_context,
                    })
                    .await
                {
                    Ok(Some(mut hit))
                        if hit.similarity.value() >= policy.semantic_cache.similarity_threshold
                            && hit.quality_score.value()
                                >= policy.semantic_cache.min_quality_score =>
                    {
                        hit.decision.cache_hit = true;
                        self.metrics.cache_lookup(CacheLookupEvent::Hit);
                        return Ok(RoutingPlanOutcome::CacheHit(hit));
                    }
                    Ok(_) => self.metrics.cache_lookup(CacheLookupEvent::Miss),
                    Err(_) => self.metrics.cache_lookup(CacheLookupEvent::Error),
                }
            }
        }

        let classification = self.classify_with_policy(input, &policy).await;
        let budget_decision = if let Some(budget) = &self.budget {
            budget
                .check(BudgetCheckInput {
                    request: input.request,
                    model_group: input.model_group,
                    pinned_context: input.pinned_context,
                    classification,
                    configured_limits: self.config.budget_limits.get(&input.model_group.name),
                })
                .await
        } else {
            BudgetDecision::Allow
        };
        self.metrics.budget_decision(budget_decision);

        if let BudgetDecision::Reject { reason } = budget_decision {
            return Ok(RoutingPlanOutcome::BudgetRejected(BudgetRejection {
                reason,
            }));
        }

        let requirement = ContextRequirement {
            input_tokens: classification
                .token_estimate
                .saturating_add(input.pinned_context.additional_input_tokens),
            reserved_output_tokens: u64::from(
                input
                    .request
                    .max_tokens
                    .unwrap_or(policy.reserved_output_tokens),
            )
            .max(input.pinned_context.reserved_output_tokens),
            provider_overhead_tokens: input
                .pinned_context
                .provider_overhead_tokens
                .max(u64::from(policy.provider_overhead_tokens)),
            safety_margin_tokens: input
                .pinned_context
                .safety_margin_tokens
                .max(u64::from(policy.context_safety_margin_tokens)),
        };
        let eligible = match filter_by_context_capacity(
            &input.model_group.models,
            requirement,
            input.pinned_context.allow_unknown_context_window
                || policy.allow_unknown_context_window,
        ) {
            ContextFilterResult::Eligible(eligible) => eligible,
            ContextFilterResult::NoCandidates { .. } => {
                return Err(RoutingPlanningError::NoCandidates)
            }
            ContextFilterResult::NoSafeCandidate(error) => {
                return Err(RoutingPlanningError::ContextCapacity(error.into()))
            }
        };

        let configured_threshold = policy.cost_quality_threshold;
        let threshold = self
            .optimizer
            .as_ref()
            .map_or(configured_threshold, |optimizer| {
                optimizer.cost_quality_threshold(
                    &input.model_group.name,
                    classification.task_type,
                    configured_threshold,
                )
            });
        let selected =
            self.decision_engine
                .decide(classification.score, &policy.tier_boundaries, threshold);
        let (tier, budget_downgraded) = apply_budget_limit(selected.tier, budget_decision);
        let excluded_for_context = eligible.excluded_count;
        let estimated_context_tokens = eligible.estimated_requirement;
        let mut context_safe_group = input.model_group.clone();
        context_safe_group.models = eligible.models;
        let tier_filter = self.filter_by_tier(
            &context_safe_group,
            tier,
            classification.task_type,
            input.pinned_context.model.as_deref(),
        );
        let tier = tier_filter.applied_tier.unwrap_or(tier);
        let mut candidates = tier_filter.model_group.models;
        if tier_filter.bypassed && input.pinned_context.model.is_some() {
            prefer_candidates(
                &mut candidates,
                classification.task_type,
                input.pinned_context,
            );
        }

        let decision = RoutingDecision {
            score: selected.score,
            adjusted_score: selected.adjusted_score,
            tier,
            task_type: classification.task_type,
            classifier: classification.classifier,
            escalated: false,
            escalation_count: 0,
            cache_hit: false,
            budget_downgraded,
            context_filtered: excluded_for_context > 0,
        };
        if let Some(evaluator) = &self.quality_evaluator {
            evaluator.observe_plan(&decision);
        }

        Ok(RoutingPlanOutcome::Route(CandidatePlan {
            decision,
            candidates,
            excluded_for_context,
            estimated_context_tokens,
            bypassed: tier_filter.bypassed,
        }))
    }

    fn policy_for(&self, input: &SmartRoutingInput<'_>) -> SmartRoutingConfig {
        let mut policy = self.config.effective_for_group(&input.model_group.name);
        if let Some(ab_policy) = self
            .ab_test
            .as_ref()
            .and_then(|ab_test| ab_test.policy(input.request_id, &input.model_group.name))
        {
            apply_policy_snapshot(&mut policy, ab_policy);
        }
        policy
    }

    async fn classify_with_policy(
        &self,
        input: &SmartRoutingInput<'_>,
        policy: &SmartRoutingConfig,
    ) -> Classification {
        let heuristic = self.heuristic_assessment(input, policy);
        let fallback = || match heuristic {
            Ok(assessment) => Classification {
                score: assessment.score,
                task_type: assessment.task_type,
                classifier: ClassifierUsed::Heuristic,
                token_estimate: assessment.token_estimate() as u64,
            },
            Err(reason) => {
                self.metrics.classifier_fallback(ClassifierFallbackEvent {
                    configured: policy.classifier,
                    reason,
                });
                Classification {
                    score: ComplexityScore::new(DEFAULT_CLASSIFICATION_SCORE),
                    task_type: TaskType::General,
                    classifier: ClassifierUsed::Heuristic,
                    token_estimate: 0,
                }
            }
        };
        let heuristic_classification = fallback();

        match policy.classifier {
            ClassifierMode::Heuristic => heuristic_classification,
            ClassifierMode::Ml => self
                .classify_optional(
                    self.ml.as_deref(),
                    ClassifierUsed::Ml,
                    input,
                    heuristic_classification,
                    policy.classifier,
                )
                .await
                .unwrap_or(heuristic_classification),
            ClassifierMode::Llm => self
                .classify_optional(
                    self.llm.as_deref(),
                    ClassifierUsed::Llm,
                    input,
                    heuristic_classification,
                    policy.classifier,
                )
                .await
                .unwrap_or(heuristic_classification),
            ClassifierMode::Composite => {
                let Some(ml) = self
                    .classify_optional(
                        self.ml.as_deref(),
                        ClassifierUsed::Ml,
                        input,
                        heuristic_classification,
                        policy.classifier,
                    )
                    .await
                else {
                    return heuristic_classification;
                };
                let weights = policy.composite_weights.clone().unwrap_or_default();
                Classification {
                    score: composite_score(heuristic_classification.score, ml.score, &weights),
                    task_type: heuristic_classification.task_type,
                    classifier: ClassifierUsed::Composite,
                    token_estimate: heuristic_classification.token_estimate,
                }
            }
        }
    }

    fn heuristic_assessment(
        &self,
        input: &SmartRoutingInput<'_>,
        policy: &SmartRoutingConfig,
    ) -> Result<HeuristicAssessment, ClassifierFailure> {
        let assessment = if policy.heuristic_weights == self.config.heuristic_weights {
            self.heuristic.score(&input.request.messages)
        } else {
            HeuristicScorer::new(policy.heuristic_weights.clone()).score(&input.request.messages)
        };
        Ok(assessment)
    }

    async fn classify_optional(
        &self,
        classifier: Option<&dyn OptionalClassifier>,
        classifier_used: ClassifierUsed,
        input: &SmartRoutingInput<'_>,
        heuristic: Classification,
        configured: ClassifierMode,
    ) -> Option<Classification> {
        let Some(classifier) = classifier else {
            self.metrics.classifier_fallback(ClassifierFallbackEvent {
                configured,
                reason: ClassifierFailure::Unavailable,
            });
            return None;
        };
        let output = match classifier
            .classify(ClassifierInput {
                request: input.request,
                model_group: input.model_group,
                pinned_context: input.pinned_context,
                heuristic_score: heuristic.score,
                heuristic_task_type: heuristic.task_type,
            })
            .await
        {
            Ok(output) if output.score.is_finite() && (0.0..=1.0).contains(&output.score) => output,
            Ok(_) => {
                self.metrics.classifier_fallback(ClassifierFallbackEvent {
                    configured,
                    reason: ClassifierFailure::InvalidOutput,
                });
                return None;
            }
            Err(reason) => {
                self.metrics
                    .classifier_fallback(ClassifierFallbackEvent { configured, reason });
                return None;
            }
        };

        Some(Classification {
            score: ComplexityScore::new(output.score),
            task_type: heuristic.task_type,
            classifier: classifier_used,
            token_estimate: heuristic.token_estimate,
        })
    }
}

/// Result of tier filtering, including whether Smart Routing was bypassed.
#[derive(Debug, Clone, PartialEq)]
pub struct TierFilterResult {
    pub model_group: ModelGroup,
    pub applied_tier: Option<SmartRoutingTier>,
    pub bypassed: bool,
}

impl SmartRouter {
    /// Apply tier selection, adjacent-tier fallback, and in-tier specialization.
    ///
    /// Pinned requests and groups without any tier tags bypass capability routing.
    /// Context safety must already have been applied to `model_group`.
    pub fn filter_by_tier(
        &self,
        model_group: &ModelGroup,
        selected_tier: SmartRoutingTier,
        task_type: TaskType,
        pinned_model: Option<&str>,
    ) -> TierFilterResult {
        if pinned_model.is_some()
            || model_group
                .models
                .iter()
                .all(|candidate| candidate.tier.is_none())
        {
            return TierFilterResult {
                model_group: model_group.clone(),
                applied_tier: None,
                bypassed: true,
            };
        }

        let search_order = tier_fallback_order(selected_tier);
        let applied_tier = search_order.into_iter().find(|tier| {
            model_group
                .models
                .iter()
                .any(|candidate| candidate.tier == Some(*tier))
        });
        let Some(applied_tier) = applied_tier else {
            return TierFilterResult {
                model_group: model_group.clone(),
                applied_tier: None,
                bypassed: true,
            };
        };

        let mut models = model_group
            .models
            .iter()
            .filter(|candidate| candidate.tier == Some(applied_tier))
            .cloned()
            .collect::<Vec<_>>();
        if models
            .iter()
            .any(|candidate| candidate.specializations.contains(&task_type))
        {
            models.retain(|candidate| candidate.specializations.contains(&task_type));
        }

        let mut filtered = model_group.clone();
        filtered.models = models;
        TierFilterResult {
            model_group: filtered,
            applied_tier: Some(applied_tier),
            bypassed: false,
        }
    }
}

fn tier_fallback_order(selected: SmartRoutingTier) -> [SmartRoutingTier; 3] {
    match selected {
        SmartRoutingTier::Fast => [
            SmartRoutingTier::Fast,
            SmartRoutingTier::Balanced,
            SmartRoutingTier::Powerful,
        ],
        SmartRoutingTier::Balanced => [
            SmartRoutingTier::Balanced,
            SmartRoutingTier::Powerful,
            SmartRoutingTier::Fast,
        ],
        SmartRoutingTier::Powerful => [
            SmartRoutingTier::Powerful,
            SmartRoutingTier::Balanced,
            SmartRoutingTier::Fast,
        ],
    }
}

fn composite_score(
    heuristic: ComplexityScore,
    ml: ComplexityScore,
    weights: &CompositeWeights,
) -> ComplexityScore {
    ComplexityScore::new(heuristic.value() * weights.heuristic + ml.value() * weights.ml)
}

fn apply_budget_limit(
    selected: SmartRoutingTier,
    budget: BudgetDecision,
) -> (SmartRoutingTier, bool) {
    let BudgetDecision::Downgrade { maximum_tier } = budget else {
        return (selected, false);
    };
    let one_step_lower = match selected {
        SmartRoutingTier::Powerful => SmartRoutingTier::Balanced,
        SmartRoutingTier::Balanced => SmartRoutingTier::Fast,
        SmartRoutingTier::Fast => SmartRoutingTier::Fast,
    };
    let capped = if tier_rank(maximum_tier) < tier_rank(one_step_lower) {
        one_step_lower
    } else if tier_rank(maximum_tier) < tier_rank(selected) {
        maximum_tier
    } else {
        selected
    };
    (capped, capped != selected)
}

fn tier_rank(tier: SmartRoutingTier) -> u8 {
    match tier {
        SmartRoutingTier::Fast => 0,
        SmartRoutingTier::Balanced => 1,
        SmartRoutingTier::Powerful => 2,
    }
}

fn prefer_candidates(
    candidates: &mut [ProviderModel],
    task_type: TaskType,
    pinned: &PinnedRoutingContext,
) {
    candidates.sort_by_key(|candidate| {
        let pinned_match = match (&pinned.provider, &pinned.model) {
            (None, None) => false,
            (provider, model) => {
                provider
                    .as_deref()
                    .is_none_or(|provider| provider == candidate.provider)
                    && model
                        .as_deref()
                        .is_none_or(|model| model == candidate.model)
            }
        };
        let specialist = candidate.specializations.contains(&task_type);
        (!pinned_match, !specialist)
    });
}

fn apply_policy_snapshot(config: &mut SmartRoutingConfig, policy: RoutingPolicySnapshot) {
    config.enabled = policy.enabled;
    config.classifier = policy.classifier;
    config.ml_model_path = policy.ml_model_path;
    config.classifier_model = policy.classifier_model;
    config.cost_quality_threshold = policy.cost_quality_threshold;
    config.cascade = policy.cascade;
    config.tier_boundaries = policy.tier_boundaries;
    config.heuristic_weights = policy.heuristic_weights;
    config.composite_weights = policy.composite_weights;
    config.streaming_cascade_mode = policy.streaming_cascade_mode;
    config.online_optimizer = policy.online_optimizer;
    config.semantic_cache = policy.semantic_cache;
    config.quality_evaluator = policy.quality_evaluator;
}

#[cfg(test)]
mod tests {
    use std::sync::Mutex;

    use proptest::prelude::*;
    use proptest::test_runner::{Config as ProptestConfig, TestRunner};
    use serde_json::{Map, Value};

    use super::*;
    use crate::models::openai::Message;

    struct FixedClassifier(Result<ClassifierOutput, ClassifierFailure>);

    #[async_trait]
    impl OptionalClassifier for FixedClassifier {
        async fn classify(
            &self,
            _input: ClassifierInput<'_>,
        ) -> Result<ClassifierOutput, ClassifierFailure> {
            self.0
        }
    }

    struct FixedBudget(BudgetDecision);

    #[async_trait]
    impl BudgetPolicy for FixedBudget {
        async fn check(&self, _input: BudgetCheckInput<'_>) -> BudgetDecision {
            self.0
        }
    }

    #[derive(Default)]
    struct CapturingMetrics {
        fallbacks: Mutex<Vec<ClassifierFallbackEvent>>,
    }

    impl SmartRoutingMetrics for CapturingMetrics {
        fn classifier_fallback(&self, event: ClassifierFallbackEvent) {
            self.fallbacks.lock().unwrap().push(event);
        }
    }

    fn config(classifier: ClassifierMode) -> SmartRoutingConfig {
        SmartRoutingConfig {
            enabled: true,
            classifier,
            ml_model_path: matches!(classifier, ClassifierMode::Ml | ClassifierMode::Composite)
                .then(|| "configured-model.onnx".to_string()),
            classifier_model: matches!(classifier, ClassifierMode::Llm)
                .then(|| "configured-llm".to_string()),
            ..SmartRoutingConfig::default()
        }
    }

    fn message(content: &str) -> Message {
        Message {
            role: "user".to_string(),
            content: Value::String(content.to_string()),
            extra: Map::new(),
        }
    }

    fn request(content: &str) -> OpenAIRequest {
        OpenAIRequest {
            model: "group".to_string(),
            messages: vec![message(content)],
            stream: false,
            temperature: None,
            max_tokens: Some(100),
            extra: Map::new(),
        }
    }

    fn candidate(name: &str, context_window: u32, tier: SmartRoutingTier) -> ProviderModel {
        ProviderModel {
            provider: "test".to_string(),
            model: name.to_string(),
            cost_per_million_input_tokens: 0.0,
            cost_per_million_output_tokens: 0.0,
            priority: 100,
            structured_output_passthrough: None,
            tier: Some(tier),
            context_window,
            specializations: Vec::new(),
        }
    }

    fn generated_candidate(
        index: usize,
        tier: Option<SmartRoutingTier>,
        specializations: Vec<TaskType>,
    ) -> ProviderModel {
        ProviderModel {
            provider: format!("provider-{index}"),
            model: format!("model-{index}"),
            cost_per_million_input_tokens: index as f64 / 10.0,
            cost_per_million_output_tokens: index as f64 / 5.0,
            priority: 100 + index as u32,
            structured_output_passthrough: None,
            tier,
            context_window: 8_192 + index as u32 * 1_024,
            specializations,
        }
    }

    fn tier_strategy() -> impl Strategy<Value = SmartRoutingTier> {
        prop_oneof![
            Just(SmartRoutingTier::Fast),
            Just(SmartRoutingTier::Balanced),
            Just(SmartRoutingTier::Powerful),
        ]
    }

    fn task_type_strategy() -> impl Strategy<Value = TaskType> {
        prop_oneof![
            Just(TaskType::CodeGeneration),
            Just(TaskType::MathReasoning),
            Just(TaskType::CreativeWriting),
            Just(TaskType::FactualQA),
            Just(TaskType::ToolUse),
            Just(TaskType::Summarization),
            Just(TaskType::General),
        ]
    }

    fn classifier_failure_strategy() -> impl Strategy<Value = ClassifierFailure> {
        prop_oneof![
            Just(ClassifierFailure::Unavailable),
            Just(ClassifierFailure::Timeout),
            Just(ClassifierFailure::InvalidOutput),
            Just(ClassifierFailure::Backend),
        ]
    }

    fn adjacent_tier(selected: SmartRoutingTier) -> SmartRoutingTier {
        match selected {
            SmartRoutingTier::Fast => SmartRoutingTier::Balanced,
            SmartRoutingTier::Balanced => SmartRoutingTier::Powerful,
            SmartRoutingTier::Powerful => SmartRoutingTier::Balanced,
        }
    }

    fn nonmatching_task(task_type: TaskType) -> TaskType {
        match task_type {
            TaskType::General => TaskType::CodeGeneration,
            _ => TaskType::General,
        }
    }

    fn group(models: Vec<ProviderModel>) -> ModelGroup {
        ModelGroup {
            name: "group".to_string(),
            version_fallback_enabled: false,
            compression: None,
            memory: None,
            structured_output: None,
            models,
        }
    }

    fn input<'a>(
        request: &'a OpenAIRequest,
        group: &'a ModelGroup,
        pinned: &'a PinnedRoutingContext,
    ) -> SmartRoutingInput<'a> {
        SmartRoutingInput {
            request_id: "request-1",
            request,
            model_group: group,
            pinned_context: pinned,
        }
    }

    #[tokio::test]
    async fn heuristic_active_produces_real_decision() {
        let router = SmartRouter::new(config(ClassifierMode::Heuristic)).unwrap();
        let request =
            request("Analyze and implement this code step by step: ```rust fn main() {} ```");
        let group = group(vec![candidate("safe", 10_000, SmartRoutingTier::Balanced)]);
        let pinned = PinnedRoutingContext::default();

        let outcome = router
            .plan(&input(&request, &group, &pinned))
            .await
            .unwrap();
        let RoutingPlanOutcome::Route(plan) = outcome else {
            panic!("routing should continue");
        };
        assert_eq!(plan.decision.classifier, ClassifierUsed::Heuristic);
        assert!(plan.decision.score.value() > 0.0);
        assert!(!plan.decision.cache_hit);
    }

    #[tokio::test]
    async fn unavailable_ml_and_llm_report_heuristic_as_actual_classifier() {
        for mode in [ClassifierMode::Ml, ClassifierMode::Llm] {
            let router = SmartRouter::new(config(mode)).unwrap();
            let request = request("Summarize this text");
            let group = group(vec![candidate("safe", 10_000, SmartRoutingTier::Fast)]);
            let pinned = PinnedRoutingContext::default();

            let classification = router.classify(&input(&request, &group, &pinned)).await;
            assert_eq!(classification.classifier, ClassifierUsed::Heuristic);
        }
    }

    #[tokio::test]
    async fn composite_uses_weighted_backend_success_and_falls_back_on_failure() {
        let request = request("hello");
        let group = group(vec![candidate("safe", 10_000, SmartRoutingTier::Fast)]);
        let pinned = PinnedRoutingContext::default();
        let smart_input = input(&request, &group, &pinned);
        let heuristic = SmartRouter::new(config(ClassifierMode::Heuristic))
            .unwrap()
            .classify(&smart_input)
            .await;
        let success = SmartRouter::new(config(ClassifierMode::Composite))
            .unwrap()
            .with_ml_classifier(Arc::new(FixedClassifier(Ok(ClassifierOutput {
                score: 0.9,
            }))));

        let composite = success.classify(&smart_input).await;
        let weights = CompositeWeights::default();
        let expected = heuristic.score.value() * weights.heuristic + 0.9 * weights.ml;
        assert_eq!(composite.classifier, ClassifierUsed::Composite);
        assert!((composite.score.value() - expected).abs() < 1.0e-12);
        assert_eq!(composite.task_type, heuristic.task_type);

        let failure = SmartRouter::new(config(ClassifierMode::Composite))
            .unwrap()
            .with_ml_classifier(Arc::new(FixedClassifier(Err(ClassifierFailure::Backend))));
        let fallback = failure.classify(&smart_input).await;
        assert_eq!(fallback.classifier, ClassifierUsed::Heuristic);
        assert_eq!(fallback.score, heuristic.score);
    }

    #[tokio::test]
    async fn no_safe_context_returns_typed_413_planning_error() {
        let router = SmartRouter::new(config(ClassifierMode::Heuristic)).unwrap();
        let request = request("small input");
        let group = group(vec![candidate("too-small", 50, SmartRoutingTier::Fast)]);
        let pinned = PinnedRoutingContext {
            additional_input_tokens: 1_000,
            ..PinnedRoutingContext::default()
        };

        let error = router
            .plan(&input(&request, &group, &pinned))
            .await
            .unwrap_err();
        let RoutingPlanningError::ContextCapacity(error) = error else {
            panic!("expected context-capacity error");
        };
        assert_eq!(error.status_code(), 413);
        assert_eq!(error.excluded_count, 1);
    }

    #[tokio::test]
    async fn budget_downgrade_caps_tier_and_reject_stays_typed() {
        let mut routing_config = config(ClassifierMode::Ml);
        routing_config.cost_quality_threshold = 1.0;
        let high_classifier = Arc::new(FixedClassifier(Ok(ClassifierOutput { score: 1.0 })));
        let request = request("route this");
        let group = group(vec![candidate("safe", 10_000, SmartRoutingTier::Balanced)]);
        let pinned = PinnedRoutingContext::default();
        let smart_input = input(&request, &group, &pinned);
        let downgraded = SmartRouter::new(routing_config.clone())
            .unwrap()
            .with_ml_classifier(high_classifier.clone())
            .with_budget(Arc::new(FixedBudget(BudgetDecision::Downgrade {
                maximum_tier: SmartRoutingTier::Fast,
            })));

        let RoutingPlanOutcome::Route(plan) = downgraded.plan(&smart_input).await.unwrap() else {
            panic!("downgrade should continue routing");
        };
        assert_eq!(plan.decision.tier, SmartRoutingTier::Balanced);
        assert!(plan.decision.budget_downgraded);

        let rejected = SmartRouter::new(routing_config)
            .unwrap()
            .with_ml_classifier(high_classifier)
            .with_budget(Arc::new(FixedBudget(BudgetDecision::Reject {
                reason: BudgetRejectionReason::DailyLimit,
            })));
        assert_eq!(
            rejected.plan(&smart_input).await.unwrap(),
            RoutingPlanOutcome::BudgetRejected(BudgetRejection {
                reason: BudgetRejectionReason::DailyLimit,
            })
        );
    }

    #[test]
    fn disabled_config_does_not_construct_router_and_invalid_config_is_rejected() {
        assert!(matches!(
            SmartRouter::new(SmartRoutingConfig::default()),
            Err(SmartRouterBuildError::Disabled)
        ));
        let mut invalid = config(ClassifierMode::Heuristic);
        invalid.cost_quality_threshold = f64::NAN;
        assert!(matches!(
            SmartRouter::new(invalid),
            Err(SmartRouterBuildError::InvalidConfig(_))
        ));
    }

    #[tokio::test]
    async fn fallback_metrics_are_bounded_and_cannot_capture_content() {
        let metrics = Arc::new(CapturingMetrics::default());
        let router = SmartRouter::new(config(ClassifierMode::Llm))
            .unwrap()
            .with_metrics(metrics.clone());
        let request = request("TOP SECRET PROMPT CONTENT");
        let group = group(vec![candidate("safe", 10_000, SmartRoutingTier::Fast)]);
        let pinned = PinnedRoutingContext::default();

        let _ = router.classify(&input(&request, &group, &pinned)).await;
        assert_eq!(
            *metrics.fallbacks.lock().unwrap(),
            vec![ClassifierFallbackEvent {
                configured: ClassifierMode::Llm,
                reason: ClassifierFailure::Unavailable,
            }]
        );
    }

    #[test]
    fn specialist_and_pinned_candidates_are_preferred_without_tier_filtering() {
        let mut specialist = candidate("specialist", 10_000, SmartRoutingTier::Powerful);
        specialist.specializations = vec![TaskType::CodeGeneration];
        let mut candidates = vec![
            candidate("general", 10_000, SmartRoutingTier::Fast),
            specialist,
            candidate("pinned", 10_000, SmartRoutingTier::Balanced),
        ];
        let pinned = PinnedRoutingContext {
            model: Some("pinned".to_string()),
            ..PinnedRoutingContext::default()
        };

        prefer_candidates(&mut candidates, TaskType::CodeGeneration, &pinned);
        assert_eq!(
            candidates
                .iter()
                .map(|candidate| candidate.model.as_str())
                .collect::<Vec<_>>(),
            vec!["pinned", "specialist", "general"]
        );
    }

    #[test]
    fn tier_filter_uses_adjacent_fallback_and_specialists() {
        let router = SmartRouter::new(config(ClassifierMode::Heuristic)).unwrap();
        let mut specialist = candidate("math", 10_000, SmartRoutingTier::Balanced);
        specialist.specializations = vec![TaskType::MathReasoning];
        let source = group(vec![
            candidate("general-balanced", 10_000, SmartRoutingTier::Balanced),
            specialist,
            candidate("powerful", 10_000, SmartRoutingTier::Powerful),
        ]);

        let result = router.filter_by_tier(
            &source,
            SmartRoutingTier::Fast,
            TaskType::MathReasoning,
            None,
        );
        assert_eq!(result.applied_tier, Some(SmartRoutingTier::Balanced));
        assert!(!result.bypassed);
        assert_eq!(result.model_group.models.len(), 1);
        assert_eq!(result.model_group.models[0].model, "math");
    }

    #[test]
    fn tier_filter_bypasses_untagged_and_pinned_groups() {
        let router = SmartRouter::new(config(ClassifierMode::Heuristic)).unwrap();
        let mut untagged = candidate("untagged", 10_000, SmartRoutingTier::Fast);
        untagged.tier = None;
        let untagged_group = group(vec![untagged]);
        let untagged_result = router.filter_by_tier(
            &untagged_group,
            SmartRoutingTier::Powerful,
            TaskType::General,
            None,
        );
        assert!(untagged_result.bypassed);
        assert_eq!(untagged_result.model_group, untagged_group);

        let tagged_group = group(vec![candidate("pinned", 10_000, SmartRoutingTier::Fast)]);
        let pinned_result = router.filter_by_tier(
            &tagged_group,
            SmartRoutingTier::Powerful,
            TaskType::General,
            Some("pinned"),
        );
        assert!(pinned_result.bypassed);
        assert_eq!(pinned_result.model_group, tagged_group);
    }

    #[test]
    fn malformed_score_normalization_is_reserved_for_explicit_failure_path() {
        assert_eq!(
            ComplexityScore::new(f64::NAN).value(),
            DEFAULT_CLASSIFICATION_SCORE
        );
    }

    proptest! {
            #![proptest_config(ProptestConfig::with_cases(64))]

            #[test]
            fn property_4_composite_average_is_bounded(
            heuristic in any::<f64>().prop_filter("finite heuristic score", |score| score.is_finite()),
            ml in any::<f64>().prop_filter("finite ML score", |score| score.is_finite()),
            heuristic_weight in 0.0f64..=1.0,
            ) {
            let weights = CompositeWeights {
            heuristic: heuristic_weight,
            ml: 1.0 - heuristic_weight,
            };
            let score = composite_score(
            ComplexityScore::new(heuristic),
            ComplexityScore::new(ml),
            &weights,
            )
            .value();
            let lower = ComplexityScore::new(heuristic)
            .value()
            .min(ComplexityScore::new(ml).value());
            let upper = ComplexityScore::new(heuristic)
            .value()
            .max(ComplexityScore::new(ml).value());

            prop_assert!(score.is_finite());
            prop_assert!((0.0..=1.0).contains(&score));
            prop_assert!(score + 1.0e-12 >= lower);
            prop_assert!(score <= upper + 1.0e-12);
            }

            #[test]
    fn property_8_tier_filter_is_exact(
    selected_tier in tier_strategy(),
    task_type in task_type_strategy(),
    selected_count in 1usize..=24,
    other_count in 1usize..=24,
    specialist_mask in prop::collection::vec(any::<bool>(), 1..=24),
    ) {
    let mut models = (0..selected_count)
    .map(|index| {
    let specializations = specialist_mask[index % specialist_mask.len()]
    .then_some(vec![task_type])
    .unwrap_or_default();
    generated_candidate(index, Some(selected_tier), specializations)
    })
    .collect::<Vec<_>>();
    let other_tier = adjacent_tier(selected_tier);
    models.extend((0..other_count).map(|index| {
    generated_candidate(
    selected_count + index,
    Some(other_tier),
    vec![nonmatching_task(task_type)],
    )
    }));

            let source = group(models);
            let router = SmartRouter::new(config(ClassifierMode::Heuristic)).unwrap();
            let result = router.filter_by_tier(&source, selected_tier, task_type, None);
            let has_specialist = source
            .models
            .iter()
            .any(|candidate| candidate.specializations.contains(&task_type));

            prop_assert!(!result.bypassed);
            prop_assert_eq!(result.applied_tier, Some(selected_tier));
            prop_assert!(!result.model_group.models.is_empty());
        let all_models_match = result.model_group.models.iter().all(|candidate| {
        candidate.tier == Some(selected_tier)
        && (!has_specialist || candidate.specializations.contains(&task_type))
        });
        prop_assert!(all_models_match);

            let expected = source
            .models
            .iter()
            .filter(|candidate| {
            candidate.tier == Some(selected_tier)
            && (!has_specialist || candidate.specializations.contains(&task_type))
            })
            .cloned()
            .collect::<Vec<_>>();
            prop_assert_eq!(result.model_group.models, expected);
            }

            #[test]
            fn property_9_adjacent_fallback_is_nonempty(
            selected_tier in tier_strategy(),
            task_type in task_type_strategy(),
            adjacent_count in 1usize..=16,
            other_count in 0usize..=16,
            ) {
            let fallback_tier = adjacent_tier(selected_tier);
            let other_tier = tier_fallback_order(selected_tier)[2];
            let mut models = (0..adjacent_count)
            .map(|index| generated_candidate(index, Some(fallback_tier), Vec::new()))
            .collect::<Vec<_>>();
            models.extend((0..other_count).map(|index| {
            generated_candidate(
            adjacent_count + index,
            Some(other_tier),
            vec![nonmatching_task(task_type)],
            )
            }));
            let source = group(models);
            let router = SmartRouter::new(config(ClassifierMode::Heuristic)).unwrap();
            let result = router.filter_by_tier(&source, selected_tier, task_type, None);

            prop_assert!(!result.bypassed);
            prop_assert_eq!(result.applied_tier, Some(fallback_tier));
            prop_assert!(!result.model_group.models.is_empty());
            prop_assert!(result
            .model_group
            .models
            .iter()
            .all(|candidate| candidate.tier == Some(fallback_tier)));
            }

            #[test]
            fn property_10_pinned_bypass_preserves_original_ordering(
            tiers in prop::collection::vec(prop::option::of(tier_strategy()), 1..=32),
            selected_tier in tier_strategy(),
            task_type in task_type_strategy(),
            pinned_index in any::<usize>(),
            ) {
            let models = tiers
            .into_iter()
            .enumerate()
            .map(|(index, tier)| generated_candidate(index, tier, vec![task_type]))
            .collect::<Vec<_>>();
            let source = group(models);
            let original_full_ordering = source.models.clone();
            let pinned_model = source.models[pinned_index % source.models.len()].model.clone();
            let router = SmartRouter::new(config(ClassifierMode::Heuristic)).unwrap();
            let result = router.filter_by_tier(
            &source,
            selected_tier,
            task_type,
            Some(pinned_model.as_str()),
            );

            prop_assert!(result.bypassed);
            prop_assert_eq!(result.applied_tier, None);
        prop_assert_eq!(result.model_group.models.as_slice(), original_full_ordering.as_slice());
        prop_assert_eq!(source.models.as_slice(), original_full_ordering.as_slice());

            }
            }

    #[tokio::test]
    async fn property_19_graceful_classification_always_completes() {
        let strategy = (
            prop_oneof![Just(ClassifierMode::Ml), Just(ClassifierMode::Composite)],
            classifier_failure_strategy(),
            any::<f64>(),
            "[a-zA-Z0-9 _-]{0,128}",
        );
        let mut runner = TestRunner::new(ProptestConfig::with_cases(64));
        let generated_cases = Mutex::new(Vec::with_capacity(256));
        runner
            .run(&strategy, |generated_case| {
                generated_cases.lock().unwrap().push(generated_case);
                Ok(())
            })
            .unwrap();
        let generated_cases = generated_cases.into_inner().unwrap();

        for (mode, failure, generated_score, content) in generated_cases {
            let router = SmartRouter::new(config(mode))
                .unwrap()
                .with_ml_classifier(Arc::new(FixedClassifier(
                    if generated_score.is_finite() && (0.0..=1.0).contains(&generated_score) {
                        Err(failure)
                    } else {
                        Ok(ClassifierOutput {
                            score: generated_score,
                        })
                    },
                )));
            let request = request(&content);
            let group = group(vec![candidate(
                "safe",
                1_000_000,
                SmartRoutingTier::Balanced,
            )]);
            let pinned = PinnedRoutingContext::default();
            let classification = router.classify(&input(&request, &group, &pinned)).await;

            assert_eq!(classification.classifier, ClassifierUsed::Heuristic);
            assert!(classification.score.value().is_finite());
            assert!((0.0..=1.0).contains(&classification.score.value()));
        }
    }
}
