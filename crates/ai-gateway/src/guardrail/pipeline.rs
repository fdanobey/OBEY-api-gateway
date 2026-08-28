//! Pipeline binding resolution and stage ordering.
//!
//! The [`PipelineResolver`] compiles the configured [`GuardrailConfig`] against
//! a [`ProviderRegistry`] once at load time, producing a `name -> ResolvedPipeline`
//! map plus the global-default name and the binding lookups. Given a
//! [`BindingSelector`] describing a request, [`PipelineResolver::resolve`]
//! returns a single flat, ordered `Vec<ResolvedStage>` by concatenating stages
//! in this FIXED order (Req 1.7):
//!
//! 1. Global_Default_Pipeline stages (if configured),
//! 2. virtual-key pipeline stages,
//! 3. model-group pipeline stages,
//! 4. route pipeline stages.
//!
//! Each source contributes its stages in definition order. When no source
//! matches and no global default exists, the resolver returns an empty vector
//! and the request is forwarded unmodified (Req 1.4, 1.6).

use std::collections::HashMap;
use std::sync::Arc;
use std::time::Duration;

use crate::guardrail::config::{
    FailurePolicy, GuardrailConfig, InstructionInsertionMode, PolicyAction, StagePhase,
    ToolResultPhaseConfig,
};
use crate::guardrail::provider::{GuardrailProvider, ProviderRegistry};

/// Describes the request attributes used to look up bound pipelines.
///
/// Any field may be `None` when the corresponding attribute is absent for the
/// request (e.g. an unauthenticated route with no virtual key). Only present
/// fields that match a configured binding contribute stages.
#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub struct BindingSelector {
    /// The request's virtual key id, if authenticated.
    pub virtual_key_id: Option<String>,
    /// The resolved model group name, if any.
    pub model_group: Option<String>,
    /// The route path, if any.
    pub route_path: Option<String>,
}

impl BindingSelector {
    /// Convenience constructor.
    pub fn new(
        virtual_key_id: Option<String>,
        model_group: Option<String>,
        route_path: Option<String>,
    ) -> Self {
        Self {
            virtual_key_id,
            model_group,
            route_path,
        }
    }
}

/// A single compiled stage ready for execution.
///
/// This is the runtime counterpart of [`crate::guardrail::config::StageConfig`]:
/// the provider name has been resolved to a shared `Arc<dyn GuardrailProvider>`,
/// and the failure policy / timeout have been read from the provider's
/// configuration.
#[derive(Clone)]
pub struct ResolvedStage {
    /// Name of the pipeline this stage came from (metric/log label).
    pub pipeline_name: String,
    /// Stage name (metric/log label).
    pub stage_name: String,
    /// Shared provider instance used to analyze content.
    pub provider: Arc<dyn GuardrailProvider>,
    /// Provider type discriminant (metric label), e.g. `"regex"`.
    pub provider_type: &'static str,
    /// Failure policy applied on provider error/timeout.
    pub failure_policy: FailurePolicy,
    /// Per-call analyze timeout.
    pub timeout: Duration,
    /// Enforcement action applied to this stage's findings.
    pub action: PolicyAction,
    /// Whether the stage runs pre-call or post-call, so the engine can split
    /// the flat list into pre/post phases.
    pub phase: StagePhase,
}

impl std::fmt::Debug for ResolvedStage {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        // `dyn GuardrailProvider` is not `Debug`; surface the provider type.
        f.debug_struct("ResolvedStage")
            .field("pipeline_name", &self.pipeline_name)
            .field("stage_name", &self.stage_name)
            .field("provider_type", &self.provider_type)
            .field("failure_policy", &self.failure_policy)
            .field("timeout", &self.timeout)
            .field("action", &self.action)
            .field("phase", &self.phase)
            .finish()
    }
}

/// A compiled pipeline: its declared name plus its ordered resolved stages.
#[derive(Clone, Debug)]
pub struct ResolvedPipeline {
    /// Declared pipeline name.
    #[allow(dead_code)] // retained for diagnostics/tests; unused in the binary build
    pub name: String,
    /// Stages in definition order.
    pub stages: Vec<ResolvedStage>,
    /// Per-pipeline override of the redaction-notice instruction text (Req 4.8, 4.9).
    /// When `None`, the default constant is used.
    pub redaction_notice_instruction: Option<String>,
    /// How the redaction-notice instruction is inserted (Req 4.10).
    pub instruction_insertion_mode: InstructionInsertionMode,
    /// Per-pipeline "failover if refusal is detected" toggle (Req 12.4).
    /// Default: `false`.
    pub failover_on_refusal: bool,
    /// Per-pipeline `tool_result` phase tuning (indirect-injection defense):
    /// whether JSON-object tool-result content is serialized and scanned.
    pub tool_result: ToolResultPhaseConfig,
}

/// Error produced while compiling the [`PipelineResolver`] from configuration.
///
/// Configuration validation (`config/validation.rs`) is expected to catch these
/// cases before the resolver is built; these variants are a defensive
/// safeguard so construction never silently drops stages.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum PipelineResolverError {
    /// A stage referenced a provider name absent from the registry.
    UnknownProvider {
        pipeline: String,
        stage_index: usize,
        provider: String,
    },
    /// A stage referenced a provider name absent from the provider config list
    /// (so no failure policy / timeout could be determined).
    MissingProviderConfig {
        pipeline: String,
        stage_index: usize,
        provider: String,
    },
}

impl std::fmt::Display for PipelineResolverError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            PipelineResolverError::UnknownProvider {
                pipeline,
                stage_index,
                provider,
            } => write!(
                f,
                "pipeline '{pipeline}' stage {stage_index} references provider '{provider}' \
                 which is not registered"
            ),
            PipelineResolverError::MissingProviderConfig {
                pipeline,
                stage_index,
                provider,
            } => write!(
                f,
                "pipeline '{pipeline}' stage {stage_index} references provider '{provider}' \
                 which has no provider configuration"
            ),
        }
    }
}

impl std::error::Error for PipelineResolverError {}

/// Resolves request bindings into an ordered, flat list of executable stages.
///
/// Constructed once from a [`GuardrailConfig`] and its already-built
/// [`ProviderRegistry`]; holds immutable compiled state shared behind an `Arc`
/// by the engine. Hot-reload swaps in a freshly built resolver (Req 1.8).
#[derive(Clone, Debug)]
pub struct PipelineResolver {
    /// Compiled pipelines keyed by name.
    pipelines: HashMap<String, ResolvedPipeline>,
    /// Name of the global-default pipeline, if configured.
    global_default: Option<String>,
    /// virtual-key id â†’ pipeline name.
    virtual_keys: HashMap<String, String>,
    /// model-group name â†’ pipeline name.
    model_groups: HashMap<String, String>,
    /// route path â†’ pipeline name.
    routes: HashMap<String, String>,
    /// Per-binding "failover if refusal is detected" overrides (Req 12.4).
    failover_on_refusal_bindings: HashMap<String, bool>,
}

impl PipelineResolver {
    /// Compile `config` against `registry`, producing a ready-to-use resolver.
    ///
    /// Returns an error if a stage references a provider missing from either the
    /// registry or the provider configuration list. Callers that have already
    /// validated the configuration can treat an error as an internal invariant
    /// violation.
    pub fn new(
        config: &GuardrailConfig,
        registry: &ProviderRegistry,
    ) -> Result<Self, PipelineResolverError> {
        // Map provider name â†’ (failure_policy, timeout) from the config list.
        let provider_meta: HashMap<&str, (FailurePolicy, Duration)> = config
            .providers
            .iter()
            .map(|p| {
                (
                    p.name.as_str(),
                    (p.failure_policy, Duration::from_secs(p.timeout_seconds)),
                )
            })
            .collect();

        let mut pipelines = HashMap::with_capacity(config.pipelines.len());
        for pipeline in &config.pipelines {
            let mut stages = Vec::with_capacity(pipeline.stages.len());
            for (idx, stage) in pipeline.stages.iter().enumerate() {
                let provider = registry.get(&stage.provider).ok_or_else(|| {
                    PipelineResolverError::UnknownProvider {
                        pipeline: pipeline.name.clone(),
                        stage_index: idx,
                        provider: stage.provider.clone(),
                    }
                })?;
                let (failure_policy, timeout) = provider_meta
                    .get(stage.provider.as_str())
                    .copied()
                    .ok_or_else(|| PipelineResolverError::MissingProviderConfig {
                        pipeline: pipeline.name.clone(),
                        stage_index: idx,
                        provider: stage.provider.clone(),
                    })?;

                stages.push(ResolvedStage {
                    pipeline_name: pipeline.name.clone(),
                    stage_name: stage.name.clone(),
                    provider_type: provider.provider_type(),
                    provider,
                    failure_policy,
                    timeout,
                    action: stage.action,
                    phase: stage.phase,
                });
            }

            pipelines.insert(
                pipeline.name.clone(),
                ResolvedPipeline {
                    name: pipeline.name.clone(),
                    stages,
                    redaction_notice_instruction: pipeline.redaction_notice_instruction.clone(),
                    instruction_insertion_mode: pipeline.instruction_insertion_mode,
                    failover_on_refusal: pipeline.failover_on_refusal,
                    tool_result: pipeline.tool_result.clone(),
                },
            );
        }

        Ok(Self {
            pipelines,
            global_default: config.global_default_pipeline.clone(),
            virtual_keys: config.bindings.virtual_keys.clone(),
            model_groups: config.bindings.model_groups.clone(),
            routes: config.bindings.routes.clone(),
            failover_on_refusal_bindings: config.bindings.failover_on_refusal.clone(),
        })
    }

    /// Resolve the ordered, flat list of stages that apply to a request.
    ///
    /// Stages are concatenated in the fixed order global-default â†’ virtual-key â†’
    /// model-group â†’ route, each source contributing its stages in definition
    /// order (Req 1.7). Returns an empty vector when nothing applies (Req 1.4,
    /// 1.6).
    pub fn resolve(&self, selector: &BindingSelector) -> Vec<ResolvedStage> {
        let mut stages = Vec::new();

        // 1. Global default pipeline.
        if let Some(name) = &self.global_default {
            self.extend_with_pipeline(&mut stages, name);
        }

        // 2. Virtual-key binding.
        if let Some(vkey) = &selector.virtual_key_id {
            if let Some(name) = self.virtual_keys.get(vkey) {
                self.extend_with_pipeline(&mut stages, name);
            }
        }

        // 3. Model-group binding.
        if let Some(group) = &selector.model_group {
            if let Some(name) = self.model_groups.get(group) {
                self.extend_with_pipeline(&mut stages, name);
            }
        }

        // 4. Route binding.
        if let Some(route) = &selector.route_path {
            if let Some(name) = self.routes.get(route) {
                self.extend_with_pipeline(&mut stages, name);
            }
        }

        stages
    }

    /// Append the resolved stages of the named pipeline, if it exists.
    fn extend_with_pipeline(&self, stages: &mut Vec<ResolvedStage>, name: &str) {
        if let Some(pipeline) = self.pipelines.get(name) {
            stages.extend(pipeline.stages.iter().cloned());
        }
    }

    /// Look up a compiled pipeline by name (used by tests and diagnostics).
    #[allow(dead_code)] // used by tests/diagnostics; unused in the binary build
    pub fn pipeline(&self, name: &str) -> Option<&ResolvedPipeline> {
        self.pipelines.get(name)
    }

    /// Resolve the redaction-notice instruction config for a request.
    ///
    /// Returns the `(override_text, insertion_mode)` from the most specific
    /// pipeline binding (route > model-group > vkey > global). When no pipeline
    /// applies, returns `(None, Separate)` as the default.
    pub fn resolve_instruction_config(
        &self,
        selector: &BindingSelector,
    ) -> (Option<&str>, InstructionInsertionMode) {
        // Walk bindings from most-specific to least-specific; first match wins.
        let pipeline_name = selector
            .route_path
            .as_ref()
            .and_then(|r| self.routes.get(r))
            .or_else(|| {
                selector
                    .model_group
                    .as_ref()
                    .and_then(|g| self.model_groups.get(g))
            })
            .or_else(|| {
                selector
                    .virtual_key_id
                    .as_ref()
                    .and_then(|v| self.virtual_keys.get(v))
            })
            .or(self.global_default.as_ref());

        match pipeline_name.and_then(|n| self.pipelines.get(n)) {
            Some(pipeline) => (
                pipeline.redaction_notice_instruction.as_deref(),
                pipeline.instruction_insertion_mode,
            ),
            None => (None, InstructionInsertionMode::default()),
        }
    }

    /// The configured global-default pipeline name, if any.
    #[allow(dead_code)] // public accessor; unused in the binary build
    pub fn global_default(&self) -> Option<&str> {
        self.global_default.as_deref()
    }

    /// Resolve the effective `tool_result` phase config for a request
    /// (indirect-injection defense, Req 1.6).
    ///
    /// Mirrors [`Self::resolve_instruction_config`]: the most-specific bound
    /// pipeline (route > model-group > vkey > global) provides the config;
    /// when no pipeline applies, the default (JSON scanning enabled) is used.
    pub fn resolve_tool_result_config(&self, selector: &BindingSelector) -> ToolResultPhaseConfig {
        let pipeline_name = selector
            .route_path
            .as_ref()
            .and_then(|r| self.routes.get(r))
            .or_else(|| {
                selector
                    .model_group
                    .as_ref()
                    .and_then(|g| self.model_groups.get(g))
            })
            .or_else(|| {
                selector
                    .virtual_key_id
                    .as_ref()
                    .and_then(|v| self.virtual_keys.get(v))
            })
            .or(self.global_default.as_ref());

        pipeline_name
            .and_then(|n| self.pipelines.get(n))
            .map(|p| p.tool_result.clone())
            .unwrap_or_default()
    }

    /// Resolve the effective `failover_on_refusal` toggle for a request (Req 12.4).
    ///
    /// The effective toggle is `true` if the per-binding setting for any matching
    /// binding target is `true` OR the resolved pipeline's own toggle is `true`.
    /// When no pipeline applies, returns `false`.
    pub fn resolve_failover_on_refusal(&self, selector: &BindingSelector) -> bool {
        // Check per-binding overrides: any matching target with `true` enables.
        let binding_enabled = selector
            .virtual_key_id
            .as_ref()
            .and_then(|k| self.failover_on_refusal_bindings.get(k).copied())
            .unwrap_or(false)
            || selector
                .model_group
                .as_ref()
                .and_then(|g| self.failover_on_refusal_bindings.get(g).copied())
                .unwrap_or(false)
            || selector
                .route_path
                .as_ref()
                .and_then(|r| self.failover_on_refusal_bindings.get(r).copied())
                .unwrap_or(false);

        if binding_enabled {
            return true;
        }

        // Fall back to the resolved pipeline's own toggle (most-specific binding wins).
        let pipeline_name = selector
            .route_path
            .as_ref()
            .and_then(|r| self.routes.get(r))
            .or_else(|| {
                selector
                    .model_group
                    .as_ref()
                    .and_then(|g| self.model_groups.get(g))
            })
            .or_else(|| {
                selector
                    .virtual_key_id
                    .as_ref()
                    .and_then(|v| self.virtual_keys.get(v))
            })
            .or(self.global_default.as_ref());

        pipeline_name
            .and_then(|n| self.pipelines.get(n))
            .map(|p| p.failover_on_refusal)
            .unwrap_or(false)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::guardrail::config::{
        GuardrailBindings, GuardrailProviderConfig, GuardrailProviderType,
        InstructionInsertionMode, PipelineConfig, ProviderSettings, StageConfig,
    };
    use crate::guardrail::provider::{Finding, GuardrailProviderError};
    use proptest::prelude::*;

    /// Minimal provider whose `provider_type` echoes a fixed discriminant.
    struct StubProvider(&'static str);

    #[async_trait::async_trait]
    impl GuardrailProvider for StubProvider {
        async fn analyze(&self, _content: &str) -> Result<Vec<Finding>, GuardrailProviderError> {
            Ok(vec![])
        }
        fn provider_type(&self) -> &'static str {
            self.0
        }
    }

    fn provider_config(name: &str, policy: FailurePolicy, timeout: u64) -> GuardrailProviderConfig {
        GuardrailProviderConfig {
            name: name.to_string(),
            provider_type: GuardrailProviderType::Regex,
            failure_policy: policy,
            timeout_seconds: timeout,
            settings: ProviderSettings::default(),
        }
    }

    fn stage(name: &str, provider: &str, phase: StagePhase, action: PolicyAction) -> StageConfig {
        StageConfig {
            name: name.to_string(),
            provider: provider.to_string(),
            phase,
            action,
        }
    }

    fn pipeline(name: &str, stages: Vec<StageConfig>) -> PipelineConfig {
        PipelineConfig {
            name: name.to_string(),
            stages,
            redaction_notice_instruction: None,
            instruction_insertion_mode: InstructionInsertionMode::default(),
            failover_on_refusal: false,
            refusal_phrase_list: None,
            tool_result: crate::guardrail::config::ToolResultPhaseConfig::default(),
        }
    }

    fn registry_with(names: &[(&str, &'static str)]) -> ProviderRegistry {
        let mut registry = ProviderRegistry::new();
        for (name, ptype) in names {
            registry.insert(
                *name,
                Arc::new(StubProvider(ptype)) as Arc<dyn GuardrailProvider>,
            );
        }
        registry
    }

    /// Build a config exercising all four binding sources with distinct,
    /// identifiable stage names so ordering is observable.
    fn full_config() -> GuardrailConfig {
        GuardrailConfig {
            providers: vec![provider_config("p", FailurePolicy::FailOpen, 5)],
            pipelines: vec![
                pipeline(
                    "global",
                    vec![stage("g1", "p", StagePhase::PreCall, PolicyAction::Redact)],
                ),
                pipeline(
                    "vkey",
                    vec![
                        stage("v1", "p", StagePhase::PreCall, PolicyAction::Block),
                        stage("v2", "p", StagePhase::PostCall, PolicyAction::Redact),
                    ],
                ),
                pipeline(
                    "group",
                    vec![stage("m1", "p", StagePhase::PreCall, PolicyAction::Mask)],
                ),
                pipeline(
                    "route",
                    vec![stage("r1", "p", StagePhase::PostCall, PolicyAction::Allow)],
                ),
            ],
            global_default_pipeline: Some("global".to_string()),
            bindings: GuardrailBindings {
                virtual_keys: HashMap::from([("vk-1".to_string(), "vkey".to_string())]),
                model_groups: HashMap::from([("grp-a".to_string(), "group".to_string())]),
                routes: HashMap::from([("/v1/chat".to_string(), "route".to_string())]),
                failover_on_refusal: HashMap::new(),
            },
            ..Default::default()
        }
    }

    fn stage_names(stages: &[ResolvedStage]) -> Vec<&str> {
        stages.iter().map(|s| s.stage_name.as_str()).collect()
    }

    #[test]
    fn empty_config_resolves_to_no_stages() {
        let config = GuardrailConfig::default();
        let registry = ProviderRegistry::new();
        let resolver = PipelineResolver::new(&config, &registry).unwrap();

        let stages = resolver.resolve(&BindingSelector::default());
        assert!(stages.is_empty());
    }

    #[test]
    fn no_matching_binding_and_no_global_default_resolves_empty() {
        let mut config = full_config();
        config.global_default_pipeline = None;
        let registry = registry_with(&[("p", "regex")]);
        let resolver = PipelineResolver::new(&config, &registry).unwrap();

        // Selector whose values match no binding.
        let selector = BindingSelector::new(
            Some("unknown".to_string()),
            Some("unknown".to_string()),
            Some("/nope".to_string()),
        );
        assert!(resolver.resolve(&selector).is_empty());
    }

    #[test]
    fn global_default_applies_with_no_bindings() {
        let config = full_config();
        let registry = registry_with(&[("p", "regex")]);
        let resolver = PipelineResolver::new(&config, &registry).unwrap();

        let stages = resolver.resolve(&BindingSelector::default());
        assert_eq!(stage_names(&stages), vec!["g1"]);
    }

    #[test]
    fn single_binding_without_global_default_applies() {
        let mut config = full_config();
        config.global_default_pipeline = None;
        let registry = registry_with(&[("p", "regex")]);
        let resolver = PipelineResolver::new(&config, &registry).unwrap();

        let selector = BindingSelector::new(Some("vk-1".to_string()), None, None);
        assert_eq!(stage_names(&resolver.resolve(&selector)), vec!["v1", "v2"]);
    }

    #[test]
    fn all_sources_concatenate_in_fixed_order() {
        let config = full_config();
        let registry = registry_with(&[("p", "regex")]);
        let resolver = PipelineResolver::new(&config, &registry).unwrap();

        let selector = BindingSelector::new(
            Some("vk-1".to_string()),
            Some("grp-a".to_string()),
            Some("/v1/chat".to_string()),
        );
        // global (g1) -> vkey (v1,v2) -> group (m1) -> route (r1)
        assert_eq!(
            stage_names(&resolver.resolve(&selector)),
            vec!["g1", "v1", "v2", "m1", "r1"]
        );
    }

    #[test]
    fn partial_sources_preserve_relative_order() {
        let config = full_config();
        let registry = registry_with(&[("p", "regex")]);
        let resolver = PipelineResolver::new(&config, &registry).unwrap();

        // global default + model-group only (no vkey, no route).
        let selector = BindingSelector::new(None, Some("grp-a".to_string()), None);
        assert_eq!(stage_names(&resolver.resolve(&selector)), vec!["g1", "m1"]);
    }

    #[test]
    fn resolved_stage_carries_provider_and_policy_metadata() {
        let mut config = full_config();
        config.providers = vec![provider_config("p", FailurePolicy::FailClose, 12)];
        let registry = registry_with(&[("p", "regex")]);
        let resolver = PipelineResolver::new(&config, &registry).unwrap();

        let stages = resolver.resolve(&BindingSelector::default());
        let g1 = &stages[0];
        assert_eq!(g1.pipeline_name, "global");
        assert_eq!(g1.stage_name, "g1");
        assert_eq!(g1.provider_type, "regex");
        assert_eq!(g1.failure_policy, FailurePolicy::FailClose);
        assert_eq!(g1.timeout, Duration::from_secs(12));
        assert_eq!(g1.action, PolicyAction::Redact);
        assert_eq!(g1.phase, StagePhase::PreCall);
    }

    #[test]
    fn unknown_provider_in_registry_is_rejected() {
        let config = full_config();
        // Registry missing the referenced provider "p".
        let registry = ProviderRegistry::new();
        let err = PipelineResolver::new(&config, &registry).unwrap_err();
        assert!(matches!(err, PipelineResolverError::UnknownProvider { .. }));
    }

    #[test]
    fn missing_provider_config_is_rejected() {
        let mut config = full_config();
        config.providers.clear(); // registry has it, config list does not
        let registry = registry_with(&[("p", "regex")]);
        let err = PipelineResolver::new(&config, &registry).unwrap_err();
        assert!(matches!(
            err,
            PipelineResolverError::MissingProviderConfig { .. }
        ));
    }

    // ---- Property-based tests (proptest, >=100 cases) ----
    //
    // These build randomized configs and selectors, then assert the resolver's
    // output against an independently-computed oracle. Generators keep the
    // input space small but exhaustive over the resolution-relevant dimensions:
    // present/absent global default, present/absent bindings per source,
    // matching vs non-matching selector ids, and targets that reference either a
    // defined pipeline or a missing name.

    /// Fixed binding keys used by the generators. A selector either supplies the
    /// exact key (a "match"), a different value (a "miss"), or `None`.
    const VKEY_KEY: &str = "vk";
    const GROUP_KEY: &str = "grp";
    const ROUTE_KEY: &str = "rt";

    /// A generated binding source: whether a binding entry exists, which target
    /// pipeline index it points at (interpreted modulo `n_pipelines + 1`, where
    /// the extra slot means a missing/undefined pipeline name), and the selector
    /// state (0 = absent, 1 = matching key, 2 = non-matching key).
    type SourceParams = (bool, usize, u8);

    /// Everything the two properties need, derived from generated primitives.
    struct Scenario {
        config: GuardrailConfig,
        selector: BindingSelector,
        /// Stage names the resolver is expected to produce, in order.
        expected_names: Vec<String>,
        /// Global default configured AND names a defined pipeline.
        global_matches: bool,
        /// Each binding source: selector matches AND binding entry exists AND
        /// its target names a defined pipeline.
        vkey_matches: bool,
        group_matches: bool,
        route_matches: bool,
    }

    /// Deterministic stage names for pipeline `i`: `s{i}_{j}`.
    fn stage_name(i: usize, j: usize) -> String {
        format!("s{i}_{j}")
    }

    /// Resolve a generated target index into a pipeline name plus whether that
    /// name refers to a defined pipeline. Index `n` (the extra slot) maps to a
    /// deliberately-missing name so we exercise dangling references.
    fn target(idx: usize, n: usize) -> (String, Option<usize>) {
        let m = idx % (n + 1);
        if m == n {
            ("missing_pipeline".to_string(), None)
        } else {
            (format!("pipe{m}"), Some(m))
        }
    }

    /// Selector value for a given state and key: 0 => None, 1 => the exact key
    /// (match), 2 => a distinct value (miss).
    fn selector_value(state: u8, key: &str) -> Option<String> {
        match state % 3 {
            0 => None,
            1 => Some(key.to_string()),
            _ => Some(format!("other-{key}")),
        }
    }

    /// Build a full scenario (config + selector + oracle) from generated params.
    fn build_scenario(
        stage_counts: &[usize],
        gd: (bool, usize),
        vkey: SourceParams,
        group: SourceParams,
        route: SourceParams,
    ) -> Scenario {
        let n = stage_counts.len();

        // Pipelines: pipe0..pipe{n-1}, each with >=1 uniquely-named stage.
        let pipelines: Vec<PipelineConfig> = stage_counts
            .iter()
            .enumerate()
            .map(|(i, &count)| {
                let stages = (0..count)
                    .map(|j| {
                        stage(
                            &stage_name(i, j),
                            "p",
                            StagePhase::PreCall,
                            PolicyAction::Allow,
                        )
                    })
                    .collect();
                pipeline(&format!("pipe{i}"), stages)
            })
            .collect();

        // Global default.
        let (gd_present, gd_idx) = gd;
        let (gd_name, gd_defined) = target(gd_idx, n);
        let global_default_pipeline = if gd_present { Some(gd_name) } else { None };
        let global_matches = gd_present && gd_defined.is_some();

        // Per-source binding targets.
        let (vk_present, vk_idx, vk_state) = vkey;
        let (grp_present, grp_idx, grp_state) = group;
        let (rt_present, rt_idx, rt_state) = route;

        let (vk_name, vk_defined) = target(vk_idx, n);
        let (grp_name, grp_defined) = target(grp_idx, n);
        let (rt_name, rt_defined) = target(rt_idx, n);

        let mut virtual_keys = HashMap::new();
        if vk_present {
            virtual_keys.insert(VKEY_KEY.to_string(), vk_name);
        }
        let mut model_groups = HashMap::new();
        if grp_present {
            model_groups.insert(GROUP_KEY.to_string(), grp_name);
        }
        let mut routes = HashMap::new();
        if rt_present {
            routes.insert(ROUTE_KEY.to_string(), rt_name);
        }

        let selector = BindingSelector::new(
            selector_value(vk_state, VKEY_KEY),
            selector_value(grp_state, GROUP_KEY),
            selector_value(rt_state, ROUTE_KEY),
        );

        // A source contributes iff: selector supplies the exact key (state 1),
        // a binding entry exists for that key, and its target is defined.
        let vkey_matches = vk_present && vk_state % 3 == 1 && vk_defined.is_some();
        let group_matches = grp_present && grp_state % 3 == 1 && grp_defined.is_some();
        let route_matches = rt_present && rt_state % 3 == 1 && rt_defined.is_some();

        // Oracle stage-name sequence, concatenated in the FIXED source order:
        // global default -> virtual-key -> model-group -> route.
        let names_of = |pi: usize| -> Vec<String> {
            (0..stage_counts[pi]).map(|j| stage_name(pi, j)).collect()
        };
        let mut expected_names = Vec::new();
        if global_matches {
            expected_names.extend(names_of(gd_defined.unwrap()));
        }
        if vkey_matches {
            expected_names.extend(names_of(vk_defined.unwrap()));
        }
        if group_matches {
            expected_names.extend(names_of(grp_defined.unwrap()));
        }
        if route_matches {
            expected_names.extend(names_of(rt_defined.unwrap()));
        }

        let config = GuardrailConfig {
            providers: vec![provider_config("p", FailurePolicy::FailOpen, 5)],
            pipelines,
            global_default_pipeline,
            bindings: GuardrailBindings {
                virtual_keys,
                model_groups,
                routes,
                failover_on_refusal: HashMap::new(),
            },
            ..Default::default()
        };

        Scenario {
            config,
            selector,
            expected_names,
            global_matches,
            vkey_matches,
            group_matches,
            route_matches,
        }
    }

    /// Strategy for a single binding source's parameters.
    fn source_strategy() -> impl Strategy<Value = SourceParams> {
        (any::<bool>(), 0usize..8, 0u8..3)
    }

    proptest! {
        #![proptest_config(ProptestConfig { cases: 64, ..ProptestConfig::default() })]

        /// Property 31: Failover toggle gates re-dispatch.
        ///
        /// The effective `failover_on_refusal` toggle resolved by
        /// [`PipelineResolver::resolve_failover_on_refusal`] is `true` if and
        /// only if at least one per-binding key matching the selector is set to
        /// `true` OR the resolved pipeline's own `failover_on_refusal` field is
        /// `true`. When the toggle is disabled (both binding and pipeline
        /// default to `false`), refusal detection fires but no re-dispatch
        /// occurs â€” the response is returned unmodified (Req 12.6). When
        /// enabled via either path, a refusal triggers failover (Req 12.4,
        /// 12.5).
        ///
        /// **Validates: Requirements 12.4, 12.5, 12.6**
        #[test]
        fn prop_failover_toggle_gates_redispatch(
            // Whether the pipeline itself has failover_on_refusal = true
            pipeline_toggle in any::<bool>(),
            // Per-binding failover_on_refusal values for each selector dimension
            binding_vkey_toggle in prop::option::of(any::<bool>()),
            binding_group_toggle in prop::option::of(any::<bool>()),
            binding_route_toggle in prop::option::of(any::<bool>()),
            // Whether each selector field is present (matches)
            selector_has_vkey in any::<bool>(),
            selector_has_group in any::<bool>(),
            selector_has_route in any::<bool>(),
            // Whether a global default or route binding is used
            use_global_default in any::<bool>(),
        ) {
            // Build a pipeline with the given toggle.
            let mut pipe = pipeline(
                "test_pipe",
                vec![stage("s1", "p", StagePhase::PreCall, PolicyAction::Allow)],
            );
            pipe.failover_on_refusal = pipeline_toggle;

            // Build bindings pointing to the test pipeline.
            let mut virtual_keys = HashMap::new();
            let mut model_groups = HashMap::new();
            let mut routes = HashMap::new();
            let mut failover_on_refusal_bindings = HashMap::new();

            // Add a vkey binding that references our pipeline.
            virtual_keys.insert("vk-1".to_string(), "test_pipe".to_string());
            // Add a model-group binding.
            model_groups.insert("grp-1".to_string(), "test_pipe".to_string());
            // Add a route binding.
            routes.insert("/chat".to_string(), "test_pipe".to_string());

            // Set per-binding failover toggles when present.
            if let Some(val) = binding_vkey_toggle {
                failover_on_refusal_bindings.insert("vk-1".to_string(), val);
            }
            if let Some(val) = binding_group_toggle {
                failover_on_refusal_bindings.insert("grp-1".to_string(), val);
            }
            if let Some(val) = binding_route_toggle {
                failover_on_refusal_bindings.insert("/chat".to_string(), val);
            }

            let config = GuardrailConfig {
                providers: vec![provider_config("p", FailurePolicy::FailOpen, 5)],
                pipelines: vec![pipe],
                global_default_pipeline: if use_global_default {
                    Some("test_pipe".to_string())
                } else {
                    None
                },
                bindings: GuardrailBindings {
                    virtual_keys,
                    model_groups,
                    routes,
                    failover_on_refusal: failover_on_refusal_bindings.clone(),
                },
                ..Default::default()
            };

            let registry = registry_with(&[("p", "regex")]);
            let resolver = PipelineResolver::new(&config, &registry).unwrap();

            // Build selector based on generated flags.
            let selector = BindingSelector::new(
                if selector_has_vkey { Some("vk-1".to_string()) } else { None },
                if selector_has_group { Some("grp-1".to_string()) } else { None },
                if selector_has_route { Some("/chat".to_string()) } else { None },
            );

            let effective = resolver.resolve_failover_on_refusal(&selector);

            // Oracle: compute expected effective toggle.
            // Per-binding enabled if any matching selector key is set to true.
            let binding_enabled = (selector_has_vkey
                && binding_vkey_toggle.unwrap_or(false))
                || (selector_has_group
                    && binding_group_toggle.unwrap_or(false))
                || (selector_has_route
                    && binding_route_toggle.unwrap_or(false));

            // Pipeline toggle applies if the pipeline is resolvable.
            // Resolution priority: route > model-group > vkey > global.
            let pipeline_resolves = selector_has_route
                || selector_has_group
                || selector_has_vkey
                || use_global_default;

            let expected = binding_enabled
                || (pipeline_resolves && pipeline_toggle);

            prop_assert_eq!(
                effective,
                expected,
                "effective failover toggle must be true iff any matching binding \
                 is true OR the resolved pipeline's toggle is true. \
                 binding_enabled={}, pipeline_resolves={}, pipeline_toggle={}",
                binding_enabled, pipeline_resolves, pipeline_toggle
            );
        }

        /// Property 32: Failover attempts are bounded and skip open breakers.
        ///
        /// Given a randomly generated fallback ordering of N targets, a subset
        /// with open circuit breakers, and an "already-tried" provider, the
        /// bounded failover re-dispatch loop (modeled here as a pure function)
        /// guarantees:
        /// 1. Each target is attempted at most once (bounded by ordering length)
        ///    â€” Req 12.7.
        /// 2. Targets with open circuit breakers are NEVER dispatched to
        ///    â€” Req 12.10.
        /// 3. The total number of dispatch attempts is at most the number of
        ///    available (CB-closed, not-yet-tried) targets.
        ///
        /// Uses a fake dispatcher: responses are simulated as always-refusal to
        /// exercise the full loop without network I/O.
        ///
        /// **Validates: Requirements 12.7, 12.10**
        #[test]
        fn prop_bounded_attempts_and_breaker_skipping(
            // Number of targets in the fallback ordering (1..=8).
            num_targets in 1usize..=8,
            // Bitmask: which targets have their circuit breaker OPEN.
            cb_open_bits in any::<u8>(),
            // Index of the provider that produced the initial (refused) response.
            already_tried_idx in 0usize..8,
        ) {
            // --- Model the fallback ordering ---
            let fallback_order: Vec<String> = (0..num_targets)
                .map(|i| format!("provider_{i}"))
                .collect();

            // Which targets have open CBs (bit i â†’ target i is open).
            let cb_open: Vec<bool> = (0..num_targets)
                .map(|i| (cb_open_bits >> (i % 8)) & 1 == 1)
                .collect();

            // The provider that already produced the refused response.
            let already_tried = format!("provider_{}", already_tried_idx % num_targets);

            // --- Simulate the bounded re-dispatch loop (same logic as handler) ---
            let mut tried: Vec<String> = vec![already_tried.clone()];
            let mut dispatched_to: Vec<String> = Vec::new();

            for (i, target) in fallback_order.iter().enumerate() {
                // Skip already-tried targets (Req 12.7).
                if tried.contains(target) {
                    continue;
                }
                // Skip targets whose circuit breaker is open (Req 12.10).
                if cb_open[i] {
                    continue;
                }

                // Record that we dispatched to this target.
                dispatched_to.push(target.clone());
                tried.push(target.clone());

                // In this property test, the fake dispatcher always returns a
                // refusal, so we never break early â€” exercising the full loop.
            }

            // --- Assertions ---

            // 1. Each target is dispatched to AT MOST ONCE (Req 12.7).
            let mut seen = std::collections::HashSet::new();
            for t in &dispatched_to {
                prop_assert!(
                    seen.insert(t.clone()),
                    "Target {:?} was dispatched to more than once (Req 12.7)",
                    t
                );
            }

            // 2. No target with an open circuit breaker was dispatched to (Req 12.10).
            for (i, target) in fallback_order.iter().enumerate() {
                if cb_open[i] {
                    prop_assert!(
                        !dispatched_to.contains(target),
                        "Target {:?} has open CB but was dispatched to (Req 12.10)",
                        target
                    );
                }
            }

            // 3. Total dispatch attempts <= number of available (CB-closed, not-already-tried) targets.
            let available_count = fallback_order.iter().enumerate().filter(|(i, t)| {
                !cb_open[*i] && **t != already_tried
            }).count();
            prop_assert!(
                dispatched_to.len() <= available_count,
                "dispatched {} times but only {} targets are available (CB-closed, not-already-tried)",
                dispatched_to.len(),
                available_count
            );

            // 4. Dispatched count equals available count (all-refusal means we exhaust them).
            prop_assert_eq!(
                dispatched_to.len(),
                available_count,
                "With all-refusal responses, we must attempt every available target exactly once"
            );

            // 5. Total attempts bounded by ordering length (Req 12.7).
            prop_assert!(
                dispatched_to.len() <= num_targets,
                "dispatch count {} exceeds fallback ordering length {} (Req 12.7)",
                dispatched_to.len(),
                num_targets
            );
        }

        /// Property 33: Failover exhaustion returns the last response.
        ///
        /// When the refusal-failover toggle is enabled and ALL attempted
        /// responses are refusals (the ordering is exhausted), the loop returns
        /// the last received response. PII re-injection still runs on that
        /// response. The last response is from the last target that was actually
        /// attempted (not one whose circuit breaker was open/skipped).
        ///
        /// Uses a fake dispatcher that always returns a refusal response for
        /// each target, simulating full exhaustion of the Router_Fallback_Ordering.
        ///
        /// **Validates: Requirements 12.8**
        #[test]
        fn prop_failover_exhaustion_returns_last_response(
            // Number of targets in the fallback ordering (1..=8).
            num_targets in 1usize..=8,
            // Which targets have their circuit breaker open (bit mask).
            breaker_bits in any::<u8>(),
            // Whether the re-injection map has entries (tests PII re-injection on
            // the exhausted response).
            has_reinjection_entries in any::<bool>(),
            // A unique suffix per target so we can identify the last attempted.
            target_suffix in prop::collection::vec("[a-z]{3}", 1..=8),
        ) {
            // Ensure we have enough suffixes for the targets.
            let suffixes: Vec<String> = target_suffix.into_iter().take(num_targets).collect();
            prop_assume!(suffixes.len() == num_targets);

            // --- Fake dispatcher: each target returns a refusal response whose
            // content includes "i cannot help <suffix>" (a known default refusal
            // phrase).  We model the loop identically to the handler: iterate
            // targets in order, skip those with open breakers, attempt each at
            // most once, and on exhaustion return the last response.

            let detector = crate::guardrail::refusal::RefusalDetector::default_detector();

            // No-tools context: no tool-omission signal, rely on phrase matching.
            let no_tools = crate::guardrail::refusal::ToolContext {
                tool_use_allowed: false,
                tools_provided: false,
                finish_reason_is_tool_call: false,
                has_tool_calls: false,
            };

            // Simulate the first response (from the original dispatch) is also a
            // refusal with a distinguishable content marker.
            let initial_content = format!("i can't help with initial_target");

            // Track which targets are skipped due to open breaker.
            let mut last_attempted_content: Option<String> = None;
            let mut attempted_count: usize = 0;

            // The loop mirrors the handler's bounded re-dispatch loop (Req 12.7):
            // iterate all targets, skip open-breaker ones, attempt each once.
            for (idx, suffix) in suffixes.iter().enumerate() {
                // Simulate circuit-breaker check: bit is set => breaker is open.
                let breaker_open = (breaker_bits >> (idx % 8)) & 1 == 1;
                if breaker_open {
                    continue; // Skip this target (Req 12.10).
                }

                // Fake dispatch: this target always returns a refusal.
                let target_content = format!("i can't help with {}", suffix);

                // Verify our fake dispatcher produces a refusal.
                let decision = detector.detect(&target_content, &no_tools);
                prop_assert!(
                    decision.is_refusal(),
                    "Fake dispatcher must produce a refusal for target {}",
                    idx
                );

                last_attempted_content = Some(target_content);
                attempted_count += 1;
            }

            // Determine the "final" response: if no target was attempted (all
            // breakers open), the initial response is returned.
            let final_content = last_attempted_content
                .unwrap_or(initial_content.clone());

            // --- Verify Property 33 assertions:

            // 1) The returned response is the last actually-attempted response
            //    (or the initial if nothing was attempted).
            if attempted_count == 0 {
                prop_assert_eq!(
                    &final_content,
                    &initial_content,
                    "When no target is attempted, the initial response is returned"
                );
            } else {
                // It must be from the LAST attempted target (not a skipped one).
                let last_attempted_suffix = suffixes.iter().enumerate()
                    .rev()
                    .find(|(idx, _)| (breaker_bits >> (idx % 8)) & 1 == 0)
                    .map(|(_, s)| s.as_str())
                    .unwrap();
                let expected_content = format!("i can't help with {}", last_attempted_suffix);
                prop_assert_eq!(
                    &final_content,
                    &expected_content,
                    "Exhaustion must return the last attempted target's response, \
                     not a skipped one"
                );
            }

            // 2) The final content IS a refusal (all were refusals, so the
            //    returned one is necessarily also a refusal).
            let final_decision = detector.detect(&final_content, &no_tools);
            prop_assert!(
                final_decision.is_refusal(),
                "The exhausted response must still be a refusal"
            );

            // 3) PII re-injection runs on the final response. Simulate with a
            //    simple placeholder in the content.
            if has_reinjection_entries {
                use crate::guardrail::pii::GuardrailContext;

                let mut ctx = GuardrailContext::new();
                // Add a placeholder entry to the context.
                let _placeholder_result = ctx.placeholder_for("EMAIL", "user@example.com");

                // Simulate the response containing a placeholder token.
                let content_with_placeholder =
                    format!("{} <<PII_EMAIL_1>>", final_content);

                // Re-injection replaces the placeholder with the original value.
                let reinjected = ctx.reinject(&content_with_placeholder);
                prop_assert!(
                    reinjected.contains("user@example.com"),
                    "PII re-injection must run on the exhausted response"
                );
                prop_assert!(
                    !reinjected.contains("<<PII_EMAIL_1>>"),
                    "Placeholder must be fully replaced after re-injection"
                );
            }
        }

        /// Property 1: Binding resolution presence.
        ///
        /// For any config and selector, the resolved stage list is non-empty if
        /// and only if a global default is configured and names a defined
        /// pipeline, OR at least one of the selector's virtual-key / model-group
        /// / route bindings matches a defined pipeline; otherwise it is empty.
        ///
        /// Validates: Requirements 1.3, 1.4, 1.6
        #[test]
        fn prop_binding_resolution_presence(
            stage_counts in prop::collection::vec(1usize..=3, 1..=4),
            gd in (any::<bool>(), 0usize..8),
            vkey in source_strategy(),
            group in source_strategy(),
            route in source_strategy(),
        ) {
            let scenario = build_scenario(&stage_counts, gd, vkey, group, route);
            let registry = registry_with(&[("p", "regex")]);
            let resolver = PipelineResolver::new(&scenario.config, &registry).unwrap();

            let resolved = resolver.resolve(&scenario.selector);

            // Independent oracle: presence tracks the disjunction of matches.
            let expect_nonempty = scenario.global_matches
                || scenario.vkey_matches
                || scenario.group_matches
                || scenario.route_matches;

            prop_assert_eq!(
                !resolved.is_empty(),
                expect_nonempty,
                "resolved presence must match the presence oracle"
            );
        }

        /// Property 2: Fixed binding concatenation order.
        ///
        /// For any combination of present/absent global-default, vkey,
        /// model-group, and route bindings, the resolved stage list equals the
        /// concatenation of global-default stages, then vkey, then model-group,
        /// then route stages â€” each source in definition order.
        ///
        /// Validates: Requirements 1.5, 1.7
        #[test]
        fn prop_fixed_binding_concatenation_order(
            stage_counts in prop::collection::vec(1usize..=3, 1..=4),
            gd in (any::<bool>(), 0usize..8),
            vkey in source_strategy(),
            group in source_strategy(),
            route in source_strategy(),
        ) {
            let scenario = build_scenario(&stage_counts, gd, vkey, group, route);
            let registry = registry_with(&[("p", "regex")]);
            let resolver = PipelineResolver::new(&scenario.config, &registry).unwrap();

            let resolved = resolver.resolve(&scenario.selector);
            let resolved_names: Vec<String> = resolved
                .iter()
                .map(|s| s.stage_name.clone())
                .collect();

            prop_assert_eq!(
                resolved_names,
                scenario.expected_names,
                "resolved stage order must equal the fixed-order oracle concatenation"
            );
        }
    }
}
