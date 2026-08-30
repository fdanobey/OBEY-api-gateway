//! Reasoning-parameter normalization between provider families (design
//! Component 3).
//!
//! Converts whatever reasoning parameter the client sent — OpenAI-style
//! `reasoning_effort`, Anthropic manual `thinking: {type:"enabled",
//! budget_tokens}`, Anthropic adaptive `thinking: {type:"adaptive"}` (with
//! `output_config.effort`), or OpenRouter-style `reasoning: {max_tokens}` —
//! into the single parameter shape the resolved target model accepts.
//!
//! Pipeline position: the router reads the client spec from the cloned
//! outgoing request with [`read_client_spec`] once, then calls
//! [`emit_for_target`] per failover attempt so every attempt carries the
//! correct parameter shape for its own target family.
//!
//! Normalization rules (requirements 3.1-3.8):
//!
//! | Target shape            | Emitted parameter                                            |
//! |-------------------------|--------------------------------------------------------------|
//! | Anthropic manual        | `thinking: {type:"enabled", budget_tokens:N}` (N >= 1024 and < `max_tokens`, raising `max_tokens` to N+1 when needed) |
//! | Anthropic adaptive      | `thinking: {type:"adaptive"}` + `output_config.effort` — NEVER `type:"enabled"` (400 on Claude 4.7+) |
//! | OpenRouter              | `reasoning: {max_tokens: N}`                                 |
//! | OpenAI / xAI            | `reasoning_effort: "minimal|low|medium|high|xhigh"`          |
//! | No reasoning support    | all reasoning parameters removed                             |
//!
//! Effort→budget uses the configurable [`EffortBudgetMap`] (defaults
//! 1024/2048/8192/16384/32768). Budget→effort uses the inverse rule: the
//! nearest effort whose mapped budget is <= the value (falling back to
//! `minimal` when the value is below every mapped budget).
//!
//! Sampling parameters (`temperature`, `top_p`, `top_k`) are dropped when
//! Anthropic thinking parameters are emitted; Anthropic rejects them
//! combined with thinking (requirement 3.8).
//!
//! No-injection rule: when the client sent no reasoning parameter, nothing
//! is injected for any target (requirement 3.7).

use crate::config::ProviderModel;
use crate::models::openai::OpenAIRequest;
use crate::reasoning_compat::config::{
    Effort, EffortBudgetMap, ReasoningCompatConfig, ReasoningFamily, ReasoningParamShape,
    MIN_REASONING_BUDGET_TOKENS,
};
use crate::reasoning_compat::detect::classify_family;
use serde_json::{json, Value};

/// What the client asked for, in family-neutral form.
///
/// Captures any of the three input shapes ([`read_client_spec`]):
/// - `reasoning_effort: "high"` → `effort: Some(High)`
/// - `thinking: {type:"enabled", budget_tokens: N}` → `budget: Some(N)`
/// - `thinking: {type:"adaptive"}` (+ `output_config.effort`) → `adaptive:
///   true` (and `effort` when the effort level was sent)
/// - `reasoning: {max_tokens: N}` → `budget: Some(N)`
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct ReasoningSpec {
    /// Client effort level (from `reasoning_effort`, `output_config.effort`,
    /// or `reasoning.effort`).
    pub effort: Option<Effort>,
    /// Client token budget (from `thinking.budget_tokens` or
    /// `reasoning.max_tokens`).
    pub budget: Option<u32>,
    /// Client explicitly requested Anthropic adaptive thinking.
    pub adaptive: bool,
}

impl ReasoningSpec {
    /// True when the client sent no reasoning parameter at all.
    pub fn is_empty(&self) -> bool {
        self.effort.is_none() && self.budget.is_none() && !self.adaptive
    }
}

/// Outcome of normalizing reasoning parameters for one target attempt.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct NormalizeReport {
    /// Shape emitted: `"reasoning_effort"`, `"thinking_enabled"`,
    /// `"thinking_adaptive"`, `"reasoning_max_tokens"`, or `"none"`.
    pub emitted_shape: &'static str,
    /// A budget below the Anthropic 1024 floor was clamped up.
    pub clamped: bool,
    /// Conflicting sampling parameters (temperature/top_p/top_k) were
    /// dropped for an Anthropic thinking attempt.
    pub sampling_dropped: bool,
    /// `max_tokens` was raised above the emitted thinking budget.
    pub max_tokens_raised: bool,
}

impl NormalizeReport {
    /// Report for an attempt where no reasoning parameter was emitted.
    fn none() -> Self {
        Self {
            emitted_shape: "none",
            clamped: false,
            sampling_dropped: false,
            max_tokens_raised: false,
        }
    }
}

/// Read the client's reasoning parameter from the outgoing request.
///
/// Precedence: the `thinking` object (the most explicit carrier) wins;
/// otherwise `reasoning_effort`; otherwise the OpenRouter `reasoning`
/// object. Unparseable values (unknown effort strings, missing numbers)
/// yield an empty spec, which [`emit_for_target`] treats as no-op
/// passthrough.
pub fn read_client_spec(outgoing: &OpenAIRequest) -> ReasoningSpec {
    let mut spec = ReasoningSpec {
        effort: None,
        budget: None,
        adaptive: false,
    };

    if let Some(thinking) = outgoing.extra.get("thinking").and_then(Value::as_object) {
        match thinking.get("type").and_then(Value::as_str) {
            Some("enabled") => {
                spec.budget = json_u32(thinking.get("budget_tokens"));
                return spec;
            }
            Some("adaptive") => {
                spec.adaptive = true;
                spec.effort = outgoing
                    .extra
                    .get("output_config")
                    .and_then(Value::as_object)
                    .and_then(|config| config.get("effort"))
                    .and_then(Value::as_str)
                    .and_then(Effort::parse);
                return spec;
            }
            _ => {}
        }
    }

    if let Some(effort) = outgoing
        .extra
        .get("reasoning_effort")
        .and_then(Value::as_str)
        .and_then(Effort::parse)
    {
        spec.effort = Some(effort);
        return spec;
    }

    if let Some(reasoning) = outgoing.extra.get("reasoning").and_then(Value::as_object) {
        spec.budget = json_u32(reasoning.get("max_tokens"));
        spec.effort = reasoning
            .get("effort")
            .and_then(Value::as_str)
            .and_then(Effort::parse);
    }

    spec
}

/// Normalize the client's reasoning spec into the parameter shape the
/// target model accepts.
///
/// Target resolution:
/// - family: `target.reasoning_family` or [`classify_family`] of the
///   target model id
/// - shape: `target.reasoning_parameter` or the family-derived default
///
/// Empty spec → nothing is emitted and nothing is removed (no-injection
/// rule, requirement 3.7). Budget violations are clamped and logged via
/// `tracing::debug!`, never forwarded (requirement 3.6).
pub fn emit_for_target(
    outgoing: &mut OpenAIRequest,
    spec: ReasoningSpec,
    target: &ProviderModel,
    cfg: &ReasoningCompatConfig,
) -> NormalizeReport {
    let family = target
        .reasoning_family
        .unwrap_or_else(|| classify_family(&target.model));
    let shape = target
        .reasoning_parameter
        .unwrap_or_else(|| family_default_shape(family));

    if spec.is_empty() {
        return NormalizeReport::none();
    }

    match shape {
        ReasoningParamShape::ReasoningEffort => emit_reasoning_effort(outgoing, spec, cfg),
        ReasoningParamShape::ThinkingBudget => emit_thinking_budget(outgoing, spec, cfg),
        ReasoningParamShape::Adaptive => emit_adaptive(outgoing, spec),
        ReasoningParamShape::ReasoningMaxTokens => emit_reasoning_max_tokens(outgoing, spec, cfg),
        ReasoningParamShape::None => {
            remove_reasoning_params(outgoing, ReasoningParamShape::None);
            NormalizeReport::none()
        }
    }
}

/// Default parameter shape for a reasoning family.
fn family_default_shape(family: ReasoningFamily) -> ReasoningParamShape {
    match family {
        ReasoningFamily::AnthropicManual => ReasoningParamShape::ThinkingBudget,
        ReasoningFamily::AnthropicAdaptive => ReasoningParamShape::Adaptive,
        ReasoningFamily::OpenAIReasoning | ReasoningFamily::XAI => {
            ReasoningParamShape::ReasoningEffort
        }
        ReasoningFamily::OpenRouter => ReasoningParamShape::ReasoningMaxTokens,
        ReasoningFamily::DeepSeek | ReasoningFamily::Gemini | ReasoningFamily::None => {
            ReasoningParamShape::None
        }
    }
}

/// Inverse effort lookup: the nearest effort whose mapped budget is <=
/// `budget`; `Effort::Minimal` when `budget` is below every mapped budget.
fn inverse_effort(budget: u32, map: &EffortBudgetMap) -> Effort {
    let mut best: Option<(u32, Effort)> = None;
    for effort in Effort::ALL {
        let effort_budget = map.budget_for(effort);
        if effort_budget <= budget {
            let better = match best {
                Some((best_budget, _)) => effort_budget > best_budget,
                None => true,
            };
            if better {
                best = Some((effort_budget, effort));
            }
        }
    }
    best.map(|(_, effort)| effort).unwrap_or(Effort::Minimal)
}

/// Emit `reasoning_effort: "..."` (OpenAI / xAI targets).
fn emit_reasoning_effort(
    outgoing: &mut OpenAIRequest,
    spec: ReasoningSpec,
    cfg: &ReasoningCompatConfig,
) -> NormalizeReport {
    let effort = match spec.effort {
        Some(effort) => effort,
        None => match spec.budget {
            Some(budget) => inverse_effort(budget, &cfg.effort_budget_map),
            None => {
                remove_reasoning_params(outgoing, ReasoningParamShape::None);
                return NormalizeReport::none();
            }
        },
    };

    outgoing
        .extra
        .insert("reasoning_effort".to_string(), json!(effort.to_string()));
    remove_reasoning_params(outgoing, ReasoningParamShape::ReasoningEffort);

    NormalizeReport {
        emitted_shape: "reasoning_effort",
        ..NormalizeReport::none()
    }
}

/// Emit `thinking: {type:"enabled", budget_tokens: N}` (Anthropic manual
/// targets) with the 1024 floor and `budget < max_tokens` constraints
/// enforced (requirements 3.1 / 3.6).
fn emit_thinking_budget(
    outgoing: &mut OpenAIRequest,
    spec: ReasoningSpec,
    cfg: &ReasoningCompatConfig,
) -> NormalizeReport {
    let mut budget = match spec.budget {
        Some(budget) => budget,
        None => match spec.effort {
            Some(effort) => cfg.effort_budget_map.budget_for(effort),
            None => {
                remove_reasoning_params(outgoing, ReasoningParamShape::None);
                return NormalizeReport::none();
            }
        },
    };

    let mut clamped = false;
    if budget < MIN_REASONING_BUDGET_TOKENS {
        tracing::debug!(
            from = budget,
            to = MIN_REASONING_BUDGET_TOKENS,
            "clamping thinking budget_tokens up to the Anthropic minimum"
        );
        budget = MIN_REASONING_BUDGET_TOKENS;
        clamped = true;
    }

    let mut max_tokens_raised = false;
    if let Some(current_max_tokens) = read_max_tokens(outgoing) {
        if current_max_tokens <= budget {
            let raised = budget.saturating_add(1);
            tracing::debug!(
                from = current_max_tokens,
                to = raised,
                budget_tokens = budget,
                "raising max_tokens above the thinking budget"
            );
            write_max_tokens(outgoing, raised);
            max_tokens_raised = true;
        }
    }

    outgoing.extra.insert(
        "thinking".to_string(),
        json!({"type": "enabled", "budget_tokens": budget}),
    );
    let sampling_dropped = drop_sampling_params(outgoing);
    remove_reasoning_params(outgoing, ReasoningParamShape::ThinkingBudget);

    NormalizeReport {
        emitted_shape: "thinking_enabled",
        clamped,
        sampling_dropped,
        max_tokens_raised,
    }
}

/// Emit `thinking: {type:"adaptive"}` (+ `output_config.effort` when the
/// client effort level is known) for Anthropic adaptive targets. Never
/// emits `type:"enabled"` — Claude 4.7+ reject it with a 400 (requirement
/// 3.2).
fn emit_adaptive(outgoing: &mut OpenAIRequest, spec: ReasoningSpec) -> NormalizeReport {
    outgoing
        .extra
        .insert("thinking".to_string(), json!({"type": "adaptive"}));
    if let Some(effort) = spec.effort {
        outgoing.extra.insert(
            "output_config".to_string(),
            json!({"effort": effort.to_string()}),
        );
    } else {
        outgoing.extra.remove("output_config");
    }
    let sampling_dropped = drop_sampling_params(outgoing);
    remove_reasoning_params(outgoing, ReasoningParamShape::Adaptive);

    NormalizeReport {
        emitted_shape: "thinking_adaptive",
        sampling_dropped,
        ..NormalizeReport::none()
    }
}

/// Emit `reasoning: {max_tokens: N}` (OpenRouter targets).
fn emit_reasoning_max_tokens(
    outgoing: &mut OpenAIRequest,
    spec: ReasoningSpec,
    cfg: &ReasoningCompatConfig,
) -> NormalizeReport {
    let budget = match spec.budget {
        Some(budget) => budget,
        None => match spec.effort {
            Some(effort) => cfg.effort_budget_map.budget_for(effort),
            None => {
                remove_reasoning_params(outgoing, ReasoningParamShape::None);
                return NormalizeReport::none();
            }
        },
    };

    outgoing
        .extra
        .insert("reasoning".to_string(), json!({"max_tokens": budget}));
    remove_reasoning_params(outgoing, ReasoningParamShape::ReasoningMaxTokens);

    NormalizeReport {
        emitted_shape: "reasoning_max_tokens",
        ..NormalizeReport::none()
    }
}

/// Remove every reasoning parameter except the ones belonging to `keep`'s
/// shape, so a foreign-shape parameter never leaks through to the target
/// (requirement 3.3). Also drops `output_config` (the adaptive-era effort
/// container) for every shape that does not emit it.
fn remove_reasoning_params(outgoing: &mut OpenAIRequest, keep: ReasoningParamShape) {
    for key in ["reasoning_effort", "thinking", "reasoning", "output_config"] {
        let kept = match keep {
            ReasoningParamShape::ReasoningEffort => key == "reasoning_effort",
            ReasoningParamShape::ThinkingBudget => key == "thinking",
            ReasoningParamShape::Adaptive => key == "thinking" || key == "output_config",
            ReasoningParamShape::ReasoningMaxTokens => key == "reasoning",
            ReasoningParamShape::None => false,
        };
        if !kept {
            outgoing.extra.remove(key);
        }
    }
}

/// Drop sampling parameters Anthropic rejects alongside thinking
/// (requirement 3.8). Returns whether anything was actually dropped.
fn drop_sampling_params(outgoing: &mut OpenAIRequest) -> bool {
    let mut dropped = outgoing.temperature.is_some();
    outgoing.temperature = None;
    for key in ["temperature", "top_p", "top_k"] {
        dropped |= outgoing.extra.remove(key).is_some();
    }
    dropped
}

/// Read the request's `max_tokens` — the typed field first, then a
/// `max_tokens` entry in `extra`.
fn read_max_tokens(outgoing: &OpenAIRequest) -> Option<u32> {
    outgoing.max_tokens.or_else(|| json_u32(outgoing.extra.get("max_tokens")))
}

/// Write `max_tokens` back to wherever it was read from, so serialization
/// never produces a duplicate key.
fn write_max_tokens(outgoing: &mut OpenAIRequest, value: u32) {
    if outgoing.max_tokens.is_some() {
        outgoing.max_tokens = Some(value);
    } else {
        outgoing
            .extra
            .insert("max_tokens".to_string(), json!(value));
    }
}

/// Coerce a JSON number into a `u32` (negative / fractional / overflowing
/// values yield `None`).
fn json_u32(value: Option<&Value>) -> Option<u32> {
    value?.as_u64().and_then(|n| u32::try_from(n).ok())
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::Map;
    use std::collections::HashMap;

    fn request() -> OpenAIRequest {
        OpenAIRequest {
            model: "test-model".to_string(),
            messages: Vec::new(),
            stream: false,
            temperature: None,
            max_tokens: None,
            extra: Map::new(),
        }
    }

    fn target(model: &str, family: Option<ReasoningFamily>, shape: Option<ReasoningParamShape>) -> ProviderModel {
        ProviderModel {
            provider: "test-provider".to_string(),
            model: model.to_string(),
            cost_per_million_input_tokens: 0.0,
            cost_per_million_output_tokens: 0.0,
            priority: 100,
            structured_output_passthrough: None,
            tier: None,
            context_window: 0,
            specializations: Vec::new(),
            cost_per_million_cache_read_input_tokens: None,
            cost_per_million_cache_creation_input_tokens: None,
            cache_min_tokens: None,
            cache_support: None,
            cost_per_million_reasoning_tokens: None,
            reasoning_family: family,
            reasoning_parameter: shape,
        }
    }

    fn custom_map(minimal: u32) -> EffortBudgetMap {
        let budgets = [minimal, 2048, 8192, 16384, 32768];
        EffortBudgetMap(
            Effort::ALL
                .iter()
                .copied()
                .zip(budgets)
                .collect::<HashMap<_, _>>(),
        )
    }

    fn effort_spec(level: &str) -> (OpenAIRequest, ReasoningSpec) {
        let mut req = request();
        req.extra
            .insert("reasoning_effort".to_string(), json!(level));
        let spec = read_client_spec(&req);
        (req, spec)
    }

    // ---- read_client_spec ----

    #[test]
    fn reads_reasoning_effort() {
        let (req, spec) = effort_spec("high");
        assert_eq!(
            spec,
            ReasoningSpec {
                effort: Some(Effort::High),
                budget: None,
                adaptive: false
            }
        );
        assert!(req.extra.contains_key("reasoning_effort"));
    }

    #[test]
    fn reads_thinking_enabled_budget() {
        let mut req = request();
        req.extra.insert(
            "thinking".to_string(),
            json!({"type": "enabled", "budget_tokens": 5000}),
        );
        assert_eq!(
            read_client_spec(&req),
            ReasoningSpec {
                effort: None,
                budget: Some(5000),
                adaptive: false
            }
        );
    }

    #[test]
    fn reads_thinking_adaptive_with_output_config_effort() {
        let mut req = request();
        req.extra
            .insert("thinking".to_string(), json!({"type": "adaptive"}));
        req.extra
            .insert("output_config".to_string(), json!({"effort": "low"}));
        assert_eq!(
            read_client_spec(&req),
            ReasoningSpec {
                effort: Some(Effort::Low),
                budget: None,
                adaptive: true
            }
        );
    }

    #[test]
    fn reads_openrouter_reasoning_max_tokens() {
        let mut req = request();
        req.extra
            .insert("reasoning".to_string(), json!({"max_tokens": 4096}));
        assert_eq!(
            read_client_spec(&req),
            ReasoningSpec {
                effort: None,
                budget: Some(4096),
                adaptive: false
            }
        );
    }

    #[test]
    fn reads_openrouter_reasoning_effort_field() {
        let mut req = request();
        req.extra
            .insert("reasoning".to_string(), json!({"effort": "medium"}));
        assert_eq!(
            read_client_spec(&req),
            ReasoningSpec {
                effort: Some(Effort::Medium),
                budget: None,
                adaptive: false
            }
        );
    }

    #[test]
    fn reads_nothing_when_absent() {
        assert_eq!(
            read_client_spec(&request()),
            ReasoningSpec {
                effort: None,
                budget: None,
                adaptive: false
            }
        );
    }

    #[test]
    fn thinking_takes_precedence_over_reasoning_effort() {
        let mut req = request();
        req.extra
            .insert("reasoning_effort".to_string(), json!("high"));
        req.extra.insert(
            "thinking".to_string(),
            json!({"type": "enabled", "budget_tokens": 2048}),
        );
        assert_eq!(
            read_client_spec(&req),
            ReasoningSpec {
                effort: None,
                budget: Some(2048),
                adaptive: false
            }
        );
    }

    #[test]
    fn unknown_effort_string_yields_empty_spec() {
        let mut req = request();
        req.extra
            .insert("reasoning_effort".to_string(), json!("bogus"));
        assert!(read_client_spec(&req).is_empty());
    }

    // ---- family matrix: reasoning_effort:"high" into every family ----

    #[test]
    fn effort_high_to_anthropic_manual_emits_default_high_budget() {
        let (mut req, spec) = effort_spec("high");
        req.temperature = Some(0.7);
        req.extra.insert("top_p".to_string(), json!(0.9));
        req.extra.insert("top_k".to_string(), json!(40));
        let target = target("claude-sonnet-4-5", None, None);
        let report = emit_for_target(&mut req, spec, &target, &ReasoningCompatConfig::default());

        assert_eq!(report.emitted_shape, "thinking_enabled");
        assert!(!report.clamped);
        assert!(report.sampling_dropped);
        assert!(!report.max_tokens_raised);
        assert_eq!(
            req.extra.get("thinking"),
            Some(&json!({"type": "enabled", "budget_tokens": 16384}))
        );
        assert!(req.temperature.is_none());
        assert!(!req.extra.contains_key("top_p"));
        assert!(!req.extra.contains_key("top_k"));
        assert!(!req.extra.contains_key("reasoning_effort"));
        assert!(!req.extra.contains_key("reasoning"));
    }

    #[test]
    fn effort_high_to_anthropic_adaptive_never_emits_enabled() {
        let (mut req, spec) = effort_spec("high");
        req.temperature = Some(0.7);
        let target = target("claude-sonnet-4-7", None, None);
        let report = emit_for_target(&mut req, spec, &target, &ReasoningCompatConfig::default());

        assert_eq!(report.emitted_shape, "thinking_adaptive");
        assert!(report.sampling_dropped);
        assert_eq!(
            req.extra.get("thinking"),
            Some(&json!({"type": "adaptive"}))
        );
        assert_eq!(
            req.extra.get("output_config"),
            Some(&json!({"effort": "high"}))
        );
        assert!(req.temperature.is_none());
        assert!(!req.extra.contains_key("reasoning_effort"));
        assert!(!req.extra.contains_key("reasoning"));
    }

    #[test]
    fn effort_high_to_openai_preserves_reasoning_effort() {
        let (mut req, spec) = effort_spec("high");
        let target = target("gpt-5.1", None, None);
        let report = emit_for_target(&mut req, spec, &target, &ReasoningCompatConfig::default());

        assert_eq!(report.emitted_shape, "reasoning_effort");
        assert_eq!(
            req.extra.get("reasoning_effort"),
            Some(&json!("high"))
        );
        assert!(!req.extra.contains_key("thinking"));
        assert!(!req.extra.contains_key("reasoning"));
    }

    #[test]
    fn effort_high_to_openrouter_emits_reasoning_max_tokens() {
        let (mut req, spec) = effort_spec("high");
        let target = target("openrouter/gpt-5.1", None, None);
        let report = emit_for_target(&mut req, spec, &target, &ReasoningCompatConfig::default());

        assert_eq!(report.emitted_shape, "reasoning_max_tokens");
        assert_eq!(
            req.extra.get("reasoning"),
            Some(&json!({"max_tokens": 16384}))
        );
        assert!(!req.extra.contains_key("reasoning_effort"));
        assert!(!req.extra.contains_key("thinking"));
    }

    // ---- clamps and max_tokens raising ----

    #[test]
    fn budget_below_floor_is_clamped_to_1024() {
        let mut req = request();
        req.extra.insert(
            "thinking".to_string(),
            json!({"type": "enabled", "budget_tokens": 500}),
        );
        let spec = read_client_spec(&req);
        let target = target("claude-sonnet-4-5", None, None);
        let report = emit_for_target(&mut req, spec, &target, &ReasoningCompatConfig::default());

        assert_eq!(report.emitted_shape, "thinking_enabled");
        assert!(report.clamped);
        assert_eq!(
            req.extra.get("thinking"),
            Some(&json!({"type": "enabled", "budget_tokens": 1024}))
        );
    }

    #[test]
    fn budget_at_or_above_max_tokens_raises_max_tokens() {
        let mut req = request();
        req.max_tokens = Some(4096);
        req.extra.insert(
            "thinking".to_string(),
            json!({"type": "enabled", "budget_tokens": 5000}),
        );
        let spec = read_client_spec(&req);
        let target = target("claude-sonnet-4-5", None, None);
        let report = emit_for_target(&mut req, spec, &target, &ReasoningCompatConfig::default());

        assert_eq!(report.emitted_shape, "thinking_enabled");
        assert!(report.max_tokens_raised);
        assert_eq!(req.max_tokens, Some(5001));
        assert_eq!(
            req.extra.get("thinking"),
            Some(&json!({"type": "enabled", "budget_tokens": 5000}))
        );
    }

    #[test]
    fn budget_just_below_max_tokens_leaves_max_tokens_alone() {
        let mut req = request();
        req.max_tokens = Some(4097);
        req.extra.insert(
            "thinking".to_string(),
            json!({"type": "enabled", "budget_tokens": 4096}),
        );
        let spec = read_client_spec(&req);
        let target = target("claude-sonnet-4-5", None, None);
        let report = emit_for_target(&mut req, spec, &target, &ReasoningCompatConfig::default());

        assert!(!report.max_tokens_raised);
        assert_eq!(req.max_tokens, Some(4097));
    }

    #[test]
    fn extra_max_tokens_is_honored_and_raised_in_place() {
        let mut req = request();
        req.extra.insert("max_tokens".to_string(), json!(2048));
        req.extra.insert(
            "thinking".to_string(),
            json!({"type": "enabled", "budget_tokens": 4096}),
        );
        let spec = read_client_spec(&req);
        let target = target("claude-sonnet-4-5", None, None);
        let report = emit_for_target(&mut req, spec, &target, &ReasoningCompatConfig::default());

        assert!(report.max_tokens_raised);
        assert!(req.max_tokens.is_none());
        assert_eq!(req.extra.get("max_tokens"), Some(&json!(4097)));
    }

    // ---- inverse map (budget → effort) ----

    #[test]
    fn openrouter_budget_maps_to_openai_effort_via_inverse_map() {
        let mut req = request();
        req.extra
            .insert("reasoning".to_string(), json!({"max_tokens": 4096}));
        let spec = read_client_spec(&req);
        let target = target("gpt-5.1", None, None);
        let report = emit_for_target(&mut req, spec, &target, &ReasoningCompatConfig::default());

        assert_eq!(report.emitted_shape, "reasoning_effort");
        // Default budgets: minimal 1024, low 2048, medium 8192 — the
        // nearest effort whose budget is <= 4096 is "low".
        assert_eq!(req.extra.get("reasoning_effort"), Some(&json!("low")));
        assert!(!req.extra.contains_key("reasoning"));
    }

    #[test]
    fn inverse_map_exact_and_boundary_values() {
        let map = EffortBudgetMap::default();
        assert_eq!(inverse_effort(8192, &map), Effort::Medium);
        assert_eq!(inverse_effort(8191, &map), Effort::Low);
        assert_eq!(inverse_effort(1024, &map), Effort::Minimal);
        assert_eq!(inverse_effort(1023, &map), Effort::Minimal);
        assert_eq!(inverse_effort(500, &map), Effort::Minimal);
        assert_eq!(inverse_effort(32768, &map), Effort::XHigh);
        assert_eq!(inverse_effort(u32::MAX, &map), Effort::XHigh);
    }

    #[test]
    fn anthropic_budget_to_openrouter_target_round_trips_through_effort() {
        let mut req = request();
        req.extra.insert(
            "thinking".to_string(),
            json!({"type": "enabled", "budget_tokens": 16384}),
        );
        let spec = read_client_spec(&req);
        let target = target("openrouter/gpt-5.1", None, None);
        let report = emit_for_target(&mut req, spec, &target, &ReasoningCompatConfig::default());

        assert_eq!(report.emitted_shape, "reasoning_max_tokens");
        assert_eq!(
            req.extra.get("reasoning"),
            Some(&json!({"max_tokens": 16384}))
        );
        assert!(!req.extra.contains_key("thinking"));
    }

    // ---- no-injection rule ----

    #[test]
    fn no_reasoning_params_emits_nothing_for_any_family() {
        for (model, expected_shape) in [
            ("claude-sonnet-4-5", "none"),
            ("claude-sonnet-4-7", "none"),
            ("gpt-5.1", "none"),
            ("openrouter/gpt-5.1", "none"),
            ("llama-3", "none"),
        ] {
            let mut req = request();
            req.temperature = Some(0.5);
            let spec = read_client_spec(&req);
            assert!(spec.is_empty());
            let target = target(model, None, None);
            let report =
                emit_for_target(&mut req, spec, &target, &ReasoningCompatConfig::default());
            assert_eq!(report, NormalizeReport::none(), "model {model}");
            assert_eq!(req.temperature, Some(0.5), "model {model}");
            assert!(req.extra.is_empty(), "model {model}");
            assert_eq!(report.emitted_shape, expected_shape);
        }
    }

    // ---- no-reasoning-support target ----

    #[test]
    fn none_shape_target_strips_all_reasoning_params() {
        let (mut req, spec) = effort_spec("high");
        req.extra
            .insert("reasoning".to_string(), json!({"max_tokens": 9999}));
        req.extra
            .insert("output_config".to_string(), json!({"effort": "high"}));
        req.temperature = Some(0.7);
        let target = target("some-model", Some(ReasoningFamily::None), Some(ReasoningParamShape::None));
        let report = emit_for_target(&mut req, spec, &target, &ReasoningCompatConfig::default());

        assert_eq!(report, NormalizeReport::none());
        assert!(!req.extra.contains_key("reasoning_effort"));
        assert!(!req.extra.contains_key("thinking"));
        assert!(!req.extra.contains_key("reasoning"));
        assert!(!req.extra.contains_key("output_config"));
        assert_eq!(req.temperature, Some(0.7));
    }

    // ---- configuration overrides ----

    #[test]
    fn custom_effort_map_is_honored() {
        let (mut req, spec) = effort_spec("minimal");
        let target = target("claude-sonnet-4-5", None, None);
        let mut cfg = ReasoningCompatConfig::default();
        cfg.effort_budget_map = custom_map(2048);
        let report = emit_for_target(&mut req, spec, &target, &cfg);

        assert_eq!(report.emitted_shape, "thinking_enabled");
        assert!(!report.clamped);
        assert_eq!(
            req.extra.get("thinking"),
            Some(&json!({"type": "enabled", "budget_tokens": 2048}))
        );
    }

    #[test]
    fn custom_effort_map_changes_inverse_lookup() {
        let mut req = request();
        req.extra
            .insert("reasoning".to_string(), json!({"max_tokens": 4096}));
        let spec = read_client_spec(&req);
        let target = target("gpt-5.1", None, None);
        let mut cfg = ReasoningCompatConfig::default();
        cfg.effort_budget_map = custom_map(4096);
        let report = emit_for_target(&mut req, spec, &target, &cfg);

        assert_eq!(report.emitted_shape, "reasoning_effort");
        assert_eq!(req.extra.get("reasoning_effort"), Some(&json!("minimal")));
    }

    // ---- target resolution ----

    #[test]
    fn explicit_shape_override_beats_family_default() {
        let (mut req, spec) = effort_spec("high");
        let target = target(
            "custom-gw-model",
            Some(ReasoningFamily::None),
            Some(ReasoningParamShape::ReasoningEffort),
        );
        let report = emit_for_target(&mut req, spec, &target, &ReasoningCompatConfig::default());

        assert_eq!(report.emitted_shape, "reasoning_effort");
        assert_eq!(req.extra.get("reasoning_effort"), Some(&json!("high")));
    }

    #[test]
    fn explicit_family_beats_model_id_classification() {
        let (mut req, spec) = effort_spec("high");
        let target = target(
            "gpt-5.1",
            Some(ReasoningFamily::AnthropicAdaptive),
            None,
        );
        let report = emit_for_target(&mut req, spec, &target, &ReasoningCompatConfig::default());

        assert_eq!(report.emitted_shape, "thinking_adaptive");
        assert_eq!(
            req.extra.get("thinking"),
            Some(&json!({"type": "adaptive"}))
        );
    }

    #[test]
    fn family_defaults_derive_from_model_id() {
        let (req, spec) = effort_spec("medium");
        for (model, expected) in [
            ("claude-opus-4-1", "thinking_enabled"),
            ("claude-opus-5", "thinking_adaptive"),
            ("o3-mini", "reasoning_effort"),
            ("grok-4", "reasoning_effort"),
            ("openrouter/anthropic/claude-sonnet-4", "reasoning_max_tokens"),
            ("deepseek-r1", "none"),
        ] {
            let mut cloned = req.clone();
            let target = target(model, None, None);
            let report =
                emit_for_target(&mut cloned, spec, &target, &ReasoningCompatConfig::default());
            assert_eq!(report.emitted_shape, expected, "model {model}");
        }
    }

    // ---- adaptive-spec edge cases ----

    #[test]
    fn adaptive_only_spec_to_manual_target_emits_nothing() {
        let mut req = request();
        req.extra
            .insert("thinking".to_string(), json!({"type": "adaptive"}));
        let spec = read_client_spec(&req);
        assert!(spec.adaptive);
        assert!(!spec.is_empty());
        let target = target("claude-sonnet-4-5", None, None);
        let report = emit_for_target(&mut req, spec, &target, &ReasoningCompatConfig::default());

        assert_eq!(report.emitted_shape, "none");
        assert!(!req.extra.contains_key("thinking"));
    }

    #[test]
    fn adaptive_spec_with_effort_to_manual_target_uses_effort_map() {
        let mut req = request();
        req.extra
            .insert("thinking".to_string(), json!({"type": "adaptive"}));
        req.extra
            .insert("output_config".to_string(), json!({"effort": "low"}));
        let spec = read_client_spec(&req);
        let target = target("claude-sonnet-4-5", None, None);
        let report = emit_for_target(&mut req, spec, &target, &ReasoningCompatConfig::default());

        assert_eq!(report.emitted_shape, "thinking_enabled");
        assert_eq!(
            req.extra.get("thinking"),
            Some(&json!({"type": "enabled", "budget_tokens": 2048}))
        );
        assert!(!req.extra.contains_key("output_config"));
    }

    #[test]
    fn adaptive_target_with_budget_only_spec_omits_output_config() {
        let mut req = request();
        req.extra
            .insert("reasoning".to_string(), json!({"max_tokens": 8192}));
        let spec = read_client_spec(&req);
        let target = target("claude-sonnet-4-7", None, None);
        let report = emit_for_target(&mut req, spec, &target, &ReasoningCompatConfig::default());

        assert_eq!(report.emitted_shape, "thinking_adaptive");
        assert_eq!(
            req.extra.get("thinking"),
            Some(&json!({"type": "adaptive"}))
        );
        assert!(!req.extra.contains_key("output_config"));
        assert!(!req.extra.contains_key("reasoning"));
    }

    #[test]
    fn enabled_never_reaches_adaptive_target_even_via_override() {
        let (mut req, spec) = effort_spec("xhigh");
        let target = target(
            "claude-sonnet-4-7",
            None,
            Some(ReasoningParamShape::Adaptive),
        );
        let report = emit_for_target(&mut req, spec, &target, &ReasoningCompatConfig::default());

        assert_eq!(report.emitted_shape, "thinking_adaptive");
        assert_eq!(
            req.extra.get("thinking"),
            Some(&json!({"type": "adaptive"}))
        );
        assert_eq!(
            req.extra.get("output_config"),
            Some(&json!({"effort": "xhigh"}))
        );
    }

    #[test]
    fn stale_output_config_is_replaced_not_merged_on_adaptive() {
        let mut req = request();
        req.extra
            .insert("thinking".to_string(), json!({"type": "adaptive"}));
        req.extra.insert(
            "output_config".to_string(),
            json!({"effort": "low", "extra_key": 1}),
        );
        let spec = read_client_spec(&req);
        let target = target("claude-sonnet-4-7", None, None);
        let _ = emit_for_target(&mut req, spec, &target, &ReasoningCompatConfig::default());

        assert_eq!(
            req.extra.get("output_config"),
            Some(&json!({"effort": "low"}))
        );
    }
}
