//! Guardrail pipelines: configurable, opt-in policy enforcement for the gateway.
//!
//! This module tree adds pre-call and post-call policy evaluation to the
//! request lifecycle. It is organized as:
//!
//! - [`config`]      — configuration data model (`guardrails` section).
//! - `provider`      — `GuardrailProvider` trait, `Finding`, registry (task 2).
//! - `pipeline`      — binding resolution and stage ordering (task 9).
//! - `pii`           — PII placeholder generation and re-injection (task 4).
//! - `stream`        — SSE buffering and re-chunking for post-call (task 13).
//! - [`providers`]   — provider backend implementations.
//!
//! Only [`config`] and [`providers`] exist so far; the remaining submodules are
//! declared as work lands in later tasks to keep the crate compiling.

pub mod config;
pub mod factory;
pub mod pii;
pub mod pipeline;
pub mod provider;
pub mod providers;
pub mod refusal;
pub mod stream;

// Placeholder public exports. The configuration data model is the only stable
// surface at this stage; engine/context/provider exports are added by later
// tasks (e.g. `GuardrailEngine`, `GuardrailContext`, `GuardrailProvider`).
// These re-exports form the guardrail module's public API and are consumed by
// the library's integration tests and dependents. The binary target
// (`main.rs`, which re-declares `mod guardrail;`) does not use every item, so
// its unused-import lint would otherwise fire on the API surface it doesn't
// touch; allow it here.
#[allow(unused_imports)]
pub use config::{
    default_provider_timeout_secs, FailurePolicy, GuardrailBindings, GuardrailConfig,
    GuardrailProviderConfig, GuardrailProviderType, InstructionInsertionMode, PipelineConfig,
    PolicyAction, ProviderSettings, RegexPatternConfig, RegexRuleMode, StageConfig, StagePhase,
};
#[allow(unused_imports)]
pub use factory::{build_engine, build_registry, RegistryBuildError};
#[allow(unused_imports)]
pub use pii::{
    inject_redaction_notice, mask, GuardrailContext, PlaceholderResult,
    DEFAULT_REDACTION_NOTICE_INSTRUCTION, MAX_CONFIGURABLE_REINJECTION_ENTRIES,
    MAX_REINJECTION_ENTRIES, MIN_REINJECTION_ENTRIES, PRESERVE_PLACEHOLDERS_INSTRUCTION,
};
#[allow(unused_imports)]
pub use pipeline::{
    BindingSelector, PipelineResolver, PipelineResolverError, ResolvedPipeline, ResolvedStage,
};
#[allow(unused_imports)]
pub use provider::{
    analyze_with_policy, Finding, GuardrailProvider, GuardrailProviderError, ProviderRegistry,
    StageDisposition,
};
// The streaming (SSE) buffering support consumed by the streaming handler
// (task 13.3) is reachable via the `stream` submodule path
// (`crate::guardrail::stream::{SseBuffer, block_frame_payload, ...}`); the
// handler imports it as `stream as guardrail_stream`.
#[allow(unused_imports)]
pub use refusal::{
    RefusalBuildError, RefusalDecision, RefusalDetector, RefusalSignal, ToolContext,
    DEFAULT_REFUSAL_PHRASES,
};

// ---------------------------------------------------------------------------
// Guardrail engine (task 10.1)
// ---------------------------------------------------------------------------
//
// The engine is the central integration point that executes resolved stages
// against a request (pre-call) and a response (post-call). It owns a
// pre-compiled [`PipelineResolver`] and applies per-stage policy actions with
// deterministic ordering and short-circuit behavior (Req 9), content clamping
// (Req 8.1), message-content extraction across roles/shapes (Req 2.5, 3.5),
// failure-policy handling (Req 8.6, 9.6, 9.7), and PII re-injection as the final
// non-halting post-call step (Req 9.5).
//
// Metric/log emission is deliberately deferred to task 12.2: the engine holds an
// optional `Arc<Metrics>` handle but performs no recording yet, so enforcement
// is fully functional without blocking on the observability wiring.

use std::sync::Arc;
use std::time::Instant;

use serde_json::Value;

// Guardrail types (`PolicyAction`, `StagePhase`, `FailurePolicy`,
// `GuardrailContext`, `mask`, `BindingSelector`, `PipelineResolver`,
// `ResolvedStage`, `Finding`, `GuardrailConfig`, `ProviderRegistry`,
// `PipelineResolverError`) are already in scope via the `pub use` re-exports
// above. Only the external types the engine depends on are imported here.
use crate::metrics::Metrics;
use crate::models::openai::{Message, OpenAIRequest, OpenAIResponse};

/// Maximum UTF-8 characters of content submitted to a provider (Req 8.1).
pub const DEFAULT_MAX_CONTENT_CHARS: usize = 100_000;

/// Default policy message substituted for a `replace_with_policy_message`
/// post-call action (Req 3.3).
///
/// `StageConfig` does not currently carry a per-stage policy message, so the
/// engine uses this fixed default. A configurable message would require a new
/// config field wired through in a later task; this keeps the engine functional
/// for enforcement today.
pub const DEFAULT_POLICY_MESSAGE: &str =
    "This content was replaced because it violated the configured content policy.";

/// A halting policy decision carrying only the triggering entity/category so a
/// 403 response body can identify *why* without leaking the original content
/// (Req 2.2, and content-safe logging per Req 11.3).
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct GuardrailBlock {
    /// Pipeline that produced the block (label only).
    pub pipeline_name: String,
    /// Stage that produced the block (label only).
    pub stage_name: String,
    /// Entity/category label of the triggering finding (never the raw value).
    pub entity_label: String,
    /// Phase in which the block occurred.
    pub phase: StagePhase,
}

/// Outcome of [`GuardrailEngine::run_pre_call`].
///
/// The handler (task 13.2) maps these to HTTP responses:
/// `Proceed` → forward to the router; `Block` → 403 policy violation;
/// `InvalidAction` → 400; `Timeout` → 503 scan timeout; `ServiceFailure` → 503
/// guardrail service failure.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum PreCallOutcome {
    /// `allow`/`redact`/`mask` applied in place; forward the (possibly mutated)
    /// request to the router.
    Proceed,
    /// A `block` action fired: return HTTP 403 without forwarding (Req 2.2, 9.2).
    Block(GuardrailBlock),
    /// A stage declared an action invalid for the pre-call phase (e.g.
    /// `replace_with_policy_message`): return HTTP 400 (Req 2.7).
    InvalidAction,
    /// A `fail_close` provider exceeded its scan latency budget: return HTTP 503
    /// scan timeout (Req 2.9).
    Timeout,
    /// A `fail_close` provider errored: return HTTP 503 guardrail service
    /// failure (Req 9.7, 8.6).
    ServiceFailure,
}

/// Outcome of [`GuardrailEngine::run_post_call`].
///
/// `Proceed` → return the (possibly redacted + re-injected) response;
/// `Block` → 403; `Replaced` → 200 with the policy message; `ServiceFailure`
/// → 503.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum PostCallOutcome {
    /// `allow`/`redact` applied and re-injection completed; return the response
    /// (HTTP 200).
    Proceed,
    /// A `block` action fired: discard the response, return HTTP 403 (Req 3.1,
    /// 9.4).
    Block(GuardrailBlock),
    /// A `replace_with_policy_message` action fired: assistant content replaced,
    /// return HTTP 200 (Req 3.3, 9.4). Re-injection is skipped.
    Replaced,
    /// A `fail_close` provider errored or timed out: return HTTP 503 (Req 9.7).
    ServiceFailure,
}

/// Central guardrail execution engine (Req 9).
///
/// Constructed once from a [`GuardrailConfig`] and its [`ProviderRegistry`],
/// held behind an `Arc` on `AppState`, and swapped wholesale on hot-reload so
/// in-flight requests keep their snapshot (Req 1.8; wiring is task 13.1).
pub struct GuardrailEngine {
    /// Pre-compiled binding resolver producing ordered stage lists.
    resolver: PipelineResolver,
    /// Metrics handle. When `Some`, per-stage counter/latency metrics are
    /// recorded best-effort (Req 11.1, 11.2, 11.7).
    metrics: Option<Arc<Metrics>>,
    /// Content clamp applied before provider analysis (Req 8.1).
    max_content_chars: usize,
    /// Message substituted for `replace_with_policy_message` (Req 3.3).
    policy_message: String,
    /// Compiled refusal detector for post-call refusal detection (Req 12.1, 12.12).
    refusal_detector: RefusalDetector,
    /// Maximum re-injection entries per request (configurable).
    max_reinjection_entries: usize,
}

impl std::fmt::Debug for GuardrailEngine {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("GuardrailEngine")
            .field("resolver", &self.resolver)
            .field("has_metrics", &self.metrics.is_some())
            .field("max_content_chars", &self.max_content_chars)
            .field("max_reinjection_entries", &self.max_reinjection_entries)
            .field("refusal_detector", &self.refusal_detector)
            .finish()
    }
}

impl GuardrailEngine {
    /// Build an engine by compiling `config` against `registry`.
    ///
    /// Returns [`PipelineResolverError`] if a stage references a provider absent
    /// from the registry or the provider config list (configuration validation
    /// is expected to have already rejected such cases).
    #[allow(dead_code)] // public API; used by tests and external callers
    pub fn new(
        config: &GuardrailConfig,
        registry: &ProviderRegistry,
        metrics: Option<Arc<Metrics>>,
    ) -> Result<Self, PipelineResolverError> {
        let resolver = PipelineResolver::new(config, registry)?;
        Ok(Self {
            resolver,
            metrics,
            max_content_chars: DEFAULT_MAX_CONTENT_CHARS,
            policy_message: DEFAULT_POLICY_MESSAGE.to_string(),
            refusal_detector: RefusalDetector::default_detector(),
            max_reinjection_entries: config.max_reinjection_entries,
        })
    }

    /// Build an engine with explicit re-injection capacity.
    pub fn new_with_capacity(
        config: &GuardrailConfig,
        registry: &ProviderRegistry,
        metrics: Option<Arc<Metrics>>,
        max_reinjection_entries: usize,
    ) -> Result<Self, PipelineResolverError> {
        let resolver = PipelineResolver::new(config, registry)?;
        Ok(Self {
            resolver,
            metrics,
            max_content_chars: DEFAULT_MAX_CONTENT_CHARS,
            policy_message: DEFAULT_POLICY_MESSAGE.to_string(),
            refusal_detector: RefusalDetector::default_detector(),
            max_reinjection_entries,
        })
    }

    /// Build an engine from an already-compiled [`PipelineResolver`].
    #[allow(dead_code)] // public API; used by tests and external callers
    pub fn with_resolver(resolver: PipelineResolver, metrics: Option<Arc<Metrics>>) -> Self {
        Self {
            resolver,
            metrics,
            max_content_chars: DEFAULT_MAX_CONTENT_CHARS,
            policy_message: DEFAULT_POLICY_MESSAGE.to_string(),
            refusal_detector: RefusalDetector::default_detector(),
            max_reinjection_entries: pii::MAX_REINJECTION_ENTRIES,
        }
    }

    /// The configured re-injection entry cap.
    #[allow(dead_code)] // public API; used by tests and external callers
    pub fn max_reinjection_entries(&self) -> usize {
        self.max_reinjection_entries
    }

    /// Borrow the underlying resolver (used by the handler and diagnostics).
    pub fn resolver(&self) -> &PipelineResolver {
        &self.resolver
    }

    /// Create a request-scoped guardrail context using this engine's
    /// configured re-injection capacity.
    pub fn new_context(&self) -> GuardrailContext {
        GuardrailContext::with_capacity(self.max_reinjection_entries)
    }

    /// The configured content clamp in UTF-8 characters.
    #[allow(dead_code)] // public accessor / test-only; unused in the binary build
    pub fn max_content_chars(&self) -> usize {
        self.max_content_chars
    }

    /// Override the content clamp (primarily for tests).
    #[allow(dead_code)] // used by tests; unused in the binary build
    pub fn set_max_content_chars(&mut self, max: usize) {
        self.max_content_chars = max;
    }

    /// Override the `replace_with_policy_message` substitution text.
    #[allow(dead_code)] // public API; unused in the binary build
    pub fn set_policy_message(&mut self, message: impl Into<String>) {
        self.policy_message = message.into();
    }

    /// Override the refusal detector (for pipelines that supply a custom phrase list).
    #[allow(dead_code)] // public API; unused in the binary build until task 17.2+
    pub fn set_refusal_detector(&mut self, detector: RefusalDetector) {
        self.refusal_detector = detector;
    }

    /// Borrow the refusal detector (used by the handler for re-dispatch loop).
    #[allow(dead_code)] // public API; consumed by handler in task 17.2
    pub fn refusal_detector(&self) -> &RefusalDetector {
        &self.refusal_detector
    }

    /// Record a refusal-failover outcome, best-effort (Req 12.11, 11.7).
    ///
    /// Called by the handler after the bounded re-dispatch loop settles.
    /// `pipeline` is the effective pipeline label, `outcome` ∈ {recovered,
    /// exhausted}, and `attempt_count` is the total number of provider targets
    /// attempted (including the original). Emits an INFO log and increments
    /// the `obey_api_guardrail_refusal_failover_total` counter. Never fails the
    /// request.
    #[allow(dead_code)] // public API; wired in handler failover loop
    pub fn record_refusal_failover(
        &self,
        pipeline: &str,
        outcome: &str,
        attempt_count: usize,
        trace_id: &str,
    ) {
        tracing::info!(
            target: "guardrail",
            pipeline = %pipeline,
            outcome = %outcome,
            attempt_count,
            trace_id = %trace_id,
            "refusal failover initiated"
        );
        if let Some(metrics) = &self.metrics {
            metrics.record_guardrail_refusal_failover(pipeline, outcome);
        }
    }

    /// Execute the pre-call stages bound to `selector` against `request`.
    ///
    /// Applies each stage's action to every message content field across all
    /// roles (Req 2.5), clamping content before analysis (Req 8.1). Halting
    /// actions short-circuit (Req 9.2). On success, if pre-call redaction
    /// populated `ctx`, a preserve-placeholders system instruction is prepended
    /// to the request (Req 4.4).
    pub async fn run_pre_call(
        &self,
        request: &mut OpenAIRequest,
        selector: &BindingSelector,
        ctx: &mut GuardrailContext,
        trace_id: &str,
    ) -> PreCallOutcome {
        let pipeline_start = Instant::now();

        let stages = self.resolver.resolve(selector);
        let pre_stages: Vec<&ResolvedStage> = stages
            .iter()
            .filter(|s| s.phase == StagePhase::PreCall)
            .collect();

        // Per-request guardrail summary accumulators (Req 11.4).
        let mut stages_executed = 0usize;
        let mut non_pass_actions: Vec<String> = Vec::new();
        // Set to the halting/terminal outcome; when `None` the pipeline passed.
        let mut terminal: Option<PreCallOutcome> = None;

        for stage in pre_stages {
            // An action invalid for the pre-call phase → HTTP 400 (Req 2.7).
            // No provider is executed, so this stage is not counted/metered.
            if stage.action == PolicyAction::ReplaceWithPolicyMessage {
                terminal = Some(PreCallOutcome::InvalidAction);
                break;
            }

            stages_executed += 1;
            let stage_start = Instant::now();

            // Per-stage observation state driving the single action label.
            let mut modified = false;
            let mut errored = false;
            let mut entity_for_log: Option<String> = None;
            let mut stage_block: Option<GuardrailBlock> = None;
            let mut stage_terminal: Option<PreCallOutcome> = None;

            // Scan every message content field regardless of role (Req 2.5).
            'fields: for message in request.messages.iter_mut() {
                let slots = collect_text_slots(&message.content);
                for (slot, text) in slots {
                    let clamped = clamp_content(&text, self.max_content_chars);
                    match evaluate_stage(stage, clamped).await {
                        StageAnalysis::Timeout => {
                            errored = true;
                            stage_terminal = Some(PreCallOutcome::Timeout);
                            break 'fields;
                        }
                        StageAnalysis::ServiceFailure => {
                            errored = true;
                            stage_terminal = Some(PreCallOutcome::ServiceFailure);
                            break 'fields;
                        }
                        // `fail_open` error/timeout: skip the field but still
                        // count it as a provider error for the counter (Req 11.6).
                        StageAnalysis::Skip => {
                            errored = true;
                            continue;
                        }
                        StageAnalysis::Findings(findings) => {
                            match apply_pre_action(stage.action, ctx, &text, &findings) {
                                FieldEffect::Pass => {}
                                FieldEffect::Modified(new_text) => {
                                    modified = true;
                                    if entity_for_log.is_none() {
                                        entity_for_log =
                                            findings.first().map(|f| f.entity_label.clone());
                                    }
                                    set_text_slot(&mut message.content, slot, new_text);
                                }
                                FieldEffect::Block(entity_label) => {
                                    entity_for_log = Some(entity_label.clone());
                                    stage_block = Some(GuardrailBlock {
                                        pipeline_name: stage.pipeline_name.clone(),
                                        stage_name: stage.stage_name.clone(),
                                        entity_label,
                                        phase: StagePhase::PreCall,
                                    });
                                    break 'fields;
                                }
                            }
                        }
                    }
                }
            }

            // Derive the single action label for this stage execution (Req 11.1).
            let action_label =
                derive_action_label(stage.action, stage_block.is_some(), errored, modified);
            let latency_ms = duration_ms(stage_start.elapsed());
            self.record_stage_metric(stage, action_label, latency_ms);
            self.log_stage_action(stage, action_label, entity_for_log.as_deref(), trace_id);
            if action_label != "pass" {
                non_pass_actions.push(action_label.to_string());
            }

            if let Some(t) = stage_terminal {
                terminal = Some(t);
                break;
            }
            if let Some(block) = stage_block {
                terminal = Some(PreCallOutcome::Block(block));
                break;
            }
        }

        // Req 4.4, 4.8–4.11, 4.13: inject the redaction-notice instruction iff
        // redaction recorded at least one Re_Injection_Map entry — only when proceeding.
        if terminal.is_none() {
            let (override_instruction, insertion_mode) =
                self.resolver.resolve_instruction_config(selector);
            inject_redaction_notice(
                &mut request.messages,
                ctx,
                override_instruction,
                insertion_mode,
            );
        }

        self.emit_summary(
            "pre_call",
            trace_id,
            stages_executed,
            &non_pass_actions,
            duration_ms(pipeline_start.elapsed()),
        );

        terminal.unwrap_or(PreCallOutcome::Proceed)
    }

    /// Execute the post-call stages bound to `selector` against `response`.
    ///
    /// Scans assistant-role message content (Req 3.5). `block` discards the
    /// response (Req 3.1); `replace_with_policy_message` rewrites assistant
    /// content (Req 3.3); both halt and skip re-injection (Req 9.4). `redact`
    /// replaces matched spans with `[REDACTED]` and continues (Req 3.2). After
    /// all stages complete without a halting action, refusal detection runs on
    /// assistant content (Req 12.12) and PII re-injection runs as the final
    /// step when no refusal-failover is triggered (Req 9.5).
    ///
    /// Returns `(PostCallOutcome, RefusalDecision)`. When the effective
    /// `failover_on_refusal` toggle is disabled (Req 12.6), the decision is
    /// always `NotRefusal` and the response is returned unmodified (re-injection
    /// still applies). When enabled and a refusal is detected, re-injection is
    /// skipped so the handler can re-dispatch before re-injecting on the final
    /// response.
    pub async fn run_post_call(
        &self,
        response: &mut OpenAIResponse,
        selector: &BindingSelector,
        ctx: &mut GuardrailContext,
        trace_id: &str,
        tool_context: &ToolContext,
    ) -> (PostCallOutcome, RefusalDecision) {
        let pipeline_start = Instant::now();

        let stages = self.resolver.resolve(selector);
        let post_stages: Vec<&ResolvedStage> = stages
            .iter()
            .filter(|s| s.phase == StagePhase::PostCall)
            .collect();

        let mut stages_executed = 0usize;
        let mut non_pass_actions: Vec<String> = Vec::new();
        let mut terminal: Option<PostCallOutcome> = None;

        for stage in post_stages {
            stages_executed += 1;
            let stage_start = Instant::now();

            let mut modified = false;
            let mut errored = false;
            let mut replaced = false;
            let mut entity_for_log: Option<String> = None;
            let mut stage_block: Option<GuardrailBlock> = None;
            let mut stage_terminal: Option<PostCallOutcome> = None;

            if stage.action == PolicyAction::ReplaceWithPolicyMessage {
                // Message-level halting action: if any assistant content field
                // yields a finding, replace all assistant content and halt.
                'scan: for choice in response.choices.iter() {
                    if !is_assistant(&choice.message) {
                        continue;
                    }
                    for (_slot, text) in collect_text_slots(&choice.message.content) {
                        let clamped = clamp_content(&text, self.max_content_chars);
                        match evaluate_stage(stage, clamped).await {
                            StageAnalysis::Timeout | StageAnalysis::ServiceFailure => {
                                errored = true;
                                stage_terminal = Some(PostCallOutcome::ServiceFailure);
                                break 'scan;
                            }
                            StageAnalysis::Skip => {
                                errored = true;
                                continue;
                            }
                            StageAnalysis::Findings(f) if f.is_empty() => continue,
                            StageAnalysis::Findings(f) => {
                                entity_for_log = f.first().map(|x| x.entity_label.clone());
                                replaced = true;
                                break 'scan;
                            }
                        }
                    }
                }
                if replaced {
                    for choice in response.choices.iter_mut() {
                        if is_assistant(&choice.message) {
                            choice.message.content = Value::String(self.policy_message.clone());
                        }
                    }
                    stage_terminal = Some(PostCallOutcome::Replaced);
                }
            } else {
                // Per-field actions (allow / block / mask / redact).
                'fields: for choice in response.choices.iter_mut() {
                    if !is_assistant(&choice.message) {
                        continue;
                    }
                    let slots = collect_text_slots(&choice.message.content);
                    for (slot, text) in slots {
                        let clamped = clamp_content(&text, self.max_content_chars);
                        match evaluate_stage(stage, clamped).await {
                            StageAnalysis::Timeout | StageAnalysis::ServiceFailure => {
                                errored = true;
                                stage_terminal = Some(PostCallOutcome::ServiceFailure);
                                break 'fields;
                            }
                            StageAnalysis::Skip => {
                                errored = true;
                                continue;
                            }
                            StageAnalysis::Findings(findings) => {
                                match apply_post_action(stage.action, &text, &findings) {
                                    FieldEffect::Pass => {}
                                    FieldEffect::Modified(new_text) => {
                                        modified = true;
                                        if entity_for_log.is_none() {
                                            entity_for_log =
                                                findings.first().map(|f| f.entity_label.clone());
                                        }
                                        set_text_slot(&mut choice.message.content, slot, new_text);
                                    }
                                    FieldEffect::Block(entity_label) => {
                                        entity_for_log = Some(entity_label.clone());
                                        stage_block = Some(GuardrailBlock {
                                            pipeline_name: stage.pipeline_name.clone(),
                                            stage_name: stage.stage_name.clone(),
                                            entity_label,
                                            phase: StagePhase::PostCall,
                                        });
                                        break 'fields;
                                    }
                                }
                            }
                        }
                    }
                }
            }

            // Derive the single action label for this stage execution (Req 11.1).
            let action_label = if stage_block.is_some() {
                "block"
            } else if errored && !replaced {
                "error"
            } else if replaced {
                "replace_with_policy_message"
            } else {
                derive_action_label(stage.action, false, false, modified)
            };
            let latency_ms = duration_ms(stage_start.elapsed());
            self.record_stage_metric(stage, action_label, latency_ms);
            self.log_stage_action(stage, action_label, entity_for_log.as_deref(), trace_id);
            if action_label != "pass" {
                non_pass_actions.push(action_label.to_string());
            }

            if let Some(t) = stage_terminal {
                terminal = Some(t);
                break;
            }
            if let Some(block) = stage_block {
                terminal = Some(PostCallOutcome::Block(block));
                break;
            }
        }

        // Req 12.12: run refusal detection as part of the post-call guardrail
        // stage, reusing the post-call observability model. Only meaningful when
        // the pipeline completed without a halting action.
        let refusal_decision = if terminal.is_none() {
            let failover_enabled = self.resolver.resolve_failover_on_refusal(selector);

            // Extract assistant content for phrase-based detection.
            let assistant_content = response
                .choices
                .iter()
                .filter(|c| is_assistant(&c.message))
                .filter_map(|c| {
                    if let Value::String(s) = &c.message.content {
                        Some(s.as_str())
                    } else {
                        None
                    }
                })
                .collect::<Vec<&str>>()
                .join(" ");

            let raw_decision = self
                .refusal_detector
                .detect(&assistant_content, tool_context);

            if failover_enabled {
                if raw_decision.is_refusal() {
                    let signal_label = match &raw_decision {
                        RefusalDecision::Refusal(RefusalSignal::Phrase) => "phrase",
                        RefusalDecision::Refusal(RefusalSignal::ToolOmission) => "tool_omission",
                        _ => unreachable!(),
                    };
                    // Derive pipeline name from resolved stages for logging/metrics.
                    let pipeline_label = stages
                        .first()
                        .map(|s| s.pipeline_name.as_str())
                        .unwrap_or("unknown");

                    // Req 12.11: INFO log on refusal detection — never includes
                    // response content.
                    tracing::info!(
                        target: "guardrail",
                        pipeline = %pipeline_label,
                        signal = %signal_label,
                        trace_id = %trace_id,
                        "refusal detected"
                    );
                    // Req 12.11: best-effort metric recording (Req 11.7).
                    if let Some(metrics) = &self.metrics {
                        metrics.record_guardrail_refusal_detected(pipeline_label, signal_label);
                    }
                }
                raw_decision
            } else {
                // Req 12.6: when the toggle is disabled, return NotRefusal so
                // the response is returned unmodified (no failover).
                RefusalDecision::NotRefusal
            }
        } else {
            RefusalDecision::NotRefusal
        };

        // Req 9.5: re-injection is the final non-halting post-call operation.
        // Skipped when refusal-failover will be triggered (handler re-dispatches
        // and re-injects on the finally selected response).
        if terminal.is_none() && !refusal_decision.is_refusal() && !ctx.is_empty() {
            let has_overflow = ctx.overflow_count() > 0;
            for choice in response.choices.iter_mut() {
                let slots = collect_text_slots(&choice.message.content);
                for (slot, text) in slots {
                    let restored = if has_overflow {
                        ctx.reinject_safe(&text)
                    } else {
                        ctx.reinject(&text)
                    };
                    if restored != text {
                        set_text_slot(&mut choice.message.content, slot, restored);
                    }
                }
            }
        }

        self.emit_summary(
            "post_call",
            trace_id,
            stages_executed,
            &non_pass_actions,
            duration_ms(pipeline_start.elapsed()),
        );

        (
            terminal.unwrap_or(PostCallOutcome::Proceed),
            refusal_decision,
        )
    }

    /// Apply PII re-injection on a response using the given context (Req 9.5).
    ///
    /// This is the public entry point for the handler's refusal-failover loop
    /// (task 17.2): after the loop settles on a final response, the handler
    /// calls this exactly once so re-injection runs on the chosen response.
    ///
    /// Uses [`GuardrailContext::reinject_safe`] when any overflow placeholders
    /// were generated (values redacted but not tracked for re-injection), so
    /// raw `<<PII_TYPE_N>>` tokens cannot leak into the user-visible response.
    pub fn reinject_response(&self, response: &mut OpenAIResponse, ctx: &GuardrailContext) {
        if ctx.is_empty() {
            return;
        }
        let has_overflow = ctx.overflow_count() > 0;
        for choice in response.choices.iter_mut() {
            let slots = collect_text_slots(&choice.message.content);
            for (slot, text) in slots {
                let restored = if has_overflow {
                    ctx.reinject_safe(&text)
                } else {
                    ctx.reinject(&text)
                };
                if restored != text {
                    set_text_slot(&mut choice.message.content, slot, restored);
                }
            }
        }
    }

    /// Record a single stage execution's counter and latency, best-effort
    /// (Req 11.1, 11.2, 11.7). A missing metrics handle is a no-op;
    /// `record_guardrail_stage` itself is lock-free and cannot fail.
    fn record_stage_metric(&self, stage: &ResolvedStage, action: &str, latency_ms: f64) {
        if let Some(metrics) = &self.metrics {
            metrics.record_guardrail_stage(
                &stage.pipeline_name,
                &stage.stage_name,
                stage.provider_type,
                action,
                latency_ms,
            );
        }
    }

    /// Emit the per-stage log for a non-pass action (Req 11.3) or a provider
    /// error (Req 11.6). Never logs the original/triggering content.
    fn log_stage_action(
        &self,
        stage: &ResolvedStage,
        action: &str,
        entity_label: Option<&str>,
        trace_id: &str,
    ) {
        match action {
            "pass" => {}
            // Provider failure/timeout (Req 11.6): WARN with provider type.
            "error" => {
                tracing::warn!(
                    target: "guardrail",
                    pipeline = %stage.pipeline_name,
                    stage = %stage.stage_name,
                    provider_type = %stage.provider_type,
                    trace_id = %trace_id,
                    "guardrail stage provider error"
                );
            }
            // Enforcement action (Req 11.3): INFO with entity label, no content.
            _ => {
                tracing::info!(
                    target: "guardrail",
                    pipeline = %stage.pipeline_name,
                    stage = %stage.stage_name,
                    entity_label = %entity_label.unwrap_or(""),
                    action = %action,
                    trace_id = %trace_id,
                    "guardrail stage action"
                );
            }
        }
    }

    /// Emit the per-request (per-phase) guardrail summary: stages executed, the
    /// list of non-pass actions taken, and total pipeline latency (Req 11.4).
    ///
    /// A per-phase summary is emitted at the end of each of `run_pre_call` and
    /// `run_post_call`; the design's "request log entry" is satisfied by these
    /// two structured entries keyed by `phase`.
    fn emit_summary(
        &self,
        phase: &str,
        trace_id: &str,
        stages_executed: usize,
        non_pass_actions: &[String],
        total_latency_ms: f64,
    ) {
        tracing::info!(
            target: "guardrail",
            phase = %phase,
            trace_id = %trace_id,
            stages_executed,
            actions = %non_pass_actions.join(","),
            total_latency_ms,
            "guardrail pipeline summary"
        );
    }
}

/// Result of analyzing a single content field for a stage, distinguishing a
/// `fail_close` timeout (Req 2.9) from a generic `fail_close` service failure
/// (Req 9.7). Mirrors the failure-policy mapping of
/// [`analyze_with_policy`](provider::analyze_with_policy) but preserves the
/// timeout signal the engine needs for the scan-timeout outcome.
enum StageAnalysis {
    /// Provider succeeded; findings drive the action.
    Findings(Vec<Finding>),
    /// `fail_open` error or timeout: skip this field, continue the pipeline
    /// (Req 9.6).
    Skip,
    /// `fail_close` timeout: pre-call scan-timeout (Req 2.9).
    Timeout,
    /// `fail_close` provider error: guardrail service failure (Req 9.7).
    ServiceFailure,
}

/// Run a stage's provider under its timeout and map errors/timeouts onto the
/// configured failure policy, preserving the timeout distinction.
async fn evaluate_stage(stage: &ResolvedStage, content: &str) -> StageAnalysis {
    match tokio::time::timeout(stage.timeout, stage.provider.analyze(content)).await {
        Ok(Ok(findings)) => StageAnalysis::Findings(findings),
        Ok(Err(_err)) => match stage.failure_policy {
            FailurePolicy::FailOpen => StageAnalysis::Skip,
            FailurePolicy::FailClose => StageAnalysis::ServiceFailure,
        },
        Err(_elapsed) => match stage.failure_policy {
            FailurePolicy::FailOpen => StageAnalysis::Skip,
            FailurePolicy::FailClose => StageAnalysis::Timeout,
        },
    }
}

/// The effect of applying a stage action to a single content field.
enum FieldEffect {
    /// No modification (allow, or no findings).
    Pass,
    /// The field text was rewritten (mask/redact).
    Modified(String),
    /// A halting `block` fired, carrying the triggering entity label.
    Block(String),
}

/// Apply a pre-call action to a content field's `text` given its `findings`.
///
/// Pre-call `redact` replaces spans with deterministic PII placeholders and
/// records them in `ctx` (Req 2.1). `mask` replaces span bytes with `*`
/// (Req 2.3). `block` halts (Req 2.2). `allow` and empty findings are identity
/// (Req 2.4).
fn apply_pre_action(
    action: PolicyAction,
    ctx: &mut GuardrailContext,
    text: &str,
    findings: &[Finding],
) -> FieldEffect {
    if findings.is_empty() {
        return FieldEffect::Pass;
    }
    match action {
        PolicyAction::Allow => FieldEffect::Pass,
        PolicyAction::Block => FieldEffect::Block(findings[0].entity_label.clone()),
        PolicyAction::Mask => FieldEffect::Modified(mask(text, findings)),
        PolicyAction::Redact => FieldEffect::Modified(ctx.redact(text, findings)),
        // Validated as invalid for the pre-call phase before analysis.
        PolicyAction::ReplaceWithPolicyMessage => FieldEffect::Pass,
    }
}

/// Apply a post-call action to a content field's `text` given its `findings`.
///
/// Post-call `redact` replaces each matched span with the literal `[REDACTED]`
/// (Req 3.2). `block` halts (Req 3.1). `mask` performs byte-preserving masking.
/// `allow` and empty findings are identity (Req 3.4). `replace_with_policy_message`
/// is handled at the message level by the caller.
fn apply_post_action(action: PolicyAction, text: &str, findings: &[Finding]) -> FieldEffect {
    if findings.is_empty() {
        return FieldEffect::Pass;
    }
    match action {
        PolicyAction::Allow => FieldEffect::Pass,
        PolicyAction::Block => FieldEffect::Block(findings[0].entity_label.clone()),
        PolicyAction::Mask => FieldEffect::Modified(mask(text, findings)),
        PolicyAction::Redact => FieldEffect::Modified(redact_literal(text, findings, "[REDACTED]")),
        PolicyAction::ReplaceWithPolicyMessage => FieldEffect::Pass,
    }
}

/// Replace each in-bounds, char-boundary-aligned finding span with `replacement`,
/// applying right-to-left so earlier offsets stay valid. Overlapping spans are
/// resolved by skipping any finding intersecting one already replaced to its
/// right. Used for the post-call `[REDACTED]` action (Req 3.2).
fn redact_literal(content: &str, findings: &[Finding], replacement: &str) -> String {
    let mut spans: Vec<&Finding> = findings
        .iter()
        .filter(|f| f.start < f.end && f.end <= content.len())
        .filter(|f| content.is_char_boundary(f.start) && content.is_char_boundary(f.end))
        .collect();
    spans.sort_by(|a, b| b.start.cmp(&a.start));

    let mut out = content.to_string();
    let mut last_consumed_start = usize::MAX;
    for finding in spans {
        if finding.end > last_consumed_start {
            continue;
        }
        out.replace_range(finding.start..finding.end, replacement);
        last_consumed_start = finding.start;
    }
    out
}

/// Clamp `content` to at most `max_chars` UTF-8 characters before analysis
/// (Req 8.1). Returns a prefix slice, so byte offsets in any resulting findings
/// remain valid within the original (longer) string.
fn clamp_content(content: &str, max_chars: usize) -> &str {
    match content.char_indices().nth(max_chars) {
        Some((idx, _)) => &content[..idx],
        None => content,
    }
}

/// Identifies which text field of a message's `content` a slot refers to.
#[derive(Clone, Copy)]
enum TextSlot {
    /// `content` is a plain string.
    Whole,
    /// `content` is a multi-part array; the text lives at `parts[i]["text"]`.
    Part(usize),
}

/// Extract every text field from a message content value across both supported
/// shapes: a plain string, or a multi-part array of `{"type":"text","text":...}`
/// elements (Req 2.5, 3.5). Returns `(slot, owned text)` pairs so the caller can
/// analyze and then write back independently.
fn collect_text_slots(content: &Value) -> Vec<(TextSlot, String)> {
    match content {
        Value::String(s) => vec![(TextSlot::Whole, s.clone())],
        Value::Array(parts) => parts
            .iter()
            .enumerate()
            .filter_map(|(i, part)| {
                if part.get("type").and_then(|t| t.as_str()) == Some("text") {
                    part.get("text")
                        .and_then(|t| t.as_str())
                        .map(|s| (TextSlot::Part(i), s.to_string()))
                } else {
                    None
                }
            })
            .collect(),
        _ => Vec::new(),
    }
}

/// Write `new_text` back into the content field identified by `slot`, preserving
/// the original content shape.
fn set_text_slot(content: &mut Value, slot: TextSlot, new_text: String) {
    match slot {
        TextSlot::Whole => *content = Value::String(new_text),
        TextSlot::Part(i) => {
            if let Some(part) = content.get_mut(i).and_then(|p| p.as_object_mut()) {
                part.insert("text".to_string(), Value::String(new_text));
            }
        }
    }
}

/// Whether a message carries the assistant role (Req 3.5).
fn is_assistant(message: &Message) -> bool {
    message.role == "assistant"
}

/// Convert a [`std::time::Duration`] to fractional milliseconds for the latency
/// histogram (Req 11.2).
fn duration_ms(elapsed: std::time::Duration) -> f64 {
    elapsed.as_secs_f64() * 1000.0
}

/// Derive the single counter action label for a stage execution (Req 11.1).
///
/// Precedence: a provider error/timeout is reported as `error` (Req 11.6); a
/// halting `block` as `block`; otherwise, if the stage's configured action
/// actually modified content, the action's own label (`redact`/`mask`); when
/// nothing was enforced the stage is a `pass`. `allow` never modifies content,
/// so it maps to `pass`. This helper covers the pre-call action set and the
/// per-field post-call actions; `replace_with_policy_message` is labeled by the
/// post-call caller directly.
fn derive_action_label(
    action: PolicyAction,
    blocked: bool,
    errored: bool,
    modified: bool,
) -> &'static str {
    if blocked {
        return "block";
    }
    if errored {
        return "error";
    }
    if !modified {
        return "pass";
    }
    match action {
        PolicyAction::Block => "block",
        PolicyAction::Redact => "redact",
        PolicyAction::Mask => "mask",
        PolicyAction::ReplaceWithPolicyMessage => "replace_with_policy_message",
        PolicyAction::Allow => "pass",
    }
}

#[cfg(test)]
mod engine_tests {
    //! Task 10.1 — core engine behavior with stub providers.

    use super::*;
    use crate::guardrail::config::{
        GuardrailBindings, GuardrailProviderConfig, GuardrailProviderType,
        InstructionInsertionMode, PipelineConfig, ProviderSettings, StageConfig,
    };
    use crate::guardrail::provider::{Finding, GuardrailProvider, GuardrailProviderError};
    use crate::models::openai::{Choice, Message, OpenAIRequest, OpenAIResponse, Usage};
    use std::sync::atomic::{AtomicUsize, Ordering};

    // --- Stub providers ---------------------------------------------------

    /// Returns a single finding covering the whole (non-empty) content.
    struct WholeMatch(&'static str);
    #[async_trait::async_trait]
    impl GuardrailProvider for WholeMatch {
        async fn analyze(&self, content: &str) -> Result<Vec<Finding>, GuardrailProviderError> {
            if content.is_empty() {
                return Ok(vec![]);
            }
            Ok(vec![Finding {
                entity_label: self.0.to_string(),
                start: 0,
                end: content.len(),
                matched_text: Some(content.to_string()),
                score: None,
            }])
        }
        fn provider_type(&self) -> &'static str {
            "regex"
        }
    }

    /// Finds every non-overlapping occurrence of a fixed needle.
    struct Substr(&'static str, &'static str); // (needle, label)
    #[async_trait::async_trait]
    impl GuardrailProvider for Substr {
        async fn analyze(&self, content: &str) -> Result<Vec<Finding>, GuardrailProviderError> {
            let mut findings = Vec::new();
            let mut from = 0usize;
            while let Some(rel) = content[from..].find(self.0) {
                let start = from + rel;
                let end = start + self.0.len();
                findings.push(Finding {
                    entity_label: self.1.to_string(),
                    start,
                    end,
                    matched_text: Some(self.0.to_string()),
                    score: None,
                });
                from = end;
            }
            Ok(findings)
        }
        fn provider_type(&self) -> &'static str {
            "regex"
        }
    }

    /// Always errors, to exercise failure-policy paths.
    struct Failing;
    #[async_trait::async_trait]
    impl GuardrailProvider for Failing {
        async fn analyze(&self, _content: &str) -> Result<Vec<Finding>, GuardrailProviderError> {
            Err(GuardrailProviderError::Unreachable("boom".to_string()))
        }
        fn provider_type(&self) -> &'static str {
            "regex"
        }
    }

    /// Records the char count of the content it received.
    struct Recording(Arc<AtomicUsize>);
    #[async_trait::async_trait]
    impl GuardrailProvider for Recording {
        async fn analyze(&self, content: &str) -> Result<Vec<Finding>, GuardrailProviderError> {
            self.0.store(content.chars().count(), Ordering::SeqCst);
            Ok(vec![])
        }
        fn provider_type(&self) -> &'static str {
            "regex"
        }
    }

    // --- Builders ---------------------------------------------------------

    fn stage(name: &str, phase: StagePhase, action: PolicyAction) -> StageConfig {
        StageConfig {
            name: name.to_string(),
            provider: "p".to_string(),
            phase,
            action,
        }
    }

    fn make_engine(
        provider: Arc<dyn GuardrailProvider>,
        policy: FailurePolicy,
        stages: Vec<StageConfig>,
    ) -> GuardrailEngine {
        let mut registry = ProviderRegistry::new();
        registry.insert("p", provider);
        let config = GuardrailConfig {
            providers: vec![GuardrailProviderConfig {
                name: "p".to_string(),
                provider_type: GuardrailProviderType::Regex,
                failure_policy: policy,
                timeout_seconds: 5,
                settings: ProviderSettings::default(),
            }],
            pipelines: vec![PipelineConfig {
                name: "pl".to_string(),
                stages,
                redaction_notice_instruction: None,
                instruction_insertion_mode: InstructionInsertionMode::default(),
                failover_on_refusal: false,
                refusal_phrase_list: None,
            }],
            global_default_pipeline: Some("pl".to_string()),
            bindings: GuardrailBindings::default(),
            ..Default::default()
        };
        GuardrailEngine::new(&config, &registry, None).unwrap()
    }

    fn user_request(text: &str) -> OpenAIRequest {
        OpenAIRequest {
            model: "m".to_string(),
            messages: vec![Message {
                role: "user".to_string(),
                content: Value::String(text.to_string()),
                extra: serde_json::Map::new(),
            }],
            stream: false,
            temperature: None,
            max_tokens: None,
            extra: serde_json::Map::new(),
        }
    }

    fn assistant_response(text: &str) -> OpenAIResponse {
        OpenAIResponse {
            id: String::new(),
            object: String::new(),
            created: 0,
            model: "m".to_string(),
            choices: vec![Choice {
                index: 0,
                message: Message {
                    role: "assistant".to_string(),
                    content: Value::String(text.to_string()),
                    extra: serde_json::Map::new(),
                },
                finish_reason: Some("stop".to_string()),
                extra: serde_json::Map::new(),
            }],
            usage: Usage::default(),
            extra: serde_json::Map::new(),
        }
    }

    fn assistant_text(resp: &OpenAIResponse) -> String {
        resp.choices[0].message.content_as_text()
    }

    /// Default ToolContext for tests: no tools involved, no refusal signal.
    fn no_tool_ctx() -> ToolContext {
        ToolContext {
            tool_use_allowed: false,
            tools_provided: false,
            finish_reason_is_tool_call: false,
            has_tool_calls: false,
        }
    }

    // --- Tests ------------------------------------------------------------

    #[tokio::test]
    async fn allow_is_identity_pre_call() {
        let engine = make_engine(
            Arc::new(WholeMatch("X")),
            FailurePolicy::FailClose,
            vec![stage("s", StagePhase::PreCall, PolicyAction::Allow)],
        );
        let mut req = user_request("hello world");
        let mut ctx = GuardrailContext::new();
        let outcome = engine
            .run_pre_call(&mut req, &BindingSelector::default(), &mut ctx, "t")
            .await;

        assert_eq!(outcome, PreCallOutcome::Proceed);
        assert_eq!(req.messages.len(), 1);
        assert_eq!(req.messages[0].content_as_text(), "hello world");
        assert!(ctx.is_empty());
    }

    #[tokio::test]
    async fn pre_call_block_halts_and_carries_entity() {
        let engine = make_engine(
            Arc::new(WholeMatch("API_KEY")),
            FailurePolicy::FailClose,
            vec![stage("secret", StagePhase::PreCall, PolicyAction::Block)],
        );
        let mut req = user_request("some secret");
        let mut ctx = GuardrailContext::new();
        let outcome = engine
            .run_pre_call(&mut req, &BindingSelector::default(), &mut ctx, "t")
            .await;

        match outcome {
            PreCallOutcome::Block(block) => {
                assert_eq!(block.entity_label, "API_KEY");
                assert_eq!(block.stage_name, "secret");
                assert_eq!(block.phase, StagePhase::PreCall);
            }
            other => panic!("expected Block, got {other:?}"),
        }
        // Request is left unmodified (no system instruction prepended).
        assert_eq!(req.messages.len(), 1);
    }

    #[tokio::test]
    async fn redact_pre_call_then_reinject_post_call_round_trip() {
        let engine = make_engine(
            Arc::new(Substr("john@x.com", "EMAIL")),
            FailurePolicy::FailClose,
            vec![stage("pii", StagePhase::PreCall, PolicyAction::Redact)],
        );
        let mut req = user_request("contact john@x.com now");
        let mut ctx = GuardrailContext::new();
        let outcome = engine
            .run_pre_call(&mut req, &BindingSelector::default(), &mut ctx, "t")
            .await;
        assert_eq!(outcome, PreCallOutcome::Proceed);

        // A preserve-placeholders system instruction was prepended (Req 4.4).
        assert_eq!(req.messages.len(), 2);
        assert_eq!(req.messages[0].role, "system");
        assert_eq!(
            req.messages[1].content_as_text(),
            "contact <<PII_EMAIL_1>> now"
        );
        assert_eq!(ctx.len(), 1);

        // The LLM echoes the placeholder; post-call re-injection restores it.
        let mut resp = assistant_response("here you go: <<PII_EMAIL_1>>");
        let post = engine
            .run_post_call(
                &mut resp,
                &BindingSelector::default(),
                &mut ctx,
                "t",
                &no_tool_ctx(),
            )
            .await;
        assert_eq!(post.0, PostCallOutcome::Proceed);
        assert_eq!(assistant_text(&resp), "here you go: john@x.com");
    }

    #[tokio::test]
    async fn fail_open_skips_stage_and_proceeds() {
        let engine = make_engine(
            Arc::new(Failing),
            FailurePolicy::FailOpen,
            vec![stage("s", StagePhase::PreCall, PolicyAction::Redact)],
        );
        let mut req = user_request("unchanged content");
        let mut ctx = GuardrailContext::new();
        let outcome = engine
            .run_pre_call(&mut req, &BindingSelector::default(), &mut ctx, "t")
            .await;

        assert_eq!(outcome, PreCallOutcome::Proceed);
        assert_eq!(req.messages[0].content_as_text(), "unchanged content");
        assert!(ctx.is_empty());
    }

    #[tokio::test]
    async fn fail_close_error_returns_service_failure() {
        let engine = make_engine(
            Arc::new(Failing),
            FailurePolicy::FailClose,
            vec![stage("s", StagePhase::PreCall, PolicyAction::Block)],
        );
        let mut req = user_request("content");
        let mut ctx = GuardrailContext::new();
        let outcome = engine
            .run_pre_call(&mut req, &BindingSelector::default(), &mut ctx, "t")
            .await;
        assert_eq!(outcome, PreCallOutcome::ServiceFailure);
    }

    #[tokio::test]
    async fn invalid_action_pre_call_replace_with_policy_message() {
        let engine = make_engine(
            Arc::new(WholeMatch("X")),
            FailurePolicy::FailClose,
            vec![stage(
                "bad",
                StagePhase::PreCall,
                PolicyAction::ReplaceWithPolicyMessage,
            )],
        );
        let mut req = user_request("content");
        let mut ctx = GuardrailContext::new();
        let outcome = engine
            .run_pre_call(&mut req, &BindingSelector::default(), &mut ctx, "t")
            .await;
        assert_eq!(outcome, PreCallOutcome::InvalidAction);
    }

    #[tokio::test]
    async fn post_call_redact_replaces_span_with_literal() {
        let engine = make_engine(
            Arc::new(Substr("bad", "TERM")),
            FailurePolicy::FailClose,
            vec![stage("out", StagePhase::PostCall, PolicyAction::Redact)],
        );
        let mut resp = assistant_response("this is bad text with bad words");
        let mut ctx = GuardrailContext::new();
        let post = engine
            .run_post_call(
                &mut resp,
                &BindingSelector::default(),
                &mut ctx,
                "t",
                &no_tool_ctx(),
            )
            .await;

        assert_eq!(post.0, PostCallOutcome::Proceed);
        assert_eq!(
            assistant_text(&resp),
            "this is [REDACTED] text with [REDACTED] words"
        );
    }

    #[tokio::test]
    async fn post_call_replace_with_policy_message_rewrites_and_halts() {
        let engine = make_engine(
            Arc::new(WholeMatch("TOXIC")),
            FailurePolicy::FailClose,
            vec![stage(
                "out",
                StagePhase::PostCall,
                PolicyAction::ReplaceWithPolicyMessage,
            )],
        );
        let mut resp = assistant_response("prohibited output");
        let mut ctx = GuardrailContext::new();
        let post = engine
            .run_post_call(
                &mut resp,
                &BindingSelector::default(),
                &mut ctx,
                "t",
                &no_tool_ctx(),
            )
            .await;

        assert_eq!(post.0, PostCallOutcome::Replaced);
        assert_eq!(assistant_text(&resp), DEFAULT_POLICY_MESSAGE);
    }

    #[tokio::test]
    async fn post_call_block_discards_response() {
        let engine = make_engine(
            Arc::new(WholeMatch("SECRET")),
            FailurePolicy::FailClose,
            vec![stage("out", StagePhase::PostCall, PolicyAction::Block)],
        );
        let mut resp = assistant_response("leaked secret");
        let mut ctx = GuardrailContext::new();
        let post = engine
            .run_post_call(
                &mut resp,
                &BindingSelector::default(),
                &mut ctx,
                "t",
                &no_tool_ctx(),
            )
            .await;

        match post {
            (PostCallOutcome::Block(block), _) => {
                assert_eq!(block.entity_label, "SECRET");
                assert_eq!(block.phase, StagePhase::PostCall);
            }
            other => panic!("expected Block, got {other:?}"),
        }
    }

    #[tokio::test]
    async fn no_bound_pipeline_proceeds_without_side_effects() {
        // Engine with no global default and no bindings: nothing resolves.
        let mut registry = ProviderRegistry::new();
        registry.insert("p", Arc::new(WholeMatch("X")) as Arc<dyn GuardrailProvider>);
        let config = GuardrailConfig {
            providers: vec![GuardrailProviderConfig {
                name: "p".to_string(),
                provider_type: GuardrailProviderType::Regex,
                failure_policy: FailurePolicy::FailClose,
                timeout_seconds: 5,
                settings: ProviderSettings::default(),
            }],
            pipelines: vec![PipelineConfig {
                name: "pl".to_string(),
                stages: vec![stage("s", StagePhase::PreCall, PolicyAction::Block)],
                redaction_notice_instruction: None,
                instruction_insertion_mode: InstructionInsertionMode::default(),
                failover_on_refusal: false,
                refusal_phrase_list: None,
            }],
            global_default_pipeline: None,
            bindings: GuardrailBindings::default(),
            ..Default::default()
        };
        let engine = GuardrailEngine::new(&config, &registry, None).unwrap();

        let mut req = user_request("anything");
        let mut ctx = GuardrailContext::new();
        let outcome = engine
            .run_pre_call(&mut req, &BindingSelector::default(), &mut ctx, "t")
            .await;
        assert_eq!(outcome, PreCallOutcome::Proceed);
        assert_eq!(req.messages.len(), 1);
    }

    #[tokio::test]
    async fn content_is_clamped_before_analysis() {
        let seen = Arc::new(AtomicUsize::new(0));
        let engine = make_engine(
            Arc::new(Recording(seen.clone())),
            FailurePolicy::FailClose,
            vec![stage("s", StagePhase::PreCall, PolicyAction::Allow)],
        );
        // Content longer than the clamp.
        let long = "a".repeat(DEFAULT_MAX_CONTENT_CHARS + 50);
        let mut req = user_request(&long);
        let mut ctx = GuardrailContext::new();
        let _ = engine
            .run_pre_call(&mut req, &BindingSelector::default(), &mut ctx, "t")
            .await;

        assert_eq!(seen.load(Ordering::SeqCst), DEFAULT_MAX_CONTENT_CHARS);
    }

    #[tokio::test]
    async fn scans_multi_part_text_array_and_all_roles() {
        // system + user messages; user content is a multi-part array.
        let engine = make_engine(
            Arc::new(Substr("secret", "TERM")),
            FailurePolicy::FailClose,
            vec![stage("s", StagePhase::PreCall, PolicyAction::Mask)],
        );
        let mut req = OpenAIRequest {
            model: "m".to_string(),
            messages: vec![
                Message {
                    role: "system".to_string(),
                    content: Value::String("keep secret safe".to_string()),
                    extra: serde_json::Map::new(),
                },
                Message {
                    role: "user".to_string(),
                    content: serde_json::json!([
                        {"type": "text", "text": "my secret here"},
                        {"type": "image_url", "image_url": {"url": "http://x"}}
                    ]),
                    extra: serde_json::Map::new(),
                },
            ],
            stream: false,
            temperature: None,
            max_tokens: None,
            extra: serde_json::Map::new(),
        };
        let mut ctx = GuardrailContext::new();
        let outcome = engine
            .run_pre_call(&mut req, &BindingSelector::default(), &mut ctx, "t")
            .await;
        assert_eq!(outcome, PreCallOutcome::Proceed);

        // "secret" masked in both the system string and the user text part.
        assert_eq!(req.messages[0].content_as_text(), "keep ****** safe");
        assert_eq!(req.messages[1].content_as_text(), "my ****** here");
        // Non-text part preserved.
        assert_eq!(
            req.messages[1].content[1]["type"].as_str(),
            Some("image_url")
        );
    }

    // =====================================================================
    // Tasks 10.2–10.11 — engine property tests and error-mapping unit tests.
    //
    // These reuse the stub providers / builders above and add a few extra
    // stubs (`Slow`, `Recorder`) plus a multi-provider engine builder for
    // ordering and failure-policy properties. Property tests run >=100 cases.
    // =====================================================================

    use proptest::prelude::*;
    use std::sync::Mutex;

    /// Build a fresh single-thread Tokio runtime (time enabled) to drive the
    /// engine's async API from inside synchronous proptest bodies.
    fn rt() -> tokio::runtime::Runtime {
        tokio::runtime::Builder::new_current_thread()
            .enable_all()
            .build()
            .unwrap()
    }

    // --- Extra stub providers --------------------------------------------

    /// Sleeps longer than any test timeout, so a `fail_close` stage configured
    /// with a 0-second timeout resolves to a scan timeout (Req 2.9, 8.6).
    struct Slow;
    #[async_trait::async_trait]
    impl GuardrailProvider for Slow {
        async fn analyze(&self, _content: &str) -> Result<Vec<Finding>, GuardrailProviderError> {
            tokio::time::sleep(std::time::Duration::from_millis(200)).await;
            Ok(vec![])
        }
        fn provider_type(&self) -> &'static str {
            "regex"
        }
    }

    /// Records its `id` into a shared log every time it is invoked, and
    /// optionally returns a whole-content finding (to drive halting actions).
    /// Used to observe stage execution order and short-circuit behavior.
    struct Recorder {
        id: usize,
        log: Arc<Mutex<Vec<usize>>>,
        produce_finding: bool,
    }
    #[async_trait::async_trait]
    impl GuardrailProvider for Recorder {
        async fn analyze(&self, content: &str) -> Result<Vec<Finding>, GuardrailProviderError> {
            self.log.lock().unwrap().push(self.id);
            if self.produce_finding && !content.is_empty() {
                Ok(vec![Finding {
                    entity_label: "T".to_string(),
                    start: 0,
                    end: content.len(),
                    matched_text: Some(content.to_string()),
                    score: None,
                }])
            } else {
                Ok(vec![])
            }
        }
        fn provider_type(&self) -> &'static str {
            "regex"
        }
    }

    // --- Multi-provider engine builder -----------------------------------

    /// One stage's provider plus its config knobs, for [`make_engine_multi`].
    struct StageSpec {
        provider: Arc<dyn GuardrailProvider>,
        action: PolicyAction,
        phase: StagePhase,
        policy: FailurePolicy,
        timeout_secs: u64,
    }

    /// Build an engine whose single global-default pipeline contains one stage
    /// per `StageSpec`, each backed by its own uniquely-named provider so
    /// per-stage failure policy / timeout / invocation can be observed.
    fn make_engine_multi(specs: Vec<StageSpec>) -> GuardrailEngine {
        let mut registry = ProviderRegistry::new();
        let mut providers = Vec::new();
        let mut stages = Vec::new();
        for (i, spec) in specs.into_iter().enumerate() {
            let name = format!("p{i}");
            registry.insert(name.clone(), spec.provider);
            providers.push(GuardrailProviderConfig {
                name: name.clone(),
                provider_type: GuardrailProviderType::Regex,
                failure_policy: spec.policy,
                timeout_seconds: spec.timeout_secs,
                settings: ProviderSettings::default(),
            });
            stages.push(StageConfig {
                name: format!("s{i}"),
                provider: name,
                phase: spec.phase,
                action: spec.action,
            });
        }
        let config = GuardrailConfig {
            providers,
            pipelines: vec![PipelineConfig {
                name: "pl".to_string(),
                stages,
                redaction_notice_instruction: None,
                instruction_insertion_mode: InstructionInsertionMode::default(),
                failover_on_refusal: false,
                refusal_phrase_list: None,
            }],
            global_default_pipeline: Some("pl".to_string()),
            bindings: GuardrailBindings::default(),
            ..Default::default()
        };
        GuardrailEngine::new(&config, &registry, None).unwrap()
    }

    /// A response whose single assistant choice carries multi-part array content.
    fn assistant_array_response(parts: Value) -> OpenAIResponse {
        OpenAIResponse {
            id: String::new(),
            object: String::new(),
            created: 0,
            model: "m".to_string(),
            choices: vec![Choice {
                index: 0,
                message: Message {
                    role: "assistant".to_string(),
                    content: parts,
                    extra: serde_json::Map::new(),
                },
                finish_reason: Some("stop".to_string()),
                extra: serde_json::Map::new(),
            }],
            usage: Usage::default(),
            extra: serde_json::Map::new(),
        }
    }

    // -----------------------------------------------------------------
    // 10.2 — Feature: guardrail-pipelines, Property 4: Allow action is identity
    // **Validates: Requirements 2.4, 3.4**
    //
    // Applying an `allow` stage (even when the provider reports a whole-content
    // finding) leaves the request/response byte-identical and yields a pass
    // outcome. Byte-identity is checked via canonical JSON serialization.
    // -----------------------------------------------------------------
    proptest! {
        #![proptest_config(ProptestConfig::with_cases(64))]

        #[test]
        fn prop4_allow_is_identity(text in ".{0,80}") {
            // Pre-call allow.
            let engine = make_engine(
                Arc::new(WholeMatch("X")),
                FailurePolicy::FailClose,
                vec![stage("s", StagePhase::PreCall, PolicyAction::Allow)],
            );
            let mut req = user_request(&text);
            let before = serde_json::to_string(&req).unwrap();
            let mut ctx = GuardrailContext::new();
            let outcome = rt().block_on(engine.run_pre_call(
                &mut req,
                &BindingSelector::default(),
                &mut ctx,
                "t",
            ));
            let after = serde_json::to_string(&req).unwrap();
            prop_assert_eq!(outcome, PreCallOutcome::Proceed);
            prop_assert_eq!(&before, &after);
            prop_assert!(ctx.is_empty());

            // Post-call allow.
            let engine2 = make_engine(
                Arc::new(WholeMatch("X")),
                FailurePolicy::FailClose,
                vec![stage("s", StagePhase::PostCall, PolicyAction::Allow)],
            );
            let mut resp = assistant_response(&text);
            let before2 = serde_json::to_string(&resp).unwrap();
            let mut ctx2 = GuardrailContext::new();
            let post = rt().block_on(engine2.run_post_call(
                &mut resp,
                &BindingSelector::default(),
                &mut ctx2,
                "t",
                &no_tool_ctx(),
            ));
            let after2 = serde_json::to_string(&resp).unwrap();
            prop_assert_eq!(post, (PostCallOutcome::Proceed, RefusalDecision::NotRefusal));
            prop_assert_eq!(&before2, &after2);
        }
    }

    // -----------------------------------------------------------------
    // 10.3 — Feature: guardrail-pipelines, Property 9: All message content is
    //        scanned regardless of role or shape
    // **Validates: Requirements 2.5, 3.5**
    //
    // Pre-call: a detectable needle embedded in every message is detected and
    // acted on regardless of role (user/system/assistant/tool). Post-call: a
    // needle is detected in assistant content whether it is a plain string or a
    // multi-part text array.
    // -----------------------------------------------------------------
    proptest! {
        #![proptest_config(ProptestConfig::with_cases(64))]

        #[test]
        fn prop9_all_content_scanned(
            role_idx in prop::collection::vec(0usize..4, 1..5),
            tail in "[a-z ]{0,20}",
        ) {
            const NEEDLE: &str = "SEEKME";
            let roles = ["user", "system", "assistant", "tool"];

            // --- Pre-call: every role scanned (mask replaces needle bytes). ---
            let engine = make_engine(
                Arc::new(Substr(NEEDLE, "TERM")),
                FailurePolicy::FailClose,
                vec![stage("s", StagePhase::PreCall, PolicyAction::Mask)],
            );
            let messages: Vec<Message> = role_idx
                .iter()
                .map(|&i| Message {
                    role: roles[i].to_string(),
                    content: Value::String(format!("x {NEEDLE} {tail}")),
                    extra: serde_json::Map::new(),
                })
                .collect();
            let mut req = OpenAIRequest {
                model: "m".to_string(),
                messages,
                stream: false,
                temperature: None,
                max_tokens: None,
                extra: serde_json::Map::new(),
            };
            let mut ctx = GuardrailContext::new();
            let outcome = rt().block_on(engine.run_pre_call(
                &mut req,
                &BindingSelector::default(),
                &mut ctx,
                "t",
            ));
            prop_assert_eq!(outcome, PreCallOutcome::Proceed);
            // ctx.system_instruction() may prepend a system message only when a
            // map entry exists; mask never records one, so message count is
            // preserved and each original message no longer contains the needle.
            for msg in req.messages.iter() {
                prop_assert!(!msg.content_as_text().contains(NEEDLE));
                prop_assert!(msg.content_as_text().contains("******"));
            }

            // --- Post-call, plain string assistant content. ---
            let engine_s = make_engine(
                Arc::new(Substr(NEEDLE, "TERM")),
                FailurePolicy::FailClose,
                vec![stage("s", StagePhase::PostCall, PolicyAction::Redact)],
            );
            let mut resp_s = assistant_response(&format!("a {NEEDLE} {tail}"));
            let mut ctx_s = GuardrailContext::new();
            let post_s = rt().block_on(engine_s.run_post_call(
                &mut resp_s,
                &BindingSelector::default(),
                &mut ctx_s,
                "t",
                &no_tool_ctx(),
            ));
            prop_assert_eq!(post_s, (PostCallOutcome::Proceed, RefusalDecision::NotRefusal));
            prop_assert!(!assistant_text(&resp_s).contains(NEEDLE));
            prop_assert!(assistant_text(&resp_s).contains("[REDACTED]"));

            // --- Post-call, multi-part text array assistant content. ---
            let engine_a = make_engine(
                Arc::new(Substr(NEEDLE, "TERM")),
                FailurePolicy::FailClose,
                vec![stage("s", StagePhase::PostCall, PolicyAction::Redact)],
            );
            let mut resp_a = assistant_array_response(serde_json::json!([
                {"type": "text", "text": format!("{NEEDLE} {tail}")},
                {"type": "image_url", "image_url": {"url": "http://x"}}
            ]));
            let mut ctx_a = GuardrailContext::new();
            let post_a = rt().block_on(engine_a.run_post_call(
                &mut resp_a,
                &BindingSelector::default(),
                &mut ctx_a,
                "t",
                &no_tool_ctx(),
            ));
            prop_assert_eq!(post_a, (PostCallOutcome::Proceed, RefusalDecision::NotRefusal));
            let text_part = resp_a.choices[0].message.content[0]["text"]
                .as_str()
                .unwrap()
                .to_string();
            prop_assert!(!text_part.contains(NEEDLE));
            prop_assert!(text_part.contains("[REDACTED]"));
            // Non-text part is left intact.
            prop_assert_eq!(
                resp_a.choices[0].message.content[1]["type"].as_str(),
                Some("image_url")
            );
        }
    }

    // -----------------------------------------------------------------
    // 10.4 — Feature: guardrail-pipelines, Property 10: Post-call block
    //        discards; replace rewrites; redact preserves structure elsewhere
    // **Validates: Requirements 3.1, 3.2, 3.3**
    // -----------------------------------------------------------------
    proptest! {
        #![proptest_config(ProptestConfig::with_cases(64))]

        #[test]
        fn prop10_post_block_replace_redact(
            pre in "[a-z ]{0,15}",
            post in "[a-z ]{0,15}",
        ) {
            let text = format!("{pre}BAD{post}");
            let sel = BindingSelector::default();

            // block → 403-mapped Block outcome, response discarded.
            let e_block = make_engine(
                Arc::new(Substr("BAD", "CAT")),
                FailurePolicy::FailClose,
                vec![stage("out", StagePhase::PostCall, PolicyAction::Block)],
            );
            let mut r_block = assistant_response(&text);
            let mut c = GuardrailContext::new();
            let o_block = rt().block_on(e_block.run_post_call(&mut r_block, &sel, &mut c, "t", &no_tool_ctx()));
            match o_block.0 {
                PostCallOutcome::Block(b) => {
                    prop_assert_eq!(b.entity_label, "CAT".to_string());
                    prop_assert_eq!(b.phase, StagePhase::PostCall);
                }
                other => prop_assert!(false, "expected Block, got {:?}", other),
            }

            // replace_with_policy_message → assistant content replaced, 200.
            let e_rep = make_engine(
                Arc::new(Substr("BAD", "CAT")),
                FailurePolicy::FailClose,
                vec![stage(
                    "out",
                    StagePhase::PostCall,
                    PolicyAction::ReplaceWithPolicyMessage,
                )],
            );
            let mut r_rep = assistant_response(&text);
            let mut c2 = GuardrailContext::new();
            let o_rep = rt().block_on(e_rep.run_post_call(&mut r_rep, &sel, &mut c2, "t", &no_tool_ctx()));
            prop_assert_eq!(o_rep.0, PostCallOutcome::Replaced);
            prop_assert_eq!(assistant_text(&r_rep), DEFAULT_POLICY_MESSAGE.to_string());
            // Non-content structure preserved.
            prop_assert_eq!(&r_rep.model, "m");
            prop_assert_eq!(r_rep.choices.len(), 1);
            prop_assert_eq!(r_rep.choices[0].finish_reason.as_deref(), Some("stop"));

            // redact → each matched span → [REDACTED]; other fields unchanged.
            let e_red = make_engine(
                Arc::new(Substr("BAD", "CAT")),
                FailurePolicy::FailClose,
                vec![stage("out", StagePhase::PostCall, PolicyAction::Redact)],
            );
            let mut r_red = assistant_response(&text);
            let mut c3 = GuardrailContext::new();
            let o_red = rt().block_on(e_red.run_post_call(&mut r_red, &sel, &mut c3, "t", &no_tool_ctx()));
            prop_assert_eq!(o_red.0, PostCallOutcome::Proceed);
            let out = assistant_text(&r_red);
            prop_assert!(!out.contains("BAD"));
            prop_assert!(out.contains("[REDACTED]"));
            prop_assert_eq!(out, format!("{pre}[REDACTED]{post}"));
            // HTTP/other fields unchanged.
            prop_assert_eq!(&r_red.model, "m");
            prop_assert_eq!(&r_red.id, "");
            prop_assert_eq!(r_red.choices[0].finish_reason.as_deref(), Some("stop"));
        }
    }

    // -----------------------------------------------------------------
    // 10.5 — Feature: guardrail-pipelines, Property 11: Pre-call block carries
    //        the triggering category and forwards nothing
    // **Validates: Requirements 2.2**
    //
    // The engine never invokes the router; a pre-call block returns immediately
    // and does not mutate/forward the request (verified via byte-identity).
    // -----------------------------------------------------------------
    proptest! {
        #![proptest_config(ProptestConfig::with_cases(64))]

        #[test]
        fn prop11_pre_call_block_carries_category(
            label in prop::sample::select(vec!["EMAIL", "US_SSN", "API_KEY", "PERSON"]),
            text in "[a-z ]{1,40}",
        ) {
            let engine = make_engine(
                Arc::new(WholeMatch(label)),
                FailurePolicy::FailClose,
                vec![stage("secret", StagePhase::PreCall, PolicyAction::Block)],
            );
            let mut req = user_request(&text);
            let before = serde_json::to_string(&req).unwrap();
            let mut ctx = GuardrailContext::new();
            let outcome = rt().block_on(engine.run_pre_call(
                &mut req,
                &BindingSelector::default(),
                &mut ctx,
                "t",
            ));
            let after = serde_json::to_string(&req).unwrap();

            match outcome {
                PreCallOutcome::Block(b) => {
                    prop_assert_eq!(b.entity_label, label.to_string());
                    prop_assert_eq!(b.phase, StagePhase::PreCall);
                    prop_assert_eq!(b.stage_name, "secret".to_string());
                }
                other => prop_assert!(false, "expected Block, got {:?}", other),
            }
            // Request neither mutated nor forwarded.
            prop_assert_eq!(&before, &after);
            prop_assert!(ctx.is_empty());
        }
    }

    // -----------------------------------------------------------------
    // 10.6 — Feature: guardrail-pipelines, Property 18: Content is clamped
    //        before provider analysis
    // **Validates: Requirements 8.1**
    // -----------------------------------------------------------------
    proptest! {
        #![proptest_config(ProptestConfig::with_cases(64))]

        #[test]
        fn prop18_content_clamped(len in 0usize..300, max in 1usize..100) {
            let seen = Arc::new(AtomicUsize::new(0));
            let mut engine = make_engine(
                Arc::new(Recording(seen.clone())),
                FailurePolicy::FailClose,
                vec![stage("s", StagePhase::PreCall, PolicyAction::Allow)],
            );
            engine.set_max_content_chars(max);

            let content = "a".repeat(len);
            let mut req = user_request(&content);
            let mut ctx = GuardrailContext::new();
            rt().block_on(engine.run_pre_call(
                &mut req,
                &BindingSelector::default(),
                &mut ctx,
                "t",
            ));

            prop_assert_eq!(seen.load(Ordering::SeqCst), len.min(max));
        }
    }

    // -----------------------------------------------------------------
    // 10.7 — Feature: guardrail-pipelines, Property 20: In-order execution and
    //        continuation for non-halting actions
    // **Validates: Requirements 9.1, 9.3**
    // -----------------------------------------------------------------
    proptest! {
        #![proptest_config(ProptestConfig::with_cases(64))]

        #[test]
        fn prop20_in_order_non_halting(
            is_pre in any::<bool>(),
            action_idx in prop::collection::vec(0usize..3, 1..7),
        ) {
            let n = action_idx.len();
            let log = Arc::new(Mutex::new(Vec::<usize>::new()));
            let phase = if is_pre { StagePhase::PreCall } else { StagePhase::PostCall };
            // pre-call non-halting: allow/redact/mask; post-call: allow/redact.
            let pre_actions = [PolicyAction::Allow, PolicyAction::Redact, PolicyAction::Mask];
            let post_actions = [PolicyAction::Allow, PolicyAction::Redact, PolicyAction::Allow];

            let specs: Vec<StageSpec> = action_idx
                .iter()
                .enumerate()
                .map(|(i, &a)| StageSpec {
                    provider: Arc::new(Recorder {
                        id: i,
                        log: log.clone(),
                        produce_finding: false,
                    }),
                    action: if is_pre { pre_actions[a] } else { post_actions[a] },
                    phase,
                    policy: FailurePolicy::FailClose,
                    timeout_secs: 5,
                })
                .collect();
            let engine = make_engine_multi(specs);
            let sel = BindingSelector::default();
            let mut ctx = GuardrailContext::new();

            if is_pre {
                let mut req = user_request("some content");
                let o = rt().block_on(engine.run_pre_call(&mut req, &sel, &mut ctx, "t"));
                prop_assert_eq!(o, PreCallOutcome::Proceed);
            } else {
                let mut resp = assistant_response("some content");
                let o = rt().block_on(engine.run_post_call(&mut resp, &sel, &mut ctx, "t", &no_tool_ctx()));
                prop_assert_eq!(o.0, PostCallOutcome::Proceed);
            }

            let recorded = log.lock().unwrap().clone();
            prop_assert_eq!(recorded, (0..n).collect::<Vec<_>>());
        }
    }

    // -----------------------------------------------------------------
    // 10.8 — Feature: guardrail-pipelines, Property 21: Halting actions
    //        short-circuit the pipeline
    // **Validates: Requirements 9.2, 9.4**
    // -----------------------------------------------------------------
    proptest! {
        #![proptest_config(ProptestConfig::with_cases(64))]

        #[test]
        fn prop21_halting_short_circuits(
            n in 1usize..7,
            halt_at in 0usize..7,
            is_pre in any::<bool>(),
            post_replace in any::<bool>(),
        ) {
            let i = halt_at % n; // halting stage index in [0, n)
            let log = Arc::new(Mutex::new(Vec::<usize>::new()));
            let phase = if is_pre { StagePhase::PreCall } else { StagePhase::PostCall };
            let halting_action = if is_pre {
                PolicyAction::Block
            } else if post_replace {
                PolicyAction::ReplaceWithPolicyMessage
            } else {
                PolicyAction::Block
            };

            let specs: Vec<StageSpec> = (0..n)
                .map(|idx| StageSpec {
                    provider: Arc::new(Recorder {
                        id: idx,
                        log: log.clone(),
                        // Only the halting stage produces a finding so it fires.
                        produce_finding: idx == i,
                    }),
                    action: if idx == i { halting_action } else { PolicyAction::Allow },
                    phase,
                    policy: FailurePolicy::FailClose,
                    timeout_secs: 5,
                })
                .collect();
            let engine = make_engine_multi(specs);
            let sel = BindingSelector::default();
            let mut ctx = GuardrailContext::new();

            if is_pre {
                let mut req = user_request("some content");
                let o = rt().block_on(engine.run_pre_call(&mut req, &sel, &mut ctx, "t"));
                prop_assert!(matches!(o, PreCallOutcome::Block(_)));
            } else {
                let mut resp = assistant_response("some content");
                let o = rt().block_on(engine.run_post_call(&mut resp, &sel, &mut ctx, "t", &no_tool_ctx()));
                prop_assert!(matches!(
                    o.0,
                    PostCallOutcome::Block(_) | PostCallOutcome::Replaced
                ));
            }

            // Stages 0..=i executed exactly once and in order; none after `i`.
            let recorded = log.lock().unwrap().clone();
            prop_assert_eq!(recorded, (0..=i).collect::<Vec<_>>());
        }
    }

    // -----------------------------------------------------------------
    // 10.9 — Feature: guardrail-pipelines, Property 22: Re-injection runs
    //        exactly once as the final post-call step
    // **Validates: Requirements 9.5**
    //
    // After non-halting post-call stages, re-injection restores every
    // placeholder occurrence (a single final pass) and only when the context is
    // non-empty; a halting action skips re-injection entirely.
    // -----------------------------------------------------------------
    proptest! {
        #![proptest_config(ProptestConfig::with_cases(64))]

        #[test]
        fn prop22_reinjection_final_once(
            value in prop::sample::select(vec!["john@x.com", "topsecret", "123-45-6789"]),
            label in prop::sample::select(vec!["EMAIL", "SECRET", "SSN"]),
            k in 1usize..4,
        ) {
            let sel = BindingSelector::default();
            let placeholder = format!("<<PII_{label}_1>>");

            // --- Non-halting: placeholder restored to original everywhere. ---
            let engine = make_engine_multi(vec![
                StageSpec {
                    provider: Arc::new(Substr(value, label)),
                    action: PolicyAction::Redact,
                    phase: StagePhase::PreCall,
                    policy: FailurePolicy::FailClose,
                    timeout_secs: 5,
                },
                StageSpec {
                    provider: Arc::new(WholeMatch("Z")),
                    action: PolicyAction::Allow,
                    phase: StagePhase::PostCall,
                    policy: FailurePolicy::FailClose,
                    timeout_secs: 5,
                },
            ]);
            let mut req = user_request(&format!("contact {value} please"));
            let mut ctx = GuardrailContext::new();
            let pre = rt().block_on(engine.run_pre_call(&mut req, &sel, &mut ctx, "t"));
            prop_assert_eq!(pre, PreCallOutcome::Proceed);
            prop_assert!(!ctx.is_empty());

            let echoed = vec![placeholder.clone(); k].join(" and ");
            let mut resp = assistant_response(&echoed);
            let post = rt().block_on(engine.run_post_call(&mut resp, &sel, &mut ctx, "t", &no_tool_ctx()));
            prop_assert_eq!(post.0, PostCallOutcome::Proceed);
            let out = assistant_text(&resp);
            prop_assert!(!out.contains(&placeholder));
            prop_assert_eq!(out.matches(value).count(), k);

            // --- Empty context: no re-injection, response unchanged. ---
            let mut ctx_empty = GuardrailContext::new();
            let mut resp2 = assistant_response(&placeholder);
            let before = serde_json::to_string(&resp2).unwrap();
            let post2 = rt().block_on(engine.run_post_call(&mut resp2, &sel, &mut ctx_empty, "t", &no_tool_ctx()));
            let after = serde_json::to_string(&resp2).unwrap();
            prop_assert_eq!(post2, (PostCallOutcome::Proceed, RefusalDecision::NotRefusal));
            prop_assert_eq!(&before, &after);

            // --- Halting action skips re-injection. ---
            let engine_halt = make_engine_multi(vec![
                StageSpec {
                    provider: Arc::new(Substr(value, label)),
                    action: PolicyAction::Redact,
                    phase: StagePhase::PreCall,
                    policy: FailurePolicy::FailClose,
                    timeout_secs: 5,
                },
                StageSpec {
                    provider: Arc::new(WholeMatch("Z")),
                    action: PolicyAction::ReplaceWithPolicyMessage,
                    phase: StagePhase::PostCall,
                    policy: FailurePolicy::FailClose,
                    timeout_secs: 5,
                },
            ]);
            let mut req_h = user_request(&format!("contact {value} please"));
            let mut ctx_h = GuardrailContext::new();
            rt().block_on(engine_halt.run_pre_call(&mut req_h, &sel, &mut ctx_h, "t"));
            prop_assert!(!ctx_h.is_empty());
            let mut resp_h = assistant_response(&placeholder);
            let post_h = rt().block_on(engine_halt.run_post_call(&mut resp_h, &sel, &mut ctx_h, "t", &no_tool_ctx()));
            prop_assert_eq!(post_h.0, PostCallOutcome::Replaced);
            // Re-injection was skipped: the original value was never restored.
            prop_assert!(!assistant_text(&resp_h).contains(value));
        }
    }

    // -----------------------------------------------------------------
    // 10.10 — Feature: guardrail-pipelines, Property 23: fail_open skips,
    //         fail_close halts
    // **Validates: Requirements 8.6, 9.6, 9.7**
    // -----------------------------------------------------------------
    proptest! {
        #![proptest_config(ProptestConfig::with_cases(64))]

        #[test]
        fn prop23_failure_policy(is_pre in any::<bool>()) {
            let sel = BindingSelector::default();
            let phase = if is_pre { StagePhase::PreCall } else { StagePhase::PostCall };

            // fail_open: failing stage skipped, subsequent stage still executes.
            let log = Arc::new(Mutex::new(Vec::<usize>::new()));
            let engine_open = make_engine_multi(vec![
                StageSpec {
                    provider: Arc::new(Failing),
                    action: PolicyAction::Redact,
                    phase,
                    policy: FailurePolicy::FailOpen,
                    timeout_secs: 5,
                },
                StageSpec {
                    provider: Arc::new(Recorder {
                        id: 1,
                        log: log.clone(),
                        produce_finding: false,
                    }),
                    action: PolicyAction::Allow,
                    phase,
                    policy: FailurePolicy::FailClose,
                    timeout_secs: 5,
                },
            ]);
            let mut ctx = GuardrailContext::new();
            if is_pre {
                let mut req = user_request("content");
                let o = rt().block_on(engine_open.run_pre_call(&mut req, &sel, &mut ctx, "t"));
                prop_assert_eq!(o, PreCallOutcome::Proceed);
            } else {
                let mut resp = assistant_response("content");
                let o = rt().block_on(engine_open.run_post_call(&mut resp, &sel, &mut ctx, "t", &no_tool_ctx()));
                prop_assert_eq!(o.0, PostCallOutcome::Proceed);
            }
            // The failing (skipped) stage recorded nothing; the next stage ran.
            prop_assert_eq!(log.lock().unwrap().clone(), vec![1usize]);

            // fail_close: failing stage halts with a service failure.
            let engine_close = make_engine(
                Arc::new(Failing),
                FailurePolicy::FailClose,
                vec![stage("s", phase, PolicyAction::Block)],
            );
            let mut ctx2 = GuardrailContext::new();
            if is_pre {
                let mut req = user_request("content");
                let o = rt().block_on(engine_close.run_pre_call(&mut req, &sel, &mut ctx2, "t"));
                prop_assert_eq!(o, PreCallOutcome::ServiceFailure);
            } else {
                let mut resp = assistant_response("content");
                let o = rt().block_on(engine_close.run_post_call(&mut resp, &sel, &mut ctx2, "t", &no_tool_ctx()));
                prop_assert_eq!(o.0, PostCallOutcome::ServiceFailure);
            }
        }
    }

    // -----------------------------------------------------------------
    // 10.11 — Unit tests for engine error mapping.
    // **Validates: Requirements 2.7, 2.9, 3.6, 8.6**
    // -----------------------------------------------------------------

    /// Req 2.9, 8.6: a `fail_close` provider that exceeds its scan-latency
    /// budget resolves to a scan-timeout outcome. A 0-second timeout makes the
    /// slow provider exceed the budget immediately.
    #[tokio::test]
    async fn scan_timeout_maps_to_timeout_outcome() {
        let engine = make_engine_multi(vec![StageSpec {
            provider: Arc::new(Slow),
            action: PolicyAction::Block,
            phase: StagePhase::PreCall,
            policy: FailurePolicy::FailClose,
            timeout_secs: 0,
        }]);
        let mut req = user_request("content");
        let mut ctx = GuardrailContext::new();
        let outcome = engine
            .run_pre_call(&mut req, &BindingSelector::default(), &mut ctx, "t")
            .await;
        assert_eq!(outcome, PreCallOutcome::Timeout);
    }

    /// Req 2.9, 8.6: the same timeout on a post-call `fail_close` stage maps to
    /// a service failure (post-call has no distinct timeout outcome).
    #[tokio::test]
    async fn scan_timeout_post_call_maps_to_service_failure() {
        let engine = make_engine_multi(vec![StageSpec {
            provider: Arc::new(Slow),
            action: PolicyAction::Block,
            phase: StagePhase::PostCall,
            policy: FailurePolicy::FailClose,
            timeout_secs: 0,
        }]);
        let mut resp = assistant_response("content");
        let mut ctx = GuardrailContext::new();
        let outcome = engine
            .run_post_call(
                &mut resp,
                &BindingSelector::default(),
                &mut ctx,
                "t",
                &no_tool_ctx(),
            )
            .await;
        assert_eq!(outcome.0, PostCallOutcome::ServiceFailure);
    }

    /// Req 3.6: a response containing no assistant-role messages is forwarded
    /// unmodified (post-call scanning is skipped).
    #[tokio::test]
    async fn assistant_less_response_forwarded_unmodified() {
        let engine = make_engine(
            Arc::new(WholeMatch("SECRET")),
            FailurePolicy::FailClose,
            vec![stage("out", StagePhase::PostCall, PolicyAction::Block)],
        );
        let mut resp = OpenAIResponse {
            id: "abc".to_string(),
            object: "chat.completion".to_string(),
            created: 1,
            model: "m".to_string(),
            choices: vec![Choice {
                index: 0,
                message: Message {
                    // Not an assistant message → skipped by the scanner.
                    role: "user".to_string(),
                    content: Value::String("some content".to_string()),
                    extra: serde_json::Map::new(),
                },
                finish_reason: Some("stop".to_string()),
                extra: serde_json::Map::new(),
            }],
            usage: Usage::default(),
            extra: serde_json::Map::new(),
        };
        let before = serde_json::to_string(&resp).unwrap();
        let mut ctx = GuardrailContext::new();
        let outcome = engine
            .run_post_call(
                &mut resp,
                &BindingSelector::default(),
                &mut ctx,
                "t",
                &no_tool_ctx(),
            )
            .await;
        let after = serde_json::to_string(&resp).unwrap();
        assert_eq!(outcome.0, PostCallOutcome::Proceed);
        assert_eq!(before, after);
    }

    /// Req 3.6: a response with no choices at all is forwarded unmodified.
    #[tokio::test]
    async fn empty_choices_response_forwarded_unmodified() {
        let engine = make_engine(
            Arc::new(WholeMatch("SECRET")),
            FailurePolicy::FailClose,
            vec![stage("out", StagePhase::PostCall, PolicyAction::Block)],
        );
        let mut resp = OpenAIResponse {
            id: String::new(),
            object: String::new(),
            created: 0,
            model: "m".to_string(),
            choices: vec![],
            usage: Usage::default(),
            extra: serde_json::Map::new(),
        };
        let before = serde_json::to_string(&resp).unwrap();
        let mut ctx = GuardrailContext::new();
        let outcome = engine
            .run_post_call(
                &mut resp,
                &BindingSelector::default(),
                &mut ctx,
                "t",
                &no_tool_ctx(),
            )
            .await;
        let after = serde_json::to_string(&resp).unwrap();
        assert_eq!(outcome.0, PostCallOutcome::Proceed);
        assert_eq!(before, after);
    }

    /// Req 2.7: an action invalid for the pre-call phase
    /// (`replace_with_policy_message`) maps to the invalid-action outcome, even
    /// when it appears after a valid earlier stage.
    #[tokio::test]
    async fn invalid_pre_call_action_after_valid_stage() {
        let engine = make_engine_multi(vec![
            StageSpec {
                provider: Arc::new(WholeMatch("X")),
                action: PolicyAction::Allow,
                phase: StagePhase::PreCall,
                policy: FailurePolicy::FailClose,
                timeout_secs: 5,
            },
            StageSpec {
                provider: Arc::new(WholeMatch("X")),
                action: PolicyAction::ReplaceWithPolicyMessage,
                phase: StagePhase::PreCall,
                policy: FailurePolicy::FailClose,
                timeout_secs: 5,
            },
        ]);
        let mut req = user_request("content");
        let mut ctx = GuardrailContext::new();
        let outcome = engine
            .run_pre_call(&mut req, &BindingSelector::default(), &mut ctx, "t")
            .await;
        assert_eq!(outcome, PreCallOutcome::InvalidAction);
    }

    // -----------------------------------------------------------------
    // 12.2 — Metrics/log emission wiring.
    // **Validates: Requirements 11.1, 11.2, 11.6, 11.7**
    //
    // Minimal coverage that per-stage counters/latency are recorded on the
    // engine's `Arc<Metrics>` with the correct action label. Detailed field
    // assertions (INFO/WARN, summary) are task 12.3.
    // -----------------------------------------------------------------

    use crate::metrics::Metrics;

    /// Build an engine wired to `metrics` with a single global-default pipeline
    /// composed of `stages`, all backed by one shared `provider`.
    fn make_engine_with_metrics(
        provider: Arc<dyn GuardrailProvider>,
        policy: FailurePolicy,
        stages: Vec<StageConfig>,
        metrics: Arc<Metrics>,
    ) -> GuardrailEngine {
        let mut registry = ProviderRegistry::new();
        registry.insert("p", provider);
        let config = GuardrailConfig {
            providers: vec![GuardrailProviderConfig {
                name: "p".to_string(),
                provider_type: GuardrailProviderType::Regex,
                failure_policy: policy,
                timeout_seconds: 5,
                settings: ProviderSettings::default(),
            }],
            pipelines: vec![PipelineConfig {
                name: "pl".to_string(),
                stages,
                redaction_notice_instruction: None,
                instruction_insertion_mode: InstructionInsertionMode::default(),
                failover_on_refusal: false,
                refusal_phrase_list: None,
            }],
            global_default_pipeline: Some("pl".to_string()),
            bindings: GuardrailBindings::default(),
            ..Default::default()
        };
        GuardrailEngine::new(&config, &registry, Some(metrics)).unwrap()
    }

    /// Req 11.1, 11.2: a redact stage bumps the execution counter with
    /// `action="redact"` and observes a latency sample for the stage.
    #[tokio::test]
    async fn records_counter_and_latency_for_redact() {
        let metrics = Arc::new(Metrics::new());
        let engine = make_engine_with_metrics(
            Arc::new(Substr("john@x.com", "EMAIL")),
            FailurePolicy::FailClose,
            vec![stage("pii", StagePhase::PreCall, PolicyAction::Redact)],
            metrics.clone(),
        );
        let mut req = user_request("contact john@x.com now");
        let mut ctx = GuardrailContext::new();
        let outcome = engine
            .run_pre_call(&mut req, &BindingSelector::default(), &mut ctx, "trace-1")
            .await;
        assert_eq!(outcome, PreCallOutcome::Proceed);

        let mut out = String::new();
        metrics.write_guardrail_prometheus(&mut out);
        assert!(out.contains(
            "obey_api_guardrail_stage_executions_total{pipeline=\"pl\",stage=\"pii\",provider=\"regex\",action=\"redact\"} 1"
        ));
        assert!(out.contains(
            "obey_api_guardrail_stage_latency_ms_count{pipeline=\"pl\",stage=\"pii\",provider=\"regex\"} 1"
        ));
    }

    /// Req 11.1: a stage whose provider reports no findings is counted as
    /// `action="pass"`.
    #[tokio::test]
    async fn records_pass_action_when_no_findings() {
        let metrics = Arc::new(Metrics::new());
        let engine = make_engine_with_metrics(
            Arc::new(Substr("absent-needle", "TERM")),
            FailurePolicy::FailClose,
            vec![stage("scan", StagePhase::PreCall, PolicyAction::Redact)],
            metrics.clone(),
        );
        let mut req = user_request("nothing to see here");
        let mut ctx = GuardrailContext::new();
        let _ = engine
            .run_pre_call(&mut req, &BindingSelector::default(), &mut ctx, "t")
            .await;

        let mut out = String::new();
        metrics.write_guardrail_prometheus(&mut out);
        assert!(out.contains(
            "obey_api_guardrail_stage_executions_total{pipeline=\"pl\",stage=\"scan\",provider=\"regex\",action=\"pass\"} 1"
        ));
    }

    /// Req 11.6: a provider error increments the counter with `action="error"`
    /// regardless of failure policy (here `fail_open`, which still proceeds).
    #[tokio::test]
    async fn records_error_action_on_provider_failure() {
        let metrics = Arc::new(Metrics::new());
        let engine = make_engine_with_metrics(
            Arc::new(Failing),
            FailurePolicy::FailOpen,
            vec![stage("scan", StagePhase::PreCall, PolicyAction::Redact)],
            metrics.clone(),
        );
        let mut req = user_request("some content");
        let mut ctx = GuardrailContext::new();
        let outcome = engine
            .run_pre_call(&mut req, &BindingSelector::default(), &mut ctx, "t")
            .await;
        assert_eq!(outcome, PreCallOutcome::Proceed);

        let mut out = String::new();
        metrics.write_guardrail_prometheus(&mut out);
        assert!(out.contains(
            "obey_api_guardrail_stage_executions_total{pipeline=\"pl\",stage=\"scan\",provider=\"regex\",action=\"error\"} 1"
        ));
    }

    /// Req 11.7: with no metrics handle, enforcement still works (best-effort
    /// recording is a no-op and never fails the request).
    #[tokio::test]
    async fn no_metrics_handle_does_not_fail_request() {
        let engine = make_engine(
            Arc::new(Substr("bad", "TERM")),
            FailurePolicy::FailClose,
            vec![stage("out", StagePhase::PostCall, PolicyAction::Redact)],
        );
        let mut resp = assistant_response("this is bad");
        let mut ctx = GuardrailContext::new();
        let outcome = engine
            .run_post_call(
                &mut resp,
                &BindingSelector::default(),
                &mut ctx,
                "t",
                &no_tool_ctx(),
            )
            .await;
        assert_eq!(outcome.0, PostCallOutcome::Proceed);
        assert_eq!(assistant_text(&resp), "this is [REDACTED]");
    }

    // -----------------------------------------------------------------
    // Task 12.3 — metric/log field coverage complementary to 12.2.
    //
    // 12.2 already asserts the `redact`, `pass`, and `error` action labels
    // and one latency `_count` sample. The tests below add the remaining
    // action labels (`mask`, `replace_with_policy_message`, `block`), verify
    // the latency histogram `_count` increments once per stage execution
    // (Req 11.2), and assert the request-summary fields (Req 11.4).
    //
    // NOTE ON SUMMARY CAPTURE: the per-phase guardrail summary
    // (`stages_executed`, non-pass `actions`, `total_latency_ms`) is emitted
    // exclusively via `tracing::info!`. `crates/ai-gateway/Cargo.toml` has no
    // tracing-capture dev-dependency (`tracing-test`/subscriber capture), so
    // the summary fields are asserted INDIRECTLY through the per-stage
    // counters that feed the summary: the sum of execution-counter samples
    // equals `stages_executed`, and the non-`pass` counter lines equal the
    // summary's `actions` list.
    // -----------------------------------------------------------------

    /// Sum of every guardrail execution-counter sample in `prom`. Equals the
    /// number of stage executions recorded, i.e. the summary `stages_executed`.
    fn total_stage_executions(prom: &str) -> u64 {
        prom.lines()
            .filter(|l| l.starts_with("obey_api_guardrail_stage_executions_total{"))
            .filter_map(|l| l.rsplit(' ').next())
            .filter_map(|v| v.trim().parse::<u64>().ok())
            .sum()
    }

    /// Req 11.1: a pre-call `mask` stage that modifies content is counted with
    /// `action="mask"` and observes one latency sample.
    #[tokio::test]
    async fn records_mask_action_label() {
        let metrics = Arc::new(Metrics::new());
        let engine = make_engine_with_metrics(
            Arc::new(Substr("john@x.com", "EMAIL")),
            FailurePolicy::FailClose,
            vec![stage("pii", StagePhase::PreCall, PolicyAction::Mask)],
            metrics.clone(),
        );
        let mut req = user_request("contact john@x.com now");
        let mut ctx = GuardrailContext::new();
        let outcome = engine
            .run_pre_call(&mut req, &BindingSelector::default(), &mut ctx, "t")
            .await;
        assert_eq!(outcome, PreCallOutcome::Proceed);
        // Byte-preserving mask applied in place.
        assert_eq!(req.messages[0].content_as_text(), "contact ********** now");

        let mut out = String::new();
        metrics.write_guardrail_prometheus(&mut out);
        assert!(out.contains(
            "obey_api_guardrail_stage_executions_total{pipeline=\"pl\",stage=\"pii\",provider=\"regex\",action=\"mask\"} 1"
        ));
        assert!(out.contains(
            "obey_api_guardrail_stage_latency_ms_count{pipeline=\"pl\",stage=\"pii\",provider=\"regex\"} 1"
        ));
    }

    /// Req 11.1: a post-call `replace_with_policy_message` stage that fires is
    /// counted with `action="replace_with_policy_message"`.
    #[tokio::test]
    async fn records_replace_with_policy_message_action_label() {
        let metrics = Arc::new(Metrics::new());
        let engine = make_engine_with_metrics(
            Arc::new(WholeMatch("POLICY")),
            FailurePolicy::FailClose,
            vec![stage(
                "out",
                StagePhase::PostCall,
                PolicyAction::ReplaceWithPolicyMessage,
            )],
            metrics.clone(),
        );
        let mut resp = assistant_response("disallowed content");
        let mut ctx = GuardrailContext::new();
        let outcome = engine
            .run_post_call(
                &mut resp,
                &BindingSelector::default(),
                &mut ctx,
                "t",
                &no_tool_ctx(),
            )
            .await;
        assert_eq!(outcome.0, PostCallOutcome::Replaced);

        let mut out = String::new();
        metrics.write_guardrail_prometheus(&mut out);
        assert!(out.contains(
            "obey_api_guardrail_stage_executions_total{pipeline=\"pl\",stage=\"out\",provider=\"regex\",action=\"replace_with_policy_message\"} 1"
        ));
        assert!(out.contains(
            "obey_api_guardrail_stage_latency_ms_count{pipeline=\"pl\",stage=\"out\",provider=\"regex\"} 1"
        ));
    }

    /// Req 11.1: a post-call `block` stage that halts is counted with
    /// `action="block"`.
    #[tokio::test]
    async fn records_block_action_label_post_call() {
        let metrics = Arc::new(Metrics::new());
        let engine = make_engine_with_metrics(
            Arc::new(WholeMatch("SECRET")),
            FailurePolicy::FailClose,
            vec![stage("blk", StagePhase::PostCall, PolicyAction::Block)],
            metrics.clone(),
        );
        let mut resp = assistant_response("leaked data");
        let mut ctx = GuardrailContext::new();
        let outcome = engine
            .run_post_call(
                &mut resp,
                &BindingSelector::default(),
                &mut ctx,
                "t",
                &no_tool_ctx(),
            )
            .await;
        assert!(matches!(outcome.0, PostCallOutcome::Block(_)));

        let mut out = String::new();
        metrics.write_guardrail_prometheus(&mut out);
        assert!(out.contains(
            "obey_api_guardrail_stage_executions_total{pipeline=\"pl\",stage=\"blk\",provider=\"regex\",action=\"block\"} 1"
        ));
    }

    /// Req 11.2: the latency histogram `_count` for a stage increments once per
    /// stage execution across repeated requests.
    #[tokio::test]
    async fn latency_histogram_count_increments_per_stage_execution() {
        let metrics = Arc::new(Metrics::new());
        let engine = make_engine_with_metrics(
            Arc::new(Substr("x", "T")),
            FailurePolicy::FailClose,
            vec![stage("scan", StagePhase::PreCall, PolicyAction::Redact)],
            metrics.clone(),
        );
        for _ in 0..3 {
            let mut req = user_request("x");
            let mut ctx = GuardrailContext::new();
            let _ = engine
                .run_pre_call(&mut req, &BindingSelector::default(), &mut ctx, "t")
                .await;
        }

        let mut out = String::new();
        metrics.write_guardrail_prometheus(&mut out);
        // One latency observation per stage execution (3 requests → 3).
        assert!(out.contains(
            "obey_api_guardrail_stage_latency_ms_count{pipeline=\"pl\",stage=\"scan\",provider=\"regex\"} 3"
        ));
        assert_eq!(total_stage_executions(&out), 3);
    }

    /// Req 11.4: the per-request summary reports `stages_executed`, the non-pass
    /// `actions` list, and `total_latency_ms`. Asserted indirectly through the
    /// per-stage counters that feed the summary (see module NOTE):
    /// two stages execute (stages_executed == 2), stage "a" redacts (a non-pass
    /// entry in the actions list) and stage "b" — seeing only the placeholder —
    /// passes (excluded from the list). One latency sample per stage confirms
    /// total latency aggregates both executions.
    #[tokio::test]
    async fn stage_counters_reflect_summary_stages_and_actions() {
        let metrics = Arc::new(Metrics::new());
        let engine = make_engine_with_metrics(
            Arc::new(Substr("SECRET", "X")),
            FailurePolicy::FailClose,
            vec![
                stage("a", StagePhase::PreCall, PolicyAction::Redact),
                stage("b", StagePhase::PreCall, PolicyAction::Redact),
            ],
            metrics.clone(),
        );
        let mut req = user_request("my SECRET data");
        let mut ctx = GuardrailContext::new();
        let outcome = engine
            .run_pre_call(&mut req, &BindingSelector::default(), &mut ctx, "t")
            .await;
        assert_eq!(outcome, PreCallOutcome::Proceed);

        let mut out = String::new();
        metrics.write_guardrail_prometheus(&mut out);

        // summary.stages_executed == 2.
        assert_eq!(total_stage_executions(&out), 2);
        // Stage "a" redacted the sole occurrence → non-pass entry in the list.
        assert!(out.contains(
            "obey_api_guardrail_stage_executions_total{pipeline=\"pl\",stage=\"a\",provider=\"regex\",action=\"redact\"} 1"
        ));
        // Stage "b" saw only the placeholder → no findings → pass (not listed).
        assert!(out.contains(
            "obey_api_guardrail_stage_executions_total{pipeline=\"pl\",stage=\"b\",provider=\"regex\",action=\"pass\"} 1"
        ));
        // total_latency_ms aggregates one sample per executed stage (Req 11.2).
        assert!(out.contains(
            "obey_api_guardrail_stage_latency_ms_count{pipeline=\"pl\",stage=\"a\",provider=\"regex\"} 1"
        ));
        assert!(out.contains(
            "obey_api_guardrail_stage_latency_ms_count{pipeline=\"pl\",stage=\"b\",provider=\"regex\"} 1"
        ));
    }

    /// Req 11.7: with no metrics handle, the INFO (enforcement) and WARN
    /// (provider error) log paths are still exercised and enforcement still
    /// applies — best-effort recording never fails the request.
    #[tokio::test]
    async fn no_metrics_handle_warn_and_info_paths_still_enforce() {
        // WARN path: provider error under fail_open still proceeds.
        let warn_engine = make_engine(
            Arc::new(Failing),
            FailurePolicy::FailOpen,
            vec![stage("scan", StagePhase::PreCall, PolicyAction::Redact)],
        );
        let mut req = user_request("some content");
        let mut ctx = GuardrailContext::new();
        assert_eq!(
            warn_engine
                .run_pre_call(&mut req, &BindingSelector::default(), &mut ctx, "t")
                .await,
            PreCallOutcome::Proceed
        );

        // INFO path: a block enforcement still halts without a metrics handle.
        let block_engine = make_engine(
            Arc::new(WholeMatch("KEY")),
            FailurePolicy::FailClose,
            vec![stage("secret", StagePhase::PreCall, PolicyAction::Block)],
        );
        let mut req2 = user_request("my key");
        let mut ctx2 = GuardrailContext::new();
        assert!(matches!(
            block_engine
                .run_pre_call(&mut req2, &BindingSelector::default(), &mut ctx2, "t")
                .await,
            PreCallOutcome::Block(_)
        ));
    }
}
