//! Reasoning-compatibility layer for the OBEY gateway.
//!
//! This module implements a reasoning-aware transformation stage that runs
//! per-failover-attempt on cloned outgoing requests. It detects reasoning state
//! that clients replay in prior assistant turns, applies a strip/preserve policy
//! keyed off the source vs target model family, normalizes reasoning parameters,
//! and feeds reasoning-token usage into the cost-attribution pipeline.
//!
//! The layer is opt-in via `reasoning_compat.enabled: true` (default: true).
//! With `enabled: false`, the gateway reproduces exact current behavior (passthrough).
//!
//! Components:
//! - **config**: Configuration types and validation (ReasoningCompatConfig, Effort, etc.)
//! - **detect**: Carrier detection (thinking blocks, reasoning_content, etc.) — Task 2
//! - **policy**: Strip/preserve decision logic — Task 3
//! - **normalize**: Parameter normalization (reasoning_effort ↔ budget_tokens, etc.) — Task 4
//! - **cost**: Reasoning-token cost attribution — Task 4
//!
//! Flow: detect → policy → normalize → cost

pub mod config;
pub mod cost;
pub mod detect;
pub mod normalize;
pub mod policy;

#[cfg(test)]
mod tests;

pub use config::ReasoningCompatConfig;

use crate::config::ProviderModel;
use crate::models::openai::OpenAIRequest;

/// Counts-only record of what one [`prepare_attempt`] pass did. Never
/// contains payloads (no thinking text, no signatures, no redacted data).
#[derive(Debug, Clone, Copy)]
pub struct AttemptReport {
    /// The strip/preserve decision that was applied.
    pub decision: policy::StripDecision,
    /// What [`policy::apply`] removed from the outgoing request.
    pub strip: policy::StripReport,
    /// What [`normalize::emit_for_target`] emitted for the target family.
    pub normalized: normalize::NormalizeReport,
}

impl AttemptReport {
    /// Compact JSON summary of the actions taken (decision, counts, and
    /// emitted shape only), or `None` when the attempt was a no-op
    /// passthrough. Safe for logging: never carries reasoning payloads,
    /// signatures, or redacted data.
    pub fn actions_json(self) -> Option<String> {
        let strip_acted = self.strip != policy::StripReport::default();
        let normalize_acted = self.normalized.emitted_shape != "none";
        if !strip_acted && !normalize_acted {
            return None;
        }
        Some(
            serde_json::json!({
                "action": self.decision.as_str(),
                "messages_touched": self.strip.messages_touched,
                "thinking_blocks": self.strip.thinking_blocks,
                "redacted_thinking_blocks": self.strip.redacted_thinking_blocks,
                "fields_removed": self.strip.fields_removed,
                "normalized_shape": self.normalized.emitted_shape,
            })
            .to_string(),
        )
    }
}

/// Build the policy [`ModelRef`](policy::ModelRef) for a failover target:
/// the configured `reasoning_family` wins, otherwise the model id is
/// classified from its name.
pub fn target_model_ref(target: &ProviderModel) -> policy::ModelRef {
    policy::ModelRef {
        provider: target.provider.clone(),
        model: target.model.clone(),
        family: target
            .reasoning_family
            .unwrap_or_else(|| detect::classify_family(&target.model)),
    }
}

/// Per-attempt transform: detect → policy → strip → normalize.
///
/// Runs on the cloned outgoing request for one failover attempt:
/// 1. detect reasoning carriers on the ORIGINAL request's messages
///    (pre-transform — carriers live there, compression aside)
/// 2. decide strip/preserve for the source→target transition and apply it
///    to `outgoing`
/// 3. read the client's reasoning parameter from `outgoing` and re-emit it
///    in the shape the target family accepts
///
/// Total (fail-open): all steps match defensively on odd shapes; there is
/// no panic path, so callers need no recovery wrapper.
pub fn prepare_attempt(
    outgoing: &mut OpenAIRequest,
    original_request: &OpenAIRequest,
    source: Option<policy::ModelRef>,
    target: &ProviderModel,
    cfg: &ReasoningCompatConfig,
) -> AttemptReport {
    let footprint = detect::detect(&original_request.messages);
    let target_ref = target_model_ref(target);
    let decision = policy::decide(&footprint, source.as_ref(), &target_ref, cfg);
    let strip = policy::apply(outgoing, decision);
    let spec = normalize::read_client_spec(outgoing);
    let normalized = normalize::emit_for_target(outgoing, spec, target, cfg);
    AttemptReport {
        decision,
        strip,
        normalized,
    }
}
