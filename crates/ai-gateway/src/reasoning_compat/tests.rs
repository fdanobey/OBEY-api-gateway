//! Cross-component tests for the reasoning-compat layer.
//!
//! Unlike the per-file inline test modules (config, detect, policy,
//! normalize, cost), this suite exercises the components TOGETHER:
//! end-to-end [`prepare_attempt`] failover flows and module-interaction
//! contracts (detection→policy, policy→normalize, normalize round-trips,
//! cost extraction→pricing chains).

use super::*;
use crate::config::ProviderModel;
use crate::models::openai::{Message, OpenAIRequest, Usage};
use crate::reasoning_compat::config::{Effort, ReasoningFamily, ReasoningParamShape};
use crate::reasoning_compat::cost::{extract_reasoning_usage, reasoning_cost, ReasoningCarrier};
use crate::reasoning_compat::detect::detect;
use crate::reasoning_compat::normalize::{emit_for_target, read_client_spec};
use crate::reasoning_compat::policy::{decide, ModelRef, StripDecision, StripReport};
use serde_json::{json, Map, Value};

// ---------------------------------------------------------------------------
// Fixture helpers
// ---------------------------------------------------------------------------

/// Assistant message with a `thinking` block (signed when `signature` is
/// true) next to a plain text block.
fn msg_with_thinking(signature: bool) -> Message {
    let mut block = json!({"type": "thinking", "thinking": "deep-reasoning-payload"});
    if signature {
        block["signature"] = json!("sig-abc");
    }
    Message {
        role: "assistant".to_string(),
        content: json!([block, {"type": "text", "text": "answer"}]),
        extra: Map::new(),
    }
}

/// Assistant message with a `redacted_thinking` block next to text.
fn msg_with_redacted() -> Message {
    Message {
        role: "assistant".to_string(),
        content: json!([
            {"type": "redacted_thinking", "data": "opaque-redacted-data"},
            {"type": "text", "text": "answer"}
        ]),
        extra: Map::new(),
    }
}

/// Assistant message carrying a DeepSeek-style `reasoning_content` field.
fn msg_with_reasoning_content(text: &str) -> Message {
    let mut extra = Map::new();
    extra.insert("reasoning_content".to_string(), json!(text));
    Message {
        role: "assistant".to_string(),
        content: Value::String("answer".to_string()),
        extra,
    }
}

/// Assistant message carrying an OpenRouter-style `reasoning` field.
fn msg_with_reasoning_field(text: &str) -> Message {
    let mut extra = Map::new();
    extra.insert("reasoning".to_string(), json!(text));
    Message {
        role: "assistant".to_string(),
        content: Value::String("answer".to_string()),
        extra,
    }
}

/// Plain user text message.
fn msg_text(content: &str) -> Message {
    Message {
        role: "user".to_string(),
        content: Value::String(content.to_string()),
        extra: Map::new(),
    }
}

/// ProviderModel fixture (input price 3.0, output price 5.0, no dedicated
/// reasoning price; family/shape `None` defer to runtime classification).
fn pm(
    provider: &str,
    model: &str,
    family: Option<ReasoningFamily>,
    shape: Option<ReasoningParamShape>,
) -> ProviderModel {
    ProviderModel {
        provider: provider.to_string(),
        model: model.to_string(),
        cost_per_million_input_tokens: 3.0,
        cost_per_million_output_tokens: 5.0,
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

/// Source/target model reference for policy decisions.
fn model_ref(provider: &str, model: &str, family: ReasoningFamily) -> ModelRef {
    ModelRef {
        provider: provider.to_string(),
        model: model.to_string(),
        family,
    }
}

/// Bare request fixture.
fn req(msgs: Vec<Message>) -> OpenAIRequest {
    OpenAIRequest {
        model: "test-model".to_string(),
        messages: msgs,
        stream: false,
        temperature: None,
        max_tokens: None,
        extra: Map::new(),
    }
}

/// Usage fixture from raw JSON (extra fields land in the flattened map).
fn usage_from_json(value: Value) -> Usage {
    serde_json::from_value(value).expect("test usage JSON is valid")
}

// ---------------------------------------------------------------------------
// 1. prepare_attempt end-to-end flows
// ---------------------------------------------------------------------------

#[test]
fn cross_model_failover_strips_all_carriers_and_reports_safe_summary() {
    let cfg = ReasoningCompatConfig::default();
    let request = req(vec![
        msg_text("check the weather"),
        msg_with_thinking(true),
        msg_with_redacted(),
    ]);
    // claude-4-5 → claude-4-7: different resolved model, so signed AND
    // redacted thinking state must both go (regression: thinking-only
    // filtering leaves redacted_thinking behind).
    let source = Some(model_ref(
        "anthropic",
        "claude-sonnet-4-5",
        ReasoningFamily::AnthropicManual,
    ));
    let target = pm("anthropic", "claude-sonnet-4-7", None, None);
    let mut outgoing = request.clone();

    let report = prepare_attempt(&mut outgoing, &request, source, &target, &cfg);

    assert_eq!(report.decision, StripDecision::StripAll);
    assert_eq!(report.strip.messages_touched, 2);
    assert_eq!(report.strip.thinking_blocks, 1);
    assert_eq!(report.strip.redacted_thinking_blocks, 1);

    // No carrier survives anywhere in the conversation; untouched user
    // turn and plain text blocks do.
    assert_eq!(outgoing.messages.len(), 3);
    for message in &outgoing.messages {
        if let Some(blocks) = message.content.as_array() {
            for block in blocks {
                let block_type = block.get("type").and_then(Value::as_str);
                assert_ne!(block_type, Some("thinking"));
                assert_ne!(block_type, Some("redacted_thinking"));
            }
        }
    }

    // Counts-only JSON summary: both block families + counts present,
    // no payloads, signatures, or redacted data.
    let actions = report.actions_json().expect("strip acted, summary due");
    assert!(actions.contains("\"action\":\"strip_all\""));
    assert!(actions.contains("\"thinking_blocks\":1"));
    assert!(actions.contains("\"redacted_thinking_blocks\":1"));
    assert!(!actions.contains("deep-reasoning-payload"));
    assert!(!actions.contains("sig-abc"));
    assert!(!actions.contains("opaque-redacted-data"));
}

#[test]
fn same_model_continuation_preserves_messages_verbatim() {
    let cfg = ReasoningCompatConfig::default();
    let request = req(vec![msg_with_thinking(true), msg_with_redacted()]);
    // Same resolved provider + model + family (mid-tool-loop echo): the
    // signed and redacted blocks must survive bit-identical.
    let source = model_ref(
        "anthropic",
        "claude-sonnet-4-5",
        ReasoningFamily::AnthropicManual,
    );
    let target = pm("anthropic", "claude-sonnet-4-5", None, None);
    let mut outgoing = request.clone();

    let report = prepare_attempt(&mut outgoing, &request, Some(source), &target, &cfg);

    assert_eq!(report.decision, StripDecision::Preserve);
    assert_eq!(report.strip, StripReport::default());
    assert_eq!(
        serde_json::to_value(&outgoing.messages).unwrap(),
        serde_json::to_value(&request.messages).unwrap()
    );
    assert_eq!(report.actions_json(), None);
}

#[test]
fn attribution_unknown_cross_family_strips_reasoning_content() {
    let cfg = ReasoningCompatConfig::default();
    let request = req(vec![
        msg_with_reasoning_content("chain-of-thought-text"),
        msg_text("continue"),
    ]);
    // No affinity record + DeepSeek carriers + OpenAI target: conservative
    // strip logged as attribution-unknown.
    let target = pm("openai", "gpt-5", None, None);
    let mut outgoing = request.clone();

    let report = prepare_attempt(&mut outgoing, &request, None, &target, &cfg);

    assert_eq!(report.decision, StripDecision::StripAttributionUnknown);
    assert_eq!(report.strip.fields_removed, 1);
    assert_eq!(report.strip.messages_touched, 1);
    assert!(!outgoing.messages[0].extra.contains_key("reasoning_content"));
    // The message itself survives: it still carries text content.
    assert_eq!(outgoing.messages[0].content.as_str(), Some("answer"));
    assert_eq!(outgoing.messages.len(), 2);
}

#[test]
fn disabled_config_gates_the_stage_and_forwards_verbatim() {
    // The enabled gate lives at the router call site (router.rs gates the
    // per-attempt stage behind `reasoning_compat_cfg.enabled`): with
    // `enabled: false` the stage is never invoked and the request —
    // reasoning carriers included — is forwarded unmodified.
    let mut cfg = ReasoningCompatConfig::default();
    cfg.enabled = false;
    let request = req(vec![msg_with_thinking(true), msg_with_redacted()]);
    let source = model_ref(
        "anthropic",
        "claude-sonnet-4-5",
        ReasoningFamily::AnthropicManual,
    );
    let target = pm("anthropic", "claude-sonnet-4-7", None, None);

    // Carriers ARE present (the router's disabled branch still detects
    // them for its debug note) — the gate is what prevents mutation.
    assert!(!detect(&request.messages).is_empty());

    let mut outgoing = request.clone();
    if cfg.enabled {
        prepare_attempt(&mut outgoing, &request, Some(source), &target, &cfg);
    }

    assert_eq!(
        serde_json::to_value(&outgoing).unwrap(),
        serde_json::to_value(&request).unwrap()
    );
}

#[test]
fn empty_messages_request_is_a_verbatim_no_op() {
    let cfg = ReasoningCompatConfig::default();
    let request = req(Vec::new());
    let target = pm("anthropic", "claude-sonnet-4-7", None, None);
    let mut outgoing = request.clone();

    let report = prepare_attempt(&mut outgoing, &request, None, &target, &cfg);

    assert_eq!(report.decision, StripDecision::Preserve);
    assert_eq!(report.strip, StripReport::default());
    assert_eq!(report.normalized.emitted_shape, "none");
    assert_eq!(report.actions_json(), None);
    assert!(outgoing.messages.is_empty());
}

// ---------------------------------------------------------------------------
// 2. Detection → policy integration
// ---------------------------------------------------------------------------

#[test]
fn detection_to_policy_matrix_same_family_vs_cross_family() {
    let cfg = ReasoningCompatConfig::default();
    let cases = [
        (
            "signed thinking block",
            vec![msg_with_thinking(true)],
            ReasoningFamily::AnthropicManual,
            pm("anthropic", "claude-sonnet-4-5", None, None),
        ),
        (
            "redacted thinking block",
            vec![msg_with_redacted()],
            ReasoningFamily::AnthropicManual,
            pm("anthropic", "claude-opus-4-5", None, None),
        ),
        (
            "deepseek reasoning_content field",
            vec![msg_with_reasoning_content("pondering")],
            ReasoningFamily::DeepSeek,
            pm("deepseek", "deepseek-reasoner", None, None),
        ),
        (
            "openrouter reasoning field",
            vec![msg_with_reasoning_field("pondering")],
            ReasoningFamily::OpenRouter,
            pm("openrouter", "openrouter/auto", None, None),
        ),
    ];
    let cross_family_target = pm("openai", "gpt-5", None, None);

    for (label, msgs, expected_family, same_family_target) in cases {
        let footprint = detect(&msgs);
        assert_eq!(footprint.source_family, expected_family, "carrier: {label}");

        // Attribution unknown + target family matches the inferred source
        // family → preserve (the carriers may be native to the target).
        assert_eq!(
            decide(&footprint, None, &target_model_ref(&same_family_target), &cfg),
            StripDecision::Preserve,
            "same-family target: {label}"
        );

        // Attribution unknown + cross-family target → conservative strip.
        assert_eq!(
            decide(&footprint, None, &target_model_ref(&cross_family_target), &cfg),
            StripDecision::StripAttributionUnknown,
            "cross-family target: {label}"
        );
    }
}

#[test]
fn known_source_different_model_same_family_strips_model_bound_state() {
    let cfg = ReasoningCompatConfig::default();
    let msgs = vec![msg_with_thinking(true), msg_with_redacted()];
    let footprint = detect(&msgs);
    // Same manual family, different model: thinking signatures are
    // model-bound, so even an intra-family transition strips.
    let source = model_ref(
        "anthropic",
        "claude-sonnet-4-5",
        ReasoningFamily::AnthropicManual,
    );
    let target = pm("anthropic", "claude-opus-4-5", None, None);

    let decision = decide(&footprint, Some(&source), &target_model_ref(&target), &cfg);
    assert_eq!(decision, StripDecision::StripAll);

    let mut outgoing = req(msgs);
    let report = policy::apply(&mut outgoing, decision);
    assert_eq!(report.thinking_blocks, 1);
    assert_eq!(report.redacted_thinking_blocks, 1);
    assert!(outgoing.messages.iter().all(|message| {
        message
            .content
            .as_array()
            .map_or(true, |blocks| {
            blocks.iter().all(|block| {
                let block_type = block.get("type").and_then(Value::as_str);
                block_type != Some("thinking") && block_type != Some("redacted_thinking")
            })
        })
    }));
}

// ---------------------------------------------------------------------------
// 3. Policy → normalize integration
// ---------------------------------------------------------------------------

#[test]
fn cross_model_strip_then_manual_target_emits_budget_and_drops_sampling() {
    let cfg = ReasoningCompatConfig::default();
    let mut request = req(vec![msg_with_thinking(true)]);
    request.extra.insert("reasoning_effort".to_string(), json!("high"));
    request.temperature = Some(0.7);
    request.extra.insert("top_p".to_string(), json!(0.9));
    request.extra.insert("top_k".to_string(), json!(40));

    let source = model_ref(
        "anthropic",
        "claude-opus-4-5",
        ReasoningFamily::AnthropicManual,
    );
    let target = pm("anthropic", "claude-sonnet-4-5", None, None);
    let mut outgoing = request.clone();

    let report = prepare_attempt(&mut outgoing, &request, Some(source), &target, &cfg);

    // Strip stage: carriers removed from history.
    assert_eq!(report.decision, StripDecision::StripAll);
    assert_eq!(report.strip.thinking_blocks, 1);

    // Normalize stage: effort high → manual budget 16384 (default map),
    // foreign effort parameter replaced, sampling parameters dropped.
    assert_eq!(report.normalized.emitted_shape, "thinking_enabled");
    assert!(report.normalized.sampling_dropped);
    assert_eq!(
        outgoing.extra.get("thinking"),
        Some(&json!({"type": "enabled", "budget_tokens": 16384}))
    );
    assert!(!outgoing.extra.contains_key("reasoning_effort"));
    assert!(!outgoing.extra.contains_key("top_p"));
    assert!(!outgoing.extra.contains_key("top_k"));
    assert_eq!(outgoing.temperature, None);

    let actions = report.actions_json().expect("acted, summary due");
    assert!(actions.contains("\"normalized_shape\":\"thinking_enabled\""));
}

#[test]
fn cross_model_clamps_below_floor_manual_budget_to_minimum() {
    let cfg = ReasoningCompatConfig::default();
    let mut request = req(vec![msg_with_thinking(true)]);
    request.extra.insert(
        "thinking".to_string(),
        json!({"type": "enabled", "budget_tokens": 500}),
    );

    let source = model_ref(
        "anthropic",
        "claude-opus-4-5",
        ReasoningFamily::AnthropicManual,
    );
    let target = pm("anthropic", "claude-sonnet-4-5", None, None);
    let mut outgoing = request.clone();

    let report = prepare_attempt(&mut outgoing, &request, Some(source), &target, &cfg);

    assert_eq!(report.strip.thinking_blocks, 1);
    assert!(report.normalized.clamped);
    assert_eq!(
        outgoing.extra.get("thinking"),
        Some(&json!({"type": "enabled", "budget_tokens": 1024}))
    );
}

// ---------------------------------------------------------------------------
// 4. Normalize round-trips (effort ↔ budget matrix)
// ---------------------------------------------------------------------------

#[test]
fn reasoning_effort_round_trips_through_manual_budget_and_back() {
    let cfg = ReasoningCompatConfig::default();
    let manual_target = pm("anthropic", "claude-sonnet-4-5", None, None);
    let openai_target = pm("openai", "gpt-5", None, None);
    let matrix = [
        (Effort::Minimal, 1024),
        (Effort::Low, 2048),
        (Effort::Medium, 8192),
        (Effort::High, 16384),
        (Effort::XHigh, 32768),
    ];

    for (effort, budget) in matrix {
        // OpenAI client sends `reasoning_effort` → manual target receives
        // the mapped `budget_tokens`.
        let mut outgoing = req(Vec::new());
        outgoing
            .extra
            .insert("reasoning_effort".to_string(), json!(effort.to_string()));
        let spec = read_client_spec(&outgoing);
        let report = emit_for_target(&mut outgoing, spec, &manual_target, &cfg);
        assert_eq!(report.emitted_shape, "thinking_enabled", "effort {effort}");
        assert_eq!(
            outgoing.extra.get("thinking"),
            Some(&json!({"type": "enabled", "budget_tokens": budget})),
            "effort {effort}"
        );
        assert!(!outgoing.extra.contains_key("reasoning_effort"));

        // Read the manual budget back → emit for an OpenAI target → the
        // same effort level comes out (semantics preserved end to end).
        let spec = read_client_spec(&outgoing);
        let report = emit_for_target(&mut outgoing, spec, &openai_target, &cfg);
        assert_eq!(report.emitted_shape, "reasoning_effort", "budget {budget}");
        assert_eq!(
            outgoing.extra.get("reasoning_effort"),
            Some(&json!(effort.to_string())),
            "budget {budget}"
        );
        assert!(!outgoing.extra.contains_key("thinking"));
    }
}

// ---------------------------------------------------------------------------
// 5. Cost extraction → reasoning_cost chain
// ---------------------------------------------------------------------------

#[test]
fn usage_to_cost_chain_openai_shape_with_price_and_fallback() {
    let usage = usage_from_json(json!({
        "prompt_tokens": 100,
        "completion_tokens": 1000,
        "completion_tokens_details": { "reasoning_tokens": 500 }
    }));

    let extracted = extract_reasoning_usage(&usage);
    assert_eq!(extracted.carrier, ReasoningCarrier::OpenAIDetails);
    assert_eq!(extracted.reasoning_tokens, 500);

    // Dedicated reasoning price wins.
    let mut priced = pm("openai", "o3", None, None);
    priced.cost_per_million_reasoning_tokens = Some(12.0);
    assert!(
        (reasoning_cost(&priced, extracted.reasoning_tokens) - 12.0 * 500.0 / 1_000_000.0).abs()
            < 1e-12
    );

    // No dedicated price → output-price fallback (legacy billing).
    let unpriced = pm("openai", "o3", None, None);
    assert!(
        (reasoning_cost(&unpriced, extracted.reasoning_tokens) - 5.0 * 500.0 / 1_000_000.0).abs()
            < 1e-12
    );
}

#[test]
fn usage_to_cost_chain_anthropic_and_streaming_shapes() {
    // Anthropic shape: thinking tokens reported in output_tokens_details.
    let anthropic_usage = usage_from_json(json!({
        "completion_tokens": 1500,
        "output_tokens_details": { "thinking_tokens": 700 }
    }));
    let extracted = extract_reasoning_usage(&anthropic_usage);
    assert_eq!(extracted.carrier, ReasoningCarrier::AnthropicDetails);
    assert_eq!(extracted.reasoning_tokens, 700);
    let mut priced = pm("anthropic", "claude-sonnet-4-5", None, None);
    priced.cost_per_million_reasoning_tokens = Some(9.0);
    assert!(
        (reasoning_cost(&priced, extracted.reasoning_tokens) - 9.0 * 700.0 / 1_000_000.0).abs()
            < 1e-12
    );

    // Flattened streaming-relay shape: top-level reasoning_tokens.
    let streaming_usage = usage_from_json(json!({
        "completion_tokens": 400,
        "reasoning_tokens": 300
    }));
    let extracted = extract_reasoning_usage(&streaming_usage);
    assert_eq!(extracted.carrier, ReasoningCarrier::Streaming);
    assert_eq!(extracted.reasoning_tokens, 300);
    let unpriced = pm("openrouter", "openrouter/auto", None, None);
    assert!(
        (reasoning_cost(&unpriced, extracted.reasoning_tokens) - 5.0 * 300.0 / 1_000_000.0).abs()
            < 1e-12
    );
}

// ---------------------------------------------------------------------------
// 6. classify_family matrix via prepare_attempt (manual vs adaptive Claude)
// ---------------------------------------------------------------------------

#[test]
fn claude_manual_target_takes_thinking_budget_path() {
    let cfg = ReasoningCompatConfig::default();
    let mut request = req(vec![msg_with_thinking(true)]);
    request
        .extra
        .insert("reasoning_effort".to_string(), json!("high"));

    // claude-4-5 classifies manual; the signed carriers infer the same
    // family, so attribution-unknown preserves history and the client's
    // effort is re-emitted as a manual thinking budget.
    let target = pm("anthropic", "claude-sonnet-4-5", None, None);
    let mut outgoing = request.clone();

    let report = prepare_attempt(&mut outgoing, &request, None, &target, &cfg);

    assert_eq!(report.decision, StripDecision::Preserve);
    assert_eq!(report.strip, StripReport::default());
    assert_eq!(report.normalized.emitted_shape, "thinking_enabled");
    assert_eq!(
        outgoing.extra.get("thinking"),
        Some(&json!({"type": "enabled", "budget_tokens": 16384}))
    );
}

#[test]
fn claude_adaptive_target_never_receives_type_enabled() {
    let cfg = ReasoningCompatConfig::default();
    let mut request = req(vec![msg_with_thinking(true)]);
    request
        .extra
        .insert("reasoning_effort".to_string(), json!("high"));

    // claude-4-7 classifies adaptive: manual-era carriers strip
    // (attribution unknown × cross family) and the parameter is re-emitted
    // as adaptive — never `type: "enabled"`, which 400s on Claude 4.7+.
    let target = pm("anthropic", "claude-sonnet-4-7", None, None);
    let mut outgoing = request.clone();

    let report = prepare_attempt(&mut outgoing, &request, None, &target, &cfg);

    assert_eq!(report.decision, StripDecision::StripAttributionUnknown);
    assert_eq!(report.strip.thinking_blocks, 1);
    assert_eq!(report.normalized.emitted_shape, "thinking_adaptive");
    assert_eq!(
        outgoing.extra.get("thinking"),
        Some(&json!({"type": "adaptive"}))
    );
    assert_eq!(
        outgoing.extra.get("output_config"),
        Some(&json!({"effort": "high"}))
    );
    assert!(!outgoing.extra.contains_key("reasoning_effort"));
}
