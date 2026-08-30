//! Property tests for the reasoning-compatibility layer.
//!
//! Run with: `PROPTEST_CASES=64 cargo test -p ai-gateway --test reasoning_compat_property`
//!
//! Tests invariants:
//! 1. strip_removes_all_foreign_carriers - cross-model transitions strip all reasoning carriers
//! 2. preserve_same_model_verbatim - same-model continuations preserve messages bit-identically
//! 3. emitted_budgets_satisfy_anthropic_constraints - Anthropic manual budgets >= 1024 and < max_tokens
//! 4. adaptive_never_enabled - Anthropic adaptive targets never emit type:"enabled"
//! 5. no_injection_when_no_spec - empty spec adds no reasoning parameters
//! 6. cost_invariant_reasoning_lte_output_priced - reasoning cost never exceeds output-priced cost
//! 7. detect_footprint_matches_carriers - detect() correctly identifies carrier types and counts

use proptest::prelude::*;

use ai_gateway::config::ProviderModel;
use ai_gateway::models::openai::{Message, OpenAIRequest, Usage};
use ai_gateway::reasoning_compat::config::{
    Effort, ReasoningCompatConfig, ReasoningFamily,
};
use ai_gateway::reasoning_compat::cost::{extract_reasoning_usage, reasoning_cost};
use ai_gateway::reasoning_compat::detect::detect;
use ai_gateway::reasoning_compat::normalize::{emit_for_target, read_client_spec, ReasoningSpec};
use ai_gateway::reasoning_compat::policy::{apply, decide, ModelRef, StripDecision, StripReport};

use serde_json::{json, Map, Value};

// ---------------------------------------------------------------------------
// Carrier type enum for test metadata tracking
// ---------------------------------------------------------------------------

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum CarrierType {
 ThinkingWithSignature,
 ThinkingWithoutSignature,
 RedactedThinking,
 ReasoningContent,
 ReasoningField,
 ResponsesReasoning,
 #[allow(dead_code)]
 PlainText,
 #[allow(dead_code)]
 ToolCalls,
}

// ---------------------------------------------------------------------------
// Generators
// ---------------------------------------------------------------------------

prop_compose! {
    fn arb_carrier()
        (
        carrier_type in prop::sample::select(&[
            CarrierType::ThinkingWithSignature,
            CarrierType::ThinkingWithoutSignature,
            CarrierType::RedactedThinking,
            CarrierType::ReasoningContent,
            CarrierType::ReasoningField,
                CarrierType::ResponsesReasoning,
            ]),
            text_len in 10usize..200,
        )
        -> (CarrierType, Value, Option<(String, Value)>)
    {
        let text: String = (0..text_len).map(|_| 'x').collect();
        match carrier_type {
            CarrierType::ThinkingWithSignature => {
                (carrier_type, json!({"type": "thinking", "thinking": text.clone(), "signature": "sig_" }), None)
            }
            CarrierType::ThinkingWithoutSignature => {
                (carrier_type, json!({"type": "thinking", "thinking": text.clone() }), None)
            }
            CarrierType::RedactedThinking => {
                (carrier_type, json!({"type": "redacted_thinking", "data": "opaque_data" }), None)
            }
            CarrierType::ReasoningContent => {
                (carrier_type, json!({"type": "text", "text": text.clone() }), Some(("reasoning_content".to_string(), json!(text))))
            }
            CarrierType::ReasoningField => {
                (carrier_type, json!({"type": "text", "text": text.clone() }), Some(("reasoning".to_string(), json!(text))))
            }
            CarrierType::ResponsesReasoning => {
                (carrier_type, json!({"type": "reasoning", "text": text.clone() }), None)
            }
            CarrierType::PlainText | CarrierType::ToolCalls => unreachable!(),
        }
    }
}

prop_compose! {
    fn arb_plain_text_block()
        (text_len in 10usize..200)
        -> Value
    {
        let text: String = (0..text_len).map(|_| 'a').collect();
        json!({"type": "text", "text": text})
    }
}

prop_compose! {
    fn arb_tool_calls_extra()
        (call_id in 0u32..10)
        -> Map<String, Value>
    {
        let mut extra = Map::new();
        extra.insert(
            "tool_calls".to_string(),
            json!([{
                "id": format!("call_{}", call_id),
                "type": "function",
                "function": {"name": "test_fn", "arguments": "{}"}
            }]),
        );
        extra
    }
}

prop_compose! {
    fn arb_message_carriers()
        (
            carriers in prop::collection::vec(arb_carrier(), 0..3),
            has_plain in any::<bool>(),
            plain in arb_plain_text_block(),
        )
        -> (Vec<CarrierType>, Value, Map<String, Value>)
    {
        let mut carrier_types = Vec::new();
        let mut blocks = Vec::new();
        let mut extra = Map::new();

        for (ctype, block, extra_field) in carriers {
            carrier_types.push(ctype);
            blocks.push(block);
            if let Some((key, value)) = extra_field {
                extra.insert(key, value);
            }
        }

        if has_plain || blocks.is_empty() {
            blocks.push(plain);
        }

        (carrier_types, Value::Array(blocks), extra)
    }
}

prop_compose! {
    fn arb_conversation()
        (
            msg_count in 1usize..8usize,
            carriers_list in prop::collection::vec(arb_message_carriers(), 1..8),
            tool_extra in prop::option::of(arb_tool_calls_extra()),
        )
        -> (Vec<Vec<CarrierType>>, Vec<Message>)
    {
        let mut all_carrier_types = Vec::new();
        let mut messages = Vec::new();

        for i in 0..msg_count {
            let is_user = i % 2 == 0;
            if is_user {
                let mut content = Map::new();
                content.insert("text".to_string(), json!(format!("User message {}", i)));
                let text_block = json!({"type": "text", "text": format!("User message {}", i)});
                messages.push(Message {
                    role: "user".to_string(),
                    content: Value::Array(vec![text_block]),
                    extra: Map::new(),
                });
                all_carrier_types.push(Vec::new());
            } else {
                let idx = (i / 2).min(carriers_list.len().saturating_sub(1));
                let (carrier_types, content, extra) = if idx < carriers_list.len() {
                    carriers_list[idx].clone()
                } else {
                    (Vec::new(), json!({"type": "text", "text": "assistant fallback"}), Map::new())
                };
                all_carrier_types.push(carrier_types);
                messages.push(Message {
                    role: "assistant".to_string(),
                    content,
                    extra,
                });
            }
        }

        if !msg_count % 2 == 0 && tool_extra.is_some() {
            if let Some(last_msg) = messages.last_mut() {
                if last_msg.role == "assistant" {
                    last_msg.extra.extend(tool_extra.clone().unwrap_or_default());
                }
            }
        }

        (all_carrier_types, messages)
    }
}

prop_compose! {
    fn arb_model_id()
        (idx in 0usize..10)
        -> &'static str
    {
        [
            "claude-4-5-sonnet",
            "claude-4-7-sonnet",
            "claude-opus-4-1",
            "deepseek-chat",
            "deepseek-reasoner",
            "gpt-4o",
            "o3-mini",
            "grok-3",
            "gemini-2.0-flash",
            "some-plain-model",
        ][idx]
    }
}

prop_compose! {
    fn arb_effort()
        (idx in 0usize..5)
        -> Effort
    {
        [Effort::Minimal, Effort::Low, Effort::Medium, Effort::High, Effort::XHigh][idx]
    }
}

prop_compose! {
    fn arb_usage()
        (
            prompt_tokens in 0u32..10000u32,
            completion_tokens in 0u32..10000u32,
            has_reasoning_tokens in any::<bool>(),
            has_thinking_tokens in any::<bool>(),
            reasoning_tokens in 0u32..5000u32,
            thinking_tokens in 0u32..5000u32,
            has_top_level in any::<bool>(),
        )
        -> Usage
    {
        let mut extra = Map::new();
        extra.insert("prompt_tokens".to_string(), json!(prompt_tokens));
        extra.insert("completion_tokens".to_string(), json!(completion_tokens));

        if has_reasoning_tokens {
            let mut details = Map::new();
            details.insert("reasoning_tokens".to_string(), json!(reasoning_tokens));
            extra.insert("completion_tokens_details".to_string(), Value::Object(details));
        }

        if has_thinking_tokens {
            let mut details = Map::new();
            details.insert("thinking_tokens".to_string(), json!(thinking_tokens));
            extra.insert("output_tokens_details".to_string(), Value::Object(details));
        }

        if has_top_level && !has_reasoning_tokens && !has_thinking_tokens {
            extra.insert("reasoning_tokens".to_string(), json!(reasoning_tokens));
        }

        Usage {
            prompt_tokens,
            completion_tokens,
            total_tokens: prompt_tokens.saturating_add(completion_tokens),
            extra,
        }
    }
}

prop_compose! {
    fn arb_source_target_pair()
        (
            source_idx in 0usize..10,
            target_idx in 0usize..10,
            same in any::<bool>(),
        )
        -> (String, String, ReasoningFamily, ReasoningFamily)
    {
        let models = [
            ("claude-4-5-sonnet", ReasoningFamily::AnthropicManual),
            ("claude-4-7-sonnet", ReasoningFamily::AnthropicAdaptive),
            ("claude-opus-4-1", ReasoningFamily::AnthropicManual),
            ("deepseek-chat", ReasoningFamily::DeepSeek),
            ("deepseek-reasoner", ReasoningFamily::DeepSeek),
            ("gpt-4o", ReasoningFamily::None),
            ("o3-mini", ReasoningFamily::OpenAIReasoning),
            ("grok-3", ReasoningFamily::XAI),
            ("gemini-2.0-flash", ReasoningFamily::Gemini),
            ("some-plain-model", ReasoningFamily::None),
        ];

        if same {
            let (model, family) = models[source_idx % models.len()];
            (model.to_string(), model.to_string(), family, family)
        } else {
            let (s_model, s_family) = models[source_idx % models.len()];
            let t_idx = if target_idx % models.len() == source_idx % models.len() {
                (target_idx + 1) % models.len()
            } else {
                target_idx % models.len()
            };
            let (t_model, t_family) = models[t_idx];
            (s_model.to_string(), t_model.to_string(), s_family, t_family)
        }
    }
}

// ---------------------------------------------------------------------------
// Helper functions
// ---------------------------------------------------------------------------

fn make_provider_model(model: &str, family: ReasoningFamily) -> ProviderModel {
    ProviderModel {
        provider: "test-provider".to_string(),
        model: model.to_string(),
        cost_per_million_input_tokens: 1.0,
        cost_per_million_output_tokens: 2.0,
        priority: 100,
        structured_output_passthrough: None,
        tier: None,
        context_window: 128000,
        specializations: Vec::new(),
        cost_per_million_cache_read_input_tokens: None,
        cost_per_million_cache_creation_input_tokens: None,
        cache_min_tokens: None,
        cache_support: None,
        cost_per_million_reasoning_tokens: None,
        reasoning_family: Some(family),
        reasoning_parameter: None,
    }
}

fn make_provider_model_with_reasoning_price(
    model: &str,
    family: ReasoningFamily,
    output_price: f64,
    reasoning_price: Option<f64>,
) -> ProviderModel {
    ProviderModel {
        provider: "test-provider".to_string(),
        model: model.to_string(),
        cost_per_million_input_tokens: 1.0,
        cost_per_million_output_tokens: output_price,
        priority: 100,
        structured_output_passthrough: None,
        tier: None,
        context_window: 128000,
        specializations: Vec::new(),
        cost_per_million_cache_read_input_tokens: None,
        cost_per_million_cache_creation_input_tokens: None,
        cache_min_tokens: None,
        cache_support: None,
        cost_per_million_reasoning_tokens: reasoning_price,
        reasoning_family: Some(family),
        reasoning_parameter: None,
    }
}

fn model_ref(model: &str, family: ReasoningFamily) -> ModelRef {
    ModelRef {
        provider: "test-provider".to_string(),
        model: model.to_string(),
        family,
    }
}

fn count_carriers_in_json(messages: &[Message]) -> (usize, usize, usize, usize, usize) {
    let mut thinking_blocks = 0usize;
    let mut redacted_blocks = 0usize;
    let mut reasoning_content_count = 0usize;
    let mut reasoning_field_count = 0usize;
    let mut responses_reasoning = 0usize;

    for msg in messages {
        if let Value::Array(blocks) = &msg.content {
            for block in blocks {
                if let Some(block_type) = block.get("type").and_then(Value::as_str) {
                    match block_type {
                        "thinking" => thinking_blocks += 1,
                        "redacted_thinking" => redacted_blocks += 1,
                        "reasoning" => responses_reasoning += 1,
                        _ => {}
                    }
                }
            }
        }
        if msg.extra.contains_key("reasoning_content") {
            reasoning_content_count += 1;
        }
        if msg.extra.contains_key("reasoning") {
            reasoning_field_count += 1;
        }
    }

    (thinking_blocks, redacted_blocks, reasoning_content_count, reasoning_field_count, responses_reasoning)
}

// ---------------------------------------------------------------------------
// Property tests
// ---------------------------------------------------------------------------

proptest! {
    #![proptest_config(ProptestConfig::with_cases(64))]

    /// Property 1: strip_removes_all_foreign_carriers
    /// For any conversation + any (source, target) with source != target:
    /// apply(StripAll) removes all thinking/redacted_thinking blocks and reasoning fields.
    #[test]
    fn prop_strip_removes_all_foreign_carriers(
        (_carrier_types, messages) in arb_conversation(),
        (source_model, target_model, source_family, target_family) in arb_source_target_pair(),
    ) {
        // Skip when source == target (verbatim preservation case, covered separately)
        if source_model == target_model && source_family == target_family {
            return Ok(());
        }

        let mut outgoing = OpenAIRequest {
            model: target_model.clone(),
            messages: messages.clone(),
            stream: false,
            temperature: None,
            max_tokens: None,
            extra: Map::new(),
        };

        let cfg = ReasoningCompatConfig::default();
        let source_ref = model_ref(&source_model, source_family);
        let target_ref = model_ref(&target_model, target_family);

        let footprint = detect(&messages);
        let decision = decide(&footprint, Some(&source_ref), &target_ref, &cfg);
        let _report = apply(&mut outgoing, decision);

        let (thinking, redacted, reasoning_content, reasoning_field, responses_reasoning) =
            count_carriers_in_json(&outgoing.messages);

        prop_assert_eq!(thinking, 0, "thinking blocks must all be removed after strip");
        prop_assert_eq!(redacted, 0, "redacted_thinking blocks must all be removed after strip");
        prop_assert_eq!(reasoning_content, 0, "reasoning_content fields must all be removed after strip");
        prop_assert_eq!(reasoning_field, 0, "reasoning fields must all be removed after strip");
        prop_assert_eq!(responses_reasoning, 0, "responses reasoning blocks must all be removed after strip");
    }

    /// Property 2: preserve_same_model_verbatim
    /// For same provider + model + family, messages are preserved bit-identically.
    #[test]
    fn prop_preserve_same_model_verbatim(
        (_carrier_types, messages) in arb_conversation(),
        model in arb_model_id(),
    ) {
        let family = ai_gateway::reasoning_compat::detect::classify_family(model);

        let mut outgoing = OpenAIRequest {
            model: model.to_string(),
            messages: messages.clone(),
            stream: false,
            temperature: None,
            max_tokens: None,
            extra: Map::new(),
        };

        let cfg = ReasoningCompatConfig::default();
        let source_ref = model_ref(model, family);
        let target_ref = model_ref(model, family);

        let footprint = detect(&messages);
        let decision = decide(&footprint, Some(&source_ref), &target_ref, &cfg);
        let _report = apply(&mut outgoing, decision);

        let original_json = serde_json::to_string(&messages).expect("messages serialize");
        let result_json = serde_json::to_string(&outgoing.messages).expect("result serialize");

        prop_assert_eq!(original_json, result_json, "same-model continuation must preserve messages verbatim");
    }

    /// Property 3: emitted_budgets_satisfy_anthropic_constraints
    /// For Anthropic manual targets: budget_tokens >= 1024 AND < resolved max_tokens.
    #[test]
    fn prop_emitted_budgets_satisfy_anthropic_constraints(
        effort in arb_effort(),
        max_tokens in 1024u32..100000u32,
    ) {
        let mut outgoing = OpenAIRequest {
            model: "claude-4-5-sonnet".to_string(),
            messages: vec![],
            stream: false,
            temperature: None,
            max_tokens: Some(max_tokens),
            extra: Map::new(),
        };

        outgoing.extra.insert("reasoning_effort".to_string(), json!(effort.to_string()));

        let spec = read_client_spec(&outgoing);
        let target = make_provider_model("claude-4-5-sonnet", ReasoningFamily::AnthropicManual);
        let cfg = ReasoningCompatConfig::default();

        let _report = emit_for_target(&mut outgoing, spec, &target, &cfg);

        if let Some(thinking) = outgoing.extra.get("thinking").and_then(Value::as_object) {
            if thinking.get("type").and_then(Value::as_str) == Some("enabled") {
                if let Some(budget) = thinking.get("budget_tokens").and_then(Value::as_u64) {
                    let budget = budget as u32;
                    prop_assert!(budget >= 1024, "budget_tokens must be >= 1024 floor, got {}", budget);

                    let resolved_max = outgoing.max_tokens
                        .or_else(|| outgoing.extra.get("max_tokens").and_then(Value::as_u64).map(|n| n as u32))
                        .unwrap_or(max_tokens);

                    prop_assert!(budget < resolved_max, "budget_tokens {} must be < resolved max_tokens {}", budget, resolved_max);
                }
            }
        }
    }

    /// Property 4: adaptive_never_enabled
    /// Anthropic adaptive targets never emit thinking.type == "enabled".
    #[test]
    fn prop_adaptive_never_enabled(
        effort in arb_effort(),
    ) {
        let mut outgoing = OpenAIRequest {
            model: "claude-4-7-sonnet".to_string(),
            messages: vec![],
            stream: false,
            temperature: None,
            max_tokens: None,
            extra: Map::new(),
        };

        outgoing.extra.insert("reasoning_effort".to_string(), json!(effort.to_string()));

        let spec = read_client_spec(&outgoing);
        let target = make_provider_model("claude-4-7-sonnet", ReasoningFamily::AnthropicAdaptive);
        let cfg = ReasoningCompatConfig::default();

        let _report = emit_for_target(&mut outgoing, spec, &target, &cfg);

        if let Some(thinking) = outgoing.extra.get("thinking").and_then(Value::as_object) {
            let thinking_type = thinking.get("type").and_then(Value::as_str);
            prop_assert_ne!(thinking_type, Some("enabled"), "adaptive targets must never emit type:enabled, got {:?}", thinking_type);
            prop_assert_eq!(thinking_type, Some("adaptive"), "adaptive targets should emit type:adaptive");
        }
    }

    /// Property 5: no_injection_when_no_spec
    /// Empty spec results in no reasoning parameters added.
    #[test]
    fn prop_no_injection_when_no_spec(
        target_family in prop::sample::select(&[
            ReasoningFamily::AnthropicManual,
            ReasoningFamily::AnthropicAdaptive,
            ReasoningFamily::OpenAIReasoning,
            ReasoningFamily::OpenRouter,
            ReasoningFamily::DeepSeek,
            ReasoningFamily::Gemini,
            ReasoningFamily::XAI,
            ReasoningFamily::None,
        ]),
    ) {
        let mut outgoing = OpenAIRequest {
            model: "test-model".to_string(),
            messages: vec![],
            stream: false,
            temperature: None,
            max_tokens: None,
            extra: Map::new(),
        };

        let spec = ReasoningSpec {
            effort: None,
            budget: None,
            adaptive: false,
        };

        let target = make_provider_model("test-model", target_family);
        let cfg = ReasoningCompatConfig::default();

        let report = emit_for_target(&mut outgoing, spec, &target, &cfg);

        prop_assert_eq!(report.emitted_shape, "none", "empty spec should emit nothing");
        prop_assert!(!outgoing.extra.contains_key("thinking"), "no thinking should be added for empty spec");
        prop_assert!(!outgoing.extra.contains_key("reasoning_effort"), "no reasoning_effort should be added for empty spec");
        prop_assert!(!outgoing.extra.contains_key("reasoning"), "no reasoning should be added for empty spec");
        prop_assert!(!outgoing.extra.contains_key("output_config"), "no output_config should be added for empty spec");
    }

    /// Property 6: cost_invariant_reasoning_lte_output_priced
    /// When reasoning_price <= output_price: reasoning_cost <= tokens * output_price / 1e6.
    #[test]
    fn prop_cost_invariant_reasoning_lte_output_priced(
        usage in arb_usage(),
        output_price in 1.0..50.0f64,
        reasoning_ratio in 0.0..=1.0f64,
    ) {
        let reasoning_price = output_price * reasoning_ratio;
        let model = make_provider_model_with_reasoning_price(
            "test-model",
            ReasoningFamily::AnthropicManual,
            output_price,
            Some(reasoning_price),
        );

        let extracted = extract_reasoning_usage(&usage);
        if extracted.reasoning_tokens == 0 {
            return Ok(());
        }

        let reasoning_cost_val = reasoning_cost(&model, extracted.reasoning_tokens);
        let output_cost_bound = (extracted.reasoning_tokens as f64) * output_price / 1_000_000.0;

        prop_assert!(
            reasoning_cost_val <= output_cost_bound + 1e-9,
            "reasoning_cost {} must be <= output-priced bound {}",
            reasoning_cost_val,
            output_cost_bound
        );
    }

    /// Property 7: detect_footprint_matches_carriers
    /// detect() correctly identifies carrier types and counts.
    #[test]
    fn prop_detect_footprint_matches_carriers(
        (_carrier_types, messages) in arb_conversation(),
    ) {
        let footprint = detect(&messages);

        let (thinking, redacted, reasoning_content, reasoning_field, responses_reasoning) =
            count_carriers_in_json(&messages);

        prop_assert_eq!(
            footprint.has_thinking_blocks,
            thinking > 0,
            "has_thinking_blocks flag must match actual thinking blocks"
        );
        prop_assert_eq!(
            footprint.has_redacted_thinking,
            redacted > 0,
            "has_redacted_thinking flag must match actual redacted blocks"
        );
        prop_assert_eq!(
            footprint.reasoning_content_msgs.len(),
            reasoning_content,
            "reasoning_content_msgs count must match actual reasoning_content fields"
        );
        prop_assert_eq!(
            footprint.reasoning_field_msgs.len(),
            reasoning_field,
            "reasoning_field_msgs count must match actual reasoning fields"
        );
        prop_assert_eq!(
            footprint.responses_items.len(),
            responses_reasoning,
            "responses_items count must match actual reasoning blocks"
        );

        let total_blocks = thinking + redacted + responses_reasoning;
        prop_assert_eq!(
            footprint.block_counts,
            total_blocks,
            "block_counts must match total reasoning content blocks"
        );
    }
}

// ---------------------------------------------------------------------------
// Additional unit-style tests for edge cases
// ---------------------------------------------------------------------------

#[test]
fn strip_all_removes_all_carrier_types() {
    let messages = vec![
        Message {
            role: "user".to_string(),
            content: json!("Hello"),
            extra: Map::new(),
        },
        Message {
            role: "assistant".to_string(),
            content: json!([
                {"type": "thinking", "thinking": "deep thought", "signature": "sig"},
                {"type": "redacted_thinking", "data": "opaque"},
                {"type": "reasoning", "text": "step by step"},
                {"type": "text", "text": "answer"},
            ]),
            extra: {
                let mut m = Map::new();
                m.insert("reasoning_content".to_string(), json!("extra reasoning"));
                m.insert("reasoning".to_string(), json!("field reasoning"));
                m
            },
        },
    ];

    let mut outgoing = OpenAIRequest {
        model: "gpt-4o".to_string(),
        messages: messages.clone(),
        stream: false,
        temperature: None,
        max_tokens: None,
        extra: Map::new(),
    };

    let report = apply(&mut outgoing, StripDecision::StripAll);

    assert_eq!(report.thinking_blocks, 1);
    assert_eq!(report.redacted_thinking_blocks, 1);
    assert_eq!(report.fields_removed, 3); // reasoning block + reasoning_content + reasoning extras
    assert!(!outgoing.messages[1].extra.contains_key("reasoning_content"));
    assert!(!outgoing.messages[1].extra.contains_key("reasoning"));

    let (t, r, rc, rf, rr) = count_carriers_in_json(&outgoing.messages);
    assert_eq!((t, r, rc, rf, rr), (0, 0, 0, 0, 0));
}

#[test]
fn preserve_returns_empty_report() {
    let messages = vec![Message {
        role: "assistant".to_string(),
        content: json!([
            {"type": "thinking", "thinking": "thought", "signature": "s"},
            {"type": "text", "text": "answer"},
        ]),
        extra: Map::new(),
    }];

    let mut outgoing = OpenAIRequest {
        model: "claude-4-5-sonnet".to_string(),
        messages: messages.clone(),
        stream: false,
        temperature: None,
        max_tokens: None,
        extra: Map::new(),
    };

    let report = apply(&mut outgoing, StripDecision::Preserve);

    assert_eq!(report, StripReport::default());
    assert_eq!(outgoing.messages.len(), 1);
}

#[test]
fn anthropic_manual_budget_clamping() {
    let mut outgoing = OpenAIRequest {
        model: "claude-4-5-sonnet".to_string(),
        messages: vec![],
        stream: false,
        temperature: None,
        max_tokens: Some(5000),
        extra: Map::new(),
    };

    outgoing.extra.insert("thinking".to_string(), json!({"type": "enabled", "budget_tokens": 500}));

    let spec = read_client_spec(&outgoing);
    let target = make_provider_model("claude-4-5-sonnet", ReasoningFamily::AnthropicManual);
    let cfg = ReasoningCompatConfig::default();

    let report = emit_for_target(&mut outgoing, spec, &target, &cfg);

    assert!(report.clamped, "budget below 1024 should be clamped");

    if let Some(thinking) = outgoing.extra.get("thinking").and_then(Value::as_object) {
        let budget = thinking.get("budget_tokens").and_then(Value::as_u64).unwrap();
        assert!(budget >= 1024, "clamped budget must be >= 1024");
    }
}

#[test]
fn adaptive_target_with_budget_spec_emits_adaptive() {
    let mut outgoing = OpenAIRequest {
        model: "claude-4-7-sonnet".to_string(),
        messages: vec![],
        stream: false,
        temperature: None,
        max_tokens: None,
        extra: Map::new(),
    };

    outgoing.extra.insert("reasoning".to_string(), json!({"max_tokens": 8192}));

    let spec = read_client_spec(&outgoing);
    let target = make_provider_model("claude-4-7-sonnet", ReasoningFamily::AnthropicAdaptive);
    let cfg = ReasoningCompatConfig::default();

    let report = emit_for_target(&mut outgoing, spec, &target, &cfg);

    assert_eq!(report.emitted_shape, "thinking_adaptive");

    let thinking = outgoing.extra.get("thinking").and_then(Value::as_object).unwrap();
    assert_eq!(thinking.get("type").and_then(Value::as_str), Some("adaptive"));
}

#[test]
fn cost_uses_output_price_when_reasoning_unset() {
    let model = ProviderModel {
        provider: "test".to_string(),
        model: "test-model".to_string(),
        cost_per_million_input_tokens: 1.0,
        cost_per_million_output_tokens: 10.0,
        cost_per_million_reasoning_tokens: None,
        priority: 100,
        structured_output_passthrough: None,
        tier: None,
        context_window: 0,
        specializations: vec![],
        cost_per_million_cache_read_input_tokens: None,
        cost_per_million_cache_creation_input_tokens: None,
        cache_min_tokens: None,
        cache_support: None,
        reasoning_family: None,
        reasoning_parameter: None,
    };

    let cost = reasoning_cost(&model, 1_000_000);
    assert!((cost - 10.0).abs() < 1e-9, "cost should fall back to output price");
}
