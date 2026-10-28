//! Property tests for the Anthropic Messages front-door translation layer.
//!
//! Run with: `cargo test -p ai-gateway --test anthropic_translation_property`
//! (respects `PROPTEST_CASES`; the pinned case count matches the sibling
//! property suites).
//!
//! Tests three invariants (spec requirements 1.2 and 2.2):
//! a) Tool-call argument fragments streamed through [`MessageStreamAssembler`]
//!    produce `input_json_delta` events whose concatenated `partial_json`
//!    equals the original tool arguments JSON (LiteLLM dropped-fragment bug
//!    regression guard), with spec-conformant event order and indices.
//! b) Every generated chat `finish_reason` (known, unknown, missing) maps to
//!    a valid Anthropic `stop_reason` serde variant that never fails to
//!    serialize, and `pause_turn` is never synthesized.
//! c) The request translator preserves `tool_use` ↔ `tool_result` id pairing
//!    (including intentionally mismatched ids), orders tool messages directly
//!    after the assistant `tool_calls` message, and satisfies the
//!    message-count invariant implemented by `translate_request.rs`.

use proptest::prelude::*;
use serde_json::{json, Map, Value};

use ai_gateway::anthropic::{
    synthesize_anthropic_response, translate, AnthropicMessage, AnthropicRequest,
    AnthropicSseEvent, AnthropicSynthesisContext, ChatDelta, ContentBlock, ContentBlockDelta,
    MessageContent, MessageStreamAssembler, StopReason, SystemPrompt, ToolResultContent, Usage,
};
use ai_gateway::models::openai::OpenAIResponse;

/// Every serde wire variant of [`StopReason`] (snake_case serialization).
const VALID_STOP_REASONS: &[&str] = &[
    "end_turn",
    "max_tokens",
    "stop_sequence",
    "tool_use",
    "pause_turn",
    "refusal",
];

// ---------------------------------------------------------------------------
// Generators
// ---------------------------------------------------------------------------

/// Bounded JSON value generator (depth 0 = leaves only).
fn arb_json_value(depth: u32) -> BoxedStrategy<Value> {
    let leaf = prop_oneof![
        4 => proptest::string::string_regex("[a-zA-Z0-9 ,.!?-]{0,12}")
            .expect("leaf string regex must compile")
            .prop_map(Value::String),
        2 => any::<i64>().prop_map(Value::from),
        1 => any::<bool>().prop_map(Value::Bool),
    ];
    if depth == 0 {
        return leaf.boxed();
    }
    prop_oneof![
        3 => leaf,
        2 => arb_json_object(depth - 1).prop_map(Value::Object),
        1 => prop::collection::vec(arb_json_value(depth - 1), 0..3).prop_map(Value::Array),
    ]
    .boxed()
}

/// Bounded JSON object generator (0..=4 entries, depth-bounded values).
fn arb_json_object(depth: u32) -> BoxedStrategy<Map<String, Value>> {
    prop::collection::hash_map(
        proptest::string::string_regex("[a-z_]{1,8}").expect("key regex must compile"),
        arb_json_value(depth),
        0..4,
    )
    .prop_map(|entries| entries.into_iter().collect::<Map<String, Value>>())
    .boxed()
}

prop_compose! {
    fn arb_tool_name()
    (name in proptest::string::string_regex("[a-z_]{1,16}").expect("tool name regex must compile"))
    -> String { name }
}

prop_compose! {
    fn arb_tool_id()
    (n in 0u32..10_000)
    -> String { format!("toolu_{n}") }
}

prop_compose! {
    fn arb_call_spec()
    (
        id in proptest::string::string_regex("toolu_A[0-9]{1,4}").expect("id regex must compile"),
        foreign in proptest::string::string_regex("toolu_B[0-9]{1,4}").expect("foreign regex must compile"),
        result in proptest::string::string_regex("[a-z0-9 ]{1,24}").expect("result regex must compile"),
        mismatch in any::<bool>(),
    )
    -> CallSpec { CallSpec { id, foreign, result, mismatch } }
}

/// One assistant tool call plus the user's tool_result for it.
#[derive(Debug, Clone)]
struct CallSpec {
    id: String,
    foreign: String,
    result: String,
    mismatch: bool,
}

impl CallSpec {
    /// The `tool_use_id` the generated user tool_result carries.
    fn effective_id(&self) -> &str {
        if self.mismatch {
            &self.foreign
        } else {
            &self.id
        }
    }
}

/// One tool exchange: assistant `tool_use` message + user `tool_result` reply.
#[derive(Debug, Clone)]
struct GroupSpec {
    preamble: String,
    calls: Vec<CallSpec>,
    followup: Option<String>,
}

prop_compose! {
    fn arb_group()
    (
        preamble in proptest::string::string_regex("[a-z ]{1,20}").expect("preamble regex must compile"),
        calls in prop::collection::vec(arb_call_spec(), 1..=3),
        followup in prop::option::of(
            proptest::string::string_regex("[a-z ]{1,20}").expect("followup regex must compile")
        ),
    )
    -> GroupSpec { GroupSpec { preamble, calls, followup } }
}

prop_compose! {
    fn arb_optional_text()
    (text in prop::option::of(
        proptest::string::string_regex("[a-z0-9 ]{1,32}").expect("text regex must compile")
    ))
    -> Option<String> { text }
}

// ---------------------------------------------------------------------------
// Stream-order state machine (shared checker for property a)
// ---------------------------------------------------------------------------

fn block_kind(content_block: &ContentBlock) -> &'static str {
    match content_block {
        ContentBlock::Text { .. } => "text",
        ContentBlock::ToolUse { .. } => "tool_use",
        ContentBlock::Thinking { .. } => "thinking",
        _ => "other",
    }
}

fn delta_kind(delta: &ContentBlockDelta) -> &'static str {
    match delta {
        ContentBlockDelta::TextDelta { .. } => "text",
        ContentBlockDelta::InputJsonDelta { .. } => "tool_use",
        ContentBlockDelta::ThinkingDelta { .. } | ContentBlockDelta::SignatureDelta { .. } => {
            "thinking"
        }
    }
}

/// Assert the documented event order:
/// `message_start` (first) → per block `content_block_start` → deltas →
/// `content_block_stop` → … → `message_delta` → `message_stop`, with
/// sequential block indices, at most one open block at a time, and every
/// delta's kind matching its open block's kind.
fn check_stream_order(events: &[AnthropicSseEvent]) -> Result<(), String> {
    let mut saw_message_start = false;
    let mut saw_message_delta = false;
    let mut saw_message_stop = false;
    let mut open: Option<usize> = None;
    let mut kinds: Vec<&'static str> = Vec::new();

    for (pos, event) in events.iter().enumerate() {
        match event {
            AnthropicSseEvent::MessageStart { .. } => {
                if pos != 0 {
                    return Err("message_start must be the first event".to_string());
                }
                if saw_message_start {
                    return Err("duplicate message_start".to_string());
                }
                saw_message_start = true;
            }
            AnthropicSseEvent::ContentBlockStart {
                index, content_block, ..
            } => {
                if saw_message_delta || saw_message_stop {
                    return Err(format!("content_block_start at pos {pos} after terminal events"));
                }
                if open.is_some() {
                    return Err(format!(
                        "content_block_start at pos {pos} while block {open:?} is still open"
                    ));
                }
                if *index != kinds.len() {
                    return Err(format!(
                        "non-sequential block index {index} at pos {pos} (expected {})",
                        kinds.len()
                    ));
                }
                kinds.push(block_kind(content_block));
                open = Some(*index);
            }
            AnthropicSseEvent::ContentBlockDelta { index, delta, .. } => {
                if open != Some(*index) {
                    return Err(format!(
                        "delta at pos {pos} for block {index} while open block is {open:?}"
                    ));
                }
                let expected = kinds[*index];
                let actual = delta_kind(delta);
                if expected != actual {
                    return Err(format!(
                        "delta kind '{actual}' does not match open block kind '{expected}' at pos {pos}"
                    ));
                }
            }
            AnthropicSseEvent::ContentBlockStop { index, .. } => {
                if open != Some(*index) {
                    return Err(format!(
                        "content_block_stop at pos {pos} for block {index} but open block is {open:?}"
                    ));
                }
                open = None;
            }
            AnthropicSseEvent::MessageDelta { .. } => {
                if open.is_some() {
                    return Err("message_delta with a block still open".to_string());
                }
                if saw_message_delta || saw_message_stop {
                    return Err("duplicate or late message_delta".to_string());
                }
                saw_message_delta = true;
            }
            AnthropicSseEvent::MessageStop { .. } => {
                if open.is_some() {
                    return Err("message_stop with a block still open".to_string());
                }
                if !saw_message_delta || saw_message_stop {
                    return Err("message_stop without a preceding message_delta".to_string());
                }
                saw_message_stop = true;
            }
            AnthropicSseEvent::Ping { .. } => {}
            AnthropicSseEvent::Error { .. } => {
                return Err("unexpected error event in a successful stream".to_string());
            }
        }
    }

    if !saw_message_start {
        return Err("message_start never emitted".to_string());
    }
    if !saw_message_stop {
        return Err("message_stop never emitted".to_string());
    }
    Ok(())
}

// ---------------------------------------------------------------------------
// Property (a): tool-call fragment assembly (Requirement 2.2)
// ---------------------------------------------------------------------------

proptest! {
    #![proptest_config(ProptestConfig::with_cases(64))]

    /// Arbitrary tool calls, split into 2..=10 argument fragments with
    /// optional thinking/text interleaves, must reassemble exactly:
    /// concatenated `input_json_delta.partial_json` parses back to the
    /// original arguments object, the accessor agrees, event order and
    /// indices conform, and reopened tool blocks keep their id/name identity.
    #[test]
    fn prop_tool_fragment_assembly_preserves_arguments(
        name in arb_tool_name(),
        id in arb_tool_id(),
        args in arb_json_object(2),
        cuts in prop::collection::vec(1u8..=99, 1..10),
        lead_thinking in arb_optional_text(),
        mid_text in arb_optional_text(),
        mid_thinking in arb_optional_text(),
    ) {
        let args_str = Value::Object(args.clone()).to_string();
        let len = args_str.len();

        // Structural fragment-count invariant: `cuts` draws 1..=9 values in
        // 1..=99, so after sort+dedup there are 1..=9 distinct interior
        // positions (each maps to (pct*len)/100 < len), yielding exactly
        // 2..=10 fragments — no post-hoc range assertion needed.
        //
        // ASCII-only precondition for the byte slicing below: every
        // generator feeding `args` ([a-z_] keys, [a-zA-Z0-9 ,.!?-] string
        // leaves, integers, booleans) serializes to pure-ASCII JSON, so
        // `args_str[prev..pos]` can never split a multi-byte char.
        let mut positions: Vec<usize> = cuts
            .iter()
            .map(|pct| (*pct as usize * len) / 100)
            .collect();
        positions.sort_unstable();
        positions.dedup();
        let mut fragments: Vec<&str> = Vec::new();
        let mut prev = 0usize;
        for pos in positions {
            fragments.push(&args_str[prev..pos]);
            prev = pos;
        }
        fragments.push(&args_str[prev..]);

        // Build the delta sequence: optional leading thinking, the tool
        // fragments (id/name on the first), optional interleaves that force
        // the tool block to close and reopen.
        let mut deltas: Vec<ChatDelta> = Vec::new();
        if let Some(thinking) = &lead_thinking {
            deltas.push(ChatDelta::Thinking { thinking: thinking.clone(), signature: None });
        }
        for (i, fragment) in fragments.iter().enumerate() {
            let (tid, tname) = if i == 0 {
                (Some(id.clone()), Some(name.clone()))
            } else {
                (None, None)
            };
            deltas.push(ChatDelta::ToolCall {
                index: 0,
                id: tid,
                name: tname,
                arguments: fragment.to_string(),
            });
            if i == 0 && fragments.len() > 1 {
                if let Some(text) = &mid_text {
                    deltas.push(ChatDelta::Text { text: text.clone() });
                }
            }
            if i == 1 && fragments.len() > 2 {
                if let Some(thinking) = &mid_thinking {
                    deltas.push(ChatDelta::Thinking { thinking: thinking.clone(), signature: None });
                }
            }
        }

        let mut assembler = MessageStreamAssembler::new("claude-group", None, Usage::default());
        let mut events: Vec<AnthropicSseEvent> = Vec::new();
        for delta in deltas {
            events.extend(assembler.feed(delta));
        }
        let accumulated = assembler.tool_arguments(0).to_string();
        let unused_accumulation = assembler.tool_arguments(7).to_string();
        events.extend(
            assembler.finish(
                StopReason::ToolUse,
                None,
                Usage { input_tokens: 3, output_tokens: 5, ..Default::default() },
            ),
        );

        // Event-order + index invariants.
        if let Err(msg) = check_stream_order(&events) {
            prop_assert!(false, "stream order violation: {msg}");
        }

        // Every tool_use block start (including reopens after an interleave)
        // carries the original id/name and an empty input object.
        for event in &events {
            if let AnthropicSseEvent::ContentBlockStart {
                content_block: ContentBlock::ToolUse { id: bid, name: bname, input, .. },
                ..
            } = event
            {
                prop_assert_eq!(bid, &id, "tool block id must survive close/reopen");
                prop_assert_eq!(bname, &name, "tool block name must survive close/reopen");
                prop_assert_eq!(input, &json!({}), "tool block starts with empty input");
            }
        }

        // The concatenation invariant: concatenated partial_json (across all
        // blocks, in stream order) parses back to the original arguments.
        let concatenated: String = events
            .iter()
            .filter_map(|event| match event {
                AnthropicSseEvent::ContentBlockDelta {
                    delta: ContentBlockDelta::InputJsonDelta { partial_json, .. },
                    ..
                } => Some(partial_json.as_str()),
                _ => None,
            })
            .collect();
        prop_assert_eq!(&concatenated, &args_str, "partial_json concatenation must equal the arguments string");
        let parsed: Value = serde_json::from_str(&concatenated)
            .map_err(|e| TestCaseError::fail(format!("concatenated partial_json must parse: {e}")))?;
        prop_assert_eq!(parsed, Value::Object(args));

        // The per-chat-index accumulator agrees with the event stream.
        prop_assert_eq!(&accumulated, args_str.as_str());
        prop_assert_eq!(unused_accumulation, "", "unused chat index has no accumulation");
    }
}

// ---------------------------------------------------------------------------
// Property (b): finish_reason → valid stop_reason (Requirement 1.2)
// ---------------------------------------------------------------------------

proptest! {
    #![proptest_config(ProptestConfig::with_cases(64))]

    /// Every generated finish_reason (known, unknown, or missing) maps to a
    /// valid Anthropic stop_reason variant, serializes without failure, and
    /// `pause_turn` is never synthesized. Stop-sequence hits are exercised
    /// via an uppercase sequence appended to lowercase text.
    #[test]
    fn prop_finish_reason_maps_to_valid_stop_reason(
        finish in prop::option::of(prop_oneof![
            4 => Just("stop".to_string()),
            3 => Just("length".to_string()),
            3 => Just("tool_calls".to_string()),
            2 => Just("content_filter".to_string()),
            3 => proptest::string::string_regex("[a-z_]{0,12}").expect("finish regex must compile"),
        ]),
        text in proptest::string::string_regex("[a-z0-9 .]{0,48}").expect("text regex must compile"),
        stop_sequences in prop::collection::vec(
            proptest::string::string_regex("[A-Z]{2,6}").expect("seq regex must compile"),
            0..3
        ),
        append_first in any::<bool>(),
    ) {
        // Lowercase text + uppercase sequences: a suffix hit can only happen
        // when deliberately appended.
        let hit = append_first && !stop_sequences.is_empty() && finish.as_deref() == Some("stop");
        let content = if append_first && !stop_sequences.is_empty() {
            format!("{text}{}", stop_sequences[0])
        } else {
            text
        };

        let chat: OpenAIResponse = serde_json::from_value(json!({
            "id": "chatcmpl-prop",
            "model": "test-model",
            "choices": [{
                "index": 0,
                "message": { "role": "assistant", "content": content },
                "finish_reason": finish,
            }],
            "usage": { "prompt_tokens": 3, "completion_tokens": 5, "total_tokens": 8 },
        }))
        .expect("chat response fixture must parse");

        let ctx = AnthropicSynthesisContext {
            request_model: "claude-group",
            stop_sequences: &stop_sequences,
        };
        let resp = synthesize_anthropic_response(&chat, &ctx);

        // Never fails to serialize; stop_reason is always a valid enum variant.
        let wire = serde_json::to_value(&resp)
            .map_err(|e| TestCaseError::fail(format!("response must serialize: {e}")))?;
        let sr_str = wire
            .get("stop_reason")
            .and_then(Value::as_str)
            .expect("stop_reason must serialize as a string");
        prop_assert!(
            VALID_STOP_REASONS.contains(&sr_str),
            "stop_reason '{sr_str}' is not a valid serde variant"
        );
        prop_assert_ne!(sr_str, "pause_turn", "pause_turn must never be synthesized");

        let sr = resp.stop_reason.expect("a present choice always yields a stop_reason");
        match finish.as_deref() {
            Some("length") => {
                prop_assert_eq!(sr, StopReason::MaxTokens);
                prop_assert!(resp.stop_sequence.is_none());
            }
            Some("tool_calls") => {
                prop_assert_eq!(sr, StopReason::ToolUse);
                prop_assert!(resp.stop_sequence.is_none());
            }
            Some("content_filter") => {
                prop_assert_eq!(sr, StopReason::Refusal);
                prop_assert!(resp.stop_sequence.is_none());
            }
            None | Some(_) => {
                if hit {
                    prop_assert_eq!(sr, StopReason::StopSequence);
                    prop_assert_eq!(resp.stop_sequence.as_deref(), Some(stop_sequences[0].as_str()));
                } else {
                    prop_assert_eq!(sr, StopReason::EndTurn);
                    prop_assert!(resp.stop_sequence.is_none());
                }
            }
        }
    }
}

// ---------------------------------------------------------------------------
// Property (c): translator invariants (Requirement 1.2)
// ---------------------------------------------------------------------------

proptest! {
    #![proptest_config(ProptestConfig::with_cases(64))]

    /// Generated multi-turn tool conversations translate with:
    /// - tool_use ↔ tool_result id pairing preserved exactly (including
    ///   intentionally mismatched ids),
    /// - tool messages ordered directly after the assistant tool_calls
    ///   message they answer,
    /// - the message-count invariant: system (0/1) + per message (string
    ///   content → 1; blocks → #tool_results + 1 when any non-tool_result
    ///   block is present, else 0).
    #[test]
    fn prop_translator_preserves_tool_pairing_and_counts(
        system in arb_optional_text(),
        leading in arb_optional_text(),
        groups in prop::collection::vec(arb_group(), 0..=3),
        trailing in arb_optional_text(),
        empty_block_user in any::<bool>(),
    ) {
        let mut messages: Vec<AnthropicMessage> = Vec::new();
        let empty_extra = Map::new();

        if let Some(text) = &leading {
            messages.push(AnthropicMessage {
                role: "user".to_string(),
                content: MessageContent::Text(text.clone()),
                extra: empty_extra.clone(),
            });
        }

        for group in &groups {
            // Assistant turn: text preamble + one tool_use block per call.
            let assistant_blocks: Vec<ContentBlock> = std::iter::once(ContentBlock::Text {
                text: group.preamble.clone(),
                extra: empty_extra.clone(),
            })
            .chain(group.calls.iter().enumerate().map(|(i, call)| ContentBlock::ToolUse {
                id: call.id.clone(),
                name: format!("fn_{i}"),
                input: json!({ "call": i }),
                extra: empty_extra.clone(),
            }))
            .collect();
            messages.push(AnthropicMessage {
                role: "assistant".to_string(),
                content: MessageContent::Blocks(assistant_blocks),
                extra: empty_extra.clone(),
            });

            // User turn: one tool_result per call (ids optionally mismatched),
            // then an optional follow-up text block in the same message.
            let mut user_blocks: Vec<ContentBlock> = group
                .calls
                .iter()
                .map(|call| ContentBlock::ToolResult {
                    tool_use_id: call.effective_id().to_string(),
                    content: ToolResultContent::Text(call.result.clone()),
                    is_error: None,
                    extra: empty_extra.clone(),
                })
                .collect();
            if let Some(text) = &group.followup {
                user_blocks.push(ContentBlock::Text {
                    text: text.clone(),
                    extra: empty_extra.clone(),
                });
            }
            messages.push(AnthropicMessage {
                role: "user".to_string(),
                content: MessageContent::Blocks(user_blocks),
                extra: empty_extra.clone(),
            });
        }

        if let Some(text) = &trailing {
            messages.push(AnthropicMessage {
                role: "assistant".to_string(),
                content: MessageContent::Text(text.clone()),
                extra: empty_extra.clone(),
            });
        }
        if empty_block_user {
            // An empty block list emits no message.
            messages.push(AnthropicMessage {
                role: "user".to_string(),
                content: MessageContent::Blocks(Vec::new()),
                extra: empty_extra.clone(),
            });
        }

        let req = AnthropicRequest {
            model: "claude-group".to_string(),
            messages,
            system: system.clone().map(SystemPrompt::Text),
            max_tokens: 1024,
            temperature: None,
            top_p: None,
            stop_sequences: Vec::new(),
            stream: false,
            tools: Vec::new(),
            tool_choice: None,
            thinking: None,
            metadata: None,
            extra: Map::new(),
        };

        let out = translate(&req).expect("generated conversations must translate");

        // Message-count invariant, computed with the translator's own rules.
        let mut expected = if system.as_ref().is_some_and(|s| !s.is_empty()) { 1 } else { 0 };
        if leading.is_some() {
            expected += 1;
        }
        for group in &groups {
            expected += 1 + group.calls.len() + usize::from(group.followup.is_some());
        }
        if trailing.is_some() {
            expected += 1;
        }
        prop_assert_eq!(
            out.messages.len(),
            expected,
            "translated message count must match the translate_request.rs expansion rule"
        );

        // Positional walk: pairing + ordering.
        let mut cursor = 0usize;
        if system.as_ref().is_some_and(|s| !s.is_empty()) {
            prop_assert_eq!(&out.messages[0].role, "system", "system must be strictly first");
            cursor = 1;
        }
        if let Some(text) = &leading {
            prop_assert_eq!(&out.messages[cursor].role, "user");
            prop_assert_eq!(&out.messages[cursor].content, &Value::String(text.clone()));
            cursor += 1;
        }
        for group in &groups {
            // Assistant tool_calls message.
            let assistant = &out.messages[cursor];
            prop_assert_eq!(&assistant.role, "assistant");
            let tool_calls = assistant
                .extra
                .get("tool_calls")
                .and_then(Value::as_array)
                .expect("assistant message must carry tool_calls");
            prop_assert_eq!(tool_calls.len(), group.calls.len());
            for (i, call) in group.calls.iter().enumerate() {
                prop_assert_eq!(&tool_calls[i]["id"], &json!(call.id), "tool_use id preserved");
                let args = tool_calls[i]["function"]["arguments"]
                    .as_str()
                    .expect("arguments must be a string");
                let parsed: Value = serde_json::from_str(args)
                    .map_err(|e| TestCaseError::fail(format!("arguments must parse: {e}")))?;
                prop_assert_eq!(parsed, json!({ "call": i }));
            }
            cursor += 1;

            // Tool messages directly follow, in call order, with the exact
            // tool_use_id the input carried (matched or mismatched).
            for call in &group.calls {
                let tool_msg = &out.messages[cursor];
                prop_assert_eq!(&tool_msg.role, "tool", "tool message must directly follow the assistant tool_calls message");
                prop_assert_eq!(
                    tool_msg.extra.get("tool_call_id"),
                    Some(&json!(call.effective_id())),
                    "tool_call_id must preserve the input tool_use_id exactly"
                );
                cursor += 1;
            }

            if let Some(text) = &group.followup {
                prop_assert_eq!(&out.messages[cursor].role, "user");
                // After a tool_result split the remaining blocks become a
                // parts array (translate_request.rs shape), never a string.
                prop_assert_eq!(
                    &out.messages[cursor].content,
                    &Value::Array(vec![json!({ "type": "text", "text": text })])
                );
                cursor += 1;
            }
        }
        if trailing.is_some() {
            prop_assert_eq!(&out.messages[cursor].role, "assistant");
            cursor += 1;
        }
        prop_assert_eq!(cursor, out.messages.len(), "walked every translated message");
    }
}

// ---------------------------------------------------------------------------
// Deterministic sanity tests (sibling-suite convention)
// ---------------------------------------------------------------------------

#[test]
fn single_fragment_emits_full_arguments_in_one_delta() {
    let mut assembler = MessageStreamAssembler::new("claude-group", None, Usage::default());
    let events = assembler
        .feed(ChatDelta::ToolCall {
            index: 0,
            id: Some("toolu_1".to_string()),
            name: Some("lookup".to_string()),
            arguments: r#"{"city":"Paris"}"#.to_string(),
        })
        .into_iter()
        .chain(assembler.finish(
            StopReason::ToolUse,
            None,
            Usage { input_tokens: 1, output_tokens: 1, ..Default::default() },
        ))
        .collect::<Vec<_>>();

    assert!(matches!(events[0], AnthropicSseEvent::MessageStart { .. }));
    let concatenated: String = events
        .iter()
        .filter_map(|event| match event {
            AnthropicSseEvent::ContentBlockDelta {
                delta: ContentBlockDelta::InputJsonDelta { partial_json, .. },
                ..
            } => Some(partial_json.as_str()),
            _ => None,
        })
        .collect();
    assert_eq!(concatenated, r#"{"city":"Paris"}"#);
    assert_eq!(
        serde_json::from_str::<Value>(&concatenated).unwrap(),
        json!({ "city": "Paris" })
    );
    assert!(check_stream_order(&events).is_ok());
}

#[test]
fn mismatched_tool_result_id_preserved_exactly() {
    let req: AnthropicRequest = serde_json::from_value(json!({
        "model": "claude-group",
        "max_tokens": 64,
        "messages": [
            { "role": "assistant", "content": [
                { "type": "tool_use", "id": "toolu_A1", "name": "fn_0", "input": {} }
            ]},
            { "role": "user", "content": [
                { "type": "tool_result", "tool_use_id": "toolu_B9", "content": "oops" }
            ]}
        ]
    }))
    .unwrap();
    let out = translate(&req).expect("translates");
    assert_eq!(out.messages.len(), 2);
    assert_eq!(out.messages[1].role, "tool");
    assert_eq!(
        out.messages[1].extra.get("tool_call_id"),
        Some(&json!("toolu_B9")),
        "the translator is a pure mapping: mismatches pass through untouched"
    );
}
