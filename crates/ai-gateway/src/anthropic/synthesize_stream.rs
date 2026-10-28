//! Streaming Chat Completions deltas → Anthropic Messages SSE synthesizer.
//!
//! [`MessageStreamAssembler`] is a pure state machine (no I/O, no clock, no
//! tokio): the caller feeds parsed chat-completions deltas and receives
//! Anthropic SSE events in the documented order
//! `message_start → (content_block_start → content_block_delta* →
//! content_block_stop)* → message_delta → message_stop`, with `ping` allowed
//! anywhere and `error` as the post-200 failure frame.
//!
//! Interleaving rule (the LiteLLM dropped-first-text-after-thinking bug class
//! is structurally impossible here): a delta never emits unless its block is
//! open; a block is closed when a delta of a different kind — or a different
//! chat `tool_calls[].index` — arrives; every open block is closed before the
//! terminal `message_delta`.
//!
//! Tool-call argument fragments are forwarded verbatim as
//! `input_json_delta.partial_json` strings (never parsed mid-flight); the
//! assembler also accumulates them per call so callers can verify the
//! concatenation invariant (accumulated `partial_json` == final arguments
//! string).
//!
//! Stop-reason and usage mapping follow [`super::synthesize_response`]'s
//! conventions; the caller performs the finish_reason → [`StopReason`]
//! mapping and passes the result to [`MessageStreamAssembler::finish`].

use std::collections::HashMap;

use crate::anthropic::models::{
    new_message_id, AnthropicResponse, AnthropicSseEvent, ContentBlock, ContentBlockDelta,
    ErrorBody, MessageDeltaBody, StopReason, Usage,
};

type Extra = serde_json::Map<String, serde_json::Value>;

/// One chat-completions streaming delta, pre-parsed by the caller.
#[derive(Debug, Clone, PartialEq)]
pub enum ChatDelta {
    /// `delta.content` text fragment (empty fragments are ignored
    /// defensively).
    Text { text: String },
    /// One entry of `delta.tool_calls[]` from a chat chunk.
    /// `index` is the chat-level tool-call index (not the Anthropic
    /// content-block index); `id`/`name` are present on the first fragment of
    /// a call; `arguments` is a raw JSON-fragment string (possibly partial).
    ToolCall {
        index: u64,
        id: Option<String>,
        name: Option<String>,
        arguments: String,
    },
    /// Reasoning/thinking fragment, plus an optional reasoning signature
    /// fragment (providers that stream signatures deliver them alongside).
    Thinking {
        thinking: String,
        signature: Option<String>,
    },
}

impl ChatDelta {
    /// Kind key for the interleaving rule: a delta whose key differs from the
    /// open block closes it. Tool calls additionally compare the chat tool
    /// index.
    fn block_key(&self) -> BlockKey {
        match self {
            Self::Text { .. } => BlockKey::Text,
            Self::ToolCall { index, .. } => BlockKey::Tool(*index),
            Self::Thinking { .. } => BlockKey::Thinking,
        }
    }
}

/// Which content block is currently open.
#[derive(Debug, Clone, PartialEq)]
enum OpenBlock {
    Text { index: usize },
    ToolUse { index: usize, chat_index: u64 },
    Thinking { index: usize },
}

/// Kind key for the interleaving rule.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum BlockKey {
    Text,
    Tool(u64),
    Thinking,
}

/// Identity of one chat tool call (first-fragment wins), so a block reopened
/// after an interleave carries the same id/name as its original block.
#[derive(Debug, Clone, Default)]
struct ToolIdentity {
    id: String,
    name: String,
}

/// Stateful chat-deltas → Anthropic SSE event assembler.
///
/// Construction captures the `message_start` scaffold parameters (model, id,
/// input-side usage). `message_start` is emitted eagerly on the first
/// [`Self::feed`] (or [`Self::finish`]/[`Self::error`] on an empty stream),
/// which keeps the caller in control of the failover commit point.
pub struct MessageStreamAssembler {
    message: AnthropicResponse,
    started: bool,
    open: Option<OpenBlock>,
    next_index: usize,
    /// Chat tool index → id/name captured from the first fragment.
    tool_identity: HashMap<u64, ToolIdentity>,
    /// Concatenated `partial_json` per chat tool index (the concatenation
    /// invariant; also the accumulation the non-streaming path parses).
    tool_arguments: HashMap<u64, String>,
}

impl MessageStreamAssembler {
    /// Create an assembler for `model` with the `message_start` usage
    /// scaffold (zero usage when the caller has no input-side estimate yet).
    /// A unique `msg_...` id is generated unless `id` is provided.
    pub fn new(model: &str, id: Option<String>, initial_usage: Usage) -> Self {
        Self {
            message: AnthropicResponse {
                kind: "message".to_string(),
                id: id.unwrap_or_else(new_message_id),
                role: "assistant".to_string(),
                content: Vec::new(),
                model: model.to_string(),
                stop_reason: None,
                stop_sequence: None,
                usage: initial_usage,
                extra: Extra::new(),
            },
            started: false,
            open: None,
            next_index: 0,
            tool_identity: HashMap::new(),
            tool_arguments: HashMap::new(),
        }
    }

    /// The `msg_...` message id (log correlation for the handler).
    #[cfg(test)]
    pub fn message_id(&self) -> &str {
        &self.message.id
    }

    /// Feed one chat delta; returns the Anthropic SSE events it produced.
    ///
    /// The first feed emits `message_start` before any block events. A delta
    /// whose kind (or chat tool index) differs from the open block closes the
    /// open block and opens a new one, so no delta is ever emitted without a
    /// matching open block.
    pub fn feed(&mut self, delta: ChatDelta) -> Vec<AnthropicSseEvent> {
        let mut events = Vec::new();

        if !self.started {
            self.started = true;
            events.push(AnthropicSseEvent::MessageStart {
                message: self.message.clone(),
                extra: Extra::new(),
            });
        }

        // Interleaving rule: close the open block when the delta belongs to a
        // different one.
        if self.open_block_key() != Some(delta.block_key()) {
            self.close_open_block(&mut events);
        }

        match delta {
            ChatDelta::Text { text } => {
                if !text.is_empty() {
                    let index = self.ensure_text_block(&mut events);
                    events.push(delta_event(
                        index,
                        ContentBlockDelta::TextDelta {
                            text,
                            extra: Extra::new(),
                        },
                    ));
                }
            }
            ChatDelta::ToolCall {
                index: chat_index,
                id,
                name,
                arguments,
            } => {
                let identity = self.tool_identity.entry(chat_index).or_default();
                if let Some(id) = id.filter(|s| !s.is_empty()) {
                    identity.id = id;
                }
                if let Some(name) = name.filter(|s| !s.is_empty()) {
                    identity.name = name;
                }
                let index = self.ensure_tool_block(chat_index, &mut events);
                if !arguments.is_empty() {
                    self.tool_arguments
                        .entry(chat_index)
                        .or_default()
                        .push_str(&arguments);
                    events.push(delta_event(
                        index,
                        ContentBlockDelta::InputJsonDelta {
                            partial_json: arguments,
                            extra: Extra::new(),
                        },
                    ));
                }
            }
            ChatDelta::Thinking {
                thinking,
                signature,
            } => {
                if !thinking.is_empty() {
                    let index = self.ensure_thinking_block(&mut events);
                    events.push(delta_event(
                        index,
                        ContentBlockDelta::ThinkingDelta {
                            thinking,
                            extra: Extra::new(),
                        },
                    ));
                }
                if let Some(signature) = signature.filter(|s| !s.is_empty()) {
                    // A signature always belongs to a thinking block; open one
                    // if the provider emits a signature-only fragment.
                    let index = self.ensure_thinking_block(&mut events);
                    events.push(delta_event(
                        index,
                        ContentBlockDelta::SignatureDelta {
                            signature,
                            extra: Extra::new(),
                        },
                    ));
                }
            }
        }

        events
    }

    /// Terminal transition: closes the open block, then emits `message_delta`
    /// (`delta.stop_reason`/`delta.stop_sequence` + cumulative usage) and
    /// `message_stop`.
    pub fn finish(
        mut self,
        stop_reason: StopReason,
        stop_sequence: Option<String>,
        cumulative_usage: Usage,
    ) -> Vec<AnthropicSseEvent> {
        let mut events = Vec::new();
        if !self.started {
            self.started = true;
            events.push(AnthropicSseEvent::MessageStart {
                message: self.message.clone(),
                extra: Extra::new(),
            });
        }
        self.close_open_block(&mut events);
        events.push(AnthropicSseEvent::MessageDelta {
            delta: MessageDeltaBody {
                stop_reason: Some(stop_reason),
                stop_sequence,
                extra: Extra::new(),
            },
            usage: Some(cumulative_usage),
            extra: Extra::new(),
        });
        events.push(AnthropicSseEvent::MessageStop {
            extra: Extra::new(),
        });
        events
    }

pub fn ping(&self) -> AnthropicSseEvent {
    AnthropicSseEvent::Ping {
        extra: Extra::new(),
    }
}

    /// Post-200 error frame: closes any open block, then emits the Anthropic
    /// `error` event so the client never sees a silent disconnect.
    pub fn error(mut self, error: ErrorBody) -> Vec<AnthropicSseEvent> {
        let mut events = Vec::new();
        self.close_open_block(&mut events);
        events.push(AnthropicSseEvent::Error {
            error,
            extra: Extra::new(),
        });
        events
    }

    /// Concatenated `partial_json` for a chat tool index (the concatenation
    /// invariant: this equals the provider's final arguments string).
    pub fn tool_arguments(&self, chat_index: u64) -> &str {
        self.tool_arguments
            .get(&chat_index)
            .map(String::as_str)
            .unwrap_or("")
    }

    /// Block key of the open block, or a sentinel that never matches a delta
    /// when no block is open (so the first delta always opens one).
    fn open_block_key(&self) -> Option<BlockKey> {
        self.open.as_ref().map(|open| match open {
            OpenBlock::Text { .. } => BlockKey::Text,
            OpenBlock::ToolUse { chat_index, .. } => BlockKey::Tool(*chat_index),
            OpenBlock::Thinking { .. } => BlockKey::Thinking,
        })
    }

    fn ensure_text_block(&mut self, events: &mut Vec<AnthropicSseEvent>) -> usize {
        if let Some(OpenBlock::Text { index }) = self.open {
            return index;
        }
        let index = self.next_index;
        self.next_index += 1;
        self.open = Some(OpenBlock::Text { index });
        events.push(block_start_event(
            index,
            ContentBlock::Text {
                text: String::new(),
                extra: Extra::new(),
            },
        ));
        index
    }

    fn ensure_tool_block(
        &mut self,
        chat_index: u64,
        events: &mut Vec<AnthropicSseEvent>,
    ) -> usize {
        if let Some(OpenBlock::ToolUse {
            index,
            chat_index: open_chat_index,
        }) = &self.open
        {
            if *open_chat_index == chat_index {
                return *index;
            }
        }
        let identity = self
            .tool_identity
            .get(&chat_index)
            .cloned()
            .unwrap_or_default();
        let index = self.next_index;
        self.next_index += 1;
        self.open = Some(OpenBlock::ToolUse { index, chat_index });
        events.push(block_start_event(
            index,
            ContentBlock::ToolUse {
                id: identity.id,
                name: identity.name,
                input: serde_json::Value::Object(serde_json::Map::new()),
                extra: Extra::new(),
            },
        ));
        index
    }

    fn ensure_thinking_block(&mut self, events: &mut Vec<AnthropicSseEvent>) -> usize {
        if let Some(OpenBlock::Thinking { index }) = self.open {
            return index;
        }
        let index = self.next_index;
        self.next_index += 1;
        self.open = Some(OpenBlock::Thinking { index });
        events.push(block_start_event(
            index,
            ContentBlock::Thinking {
                thinking: String::new(),
                signature: String::new(),
                extra: Extra::new(),
            },
        ));
        index
    }

    fn close_open_block(&mut self, events: &mut Vec<AnthropicSseEvent>) {
        if let Some(open) = self.open.take() {
            let index = match open {
                OpenBlock::Text { index }
                | OpenBlock::ToolUse { index, .. }
                | OpenBlock::Thinking { index } => index,
            };
            events.push(AnthropicSseEvent::ContentBlockStop {
                index,
                extra: Extra::new(),
            });
        }
    }
}

fn block_start_event(index: usize, content_block: ContentBlock) -> AnthropicSseEvent {
    AnthropicSseEvent::ContentBlockStart {
        index,
        content_block,
        extra: Extra::new(),
    }
}

fn delta_event(index: usize, delta: ContentBlockDelta) -> AnthropicSseEvent {
    AnthropicSseEvent::ContentBlockDelta {
        index,
        delta,
        extra: Extra::new(),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn zero_usage() -> Usage {
        Usage::default()
    }

    fn input_usage(n: u64) -> Usage {
        Usage {
            input_tokens: n,
            ..Usage::default()
        }
    }

    fn cumulative_usage(input: u64, output: u64) -> Usage {
        Usage {
            input_tokens: input,
            output_tokens: output,
            ..Usage::default()
        }
    }

    fn text(s: &str) -> ChatDelta {
        ChatDelta::Text {
            text: s.to_string(),
        }
    }

    fn tool_head(index: u64, id: &str, name: &str) -> ChatDelta {
        ChatDelta::ToolCall {
            index,
            id: Some(id.to_string()),
            name: Some(name.to_string()),
            arguments: String::new(),
        }
    }

    fn tool_args(index: u64, arguments: &str) -> ChatDelta {
        ChatDelta::ToolCall {
            index,
            id: None,
            name: None,
            arguments: arguments.to_string(),
        }
    }

    fn thinking(s: &str) -> ChatDelta {
        ChatDelta::Thinking {
            thinking: s.to_string(),
            signature: None,
        }
    }

    /// Event-name shorthand for sequence assertions.
    fn names(events: &[AnthropicSseEvent]) -> Vec<&'static str> {
        events.iter().map(|e| e.name()).collect()
    }

    /// Indices of `content_block_start` events, in emission order.
    fn start_indexes(events: &[AnthropicSseEvent]) -> Vec<usize> {
        events
            .iter()
            .filter_map(|e| match e {
                AnthropicSseEvent::ContentBlockStart { index, .. } => Some(*index),
                _ => None,
            })
            .collect()
    }

    fn stop_indexes(events: &[AnthropicSseEvent]) -> Vec<usize> {
        events
            .iter()
            .filter_map(|e| match e {
                AnthropicSseEvent::ContentBlockStop { index, .. } => Some(*index),
                _ => None,
            })
            .collect()
    }

    fn block_kind_at_start(events: &[AnthropicSseEvent], index: usize) -> &'static str {
        events
            .iter()
            .find_map(|e| match e {
                AnthropicSseEvent::ContentBlockStart {
                    index: i,
                    content_block,
                    ..
                } if *i == index => Some(content_block.name()),
                _ => None,
            })
            .expect("content_block_start for index")
    }

    fn run(mut asm: MessageStreamAssembler, deltas: Vec<ChatDelta>) -> Vec<AnthropicSseEvent> {
        let mut events = Vec::new();
        for delta in deltas {
            events.extend(asm.feed(delta));
        }
        events
    }

    #[test]
    fn message_start_emitted_eagerly_on_first_feed_with_scaffold() {
        let mut asm = MessageStreamAssembler::new("claude-sonnet-4-5", None, input_usage(12));
        assert!(asm.message_id().starts_with("msg_"));

        let events = asm.feed(text("Hi"));
        assert_eq!(names(&events)[0], "message_start");
        let AnthropicSseEvent::MessageStart { message, .. } = &events[0] else {
            panic!("first event must be message_start");
        };
        assert_eq!(message.id, asm.message_id());
        assert_eq!(message.model, "claude-sonnet-4-5");
        assert_eq!(message.role, "assistant");
        assert!(message.content.is_empty());
        assert_eq!(message.usage.input_tokens, 12);
        assert_eq!(message.usage.output_tokens, 0);
        assert_eq!(message.stop_reason, None);
    }

    #[test]
    fn provided_id_is_used_in_message_start() {
        let asm = MessageStreamAssembler::new("m", Some("msg_fixed".into()), zero_usage());
        let events = run(asm, vec![text("x")]);
        match &events[0] {
            AnthropicSseEvent::MessageStart { message, .. } => {
                assert_eq!(message.id, "msg_fixed");
            }
            _ => panic!(),
        }
    }

    #[test]
    fn no_events_before_first_feed() {
        let asm = MessageStreamAssembler::new("m", None, zero_usage());
        let events = run(asm, vec![]);
        assert!(events.is_empty());
    }

    #[test]
    fn text_stream_full_lifecycle() {
        let asm = MessageStreamAssembler::new("m", None, zero_usage());
        let events = run(asm, vec![text("Hello"), text(" world")]);
        assert_eq!(
            names(&events),
            vec![
                "message_start",
                "content_block_start",
                "content_block_delta",
                "content_block_delta"
            ]
        );
        assert_eq!(start_indexes(&events), vec![0]);
        match &events[1] {
            AnthropicSseEvent::ContentBlockStart { content_block, .. } => {
                assert_eq!(content_block.name(), "text");
            }
            _ => panic!(),
        }

        let concatenated: String = events
            .iter()
            .filter_map(|e| match e {
                AnthropicSseEvent::ContentBlockDelta {
                    delta: ContentBlockDelta::TextDelta { text, .. },
                    ..
                } => Some(text.clone()),
                _ => None,
            })
            .collect();
        assert_eq!(concatenated, "Hello world");
    }

    #[test]
    fn interleaved_text_tool_thinking_sequential_indices() {
        let asm = MessageStreamAssembler::new("m", None, zero_usage());
        let events = run(
            asm,
            vec![
                thinking("pondering"),
                text("answer"),
                tool_head(0, "call_a", "get_weather"),
                tool_args(0, "{\"city\":"),
                tool_args(0, "\"Paris\"}"),
                tool_head(1, "call_b", "get_time"),
                tool_args(1, "{\"tz\":\"UTC\"}"),
                text(" and done"),
            ],
        );

        assert_eq!(
            names(&events),
            vec![
                "message_start",
                // thinking block (0)
                "content_block_start",
                "content_block_delta",
                "content_block_stop",
                // text block (1)
                "content_block_start",
                "content_block_delta",
                "content_block_stop",
                // tool 0 block (2)
                "content_block_start",
                "content_block_delta",
                "content_block_delta",
                "content_block_stop",
                // tool 1 block (3)
                "content_block_start",
                "content_block_delta",
                "content_block_stop",
                // text block (4)
                "content_block_start",
                "content_block_delta",
            ]
        );

        assert_eq!(start_indexes(&events), vec![0, 1, 2, 3, 4]);
        assert_eq!(block_kind_at_start(&events, 0), "thinking");
        assert_eq!(block_kind_at_start(&events, 1), "text");
        assert_eq!(block_kind_at_start(&events, 2), "tool_use");
        assert_eq!(block_kind_at_start(&events, 3), "tool_use");
        assert_eq!(block_kind_at_start(&events, 4), "text");

        // Starts/stops are strictly LIFO: every block opened except the last
        // is closed before the next opens.
        let mut open_stack: Vec<usize> = Vec::new();
        for event in &events {
            match event {
                AnthropicSseEvent::ContentBlockStart { index, .. } => open_stack.push(*index),
                AnthropicSseEvent::ContentBlockStop { index, .. } => {
                    assert_eq!(open_stack.pop(), Some(*index));
                }
                _ => {}
            }
        }
        assert_eq!(open_stack, vec![4]);
    }

    #[test]
    fn multiple_tool_calls_get_separate_blocks_with_correct_indices() {
        let asm = MessageStreamAssembler::new("m", None, zero_usage());
        let events = run(
            asm,
            vec![
                tool_head(0, "call_a", "f1"),
                tool_head(1, "call_b", "f2"),
                tool_args(1, "{\"x\":1}"),
                // A late fragment for tool 0 reopens it as a NEW block (chat
                // providers keep per-call fragments contiguous, but the
                // assembler stays correct for adversarial orders).
                tool_args(0, "{\"y\":2}"),
            ],
        );

        assert_eq!(start_indexes(&events), vec![0, 1, 2]);
        assert_eq!(stop_indexes(&events), vec![0, 1]);

        // The reopened block carries the identity captured from tool 0's
        // first fragment.
        match events.iter().find(|e| matches!(e, AnthropicSseEvent::ContentBlockStart { index: 2, .. })) {
            Some(AnthropicSseEvent::ContentBlockStart { content_block, .. }) => {
                assert_eq!(content_block.name(), "tool_use");
                match content_block {
                    ContentBlock::ToolUse { id, name, input, .. } => {
                assert_eq!(id, "call_a");
                assert_eq!(name, "f1");
                assert_eq!(input, &serde_json::json!({}));
                    }
                    _ => panic!(),
                }
            }
            _ => panic!("expected content_block_start with index 2"),
        }
    }

    #[test]
    fn partial_json_fragments_forwarded_verbatim_concatenation_invariant() {
        let fragments = [
            "{",
            "\"city\":",
            "\"Paris\",",
            "\"unit\":\"C\",",
            "\"notes\":\"a,b}c\"",
            "}",
        ];
        let final_arguments = fragments.concat();

        let mut asm = MessageStreamAssembler::new("m", None, zero_usage());
        let mut events = asm.feed(tool_head(0, "call_a", "get_weather"));
        for fragment in fragments {
            events.extend(asm.feed(tool_args(0, fragment)));
        }
        // Accumulation invariant via the accessor (checked before finish
        // consumes the assembler).
        assert_eq!(asm.tool_arguments(0), final_arguments);

        events.extend(asm.finish(StopReason::ToolUse, None, cumulative_usage(10, 5)));

        // content_block_start carries empty input: {} per the wire contract.
        match events.iter().find_map(|e| match e {
            AnthropicSseEvent::ContentBlockStart {
                content_block: ContentBlock::ToolUse { id, name, input, .. },
                ..
            } => Some((id.clone(), name.clone(), input.clone())),
            _ => None,
        }) {
        Some((id, name, input)) => {
            assert_eq!(id, "call_a");
            assert_eq!(name, "get_weather");
            assert_eq!(input, serde_json::json!({}));
        }
            None => panic!("expected tool_use content_block_start"),
        }

        // Concatenated partial_json == final arguments string, unmodified
        // (fragments include braces/commas that must not be re-encoded).
        let concatenated: String = events
            .iter()
            .filter_map(|e| match e {
                AnthropicSseEvent::ContentBlockDelta {
                    delta: ContentBlockDelta::InputJsonDelta { partial_json, .. },
                    ..
                } => Some(partial_json.clone()),
                _ => None,
            })
            .collect();
        assert_eq!(concatenated, final_arguments);
        assert_eq!(
            serde_json::from_str::<serde_json::Value>(&concatenated).unwrap(),
            serde_json::json!({"city": "Paris", "unit": "C", "notes": "a,b}c"})
        );
    }

    #[test]
    fn tool_arguments_accessor_accumulates_per_chat_index() {
        let mut asm = MessageStreamAssembler::new("m", None, zero_usage());
        asm.feed(tool_head(0, "call_a", "f"));
        asm.feed(tool_args(0, "{\"a\":"));
        asm.feed(tool_args(0, "1}"));
        asm.feed(tool_head(1, "call_b", "g"));
        asm.feed(tool_args(1, "{}"));
        assert_eq!(asm.tool_arguments(0), "{\"a\":1}");
        assert_eq!(asm.tool_arguments(1), "{}");
        assert_eq!(asm.tool_arguments(9), "");
    }

    #[test]
    fn thinking_block_emits_thinking_and_signature_deltas() {
        let asm = MessageStreamAssembler::new("m", None, zero_usage());
        let events = run(
            asm,
            vec![
                ChatDelta::Thinking {
                    thinking: "step 1".into(),
                    signature: None,
                },
                ChatDelta::Thinking {
                    thinking: " step 2".into(),
                    signature: Some("sig_abc".into()),
                },
            ],
        );

        assert_eq!(
            names(&events),
            vec![
                "message_start",
                "content_block_start",
                "content_block_delta",
                "content_block_delta",
                "content_block_delta"
            ]
        );
        assert_eq!(block_kind_at_start(&events, 0), "thinking");
        match &events[2] {
            AnthropicSseEvent::ContentBlockDelta {
                index,
                delta: ContentBlockDelta::ThinkingDelta { thinking, .. },
                ..
            } => {
                assert_eq!(*index, 0);
                assert_eq!(thinking, "step 1");
            }
            _ => panic!("expected thinking_delta"),
        }
        match &events[3] {
            AnthropicSseEvent::ContentBlockDelta {
                delta: ContentBlockDelta::ThinkingDelta { thinking, .. },
                ..
            } => assert_eq!(thinking, " step 2"),
            _ => panic!("expected thinking_delta"),
        }
        match &events[4] {
            AnthropicSseEvent::ContentBlockDelta {
                delta: ContentBlockDelta::SignatureDelta { signature, .. },
                ..
            } => assert_eq!(signature, "sig_abc"),
            _ => panic!("expected signature_delta"),
        }
    }

    #[test]
    fn signature_only_fragment_opens_thinking_block() {
        let asm = MessageStreamAssembler::new("m", None, zero_usage());
        let events = run(
            asm,
            vec![ChatDelta::Thinking {
                thinking: String::new(),
                signature: Some("sig_only".into()),
            }],
        );
        assert_eq!(
            names(&events),
            vec![
                "message_start",
                "content_block_start",
                "content_block_delta"
            ]
        );
        assert_eq!(block_kind_at_start(&events, 0), "thinking");
    }

    #[test]
    fn first_text_after_thinking_opens_new_text_block_not_dropped() {
        // Regression shape of the LiteLLM dropped-first-text-after-thinking
        // bug: the first text delta after a thinking block must produce
        // content_block_start for a NEW text block followed by its
        // text_delta — never a bare delta without an open block.
        let asm = MessageStreamAssembler::new("m", None, zero_usage());
        let events = run(asm, vec![thinking("hmm"), text("the answer")]);

        assert_eq!(
            names(&events),
            vec![
                "message_start",
                "content_block_start",
                "content_block_delta",
                "content_block_stop",
                "content_block_start",
                "content_block_delta"
            ]
        );
        assert_eq!(block_kind_at_start(&events, 1), "text");
        match &events[events.len() - 1] {
            AnthropicSseEvent::ContentBlockDelta {
                index,
                delta: ContentBlockDelta::TextDelta { text, .. },
                ..
            } => {
                assert_eq!(*index, 1);
                assert_eq!(text, "the answer");
            }
            _ => panic!("expected final text_delta on new block index 1"),
        }
    }

    #[test]
    fn finish_closes_open_block_then_message_delta_and_stop() {
        let mut asm = MessageStreamAssembler::new("m", None, zero_usage());
        let mut events = asm.feed(text("Hi"));
        events.extend(asm.finish(
            StopReason::StopSequence,
            Some("END".to_string()),
            cumulative_usage(7, 3),
        ));

        assert_eq!(
            names(&events),
            vec![
                "message_start",
                "content_block_start",
                "content_block_delta",
                "content_block_stop",
                "message_delta",
                "message_stop"
            ]
        );
        match &events[4] {
            AnthropicSseEvent::MessageDelta { delta, usage, .. } => {
                assert_eq!(delta.stop_reason, Some(StopReason::StopSequence));
                assert_eq!(delta.stop_sequence.as_deref(), Some("END"));
                let usage = usage.as_ref().expect("cumulative usage");
                assert_eq!(usage.input_tokens, 7);
                assert_eq!(usage.output_tokens, 3);
            }
            _ => panic!("expected message_delta"),
        }
    }

    #[test]
    fn finish_stop_reason_variants_pass_through() {
        for reason in [
            StopReason::EndTurn,
            StopReason::MaxTokens,
            StopReason::ToolUse,
            StopReason::Refusal,
        ] {
            let asm = MessageStreamAssembler::new("m", None, zero_usage());
            let events = asm.finish(reason, None, cumulative_usage(1, 1));
            match &events[1] {
                AnthropicSseEvent::MessageDelta { delta, .. } => {
                    assert_eq!(delta.stop_reason, Some(reason));
                    assert_eq!(delta.stop_sequence, None);
                }
                _ => panic!("expected message_delta"),
            }
            assert_eq!(events.last().unwrap().name(), "message_stop");
        }
    }

    #[test]
    fn finish_without_open_block_emits_no_content_block_stop() {
        let asm = MessageStreamAssembler::new("m", None, zero_usage());
        let events = asm.finish(StopReason::EndTurn, None, cumulative_usage(1, 1));
        assert_eq!(
            names(&events),
            vec!["message_start", "message_delta", "message_stop"]
        );
    }

    #[test]
    fn finish_on_never_started_stream_still_emits_message_start_first() {
        let asm = MessageStreamAssembler::new("m", None, zero_usage());
        let events = asm.finish(StopReason::EndTurn, None, cumulative_usage(1, 2));
        assert_eq!(names(&events)[0], "message_start");
    }

#[test]
fn ping_event_shape_and_cadence_with_injected_time() {
    let mut asm = MessageStreamAssembler::new("m", None, zero_usage());
    // Keep-alive loop over injected time; cadence is the caller's choice.
    for _now_ms in [0u64, 5_000, 10_000] {
        let event = asm.ping();
        assert_eq!(event.name(), "ping");
        match &event {
            AnthropicSseEvent::Ping { extra } => assert!(extra.is_empty()),
            _ => unreachable!(),
        }
    }

    // ping does not disturb block state or start the message.
    let mut events = vec![asm.ping()];
    events.extend(asm.feed(text("x")));
        assert_eq!(
            names(&events),
            vec![
                "ping",
                "message_start",
                "content_block_start",
                "content_block_delta"
            ]
        );
    }

    #[test]
    fn error_frame_closes_open_block_then_emits_error_event() {
        let mut asm = MessageStreamAssembler::new("m", None, zero_usage());
        let mut events = asm.feed(text("partial"));
        events.extend(asm.error(ErrorBody {
            kind: "overloaded_error".to_string(),
            message: "upstream busy".to_string(),
            extra: Extra::new(),
        }));

        assert_eq!(
            names(&events),
            vec![
                "message_start",
                "content_block_start",
                "content_block_delta",
                "content_block_stop",
                "error"
            ]
        );
        match &events[events.len() - 1] {
            AnthropicSseEvent::Error { error, .. } => {
                assert_eq!(error.kind, "overloaded_error");
                assert_eq!(error.message, "upstream busy");
            }
            _ => panic!("expected error event"),
        }
    }

    #[test]
    fn error_before_any_feed_is_just_the_error_frame() {
        let asm = MessageStreamAssembler::new("m", None, zero_usage());
        let events = asm.error(ErrorBody {
            kind: "api_error".to_string(),
            message: "boom".to_string(),
            extra: Extra::new(),
        });
        assert_eq!(names(&events), vec!["error"]);
    }

    #[test]
    fn empty_fragments_never_emit_deltas_or_spurious_blocks() {
        let asm = MessageStreamAssembler::new("m", None, zero_usage());
        let events = run(
            asm,
            vec![
                text(""),
                tool_head(0, "c", "f"),
                ChatDelta::Thinking {
                    thinking: String::new(),
                    signature: Some(String::new()),
                },
            ],
        );
        // message_start + the tool_use block head only; the empty thinking
        // fragment emitted no delta and opened no thinking block. It still
        // closed the tool block — correct per the interleaving rule, since
        // it is a different-kind delta.
        assert_eq!(
            names(&events),
            vec!["message_start", "content_block_start", "content_block_stop"]
        );
        assert_eq!(block_kind_at_start(&events, 0), "tool_use");
    }
}
