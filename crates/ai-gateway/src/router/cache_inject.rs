//! Cache-control breakpoint injection and advancement for explicit-cache providers.
//!
//! Providers with [`crate::config::PromptCacheSupport::Explicit`] (Anthropic-style)
//! require `cache_control: {"type": "ephemeral"}` markers on content blocks to
//! activate upstream prompt caching. This module computes where those markers
//! belong and injects them into an [`OpenAIRequest`] immediately before the
//! request is forwarded (design.md, "CacheInjector").
//!
//! Placement strategy, in priority order (Requirement 2):
//!
//! 1. **Last tool definition** — the most stable prefix content (Req 2.2).
//! 2. **System prompt** — the first `system` message (Req 2.3).
//! 3. **Advancement** — when the previous turn wrote to the cache
//!    (`prior_usage.cache_creation_input_tokens > 0`), a marker is placed on
//!    the newest cacheable non-user message (preferring recent tool results,
//!    then assistant replies) so the growing conversation tail stays cached
//!    (Req 2.6). If all explicit slots are taken, the OLDEST marker is moved
//!    onto that block instead of adding a new one.
//!
//! Invariants:
//!
//! - Client-supplied `cache_control` markers are stripped first (Req 2.5).
//! - At most `max_breakpoints - 1` markers are injected; one slot stays
//!   reserved for automatic/advancement caching (Req 2.4).
//! - A marker is only placed on a block when the estimated token count of all
//!   content preceding and including that block (tools first, then messages in
//!   order) reaches `cache_min_tokens * 1.1` — a 10% safety margin so
//!   under-estimation cannot produce sub-minimum cache writes.
//! - Message ordering is never changed; markers are added in place.
//!
//! The request is mutated in place. Callers are expected to invoke this only
//! after matching the provider model's cache support to
//! [`crate::config::PromptCacheSupport::Explicit`]; the numbers in
//! [`CacheInjectorConfig`] are derived from that variant.
//!
//! # Token estimation
//!
//! `ContextManager::estimate_request_tokens` (context/manager.rs:163) and the
//! quality evaluator's message estimator (smart_routing/quality_evaluator.rs:257)
//! both operate on whole requests, while breakpoint placement needs cumulative
//! per-prefix token sums. This module therefore estimates tokens per part with
//! the codebase's existing chars-per-token heuristic
//! [`TokenCounter::estimate_heuristic`] (~4 chars/token, CJK-aware, rounds up) —
//! the same heuristic `ContextManager::estimate_tokens` (context/manager.rs:140)
//! applies to message content. Non-text JSON parts (tool definitions,
//! `tool_calls`, non-text content blocks) contribute their serialized character
//! count, mirroring how `TokenCounter::count_request` feeds `value.to_string()`
//! through the tokenizer. Per-part division slightly over-estimates mixed
//! content, which is the safe direction for the `cache_min_tokens` gate.

use std::fmt;

use serde_json::{Map, Value};

use crate::compression::token_counter::TokenCounter;
use crate::models::openai::{Message, OpenAIRequest};
use crate::router::sticky_cache::CacheUsage;

const CACHE_CONTROL_KEY: &str = "cache_control";

/// Tuning inputs for [`inject_explicit_cache_breakpoints`], derived from a
/// provider model's `PromptCacheSupport::Explicit { max_breakpoints }` entry
/// and its resolved `cache_min_tokens`.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct CacheInjectorConfig {
    /// Maximum number of cache breakpoints the provider accepts (1..=4,
    /// enforced by config validation). At most `max_breakpoints - 1` markers
    /// are ever injected.
    pub max_breakpoints: u32,
    /// Minimum uncached prefix tokens required before a breakpoint may be
    /// placed; the injector applies a further 10% safety margin.
    pub cache_min_tokens: u32,
}

/// Failures of breakpoint injection.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum CacheInjectError {
    /// More markers ended up on the request than the provider accepts.
    /// Unreachable by construction (the injector budgets markers up front);
    /// kept as a defensive post-condition check.
    TooManyBreakpoints {
        /// Total marker count observed on the request.
        placed: u32,
        /// The provider's maximum accepted breakpoints.
        max_breakpoints: u32,
    },
}

impl fmt::Display for CacheInjectError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            CacheInjectError::TooManyBreakpoints {
                placed,
                max_breakpoints,
            } => write!(
                f,
                "cache breakpoint injection placed {placed} markers but the provider accepts at most {max_breakpoints}"
            ),
        }
    }
}

impl std::error::Error for CacheInjectError {}

/// Where an injected marker currently sits, in conversation order.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum Marker {
    /// On the last tool definition in `request.extra["tools"]`.
    LastToolDefinition,
    /// On the message at this index (marker on its last content block, or on
    /// the message itself for string/other content shapes).
    Message(usize),
}

fn ephemeral_marker() -> Value {
    serde_json::json!({ "type": "ephemeral" })
}

/// Strips every client-supplied `cache_control` marker from tool definitions,
/// message content blocks, and message extension fields (Req 2.5).
fn strip_client_markers(request: &mut OpenAIRequest) {
    if let Some(Value::Array(tools)) = request.extra.get_mut("tools") {
        for tool in tools.iter_mut() {
            if let Value::Object(map) = tool {
                map.remove(CACHE_CONTROL_KEY);
            }
        }
    }
    for message in request.messages.iter_mut() {
        if let Value::Array(blocks) = &mut message.content {
            for block in blocks.iter_mut() {
                if let Value::Object(map) = block {
                    map.remove(CACHE_CONTROL_KEY);
                }
            }
        }
        message.extra.remove(CACHE_CONTROL_KEY);
    }
}

/// Returns the last tool definition object from `request.extra["tools"]`,
/// provided tools exist and the final entry is an object.
fn last_tool_definition_mut(request: &mut OpenAIRequest) -> Option<&mut Map<String, Value>> {
    match request.extra.get_mut("tools") {
        Some(Value::Array(tools)) => tools.last_mut()?.as_object_mut(),
        _ => None,
    }
}

/// True if the request's last tool definition carries a marker.
fn last_tool_definition_has_marker(request: &OpenAIRequest) -> bool {
    match request.extra.get("tools") {
        Some(Value::Array(tools)) => tools
            .last()
            .and_then(|tool| tool.get(CACHE_CONTROL_KEY))
            .is_some(),
        _ => false,
    }
}

/// Removes a marker from the last tool definition.
fn unmark_last_tool_definition(request: &mut OpenAIRequest) {
    if let Some(tool) = last_tool_definition_mut(request) {
        tool.remove(CACHE_CONTROL_KEY);
    }
}

/// True if the message carries a `cache_control` marker (on a content block
/// or on its extension fields).
fn message_has_marker(message: &Message) -> bool {
    if message.extra.contains_key(CACHE_CONTROL_KEY) {
        return true;
    }
    match &message.content {
        Value::Array(blocks) => blocks
            .iter()
            .any(|block| block.get(CACHE_CONTROL_KEY).is_some()),
        _ => false,
    }
}

/// Places an ephemeral marker on a message.
///
/// - Array content: the marker goes on the LAST content block, so the cache
///   prefix covers the whole message.
/// - String content: the string is wrapped in a synthetic text block
///   (`[{"type":"text","text":...,"cache_control":...}]`) — semantically
///   equivalent in the OpenAI request shape and the block form explicit-cache
///   (Anthropic-style) providers require.
/// - Other/null content: the marker falls back to the message's extension
///   fields.
fn mark_message(message: &mut Message) {
    if let Value::String(text) = &message.content {
        message.content = Value::Array(vec![serde_json::json!({
            "type": "text",
            "text": text,
            CACHE_CONTROL_KEY: ephemeral_marker(),
        })]);
        return;
    }
    if let Value::Array(blocks) = &mut message.content {
        for block in blocks.iter_mut().rev() {
            if let Value::Object(map) = block {
                map.insert(CACHE_CONTROL_KEY.to_string(), ephemeral_marker());
                return;
            }
        }
    }
    message
        .extra
        .insert(CACHE_CONTROL_KEY.to_string(), ephemeral_marker());
}

/// Removes every marker from a message (content blocks and extension fields).
fn unmark_message(message: &mut Message) {
    message.extra.remove(CACHE_CONTROL_KEY);
    if let Value::Array(blocks) = &mut message.content {
        for block in blocks.iter_mut() {
            if let Value::Object(map) = block {
                map.remove(CACHE_CONTROL_KEY);
            }
        }
    }
}

/// Estimated tokens for one message (chars/4 heuristic; see module docs).
fn estimate_message_tokens(message: &Message) -> u64 {
    let mut tokens = estimate_text_tokens(&message.role).saturating_add(4);
    tokens = tokens.saturating_add(estimate_content_tokens(&message.content));
    for (key, value) in &message.extra {
        tokens = tokens
            .saturating_add(estimate_text_tokens(key))
            .saturating_add(estimate_text_tokens(&value.to_string()));
    }
    tokens
}

/// Estimated tokens for a message content value. Text blocks count their text;
/// non-text blocks and structured values count their serialized form.
fn estimate_content_tokens(content: &Value) -> u64 {
    match content {
        Value::Null => 0,
        Value::String(text) => estimate_text_tokens(text),
        Value::Array(blocks) => blocks
            .iter()
            .map(|block| match block.get("text").and_then(Value::as_str) {
                Some(text) => estimate_text_tokens(text),
                None => estimate_text_tokens(&block.to_string()),
            })
            .fold(0u64, u64::saturating_add),
        other => estimate_text_tokens(&other.to_string()),
    }
}

/// Estimated tokens for an arbitrary JSON value (e.g. the tools array).
fn estimate_value_tokens(value: &Value) -> u64 {
    estimate_text_tokens(&value.to_string())
}

fn estimate_text_tokens(text: &str) -> u64 {
    u64::from(TokenCounter::estimate_heuristic(text))
}

/// Index of the newest cacheable message for advancement, or `None`.
///
/// Candidates are non-user, non-system messages (the conversation tail);
/// recent tool results are preferred, then assistant replies (Req 2.6).
/// Already-marked messages are skipped — advancing onto a marked message
/// would be a no-op.
fn newest_cacheable_message_index(messages: &[Message]) -> Option<usize> {
    let is_candidate = |message: &Message| {
        !message_has_marker(message) && !matches!(&message.role[..], "user" | "system")
    };
    for preferred in ["tool", "assistant"] {
        if let Some(index) = messages
            .iter()
            .rposition(|m| m.role == preferred && is_candidate(m))
        {
            return Some(index);
        }
    }
    messages.iter().rposition(is_candidate)
}

/// Counts all `cache_control` markers on the request (tools + messages).
fn count_markers(request: &OpenAIRequest) -> u32 {
    let mut total = 0u32;
    if last_tool_definition_has_marker(request) {
        total += 1;
    }
    total += request
        .messages
        .iter()
        .filter(|message| message_has_marker(message))
        .count() as u32;
    total
}

/// Injects gateway-computed `cache_control: {"type": "ephemeral"}` breakpoints
/// into `request`, in place.
///
/// See the module docs for the placement strategy and invariants. The caller
/// is responsible for only invoking this for models configured with
/// `PromptCacheSupport::Explicit`; `cfg` carries the numbers derived from that
/// configuration.
///
/// Returns [`CacheInjectError::TooManyBreakpoints`] only if the post-injection
/// marker count exceeds `cfg.max_breakpoints`, which the internal budget makes
/// impossible — the check is a defensive post-condition.
pub fn inject_explicit_cache_breakpoints(
    request: &mut OpenAIRequest,
    cfg: &CacheInjectorConfig,
    prior_usage: Option<&CacheUsage>,
) -> Result<(), CacheInjectError> {
    // Req 2.5: client markers are always replaced by gateway-computed ones.
    strip_client_markers(request);

    // Req 2.4: one slot stays reserved for automatic/advancement caching.
    let explicit_budget = cfg.max_breakpoints.saturating_sub(1);
    if explicit_budget == 0 {
        return Ok(());
    }

    // 10% safety margin over cache_min_tokens.
    let threshold = (u64::from(cfg.cache_min_tokens) * 11).div_ceil(10);

    let tools_tokens = request
        .extra
        .get("tools")
        .map(estimate_value_tokens)
        .unwrap_or(0);

    // Cumulative token estimate of all content preceding and including each
    // message (tools sit at the front of the prompt, then messages in order).
    let mut cumulative = tools_tokens;
    let mut prefix_inclusive = Vec::with_capacity(request.messages.len());
    for message in &request.messages {
        cumulative = cumulative.saturating_add(estimate_message_tokens(message));
        prefix_inclusive.push(cumulative);
    }

    let mut placed: u32 = 0;
    // Injected markers, in conversation order (front-most first).
    let mut markers: Vec<Marker> = Vec::new();

    // Priority 1 (Req 2.2): the last tool definition — the most stable prefix.
    if tools_tokens >= threshold && last_tool_definition_mut(request).is_some() {
        if let Some(tool) = last_tool_definition_mut(request) {
            tool.insert(CACHE_CONTROL_KEY.to_string(), ephemeral_marker());
        }
        placed += 1;
        markers.push(Marker::LastToolDefinition);
    }

    // Priority 2 (Req 2.3): the system prompt (first system message).
    if placed < explicit_budget {
        if let Some(index) = request.messages.iter().position(|m| m.role == "system") {
            if prefix_inclusive[index] >= threshold {
                mark_message(&mut request.messages[index]);
                placed += 1;
                markers.push(Marker::Message(index));
            }
        }
    }

    // Priority 3 (Req 2.6): advancement of the cache window onto the growing
    // tail when the previous turn wrote to the cache.
    let prior_created_cache =
        prior_usage.is_some_and(|usage| usage.cache_creation_input_tokens > 0);
    if prior_created_cache {
        if let Some(index) = newest_cacheable_message_index(&request.messages) {
            if prefix_inclusive[index] >= threshold {
                if placed < explicit_budget {
                    // Free slot: place an additional marker on the tail.
                    mark_message(&mut request.messages[index]);
                    markers.push(Marker::Message(index));
                } else if let Some(oldest) = markers.first().copied() {
                    // All slots used: MOVE the oldest marker onto the tail.
                    match oldest {
                        Marker::LastToolDefinition => unmark_last_tool_definition(request),
                        Marker::Message(message_index) => {
                            unmark_message(&mut request.messages[message_index]);
                        }
                    }
                    markers.remove(0);
                    mark_message(&mut request.messages[index]);
                    markers.push(Marker::Message(index));
                }
            }
        }
    }

    let total = count_markers(request);
    debug_assert!(total <= cfg.max_breakpoints);
    if total > cfg.max_breakpoints {
        return Err(CacheInjectError::TooManyBreakpoints {
            placed: total,
            max_breakpoints: cfg.max_breakpoints,
        });
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    fn request_from_json(json: &str) -> OpenAIRequest {
        serde_json::from_str(json).expect("valid test request JSON")
    }

    fn cfg(max_breakpoints: u32, cache_min_tokens: u32) -> CacheInjectorConfig {
        CacheInjectorConfig {
            max_breakpoints,
            cache_min_tokens,
        }
    }

    fn prior_usage(cache_creation: u64) -> CacheUsage {
        CacheUsage {
            cache_read_input_tokens: 0,
            cache_creation_input_tokens: cache_creation,
            uncached_input_tokens: 0,
        }
    }

    /// Marker on a message's last content block, or on its extra map.
    fn message_marker(message: &Message) -> Option<Value> {
        if let Value::Array(blocks) = &message.content {
            return blocks.last().and_then(|b| b.get("cache_control")).cloned();
        }
        message.extra.get("cache_control").cloned()
    }

    const TOOLS: &str = r#"[{"type":"function","function":{"name":"get_weather","parameters":{"type":"object","properties":{"city":{"type":"string"}}}}}]"#;

    #[test]
    fn strips_client_supplied_markers_first() {
        let mut request = request_from_json(&format!(
            r#"{{"model":"m","messages":[
                {{"role":"system","content":[{{"type":"text","text":"Sys.","cache_control":{{"type":"ephemeral"}}}}]}},
                {{"role":"user","content":"Hi","cache_control":{{"type":"ephemeral"}}}}
            ],"tools":[{{"type":"function","function":{{"name":"t"}},"cache_control":{{"type":"ephemeral"}}}}]}}"#
        ));
        inject_explicit_cache_breakpoints(&mut request, &cfg(4, 10_000), None).unwrap();
        assert!(!message_has_marker(&request.messages[0]));
        assert!(!message_has_marker(&request.messages[1]));
        assert!(!last_tool_definition_has_marker(&request));
    }

    #[test]
    fn places_first_breakpoint_on_last_tool_definition() {
        let mut request = request_from_json(&format!(
            r#"{{"model":"m","messages":[{{"role":"user","content":"Hello"}}],"tools":{TOOLS}}}"#
        ));
        inject_explicit_cache_breakpoints(&mut request, &cfg(4, 10), None).unwrap();
        let tools = request.extra["tools"].as_array().unwrap();
        let last = tools.last().unwrap();
        assert_eq!(
            last.get("cache_control"),
            Some(&json!({"type": "ephemeral"}))
        );
        assert!(!tools[0..tools.len() - 1]
            .iter()
            .any(|t| t.get("cache_control").is_some()));
        assert!(!message_has_marker(&request.messages[0]));
    }

    #[test]
    fn places_breakpoint_on_system_prompt_wrapping_string_content() {
        let mut request = request_from_json(
            r#"{"model":"m","messages":[
                {"role":"system","content":"You are a helpful assistant."},
                {"role":"user","content":"Hello"}
            ]}"#,
        );
        inject_explicit_cache_breakpoints(&mut request, &cfg(4, 1), None).unwrap();
        let system = &request.messages[0];
        match &system.content {
            Value::Array(blocks) => {
                assert_eq!(blocks.len(), 1);
                assert_eq!(blocks[0].get("type"), Some(&json!("text")));
                assert_eq!(
                    blocks[0].get("text"),
                    Some(&json!("You are a helpful assistant."))
                );
                assert_eq!(
                    blocks[0].get("cache_control"),
                    Some(&json!({"type": "ephemeral"}))
                );
            }
            other => panic!("expected wrapped block content, got {other:?}"),
        }
        assert!(!message_has_marker(&request.messages[1]));
    }

    #[test]
    fn places_marker_on_last_block_of_array_content() {
        let mut request = request_from_json(
            r#"{"model":"m","messages":[
                {"role":"system","content":[
                    {"type":"text","text":"Part one. "},
                    {"type":"text","text":"Part two, the tail."}
                ]},
                {"role":"user","content":"Hello"}
            ]}"#,
        );
        inject_explicit_cache_breakpoints(&mut request, &cfg(4, 1), None).unwrap();
        match &request.messages[0].content {
            Value::Array(blocks) => {
                assert!(blocks[0].get("cache_control").is_none());
                assert_eq!(
                    blocks[1].get("cache_control"),
                    Some(&json!({"type": "ephemeral"}))
                );
            }
            other => panic!("expected block content, got {other:?}"),
        }
    }

    #[test]
    fn never_exceeds_max_breakpoints_minus_one() {
        // Tools AND system both clear the threshold, but max_breakpoints = 2
        // reserves one slot, so only the tool definition may be marked.
        let mut request = request_from_json(&format!(
            r#"{{"model":"m","messages":[
                {{"role":"system","content":"You are a helpful assistant with a long persona description."}},
                {{"role":"user","content":"Hello"}}
            ],"tools":{TOOLS}}}"#
        ));
        inject_explicit_cache_breakpoints(&mut request, &cfg(2, 1), None).unwrap();
        assert!(last_tool_definition_has_marker(&request));
        assert!(!message_has_marker(&request.messages[0]));
        assert_eq!(count_markers(&request), 1);
    }

    #[test]
    fn max_breakpoints_of_one_injects_nothing() {
        let mut request = request_from_json(&format!(
            r#"{{"model":"m","messages":[{{"role":"system","content":"System prompt long enough."}}],"tools":{TOOLS}}}"#
        ));
        inject_explicit_cache_breakpoints(&mut request, &cfg(1, 1), None).unwrap();
        assert_eq!(count_markers(&request), 0);
    }

    #[test]
    fn places_no_markers_below_threshold() {
        let mut request = request_from_json(
            r#"{"model":"m","messages":[
                {"role":"system","content":"Short."},
                {"role":"user","content":"Hi"}
            ]}"#,
        );
        // threshold = ceil(1000 * 1.1) = 1100 tokens, far above the request.
        inject_explicit_cache_breakpoints(&mut request, &cfg(4, 1_000), None).unwrap();
        assert_eq!(count_markers(&request), 0);
    }

    #[test]
    fn advancement_moves_oldest_breakpoint_to_newest_tool_result() {
        // Budget = 2: tool definition + system get the explicit slots.
        // Prior turn created cache, so the OLDEST marker (tools) moves onto
        // the trailing tool-result message.
        let mut request = request_from_json(&format!(
            r#"{{"model":"m","messages":[
                {{"role":"system","content":"You are a helpful assistant with a long persona description."}},
                {{"role":"user","content":"What is the weather in Paris?"}},
                {{"role":"assistant","content":null,"tool_calls":[{{"id":"call_1","type":"function","function":{{"name":"get_weather","arguments":"{{\"city\":\"Paris\"}}"}}}}]}},
                {{"role":"tool","tool_call_id":"call_1","content":"18 degrees and sunny in Paris today."}}
            ],"tools":{TOOLS}}}"#
        ));
        let usage = prior_usage(512);
        inject_explicit_cache_breakpoints(&mut request, &cfg(3, 1), Some(&usage)).unwrap();
        assert!(
            !last_tool_definition_has_marker(&request),
            "oldest marker should have been moved off the tool definition"
        );
        assert!(
            message_has_marker(&request.messages[0]),
            "system keeps its slot"
        );
        assert!(!message_has_marker(&request.messages[1]));
        assert!(
            !message_has_marker(&request.messages[2]),
            "assistant is skipped in favor of the newer tool result"
        );
        assert!(message_has_marker(&request.messages[3]));
        assert_eq!(count_markers(&request), 2);
    }

    #[test]
    fn advancement_uses_free_slot_when_available() {
        // System prompt is below the threshold on its own, so its explicit
        // slot stays free; advancement fills it with the newest tool result.
        let mut request = request_from_json(
            r#"{"model":"m","messages":[
                {"role":"system","content":"Short."},
                {"role":"user","content":"What is the weather in Paris right now please?"},
                {"role":"assistant","content":null,"tool_calls":[{"id":"c1","type":"function","function":{"name":"get_weather","arguments":"{}"}}]},
                {"role":"tool","tool_call_id":"c1","content":"18 degrees and sunny in Paris today, quite warm for the season."}
            ]}"#,
        );
        let usage = prior_usage(64);
        inject_explicit_cache_breakpoints(&mut request, &cfg(3, 8), Some(&usage)).unwrap();
        assert!(
            !message_has_marker(&request.messages[0]),
            "system below threshold"
        );
        assert!(!message_has_marker(&request.messages[1]));
        assert!(!message_has_marker(&request.messages[2]));
        assert!(
            message_has_marker(&request.messages[3]),
            "tool result marked"
        );
        assert_eq!(count_markers(&request), 1);
    }

    #[test]
    fn no_advancement_without_prior_cache_creation() {
        let mut request = request_from_json(
            r#"{"model":"m","messages":[
                {"role":"system","content":"Short."},
                {"role":"user","content":"What is the weather in Paris right now please?"},
                {"role":"assistant","content":null,"tool_calls":[{"id":"c1","type":"function","function":{"name":"get_weather","arguments":"{}"}}]},
                {"role":"tool","tool_call_id":"c1","content":"18 degrees and sunny in Paris today, quite warm for the season."}
            ]}"#,
        );
        let usage = CacheUsage {
            cache_read_input_tokens: 900,
            cache_creation_input_tokens: 0,
            uncached_input_tokens: 10,
        };
        inject_explicit_cache_breakpoints(&mut request, &cfg(3, 8), Some(&usage)).unwrap();
        assert_eq!(count_markers(&request), 0);
    }

    #[test]
    fn advancement_never_marks_the_newest_user_turn() {
        let mut request = request_from_json(
            r#"{"model":"m","messages":[
                {"role":"user","content":"First question that is long enough to matter here."},
                {"role":"assistant","content":"An answer of some length to the first question."},
                {"role":"user","content":"Follow-up question, also reasonably long."}
            ]}"#,
        );
        let usage = prior_usage(32);
        inject_explicit_cache_breakpoints(&mut request, &cfg(2, 4), Some(&usage)).unwrap();
        assert!(
            !message_has_marker(&request.messages[2]),
            "user turn stays unmarked"
        );
        assert!(
            message_has_marker(&request.messages[1]),
            "assistant tail is marked instead"
        );
        assert_eq!(count_markers(&request), 1);
    }

    #[test]
    fn marker_value_is_ephemeral_object() {
        let mut request = request_from_json(
            r#"{"model":"m","messages":[
                {"role":"system","content":"You are a helpful assistant."},
                {"role":"user","content":"Hello"}
            ]}"#,
        );
        inject_explicit_cache_breakpoints(&mut request, &cfg(4, 1), None).unwrap();
        assert_eq!(
            message_marker(&request.messages[0]),
            Some(json!({"type": "ephemeral"}))
        );
    }

    #[test]
    fn message_ordering_is_preserved() {
        let json = format!(
            r#"{{"model":"m","messages":[
                {{"role":"system","content":"You are a helpful assistant with a long persona description."}},
                {{"role":"user","content":"What is the weather in Paris?"}},
                {{"role":"assistant","content":null,"tool_calls":[{{"id":"call_1","type":"function","function":{{"name":"get_weather","arguments":"{{}}"}}}}]}},
                {{"role":"tool","tool_call_id":"call_1","content":"18 degrees and sunny."}},
                {{"role":"user","content":"Thanks!"}}
            ],"tools":{TOOLS}}}"#
        );
        let mut request = request_from_json(&json);
        let roles_before: Vec<String> = request
            .messages
            .iter()
            .map(|m| m.role.clone())
            .collect();
        let usage = prior_usage(128);
        inject_explicit_cache_breakpoints(&mut request, &cfg(4, 1), Some(&usage)).unwrap();
        let roles_after: Vec<String> = request
            .messages
            .iter()
            .map(|m| m.role.clone())
            .collect();
        assert_eq!(roles_before, roles_after);
        assert_eq!(count_markers(&request), 3);
    }

    #[test]
    fn error_display_mentions_limits() {
        let error = CacheInjectError::TooManyBreakpoints {
            placed: 5,
            max_breakpoints: 4,
        };
        let message = error.to_string();
        assert!(message.contains('5') && message.contains('4'), "{message}");
    }
}
