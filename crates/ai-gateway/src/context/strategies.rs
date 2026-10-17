//! Context truncation strategies.
//!
//! Different strategies for handling requests that exceed model context windows.

use crate::models::openai::Message;
use serde_json::Value;

/// Strategy for truncating context when it exceeds model limits
#[derive(Debug, Clone)]
pub enum TruncationStrategy {
    /// Remove oldest messages until context fits within limits
    /// Preserves system messages and most recent user/assistant exchanges
    RemoveOldest,

    /// Keep only the most recent N messages (sliding window)
    /// Preserves system messages and last N user/assistant exchanges
    SlidingWindow { window_size: usize },
}

/// Result of context truncation
#[derive(Debug, Clone)]
pub struct TruncationResult {
    /// Whether any truncation was performed
    pub truncated: bool,
    /// Number of messages removed
    pub messages_removed: usize,
    /// Estimated tokens before truncation
    pub original_tokens: u32,
    /// Estimated tokens after truncation
    pub final_tokens: u32,
}

impl TruncationResult {
    /// Create a result indicating no truncation was needed
    pub fn no_truncation(original_tokens: u32) -> Self {
        Self {
            truncated: false,
            messages_removed: 0,
            original_tokens,
            final_tokens: original_tokens,
        }
    }

    /// Create a result indicating truncation was performed
    pub fn truncated(messages_removed: usize, original_tokens: u32, final_tokens: u32) -> Self {
        Self {
            truncated: true,
            messages_removed,
            original_tokens,
            final_tokens,
        }
    }
}

/// Apply truncation strategy to request messages
pub fn apply_truncation_strategy(
    messages: &mut Vec<Message>,
    max_tokens: u32,
    strategy: TruncationStrategy,
    estimate_fn: impl Fn(&[Message]) -> u32,
) -> TruncationResult {
    let original_tokens = estimate_fn(messages);

    // If already within limits, no truncation needed
    if original_tokens <= max_tokens {
        return TruncationResult::no_truncation(original_tokens);
    }

    let messages_removed = match strategy {
        TruncationStrategy::RemoveOldest => {
            truncate_remove_oldest(messages, max_tokens, &estimate_fn)
        }
        TruncationStrategy::SlidingWindow { window_size } => {
            truncate_sliding_window(messages, window_size)
        }
    };

    let final_tokens = estimate_fn(messages);
    TruncationResult::truncated(messages_removed, original_tokens, final_tokens)
}

/// Remove oldest messages until context fits.
///
/// Preserves system messages and tool-use message groups.  A "tool group"
/// is an assistant message that contains `tool_calls` followed by one or
/// more `tool` role messages.  Removing only part of a group produces a
/// malformed conversation that providers reject or that causes models to
/// fall back to XML-style tool use in plain text.
fn truncate_remove_oldest(
    messages: &mut Vec<Message>,
    max_tokens: u32,
    estimate_fn: &impl Fn(&[Message]) -> u32,
) -> usize {
    let mut removed = 0;

    while estimate_fn(messages) > max_tokens {
        // Find the oldest non-system message
        let Some(idx) = messages.iter().position(|m| m.role != "system") else {
            break; // Only system messages left
        };

        let msg = &messages[idx];

        if msg.role == "assistant" && msg.extra.contains_key("tool_calls") {
            // This is the start of a tool-use group.  Remove the assistant
            // message AND all immediately following "tool" messages.
            messages.remove(idx);
            removed += 1;
            while idx < messages.len() && messages[idx].role == "tool" {
                messages.remove(idx);
                removed += 1;
            }
        } else if msg.role == "tool" {
            // Orphaned tool result (its assistant message was already removed
            // or is missing).  Remove it.
            messages.remove(idx);
            removed += 1;
        } else {
            // Regular user/assistant message — safe to remove individually.
            messages.remove(idx);
            removed += 1;
        }
    }

    removed
}

/// Keep only the most recent N messages (preserving tool-use groups).
///
/// When the window boundary falls inside a tool-use group (assistant with
/// tool_calls + following tool messages), the entire group is included so
/// the conversation stays well-formed.
fn truncate_sliding_window(messages: &mut Vec<Message>, window_size: usize) -> usize {
    if messages.len() <= window_size {
        return 0;
    }

    // Preserve system messages
    let system_messages: Vec<Message> = messages
        .iter()
        .filter(|m| m.role == "system")
        .cloned()
        .collect();

    let non_system_messages: Vec<Message> = messages
        .iter()
        .filter(|m| m.role != "system")
        .cloned()
        .collect();

    if non_system_messages.len() <= window_size {
        return 0;
    }

    // Take the last `window_size` non-system messages
    let start = non_system_messages.len() - window_size;
    let mut recent_messages: Vec<Message> = non_system_messages[start..].to_vec();

    // If the first kept message is a "tool" result, we need to also include
    // the preceding assistant message (with tool_calls) so the conversation
    // is well-formed.  Walk backwards from `start` to find the group head.
    if !recent_messages.is_empty() && recent_messages[0].role == "tool" {
        let mut prepend_from = start;
        while prepend_from > 0 {
            prepend_from -= 1;
            let msg = &non_system_messages[prepend_from];
            if msg.role == "assistant" && msg.extra.contains_key("tool_calls") {
                // Found the group head — include it and everything between
                let mut prefix: Vec<Message> = non_system_messages[prepend_from..start].to_vec();
                prefix.append(&mut recent_messages);
                recent_messages = prefix;
                break;
            }
            if msg.role != "tool" {
                // Hit a non-tool, non-assistant-with-tool_calls message — stop
                break;
            }
        }
    }

    let removed = messages.len() - (system_messages.len() + recent_messages.len());

    *messages = system_messages
        .into_iter()
        .chain(recent_messages.into_iter())
        .collect();

    removed
}

/// Clear the contents of old tool results, keeping structure intact.
///
/// Walks messages chronologically and replaces the `content` of
/// `role == "tool"` messages older than the last `keep_recent` tool results
/// with `placeholder`. `tool_call_id` (and all other structure) is
/// preserved, so every assistant `tool_calls` message keeps its responses.
/// Tool results whose paired tool name appears in `exclude_tools` are never
/// cleared. The `keep_recent` boundary is **batch-atomic**: an assistant
/// `tool_calls` message and all of its parallel tool results share one
/// boundary slot, so a multi-call batch is never partially cleared at the
/// keep boundary (a model that sees call B's output but not call A's output
/// from the same turn receives a malformed conversation). Returns the
/// number of tool results cleared.
pub fn clear_tool_results(
    messages: &mut [Message],
    keep_recent: usize,
    exclude_tools: &[String],
    placeholder: &str,
) -> usize {
    // Tool-result batches: each entry is (assistant_index, [tool_indexes])
    // grouped by the assistant `tool_calls` message the results answer to.
    // A result with no matching assistant (orphan) forms its own batch.
    let mut batches: Vec<(usize, Vec<usize>)> = Vec::new();
    for (index, message) in messages.iter().enumerate() {
        if message.role != "tool" {
            if message.role == "assistant" && message.extra.contains_key("tool_calls") {
                batches.push((index, Vec::new()));
            }
            continue;
        }
        let tool_id = message.extra.get("tool_call_id").and_then(Value::as_str);
        let owner = tool_id.and_then(|id| {
            batches
                .iter()
                .enumerate()
                .rev()
                .find_map(|(batch_index, (assistant_index, _))| {
                    let owner = messages[*assistant_index]
                        .extra
                        .get("tool_calls")
                        .and_then(Value::as_array)
                        .is_some_and(|calls| {
                            calls
                                .iter()
                                .any(|call| call.get("id").and_then(Value::as_str) == Some(id))
                        });
                    owner.then_some(batch_index)
                })
        });
        match owner {
            Some(batch_index) => batches[batch_index].1.push(index),
            None => batches.push((index, vec![index])), // orphan: its own batch
        }
    }

    // Count from the newest batch backwards; a batch consumes exactly one
    // keep slot regardless of how many parallel results it contains. All
    // results in a kept batch stay intact (never partially cleared).
    let mut kept_batches = 0usize;
    let mut clearable_from = batches.len();
    for batch in batches.iter().rev() {
        if kept_batches >= keep_recent {
            break;
        }
        if batch.1.is_empty() {
            continue; // assistant tool_calls with no results yet: nothing to keep
        }
        kept_batches += 1;
        clearable_from -= 1;
    }

    let mut cleared = 0;
    for batch_index in 0..clearable_from {
        for &position in &batches[batch_index].1 {
            let tool_id = messages[position]
                .extra
                .get("tool_call_id")
                .and_then(Value::as_str)
                .map(str::to_owned);

            let Some(tool_id) = tool_id else {
                messages[position].content = Value::String(placeholder.to_owned());
                cleared += 1;
                continue;
            };

            let excluded = messages[..position].iter().any(|m| {
                m.role == "assistant"
                    && m
                        .extra
                        .get("tool_calls")
                        .and_then(Value::as_array)
                        .is_some_and(|calls| {
                            calls.iter().any(|call| {
                                call.get("id").and_then(Value::as_str) == Some(tool_id.as_str())
                                    && call
                                        .pointer("/function/name")
                                        .and_then(Value::as_str)
                                        .is_some_and(|name| {
                                            exclude_tools.iter().any(|e| e == name)
                                        })
                            })
                        })
            });

            if !excluded {
                messages[position].content = Value::String(placeholder.to_owned());
                cleared += 1;
            }
        }
    }

    cleared
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::models::openai::Message;

    fn create_test_message(role: &str, content: &str) -> Message {
        Message {
            role: role.to_string(),
            content: serde_json::Value::String(content.to_string()),
            extra: Default::default(),
        }
    }

    fn simple_estimate(messages: &[Message]) -> u32 {
        // Rough estimate: 4 characters per token
        messages
            .iter()
            .map(|m| m.content_as_text().len() as u32 / 4)
            .sum()
    }

    #[test]
    fn test_no_truncation_when_within_limits() {
        let mut messages = vec![
            create_test_message("system", "You are helpful"),
            create_test_message("user", "Hello"),
            create_test_message("assistant", "Hi there"),
        ];

        let result = apply_truncation_strategy(
            &mut messages,
            1000, // well above estimate
            TruncationStrategy::RemoveOldest,
            &simple_estimate,
        );

        assert!(!result.truncated);
        assert_eq!(result.messages_removed, 0);
        assert_eq!(messages.len(), 3);
    }

    #[test]
    fn test_remove_oldest_preserves_system() {
        let mut messages = vec![
            create_test_message("system", "You are helpful"),
            create_test_message("user", "First message"),
            create_test_message("assistant", "First response"),
            create_test_message("user", "Second message"),
            create_test_message("assistant", "Second response"),
        ];

        let result = apply_truncation_strategy(
            &mut messages,
            5, // Low enough to trigger truncation (total estimate ~15 tokens)
            TruncationStrategy::RemoveOldest,
            &simple_estimate,
        );

        assert!(result.truncated);
        assert!(result.messages_removed > 0);
        assert!(messages.iter().any(|m| m.role == "system"));
        // System message should still be present
        assert_eq!(messages[0].role, "system");
    }

    #[test]
    fn test_sliding_window() {
        let mut messages = vec![
            create_test_message("system", "You are helpful"),
            create_test_message("user", "Old message 1"),
            create_test_message("assistant", "Response 1"),
            create_test_message("user", "Old message 2"),
            create_test_message("assistant", "Response 2"),
            create_test_message("user", "Recent message"),
            create_test_message("assistant", "Recent response"),
        ];

        let result = apply_truncation_strategy(
            &mut messages,
            5, // Low enough to trigger truncation (total estimate ~20 tokens)
            TruncationStrategy::SlidingWindow { window_size: 2 },
            &simple_estimate,
        );

        assert!(result.truncated);
        assert!(messages.iter().any(|m| m.role == "system"));
        // Should keep system + 2 most recent non-system messages
        assert_eq!(messages.len(), 3); // system + 2 recent
    }

    fn assistant_with_tool_call(id: &str, name: &str) -> Message {
        Message {
            role: "assistant".to_string(),
            content: serde_json::Value::Null,
            extra: {
                let mut extra = serde_json::Map::new();
                extra.insert(
                    "tool_calls".to_string(),
                    serde_json::json!([{
                        "id": id,
                        "type": "function",
                        "function": {"name": name, "arguments": "{}"}
                    }]),
                );
                extra
            },
        }
    }

    fn tool_result_message(id: &str, content: &str) -> Message {
        Message {
            role: "tool".to_string(),
            content: serde_json::Value::String(content.to_string()),
            extra: {
                let mut extra = serde_json::Map::new();
                extra.insert(
                    "tool_call_id".to_string(),
                    serde_json::Value::String(id.to_string()),
                );
                extra
            },
        }
    }

    fn tool_call_ids(message: &Message) -> Vec<&str> {
        message
            .extra
            .get("tool_calls")
            .and_then(|v| v.as_array())
            .map(|calls| {
                calls
                    .iter()
                    .filter_map(|c| c.get("id").and_then(|v| v.as_str()))
                    .collect()
            })
            .unwrap_or_default()
    }

    fn assert_tool_structure_well_formed(messages: &[Message]) {
        for (index, msg) in messages.iter().enumerate() {
            if msg.role != "tool" {
                continue;
            }
            let id = msg
                .extra
                .get("tool_call_id")
                .and_then(|v| v.as_str())
                .expect("tool result keeps tool_call_id");
            let paired = messages[..index].iter().any(|m| {
                m.role == "assistant" && tool_call_ids(m).contains(&id)
            });
            assert!(paired, "orphaned tool result {id} at index {index}");
        }
    }

    #[test]
    fn test_clear_tool_results_honors_keep_recent() {
        let mut messages = vec![
            create_test_message("system", "You are helpful"),
            assistant_with_tool_call("call-0", "fetch"),
            tool_result_message("call-0", &"old output ".repeat(50)),
            assistant_with_tool_call("call-1", "fetch"),
            tool_result_message("call-1", &"mid output ".repeat(50)),
            assistant_with_tool_call("call-2", "fetch"),
            tool_result_message("call-2", &"recent output ".repeat(50)),
        ];

        let cleared = clear_tool_results(&mut messages, 1, &[], "[cleared]");

        assert_eq!(cleared, 2);
        assert_eq!(
            messages[2].content_as_text(),
            "[cleared]",
            "oldest tool result cleared"
        );
        assert_eq!(
            messages[4].content_as_text(),
            "[cleared]",
            "middle tool result cleared"
        );
        assert_eq!(
            messages[6].content_as_text(),
            "recent output ".repeat(50),
            "most recent tool result kept"
        );
        assert_eq!(messages.len(), 7, "no messages removed");
        assert_tool_structure_well_formed(&messages);
    }

    #[test]
    fn test_clear_tool_results_honors_exclude_list() {
        let excluded: Vec<String> = vec!["precious".to_string()];
        let mut messages = vec![
            assistant_with_tool_call("call-0", "precious"),
            tool_result_message("call-0", &"precious output ".repeat(50)),
            assistant_with_tool_call("call-1", "fetch"),
            tool_result_message("call-1", &"normal output ".repeat(50)),
            assistant_with_tool_call("call-2", "fetch"),
            tool_result_message("call-2", &"recent output ".repeat(50)),
        ];

        let cleared = clear_tool_results(&mut messages, 1, &excluded, "[cleared]");

        assert_eq!(cleared, 1);
        assert_eq!(
            messages[1].content_as_text(),
            "precious output ".repeat(50),
            "excluded tool result untouched"
        );
        assert_eq!(
            messages[3].content_as_text(),
            "[cleared]",
            "non-excluded old tool result cleared"
        );
        assert_eq!(
            messages[5].content_as_text(),
            "recent output ".repeat(50),
            "most recent tool result kept"
        );
        assert_tool_structure_well_formed(&messages);
    }

    #[test]
    fn test_clear_tool_results_preserves_tool_call_id_and_structure() {
        let mut messages = vec![
            assistant_with_tool_call("call-0", "fetch"),
            tool_result_message("call-0", &"old output ".repeat(50)),
            assistant_with_tool_call("call-1", "fetch"),
            tool_result_message("call-1", &"recent output ".repeat(50)),
        ];

        let cleared = clear_tool_results(&mut messages, 1, &[], "[cleared]");

        assert_eq!(cleared, 1);
        assert_eq!(
            messages[1]
                .extra
                .get("tool_call_id")
                .and_then(|v| v.as_str()),
            Some("call-0")
        );
        assert_eq!(messages[1].role, "tool");
        assert_eq!(
            messages[1].content_as_text(),
            "[cleared]",
            "content replaced with placeholder"
        );
        assert_eq!(
            tool_call_ids(&messages[0]),
            vec!["call-0"],
            "adjacent assistant tool_calls intact"
        );
        assert_tool_structure_well_formed(&messages);
    }

    #[test]
    fn test_clear_tool_results_keep_all_when_keep_recent_covers_everything() {
        let mut messages = vec![
            assistant_with_tool_call("call-0", "fetch"),
            tool_result_message("call-0", "output 0"),
            assistant_with_tool_call("call-1", "fetch"),
            tool_result_message("call-1", "output 1"),
        ];

        let cleared = clear_tool_results(&mut messages, 2, &[], "[cleared]");

        assert_eq!(cleared, 0);
        assert_eq!(messages[1].content_as_text(), "output 0");
        assert_eq!(messages[3].content_as_text(), "output 1");
    }

#[test]
fn test_remove_oldest_never_orphans_tool_results() {
    let mut messages = vec![
        create_test_message("system", "You are helpful"),
        assistant_with_tool_call("call-0", "fetch"),
        tool_result_message("call-0", &"output ".repeat(200)),
        assistant_with_tool_call("call-1", "fetch"),
        tool_result_message("call-1", &"output ".repeat(200)),
        create_test_message("user", "Final question"),
    ];

    let result = apply_truncation_strategy(
        &mut messages,
        5,
        TruncationStrategy::RemoveOldest,
        &simple_estimate,
    );

    assert!(result.truncated);
    assert!(result.messages_removed > 0);
    assert_tool_structure_well_formed(&messages);
    assert_eq!(messages[0].role, "system");
}

/// One assistant turn issuing two parallel tool calls: both results form a
/// single batch, so a `keep_recent` boundary falling inside that turn can
/// never clear only one of the pair (a model that sees call B's output but
/// not call A's receives a malformed conversation).
fn assistant_with_parallel_tool_calls(first: (&str, &str), second: (&str, &str)) -> Message {
    Message {
        role: "assistant".to_string(),
        content: serde_json::Value::Null,
        extra: {
            let mut extra = serde_json::Map::new();
            extra.insert(
                "tool_calls".to_string(),
                serde_json::json!([
                    {
                        "id": first.0,
                        "type": "function",
                        "function": {"name": first.1, "arguments": "{}"}
                    },
                    {
                        "id": second.0,
                        "type": "function",
                        "function": {"name": second.1, "arguments": "{}"}
                    }
                ]),
            );
            extra
        },
    }
}

#[test]
fn clear_tool_results_batch_boundary_never_splits_parallel_tool_calls() {
    // Two batches: an older parallel pair (call-0a/call-0b) and a newer
    // parallel pair (call-1a/call-1b). keep_recent=1 keeps the newest
    // BATCH; the older pair must be cleared atomically (both or — with an
    // exclude on one — neither eligible result partially cleared by the
    // boundary itself).
    let mut messages = vec![
        create_test_message("system", "You are helpful"),
        assistant_with_parallel_tool_calls(("call-0a", "fetch"), ("call-0b", "search")),
        tool_result_message("call-0a", &"older fetch output ".repeat(30)),
        tool_result_message("call-0b", &"older search output ".repeat(30)),
        assistant_with_parallel_tool_calls(("call-1a", "fetch"), ("call-1b", "search")),
        tool_result_message("call-1a", &"newer fetch output ".repeat(30)),
        tool_result_message("call-1b", &"newer search output ".repeat(30)),
        create_test_message("user", "Continue"),
    ];

    let cleared = clear_tool_results(&mut messages, 1, &[], "[cleared]");

    assert_eq!(cleared, 2, "the whole older batch is cleared together");
    assert_eq!(messages[2].content_as_text(), "[cleared]");
    assert_eq!(messages[3].content_as_text(), "[cleared]");
    assert_eq!(
        messages[5].content_as_text(),
        "newer fetch output ".repeat(30),
        "kept batch never partially cleared"
    );
    assert_eq!(
        messages[6].content_as_text(),
        "newer search output ".repeat(30),
        "kept batch never partially cleared"
    );
    assert_tool_structure_well_formed(&messages);
}

#[test]
fn clear_tool_results_keep_boundary_counts_batches_not_individual_results() {
    // Three batches where the middle one is a parallel pair: with
    // keep_recent=2 the two NEWEST batches (the pair + the last single)
    // survive; only the oldest single-call batch is cleared. Under
    // per-result counting the pair would count as two, clearing the
    // single call at the boundary.
    let mut messages = vec![
        assistant_with_tool_call("call-0", "fetch"),
        tool_result_message("call-0", &"oldest output ".repeat(30)),
        assistant_with_parallel_tool_calls(("call-1a", "fetch"), ("call-1b", "search")),
        tool_result_message("call-1a", &"mid fetch output ".repeat(30)),
        tool_result_message("call-1b", &"mid search output ".repeat(30)),
        assistant_with_tool_call("call-2", "fetch"),
        tool_result_message("call-2", &"recent output ".repeat(30)),
    ];

    let cleared = clear_tool_results(&mut messages, 2, &[], "[cleared]");

    assert_eq!(cleared, 1, "only the oldest batch is cleared");
    assert_eq!(messages[1].content_as_text(), "[cleared]");
    assert_eq!(
        messages[3].content_as_text(),
        "mid fetch output ".repeat(30),
        "parallel pair kept intact by batch counting"
    );
    assert_eq!(
        messages[4].content_as_text(),
        "mid search output ".repeat(30),
        "parallel pair kept intact by batch counting"
    );
    assert_eq!(
        messages[6].content_as_text(),
        "recent output ".repeat(30)
    );
    assert_tool_structure_well_formed(&messages);
}

#[test]
fn clear_tool_results_exclude_interacts_with_parallel_batches() {
    // The older parallel batch mixes an excluded tool (search) with a
    // normal one (fetch): the excluded result stays verbatim while its
    // batch partner is cleared — and the batch still consumes exactly one
    // keep slot, so the newer batch survives on its own merits.
    let excluded: Vec<String> = vec!["search".to_string()];
    let mut messages = vec![
        assistant_with_parallel_tool_calls(("call-0a", "fetch"), ("call-0b", "search")),
        tool_result_message("call-0a", &"older fetch output ".repeat(30)),
        tool_result_message("call-0b", &"older search output ".repeat(30)),
        assistant_with_parallel_tool_calls(("call-1a", "fetch"), ("call-1b", "fetch")),
        tool_result_message("call-1a", &"newer fetch output a ".repeat(30)),
        tool_result_message("call-1b", &"newer fetch output b ".repeat(30)),
    ];

    let cleared = clear_tool_results(&mut messages, 1, &excluded, "[cleared]");

    assert_eq!(cleared, 1, "only the non-excluded member is cleared");
    assert_eq!(
        messages[1].content_as_text(),
        "[cleared]",
        "excluded tool's batch partner still cleared"
    );
    assert_eq!(
        messages[2].content_as_text(),
        "older search output ".repeat(30),
        "excluded tool result untouched"
    );
    assert_eq!(
        messages[4].content_as_text(),
        "newer fetch output a ".repeat(30),
        "kept batch untouched"
    );
    assert_eq!(
        messages[5].content_as_text(),
        "newer fetch output b ".repeat(30),
        "kept batch untouched"
    );
    // The ids and the tool_calls arrays survive every replacement.
    assert_eq!(
        messages[1].extra.get("tool_call_id").and_then(|v| v.as_str()),
        Some("call-0a")
    );
    assert_eq!(tool_call_ids(&messages[0]), vec!["call-0a", "call-0b"]);
    assert_eq!(tool_call_ids(&messages[3]), vec!["call-1a", "call-1b"]);
    assert_tool_structure_well_formed(&messages);
}
}
