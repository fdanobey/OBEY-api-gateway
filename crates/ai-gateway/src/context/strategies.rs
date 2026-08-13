//! Context truncation strategies.
//!
//! Different strategies for handling requests that exceed model context windows.

use crate::models::openai::Message;

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
}
