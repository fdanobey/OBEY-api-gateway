//! Proactive context-overflow enforcement primitives.
//!
//! [`ContextBudgetGate`] mirrors the reactive `ContextManager` budget math
//! (`input_token_budget`) so pre-dispatch checks and reactive truncation
//! agree on the effective input budget for every attempt.

use crate::compression::token_counter::TokenCounter;
use crate::config::ContextConfig;
use crate::models::openai::OpenAIRequest;

use super::manager::ContextManager;
use super::strategies::{apply_truncation_strategy, clear_tool_results, TruncationStrategy};

/// Policy selected when an outgoing request exceeds a model's input budget.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ContextOverflowPolicy {
    /// Return a 413 context-capacity error without dispatching.
    Reject,
    /// Fail over to a candidate with a larger context window.
    RouteLargerContext,
    /// Reduce the request copy so it fits before dispatch.
    Reduce,
    /// Try larger-context candidates first; reduce once none remain.
    RouteThenReduce,
}

impl ContextOverflowPolicy {
    /// Parses the configured `context.overflow_policy` value, falling back
    /// to `Reduce` (with a warning) on unknown input.
    pub fn from_config(value: &str) -> Self {
        match value {
            "reject" => Self::Reject,
            "route_larger_context" => Self::RouteLargerContext,
            "reduce" => Self::Reduce,
            "route_then_reduce" => Self::RouteThenReduce,
            other => {
                tracing::warn!(
                    value = other,
                    "invalid context.overflow_policy, falling back to reduce"
                );
                Self::Reduce
            }
        }
    }
}

/// Strategy applied when the overflow policy calls for reduction.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ContextReductionStrategy {
    /// Remove oldest message groups until the request fits.
    RemoveOldest,
    /// Keep only the most recent messages (sliding window).
    SlidingWindow,
    /// Clear old tool results first, then remove oldest groups if needed.
    ClearToolsThenRemove,
    /// Summarize older history, then trim; the async summary call is
    /// orchestrated by the caller, with these sync stages as its fallback.
    SummarizeThenTrim,
}

impl ContextReductionStrategy {
    /// Parses the configured `context.reduction_strategy` value, falling
    /// back to `RemoveOldest` (with a warning) on unknown input.
    pub fn from_config(value: &str) -> Self {
        match value {
            "remove_oldest" => Self::RemoveOldest,
            "sliding_window" => Self::SlidingWindow,
            "clear_tools_then_remove" => Self::ClearToolsThenRemove,
            "summarize_then_trim" => Self::SummarizeThenTrim,
            other => {
                tracing::warn!(
                    value = other,
                    "invalid context.reduction_strategy, falling back to remove_oldest"
                );
                Self::RemoveOldest
            }
        }
    }
}

/// Result of comparing an outgoing request against its input budget.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct BudgetCheck {
    /// Estimated tokens for the complete outgoing request.
    pub estimate: u32,
    /// Effective input budget (context window minus output reservation and
    /// safety margin).
    pub budget: u32,
    /// Whether the estimate is within the budget.
    pub fits: bool,
    /// The output reservation subtracted from the window
    /// (`request.max_tokens`), exposed so callers can report a
    /// total-requirement figure consistent with the smart-routing capacity
    /// filter (prompt + reserved output + margin).
    pub reserved_output_tokens: u32,
    /// The safety margin subtracted from the window
    /// (`max(context_window/20, 1024)`), same purpose as above.
    pub safety_margin_tokens: u32,
}

/// Knobs for a single reduction pipeline run.
#[derive(Debug, Clone)]
pub struct ReductionOptions<'a> {
    pub strategy: ContextReductionStrategy,
    pub budget: u32,
    pub sliding_window_size: usize,
    pub tool_result_keep_recent: usize,
    pub tool_result_exclude_tools: &'a [String],
    pub tool_result_placeholder: &'a str,
}

impl<'a> ReductionOptions<'a> {
    /// Builds reduction knobs from `ContextConfig` plus the target budget.
    pub fn from_config(
        strategy: ContextReductionStrategy,
        budget: u32,
        config: &'a ContextConfig,
    ) -> Self {
        Self {
            strategy,
            budget,
            sliding_window_size: config.sliding_window_size,
            tool_result_keep_recent: config.tool_result_keep_recent,
            tool_result_exclude_tools: &config.tool_result_exclude_tools,
            tool_result_placeholder: &config.tool_result_placeholder,
        }
    }
}

/// Outcome of one reduction pipeline run; consumed by the router and logging.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ReductionOutcome {
    pub strategy_applied: ContextReductionStrategy,
    pub original_tokens: u32,
    pub final_tokens: u32,
    pub budget: u32,
    pub groups_removed: usize,
    pub tool_results_cleared: usize,
    pub summarized: bool,
    pub fallback_from_summarization: bool,
    pub fits: bool,
}

/// Pure, sync pre-dispatch context budget gate.
pub struct ContextBudgetGate;

impl ContextBudgetGate {
    /// Effective input budget: context window minus reserved output tokens
    /// minus a safety margin — mirrors `ContextManager`'s
    /// `input_token_budget` math.
    pub fn input_budget(context_window: u32, max_tokens: u32) -> u32 {
        let safety_margin = (context_window / 20).max(1_024);
        context_window
            .saturating_sub(max_tokens)
            .saturating_sub(safety_margin)
    }

    /// Counts the complete outgoing request (messages, tools, extras, reply
    /// priming) and compares it against the effective input budget.
    pub fn check(request: &OpenAIRequest, context_window: u32, max_tokens: u32) -> BudgetCheck {
        let safety_margin_tokens = (context_window / 20).max(1_024);
        let budget = Self::input_budget(context_window, max_tokens);
        let estimate = TokenCounter::new().count_request(request);
        BudgetCheck {
            estimate,
            budget,
            fits: estimate <= budget,
            reserved_output_tokens: max_tokens,
            safety_margin_tokens,
        }
    }

    /// Applies the reduction stages to a mutable copy of the request:
    /// [sliding window?] / [clear tool results?] → remove-oldest-to-budget,
    /// with a single recount after each stage, stopping at the first stage
    /// that fits. `SummarizeThenTrim` runs the `ClearToolsThenRemove` sync
    /// stages here; the async summary call (and its
    /// `fallback_from_summarization` flag) is orchestrated by the caller,
    /// which may also set `summarized` on the returned outcome.
    pub fn reduce(
        request: &mut OpenAIRequest,
        options: &ReductionOptions<'_>,
    ) -> ReductionOutcome {
        let counter = TokenCounter::new();
        let original_tokens = counter.count_request(request);
        let mut outcome = ReductionOutcome {
            strategy_applied: options.strategy,
            original_tokens,
            final_tokens: original_tokens,
            budget: options.budget,
            groups_removed: 0,
            tool_results_cleared: 0,
            summarized: false,
            fallback_from_summarization: false,
            fits: original_tokens <= options.budget,
        };
        if outcome.fits {
            return outcome;
        }

        let non_message_tokens = original_tokens
            .saturating_sub(ContextManager::estimate_tokens(&request.messages));
        let message_budget = options.budget.saturating_sub(non_message_tokens);

        match options.strategy {
            ContextReductionStrategy::SlidingWindow => {
                let removed = apply_truncation_strategy(
                    &mut request.messages,
                    message_budget,
                    TruncationStrategy::SlidingWindow {
                        window_size: options.sliding_window_size,
                    },
                    ContextManager::estimate_tokens,
                );
                outcome.groups_removed += removed.messages_removed;
                outcome.final_tokens = counter.count_request(request);
                outcome.fits = outcome.final_tokens <= options.budget;
            }
            ContextReductionStrategy::ClearToolsThenRemove
            | ContextReductionStrategy::SummarizeThenTrim => {
                outcome.tool_results_cleared = clear_tool_results(
                    &mut request.messages,
                    options.tool_result_keep_recent,
                    options.tool_result_exclude_tools,
                    options.tool_result_placeholder,
                );
                outcome.final_tokens = counter.count_request(request);
                outcome.fits = outcome.final_tokens <= options.budget;
            }
            ContextReductionStrategy::RemoveOldest => {}
        }

        if !outcome.fits {
            let removed = apply_truncation_strategy(
                &mut request.messages,
                message_budget,
                TruncationStrategy::RemoveOldest,
                ContextManager::estimate_tokens,
            );
            outcome.groups_removed += removed.messages_removed;
            outcome.final_tokens = counter.count_request(request);
            outcome.fits = outcome.final_tokens <= options.budget;
        }

        outcome
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::models::openai::Message;
    use serde_json::Map;

    fn message(role: &str, content: &str) -> Message {
        Message {
            role: role.to_string(),
            content: serde_json::Value::String(content.to_string()),
            extra: Map::new(),
        }
    }

    fn assistant_tool_call(id: &str, name: &str) -> Message {
        Message {
            role: "assistant".to_string(),
            content: serde_json::Value::Null,
            extra: {
                let mut extra = Map::new();
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

    fn tool_result(id: &str, content: &str) -> Message {
        Message {
            role: "tool".to_string(),
            content: serde_json::Value::String(content.to_string()),
            extra: {
                let mut extra = Map::new();
                extra.insert(
                    "tool_call_id".to_string(),
                    serde_json::Value::String(id.to_string()),
                );
                extra
            },
        }
    }

    fn request(messages: Vec<Message>) -> OpenAIRequest {
        OpenAIRequest {
            model: "gpt-4o-mini".to_string(),
            messages,
            stream: false,
            temperature: None,
            max_tokens: None,
            extra: Map::new(),
        }
    }

    fn options(strategy: ContextReductionStrategy, budget: u32) -> ReductionOptions<'static> {
        ReductionOptions {
            strategy,
            budget,
            sliding_window_size: 3,
            tool_result_keep_recent: 1,
            tool_result_exclude_tools: &[],
            tool_result_placeholder: "[cleared]",
        }
    }

    fn assert_no_orphaned_tool_results(messages: &[Message]) {
        for (index, msg) in messages.iter().enumerate() {
            if msg.role != "tool" {
                continue;
            }
            let Some(id) = msg.extra.get("tool_call_id").and_then(|v| v.as_str()) else {
                panic!("tool result at {index} missing tool_call_id");
            };
            let paired = messages[..index].iter().any(|m| {
                m.role == "assistant"
                    && m.extra
                        .get("tool_calls")
                        .and_then(|v| v.as_array())
                        .is_some_and(|calls| {
                            calls
                                .iter()
                                .any(|c| c.get("id").and_then(|v| v.as_str()) == Some(id))
                        })
            });
            assert!(paired, "orphaned tool result {id} at index {index}");
        }
    }

    #[test]
    fn budget_math_mirrors_input_token_budget() {
        for (window, max_tokens) in [
            (200_000u32, 8_192u32),
            (100_000, 0),
            (32_768, 4_096),
            (20_480, 1_024),
            (8_192, 512),
            (2_048, 0),
            (512, 100),
            (0, 0),
        ] {
            let expected = window
                .saturating_sub(max_tokens)
                .saturating_sub((window / 20).max(1_024));
            assert_eq!(
                ContextBudgetGate::input_budget(window, max_tokens),
                expected,
                "window={window} max_tokens={max_tokens}"
            );
        }
    }

    #[test]
    fn check_matches_manager_fits_within_limits() {
        let mut req = request(vec![message("user", &"hello world ".repeat(50))]);
        req.max_tokens = Some(1_000);
        let manager = ContextManager::new();

        let fitting = ContextBudgetGate::check(&req, 60_000, req.max_tokens.unwrap_or(0));
        assert!(fitting.fits);
        assert_eq!(
            fitting.estimate,
            ContextManager::estimate_request_tokens(&req)
        );
        assert_eq!(fitting.fits, manager.fits_within_limits(&req, 60_000));

        req.max_tokens = Some(59_000);
        let over = ContextBudgetGate::check(&req, 60_000, req.max_tokens.unwrap_or(0));
        assert!(!over.fits);
        assert_eq!(over.fits, manager.fits_within_limits(&req, 60_000));
    }

    #[test]
    fn check_fits_and_over_boundary() {
        let req = request(vec![message("user", "Hello there")]);
        let estimate = TokenCounter::new().count_request(&req);
        assert!(estimate < 19_456);
        let window = 20_480u32;

        let at_boundary = ContextBudgetGate::check(&req, window, 19_456 - estimate);
        assert_eq!(at_boundary.budget, estimate);
        assert!(at_boundary.fits);

        let over_boundary = ContextBudgetGate::check(&req, window, 19_457 - estimate);
        assert_eq!(over_boundary.budget, estimate - 1);
        assert!(!over_boundary.fits);
    }

    #[test]
    fn overflow_policy_parses_known_values_and_falls_back() {
        assert_eq!(
            ContextOverflowPolicy::from_config("reject"),
            ContextOverflowPolicy::Reject
        );
        assert_eq!(
            ContextOverflowPolicy::from_config("route_larger_context"),
            ContextOverflowPolicy::RouteLargerContext
        );
        assert_eq!(
            ContextOverflowPolicy::from_config("reduce"),
            ContextOverflowPolicy::Reduce
        );
        assert_eq!(
            ContextOverflowPolicy::from_config("route_then_reduce"),
            ContextOverflowPolicy::RouteThenReduce
        );
        assert_eq!(
            ContextOverflowPolicy::from_config("bogus"),
            ContextOverflowPolicy::Reduce
        );
        assert_eq!(ContextOverflowPolicy::from_config(""), ContextOverflowPolicy::Reduce);
    }

    #[test]
    fn reduction_strategy_parses_known_values_and_falls_back() {
        assert_eq!(
            ContextReductionStrategy::from_config("remove_oldest"),
            ContextReductionStrategy::RemoveOldest
        );
        assert_eq!(
            ContextReductionStrategy::from_config("sliding_window"),
            ContextReductionStrategy::SlidingWindow
        );
        assert_eq!(
            ContextReductionStrategy::from_config("clear_tools_then_remove"),
            ContextReductionStrategy::ClearToolsThenRemove
        );
        assert_eq!(
            ContextReductionStrategy::from_config("summarize_then_trim"),
            ContextReductionStrategy::SummarizeThenTrim
        );
        assert_eq!(
            ContextReductionStrategy::from_config("bogus"),
            ContextReductionStrategy::RemoveOldest
        );
        assert_eq!(
            ContextReductionStrategy::from_config(""),
            ContextReductionStrategy::RemoveOldest
        );
    }

    #[test]
    fn reduce_already_fitting_request_is_noop() {
        let mut req = request(vec![message("user", "Hello there")]);
        let estimate = TokenCounter::new().count_request(&req);
        let outcome = ContextBudgetGate::reduce(&mut req, &options(ContextReductionStrategy::RemoveOldest, estimate));

        assert!(outcome.fits);
        assert_eq!(outcome.original_tokens, estimate);
        assert_eq!(outcome.final_tokens, estimate);
        assert_eq!(outcome.groups_removed, 0);
        assert_eq!(outcome.tool_results_cleared, 0);
        assert!(!outcome.summarized);
        assert!(!outcome.fallback_from_summarization);
        assert_eq!(req.messages.len(), 1);
    }

    #[test]
    fn reduce_remove_oldest_fits_after_group_removal() {
        let mut messages = vec![message("system", "You are helpful")];
        for i in 0..60 {
            messages.push(message("user", &format!("message number {i} with some padding text")));
        }
        let mut req = request(messages);
        let outcome =
            ContextBudgetGate::reduce(&mut req, &options(ContextReductionStrategy::RemoveOldest, 300));

        assert!(outcome.fits);
        assert!(outcome.original_tokens > 300);
        assert!(outcome.final_tokens <= 300);
        assert_eq!(outcome.strategy_applied, ContextReductionStrategy::RemoveOldest);
        assert!(outcome.groups_removed > 0);
        assert_eq!(outcome.tool_results_cleared, 0);
        assert_eq!(req.messages[0].role, "system");
        assert_no_orphaned_tool_results(&req.messages);
    }

    #[test]
    fn reduce_clear_tools_stage_stops_before_group_removal() {
        let mut messages = vec![message("system", "You are helpful")];
        for i in 0..4 {
            messages.push(assistant_tool_call(&format!("call-{i}"), "fetch"));
            messages.push(tool_result(&format!("call-{i}"), &"x".repeat(2_000)));
        }
        messages.push(message("user", "Final question"));

        let mut dry = request(messages.clone());
        clear_tool_results(&mut dry.messages, 1, &[], "[cleared]");
        let budget = TokenCounter::new().count_request(&dry);
        assert!(budget > 0);

        let mut req = request(messages);
        let original = TokenCounter::new().count_request(&req);
        let outcome = ContextBudgetGate::reduce(
            &mut req,
            &options(ContextReductionStrategy::ClearToolsThenRemove, budget),
        );

        assert!(outcome.fits);
        assert_eq!(outcome.original_tokens, original);
        assert_eq!(outcome.final_tokens, budget);
        assert_eq!(outcome.tool_results_cleared, 3);
        assert_eq!(outcome.groups_removed, 0);
        assert_eq!(req.messages.len(), 10);
        assert_eq!(
            req.messages[8].content_as_text(),
            "x".repeat(2_000),
            "kept recent tool result retains content"
        );
        assert_eq!(req.messages[9].content_as_text(), "Final question");
        assert_no_orphaned_tool_results(&req.messages);
    }

    #[test]
    fn reduce_summarize_then_trim_runs_sync_fallback_stages() {
        let mut messages = vec![message("system", "You are helpful")];
        for i in 0..4 {
            messages.push(assistant_tool_call(&format!("call-{i}"), "fetch"));
            messages.push(tool_result(&format!("call-{i}"), &"x".repeat(2_000)));
        }
        messages.push(message("user", "Final question"));

        let mut dry = request(messages.clone());
        clear_tool_results(&mut dry.messages, 1, &[], "[cleared]");
        let budget = TokenCounter::new().count_request(&dry) - 1;

        let mut req = request(messages);
        let outcome = ContextBudgetGate::reduce(
            &mut req,
            &options(ContextReductionStrategy::SummarizeThenTrim, budget),
        );

        assert!(outcome.fits);
        assert_eq!(
            outcome.strategy_applied,
            ContextReductionStrategy::SummarizeThenTrim
        );
        assert_eq!(outcome.tool_results_cleared, 3);
        assert!(outcome.groups_removed > 0);
        assert!(!outcome.summarized);
        assert!(!outcome.fallback_from_summarization);
        assert_no_orphaned_tool_results(&req.messages);
    }

    #[test]
    fn reduce_sliding_window_stage_stops_before_group_removal() {
        let mut messages = vec![message("system", "You are helpful")];
        for i in 0..60 {
            messages.push(message("user", &format!("message number {i} with some padding text")));
        }
        let mut req = request(messages);
        let original_len = req.messages.len();
        let outcome = ContextBudgetGate::reduce(
            &mut req,
            &options(ContextReductionStrategy::SlidingWindow, 300),
        );

        assert!(outcome.fits);
        assert_eq!(
            outcome.strategy_applied,
            ContextReductionStrategy::SlidingWindow
        );
        assert!(outcome.groups_removed > 0);
        assert_eq!(outcome.tool_results_cleared, 0);
        assert_eq!(req.messages.len(), 4);
        assert!(req.messages.len() < original_len);
        assert_eq!(req.messages[0].role, "system");
    }

    #[test]
    fn reduce_never_orphans_tool_results() {
        let mut messages = vec![message("system", "You are helpful")];
        for i in 0..8 {
            messages.push(assistant_tool_call(&format!("call-{i}"), "fetch"));
            messages.push(tool_result(&format!("call-{i}"), &"result ".repeat(100)));
            messages.push(message("assistant", &format!("assistant turn {i}")));
        }
        messages.push(message("user", "Final question"));

        let mut req = request(messages);
        let outcome =
            ContextBudgetGate::reduce(&mut req, &options(ContextReductionStrategy::RemoveOldest, 200));

        assert!(outcome.fits);
        assert!(outcome.groups_removed > 0);
        assert_no_orphaned_tool_results(&req.messages);
        assert_eq!(req.messages[0].role, "system");
    }
}
