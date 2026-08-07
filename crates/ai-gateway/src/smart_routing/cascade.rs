//! Failure-signal evaluation and bounded tier escalation for smart routing.

use crate::models::openai::{OpenAIRequest, OpenAIResponse};
use crate::smart_routing::config::CascadeConfig;
use crate::smart_routing::tier::SmartRoutingTier;

const LONG_INPUT_TOKEN_THRESHOLD: u32 = 500;
const REFUSAL_PREFIXES: &[&str] = &["I cannot", "I'm sorry, I can't"];
pub const SMART_ROUTE_ESCALATION_REQUIRED: &str = "smart_route_escalation_required";

/// A typed terminal event emitted when escalation is requested after output
/// has already been committed to the client.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum EarlySignalTerminalEvent {
    SmartRouteEscalationRequired,
}

impl EarlySignalTerminalEvent {
    /// Stable event code for stream protocol adapters.
    pub const fn code(self) -> &'static str {
        match self {
            Self::SmartRouteEscalationRequired => SMART_ROUTE_ESCALATION_REQUIRED,
        }
    }
}

/// The action selected by [`EarlySignalGate`] for a stream-control input.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum EarlySignalGateAction {
    Buffering,
    Commit,
    PassThrough,
    EscalateTransparently,
    Terminal(EarlySignalTerminalEvent),
}

/// Pure stream-control state for early-signal cascade handling.
///
/// The gate counts token units until its configured limit, then permanently
/// commits the source stream. Escalation before that point is transparent;
/// escalation afterwards can only produce a typed terminal event. Replacement
/// content is intentionally not representable by this state machine.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct EarlySignalGate {
    max_buffered_token_units: u32,
    buffered_token_units: u32,
    committed: bool,
    source_cancellation_requested: bool,
}

impl EarlySignalGate {
    pub fn new(max_buffered_token_units: u32) -> Self {
        Self {
            max_buffered_token_units,
            buffered_token_units: 0,
            committed: max_buffered_token_units == 0,
            source_cancellation_requested: false,
        }
    }

    /// Account for source token units and select buffering or relay behavior.
    pub fn observe_token_units(&mut self, token_units: u32) -> EarlySignalGateAction {
        if self.committed {
            return EarlySignalGateAction::PassThrough;
        }

        self.buffered_token_units = self
            .buffered_token_units
            .saturating_add(token_units)
            .min(self.max_buffered_token_units);
        if self.buffered_token_units >= self.max_buffered_token_units {
            self.committed = true;
            EarlySignalGateAction::Commit
        } else {
            EarlySignalGateAction::Buffering
        }
    }

    /// Handle a failure signal and record that the active source must be cancelled.
    pub fn request_escalation(&mut self) -> EarlySignalGateAction {
        self.source_cancellation_requested = true;
        if self.committed {
            EarlySignalGateAction::Terminal(EarlySignalTerminalEvent::SmartRouteEscalationRequired)
        } else {
            EarlySignalGateAction::EscalateTransparently
        }
    }

    pub const fn buffered_token_units(&self) -> u32 {
        self.buffered_token_units
    }

    pub const fn is_committed(&self) -> bool {
        self.committed
    }

    pub const fn source_cancellation_requested(&self) -> bool {
        self.source_cancellation_requested
    }
}

/// A typed reason why a completed cascade attempt should be escalated.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum FailureSignal {
    /// The provider returned an HTTP error after its configured retries ended.
    HttpErrorAfterRetries { status: u16 },
    /// No choice contained text or a tool call.
    EmptyResponse,
    /// Every non-empty textual choice began with a recognized refusal prefix.
    Refusal,
    /// The request supplied tools but no response choice called one.
    MissingToolCalls,
    /// A long request received an unexpectedly small textual response.
    ShortResponse {
        input_tokens: u32,
        response_tokens: u32,
    },
}

/// The terminal result of one provider/tier attempt.
///
/// Live retry errors are intentionally not representable. Callers construct the
/// error variant only after the existing provider retry loop is exhausted.
#[derive(Debug, Clone, Copy)]
pub enum CascadeAttemptResult<'a> {
    Response(&'a OpenAIResponse),
    ExhaustedProviderError { http_status: u16 },
}

/// Inputs required to evaluate a completed cascade attempt.
#[derive(Debug, Clone, Copy)]
pub struct CascadeEvaluationInput<'a> {
    pub request: &'a OpenAIRequest,
    pub result: CascadeAttemptResult<'a>,
}

impl<'a> CascadeEvaluationInput<'a> {
    pub fn response(request: &'a OpenAIRequest, response: &'a OpenAIResponse) -> Self {
        Self {
            request,
            result: CascadeAttemptResult::Response(response),
        }
    }

    pub fn exhausted_provider_error(request: &'a OpenAIRequest, http_status: u16) -> Self {
        Self {
            request,
            result: CascadeAttemptResult::ExhaustedProviderError { http_status },
        }
    }
}

/// Stateless response evaluator used by cascade orchestration.
#[derive(Debug, Default, Clone, Copy)]
pub struct CascadeEvaluator;

impl CascadeEvaluator {
    /// Return the first conservative failure signal for a completed attempt.
    pub fn is_failure_signal(
        &self,
        input: CascadeEvaluationInput<'_>,
        config: &CascadeConfig,
    ) -> Option<FailureSignal> {
        let response = match input.result {
            CascadeAttemptResult::Response(response) => response,
            CascadeAttemptResult::ExhaustedProviderError { http_status } => {
                return Some(FailureSignal::HttpErrorAfterRetries {
                    status: http_status,
                });
            }
        };

        let has_tool_calls = response.choices.iter().any(|choice| {
            choice
                .message
                .extra
                .get("tool_calls")
                .and_then(|value| value.as_array())
                .map_or(false, |calls| !calls.is_empty())
        });
        let contents: Vec<String> = response
            .choices
            .iter()
            .map(|choice| choice.message.content_as_text())
            .collect();
        let non_empty_contents: Vec<&str> = contents
            .iter()
            .map(String::as_str)
            .filter(|content| !content.is_empty())
            .collect();

        if response.choices.is_empty() || (non_empty_contents.is_empty() && !has_tool_calls) {
            return Some(FailureSignal::EmptyResponse);
        }

        if !non_empty_contents.is_empty()
            && non_empty_contents
                .iter()
                .all(|content| has_refusal_prefix(content))
        {
            return Some(FailureSignal::Refusal);
        }

        if request_has_tools(input.request) && !has_tool_calls {
            return Some(FailureSignal::MissingToolCalls);
        }

        if !has_tool_calls {
            let input_tokens = input_token_count(input.request, response);
            let response_tokens = response_token_count(response, &contents);
            if input_tokens > LONG_INPUT_TOKEN_THRESHOLD
                && response_tokens < config.min_response_tokens
            {
                return Some(FailureSignal::ShortResponse {
                    input_tokens,
                    response_tokens,
                });
            }
        }

        None
    }

    /// Return exactly one higher tier while respecting both escalation bounds.
    pub fn next_tier(
        current: SmartRoutingTier,
        escalations: u8,
        max: u8,
    ) -> Option<SmartRoutingTier> {
        let effective_max = max.min(2);
        if escalations >= effective_max {
            return None;
        }

        current.escalate()
    }
}

fn request_has_tools(request: &OpenAIRequest) -> bool {
    request
        .extra
        .get("tools")
        .and_then(|value| value.as_array())
        .map_or(false, |tools| !tools.is_empty())
}

fn has_refusal_prefix(content: &str) -> bool {
    REFUSAL_PREFIXES.iter().any(|prefix| {
        content
            .get(..prefix.len())
            .map_or(false, |candidate| candidate.eq_ignore_ascii_case(prefix))
    })
}

fn input_token_count(request: &OpenAIRequest, response: &OpenAIResponse) -> u32 {
    if response.usage.prompt_tokens > 0 {
        response.usage.prompt_tokens
    } else {
        estimate_tokens(
            request
                .messages
                .iter()
                .map(|message| message.content_as_text().chars().count()),
        )
    }
}

fn response_token_count(response: &OpenAIResponse, contents: &[String]) -> u32 {
    if response.usage.completion_tokens > 0 {
        response.usage.completion_tokens
    } else {
        estimate_tokens(contents.iter().map(|content| content.chars().count()))
    }
}

fn estimate_tokens(character_counts: impl Iterator<Item = usize>) -> u32 {
    let characters = character_counts.fold(0usize, usize::saturating_add);
    u32::try_from(characters / 4).unwrap_or(u32::MAX)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::models::openai::{Choice, Message, Usage};
    use proptest::prelude::*;
    use serde_json::{json, Map, Value};

    fn request(content: &str) -> OpenAIRequest {
        OpenAIRequest {
            model: "logical-model".to_string(),
            messages: vec![message("user", content)],
            stream: false,
            temperature: None,
            max_tokens: None,
            extra: Map::new(),
        }
    }

    fn response(contents: &[&str]) -> OpenAIResponse {
        OpenAIResponse {
            id: String::new(),
            object: String::new(),
            created: 0,
            model: "provider-model".to_string(),
            choices: contents
                .iter()
                .enumerate()
                .map(|(index, content)| Choice {
                    index: index as u32,
                    message: message("assistant", content),
                    finish_reason: Some("stop".to_string()),
                    extra: Map::new(),
                })
                .collect(),
            usage: Usage::default(),
            extra: Map::new(),
        }
    }

    fn message(role: &str, content: &str) -> Message {
        Message {
            role: role.to_string(),
            content: Value::String(content.to_string()),
            extra: Map::new(),
        }
    }

    fn config() -> CascadeConfig {
        CascadeConfig {
            enabled: true,
            max_escalations: 2,
            min_response_tokens: 10,
            early_signal_tokens: 50,
        }
    }

    #[test]
    fn exhausted_provider_error_is_a_terminal_failure_signal() {
        let request = request("hello");

        assert_eq!(
            CascadeEvaluator.is_failure_signal(
                CascadeEvaluationInput::exhausted_provider_error(&request, 503),
                &config(),
            ),
            Some(FailureSignal::HttpErrorAfterRetries { status: 503 })
        );
    }

    #[test]
    fn empty_choices_and_empty_content_are_failures() {
        let request = request("hello");
        let no_choices = response(&[]);
        let empty_choices = response(&["", ""]);

        assert_eq!(
            CascadeEvaluator.is_failure_signal(
                CascadeEvaluationInput::response(&request, &no_choices),
                &config(),
            ),
            Some(FailureSignal::EmptyResponse)
        );
        assert_eq!(
            CascadeEvaluator.is_failure_signal(
                CascadeEvaluationInput::response(&request, &empty_choices),
                &config(),
            ),
            Some(FailureSignal::EmptyResponse)
        );
    }

    #[test]
    fn refusal_prefixes_are_exact_case_insensitive_and_conservative() {
        let request = request("hello");
        let refusal = response(&["i CaNnOt provide that", "I'M SORRY, I CAN'T do that"]);
        let mixed = response(&["I cannot provide that", "Here is a useful answer"]);
        let non_prefix = response(&["Perhaps I cannot, but here is an answer"]);

        assert_eq!(
            CascadeEvaluator.is_failure_signal(
                CascadeEvaluationInput::response(&request, &refusal),
                &config(),
            ),
            Some(FailureSignal::Refusal)
        );
        assert_eq!(
            CascadeEvaluator.is_failure_signal(
                CascadeEvaluationInput::response(&request, &mixed),
                &config(),
            ),
            None
        );
        assert_eq!(
            CascadeEvaluator.is_failure_signal(
                CascadeEvaluationInput::response(&request, &non_prefix),
                &config(),
            ),
            None
        );
    }

    #[test]
    fn tool_expectation_and_calls_use_flattened_extra_fields_across_choices() {
        let mut request = request("Use the weather tool");
        request
            .extra
            .insert("tools".to_string(), json!([{"type": "function"}]));
        let missing = response(&["The weather is sunny"]);
        let mut called = response(&["", ""]);
        called.choices[1]
            .message
            .extra
            .insert("tool_calls".to_string(), json!([{"id": "call-1"}]));

        assert_eq!(
            CascadeEvaluator.is_failure_signal(
                CascadeEvaluationInput::response(&request, &missing),
                &config(),
            ),
            Some(FailureSignal::MissingToolCalls)
        );
        assert_eq!(
            CascadeEvaluator.is_failure_signal(
                CascadeEvaluationInput::response(&request, &called),
                &config(),
            ),
            None
        );
    }

    #[test]
    fn short_response_prefers_usage_and_falls_back_to_character_estimates() {
        let request = request(&"x".repeat(2_004));
        let estimated_short = response(&["12345678901234567890"]);
        let mut usage_short = response(&[&"long answer ".repeat(100)]);
        usage_short.usage.prompt_tokens = 501;
        usage_short.usage.completion_tokens = 9;

        assert_eq!(
            CascadeEvaluator.is_failure_signal(
                CascadeEvaluationInput::response(&request, &estimated_short),
                &config(),
            ),
            Some(FailureSignal::ShortResponse {
                input_tokens: 501,
                response_tokens: 5,
            })
        );
        assert_eq!(
            CascadeEvaluator.is_failure_signal(
                CascadeEvaluationInput::response(&request, &usage_short),
                &config(),
            ),
            Some(FailureSignal::ShortResponse {
                input_tokens: 501,
                response_tokens: 9,
            })
        );
    }

    #[test]
    fn short_response_requires_strictly_more_than_500_input_tokens() {
        let request = request(&"x".repeat(2_000));
        let response = response(&["short"]);

        assert_eq!(
            CascadeEvaluator.is_failure_signal(
                CascadeEvaluationInput::response(&request, &response),
                &config(),
            ),
            None
        );
    }

    #[test]
    fn response_estimate_counts_content_across_choices() {
        let request = request(&"x".repeat(2_004));
        let response = response(&["12345678901234567890", "12345678901234567890"]);

        assert_eq!(
            CascadeEvaluator.is_failure_signal(
                CascadeEvaluationInput::response(&request, &response),
                &config(),
            ),
            None
        );
    }

    #[test]
    fn next_tier_obeys_configured_hard_and_physical_bounds() {
        assert_eq!(
            CascadeEvaluator::next_tier(SmartRoutingTier::Fast, 0, 2),
            Some(SmartRoutingTier::Balanced)
        );
        assert_eq!(
            CascadeEvaluator::next_tier(SmartRoutingTier::Balanced, 1, u8::MAX),
            Some(SmartRoutingTier::Powerful)
        );
        assert_eq!(
            CascadeEvaluator::next_tier(SmartRoutingTier::Fast, 1, 1),
            None
        );
        assert_eq!(
            CascadeEvaluator::next_tier(SmartRoutingTier::Fast, 2, u8::MAX),
            None
        );
        assert_eq!(
            CascadeEvaluator::next_tier(SmartRoutingTier::Powerful, 0, 2),
            None
        );
        assert_eq!(
            CascadeEvaluator::next_tier(SmartRoutingTier::Fast, 0, 0),
            None
        );
    }

    fn tier_from_index(index: u8) -> SmartRoutingTier {
        match index % 3 {
            0 => SmartRoutingTier::Fast,
            1 => SmartRoutingTier::Balanced,
            _ => SmartRoutingTier::Powerful,
        }
    }

    fn tier_index(tier: SmartRoutingTier) -> u8 {
        match tier {
            SmartRoutingTier::Fast => 0,
            SmartRoutingTier::Balanced => 1,
            SmartRoutingTier::Powerful => 2,
        }
    }

    fn escalation_count(start: SmartRoutingTier, configured_max: u8) -> u8 {
        let mut tier = start;
        let mut escalations = 0;
        while let Some(next) = CascadeEvaluator::next_tier(tier, escalations, configured_max) {
            tier = next;
            escalations = escalations.saturating_add(1);
        }
        escalations
    }

    fn next_tier_if_enabled(
        enabled: bool,
        current: SmartRoutingTier,
        escalations: u8,
        configured_max: u8,
    ) -> Option<SmartRoutingTier> {
        enabled
            .then(|| CascadeEvaluator::next_tier(current, escalations, configured_max))
            .flatten()
    }

    proptest! {
            #![proptest_config(ProptestConfig::with_cases(256))]

            // Feature: smart-routing, Property 11: Cascade escalation count never exceeds two.
            #[test]
            fn property_11_max_escalation_is_at_most_two(
            start_index in any::<u8>(),
            configured_max in any::<u8>(),
            ) {
            let count = escalation_count(tier_from_index(start_index), configured_max);
            prop_assert!(count <= 2);
            prop_assert!(count <= configured_max);
            }

            // Feature: smart-routing, Property 12: Disabled cascade performs zero escalations.
            #[test]
            fn property_12_disabled_cascade_has_zero_escalation(
            start_index in any::<u8>(),
            escalations in any::<u8>(),
            configured_max in any::<u8>(),
            ) {
            prop_assert_eq!(
    next_tier_if_enabled(
    false,
            tier_from_index(start_index),
            escalations,
            configured_max,
            ),
            None,
            );
            }

            // Feature: smart-routing, Property 13: Only the defined completed-attempt patterns fail.
            #[test]
            fn property_13_defined_failure_patterns(
            pattern in 0u8..=5,
            http_status in any::<u16>(),
            ) {
            let mut request = request("hello");
        let mut candidate_response = response(&["useful answer with enough detail"]);
        let actual = match pattern {
        0 => CascadeEvaluator.is_failure_signal(
        CascadeEvaluationInput::exhausted_provider_error(&request, http_status),
        &config(),
        ),
        1 => {
        candidate_response.choices.clear();
        CascadeEvaluator.is_failure_signal(
        CascadeEvaluationInput::response(&request, &candidate_response),
        &config(),
        )
        }
        2 => {
        candidate_response = response(&["I cannot comply"]);
        CascadeEvaluator.is_failure_signal(
        CascadeEvaluationInput::response(&request, &candidate_response),
        &config(),
        )
        }
        3 => {
        request.extra.insert("tools".to_string(), json!([{"type": "function"}]));
        CascadeEvaluator.is_failure_signal(
        CascadeEvaluationInput::response(&request, &candidate_response),
        &config(),
        )
        }
        4 => {
        request.messages[0].content = Value::String("x".repeat(2_004));
        candidate_response = response(&["short"]);
        CascadeEvaluator.is_failure_signal(
        CascadeEvaluationInput::response(&request, &candidate_response),
        &config(),
        )
        }
        _ => CascadeEvaluator.is_failure_signal(
        CascadeEvaluationInput::response(&request, &candidate_response),
        &config(),
        ),
        };

            match pattern {
            0 => prop_assert_eq!(actual, Some(FailureSignal::HttpErrorAfterRetries { status: http_status })),
            1 => prop_assert_eq!(actual, Some(FailureSignal::EmptyResponse)),
            2 => prop_assert_eq!(actual, Some(FailureSignal::Refusal)),
            3 => prop_assert_eq!(actual, Some(FailureSignal::MissingToolCalls)),
            4 => prop_assert_eq!(
            actual,
            Some(FailureSignal::ShortResponse {
            input_tokens: 501,
            response_tokens: 1,
            }),
            ),
            _ => prop_assert_eq!(actual, None),
            }
            }

            // Feature: smart-routing, Property 14: Every escalation moves exactly one tier upward.
            #[test]
            fn property_14_escalation_is_exactly_one_tier_upward(
            start_index in any::<u8>(),
            escalations in any::<u8>(),
            configured_max in any::<u8>(),
            ) {
            let current = tier_from_index(start_index);
            if let Some(next) = CascadeEvaluator::next_tier(current, escalations, configured_max) {
            prop_assert_eq!(tier_index(next), tier_index(current) + 1);
            }
            }

            // Feature: smart-routing, Property 33: Early-signal commitment is irreversible.
            #[test]
            fn property_33_early_signal_gate_controls_commit_boundary(
            limit in 1u32..=65_536,
            first_units in any::<u32>(),
            overflow_units in any::<u32>(),
            ) {
            let precommit_units = first_units % limit;
            let mut precommit_gate = EarlySignalGate::new(limit);
            prop_assert_eq!(
            precommit_gate.observe_token_units(precommit_units),
            EarlySignalGateAction::Buffering,
            );
            prop_assert!(precommit_gate.buffered_token_units() <= limit);
            prop_assert!(!precommit_gate.is_committed());
            prop_assert_eq!(
            precommit_gate.request_escalation(),
            EarlySignalGateAction::EscalateTransparently,
            );
            prop_assert!(precommit_gate.source_cancellation_requested());

            let mut committed_gate = EarlySignalGate::new(limit);
            prop_assert_eq!(
            committed_gate.observe_token_units(limit.saturating_add(overflow_units)),
            EarlySignalGateAction::Commit,
            );
            prop_assert_eq!(committed_gate.buffered_token_units(), limit);
            prop_assert!(committed_gate.is_committed());
            prop_assert_eq!(
            committed_gate.request_escalation(),
            EarlySignalGateAction::Terminal(
            EarlySignalTerminalEvent::SmartRouteEscalationRequired,
            ),
            );
            prop_assert!(committed_gate.source_cancellation_requested());
            prop_assert_eq!(
            EarlySignalTerminalEvent::SmartRouteEscalationRequired.code(),
            "smart_route_escalation_required",
            );
            }
            }
}
