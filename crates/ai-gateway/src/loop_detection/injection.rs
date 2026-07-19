use crate::loop_detection::{
    config::{InjectionStrategy, LoopDetectionConfig},
    enforcement::EnforcementDecision,
    session::SessionState,
};
use crate::models::openai::{Message, OpenAIRequest};
use serde_json::Value;

pub const DEFAULT_BREAK_INSTRUCTION: &str = "IMPORTANT: Loop detected. You are repeating the same actions without progress. Stop the current approach. Summarize what you have tried, identify why it is failing, and propose a fundamentally different strategy.";
pub const ERROR_CYCLING_INSTRUCTION: &str = "IMPORTANT: Loop detected. You are retrying the same operation that keeps failing with the same error. The error will not resolve by retrying. Acknowledge the limitation, explain the error to the user, and suggest alternative approaches.";

pub struct InjectionEngine;

impl InjectionEngine {
    pub fn inject(
        request: &mut OpenAIRequest,
        decision: &EnforcementDecision,
        session: &mut SessionState,
        config: &LoopDetectionConfig,
        custom_template: Option<&str>,
    ) -> bool {
        if !decision.should_inject || !decision.transitioned || session.injected_at_level {
            return false;
        }

        let instruction = select_instruction(request, decision, config, custom_template);
        append_system_instruction(request, &instruction);
        session.injected_at_level = true;
        true
    }
}

fn select_instruction(
    request: &OpenAIRequest,
    decision: &EnforcementDecision,
    config: &LoopDetectionConfig,
    custom_template: Option<&str>,
) -> String {
    if let Some(template) = custom_template.or(config.break_instruction_template.as_deref()) {
        return template.to_string();
    }
    if config.injection_strategy != InjectionStrategy::ContextAware {
        return DEFAULT_BREAK_INSTRUCTION.to_string();
    }

    match decision.dominant_signal {
        "tool_call_repetition" => most_frequent_tool_name(request)
            .map(|tool_name| format!("IMPORTANT: Loop detected. You are calling the same tools repeatedly with similar arguments. The tool calls are not producing new results. Stop calling {tool_name} and try a completely different approach to solve the problem."))
            .unwrap_or_else(|| DEFAULT_BREAK_INSTRUCTION.to_string()),
        "error_cycling" => ERROR_CYCLING_INSTRUCTION.to_string(),
        _ => DEFAULT_BREAK_INSTRUCTION.to_string(),
    }
}

fn append_system_instruction(request: &mut OpenAIRequest, instruction: &str) {
    if let Some(system) = request
        .messages
        .iter_mut()
        .find(|message| message.role == "system")
    {
        match &mut system.content {
            Value::String(content) => {
                if !content.is_empty() {
                    content.push_str("\n\n");
                }
                content.push_str(instruction);
            }
            Value::Array(parts) => parts.push(serde_json::json!({
                "type": "text",
                "text": instruction,
            })),
            Value::Null => system.content = Value::String(instruction.to_string()),
            other => {
                let existing = other.to_string();
                system.content = Value::String(format!("{existing}\n\n{instruction}"));
            }
        }
        return;
    }

    request.messages.insert(
        0,
        Message {
            role: "system".to_string(),
            content: Value::String(instruction.to_string()),
            extra: serde_json::Map::new(),
        },
    );
}

fn most_frequent_tool_name(request: &OpenAIRequest) -> Option<String> {
    let mut counts = std::collections::HashMap::<String, usize>::new();
    for message in &request.messages {
        let Some(tool_calls) = message.extra.get("tool_calls").and_then(Value::as_array) else {
            continue;
        };
        for tool_call in tool_calls {
            if let Some(name) = tool_call
                .get("function")
                .and_then(|function| function.get("name"))
                .and_then(Value::as_str)
            {
                *counts.entry(name.to_string()).or_default() += 1;
            }
        }
    }
    counts
        .into_iter()
        .max_by(|left, right| left.1.cmp(&right.1).then_with(|| right.0.cmp(&left.0)))
        .map(|(name, _)| name)
}
