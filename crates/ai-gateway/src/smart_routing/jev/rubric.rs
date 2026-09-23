//! The Jev classification rubric: the fixed set of typed questions sent to the
//! System One endpoint, and the char-bounded state assembled from a request.
//!
//! Six orthogonal Score dimensions follow TypeSafe's composite-scoring
//! guidance: one dimension per question, level descriptions describe
//! situations rather than degrees, and structured `{what, examples}` objects
//! sharpen levels where plain strings split. One Choice question detects the
//! task type. Question ids are stable contracts with `compose`.

use std::collections::HashMap;

use serde_json::{json, Value};

use crate::models::openai::{Message, OpenAIRequest};
use crate::smart_routing::config::JevConfig;

use super::models::Question;

/// Question id for the task-type choice.
pub const TASK_TYPE_QUESTION_ID: &str = "task_type";

/// Stable dimension ids in weighting order (matches `DimensionWeights`).
pub const DIMENSION_QUESTION_IDS: [&str; 6] = [
    "reasoning_depth",
    "tool_coupling",
    "context_synthesis",
    "output_precision",
    "domain_load",
    "ambiguity",
];

/// Number of levels per dimension Score question.
pub const DIMENSION_LEVELS: usize = 4;

/// Build the fixed question map: six Score dimensions plus one task-type
/// Choice. Level texts are constants here so the rubric is reviewable in one
/// place; weights are applied later in `compose`, never here.
pub fn build_questions() -> HashMap<String, Question> {
    let mut questions = HashMap::with_capacity(DIMENSION_QUESTION_IDS.len() + 1);

    questions.insert(
        "reasoning_depth".to_string(),
        Question {
            question_type: "score",
            instructions: json!("How much multi-step reasoning does answering this request require?"),
            criteria: json!([
                { "what": "Single-step recall or trivial transform; the answer follows directly from the request",
                  "examples": ["greeting", "asking for a definition", "reformatting one field"] },
                { "what": "A short chain of two or three straightforward steps",
                  "examples": ["summarizing a short text", "a simple lookup plus one join", "one-file code tweak"] },
                { "what": "Multi-step reasoning with dependencies between steps, or a nontrivial plan",
                  "examples": ["debugging across modules", "comparing several options with trade-offs", "structured analysis"] },
                { "what": "Deep, novel reasoning: proofs, intricate causal chains, or problems with no obvious solution path",
                  "examples": ["formal proof", "novel algorithm design", "complex root-cause analysis under uncertainty"] }
            ]),
        },
    );

    questions.insert(
        "tool_coupling".to_string(),
        Question {
            question_type: "score",
            instructions: json!("How tightly does this request depend on tool calls, function results, or agentic loops?"),
            criteria: json!([
                { "what": "No tools involved; a plain conversational answer suffices",
                  "examples": ["chat reply", "writing a note from memory"] },
                { "what": "One optional tool call could help but is not required",
                  "examples": ["asking for current weather", "a single web lookup"] },
                { "what": "The task needs a few coordinated tool calls whose results feed later steps",
                  "examples": ["run tests then fix the failure", "search then summarize findings"] },
                { "what": "An agentic loop where many tool results drive subsequent decisions and correctness depends on them",
                  "examples": ["multi-file refactor driven by test output", "research task with iterative search"] }
            ]),
        },
    );

    questions.insert(
        "context_synthesis".to_string(),
        Question {
            question_type: "score",
            instructions: json!("How much synthesis across prior conversation context or supplied material does this request need?"),
            criteria: json!([
                { "what": "Self-contained; no prior context or supplied material matters",
                  "examples": ["one-shot question with all facts included"] },
                { "what": "Light use of earlier turns or one short attached text",
                  "examples": ["follow-up referring to the previous message", "summary of one document"] },
                { "what": "Requires reconciling several messages, files, or constraints into one answer",
                  "examples": ["combine notes from three sources", "answer using constraints stated earlier"] },
                { "what": "Heavy synthesis across a long conversation or many artifacts, with contradictions or partial information",
                  "examples": ["resolve conflicting statements across a long thread", "cross-reference many documents"] }
            ]),
        },
    );

    questions.insert(
        "output_precision".to_string(),
        Question {
            question_type: "score",
            instructions: json!("How precise, structured, or constrained does the required output be?"),
            criteria: json!([
                { "what": "Free-form short reply; no format constraints",
                  "examples": ["casual chat", "a quick opinion"] },
                { "what": "A specific format or tone is expected",
                  "examples": ["bullet list", "polite professional email", "fixed length"] },
                { "what": "Strict structure or exact values required: code that must compile, valid JSON, tables",
                  "examples": ["a code patch", "JSON matching a schema", "exact translation"] },
                { "what": "Zero-tolerance output: legal or financial text, formal proofs, safety-critical instructions",
                  "examples": ["contract clause", "medical dosage wording", "verified proof steps"] }
            ]),
        },
    );

    questions.insert(
        "domain_load".to_string(),
        Question {
            question_type: "score",
            instructions: json!("How much specialized domain knowledge — code, math, science, legal, financial — does this request carry or demand?"),
            criteria: json!([
                { "what": "Everyday language and common knowledge",
                  "examples": ["small talk", "general trivia"] },
                { "what": "Some specialized terms but a generalist handles it",
                  "examples": ["explain a technical concept simply", "basic arithmetic"] },
                { "what": "Real domain expertise needed: nontrivial code, applied math, or professional field knowledge",
                  "examples": ["implement a feature in a known framework", "solve an engineering calculation"] },
                { "what": "Expert-level or cross-domain depth: advanced algorithms, formal math, rare specialties",
                  "examples": ["compiler optimization", "statistical modeling", "specialized regulatory analysis"] }
            ]),
        },
    );

    questions.insert(
        "ambiguity".to_string(),
        Question {
            question_type: "score",
            instructions: json!("How ambiguous or underspecified is this request?"),
            criteria: json!([
                { "what": "Fully specified; the intent and expected result are clear",
                  "examples": ["translate this exact sentence", "answer with this data"] },
                { "what": "Minor ambiguity that a reasonable default resolves",
                  "examples": ["summarize this (length unstated)"] },
                { "what": "Materially underspecified; the answer depends on assumptions the model must choose",
                  "examples": ["make it better", "write something about our product"] },
                { "what": "Deeply ambiguous, contradictory, or requires guessing unstated goals",
                  "examples": ["fix it without saying what is broken", "conflicting instructions in one request"] }
            ]),
        },
    );

    questions.insert(
        TASK_TYPE_QUESTION_ID.to_string(),
        Question {
            question_type: "choice",
            instructions: json!("What kind of task is this request, primarily?"),
            criteria: json!({
                "code_generation": "Writing, modifying, debugging, or reviewing source code or configuration",
                "math_reasoning": "Mathematics, quantitative analysis, formal logic, or calculation",
                "creative_writing": "Fiction, marketing copy, style-driven or expressive writing",
                "factual_qa": "Questions about facts, definitions, or knowledge lookup",
                "tool_use": "Orchestrating tool calls, function invocations, or external systems",
                "summarization": "Condensing, rewriting, or restructuring supplied material",
                "general": "Conversation, advice, mixed or unclassified requests"
            }),
        },
    );

    questions
}

/// Assemble the state value sent to the endpoint from a request.
///
/// Reuses the `LlmClassifier` extraction rules: the most recent user message
/// is preferred, tool-result content is skipped, and text is truncated to the
/// configured character budget. Content-free metadata (tool presence, message
/// count, token estimate) gives the rubric signals without sending more text.
pub fn build_state(request: &OpenAIRequest, config: &JevConfig) -> Value {
    let mut budget = config.char_budget;
    let mut excerpts: Vec<Value> = Vec::new();

    let mut selected = select_messages(request);
    selected.reverse();
    for message in selected {
        if budget == 0 {
            break;
        }
        let mut message_text = String::new();
        visit_message_text(message, |text| {
            if budget == 0 {
                return;
            }
            let remaining_chars = text.chars().take(budget).collect::<String>();
            budget -= remaining_chars.chars().count();
            if !message_text.is_empty() {
                message_text.push('\n');
            }
            message_text.push_str(&remaining_chars);
        });
        if !message_text.is_empty() {
            excerpts.push(json!({
                "role": message.role,
                "content": message_text,
            }));
        }
    }

    json!({
        "messages": excerpts,
        "meta": {
            "tools_present": request.messages.iter().any(|message| message.role.eq_ignore_ascii_case("tool")),
            "message_count": request.messages.len(),
        },
    })
}

/// Message selection: the most recent user message when one exists, otherwise
/// the most recent non-tool message. Returns at most one message today; the
/// Vec shape keeps the door open for multi-message states within budget.
fn select_messages(request: &OpenAIRequest) -> Vec<&Message> {
    if let Some(message) = request
        .messages
        .iter()
        .rev()
        .find(|message| message.role.eq_ignore_ascii_case("user"))
    {
        return vec![message];
    }
    request
        .messages
        .iter()
        .rev()
        .find(|message| !message.role.eq_ignore_ascii_case("tool"))
        .map(|message| vec![message])
        .unwrap_or_default()
}

/// Visit string and text-array content, skipping tool results — identical
/// rules to `LlmClassifier::visit_text_content`.
pub fn visit_message_text(message: &Message, mut visit: impl FnMut(&str)) {
    match &message.content {
        Value::String(text) => visit(text),
        Value::Array(parts) => {
            for part in parts {
                if part.get("type").and_then(Value::as_str) == Some("tool_result") {
                    continue;
                }
                if let Some(text) = part.get("text").and_then(Value::as_str) {
                    visit(text);
                }
            }
        }
        _ => {}
    }
}

#[cfg(test)]
mod tests {
    use serde_json::{Map, Value};

    use super::*;

    fn request(messages: Vec<(&str, Value)>) -> OpenAIRequest {
        OpenAIRequest {
            model: "group-model".to_string(),
            messages: messages
                .into_iter()
                .map(|(role, content)| Message {
                    role: role.to_string(),
                    content,
                    extra: Map::new(),
                })
                .collect(),
            stream: false,
            temperature: None,
            max_tokens: None,
            extra: Map::new(),
        }
    }

    fn config() -> JevConfig {
        JevConfig::default()
    }

    #[test]
    fn question_map_shape_is_stable() {
        let questions = build_questions();

        assert_eq!(questions.len(), DIMENSION_QUESTION_IDS.len() + 1);
        for id in DIMENSION_QUESTION_IDS {
            let question = questions.get(id).unwrap_or_else(|| panic!("missing {id}"));
            assert_eq!(question.question_type, "score");
            let criteria = question.criteria.as_array().unwrap();
            assert_eq!(criteria.len(), DIMENSION_LEVELS);
            for level in criteria {
                assert!(level.get("what").is_some(), "level needs a what: {id}");
            }
        }

        let task = questions.get(TASK_TYPE_QUESTION_ID).unwrap();
        assert_eq!(task.question_type, "choice");
        let options = task.criteria.as_object().unwrap();
        assert_eq!(
            options.len(),
            7,
            "must cover every TaskType variant plus general"
        );
        for option in [
            "code_generation",
            "math_reasoning",
            "creative_writing",
            "factual_qa",
            "tool_use",
            "summarization",
            "general",
        ] {
            assert!(options.contains_key(option), "missing option {option}");
        }
    }

    #[test]
    fn state_prefers_last_user_message_and_respects_budget() {
        let long_text = "x".repeat(5_000);
        let request = request(vec![
            ("user", json!("earlier question")),
            ("assistant", json!("earlier answer")),
            ("user", json!(long_text)),
        ]);

        let state = build_state(&request, &config());
        let messages = state["messages"].as_array().unwrap();
        assert_eq!(messages.len(), 1);
        assert_eq!(messages[0]["role"], "user");
        let content = messages[0]["content"].as_str().unwrap();
        assert_eq!(content.chars().count(), config().char_budget);
        assert_eq!(state["meta"]["message_count"], 3);
        assert_eq!(state["meta"]["tools_present"], false);
    }

    #[test]
    fn state_skips_tool_results_and_flags_tool_presence() {
        let request = request(vec![
            (
                "user",
                json!([{"type": "tool_result", "text": "tool output secret"}]),
            ),
            ("tool", json!("tool role content")),
            ("assistant", json!("assistant text")),
            ("user", json!("final question")),
        ]);

        let state = build_state(&request, &config());
        let serialized = serde_json::to_string(&state).unwrap();
        assert!(!serialized.contains("tool output secret"));
        assert!(!serialized.contains("tool role content"));
        assert!(serialized.contains("final question"));
        assert_eq!(state["meta"]["tools_present"], true);
        assert_eq!(state["meta"]["message_count"], 4);
    }

    #[test]
    fn state_falls_back_to_most_recent_non_tool_message() {
        let request = request(vec![
            ("system", json!("system prompt")),
            ("assistant", json!("only an answer exists")),
        ]);

        let state = build_state(&request, &config());
        let messages = state["messages"].as_array().unwrap();
        assert_eq!(messages.len(), 1);
        assert_eq!(messages[0]["role"], "assistant");
    }

    #[test]
    fn state_with_no_messages_is_well_formed() {
        let request = request(vec![]);
        let state = build_state(&request, &config());
        assert_eq!(state["messages"].as_array().unwrap().len(), 0);
        assert_eq!(state["meta"]["message_count"], 0);
        assert_eq!(state["meta"]["tools_present"], false);
    }

    #[test]
    fn small_budget_truncates_without_panicking() {
        let mut small = config();
        small.char_budget = 300;
        let request = request(vec![("user", json!("short but multi paragraph text"))]);

        let state = build_state(&request, &small);
        let messages = state["messages"].as_array().unwrap();
        assert_eq!(messages.len(), 1);
    }
}
