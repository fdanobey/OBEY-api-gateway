//! Deterministic, allocation-conscious heuristic complexity scoring.

use crate::models::openai::Message;
use crate::smart_routing::config::HeuristicWeights;
use crate::smart_routing::tier::{ComplexityScore, TaskType};

const MESSAGE_COUNT_SCALE: f64 = 8.0;
const TOKEN_ESTIMATE_SCALE: f64 = 1_000.0;
const CODE_BLOCK_SCALE: f64 = 3.0;
const TOOL_CALL_SCALE: f64 = 3.0;
const MATH_EXPRESSION_SCALE: f64 = 4.0;
const REASONING_KEYWORD_SCALE: f64 = 6.0;
const FEATURE_COUNT: usize = 6;

const DEFAULT_REASONING_KEYWORDS: &[&str] = &[
    "analyze",
    "analyse",
    "compare",
    "derive",
    "evaluate",
    "explain why",
    "reason",
    "step by step",
    "think through",
    "trade-off",
    "tradeoff",
];

const TOOL_PATTERNS: &[&str] = &[
    "call a tool",
    "call the function",
    "call the tool",
    "function call",
    "use a tool",
    "use the tool",
];
const CODE_PATTERNS: &[&str] = &[
    "debug this code",
    "fix this code",
    "generate code",
    "implement a function",
    "implement the code",
    "refactor this code",
    "write a function",
    "write code",
];
const MATH_PATTERNS: &[&str] = &[
    "calculate the",
    "derive the equation",
    "mathematical proof",
    "prove that",
    "solve the equation",
    "solve this equation",
];
const SUMMARY_PATTERNS: &[&str] = &[
    "condense",
    "provide a summary",
    "summarise",
    "summarize",
    "summarization",
    "summary of",
    "tl;dr",
    "write a summary",
];
const CREATIVE_PATTERNS: &[&str] = &[
    "compose a song",
    "creative writing",
    "write a poem",
    "write a screenplay",
    "write a story",
];
const FACTUAL_PATTERNS: &[&str] = &["answer this question", "fact check", "factual question"];
const QUESTION_WORDS: &[&str] = &["how", "what", "when", "where", "which", "who", "why"];

/// Raw structural features extracted without mutating or retaining request content.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub struct HeuristicFeatures {
    pub message_count: usize,
    pub character_count: usize,
    pub token_estimate: usize,
    pub code_blocks: usize,
    pub tool_calls: usize,
    pub math_expressions: usize,
    pub reasoning_keywords: usize,
}

/// Complexity and task classification produced by the same extraction pass.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct HeuristicAssessment {
    pub score: ComplexityScore,
    pub task_type: TaskType,
    pub features: HeuristicFeatures,
}

impl HeuristicAssessment {
    /// Estimated prompt tokens, exposed for context filtering and orchestration.
    pub fn token_estimate(self) -> usize {
        self.features.token_estimate
    }
}

/// One-pass structural scorer with configurable reasoning keywords.
#[derive(Debug, Clone)]
pub struct HeuristicScorer {
    weights: [f64; FEATURE_COUNT],
    reasoning_keywords: Vec<Vec<u8>>,
}

impl Default for HeuristicScorer {
    fn default() -> Self {
        Self::new(None)
    }
}

impl HeuristicScorer {
    /// Construct a scorer. Omitted weights use the equal `HeuristicWeights` default.
    pub fn new(weights: Option<HeuristicWeights>) -> Self {
        Self::with_reasoning_keywords(weights, DEFAULT_REASONING_KEYWORDS.iter().copied())
    }

    /// Construct a scorer with a request-independent reasoning keyword list.
    pub fn with_reasoning_keywords<I, S>(
        weights: Option<HeuristicWeights>,
        reasoning_keywords: I,
    ) -> Self
    where
        I: IntoIterator<Item = S>,
        S: AsRef<str>,
    {
        let mut normalized_keywords: Vec<Vec<u8>> = Vec::new();
        for keyword in reasoning_keywords {
            let keyword = keyword.as_ref().trim().as_bytes();
            if keyword.is_empty()
                || normalized_keywords
                    .iter()
                    .any(|existing| existing.eq_ignore_ascii_case(keyword))
            {
                continue;
            }
            normalized_keywords.push(keyword.to_vec());
        }

        Self {
            weights: normalized_weights(weights.unwrap_or_default()),
            reasoning_keywords: normalized_keywords,
        }
    }

    /// Extract features, normalize them, and classify the task in one content pass.
    pub fn score(&self, messages: &[Message]) -> HeuristicAssessment {
        let mut extraction = Extraction::default();
        extraction.features.message_count = messages.len();

        for message in messages {
            extraction.scan_message_structure(message);
            visit_text_content(message, |text| {
                extraction.scan_text(text, &self.reasoning_keywords)
            });
        }

        extraction.features.token_estimate = extraction.features.character_count / 4;
        let task_type = extraction.task_type();
        let score = if extraction.has_meaningful_signal {
            let values = [
                sigmoid(extraction.features.message_count, MESSAGE_COUNT_SCALE),
                sigmoid(extraction.features.token_estimate, TOKEN_ESTIMATE_SCALE),
                sigmoid(extraction.features.code_blocks, CODE_BLOCK_SCALE),
                sigmoid(extraction.features.tool_calls, TOOL_CALL_SCALE),
                sigmoid(extraction.features.math_expressions, MATH_EXPRESSION_SCALE),
                sigmoid(
                    extraction.features.reasoning_keywords,
                    REASONING_KEYWORD_SCALE,
                ),
            ];
            let weighted_sum = self
                .weights
                .iter()
                .zip(values)
                .map(|(weight, value)| weight * value)
                .sum::<f64>()
                .clamp(0.0, 1.0);
            ComplexityScore::new(weighted_sum)
        } else {
            ComplexityScore::new(0.0)
        };

        HeuristicAssessment {
            score,
            task_type,
            features: extraction.features,
        }
    }
}

#[derive(Debug, Default)]
struct Extraction {
    features: HeuristicFeatures,
    has_meaningful_signal: bool,
    tool_use: bool,
    code_generation: bool,
    math_reasoning: bool,
    summarization: bool,
    creative_writing: bool,
    factual_qa: bool,
}

impl Extraction {
    fn scan_message_structure(&mut self, message: &Message) {
        let mut message_tool_signals = 0;

        if let Some(value) = message.extra.get("tool_calls") {
            message_tool_signals += structural_item_count(value);
        }
        if message
            .extra
            .get("function_call")
            .is_some_and(is_present_structure)
        {
            message_tool_signals += 1;
        }
        if let Some(value) = message.extra.get("tools") {
            message_tool_signals += structural_item_count(value);
        }
        if message.role.eq_ignore_ascii_case("tool")
            || message
                .extra
                .get("tool_call_id")
                .is_some_and(is_present_structure)
        {
            message_tool_signals = message_tool_signals.max(1);
        }

        if message_tool_signals > 0 {
            self.features.tool_calls = self
                .features
                .tool_calls
                .saturating_add(message_tool_signals);
            self.has_meaningful_signal = true;
            self.tool_use = true;
        }
    }

    fn scan_text(&mut self, text: &str, reasoning_keywords: &[Vec<u8>]) {
        let bytes = text.as_bytes();
        let mut index = 0;
        let mut at_text_start = true;
        let mut code_fence_open = false;
        let mut display_math_open = false;

        while index < bytes.len() {
            let byte = bytes[index];
            if !is_utf8_continuation(byte) {
                self.features.character_count = self.features.character_count.saturating_add(1);
            }
            if !byte.is_ascii_whitespace() {
                self.has_meaningful_signal = true;
            }

            if starts_with(bytes, index, b"```") {
                self.code_generation = true;
                if !code_fence_open {
                    self.features.code_blocks = self.features.code_blocks.saturating_add(1);
                }
                code_fence_open = !code_fence_open;
            }

            if starts_with(bytes, index, b"$$") {
                self.math_reasoning = true;
                if !display_math_open {
                    self.features.math_expressions =
                        self.features.math_expressions.saturating_add(1);
                }
                display_math_open = !display_math_open;
            } else if starts_with(bytes, index, br"\(")
                || starts_with(bytes, index, br"\[")
                || starts_with_ignore_ascii_case(bytes, index, br"\begin{equation")
                || starts_with_ignore_ascii_case(bytes, index, br"\begin{align")
                || starts_with_ignore_ascii_case(bytes, index, br"\begin{gather")
            {
                self.features.math_expressions = self.features.math_expressions.saturating_add(1);
                self.math_reasoning = true;
            }

            if at_text_start && !byte.is_ascii_whitespace() {
                self.factual_qa |= matches_any_pattern(bytes, index, QUESTION_WORDS);
                at_text_start = false;
            }

            self.tool_use |= matches_any_pattern(bytes, index, TOOL_PATTERNS);
            self.code_generation |= matches_any_pattern(bytes, index, CODE_PATTERNS);
            self.math_reasoning |= matches_any_pattern(bytes, index, MATH_PATTERNS);
            self.summarization |= matches_any_pattern(bytes, index, SUMMARY_PATTERNS);
            self.creative_writing |= matches_any_pattern(bytes, index, CREATIVE_PATTERNS);
            self.factual_qa |= matches_any_pattern(bytes, index, FACTUAL_PATTERNS);

            for keyword in reasoning_keywords {
                if matches_bounded(bytes, index, keyword) {
                    self.features.reasoning_keywords =
                        self.features.reasoning_keywords.saturating_add(1);
                }
            }

            index += 1;
        }
    }

    fn task_type(&self) -> TaskType {
        if self.tool_use {
            TaskType::ToolUse
        } else if self.code_generation {
            TaskType::CodeGeneration
        } else if self.math_reasoning {
            TaskType::MathReasoning
        } else if self.summarization {
            TaskType::Summarization
        } else if self.creative_writing {
            TaskType::CreativeWriting
        } else if self.factual_qa {
            TaskType::FactualQA
        } else {
            TaskType::General
        }
    }
}

pub fn visit_text_content(message: &Message, mut visit: impl FnMut(&str)) {
    match &message.content {
        serde_json::Value::String(text) => visit(text),
        serde_json::Value::Array(parts) => {
            for part in parts {
                let part_type = part.get("type").and_then(serde_json::Value::as_str);
                if matches!(part_type, Some("text" | "input_text")) {
                    if let Some(text) = part.get("text").and_then(serde_json::Value::as_str) {
                        visit(text);
                    }
                }
            }
        }
        _ => {}
    }
}

fn structural_item_count(value: &serde_json::Value) -> usize {
    match value {
        serde_json::Value::Null | serde_json::Value::Bool(false) => 0,
        serde_json::Value::Array(items) => items.len(),
        _ => 1,
    }
}

fn is_present_structure(value: &serde_json::Value) -> bool {
    !matches!(
        value,
        serde_json::Value::Null | serde_json::Value::Bool(false)
    )
}

fn normalized_weights(weights: HeuristicWeights) -> [f64; FEATURE_COUNT] {
    let mut values = [
        weights.message_count,
        weights.token_estimate,
        weights.code_blocks,
        weights.tool_calls,
        weights.math_expressions,
        weights.reasoning_keywords,
    ];
    for value in &mut values {
        if !value.is_finite() || *value < 0.0 {
            *value = 0.0;
        }
    }

    let total = values.iter().sum::<f64>();
    if total <= 0.0 {
        [1.0 / FEATURE_COUNT as f64; FEATURE_COUNT]
    } else {
        values.map(|value| value / total)
    }
}

fn sigmoid(value: usize, scale: f64) -> f64 {
    1.0 - (-(value as f64) / scale).exp()
}

fn matches_any_pattern(bytes: &[u8], index: usize, patterns: &[&str]) -> bool {
    patterns
        .iter()
        .any(|pattern| matches_bounded(bytes, index, pattern.as_bytes()))
}

fn matches_bounded(bytes: &[u8], index: usize, pattern: &[u8]) -> bool {
    starts_with_ignore_ascii_case(bytes, index, pattern)
        && is_word_boundary_before(bytes, index)
        && is_word_boundary_after(bytes, index.saturating_add(pattern.len()))
}

fn starts_with(bytes: &[u8], index: usize, pattern: &[u8]) -> bool {
    bytes
        .get(index..index.saturating_add(pattern.len()))
        .is_some_and(|candidate| candidate == pattern)
}

fn starts_with_ignore_ascii_case(bytes: &[u8], index: usize, pattern: &[u8]) -> bool {
    bytes
        .get(index..index.saturating_add(pattern.len()))
        .is_some_and(|candidate| candidate.eq_ignore_ascii_case(pattern))
}

fn is_word_boundary_before(bytes: &[u8], index: usize) -> bool {
    index == 0 || !bytes[index - 1].is_ascii_alphanumeric() && bytes[index - 1] != b'_'
}

fn is_word_boundary_after(bytes: &[u8], index: usize) -> bool {
    index >= bytes.len() || !bytes[index].is_ascii_alphanumeric() && bytes[index] != b'_'
}

fn is_utf8_continuation(byte: u8) -> bool {
    byte & 0b1100_0000 == 0b1000_0000
}

#[cfg(test)]
mod tests {
    use super::*;
    use proptest::prelude::*;
    use serde_json::{json, Map, Value};

    fn message(content: Value) -> Message {
        Message {
            role: "user".to_string(),
            content,
            extra: Map::new(),
        }
    }

    fn text(content: &str) -> Message {
        message(Value::String(content.to_string()))
    }

    fn one_hot(feature: usize) -> HeuristicWeights {
        let mut values = [0.0; FEATURE_COUNT];
        values[feature] = 1.0;
        HeuristicWeights {
            message_count: values[0],
            token_estimate: values[1],
            code_blocks: values[2],
            tool_calls: values[3],
            math_expressions: values[4],
            reasoning_keywords: values[5],
        }
    }

    fn assert_close(actual: f64, expected: f64) {
        assert!(
            (actual - expected).abs() < 1.0e-12,
            "{actual} != {expected}"
        );
    }

    fn arb_text_payload() -> impl Strategy<Value = String> {
        (
            prop::collection::vec(any::<char>(), 0..=48),
            0usize..=3,
            0usize..=3,
            prop::sample::select(vec![
                "",
                " analyze step by step",
                " use a tool",
                " summarize this",
                " write a poem",
                " what is the answer?",
            ]),
        )
            .prop_map(|(characters, code_blocks, math_expressions, prompt)| {
                let mut content = characters.into_iter().collect::<String>();
                content.push_str(prompt);
                content.push_str(&" ```rust\nfn generated() {}\n```".repeat(code_blocks));
                content.push_str(&r" \(x + y = z\)".repeat(math_expressions));
                content
            })
    }

    fn arb_content() -> impl Strategy<Value = Value> {
        prop_oneof![
            3 => arb_text_payload().prop_map(Value::String),
            2 => prop::collection::vec(arb_text_payload(), 0..=4).prop_map(|texts| {
                Value::Array(
                    texts
                        .into_iter()
                        .enumerate()
                        .map(|(index, text)| {
                            if index % 2 == 0 {
                                json!({"type": "text", "text": text})
                            } else {
                                json!({"type": "input_text", "text": text})
                            }
                        })
                        .chain(std::iter::once(json!({
                            "type": "image_url",
                            "image_url": {"url": "https://example.invalid/image.png"}
                        })))
                        .collect(),
                )
            }),
            1 => Just(Value::Null),
            1 => Just(json!({"unexpected": "object"})),
        ]
    }

    fn arb_message() -> impl Strategy<Value = Message> {
        (
            prop::sample::select(vec!["user", "assistant", "system", "tool"]),
            arb_content(),
            0u8..=4,
            0usize..=4,
        )
            .prop_map(|(role, content, tool_extra_kind, tool_call_count)| {
                let mut extra = Map::new();
                match tool_extra_kind {
                    1 => {
                        extra.insert(
                            "tool_calls".to_string(),
                            Value::Array(
                                (0..tool_call_count)
                                    .map(|index| json!({"id": format!("call-{index}")}))
                                    .collect(),
                            ),
                        );
                    }
                    2 => {
                        extra.insert(
                            "function_call".to_string(),
                            json!({"name": "bounded_tool", "arguments": "{}"}),
                        );
                    }
                    3 => {
                        extra.insert("tool_call_id".to_string(), json!("call-bounded"));
                    }
                    4 => {
                        extra.insert(
                            "tool_calls".to_string(),
                            Value::Array(
                                (0..tool_call_count)
                                    .map(|index| json!({"id": format!("call-{index}")}))
                                    .collect(),
                            ),
                        );
                        extra.insert(
                            "function_call".to_string(),
                            json!({"name": "bounded_tool", "arguments": "{}"}),
                        );
                        extra.insert("tool_call_id".to_string(), json!("call-bounded"));
                    }
                    _ => {}
                }

                Message {
                    role: role.to_string(),
                    content,
                    extra,
                }
            })
    }

    fn arb_messages() -> impl Strategy<Value = Vec<Message>> {
        prop::collection::vec(arb_message(), 0..=8)
    }

    fn arb_valid_normalized_weights() -> impl Strategy<Value = HeuristicWeights> {
        prop::array::uniform5(any::<u8>()).prop_map(|mut cuts| {
            cuts.sort_unstable();
            let boundaries = [
                0u16,
                cuts[0] as u16,
                cuts[1] as u16,
                cuts[2] as u16,
                cuts[3] as u16,
                cuts[4] as u16,
                256u16,
            ];
            let mut values = [0.0; FEATURE_COUNT];
            for (index, interval) in boundaries.windows(2).enumerate() {
                values[index] = f64::from(interval[1] - interval[0]) / 256.0;
            }

            HeuristicWeights {
                message_count: values[0],
                token_estimate: values[1],
                code_blocks: values[2],
                tool_calls: values[3],
                math_expressions: values[4],
                reasoning_keywords: values[5],
            }
        })
    }

    fn weight_values(weights: &HeuristicWeights) -> [f64; FEATURE_COUNT] {
        [
            weights.message_count,
            weights.token_estimate,
            weights.code_blocks,
            weights.tool_calls,
            weights.math_expressions,
            weights.reasoning_keywords,
        ]
    }

    #[test]
    fn empty_whitespace_and_non_text_content_score_zero() {
        let scorer = HeuristicScorer::default();
        for messages in [
            Vec::new(),
            vec![text("")],
            vec![text(" \n\t")],
            vec![message(
                json!([{"type": "image_url", "image_url": {"url": "x"}}]),
            )],
            vec![message(json!({"unexpected": "object"}))],
        ] {
            let assessment = scorer.score(&messages);
            assert_eq!(assessment.score.value(), 0.0);
            assert_eq!(assessment.task_type, TaskType::General);
        }
    }

    #[test]
    fn message_count_feature_uses_the_sigmoid_scale() {
        let scorer = HeuristicScorer::new(Some(one_hot(0)));
        let assessment = scorer.score(&[text("a"), text("b")]);
        assert_eq!(assessment.features.message_count, 2);
        assert_close(assessment.score.value(), sigmoid(2, MESSAGE_COUNT_SCALE));
    }

    #[test]
    fn token_estimate_counts_unicode_characters_and_uses_integer_division() {
        let scorer = HeuristicScorer::new(Some(one_hot(1)));
        let assessment = scorer.score(&[text("aé日🙂zabc")]);
        assert_eq!(assessment.features.character_count, 8);
        assert_eq!(assessment.features.token_estimate, 2);
        assert_eq!(assessment.token_estimate(), 2);
        assert_close(assessment.score.value(), sigmoid(2, TOKEN_ESTIMATE_SCALE));
    }

    #[test]
    fn code_blocks_are_counted_and_detect_code_generation() {
        let scorer = HeuristicScorer::new(Some(one_hot(2)));
        let assessment = scorer.score(&[text("```rust\nfn main() {}\n```")]);
        assert_eq!(assessment.features.code_blocks, 1);
        assert_eq!(assessment.task_type, TaskType::CodeGeneration);
        assert_close(assessment.score.value(), sigmoid(1, CODE_BLOCK_SCALE));
    }

    #[test]
    fn tool_structures_count_without_text_or_content_copying() {
        let scorer = HeuristicScorer::new(Some(one_hot(3)));
        let mut tool_message = message(Value::Null);
        tool_message.extra.insert(
            "tool_calls".to_string(),
            json!([{"id": "one"}, {"id": "two"}]),
        );
        let assessment = scorer.score(&[tool_message]);
        assert_eq!(assessment.features.tool_calls, 2);
        assert_eq!(assessment.task_type, TaskType::ToolUse);
        assert_close(assessment.score.value(), sigmoid(2, TOOL_CALL_SCALE));
    }

    #[test]
    fn tool_role_is_a_structural_signal() {
        let scorer = HeuristicScorer::default();
        let mut tool_message = message(Value::Null);
        tool_message.role = "tool".to_string();
        let assessment = scorer.score(&[tool_message]);
        assert_eq!(assessment.features.tool_calls, 1);
        assert_eq!(assessment.task_type, TaskType::ToolUse);
        assert!(assessment.score.value() > 0.0);
    }

    #[test]
    fn latex_math_expressions_are_counted() {
        let scorer = HeuristicScorer::new(Some(one_hot(4)));
        let assessment = scorer.score(&[text(r"Solve \(x + 1 = 2\) and $$y = x^2$$")]);
        assert_eq!(assessment.features.math_expressions, 2);
        assert_eq!(assessment.task_type, TaskType::MathReasoning);
        assert_close(assessment.score.value(), sigmoid(2, MATH_EXPRESSION_SCALE));
    }

    #[test]
    fn reasoning_keywords_are_counted_and_configurable() {
        let default_scorer = HeuristicScorer::new(Some(one_hot(5)));
        let default_assessment = default_scorer.score(&[text("Analyze and compare step by step.")]);
        assert_eq!(default_assessment.features.reasoning_keywords, 3);

        let custom = HeuristicScorer::with_reasoning_keywords(
            Some(one_hot(5)),
            ["inspect deeply", "inspect deeply", ""],
        );
        let custom_assessment = custom.score(&[text("Please INSPECT DEEPLY, then analyze.")]);
        assert_eq!(custom_assessment.features.reasoning_keywords, 1);
        assert_close(
            custom_assessment.score.value(),
            sigmoid(1, REASONING_KEYWORD_SCALE),
        );
    }

    #[test]
    fn multimodal_text_parts_are_scanned_without_string_assembly() {
        let assessment = HeuristicScorer::default().score(&[message(json!([
            {"type": "text", "text": "Summarize this."},
            {"type": "image_url", "image_url": {"url": "x"}},
            {"type": "input_text", "text": " Be concise."}
        ]))]);
        assert_eq!(assessment.features.character_count, 27);
        assert_eq!(assessment.task_type, TaskType::Summarization);
    }

    #[test]
    fn task_detection_uses_fixed_precedence() {
        let scorer = HeuristicScorer::default();
        let cases = [
            (
                "Use a tool to write code, solve the equation, summarize it, and write a poem. What is it?",
                TaskType::ToolUse,
            ),
            (
                "Write code to solve the equation, summarize it, and write a poem.",
                TaskType::CodeGeneration,
            ),
            (
                "Solve the equation, summarize it, and write a poem.",
                TaskType::MathReasoning,
            ),
            ("Summarize this and write a poem.", TaskType::Summarization),
            ("Write a poem. What is the theme?", TaskType::CreativeWriting),
            ("What is the capital of France?", TaskType::FactualQA),
            ("Hello there.", TaskType::General),
        ];

        for (content, expected) in cases {
            assert_eq!(scorer.score(&[text(content)]).task_type, expected);
        }
    }

    #[test]
    fn default_and_explicit_equal_weights_match() {
        let messages = [text("Analyze this ```code``` and solve \\(x=1\\).")];
        let default_score = HeuristicScorer::new(None).score(&messages).score;
        let explicit_score = HeuristicScorer::new(Some(HeuristicWeights::default()))
            .score(&messages)
            .score;
        assert_eq!(default_score, explicit_score);
    }

    proptest! {
        #![proptest_config(ProptestConfig::with_cases(256))]

        #[test]
        fn heuristic_score_is_finite_and_in_range(
            messages in arb_messages(),
            weights in arb_valid_normalized_weights(),
        ) {
            let values = weight_values(&weights);
            prop_assert!(values.iter().all(|value| value.is_finite() && *value >= 0.0));
            prop_assert_eq!(values.iter().sum::<f64>(), 1.0);

            let score = HeuristicScorer::new(Some(weights)).score(&messages).score.value();
            prop_assert!(score.is_finite());
            prop_assert!((0.0..=1.0).contains(&score));
        }

        #[test]
        fn heuristic_scoring_is_deterministic(
            messages in arb_messages(),
            weights in arb_valid_normalized_weights(),
        ) {
            let scorer = HeuristicScorer::new(Some(weights));
            let first = scorer.score(&messages);
            let second = scorer.score(&messages);

            prop_assert_eq!(first, second);
        }
    }

    #[test]
    fn invalid_or_zero_weights_still_produce_a_bounded_score() {
        let weights = HeuristicWeights {
            message_count: f64::NAN,
            token_estimate: -1.0,
            code_blocks: 0.0,
            tool_calls: 0.0,
            math_expressions: 0.0,
            reasoning_keywords: 0.0,
        };
        let assessment = HeuristicScorer::new(Some(weights)).score(&[text("Hello")]);
        assert!(assessment.score.value().is_finite());
        assert!((0.0..=1.0).contains(&assessment.score.value()));
    }
}
