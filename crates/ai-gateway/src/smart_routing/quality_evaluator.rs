//! Fast, structural response-quality scoring for smart routing.
//!
//! The evaluator intentionally measures only observable completion signals. It
//! does not attempt to determine whether response content is factually or
//! semantically correct, and it never logs request or response content.

use std::borrow::Borrow;

use serde_json::{Map, Value};

use crate::models::openai::{Choice, Message, OpenAIRequest, OpenAIResponse, Usage};
use crate::smart_routing::config::QualityEvaluatorConfig;

const DEFAULT_QUALITY_THRESHOLD: f64 = 0.3;
const FINISH_REASON_WEIGHT: f64 = 0.30;
const RESPONSE_RATIO_WEIGHT: f64 = 0.20;
const EXPECTED_STRUCTURE_WEIGHT: f64 = 0.20;
const STRUCTURAL_COMPLETENESS_WEIGHT: f64 = 0.20;
const LOGPROB_WEIGHT: f64 = 0.10;

const VALIDATION_EXTRA_KEYS: &[&str] = &[
    "structured_output_validation",
    "structured_output_validation_result",
    "structured_output_outcome",
    "gateway_structured_output_validation",
    "gateway_structured_output_outcome",
];

/// Synchronous response-quality evaluator configured for smart routing.
///
/// Evaluation performs linear scans over response text and optional log-probability
/// arrays. JSON is parsed only when the request explicitly expects JSON or tool
/// arguments need structural validation. No JSON Schema is compiled here.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct ResponseQualityEvaluator {
    enabled: bool,
    quality_threshold: f64,
}

impl ResponseQualityEvaluator {
    /// Build an evaluator from the validated smart-routing configuration.
    ///
    /// The threshold is still sanitized defensively because callers can construct
    /// `QualityEvaluatorConfig` directly without running global validation.
    pub fn new(config: impl Borrow<QualityEvaluatorConfig>) -> Self {
        let config = config.borrow();
        let quality_threshold = if config.threshold.is_finite() {
            config.threshold.clamp(0.0, 1.0)
        } else {
            DEFAULT_QUALITY_THRESHOLD
        };

        Self {
            enabled: config.enabled,
            quality_threshold,
        }
    }

    /// Evaluate observable response quality and return a finite score in `[0, 1]`.
    ///
    /// This is a structural confidence score, not a semantic-correctness score.
    pub fn evaluate(&self, response: &OpenAIResponse, request: &OpenAIRequest) -> f64 {
        let mut score = WeightedScore::default();
        score.add(finish_reason_score(&response.choices), FINISH_REASON_WEIGHT);
        score.add(
            response_to_input_ratio_score(response, request),
            RESPONSE_RATIO_WEIGHT,
        );

        let expectation = StructuredExpectation::from_request(request);
        if expectation.is_expected() {
            score.add(
                expected_structure_score(response, request, expectation),
                EXPECTED_STRUCTURE_WEIGHT,
            );
        }

        score.add(
            structural_completeness_score(response, expectation),
            STRUCTURAL_COMPLETENESS_WEIGHT,
        );

        if let Some(confidence) = average_logprob_confidence(response) {
            score.add(confidence, LOGPROB_WEIGHT);
        }

        score.finish()
    }

    /// Return whether an evaluated score is a cascade failure signal.
    ///
    /// Disabled evaluators never signal failure. Non-finite scores are treated as
    /// low quality rather than allowing invalid numeric state into cascade logic.
    pub fn is_low_quality(&self, score: f64) -> bool {
        self.enabled && (!score.is_finite() || score < self.quality_threshold)
    }

    /// Configured, sanitized quality threshold.
    pub fn threshold(&self) -> f64 {
        self.quality_threshold
    }

    /// Whether quality-based failure signaling is enabled.
    pub fn is_enabled(&self) -> bool {
        self.enabled
    }
}

impl From<QualityEvaluatorConfig> for ResponseQualityEvaluator {
    fn from(config: QualityEvaluatorConfig) -> Self {
        Self::new(config)
    }
}

impl From<&QualityEvaluatorConfig> for ResponseQualityEvaluator {
    fn from(config: &QualityEvaluatorConfig) -> Self {
        Self::new(config.clone())
    }
}

#[derive(Debug, Default)]
struct WeightedScore {
    weighted_sum: f64,
    total_weight: f64,
}

impl WeightedScore {
    fn add(&mut self, value: f64, weight: f64) {
        let value = finite_unit(value);
        if weight.is_finite() && weight > 0.0 {
            self.weighted_sum += value * weight;
            self.total_weight += weight;
        }
    }

    fn finish(self) -> f64 {
        if !self.weighted_sum.is_finite() || self.total_weight <= 0.0 {
            return 0.0;
        }
        finite_unit(self.weighted_sum / self.total_weight)
    }
}

#[derive(Debug, Clone, Copy, Default)]
struct StructuredExpectation {
    json: bool,
    strict_schema: bool,
    tool_call: bool,
    strict_tool: bool,
}

impl StructuredExpectation {
    fn from_request(request: &OpenAIRequest) -> Self {
        let mut expectation = Self::default();

        if let Some(response_format) = request
            .extra
            .get("response_format")
            .and_then(Value::as_object)
        {
            let format_type = response_format.get("type").and_then(Value::as_str);
            expectation.json = matches!(format_type, Some("json_object" | "json_schema"));
            expectation.strict_schema = response_format
                .get("json_schema")
                .and_then(Value::as_object)
                .and_then(|schema| schema.get("strict"))
                .and_then(Value::as_bool)
                .unwrap_or(false)
                || response_format
                    .get("strict")
                    .and_then(Value::as_bool)
                    .unwrap_or(false);
        }

        let has_tools = request
            .extra
            .get("tools")
            .is_some_and(has_nonempty_structure);
        expectation.strict_tool = request
            .extra
            .get("tools")
            .is_some_and(tools_include_strict_schema);
        expectation.tool_call =
            has_tools && tool_choice_requires_call(request.extra.get("tool_choice"));
        expectation
    }

    fn is_expected(self) -> bool {
        self.json || self.strict_schema || self.tool_call || self.strict_tool
    }
}

fn finish_reason_score(choices: &[Choice]) -> f64 {
    if choices.is_empty() {
        return 0.0;
    }

    let total = choices
        .iter()
        .map(|choice| match choice.finish_reason.as_deref() {
            Some("stop" | "tool_calls" | "function_call") => 1.0,
            Some("length" | "max_tokens") => 0.5,
            Some("content_filter" | "safety") => 0.3,
            Some("error" | "failed" | "cancelled") => 0.0,
            None => 0.5,
            Some(_) => 0.5,
        })
        .sum::<f64>();

    total / choices.len() as f64
}

fn response_to_input_ratio_score(response: &OpenAIResponse, request: &OpenAIRequest) -> f64 {
    let input_tokens = token_count(
        &response.usage,
        response.usage.prompt_tokens,
        &["input_tokens", "prompt_token_count"],
    )
    .or_else(|| extra_token_count(&request.extra, &["input_tokens", "prompt_tokens"]))
    .unwrap_or_else(|| estimate_request_tokens(request));
    let output_tokens = token_count(
        &response.usage,
        response.usage.completion_tokens,
        &["output_tokens", "completion_token_count"],
    )
    .unwrap_or_else(|| estimate_response_tokens(response));

    if output_tokens == 0 {
        return 0.0;
    }

    let ratio = output_tokens as f64 / input_tokens.max(1) as f64;
    let bounded_ratio = ratio.min(8.0);
    let raw = 1.0 / (1.0 + (-8.0 * (bounded_ratio - 0.25)).exp());
    let floor = 1.0 / (1.0 + 2.0_f64.exp());
    finite_unit((raw - floor) / (1.0 - floor))
}

fn token_count(usage: &Usage, explicit: u32, aliases: &[&str]) -> Option<u64> {
    if explicit > 0 {
        Some(u64::from(explicit))
    } else {
        extra_token_count(&usage.extra, aliases)
    }
}

fn extra_token_count(extra: &Map<String, Value>, keys: &[&str]) -> Option<u64> {
    keys.iter().find_map(|key| {
        let value = extra.get(*key)?;
        value.as_u64().or_else(|| {
            let numeric = value.as_f64()?;
            (numeric.is_finite() && numeric >= 0.0).then_some(numeric as u64)
        })
    })
}

fn estimate_request_tokens(request: &OpenAIRequest) -> u64 {
    let chars = request
        .messages
        .iter()
        .map(|message| text_char_count(&message.content))
        .fold(0_u64, u64::saturating_add);
    chars.saturating_add(3) / 4
}

fn estimate_response_tokens(response: &OpenAIResponse) -> u64 {
    let chars = response
        .choices
        .iter()
        .map(|choice| text_char_count(&choice.message.content))
        .fold(0_u64, u64::saturating_add);
    chars.saturating_add(3) / 4
}

fn text_char_count(content: &Value) -> u64 {
    match content {
        Value::String(text) => text.chars().count() as u64,
        Value::Array(parts) => parts
            .iter()
            .filter_map(|part| {
                (part.get("type").and_then(Value::as_str) == Some("text"))
                    .then(|| part.get("text").and_then(Value::as_str))
                    .flatten()
            })
            .map(|text| text.chars().count() as u64)
            .fold(0_u64, u64::saturating_add),
        Value::Null => 0,
        other => other.to_string().chars().count() as u64,
    }
}

fn expected_structure_score(
    response: &OpenAIResponse,
    request: &OpenAIRequest,
    expectation: StructuredExpectation,
) -> f64 {
    if expectation.strict_schema || expectation.strict_tool {
        if let Some(result) = reused_validation_score(response, request) {
            return result;
        }
    }

    let mut total = 0.0;
    let mut count = 0_u32;
    for choice in &response.choices {
        if expectation.json || expectation.strict_schema {
            total += if message_contains_valid_json(&choice.message) {
                1.0
            } else {
                0.0
            };
            count += 1;
        }

        let choice_has_tool_call = choice_has_tool_calls(choice);
        if expectation.tool_call {
            total += if choice_has_tool_call { 1.0 } else { 0.0 };
            count += 1;
        }
        if expectation.strict_tool && choice_has_tool_call {
            total += tool_arguments_score(choice);
            count += 1;
        }
    }

    if count == 0 {
        0.0
    } else {
        total / f64::from(count)
    }
}

fn structural_completeness_score(
    response: &OpenAIResponse,
    expectation: StructuredExpectation,
) -> f64 {
    if response.choices.is_empty() {
        return 0.0;
    }

    let total = response
        .choices
        .iter()
        .map(|choice| {
            let mut scanner = StructureScanner::default();
            visit_message_text(&choice.message, |text| scanner.scan(text));
            let has_tool_calls = choice_has_tool_calls(choice);
            let mut score = if scanner.has_non_whitespace() {
                scanner.score()
            } else if has_tool_calls {
                1.0
            } else {
                0.0
            };

            if expectation.json || expectation.strict_schema {
                score = (score
                    + if message_contains_valid_json(&choice.message) {
                        1.0
                    } else {
                        0.0
                    })
                    / 2.0;
            }
            if has_tool_calls {
                score = (score + tool_arguments_score(choice)) / 2.0;
            }
            score
        })
        .sum::<f64>();

    total / response.choices.len() as f64
}

#[derive(Debug, Default)]
struct StructureScanner {
    delimiters: Vec<u8>,
    delimiter_error: bool,
    fence_open: bool,
    fence_marker: Option<u8>,
    quote: Option<u8>,
    escaped: bool,
    has_non_whitespace: bool,
}

impl StructureScanner {
    fn scan(&mut self, text: &str) {
        let bytes = text.as_bytes();
        let mut index = 0;
        while index < bytes.len() {
            let byte = bytes[index];
            if !byte.is_ascii_whitespace() {
                self.has_non_whitespace = true;
            }

            if self.quote.is_none()
                && matches!(byte, b'`' | b'~')
                && bytes.get(index + 1) == Some(&byte)
                && bytes.get(index + 2) == Some(&byte)
            {
                match self.fence_marker {
                    Some(marker) if marker == byte => {
                        self.fence_open = !self.fence_open;
                        if !self.fence_open {
                            self.fence_marker = None;
                        }
                    }
                    None => {
                        self.fence_open = true;
                        self.fence_marker = Some(byte);
                    }
                    Some(_) => {}
                }
                index += 3;
                continue;
            }

            if let Some(quote) = self.quote {
                if self.escaped {
                    self.escaped = false;
                } else if byte == b'\\' {
                    self.escaped = true;
                } else if byte == quote {
                    self.quote = None;
                }
                index += 1;
                continue;
            }

            if matches!(byte, b'\'' | b'"') {
                self.quote = Some(byte);
                index += 1;
                continue;
            }

            match byte {
                b'(' | b'[' | b'{' => self.delimiters.push(byte),
                b')' | b']' | b'}' => {
                    let expected = match byte {
                        b')' => b'(',
                        b']' => b'[',
                        b'}' => b'{',
                        _ => unreachable!(),
                    };
                    if self.delimiters.pop() != Some(expected) {
                        self.delimiter_error = true;
                    }
                }
                _ => {}
            }
            index += 1;
        }
    }

    fn score(&self) -> f64 {
        if !self.has_non_whitespace {
            0.0
        } else if self.fence_open
            || self.delimiter_error
            || !self.delimiters.is_empty()
            || self.quote.is_some()
        {
            0.25
        } else {
            1.0
        }
    }

    fn has_non_whitespace(&self) -> bool {
        self.has_non_whitespace
    }
}

fn message_contains_valid_json(message: &Message) -> bool {
    match &message.content {
        Value::String(text) => serde_json::from_str::<Value>(text.trim()).is_ok(),
        Value::Object(_) => true,
        Value::Array(parts) if content_parts_are_text(parts) => {
            let mut combined = String::new();
            for part in parts {
                if let Some(text) = part.get("text").and_then(Value::as_str) {
                    combined.push_str(text);
                }
            }
            serde_json::from_str::<Value>(combined.trim()).is_ok()
        }
        Value::Array(_) => true,
        _ => false,
    }
}

fn content_parts_are_text(parts: &[Value]) -> bool {
    !parts.is_empty()
        && parts.iter().all(|part| {
            part.get("type").and_then(Value::as_str) == Some("text")
                && part.get("text").and_then(Value::as_str).is_some()
        })
}

fn visit_message_text(message: &Message, mut visit: impl FnMut(&str)) {
    match &message.content {
        Value::String(text) => visit(text),
        Value::Array(parts) => {
            for part in parts {
                if part.get("type").and_then(Value::as_str) == Some("text") {
                    if let Some(text) = part.get("text").and_then(Value::as_str) {
                        visit(text);
                    }
                }
            }
        }
        _ => {}
    }
}

fn choice_has_tool_calls(choice: &Choice) -> bool {
    choice
        .message
        .extra
        .get("tool_calls")
        .is_some_and(has_nonempty_structure)
        || choice
            .message
            .extra
            .get("function_call")
            .is_some_and(has_nonempty_structure)
}

fn tool_arguments_score(choice: &Choice) -> f64 {
    let Some(tool_calls) = choice.message.extra.get("tool_calls") else {
        return choice
            .message
            .extra
            .get("function_call")
            .map_or(0.0, valid_function_call_score);
    };

    let Some(tool_calls) = tool_calls.as_array() else {
        return 0.0;
    };
    if tool_calls.is_empty() {
        return 0.0;
    }

    tool_calls
        .iter()
        .map(valid_function_call_score)
        .sum::<f64>()
        / tool_calls.len() as f64
}

fn valid_function_call_score(call: &Value) -> f64 {
    let function = call.get("function").unwrap_or(call);
    let has_name = function
        .get("name")
        .and_then(Value::as_str)
        .is_some_and(|name| !name.trim().is_empty());
    let valid_arguments = match function.get("arguments") {
        Some(Value::String(arguments)) => serde_json::from_str::<Value>(arguments.trim()).is_ok(),
        Some(Value::Object(_) | Value::Array(_)) => true,
        _ => false,
    };
    if has_name && valid_arguments {
        1.0
    } else {
        0.0
    }
}

fn tool_choice_requires_call(tool_choice: Option<&Value>) -> bool {
    match tool_choice {
        Some(Value::String(choice)) => !matches!(choice.as_str(), "none" | "auto"),
        Some(Value::Object(choice)) => !choice.is_empty(),
        _ => false,
    }
}

fn tools_include_strict_schema(value: &Value) -> bool {
    value.as_array().is_some_and(|tools| {
        tools.iter().any(|tool| {
            tool.get("function")
                .and_then(Value::as_object)
                .and_then(|function| function.get("strict"))
                .and_then(Value::as_bool)
                .unwrap_or(false)
        })
    })
}

fn has_nonempty_structure(value: &Value) -> bool {
    match value {
        Value::Array(values) => !values.is_empty(),
        Value::Object(values) => !values.is_empty(),
        Value::String(value) => !value.trim().is_empty(),
        Value::Bool(value) => *value,
        Value::Number(_) => true,
        Value::Null => false,
    }
}

fn reused_validation_score(response: &OpenAIResponse, request: &OpenAIRequest) -> Option<f64> {
    validation_score_from_map(&response.extra)
        .or_else(|| {
            response
                .choices
                .iter()
                .find_map(|choice| validation_score_from_map(&choice.extra))
        })
        .or_else(|| validation_score_from_map(&request.extra))
}

fn validation_score_from_map(extra: &Map<String, Value>) -> Option<f64> {
    VALIDATION_EXTRA_KEYS.iter().find_map(|key| {
        extra
            .get(*key)
            .and_then(|value| validation_value_score(value, 0))
    })
}

fn validation_value_score(value: &Value, depth: u8) -> Option<f64> {
    if depth > 3 {
        return None;
    }
    match value {
        Value::Bool(valid) => Some(if *valid { 1.0 } else { 0.0 }),
        Value::String(status) => validation_status_score(status),
        Value::Object(object) => {
            for key in ["outcome", "status", "result", "valid", "passed", "success"] {
                if let Some(score) = object
                    .get(key)
                    .and_then(|nested| validation_value_score(nested, depth + 1))
                {
                    return Some(score);
                }
            }
            object
                .get("choices")
                .and_then(|choices| validation_value_score(choices, depth + 1))
        }
        Value::Array(values) => {
            let mut total = 0.0;
            let mut count = 0_u32;
            for item in values {
                if let Some(score) = validation_value_score(item, depth + 1) {
                    total += score;
                    count += 1;
                }
            }
            (count > 0).then_some(total / f64::from(count))
        }
        _ => None,
    }
}

fn validation_status_score(status: &str) -> Option<f64> {
    if ["pass", "passed", "valid", "success", "succeeded"]
        .iter()
        .any(|candidate| status.eq_ignore_ascii_case(candidate))
    {
        Some(1.0)
    } else if [
        "fail",
        "failed",
        "invalid",
        "error",
        "json_parse_error",
        "schema_violations",
    ]
    .iter()
    .any(|candidate| status.eq_ignore_ascii_case(candidate))
    {
        Some(0.0)
    } else {
        None
    }
}

fn average_logprob_confidence(response: &OpenAIResponse) -> Option<f64> {
    let mut sum = 0.0;
    let mut count = 0_u64;

    add_direct_average(&response.extra, &mut sum, &mut count);
    add_direct_average(&response.usage.extra, &mut sum, &mut count);
    for choice in &response.choices {
        add_direct_average(&choice.extra, &mut sum, &mut count);
        if let Some(logprobs) = choice.extra.get("logprobs") {
            collect_logprobs(logprobs, &mut sum, &mut count);
        }
    }

    if count == 0 {
        None
    } else {
        let average = sum / count as f64;
        Some(finite_unit(average.exp()))
    }
}

fn add_direct_average(extra: &Map<String, Value>, sum: &mut f64, count: &mut u64) {
    for key in ["average_logprob", "avg_logprob", "mean_logprob"] {
        if let Some(value) = extra.get(key).and_then(Value::as_f64) {
            if value.is_finite() {
                *sum += value;
                *count += 1;
                return;
            }
        }
    }
}

fn collect_logprobs(value: &Value, sum: &mut f64, count: &mut u64) {
    match value {
        Value::Array(entries) => collect_logprob_entries(entries, sum, count),
        Value::Object(object) => {
            add_direct_average(object, sum, count);
            if let Some(entries) = object.get("content").and_then(Value::as_array) {
                collect_logprob_entries(entries, sum, count);
            }
        }
        _ => {}
    }
}

fn collect_logprob_entries(entries: &[Value], sum: &mut f64, count: &mut u64) {
    for entry in entries {
        if let Some(logprob) = entry.get("logprob").and_then(Value::as_f64) {
            if logprob.is_finite() {
                *sum += logprob;
                *count += 1;
            }
        }
    }
}

fn finite_unit(value: f64) -> f64 {
    if value.is_finite() {
        value.clamp(0.0, 1.0)
    } else {
        0.0
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use proptest::prelude::*;
    use serde_json::json;

    fn arb_json_value() -> impl Strategy<Value = Value> {
        let leaf = prop_oneof![
            Just(Value::Null),
            any::<bool>().prop_map(Value::Bool),
            any::<i64>().prop_map(|value| Value::Number(value.into())),
            any::<String>().prop_map(Value::String),
        ];

        leaf.prop_recursive(3, 64, 8, |inner| {
            prop_oneof![
                prop::collection::vec(inner.clone(), 0..8).prop_map(Value::Array),
                prop::collection::btree_map(any::<String>(), inner, 0..8)
                    .prop_map(|entries| { Value::Object(entries.into_iter().collect()) }),
            ]
        })
    }

    fn arb_extra() -> impl Strategy<Value = Map<String, Value>> {
        prop::collection::btree_map(any::<String>(), arb_json_value(), 0..8)
            .prop_map(|entries| entries.into_iter().collect())
    }

    fn arb_message() -> impl Strategy<Value = Message> {
        (any::<String>(), arb_json_value(), arb_extra()).prop_map(|(role, content, extra)| {
            Message {
                role,
                content,
                extra,
            }
        })
    }

    fn arb_request() -> impl Strategy<Value = OpenAIRequest> {
        (
            any::<String>(),
            prop::collection::vec(arb_message(), 0..8),
            any::<bool>(),
            proptest::option::of(any::<f32>()),
            proptest::option::of(any::<u32>()),
            arb_extra(),
        )
            .prop_map(
                |(model, messages, stream, temperature, max_tokens, extra)| OpenAIRequest {
                    model,
                    messages,
                    stream,
                    temperature,
                    max_tokens,
                    extra,
                },
            )
    }

    fn arb_choice() -> impl Strategy<Value = Choice> {
        (
            any::<u32>(),
            arb_message(),
            proptest::option::of(any::<String>()),
            arb_extra(),
        )
            .prop_map(|(index, message, finish_reason, extra)| Choice {
                index,
                message,
                finish_reason,
                extra,
            })
    }

    fn arb_usage() -> impl Strategy<Value = Usage> {
        (any::<u32>(), any::<u32>(), any::<u32>(), arb_extra()).prop_map(
            |(prompt_tokens, completion_tokens, total_tokens, extra)| Usage {
                prompt_tokens,
                completion_tokens,
                total_tokens,
                extra,
            },
        )
    }

    fn arb_response() -> impl Strategy<Value = OpenAIResponse> {
        (
            any::<String>(),
            any::<String>(),
            any::<i64>(),
            any::<String>(),
            prop::collection::vec(arb_choice(), 0..8),
            arb_usage(),
            arb_extra(),
        )
            .prop_map(|(id, object, created, model, choices, usage, extra)| {
                OpenAIResponse {
                    id,
                    object,
                    created,
                    model,
                    choices,
                    usage,
                    extra,
                }
            })
    }

    fn evaluator(enabled: bool, threshold: f64) -> ResponseQualityEvaluator {
        ResponseQualityEvaluator::new(QualityEvaluatorConfig { enabled, threshold })
    }

    fn request(response_format: Option<Value>) -> OpenAIRequest {
        let mut extra = Map::new();
        if let Some(response_format) = response_format {
            extra.insert("response_format".to_owned(), response_format);
        }
        OpenAIRequest {
            model: "logical-model".to_owned(),
            messages: vec![Message {
                role: "user".to_owned(),
                content: json!("Provide a concise response."),
                extra: Map::new(),
            }],
            stream: false,
            temperature: None,
            max_tokens: None,
            extra,
        }
    }

    fn response(content: Value, finish_reason: &str) -> OpenAIResponse {
        OpenAIResponse {
            id: "response".to_owned(),
            object: "chat.completion".to_owned(),
            created: 0,
            model: "provider-model".to_owned(),
            choices: vec![Choice {
                index: 0,
                message: Message {
                    role: "assistant".to_owned(),
                    content,
                    extra: Map::new(),
                },
                finish_reason: Some(finish_reason.to_owned()),
                extra: Map::new(),
            }],
            usage: Usage {
                prompt_tokens: 20,
                completion_tokens: 20,
                total_tokens: 40,
                extra: Map::new(),
            },
            extra: Map::new(),
        }
    }

    proptest! {
    #![proptest_config(ProptestConfig::with_cases(64))]

    #[test]
    fn property_25_quality_score_is_finite_and_bounded(
    request in arb_request(),
    response in arb_response(),
    ) {
    let score = evaluator(true, 0.3).evaluate(&response, &request);

    prop_assert!(score.is_finite(), "quality score must be finite, got {score}");
    prop_assert!((0.0..=1.0).contains(&score), "quality score must be in [0, 1], got {score}");
    }
    }

    #[test]
    fn property_25_targeted_fault_and_success_cases_remain_finite_and_bounded() {
        let plain_request = request(None);
        let json_request = request(Some(json!({"type": "json_object"})));
        let mut high_logprobs = response(json!("Answer."), "stop");
        high_logprobs.choices[0].extra.insert(
            "logprobs".to_owned(),
            json!({"content": [{"token": "A", "logprob": -0.01}]}),
        );
        let mut low_logprobs = high_logprobs.clone();
        low_logprobs.choices[0].extra.insert(
            "logprobs".to_owned(),
            json!({"content": [{"token": "A", "logprob": -8.0}]}),
        );
        let cases = [
            (&json_request, response(json!(r#"{"ok":true}"#), "stop")),
            (&json_request, response(json!(r#"{"ok":true"#), "stop")),
            (
                &plain_request,
                response(json!("```rust\nfn main() { println!(\"ok\");"), "length"),
            ),
            (
                &plain_request,
                response(json!("A complete answer."), "content_filter"),
            ),
            (
                &plain_request,
                response(json!("A complete answer."), "stop"),
            ),
            (&plain_request, high_logprobs),
            (&plain_request, low_logprobs),
        ];

        for (request, response) in cases {
            let score = evaluator(true, 0.3).evaluate(&response, request);
            assert!(
                score.is_finite(),
                "quality score must be finite, got {score}"
            );
            assert!(
                (0.0..=1.0).contains(&score),
                "quality score must be in [0, 1], got {score}"
            );
        }
    }

    #[test]
    fn normal_completion_produces_finite_bounded_score() {
        let score = evaluator(true, 0.3).evaluate(
            &response(json!("Complete answer with (balanced) structure."), "stop"),
            &request(None),
        );
        assert!(score.is_finite());
        assert!((0.0..=1.0).contains(&score));
        assert!(score > 0.8);
    }

    #[test]
    fn invalid_json_scores_below_valid_json() {
        let evaluator = evaluator(true, 0.3);
        let request = request(Some(json!({"type": "json_object"})));
        let valid = evaluator.evaluate(&response(json!(r#"{"ok":true}"#), "stop"), &request);
        let invalid = evaluator.evaluate(&response(json!(r#"{"ok":true"#), "stop"), &request);
        assert!(valid > invalid);
    }

    #[test]
    fn strict_schema_reuses_validation_failure_without_compiling_schema() {
        let evaluator = evaluator(true, 0.3);
        let request = request(Some(json!({
            "type": "json_schema",
            "json_schema": {
                "strict": true,
                "schema": {"type": "object", "required": ["id"]}
            }
        })));
        let mut failed = response(json!(r#"{"id":1}"#), "stop");
        failed.extra.insert(
            "structured_output_validation".to_owned(),
            json!({"outcome": "fail"}),
        );
        let mut passed = failed.clone();
        passed.extra.insert(
            "structured_output_validation".to_owned(),
            json!({"outcome": "pass"}),
        );
        assert!(evaluator.evaluate(&passed, &request) > evaluator.evaluate(&failed, &request));
    }

    #[test]
    fn truncated_code_and_finish_reason_reduce_score() {
        let evaluator = evaluator(true, 0.3);
        let request = request(None);
        let complete = response(
            json!("```rust\nfn main() { println!(\"ok\"); }\n```"),
            "stop",
        );
        let truncated = response(json!("```rust\nfn main() { println!(\"ok\");"), "length");
        assert!(evaluator.evaluate(&complete, &request) > evaluator.evaluate(&truncated, &request));
    }

    #[test]
    fn content_filter_scores_below_normal_finish() {
        let evaluator = evaluator(true, 0.3);
        let request = request(None);
        let normal = response(json!("A complete answer."), "stop");
        let filtered = response(json!("A complete answer."), "content_filter");
        assert!(evaluator.evaluate(&normal, &request) > evaluator.evaluate(&filtered, &request));
    }

    #[test]
    fn average_token_logprobs_adjust_confidence() {
        let evaluator = evaluator(true, 0.3);
        let request = request(None);
        let mut confident = response(json!("Answer."), "stop");
        confident.choices[0].extra.insert(
            "logprobs".to_owned(),
            json!({"content": [{"token": "A", "logprob": -0.01}]}),
        );
        let mut uncertain = confident.clone();
        uncertain.choices[0].extra.insert(
            "logprobs".to_owned(),
            json!({"content": [{"token": "A", "logprob": -8.0}]}),
        );
        assert!(
            evaluator.evaluate(&confident, &request) > evaluator.evaluate(&uncertain, &request)
        );
    }

    #[test]
    fn required_tool_call_checks_presence_and_json_arguments() {
        let evaluator = evaluator(true, 0.3);
        let mut request = request(None);
        request.extra.insert(
            "tools".to_owned(),
            json!([{"type": "function", "function": {"name": "lookup", "strict": true}}]),
        );
        request
            .extra
            .insert("tool_choice".to_owned(), json!("required"));

        let missing = response(json!("I did not call the tool."), "stop");
        let mut present = response(Value::Null, "tool_calls");
        present.choices[0].message.extra.insert(
            "tool_calls".to_owned(),
            json!([{"function": {"name": "lookup", "arguments": "{\"id\":1}"}}]),
        );
        assert!(evaluator.evaluate(&present, &request) > evaluator.evaluate(&missing, &request));
    }

    #[test]
    fn flattened_usage_aliases_are_supported() {
        let evaluator = evaluator(true, 0.3);
        let request = request(None);
        let mut response = response(json!("Answer."), "stop");
        response.usage.prompt_tokens = 0;
        response.usage.completion_tokens = 0;
        response
            .usage
            .extra
            .insert("input_tokens".to_owned(), json!(100));
        response
            .usage
            .extra
            .insert("output_tokens".to_owned(), json!(25));
        let score = evaluator.evaluate(&response, &request);
        assert!(score.is_finite());
        assert!((0.0..=1.0).contains(&score));
    }

    #[test]
    fn low_quality_signal_respects_enabled_flag_and_sanitizes_threshold() {
        let disabled = evaluator(false, 0.8);
        assert!(!disabled.is_low_quality(0.1));

        let enabled = evaluator(true, 0.8);
        assert!(enabled.is_low_quality(0.79));
        assert!(!enabled.is_low_quality(0.8));
        assert!(enabled.is_low_quality(f64::NAN));

        let sanitized = evaluator(true, f64::NAN);
        assert_eq!(sanitized.threshold(), DEFAULT_QUALITY_THRESHOLD);
    }
}
