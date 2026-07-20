//! Aggressive compression engine.

use super::{
    lite::LiteEngine, standard::StandardEngine, CompressibleMessage, CompressiblePayload,
    CompressionContext, CompressionEngine, EngineResult, MessageContent,
};
use async_trait::async_trait;
use serde_json::Value;
use std::{collections::HashMap, time::Instant};

const TOOL_RESULT_DIGEST_TRIGGER_TOKENS: u32 = 500;
const RECENT_USER_MESSAGES_TO_PRESERVE: usize = 2;

#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord)]
enum Treatment {
    Untouched,
    Lite,
    Standard,
}

#[derive(Debug)]
struct TreatmentUnit {
    positions: Vec<usize>,
    treatment: Treatment,
    oldest_original_index: usize,
}

/// Age-aware compression for long conversations and large textual tool results.
#[derive(Debug, Clone, Copy, Default)]
pub struct AggressiveEngine;

impl AggressiveEngine {
    pub fn new() -> Self {
        Self
    }
}

#[async_trait]
impl CompressionEngine for AggressiveEngine {
    fn name(&self) -> &str {
        "aggressive"
    }

    async fn compress(
        &self,
        payload: &mut CompressiblePayload,
        context: &CompressionContext,
    ) -> EngineResult {
        let started = Instant::now();
        let original = payload.clone();
        let tokens_before = count_payload_tokens(&original, context);
        let target = compression_target(context);

        if target.is_some_and(|target| tokens_before <= target) {
            return EngineResult {
                engine_name: self.name().to_owned(),
                tokens_before,
                tokens_after: tokens_before,
                duration_ms: elapsed_millis(started),
                applied: false,
            };
        }

        let units = build_treatment_units(payload);
        let mut current_tokens = tokens_before;
        let mut changed = false;

        for unit in units {
            if unit.treatment == Treatment::Untouched {
                continue;
            }

            let mut candidate = payload.clone();
            if !apply_treatment_unit(&mut candidate, &unit, context).await {
                continue;
            }

            candidate.refresh_metadata();
            refresh_message_token_counts(&mut candidate, context);
            let candidate_tokens = count_payload_tokens(&candidate, context);
            if candidate_tokens > current_tokens {
                continue;
            }

            *payload = candidate;
            current_tokens = candidate_tokens;
            changed = true;

            if target.is_some_and(|target| current_tokens <= target) {
                break;
            }
        }

        if changed {
            payload.refresh_metadata();
            refresh_message_token_counts(payload, context);
        }

        let mut final_tokens = count_payload_tokens(payload, context);
        if final_tokens > tokens_before {
            *payload = original.clone();
            final_tokens = tokens_before;
        }
        let applied = *payload != original;

        EngineResult {
            engine_name: self.name().to_owned(),
            tokens_before,
            tokens_after: final_tokens,
            duration_ms: elapsed_millis(started),
            applied,
        }
    }
}

fn elapsed_millis(started: Instant) -> u64 {
    started.elapsed().as_millis().min(u64::MAX as u128) as u64
}

fn compression_target(context: &CompressionContext) -> Option<u32> {
    let window_target = (context.context_window > 0)
        .then(|| ((u64::from(context.context_window) * 9) / 10).min(u64::from(u32::MAX)) as u32);

    match (window_target, context.target_token_budget) {
        (Some(window), Some(budget)) => Some(window.min(budget)),
        (Some(window), None) => Some(window),
        (None, Some(budget)) => Some(budget),
        (None, None) => None,
    }
}

fn build_treatment_units(payload: &CompressiblePayload) -> Vec<TreatmentUnit> {
    let mut recent_users = payload
        .messages
        .iter()
        .enumerate()
        .rev()
        .filter(|(_, message)| message.role == "user")
        .take(RECENT_USER_MESSAGES_TO_PRESERVE)
        .map(|(position, _)| position)
        .collect::<Vec<_>>();
    recent_users.sort_unstable();

    let base_treatments = payload
        .messages
        .iter()
        .enumerate()
        .map(|(position, message)| {
            if message.is_system()
                || message.cache_protected
                || recent_users.binary_search(&position).is_ok()
                || !message.relationships.unresolved_tool_call_ids.is_empty()
            {
                Treatment::Untouched
            } else {
                treatment_for_age(message.age)
            }
        })
        .collect::<Vec<_>>();

    let positions_by_original_index = payload
        .messages
        .iter()
        .enumerate()
        .map(|(position, message)| (message.original_index, position))
        .collect::<HashMap<_, _>>();
    let mut adjacency = vec![Vec::new(); payload.messages.len()];
    for (position, message) in payload.messages.iter().enumerate() {
        for related_original_index in &message.relationships.related_message_indices {
            let Some(&related_position) = positions_by_original_index.get(related_original_index)
            else {
                continue;
            };
            push_unique(&mut adjacency[position], related_position);
            push_unique(&mut adjacency[related_position], position);
        }
    }

    let mut visited = vec![false; payload.messages.len()];
    let mut units = Vec::with_capacity(payload.messages.len());
    for start in 0..payload.messages.len() {
        if visited[start] {
            continue;
        }

        let mut stack = vec![start];
        let mut positions = Vec::new();
        visited[start] = true;
        while let Some(position) = stack.pop() {
            positions.push(position);
            for &related in &adjacency[position] {
                if !visited[related] {
                    visited[related] = true;
                    stack.push(related);
                }
            }
        }

        positions.sort_unstable_by_key(|position| payload.messages[*position].original_index);
        let treatment = positions
            .iter()
            .map(|position| base_treatments[*position])
            .min()
            .unwrap_or(Treatment::Untouched);
        let oldest_original_index = positions
            .iter()
            .map(|position| payload.messages[*position].original_index)
            .min()
            .unwrap_or(usize::MAX);
        units.push(TreatmentUnit {
            positions,
            treatment,
            oldest_original_index,
        });
    }

    units.sort_unstable_by_key(|unit| unit.oldest_original_index);
    units
}

fn treatment_for_age(age: usize) -> Treatment {
    match age {
        0..=2 => Treatment::Untouched,
        3..=6 => Treatment::Lite,
        _ => Treatment::Standard,
    }
}

async fn apply_treatment_unit(
    payload: &mut CompressiblePayload,
    unit: &TreatmentUnit,
    context: &CompressionContext,
) -> bool {
    let selected = unit
        .positions
        .iter()
        .copied()
        .map(|position| (position, true))
        .collect::<HashMap<_, _>>();
    let digest_positions = unit
        .positions
        .iter()
        .copied()
        .filter(|position| should_digest_tool_result(&payload.messages[*position], context))
        .collect::<Vec<_>>();
    let preserved_tool_results = unit
        .positions
        .iter()
        .copied()
        .filter(|position| is_structured_tool_result(&payload.messages[*position]))
        .collect::<Vec<_>>();

    let mut transformed = payload.clone();
    for (position, message) in transformed.messages.iter_mut().enumerate() {
        message.cache_protected = !selected.contains_key(&position)
            || digest_positions.contains(&position)
            || preserved_tool_results.contains(&position);
    }

    match unit.treatment {
        Treatment::Untouched => return false,
        Treatment::Lite => {
            LiteEngine::new().compress(&mut transformed, context).await;
        }
        Treatment::Standard => {
            StandardEngine::new()
                .compress(&mut transformed, context)
                .await;
        }
    }

    let mut changed = false;
    for &position in &unit.positions {
        if digest_positions.contains(&position) || preserved_tool_results.contains(&position) {
            continue;
        }
        if transformed.messages[position].content != payload.messages[position].content {
            payload.messages[position].content = transformed.messages[position].content.clone();
            changed = true;
        }
    }

    for position in digest_positions {
        let tool_name = tool_name_for_unit(payload, unit, position);
        let status = tool_status(&payload.messages[position]);
        let Some(text) = payload.messages[position].content.as_text() else {
            continue;
        };
        let digest = build_tool_result_digest(text, &tool_name, status, context);
        if digest != text {
            payload.messages[position].content = MessageContent::new(Value::String(digest));
            changed = true;
        }
    }

    changed
}

fn is_tool_result(message: &CompressibleMessage) -> bool {
    matches!(message.role.as_str(), "tool" | "function")
        || !message.relationships.tool_result_for_ids.is_empty()
}

fn is_structured_tool_result(message: &CompressibleMessage) -> bool {
    if !is_tool_result(message) {
        return false;
    }

    match message.content.as_value() {
        Value::String(text) => serde_json::from_str::<Value>(text.trim())
            .is_ok_and(|value| matches!(value, Value::Array(_) | Value::Object(_))),
        _ => true,
    }
}

fn should_digest_tool_result(message: &CompressibleMessage, context: &CompressionContext) -> bool {
    if !is_tool_result(message) || is_structured_tool_result(message) {
        return false;
    }
    let Some(text) = message.content.as_text() else {
        return false;
    };
    context.token_counter.count_text(&context.model, text) > TOOL_RESULT_DIGEST_TRIGGER_TOKENS
}

fn tool_name_for_unit(
    payload: &CompressiblePayload,
    unit: &TreatmentUnit,
    result_position: usize,
) -> String {
    payload.messages[result_position]
        .extra
        .get("name")
        .and_then(Value::as_str)
        .or_else(|| {
            unit.positions.iter().find_map(|position| {
                payload.messages[*position]
                    .relationships
                    .tool_names
                    .first()
                    .map(String::as_str)
            })
        })
        .or_else(|| {
            unit.positions.iter().find_map(|position| {
                payload.messages[*position]
                    .extra
                    .get("name")
                    .and_then(Value::as_str)
            })
        })
        .unwrap_or("unknown")
        .to_owned()
}

fn tool_status(message: &CompressibleMessage) -> &'static str {
    if value_indicates_error(&Value::Object(message.extra.clone()))
        || message.content.as_text().is_some_and(text_indicates_error)
    {
        "error"
    } else {
        "success"
    }
}

fn value_indicates_error(value: &Value) -> bool {
    match value {
        Value::Array(values) => values.iter().any(value_indicates_error),
        Value::Object(object) => object.iter().any(|(key, value)| {
            let key = key.to_ascii_lowercase();
            (key == "is_error" && value == &Value::Bool(true))
                || (key == "error" && !value.is_null() && value != &Value::Bool(false))
                || ((key == "status" || key == "state")
                    && value.as_str().is_some_and(|status| {
                        matches!(
                            status.to_ascii_lowercase().as_str(),
                            "error" | "failed" | "failure"
                        )
                    }))
                || value_indicates_error(value)
        }),
        _ => false,
    }
}

fn text_indicates_error(text: &str) -> bool {
    text.lines().any(|line| {
        let line = line.trim_start().to_ascii_lowercase();
        line.starts_with("error")
            || line.starts_with("failed")
            || line.starts_with("failure")
            || line.contains("status: error")
            || line.contains("status=error")
    })
}

fn build_tool_result_digest(
    text: &str,
    tool_name: &str,
    status: &str,
    context: &CompressionContext,
) -> String {
    let lines = text.split('\n').collect::<Vec<_>>();
    let first_lines = lines.iter().take(3).copied().collect::<Vec<_>>();
    let last_start = lines.len().saturating_sub(3);
    let last_lines = lines[last_start..].to_vec();
    let mut digest = format!(
        "[tool result digest]\ntool_name: {tool_name}\nstatus: {status}\noriginal_byte_count: {}\nfirst_3_lines:\n",
        text.len()
    );
    append_digest_lines(&mut digest, &first_lines);
    digest.push_str("last_3_lines:\n");
    append_digest_lines(&mut digest, &last_lines);

    let protected = context.protection_scanner.scan(text);
    let missing_protected = protected
        .into_iter()
        .map(|range| &text[range])
        .filter(|region| !region.is_empty() && !digest.contains(region))
        .collect::<Vec<_>>();
    if !missing_protected.is_empty() {
        digest.push_str("protected_regions:\n");
        for region in missing_protected {
            digest.push_str("--- protected region ---\n");
            digest.push_str(region);
            if !region.ends_with('\n') {
                digest.push('\n');
            }
        }
    }

    digest
}

fn append_digest_lines(output: &mut String, lines: &[&str]) {
    for line in lines {
        output.push_str(line);
        output.push('\n');
    }
}

fn count_payload_tokens(payload: &CompressiblePayload, context: &CompressionContext) -> u32 {
    context
        .token_counter
        .count_request(&payload.clone().into_openai_request())
}

fn refresh_message_token_counts(payload: &mut CompressiblePayload, context: &CompressionContext) {
    for message in &mut payload.messages {
        let content_tokens = match message.content.as_value() {
            Value::Null => 0,
            Value::String(text) => context.token_counter.count_text(&context.model, text),
            structured => context
                .token_counter
                .count_text(&context.model, &structured.to_string()),
        };
        let extra_tokens = if message.extra.is_empty() {
            0
        } else {
            context.token_counter.count_text(
                &context.model,
                &Value::Object(message.extra.clone()).to_string(),
            )
        };
        message.token_count = 4u32
            .saturating_add(
                context
                    .token_counter
                    .count_text(&context.model, &message.role),
            )
            .saturating_add(content_tokens)
            .saturating_add(extra_tokens);
    }
}

fn push_unique(values: &mut Vec<usize>, value: usize) {
    if !values.contains(&value) {
        values.push(value);
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::models::openai::OpenAIRequest;
    use serde_json::{json, Map};

    fn payload(value: Value) -> CompressiblePayload {
        let request: OpenAIRequest = serde_json::from_value(value).unwrap();
        CompressiblePayload::from(request)
    }

    fn context() -> CompressionContext {
        CompressionContext::new("gpt-4o", "test")
    }

    fn aging_payload(turns: usize) -> CompressiblePayload {
        let mut messages =
            vec![json!({"role": "system", "content": "System  policy stays exact."})];
        for turn in 0..turns {
            let age = turns - turn - 1;
            messages.push(json!({
                "role": "user",
                "name": format!("user-{turn}"),
                "content": format!("User age {age}: Please   actually keep this request.")
            }));
            messages.push(json!({
                "role": "assistant",
                "name": format!("assistant-{turn}"),
                "content": format!("Assistant age {age}: Please   actually use this response.")
            }));
        }
        payload(json!({"model": "gpt-4o", "messages": messages}))
    }

    fn message_at_age<'a>(
        payload: &'a CompressiblePayload,
        role: &str,
        age: usize,
    ) -> &'a CompressibleMessage {
        payload
            .messages
            .iter()
            .find(|message| message.role == role && message.age == age)
            .unwrap()
    }

    #[tokio::test]
    async fn applies_progressive_age_buckets_without_exceeding_each_bucket() {
        let mut payload = aging_payload(10);
        let age_zero = message_at_age(&payload, "assistant", 0).content.clone();
        let age_two = message_at_age(&payload, "assistant", 2).content.clone();

        let result = AggressiveEngine::new()
            .compress(&mut payload, &context())
            .await;

        assert!(result.applied);
        assert_eq!(message_at_age(&payload, "assistant", 0).content, age_zero);
        assert_eq!(message_at_age(&payload, "assistant", 2).content, age_two);
        assert_eq!(
            message_at_age(&payload, "assistant", 3).content.as_text(),
            Some("Assistant age 3: Please actually use this response.")
        );
        assert_eq!(
            message_at_age(&payload, "assistant", 6).content.as_text(),
            Some("Assistant age 6: Please actually use this response.")
        );
        assert_eq!(
            message_at_age(&payload, "assistant", 7).content.as_text(),
            Some("Assistant age 7: use this response.")
        );
    }

    #[tokio::test]
    async fn preserves_system_recent_users_and_cache_protected_messages_exactly() {
        let mut messages =
            vec![json!({"role": "system", "content": "Please   actually preserve system."})];
        for turn in 0..8 {
            let content = if turn == 0 {
                json!([{"type": "text", "text": "Please   preserve cached user.", "cache_control": {"type": "ephemeral"}}])
            } else {
                json!(format!("Please   actually preserve user {turn}."))
            };
            messages
                .push(json!({"role": "user", "name": format!("user-{turn}"), "content": content}));
            messages.push(
                json!({"role": "assistant", "content": "Please   actually compress response."}),
            );
        }
        let mut payload = payload(json!({"model": "gpt-4o", "messages": messages}));
        let system = payload.messages[0].clone();
        let cached = payload.messages[1].clone();
        let recent_users = payload
            .messages
            .iter()
            .filter(|message| message.role == "user")
            .rev()
            .take(2)
            .cloned()
            .collect::<Vec<_>>();

        AggressiveEngine::new()
            .compress(&mut payload, &context())
            .await;

        assert_eq!(payload.messages[0].content, system.content);
        assert_eq!(payload.messages[0].extra, system.extra);
        assert_eq!(payload.messages[1].content, cached.content);
        assert_eq!(payload.messages[1].extra, cached.extra);
        for original in recent_users {
            let current = payload
                .messages
                .iter()
                .find(|message| message.original_index == original.original_index)
                .unwrap();
            assert_eq!(current.content, original.content);
            assert_eq!(current.extra, original.extra);
        }
    }

    #[tokio::test]
    async fn summarizes_large_text_tool_result_with_required_digest_fields() {
        let mut lines = (0..8)
            .map(|line| format!("line {line}: {}", "output ".repeat(100)))
            .collect::<Vec<_>>();
        lines[3] = format!("ERROR: command failed {}", "detail ".repeat(100));
        lines[4].push_str(" https://example.test/protected/path");
        let output = lines.join("\n");
        let original_bytes = output.len();
        let mut messages = vec![
            json!({
                "role": "assistant",
                "content": "Please actually call lookup.",
                "tool_calls": [{"id": "call-1", "type": "function", "function": {"name": "lookup", "arguments": "{}"}}]
            }),
            json!({
                "role": "tool",
                "tool_call_id": "call-1",
                "name": "lookup",
                "content": output
            }),
        ];
        for turn in 0..8 {
            messages.push(json!({"role": "user", "content": format!("later {turn}")}));
        }
        let mut payload = payload(json!({"model": "gpt-4o", "messages": messages}));

        let result = AggressiveEngine::new()
            .compress(&mut payload, &context())
            .await;
        let digest = payload.messages[1].content.as_text().unwrap();

        assert!(result.applied);
        assert!(digest.starts_with("[tool result digest]"));
        assert!(digest.contains("tool_name: lookup"));
        assert!(digest.contains("status: error"));
        assert!(digest.contains(&format!("original_byte_count: {original_bytes}")));
        for line in &lines[..3] {
            assert!(digest.contains(line));
        }
        for line in &lines[lines.len() - 3..] {
            assert!(digest.contains(line));
        }
        assert!(digest.contains("https://example.test/protected/path"));
    }

    #[tokio::test]
    async fn linked_tool_messages_use_same_least_aggressive_treatment_bucket() {
        let mut messages = vec![
            json!({"role": "user", "content": "first"}),
            json!({
                "role": "assistant",
                "content": "Please   actually call the tool.",
                "tool_calls": [{"id": "call-pair", "type": "function", "function": {"name": "pair_lookup", "arguments": "{}"}}]
            }),
            json!({"role": "user", "content": "second"}),
            json!({
                "role": "tool",
                "tool_call_id": "call-pair",
                "name": "pair_lookup",
                "content": "Please   actually return the result."
            }),
        ];
        for turn in 2..8 {
            messages.push(json!({"role": "user", "content": format!("later {turn}")}));
        }
        let mut payload = payload(json!({"model": "gpt-4o", "messages": messages}));
        let original_indices = payload
            .messages
            .iter()
            .map(|message| message.original_index)
            .collect::<Vec<_>>();
        let original_extras = payload
            .messages
            .iter()
            .map(|message| message.extra.clone())
            .collect::<Vec<_>>();
        assert_eq!(payload.messages[1].age, 7);
        assert_eq!(payload.messages[3].age, 6);

        AggressiveEngine::new()
            .compress(&mut payload, &context())
            .await;

        assert_eq!(
            payload.messages[1].content.as_text(),
            Some("Please actually call the tool.")
        );
        assert_eq!(
            payload.messages[3].content.as_text(),
            Some("Please actually return the result.")
        );
        assert_eq!(
            payload
                .messages
                .iter()
                .map(|message| message.original_index)
                .collect::<Vec<_>>(),
            original_indices
        );
        assert_eq!(
            payload
                .messages
                .iter()
                .map(|message| message.extra.clone())
                .collect::<Vec<_>>(),
            original_extras
        );
        assert_eq!(
            payload.messages[1].relationships.related_message_indices,
            [3]
        );
        assert_eq!(
            payload.messages[3].relationships.related_message_indices,
            [1]
        );
    }

    #[tokio::test]
    async fn processes_oldest_units_first_and_stops_at_context_budget() {
        let mut payload = aging_payload(10);
        for message in &mut payload.messages {
            if message.age >= 7 && !message.is_system() {
                message.content = MessageContent::new(Value::String(format!(
                    "Please actually use this very verbose content in order to finish. {}",
                    "padding ".repeat(80)
                )));
            }
        }
        let base_context = context();
        let units = build_treatment_units(&payload);
        let first_unit = units
            .iter()
            .find(|unit| unit.treatment != Treatment::Untouched)
            .unwrap();
        let first_position = first_unit.positions[0];
        let second_position = units
            .iter()
            .filter(|unit| unit.treatment != Treatment::Untouched)
            .nth(1)
            .unwrap()
            .positions[0];
        let first_original = payload.messages[first_position].content.clone();
        let second_original = payload.messages[second_position].content.clone();
        let before = count_payload_tokens(&payload, &base_context);
        let mut after_first_payload = payload.clone();
        assert!(apply_treatment_unit(&mut after_first_payload, first_unit, &base_context).await);
        after_first_payload.refresh_metadata();
        refresh_message_token_counts(&mut after_first_payload, &base_context);
        let after_first = count_payload_tokens(&after_first_payload, &base_context);
        assert!(after_first < before);

        let mut context = base_context;
        let mut context_window = after_first.saturating_mul(10) / 9;
        while (u64::from(context_window) * 9 / 10) < u64::from(after_first) {
            context_window = context_window.saturating_add(1);
        }
        context.context_window = context_window;
        let target = compression_target(&context).unwrap();
        assert!(after_first <= target);
        assert!(target < before);

        AggressiveEngine::new()
            .compress(&mut payload, &context)
            .await;

        assert_ne!(payload.messages[first_position].content, first_original);
        assert_eq!(payload.messages[second_position].content, second_original);
        assert!(count_payload_tokens(&payload, &context) <= target);
    }

    #[tokio::test]
    async fn preserves_structured_and_textual_json_tool_outputs() {
        let structured =
            json!({"rows": [{"value": "Please actually keep this structured output".repeat(100)}]});
        let textual = serde_json::to_string(&structured).unwrap();
        let mut messages = vec![
            json!({"role": "tool", "tool_call_id": "structured", "name": "lookup", "content": structured}),
            json!({"role": "tool", "tool_call_id": "text-json", "name": "lookup", "content": textual}),
        ];
        for turn in 0..8 {
            messages.push(json!({"role": "user", "content": format!("later {turn}")}));
        }
        let mut payload = payload(json!({"model": "gpt-4o", "messages": messages}));
        let original_structured = payload.messages[0].content.clone();
        let original_textual = payload.messages[1].content.clone();

        AggressiveEngine::new()
            .compress(&mut payload, &context())
            .await;

        assert_eq!(payload.messages[0].content, original_structured);
        assert_eq!(payload.messages[1].content, original_textual);
    }

    #[tokio::test]
    async fn preserves_active_unresolved_tool_calls_exactly() {
        let mut messages = vec![json!({
            "role": "assistant",
            "name": "caller",
            "content": "Please   actually keep this unresolved call exactly.",
            "tool_calls": [{"id": "active-call", "type": "function", "function": {"name": "lookup", "arguments": "{\"id\":1}"}}]
        })];
        for turn in 0..8 {
            messages.push(json!({"role": "user", "content": format!("later {turn}")}));
        }
        let mut payload = payload(json!({"model": "gpt-4o", "messages": messages}));
        let active = payload.messages[0].clone();
        assert_eq!(
            active.relationships.unresolved_tool_call_ids,
            ["active-call"]
        );

        AggressiveEngine::new()
            .compress(&mut payload, &context())
            .await;

        assert_eq!(payload.messages[0].content, active.content);
        assert_eq!(payload.messages[0].extra, active.extra);
        assert_eq!(payload.messages[0].original_index, active.original_index);
        assert!(payload.messages[0].critical);
    }

    #[tokio::test]
    async fn reports_full_request_counts_and_never_increases_tokens() {
        let cases = [
            json!({"model": "gpt-4o", "messages": [{"role": "user", "content": "Already concise."}]}),
            json!({"model": "gpt-4o", "messages": [{"role": "system", "content": "Please   preserve."}, {"role": "user", "content": "Recent   exact."}]}),
            serde_json::to_value(aging_payload(10).into_openai_request()).unwrap(),
        ];
        let context = context();

        for case in cases {
            let mut payload = payload(case);
            let original = payload.clone();
            let expected_before = count_payload_tokens(&original, &context);
            let result = AggressiveEngine::new()
                .compress(&mut payload, &context)
                .await;
            let actual_after = count_payload_tokens(&payload, &context);

            assert_eq!(result.engine_name, "aggressive");
            assert_eq!(result.tokens_before, expected_before);
            assert_eq!(result.tokens_after, actual_after);
            assert!(result.tokens_after <= result.tokens_before);
            assert_eq!(result.applied, payload != original);
        }
    }

    #[tokio::test]
    async fn zero_context_and_no_target_processes_every_eligible_unit() {
        let mut payload = aging_payload(10);
        let eligible_before = payload
            .messages
            .iter()
            .filter(|message| {
                !message.is_system() && ((3..=6).contains(&message.age) || message.age >= 7)
            })
            .map(|message| (message.original_index, message.content.clone()))
            .collect::<HashMap<_, _>>();
        let context = context();
        assert_eq!(context.context_window, 0);
        assert_eq!(context.target_token_budget, None);

        AggressiveEngine::new()
            .compress(&mut payload, &context)
            .await;

        for message in payload.messages.iter().filter(|message| {
            !message.is_system() && ((3..=6).contains(&message.age) || message.age >= 7)
        }) {
            assert_ne!(
                message.content, eligible_before[&message.original_index],
                "eligible original index {} was not processed",
                message.original_index
            );
        }
    }

    #[test]
    fn target_budget_further_constrains_context_window() {
        let mut context = context();
        context.context_window = 10_000;
        context.target_token_budget = Some(7_500);
        assert_eq!(compression_target(&context), Some(7_500));
        context.target_token_budget = Some(9_500);
        assert_eq!(compression_target(&context), Some(9_000));
    }

    #[test]
    fn digest_status_uses_structured_error_metadata() {
        let mut extra = Map::new();
        extra.insert("status".to_owned(), json!("failed"));
        let message = CompressibleMessage {
            role: "tool".to_owned(),
            content: MessageContent::new(json!("ordinary output")),
            extra,
            ..CompressibleMessage::default()
        };
        assert_eq!(tool_status(&message), "error");
    }
}
