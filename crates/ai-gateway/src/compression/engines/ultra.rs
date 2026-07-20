//! Ultra compression engine.

use super::{
    CompressibleMessage, CompressiblePayload, CompressionContext, CompressionEngine, EngineResult,
};
use async_trait::async_trait;
use regex::Regex;
use serde_json::Value;
use std::{
    collections::{HashMap, HashSet},
    sync::OnceLock,
    time::Instant,
};

const DEFAULT_RELEVANCE_THRESHOLD: f32 = 0.3;
const RECENCY_WEIGHT: f32 = 0.5;
const RECENCY_DECAY_RATE: f32 = 0.55;
const CODE_BLOCK_BONUS: f32 = 0.18;
const RECENT_FILE_PATH_BONUS: f32 = 0.22;
const RECENT_PATH_MAX_AGE: usize = 2;
const CODE_BLOCK_LINE_THRESHOLD: usize = 50;
const CODE_EDGE_LINES: usize = 10;
const OMITTED_MARKER_PREFIX: &str = "[... ";
const OMITTED_MARKER_SUFFIX: &str = " lines omitted ...]";

/// Maximum-compression engine using deterministic relevance pruning and truncation.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct UltraEngine {
    /// Messages scoring below this value are eligible for removal.
    pub relevance_threshold: f32,
}

impl Default for UltraEngine {
    fn default() -> Self {
        Self {
            relevance_threshold: DEFAULT_RELEVANCE_THRESHOLD,
        }
    }
}

impl UltraEngine {
    /// Creates an engine with the default relevance threshold of `0.3`.
    pub fn new() -> Self {
        Self::default()
    }

    /// Creates an engine with a request-independent relevance threshold.
    pub fn with_relevance_threshold(relevance_threshold: f32) -> Self {
        Self {
            relevance_threshold: normalize_threshold(relevance_threshold),
        }
    }

    fn threshold(&self) -> f32 {
        normalize_threshold(self.relevance_threshold)
    }
}

#[async_trait]
impl CompressionEngine for UltraEngine {
    fn name(&self) -> &str {
        "ultra"
    }

    async fn compress(
        &self,
        payload: &mut CompressiblePayload,
        context: &CompressionContext,
    ) -> EngineResult {
        let started = Instant::now();
        let original = payload.clone();
        let tokens_before = count_payload_tokens(&original, context);

        payload.refresh_metadata();
        prune_by_relevance(payload, self.threshold());
        thin_large_code_blocks(payload);
        payload.refresh_metadata();
        refresh_message_token_counts(payload, context);

        if let Some(target) = compression_target(context) {
            if count_payload_tokens(payload, context) > target {
                binary_search_oldest_cut(payload, context, target);
            }
        }

        payload.refresh_metadata();
        refresh_message_token_counts(payload, context);
        let mut tokens_after = count_payload_tokens(payload, context);
        if tokens_after > tokens_before {
            *payload = original.clone();
            tokens_after = tokens_before;
        }

        EngineResult {
            engine_name: self.name().to_owned(),
            tokens_before,
            tokens_after,
            duration_ms: elapsed_millis(started),
            applied: *payload != original,
        }
    }
}

fn normalize_threshold(threshold: f32) -> f32 {
    if threshold.is_finite() {
        threshold.clamp(0.0, 1.0)
    } else {
        DEFAULT_RELEVANCE_THRESHOLD
    }
}

fn elapsed_millis(started: Instant) -> u64 {
    started.elapsed().as_millis().min(u64::MAX as u128) as u64
}

fn compression_target(context: &CompressionContext) -> Option<u32> {
    match (context.context_window, context.target_token_budget) {
        (0, None) => None,
        (0, Some(budget)) => Some(budget),
        (window, None) => Some(window),
        (window, Some(budget)) => Some(window.min(budget)),
    }
}

fn prune_by_relevance(payload: &mut CompressiblePayload, threshold: f32) {
    if payload.messages.is_empty() {
        return;
    }

    let recent_paths = recent_file_paths(payload);
    let latest_user = latest_user_original_index(payload);
    let scores = payload
        .messages
        .iter()
        .map(|message| relevance_score(message, &recent_paths))
        .collect::<Vec<_>>();
    let groups = relationship_groups(payload);
    let mut retained = vec![false; payload.messages.len()];

    for group in groups {
        let protected = group
            .iter()
            .any(|position| must_preserve(&payload.messages[*position], latest_user));
        let orphan_result = group.len() == 1
            && is_tool_result(&payload.messages[group[0]])
            && payload.messages[group[0]]
                .relationships
                .related_message_indices
                .is_empty();
        let relevant = group.iter().any(|position| scores[*position] >= threshold);
        let keep_group = protected || (!orphan_result && relevant);
        if keep_group {
            for position in group {
                retained[position] = true;
            }
        }
    }

    payload.messages = payload
        .messages
        .drain(..)
        .enumerate()
        .filter_map(|(position, message)| retained[position].then_some(message))
        .collect();
    payload.refresh_metadata();
}

fn relevance_score(message: &CompressibleMessage, recent_paths: &HashSet<String>) -> f32 {
    if message.is_system() {
        return 1.0;
    }

    let age = message.age.min(10_000) as f32;
    let recency = RECENCY_WEIGHT * (-RECENCY_DECAY_RATE * age).exp();
    let role = match message.role.as_str() {
        "user" => 0.20,
        "assistant" => 0.12,
        "tool" | "function" => 0.10,
        _ => 0.08,
    };
    let code = if content_has_code_block(message.content.as_value()) {
        CODE_BLOCK_BONUS
    } else {
        0.0
    };
    let path_overlap = if message_file_paths(message)
        .iter()
        .any(|path| recent_paths.contains(path))
    {
        RECENT_FILE_PATH_BONUS
    } else {
        0.0
    };

    (recency + role + code + path_overlap).min(0.99)
}

fn recent_file_paths(payload: &CompressiblePayload) -> HashSet<String> {
    payload
        .messages
        .iter()
        .filter(|message| !message.is_system() && message.age <= RECENT_PATH_MAX_AGE)
        .flat_map(message_file_paths)
        .collect()
}

fn message_file_paths(message: &CompressibleMessage) -> HashSet<String> {
    let mut paths = HashSet::new();
    for text in visible_text_leaves(message.content.as_value()) {
        for captures in file_path_regex().captures_iter(text) {
            if let Some(path) = captures.name("path") {
                paths.insert(
                    path.as_str()
                        .trim_end_matches(['.', ',', ':', ';', '!', '?'])
                        .to_owned(),
                );
            }
        }
    }
    paths
}

fn file_path_regex() -> &'static Regex {
    static FILE_PATH_REGEX: OnceLock<Regex> = OnceLock::new();
    FILE_PATH_REGEX.get_or_init(|| {
        Regex::new(
            r#"(?x)(?:^|[\s\(\[\{\"'])
            (?P<path>
                [A-Za-z]:\\[^\s\"'<>|]+ |
                (?:\.\.?/|/)[A-Za-z0-9_.-]+(?:/[A-Za-z0-9_.-]+)+ |
                [A-Za-z0-9_.-]+(?:/[A-Za-z0-9_.-]+)+\.[A-Za-z0-9_-]+
            )"#,
        )
        .expect("UltraEngine file-path regex must compile")
    })
}

fn visible_text_leaves(value: &Value) -> Vec<&str> {
    let mut leaves = Vec::new();
    collect_visible_text_leaves(value, true, &mut leaves);
    leaves
}

fn collect_visible_text_leaves<'a>(value: &'a Value, root: bool, leaves: &mut Vec<&'a str>) {
    match value {
        Value::String(text) if root => leaves.push(text),
        Value::Array(parts) if root => {
            for part in parts {
                match part {
                    Value::String(text) => leaves.push(text),
                    Value::Object(_) => collect_content_block_text(part, leaves),
                    _ => {}
                }
            }
        }
        Value::Object(_) if root => collect_content_block_text(value, leaves),
        _ => {}
    }
}

fn collect_content_block_text<'a>(value: &'a Value, leaves: &mut Vec<&'a str>) {
    let Some(object) = value.as_object() else {
        return;
    };
    match object.get("type").and_then(Value::as_str) {
        Some("text" | "input_text" | "output_text") => {
            if let Some(text) = object.get("text").and_then(Value::as_str) {
                leaves.push(text);
            }
        }
        Some("tool_result") => {
            if let Some(content) = object.get("content") {
                collect_visible_text_leaves(content, true, leaves);
            }
        }
        _ => {}
    }
}

fn content_has_code_block(value: &Value) -> bool {
    visible_text_leaves(value)
        .iter()
        .any(|text| contains_fenced_code_block(text))
}

fn contains_fenced_code_block(text: &str) -> bool {
    let lines = lines_preserving_endings(text);
    let mut position = 0;
    while position < lines.len() {
        if let Some((fence, width)) = opening_fence(lines[position]) {
            if lines[position + 1..]
                .iter()
                .any(|line| is_closing_fence(line, fence, width))
            {
                return true;
            }
        }
        position += 1;
    }
    false
}

fn must_preserve(message: &CompressibleMessage, latest_user: Option<usize>) -> bool {
    message.is_system()
        || latest_user == Some(message.original_index)
        || message.cache_protected
        || !message.relationships.unresolved_tool_call_ids.is_empty()
}

fn latest_user_original_index(payload: &CompressiblePayload) -> Option<usize> {
    payload
        .messages
        .iter()
        .rev()
        .find(|message| message.role == "user")
        .map(|message| message.original_index)
}

fn relationship_groups(payload: &CompressiblePayload) -> Vec<Vec<usize>> {
    let positions_by_original_index = payload
        .messages
        .iter()
        .enumerate()
        .map(|(position, message)| (message.original_index, position))
        .collect::<HashMap<_, _>>();
    let mut adjacency = vec![Vec::new(); payload.messages.len()];

    for (position, message) in payload.messages.iter().enumerate() {
        for original_index in &message.relationships.related_message_indices {
            let Some(&related) = positions_by_original_index.get(original_index) else {
                continue;
            };
            push_unique(&mut adjacency[position], related);
            push_unique(&mut adjacency[related], position);
        }
    }

    let mut visited = vec![false; payload.messages.len()];
    let mut groups = Vec::new();
    for start in 0..payload.messages.len() {
        if visited[start] {
            continue;
        }
        visited[start] = true;
        let mut stack = vec![start];
        let mut group = Vec::new();
        while let Some(position) = stack.pop() {
            group.push(position);
            for &related in &adjacency[position] {
                if !visited[related] {
                    visited[related] = true;
                    stack.push(related);
                }
            }
        }
        group.sort_unstable();
        groups.push(group);
    }
    groups
}

fn is_tool_result(message: &CompressibleMessage) -> bool {
    matches!(message.role.as_str(), "tool" | "function")
        || !message.relationships.tool_result_for_ids.is_empty()
}

fn thin_large_code_blocks(payload: &mut CompressiblePayload) {
    for message in &mut payload.messages {
        if message.cache_protected {
            continue;
        }
        message
            .content
            .transform_text_leaves(thin_large_code_blocks_in_text);
    }
}

fn thin_large_code_blocks_in_text(text: &str) -> String {
    let lines = lines_preserving_endings(text);
    let mut output = String::with_capacity(text.len());
    let mut position = 0;

    while position < lines.len() {
        let Some((fence, width)) = opening_fence(lines[position]) else {
            output.push_str(lines[position]);
            position += 1;
            continue;
        };
        let Some(close_offset) = lines[position + 1..]
            .iter()
            .position(|line| is_closing_fence(line, fence, width))
        else {
            output.push_str(lines[position]);
            position += 1;
            continue;
        };
        let close = position + 1 + close_offset;
        let content_start = position + 1;
        let content_lines = close.saturating_sub(content_start);

        if content_lines <= CODE_BLOCK_LINE_THRESHOLD {
            for line in &lines[position..=close] {
                output.push_str(line);
            }
            position = close + 1;
            continue;
        }

        let first_end = content_start + CODE_EDGE_LINES;
        let last_start = close - CODE_EDGE_LINES;
        let signature_positions = signature_line_positions(&lines, first_end, last_start);
        let retained = CODE_EDGE_LINES * 2 + signature_positions.len();
        let omitted = content_lines.saturating_sub(retained);
        if omitted == 0 {
            for line in &lines[position..=close] {
                output.push_str(line);
            }
            position = close + 1;
            continue;
        }

        output.push_str(lines[position]);
        for line in &lines[content_start..first_end] {
            output.push_str(line);
        }
        for signature_position in signature_positions {
            output.push_str(lines[signature_position]);
        }
        output.push_str(OMITTED_MARKER_PREFIX);
        output.push_str(&omitted.to_string());
        output.push_str(OMITTED_MARKER_SUFFIX);
        output.push_str(preferred_line_ending(
            &lines[content_start..close],
            lines[position],
        ));
        for line in &lines[last_start..close] {
            output.push_str(line);
        }
        output.push_str(lines[close]);
        position = close + 1;
    }

    output
}

fn lines_preserving_endings(text: &str) -> Vec<&str> {
    let mut lines = Vec::new();
    let mut start = 0;
    for (position, character) in text.char_indices() {
        if character == '\n' {
            lines.push(&text[start..=position]);
            start = position + 1;
        }
    }
    if start < text.len() {
        lines.push(&text[start..]);
    }
    lines
}

fn line_body(line: &str) -> &str {
    line.strip_suffix("\r\n")
        .or_else(|| line.strip_suffix('\n'))
        .unwrap_or(line)
}

fn opening_fence(line: &str) -> Option<(char, usize)> {
    let trimmed = line_body(line).trim_start();
    let fence = trimmed.chars().next()?;
    if !matches!(fence, '`' | '~') {
        return None;
    }
    let width = trimmed
        .chars()
        .take_while(|character| *character == fence)
        .count();
    (width >= 3).then_some((fence, width))
}

fn is_closing_fence(line: &str, fence: char, minimum_width: usize) -> bool {
    let trimmed = line_body(line).trim();
    let width = trimmed
        .chars()
        .take_while(|character| *character == fence)
        .count();
    width >= minimum_width && trimmed.chars().skip(width).all(char::is_whitespace)
}

fn preferred_line_ending(lines: &[&str], opening: &str) -> &'static str {
    if lines
        .iter()
        .chain(std::iter::once(&opening))
        .any(|line| line.ends_with("\r\n"))
    {
        "\r\n"
    } else {
        "\n"
    }
}

fn signature_line_positions(lines: &[&str], start: usize, end: usize) -> Vec<usize> {
    let mut positions = Vec::new();
    let mut position = start;
    while position < end {
        if !looks_like_signature(lines[position]) {
            position += 1;
            continue;
        }
        positions.push(position);
        if signature_is_incomplete(lines[position]) {
            position += 1;
            while position < end {
                positions.push(position);
                if signature_continuation_ends(lines[position]) {
                    break;
                }
                position += 1;
            }
        }
        position += 1;
    }
    positions
}

fn looks_like_signature(line: &str) -> bool {
    let trimmed = line_body(line).trim();
    if trimmed.is_empty()
        || trimmed.starts_with("//")
        || trimmed.starts_with('#')
        || trimmed.starts_with("/*")
    {
        return false;
    }

    let without_visibility = trimmed
        .strip_prefix("pub ")
        .or_else(|| trimmed.strip_prefix("pub(crate) "))
        .or_else(|| trimmed.strip_prefix("pub(super) "))
        .unwrap_or(trimmed);
    let without_modifiers = without_visibility
        .strip_prefix("async ")
        .unwrap_or(without_visibility);
    if without_modifiers.starts_with("fn ")
        || without_modifiers.starts_with("unsafe fn ")
        || without_modifiers.starts_with("def ")
        || without_modifiers.starts_with("function ")
        || without_modifiers.starts_with("async function ")
    {
        return true;
    }

    let Some(open_parenthesis) = trimmed.find('(') else {
        return false;
    };
    let Some(close_parenthesis) = trimmed.rfind(')') else {
        return false;
    };
    if close_parenthesis < open_parenthesis
        || trimmed[..open_parenthesis].contains('=')
        || !trimmed[close_parenthesis + 1..]
            .trim()
            .starts_with(['{', ':', '-'])
    {
        return false;
    }
    let prefix = trimmed[..open_parenthesis].trim();
    let first_word = prefix.split_whitespace().next().unwrap_or_default();
    !matches!(
        first_word,
        "if" | "for" | "while" | "match" | "switch" | "catch" | "return" | "Some" | "Ok"
    ) && prefix
        .chars()
        .last()
        .is_some_and(|character| character.is_alphanumeric() || matches!(character, '_' | '$'))
}

fn signature_is_incomplete(line: &str) -> bool {
    let body = line_body(line).trim();
    delimiter_balance(body, '(', ')') > 0
        || delimiter_balance(body, '<', '>') > 0
        || (!body.contains(')') && !body.ends_with(['{', ':', ';']))
}

fn signature_continuation_ends(line: &str) -> bool {
    let body = line_body(line).trim_end();
    body.ends_with(['{', ':', ';'])
}

fn delimiter_balance(text: &str, opening: char, closing: char) -> isize {
    text.chars().fold(0isize, |balance, character| {
        if character == opening {
            balance + 1
        } else if character == closing {
            balance - 1
        } else {
            balance
        }
    })
}

fn binary_search_oldest_cut(
    payload: &mut CompressiblePayload,
    context: &CompressionContext,
    target: u32,
) {
    if payload.messages.is_empty() || count_payload_tokens(payload, context) <= target {
        return;
    }

    let source = payload.clone();
    let maximum_cut = source.messages.len();
    let maximally_truncated = payload_for_cut(&source, maximum_cut);
    if count_payload_tokens(&maximally_truncated, context) > target {
        *payload = maximally_truncated;
        payload.refresh_metadata();
        refresh_message_token_counts(payload, context);
        return;
    }

    let mut low = 1usize;
    let mut high = maximum_cut;
    while low < high {
        let middle = low + (high - low) / 2;
        let candidate = payload_for_cut(&source, middle);
        if count_payload_tokens(&candidate, context) <= target {
            high = middle;
        } else {
            low = middle + 1;
        }
    }

    *payload = payload_for_cut(&source, low);
    payload.refresh_metadata();
    refresh_message_token_counts(payload, context);
}

fn payload_for_cut(source: &CompressiblePayload, cut: usize) -> CompressiblePayload {
    let latest_user = latest_user_original_index(source);
    let groups = relationship_groups(source);
    let mut retained = vec![false; source.messages.len()];

    for group in groups {
        let protected = group
            .iter()
            .any(|position| must_preserve(&source.messages[*position], latest_user));
        let newest_position = group.iter().copied().max().unwrap_or(usize::MAX);
        if protected || newest_position >= cut {
            for position in group {
                retained[position] = true;
            }
        }
    }

    let mut candidate = source.clone();
    candidate.messages = source
        .messages
        .iter()
        .cloned()
        .enumerate()
        .filter_map(|(position, message)| retained[position].then_some(message))
        .collect();
    candidate.refresh_metadata();
    candidate
}

fn count_payload_tokens(payload: &CompressiblePayload, context: &CompressionContext) -> u32 {
    context
        .token_counter
        .count_request(&payload.clone().into_openai_request())
}

fn refresh_message_token_counts(payload: &mut CompressiblePayload, context: &CompressionContext) {
    let model = if payload.model.is_empty() {
        context.model.as_str()
    } else {
        payload.model.as_str()
    };
    for message in &mut payload.messages {
        let content_tokens = match message.content.as_value() {
            Value::Null => 0,
            Value::String(text) => context.token_counter.count_text(model, text),
            structured => context
                .token_counter
                .count_text(model, &structured.to_string()),
        };
        let extra_tokens = if message.extra.is_empty() {
            0
        } else {
            context
                .token_counter
                .count_text(model, &Value::Object(message.extra.clone()).to_string())
        };
        message.token_count = 4u32
            .saturating_add(context.token_counter.count_text(model, &message.role))
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
    use serde_json::{json, Value};

    fn payload(value: Value) -> CompressiblePayload {
        let request: OpenAIRequest = serde_json::from_value(value).unwrap();
        CompressiblePayload::from(request)
    }

    fn context() -> CompressionContext {
        CompressionContext::new("gpt-4o", "test")
    }

    fn message(role: &str, age: usize, content: &str) -> CompressibleMessage {
        CompressibleMessage {
            role: role.to_owned(),
            age,
            content: Value::String(content.to_owned()).into(),
            ..CompressibleMessage::default()
        }
    }

    fn ids(payload: &CompressiblePayload) -> Vec<String> {
        payload
            .messages
            .iter()
            .filter_map(|message| message.extra.get("name").and_then(Value::as_str))
            .map(str::to_owned)
            .collect()
    }

    #[test]
    fn default_and_custom_thresholds_are_deterministic() {
        assert_eq!(UltraEngine::new().relevance_threshold, 0.3);
        assert_eq!(UltraEngine::with_relevance_threshold(0.7).threshold(), 0.7);
        assert_eq!(UltraEngine::with_relevance_threshold(-1.0).threshold(), 0.0);
        assert_eq!(UltraEngine::with_relevance_threshold(2.0).threshold(), 1.0);
        assert_eq!(
            UltraEngine::with_relevance_threshold(f32::NAN).threshold(),
            0.3
        );
    }

    #[test]
    fn scoring_accounts_for_recency_code_paths_and_role_priority() {
        let recent_paths = HashSet::from(["crates/app/src/lib.rs".to_owned()]);
        let recent = relevance_score(&message("assistant", 0, "plain"), &recent_paths);
        let old = relevance_score(&message("assistant", 8, "plain"), &recent_paths);
        let code = relevance_score(
            &message("assistant", 8, "```rust\nlet value = 1;\n```"),
            &recent_paths,
        );
        let path = relevance_score(
            &message("assistant", 8, "See crates/app/src/lib.rs"),
            &recent_paths,
        );
        let user = relevance_score(&message("user", 8, "plain"), &recent_paths);
        let system = relevance_score(&message("system", usize::MAX, "plain"), &recent_paths);

        assert!(recent > old);
        assert!(code > old);
        assert!(path > old);
        assert!(user > old);
        assert_eq!(system, 1.0);
        assert!(system > recent);
    }

    #[tokio::test]
    async fn threshold_prunes_low_relevance_messages() {
        let mut payload = payload(json!({
            "model": "gpt-4o",
            "messages": [
                {"role": "system", "content": "policy", "name": "system"},
                {"role": "user", "content": "old question", "name": "old-user"},
                {"role": "assistant", "content": "stale plain answer", "name": "stale"},
                {"role": "user", "content": "new question", "name": "latest-user"}
            ]
        }));

        UltraEngine::with_relevance_threshold(0.5)
            .compress(&mut payload, &context())
            .await;

        assert_eq!(ids(&payload), ["system", "latest-user"]);
    }

    #[tokio::test]
    async fn preserves_system_latest_user_and_active_tool_use() {
        let mut payload = payload(json!({
            "model": "gpt-4o",
            "messages": [
                {"role": "system", "content": "exact policy", "name": "system"},
                {"role": "assistant", "content": "active", "name": "active", "tool_calls": [{"id": "active-call", "type": "function", "function": {"name": "lookup", "arguments": "{}"}}]},
                {"role": "assistant", "content": "remove me", "name": "stale"},
                {"role": "user", "content": "latest", "name": "latest-user"}
            ]
        }));
        let originals = payload
            .messages
            .iter()
            .map(|message| {
                (
                    message.original_index,
                    message.content.clone(),
                    message.extra.clone(),
                )
            })
            .collect::<Vec<_>>();

        UltraEngine::with_relevance_threshold(1.0)
            .compress(&mut payload, &context())
            .await;

        assert_eq!(ids(&payload), ["system", "active", "latest-user"]);
        for message in &payload.messages {
            let original = &originals[message.original_index];
            assert_eq!(message.content, original.1);
            assert_eq!(message.extra, original.2);
        }
    }

    #[tokio::test]
    async fn tool_pairs_are_retained_or_removed_as_an_indivisible_group() {
        let mut payload = payload(json!({
            "model": "gpt-4o",
            "messages": [
                {"role": "user", "content": "first"},
                {"role": "assistant", "name": "kept-call", "content": "```text\nimportant\n```", "tool_calls": [{"id": "keep", "type": "function", "function": {"name": "lookup", "arguments": "{}"}}]},
                {"role": "tool", "name": "kept-result", "tool_call_id": "keep", "content": "plain result"},
                {"role": "assistant", "name": "dropped-call", "content": "plain", "tool_calls": [{"id": "drop", "type": "function", "function": {"name": "lookup", "arguments": "{}"}}]},
                {"role": "tool", "name": "dropped-result", "tool_call_id": "drop", "content": "plain result"},
                {"role": "user", "name": "latest", "content": "latest"}
            ]
        }));

        UltraEngine::with_relevance_threshold(0.5)
            .compress(&mut payload, &context())
            .await;

        assert_eq!(ids(&payload), ["kept-call", "kept-result", "latest"]);
        payload.refresh_metadata();
        for message in &payload.messages {
            if !message.relationships.tool_result_for_ids.is_empty() {
                assert!(!message.relationships.related_message_indices.is_empty());
            }
            if message.relationships.unresolved_tool_call_ids.is_empty()
                && !message.relationships.tool_call_ids.is_empty()
            {
                assert!(!message.relationships.related_message_indices.is_empty());
            }
        }
    }

    #[tokio::test]
    async fn cache_protected_prefix_and_marker_remain_byte_stable() {
        let mut payload = payload(json!({
            "model": "gpt-4o",
            "messages": [
                {"role": "system", "content": "cached policy"},
                {"role": "user", "name": "cached", "content": [{"type": "text", "text": "cached text", "cache_control": {"type": "ephemeral"}}]},
                {"role": "assistant", "name": "discard", "content": "stale"},
                {"role": "user", "name": "latest", "content": "latest"}
            ]
        }));
        let cached = payload.messages[..2].to_vec();

        UltraEngine::with_relevance_threshold(1.0)
            .compress(&mut payload, &context())
            .await;

        assert_eq!(payload.messages[0], cached[0]);
        assert_eq!(payload.messages[1], cached[1]);
        assert!(payload.messages[1].has_cache_marker());
    }

    #[test]
    fn thins_only_large_code_blocks_with_exact_edges_signatures_and_marker() {
        let mut content = (0..60)
            .map(|line| format!("line {line}: value\n"))
            .collect::<Vec<_>>();
        content[25] = "pub async fn retained_signature(value: usize) -> usize {\n".to_owned();
        let code = format!("before\n```rust\n{}```\nafter", content.concat());

        let thinned = thin_large_code_blocks_in_text(&code);

        assert!(thinned.starts_with("before\n```rust\n"));
        assert!(thinned.ends_with("```\nafter"));
        for line in &content[..10] {
            assert!(thinned.contains(line));
        }
        for line in &content[50..] {
            assert!(thinned.contains(line));
        }
        assert!(thinned.contains(&content[25]));
        assert!(thinned.contains("[... 39 lines omitted ...]"));
        assert!(!thinned.contains(&content[24]));
    }

    #[test]
    fn leaves_small_blocks_and_non_code_regions_byte_exact() {
        let content = (0..50)
            .map(|line| format!("small {line}\n"))
            .collect::<String>();
        let text = format!("outside  bytes\n~~~rust\n{content}~~~\ntail  bytes");
        assert_eq!(thin_large_code_blocks_in_text(&text), text);
    }

    #[test]
    fn code_thinning_supports_crlf_and_tilde_fences() {
        let content = (0..51)
            .map(|line| format!("line {line}\r\n"))
            .collect::<String>();
        let text = format!("~~~ts\r\n{content}~~~\r\n");
        let thinned = thin_large_code_blocks_in_text(&text);
        assert!(thinned.contains("[... 31 lines omitted ...]\r\n"));
        assert!(thinned.starts_with("~~~ts\r\nline 0\r\n"));
        assert!(thinned.ends_with("line 50\r\n~~~\r\n"));
    }

    #[test]
    fn binary_search_finds_minimum_fitting_cut_and_maximizes_recent_context() {
        let mut payload = payload(json!({
            "model": "gpt-4o",
            "messages": [
                {"role": "system", "content": "policy", "name": "system"},
                {"role": "assistant", "content": "old zero ".repeat(80), "name": "zero"},
                {"role": "assistant", "content": "old one ".repeat(80), "name": "one"},
                {"role": "assistant", "content": "old two ".repeat(80), "name": "two"},
                {"role": "assistant", "content": "recent three ".repeat(80), "name": "three"},
                {"role": "user", "content": "latest", "name": "latest"}
            ]
        }));
        let context = context();
        let expected = payload_for_cut(&payload, 3);
        let target = count_payload_tokens(&expected, &context);
        assert!(count_payload_tokens(&payload_for_cut(&payload, 2), &context) > target);

        binary_search_oldest_cut(&mut payload, &context, target);

        assert_eq!(ids(&payload), ids(&expected));
        assert!(count_payload_tokens(&payload, &context) <= target);
        assert!(ids(&payload).contains(&"two".to_owned()));
    }

    #[test]
    fn binary_cut_preserves_critical_cache_and_tool_groups() {
        let mut payload = payload(json!({
            "model": "gpt-4o",
            "messages": [
                {"role": "system", "content": "policy", "name": "system"},
                {"role": "user", "content": [{"type": "text", "text": "cached", "cache_control": {"type": "ephemeral"}}], "name": "cached"},
                {"role": "assistant", "content": "call", "name": "call", "tool_calls": [{"id": "pair", "type": "function", "function": {"name": "lookup", "arguments": "{}"}}]},
                {"role": "tool", "content": "result", "name": "result", "tool_call_id": "pair"},
                {"role": "assistant", "content": "removable", "name": "removable"},
                {"role": "assistant", "content": "active", "name": "active", "tool_calls": [{"id": "active", "type": "function", "function": {"name": "lookup", "arguments": "{}"}}]},
                {"role": "user", "content": "latest", "name": "latest"}
            ]
        }));

        payload = payload_for_cut(&payload, payload.messages.len());

        assert_eq!(ids(&payload), ["system", "cached", "active", "latest"]);
        assert!(payload
            .messages
            .iter()
            .any(|message| { message.relationships.unresolved_tool_call_ids == ["active"] }));
    }

    #[tokio::test]
    async fn uses_target_budget_when_it_is_tighter_than_context_window() {
        let mut payload = payload(json!({
            "model": "gpt-4o",
            "messages": [
                {"role": "system", "content": "policy", "name": "system"},
                {"role": "assistant", "content": "old ".repeat(200), "name": "old"},
                {"role": "assistant", "content": "recent ".repeat(200), "name": "recent"},
                {"role": "user", "content": "latest", "name": "latest"}
            ]
        }));
        let mut context = context();
        let desired = payload_for_cut(&payload, 2);
        context.context_window = count_payload_tokens(&payload, &context) + 100;
        context.target_token_budget = Some(count_payload_tokens(&desired, &context));

        UltraEngine::with_relevance_threshold(0.0)
            .compress(&mut payload, &context)
            .await;

        assert_eq!(ids(&payload), ids(&desired));
        assert!(count_payload_tokens(&payload, &context) <= context.target_token_budget.unwrap());
    }

    #[tokio::test]
    async fn rolls_back_if_code_thinning_would_increase_tokens() {
        let mut middle = (0..30)
            .map(|index| format!("fn signature_{index}() {{\n"))
            .collect::<Vec<_>>();
        middle.push("\n".to_owned());
        let mut lines = (0..10).map(|_| "\n".to_owned()).collect::<Vec<_>>();
        lines.extend(middle);
        lines.extend((0..10).map(|_| "\n".to_owned()));
        assert_eq!(lines.len(), 51);
        let code = format!("```rust\n{}```", lines.concat());
        let mut payload = payload(json!({
            "model": "gpt-4o",
            "messages": [{"role": "user", "content": code}]
        }));
        let original = payload.clone();

        let result = UltraEngine::with_relevance_threshold(0.0)
            .compress(&mut payload, &context())
            .await;

        assert_eq!(payload, original);
        assert!(!result.applied);
        assert_eq!(result.tokens_after, result.tokens_before);
    }

    #[tokio::test]
    async fn reports_accurate_counts_and_never_increases_tokens() {
        let mut payload = payload(json!({
            "model": "gpt-4o",
            "messages": [
                {"role": "system", "content": "policy"},
                {"role": "user", "content": "old"},
                {"role": "assistant", "content": "irrelevant old prose ".repeat(100)},
                {"role": "user", "content": "middle"},
                {"role": "assistant", "content": "middle answer"},
                {"role": "user", "content": "latest"}
            ]
        }));
        let context = context();
        let expected_before = count_payload_tokens(&payload, &context);

        let result = UltraEngine::new().compress(&mut payload, &context).await;
        let actual_after = count_payload_tokens(&payload, &context);

        assert_eq!(result.engine_name, "ultra");
        assert_eq!(result.tokens_before, expected_before);
        assert_eq!(result.tokens_after, actual_after);
        assert!(result.tokens_after <= result.tokens_before);
        assert!(result.applied);
    }
}
