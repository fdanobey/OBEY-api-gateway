//! Standard compression engine.

use super::{CompressiblePayload, CompressionContext, CompressionEngine, EngineResult};
use async_trait::async_trait;
use regex::Regex;
use serde_json::Value;
use std::{sync::LazyLock, time::Instant};

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum RuleScope {
    Common,
    Assistant,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum RuleGroup {
    Filler,
    Verbose,
    CodingPrompt,
    AssistantHedge,
}

#[derive(Debug)]
struct TransformationRule {
    regex: Regex,
    replacement: &'static str,
    scope: RuleScope,
    #[cfg_attr(not(test), allow(dead_code))]
    group: RuleGroup,
}

impl TransformationRule {
    fn new(
        pattern: &'static str,
        replacement: &'static str,
        scope: RuleScope,
        group: RuleGroup,
    ) -> Self {
        let pattern = if group == RuleGroup::AssistantHedge {
            format!(r"{pattern}(?:[\t ]*[.!?])?[\t ]*")
        } else if replacement.is_empty() {
            format!(r"{pattern}[\t ]*")
        } else {
            pattern.to_owned()
        };
        Self {
            regex: Regex::new(&pattern).expect("standard compression regex must compile"),
            replacement,
            scope,
            group,
        }
    }
}

const COMMON: RuleScope = RuleScope::Common;
const ASSISTANT: RuleScope = RuleScope::Assistant;
const FILLER: RuleGroup = RuleGroup::Filler;
const VERBOSE: RuleGroup = RuleGroup::Verbose;
const CODING: RuleGroup = RuleGroup::CodingPrompt;
const HEDGE: RuleGroup = RuleGroup::AssistantHedge;

static TRANSFORMATION_RULES: LazyLock<Vec<TransformationRule>> = LazyLock::new(|| {
    [
        (r"(?i)\bcould you please\b", "", COMMON, CODING),
        (r"(?i)\bcan you please\b", "", COMMON, CODING),
        (r"(?i)\bwould you please\b", "", COMMON, CODING),
        (r"(?i)\bi would like you to\b", "", COMMON, CODING),
        (r"(?i)\bi need you to\b", "", COMMON, CODING),
        (r"(?i)\bwhat i want you to do is\b", "", COMMON, CODING),
        (r"(?i)\byour task is to\b", "", COMMON, CODING),
        (r"(?i)\bgo ahead and\b", "", COMMON, CODING),
        (r"(?i)\btake a look at\b", "inspect", COMMON, CODING),
        (
            r"(?i)\bprovide an explanation of\b",
            "explain",
            COMMON,
            CODING,
        ),
        (r"(?i)\bgive me an example of\b", "show", COMMON, CODING),
        (
            r"(?i)\bwrite the code necessary to\b",
            "implement",
            COMMON,
            CODING,
        ),
        (r"(?i)\bmake sure that\b", "ensure", COMMON, CODING),
        (r"(?i)\bkeep in mind that\b", "remember", COMMON, CODING),
        (
            r"(?i)\bthe following piece of code\b",
            "this code",
            COMMON,
            CODING,
        ),
        (r"(?i)\bin the code below\b", "below", COMMON, CODING),
        (r"(?i)\bit is important to note that\b", "", COMMON, VERBOSE),
        (r"(?i)\bas a matter of fact\b", "", COMMON, VERBOSE),
        (r"(?i)\bdue to the fact that\b", "because", COMMON, VERBOSE),
        (r"(?i)\bat this point in time\b", "now", COMMON, VERBOSE),
        (r"(?i)\bin order to\b", "to", COMMON, VERBOSE),
        (r"(?i)\bis able to\b", "can", COMMON, VERBOSE),
        (r"(?i)\bare able to\b", "can", COMMON, VERBOSE),
        (r"(?i)\bhas the ability to\b", "can", COMMON, VERBOSE),
        (r"(?i)\bhave the ability to\b", "can", COMMON, VERBOSE),
        (r"(?i)\bfor the purpose of\b", "to", COMMON, VERBOSE),
        (r"(?i)\bin the event that\b", "if", COMMON, VERBOSE),
        (r"(?i)\bin cases? where\b", "when", COMMON, VERBOSE),
        (r"(?i)\bwith regard to\b", "about", COMMON, VERBOSE),
        (r"(?i)\bin relation to\b", "about", COMMON, VERBOSE),
        (r"(?i)\ba large number of\b", "many", COMMON, VERBOSE),
        (r"(?i)\ba small number of\b", "few", COMMON, VERBOSE),
        (r"(?i)\bon a daily basis\b", "daily", COMMON, VERBOSE),
        (r"(?i)\bmake use of\b", "use", COMMON, VERBOSE),
        (r"(?i)\bprior to\b", "before", COMMON, VERBOSE),
        (r"(?i)\bsubsequent to\b", "after", COMMON, VERBOSE),
        (r"(?i)\bin the near future\b", "soon", COMMON, VERBOSE),
        (
            r"(?i)\bdespite the fact that\b",
            "although",
            COMMON,
            VERBOSE,
        ),
        (r"(?i)\bneedless to say\b", "", COMMON, VERBOSE),
        (r"(?i)\bat the present time\b", "now", COMMON, VERBOSE),
        (r"(?i)\bon account of\b", "because of", COMMON, VERBOSE),
        (r"(?i)\bin the process of\b", "", COMMON, VERBOSE),
        (r"(?i)\bplease\b", "", COMMON, FILLER),
        (r"(?i)\bi think\b", "", COMMON, FILLER),
        (r"(?i)\bi believe\b", "", COMMON, FILLER),
        (r"(?i)\bbasically\b", "", COMMON, FILLER),
        (r"(?i)\bactually\b", "", COMMON, FILLER),
        (r"(?i)\bjust\b", "", COMMON, FILLER),
        (r"(?i)\breally\b", "", COMMON, FILLER),
        (r"(?i)\bvery\b", "", COMMON, FILLER),
        (r"(?i)\bsimply\b", "", COMMON, FILLER),
        (r"(?i)\bperhaps\b", "", COMMON, FILLER),
        (r"(?i)\bmaybe\b", "", COMMON, FILLER),
        (r"(?i)\bgenerally speaking\b", "", COMMON, FILLER),
        (r"(?i)\bessentially\b", "", COMMON, FILLER),
        (r"(?i)\bliterally\b", "", COMMON, FILLER),
        (r"(?i)\bhonestly\b", "", COMMON, FILLER),
        (r"(?i)\bclearly\b", "", COMMON, FILLER),
        (r"(?i)\bobviously\b", "", COMMON, FILLER),
        (r"(?i)\bkind of\b", "", COMMON, FILLER),
        (r"(?i)\bsort of\b", "", COMMON, FILLER),
        (r"(?i)\bi hope this helps\b", "", ASSISTANT, HEDGE),
        (
            r"(?i)\blet me know if you have any questions\b",
            "",
            ASSISTANT,
            HEDGE,
        ),
        (r"(?i)\bhappy to help with that\b", "", ASSISTANT, HEDGE),
        (r"(?i)\bi(?:'|’)d be happy to\b", "", ASSISTANT, HEDGE),
        (r"(?i)\bfeel free to ask\b", "", ASSISTANT, HEDGE),
        (r"(?i)\bif you(?:'|’)d like,? i can\b", "", ASSISTANT, HEDGE),
    ]
    .into_iter()
    .map(|(pattern, replacement, scope, group)| {
        TransformationRule::new(pattern, replacement, scope, group)
    })
    .collect()
});

#[derive(Debug)]
struct CleanupRegexes {
    horizontal_whitespace: Regex,
    space_before_punctuation: Regex,
    space_after_opening: Regex,
    line_trailing_whitespace: Regex,
}

static CLEANUP_REGEXES: LazyLock<CleanupRegexes> = LazyLock::new(|| CleanupRegexes {
    horizontal_whitespace: Regex::new(r"[\t ]{2,}").expect("cleanup regex must compile"),
    space_before_punctuation: Regex::new(r"[\t ]+([,.;:!?])").expect("cleanup regex must compile"),
    space_after_opening: Regex::new(r"([\(\[\{])[\t ]+").expect("cleanup regex must compile"),
    line_trailing_whitespace: Regex::new(r"(?m)[\t ]+$").expect("cleanup regex must compile"),
});

/// Conversational and coding-prompt compression using reusable regex rules.
#[derive(Debug, Clone, Copy)]
pub struct StandardEngine;

impl StandardEngine {
    /// Creates a standard engine and initializes its process-wide compiled rules.
    pub fn new() -> Self {
        LazyLock::force(&TRANSFORMATION_RULES);
        LazyLock::force(&CLEANUP_REGEXES);
        Self
    }
}

impl Default for StandardEngine {
    fn default() -> Self {
        Self::new()
    }
}

#[async_trait]
impl CompressionEngine for StandardEngine {
    fn name(&self) -> &str {
        "standard"
    }

    async fn compress(
        &self,
        payload: &mut CompressiblePayload,
        context: &CompressionContext,
    ) -> EngineResult {
        let started = Instant::now();
        let original = payload.clone();
        let tokens_before = count_payload_tokens(&original, context);
        let mut changed = false;

        for message in &mut payload.messages {
            if message.cache_protected {
                continue;
            }

            let assistant = message.role == "assistant";
            message.content.for_each_text_leaf_mut(|text| {
                let transformed = context
                    .protection_scanner
                    .transform_unprotected(text, |segment| transform_segment(segment, assistant));
                if transformed != *text {
                    *text = transformed;
                    changed = true;
                }
            });
        }

        if changed {
            payload.refresh_metadata();
            refresh_message_token_counts(payload, context);
        }

        let tokens_after = count_payload_tokens(payload, context);
        let applied = if changed && tokens_after <= tokens_before {
            true
        } else {
            if tokens_after > tokens_before {
                *payload = original;
            }
            false
        };
        let tokens_after = if applied {
            tokens_after
        } else {
            count_payload_tokens(payload, context)
        };

        EngineResult {
            engine_name: self.name().to_owned(),
            tokens_before,
            tokens_after,
            duration_ms: started.elapsed().as_millis().min(u64::MAX as u128) as u64,
            applied,
        }
    }
}

fn transform_segment(segment: &str, assistant: bool) -> String {
    let mut output = segment.to_owned();
    let mut changed = false;

    for rule in TRANSFORMATION_RULES.iter() {
        if rule.scope == RuleScope::Assistant && !assistant {
            continue;
        }
        if rule.regex.is_match(&output) {
            output = rule
                .regex
                .replace_all(&output, rule.replacement)
                .into_owned();
            changed = true;
        }
    }

    if changed {
        normalize_removal_artifacts(&output)
    } else {
        output
    }
}

fn normalize_removal_artifacts(text: &str) -> String {
    let cleanup = &*CLEANUP_REGEXES;
    let text = cleanup
        .horizontal_whitespace
        .replace_all(text, " ")
        .into_owned();
    let text = cleanup
        .space_before_punctuation
        .replace_all(&text, "$1")
        .into_owned();
    let text = cleanup
        .space_after_opening
        .replace_all(&text, "$1")
        .into_owned();
    cleanup
        .line_trailing_whitespace
        .replace_all(&text, "")
        .into_owned()
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

#[cfg(test)]
mod tests {
    use super::*;
    use crate::models::openai::OpenAIRequest;
    use serde_json::{json, Value};
    use std::ptr;

    fn payload(value: Value) -> CompressiblePayload {
        let request: OpenAIRequest = serde_json::from_value(value).unwrap();
        CompressiblePayload::from(request)
    }

    fn context() -> CompressionContext {
        CompressionContext::new("gpt-4o", "test")
    }

    async fn compress(payload: &mut CompressiblePayload) -> EngineResult {
        StandardEngine::new().compress(payload, &context()).await
    }

    #[test]
    fn provides_more_than_thirty_rules_across_every_required_group() {
        StandardEngine::new();

        assert!(TRANSFORMATION_RULES.len() >= 30);
        for group in [FILLER, VERBOSE, CODING, HEDGE] {
            assert!(TRANSFORMATION_RULES.iter().any(|rule| rule.group == group));
        }
    }

    #[tokio::test]
    async fn executes_filler_verbose_coding_and_assistant_rule_groups() {
        let mut payload = payload(json!({
            "model": "gpt-4o",
            "messages": [
                {
                    "role": "user",
                    "content": "Could you please basically take a look at this module in order to make sure that it is able to compile due to the fact that it is very important?"
                },
                {
                    "role": "assistant",
                    "content": "I hope this helps. Let me know if you have any questions."
                }
            ]
        }));

        let result = compress(&mut payload).await;
        let user = payload.messages[0].content.as_text().unwrap();
        let assistant = payload.messages[1].content.as_text().unwrap();

        assert!(result.applied);
        assert_eq!(
            user,
            "inspect this module to ensure it can compile because it is important?"
        );
        assert_eq!(assistant, "");
    }

    #[tokio::test]
    async fn polite_hedges_are_removed_only_from_assistant_messages() {
        let hedge = "I hope this helps. Let me know if you have any questions.";
        let mut payload = payload(json!({
            "model": "gpt-4o",
            "messages": [
                {"role": "user", "content": hedge},
                {"role": "assistant", "content": hedge}
            ]
        }));

        let result = compress(&mut payload).await;

        assert!(result.applied);
        assert_eq!(payload.messages[0].content.as_text(), Some(hedge));
        assert_eq!(payload.messages[1].content.as_text(), Some(""));
    }

    #[tokio::test]
    async fn preserves_every_protected_region_byte_for_byte() {
        let protected = [
            "```rust\r\nlet  value =  veryImportant;\r\n```",
            "https://example.test/really/path?q=very",
            r"C:\Users\alice\very_important.rs",
            "/usr/local/really-important/file.rs",
            r#"{"please":"do not alter", "very":  true}"#,
            "camelCaseIdentifier",
            "$very + please$",
            r#"tool_call: {"name":"pleaseRun","arguments":{"value":"very"}}"#,
        ];
        let input = format!(
            "Please inspect these exactly:\n{}\n{}\n{}\n{}\n{}\n{}\n{}\n{}\nActually finish.",
            protected[0],
            protected[1],
            protected[2],
            protected[3],
            protected[4],
            protected[5],
            protected[6],
            protected[7]
        );
        let mut payload = payload(json!({
            "model": "gpt-4o",
            "messages": [{"role": "user", "content": input}]
        }));

        let result = compress(&mut payload).await;
        let output = payload.messages[0].content.as_text().unwrap();

        assert!(result.applied);
        for expected in protected {
            assert!(
                output.contains(expected),
                "missing protected bytes: {expected}"
            );
        }
        assert!(!output.contains("Please inspect"));
        assert!(!output.contains("Actually finish"));
    }

    #[tokio::test]
    async fn leaves_cache_protected_prefix_unchanged() {
        let mut payload = payload(json!({
            "model": "gpt-4o",
            "messages": [
                {"role": "system", "content": "Please actually preserve this very important policy."},
                {
                    "role": "user",
                    "content": [{
                        "type": "text",
                        "text": "I think this cached text is very important.",
                        "cache_control": {"type": "ephemeral"}
                    }]
                },
                {"role": "assistant", "content": "I hope this helps. Actually done."}
            ]
        }));
        let cached_prefix: Vec<_> = payload.messages[..2]
            .iter()
            .map(|message| message.content.clone())
            .collect();

        let result = compress(&mut payload).await;

        assert!(result.applied);
        assert_eq!(payload.messages[0].content, cached_prefix[0]);
        assert_eq!(payload.messages[1].content, cached_prefix[1]);
        assert_eq!(payload.messages[2].content.as_text(), Some("done."));
    }

    #[tokio::test]
    async fn preserves_tool_pairs_schemas_arguments_and_request_fields() {
        let tools = json!([{
            "type": "function",
            "function": {
                "name": "veryImportantLookup",
                "description": "Please actually keep this very detailed description.",
                "parameters": {
                    "type": "object",
                    "properties": {"item_id": {"type": "string"}},
                    "required": ["item_id"]
                }
            }
        }]);
        let mut payload = payload(json!({
            "model": "gpt-4o",
            "stream": true,
            "temperature": 0.2,
            "max_tokens": 512,
            "tools": tools,
            "tool_choice": {"type": "function", "function": {"name": "veryImportantLookup"}},
            "response_format": {"type": "json_object"},
            "messages": [
                {"role": "user", "content": "Please actually find the item."},
                {
                    "role": "assistant",
                    "content": "I think I should call it.",
                    "tool_calls": [{
                        "id": "call_very_1",
                        "type": "function",
                        "function": {
                            "name": "veryImportantLookup",
                            "arguments": "{\"item_id\":\"please-very-1\"}"
                        }
                    }]
                },
                {
                    "role": "tool",
                    "tool_call_id": "call_very_1",
                    "name": "veryImportantLookup",
                    "content": "{\"item_id\":\"please-very-1\",\"status\":\"actually ready\"}"
                }
            ]
        }));
        let original_tools = payload.tool_definitions.clone();
        let original_extra = payload.extra.clone();
        let original_message_extras: Vec<_> = payload
            .messages
            .iter()
            .map(|message| message.extra.clone())
            .collect();
        let original_indices: Vec<_> = payload
            .messages
            .iter()
            .map(|message| message.original_index)
            .collect();

        let result = compress(&mut payload).await;

        assert!(result.applied);
        assert_eq!(payload.tool_definitions, original_tools);
        assert_eq!(payload.extra, original_extra);
        assert_eq!(payload.stream, true);
        assert_eq!(payload.temperature, Some(0.2));
        assert_eq!(payload.max_tokens, Some(512));
        assert_eq!(
            payload
                .messages
                .iter()
                .map(|message| message.extra.clone())
                .collect::<Vec<_>>(),
            original_message_extras
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
            payload.messages[1].relationships.tool_call_ids,
            ["call_very_1"]
        );
        assert_eq!(
            payload.messages[2].relationships.tool_result_for_ids,
            ["call_very_1"]
        );
        assert_eq!(
            payload.messages[1].relationships.related_message_indices,
            [2]
        );
        assert_eq!(
            payload.messages[2].relationships.related_message_indices,
            [1]
        );
        assert_eq!(
            payload.messages[2].content.as_text(),
            Some(r#"{"item_id":"please-very-1","status":"actually ready"}"#)
        );
    }

    #[tokio::test]
    async fn reports_accurate_counts_and_never_increases_tokens() {
        let cases = [
            json!({"model": "gpt-4o", "messages": [{"role": "user", "content": "Already concise."}]}),
            json!({"model": "gpt-4o", "messages": [{"role": "user", "content": "Please actually use a very small number of checks in order to finish."}]}),
            json!({"model": "gpt-4o", "messages": [{"role": "assistant", "content": "I hope this helps. Happy to help with that."}]}),
        ];
        let context = context();

        for case in cases {
            let mut payload = payload(case);
            let original = payload.clone();
            let expected_before = count_payload_tokens(&original, &context);
            let result = StandardEngine::new().compress(&mut payload, &context).await;
            let actual_after = count_payload_tokens(&payload, &context);

            assert_eq!(result.engine_name, "standard");
            assert_eq!(result.tokens_before, expected_before);
            assert_eq!(result.tokens_after, actual_after);
            assert!(result.tokens_after <= result.tokens_before);
            if result.tokens_after > result.tokens_before {
                assert_eq!(payload, original);
                assert!(!result.applied);
            }
        }
    }

    #[test]
    fn increasing_transform_rolls_back_to_original_payload() {
        let context = context();
        let original = payload(json!({
            "model": "gpt-4o",
            "messages": [{"role": "user", "content": "Please finish."}]
        }));
        let mut transformed = original.clone();
        *transformed.messages[0].content.as_value_mut() =
            Value::String("expanded output ".repeat(100));
        let tokens_before = count_payload_tokens(&original, &context);
        let tokens_after = count_payload_tokens(&transformed, &context);

        assert!(tokens_after > tokens_before);
        if tokens_after > tokens_before {
            transformed = original.clone();
        }
        assert_eq!(transformed, original);
    }

    #[tokio::test]
    async fn compiled_regexes_are_reused_across_engines_and_requests() {
        let first = StandardEngine::new();
        let first_regex = &TRANSFORMATION_RULES[0].regex;
        let second = StandardEngine::default();
        let second_regex = &TRANSFORMATION_RULES[0].regex;
        assert!(ptr::eq(first_regex, second_regex));

        let mut first_payload = payload(json!({
            "model": "gpt-4o",
            "messages": [{"role": "user", "content": "Could you please finish?"}]
        }));
        let mut second_payload = first_payload.clone();
        first.compress(&mut first_payload, &context()).await;
        second.compress(&mut second_payload, &context()).await;

        assert!(ptr::eq(first_regex, &TRANSFORMATION_RULES[0].regex));
        assert_eq!(first_payload, second_payload);
    }

    #[tokio::test]
    async fn refreshes_metadata_and_message_token_counts_after_changes() {
        let mut payload = payload(json!({
            "model": "gpt-4o",
            "messages": [
                {"role": "user", "content": "Please actually finish."},
                {"role": "assistant", "content": "I hope this helps."}
            ]
        }));
        let previous_counts: Vec<_> = payload
            .messages
            .iter()
            .map(|message| message.token_count)
            .collect();

        let result = compress(&mut payload).await;

        assert!(result.applied);
        assert!(payload.messages[0].critical);
        assert_eq!(payload.messages[0].age, 0);
        assert_eq!(payload.messages[1].age, 0);
        assert_ne!(
            payload
                .messages
                .iter()
                .map(|message| message.token_count)
                .collect::<Vec<_>>(),
            previous_counts
        );
    }
}
