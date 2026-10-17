//! Async conversation summarizer for the `summarize_then_trim` context
//! reduction strategy.
//!
//! Follows the memory-extraction adapter pattern
//! (`memory/extraction_adapter.rs`): the provider is resolved directly from
//! configuration by model name, the call is tagged internal
//! ([`CONTEXT_SUMMARIZATION_INTERNAL_TAG`]) so gateway features that inspect
//! internal traffic can recognize it, and the HTTP request goes straight to
//! the provider's chat-completions endpoint. The router bypass is
//! **structural**, not marker-enforced: [`GatewaySummarizer`] never calls
//! into the router (or its health/rate-limit state) at all, so summarization
//! cannot recurse and never debits provider rate-limit tokens. The internal
//! marker is additionally surfaced on the wire as the sanitized,
//! constant-value [`CONTEXT_SUMMARIZATION_HEADER`] informational header so
//! providers and observers can recognize gateway-internal traffic. On
//! timeout or any provider failure the orchestrating
//! [`GatewaySummarizer::reduce_with_summarizer`] falls back to the
//! `clear_tools_then_remove` sync stages via [`ContextBudgetGate::reduce`]
//! and flags `fallback_from_summarization` for logging.

use std::sync::Arc;
use std::time::Duration;

use serde_json::Value;
use tokio::sync::RwLock;

use crate::compression::token_counter::TokenCounter;
use crate::config::{Config, ContextConfig, ModelGroup, Provider, ProviderModel};
use crate::models::openai::{Message, OpenAIRequest};

use super::manager::ContextManager;
use super::overflow::{
    ContextBudgetGate, ContextReductionStrategy, ReductionOptions, ReductionOutcome,
};
use super::strategies::{apply_truncation_strategy, TruncationStrategy};

/// Internal tag marking context-summarizer traffic, mirroring
/// `MEMORY_EXTRACTION_INTERNAL_TAG` for the memory-extraction adapter.
pub const CONTEXT_SUMMARIZATION_INTERNAL_TAG: &str = "context_summarization";

/// Sanitized, constant-value informational header attached to every
/// summarizer call so providers and observers can recognize gateway-internal
/// traffic on the wire. Constant by construction — it can never echo request
/// or conversation content.
pub const CONTEXT_SUMMARIZATION_HEADER: &str = "x-gateway-internal-context-summarization";

/// Upper bound on conversation messages rendered into the summary prompt.
const MAX_CONVERSATION_MESSAGES: usize = 200;
/// Upper bound on characters rendered per conversation message.
const MAX_MESSAGE_CHARS: usize = 2_000;

/// Prompt requesting a structured summary. Sent as the user message; the
/// conversation transcript is appended compactly below it.
const SUMMARIZATION_PROMPT: &str = "Summarize the conversation below so work can continue once the older messages are removed. Recent messages will be kept; your summary replaces everything older.\n\nRespond with exactly these sections, each as a short bullet list:\n1. Goals and constraints\n2. Decisions made\n3. Facts, file paths, and identifiers\n4. Unresolved tasks and open questions\n\nPreserve every detail needed to continue the work: exact file paths, identifiers, commands, error messages, and constraints. Omit a section only if truly empty. No commentary.";

/// Sanitized summarizer failure. Never includes conversation content.
#[derive(Debug, Clone, PartialEq, Eq, thiserror::Error)]
#[error("{message}")]
pub struct SummarizationError {
    pub message: String,
}

/// Internal, router-bypassing summarizer call, mirroring the
/// memory-extraction provider request shape.
#[derive(Debug, Clone)]
pub struct SummarizerRequest<'a> {
    /// Configured `context.summarize_model` value (model or group name).
    pub model: &'a str,
    /// Conversation messages to summarize.
    pub messages: &'a [Message],
    /// Timeout for the provider call in seconds.
    pub timeout_seconds: u64,
    pub internal_tag: &'static str,
    pub bypass_memory: bool,
}

impl<'a> SummarizerRequest<'a> {
    /// Builds an internal summarizer request that bypasses memory injection.
    pub fn internal(model: &'a str, messages: &'a [Message], timeout_seconds: u64) -> Self {
        Self {
            model,
            messages,
            timeout_seconds,
            internal_tag: CONTEXT_SUMMARIZATION_INTERNAL_TAG,
            bypass_memory: true,
        }
    }
}

/// Production summarizer adapter that calls a configured provider via the
/// standard OpenAI-compatible chat-completions endpoint.
pub struct GatewaySummarizer {
    config: Arc<RwLock<Config>>,
    http_client: reqwest::Client,
}

impl GatewaySummarizer {
    pub fn new(config: Arc<RwLock<Config>>) -> Self {
        let http_client = reqwest::Client::builder()
            .timeout(Duration::from_secs(60))
            .build()
            .unwrap_or_else(|_| reqwest::Client::new());
        Self {
            config,
            http_client,
        }
    }

    /// Resolves the provider backing `model_name` directly from config —
    /// exact provider-model match first (lowest priority wins, mirroring the
    /// router's ascending-priority sort), then model-group name. Never
    /// touches the router or its health/rate-limit state.
    async fn resolve_provider(
        &self,
        model_name: &str,
    ) -> Result<(Provider, String), SummarizationError> {
        let config = self.config.read().await;
        let candidate = select_candidate(&config.model_groups, model_name).ok_or_else(|| {
            SummarizationError {
                message: format!("summarize_model {model_name:?} not found in any model group"),
            }
        })?;
        let provider = config
            .providers
            .iter()
            .find(|p| p.name == candidate.provider)
            .cloned()
            .ok_or_else(|| SummarizationError {
                message: format!(
                    "provider {:?} for summarize_model {model_name:?} not found",
                    candidate.provider
                ),
            })?;
        Ok((provider, candidate.model.clone()))
    }

/// Makes the internal, timeout-bounded summarizer call and returns the
/// trimmed summary text. Direct provider HTTP call — no router recursion,
/// no provider rate-token debit.
///
/// The configured timeout bounds the **entire** exchange: connect, send,
/// status/headers, and the full body read + JSON parse. A provider that
/// streams its body slower than the timeout budget (or stalls mid-body)
/// triggers the fallback exactly like a slow TTFB, instead of hanging on
/// the client's 60s HTTP default.
pub async fn summarize_conversation(
    &self,
    request: SummarizerRequest<'_>,
) -> Result<String, SummarizationError> {
    debug_assert_eq!(request.internal_tag, CONTEXT_SUMMARIZATION_INTERNAL_TAG);
    debug_assert!(request.bypass_memory);

    let (provider, model) = self.resolve_provider(request.model).await?;
    let api_key = provider.resolve_api_key().unwrap_or_default();

    // Build base URL — strip trailing slash, append /v1 if missing
    let mut base_url = provider.base_url.clone().unwrap_or_default();
    base_url = base_url.trim_end_matches('/').to_string();
    if !base_url.ends_with("/v1") {
        base_url.push_str("/v1");
    }
    let url = format!("{}/chat/completions", base_url);

    let body = Self::build_request_body(&model, request.messages);

        let mut builder = self
            .http_client
            .post(&url)
            .header("Authorization", format!("Bearer {}", api_key))
            .header("Content-Type", "application/json")
            .header(CONTEXT_SUMMARIZATION_HEADER, CONTEXT_SUMMARIZATION_INTERNAL_TAG)
            .json(&body);

    for (key, value) in provider.effective_custom_headers() {
        builder = builder.header(key.as_str(), value);
    }

let timeout = Duration::from_secs(request.timeout_seconds);
// Send + status/headers + full body read/parse, all under one budget.
let response_json = tokio::time::timeout(timeout, async {
    let response = builder
        .send()
        .await
        .map_err(|e| SummarizationError {
            message: format!("summarizer HTTP request failed: {e}"),
        })?;
    let status = response.status();
    if !status.is_success() {
        let error_text = response
            .text()
            .await
            .unwrap_or_else(|_| "unknown error".to_string());
        return Err(SummarizationError {
            message: format!(
                "summarizer provider returned HTTP {}: {}",
                status.as_u16(),
                error_text
            ),
        });
    }
    response
        .json::<serde_json::Value>()
        .await
        .map_err(|e| SummarizationError {
            message: format!("failed to parse summarizer response body: {e}"),
        })
})
.await
.map_err(|_| SummarizationError {
    message: format!("summarizer timed out after {}s", request.timeout_seconds),
    })??;

    let content = response_json
        .get("choices")
        .and_then(|c| c.get(0))
        .and_then(|c| c.get("message"))
        .and_then(|m| m.get("content"))
        .and_then(|c| c.as_str())
        .unwrap_or("");

        if content.trim().is_empty() {
            return Err(SummarizationError {
                message: "summarizer response contained no summary content".to_string(),
            });
        }

        Ok(content.trim().to_string())
    }

    /// Orchestrates US-3 `summarize_then_trim` semantics for the router:
    /// try the summarizer → insert the `[Conversation summary]` system
    /// message in place of the removed older groups (trimming to the
    /// configured target percentage of the budget) → verify fit; on any
    /// failure (timeout, HTTP error, unconfigured model) fall back to
    /// `clear_tools_then_remove` via [`ContextBudgetGate::reduce`] and set
    /// `fallback_from_summarization`.
    pub async fn reduce_with_summarizer(
        &self,
        request: &mut OpenAIRequest,
        options: &ReductionOptions<'_>,
        context_config: &ContextConfig,
    ) -> ReductionOutcome {
        let counter = TokenCounter::new();
        let original_tokens = counter.count_request(request);
        let mut outcome = ReductionOutcome {
            strategy_applied: ContextReductionStrategy::SummarizeThenTrim,
            original_tokens,
            final_tokens: original_tokens,
            budget: options.budget,
            groups_removed: 0,
            tool_results_cleared: 0,
            summarized: false,
            fallback_from_summarization: false,
            fits: original_tokens <= options.budget,
        };
if u64::from(original_tokens) * 100
<= u64::from(options.budget) * u64::from(context_config.summarize_trigger_percent)
{
let mut trim_options = options.clone();
trim_options.strategy = ContextReductionStrategy::ClearToolsThenRemove;
return ContextBudgetGate::reduce(request, &trim_options);
}

        let summary_result = match context_config.summarize_model.as_deref() {
            Some(model) => {
                self.summarize_conversation(SummarizerRequest::internal(
                    model,
                    &request.messages,
                    context_config.summarize_timeout_seconds,
                ))
                .await
            }
            None => Err(SummarizationError {
                message: "context.summarize_model is not configured".to_string(),
            }),
        };

        match summary_result {
            Ok(summary_text) => {
                let non_message_tokens = original_tokens
                    .saturating_sub(ContextManager::estimate_tokens(&request.messages));
                let target_budget = ((options.budget as u64
                    * context_config.summarize_target_percent as u64)
                    / 100)
                    .min(options.budget as u64) as u32;
                let target_message_budget = target_budget.saturating_sub(non_message_tokens);

                // The summary replaces the removed older messages: trim
                // oldest groups down to the target, then insert the summary
                // system message after the leading system prompt.
                let trimmed = apply_truncation_strategy(
                    &mut request.messages,
                    target_message_budget,
                    TruncationStrategy::RemoveOldest,
                    ContextManager::estimate_tokens,
                );
                outcome.groups_removed = trimmed.messages_removed;

                insert_summary_message(&mut request.messages, &summary_text);
                outcome.summarized = true;
                outcome.final_tokens = counter.count_request(request);
                outcome.fits = outcome.final_tokens <= options.budget;

                // Verify fit; if the summary itself pushed the request over
                // the budget, trim further (never removes system messages,
                // so the summary survives).
                if !outcome.fits {
                    let message_budget = options.budget.saturating_sub(non_message_tokens);
                    let extra = apply_truncation_strategy(
                        &mut request.messages,
                        message_budget,
                        TruncationStrategy::RemoveOldest,
                        ContextManager::estimate_tokens,
                    );
                    outcome.groups_removed += extra.messages_removed;
                    outcome.final_tokens = counter.count_request(request);
                    outcome.fits = outcome.final_tokens <= options.budget;
                }
                outcome
            }
            Err(error) => {
                tracing::warn!(
                    error = %error.message,
                    "context summarizer failed; falling back to clear_tools_then_remove"
                );
                let mut fallback_options = options.clone();
                fallback_options.strategy = ContextReductionStrategy::ClearToolsThenRemove;
                let mut fallback = ContextBudgetGate::reduce(request, &fallback_options);
                fallback.fallback_from_summarization = true;
                fallback
            }
        }
    }

    /// Builds the outgoing summarizer request body: the structured-summary
    /// prompt plus the compact conversation transcript as a single user
    /// message, mirroring the extraction adapter's non-streaming shape.
    fn build_request_body(model: &str, messages: &[Message]) -> serde_json::Value {
        serde_json::json!({
            "model": model,
            "messages": [
                {
                    "role": "user",
                    "content": format!("{SUMMARIZATION_PROMPT}\n\n{}", render_conversation(messages)),
                },
            ],
            "temperature": 0.1,
            "stream": false,
        })
    }
}

/// Picks the provider-model entry for a summarize_model config value: an
/// exact provider-model match wins (lowest priority among duplicates), then
/// a model-group name match (lowest priority within the group).
fn select_candidate<'a>(
    model_groups: &'a [ModelGroup],
    model_name: &str,
) -> Option<&'a ProviderModel> {
    let mut exact: Option<&ProviderModel> = None;
    for group in model_groups {
        for candidate in &group.models {
            if candidate.model != model_name {
                continue;
            }
            if exact.is_none_or(|current| candidate.priority < current.priority) {
                exact = Some(candidate);
            }
        }
    }
    if exact.is_some() {
        return exact;
    }
    model_groups
        .iter()
        .filter(|group| group.name == model_name)
        .flat_map(|group| group.models.iter())
        .min_by_key(|candidate| candidate.priority)
}

/// Inserts the summary as a system message prefixed exactly with
/// `[Conversation summary]`, placed after the leading system messages so it
/// represents the removed older history while recent messages are retained.
pub fn insert_summary_message(messages: &mut Vec<Message>, summary_text: &str) {
    let insert_at = messages
        .iter()
        .take_while(|m| m.role == "system")
        .count()
        .min(messages.len());
    let content = format!("[Conversation summary]\n{}", summary_text.trim());
    messages.insert(
        insert_at,
        Message {
            role: "system".to_string(),
            content: Value::String(content),
            extra: serde_json::Map::new(),
        },
    );
}

/// Renders the conversation compactly for the summary prompt: one
/// `[role] text` line per message, tool-call function names inlined for
/// assistant messages, per-message and total caps to bound the prompt.
fn render_conversation(messages: &[Message]) -> String {
    let mut out = String::from("Conversation to summarize:\n");
    for message in messages.iter().take(MAX_CONVERSATION_MESSAGES) {
        out.push_str(&format!("[{}] ", message.role));
        let text = message.content_as_text();
        let text = text.trim();
        if text.chars().count() > MAX_MESSAGE_CHARS {
            let truncated: String = text.chars().take(MAX_MESSAGE_CHARS).collect();
            out.push_str(&truncated);
            out.push_str("...[truncated]");
        } else {
            out.push_str(text);
        }
        if let Some(names) = tool_call_names(message) {
            out.push_str(&format!(" (tool_calls: {})", names.join(", ")));
        }
        out.push('\n');
    }
    out
}

/// Extracts the function names from an assistant message's `tool_calls`.
fn tool_call_names(message: &Message) -> Option<Vec<String>> {
    message
        .extra
        .get("tool_calls")
        .and_then(Value::as_array)
        .map(|calls| {
            calls
                .iter()
                .filter_map(|call| {
                    call.pointer("/function/name")
                        .and_then(Value::as_str)
                        .map(str::to_string)
                })
                .collect()
        })
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::{json, Map};
    use wiremock::matchers::{method, path};
    use wiremock::{Mock, MockServer, ResponseTemplate};

    fn message(role: &str, content: &str) -> Message {
        Message {
            role: role.to_string(),
            content: Value::String(content.to_string()),
            extra: Map::new(),
        }
    }

    fn assistant_tool_call(id: &str, name: &str) -> Message {
        Message {
            role: "assistant".to_string(),
            content: Value::Null,
            extra: {
                let mut extra = Map::new();
                extra.insert(
                    "tool_calls".to_string(),
                    json!([{
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
            content: Value::String(content.to_string()),
            extra: {
                let mut extra = Map::new();
                extra.insert(
                    "tool_call_id".to_string(),
                    Value::String(id.to_string()),
                );
                extra
            },
        }
    }

    fn openai_request(messages: Vec<Message>) -> OpenAIRequest {
        OpenAIRequest {
            model: "gpt-4o-mini".to_string(),
            messages,
            stream: false,
            temperature: None,
            max_tokens: None,
            extra: Map::new(),
        }
    }

    fn test_config(base_url: &str) -> Arc<RwLock<Config>> {
        let yaml = format!(
            r#"
server:
  host: 127.0.0.1
  port: 0
providers:
  - name: sump
    type: openai
    base_url: "{base_url}"
model_groups:
  - name: grp
    models:
      - provider: sump
        model: sum-model
"#
        );
        let config: Config = serde_yaml::from_str(&yaml).expect("test config must parse");
        Arc::new(RwLock::new(config))
    }

    fn context_config_with(model: &str, timeout_seconds: u64, keep_recent: usize) -> ContextConfig {
        ContextConfig {
            summarize_model: Some(model.to_string()),
            summarize_timeout_seconds: timeout_seconds,
            tool_result_keep_recent: keep_recent,
            ..Default::default()
        }
    }

    #[test]
    fn select_candidate_prefers_exact_model_then_lowest_priority() {
        let groups: Vec<ModelGroup> = serde_yaml::from_str(
            r#"
- name: grp-a
  models:
    - provider: p1
      model: sum-model
      priority: 50
    - provider: p2
      model: other-model
      priority: 10
- name: grp-b
  models:
    - provider: p3
      model: third-model
      priority: 5
"#,
        )
        .expect("groups must parse");

        let exact = select_candidate(&groups, "sum-model").expect("exact match");
        assert_eq!((exact.provider.as_str(), exact.model.as_str()), ("p1", "sum-model"));

        let by_group = select_candidate(&groups, "grp-a").expect("group match");
        assert_eq!(
            (by_group.provider.as_str(), by_group.model.as_str()),
            ("p2", "other-model")
        );

        assert!(select_candidate(&groups, "missing").is_none());
    }

    #[test]
    fn build_request_body_sends_prompt_and_transcript_as_user_message() {
        let body = GatewaySummarizer::build_request_body(
            "sum-model",
            &[
                message("system", "You are helpful."),
                message("user", "hello there"),
                assistant_tool_call("call_1", "get_files"),
            ],
        );

        assert_eq!(body["model"], "sum-model");
        assert_eq!(body["stream"], false);
        assert_eq!(body["temperature"], 0.1);
        let messages = body["messages"].as_array().expect("messages array");
        assert_eq!(messages.len(), 1);
        assert_eq!(messages[0]["role"], "user");
        let content = messages[0]["content"].as_str().expect("content");
        assert!(content.contains("Goals and constraints"));
        assert!(content.contains("Unresolved tasks and open questions"));
        assert!(content.contains("[system] You are helpful."));
        assert!(content.contains("[user] hello there"));
        assert!(content.contains("(tool_calls: get_files)"));
    }

    #[test]
    fn insert_summary_message_prefixes_and_positions_after_system_prompt() {
        let mut messages = vec![
            message("system", "You are helpful."),
            message("system", "Extra rules."),
            message("user", "recent question"),
        ];

        insert_summary_message(&mut messages, "1. Goals and constraints\n- ship it");

        assert_eq!(messages.len(), 4);
        assert_eq!(messages[0].content_as_text(), "You are helpful.");
        assert_eq!(messages[1].content_as_text(), "Extra rules.");
        assert_eq!(messages[2].role, "system");
        assert!(messages[2]
            .content_as_text()
            .starts_with("[Conversation summary]\n1. Goals and constraints"));
        assert_eq!(messages[3].content_as_text(), "recent question");
    }

    #[tokio::test]
    async fn summarize_success_inserts_summary_and_trims_old_messages() {
        let server = MockServer::start().await;
        let summary = "1. Goals and constraints\n- ship the overflow feature\n2. Decisions made\n- reuse the memory adapter pattern\n3. Facts, file paths, and identifiers\n- crates/ai-gateway/src/context/summarizer.rs\n4. Unresolved tasks and open questions\n- router integration pending";
        Mock::given(method("POST"))
            .and(path("/v1/chat/completions"))
            .respond_with(ResponseTemplate::new(200).set_body_json(json!({
                "choices": [{"message": {"role": "assistant", "content": summary}}]
            })))
            .expect(1)
            .mount(&server)
            .await;

        let summarizer = GatewaySummarizer::new(test_config(&server.uri()));

        let mut messages = vec![message("system", "You are helpful.")];
        for turn in 0..12 {
            messages.push(message(
                "user",
                &format!("early turn marker {turn}: we need to ship the overflow feature soon"),
            ));
            messages.push(message(
                "assistant",
                &format!("acknowledged reply {turn} with some detail about the plan"),
            ));
        }
        messages.push(message("user", "final question about the summary"));
        let mut request = openai_request(messages);

        let total = TokenCounter::new().count_request(&request);
        let budget = total * 3 / 4;
        let context_config = context_config_with("sum-model", 10, 3);
        let options = ReductionOptions::from_config(
            ContextReductionStrategy::SummarizeThenTrim,
            budget,
            &context_config,
        );

        let outcome = summarizer
            .reduce_with_summarizer(&mut request, &options, &context_config)
            .await;

        assert!(outcome.summarized);
        assert!(!outcome.fallback_from_summarization);
        assert!(outcome.fits);
        assert!(outcome.final_tokens <= budget);
        assert!(outcome.groups_removed >= 1);

        assert_eq!(request.messages[0].content_as_text(), "You are helpful.");
        let summary_message = &request.messages[1];
        assert_eq!(summary_message.role, "system");
        let summary_text = summary_message.content_as_text();
        assert!(summary_text.starts_with("[Conversation summary]\n"));
        assert!(summary_text.contains("overflow feature"));

        assert!(!request
            .messages
            .iter()
            .any(|m| m.content_as_text().contains("early turn marker 0")));
        assert!(request
            .messages
            .iter()
            .any(|m| m.content_as_text().contains("final question about the summary")));

        server.verify().await;
    }

#[tokio::test]
async fn summary_trigger_skips_below_boundary_and_summarizes_above_it() {
let server = MockServer::start().await;
Mock::given(method("POST"))
.and(path("/v1/chat/completions"))
.respond_with(ResponseTemplate::new(200).set_body_json(json!({
"choices": [{"message": {"role": "assistant", "content": "Goals: finish the work."}}]
})))
.expect(1)
.mount(&server).await;
let summarizer = GatewaySummarizer::new(test_config(&server.uri()));
let original = openai_request((0..40).map(|turn| {
message(if turn % 2 == 0 { "user" } else { "assistant" },
&format!("history turn {turn} with enough detail for useful summarization and trimming"))
}).collect());
let estimate = TokenCounter::new().count_request(&original);
let budget = estimate * 2;
for (trigger, summarized) in [(51, false), (50, false), (49, true)] {
let mut config = context_config_with("sum-model", 10, 3);
config.summarize_target_percent = 40;
config.summarize_trigger_percent = trigger;
let options = ReductionOptions::from_config(ContextReductionStrategy::SummarizeThenTrim, budget, &config);
let mut request = original.clone();
let outcome = summarizer.reduce_with_summarizer(&mut request, &options, &config).await;
assert_eq!(outcome.summarized, summarized, "trigger={trigger}");
assert!(!outcome.fallback_from_summarization);
assert!(outcome.fits);
if summarized {
assert!(request.messages.iter().any(|message| message.content_as_text().starts_with("[Conversation summary]")));
} else {
assert_eq!(outcome.strategy_applied, ContextReductionStrategy::ClearToolsThenRemove);
assert_eq!(request.messages.len(), original.messages.len());
}
}
server.verify().await;
}

#[tokio::test]
async fn summarize_timeout_falls_back_to_clear_tools_then_remove() {
        let server = MockServer::start().await;
        Mock::given(method("POST"))
            .and(path("/v1/chat/completions"))
            .respond_with(
                ResponseTemplate::new(200)
                    .set_body_json(json!({
                        "choices": [{"message": {"role": "assistant", "content": "irrelevant"}}]
                    }))
                    .set_delay(Duration::from_secs(10)),
            )
            .expect(1)
            .mount(&server)
            .await;

        let summarizer = GatewaySummarizer::new(test_config(&server.uri()));
        let mut request = openai_request(vec![
            message("system", "You are helpful."),
            message("user", "list the files"),
            assistant_tool_call("call_1", "get_files"),
            tool_result("call_1", &"file listing row ".repeat(40)),
            assistant_tool_call("call_2", "get_more"),
            tool_result("call_2", &"more file listing row ".repeat(40)),
            message("user", "summarize what you found"),
        ]);

        let total = TokenCounter::new().count_request(&request);
        let budget = total * 3 / 4;
        let context_config = context_config_with("grp", 1, 1);
        let options = ReductionOptions::from_config(
            ContextReductionStrategy::SummarizeThenTrim,
            budget,
            &context_config,
        );

        let start = std::time::Instant::now();
        let outcome = summarizer
            .reduce_with_summarizer(&mut request, &options, &context_config)
            .await;

        assert!(
            start.elapsed() < Duration::from_secs(5),
            "fallback must not wait for the delayed provider response"
        );
        assert!(!outcome.summarized);
        assert!(outcome.fallback_from_summarization);
        assert_eq!(
            outcome.strategy_applied,
            ContextReductionStrategy::ClearToolsThenRemove
        );
        assert_eq!(outcome.tool_results_cleared, 1);
        assert!(outcome.fits);

        let recent = request
            .messages
            .iter()
            .find(|m| {
                m.extra.get("tool_call_id").and_then(Value::as_str) == Some("call_2")
            })
            .expect("recent tool result retained");
        assert!(recent.content_as_text().contains("more file listing row"));
    }

    #[tokio::test]
    async fn summarize_provider_error_falls_back_to_clear_tools_then_remove() {
        let server = MockServer::start().await;
        Mock::given(method("POST"))
            .and(path("/v1/chat/completions"))
            .respond_with(ResponseTemplate::new(500).set_body_string("upstream exploded"))
            .expect(1)
            .mount(&server)
            .await;

        let summarizer = GatewaySummarizer::new(test_config(&server.uri()));
        let mut request = openai_request(vec![
            message("system", "You are helpful."),
            message("user", "list the files"),
            assistant_tool_call("call_1", "get_files"),
            tool_result("call_1", &"file listing row ".repeat(40)),
            assistant_tool_call("call_2", "get_more"),
            tool_result("call_2", &"more file listing row ".repeat(40)),
            message("user", "summarize what you found"),
        ]);

        let total = TokenCounter::new().count_request(&request);
        let budget = total * 3 / 4;
        let context_config = context_config_with("sum-model", 10, 1);
        let options = ReductionOptions::from_config(
            ContextReductionStrategy::SummarizeThenTrim,
            budget,
            &context_config,
        );

        let outcome = summarizer
            .reduce_with_summarizer(&mut request, &options, &context_config)
            .await;

        assert!(!outcome.summarized);
        assert!(outcome.fallback_from_summarization);
        assert_eq!(
            outcome.strategy_applied,
            ContextReductionStrategy::ClearToolsThenRemove
        );
        assert_eq!(outcome.tool_results_cleared, 1);
        assert!(outcome.fits);
        assert_eq!(outcome.original_tokens, total);
    }

#[tokio::test]
async fn summarize_slow_body_times_out_and_falls_back() {
// M-1 regression: headers arrive promptly but the BODY dribbles in slower
// than the configured timeout. The whole send+read/parse exchange is
// bounded — the fallback must fire within the timeout budget instead of
// waiting on the client's 60s default.
let server = MockServer::start().await;
Mock::given(method("POST"))
.and(path("/v1/chat/completions"))
.respond_with(
ResponseTemplate::new(200)
.set_body_json(json!({
"choices": [{"message": {"role": "assistant", "content": "slow summary"}}]
}))
// Body slower than the 1s summarizer timeout below.
.set_delay(Duration::from_secs(5)),
)
.expect(1)
.mount(&server)
.await;

let summarizer = GatewaySummarizer::new(test_config(&server.uri()));
let mut request = openai_request(vec![
message("system", "You are helpful."),
message("user", "list the files"),
assistant_tool_call("call_1", "get_files"),
tool_result("call_1", &"file listing row ".repeat(40)),
assistant_tool_call("call_2", "get_more"),
tool_result("call_2", &"more file listing row ".repeat(40)),
message("user", "summarize what you found"),
]);

let total = TokenCounter::new().count_request(&request);
let budget = total * 3 / 4;
let context_config = context_config_with("sum-model", 1, 1);
let options = ReductionOptions::from_config(
ContextReductionStrategy::SummarizeThenTrim,
budget,
&context_config,
);

let start = std::time::Instant::now();
let outcome = summarizer
.reduce_with_summarizer(&mut request, &options, &context_config)
.await;

assert!(
start.elapsed() < Duration::from_secs(4),
"slow body must be bounded by the configured timeout, not the 60s client default"
);
assert!(!outcome.summarized);
assert!(outcome.fallback_from_summarization);
assert_eq!(
outcome.strategy_applied,
ContextReductionStrategy::ClearToolsThenRemove
);
assert!(outcome.fits);
}

#[tokio::test]
async fn summarize_internal_request_carries_tag_and_informational_header() {
// M-2: the internal marker is enforced on the request struct AND surfaced
// as a sanitized constant-value informational header on the wire, while
// the call itself is a structural direct provider call (no router
// recursion, no rate-token debit — proven by the mock receiving the
// request directly and the tag/header assertions below).
let server = MockServer::start().await;
Mock::given(method("POST"))
.and(path("/v1/chat/completions"))
.respond_with(ResponseTemplate::new(200).set_body_json(json!({
"choices": [{"message": {"role": "assistant", "content": "Goals: done."}}]
})))
.expect(1)
.mount(&server)
.await;

let internal_messages = [
message("system", "You are helpful."),
message("user", "hello"),
];
let internal = SummarizerRequest::internal(
"sum-model",
&internal_messages,
5,
);
assert_eq!(internal.internal_tag, CONTEXT_SUMMARIZATION_INTERNAL_TAG);
assert!(internal.bypass_memory, "bypass is structural: the direct provider call never enters memory injection");

let summarizer = GatewaySummarizer::new(test_config(&server.uri()));
let summary = summarizer.summarize_conversation(internal).await.unwrap();
assert_eq!(summary, "Goals: done.");

let received = server.received_requests().await.expect("captured");
assert_eq!(received.len(), 1, "exactly one direct provider call — no router recursion/retry");
let sent = &received[0];
assert_eq!(
sent.headers
.get(CONTEXT_SUMMARIZATION_HEADER)
.and_then(|value| value.to_str().ok()),
Some(CONTEXT_SUMMARIZATION_INTERNAL_TAG),
"sanitized constant-value informational header present"
);
}
}
