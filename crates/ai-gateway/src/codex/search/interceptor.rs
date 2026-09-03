//! Tool interceptor — agent-loop wrapper for gateway-handled tool calls.
//!
//! Wraps a Codex provider completion with an agent loop that detects
//! `codex_search`/`codex_web` tool calls, executes them, and resubmits
//! results until the model produces a final answer or the iteration
//! limit is reached.

use std::sync::Arc;
use std::time::{Duration, Instant};

use serde_json::{json, Value};

use crate::codex::search::executor::SearchExecutor;
use crate::codex::search::models::{CodexSearchArgs, CodexWebArgs, ToolResult};
use crate::error::GatewayError;
use crate::models::openai::{Message, OpenAIRequest, OpenAIResponse};
use crate::providers::{ProviderClient, ProviderResponse};

const GATEWAY_TOOLS: [&str; 2] = ["codex_search", "codex_web"];
const ITERATION_LIMIT_MESSAGE: &str = "Codex search agent loop iteration limit reached. \
Please refine your request or provide the information directly.";
const CONTINUATION_NUDGE_MESSAGE: &str = "The gateway executed the codex_search/codex_web tool \
call server-side; the result is already in the conversation above. This is not a new user \
request: do not acknowledge, summarize, or restate the search result. Continue the user's \
original task. If any work remains, emit your next native tool call now; reply with plain \
text only when the entire task is complete.";

/// Result of intercepting a provider response through the agent loop.
#[allow(dead_code)]
pub struct InterceptResult {
    pub response: OpenAIResponse,
    pub pending_client_tool_calls: Vec<Value>,
    pub iteration_limit_reached: bool,
    pub total_latency_ms: u64,
}

/// Wraps a Codex provider completion with an agent loop.
pub struct ToolInterceptor {
    executor: Arc<SearchExecutor>,
    max_iterations: u32,
    output_to_chat: bool,
    /// Wall-clock ceiling for the whole loop.
    ///
    /// `max_iterations` alone bounds the *number* of model round trips, not
    /// their duration: each resubmit runs through the normal retry path under
    /// the provider's own `total_timeout_seconds`, so the worst case is
    /// iterations × that timeout. From the client's point of view this is still
    /// one request, and on a streaming request the global gateway deadline does
    /// not apply (it only bounds producing response headers), so without this
    /// budget the loop can outlive every configured limit and the connection
    /// just sits there.
    budget: Duration,
}

impl ToolInterceptor {
    pub fn new(
        executor: Arc<SearchExecutor>,
        max_iterations: u32,
        output_to_chat: bool,
        budget: Duration,
    ) -> Self {
        Self {
            executor,
            max_iterations,
            output_to_chat,
            budget,
        }
    }

    /// Run the agent-loop interception.
    ///
    /// `provider` is the same Codex provider that generated the initial
    /// response — the loop resubmits through it directly without
    /// re-running model-group routing or failover.
    pub async fn intercept(
        &self,
        provider: &dyn ProviderClient,
        mut request: OpenAIRequest,
        initial_response: OpenAIResponse,
    ) -> Result<InterceptResult, GatewayError> {
        let start = Instant::now();
        let mut current_response = initial_response;
        let mut iterations: u32 = 0;
 let mut all_results: Vec<Value> = Vec::new();
 let mut seen_call_ids: std::collections::HashSet<String> = std::collections::HashSet::new();
 // One-shot guard for the reactive continuation nudge (see the
 // tool_calls-empty branch of the loop).
 let mut nudged = false;

        loop {
            let tool_calls = extract_tool_calls(&current_response);

 if tool_calls.is_empty() {
 // Reactive continuation nudge: the model produced a text-only
 // response right after gateway-executed search results. Client
 // harnesses treat "text + no tool_calls" as end of the whole
 // turn, so an interim answer here ends the task prematurely.
 // Give the model exactly one extra round-trip to continue via
 // its own (client) tools before accepting the stop. The nudge
 // lives only in the resubmitted request — the client never
 // sees it. Skipped when no search ran this turn, when the
 // nudge already fired, or when the iteration budget is spent
 // (the explicit limit branch handles that case).
 if !all_results.is_empty()
 && !nudged
 && iterations < self.max_iterations
 && start.elapsed() < self.budget
 {
 append_continuation_nudge(&mut request, &current_response);
 nudged = true;
 iterations += 1;
 let next_response = self.resubmit(provider, &request).await?;
 current_response = next_response;
 continue;
 }
 if self.output_to_chat {
 append_results_to_chat(&mut current_response, &all_results);
 }
 attach_results_to_response(&mut current_response, &all_results);
 return Ok(InterceptResult {
 response: current_response,
 pending_client_tool_calls: Vec::new(),
 iteration_limit_reached: false,
 total_latency_ms: start.elapsed().as_millis() as u64,
 });
 }

            let (gateway_calls, client_calls): (Vec<&Value>, Vec<&Value>) =
                tool_calls.iter().partition(|tc| is_gateway_tool_call(tc));

            if !client_calls.is_empty() {
                append_assistant_message(&mut request, &current_response);
                let executed_results = self
                    .execute_and_append_tool_results(&mut request, &gateway_calls)
                    .await;
                extend_results_deduped(&mut all_results, &mut seen_call_ids, executed_results);
                if self.output_to_chat {
                    append_results_to_chat(&mut current_response, &all_results);
                }
                attach_results_to_response(&mut current_response, &all_results);
                strip_gateway_tool_calls(&mut current_response);

                return Ok(InterceptResult {
                    response: current_response,
                    pending_client_tool_calls: client_calls.into_iter().cloned().collect(),
                    iteration_limit_reached: false,
                    total_latency_ms: start.elapsed().as_millis() as u64,
                });
            }

            // Stop looping when either budget is spent. The iteration limit gets
            // one final resubmit so the model can answer with what it already
            // has; a spent wall-clock budget does not, because another round
            // trip is precisely what there is no time left for.
            let iterations_spent = iterations >= self.max_iterations;
            let time_spent = start.elapsed() >= self.budget;
            if iterations_spent || time_spent {
                let mut final_response = if time_spent {
                    tracing::warn!(
                        elapsed_ms = start.elapsed().as_millis() as u64,
                        budget_ms = self.budget.as_millis() as u64,
                        iterations,
                        "Codex search loop exceeded its wall-clock budget; returning current turn without another model round trip"
                    );
                    current_response
                } else {
                    append_iteration_limit_message(&mut request);
                    // Drop the gateway tool definitions before the final turn so
                    // the model cannot emit yet another search call. If it did,
                    // `strip_gateway_tool_calls` below would delete it and leave
                    // an assistant message with neither content nor tool calls,
                    // which clients read as a finished-but-empty turn and stop on.
                    remove_gateway_tools(&mut request);
                    self.resubmit(provider, &request).await?
                };
                if self.output_to_chat {
                    append_results_to_chat(&mut final_response, &all_results);
                }
                attach_results_to_response(&mut final_response, &all_results);
                strip_gateway_tool_calls(&mut final_response);
                return Ok(InterceptResult {
                    response: final_response,
                    pending_client_tool_calls: Vec::new(),
                    iteration_limit_reached: true,
                    total_latency_ms: start.elapsed().as_millis() as u64,
                });
            }

            append_assistant_message(&mut request, &current_response);
            let executed_results = self
                .execute_and_append_tool_results(&mut request, &gateway_calls)
                .await;
            extend_results_deduped(&mut all_results, &mut seen_call_ids, executed_results);

            iterations += 1;

            let next_response = self.resubmit(provider, &request).await?;
            current_response = next_response;
        }
    }

    async fn execute_and_append_tool_results(
        &self,
        request: &mut OpenAIRequest,
        gateway_calls: &[&Value],
    ) -> Vec<Value> {
        let mut executed_results = Vec::with_capacity(gateway_calls.len());
        for tc in gateway_calls {
            let call_id = tc
                .get("id")
                .and_then(|v| v.as_str())
                .unwrap_or("unknown")
                .to_string();
            let tool_name = tc
                .get("function")
                .and_then(|f| f.get("name"))
                .and_then(|n| n.as_str())
                .unwrap_or("")
                .to_string();
            let args_str = tc
                .get("function")
                .and_then(|f| f.get("arguments"))
                .and_then(|a| a.as_str())
                .unwrap_or("{}");
        let args_value: Value =
            serde_json::from_str(args_str).unwrap_or(Value::Object(Default::default()));

        let mut query: Option<String> = None;
        let result = match tool_name.as_str() {
            "codex_search" => {
            let parsed: CodexSearchArgs = serde_json::from_value(args_value.clone())
                .unwrap_or(CodexSearchArgs {
                    q: String::new(),
                    domains: None,
                    recency: None,
                    response_length: None,
                });
                query = Some(parsed.q.clone());
                self.executor.execute_search(parsed).await
            }
                "codex_web" => {
                    let parsed: CodexWebArgs = serde_json::from_value(args_value.clone())
                        .unwrap_or(CodexWebArgs {
                            session_id: None,
                            commands: None,
                            response_length: None,
                        });
                    self.executor.execute_web(parsed).await
                }
                _ => ToolResult {
                    content: "Unknown gateway tool".to_string(),
                    is_error: true,
                    session_id: None,
                },
            };

        let tool_message = build_tool_message(&call_id, &result);
        let mut executed = json!({
            "tool_call_id": call_id,
            "name": tool_name,
            "content": result.content,
            "is_error": result.is_error,
            "session_id": result.session_id,
        });
        if let Some(q) = query {
            executed["query"] = Value::String(q);
        }
        executed_results.push(executed);
        request.messages.push(tool_message);
        }
        executed_results
    }

    async fn resubmit(
        &self,
        provider: &dyn ProviderClient,
        request: &OpenAIRequest,
    ) -> Result<OpenAIResponse, GatewayError> {
        let ProviderResponse { response, .. } = provider.chat_completion(request.clone()).await?;
        Ok(response)
    }
}

fn extract_tool_calls(response: &OpenAIResponse) -> Vec<Value> {
    response
        .choices
        .first()
        .and_then(|c| c.message.extra.get("tool_calls"))
        .and_then(|v| v.as_array())
        .cloned()
        .unwrap_or_default()
}

fn is_gateway_tool_call(tc: &Value) -> bool {
    tc.get("function")
        .and_then(|f| f.get("name"))
        .and_then(|n| n.as_str())
        .map(|name| GATEWAY_TOOLS.contains(&name))
        .unwrap_or(false)
}

fn append_assistant_message(request: &mut OpenAIRequest, response: &OpenAIResponse) {
    if let Some(choice) = response.choices.first() {
        let mut extra = choice.message.extra.clone();
        if let Some(tool_calls) = choice.message.extra.get("tool_calls") {
            extra.insert("tool_calls".to_string(), tool_calls.clone());
        }
        request.messages.push(Message {
            role: "assistant".to_string(),
            content: choice.message.content.clone(),
            extra,
        });
    }
}

fn build_tool_message(call_id: &str, result: &ToolResult) -> Message {
    let mut extra = serde_json::Map::new();
    extra.insert(
        "tool_call_id".to_string(),
        Value::String(call_id.to_string()),
    );
    Message {
        role: "tool".to_string(),
        content: Value::String(result.content.clone()),
        extra,
    }
}

/// Remove the gateway's own tool definitions from a request.
///
/// Used for the final, limit-reached resubmit so the model is not offered a tool
/// whose call would immediately be stripped back out. The `tools` key is dropped
/// entirely when nothing else remains, because several providers reject an empty
/// `tools` array.
fn remove_gateway_tools(request: &mut OpenAIRequest) {
    let Some(tools) = request
        .extra
        .get_mut("tools")
        .and_then(Value::as_array_mut)
    else {
        return;
    };
    tools.retain(|tool| {
        tool.get("function")
            .and_then(|f| f.get("name"))
            .and_then(|n| n.as_str())
            .map(|name| !GATEWAY_TOOLS.contains(&name))
            .unwrap_or(true)
    });
    if tools.is_empty() {
        request.extra.remove("tools");
    }
}

fn append_iteration_limit_message(request: &mut OpenAIRequest) {
 request.messages.push(Message {
 role: "system".to_string(),
 content: Value::String(ITERATION_LIMIT_MESSAGE.to_string()),
 extra: serde_json::Map::new(),
 });
}

/// Append the model's text-only response as an assistant message,
/// followed by a one-shot user-role nudge to continue the task. The
/// nudge is placed after any tool messages (assistant -> tool ->
/// user), which strict providers accept; interleaving it between the
/// assistant tool_calls message and its tool results would not.
fn append_continuation_nudge(request: &mut OpenAIRequest, response: &OpenAIResponse) {
 append_assistant_message(request, response);
 request.messages.push(Message {
 role: "user".to_string(),
 content: Value::String(CONTINUATION_NUDGE_MESSAGE.to_string()),
 extra: serde_json::Map::new(),
 });
}

/// Append executed results to `all_results`, skipping call IDs already seen.
fn extend_results_deduped(
    all_results: &mut Vec<Value>,
    seen_call_ids: &mut std::collections::HashSet<String>,
    executed_results: Vec<Value>,
) {
    for result in executed_results {
        let call_id = result
            .get("tool_call_id")
            .and_then(|v| v.as_str())
            .unwrap_or("unknown")
            .to_string();
        if seen_call_ids.insert(call_id) {
            all_results.push(result);
        }
    }
}

/// Attach executed results to the response as `codex_search_tool_results`
/// metadata for programmatic consumption by API clients.
fn attach_results_to_response(response: &mut OpenAIResponse, all_results: &[Value]) {
    if all_results.is_empty() {
        return;
    }
    response.extra.insert(
        "codex_search_tool_results".to_string(),
        Value::Array(all_results.to_vec()),
    );
}

/// Append executed results to the assistant message content so they are
/// visible in chat history. Unlike tool-call metadata, plain content
/// survives downstream context/tool compression, preventing the model
/// from forgetting that the search already ran.
fn append_results_to_chat(response: &mut OpenAIResponse, all_results: &[Value]) {
    if all_results.is_empty() {
        return;
    }
    if let Some(choice) = response.choices.first_mut() {
        let block = format_results_block(all_results);
        let existing = choice.message.content_as_text();
        let combined = if existing.trim().is_empty() {
            block
        } else {
            format!("{}\n\n{}", existing, block)
        };
        choice.message.content = Value::String(combined);
    }
}

fn format_results_block(results: &[Value]) -> String {
    let mut lines = vec!["--- codex search results ---".to_string()];
    for result in results {
        let name = result.get("name").and_then(|v| v.as_str()).unwrap_or("tool");
        let header = match result.get("query").and_then(|v| v.as_str()) {
            Some(q) if !q.is_empty() => format!("[{} \"{}\"]", name, q),
            _ => format!("[{}]", name),
        };
        lines.push(header);
        if let Some(content) = result.get("content").and_then(|v| v.as_str()) {
            lines.push(content.to_string());
        }
    }
    lines.push("--- end codex search results ---".to_string());
    lines.join("\n")
}

/// Remove gateway-executed tool calls (codex_search/codex_web) from the
/// response's tool_calls array. The gateway has already executed them
/// server-side; leaking them back to the client causes harnesses to reject
/// the response ("tried to call unavailable tool"). Client tool calls are
/// preserved untouched.
pub(crate) fn strip_gateway_tool_calls(response: &mut OpenAIResponse) {
    if let Some(choice) = response.choices.first_mut() {
        let should_remove = {
            let calls = choice
                .message
                .extra
                .get_mut("tool_calls")
                .and_then(|v| v.as_array_mut());
            if let Some(calls) = calls {
                calls.retain(|tc| !is_gateway_tool_call(tc));
                calls.is_empty()
            } else {
                false
            }
        };
        if should_remove {
            choice.message.extra.remove("tool_calls");
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::codex::search::executor::Sleeper;
    use crate::codex::search::metrics::SearchMetrics;
    use crate::oauth::{OAuthManager, UsageTracker};
    use crate::providers::{Model, ProviderResponse, SSEEvent};
    use async_trait::async_trait;
    use serde_json::json;
    use std::pin::Pin;
    use std::time::Duration;

    fn make_tool_call(id: &str, name: &str, args: &str) -> Value {
        json!({
            "id": id,
            "type": "function",
            "function": {
                "name": name,
                "arguments": args
            }
        })
    }

    fn make_response_with_tool_calls(tool_calls: Vec<Value>) -> OpenAIResponse {
        let mut extra = serde_json::Map::new();
        let has_calls = !tool_calls.is_empty();
        if has_calls {
            extra.insert("tool_calls".to_string(), Value::Array(tool_calls));
        }
        OpenAIResponse {
            id: "test".to_string(),
            object: "chat.completion".to_string(),
            created: 0,
            model: "gpt-4o".to_string(),
            choices: vec![crate::models::openai::Choice {
                index: 0,
                message: Message {
                    role: "assistant".to_string(),
                    content: Value::String(String::new()),
                    extra,
                },
                finish_reason: if has_calls {
                    Some("stop".to_string())
                } else {
                    Some("tool_calls".to_string())
                },
                extra: serde_json::Map::new(),
            }],
            usage: Default::default(),
            extra: serde_json::Map::new(),
        }
    }

    fn make_final_response(content: &str) -> OpenAIResponse {
        OpenAIResponse {
            id: "test".to_string(),
            object: "chat.completion".to_string(),
            created: 0,
            model: "gpt-4o".to_string(),
            choices: vec![crate::models::openai::Choice {
                index: 0,
                message: Message {
                    role: "assistant".to_string(),
                    content: Value::String(content.to_string()),
                    extra: serde_json::Map::new(),
                },
                finish_reason: Some("stop".to_string()),
                extra: serde_json::Map::new(),
            }],
            usage: Default::default(),
            extra: serde_json::Map::new(),
        }
    }

 struct MockProvider {
 responses: Vec<OpenAIResponse>,
 call_count: std::sync::atomic::AtomicU32,
 requests: std::sync::Mutex<Vec<OpenAIRequest>>,
 }

 impl MockProvider {
 fn new(responses: Vec<OpenAIResponse>) -> Self {
 Self {
 responses,
 call_count: std::sync::atomic::AtomicU32::new(0),
 requests: std::sync::Mutex::new(Vec::new()),
 }
 }

 fn calls(&self) -> u32 {
 self.call_count.load(std::sync::atomic::Ordering::Relaxed)
 }

 fn requests(&self) -> Vec<OpenAIRequest> {
 self.requests.lock().expect("mock requests mutex").clone()
 }
 }

    #[async_trait]
    impl ProviderClient for MockProvider {
 async fn chat_completion(
 &self,
 request: OpenAIRequest,
 ) -> Result<ProviderResponse, GatewayError> {
 self.requests
 .lock()
 .expect("mock requests mutex")
 .push(request);
 let idx = self
                .call_count
                .fetch_add(1, std::sync::atomic::Ordering::Relaxed) as usize;
            let resp = self
                .responses
                .get(idx)
                .cloned()
                .unwrap_or_else(|| make_final_response("fallback"));
            Ok(ProviderResponse {
                response: resp,
                provider_name: "mock".to_string(),
                latency_ms: 0,
            })
        }

        async fn chat_completion_stream(
            &self,
            _request: OpenAIRequest,
        ) -> Result<
            Pin<Box<dyn futures::Stream<Item = Result<SSEEvent, GatewayError>> + Send>>,
            GatewayError,
        > {
            Err(GatewayError::Provider {
                provider: "mock".to_string(),
                message: "streaming not supported".to_string(),
                status_code: None,
            })
        }

        async fn list_models(&self) -> Result<Vec<Model>, GatewayError> {
            Ok(vec![])
        }

        fn provider_name(&self) -> &str {
            "mock"
        }
    }

    struct NoopSleeper;

    #[async_trait]
    impl Sleeper for NoopSleeper {
        async fn sleep(&self, _duration: Duration) {}
    }

    fn make_executor() -> Arc<SearchExecutor> {
        let store = crate::oauth::OAuthTokenStore::new(
            std::env::temp_dir().join(format!("oauth-test-{}", uuid::Uuid::new_v4())),
        );
        let oauth = Arc::new(OAuthManager::new(store, reqwest::Client::new()));
        let usage_tracker = Arc::new(UsageTracker::new());
        let metrics = Arc::new(SearchMetrics::new());
        Arc::new(SearchExecutor::with_sleeper(
            reqwest::Client::new(),
            oauth,
            usage_tracker,
            metrics,
            "http://localhost/search".to_string(),
            Duration::from_secs(15),
            Arc::new(NoopSleeper),
        ))
    }

    fn make_request() -> OpenAIRequest {
        OpenAIRequest {
            model: "gpt-4o".to_string(),
            messages: vec![Message {
                role: "user".to_string(),
                content: Value::String("test".to_string()),
                extra: serde_json::Map::new(),
            }],
            stream: false,
            temperature: None,
            max_tokens: None,
            extra: serde_json::Map::new(),
        }
    }

/// Build an interceptor with a budget large enough never to fire, so the
/// existing tests exercise the iteration limit rather than the clock. The
/// budget itself is covered by `wall_clock_budget_stops_the_loop_without_another_round_trip`.
fn test_interceptor(
    executor: Arc<SearchExecutor>,
    max_iterations: u32,
    output_to_chat: bool,
) -> ToolInterceptor {
    ToolInterceptor::new(
        executor,
        max_iterations,
        output_to_chat,
        Duration::from_secs(3600),
    )
}

#[tokio::test]
async fn no_tool_calls_returns_immediately() {
    let executor = make_executor();
    let interceptor = test_interceptor(executor, 5, false);
    let provider = MockProvider::new(vec![make_final_response("hello")]);

    let result = interceptor
        .intercept(&provider, make_request(), make_final_response("hello"))
        .await
        .unwrap();

    assert!(!result.iteration_limit_reached);
    assert!(result.pending_client_tool_calls.is_empty());
    assert_eq!(provider.calls(), 0);
}

#[tokio::test]
async fn iteration_limit_enforced() {
    let executor = make_executor();
    let interceptor = test_interceptor(executor, 2, false);
        let search_call = make_tool_call("call_1", "codex_search", r#"{"q":"test"}"#);
        let provider = MockProvider::new(vec![
            make_response_with_tool_calls(vec![search_call.clone()]),
            make_response_with_tool_calls(vec![search_call.clone()]),
            make_response_with_tool_calls(vec![search_call.clone()]),
            make_response_with_tool_calls(vec![search_call.clone()]),
        ]);
        let initial = make_response_with_tool_calls(vec![search_call.clone()]);
        let result = interceptor
            .intercept(&provider, make_request(), initial)
            .await
            .unwrap();

        assert!(result.iteration_limit_reached);
    }

#[tokio::test]
async fn mixed_gateway_and_client_tools_stops_loop() {
    let executor = make_executor();
    let interceptor = test_interceptor(executor, 5, false);
        let search_call = make_tool_call("call_1", "codex_search", r#"{"q":"test"}"#);
        let client_call = make_tool_call("call_2", "client_custom_tool", r#"{"x":1}"#);
        let provider = MockProvider::new(vec![]);

        let initial = make_response_with_tool_calls(vec![search_call, client_call.clone()]);
        let result = interceptor
            .intercept(&provider, make_request(), initial)
            .await
            .unwrap();

        assert!(!result.iteration_limit_reached);
        assert_eq!(result.pending_client_tool_calls.len(), 1);
        assert_eq!(
            result.pending_client_tool_calls[0]
                .get("function")
                .and_then(|f| f.get("name"))
                .and_then(|n| n.as_str()),
            Some("client_custom_tool")
        );
        assert_eq!(provider.calls(), 0);
        let executed = result
            .response
            .extra
            .get("codex_search_tool_results")
            .and_then(|v| v.as_array())
            .expect("executed gateway results should be returned");
        assert_eq!(executed.len(), 1);
        assert_eq!(
            executed[0].get("tool_call_id").and_then(Value::as_str),
            Some("call_1")
        );
    assert_eq!(
        executed[0].get("name").and_then(Value::as_str),
        Some("codex_search")
    );
    // Gateway tool calls must be stripped from the response; the client
    // harness only knows its own tools and rejects codex_search calls.
    let remaining = result
        .response
        .choices
        .first()
        .and_then(|c| c.message.extra.get("tool_calls"))
        .and_then(|v| v.as_array())
        .expect("client tool_calls must survive");
    assert_eq!(remaining.len(), 1, "only the client call remains");
    assert_eq!(
        remaining[0]
            .get("function")
            .and_then(|f| f.get("name"))
            .and_then(|n| n.as_str()),
        Some("client_custom_tool")
    );
}

/// `max_iterations` bounds the number of model round trips, not their duration.
/// A zero budget must stop the loop immediately and, crucially, WITHOUT another
/// resubmit — another round trip is exactly what there is no time for. The
/// `MockProvider` is primed with no responses, so any resubmit attempt fails the
/// test rather than silently passing.
#[tokio::test]
async fn wall_clock_budget_stops_the_loop_without_another_round_trip() {
    let executor = make_executor();
    let interceptor = ToolInterceptor::new(executor, 5, false, Duration::ZERO);
    let search_call = make_tool_call("call_1", "codex_search", r#"{"q":"test"}"#);
    let provider = MockProvider::new(vec![]);

    let initial = make_response_with_tool_calls(vec![search_call]);
    let result = interceptor
        .intercept(&provider, make_request(), initial)
        .await
        .unwrap();

    assert!(result.iteration_limit_reached, "loop terminated on a budget");
    // The unexecutable gateway call must not survive to the client.
    assert!(
        result
            .response
            .choices
            .first()
            .and_then(|c| c.message.extra.get("tool_calls"))
            .is_none(),
        "gateway tool call stripped even on the budget path"
    );
}

/// The final limit-reached resubmit must not offer the gateway tools again.
/// Leaving them in lets the model emit another `codex_search` that the strip
/// then deletes, leaving a turn with neither content nor tool calls — which
/// clients read as a finished, empty turn and stop on.
#[test]
fn remove_gateway_tools_leaves_only_client_tools() {
    let mut request = make_request();
    request.extra.insert(
        "tools".to_string(),
        json!([
            {"type": "function", "function": {"name": "client_custom_tool"}},
            {"type": "function", "function": {"name": "codex_search"}},
            {"type": "function", "function": {"name": "codex_web"}},
        ]),
    );

    remove_gateway_tools(&mut request);

    let names: Vec<&str> = request.extra["tools"]
        .as_array()
        .unwrap()
        .iter()
        .filter_map(|t| t.pointer("/function/name").and_then(Value::as_str))
        .collect();
    assert_eq!(names, vec!["client_custom_tool"]);
}

/// When only gateway tools were injected, the key is removed rather than left as
/// an empty array — several providers reject `tools: []`.
#[test]
fn remove_gateway_tools_drops_an_emptied_tools_key() {
    let mut request = make_request();
    request.extra.insert(
        "tools".to_string(),
        json!([{"type": "function", "function": {"name": "codex_search"}}]),
    );

    remove_gateway_tools(&mut request);

    assert!(!request.extra.contains_key("tools"));
}

#[tokio::test]
async fn gateway_only_calls_are_stripped_before_client_return() {
    let executor = make_executor();
    // output_to_chat = false keeps content assertions out of the way; the
    // strip must still happen on the client-tool return path.
    let interceptor = test_interceptor(executor, 5, false);
    let search_call = make_tool_call("call_1", "codex_search", r#"{"q":"test"}"#);
    let client_call = make_tool_call("call_2", "client_custom_tool", r#"{"x":1}"#);
    let provider = MockProvider::new(vec![]);

    let initial = make_response_with_tool_calls(vec![search_call, client_call]);
    let result = interceptor
        .intercept(&provider, make_request(), initial)
        .await
        .unwrap();

    let remaining = result
        .response
        .choices
        .first()
        .and_then(|c| c.message.extra.get("tool_calls"))
        .and_then(|v| v.as_array())
        .expect("client tool_calls must survive");
    assert!(remaining
        .iter()
        .all(|tc| tc.get("function")
            .and_then(|f| f.get("name"))
            .and_then(|n| n.as_str())
            .map(|n| n != "codex_search" && n != "codex_web")
            .unwrap_or(true)));
}

#[tokio::test]
async fn single_tool_call_then_final_answer() {
 let executor = make_executor();
 let interceptor = test_interceptor(executor, 5, false);
 let search_call = make_tool_call("call_1", "codex_search", r#"{"q":"test"}"#);
 // First resubmit returns an interim text-only answer, which triggers
 // the one-shot continuation nudge; the second resubmit completes.
 let provider = MockProvider::new(vec![
 make_final_response("interim answer"),
 make_final_response("task complete"),
 ]);

 let initial = make_response_with_tool_calls(vec![search_call]);
 let result = interceptor
 .intercept(&provider, make_request(), initial)
 .await
 .unwrap();

 assert!(!result.iteration_limit_reached);
 assert!(result.pending_client_tool_calls.is_empty());
 assert_eq!(provider.calls(), 2);
 // Metadata attachment happens on the final-answer path too.
 let executed = result
 .response
 .extra
 .get("codex_search_tool_results")
 .and_then(|v| v.as_array())
 .expect("results should be attached on final-answer path");
 assert_eq!(executed.len(), 1);
 assert_eq!(
 executed[0].get("query").and_then(Value::as_str),
 Some("test")
 );
 // With output_to_chat disabled the assistant content is untouched.
 let content = result
 .response
 .choices
 .first()
 .map(|c| c.message.content_as_text())
 .unwrap_or_default();
 assert_eq!(content, "task complete");
 // The nudge resubmission carried the interim assistant answer
 // followed by a user-role continuation instruction.
 let requests = provider.requests();
 let nudged_request = &requests[1];
 let nudge = nudged_request
 .messages
 .last()
 .expect("nudge message should be present");
 assert_eq!(nudge.role, "user");
 assert!(nudge.content_as_text().contains("server-side"));
 let interim = &nudged_request.messages[nudged_request.messages.len() - 2];
 assert_eq!(interim.role, "assistant");
 assert_eq!(interim.content_as_text(), "interim answer");
}

#[tokio::test]
async fn output_to_chat_appends_results_to_content() {
    let executor = make_executor();
    let interceptor = test_interceptor(executor, 5, true);
    let search_call = make_tool_call("call_1", "codex_search", r#"{"q":"rust news"}"#);
    // Second response answers the one-shot continuation nudge so the
    // returned content is a genuine final answer.
    let provider = MockProvider::new(vec![
        make_final_response("final answer"),
        make_final_response("final answer"),
    ]);

    let initial = make_response_with_tool_calls(vec![search_call]);
    let result = interceptor
        .intercept(&provider, make_request(), initial)
        .await
        .unwrap();

    let content = result
        .response
        .choices
        .first()
        .map(|c| c.message.content_as_text())
        .expect("content should exist");
    assert!(
        content.starts_with("final answer"),
        "original content must be preserved, got: {content}"
    );
    assert!(content.contains("--- codex search results ---"));
    assert!(content.contains("[codex_search \"rust news\"]"));
    assert!(content.contains("--- end codex search results ---"));
    // Metadata is still attached alongside the chat block.
    assert!(result
        .response
        .extra
        .get("codex_search_tool_results")
        .and_then(|v| v.as_array())
        .is_some());
}

#[tokio::test]
async fn output_to_chat_appends_on_client_tool_stop() {
    let executor = make_executor();
    let interceptor = test_interceptor(executor, 5, true);
    let search_call = make_tool_call("call_1", "codex_search", r#"{"q":"test"}"#);
    let client_call = make_tool_call("call_2", "client_custom_tool", r#"{"x":1}"#);
    let provider = MockProvider::new(vec![]);

    let initial = make_response_with_tool_calls(vec![search_call, client_call]);
    let result = interceptor
        .intercept(&provider, make_request(), initial)
        .await
        .unwrap();

    let content = result
        .response
        .choices
        .first()
        .map(|c| c.message.content_as_text())
        .expect("content should exist");
    assert!(content.contains("--- codex search results ---"));
    assert!(content.contains("[codex_search \"test\"]"));
}

#[tokio::test]
async fn repeated_call_ids_are_deduplicated() {
    let executor = make_executor();
    let interceptor = test_interceptor(executor, 5, false);
    // Same call ID echoed again after resubmit — must not duplicate results.
    let search_call = make_tool_call("call_1", "codex_search", r#"{"q":"test"}"#);
    let provider = MockProvider::new(vec![
        make_response_with_tool_calls(vec![search_call.clone()]),
        make_final_response("done"),
    ]);

    let initial = make_response_with_tool_calls(vec![search_call]);
    let result = interceptor
        .intercept(&provider, make_request(), initial)
        .await
        .unwrap();

    let executed = result
        .response
        .extra
        .get("codex_search_tool_results")
        .and_then(|v| v.as_array())
        .expect("results should be attached");
    assert_eq!(
        executed.len(),
        1,
        "duplicate call_id must be deduplicated, got {executed:?}"
    );
}

#[tokio::test]
async fn continuation_nudge_fires_once_after_text_stop() {
    let executor = make_executor();
    let interceptor = test_interceptor(executor, 5, false);
    let search_call = make_tool_call("call_1", "codex_search", r#"{"q":"test"}"#);
    // The model keeps answering with text after the nudge: the stop is
    // accepted on the second text-only response (nudge is one-shot).
    let provider = MockProvider::new(vec![
        make_final_response("stop one"),
        make_final_response("stop two"),
    ]);

    let initial = make_response_with_tool_calls(vec![search_call]);
    let result = interceptor
        .intercept(&provider, make_request(), initial)
        .await
        .unwrap();

    assert_eq!(provider.calls(), 2);
    let content = result
        .response
        .choices
        .first()
        .map(|c| c.message.content_as_text())
        .unwrap_or_default();
    assert_eq!(content, "stop two");
    // Exactly one nudge message was appended across all resubmits.
    let nudges = provider
        .requests()
        .iter()
        .flat_map(|r| r.messages.iter())
        .filter(|m| m.role == "user" && m.content_as_text().contains("server-side"))
        .count();
    assert_eq!(nudges, 1);
}

#[tokio::test]
async fn continuation_nudge_respects_iteration_budget() {
    let executor = make_executor();
    // max_iterations=1: the single resubmit slot is consumed by the
    // search round-trip, so the following text-only stop must finalize
    // without a nudge.
    let interceptor = test_interceptor(executor, 1, false);
    let search_call = make_tool_call("call_1", "codex_search", r#"{"q":"test"}"#);
    let provider = MockProvider::new(vec![make_final_response("answer")]);

    let initial = make_response_with_tool_calls(vec![search_call]);
    let result = interceptor
        .intercept(&provider, make_request(), initial)
        .await
        .unwrap();

    assert_eq!(provider.calls(), 1);
    let content = result
        .response
        .choices
        .first()
        .map(|c| c.message.content_as_text())
        .unwrap_or_default();
    assert_eq!(content, "answer");
}

#[tokio::test]
async fn continuation_nudge_revives_client_tool_call() {
    let executor = make_executor();
    let interceptor = test_interceptor(executor, 5, false);
    let search_call = make_tool_call("call_1", "codex_search", r#"{"q":"test"}"#);
    let client_call = make_tool_call("call_2", "client_custom_tool", r#"{"x":1}"#);
    // After the nudge the model continues via a client tool: the loop
    // must hand the call back instead of accepting the interim text.
    let provider = MockProvider::new(vec![
        make_final_response("interim"),
        make_response_with_tool_calls(vec![client_call.clone()]),
    ]);

    let initial = make_response_with_tool_calls(vec![search_call]);
    let result = interceptor
        .intercept(&provider, make_request(), initial)
        .await
        .unwrap();

    assert_eq!(provider.calls(), 2);
    assert_eq!(result.pending_client_tool_calls.len(), 1);
    assert_eq!(
        result.pending_client_tool_calls[0]
            .get("function")
            .and_then(|f| f.get("name"))
            .and_then(|n| n.as_str()),
        Some("client_custom_tool")
    );
}
}
