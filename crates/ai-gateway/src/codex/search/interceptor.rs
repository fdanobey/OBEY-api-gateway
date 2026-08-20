//! Tool interceptor — agent-loop wrapper for gateway-handled tool calls.
//!
//! Wraps a Codex provider completion with an agent loop that detects
//! `codex_search`/`codex_web` tool calls, executes them, and resubmits
//! results until the model produces a final answer or the iteration
//! limit is reached.

use std::sync::Arc;
use std::time::Instant;

use serde_json::{json, Value};

use crate::codex::search::executor::SearchExecutor;
use crate::codex::search::models::{CodexSearchArgs, CodexWebArgs, ToolResult};
use crate::error::GatewayError;
use crate::models::openai::{Message, OpenAIRequest, OpenAIResponse};
use crate::providers::{ProviderClient, ProviderResponse};

const GATEWAY_TOOLS: [&str; 2] = ["codex_search", "codex_web"];
const ITERATION_LIMIT_MESSAGE: &str = "Codex search agent loop iteration limit reached. \
     Please refine your request or provide the information directly.";

/// Result of intercepting a provider response through the agent loop.
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
}

impl ToolInterceptor {
    pub fn new(
        executor: Arc<SearchExecutor>,
        max_iterations: u32,
        output_to_chat: bool,
    ) -> Self {
        Self {
            executor,
            max_iterations,
            output_to_chat,
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

        loop {
            let tool_calls = extract_tool_calls(&current_response);

            if tool_calls.is_empty() {
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

                return Ok(InterceptResult {
                    response: current_response,
                    pending_client_tool_calls: client_calls.into_iter().cloned().collect(),
                    iteration_limit_reached: false,
                    total_latency_ms: start.elapsed().as_millis() as u64,
                });
            }

            if iterations >= self.max_iterations {
                append_iteration_limit_message(&mut request);
                let mut final_response = self.resubmit(provider, &request).await?;
                if self.output_to_chat {
                    append_results_to_chat(&mut final_response, &all_results);
                }
                attach_results_to_response(&mut final_response, &all_results);
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

fn append_iteration_limit_message(request: &mut OpenAIRequest) {
    request.messages.push(Message {
        role: "system".to_string(),
        content: Value::String(ITERATION_LIMIT_MESSAGE.to_string()),
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
    }

    impl MockProvider {
        fn new(responses: Vec<OpenAIResponse>) -> Self {
            Self {
                responses,
                call_count: std::sync::atomic::AtomicU32::new(0),
            }
        }

        fn calls(&self) -> u32 {
            self.call_count.load(std::sync::atomic::Ordering::Relaxed)
        }
    }

    #[async_trait]
    impl ProviderClient for MockProvider {
        async fn chat_completion(
            &self,
            _request: OpenAIRequest,
        ) -> Result<ProviderResponse, GatewayError> {
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

#[tokio::test]
async fn no_tool_calls_returns_immediately() {
    let executor = make_executor();
    let interceptor = ToolInterceptor::new(executor, 5, false);
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
    let interceptor = ToolInterceptor::new(executor, 2, false);
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
    let interceptor = ToolInterceptor::new(executor, 5, false);
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
    }

#[tokio::test]
async fn single_tool_call_then_final_answer() {
    let executor = make_executor();
    let interceptor = ToolInterceptor::new(executor, 5, false);
    let search_call = make_tool_call("call_1", "codex_search", r#"{"q":"test"}"#);
    let provider = MockProvider::new(vec![make_final_response("final answer")]);

    let initial = make_response_with_tool_calls(vec![search_call]);
    let result = interceptor
        .intercept(&provider, make_request(), initial)
        .await
        .unwrap();

    assert!(!result.iteration_limit_reached);
    assert!(result.pending_client_tool_calls.is_empty());
    assert_eq!(provider.calls(), 1);
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
    assert_eq!(content, "final answer");
}

#[tokio::test]
async fn output_to_chat_appends_results_to_content() {
    let executor = make_executor();
    let interceptor = ToolInterceptor::new(executor, 5, true);
    let search_call = make_tool_call("call_1", "codex_search", r#"{"q":"rust news"}"#);
    let provider = MockProvider::new(vec![make_final_response("final answer")]);

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
    let interceptor = ToolInterceptor::new(executor, 5, true);
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
    let interceptor = ToolInterceptor::new(executor, 5, false);
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
}
