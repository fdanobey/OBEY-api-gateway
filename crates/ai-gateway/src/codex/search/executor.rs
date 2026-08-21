//! Search executor — dispatches search commands to the Codex upstream.
//!
//! Tasks 4.1–4.4: struct, constructor, input validation, HTTP dispatch with
//! OAuth authentication, retry logic, timeout handling, and metrics.

use std::sync::Arc;
use std::time::{Duration, Instant};

use async_trait::async_trait;
use serde_json::Value;
use uuid::Uuid;

use crate::codex::jwt::extract_chatgpt_account_id;
use crate::codex::search::metrics::SearchMetrics;
use crate::codex::search::models::{
    CodexSearchArgs, CodexSearchRequest, CodexSearchRequestCommands, CodexWebArgs,
    CodexWebCommands, ResponseLength, ToolResult,
};
use crate::oauth::{OAuthManager, UsageTracker};

const SEARCH_MODEL: &str = "gpt-4o";
const MAX_RESULT_CHARS: usize = 4000;
const TRUNCATION_SUFFIX: &str = "[truncated]";
const MAX_LOG_BODY_CHARS: usize = 500;
const SERVER_ERROR_BACKOFF: [Duration; 2] = [Duration::from_secs(1), Duration::from_secs(3)];

#[async_trait]
pub trait Sleeper: Send + Sync {
    async fn sleep(&self, duration: Duration);
}

pub struct TokioSleeper;

#[async_trait]
impl Sleeper for TokioSleeper {
    async fn sleep(&self, duration: Duration) {
        tokio::time::sleep(duration).await;
    }
}

/// Executes `codex_search` and `codex_web` tool calls against the Codex
/// upstream search endpoint.
pub struct SearchExecutor {
    http: reqwest::Client,
    oauth: Arc<OAuthManager>,
    usage_tracker: Arc<UsageTracker>,
    metrics: Arc<SearchMetrics>,
    base_url: String,
    timeout: Duration,
    sleeper: Arc<dyn Sleeper>,
}

impl SearchExecutor {
    pub fn new(
        http: reqwest::Client,
        oauth: Arc<OAuthManager>,
        usage_tracker: Arc<UsageTracker>,
        metrics: Arc<SearchMetrics>,
        base_url: String,
        timeout: Duration,
    ) -> Self {
        Self {
            http,
            oauth,
            usage_tracker,
            metrics,
            base_url,
            timeout,
            sleeper: Arc::new(TokioSleeper),
        }
    }

    #[allow(dead_code)]
    pub fn with_sleeper(
        http: reqwest::Client,
        oauth: Arc<OAuthManager>,
        usage_tracker: Arc<UsageTracker>,
        metrics: Arc<SearchMetrics>,
        base_url: String,
        timeout: Duration,
        sleeper: Arc<dyn Sleeper>,
    ) -> Self {
        Self {
            http,
            oauth,
            usage_tracker,
            metrics,
            base_url,
            timeout,
            sleeper,
        }
    }

    /// Execute a `codex_search` tool call.
    pub async fn execute_search(&self, args: CodexSearchArgs) -> ToolResult {
        if let Err(e) = Self::validate_search_args(&args) {
            self.metrics.record_execution("codex_search");
            return e;
        }

    let session_id = Uuid::new_v4().to_string();
    let commands = CodexSearchRequestCommands {
        search_query: Some(vec![crate::codex::search::models::SearchQueryCommand {
            q: args.q,
            domains: args.domains,
            recency: args.recency,
        }]),
        open: None,
        find: None,
        click: None,
        response_length: Some(
            args.response_length.unwrap_or_else(ResponseLength::short),
        ),
        extra: serde_json::Map::new(),
    };

    let request = CodexSearchRequest {
        id: session_id,
        model: SEARCH_MODEL.to_string(),
        commands,
        extra: serde_json::Map::new(),
    };

    let result = self.dispatch_upstream("codex_search", &request).await;
        self.metrics.record_execution("codex_search");
        result
    }

    /// Execute a `codex_web` tool call.
    pub async fn execute_web(&self, args: CodexWebArgs) -> ToolResult {
        if let Err(e) = Self::validate_web_args(&args) {
            self.metrics.record_execution("codex_web");
            return e;
        }

        let session_id = args
            .session_id
            .clone()
            .unwrap_or_else(|| Uuid::new_v4().to_string());

        let commands = if let Some(c) = args.commands {
            CodexSearchRequestCommands {
                search_query: c.search_query,
                open: c.open,
                find: c.find,
                click: c.click,
                response_length: Some(args.response_length.unwrap_or_else(ResponseLength::medium)),
                extra: serde_json::Map::new(),
            }
        } else {
            CodexSearchRequestCommands {
                search_query: None,
                open: None,
                find: None,
                click: None,
                response_length: Some(ResponseLength::medium()),
                extra: serde_json::Map::new(),
            }
        };

    let request = CodexSearchRequest {
        id: session_id.clone(),
        model: SEARCH_MODEL.to_string(),
        commands,
        extra: serde_json::Map::new(),
    };

    let mut result = self.dispatch_upstream("codex_web", &request).await;
        self.metrics.record_execution("codex_web");
        if result.session_id.is_none() {
            result.session_id = Some(session_id);
        }
        result
    }

    async fn dispatch_upstream(&self, tool: &str, request: &CodexSearchRequest) -> ToolResult {
        let body = match serde_json::to_value(request) {
            Ok(v) => v,
            Err(_) => return Self::error_tool_result("Search request serialization failed"),
        };

        let access_token = match self.oauth.get_access_token().await {
            Some(t) => t,
            None => return Self::error_tool_result("Search authentication unavailable."),
        };

        let account_id = match extract_chatgpt_account_id(&access_token) {
            Ok(id) => id,
            Err(_) => return Self::error_tool_result("Search authentication configuration error."),
        };

        match self
            .send_with_retry(tool, &access_token, &account_id, &body)
            .await
        {
            Ok(content) => ToolResult {
                content: truncate_content(&content),
                is_error: false,
                session_id: None,
            },
            Err(tool_err) => tool_err,
        }
    }

    async fn send_with_retry(
        &self,
        tool: &str,
        access_token: &str,
        account_id: &str,
        body: &Value,
    ) -> Result<String, ToolResult> {
        let mut current_token = access_token.to_string();
        let mut current_account_id = account_id.to_string();
        let mut auth_retried = false;
        let mut server_error_attempts = 0u32;

        for server_attempt in 0..3u32 {
            if server_attempt > 0 {
                let backoff = SERVER_ERROR_BACKOFF[(server_attempt - 1) as usize];
                self.sleeper.sleep(backoff).await;
            }

            let result = loop {
                let inner = self
                    .send_once(tool, &current_token, &current_account_id, body)
                    .await;

                match inner {
                    SendOutcome::AuthError(status, latency_ms, headers)
                        if !auth_retried && status == 401 =>
                    {
                        self.metrics.record_latency(tool, latency_ms);
                        self.usage_tracker.update_from_headers(&headers).await;
                        auth_retried = true;
                        match self.oauth.force_refresh().await {
                            Ok(new_token) => match extract_chatgpt_account_id(&new_token) {
                                Ok(new_account_id) => {
                                    current_token = new_token;
                                    current_account_id = new_account_id;
                                    continue;
                                }
                                Err(_) => {
                                    break SendOutcome::AuthError(status, latency_ms, headers);
                                }
                            },
                            Err(_) => {
                                break SendOutcome::AuthError(status, latency_ms, headers);
                            }
                        }
                    }
                    other => break other,
                }
            };

            match result {
                SendOutcome::Success(content, latency_ms, headers) => {
                    self.metrics.record_latency(tool, latency_ms);
                    self.usage_tracker.update_from_headers(&headers).await;
                    return Ok(content);
                }
                SendOutcome::AuthError(_status, latency_ms, headers) => {
                    self.metrics.record_latency(tool, latency_ms);
                    self.usage_tracker.update_from_headers(&headers).await;
                    return Err(Self::error_tool_result("Search authentication failed."));
                }
                SendOutcome::RateLimited(retry_after, latency_ms, headers) => {
                    self.metrics.record_latency(tool, latency_ms);
                    self.usage_tracker.update_from_headers(&headers).await;
                    let msg = if let Some(ra) = retry_after {
                        format!("Search request was rate limited. Retry-After: {ra}.")
                    } else {
                        "Search request was rate limited.".to_string()
                    };
                    return Err(Self::error_tool_result(&msg));
                }
                SendOutcome::ServerError(status, latency_ms, headers) => {
                    self.metrics.record_latency(tool, latency_ms);
                    if let Some(h) = headers {
                        self.usage_tracker.update_from_headers(&h).await;
                    }
                    server_error_attempts += 1;
                    if server_attempt >= 2 {
                        return Err(Self::error_tool_result(&format!(
"Search service is temporarily unavailable after {server_error_attempts} attempts (HTTP {status})."
)));
                    }
                    tracing::warn!(
                        tool,
                        status,
                        attempt = server_attempt + 1,
                        "Upstream server error, will retry"
                    );
                }
                SendOutcome::NonRetryableError(msg) => {
                    return Err(Self::error_tool_result(&msg));
                }
                SendOutcome::Timeout => {
                    let secs = self.timeout.as_secs();
                    return Err(Self::error_tool_result(&format!(
                        "Search request timed out after {secs} seconds."
                    )));
                }
                SendOutcome::ConnectionError => {
                    return Err(Self::error_tool_result(
                        "Search service is temporarily unavailable.",
                    ));
                }
                SendOutcome::InvalidRefId => {
                    return Err(Self::error_tool_result("Search reference was not found."));
                }
            }
        }

        Err(Self::error_tool_result(
            "Search service is temporarily unavailable after 3 attempts.",
        ))
    }

    async fn send_once(
        &self,
        _tool: &str,
        access_token: &str,
        account_id: &str,
        body: &Value,
    ) -> SendOutcome {
        let start = Instant::now();

    let resp = self
        .http
        .post(&self.base_url)
        .header("authorization", format!("Bearer {access_token}"))
        .header("chatgpt-account-id", account_id)
        .header("content-type", "application/json")
        .header("user-agent", "codex-cli/0.147.0-alpha.6.5")
        .timeout(self.timeout)
        .json(body)
        .send()
        .await;

        let resp = match resp {
            Ok(r) => r,
            Err(e) => {
                if e.is_timeout() {
                    return SendOutcome::Timeout;
                }
                tracing::warn!(error = %e, "Search upstream connection error");
                return SendOutcome::ConnectionError;
            }
        };

        let status = resp.status().as_u16();
        let headers = resp.headers().clone();
        let latency_ms = start.elapsed().as_millis() as u64;

        if status == 401 || status == 403 {
            return SendOutcome::AuthError(status, latency_ms, headers);
        }

        if status == 429 {
            let retry_after = headers
                .get("retry-after")
                .and_then(|v| v.to_str().ok())
                .and_then(|s| s.parse::<u64>().ok());
            return SendOutcome::RateLimited(retry_after, latency_ms, headers);
        }

        if status == 502 || status == 503 || status == 504 {
            let body_text = resp.text().await.unwrap_or_default();
            tracing::warn!(
            status,
            body = %body_text.chars().take(MAX_LOG_BODY_CHARS).collect::<String>(),
            "Search upstream server error"
            );
            return SendOutcome::ServerError(status, latency_ms, Some(headers));
        }

        if !resp.status().is_success() {
            self.metrics.record_latency(_tool, latency_ms);
            self.usage_tracker.update_from_headers(&headers).await;
            let body_text = resp.text().await.unwrap_or_default();
            tracing::warn!(
            status,
            body = %body_text.chars().take(MAX_LOG_BODY_CHARS).collect::<String>(),
            "Search upstream non-2xx"
            );
            if body_text.contains("invalid") && body_text.contains("ref_id") {
                return SendOutcome::InvalidRefId;
            }
            return SendOutcome::NonRetryableError(format!(
                "Search request failed with HTTP {status}."
            ));
        }

        let content = resp.text().await.unwrap_or_default();
        SendOutcome::Success(content, latency_ms, headers)
    }

    /// Validate `codex_search` tool arguments.
    pub fn validate_search_args(args: &CodexSearchArgs) -> Result<(), ToolResult> {
        if let Err(msg) = Self::validate_query(&args.q) {
            return Err(Self::error_tool_result(&msg));
        }
        if let Err(msg) = Self::validate_domains(&args.domains) {
            return Err(Self::error_tool_result(&msg));
        }
        if let Err(msg) = Self::validate_recency(&args.recency) {
            return Err(Self::error_tool_result(&msg));
        }
        Ok(())
    }

    /// Validate `codex_web` tool arguments.
    pub fn validate_web_args(args: &CodexWebArgs) -> Result<(), ToolResult> {
        if let Some(session_id) = &args.session_id {
            if session_id.chars().count() > 128 {
                return Err(Self::error_tool_result("Session ID exceeds 128 characters"));
            }
        }
        if let Some(commands) = &args.commands {
            Self::validate_web_commands(commands)?;
        }
        Ok(())
    }

    fn validate_web_commands(commands: &CodexWebCommands) -> Result<(), ToolResult> {
        if let Some(sq) = &commands.search_query {
            if sq.len() > 10 {
                return Err(Self::error_tool_result("Command array exceeds 10 entries"));
            }
            for cmd in sq {
                if let Err(msg) = Self::validate_query(&cmd.q) {
                    return Err(Self::error_tool_result(&msg));
                }
                if let Err(msg) = Self::validate_domains(&cmd.domains) {
                    return Err(Self::error_tool_result(&msg));
                }
                if let Err(msg) = Self::validate_recency(&cmd.recency) {
                    return Err(Self::error_tool_result(&msg));
                }
            }
        }
        if let Some(v) = &commands.open {
            if v.len() > 10 {
                return Err(Self::error_tool_result("Command array exceeds 10 entries"));
            }
        }
        if let Some(v) = &commands.find {
            if v.len() > 10 {
                return Err(Self::error_tool_result("Command array exceeds 10 entries"));
            }
        }
        if let Some(v) = &commands.click {
            if v.len() > 10 {
                return Err(Self::error_tool_result("Command array exceeds 10 entries"));
            }
        }
        Ok(())
    }

    fn validate_query(q: &str) -> Result<(), String> {
        let len = q.chars().count();
        if len == 0 {
            return Err("Search query is required".to_string());
        }
        if len > 2000 {
            return Err("Search query exceeds 2000 characters".to_string());
        }
        Ok(())
    }

    fn validate_domains(domains: &Option<Vec<String>>) -> Result<(), String> {
        match domains {
            None => Ok(()),
            Some(d) => {
                if d.len() > 10 {
                    return Err("Domains list exceeds 10 entries".to_string());
                }
                Ok(())
            }
        }
    }

    fn validate_recency(recency: &Option<u32>) -> Result<(), String> {
        match recency {
            None => Ok(()),
            Some(r) => {
                if *r < 1 || *r > 365 {
                    return Err("Recency must be between 1 and 365 days".to_string());
                }
                Ok(())
            }
        }
    }

    fn error_tool_result(msg: &str) -> ToolResult {
        ToolResult {
            content: msg.to_string(),
            is_error: true,
            session_id: None,
        }
    }
}

enum SendOutcome {
    Success(String, u64, reqwest::header::HeaderMap),
    AuthError(u16, u64, reqwest::header::HeaderMap),
    RateLimited(Option<u64>, u64, reqwest::header::HeaderMap),
    ServerError(u16, u64, Option<reqwest::header::HeaderMap>),
    NonRetryableError(String),
    Timeout,
    ConnectionError,
    InvalidRefId,
}

fn truncate_content(content: &str) -> String {
    let char_count = content.chars().count();
    if char_count <= MAX_RESULT_CHARS {
        return content.to_string();
    }
    let truncated: String = content.chars().take(MAX_RESULT_CHARS).collect();
    format!("{truncated}{TRUNCATION_SUFFIX}")
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::codex::search::models::{
        ClickCommand, FindCommand, OpenCommand, SearchQueryCommand,
    };

    fn domains(n: usize) -> Option<Vec<String>> {
        Some((0..n).map(|i| format!("example{i}.com")).collect())
    }

    #[test]
    fn domains_length_10_accepted() {
    let args = CodexSearchArgs {
        q: "hello".to_string(),
        domains: domains(10),
        recency: None,
        response_length: None,
    };
    assert!(SearchExecutor::validate_search_args(&args).is_ok());
}

    #[test]
    fn domains_length_11_rejected() {
    let args = CodexSearchArgs {
        q: "hello".to_string(),
        domains: domains(11),
        recency: None,
        response_length: None,
    };
    let err = SearchExecutor::validate_search_args(&args).unwrap_err();
        assert!(err.is_error);
        assert_eq!(err.content, "Domains list exceeds 10 entries");
    }

    #[test]
    fn query_length_exactly_2000_accepted() {
        let q = "a".repeat(2000);
        let args = CodexSearchArgs {
            q,
            domains: None,
            recency: None,
        response_length: None,
        };
        assert!(SearchExecutor::validate_search_args(&args).is_ok());
    }

    #[test]
    fn query_length_2001_rejected() {
        let q = "a".repeat(2001);
        let args = CodexSearchArgs {
            q,
            domains: None,
            recency: None,
        response_length: None,
        };
        let err = SearchExecutor::validate_search_args(&args).unwrap_err();
        assert!(err.is_error);
        assert_eq!(err.content, "Search query exceeds 2000 characters");
    }

    #[test]
    fn recency_1_accepted() {
        let args = CodexSearchArgs {
            q: "hello".to_string(),
            domains: None,
            recency: Some(1),
        response_length: None,
        };
        assert!(SearchExecutor::validate_search_args(&args).is_ok());
    }

    #[test]
    fn recency_365_accepted() {
        let args = CodexSearchArgs {
            q: "hello".to_string(),
            domains: None,
            recency: Some(365),
        response_length: None,
        };
        assert!(SearchExecutor::validate_search_args(&args).is_ok());
    }

    #[test]
    fn recency_0_rejected() {
        let args = CodexSearchArgs {
            q: "hello".to_string(),
            domains: None,
            recency: Some(0),
        response_length: None,
        };
        let err = SearchExecutor::validate_search_args(&args).unwrap_err();
        assert!(err.is_error);
        assert_eq!(err.content, "Recency must be between 1 and 365 days");
    }

    #[test]
    fn recency_366_rejected() {
        let args = CodexSearchArgs {
            q: "hello".to_string(),
            domains: None,
            recency: Some(366),
        response_length: None,
        };
        let err = SearchExecutor::validate_search_args(&args).unwrap_err();
        assert!(err.is_error);
        assert_eq!(err.content, "Recency must be between 1 and 365 days");
    }

    #[test]
    fn empty_query_rejected() {
        let args = CodexSearchArgs {
            q: String::new(),
            domains: None,
            recency: None,
        response_length: None,
        };
        let err = SearchExecutor::validate_search_args(&args).unwrap_err();
        assert!(err.is_error);
        assert_eq!(err.content, "Search query is required");
    }

    #[test]
    fn web_args_valid() {
        let args = CodexWebArgs {
            session_id: Some("sess123".to_string()),
            commands: Some(CodexWebCommands {
                search_query: Some(vec![SearchQueryCommand {
                    q: "rust async".to_string(),
                    domains: Some(vec!["rust-lang.org".to_string()]),
                    recency: Some(7),
                }]),
        open: Some(vec![OpenCommand {
            ref_id: "1".to_string(),
            lineno: None,
        }]),
        find: Some(vec![FindCommand {
            ref_id: "1".to_string(),
            pattern: "tokio".to_string(),
        }]),
        click: Some(vec![ClickCommand {
            ref_id: "1".to_string(),
            id: None,
        }]),
            }),
            response_length: None,
        };
        assert!(SearchExecutor::validate_web_args(&args).is_ok());
    }

    #[test]
    fn web_command_array_exceeds_10_rejected() {
        let sq: Vec<SearchQueryCommand> = std::iter::repeat_with(|| SearchQueryCommand {
            q: "x".to_string(),
            domains: None,
            recency: None,
        })
        .take(11)
        .collect();
        let args = CodexWebArgs {
            session_id: None,
            commands: Some(CodexWebCommands {
                search_query: Some(sq),
                open: None,
                find: None,
                click: None,
            }),
            response_length: None,
        };
        let err = SearchExecutor::validate_web_args(&args).unwrap_err();
        assert!(err.is_error);
        assert_eq!(err.content, "Command array exceeds 10 entries");
    }

    #[test]
    fn web_session_id_too_long_rejected() {
        let sid = "a".repeat(129);
        let args = CodexWebArgs {
            session_id: Some(sid),
            commands: None,
            response_length: None,
        };
        let err = SearchExecutor::validate_web_args(&args).unwrap_err();
        assert!(err.is_error);
        assert_eq!(err.content, "Session ID exceeds 128 characters");
    }

    #[test]
    fn web_session_id_128_accepted() {
        let sid = "a".repeat(128);
        let args = CodexWebArgs {
            session_id: Some(sid),
            commands: None,
            response_length: None,
        };
        assert!(SearchExecutor::validate_web_args(&args).is_ok());
    }

    #[test]
    fn truncation_preserves_utf8_and_adds_suffix() {
        let content = "世".repeat(5000);
        let truncated = truncate_content(&content);
        assert!(truncated.ends_with(TRUNCATION_SUFFIX));
        let without_suffix = truncated.strip_suffix(TRUNCATION_SUFFIX).unwrap();
        assert_eq!(without_suffix.chars().count(), MAX_RESULT_CHARS);
    }

    #[test]
    fn truncation_noop_when_under_limit() {
        let content = "short result";
        let truncated = truncate_content(content);
        assert_eq!(truncated, content);
    }

    mod http_retry_tests {
        use super::*;

        #[test]
        fn send_outcome_server_error_carries_latency_and_headers() {
            let outcome = SendOutcome::ServerError(502, 123, None);
            match outcome {
                SendOutcome::ServerError(status, latency_ms, headers) => {
                    assert_eq!(status, 502);
                    assert_eq!(latency_ms, 123);
                    assert!(headers.is_none());
                }
                _ => panic!("expected ServerError"),
            }
        }

        #[test]
        fn send_outcome_auth_error_carries_latency_and_headers() {
            let outcome = SendOutcome::AuthError(401, 456, reqwest::header::HeaderMap::new());
            match outcome {
                SendOutcome::AuthError(status, latency_ms, _headers) => {
                    assert_eq!(status, 401);
                    assert_eq!(latency_ms, 456);
                }
                _ => panic!("expected AuthError"),
            }
        }

        #[test]
        fn send_outcome_rate_limited_carries_latency_and_headers() {
            let outcome =
                SendOutcome::RateLimited(Some(30), 789, reqwest::header::HeaderMap::new());
            match outcome {
                SendOutcome::RateLimited(retry_after, latency_ms, _headers) => {
                    assert_eq!(retry_after, Some(30));
                    assert_eq!(latency_ms, 789);
                }
                _ => panic!("expected RateLimited"),
            }
        }

        #[test]
        fn send_outcome_non_retryable_error_message_used() {
            let outcome = SendOutcome::NonRetryableError("Custom error".to_string());
            match outcome {
                SendOutcome::NonRetryableError(msg) => {
                    assert_eq!(msg, "Custom error");
                }
                _ => panic!("expected NonRetryableError"),
            }
        }
    }

    mod property_tests {
        use super::*;
        use proptest::prelude::*;

        proptest! {
            #![proptest_config(ProptestConfig::with_cases(100))]

            // Feature: codex-search, Property 5: Invalid queries rejected
            #[test]
            fn prop_invalid_queries_rejected(
                q_len in 0u32..=3000u32,
            ) {
                let q = "a".repeat(q_len as usize);
                let args = CodexSearchArgs {
                    q,
                    domains: None,
                    recency: None,
                response_length: None,
                };
                let result = SearchExecutor::validate_search_args(&args);
                let char_len = q_len as usize;
                if char_len == 0 || char_len > 2000 {
                    prop_assert!(result.is_err(), "should reject q_len={}", q_len);
                    let err = result.unwrap_err();
                    prop_assert!(err.is_error);
                } else {
                    prop_assert!(result.is_ok(), "should accept q_len={}", q_len);
                }
            }

            // Feature: codex-search, Property 6: Invalid recency rejected
            #[test]
            fn prop_invalid_recency_rejected(
                recency in 0u32..=500u32,
            ) {
                let args = CodexSearchArgs {
                    q: "test".to_string(),
                    domains: None,
                    recency: Some(recency),
                response_length: None,
                };
                let result = SearchExecutor::validate_search_args(&args);
                if (1..=365).contains(&recency) {
                    prop_assert!(result.is_ok(), "should accept recency={}", recency);
                } else {
                    prop_assert!(result.is_err(), "should reject recency={}", recency);
                    let err = result.unwrap_err();
                    prop_assert!(err.is_error);
                }
            }

            // Feature: codex-search, Property 7: Truncation prefix preservation
            #[test]
            fn prop_truncation_preserves_prefix(
                content in "[a-zA-Z0-9 ]{1,6000}",
            ) {
                let truncated = truncate_content(&content);
                let char_count = content.chars().count();
                if char_count <= MAX_RESULT_CHARS {
                    prop_assert_eq!(truncated, content);
                } else {
                    prop_assert!(truncated.ends_with(TRUNCATION_SUFFIX));
                    let prefix = truncated.strip_suffix(TRUNCATION_SUFFIX).unwrap();
                    prop_assert_eq!(prefix.chars().count(), MAX_RESULT_CHARS);
                    let original_prefix: String = content.chars().take(MAX_RESULT_CHARS).collect();
                    prop_assert_eq!(prefix, original_prefix);
                }
            }
        }
    }
}