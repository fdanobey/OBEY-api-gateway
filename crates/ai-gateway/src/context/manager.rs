//! Context manager for automatic context window handling.
//!
//! Provides token estimation, model capabilities caching, and automatic
//! context truncation when requests exceed model limits.

use crate::config::ContextConfig;
use crate::models::openai::{Message, OpenAIRequest};
use crate::providers::Model;
use dashmap::DashMap;
use std::sync::Arc;
use std::time::{Duration, Instant};

use super::strategies::{apply_truncation_strategy, TruncationResult, TruncationStrategy};

/// Model capabilities discovered from provider
#[derive(Debug, Clone)]
#[allow(dead_code)]
pub struct ModelCapabilities {
    /// Model identifier
    pub model_id: String,
    /// Maximum context window in tokens
    pub context_window: u32,
    /// Maximum output tokens
    pub max_completion_tokens: Option<u32>,
    /// When this capability was discovered
    pub discovered_at: Instant,
}

#[allow(dead_code)]
impl ModelCapabilities {
    /// Create new model capabilities
    pub fn new(model_id: String, context_window: u32, max_completion_tokens: Option<u32>) -> Self {
        Self {
            model_id,
            context_window,
            max_completion_tokens,
            discovered_at: Instant::now(),
        }
    }

    /// Create from provider Model struct
    pub fn from_model(model: &Model) -> Option<Self> {
        model.context_window.map(|cw| {
            Self::new(model.id.clone(), cw, model.max_completion_tokens)
        })
    }

    /// Check if capabilities are stale
    pub fn is_stale(&self, ttl: Duration) -> bool {
        self.discovered_at.elapsed() > ttl
    }
}

/// Context manager for handling context window limits
pub struct ContextManager {
    /// Cache of model capabilities by model ID
    capabilities_cache: Arc<DashMap<String, ModelCapabilities>>,
    /// Configuration for context management
    config: ContextConfig,
    /// TTL for cached capabilities
    cache_ttl: Duration,
    /// Maximum retries for truncation
    max_truncation_retries: usize,
}

impl ContextManager {
    /// Create a new context manager with default settings
    pub fn new() -> Self {
        Self::with_config(ContextConfig::default())
    }

    /// Create a new context manager with configuration
    pub fn with_config(config: ContextConfig) -> Self {
        let cache_ttl = Duration::from_secs(config.capabilities_cache_ttl_seconds);
        let max_truncation_retries = config.max_truncation_retries;
        Self {
            capabilities_cache: Arc::new(DashMap::new()),
            config,
            cache_ttl,
            max_truncation_retries,
        }
    }

    /// Get cached capabilities for a model
    pub fn get_capabilities(&self, model_id: &str) -> Option<ModelCapabilities> {
        self.capabilities_cache.get(model_id).and_then(|caps| {
            if caps.is_stale(self.cache_ttl) {
                None // Stale, need to refresh
            } else {
                Some(caps.value().clone())
            }
        })
    }

    /// Store capabilities for a model
    #[allow(dead_code)]
    pub fn store_capabilities(&self, capabilities: ModelCapabilities) {
        self.capabilities_cache
            .insert(capabilities.model_id.clone(), capabilities);
    }

    /// Store capabilities from a provider Model
    #[allow(dead_code)]
    pub fn store_model(&self, model: &Model) {
        if let Some(caps) = ModelCapabilities::from_model(model) {
            self.store_capabilities(caps);
        }
    }

    /// Store multiple models at once
    #[allow(dead_code)]
    pub fn store_models(&self, models: &[Model]) {
        for model in models {
            self.store_model(model);
        }
    }

    /// Clear the capabilities cache
    pub fn clear_cache(&self) {
        self.capabilities_cache.clear();
    }

    /// Estimate token count for a slice of messages
    /// Uses a simple heuristic of ~4 characters per token
    pub fn estimate_tokens(messages: &[Message]) -> u32 {
        let mut total = 0u32;
        for msg in messages {
            // Account for role prefix (~1 token)
            total += 1;
            // Count content tokens (rough estimate: 4 chars per token)
            let content = msg.content_as_text();
            total += (content.len() as u32 + 3) / 4;
            // Account for tool_calls in assistant messages (function names + arguments)
            if let Some(tool_calls) = msg.extra.get("tool_calls") {
                let tc_str = tool_calls.to_string();
                total += (tc_str.len() as u32 + 3) / 4;
            }
            // Account for tool_call_id in tool result messages
            if let Some(tc_id) = msg.extra.get("tool_call_id") {
                let id_str = tc_id.to_string();
                total += (id_str.len() as u32 + 3) / 4;
            }
        }
        total
    }

    /// Estimate token count for a full request
    pub fn estimate_request_tokens(request: &OpenAIRequest) -> u32 {
        let mut total = Self::estimate_tokens(&request.messages);
        // Account for tools definitions in the request (can be thousands of tokens)
        if let Some(tools) = request.extra.get("tools") {
            let tools_str = tools.to_string();
            total += (tools_str.len() as u32 + 3) / 4;
        }
        total
    }

    /// Check if a request fits within context limits
    pub fn fits_within_limits(
        &self,
        request: &OpenAIRequest,
        context_window: u32,
    ) -> bool {
        let estimated = Self::estimate_request_tokens(request);
        // Leave some buffer for output tokens
        let effective_limit = (context_window as f64 * 0.75) as u32;
        estimated <= effective_limit
    }

    /// Truncate a request to fit within context limits
    /// Returns the truncation result
    pub fn truncate_request(
        &self,
        request: &mut OpenAIRequest,
        context_window: u32,
    ) -> TruncationResult {
        let strategy = match self.config.truncation_strategy.as_str() {
            "sliding_window" => TruncationStrategy::SlidingWindow {
                window_size: self.config.sliding_window_size,
            },
            "remove_oldest" | _ => TruncationStrategy::RemoveOldest,
        };

        // Leave some buffer for output tokens
        let effective_limit = (context_window as f64 * 0.75) as u32;

        apply_truncation_strategy(
            &mut request.messages,
            effective_limit,
            strategy,
            Self::estimate_tokens,
        )
    }

    /// Handle a context length error by truncating and retrying
    /// Returns true if truncation was performed, false if no more truncation possible
    pub fn handle_context_error(
        &self,
        request: &mut OpenAIRequest,
        attempt: usize,
        error_body: Option<&str>,
    ) -> Result<TruncationResult, ContextError> {
        if attempt >= self.max_truncation_retries {
            return Err(ContextError::MaxRetriesExceeded);
        }

        // Priority order for determining context window:
        // 1. Cached model capabilities (from /models endpoint or prior calls)
        // 2. Limit parsed from the provider's error response body
        // 3. Configured default_context_window (defaults to 32768)
        let context_window = self
            .get_capabilities(&request.model)
            .map(|caps| caps.context_window)
            .or_else(|| error_body.and_then(Self::parse_context_limit_from_error))
            .unwrap_or(self.config.default_context_window);

        let result = self.truncate_request(request, context_window);

        if !result.truncated {
            return Err(ContextError::CannotTruncateFurther);
        }

        Ok(result)
    }

    /// Try to extract the maximum token limit from a provider error body.
    ///
    /// Many providers include the limit in their error messages, e.g.:
    /// - "maximum context length is 32768 tokens"
    /// - "This model's maximum context length is 131072 tokens"
    /// - "token limit: 32768"
    /// - {"detail":"This model's maximum context length is 32768 tokens. However, your messages resulted in 186686 tokens."}
    fn parse_context_limit_from_error(body: &str) -> Option<u32> {
        let body_lower = body.to_lowercase();

        // Pattern: "maximum context length is N tokens"
        if let Some(pos) = body_lower.find("maximum context length is ") {
            let after = &body[pos + "maximum context length is ".len()..];
            if let Some(num) = after.split(|c: char| !c.is_ascii_digit()).next() {
                if let Ok(limit) = num.parse::<u32>() {
                    if limit > 0 {
                        return Some(limit);
                    }
                }
            }
        }

        // Pattern: "token limit" or "token_limit" followed by a number
        for pattern in &["token limit", "token_limit", "max_tokens"] {
            if let Some(pos) = body_lower.find(pattern) {
                let after = &body[pos + pattern.len()..];
                // Skip non-digit chars (colon, space, equals, etc.)
                let num_str: String = after
                    .chars()
                    .skip_while(|c| !c.is_ascii_digit())
                    .take_while(|c| c.is_ascii_digit())
                    .collect();
                if let Ok(limit) = num_str.parse::<u32>() {
                    if limit > 0 {
                        return Some(limit);
                    }
                }
            }
        }

        None
    }

    /// Check if an error message indicates a context length problem
    pub fn is_context_length_error(&self, status: u16, body: &str) -> bool {
        if status != 400 && status != 413 {
            return false;
        }

        let body_lower = body.to_lowercase();
        body_lower.contains("context_length")
            || body_lower.contains("context length")
            || body_lower.contains("maximum context length")
            || body_lower.contains("token limit")
            || body_lower.contains("context window")
            || body_lower.contains("too many tokens")
            || body_lower.contains("input is too long")
    }

    /// Get the current configuration
    #[allow(dead_code)]
    pub fn config(&self) -> &ContextConfig {
        &self.config
    }
}

impl Default for ContextManager {
    fn default() -> Self {
        Self::new()
    }
}

/// Errors that can occur during context management
#[derive(Debug, Clone)]
#[allow(dead_code)]
pub enum ContextError {
    /// Maximum truncation retries exceeded
    MaxRetriesExceeded,
    /// Cannot truncate further (minimum context reached)
    CannotTruncateFurther,
    /// Model capabilities not known
    UnknownModelCapabilities,
    /// Request is empty (no messages to truncate)
    EmptyRequest,
}

impl std::fmt::Display for ContextError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::MaxRetriesExceeded => {
                write!(f, "Maximum context truncation retries exceeded")
            }
            Self::CannotTruncateFurther => {
                write!(f, "Cannot truncate context further")
            }
            Self::UnknownModelCapabilities => {
                write!(f, "Model context capabilities not available")
            }
            Self::EmptyRequest => {
                write!(f, "Request contains no messages")
            }
        }
    }
}

impl std::error::Error for ContextError {}

#[cfg(test)]
mod tests {
    use super::*;

    fn create_test_message(role: &str, content: &str) -> Message {
        Message {
            role: role.to_string(),
            content: serde_json::Value::String(content.to_string()),
            extra: Default::default(),
        }
    }

    #[test]
    fn test_token_estimation() {
        let messages = vec![
            create_test_message("system", "You are a helpful assistant"), // ~7 tokens
            create_test_message("user", "Hello, how are you?"), // ~6 tokens
        ];

        let estimate = ContextManager::estimate_tokens(&messages);
        // Should be approximately 13 tokens (7 + 6)
        assert!(estimate > 0 && estimate < 50);
    }

    #[test]
    fn test_capability_caching() {
        let manager = ContextManager::new();

        let caps = ModelCapabilities::new("gpt-4".to_string(), 128000, Some(4096));
        manager.store_capabilities(caps);

        let retrieved = manager.get_capabilities("gpt-4");
        assert!(retrieved.is_some());
        assert_eq!(retrieved.unwrap().context_window, 128000);
    }

    #[test]
    fn test_context_length_error_detection() {
        let manager = ContextManager::new();

        assert!(manager.is_context_length_error(
            400,
            "This model's maximum context length is 4096 tokens"
        ));
        assert!(manager.is_context_length_error(
            400,
            "context_length_exceeded"
        ));
        assert!(manager.is_context_length_error(
            413,
            "Input is too long"
        ));
        assert!(!manager.is_context_length_error(
            400,
            "Invalid API key"
        ));
        assert!(!manager.is_context_length_error(
            500,
            "Internal server error"
        ));
    }

    #[test]
    fn test_truncate_request() {
        let manager = ContextManager::new();
        manager.store_capabilities(ModelCapabilities::new(
            "test-model".to_string(),
            100, // Small context for testing
            None,
        ));

        let mut request = OpenAIRequest {
            model: "test-model".to_string(),
            messages: vec![
                create_test_message("system", "You are helpful"),
                create_test_message("user", &"x".repeat(200)), // Long message
                create_test_message("assistant", "Ok"),
                create_test_message("user", &"y".repeat(200)), // Long message
            ],
            stream: false,
            temperature: None,
            max_tokens: None,
            extra: Default::default(),
        };

        let result = manager.truncate_request(&mut request, 100);
        assert!(result.truncated);
        assert!(result.messages_removed > 0);
    }
}
