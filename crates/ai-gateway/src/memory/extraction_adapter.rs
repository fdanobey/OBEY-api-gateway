//! Production extraction adapter that calls a configured provider via the
//! standard OpenAI-compatible chat-completions endpoint.
//!
//! This adapter bridges the [`MemoryExtractionProvider`] trait to the
//! gateway's provider configuration, allowing automatic memory extraction
//! to use any OpenAI-compatible provider declared in the config.

use std::sync::Arc;

use async_trait::async_trait;
use serde::Deserialize;
use tokio::sync::RwLock;

use crate::config::Config;
use crate::memory::extractor::{
    ExtractionMessage, ExtractionRole, MemoryExtractionProvider, MemoryExtractionProviderError,
    MemoryExtractionProviderRequest, StructuredMemoryCandidate, MEMORY_EXTRACTION_INTERNAL_TAG,
};
use crate::memory::MemoryType;

/// System prompt instructing the model to extract structured memories.
///
/// The prompt enforces a strict quality bar: only durable, reusable
/// information that would genuinely save effort in a future conversation
/// should be extracted. Transient state, task progress, and one-off details
/// are explicitly excluded to prevent noise accumulation.
const EXTRACTION_SYSTEM_PROMPT: &str = r#"You are a memory extraction assistant. Extract ONLY durable, high-value information that will be useful in future conversations with this user or on this project.

QUALITY BAR — a memory must satisfy ALL of these:
1. Durable: will remain true or relevant well beyond this conversation
2. Reusable: would save time, prevent re-explanation, or avoid repeating a mistake in a future session
3. Self-contained: fully understandable without the surrounding conversation
4. Non-obvious: not trivially re-derivable from the project's code, docs, or standard tooling

DO NOT EXTRACT:
- Transient state: current task, current bug, what you are doing right now, work in progress
- One-off questions, clarifications, or tentative ideas that were not adopted
- Code snippets, implementation details, or syntax (unless they encode a durable convention)
- Greetings, acknowledgments, status updates, or meta-conversation
- File paths, build commands, or config values that are evident from the project itself
- Anything the user is merely considering but has not committed to
- Trivial facts (e.g., "the user writes code", "the project uses Git")

Prefer extracting FEWER, higher-quality memories. When uncertain whether something is worth persisting, do NOT extract it.

Classify each memory as one of:
- "preference": Stable user preferences, habits, or conventions (e.g., "always use snake_case for Rust variables")
- "fact": Objective, durable facts about the user, project, or environment (e.g., "the project targets Rust 1.78 and uses Tokio")
- "context": Stable project context that is not obvious from the repo (e.g., "deployment runs on AWS via Terraform in the infra/ directory")
- "decision": Concrete decisions that were committed to (e.g., "chose SQLite over Postgres for the memory store to keep deployment single-binary")

Respond as a JSON array of objects with "content" and "memory_type" fields:
[{"content": "...", "memory_type": "preference"}, ...]

If nothing durable and reusable can be extracted, respond with an empty array: []"#;

/// Production adapter that calls a configured provider's chat-completions
/// endpoint to extract structured memories from conversation messages.
pub struct GatewayExtractionAdapter {
    config: Arc<RwLock<Config>>,
    http_client: reqwest::Client,
}

impl GatewayExtractionAdapter {
    pub fn new(config: Arc<RwLock<Config>>) -> Self {
        let http_client = reqwest::Client::builder()
            .timeout(std::time::Duration::from_secs(60))
            .build()
            .unwrap_or_else(|_| reqwest::Client::new());
        Self {
            config,
            http_client,
        }
    }

    /// Resolve the provider configuration for the named extraction provider.
    async fn resolve_provider(
        &self,
        provider_name: &str,
    ) -> Result<crate::config::Provider, MemoryExtractionProviderError> {
        let config = self.config.read().await;
        config
            .providers
            .iter()
            .find(|p| p.name == provider_name)
            .cloned()
            .ok_or_else(|| MemoryExtractionProviderError {
                message: format!(
                    "auto_extract_provider {:?} not found in configured providers",
                    provider_name
                ),
            })
    }

    /// Build the outgoing request body as serde_json::Value.
    fn build_request_body(model: &str, messages: &[ExtractionMessage]) -> serde_json::Value {
        let mut chat_messages = Vec::with_capacity(messages.len() + 1);

        // System prompt
        chat_messages.push(serde_json::json!({
            "role": "system",
            "content": EXTRACTION_SYSTEM_PROMPT,
        }));

        // Conversation messages
        for msg in messages {
            let role = match msg.role {
                ExtractionRole::User => "user",
                ExtractionRole::Assistant => "assistant",
                ExtractionRole::Other => "system",
            };
            chat_messages.push(serde_json::json!({
            "role": role,
            "content": msg.content,
            }));
        }

        serde_json::json!({
            "model": model,
            "messages": chat_messages,
            "temperature": 0.1,
            "stream": false,
        })
    }

    /// Parse the LLM response content into structured memory candidates.
    fn parse_response(
        content: &str,
    ) -> Result<Vec<StructuredMemoryCandidate>, MemoryExtractionProviderError> {
        let trimmed = content.trim();

        // Strip markdown code fences if present
        let json_str = if trimmed.starts_with("```") {
            let inner = trimmed
                .trim_start_matches("```json")
                .trim_start_matches("```")
                .trim_end_matches("```")
                .trim();
            inner
        } else {
            trimmed
        };

        // Empty array or empty response = no candidates
        if json_str == "[]" || json_str.is_empty() {
            return Ok(Vec::new());
        }

        #[derive(Deserialize)]
        struct RawCandidate {
            content: String,
            memory_type: String,
        }

        let raw_candidates: Vec<RawCandidate> =
            serde_json::from_str(json_str).map_err(|e| MemoryExtractionProviderError {
                message: format!("failed to parse extraction response as JSON: {e}"),
            })?;

        Ok(raw_candidates
            .into_iter()
            .filter_map(|c| {
                let memory_type = match c.memory_type.to_lowercase().as_str() {
                    "preference" => MemoryType::Preference,
                    "fact" => MemoryType::Fact,
                    "context" => MemoryType::Context,
                    "decision" => MemoryType::Decision,
                    _ => MemoryType::Fact,
                };
                Some(StructuredMemoryCandidate {
                    content: c.content,
                    memory_type,
                })
            })
            .collect())
    }
}

#[async_trait]
impl MemoryExtractionProvider for GatewayExtractionAdapter {
    async fn extract(
        &self,
        request: MemoryExtractionProviderRequest,
    ) -> Result<Vec<StructuredMemoryCandidate>, MemoryExtractionProviderError> {
        debug_assert_eq!(request.internal_tag, MEMORY_EXTRACTION_INTERNAL_TAG);

        let provider = self.resolve_provider(&request.provider).await?;

        let api_key = provider.resolve_api_key().unwrap_or_default();

        // Build base URL — strip trailing slash, append /v1 if missing
        let mut base_url = provider.base_url.clone().unwrap_or_default();
        base_url = base_url.trim_end_matches('/').to_string();
        if !base_url.ends_with("/v1") {
            base_url.push_str("/v1");
        }
        let url = format!("{}/chat/completions", base_url);

        let body = Self::build_request_body(&request.model, &request.messages);

        let mut builder = self
            .http_client
            .post(&url)
            .header("Authorization", format!("Bearer {}", api_key))
            .header("Content-Type", "application/json")
            .json(&body);

        // Apply custom headers
        for (key, value) in &provider.custom_headers {
            let resolved = if value.trim().starts_with("${") && value.trim().ends_with('}') {
                let var_name = &value.trim()[2..value.trim().len() - 1];
                std::env::var(var_name).unwrap_or_else(|_| value.clone())
            } else {
                value.clone()
            };
            builder = builder.header(key.as_str(), resolved);
        }

        let response = builder
            .send()
            .await
            .map_err(|e| MemoryExtractionProviderError {
                message: format!("extraction HTTP request failed: {e}"),
            })?;

        let status = response.status();
        if !status.is_success() {
            let error_text = response
                .text()
                .await
                .unwrap_or_else(|_| "unknown error".to_string());
            return Err(MemoryExtractionProviderError {
                message: format!(
                    "extraction provider returned HTTP {}: {}",
                    status.as_u16(),
                    error_text
                ),
            });
        }

        let response_json: serde_json::Value =
            response
                .json()
                .await
                .map_err(|e| MemoryExtractionProviderError {
                    message: format!("failed to parse extraction response body: {e}"),
                })?;

        // Extract the assistant message content from the OpenAI-format response
        let content = response_json
            .get("choices")
            .and_then(|c| c.get(0))
            .and_then(|c| c.get("message"))
            .and_then(|m| m.get("content"))
            .and_then(|c| c.as_str())
            .unwrap_or("");

        Self::parse_response(content)
    }
}
