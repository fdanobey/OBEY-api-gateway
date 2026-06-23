//! `CodexProviderClient` — `ProviderClient` implementation for the Codex backend.
//!
//! Ties together JWT extraction, request translation, response translation,
//! outgoing headers, and the 401-refresh retry dance into a single
//! [`ProviderClient`] implementation that the gateway router can dispatch to.

use std::pin::Pin;
use std::sync::Arc;
use std::time::Instant;

use async_trait::async_trait;
use futures::{Stream, StreamExt};
use serde_json::Value;

use crate::codex::effort_map::map_effort;
use crate::codex::errors::CodexError;
use crate::codex::instructions::InstructionsStore;
use crate::codex::jwt::extract_chatgpt_account_id;
use crate::codex::model_map::{
    is_reasoning, resolve_model, REASONING_MODEL_PATTERNS, XHIGH_MODEL_PATTERNS,
};
use crate::codex::sse::{ResponsesEvent, SseLineParser};
use crate::codex::translate_request::ChatToResponsesTranslator;
use crate::codex::translate_response::{accumulate, stream_translate};
use crate::error::GatewayError;
use crate::metrics::Metrics;
use crate::models::openai::{OpenAIRequest, OpenAIResponse};
use crate::oauth::OAuthManager;
use crate::providers::{Model, ProviderClient, ProviderResponse, SSEEvent};

const DEFAULT_CODEX_BASE_URL: &str = "https://chatgpt.com/backend-api/codex/responses";

/// Provider client that dispatches Chat Completions requests through the
/// Codex Responses API backend, translating on the fly.
pub struct CodexProviderClient {
    provider_name: String,
    http: reqwest::Client,
    oauth: Arc<OAuthManager>,
    instructions: Arc<InstructionsStore>,
    base_url: String,
    instructions_override: Option<String>,
    model_override: Option<String>,
    #[allow(dead_code)] // stored for future use; not yet read
    metrics: Arc<Metrics>,
    /// Operator-configured allowlists (from gateway config).
    xhigh_models_allowlist: Vec<String>,
    reasoning_models_allowlist: Vec<String>,
}

impl CodexProviderClient {
    /// Construct a new Codex provider client.
    ///
    /// Fields that will later come from `Provider` config (task 17) are
    /// accepted as explicit parameters for now.
    pub fn new(
        provider_name: String,
        oauth: Arc<OAuthManager>,
        instructions: Arc<InstructionsStore>,
        http: reqwest::Client,
        metrics: Arc<Metrics>,
        base_url_override: Option<String>,
        model_override: Option<String>,
        instructions_override: Option<String>,
        xhigh_models_allowlist: Vec<String>,
        reasoning_models_allowlist: Vec<String>,
    ) -> Self {
        let base_url = base_url_override.unwrap_or_else(|| DEFAULT_CODEX_BASE_URL.to_string());
        Self {
            provider_name,
            http,
            oauth,
            instructions,
            base_url,
            instructions_override,
            model_override,
            metrics,
            xhigh_models_allowlist,
            reasoning_models_allowlist,
        }
    }
}

// ─── ProviderClient trait implementation ─────────────────────────────────────

#[async_trait]
impl ProviderClient for CodexProviderClient {
    async fn chat_completion(
        &self,
        request: OpenAIRequest,
    ) -> Result<ProviderResponse, GatewayError> {
        let start = Instant::now();
        let result = self.chat_completion_with_retry(&request).await?;
        let latency_ms = start.elapsed().as_millis() as u64;

        let openai_response: OpenAIResponse = serde_json::from_value(result).map_err(|e| {
            GatewayError::Provider {
                provider: self.provider_name.clone(),
                message: format!("response deserialization failed: {e}"),
                status_code: None,
            }
        })?;

        Ok(ProviderResponse {
            response: openai_response,
            provider_name: self.provider_name.clone(),
            latency_ms,
        })
    }

    async fn chat_completion_stream(
        &self,
        request: OpenAIRequest,
    ) -> Result<Pin<Box<dyn Stream<Item = Result<SSEEvent, GatewayError>> + Send>>, GatewayError>
    {
        let events_stream = self.dispatch_streaming(&request).await?;
        let resolved_model = self.resolve_model_for_request(&request);
        let provider_name = self.provider_name.clone();

        let sse_stream = stream_translate(events_stream, resolved_model);
        let mapped = sse_stream.map(move |result| {
            result
                .map(|data| SSEEvent::new(data))
                .map_err(|e| GatewayError::Provider {
                    provider: provider_name.clone(),
                    message: format!("Codex stream error: {e}"),
                    status_code: Some(502),
                })
        });

        Ok(Box::pin(mapped))
    }

    async fn list_models(&self) -> Result<Vec<Model>, GatewayError> {
        // Codex backend has no /v1/models; return static hint list.
        let mut models: Vec<Model> = Vec::new();

        for pattern in XHIGH_MODEL_PATTERNS.iter().chain(REASONING_MODEL_PATTERNS.iter()) {
            models.push(Model {
                id: pattern.to_string(),
                object: "model".to_string(),
                owned_by: "openai-codex".to_string(),
                created: None,
                context_window: None,
                max_completion_tokens: None,
            });
        }

        // Add allowlist entries that aren't already present.
        for m in &self.xhigh_models_allowlist {
            if !models.iter().any(|existing| existing.id == *m) {
                models.push(Model {
                    id: m.clone(),
                    object: "model".to_string(),
                    owned_by: "openai-codex".to_string(),
                    created: None,
                    context_window: None,
                    max_completion_tokens: None,
                });
            }
        }
        for m in &self.reasoning_models_allowlist {
            if !models.iter().any(|existing| existing.id == *m) {
                models.push(Model {
                    id: m.clone(),
                    object: "model".to_string(),
                    owned_by: "openai-codex".to_string(),
                    created: None,
                    context_window: None,
                    max_completion_tokens: None,
                });
            }
        }

        Ok(models)
    }

    fn provider_name(&self) -> &str {
        &self.provider_name
    }
}

// ─── Outgoing headers and URL composition (Task 15.2) ────────────────────────

impl CodexProviderClient {
    /// Build the outgoing HTTP request with all required Codex headers.
    #[tracing::instrument(skip(self, access_token, account_id, body))]
    fn build_request(
        &self,
        access_token: &str,
        account_id: &str,
        body: &Value,
    ) -> reqwest::RequestBuilder {
        self.http
            .post(&self.base_url)
            .header("authorization", format!("Bearer {access_token}"))
            .header("chatgpt-account-id", account_id)
            .header("openai-beta", "responses=experimental")
            .header("originator", "codex_cli_rs")
            .header("accept", "text/event-stream")
            .header("content-type", "application/json")
            .json(body)
    }
}

// ─── Retry logic and dispatch (Task 15.3) ────────────────────────────────────

impl CodexProviderClient {
    #[allow(dead_code)]
    fn resolve_model_for_request(&self, req: &OpenAIRequest) -> String {
        resolve_model(&req.model, self.model_override.as_deref()).to_string()
    }

    /// Non-streaming dispatch with 401-refresh retry dance.
    async fn chat_completion_with_retry(&self, req: &OpenAIRequest) -> Result<Value, GatewayError> {
        let access_token = self.oauth.get_access_token().await.ok_or_else(|| {
            tracing::debug!(provider = %self.provider_name, "Codex provider skipped: no valid OAuth token");
            GatewayError::Provider {
                provider: self.provider_name.clone(),
                message: "OAuth session not authenticated".to_string(),
                status_code: Some(401),
            }
        })?;

        let account_id = extract_chatgpt_account_id(&access_token).map_err(|e| {
            tracing::warn!(provider = %self.provider_name, error = %e, "JWT account-id extraction failed");
            GatewayError::Provider {
                provider: self.provider_name.clone(),
                message: format!("JWT extraction failed: {e}"),
                status_code: Some(401),
            }
        })?;

        let body = self.build_translated_body(req).await?;

        // First dispatch attempt.
        match self.dispatch_once(&access_token, &account_id, &body).await {
            Ok(value) => Ok(value),
            Err(GatewayError::Provider {
                status_code: Some(401),
                ..
            }) => {
                // 401 → force refresh before consuming retry budget (Req 10.4).
                tracing::info!(provider = %self.provider_name, "Upstream 401; forcing OAuth refresh");
                let new_token = self.oauth.force_refresh().await.map_err(|e| {
                    GatewayError::Provider {
                        provider: self.provider_name.clone(),
                        message: format!("OAuth refresh after 401 failed: {e}"),
                        status_code: Some(401),
                    }
                })?;
                let new_account_id =
                    extract_chatgpt_account_id(&new_token).map_err(|e| {
                        GatewayError::Provider {
                            provider: self.provider_name.clone(),
                            message: format!("JWT extraction failed after refresh: {e}"),
                            status_code: Some(401),
                        }
                    })?;
                // Retry once with fresh token.
                self.dispatch_once(&new_token, &new_account_id, &body).await
            }
            Err(e @ GatewayError::Provider {
                status_code: Some(403),
                ..
            }) => {
                // 403 → surface immediately, no refresh (Req 10.5).
                tracing::warn!(provider = %self.provider_name, "Upstream 403; not retrying");
                Err(e)
            }
            Err(e) => Err(e),
        }
    }

    /// Translate the incoming Chat Completions request into a Responses API body.
    async fn build_translated_body(&self, req: &OpenAIRequest) -> Result<Value, GatewayError> {
        let resolved_model = resolve_model(&req.model, self.model_override.as_deref());
        let reasoning_effort_str = req.extra.get("reasoning_effort").and_then(|v| v.as_str());
        let mapped_effort =
            map_effort(reasoning_effort_str, resolved_model, &self.xhigh_models_allowlist);
        let emit_reasoning = is_reasoning(resolved_model, &self.reasoning_models_allowlist);
        let instructions = self.instructions.get(self.instructions_override.as_deref()).await;

        let translator = ChatToResponsesTranslator {
            resolved_model,
            instructions: &instructions,
            mapped_effort,
            emit_reasoning,
        };

        translator.translate(req).map_err(|e| match e {
            CodexError::UnsupportedFeature { feature } => GatewayError::InvalidRequest(format!(
                "Unsupported feature for Codex backend: {feature}"
            )),
            other => GatewayError::Provider {
                provider: self.provider_name.clone(),
                message: format!("Request translation failed: {other}"),
                status_code: None,
            },
        })
    }

    /// Single HTTP round-trip: send the request, read the full SSE body,
    /// parse events, and accumulate into a Chat Completions JSON value.
    async fn dispatch_once(
        &self,
        access_token: &str,
        account_id: &str,
        body: &Value,
    ) -> Result<Value, GatewayError> {
        let resp = self
            .build_request(access_token, account_id, body)
            .send()
            .await
            .map_err(|e| GatewayError::Provider {
                provider: self.provider_name.clone(),
                message: format!("HTTP request failed: {e}"),
                status_code: None,
            })?;

        let status = resp.status().as_u16();
        if status == 401 || status == 403 {
            tracing::warn!(provider = %self.provider_name, status, "Codex upstream auth error");
            return Err(GatewayError::Provider {
                provider: self.provider_name.clone(),
                message: format!("Upstream returned {status}"),
                status_code: Some(status),
            });
        }
        if !resp.status().is_success() {
            let body_text = resp.text().await.unwrap_or_default();
            tracing::warn!(
                provider = %self.provider_name,
                status,
                body = %body_text.chars().take(500).collect::<String>(),
                "Codex upstream returned non-2xx"
            );
            return Err(GatewayError::Provider {
                provider: self.provider_name.clone(),
                message: format!("Upstream HTTP {status}: {body_text}"),
                status_code: Some(status),
            });
        }

        // Read the full body and parse SSE events.
        let bytes = resp.bytes().await.map_err(|e| GatewayError::Provider {
            provider: self.provider_name.clone(),
            message: format!("Failed to read response body: {e}"),
            status_code: None,
        })?;

        let mut parser = SseLineParser::new();
        let parse_results = parser.feed(&bytes);
        let mut events: Vec<ResponsesEvent> = Vec::new();
        for result in parse_results {
            match result {
                Ok(event) => events.push(event),
                Err(e) => {
                    tracing::warn!(provider = %self.provider_name, error = %e, "SSE parse error on event");
                }
            }
        }

        if events.is_empty() {
            let body_preview = String::from_utf8_lossy(&bytes);
            let truncated: String = body_preview.chars().take(500).collect();
            tracing::warn!(
                provider = %self.provider_name,
                body_preview = %truncated,
                "Codex upstream returned 2xx but SSE parser produced zero events"
            );
            return Err(GatewayError::Provider {
                provider: self.provider_name.clone(),
                message: format!("Upstream returned 2xx but no parseable SSE events. Body preview: {truncated}"),
                status_code: Some(502),
            });
        }

        tracing::debug!(
            provider = %self.provider_name,
            event_count = events.len(),
            "Parsed SSE events from Codex upstream"
        );

        let event_stream = futures::stream::iter(events);
        accumulate(event_stream).await.map_err(|e| {
            tracing::warn!(
                provider = %self.provider_name,
                error = %e,
                "Response accumulation failed after parsing SSE events"
            );
            GatewayError::Provider {
                provider: self.provider_name.clone(),
                message: format!("Response accumulation failed: {e}"),
                status_code: Some(502),
            }
        })
    }

    /// Dispatch a streaming request and return a stream of `ResponsesEvent`.
    #[allow(dead_code)]
    async fn dispatch_streaming(
        &self,
        req: &OpenAIRequest,
    ) -> Result<Pin<Box<dyn Stream<Item = ResponsesEvent> + Send>>, GatewayError> {
        let access_token = self.oauth.get_access_token().await.ok_or_else(|| {
            GatewayError::Provider {
                provider: self.provider_name.clone(),
                message: "OAuth session not authenticated".to_string(),
                status_code: Some(401),
            }
        })?;

        let account_id = extract_chatgpt_account_id(&access_token).map_err(|e| {
            GatewayError::Provider {
                provider: self.provider_name.clone(),
                message: format!("JWT extraction failed: {e}"),
                status_code: Some(401),
            }
        })?;

        let body = self.build_translated_body(req).await?;

        let resp = self
            .build_request(&access_token, &account_id, &body)
            .send()
            .await
            .map_err(|e| GatewayError::Provider {
                provider: self.provider_name.clone(),
                message: format!("HTTP request failed: {e}"),
                status_code: None,
            })?;

        let status = resp.status().as_u16();
        if !resp.status().is_success() {
            return Err(GatewayError::Provider {
                provider: self.provider_name.clone(),
                message: format!("Upstream HTTP {status}"),
                status_code: Some(status),
            });
        }

        // Stream the response bytes and parse SSE events on the fly.
        let byte_stream = resp.bytes_stream();
        let event_stream = byte_stream
            .scan(SseLineParser::new(), |parser, chunk_result| {
                let events = match chunk_result {
                    Ok(bytes) => parser
                        .feed(&bytes)
                        .into_iter()
                        .filter_map(|r| r.ok())
                        .collect::<Vec<_>>(),
                    Err(_) => vec![],
                };
                futures::future::ready(Some(futures::stream::iter(events)))
            })
            .flatten();

        Ok(Box::pin(event_stream))
    }
}
