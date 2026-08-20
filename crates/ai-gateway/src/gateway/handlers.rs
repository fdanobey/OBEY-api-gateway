//! OpenAI API endpoint handlers for the OBEY-API gateway.
//!
//! Requirements: 2.1-2.12

use axum::{
    body::{Body, Bytes},
    extract::{FromRequestParts, Json, Path, Query, State},
    http::{header, HeaderMap, HeaderName, HeaderValue, StatusCode},
    response::{
        sse::{Event, KeepAlive, Sse},
        IntoResponse, Response,
    },
    Extension,
};
use futures::StreamExt;
use serde::Serialize;
use std::convert::Infallible;
use std::error::Error as StdError;
use std::sync::atomic::Ordering;
use std::sync::Arc;

use std::time::{Duration, Instant};

use crate::cache::ExactCache;
use crate::compression::stats::CompressionStats;
use crate::compression::token_counter::TokenCounter;
use crate::config::{load_and_validate_config, ModelGroup, ProviderModel, StreamingConfig};
use crate::error::{AggregatedError, GatewayError, ProviderAttempt};
use crate::gateway::apply_runtime_config_update;
use crate::guardrail::{
    stream as guardrail_stream, BindingSelector, GuardrailContext, GuardrailEngine,
    PostCallOutcome, PreCallOutcome, RefusalDecision, StagePhase, ToolContext,
};
use crate::logger::{CompressionLogMetadata, LogEntry};
use crate::memory::{
    format_feedback_suffix, ContextType, EffectiveMemoryConfig, ExtractionCounts,
    ExtractionMessage, ExtractionRole, InjectionResult, MemoryRequestResult, ResolvedNamespace,
};
use crate::metrics::Metrics;
use crate::models::openai::{Choice, OpenAIRequest, OpenAIResponse, Usage};
use crate::providers::Model;
use crate::router::router::{ProviderPassThroughEndpoint, ProviderPassThroughResponse};
use crate::router::trace_id::generate_trace_id;
use crate::router::StreamingResponse;
use crate::smart_routing::tier::{ClassifierUsed, RoutingDecision, SmartRoutingTier, TaskType};
use crate::structured_output::validator::{
    ChoiceValidationOutcome, ChoiceValidationResult, SchemaViolation,
};
use crate::structured_output::{
    StructuredOutputEngine, StructuredOutputOutcome, ValidationDecision, ValidationSkipReason,
};
use crate::virtual_keys::access::AccessError;
use crate::virtual_keys::models::AuthenticatedKey;

pub struct AssistantsIdentity(AuthenticatedKey);

impl<S> FromRequestParts<S> for AssistantsIdentity
where
    S: Send + Sync,
{
    type Rejection = Response;

    async fn from_request_parts(
        parts: &mut axum::http::request::Parts,
        _state: &S,
    ) -> Result<Self, Self::Rejection> {
        parts
            .extensions
            .get::<AuthenticatedKey>()
            .cloned()
            .map(Self)
            .ok_or_else(assistants_authentication_required)
    }
}

#[derive(Debug, Clone)]
struct RequestLogContext {
    trace_id: String,
    status_code: u16,
    duration_ms: u64,
    provider: String,
    requested_model: String,
    responded_model: Option<String>,
    cost: f64,
    /// Detailed error message for failed requests (shown in dashboard log viewer).
    error_message: Option<String>,
    compression: Option<CompressionLogMetadata>,
    memories_injected: u32,
    memories_stored: u32,
    injection_tokens: u32,
    detected_project: Option<String>,
}

impl RequestLogContext {
    fn with_memory(
        mut self,
        memory: Option<(&ContextType, &InjectionResult)>,
        extraction: Option<ExtractionCounts>,
    ) -> Self {
        if let Some((context, injection)) = memory {
            self.memories_injected = injection.memories_injected;
            self.injection_tokens = injection.injection_tokens;
            self.detected_project = match context {
                ContextType::Project(hash) => Some(hash.clone()),
                ContextType::Agent(_) | ContextType::User => None,
            };
        }
        if let Some(extraction) = extraction {
            self.memories_stored = extraction.stored;
        }
        self
    }

    fn from_response(
        request: &OpenAIRequest,
        trace_id: String,
        duration_ms: u64,
        response: &crate::models::openai::OpenAIResponse,
    ) -> Self {
        Self::from_response_with_compression(request, trace_id, duration_ms, response, None)
    }

    fn from_response_with_compression(
        request: &OpenAIRequest,
        trace_id: String,
        duration_ms: u64,
        response: &crate::models::openai::OpenAIResponse,
        fallback_compression: Option<&CompressionStats>,
    ) -> Self {
        Self {
            trace_id,
            status_code: StatusCode::OK.as_u16(),
            duration_ms,
            provider: response
                .extra
                .get("gateway_provider")
                .and_then(|v| v.as_str())
                .unwrap_or_default()
                .to_string(),
            requested_model: request.model.clone(),
            responded_model: response
                .extra
                .get("gateway_responded_model")
                .and_then(|v| v.as_str())
                .map(|s| s.to_string())
                .or_else(|| {
                    if response.model.is_empty() {
                        None
                    } else {
                        Some(response.model.clone())
                    }
                }),
            cost: response
                .extra
                .get("gateway_cost")
                .and_then(|v| v.as_f64())
                .unwrap_or(0.0),
            error_message: None,
            compression: response
                .extra
                .get("gateway_compression")
                .and_then(|value| serde_json::from_value::<CompressionStats>(value.clone()).ok())
                .as_ref()
                .map(CompressionLogMetadata::from)
                .or_else(|| fallback_compression.map(CompressionLogMetadata::from)),
            memories_injected: 0,
            memories_stored: 0,
            injection_tokens: 0,
            detected_project: None,
        }
    }

    fn from_streaming_success(
        request: &OpenAIRequest,
        trace_id: String,
        duration_ms: u64,
        provider: String,
        model: String,
        compression: CompressionStats,
    ) -> Self {
        Self {
            trace_id,
            status_code: StatusCode::OK.as_u16(),
            duration_ms,
            provider,
            requested_model: request.model.clone(),
            responded_model: Some(model),
            cost: 0.0,
            error_message: None,
            compression: Some(CompressionLogMetadata::from(&compression)),
            memories_injected: 0,
            memories_stored: 0,
            injection_tokens: 0,
            detected_project: None,
        }
    }

    fn from_error(
        request: &OpenAIRequest,
        trace_id: String,
        duration_ms: u64,
        error: &GatewayError,
    ) -> Self {
        let provider = match error {
            GatewayError::Provider { provider, .. } => provider.clone(),
            GatewayError::AllProvidersFailed(agg) => agg
                .attempts
                .first()
                .map(|attempt| attempt.provider.clone())
                .unwrap_or_default(),
            _ => String::new(),
        };
        let error_message = match error {
            GatewayError::Provider { message, .. } => Some(message.clone()),
            GatewayError::AllProvidersFailed(agg) => Some(
                agg.attempts
                    .iter()
                    .map(|a| format!("[{}] {}", a.provider, a.error))
                    .collect::<Vec<_>>()
                    .join("; "),
            ),
            other => Some(other.to_string()),
        };
        Self {
            trace_id,
            status_code: error.status_code().as_u16(),
            duration_ms,
            provider,
            requested_model: request.model.clone(),
            responded_model: None,
            cost: 0.0,
            error_message,
            compression: None,
            memories_injected: 0,
            memories_stored: 0,
            injection_tokens: 0,
            detected_project: None,
        }
    }
}

/// Log a completed request to the SQLite database for the dashboard log viewer.
fn log_request(state: &super::AppState, request: &OpenAIRequest, context: &RequestLogContext) {
    let entry = LogEntry {
        trace_id: context.trace_id.clone(),
        timestamp: chrono::Utc::now(),
        method: "POST".to_string(),
        path: "/v1/chat/completions".to_string(),
        model: context
            .responded_model
            .clone()
            .unwrap_or_else(|| context.requested_model.clone()),
        provider: context.provider.clone(),
        status_code: context.status_code,
        duration_ms: context.duration_ms,
        cost: context.cost,
        request_body: serde_json::to_string(request).ok(),
        response_body: context.error_message.clone(),
        requested_model: Some(request.model.clone()),
        responded_model: context.responded_model.clone(),
        compression: context.compression.clone(),
        memories_injected: context.memories_injected,
        memories_stored: context.memories_stored,
        injection_tokens: context.injection_tokens,
        detected_project: context.detected_project.clone(),
    };
    if let Err(e) = state.logger.log(entry) {
        tracing::warn!(error = %e, trace_id = %context.trace_id, "Failed to write request log entry");
    }
}

fn strip_gateway_response_metadata(response: &mut OpenAIResponse) {
    response.extra.remove("gateway_provider");
    response.extra.remove("gateway_responded_model");
    response.extra.remove("gateway_cost");
    response.extra.remove("gateway_compression");
    response.extra.remove("gateway_smart_routing");
}

fn prepare_response_for_client(response: &OpenAIResponse) -> OpenAIResponse {
    let mut client_response = response.clone();
    strip_gateway_response_metadata(&mut client_response);
    client_response
}

const SMART_ROUTE_TIER_HEADER: &str = "x-smart-route-tier";
const SMART_ROUTE_SCORE_HEADER: &str = "x-smart-route-score";
const SMART_ROUTE_CLASSIFIER_HEADER: &str = "x-smart-route-classifier";
const SMART_ROUTE_TASK_TYPE_HEADER: &str = "x-smart-route-task-type";
const SMART_ROUTE_ESCALATED_HEADER: &str = "x-smart-route-escalated";
const SMART_ROUTE_CACHE_HIT_HEADER: &str = "x-smart-route-cache-hit";
const BUDGET_EXCEEDED_HEADER: &str = "x-budget-exceeded";
const MAX_DYNAMIC_HEADER_VALUE_LEN: usize = 128;

fn ascii_bounded_header_value(value: &str) -> Option<HeaderValue> {
    if value.is_empty() || value.len() > MAX_DYNAMIC_HEADER_VALUE_LEN || !value.is_ascii() {
        return None;
    }
    HeaderValue::from_str(value).ok()
}

fn smart_routing_tier_value(tier: SmartRoutingTier) -> &'static str {
    match tier {
        SmartRoutingTier::Fast => "fast",
        SmartRoutingTier::Balanced => "balanced",
        SmartRoutingTier::Powerful => "powerful",
    }
}

fn smart_routing_classifier_value(classifier: ClassifierUsed) -> &'static str {
    match classifier {
        ClassifierUsed::Heuristic => "heuristic",
        ClassifierUsed::Ml => "ml",
        ClassifierUsed::Llm => "llm",
        ClassifierUsed::Composite => "composite",
    }
}

fn smart_routing_task_type_value(task_type: TaskType) -> &'static str {
    match task_type {
        TaskType::CodeGeneration => "code_generation",
        TaskType::MathReasoning => "math_reasoning",
        TaskType::CreativeWriting => "creative_writing",
        TaskType::FactualQA => "factual_qa",
        TaskType::ToolUse => "tool_use",
        TaskType::Summarization => "summarization",
        TaskType::General => "general",
    }
}

fn smart_routing_headers(response: &OpenAIResponse) -> HeaderMap {
    let Some(decision) = response
        .extra
        .get("gateway_smart_routing")
        .and_then(|value| serde_json::from_value::<RoutingDecision>(value.clone()).ok())
    else {
        return HeaderMap::new();
    };

    let mut headers = HeaderMap::new();
    headers.insert(
        HeaderName::from_static(SMART_ROUTE_TIER_HEADER),
        HeaderValue::from_static(smart_routing_tier_value(decision.tier)),
    );
    let score = decision.score.value();
    let score = if score == 0.0 { 0.0 } else { score };
    if let Ok(score) = HeaderValue::from_str(&format!("{score:.2}")) {
        headers.insert(HeaderName::from_static(SMART_ROUTE_SCORE_HEADER), score);
    }
    headers.insert(
        HeaderName::from_static(SMART_ROUTE_CLASSIFIER_HEADER),
        HeaderValue::from_static(smart_routing_classifier_value(decision.classifier)),
    );
    headers.insert(
        HeaderName::from_static(SMART_ROUTE_TASK_TYPE_HEADER),
        HeaderValue::from_static(smart_routing_task_type_value(decision.task_type)),
    );
    if decision.escalated {
        headers.insert(
            HeaderName::from_static(SMART_ROUTE_ESCALATED_HEADER),
            HeaderValue::from_static("true"),
        );
    }
    if decision.cache_hit {
        headers.insert(
            HeaderName::from_static(SMART_ROUTE_CACHE_HIT_HEADER),
            HeaderValue::from_static("true"),
        );
    }
    headers
}

fn attach_smart_routing_headers(response: &mut Response, routing_headers: HeaderMap) {
    response.headers_mut().extend(routing_headers);
}

fn openai_json_response(response: &OpenAIResponse) -> Response {
    let routing_headers = smart_routing_headers(response);
    let client_response = prepare_response_for_client(response);
    let mut http_response = Json(client_response).into_response();
    attach_smart_routing_headers(&mut http_response, routing_headers);
    http_response
}

fn trace_id_from_headers(headers: &HeaderMap) -> String {
    let request_id = headers
        .get("x-request-id")
        .or_else(|| headers.get("x-trace-id"))
        .and_then(|value| value.to_str().ok());
    generate_trace_id(request_id)
}

fn attach_trace_id_header(response: &mut Response, trace_id: &str) {
    let header_name = HeaderName::from_static("x-trace-id");
    if let Ok(header_value) = HeaderValue::from_str(trace_id) {
        response.headers_mut().insert(header_name, header_value);
    }
}

const VALIDATION_STATUS_HEADER: &str = "x-obey-validation-status";

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum ValidationResponseStatus {
    NotApplicable,
    Passed,
    Failed,
    Skipped,
}

impl ValidationResponseStatus {
    fn from_outcome(outcome: StructuredOutputOutcome) -> Self {
        match outcome {
            StructuredOutputOutcome::NotApplicable => Self::NotApplicable,
            StructuredOutputOutcome::Pass => Self::Passed,
            StructuredOutputOutcome::Fail => Self::Failed,
            StructuredOutputOutcome::Skipped => Self::Skipped,
        }
    }

    fn header_value(self) -> Option<&'static str> {
        match self {
            Self::NotApplicable => None,
            Self::Passed => Some("passed"),
            Self::Failed => Some("failed"),
            Self::Skipped => Some("skipped"),
        }
    }
}

fn attach_validation_status_header(
    response: &mut Response,
    status: ValidationResponseStatus,
    applicable_json_schema: bool,
) {
    if !applicable_json_schema {
        return;
    }
    let Some(value) = status.header_value() else {
        return;
    };
    response.headers_mut().insert(
        HeaderName::from_static(VALIDATION_STATUS_HEADER),
        HeaderValue::from_static(value),
    );
}

fn requests_structured_output(request: &OpenAIRequest) -> bool {
    matches!(
        request
            .extra
            .get("response_format")
            .and_then(|value| value.as_object())
            .and_then(|response_format| response_format.get("type"))
            .and_then(|value| value.as_str()),
        Some("json_object" | "json_schema")
    )
}

fn requests_json_schema(request: &OpenAIRequest) -> bool {
    request
        .extra
        .get("response_format")
        .and_then(|value| value.as_object())
        .and_then(|response_format| response_format.get("type"))
        .and_then(|value| value.as_str())
        == Some("json_schema")
}

fn cache_allowed_for_validation(
    request: &OpenAIRequest,
    validation_status: ValidationResponseStatus,
) -> bool {
    !requests_json_schema(request) || matches!(validation_status, ValidationResponseStatus::Passed)
}

fn force_eager_structured_stream(request: &OpenAIRequest) -> bool {
    request.stream && requests_json_schema(request)
}

fn structured_stream_overflow_events(trace_id: &str) -> Vec<Event> {
    let message = format!(
        "Buffered streaming response exceeded the {} byte structured output buffer limit",
        guardrail_stream::MAX_STREAM_BUFFER_BYTES
    );
    emit_sse_error_event("structured_output_buffer_overflow", &message, trace_id)
}

fn should_cache_eager_structured(
    request: &OpenAIRequest,
    response: Option<&OpenAIResponse>,
    validation_status: Option<ValidationResponseStatus>,
) -> bool {
    validation_status.is_some_and(|status| cache_allowed_for_validation(request, status))
        && response.is_some_and(crate::router::router::Router::should_cache_response)
}

fn rechunk_structured_response(response: &OpenAIResponse) -> Vec<Event> {
    let client_response = prepare_response_for_client(response);
    let mut events = guardrail_stream::rechunk_full(&client_response)
        .into_iter()
        .map(|chunk| Event::default().data(chunk.to_string()))
        .collect::<Vec<_>>();
    events.push(Event::default().data("[DONE]"));
    events
}

fn eager_sse_response(
    events: Vec<Event>,
    streaming_config: &StreamingConfig,
    trace_id: &str,
    validation_status: Option<ValidationResponseStatus>,
    routing_headers: Option<HeaderMap>,
) -> Response {
    let stream = futures::stream::iter(events.into_iter().map(Ok::<_, Infallible>));
    let mut response = Sse::new(stream)
        .keep_alive(build_keepalive(streaming_config))
        .into_response();
    if let Some(routing_headers) = routing_headers {
        attach_smart_routing_headers(&mut response, routing_headers);
    }
    if let Some(status) = validation_status {
        attach_validation_status_header(&mut response, status, true);
    }
    attach_trace_id_header(&mut response, trace_id);
    response
}

#[derive(Debug, Clone, PartialEq, Eq)]
struct StructuredOutputFailure {
    violations: Vec<SchemaViolation>,
    previous_output: String,
}

fn collect_structured_output_failure(
    response: &OpenAIResponse,
    choices: &[ChoiceValidationOutcome],
) -> StructuredOutputFailure {
    let mut violations = Vec::new();
    let mut previous_outputs = Vec::new();

    for (index, choice) in choices.iter().enumerate() {
        match &choice.result {
            ChoiceValidationResult::Pass | ChoiceValidationResult::Skipped => {}
            ChoiceValidationResult::JsonParseError {
                byte_offset,
                expected,
            } => {
                violations.push(SchemaViolation {
                    path: format!("/choices/{index}"),
                    expected: format!("valid JSON at byte {byte_offset}: {expected}"),
                    actual: "invalid JSON".to_owned(),
                });
                if let Some(response_choice) = response.choices.get(index) {
                    previous_outputs.push(response_choice.message.content_as_text());
                }
            }
            ChoiceValidationResult::SchemaViolations(choice_violations) => {
                violations.extend(choice_violations.iter().cloned());
                if let Some(response_choice) = response.choices.get(index) {
                    previous_outputs.push(response_choice.message.content_as_text());
                }
            }
        }
    }

    StructuredOutputFailure {
        violations,
        previous_output: previous_outputs.join("\n"),
    }
}

fn response_provider_model(response: &OpenAIResponse) -> Option<(&str, &str)> {
    Some((
        response.extra.get("gateway_provider")?.as_str()?,
        response.extra.get("gateway_responded_model")?.as_str()?,
    ))
}

fn find_selected_provider_model(
    model_group: &ModelGroup,
    provider: &str,
    model: &str,
) -> Option<ProviderModel> {
    model_group
        .models
        .iter()
        .find(|candidate| candidate.provider == provider && candidate.model == model)
        .cloned()
}

fn tool_context(request: &OpenAIRequest, response: &OpenAIResponse) -> ToolContext {
    ToolContext {
        tool_use_allowed: request
            .extra
            .get("tool_choice")
            .and_then(|value| value.as_str())
            .map_or(true, |tool_choice| tool_choice != "none"),
        tools_provided: request
            .extra
            .get("tools")
            .and_then(|value| value.as_array())
            .is_some_and(|tools| !tools.is_empty()),
        finish_reason_is_tool_call: response
            .choices
            .first()
            .and_then(|choice| choice.finish_reason.as_deref())
            == Some("tool_calls"),
        has_tool_calls: response.choices.first().is_some_and(|choice| {
            choice
                .message
                .extra
                .get("tool_calls")
                .and_then(|value| value.as_array())
                .is_some_and(|tool_calls| !tool_calls.is_empty())
        }),
    }
}

fn consume_provider_error_attempts(error: &GatewayError) -> usize {
    match error {
        GatewayError::AllProvidersFailed(aggregated) => aggregated.attempts.len().max(1),
        _ => 1,
    }
}

fn validation_skip_category(reason: &ValidationSkipReason) -> &'static str {
    match reason {
        ValidationSkipReason::Disabled => "disabled",
        ValidationSkipReason::Passthrough => "passthrough",
        ValidationSkipReason::Malformed(_) => "malformed",
        ValidationSkipReason::CompileFailed(_) => "compile_failure",
    }
}

fn default_context_window() -> usize {
    crate::config::ContextConfig::default().default_context_window as usize
}

#[derive(Clone)]
struct MemoryRequestContext {
    system: Arc<crate::memory::MemorySystem>,
    context: ContextType,
    namespace: ResolvedNamespace,
    effective: EffectiveMemoryConfig,
    injection: InjectionResult,
    extraction_messages: Vec<ExtractionMessage>,
}

async fn preprocess_memory_request(
    state: &AppState,
    request: &mut OpenAIRequest,
    virtual_key_id: Option<&str>,
) -> Option<MemoryRequestContext> {
    let system = state.memory_system.read().await.clone()?;
    let model_group = state.router.find_model_group(&request.model).await.ok()?;
    let ordered = state.router.select_provider_order(&model_group).await;
    let provider_model = ordered.first().or_else(|| model_group.models.first())?;
    let (effective, default_context_window) = {
        let config = state.config.read().await;
        let memory = config.memory.as_ref()?;
        let provider_override = config
            .providers
            .iter()
            .find(|provider| provider.name == provider_model.provider)
            .and_then(|provider| provider.memory.as_ref());
        (
            memory.resolve(provider_override, model_group.memory.as_ref()),
            config.context.default_context_window,
        )
    };
    if !effective.enabled {
        return None;
    }

    let extraction_messages = extraction_messages(request);
    let query = request
        .messages
        .iter()
        .map(|message| message.content_as_text())
        .filter(|content| !content.is_empty())
        .collect::<Vec<_>>()
        .join("\n");
    let context_window = state
        .router
        .context_manager()
        .get_capabilities(&provider_model.model)
        .map(|capabilities| capabilities.context_window)
        .unwrap_or(default_context_window);
    let post_truncation_tokens = TokenCounter::new().count_request(request);
    let result: MemoryRequestResult = match system
        .process_request(
            request,
            &query,
            context_window,
            post_truncation_tokens,
            effective,
            virtual_key_id,
        )
        .await
    {
        Ok(result) => result,
        Err(error) => {
            tracing::warn!(error = %error, "Memory request preprocessing failed; routing unchanged");
            return None;
        }
    };

    if result.injection.injection_tokens > 200 {
        tracing::info!(
            memories_injected = result.injection.memories_injected,
            injection_tokens = result.injection.injection_tokens,
            "Memory injection added more than 200 tokens"
        );
    }
    if result.injection.memories_injected > 0 {
        state
            .memory_events
            .publish(crate::dashboard::MemoryDashboardEvent::new(
                crate::dashboard::MemoryEventType::Injection,
                result
                    .namespace
                    .context_scope
                    .as_deref()
                    .unwrap_or(&result.namespace.user_scope),
                result.injection.memories_injected,
            ));
    }

    Some(MemoryRequestContext {
        system,
        context: result.context,
        namespace: result.namespace,
        effective,
        injection: result.injection,
        extraction_messages,
    })
}

fn extraction_messages(request: &OpenAIRequest) -> Vec<ExtractionMessage> {
    request
        .messages
        .iter()
        .map(|message| {
            let role = match message.role.as_str() {
                "user" => ExtractionRole::User,
                "assistant" => ExtractionRole::Assistant,
                _ => ExtractionRole::Other,
            };
            ExtractionMessage::caller(role, message.content_as_text())
        })
        .collect()
}

fn append_feedback_to_response(response: &mut OpenAIResponse, suffix: &str) -> bool {
    let Some(choice) = response.choices.first_mut() else {
        return false;
    };
    if !choice.message.extra.get("tool_calls").is_none() {
        return false;
    }
    match &mut choice.message.content {
        serde_json::Value::String(content) => {
            content.push_str(suffix);
            true
        }
        _ => false,
    }
}

async fn finalize_memory_response(
    state: &AppState,
    request: &OpenAIRequest,
    response: &mut OpenAIResponse,
    memory: Option<&MemoryRequestContext>,
    request_id: Option<uuid::Uuid>,
    is_thread_start: bool,
) -> (Option<String>, ExtractionCounts) {
    let Some(memory) = memory else {
        return (None, ExtractionCounts::default());
    };
    let response_content = response
        .choices
        .first()
        .map(|choice| choice.message.content_as_text())
        .unwrap_or_default();
    let extraction = match memory
        .system
        .extract_explicit_response(&memory.extraction_messages, &memory.namespace, request_id)
        .await
    {
        Ok(extraction) => extraction,
        Err(error) => {
            tracing::warn!(error = %error, "Memory response extraction failed");
            return (None, ExtractionCounts::default());
        }
    };
    if extraction.stored > 0 {
        state
            .memory_events
            .publish(crate::dashboard::MemoryDashboardEvent::new(
                crate::dashboard::MemoryEventType::Extraction,
                memory
                    .namespace
                    .context_scope
                    .as_deref()
                    .unwrap_or(&memory.namespace.user_scope),
                extraction.stored,
            ));
    }
    let _automatic = memory
        .system
        .schedule_automatic_extraction(
            &memory.extraction_messages,
            &response_content,
            &memory.namespace,
            request_id,
        )
        .await;
    let suffix = if !memory.effective.show_feedback || requests_structured_output(request) {
        None
    } else {
        format_feedback_suffix(
            memory.injection.memories_injected,
            extraction.stored,
            extraction.sensitive_rejected,
            is_thread_start,
        )
    };
    (suffix, extraction)
}

fn memory_feedback_chunk(request: &OpenAIRequest, suffix: &str) -> serde_json::Value {
    serde_json::json!({
        "id": format!("chatcmpl-memory-{}", uuid::Uuid::new_v4()),
        "object": "chat.completion.chunk",
        "created": chrono::Utc::now().timestamp(),
        "model": request.model,
        "choices": [{
            "index": 0,
            "delta": { "content": suffix },
            "finish_reason": null
        }]
    })
}

use super::AppState;

/// Finalize a guardrail-produced [`GatewayError`] into an HTTP response,
/// completing the request-metrics guard, writing the request log entry, and
/// attaching the trace-id header — mirroring the error path in
/// `chat_completions_non_stream` so guardrail rejections are observed
/// identically to provider failures.
fn guardrail_error_response(
    state: &AppState,
    request: &OpenAIRequest,
    request_guard: &mut RequestCompleteGuard,
    trace_id: &str,
    error: GatewayError,
) -> Response {
    let duration_ms = request_guard.complete();
    let log_context =
        RequestLogContext::from_error(request, trace_id.to_string(), duration_ms, &error);
    log_request(state, request, &log_context);
    let mut response = error.into_response();
    attach_trace_id_header(&mut response, trace_id);
    response
}

// ---------------------------------------------------------------------------
// Error → HTTP response mapping
// ---------------------------------------------------------------------------

impl IntoResponse for GatewayError {
    fn into_response(self) -> Response {
        let (status, body) = match &self {
            GatewayError::InvalidRequest(msg) => (
                StatusCode::BAD_REQUEST,
                serde_json::json!({ "error": { "message": msg, "type": "invalid_request_error" } }),
            ),
            GatewayError::ContextCapacityExceeded {
                estimated_requirement,
                largest_supported_context,
            } => (
                StatusCode::PAYLOAD_TOO_LARGE,
                serde_json::json!({
                    "error": {
                        "message": "No configured model can safely accommodate the request context",
                        "type": "context_capacity_exceeded",
                        "estimated_requirement": estimated_requirement,
                        "largest_supported_context": largest_supported_context,
                    }
                }),
            ),
            GatewayError::SmartRoutingBudgetExceeded { period } => (
                StatusCode::TOO_MANY_REQUESTS,
                serde_json::json!({
                    "error": {
                        "message": "Smart-routing budget exhausted",
                        "type": "smart_routing_budget_exceeded",
                        "period": period,
                    }
                }),
            ),
            GatewayError::Authentication(msg) => (
                StatusCode::UNAUTHORIZED,
                serde_json::json!({ "error": { "message": msg, "type": "authentication_error" } }),
            ),
            GatewayError::AllProvidersFailed(agg) => (
                StatusCode::BAD_GATEWAY,
                serde_json::json!({
                    "error": {
                        "message": "All providers failed to process the request",
                        "type": "all_providers_failed",
                        "attempts": agg.attempts,
                    }
                }),
            ),
            GatewayError::RateLimitExceeded(provider) => (
                StatusCode::TOO_MANY_REQUESTS,
                serde_json::json!({ "error": { "message": format!("Rate limit exceeded for provider: {}", provider), "type": "rate_limit_error" } }),
            ),
            GatewayError::TtfbTimeout(secs) => (
                StatusCode::GATEWAY_TIMEOUT,
                serde_json::json!({ "error": { "message": format!("Provider did not respond within {}s (time-to-first-byte timeout). The model may need more time to start generating — consider increasing ttfb_timeout_seconds.", secs), "type": "ttfb_timeout_error" } }),
            ),
            GatewayError::TotalTimeout(secs) => (
                StatusCode::GATEWAY_TIMEOUT,
                serde_json::json!({ "error": { "message": format!("Request exceeded {}s total round-trip timeout. The response may be too large or the model too slow — consider increasing total_timeout_seconds.", secs), "type": "total_timeout_error" } }),
            ),
            GatewayError::Provider {
                provider: _,
                message: _,
                status_code,
            } => {
                let sc = status_code
                    .and_then(|c| StatusCode::from_u16(c).ok())
                    .unwrap_or(StatusCode::BAD_GATEWAY);
                (
                    sc,
                    serde_json::json!({ "error": { "message": self.to_string(), "type": "provider_error" } }),
                )
            }
            // Guardrail policy block (pre- or post-call): 403 with the
            // triggering category only, never the raw content (Req 2.2, 3.1).
            GatewayError::GuardrailPolicyViolation { category } => (
                StatusCode::FORBIDDEN,
                serde_json::json!({ "error": { "message": "Request blocked by guardrail policy", "type": "guardrail_policy_violation", "category": category } }),
            ),
            // Stage action invalid for its phase (Req 2.7).
            GatewayError::GuardrailInvalidAction => (
                StatusCode::BAD_REQUEST,
                serde_json::json!({ "error": { "message": "Invalid guardrail stage action", "type": "invalid_request_error" } }),
            ),
            // fail_close provider timeout/error (Req 2.9, 9.7).
            GatewayError::GuardrailUnavailable(msg) => (
                StatusCode::SERVICE_UNAVAILABLE,
                serde_json::json!({ "error": { "message": msg, "type": "guardrail_unavailable" } }),
            ),
            _ => (
                StatusCode::INTERNAL_SERVER_ERROR,
                serde_json::json!({ "error": { "message": self.to_string(), "type": "server_error" } }),
            ),
        };

        let mut response = (status, Json(body)).into_response();
        if let GatewayError::SmartRoutingBudgetExceeded { period } = self {
            if let Some(period) = ascii_bounded_header_value(&period) {
                response
                    .headers_mut()
                    .insert(HeaderName::from_static(BUDGET_EXCEEDED_HEADER), period);
            }
        }
        response
    }
}

/// Drop guard that ensures `metrics.complete_request()` is called even when an
/// SSE stream is cancelled (client disconnect, timeout, connection reset).
/// Without this, `active_requests` increments on `start_request()` but never
/// decrements when the stream generator is dropped mid-flight.
struct RequestCompleteGuard {
    metrics: Arc<Metrics>,
    start: std::time::Instant,
    completed: bool,
    /// Optional in-flight registry handle so the request is removed when the
    /// guard is dropped (including mid-stream cancellations).
    active: Option<(Arc<crate::active_requests::ActiveRequestRegistry>, String)>,
}

impl RequestCompleteGuard {
    fn new(
        metrics: Arc<Metrics>,
        start: std::time::Instant,
        active: Option<(Arc<crate::active_requests::ActiveRequestRegistry>, String)>,
    ) -> Self {
        Self {
            metrics,
            start,
            completed: false,
            active,
        }
    }

    /// Mark the request as completed normally (prevents the Drop impl from
    /// double-decrementing) and return the measured duration.
    fn complete(&mut self) -> u64 {
        let duration_ms = self.start.elapsed().as_millis() as u64;
        if !self.completed {
            self.completed = true;
            self.metrics.complete_request(duration_ms);
        }
        duration_ms
    }
}

impl Drop for RequestCompleteGuard {
    fn drop(&mut self) {
        if !self.completed {
            self.completed = true;
            let duration_ms = self.start.elapsed().as_millis() as u64;
            tracing::debug!(
                duration_ms,
                source = "drop",
                "RequestCompleteGuard dropped without explicit complete — completing request metrics via Drop (this is expected for cancelled streams)"
            );
            self.metrics.complete_request(duration_ms);
        }
        // Remove the entry from the in-flight registry if one was registered.
        if let Some((registry, trace_id)) = &self.active {
            registry.deregister(trace_id);
        }
    }
}

/// Build the initial `ActiveRequestInfo` for a freshly-started request so it can
/// be tracked in the dashboard's in-flight registry.
fn build_active_request_info(
    trace_id: &str,
    requested_model: &str,
    virtual_key_id: Option<&str>,
    kind: crate::active_requests::RequestKind,
) -> crate::active_requests::ActiveRequestInfo {
    let started_at_ms = std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .map(|d| d.as_millis() as i64)
        .unwrap_or(0);
    crate::active_requests::ActiveRequestInfo {
        trace_id: trace_id.to_string(),
        requested_model: requested_model.to_string(),
        model_group: None,
        provider: None,
        model: None,
        attempt: 0,
        phase: crate::active_requests::ActivePhase::Pending,
        last_error: None,
        virtual_key_id: virtual_key_id.map(|s| s.to_string()),
        started_at_ms,
        kind,
    }
}

// ---------------------------------------------------------------------------
// GET /health  (Req 20.1-20.3)
// ---------------------------------------------------------------------------

/// Health check endpoint — returns 200 when operational, 503 when shutting down.
pub async fn health_check(State(state): State<AppState>) -> Response {
    if state.shutting_down.load(Ordering::Relaxed) {
        (
            StatusCode::SERVICE_UNAVAILABLE,
            Json(serde_json::json!({ "status": "shutting_down" })),
        )
            .into_response()
    } else {
        (StatusCode::OK, Json(serde_json::json!({ "status": "ok" }))).into_response()
    }
}

// ---------------------------------------------------------------------------
// POST /v1/chat/completions  (Req 2.1)
// ---------------------------------------------------------------------------

/// Chat completions handler — streaming and non-streaming.
///
/// The optional `Extension<AuthenticatedKey>` is inserted by the virtual-key
/// enforcement middleware (`virtual_keys::auth`) when key enforcement is active.
/// Its id is threaded into the guardrail `BindingSelector` so per-virtual-key
/// guardrail bindings resolve (Req 1.3, 1.7). When enforcement is disabled the
/// extension is absent and the id is `None`.
pub async fn chat_completions(
    State(state): State<AppState>,
    headers: HeaderMap,
    authenticated_key: Option<Extension<AuthenticatedKey>>,
    Json(request): Json<OpenAIRequest>,
) -> Response {
    tracing::info!(model = %request.model, stream = request.stream, "Received chat completion request");
    let trace_id = trace_id_from_headers(&headers);
    let virtual_key_id = authenticated_key.map(|Extension(key)| key.id);
    if request.stream {
        chat_completions_stream(state, request, trace_id, virtual_key_id).await
    } else {
        chat_completions_non_stream(state, request, trace_id, virtual_key_id).await
    }
}

async fn chat_completions_non_stream(
    state: AppState,
    mut request: OpenAIRequest,
    trace_id: String,
    virtual_key_id: Option<String>,
) -> Response {
    state.metrics.start_request();
    let start = std::time::Instant::now();
    let active_handle = state.active_requests.register(build_active_request_info(
        &trace_id,
        &request.model,
        virtual_key_id.as_deref(),
        crate::active_requests::RequestKind::Chat,
    ));
    let mut request_guard = RequestCompleteGuard::new(
        state.metrics.clone(),
        start,
        Some((state.active_requests.clone(), trace_id.clone())),
    );
    tracing::debug!(model = %request.model, "Routing non-stream request");

    // Snapshot request-scoped engines once and release both hot-reload locks
    // before cache lookup, provider routing, validation, or corrective retries.
    let guardrail_engine = state.guardrail_engine.read().await.clone();
    let structured_output_engine = state.structured_output_engine.read().await.clone();

    // Tier-1: exact-match in-memory cache.  Lookup is always safe — eligibility
    // (deterministic temperature, n=1) is enforced internally.  Tool-using
    // requests ARE looked up here; only writes are gated below by
    // `should_cache_response`.
    if let Some(cached_json) = state.exact_cache.get(&request) {
        if let Ok(resp) =
            serde_json::from_str::<crate::models::openai::OpenAIResponse>(&cached_json)
        {
            state.metrics.record_cache_hit();
            request_guard.complete();
            let mut http = openai_json_response(&resp);
            attach_trace_id_header(&mut http, &trace_id);
            return http;
        }
    } else if state.exact_cache.is_eligible(&request) {
        state.metrics.record_cache_miss();
    }

    // Tier-2: semantic cache (paraphrase match).  Skipped for tool-using
    // requests — semantic similarity across different tool surfaces is too
    // risky for code agents.
    let skip_semantic =
        request.extra.contains_key("tools") || request.extra.contains_key("tool_choice");
    if !skip_semantic {
        if let Some(ref cache) = state.cache {
            match cache.get(&request).await {
                Ok(Some(cached_response)) => {
                    state.metrics.record_cache_hit();
                    match serde_json::from_str::<crate::models::openai::OpenAIResponse>(
                        &cached_response,
                    ) {
                        Ok(resp) => {
                            request_guard.complete();
                            let mut response = openai_json_response(&resp);
                            attach_trace_id_header(&mut response, &trace_id);
                            return response;
                        }
                        Err(_) => {
                            tracing::warn!(
                                "Failed to parse cached response, falling through to provider"
                            );
                        }
                    }
                }
                Ok(None) => {
                    state.metrics.record_cache_miss();
                }
                Err(e) => {
                    tracing::warn!("Cache lookup failed: {}, falling through to provider", e);
                    state.metrics.record_cache_miss();
                }
            }
        }
    }

    // -----------------------------------------------------------------
    // Guardrail hooks (opt-in). NOTE: cache hits above are served WITHOUT
    // guardrail evaluation. This matches the design's "pre-call before
    // route_request" placement and keeps this task scoped; a cached reply is a
    // verbatim replay of a response that already passed guardrails when first
    // produced (post-call runs before caching below), so redact/replace effects
    // are preserved in the cached payload. A pre-call `block` can still be
    // bypassed by an exact/semantic cache hit — a known limitation to revisit if
    // guardrails must gate cached replays.
    // -----------------------------------------------------------------
    // A request-scoped context carries the PII Re_Injection_Map from pre-call
    // redaction into post-call re-injection (Req 9.5); it is dropped when this
    // function returns (Req 2.6 / 4.6).
    let mut guardrail_ctx = guardrail_engine
        .as_ref()
        .map(|e| e.new_context())
        .unwrap_or_default();
    // Bindings resolve from the authenticated virtual-key id (when key
    // enforcement is active), the requested model group, and the route path
    // (Req 1.3, 1.7).
    let selector = BindingSelector::new(
        virtual_key_id.clone(),
        Some(request.model.clone()),
        Some("/v1/chat/completions".to_string()),
    );

    if let Some(engine) = guardrail_engine.as_ref() {
        match engine
            .run_pre_call(&mut request, &selector, &mut guardrail_ctx, &trace_id)
            .await
        {
            // Forward the (possibly redacted/masked) request to the router.
            PreCallOutcome::Proceed => {}
            // `block` fired → HTTP 403, do NOT route (Req 2.2).
            PreCallOutcome::Block(block) => {
                let err = GatewayError::GuardrailPolicyViolation {
                    category: block.entity_label,
                };
                return guardrail_error_response(
                    &state,
                    &request,
                    &mut request_guard,
                    &trace_id,
                    err,
                );
            }
            // Action invalid for the pre-call phase → HTTP 400 (Req 2.7).
            PreCallOutcome::InvalidAction => {
                return guardrail_error_response(
                    &state,
                    &request,
                    &mut request_guard,
                    &trace_id,
                    GatewayError::GuardrailInvalidAction,
                );
            }
            // fail_close scan timeout → HTTP 503 (Req 2.9).
            PreCallOutcome::Timeout => {
                let err = GatewayError::GuardrailUnavailable("guardrail scan timeout".to_string());
                return guardrail_error_response(
                    &state,
                    &request,
                    &mut request_guard,
                    &trace_id,
                    err,
                );
            }
            // fail_close provider error → HTTP 503 (Req 9.7).
            PreCallOutcome::ServiceFailure => {
                let err =
                    GatewayError::GuardrailUnavailable("guardrail service unavailable".to_string());
                return guardrail_error_response(
                    &state,
                    &request,
                    &mut request_guard,
                    &trace_id,
                    err,
                );
            }
        }
    }

    // Context truncation is centralized here so memory observes the exact
    // post-truncation request. Router compression remains provider-specific and
    // therefore runs only after this handler-side injection boundary.
    request = state.router.prepare_post_truncation_request(&request);
    let memory_context =
        preprocess_memory_request(&state, &mut request, virtual_key_id.as_deref()).await;

    match state
        .router
        .route_request(&request, Some(active_handle.clone()))
        .await
    {
        Ok(mut response) => {
            // Post-call guardrails run on the freshly routed response BEFORE it
            // is cached or returned, using the SAME ctx so PII re-injection
            // works (Req 9.5). Caching the post-guardrail response means later
            // cache replays already reflect redact/replace/re-injection.
            if let Some(engine) = guardrail_engine.as_ref() {
                // Build ToolContext from request/response for refusal detection (Req 12.3).
                let tool_ctx = ToolContext {
                    tool_use_allowed: request
                        .extra
                        .get("tool_choice")
                        .and_then(|v| v.as_str())
                        .map_or(true, |tc| tc != "none"),
                    tools_provided: request
                        .extra
                        .get("tools")
                        .and_then(|v| v.as_array())
                        .map_or(false, |t| !t.is_empty()),
                    finish_reason_is_tool_call: response
                        .choices
                        .first()
                        .and_then(|c| c.finish_reason.as_deref())
                        .map_or(false, |r| r == "tool_calls"),
                    has_tool_calls: response.choices.first().map_or(false, |c| {
                        c.message
                            .extra
                            .get("tool_calls")
                            .and_then(|v| v.as_array())
                            .map_or(false, |a| !a.is_empty())
                    }),
                };
                match engine
                    .run_post_call(
                        &mut response,
                        &selector,
                        &mut guardrail_ctx,
                        &trace_id,
                        &tool_ctx,
                    )
                    .await
                {
                    // Proceed without refusal: re-injection already ran inside
                    // run_post_call. Replaced also returns as-is (Req 9.4: a
                    // halting action skips re-injection and is final regardless
                    // of refusal detection).
                    (PostCallOutcome::Proceed, RefusalDecision::NotRefusal)
                    | (PostCallOutcome::Replaced, _) => {}
                    // Refusal detected with failover enabled (Req 12.5): run the
                    // bounded re-dispatch loop. Re-injection was skipped inside
                    // run_post_call so we do it exactly once on the final response.
                    (PostCallOutcome::Proceed, RefusalDecision::Refusal(_signal)) => {
                        // Compute the fallback ordering once (Req 12.5, 12.7).
                        let model_group = match state.router.find_model_group(&request.model).await
                        {
                            Ok(mg) => mg,
                            Err(_) => {
                                // Cannot resolve model group — re-inject on current response and return.
                                engine.reinject_response(&mut response, &guardrail_ctx);
                                // fall through to caching/return below
                                let cacheable = cache_allowed_for_validation(
                                    &request,
                                    ValidationResponseStatus::NotApplicable,
                                )
                                    && crate::router::router::Router::should_cache_response(
                                        &response,
                                    );
                                if cacheable {
                                    let response_json =
                                        serde_json::to_string(&response).unwrap_or_default();
                                    if !response_json.is_empty() {
                                        state.exact_cache.set(&request, response_json.clone());
                                    }
                                    if !skip_semantic {
                                        if let Some(ref cache) = state.cache {
                                            if let Err(e) =
                                                cache.set(&request, &response_json, 0.0).await
                                            {
                                                tracing::warn!("Failed to cache response: {}", e);
                                            }
                                        }
                                    }
                                }
                                let duration_ms = request_guard.complete();
                                let log_context = RequestLogContext::from_response(
                                    &request,
                                    trace_id.clone(),
                                    duration_ms,
                                    &response,
                                )
                                .with_memory(
                                    memory_context
                                        .as_ref()
                                        .map(|memory| (&memory.context, &memory.injection)),
                                    None,
                                );
                                log_request(&state, &request, &log_context);
                                let mut http_response = openai_json_response(&response);
                                attach_trace_id_header(&mut http_response, &trace_id);
                                return http_response;
                            }
                        };
                        let fallback_order = state.router.select_provider_order(&model_group).await;

                        // Identify the provider that produced the current (refused) response.
                        let already_tried_provider = response
                            .extra
                            .get("gateway_provider")
                            .and_then(|v| v.as_str())
                            .unwrap_or("")
                            .to_string();

                        let mut tried: Vec<String> = vec![already_tried_provider.clone()];
                        let mut last_response = response;
                        // Track whether the final response was a halting action
                        // (replaced/block) that skips re-injection (Req 9.4).
                        let mut skip_reinjection = false;

                        // Bounded re-dispatch loop (Req 12.7): attempt each
                        // target at most once, bounded by ordering length.
                        for target in &fallback_order {
                            // Skip already-tried targets (Req 12.7).
                            let target_key = format!("{}:{}", target.provider, target.model);
                            if tried.contains(&target.provider) {
                                continue;
                            }
                            // Skip targets whose circuit breaker is open (Req 12.10).
                            let cb = state.router.get_circuit_breaker(&target_key).await;
                            if !cb.is_available().await {
                                tracing::debug!(
                                    provider = %target.provider,
                                    model = %target.model,
                                    "Refusal failover: circuit breaker open, skipping"
                                );
                                continue;
                            }

                            tried.push(target.provider.clone());

                            // Re-dispatch the already-redacted request to this
                            // single target via route_with_failover (Req 12.5).
                            let attempt_result = state
                                .router
                                .route_with_failover(&request, vec![target.clone()], None)
                                .await;

                            match attempt_result {
                                Ok(mut new_response) => {
                                    // Re-run post-call + refusal detection on the new response.
                                    let new_tool_ctx = ToolContext {
                                        tool_use_allowed: request
                                            .extra
                                            .get("tool_choice")
                                            .and_then(|v| v.as_str())
                                            .map_or(true, |tc| tc != "none"),
                                        tools_provided: request
                                            .extra
                                            .get("tools")
                                            .and_then(|v| v.as_array())
                                            .map_or(false, |t| !t.is_empty()),
                                        finish_reason_is_tool_call: new_response
                                            .choices
                                            .first()
                                            .and_then(|c| c.finish_reason.as_deref())
                                            .map_or(false, |r| r == "tool_calls"),
                                        has_tool_calls: new_response.choices.first().map_or(
                                            false,
                                            |c| {
                                                c.message
                                                    .extra
                                                    .get("tool_calls")
                                                    .and_then(|v| v.as_array())
                                                    .map_or(false, |a| !a.is_empty())
                                            },
                                        ),
                                    };
                                    let (post_outcome, refusal_decision) = engine
                                        .run_post_call(
                                            &mut new_response,
                                            &selector,
                                            &mut guardrail_ctx,
                                            &trace_id,
                                            &new_tool_ctx,
                                        )
                                        .await;

                                    match post_outcome {
                                        // Block/ServiceFailure from post-call on failover target:
                                        // return the error immediately.
                                        PostCallOutcome::Block(block) => {
                                            let err = GatewayError::GuardrailPolicyViolation {
                                                category: block.entity_label,
                                            };
                                            return guardrail_error_response(
                                                &state,
                                                &request,
                                                &mut request_guard,
                                                &trace_id,
                                                err,
                                            );
                                        }
                                        PostCallOutcome::ServiceFailure => {
                                            let err = GatewayError::GuardrailUnavailable(
                                                "guardrail service unavailable".to_string(),
                                            );
                                            return guardrail_error_response(
                                                &state,
                                                &request,
                                                &mut request_guard,
                                                &trace_id,
                                                err,
                                            );
                                        }
                                        // Replaced means we got a policy message; treat as non-refusal.
                                        // Re-injection is skipped for replaced responses (Req 9.4).
                                        PostCallOutcome::Replaced => {
                                            last_response = new_response;
                                            skip_reinjection = true;
                                            break;
                                        }
                                        PostCallOutcome::Proceed => {
                                            if !refusal_decision.is_refusal() {
                                                // First non-refusal: use it (Req 12.5).
                                                // Re-injection runs once below on the final response.
                                                last_response = new_response;
                                                break;
                                            }
                                            // Still a refusal — record and continue loop.
                                            last_response = new_response;
                                        }
                                    }
                                }
                                Err(_e) => {
                                    // Provider error during failover — continue to next target.
                                    tracing::debug!(
                                        provider = %target.provider,
                                        "Refusal failover: provider error, trying next"
                                    );
                                    continue;
                                }
                            }
                        }

                        // PII re-injection runs exactly once on the finally
                        // selected response (Req 9.5, 12.5), unless the response
                        // was replaced by a policy message (Req 9.4).
                        if !skip_reinjection {
                            engine.reinject_response(&mut last_response, &guardrail_ctx);
                        }
                        response = last_response;
                    }
                    // `block` → discard response, HTTP 403 (Req 3.1).
                    (PostCallOutcome::Block(block), _) => {
                        let err = GatewayError::GuardrailPolicyViolation {
                            category: block.entity_label,
                        };
                        return guardrail_error_response(
                            &state,
                            &request,
                            &mut request_guard,
                            &trace_id,
                            err,
                        );
                    }
                    // fail_close provider error/timeout → HTTP 503 (Req 9.7).
                    (PostCallOutcome::ServiceFailure, _) => {
                        let err = GatewayError::GuardrailUnavailable(
                            "guardrail service unavailable".to_string(),
                        );
                        return guardrail_error_response(
                            &state,
                            &request,
                            &mut request_guard,
                            &trace_id,
                            err,
                        );
                    }
                }
            }
            // Structured-output validation runs only after the complete
            // guardrail/refusal pipeline and before either cache write.
            let mut validation_status = ValidationResponseStatus::NotApplicable;
            if let Some(engine) = structured_output_engine.as_ref() {
                let model_group = state.router.find_model_group(&request.model).await.ok();
                if let (Some(model_group), Some((provider, model))) =
                    (model_group.as_ref(), response_provider_model(&response))
                {
                    let decision_started = Instant::now();
                    let validation_decision =
                        engine.should_validate(&request, &model_group.name, provider, model);
                    let decision_latency_ms = decision_started.elapsed().as_secs_f64() * 1_000.0;
                    match validation_decision {
                        ValidationDecision::NotApplicable => {}
                        ValidationDecision::Skipped(reason) => {
                            validation_status = ValidationResponseStatus::Skipped;
                            state.metrics.observe_structured_output_latency(
                                provider,
                                model,
                                decision_latency_ms,
                            );
                            tracing::warn!(
                                trace_id = %trace_id,
                                category = validation_skip_category(&reason),
                                "structured output validation skipped"
                            );
                        }
                        ValidationDecision::Validate(schema_context) => {
                            let initial_provider = provider.to_owned();
                            let initial_model = model.to_owned();
                            let mut gateway_processing_ms = decision_latency_ms;
                            let mut validation = engine
                                .validate_response(
                                    &schema_context,
                                    &response,
                                    &initial_provider,
                                    &initial_model,
                                )
                                .await;
                            gateway_processing_ms += validation.latency_ms;
                            validation_status =
                                ValidationResponseStatus::from_outcome(validation.outcome);

                            if validation.outcome == StructuredOutputOutcome::Fail {
                                let initial_failure = collect_structured_output_failure(
                                    &response,
                                    &validation.choices,
                                );
                                tracing::info!(
                                    trace_id = %trace_id,
                                    provider = %initial_provider,
                                    model = %initial_model,
                                    error_count = initial_failure.violations.len(),
                                    retry_attempt = 0,
                                    category = "validation_failed",
                                    "structured output validation failed"
                                );

                                let original_failed_response = response.clone();
                                let effective_config = engine.effective_config(
                                    &model_group.name,
                                    &initial_provider,
                                    &initial_model,
                                );
                                let target = find_selected_provider_model(
                                    model_group,
                                    &initial_provider,
                                    &initial_model,
                                );

                                if effective_config.max_retries == 0 {
                                    validation_status = ValidationResponseStatus::Failed;
                                } else if let Some(target) = target {
                                    let context_window = state
                                        .router
                                        .context_manager()
                                        .get_capabilities(&target.model)
                                        .map(|capabilities| capabilities.context_window as usize)
                                        .unwrap_or_else(default_context_window);
                                    let mut remaining_attempts =
                                        usize::from(effective_config.max_retries);
                                    let mut last_successful_response = response.clone();
                                    let mut successful_retry_count = 0usize;
                                    let mut provider_error_count = 0usize;

                                    while remaining_attempts > 0 {
                                        let failure = collect_structured_output_failure(
                                            &last_successful_response,
                                            &validation.choices,
                                        );
                                        let request_tokens =
                                            TokenCounter::new().count_request(&request) as usize;
                                        let prompt_started = Instant::now();
                                        let retry_request = engine.build_retry_request(
                                            &request,
                                            &schema_context,
                                            &failure.violations,
                                            &failure.previous_output,
                                            &effective_config,
                                            false,
                                            context_window,
                                            request_tokens,
                                        );
                                        gateway_processing_ms +=
                                            prompt_started.elapsed().as_secs_f64() * 1_000.0;

                                        let retry_result = state
                                            .router
                                            .route_with_failover(
                                                &retry_request,
                                                vec![target.clone()],
                                                None,
                                            )
                                            .await;
                                        match retry_result {
                                            Ok(mut retry_response) => {
                                                successful_retry_count += 1;
                                                remaining_attempts -= 1;

                                                if let Some(guardrail) = guardrail_engine.as_ref() {
                                                    let retry_tool_context =
                                                        tool_context(&request, &retry_response);
                                                    let guardrail_started = Instant::now();
                                                    let guardrail_result = guardrail
                                                        .run_post_call(
                                                            &mut retry_response,
                                                            &selector,
                                                            &mut guardrail_ctx,
                                                            &trace_id,
                                                            &retry_tool_context,
                                                        )
                                                        .await;
                                                    gateway_processing_ms +=
                                                        guardrail_started.elapsed().as_secs_f64()
                                                            * 1_000.0;
                                                    match guardrail_result {
                                                        (PostCallOutcome::Block(block), _) => {
                                                            return guardrail_error_response(
                                                                &state,
                                                                &request,
                                                                &mut request_guard,
                                                                &trace_id,
                                                                GatewayError::GuardrailPolicyViolation {
                                                                    category: block.entity_label,
                                                                },
                                                            );
                                                        }
                                                        (PostCallOutcome::ServiceFailure, _) => {
                                                            return guardrail_error_response(
                                                                &state,
                                                                &request,
                                                                &mut request_guard,
                                                                &trace_id,
                                                                GatewayError::GuardrailUnavailable(
                                                                    "guardrail service unavailable"
                                                                        .to_owned(),
                                                                ),
                                                            );
                                                        }
                                                        (
                                                            PostCallOutcome::Proceed,
                                                            RefusalDecision::Refusal(_),
                                                        ) => {
                                                            last_successful_response =
                                                                retry_response;
                                                            break;
                                                        }
                                                        _ => {}
                                                    }
                                                }

                                                let (retry_provider, retry_model) =
                                                    response_provider_model(&retry_response)
                                                        .map(|(provider, model)| {
                                                            (provider.to_owned(), model.to_owned())
                                                        })
                                                        .unwrap_or_else(|| {
                                                            (
                                                                target.provider.clone(),
                                                                target.model.clone(),
                                                            )
                                                        });
                                                validation = engine
                                                    .validate_response(
                                                        &schema_context,
                                                        &retry_response,
                                                        &retry_provider,
                                                        &retry_model,
                                                    )
                                                    .await;
                                                gateway_processing_ms += validation.latency_ms;
                                                validation_status =
                                                    ValidationResponseStatus::from_outcome(
                                                        validation.outcome,
                                                    );
                                                last_successful_response = retry_response;

                                                if validation.outcome
                                                    == StructuredOutputOutcome::Fail
                                                {
                                                    let failure = collect_structured_output_failure(
                                                        &last_successful_response,
                                                        &validation.choices,
                                                    );
                                                    tracing::info!(
                                                        trace_id = %trace_id,
                                                        provider = %retry_provider,
                                                        model = %retry_model,
                                                        error_count = failure.violations.len(),
                                                        retry_attempt = successful_retry_count,
                                                        category = "validation_failed",
                                                        "structured output validation failed"
                                                    );
                                                }

                                                if validation.outcome
                                                    == StructuredOutputOutcome::Pass
                                                {
                                                    response = last_successful_response.clone();
                                                    state.metrics.record_structured_output_retry(
                                                        &initial_provider,
                                                        &initial_model,
                                                        "recovered",
                                                    );
                                                    break;
                                                }
                                                if validation.outcome
                                                    == StructuredOutputOutcome::Skipped
                                                {
                                                    response = last_successful_response.clone();
                                                    validation_status =
                                                        ValidationResponseStatus::Skipped;
                                                    tracing::warn!(
                                                        trace_id = %trace_id,
                                                        category = "validation_internal_skip",
                                                        "structured output validation skipped"
                                                    );
                                                    break;
                                                }
                                            }
                                            Err(error) => {
                                                let consumed =
                                                    consume_provider_error_attempts(&error)
                                                        .min(remaining_attempts);
                                                provider_error_count += consumed;
                                                remaining_attempts -= consumed;
                                                tracing::warn!(
                                                    trace_id = %trace_id,
                                                    category = "provider_error",
                                                    "structured output retry attempt failed"
                                                );
                                            }
                                        }
                                    }

                                    if validation_status == ValidationResponseStatus::Failed {
                                        response = last_successful_response.clone();
                                        if successful_retry_count + provider_error_count > 0 {
                                            state.metrics.record_structured_output_retry(
                                                &initial_provider,
                                                &initial_model,
                                                "exhausted",
                                            );
                                        }
                                        tracing::warn!(
                                            trace_id = %trace_id,
                                            category = "retry_exhausted",
                                            successful_retry_count,
                                            provider_error_count,
                                            "structured output retry exhausted"
                                        );
                                    }
                                } else {
                                    validation_status = ValidationResponseStatus::Failed;
                                    response = original_failed_response;
                                    tracing::warn!(
                                        trace_id = %trace_id,
                                        category = "retry_dispatch_setup",
                                        "structured output retry setup failed"
                                    );
                                }
                            } else if validation.outcome == StructuredOutputOutcome::Skipped {
                                tracing::warn!(
                                    trace_id = %trace_id,
                                    category = "validation_internal_skip",
                                    "structured output validation skipped"
                                );
                            }

                            state.metrics.observe_structured_output_latency(
                                &initial_provider,
                                &initial_model,
                                gateway_processing_ms,
                            );
                        }
                    }
                } else if requests_json_schema(&request) {
                    validation_status = ValidationResponseStatus::Skipped;
                    tracing::warn!(
                        trace_id = %trace_id,
                        category = "routing_metadata",
                        "structured output validation skipped"
                    );
                }
            }

            // Cache responses that are safe to replay. Structured-output skips
            // and failures are never written to either cache tier.
            let cacheable = cache_allowed_for_validation(&request, validation_status)
                && crate::router::router::Router::should_cache_response(&response);
            if cacheable {
                let response_json = serde_json::to_string(&response).unwrap_or_default();
                if !response_json.is_empty() {
                    state.exact_cache.set(&request, response_json.clone());
                }
                if !skip_semantic {
                    if let Some(ref cache) = state.cache {
                        if let Err(e) = cache.set(&request, &response_json, 0.0).await {
                            tracing::warn!("Failed to cache response: {}", e);
                        }
                    }
                }
            }
            let request_uuid = uuid::Uuid::parse_str(&trace_id).ok();
            let is_thread_start = request.messages.iter().filter(|m| m.role == "user").count() <= 1;
            let (memory_suffix, memory_extraction) = finalize_memory_response(
                &state,
                &request,
                &mut response,
                memory_context.as_ref(),
                request_uuid,
                is_thread_start,
            )
            .await;
            if let Some(suffix) = memory_suffix {
                append_feedback_to_response(&mut response, &suffix);
            }
            let duration_ms = request_guard.complete();
            let log_context = RequestLogContext::from_response(
                &request,
                trace_id.clone(),
                duration_ms,
                &response,
            )
            .with_memory(
                memory_context
                    .as_ref()
                    .map(|memory| (&memory.context, &memory.injection)),
                Some(memory_extraction),
            );
            log_request(&state, &request, &log_context);
            let mut http_response = openai_json_response(&response);
            attach_validation_status_header(
                &mut http_response,
                validation_status,
                requests_json_schema(&request),
            );
            attach_trace_id_header(&mut http_response, &trace_id);
            http_response
        }
        Err(e) => {
            let duration_ms = request_guard.complete();
            let log_context =
                RequestLogContext::from_error(&request, trace_id.clone(), duration_ms, &e);
            log_request(&state, &request, &log_context);
            let mut response = e.into_response();
            attach_trace_id_header(&mut response, &trace_id);
            response
        }
    }
}

async fn chat_completions_stream(
    state: AppState,
    mut request: OpenAIRequest,
    trace_id: String,
    virtual_key_id: Option<String>,
) -> Response {
    state.metrics.start_request();
    let start = Instant::now();
    let active_handle = state.active_requests.register(build_active_request_info(
        &trace_id,
        &request.model,
        virtual_key_id.as_deref(),
        crate::active_requests::RequestKind::Stream,
    ));
    let mut request_guard = RequestCompleteGuard::new(
        state.metrics.clone(),
        start,
        Some((state.active_requests.clone(), trace_id.clone())),
    );
    tracing::debug!(
        trace_id = %trace_id,
        model = %request.model,
        "Client requested streaming response; gateway currently buffers the full upstream response before synthesizing SSE"
    );

    request = state.router.prepare_post_truncation_request(&request);
    let memory_context =
        preprocess_memory_request(&state, &mut request, virtual_key_id.as_deref()).await;

    let memory_extraction = if let Some(memory) = memory_context.as_ref() {
        match memory
            .system
            .extract_explicit_response(
                &memory.extraction_messages,
                &memory.namespace,
                uuid::Uuid::parse_str(&trace_id).ok(),
            )
            .await
        {
            Ok(extraction) => Some(extraction),
            Err(error) => {
                tracing::warn!(error = %error, "Streaming memory explicit extraction failed");
                None
            }
        }
    } else {
        None
    };
    let memory_suffix = memory_context.as_ref().and_then(|memory| {
        if !memory.effective.show_feedback || requests_structured_output(&request) {
            return None;
        }
        let extraction = memory_extraction.unwrap_or_default();
        let is_thread_start = request.messages.iter().filter(|m| m.role == "user").count() <= 1;
        format_feedback_suffix(
            memory.injection.memories_injected,
            extraction.stored,
            extraction.sensitive_rejected,
            is_thread_start,
        )
    });

    // Streaming Reliability (Req 2): resolve the effective streaming settings
    // up front so every SSE path below (cache replay, early event, and the
    // buffer-and-replay fallback) can apply the configured keep-alive interval.
    // An absent `streaming` section falls back to defaults.
    let streaming_config = state
        .config
        .read()
        .await
        .streaming
        .clone()
        .unwrap_or_default();

    // Snapshot request-scoped engines once before cache lookup, pre-call
    // processing, routing, or validation. The hot-reload locks are never held
    // across an await below.
    let guardrail_engine = state.guardrail_engine.read().await.clone();
    let structured_output_engine = state.structured_output_engine.read().await.clone();

    // Tier-1 cache lookup for streaming requests. The cached payload is a
    // full non-streaming `OpenAIResponse` JSON; we re-emit it as SSE chunks
    // using the same path as a fresh provider response.  This means a single
    // cached entry serves both stream and non-stream callers identically.
    if let Some(cached_json) = state.exact_cache.get(&request) {
        if let Ok(cached_resp) = serde_json::from_str::<OpenAIResponse>(&cached_json) {
            state.metrics.record_cache_hit();
            request_guard.complete();
            let stream_trace_id = trace_id.clone();

            let stream = async_stream::stream! {
                tracing::debug!(trace_id = %stream_trace_id, "Streaming cached response from exact cache");
                for chunk in streaming_chunks_from_response(&cached_resp) {
                    yield Ok::<_, Infallible>(Event::default().data(chunk.to_string()));
                }
                yield Ok(Event::default().data("[DONE]"));
            };
            let mut sse = Sse::new(stream)
                .keep_alive(build_keepalive(&streaming_config))
                .into_response();
            attach_trace_id_header(&mut sse, &trace_id);
            return sse;
        }
    } else if state.exact_cache.is_eligible(&request) {
        state.metrics.record_cache_miss();
    }

    // -----------------------------------------------------------------
    // Guardrail hooks (opt-in) for streaming (Req 10). Run pre-call before
    // routing; any bound post-call stage later forces full-response buffering.
    // Cache hits above intentionally bypass both guardrails and validation.
    // -----------------------------------------------------------------
    let mut guardrail_ctx = guardrail_engine
        .as_ref()
        .map(|e| e.new_context())
        .unwrap_or_default();
    let selector = BindingSelector::new(
        virtual_key_id,
        Some(request.model.clone()),
        Some("/v1/chat/completions".to_string()),
    );
    if let Some(engine) = guardrail_engine.as_ref() {
        // Request-scoped context carrying the PII Re_Injection_Map from pre-call
        // redaction into post-call re-injection (Req 9.5).

        // Pre-call runs before routing. Nothing has streamed yet, so on a
        // terminal pre-call outcome we return a plain JSON error response
        // identical to the non-stream handler.
        match engine
            .run_pre_call(&mut request, &selector, &mut guardrail_ctx, &trace_id)
            .await
        {
            PreCallOutcome::Proceed => {}
            PreCallOutcome::Block(block) => {
                let err = GatewayError::GuardrailPolicyViolation {
                    category: block.entity_label,
                };
                return guardrail_error_response(
                    &state,
                    &request,
                    &mut request_guard,
                    &trace_id,
                    err,
                );
            }
            PreCallOutcome::InvalidAction => {
                return guardrail_error_response(
                    &state,
                    &request,
                    &mut request_guard,
                    &trace_id,
                    GatewayError::GuardrailInvalidAction,
                );
            }
            PreCallOutcome::Timeout => {
                let err = GatewayError::GuardrailUnavailable("guardrail scan timeout".to_string());
                return guardrail_error_response(
                    &state,
                    &request,
                    &mut request_guard,
                    &trace_id,
                    err,
                );
            }
            PreCallOutcome::ServiceFailure => {
                let err =
                    GatewayError::GuardrailUnavailable("guardrail service unavailable".to_string());
                return guardrail_error_response(
                    &state,
                    &request,
                    &mut request_guard,
                    &trace_id,
                    err,
                );
            }
        }

        // Force the buffered path when either:
        //   (a) a post-call pipeline is bound (at least one resolved stage runs
        //       in the post-call phase for this selector), so the assembled
        //       response can be analyzed before any bytes reach the caller; OR
        //   (b) pre-call redaction populated the Re_Injection_Map — even with no
        //       post-call stage, the assembled response must be buffered so PII
        //       placeholders are re-injected as the final post-call step (Req
        //       9.5). Without this, a streaming response would leak raw
        //       `<<PII_..>>` placeholders instead of the restored originals.
        // `run_post_call` handles the no-post-call-stage case: it runs no stages
        // and performs re-injection when the context is non-empty.
        let post_call_bound = engine
            .resolver()
            .resolve(&selector)
            .iter()
            .any(|s| s.phase == StagePhase::PostCall);
        let needs_buffering = post_call_bound || !guardrail_ctx.is_empty();
        if needs_buffering && !force_eager_structured_stream(&request) {
            return stream_buffered_with_post_call(
                state,
                request,
                trace_id,
                start,
                streaming_config,
                engine.clone(),
                selector,
                guardrail_ctx,
                request_guard,
                Some(active_handle.clone()),
            )
            .await;
        }
        // No post-call pipeline and no pending re-injection: fall through to the
        // normal SSE path unless structured output below forces eager buffering.
    }

    // A streaming JSON-schema request is always resolved eagerly. This occurs
    // before synthetic early events or construction of a lazy SSE body, so the
    // validation status header is final before the first response byte.
    if force_eager_structured_stream(&request) {
        return stream_eager_structured_output(
            state,
            request,
            trace_id,
            streaming_config,
            guardrail_engine,
            structured_output_engine,
            selector,
            guardrail_ctx,
            request_guard,
            Some(active_handle.clone()),
        )
        .await;
    }

    // Req 1: resolve the effective streaming settings — done above so all SSE
    // paths share it.

    // Req 1.1/1.2/1.4/1.6: when enabled (cache hits are handled above and skip
    // this path), emit a synthetic `role: assistant` event BEFORE the provider
    // responds so the client's idle timer resets within 500ms. route_request()
    // therefore runs INSIDE the stream, after the early event is flushed.
    if streaming_config.emit_early_event {
        // Pre-generate a stable id + timestamp so the early event and every
        // subsequent chunk can share them (Req 1.3; threaded into downstream
        // chunks by task 2.2).
        let response_id = format!("chatcmpl-{}", uuid::Uuid::new_v4());
        let created = chrono::Utc::now().timestamp();
        let requested_model = request.model.clone();
        let early_chunk = early_event_chunk(&response_id, created, &requested_model);
        let stream_trace_id = trace_id.clone();
        let active_handle_stream = active_handle.clone();

        let streaming_config_relay = streaming_config.clone();
        let stream = async_stream::stream! {
                    // Drop guard: ensures active_requests is decremented even if the
                    // client disconnects and the stream is cancelled mid-flight.
                    let mut _guard = request_guard;

                    // Early synthetic event (Req 1.1, 1.2, 1.3).
                    yield Ok::<_, Infallible>(Event::default().data(early_chunk.to_string()));

                    // Task 5.5: dispatch through the streaming router so capable
                    // providers stream in real time (PassThrough) while providers that
                    // need response transformation fall back to buffer-and-replay
                    // (Buffered) — both behind the early event already flushed above.
                    match state.router.route_request_streaming(&request, Some(active_handle_stream.clone())).await {
                        Ok(StreamingResponse::Buffered(response)) => {
                            // Buffer-and-replay: cache the assembled response so a later
                            // identical request replays without hitting the provider.
                            if crate::router::router::Router::should_cache_response(&response) {
                                if let Ok(json) = serde_json::to_string(&response) {
                                    state.exact_cache.set(&request, json);
                                }
                            }

                            let duration_ms = start.elapsed().as_millis() as u64;
                            let log_context = RequestLogContext::from_response(&request, stream_trace_id.clone(), duration_ms, &response);
                            log_request(&state, &request, &log_context);

                            // Req 1.5: continue emitting content chunks after the early
                            // event, reusing its id/created and skipping the duplicate
                            // role delta (task 2.2).
                            for chunk in streaming_chunks_after_early_event(&response, &response_id, created) {
                                yield Ok(Event::default().data(chunk.to_string()));
                            }
                            if let Some(suffix) = memory_suffix.as_deref() {
                                yield Ok(Event::default().data(memory_feedback_chunk(&request, suffix).to_string()));
                            }
                            yield Ok(Event::default().data("[DONE]"));
                        }
                        Ok(StreamingResponse::PassThrough { byte_stream, provider, model, compression, concurrency_permit }) => {
                            // True streaming pass-through (Req 3.1, 3.2). The early event
                            // above already reset the client idle timer; now relay the
                            // upstream chunks verbatim.
                            //
                            // EARLY-EVENT ID TRADEOFF: the relayed chunks carry the
                            // provider's own `id`, which differs from the synthetic
                            // early event's fresh uuid. We do NOT rewrite per-chunk ids
                            // (costly and unnecessary): OpenAI-compatible clients merge
                            // deltas by `choices[].index`, not by envelope `id`, and the
                            // role-only early event is idempotent. So we forward upstream
                            // chunks as-is.
                            //
                            // Task 6.1 — PRE-CONTENT FAILOVER LOOP (Req 4.1, 4.4, 4.5):
                            // relay the current provider; if it fails BEFORE any content
                            // reached the client, record a circuit-breaker failure, add
                            // the provider to the exclusion list, and retry the next
                            // eligible provider — WITHOUT emitting a second early/role
                            // event (the early event was emitted once, above). The loop
                            // is bounded because every retry excludes the failed
                            // provider, so `route_request_streaming_excluding` eventually
                            // returns `Buffered`/`Err` (no eligible pass-through left).
                            //
                    // Task 6.3 — RETRY/FAILOVER LIMITS + AGGREGATED ERROR (Req 4.3):
                    // - Provider ordering: `route_request_streaming_excluding`
                    //   picks from the SAME `select_provider_order()` list as the
                    //   non-streaming path, skipping `tried_providers` (keyed per
                    //   `provider:model`, matching the circuit-breaker key, so a
                    //   provider offering several models stays eligible via its
                    //   other models). That list is the natural bound — each
                    //   provider:model entry is tried for pass-through at most once.
                            // - `max_retries_per_provider` mapping: the non-streaming
                            //   path applies it INSIDE `attempt_with_retry` (inline
                            //   same-provider retries). A live SSE relay cannot be safely
                            //   retried inline once response headers/bytes have arrived,
                            //   so each provider gets exactly ONE pass-through attempt and
                            //   failover advances to the NEXT provider. The buffered
                            //   fallback (`route_request`) still honors
                            //   `max_retries_per_provider` via `attempt_with_retry`.
                            // - Defensive hard cap: even though the exclusion list bounds
                            //   the loop, cap total pass-through attempts at
                            //   (provider count + 1) so a logic error can never spin
                            //   forever.
                            // - Aggregated error: each pre-content failure is recorded as
                            //   a `ProviderAttempt`; if every provider fails they are
                            //   merged into a single `AllProvidersFailed` error.
                let (max_retries_per_provider, max_failover_attempts) = {
                    let cfg = state.config.read().await;
                    // Exclusion is per provider:model, so the natural loop bound is
                    // the number of distinct model entries across all groups.
                    let model_entries: usize = cfg.model_groups.iter().map(|g| g.models.len()).sum();
                    (cfg.retry.max_retries_per_provider, cfg.providers.len() + model_entries + 1)
                };
                            tracing::debug!(
                                trace_id = %stream_trace_id,
                                max_retries_per_provider,
                                max_failover_attempts,
                                "Streaming failover policy: one pass-through attempt per provider; max_retries_per_provider applies to the buffered fallback only"
                            );

                            let mut tried_providers: Vec<String> = Vec::new();
                            // Req 4.3: accumulate each failed pass-through attempt so a
                            // total failure surfaces every provider, not just the last.
                            let mut streaming_attempts: Vec<ProviderAttempt> = Vec::new();
                            let mut failover_attempts: usize = 0;
                                let mut _current_concurrency_permit = Some(concurrency_permit);
                                let mut current_stream = byte_stream;
                            let mut current_provider = provider;
                            let mut current_model = model;
                            let mut current_compression = compression;

                            'failover: loop {
                                // Defensive bound (see note above): unreachable in normal
                                // operation because the exclusion list already bounds the
                                // loop. If ever tripped, emit whatever was accumulated.
                                failover_attempts += 1;
                                if failover_attempts > max_failover_attempts {
                                    tracing::error!(
                                        trace_id = %stream_trace_id,
                                        failover_attempts,
                                        max_failover_attempts,
                                        "Streaming failover exceeded safety cap; aborting with aggregated error"
                                    );
                                    let aggregated = GatewayError::AllProvidersFailed(
                                        AggregatedError::new(std::mem::take(&mut streaming_attempts)),
                                    );
                                    let (error_type, message) = classify_stream_error(&aggregated);
                                    for event in emit_sse_error_event(error_type, &message, &stream_trace_id) {
                                        yield Ok(event);
                                    }
                                    break 'failover;
                                }
                                // Resolve the chosen provider's effective total timeout
                                // for the relay budget (Req 3.11). Short-lived guard,
                                // dropped before relaying — never held across `.await`s.
                                let total_timeout = {
                                    let cfg = state.config.read().await;
                                    let secs = cfg
                                        .providers
                                        .iter()
                                        .find(|p| p.name == current_provider)
                                        .map(|p| p.effective_total_timeout(&current_model))
                                        .unwrap_or(600);
                                    Duration::from_secs(secs)
                                };

                // Shared handle the relay writes its terminal outcome to.
                let outcome = Arc::new(tokio::sync::Mutex::new(RelayOutcome::Completed {
                    usage: Usage::default(),
                }));
                                // Enable adaptive XML-tool detection only when the
                                // request carries `tools` (XML tool use is irrelevant
                                // otherwise). A learned combo would already have been
                                // routed to the buffered path, so here it is always an
                                // as-yet-unlearned combo.
                                let xml_detect = if request.extra.contains_key("tools") {
                                    Some(XmlToolDetect {
                                        router: state.router.clone(),
                                        provider: current_provider.clone(),
                                        model: current_model.clone(),
                                    })
                                } else {
                                    None
                                };
                                let relay = relay_passthrough_stream(
                                    current_stream,
                                    streaming_config_relay.clone(),
                                    stream_trace_id.clone(),
                                    total_timeout,
                                    state.exact_cache.clone(),
                                    state.metrics.clone(),
                                    request.clone(),
                                    outcome.clone(),
                                    xml_detect,
                                    memory_suffix.clone(),
                                );
                                // The relay emits its own terminal `[DONE]` (or a graceful
                                // error event that appends one, or — on pre-content
                                // failure — nothing), so we must NOT emit another here.
                                // `relay_passthrough_stream` returns an `!Unpin` async
                                // stream, so pin it on the stack before polling.
                                futures::pin_mut!(relay);
                                while let Some(ev) = relay.next().await {
                                    yield ev;
                                }
                                drop(relay);

                let final_outcome = { outcome.lock().await.clone() };
                match final_outcome {
                    // Clean finish — relay already emitted `[DONE]`.
                    RelayOutcome::Completed { usage } => {
                        let duration = start.elapsed();
                        // Success accounting for pass-through streams (mirror of the
                        // buffered path): close/record circuit-breaker success, feed
                        // the latency tracker, accrue cost from the relay's
                        // reassembled usage, and clear any upstream-driven cooldown
                        // so a recovered provider is reselected immediately.
                        state
                            .router
                            .record_streaming_success(
                                &current_provider,
                                &current_model,
                                duration,
                                &usage,
                            )
                            .await;
                        let duration_ms = duration.as_millis() as u64;
                        let log_context = RequestLogContext::from_streaming_success(
                            &request,
                            stream_trace_id.clone(),
                            duration_ms,
                            current_provider.clone(),
                            current_model.clone(),
                            current_compression.clone(),
                        );
                        log_request(&state, &request, &log_context);
                        break 'failover;
                    }
                                    // Post-content failure (Req 4.2): the relay already
                                    // emitted the graceful error event + `[DONE]`. We
                                    // cannot transparently fail over mid-content, so
                                    // account the failed attempt against the circuit
                                    // breaker + metrics (Req 4.5) and stop — no retry.
                                    RelayOutcome::FailedAfterContent(reason) => {
                                        let duration_ms = start.elapsed().as_millis() as u64;
                                        let log_context = RequestLogContext::from_streaming_success(
                                            &request,
                                            stream_trace_id.clone(),
                                            duration_ms,
                                            current_provider.clone(),
                                            current_model.clone(),
                                            current_compression.clone(),
                                        );
                                        log_request(&state, &request, &log_context);
                                        state
                                            .router
                                            .record_streaming_failure(
                                                &current_provider,
                                                &current_model,
                                                Some(reason.clone()),
                                            )
                                            .await;
                                        tracing::warn!(
                                            trace_id = %stream_trace_id,
                                            provider = %current_provider,
                                            reason = %reason,
                                            "Streaming provider failed after content was sent; closing stream (no failover)"
                                        );
                                        break 'failover;
                                    }
                                    // Pre-content failure — transparently fail over.
                                    RelayOutcome::FailedBeforeContent(reason) => {
                                        // Req 4.5: account the failed attempt against the
                                        // circuit breaker before retrying.
                                        state
                                            .router
                                            .record_streaming_failure(
                                                &current_provider,
                                                &current_model,
                                                Some(reason.clone()),
                                            )
                                            .await;
                                        tracing::warn!(
                                            trace_id = %stream_trace_id,
                                            provider = %current_provider,
                                            reason = %reason,
                                            "Streaming provider failed before any content; attempting pre-content failover"
                                        );
                                        tried_providers.push(format!("{}:{}", current_provider, current_model));
                                        // Req 4.3: record this pre-content failure for the
                                        // aggregated error in case every provider fails.
                                        streaming_attempts.push(ProviderAttempt::new(
                                            current_provider.clone(),
        current_model.clone(),
        reason.clone(),
        None,
        ));
        drop(_current_concurrency_permit.take());

        match state
                                            .router
                                            .route_request_streaming_excluding(&request, &tried_providers, Some(active_handle.clone()))
                                            .await
                                        {
                                            // Another eligible provider — relay it,
                                            // reusing the SAME early-event id (Req 4.4:
                                            // do NOT emit a second role event).
        Ok(StreamingResponse::PassThrough { byte_stream, provider, model, compression, concurrency_permit }) => {
        _current_concurrency_permit = Some(concurrency_permit);
        current_stream = byte_stream;
                                                current_provider = provider;
                                                current_model = model;
                                                current_compression = compression;
                                                continue 'failover;
                                            }
                                            // No eligible pass-through provider remains —
                                            // replay the buffered fallback after the early
                                            // event, then terminate.
                                            Ok(StreamingResponse::Buffered(response)) => {
                                                if crate::router::router::Router::should_cache_response(&response) {
                                                    if let Ok(json) = serde_json::to_string(&response) {
                                                        state.exact_cache.set(&request, json);
                                                    }
                                                }
                                                let duration_ms = start.elapsed().as_millis() as u64;
                                                let log_context = RequestLogContext::from_response(&request, stream_trace_id.clone(), duration_ms, &response);
                                                log_request(&state, &request, &log_context);
                                                for chunk in streaming_chunks_after_early_event(&response, &response_id, created) {
                                                    yield Ok(Event::default().data(chunk.to_string()));
                                                }
                                                if let Some(suffix) = memory_suffix.as_deref() {
                                                    yield Ok(Event::default().data(memory_feedback_chunk(&request, suffix).to_string()));
                                                }
                                                yield Ok(Event::default().data("[DONE]"));
                                                break 'failover;
                                            }
                                            // All providers exhausted/failed — merge the
                                            // accumulated pass-through attempts with the
                                            // error from the excluding call (Req 4.3) so
                                            // the client sees every failed provider, then
                                            // emit a single graceful aggregated error
                                            // event (client is in SSE mode).
                                            Err(e) => {
                                                let aggregated = merge_streaming_attempts(
                                                    std::mem::take(&mut streaming_attempts),
                                                    e,
                                                );
                                                let duration_ms = start.elapsed().as_millis() as u64;
                                                let log_context = RequestLogContext::from_error(&request, stream_trace_id.clone(), duration_ms, &aggregated);
                                                log_request(&state, &request, &log_context);
                                                let (error_type, message) = classify_stream_error(&aggregated);
                                                for event in emit_sse_error_event(error_type, &message, &stream_trace_id) {
                                                    yield Ok(event);
                                                }
                                                break 'failover;
                                            }
                                        }
                                    }
                                }
                            }
                        }
                        Err(e) => {
                            // The early event already put the client in SSE parsing mode,
                            // so an HTTP error status is no longer possible. Emit a
                            // graceful SSE error event so the client always gets a reason
                            // before the stream terminates (Req 5.1, 5.2, 5.4).
                            let duration_ms = start.elapsed().as_millis() as u64;
                            let log_context = RequestLogContext::from_error(&request, stream_trace_id.clone(), duration_ms, &e);
                            log_request(&state, &request, &log_context);

                            // Map the error variant to an SSE error frame. emit_sse_error_event
                            // already appends [DONE], so we must NOT yield a separate one.
                            let (error_type, message) = classify_stream_error(&e);
                            for event in emit_sse_error_event(error_type, &message, &stream_trace_id) {
                                yield Ok(event);
                            }
                        }
                    }

                    _guard.complete();
                };

        let mut sse = Sse::new(stream)
            .keep_alive(build_keepalive(&streaming_config))
            .into_response();
        attach_trace_id_header(&mut sse, &trace_id);
        return sse;
    }

    // Early event disabled (Req 1.6): preserve the original buffer-and-replay
    // flow where route_request() runs first and pre-stream errors return a
    // proper HTTP status code.
    //
    // Task 5.5 deliberately leaves this path on the buffered `route_request()`
    // (NOT `route_request_streaming`): the value of pass-through is realized
    // alongside the early event, which is the default. Keeping this path
    // buffered preserves the "pre-stream errors return proper HTTP status"
    // behavior (Req 1.6) with the smallest change.
    //
    // Route the request first (provider always returns non-streaming JSON).
    // Errors here happen BEFORE any SSE chunks are sent, so we return a
    // normal JSON error response with the proper HTTP status code.
    let response = match state
        .router
        .route_request(&request, Some(active_handle.clone()))
        .await
    {
        Ok(resp) => resp,
        Err(e) => {
            let duration_ms = request_guard.complete();
            let log_context =
                RequestLogContext::from_error(&request, trace_id.clone(), duration_ms, &e);
            log_request(&state, &request, &log_context);
            let mut response = e.into_response();
            attach_trace_id_header(&mut response, &trace_id);
            return response;
        }
    };

    // Buffer-and-replay: store the assembled response in the exact cache so
    // a subsequent identical request (streaming or not) replays without
    // hitting the provider.  Gated by `should_cache_response` (no tool_calls,
    // finish_reason == stop, etc.).
    if crate::router::router::Router::should_cache_response(&response) {
        if let Ok(json) = serde_json::to_string(&response) {
            state.exact_cache.set(&request, json);
        }
    }

    // Log the successful routed request before streaming begins
    let duration_ms = start.elapsed().as_millis() as u64;
    let log_context =
        RequestLogContext::from_response(&request, trace_id.clone(), duration_ms, &response);
    log_request(&state, &request, &log_context);

    // Success — convert the complete response into SSE chunk format for the client.
    //
    // The gateway always fetches a complete non-streaming response from the
    // provider, then re-chunks it as SSE for the client.  The chunk format
    // must exactly match the OpenAI streaming spec so that clients like
    // Roo Code and Kilo Code can parse tool_calls correctly.
    //
    // Reference (real OpenAI stream for tool_calls):
    //   Chunk 1: delta has role, content:null, tool_calls[0] with index/id/type/function.name/arguments:""
    //   Chunk 2..N: delta has tool_calls[0] with index + function.arguments fragment
    //   Final: delta:{}, finish_reason:"tool_calls", usage:{...}
    let stream_trace_id = trace_id.clone();
    let stream = async_stream::stream! {
        let choice = response.choices.first();

        // Extract tool_calls from message extra fields
        let tool_calls = choice
            .and_then(|c| c.message.extra.get("tool_calls"))
            .and_then(|v| v.as_array())
            .cloned();

        let has_tool_calls = tool_calls.as_ref().is_some_and(|tc| !tc.is_empty());
        let reasoning_text = choice
            .and_then(|c| {
                c.message
                    .extra
                    .get("reasoning")
                    .and_then(|v| v.as_str())
                    .or_else(|| {
                        c.message
                            .extra
                            .get("reasoning_content")
                            .and_then(|v| v.as_str())
                    })
            })
            .unwrap_or("");

        if !reasoning_text.is_empty() {
            tracing::warn!(
                trace_id = %stream_trace_id,
                model = %response.model,
                reasoning_len = reasoning_text.len(),
                has_tool_calls,
                finish_reason = ?choice.and_then(|c| c.finish_reason.as_deref()),
                "Buffered provider response contains reasoning content, but synthesized SSE currently emits only content/tool_calls chunks"
            );
        }

    for chunk in streaming_chunks_from_response(&response) {
        yield Ok::<_, Infallible>(Event::default().data(chunk.to_string()));
    }
    if let Some(suffix) = memory_suffix.as_deref() {
        yield Ok(Event::default().data(memory_feedback_chunk(&request, suffix).to_string()));
    }
    yield Ok(Event::default().data("[DONE]"));
    };

    request_guard.complete();

    Sse::new(stream)
        .keep_alive(build_keepalive(&streaming_config))
        .into_response()
}

#[derive(Debug)]
enum EagerPostCallResult {
    Response(OpenAIResponse),
    Terminal {
        events: Vec<Event>,
        error: GatewayError,
    },
}

enum EagerSinglePostCall {
    Response {
        response: OpenAIResponse,
        refusal: bool,
        replaced: bool,
    },
    Terminal(EagerPostCallResult),
}

async fn finalize_single_eager_post_call(
    engine: &GuardrailEngine,
    request: &OpenAIRequest,
    mut response: OpenAIResponse,
    selector: &BindingSelector,
    guardrail_ctx: &mut GuardrailContext,
    trace_id: &str,
) -> EagerSinglePostCall {
    let tool_ctx = tool_context(request, &response);
    let (post_outcome, refusal) = engine
        .run_post_call(&mut response, selector, guardrail_ctx, trace_id, &tool_ctx)
        .await;
    match post_outcome {
        PostCallOutcome::Proceed => EagerSinglePostCall::Response {
            response,
            refusal: refusal.is_refusal(),
            replaced: false,
        },
        PostCallOutcome::Replaced => EagerSinglePostCall::Response {
            response,
            refusal: false,
            replaced: true,
        },
        PostCallOutcome::Block(block) => {
            EagerSinglePostCall::Terminal(EagerPostCallResult::Terminal {
                events: vec![
                    Event::default().data(
                        guardrail_stream::block_frame_payload(&block.entity_label).to_string(),
                    ),
                    Event::default().data("[DONE]"),
                ],
                error: GatewayError::GuardrailPolicyViolation {
                    category: block.entity_label,
                },
            })
        }
        PostCallOutcome::ServiceFailure => {
            let message = "guardrail service unavailable".to_owned();
            EagerSinglePostCall::Terminal(EagerPostCallResult::Terminal {
                events: emit_sse_error_event("guardrail_unavailable", &message, trace_id),
                error: GatewayError::GuardrailUnavailable(message),
            })
        }
    }
}

async fn finalize_eager_post_call(
    state: &AppState,
    request: &OpenAIRequest,
    mut response: OpenAIResponse,
    guardrail_engine: Option<&Arc<GuardrailEngine>>,
    selector: &BindingSelector,
    guardrail_ctx: &mut GuardrailContext,
    trace_id: &str,
) -> EagerPostCallResult {
    let Some(engine) = guardrail_engine else {
        return EagerPostCallResult::Response(response);
    };

    let tool_ctx = tool_context(request, &response);
    match engine
        .run_post_call(&mut response, selector, guardrail_ctx, trace_id, &tool_ctx)
        .await
    {
        (PostCallOutcome::Proceed, RefusalDecision::NotRefusal)
        | (PostCallOutcome::Replaced, _) => EagerPostCallResult::Response(response),
        (PostCallOutcome::Block(block), _) => EagerPostCallResult::Terminal {
            events: vec![
                Event::default()
                    .data(guardrail_stream::block_frame_payload(&block.entity_label).to_string()),
                Event::default().data("[DONE]"),
            ],
            error: GatewayError::GuardrailPolicyViolation {
                category: block.entity_label,
            },
        },
        (PostCallOutcome::ServiceFailure, _) => {
            let message = "guardrail service unavailable".to_owned();
            EagerPostCallResult::Terminal {
                events: emit_sse_error_event("guardrail_unavailable", &message, trace_id),
                error: GatewayError::GuardrailUnavailable(message),
            }
        }
        (PostCallOutcome::Proceed, RefusalDecision::Refusal(_)) => {
            let model_group = match state.router.find_model_group(&request.model).await {
                Ok(model_group) => model_group,
                Err(_) => {
                    engine.reinject_response(&mut response, guardrail_ctx);
                    return EagerPostCallResult::Response(response);
                }
            };
            let fallback_order = state.router.select_provider_order(&model_group).await;
            let already_tried_provider = response
                .extra
                .get("gateway_provider")
                .and_then(|value| value.as_str())
                .unwrap_or_default()
                .to_owned();
            let mut tried = vec![already_tried_provider];
            let mut last_response = response;
            let mut skip_reinjection = false;

            for target in &fallback_order {
                if tried.contains(&target.provider) {
                    continue;
                }
                let target_key = format!("{}:{}", target.provider, target.model);
                let circuit_breaker = state.router.get_circuit_breaker(&target_key).await;
                if !circuit_breaker.is_available().await {
                    continue;
                }
                tried.push(target.provider.clone());

                let Ok(candidate) = state
                    .router
                    .route_with_failover(request, vec![target.clone()], None)
                    .await
                else {
                    tracing::debug!(
                        trace_id,
                        provider = %target.provider,
                        model = %target.model,
                        category = "refusal_provider_error",
                        "structured streaming refusal failover provider failed"
                    );
                    continue;
                };

                match finalize_single_eager_post_call(
                    engine,
                    request,
                    candidate,
                    selector,
                    guardrail_ctx,
                    trace_id,
                )
                .await
                {
                    EagerSinglePostCall::Response {
                        response: candidate,
                        refusal,
                        replaced,
                    } => {
                        last_response = candidate;
                        if replaced {
                            skip_reinjection = true;
                            break;
                        }
                        if !refusal {
                            break;
                        }
                    }
                    EagerSinglePostCall::Terminal(result) => return result,
                }
            }

            if !skip_reinjection {
                engine.reinject_response(&mut last_response, guardrail_ctx);
            }
            EagerPostCallResult::Response(last_response)
        }
    }
}

#[allow(clippy::too_many_arguments)]
async fn validate_eager_structured_response(
    state: &AppState,
    request: &OpenAIRequest,
    mut response: OpenAIResponse,
    engine: Option<&Arc<StructuredOutputEngine>>,
    guardrail_engine: Option<&Arc<GuardrailEngine>>,
    selector: &BindingSelector,
    guardrail_ctx: &mut GuardrailContext,
    trace_id: &str,
) -> Result<(OpenAIResponse, ValidationResponseStatus), EagerPostCallResult> {
    let Some(engine) = engine else {
        tracing::warn!(
            trace_id,
            category = "disabled",
            "structured streaming validation skipped"
        );
        return Ok((response, ValidationResponseStatus::Skipped));
    };
    let model_group = match state.router.find_model_group(&request.model).await {
        Ok(model_group) => model_group,
        Err(_) => {
            tracing::warn!(
                trace_id,
                category = "routing_metadata",
                "structured streaming validation skipped"
            );
            return Ok((response, ValidationResponseStatus::Skipped));
        }
    };
    let Some((provider, model)) = response_provider_model(&response) else {
        tracing::warn!(
            trace_id,
            category = "routing_metadata",
            "structured streaming validation skipped"
        );
        return Ok((response, ValidationResponseStatus::Skipped));
    };

    let decision_started = Instant::now();
    let validation_decision = engine.should_validate(request, &model_group.name, provider, model);
    let decision_latency_ms = decision_started.elapsed().as_secs_f64() * 1_000.0;
    match validation_decision {
        ValidationDecision::NotApplicable => {
            Ok((response, ValidationResponseStatus::NotApplicable))
        }
        ValidationDecision::Skipped(reason) => {
            state
                .metrics
                .observe_structured_output_latency(provider, model, decision_latency_ms);
            tracing::warn!(
                trace_id,
                category = validation_skip_category(&reason),
                "structured streaming validation skipped"
            );
            Ok((response, ValidationResponseStatus::Skipped))
        }
        ValidationDecision::Validate(schema_context) => {
            let initial_provider = provider.to_owned();
            let initial_model = model.to_owned();
            let mut gateway_processing_ms = decision_latency_ms;
            let mut validation = engine
                .validate_response(
                    &schema_context,
                    &response,
                    &initial_provider,
                    &initial_model,
                )
                .await;
            gateway_processing_ms += validation.latency_ms;
            let mut status = ValidationResponseStatus::from_outcome(validation.outcome);

            if validation.outcome == StructuredOutputOutcome::Fail {
                let initial_failure =
                    collect_structured_output_failure(&response, &validation.choices);
                tracing::info!(
                    trace_id,
                    provider = %initial_provider,
                    model = %initial_model,
                    error_count = initial_failure.violations.len(),
                    retry_attempt = 0,
                    category = "validation_failed",
                    "structured streaming output validation failed"
                );
                let original_failed_response = response.clone();
                let effective_config =
                    engine.effective_config(&model_group.name, &initial_provider, &initial_model);
                let target =
                    find_selected_provider_model(&model_group, &initial_provider, &initial_model);

                if effective_config.max_retries == 0 {
                    status = ValidationResponseStatus::Failed;
                } else if let Some(target) = target {
                    let context_window = state
                        .router
                        .context_manager()
                        .get_capabilities(&target.model)
                        .map(|capabilities| capabilities.context_window as usize)
                        .unwrap_or_else(default_context_window);
                    let mut remaining_attempts = usize::from(effective_config.max_retries);
                    let mut last_successful_response = response.clone();
                    let mut successful_retry_count = 0usize;
                    let mut provider_error_count = 0usize;

                    while remaining_attempts > 0 {
                        let failure = collect_structured_output_failure(
                            &last_successful_response,
                            &validation.choices,
                        );
                        let request_tokens = TokenCounter::new().count_request(request) as usize;
                        let prompt_started = Instant::now();
                        let retry_request = engine.build_retry_request(
                            request,
                            &schema_context,
                            &failure.violations,
                            &failure.previous_output,
                            &effective_config,
                            true,
                            context_window,
                            request_tokens,
                        );
                        gateway_processing_ms += prompt_started.elapsed().as_secs_f64() * 1_000.0;
                        debug_assert!(!retry_request.stream);

                        match state
                            .router
                            .route_with_failover(&retry_request, vec![target.clone()], None)
                            .await
                        {
                            Ok(retry_response) => {
                                successful_retry_count += 1;
                                remaining_attempts -= 1;
                                let retry_response = if let Some(guardrail) = guardrail_engine {
                                    let guardrail_started = Instant::now();
                                    let post_call = finalize_single_eager_post_call(
                                        guardrail,
                                        request,
                                        retry_response,
                                        selector,
                                        guardrail_ctx,
                                        trace_id,
                                    )
                                    .await;
                                    gateway_processing_ms +=
                                        guardrail_started.elapsed().as_secs_f64() * 1_000.0;
                                    match post_call {
                                        EagerSinglePostCall::Response {
                                            response,
                                            refusal: true,
                                            ..
                                        } => {
                                            last_successful_response = response;
                                            break;
                                        }
                                        EagerSinglePostCall::Response { response, .. } => response,
                                        EagerSinglePostCall::Terminal(terminal) => {
                                            return Err(terminal)
                                        }
                                    }
                                } else {
                                    retry_response
                                };
                                let (retry_provider, retry_model) =
                                    response_provider_model(&retry_response)
                                        .map(|(provider, model)| {
                                            (provider.to_owned(), model.to_owned())
                                        })
                                        .unwrap_or_else(|| {
                                            (target.provider.clone(), target.model.clone())
                                        });
                                validation = engine
                                    .validate_response(
                                        &schema_context,
                                        &retry_response,
                                        &retry_provider,
                                        &retry_model,
                                    )
                                    .await;
                                gateway_processing_ms += validation.latency_ms;
                                status = ValidationResponseStatus::from_outcome(validation.outcome);
                                last_successful_response = retry_response;

                                if validation.outcome == StructuredOutputOutcome::Pass {
                                    response = last_successful_response.clone();
                                    state.metrics.record_structured_output_retry(
                                        &initial_provider,
                                        &initial_model,
                                        "recovered",
                                    );
                                    break;
                                }
                                if validation.outcome == StructuredOutputOutcome::Skipped {
                                    response = last_successful_response.clone();
                                    tracing::warn!(
                                        trace_id,
                                        category = "validation_internal_skip",
                                        "structured streaming validation skipped"
                                    );
                                    break;
                                }
                                let retry_failure = collect_structured_output_failure(
                                    &last_successful_response,
                                    &validation.choices,
                                );
                                tracing::info!(
                                    trace_id,
                                    provider = %retry_provider,
                                    model = %retry_model,
                                    error_count = retry_failure.violations.len(),
                                    retry_attempt = successful_retry_count,
                                    category = "validation_failed",
                                    "structured streaming output validation failed"
                                );
                            }
                            Err(error) => {
                                let consumed =
                                    consume_provider_error_attempts(&error).min(remaining_attempts);
                                provider_error_count += consumed;
                                remaining_attempts -= consumed;
                                tracing::warn!(
                                    trace_id,
                                    category = "provider_error",
                                    "structured streaming retry provider failed"
                                );
                            }
                        }
                    }

                    if status == ValidationResponseStatus::Failed {
                        response = last_successful_response;
                        if successful_retry_count + provider_error_count > 0 {
                            state.metrics.record_structured_output_retry(
                                &initial_provider,
                                &initial_model,
                                "exhausted",
                            );
                        }
                        tracing::warn!(
                            trace_id,
                            category = "retry_exhausted",
                            successful_retry_count,
                            provider_error_count,
                            "structured streaming retry exhausted"
                        );
                    }
                } else {
                    response = original_failed_response;
                    status = ValidationResponseStatus::Failed;
                    tracing::warn!(
                        trace_id,
                        category = "retry_dispatch_setup",
                        "structured streaming retry setup failed"
                    );
                }
            } else if validation.outcome == StructuredOutputOutcome::Skipped {
                tracing::warn!(
                    trace_id,
                    category = "validation_internal_skip",
                    "structured streaming validation skipped"
                );
            }

            state.metrics.observe_structured_output_latency(
                &initial_provider,
                &initial_model,
                gateway_processing_ms,
            );
            Ok((response, status))
        }
    }
}

#[allow(clippy::too_many_arguments)]
async fn stream_eager_structured_output(
    state: AppState,
    request: OpenAIRequest,
    trace_id: String,
    streaming_config: StreamingConfig,
    guardrail_engine: Option<Arc<GuardrailEngine>>,
    structured_output_engine: Option<Arc<StructuredOutputEngine>>,
    selector: BindingSelector,
    mut guardrail_ctx: GuardrailContext,
    mut request_guard: RequestCompleteGuard,
    active: Option<crate::active_requests::ActiveRequestHandle>,
) -> Response {
    let mut response = match state
        .router
        .route_request_streaming(&request, active.clone())
        .await
    {
        Ok(StreamingResponse::Buffered(response)) => response,
        Ok(StreamingResponse::PassThrough {
            byte_stream,
            provider,
            model,
            compression,
            concurrency_permit,
        }) => {
            let _concurrency_permit = concurrency_permit;
            let mut buffer = guardrail_stream::SseBuffer::with_default_cap();
            let mut bytes = byte_stream.bytes_stream();
            while let Some(chunk) = bytes.next().await {
                match chunk {
                    Ok(chunk) if buffer.push_bytes(&chunk).is_ok() => {}
                    Ok(_) => {
                        let message = format!(
                            "Buffered streaming response exceeded the {} byte structured output buffer limit",
                            guardrail_stream::MAX_STREAM_BUFFER_BYTES
                        );
                        let error = GatewayError::GuardrailUnavailable(message);
                        let duration_ms = request_guard.complete();
                        let log_context = RequestLogContext::from_error(
                            &request,
                            trace_id.clone(),
                            duration_ms,
                            &error,
                        );
                        log_request(&state, &request, &log_context);
                        return eager_sse_response(
                            structured_stream_overflow_events(&trace_id),
                            &streaming_config,
                            &trace_id,
                            None,
                            None,
                        );
                    }
                    Err(error) => {
                        let gateway_error = GatewayError::GuardrailUnavailable(format!(
                            "Upstream streaming response could not be buffered: {error}"
                        ));
                        let duration_ms = request_guard.complete();
                        let log_context = RequestLogContext::from_error(
                            &request,
                            trace_id.clone(),
                            duration_ms,
                            &gateway_error,
                        );
                        log_request(&state, &request, &log_context);
                        return eager_sse_response(
                            emit_sse_error_event(
                                "stream_error",
                                &gateway_error.to_string(),
                                &trace_id,
                            ),
                            &streaming_config,
                            &trace_id,
                            None,
                            None,
                        );
                    }
                }
            }

            let assembled = match buffer.assemble() {
                Ok(assembled) => assembled,
                Err(message) => {
                    let gateway_error = GatewayError::GuardrailUnavailable(message);
                    let duration_ms = request_guard.complete();
                    let log_context = RequestLogContext::from_error(
                        &request,
                        trace_id.clone(),
                        duration_ms,
                        &gateway_error,
                    );
                    log_request(&state, &request, &log_context);
                    return eager_sse_response(
                        emit_sse_error_event("stream_error", &gateway_error.to_string(), &trace_id),
                        &streaming_config,
                        &trace_id,
                        None,
                        None,
                    );
                }
            };
            if !assembled.complete {
                let message = "Upstream stream ended before a complete structured output response";
                let gateway_error = GatewayError::GuardrailUnavailable(message.to_owned());
                let duration_ms = request_guard.complete();
                let log_context = RequestLogContext::from_error(
                    &request,
                    trace_id.clone(),
                    duration_ms,
                    &gateway_error,
                );
                log_request(&state, &request, &log_context);
                return eager_sse_response(
                    emit_sse_error_event("stream_error", message, &trace_id),
                    &streaming_config,
                    &trace_id,
                    None,
                    None,
                );
            }

            let mut response = assembled.response;
            response.extra.insert(
                "gateway_provider".to_owned(),
                serde_json::Value::String(provider),
            );
            response.extra.insert(
                "gateway_responded_model".to_owned(),
                serde_json::Value::String(model),
            );
            response.extra.insert(
                "gateway_compression".to_owned(),
                serde_json::to_value(compression)
                    .expect("CompressionStats serialization must succeed"),
            );
            response
        }
        Err(error) => {
            let duration_ms = request_guard.complete();
            let log_context =
                RequestLogContext::from_error(&request, trace_id.clone(), duration_ms, &error);
            log_request(&state, &request, &log_context);
            let (error_type, message) = classify_stream_error(&error);
            return eager_sse_response(
                emit_sse_error_event(error_type, &message, &trace_id),
                &streaming_config,
                &trace_id,
                None,
                None,
            );
        }
    };

    response = match finalize_eager_post_call(
        &state,
        &request,
        response,
        guardrail_engine.as_ref(),
        &selector,
        &mut guardrail_ctx,
        &trace_id,
    )
    .await
    {
        EagerPostCallResult::Response(response) => response,
        EagerPostCallResult::Terminal { events, error } => {
            let duration_ms = request_guard.complete();
            let log_context =
                RequestLogContext::from_error(&request, trace_id.clone(), duration_ms, &error);
            log_request(&state, &request, &log_context);
            return eager_sse_response(events, &streaming_config, &trace_id, None, None);
        }
    };

    let (response, validation_status) = match validate_eager_structured_response(
        &state,
        &request,
        response,
        structured_output_engine.as_ref(),
        guardrail_engine.as_ref(),
        &selector,
        &mut guardrail_ctx,
        &trace_id,
    )
    .await
    {
        Ok(result) => result,
        Err(EagerPostCallResult::Terminal { events, error }) => {
            let duration_ms = request_guard.complete();
            let log_context =
                RequestLogContext::from_error(&request, trace_id.clone(), duration_ms, &error);
            log_request(&state, &request, &log_context);
            return eager_sse_response(events, &streaming_config, &trace_id, None, None);
        }
        Err(EagerPostCallResult::Response(_)) => unreachable!(),
    };

    if should_cache_eager_structured(&request, Some(&response), Some(validation_status)) {
        if let Ok(json) = serde_json::to_string(&response) {
            state.exact_cache.set(&request, json.clone());
            let skip_semantic =
                request.extra.contains_key("tools") || request.extra.contains_key("tool_choice");
            if !skip_semantic {
                if let Some(cache) = state.cache.as_ref() {
                    if let Err(error) = cache.set(&request, &json, 0.0).await {
                        tracing::warn!(
                            trace_id,
                            category = "semantic_cache_write",
                            "failed to cache validated structured streaming response: {error}"
                        );
                    }
                }
            }
        }
    }

    let routing_headers = smart_routing_headers(&response);
    let events = rechunk_structured_response(&response);
    let duration_ms = request_guard.complete();
    let log_context =
        RequestLogContext::from_response(&request, trace_id.clone(), duration_ms, &response);
    log_request(&state, &request, &log_context);
    eager_sse_response(
        events,
        &streaming_config,
        &trace_id,
        Some(validation_status),
        Some(routing_headers),
    )
}

/// Streaming handler variant used when a post-call guardrail pipeline is bound
/// (task 13.3, Req 10). It FORCES the buffered path: rather than relaying
/// upstream SSE chunks verbatim, it assembles the complete response, runs the
/// post-call pipeline on it, then re-chunks the result back into SSE.
///
/// - Buffers the upstream response under a 10 MB cap, aborting with a gateway
///   error beyond it (Req 10.1), emitting keep-alive comments during idle gaps
///   at `keepalive_interval_seconds` (Req 10.2).
/// - On a premature disconnect (no `finish_reason`), applies the bound post-call
///   stages' failure policy: `fail_close` discards, `fail_open` forwards the
///   partial content (Req 10.5).
/// - Maps post-call outcomes to SSE: pass/replace → re-chunk (Req 10.4) and
///   forward within 500 ms of analysis (Req 10.6, no artificial delay);
///   `block` → a terminal policy-violation event + `[DONE]` (Req 10.3);
///   `ServiceFailure` → a 503-style graceful SSE error termination (Req 9.7).
///
/// Pre-call has already run against `request` before this function is called;
/// `guardrail_ctx` carries the pre-call PII Re_Injection_Map so post-call
/// re-injection restores originals in the assembled response (Req 9.5).
#[allow(clippy::too_many_arguments)]
async fn stream_buffered_with_post_call(
    state: AppState,
    request: OpenAIRequest,
    trace_id: String,
    start: std::time::Instant,
    streaming_config: StreamingConfig,
    engine: Arc<GuardrailEngine>,
    selector: BindingSelector,
    mut guardrail_ctx: GuardrailContext,
    request_guard: RequestCompleteGuard,
    active: Option<crate::active_requests::ActiveRequestHandle>,
) -> Response {
    let emit_early = streaming_config.emit_early_event;
    let response_id = format!("chatcmpl-{}", uuid::Uuid::new_v4());
    let created = chrono::Utc::now().timestamp();
    let requested_model = request.model.clone();
    // Keep-alive cadence while buffering (Req 10.2). Clamp to at least 1s so a
    // configured `0` (axum-default keepalive) does not spin the buffering loop.
    let keepalive_interval =
        Duration::from_secs(streaming_config.keepalive_interval_seconds.max(1));
    // Premature-disconnect failure policy from the bound post-call stages (Req 10.5).
    let discard_on_disconnect =
        guardrail_stream::disconnect_discards_partial(&engine.resolver().resolve(&selector));

    let stream_trace_id = trace_id.clone();
    let active_stream = active.clone();

    let stream = async_stream::stream! {
            let mut _guard = request_guard;

            // Early synthetic role event (Req 1.1-1.3) so the client's idle timer
            // resets while we buffer; re-chunking below reuses its id/created.
            if emit_early {
                let early = early_event_chunk(&response_id, created, &requested_model);
                yield Ok::<_, Infallible>(Event::default().data(early.to_string()));
            }

            // Force the buffered path: assemble the full upstream response (never
            // relay pass-through chunks to the caller) so the post-call pipeline
            // sees the complete content.
            let mut assembled: Option<OpenAIResponse> = None;
            let mut streaming_compression: Option<CompressionStats> = None;
            let mut disconnected = false;

            match state.router.route_request_streaming(&request, active_stream.clone()).await {
                Ok(StreamingResponse::Buffered(response)) => {
                    assembled = Some(response);
                }
    Ok(StreamingResponse::PassThrough { byte_stream, compression, concurrency_permit, .. }) => {
    let _concurrency_permit = concurrency_permit;
    streaming_compression = Some(compression.clone());
                    // Buffer the live SSE body under the 10 MB cap while emitting
                    // keep-alive comments during idle gaps (Req 10.1, 10.2).
                    let mut buf = guardrail_stream::SseBuffer::with_default_cap();
                    let mut bytes = byte_stream.bytes_stream();
                    let mut too_large = false;
                    loop {
                        match tokio::time::timeout(keepalive_interval, bytes.next()).await {
                            // Idle gap while buffering: keep the client alive (Req 10.2).
                            Err(_elapsed) => {
                                yield Ok(Event::default().comment("keepalive"));
                            }
                            // Upstream stream ended.
                            Ok(None) => break,
                            Ok(Some(Ok(b))) => {
                                if buf.push_bytes(&b).is_err() {
                                    too_large = true;
                                    break;
                                }
                            }
                            // Transport error mid-stream → premature disconnect (Req 10.5).
                            Ok(Some(Err(_e))) => {
                                disconnected = true;
                                break;
                            }
                        }
                    }

                    if too_large {
                        // Req 10.1: abort with a gateway error when the cap is exceeded.
                        let msg = format!(
                            "Buffered streaming response exceeded the {} byte guardrail buffer limit",
                            guardrail_stream::MAX_STREAM_BUFFER_BYTES
                        );
                        let duration_ms = start.elapsed().as_millis() as u64;
                        let err = GatewayError::GuardrailUnavailable(msg.clone());
                        let log_context = RequestLogContext::from_error(&request, stream_trace_id.clone(), duration_ms, &err);
                        log_request(&state, &request, &log_context);
                        for event in emit_sse_error_event("guardrail_buffer_overflow", &msg, &stream_trace_id) {
                            yield Ok(event);
                        }
                        _guard.complete();
                        return;
                    }

                    match buf.assemble() {
                        Ok(a) => {
                            // No finish_reason ⇒ premature disconnect (Req 10.5).
                            if !a.complete {
                                disconnected = true;
                            }
                            assembled = Some(a.response);
                        }
                        // Empty/mid-error buffer: treat as a disconnect (Req 10.5).
                        Err(_e) => {
                            disconnected = true;
                        }
                    }
                }
                Err(e) => {
                    // Routing failed before any streaming — graceful SSE error frame.
                    let duration_ms = start.elapsed().as_millis() as u64;
                    let log_context = RequestLogContext::from_error(&request, stream_trace_id.clone(), duration_ms, &e);
                    log_request(&state, &request, &log_context);
                    let (error_type, message) = classify_stream_error(&e);
                    for event in emit_sse_error_event(error_type, &message, &stream_trace_id) {
                        yield Ok(event);
                    }
                    _guard.complete();
                    return;
                }
            }

            // Premature disconnect handling (Req 10.5): fail_close discards, fail_open
            // forwards the partial content through post-call.
            if disconnected && (discard_on_disconnect || assembled.is_none()) {
                let msg = "Upstream stream ended before a complete response; guardrail failure policy is fail-close".to_string();
                let duration_ms = start.elapsed().as_millis() as u64;
                let err = GatewayError::GuardrailUnavailable(msg.clone());
                let log_context = RequestLogContext::from_error(&request, stream_trace_id.clone(), duration_ms, &err);
                log_request(&state, &request, &log_context);
                for event in emit_sse_error_event("guardrail_unavailable", &msg, &stream_trace_id) {
                    yield Ok(event);
                }
                _guard.complete();
                return;
            }

            let mut response = match assembled {
                Some(r) => r,
                None => {
                    yield Ok(Event::default().data("[DONE]"));
                    _guard.complete();
                    return;
                }
            };
            if let Some(compression) = streaming_compression.take() {
                response.extra.insert(
                    "gateway_compression".to_string(),
                    serde_json::to_value(compression)
                        .expect("CompressionStats serialization must succeed"),
                );
            }

            // Post-call guardrails on the assembled response, using the SAME ctx as
            // pre-call so PII re-injection works (Req 9.5). No artificial delay is
            // added, so re-chunked events forward within 500 ms of analysis (Req 10.6).
            //
            // Refusal detection runs on the assembled buffered response BEFORE the
            // failover decision (Req 12.9). On premature stream termination (no
            // finish_reason), detection runs on the partially assembled content and
            // the same failover decision applies (Req 12.14).
            let tool_ctx = ToolContext {
                tool_use_allowed: request.extra.get("tool_choice").and_then(|v| v.as_str()).map_or(true, |tc| tc != "none"),
                tools_provided: request.extra.get("tools").and_then(|v| v.as_array()).map_or(false, |t| !t.is_empty()),
                finish_reason_is_tool_call: response.choices.first().and_then(|c| c.finish_reason.as_deref()).map_or(false, |r| r == "tool_calls"),
                has_tool_calls: response.choices.first().map_or(false, |c| {
                    c.message.extra.get("tool_calls").and_then(|v| v.as_array()).map_or(false, |a| !a.is_empty())
                }),
            };
            match engine
                .run_post_call(&mut response, &selector, &mut guardrail_ctx, &stream_trace_id, &tool_ctx)
                .await
            {
                // Proceed without refusal, or replaced (halting action skips
                // re-injection, Req 9.4) → re-chunk normally.
                (PostCallOutcome::Proceed, RefusalDecision::NotRefusal)
                | (PostCallOutcome::Replaced, _) => {
                    let chunks = if emit_early {
                        guardrail_stream::rechunk_after_early_event(&response, &response_id, created)
                    } else {
                        guardrail_stream::rechunk_full(&response)
                    };
                    if crate::router::router::Router::should_cache_response(&response) {
                        if let Ok(json) = serde_json::to_string(&response) {
                            state.exact_cache.set(&request, json);
                        }
                    }
                    let duration_ms = start.elapsed().as_millis() as u64;
                    let log_context = RequestLogContext::from_response(&request, stream_trace_id.clone(), duration_ms, &response);
                    log_request(&state, &request, &log_context);
                    for chunk in chunks {
                        yield Ok(Event::default().data(chunk.to_string()));
                    }
                    yield Ok(Event::default().data("[DONE]"));
                }
                // Refusal detected with failover enabled (Req 12.5, 12.9, 12.14):
                // run the bounded re-dispatch loop, buffering each re-dispatched
                // target before detection (same pattern as the non-streaming 17.2).
                (PostCallOutcome::Proceed, RefusalDecision::Refusal(_signal)) => {
                    // Compute the fallback ordering once (Req 12.5, 12.7).
                    let model_group = match state.router.find_model_group(&request.model).await {
                        Ok(mg) => mg,
                        Err(_) => {
                            // Cannot resolve model group — re-inject on current response and return.
                            engine.reinject_response(&mut response, &guardrail_ctx);
                            let chunks = if emit_early {
                                guardrail_stream::rechunk_after_early_event(&response, &response_id, created)
                            } else {
                                guardrail_stream::rechunk_full(&response)
                            };
                            let duration_ms = start.elapsed().as_millis() as u64;
                            let log_context = RequestLogContext::from_response(&request, stream_trace_id.clone(), duration_ms, &response);
                            log_request(&state, &request, &log_context);
                            for chunk in chunks {
                                yield Ok(Event::default().data(chunk.to_string()));
                            }
                            yield Ok(Event::default().data("[DONE]"));
                            _guard.complete();
                            return;
                        }
                    };
                    let fallback_order = state.router.select_provider_order(&model_group).await;

                    // Identify the provider that produced the current (refused) response.
                    let already_tried_provider = response
                        .extra
                        .get("gateway_provider")
                        .and_then(|v| v.as_str())
                        .unwrap_or("")
                        .to_string();

                    let mut tried: Vec<String> = vec![already_tried_provider.clone()];
                    let mut last_response = response;
                    let mut skip_reinjection = false;

                    // Bounded re-dispatch loop (Req 12.7): attempt each target at
                    // most once, bounded by ordering length. Re-dispatched targets in
                    // a streaming context are likewise buffered before detection.
                    for target in &fallback_order {
                        let target_key = format!("{}:{}", target.provider, target.model);
                        if tried.contains(&target.provider) {
                            continue;
                        }
                        // Skip targets whose circuit breaker is open (Req 12.10).
                        let cb = state.router.get_circuit_breaker(&target_key).await;
                        if !cb.is_available().await {
                            tracing::debug!(
                                provider = %target.provider,
                                model = %target.model,
                                "Streaming refusal failover: circuit breaker open, skipping"
                            );
                            continue;
                        }

                        tried.push(target.provider.clone());

                        // Re-dispatch the already-redacted request to this single
                        // target (Req 12.5). Buffer the response for detection.
                        let attempt_result = state
                            .router
                            .route_with_failover(&request, vec![target.clone()], None)
                            .await;

                        match attempt_result {
                            Ok(mut new_response) => {
                                // Re-run post-call + refusal detection on the new response.
                                let new_tool_ctx = ToolContext {
                                    tool_use_allowed: request.extra.get("tool_choice").and_then(|v| v.as_str()).map_or(true, |tc| tc != "none"),
                                    tools_provided: request.extra.get("tools").and_then(|v| v.as_array()).map_or(false, |t| !t.is_empty()),
                                    finish_reason_is_tool_call: new_response.choices.first().and_then(|c| c.finish_reason.as_deref()).map_or(false, |r| r == "tool_calls"),
                                    has_tool_calls: new_response.choices.first().map_or(false, |c| {
                                        c.message.extra.get("tool_calls").and_then(|v| v.as_array()).map_or(false, |a| !a.is_empty())
                                    }),
                                };
                                let (post_outcome, refusal_decision) = engine
                                    .run_post_call(&mut new_response, &selector, &mut guardrail_ctx, &stream_trace_id, &new_tool_ctx)
                                    .await;

                                match post_outcome {
                                    PostCallOutcome::Block(block) => {
                                        // Block from failover target → terminal SSE event (Req 10.3).
                                        let payload = guardrail_stream::block_frame_payload(&block.entity_label);
                                        let duration_ms = start.elapsed().as_millis() as u64;
                                        let err = GatewayError::GuardrailPolicyViolation { category: block.entity_label.clone() };
                                        let log_context = RequestLogContext::from_error(&request, stream_trace_id.clone(), duration_ms, &err);
                                        log_request(&state, &request, &log_context);
                                        yield Ok(Event::default().data(payload.to_string()));
                                        yield Ok(Event::default().data("[DONE]"));
                                        _guard.complete();
                                        return;
                                    }
                                    PostCallOutcome::ServiceFailure => {
                                        let msg = "guardrail service unavailable".to_string();
                                        let duration_ms = start.elapsed().as_millis() as u64;
                                        let err = GatewayError::GuardrailUnavailable(msg.clone());
                                        let log_context = RequestLogContext::from_error(&request, stream_trace_id.clone(), duration_ms, &err);
                                        log_request(&state, &request, &log_context);
                                        for event in emit_sse_error_event("guardrail_unavailable", &msg, &stream_trace_id) {
                                            yield Ok(event);
                                        }
                                        _guard.complete();
                                        return;
                                    }
                                    PostCallOutcome::Replaced => {
                                        last_response = new_response;
                                        skip_reinjection = true;
                                        break;
                                    }
                                    PostCallOutcome::Proceed => {
                                        if !refusal_decision.is_refusal() {
                                            // First non-refusal: use it (Req 12.5).
                                            last_response = new_response;
                                            break;
                                        }
                                        // Still a refusal — record and continue loop.
                                        last_response = new_response;
                                    }
                                }
                            }
                            Err(_e) => {
                                tracing::debug!(
                                    provider = %target.provider,
                                    "Streaming refusal failover: provider error, trying next"
                                );
                                continue;
                            }
                        }
                    }

                    // PII re-injection runs exactly once on the finally selected
                    // response (Req 9.5, 12.5), unless replaced (Req 9.4).
                    if !skip_reinjection {
                        engine.reinject_response(&mut last_response, &guardrail_ctx);
                    }

                    // Re-chunk the final response into SSE events (Req 10.4).
                    let chunks = if emit_early {
                        guardrail_stream::rechunk_after_early_event(&last_response, &response_id, created)
                    } else {
                        guardrail_stream::rechunk_full(&last_response)
                    };
                    if crate::router::router::Router::should_cache_response(&last_response) {
                        if let Ok(json) = serde_json::to_string(&last_response) {
                            state.exact_cache.set(&request, json);
                        }
                    }
                    let duration_ms = start.elapsed().as_millis() as u64;
                    let log_context = RequestLogContext::from_response(&request, stream_trace_id.clone(), duration_ms, &last_response);
                    log_request(&state, &request, &log_context);
                    for chunk in chunks {
                        yield Ok(Event::default().data(chunk.to_string()));
                    }
                    yield Ok(Event::default().data("[DONE]"));
                }
                // Block → terminal policy-violation event + [DONE] (Req 10.3).
                (PostCallOutcome::Block(block), _) => {
                    let payload = guardrail_stream::block_frame_payload(&block.entity_label);
                    let duration_ms = start.elapsed().as_millis() as u64;
                    let err = GatewayError::GuardrailPolicyViolation { category: block.entity_label.clone() };
                    let log_context = RequestLogContext::from_error(&request, stream_trace_id.clone(), duration_ms, &err);
                    log_request(&state, &request, &log_context);
                    yield Ok(Event::default().data(payload.to_string()));
                    yield Ok(Event::default().data("[DONE]"));
                }
                // fail_close provider error/timeout → 503-style SSE termination (Req 9.7).
                (PostCallOutcome::ServiceFailure, _) => {
                    let msg = "guardrail service unavailable".to_string();
                    let duration_ms = start.elapsed().as_millis() as u64;
                    let err = GatewayError::GuardrailUnavailable(msg.clone());
                    let log_context = RequestLogContext::from_error(&request, stream_trace_id.clone(), duration_ms, &err);
                    log_request(&state, &request, &log_context);
                    for event in emit_sse_error_event("guardrail_unavailable", &msg, &stream_trace_id) {
                        yield Ok(event);
                    }
                }
            }

            _guard.complete();
        };

    let mut sse = Sse::new(stream)
        .keep_alive(build_keepalive(&streaming_config))
        .into_response();
    attach_trace_id_header(&mut sse, &trace_id);
    sse
}

/// Build the SSE keep-alive policy from streaming config (Req 2.1-2.5).
///
/// When `keepalive_interval_seconds == 0`, keep-alive falls back to axum's
/// default behavior. Otherwise emits a `:keepalive` comment at the configured
/// interval to keep client idle timers from firing during slow responses.
fn build_keepalive(streaming_config: &StreamingConfig) -> KeepAlive {
    if streaming_config.keepalive_interval_seconds == 0 {
        KeepAlive::default()
    } else {
        KeepAlive::new()
            .interval(Duration::from_secs(
                streaming_config.keepalive_interval_seconds,
            ))
            .text("keepalive")
    }
}

/// Build the synthetic "early" SSE chunk emitted before the upstream provider
/// responds. It carries a `role: assistant` delta so the client's idle timer
/// resets immediately on streaming requests.
///
/// Streaming Reliability Req 1.1, 1.2, 1.3 — the `id`/`created`/`model` are
/// pre-generated by the caller so subsequent chunks can reuse them (task 2.2).
fn early_event_chunk(id: &str, created: i64, model: &str) -> serde_json::Value {
    serde_json::json!({
        "id": id,
        "object": "chat.completion.chunk",
        "created": created,
        "model": model,
        "choices": [{
            "index": 0,
            "delta": { "role": "assistant" },
            "finish_reason": null
        }]
    })
}

/// Build a graceful SSE error termination: an `error` data event followed by
/// the `[DONE]` sentinel (Req 5.1-5.5).
///
/// The payload is `{"error":{"message":"...","type":"...","trace_id":"..."}}`
/// so clients in SSE parsing mode see a proper error frame and stream
/// termination instead of a silent TCP close. The caller is responsible for
/// having already emitted the early event so the client is in SSE mode.
fn emit_sse_error_event(error_type: &str, message: &str, trace_id: &str) -> Vec<Event> {
    let payload = sse_error_payload(error_type, message, trace_id);
    vec![
        Event::default().data(payload.to_string()),
        Event::default().data("[DONE]"),
    ]
}

/// Pure builder for the SSE error frame payload (Req 5.1, 5.2, 5.5).
///
/// Returns `{"error":{"message":"...","type":"...","trace_id":"..."}}`. Kept
/// separate from `emit_sse_error_event` because axum's `Event` does not expose
/// its data for assertion — testing this helper directly is the deterministic
/// way to verify the exact error shape and `trace_id` correlation.
fn sse_error_payload(error_type: &str, message: &str, trace_id: &str) -> serde_json::Value {
    serde_json::json!({
        "error": {
            "message": message,
            "type": error_type,
            "trace_id": trace_id,
        }
    })
}

/// Classify a `GatewayError` into an `(error_type, message)` pair for the SSE
/// error frame (Req 5.1, 5.2).
///
/// `router.route_request()` wraps single-provider timeouts in
/// `GatewayError::AllProvidersFailed(AggregatedError { attempts })`, so the
/// direct `TtfbTimeout`/`TotalTimeout` arms never fire for the end-to-end
/// streaming path. We must therefore inspect the aggregated attempts and
/// recover the timeout kind from each attempt's recorded error string (the
/// per-attempt `e.to_string()`), matching against the stable in-crate
/// `GatewayError` Display signatures:
///   - TtfbTimeout  => "...waiting for first byte from provider"
///   - TotalTimeout => "...total round-trip time"
/// The direct arms are kept for any code path that returns those variants
/// unwrapped (they carry the precise `{secs}` in the message).
fn classify_stream_error(e: &GatewayError) -> (&'static str, String) {
    /// TTFB-timeout signature from `GatewayError::TtfbTimeout` Display text.
    const TTFB_SIGNATURE: &str = "waiting for first byte";
    /// Total-timeout signature from `GatewayError::TotalTimeout` Display text.
    const TOTAL_SIGNATURE: &str = "total round-trip";

    match e {
        GatewayError::TtfbTimeout(secs) => (
            "ttfb_timeout_error",
            format!("Provider did not respond within {}s", secs),
        ),
        GatewayError::TotalTimeout(secs) => (
            "total_timeout_error",
            format!("Response exceeded {}s total timeout", secs),
        ),
        GatewayError::AllProvidersFailed(agg) => {
            let any_attempt_contains = |needle: &str| {
                agg.attempts
                    .iter()
                    .any(|attempt| attempt.error.contains(needle))
            };

            if any_attempt_contains(TTFB_SIGNATURE) {
                (
                    "ttfb_timeout_error",
                    "Provider did not respond before the time-to-first-byte timeout".to_string(),
                )
            } else if any_attempt_contains(TOTAL_SIGNATURE) {
                (
                    "total_timeout_error",
                    "Response exceeded the total timeout".to_string(),
                )
            } else {
                ("stream_error", e.to_string())
            }
        }
        other => ("stream_error", other.to_string()),
    }
}

/// Combine the streaming pass-through attempts recorded by the handler's
/// failover loop with the error returned by the final
/// `route_request_streaming_excluding` call (task 6.3, Req 4.3).
///
/// If that error is itself a `GatewayError::AllProvidersFailed` aggregate (the
/// buffered fallback's own failover produced one), its attempts are appended so
/// the client sees a single, complete list of every provider that failed.
/// Otherwise the error is wrapped as a synthetic attempt, preserving any
/// provider status code. The result is always an `AllProvidersFailed` error so
/// `classify_stream_error` can recover timeout kinds from the merged attempts.
fn merge_streaming_attempts(
    mut attempts: Vec<ProviderAttempt>,
    error: GatewayError,
) -> GatewayError {
    match error {
        GatewayError::AllProvidersFailed(agg) => {
            attempts.extend(agg.attempts);
        }
        other => {
            let status = match &other {
                GatewayError::Provider { status_code, .. } => *status_code,
                _ => None,
            };
            attempts.push(ProviderAttempt::new(
                "streaming-failover".to_string(),
                String::new(),
                other.to_string(),
                status,
            ));
        }
    }
    GatewayError::AllProvidersFailed(AggregatedError::new(attempts))
}

/// Decision for a single SSE `data:` payload encountered during true streaming
/// pass-through relay (task 5.3, Req 3.2, 3.3, 3.6).
///
/// Factored out of [`relay_passthrough_stream`] so the per-line parsing/
/// validation rules are unit-testable without constructing a live
/// `reqwest::Response`.
#[derive(Debug, PartialEq)]
enum RelayLineAction {
    /// Well-formed chunk (has a `choices` array) — forward the payload verbatim.
    Forward,
    /// Payload was not valid JSON — skip with a warning (Req 3.3).
    SkipMalformed,
    /// Valid JSON but not a recognizable chunk (no `choices`, no `error`) — skip quietly.
    SkipNonChunk,
    /// The upstream `[DONE]` sentinel — stop reading; we emit our own `[DONE]`.
    Done,
    /// Mid-stream error frame (top-level `error` object or `finish_reason == "error"`).
    /// Carries the upstream message for the graceful SSE error event (Req 3.6, Req 5).
    Error(String),
}

/// Classify a single SSE `data:` payload (already stripped of the `data:`
/// prefix and trimmed) into a [`RelayLineAction`] (Req 3.2, 3.3, 3.6).
fn classify_relay_line(payload: &str) -> RelayLineAction {
    if payload == "[DONE]" {
        return RelayLineAction::Done;
    }

    let value: serde_json::Value = match serde_json::from_str(payload) {
        Ok(v) => v,
        // Req 3.3: malformed chunks are skipped with a warning (logged by caller).
        Err(_) => return RelayLineAction::SkipMalformed,
    };

    // Req 3.6 / Req 5: a top-level `error` object is a mid-stream failure.
    if let Some(err) = value.get("error") {
        let message = err
            .get("message")
            .and_then(|m| m.as_str())
            .unwrap_or("Provider returned a stream error")
            .to_string();
        return RelayLineAction::Error(message);
    }

    // Req 3.6: `finish_reason: "error"` on the first choice is also a failure.
    let finish_is_error = value
        .get("choices")
        .and_then(|c| c.as_array())
        .and_then(|choices| choices.first())
        .and_then(|choice| choice.get("finish_reason"))
        .and_then(|fr| fr.as_str())
        == Some("error");
    if finish_is_error {
        return RelayLineAction::Error(
            "Provider signaled a mid-stream error (finish_reason=error)".to_string(),
        );
    }

    if value.get("choices").is_some() {
        RelayLineAction::Forward
    } else {
        // Valid JSON without choices (e.g. a stray usage-only frame the client
        // does not expect mid-stream). Skip quietly; accumulation/usage handling
        // is task 5.4's responsibility.
        RelayLineAction::SkipNonChunk
    }
}

/// Outcome of a single true-streaming pass-through relay, written by
/// [`relay_passthrough_stream`] through a shared handle and read by the
/// streaming handler's failover loop (task 6.1, Req 4.1, 4.2).
///
/// - `Completed`: the upstream stream finished cleanly (`[DONE]` or
///   end-of-stream with no error). The relay emitted its terminal `[DONE]`.
/// - `FailedBeforeContent`: the provider disconnected/errored/timed out before
///   any content or tool_call delta reached the client. The relay stayed silent
///   (no error event, no `[DONE]`) so the handler can transparently retry the
///   next provider without confusing the client (Req 4.1, 4.4).
/// - `FailedAfterContent`: the provider failed after content was already
///   forwarded. The relay emitted a graceful error event + `[DONE]`; the
///   handler records the failure (cb + metrics) and must NOT retry (Req 4.2,
///   4.5).
#[derive(Debug, Clone)]
enum RelayOutcome {
    /// Clean finish. Carries the reassembled usage from the streamed chunks
    /// (zeros when the provider omitted usage frames) so the handler can
    /// feed success accounting — circuit-breaker recovery, latency, and
    /// cost accrual — via `Router::record_streaming_success`.
    Completed {
        usage: Usage,
    },
    FailedBeforeContent(String),
    /// The provider failed AFTER content was already forwarded. The relay
    /// emitted a graceful error event + `[DONE]`; the handler records the
    /// failure (cb + metrics) and must NOT retry (Req 4.2, 4.5). Carries a
    /// human-readable reason for the dashboard/log.
    FailedAfterContent(String),
}

/// True iff an SSE chunk payload carries a content / tool_call / reasoning
/// delta that reaches the client (task 6.1). A role-only delta (e.g. the
/// upstream's first `{"delta":{"role":"assistant"}}` chunk) does NOT count —
/// that is idempotent with our early event and does not block pre-content
/// failover (Req 4.1, 4.4).
fn chunk_carries_content(payload: &str) -> bool {
    let value: serde_json::Value = match serde_json::from_str(payload) {
        Ok(v) => v,
        Err(_) => return false,
    };
    let delta = value
        .get("choices")
        .and_then(|c| c.as_array())
        .and_then(|choices| choices.first())
        .and_then(|choice| choice.get("delta"));
    let Some(delta) = delta else {
        return false;
    };

    // Non-empty textual content.
    if delta
        .get("content")
        .and_then(|v| v.as_str())
        .is_some_and(|s| !s.is_empty())
    {
        return true;
    }
    // Any tool_call delta (function name / arguments fragments).
    if delta
        .get("tool_calls")
        .and_then(|v| v.as_array())
        .is_some_and(|a| !a.is_empty())
    {
        return true;
    }
    // Non-empty reasoning / reasoning_content (thinking models).
    for key in ["reasoning", "reasoning_content"] {
        if delta
            .get(key)
            .and_then(|v| v.as_str())
            .is_some_and(|s| !s.is_empty())
        {
            return true;
        }
    }
    false
}

fn reqwest_error_chain(error: &reqwest::Error) -> String {
    let mut messages = Vec::new();
    let mut current: Option<&(dyn StdError + 'static)> = Some(error);
    while let Some(cause) = current {
        let message = cause.to_string();
        if messages.last() != Some(&message) {
            messages.push(message);
        }
        current = cause.source();
    }
    messages.join(": ")
}

fn upstream_stream_metadata(
    upstream: &reqwest::Response,
) -> (reqwest::Version, String, String, Option<u64>) {
    let version = upstream.version();
    let content_encoding = upstream
        .headers()
        .get(header::CONTENT_ENCODING)
        .and_then(|value| value.to_str().ok())
        .unwrap_or("identity")
        .to_owned();
    let transfer_encoding = upstream
        .headers()
        .get(header::TRANSFER_ENCODING)
        .and_then(|value| value.to_str().ok())
        .unwrap_or("none")
        .to_owned();
    (
        version,
        content_encoding,
        transfer_encoding,
        upstream.content_length(),
    )
}

/// Relay a true-streaming pass-through upstream response to the client as SSE
/// events (task 5.3, Req 3.2, 3.3, 3.6, 3.11, 3.12).
///
/// Reads the upstream `reqwest::Response` body as a byte stream, reassembles SSE
/// lines across arbitrary chunk boundaries (handling `\n` and `\r\n`, multiple
/// `data:` lines per chunk, and lines spanning chunks), validates each
/// `data: {...}` payload as JSON, and forwards well-formed chunks verbatim. The
/// payload `id`/`model` are intentionally **not** rewritten here — early-event
/// id reconciliation is decided by the wiring in task 5.5.
///
/// Timeouts (both emitted as graceful SSE error events followed by `[DONE]`):
/// - **Inter-chunk** (`chunk_timeout_seconds`, Req 3.12): max gap between data events.
/// - **Total** (`total_timeout`, Req 3.11): cap on the whole streaming duration.
///
/// The per-iteration wait is `min(chunk_timeout, remaining_total)`; when the wait
/// elapses we compare against the deadline to attribute it to the correct timeout.
///
/// Always terminates the stream with exactly one `[DONE]`: emitted directly on a
/// clean finish, or via [`emit_sse_error_event`] (which appends its own `[DONE]`)
/// on a post-content error/timeout path.
///
/// ## Pre-content failover signal (task 6.1, Req 4.1, 4.2, 4.4)
///
/// The relay tracks whether any content/tool_call/reasoning delta has reached
/// the client ([`chunk_carries_content`]). On any failure (network error,
/// mid-stream error frame, inter-chunk/total timeout) it branches on that flag:
///
/// - **Before content**: stays completely silent — no error event, no `[DONE]` —
///   and records [`RelayOutcome::FailedBeforeContent`] through the shared
///   `outcome` handle so the handler can transparently retry the next provider
///   reusing the same early-event id (Req 4.1, 4.4).
/// - **After content**: keeps the existing behavior — emits a graceful error
///   event + `[DONE]` and records [`RelayOutcome::FailedAfterContent`] (task 6.2
///   refines this path).
///
/// A clean finish records [`RelayOutcome::Completed`].
///
/// ## Background accumulation for caching (task 5.4, Req 3.7, 3.10)
///
/// While relaying, every forwarded chunk payload is appended to an in-memory SSE
/// buffer. On a **clean** completion only (upstream `[DONE]` or end-of-stream with
/// no error/timeout), the buffer is reassembled into a full [`OpenAIResponse`] via
/// [`Router::reassemble_sse_response`]. If the assembled response is cacheable
/// (`Router::should_cache_response`, Req 3.10) it is written to the exact cache
/// keyed by `request`. Usage extracted from the final chunk is recorded in
/// metrics/logging (Req 3.7). Partial or errored streams are never cached.
///
// Wired into `chat_completions_stream` by task 5.5; failover wiring task 6.1.
/// Inputs the streaming relay needs to adaptively learn that a provider/model
/// combo emits XML-style tool calls. When `Some` (the request carried `tools`),
/// a clean pass-through completion is scanned for XML tool use; if found, the
/// combo is recorded via [`Router::mark_xml_tool_combo`] so future tool requests
/// for it take the buffer-and-translate path.
struct XmlToolDetect {
    router: Arc<crate::router::router::Router>,
    provider: String,
    model: String,
}

fn relay_passthrough_stream(
    upstream: reqwest::Response,
    streaming_config: StreamingConfig,
    trace_id: String,
    total_timeout: Duration,
    exact_cache: Arc<ExactCache>,
    metrics: Arc<Metrics>,
    request: OpenAIRequest,
    outcome: Arc<tokio::sync::Mutex<RelayOutcome>>,
    xml_detect: Option<XmlToolDetect>,
    memory_suffix: Option<String>,
) -> impl futures::Stream<Item = Result<Event, Infallible>> {
    async_stream::stream! {
    let chunk_timeout = Duration::from_secs(streaming_config.chunk_timeout_seconds);
    let deadline = tokio::time::Instant::now() + total_timeout;
    let (
    upstream_version,
    upstream_content_encoding,
    upstream_transfer_encoding,
    upstream_content_length,
    ) = upstream_stream_metadata(&upstream);
    let mut byte_stream = upstream.bytes_stream();

    let mut buffer = String::new();
    let mut bytes_received = 0usize;


           // Req 3.10: accumulate forwarded chunk payloads into an SSE buffer so a
           // clean completion can be reassembled into a cacheable response.
           let mut sse_accumulator = String::new();

           // Task 6.1: track whether any content/tool_call/reasoning delta has been
           // forwarded. Drives the pre- vs post-content failure branch below.
           let mut content_forwarded = false;

           // `terminated` => the relay already wrote its own terminal frame(s) — a
           // graceful error event (post-content, appends `[DONE]`) OR deliberate
           // silence (pre-content failover) — so we must NOT emit a final `[DONE]`.
           let mut terminated = false;

           'relay: loop {
               // Req 3.11: enforce the overall streaming budget.
               let now = tokio::time::Instant::now();
               if now >= deadline {
                   let message = format!("Response exceeded {}s total timeout", total_timeout.as_secs());
                   if content_forwarded {
                       for event in emit_sse_error_event("total_timeout_error", &message, &trace_id) {
                           yield Ok(event);
                       }
                       *outcome.lock().await = RelayOutcome::FailedAfterContent(message);
                   } else {
                       // Req 4.1: stay silent so the handler can fail over.
                       *outcome.lock().await = RelayOutcome::FailedBeforeContent(message);
                   }
                   terminated = true;
                   break 'relay;
               }
               let per_chunk_wait = chunk_timeout.min(deadline - now);

               let mut stream_ended = false;
               match tokio::time::timeout(per_chunk_wait, byte_stream.next()).await {
    Ok(Some(Ok(bytes))) => {
    bytes_received = bytes_received.saturating_add(bytes.len());
    buffer.push_str(&String::from_utf8_lossy(&bytes));
    }
    Ok(Some(Err(e))) => {
    // Reqwest labels every HTTP body/frame failure as a decode error,
    // including HTTP/1 truncation and HTTP/2 resets. Preserve the source
    // chain so the client and logs expose the actual transport failure.
    let error_chain = reqwest_error_chain(&e);
    let message = format!("Stream error: {error_chain}");
    tracing::warn!(
    trace_id = %trace_id,
    error = %e,
    error_debug = ?e,
    error_chain = %error_chain,
    http_version = ?upstream_version,
    content_encoding = %upstream_content_encoding,
    transfer_encoding = %upstream_transfer_encoding,
    content_length = ?upstream_content_length,
    bytes_received,
    content_forwarded,
    "Upstream streaming response body failed"
    );

                       if content_forwarded {
                           for event in emit_sse_error_event("stream_error", &message, &trace_id) {
                               yield Ok(event);
                           }
                           *outcome.lock().await = RelayOutcome::FailedAfterContent(message);
                       } else {
                           // Req 4.1: silent pre-content failure → handler retries.
                           *outcome.lock().await = RelayOutcome::FailedBeforeContent(message);
                       }
                       terminated = true;
                       break 'relay;
                   }
                   Ok(None) => {
                       // Upstream finished — flush any trailing partial line by
                       // forcing the line-drain loop to process the remainder.
                       stream_ended = true;
                       if !buffer.is_empty() && !buffer.ends_with('\n') {
                           buffer.push('\n');
                       }
                   }
                   Err(_) => {
                       // Distinguish total-timeout (Req 3.11) from inter-chunk
                       // timeout (Req 3.12): the wait was min(chunk, remaining_total).
                       let (error_type, message) = if tokio::time::Instant::now() >= deadline {
                           (
                               "total_timeout_error",
                               format!("Response exceeded {}s total timeout", total_timeout.as_secs()),
                           )
                       } else {
                           (
                               "chunk_timeout_error",
                               format!(
                                   "Provider stopped sending data for {}s",
                                   streaming_config.chunk_timeout_seconds
                               ),
                           )
                       };
                       if content_forwarded {
                           for event in emit_sse_error_event(error_type, &message, &trace_id) {
                               yield Ok(event);
                           }
                           *outcome.lock().await = RelayOutcome::FailedAfterContent(message);
                       } else {
                           // Req 4.1: silent pre-content timeout → handler retries.
                           *outcome.lock().await = RelayOutcome::FailedBeforeContent(message);
                       }
                       terminated = true;
                       break 'relay;
                   }
               }

               // Drain all complete lines currently in the buffer.
               while let Some(newline_idx) = buffer.find('\n') {
                   let raw: String = buffer.drain(..=newline_idx).collect();
                   let line = raw.trim_end_matches(|c| c == '\n' || c == '\r');
                   if line.is_empty() {
                       continue; // SSE frame separator / blank line.
                   }

                   // Only `data:` lines carry payloads; skip `event:`/`id:`/`:comment`.
                   let payload = match line.strip_prefix("data:") {
                       Some(rest) => rest.trim_start(),
                       None => continue,
                   };

                   match classify_relay_line(payload) {
                       RelayLineAction::Forward => {
                           // Req 3.2: forward the validated chunk verbatim.
                           // Req 3.10: also retain it for background reassembly so a
                           // clean completion can be cached.
                           // Task 6.1: once a real content delta is forwarded,
                           // pre-content failover is no longer possible.
                           if !content_forwarded && chunk_carries_content(payload) {
                               content_forwarded = true;
                           }
                           sse_accumulator.push_str("data: ");
                           sse_accumulator.push_str(payload);
                           sse_accumulator.push_str("\n\n");
                           yield Ok(Event::default().data(payload.to_string()));
                       }
                       RelayLineAction::SkipMalformed => {
                           tracing::warn!(
                               trace_id = %trace_id,
                               "Skipping malformed SSE chunk from provider"
                           );
                       }
                       RelayLineAction::SkipNonChunk => {
                           tracing::debug!(
                               trace_id = %trace_id,
                               "Skipping non-chunk SSE data frame (no choices)"
                           );
                       }
                       RelayLineAction::Done => {
                           // Upstream `[DONE]`: stop reading; we emit our own below.
                           break 'relay;
                       }
                       RelayLineAction::Error(message) => {
                           if content_forwarded {
                               for event in emit_sse_error_event("stream_error", &message, &trace_id) {
                                   yield Ok(event);
                               }
                               *outcome.lock().await = RelayOutcome::FailedAfterContent(message);
                           } else {
                               // Req 4.1: silent pre-content error frame → retry.
                               *outcome.lock().await = RelayOutcome::FailedBeforeContent(message);
                           }
                           terminated = true;
                           break 'relay;
                       }
                   }
               }

               if stream_ended {
                   break 'relay;
               }
           }

           // Req 3.6: always terminate with `[DONE]`, unless an error path already
           // appended one via `emit_sse_error_event`.
           if !terminated {
               // Guard: if the upstream closed cleanly but never sent any content
               // delta (e.g. provider returned HTTP 200 then immediately closed,
               // sent only role-only/empty frames, or only non-data lines), treat
               // this as a pre-content failure so the failover loop can retry the
               // next provider. Without this, the client would see only the early
               // event (+ maybe a duplicate role delta) + [DONE] — an apparently
               // empty response with no error.
               if !content_forwarded {
                   tracing::warn!(
                       trace_id = %trace_id,
                       sse_accumulator_len = sse_accumulator.len(),
                       "Upstream closed cleanly but sent no content delta; treating as pre-content failure for failover"
                   );
                   *outcome.lock().await = RelayOutcome::FailedBeforeContent(
                       "Provider stream ended without sending any content".to_string(),
                   );
                   // Stay silent (no error event, no [DONE]) so handler can retry.
                   // Any role-only chunks we already forwarded are idempotent with
                   // the early event and won't confuse the client on retry.
                   return;
               }

               // Req 3.7 / 3.10: clean completion — reassemble the accumulated chunks
               // into a full response, cache it if eligible, and capture the
               // reported usage for success accounting.
               let mut completed_usage = Usage::default();
               if !sse_accumulator.is_empty() {
                   match crate::router::router::Router::reassemble_sse_response(&sse_accumulator) {
                       Ok(assembled) => {
                           // Capture the reassembled usage (zeros when the
                           // provider omitted usage frames) so the handler can
                           // run cost accrual via record_streaming_success.
                           completed_usage = assembled.usage.clone();
                           // Adaptive XML-tool detection (only when the request
                           // carried `tools`): if this provider/model streamed
                           // XML-style tool calls in plain text instead of native
                           // `tool_calls`, learn the combo so the NEXT tool request
                           // for it takes the buffer-and-translate path. This one
                           // request still streamed the raw XML — learning is for
                           // subsequent requests.
                           if let Some(det) = xml_detect.as_ref() {
                               let choice = assembled.choices.first();
                               let has_native_tc = choice
                                   .map(|c| c.message.extra.contains_key("tool_calls"))
                                   .unwrap_or(false);
                               let content_text = choice
                                   .map(|c| c.message.content_as_text())
                                   .unwrap_or_default();
                   if !has_native_tc
                       && crate::router::router::Router::looks_like_xml_tool_use(&content_text)
                   {
                       det.router.mark_xml_tool_combo(&det.provider, &det.model);
                       tracing::warn!(
                           trace_id = %trace_id,
                           provider = %det.provider,
                           model = %det.model,
                           "Detected XML-style tool use in streamed response; future tool requests for this provider/model will use the buffered translate path"
                       );
                   } else if has_native_tc {
                       // Mirror of the buffered-path diagnostic: native
                       // tool_calls count toward forgiving a learned XML
                       // combo (hint injection + buffer-and-translate stand
                       // down after TOOL_HINT_RECOVERY_SUCCESSES).
                       det.router
                           .record_native_tool_success(&det.provider, &det.model);
                   }
                           }
                           // Req 3.7: surface usage from the final chunk in logs.
                           tracing::info!(
                               trace_id = %trace_id,
                               prompt_tokens = assembled.usage.prompt_tokens,
                               completion_tokens = assembled.usage.completion_tokens,
                               total_tokens = assembled.usage.total_tokens,
                               "Streaming pass-through completed; recorded usage"
                           );
                           // Req 3.10: cache only responses safe to replay (gate
                           // identical to the buffer-and-replay path).
                           if crate::router::router::Router::should_cache_response(&assembled) {
                               if let Ok(json) = serde_json::to_string(&assembled) {
                                   if !json.is_empty() {
                                       exact_cache.set(&request, json);
                                       tracing::debug!(
                                           trace_id = %trace_id,
                                           "Cached reassembled streaming response in exact cache"
                                       );
                                   }
                               }
                           }
                           // Touch metrics so the dependency is exercised even when the
                           // response is not cost-attributable here (no provider cost
                           // rates in the relay path); usage is logged above per Req 3.7.
                           // Task 5.5 may extend this to record provider-scoped cost.
                           let _ = &metrics;
                       }
                       Err(e) => {
                           // Reassembly failure is non-fatal for the client (the stream
                           // already completed); just skip caching.
                           tracing::warn!(
                               trace_id = %trace_id,
                               error = %e,
                               "Failed to reassemble streaming response for caching"
                           );
                       }
                   }
               }
               // Task 6.1: a clean finish — record the outcome (carrying the
               // reassembled usage, zeros when unavailable) so the handler's
               // failover loop stops here (no retry) and can run success
               // accounting via `record_streaming_success`.
               *outcome.lock().await = RelayOutcome::Completed {
                   usage: completed_usage,
               };
               if let Some(suffix) = memory_suffix.as_deref() {
                   yield Ok(Event::default().data(memory_feedback_chunk(&request, suffix).to_string()));
               }
               yield Ok(Event::default().data("[DONE]"));
           }
       }
}

pub(crate) fn streaming_chunks_from_response(response: &OpenAIResponse) -> Vec<serde_json::Value> {
    let response = prepare_response_for_client(response);
    build_streaming_chunks(&response, None)
}

/// Variant used after an early synthetic `role: assistant` event has already
/// been emitted (Req 1.5). It suppresses the duplicate role delta and reuses
/// the early event's pre-generated `id`/`created` for every subsequent chunk so
/// the whole stream shares a single id (task 2.2).
pub(crate) fn streaming_chunks_after_early_event(
    response: &OpenAIResponse,
    id: &str,
    created: i64,
) -> Vec<serde_json::Value> {
    let response = prepare_response_for_client(response);
    build_streaming_chunks(&response, Some((id, created)))
}

/// Core chunk synthesizer. When `early_event` is `Some`, the leading
/// `role: assistant` delta is skipped (the early event already sent it) and the
/// supplied `id`/`created` override the provider response's own envelope values
/// so all chunks line up with the early event. When `None` (cache replay and
/// the `emit_early_event: false` path), the role chunk is emitted and the
/// provider's `id`/`created` are used unchanged.
fn build_streaming_chunks(
    response: &OpenAIResponse,
    early_event: Option<(&str, i64)>,
) -> Vec<serde_json::Value> {
    let skip_role = early_event.is_some();

    // Override the envelope `id`/`created` with the early-event values by
    // operating on an owned copy, so the existing builders (which read these
    // from `response`) emit the shared id without further plumbing.
    let owned;
    let response: &OpenAIResponse = match early_event {
        Some((id, created)) => {
            let mut overridden = response.clone();
            overridden.id = id.to_string();
            overridden.created = created;
            owned = overridden;
            &owned
        }
        None => response,
    };

    let choice = response.choices.first();

    let content = choice
        .map(|c| match &c.message.content {
            serde_json::Value::String(s) => serde_json::Value::String(s.clone()),
            serde_json::Value::Null => serde_json::Value::Null,
            other => other.clone(),
        })
        .unwrap_or(serde_json::Value::Null);

    let tool_calls = choice
        .and_then(|c| c.message.extra.get("tool_calls"))
        .and_then(|v| v.as_array())
        .cloned();

    let has_tool_calls = tool_calls.as_ref().is_some_and(|tc| !tc.is_empty());
    let reasoning_delta = reasoning_delta(choice);
    let mut chunks = Vec::new();

    if has_tool_calls {
        let tcs = tool_calls.as_ref().unwrap();
        let first_tc = &tcs[0];
        let tc_id = first_tc.get("id").and_then(|v| v.as_str()).unwrap_or("");
        let tc_type = first_tc
            .get("type")
            .and_then(|v| v.as_str())
            .unwrap_or("function");
        let fn_name = first_tc
            .get("function")
            .and_then(|f| f.get("name"))
            .and_then(|n| n.as_str())
            .unwrap_or("");

        // The first tool-call chunk also carries the `role` marker. When the
        // early event already sent it, drop just the role field and keep the
        // tool-call metadata (Req 1.5).
        let mut first_delta = serde_json::json!({
            "content": null,
            "tool_calls": [{
                "index": 0,
                "id": tc_id,
                "type": tc_type,
                "function": {
                    "name": fn_name,
                    "arguments": ""
                }
            }]
        });
        if !skip_role {
            first_delta["role"] = serde_json::json!("assistant");
        }
        chunks.push(build_chunk_payload(response, first_delta, None));

        if let Some(delta) = reasoning_delta.clone() {
            chunks.push(build_chunk_payload(response, delta, None));
        }

        let fn_args = first_tc
            .get("function")
            .and_then(|f| f.get("arguments"))
            .and_then(|a| a.as_str())
            .unwrap_or("{}");
        chunks.push(build_chunk_payload(
            response,
            serde_json::json!({
                "tool_calls": [{
                    "index": 0,
                    "function": { "arguments": fn_args }
                }]
            }),
            None,
        ));

        for (i, tc) in tcs.iter().enumerate().skip(1) {
            let tc_id = tc.get("id").and_then(|v| v.as_str()).unwrap_or("");
            let tc_type = tc
                .get("type")
                .and_then(|v| v.as_str())
                .unwrap_or("function");
            let fn_name = tc
                .get("function")
                .and_then(|f| f.get("name"))
                .and_then(|n| n.as_str())
                .unwrap_or("");
            let fn_args = tc
                .get("function")
                .and_then(|f| f.get("arguments"))
                .and_then(|a| a.as_str())
                .unwrap_or("{}");

            chunks.push(build_chunk_payload(
                response,
                serde_json::json!({
                    "tool_calls": [{
                        "index": i,
                        "id": tc_id,
                        "type": tc_type,
                        "function": {
                            "name": fn_name,
                            "arguments": ""
                        }
                    }]
                }),
                None,
            ));

            chunks.push(build_chunk_payload(
                response,
                serde_json::json!({
                    "tool_calls": [{
                        "index": i,
                        "function": { "arguments": fn_args }
                    }]
                }),
                None,
            ));
        }
    } else {
        if !skip_role {
            chunks.push(build_chunk_payload(
                response,
                serde_json::json!({ "role": "assistant", "content": "" }),
                None,
            ));
        }

        if let Some(delta) = reasoning_delta {
            chunks.push(build_chunk_payload(response, delta, None));
        }

        if !content.is_null() && content.as_str().map(|s| !s.is_empty()).unwrap_or(true) {
            chunks.push(build_chunk_payload(
                response,
                serde_json::json!({ "content": content }),
                None,
            ));
        }
    }

    let finish_reason = if has_tool_calls {
        "tool_calls"
    } else {
        choice
            .and_then(|c| c.finish_reason.as_deref())
            .unwrap_or("stop")
    };
    chunks.push(serde_json::json!({
        "id": response.id,
        "object": "chat.completion.chunk",
        "created": response.created,
        "model": response.model,
        "choices": [{
            "index": 0,
            "delta": {},
            "finish_reason": finish_reason
        }],
        "usage": response.usage
    }));

    chunks
}

fn build_chunk_payload(
    response: &OpenAIResponse,
    delta: serde_json::Value,
    finish_reason: Option<&str>,
) -> serde_json::Value {
    serde_json::json!({
        "id": response.id,
        "object": "chat.completion.chunk",
        "created": response.created,
        "model": response.model,
        "choices": [{
            "index": 0,
            "delta": delta,
            "finish_reason": finish_reason
        }]
    })
}

fn reasoning_delta(choice: Option<&Choice>) -> Option<serde_json::Value> {
    let choice = choice?;

    for field in ["reasoning", "reasoning_content"] {
        let Some(value) = choice.message.extra.get(field) else {
            continue;
        };
        if value.is_null() || value.as_str().is_some_and(|s| s.is_empty()) {
            continue;
        }

        let mut delta = serde_json::Map::new();
        delta.insert(field.to_string(), value.clone());
        return Some(serde_json::Value::Object(delta));
    }

    None
}

#[cfg(test)]
mod tests {
    use super::{
        append_feedback_to_response, attach_smart_routing_headers, attach_validation_status_header,
        build_keepalive, cache_allowed_for_validation, chunk_carries_content, classify_relay_line,
        classify_stream_error, collect_structured_output_failure, eager_sse_response,
        early_event_chunk, emit_sse_error_event, force_eager_structured_stream, json_model,
        memory_feedback_chunk, multipart_model, openai_json_response, prepare_response_for_client,
        provider_pass_through_response, rechunk_structured_response, relay_passthrough_stream,
        requests_structured_output, should_cache_eager_structured, smart_routing_headers,
        sse_error_payload, streaming_chunks_after_early_event, streaming_chunks_from_response,
        structured_stream_overflow_events, upstream_stream_metadata, RelayLineAction, RelayOutcome,
        RequestCompleteGuard, RequestLogContext, ValidationResponseStatus,
    };
    use crate::compression::{stats::CompressionStats, CompressionLevel};
    use crate::config::StreamingConfig;
    use crate::error::{AggregatedError, GatewayError, ProviderAttempt};
    use crate::memory::{ContextType, ExtractionCounts, InjectionResult};
    use crate::metrics::Metrics;
    use crate::models::openai::{Choice, Message, OpenAIRequest, OpenAIResponse, Usage};
    use crate::router::router::ProviderPassThroughResponse;
    use crate::smart_routing::tier::{
        ClassifierUsed, ComplexityScore, RoutingDecision, SmartRoutingTier, TaskType,
    };
    use crate::structured_output::validator::{
        ChoiceValidationOutcome, ChoiceValidationResult, SchemaViolation,
    };
    use crate::structured_output::StructuredOutputOutcome;
    use axum::response::{IntoResponse, Response};
    use axum::Json;
    use futures::StreamExt;
    use proptest::prelude::*;
    use std::sync::Arc;
    use std::time::{Duration, Instant};

    #[test]
    fn pass_through_extracts_models_without_mutating_bodies() {
        let json = br#"{"model":"embedding-group","input":["hello"]}"#;
        assert_eq!(json_model(json).unwrap(), "embedding-group");

        let multipart = b"--raw-boundary\r\nContent-Disposition: form-data; name=\"model\"\r\n\r\naudio-group\r\n--raw-boundary\r\nContent-Disposition: form-data; name=\"file\"; filename=\"sample.wav\"\r\nContent-Type: application/octet-stream\r\n\r\n\x00\xff\x80raw-audio\r\n--raw-boundary--\r\n";
        assert_eq!(
            multipart_model("multipart/form-data; boundary=raw-boundary", multipart).unwrap(),
            "audio-group"
        );
        let binary_offset = multipart
            .windows(3)
            .position(|window| window == b"\x00\xff\x80")
            .expect("binary payload must remain intact");
        assert_eq!(
            &multipart[binary_offset..binary_offset + 3],
            b"\x00\xff\x80"
        );
    }

    #[test]
    fn provider_pass_through_response_applies_upstream_status_and_body() {
        let response = provider_pass_through_response(ProviderPassThroughResponse {
            status: 202,
            headers: reqwest::header::HeaderMap::new(),
            body: br#"{"queued":true}"#.to_vec(),
        });

        assert_eq!(response.status(), axum::http::StatusCode::ACCEPTED);
        let body = futures::executor::block_on(axum::body::to_bytes(response.into_body(), 1024))
            .expect("response body should be readable");
        assert_eq!(body.as_ref(), br#"{"queued":true}"#);
    }

    #[test]
    fn request_guard_drop_completes_active_request() {
        let metrics = Arc::new(Metrics::new());
        metrics.start_request();

        {
            let _guard = RequestCompleteGuard::new(metrics.clone(), Instant::now(), None);
            assert_eq!(metrics.snapshot().active_requests, 1);
        }

        assert_eq!(metrics.snapshot().active_requests, 0);
    }

    fn base_response(message: Message) -> OpenAIResponse {
        OpenAIResponse {
            id: "chatcmpl-test".to_string(),
            object: "chat.completion".to_string(),
            created: 123,
            model: "test-model".to_string(),
            choices: vec![Choice {
                index: 0,
                message,
                finish_reason: Some("stop".to_string()),
                extra: Default::default(),
            }],
            usage: Usage {
                prompt_tokens: 1,
                completion_tokens: 2,
                total_tokens: 3,
                extra: Default::default(),
            },
            extra: Default::default(),
        }
    }

    fn routing_decision(score: f64) -> RoutingDecision {
        RoutingDecision {
            score: ComplexityScore::new(score),
            adjusted_score: ComplexityScore::new(score),
            tier: SmartRoutingTier::Balanced,
            task_type: TaskType::CodeGeneration,
            classifier: ClassifierUsed::Composite,
            escalated: true,
            escalation_count: 1,
            cache_hit: true,
            budget_downgraded: false,
            context_filtered: false,
        }
    }

    fn test_compression_stats() -> CompressionStats {
        CompressionStats {
            request_id: "route-safe".to_owned(),
            level: CompressionLevel::Standard,
            engines_applied: vec!["standard".to_owned()],
            original_tokens: 100,
            compressed_tokens: 75,
            savings_percent: 25.0,
            compression_time_ms: 4,
            auto_triggered: false,
            cache_downgrade_applied: false,
            tool_definitions_tokens_saved: 0,
            caveman_applied: false,
            timed_out: false,
            error: false,
            provider: "provider".to_owned(),
            model: "model".to_owned(),
            engine_results: Vec::new(),
        }
    }

    fn request_with_response_format(
        stream: bool,
        response_format: serde_json::Value,
    ) -> OpenAIRequest {
        let mut request = OpenAIRequest {
            model: "test-model".to_owned(),
            messages: Vec::new(),
            stream,
            temperature: None,
            max_tokens: None,
            extra: Default::default(),
        };
        request
            .extra
            .insert("response_format".to_owned(), response_format);
        request
    }

    fn validation_outcome_strategy() -> impl Strategy<Value = StructuredOutputOutcome> {
        prop_oneof![
            Just(StructuredOutputOutcome::Pass),
            Just(StructuredOutputOutcome::Fail),
            Just(StructuredOutputOutcome::Skipped),
        ]
    }

    fn json_value_strategy() -> impl Strategy<Value = serde_json::Value> {
        let leaf = prop_oneof![
            Just(serde_json::Value::Null),
            any::<bool>().prop_map(serde_json::Value::Bool),
            any::<i64>().prop_map(|value| serde_json::json!(value)),
            prop::collection::vec(any::<char>(), 0..64)
                .prop_map(|characters| serde_json::Value::String(characters.into_iter().collect())),
        ];

        leaf.prop_recursive(4, 64, 8, |inner| {
            let key = prop::collection::vec(any::<char>(), 0..24)
                .prop_map(|characters| characters.into_iter().collect::<String>());
            prop_oneof![
                prop::collection::vec(inner.clone(), 0..8).prop_map(serde_json::Value::Array),
                prop::collection::btree_map(key, inner, 0..8).prop_map(|entries| {
                    serde_json::Value::Object(entries.into_iter().collect())
                }),
            ]
        })
    }

    #[derive(Clone, Copy, Debug)]
    enum SkipScenario {
        SchemaCompilationFailure,
        InternalError,
        Timeout,
    }

    fn skip_scenario_strategy() -> impl Strategy<Value = SkipScenario> {
        prop_oneof![
            Just(SkipScenario::SchemaCompilationFailure),
            Just(SkipScenario::InternalError),
            Just(SkipScenario::Timeout),
        ]
    }

    fn response_body_bytes(response: Response) -> Vec<u8> {
        futures::executor::block_on(axum::body::to_bytes(response.into_body(), usize::MAX))
            .expect("in-memory response body must be readable")
            .to_vec()
    }

    fn assert_skipped_header_is_only_header_change(
        before: &axum::http::HeaderMap,
        response: &Response,
    ) {
        assert_eq!(response.headers().len(), before.len() + 1);
        for (name, value) in before {
            assert_eq!(response.headers().get(name), Some(value));
        }
        assert_eq!(
            response
                .headers()
                .get("x-obey-validation-status")
                .and_then(|value| value.to_str().ok()),
            Some("skipped")
        );
    }

    // Feature: structured-output-validation, Property 12: Cache Write Gating
    proptest! {
        #![proptest_config(ProptestConfig::with_cases(64))]

        #[test]
        fn prop_cache_write_gating(
            outcome in validation_outcome_strategy(),
            otherwise_cacheable in any::<bool>(),
        ) {
            let request = request_with_response_format(
                false,
                serde_json::json!({
                    "type": "json_schema",
                    "json_schema": {"schema": {"type": "object"}},
                }),
            );
            let status = ValidationResponseStatus::from_outcome(outcome);
            let mut response = base_response(Message {
                role: "assistant".to_owned(),
                content: serde_json::json!({"arbitrary": "body"}),
                extra: Default::default(),
            });
            response.choices[0].finish_reason = Some(
                if otherwise_cacheable { "stop" } else { "length" }.to_owned(),
            );

            let router_allows = crate::router::router::Router::should_cache_response(&response);
            let cache_write_allowed =
                cache_allowed_for_validation(&request, status) && router_allows;

            prop_assert_eq!(router_allows, otherwise_cacheable);
            prop_assert_eq!(
                cache_write_allowed,
                outcome == StructuredOutputOutcome::Pass && otherwise_cacheable,
            );
        }
    }

    // Feature: structured-output-validation, Property 13: Skip Scenario Identity
    proptest! {
        #![proptest_config(ProptestConfig::with_cases(64))]

        #[test]
        fn prop_skip_scenario_identity(
            scenario in skip_scenario_strategy(),
            json_body in json_value_strategy(),
            utf8_characters in prop::collection::vec(any::<char>(), 0..512),
        ) {
            let expected_json = serde_json::to_vec(&json_body)
                .expect("generated JSON body must serialize");
            let mut json_response = Json(json_body).into_response();
            let json_headers_before = json_response.headers().clone();
            attach_validation_status_header(
                &mut json_response,
                ValidationResponseStatus::Skipped,
                true,
            );
            assert_skipped_header_is_only_header_change(&json_headers_before, &json_response);
            prop_assert_eq!(
                response_body_bytes(json_response),
                expected_json,
                "JSON body changed for {:?}",
                scenario,
            );

            let expected_utf8 = utf8_characters.into_iter().collect::<String>().into_bytes();
            let mut byte_response = Response::new(axum::body::Body::from(expected_utf8.clone()));
            let byte_headers_before = byte_response.headers().clone();
            attach_validation_status_header(
                &mut byte_response,
                ValidationResponseStatus::Skipped,
                true,
            );
            assert_skipped_header_is_only_header_change(&byte_headers_before, &byte_response);
            prop_assert_eq!(
                response_body_bytes(byte_response),
                expected_utf8,
                "UTF-8 body bytes changed for {:?}",
                scenario,
            );
        }
    }

    #[test]
    fn structured_streaming_forces_eager_buffering_only_for_json_schema() {
        let mut request = request_with_response_format(
            true,
            serde_json::json!({"type": "json_schema", "json_schema": {"schema": {"type": "object"}}}),
        );
        assert!(force_eager_structured_stream(&request));

        request.stream = false;
        assert!(!force_eager_structured_stream(&request));
        request.stream = true;
        request.extra.insert(
            "response_format".to_owned(),
            serde_json::json!({"type": "json_object"}),
        );
        assert!(!force_eager_structured_stream(&request));
    }

    #[tokio::test]
    async fn eager_structured_sse_attaches_header_before_complete_body() {
        let response = base_response(Message {
            role: "assistant".to_owned(),
            content: serde_json::json!("{\"id\":1}"),
            extra: Default::default(),
        });
        let events = rechunk_structured_response(&response);
        let response = eager_sse_response(
            events,
            &StreamingConfig::default(),
            "trace-eager",
            Some(ValidationResponseStatus::Passed),
            None,
        );
        assert_eq!(
            response
                .headers()
                .get("x-obey-validation-status")
                .and_then(|value| value.to_str().ok()),
            Some("passed")
        );
        let body = axum::body::to_bytes(response.into_body(), 1024 * 1024)
            .await
            .unwrap();
        let body = String::from_utf8(body.to_vec()).unwrap();
        assert!(body.contains("{\\\"id\\\":1}"));
        assert!(body.contains("data: [DONE]"));
        assert!(body.trim_end().ends_with("data: [DONE]"));
    }

    #[tokio::test]
    async fn structured_overflow_is_terminal_and_skips_validation_header() {
        let request = request_with_response_format(
            true,
            serde_json::json!({"type": "json_schema", "json_schema": {"schema": {"type": "object"}}}),
        );
        assert!(!should_cache_eager_structured(&request, None, None));
        assert!(!should_cache_eager_structured(
            &request,
            None,
            Some(ValidationResponseStatus::Passed)
        ));
        let response = eager_sse_response(
            structured_stream_overflow_events("trace-overflow"),
            &StreamingConfig::default(),
            "trace-overflow",
            None,
            None,
        );
        assert!(response.headers().get("x-obey-validation-status").is_none());
        let body = axum::body::to_bytes(response.into_body(), 1024 * 1024)
            .await
            .unwrap();
        let body = String::from_utf8(body.to_vec()).unwrap();
        assert!(body.contains("structured_output_buffer_overflow"));
        assert!(body.contains("data: [DONE]"));
        assert!(body.trim_end().ends_with("data: [DONE]"));
    }

    #[test]
    fn validation_cache_gate_allows_only_pass_for_applicable_requests() {
        let structured_request = request_with_response_format(
            false,
            serde_json::json!({"type": "json_schema", "json_schema": {"schema": {"type": "object"}}}),
        );
        let non_structured_requests = [
            request_with_response_format(false, serde_json::json!({"type": "json_object"})),
            request_with_response_format(false, serde_json::json!({"type": "text"})),
        ];
        let statuses = [
            ValidationResponseStatus::NotApplicable,
            ValidationResponseStatus::Passed,
            ValidationResponseStatus::Failed,
            ValidationResponseStatus::Skipped,
        ];

        for status in statuses {
            assert_eq!(
                cache_allowed_for_validation(&structured_request, status),
                status == ValidationResponseStatus::Passed
            );
            for request in &non_structured_requests {
                assert!(cache_allowed_for_validation(request, status));
            }
        }
    }

    #[test]
    fn validation_status_centralizes_header_policy() {
        let cases = [
            (StructuredOutputOutcome::NotApplicable, None),
            (StructuredOutputOutcome::Pass, Some("passed")),
            (StructuredOutputOutcome::Fail, Some("failed")),
            (StructuredOutputOutcome::Skipped, Some("skipped")),
        ];

        for (outcome, expected_header) in cases {
            let status = ValidationResponseStatus::from_outcome(outcome);
            assert_eq!(status.header_value(), expected_header);

            let mut response = ().into_response();
            attach_validation_status_header(&mut response, status, true);
            assert_eq!(
                response
                    .headers()
                    .get("x-obey-validation-status")
                    .and_then(|value| value.to_str().ok()),
                expected_header
            );
        }
        let mut response = ().into_response();
        attach_validation_status_header(&mut response, ValidationResponseStatus::Skipped, false);
        assert!(response.headers().get("x-obey-validation-status").is_none());
    }

    #[test]
    fn structured_output_failure_collects_every_failed_choice() {
        let response = OpenAIResponse {
            choices: vec![
                Choice {
                    index: 0,
                    message: Message {
                        role: "assistant".to_owned(),
                        content: serde_json::json!("not-json"),
                        extra: Default::default(),
                    },
                    finish_reason: Some("stop".to_owned()),
                    extra: Default::default(),
                },
                Choice {
                    index: 1,
                    message: Message {
                        role: "assistant".to_owned(),
                        content: serde_json::json!("{\"id\":\"wrong\"}"),
                        extra: Default::default(),
                    },
                    finish_reason: Some("stop".to_owned()),
                    extra: Default::default(),
                },
            ],
            ..base_response(Message {
                role: "assistant".to_owned(),
                content: serde_json::json!(null),
                extra: Default::default(),
            })
        };
        let choices = vec![
            ChoiceValidationOutcome {
                result: ChoiceValidationResult::JsonParseError {
                    byte_offset: 2,
                    expected: "object".to_owned(),
                },
                internal_skip: None,
            },
            ChoiceValidationOutcome {
                result: ChoiceValidationResult::SchemaViolations(vec![SchemaViolation {
                    path: "/id".to_owned(),
                    expected: "integer".to_owned(),
                    actual: "string".to_owned(),
                }]),
                internal_skip: None,
            },
        ];

        let failure = collect_structured_output_failure(&response, &choices);
        assert_eq!(failure.violations.len(), 2);
        assert_eq!(failure.violations[0].path, "/choices/0");
        assert_eq!(failure.violations[1].path, "/id");
        assert!(failure.previous_output.contains("not-json"));
        assert!(failure.previous_output.contains("wrong"));
    }

    #[test]
    fn smart_routing_headers_use_contract_names_and_two_decimal_score() {
        let mut response = base_response(Message {
            role: "assistant".to_owned(),
            content: serde_json::json!("ok"),
            extra: Default::default(),
        });
        response.extra.insert(
            "gateway_smart_routing".to_owned(),
            serde_json::to_value(routing_decision(0.476)).unwrap(),
        );

        let headers = smart_routing_headers(&response);
        assert_eq!(headers.get("x-smart-route-tier").unwrap(), "balanced");
        assert_eq!(headers.get("x-smart-route-score").unwrap(), "0.48");
        assert_eq!(
            headers.get("x-smart-route-classifier").unwrap(),
            "composite"
        );
        assert_eq!(
            headers.get("x-smart-route-task-type").unwrap(),
            "code_generation"
        );
        assert_eq!(headers.get("x-smart-route-escalated").unwrap(), "true");
        assert_eq!(headers.get("x-smart-route-cache-hit").unwrap(), "true");
        assert_eq!(headers.len(), 6);

        let http_response = openai_json_response(&response);
        assert_eq!(
            http_response.headers().get("x-smart-route-score").unwrap(),
            "0.48"
        );
    }

    // Feature: smart-routing, Property 21: Exact Score Header Format
    proptest! {
    #![proptest_config(ProptestConfig::with_cases(64))]

    #[test]
    fn prop_smart_routing_score_header_has_exact_format_and_normalized_value(score in any::<f64>()) {
    let normalized_score = ComplexityScore::new(score).value();
    let mut response = base_response(Message {
    role: "assistant".to_owned(),
    content: serde_json::json!("ok"),
    extra: Default::default(),
    });
    response.extra.insert(
    "gateway_smart_routing".to_owned(),
    serde_json::to_value(routing_decision(score)).unwrap(),
    );

    let headers = smart_routing_headers(&response);
    let score_header = headers
    .get("x-smart-route-score")
    .expect("routing metadata must emit a score header")
    .to_str()
    .expect("score header must be ASCII");
    let (integer_digits, fractional_digits) = score_header
    .split_once('.')
    .expect("score header must contain a decimal point");

    prop_assert!(!integer_digits.is_empty());
    prop_assert!(integer_digits.bytes().all(|byte| byte.is_ascii_digit()));
    prop_assert_eq!(fractional_digits.len(), 2);
    prop_assert!(fractional_digits.bytes().all(|byte| byte.is_ascii_digit()));
    prop_assert!(!fractional_digits.contains('.'));
    let expected_score = if normalized_score == 0.0 {
    "0.00".to_owned()
    } else {
    format!("{normalized_score:.2}")
    };
    prop_assert_eq!(score_header, expected_score);
    let parsed_score = score_header.parse::<f64>().expect("score header must parse");
    prop_assert!((0.0..=1.0).contains(&parsed_score));
    }
    }

    #[derive(Clone, Copy, Debug)]
    enum SmartRoutingHeaderAbsenceScenario {
        NoInternalMetadata,
        Bypassed,
    }

    fn smart_routing_header_absence_scenario_strategy(
    ) -> impl Strategy<Value = SmartRoutingHeaderAbsenceScenario> {
        prop_oneof![
            Just(SmartRoutingHeaderAbsenceScenario::NoInternalMetadata),
            Just(SmartRoutingHeaderAbsenceScenario::Bypassed),
        ]
    }

    // Feature: smart-routing, Property 22: Header Absence Without Routing Metadata
    proptest! {
    #![proptest_config(ProptestConfig::with_cases(64))]

    #[test]
    fn prop_smart_routing_headers_are_absent_without_metadata_or_when_bypassed(
    scenario in smart_routing_header_absence_scenario_strategy(),
    unrelated_value in json_value_strategy(),
    ) {
    let mut response = base_response(Message {
    role: "assistant".to_owned(),
    content: serde_json::json!("ok"),
    extra: Default::default(),
    });
    response
    .extra
    .insert("unrelated_internal_metadata".to_owned(), unrelated_value);

    let http_response = match scenario {
    SmartRoutingHeaderAbsenceScenario::NoInternalMetadata => openai_json_response(&response),
    SmartRoutingHeaderAbsenceScenario::Bypassed => {
    response.extra.insert(
    "gateway_smart_routing".to_owned(),
    serde_json::to_value(routing_decision(0.73)).unwrap(),
    );
    ().into_response()
    }
    };

    prop_assert!(http_response
    .headers()
    .keys()
    .all(|name| !name.as_str().starts_with("x-smart-route-")));
    }
    }

    #[test]
    fn smart_routing_headers_are_absent_without_internal_metadata() {
        let response = base_response(Message {
            role: "assistant".to_owned(),
            content: serde_json::json!("ok"),
            extra: Default::default(),
        });
        let mut http_response = ().into_response();
        attach_smart_routing_headers(&mut http_response, smart_routing_headers(&response));

        assert!(http_response
            .headers()
            .keys()
            .all(|name| !name.as_str().starts_with("x-smart-route-")));
    }

    #[test]
    fn smart_routing_optional_true_headers_are_omitted_when_false() {
        let mut decision = routing_decision(0.5);
        decision.escalated = false;
        decision.cache_hit = false;
        let mut response = base_response(Message {
            role: "assistant".to_owned(),
            content: serde_json::json!("ok"),
            extra: Default::default(),
        });
        response.extra.insert(
            "gateway_smart_routing".to_owned(),
            serde_json::to_value(decision).unwrap(),
        );

        let headers = smart_routing_headers(&response);
        assert!(headers.get("x-smart-route-escalated").is_none());
        assert!(headers.get("x-smart-route-cache-hit").is_none());
    }

    #[test]
    fn smart_routing_budget_error_sets_exceeded_period_header() {
        let response = GatewayError::SmartRoutingBudgetExceeded {
            period: "monthly".to_owned(),
        }
        .into_response();

        assert_eq!(response.status(), axum::http::StatusCode::TOO_MANY_REQUESTS);
        assert_eq!(
            response.headers().get("x-budget-exceeded").unwrap(),
            "monthly"
        );
    }

    #[test]
    fn response_metadata_is_logged_but_removed_from_client_payload() {
        let request = OpenAIRequest {
            model: "requested-model".to_owned(),
            messages: Vec::new(),
            stream: false,
            temperature: None,
            max_tokens: None,
            extra: Default::default(),
        };
        let mut response = base_response(Message {
            role: "assistant".to_owned(),
            content: serde_json::json!("ok"),
            extra: Default::default(),
        });
        response
            .extra
            .insert("gateway_provider".to_owned(), serde_json::json!("provider"));
        response.extra.insert(
            "gateway_compression".to_owned(),
            serde_json::to_value(test_compression_stats()).unwrap(),
        );
        response.extra.insert(
            "gateway_smart_routing".to_owned(),
            serde_json::to_value(routing_decision(0.42)).unwrap(),
        );

        let context = RequestLogContext::from_response(&request, "trace".to_owned(), 10, &response);
        assert_eq!(
            context
                .compression
                .as_ref()
                .map(|metadata| metadata.compression_level.as_str()),
            Some("standard")
        );
        let client_response = prepare_response_for_client(&response);
        assert!(!client_response.extra.contains_key("gateway_compression"));
        assert!(!client_response.extra.contains_key("gateway_provider"));
        assert!(!client_response.extra.contains_key("gateway_smart_routing"));
        assert!(!serde_json::to_string(&client_response)
            .unwrap()
            .contains("gateway_compression"));
    }

    #[test]
    fn streaming_log_context_uses_explicit_compression_metadata() {
        let request = OpenAIRequest {
            model: "requested-model".to_owned(),
            messages: Vec::new(),
            stream: true,
            temperature: None,
            max_tokens: None,
            extra: Default::default(),
        };
        let context = RequestLogContext::from_streaming_success(
            &request,
            "trace".to_owned(),
            10,
            "provider".to_owned(),
            "model".to_owned(),
            test_compression_stats(),
        );
        assert_eq!(context.provider, "provider");
        assert_eq!(context.responded_model.as_deref(), Some("model"));
        assert_eq!(
            context
                .compression
                .as_ref()
                .map(|metadata| metadata.original_tokens),
            Some(100)
        );
    }

    #[test]
    fn memory_log_context_records_request_counts_and_project_hash() {
        let request = OpenAIRequest {
            model: "requested-model".to_owned(),
            messages: Vec::new(),
            stream: false,
            temperature: None,
            max_tokens: None,
            extra: Default::default(),
        };
        let response = base_response(Message {
            role: "assistant".to_owned(),
            content: serde_json::json!("ok"),
            extra: Default::default(),
        });
        let context = RequestLogContext::from_response(&request, "trace".to_owned(), 10, &response)
            .with_memory(
                Some((
                    &ContextType::Project("0123456789abcdef".to_owned()),
                    &InjectionResult {
                        memories_injected: 3,
                        injection_tokens: 240,
                        ..InjectionResult::default()
                    },
                )),
                Some(ExtractionCounts {
                    stored: 2,
                    ..ExtractionCounts::default()
                }),
            );
        assert_eq!(context.memories_injected, 3);
        assert_eq!(context.memories_stored, 2);
        assert_eq!(context.injection_tokens, 240);
        assert_eq!(
            context.detected_project.as_deref(),
            Some("0123456789abcdef")
        );
    }

    #[test]
    fn memory_feedback_helpers_suppress_structured_and_require_string_content() {
        let mut request =
            request_with_response_format(false, serde_json::json!({"type": "json_object"}));
        assert!(requests_structured_output(&request));
        request.extra.remove("response_format");
        assert!(!requests_structured_output(&request));

        let mut response = base_response(Message {
            role: "assistant".to_owned(),
            content: serde_json::json!("answer"),
            extra: Default::default(),
        });
        assert!(append_feedback_to_response(&mut response, " suffix"));
        assert_eq!(
            response.choices[0].message.content,
            serde_json::json!("answer suffix")
        );
        response.choices[0].message.content = serde_json::json!({"value": 1});
        assert!(!append_feedback_to_response(&mut response, " suffix"));
    }

    #[test]
    fn memory_feedback_chunk_is_content_delta_for_before_done_insertion() {
        let request = request_with_response_format(false, serde_json::json!({"type": "text"}));
        let chunk = memory_feedback_chunk(&request, "\n\n---\nfeedback");
        assert_eq!(chunk["choices"][0]["delta"]["content"], "\n\n---\nfeedback");
        assert!(chunk["choices"][0]["finish_reason"].is_null());
    }

    #[test]
    fn streaming_chunks_include_reasoning_before_content() {
        let mut extra = serde_json::Map::new();
        extra.insert("reasoning".to_string(), serde_json::json!("thinking step"));

        let response = base_response(Message {
            role: "assistant".to_string(),
            content: serde_json::json!("final answer"),
            extra,
        });

        let chunks = streaming_chunks_from_response(&response);

        assert_eq!(chunks[0]["choices"][0]["delta"]["role"], "assistant");
        assert_eq!(
            chunks[1]["choices"][0]["delta"]["reasoning"],
            "thinking step"
        );
        assert_eq!(chunks[2]["choices"][0]["delta"]["content"], "final answer");
    }

    #[test]
    fn streaming_chunks_preserve_reasoning_content_field_name() {
        let mut extra = serde_json::Map::new();
        extra.insert(
            "reasoning_content".to_string(),
            serde_json::json!("hidden chain"),
        );

        let response = base_response(Message {
            role: "assistant".to_string(),
            content: serde_json::json!("visible answer"),
            extra,
        });

        let chunks = streaming_chunks_from_response(&response);

        assert_eq!(
            chunks[1]["choices"][0]["delta"]["reasoning_content"],
            "hidden chain"
        );
        assert_eq!(
            chunks[2]["choices"][0]["delta"]["content"],
            "visible answer"
        );
    }

    // -- Early synthetic SSE event (task 2.3) --------------------------------

    /// Req 1.1, 1.3: the early event carries a `role: assistant` delta and a
    /// well-formed chunk envelope (id/object/created/model) with a null
    /// finish_reason, and does NOT prematurely emit content.
    #[test]
    fn early_event_chunk_has_role_delta_and_chunk_envelope() {
        let chunk = early_event_chunk("chatcmpl-early-id", 1700, "gpt-4o");

        assert_eq!(chunk["id"], "chatcmpl-early-id");
        assert_eq!(chunk["object"], "chat.completion.chunk");
        assert_eq!(chunk["created"], 1700);
        assert_eq!(chunk["model"], "gpt-4o");
        assert_eq!(chunk["choices"][0]["index"], 0);
        assert_eq!(chunk["choices"][0]["delta"]["role"], "assistant");
        assert!(chunk["choices"][0]["finish_reason"].is_null());
        // The early event only signals the role; it must not carry content.
        assert!(chunk["choices"][0]["delta"].get("content").is_none());
    }

    /// Req 1.5 + same-id: chunks emitted after the early event reuse the
    /// pre-generated `id`/`created` (not the provider's own envelope values)
    /// and suppress the duplicate leading `role: assistant` delta.
    #[test]
    fn chunks_after_early_event_share_id_and_suppress_role() {
        // Provider response carries its own id/created which must be overridden.
        let mut response = base_response(Message {
            role: "assistant".to_string(),
            content: serde_json::json!("hello world"),
            extra: Default::default(),
        });
        response.id = "chatcmpl-provider-original".to_string();
        response.created = 123;

        let chunks = streaming_chunks_after_early_event(&response, "chatcmpl-early-id", 1700);

        // Every chunk must line up with the early event's id/created.
        for chunk in &chunks {
            assert_eq!(chunk["id"], "chatcmpl-early-id");
            assert_eq!(chunk["created"], 1700);
        }

        // The leading role-only delta is suppressed: the first chunk after the
        // early event goes straight to content (no `role` field).
        assert!(chunks[0]["choices"][0]["delta"].get("role").is_none());
        assert_eq!(chunks[0]["choices"][0]["delta"]["content"], "hello world");
    }

    /// Req 1.4 / 1.6: the cache-replay and `emit_early_event: false` paths use
    /// `streaming_chunks_from_response`, which is self-contained — it emits its
    /// own `role: assistant` chunk and keeps the provider's `id`. This proves
    /// no separate synthetic early event is prepended on those paths.
    #[test]
    fn chunks_without_early_event_keep_role_and_provider_id() {
        let mut response = base_response(Message {
            role: "assistant".to_string(),
            content: serde_json::json!("cached answer"),
            extra: Default::default(),
        });
        response.id = "chatcmpl-provider-original".to_string();

        let chunks = streaming_chunks_from_response(&response);

        // First chunk is the role delta (not suppressed) ...
        assert_eq!(chunks[0]["choices"][0]["delta"]["role"], "assistant");
        // ... and the provider's own id is preserved across all chunks.
        for chunk in &chunks {
            assert_eq!(chunk["id"], "chatcmpl-provider-original");
        }
    }

    // -- Graceful SSE error events (task 4.1) --------------------------------

    /// Req 5.1-5.5: the error helper returns an error data event followed by a
    /// `[DONE]` event. axum's `Event` does not expose its data for assertion, so
    /// this sanity test guards the event count; the payload shape and trace_id
    /// correlation are exercised by the task 4.3 integration tests.
    #[test]
    fn emit_sse_error_event_returns_error_then_done() {
        let events = emit_sse_error_event("chunk_timeout_error", "no data for 60s", "tr-abc123");
        assert_eq!(events.len(), 2);
    }

    /// Req 5.1 / 5.5: the TTFB-timeout SSE error frame carries the exact
    /// `ttfb_timeout_error` type, the wired message, and the trace_id for
    /// correlation. Replicates the strings emitted by the timeout path in
    /// `chat_completions_stream` (task 4.2).
    #[test]
    fn sse_error_payload_ttfb_timeout_shape() {
        let payload = sse_error_payload(
            "ttfb_timeout_error",
            &format!("Provider did not respond within {}s", 30),
            "tr-ttfb-1",
        );
        assert_eq!(payload["error"]["type"], "ttfb_timeout_error");
        assert_eq!(
            payload["error"]["message"],
            "Provider did not respond within 30s"
        );
        assert_eq!(payload["error"]["trace_id"], "tr-ttfb-1");
    }

    /// Req 5.2 / 5.5: the total-timeout SSE error frame carries the exact
    /// `total_timeout_error` type, the wired message, and the trace_id.
    #[test]
    fn sse_error_payload_total_timeout_shape() {
        let payload = sse_error_payload(
            "total_timeout_error",
            &format!("Response exceeded {}s total timeout", 120),
            "tr-total-1",
        );
        assert_eq!(payload["error"]["type"], "total_timeout_error");
        assert_eq!(
            payload["error"]["message"],
            "Response exceeded 120s total timeout"
        );
        assert_eq!(payload["error"]["trace_id"], "tr-total-1");
    }

    /// Req 5.5: every error frame includes a non-empty `trace_id` field, and the
    /// `emit_sse_error_event` data event is built from this exact payload (its
    /// first event's data string equals the serialized payload).
    #[test]
    fn sse_error_payload_includes_trace_id() {
        let payload = sse_error_payload("chunk_timeout_error", "stalled", "tr-corr-99");
        assert_eq!(payload["error"]["trace_id"], "tr-corr-99");
        assert!(payload["error"]["trace_id"]
            .as_str()
            .is_some_and(|s: &str| !s.is_empty()));
    }

    // -- Stream error classification (task 4.2) ------------------------------

    fn attempt_with_error(error: String) -> ProviderAttempt {
        ProviderAttempt::new("openai".to_string(), "gpt-4".to_string(), error, Some(504))
    }

    /// Req 5.1: `router.route_request()` wraps a single-provider TTFB timeout in
    /// `AllProvidersFailed`, so classification must recover the `ttfb_timeout_error`
    /// type from the aggregated attempt's recorded Display string rather than
    /// falling through to the generic `stream_error`.
    #[test]
    fn classify_stream_error_recovers_ttfb_from_aggregated() {
        let inner = GatewayError::TtfbTimeout(30).to_string();
        let agg = AggregatedError::new(vec![attempt_with_error(inner)]);
        let (error_type, message) = classify_stream_error(&GatewayError::AllProvidersFailed(agg));
        assert_eq!(error_type, "ttfb_timeout_error");
        assert!(!message.is_empty());
    }

    /// Req 5.2: a total-timeout wrapped in `AllProvidersFailed` is classified as
    /// `total_timeout_error`.
    #[test]
    fn classify_stream_error_recovers_total_from_aggregated() {
        let inner = GatewayError::TotalTimeout(120).to_string();
        let agg = AggregatedError::new(vec![attempt_with_error(inner)]);
        let (error_type, message) = classify_stream_error(&GatewayError::AllProvidersFailed(agg));
        assert_eq!(error_type, "total_timeout_error");
        assert!(!message.is_empty());
    }

    /// A non-timeout aggregated failure keeps the generic `stream_error` type.
    #[test]
    fn classify_stream_error_aggregated_non_timeout_is_generic() {
        let agg = AggregatedError::new(vec![attempt_with_error(
            "Provider error: openai - 500 internal".to_string(),
        )]);
        let (error_type, _message) = classify_stream_error(&GatewayError::AllProvidersFailed(agg));
        assert_eq!(error_type, "stream_error");
    }

    /// Direct (unwrapped) timeout variants keep their precise `{secs}` messages.
    #[test]
    fn classify_stream_error_direct_variants_keep_precise_message() {
        let (ttfb_type, ttfb_msg) = classify_stream_error(&GatewayError::TtfbTimeout(30));
        assert_eq!(ttfb_type, "ttfb_timeout_error");
        assert_eq!(ttfb_msg, "Provider did not respond within 30s");

        let (total_type, total_msg) = classify_stream_error(&GatewayError::TotalTimeout(120));
        assert_eq!(total_type, "total_timeout_error");
        assert_eq!(total_msg, "Response exceeded 120s total timeout");
    }

    // -- Configurable keep-alive (task 3.2) ----------------------------------
    /// Req 2.4 / 2.5: `build_keepalive` constructs a value for both the custom
    /// interval path (interval > 0) and the disabled/default path (interval == 0)
    /// without panicking. axum's `KeepAlive` does not expose its interval/text
    /// for assertion, so the observable behavioral guarantees (a working SSE
    /// stream under each setting) are covered by the integration tests; this
    /// unit test only guards the two construction branches.
    #[test]
    fn build_keepalive_handles_custom_and_disabled_intervals() {
        // Custom interval (Req 2.4: within the 1–60 range).
        let custom = StreamingConfig {
            keepalive_interval_seconds: 5,
            ..StreamingConfig::default()
        };
        let _ = build_keepalive(&custom);

        // Disabled → falls back to axum's default keep-alive (Req 2.5).
        let disabled = StreamingConfig {
            keepalive_interval_seconds: 0,
            ..StreamingConfig::default()
        };
        let _ = build_keepalive(&disabled);
    }

    // -- True streaming pass-through relay (task 5.3) ------------------------

    /// Req 3.2: a well-formed chunk (carries a `choices` array) is forwarded.
    #[test]
    fn classify_relay_line_forwards_well_formed_chunk() {
        let payload = r#"{"choices":[{"index":0,"delta":{"content":"hi"},"finish_reason":null}]}"#;
        assert_eq!(classify_relay_line(payload), RelayLineAction::Forward);
    }

    /// Req 3.3: malformed (non-JSON) chunks are skipped.
    #[test]
    fn classify_relay_line_skips_malformed_json() {
        assert_eq!(
            classify_relay_line("{not json"),
            RelayLineAction::SkipMalformed
        );
    }

    /// Valid JSON without `choices`/`error` is skipped quietly (accumulation is task 5.4).
    #[test]
    fn classify_relay_line_skips_non_chunk_json() {
        let payload = r#"{"usage":{"prompt_tokens":1,"completion_tokens":2}}"#;
        assert_eq!(classify_relay_line(payload), RelayLineAction::SkipNonChunk);
    }

    /// The upstream `[DONE]` sentinel maps to `Done` (we emit our own).
    #[test]
    fn classify_relay_line_detects_done_sentinel() {
        assert_eq!(classify_relay_line("[DONE]"), RelayLineAction::Done);
    }

    /// Req 3.6: a top-level `error` object is a mid-stream failure carrying the message.
    #[test]
    fn classify_relay_line_detects_top_level_error_frame() {
        let payload = r#"{"error":{"message":"upstream exploded","type":"server_error"}}"#;
        assert_eq!(
            classify_relay_line(payload),
            RelayLineAction::Error("upstream exploded".to_string())
        );
    }

    /// Req 3.6: `finish_reason: "error"` on the first choice is a mid-stream failure.
    #[test]
    fn classify_relay_line_detects_finish_reason_error() {
        let payload = r#"{"choices":[{"index":0,"delta":{},"finish_reason":"error"}]}"#;
        assert!(matches!(
            classify_relay_line(payload),
            RelayLineAction::Error(_)
        ));
    }

    /// Build a synthetic streaming `reqwest::Response` from raw SSE bytes so the
    /// relay loop can be driven without a live server.
    fn fake_streaming_response(body: &'static str) -> reqwest::Response {
        let stream = futures::stream::once(async move { Ok::<_, std::io::Error>(body.as_bytes()) });
        let http_response = axum::http::Response::new(reqwest::Body::wrap_stream(stream));
        reqwest::Response::from(http_response)
    }

    /// Build the caching dependencies the relay needs (task 5.4). Caching is a
    /// no-op for these tests because the requests are streaming/ineligible, but
    /// the arguments must be supplied to drive the relay.
    fn relay_cache_deps() -> (
        std::sync::Arc<crate::cache::ExactCache>,
        std::sync::Arc<crate::metrics::Metrics>,
        OpenAIRequest,
    ) {
        let exact_cache = std::sync::Arc::new(crate::cache::ExactCache::new(
            &crate::config::ExactCacheConfig::default(),
        ));
        let metrics = std::sync::Arc::new(crate::metrics::Metrics::new());
        let request = OpenAIRequest {
            model: "test-model".to_string(),
            messages: vec![],
            stream: true,
            temperature: None,
            max_tokens: None,
            extra: Default::default(),
        };
        (exact_cache, metrics, request)
    }

    /// Throwaway outcome handle for relay tests that don't assert the signal.
    fn mk_outcome() -> std::sync::Arc<tokio::sync::Mutex<RelayOutcome>> {
        std::sync::Arc::new(tokio::sync::Mutex::new(RelayOutcome::Completed {
            usage: Usage::default(),
        }))
    }

    /// Req 3.2 / 3.6: forwarded chunks reach the client and the relay always
    /// terminates with exactly one `[DONE]` on a clean finish. Malformed and
    /// non-chunk lines are dropped (Req 3.3); the upstream `[DONE]` is swallowed.
    #[tokio::test]
    async fn relay_forwards_chunks_and_emits_single_done() {
        let body = "data: {\"choices\":[{\"index\":0,\"delta\":{\"content\":\"a\"}}]}\n\n\
                    data: not-json\n\n\
                    data: {\"choices\":[{\"index\":0,\"delta\":{\"content\":\"b\"}}]}\n\n\
                    data: [DONE]\n\n";
        let response = fake_streaming_response(body);
        let (exact_cache, metrics, request) = relay_cache_deps();
        let stream = relay_passthrough_stream(
            response,
            StreamingConfig::default(),
            "tr-relay".to_string(),
            Duration::from_secs(30),
            exact_cache,
            metrics,
            request,
            mk_outcome(),
            None,
            None,
        );
        let events: Vec<_> = stream.collect().await;
        // 2 forwarded chunks + 1 terminal [DONE]. Malformed + upstream [DONE] dropped.
        assert_eq!(events.len(), 3);
        assert!(events.iter().all(|e| e.is_ok()));
    }

    /// Req 3.6 / Req 5: a mid-stream error frame produces a graceful error event
    /// followed by `[DONE]` (via `emit_sse_error_event`), and no extra `[DONE]`.
    #[tokio::test]
    async fn relay_emits_error_then_done_on_error_frame() {
        let body = "data: {\"choices\":[{\"index\":0,\"delta\":{\"content\":\"a\"}}]}\n\n\
                    data: {\"error\":{\"message\":\"boom\",\"type\":\"server_error\"}}\n\n";
        let response = fake_streaming_response(body);
        let (exact_cache, metrics, request) = relay_cache_deps();
        let stream = relay_passthrough_stream(
            response,
            StreamingConfig::default(),
            "tr-relay-err".to_string(),
            Duration::from_secs(30),
            exact_cache,
            metrics,
            request,
            mk_outcome(),
            None,
            None,
        );
        let events: Vec<_> = stream.collect().await;
        // 1 forwarded chunk + error event + [DONE] (the error path's own DONE).
        assert_eq!(events.len(), 3);
    }

    /// Req 3.10: a clean completion of an eligible request reassembles the
    /// forwarded chunks and writes them to the exact cache.
    #[tokio::test]
    async fn relay_caches_assembled_response_on_clean_completion() {
        let body = "data: {\"id\":\"chatcmpl-x\",\"object\":\"chat.completion.chunk\",\"created\":1,\"model\":\"test-model\",\"choices\":[{\"index\":0,\"delta\":{\"content\":\"Hello\"},\"finish_reason\":null}]}\n\n\
                    data: {\"id\":\"chatcmpl-x\",\"object\":\"chat.completion.chunk\",\"created\":1,\"model\":\"test-model\",\"choices\":[{\"index\":0,\"delta\":{\"content\":\" world\"},\"finish_reason\":\"stop\"}]}\n\n\
                    data: [DONE]\n\n";
        let response = fake_streaming_response(body);
        let (exact_cache, metrics, request) = relay_cache_deps();
        assert!(exact_cache.get(&request).is_none());
        let stream = relay_passthrough_stream(
            response,
            StreamingConfig::default(),
            "tr-relay-cache".to_string(),
            Duration::from_secs(30),
            exact_cache.clone(),
            metrics,
            request.clone(),
            mk_outcome(),
            None,
            None,
        );
        let _events: Vec<_> = stream.collect().await;

        let cached = exact_cache
            .get(&request)
            .expect("response should be cached");
        let resp: OpenAIResponse =
            serde_json::from_str(&cached).expect("cached payload is valid JSON");
        assert_eq!(
            resp.choices[0].message.content,
            serde_json::json!("Hello world")
        );
        assert_eq!(resp.choices[0].finish_reason.as_deref(), Some("stop"));
    }

    /// Req 3.10: an errored stream must NOT be cached (partial/failed result).
    #[tokio::test]
    async fn relay_does_not_cache_on_error_frame() {
        let body = "data: {\"id\":\"chatcmpl-y\",\"object\":\"chat.completion.chunk\",\"created\":1,\"model\":\"test-model\",\"choices\":[{\"index\":0,\"delta\":{\"content\":\"partial\"},\"finish_reason\":null}]}\n\n\
                    data: {\"error\":{\"message\":\"boom\",\"type\":\"server_error\"}}\n\n";
        let response = fake_streaming_response(body);
        let (exact_cache, metrics, request) = relay_cache_deps();
        let stream = relay_passthrough_stream(
            response,
            StreamingConfig::default(),
            "tr-relay-noerr-cache".to_string(),
            Duration::from_secs(30),
            exact_cache.clone(),
            metrics,
            request.clone(),
            mk_outcome(),
            None,
            None,
        );
        let _events: Vec<_> = stream.collect().await;
        assert!(
            exact_cache.get(&request).is_none(),
            "errored stream must not populate the cache"
        );
    }

    /// Build a synthetic streaming `reqwest::Response` that yields `first_chunk`
    /// immediately, then stalls for `stall` before any further data — used to
    /// drive the relay's inter-chunk / total timeout paths deterministically.
    fn stalling_streaming_response(
        first_chunk: &'static str,
        stall: Duration,
    ) -> reqwest::Response {
        let stream = async_stream::stream! {
            yield Ok::<_, std::io::Error>(first_chunk.as_bytes());
            tokio::time::sleep(stall).await;
            // Trailing data that the timeout should pre-empt before it is read.
            yield Ok::<_, std::io::Error>("data: [DONE]\n\n".as_bytes());
        };
        let http_response = axum::http::Response::new(reqwest::Body::wrap_stream(stream));
        reqwest::Response::from(http_response)
    }

    /// Render a relay stream to its SSE wire text so timeout error frames can be
    /// inspected (axum `Event` has no public data accessor).
    async fn relay_to_sse_text(
        stream: impl futures::Stream<Item = Result<super::Event, std::convert::Infallible>>
            + Send
            + 'static,
    ) -> String {
        use axum::response::{IntoResponse, Sse};
        let resp = Sse::new(stream).into_response();
        let body = axum::body::to_bytes(resp.into_body(), 1024 * 1024)
            .await
            .unwrap();
        String::from_utf8(body.to_vec()).unwrap()
    }

    /// Req 3.12: when the provider sends an initial chunk then goes silent past
    /// the inter-chunk window, the relay emits a `chunk_timeout_error` SSE event
    /// followed by `[DONE]`. A 1s inter-chunk timeout keeps the test fast; the
    /// long stall future is dropped the moment the relay terminates, so the
    /// test never actually waits 30s.
    #[tokio::test]
    async fn relay_emits_chunk_timeout_error_when_provider_stalls() {
        let first = "data: {\"choices\":[{\"index\":0,\"delta\":{\"content\":\"a\"}}]}\n\n";
        // Provider stalls far longer than the 1s inter-chunk timeout.
        let response = stalling_streaming_response(first, Duration::from_secs(30));
        let cfg = StreamingConfig {
            chunk_timeout_seconds: 1,
            ..StreamingConfig::default()
        };
        let (exact_cache, metrics, request) = relay_cache_deps();
        let stream = relay_passthrough_stream(
            response,
            cfg,
            "tr-chunk-timeout".to_string(),
            // Large total budget so the inter-chunk timeout fires first (Req 3.12).
            Duration::from_secs(3600),
            exact_cache,
            metrics,
            request,
            mk_outcome(),
            None,
            None,
        );

        let text = relay_to_sse_text(stream).await;
        // The first chunk was forwarded before the stall.
        assert!(
            text.contains("\"content\":\"a\""),
            "first chunk forwarded before timeout"
        );
        // The inter-chunk timeout surfaces as the precise type (Req 3.12, 5.3).
        assert!(
            text.contains("\"type\":\"chunk_timeout_error\""),
            "stall must produce a chunk_timeout_error frame, got: {text}"
        );
        assert!(
            text.trim_end().ends_with("data: [DONE]"),
            "error frame followed by [DONE]"
        );
    }

    /// Req 3.11: the total streaming budget caps the whole duration. With a 1s
    /// total timeout and a provider that stalls after the first chunk, the relay
    /// emits a `total_timeout_error` SSE event followed by `[DONE]`. The 60s
    /// inter-chunk window never fires because the total budget elapses first.
    #[tokio::test]
    async fn relay_emits_total_timeout_error_when_stream_exceeds_budget() {
        let first = "data: {\"choices\":[{\"index\":0,\"delta\":{\"content\":\"a\"}}]}\n\n";
        let response = stalling_streaming_response(first, Duration::from_secs(30));
        let cfg = StreamingConfig {
            // Inter-chunk window is large; the total budget should bound first.
            chunk_timeout_seconds: 60,
            ..StreamingConfig::default()
        };
        let (exact_cache, metrics, request) = relay_cache_deps();
        let stream = relay_passthrough_stream(
            response,
            cfg,
            "tr-total-timeout".to_string(),
            Duration::from_secs(1),
            exact_cache,
            metrics,
            request,
            mk_outcome(),
            None,
            None,
        );

        let text = relay_to_sse_text(stream).await;
        assert!(
            text.contains("\"content\":\"a\""),
            "first chunk forwarded before timeout"
        );
        assert!(
            text.contains("\"type\":\"total_timeout_error\""),
            "exceeding the total budget must produce a total_timeout_error frame, got: {text}"
        );
        assert!(
            text.trim_end().ends_with("data: [DONE]"),
            "error frame followed by [DONE]"
        );
    }

    /// Task 6.1 / Req 4.1: a `data:` payload carrying real content/tool_call/
    /// reasoning deltas is detected, while a role-only delta is not (so it does
    /// not block pre-content failover).
    #[test]
    fn chunk_carries_content_distinguishes_role_only_from_content() {
        let role_only = r#"{"choices":[{"index":0,"delta":{"role":"assistant"}}]}"#;
        assert!(
            !chunk_carries_content(role_only),
            "role-only delta is not content"
        );

        let content = r#"{"choices":[{"index":0,"delta":{"content":"hi"}}]}"#;
        assert!(chunk_carries_content(content), "content delta counts");

        let empty_content = r#"{"choices":[{"index":0,"delta":{"content":""}}]}"#;
        assert!(
            !chunk_carries_content(empty_content),
            "empty content does not count"
        );

        let tool = r#"{"choices":[{"index":0,"delta":{"tool_calls":[{"index":0,"function":{"name":"f"}}]}}]}"#;
        assert!(chunk_carries_content(tool), "tool_calls delta counts");

        let reasoning = r#"{"choices":[{"index":0,"delta":{"reasoning_content":"think"}}]}"#;
        assert!(chunk_carries_content(reasoning), "reasoning_content counts");

        assert!(
            !chunk_carries_content("not json"),
            "malformed is not content"
        );
    }

    /// Build a streaming `reqwest::Response` whose body errors immediately,
    /// before any byte is delivered — drives the relay's pre-content failure
    /// path.
    fn erroring_streaming_response() -> reqwest::Response {
        let stream = futures::stream::once(async move {
            Err::<&[u8], _>(std::io::Error::new(
                std::io::ErrorKind::ConnectionReset,
                "reset",
            ))
        });
        let http_response = axum::http::Response::new(reqwest::Body::wrap_stream(stream));
        reqwest::Response::from(http_response)
    }

    #[test]
    fn reqwest_error_chain_includes_transport_cause() {
        let response = erroring_streaming_response();
        let (version, content_encoding, transfer_encoding, content_length) =
            upstream_stream_metadata(&response);
        assert_eq!(version, reqwest::Version::HTTP_11);
        assert_eq!(content_encoding, "identity");
        assert_eq!(transfer_encoding, "none");
        assert_eq!(content_length, None);
    }

    /// Task 6.1 / Req 4.1, 4.4: when the upstream errors BEFORE any content is

    /// forwarded, the relay stays silent (no error event, no `[DONE]`) and
    /// records `FailedBeforeContent` so the handler can fail over without
    /// emitting a duplicate role event.
    #[tokio::test]
    async fn relay_signals_failed_before_content_and_stays_silent() {
        let response = erroring_streaming_response();
        let (exact_cache, metrics, request) = relay_cache_deps();
        let outcome = mk_outcome();
        let stream = relay_passthrough_stream(
            response,
            StreamingConfig::default(),
            "tr-pre-content-fail".to_string(),
            Duration::from_secs(30),
            exact_cache,
            metrics,
            request,
            outcome.clone(),
            None,
            None,
        );
        let events: Vec<_> = stream.collect().await;
        assert!(
            events.is_empty(),
            "pre-content failure must emit no SSE events"
        );
        let guard = outcome.lock().await;
        match &*guard {
            RelayOutcome::FailedBeforeContent(reason) => assert!(reason.contains("reset")),
            other => panic!("expected FailedBeforeContent, got {other:?}"),
        }
    }

    /// Task 6.1: a clean completion records `Completed` so the handler's
    /// failover loop terminates without retrying.
    #[tokio::test]
    async fn relay_records_completed_on_clean_finish() {
        let body = "data: {\"choices\":[{\"index\":0,\"delta\":{\"content\":\"a\"}}]}\n\n\
                    data: [DONE]\n\n";
        let response = fake_streaming_response(body);
        let (exact_cache, metrics, request) = relay_cache_deps();
        let outcome = mk_outcome();
        let stream = relay_passthrough_stream(
            response,
            StreamingConfig::default(),
            "tr-clean".to_string(),
            Duration::from_secs(30),
            exact_cache,
            metrics,
            request,
            outcome.clone(),
            None,
            None,
        );
        let _events: Vec<_> = stream.collect().await;
        assert!(matches!(
            &*outcome.lock().await,
            RelayOutcome::Completed { .. }
        ));
    }
}

// ---------------------------------------------------------------------------
// POST /v1/completions  (Req 2.2)
// ---------------------------------------------------------------------------

/// Legacy completions endpoint — pass-through proxy.
pub async fn completions(
    State(state): State<AppState>,
    headers: HeaderMap,
    authenticated_key: Option<Extension<AuthenticatedKey>>,
    Json(request): Json<OpenAIRequest>,
) -> Response {
    // Reuse chat completions routing; the OpenAI completions format is close enough
    // for provider pass-through. Full translation can be refined later.
    let trace_id = trace_id_from_headers(&headers);
    let virtual_key_id = authenticated_key.map(|Extension(key)| key.id);
    chat_completions_non_stream(state, request, trace_id, virtual_key_id).await
}
// ---------------------------------------------------------------------------
// Provider pass-through endpoints (Req 2.3-2.5)
// ---------------------------------------------------------------------------

fn json_model(body: &[u8]) -> Result<String, GatewayError> {
    let value: serde_json::Value = serde_json::from_slice(body)
        .map_err(|error| GatewayError::InvalidRequest(format!("Invalid JSON body: {}", error)))?;
    value
        .get("model")
        .and_then(serde_json::Value::as_str)
        .filter(|model| !model.is_empty())
        .map(str::to_string)
        .ok_or_else(|| GatewayError::InvalidRequest("Request must include a model".to_string()))
}

pub fn multipart_model(content_type: &str, body: &[u8]) -> Result<String, GatewayError> {
    let boundary = content_type
        .split(';')
        .map(str::trim)
        .find_map(|part| part.strip_prefix("boundary="))
        .map(|boundary| boundary.trim_matches('"'))
        .filter(|boundary| !boundary.is_empty())
        .ok_or_else(|| GatewayError::InvalidRequest("Multipart boundary is missing".to_string()))?;
    let delimiter = format!("--{}", boundary).into_bytes();
    let field_marker = b"name=\"model\"";
    let header_separator = b"\r\n\r\n";
    let line_end = b"\r\n";

    for start in body
        .windows(delimiter.len())
        .enumerate()
        .filter_map(|(index, window)| (window == delimiter.as_slice()).then_some(index))
    {
        let part = &body[start + delimiter.len()..];
        let end = part
            .windows(delimiter.len())
            .position(|window| window == delimiter.as_slice())
            .unwrap_or(part.len());
        let part = &part[..end];
        if !part
            .windows(field_marker.len())
            .any(|window| window == field_marker)
        {
            continue;
        }
        let Some(header_end) = part
            .windows(header_separator.len())
            .position(|window| window == header_separator)
        else {
            continue;
        };
        let value = &part[header_end + header_separator.len()..];
        let value_end = value
            .windows(line_end.len())
            .position(|window| window == line_end)
            .unwrap_or(value.len());
        let model = std::str::from_utf8(&value[..value_end])
            .map_err(|_| GatewayError::InvalidRequest("Multipart model must be UTF-8".to_string()))?
            .trim();
        if !model.is_empty() {
            return Ok(model.to_string());
        }
    }
    Err(GatewayError::InvalidRequest(
        "Multipart request must include a model field".to_string(),
    ))
}

fn provider_pass_through_response(upstream: ProviderPassThroughResponse) -> Response {
    let status = StatusCode::from_u16(upstream.status).unwrap_or(StatusCode::BAD_GATEWAY);
    let mut response = Response::new(Body::from(upstream.body));
    *response.status_mut() = status;
    for name in [
        header::CONTENT_TYPE,
        header::CONTENT_LENGTH,
        header::CONTENT_DISPOSITION,
        header::CACHE_CONTROL,
        header::RETRY_AFTER,
    ] {
        if let Some(value) = upstream.headers.get(&name) {
            response.headers_mut().insert(name, value.clone());
        }
    }
    response
}

async fn provider_pass_through(
    state: AppState,
    endpoint: ProviderPassThroughEndpoint,
    content_type: String,
    model: String,
    body: Bytes,
) -> Response {
    let started = Instant::now();
    let trace_id = generate_trace_id(None);
    let endpoint_label = match endpoint {
        ProviderPassThroughEndpoint::Embeddings => "embeddings",
        ProviderPassThroughEndpoint::ImageGenerations => "images/generations",
        ProviderPassThroughEndpoint::AudioTranscriptions => "audio/transcriptions",
        ProviderPassThroughEndpoint::AudioTranslations => "audio/translations",
    };
    let result = state
        .router
        .route_provider_pass_through(endpoint, &model, &content_type, body.to_vec())
        .await;
    let duration_ms = started.elapsed().as_millis() as u64;
    match result {
        Ok(response) => {
            tracing::info!(
                trace_id = %trace_id,
                endpoint = endpoint_label,
                model = %model,
                status = response.status,
                duration_ms,
                "Provider pass-through request completed"
            );
            provider_pass_through_response(response)
        }
        Err(error) => {
            tracing::warn!(
                trace_id = %trace_id,
                endpoint = endpoint_label,
                model = %model,
                duration_ms,
                error = %error,
                "Provider pass-through request failed"
            );
            error.into_response()
        }
    }
}

pub async fn embeddings(
    State(state): State<AppState>,
    headers: HeaderMap,
    body: Bytes,
) -> Response {
    let content_type = headers
        .get(header::CONTENT_TYPE)
        .and_then(|value| value.to_str().ok())
        .unwrap_or("application/json")
        .to_string();
    let model = match json_model(&body) {
        Ok(model) => model,
        Err(error) => return error.into_response(),
    };
    provider_pass_through(
        state,
        ProviderPassThroughEndpoint::Embeddings,
        content_type,
        model,
        body,
    )
    .await
}

pub async fn image_generations(
    State(state): State<AppState>,
    headers: HeaderMap,
    body: Bytes,
) -> Response {
    let content_type = headers
        .get(header::CONTENT_TYPE)
        .and_then(|value| value.to_str().ok())
        .unwrap_or("application/json")
        .to_string();
    let model = match json_model(&body) {
        Ok(model) => model,
        Err(error) => return error.into_response(),
    };
    provider_pass_through(
        state,
        ProviderPassThroughEndpoint::ImageGenerations,
        content_type,
        model,
        body,
    )
    .await
}

async fn audio_pass_through(
    state: AppState,
    headers: HeaderMap,
    body: Bytes,
    endpoint: ProviderPassThroughEndpoint,
) -> Response {
    let content_type = match headers
        .get(header::CONTENT_TYPE)
        .and_then(|value| value.to_str().ok())
        .filter(|value| {
            value
                .to_ascii_lowercase()
                .starts_with("multipart/form-data")
        }) {
        Some(value) => value.to_string(),
        None => {
            return GatewayError::InvalidRequest(
                "Audio requests must use multipart/form-data".to_string(),
            )
            .into_response()
        }
    };
    let model = match multipart_model(&content_type, &body) {
        Ok(model) => model,
        Err(error) => return error.into_response(),
    };
    provider_pass_through(state, endpoint, content_type, model, body).await
}

pub async fn audio_transcriptions(
    State(state): State<AppState>,
    headers: HeaderMap,
    body: Bytes,
) -> Response {
    audio_pass_through(
        state,
        headers,
        body,
        ProviderPassThroughEndpoint::AudioTranscriptions,
    )
    .await
}

pub async fn audio_translations(
    State(state): State<AppState>,
    headers: HeaderMap,
    body: Bytes,
) -> Response {
    audio_pass_through(
        state,
        headers,
        body,
        ProviderPassThroughEndpoint::AudioTranslations,
    )
    .await
}

// ---------------------------------------------------------------------------
// GET /v1/models  (Req 2.6, 2.12, 24.1-24.7)
// ---------------------------------------------------------------------------

/// Models list response in OpenAI format.
#[derive(Debug, Serialize)]
pub struct ModelsListResponse {
    pub object: String,
    pub data: Vec<Model>,
}

/// Aggregated models endpoint — queries all configured providers.
pub async fn list_models(
    State(state): State<AppState>,
    request: axum::extract::Request,
) -> Response {
    // Extract the authenticated key from request extensions (set by the vk
    // auth middleware when a valid virtual key is presented).
    let authenticated_key = request.extensions().get::<AuthenticatedKey>().cloned();

    let config = state.config.read().await;
    let mut all_models: Vec<Model> = Vec::new();
    let mut seen_ids = std::collections::HashSet::new();

    // List model group names first so clients can target groups directly
    for group in &config.model_groups {
        if seen_ids.insert(group.name.clone()) {
            all_models.push(Model {
                id: group.name.clone(),
                object: "model".to_string(),
                owned_by: "gateway".to_string(),
                created: None,
                context_window: None,
                max_completion_tokens: None,
                supports_vision: false,
            });
        }
    }

    // Also list individual model names for backward compatibility
    for group in &config.model_groups {
        for pm in &group.models {
            if seen_ids.insert(pm.model.clone()) {
                all_models.push(Model {
                    id: pm.model.clone(),
                    object: "model".to_string(),
                    owned_by: pm.provider.clone(),
                    created: None,
                    context_window: None,
                    max_completion_tokens: None,
                    supports_vision: false,
                });
            }
        }
    }

    // Include manually specified models from provider configs first, then add
    // the NVIDIA NIM built-in fallback. The existing dedup preserves manual
    // entries as explicit overrides.
    for provider in &config.providers {
        for model_id in &provider.manual_models {
            if seen_ids.insert(model_id.clone()) {
                all_models.push(Model {
                    id: model_id.clone(),
                    object: "model".to_string(),
                    owned_by: provider.name.clone(),
                    created: None,
                    context_window: None,
                    max_completion_tokens: None,
                    supports_vision: false,
                });
            }
        }

        if provider.provider_type == "nvidia_nim" {
            for mut model in crate::providers::nvidia_nim::fallback_models() {
                model.owned_by = provider.name.clone();
                if seen_ids.insert(model.id.clone()) {
                    all_models.push(model);
                }
            }
        }
    }

    // When the caller authenticated with a virtual key that has a model_access
    // restriction, filter the list to only models/groups in the allowed set.
    // A permitted model group name also grants visibility to all individual
    // models within that group.
    if let Some(ref authenticated) = authenticated_key {
        if let Some(ref allowed) = authenticated.model_access {
            // Build the expanded set: explicit allowed entries + all individual
            // model IDs belonging to allowed model groups.
            let mut visible: std::collections::HashSet<&str> =
                allowed.iter().map(|s| s.as_str()).collect();
            for group in &config.model_groups {
                if allowed.iter().any(|a| a == &group.name) {
                    for pm in &group.models {
                        visible.insert(&pm.model);
                    }
                }
            }
            all_models.retain(|m| visible.contains(m.id.as_str()));
        }
    }

    let response = ModelsListResponse {
        object: "list".to_string(),
        data: all_models,
    };

    Json(response).into_response()
}

// ---------------------------------------------------------------------------
// Assistants / Threads / Runs / Files / Fine-tuning (Req 2.7-2.11)
// ---------------------------------------------------------------------------

// --- Assistants (Req 2.7) ---
pub async fn create_assistant(
    State(state): State<AppState>,
    AssistantsIdentity(authenticated_key): AssistantsIdentity,
    Json(body): Json<serde_json::Value>,
) -> Response {
    assistants_result(
        state
            .assistants_store
            .create_assistant(authenticated_key.id.as_str(), body),
    )
}

pub async fn list_assistants(
    State(state): State<AppState>,
    AssistantsIdentity(authenticated_key): AssistantsIdentity,
    Query(params): Query<crate::assistants::ListParams>,
) -> Response {
    assistants_list_result(
        state
            .assistants_store
            .list_assistants(authenticated_key.id.as_str(), &params),
    )
}

pub async fn get_assistant(
    State(state): State<AppState>,
    Path(assistant_id): Path<String>,
    AssistantsIdentity(authenticated_key): AssistantsIdentity,
) -> Response {
    assistants_result(
        state
            .assistants_store
            .get_assistant(authenticated_key.id.as_str(), &assistant_id),
    )
}

pub async fn modify_assistant(
    State(state): State<AppState>,
    Path(assistant_id): Path<String>,
    AssistantsIdentity(authenticated_key): AssistantsIdentity,
    Json(body): Json<serde_json::Value>,
) -> Response {
    assistants_result(state.assistants_store.modify_assistant(
        authenticated_key.id.as_str(),
        &assistant_id,
        body,
    ))
}

pub async fn delete_assistant(
    State(state): State<AppState>,
    Path(assistant_id): Path<String>,
    AssistantsIdentity(authenticated_key): AssistantsIdentity,
) -> Response {
    assistants_result(
        state
            .assistants_store
            .delete_assistant(authenticated_key.id.as_str(), &assistant_id),
    )
}

// --- Threads (Req 2.8) ---
pub async fn create_thread(
    State(state): State<AppState>,
    AssistantsIdentity(authenticated_key): AssistantsIdentity,
    Json(body): Json<serde_json::Value>,
) -> Response {
    assistants_result(
        state
            .assistants_store
            .create_thread(authenticated_key.id.as_str(), body),
    )
}

pub async fn list_threads(
    State(state): State<AppState>,
    AssistantsIdentity(authenticated_key): AssistantsIdentity,
    Query(params): Query<crate::assistants::ListParams>,
) -> Response {
    assistants_list_result(
        state
            .assistants_store
            .list_threads(authenticated_key.id.as_str(), &params),
    )
}

pub async fn get_thread(
    State(state): State<AppState>,
    Path(thread_id): Path<String>,
    AssistantsIdentity(authenticated_key): AssistantsIdentity,
) -> Response {
    assistants_result(
        state
            .assistants_store
            .get_thread(authenticated_key.id.as_str(), &thread_id),
    )
}

pub async fn modify_thread(
    State(state): State<AppState>,
    Path(thread_id): Path<String>,
    AssistantsIdentity(authenticated_key): AssistantsIdentity,
    Json(body): Json<serde_json::Value>,
) -> Response {
    assistants_result(state.assistants_store.modify_thread(
        authenticated_key.id.as_str(),
        &thread_id,
        body,
    ))
}

pub async fn delete_thread(
    State(state): State<AppState>,
    Path(thread_id): Path<String>,
    AssistantsIdentity(authenticated_key): AssistantsIdentity,
) -> Response {
    assistants_result(
        state
            .assistants_store
            .delete_thread(authenticated_key.id.as_str(), &thread_id),
    )
}

// --- Runs (Req 2.9) ---
pub async fn create_run(
    State(state): State<AppState>,
    Path(thread_id): Path<String>,
    AssistantsIdentity(authenticated_key): AssistantsIdentity,
    Json(body): Json<serde_json::Value>,
) -> Response {
    let owner = authenticated_key.id.as_str().to_string();
    let execution = match state.assistants_store.start_run(&owner, &thread_id, body) {
        Ok(execution) => execution,
        Err(error) => return assistants_error_response(error),
    };
    let run_id = execution.run["id"].as_str().unwrap_or_default().to_string();
    let effective_model = execution.request.model.clone();
    if let Err(err) = state
        .virtual_key_manager
        .check_model_access(&authenticated_key, &effective_model)
    {
        if let Err(store_error) = state.assistants_store.fail_run(
            &owner,
            &thread_id,
            &run_id,
            "model not permitted for this key",
        ) {
            tracing::error!(error = %store_error, run_id, "Failed to persist run failure");
        }
        return access_denied_response_run(&err);
    }

    let (abort_tx, mut abort_rx) = tokio::sync::watch::channel(false);
    state.active_runs.insert(run_id.clone(), abort_tx);

    let router = Arc::clone(&state.router);
    let request = execution.request.clone();
    let route_future = router.route_request(&request, None);

    tokio::pin!(route_future);

    let route_outcome = tokio::select! {
        _ = abort_rx.changed() => {
            let cancel_msg = "run was cancelled by the client";
            if let Err(store_error) = state
                .assistants_store
                .cancel_run(&owner, &thread_id, &run_id)
            {
                tracing::error!(error = %store_error, run_id, "Failed to persist run cancellation");
            }
            return (
                StatusCode::OK,
                Json(serde_json::json!({
                    "id": run_id,
                    "object": "thread.run",
                    "status": "cancelled",
                    "message": cancel_msg
                })),
            )
                .into_response();
        }
        result = &mut route_future => result,
    };

    state.active_runs.remove(&run_id);

    match route_outcome {
        Ok(response) => {
            let completion_result = state
                .assistants_store
                .complete_run(&owner, &thread_id, &run_id, &response);
            match &completion_result {
                Ok(_) => {
                    extract_run_usage_and_record(
                        &state,
                        &authenticated_key,
                        &effective_model,
                        &response,
                    )
                    .await;
                }
                Err(store_err) => {
                    let err_msg = store_err.to_string();
                    if let Err(fail_err) = state
                        .assistants_store
                        .fail_run(&owner, &thread_id, &run_id, &err_msg)
                    {
                        tracing::error!(error = %fail_err, run_id, "Failed to persist run failure after completion error");
                    }
                }
            }
            assistants_result(completion_result)
        }
        Err(error) => {
            let message = error.to_string();
            if let Err(store_error) = state
                .assistants_store
                .fail_run(&owner, &thread_id, &run_id, &message)
            {
                tracing::error!(error = %store_error, run_id, "Failed to persist run failure");
            }
            error.into_response()
        }
    }
}

pub async fn list_runs(
    State(state): State<AppState>,
    Path(thread_id): Path<String>,
    AssistantsIdentity(authenticated_key): AssistantsIdentity,
    Query(params): Query<crate::assistants::ListParams>,
) -> Response {
    assistants_list_result(state.assistants_store.list_runs(
        authenticated_key.id.as_str(),
        &thread_id,
        &params,
    ))
}

pub async fn get_run(
    State(state): State<AppState>,
    Path((thread_id, run_id)): Path<(String, String)>,
    AssistantsIdentity(authenticated_key): AssistantsIdentity,
) -> Response {
    assistants_result(state.assistants_store.get_run(
        authenticated_key.id.as_str(),
        &thread_id,
        &run_id,
    ))
}

pub async fn cancel_run(
    State(state): State<AppState>,
    Path((thread_id, run_id)): Path<(String, String)>,
    AssistantsIdentity(authenticated_key): AssistantsIdentity,
) -> Response {
    if let Some(entry) = state.active_runs.remove(&run_id) {
        let _ = entry.1.send(true);
    }
    assistants_result(state.assistants_store.cancel_run(
        authenticated_key.id.as_str(),
        &thread_id,
        &run_id,
    ))
}

pub async fn list_run_steps(
    State(state): State<AppState>,
    Path((thread_id, run_id)): Path<(String, String)>,
    AssistantsIdentity(authenticated_key): AssistantsIdentity,
    Query(params): Query<crate::assistants::ListParams>,
) -> Response {
    assistants_list_result(state.assistants_store.list_run_steps(
        authenticated_key.id.as_str(),
        &thread_id,
        &run_id,
        &params,
    ))
}

// --- Messages on threads ---
pub async fn create_message(
    State(state): State<AppState>,
    Path(thread_id): Path<String>,
    AssistantsIdentity(authenticated_key): AssistantsIdentity,
    Json(body): Json<serde_json::Value>,
) -> Response {
    assistants_result(state.assistants_store.create_message(
        authenticated_key.id.as_str(),
        &thread_id,
        body,
    ))
}

pub async fn list_messages(
    State(state): State<AppState>,
    Path(thread_id): Path<String>,
    AssistantsIdentity(authenticated_key): AssistantsIdentity,
    Query(params): Query<crate::assistants::ListParams>,
) -> Response {
    assistants_list_result(state.assistants_store.list_messages(
        authenticated_key.id.as_str(),
        &thread_id,
        &params,
    ))
}

pub async fn get_message(
    State(state): State<AppState>,
    Path((thread_id, message_id)): Path<(String, String)>,
    AssistantsIdentity(authenticated_key): AssistantsIdentity,
) -> Response {
    assistants_result(state.assistants_store.get_message(
        authenticated_key.id.as_str(),
        &thread_id,
        &message_id,
    ))
}

pub async fn modify_message(
    State(state): State<AppState>,
    Path((thread_id, message_id)): Path<(String, String)>,
    AssistantsIdentity(authenticated_key): AssistantsIdentity,
    Json(body): Json<serde_json::Value>,
) -> Response {
    assistants_result(state.assistants_store.modify_message(
        authenticated_key.id.as_str(),
        &thread_id,
        &message_id,
        body,
    ))
}

pub async fn delete_message(
    State(state): State<AppState>,
    Path((thread_id, message_id)): Path<(String, String)>,
    AssistantsIdentity(authenticated_key): AssistantsIdentity,
) -> Response {
    assistants_result(state.assistants_store.delete_message(
        authenticated_key.id.as_str(),
        &thread_id,
        &message_id,
    ))
}

fn assistants_authentication_required() -> Response {
    (
        StatusCode::UNAUTHORIZED,
        Json(serde_json::json!({
            "error": {
                "message": "Authentication is required for Assistants API state",
                "type": "authentication_error",
                "code": "authentication_required"
            }
        })),
    )
        .into_response()
}

fn access_denied_response_run(err: &AccessError) -> Response {
    let AccessError::ModelDenied { model, allowed } = err;
    (
        StatusCode::FORBIDDEN,
        Json(serde_json::json!({
            "error": {
                "message": "Model not permitted for this virtual key",
                "type": "invalid_request_error",
                "code": "model_access_denied",
                "model": model,
                "allowed": allowed,
            }
        })),
    )
        .into_response()
}

async fn extract_run_usage_and_record(
    state: &AppState,
    authenticated_key: &AuthenticatedKey,
    model_group: &str,
    response: &OpenAIResponse,
) -> Option<String> {
    let input_tokens = response.usage.prompt_tokens as u64;
    let output_tokens = response.usage.completion_tokens as u64;
    if input_tokens == 0 && output_tokens == 0 {
        tracing::warn!(
            key_id = %authenticated_key.id,
            "run usage not recorded: response has zero token usage"
        );
        return None;
    }

    let cost_usd = {
        let cfg = state.config.read().await;
        crate::virtual_keys::auth::lookup_model_rates(&cfg, model_group, &response.model)
            .map(|(input_rate, output_rate)| {
                crate::virtual_keys::compute_cost(
                    input_tokens,
                    output_tokens,
                    input_rate,
                    output_rate,
                )
            })
            .unwrap_or(0.0)
    };

    let record = crate::virtual_keys::models::UsageRecord {
        key_id: authenticated_key.id.clone(),
        model_group: model_group.to_string(),
        model: response.model.clone(),
        input_tokens,
        output_tokens,
        cost_usd,
        timestamp: chrono::Utc::now(),
    };

    let tpm_tokens = input_tokens + output_tokens;
    let manager = Arc::clone(&state.virtual_key_manager);
    manager.record_tpm_usage(&record.key_id, tpm_tokens);
    let record_clone = record.clone();
    let manager_clone = Arc::clone(&manager);
    tokio::spawn(async move {
        if let Err(e) = manager_clone.record_usage(record_clone).await {
            tracing::warn!(error = %e, "failed to record virtual key run usage");
        }
    });

    Some(record.key_id)
}

fn assistants_list_result(
    result: Result<crate::assistants::ListPage, crate::assistants::StoreError>,
) -> Response {
    match result {
        Ok(page) => Json(page.into_openai_response()).into_response(),
        Err(error) => assistants_error_response(error),
    }
}

fn assistants_result(result: Result<serde_json::Value, crate::assistants::StoreError>) -> Response {
    match result {
        Ok(value) => Json(value).into_response(),
        Err(error) => assistants_error_response(error),
    }
}

fn assistants_error_response(error: crate::assistants::StoreError) -> Response {
    use crate::assistants::StoreError;
    let (status, error_type, message) = match error {
        StoreError::NotFound { object, id } => (
            StatusCode::NOT_FOUND,
            "invalid_request_error",
            format!("No {object} found with id '{id}'"),
        ),
        StoreError::InvalidRequest(message) => {
            (StatusCode::BAD_REQUEST, "invalid_request_error", message)
        }
        StoreError::TooLarge { object, max_bytes } => (
            StatusCode::PAYLOAD_TOO_LARGE,
            "invalid_request_error",
            format!("{object} exceeds the {max_bytes} byte storage limit"),
        ),
        StoreError::Database(_)
        | StoreError::Serialization(_)
        | StoreError::Io(_)
        | StoreError::Lock => {
            tracing::error!(error = %error, "Assistants store operation failed");
            (
                StatusCode::INTERNAL_SERVER_ERROR,
                "server_error",
                "The assistants store could not complete the request".to_string(),
            )
        }
    };
    (
        status,
        Json(serde_json::json!({
        "error": {
        "message": message,
        "type": error_type
        }
        })),
    )
        .into_response()
}

// --- Files (Req 2.10) ---
pub async fn upload_file(
    State(state): State<AppState>,
    AssistantsIdentity(authenticated_key): AssistantsIdentity,
    headers: HeaderMap,
    body: Bytes,
) -> Response {
    let content_type = match headers
        .get(header::CONTENT_TYPE)
        .and_then(|value| value.to_str().ok())
    {
        Some(content_type) => content_type,
        None => {
            return GatewayError::InvalidRequest("Content-Type is required".into()).into_response()
        }
    };
    let parts = match parse_multipart_fields(content_type, &body) {
        Ok(parts) => parts,
        Err(error) => return error.into_response(),
    };
    let file = match parts.iter().find(|part| part.name == "file") {
        Some(file) => file,
        None => {
            return GatewayError::InvalidRequest(
                "Multipart request must include a file field".into(),
            )
            .into_response()
        }
    };
    let purpose = parts
        .iter()
        .find(|part| part.name == "purpose")
        .and_then(|part| std::str::from_utf8(&part.data).ok())
        .unwrap_or("assistants")
        .trim()
        .to_string();
    assistants_result(state.assistants_store.create_file(
        authenticated_key.id.as_str(),
        file.filename.clone().unwrap_or_else(|| "upload".into()),
        purpose,
        file.data.clone(),
    ))
}

pub async fn list_files(
    State(state): State<AppState>,
    AssistantsIdentity(authenticated_key): AssistantsIdentity,
    Query(params): Query<crate::assistants::ListParams>,
) -> Response {
    assistants_list_result(
        state
            .assistants_store
            .list_files(authenticated_key.id.as_str(), &params),
    )
}

pub async fn get_file(
    State(state): State<AppState>,
    Path(file_id): Path<String>,
    AssistantsIdentity(authenticated_key): AssistantsIdentity,
) -> Response {
    assistants_result(
        state
            .assistants_store
            .get_file(authenticated_key.id.as_str(), &file_id),
    )
}

pub async fn delete_file(
    State(state): State<AppState>,
    Path(file_id): Path<String>,
    AssistantsIdentity(authenticated_key): AssistantsIdentity,
) -> Response {
    assistants_result(
        state
            .assistants_store
            .delete_file(authenticated_key.id.as_str(), &file_id),
    )
}

pub async fn get_file_content(
    State(state): State<AppState>,
    Path(file_id): Path<String>,
    AssistantsIdentity(authenticated_key): AssistantsIdentity,
) -> Response {
    match state
        .assistants_store
        .get_file_content(authenticated_key.id.as_str(), &file_id)
    {
        Ok(file) => {
            let mut response = Response::new(Body::from(file.content));
            response.headers_mut().insert(
                header::CONTENT_TYPE,
                HeaderValue::from_static("application/octet-stream"),
            );
            if let Ok(value) = HeaderValue::from_str(&format!(
                "attachment; filename=\"{}\"",
                file.filename.replace(['\\', '"'], "_")
            )) {
                response
                    .headers_mut()
                    .insert(header::CONTENT_DISPOSITION, value);
            }
            response
        }
        Err(error) => assistants_error_response(error),
    }
}

#[derive(Debug)]
struct MultipartField {
    name: String,
    filename: Option<String>,
    data: Vec<u8>,
}

fn parse_multipart_fields(
    content_type: &str,
    body: &[u8],
) -> Result<Vec<MultipartField>, GatewayError> {
    let boundary = content_type
        .split(';')
        .map(str::trim)
        .find_map(|part| part.strip_prefix("boundary="))
        .map(|boundary| boundary.trim_matches('"'))
        .filter(|boundary| !boundary.is_empty())
        .ok_or_else(|| GatewayError::InvalidRequest("Multipart boundary is missing".into()))?;
    let delimiter = format!("--{boundary}").into_bytes();
    let mut fields = Vec::new();
    for part in body.split(|byte| *byte == b'\n') {
        let _ = part;
    }
    let mut cursor = 0;
    while let Some(relative_start) = body[cursor..]
        .windows(delimiter.len())
        .position(|window| window == delimiter.as_slice())
    {
        let start = cursor + relative_start + delimiter.len();
        let Some(relative_end) = body[start..]
            .windows(delimiter.len())
            .position(|window| window == delimiter.as_slice())
        else {
            break;
        };
        let mut part = &body[start..start + relative_end];
        if part.starts_with(b"\r\n") {
            part = &part[2..];
        }
        if part.ends_with(b"\r\n") {
            part = &part[..part.len() - 2];
        }
        let Some(header_end) = part.windows(4).position(|window| window == b"\r\n\r\n") else {
            cursor = start + relative_end;
            continue;
        };
        let headers = std::str::from_utf8(&part[..header_end])
            .map_err(|_| GatewayError::InvalidRequest("Multipart headers must be UTF-8".into()))?;
        let disposition = headers
            .lines()
            .find(|line| {
                line.to_ascii_lowercase()
                    .starts_with("content-disposition:")
            })
            .ok_or_else(|| {
                GatewayError::InvalidRequest(
                    "Multipart field is missing Content-Disposition".into(),
                )
            })?;
        let name = disposition_parameter(disposition, "name").ok_or_else(|| {
            GatewayError::InvalidRequest("Multipart field name is missing".into())
        })?;
        let filename = disposition_parameter(disposition, "filename");
        fields.push(MultipartField {
            name,
            filename,
            data: part[header_end + 4..].to_vec(),
        });
        cursor = start + relative_end;
    }
    Ok(fields)
}

fn disposition_parameter(disposition: &str, parameter: &str) -> Option<String> {
    disposition.split(';').map(str::trim).find_map(|part| {
        let value = part.strip_prefix(&format!("{parameter}="))?;
        Some(value.trim_matches('"').to_string())
    })
}

// --- Fine-tuning (Req 2.11) ---
fn fine_tuning_unsupported() -> Response {
    (
        StatusCode::NOT_IMPLEMENTED,
        Json(serde_json::json!({
        "error": {
        "message": "Fine-tuning is not supported by the gateway's configured provider routing",
        "type": "unsupported_feature",
        "code": "unsupported_feature"
        }
        })),
    )
        .into_response()
}

/// Proxy a fine-tuning request to the OpenAI-compatible provider. Returns the
/// structured unsupported-feature response only when capability selection
/// fails (no OpenAI-compatible provider configured).
async fn fine_tuning_proxy(
    state: AppState,
    method: reqwest::Method,
    path_suffix: &str,
    body: Option<Vec<u8>>,
) -> Response {
    match state
        .router
        .route_fine_tuning_pass_through(method, path_suffix, body)
        .await
    {
        Ok(response) => provider_pass_through_response(response),
        Err(GatewayError::Provider {
            status_code: Some(501),
            ..
        }) => fine_tuning_unsupported(),
        Err(error) => error.into_response(),
    }
}

pub async fn create_fine_tuning_job(State(state): State<AppState>, body: Bytes) -> Response {
    fine_tuning_proxy(state, reqwest::Method::POST, "", Some(body.to_vec())).await
}

pub async fn list_fine_tuning_jobs(State(state): State<AppState>) -> Response {
    fine_tuning_proxy(state, reqwest::Method::GET, "", None).await
}

pub async fn get_fine_tuning_job(
    State(state): State<AppState>,
    Path(fine_tuning_id): Path<String>,
) -> Response {
    fine_tuning_proxy(
        state,
        reqwest::Method::GET,
        &format!("/{fine_tuning_id}"),
        None,
    )
    .await
}

pub async fn cancel_fine_tuning_job(
    State(state): State<AppState>,
    Path(fine_tuning_id): Path<String>,
) -> Response {
    fine_tuning_proxy(
        state,
        reqwest::Method::POST,
        &format!("/{fine_tuning_id}/cancel"),
        None,
    )
    .await
}

pub async fn list_fine_tuning_events(
    State(state): State<AppState>,
    Path(fine_tuning_id): Path<String>,
) -> Response {
    fine_tuning_proxy(
        state,
        reqwest::Method::GET,
        &format!("/{fine_tuning_id}/events"),
        None,
    )
    .await
}

// ---------------------------------------------------------------------------
// GET /metrics  (Req 20.7-20.11) — Prometheus exposition format
// ---------------------------------------------------------------------------

/// Prometheus metrics endpoint — returns metrics in Prometheus text exposition format.
/// No external prometheus client library; we format the text directly from MetricsSnapshot.
pub async fn prometheus_metrics(State(state): State<AppState>) -> Response {
    let snap = state.metrics.snapshot();
    let mut out = String::with_capacity(2048);

    // Helper: append a metric block
    macro_rules! metric {
        (counter $name:expr, $help:expr, $val:expr) => {
            out.push_str(&format!(
                "# HELP {} {}\n# TYPE {} counter\n{} {}\n",
                $name, $help, $name, $name, $val
            ));
        };
        (gauge $name:expr, $help:expr, $val:expr) => {
            out.push_str(&format!(
                "# HELP {} {}\n# TYPE {} gauge\n{} {}\n",
                $name, $help, $name, $name, $val
            ));
        };
    }

    // Req 20.8: request count
    metric!(counter "obey_api_requests_total", "Total number of requests", snap.request_count);

    // Req 20.8: active requests
    metric!(gauge "obey_api_active_requests", "Current active requests", snap.active_requests);

    // Req 20.9: response time (avg as gauge — histogram buckets would need raw data)
    metric!(gauge "obey_api_response_time_avg_ms", "Average response time in milliseconds", snap.avg_response_time_ms);

    // Request rate
    metric!(gauge "obey_api_request_rate_per_min", "Requests per minute", snap.request_rate_per_min);

    // Cumulative cost
    metric!(gauge "obey_api_cumulative_cost_dollars", "Cumulative cost in dollars", snap.cumulative_cost);

    // Req 20.8: per-provider request counts
    if !snap.provider_health.is_empty() {
        out.push_str("# HELP obey_api_provider_requests_total Total requests by provider\n");
        out.push_str("# TYPE obey_api_provider_requests_total counter\n");
        for ph in &snap.provider_health {
            out.push_str(&format!(
                "obey_api_provider_requests_total{{provider=\"{}\"}} {}\n",
                ph.provider, ph.total_requests
            ));
        }

        out.push_str("# HELP obey_api_provider_success_total Successful requests by provider\n");
        out.push_str("# TYPE obey_api_provider_success_total counter\n");
        for ph in &snap.provider_health {
            out.push_str(&format!(
                "obey_api_provider_success_total{{provider=\"{}\"}} {}\n",
                ph.provider, ph.successful_requests
            ));
        }

        out.push_str("# HELP obey_api_provider_failures_total Failed requests by provider\n");
        out.push_str("# TYPE obey_api_provider_failures_total counter\n");
        for ph in &snap.provider_health {
            out.push_str(&format!(
                "obey_api_provider_failures_total{{provider=\"{}\"}} {}\n",
                ph.provider, ph.failed_requests
            ));
        }

        // Req 20.9: per-provider avg response time (histogram proxy)
        out.push_str("# HELP obey_api_provider_response_time_avg_ms Average response time by provider in milliseconds\n");
        out.push_str("# TYPE obey_api_provider_response_time_avg_ms gauge\n");
        for ph in &snap.provider_health {
            out.push_str(&format!(
                "obey_api_provider_response_time_avg_ms{{provider=\"{}\"}} {}\n",
                ph.provider, ph.avg_response_time_ms
            ));
        }
    }

    // Req 20.10: circuit breaker state gauges
    let cb_states = state.router.get_circuit_breaker_states().await;
    if !cb_states.is_empty() {
        out.push_str("# HELP obey_api_circuit_breaker_state Circuit breaker state (0=closed, 1=open, 2=half_open)\n");
        out.push_str("# TYPE obey_api_circuit_breaker_state gauge\n");
        for (provider, state_label) in &cb_states {
            let val = match state_label.as_str() {
                "closed" => 0,
                "open" => 1,
                "half_open" => 2,
                _ => 0,
            };
            out.push_str(&format!(
                "obey_api_circuit_breaker_state{{provider=\"{}\",state=\"{}\"}} {}\n",
                provider, state_label, val
            ));
        }
    }

    // Req 20.11: cache hit rate gauge
    if let Some(rate) = snap.cache_hit_rate {
        metric!(gauge "obey_api_cache_hit_rate", "Cache hit rate (0.0 to 1.0)", rate);
    }

    // Cost by provider
    if !snap.cost_by_provider.is_empty() {
        out.push_str(
            "# HELP obey_api_cost_by_provider_dollars Cumulative cost by provider in dollars\n",
        );
        out.push_str("# TYPE obey_api_cost_by_provider_dollars gauge\n");
        for (provider, cost) in &snap.cost_by_provider {
            out.push_str(&format!(
                "obey_api_cost_by_provider_dollars{{provider=\"{}\"}} {}\n",
                provider, cost
            ));
        }
    }

    if !snap.retry_count_by_provider.is_empty() {
        out.push_str("# HELP obey_api_provider_retries_total Total retry attempts by provider\n");
        out.push_str("# TYPE obey_api_provider_retries_total counter\n");
        for (provider, retry_count) in &snap.retry_count_by_provider {
            out.push_str(&format!(
                "obey_api_provider_retries_total{{provider=\"{}\"}} {}\n",
                provider, retry_count
            ));
        }
    }

    if !snap.retry_delay_ms_by_provider.is_empty() {
        out.push_str("# HELP obey_api_provider_retry_delay_ms_total Total retry delay applied by provider in milliseconds\n");
        out.push_str("# TYPE obey_api_provider_retry_delay_ms_total counter\n");
        for (provider, retry_delay_ms) in &snap.retry_delay_ms_by_provider {
            out.push_str(&format!(
                "obey_api_provider_retry_delay_ms_total{{provider=\"{}\"}} {}\n",
                provider, retry_delay_ms
            ));
        }
    }

    if !snap.budget_limit_by_provider.is_empty() {
        out.push_str("# HELP obey_api_provider_budget_limit_dollars Configured budget limit by provider in dollars\n");
        out.push_str("# TYPE obey_api_provider_budget_limit_dollars gauge\n");
        for (provider, budget_limit) in &snap.budget_limit_by_provider {
            out.push_str(&format!(
                "obey_api_provider_budget_limit_dollars{{provider=\"{}\"}} {}\n",
                provider, budget_limit
            ));
        }
    }

    if !snap.budget_exhaustions_by_provider.is_empty() {
        out.push_str("# HELP obey_api_provider_budget_exhaustions_total Total provider budget exhaustion events\n");
        out.push_str("# TYPE obey_api_provider_budget_exhaustions_total counter\n");
        for (provider, budget_exhaustions) in &snap.budget_exhaustions_by_provider {
            out.push_str(&format!(
                "obey_api_provider_budget_exhaustions_total{{provider=\"{}\"}} {}\n",
                provider, budget_exhaustions
            ));
        }
    }

    if !snap.unknown_cost_by_provider.is_empty() {
        out.push_str("# HELP obey_api_provider_unknown_cost_total Total successful responses without usable usage data by provider\n");
        out.push_str("# TYPE obey_api_provider_unknown_cost_total counter\n");
        for (provider, unknown_cost) in &snap.unknown_cost_by_provider {
            out.push_str(&format!(
                "obey_api_provider_unknown_cost_total{{provider=\"{}\"}} {}\n",
                provider, unknown_cost
            ));
        }
    }

    if !snap.rate_limit_exhaustions_by_provider.is_empty() {
        out.push_str("# HELP obey_api_provider_rate_limit_exhaustions_total Total provider skips caused by local rate-limit exhaustion\n");
        out.push_str("# TYPE obey_api_provider_rate_limit_exhaustions_total counter\n");
        for (provider, rate_limit_exhaustions) in &snap.rate_limit_exhaustions_by_provider {
            out.push_str(&format!(
                "obey_api_provider_rate_limit_exhaustions_total{{provider=\"{}\"}} {}\n",
                provider, rate_limit_exhaustions
            ));
        }
    }

    // Compression metrics are emitted alongside guardrail metrics from the
    // same in-repo metrics state.
    state.metrics.write_guardrail_prometheus(&mut out);
    state.metrics.write_compression_prometheus(&mut out);
    state.metrics.write_tool_compression_prometheus(&mut out);
    state.metrics.write_structured_output_prometheus(&mut out);
    state
        .loop_detector
        .metrics
        .write_prometheus(&mut out, state.loop_detector.sessions.len());
    state.router.search_metrics().write_prometheus(&mut out);
    if let Some(memory) = state.memory_system.read().await.clone() {
        memory.metrics.write_prometheus(&mut out);
    }

    (
        StatusCode::OK,
        [(
            axum::http::header::CONTENT_TYPE,
            "text/plain; version=0.0.4; charset=utf-8",
        )],
        out,
    )
        .into_response()
}

// ---------------------------------------------------------------------------
// POST /admin/config/reload  (Req 26.1-26.7)
// ---------------------------------------------------------------------------

/// Reload configuration from disk without restarting the gateway.
///
/// On success the new config is applied to future requests, circuit breaker
/// states are reset, and the models list cache is invalidated.
/// On validation failure the existing config is kept and an error is returned.
#[allow(dead_code)]
pub async fn reload_config(State(state): State<AppState>) -> Response {
    let config_path = state.config_path.as_ref();

    // Read & validate new config from disk (Req 26.1, 26.2)
    let new_config = match load_and_validate_config(config_path) {
        Ok(cfg) => cfg,
        Err(e) => {
            // Req 26.3: keep existing config, return error
            tracing::warn!("Config reload validation failed: {}", e);
            return (
                StatusCode::BAD_REQUEST,
                Json(serde_json::json!({
                    "error": {
                        "message": format!("Configuration validation failed: {}", e),
                        "type": "configuration_error"
                    }
                })),
            )
                .into_response();
        }
    };

    // Apply new config (Req 26.4)
    apply_runtime_config_update(&state, new_config).await;

    // Req 26.6: models list cache is implicitly cleared because list_models
    // reads from the config on every call.

    tracing::info!(
        "Configuration reloaded successfully from {}",
        config_path.display()
    );

    (
        StatusCode::OK,
        Json(serde_json::json!({
            "status": "ok",
            "message": "Configuration reloaded successfully"
        })),
    )
        .into_response()
}
