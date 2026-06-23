//! OpenAI API endpoint handlers for the OBEY-API gateway.
//!
//! Requirements: 2.1-2.12

use axum::{
    extract::{Json, State},
    http::{HeaderMap, HeaderName, HeaderValue, StatusCode},
    response::{
        sse::{Event, KeepAlive, Sse},
        IntoResponse, Response,
    },
};
use futures::StreamExt;
use serde::{Deserialize, Serialize};
use std::convert::Infallible;
use std::sync::atomic::Ordering;
use std::sync::Arc;

use std::time::Duration;

use crate::cache::ExactCache;
use crate::config::{load_and_validate_config, StreamingConfig};
use crate::error::{AggregatedError, GatewayError, ProviderAttempt};
use crate::gateway::apply_runtime_config_update;
use crate::logger::LogEntry;
use crate::metrics::Metrics;
use crate::models::openai::{Choice, OpenAIRequest, OpenAIResponse};
use crate::providers::Model;
use crate::router::trace_id::generate_trace_id;
use crate::router::StreamingResponse;

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
}

impl RequestLogContext {
    fn from_response(request: &OpenAIRequest, trace_id: String, duration_ms: u64, response: &crate::models::openai::OpenAIResponse) -> Self {
        Self {
            trace_id,
            status_code: StatusCode::OK.as_u16(),
            duration_ms,
            provider: response.extra.get("gateway_provider").and_then(|v| v.as_str()).unwrap_or_default().to_string(),
            requested_model: request.model.clone(),
            responded_model: response.extra.get("gateway_responded_model").and_then(|v| v.as_str()).map(|s| s.to_string()).or_else(|| if response.model.is_empty() { None } else { Some(response.model.clone()) }),
            cost: response.extra.get("gateway_cost").and_then(|v| v.as_f64()).unwrap_or(0.0),
            error_message: None,
        }
    }

    fn from_error(request: &OpenAIRequest, trace_id: String, duration_ms: u64, error: &GatewayError) -> Self {
        let provider = match error {
            GatewayError::Provider { provider, .. } => provider.clone(),
            GatewayError::AllProvidersFailed(agg) => agg.attempts.first().map(|attempt| attempt.provider.clone()).unwrap_or_default(),
            _ => String::new(),
        };
        let error_message = match error {
            GatewayError::Provider { message, .. } => Some(message.clone()),
            GatewayError::AllProvidersFailed(agg) => {
                Some(agg.attempts.iter()
                    .map(|a| format!("[{}] {}", a.provider, a.error))
                    .collect::<Vec<_>>()
                    .join("; "))
            }
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
        model: context.responded_model.clone().unwrap_or_else(|| context.requested_model.clone()),
        provider: context.provider.clone(),
        status_code: context.status_code,
        duration_ms: context.duration_ms,
        cost: context.cost,
        request_body: None,
        response_body: context.error_message.clone(),
        requested_model: Some(request.model.clone()),
        responded_model: context.responded_model.clone(),
    };
    if let Err(e) = state.logger.log(entry) {
        tracing::warn!(error = %e, trace_id = %context.trace_id, "Failed to write request log entry");
    }
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

use super::AppState;

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
            GatewayError::Provider { provider: _, message: _, status_code } => {
                let sc = status_code
                    .and_then(|c| StatusCode::from_u16(c).ok())
                    .unwrap_or(StatusCode::BAD_GATEWAY);
                (
                    sc,
                    serde_json::json!({ "error": { "message": self.to_string(), "type": "provider_error" } }),
                )
            },
            _ => (
                StatusCode::INTERNAL_SERVER_ERROR,
                serde_json::json!({ "error": { "message": self.to_string(), "type": "server_error" } }),
            ),
        };

        (status, Json(body)).into_response()
    }
}

// ---------------------------------------------------------------------------
// GET /health  (Req 20.1-20.3)
// ---------------------------------------------------------------------------

/// Health check endpoint — returns 200 when operational, 503 when shutting down.
pub async fn health_check(State(state): State<AppState>) -> Response {
    if state.shutting_down.load(Ordering::Relaxed) {
        (StatusCode::SERVICE_UNAVAILABLE, Json(serde_json::json!({ "status": "shutting_down" }))).into_response()
    } else {
        (StatusCode::OK, Json(serde_json::json!({ "status": "ok" }))).into_response()
    }
}

// ---------------------------------------------------------------------------
// POST /v1/chat/completions  (Req 2.1)
// ---------------------------------------------------------------------------

/// Chat completions handler — streaming and non-streaming.
pub async fn chat_completions(
    State(state): State<AppState>,
    headers: HeaderMap,
    Json(request): Json<OpenAIRequest>,
) -> Response {
    tracing::info!(model = %request.model, stream = request.stream, "Received chat completion request");
    let trace_id = trace_id_from_headers(&headers);
    if request.stream {
        chat_completions_stream(state, request, trace_id).await
    } else {
        chat_completions_non_stream(state, request, trace_id).await
    }
}

async fn chat_completions_non_stream(state: AppState, request: OpenAIRequest, trace_id: String) -> Response {
    state.metrics.start_request();
    let start = std::time::Instant::now();
    tracing::debug!(model = %request.model, "Routing non-stream request");

    // Tier-1: exact-match in-memory cache.  Lookup is always safe — eligibility
    // (deterministic temperature, n=1) is enforced internally.  Tool-using
    // requests ARE looked up here; only writes are gated below by
    // `should_cache_response`.
    if let Some(cached_json) = state.exact_cache.get(&request) {
        if let Ok(resp) = serde_json::from_str::<crate::models::openai::OpenAIResponse>(&cached_json) {
            state.metrics.record_cache_hit();
            state.metrics.complete_request(start.elapsed().as_millis() as u64);
            let mut http = Json(resp).into_response();
            attach_trace_id_header(&mut http, &trace_id);
            return http;
        }
    } else if state.exact_cache.is_eligible(&request) {
        state.metrics.record_cache_miss();
    }

    // Tier-2: semantic cache (paraphrase match).  Skipped for tool-using
    // requests — semantic similarity across different tool surfaces is too
    // risky for code agents.
    let skip_semantic = request.extra.contains_key("tools") || request.extra.contains_key("tool_choice");
    if !skip_semantic {
        if let Some(ref cache) = state.cache {
            match cache.get(&request).await {
                Ok(Some(cached_response)) => {
                    state.metrics.record_cache_hit();
                    state.metrics.complete_request(start.elapsed().as_millis() as u64);
                    match serde_json::from_str::<crate::models::openai::OpenAIResponse>(&cached_response) {
                        Ok(resp) => {
                            let mut response = Json(resp).into_response();
                            attach_trace_id_header(&mut response, &trace_id);
                            return response;
                        }
                        Err(_) => {
                            tracing::warn!("Failed to parse cached response, falling through to provider");
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

    match state.router.route_request(&request).await {
        Ok(response) => {
            // Cache responses that are safe to replay.  Filter applies to
            // both tiers (no tool_calls, complete finish_reason, etc.).
            let cacheable = crate::router::router::Router::should_cache_response(&response);
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
            let duration_ms = start.elapsed().as_millis() as u64;
            state.metrics.complete_request(duration_ms);
            let log_context = RequestLogContext::from_response(&request, trace_id.clone(), duration_ms, &response);
            log_request(&state, &request, &log_context);
            let mut http_response = Json(response).into_response();
            attach_trace_id_header(&mut http_response, &trace_id);
            http_response
        }
        Err(e) => {
            let duration_ms = start.elapsed().as_millis() as u64;
            state.metrics.complete_request(duration_ms);
            let log_context = RequestLogContext::from_error(&request, trace_id.clone(), duration_ms, &e);
            log_request(&state, &request, &log_context);
            let mut response = e.into_response();
            attach_trace_id_header(&mut response, &trace_id);
            response
        }
    }
}

async fn chat_completions_stream(state: AppState, request: OpenAIRequest, trace_id: String) -> Response {
    state.metrics.start_request();
    let start = std::time::Instant::now();
    tracing::debug!(
        trace_id = %trace_id,
        model = %request.model,
        "Client requested streaming response; gateway currently buffers the full upstream response before synthesizing SSE"
    );

    // Streaming Reliability (Req 2): resolve the effective streaming settings
    // up front so every SSE path below (cache replay, early event, and the
    // buffer-and-replay fallback) can apply the configured keep-alive interval.
    // An absent `streaming` section falls back to defaults.
    let streaming_config = state.config.read().await.streaming.clone().unwrap_or_default();

    // Tier-1 cache lookup for streaming requests.  The cached payload is a
    // full non-streaming `OpenAIResponse` JSON; we re-emit it as SSE chunks
    // using the same path as a fresh provider response.  This means a single
    // cached entry serves both stream and non-stream callers identically.
    if let Some(cached_json) = state.exact_cache.get(&request) {
        if let Ok(cached_resp) = serde_json::from_str::<OpenAIResponse>(&cached_json) {
            state.metrics.record_cache_hit();
            state.metrics.complete_request(start.elapsed().as_millis() as u64);
            let stream_trace_id = trace_id.clone();
            let stream = async_stream::stream! {
                tracing::debug!(trace_id = %stream_trace_id, "Streaming cached response from exact cache");
                for chunk in streaming_chunks_from_response(&cached_resp) {
                    yield Ok::<_, Infallible>(Event::default().data(chunk.to_string()));
                }
                yield Ok(Event::default().data("[DONE]"));
            };
            let mut sse = Sse::new(stream).keep_alive(build_keepalive(&streaming_config)).into_response();
            attach_trace_id_header(&mut sse, &trace_id);
            return sse;
        }
    } else if state.exact_cache.is_eligible(&request) {
        state.metrics.record_cache_miss();
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

        let streaming_config_relay = streaming_config.clone();
        let stream = async_stream::stream! {
            // Early synthetic event (Req 1.1, 1.2, 1.3).
            yield Ok::<_, Infallible>(Event::default().data(early_chunk.to_string()));

            // Task 5.5: dispatch through the streaming router so capable
            // providers stream in real time (PassThrough) while providers that
            // need response transformation fall back to buffer-and-replay
            // (Buffered) — both behind the early event already flushed above.
            match state.router.route_request_streaming(&request).await {
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
                    yield Ok(Event::default().data("[DONE]"));
                }
                Ok(StreamingResponse::PassThrough { byte_stream, provider, model }) => {
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
                    //   non-streaming path, skipping `tried_providers`. That list
                    //   is the natural bound — each provider is tried for
                    //   pass-through at most once.
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
                        (cfg.retry.max_retries_per_provider, cfg.providers.len() + 1)
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
                    let mut current_stream = byte_stream;
                    let mut current_provider = provider;
                    let mut current_model = model;

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
                        let outcome = Arc::new(tokio::sync::Mutex::new(RelayOutcome::Completed));
                        let relay = relay_passthrough_stream(
                            current_stream,
                            streaming_config_relay.clone(),
                            stream_trace_id.clone(),
                            total_timeout,
                            state.exact_cache.clone(),
                            state.metrics.clone(),
                            request.clone(),
                            outcome.clone(),
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
                            RelayOutcome::Completed => break 'failover,
                            // Post-content failure (Req 4.2): the relay already
                            // emitted the graceful error event + `[DONE]`. We
                            // cannot transparently fail over mid-content, so
                            // account the failed attempt against the circuit
                            // breaker + metrics (Req 4.5) and stop — no retry.
                            RelayOutcome::FailedAfterContent(reason) => {
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
                                tried_providers.push(current_provider.clone());
                                // Req 4.3: record this pre-content failure for the
                                // aggregated error in case every provider fails.
                                streaming_attempts.push(ProviderAttempt::new(
                                    current_provider.clone(),
                                    current_model.clone(),
                                    reason.clone(),
                                    None,
                                ));

                                match state
                                    .router
                                    .route_request_streaming_excluding(&request, &tried_providers)
                                    .await
                                {
                                    // Another eligible provider — relay it,
                                    // reusing the SAME early-event id (Req 4.4:
                                    // do NOT emit a second role event).
                                    Ok(StreamingResponse::PassThrough { byte_stream, provider, model }) => {
                                        current_stream = byte_stream;
                                        current_provider = provider;
                                        current_model = model;
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

            state.metrics.complete_request(start.elapsed().as_millis() as u64);
        };

        let mut sse = Sse::new(stream).keep_alive(build_keepalive(&streaming_config)).into_response();
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
    let response = match state.router.route_request(&request).await {
        Ok(resp) => resp,
        Err(e) => {
            let duration_ms = start.elapsed().as_millis() as u64;
            state.metrics.complete_request(duration_ms);
            let log_context = RequestLogContext::from_error(&request, trace_id.clone(), duration_ms, &e);
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
    let log_context = RequestLogContext::from_response(&request, trace_id.clone(), duration_ms, &response);
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
        yield Ok(Event::default().data("[DONE]"));
    };

    state.metrics.complete_request(start.elapsed().as_millis() as u64);

    Sse::new(stream)
        .keep_alive(build_keepalive(&streaming_config))
        .into_response()
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
            .interval(Duration::from_secs(streaming_config.keepalive_interval_seconds))
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
fn merge_streaming_attempts(mut attempts: Vec<ProviderAttempt>, error: GatewayError) -> GatewayError {
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
    Completed,
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
fn relay_passthrough_stream(
    upstream: reqwest::Response,
    streaming_config: StreamingConfig,
    trace_id: String,
    total_timeout: Duration,
    exact_cache: Arc<ExactCache>,
    metrics: Arc<Metrics>,
    request: OpenAIRequest,
    outcome: Arc<tokio::sync::Mutex<RelayOutcome>>,
) -> impl futures::Stream<Item = Result<Event, Infallible>> {
    async_stream::stream! {
        let chunk_timeout = Duration::from_secs(streaming_config.chunk_timeout_seconds);
        let deadline = tokio::time::Instant::now() + total_timeout;
        let mut byte_stream = upstream.bytes_stream();
        let mut buffer = String::new();

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
                    buffer.push_str(&String::from_utf8_lossy(&bytes));
                }
                Ok(Some(Err(e))) => {
                    // Req 3.4/3.6: network error mid-stream.
                    let message = format!("Stream error: {}", e);
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

            // Task 6.1: a clean finish — record the outcome so the handler's
            // failover loop stops here (no retry).
            *outcome.lock().await = RelayOutcome::Completed;
            // Req 3.7 / 3.10: clean completion — reassemble the accumulated chunks
            // into a full response, cache it if eligible, and record usage.
            if !sse_accumulator.is_empty() {
                match crate::router::router::Router::reassemble_sse_response(&sse_accumulator) {
                    Ok(assembled) => {
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
            yield Ok(Event::default().data("[DONE]"));
        }
    }
}

fn streaming_chunks_from_response(response: &OpenAIResponse) -> Vec<serde_json::Value> {
    build_streaming_chunks(response, None)
}

/// Variant used after an early synthetic `role: assistant` event has already
/// been emitted (Req 1.5). It suppresses the duplicate role delta and reuses
/// the early event's pre-generated `id`/`created` for every subsequent chunk so
/// the whole stream shares a single id (task 2.2).
fn streaming_chunks_after_early_event(
    response: &OpenAIResponse,
    id: &str,
    created: i64,
) -> Vec<serde_json::Value> {
    build_streaming_chunks(response, Some((id, created)))
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
        .map(|c| {
            match &c.message.content {
                serde_json::Value::String(s) => serde_json::Value::String(s.clone()),
                serde_json::Value::Null => serde_json::Value::Null,
                other => other.clone(),
            }
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
        let tc_type = first_tc.get("type").and_then(|v| v.as_str()).unwrap_or("function");
        let fn_name = first_tc.get("function")
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

        let fn_args = first_tc.get("function")
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
            let tc_type = tc.get("type").and_then(|v| v.as_str()).unwrap_or("function");
            let fn_name = tc.get("function")
                .and_then(|f| f.get("name"))
                .and_then(|n| n.as_str())
                .unwrap_or("");
            let fn_args = tc.get("function")
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
        build_keepalive, classify_relay_line, classify_stream_error, chunk_carries_content,
        early_event_chunk,
        emit_sse_error_event, relay_passthrough_stream, sse_error_payload,
        streaming_chunks_after_early_event, streaming_chunks_from_response, RelayLineAction,
        RelayOutcome,
    };
    use crate::config::StreamingConfig;
    use crate::error::{AggregatedError, GatewayError, ProviderAttempt};
    use crate::models::openai::{Choice, Message, OpenAIRequest, OpenAIResponse, Usage};
    use futures::StreamExt;
    use std::time::Duration;

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
        assert_eq!(chunks[1]["choices"][0]["delta"]["reasoning"], "thinking step");
        assert_eq!(chunks[2]["choices"][0]["delta"]["content"], "final answer");
    }

    #[test]
    fn streaming_chunks_preserve_reasoning_content_field_name() {
        let mut extra = serde_json::Map::new();
        extra.insert("reasoning_content".to_string(), serde_json::json!("hidden chain"));

        let response = base_response(Message {
            role: "assistant".to_string(),
            content: serde_json::json!("visible answer"),
            extra,
        });

        let chunks = streaming_chunks_from_response(&response);

        assert_eq!(chunks[1]["choices"][0]["delta"]["reasoning_content"], "hidden chain");
        assert_eq!(chunks[2]["choices"][0]["delta"]["content"], "visible answer");
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
        assert_eq!(payload["error"]["message"], "Provider did not respond within 30s");
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
        assert_eq!(payload["error"]["message"], "Response exceeded 120s total timeout");
        assert_eq!(payload["error"]["trace_id"], "tr-total-1");
    }

    /// Req 5.5: every error frame includes a non-empty `trace_id` field, and the
    /// `emit_sse_error_event` data event is built from this exact payload (its
    /// first event's data string equals the serialized payload).
    #[test]
    fn sse_error_payload_includes_trace_id() {
        let payload = sse_error_payload("chunk_timeout_error", "stalled", "tr-corr-99");
        assert_eq!(payload["error"]["trace_id"], "tr-corr-99");
        assert!(payload["error"]["trace_id"].as_str().is_some_and(|s: &str| !s.is_empty()));
    }

    // -- Stream error classification (task 4.2) ------------------------------

    fn attempt_with_error(error: String) -> ProviderAttempt {
        ProviderAttempt::new(
            "openai".to_string(),
            "gpt-4".to_string(),
            error,
            Some(504),
        )
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
        assert_eq!(classify_relay_line("{not json"), RelayLineAction::SkipMalformed);
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
        let stream = futures::stream::once(async move {
            Ok::<_, std::io::Error>(body.as_bytes())
        });
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
        std::sync::Arc::new(tokio::sync::Mutex::new(RelayOutcome::Completed))
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
        );
        let _events: Vec<_> = stream.collect().await;

        let cached = exact_cache.get(&request).expect("response should be cached");
        let resp: OpenAIResponse =
            serde_json::from_str(&cached).expect("cached payload is valid JSON");
        assert_eq!(resp.choices[0].message.content, serde_json::json!("Hello world"));
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
        let body = axum::body::to_bytes(resp.into_body(), 1024 * 1024).await.unwrap();
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
        );

        let text = relay_to_sse_text(stream).await;
        // The first chunk was forwarded before the stall.
        assert!(text.contains("\"content\":\"a\""), "first chunk forwarded before timeout");
        // The inter-chunk timeout surfaces as the precise type (Req 3.12, 5.3).
        assert!(
            text.contains("\"type\":\"chunk_timeout_error\""),
            "stall must produce a chunk_timeout_error frame, got: {text}"
        );
        assert!(text.trim_end().ends_with("data: [DONE]"), "error frame followed by [DONE]");
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
        );

        let text = relay_to_sse_text(stream).await;
        assert!(text.contains("\"content\":\"a\""), "first chunk forwarded before timeout");
        assert!(
            text.contains("\"type\":\"total_timeout_error\""),
            "exceeding the total budget must produce a total_timeout_error frame, got: {text}"
        );
        assert!(text.trim_end().ends_with("data: [DONE]"), "error frame followed by [DONE]");
    }

    /// Task 6.1 / Req 4.1: a `data:` payload carrying real content/tool_call/
    /// reasoning deltas is detected, while a role-only delta is not (so it does
    /// not block pre-content failover).
    #[test]
    fn chunk_carries_content_distinguishes_role_only_from_content() {
        let role_only = r#"{"choices":[{"index":0,"delta":{"role":"assistant"}}]}"#;
        assert!(!chunk_carries_content(role_only), "role-only delta is not content");

        let content = r#"{"choices":[{"index":0,"delta":{"content":"hi"}}]}"#;
        assert!(chunk_carries_content(content), "content delta counts");

        let empty_content = r#"{"choices":[{"index":0,"delta":{"content":""}}]}"#;
        assert!(!chunk_carries_content(empty_content), "empty content does not count");

        let tool = r#"{"choices":[{"index":0,"delta":{"tool_calls":[{"index":0,"function":{"name":"f"}}]}}]}"#;
        assert!(chunk_carries_content(tool), "tool_calls delta counts");

        let reasoning = r#"{"choices":[{"index":0,"delta":{"reasoning_content":"think"}}]}"#;
        assert!(chunk_carries_content(reasoning), "reasoning_content counts");

        assert!(!chunk_carries_content("not json"), "malformed is not content");
    }

    /// Build a streaming `reqwest::Response` whose body errors immediately,
    /// before any byte is delivered — drives the relay's pre-content failure
    /// path.
    fn erroring_streaming_response() -> reqwest::Response {
        let stream = futures::stream::once(async move {
            Err::<&[u8], _>(std::io::Error::new(std::io::ErrorKind::ConnectionReset, "reset"))
        });
        let http_response = axum::http::Response::new(reqwest::Body::wrap_stream(stream));
        reqwest::Response::from(http_response)
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
        );
        let events: Vec<_> = stream.collect().await;
        assert!(events.is_empty(), "pre-content failure must emit no SSE events");
        let guard = outcome.lock().await;
        match &*guard {
            RelayOutcome::FailedBeforeContent(_) => {}
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
        );
        let _events: Vec<_> = stream.collect().await;
        assert!(matches!(&*outcome.lock().await, RelayOutcome::Completed));
    }
}

// ---------------------------------------------------------------------------
// POST /v1/completions  (Req 2.2)
// ---------------------------------------------------------------------------

/// Legacy completions endpoint — pass-through proxy.
pub async fn completions(
    State(state): State<AppState>,
    headers: HeaderMap,
    Json(request): Json<OpenAIRequest>,
) -> Response {
    // Reuse chat completions routing; the OpenAI completions format is close enough
    // for provider pass-through. Full translation can be refined later.
    let trace_id = trace_id_from_headers(&headers);
    chat_completions_non_stream(state, request, trace_id).await
}

// ---------------------------------------------------------------------------
// POST /v1/embeddings  (Req 2.3)
// ---------------------------------------------------------------------------

#[derive(Debug, Deserialize)]
#[allow(dead_code)]
pub struct EmbeddingRequest {
    pub model: String,
    pub input: serde_json::Value,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub encoding_format: Option<String>,
}

/// Embeddings endpoint — pass-through proxy to configured provider.
pub async fn embeddings(
    State(_state): State<AppState>,
    Json(_request): Json<EmbeddingRequest>,
) -> Response {
    // Pass-through proxy placeholder — will forward to the provider that owns the model.
    (
        StatusCode::NOT_IMPLEMENTED,
        Json(serde_json::json!({
            "error": { "message": "Embeddings endpoint: pass-through not yet wired to provider client", "type": "not_implemented" }
        })),
    )
        .into_response()
}

// ---------------------------------------------------------------------------
// POST /v1/images/generations  (Req 2.4)
// ---------------------------------------------------------------------------

#[derive(Debug, Deserialize)]
#[allow(dead_code)]
pub struct ImageGenerationRequest {
    pub model: Option<String>,
    pub prompt: String,
    #[serde(default = "default_image_count")]
    pub n: u32,
    pub size: Option<String>,
}

fn default_image_count() -> u32 {
    1
}

pub async fn image_generations(
    State(_state): State<AppState>,
    Json(_request): Json<ImageGenerationRequest>,
) -> Response {
    (
        StatusCode::NOT_IMPLEMENTED,
        Json(serde_json::json!({
            "error": { "message": "Images endpoint: pass-through not yet wired to provider client", "type": "not_implemented" }
        })),
    )
        .into_response()
}

// ---------------------------------------------------------------------------
// POST /v1/audio/transcriptions  (Req 2.5)
// POST /v1/audio/translations    (Req 2.5)
// ---------------------------------------------------------------------------

pub async fn audio_transcriptions(headers: HeaderMap) -> Response {
    let _ = headers; // multipart handling deferred to provider pass-through
    (
        StatusCode::NOT_IMPLEMENTED,
        Json(serde_json::json!({
            "error": { "message": "Audio transcriptions endpoint: pass-through not yet wired", "type": "not_implemented" }
        })),
    )
        .into_response()
}

pub async fn audio_translations(headers: HeaderMap) -> Response {
    let _ = headers;
    (
        StatusCode::NOT_IMPLEMENTED,
        Json(serde_json::json!({
            "error": { "message": "Audio translations endpoint: pass-through not yet wired", "type": "not_implemented" }
        })),
    )
        .into_response()
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
pub async fn list_models(State(state): State<AppState>) -> Response {
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
                });
            }
        }
    }

    // Include manually specified models from provider configs
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
                });
            }
        }
    }

    let response = ModelsListResponse {
        object: "list".to_string(),
        data: all_models,
    };

    Json(response).into_response()
}

// ---------------------------------------------------------------------------
// Assistants / Threads / Runs / Files / Fine-tuning  (Req 2.7-2.11)
// Pass-through stubs — these forward to the upstream provider once wired.
// ---------------------------------------------------------------------------

/// Generic pass-through stub returning 501 for unimplemented endpoints.
async fn not_implemented_stub(endpoint: &str) -> Response {
    (
        StatusCode::NOT_IMPLEMENTED,
        Json(serde_json::json!({
            "error": {
                "message": format!("{} endpoint: pass-through not yet wired to provider client", endpoint),
                "type": "not_implemented"
            }
        })),
    )
        .into_response()
}

// --- Assistants (Req 2.7) ---
pub async fn create_assistant(State(_s): State<AppState>, body: String) -> Response {
    let _ = body;
    not_implemented_stub("Assistants").await
}
pub async fn list_assistants(State(_s): State<AppState>) -> Response {
    not_implemented_stub("Assistants").await
}
pub async fn get_assistant(State(_s): State<AppState>) -> Response {
    not_implemented_stub("Assistants").await
}
pub async fn modify_assistant(State(_s): State<AppState>, body: String) -> Response {
    let _ = body;
    not_implemented_stub("Assistants").await
}
pub async fn delete_assistant(State(_s): State<AppState>) -> Response {
    not_implemented_stub("Assistants").await
}

// --- Threads (Req 2.8) ---
pub async fn create_thread(State(_s): State<AppState>, body: String) -> Response {
    let _ = body;
    not_implemented_stub("Threads").await
}
pub async fn get_thread(State(_s): State<AppState>) -> Response {
    not_implemented_stub("Threads").await
}
pub async fn modify_thread(State(_s): State<AppState>, body: String) -> Response {
    let _ = body;
    not_implemented_stub("Threads").await
}
pub async fn delete_thread(State(_s): State<AppState>) -> Response {
    not_implemented_stub("Threads").await
}

// --- Runs (Req 2.9) ---
pub async fn create_run(State(_s): State<AppState>, body: String) -> Response {
    let _ = body;
    not_implemented_stub("Runs").await
}
pub async fn list_runs(State(_s): State<AppState>) -> Response {
    not_implemented_stub("Runs").await
}
pub async fn get_run(State(_s): State<AppState>) -> Response {
    not_implemented_stub("Runs").await
}
pub async fn cancel_run(State(_s): State<AppState>) -> Response {
    not_implemented_stub("Runs").await
}

// --- Messages on threads ---
pub async fn create_message(State(_s): State<AppState>, body: String) -> Response {
    let _ = body;
    not_implemented_stub("Messages").await
}
pub async fn list_messages(State(_s): State<AppState>) -> Response {
    not_implemented_stub("Messages").await
}

// --- Files (Req 2.10) ---
pub async fn upload_file(headers: HeaderMap) -> Response {
    let _ = headers;
    not_implemented_stub("Files").await
}
pub async fn list_files(State(_s): State<AppState>) -> Response {
    not_implemented_stub("Files").await
}
pub async fn get_file(State(_s): State<AppState>) -> Response {
    not_implemented_stub("Files").await
}
pub async fn delete_file(State(_s): State<AppState>) -> Response {
    not_implemented_stub("Files").await
}
pub async fn get_file_content(State(_s): State<AppState>) -> Response {
    not_implemented_stub("Files").await
}

// --- Fine-tuning (Req 2.11) ---
pub async fn create_fine_tuning_job(State(_s): State<AppState>, body: String) -> Response {
    let _ = body;
    not_implemented_stub("Fine-tuning").await
}
pub async fn list_fine_tuning_jobs(State(_s): State<AppState>) -> Response {
    not_implemented_stub("Fine-tuning").await
}
pub async fn get_fine_tuning_job(State(_s): State<AppState>) -> Response {
    not_implemented_stub("Fine-tuning").await
}
pub async fn cancel_fine_tuning_job(State(_s): State<AppState>) -> Response {
    not_implemented_stub("Fine-tuning").await
}
pub async fn list_fine_tuning_events(State(_s): State<AppState>) -> Response {
    not_implemented_stub("Fine-tuning").await
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
        out.push_str("# HELP obey_api_cost_by_provider_dollars Cumulative cost by provider in dollars\n");
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

    (
        StatusCode::OK,
        [(axum::http::header::CONTENT_TYPE, "text/plain; version=0.0.4; charset=utf-8")],
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

    tracing::info!("Configuration reloaded successfully from {}", config_path.display());

    (
        StatusCode::OK,
        Json(serde_json::json!({
            "status": "ok",
            "message": "Configuration reloaded successfully"
        })),
    )
        .into_response()
}
