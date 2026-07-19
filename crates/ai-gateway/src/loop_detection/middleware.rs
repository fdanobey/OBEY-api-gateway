use crate::config::Config;
use crate::loop_detection::{
    enforcement::EnforcementEngine,
    events::{LoopDetectionEvent, LoopEventBus},
    eviction::{eviction_loop, insert_bounded},
    fingerprint::ToolCallFingerprint,
    injection::InjectionEngine,
    metrics::LoopDetectionMetrics,
    scorer::ConfidenceScorer,
    session::{RequestRecord, ResponseDescriptor, SessionResolver, SessionState},
    signals::SignalComputer,
    simhash,
};
use crate::models::openai::OpenAIRequest;
use crate::virtual_keys::models::AuthenticatedKey;
use axum::{
    body::{to_bytes, Body},
    http::{header, HeaderValue, Request, Response, StatusCode},
};
use dashmap::DashMap;
use serde_json::json;
use std::{
    future::Future,
    pin::Pin,
    sync::{
        atomic::{AtomicBool, Ordering},
        Arc,
    },
    task::{Context, Poll},
    time::{Duration, Instant},
};
use tokio::sync::{Mutex, RwLock};
use tower::{Layer, Service};

const CHAT_COMPLETIONS_PATH: &str = "/v1/chat/completions";

#[derive(Clone)]
pub struct LoopDetectorState {
    pub sessions: Arc<DashMap<String, SessionState>>,
    pub config: Arc<RwLock<Config>>,
    pub detector_config: Arc<RwLock<crate::loop_detection::LoopDetectionConfig>>,
    pub enabled: Arc<AtomicBool>,
    pub metrics: Arc<LoopDetectionMetrics>,
    pub events: Arc<LoopEventBus>,
    session_locks: Arc<DashMap<String, Arc<Mutex<()>>>>,
}

impl LoopDetectorState {
    pub fn new(config: Arc<RwLock<Config>>, detector_config: crate::loop_detection::LoopDetectionConfig) -> Self {
        let enabled = detector_config.enabled;
        Self {
            sessions: Arc::new(DashMap::new()),
            config,
            detector_config: Arc::new(RwLock::new(detector_config)),
            enabled: Arc::new(AtomicBool::new(enabled)),
            metrics: Arc::new(LoopDetectionMetrics::new()),
            events: Arc::new(LoopEventBus::new()),
            session_locks: Arc::new(DashMap::new()),
        }
    }

    pub fn clear_sessions(&self) {
        self.sessions.clear();
        self.session_locks.clear();
    }

    pub async fn apply_config(&self, config: crate::loop_detection::LoopDetectionConfig) {
        self.enabled.store(config.enabled, Ordering::Relaxed);
        *self.detector_config.write().await = config;
        self.clear_sessions();
    }

    pub fn spawn_eviction(&self, config: &crate::loop_detection::LoopDetectionConfig) {
        let sessions = Arc::clone(&self.sessions);
        let metrics = Arc::clone(&self.metrics);
        let interval = Duration::from_secs(u64::from(config.eviction_interval_seconds));
        let ttl = Duration::from_secs(u64::from(config.session_timeout_minutes) * 60);
        tokio::spawn(eviction_loop(sessions, metrics, interval, ttl, 1_000));
    }
}

#[derive(Clone)]
pub struct LoopDetectorLayer {
    state: Arc<LoopDetectorState>,
}

impl LoopDetectorLayer {
    pub fn new(state: Arc<LoopDetectorState>) -> Self {
        Self { state }
    }
}

impl<S> Layer<S> for LoopDetectorLayer {
    type Service = LoopDetectorService<S>;

    fn layer(&self, inner: S) -> Self::Service {
        LoopDetectorService {
            inner,
            state: Arc::clone(&self.state),
        }
    }
}

#[derive(Clone)]
pub struct LoopDetectorService<S> {
    inner: S,
    state: Arc<LoopDetectorState>,
}

impl<S> Service<Request<Body>> for LoopDetectorService<S>
where
    S: Service<Request<Body>, Response = Response<Body>> + Clone + Send + 'static,
    S::Future: Send + 'static,
    S::Error: Send + 'static,
{
    type Response = Response<Body>;
    type Error = S::Error;
    type Future = Pin<Box<dyn Future<Output = Result<Self::Response, Self::Error>> + Send>>;

    fn poll_ready(&mut self, context: &mut Context<'_>) -> Poll<Result<(), Self::Error>> {
        self.inner.poll_ready(context)
    }

    fn call(&mut self, request: Request<Body>) -> Self::Future {
        let mut inner = self.inner.clone();
        let state = Arc::clone(&self.state);
        if !state.enabled.load(Ordering::Relaxed) || !is_loop_detection_request(&request) {
            return Box::pin(async move { inner.call(request).await });
        }

        Box::pin(async move {
            let config = state.detector_config.read().await.clone();
            if !config.enabled {
                return inner.call(request).await;
            }

            let vk = request.extensions().get::<AuthenticatedKey>().cloned();
            let session_id = SessionResolver::resolve(
                &request,
                &state.sessions,
                vk.as_ref().map(|key| key.id.as_str()),
                Duration::from_secs(u64::from(config.session_timeout_minutes) * 60),
            );
            let Some(session_id) = session_id else {
                return inner.call(request).await;
            };

            let content_length = request
                .headers()
                .get(header::CONTENT_LENGTH)
                .and_then(|value| value.to_str().ok())
                .and_then(|value| value.parse::<usize>().ok());
            let max_body = {
                let config = state.config.read().await;
                config.server.max_request_size_mb as usize * 1024 * 1024
            };
            if content_length.is_some_and(|length| length > max_body) {
                return inner.call(request).await;
            }

            let (parts, body) = request.into_parts();
            let bytes = match to_bytes(body, max_body).await {
                Ok(bytes) => bytes,
                Err(_) => {
                    return inner.call(Request::from_parts(parts, Body::empty())).await;
                }
            };
            let mut openai_request = match serde_json::from_slice::<OpenAIRequest>(&bytes) {
                Ok(request) => request,
                Err(_) => {
                    return inner.call(Request::from_parts(parts, Body::from(bytes))).await;
                }
            };

            let effective_config = vk
                .as_ref()
                .and_then(|key| key.loop_detection.as_ref())
                .and_then(|override_config| override_config.merge(&config).ok())
                .unwrap_or(config);
            let custom_template = vk
                .as_ref()
                .and_then(|key| key.loop_detection.as_ref())
                .and_then(|override_config| override_config.break_instruction_template.clone());
            let record = request_record(&openai_request);
            let session_lock = state
                .session_locks
                .entry(session_id.clone())
                .or_insert_with(|| Arc::new(Mutex::new(())))
                .clone();
            let _session_guard = session_lock.lock().await;

            if !state.sessions.contains_key(&session_id) {
                insert_bounded(
                    &state.sessions,
                    session_id.clone(),
                    SessionState::new(
                        vk.as_ref().map(|key| key.id.clone()),
                        effective_config.history_depth as usize,
                    ),
                    effective_config.max_sessions as usize,
                    Some(&state.metrics),
                );
            }

            let decision = {
                let mut session = state.sessions.get_mut(&session_id).expect("session inserted");
                let signals = SignalComputer::compute(
                    &session,
                    &record,
                    None,
                    &effective_config,
                    None,
                );
                let score = ConfidenceScorer::score(
                    &signals,
                    &effective_config.weights,
                    session.smoothed_confidence,
                    effective_config.ema_alpha,
                    session.request_count.saturating_add(1) as usize,
                );
                session.dominant_signal = score.dominant_signal;
                session.record_signals(signals);
                state.metrics.record_confidence(
                    session.vk_id.as_deref(),
                    score.confidence,
                );
                let decision = EnforcementEngine::evaluate(
                    score.confidence,
                    &mut session,
                    &effective_config,
                );
                InjectionEngine::inject(
                    &mut openai_request,
                    &decision,
                    &mut session,
                    &effective_config,
                    custom_template.as_deref(),
                );
                session.record_request(&record);
                if decision.transitioned {
                    state.metrics.record_enforcement(
                        enforcement_label(decision.level),
                        session.vk_id.as_deref(),
                    );
                    tracing::info!(
                        session_id = %session_id,
                        virtual_key = session.vk_id.as_deref().unwrap_or("none"),
                        level = enforcement_label(decision.level),
                        confidence = score.confidence,
                        dominant_signal = decision.dominant_signal,
                        content_similarity = signals.content_similarity,
                        tool_call_repetition = signals.tool_call_repetition,
                        response_stagnation = signals.response_stagnation,
                        token_velocity = signals.token_velocity,
                        error_cycling = signals.error_cycling,
                        context_growth = signals.context_growth,
                        cost_velocity = signals.cost_velocity,
                        consecutive_count = session.consecutive_high,
                        request_id = request_id(&parts),
                        timestamp = %chrono::Utc::now(),
                        "Agent loop enforcement action"
                    );
                }
                decision
            };

            if decision.should_hard_stop {
                let confidence = state
                    .sessions
                    .get(&session_id)
                    .map_or(0.0, |session| session.smoothed_confidence);
                state.events.publish(LoopDetectionEvent::new(
                    session_id.clone(),
                    confidence,
                    decision.level,
                    decision.dominant_signal,
                ));
                if let Some(session) = state.sessions.get(&session_id) {
                    tracing::error!(
                        session_id = %session_id,
                        virtual_key = session.vk_id.as_deref().unwrap_or("none"),
                        confidence = session.smoothed_confidence,
                        dominant_signal = session.dominant_signal,
                        request_count = session.request_count,
                        total_tokens = session.total_tokens,
                        total_cost = session.total_cost,
                        recent_request_hashes = ?session.request_hashes,
                        recent_tool_fingerprints = ?session.tool_fingerprints,
                        recent_responses = ?session.response_descriptors,
                        escalation_history = ?session.escalation_history,
                        "Agent loop hard-stop"
                    );
                }
                return Ok(hard_stop_response(
                    &session_id,
                    state
                        .sessions
                        .get(&session_id)
                        .map_or(0.0, |session| session.smoothed_confidence),
                    decision.dominant_signal,
                ));
            }
            if decision.should_throttle {
                tokio::time::sleep(Duration::from_secs(u64::from(
                    effective_config.throttle_delay_seconds,
                )))
                .await;
            }
            drop(_session_guard);

            let body = serde_json::to_vec(&openai_request).expect("OpenAI request serializes");
            let mut response = inner
                .call(Request::from_parts(parts, Body::from(body)))
                .await?;
            if !is_streaming_response(&response) {
                record_response_descriptor(&state, &session_id, &mut response).await;
            }
            let confidence = state
                .sessions
                .get(&session_id)
                .map_or(0.0, |session| session.smoothed_confidence);
            state.events.publish(LoopDetectionEvent::new(
                session_id.clone(),
                confidence,
                decision.level,
                decision.dominant_signal,
            ));
            if decision.should_warn {
                if let Ok(value) = HeaderValue::from_str(&format!(
                    "{confidence:.3}; dominant_signal={}",
                    decision.dominant_signal
                )) {
                    response.headers_mut().insert("x-loop-warning", value);
                }
            }
            Ok(response)
        })
    }
}

fn is_loop_detection_request(request: &Request<Body>) -> bool {
    request.method() == axum::http::Method::POST
        && request.uri().path() == CHAT_COMPLETIONS_PATH
        && request
            .headers()
            .get(header::CONTENT_TYPE)
            .and_then(|value| value.to_str().ok())
            .is_some_and(|value| value.to_ascii_lowercase().contains("application/json"))
}

fn request_record(request: &OpenAIRequest) -> RequestRecord {
    let messages = serde_json::to_value(&request.messages).unwrap_or_default();
    let tool_calls = request
        .messages
        .iter()
        .flat_map(|message| {
            message
                .extra
                .get("tool_calls")
                .and_then(serde_json::Value::as_array)
                .cloned()
                .unwrap_or_default()
        })
        .collect::<Vec<_>>();
    let content_tokens = request
        .messages
        .iter()
        .map(|message| message.content_as_text().split_whitespace().count() as u32)
        .sum();
    let new_information_tokens = estimate_new_information_tokens(request);
    RequestRecord {
        content_simhash: simhash::compute_messages(&messages),
        tool_call_fingerprint: ToolCallFingerprint::from_json(&serde_json::Value::Array(tool_calls.clone())),
        context_token_count: content_tokens,
        new_information_tokens,
        token_count: content_tokens,
        cost: 0.0,
        has_tool_calls: !tool_calls.is_empty(),
        tool_names: tool_calls
            .iter()
            .filter_map(|call| call.pointer("/function/name").and_then(serde_json::Value::as_str))
            .map(str::to_string)
            .collect(),
        timestamp: Instant::now(),
    }
}

fn estimate_new_information_tokens(request: &OpenAIRequest) -> u32 {
    let Some(message) = request.messages.last() else {
        return 0;
    };
    let mut unique = std::collections::HashSet::new();
    for token in message.content_as_text().split_whitespace() {
        unique.insert(token.to_ascii_lowercase());
    }
    if let Some(tool_calls) = message.extra.get("tool_calls").and_then(serde_json::Value::as_array) {
        for call in tool_calls {
            if let Some(arguments) = call.pointer("/function/arguments").and_then(serde_json::Value::as_str) {
                unique.extend(arguments.split_whitespace().map(str::to_ascii_lowercase));
            }
        }
    }
    unique.len().min(u32::MAX as usize) as u32
}

fn enforcement_label(level: crate::loop_detection::EnforcementLevel) -> &'static str {
    match level {
        crate::loop_detection::EnforcementLevel::None => "none",
        crate::loop_detection::EnforcementLevel::Warn => "warn",
        crate::loop_detection::EnforcementLevel::Throttle => "throttle",
        crate::loop_detection::EnforcementLevel::Inject => "inject",
        crate::loop_detection::EnforcementLevel::HardStop => "hard_stop",
    }
}

fn request_id(parts: &axum::http::request::Parts) -> &str {
    parts
        .headers
        .get("x-request-id")
        .and_then(|value| value.to_str().ok())
        .unwrap_or("none")
}

fn is_streaming_response(response: &Response<Body>) -> bool {
    response
        .headers()
        .get(header::CONTENT_TYPE)
        .and_then(|value| value.to_str().ok())
        .is_some_and(|value| value.to_ascii_lowercase().contains("text/event-stream"))
}

async fn record_response_descriptor(
    state: &LoopDetectorState,
    session_id: &str,
    response: &mut Response<Body>,
) {
    let status_error = response.status().is_client_error() || response.status().is_server_error();
    let content_type_json = response
        .headers()
        .get(header::CONTENT_TYPE)
        .and_then(|value| value.to_str().ok())
        .is_some_and(|value| value.to_ascii_lowercase().contains("application/json"));
    if !content_type_json {
        return;
    }
    let body = std::mem::replace(response.body_mut(), Body::empty());
    let bytes = match to_bytes(body, 64 * 1024 * 1024).await {
        Ok(bytes) => bytes,
        Err(_) => return,
    };
    let value = serde_json::from_slice::<serde_json::Value>(&bytes).ok();
    let token_count = value
        .as_ref()
        .and_then(|value| value.pointer("/usage/total_tokens"))
        .and_then(serde_json::Value::as_u64)
        .unwrap_or_else(|| String::from_utf8_lossy(&bytes).split_whitespace().count() as u64)
        .min(u64::from(u32::MAX)) as u32;
    let provider_error = value
        .as_ref()
        .is_some_and(|value| value.get("error").is_some());
    let response_cost = value
        .as_ref()
        .and_then(|value| value.get("gateway_cost"))
        .and_then(serde_json::Value::as_f64)
        .filter(|cost| cost.is_finite() && *cost >= 0.0)
        .unwrap_or(0.0);
    if let Some(mut session) = state.sessions.get_mut(session_id) {
        let previous_error = session
            .response_descriptors
            .back()
            .is_some_and(|descriptor| descriptor.is_error);
        let content_similarity = session
            .signal_history
            .back()
            .map_or(0.0, |signals| signals.content_similarity);
        let is_error = status_error || provider_error;
        session.total_cost += response_cost;
        if let Some(cost) = session.recent_costs.back_mut() {
            *cost += response_cost;
        }
        if is_error && previous_error && content_similarity > 0.8 {
            session.error_retry_cycles = session.error_retry_cycles.saturating_add(1);
        } else if !is_error {
            session.error_retry_cycles = 0;
        }
        session.record_response(ResponseDescriptor {
            token_count,
            block_type_hash: response_block_hash(&bytes),
            is_error,
        });
    }
    *response.body_mut() = Body::from(bytes);
}

fn response_block_hash(bytes: &[u8]) -> u64 {
    use std::hash::{Hash, Hasher};
    let text = String::from_utf8_lossy(bytes);
    let sequence = text
        .lines()
        .map(|line| {
            let trimmed = line.trim_start();
            if trimmed.starts_with("```") {
                "code"
            } else if trimmed.starts_with("- ") || trimmed.starts_with("* ") {
                "list"
            } else {
                "text"
            }
        })
        .collect::<Vec<_>>();
    let mut hasher = std::collections::hash_map::DefaultHasher::new();
    sequence.hash(&mut hasher);
    hasher.finish()
}

fn hard_stop_response(session_id: &str, confidence: f32, dominant_signal: &str) -> Response<Body> {
    Response::builder()
        .status(StatusCode::TOO_MANY_REQUESTS)
        .header(header::CONTENT_TYPE, "application/json")
        .header(header::RETRY_AFTER, "60")
        .body(Body::from(
            json!({
                "error": {
                    "reason": "loop_detected",
                    "session_id": session_id,
                    "confidence": confidence,
                    "dominant_signal": dominant_signal,
                    "enforcement_level": "hard_stop"
                }
            })
            .to_string(),
        ))
        .expect("hard-stop response")
}
