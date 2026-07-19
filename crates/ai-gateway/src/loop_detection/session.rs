use crate::loop_detection::signals::SignalValues;
use axum::{body::Body, http::Request};
use chrono::{DateTime, Utc};
use dashmap::DashMap;
use std::{
    collections::VecDeque,
    time::{Duration, Instant},
};
use uuid::Uuid;

pub const SESSION_ID_HEADER: &str = "x-session-id";
pub type SessionId = String;

#[derive(Debug, Clone, Copy, Default, PartialEq, Eq, PartialOrd, Ord)]
pub enum EnforcementLevel {
    #[default]
    None,
    Warn,
    Throttle,
    Inject,
    HardStop,
}

impl EnforcementLevel {
    pub fn next(self) -> Self {
        match self {
            Self::None => Self::Warn,
            Self::Warn => Self::Throttle,
            Self::Throttle => Self::Inject,
            Self::Inject => Self::HardStop,
            Self::HardStop => Self::HardStop,
        }
    }

    pub fn previous(self) -> Self {
        match self {
            Self::None => Self::None,
            Self::Warn => Self::None,
            Self::Throttle => Self::Warn,
            Self::Inject => Self::Throttle,
            Self::HardStop => Self::Inject,
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct ResponseDescriptor {
    pub token_count: u32,
    pub block_type_hash: u64,
    pub is_error: bool,
}

#[derive(Debug, Clone, PartialEq)]
pub struct EscalationEvent {
    pub timestamp: DateTime<Utc>,
    pub from_level: EnforcementLevel,
    pub to_level: EnforcementLevel,
    pub confidence: f32,
}

#[derive(Debug, Clone)]
pub struct RequestRecord {
    pub content_simhash: u64,
    pub tool_call_fingerprint: Option<u64>,
    pub context_token_count: u32,
    pub new_information_tokens: u32,
    pub token_count: u32,
    pub cost: f64,
    pub has_tool_calls: bool,
    pub tool_names: Vec<String>,
    pub timestamp: Instant,
}

#[derive(Debug, Clone)]
pub struct SessionState {
    pub vk_id: Option<String>,
    pub request_hashes: VecDeque<u64>,
    pub tool_fingerprints: VecDeque<u64>,
    pub response_descriptors: VecDeque<ResponseDescriptor>,
    pub timestamps: VecDeque<Instant>,
    pub recent_token_counts: VecDeque<u32>,
    pub recent_costs: VecDeque<f64>,
    pub context_token_counts: VecDeque<u32>,
    pub total_tokens: u64,
    pub total_cost: f64,
    pub error_count: u32,
    pub error_retry_cycles: u32,
    pub consecutive_tool_fingerprint_count: u32,
    pub enforcement_level: EnforcementLevel,
    pub consecutive_high: u32,
    pub consecutive_low: u32,
    pub smoothed_confidence: f32,
    pub injected_at_level: bool,
    pub request_count: u32,
    pub peak_confidence: f32,
    pub last_active: Instant,
    pub escalation_history: Vec<EscalationEvent>,
    pub signal_history: VecDeque<SignalValues>,
    pub dominant_signal: &'static str,
    history_depth: usize,
}

impl SessionState {
    pub fn new(vk_id: Option<String>, history_depth: usize) -> Self {
        let history_depth = history_depth.max(1);
        Self {
            vk_id,
            request_hashes: VecDeque::with_capacity(history_depth),
            tool_fingerprints: VecDeque::with_capacity(history_depth),
            response_descriptors: VecDeque::with_capacity(history_depth),
            timestamps: VecDeque::with_capacity(history_depth),
            recent_token_counts: VecDeque::with_capacity(history_depth),
            recent_costs: VecDeque::with_capacity(history_depth),
            context_token_counts: VecDeque::with_capacity(history_depth),
            total_tokens: 0,
            total_cost: 0.0,
            error_count: 0,
            error_retry_cycles: 0,
            consecutive_tool_fingerprint_count: 0,
            enforcement_level: EnforcementLevel::None,
            consecutive_high: 0,
            consecutive_low: 0,
            smoothed_confidence: 0.0,
            injected_at_level: false,
            request_count: 0,
            peak_confidence: 0.0,
            last_active: Instant::now(),
            escalation_history: Vec::new(),
            signal_history: VecDeque::with_capacity(history_depth),
            dominant_signal: "none",
            history_depth,
        }
    }

    pub fn history_depth(&self) -> usize {
        self.history_depth
    }

    pub fn record_request(&mut self, request: &RequestRecord) {
        push_bounded(
            &mut self.request_hashes,
            request.content_simhash,
            self.history_depth,
        );
        if let Some(fingerprint) = request.tool_call_fingerprint {
            push_bounded(&mut self.tool_fingerprints, fingerprint, self.history_depth);
        }
        push_bounded(&mut self.timestamps, request.timestamp, self.history_depth);
        push_bounded(
            &mut self.recent_token_counts,
            request.token_count,
            self.history_depth,
        );
        push_bounded(&mut self.recent_costs, request.cost, self.history_depth);
        push_bounded(
            &mut self.context_token_counts,
            request.context_token_count,
            self.history_depth,
        );
        self.total_tokens = self
            .total_tokens
            .saturating_add(u64::from(request.token_count));
        self.total_cost += request.cost;
        self.request_count = self.request_count.saturating_add(1);
        self.last_active = request.timestamp;
    }

    pub fn record_response(&mut self, response: ResponseDescriptor) {
        if response.is_error {
            self.error_count = self.error_count.saturating_add(1);
        }
        push_bounded(&mut self.response_descriptors, response, self.history_depth);
    }

    pub fn record_signals(&mut self, signals: SignalValues) {
        push_bounded(&mut self.signal_history, signals, self.history_depth);
    }
}

fn push_bounded<T>(values: &mut VecDeque<T>, value: T, limit: usize) {
    if values.len() == limit {
        values.pop_front();
    }
    values.push_back(value);
}

pub struct SessionResolver;

impl SessionResolver {
    pub fn resolve(
        request: &Request<Body>,
        sessions: &DashMap<SessionId, SessionState>,
        vk_id: Option<&str>,
        session_timeout: Duration,
    ) -> Option<SessionId> {
        if let Some(session_id) = request
            .headers()
            .get(SESSION_ID_HEADER)
            .and_then(|value| value.to_str().ok())
            .filter(|value| is_valid_explicit_session_id(value))
        {
            return Some(session_id.to_string());
        }

        let vk_id = vk_id.filter(|value| !value.is_empty())?;
        let now = Instant::now();
        sessions
            .iter()
            .filter(|entry| entry.value().vk_id.as_deref() == Some(vk_id))
            .filter_map(|entry| {
                let idle = now.saturating_duration_since(entry.value().last_active);
                (idle <= session_timeout).then(|| (entry.key().clone(), entry.value().last_active))
            })
            .max_by_key(|(_, last_active)| *last_active)
            .map(|(session_id, _)| session_id)
            .or_else(|| Some(format!("vk:{vk_id}:{}", Uuid::new_v4())))
    }
}

pub fn is_valid_explicit_session_id(value: &str) -> bool {
    !value.is_empty()
        && value.len() <= 256
        && value.bytes().all(|byte| (0x20..=0x7e).contains(&byte))
}
