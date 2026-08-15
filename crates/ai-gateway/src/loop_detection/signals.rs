use crate::loop_detection::{
    config::LoopDetectionConfig,
    fingerprint::repetition_score,
    session::{RequestRecord, ResponseDescriptor, SessionState},
    simhash,
};
use std::time::Duration;

#[derive(Debug, Clone, Copy, Default, PartialEq)]
pub struct SignalValues {
    pub content_similarity: f32,
    pub tool_call_repetition: f32,
    pub response_stagnation: f32,
    pub token_velocity: f32,
    pub error_cycling: f32,
    pub context_growth: f32,
    pub cost_velocity: f32,
}

impl SignalValues {
    pub fn iter(self) -> [(&'static str, f32); 7] {
        [
            ("content_similarity", self.content_similarity),
            ("tool_call_repetition", self.tool_call_repetition),
            ("response_stagnation", self.response_stagnation),
            ("token_velocity", self.token_velocity),
            ("error_cycling", self.error_cycling),
            ("context_growth", self.context_growth),
            ("cost_velocity", self.cost_velocity),
        ]
    }
}

pub struct SignalComputer;

impl SignalComputer {
    pub fn compute(
        session: &SessionState,
        request: &RequestRecord,
        response: Option<&ResponseDescriptor>,
        config: &LoopDetectionConfig,
        cost_rate: Option<f64>,
    ) -> SignalValues {
        if session.request_count < 2 {
            return SignalValues::default();
        }

        let content_similarity = content_similarity(session, request.content_simhash);
        let tool_call_repetition_value =
            tool_call_repetition(session, request).max(discovery_repetition(session, request));
        SignalValues {
            content_similarity,
            tool_call_repetition: tool_call_repetition_value,
            response_stagnation: response_stagnation(session, response),
            token_velocity: token_velocity(session, request, config.token_velocity_threshold),
            error_cycling: error_cycling(session, response, content_similarity),
            context_growth: context_growth(session, request),
            cost_velocity: cost_velocity(
                cost_rate.unwrap_or_else(|| cost_rate_per_minute(session, request)),
                config.cost_velocity_threshold,
            ),
        }
    }
}

fn content_similarity(session: &SessionState, current_hash: u64) -> f32 {
    session
        .request_hashes
        .iter()
        .map(|previous_hash| simhash::similarity(current_hash, *previous_hash))
        .fold(0.0, f32::max)
}

fn tool_call_repetition(session: &SessionState, request: &RequestRecord) -> f32 {
    let Some(current) = request.tool_call_fingerprint else {
        return 0.0;
    };
    let consecutive_previous = session
        .tool_fingerprints
        .iter()
        .rev()
        .take_while(|fingerprint| **fingerprint == current)
        .count() as u32;
    repetition_score(consecutive_previous.saturating_add(1))
}

/// Monitors tool-compression discovery loops: repeated `get_tools_in_namespace`,
/// `get_tool_schema`, or `ns_*` drill-downs into a namespace/tool already revealed
/// earlier in the session. Unlike `tool_call_repetition` (consecutive only), this
/// counts re-observations anywhere in the session history so a slow, non-consecutive
/// discovery loop is still detected. Returns a value in `[0, 1]`.
fn discovery_repetition(session: &SessionState, request: &RequestRecord) -> f32 {
    if request.discovery_keys.is_empty() {
        return 0.0;
    }
    let mut re_observations = 0u32;
    let mut seen = std::collections::HashSet::new();
    for key in &request.discovery_keys {
        if seen.insert(key.clone()) && session.discovery_history.contains(key) {
            re_observations = re_observations.saturating_add(1);
        }
    }
    if re_observations == 0 {
        return 0.0;
    }
    // `discovery_repeat` holds the cumulative count prior to this request; add the
    // re-observations seen now. Map to the same 0/0.4/0.7/1.0 ladder as repetition.
    let total = session.discovery_repeat.saturating_add(re_observations);
    repetition_score(total.saturating_add(1))
}

fn response_stagnation(session: &SessionState, response: Option<&ResponseDescriptor>) -> f32 {
    let consecutive_previous = match response {
        Some(response) => session
            .response_descriptors
            .iter()
            .rev()
            .take_while(|previous| response_matches(previous, response))
            .count(),
        None => {
            let Some(response) = session.response_descriptors.back() else {
                return 0.0;
            };
            session
                .response_descriptors
                .iter()
                .rev()
                .skip(1)
                .take_while(|previous| response_matches(previous, response))
                .count()
        }
    };
    match consecutive_previous.saturating_add(1) {
        0..=2 => 0.0,
        3 => 0.6,
        4 => 0.8,
        _ => 1.0,
    }
}

fn response_matches(left: &ResponseDescriptor, right: &ResponseDescriptor) -> bool {
    if left.block_type_hash != right.block_type_hash {
        return false;
    }
    let maximum = left.token_count.max(right.token_count).max(1) as f32;
    let difference = left.token_count.abs_diff(right.token_count) as f32;
    difference / maximum < 0.05
}

fn token_velocity(session: &SessionState, request: &RequestRecord, threshold: f32) -> f32 {
    let window_start = request.timestamp.checked_sub(Duration::from_secs(60));
    let previous_tokens: u64 = session
        .timestamps
        .iter()
        .zip(session.recent_token_counts.iter())
        .filter(|(timestamp, _)| window_start.is_none_or(|start| **timestamp >= start))
        .map(|(_, tokens)| u64::from(*tokens))
        .sum();
    rate_score(
        previous_tokens.saturating_add(u64::from(request.token_count)) as f64,
        threshold as f64,
    )
}

fn error_cycling(
    session: &SessionState,
    response: Option<&ResponseDescriptor>,
    content_similarity: f32,
) -> f32 {
    let current_error = response.is_some_and(|response| response.is_error);
    let retry_after_error = response.is_none()
        && session
            .response_descriptors
            .back()
            .is_some_and(|response| response.is_error)
        && content_similarity > 0.8;
    let consecutive_error_retries = if retry_after_error {
        session
            .response_descriptors
            .iter()
            .rev()
            .take_while(|response| response.is_error)
            .count() as u32
    } else if current_error && content_similarity > 0.8 {
        session
            .response_descriptors
            .iter()
            .rev()
            .take_while(|response| response.is_error)
            .count() as u32
            + 1
    } else {
        0
    };
    let cycles = session.error_retry_cycles.max(consecutive_error_retries);
    match cycles {
        0 | 1 => 0.0,
        2 => 0.5,
        _ => 1.0,
    }
}

fn context_growth(session: &SessionState, request: &RequestRecord) -> f32 {
    let Some(previous_context) = session.context_token_counts.back() else {
        return 0.0;
    };
    let growth = request
        .context_token_count
        .saturating_sub(*previous_context);
    if growth == 0 {
        return 0.0;
    }
    let ratio = growth as f32 / request.new_information_tokens.max(1) as f32;
    (ratio / 10.0).clamp(0.0, 1.0)
}

fn cost_rate_per_minute(session: &SessionState, request: &RequestRecord) -> f64 {
    let window_start = request.timestamp.checked_sub(Duration::from_secs(60));
    session
        .timestamps
        .iter()
        .zip(session.recent_costs.iter())
        .filter(|(timestamp, _)| window_start.is_none_or(|start| **timestamp >= start))
        .map(|(_, cost)| *cost)
        .sum::<f64>()
        + request.cost
}

fn cost_velocity(actual_rate: f64, threshold: f64) -> f32 {
    rate_score(actual_rate, threshold)
}

fn rate_score(actual_rate: f64, threshold: f64) -> f32 {
    if !actual_rate.is_finite()
        || !threshold.is_finite()
        || threshold <= 0.0
        || actual_rate <= threshold
    {
        return 0.0;
    }
    ((actual_rate - threshold) / threshold).clamp(0.0, 1.0) as f32
}
