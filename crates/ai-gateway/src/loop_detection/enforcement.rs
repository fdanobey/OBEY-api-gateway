use crate::loop_detection::{
    config::LoopDetectionConfig,
    session::{EnforcementLevel, EscalationEvent, SessionState},
};
use chrono::Utc;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct EnforcementDecision {
    pub level: EnforcementLevel,
    pub transitioned: bool,
    pub dominant_signal: &'static str,
    pub should_warn: bool,
    pub should_throttle: bool,
    pub should_inject: bool,
    pub should_hard_stop: bool,
}

pub struct EnforcementEngine;

impl EnforcementEngine {
    pub fn evaluate(
        confidence: f32,
        session: &mut SessionState,
        config: &LoopDetectionConfig,
    ) -> EnforcementDecision {
        let confidence = if confidence.is_nan() {
            0.0
        } else {
            confidence.clamp(0.0, 1.0)
        };
        let previous_level = session.enforcement_level;
        let mut transitioned = false;

        if confidence < config.thresholds.warn_confidence {
            session.consecutive_high = 0;
            session.consecutive_low = session.consecutive_low.saturating_add(1);
            if session.consecutive_low >= 5
                && matches!(
                    session.enforcement_level,
                    EnforcementLevel::Inject | EnforcementLevel::HardStop
                )
            {
                session.enforcement_level = session.enforcement_level.previous();
                session.consecutive_low = 0;
                session.injected_at_level = false;
                transitioned = true;
            }
        } else {
            session.consecutive_low = 0;
            session.consecutive_high = session.consecutive_high.saturating_add(1);
            let next_level = session.enforcement_level.next();
            if next_level != session.enforcement_level
                && confidence >= threshold_for(next_level, config)
                && session.consecutive_high >= consecutive_count_for(next_level, config)
            {
                session.enforcement_level = next_level;
                session.injected_at_level = false;
                transitioned = true;
            }
        }

        session.smoothed_confidence = confidence;
        session.peak_confidence = session.peak_confidence.max(confidence);
        if transitioned {
            session.escalation_history.push(EscalationEvent {
                timestamp: Utc::now(),
                from_level: previous_level,
                to_level: session.enforcement_level,
                confidence,
            });
            tracing::info!(
                previous_level = ?previous_level,
                new_level = ?session.enforcement_level,
                confidence,
                consecutive_count = session.consecutive_high,
                dominant_signal = session.dominant_signal,
                "Loop detection enforcement level transitioned"
            );
        }

        let level = session.enforcement_level;
        EnforcementDecision {
            level,
            transitioned,
            dominant_signal: session.dominant_signal,
            should_warn: level >= EnforcementLevel::Warn,
            should_throttle: level >= EnforcementLevel::Throttle,
            should_inject: transitioned && level == EnforcementLevel::Inject,
            should_hard_stop: level == EnforcementLevel::HardStop,
        }
    }
}

fn threshold_for(level: EnforcementLevel, config: &LoopDetectionConfig) -> f32 {
    match level {
        EnforcementLevel::None => 0.0,
        EnforcementLevel::Warn => config.thresholds.warn_confidence,
        EnforcementLevel::Throttle => config.thresholds.throttle_confidence,
        EnforcementLevel::Inject => config.thresholds.inject_confidence,
        EnforcementLevel::HardStop => config.thresholds.hardstop_confidence,
    }
}

fn consecutive_count_for(level: EnforcementLevel, config: &LoopDetectionConfig) -> u32 {
    match level {
        EnforcementLevel::None => 0,
        EnforcementLevel::Warn => config.consecutive_counts.warn,
        EnforcementLevel::Throttle => config.consecutive_counts.throttle,
        EnforcementLevel::Inject => config.consecutive_counts.inject,
        EnforcementLevel::HardStop => config.consecutive_counts.hardstop,
    }
}
