use crate::loop_detection::EnforcementLevel;
use chrono::{DateTime, Utc};
use serde::Serialize;
use std::{collections::VecDeque, sync::Mutex};
use tokio::sync::broadcast;

const EVENT_CHANNEL_CAPACITY: usize = 100;
const REPLAY_CAPACITY: usize = 100;

#[derive(Debug, Clone, Serialize, PartialEq)]
pub struct LoopDetectionEvent {
    pub session_id: String,
    pub confidence_score: f32,
    pub enforcement_level: &'static str,
    pub dominant_signal: &'static str,
    pub timestamp: DateTime<Utc>,
}

impl LoopDetectionEvent {
    pub fn new(
        session_id: String,
        confidence_score: f32,
        enforcement_level: EnforcementLevel,
        dominant_signal: &'static str,
    ) -> Self {
        Self {
            session_id,
            confidence_score,
            enforcement_level: level_label(enforcement_level),
            dominant_signal,
            timestamp: Utc::now(),
        }
    }
}

#[derive(Debug)]
pub struct LoopEventBus {
    sender: broadcast::Sender<LoopDetectionEvent>,
    replay: Mutex<VecDeque<LoopDetectionEvent>>,
}

impl LoopEventBus {
    pub fn new() -> Self {
        let (sender, _) = broadcast::channel(EVENT_CHANNEL_CAPACITY);
        Self {
            sender,
            replay: Mutex::new(VecDeque::with_capacity(REPLAY_CAPACITY)),
        }
    }

    pub fn publish(&self, event: LoopDetectionEvent) {
        if self.sender.receiver_count() == 0 {
            let mut replay = self
                .replay
                .lock()
                .expect("loop event replay mutex poisoned");
            if replay.len() == REPLAY_CAPACITY {
                replay.pop_front();
            }
            replay.push_back(event.clone());
        }
        let _ = self.sender.send(event);
    }

    pub fn subscribe(&self) -> LoopEventSubscription {
        let replay = {
            let mut replay = self
                .replay
                .lock()
                .expect("loop event replay mutex poisoned");
            replay.drain(..).collect()
        };
        LoopEventSubscription {
            replay,
            receiver: self.sender.subscribe(),
        }
    }

    pub fn buffered_len(&self) -> usize {
        self.replay
            .lock()
            .expect("loop event replay mutex poisoned")
            .len()
    }
}

impl Default for LoopEventBus {
    fn default() -> Self {
        Self::new()
    }
}

pub struct LoopEventSubscription {
    pub replay: Vec<LoopDetectionEvent>,
    pub receiver: broadcast::Receiver<LoopDetectionEvent>,
}

fn level_label(level: EnforcementLevel) -> &'static str {
    match level {
        EnforcementLevel::None => "none",
        EnforcementLevel::Warn => "warn",
        EnforcementLevel::Throttle => "throttle",
        EnforcementLevel::Inject => "inject",
        EnforcementLevel::HardStop => "hard_stop",
    }
}
