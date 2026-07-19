use ai_gateway::loop_detection::{
    enforcement::EnforcementEngine, EnforcementLevel, LoopDetectionConfig, SessionState,
};
use proptest::prelude::*;

fn immediate_config() -> LoopDetectionConfig {
    let mut config = LoopDetectionConfig::default();
    config.consecutive_counts.warn = 1;
    config.consecutive_counts.throttle = 1;
    config.consecutive_counts.inject = 1;
    config.consecutive_counts.hardstop = 1;
    config
}

proptest! {
    #![proptest_config(ProptestConfig::with_cases(128))]

    // Feature: agent-loop-detection, Property 4: Sequential Enforcement Escalation
    #[test]
    fn prop_enforcement_never_skips_levels(confidences in prop::collection::vec(0.0f32..=1.0, 1..100)) {
        let config = immediate_config();
        let mut session = SessionState::new(Some("vk".into()), 5);
        for confidence in confidences {
            let previous = session.enforcement_level;
            let decision = EnforcementEngine::evaluate(confidence, &mut session, &config);
            prop_assert!(decision.level <= previous.next());
            if decision.level > previous {
                prop_assert_eq!(decision.level, previous.next());
            }
        }
    }
}

#[test]
fn escalation_reaches_each_level_sequentially() {
    let config = immediate_config();
    let mut session = SessionState::new(Some("vk".into()), 5);
    let expected = [
        EnforcementLevel::Warn,
        EnforcementLevel::Throttle,
        EnforcementLevel::Inject,
        EnforcementLevel::HardStop,
    ];
    for expected_level in expected {
        let decision = EnforcementEngine::evaluate(1.0, &mut session, &config);
        assert!(decision.transitioned);
        assert_eq!(decision.level, expected_level);
    }
}

// Feature: agent-loop-detection, Property 12: Injection Idempotency Per Level
#[test]
fn injection_is_signaled_once_on_inject_transition() {
    let config = immediate_config();
    let mut session = SessionState::new(Some("vk".into()), 5);
    EnforcementEngine::evaluate(1.0, &mut session, &config);
    EnforcementEngine::evaluate(1.0, &mut session, &config);
    let transition = EnforcementEngine::evaluate(1.0, &mut session, &config);
    assert!(transition.should_inject);
    assert!(!session.injected_at_level);

    let stable = EnforcementEngine::evaluate(0.8, &mut session, &config);
    assert!(!stable.should_inject);
    assert!(!stable.transitioned);
}

// Feature: agent-loop-detection, Property 13: De-escalation After Recovery
#[test]
fn five_low_requests_deescalate_exactly_one_level() {
    let config = immediate_config();
    let mut session = SessionState::new(Some("vk".into()), 5);
    session.enforcement_level = EnforcementLevel::HardStop;

    for _ in 0..4 {
        let decision = EnforcementEngine::evaluate(0.0, &mut session, &config);
        assert_eq!(decision.level, EnforcementLevel::HardStop);
        assert!(!decision.transitioned);
    }
    let decision = EnforcementEngine::evaluate(0.0, &mut session, &config);
    assert_eq!(decision.level, EnforcementLevel::Inject);
    assert!(decision.transitioned);
    assert_eq!(session.consecutive_low, 0);
}

#[test]
fn escalation_uses_session_wide_consecutive_streak() {
    let mut config = LoopDetectionConfig::default();
    config.consecutive_counts.warn = 2;
    config.consecutive_counts.throttle = 3;
    config.consecutive_counts.inject = 4;
    config.consecutive_counts.hardstop = 5;
    let mut session = SessionState::new(Some("vk".into()), 5);

    assert_eq!(EnforcementEngine::evaluate(1.0, &mut session, &config).level, EnforcementLevel::None);
    assert_eq!(EnforcementEngine::evaluate(1.0, &mut session, &config).level, EnforcementLevel::Warn);
    assert_eq!(EnforcementEngine::evaluate(1.0, &mut session, &config).level, EnforcementLevel::Throttle);
    assert_eq!(EnforcementEngine::evaluate(1.0, &mut session, &config).level, EnforcementLevel::Inject);
    assert_eq!(EnforcementEngine::evaluate(1.0, &mut session, &config).level, EnforcementLevel::HardStop);
}

#[test]
fn default_consecutive_count_is_enforced() {
    let config = LoopDetectionConfig::default();
    let mut session = SessionState::new(Some("vk".into()), 5);
    for _ in 0..2 {
        assert_eq!(
            EnforcementEngine::evaluate(0.4, &mut session, &config).level,
            EnforcementLevel::None
        );
    }
    assert_eq!(
        EnforcementEngine::evaluate(0.4, &mut session, &config).level,
        EnforcementLevel::Warn
    );
}
