use ai_gateway::loop_detection::{
    events::{LoopDetectionEvent, LoopEventBus},
    EnforcementLevel,
};

#[tokio::test]
async fn live_subscriber_receives_nonblocking_event() {
    let bus = LoopEventBus::new();
    let mut subscription = bus.subscribe();
    bus.publish(LoopDetectionEvent::new(
        "session-live".into(),
        0.8,
        EnforcementLevel::Inject,
        "tool_call_repetition",
    ));
    let event = tokio::time::timeout(
        std::time::Duration::from_millis(500),
        subscription.receiver.recv(),
    )
    .await
    .unwrap()
    .unwrap();
    assert_eq!(event.session_id, "session-live");
    assert_eq!(event.enforcement_level, "inject");
}

#[test]
fn disconnected_replay_is_bounded_to_newest_hundred() {
    let bus = LoopEventBus::new();
    for index in 0..150 {
        bus.publish(LoopDetectionEvent::new(
            format!("session-{index}"),
            index as f32 / 150.0,
            EnforcementLevel::Warn,
            "content_similarity",
        ));
    }
    assert_eq!(bus.buffered_len(), 100);
    let subscription = bus.subscribe();
    assert_eq!(subscription.replay.len(), 100);
    assert_eq!(
        subscription.replay.first().unwrap().session_id,
        "session-50"
    );
    assert_eq!(
        subscription.replay.last().unwrap().session_id,
        "session-149"
    );
    assert_eq!(bus.buffered_len(), 0);
}
