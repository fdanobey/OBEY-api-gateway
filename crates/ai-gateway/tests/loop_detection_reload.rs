use ai_gateway::{
    config::Config,
    gateway::{apply_runtime_config_update, GatewayServer},
    loop_detection::SessionState,
};

fn minimal_config() -> Config {
    serde_yaml::from_str("server:\n  host: 127.0.0.1\n  port: 8080\nproviders:\n  - name: p\n    type: openai\n    base_url: http://localhost\n    timeout_seconds: 30\nmodel_groups:\n  - name: g\n    models:\n      - provider: p\n        model: gpt-4\n").unwrap()
}

#[tokio::test]
async fn successful_reload_swaps_detector_snapshot_and_clears_sessions() {
    let server = GatewayServer::new(minimal_config(), None).await.unwrap();
    server
        .state
        .loop_detector
        .sessions
        .insert("active".into(), SessionState::new(Some("vk-id".into()), 5));
    let mut updated = minimal_config();
    updated.loop_detection.enabled = true;
    updated.loop_detection.throttle_delay_seconds = 7;

    apply_runtime_config_update(&server.state, updated).await;

    assert!(server
        .state
        .loop_detector
        .enabled
        .load(std::sync::atomic::Ordering::Relaxed));
    assert_eq!(
        server
            .state
            .loop_detector
            .detector_config
            .read()
            .await
            .throttle_delay_seconds,
        7
    );
    assert!(server.state.loop_detector.sessions.is_empty());
}

#[tokio::test]
async fn invalid_candidate_is_not_applied() {
    let server = GatewayServer::new(minimal_config(), None).await.unwrap();
    let old = server.state.config.read().await.clone();
    let mut invalid = old.clone();
    invalid.loop_detection.thresholds.warn_confidence = 0.95;
    assert!(invalid.validate().is_err());

    assert_eq!(
        server.state.config.read().await.loop_detection,
        old.loop_detection
    );
    assert!(!server
        .state
        .loop_detector
        .enabled
        .load(std::sync::atomic::Ordering::Relaxed));
}
