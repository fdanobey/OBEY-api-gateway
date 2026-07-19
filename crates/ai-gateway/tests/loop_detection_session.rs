use ai_gateway::loop_detection::{SessionResolver, SessionState};
use axum::{body::Body, http::Request};
use dashmap::DashMap;
use proptest::prelude::*;
use std::time::Duration;

proptest! {
    #![proptest_config(ProptestConfig::with_cases(128))]

    // Feature: agent-loop-detection, Property 7: Session ID Extraction Round-Trip
    #[test]
    fn prop_valid_session_id_round_trip(bytes in prop::collection::vec(0x20u8..=0x7e, 1..=256)) {
        let session_id = String::from_utf8(bytes).unwrap();
        let request = Request::builder()
            .header("X-Session-ID", &session_id)
            .body(Body::empty())
            .unwrap();
        let sessions = DashMap::<String, SessionState>::new();

        prop_assert_eq!(
            SessionResolver::resolve(&request, &sessions, Some("vk-test"), Duration::from_secs(1_800)),
            Some(session_id)
        );
    }
}

#[test]
fn empty_session_id_falls_back_to_virtual_key() {
    let request = Request::builder()
        .header("X-Session-ID", "")
        .body(Body::empty())
        .unwrap();
    let sessions = DashMap::<String, SessionState>::new();

    let resolved = SessionResolver::resolve(
        &request,
        &sessions,
        Some("vk-test"),
        Duration::from_secs(1_800),
    )
    .unwrap();
    assert!(resolved.starts_with("vk:vk-test:"));
}

#[test]
fn oversized_session_id_falls_back_to_virtual_key() {
    let request = Request::builder()
        .header("X-Session-ID", "x".repeat(257))
        .body(Body::empty())
        .unwrap();
    let sessions = DashMap::<String, SessionState>::new();

    let resolved = SessionResolver::resolve(
        &request,
        &sessions,
        Some("vk-test"),
        Duration::from_secs(1_800),
    )
    .unwrap();
    assert!(resolved.starts_with("vk:vk-test:"));
}

#[test]
fn unauthenticated_request_without_valid_header_is_untracked() {
    let request = Request::new(Body::empty());
    let sessions = DashMap::<String, SessionState>::new();

    assert_eq!(
        SessionResolver::resolve(&request, &sessions, None, Duration::from_secs(1_800)),
        None
    );
}

#[test]
fn active_virtual_key_session_is_reused() {
    let request = Request::new(Body::empty());
    let sessions = DashMap::<String, SessionState>::new();
    sessions.insert(
        "existing-session".to_string(),
        SessionState::new(Some("vk-test".to_string()), 5),
    );

    assert_eq!(
        SessionResolver::resolve(
            &request,
            &sessions,
            Some("vk-test"),
            Duration::from_secs(1_800),
        ),
        Some("existing-session".to_string())
    );
}
