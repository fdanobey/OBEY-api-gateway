use ai_gateway::{
    config::Config,
    loop_detection::{
        middleware::{LoopDetectorLayer, LoopDetectorState},
        LoopDetectionConfig,
    },
};
use axum::{
    body::{to_bytes, Body},
    http::{Request, Response, StatusCode},
};
use std::{convert::Infallible, sync::Arc};
use tokio::sync::RwLock;
use tower::{service_fn, Layer, ServiceExt};

fn minimal_config(loop_detection: LoopDetectionConfig) -> Config {
    let yaml = "server:\n  host: 127.0.0.1\n  port: 8080\nproviders:\n  - name: p\n    type: openai\n    base_url: http://localhost\n    timeout_seconds: 30\nmodel_groups:\n  - name: g\n    models:\n      - provider: p\n        model: gpt-4\n";
    let mut config = serde_yaml::from_str::<Config>(yaml).unwrap();
    config.loop_detection = loop_detection;
    config
}

fn chat_request(session: &str) -> Request<Body> {
    let body = serde_json::json!({
        "model":"gpt-4",
        "messages":[{"role":"user","content":"repeat repeat repeat"}],
        "stream":false
    })
    .to_string();
    Request::post("/v1/chat/completions")
        .header("content-type", "application/json")
        .header("content-length", body.len().to_string())
        .header("x-session-id", session)
        .body(Body::from(body))
        .unwrap()
}

#[tokio::test]
async fn disabled_fast_path_does_not_create_session() {
    let config = minimal_config(LoopDetectionConfig::default());
    let detector_config = config.loop_detection.clone();
    let state = Arc::new(LoopDetectorState::new(
        Arc::new(RwLock::new(config)),
        detector_config,
    ));
    let service =
        LoopDetectorLayer::new(state.clone()).layer(service_fn(
            |request: Request<Body>| async move {
                Ok::<_, Infallible>(Response::new(request.into_body()))
            },
        ));
    let response = service.oneshot(chat_request("disabled")).await.unwrap();
    assert_eq!(response.status(), StatusCode::OK);
    assert!(state.sessions.is_empty());
}

#[tokio::test]
async fn non_json_and_unrelated_routes_pass_through() {
    let mut loop_config = LoopDetectionConfig::default();
    loop_config.enabled = true;
    let config = minimal_config(loop_config);
    let detector_config = config.loop_detection.clone();
    let state = Arc::new(LoopDetectorState::new(
        Arc::new(RwLock::new(config)),
        detector_config,
    ));
    let service =
        LoopDetectorLayer::new(state.clone()).layer(service_fn(
            |request: Request<Body>| async move {
                Ok::<_, Infallible>(Response::new(request.into_body()))
            },
        ));
    let request = Request::post("/v1/embeddings")
        .header("content-type", "application/json")
        .body(Body::from("not-json"))
        .unwrap();
    let response = service.oneshot(request).await.unwrap();
    assert_eq!(
        to_bytes(response.into_body(), 1024).await.unwrap(),
        "not-json"
    );
    assert!(state.sessions.is_empty());
}

#[tokio::test]
async fn repeated_requests_reach_warn_and_add_header() {
    let mut loop_config = LoopDetectionConfig::default();
    loop_config.enabled = true;
    loop_config.ema_alpha = 1.0;
    loop_config.thresholds.warn_confidence = 0.1;
    loop_config.consecutive_counts.warn = 1;
    let config = minimal_config(loop_config);
    let detector_config = config.loop_detection.clone();
    let state = Arc::new(LoopDetectorState::new(
        Arc::new(RwLock::new(config)),
        detector_config,
    ));
    let service = LoopDetectorLayer::new(state.clone()).layer(service_fn(
        |_request: Request<Body>| async move {
            Ok::<_, Infallible>(
                Response::builder()
                    .header("content-type", "application/json")
                    .body(Body::from("{}"))
                    .unwrap(),
            )
        },
    ));
    for _ in 0..2 {
        service.clone().oneshot(chat_request("warn")).await.unwrap();
    }
    let response = service.oneshot(chat_request("warn")).await.unwrap();
    assert!(response.headers().contains_key("x-loop-warning"));
}
