use ai_gateway::{
    config::Config,
    gateway::{apply_runtime_config_update, GatewayServer},
    loop_detection::{
        middleware::{LoopDetectorLayer, LoopDetectorState},
        EnforcementLevel, LoopDetectionConfig, SessionState,
    },
    virtual_keys::models::{AuthenticatedKey, KeyStatus},
};
use axum::{
    body::{to_bytes, Body},
    http::{Request, Response, StatusCode},
    Extension,
};
use serde_json::Value;
use std::{
    convert::Infallible,
    sync::{Arc, Mutex},
    time::{Duration, Instant},
};
use tokio::sync::RwLock;
use tower::{service_fn, Layer, ServiceExt};

fn config(loop_config: LoopDetectionConfig) -> Config {
    let mut config: Config = serde_yaml::from_str("server:\n  host: 127.0.0.1\n  port: 8080\nproviders:\n  - name: p\n    type: openai\n    base_url: http://localhost\n    timeout_seconds: 30\nmodel_groups:\n  - name: g\n    models:\n      - provider: p\n        model: gpt-4\n").unwrap();
    config.loop_detection = loop_config;
    config
}

fn request(session: &str) -> Request<Body> {
    let body = serde_json::json!({"model":"gpt-4","messages":[{"role":"user","content":"repeat same operation"}],"stream":false}).to_string();
    Request::post("/v1/chat/completions")
        .header("content-type", "application/json")
        .header("content-length", body.len().to_string())
        .header("x-session-id", session)
        .body(Body::from(body))
        .unwrap()
}

fn immediate_config() -> LoopDetectionConfig {
    let mut config = LoopDetectionConfig::default();
    config.enabled = true;
    config.ema_alpha = 1.0;
    config.thresholds.warn_confidence = 0.01;
    config.thresholds.throttle_confidence = 0.02;
    config.thresholds.inject_confidence = 0.03;
    config.thresholds.hardstop_confidence = 0.04;
    config.consecutive_counts.warn = 1;
    config.consecutive_counts.throttle = 1;
    config.consecutive_counts.inject = 1;
    config.consecutive_counts.hardstop = 1;
    config.weights.content_similarity = 1.0;
    config.weights.tool_call_repetition = 0.0;
    config.weights.response_stagnation = 0.0;
    config.weights.token_velocity = 0.0;
    config.weights.error_cycling = 0.0;
    config.weights.context_growth = 0.0;
    config.weights.cost_velocity = 0.0;
    config
}

fn detector(loop_config: LoopDetectionConfig) -> (Arc<LoopDetectorState>, LoopDetectorLayer) {
    let config = config(loop_config.clone());
    let state = Arc::new(LoopDetectorState::new(
        Arc::new(RwLock::new(config)),
        loop_config,
    ));
    (state.clone(), LoopDetectorLayer::new(state))
}

#[tokio::test]
async fn hard_stop_short_circuits_inner_service() {
    let (state, layer) = detector(immediate_config());
    let calls = Arc::new(std::sync::atomic::AtomicUsize::new(0));
    let calls_inner = calls.clone();
    let service = layer.layer(service_fn(move |_request: Request<Body>| {
        let calls = calls_inner.clone();
        async move {
            calls.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
            Ok::<_, Infallible>(Response::new(Body::from("{}")))
        }
    }));
    for _ in 0..5 {
        service.clone().oneshot(request("hard-stop")).await.unwrap();
    }
    let response = service.oneshot(request("hard-stop")).await.unwrap();
    assert_eq!(response.status(), StatusCode::TOO_MANY_REQUESTS);
    assert_eq!(calls.load(std::sync::atomic::Ordering::Relaxed), 5);
    assert_eq!(
        state.sessions.get("hard-stop").unwrap().enforcement_level,
        EnforcementLevel::HardStop
    );
}

#[tokio::test]
async fn throttle_applies_minimum_delay_and_warning_header() {
    let mut loop_config = immediate_config();
    loop_config.throttle_delay_seconds = 1;
    let (_state, layer) = detector(loop_config);
    let service = layer.layer(service_fn(|_request: Request<Body>| async {
        Ok::<_, Infallible>(
            Response::builder()
                .header("content-type", "application/json")
                .body(Body::from("{}"))
                .unwrap(),
        )
    }));
    for _ in 0..3 {
        service.clone().oneshot(request("throttle")).await.unwrap();
    }
    let started = Instant::now();
    let response = service.oneshot(request("throttle")).await.unwrap();
    assert!(started.elapsed() >= Duration::from_secs(1));
    assert!(response.headers().contains_key("x-loop-warning"));
}

#[tokio::test]
async fn injection_mutation_is_observed_by_inner_service() {
    let (_state, layer) = detector(immediate_config());
    let observed = Arc::new(Mutex::new(Vec::<Value>::new()));
    let observed_inner = observed.clone();
    let service = layer.layer(service_fn(move |request: Request<Body>| {
        let observed = observed_inner.clone();
        async move {
            let body = to_bytes(request.into_body(), 1024 * 1024).await.unwrap();
            observed
                .lock()
                .unwrap()
                .push(serde_json::from_slice(&body).unwrap());
            Ok::<_, Infallible>(
                Response::builder()
                    .header("content-type", "application/json")
                    .body(Body::from("{}"))
                    .unwrap(),
            )
        }
    }));
    for _ in 0..5 {
        service.clone().oneshot(request("inject")).await.unwrap();
    }
    let observed = observed.lock().unwrap();
    let messages = &observed[4]["messages"];
    assert!(messages
        .as_array()
        .unwrap()
        .iter()
        .any(|message| message["role"] == "system"));
}

#[tokio::test]
async fn disabled_mode_creates_no_state_and_reload_clears_state() {
    let (state, layer) = detector(LoopDetectionConfig::default());
    let service = layer.layer(service_fn(|_request: Request<Body>| async {
        Ok::<_, Infallible>(Response::new(Body::empty()))
    }));
    service.oneshot(request("disabled")).await.unwrap();
    assert!(state.sessions.is_empty());

    let server = GatewayServer::new(config(LoopDetectionConfig::default()), None)
        .await
        .unwrap();
    server
        .state
        .loop_detector
        .sessions
        .insert("old".into(), SessionState::new(None, 5));
    let mut updated = server.state.config.read().await.clone();
    updated.loop_detection.enabled = true;
    apply_runtime_config_update(&server.state, updated).await;
    assert!(server.state.loop_detector.sessions.is_empty());
}

#[tokio::test]
async fn event_delivery_reaches_live_subscriber() {
    let (state, layer) = detector(immediate_config());
    let mut subscription = state.events.subscribe();
    let service = layer.layer(service_fn(|_request: Request<Body>| async {
        Ok::<_, Infallible>(
            Response::builder()
                .header("content-type", "application/json")
                .body(Body::from("{}"))
                .unwrap(),
        )
    }));
    service.oneshot(request("event")).await.unwrap();
    let event = tokio::time::timeout(Duration::from_millis(500), subscription.receiver.recv())
        .await
        .unwrap()
        .unwrap();
    assert_eq!(event.session_id, "event");
}

fn authenticated_key() -> AuthenticatedKey {
    AuthenticatedKey {
        id: "vk-ordering".into(),
        name: None,
        status: KeyStatus::Active,
        budget_limit_usd: None,
        token_budget: None,
        budget_window: None,
        current_spend_usd: 0.0,
        current_tokens_used: 0,
        window_start: None,
        requests_per_minute: None,
        tokens_per_minute: None,
        model_access: None,
        expires_at: None,
        loop_detection: None,
    }
}

#[tokio::test]
async fn actual_gateway_router_places_loop_after_auth_and_before_guardrail() {
    let mut cfg = config(immediate_config());
    cfg.guardrails = serde_yaml::from_str("providers:\n  - name: regex\n    type: regex\n    failure_policy: fail_close\n    patterns:\n      - name: injected\n        regex: 'Loop detected'\n        entity: LOOP\n        mode: deny\npipelines:\n  - name: loop-check\n    stages:\n      - name: check\n        provider: regex\n        phase: pre_call\n        action: block\nglobal_default_pipeline: loop-check\n").ok();
    let server = GatewayServer::new(cfg, None).await.unwrap();
    let app = server.build_router().layer(Extension(authenticated_key()));
    for _ in 0..4 {
        let _ = app.clone().oneshot(request("router-order")).await;
    }
    let response = app.oneshot(request("router-order")).await.unwrap();
    assert_eq!(
        response.status(),
        StatusCode::FORBIDDEN,
        "guardrail must observe loop-injected prompt"
    );
    assert_eq!(
        server
            .state
            .loop_detector
            .sessions
            .get("router-order")
            .unwrap()
            .vk_id
            .as_deref(),
        Some("vk-ordering")
    );
}
