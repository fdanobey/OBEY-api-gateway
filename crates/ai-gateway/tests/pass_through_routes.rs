use ai_gateway::config::{Config, EnforcementMode};
use ai_gateway::gateway::GatewayServer;
use axum::body::Body;
use axum::http::{Request, StatusCode};
use serde_json::{json, Value};
use tower::ServiceExt;
use wiremock::matchers::{method, path};
use wiremock::{Mock, MockServer, ResponseTemplate};

async fn test_server() -> (GatewayServer, MockServer, tempfile::TempDir) {
    let mock = MockServer::start().await;
    for (route, body) in [
        (
            "/v1/embeddings",
            json!({"object":"list","data":[],"model":"upstream-model","usage":{"prompt_tokens":1,"total_tokens":1}}),
        ),
        ("/v1/images/generations", json!({"created":1,"data":[]})),
        ("/v1/audio/transcriptions", json!({"text":"transcribed"})),
        ("/v1/audio/translations", json!({"text":"translated"})),
    ] {
        Mock::given(method("POST"))
            .and(path(route))
            .respond_with(ResponseTemplate::new(200).set_body_json(body))
            .mount(&mock)
            .await;
    }

    let temp = tempfile::tempdir().unwrap();
    let mut config: Config = serde_yaml::from_str(&format!(
        "server:\n  host: 127.0.0.1\n  port: 0\nproviders:\n  - name: p\n    type: openai\n    base_url: {}\n    timeout_seconds: 30\nmodel_groups:\n  - name: g\n    models:\n      - provider: p\n        model: upstream-model\n",
        mock.uri()
    ))
    .unwrap();
    config.logging.database_path = temp.path().join("logs.db").to_string_lossy().into_owned();
    config.virtual_keys.database_path = temp.path().join("keys.db").to_string_lossy().into_owned();
    config.virtual_keys.enforcement = EnforcementMode::Disabled;
    let server = GatewayServer::new(config, None).await.unwrap();
    (server, mock, temp)
}

async fn json_post(app: axum::Router, uri: &str, body: Value) -> (StatusCode, Value) {
    let response = app
        .oneshot(
            Request::post(uri)
                .header("content-type", "application/json")
                .body(Body::from(body.to_string()))
                .unwrap(),
        )
        .await
        .unwrap();
    let status = response.status();
    let bytes = axum::body::to_bytes(response.into_body(), 1024 * 1024)
        .await
        .unwrap();
    (status, serde_json::from_slice(&bytes).unwrap())
}

async fn audio_post(app: axum::Router, uri: &str) -> (StatusCode, Value) {
    let boundary = "pass-through-test";
    let body = format!(
        "--{boundary}\r\nContent-Disposition: form-data; name=\"model\"\r\n\r\ng\r\n--{boundary}\r\nContent-Disposition: form-data; name=\"file\"; filename=\"a.wav\"\r\nContent-Type: application/octet-stream\r\n\r\naudio\r\n--{boundary}--\r\n"
    );
    let response = app
        .oneshot(
            Request::post(uri)
                .header(
                    "content-type",
                    format!("multipart/form-data; boundary={boundary}"),
                )
                .body(Body::from(body))
                .unwrap(),
        )
        .await
        .unwrap();
    let status = response.status();
    let bytes = axum::body::to_bytes(response.into_body(), 1024 * 1024)
        .await
        .unwrap();
    (status, serde_json::from_slice(&bytes).unwrap())
}

#[tokio::test]
async fn all_provider_pass_through_routes_dispatch_through_axum() {
    let (server, _mock, _temp) = test_server().await;
    let app = server.build_router();

    let (status, embeddings) = json_post(
        app.clone(),
        "/v1/embeddings",
        json!({"model":"g","input":"hello"}),
    )
    .await;
    assert_eq!(status, StatusCode::OK);
    assert_eq!(embeddings["object"], "list");

    let (status, images) = json_post(
        app.clone(),
        "/v1/images/generations",
        json!({"model":"g","prompt":"draw"}),
    )
    .await;
    assert_eq!(status, StatusCode::OK);
    assert_eq!(images["created"], 1);

    let (status, transcription) = audio_post(app.clone(), "/v1/audio/transcriptions").await;
    assert_eq!(status, StatusCode::OK);
    assert_eq!(transcription["text"], "transcribed");

    let (status, translation) = audio_post(app, "/v1/audio/translations").await;
    assert_eq!(status, StatusCode::OK);
    assert_eq!(translation["text"], "translated");
}
