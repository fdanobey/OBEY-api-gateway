//! Routes memory consolidation batches through the gateway router so they
//! behave like normal requests: model-group routing or a pinned
//! `provider:model`, failover, rate limits, circuit breakers, the attempt
//! ledger, request log rows and an in-flight dashboard entry.
//!
//! Auto-extract requests keep using the direct [`GatewayExtractionAdapter`].

use std::sync::{Arc, OnceLock, Weak};

use async_trait::async_trait;
use tokio::sync::RwLock;

use super::AppState;
use crate::active_requests::{ActiveRequestRegistry, RequestOrigin};
use crate::config::Config;
use crate::logger::RequestLogger;
use crate::memory::{
    ConsolidationRoute, GatewayExtractionAdapter, MemoryExtractionProvider,
    MemoryExtractionProviderError, MemoryExtractionProviderRequest, StructuredMemoryCandidate,
};
use crate::metrics::Metrics;
use crate::models::openai::OpenAIRequest;
use crate::router::router::Router as RequestRouter;
use crate::runtime_limits::RuntimeLimits;

/// Gateway handles an internal request needs to run like a client request.
///
/// The router is held weakly: the router owns the memory system, which owns
/// the adapter holding this context.
pub struct InternalDispatchContext {
    pub(crate) router: Weak<RequestRouter>,
    pub(crate) logger: Arc<RequestLogger>,
    pub(crate) metrics: Arc<Metrics>,
    pub(crate) active_requests: Arc<ActiveRequestRegistry>,
    pub(crate) runtime_limits: Arc<RuntimeLimits>,
}

impl InternalDispatchContext {
    pub fn from_state(state: &AppState) -> Self {
        Self {
            router: Arc::downgrade(&state.router),
            logger: state.logger.clone(),
            metrics: state.metrics.clone(),
            active_requests: state.active_requests.clone(),
            runtime_limits: state.runtime_limits.clone(),
        }
    }
}

/// Content-free failure of an internal dispatch.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct InternalDispatchFailure {
    pub error_class: String,
    pub message: String,
    pub trace_id: Option<String>,
}

/// Memory extraction provider that sends consolidation batches through the
/// router and delegates every other request to the direct adapter.
pub struct RoutedMemoryAdapter {
    direct: GatewayExtractionAdapter,
    dispatch: Arc<OnceLock<InternalDispatchContext>>,
}

impl RoutedMemoryAdapter {
    pub fn new(config: Arc<RwLock<Config>>, dispatch: Arc<OnceLock<InternalDispatchContext>>) -> Self {
        Self {
            direct: GatewayExtractionAdapter::new(config),
            dispatch,
        }
    }
}

#[async_trait]
impl MemoryExtractionProvider for RoutedMemoryAdapter {
    async fn extract(
        &self,
        request: MemoryExtractionProviderRequest,
    ) -> Result<Vec<StructuredMemoryCandidate>, MemoryExtractionProviderError> {
        if !GatewayExtractionAdapter::is_consolidation_request(&request.messages) {
            return self.direct.extract(request).await;
        }
        let Some(context) = self.dispatch.get() else {
            return Err(MemoryExtractionProviderError::new("gateway dispatch unavailable")
                .with_error_class("gateway_unavailable"));
        };
        // Client-side `stream: false` is intended: buffered dispatch streams
        // upstream on its own (ADR 0001).
        let body = GatewayExtractionAdapter::build_request_body(&request.model, &request.messages);
        let chat_request = serde_json::from_value::<OpenAIRequest>(body).map_err(|_| {
            MemoryExtractionProviderError::new("failed to build consolidation request")
                .with_error_class("other")
        })?;
        let route = if request.provider.is_empty() {
            ConsolidationRoute::ModelGroup(request.model)
        } else {
            ConsolidationRoute::Pinned {
                provider: request.provider,
                model: request.model,
            }
        };
        match super::handlers::dispatch_internal_chat(
            context,
            chat_request,
            route,
            RequestOrigin::MemoryConsolidation,
        )
        .await
        {
            Ok(response) => GatewayExtractionAdapter::parse_provider_payload(
                &serde_json::to_value(&response).unwrap_or_default(),
                true,
            ),
            Err(failure) => {
                let error = MemoryExtractionProviderError::new(failure.message)
                    .with_error_class(failure.error_class);
                Err(match failure.trace_id {
                    Some(trace_id) => error.with_trace_id(trace_id),
                    None => error,
                })
            }
        }
    }

    fn enforces_own_deadline(&self) -> bool {
        true
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::gateway::GatewayServer;
    use crate::logger::LogEntry;
    use crate::memory::{
        ConsolidationJobState, ExtractionMessage, ExtractionRole, MemoryConfig, MemorySystem,
        MemoryType, NewMemoryEntry, MEMORY_EXTRACTION_INTERNAL_TAG,
    };
    use axum::body::Body;
    use axum::http::{Request, StatusCode};
    use serde_json::{json, Value};
    use std::time::Duration;
    use tower::ServiceExt;
    use uuid::Uuid;
    use wiremock::matchers::{body_partial_json, method, path};
    use wiremock::{Mock, MockServer, ResponseTemplate};

    const OUTCOME_PATH: &str = "/v1/chat/completions";
    const ATTEMPT_PATH: &str = "/v1/chat/completions#attempt";
    const GROUP: &str = "consolidation-group";

    struct Harness {
        server: GatewayServer,
        memory: Arc<MemorySystem>,
        _dir: tempfile::TempDir,
    }

    enum Target {
        Group,
        Pinned(&'static str, &'static str),
    }

    async fn harness(
        primary: &MockServer,
        backup: &MockServer,
        primary_rate_limit: u32,
        target: Target,
    ) -> Harness {
        let dir = tempfile::tempdir().unwrap();
        let mut config: Config = serde_json::from_value(json!({
            "server": { "host": "127.0.0.1", "port": 0 },
            "providers": [
                {
                    "name": "primary",
                    "type": "openai",
                    "base_url": primary.uri(),
                    "rate_limit_per_minute": primary_rate_limit
                },
                { "name": "backup", "type": "openai", "base_url": backup.uri() }
            ],
            "model_groups": [{
                "name": GROUP,
                "models": [
                    { "provider": "primary", "model": "m1", "priority": 1 },
                    { "provider": "backup", "model": "m2", "priority": 2 }
                ]
            }]
        }))
        .expect("test configuration should deserialize");
        config.logging.database_path = dir.path().join("logs.db").to_string_lossy().into_owned();
        config.virtual_keys.database_path =
            dir.path().join("keys.db").to_string_lossy().into_owned();
        config.retry.max_retries_per_provider = 0;
        config.retry.backoff_sequence_seconds = vec![0];
        config.retry.jitter_enabled = false;
        let mut memory = MemoryConfig {
            enabled: true,
            database_path: dir.path().join("memory.db").to_string_lossy().into_owned(),
            consolidation_enabled: false,
            ..Default::default()
        };
        match target {
            Target::Group => memory.consolidation_model_group = GROUP.to_owned(),
            Target::Pinned(provider, model) => {
                memory.consolidation_provider = provider.to_owned();
                memory.consolidation_model = model.to_owned();
            }
        }
        config.memory = Some(memory);
        let server = GatewayServer::new(config, None).await.unwrap();
        let memory = server
            .state
            .memory_system
            .read()
            .await
            .clone()
            .expect("memory enabled");
        Harness {
            server,
            memory,
            _dir: dir,
        }
    }

    fn seed_near_duplicates(memory: &MemorySystem) -> [Uuid; 2] {
        ["The deploy target is fly.io eu-west", "The deploy target is fly.io eu-west region"].map(
            |content| {
                memory
                    .store
                    .store_entry(
                        NewMemoryEntry {
                            namespace: "user::test".to_owned(),
                            content: content.to_owned(),
                            memory_type: MemoryType::Fact,
                            source_request_id: None,
                        },
                        None,
                    )
                    .unwrap()
                    .id
            },
        )
    }

    /// Upstream SSE whose assembled content keeps both entries.
    fn keep_decisions_sse(ids: &[Uuid]) -> String {
        let decisions = Value::Array(
            ids.iter()
                .map(|id| json!({ "action": "keep", "id": id.to_string() }))
                .collect(),
        )
        .to_string();
        let chunks = [
            json!({
                "id": "c1", "object": "chat.completion.chunk", "created": 1, "model": "m",
                "choices": [{ "index": 0, "delta": { "role": "assistant", "content": decisions } }]
            }),
            json!({
                "id": "c1", "object": "chat.completion.chunk", "created": 1, "model": "m",
                "choices": [{ "index": 0, "delta": {}, "finish_reason": "stop" }]
            }),
            json!({
                "id": "c1", "object": "chat.completion.chunk", "created": 1, "model": "m",
                "choices": [],
                "usage": { "prompt_tokens": 50, "completion_tokens": 10, "total_tokens": 60 }
            }),
        ];
        let mut body = String::new();
        for chunk in chunks {
            body.push_str(&format!("data: {chunk}\n\n"));
        }
        body.push_str("data: [DONE]\n\n");
        body
    }

    fn sse_success(ids: &[Uuid]) -> ResponseTemplate {
        ResponseTemplate::new(200).set_body_raw(keep_decisions_sse(ids), "text/event-stream")
    }

    fn streamed_chat() -> wiremock::MockBuilder {
        Mock::given(method("POST"))
            .and(path("/v1/chat/completions"))
            .and(body_partial_json(
                json!({ "stream": true, "stream_options": { "include_usage": true } }),
            ))
    }

    async fn mount_failure(server: &MockServer) {
        Mock::given(method("POST"))
            .and(path("/v1/chat/completions"))
            .respond_with(ResponseTemplate::new(500).set_body_string("boom"))
            .mount(server)
            .await;
    }

    async fn wait_for_job(memory: &MemorySystem) -> crate::memory::ConsolidationJobStatus {
        let deadline = tokio::time::Instant::now() + Duration::from_secs(10);
        loop {
            let status = memory.consolidation_job_status().await.expect("job status");
            if status.state != ConsolidationJobState::Running
                || tokio::time::Instant::now() >= deadline
            {
                return status;
            }
            tokio::time::sleep(Duration::from_millis(25)).await;
        }
    }

    async fn run_job(memory: &MemorySystem) -> crate::memory::ConsolidationJobStatus {
        memory.start_consolidation_job().await.expect("job starts");
        wait_for_job(memory).await
    }

    async fn get_json(server: &GatewayServer, uri: &str) -> Value {
        let response = server
            .build_router()
            .oneshot(Request::get(uri).body(Body::empty()).unwrap())
            .await
            .unwrap();
        assert_eq!(response.status(), StatusCode::OK, "GET {uri}");
        let body = axum::body::to_bytes(response.into_body(), 4 * 1024 * 1024)
            .await
            .unwrap();
        serde_json::from_slice(&body).unwrap()
    }

    /// Poll the request log until `done` holds (or 5 s pass).
    async fn poll_logs(server: &GatewayServer, done: impl Fn(&[LogEntry]) -> bool) -> Vec<LogEntry> {
        let deadline = tokio::time::Instant::now() + Duration::from_secs(5);
        loop {
            let rows: Vec<LogEntry> =
                serde_json::from_value(get_json(server, "/dashboard/logs").await).unwrap();
            if done(&rows) || tokio::time::Instant::now() >= deadline {
                return rows;
            }
            tokio::time::sleep(Duration::from_millis(50)).await;
        }
    }

    fn outcome_rows(rows: &[LogEntry]) -> Vec<&LogEntry> {
        rows.iter().filter(|row| row.path == OUTCOME_PATH).collect()
    }

    async fn chat_requests(server: &MockServer) -> usize {
        server
            .received_requests()
            .await
            .unwrap_or_default()
            .iter()
            .filter(|request| request.url.path() == "/v1/chat/completions")
            .count()
    }

    #[tokio::test]
    async fn consolidation_batch_streams_upstream_and_takes_one_token_per_request() {
        let primary = MockServer::start().await;
        let backup = MockServer::start().await;
        let h = harness(&primary, &backup, 6, Target::Group).await;
        let ids = seed_near_duplicates(&h.memory);
        streamed_chat()
            .respond_with(sse_success(&ids))
            .expect(1)
            .mount(&primary)
            .await;

        let limiter = h.server.state.router.get_rate_limiter("primary").await;
        let before = limiter.get_tokens().await;
        let status = run_job(&h.memory).await;
        let after = limiter.get_tokens().await;

        assert_eq!(status.state, ConsolidationJobState::Completed);
        assert_eq!(status.counts.expect("counts").kept, 2);
        let upstream = chat_requests(&primary).await;
        assert_eq!(upstream, 1);
        // 6/min refills 0.1 token/s: negligible over a sub-second pass.
        assert!(
            ((before - after) - upstream as f64).abs() < 0.5,
            "tokens before {before}, after {after}, upstream {upstream}"
        );

        let rows = poll_logs(&h.server, |rows| !outcome_rows(rows).is_empty()).await;
        let outcomes = outcome_rows(&rows);
        assert_eq!(outcomes.len(), 1, "{rows:?}");
        assert_eq!(outcomes[0].status_code, 200);
        assert_eq!(outcomes[0].provider, "primary");
        assert_eq!(
            outcomes[0].client_protocol.as_deref(),
            Some("memory_consolidation")
        );
    }

    #[tokio::test]
    async fn consolidation_batch_fails_over_and_logs_attempt_rows() {
        let primary = MockServer::start().await;
        let backup = MockServer::start().await;
        let h = harness(&primary, &backup, 0, Target::Group).await;
        let ids = seed_near_duplicates(&h.memory);
        mount_failure(&primary).await;
        streamed_chat()
            .respond_with(sse_success(&ids))
            .expect(1)
            .mount(&backup)
            .await;

        let status = run_job(&h.memory).await;

        assert_eq!(status.state, ConsolidationJobState::Completed);
        assert_eq!(chat_requests(&backup).await, 1);
        let rows = poll_logs(&h.server, |rows| {
            !outcome_rows(rows).is_empty() && rows.iter().any(|row| row.path == ATTEMPT_PATH)
        })
        .await;
        let attempt = rows
            .iter()
            .find(|row| row.path == ATTEMPT_PATH && row.provider == "primary")
            .unwrap_or_else(|| panic!("primary attempt row missing: {rows:?}"));
        assert_eq!(attempt.error_class.as_deref(), Some("upstream_http_500"));
        let outcomes = outcome_rows(&rows);
        assert_eq!(outcomes.len(), 1, "{rows:?}");
        assert_eq!(outcomes[0].provider, "backup");
        assert_eq!(outcomes[0].trace_id, attempt.trace_id);
    }

    #[tokio::test]
    async fn consolidation_batch_is_visible_in_flight() {
        let primary = MockServer::start().await;
        let backup = MockServer::start().await;
        let h = harness(&primary, &backup, 0, Target::Group).await;
        let ids = seed_near_duplicates(&h.memory);
        streamed_chat()
            .respond_with(sse_success(&ids).set_delay(Duration::from_millis(1500)))
            .mount(&primary)
            .await;

        h.memory.start_consolidation_job().await.expect("job starts");
        let deadline = tokio::time::Instant::now() + Duration::from_secs(5);
        let entry = loop {
            let found = h.server.state.active_requests.snapshot().into_iter().find(|info| {
                info.origin == Some(RequestOrigin::MemoryConsolidation)
                    && info.provider.as_deref() == Some("primary")
            });
            if let Some(info) = found {
                break info;
            }
            assert!(
                tokio::time::Instant::now() < deadline,
                "consolidation request never appeared in flight"
            );
            tokio::time::sleep(Duration::from_millis(20)).await;
        };
        assert_eq!(entry.phase, crate::active_requests::ActivePhase::Primary);
        assert_eq!(entry.model_group.as_deref(), Some(GROUP));

        let metrics = get_json(&h.server, "/dashboard/metrics").await;
        assert_eq!(
            metrics["active_requests_list"][0]["origin"],
            json!("memory_consolidation"),
            "{metrics}"
        );

        let status = wait_for_job(&h.memory).await;
        assert_eq!(status.state, ConsolidationJobState::Completed);
        assert!(h.server.state.active_requests.snapshot().is_empty());
    }

    #[tokio::test]
    async fn consolidation_failure_reports_class_and_trace() {
        let primary = MockServer::start().await;
        let backup = MockServer::start().await;
        let h = harness(&primary, &backup, 0, Target::Group).await;
        seed_near_duplicates(&h.memory);
        mount_failure(&primary).await;
        mount_failure(&backup).await;

        let status = run_job(&h.memory).await;

        assert_eq!(status.state, ConsolidationJobState::Failed);
        assert!(
            status
                .provider_error_classes
                .iter()
                .any(|class| class.class == "upstream_http_500"),
            "{:?}",
            status.provider_error_classes
        );
        let trace_id = status.last_error_trace_id.clone().expect("trace id");
        let rows = poll_logs(&h.server, |rows| !outcome_rows(rows).is_empty()).await;
        let outcome = outcome_rows(&rows)
            .into_iter()
            .find(|row| row.trace_id == trace_id)
            .unwrap_or_else(|| panic!("no outcome row for {trace_id}: {rows:?}"));
        assert_eq!(outcome.status_code, 502);
        assert_eq!(outcome.client_protocol.as_deref(), Some("memory_consolidation"));
    }

    #[tokio::test]
    async fn legacy_pair_routes_pinned_through_router() {
        let primary = MockServer::start().await;
        let backup = MockServer::start().await;
        let h = harness(&primary, &backup, 0, Target::Pinned("backup", "m2")).await;
        let ids = seed_near_duplicates(&h.memory);
        Mock::given(method("POST"))
            .and(path("/v1/chat/completions"))
            .respond_with(sse_success(&ids))
            .expect(0)
            .mount(&primary)
            .await;
        streamed_chat()
            .respond_with(sse_success(&ids))
            .expect(1)
            .mount(&backup)
            .await;

        let status = run_job(&h.memory).await;

        assert_eq!(status.state, ConsolidationJobState::Completed);
        assert_eq!(status.counts.expect("counts").kept, 2);
        let rows = poll_logs(&h.server, |rows| !outcome_rows(rows).is_empty()).await;
        let outcomes = outcome_rows(&rows);
        assert_eq!(outcomes.len(), 1, "{rows:?}");
        assert_eq!(outcomes[0].provider, "backup");
        assert_eq!(
            outcomes[0].client_protocol.as_deref(),
            Some("memory_consolidation")
        );
    }

    #[tokio::test]
    async fn auto_extract_requests_bypass_the_router() {
        let primary = MockServer::start().await;
        let backup = MockServer::start().await;
        let h = harness(&primary, &backup, 0, Target::Group).await;
        Mock::given(method("POST"))
            .and(path("/v1/chat/completions"))
            .and(body_partial_json(json!({ "stream": false })))
            .respond_with(ResponseTemplate::new(200).set_body_json(json!({
                "id": "chatcmpl-1",
                "object": "chat.completion",
                "created": 1,
                "model": "m1",
                "choices": [{
                    "index": 0,
                    "message": {
                        "role": "assistant",
                        "content": "[{\"content\":\"uses snake_case\",\"memory_type\":\"preference\"}]"
                    },
                    "finish_reason": "stop"
                }]
            })))
            .expect(1)
            .mount(&primary)
            .await;
        let adapter = RoutedMemoryAdapter::new(
            h.server.state.config.clone(),
            Arc::new(OnceLock::from(InternalDispatchContext::from_state(&h.server.state))),
        );

        let candidates = adapter
            .extract(MemoryExtractionProviderRequest {
                provider: "primary".to_owned(),
                model: "m1".to_owned(),
                messages: vec![ExtractionMessage::caller(
                    ExtractionRole::User,
                    "I always use snake_case",
                )],
                internal_tag: MEMORY_EXTRACTION_INTERNAL_TAG,
                bypass_memory: true,
            })
            .await
            .expect("direct extraction succeeds");

        assert_eq!(candidates.len(), 1);
        assert_eq!(candidates[0].content, "uses snake_case");
        assert!(h.server.state.active_requests.snapshot().is_empty());
        tokio::time::sleep(Duration::from_millis(200)).await;
        let rows: Vec<LogEntry> =
            serde_json::from_value(get_json(&h.server, "/dashboard/logs").await).unwrap();
        assert!(rows.is_empty(), "{rows:?}");
    }
}
