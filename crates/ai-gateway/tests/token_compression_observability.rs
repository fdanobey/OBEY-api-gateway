//! Task 12.5 integration coverage for token-compression observability.

use ai_gateway::compression::{
    config::{CompressionConfig, EffectiveCompressionConfig},
    pipeline::{CompressionPipeline, CompressionRequestMetadata},
    stats::CompressionStats,
    CompressiblePayload, CompressionContext, CompressionLevel,
};
use ai_gateway::config::{Config, PrometheusConfig};
use ai_gateway::dashboard::CompressionEventHub;
use ai_gateway::gateway::GatewayServer;
use ai_gateway::logger::{CompressionLogMetadata, LogEntry};
use ai_gateway::models::openai::OpenAIRequest;
use axum::body::Body;
use axum::http::{Request, StatusCode};
use serde_json::{json, Value};
use tempfile::TempDir;
use tower::ServiceExt;

fn test_config(temp_dir: &TempDir) -> Config {
    let mut config: Config = serde_json::from_value(json!({
        "server": {
            "host": "127.0.0.1",
            "port": 0
        },
        "providers": [{
            "name": "observability-provider",
            "type": "openai",
            "base_url": "http://127.0.0.1:9"
        }],
        "model_groups": [{
            "name": "observability-group",
            "models": [{
                "provider": "observability-provider",
                "model": "gpt-4"
            }]
        }],
        "logging": {
            "database_path": temp_dir.path().join("logs.db")
        },
        "virtual_keys": {
            "database_path": temp_dir.path().join("keys.db")
        }
    }))
    .expect("test configuration should deserialize");
    config.prometheus = Some(PrometheusConfig {
        enabled: true,
        path: "/metrics".to_owned(),
    });
    config
}

fn effective(level: CompressionLevel) -> EffectiveCompressionConfig {
    EffectiveCompressionConfig {
        enabled: true,
        level,
        auto_threshold_tokens: 0,
        caveman_output: false,
    }
}

fn payload(model: &str, content: &str) -> CompressiblePayload {
    let request: OpenAIRequest = serde_json::from_value(json!({
        "model": model,
        "messages": [{"role": "user", "content": content}]
    }))
    .expect("compression request should deserialize");
    request.into()
}

async fn pipeline_stats(
    pipeline: &CompressionPipeline,
    level: CompressionLevel,
    request_id: &str,
    provider: &str,
) -> CompressionStats {
    let model = "gpt-4";
    let result = pipeline
        .compress_explicit(
            payload(
                model,
                "This is deliberately verbose input that should be processed by the configured token compression pipeline. This is deliberately verbose input that should be processed by the configured token compression pipeline.",
            ),
            CompressionContext::new(model, provider),
            effective(level),
            CompressionRequestMetadata {
                request_id: request_id.to_owned(),
                ..Default::default()
            },
        )
        .await;
    CompressionStats::from_pipeline_result(&result, false, provider, model)
}

async fn response_body(app: axum::Router, uri: &str) -> (StatusCode, Vec<u8>) {
    let response = app
        .oneshot(Request::get(uri).body(Body::empty()).unwrap())
        .await
        .unwrap();
    let status = response.status();
    let body = axum::body::to_bytes(response.into_body(), 1024 * 1024)
        .await
        .unwrap();
    (status, body.to_vec())
}

fn log_entry(trace_id: &str, compression: Option<CompressionLogMetadata>) -> LogEntry {
    LogEntry {
        trace_id: trace_id.to_owned(),
        timestamp: chrono::Utc::now(),
        method: "POST".to_owned(),
        path: "/v1/chat/completions".to_owned(),
        model: "gpt-4".to_owned(),
        provider: "observability-provider".to_owned(),
        status_code: 200,
        duration_ms: 10,
        cost: 0.0,
        request_body: None,
        response_body: None,
        requested_model: Some("gpt-4".to_owned()),
        responded_model: Some("gpt-4".to_owned()),
        compression,
    }
}

#[tokio::test]
async fn compression_event_hub_replays_and_streams_pipeline_stats_with_dashboard_hooks() {
    let pipeline = CompressionPipeline::from_config(CompressionConfig::default());
    let base = pipeline_stats(
        &pipeline,
        CompressionLevel::Standard,
        "compression-event-0",
        "observability-provider",
    )
    .await;
    let hub = CompressionEventHub::new();

    for index in 0..105 {
        let mut stats = base.clone();
        stats.request_id = format!("compression-event-{index}");
        hub.publish(stats);
    }

    let mut subscription = hub.subscribe();
    assert_eq!(subscription.replay.len(), 100);
    assert_eq!(subscription.replay[0].request_id, "compression-event-5");
    assert_eq!(
        subscription.replay.last().unwrap().request_id,
        "compression-event-104"
    );
    assert!(subscription
        .replay
        .iter()
        .all(|stats| stats.level == CompressionLevel::Standard));

    let mut live = base;
    live.request_id = "compression-event-live".to_owned();
    hub.publish(live);
    let received = tokio::time::timeout(
        std::time::Duration::from_secs(1),
        subscription.receiver.recv(),
    )
    .await
    .expect("live compression event should arrive")
    .expect("compression event channel should remain open");
    assert_eq!(received.request_id, "compression-event-live");
    assert_eq!(received.provider, "observability-provider");

    let temp_dir = tempfile::tempdir().unwrap();
    let server = GatewayServer::new(test_config(&temp_dir), None)
        .await
        .unwrap();
    let (status, body) = response_body(server.build_router(), "/dashboard").await;
    assert_eq!(status, StatusCode::OK);
    let html = std::str::from_utf8(&body).unwrap();
    assert!(html.contains("(window.__dashboardBasePath||'/dashboard')+'/ws'"));
    assert!(html.contains("new WebSocket(url)"));
    assert!(html.contains("msg.type==='compression'&&msg.data"));
    assert!(html.contains("handleCompressionEvent(msg.data)"));
}

#[tokio::test]
async fn dashboard_logs_filter_returns_only_matching_persisted_compression_metadata() {
    let temp_dir = tempfile::tempdir().unwrap();
    let server = GatewayServer::new(test_config(&temp_dir), None)
        .await
        .unwrap();
    let pipeline = CompressionPipeline::from_config(CompressionConfig::default());
    let standard = pipeline_stats(
        &pipeline,
        CompressionLevel::Standard,
        "trace-standard",
        "observability-provider",
    )
    .await;
    let lite = pipeline_stats(
        &pipeline,
        CompressionLevel::Lite,
        "trace-lite",
        "observability-provider",
    )
    .await;

    server
        .state
        .logger
        .log(log_entry(
            "trace-standard",
            Some(CompressionLogMetadata::from(&standard)),
        ))
        .unwrap();
    server
        .state
        .logger
        .log(log_entry(
            "trace-lite",
            Some(CompressionLogMetadata::from(&lite)),
        ))
        .unwrap();
    server
        .state
        .logger
        .log(log_entry("trace-uncompressed", None))
        .unwrap();

    let (status, body) = response_body(
        server.build_router(),
        "/dashboard/logs?compression_level=standard",
    )
    .await;
    assert_eq!(status, StatusCode::OK);
    let entries: Vec<LogEntry> = serde_json::from_slice(&body).unwrap();
    assert_eq!(entries.len(), 1);
    assert_eq!(entries[0].trace_id, "trace-standard");
    let metadata = entries[0].compression.as_ref().unwrap();
    assert_eq!(metadata.compression_level, "standard");
    assert_eq!(metadata.original_tokens, standard.original_tokens);
    assert_eq!(metadata.compressed_tokens, standard.compressed_tokens);
    assert_eq!(metadata.engines_applied, standard.engines_applied);
}

#[tokio::test]
async fn prometheus_endpoint_exposes_exact_compression_metric_names_after_recording() {
    let temp_dir = tempfile::tempdir().unwrap();
    let server = GatewayServer::new(test_config(&temp_dir), None)
        .await
        .unwrap();
    let pipeline = CompressionPipeline::from_config(CompressionConfig::default());
    let stats = pipeline_stats(
        &pipeline,
        CompressionLevel::Stacked,
        "metrics-stacked",
        "observability-provider",
    )
    .await;
    server.state.metrics.record_compression(&stats);

    let (status, body) = response_body(server.build_router(), "/metrics").await;
    assert_eq!(status, StatusCode::OK);
    let metrics = std::str::from_utf8(&body).unwrap();

    assert!(metrics.contains("# TYPE obey_compression_tokens_saved_total counter"));
    assert!(metrics.contains(
        "obey_compression_tokens_saved_total{level=\"stacked\",provider=\"observability-provider\"}"
    ));
    assert!(metrics.contains("# TYPE obey_compression_ratio histogram"));
    assert!(metrics.contains(
        "obey_compression_ratio_count{level=\"stacked\",provider=\"observability-provider\"} 1"
    ));
    assert!(metrics.contains("# TYPE obey_compression_duration_seconds histogram"));
    assert!(metrics.contains(
        "obey_compression_duration_seconds_count{level=\"stacked\",provider=\"observability-provider\"} 1"
    ));
    assert!(!metrics.contains("obey_api_compression_"));
}

#[tokio::test]
async fn cache_aware_pipeline_keeps_the_cached_prefix_byte_stable() {
    let suffix = (0..80)
        .map(|index| format!("Compiling cached_suffix_crate_{index}   v1.0.0"))
        .chain(std::iter::once("Finished   release   target(s)".to_owned()))
        .collect::<Vec<_>>()
        .join("\n");
    let request: OpenAIRequest = serde_json::from_value(json!({
        "model": "claude-test",
        "messages": [
            {"role": "user", "content": "cached prefix   must remain exactly stable"},
            {
                "role": "assistant",
                "content": [{
                    "type": "text",
                    "text": "cache boundary   remains exact",
                    "cache_control": {"type": "ephemeral"}
                }]
            },
            {
                "role": "tool",
                "content": suffix,
                "command": "cargo build --release"
            }
        ]
    }))
    .unwrap();
    let original: CompressiblePayload = request.into();
    let prefix = original.messages[..=1].to_vec();
    let original_suffix: Value = original.messages[2].content.as_value().clone();
    let pipeline = CompressionPipeline::from_config(CompressionConfig::default());

    for level in [
        CompressionLevel::Aggressive,
        CompressionLevel::Ultra,
        CompressionLevel::Rtk,
        CompressionLevel::Stacked,
    ] {
        let context = CompressionContext {
            model: "claude-test".to_owned(),
            provider_name: "anthropic".to_owned(),
            prompt_caching_enabled: true,
            ..Default::default()
        };
        let result = pipeline
            .compress_explicit(
                original.clone(),
                context,
                effective(level),
                CompressionRequestMetadata {
                    request_id: format!("cache-prefix-{level:?}"),
                    ..Default::default()
                },
            )
            .await;

        assert_eq!(result.payload.messages[..=1], prefix, "level {level:?}");
        assert!(result.cache_downgrade_applied, "level {level:?}");
        let downgrade = result.cache_downgrade.as_ref().unwrap();
        assert_eq!(downgrade.requested_level, level);
        assert_eq!(downgrade.actual_prefix_level, CompressionLevel::None);
        assert_eq!(downgrade.boundary_message_index, 1);
        assert_eq!(downgrade.provider, "anthropic");

        if level == CompressionLevel::Rtk {
            assert_ne!(
                result.payload.messages[2].content.as_value(),
                &original_suffix,
                "the uncached suffix should remain eligible for compression"
            );
        }
    }
}
