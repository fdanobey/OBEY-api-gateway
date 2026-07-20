use ai_gateway::{
    compression::{
        config::{
            reload_compression_config, shared_compression_config, CompressionConfig,
            CustomPipelineConfig, EffectiveCompressionConfig, TimeBudgetConfig,
        },
        pipeline::{CompressionPipeline, CompressionRequestMetadata},
        protection::ProtectionScanner,
        token_counter::TokenCounter,
        CompressiblePayload, CompressionContext, CompressionEngine, CompressionLevel, EngineResult,
    },
    models::openai::{OpenAIRequest, OpenAIResponse},
};
use async_trait::async_trait;
use serde_json::{json, Value};
use std::{collections::HashMap, sync::Arc, time::Duration};

const LONG_BUDGET: u64 = 60_000;

struct SlowEngine;

#[async_trait]
impl CompressionEngine for SlowEngine {
    fn name(&self) -> &str {
        "lite"
    }

    async fn compress(
        &self,
        payload: &mut CompressiblePayload,
        context: &CompressionContext,
    ) -> EngineResult {
        let tokens_before = context
            .token_counter
            .count_request(&payload.clone().into_openai_request());
        if let Some(message) = payload.messages.first_mut() {
            *message.content.as_value_mut() = Value::String("mutated before timeout".to_owned());
        }
        tokio::time::sleep(Duration::from_millis(50)).await;
        let tokens_after = context
            .token_counter
            .count_request(&payload.clone().into_openai_request());
        EngineResult {
            engine_name: self.name().to_owned(),
            tokens_before,
            tokens_after,
            duration_ms: 50,
            applied: true,
        }
    }
}

fn production_config() -> CompressionConfig {
    CompressionConfig {
        enabled: true,
        default_level: CompressionLevel::Standard,
        time_budget_ms: TimeBudgetConfig {
            lite: LONG_BUDGET,
            standard: LONG_BUDGET,
            aggressive: LONG_BUDGET,
            ultra: LONG_BUDGET,
            rtk: LONG_BUDGET,
            stacked: LONG_BUDGET,
        },
        ..CompressionConfig::default()
    }
}

fn effective(enabled: bool, level: CompressionLevel) -> EffectiveCompressionConfig {
    EffectiveCompressionConfig {
        enabled,
        level,
        auto_threshold_tokens: 0,
        caveman_output: false,
    }
}

fn verbose_request(stream: bool) -> OpenAIRequest {
    serde_json::from_value(json!({
        "model": "gpt-4o",
        "stream": stream,
        "messages": [{
            "role": "user",
            "content": "Could you please basically take a look at this module in order to make sure that it is able to compile due to the fact that it is very important?"
        }]
    }))
    .expect("test request must deserialize")
}

fn full_openai_request() -> (OpenAIRequest, Value) {
    let wire = json!({
        "model": "gpt-4o",
        "stream": true,
        "temperature": 0.25,
        "max_tokens": 512,
        "top_p": 0.9,
        "metadata": {"tenant": "compression-e2e", "trace": true},
        "response_format": {"type": "json_object"},
        "tool_choice": {"type": "function", "function": {"name": "lookup_record"}},
        "tools": [{
            "type": "function",
            "provider_extension": {"strict": true},
            "function": {
                "name": "lookup_record",
                "description": "The purpose of this tool is to look up a record by its stable identifier. For example, pass record_123. Note: archived records may not be available. This repeated explanatory material is intentionally verbose.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "record_id": {
                            "type": "string",
                            "description": "The purpose of this function is to identify the record. Example: record_123.",
                            "enum": ["record_123", "record_456"]
                        }
                    },
                    "required": ["record_id"],
                    "additionalProperties": false
                }
            }
        }],
        "messages": [
            {
                "role": "system",
                "content": "Keep tool identifiers and all structured values exact.",
                "name": "policy"
            },
            {
                "role": "user",
                "content": [
                    {
                        "type": "text",
                        "text": "Could you please basically take a look at this image in order to identify the record?"
                    },
                    {
                        "type": "image_url",
                        "image_url": {"url": "data:image/png;base64,AAEC", "detail": "high"},
                        "vendor_field": {"preserve": true}
                    }
                ]
            },
            {
                "role": "assistant",
                "content": "I would like you to know that I am calling the lookup tool.",
                "tool_calls": [{
                    "id": "call_record_1",
                    "type": "function",
                    "function": {
                        "name": "lookup_record",
                        "arguments": "{\"record_id\":\"record_123\"}"
                    },
                    "vendor_trace": "trace-1"
                }]
            },
            {
                "role": "tool",
                "tool_call_id": "call_record_1",
                "name": "lookup_record",
                "content": "{\"record_id\":\"record_123\",\"status\":\"active\"}",
                "vendor_status": {"ok": true}
            },
            {
                "role": "user",
                "content": "Could you please provide an explanation of the active record in order to make sure that it is very clear?"
            }
        ]
    });
    let request = serde_json::from_value(wire.clone()).expect("full request must deserialize");
    (request, wire)
}

fn without_descriptions(value: &Value) -> Value {
    match value {
        Value::Array(values) => Value::Array(values.iter().map(without_descriptions).collect()),
        Value::Object(object) => Value::Object(
            object
                .iter()
                .filter(|(field, _)| field.as_str() != "description")
                .map(|(field, value)| (field.clone(), without_descriptions(value)))
                .collect(),
        ),
        primitive => primitive.clone(),
    }
}

#[tokio::test]
async fn production_pipeline_preserves_full_openai_structure_and_stream_roundtrip() {
    let mut config = production_config();
    config.compress_tool_definitions = true;
    let shared = shared_compression_config(config.clone()).expect("config must validate");
    let pipeline = CompressionPipeline::new(shared).await;
    let (request, original_wire) = full_openai_request();

    let result = pipeline
        .compress_explicit(
            CompressiblePayload::from_openai_request(request),
            CompressionContext::new("gpt-4o", "openai"),
            config.resolve(None, None),
            CompressionRequestMetadata {
                request_id: "full-openai-e2e".to_owned(),
                ..CompressionRequestMetadata::default()
            },
        )
        .await;
    let outgoing = result.payload.into_openai_request();
    let outgoing_wire = serde_json::to_value(&outgoing).expect("output must serialize");

    assert!(!result.timed_out);
    assert!(result.errors.is_empty());
    assert!(result.final_tokens <= result.original_tokens);
    assert!(result.tool_definitions_compressed);
    assert_eq!(outgoing.model, "gpt-4o");
    assert!(outgoing.stream);
    assert_eq!(outgoing_wire["stream"], json!(true));
    assert_eq!(outgoing_wire["temperature"], original_wire["temperature"]);
    assert_eq!(outgoing_wire["max_tokens"], original_wire["max_tokens"]);
    assert_eq!(outgoing_wire["top_p"], original_wire["top_p"]);
    assert_eq!(outgoing_wire["metadata"], original_wire["metadata"]);
    assert_eq!(
        outgoing_wire["response_format"],
        original_wire["response_format"]
    );
    assert_eq!(outgoing_wire["tool_choice"], original_wire["tool_choice"]);
    assert_eq!(
        without_descriptions(&outgoing_wire["tools"]),
        without_descriptions(&original_wire["tools"])
    );
    assert_ne!(outgoing_wire["tools"], original_wire["tools"]);
    assert_eq!(outgoing.messages.len(), 5);
    assert_eq!(
        outgoing
            .messages
            .iter()
            .map(|message| message.role.as_str())
            .collect::<Vec<_>>(),
        ["system", "user", "assistant", "tool", "user"]
    );
    assert_eq!(
        outgoing_wire["messages"][1]["content"][1],
        original_wire["messages"][1]["content"][1]
    );
    assert_eq!(
        outgoing_wire["messages"][2]["tool_calls"],
        original_wire["messages"][2]["tool_calls"]
    );
    assert_eq!(
        outgoing_wire["messages"][3]["tool_call_id"],
        original_wire["messages"][3]["tool_call_id"]
    );
    assert_eq!(
        outgoing_wire["messages"][3]["content"],
        original_wire["messages"][3]["content"]
    );
}

#[tokio::test]
async fn existing_pipeline_uses_reloaded_enabled_level_and_custom_chain() {
    let initial = CompressionConfig {
        enabled: false,
        default_level: CompressionLevel::Lite,
        ..production_config()
    };
    let shared = shared_compression_config(initial).expect("initial config must validate");
    let pipeline = CompressionPipeline::new(Arc::clone(&shared)).await;
    let original = CompressiblePayload::from_openai_request(verbose_request(false));
    let disabled_effective = shared.read().await.resolve(None, None);

    let disabled = pipeline
        .compress_explicit(
            original.clone(),
            CompressionContext::new("gpt-4o", "openai"),
            disabled_effective,
            CompressionRequestMetadata {
                request_id: "before-reload".to_owned(),
                ..CompressionRequestMetadata::default()
            },
        )
        .await;
    assert_eq!(disabled.payload, original);
    assert!(disabled.engine_results.is_empty());

    let mut replacement = production_config();
    replacement.custom_pipelines.insert(
        "live_standard".to_owned(),
        CustomPipelineConfig {
            engines: vec!["standard".to_owned()],
        },
    );
    reload_compression_config(&shared, replacement)
        .await
        .expect("replacement config must validate");
    let enabled_effective = shared.read().await.resolve(None, None);
    assert_eq!(enabled_effective.level, CompressionLevel::Standard);

    let enabled = pipeline
        .compress_explicit(
            original.clone(),
            CompressionContext::new("gpt-4o", "openai"),
            enabled_effective,
            CompressionRequestMetadata {
                request_id: "after-reload".to_owned(),
                custom_pipeline: Some("live_standard".to_owned()),
                ..CompressionRequestMetadata::default()
            },
        )
        .await;

    assert!(!enabled.timed_out);
    assert!(enabled.errors.is_empty());
    assert_eq!(enabled.engine_results.len(), 1);
    assert_eq!(enabled.engine_results[0].engine_name, "standard");
    assert_eq!(enabled.engines_applied, ["standard"]);
    assert!(enabled.final_tokens < enabled.original_tokens);
    assert_ne!(enabled.payload, original);
}

#[tokio::test]
async fn one_millisecond_budget_returns_exact_original_and_marks_timeout() {
    let config = CompressionConfig {
        enabled: true,
        default_level: CompressionLevel::Lite,
        time_budget_ms: TimeBudgetConfig {
            lite: 1,
            ..TimeBudgetConfig::default()
        },
        ..CompressionConfig::default()
    };
    let pipeline = CompressionPipeline::with_engines(
        shared_compression_config(config).expect("timeout config must validate"),
        Arc::new(TokenCounter::new()),
        Arc::new(ProtectionScanner::default()),
        HashMap::from([(
            "lite".to_owned(),
            Arc::new(SlowEngine) as Arc<dyn CompressionEngine>,
        )]),
    );
    let original = CompressiblePayload::from_openai_request(verbose_request(false));

    let result = pipeline
        .compress_explicit(
            original.clone(),
            CompressionContext::new("gpt-4o", "openai"),
            effective(true, CompressionLevel::Lite),
            CompressionRequestMetadata {
                request_id: "timeout-e2e".to_owned(),
                ..CompressionRequestMetadata::default()
            },
        )
        .await;

    assert!(result.timed_out);
    assert_eq!(result.payload, original);
    assert_eq!(result.final_tokens, result.original_tokens);
}

#[tokio::test]
async fn streaming_request_compression_does_not_mutate_separate_response() {
    let config = production_config();
    let pipeline = CompressionPipeline::from_config(config.clone());
    let request = verbose_request(true);
    let original_request = serde_json::to_value(&request).expect("request must serialize");
    let response: OpenAIResponse = serde_json::from_value(json!({
        "id": "chatcmpl-response-isolation",
        "object": "chat.completion",
        "created": 1_721_500_000,
        "model": "gpt-4o",
        "choices": [{
            "index": 0,
            "message": {
                "role": "assistant",
                "content": "I hope this response remains exactly untouched."
            },
            "finish_reason": "stop",
            "provider_choice_field": {"stable": true}
        }],
        "usage": {
            "prompt_tokens": 100,
            "completion_tokens": 10,
            "total_tokens": 110
        },
        "system_fingerprint": "fp_stable"
    }))
    .expect("response must deserialize");
    let response_before = serde_json::to_value(&response).expect("response must serialize");

    let result = pipeline
        .compress_explicit(
            CompressiblePayload::from_openai_request(request),
            CompressionContext::new("gpt-4o", "openai"),
            config.resolve(None, None),
            CompressionRequestMetadata {
                request_id: "stream-isolation-e2e".to_owned(),
                ..CompressionRequestMetadata::default()
            },
        )
        .await;
    let outgoing = result.payload.into_openai_request();
    let outgoing_wire = serde_json::to_value(&outgoing).expect("output must serialize");

    assert!(outgoing.stream);
    assert_ne!(outgoing_wire, original_request);
    assert!(result.final_tokens < result.original_tokens);
    assert_eq!(
        serde_json::to_value(&response).expect("response must still serialize"),
        response_before
    );
}
