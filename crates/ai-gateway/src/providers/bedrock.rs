use async_trait::async_trait;
use aws_config::BehaviorVersion;
use aws_sdk_bedrock::Client as BedrockControlClient;
use aws_sdk_bedrockruntime::{
    operation::{converse::ConverseOutput as SdkConverseOutput, invoke_model::InvokeModelOutput},
    types::{
        ContentBlock, ConversationRole, InferenceConfiguration, Message as BedrockMessage,
        SystemContentBlock,
    },
    Client as BedrockClient,
};
use futures::stream::{Stream, StreamExt};
use reqwest::Client;
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::pin::Pin;
use std::time::Instant;

use crate::error::GatewayError;
use crate::models::openai::{Choice, Message, OpenAIRequest, OpenAIResponse, Usage};
use crate::providers::{Model, ProviderClient, ProviderResponse, SSEEvent};

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub(crate) enum MantleApi {
    Chat,
    Responses,
    Messages,
}

fn mantle_api_for_model(model_id: &str) -> MantleApi {
    let id = model_id.to_ascii_lowercase();
    if id.starts_with("openai.gpt-5.6")
        || id.starts_with("openai.gpt-5.5")
        || id.starts_with("openai.gpt-5.4")
    {
        MantleApi::Responses
    } else if id.starts_with("anthropic.claude") {
        MantleApi::Messages
    } else {
        MantleApi::Chat
    }
}

pub(crate) fn normalize_mantle_chat_messages(request: &mut OpenAIRequest) -> usize {
    let mut normalized = 0;
    for message in &mut request.messages {
        if message.role == "developer" {
            message.role = "system".to_string();
            normalized += 1;
        }

        if let serde_json::Value::Array(parts) = &mut message.content {
            for part in parts {
                let Some(object) = part.as_object_mut() else {
                    continue;
                };
                match object.get("type").and_then(serde_json::Value::as_str) {
                    Some("input_text" | "output_text") => {
                        object.insert("type".to_string(), serde_json::json!("text"));
                        normalized += 1;
                    }
                    _ => {}
                }
                if object.remove("cache_control").is_some() {
                    normalized += 1;
                }
            }
        }
    }
    normalized
}

pub(crate) fn sanitize_mantle_chat_request(request: &mut OpenAIRequest) -> usize {
    const MANTLE_CHAT_ALLOWED: &[&str] = &[
        "tools",
        "tool_choice",
        "parallel_tool_calls",
        "top_p",
        "frequency_penalty",
        "presence_penalty",
        "stop",
        "seed",
        "response_format",
        "reasoning_effort",
        "logprobs",
        "top_logprobs",
        "service_tier",
        "user",
    ];

    let before = request.extra.len();
    request
        .extra
        .retain(|key, _| MANTLE_CHAT_ALLOWED.contains(&key.as_str()));
    before - request.extra.len()
}

/// Default pool idle timeout in seconds
const DEFAULT_POOL_IDLE_TIMEOUT_SECS: u64 = 90;

/// Supported AWS Bedrock regions.
pub const BEDROCK_REGIONS: &[&str] = &[
    "us-east-1",
    "us-east-2",
    "us-west-2",
    "eu-west-1",
    "eu-west-3",
    "eu-central-1",
    "ap-northeast-1",
    "ap-southeast-1",
    "ap-southeast-2",
    "ap-south-1",
    "sa-east-1",
    "ca-central-1",
    "us-gov-west-1",
];

/// A fallback catalog entry for Bedrock model discovery.
/// One struct backs both the Mantle Chat and runtime fallback lists; the
/// generated block per catalog is delimited so `scripts/sync-bedrock-fallback.ps1`
/// can rewrite it without touching surrounding code.
pub struct BedrockFallbackModel {
    pub id: &'static str,
    pub owned_by: &'static str,
    pub supports_vision: bool,
    pub context_window: Option<u32>,
    pub max_completion_tokens: Option<u32>,
    pub source_url: &'static str,
}

// BEGIN BEDROCK MANTLE CHAT FALLBACK MODELS
// Source: AWS Bedrock model cards (Programmatic Access) + API compatibility table.
// Only Chat Completions-capable bedrock-mantle models with verified IDs.
// Responses-only (GPT-5.5/5.4/GPT-5.6) and Messages-only (Claude) models are
// intentionally excluded pending dedicated adapters.
pub const BEDROCK_MANTLE_CHAT_FALLBACK: &[BedrockFallbackModel] = &[
    BedrockFallbackModel {
        id: "openai.gpt-oss-120b",
        owned_by: "openai",
        supports_vision: false,
        context_window: Some(128_000),
        max_completion_tokens: None,
        source_url: "https://docs.aws.amazon.com/bedrock/latest/userguide/model-card-openai-gpt-oss-120b.html",
    },
    BedrockFallbackModel {
        id: "openai.gpt-oss-20b",
        owned_by: "openai",
        supports_vision: false,
        context_window: Some(128_000),
        max_completion_tokens: None,
        source_url: "https://docs.aws.amazon.com/bedrock/latest/userguide/model-card-openai-gpt-oss-20b.html",
    },
    BedrockFallbackModel {
        id: "deepseek.v3.2",
        owned_by: "deepseek",
        supports_vision: false,
        context_window: Some(164_000),
        max_completion_tokens: Some(8_192),
        source_url: "https://docs.aws.amazon.com/bedrock/latest/userguide/model-card-deepseek-deepseek-v3-2.html",
    },
    BedrockFallbackModel {
        id: "deepseek.v3.1",
        owned_by: "deepseek",
        supports_vision: false,
        context_window: Some(128_000),
        max_completion_tokens: Some(8_192),
        source_url: "https://docs.aws.amazon.com/bedrock/latest/userguide/model-card-deepseek-deepseek-v3-1.html",
    },
    BedrockFallbackModel {
        id: "mistral.mistral-large-3-675b-instruct",
        owned_by: "mistral",
        supports_vision: false,
        context_window: None,
        max_completion_tokens: None,
        source_url: "https://docs.aws.amazon.com/bedrock/latest/userguide/model-card-mistral-ai-mistral-large-3.html",
    },
    BedrockFallbackModel {
        id: "qwen.qwen3-32b",
        owned_by: "qwen",
        supports_vision: false,
        context_window: None,
        max_completion_tokens: None,
        source_url: "https://docs.aws.amazon.com/bedrock/latest/userguide/model-card-qwen-qwen3-32b.html",
    },
    BedrockFallbackModel {
        id: "qwen.qwen3-235b-a22b-2507",
        owned_by: "qwen",
        supports_vision: false,
        context_window: None,
        max_completion_tokens: None,
        source_url: "https://docs.aws.amazon.com/bedrock/latest/userguide/model-cards.html",
    },
    BedrockFallbackModel {
        id: "qwen.qwen3-coder-480b-a35b-instruct",
        owned_by: "qwen",
        supports_vision: false,
        context_window: None,
        max_completion_tokens: None,
        source_url: "https://docs.aws.amazon.com/bedrock/latest/userguide/model-cards.html",
    },
];
// END BEDROCK MANTLE CHAT FALLBACK MODELS

// BEGIN BEDROCK MANTLE RESPONSES FALLBACK MODELS
pub const BEDROCK_MANTLE_RESPONSES_FALLBACK: &[BedrockFallbackModel] = &[
    BedrockFallbackModel {
        id: "openai.gpt-5.6-sol",
        owned_by: "openai",
        supports_vision: true,
        context_window: Some(272_000),
        max_completion_tokens: None,
        source_url: "https://docs.aws.amazon.com/bedrock/latest/userguide/model-card-openai-gpt-56-sol.html",
    },
    BedrockFallbackModel {
        id: "openai.gpt-5.6-terra",
        owned_by: "openai",
        supports_vision: true,
        context_window: Some(272_000),
        max_completion_tokens: None,
        source_url: "https://docs.aws.amazon.com/bedrock/latest/userguide/model-card-openai-gpt-56-terra.html",
    },
    BedrockFallbackModel {
        id: "openai.gpt-5.6-luna",
        owned_by: "openai",
        supports_vision: true,
        context_window: Some(272_000),
        max_completion_tokens: None,
        source_url: "https://docs.aws.amazon.com/bedrock/latest/userguide/model-card-openai-gpt-56-luna.html",
    },
    BedrockFallbackModel {
        id: "openai.gpt-5.5",
        owned_by: "openai",
        supports_vision: true,
        context_window: Some(272_000),
        max_completion_tokens: None,
        source_url: "https://docs.aws.amazon.com/bedrock/latest/userguide/model-card-openai-gpt-55.html",
    },
    BedrockFallbackModel {
        id: "openai.gpt-5.4",
        owned_by: "openai",
        supports_vision: true,
        context_window: Some(272_000),
        max_completion_tokens: None,
        source_url: "https://docs.aws.amazon.com/bedrock/latest/userguide/model-card-openai-gpt-54.html",
    },
];
// END BEDROCK MANTLE RESPONSES FALLBACK MODELS

// BEGIN BEDROCK MANTLE MESSAGES FALLBACK MODELS
pub const BEDROCK_MANTLE_MESSAGES_FALLBACK: &[BedrockFallbackModel] = &[
    BedrockFallbackModel {
        id: "anthropic.claude-sonnet-5",
        owned_by: "anthropic",
        supports_vision: true,
        context_window: Some(1_000_000),
        max_completion_tokens: Some(131_072),
        source_url: "https://docs.aws.amazon.com/bedrock/latest/userguide/model-card-anthropic-claude-sonnet-5.html",
    },
    BedrockFallbackModel {
        id: "anthropic.claude-fable-5",
        owned_by: "anthropic",
        supports_vision: true,
        context_window: Some(1_000_000),
        max_completion_tokens: Some(131_072),
        source_url: "https://docs.aws.amazon.com/bedrock/latest/userguide/model-card-anthropic-claude-fable-5.html",
    },
    BedrockFallbackModel {
        id: "anthropic.claude-opus-4-8",
        owned_by: "anthropic",
        supports_vision: true,
        context_window: None,
        max_completion_tokens: None,
        source_url: "https://docs.aws.amazon.com/bedrock/latest/userguide/model-card-anthropic-claude-opus-4-8.html",
    },
    BedrockFallbackModel {
        id: "anthropic.claude-opus-4-7",
        owned_by: "anthropic",
        supports_vision: true,
        context_window: None,
        max_completion_tokens: None,
        source_url: "https://docs.aws.amazon.com/bedrock/latest/userguide/model-cards.html",
    },
];
// END BEDROCK MANTLE MESSAGES FALLBACK MODELS

// BEGIN BEDROCK RUNTIME FALLBACK MODELS
// Source: AWS Bedrock model cards (Programmatic Access) + ListFoundationModels.
// Only Converse/Invoke-capable runtime models. Legacy models (Nova Premier,
// meta.llama3-1-405b, AI21 Jamba, Cohere Command R/R+) are excluded.
// SDK chat currently routes through InvokeModel with legacy translators; once
// Converse migration lands, this catalog's IDs all become truthfully invocable.
pub const BEDROCK_RUNTIME_FALLBACK: &[BedrockFallbackModel] = &[
    BedrockFallbackModel {
        id: "anthropic.claude-sonnet-5",
        owned_by: "anthropic",
        supports_vision: true,
        context_window: Some(1_000_000),
        max_completion_tokens: Some(131_072),
        source_url: "https://docs.aws.amazon.com/bedrock/latest/userguide/model-card-anthropic-claude-sonnet-5.html",
    },
    BedrockFallbackModel {
        id: "anthropic.claude-opus-4-8",
        owned_by: "anthropic",
        supports_vision: true,
        context_window: None,
        max_completion_tokens: None,
        source_url: "https://docs.aws.amazon.com/bedrock/latest/userguide/model-card-anthropic-claude-opus-4-8.html",
    },
    BedrockFallbackModel {
        id: "anthropic.claude-opus-4-7",
        owned_by: "anthropic",
        supports_vision: true,
        context_window: None,
        max_completion_tokens: None,
        source_url: "https://docs.aws.amazon.com/bedrock/latest/userguide/model-cards.html",
    },
    BedrockFallbackModel {
        id: "anthropic.claude-haiku-4-5-20251001-v1:0",
        owned_by: "anthropic",
        supports_vision: true,
        context_window: None,
        max_completion_tokens: None,
        source_url: "https://docs.aws.amazon.com/bedrock/latest/userguide/model-cards.html",
    },
    BedrockFallbackModel {
        id: "amazon.nova-2-lite-v1:0",
        owned_by: "amazon",
        supports_vision: true,
        context_window: Some(1_000_000),
        max_completion_tokens: None,
        source_url: "https://docs.aws.amazon.com/bedrock/latest/userguide/model-cards.html",
    },
    BedrockFallbackModel {
        id: "amazon.nova-pro-v1:0",
        owned_by: "amazon",
        supports_vision: true,
        context_window: None,
        max_completion_tokens: Some(25_000),
        source_url: "https://docs.aws.amazon.com/bedrock/latest/userguide/model-cards.html",
    },
    BedrockFallbackModel {
        id: "amazon.nova-lite-v1:0",
        owned_by: "amazon",
        supports_vision: true,
        context_window: Some(300_000),
        max_completion_tokens: None,
        source_url: "https://docs.aws.amazon.com/bedrock/latest/userguide/model-cards.html",
    },
    BedrockFallbackModel {
        id: "amazon.nova-micro-v1:0",
        owned_by: "amazon",
        supports_vision: false,
        context_window: Some(128_000),
        max_completion_tokens: None,
        source_url: "https://docs.aws.amazon.com/bedrock/latest/userguide/model-cards.html",
    },
    BedrockFallbackModel {
        id: "openai.gpt-oss-120b-1:0",
        owned_by: "openai",
        supports_vision: false,
        context_window: Some(128_000),
        max_completion_tokens: None,
        source_url: "https://docs.aws.amazon.com/bedrock/latest/userguide/model-card-openai-gpt-oss-120b.html",
    },
    BedrockFallbackModel {
        id: "openai.gpt-oss-20b-1:0",
        owned_by: "openai",
        supports_vision: false,
        context_window: Some(128_000),
        max_completion_tokens: None,
        source_url: "https://docs.aws.amazon.com/bedrock/latest/userguide/model-card-openai-gpt-oss-20b.html",
    },
    BedrockFallbackModel {
        id: "deepseek.v3.2",
        owned_by: "deepseek",
        supports_vision: false,
        context_window: Some(164_000),
        max_completion_tokens: Some(8_192),
        source_url: "https://docs.aws.amazon.com/bedrock/latest/userguide/model-card-deepseek-deepseek-v3-2.html",
    },
    BedrockFallbackModel {
        id: "deepseek.v3.1",
        owned_by: "deepseek",
        supports_vision: false,
        context_window: Some(128_000),
        max_completion_tokens: Some(8_192),
        source_url: "https://docs.aws.amazon.com/bedrock/latest/userguide/model-card-deepseek-deepseek-v3-1.html",
    },
    BedrockFallbackModel {
        id: "meta.llama3-1-70b-instruct-v1:0",
        owned_by: "meta",
        supports_vision: false,
        context_window: Some(128_000),
        max_completion_tokens: None,
        source_url: "https://docs.aws.amazon.com/bedrock/latest/userguide/model-cards.html",
    },
    BedrockFallbackModel {
        id: "meta.llama3-1-8b-instruct-v1:0",
        owned_by: "meta",
        supports_vision: false,
        context_window: Some(128_000),
        max_completion_tokens: None,
        source_url: "https://docs.aws.amazon.com/bedrock/latest/userguide/model-cards.html",
    },
    BedrockFallbackModel {
        id: "meta.llama3-3-70b-instruct-v1:0",
        owned_by: "meta",
        supports_vision: false,
        context_window: None,
        max_completion_tokens: None,
        source_url: "https://docs.aws.amazon.com/bedrock/latest/userguide/model-cards.html",
    },
    BedrockFallbackModel {
        id: "mistral.mistral-large-3-675b-instruct",
        owned_by: "mistral",
        supports_vision: false,
        context_window: None,
        max_completion_tokens: None,
        source_url: "https://docs.aws.amazon.com/bedrock/latest/userguide/model-card-mistral-ai-mistral-large-3.html",
    },
    BedrockFallbackModel {
        id: "qwen.qwen3-32b-v1:0",
        owned_by: "qwen",
        supports_vision: false,
        context_window: None,
        max_completion_tokens: None,
        source_url: "https://docs.aws.amazon.com/bedrock/latest/userguide/model-card-qwen-qwen3-32b.html",
    },
];
// END BEDROCK RUNTIME FALLBACK MODELS

fn fallback_entries_to_models(entries: &[BedrockFallbackModel]) -> Vec<Model> {
    entries
        .iter()
        .map(|entry| Model {
            id: entry.id.to_string(),
            object: "model".to_string(),
            owned_by: entry.owned_by.to_string(),
            created: None,
            context_window: entry.context_window,
            max_completion_tokens: entry.max_completion_tokens,
            supports_vision: entry.supports_vision,
        })
        .collect()
}

/// Derive the region group code from an AWS region string.
/// Maps region prefixes to group codes used for global inference profile model ID prefixing.
/// Returns `"us"`, `"eu"`, `"ap"`, `"sa"`, `"ca"`, or `""` for unknown prefixes.
pub fn derive_region_group(region: &str) -> &str {
    if region.starts_with("us-") {
        "us"
    } else if region.starts_with("eu-") {
        "eu"
    } else if region.starts_with("ap-") {
        "ap"
    } else if region.starts_with("sa-") {
        "sa"
    } else if region.starts_with("ca-") {
        "ca"
    } else {
        ""
    }
}

/// Apply global inference profile prefix to a model ID.
///
/// When `enabled` is `true`, prepends the region group (e.g., `us.`) derived from `region`
/// to the model ID. If the model ID already starts with the region group prefix, it is
/// returned unchanged to avoid double-prefixing. When `enabled` is `false`, the model ID
/// is returned unchanged.
pub fn apply_global_inference_prefix(model_id: &str, region: &str, enabled: bool) -> String {
    if !enabled || !model_supports_geo_inference(model_id) {
        return model_id.to_string();
    }
    let region_group = derive_region_group(region);
    if region_group.is_empty() {
        return model_id.to_string();
    }
    let prefix = format!("{}.", region_group);
    if model_id.starts_with(&prefix) {
        model_id.to_string()
    } else {
        format!("{}{}", prefix, model_id)
    }
}

pub fn apply_global_inference_profile(model_id: &str, enabled: bool) -> String {
    if !enabled || !model_supports_global_inference(model_id) {
        return model_id.to_string();
    }
    if model_id.starts_with("global.") {
        model_id.to_string()
    } else {
        format!("global.{}", model_id)
    }
}

pub fn model_supports_geo_inference(model_id: &str) -> bool {
    matches!(
        bedrock_base_model_id(model_id),
        "anthropic.claude-sonnet-5"
            | "anthropic.claude-sonnet-4-5-20250929-v1:0"
            | "anthropic.claude-opus-4-8"
            | "anthropic.claude-opus-4-7"
            | "anthropic.claude-haiku-4-5-20251001-v1:0"
            | "amazon.nova-pro-v1:0"
            | "amazon.nova-lite-v1:0"
            | "amazon.nova-micro-v1:0"
            | "writer.palmyra-x5-v1:0"
    )
}

pub fn model_supports_global_inference(model_id: &str) -> bool {
    matches!(
        bedrock_base_model_id(model_id),
        "anthropic.claude-sonnet-5" | "anthropic.claude-sonnet-4-5-20250929-v1:0"
    )
}

fn bedrock_base_model_id(model_id: &str) -> &str {
    const PROFILE_PREFIXES: &[&str] = &["global.", "us.", "eu.", "ap.", "au.", "jp."];
    PROFILE_PREFIXES
        .iter()
        .find_map(|prefix| model_id.strip_prefix(prefix))
        .unwrap_or(model_id)
}

/// Check whether a model ID refers to a reasoning-capable model.
/// Returns `true` for Claude 3.5 Sonnet v2 and later model patterns that support
/// extended thinking via the `thinking` parameter.
pub fn model_supports_reasoning(model_id: &str) -> bool {
    // Claude 3.5 Sonnet v2+ (e.g. anthropic.claude-3-5-sonnet-20241022-v2:0, us.anthropic.claude-3-5-sonnet-...)
    // Claude 3.5 Haiku models
    // Claude 3 Opus and later
    // Claude 4+ family
    let id = model_id.to_lowercase();

    // Claude 3.5 Sonnet v2 or later — contains "claude-3-5-sonnet" with v2+
    if id.contains("claude-3-5-sonnet") {
        // Check for v2 or later version suffix
        if let Some(pos) = id.find("-v") {
            let after_v = &id[pos + 2..];
            if let Some(version) = after_v.chars().next().and_then(|c| c.to_digit(10)) {
                return version >= 2;
            }
        }
        return false;
    }

    // Claude 3 Opus supports reasoning
    if id.contains("claude-3-opus") {
        return true;
    }

    // Current Claude naming uses the family before the major version (for
    // example claude-sonnet-5 and claude-opus-4-8).
    if id.contains("claude-sonnet-5")
        || id.contains("claude-opus-4")
        || id.contains("claude-fable-5")
        || id.contains("claude-haiku-4-5")
    {
        return true;
    }

    // Claude 4+ legacy naming (future-proofing)
    if id.contains("claude-4") || id.contains("claude-5") {
        return true;
    }

    false
}

/// Build the Bedrock Mantle endpoint URL for a given region.
/// The Mantle endpoint is OpenAI-compatible and supports API key authentication.
fn build_mantle_base_url(region: &str) -> String {
    format!("https://bedrock-mantle.{}.api.aws/v1", region)
}

/// Resolve environment variable references in a header value.
/// If the value matches `${VAR_NAME}`, resolve from env. Otherwise return as-is.
fn resolve_header_value(value: &str) -> String {
    let trimmed = value.trim();
    if trimmed.starts_with("${") && trimmed.ends_with('}') {
        let var_name = &trimmed[2..trimmed.len() - 1];
        std::env::var(var_name).unwrap_or_else(|_| value.to_string())
    } else {
        value.to_string()
    }
}

/// Authentication mode for Bedrock provider.
/// Supports either API key (bearer token) authentication via HTTP to the Bedrock Mantle endpoint,
/// or traditional AWS SDK authentication using the credential chain.
pub enum BedrockAuthMode {
    /// API key authentication via HTTP to Bedrock Mantle endpoint (OpenAI-compatible)
    ApiKey {
        /// HTTP client with connection pooling
        http_client: Client,
        /// Bearer token for Authorization header
        api_key: String,
        /// Base URL (e.g., "https://bedrock-mantle.us-east-1.api.aws/v1")
        base_url: String,
        /// Custom headers to include in requests
        custom_headers: HashMap<String, String>,
    },
    /// AWS SDK authentication using credential chain (environment variables, credentials file, IAM role)
    AwsSdk {
        /// AWS Bedrock Runtime client (inference)
        client: BedrockClient,
        /// AWS Bedrock control-plane client (ListFoundationModels)
        control_client: BedrockControlClient,
    },
}

/// AWS Bedrock provider client
/// Supports dual authentication: API key (bearer token) via HTTP or AWS SDK credentials.
/// When API key is configured, uses the OpenAI-compatible Bedrock Mantle endpoint.
/// When no API key is present, falls back to AWS SDK authentication.
pub struct BedrockProvider {
    /// Provider name for identification
    name: String,
    /// AWS region (used in tests for assertions)
    #[allow(dead_code)]
    region: String,
    /// Authentication mode (API key or AWS SDK)
    auth_mode: BedrockAuthMode,
}

impl BedrockProvider {
    /// Create a new Bedrock provider client using AWS SDK authentication.
    /// This is the backward-compatible constructor that uses the AWS credential chain.
    pub async fn new(name: String, region: String) -> Result<Self, GatewayError> {
        Self::new_with_config(name, region, None, None, None, HashMap::new()).await
    }

    /// Create a new Bedrock provider client with full configuration options.
    ///
    /// If `api_key` is provided, uses HTTP-based authentication to the Bedrock Mantle endpoint.
    /// Otherwise, falls back to AWS SDK authentication using the credential chain.
    ///
    /// # Arguments
    /// * `name` - Provider name for identification
    /// * `region` - AWS region (e.g., "us-east-1")
    /// * `api_key` - Optional API key for bearer token authentication
    /// * `max_connections` - Optional max connections for HTTP client pool (default: 100)
    /// * `timeout_seconds` - Optional request timeout in seconds (default: 30)
    /// * `custom_headers` - Custom headers to include in requests (supports ${ENV_VAR} syntax)
    pub async fn new_with_config(
        name: String,
        region: String,
        api_key: Option<String>,
        max_connections: Option<u32>,
        timeout_seconds: Option<u64>,
        custom_headers: HashMap<String, String>,
    ) -> Result<Self, GatewayError> {
        let auth_mode = if let Some(key) = api_key {
            // API key mode: create HTTP client for Bedrock Mantle endpoint
            let pool_size = max_connections.unwrap_or(100) as usize;
            let timeout = std::time::Duration::from_secs(timeout_seconds.unwrap_or(30));

            let http_client = Client::builder()
                .pool_max_idle_per_host(pool_size)
                .pool_idle_timeout(std::time::Duration::from_secs(
                    DEFAULT_POOL_IDLE_TIMEOUT_SECS,
                ))
                .timeout(timeout)
                .build()
                .map_err(|e| {
                    GatewayError::Configuration(format!("Failed to create HTTP client: {}", e))
                })?;

            let base_url = build_mantle_base_url(&region);

            BedrockAuthMode::ApiKey {
                http_client,
                api_key: key,
                base_url,
                custom_headers,
            }
        } else {
            // AWS SDK mode: use credential chain
            let config = aws_config::defaults(BehaviorVersion::latest())
                .region(aws_config::Region::new(region.clone()))
                .load()
                .await;

            let client = BedrockClient::new(&config);
            let control_client = BedrockControlClient::new(&config);
            BedrockAuthMode::AwsSdk {
                client,
                control_client,
            }
        };

        Ok(Self {
            name,
            region,
            auth_mode,
        })
    }

    /// Get a reference to the AWS SDK client if using SDK authentication mode.
    /// Returns None if using API key authentication.
    #[allow(dead_code)] // used in tests
    fn get_sdk_client(&self) -> Option<&BedrockClient> {
        match &self.auth_mode {
            BedrockAuthMode::AwsSdk { client, .. } => Some(client),
            BedrockAuthMode::ApiKey { .. } => None,
        }
    }

    /// Check if this provider is using API key authentication mode.
    #[allow(dead_code)]
    pub fn is_api_key_mode(&self) -> bool {
        matches!(&self.auth_mode, BedrockAuthMode::ApiKey { .. })
    }

    /// Translate OpenAI model name to Bedrock model ID
    fn translate_model_id(&self, openai_model: &str) -> String {
        // Support common model name patterns
        match openai_model {
            // Current Claude families
            m if m.contains("claude-sonnet-5") => "anthropic.claude-sonnet-5".to_string(),
            m if m.contains("claude-opus-4-8") => "anthropic.claude-opus-4-8".to_string(),
            m if m.contains("claude-opus-4-7") => "anthropic.claude-opus-4-7".to_string(),
            m if m.contains("claude-haiku-4-5") => {
                "anthropic.claude-haiku-4-5-20251001-v1:0".to_string()
            }
            // Claude 3 models
            m if m.contains("claude-3-opus") => "anthropic.claude-3-opus-20240229-v1:0".to_string(),
            m if m.contains("claude-3-sonnet") => {
                "anthropic.claude-3-sonnet-20240229-v1:0".to_string()
            }
            m if m.contains("claude-3-haiku") => {
                "anthropic.claude-3-haiku-20240307-v1:0".to_string()
            }
            m if m.contains("claude-2.1") => "anthropic.claude-v2:1".to_string(),
            m if m.contains("claude-2") => "anthropic.claude-v2".to_string(),
            m if m.contains("claude-instant") => "anthropic.claude-instant-v1".to_string(),

            // Titan models
            m if m.contains("titan-text-express") => "amazon.titan-text-express-v1".to_string(),
            m if m.contains("titan-text-lite") => "amazon.titan-text-lite-v1".to_string(),
            m if m.contains("titan-embed") => "amazon.titan-embed-text-v1".to_string(),

            // Jurassic models
            m if m.contains("jurassic-2-ultra") => "ai21.j2-ultra-v1".to_string(),
            m if m.contains("jurassic-2-mid") => "ai21.j2-mid-v1".to_string(),

            // Command models (Cohere)
            m if m.contains("command-text") => "cohere.command-text-v14".to_string(),
            m if m.contains("command-light") => "cohere.command-light-text-v14".to_string(),

            // If already in ARN format, use as-is
            _ => openai_model.to_string(),
        }
    }

    /// Translate OpenAI request to Bedrock format
    fn translate_request(
        &self,
        request: &OpenAIRequest,
        model_id: &str,
    ) -> Result<String, GatewayError> {
        // Determine model family from model_id
        if model_id.starts_with("anthropic.claude") {
            self.translate_claude_request(request)
        } else if model_id.starts_with("amazon.titan") {
            self.translate_titan_request(request)
        } else if model_id.starts_with("ai21.j2") {
            self.translate_jurassic_request(request)
        } else if model_id.starts_with("cohere.command") {
            self.translate_command_request(request)
        } else {
            Err(GatewayError::Configuration(format!(
                "Unsupported Bedrock model: {}",
                model_id
            )))
        }
    }

    /// Convert OpenAI messages into the Bedrock Converse message and system
    /// shapes. Converse supports a common schema across current Bedrock models,
    /// avoiding the legacy per-provider InvokeModel translators.
    fn build_converse_input(
        &self,
        request: &OpenAIRequest,
    ) -> Result<
        (
            Vec<BedrockMessage>,
            Vec<SystemContentBlock>,
            InferenceConfiguration,
        ),
        GatewayError,
    > {
        let mut messages = Vec::new();
        let mut system = Vec::new();

        for message in &request.messages {
            if message.role == "system" {
                system.push(SystemContentBlock::Text(message.content_as_text()));
                continue;
            }

            let role = match message.role.as_str() {
                "assistant" => ConversationRole::Assistant,
                "user" => ConversationRole::User,
                other => {
                    return Err(GatewayError::Configuration(format!(
                        "Unsupported Converse message role: {}",
                        other
                    )))
                }
            };
            let built = BedrockMessage::builder()
                .role(role)
                .content(ContentBlock::Text(message.content_as_text()))
                .build()
                .map_err(|error| {
                    GatewayError::Configuration(format!(
                        "Failed to build Bedrock Converse message: {}",
                        error
                    ))
                })?;
            messages.push(built);
        }

        if messages.is_empty() {
            return Err(GatewayError::Configuration(
                "Bedrock Converse requires at least one user or assistant message".to_string(),
            ));
        }

        let max_tokens = request.max_tokens.unwrap_or(2048).min(i32::MAX as u32) as i32;
        let inference = InferenceConfiguration::builder()
            .max_tokens(max_tokens)
            .set_temperature(request.temperature)
            .build();

        Ok((messages, system, inference))
    }

    fn translate_converse_output(
        &self,
        output: SdkConverseOutput,
        original_model: &str,
    ) -> Result<OpenAIResponse, GatewayError> {
        let message = output
            .output()
            .and_then(|value| value.as_message().ok())
            .ok_or_else(|| GatewayError::Provider {
                provider: self.name.clone(),
                message: "Bedrock Converse response did not contain a message".to_string(),
                status_code: None,
            })?;

        let content = message
            .content()
            .iter()
            .filter_map(|block| block.as_text().ok())
            .cloned()
            .collect::<Vec<_>>()
            .join("");
        let usage = output.usage();
        let finish_reason = match output.stop_reason().as_str() {
            "max_tokens" => "length",
            "end_turn" | "stop_sequence" => "stop",
            "tool_use" => "tool_calls",
            other => other,
        };

        Ok(OpenAIResponse {
            id: format!("chatcmpl-{}", uuid::Uuid::new_v4()),
            object: "chat.completion".to_string(),
            created: chrono::Utc::now().timestamp(),
            model: original_model.to_string(),
            choices: vec![Choice {
                index: 0,
                message: Message {
                    role: "assistant".to_string(),
                    content: serde_json::Value::String(content),
                    extra: Default::default(),
                },
                finish_reason: Some(finish_reason.to_string()),
                extra: Default::default(),
            }],
            usage: Usage {
                prompt_tokens: usage.map_or(0, |value| value.input_tokens().max(0) as u32),
                completion_tokens: usage.map_or(0, |value| value.output_tokens().max(0) as u32),
                total_tokens: usage.map_or(0, |value| value.total_tokens().max(0) as u32),
                extra: Default::default(),
            },
            extra: Default::default(),
        })
    }

    /// Translate OpenAI request to Claude format
    fn translate_claude_request(&self, request: &OpenAIRequest) -> Result<String, GatewayError> {
        #[derive(Serialize)]
        struct ClaudeRequest {
            prompt: String,
            max_tokens_to_sample: u32,
            #[serde(skip_serializing_if = "Option::is_none")]
            temperature: Option<f32>,
            #[serde(skip_serializing_if = "Option::is_none")]
            stop_sequences: Option<Vec<String>>,
        }

        // Convert messages to Claude prompt format
        let mut prompt = String::new();
        for msg in &request.messages {
            match msg.role.as_str() {
                "system" => prompt.push_str(&format!("\n\nSystem: {}", msg.content_as_text())),
                "user" => prompt.push_str(&format!("\n\nHuman: {}", msg.content_as_text())),
                "assistant" => {
                    prompt.push_str(&format!("\n\nAssistant: {}", msg.content_as_text()))
                }
                _ => {}
            }
        }
        prompt.push_str("\n\nAssistant:");

        let claude_req = ClaudeRequest {
            prompt,
            max_tokens_to_sample: request.max_tokens.unwrap_or(2048),
            temperature: request.temperature,
            stop_sequences: None,
        };

        serde_json::to_string(&claude_req).map_err(|e| GatewayError::Serialization(e))
    }

    /// Translate OpenAI request to Titan format
    fn translate_titan_request(&self, request: &OpenAIRequest) -> Result<String, GatewayError> {
        #[derive(Serialize)]
        struct TitanRequest {
            #[serde(rename = "inputText")]
            input_text: String,
            #[serde(rename = "textGenerationConfig")]
            text_generation_config: TitanConfig,
        }

        #[derive(Serialize)]
        struct TitanConfig {
            #[serde(rename = "maxTokenCount")]
            max_token_count: u32,
            #[serde(skip_serializing_if = "Option::is_none")]
            temperature: Option<f32>,
        }

        // Combine messages into single input text
        let input_text = request
            .messages
            .iter()
            .map(|m| format!("{}: {}", m.role, m.content_as_text()))
            .collect::<Vec<_>>()
            .join("\n");

        let titan_req = TitanRequest {
            input_text,
            text_generation_config: TitanConfig {
                max_token_count: request.max_tokens.unwrap_or(2048),
                temperature: request.temperature,
            },
        };

        serde_json::to_string(&titan_req).map_err(|e| GatewayError::Serialization(e))
    }

    /// Translate OpenAI request to Jurassic format
    fn translate_jurassic_request(&self, request: &OpenAIRequest) -> Result<String, GatewayError> {
        #[derive(Serialize)]
        struct JurassicRequest {
            prompt: String,
            #[serde(rename = "maxTokens")]
            max_tokens: u32,
            #[serde(skip_serializing_if = "Option::is_none")]
            temperature: Option<f32>,
        }

        let prompt = request
            .messages
            .iter()
            .map(|m| format!("{}: {}", m.role, m.content_as_text()))
            .collect::<Vec<_>>()
            .join("\n");

        let jurassic_req = JurassicRequest {
            prompt,
            max_tokens: request.max_tokens.unwrap_or(2048),
            temperature: request.temperature,
        };

        serde_json::to_string(&jurassic_req).map_err(|e| GatewayError::Serialization(e))
    }

    /// Translate OpenAI request to Command format
    fn translate_command_request(&self, request: &OpenAIRequest) -> Result<String, GatewayError> {
        #[derive(Serialize)]
        struct CommandRequest {
            prompt: String,
            #[serde(rename = "max_tokens")]
            max_tokens: u32,
            #[serde(skip_serializing_if = "Option::is_none")]
            temperature: Option<f32>,
        }

        let prompt = request
            .messages
            .iter()
            .map(|m| m.content_as_text())
            .collect::<Vec<_>>()
            .join("\n");

        let command_req = CommandRequest {
            prompt,
            max_tokens: request.max_tokens.unwrap_or(2048),
            temperature: request.temperature,
        };

        serde_json::to_string(&command_req).map_err(|e| GatewayError::Serialization(e))
    }

    /// Translate Bedrock response to OpenAI format
    fn translate_response(
        &self,
        output: InvokeModelOutput,
        model_id: &str,
        original_model: &str,
    ) -> Result<OpenAIResponse, GatewayError> {
        let body = output.body().as_ref();
        let response_text = String::from_utf8_lossy(body);

        if model_id.starts_with("anthropic.claude") {
            self.translate_claude_response(&response_text, original_model)
        } else if model_id.starts_with("amazon.titan") {
            self.translate_titan_response(&response_text, original_model)
        } else if model_id.starts_with("ai21.j2") {
            self.translate_jurassic_response(&response_text, original_model)
        } else if model_id.starts_with("cohere.command") {
            self.translate_command_response(&response_text, original_model)
        } else {
            Err(GatewayError::Configuration(format!(
                "Unsupported Bedrock model: {}",
                model_id
            )))
        }
    }

    /// Translate Claude response to OpenAI format
    fn translate_claude_response(
        &self,
        response_text: &str,
        model: &str,
    ) -> Result<OpenAIResponse, GatewayError> {
        #[derive(Deserialize)]
        struct ClaudeResponse {
            completion: String,
            stop_reason: Option<String>,
        }

        let claude_resp: ClaudeResponse =
            serde_json::from_str(response_text).map_err(|e| GatewayError::Provider {
                provider: self.name.clone(),
                message: format!("Failed to parse Claude response: {}", e),
                status_code: None,
            })?;

        Ok(OpenAIResponse {
            id: format!("chatcmpl-{}", uuid::Uuid::new_v4()),
            object: "chat.completion".to_string(),
            created: chrono::Utc::now().timestamp(),
            model: model.to_string(),
            choices: vec![Choice {
                index: 0,
                message: Message {
                    role: "assistant".to_string(),
                    content: serde_json::Value::String(claude_resp.completion),
                    extra: Default::default(),
                },
                finish_reason: claude_resp.stop_reason,
                extra: Default::default(),
            }],
            usage: Usage {
                prompt_tokens: 0,
                completion_tokens: 0,
                total_tokens: 0,
                extra: Default::default(),
            },
            extra: Default::default(),
        })
    }

    /// Translate Titan response to OpenAI format
    fn translate_titan_response(
        &self,
        response_text: &str,
        model: &str,
    ) -> Result<OpenAIResponse, GatewayError> {
        #[derive(Deserialize)]
        struct TitanResponse {
            results: Vec<TitanResult>,
        }

        #[derive(Deserialize)]
        struct TitanResult {
            #[serde(rename = "outputText")]
            output_text: String,
        }

        let titan_resp: TitanResponse =
            serde_json::from_str(response_text).map_err(|e| GatewayError::Provider {
                provider: self.name.clone(),
                message: format!("Failed to parse Titan response: {}", e),
                status_code: None,
            })?;

        let content = titan_resp
            .results
            .first()
            .map(|r| r.output_text.clone())
            .unwrap_or_default();

        Ok(OpenAIResponse {
            id: format!("chatcmpl-{}", uuid::Uuid::new_v4()),
            object: "chat.completion".to_string(),
            created: chrono::Utc::now().timestamp(),
            model: model.to_string(),
            choices: vec![Choice {
                index: 0,
                message: Message {
                    role: "assistant".to_string(),
                    content: serde_json::Value::String(content),
                    extra: Default::default(),
                },
                finish_reason: Some("stop".to_string()),
                extra: Default::default(),
            }],
            usage: Usage {
                prompt_tokens: 0,
                completion_tokens: 0,
                total_tokens: 0,
                extra: Default::default(),
            },
            extra: Default::default(),
        })
    }

    /// Translate Jurassic response to OpenAI format
    fn translate_jurassic_response(
        &self,
        response_text: &str,
        model: &str,
    ) -> Result<OpenAIResponse, GatewayError> {
        #[derive(Deserialize)]
        struct JurassicResponse {
            completions: Vec<JurassicCompletion>,
        }

        #[derive(Deserialize)]
        struct JurassicCompletion {
            data: JurassicData,
        }

        #[derive(Deserialize)]
        struct JurassicData {
            text: String,
        }

        let jurassic_resp: JurassicResponse =
            serde_json::from_str(response_text).map_err(|e| GatewayError::Provider {
                provider: self.name.clone(),
                message: format!("Failed to parse Jurassic response: {}", e),
                status_code: None,
            })?;

        let content = jurassic_resp
            .completions
            .first()
            .map(|c| c.data.text.clone())
            .unwrap_or_default();

        Ok(OpenAIResponse {
            id: format!("chatcmpl-{}", uuid::Uuid::new_v4()),
            object: "chat.completion".to_string(),
            created: chrono::Utc::now().timestamp(),
            model: model.to_string(),
            choices: vec![Choice {
                index: 0,
                message: Message {
                    role: "assistant".to_string(),
                    content: serde_json::Value::String(content),
                    extra: Default::default(),
                },
                finish_reason: Some("stop".to_string()),
                extra: Default::default(),
            }],
            usage: Usage {
                prompt_tokens: 0,
                completion_tokens: 0,
                total_tokens: 0,
                extra: Default::default(),
            },
            extra: Default::default(),
        })
    }

    /// Translate Command response to OpenAI format
    fn translate_command_response(
        &self,
        response_text: &str,
        model: &str,
    ) -> Result<OpenAIResponse, GatewayError> {
        #[derive(Deserialize)]
        struct CommandResponse {
            generations: Vec<CommandGeneration>,
        }

        #[derive(Deserialize)]
        struct CommandGeneration {
            text: String,
        }

        let command_resp: CommandResponse =
            serde_json::from_str(response_text).map_err(|e| GatewayError::Provider {
                provider: self.name.clone(),
                message: format!("Failed to parse Command response: {}", e),
                status_code: None,
            })?;

        let content = command_resp
            .generations
            .first()
            .map(|g| g.text.clone())
            .unwrap_or_default();

        Ok(OpenAIResponse {
            id: format!("chatcmpl-{}", uuid::Uuid::new_v4()),
            object: "chat.completion".to_string(),
            created: chrono::Utc::now().timestamp(),
            model: model.to_string(),
            choices: vec![Choice {
                index: 0,
                message: Message {
                    role: "assistant".to_string(),
                    content: serde_json::Value::String(content),
                    extra: Default::default(),
                },
                finish_reason: Some("stop".to_string()),
                extra: Default::default(),
            }],
            usage: Usage {
                prompt_tokens: 0,
                completion_tokens: 0,
                total_tokens: 0,
                extra: Default::default(),
            },
            extra: Default::default(),
        })
    }

    /// Static fallback catalog used when the live Mantle `/v1/models` listing
    /// is unavailable or empty in API key mode. Contains only models confirmed
    /// OpenAI Chat Completions-compatible on the bedrock-mantle endpoint by AWS
    /// model cards. Mantle Responses-only (GPT-5.5/5.4/GPT-5.6) and Anthropic
    /// Messages-only (Claude) families are intentionally excluded until dedicated
    /// adapters are wired (see sync-bedrock-fallback verifier).
    pub fn mantle_chat_fallback_models() -> Vec<Model> {
        fallback_entries_to_models(BEDROCK_MANTLE_CHAT_FALLBACK)
    }

    pub fn mantle_responses_fallback_models() -> Vec<Model> {
        fallback_entries_to_models(BEDROCK_MANTLE_RESPONSES_FALLBACK)
    }

    pub fn mantle_messages_fallback_models() -> Vec<Model> {
        fallback_entries_to_models(BEDROCK_MANTLE_MESSAGES_FALLBACK)
    }

    /// Static fallback catalog used when `ListFoundationModels` is unavailable
    /// in AWS SDK mode. Contains only runtime IDs verified on AWS model cards
    /// as Converse/Invoke-capable. Legacy models and `meta.llama3-1-405b` are
    /// excluded. Will route through Converse once the SDK chat migration lands.
    pub fn runtime_fallback_models() -> Vec<Model> {
        fallback_entries_to_models(BEDROCK_RUNTIME_FALLBACK)
    }

    /// Query the OpenAI-compatible `/models` endpoint on the Bedrock Mantle
    /// endpoint (API key mode only). Returns the live list of available models.
    ///
    /// Queries both the standard path (`/v1/models` for GPT-OSS, Claude, etc.)
    /// and the OpenAI-specific path (`/openai/v1/models` for GPT-5.5, GPT-5.4)
    /// and merges the results.
    async fn list_models_api_key(
        &self,
        http_client: &Client,
        api_key: &str,
        base_url: &str,
        custom_headers: &HashMap<String, String>,
    ) -> Result<Vec<Model>, GatewayError> {
        let mut all_models: Vec<Model> = Vec::new();

        // 1) Standard Mantle path: /v1/models (GPT-OSS, open-weight models, etc.)
        let standard_url = format!("{}/models", base_url.trim_end_matches('/'));
        if let Ok(models) = self
            .fetch_models_from_url(http_client, api_key, &standard_url, custom_headers)
            .await
        {
            all_models.extend(models);
        }

        // 2) OpenAI-specific Mantle path: /openai/v1/models (GPT-5.5, GPT-5.4)
        // The base_url is typically https://bedrock-mantle.{region}.api.aws/v1
        // We need https://bedrock-mantle.{region}.api.aws/openai/v1/models
        let openai_base = base_url.trim_end_matches('/').trim_end_matches("/v1");
        let openai_url = format!("{}/openai/v1/models", openai_base);
        if let Ok(models) = self
            .fetch_models_from_url(http_client, api_key, &openai_url, custom_headers)
            .await
        {
            // Deduplicate by model ID
            let existing_ids: std::collections::HashSet<String> =
                all_models.iter().map(|m| m.id.clone()).collect();
            for m in models {
                if !existing_ids.contains(&m.id) {
                    all_models.push(m);
                }
            }
        }

        if all_models.is_empty() {
            return Err(GatewayError::Provider {
                provider: self.name.clone(),
                message: "Both Mantle /models endpoints returned empty".to_string(),
                status_code: None,
            });
        }

        Ok(all_models)
    }

    /// Fetch models from a single URL endpoint. Returns Ok(vec) or Err on failure.
    async fn fetch_models_from_url(
        &self,
        http_client: &Client,
        api_key: &str,
        url: &str,
        custom_headers: &HashMap<String, String>,
    ) -> Result<Vec<Model>, GatewayError> {
        #[derive(Deserialize)]
        struct ModelsResponse {
            #[serde(default)]
            data: Vec<Model>,
        }

        let mut req_builder = http_client
            .get(url)
            .header("Authorization", format!("Bearer {}", api_key));

        for (key, value) in custom_headers {
            let resolved = resolve_header_value(value);
            req_builder = req_builder.header(key.as_str(), resolved);
        }

        let response = req_builder
            .send()
            .await
            .map_err(|e| GatewayError::Network(format!("Request to {} failed: {}", url, e)))?;

        let status = response.status();
        if !status.is_success() {
            let error_text = response
                .text()
                .await
                .unwrap_or_else(|_| "Unknown error".to_string());
            return Err(GatewayError::Provider {
                provider: self.name.clone(),
                message: format!("HTTP {}: {}", status.as_u16(), error_text),
                status_code: Some(status.as_u16()),
            });
        }

        let parsed: ModelsResponse = response.json().await.map_err(|e| {
            GatewayError::Network(format!("Failed to parse models response: {}", e))
        })?;

        Ok(parsed.data)
    }

    /// Perform chat completion using API key authentication via HTTP.
    /// Sends request to the Bedrock Mantle endpoint which is OpenAI-compatible.
    async fn chat_completion_api_key(
        &self,
        mut request: OpenAIRequest,
        http_client: &Client,
        api_key: &str,
        base_url: &str,
        custom_headers: &HashMap<String, String>,
    ) -> Result<ProviderResponse, GatewayError> {
        let start = Instant::now();
        let url = format!("{}/chat/completions", base_url);
        let normalized = normalize_mantle_chat_messages(&mut request);
        let stripped = sanitize_mantle_chat_request(&mut request);
        if normalized > 0 {
            tracing::debug!(
                provider = %self.name,
                model = %request.model,
                fields_normalized = normalized,
                "Normalized Bedrock Mantle Chat Completions messages"
            );
        }
        if stripped > 0 {
            tracing::debug!(
                provider = %self.name,
                model = %request.model,
                fields_removed = stripped,
                "Sanitized Bedrock Mantle Chat Completions request"
            );
        }

        // Build request with Bearer token and custom headers
        let mut req_builder = http_client
            .post(&url)
            .header("Authorization", format!("Bearer {}", api_key))
            .header("Content-Type", "application/json");

        // Apply custom headers with environment variable resolution
        for (key, value) in custom_headers {
            let resolved = resolve_header_value(value);
            req_builder = req_builder.header(key.as_str(), resolved);
        }

        // Send request (OpenAI format - no translation needed for Mantle endpoint)
        let response = req_builder
            .json(&request)
            .send()
            .await
            .map_err(|e| GatewayError::Network(format!("Request to {} failed: {}", url, e)))?;

        let status = response.status();
        let latency_ms = start.elapsed().as_millis() as u64;

        if !status.is_success() {
            let error_text = response
                .text()
                .await
                .unwrap_or_else(|_| "Unknown error".to_string());

            // Handle authentication failures specifically
            if status.as_u16() == 401 || status.as_u16() == 403 {
                return Err(GatewayError::Provider {
                    provider: self.name.clone(),
                    message: format!(
                        "Bedrock API key authentication failed: HTTP {}: {}",
                        status.as_u16(),
                        error_text
                    ),
                    status_code: Some(status.as_u16()),
                });
            }

            return Err(GatewayError::Provider {
                provider: self.name.clone(),
                message: format!("HTTP {}: {}", status.as_u16(), error_text),
                status_code: Some(status.as_u16()),
            });
        }

        // Parse response as OpenAI format (no translation needed)
        let openai_response: OpenAIResponse = response
            .json()
            .await
            .map_err(|e| GatewayError::Network(format!("Failed to parse response: {}", e)))?;

        Ok(ProviderResponse {
            response: openai_response,
            provider_name: self.name.clone(),
            latency_ms,
        })
    }

    async fn chat_completion_responses_api(
        &self,
        request: OpenAIRequest,
        http_client: &Client,
        api_key: &str,
        base_url: &str,
        custom_headers: &HashMap<String, String>,
    ) -> Result<ProviderResponse, GatewayError> {
        let start = Instant::now();
        let root = base_url.trim_end_matches('/').trim_end_matches("/v1");
        let url = format!("{}/openai/v1/responses", root);
        let OpenAIRequest {
            model,
            messages,
            temperature,
            max_tokens,
            extra,
            ..
        } = request;
        let input = messages
            .iter()
            .map(|message| {
                serde_json::json!({
                    "role": message.role,
                    "content": message.content_as_text()
                })
            })
            .collect::<Vec<_>>();
        let mut body = serde_json::json!({
            "model": model.clone(),
            "input": input,
            "max_output_tokens": max_tokens.unwrap_or(2048),
            "stream": false
        });
        if let Some(temperature) = temperature {
            body["temperature"] = serde_json::json!(temperature);
        }
        for key in [
            "tools",
            "tool_choice",
            "parallel_tool_calls",
            "reasoning",
            "metadata",
            "service_tier",
            "store",
            "truncation",
        ] {
            if let Some(value) = extra.get(key) {
                body[key] = value.clone();
            }
        }
        let value = self
            .post_mantle_json(http_client, api_key, &url, custom_headers, &body)
            .await?;
        let content = value
            .get("output_text")
            .and_then(|value| value.as_str())
            .map(str::to_string)
            .or_else(|| {
                value
                    .get("output")
                    .and_then(|value| value.as_array())
                    .into_iter()
                    .flatten()
                    .flat_map(|item| item.get("content").and_then(|value| value.as_array()))
                    .flatten()
                    .find_map(|item| {
                        item.get("text")
                            .and_then(|value| value.as_str())
                            .map(str::to_string)
                    })
            })
            .unwrap_or_default();
        let usage = value.get("usage");
        Ok(ProviderResponse {
            response: self.openai_response_from_text(
                &model,
                content,
                usage
                    .and_then(|value| value.get("input_tokens"))
                    .and_then(|value| value.as_u64())
                    .unwrap_or(0) as u32,
                usage
                    .and_then(|value| value.get("output_tokens"))
                    .and_then(|value| value.as_u64())
                    .unwrap_or(0) as u32,
                "stop",
            ),
            provider_name: self.name.clone(),
            latency_ms: start.elapsed().as_millis() as u64,
        })
    }

    async fn chat_completion_messages_api(
        &self,
        request: OpenAIRequest,
        http_client: &Client,
        api_key: &str,
        base_url: &str,
        custom_headers: &HashMap<String, String>,
    ) -> Result<ProviderResponse, GatewayError> {
        let start = Instant::now();
        let root = base_url.trim_end_matches('/').trim_end_matches("/v1");
        let url = format!("{}/v1/messages", root);
        let system = request
            .messages
            .iter()
            .filter(|message| message.role == "system")
            .map(|message| message.content_as_text())
            .collect::<Vec<_>>()
            .join("\n");
        let messages = request
            .messages
            .iter()
            .filter(|message| message.role != "system")
            .map(|message| {
                serde_json::json!({
                    "role": message.role,
                    "content": message.content_as_text()
                })
            })
            .collect::<Vec<_>>();
        let mut body = serde_json::json!({
            "model": request.model,
            "messages": messages,
            "max_tokens": request.max_tokens.unwrap_or(2048),
            "stream": false
        });
        if !system.is_empty() {
            body["system"] = serde_json::json!(system);
        }
        if let Some(temperature) = request.temperature {
            body["temperature"] = serde_json::json!(temperature);
        }
        let value = self
            .post_mantle_json(http_client, api_key, &url, custom_headers, &body)
            .await?;
        let content = value
            .get("content")
            .and_then(|value| value.as_array())
            .into_iter()
            .flatten()
            .filter_map(|item| item.get("text").and_then(|value| value.as_str()))
            .collect::<Vec<_>>()
            .join("");
        let usage = value.get("usage");
        let stop_reason = value
            .get("stop_reason")
            .and_then(|value| value.as_str())
            .unwrap_or("end_turn");
        Ok(ProviderResponse {
            response: self.openai_response_from_text(
                &request.model,
                content,
                usage
                    .and_then(|value| value.get("input_tokens"))
                    .and_then(|value| value.as_u64())
                    .unwrap_or(0) as u32,
                usage
                    .and_then(|value| value.get("output_tokens"))
                    .and_then(|value| value.as_u64())
                    .unwrap_or(0) as u32,
                if stop_reason == "max_tokens" {
                    "length"
                } else {
                    "stop"
                },
            ),
            provider_name: self.name.clone(),
            latency_ms: start.elapsed().as_millis() as u64,
        })
    }

    async fn post_mantle_json(
        &self,
        http_client: &Client,
        api_key: &str,
        url: &str,
        custom_headers: &HashMap<String, String>,
        body: &serde_json::Value,
    ) -> Result<serde_json::Value, GatewayError> {
        let mut request = http_client
            .post(url)
            .header("Authorization", format!("Bearer {}", api_key))
            .header("Content-Type", "application/json")
            .json(body);
        for (key, value) in custom_headers {
            request = request.header(key.as_str(), resolve_header_value(value));
        }
        let response = request.send().await.map_err(|error| {
            GatewayError::Network(format!("Request to {} failed: {}", url, error))
        })?;
        let status = response.status();
        if !status.is_success() {
            let text = response.text().await.unwrap_or_default();
            return Err(GatewayError::Provider {
                provider: self.name.clone(),
                message: format!("HTTP {}: {}", status.as_u16(), text),
                status_code: Some(status.as_u16()),
            });
        }
        response.json().await.map_err(|error| {
            GatewayError::Network(format!("Failed to parse response from {}: {}", url, error))
        })
    }

    fn openai_response_from_text(
        &self,
        model: &str,
        content: String,
        prompt_tokens: u32,
        completion_tokens: u32,
        finish_reason: &str,
    ) -> OpenAIResponse {
        OpenAIResponse {
            id: format!("chatcmpl-{}", uuid::Uuid::new_v4()),
            object: "chat.completion".to_string(),
            created: chrono::Utc::now().timestamp(),
            model: model.to_string(),
            choices: vec![Choice {
                index: 0,
                message: Message {
                    role: "assistant".to_string(),
                    content: serde_json::Value::String(content),
                    extra: Default::default(),
                },
                finish_reason: Some(finish_reason.to_string()),
                extra: Default::default(),
            }],
            usage: Usage {
                prompt_tokens,
                completion_tokens,
                total_tokens: prompt_tokens.saturating_add(completion_tokens),
                extra: Default::default(),
            },
            extra: Default::default(),
        }
    }

    /// Perform streaming chat completion using API key authentication via HTTP.
    /// Sends request to the Bedrock Mantle endpoint which is OpenAI-compatible.
    /// Returns a stream of SSE events.
    async fn chat_completion_stream_api_key(
        &self,
        request: OpenAIRequest,
        http_client: &Client,
        api_key: &str,
        base_url: &str,
        custom_headers: &HashMap<String, String>,
    ) -> Result<Pin<Box<dyn Stream<Item = Result<SSEEvent, GatewayError>> + Send>>, GatewayError>
    {
        let url = format!("{}/chat/completions", base_url);

        // Build request with Bearer token and custom headers
        let mut req_builder = http_client
            .post(&url)
            .header("Authorization", format!("Bearer {}", api_key))
            .header("Content-Type", "application/json");

        // Apply custom headers with environment variable resolution
        for (key, value) in custom_headers {
            let resolved = resolve_header_value(value);
            req_builder = req_builder.header(key.as_str(), resolved);
        }

        // Send request (OpenAI format - no translation needed for Mantle endpoint)
        let response = req_builder
            .json(&request)
            .send()
            .await
            .map_err(|e| GatewayError::Network(format!("Request to {} failed: {}", url, e)))?;

        let status = response.status();

        if !status.is_success() {
            let error_text = response
                .text()
                .await
                .unwrap_or_else(|_| "Unknown error".to_string());

            // Handle authentication failures specifically
            if status.as_u16() == 401 || status.as_u16() == 403 {
                return Err(GatewayError::Provider {
                    provider: self.name.clone(),
                    message: format!(
                        "Bedrock API key authentication failed: HTTP {}: {}",
                        status.as_u16(),
                        error_text
                    ),
                    status_code: Some(status.as_u16()),
                });
            }

            return Err(GatewayError::Provider {
                provider: self.name.clone(),
                message: format!("HTTP {}: {}", status.as_u16(), error_text),
                status_code: Some(status.as_u16()),
            });
        }

        // Get the byte stream from the response
        let mut stream = response.bytes_stream();
        let provider_name = self.name.clone();

        let sse_stream = async_stream::stream! {
            while let Some(chunk_result) = stream.next().await {
                match chunk_result {
                    Ok(bytes) => {
                        let text = String::from_utf8_lossy(&bytes);
                        for line in text.lines() {
                            let trimmed = line.trim();
                            if trimmed.is_empty() {
                                continue;
                            }
                            match parse_sse_chunk_api_key(trimmed, &provider_name) {
                                Ok(Some(event)) => yield Ok(event),
                                Ok(None) => {}
                                Err(e) => {
                                    yield Err(e);
                                    break;
                                }
                            }
                        }
                    }
                    Err(e) => {
                        yield Err(GatewayError::Network(format!("Stream error: {}", e)));
                        break;
                    }
                }
            }
        };

        Ok(Box::pin(sse_stream))
    }
}

#[async_trait]
impl ProviderClient for BedrockProvider {
    async fn chat_completion(
        &self,
        request: OpenAIRequest,
    ) -> Result<ProviderResponse, GatewayError> {
        match &self.auth_mode {
            BedrockAuthMode::ApiKey {
                http_client,
                api_key,
                base_url,
                custom_headers,
            } => match mantle_api_for_model(&request.model) {
                MantleApi::Chat => {
                    self.chat_completion_api_key(
                        request,
                        http_client,
                        api_key,
                        base_url,
                        custom_headers,
                    )
                    .await
                }
                MantleApi::Responses => {
                    self.chat_completion_responses_api(
                        request,
                        http_client,
                        api_key,
                        base_url,
                        custom_headers,
                    )
                    .await
                }
                MantleApi::Messages => {
                    self.chat_completion_messages_api(
                        request,
                        http_client,
                        api_key,
                        base_url,
                        custom_headers,
                    )
                    .await
                }
            },
            BedrockAuthMode::AwsSdk { client, .. } => {
                // AWS SDK mode: Converse provides one request/response schema
                // across current Bedrock model families. Always set max_tokens
                // explicitly to avoid reserving each model's maximum quota.
                let start = Instant::now();
                let model_id = self.translate_model_id(&request.model);
                let (messages, system, inference_config) = self.build_converse_input(&request)?;

                let output = client
                    .converse()
                    .model_id(&model_id)
                    .set_messages(Some(messages))
                    .set_system(if system.is_empty() {
                        None
                    } else {
                        Some(system)
                    })
                    .inference_config(inference_config)
                    .send()
                    .await
                    .map_err(|error| GatewayError::Provider {
                        provider: self.name.clone(),
                        message: format!("Bedrock Converse failed: {}", error),
                        status_code: None,
                    })?;

                let latency_ms = start.elapsed().as_millis() as u64;
                let response = self.translate_converse_output(output, &request.model)?;

                Ok(ProviderResponse {
                    response,
                    provider_name: self.name.clone(),
                    latency_ms,
                })
            }
        }
    }

    async fn chat_completion_stream(
        &self,
        request: OpenAIRequest,
    ) -> Result<Pin<Box<dyn Stream<Item = Result<SSEEvent, GatewayError>> + Send>>, GatewayError>
    {
        match &self.auth_mode {
            BedrockAuthMode::ApiKey {
                http_client,
                api_key,
                base_url,
                custom_headers,
            } => {
                if mantle_api_for_model(&request.model) == MantleApi::Chat {
                    let mut stream_request = request;
                    stream_request.stream = true;
                    self.chat_completion_stream_api_key(
                        stream_request,
                        http_client,
                        api_key,
                        base_url,
                        custom_headers,
                    )
                    .await
                } else {
                    // Responses and Messages require translation into the
                    // gateway's OpenAI response schema, so buffer the upstream
                    // response and emit one translated event for replay.
                    let provider_response = match mantle_api_for_model(&request.model) {
                        MantleApi::Responses => {
                            self.chat_completion_responses_api(
                                request,
                                http_client,
                                api_key,
                                base_url,
                                custom_headers,
                            )
                            .await?
                        }
                        MantleApi::Messages => {
                            self.chat_completion_messages_api(
                                request,
                                http_client,
                                api_key,
                                base_url,
                                custom_headers,
                            )
                            .await?
                        }
                        MantleApi::Chat => unreachable!(),
                    };
                    let payload = serde_json::to_string(&provider_response.response)
                        .map_err(GatewayError::Serialization)?;
                    Ok(Box::pin(futures::stream::once(async move {
                        Ok(SSEEvent::new(payload))
                    })))
                }
            }
            BedrockAuthMode::AwsSdk { client, .. } => {
                // Bedrock-translated providers use the gateway's buffer-and-
                // replay path. Perform a non-streaming Converse request here and
                // expose one OpenAI-shaped SSE event; the handler re-chunks the
                // complete response for clients.
                let model_id = self.translate_model_id(&request.model);
                let (messages, system, inference_config) = self.build_converse_input(&request)?;
                let output = client
                    .converse()
                    .model_id(&model_id)
                    .set_messages(Some(messages))
                    .set_system(if system.is_empty() {
                        None
                    } else {
                        Some(system)
                    })
                    .inference_config(inference_config)
                    .send()
                    .await
                    .map_err(|error| GatewayError::Provider {
                        provider: self.name.clone(),
                        message: format!("Bedrock Converse failed: {}", error),
                        status_code: None,
                    })?;
                let response = self.translate_converse_output(output, &request.model)?;
                let provider_name = self.name.clone();
                let payload =
                    serde_json::to_string(&response).map_err(GatewayError::Serialization)?;
                let stream = futures::stream::once(async move {
                    if payload.is_empty() {
                        Err(GatewayError::Provider {
                            provider: provider_name,
                            message: "Bedrock Converse returned an empty response".to_string(),
                            status_code: None,
                        })
                    } else {
                        Ok(SSEEvent::new(payload))
                    }
                });
                Ok(Box::pin(stream))
            }
        }
    }

    async fn list_models(&self) -> Result<Vec<Model>, GatewayError> {
        let mut live_models: Vec<Model> = Vec::new();

        match &self.auth_mode {
            // API key mode: query the OpenAI-compatible Bedrock Mantle /models endpoint.
            BedrockAuthMode::ApiKey {
                http_client,
                api_key,
                base_url,
                custom_headers,
            } => {
                match self
                    .list_models_api_key(http_client, api_key, base_url, custom_headers)
                    .await
                {
                    Ok(models) => {
                        live_models = models;
                    }
                    Err(e) => {
                        tracing::warn!(
                            provider = %self.name,
                            error = %e,
                            "Bedrock Mantle /models listing failed, merging backup list only"
                        );
                    }
                }
            }
            // SDK mode: use the ListFoundationModels control-plane API.
            BedrockAuthMode::AwsSdk { control_client, .. } => {
                match control_client.list_foundation_models().send().await {
                    Ok(output) => {
                        let summaries = output.model_summaries();
                        live_models = summaries
                            .iter()
                            .map(|s| {
                                let id = s.model_id().to_string();
                                let owner = s.provider_name().unwrap_or("unknown").to_string();
                                let has_vision = id.contains("claude-3")
                                    || id.contains("nova-pro")
                                    || id.contains("nova-lite")
                                    || id.contains("gpt-oss");
                                Model {
                                    id,
                                    object: "model".to_string(),
                                    owned_by: owner,
                                    created: None,
                                    context_window: None,
                                    max_completion_tokens: None,
                                    supports_vision: has_vision,
                                }
                            })
                            .collect();
                    }
                    Err(e) => {
                        tracing::warn!(
                            provider = %self.name,
                            error = %e,
                            "Bedrock ListFoundationModels failed, merging backup list only"
                        );
                    }
                }
            }
        }

        // Merge the compatibility-correct fallback catalog for the active auth
        // mode so the gateway stays useful when the live listing is incomplete
        // or unavailable. API key mode merges the Mantle Chat catalog (only
        // Chat Completions-capable IDs); AWS SDK mode merges the runtime
        // catalog (Converse/Invoke-capable IDs). Existing live IDs win.
        let existing_ids: std::collections::HashSet<String> =
            live_models.iter().map(|m| m.id.clone()).collect();
        let fallback: Vec<Model> = match &self.auth_mode {
            BedrockAuthMode::ApiKey { .. } => {
                let mut models = Self::mantle_chat_fallback_models();
                models.extend(Self::mantle_responses_fallback_models());
                models.extend(Self::mantle_messages_fallback_models());
                models
            }
            BedrockAuthMode::AwsSdk { .. } => Self::runtime_fallback_models(),
        };
        for backup in fallback {
            if !existing_ids.contains(&backup.id) {
                live_models.push(backup);
            }
        }

        Ok(live_models)
    }

    fn provider_name(&self) -> &str {
        &self.name
    }
}

/// Parse Bedrock streaming chunk to SSE event
#[allow(dead_code)]
fn parse_bedrock_chunk(text: &str, model: &str) -> Result<SSEEvent, GatewayError> {
    // Convert Bedrock chunk to OpenAI SSE format
    #[derive(Serialize)]
    struct StreamChunk {
        id: String,
        object: String,
        created: i64,
        model: String,
        choices: Vec<StreamChoice>,
    }

    #[derive(Serialize)]
    struct StreamChoice {
        index: u32,
        delta: Delta,
        finish_reason: Option<String>,
    }

    #[derive(Serialize)]
    struct Delta {
        content: String,
    }

    let chunk = StreamChunk {
        id: format!("chatcmpl-{}", uuid::Uuid::new_v4()),
        object: "chat.completion.chunk".to_string(),
        created: chrono::Utc::now().timestamp(),
        model: model.to_string(),
        choices: vec![StreamChoice {
            index: 0,
            delta: Delta {
                content: text.to_string(),
            },
            finish_reason: None,
        }],
    };

    let json = serde_json::to_string(&chunk).map_err(|e| GatewayError::Serialization(e))?;

    Ok(SSEEvent::new(json))
}

/// Parse SSE chunk from API key mode (OpenAI-compatible format).
/// Returns None for empty lines or [DONE] terminator.
/// Returns Some(SSEEvent) for valid data lines.
fn parse_sse_chunk_api_key(
    text: &str,
    _provider_name: &str,
) -> Result<Option<SSEEvent>, GatewayError> {
    // SSE format: "data: {...}\n\n" or "data: [DONE]\n\n"
    for line in text.lines() {
        let line = line.trim();

        // Skip empty lines
        if line.is_empty() {
            continue;
        }

        // Handle data lines
        if let Some(data) = line.strip_prefix("data: ") {
            let data = data.trim();

            // Handle [DONE] terminator
            if data == "[DONE]" {
                return Ok(None);
            }

            // Return the JSON data as-is (already in OpenAI format)
            return Ok(Some(SSEEvent::new(data.to_string())));
        }
    }

    // No valid data found in this chunk
    Ok(None)
}

/// Merge two model ID lists into a deduplicated, lexicographically sorted union.
///
/// Used by the admin UI and backend to combine auto-discovered models with
/// manually specified models. Duplicates are removed and the result is sorted.
pub fn merge_model_lists(list_a: Vec<String>, list_b: Vec<String>) -> Vec<String> {
    let mut set: std::collections::BTreeSet<String> = list_a.into_iter().collect();
    set.extend(list_b);
    set.into_iter().collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Helper function to create a test BedrockProvider with AWS SDK auth mode
    fn create_test_provider(name: &str, region: &str) -> BedrockProvider {
        let client = BedrockClient::from_conf(
            aws_sdk_bedrockruntime::Config::builder()
                .behavior_version(aws_sdk_bedrockruntime::config::BehaviorVersion::latest())
                .build(),
        );
        let control_client = BedrockControlClient::from_conf(
            aws_sdk_bedrock::Config::builder()
                .behavior_version(aws_sdk_bedrock::config::BehaviorVersion::latest())
                .build(),
        );
        BedrockProvider {
            name: name.to_string(),
            region: region.to_string(),
            auth_mode: BedrockAuthMode::AwsSdk {
                client,
                control_client,
            },
        }
    }

    /// Helper function to create a test BedrockProvider with API key auth mode
    fn create_test_provider_with_api_key(
        name: &str,
        region: &str,
        api_key: &str,
    ) -> BedrockProvider {
        let http_client = Client::builder()
            .build()
            .expect("Failed to create HTTP client");

        BedrockProvider {
            name: name.to_string(),
            region: region.to_string(),
            auth_mode: BedrockAuthMode::ApiKey {
                http_client,
                api_key: api_key.to_string(),
                base_url: build_mantle_base_url(region),
                custom_headers: HashMap::new(),
            },
        }
    }

    #[test]
    fn test_build_mantle_base_url() {
        assert_eq!(
            build_mantle_base_url("us-east-1"),
            "https://bedrock-mantle.us-east-1.api.aws/v1"
        );
        assert_eq!(
            build_mantle_base_url("us-west-2"),
            "https://bedrock-mantle.us-west-2.api.aws/v1"
        );
        assert_eq!(
            build_mantle_base_url("eu-west-1"),
            "https://bedrock-mantle.eu-west-1.api.aws/v1"
        );
    }

    #[test]
    fn test_build_mantle_base_url_additional_regions() {
        // Test additional AWS regions
        assert_eq!(
            build_mantle_base_url("ap-northeast-1"),
            "https://bedrock-mantle.ap-northeast-1.api.aws/v1"
        );
        assert_eq!(
            build_mantle_base_url("ap-southeast-2"),
            "https://bedrock-mantle.ap-southeast-2.api.aws/v1"
        );
        assert_eq!(
            build_mantle_base_url("eu-central-1"),
            "https://bedrock-mantle.eu-central-1.api.aws/v1"
        );
    }

    #[test]
    fn test_is_api_key_mode_with_api_key() {
        let provider = create_test_provider_with_api_key("test", "us-east-1", "test-api-key");
        assert!(provider.is_api_key_mode());
    }

    #[test]
    fn test_is_api_key_mode_with_sdk() {
        let provider = create_test_provider("test", "us-east-1");
        assert!(!provider.is_api_key_mode());
    }

    #[test]
    fn test_get_sdk_client_returns_none_for_api_key_mode() {
        let provider = create_test_provider_with_api_key("test", "us-east-1", "test-api-key");
        assert!(provider.get_sdk_client().is_none());
    }

    #[test]
    fn test_get_sdk_client_returns_some_for_sdk_mode() {
        let provider = create_test_provider("test", "us-east-1");
        assert!(provider.get_sdk_client().is_some());
    }

    #[tokio::test]
    async fn test_new_with_api_key_creates_http_mode() {
        let provider = BedrockProvider::new_with_config(
            "test-bedrock".to_string(),
            "us-east-1".to_string(),
            Some("test-api-key-12345".to_string()),
            Some(50),
            Some(60),
            HashMap::new(),
        )
        .await
        .expect("Failed to create provider");

        // Verify API key mode is selected
        assert!(provider.is_api_key_mode());
        assert!(provider.get_sdk_client().is_none());

        // Verify provider name and region are set correctly
        assert_eq!(provider.name, "test-bedrock");
        assert_eq!(provider.region, "us-east-1");

        // Verify base_url is constructed correctly
        match &provider.auth_mode {
            BedrockAuthMode::ApiKey {
                base_url, api_key, ..
            } => {
                assert_eq!(base_url, "https://bedrock-mantle.us-east-1.api.aws/v1");
                assert_eq!(api_key, "test-api-key-12345");
            }
            _ => panic!("Expected ApiKey auth mode"),
        }
    }

    #[tokio::test]
    async fn test_new_without_api_key_creates_sdk_mode() {
        let provider = BedrockProvider::new_with_config(
            "test-bedrock".to_string(),
            "us-west-2".to_string(),
            None, // No API key
            None,
            None,
            HashMap::new(),
        )
        .await
        .expect("Failed to create provider");

        // Verify SDK mode is selected
        assert!(!provider.is_api_key_mode());
        assert!(provider.get_sdk_client().is_some());

        // Verify provider name and region are set correctly
        assert_eq!(provider.name, "test-bedrock");
        assert_eq!(provider.region, "us-west-2");
    }

    #[tokio::test]
    async fn test_new_backward_compatible() {
        // Test the backward-compatible new() constructor
        let provider = BedrockProvider::new("test-bedrock".to_string(), "eu-west-1".to_string())
            .await
            .expect("Failed to create provider");

        // Should use SDK mode by default
        assert!(!provider.is_api_key_mode());
        assert!(provider.get_sdk_client().is_some());
        assert_eq!(provider.name, "test-bedrock");
        assert_eq!(provider.region, "eu-west-1");
    }

    #[test]
    fn test_resolve_header_value_env_var() {
        // Set a test environment variable
        std::env::set_var("TEST_BEDROCK_HEADER", "resolved-value");

        let result = resolve_header_value("${TEST_BEDROCK_HEADER}");
        assert_eq!(result, "resolved-value");

        // Clean up
        std::env::remove_var("TEST_BEDROCK_HEADER");
    }

    #[test]
    fn test_resolve_header_value_literal() {
        let result = resolve_header_value("literal-value");
        assert_eq!(result, "literal-value");
    }

    #[test]
    fn test_resolve_header_value_unset_env_var() {
        // Ensure the env var doesn't exist
        std::env::remove_var("NONEXISTENT_VAR_12345");

        let result = resolve_header_value("${NONEXISTENT_VAR_12345}");
        // Should return the original value when env var is not set
        assert_eq!(result, "${NONEXISTENT_VAR_12345}");
    }

    #[test]
    fn test_bedrock_regions_count() {
        assert_eq!(BEDROCK_REGIONS.len(), 13);
    }

    #[test]
    fn test_bedrock_regions_contains_expected() {
        assert!(BEDROCK_REGIONS.contains(&"us-east-1"));
        assert!(BEDROCK_REGIONS.contains(&"us-west-2"));
        assert!(BEDROCK_REGIONS.contains(&"eu-west-1"));
        assert!(BEDROCK_REGIONS.contains(&"us-gov-west-1"));
        assert!(BEDROCK_REGIONS.contains(&"sa-east-1"));
        assert!(BEDROCK_REGIONS.contains(&"ca-central-1"));
    }

    #[test]
    fn test_derive_region_group() {
        assert_eq!(derive_region_group("us-east-1"), "us");
        assert_eq!(derive_region_group("us-west-2"), "us");
        assert_eq!(derive_region_group("us-gov-west-1"), "us");
        assert_eq!(derive_region_group("eu-west-1"), "eu");
        assert_eq!(derive_region_group("eu-west-3"), "eu");
        assert_eq!(derive_region_group("eu-central-1"), "eu");
        assert_eq!(derive_region_group("ap-northeast-1"), "ap");
        assert_eq!(derive_region_group("ap-southeast-1"), "ap");
        assert_eq!(derive_region_group("ap-south-1"), "ap");
        assert_eq!(derive_region_group("sa-east-1"), "sa");
        assert_eq!(derive_region_group("ca-central-1"), "ca");
        assert_eq!(derive_region_group("unknown-region"), "");
        assert_eq!(derive_region_group(""), "");
    }

    #[test]
    fn test_model_supports_reasoning_sonnet_v2() {
        // Claude 3.5 Sonnet v2 should support reasoning
        assert!(model_supports_reasoning(
            "anthropic.claude-3-5-sonnet-20241022-v2:0"
        ));
        assert!(model_supports_reasoning(
            "us.anthropic.claude-3-5-sonnet-20241022-v2:0"
        ));
    }

    #[test]
    fn test_model_supports_reasoning_sonnet_v1_no() {
        // Claude 3.5 Sonnet v1 should NOT support reasoning
        assert!(!model_supports_reasoning(
            "anthropic.claude-3-5-sonnet-20240620-v1:0"
        ));
    }

    #[test]
    fn test_model_supports_reasoning_opus() {
        // Claude 3 Opus supports reasoning
        assert!(model_supports_reasoning(
            "anthropic.claude-3-opus-20240229-v1:0"
        ));
        assert!(model_supports_reasoning(
            "us.anthropic.claude-3-opus-20240229-v1:0"
        ));
    }

    #[test]
    fn test_model_supports_reasoning_non_reasoning_models() {
        assert!(!model_supports_reasoning(
            "anthropic.claude-3-sonnet-20240229-v1:0"
        ));
        assert!(!model_supports_reasoning(
            "anthropic.claude-3-haiku-20240307-v1:0"
        ));
        assert!(!model_supports_reasoning("amazon.titan-text-express-v1"));
        assert!(!model_supports_reasoning("cohere.command-r-plus-v1:0"));
        assert!(!model_supports_reasoning("meta.llama3-1-70b-instruct-v1:0"));
        assert!(!model_supports_reasoning(""));
    }

    #[test]
    fn test_model_supports_reasoning_future_claude4() {
        assert!(model_supports_reasoning("anthropic.claude-4-sonnet-v1:0"));
    }

    #[test]
    fn test_translate_model_id_claude() {
        let provider = create_test_provider("test", "us-east-1");

        assert_eq!(
            provider.translate_model_id("claude-3-opus"),
            "anthropic.claude-3-opus-20240229-v1:0"
        );
        assert_eq!(
            provider.translate_model_id("claude-3-sonnet"),
            "anthropic.claude-3-sonnet-20240229-v1:0"
        );
    }

    #[test]
    fn test_translate_model_id_titan() {
        let provider = create_test_provider("test", "us-east-1");

        assert_eq!(
            provider.translate_model_id("titan-text-express"),
            "amazon.titan-text-express-v1"
        );
    }

    #[test]
    fn test_parse_sse_chunk_api_key_valid_data() {
        let chunk = "data: {\"id\":\"test\",\"choices\":[{\"delta\":{\"content\":\"hello\"}}]}\n\n";
        let result = parse_sse_chunk_api_key(chunk, "test-provider");

        assert!(result.is_ok());
        let event = result.unwrap();
        assert!(event.is_some());
    }

    #[test]
    fn test_parse_sse_chunk_api_key_done() {
        let chunk = "data: [DONE]\n\n";
        let result = parse_sse_chunk_api_key(chunk, "test-provider");

        assert!(result.is_ok());
        let event = result.unwrap();
        assert!(event.is_none()); // [DONE] should return None
    }

    #[test]
    fn test_parse_sse_chunk_api_key_empty() {
        let chunk = "\n\n";
        let result = parse_sse_chunk_api_key(chunk, "test-provider");

        assert!(result.is_ok());
        let event = result.unwrap();
        assert!(event.is_none()); // Empty chunk should return None
    }

    fn create_api_key_mode_provider_for_base_url(
        name: &str,
        base_url: String,
        api_key: &str,
    ) -> BedrockProvider {
        let http_client = Client::builder()
            .build()
            .expect("Failed to create HTTP client");

        BedrockProvider {
            name: name.to_string(),
            region: "us-east-1".to_string(),
            auth_mode: BedrockAuthMode::ApiKey {
                http_client,
                api_key: api_key.to_string(),
                base_url,
                custom_headers: HashMap::new(),
            },
        }
    }

    fn create_test_chat_request(stream: bool) -> OpenAIRequest {
        OpenAIRequest {
            model: "gpt-test".to_string(),
            messages: vec![Message {
                role: "user".to_string(),
                content: serde_json::Value::String("hello".to_string()),
                extra: Default::default(),
            }],
            stream,
            temperature: None,
            max_tokens: None,
            extra: Default::default(),
        }
    }

    #[tokio::test]
    async fn test_api_key_chat_completion_success() {
        use wiremock::matchers::{header, method, path};
        use wiremock::{Mock, MockServer, ResponseTemplate};

        let mock_server = MockServer::start().await;

        Mock::given(method("POST"))
            .and(path("/chat/completions"))
            .and(header("Authorization", "Bearer test-api-key"))
            .respond_with(ResponseTemplate::new(200).set_body_json(serde_json::json!({
                "id": "chatcmpl-test",
                "object": "chat.completion",
                "created": 1234567890i64,
                "model": "gpt-test",
                "choices": [{
                    "index": 0,
                    "message": {"role": "assistant", "content": "hi from bedrock"},
                    "finish_reason": "stop"
                }],
                "usage": {"prompt_tokens": 1, "completion_tokens": 2, "total_tokens": 3}
            })))
            .expect(1)
            .mount(&mock_server)
            .await;

        let provider = create_api_key_mode_provider_for_base_url(
            "bedrock-test",
            mock_server.uri(),
            "test-api-key",
        );

        let result = provider
            .chat_completion(create_test_chat_request(false))
            .await;
        assert!(result.is_ok(), "API key mode request should succeed");

        let response = result.unwrap();
        assert_eq!(response.provider_name, "bedrock-test");
        assert_eq!(
            response.response.choices[0].message.content_as_text(),
            "hi from bedrock"
        );
    }

    #[tokio::test]
    async fn test_api_key_auth_failure_401() {
        use wiremock::matchers::{header, method, path};
        use wiremock::{Mock, MockServer, ResponseTemplate};

        let mock_server = MockServer::start().await;

        Mock::given(method("POST"))
            .and(path("/chat/completions"))
            .and(header("Authorization", "Bearer bad-key"))
            .respond_with(ResponseTemplate::new(401).set_body_string("Unauthorized"))
            .expect(1)
            .mount(&mock_server)
            .await;

        let provider =
            create_api_key_mode_provider_for_base_url("bedrock-test", mock_server.uri(), "bad-key");

        let result = provider
            .chat_completion(create_test_chat_request(false))
            .await;
        assert!(result.is_err(), "401 response should return an error");

        match result {
            Err(GatewayError::Provider {
                provider,
                status_code,
                message,
            }) => {
                assert_eq!(provider, "bedrock-test");
                assert_eq!(status_code, Some(401));
                assert!(message.contains("authentication failed"));
            }
            other => panic!("Expected provider error, got {:?}", other),
        }
    }

    #[tokio::test]
    async fn test_list_models_api_key_live_listing() {
        use wiremock::matchers::{header, method, path};
        use wiremock::{Mock, MockServer, ResponseTemplate};

        let mock_server = MockServer::start().await;

        // Mantle exposes an OpenAI-compatible /models listing. Verify the
        // provider reads it live so new models (e.g. openai.gpt-5.5) appear.
        Mock::given(method("GET"))
            .and(path("/models"))
            .and(header("Authorization", "Bearer test-api-key"))
            .respond_with(ResponseTemplate::new(200).set_body_json(serde_json::json!({
                "object": "list",
                "data": [
                    {"id": "openai.gpt-5.5", "object": "model", "owned_by": "openai"},
                    {"id": "openai.gpt-5.4", "object": "model", "owned_by": "openai"}
                ]
            })))
            .expect(1)
            .mount(&mock_server)
            .await;

        let provider = create_api_key_mode_provider_for_base_url(
            "bedrock-test",
            mock_server.uri(),
            "test-api-key",
        );

        let models = provider
            .list_models()
            .await
            .expect("list_models should succeed");
        let ids: Vec<String> = models.into_iter().map(|m| m.id).collect();
        assert!(ids.contains(&"openai.gpt-5.5".to_string()));
        assert!(ids.contains(&"openai.gpt-5.4".to_string()));
    }

    #[tokio::test]
    async fn test_list_models_falls_back_to_mantle_chat_catalog_on_failure() {
        use wiremock::matchers::{method, path};
        use wiremock::{Mock, MockServer, ResponseTemplate};

        let mock_server = MockServer::start().await;

        // Simulate the live Mantle /models listing being unavailable.
        Mock::given(method("GET"))
            .and(path("/models"))
            .respond_with(ResponseTemplate::new(500).set_body_string("error"))
            .mount(&mock_server)
            .await;

        let provider = create_api_key_mode_provider_for_base_url(
            "bedrock-test",
            mock_server.uri(),
            "test-api-key",
        );

        let models = provider
            .list_models()
            .await
            .expect("list_models should fall back");
        let ids: Vec<String> = models.into_iter().map(|m| m.id).collect();
        // Mantle Chat fallback must surface verified Chat Completions-capable models.
        assert!(ids.contains(&"openai.gpt-oss-120b".to_string()));
        assert!(ids.contains(&"openai.gpt-oss-20b".to_string()));
        assert!(ids.contains(&"deepseek.v3.2".to_string()));
        // Responses-only IDs are included because the provider now dispatches
        // them through the dedicated Mantle Responses adapter.
        assert!(ids.contains(&"openai.gpt-5.5".to_string()));
        assert!(ids.contains(&"openai.gpt-5.4".to_string()));
        assert!(ids.contains(&"anthropic.claude-sonnet-5".to_string()));
    }

    #[test]
    fn test_mantle_chat_fallback_includes_only_chat_compatible_models() {
        let ids: Vec<String> = BedrockProvider::mantle_chat_fallback_models()
            .into_iter()
            .map(|m| m.id)
            .collect();
        // Verified Mantle Chat IDs.
        assert!(ids.contains(&"openai.gpt-oss-120b".to_string()));
        assert!(ids.contains(&"openai.gpt-oss-20b".to_string()));
        assert!(ids.contains(&"deepseek.v3.2".to_string()));
        assert!(ids.contains(&"mistral.mistral-large-3-675b-instruct".to_string()));
        assert!(ids.contains(&"qwen.qwen3-32b".to_string()));
        // Runtime-only IDs must not appear in the Mantle Chat catalog.
        assert!(!ids.contains(&"openai.gpt-oss-120b-1:0".to_string()));
        assert!(!ids.contains(&"openai.gpt-oss-20b-1:0".to_string()));
        // Responses-only OpenAI models are intentionally absent.
        assert!(!ids.contains(&"openai.gpt-5.5".to_string()));
        assert!(!ids.contains(&"openai.gpt-5.4".to_string()));
        // Responses and Messages catalogs become visible only because their
        // dedicated adapters dispatch those IDs to compatible Mantle APIs.
        let responses: Vec<String> = BedrockProvider::mantle_responses_fallback_models()
            .into_iter()
            .map(|model| model.id)
            .collect();
        let messages: Vec<String> = BedrockProvider::mantle_messages_fallback_models()
            .into_iter()
            .map(|model| model.id)
            .collect();
        assert!(responses.contains(&"openai.gpt-5.6-sol".to_string()));
        assert!(responses.contains(&"openai.gpt-5.5".to_string()));
        assert!(messages.contains(&"anthropic.claude-sonnet-5".to_string()));
    }

    #[test]
    fn test_runtime_fallback_uses_converse_ids() {
        let ids: Vec<String> = BedrockProvider::runtime_fallback_models()
            .into_iter()
            .map(|m| m.id)
            .collect();
        // Verified runtime (Converse/Invoke) IDs.
        assert!(ids.contains(&"openai.gpt-oss-120b-1:0".to_string()));
        assert!(ids.contains(&"openai.gpt-oss-20b-1:0".to_string()));
        assert!(ids.contains(&"anthropic.claude-sonnet-5".to_string()));
        assert!(ids.contains(&"anthropic.claude-opus-4-8".to_string()));
        assert!(ids.contains(&"amazon.nova-2-lite-v1:0".to_string()));
        assert!(ids.contains(&"amazon.nova-pro-v1:0".to_string()));
        // Mantle-only IDs do not belong in the runtime catalog.
        assert!(!ids.contains(&"openai.gpt-oss-120b".to_string()));
        // Some providers use the same verified ID on both endpoints.
        assert!(ids.contains(&"deepseek.v3.2".to_string()));
        // Responses-only OpenAI models are absent.
        assert!(!ids.contains(&"openai.gpt-5.5".to_string()));
        assert!(!ids.contains(&"openai.gpt-5.4".to_string()));
        // Legacy models are excluded.
        assert!(!ids.contains(&"amazon.nova-premier-v1:0".to_string()));
        assert!(!ids.contains(&"meta.llama3-1-405b-instruct-v1:0".to_string()));
    }

    #[test]
    fn test_mantle_message_normalizer_converts_responses_content_parts() {
        let mut request = create_test_chat_request(false);
        request.messages = vec![
            Message {
                role: "developer".to_string(),
                content: serde_json::json!([{
                    "type": "input_text",
                    "text": "instructions",
                    "cache_control": {"type": "ephemeral"}
                }]),
                extra: Default::default(),
            },
            Message {
                role: "assistant".to_string(),
                content: serde_json::json!([{"type": "output_text", "text": "previous"}]),
                extra: Default::default(),
            },
        ];

        assert_eq!(normalize_mantle_chat_messages(&mut request), 4);
        assert_eq!(request.messages[0].role, "system");
        assert_eq!(request.messages[0].content[0]["type"], "text");
        assert!(request.messages[0].content[0]
            .get("cache_control")
            .is_none());
        assert_eq!(request.messages[1].content[0]["type"], "text");
    }

    #[test]
    fn test_mantle_chat_sanitizer_removes_gateway_only_fields() {
        let mut request = create_test_chat_request(false);
        request
            .extra
            .insert("reasoning_effort".to_string(), serde_json::json!("high"));
        request
            .extra
            .insert("store".to_string(), serde_json::json!(false));
        request
            .extra
            .insert("parallel_tool_calls".to_string(), serde_json::json!(true));
        request.extra.insert(
            "tools".to_string(),
            serde_json::json!([{"type":"function","function":{"name":"read_file","parameters":{"type":"object"}}}]),
        );

        assert_eq!(sanitize_mantle_chat_request(&mut request), 1);
        assert_eq!(
            request.extra.get("reasoning_effort"),
            Some(&serde_json::json!("high"))
        );
        assert!(!request.extra.contains_key("store"));
        assert!(request.extra.contains_key("parallel_tool_calls"));
        assert!(request.extra.contains_key("tools"));
    }

    #[test]
    fn test_mantle_api_dispatch() {
        assert_eq!(mantle_api_for_model("openai.gpt-oss-120b"), MantleApi::Chat);
        assert_eq!(
            mantle_api_for_model("openai.gpt-5.6-sol"),
            MantleApi::Responses
        );
        assert_eq!(mantle_api_for_model("openai.gpt-5.5"), MantleApi::Responses);
        assert_eq!(
            mantle_api_for_model("openai.gpt-5.5-2026-04-23"),
            MantleApi::Responses
        );
        assert_eq!(
            mantle_api_for_model("anthropic.claude-sonnet-5"),
            MantleApi::Messages
        );
    }

    #[tokio::test]
    async fn test_responses_api_translation() {
        use wiremock::matchers::{header, method, path};
        use wiremock::{Mock, MockServer, ResponseTemplate};

        let server = MockServer::start().await;
        Mock::given(method("POST"))
            .and(path("/openai/v1/responses"))
            .and(header("Authorization", "Bearer test-api-key"))
            .respond_with(ResponseTemplate::new(200).set_body_json(serde_json::json!({
                "id": "resp_test",
                "output_text": "response adapter works",
                "usage": {"input_tokens": 3, "output_tokens": 4}
            })))
            .expect(1)
            .mount(&server)
            .await;
        let provider = create_api_key_mode_provider_for_base_url(
            "bedrock-test",
            format!("{}/v1", server.uri()),
            "test-api-key",
        );
        let mut request = create_test_chat_request(false);
        request.model = "openai.gpt-5.6-sol".to_string();
        let response = provider.chat_completion(request).await.unwrap().response;
        assert_eq!(
            response.choices[0].message.content_as_text(),
            "response adapter works"
        );
        assert_eq!(response.usage.total_tokens, 7);
    }

    #[tokio::test]
    async fn test_versioned_gpt_55_uses_responses_api() {
        use wiremock::matchers::{method, path};
        use wiremock::{Mock, MockServer, ResponseTemplate};

        let server = MockServer::start().await;
        Mock::given(method("POST"))
            .and(path("/openai/v1/responses"))
            .respond_with(ResponseTemplate::new(200).set_body_json(serde_json::json!({
                "id": "resp_versioned",
                "output_text": "versioned response adapter works",
                "usage": {"input_tokens": 3, "output_tokens": 4}
            })))
            .expect(1)
            .mount(&server)
            .await;
        let provider = create_api_key_mode_provider_for_base_url(
            "bedrock-test",
            format!("{}/v1", server.uri()),
            "test-api-key",
        );
        let mut request = create_test_chat_request(false);
        request.model = "openai.gpt-5.5-2026-04-23".to_string();
        let response = provider.chat_completion(request).await.unwrap().response;
        assert_eq!(
            response.choices[0].message.content_as_text(),
            "versioned response adapter works"
        );
    }

    #[tokio::test]
    async fn test_messages_api_translation() {
        use wiremock::matchers::{header, method, path};
        use wiremock::{Mock, MockServer, ResponseTemplate};

        let server = MockServer::start().await;
        Mock::given(method("POST"))
            .and(path("/v1/messages"))
            .and(header("Authorization", "Bearer test-api-key"))
            .respond_with(ResponseTemplate::new(200).set_body_json(serde_json::json!({
                "id": "msg_test",
                "content": [{"type": "text", "text": "messages adapter works"}],
                "stop_reason": "end_turn",
                "usage": {"input_tokens": 5, "output_tokens": 6}
            })))
            .expect(1)
            .mount(&server)
            .await;
        let provider = create_api_key_mode_provider_for_base_url(
            "bedrock-test",
            format!("{}/v1", server.uri()),
            "test-api-key",
        );
        let mut request = create_test_chat_request(false);
        request.model = "anthropic.claude-sonnet-5".to_string();
        let response = provider.chat_completion(request).await.unwrap().response;
        assert_eq!(
            response.choices[0].message.content_as_text(),
            "messages adapter works"
        );
        assert_eq!(response.usage.total_tokens, 11);
    }

    #[tokio::test]
    async fn test_api_key_streaming_success() {
        use wiremock::matchers::{header, method, path};
        use wiremock::{Mock, MockServer, ResponseTemplate};

        let mock_server = MockServer::start().await;
        let sse_body = concat!(
            "data: {\"id\":\"chatcmpl-test\",\"object\":\"chat.completion.chunk\",\"choices\":[{\"index\":0,\"delta\":{\"content\":\"hello\"},\"finish_reason\":null}]}\n\n",
            "data: {\"id\":\"chatcmpl-test\",\"object\":\"chat.completion.chunk\",\"choices\":[{\"index\":0,\"delta\":{\"content\":\" world\"},\"finish_reason\":null}]}\n\n",
            "data: [DONE]\n\n"
        );

        Mock::given(method("POST"))
            .and(path("/chat/completions"))
            .and(header("Authorization", "Bearer test-api-key"))
            .respond_with(
                ResponseTemplate::new(200)
                    .insert_header("content-type", "text/event-stream")
                    .set_body_string(sse_body),
            )
            .expect(1)
            .mount(&mock_server)
            .await;

        let provider = create_api_key_mode_provider_for_base_url(
            "bedrock-test",
            mock_server.uri(),
            "test-api-key",
        );

        let stream = provider
            .chat_completion_stream(create_test_chat_request(true))
            .await
            .expect("stream should be created");

        let events: Vec<Result<SSEEvent, GatewayError>> = stream.collect().await;
        assert_eq!(
            events.len(),
            2,
            "[DONE] terminator should not produce an event"
        );

        let first = events[0].as_ref().expect("first SSE event should be ok");
        let second = events[1].as_ref().expect("second SSE event should be ok");
        assert!(first.data.contains("hello"));
        assert!(second.data.contains(" world"));
    }
}

#[cfg(test)]
mod property_tests {
    use super::*;
    use crate::models::openai::Message;
    use proptest::prelude::*;

    /// Helper function to create a test BedrockProvider with AWS SDK auth mode
    fn create_test_provider(name: &str, region: &str) -> BedrockProvider {
        let client = BedrockClient::from_conf(
            aws_sdk_bedrockruntime::Config::builder()
                .behavior_version(aws_sdk_bedrockruntime::config::BehaviorVersion::latest())
                .build(),
        );
        let control_client = BedrockControlClient::from_conf(
            aws_sdk_bedrock::Config::builder()
                .behavior_version(aws_sdk_bedrock::config::BehaviorVersion::latest())
                .build(),
        );
        BedrockProvider {
            name: name.to_string(),
            region: region.to_string(),
            auth_mode: BedrockAuthMode::AwsSdk {
                client,
                control_client,
            },
        }
    }

    fn arb_openai_request() -> impl Strategy<Value = OpenAIRequest> {
        (
            prop::sample::select(vec![
                "claude-3-opus",
                "claude-3-sonnet",
                "claude-2",
                "titan-text-express",
                "titan-text-lite",
                "jurassic-2-ultra",
                "jurassic-2-mid",
                "command-text",
                "command-light",
            ]),
            prop::collection::vec(
                (
                    prop::sample::select(vec!["system", "user", "assistant"]),
                    "[a-zA-Z0-9 ]{10,50}",
                ),
                1..5,
            ),
            prop::option::of(0.0f32..2.0f32),
            prop::option::of(100u32..2048u32),
        )
            .prop_map(|(model, messages, temperature, max_tokens)| OpenAIRequest {
                model: model.to_string(),
                messages: messages
                    .into_iter()
                    .map(|(role, content)| Message {
                        role: role.to_string(),
                        content: serde_json::Value::String(content),
                        extra: Default::default(),
                    })
                    .collect(),
                stream: false,
                temperature,
                max_tokens,
                extra: Default::default(),
            })
    }

    // Feature: ai-gateway, Property 17: Bedrock Translation Round-Trip
    // **Validates: Requirements 3.11, 3.12, 23.1-23.5**
    proptest! {
        #[test]
        fn prop_bedrock_translation_round_trip(request in arb_openai_request()) {
            let provider = create_test_provider("test-bedrock", "us-east-1");

            let model_id = provider.translate_model_id(&request.model);

            // Step 1: Translate OpenAI request to Bedrock format
            let bedrock_request = provider.translate_request(&request, &model_id);
            prop_assert!(bedrock_request.is_ok(), "Request translation must succeed for valid OpenAI request");

            let bedrock_json = bedrock_request.unwrap();
            prop_assert!(!bedrock_json.is_empty(), "Bedrock request must not be empty");

            // Step 2: Create mock Bedrock response based on model family
            let mock_response = if model_id.starts_with("anthropic.claude") {
                r#"{"completion":"test response","stop_reason":"stop"}"#
            } else if model_id.starts_with("amazon.titan") {
                r#"{"results":[{"outputText":"test response"}]}"#
            } else if model_id.starts_with("ai21.j2") {
                r#"{"completions":[{"data":{"text":"test response"}}]}"#
            } else if model_id.starts_with("cohere.command") {
                r#"{"generations":[{"text":"test response"}]}"#
            } else {
                panic!("Unsupported model family");
            };

            // Step 3: Translate Bedrock response back to OpenAI format
            let openai_response = provider.translate_claude_response(mock_response, &request.model)
                .or_else(|_| provider.translate_titan_response(mock_response, &request.model))
                .or_else(|_| provider.translate_jurassic_response(mock_response, &request.model))
                .or_else(|_| provider.translate_command_response(mock_response, &request.model));

            prop_assert!(openai_response.is_ok(), "Response translation must succeed");

            let response = openai_response.unwrap();

            // Verify OpenAI response structure
            prop_assert_eq!(response.object, "chat.completion");
            prop_assert_eq!(response.model, request.model);
            prop_assert_eq!(response.choices.len(), 1);
            prop_assert_eq!(response.choices[0].index, 0);
            prop_assert_eq!(&response.choices[0].message.role, "assistant");
            prop_assert!(response.choices[0].message.content != serde_json::Value::Null,
                "Response content must not be null");

            // Verify semantic content preserved (response contains expected text)
            prop_assert_eq!(response.choices[0].message.content.clone(), serde_json::Value::String("test response".to_string()));
        }
    }

    /// Strategy: alphanumeric + hyphens, 1..30 chars (mimics valid AWS region strings)
    fn arb_region_string() -> impl Strategy<Value = String> {
        proptest::string::string_regex("[a-zA-Z0-9][a-zA-Z0-9-]{0,29}").expect("valid regex")
    }

    // Feature: bedrock-ui-integration, Property 1: Mantle URL generation is deterministic and well-formed
    // **Validates: Requirements 2.1, 9.3**
    proptest! {
        #![proptest_config(ProptestConfig::with_cases(100))]

        #[test]
        fn prop_mantle_url_generation_deterministic_and_well_formed(region in arb_region_string()) {
            let url = build_mantle_base_url(&region);

            // URL matches the exact expected format
            let expected = format!("https://bedrock-mantle.{}.api.aws/v1", region);
            prop_assert_eq!(&url, &expected, "URL must match https://bedrock-mantle.{{region}}.api.aws/v1");

            // The region substring in the output equals the input
            let prefix = "https://bedrock-mantle.";
            let suffix = ".api.aws/v1";
            prop_assert!(url.starts_with(prefix), "URL must start with {}", prefix);
            prop_assert!(url.ends_with(suffix), "URL must end with {}", suffix);
            let extracted_region = &url[prefix.len()..url.len() - suffix.len()];
            prop_assert_eq!(extracted_region, region.as_str(), "Extracted region must equal input region");

            // Determinism: calling twice yields the same result
            let url2 = build_mantle_base_url(&region);
            prop_assert_eq!(&url, &url2, "build_mantle_base_url must be deterministic");
        }
    }

    /// Strategy: generate a model ID string (alphanumeric + dots + colons + hyphens)
    fn arb_model_id() -> impl Strategy<Value = String> {
        proptest::string::string_regex("[a-zA-Z][a-zA-Z0-9._:-]{0,59}").expect("valid regex")
    }

    /// Strategy: pick one of the supported Bedrock regions
    fn arb_bedrock_region() -> impl Strategy<Value = &'static str> {
        prop::sample::select(BEDROCK_REGIONS)
    }

    fn arb_profile_model_id() -> impl Strategy<Value = String> {
        prop::sample::select(vec![
            "anthropic.claude-sonnet-5".to_string(),
            "anthropic.claude-sonnet-4-5-20250929-v1:0".to_string(),
            "anthropic.claude-opus-4-8".to_string(),
            "amazon.nova-pro-v1:0".to_string(),
            "writer.palmyra-x5-v1:0".to_string(),
        ])
    }

    // Feature: bedrock-ui-integration, Property 2: supported geo inference profile prefixing
    // **Validates: Requirements 4.3, 4.4**
    proptest! {
        #![proptest_config(ProptestConfig::with_cases(100))]

        #[test]
        fn prop_global_inference_prefix_when_enabled(
            model_id in arb_profile_model_id(),
            region in arb_bedrock_region(),
        ) {
            let result = apply_global_inference_prefix(&model_id, region, true);
            let region_group = derive_region_group(region);

            // Region group must be non-empty for all supported regions
            prop_assert!(!region_group.is_empty(), "Supported region must have a region group");

            let expected_prefix = format!("{}.", region_group);

            // Result must start with the region group prefix
            prop_assert!(
                result.starts_with(&expected_prefix),
                "Prefixed model ID '{}' must start with '{}'", result, expected_prefix
            );

            // Result must end with the original model ID
            prop_assert!(
                result.ends_with(&model_id),
                "Prefixed model ID '{}' must end with original '{}'", result, model_id
            );
        }

        #[test]
        fn prop_global_inference_prefix_when_disabled(
            model_id in arb_model_id(),
            region in arb_bedrock_region(),
        ) {
            let result = apply_global_inference_prefix(&model_id, region, false);

            // When disabled, model ID must be unchanged
            prop_assert_eq!(
                &result, &model_id,
                "Model ID must be unchanged when global_inference_profile is false"
            );
        }

        #[test]
        fn prop_global_inference_no_double_prefix(
            model_id in arb_profile_model_id(),
            region in arb_bedrock_region(),
        ) {
            // Apply prefix once
            let once = apply_global_inference_prefix(&model_id, region, true);
            // Apply prefix again on the already-prefixed result
            let twice = apply_global_inference_prefix(&once, region, true);

            // No double-prefixing: applying twice must equal applying once
            prop_assert_eq!(
                &once, &twice,
                "Double-prefixing must not occur: first='{}', second='{}'", once, twice
            );
        }
    }

    #[test]
    fn inference_profiles_leave_unsupported_mantle_models_unchanged() {
        for model_id in [
            "zai.glm-5",
            "moonshotai.kimi-k2.5",
            "openai.gpt-5.6-sol",
            "openai.gpt-5.5-2026-04-23",
        ] {
            assert_eq!(
                apply_global_inference_prefix(model_id, "us-east-2", true),
                model_id
            );
            assert_eq!(apply_global_inference_profile(model_id, true), model_id);
        }
    }

    #[test]
    fn global_profile_uses_global_prefix_only_for_supported_models() {
        assert_eq!(
            apply_global_inference_profile("anthropic.claude-sonnet-5", true),
            "global.anthropic.claude-sonnet-5"
        );
        assert_eq!(
            apply_global_inference_profile("global.anthropic.claude-sonnet-5", true),
            "global.anthropic.claude-sonnet-5"
        );
        assert_eq!(
            apply_global_inference_profile("anthropic.claude-opus-4-8", true),
            "anthropic.claude-opus-4-8"
        );
    }

    /// Known reasoning-capable model IDs that MUST return true.
    fn arb_known_reasoning_model() -> impl Strategy<Value = String> {
        prop::sample::select(vec![
            // Claude 3.5 Sonnet v2+
            "anthropic.claude-3-5-sonnet-20241022-v2:0".to_string(),
            "us.anthropic.claude-3-5-sonnet-20241022-v2:0".to_string(),
            "anthropic.claude-3-5-sonnet-20241022-v3:0".to_string(),
            "anthropic.claude-3-5-sonnet-20250101-v9:0".to_string(),
            // Claude 3 Opus
            "anthropic.claude-3-opus-20240229-v1:0".to_string(),
            "us.anthropic.claude-3-opus-20240229-v1:0".to_string(),
            "eu.anthropic.claude-3-opus-20240229-v1:0".to_string(),
            // Claude 4+ family
            "anthropic.claude-4-sonnet-20250514-v1:0".to_string(),
            "us.anthropic.claude-4-opus-20250601-v1:0".to_string(),
            "anthropic.claude-5-sonnet-20260101-v1:0".to_string(),
            // Current Anthropic Claude verified IDs (AWS model cards, 2026-07)
            "anthropic.claude-sonnet-5".to_string(),
            "us.anthropic.claude-sonnet-5".to_string(),
            "anthropic.claude-opus-4-8".to_string(),
            "us.anthropic.claude-opus-4-8".to_string(),
            "anthropic.claude-opus-4-7".to_string(),
            "us.anthropic.claude-opus-4-7".to_string(),
            "anthropic.claude-haiku-4-5-20251001-v1:0".to_string(),
        ])
    }

    /// Known non-reasoning model IDs that MUST return false.
    fn arb_known_non_reasoning_model() -> impl Strategy<Value = String> {
        prop::sample::select(vec![
            // Claude 3.5 Sonnet v1 (not v2+)
            "anthropic.claude-3-5-sonnet-20240620-v1:0".to_string(),
            // Claude 3.5 Haiku
            "anthropic.claude-3-5-haiku-20241022-v1:0".to_string(),
            // Claude 3 Haiku
            "anthropic.claude-3-haiku-20240307-v1:0".to_string(),
            // Claude 3 Sonnet (not 3.5)
            "anthropic.claude-3-sonnet-20240229-v1:0".to_string(),
            // Titan models
            "amazon.titan-text-express-v1".to_string(),
            "amazon.titan-text-lite-v1".to_string(),
            // Llama models
            "meta.llama3-1-70b-instruct-v1:0".to_string(),
            "meta.llama3-1-8b-instruct-v1:0".to_string(),
            // Mistral models
            "mistral.mistral-large-2407-v1:0".to_string(),
            // Cohere models
            "cohere.command-r-plus-v1:0".to_string(),
            "cohere.command-r-v1:0".to_string(),
        ])
    }

    // Feature: bedrock-ui-integration, Property 5: Reasoning model support detection
    // **Validates: Requirements 7.5**
    proptest! {
        #![proptest_config(ProptestConfig::with_cases(100))]

        #[test]
        fn prop_reasoning_known_models_return_true(model_id in arb_known_reasoning_model()) {
            prop_assert!(
                model_supports_reasoning(&model_id),
                "Known reasoning model '{}' must return true", model_id
            );
        }

        #[test]
        fn prop_reasoning_known_non_reasoning_models_return_false(model_id in arb_known_non_reasoning_model()) {
            prop_assert!(
                !model_supports_reasoning(&model_id),
                "Known non-reasoning model '{}' must return false", model_id
            );
        }

        #[test]
        fn prop_reasoning_arbitrary_strings_return_false(
            s in "[a-zA-Z0-9._:-]{1,60}"
                .prop_filter("must not match any reasoning pattern", |s| {
                    let lower = s.to_lowercase();
                    !lower.contains("claude-3-5-sonnet")
                        && !lower.contains("claude-3-opus")
                        && !lower.contains("claude-4")
                        && !lower.contains("claude-5")
                })
        ) {
            prop_assert!(
                !model_supports_reasoning(&s),
                "Arbitrary non-reasoning string '{}' must return false", s
            );
        }
    }

    // Feature: bedrock-ui-integration, Property 3: Model list merge produces deduplicated sorted union
    // **Validates: Requirements 5.4**
    proptest! {
        #![proptest_config(ProptestConfig::with_cases(100))]

        #[test]
        fn prop_merge_model_lists_dedup_sorted_union(
            list_a in prop::collection::vec("[a-zA-Z0-9._:-]{1,40}", 0..20),
            list_b in prop::collection::vec("[a-zA-Z0-9._:-]{1,40}", 0..20),
        ) {
            let merged = merge_model_lists(list_a.clone(), list_b.clone());

            // 1. Result contains all elements from both lists (set union)
            let expected_set: std::collections::BTreeSet<String> =
                list_a.iter().chain(list_b.iter()).cloned().collect();
            let merged_set: std::collections::BTreeSet<String> =
                merged.iter().cloned().collect();
            prop_assert_eq!(
                &merged_set, &expected_set,
                "Merged result must be the set union of both inputs"
            );

            // 2. No duplicates in result
            prop_assert_eq!(
                merged.len(), merged_set.len(),
                "Merged result must contain no duplicates"
            );

            // 3. Result is sorted lexicographically
            for w in merged.windows(2) {
                prop_assert!(
                    w[0] <= w[1],
                    "Merged result must be sorted: '{}' should come before '{}'", w[0], w[1]
                );
            }
        }
    }
}
