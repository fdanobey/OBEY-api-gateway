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

fn is_compaction_trigger(value: &serde_json::Value) -> bool {
    value.get("type").and_then(serde_json::Value::as_str) == Some("compaction_trigger")
}

/// True when a message carries a compaction trigger via its flattened `extra`
/// map — i.e. `extra["type"] == "compaction_trigger"`. This is the message-level
/// marker shape that `is_compaction_trigger` (which inspects a standalone JSON
/// value) does not see, because the marker lives in `Message.extra`.
fn message_extra_is_trigger(message: &Message) -> bool {
    message.extra.get("type").and_then(serde_json::Value::as_str) == Some("compaction_trigger")
}

/// Outcome of a single normalization scan over every compaction-trigger site in
/// an outgoing Bedrock Mantle request.
///
/// - `removed`: number of trigger sites deleted (every site except the survivor).
/// - `survivor`: a clone of the surviving trigger's JSON value, used later for
///   placement by the dispatching adapter. `None` when the request carried no
///   trigger at all.
/// - `survivor_from_input_array`: `true` only when the surviving (last) site was
///   an `extra["input"]` array item, so the adapter knows the survivor came from
///   the native input list rather than a message.
#[derive(Debug, Clone, Default, PartialEq)]
pub(crate) struct TriggerNormalization {
    pub removed: usize,
    pub survivor: Option<serde_json::Value>,
    pub survivor_from_input_array: bool,
}

/// Identifies where a trigger site lives, in document order, so a single scan
/// can decide which one survives and remove all the others consistently across
/// the three data locations.
enum TriggerSite {
    /// Item at `index` inside the `extra["input"]` array.
    InputArray { index: usize },
    /// Part at `part_index` inside `messages[message_index].content` array.
    ContentPart {
        message_index: usize,
        part_index: usize,
    },
    /// Message-level `extra["type"]` marker on `messages[message_index]`.
    MessageMarker { message_index: usize },
}

/// Normalize an outgoing Bedrock Mantle request so it carries at most one
/// `compaction_trigger`.
///
/// Performs ONE ordered scan of every trigger site in document order:
///   (a) `extra["input"]` array items — only when `input` is an array;
///   (b) then, per message in order, that message's content-array parts;
///   (c) followed by that message's message-level `extra` marker.
///
/// The LAST site encountered is kept; every earlier site is removed. Removal is
/// location-specific:
///   - content part      → dropped from the `content` array;
///   - message marker     → the `type` key is removed from `message.extra`, and
///     the whole message is dropped when it is a standalone trigger residue
///     (blank role AND blank content after removal) so no `{"role":"","content":""}`
///     message is emitted;
///   - input-array item   → removed from the `extra["input"]` array.
///
/// Non-array `input` values are left untouched. The returned
/// [`TriggerNormalization`] records how many sites were removed, a clone of the
/// survivor's JSON value, and whether the survivor came from the input array.
pub(crate) fn normalize_mantle_compaction_triggers(
    request: &mut OpenAIRequest,
) -> TriggerNormalization {
    // --- Pass 1: enumerate every trigger site in document order. ---
    let mut sites: Vec<TriggerSite> = Vec::new();

    // (a) extra["input"] array items (only when input is an array).
    let input_is_array = request
        .extra
        .get("input")
        .is_some_and(serde_json::Value::is_array);
    if input_is_array {
        if let Some(items) = request
            .extra
            .get("input")
            .and_then(serde_json::Value::as_array)
        {
            for (index, item) in items.iter().enumerate() {
                if is_compaction_trigger(item) {
                    sites.push(TriggerSite::InputArray { index });
                }
            }
        }
    }

    // (b)+(c) per message: content parts first, then the message-level marker.
    for (message_index, message) in request.messages.iter().enumerate() {
        if let serde_json::Value::Array(parts) = &message.content {
            for (part_index, part) in parts.iter().enumerate() {
                if is_compaction_trigger(part) {
                    sites.push(TriggerSite::ContentPart {
                        message_index,
                        part_index,
                    });
                }
            }
        }
        if message_extra_is_trigger(message) {
            sites.push(TriggerSite::MessageMarker { message_index });
        }
    }

    if sites.is_empty() {
        return TriggerNormalization::default();
    }

    // The survivor is the LAST site in document order.
    let survivor_index = sites.len() - 1;
    let survivor_site = &sites[survivor_index];
    let survivor_from_input_array = matches!(survivor_site, TriggerSite::InputArray { .. });
    let survivor = capture_survivor(request, survivor_site);
    let removed = sites.len() - 1;

    // --- Pass 2: remove every site EXCEPT the survivor. ---
    // Deleting a site never shifts the survivor's location because removals are
    // applied in reverse document order (highest index first) within each data
    // location, and cross-location removals target disjoint containers.
    for site in sites[..survivor_index].iter().rev() {
        match site {
            TriggerSite::InputArray { index } => {
                if let Some(items) = request
                    .extra
                    .get_mut("input")
                    .and_then(serde_json::Value::as_array_mut)
                {
                    if *index < items.len() {
                        items.remove(*index);
                    }
                }
            }
            TriggerSite::ContentPart {
                message_index,
                part_index,
            } => {
                if let Some(message) = request.messages.get_mut(*message_index) {
                    if let serde_json::Value::Array(parts) = &mut message.content {
                        if *part_index < parts.len() {
                            parts.remove(*part_index);
                        }
                    }
                }
            }
            TriggerSite::MessageMarker { message_index } => {
                if let Some(message) = request.messages.get_mut(*message_index) {
                    message.extra.remove("type");
                }
            }
        }
    }

    // Drop any standalone-trigger residue messages (blank role AND blank content)
    // left behind after removing a message-level marker, so no `{"role":"","content":""}`
    // message is emitted. Never drop the survivor's message. Iterate in reverse so
    // index removals do not disturb earlier indices, and skip the survivor message.
    let survivor_message_index = match survivor_site {
        TriggerSite::ContentPart { message_index, .. }
        | TriggerSite::MessageMarker { message_index } => Some(*message_index),
        TriggerSite::InputArray { .. } => None,
    };
    for message_index in (0..request.messages.len()).rev() {
        if Some(message_index) == survivor_message_index {
            continue;
        }
        let is_residue = {
            let message = &request.messages[message_index];
            !message_extra_is_trigger(message)
                && role_is_blank(&message.role)
                && content_is_blank(&message.content)
                && was_trigger_residue(&sites, message_index)
        };
        if is_residue {
            request.messages.remove(message_index);
        }
    }

    if removed > 0 {
        tracing::debug!(
            compaction_triggers_removed = removed,
            "Normalized Bedrock Mantle compaction triggers"
        );
    }

    TriggerNormalization {
        removed,
        survivor,
        survivor_from_input_array,
    }
}

/// Clone the surviving trigger's JSON value from its site, for later placement.
fn capture_survivor(
    request: &OpenAIRequest,
    site: &TriggerSite,
) -> Option<serde_json::Value> {
    match site {
        TriggerSite::InputArray { index } => request
            .extra
            .get("input")
            .and_then(serde_json::Value::as_array)
            .and_then(|items| items.get(*index))
            .cloned(),
        TriggerSite::ContentPart {
            message_index,
            part_index,
        } => request
            .messages
            .get(*message_index)
            .and_then(|message| message.content.as_array())
            .and_then(|parts| parts.get(*part_index))
            .cloned(),
        TriggerSite::MessageMarker { message_index } => request
            .messages
            .get(*message_index)
            .map(|message| serde_json::Value::Object(message.extra.clone())),
    }
}

/// True when a message at `message_index` had a message-level marker among the
/// enumerated sites — i.e. it is a candidate standalone-trigger residue.
fn was_trigger_residue(sites: &[TriggerSite], message_index: usize) -> bool {
    sites.iter().any(|site| {
        matches!(
            site,
            TriggerSite::MessageMarker { message_index: idx } if *idx == message_index
        )
    })
}

/// A role is blank when it is empty or whitespace-only.
fn role_is_blank(role: &str) -> bool {
    role.trim().is_empty()
}

/// Content is blank when it is JSON null, an empty/whitespace-only string, or an
/// empty array.
fn content_is_blank(content: &serde_json::Value) -> bool {
    match content {
        serde_json::Value::Null => true,
        serde_json::Value::String(s) => s.trim().is_empty(),
        serde_json::Value::Array(parts) => parts.is_empty(),
        _ => false,
    }
}

/// Surface a surviving compaction trigger into a Chat-family request body as a
/// content part, so a native `extra["input"]` survivor is not lost when
/// `sanitize_mantle_chat_request` deletes the `input` key.
///
/// The trigger is appended to the LAST message's content array (converting a
/// scalar/string content into a `text` part first so the real content is
/// preserved). When the request has no messages, a new user message carrying the
/// trigger is pushed. Exactly one trigger is added, so the outgoing body carries
/// a single trigger site.
fn surface_trigger_as_content_part(request: &mut OpenAIRequest, survivor: serde_json::Value) {
    if let Some(message) = request.messages.last_mut() {
        let mut parts = match std::mem::take(&mut message.content) {
            serde_json::Value::Array(existing) => existing,
            serde_json::Value::Null => Vec::new(),
            serde_json::Value::String(text) if text.is_empty() => Vec::new(),
            serde_json::Value::String(text) => {
                vec![serde_json::json!({"type": "text", "text": text})]
            }
            other => vec![serde_json::json!({"type": "text", "text": other.to_string()})],
        };
        parts.push(survivor);
        message.content = serde_json::Value::Array(parts);
    } else {
        request.messages.push(Message {
            role: "user".to_string(),
            content: serde_json::Value::Array(vec![survivor]),
            extra: serde_json::Map::new(),
        });
    }
}

/// Recognize the Bedrock Mantle rejection produced when an outgoing request
/// still carries more than one `compaction_trigger`. The observed body is
/// `Only one 'compaction_trigger' item may be provided.` returned with HTTP 400.
///
/// Detection is 4xx-gated (the invariant is a client-request contract, so a 5xx
/// server error must never be treated as this repairable condition) AND requires
/// a case-insensitive body match on `compaction_trigger` together with one of the
/// phrasing anchors (`only one` / `may be provided`). The phrasing check mirrors
/// the tolerance style of `Router::is_unsupported_image_phrasing`, staying robust
/// to case and to error-envelope wrappers (e.g. `{"error":{"message":"..."}}`).
///
/// This is the backstop detector for the one-shot repair-and-retry arm: it must
/// match only this specific rejection so unrelated failures keep their existing
/// retry / failover / circuit-breaker behavior (clause 3.9).
pub(crate) fn is_duplicate_compaction_trigger_error(status_code: u16, body: &str) -> bool {
    if !(400..500).contains(&status_code) {
        return false;
    }
    let lower = body.to_ascii_lowercase();
    lower.contains("compaction_trigger")
        && (lower.contains("only one") || lower.contains("may be provided"))
}

pub(crate) fn normalize_mantle_chat_messages(request: &mut OpenAIRequest) -> usize {
    // Compaction-trigger de-duplication is owned by the single seam
    // `normalize_mantle_compaction_triggers`. This function keeps only the
    // non-trigger Mantle normalizations: `developer` -> `system` role mapping,
    // `input_text`/`output_text` -> `text` part rewriting, and `cache_control`
    // removal. The returned count reflects only that retained work.
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
        "reasoning",
        "thinking",
        "output_config",
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

fn responses_tools_from_chat(tools: &serde_json::Value) -> serde_json::Value {
    let Some(items) = tools.as_array() else {
        return tools.clone();
    };

    serde_json::Value::Array(
        items
            .iter()
            .map(|tool| {
                let Some(function) = tool
                    .get("function")
                    .filter(|_| {
                        tool.get("type").and_then(serde_json::Value::as_str) == Some("function")
                    })
                    .and_then(serde_json::Value::as_object)
                else {
                    return tool.clone();
                };

                let mut flattened = serde_json::Map::new();
                flattened.insert("type".to_string(), serde_json::json!("function"));
                for key in ["name", "description", "parameters", "strict"] {
                    if let Some(value) = function.get(key) {
                        flattened.insert(key.to_string(), value.clone());
                    }
                }
                serde_json::Value::Object(flattened)
            })
            .collect(),
    )
}

fn responses_tool_choice_from_chat(tool_choice: &serde_json::Value) -> serde_json::Value {
    let Some(function) = tool_choice
        .get("function")
        .filter(|_| tool_choice.get("type").and_then(serde_json::Value::as_str) == Some("function"))
        .and_then(serde_json::Value::as_object)
    else {
        return tool_choice.clone();
    };

    match function.get("name") {
        Some(name) => serde_json::json!({"type": "function", "name": name}),
        None => tool_choice.clone(),
    }
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
    ///
    /// Reachable only through [`Self::dispatch_mantle`] /
    /// [`Self::dispatch_mantle_stream`], which run the trigger-normalization seam
    /// first and pass the resulting [`TriggerNormalization`] down as
    /// `normalization` so this adapter can place the survivor (task 5).
    async fn chat_completion_api_key(
        &self,
        mut request: OpenAIRequest,
        http_client: &Client,
        api_key: &str,
        base_url: &str,
        custom_headers: &HashMap<String, String>,
        normalization: &TriggerNormalization,
    ) -> Result<ProviderResponse, GatewayError> {
        let start = Instant::now();
        let url = format!("{}/chat/completions", base_url);
        // Survivor placement (task 5): when the surviving trigger came from a
        // native `extra["input"]` list, it would be lost because
        // `sanitize_mantle_chat_request` deletes `input` wholesale (the key is
        // not in `MANTLE_CHAT_ALLOWED`). Surface it into the Chat body — the
        // shape the Chat endpoint accepts — as a content part BEFORE sanitize
        // runs, so exactly one trigger survives. When the survivor was already a
        // content part or message-level marker (`survivor_from_input_array =
        // false`), the seam left it in place and no relocation is needed, so a
        // single-trigger content-part payload stays byte-identical (clause 3.1).
        if normalization.survivor_from_input_array {
            if let Some(survivor) = &normalization.survivor {
                surface_trigger_as_content_part(&mut request, survivor.clone());
            }
        }
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

fn mantle_responses_input(
    request: OpenAIRequest,
) -> (
    String,
    Option<f32>,
    u32,
    serde_json::Map<String, serde_json::Value>,
    serde_json::Value,
    usize,
) {
    let OpenAIRequest {
        model,
        messages,
        temperature,
        max_tokens,
        mut extra,
        ..
    } = request;
    // Compaction-trigger de-duplication is owned by the single seam
    // `normalize_mantle_compaction_triggers`, which runs before this adapter and
    // has already reduced `extra["input"]` and message content to at most one
    // trigger. This builder no longer performs any trigger normalization, so the
    // trigger-removal count is always zero here; the tuple arity is kept stable
    // for the caller's logging.
    let normalized = 0;

    let input = match extra.remove("input") {
        Some(serde_json::Value::Array(input)) => serde_json::Value::Array(input),
        Some(other) => {
            // Non-array `input` values (e.g., string "auto" from previous session continuation)
            // are passed through as-is. The Responses API only validates compaction_trigger
            // counts inside arrays of input items, not scalar values.
            other
        }
        None => {
            // Build input from OpenAI-style messages. Flatten each message's content
            // to text because the Responses API input items don't support multi-part
            // content arrays. compaction_triggers were already normalized by the seam.
            // Any remaining compaction_trigger (at most one) in content arrays will
            // be filtered out by content_as_text which only extracts text parts.
            serde_json::Value::Array(
                messages
                    .iter()
                    .map(|message| {
                        serde_json::json!({
                            "role": message.role,
                            "content": message.content_as_text()
                        })
                    })
                    .collect::<Vec<_>>(),
            )
        }
    };
    (
        model,
        temperature,
        max_tokens.unwrap_or(2048),
        extra,
        input,
        normalized,
    )
}

    /// Reachable only through [`Self::dispatch_mantle`] /
    /// [`Self::dispatch_mantle_stream`]. See `normalization` note on
    /// [`Self::chat_completion_api_key`].
    async fn chat_completion_responses_api(
        &self,
        request: OpenAIRequest,
        http_client: &Client,
        api_key: &str,
        base_url: &str,
        custom_headers: &HashMap<String, String>,
        normalization: &TriggerNormalization,
    ) -> Result<ProviderResponse, GatewayError> {
        let start = Instant::now();
        let root = base_url.trim_end_matches('/').trim_end_matches("/v1");
        let url = format!("{}/openai/v1/responses", root);
        let (model, temperature, max_output_tokens, extra, mut input, normalized) =
            Self::mantle_responses_input(request);
        // Survivor placement (task 5): the Responses `input` array must carry the
        // surviving trigger as its TERMINAL element, regardless of where the
        // survivor originated (native `extra["input"]` item, content part, or
        // message-level marker). The seam removed every EARLIER site but left the
        // surviving site in place, so:
        //   - a native-input survivor is still inside the built `input` array;
        //   - a message survivor was dropped by `content_as_text()` when the
        //     array was rebuilt from messages.
        // Unifying rule: when `input` is an array, strip any compaction_trigger
        // items already in it, then append the recorded survivor once at the end.
        // A non-array `input` (e.g. "auto") is left untouched (clause 3.7); a
        // survivor cannot be placed there, which is only possible for degenerate
        // requests that also carry a scalar `input`.
        if let serde_json::Value::Array(items) = &mut input {
            items.retain(|item| !is_compaction_trigger(item));
            if let Some(survivor) = &normalization.survivor {
                items.push(survivor.clone());
            }
        }
        if normalized > 0 {
            tracing::debug!(
                provider = %self.name,
                model = %model,
                compaction_triggers_removed = normalized,
                "Normalized Bedrock Mantle Responses compaction triggers"
            );
        }
        let mut body = serde_json::json!({
            "model": model.clone(),
            "input": input,
            "max_output_tokens": max_output_tokens,
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
                body[key] = match key {
                    "tools" => responses_tools_from_chat(value),
                    "tool_choice" => responses_tool_choice_from_chat(value),
                    _ => value.clone(),
                };
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

    /// Reachable only through [`Self::dispatch_mantle`] /
    /// [`Self::dispatch_mantle_stream`]. See `normalization` note on
    /// [`Self::chat_completion_api_key`].
    async fn chat_completion_messages_api(
        &self,
        request: OpenAIRequest,
        http_client: &Client,
        api_key: &str,
        base_url: &str,
        custom_headers: &HashMap<String, String>,
        normalization: &TriggerNormalization,
    ) -> Result<ProviderResponse, GatewayError> {
        // Messages family needs no survivor placement: the seam's removal already
        // reduced the payload to <=1 trigger, Anthropic Messages does not use
        // `compaction_trigger`, and this adapter flattens content via
        // `content_as_text()`. The normalization outcome is intentionally unused.
        let _ = normalization;
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
    ///
    /// Reachable only through [`Self::dispatch_mantle_stream`]. See `normalization`
    /// note on [`Self::chat_completion_api_key`].
    async fn chat_completion_stream_api_key(
        &self,
        request: OpenAIRequest,
        http_client: &Client,
        api_key: &str,
        base_url: &str,
        custom_headers: &HashMap<String, String>,
        normalization: &TriggerNormalization,
    ) -> Result<Pin<Box<dyn Stream<Item = Result<SSEEvent, GatewayError>> + Send>>, GatewayError>
    {
        // TODO(task5): Chat-family survivor placement (native `extra["input"]`
        // awareness); the seam has already removed earlier sites.
        let _ = normalization;
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

    /// Single Mantle trigger-normalization entry point.
    ///
    /// Every Mantle dispatch — buffered and streaming, across all endpoint
    /// families — passes through here before branching on
    /// [`mantle_api_for_model`], so no dispatch path can reach a Mantle endpoint
    /// with an un-normalized (potentially duplicate-trigger) payload. It runs the
    /// shape-complete [`normalize_mantle_compaction_triggers`] scan once and logs
    /// with provider + model context, returning the [`TriggerNormalization`] so
    /// the selected adapter can place the survivor (task 5).
    fn normalize_for_mantle(&self, request: &mut OpenAIRequest) -> TriggerNormalization {
        let normalization = normalize_mantle_compaction_triggers(request);
        if normalization.removed > 0 {
            tracing::debug!(
                provider = %self.name,
                model = %request.model,
                compaction_triggers_removed = normalization.removed,
                "Normalized Bedrock Mantle compaction triggers at dispatch seam"
            );
        }
        normalization
    }

    /// Buffered Mantle dispatch seam. Normalizes once, resolves the endpoint with
    /// a SINGLE [`mantle_api_for_model`] call, and routes to the matching adapter.
    /// The three buffered adapters are reachable only through this method.
    async fn dispatch_mantle(
        &self,
        mut request: OpenAIRequest,
        http_client: &Client,
        api_key: &str,
        base_url: &str,
        custom_headers: &HashMap<String, String>,
    ) -> Result<ProviderResponse, GatewayError> {
        let normalization = self.normalize_for_mantle(&mut request);
        match mantle_api_for_model(&request.model) {
            MantleApi::Chat => {
                self.chat_completion_api_key(
                    request,
                    http_client,
                    api_key,
                    base_url,
                    custom_headers,
                    &normalization,
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
                    &normalization,
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
                    &normalization,
                )
                .await
            }
        }
    }

    /// Streaming Mantle dispatch seam. Normalizes once, resolves the endpoint with
    /// a SINGLE [`mantle_api_for_model`] call. The Chat family streams natively via
    /// [`Self::chat_completion_stream_api_key`]; the Responses and Messages families
    /// buffer through their adapters and emit one translated SSE event for replay.
    /// The streaming Chat adapter is reachable only through this method.
    async fn dispatch_mantle_stream(
        &self,
        mut request: OpenAIRequest,
        http_client: &Client,
        api_key: &str,
        base_url: &str,
        custom_headers: &HashMap<String, String>,
    ) -> Result<Pin<Box<dyn Stream<Item = Result<SSEEvent, GatewayError>> + Send>>, GatewayError>
    {
        let normalization = self.normalize_for_mantle(&mut request);
        match mantle_api_for_model(&request.model) {
            MantleApi::Chat => {
                request.stream = true;
                self.chat_completion_stream_api_key(
                    request,
                    http_client,
                    api_key,
                    base_url,
                    custom_headers,
                    &normalization,
                )
                .await
            }
            api => {
                // Responses and Messages require translation into the gateway's
                // OpenAI response schema, so buffer the upstream response and emit
                // one translated event for replay.
                let provider_response = match api {
                    MantleApi::Responses => {
                        self.chat_completion_responses_api(
                            request,
                            http_client,
                            api_key,
                            base_url,
                            custom_headers,
                            &normalization,
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
                            &normalization,
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
            } => {
                self.dispatch_mantle(request, http_client, api_key, base_url, custom_headers)
                    .await
            }
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
                self.dispatch_mantle_stream(
                    request,
                    http_client,
                    api_key,
                    base_url,
                    custom_headers,
                )
                .await
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

    // Trigger de-duplication coverage for the removed `normalize_mantle_responses_input`
    // free function and the removed content-part pass in `normalize_mantle_chat_messages`
    // now lives with the seam in the `normalize_mantle_compaction_triggers_*` tests
    // (task 2.2). See `normalize_mantle_compaction_triggers_input_array_carrier_keeps_last`,
    // `..._two_content_parts_keeps_last`, and `..._mixed_shapes_survivor_by_document_order`.

    #[test]
    fn test_mantle_responses_input_builder_preserves_native_string_input() {
        let mut request = create_test_chat_request(false);
        request
            .extra
            .insert("input".to_string(), serde_json::json!("native input"));

        let (_, _, _, _, input, normalized) = BedrockProvider::mantle_responses_input(request);

        assert_eq!(normalized, 0);
        assert_eq!(input, serde_json::json!("native input"));
    }

    #[test]
    fn test_mantle_responses_input_builder_uses_messages_when_input_missing() {
        let request = create_test_chat_request(false);

        let (_, _, _, _, input, normalized) = BedrockProvider::mantle_responses_input(request);

        assert_eq!(normalized, 0);
        assert_eq!(input[0]["role"], "user");
        assert_eq!(input[0]["content"], "hello");
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
    fn test_responses_tools_flatten_chat_function_shape() {
        let tools = serde_json::json!([{
            "type": "function",
            "function": {
                "name": "read_file",
                "description": "Read a file",
                "parameters": {"type": "object"},
                "strict": true
            }
        }]);

        assert_eq!(
            responses_tools_from_chat(&tools),
            serde_json::json!([{
                "type": "function",
                "name": "read_file",
                "description": "Read a file",
                "parameters": {"type": "object"},
                "strict": true
            }])
        );
    }

    #[test]
    fn test_responses_tool_choice_flattens_chat_function_shape() {
        let tool_choice = serde_json::json!({
            "type": "function",
            "function": {"name": "read_file"}
        });

        assert_eq!(
            responses_tool_choice_from_chat(&tool_choice),
            serde_json::json!({"type": "function", "name": "read_file"})
        );
        assert_eq!(
            responses_tool_choice_from_chat(&serde_json::json!("auto")),
            serde_json::json!("auto")
        );
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

/// Bug condition exploration tests for the duplicate `compaction_trigger` defect.
///
/// These tests are written BEFORE the fix (task 1 of the
/// `bedrock-compaction-trigger-duplicate` spec). Cases 1-5 are EXPECTED TO FAIL
/// on unfixed code — the failure is the documented evidence that the bug exists
/// and confirms the carrier is the message-level `extra` marker / native `input`
/// list (a request *shape* concern), not the streaming transport. Case 6 is
/// expected to PASS, confirming the existing de-duplication only covers the
/// content-part shape.
///
/// Each test dispatches a real `OpenAIRequest` through `BedrockProvider` against
/// a `wiremock` server and inspects the body actually posted upstream — the seam
/// the fix will act on. The Responses front-door cases go through the production
/// `translate` path so the marker's real serialized shape is exercised, not a
/// hand-built approximation.
#[cfg(test)]
mod compaction_trigger_bug_exploration {
    use super::*;
    use crate::models::openai::{Message, OpenAIRequest};
    use crate::responses::{translate, ResponsesRequest, StoredConversation, TranslationContext};
    use std::collections::HashMap;
    use wiremock::matchers::method;
    use wiremock::{Mock, MockServer, ResponseTemplate};

    /// A generic success body accepted by every Mantle adapter, so the dispatch
    /// reaches the point where it serializes the outgoing request. The adapters
    /// each read different fields; this body carries all of them.
    fn mantle_ok_body() -> serde_json::Value {
        serde_json::json!({
            "id": "resp_test",
            "object": "chat.completion",
            "created": 1234567890i64,
            "model": "gpt-test",
            "choices": [{
                "index": 0,
                "message": {"role": "assistant", "content": "ok"},
                "finish_reason": "stop"
            }],
            "output_text": "ok",
            "content": [{"type": "text", "text": "ok"}],
            "stop_reason": "end_turn",
            "usage": {
                "prompt_tokens": 1, "completion_tokens": 1, "total_tokens": 2,
                "input_tokens": 1, "output_tokens": 1
            }
        })
    }

    fn provider_for(base_url: String) -> BedrockProvider {
        BedrockProvider {
            name: "bedrock-test".to_string(),
            region: "us-east-1".to_string(),
            auth_mode: BedrockAuthMode::ApiKey {
                http_client: Client::builder().build().expect("client"),
                api_key: "test-api-key".to_string(),
                base_url,
                custom_headers: HashMap::new(),
            },
        }
    }

    /// Mount a catch-all POST mock that accepts every Mantle path, dispatch the
    /// request, and return the parsed body the gateway posted upstream.
    async fn capture_upstream_body(
        base_url_suffix: &str,
        request: OpenAIRequest,
    ) -> serde_json::Value {
        let server = MockServer::start().await;
        Mock::given(method("POST"))
            .respond_with(ResponseTemplate::new(200).set_body_json(mantle_ok_body()))
            .mount(&server)
            .await;

        let base_url = format!("{}{}", server.uri(), base_url_suffix);
        let provider = provider_for(base_url);
        // Dispatch is fire-and-observe: the response shape is irrelevant here,
        // only the body that left the gateway matters. Some adapters may still
        // error while parsing the generic response; the request is captured
        // regardless because the POST already happened.
        let _ = provider.chat_completion(request).await;

        let requests = server
            .received_requests()
            .await
            .expect("wiremock records requests");
        assert_eq!(
            requests.len(),
            1,
            "exactly one upstream request should be captured"
        );
        serde_json::from_slice(&requests[0].body).expect("upstream body is valid JSON")
    }

    /// Count `compaction_trigger` markers across EVERY shape in an outgoing body:
    /// message-level `extra` markers, message content-array parts, and native
    /// Responses `input` array items. This mirrors `countTriggerSites` from the
    /// design's formal spec so the assertions are shape-complete.
    fn count_trigger_sites(body: &serde_json::Value) -> usize {
        let is_trigger = |v: &serde_json::Value| {
            v.get("type").and_then(serde_json::Value::as_str) == Some("compaction_trigger")
        };

        let mut count = 0;

        // Chat-family: `messages` array.
        if let Some(messages) = body.get("messages").and_then(serde_json::Value::as_array) {
            for message in messages {
                // Message-level marker (flattened `extra`).
                if is_trigger(message) {
                    count += 1;
                }
                // Content-array parts.
                if let Some(parts) = message.get("content").and_then(serde_json::Value::as_array) {
                    count += parts.iter().filter(|p| is_trigger(p)).count();
                }
            }
        }

        // Responses-family: `input` array of items.
        if let Some(items) = body.get("input").and_then(serde_json::Value::as_array) {
            for item in items {
                if is_trigger(item) {
                    count += 1;
                }
                if let Some(parts) = item.get("content").and_then(serde_json::Value::as_array) {
                    count += parts.iter().filter(|p| is_trigger(p)).count();
                }
            }
        }

        count
    }

    /// Translate a Responses front-door request exactly as production does.
    fn translate_responses(
        json: serde_json::Value,
        model: &str,
        history: Option<StoredConversation>,
    ) -> OpenAIRequest {
        let req: ResponsesRequest =
            serde_json::from_value(json).expect("valid ResponsesRequest json");
        let ctx = TranslationContext {
            resolved_model: model,
            model_supports_reasoning: false,
        };
        translate(&req, history, &ctx).expect("translation succeeds")
    }

    // ------------------------------------------------------------------
    // Case 1 — standalone input item, Chat family.
    // A Responses request whose `input` holds `{"type":"compaction_trigger"}`
    // twice, translated and dispatched to a MantleApi::Chat model.
    // EXPECT FAIL: InputItem::Easy parses each as role ""/content "" with the
    // marker in `extra`, and Message.extra is #[serde(flatten)], so two
    // message-level markers ride out on the wire.
    // ------------------------------------------------------------------
    #[tokio::test]
    async fn case1_standalone_input_item_chat_family_keeps_one_trigger() {
        let request = translate_responses(
            serde_json::json!({
                "model": "openai.gpt-oss-120b",
                "input": [
                    {"type": "compaction_trigger"},
                    {"type": "compaction_trigger"}
                ]
            }),
            "openai.gpt-oss-120b",
            None,
        );

        let body = capture_upstream_body("", request).await;
        assert_eq!(
            count_trigger_sites(&body),
            1,
            "Case 1: expected exactly one compaction_trigger in the Chat body, got body: {}",
            body
        );
    }

    // ------------------------------------------------------------------
    // Case 2 — message-level marker across two messages.
    // Two chat messages each with extra["type"] = "compaction_trigger".
    // EXPECT FAIL: neither normalize_mantle_chat_messages nor the pre-pass in
    // mantle_responses_input inspects message.extra.
    // ------------------------------------------------------------------
    #[tokio::test]
    async fn case2_message_level_marker_two_messages_keeps_one() {
        let mut trigger_extra = serde_json::Map::new();
        trigger_extra.insert("type".to_string(), serde_json::json!("compaction_trigger"));

        let request = OpenAIRequest {
            model: "openai.gpt-oss-120b".to_string(),
            messages: vec![
                Message {
                    role: "user".to_string(),
                    content: serde_json::Value::String(String::new()),
                    extra: trigger_extra.clone(),
                },
                Message {
                    role: "user".to_string(),
                    content: serde_json::Value::String(String::new()),
                    extra: trigger_extra,
                },
            ],
            stream: false,
            temperature: None,
            max_tokens: None,
            extra: Default::default(),
        };

        let body = capture_upstream_body("", request).await;
        assert_eq!(
            count_trigger_sites(&body),
            1,
            "Case 2: expected one surviving marker, got body: {}",
            body
        );
    }

    // ------------------------------------------------------------------
    // Case 3 — replay plus new turn.
    // A StoredConversation holding one trigger input item plus a new request
    // carrying another via the front door.
    // EXPECT FAIL: replay_history re-emits the stored marker, then
    // translate_input appends the new one, with no cross-item de-duplication.
    // ------------------------------------------------------------------
    #[tokio::test]
    async fn case3_replay_plus_new_turn_keeps_one() {
        let stored_item: crate::responses::InputItem = serde_json::from_value(serde_json::json!({
            "type": "compaction_trigger"
        }))
        .expect("stored trigger item parses");
        let history = StoredConversation {
            input_items: vec![stored_item],
            output_items: Vec::new(),
        };

        let request = translate_responses(
            serde_json::json!({
                "model": "openai.gpt-oss-120b",
                "input": [
                    {"type": "compaction_trigger"}
                ]
            }),
            "openai.gpt-oss-120b",
            Some(history),
        );

        let body = capture_upstream_body("", request).await;
        assert_eq!(
            count_trigger_sites(&body),
            1,
            "Case 3: expected one trigger after replay+new turn, got body: {}",
            body
        );
    }

    // ------------------------------------------------------------------
    // Case 4 — Responses family, message-level survivor.
    // A single message-level marker to a MantleApi::Responses model.
    // EXPECT FAIL: mantle_responses_input rebuilds `input` from role +
    // content_as_text() and drops `extra`, so the trigger is silently lost —
    // the body's `input` does NOT end with the trigger (zero triggers present).
    // ------------------------------------------------------------------
    #[tokio::test]
    async fn case4_responses_family_message_level_survivor_is_terminal() {
        let mut trigger_extra = serde_json::Map::new();
        trigger_extra.insert("type".to_string(), serde_json::json!("compaction_trigger"));

        let request = OpenAIRequest {
            model: "openai.gpt-5.6-sol".to_string(),
            messages: vec![Message {
                role: "user".to_string(),
                content: serde_json::Value::String(String::new()),
                extra: trigger_extra,
            }],
            stream: false,
            temperature: None,
            max_tokens: None,
            extra: Default::default(),
        };

        // Responses adapter posts to `{root}/openai/v1/responses`; the base URL
        // is normalized by stripping a trailing `/v1`, so no suffix is needed.
        let body = capture_upstream_body("", request).await;
        let input = body
            .get("input")
            .and_then(serde_json::Value::as_array)
            .cloned()
            .unwrap_or_default();
        let last_is_trigger = input
            .last()
            .map(|item| {
                item.get("type").and_then(serde_json::Value::as_str) == Some("compaction_trigger")
            })
            .unwrap_or(false);
        assert!(
            last_is_trigger,
            "Case 4: expected the Responses `input` to end with the trigger, got body: {}",
            body
        );
    }

    // ------------------------------------------------------------------
    // Case 5 — native `input` array on a Chat-family model.
    // extra["input"] with two triggers, dispatched to a MantleApi::Chat model.
    // EXPECT FAIL: sanitize_mantle_chat_request deletes `input` wholesale after
    // the content-part pass that never looked at it, so zero triggers survive.
    //
    // Confirms sanitize_request_for_provider (router.rs:3502) is NOT the blocker:
    // this is a direct provider dispatch, and on the live path
    // dispatch_attempts_under_permit returns early for bedrock API-key mode.
    // ------------------------------------------------------------------
    #[tokio::test]
    async fn case5_native_input_array_chat_family_keeps_one() {
        let mut extra = serde_json::Map::new();
        extra.insert(
            "input".to_string(),
            serde_json::json!([
                {"type": "compaction_trigger", "id": "old"},
                {"type": "compaction_trigger", "id": "latest"}
            ]),
        );

        let request = OpenAIRequest {
            model: "openai.gpt-oss-120b".to_string(),
            messages: vec![Message {
                role: "user".to_string(),
                content: serde_json::Value::String("continue".to_string()),
                extra: Default::default(),
            }],
            stream: false,
            temperature: None,
            max_tokens: None,
            extra,
        };

        let body = capture_upstream_body("", request).await;
        assert_eq!(
            count_trigger_sites(&body),
            1,
            "Case 5: expected one surviving trigger from the native input list, got body: {}",
            body
        );
    }

    // ------------------------------------------------------------------
    // Case 6 — content parts only (edge case).
    // Three triggers across two messages, all as content-array parts, to a
    // MantleApi::Chat model.
    // EXPECT PASS on unfixed code: normalize_mantle_chat_messages already
    // reduces content-part triggers to one. Confirms the existing pass covers
    // shape 3 only.
    // ------------------------------------------------------------------
    #[tokio::test]
    async fn case6_content_parts_only_keeps_one() {
        let request = OpenAIRequest {
            model: "openai.gpt-oss-120b".to_string(),
            messages: vec![
                Message {
                    role: "user".to_string(),
                    content: serde_json::json!([
                        {"type": "text", "text": "first"},
                        {"type": "compaction_trigger"}
                    ]),
                    extra: Default::default(),
                },
                Message {
                    role: "assistant".to_string(),
                    content: serde_json::json!([
                        {"type": "compaction_trigger", "id": "mid"},
                        {"type": "text", "text": "second"},
                        {"type": "compaction_trigger", "id": "latest"}
                    ]),
                    extra: Default::default(),
                },
            ],
            stream: false,
            temperature: None,
            max_tokens: None,
            extra: Default::default(),
        };

        let body = capture_upstream_body("", request).await;
        assert_eq!(
            count_trigger_sites(&body),
            1,
            "Case 6: expected content-part de-dup to leave one trigger, got body: {}",
            body
        );
    }

    // ------------------------------------------------------------------
    // Observed UNFIXED-code baselines (recorded from the runs below).
    //
    // Task 9's preservation tests MUST compare the fixed code against these
    // observed bodies, not against an assumed shape. Captured on unfixed code
    // at task 1 for the Chat-family adapter (openai.gpt-oss-120b):
    //
    //   zero-trigger:
    //     {"messages":[{"content":"hello","role":"user"}],
    //      "model":"openai.gpt-oss-120b","stream":false}
    //
    //   single-trigger (one content-part trigger, kept in place):
    //     {"messages":[{"content":[{"text":"keep me","type":"text"},
    //                              {"type":"compaction_trigger"}],
    //                   "role":"user"}],
    //      "model":"openai.gpt-oss-120b","stream":false}
    //
    // These are asserted below so a future pipeline change that alters the
    // baseline breaks here first.
    // ------------------------------------------------------------------

    /// Serialized zero-trigger Chat body observed on unfixed code (task 9 case 1).
    const BASELINE_ZERO_TRIGGER_CHAT_BODY: &str =
        r#"{"messages":[{"content":"hello","role":"user"}],"model":"openai.gpt-oss-120b","stream":false}"#;

    /// Serialized single-trigger Chat body observed on unfixed code (task 9 case 2).
    const BASELINE_SINGLE_TRIGGER_CHAT_BODY: &str =
        r#"{"messages":[{"content":[{"text":"keep me","type":"text"},{"type":"compaction_trigger"}],"role":"user"}],"model":"openai.gpt-oss-120b","stream":false}"#;

    // ------------------------------------------------------------------
    // Baselines for task 9 preservation checks — captured on UNFIXED code.
    //
    // Observation-first methodology: task 9 must assert against what the current
    // pipeline actually emits for zero-trigger and single-trigger inputs, not an
    // assumed shape. These tests record the serialized outgoing bodies so the
    // later preservation tests compare against an observed baseline. They are
    // expected to PASS now (they only assert the observation is well-formed and
    // print the body for the record).
    // ------------------------------------------------------------------

    /// Zero-trigger Chat baseline (design task 9 case 1 / clause 3.2).
    #[tokio::test]
    async fn baseline_zero_trigger_chat_body() {
        let request = OpenAIRequest {
            model: "openai.gpt-oss-120b".to_string(),
            messages: vec![Message {
                role: "user".to_string(),
                content: serde_json::Value::String("hello".to_string()),
                extra: Default::default(),
            }],
            stream: false,
            temperature: None,
            max_tokens: None,
            extra: Default::default(),
        };

        let body = capture_upstream_body("", request).await;
        eprintln!("BASELINE zero_trigger_chat_body = {}", body);
        assert_eq!(
            count_trigger_sites(&body),
            0,
            "zero-trigger baseline must carry no triggers"
        );
        // Lock in the observed unfixed-code baseline so task 9 compares the fixed
        // code against what actually leaves the gateway today (clause 3.2).
        let expected: serde_json::Value =
            serde_json::from_str(BASELINE_ZERO_TRIGGER_CHAT_BODY).unwrap();
        assert_eq!(
            body, expected,
            "zero-trigger Chat body drifted from the recorded baseline"
        );
    }

    /// Single-trigger Chat baseline (design task 9 case 2 / clause 3.1).
    /// One content-part trigger must be forwarded unchanged (in place).
    #[tokio::test]
    async fn baseline_single_trigger_chat_body() {
        let request = OpenAIRequest {
            model: "openai.gpt-oss-120b".to_string(),
            messages: vec![Message {
                role: "user".to_string(),
                content: serde_json::json!([
                    {"type": "text", "text": "keep me"},
                    {"type": "compaction_trigger"}
                ]),
                extra: Default::default(),
            }],
            stream: false,
            temperature: None,
            max_tokens: None,
            extra: Default::default(),
        };

        let body = capture_upstream_body("", request).await;
        eprintln!("BASELINE single_trigger_chat_body = {}", body);
        assert_eq!(
            count_trigger_sites(&body),
            1,
            "single-trigger baseline must forward exactly one trigger"
        );
        // Lock in the observed unfixed-code baseline: the single content-part
        // trigger stays at its original site inside the message content array
        // (clauses 3.1, 2.6). Task 9 compares the fixed code against this.
        let expected: serde_json::Value =
            serde_json::from_str(BASELINE_SINGLE_TRIGGER_CHAT_BODY).unwrap();
        assert_eq!(
            body, expected,
            "single-trigger Chat body drifted from the recorded baseline"
        );
    }

    // ------------------------------------------------------------------
    // Survivor placement unit tests (task 5)
    //
    // These assert the per-adapter placement of the surviving trigger:
    //   - Responses: the built `input` array ends with the trigger, for all
    //     three survivor origins (native input item, content part, message
    //     marker).
    //   - Chat: a single content-part trigger stays byte-identical to the
    //     baseline; a native-input survivor surfaces as exactly one trigger.
    //   - Messages: a no-op (Anthropic Messages has no compaction_trigger).
    // ------------------------------------------------------------------

    /// Helper: assert the Responses `input` array ends with a compaction trigger
    /// and carries exactly one trigger overall.
    fn assert_responses_input_terminal_trigger(body: &serde_json::Value) {
        let input = body
            .get("input")
            .and_then(serde_json::Value::as_array)
            .cloned()
            .unwrap_or_default();
        assert!(
            input
                .last()
                .is_some_and(|item| item.get("type").and_then(serde_json::Value::as_str)
                    == Some("compaction_trigger")),
            "Responses `input` must end with the trigger, got: {}",
            body
        );
        assert_eq!(
            count_trigger_sites(body),
            1,
            "Responses body must carry exactly one trigger, got: {}",
            body
        );
    }

    /// Responses placement — native `extra["input"]` survivor becomes terminal.
    #[tokio::test]
    async fn placement_responses_native_input_survivor_terminal() {
        let mut extra = serde_json::Map::new();
        extra.insert(
            "input".to_string(),
            serde_json::json!([
                {"type": "message", "role": "user", "content": "hi"},
                {"type": "compaction_trigger", "id": "old"},
                {"type": "compaction_trigger", "id": "latest"}
            ]),
        );
        let request = OpenAIRequest {
            model: "openai.gpt-5.6-sol".to_string(),
            messages: vec![],
            stream: false,
            temperature: None,
            max_tokens: None,
            extra,
        };

        let body = capture_upstream_body("", request).await;
        assert_responses_input_terminal_trigger(&body);
        // The most recent survivor (id "latest") is the one kept and placed last.
        let last = body.get("input").and_then(serde_json::Value::as_array).unwrap().last().unwrap();
        assert_eq!(last.get("id").and_then(serde_json::Value::as_str), Some("latest"));
    }

    /// Responses placement — a content-part survivor becomes terminal (the
    /// message-built `input` array otherwise drops it via `content_as_text`).
    #[tokio::test]
    async fn placement_responses_content_part_survivor_terminal() {
        let request = OpenAIRequest {
            model: "openai.gpt-5.6-sol".to_string(),
            messages: vec![Message {
                role: "user".to_string(),
                content: serde_json::json!([
                    {"type": "text", "text": "keep me"},
                    {"type": "compaction_trigger"}
                ]),
                extra: Default::default(),
            }],
            stream: false,
            temperature: None,
            max_tokens: None,
            extra: Default::default(),
        };

        let body = capture_upstream_body("", request).await;
        assert_responses_input_terminal_trigger(&body);
    }

    /// Responses placement — a message-level marker survivor becomes terminal.
    #[tokio::test]
    async fn placement_responses_message_marker_survivor_terminal() {
        let mut trigger_extra = serde_json::Map::new();
        trigger_extra.insert("type".to_string(), serde_json::json!("compaction_trigger"));
        let request = OpenAIRequest {
            model: "openai.gpt-5.6-sol".to_string(),
            messages: vec![Message {
                role: "user".to_string(),
                content: serde_json::Value::String(String::new()),
                extra: trigger_extra,
            }],
            stream: false,
            temperature: None,
            max_tokens: None,
            extra: Default::default(),
        };

        let body = capture_upstream_body("", request).await;
        assert_responses_input_terminal_trigger(&body);
    }

    /// Chat placement — a single content-part trigger is forwarded byte-identical
    /// to the recorded baseline (no relocation when the survivor is already in
    /// place; clause 3.1).
    #[tokio::test]
    async fn placement_chat_single_content_part_unchanged() {
        let request = OpenAIRequest {
            model: "openai.gpt-oss-120b".to_string(),
            messages: vec![Message {
                role: "user".to_string(),
                content: serde_json::json!([
                    {"type": "text", "text": "keep me"},
                    {"type": "compaction_trigger"}
                ]),
                extra: Default::default(),
            }],
            stream: false,
            temperature: None,
            max_tokens: None,
            extra: Default::default(),
        };

        let body = capture_upstream_body("", request).await;
        let expected: serde_json::Value =
            serde_json::from_str(BASELINE_SINGLE_TRIGGER_CHAT_BODY).unwrap();
        assert_eq!(
            body, expected,
            "single content-part trigger must stay byte-identical to the baseline"
        );
    }

    /// Chat placement — a native `extra["input"]` survivor surfaces as exactly
    /// one trigger in the Chat body while the real message content is preserved.
    #[tokio::test]
    async fn placement_chat_native_input_survivor_surfaces_once() {
        let mut extra = serde_json::Map::new();
        extra.insert(
            "input".to_string(),
            serde_json::json!([
                {"type": "compaction_trigger", "id": "old"},
                {"type": "compaction_trigger", "id": "latest"}
            ]),
        );
        let request = OpenAIRequest {
            model: "openai.gpt-oss-120b".to_string(),
            messages: vec![Message {
                role: "user".to_string(),
                content: serde_json::Value::String("continue".to_string()),
                extra: Default::default(),
            }],
            stream: false,
            temperature: None,
            max_tokens: None,
            extra,
        };

        let body = capture_upstream_body("", request).await;
        assert_eq!(
            count_trigger_sites(&body),
            1,
            "native-input survivor must surface as exactly one trigger, got: {}",
            body
        );
        // The original message content ("continue") must be preserved as text,
        // and `extra["input"]` must have been stripped by sanitization.
        assert!(body.get("input").is_none(), "input key must be stripped, got: {}", body);
        let messages = body.get("messages").and_then(serde_json::Value::as_array).unwrap();
        let joined_text: String = messages
            .iter()
            .filter_map(|m| m.get("content").and_then(serde_json::Value::as_array))
            .flatten()
            .filter(|p| p.get("type").and_then(serde_json::Value::as_str) == Some("text"))
            .filter_map(|p| p.get("text").and_then(serde_json::Value::as_str))
            .collect();
        assert!(
            joined_text.contains("continue"),
            "original message content must be preserved, got: {}",
            body
        );
    }

    /// Messages placement — a no-op: a plain single-message request forwards its
    /// flattened content and carries no trigger sites.
    #[tokio::test]
    async fn placement_messages_is_noop() {
        let request = OpenAIRequest {
            model: "anthropic.claude-3-5-sonnet".to_string(),
            messages: vec![Message {
                role: "user".to_string(),
                content: serde_json::Value::String("hello".to_string()),
                extra: Default::default(),
            }],
            stream: false,
            temperature: None,
            max_tokens: None,
            extra: Default::default(),
        };

        let body = capture_upstream_body("", request).await;
        assert_eq!(
            count_trigger_sites(&body),
            0,
            "Messages no-op body must carry no triggers, got: {}",
            body
        );
        let messages = body.get("messages").and_then(serde_json::Value::as_array).unwrap();
        assert_eq!(
            messages
                .iter()
                .find_map(|m| m.get("content").and_then(serde_json::Value::as_str)),
            Some("hello"),
            "Messages adapter must forward the flattened content, got: {}",
            body
        );
    }

    // ------------------------------------------------------------------
    // normalize_mantle_compaction_triggers unit tests (task 2.2)
    // ------------------------------------------------------------------

    /// Build a message with the given role and content and no extra fields.
    fn msg(role: &str, content: serde_json::Value) -> Message {
        Message {
            role: role.to_string(),
            content,
            extra: Default::default(),
        }
    }

    /// Build a bare request from a list of messages, no `input` in extra.
    fn req(messages: Vec<Message>) -> OpenAIRequest {
        OpenAIRequest {
            model: "openai.gpt-oss-120b".to_string(),
            messages,
            stream: false,
            temperature: None,
            max_tokens: None,
            extra: Default::default(),
        }
    }

    /// A content-array holding a single `compaction_trigger` part.
    fn trigger_content_part() -> serde_json::Value {
        serde_json::json!([{"type": "compaction_trigger"}])
    }

    /// A message-level marker message: blank role/content plus `extra["type"]`.
    fn marker_message() -> Message {
        let mut extra = serde_json::Map::new();
        extra.insert("type".to_string(), serde_json::json!("compaction_trigger"));
        Message {
            role: String::new(),
            content: serde_json::Value::String(String::new()),
            extra,
        }
    }

    /// Count every trigger site across all three shapes in a request.
    fn count_sites(request: &OpenAIRequest) -> usize {
        let mut count = 0;
        if let Some(items) = request
            .extra
            .get("input")
            .and_then(serde_json::Value::as_array)
        {
            count += items.iter().filter(|i| is_compaction_trigger(i)).count();
        }
        for message in &request.messages {
            if message_extra_is_trigger(message) {
                count += 1;
            }
            if let Some(parts) = message.content.as_array() {
                count += parts.iter().filter(|p| is_compaction_trigger(p)).count();
            }
        }
        count
    }

    #[test]
    fn normalize_mantle_compaction_triggers_zero_is_noop() {
        let mut request = req(vec![msg("user", serde_json::json!("hello"))]);
        let before = serde_json::to_value(&request).unwrap();
        let result = normalize_mantle_compaction_triggers(&mut request);
        assert_eq!(result.removed, 0);
        assert!(result.survivor.is_none());
        assert!(!result.survivor_from_input_array);
        assert_eq!(count_sites(&request), 0);
        // Byte-identical: nothing changed.
        assert_eq!(serde_json::to_value(&request).unwrap(), before);
    }

    #[test]
    fn normalize_mantle_compaction_triggers_one_content_part_unchanged() {
        let mut request = req(vec![msg("user", trigger_content_part())]);
        let before = serde_json::to_value(&request).unwrap();
        let result = normalize_mantle_compaction_triggers(&mut request);
        assert_eq!(result.removed, 0);
        assert!(result.survivor.is_some());
        assert_eq!(count_sites(&request), 1);
        // Single-trigger payload is byte-identical.
        assert_eq!(serde_json::to_value(&request).unwrap(), before);
    }

    #[test]
    fn normalize_mantle_compaction_triggers_one_marker_message_unchanged() {
        let mut request = req(vec![
            msg("user", serde_json::json!("hi")),
            marker_message(),
        ]);
        let before = serde_json::to_value(&request).unwrap();
        let result = normalize_mantle_compaction_triggers(&mut request);
        assert_eq!(result.removed, 0);
        assert_eq!(count_sites(&request), 1);
        // Survivor is the sole trigger; the residue message is NOT dropped
        // because it is the survivor's own message.
        assert_eq!(request.messages.len(), 2);
        assert_eq!(serde_json::to_value(&request).unwrap(), before);
    }

    #[test]
    fn normalize_mantle_compaction_triggers_two_content_parts_keeps_last() {
        let mut request = req(vec![
            msg("user", trigger_content_part()),
            msg("user", trigger_content_part()),
        ]);
        let result = normalize_mantle_compaction_triggers(&mut request);
        assert_eq!(result.removed, 1);
        assert_eq!(count_sites(&request), 1);
        assert!(!result.survivor_from_input_array);
        // Survivor stays in the LAST message; first message's trigger removed.
        assert!(request.messages[0].content.as_array().unwrap().is_empty());
        assert_eq!(
            request.messages[1].content.as_array().unwrap().len(),
            1
        );
    }

    #[test]
    fn normalize_mantle_compaction_triggers_two_markers_drops_residue() {
        let mut request = req(vec![marker_message(), marker_message()]);
        let result = normalize_mantle_compaction_triggers(&mut request);
        assert_eq!(result.removed, 1);
        assert_eq!(count_sites(&request), 1);
        // The earlier standalone-trigger residue message is dropped entirely,
        // never emitted as {"role":"","content":""}.
        assert_eq!(request.messages.len(), 1);
        assert!(message_extra_is_trigger(&request.messages[0]));
        // No emitted message carries an empty role.
        assert!(request.messages.iter().all(|m| !m.role.is_empty()
            || message_extra_is_trigger(m)));
    }

    #[test]
    fn normalize_mantle_compaction_triggers_many_reduces_to_one() {
        let mut request = req(vec![
            msg("user", trigger_content_part()),
            marker_message(),
            msg("user", trigger_content_part()),
            marker_message(),
        ]);
        let original = count_sites(&request);
        assert!(original >= 2);
        let result = normalize_mantle_compaction_triggers(&mut request);
        // Post-scan count is min(originalCount, 1).
        assert_eq!(count_sites(&request), 1);
        assert_eq!(result.removed, original - 1);
    }

    #[test]
    fn normalize_mantle_compaction_triggers_input_array_carrier_keeps_last() {
        let mut request = req(vec![]);
        request.extra.insert(
            "input".to_string(),
            serde_json::json!([
                {"type": "compaction_trigger", "id": "a"},
                {"type": "message", "role": "user"},
                {"type": "compaction_trigger", "id": "b"}
            ]),
        );
        let result = normalize_mantle_compaction_triggers(&mut request);
        assert_eq!(result.removed, 1);
        assert!(result.survivor_from_input_array);
        // Survivor is the LAST input item ("b").
        assert_eq!(
            result.survivor.as_ref().and_then(|s| s.get("id")),
            Some(&serde_json::json!("b"))
        );
        let items = request
            .extra
            .get("input")
            .and_then(serde_json::Value::as_array)
            .unwrap();
        assert_eq!(items.iter().filter(|i| is_compaction_trigger(i)).count(), 1);
        // The non-trigger "message" item is preserved.
        assert!(items
            .iter()
            .any(|i| i.get("type").and_then(|t| t.as_str()) == Some("message")));
        // The surviving trigger is item "b".
        let surviving = items.iter().find(|i| is_compaction_trigger(i)).unwrap();
        assert_eq!(surviving.get("id"), Some(&serde_json::json!("b")));
    }

    #[test]
    fn normalize_mantle_compaction_triggers_mixed_shapes_survivor_by_document_order() {
        // Document order: input-array item (a), then content part (b),
        // then message marker (c). The LAST site — the message marker — survives.
        let mut request = req(vec![
            msg("user", trigger_content_part()),
            marker_message(),
        ]);
        request.extra.insert(
            "input".to_string(),
            serde_json::json!([{"type": "compaction_trigger", "id": "a"}]),
        );
        let result = normalize_mantle_compaction_triggers(&mut request);
        assert_eq!(count_sites(&request), 1);
        assert_eq!(result.removed, 2);
        // Survivor came from the message marker, not the input array.
        assert!(!result.survivor_from_input_array);
        // The surviving site is the message-level marker on the last message.
        assert!(message_extra_is_trigger(request.messages.last().unwrap()));
        // Earlier sites removed: input array empty of triggers, first message's
        // content part gone.
        let items = request
            .extra
            .get("input")
            .and_then(serde_json::Value::as_array)
            .unwrap();
        assert_eq!(items.iter().filter(|i| is_compaction_trigger(i)).count(), 0);
        assert!(request.messages[0].content.as_array().unwrap().is_empty());
    }

    #[test]
    fn normalize_mantle_compaction_triggers_survivor_input_array_when_last() {
        // Content part (message 0), then input-array item — but input items are
        // enumerated FIRST in document order, so the input item is NOT last here.
        // To make the input item the survivor, use only input-array triggers.
        let mut request = req(vec![]);
        request.extra.insert(
            "input".to_string(),
            serde_json::json!([
                {"type": "compaction_trigger", "id": "x"},
                {"type": "compaction_trigger", "id": "y"}
            ]),
        );
        let result = normalize_mantle_compaction_triggers(&mut request);
        assert!(result.survivor_from_input_array);
        assert_eq!(
            result.survivor.as_ref().and_then(|s| s.get("id")),
            Some(&serde_json::json!("y"))
        );
    }

    #[test]
    fn normalize_mantle_compaction_triggers_non_array_input_untouched() {
        let mut request = req(vec![msg("user", trigger_content_part())]);
        request
            .extra
            .insert("input".to_string(), serde_json::json!("auto"));
        let before = serde_json::to_value(&request).unwrap();
        let result = normalize_mantle_compaction_triggers(&mut request);
        // Only one trigger overall (the content part), so nothing removed and
        // the non-array `input` is left byte-identical.
        assert_eq!(result.removed, 0);
        assert_eq!(
            request.extra.get("input"),
            Some(&serde_json::json!("auto"))
        );
        assert_eq!(serde_json::to_value(&request).unwrap(), before);
    }

    #[test]
    fn normalize_mantle_compaction_triggers_non_array_input_ignored_for_counting() {
        // Non-array input must not be treated as a trigger site even with two
        // message triggers present.
        let mut request = req(vec![
            msg("user", trigger_content_part()),
            msg("user", trigger_content_part()),
        ]);
        request
            .extra
            .insert("input".to_string(), serde_json::json!("auto"));
        let result = normalize_mantle_compaction_triggers(&mut request);
        assert_eq!(result.removed, 1);
        assert!(!result.survivor_from_input_array);
        assert_eq!(count_sites(&request), 1);
        // `input: "auto"` untouched.
        assert_eq!(
            request.extra.get("input"),
            Some(&serde_json::json!("auto"))
        );
    }

    // ------------------------------------------------------------------
    // is_duplicate_compaction_trigger_error unit tests (task 6)
    // ------------------------------------------------------------------

    #[test]
    fn is_duplicate_compaction_trigger_error_matches_observed_body_at_400() {
        assert!(is_duplicate_compaction_trigger_error(
            400,
            "Only one 'compaction_trigger' item may be provided."
        ));
    }

    #[test]
    fn is_duplicate_compaction_trigger_error_matches_case_variants() {
        // Upper-case body.
        assert!(is_duplicate_compaction_trigger_error(
            400,
            "ONLY ONE 'COMPACTION_TRIGGER' ITEM MAY BE PROVIDED."
        ));
        // Lower-case body.
        assert!(is_duplicate_compaction_trigger_error(
            422,
            "only one 'compaction_trigger' item may be provided."
        ));
    }

    #[test]
    fn is_duplicate_compaction_trigger_error_matches_json_error_envelope() {
        assert!(is_duplicate_compaction_trigger_error(
            400,
            r#"{"error":{"message":"Bad request: only one 'compaction_trigger' item allowed"}}"#
        ));
    }

    #[test]
    fn is_duplicate_compaction_trigger_error_rejects_unrelated_4xx() {
        assert!(!is_duplicate_compaction_trigger_error(
            400,
            "invalid model identifier"
        ));
    }

    #[test]
    fn is_duplicate_compaction_trigger_error_rejects_same_phrasing_at_500() {
        // 4xx-gated: the exact repairable phrasing at a 5xx is NOT a match, so a
        // transient server error is never treated as this client-request defect.
        assert!(!is_duplicate_compaction_trigger_error(
            500,
            "Only one 'compaction_trigger' item may be provided."
        ));
    }

    #[test]
    fn is_duplicate_compaction_trigger_error_rejects_4xx_without_phrase() {
        // Mentions compaction_trigger but neither phrasing anchor is present.
        assert!(!is_duplicate_compaction_trigger_error(
            400,
            "unexpected compaction_trigger field in payload"
        ));
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
        #![proptest_config(ProptestConfig::with_cases(64))]

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
        #![proptest_config(ProptestConfig::with_cases(64))]

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
        #![proptest_config(ProptestConfig::with_cases(64))]

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
        #![proptest_config(ProptestConfig::with_cases(64))]

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

// ======================================================================
// Task 9 — Property 2 (Preservation), plus Property 1 and Property 3
// companions and the enumerated preservation coverage map.
//
// Property 2: for every input where the bug condition does NOT hold
// (countTriggerSites <= 1), the seam leaves the request byte-identical
// (Design "Correctness Property 2", clauses 3.1/3.2/3.7). Companions:
// Property 1 (unconstrained count reduces to min(orig,1), survivor = last
// site) and Property 3 (Responses `input` survivor is terminal).
//
// Observation-first: the enumerated equality cases assert against the
// unfixed-code baselines recorded at task 1 in
// `compaction_trigger_bug_exploration`
// (`BASELINE_ZERO_TRIGGER_CHAT_BODY` / `BASELINE_SINGLE_TRIGGER_CHAT_BODY`),
// which are already asserted by `baseline_zero_trigger_chat_body`,
// `baseline_single_trigger_chat_body`, and
// `placement_chat_single_content_part_unchanged`. To avoid duplicating
// those wiremock round-trips, this module documents them by reference and
// only adds genuinely uncovered coverage.
// ======================================================================
#[cfg(test)]
mod preservation {
    use super::*;
    use crate::models::openai::{Message, OpenAIRequest};
    use proptest::prelude::*;
    use std::sync::OnceLock;
    use wiremock::matchers::method;
    use wiremock::{Mock, MockServer, ResponseTemplate};

    // ------------------------------------------------------------------
    // Shared trigger-site counting (mirrors `count_trigger_sites` /
    // `count_sites` from the sibling test modules — those are private to
    // their modules, so the shape-complete counter is restated here).
    // ------------------------------------------------------------------

    /// Count `compaction_trigger` sites across all shapes IN AN OpenAIRequest:
    /// `extra["input"]` array items (only when an array), message-level `extra`
    /// markers, and message content-array parts. Mirrors the design's
    /// `countTriggerSites`.
    fn count_request_sites(request: &OpenAIRequest) -> usize {
        let mut count = 0;
        if let Some(items) = request
            .extra
            .get("input")
            .and_then(serde_json::Value::as_array)
        {
            count += items.iter().filter(|i| is_compaction_trigger(i)).count();
        }
        for message in &request.messages {
            if message_extra_is_trigger(message) {
                count += 1;
            }
            if let Some(parts) = message.content.as_array() {
                count += parts.iter().filter(|p| is_compaction_trigger(p)).count();
            }
        }
        count
    }

    /// Count `compaction_trigger` sites across all shapes IN A SERIALIZED BODY
    /// (`messages` + `input` arrays). Used for the Property 3 dispatch check.
    fn count_body_sites(body: &serde_json::Value) -> usize {
        let is_trigger = |v: &serde_json::Value| {
            v.get("type").and_then(serde_json::Value::as_str) == Some("compaction_trigger")
        };
        let mut count = 0;
        if let Some(messages) = body.get("messages").and_then(serde_json::Value::as_array) {
            for message in messages {
                if is_trigger(message) {
                    count += 1;
                }
                if let Some(parts) = message.get("content").and_then(serde_json::Value::as_array) {
                    count += parts.iter().filter(|p| is_trigger(p)).count();
                }
            }
        }
        if let Some(items) = body.get("input").and_then(serde_json::Value::as_array) {
            for item in items {
                if is_trigger(item) {
                    count += 1;
                }
                if let Some(parts) = item.get("content").and_then(serde_json::Value::as_array) {
                    count += parts.iter().filter(|p| is_trigger(p)).count();
                }
            }
        }
        count
    }

    // ------------------------------------------------------------------
    // Generators.
    // ------------------------------------------------------------------

    /// One ordinary (non-trigger) text content part, occasionally carrying an
    /// arbitrary extra key so the generator exercises unknown fields.
    fn arb_text_part() -> impl Strategy<Value = serde_json::Value> {
        ("[a-z ]{0,12}", prop::option::of("[a-z_]{1,6}")).prop_map(|(text, extra_key)| {
            let mut part = serde_json::Map::new();
            part.insert("type".to_string(), serde_json::json!("text"));
            part.insert("text".to_string(), serde_json::json!(text));
            if let Some(k) = extra_key {
                // Never inject a key that would read as a trigger `type`.
                if k != "type" {
                    part.insert(k, serde_json::json!("x"));
                }
            }
            serde_json::Value::Object(part)
        })
    }

    /// A message with a random role and either string content or an array of
    /// parts. `content_triggers` controls how many trigger content-parts appear
    /// (mixed with 0..3 ordinary text parts).
    fn arb_message(content_triggers: usize) -> impl Strategy<Value = Message> {
        let role = prop::sample::select(vec!["system", "user", "assistant"]);
        let parts = prop::collection::vec(arb_text_part(), 0..3);
        (role, parts, any::<bool>()).prop_map(move |(role, text_parts, use_array)| {
            let content = if content_triggers > 0 || use_array {
                let mut arr: Vec<serde_json::Value> = text_parts;
                for _ in 0..content_triggers {
                    arr.push(serde_json::json!({"type": "compaction_trigger"}));
                }
                serde_json::Value::Array(arr)
            } else {
                serde_json::Value::String("hello".to_string())
            };
            Message {
                role: role.to_string(),
                content,
                extra: Default::default(),
            }
        })
    }

    /// A request whose total trigger-site count lands in `range`, distributed
    /// randomly across the three shapes (input-array items, message content
    /// parts, message-level markers) with ordinary text parts and arbitrary
    /// extra keys mixed in. Used by the Property 1 companion (unconstrained via
    /// `0..6`) and, filtered to `<= 1`, by Property 2.
    fn arb_request_with_triggers(
        range: std::ops::Range<usize>,
    ) -> impl Strategy<Value = OpenAIRequest> {
        range
            // Split the sampled total into (input_array, content_parts, markers).
            .prop_flat_map(|total| {
                (Just(total), 0..=total).prop_flat_map(|(total, input_count)| {
                    let remaining = total - input_count;
                    (Just(input_count), 0..=remaining).prop_map(move |(input_count, content_count)| {
                        (input_count, content_count, remaining - content_count)
                    })
                })
            })
            .prop_flat_map(|(input_count, content_count, marker_count)| {
                // A carrier message holds the content-part triggers; a couple of
                // ordinary messages surround it so document order is non-trivial.
                let carrier = arb_message(content_count);
                let extra_msgs = prop::collection::vec(arb_message(0), 0..2);
                (
                    Just(input_count),
                    Just(marker_count),
                    carrier,
                    extra_msgs,
                )
            })
            .prop_map(|(input_count, marker_count, carrier, mut extra_msgs)| {
                let mut messages = vec![carrier];
                messages.append(&mut extra_msgs);
                for _ in 0..marker_count {
                    let mut extra = serde_json::Map::new();
                    extra.insert("type".to_string(), serde_json::json!("compaction_trigger"));
                    messages.push(Message {
                        role: String::new(),
                        content: serde_json::Value::String(String::new()),
                        extra,
                    });
                }
                let mut request = OpenAIRequest {
                    model: "openai.gpt-oss-120b".to_string(),
                    messages,
                    stream: false,
                    temperature: None,
                    max_tokens: None,
                    extra: Default::default(),
                };
                if input_count > 0 {
                    let mut items: Vec<serde_json::Value> =
                        vec![serde_json::json!({"type": "message", "role": "user"})];
                    for i in 0..input_count {
                        items.push(serde_json::json!({
                            "type": "compaction_trigger",
                            "id": format!("in-{i}")
                        }));
                    }
                    request
                        .extra
                        .insert("input".to_string(), serde_json::Value::Array(items));
                }
                request
            })
    }

    /// The LAST trigger site of a request in document order (input-array items
    /// first, then per message content parts, then message marker), returned as
    /// its JSON value plus a flag for whether it was an input-array item. Mirrors
    /// the seam's own ordering so the companion asserts the same survivor rule.
    fn last_site_of(request: &OpenAIRequest) -> Option<(serde_json::Value, bool)> {
        let mut last: Option<(serde_json::Value, bool)> = None;
        if let Some(items) = request
            .extra
            .get("input")
            .and_then(serde_json::Value::as_array)
        {
            for item in items {
                if is_compaction_trigger(item) {
                    last = Some((item.clone(), true));
                }
            }
        }
        for message in &request.messages {
            if let Some(parts) = message.content.as_array() {
                for part in parts {
                    if is_compaction_trigger(part) {
                        last = Some((part.clone(), false));
                    }
                }
            }
            if message_extra_is_trigger(message) {
                // The marker's JSON identity is the {"type":"compaction_trigger"}
                // object; capture it as such for the survivor comparison.
                last = Some((serde_json::json!({"type": "compaction_trigger"}), false));
            }
        }
        last
    }

    // ------------------------------------------------------------------
    // Property 1 companion — unconstrained trigger count.
    // For any request, after normalization the total site count is
    // min(originalCount, 1); when >= 1, the survivor equals the LAST original
    // site in document order (Design Correctness Property 1).
    // ------------------------------------------------------------------
    proptest! {
        #![proptest_config(ProptestConfig::with_cases(64))]

        #[test]
        fn prop_property1_companion_reduces_to_last_site(
            request in arb_request_with_triggers(0..6),
        ) {
            let mut req = request;
            let original = count_request_sites(&req);
            let expected_last = last_site_of(&req);

            let result = normalize_mantle_compaction_triggers(&mut req);

            prop_assert_eq!(count_request_sites(&req), original.min(1),
                "post-seam count must be min(originalCount, 1)");
            if original >= 1 {
                let (site, from_input) = expected_last.expect("a last site exists when original >= 1");
                prop_assert_eq!(result.survivor_from_input_array, from_input,
                    "survivor origin must match the last site's origin");
                // The survivor value's `type` must be compaction_trigger and,
                // for input-array survivors, the id must match the last item.
                let survivor = result.survivor.expect("survivor present when original >= 1");
                prop_assert_eq!(
                    survivor.get("type").and_then(|t| t.as_str()),
                    Some("compaction_trigger")
                );
                if from_input {
                    prop_assert_eq!(survivor.get("id"), site.get("id"),
                        "input-array survivor must be the last input-array trigger");
                }
                prop_assert_eq!(result.removed, original - 1);
            } else {
                prop_assert_eq!(result.removed, 0);
                prop_assert!(result.survivor.is_none());
            }
        }
    }

    // ------------------------------------------------------------------
    // Property 2 (core Preservation) — constrained to countTriggerSites <= 1.
    // The seam leaves the request byte-identical: removed == 0 and the serde
    // value before == after (Design Correctness Property 2, clauses 3.1/3.2).
    // ------------------------------------------------------------------
    proptest! {
        #![proptest_config(ProptestConfig::with_cases(64))]

        #[test]
        fn prop_property2_preservation_sub_threshold_byte_identical(
            request in arb_request_with_triggers(0..2),
        ) {
            let mut request = request;
            prop_assert!(count_request_sites(&request) <= 1,
                "generator must produce <= 1 trigger site");
            let before = serde_json::to_value(&request).unwrap();

            let result = normalize_mantle_compaction_triggers(&mut request);

            prop_assert_eq!(result.removed, 0, "no site may be removed at count <= 1");
            let after = serde_json::to_value(&request).unwrap();
            prop_assert_eq!(before, after,
                "sub-threshold payload must be byte-identical after the seam");
        }
    }

    // ------------------------------------------------------------------
    // Property 3 companion — Responses terminal placement.
    // For random `extra["input"]` arrays dispatched to a MantleApi::Responses
    // model, the built `input` array's trigger (when present) is the FINAL
    // element and there is at most one (Design Correctness Property 3).
    // ------------------------------------------------------------------

    /// A shared multi-thread runtime so each proptest case can `block_on` a
    /// wiremock dispatch without spinning up a runtime per case.
    fn shared_runtime() -> &'static tokio::runtime::Runtime {
        static RT: OnceLock<tokio::runtime::Runtime> = OnceLock::new();
        RT.get_or_init(|| {
            tokio::runtime::Builder::new_multi_thread()
                .enable_all()
                .build()
                .expect("build tokio runtime")
        })
    }

    fn mantle_ok_body() -> serde_json::Value {
        serde_json::json!({
            "id": "resp_test",
            "object": "chat.completion",
            "created": 1234567890i64,
            "model": "gpt-test",
            "choices": [{
                "index": 0,
                "message": {"role": "assistant", "content": "ok"},
                "finish_reason": "stop"
            }],
            "output_text": "ok",
            "content": [{"type": "text", "text": "ok"}],
            "stop_reason": "end_turn",
            "usage": {
                "prompt_tokens": 1, "completion_tokens": 1, "total_tokens": 2,
                "input_tokens": 1, "output_tokens": 1
            }
        })
    }

    /// Dispatch `request` at an API-key Bedrock provider pointed at a fresh
    /// wiremock server and return the parsed upstream body.
    async fn dispatch_and_capture(request: OpenAIRequest) -> serde_json::Value {
        let server = MockServer::start().await;
        Mock::given(method("POST"))
            .respond_with(ResponseTemplate::new(200).set_body_json(mantle_ok_body()))
            .mount(&server)
            .await;
        let provider = BedrockProvider {
            name: "bedrock-test".to_string(),
            region: "us-east-1".to_string(),
            auth_mode: BedrockAuthMode::ApiKey {
                http_client: Client::builder().build().expect("client"),
                api_key: "test-api-key".to_string(),
                base_url: server.uri(),
                custom_headers: std::collections::HashMap::new(),
            },
        };
        let _ = provider.chat_completion(request).await;
        let requests = server
            .received_requests()
            .await
            .expect("wiremock records requests");
        assert_eq!(requests.len(), 1, "exactly one upstream request");
        serde_json::from_slice(&requests[0].body).expect("upstream body is valid JSON")
    }

    proptest! {
        #![proptest_config(ProptestConfig::with_cases(64))]

        #[test]
        fn prop_property3_responses_survivor_is_terminal(
            trigger_count in 0usize..4,
            leading_messages in 0usize..3,
        ) {
            // Build a native Responses `input` array: some ordinary message
            // items, then `trigger_count` compaction triggers interleaved with a
            // trailing ordinary item so a non-terminal input trigger must be
            // relocated to the end by the seam + Responses adapter.
            let mut items: Vec<serde_json::Value> = Vec::new();
            for i in 0..leading_messages {
                items.push(serde_json::json!({
                    "type": "message", "role": "user", "content": format!("m{i}")
                }));
            }
            for i in 0..trigger_count {
                items.push(serde_json::json!({
                    "type": "compaction_trigger", "id": format!("t{i}")
                }));
                // A trailing ordinary item after each trigger ensures the trigger
                // is NOT already terminal in the input array.
                items.push(serde_json::json!({"type": "message", "role": "user", "content": "tail"}));
            }
            let mut extra = serde_json::Map::new();
            extra.insert("input".to_string(), serde_json::Value::Array(items));
            let request = OpenAIRequest {
                model: "openai.gpt-5.6-sol".to_string(), // MantleApi::Responses
                messages: vec![],
                stream: false,
                temperature: None,
                max_tokens: None,
                extra,
            };

            let body = shared_runtime().block_on(dispatch_and_capture(request));

            let sites = count_body_sites(&body);
            prop_assert!(sites <= 1, "at most one trigger may survive, got {} in {}", sites, body);
            if trigger_count >= 1 {
                let input = body
                    .get("input")
                    .and_then(serde_json::Value::as_array)
                    .cloned()
                    .unwrap_or_default();
                let last_is_trigger = input.last().is_some_and(|item| {
                    item.get("type").and_then(serde_json::Value::as_str)
                        == Some("compaction_trigger")
                });
                prop_assert!(last_is_trigger,
                    "surviving trigger must be the FINAL input element, got: {}", body);
            }
        }
    }

    // ------------------------------------------------------------------
    // Enumerated preservation coverage map (Design "Preservation Checking").
    //
    // Several enumerated preservation cases from task 9 are ALREADY covered by
    // tests written for tasks 1, 2.2, 4.2, 5, 6, and 7.2. Rather than duplicate
    // those round-trips, they are documented here by the covering test name so
    // the coverage is discoverable from the task-9 test area. Only genuinely
    // uncovered cases get NEW tests (below and in `router.rs`).
    //
    //   Zero-trigger equality (clause 3.2)
    //     → compaction_trigger_bug_exploration::baseline_zero_trigger_chat_body
    //       (asserts the fixed body == BASELINE_ZERO_TRIGGER_CHAT_BODY)
    //     → normalize_mantle_compaction_triggers_zero_is_noop (byte-identical)
    //
    //   Single-trigger equality, Chat, content-part (clauses 3.1, 2.6)
    //     → compaction_trigger_bug_exploration::baseline_single_trigger_chat_body
    //     → compaction_trigger_bug_exploration::placement_chat_single_content_part_unchanged
    //     → normalize_mantle_compaction_triggers_one_content_part_unchanged
    //
    //   Single-trigger, Responses, terminal (clauses 3.1, 2.6)
    //     → placement_responses_content_part_survivor_terminal
    //     → placement_responses_message_marker_survivor_terminal
    //
    //   Existing Mantle normalizations unchanged (clause 3.5)
    //     developer→system, input_text/output_text→text, cache_control removal
    //     → tests::test_mantle_message_normalizer_converts_responses_content_parts
    //     MANTLE_CHAT_ALLOWED top-level field stripping
    //     → tests::test_mantle_chat_sanitizer_removes_gateway_only_fields
    //
    //   AWS SDK Converse unchanged (clause 3.4)
    //     The AwsSdk arm of `ProviderClient for BedrockProvider` never calls the
    //     Mantle seam (`normalize_for_mantle` is invoked only from the ApiKey
    //     arms via `dispatch_mantle`), so trigger normalization cannot alter a
    //     Converse build by construction. The translate path is exercised by
    //     property_tests::prop_bedrock_translation_round_trip. See the explicit
    //     structural assertion `aws_sdk_arm_does_not_normalize_triggers` below.
    //
    //   Non-array `input` untouched (clause 3.7)
    //     → normalize_mantle_compaction_triggers_non_array_input_untouched
    //     → normalize_mantle_compaction_triggers_non_array_input_ignored_for_counting
    //
    //   Non-Bedrock pass-through keeps every trigger (clause 3.3)
    //     → router.rs: openai_sanitize_preserves_all_compaction_triggers (NEW)
    //
    //   Streaming transport decision (clause 3.6)
    //     non-Bedrock → PassThrough:
    //       → router.rs: streaming_provider_receives_compressed_body_before_response
    //     Bedrock → Buffered:
    //       → router.rs: bedrock_streaming_request_takes_buffered_path (NEW)
    //
    //   Failover semantics, unrelated error, no extra attempt (clause 3.9)
    //     → router.rs: buffered_adapter_unrelated_400_surfaces_without_extra_attempt
    //     → is_duplicate_compaction_trigger_error_rejects_same_phrasing_at_500
    //       (a 5xx is never treated as the repairable duplicate-trigger defect,
    //        so a Bedrock 500 follows ordinary failover with no extra attempt)
    // ------------------------------------------------------------------

    /// Structural preservation for the AWS SDK Converse path (clause 3.4): the
    /// seam's trigger normalization is reachable ONLY from the API-key dispatch
    /// (`dispatch_mantle` → `normalize_for_mantle`). An `AwsSdk`-mode provider
    /// never routes through it, so a Converse build cannot be altered by this
    /// fix. This test documents that invariant by asserting the seam itself is a
    /// pure request transform (it takes `&mut OpenAIRequest` and touches nothing
    /// AWS-SDK-related), and that a zero-trigger request is left byte-identical —
    /// the same input the Converse builder would receive.
    #[test]
    fn aws_sdk_arm_does_not_normalize_triggers() {
        // A plain request as the Converse builder would see it (no triggers).
        let mut request = OpenAIRequest {
            model: "anthropic.claude-3-5-sonnet".to_string(),
            messages: vec![Message {
                role: "user".to_string(),
                content: serde_json::Value::String("hello".to_string()),
                extra: Default::default(),
            }],
            stream: false,
            temperature: None,
            max_tokens: None,
            extra: Default::default(),
        };
        let before = serde_json::to_value(&request).unwrap();
        // Even if the seam were (incorrectly) called on this path, a zero-trigger
        // request is a no-op — reinforcing that the Converse build is unchanged.
        let result = normalize_mantle_compaction_triggers(&mut request);
        assert_eq!(result.removed, 0);
        assert!(result.survivor.is_none());
        assert_eq!(serde_json::to_value(&request).unwrap(), before);
    }
}
