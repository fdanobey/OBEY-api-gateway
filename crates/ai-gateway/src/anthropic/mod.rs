//! Anthropic Messages API front door (`/v1/messages`).
//!
//! Gateway-owned protocol types for the Anthropic Messages wire format,
//! mirroring the conventions of [`crate::responses`].

pub mod models;
pub mod translate_request;

pub use translate_request::translate;
pub mod passthrough;
pub mod synthesize_response;
pub mod count_tokens;
pub mod errors;

pub use count_tokens::count_tokens;
pub use errors::{from_gateway_error, from_translator_rejection};

/// Input-side usage estimate for `message_start` (Req 2.1): tokenizer
/// estimate on the translated chat request, using the same
/// [`crate::compression::token_counter::TokenCounter`] counting helper as
/// `/v1/messages/count_tokens`.
pub fn anthropic_input_usage_estimate(
    chat_request: &crate::models::openai::OpenAIRequest,
) -> Usage {
    let counter = crate::compression::token_counter::TokenCounter::new();
    let input_tokens = u64::from(counter.count_request(chat_request));
    Usage {
        input_tokens,
        ..Usage::default()
    }
}

pub mod synthesize_stream;

pub use models::{
    AnthropicErrorResponse, AnthropicMessage, AnthropicRequest, AnthropicResponse,
    AnthropicSseEvent, AnthropicToolChoice, AnthropicToolDefinition, AnthropicToolKind,
    ContentBlock, ContentBlockDelta, DocumentSource, ErrorBody, ImageSource, MessageContent,
    MessageDeltaBody, RequestMetadata, StopReason, SystemBlock, SystemPrompt, TextOrImageBlock,
    ThinkingConfig, ToolResultContent, Usage,
};
pub use synthesize_response::{synthesize_anthropic_response, AnthropicSynthesisContext};
pub use synthesize_stream::{ChatDelta, MessageStreamAssembler};

/// Serialize one [`AnthropicSseEvent`] as an SSE frame:
/// `event: <type>\ndata: <json>\n\n` (spec Requirement 2.1).
///
/// Returns `None` when the event fails to serialize (never expected: the
/// wire types are plain serde structs) so callers can skip the frame
/// instead of emitting a malformed one.
#[cfg(test)]
pub fn sse_frame(event: &AnthropicSseEvent) -> Option<String> {
    let data = serde_json::to_string(event).ok()?;
    Some(format!("event: {}\ndata: {}\n\n", event.name(), data))
}

/// `(event name, JSON data)` parts for SSE emitters that compose frames
/// themselves (axum's `Event::event(..).data(..)`); avoids `Event::data`
/// splitting embedded newlines across multiple `data:` fields.
pub fn sse_event_parts(event: &AnthropicSseEvent) -> Option<(&'static str, String)> {
    let data = serde_json::to_string(event).ok()?;
    Some((event.name(), data))
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::anthropic::models::error_types;

    #[test]
    fn sse_frame_formats_event_name_and_json_data() {
        let event = AnthropicSseEvent::Ping {
            extra: serde_json::Map::new(),
        };
        let frame = sse_frame(&event).expect("ping serializes");
        assert_eq!(frame, "event: ping\ndata: {\"type\":\"ping\"}\n\n");
    }

    #[test]
    fn sse_frame_message_stop_matches_documented_shape() {
        let event = AnthropicSseEvent::MessageStop {
            extra: serde_json::Map::new(),
        };
        let frame = sse_frame(&event).expect("message_stop serializes");
        assert_eq!(frame, "event: message_stop\ndata: {\"type\":\"message_stop\"}\n\n");
    }

    #[test]
    fn sse_frame_error_event_carries_error_payload() {
        let event = AnthropicSseEvent::Error {
            error: crate::anthropic::errors::error_response(
                error_types::OVERLOADED_ERROR,
                "upstream saturated",
            )
            .error,
            extra: serde_json::Map::new(),
        };
        let frame = sse_frame(&event).expect("error serializes");
        assert!(frame.starts_with("event: error\ndata: "));
        assert!(frame.contains("overloaded_error"));
        assert!(frame.ends_with("\n\n"));
    }
}
