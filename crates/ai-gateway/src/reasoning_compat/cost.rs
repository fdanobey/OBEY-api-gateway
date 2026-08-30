//! Reasoning-token cost attribution (design Component 4).
//!
//! Extracts reasoning/thinking token usage from the usage shapes providers
//! report and prices it, so reasoning spend (including invisible reasoning
//! that never reaches the client) is attributed accurately.
//!
//! Usage shapes handled ([`extract_reasoning_usage`]):
//! - **OpenAI**: `completion_tokens_details.reasoning_tokens` — reasoning is
//!   a SUBSET of `completion_tokens`.
//! - **Anthropic**: `output_tokens_details.thinking_tokens` — thinking is
//!   ADDITIVE to the reported output/completion tokens.
//! - **Streaming relays**: some reassembled/relayed streams flatten the
//!   count to a top-level `reasoning_tokens` field.
//!
//! When a usage object carries more than one of these fields they describe
//! the same tokens: the larger count wins (never the sum).
//!
//! Pricing ([`reasoning_cost`]) uses `cost_per_million_reasoning_tokens`
//! when configured and falls back to the output price otherwise, mirroring
//! the fallback discipline of the cache prices in `router/cache_cost.rs`.

use crate::config::ProviderModel;
use crate::models::openai::Usage;
use crate::router::cache_cost::token_field_from;

/// Reasoning-token usage extracted from a provider usage object.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct ReasoningUsage {
    /// Total reasoning/thinking tokens (never double-counted across shapes).
    pub reasoning_tokens: u32,
    /// Which usage shape the count was extracted from.
    pub carrier: ReasoningCarrier,
}

/// The usage shape a reasoning-token count was extracted from.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ReasoningCarrier {
    /// OpenAI shape: `completion_tokens_details.reasoning_tokens`.
    OpenAIDetails,
    /// Anthropic shape: `output_tokens_details.thinking_tokens`.
    AnthropicDetails,
    /// Streaming relays that flatten the count to a top-level
    /// `reasoning_tokens` field.
    Streaming,
    /// No reasoning-token fields were present.
    None,
}

/// Extracts the reasoning/thinking token count from a provider usage object.
///
/// Reads, from the usage object's flattened extra JSON:
/// - `completion_tokens_details.reasoning_tokens` (OpenAI shape)
/// - `output_tokens_details.thinking_tokens` (Anthropic shape)
/// - top-level `reasoning_tokens` (flattened streaming relays) — only when
///   neither detailed field is present.
///
/// When both detailed fields are present they describe the same tokens, so
/// the larger count wins (never the sum) and the carrier reflects the
/// winning field. Values are coerced from u64/f64 JSON numbers via the same
/// [`token_field_from`] discipline as the cache extraction in
/// `router/cache_cost.rs`; counts that overflow `u32` saturate.
pub fn extract_reasoning_usage(usage: &Usage) -> ReasoningUsage {
    let openai = usage
        .extra
        .get("completion_tokens_details")
        .and_then(|details| token_field_from(details.get("reasoning_tokens")));
    let anthropic = usage
        .extra
        .get("output_tokens_details")
        .and_then(|details| token_field_from(details.get("thinking_tokens")));

    let (tokens, carrier) = match (openai, anthropic) {
        (Some(openai), Some(anthropic)) if openai >= anthropic => {
            (openai, ReasoningCarrier::OpenAIDetails)
        }
        (Some(_), Some(anthropic)) => (anthropic, ReasoningCarrier::AnthropicDetails),
        (Some(openai), None) => (openai, ReasoningCarrier::OpenAIDetails),
        (None, Some(anthropic)) => (anthropic, ReasoningCarrier::AnthropicDetails),
        (None, None) => match token_field_from(usage.extra.get("reasoning_tokens")) {
            Some(tokens) => (tokens, ReasoningCarrier::Streaming),
            None => (0, ReasoningCarrier::None),
        },
    };

    ReasoningUsage {
        reasoning_tokens: u32::try_from(tokens).unwrap_or(u32::MAX),
        carrier,
    }
}

/// Prices reasoning tokens at `cost_per_million_reasoning_tokens`, falling
/// back to the output price when no dedicated reasoning price is configured
/// (legacy behavior: reasoning billed as regular output tokens).
pub fn reasoning_cost(model: &ProviderModel, reasoning_tokens: u32) -> f64 {
    model
        .cost_per_million_reasoning_tokens
        .unwrap_or(model.cost_per_million_output_tokens)
        * f64::from(reasoning_tokens)
        / 1_000_000.0
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    fn model(output: f64, reasoning: Option<f64>) -> ProviderModel {
        ProviderModel {
            provider: "test-provider".to_string(),
            model: "test-model".to_string(),
            cost_per_million_input_tokens: 3.0,
            cost_per_million_output_tokens: output,
            priority: 100,
            structured_output_passthrough: None,
            tier: None,
            context_window: 0,
            specializations: vec![],
            cost_per_million_cache_read_input_tokens: None,
            cost_per_million_cache_creation_input_tokens: None,
            cache_min_tokens: None,
            cache_support: None,
            cost_per_million_reasoning_tokens: reasoning,
            reasoning_family: None,
            reasoning_parameter: None,
        }
    }

    fn usage_from_json(value: serde_json::Value) -> Usage {
        serde_json::from_value(value).expect("test usage JSON is valid")
    }

    #[test]
    fn extracts_openai_shape() {
        let usage = usage_from_json(json!({
            "completion_tokens": 1000,
            "completion_tokens_details": { "reasoning_tokens": 500 }
        }));
        assert_eq!(
            extract_reasoning_usage(&usage),
            ReasoningUsage {
                reasoning_tokens: 500,
                carrier: ReasoningCarrier::OpenAIDetails
            }
        );
    }

    #[test]
    fn extracts_anthropic_shape() {
        let usage = usage_from_json(json!({
            "output_tokens": 1500,
            "output_tokens_details": { "thinking_tokens": 700 }
        }));
        assert_eq!(
            extract_reasoning_usage(&usage),
            ReasoningUsage {
                reasoning_tokens: 700,
                carrier: ReasoningCarrier::AnthropicDetails
            }
        );
    }

    #[test]
    fn both_present_takes_max_not_sum() {
        let usage = usage_from_json(json!({
            "completion_tokens": 1000,
            "completion_tokens_details": { "reasoning_tokens": 750 },
            "output_tokens_details": { "thinking_tokens": 300 }
        }));
        assert_eq!(
            extract_reasoning_usage(&usage),
            ReasoningUsage {
                reasoning_tokens: 750,
                carrier: ReasoningCarrier::OpenAIDetails
            }
        );
    }

    #[test]
    fn both_present_carrier_follows_larger_field() {
        let usage = usage_from_json(json!({
            "completion_tokens_details": { "reasoning_tokens": 300 },
            "output_tokens_details": { "thinking_tokens": 750 }
        }));
        assert_eq!(
            extract_reasoning_usage(&usage),
            ReasoningUsage {
                reasoning_tokens: 750,
                carrier: ReasoningCarrier::AnthropicDetails
            }
        );
    }

    #[test]
    fn neither_present_reports_zero_with_none_carrier() {
        let usage = usage_from_json(json!({
            "prompt_tokens": 100,
            "completion_tokens": 50
        }));
        assert_eq!(
            extract_reasoning_usage(&usage),
            ReasoningUsage {
                reasoning_tokens: 0,
                carrier: ReasoningCarrier::None
            }
        );
    }

    #[test]
    fn top_level_reasoning_tokens_maps_to_streaming_carrier() {
        let usage = usage_from_json(json!({
            "completion_tokens": 800,
            "reasoning_tokens": 200
        }));
        assert_eq!(
            extract_reasoning_usage(&usage),
            ReasoningUsage {
                reasoning_tokens: 200,
                carrier: ReasoningCarrier::Streaming
            }
        );
    }

    #[test]
    fn float_json_numbers_are_coerced_and_non_numeric_ignored() {
        let usage = usage_from_json(json!({
            "completion_tokens_details": { "reasoning_tokens": 500.0 }
        }));
        assert_eq!(extract_reasoning_usage(&usage).reasoning_tokens, 500);

        let usage = usage_from_json(json!({
            "completion_tokens": 100,
            "reasoning_tokens": "500"
        }));
        assert_eq!(
            extract_reasoning_usage(&usage),
            ReasoningUsage {
                reasoning_tokens: 0,
                carrier: ReasoningCarrier::None
            }
        );
    }

    #[test]
    fn reasoning_cost_uses_explicit_price() {
        let m = model(15.0, Some(1.0));
        // $1/M * 2000 tokens = 0.002.
        assert!((reasoning_cost(&m, 2000) - 0.002).abs() < 1e-15);
    }

    #[test]
    fn reasoning_cost_falls_back_to_output_price() {
        let m = model(3.0, None);
        // $3/M output * 1000 tokens = 0.003.
        assert!((reasoning_cost(&m, 1000) - 0.003).abs() < 1e-15);
    }

    #[test]
    fn reasoning_cost_is_zero_for_zero_tokens() {
        let m = model(15.0, Some(1.0));
        assert_eq!(reasoning_cost(&m, 0), 0.0);
    }
}
