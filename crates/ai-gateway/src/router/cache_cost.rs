//! Cache-aware cost model: assumed-hit-rate routing estimates and actual cost computation.
//!
//! Two cost surfaces live here:
//!
//! 1. **Routing estimate** â€” [`ProviderModel::total_cost_with_hit_rate`] blends
//!    the cache-read and cache-creation prices with an assumed hit rate. When
//!    the hit rate is `None` or `0.0` it is bit-identical to the existing
//!    [`ProviderModel::total_cost`] comparator, so wiring can switch over
//!    without changing any behavior for cache-unaware providers.
//! 2. **Actual cost** â€” [`compute_actual_cost`] prices a completed response's
//!    usage split into uncached, cache-read, and cache-creation prompt tokens
//!    (plus completion tokens) at their respective per-million rates.
//!
//! Units convention (mirrors `config/mod.rs` and `router.rs`): all
//! `cost_per_million_*` fields are dollars per million tokens. Estimate
//! functions return summed per-million rates; actual-cost functions return
//! dollars by dividing token-weighted sums by `1_000_000.0`.

use crate::config::ProviderModel;
use crate::models::openai::Usage as OpenAIUsage;
use crate::reasoning_compat::cost::{extract_reasoning_usage, reasoning_cost, ReasoningCarrier};
use crate::router::sticky_cache::CacheUsage;

impl ProviderModel {
    /// Cache-aware routing estimate: total per-million cost assuming a
    /// fraction `h` of prompt tokens is served from cache.
    ///
    /// `effective_input = read_price * h + write_price * (1 - h)`, where the
    /// read price falls back to [`ProviderModel::cost_per_million_input_tokens`]
    /// when no cache-read price is configured, and likewise for the write
    /// price. The result is `effective_input + output_price`, matching the
    /// units of [`ProviderModel::total_cost`].
    ///
    /// When `assumed_hit_rate` is `None` or clamps to `0.0`, this returns
    /// exactly `total_cost()` (bit-identical float ops) so the uncached
    /// behavior is unchanged.
    #[inline]
    pub fn total_cost_with_hit_rate(&self, assumed_hit_rate: Option<f64>) -> f64 {
        let hit_rate = assumed_hit_rate.unwrap_or(0.0).clamp(0.0, 1.0);
        if hit_rate == 0.0 {
            return self.cost_per_million_input_tokens + self.cost_per_million_output_tokens;
        }
        let read_price = self
            .cost_per_million_cache_read_input_tokens
            .unwrap_or(self.cost_per_million_input_tokens);
        let write_price = self
            .cost_per_million_cache_creation_input_tokens
            .unwrap_or(self.cost_per_million_input_tokens);
        let effective_input = read_price * hit_rate + write_price * (1.0 - hit_rate);
        effective_input + self.cost_per_million_output_tokens
    }
}

/// Computes the actual dollar cost of a completed response from its usage
/// split, pricing uncached, cache-read, and cache-creation prompt tokens at
/// their own per-million rates (falling back to the base input price when a
/// cache price is unconfigured).
///
/// Cache token fields are parsed from the usage object's flattened extra JSON:
/// - Anthropic shape: `cache_read_input_tokens`, `cache_creation_input_tokens`
/// - OpenAI shape: `prompt_tokens_details.cached_tokens` (treated as read)
///
/// `uncached = prompt_tokens - read - creation`, clamped at zero when a
/// provider over-reports. When no cache fields are present the result is
/// bit-identical to the existing base-price recording formula in `router.rs`.
///
/// # Reasoning-token billing rule (design Component 4)
///
/// Reasoning tokens are billed as output regardless of cache state, but the
/// charge differs by usage shape:
///
/// - **Anthropic shape** (`output_tokens_details.thinking_tokens` present):
///   thinking tokens are ADDITIVE to the reported output tokens, so they are
///   billed on top of the full output-token charge at
///   `cost_per_million_reasoning_tokens` (falling back to the output price):
///   `completion_tokens × output_price + thinking_tokens × reasoning_price`.
/// - **OpenAI shape** (`completion_tokens_details.reasoning_tokens`) and
///   flattened streaming relays (top-level `reasoning_tokens`): reasoning
///   tokens are a SUBSET of `completion_tokens` and providers bill the full
///   completion count, so NO extra charge is added here — the reasoning split
///   is surfaced separately via [`extract_reasoning_usage`] for logs/metrics
///   without double-counting.
pub fn compute_actual_cost(model: &ProviderModel, usage: &OpenAIUsage) -> f64 {
    let cache = extract_cache_usage(usage);
    let read_price = model
        .cost_per_million_cache_read_input_tokens
        .unwrap_or(model.cost_per_million_input_tokens);
    let write_price = model
        .cost_per_million_cache_creation_input_tokens
        .unwrap_or(model.cost_per_million_input_tokens);

    let reasoning = extract_reasoning_usage(usage);
    // Additive only for the Anthropic shape; subset shapes (OpenAI/streaming)
    // are already billed inside completion_tokens — adding here would
    // double-count (see the billing-rule doc above). A 0.0 addend preserves
    // the legacy bit-identical results.
    let reasoning_charge = match reasoning.carrier {
        ReasoningCarrier::AnthropicDetails => reasoning_cost(model, reasoning.reasoning_tokens),
        _ => 0.0,
    };

    let cached_prompt = cache.cache_read_input_tokens.saturating_add(cache.cache_creation_input_tokens);
    if cached_prompt == 0 {
        return (usage.prompt_tokens as f64 * model.cost_per_million_input_tokens
            / 1_000_000.0)
            + (usage.completion_tokens as f64 * model.cost_per_million_output_tokens
                / 1_000_000.0)
            + reasoning_charge;
    }

    (cache.uncached_input_tokens as f64 * model.cost_per_million_input_tokens
        + cache.cache_read_input_tokens as f64 * read_price
        + cache.cache_creation_input_tokens as f64 * write_price
        + usage.completion_tokens as f64 * model.cost_per_million_output_tokens)
        / 1_000_000.0
        + reasoning_charge
}

/// Extracts the cache token breakdown from a provider usage object.
///
/// Maps the Anthropic (`cache_read_input_tokens`,
/// `cache_creation_input_tokens`) and OpenAI (`prompt_tokens_details.cached_tokens`)
/// usage shapes into [`CacheUsage`]. Missing fields default to zero and the
/// uncached count is clamped at zero when a provider over-reports cached
/// tokens relative to `prompt_tokens`.
pub fn extract_cache_usage(usage: &OpenAIUsage) -> CacheUsage {
    let cache_read = token_field(&usage.extra, "cache_read_input_tokens")
        .or_else(|| {
            usage
                .extra
                .get("prompt_tokens_details")
                .and_then(|details| token_field_from(details.get("cached_tokens")))
        })
        .unwrap_or(0);
    let cache_creation = token_field(&usage.extra, "cache_creation_input_tokens").unwrap_or(0);
    let cached = cache_read.saturating_add(cache_creation);
    CacheUsage {
        cache_read_input_tokens: cache_read,
        cache_creation_input_tokens: cache_creation,
        uncached_input_tokens: (usage.prompt_tokens as u64).saturating_sub(cached),
    }
}

/// Reads a non-negative integer token count from a flattened extra map.
fn token_field(extra: &serde_json::Map<String, serde_json::Value>, key: &str) -> Option<u64> {
    token_field_from(extra.get(key))
}

/// Coerces a JSON value into a non-negative integer token count.
pub(crate) fn token_field_from(value: Option<&serde_json::Value>) -> Option<u64> {
    match value {
        Some(serde_json::Value::Number(n)) => {
            if let Some(u) = n.as_u64() {
                Some(u)
            } else {
                n.as_f64()
                    .filter(|f| *f >= 0.0 && f.fract() == 0.0)
                    .map(|f| f as u64)
            }
        }
        _ => None,
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    fn model(
        input: f64,
        output: f64,
        cache_read: Option<f64>,
        cache_creation: Option<f64>,
    ) -> ProviderModel {
        ProviderModel {
            provider: "test-provider".to_string(),
            model: "test-model".to_string(),
            cost_per_million_input_tokens: input,
            cost_per_million_output_tokens: output,
            priority: 100,
            structured_output_passthrough: None,
            tier: None,
            context_window: 0,
            specializations: vec![],
            cost_per_million_cache_read_input_tokens: cache_read,
            cost_per_million_cache_creation_input_tokens: cache_creation,
            cache_min_tokens: None,
            cache_support: None,
        cost_per_million_reasoning_tokens: None,
        reasoning_family: None,
        reasoning_parameter: None,
        }
    }

    fn usage_from_json(value: serde_json::Value) -> OpenAIUsage {
        serde_json::from_value(value).expect("test usage JSON is valid")
    }

    #[test]
    fn none_and_zero_hit_rate_match_total_cost_exactly() {
        let models = vec![
            model(3.0, 15.0, Some(0.30), Some(3.75)),
            model(3.0, 15.0, Some(0.30), None),
            model(3.0, 15.0, None, Some(3.75)),
            model(3.0, 15.0, None, None),
            model(0.0, 0.0, None, None),
            model(10.5, 31.25, Some(1.5), Some(2.0)),
        ];
        for m in &models {
            assert_eq!(
                m.total_cost_with_hit_rate(None).to_bits(),
                m.total_cost().to_bits(),
                "None hit rate must be bit-identical for {:?}",
                m
            );
            assert_eq!(
                m.total_cost_with_hit_rate(Some(0.0)).to_bits(),
                m.total_cost().to_bits(),
                "0.0 hit rate must be bit-identical for {:?}",
                m
            );
        }
    }

    #[test]
    fn full_hit_rate_uses_read_only_pricing() {
        // input=$3/M, cache_read=$0.30/M, cache_creation=$3.75/M, output=$15/M.
        let m = model(3.0, 15.0, Some(0.30), Some(3.75));
        // h=1.0: effective input = 0.30 * 1.0 + 3.75 * 0.0 = 0.30.
        assert_eq!(m.total_cost_with_hit_rate(Some(1.0)), 0.30 + 15.0);
        // Scaled to 1000 prompt tokens (rate-sum units are per-million):
        // (0.30 + 15.0) * 1000 / 1_000_000 = 0.0153.
        assert!((m.total_cost_with_hit_rate(Some(1.0)) * 1000.0 / 1_000_000.0 - 0.0153).abs() < 1e-12);
    }

    #[test]
    fn half_hit_rate_mixes_read_and_write_prices() {
        let m = model(3.0, 15.0, Some(0.30), Some(3.75));
        // h=0.5: effective input = 0.30*0.5 + 3.75*0.5 = 2.025; total = 17.025.
        assert_eq!(
            m.total_cost_with_hit_rate(Some(0.5)),
            (0.30 * 0.5 + 3.75 * 0.5) + 15.0
        );
        assert!((m.total_cost_with_hit_rate(Some(0.5)) - 17.025).abs() < 1e-12);
    }

    #[test]
    fn hit_rate_is_clamped_to_unit_interval() {
        let m = model(3.0, 15.0, Some(0.30), Some(3.75));
        assert_eq!(
            m.total_cost_with_hit_rate(Some(2.0)),
            m.total_cost_with_hit_rate(Some(1.0))
        );
        assert_eq!(
            m.total_cost_with_hit_rate(Some(-1.0)).to_bits(),
            m.total_cost().to_bits()
        );
    }

    #[test]
    fn missing_cache_prices_fall_back_to_base_input_price() {
        let m = model(3.0, 15.0, None, None);
        // With no cache prices, every hit rate reduces to base pricing.
        assert_eq!(
            m.total_cost_with_hit_rate(Some(1.0)).to_bits(),
            m.total_cost().to_bits()
        );
        assert_eq!(
            m.total_cost_with_hit_rate(Some(0.5)).to_bits(),
            m.total_cost().to_bits()
        );
    }

    #[test]
    fn compute_actual_cost_anthropic_shape() {
        // prompt=1000, cache_read=800, cache_creation=200 -> uncached=0.
        let usage = usage_from_json(json!({
            "prompt_tokens": 1000,
            "completion_tokens": 100,
            "total_tokens": 1100,
            "cache_read_input_tokens": 800,
            "cache_creation_input_tokens": 200
        }));
        let m = model(3.0, 15.0, Some(0.30), Some(3.75));
        // (0*3 + 800*0.30 + 200*3.75 + 100*15) / 1e6 = 2490 / 1e6 = 0.00249.
        let cost = compute_actual_cost(&m, &usage);
        let expected = (800.0 * 0.30 + 200.0 * 3.75 + 100.0 * 15.0) / 1_000_000.0;
        assert!((cost - 0.00249).abs() < 1e-12);
        assert_eq!(cost, expected);
    }

    #[test]
    fn compute_actual_cost_openai_shape() {
        // prompt=1000, cached_tokens=500 -> 500 read, 0 creation, 500 uncached.
        let usage = usage_from_json(json!({
            "prompt_tokens": 1000,
            "completion_tokens": 200,
            "total_tokens": 1200,
            "prompt_tokens_details": { "cached_tokens": 500 }
        }));
        let m = model(3.0, 15.0, Some(1.5), None);
        // (500*3 + 500*1.5 + 0*3 + 200*15) / 1e6 = (1500+750+3000)/1e6 = 0.00525.
        let cost = compute_actual_cost(&m, &usage);
        assert!((cost - 0.00525).abs() < 1e-12);
    }

    #[test]
    fn compute_actual_cost_falls_back_to_base_price_without_cache_fields() {
        let usage = usage_from_json(json!({
            "prompt_tokens": 1000,
            "completion_tokens": 100,
            "total_tokens": 1100
        }));
        let m = model(3.0, 15.0, Some(0.30), Some(3.75));
        // Must equal the legacy router.rs formula bit-for-bit.
        let legacy: f64 = (1000.0 * 3.0 / 1_000_000.0) + (100.0 * 15.0 / 1_000_000.0);
        assert_eq!(compute_actual_cost(&m, &usage).to_bits(), legacy.to_bits());
    }

    #[test]
    fn compute_actual_cost_missing_cache_prices_fall_back_to_base_rates() {
        let usage = usage_from_json(json!({
            "prompt_tokens": 1000,
            "completion_tokens": 0,
            "cache_read_input_tokens": 800,
            "cache_creation_input_tokens": 200
        }));
        let m = model(3.0, 15.0, None, None);
        // read+creation at base price: (0*3 + 800*3 + 200*3 + 0) / 1e6 = 0.003.
        let cost = compute_actual_cost(&m, &usage);
        assert!((cost - 0.003).abs() < 1e-12);
    }

    #[test]
    fn over_reported_cache_tokens_clamp_uncached_at_zero() {
        let usage = usage_from_json(json!({
            "prompt_tokens": 100,
            "completion_tokens": 10,
            "cache_read_input_tokens": 90,
            "cache_creation_input_tokens": 50
        }));
        let cache = extract_cache_usage(&usage);
        assert_eq!(cache.uncached_input_tokens, 0);
        assert_eq!(cache.cache_read_input_tokens, 90);
        assert_eq!(cache.cache_creation_input_tokens, 50);

        let m = model(3.0, 15.0, Some(0.30), Some(3.75));
        // (0*3 + 90*0.30 + 50*3.75 + 10*15) / 1e6 = (27 + 187.5 + 150)/1e6.
        let cost = compute_actual_cost(&m, &usage);
        let expected = (90.0 * 0.30 + 50.0 * 3.75 + 10.0 * 15.0) / 1_000_000.0;
        assert_eq!(cost, expected);
    }

    #[test]
    fn extract_cache_usage_maps_anthropic_shape() {
        let usage = usage_from_json(json!({
            "prompt_tokens": 1000,
            "completion_tokens": 50,
            "cache_read_input_tokens": 700,
            "cache_creation_input_tokens": 200
        }));
        assert_eq!(
            extract_cache_usage(&usage),
            CacheUsage {
                cache_read_input_tokens: 700,
                cache_creation_input_tokens: 200,
                uncached_input_tokens: 100,
            }
        );
    }

    #[test]
    fn extract_cache_usage_maps_openai_shape() {
        let usage = usage_from_json(json!({
            "prompt_tokens": 1000,
            "completion_tokens": 50,
            "prompt_tokens_details": { "cached_tokens": 500 }
        }));
        assert_eq!(
            extract_cache_usage(&usage),
            CacheUsage {
                cache_read_input_tokens: 500,
                cache_creation_input_tokens: 0,
                uncached_input_tokens: 500,
            }
        );
    }

#[test]
fn extract_cache_usage_defaults_to_zero_without_cache_fields() {
let usage = usage_from_json(json!({
"prompt_tokens": 1000,
"completion_tokens": 50
}));
assert_eq!(
extract_cache_usage(&usage),
CacheUsage {
cache_read_input_tokens: 0,
cache_creation_input_tokens: 0,
uncached_input_tokens: 1000,
}
);
}

#[test]
fn compute_actual_cost_anthropic_reasoning_tokens_are_additive() {
// Anthropic shape: thinking tokens are additive to completion tokens.
let usage = usage_from_json(json!({
"prompt_tokens": 1000,
"completion_tokens": 100,
"output_tokens_details": { "thinking_tokens": 150 }
}));
let mut m = model(3.0, 15.0, None, None);
m.cost_per_million_reasoning_tokens = Some(30.0);
// (1000*3 + 100*15 + 150*30) / 1e6 = (3000 + 1500 + 4500) / 1e6 = 0.009.
let cost = compute_actual_cost(&m, &usage);
assert!((cost - 0.009).abs() < 1e-12);
}

#[test]
fn compute_actual_cost_anthropic_reasoning_falls_back_to_output_price() {
let usage = usage_from_json(json!({
"prompt_tokens": 1000,
"completion_tokens": 100,
"output_tokens_details": { "thinking_tokens": 150 }
}));
let m = model(3.0, 15.0, None, None);
// Reasoning priced at the output rate: 1000*3 + 100*15 + 150*15 (per 1e6).
let cost = compute_actual_cost(&m, &usage);
let expected = (1000.0 * 3.0 / 1_000_000.0)
+ (100.0 * 15.0 / 1_000_000.0)
+ (150.0 * 15.0 / 1_000_000.0);
assert_eq!(cost, expected);
assert!((cost - 0.00675).abs() < 1e-9);
}

#[test]
fn compute_actual_cost_openai_reasoning_tokens_are_not_double_billed() {
// OpenAI shape: reasoning is a subset of completion_tokens, so an explicit
// reasoning price must NOT add an extra charge.
let usage = usage_from_json(json!({
"prompt_tokens": 1000,
"completion_tokens": 1000,
"completion_tokens_details": { "reasoning_tokens": 500 }
}));
let mut m = model(3.0, 15.0, None, None);
m.cost_per_million_reasoning_tokens = Some(60.0);
let legacy: f64 = (1000.0 * 3.0 / 1_000_000.0) + (1000.0 * 15.0 / 1_000_000.0);
assert_eq!(compute_actual_cost(&m, &usage).to_bits(), legacy.to_bits());
}

#[test]
fn compute_actual_cost_streaming_reasoning_tokens_are_not_double_billed() {
// Flattened streaming relay shape: top-level reasoning_tokens is a subset
// of completion_tokens, so no extra charge.
let usage = usage_from_json(json!({
"prompt_tokens": 1000,
"completion_tokens": 800,
"reasoning_tokens": 200
}));
let mut m = model(3.0, 15.0, None, None);
m.cost_per_million_reasoning_tokens = Some(60.0);
let legacy: f64 = (1000.0 * 3.0 / 1_000_000.0) + (800.0 * 15.0 / 1_000_000.0);
assert_eq!(compute_actual_cost(&m, &usage).to_bits(), legacy.to_bits());
}

#[test]
fn compute_actual_cost_with_cache_and_reasoning_tokens_sums_correctly() {
let usage = usage_from_json(json!({
"prompt_tokens": 1000,
"completion_tokens": 100,
"cache_read_input_tokens": 800,
"cache_creation_input_tokens": 200,
"output_tokens_details": { "thinking_tokens": 150 }
}));
let mut m = model(3.0, 15.0, Some(0.30), Some(3.75));
m.cost_per_million_reasoning_tokens = Some(30.0);
// (0*3 + 800*0.30 + 200*3.75 + 100*15) / 1e6 + 150*30 / 1e6 = 0.00699.
let cost = compute_actual_cost(&m, &usage);
assert!((cost - 0.00699).abs() < 1e-9);
let expected = (0.0 * 3.0 + 800.0 * 0.30 + 200.0 * 3.75 + 100.0 * 15.0) / 1_000_000.0
+ (30.0 * 150.0 / 1_000_000.0);
assert_eq!(cost, expected);
}
}
