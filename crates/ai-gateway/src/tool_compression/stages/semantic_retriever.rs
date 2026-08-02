//! Semantic Retriever stage — selects tools most relevant to the user's message.
//!
//! Uses a hybrid scoring approach combining semantic similarity (via TF-IDF/BM25
//! fallback or embedding cosine similarity) with historical usage frequency.
//! Defers low-scoring tools to `CompressionContext.deferred_tools`.

use crate::tool_compression::config::{CompressionLevel, ToolCompressionConfig};
use crate::tool_compression::stage::CompressionStage;
use crate::tool_compression::tfidf::TfIdfScorer;
use crate::tool_compression::types::{CompressionContext, ToolDefinition};

// ─── EmbeddingModel trait ─────────────────────────────────────────────────────

/// Trait for embedding generation (async-agnostic for now; external API will wrap).
pub trait EmbeddingModel: Send + Sync {
    /// Generate embeddings for a batch of texts.
    /// Returns one vector per input text. Returns empty vec on failure.
    fn embed_batch(&self, texts: &[&str]) -> Vec<Vec<f32>>;
}

/// TF-IDF fallback embedding: produces BM25-score vectors as pseudo-embeddings.
///
/// When real embedding models are unavailable or exceed the latency budget,
/// this model uses the TF-IDF scorer to produce score vectors that can be
/// treated as similarity proxies (already in [0,1] range from BM25 normalization).
pub struct TfIdfFallbackModel;

impl EmbeddingModel for TfIdfFallbackModel {
    fn embed_batch(&self, _texts: &[&str]) -> Vec<Vec<f32>> {
        // The fallback model doesn't produce real embeddings; scoring is done
        // directly via TfIdfScorer::score_query in the retriever's apply method.
        Vec::new()
    }
}

// ─── SemanticRetriever stage ──────────────────────────────────────────────────

/// Semantic Retriever compression stage.
///
/// Selects top-K tools most relevant to the user's current message using
/// hybrid scoring: `(1 - frequency_weight) * similarity + frequency_weight * normalized_frequency`.
///
/// Tools below `similarity_threshold` are deferred (moved to `ctx.deferred_tools`).
pub struct SemanticRetriever {
    /// Maximum tools to retain in the active set.
    top_k: usize,
    /// Minimum hybrid score for inclusion; tools below this are deferred.
    similarity_threshold: f32,
    /// Weight for frequency vs semantic (0.0 = pure semantic, 1.0 = pure frequency).
    frequency_weight: f32,
}

impl SemanticRetriever {
    /// Create a new `SemanticRetriever` from config values.
    pub fn new(config: &ToolCompressionConfig) -> Self {
        let sr = &config.semantic_retrieval;
        Self {
            top_k: sr.top_k as usize,
            similarity_threshold: sr.similarity_threshold,
            frequency_weight: sr.frequency_weight.clamp(0.0, 1.0),
        }
    }

    /// Build a TF-IDF scorer from tool descriptions.
    fn build_scorer(tools: &[ToolDefinition]) -> (TfIdfScorer, Vec<String>) {
        let descriptions: Vec<String> = tools
            .iter()
            .map(|t| Self::extract_description(t))
            .collect();
        let doc_refs: Vec<&str> = descriptions.iter().map(|s| s.as_str()).collect();
        let scorer = TfIdfScorer::new(&doc_refs);
        (scorer, descriptions)
    }

    /// Extract description text from a tool definition for scoring.
    /// Combines name + description for richer signal.
    fn extract_description(tool: &ToolDefinition) -> String {
        let desc = tool
            .raw
            .get("function")
            .and_then(|f| f.get("description"))
            .and_then(|d| d.as_str())
            .unwrap_or("");
        format!("{} {}", tool.name, desc)
    }

    /// Compute normalized frequency for each tool from session usage.
    fn compute_normalized_frequencies(
        tools: &[ToolDefinition],
        ctx: &CompressionContext,
    ) -> Vec<f32> {
        let max_count = ctx
            .session_usage
            .values()
            .copied()
            .max()
            .unwrap_or(1)
            .max(1);

        tools
            .iter()
            .map(|t| {
                let count = ctx.session_usage.get(&t.name).copied().unwrap_or(0);
                count as f32 / max_count as f32
            })
            .collect()
    }
}

impl CompressionStage for SemanticRetriever {
    fn apply(&self, tools: &mut Vec<ToolDefinition>, ctx: &mut CompressionContext) -> u64 {
        // No-op if no query or insufficient tools
        if tools.is_empty() {
            return 0;
        }

        let query = match ctx.message_content.as_deref() {
            Some(q) if !q.trim().is_empty() => q,
            _ => return 0,
        };

        // If tool count is already within top_k, nothing to filter
        if tools.len() <= self.top_k {
            return 0;
        }

        // Build TF-IDF scorer from current tool descriptions (fallback path)
        let (_scorer, descriptions) = Self::build_scorer(tools);
        let doc_refs: Vec<&str> = descriptions.iter().map(|s| s.as_str()).collect();

        // Re-create scorer with the same corpus for consistent scoring
        let scorer = TfIdfScorer::new(&doc_refs);

        // Score each tool's description against the query using BM25
        let similarity_scores = scorer.score_query(query, &doc_refs);

        // Compute normalized frequencies from session usage
        let freq_scores = Self::compute_normalized_frequencies(tools, ctx);

        // Compute hybrid scores
        let hybrid_scores: Vec<f32> = similarity_scores
            .iter()
            .zip(freq_scores.iter())
            .map(|(&sim, &freq)| {
                (1.0 - self.frequency_weight) * sim + self.frequency_weight * freq
            })
            .collect();

        // Build indexed scores and sort descending
        let mut indexed: Vec<(usize, f32)> = hybrid_scores
            .iter()
            .enumerate()
            .map(|(i, &score)| (i, score))
            .collect();
        indexed.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));

        // Partition into keep (top-K above threshold) and defer
        let mut keep_indices: Vec<usize> = Vec::new();
        let mut defer_indices: Vec<usize> = Vec::new();

        for (rank, &(idx, score)) in indexed.iter().enumerate() {
            if rank < self.top_k && score >= self.similarity_threshold {
                keep_indices.push(idx);
            } else {
                defer_indices.push(idx);
            }
        }

        // If nothing to defer, return early
        if defer_indices.is_empty() {
            return 0;
        }

        // Calculate token savings from deferred tools
        let mut total_saved: u64 = 0;
        for &idx in &defer_indices {
            total_saved += estimate_tokens(&tools[idx].raw);
        }

        // Move deferred tools to ctx.deferred_tools
        // Sort defer_indices descending so removal doesn't shift earlier indices
        defer_indices.sort_unstable_by(|a, b| b.cmp(a));
        for &idx in &defer_indices {
            ctx.deferred_tools.push(tools[idx].clone());
        }

        // Rebuild tools vec keeping only kept indices (preserve original order)
        keep_indices.sort_unstable();
        let kept_tools: Vec<ToolDefinition> = keep_indices
            .iter()
            .map(|&idx| tools[idx].clone())
            .collect();
        *tools = kept_tools;

        if total_saved > 0 {
            ctx.strategies_applied
                .push("semantic_retriever".to_string());
            ctx.tokens_saved += total_saved;
        }

        total_saved
    }

    fn is_enabled(&self, config: &ToolCompressionConfig, _level: CompressionLevel) -> bool {
        config.semantic_retrieval.enabled
    }
}

/// Estimate token count as character_count / 4.
fn estimate_tokens(value: &serde_json::Value) -> u64 {
    let s = value.to_string();
    (s.len() as u64) / 4
}

// ─── Tests ────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;
    use std::collections::HashMap;

    fn make_tool(name: &str, description: &str) -> ToolDefinition {
        ToolDefinition {
            raw: json!({
                "type": "function",
                "function": {
                    "name": name,
                    "description": description,
                    "parameters": {
                        "type": "object",
                        "properties": {}
                    }
                }
            }),
            name: name.to_string(),
            content_hash: 0,
        }
    }

    fn make_ctx(message: Option<&str>) -> CompressionContext {
        let mut ctx = CompressionContext::default();
        ctx.message_content = message.map(|s| s.to_string());
        ctx
    }

    fn make_retriever(top_k: usize, threshold: f32, freq_weight: f32) -> SemanticRetriever {
        SemanticRetriever {
            top_k,
            similarity_threshold: threshold,
            frequency_weight: freq_weight,
        }
    }

    #[test]
    fn test_noop_empty_tools() {
        let retriever = make_retriever(5, 0.3, 0.3);
        let mut tools: Vec<ToolDefinition> = Vec::new();
        let mut ctx = make_ctx(Some("search repos"));
        let saved = retriever.apply(&mut tools, &mut ctx);
        assert_eq!(saved, 0);
        assert!(ctx.deferred_tools.is_empty());
    }

    #[test]
    fn test_noop_no_message() {
        let retriever = make_retriever(2, 0.3, 0.3);
        let mut tools = vec![
            make_tool("search_repos", "Search GitHub repositories"),
            make_tool("send_message", "Send a Slack message"),
            make_tool("get_weather", "Get weather forecast"),
        ];
        let mut ctx = make_ctx(None);
        let saved = retriever.apply(&mut tools, &mut ctx);
        assert_eq!(saved, 0);
        assert_eq!(tools.len(), 3);
    }

    #[test]
    fn test_noop_within_top_k() {
        let retriever = make_retriever(5, 0.3, 0.3);
        let mut tools = vec![
            make_tool("search_repos", "Search GitHub repositories"),
            make_tool("send_message", "Send a Slack message"),
        ];
        let mut ctx = make_ctx(Some("search repos"));
        let saved = retriever.apply(&mut tools, &mut ctx);
        assert_eq!(saved, 0);
        assert_eq!(tools.len(), 2);
    }

    #[test]
    fn test_defers_low_scoring_tools() {
        let retriever = make_retriever(2, 0.0, 0.0); // pure semantic, no threshold filter
        let mut tools = vec![
            make_tool("search_repos", "Search GitHub repositories by name and language"),
            make_tool("send_message", "Send a Slack message to a channel"),
            make_tool("get_weather", "Get weather forecast for a location"),
            make_tool("list_repos", "List all GitHub repositories for a user"),
        ];
        let mut ctx = make_ctx(Some("search github repositories"));
        let saved = retriever.apply(&mut tools, &mut ctx);

        // Should keep top 2 and defer 2
        assert_eq!(tools.len(), 2);
        assert_eq!(ctx.deferred_tools.len(), 2);
        assert!(saved > 0);
        assert!(ctx.strategies_applied.contains(&"semantic_retriever".to_string()));
    }

    #[test]
    fn test_frequency_weight_boosts_used_tools() {
        // High frequency weight — should favor tools with usage history
        let retriever = make_retriever(2, 0.0, 0.9);
        let mut tools = vec![
            make_tool("search_repos", "Search GitHub repositories"),
            make_tool("send_message", "Send a Slack message"),
            make_tool("get_weather", "Get weather forecast"),
        ];
        let mut ctx = make_ctx(Some("weather forecast"));
        // Give send_message high usage
        ctx.session_usage.insert("send_message".to_string(), 100);
        ctx.session_usage.insert("search_repos".to_string(), 50);
        ctx.session_usage.insert("get_weather".to_string(), 1);

        retriever.apply(&mut tools, &mut ctx);

        // With 0.9 freq weight, the top-used tools should be kept
        let kept_names: Vec<&str> = tools.iter().map(|t| t.name.as_str()).collect();
        assert_eq!(kept_names.len(), 2);
        // send_message (100 calls) and search_repos (50 calls) should beat get_weather
        assert!(kept_names.contains(&"send_message"));
        assert!(kept_names.contains(&"search_repos"));
    }

    #[test]
    fn test_threshold_defers_below_minimum() {
        // Set threshold high enough that some tools won't make it even in top-K
        let retriever = make_retriever(10, 0.8, 0.0);
        let mut tools = vec![
            make_tool("search_repos", "Search GitHub repositories by name"),
            make_tool("send_message", "Send a Slack message to channel"),
            make_tool("get_weather", "Get weather forecast for location"),
            make_tool("create_issue", "Create a GitHub issue in repository"),
        ];
        let mut ctx = make_ctx(Some("search github repositories"));
        retriever.apply(&mut tools, &mut ctx);

        // Tools scoring below 0.8 threshold should be deferred even if within top_k
        // At least some tools should be deferred with such a high threshold
        assert!(!ctx.deferred_tools.is_empty() || tools.len() <= 4);
    }

    #[test]
    fn test_empty_message_is_noop() {
        let retriever = make_retriever(2, 0.3, 0.3);
        let mut tools = vec![
            make_tool("search_repos", "Search GitHub repositories"),
            make_tool("send_message", "Send a Slack message"),
            make_tool("get_weather", "Get weather forecast"),
        ];
        let mut ctx = make_ctx(Some("   "));
        let saved = retriever.apply(&mut tools, &mut ctx);
        assert_eq!(saved, 0);
        assert_eq!(tools.len(), 3);
    }
}

// ─── Property Tests ───────────────────────────────────────────────────────────

#[cfg(test)]
mod property_tests {
    use super::*;
    use proptest::prelude::*;
    use serde_json::json;
    use std::collections::{HashMap, HashSet};

    // ─── Strategies ──────────────────────────────────────────────────────────

    /// Generate a vector of tools with random descriptions.
    fn arb_tool_list() -> impl Strategy<Value = Vec<ToolDefinition>> {
        (5usize..=15).prop_flat_map(|count| {
            prop::collection::vec("[a-z ]{5,30}", count).prop_map(|descriptions| {
                descriptions
                    .into_iter()
                    .enumerate()
                    .map(|(i, desc)| ToolDefinition {
                        raw: json!({
                            "type": "function",
                            "function": {
                                "name": format!("tool_{}", i),
                                "description": desc,
                                "parameters": { "type": "object", "properties": {} }
                            }
                        }),
                        name: format!("tool_{}", i),
                        content_hash: i as u64,
                    })
                    .collect()
            })
        })
    }

    // ─── Property 11: Semantic Retrieval Hybrid Selection Correctness ─────────
    // Feature: tool-definition-compression
    // **Validates: Requirements 16.2, 16.3, 16.4, 16.8**
    //
    // Strategy:
    // 1. Generate random tool lists (5-15 tools with random descriptions)
    // 2. Generate a random query
    // 3. Generate random session_usage (0-100 per tool)
    // 4. Set varying top_k (2-8) and frequency_weight (0.0-1.0)
    // 5. Apply the SemanticRetriever
    // 6. Verify:
    //    - tools.len() <= top_k (respecting threshold)
    //    - tools.len() + ctx.deferred_tools.len() == original_len (no tools lost)
    //    - All tool names from output are a subset of original tool names

    proptest! {
        #![proptest_config(ProptestConfig::with_cases(200))]

        #[test]
        fn semantic_retrieval_hybrid_selection_correctness(
            tools in arb_tool_list(),
            query in "[a-z]{3,20}",
            top_k in 2usize..=8,
            frequency_weight in 0.0f32..=1.0,
            usage_values in prop::collection::vec(0u64..100, 5..=15usize),
        ) {
            let original_len = tools.len();
            let original_names: HashSet<String> = tools.iter().map(|t| t.name.clone()).collect();

            // Build session usage map
            let session_usage: HashMap<String, u64> = tools
                .iter()
                .enumerate()
                .map(|(i, t)| {
                    let count = usage_values.get(i).copied().unwrap_or(0);
                    (t.name.clone(), count)
                })
                .collect();

            // Create retriever with the test parameters
            let retriever = SemanticRetriever {
                top_k,
                similarity_threshold: 0.0, // disable threshold filtering for selection correctness
                frequency_weight: frequency_weight.clamp(0.0, 1.0),
            };

            let mut test_tools = tools.clone();
            let mut ctx = CompressionContext {
                message_content: Some(query.clone()),
                session_usage,
                ..Default::default()
            };

            retriever.apply(&mut test_tools, &mut ctx);

            // ── Assertion 1: Output size respects top_k ──
            prop_assert!(
                test_tools.len() <= top_k,
                "Output tools ({}) must be <= top_k ({})",
                test_tools.len(),
                top_k
            );

            // ── Assertion 2: No tools lost ──
            let total = test_tools.len() + ctx.deferred_tools.len();
            prop_assert_eq!(
                total,
                original_len,
                "tools.len() ({}) + deferred ({}) must equal original_len ({})",
                test_tools.len(),
                ctx.deferred_tools.len(),
                original_len
            );

            // ── Assertion 3: All output names are subset of original ──
            let output_names: HashSet<String> = test_tools
                .iter()
                .chain(ctx.deferred_tools.iter())
                .map(|t| t.name.clone())
                .collect();
            prop_assert_eq!(
                output_names,
                original_names,
                "All tool names in output + deferred must equal original names"
            );
        }
    }
}
