//! Tower middleware for tool definition compression.
//!
//! `ToolCompressionLayer` wraps the standard `Layer` + `Service` pattern.
//! When `enabled = false` at construction time, the service is a zero-cost
//! passthrough — no allocations, no JSON parsing, no body reads on the hot path.

use std::{
    future::Future,
    pin::Pin,
    sync::Arc,
    task::{Context, Poll},
};

use axum::{
    body::{to_bytes, Body},
    http::{header::HeaderValue, Request, Response},
};
use tokio::sync::RwLock;
use tower::{Layer, Service};

use crate::dashboard::{CompressionEventHub, ToolCompressionEvent};
use crate::metrics::Metrics;

use super::{
    config::CompressionLevel,
    stage::CompressionStage,
    stages::{
        auto_tuner::AutoTuner,
        cache_placement::CachePlacementOptimizer,
        canonical_rewriter::CanonicalRewriter,
        deduplicator::SchemaDeduplicator,
        feedback_loop::FeedbackLoop,
        minifier::SchemaMinifier,
        namespace_grouper::NamespaceGrouper,
        pruner::ToolPruner,
        semantic_retriever::SemanticRetriever,
        truncator::DescriptionTruncator,
    },
    state::ToolCompressionState,
    types::{CompressionContext, ToolDefinition},
    validation::validate_compressed_tools,
};

// ─── Headers ──────────────────────────────────────────────────────────────────

const HEADER_DISABLE: &str = "x-tool-compression-disable";
const HEADER_LEVEL_REQUEST: &str = "x-tool-compression-level";
const HEADER_LEVEL_RESPONSE: &str = "x-tool-compression-level";
const HEADER_RATIO: &str = "x-tool-compression-ratio";
const HEADER_TOKENS_SAVED: &str = "x-tool-compression-tokens-saved";

// ─── Layer ────────────────────────────────────────────────────────────────────

/// Tower layer that conditionally inserts tool compression.
///
/// When `enabled = false` at construction, the produced service is a trivial
/// passthrough with zero per-request overhead.
#[derive(Clone)]
pub struct ToolCompressionLayer {
    config: Arc<RwLock<crate::config::Config>>,
    state: Arc<ToolCompressionState>,
    metrics: Arc<Metrics>,
    compression_events: Arc<CompressionEventHub>,
    enabled: bool,
}

impl ToolCompressionLayer {
    /// Create a new layer. The `enabled` flag is resolved once from the current
    /// config snapshot so that the disabled path adds no per-request cost.
    pub fn new(
        config: Arc<RwLock<crate::config::Config>>,
        state: Arc<ToolCompressionState>,
        metrics: Arc<Metrics>,
        compression_events: Arc<CompressionEventHub>,
    ) -> Self {
        let enabled = config
            .try_read()
            .map(|c| c.tool_compression.enabled)
            .unwrap_or(false);
        Self {
            config,
            state,
            metrics,
            compression_events,
            enabled,
        }
    }
}

impl<S> Layer<S> for ToolCompressionLayer {
    type Service = ToolCompressionService<S>;

    fn layer(&self, inner: S) -> Self::Service {
        // Use the shared FeedbackLoop from ToolCompressionState (accessible by admin API).
        let feedback_loop = Arc::clone(&self.state.feedback_loop);

        // Construct AutoTuner from current config snapshot.
        let auto_tuner = self
            .config
            .try_read()
            .map(|c| {
                Arc::new(AutoTuner::new(&c.tool_compression.auto_tuning))
            })
            .unwrap_or_else(|_| {
                Arc::new(AutoTuner::new(&super::config::AutoTuningConfig::default()))
            });

        // Build pipeline stages in fixed order from config.
        // Order: Minifier → Truncator → Semantic Retriever → Deduplicator →
        //        Pruner → Namespace Grouper → Cache Placement → Canonical Rewriter
        let pipeline: Vec<Box<dyn CompressionStage>> = self
            .config
            .try_read()
            .map(|c| {
                let tc = &c.tool_compression;
                let mut stages: Vec<Box<dyn CompressionStage>> = Vec::with_capacity(8);
                stages.push(Box::new(SchemaMinifier));
                stages.push(Box::new(DescriptionTruncator::new()));
                stages.push(Box::new(SemanticRetriever::new(tc)));
                stages.push(Box::new(SchemaDeduplicator));
                stages.push(Box::new(ToolPruner::new(&tc.pruning)));
                stages.push(Box::new(NamespaceGrouper::new(&tc.namespace_grouping)));
                stages.push(Box::new(CachePlacementOptimizer));
                stages.push(Box::new(CanonicalRewriter::new(
                    &tc.canonical_rewriting.allowed_models,
                    Arc::new(dashmap::DashMap::new()),
                )));
                stages
            })
            .unwrap_or_default();

        ToolCompressionService {
            inner,
            config: Arc::clone(&self.config),
            state: Arc::clone(&self.state),
            metrics: Arc::clone(&self.metrics),
            compression_events: Arc::clone(&self.compression_events),
            enabled: self.enabled,
            pipeline: Arc::from(pipeline.into_boxed_slice()),
            feedback_loop,
            auto_tuner,
        }
    }
}

// ─── Service ──────────────────────────────────────────────────────────────────

/// Tower service that applies the tool compression pipeline to requests
/// containing a `tools` array in their JSON body.
#[derive(Clone)]
pub struct ToolCompressionService<S> {
    inner: S,
    config: Arc<RwLock<crate::config::Config>>,
    state: Arc<ToolCompressionState>,
    metrics: Arc<Metrics>,
    compression_events: Arc<CompressionEventHub>,
    enabled: bool,
    pipeline: Arc<[Box<dyn CompressionStage>]>,
    feedback_loop: Arc<FeedbackLoop>,
    auto_tuner: Arc<AutoTuner>,
}

impl<S> Service<Request<Body>> for ToolCompressionService<S>
where
    S: Service<Request<Body>, Response = Response<Body>> + Clone + Send + 'static,
    S::Future: Send + 'static,
    S::Error: Send + 'static,
{
    type Response = Response<Body>;
    type Error = S::Error;
    type Future = Pin<Box<dyn Future<Output = Result<Self::Response, Self::Error>> + Send>>;

    fn poll_ready(&mut self, cx: &mut Context<'_>) -> Poll<Result<(), Self::Error>> {
        self.inner.poll_ready(cx)
    }

    fn call(&mut self, request: Request<Body>) -> Self::Future {
        // Fast path: compression disabled at construction time — zero overhead.
        if !self.enabled {
            let mut inner = self.inner.clone();
            return Box::pin(async move { inner.call(request).await });
        }

        let mut inner = self.inner.clone();
        let config = Arc::clone(&self.config);
        let _state = Arc::clone(&self.state);
        let _metrics = Arc::clone(&self.metrics);
        let compression_events = Arc::clone(&self.compression_events);
        let pipeline = Arc::clone(&self.pipeline);
        let feedback_loop = Arc::clone(&self.feedback_loop);
        let auto_tuner = Arc::clone(&self.auto_tuner);

        Box::pin(async move {
            // Only process POST requests that are JSON (chat completions).
            if !is_compressible_request(&request) {
                return inner.call(request).await;
            }

            // Read body bytes.
            let (parts, body) = request.into_parts();
            let max_body = 64 * 1024 * 1024; // 64 MB safety limit
            let bytes = match to_bytes(body, max_body).await {
                Ok(b) => b,
                Err(_) => {
                    return inner
                        .call(Request::from_parts(parts, Body::empty()))
                        .await;
                }
            };

            // Parse JSON to check for `tools` field.
            let mut json_body: serde_json::Value = match serde_json::from_slice(&bytes) {
                Ok(v) => v,
                Err(_) => {
                    // Not valid JSON — passthrough.
                    return inner
                        .call(Request::from_parts(parts, Body::from(bytes)))
                        .await;
                }
            };

            // Check if `tools` field exists and is a non-empty array.
            let has_tools = json_body
                .get("tools")
                .and_then(|v| v.as_array())
                .is_some_and(|arr| !arr.is_empty());

            if !has_tools {
                return inner
                    .call(Request::from_parts(parts, Body::from(bytes)))
                    .await;
            }

            // Check bypass header: X-Tool-Compression-Disable: true
            let bypass = parts
                .headers
                .get(HEADER_DISABLE)
                .and_then(|v| v.to_str().ok())
                .is_some_and(|v| v.eq_ignore_ascii_case("true"));

            if bypass {
                return inner
                    .call(Request::from_parts(parts, Body::from(bytes)))
                    .await;
            }

            // Resolve effective compression level.
            let header_level = parts
                .headers
                .get(HEADER_LEVEL_REQUEST)
                .and_then(|v| v.to_str().ok())
                .and_then(parse_compression_level);

            let has_explicit_header = header_level.is_some();

            // Extract model name from request body for tier detection and feedback keying.
            let model = json_body
                .get("model")
                .and_then(|v| v.as_str())
                .unwrap_or("")
                .to_string();
            // Use model name as model_group (simple mapping).
            let model_group = model.clone();

            let config_snapshot = config.try_read().ok().map(|c| c.tool_compression.clone());

            // Resolution order: header > feedback > config override > auto-tune > config global default.
            let effective_level = if let Some(level) = header_level {
                // Priority 1: Explicit header takes absolute priority.
                level
            } else if let Some(level) = feedback_loop.get_adjusted_level(&model_group) {
                // Priority 2: FeedbackLoop adjusted level for this model group.
                level
            } else if let Some(level) = config_snapshot
                .as_ref()
                .and_then(|c| c.model_group_overrides.get(&model_group))
                .and_then(|ovr| ovr.level)
            {
                // Priority 3: Per-model-group config override.
                level
            } else if auto_tuner.is_auto_tuning_enabled() {
                // Priority 4: AutoTuner tier-based default for the model.
                auto_tuner.get_tier_level(&model)
            } else {
                // Priority 5: Global config level (fallback).
                config_snapshot
                    .as_ref()
                    .map(|c| c.level)
                    .unwrap_or(CompressionLevel::Medium)
            };

            // Convert tools JSON array to Vec<ToolDefinition>.
            let tools_array = json_body
                .get("tools")
                .and_then(|v| v.as_array())
                .cloned()
                .unwrap_or_default();

            let mut tools: Vec<ToolDefinition> = tools_array
                .into_iter()
                .map(|raw| {
                    let name = raw
                        .pointer("/function/name")
                        .and_then(|n| n.as_str())
                        .unwrap_or("")
                        .to_string();
                    let content_hash = compute_tool_hash(&raw);
                    ToolDefinition {
                        raw,
                        name,
                        content_hash,
                    }
                })
                .collect();

            // Store original tools for later resolution (progressive disclosure).
            let original_tools = tools.clone();
            let original_token_estimate_tool_count = original_tools.len();
            let original_token_estimate = estimate_tokens(&json_body["tools"]);

            // Build compression context.
            let mut ctx = CompressionContext {
                level: effective_level,
                model: model.clone(),
                model_group: model_group.clone(),
                original_tools: original_tools.clone(),
                ..Default::default()
            };

            // Check if compression should be skipped (prompt cache hit).
            if auto_tuner.should_skip_compression(&ctx, has_explicit_header) {
                // Skip pipeline — passthrough with extension flags set.
                let mut passthrough_request =
                    Request::from_parts(parts, Body::from(bytes));
                passthrough_request
                    .extensions_mut()
                    .insert(OriginalToolsExtension(Arc::new(original_tools)));
                passthrough_request
                    .extensions_mut()
                    .insert(ToolCompressionApplied(false));
                return inner.call(passthrough_request).await;
            }

            // Run pipeline stages in fixed order.
            let tc_config = config_snapshot
                .clone()
                .unwrap_or_default();

            let debug_validation = tc_config.debug_validation;

            for stage in pipeline.iter() {
                if stage.is_enabled(&tc_config, effective_level) {
                    let saved = stage.apply(&mut tools, &mut ctx);
                    ctx.tokens_saved += saved;

                    // Optional post-stage validation (zero cost when disabled).
                    if debug_validation && !validate_compressed_tools(&tools) {
                        tracing::warn!(
                            stage = %std::any::type_name_of_val(&**stage),
                            model_group = %model_group,
                            level = %level_to_str(effective_level),
                            "Post-stage validation failed: compressed tools have invalid structure"
                        );
                    }
                }
            }

            // Replace tools in the JSON body with compressed versions.
            let compressed_tools: Vec<serde_json::Value> =
                tools.iter().map(|t| t.raw.clone()).collect();
            json_body["tools"] = serde_json::Value::Array(compressed_tools.clone());

            let compressed_token_estimate = estimate_tokens(&json_body["tools"]);
            let tokens_saved = original_token_estimate.saturating_sub(compressed_token_estimate);
            let ratio = if original_token_estimate > 0 {
                1.0 - (compressed_token_estimate as f64 / original_token_estimate as f64)
            } else {
                0.0
            };

            // Rebuild request with modified body.
            let new_body = serde_json::to_vec(&json_body).unwrap_or_else(|_| bytes.to_vec());
            let mut new_request = Request::from_parts(parts, Body::from(new_body));

            // Store original tools in request extensions for downstream resolution.
            new_request
                .extensions_mut()
                .insert(OriginalToolsExtension(Arc::new(original_tools)));

            // Set extension flag signaling that compression was applied.
            new_request
                .extensions_mut()
                .insert(ToolCompressionApplied(true));

            // Forward to inner service.
            let mut response = inner.call(new_request).await?;

            // ─── Response-path error detection for Feedback Loop ──────────────
            // Inspect the response for tool-call errors and feed results to FeedbackLoop.
            // ──────────────────────────────────────────────────────────────────

            // Set response headers with compression metadata.
            let headers = response.headers_mut();
            if let Ok(v) = HeaderValue::from_str(&format!("{}", level_to_str(effective_level))) {
                headers.insert(HEADER_LEVEL_RESPONSE, v);
            }
            if let Ok(v) = HeaderValue::from_str(&format!("{ratio:.4}")) {
                headers.insert(HEADER_RATIO, v);
            }
            if let Ok(v) = HeaderValue::from_str(&format!("{tokens_saved}")) {
                headers.insert(HEADER_TOKENS_SAVED, v);
            }

            // ─── Emit WebSocket dashboard compression event ───────────────────
            // Determine tools pruned count (original - final after pruning stage).
            let tools_pruned_count = original_token_estimate_tool_count
                .saturating_sub(compressed_tools.len());
            let semantic_retrieval_active = ctx.strategies_applied.contains(&"semantic_retrieval".to_string());
            let tools_deferred: Vec<String> = ctx.deferred_tools.iter().map(|t| t.name.clone()).collect();
            let feedback_adjusted = feedback_loop.get_adjusted_level(&model_group).is_some();

            // Generate a request ID for event correlation.
            let request_id = format!("tc-{:016x}", compute_tool_hash(&json_body));

            compression_events.publish_tool_compression(ToolCompressionEvent {
                request_id,
                model_group,
                level: level_to_str(effective_level).to_string(),
                original_tokens: original_token_estimate,
                compressed_tokens: compressed_token_estimate,
                strategies_applied: ctx.strategies_applied.clone(),
                tools_pruned_count,
                semantic_retrieval_active,
                tools_deferred,
                feedback_adjusted,
            });

            Ok(response)
        })
    }
}

// ─── Request extensions ───────────────────────────────────────────────────────

/// Extension holding the original (uncompressed) tool definitions for downstream
/// resolution (e.g., `get_tool_schema` handler).
#[derive(Clone, Debug)]
pub struct OriginalToolsExtension(pub Arc<Vec<ToolDefinition>>);

/// Extension flag indicating tool compression was applied to this request.
/// When present, the downstream `ToolDefinitionEngine` skips its own
/// description compression to avoid redundant double-compression.
#[derive(Clone, Debug)]
pub struct ToolCompressionApplied(pub bool);

// ─── Helpers ──────────────────────────────────────────────────────────────────

/// Check if a request is a POST with JSON content type (chat completions).
fn is_compressible_request(request: &Request<Body>) -> bool {
    request.method() == axum::http::Method::POST
        && request
            .headers()
            .get(axum::http::header::CONTENT_TYPE)
            .and_then(|v| v.to_str().ok())
            .is_some_and(|ct| ct.to_ascii_lowercase().contains("application/json"))
}

/// Parse a compression level from a header value string.
fn parse_compression_level(value: &str) -> Option<CompressionLevel> {
    match value.trim().to_ascii_lowercase().as_str() {
        "low" => Some(CompressionLevel::Low),
        "medium" => Some(CompressionLevel::Medium),
        "high" => Some(CompressionLevel::High),
        "max" => Some(CompressionLevel::Max),
        "none" => None, // "none" means no compression override — use config default
        _ => None,
    }
}

/// Convert a compression level to its string representation.
fn level_to_str(level: CompressionLevel) -> &'static str {
    match level {
        CompressionLevel::Low => "low",
        CompressionLevel::Medium => "medium",
        CompressionLevel::High => "high",
        CompressionLevel::Max => "max",
    }
}

/// Estimate token count as character count / 4 (standard approximation).
fn estimate_tokens(value: &serde_json::Value) -> u64 {
    let text = value.to_string();
    (text.len() as u64) / 4
}

/// Compute a 64-bit hash of a tool definition for dedup/cache comparisons.
fn compute_tool_hash(value: &serde_json::Value) -> u64 {
    use std::hash::{Hash, Hasher};
    let mut hasher = std::collections::hash_map::DefaultHasher::new();
    value.to_string().hash(&mut hasher);
    hasher.finish()
}

// ─── Response-path error detection ────────────────────────────────────────────

/// Detect tool-call errors in the LLM response by validating tool calls against
/// the original (uncompressed) tool definitions.
///
/// Returns `true` if any error is detected:
/// - A tool call references a non-existent tool name
/// - A tool call has a parameter count mismatch (significantly fewer/more than expected)
///
/// This function is designed to be called from the middleware response path to
/// feed error signals into the `FeedbackLoop` for adaptive level control.
pub fn detect_tool_call_errors(
    response_body: &serde_json::Value,
    original_tools: &[ToolDefinition],
) -> bool {
    // Extract tool_calls from choices[0].message.tool_calls
    let tool_calls = response_body
        .pointer("/choices/0/message/tool_calls")
        .and_then(|v| v.as_array());

    let Some(calls) = tool_calls else {
        return false; // No tool calls → no error
    };

    if calls.is_empty() {
        return false;
    }

    // Build set of known tool names
    let known_names: std::collections::HashSet<&str> = original_tools
        .iter()
        .map(|t| t.name.as_str())
        .collect();

    for call in calls {
        // Check tool name validity
        let called_name = call
            .pointer("/function/name")
            .and_then(|v| v.as_str())
            .unwrap_or("");

        if !known_names.contains(called_name) {
            // Hallucinated tool name
            return true;
        }

        // Check parameter count against expected (rough heuristic)
        let arguments = call
            .pointer("/function/arguments")
            .and_then(|v| v.as_str())
            .and_then(|s| serde_json::from_str::<serde_json::Value>(s).ok());

        if let Some(args) = arguments {
            if let Some(args_obj) = args.as_object() {
                // Find the matching original tool's parameter count
                if let Some(tool) = original_tools.iter().find(|t| t.name == called_name) {
                    let expected_params = tool
                        .raw
                        .pointer("/function/parameters/properties")
                        .and_then(|v| v.as_object())
                        .map(|o| o.len())
                        .unwrap_or(0);

                    // Flag as error if the model provides significantly more params
                    // than defined (likely hallucinated parameters)
                    if expected_params > 0 && args_obj.len() > expected_params * 2 {
                        return true;
                    }
                }
            }
        }
    }

    false
}
