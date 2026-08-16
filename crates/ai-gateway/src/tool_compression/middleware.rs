//! Tower middleware for tool definition compression.
//!
//! `ToolCompressionLayer` wraps the standard `Layer` + `Service` pattern.
//! When `enabled = false` at construction time, the service is a zero-cost
//! passthrough — no allocations, no JSON parsing, no body reads on the hot path.

use std::{
    collections::HashSet,
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
    resolver,
    stage::CompressionStage,
    stages::{
        auto_tuner::AutoTuner, cache_placement::CachePlacementOptimizer,
        canonical_rewriter::CanonicalRewriter, deduplicator::SchemaDeduplicator,
        description_compressor::DescriptionCompressor, disclosure::ProgressiveDisclosureEngine,
        feedback_loop::FeedbackLoop, minifier::SchemaMinifier, namespace_grouper::NamespaceGrouper,
        pruner::ToolPruner, semantic_retriever::SemanticRetriever, truncator::DescriptionTruncator,
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
const HEADER_SESSION_ID: &str = "x-session-id";

/// Baseline number of synthetic drill-down resolution steps per request.
///
/// The effective budget scales with the number of namespaces offered (see
/// [`synthetic_step_budget`]): a model legitimately exploring a large tool set calls
/// `get_tools_in_namespace` once per namespace, so a fixed budget of 6 was exhausted
/// by real workloads with a dozen namespaces.
const MAX_SYNTHETIC_RESOLUTION_STEPS: usize = 6;

/// Hard ceiling on resolution steps, bounding work for a misbehaving model.
const MAX_SYNTHETIC_RESOLUTION_STEPS_CEILING: usize = 32;

/// Effective resolution budget for a request offering `namespace_count` namespaces.
///
/// Allows one drill-down per namespace plus headroom for follow-up `get_tool_schema`
/// calls, never below the baseline and never above the ceiling.
fn synthetic_step_budget(namespace_count: usize) -> usize {
    namespace_count
        .saturating_add(4)
        .max(MAX_SYNTHETIC_RESOLUTION_STEPS)
        .min(MAX_SYNTHETIC_RESOLUTION_STEPS_CEILING)
}

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
            .map(|c| Arc::new(AutoTuner::new(&c.tool_compression.auto_tuning)))
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
                stages.push(Box::new(DescriptionCompressor::new(
                    &tc.precomputed_descriptions,
                    &[],
                )));
                stages.push(Box::new(SemanticRetriever::new(tc)));
                stages.push(Box::new(SchemaDeduplicator));
                stages.push(Box::new(ToolPruner::new(&tc.pruning)));
                stages.push(Box::new(NamespaceGrouper::new(&tc.namespace_grouping)));
                stages.push(Box::new(ProgressiveDisclosureEngine::new()));
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
        // Re-check the live config so a `tool_compression.enabled` toggle applied via
        // hot-reload takes effect without a restart (the layer is built once at startup).
        let enabled = self.enabled
            || self
                .config
                .try_read()
                .map(|c| c.tool_compression.enabled)
                .unwrap_or(self.enabled);
        if !enabled {
            let mut inner = self.inner.clone();
            return Box::pin(async move { inner.call(request).await });
        }

        let mut inner = self.inner.clone();
        let config = Arc::clone(&self.config);
        let state = Arc::clone(&self.state);
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
                    return inner.call(Request::from_parts(parts, Body::empty())).await;
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

            // Optional stable session id used for multi-turn disclosure tracking.
            let session_id = parts
                .headers
                .get(HEADER_SESSION_ID)
                .and_then(|v| v.to_str().ok())
                .map(|s| s.to_string());

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
                session_id: session_id.clone(),
                ..Default::default()
            };

            // Check if compression should be skipped (prompt cache hit).
            if auto_tuner.should_skip_compression(&ctx, has_explicit_header) {
                // Skip pipeline — passthrough with extension flags set.
                let mut passthrough_request = Request::from_parts(parts, Body::from(bytes));
                passthrough_request
                    .extensions_mut()
                    .insert(OriginalToolsExtension(Arc::new(original_tools)));
                passthrough_request
                    .extensions_mut()
                    .insert(ToolCompressionApplied(false));
                return inner.call(passthrough_request).await;
            }

            // Run pipeline stages in fixed order.
            let tc_config = config_snapshot.clone().unwrap_or_default();

            let debug_validation = tc_config.debug_validation;
            let disclosure_max_tools = tc_config.disclosure_max_tools;

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

            // Re-inject previously disclosed full tool schemas so they remain callable
            // across turns. Namespace grouping removes individual tools from the listing;
            // progressive disclosure keeps them (so we skip names already present to avoid
            // duplicate tool definitions).
            if let Some(sid) = &session_id {
                if let Some(disclosed) = state.disclosure_state.get(sid) {
                    if !disclosed.is_empty() {
                        let mut added = false;
                        let mut reinjected = 0usize;
                        for tool in original_tools.iter() {
                            if disclosed.contains(&tool.name)
                                && !tools.iter().any(|t| t.name == tool.name)
                            {
                                // Bound callable-tool count re-injected across turns.
                                if disclosure_max_tools > 0
                                    && reinjected >= disclosure_max_tools as usize
                                {
                                    break;
                                }
                                tools.push(ToolDefinition {
                                    raw: tool.raw.clone(),
                                    name: tool.name.clone(),
                                    content_hash: 0,
                                });
                                reinjected += 1;
                                added = true;
                            }
                        }
                        if added {
                            ctx.strategies_applied
                                .push("disclosure_reinject".to_string());
                        }
                    }
                }
            }

            // Replace tools in the JSON body with compressed versions.
            let compressed_tools: Vec<serde_json::Value> =
                tools.iter().map(|t| t.raw.clone()).collect();
            json_body["tools"] = serde_json::Value::Array(compressed_tools.clone());

            // Did any stage inject a synthetic drill-down tool? Only then must the
            // response be inspected (and, for SSE, buffered) so the synthetic call is
            // resolved here instead of leaking to the client.
            let synthetic_injected = tools.iter().any(|t| {
                t.name == resolver::GET_TOOLS_IN_NAMESPACE
                    || t.name == resolver::GET_TOOL_SCHEMA
                    || t.name.starts_with(resolver::NS_PREFIX)
            });

            // Number of namespaces offered this turn. The model may legitimately drill
            // into each one, so the resolution budget scales with this count.
            let namespace_count = tools
                .iter()
                .filter(|t| t.name.starts_with(resolver::NS_PREFIX))
                .count();

            let compressed_token_estimate = estimate_tokens(&json_body["tools"]);
            let tokens_saved = original_token_estimate.saturating_sub(compressed_token_estimate);
            let ratio = if original_token_estimate > 0 {
                1.0 - (compressed_token_estimate as f64 / original_token_estimate as f64)
            } else {
                0.0
            };

            // Rebuild request with modified body.
            let new_body = serde_json::to_vec(&json_body).unwrap_or_else(|_| bytes.to_vec());
            let parts_clone = parts.clone();
            let mut new_request = Request::from_parts(parts, Body::from(new_body));

            // Store original tools in request extensions for downstream resolution.
            new_request
                .extensions_mut()
                .insert(OriginalToolsExtension(Arc::new(original_tools.clone())));

            // Set extension flag signaling that compression was applied.
            new_request
                .extensions_mut()
                .insert(ToolCompressionApplied(true));

            // Forward to inner service.
            let response = inner.call(new_request).await?;

            // ─── Synthetic drill-down resolution loop ─────────────────────────
            // If the model calls a synthetic tool emitted by the compression stages
            // (get_tool_schema / get_tools_in_namespace), resolve it against the
            // original tools, feed the result back, and re-call so the model can
            // invoke the real tool. Bounded by MAX_SYNTHETIC_RESOLUTION_STEPS.
            //
            // The gateway returns an SSE response whenever the client requested
            // streaming, so the loop must buffer and reassemble SSE bodies to see the
            // assistant's tool_calls — otherwise a synthetic call would be relayed to
            // the client, which cannot execute it ("unavailable tool"). Buffering only
            // happens when synthetic tools were actually injected this request, so
            // ordinary streaming traffic keeps true pass-through behaviour.
            let mut final_response = response;
            let mut req_json = json_body.clone();
            let mut steps = 0usize;
            let step_budget = synthetic_step_budget(namespace_count);
            // Canonical discovery keys already revealed this session (namespace/grouping
            // only). Used so re-drills are reminded from session cache instead of being
            // silently re-resolved, strengthening multi-turn memory of disclosed tools.
            let already_disclosed: HashSet<String> = session_id
                .as_ref()
                .and_then(|sid| state.disclosure_targets.get(sid).map(|set| set.clone()))
                .unwrap_or_default();
            // Tools/namespaces disclosed by synthetic drill-downs this turn. Used to append
            // a benign "extracted compressed tools" note to the client response so the
            // model's own chat history records the discovery and stops re-drilling.
            let mut disclosed_tools_this_turn: Vec<String> = Vec::new();
            let mut disclosed_ns_this_turn: Vec<String> = Vec::new();
            while synthetic_injected {
                let sse = response_is_sse(&final_response);
                let (resp_parts, resp_body) = final_response.into_parts();
                let resp_bytes = match to_bytes(resp_body, max_body).await {
                    Ok(b) => b,
                    Err(_) => {
                        final_response = Response::from_parts(resp_parts, Body::empty());
                        break;
                    }
                };
                let resp_json: serde_json::Value = match response_bytes_to_json(&resp_bytes, sse) {
                    Some(v) => v,
                    None => {
                        final_response = Response::from_parts(resp_parts, Body::from(resp_bytes));
                        break;
                    }
                };

                // Budget exhausted: a synthetic call must still never be relayed, so
                // answer it locally and terminate the turn instead of forwarding it.
                if steps >= step_budget {
                    match sanitize_synthetic_response(&resp_json, &original_tools, sse) {
                        Some(sanitized) => {
                            tracing::warn!(
                                model_group = %model_group,
                                steps,
                                "Synthetic drill-down budget exhausted; answering locally instead of relaying the synthetic call to the client"
                            );
                            let mut parts = resp_parts;
                            parts.headers.remove(axum::http::header::CONTENT_LENGTH);
                            final_response = Response::from_parts(parts, Body::from(sanitized));
                        }
                        None => {
                            final_response =
                                Response::from_parts(resp_parts, Body::from(resp_bytes));
                        }
                    }
                    break;
                }

                match resolver::resolve_synthetic_in_response(
                    &resp_json,
                    &mut req_json,
                    &original_tools,
                    disclosure_max_tools,
                    &already_disclosed,
                ) {
                    Some(disclosed) => {
                        // Persist disclosed tools for multi-turn re-injection.
                        disclosed_tools_this_turn.extend(disclosed.iter().cloned());
                        if let Some(sid) = &session_id {
                            let mut entry = state.disclosure_state.entry(sid.clone()).or_default();
                            for n in &disclosed {
                                entry.insert(n.clone());
                            }
                            // Record the discovery targets this turn so a later re-drill
                            // is met with the stronger REDRILL_HINT reminder.
                            let mut targets =
                                state.disclosure_targets.entry(sid.clone()).or_default();
                            if let Some(calls) = resp_json
                                .pointer("/choices/0/message/tool_calls")
                                .and_then(|v| v.as_array())
                            {
                                for call in calls {
                                    let Some(fn_obj) = call.get("function").or(Some(call)) else {
                                        continue;
                                    };
                                    let Some(name) =
                                        fn_obj.get("name").and_then(|n| n.as_str())
                                    else {
                                        continue;
                                    };
                                    let args = fn_obj
                                        .get("arguments")
                                        .and_then(|a| a.as_str())
                                        .unwrap_or("{}");
                                    if let Some(key) = resolver::discovery_key(name, args) {
                                        targets.insert(key.clone());
                                        if let Some(ns) = key.strip_prefix("ns:") {
                                            if !disclosed_ns_this_turn
                                                .iter()
                                                .any(|n| n == ns)
                                            {
                                                disclosed_ns_this_turn.push(ns.to_string());
                                            }
                                        }
                                    }
                                }
                            }
                        }
                    }
                    None => {
                        final_response = Response::from_parts(resp_parts, Body::from(resp_bytes));
                        break;
                    }
                }

                let next_body = match serde_json::to_vec(&req_json) {
                    Ok(b) => b,
                    Err(_) => {
                        final_response = Response::from_parts(resp_parts, Body::from(resp_bytes));
                        break;
                    }
                };
                let mut next_parts = parts_clone.clone();
                let len_value = HeaderValue::from_str(&next_body.len().to_string())
                    .unwrap_or_else(|_| HeaderValue::from_static("0"));
                next_parts
                    .headers
                    .insert(axum::http::header::CONTENT_LENGTH, len_value);
                let mut next_req = Request::from_parts(next_parts, Body::from(next_body));
                next_req
                    .extensions_mut()
                    .insert(OriginalToolsExtension(Arc::new(original_tools.clone())));
                next_req
                    .extensions_mut()
                    .insert(ToolCompressionApplied(true));
                match inner.call(next_req).await {
                    Ok(r) => final_response = r,
                    Err(e) => return Err(e),
                }
                steps += 1;
            }

            // ─── Benign discovery note for the client transcript ──────────────────
            // Surface the tool-compression discovery in the model's own chat history as a
            // friendly note (no synthetic tool names / raw schemas). This lets the model
            // remember it already extracted these tools and call them directly instead of
            // re-drilling the same namespace. The synthetic tool name never reaches the
            // client (existing contract), so a router consumer never sees internal plumbing.
            let mut response = final_response;
            if !disclosed_tools_this_turn.is_empty() {
                let note = benign_extraction_note(&disclosed_ns_this_turn, &disclosed_tools_this_turn);
                if !note.is_empty() {
                    response = append_content_to_response(response, &note).await;
                }
            }

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
            let tools_pruned_count =
                original_token_estimate_tool_count.saturating_sub(compressed_tools.len());
            let semantic_retrieval_active = ctx
                .strategies_applied
                .contains(&"semantic_retrieval".to_string());
            let tools_deferred: Vec<String> =
                ctx.deferred_tools.iter().map(|t| t.name.clone()).collect();
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
    let known_names: std::collections::HashSet<&str> =
        original_tools.iter().map(|t| t.name.as_str()).collect();

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

/// Returns `true` when the response is an SSE stream (`text/event-stream`).
///
/// The gateway emits SSE whenever the client requested streaming, so a synthetic
/// drill-down call arrives as streamed `tool_calls` deltas rather than a JSON body.
fn response_is_sse(resp: &Response<Body>) -> bool {
    resp.headers()
        .get(axum::http::header::CONTENT_TYPE)
        .and_then(|v| v.to_str().ok())
        .is_some_and(|ct| ct.to_ascii_lowercase().contains("text/event-stream"))
}

/// Rewrite a response that still carries synthetic drill-down tool calls into one the
/// client can consume, answering the call locally instead of relaying it.
///
/// The client has no `get_tools_in_namespace` / `get_tool_schema` / `ns_*` tool, so
/// relaying such a call surfaces as "Model tried to call unavailable tool". This is the
/// last-resort guarantee that never happens: the synthetic calls are replaced with an
/// assistant message naming the tools the model was asking about, and the turn is
/// terminated with `finish_reason: "stop"`.
///
/// Returns `None` when the response carries no synthetic calls (nothing to rewrite).
/// `sse` selects the output framing so the client still receives the shape it expects.
fn sanitize_synthetic_response(
    resp_json: &serde_json::Value,
    original_tools: &[ToolDefinition],
    sse: bool,
) -> Option<Vec<u8>> {
    let calls = resp_json
        .pointer("/choices/0/message/tool_calls")
        .and_then(|v| v.as_array())?;

    let mut notes: Vec<String> = Vec::new();
    for call in calls {
        let name = call
            .pointer("/function/name")
            .and_then(|v| v.as_str())
            .unwrap_or_default();
        let args = call
            .pointer("/function/arguments")
            .and_then(|v| v.as_str())
            .unwrap_or("{}");

        let is_synthetic = name == resolver::GET_TOOLS_IN_NAMESPACE
            || name == resolver::GET_TOOL_SCHEMA
            || name.starts_with(resolver::NS_PREFIX);
        if !is_synthetic {
            continue;
        }

        // Work out which namespace the model was asking about, if any.
        let ns = if let Some(stripped) = name.strip_prefix(resolver::NS_PREFIX) {
            Some(stripped.to_string())
        } else {
            serde_json::from_str::<serde_json::Value>(args)
                .ok()
                .and_then(|v| {
                    v.get("namespace")
                        .and_then(|n| n.as_str())
                        .map(|s| s.to_string())
                })
        };

        if let Some(ns) = ns {
            let names: Vec<String> = resolver::tools_in_namespace(&ns, original_tools)
                .iter()
                .filter_map(|t| {
                    t.pointer("/function/name")
                        .and_then(|n| n.as_str())
                        .map(|s| s.to_string())
                })
                .collect();
            if names.is_empty() {
                notes.push(format!("Namespace '{ns}' contains no tools."));
            } else {
                notes.push(format!(
                    "Tools in namespace '{}': {}.",
                    ns,
                    names.join(", ")
                ));
            }
        } else if let Some(tool_name) = serde_json::from_str::<serde_json::Value>(args)
            .ok()
            .and_then(|v| {
                v.get("tool_name")
                    .and_then(|n| n.as_str())
                    .map(|s| s.to_string())
            })
        {
            notes.push(format!("Requested schema for tool '{tool_name}'."));
        } else {
            notes.push(format!("Unresolved tool-discovery request '{name}'."));
        }
    }

    // No synthetic calls present — caller should relay the original bytes untouched.
    if notes.is_empty() {
        return None;
    }

    let content = format!(
        "Tool discovery limit reached, so this was answered without another model turn. {} Call the listed tools directly by name.",
        notes.join(" ")
    );

    if sse {
        let chunk = serde_json::json!({
            "id": resp_json.get("id").and_then(|v| v.as_str()).unwrap_or("chatcmpl-tc"),
            "object": "chat.completion.chunk",
            "created": resp_json.get("created").and_then(|v| v.as_i64()).unwrap_or(0),
            "model": resp_json.get("model").and_then(|v| v.as_str()).unwrap_or(""),
            "choices": [{
                "index": 0,
                "delta": {"role": "assistant", "content": content},
                "finish_reason": "stop"
            }]
        });
        Some(format!("data: {chunk}\n\ndata: [DONE]\n\n").into_bytes())
    } else {
        let mut out = resp_json.clone();
        out["choices"] = serde_json::json!([{
            "index": 0,
            "message": {"role": "assistant", "content": content},
            "finish_reason": "stop"
        }]);
        serde_json::to_vec(&out).ok()
    }
}

/// Normalise a response body into a non-streaming chat-completion JSON value so
/// synthetic tool calls can be inspected.
///
/// SSE bodies are reassembled with the same accumulator the router uses for its
/// buffered/cache path, which merges `tool_calls` deltas by index into complete
/// tool-call objects. `Message.extra` is `#[serde(flatten)]`, so the reassembled
/// value exposes `/choices/0/message/tool_calls` exactly like a buffered response.
fn response_bytes_to_json(bytes: &[u8], sse: bool) -> Option<serde_json::Value> {
    let looks_like_sse = sse
        || std::str::from_utf8(bytes)
            .ok()
            .is_some_and(|t| t.trim_start().starts_with("data:"));

    if looks_like_sse {
        let text = std::str::from_utf8(bytes).ok()?;
        let assembled = crate::router::router::Router::reassemble_sse_response(text).ok()?;
        serde_json::to_value(assembled).ok()
    } else {
        serde_json::from_slice(bytes).ok()
    }
}

/// Build a benign, consumer-friendly note recording that tool-compression just revealed
/// some tools, without ever naming the synthetic drill-down tool or dumping raw schemas.
/// The note lists the actual (now-callable) tool names so the model's transcript reminds
/// it to call them directly instead of re-discovering the namespace.
fn benign_extraction_note(namespaces: &[String], tools: &[String]) -> String {
    if tools.is_empty() {
        return String::new();
    }
    let listed: Vec<&String> = tools.iter().take(16).collect();
    let mut note = String::new();
    if !namespaces.is_empty() {
        note.push_str(&format!(
            "Compressed tools for namespace(s) {} were extracted",
            namespaces.join(", ")
        ));
    } else {
        note.push_str("A compressed tool schema was extracted");
    }
    note.push_str(&format!(
        " and are now available: {}",
        listed
            .iter()
            .map(|t| t.as_str())
            .collect::<Vec<_>>()
            .join(", ")
    ));
    if tools.len() > listed.len() {
        note.push_str(&format!(" (and {} more)", tools.len() - listed.len()));
    }
    note.push_str(". Call them directly by name rather than re-discovering the namespace.");
    note
}

/// Append `addition` to the textual content of a chat-completion response, handling both
/// JSON and SSE shapings, so the note reaches the client's stored transcript.
async fn append_content_to_response(response: Response<Body>, addition: &str) -> Response<Body> {
    let (parts, body) = response.into_parts();
    let bytes = match to_bytes(body, 64 * 1024 * 1024).await {
        Ok(bytes) => bytes,
        Err(_) => return Response::from_parts(parts, Body::empty()),
    };
    let is_sse = parts
        .headers
        .get(axum::http::header::CONTENT_TYPE)
        .and_then(|value| value.to_str().ok())
        .is_some_and(|value| value.to_ascii_lowercase().contains("text/event-stream"));
    let new_bytes = if is_sse {
        append_to_sse(&bytes, addition)
    } else {
        append_to_json(&bytes, addition)
    };
    let mut parts = parts;
    parts.headers.remove(axum::http::header::CONTENT_LENGTH);
    Response::from_parts(parts, Body::from(new_bytes))
}

fn append_to_json(bytes: &[u8], addition: &str) -> Vec<u8> {
    let mut value: serde_json::Value = match serde_json::from_slice(bytes) {
        Ok(value) => value,
        Err(_) => return bytes.to_vec(),
    };
    if let Some(choices) = value.get_mut("choices").and_then(|choices| choices.as_array_mut()) {
        for choice in choices.iter_mut() {
            if let Some(message) = choice.get_mut("message") {
                match message.get_mut("content") {
                    Some(serde_json::Value::String(existing)) => {
                        existing.push('\n');
                        existing.push_str(addition);
                    }
                    Some(null) if null.is_null() => {
                        *null = serde_json::Value::String(addition.to_string());
                    }
                    _ => {}
                }
            }
        }
    }
    serde_json::to_vec(&value).unwrap_or_else(|_| bytes.to_vec())
}

fn append_to_sse(bytes: &[u8], addition: &str) -> Vec<u8> {
    let text = String::from_utf8_lossy(bytes);
    let escaped = addition.replace('\\', "\\\\").replace('"', "\\\"").replace('\n', "\\n");
    let chunk = format!(
        "data: {{\"choices\":[{{\"index\":0,\"delta\":{{\"content\":\"{escaped}\"}}}}]}}\n\n"
    );
    if let Some(pos) = text.rfind("data: [DONE]") {
        let mut out = String::with_capacity(text.len() + chunk.len());
        out.push_str(&text[..pos]);
        out.push_str(&chunk);
        out.push_str(&text[pos..]);
        out.into_bytes()
    } else {
        let mut out = text.to_string();
        out.push_str(&chunk);
        out.into_bytes()
    }
}

// ─── Unit Tests ───────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    fn tool(name: &str) -> ToolDefinition {
        ToolDefinition {
            raw: json!({
                "type": "function",
                "function": {
                    "name": name,
                    "description": "d",
                    "parameters": {"type": "object", "properties": {}}
                }
            }),
            name: name.to_string(),
            content_hash: 0,
        }
    }

    /// An SSE body streaming a `get_tools_in_namespace` call must normalise into a
    /// buffered-shaped JSON value exposing `/choices/0/message/tool_calls`, so the
    /// resolution loop can see it. Regression: previously SSE bodies were skipped and
    /// the synthetic call was relayed to the client as an "unavailable tool".
    #[test]
    fn sse_body_with_synthetic_tool_call_is_normalised() {
        let body = concat!(
            "data: {\"id\":\"c1\",\"model\":\"m\",\"created\":1,\"choices\":[{\"index\":0,\"delta\":{\"tool_calls\":[{\"index\":0,\"id\":\"call_1\",\"type\":\"function\",\"function\":{\"name\":\"get_tools_in_namespace\",\"arguments\":\"\"}}]},\"finish_reason\":null}]}\n\n",
            "data: {\"id\":\"c1\",\"choices\":[{\"index\":0,\"delta\":{\"tool_calls\":[{\"index\":0,\"function\":{\"arguments\":\"{\\\"namespace\\\":\\\"fs\\\"}\"}}]},\"finish_reason\":null}]}\n\n",
            "data: {\"id\":\"c1\",\"choices\":[{\"index\":0,\"delta\":{},\"finish_reason\":\"tool_calls\"}]}\n\n",
            "data: [DONE]\n\n",
        );

        let value = response_bytes_to_json(body.as_bytes(), true)
            .expect("SSE body should reassemble into JSON");

        let calls = value
            .pointer("/choices/0/message/tool_calls")
            .and_then(|v| v.as_array())
            .expect("tool_calls must be exposed at the buffered path");
        assert_eq!(calls.len(), 1);
        assert_eq!(
            calls[0].pointer("/function/name").and_then(|v| v.as_str()),
            Some("get_tools_in_namespace")
        );
        assert_eq!(
            calls[0]
                .pointer("/function/arguments")
                .and_then(|v| v.as_str()),
            Some(r#"{"namespace":"fs"}"#)
        );
    }

    /// End-to-end of the interception contract: a streamed synthetic call is resolved
    /// against the original tools rather than passed through.
    #[test]
    fn streamed_synthetic_call_resolves_and_injects_real_tools() {
        let body = concat!(
            "data: {\"id\":\"c1\",\"model\":\"m\",\"created\":1,\"choices\":[{\"index\":0,\"delta\":{\"tool_calls\":[{\"index\":0,\"id\":\"call_1\",\"type\":\"function\",\"function\":{\"name\":\"get_tools_in_namespace\",\"arguments\":\"{\\\"namespace\\\":\\\"fs\\\"}\"}}]},\"finish_reason\":\"tool_calls\"}]}\n\n",
            "data: [DONE]\n\n",
        );
        let resp_json = response_bytes_to_json(body.as_bytes(), true).expect("reassembles");

        let originals = vec![tool("fs_read"), tool("fs_write"), tool("git_log")];
        let mut req_json = json!({
            "tools": [{"type":"function","function":{"name":"get_tools_in_namespace"}}],
            "messages": [{"role":"user","content":"list files"}]
        });

        let disclosed =
            resolver::resolve_synthetic_in_response(&resp_json, &mut req_json, &originals, 0, &HashSet::new())
                .expect("synthetic call must be resolved, not relayed");

        assert!(disclosed.contains(&"fs_read".to_string()));
        assert!(disclosed.contains(&"fs_write".to_string()));

        // A tool result was fed back for the synthetic call id.
        let messages = req_json["messages"].as_array().unwrap();
        assert!(messages.iter().any(|m| {
            m.get("role").and_then(|r| r.as_str()) == Some("tool")
                && m.get("tool_call_id").and_then(|i| i.as_str()) == Some("call_1")
        }));

        // The real namespace tools became callable.
        let tool_names: Vec<&str> = req_json["tools"]
            .as_array()
            .unwrap()
            .iter()
            .filter_map(|t| t.pointer("/function/name").and_then(|n| n.as_str()))
            .collect();
        assert!(tool_names.contains(&"fs_read"));
        assert!(tool_names.contains(&"fs_write"));
    }

    /// A plain JSON (non-SSE) body still parses directly.
    #[test]
    fn json_body_parses_without_sse_path() {
        let body = json!({"choices":[{"index":0,"message":{"role":"assistant","content":"hi"}}]});
        let bytes = serde_json::to_vec(&body).unwrap();
        let value = response_bytes_to_json(&bytes, false).expect("JSON parses");
        assert_eq!(
            value.pointer("/choices/0/message/content").unwrap(),
            &json!("hi")
        );
    }

    #[test]
    fn benign_extraction_note_lists_tools_and_hides_synthetic() {
        let note = benign_extraction_note(&["fs".to_string()], &["fs_read".to_string(), "fs_write".to_string()]);
        assert!(note.contains("fs_read"));
        assert!(note.contains("fs_write"));
        assert!(note.contains("fs"));
        assert!(!note.contains("get_tools_in_namespace"));
        assert!(note.contains("Call them directly"));
    }

    #[test]
    fn append_to_json_adds_content() {
        let body = json!({"choices":[{"index":0,"message":{"role":"assistant","content":"done"},"finish_reason":"stop"}]});
        let out = append_to_json(&serde_json::to_vec(&body).unwrap(), "EXTRA");
        let value: serde_json::Value = serde_json::from_slice(&out).unwrap();
        assert_eq!(
            value.pointer("/choices/0/message/content").unwrap(),
            &json!("done\nEXTRA")
        );
    }

    #[test]
    fn append_to_sse_inserts_before_done() {
        let body = "data: {\"choices\":[{\"delta\":{\"content\":\"done\"},\"finish_reason\":\"stop\"}]}\n\ndata: [DONE]\n\n";
        let out = append_to_sse(body.as_bytes(), "EXTRA");
        let text = String::from_utf8(out).unwrap();
        assert!(text.contains("EXTRA"));
        assert!(text.contains("data: [DONE]"));
        // The EXTRA delta must come before [DONE].
        assert!(text.find("EXTRA").unwrap() < text.find("data: [DONE]").unwrap());
    }

    /// SSE bodies are detected by payload shape even when the Content-Type was lost.
    #[test]
    fn sse_detected_from_body_shape_when_header_missing() {
        let body = "data: {\"id\":\"c1\",\"choices\":[{\"index\":0,\"delta\":{\"content\":\"hi\"},\"finish_reason\":\"stop\"}]}\n\ndata: [DONE]\n\n";
        let value = response_bytes_to_json(body.as_bytes(), false)
            .expect("body-shape sniffing should still reassemble");
        assert_eq!(
            value.pointer("/choices/0/message/content").unwrap(),
            &json!("hi")
        );
    }
}

// ─── End-to-end service tests ─────────────────────────────────────────────────
//
// These drive `ToolCompressionService::call` with a mock inner service so the
// whole path is exercised: pipeline → synthetic injection → response inspection →
// resolution. Helper-level tests above cannot catch a loop that never runs.

#[cfg(test)]
mod service_tests {
    use super::*;
    use serde_json::json;
    use std::sync::atomic::{AtomicUsize, Ordering};

    /// Mock inner service. Returns a scripted response per call, recording the
    /// request bodies it saw so we can assert what the provider was actually sent.
    #[derive(Clone)]
    struct MockInner {
        /// Responses returned in order; the last is repeated once exhausted.
        responses: Arc<Vec<(String, &'static str)>>,
        calls: Arc<AtomicUsize>,
        seen_bodies: Arc<std::sync::Mutex<Vec<serde_json::Value>>>,
    }

    impl Service<Request<Body>> for MockInner {
        type Response = Response<Body>;
        type Error = std::convert::Infallible;
        type Future = Pin<Box<dyn Future<Output = Result<Self::Response, Self::Error>> + Send>>;

        fn poll_ready(&mut self, _cx: &mut Context<'_>) -> Poll<Result<(), Self::Error>> {
            Poll::Ready(Ok(()))
        }

        fn call(&mut self, req: Request<Body>) -> Self::Future {
            let responses = Arc::clone(&self.responses);
            let calls = Arc::clone(&self.calls);
            let seen = Arc::clone(&self.seen_bodies);
            Box::pin(async move {
                let body = to_bytes(req.into_body(), 64 * 1024 * 1024)
                    .await
                    .unwrap_or_default();
                if let Ok(v) = serde_json::from_slice::<serde_json::Value>(&body) {
                    seen.lock().unwrap().push(v);
                }
                let idx = calls.fetch_add(1, Ordering::SeqCst);
                let (body, ctype) = responses
                    .get(idx)
                    .cloned()
                    .unwrap_or_else(|| responses.last().cloned().unwrap());
                let resp = Response::builder()
                    .status(200)
                    .header(axum::http::header::CONTENT_TYPE, ctype)
                    .body(Body::from(body))
                    .unwrap();
                Ok(resp)
            })
        }
    }

    fn test_config() -> Arc<RwLock<crate::config::Config>> {
        let cfg: crate::config::Config = serde_json::from_value(json!({
            "server": {"host": "127.0.0.1", "port": 8080},
            "providers": [],
            "model_groups": [],
            "tool_compression": {
                "enabled": true,
                "level": "high",
                "namespace_grouping": {
                    "enabled": true,
                    "min_tools_for_grouping": 5
                }
            }
        }))
        .expect("minimal test config must deserialize");
        Arc::new(RwLock::new(cfg))
    }

    fn build_service(
        responses: Vec<(String, &'static str)>,
    ) -> (ToolCompressionService<MockInner>, MockInner) {
        let config = test_config();
        let tc = config
            .try_read()
            .map(|c| c.tool_compression.clone())
            .unwrap();
        let state = Arc::new(ToolCompressionState::new(&tc));
        let inner = MockInner {
            responses: Arc::new(responses),
            calls: Arc::new(AtomicUsize::new(0)),
            seen_bodies: Arc::new(std::sync::Mutex::new(Vec::new())),
        };
        let layer = ToolCompressionLayer::new(
            Arc::clone(&config),
            state,
            Arc::new(Metrics::new()),
            Arc::new(CompressionEventHub::new()),
        );
        let svc = layer.layer(inner.clone());
        (svc, inner)
    }

    /// 20 tools across namespaces so the grouper activates.
    fn request_with_many_tools(stream: bool) -> Request<Body> {
        let tools: Vec<serde_json::Value> = (0..10)
            .map(|i| {
                json!({
                    "type": "function",
                    "function": {
                        "name": format!("fs_op{}", i),
                        "description": "a filesystem operation",
                        "parameters": {"type":"object","properties":{"path":{"type":"string"}}}
                    }
                })
            })
            .chain((0..10).map(|i| {
                json!({
                    "type": "function",
                    "function": {
                        "name": format!("git_op{}", i),
                        "description": "a git operation",
                        "parameters": {"type":"object","properties":{"ref":{"type":"string"}}}
                    }
                })
            }))
            .collect();

        let body = json!({
            "model": "gpt-4o",
            "stream": stream,
            "messages": [{"role":"user","content":"do something"}],
            "tools": tools
        });

        Request::builder()
            .method("POST")
            .uri("/v1/chat/completions")
            .header(axum::http::header::CONTENT_TYPE, "application/json")
            .body(Body::from(serde_json::to_vec(&body).unwrap()))
            .unwrap()
    }

    fn sse_tool_call(name: &str, args: &str) -> String {
        format!(
            "data: {}\n\ndata: [DONE]\n\n",
            json!({
                "id": "c1",
                "model": "gpt-4o",
                "created": 1,
                "choices": [{
                    "index": 0,
                    "delta": {"tool_calls": [{
                        "index": 0,
                        "id": "call_1",
                        "type": "function",
                        "function": {"name": name, "arguments": args}
                    }]},
                    "finish_reason": "tool_calls"
                }]
            })
        )
    }

    fn json_tool_call(name: &str, args: &str) -> String {
        json!({
            "id": "c1",
            "choices": [{
                "index": 0,
                "message": {
                    "role": "assistant",
                    "content": null,
                    "tool_calls": [{
                        "id": "call_1",
                        "type": "function",
                        "function": {"name": name, "arguments": args}
                    }]
                },
                "finish_reason": "tool_calls"
            }]
        })
        .to_string()
    }

    fn json_text(text: &str) -> String {
        json!({
            "id": "c1",
            "choices": [{
                "index": 0,
                "message": {"role": "assistant", "content": text},
                "finish_reason": "stop"
            }]
        })
        .to_string()
    }

    async fn body_string(resp: Response<Body>) -> String {
        let bytes = to_bytes(resp.into_body(), 64 * 1024 * 1024).await.unwrap();
        String::from_utf8_lossy(&bytes).to_string()
    }

    /// Sanity: the grouper actually collapses the tools and injects the synthetic
    /// drill-down tool. If this fails, nothing downstream can be trusted.
    #[tokio::test]
    async fn grouper_collapses_tools_and_injects_synthetic() {
        let (mut svc, inner) = build_service(vec![(json_text("hi"), "application/json")]);
        let _ = svc.call(request_with_many_tools(false)).await.unwrap();

        let seen = inner.seen_bodies.lock().unwrap();
        let sent = seen.first().expect("inner must have been called");
        let names: Vec<String> = sent["tools"]
            .as_array()
            .unwrap()
            .iter()
            .filter_map(|t| {
                t.pointer("/function/name")
                    .and_then(|n| n.as_str())
                    .map(|s| s.to_string())
            })
            .collect();

        assert!(
            names.len() < 20,
            "grouper must reduce the 20-tool payload, got {names:?}"
        );
        assert!(
            names.iter().any(|n| n == resolver::GET_TOOLS_IN_NAMESPACE),
            "synthetic drill-down tool must be offered, got {names:?}"
        );
    }

    /// THE regression: a non-streaming response calling the synthetic tool must be
    /// resolved internally and must never reach the client.
    #[tokio::test]
    async fn json_synthetic_call_is_not_relayed_to_client() {
        let (mut svc, _inner) = build_service(vec![
            (
                json_tool_call(resolver::GET_TOOLS_IN_NAMESPACE, r#"{"namespace":"fs"}"#),
                "application/json",
            ),
            (json_text("done"), "application/json"),
        ]);

        let resp = svc.call(request_with_many_tools(false)).await.unwrap();
        let body = body_string(resp).await;

        assert!(
            !body.contains(resolver::GET_TOOLS_IN_NAMESPACE),
            "synthetic call leaked to client: {body}"
        );
    }

    /// The discovery must surface in the client's transcript as a benign note (so the
    /// model remembers it already extracted the tools) WITHOUT leaking the synthetic tool
    /// name or raw schemas — a router consumer should never see internal plumbing.
    #[tokio::test]
    async fn benign_extraction_note_injected_without_synthetic_leak() {
        let (mut svc, _inner) = build_service(vec![
            (
                sse_tool_call(resolver::GET_TOOLS_IN_NAMESPACE, r#"{"namespace":"fs"}"#),
                "text/event-stream",
            ),
            (
                "data: {\"id\":\"c1\",\"choices\":[{\"index\":0,\"delta\":{\"content\":\"all done\"},\"finish_reason\":\"stop\"}]}\n\ndata: [DONE]\n\n".to_string(),
                "text/event-stream",
            ),
        ]);

        let resp = svc.call(request_with_many_tools(true)).await.unwrap();
        let body = body_string(resp).await;

        assert!(
            !body.contains(resolver::GET_TOOLS_IN_NAMESPACE),
            "synthetic call leaked to client: {body}"
        );
        assert!(
            body.contains("extracted"),
            "benign extraction note missing from client transcript: {body}"
        );
        assert!(
            body.contains("fs_op0"),
            "extracted tool name should be listed in the note: {body}"
        );
        assert!(
            body.contains("all done"),
            "model's real answer must still be present: {body}"
        );
    }

    /// THE reported failure: client requested streaming, so the gateway replies with
    /// SSE. The synthetic call must still be intercepted.
    #[tokio::test]
    async fn sse_synthetic_call_is_not_relayed_to_client() {
        let (mut svc, _inner) = build_service(vec![
            (
                sse_tool_call(resolver::GET_TOOLS_IN_NAMESPACE, r#"{"namespace":"fs"}"#),
                "text/event-stream",
            ),
            (
                "data: {\"id\":\"c1\",\"choices\":[{\"index\":0,\"delta\":{\"content\":\"done\"},\"finish_reason\":\"stop\"}]}\n\ndata: [DONE]\n\n".to_string(),
                "text/event-stream",
            ),
        ]);

        let resp = svc.call(request_with_many_tools(true)).await.unwrap();
        let body = body_string(resp).await;

        assert!(
            !body.contains(resolver::GET_TOOLS_IN_NAMESPACE),
            "synthetic call leaked to client over SSE: {body}"
        );
    }

    /// A namespace summary pseudo-tool called directly must also be intercepted.
    #[tokio::test]
    async fn ns_prefix_call_is_not_relayed_to_client() {
        let (mut svc, _inner) = build_service(vec![
            (sse_tool_call("ns_other", "{}"), "text/event-stream"),
            (json_text("done"), "application/json"),
        ]);

        let resp = svc.call(request_with_many_tools(true)).await.unwrap();
        let body = body_string(resp).await;

        assert!(
            !body.contains("\"ns_other\""),
            "ns_other call leaked to client: {body}"
        );
    }

    /// A model stuck in a synthetic-call loop must not leave a synthetic call in the
    /// final response once the step budget is exhausted.
    #[tokio::test]
    async fn exhausted_budget_does_not_leak_synthetic_call() {
        // Always answer with the same synthetic call, far more than the budget.
        let repeated = (
            sse_tool_call(resolver::GET_TOOLS_IN_NAMESPACE, r#"{"namespace":"fs"}"#),
            "text/event-stream",
        );
        let (mut svc, _inner) = build_service(vec![repeated]);

        let resp = svc.call(request_with_many_tools(true)).await.unwrap();
        let body = body_string(resp).await;

        assert!(
            !body.contains(resolver::GET_TOOLS_IN_NAMESPACE),
            "synthetic call leaked after budget exhaustion: {body}"
        );
    }

    /// Mirrors the reported production shape: a large multi-namespace tool set where
    /// the model drills into many namespaces in sequence. The scaled budget must absorb
    /// legitimate exploration, and no synthetic call may reach the client either way.
    #[tokio::test]
    async fn many_namespace_exploration_completes_without_leaking() {
        // 12 namespaces × 3 tools — comparable to a real MCP-heavy tool set.
        let namespaces = [
            "magicuidesign",
            "agent",
            "aws",
            "chrome",
            "context7",
            "kilo",
            "mui",
            "nanogpt",
            "searxng",
            "shadcn",
            "spec",
            "misc",
        ];
        let tools: Vec<serde_json::Value> = namespaces
            .iter()
            .flat_map(|ns| {
                (0..3).map(move |i| {
                    json!({
                        "type": "function",
                        "function": {
                            "name": format!("{ns}_tool{i}"),
                            "description": "a tool",
                            "parameters": {"type":"object","properties":{}}
                        }
                    })
                })
            })
            .collect();

        let body = json!({
            "model": "gpt-4o",
            "stream": true,
            "messages": [{"role":"user","content":"explore"}],
            "tools": tools
        });
        let req = Request::builder()
            .method("POST")
            .uri("/v1/chat/completions")
            .header(axum::http::header::CONTENT_TYPE, "application/json")
            .body(Body::from(serde_json::to_vec(&body).unwrap()))
            .unwrap();

        // Model drills into 10 namespaces one per turn, then answers normally.
        let mut responses: Vec<(String, &'static str)> = namespaces
            .iter()
            .take(10)
            .map(|ns| {
                (
                    sse_tool_call(
                        resolver::GET_TOOLS_IN_NAMESPACE,
                        &format!(r#"{{"namespace":"{ns}"}}"#),
                    ),
                    "text/event-stream",
                )
            })
            .collect();
        responses.push((
            "data: {\"id\":\"c1\",\"choices\":[{\"index\":0,\"delta\":{\"content\":\"all done\"},\"finish_reason\":\"stop\"}]}\n\ndata: [DONE]\n\n".to_string(),
            "text/event-stream",
        ));

        let (mut svc, _inner) = build_service(responses);
        let resp = svc.call(req).await.unwrap();
        let body = body_string(resp).await;

        assert!(
            !body.contains(resolver::GET_TOOLS_IN_NAMESPACE),
            "synthetic call leaked during multi-namespace exploration: {body}"
        );
        assert!(
            body.contains("all done"),
            "exploration should finish on the model's real answer, got: {body}"
        );
    }

    /// The budget must scale with namespace count rather than staying at the old
    /// fixed 6, which real multi-namespace tool sets exhausted.
    #[test]
    fn step_budget_scales_with_namespace_count() {
        assert_eq!(synthetic_step_budget(0), MAX_SYNTHETIC_RESOLUTION_STEPS);
        assert_eq!(synthetic_step_budget(1), MAX_SYNTHETIC_RESOLUTION_STEPS);
        assert!(
            synthetic_step_budget(12) > MAX_SYNTHETIC_RESOLUTION_STEPS,
            "12 namespaces must get more than the baseline budget"
        );
        assert_eq!(
            synthetic_step_budget(10_000),
            MAX_SYNTHETIC_RESOLUTION_STEPS_CEILING,
            "budget must stay bounded"
        );
    }

    /// Ordinary streaming traffic (no synthetic tools involved) must pass through
    /// untouched — the fix must not buffer or rewrite normal responses.
    #[tokio::test]
    async fn real_tool_call_passes_through_unchanged() {
        let (mut svc, _inner) = build_service(vec![(
            sse_tool_call("fs_op1", r#"{"path":"a.txt"}"#),
            "text/event-stream",
        )]);

        let resp = svc.call(request_with_many_tools(true)).await.unwrap();
        let body = body_string(resp).await;

        assert!(
            body.contains("fs_op1"),
            "real tool call must reach the client: {body}"
        );
    }
}
