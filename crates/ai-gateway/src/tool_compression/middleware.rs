//! Tower middleware for tool definition compression.
//!
//! `ToolCompressionLayer` wraps the standard `Layer` + `Service` pattern.
//! When `enabled = false` at construction time, the service is a zero-cost
//! passthrough — no allocations, no JSON parsing, no body reads on the hot path.

use std::{
    collections::{HashMap, HashSet},
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
            // Fallback: clients that omit x-session-id (common for anonymous API-key
            // traffic) would otherwise get zero multi-turn disclosure memory and
            // re-drill every namespace on every request. Derive a stable,
            // non-reversible bucket from the Authorization header so disclosure
            // state persists per key, mirroring loop_detection's vk-id fallback.
            let session_id = parts
                .headers
                .get(HEADER_SESSION_ID)
                .and_then(|v| v.to_str().ok())
                .map(|s| s.to_string())
                .or_else(|| {
                    parts
                        .headers
                        .get(axum::http::header::AUTHORIZATION)
                        .and_then(|v| v.to_str().ok())
                        .map(|auth| format!("vk-{:016x}", hash_auth_value(auth)))
                });

            // Register the session before any per-session map is touched so the
            // LRU registry can bound their growth. Session ids are
            // caller-supplied, so this is what keeps them from accumulating for
            // the process lifetime.
            if let Some(sid) = &session_id {
                state.touch_session(sid);
            }

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

            // Strengthen session memory: annotate the synthetic drill-down tool descriptions
            // with tools already extracted this session, so the model sees them in its tool
            // list every turn and stops re-drilling. Nothing is appended to message outputs.
            if let Some(sid) = &session_id {
                if let Some(disclosed) = state.disclosure_state.get(sid) {
                    if !disclosed.is_empty() {
                        annotate_synthetic_descriptions(&mut tools, &disclosed);
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
            // Set once the discovery tools have been pulled from the outbound
            // request for the final turn — see the budget branch below.
            let mut synthetic_tools_withdrawn = false;
            let mut already_disclosed: HashSet<String> = session_id
                .as_ref()
                .and_then(|sid| state.disclosure_targets.get(sid).map(|set| set.clone()))
                .unwrap_or_default();
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

                if steps >= step_budget {
                    if synthetic_tools_withdrawn {
                        // The discovery tools are already gone from the outbound
                        // request, so a synthetic call here should be impossible.
                        // If one appears anyway it must still never be relayed —
                        // the client cannot execute it. `sanitize_synthetic_response`
                        // returns `None` when there is no synthetic call, which is
                        // the normal exit and relays the model's real answer
                        // untouched.
                        match sanitize_synthetic_response(&resp_json, &original_tools, sse) {
                            Some(sanitized) => {
                                tracing::warn!(
                                    model_group = %model_group,
                                    steps,
                                    "Model emitted a synthetic call after the discovery tools were withdrawn; answering locally rather than relaying a tool the client cannot execute"
                                );
                                let mut parts = resp_parts;
                                parts.headers.remove(axum::http::header::CONTENT_LENGTH);
                                final_response =
                                    Response::from_parts(parts, Body::from(sanitized));
                            }
                            None => {
                                final_response =
                                    Response::from_parts(resp_parts, Body::from(resp_bytes));
                            }
                        }
                        break;
                    }

                    // Budget spent. Terminating here is what strands the client: a
                    // gateway-authored assistant message reads as the final answer
                    // and the agent stops mid-task. Instead withdraw the discovery
                    // tools so the model cannot ask again, let the pending call
                    // resolve below so it has the schemas, and give it one final
                    // turn to either answer or call a real tool.
                    tracing::debug!(
                        model_group = %model_group,
                        steps,
                        "Synthetic drill-down budget exhausted; withdrawing the discovery tools for one final turn"
                    );
                    withdraw_synthetic_tools(&mut req_json, &original_tools);
                    synthetic_tools_withdrawn = true;
                }

                // ─── Re-drill detection ───────────────────────────────────
                // Every synthetic call this turn targets something already
                // disclosed AND already callable. Record it so chronic
                // re-drillers get their compression level stepped down, then fall
                // through to normal resolution.
                //
                // This used to answer locally and terminate the turn with
                // `finish_reason: "stop"` to save a provider round trip. That
                // saving is not actually available: a tool call the gateway
                // answers still needs a model turn to act on the answer. So
                // terminating handed the client a gateway-authored assistant
                // message ("Session cache hit: …") which harnesses render as the
                // final reply and stop on — killing the agent's task mid-work to
                // avoid one provider call. `resolve_synthetic_in_response` already
                // appends the sharper `REDRILL_HINT` for disclosed targets, so the
                // resubmit below carries the reminder that was the point.
                if is_pure_redrill(&resp_json, &req_json, &original_tools, &already_disclosed) {
                    feedback_loop.record_outcome(&model_group, true);
                    tracing::debug!(
                        model_group = %model_group,
                        "Model re-drilled targets already disclosed and callable; resolving with the re-drill reminder instead of ending the turn"
                    );
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
                                    let Some(name) = fn_obj.get("name").and_then(|n| n.as_str())
                                    else {
                                        continue;
                                    };
                                    let args = fn_obj
                                        .get("arguments")
                                        .and_then(|a| a.as_str())
                                        .unwrap_or("{}");
                                    if let Some(key) = resolver::discovery_key(name, args) {
                                        already_disclosed.insert(key.clone());
                                        targets.insert(key);
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

            let mut response = final_response;

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

/// Hash an Authorization header value into a stable, non-reversible session
/// bucket key. The raw credential is never stored or logged — only this digest
/// is used as the fallback disclosure-session identifier.
fn hash_auth_value(auth: &str) -> u64 {
    use std::hash::{Hash, Hasher};
    let mut hasher = std::collections::hash_map::DefaultHasher::new();
    auth.hash(&mut hasher);
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
        "Tool discovery could not be completed within the configured budget. {} Call the listed tools directly by name.",
        notes.join(" ")
    );

    frame_local_assistant_response(resp_json, &content, sse)
}

/// Withdraw the compression stages' synthetic drill-down tools from an outbound
/// request body, so the model cannot ask for another disclosure.
///
/// Used for the final turn once the resolution budget is spent. If withdrawal would
/// leave no tools at all — every real tool still hidden behind a namespace — the
/// original definitions are restored instead: the budget is spent either way, and a
/// final turn where the model can call a real tool beats one where it can only
/// reply in prose. That trades the compression saving on this single turn for the
/// agent being able to keep working.
fn withdraw_synthetic_tools(req_json: &mut serde_json::Value, original_tools: &[ToolDefinition]) {
    let Some(object) = req_json.as_object_mut() else {
        return;
    };
    let emptied = {
        let Some(tools) = object.get_mut("tools").and_then(|v| v.as_array_mut()) else {
            return;
        };
        tools.retain(|tool| {
            tool.pointer("/function/name")
                .and_then(|n| n.as_str())
                .map(|name| {
                    name != resolver::GET_TOOLS_IN_NAMESPACE
                        && name != resolver::GET_TOOL_SCHEMA
                        && !name.starts_with(resolver::NS_PREFIX)
                })
                .unwrap_or(true)
        });
        tools.is_empty()
    };
    if emptied {
        if original_tools.is_empty() {
            // Several providers reject `tools: []`.
            object.remove("tools");
        } else {
            object.insert(
                "tools".to_string(),
                serde_json::Value::Array(
                    original_tools.iter().map(|tool| tool.raw.clone()).collect(),
                ),
            );
        }
    }
}

/// Check whether every synthetic call in the response is a pure re-drill:
/// a discovery target already disclosed this session whose underlying tools
/// are already callable in the current request's tool list. In that case a
/// provider round-trip would return information the model already has, so
/// the turn is answered locally and terminated instead — saving one full
/// provider call per repeat.
///
/// Re-drills are also recorded into the `FeedbackLoop` as error outcomes so
/// chronic re-drillers get their compression level stepped down (fewer
/// synthetic tools offered → fewer opportunities to re-drill).
///
/// Returns `true` when the turn is nothing but re-drills, so the caller can
/// record the wasted round trip. Resolution proceeds either way — the answer to a
/// tool call only reaches the model on its next turn, so there is no way to
/// "answer locally" without either resubmitting or ending the client's task.
fn is_pure_redrill(
    resp_json: &serde_json::Value,
    req_json: &serde_json::Value,
    original_tools: &[ToolDefinition],
    already_disclosed: &HashSet<String>,
) -> bool {
    let Some(calls) = resp_json
        .pointer("/choices/0/message/tool_calls")
        .and_then(|v| v.as_array())
    else {
        return false;
    };

    // Names callable in the current (possibly re-injected) outbound request.
    let callable_names: HashSet<String> = req_json
        .get("tools")
        .and_then(|v| v.as_array())
        .map(|arr| {
            arr.iter()
                .filter_map(|t| {
                    t.pointer("/function/name")
                        .and_then(|n| n.as_str())
                        .map(String::from)
                })
                .collect()
        })
        .unwrap_or_default();

    let mut saw_synthetic = false;
    let mut all_redrills = true;

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
            // A real tool call mixed in: the provider turn is doing useful work.
            all_redrills = false;
            continue;
        }
        saw_synthetic = true;

        let Some(key) = resolver::discovery_key(name, args) else {
            all_redrills = false;
            continue;
        };
        if !already_disclosed.contains(&key) {
            // First drill for this target — resolve normally.
            all_redrills = false;
            continue;
        }

        // Target was disclosed, but it only counts as a wasted re-drill when every
        // underlying tool is already callable this turn. Otherwise the drill is
        // doing real work: resolution still has schemas to re-inject.
        let members: Vec<String> = if let Some(ns) = key.strip_prefix("ns:") {
            resolver::tools_in_namespace(ns, original_tools)
                .iter()
                .filter_map(|t| {
                    t.pointer("/function/name")
                        .and_then(|n| n.as_str())
                        .map(String::from)
                })
                .collect()
        } else if let Some(tool_name) = key.strip_prefix("tool:") {
            vec![tool_name.to_string()]
        } else {
            Vec::new()
        };
        if members.is_empty() || !members.iter().all(|m| callable_names.contains(m)) {
            all_redrills = false;
        }
    }

    saw_synthetic && all_redrills
}

/// Frame a locally generated assistant message in the response shape the client
/// expects (SSE chunk stream or plain JSON), terminating the turn with
/// `finish_reason: "stop"`.
fn frame_local_assistant_response(
    resp_json: &serde_json::Value,
    content: &str,
    sse: bool,
) -> Option<Vec<u8>> {
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

/// Annotate synthetic drill-down tool descriptions with the tools already extracted this
/// session. For each `ns_<prefix>` tool, append the already-extracted tools in that
/// namespace; for `get_tools_in_namespace`, append a summary of every extracted namespace.
///
/// This keeps the reminder in the model's tool list (visible every turn) instead of in the
/// chat output, so re-drills are discouraged without spamming the transcript with a note on
/// every response.
fn annotate_synthetic_descriptions(tools: &mut [ToolDefinition], disclosed: &HashSet<String>) {
    if disclosed.is_empty() {
        return;
    }
    let mut by_ns: HashMap<String, Vec<String>> = HashMap::new();
    for name in disclosed {
        let ns = resolver::namespace_of(name).unwrap_or_else(|| "other".to_string());
        by_ns.entry(ns).or_default().push(name.clone());
    }
    for names in by_ns.values_mut() {
        names.sort();
        names.dedup();
    }

    for tool in tools.iter_mut() {
        if tool.name == resolver::GET_TOOLS_IN_NAMESPACE {
            if let Some(desc) = tool.raw.pointer_mut("/function/description") {
                if let Some(existing) = desc.as_str().map(str::to_string) {
                    let note = build_disclosure_note(&by_ns);
                    *desc = serde_json::Value::String(format!("{existing}\n\n{note}"));
                }
            }
        } else if let Some(ns) = tool.name.strip_prefix(resolver::NS_PREFIX) {
            if let Some(names) = by_ns.get(ns) {
                if let Some(desc) = tool.raw.pointer_mut("/function/description") {
                    if let Some(existing) = desc.as_str().map(str::to_string) {
                        let listed = names
                            .iter()
                            .take(12)
                            .cloned()
                            .collect::<Vec<_>>()
                            .join(", ");
                        let more = if names.len() > 12 {
                            format!(" (and {} more)", names.len() - 12)
                        } else {
                            String::new()
                        };
                        *desc = serde_json::Value::String(format!(
                            "{existing}\n\nAlready extracted this session: {listed}{more}. \
                             Call these tools directly by name — do not re-discover this namespace."
                        ));
                    }
                }
            }
        }
    }
}

fn build_disclosure_note(by_ns: &HashMap<String, Vec<String>>) -> String {
    let mut parts: Vec<String> = Vec::new();
    for (ns, names) in by_ns {
        let listed = names
            .iter()
            .take(12)
            .cloned()
            .collect::<Vec<_>>()
            .join(", ");
        let more = if names.len() > 12 {
            format!(" (and {} more)", names.len() - 12)
        } else {
            String::new()
        };
        parts.push(format!("{ns}: {listed}{more}"));
    }
    parts.sort();
    if parts.is_empty() {
        String::new()
    } else {
        format!(
            "Already extracted this session — call directly: {}. Do not re-discover these namespaces.",
            parts.join("; ")
        )
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

        let disclosed = resolver::resolve_synthetic_in_response(
            &resp_json,
            &mut req_json,
            &originals,
            0,
            &HashSet::new(),
        )
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
    fn annotate_synthetic_descriptions_lists_session_tools() {
        let mut tools = vec![
            ToolDefinition {
                raw: json!({
                    "type": "function",
                    "function": {
                        "name": "ns_fs",
                        "description": "namespace: fs (2 tools) - Tools: fs_read, fs_write"
                    }
                }),
                name: "ns_fs".to_string(),
                content_hash: 0,
            },
            ToolDefinition {
                raw: json!({
                    "type": "function",
                    "function": {
                        "name": "get_tools_in_namespace",
                        "description": "Retrieve all tools in a specific namespace."
                    }
                }),
                name: "get_tools_in_namespace".to_string(),
                content_hash: 0,
            },
        ];
        let disclosed: HashSet<String> = ["fs_read".to_string(), "fs_write".to_string()]
            .into_iter()
            .collect();

        annotate_synthetic_descriptions(&mut tools, &disclosed);

        let ns_desc = tools[0]
            .raw
            .pointer("/function/description")
            .and_then(|v| v.as_str())
            .unwrap();
        assert!(
            ns_desc.contains("fs_read") && ns_desc.contains("fs_write"),
            "ns_fs description must list extracted tools: {ns_desc}"
        );
        assert!(
            ns_desc.contains("do not re-discover"),
            "ns_fs description must discourage re-discovery: {ns_desc}"
        );

        let gtin_desc = tools[1]
            .raw
            .pointer("/function/description")
            .and_then(|v| v.as_str())
            .unwrap();
        assert!(
            gtin_desc.contains("fs: fs_read, fs_write"),
            "get_tools_in_namespace must summarise extracted namespaces: {gtin_desc}"
        );
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

    /// THE redrill regression: drilling the same namespace again on a later request
    /// in the same session must be resolved and resubmitted, exactly like a first
    /// drill, so the model gets its (re-drill-flagged) answer and can keep working.
    ///
    /// This previously short-circuited into a gateway-authored assistant message
    /// ("Session cache hit: …") with `finish_reason: "stop"` to save the provider
    /// round trip. That saving does not exist — a tool call the gateway answers
    /// still needs a model turn to act on the answer — and harnesses render the
    /// synthetic message as the final reply and stop, ending the task mid-work.
    #[tokio::test]
    async fn redrill_is_resolved_and_resubmitted_not_answered_locally() {
        let config = test_config();
        let tc = config
            .try_read()
            .map(|c| c.tool_compression.clone())
            .unwrap();
        let state = Arc::new(ToolCompressionState::new(&tc));
        let inner = MockInner {
            responses: Arc::new(vec![
                (
                    json_tool_call(resolver::GET_TOOLS_IN_NAMESPACE, r#"{"namespace":"fs"}"#),
                    "application/json",
                ),
                (json_text("ok"), "application/json"),
                (
                    json_tool_call(resolver::GET_TOOLS_IN_NAMESPACE, r#"{"namespace":"fs"}"#),
                    "application/json",
                ),
                // The re-drill is resolved and resubmitted; this is the turn the
                // model gets to actually continue on.
                (json_text("continuing"), "application/json"),
            ]),
            calls: Arc::new(AtomicUsize::new(0)),
            seen_bodies: Arc::new(std::sync::Mutex::new(Vec::new())),
        };
        let layer = ToolCompressionLayer::new(
            Arc::clone(&config),
            Arc::clone(&state),
            Arc::new(Metrics::new()),
            Arc::new(CompressionEventHub::new()),
        );

        let req = || {
            Request::builder()
                .method("POST")
                .uri("/v1/chat/completions")
                .header(axum::http::header::CONTENT_TYPE, "application/json")
                .header(HEADER_SESSION_ID, "sess-redrill")
                .body(Body::from(
                    serde_json::to_vec(&many_tools_body(false)).unwrap(),
                ))
                .unwrap()
        };

        // Request 1: drill fs → resolved → one follow-up turn finishing on "ok".
        let mut svc1 = layer.layer(inner.clone());
        let _ = svc1.call(req()).await.unwrap();

        // Request 2: model re-drills fs (inner call #3). The middleware must resolve
        // it and resubmit (call #4) rather than ending the turn locally.
        let mut svc2 = layer.layer(inner.clone());
        let resp = svc2.call(req()).await.unwrap();
        let body = body_string(resp).await;

        assert!(
            !body.contains("Session cache hit"),
            "the gateway must not hand the client a synthetic assistant turn: {body}"
        );
        assert!(
            !body.contains(resolver::GET_TOOLS_IN_NAMESPACE),
            "synthetic re-drill leaked to client: {body}"
        );
        assert_eq!(
            inner.calls.load(Ordering::SeqCst),
            4,
            "re-drill must be resolved and resubmitted so the model can continue"
        );

        // The resubmitted request carries the re-drill reminder as a tool result,
        // which is the reminder the local short-circuit was trying to deliver.
        let bodies = inner.seen_bodies.lock().unwrap().clone();
        let last = bodies
            .last()
            .expect("a resubmitted body must exist")
            .to_string();
        assert!(
            last.contains("Session cache"),
            "resubmit must carry the re-drill reminder to the model"
        );
    }

    /// Once the resolution budget is spent the middleware must withdraw the
    /// discovery tools and take one final turn, not terminate with a
    /// gateway-authored assistant message. The client's task has to be able to
    /// continue.
    #[tokio::test]
    async fn exhausted_discovery_budget_withdraws_tools_and_takes_a_final_turn() {
        let config = test_config();
        let tc = config
            .try_read()
            .map(|c| c.tool_compression.clone())
            .unwrap();
        let state = Arc::new(ToolCompressionState::new(&tc));
        // The model drills forever; the middleware must stop it and still end on a
        // usable turn. Plenty of synthetic responses, then a real answer.
        let drill = || {
            (
                json_tool_call(resolver::GET_TOOLS_IN_NAMESPACE, r#"{"namespace":"fs"}"#),
                "application/json",
            )
        };
        let inner = MockInner {
            responses: Arc::new(vec![
                drill(),
                drill(),
                drill(),
                drill(),
                drill(),
                drill(),
                drill(),
                drill(),
                (json_text("done"), "application/json"),
            ]),
            calls: Arc::new(AtomicUsize::new(0)),
            seen_bodies: Arc::new(std::sync::Mutex::new(Vec::new())),
        };
        let layer = ToolCompressionLayer::new(
            Arc::clone(&config),
            Arc::clone(&state),
            Arc::new(Metrics::new()),
            Arc::new(CompressionEventHub::new()),
        );

        let mut svc = layer.layer(inner.clone());
        let resp = svc
            .call(
                Request::builder()
                    .method("POST")
                    .uri("/v1/chat/completions")
                    .header(axum::http::header::CONTENT_TYPE, "application/json")
                    .header(HEADER_SESSION_ID, "sess-budget")
                    .body(Body::from(
                        serde_json::to_vec(&many_tools_body(false)).unwrap(),
                    ))
                    .unwrap(),
            )
            .await
            .unwrap();
        let body = body_string(resp).await;

        assert!(
            !body.contains(resolver::GET_TOOLS_IN_NAMESPACE),
            "synthetic call must never reach the client: {body}"
        );
        // The last outbound request must have had the discovery tools withdrawn, so
        // the model physically cannot drill again on its final turn. Assert on the
        // `tools` array specifically — the message history legitimately still
        // mentions the synthetic name in the assistant's earlier tool_calls.
        let bodies = inner.seen_bodies.lock().unwrap().clone();
        let last = bodies.last().expect("a resubmitted body must exist").clone();
        let offered: Vec<String> = last
            .get("tools")
            .and_then(|t| t.as_array())
            .map(|arr| {
                arr.iter()
                    .filter_map(|t| {
                        t.pointer("/function/name")
                            .and_then(|n| n.as_str())
                            .map(String::from)
                    })
                    .collect()
            })
            .unwrap_or_default();
        assert!(
            !offered.iter().any(|name| name
                == resolver::GET_TOOLS_IN_NAMESPACE
                || name == resolver::GET_TOOL_SCHEMA
                || name.starts_with(resolver::NS_PREFIX)),
            "discovery tools must be withdrawn for the final turn, got: {offered:?}"
        );
        assert!(
            !offered.is_empty(),
            "the final turn must still offer real tools so the model can act"
        );
    }

    /// Request body with 20 tools across two namespaces so the grouper activates.
    fn many_tools_body(stream: bool) -> serde_json::Value {
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
        json!({
            "model": "gpt-4o",
            "stream": stream,
            "messages": [{"role":"user","content":"do something"}],
            "tools": tools
        })
    }
}
