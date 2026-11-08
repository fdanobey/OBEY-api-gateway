//! Gateway-side telemetry wiring (opentelemetry-trace-export spec,
//! task 6): the cheap per-request config snapshot, inbound W3C
//! context extraction, and the sample-and-emit gate invoked from the
//! completion seams in `handlers.rs` (`log_outcome` and the
//! abandoned-request logger). Disabled path is a single relaxed
//! atomic load; emission is synchronous, allocation-bounded, and can
//! never fail a request (queue drops are counted inside the
//! exporter).

use std::sync::atomic::{AtomicBool, Ordering};
use std::time::{Duration, SystemTime};

use crate::config::LoggingConfig;
use crate::error::ProviderAttempt;
use crate::telemetry::config::TelemetryConfig;
use crate::telemetry::exporter::TelemetryExporter;
use crate::telemetry::propagation::{new_trace_id, parse_traceparent, TraceId, W3cContext};
use crate::telemetry::sampler::Sampler;
use crate::telemetry::otlp::SpanData;
use crate::telemetry::spans::{
    build_request_spans, AttemptCtx, AttemptOutcome, ContentCtx, RequestCompletionCtx, UsageCtx,
};

/// Lock-friendly hot-path snapshot of the telemetry settings (mirrors
/// the `RuntimeLimits` pattern: refreshed at startup and hot-reload,
/// read per request without taking the config lock). `enabled == false`
/// keeps the disabled path to one atomic load.
pub struct TelemetrySnapshot {
    enabled: AtomicBool,
    inner: std::sync::RwLock<TelemetrySnapshotInner>,
}

struct TelemetrySnapshotInner {
    sampling_ratio: f64,
    sampler: Sampler,
    capture_content: bool,
    content_limit_bytes: usize,
}

impl TelemetrySnapshot {
    pub fn new() -> Self {
        Self {
            enabled: AtomicBool::new(false),
            inner: std::sync::RwLock::new(TelemetrySnapshotInner {
                sampling_ratio: 1.0,
                sampler: Sampler::new(1.0),
                capture_content: false,
                content_limit_bytes: 10_000,
            }),
        }
    }

    /// Refresh from the telemetry + logging config sections. Header
    /// env-resolution is not needed here: the exporter resolves
    /// headers itself in `ExportTarget::from_config` at apply time.
    pub fn apply(&self, telemetry: &TelemetryConfig, logging: &LoggingConfig) {
        // Mirror the exporter's dormancy rule (Error Scenario 3): an
        // enabled section with an unusable endpoint must not build
        // spans that can only be dropped.
        let endpoint_usable = telemetry
            .endpoint
            .as_deref()
            .map(|endpoint| {
                !endpoint.trim().is_empty()
                    && reqwest::Url::parse(endpoint.trim())
                        .map(|url| matches!(url.scheme(), "http" | "https"))
                        .unwrap_or(false)
            })
            .unwrap_or(false);
        let enabled = telemetry.enabled && endpoint_usable;
        let mut inner = self.inner.write().unwrap_or_else(|poisoned| poisoned.into_inner()); // fail-open on lock poison
        inner.sampling_ratio = telemetry.sampling_ratio;
        inner.sampler = Sampler::new(telemetry.sampling_ratio);
        inner.capture_content = telemetry.capture_content;
        inner.content_limit_bytes = logging.max_body_size_bytes;
        self.enabled.store(enabled, Ordering::Relaxed);
    }

    #[inline]
    pub fn enabled(&self) -> bool {
        self.enabled.load(Ordering::Relaxed)
    }

    /// Whether content capture is on AND telemetry is enabled — the
    /// single gate callers use before touching request/response bodies
    /// (Req 3.5/3.6). Reading it takes the same read lock the emit
    /// path takes, so it is only called after the cheap `enabled()`
    /// check passed.
    pub fn capture_content(&self) -> bool {
        if !self.enabled() {
            return false;
        }
        self.inner
            .read()
            .unwrap_or_else(|poisoned| poisoned.into_inner()) // fail-open on lock poison
            .capture_content
    }
}

/// Parse the inbound `traceparent` header at request entry (Req 4):
/// valid → the W3C context whose trace id + sampled flag drive span
/// identity and parent-based sampling; absent or invalid → `None`
/// (invalid values additionally log at debug level, Req 4.3). The
/// header is not read at all while telemetry is disabled (Req 4.4).
pub fn extract_inbound_context(
    headers: &axum::http::HeaderMap,
    telemetry_enabled: bool,
) -> Option<W3cContext> {
    if !telemetry_enabled {
        return None;
    }
    let raw = headers.get("traceparent")?;
    let value = raw.to_str().ok()?;
    if value.trim().is_empty() {
        return None;
    }
match parse_traceparent(value) {
Some(context) => Some(context),
None => {
tracing::debug!(
value = value.chars().take(64).collect::<String>(),
"ignoring invalid traceparent header"
);
None
}
}
}

impl TelemetrySnapshot {
    /// snapshot in effect at its START, not the live global state at
    /// completion time. Cheap: one read-lock acquisition per request.
    /// Returns `None` while telemetry is disabled so the disabled path
    /// stays a single atomic check (NFR performance).
    pub fn emission_config(&self) -> Option<EmissionConfig> {
        if !self.enabled() {
            return None;
        }
        let inner = self.inner.read().unwrap_or_else(|poisoned| poisoned.into_inner()); // fail-open on lock poison
        Some(EmissionConfig {
            sampler: inner.sampler.clone(),
            capture_content: inner.capture_content,
            content_limit_bytes: inner.content_limit_bytes,
        })
    }
}

/// The immutable per-request telemetry settings (Req 9.3): captured at
/// entry from the snapshot and carried by the request guard so
/// completion-time emission uses entry-time sampling/capture decisions
/// even if the global snapshot was hot-reloaded mid-flight.
#[derive(Clone)]
pub struct EmissionConfig {
    pub sampler: Sampler,
    pub capture_content: bool,
    pub content_limit_bytes: usize,
}

/// The serving (successful) provider attempt, from log-context fields.
pub struct ServingAttempt {
    pub provider: String,
    pub model: String,
}

/// Plain-data completion inputs assembled at the seam from the values
/// already in scope (`RequestLogContext` fields + attempt ledger).
pub struct TelemetryEmitArgs {
    pub request_id: String,
    pub inbound_context: Option<W3cContext>,
    pub method: &'static str,
    pub route: &'static str,
    pub status_code: u16,
    pub model_group: String,
    pub virtual_key_id: Option<String>,
    pub usage: Option<UsageCtx>,
    pub cost: Option<f64>,
    pub duration_ms: u64,
    pub serving: Option<ServingAttempt>,
    pub failed_attempts: Vec<ProviderAttempt>,
    /// Terminal error class for abandoned requests (499/504); used as
    /// the serving attempt's failure class when the request died
    /// mid-attempt.
    pub terminal_error_class: Option<String>,
    /// Responded model from the actual response object (Req 3.1);
    /// `None` means no response or unknown model.
    pub response_model: Option<String>,
    /// Request-level prompt/completion content (Req 3.6). Built by the
    /// caller ONLY when the snapshot's `capture_content` is on; `None`
    /// keeps the default-off path zero-content (Req 3.5). Never
    /// contains authorization material — request body text and
    /// completion text only.
    pub content: Option<ContentCtx>,
    /// Per-request emission settings frozen at entry (Req 9.3). When
    /// present, sampling and content bounding follow this config, not
    /// the live snapshot; `None` resolves from the snapshot at emit
    /// time (tests, guard-free paths).
    pub emission: Option<EmissionConfig>,
}

/// Sample-and-emit gate (Req 2.6/5, D12): single enabled-check;
/// derive the trace id; apply the sampler (inbound sampled flag wins,
/// else deterministic ratio hash); only when sampled build the spans
/// and enqueue them. Returns the number of spans offered to the
/// exporter (0 when disabled or unsampled) — enqueue rejections are
/// already counted inside the exporter and are silent here.
pub fn emit_request_spans(
    exporter: &TelemetryExporter,
    snapshot: &TelemetrySnapshot,
    args: &TelemetryEmitArgs,
) -> usize {
    // Disabled fast path (Req 9.3/NFR perf): one atomic load, no
    // locks, no clones, no allocation. The sampler decision is
    // frozen here; this helper remains the stateless completion seam
    // for callers without a guard-carried config (tests, abandon
    // paths without entry config).
    let emission = match &args.emission {
        Some(config) => {
            if !snapshot.enabled() {
                return 0;
            }
            config.clone()
        }
        None => match snapshot.emission_config() {
            Some(config) => config,
            None => return 0,
        },
    };
    let (spans, sampled) = build_sampled_spans_with(&emission, args);
    if !sampled {
        return 0;
    }
    let count = spans.len();
    exporter.try_enqueue(spans);
    count
}

/// Sampling helper for tests: resolve the emission config from the
/// snapshot at call time, then build the sampled spans (mirrors the
/// pre-Req-9.3 behavior; production paths carry the entry-time
/// config on the request guard).
#[cfg(test)]
fn build_sampled_spans(snapshot: &TelemetrySnapshot, args: &TelemetryEmitArgs) -> (Vec<SpanData>, bool) {
    if !snapshot.enabled() {
        return (Vec::new(), false);
    }
    let emission = match snapshot.emission_config() {
        Some(config) => config,
        None => return (Vec::new(), false),
    };
    build_sampled_spans_with(&emission, args)
}

/// Sampled-span construction from the per-request [`EmissionConfig`]
/// (Req 9.3): the sampler decision and content gate come from the
/// config captured at request entry, never the live snapshot. Returns
/// `(spans, sampled)`; `sampled == false` means unsampled (spans empty).
fn build_sampled_spans_with(
    emission: &EmissionConfig,
    args: &TelemetryEmitArgs,
) -> (Vec<SpanData>, bool) {
    let (trace_id, inbound_sampled) = resolve_trace_identity(args);
    if !emission.sampler.sample(&trace_id, inbound_sampled) {
        return (Vec::new(), false);
    }
    let start_time = SystemTime::now()
        .checked_sub(Duration::from_millis(args.duration_ms))
        .unwrap_or(SystemTime::UNIX_EPOCH);
    let operation = operation_for(args.route);
    let mut attempts: Vec<AttemptCtx> = Vec::with_capacity(args.failed_attempts.len() + 1);
    for (index, attempt) in args.failed_attempts.iter().enumerate() {
        attempts.push(AttemptCtx {
            index,
            provider: attempt.provider.clone(),
            provider_model: attempt.model.clone(),
            operation: operation.clone(),
            outcome: AttemptOutcome::Failure,
            error_class: Some(
                attempt
                    .error_class
                    .clone()
                    .unwrap_or_else(|| "other".to_string()),
            ),
            duration_ms: attempt.duration_ms,
            usage: None,
            response_model: None,
            start_time,
        });
    }
        if let Some(serving) = &args.serving {
            let outcome = if args.status_code >= 400 {
                AttemptOutcome::Failure
            } else {
                AttemptOutcome::Success
            };
            attempts.push(AttemptCtx {
                index: attempts.len(),
                provider: serving.provider.clone(),
                provider_model: serving.model.clone(),
                operation: operation.clone(),
                outcome,
                error_class: match outcome {
                    AttemptOutcome::Failure => Some(
                        args.terminal_error_class
                            .clone()
                            .unwrap_or_else(|| "other".to_string()),
                    ),
                    AttemptOutcome::Success => None,
                },
                duration_ms: Some(args.duration_ms),
                usage: args.usage,
                // Req 3.1: the responded model from the actual
                // response object; spans.rs emits it only when it
                // differs from the request model.
                response_model: args.response_model.clone(),
                start_time,
            });
        }
    let ctx = RequestCompletionCtx {
        request_id: args.request_id.clone(),
        inbound_context: args.inbound_context.clone(),
        http_method: args.method.to_string(),
        route: args.route.to_string(),
        response_status: args.status_code,
        model_group: args.model_group.clone(),
        virtual_key_id: args.virtual_key_id.clone(),
        usage: args.usage,
        cost: args.cost,
        currency: None,
        capture_content: emission.capture_content,
        // Content capture (Req 3.5/3.6): the caller only builds
        // this when capture is on; the span builder ignores None.
        content: args.content.clone(),
        content_limit_bytes: emission.content_limit_bytes,
        // Root-only abandonment (Req 2.4): the terminal class lands on
        // the root span when no attempt span was built for the
        // request (no serving attempt, no failed attempts).
        root_error_class: if attempts.is_empty() {
            args.terminal_error_class.clone()
        } else {
            None
        },
        start_time,
        duration_ms: args.duration_ms,
        attempts,
    };
        let spans = build_request_spans(&ctx);
        (spans, true)
    }

/// D12 trace-id derivation + D6 sampling inputs: inbound traceparent
/// (trace id + sampled flag), else the gateway request id when it is
/// a UUID (its 16 bytes), else a fresh 128-bit id.
fn resolve_trace_identity(args: &TelemetryEmitArgs) -> (TraceId, Option<bool>) {
    if let Some(inbound) = &args.inbound_context {
        return (inbound.trace_id, Some(inbound.sampled()));
    }
    match uuid::Uuid::parse_str(&args.request_id) {
        Ok(uuid) => (*uuid.as_bytes(), None),
        Err(_) => (new_trace_id(), None),
    }
}

fn operation_for(route: &str) -> String {
    if route == "/v1/responses" {
        "responses".to_string()
    } else {
        "chat".to_string()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::metrics::telemetry::TelemetryMetrics;
    use crate::telemetry::otlp::AttrValue;
    use std::sync::Arc;

    fn exporter() -> TelemetryExporter {
        TelemetryExporter::new(Arc::new(TelemetryMetrics::new()))
    }

    fn snapshot(enabled: bool, ratio: f64) -> TelemetrySnapshot {
        let snapshot = TelemetrySnapshot::new();
        let telemetry = TelemetryConfig {
            enabled,
            sampling_ratio: ratio,
            endpoint: Some("http://127.0.0.1:4318/v1/traces".to_string()),
            ..TelemetryConfig::default()
        };
        snapshot.apply(&telemetry, &LoggingConfig::default());
        snapshot
    }

fn args(request_id: &str, inbound: Option<W3cContext>) -> TelemetryEmitArgs {
    TelemetryEmitArgs {
        request_id: request_id.to_string(),
        inbound_context: inbound,
        method: "POST",
        route: "/v1/chat/completions",
        status_code: 200,
        model_group: "gpt-4o".to_string(),
        virtual_key_id: None,
        usage: Some(UsageCtx {
            input_tokens: 10,
            output_tokens: 20,
        }),
        cost: Some(0.001),
        duration_ms: 42,
        serving: Some(ServingAttempt {
            provider: "openai".to_string(),
            model: "gpt-4o-2024-08-06".to_string(),
        }),
        failed_attempts: Vec::new(),
        terminal_error_class: None,
        response_model: None,
        content: None,
        emission: None,
    }
}

    fn failed_attempt(provider: &str, class: &str) -> ProviderAttempt {
        ProviderAttempt {
            provider: provider.to_string(),
            model: "gpt-4o".to_string(),
            error: "upstream 429".to_string(),
            status_code: Some(429),
            timestamp: chrono::Utc::now(),
            duration_ms: Some(120),
            error_class: Some(class.to_string()),
        }
    }

    #[test]
    fn disabled_snapshot_emits_nothing_and_never_touches_the_exporter() {
        let exporter = exporter();
        let snapshot = snapshot(false, 1.0);
        assert_eq!(emit_request_spans(&exporter, &snapshot, &args("id", None)), 0);
        let metrics = exporter.metrics();
        assert_eq!(metrics.exported(), 0);
        assert_eq!(metrics.dropped(), 0);
    }

    #[test]
    fn ratio_zero_without_inbound_flag_builds_no_spans() {
        let snapshot = snapshot(true, 0.0);
        let exporter = exporter();
        let request_id = "123e4567-e89b-42d3-a456-426614174000";
        assert_eq!(
            emit_request_spans(&exporter, &snapshot, &args(request_id, None)),
            0
        );
    }

    #[test]
    fn inbound_sampled_flag_wins_over_ratio_zero() {
        let snapshot = snapshot(true, 0.0);
        let exporter = exporter();
        let inbound =
            parse_traceparent("00-4bf92f3577b34da6a3ce929d0e0e4736-00f067aa0ba902b7-01")
                .expect("valid traceparent");
        assert_eq!(
            emit_request_spans(&exporter, &snapshot, &args("req", Some(inbound.clone()))),
            2,
            "root + serving attempt"
        );
    }

    #[test]
    fn inbound_unsampled_flag_skips_even_at_ratio_one() {
        let snapshot = snapshot(true, 1.0);
        let exporter = exporter();
        let inbound =
            parse_traceparent("00-4bf92f3577b34da6a3ce929d0e0e4736-00f067aa0ba902b7-00")
                .expect("valid traceparent");
        assert_eq!(
            emit_request_spans(&exporter, &snapshot, &args("req", Some(inbound))),
            0
        );
    }

    #[test]
    fn uuid_request_id_drives_the_sampling_identity() {
        let request_id = "123e4567-e89b-42d3-a456-426614174000";
        let (trace_id, inbound_sampled) = resolve_trace_identity(&args(request_id, None));
        assert_eq!(
            trace_id,
            *uuid::Uuid::parse_str(request_id).unwrap().as_bytes()
        );
        assert_eq!(inbound_sampled, None);
    }

    #[test]
    fn non_uuid_request_id_gets_fresh_identity_without_inbound_flag() {
        let (_, inbound_sampled) = resolve_trace_identity(&args("short-id", None));
        assert_eq!(inbound_sampled, None);
    }

    #[test]
    fn failed_attempts_and_serving_attempt_all_built() {
        let snapshot = snapshot(true, 1.0);
        let exporter = exporter();
        let mut a = args("123e4567-e89b-42d3-a456-426614174000", None);
        a.failed_attempts = vec![failed_attempt("openai", "rate_limited")];
        assert_eq!(
            emit_request_spans(&exporter, &snapshot, &a),
            3,
            "root + failed attempt + serving attempt"
        );
    }

    #[test]
    fn enabled_section_with_unusable_endpoint_stays_dormant() {
        let snapshot = TelemetrySnapshot::new();
        let telemetry = TelemetryConfig {
            enabled: true,
            endpoint: None,
            ..TelemetryConfig::default()
        };
        snapshot.apply(&telemetry, &LoggingConfig::default());
        assert!(!snapshot.enabled());
    }

    #[test]
    fn responses_route_builds_responses_operation_root_span() {
        let snapshot = snapshot(true, 1.0);
        let mut a = args("123e4567-e89b-42d3-a456-426614174000", None);
        a.route = "/v1/responses";
        let (spans, sampled) = build_sampled_spans(&snapshot, &a);
        assert!(sampled);
        assert_eq!(spans.len(), 2, "root + serving attempt");
        assert_eq!(spans[0].name, "POST /v1/responses");
        assert!(spans[1].name.starts_with("responses "));
    }

    fn attr<'a>(span: &'a SpanData, key: &str) -> Option<&'a AttrValue> {
        span.attributes
            .iter()
            .find(|(k, _)| k == key)
            .map(|(_, v)| v)
    }

    #[test]
    fn capture_content_on_feeds_prompt_and_completion_attributes() {
        let snapshot = TelemetrySnapshot::new();
        snapshot.apply(
            &TelemetryConfig {
                enabled: true,
                sampling_ratio: 1.0,
                capture_content: true,
                endpoint: Some("http://127.0.0.1:4318/v1/traces".to_string()),
                ..TelemetryConfig::default()
            },
            &LoggingConfig::default(),
        );
        let mut a = args("123e4567-e89b-42d3-a456-426614174000", None);
        a.content = Some(ContentCtx {
            prompt: "{\"model\":\"gpt-4o\"}".to_string(),
            completion: "Hello!".to_string(),
        });
        let (spans, sampled) = build_sampled_spans(&snapshot, &a);
        assert!(sampled);
        // Content rides the serving attempt (last client span).
        let serving = spans.last().expect("serving attempt span");
        assert_eq!(
            attr(serving, "gen_ai.prompt"),
            Some(&AttrValue::String("{\"model\":\"gpt-4o\"}".to_string()))
        );
        assert_eq!(
            attr(serving, "gen_ai.completion"),
            Some(&AttrValue::String("Hello!".to_string()))
        );
    }

    #[test]
    fn capture_content_off_leaves_content_attributes_absent() {
        let snapshot = snapshot(true, 1.0);
        let mut a = args("123e4567-e89b-42d3-a456-426614174000", None);
        // Snapshot default: capture_content == false even though the
        // caller (incorrectly) supplied content — default-off stays
        // zero-content (Req 3.5).
        a.content = Some(ContentCtx {
            prompt: "p".to_string(),
            completion: "c".to_string(),
        });
        let (spans, sampled) = build_sampled_spans(&snapshot, &a);
        assert!(sampled);
        for span in &spans {
            assert!(attr(span, "gen_ai.prompt").is_none());
            assert!(attr(span, "gen_ai.completion").is_none());
        }
    }

    #[test]
    fn invalid_traceparent_is_ignored() {
        let mut headers = axum::http::HeaderMap::new();
        headers.insert(
            axum::http::HeaderName::from_static("traceparent"),
            axum::http::HeaderValue::from_static("garbage"),
        );
        assert!(extract_inbound_context(&headers, true).is_none());
    }

#[test]
fn traceparent_not_read_when_telemetry_disabled() {
    let mut headers = axum::http::HeaderMap::new();
    headers.insert(
        axum::http::HeaderName::from_static("traceparent"),
        axum::http::HeaderValue::from_static(
            "00-4bf92f3577b34da6a3ce929d0e0e4736-00f067aa0ba902b7-01",
        ),
    );
    assert!(extract_inbound_context(&headers, false).is_none());
}

#[test]
fn root_only_abandonment_carries_terminal_error_type_on_root_span() {
    // Req 2.4: abandonment before any serving/failed attempt exists
    // still surfaces the terminal outcome — the root span carries
    // error.type = client_disconnect (499) / gateway_deadline (504).
    let snapshot = snapshot(true, 1.0);
    let mut a = args("123e4567-e89b-42d3-a456-426614174000", None);
    a.status_code = 499;
    a.serving = None;
    a.failed_attempts = Vec::new();
    a.terminal_error_class = Some("client_disconnect".to_string());
    let (spans, sampled) = build_sampled_spans(&snapshot, &a);
    assert!(sampled);
    assert_eq!(spans.len(), 1, "root only, no attempt spans");
    assert_eq!(
        attr(&spans[0], "error.type"),
        Some(&AttrValue::String("client_disconnect".to_string()))
    );
}

#[test]
fn root_only_gateway_deadline_abandonment_carries_deadline_error_type() {
    let snapshot = snapshot(true, 1.0);
    let mut a = args("123e4567-e89b-42d3-a456-426614174000", None);
    a.status_code = 504;
    a.serving = None;
    a.failed_attempts = Vec::new();
    a.terminal_error_class = Some("gateway_deadline".to_string());
    let (spans, _) = build_sampled_spans(&snapshot, &a);
    assert_eq!(spans.len(), 1);
    assert_eq!(
        attr(&spans[0], "error.type"),
        Some(&AttrValue::String("gateway_deadline".to_string()))
    );
}

#[test]
fn response_model_emitted_only_when_it_differs_from_request_model() {
    // Req 3.1: gen_ai.response.model rides the serving attempt only
    // when the responded model differs from the request model.
    let snapshot = snapshot(true, 1.0);
    let mut a = args("123e4567-e89b-42d3-a456-426614174000", None);
    a.serving = Some(ServingAttempt {
        provider: "openai".to_string(),
        model: "gpt-4o".to_string(),
    });
    a.response_model = Some("gpt-4o".to_string());
    let (spans, _) = build_sampled_spans(&snapshot, &a);
    let serving = spans.last().expect("serving attempt");
    assert!(
        attr(serving, "gen_ai.response.model").is_none(),
        "same model must not emit gen_ai.response.model"
    );

    a.response_model = Some("gpt-4o-2024-08-06".to_string());
    let (spans, _) = build_sampled_spans(&snapshot, &a);
    let serving = spans.last().expect("serving attempt");
    assert_eq!(
        attr(serving, "gen_ai.response.model"),
        Some(&AttrValue::String("gpt-4o-2024-08-06".to_string()))
    );
}

#[test]
fn completion_uses_entry_time_emission_config_when_snapshot_changed_mid_flight() {
    // Req 9.3: emission follows the config snapshot in effect at
    // request START. Entry: ratio 1.0 + capture on. Mid-flight the
    // operator reloads to ratio 0.0 + capture off; the completion
    // seam must still sample and attach content per the entry config.
    let snap = TelemetrySnapshot::new();
    snap.apply(
        &TelemetryConfig {
            enabled: true,
            sampling_ratio: 1.0,
            capture_content: true,
            endpoint: Some("http://127.0.0.1:4318/v1/traces".to_string()),
            ..TelemetryConfig::default()
        },
        &LoggingConfig::default(),
    );
    // Entry-time config captured by the guard.
    let entry = snap.emission_config().expect("entry config");
    // Mid-flight reload: ratio 0 + capture off.
    snap.apply(
        &TelemetryConfig {
            enabled: true,
            sampling_ratio: 0.0,
            capture_content: false,
            endpoint: Some("http://127.0.0.1:4318/v1/traces".to_string()),
            ..TelemetryConfig::default()
        },
        &LoggingConfig::default(),
    );
    let exporter = exporter();
    let mut a = args("123e4567-e89b-42d3-a456-426614174000", None);
    a.content = Some(ContentCtx {
        prompt: "p".to_string(),
        completion: "c".to_string(),
    });
    a.emission = Some(entry);
    // Entry config (ratio 1.0) wins: spans are built and content is
    // attached even though the live sampler now rejects everything.
    assert_eq!(emit_request_spans(&exporter, &snap, &a), 2);

    // Inverse direction: entry ratio 0.0, mid-flight reload to 1.0 —
    // the request stays unsampled per its entry config.
    let snap2 = snapshot(true, 0.0);
    let entry2 = snap2.emission_config().expect("entry config");
    snap2.apply(
        &TelemetryConfig {
            enabled: true,
            sampling_ratio: 1.0,
            endpoint: Some("http://127.0.0.1:4318/v1/traces".to_string()),
            ..TelemetryConfig::default()
        },
        &LoggingConfig::default(),
    );
    let mut b = args("123e4567-e89b-42d3-a456-426614174000", None);
    b.emission = Some(entry2);
    assert_eq!(
        emit_request_spans(&exporter, &snap2, &b),
        0,
        "entry-time ratio 0.0 must keep the request unsampled"
    );
}

#[test]
fn disabled_snapshot_returns_none_from_emission_config_without_locking() {
    // NFR perf: the disabled path resolves to None with a single
    // atomic load — no Arc clones, no lock acquisition.
    let snapshot = snapshot(false, 1.0);
    assert!(snapshot.emission_config().is_none());
}
}
