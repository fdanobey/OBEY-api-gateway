//! Span assembly from request-completion seam data (spec
//! opentelemetry-trace-export, Req 2/3, decision D12). Pure assembly
//! from passed-in values: no I/O, no globals.

use crate::telemetry::otlp::{truncate_attr_value, AttrValue, SpanData, SpanKind, SpanStatus};
use crate::telemetry::propagation::{new_trace_id, W3cContext};
use std::time::{Duration, SystemTime};
use uuid::Uuid;

/// Token pair; used for request totals (root span) and per-attempt usage.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct UsageCtx {
    pub input_tokens: i64,
    pub output_tokens: i64,
}

/// Terminal result of one provider attempt.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum AttemptOutcome {
    Success,
    Failure,
}

/// Request-level prompt/completion bodies. Only read when
/// `capture_content` is set; never contains authorization material.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ContentCtx {
    pub prompt: String,
    pub completion: String,
}

/// One attempt-ledger row (Req 2.5: same fields the `#attempt` log row
/// carries) plus per-attempt usage when the provider reported it.
#[derive(Debug, Clone, PartialEq)]
pub struct AttemptCtx {
    pub index: usize,
    pub provider: String,
    pub provider_model: String,
    /// Inference operation, e.g. "chat" (span name `{operation} {model}`).
    pub operation: String,
    pub outcome: AttemptOutcome,
    /// Content-free error class (`error_class_for`); failure rows always
    /// store one, `None` degrades to "other".
    pub error_class: Option<String>,
    pub duration_ms: Option<u64>,
    pub usage: Option<UsageCtx>,
    /// Responded model; emitted as `gen_ai.response.model` only when it
    /// differs from `provider_model` (Req 3.1).
    pub response_model: Option<String>,
    pub start_time: SystemTime,
}

/// Plain data filled by the caller at the completion seam (task 6
/// constructs it from `RequestLogContext` + the attempt ledger).
#[derive(Debug, Clone, PartialEq)]
pub struct RequestCompletionCtx {
    /// Existing request trace id string (`X-Trace-Id`/`X-Request-Id`).
    pub request_id: String,
    /// Parsed inbound W3C context; its trace id wins (D12, Req 4.1).
    pub inbound_context: Option<W3cContext>,
    pub http_method: String,
    /// Route template, e.g. "/v1/chat/completions".
    pub route: String,
    pub response_status: u16,
    pub model_group: String,
    /// Virtual key ID only — never the key token.
    pub virtual_key_id: Option<String>,
    /// Request-aggregated usage (root span, Req 3.2).
    pub usage: Option<UsageCtx>,
    pub cost: Option<f64>,
    /// ISO 4217; `None` → "USD" (Req 3.3).
    pub currency: Option<String>,
    pub capture_content: bool,
    pub content: Option<ContentCtx>,
    /// Body byte bound for content attributes (`logging.max_body_size_bytes`).
    pub content_limit_bytes: usize,
    /// Terminal error class for root-only abandonment (Req 2.4):
    /// `client_disconnect`/`gateway_deadline` when the request died
    /// before any serving/failed attempt existed; the root span then
    /// carries `error.type` so the terminal outcome is visible in the
    /// trace view. `None` for normally completed requests.
    pub root_error_class: Option<String>,
    pub start_time: SystemTime,
    pub duration_ms: u64,
    pub attempts: Vec<AttemptCtx>,
}

/// Build the root gateway span (SERVER) plus one CLIENT span per attempt,
/// all sharing one trace id (Req 2.1-2.3). Trace-id derivation per D12:
/// inbound W3C trace id, else the request id when it is a UUID, else a
/// fresh 128-bit id (the original id always rides along as
/// `obey.request_id`).
pub fn build_request_spans(ctx: &RequestCompletionCtx) -> Vec<SpanData> {
    let trace_id = derive_trace_id(ctx);
    let root_span_id = new_span_id();

    let mut attrs = Vec::new();
    push_str(&mut attrs, "http.request.method", &ctx.http_method);
    push_str(&mut attrs, "http.route", &ctx.route);
    push_int(&mut attrs, "http.response.status_code", i64::from(ctx.response_status));
    push_str(&mut attrs, "obey.request_id", &ctx.request_id);
    push_str(&mut attrs, "obey.model.group", &ctx.model_group);
    if let Some(virtual_key_id) = &ctx.virtual_key_id {
        push_str(&mut attrs, "obey.virtual_key_id", virtual_key_id);
    }
    if let Some(usage) = &ctx.usage {
        push_int(&mut attrs, "gen_ai.usage.input_tokens", usage.input_tokens);
        push_int(&mut attrs, "gen_ai.usage.output_tokens", usage.output_tokens);
    }
    if let Some(cost) = ctx.cost {
        attrs.push(("gen_ai.usage.cost".to_string(), AttrValue::Double(cost)));
        push_str(
            &mut attrs,
            "gen_ai.usage.currency",
            ctx.currency.as_deref().unwrap_or("USD"),
        );
    }
    // Root-only abandonment (Req 2.4): when the request died before
    // any attempt span exists, the terminal outcome rides the root
    // span as `error.type` (bounded class) so the trace view shows
    // client_disconnect/gateway_deadline without a serving child.
    if let Some(class) = &ctx.root_error_class {
        attrs.push((
            "error.type".to_string(),
            AttrValue::String(truncate_attr_value(class, ctx.content_limit_bytes)),
        ));
    }

    let root = SpanData {
        trace_id,
        span_id: root_span_id,
        parent_span_id: ctx
            .inbound_context
            .as_ref()
            .map(|context| context.parent_span_id),
        name: format!("{} {}", ctx.http_method, ctx.route),
        kind: SpanKind::Server,
        status: Some(if ctx.response_status >= 400 {
            SpanStatus::Error
        } else {
            SpanStatus::Ok
        }),
        start_time: ctx.start_time,
        end_time: ctx.start_time + Duration::from_millis(ctx.duration_ms),
        attributes: attrs,
    };

    let mut spans = Vec::with_capacity(ctx.attempts.len() + 1);
    spans.push(root);
    for attempt in &ctx.attempts {
        spans.push(build_attempt_span(ctx, trace_id, root_span_id, attempt));
    }

    // Content attributes ride on the (last) serving attempt — the
    // inference call the bodies belong to — and fall back to the root
    // when the request never got an attempt span.
    if ctx.capture_content {
        if let Some(content) = &ctx.content {
            let target = if spans.len() > 1 { spans.len() - 1 } else { 0 };
            let limit = ctx.content_limit_bytes;
            let attrs = &mut spans[target].attributes;
            attrs.push((
                "gen_ai.prompt".to_string(),
                AttrValue::String(truncate_attr_value(&content.prompt, limit)),
            ));
            attrs.push((
                "gen_ai.completion".to_string(),
                AttrValue::String(truncate_attr_value(&content.completion, limit)),
            ));
        }
    }

    spans
}

fn build_attempt_span(
    ctx: &RequestCompletionCtx,
    trace_id: [u8; 16],
    parent_span_id: [u8; 8],
    attempt: &AttemptCtx,
) -> SpanData {
    let mut attrs = Vec::new();
    push_str(&mut attrs, "gen_ai.operation.name", &attempt.operation);
    push_str(&mut attrs, "gen_ai.provider.name", &attempt.provider);
    push_str(&mut attrs, "gen_ai.system", &provider_system(&attempt.provider));
    push_str(&mut attrs, "gen_ai.request.model", &attempt.provider_model);
    if let Some(response_model) = &attempt.response_model {
        if response_model != &attempt.provider_model {
            push_str(&mut attrs, "gen_ai.response.model", response_model);
        }
    }
    push_int(&mut attrs, "obey.attempt.index", attempt.index as i64);
    if let Some(usage) = &attempt.usage {
        push_int(&mut attrs, "gen_ai.usage.input_tokens", usage.input_tokens);
        push_int(&mut attrs, "gen_ai.usage.output_tokens", usage.output_tokens);
    }

    let status = match attempt.outcome {
        AttemptOutcome::Success => SpanStatus::Ok,
        AttemptOutcome::Failure => {
            let class = attempt.error_class.as_deref().unwrap_or("other");
            attrs.push((
                "error.type".to_string(),
                AttrValue::String(truncate_attr_value(class, ctx.content_limit_bytes)),
            ));
            SpanStatus::Error
        }
    };

    SpanData {
        trace_id,
        span_id: new_span_id(),
        parent_span_id: Some(parent_span_id),
        name: format!("{} {}", attempt.operation, attempt.provider_model),
        kind: SpanKind::Client,
        status: Some(status),
        start_time: attempt.start_time,
        end_time: attempt.start_time + Duration::from_millis(attempt.duration_ms.unwrap_or(0)),
        attributes: attrs,
    }
}

fn derive_trace_id(ctx: &RequestCompletionCtx) -> [u8; 16] {
    if let Some(inbound) = &ctx.inbound_context {
        return inbound.trace_id;
    }
    if let Some(trace_id) = parse_uuid_bytes(&ctx.request_id) {
        return trace_id;
    }
    new_trace_id()
}

fn parse_uuid_bytes(value: &str) -> Option<[u8; 16]> {
    let uuid = Uuid::parse_str(value).ok()?;
    Some(*uuid.as_bytes())
}

fn new_span_id() -> [u8; 8] {
    let bytes = *Uuid::new_v4().as_bytes();
    let mut span_id = [0u8; 8];
    span_id.copy_from_slice(&bytes[..8]);
    span_id
}

/// Trivial `gen_ai.system` mapping: provider name lowercased
/// (e.g. "OpenAI" -> "openai"); no per-provider table.
fn provider_system(provider: &str) -> String {
    provider.to_lowercase()
}

fn push_str(attrs: &mut Vec<(String, AttrValue)>, key: &str, value: &str) {
    attrs.push((key.to_string(), AttrValue::String(value.to_string())));
}

fn push_int(attrs: &mut Vec<(String, AttrValue)>, key: &str, value: i64) {
    attrs.push((key.to_string(), AttrValue::Int(value)));
}

#[cfg(test)]
mod tests {
    use super::*;

    const UUID_REQUEST_ID: &str = "123e4567-e89b-42d3-a456-426614174000";

    fn base_ctx() -> RequestCompletionCtx {
        RequestCompletionCtx {
            request_id: UUID_REQUEST_ID.to_string(),
            inbound_context: None,
            http_method: "POST".to_string(),
            route: "/v1/chat/completions".to_string(),
            response_status: 200,
            model_group: "default".to_string(),
            virtual_key_id: Some("vk-9".to_string()),
            usage: Some(UsageCtx {
                input_tokens: 100,
                output_tokens: 40,
            }),
            cost: Some(0.0123),
            currency: None,
        capture_content: false,
        content: None,
        content_limit_bytes: 10_000,
        root_error_class: None,
            start_time: SystemTime::UNIX_EPOCH,
            duration_ms: 1_500,
            attempts: vec![
                AttemptCtx {
                    index: 0,
                    provider: "OpenAI".to_string(),
                    provider_model: "gpt-4o".to_string(),
                    operation: "chat".to_string(),
                    outcome: AttemptOutcome::Failure,
                    error_class: Some("ttfb_timeout".to_string()),
                    duration_ms: Some(500),
                    usage: None,
                    response_model: None,
                    start_time: SystemTime::UNIX_EPOCH,
                },
                AttemptCtx {
                    index: 1,
                    provider: "Anthropic".to_string(),
                    provider_model: "claude-3-5".to_string(),
                    operation: "chat".to_string(),
                    outcome: AttemptOutcome::Success,
                    error_class: None,
                    duration_ms: Some(1_000),
                    usage: Some(UsageCtx {
                        input_tokens: 100,
                        output_tokens: 40,
                    }),
                    response_model: Some("claude-3-5-20241022".to_string()),
                    start_time: SystemTime::UNIX_EPOCH + Duration::from_millis(500),
                },
            ],
        }
    }

    fn attr<'a>(span: &'a SpanData, key: &str) -> Option<&'a AttrValue> {
        span.attributes
            .iter()
            .find(|(k, _)| k == key)
            .map(|(_, v)| v)
    }

    #[test]
    fn builds_root_server_and_client_attempt_spans_with_parent_linkage() {
        let spans = build_request_spans(&base_ctx());
        assert_eq!(spans.len(), 3);
        let (root, attempts) = spans.split_first().unwrap();
        assert_eq!(root.kind, SpanKind::Server);
        assert_eq!(root.name, "POST /v1/chat/completions");
        assert_eq!(root.parent_span_id, None);
        for attempt in attempts {
            assert_eq!(attempt.kind, SpanKind::Client);
            assert_eq!(attempt.trace_id, root.trace_id);
            assert_eq!(attempt.parent_span_id, Some(root.span_id));
        }
        assert_eq!(attempts[0].name, "chat gpt-4o");
        assert_eq!(attempts[1].name, "chat claude-3-5");
    }

    #[test]
    fn root_carries_http_obey_usage_and_cost_attributes() {
        let spans = build_request_spans(&base_ctx());
        let root = &spans[0];
        assert_eq!(
            attr(root, "http.request.method"),
            Some(&AttrValue::String("POST".to_string()))
        );
        assert_eq!(
            attr(root, "http.route"),
            Some(&AttrValue::String("/v1/chat/completions".to_string()))
        );
        assert_eq!(
            attr(root, "http.response.status_code"),
            Some(&AttrValue::Int(200))
        );
        assert_eq!(
            attr(root, "obey.request_id"),
            Some(&AttrValue::String(UUID_REQUEST_ID.to_string()))
        );
        assert_eq!(
            attr(root, "obey.model.group"),
            Some(&AttrValue::String("default".to_string()))
        );
        assert_eq!(
            attr(root, "obey.virtual_key_id"),
            Some(&AttrValue::String("vk-9".to_string()))
        );
        assert_eq!(
            attr(root, "gen_ai.usage.input_tokens"),
            Some(&AttrValue::Int(100))
        );
        assert_eq!(
            attr(root, "gen_ai.usage.output_tokens"),
            Some(&AttrValue::Int(40))
        );
        assert_eq!(
            attr(root, "gen_ai.usage.cost"),
            Some(&AttrValue::Double(0.0123))
        );
        assert_eq!(
            attr(root, "gen_ai.usage.currency"),
            Some(&AttrValue::String("USD".to_string()))
        );
    }

    #[test]
    fn optional_root_attributes_absent_when_unavailable() {
        let mut ctx = base_ctx();
        ctx.virtual_key_id = None;
        ctx.usage = None;
        ctx.cost = None;
        let root = &build_request_spans(&ctx)[0];
        assert!(attr(root, "obey.virtual_key_id").is_none());
        assert!(attr(root, "gen_ai.usage.input_tokens").is_none());
        assert!(attr(root, "gen_ai.usage.output_tokens").is_none());
        assert!(attr(root, "gen_ai.usage.cost").is_none());
        assert!(attr(root, "gen_ai.usage.currency").is_none());
    }

    #[test]
    fn custom_currency_is_respected() {
        let mut ctx = base_ctx();
        ctx.currency = Some("EUR".to_string());
        let root = &build_request_spans(&ctx)[0];
        assert_eq!(
            attr(root, "gen_ai.usage.currency"),
            Some(&AttrValue::String("EUR".to_string()))
        );
    }

    #[test]
    fn attempt_carries_semconv_and_usage_attributes() {
        let spans = build_request_spans(&base_ctx());
        let success = &spans[2];
        assert_eq!(
            attr(success, "gen_ai.operation.name"),
            Some(&AttrValue::String("chat".to_string()))
        );
        assert_eq!(
            attr(success, "gen_ai.provider.name"),
            Some(&AttrValue::String("Anthropic".to_string()))
        );
        assert_eq!(
            attr(success, "gen_ai.system"),
            Some(&AttrValue::String("anthropic".to_string()))
        );
        assert_eq!(
            attr(success, "gen_ai.request.model"),
            Some(&AttrValue::String("claude-3-5".to_string()))
        );
        assert_eq!(attr(success, "obey.attempt.index"), Some(&AttrValue::Int(1)));
        assert_eq!(
            attr(success, "gen_ai.usage.input_tokens"),
            Some(&AttrValue::Int(100))
        );
        assert_eq!(
            attr(success, "gen_ai.usage.output_tokens"),
            Some(&AttrValue::Int(40))
        );
    }

    #[test]
    fn response_model_emitted_only_when_it_differs() {
        let spans = build_request_spans(&base_ctx());
        assert!(attr(&spans[1], "gen_ai.response.model").is_none());
        assert_eq!(
            attr(&spans[2], "gen_ai.response.model"),
            Some(&AttrValue::String("claude-3-5-20241022".to_string()))
        );

        let mut ctx = base_ctx();
        ctx.attempts[1].response_model = Some("claude-3-5".to_string());
        let spans = build_request_spans(&ctx);
        assert!(attr(&spans[2], "gen_ai.response.model").is_none());
    }

    #[test]
    fn failed_attempt_carries_error_type_and_error_status() {
        let spans = build_request_spans(&base_ctx());
        assert_eq!(spans[1].status, Some(SpanStatus::Error));
        assert_eq!(
            attr(&spans[1], "error.type"),
            Some(&AttrValue::String("ttfb_timeout".to_string()))
        );
        assert_eq!(spans[2].status, Some(SpanStatus::Ok));
        assert!(attr(&spans[2], "error.type").is_none());
    }

    #[test]
    fn missing_error_class_degrades_to_other() {
        let mut ctx = base_ctx();
        ctx.attempts[0].error_class = None;
        let spans = build_request_spans(&ctx);
        assert_eq!(
            attr(&spans[1], "error.type"),
            Some(&AttrValue::String("other".to_string()))
        );
    }

    #[test]
    fn content_absent_by_default() {
        let mut ctx = base_ctx();
        ctx.content = Some(ContentCtx {
            prompt: "hello".to_string(),
            completion: "world".to_string(),
        });
        for span in build_request_spans(&ctx) {
            assert!(attr(&span, "gen_ai.prompt").is_none());
            assert!(attr(&span, "gen_ai.completion").is_none());
        }
    }

    #[test]
    fn content_truncated_to_limit_on_last_attempt() {
        let mut ctx = base_ctx();
        ctx.capture_content = true;
        ctx.content_limit_bytes = 8;
        ctx.content = Some(ContentCtx {
            prompt: "a".repeat(20),
            completion: "b".repeat(20),
        });
        let spans = build_request_spans(&ctx);
        let serving = spans.last().unwrap();
        assert_eq!(
            attr(serving, "gen_ai.prompt"),
            Some(&AttrValue::String(
                "aaaaaaaa... [truncated by 12 bytes]".to_string()
            ))
        );
        assert_eq!(
            attr(serving, "gen_ai.completion"),
            Some(&AttrValue::String(
                "bbbbbbbb... [truncated by 12 bytes]".to_string()
            ))
        );
        assert!(attr(&spans[0], "gen_ai.prompt").is_none());
        assert!(attr(&spans[1], "gen_ai.prompt").is_none());
    }

    #[test]
    fn content_falls_back_to_root_without_attempts() {
        let mut ctx = base_ctx();
        ctx.attempts.clear();
        ctx.capture_content = true;
        ctx.content = Some(ContentCtx {
            prompt: "hello".to_string(),
            completion: "world".to_string(),
        });
        let root = &build_request_spans(&ctx)[0];
        assert_eq!(
            attr(root, "gen_ai.prompt"),
            Some(&AttrValue::String("hello".to_string()))
        );
        assert_eq!(
            attr(root, "gen_ai.completion"),
            Some(&AttrValue::String("world".to_string()))
        );
    }

    #[test]
    fn trace_id_from_inbound_w3c_context() {
        let mut ctx = base_ctx();
        ctx.request_id = "not-a-uuid".to_string();
        ctx.inbound_context = Some(W3cContext {
            version: 0,
            trace_id: [0x4b; 16],
            parent_span_id: [0x0f; 8],
            trace_flags: 1,
        });
        let spans = build_request_spans(&ctx);
        let root = &spans[0];
        assert_eq!(root.trace_id, [0x4b; 16]);
        assert_eq!(root.parent_span_id, Some([0x0f; 8]));
        for span in &spans[1..] {
            assert_eq!(span.trace_id, [0x4b; 16]);
        }
    }

    #[test]
    fn trace_id_reuses_uuid_request_id() {
        let expected = *Uuid::parse_str(UUID_REQUEST_ID)
            .unwrap()
            .as_bytes();
        let ctx = base_ctx();
        let root = &build_request_spans(&ctx)[0];
        assert_eq!(root.trace_id, expected);
        assert_eq!(root.parent_span_id, None);
    }

    #[test]
    fn non_uuid_request_id_gets_fresh_trace_id_and_request_id_attr() {
        let mut ctx = base_ctx();
        ctx.request_id = "req-abc-123".to_string();
        let spans = build_request_spans(&ctx);
        let root = &spans[0];
        assert_ne!(root.trace_id, [0u8; 16]);
        assert_eq!(root.parent_span_id, None);
        assert_eq!(
            attr(root, "obey.request_id"),
            Some(&AttrValue::String("req-abc-123".to_string()))
        );
        for span in &spans[1..] {
            assert_eq!(span.trace_id, root.trace_id);
        }

        let fresh = &build_request_spans(&ctx)[0];
        assert_ne!(fresh.trace_id, root.trace_id);
    }

#[test]
fn end_minus_start_equals_duration_ms() {
    let ctx = base_ctx();
    let spans = build_request_spans(&ctx);
    let root = &spans[0];
    assert_eq!(
        root.end_time
            .duration_since(root.start_time)
            .unwrap()
            .as_millis(),
        1_500
    );
    for (span, expected) in spans[1..].iter().zip([500u64, 1_000]) {
        assert_eq!(
            span.end_time
                .duration_since(span.start_time)
                .unwrap()
                .as_millis(),
            u128::from(expected)
        );
    }
}

#[test]
fn root_only_abandonment_carries_terminal_error_type() {
    // Req 2.4: abandonment before any serving/failed attempt exists —
    // the root span alone carries the bounded terminal class.
    let mut ctx = base_ctx();
    ctx.attempts.clear();
    ctx.response_status = 499;
    ctx.root_error_class = Some("client_disconnect".to_string());
    let spans = build_request_spans(&ctx);
    assert_eq!(spans.len(), 1, "root only");
    assert_eq!(
        attr(&spans[0], "error.type"),
        Some(&AttrValue::String("client_disconnect".to_string()))
    );
    assert_eq!(spans[0].status, Some(SpanStatus::Error));
}

#[test]
fn root_error_class_absent_for_normal_completion() {
    let spans = build_request_spans(&base_ctx());
    assert!(attr(&spans[0], "error.type").is_none());
}

#[test]
fn content_truncation_marker_names_dropped_bytes() {
    // Req 3.6: truncation is explicit — the marker must appear in the
    // emitted attribute when truncation occurred, and must not be
    // double-applied when the value already fits.
    let mut ctx = base_ctx();
    ctx.capture_content = true;
    ctx.content_limit_bytes = 16;
    ctx.content = Some(ContentCtx {
        prompt: "a".repeat(64),
        completion: "short".to_string(),
    });
    let spans = build_request_spans(&ctx);
    let serving = spans.last().unwrap();
    let prompt = attr(serving, "gen_ai.prompt").expect("prompt attr");
    let AttrValue::String(value) = prompt else {
        panic!("string attr");
    };
    assert!(
        value.contains("[truncated by "),
        "truncation marker must be present: {value}"
    );
    assert!(value.starts_with("aaaaaaaaaaaaaaaa"));
    // Untruncated values pass through without any marker.
    let completion = attr(serving, "gen_ai.completion").expect("completion attr");
    assert_eq!(completion, &AttrValue::String("short".to_string()));
}
}
