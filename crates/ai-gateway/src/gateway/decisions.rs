//! System One decisions front door (`POST /v1/decisions`, alias
//! `/v1/systemone`) — typed decision-model requests through the gateway
//! (decisions-api spec, Req 1.1-1.6, 4.1-4.4).
//!
//! `state`/`questions`/`answers` payloads never enter logs or telemetry:
//! request-log rows and OTLP spans are content-free by construction and
//! the telemetry `content` context is always `None`.

use std::collections::HashMap;
use std::time::Instant;

use axum::{
    body::Bytes,
    extract::State,
    http::{header, HeaderMap, StatusCode},
    response::{IntoResponse, Response},
    Extension, Json,
};
use serde_json::Value;

use crate::error::{error_class_for, GatewayError, ProviderAttempt};
use crate::logger::LogEntry;
use crate::router::trace_id::generate_trace_id;
use crate::smart_routing::systemone::models::{Question, SystemOneRequest, SystemOneResponse};
use crate::telemetry::spans::UsageCtx;
use crate::virtual_keys::models::AuthenticatedKey;

use super::telemetry_seam::{self, ServingAttempt, TelemetryEmitArgs};
use super::AppState;

/// Log-row path and telemetry route for both the canonical route and the
/// `/v1/systemone` alias (Req 4.1: one path, `/v1/decisions`).
const DECISIONS_PATH: &str = "/v1/decisions";

/// Per-attempt log-row path for failed decision tries — the chat path's
/// `#attempt` convention (handlers.rs `ATTEMPT_LOG_PATH`), under the
/// decisions canonical path so rows correlate with the outcome row's
/// shared `trace_id`; the `#attempt` suffix keeps them excluded from
/// cost reports (`path NOT LIKE '%#attempt'`) and included in the
/// dashboard Errors tab.
const DECISION_ATTEMPT_LOG_PATH: &str = "/v1/decisions#attempt";

/// Status code used for an attempt whose own `status_code` is `None`
/// (e.g. a connection error or admission skip with no HTTP status);
/// mirrors handlers.rs `ATTEMPT_FALLBACK_STATUS` and the
/// `AllProvidersFailed` → 502 mapping.
const ATTEMPT_FALLBACK_STATUS: u16 = 502;

pub async fn create_decision(
    State(state): State<AppState>,
    headers: HeaderMap,
    authenticated_key: Option<Extension<AuthenticatedKey>>,
    body: Bytes,
) -> Response {
    let started = Instant::now();
    let trace_id = decision_trace_id(&headers);
    let virtual_key_id = authenticated_key.map(|Extension(key)| key.id);

    let is_json = headers
        .get(header::CONTENT_TYPE)
        .and_then(|value| value.to_str().ok())
        .is_some_and(|value| value.to_ascii_lowercase().starts_with("application/json"));
    if !is_json {
        return decision_rejected(
            &state,
            &trace_id,
            None,
            started.elapsed().as_millis() as u64,
            virtual_key_id.as_deref(),
            "Content-Type must be application/json",
        );
    }

    let value: Value = match serde_json::from_slice(&body) {
        Ok(value) => value,
        Err(error) => {
            return decision_rejected(
                &state,
                &trace_id,
                None,
                started.elapsed().as_millis() as u64,
                virtual_key_id.as_deref(),
                &format!("invalid JSON body: {error}"),
            );
        }
    };
let model_hint = value
    .get("model")
    .and_then(Value::as_str)
    .map(|model| sanitize_bounded(model, 128));
    let request = match parse_decision_request(value) {
        Ok(request) => request,
        Err(message) => {
            return decision_rejected(
                &state,
                &trace_id,
                model_hint,
                started.elapsed().as_millis() as u64,
                virtual_key_id.as_deref(),
                &message,
            );
        }
    };

    let result = state.router.route_decision(&request).await;
    let duration_ms = started.elapsed().as_millis() as u64;
    // H3: bound the request-controlled model string at every
    // reflection site this handler owns (tracing fields, log-row
    // model columns, telemetry model_group); the success path's
    // response body echoes the provider's model verbatim (Req 1.5).
    let logged_model = sanitize_bounded(&request.model, 128);
// Req 2.7: failed tries are observable exactly like the chat path —
// per-attempt `#attempt` log rows (outcome-row fields, shared trace
// id) plus the failed-attempt spans inside the root OTLP span. The
// telemetry snapshot decides span emission; the log rows always go
// through when tries exist.
let attempts = result_attempts(&result);
let telemetry_attempts = state
    .telemetry_snapshot
    .enabled()
    .then(|| attempts.to_vec())
    .unwrap_or_default();
log_decision_attempts(&state, &trace_id, &attempts);
emit_decision_telemetry(
    &state,
    &trace_id,
    &logged_model,
    duration_ms,
    virtual_key_id.as_deref(),
    &result,
    telemetry_attempts,
);

match result {
    // Ok carries (response, provider, prior failed attempts); the
    // attempts were already surfaced above via the `#attempt` rows
    // and telemetry spans.
    Ok((response, provider_name, _attempts)) => {
        tracing::info!(
            trace_id = %trace_id,
            model = %logged_model,
            resolved_model = %sanitize_bounded(&response.model, 128),
            provider = %provider_name,
            status = 200u16,
            duration_ms,
            "Decision request completed"
        );
        write_decision_log(
            &state,
            &trace_id,
            DecisionLogRow {
                requested_model: Some(logged_model.clone()),
                responded_model: Some(sanitize_bounded(&response.model, 128)),
                    provider: provider_name,
                    status_code: StatusCode::OK.as_u16(),
                    duration_ms,
                    input_tokens: usage_column(response.usage.input_tokens),
                    output_tokens: usage_column(response.usage.output_tokens),
            virtual_key_id: virtual_key_id.clone(),
            error_class: None,
        },
        );
        // Verbatim passthrough (Req 1.5): the typed response's own
        // Serialize emits model, answers (caller ids, extras,
        // absent-answer_confidence omitted), and usage (zeros omitted,
        // provider extras round-tripped) exactly as deserialized — no
        // reassembly, no renormalization.
        Json(response).into_response()
    }
    // Err carries (error, attempts); the attempts were already
    // surfaced above via the `#attempt` rows and telemetry spans.
    Err((error, _attempts)) => {
            // No-candidate case (Req 3.5): the router marks it
            // `Provider { status_code: Some(404), provider: "decisions" }`
            // with a message naming the model; surface it as the
            // 404-class `invalid_request_error` body instead of the
            // generic provider-error shape.
            if let GatewayError::Provider {
                status_code: Some(404),
                provider,
                message,
            } = &error
            {
        if provider.as_str() == "decisions" {
            tracing::warn!(
                trace_id = %trace_id,
                model = %logged_model,
                duration_ms,
                "Decision request found no configured decision model"
            );
            write_decision_log(
                &state,
                &trace_id,
                DecisionLogRow {
                    requested_model: Some(logged_model.clone()),
                            responded_model: None,
                            provider: provider.clone(),
                            status_code: StatusCode::NOT_FOUND.as_u16(),
                            duration_ms,
                            input_tokens: None,
                            output_tokens: None,
                            virtual_key_id: virtual_key_id.clone(),
                            error_class: Some(error_class_for(&error)),
                        },
                    );
                    return (
                        StatusCode::NOT_FOUND,
                        Json(serde_json::json!({
                            "error": { "message": message, "type": "invalid_request_error" }
                        })),
                    )
                        .into_response();
                }
            }
        tracing::warn!(
            trace_id = %trace_id,
            model = %logged_model,
            duration_ms,
            error = %error,
            "Decision request failed"
        );
        write_decision_log(
            &state,
            &trace_id,
            DecisionLogRow {
                requested_model: Some(logged_model.clone()),
                    responded_model: None,
                    provider: decision_error_provider(&error),
                    status_code: error.status_code().as_u16(),
                    duration_ms,
                    input_tokens: None,
                    output_tokens: None,
                    virtual_key_id: virtual_key_id.clone(),
                    error_class: Some(error_class_for(&error)),
                },
            );
            error.into_response()
        }
    }
}

fn decision_trace_id(headers: &HeaderMap) -> String {
    let request_id = headers
        .get("x-request-id")
        .or_else(|| headers.get("x-trace-id"))
        .and_then(|value| value.to_str().ok())
        .and_then(sanitize_trace_header);
    match request_id {
        Some(id) => generate_trace_id(Some(&id)),
        None => generate_trace_id(None),
    }
}

/// Bound a caller-controlled string before it enters an error body,
/// log field, or telemetry attribute (log-forging/amplification
/// guard): control characters stripped, then capped at `max_chars`
/// with a `...` marker kept inside the cap.
fn sanitize_bounded(value: &str, max_chars: usize) -> String {
    let mut clean: String = value.chars().filter(|c| !c.is_control()).collect();
    if clean.chars().count() > max_chars {
        let prefix: String = clean.chars().take(max_chars.saturating_sub(3)).collect();
        clean = format!("{prefix}...");
    }
    clean
}

/// H4: trace-id headers are echoed into logs — keep only the
/// `[A-Za-z0-9_-]` alphabet, capped at 64 chars; an empty result is
/// treated as absent so [`generate_trace_id`] mints a fresh UUID.
fn sanitize_trace_header(value: &str) -> Option<String> {
    let cleaned: String = value
        .chars()
        .filter(|c| c.is_ascii_alphanumeric() || matches!(c, '_' | '-'))
        .take(64)
        .collect();
    (!cleaned.is_empty()).then_some(cleaned)
}

/// Parse and validate a decision request body (Req 1.4): `model` a
/// non-empty string, `state` present (any JSON value), `questions` a
/// non-empty object of typed questions. Errors name the field.
fn parse_decision_request(value: Value) -> Result<SystemOneRequest, String> {
    let object = value
        .as_object()
        .ok_or_else(|| "body: request must be a JSON object".to_string())?;
    let model = object
        .get("model")
        .and_then(Value::as_str)
        .map(str::trim)
        .filter(|model| !model.is_empty())
        .ok_or_else(|| "model: required and must be a non-empty string".to_string())?
        .to_string();
    let state = object
        .get("state")
        .cloned()
        .ok_or_else(|| "state: required field".to_string())?;
    let questions_value = object
        .get("questions")
        .ok_or_else(|| "questions: required field".to_string())?;
    let questions = parse_questions(questions_value)?;
    Ok(SystemOneRequest {
        state,
        model,
        questions,
    })
}

fn parse_questions(value: &Value) -> Result<HashMap<String, Question>, String> {
    let map = value
        .as_object()
        .ok_or_else(|| "questions: must be a non-empty object".to_string())?;
    if map.is_empty() {
        return Err("questions: must be a non-empty object".to_string());
    }
    let mut questions = HashMap::with_capacity(map.len());
    for (id, entry) in map {
        // H2: the caller-controlled id is reflected into the 400 body
        // and tracing — strip control characters and cap at 64 chars
        // (marker included) at the interpolation sites. The map key
        // keeps the ORIGINAL id so answers return under it (Req 1.3).
        let logged_id = sanitize_bounded(id, 64);
        if !entry.is_object() {
            return Err(format!("questions.{logged_id}: must be a question object"));
        }
        let question_type = match entry.get("type").and_then(Value::as_str) {
            Some("score") => "score",
            Some("choice") => "choice",
            Some("noul") => "noul",
            _ => {
                return Err(format!(
                    "questions.{logged_id}: 'type' must be one of 'score', 'choice', or 'noul'"
                ));
            }
        };
        questions.insert(
            id.clone(),
            Question {
            question_type,
            instructions: entry.get("instructions").cloned().unwrap_or(Value::Null),
            criteria: entry.get("criteria").cloned().unwrap_or(Value::Null),
            // Unknown fields round-trip verbatim (Req 1.3).
            extra: entry
                .as_object()
                .map(|object| {
                    object
                        .iter()
                        .filter(|(key, _)| {
                            !matches!(key.as_str(), "type" | "instructions" | "criteria")
                        })
                        .map(|(key, value)| (key.clone(), value.clone()))
                        .collect()
                })
                .unwrap_or_default(),
        },
    );
    }
    Ok(questions)
}

/// Log the validation failure and return the standard 400
/// `invalid_request_error` body (the message names the offending field).
fn decision_rejected(
    state: &AppState,
    trace_id: &str,
    requested_model: Option<String>,
    duration_ms: u64,
    virtual_key_id: Option<&str>,
    message: &str,
) -> Response {
    tracing::warn!(
        trace_id = %trace_id,
        status = 400u16,
        reason = %message,
        "Decision request rejected"
    );
    let error = GatewayError::InvalidRequest(message.to_string());
    write_decision_log(
        state,
        trace_id,
        DecisionLogRow {
            requested_model,
            responded_model: None,
            provider: String::new(),
            status_code: StatusCode::BAD_REQUEST.as_u16(),
            duration_ms,
            input_tokens: None,
            output_tokens: None,
            virtual_key_id: virtual_key_id.map(str::to_string),
            error_class: Some(error_class_for(&error)),
        },
    );
    error.into_response()
}

/// Content-free request-log row inputs (Req 4.1-4.3): trace id, models,
/// provider, status, duration, usage, key attribution, error class.
/// Request/response bodies are always `None` (Req 4.2).
struct DecisionLogRow {
    requested_model: Option<String>,
    responded_model: Option<String>,
    provider: String,
    status_code: u16,
    duration_ms: u64,
    input_tokens: Option<i64>,
    output_tokens: Option<i64>,
    virtual_key_id: Option<String>,
    error_class: Option<String>,
}

fn write_decision_log(state: &AppState, trace_id: &str, row: DecisionLogRow) {
    let requested_model = row.requested_model.clone();
    let entry = LogEntry {
        trace_id: trace_id.to_string(),
        timestamp: chrono::Utc::now(),
        method: "POST".to_string(),
        path: DECISIONS_PATH.to_string(),
        model: row
            .responded_model
            .clone()
            .or_else(|| requested_model.clone())
            .unwrap_or_default(),
        provider: row.provider,
        status_code: row.status_code,
        duration_ms: row.duration_ms,
        cost: 0.0,
        request_body: None,
        response_body: None,
        requested_model,
        responded_model: row.responded_model,
        compression: None,
        memories_injected: 0,
        memories_stored: 0,
        injection_tokens: 0,
        detected_project: None,
        cache_read_tokens: None,
        cache_creation_tokens: None,
        cache_savings_cents: None,
        prefix_hash: None,
        reasoning_tokens: None,
        reasoning_compat_actions: None,
        error_class: row.error_class,
        context_reduction: None,
        client_protocol: None,
        claude_session_id: None,
        input_tokens: row.input_tokens,
        output_tokens: row.output_tokens,
        virtual_key_id: row.virtual_key_id,
    prompt_id: None,
    prompt_version: None,
    prompt_label: None,
    };
    if let Err(error) = state.logger.log(entry) {
        tracing::warn!(%error, trace_id = %trace_id, "Failed to write request log entry");
    }
}

/// Provider attribution for a failed decision call, mirroring
/// `RequestLogContext::from_error`.
fn decision_error_provider(error: &GatewayError) -> String {
    match error {
        GatewayError::Provider { provider, .. } => provider.clone(),
        GatewayError::AllProvidersFailed(agg) => agg
            .attempts
            .first()
            .map(|attempt| attempt.provider.clone())
            .unwrap_or_default(),
        _ => String::new(),
    }
}

/// The failed-try ledger from a `route_decision` result, both ways:
/// prior failed attempts on success, every attempt on failure. For
/// `AllProvidersFailed` the attempts are ALSO carried inside the
/// `AggregatedError` (client-visible body); the ledger here feeds the
/// internal `#attempt` rows, which the error body does not duplicate —
/// no double logging.
fn result_attempts(
    result: &Result<
        (SystemOneResponse, String, Vec<ProviderAttempt>),
        (GatewayError, Vec<ProviderAttempt>),
    >,
) -> &[ProviderAttempt] {
    match result {
        Ok((_response, _provider, attempts)) => attempts,
        Err((_error, attempts)) => attempts,
    }
}

/// Write one content-free failure log row per failed decision try —
/// the chat path's `#attempt` mechanism (`log_failed_attempts`,
/// handlers.rs) reused verbatim for field semantics: each row carries
/// the attempt's own `timestamp`, target `model`/`provider`, its
/// `status_code` (or [`ATTEMPT_FALLBACK_STATUS`] when it has none),
/// `duration_ms` and `error_class` (always stored), the error string
/// in `response_body`, and the SHARED parent `trace_id` so attempt
/// rows correlate with the final-outcome row. The provider/model
/// failure counter is bumped the same way. No request body is ever
/// logged; best-effort (failures only warned).
fn log_decision_attempts(state: &AppState, trace_id: &str, attempts: &[ProviderAttempt]) {
    for attempt in attempts {
        let status_code = attempt.status_code.unwrap_or(ATTEMPT_FALLBACK_STATUS);
        state
            .metrics
            .record_provider_attempt_failure(&attempt.provider, &attempt.model, status_code);

        let entry = LogEntry {
            trace_id: trace_id.to_string(),
            timestamp: attempt.timestamp,
            method: "POST".to_string(),
            path: DECISION_ATTEMPT_LOG_PATH.to_string(),
            model: attempt.model.clone(),
            provider: attempt.provider.clone(),
            status_code,
            duration_ms: attempt.duration_ms.unwrap_or(0),
            cost: 0.0,
            requested_model: None,
            responded_model: None,
            response_body: Some(attempt.error.clone()),
            request_body: None,
            compression: None,
            memories_injected: 0,
            memories_stored: 0,
            injection_tokens: 0,
            detected_project: None,
            cache_read_tokens: None,
            cache_creation_tokens: None,
            cache_savings_cents: None,
            prefix_hash: None,
            reasoning_tokens: None,
            reasoning_compat_actions: None,
            context_reduction: None,
            client_protocol: None,
            claude_session_id: None,
            // Attempt rows never carry usage; they are excluded from
            // cost reports by the `#attempt` path filter.
            input_tokens: None,
            output_tokens: None,
            virtual_key_id: None,
            prompt_id: None,
            prompt_version: None,
            prompt_label: None,
            error_class: Some(
                attempt
                    .error_class
                    .clone()
                    .unwrap_or_else(|| "other".to_string()),
            ),
        };
        if let Err(error) = state.logger.log(entry) {
            tracing::warn!(
                error = %error,
                trace_id = %trace_id,
                model = %attempt.model,
                provider = %attempt.provider,
                status_code,
                "Failed to write per-attempt failure log entry"
            );
        }
    }
}

/// Token-usage log column: `None` keeps "no usage reported" as NULL
/// rather than zero (cost-reporting convention).
fn usage_column(tokens: u64) -> Option<i64> {
    (tokens > 0).then(|| i64::try_from(tokens).unwrap_or(i64::MAX))
}

/// Root-only OTLP span for a decision request (Req 4.4, pass-through
/// pattern): the root span is emitted at completion with one failure
/// span per prior failed try (`failed_attempts`, the chat path's
/// attempt-ledger consumption); disabled path is a single atomic
/// check. Content contexts are never populated (NFR Security).
fn emit_decision_telemetry(
    state: &AppState,
    trace_id: &str,
    requested_model: &str,
    duration_ms: u64,
    virtual_key_id: Option<&str>,
    result: &Result<
        (SystemOneResponse, String, Vec<ProviderAttempt>),
        (GatewayError, Vec<ProviderAttempt>),
    >,
    failed_attempts: Vec<ProviderAttempt>,
) {
    if !state.telemetry_snapshot.enabled() {
        return;
    }
    let Some(emission) = state.telemetry_snapshot.emission_config() else {
        return;
    };
    let (status_code, usage, serving, response_model, error_class) = match result {
        Ok((response, provider, _attempts)) => (
            StatusCode::OK.as_u16(),
            Some(UsageCtx {
                input_tokens: i64::try_from(response.usage.input_tokens).unwrap_or(i64::MAX),
                output_tokens: i64::try_from(response.usage.output_tokens).unwrap_or(i64::MAX),
            }),
            Some(ServingAttempt {
                provider: provider.clone(),
                model: response.model.clone(),
            }),
            (response.model != requested_model).then(|| response.model.clone()),
            None,
        ),
        Err((error, _attempts)) => (
            error.status_code().as_u16(),
            None,
            None,
            None,
            Some(error_class_for(error)),
        ),
    };
    let args = TelemetryEmitArgs {
        request_id: trace_id.to_string(),
        inbound_context: None,
        method: "POST",
        route: DECISIONS_PATH,
        status_code,
        model_group: requested_model.to_string(),
        virtual_key_id: virtual_key_id.map(str::to_string),
        usage,
        cost: None,
        duration_ms,
        serving,
        failed_attempts,
        terminal_error_class: error_class,
        response_model,
        content: None,
        emission: Some(emission),
    };
    telemetry_seam::emit_request_spans(
        &state.telemetry_exporter,
        &state.telemetry_snapshot,
        &args,
    );
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Happy-path body used by the preservation test (wire-golden style,
    /// mirroring the systemone/models.rs test fixtures).
    fn valid_body() -> Value {
        serde_json::json!({
            "model": "jev-1.14.0",
            "state": { "messages": [{ "role": "user", "content": "help" }] },
            "questions": {
                "is_urgent": {
                    "type": "noul",
                    "instructions": "Is this urgent?",
                    "criteria": { "levels": ["no", "yes"] }
                },
                "department": {
                    "type": "choice",
                    "instructions": "Pick the team",
                    "criteria": { "options": ["billing", "technical"] }
                }
            }
        })
    }

    #[test]
    fn parse_decision_request_rejects_missing_or_empty_model() {
        let mut body = valid_body();
        body.as_object_mut().unwrap().remove("model");
        let error = parse_decision_request(body).expect_err("missing model must be rejected");
        assert!(error.contains("model"), "error must name the field: {error}");

        let mut body = valid_body();
        body["model"] = serde_json::json!("   ");
        let error = parse_decision_request(body).expect_err("blank model must be rejected");
        assert!(error.contains("model"), "error must name the field: {error}");

        // Non-string model is rejected with the same field-naming shape.
        let mut body = valid_body();
        body["model"] = serde_json::json!(42);
        let error = parse_decision_request(body).expect_err("non-string model must be rejected");
        assert!(error.contains("model"), "error must name the field: {error}");
    }

    #[test]
    fn parse_decision_request_rejects_missing_state() {
        let mut body = valid_body();
        body.as_object_mut().unwrap().remove("state");
        let error = parse_decision_request(body).expect_err("missing state must be rejected");
        assert!(error.contains("state"), "error must name the field: {error}");
    }

    #[test]
    fn parse_decision_request_rejects_missing_empty_or_non_object_questions() {
        let mut body = valid_body();
        body.as_object_mut().unwrap().remove("questions");
        let error = parse_decision_request(body).expect_err("missing questions must be rejected");
        assert!(error.contains("questions"), "{error}");

        let mut body = valid_body();
        body["questions"] = serde_json::json!({});
        let error = parse_decision_request(body).expect_err("empty questions must be rejected");
        assert!(error.contains("questions"), "{error}");

        let mut body = valid_body();
        body["questions"] = serde_json::json!([]);
        let error =
            parse_decision_request(body).expect_err("array questions must be rejected");
        assert!(error.contains("questions"), "{error}");
    }

    #[test]
    fn parse_decision_request_rejects_invalid_question_entries_naming_the_id() {
        let mut body = valid_body();
        body["questions"]["is_urgent"] = serde_json::json!("just a string");
        let error =
            parse_decision_request(body).expect_err("non-object question must be rejected");
        assert!(
            error.contains("questions.is_urgent"),
            "error must name questions.<id>: {error}"
        );

        let mut body = valid_body();
        body["questions"]["department"] = serde_json::json!({
            "type": "essay",
            "instructions": "write"
        });
        let error =
            parse_decision_request(body).expect_err("unknown question type must be rejected");
        assert!(
            error.contains("questions.department"),
            "error must name questions.<id>: {error}"
        );
        assert!(error.contains("type"), "{error}");

        // A missing type is equally invalid.
        let mut body = valid_body();
        body["questions"]["department"]
            .as_object_mut()
            .unwrap()
            .remove("type");
        let error =
            parse_decision_request(body).expect_err("missing question type must be rejected");
        assert!(error.contains("questions.department"), "{error}");
    }

    #[test]
    fn parse_decision_request_preserves_question_ids_contents_and_state() {
        let request = parse_decision_request(valid_body()).expect("valid body must parse");
        assert_eq!(request.model, "jev-1.14.0");
        assert_eq!(
            request.state,
            serde_json::json!({ "messages": [{ "role": "user", "content": "help" }] })
        );
        assert_eq!(request.questions.len(), 2, "both questions must parse");

        let urgent = request.questions.get("is_urgent").expect("id preserved");
        assert_eq!(urgent.question_type, "noul");
        assert_eq!(urgent.instructions, serde_json::json!("Is this urgent?"));
        assert_eq!(
            urgent.criteria,
            serde_json::json!({ "levels": ["no", "yes"] })
        );

        let department = request
            .questions
            .get("department")
            .expect("id preserved");
        assert_eq!(department.question_type, "choice");
        assert_eq!(department.instructions, serde_json::json!("Pick the team"));
    }

#[test]
fn parse_decision_request_rejects_non_object_body() {
    let error = parse_decision_request(serde_json::json!([]))
        .expect_err("array body must be rejected");
    assert!(error.contains("object"), "{error}");
}

#[test]
fn sanitize_bounded_strips_control_chars_and_caps_length() {
    // Control characters (newlines etc.) are stripped — no log
    // forging; plain spaces are kept.
    assert_eq!(sanitize_bounded("a\nb\r c", 64), "ab c");
    // Within the cap: unchanged.
    assert_eq!(sanitize_bounded("short", 64), "short");
    // Over the cap: truncated to the cap with the ellipsis marker
    // inside it.
    let long = "x".repeat(200);
    let bounded = sanitize_bounded(&long, 128);
    assert_eq!(bounded.chars().count(), 128);
    assert!(bounded.ends_with("..."), "{bounded}");
    // 64-char cap for question ids keeps the marker inside the cap.
    let id = "q".repeat(100);
    let bounded = sanitize_bounded(&id, 64);
    assert_eq!(bounded.chars().count(), 64);
    assert!(bounded.ends_with("..."));
}

#[test]
fn sanitize_trace_header_keeps_safe_alphabet_and_drops_the_rest() {
    assert_eq!(
        sanitize_trace_header("abc-DEF_123").as_deref(),
        Some("abc-DEF_123")
    );
    // Invalid characters are dropped, not rejected wholesale.
    assert_eq!(
        sanitize_trace_header("bad id\nwith junk!!!").as_deref(),
        Some("badidwithjunk")
    );
    // Over 64 chars: capped.
    let long = "a".repeat(100);
    assert_eq!(
        sanitize_trace_header(&long).as_deref().map(str::len),
        Some(64)
    );
    // Nothing left after sanitizing: treated as absent.
    assert_eq!(sanitize_trace_header("!!!\n\r\t"), None);
    assert_eq!(sanitize_trace_header(""), None);
}
}
