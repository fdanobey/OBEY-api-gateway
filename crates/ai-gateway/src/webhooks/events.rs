//! Pure event-envelope types for budget-alert webhooks
//! (budget-alerts-unified-webhooks spec).
//!
//! The envelope is the wire contract with every receiver: `POST` JSON body
//! `{"id","type","created_at","signed","data":{...}}`, byte-stable under
//! [`serde_json`] so the Standard Webhooks HMAC in the delivery task signs
//! exactly the bytes the wiki example documents. Event names and the
//! [`crate::config::BudgetEventType`] enum are owned by the config module
//! (single config authority); this module owns the envelope and its typed
//! `data` body.
//!
//! _Requirements: 2.1 (envelope fields), 2.3 (reset payload: scope, source,
//! new window start), 1.3/1.4 (event types); spend-pace-forecasting 3.4
//! (forecast payload)_

use chrono::{DateTime, NaiveDate, Utc};
use serde::{Deserialize, Serialize};

use crate::config::BudgetEventType;
use crate::metrics::forecast::ForecastMethod;
use crate::performance_alerts::state::{Signal, Unit};

/// What the alert is about: a provider budget or a virtual key's budget.
///
/// Serialized verbatim as `"provider" | "virtual_key"` in the event `data`
/// body (design "Event envelope").
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum Scope {
    /// Provider budget scope; `key_id` stays `None`.
    Provider,
    /// Virtual-key budget scope; `key_id` carries the key id (never the
    /// `vk_` secret value itself).
    VirtualKey,
}

/// What re-armed a budget: the window rolled over, or an admin zeroed spend.
/// Reset events carry this in their `data` body (Req 2.3).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum ResetSource {
    /// Lazy daily/weekly/monthly window rollover (Req 2.1).
    #[serde(rename = "window_rollover")]
    WindowRollover,
    /// Admin reset via the existing admin endpoints (Req 2.2).
    #[serde(rename = "admin")]
    Admin,
}

/// Typed `data` body of a [`BudgetAlertEvent`] (design "Event envelope",
/// fields verbatim). Fields are `Option` because the three event types
/// populate different subsets: warnings carry spend and `percent`; exhausted
/// events carry spend; reset events carry the NEW `window_start` anchor and
/// `reset_source` (Req 2.3).
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct BudgetEventData {
    /// `"provider"` or `"virtual_key"`.
    pub scope: Scope,
    /// Provider name or virtual-key name.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub name: Option<String>,
    /// Virtual-key id (providers: `None`) — never the `vk_` value.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub key_id: Option<String>,
    /// Configured USD limit at evaluation time.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub limit_usd: Option<f64>,
    /// Current-window USD spend at evaluation time.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub spent_usd: Option<f64>,
    /// Warn threshold crossed (warning events; Req 1.3).
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub percent: Option<u8>,
    /// `"daily" | "weekly" | "monthly" | "manual" | "lifetime"`.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub window: Option<String>,
    /// UTC epoch seconds of the current window anchor; reset events carry
    /// the NEW anchor (Req 2.3).
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub window_start: Option<i64>,
    /// Present only on `budget.reset` events.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub reset_source: Option<ResetSource>,
    /// Present only on `budget.forecast.warning` events ( Req 3.4):
    /// the projection that fired. Threshold/reset events leave it
    /// `None`, so their wire bytes are unchanged.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub forecast: Option<BudgetForecastEventData>,
}

/// Typed `data` body of a `budget.forecast.warning` event
/// (spend-pace-forecasting design "Event payload", fields verbatim):
/// the projection that fired alongside the scope identity and spend
/// facts. `window_end` carries the windowed scope's window end;
/// `horizon_end` carries the manual scope's 30-day pace-horizon end
/// (exactly one of the two is `Some`).
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct BudgetForecastEventData {
    /// `"provider"` or `"virtual_key"`.
    pub scope: Scope,
    /// Provider name or virtual-key name.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub name: Option<String>,
    /// Virtual-key id (providers: `None`) — never the `vk_` value.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub key_id: Option<String>,
    /// Current-window USD spend at evaluation time.
    pub spent_usd: f64,
    /// Configured USD limit at evaluation time.
    pub limit_usd: f64,
    /// Projected end-of-window (or 30-day-horizon) spend.
    pub projected_usd: f64,
    /// Projection method: `window_pace` or `trailing_run_rate`.
    pub method: ForecastMethod,
    /// UTC epoch seconds of the window end (windowed scopes).
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub window_end: Option<i64>,
    /// UTC epoch seconds of the 30-day pace-horizon end (manual scopes).
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub horizon_end: Option<i64>,
    /// Projected UTC date the limit is reached (ISO-8601).
    pub projected_exhaustion_date: NaiveDate,
    /// Discriminator: always `"forecast"` (design Data Models).
    pub threshold_kind: String,
}

/// Typed `data` body of a `performance.anomaly` event
/// (performance-alerts design "Event payload", fields verbatim):
/// the finding that fired for one (scope, signal, day) against the
/// 7-day rolling baseline. `signal` and `unit` reuse the evaluation
/// core's enums (their snake_case serde forms match the design
/// strings byte-for-byte); `threshold` is the human-readable fixed
/// condition (e.g. `">=3x baseline and >=$50 over"`). The struct
/// carries only aggregates — no key ids, no request bodies — so the
/// no-secret invariant holds by construction.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct PerformanceAnomalyData {
    /// Signal that fired: `spend` | `traffic` | `error_rate` | `latency`.
    pub signal: Signal,
    /// `"gateway"` | `"provider"` | `"model_group"`.
    pub scope: String,
    /// Provider or model-group name; `"gateway"` for gateway scope.
    pub name: String,
    /// Observed day's value for the signal.
    pub observed_value: f64,
    /// 7-day rolling baseline value at evaluation time.
    pub baseline_value: f64,
    /// `usd` | `requests` | `ratio` | `ms`.
    pub unit: Unit,
    /// UTC day the finding evaluated (ISO `YYYY-MM-DD`).
    pub evaluated_day: NaiveDate,
    /// Human-readable fixed condition that fired.
    pub threshold: String,
}

/// Typed `data` body of a `usage.anomaly` event
/// (usage-anomaly-alerts design "Event payload", fields verbatim):
/// the per-key finding that fired for one (key, signal, day) against
/// the key's own 7-day baseline. `signal` and `unit` are the wire
/// names as plain strings so this payload stands alone from the
/// evaluation core's enums; `key_id` is the dedup identity — never
/// the `vk_` secret value — and `name` is the cosmetic display name.
/// Like the performance payload, the struct carries only aggregates
/// — no request bodies, no key material — so the no-secret
/// invariant holds by construction.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct UsageAnomalyData {
    /// Signal that fired: `spend` | `traffic` | `tokens` |
    /// `new_activity` | `odd_hours` | `new_model` | `probes`.
    pub signal: String,
    /// Always `"virtual_key"` (the only usage-anomaly scope).
    pub scope: String,
    /// Virtual-key id — the dedup identity; never the `vk_` value.
    pub key_id: String,
    /// Key display name (cosmetic only).
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub name: Option<String>,
    /// Observed day's value for the signal.
    pub observed_value: f64,
    /// 7-day rolling baseline value at evaluation time.
    pub baseline_value: f64,
    /// `usd` | `requests` | `tokens` | `ratio`.
    pub unit: String,
    /// UTC day the finding evaluated (ISO `YYYY-MM-DD`).
    pub evaluated_day: String,
    /// Signal-specific context (bucket share, model name,
    /// validation-failure count).
    pub detail: String,
    /// Human-readable fixed condition that fired.
    pub threshold: String,
}

impl UsageAnomalyData {
    /// Build a `usage.anomaly` data body for one (key, signal, day)
    /// finding: scope is fixed `"virtual_key"`; `key_id` is the
    /// virtual key's id (never the `vk_` secret); `name` is the
    /// cosmetic display name.
    pub fn new(
        signal: String,
        key_id: String,
        name: Option<String>,
        observed_value: f64,
        baseline_value: f64,
        unit: String,
        evaluated_day: String,
        detail: String,
        threshold: String,
    ) -> Self {
        Self {
            signal,
            scope: "virtual_key".to_string(),
            key_id,
            name,
            observed_value,
            baseline_value,
            unit,
            evaluated_day,
            detail,
            threshold,
        }
    }
}

/// The webhook event envelope (design "Event envelope"). Field names are
/// verbatim on the wire: `{"id","type","created_at","signed","data":{...}}`.
/// `type` reuses the config module's [`BudgetEventType`] (Task 1) — one
/// event-name authority, no duplicate enum.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct BudgetAlertEvent {
    /// Fresh UUID v4 per event; also the `webhook-id` header value when
    /// signed.
    pub id: uuid::Uuid,
    /// Event name: `budget.warning` | `budget.exhausted` | `budget.reset`
    /// | `budget.forecast.warning`.
    #[serde(rename = "type")]
    pub event_type: BudgetEventType,
    /// UTC creation instant; RFC 3339 via chrono's serde impl.
    pub created_at: DateTime<Utc>,
    /// `false` when the receiving endpoint has no secret (unsigned
    /// delivery). Filled per-endpoint by the delivery task; constructors
    /// here default it to `false`.
    #[serde(default)]
    pub signed: bool,
    /// Typed event body.
    pub data: BudgetEventData,
}

impl BudgetAlertEvent {
    /// Build a `budget.warning` event: spend crossed `percent`% of the limit
    /// for the first time this window (Req 1.3).
    pub fn warning(data: BudgetEventData) -> Self {
        Self {
            id: uuid::Uuid::new_v4(),
            event_type: BudgetEventType::BudgetWarning,
            created_at: Utc::now(),
            signed: false,
            data,
        }
    }

    /// Build a `budget.exhausted` event: spend met or crossed the limit for
    /// the first time this window (Req 1.4).
    pub fn exhausted(data: BudgetEventData) -> Self {
        Self {
            id: uuid::Uuid::new_v4(),
            event_type: BudgetEventType::BudgetExhausted,
            created_at: Utc::now(),
            signed: false,
            data,
        }
    }

    /// Build a `budget.reset` event: the budget re-armed, `data` carrying
    /// the NEW `window_start` anchor and `reset_source` (Reqs 2.1-2.3).
    pub fn reset(data: BudgetEventData) -> Self {
        Self {
            id: uuid::Uuid::new_v4(),
            event_type: BudgetEventType::BudgetReset,
            created_at: Utc::now(),
            signed: false,
            data,
        }
    }

    /// Build a `budget.forecast.warning` event: the projection first
    /// shows exhaustion within the window (spend-pace-forecasting
    /// Req 3.1); `data.forecast` carries the projection payload
    /// (Req 3.4).
    pub fn forecast_warning(mut data: BudgetEventData, forecast: BudgetForecastEventData) -> Self {
        data.forecast = Some(forecast);
        Self {
            id: uuid::Uuid::new_v4(),
            event_type: BudgetEventType::BudgetForecastWarning,
            created_at: Utc::now(),
            signed: false,
            data,
        }
    }
}

/// Event type string for admin test sends (Req 4.4). Deliberately NOT a
/// [`BudgetEventType`] variant: `webhook.test` is synthesized by the
/// webhooks module and is never a configurable event filter value (the
/// config enum owns that vocabulary; see its doc comment).
pub const WEBHOOK_TEST_EVENT_TYPE: &str = "webhook.test";

/// Synthetic `data` body of a [`WebhookTestEvent`]: `{"scope":"test",
/// "url":<endpoint url>}` (design "Event envelope", `webhook.test`).
#[derive(Debug, Clone, PartialEq, Serialize)]
pub struct WebhookTestData {
    /// Always the literal `"test"`.
    pub scope: String,
    /// The endpoint URL the test send addressed.
    pub url: String,
}

/// Envelope for admin test sends: same wire shape as
/// [`BudgetAlertEvent`] but with `type: "webhook.test"` and synthetic
/// `data` (Req 4.4). Serialize-only — the gateway emits it, receivers
/// never send it back.
#[derive(Debug, Clone, PartialEq, Serialize)]
pub struct WebhookTestEvent {
    /// Fresh UUID v4 per test send; `webhook-id` header when signed.
    pub id: uuid::Uuid,
    /// Always [`WEBHOOK_TEST_EVENT_TYPE`].
    #[serde(rename = "type")]
    pub event_type: String,
    /// UTC creation instant; RFC 3339 via chrono's serde impl.
    pub created_at: DateTime<Utc>,
    /// `false` when the receiving endpoint has no secret; the delivery
    /// task flips it per-endpoint before serialization.
    #[serde(default)]
    pub signed: bool,
    /// Synthetic test body.
    pub data: WebhookTestData,
}

impl WebhookTestEvent {
    /// Build a `webhook.test` event addressed at `url` (Req 4.4). The
    /// delivery task sets `signed` per-endpoint before serializing.
    pub fn new(url: &str) -> Self {
        Self {
            id: uuid::Uuid::new_v4(),
            event_type: WEBHOOK_TEST_EVENT_TYPE.to_string(),
            created_at: Utc::now(),
            signed: false,
            data: WebhookTestData {
                scope: "test".to_string(),
                url: url.to_string(),
            },
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Envelope serialization is byte-stable: exact field set, exact names,
    /// `Option::None` omitted. This underpins the HMAC doc example (the
    /// delivery task signs precisely these bytes) — any drift here is a
    /// receiver-visible wire break.
    // Validates: Requirement 2.1 (envelope fields)
    #[test]
    fn warning_envelope_serializes_to_exact_json() {
        let data = BudgetEventData {
            scope: Scope::Provider,
            name: Some("openai".to_string()),
            key_id: None,
            limit_usd: Some(10.0),
            spent_usd: Some(8.0),
            percent: Some(80),
            window: Some("daily".to_string()),
            window_start: Some(1_735_689_600),
            reset_source: None,
            forecast: None,
        };
        let event = BudgetAlertEvent {
            id: uuid::Uuid::parse_str("00000000-0000-4000-8000-000000000001")
                .expect("valid uuid"),
            event_type: BudgetEventType::BudgetWarning,
            created_at: DateTime::parse_from_rfc3339("2025-01-01T00:00:00Z")
                .expect("valid rfc3339")
                .with_timezone(&Utc),
            signed: false,
            data,
        };

        let json = serde_json::to_string(&event).expect("serializes");
        let expected = concat!(
            r#"{"id":"00000000-0000-4000-8000-000000000001","type":"budget.warning","#,
            r#""created_at":"2025-01-01T00:00:00Z","signed":false,"data":{"scope":"provider","#,
            r#""name":"openai","limit_usd":10.0,"spent_usd":8.0,"percent":80,"window":"daily","#,
            r#""window_start":1735689600}}"#
        );
        assert_eq!(json, expected);
    }

    /// Exhausted envelope: same shape, `percent` absent (warn-only field).
    // Validates: Requirements 1.4, 2.1
    #[test]
    fn exhausted_envelope_serializes_to_exact_json() {
        let data = BudgetEventData {
            scope: Scope::VirtualKey,
            name: Some("ci-runner".to_string()),
            key_id: Some("key_123".to_string()),
            limit_usd: Some(5.0),
            spent_usd: Some(5.0),
            percent: None,
            window: Some("lifetime".to_string()),
            window_start: None,
            reset_source: None,
            forecast: None,
        };
        let event = BudgetAlertEvent {
            id: uuid::Uuid::parse_str("00000000-0000-4000-8000-000000000002")
                .expect("valid uuid"),
            event_type: BudgetEventType::BudgetExhausted,
            created_at: DateTime::parse_from_rfc3339("2025-01-01T00:00:00Z")
                .expect("valid rfc3339")
                .with_timezone(&Utc),
            signed: false,
            data,
        };

        let json = serde_json::to_string(&event).expect("serializes");
        let expected = concat!(
            r#"{"id":"00000000-0000-4000-8000-000000000002","type":"budget.exhausted","#,
            r#""created_at":"2025-01-01T00:00:00Z","signed":false,"data":{"scope":"virtual_key","#,
            r#""name":"ci-runner","key_id":"key_123","limit_usd":5.0,"spent_usd":5.0,"window":"lifetime"}}"#
        );
        assert_eq!(json, expected);
    }

    /// Reset envelope: carries the NEW `window_start` anchor and
    /// `reset_source`; no spend/percent fields.
    // Validates: Requirement 2.3
    #[test]
    fn reset_envelope_serializes_to_exact_json() {
        let data = BudgetEventData {
            scope: Scope::Provider,
            name: Some("openai".to_string()),
            key_id: None,
            limit_usd: None,
            spent_usd: None,
            percent: None,
            window: Some("daily".to_string()),
            window_start: Some(1_735_776_000),
            reset_source: Some(ResetSource::WindowRollover),
            forecast: None,
        };
        let event = BudgetAlertEvent {
            id: uuid::Uuid::parse_str("00000000-0000-4000-8000-000000000003")
                .expect("valid uuid"),
            event_type: BudgetEventType::BudgetReset,
            created_at: DateTime::parse_from_rfc3339("2025-01-02T00:00:00Z")
                .expect("valid rfc3339")
                .with_timezone(&Utc),
            signed: false,
            data,
        };

        let json = serde_json::to_string(&event).expect("serializes");
        let expected = concat!(
            r#"{"id":"00000000-0000-4000-8000-000000000003","type":"budget.reset","#,
            r#""created_at":"2025-01-02T00:00:00Z","signed":false,"data":{"scope":"provider","#,
            r#""name":"openai","window":"daily","window_start":1735776000,"reset_source":"window_rollover"}}"#
        );
        assert_eq!(json, expected);
    }

    /// Constructors stamp the right type and fresh UUID v4s; `signed`
    /// defaults to `false` (delivery flips it per-endpoint).
    // Validates: Requirements 1.3, 1.4, 2.1
    #[test]
    fn constructors_set_event_type_and_fresh_ids() {
        let data = BudgetEventData {
            scope: Scope::Provider,
            name: None,
            key_id: None,
            limit_usd: None,
            spent_usd: None,
            percent: None,
            window: None,
            window_start: None,
            reset_source: None,
            forecast: None,
        };
        let w = BudgetAlertEvent::warning(data.clone());
        let e = BudgetAlertEvent::exhausted(data.clone());
        let r = BudgetAlertEvent::reset(data);
        assert_eq!(w.event_type, BudgetEventType::BudgetWarning);
        assert_eq!(e.event_type, BudgetEventType::BudgetExhausted);
        assert_eq!(r.event_type, BudgetEventType::BudgetReset);
        assert!(!w.signed && !e.signed && !r.signed);
        // UUID v4 variant bits set, ids distinct.
        assert_ne!(w.id, e.id);
        assert_ne!(e.id, r.id);
        assert_eq!(w.id.get_version_num(), 4);
}

/// Forecast envelope: `type` is `budget.forecast.warning`, `data.forecast`
/// carries the projection payload per the design Data Models (method,
/// projected spend, window end or horizon end, exhaustion date,
/// `threshold_kind: "forecast"`). Byte-stable like every other envelope —
/// any drift here is a receiver-visible wire break.
// Validates: spend-pace-forecasting Requirement 3.4
#[test]
fn forecast_envelope_serializes_to_exact_json() {
    let data = BudgetEventData {
        scope: Scope::VirtualKey,
        name: Some("ci-runner".to_string()),
        key_id: Some("key_123".to_string()),
        limit_usd: Some(100.0),
        spent_usd: Some(40.0),
        percent: None,
        window: Some("manual".to_string()),
        window_start: Some(0),
        reset_source: None,
        forecast: Some(BudgetForecastEventData {
            scope: Scope::VirtualKey,
            name: Some("ci-runner".to_string()),
            key_id: Some("key_123".to_string()),
            spent_usd: 40.0,
            limit_usd: 100.0,
            projected_usd: 160.0,
            method: ForecastMethod::TrailingRunRate,
            window_end: None,
            horizon_end: Some(1_761_913_600),
            projected_exhaustion_date: NaiveDate::from_ymd_opt(2025, 11, 27).expect("valid date"),
            threshold_kind: "forecast".to_string(),
        }),
    };
    let event = BudgetAlertEvent {
        id: uuid::Uuid::parse_str("00000000-0000-4000-8000-000000000004")
            .expect("valid uuid"),
        event_type: BudgetEventType::BudgetForecastWarning,
        created_at: DateTime::parse_from_rfc3339("2025-01-01T00:00:00Z")
            .expect("valid rfc3339")
            .with_timezone(&Utc),
        signed: false,
        data,
    };

    let json = serde_json::to_string(&event).expect("serializes");
    let expected = concat!(
        r#"{"id":"00000000-0000-4000-8000-000000000004","type":"budget.forecast.warning","#,
        r#""created_at":"2025-01-01T00:00:00Z","signed":false,"data":{"scope":"virtual_key","#,
        r#""name":"ci-runner","key_id":"key_123","limit_usd":100.0,"spent_usd":40.0,"window":"manual","#,
        r#""window_start":0,"forecast":{"scope":"virtual_key","name":"ci-runner","#,
        r#""key_id":"key_123","spent_usd":40.0,"limit_usd":100.0,"projected_usd":160.0,"#,
        r#""method":"trailing_run_rate","horizon_end":1761913600,"projected_exhaustion_date":"2025-11-27","#,
        r#""threshold_kind":"forecast"}}}"#
    );
    assert_eq!(json, expected);
}

/// Windowed forecast envelope: the payload carries `window_end` (not
/// `horizon_end`) for windowed scopes and the `window_pace` method.
// Validates: spend-pace-forecasting Requirement 3.4
#[test]
fn forecast_envelope_windowed_carries_window_end() {
    let data = BudgetEventData {
        scope: Scope::Provider,
        name: Some("openai".to_string()),
        key_id: None,
        limit_usd: Some(100.0),
        spent_usd: Some(90.0),
        percent: None,
        window: Some("monthly".to_string()),
        window_start: Some(1_735_689_600),
        reset_source: None,
        forecast: Some(BudgetForecastEventData {
            scope: Scope::Provider,
            name: Some("openai".to_string()),
            key_id: None,
            spent_usd: 90.0,
            limit_usd: 100.0,
            projected_usd: 120.0,
            method: ForecastMethod::WindowPace,
            window_end: Some(1_738_368_000),
            horizon_end: None,
            projected_exhaustion_date: NaiveDate::from_ymd_opt(2025, 1, 20).expect("valid date"),
            threshold_kind: "forecast".to_string(),
        }),
    };
    let forecast_payload = data.forecast.clone().expect("some");
    let event = BudgetAlertEvent::forecast_warning(data, forecast_payload);
    assert_eq!(event.event_type, BudgetEventType::BudgetForecastWarning);
    let json = serde_json::to_value(&event).expect("serializes");
    let forecast = json
        .get("data")
        .and_then(|data| data.get("forecast"))
        .expect("forecast payload present");
    assert_eq!(forecast["method"], serde_json::json!("window_pace"));
    assert_eq!(forecast["window_end"], serde_json::json!(1_738_368_000));
    assert!(forecast.get("horizon_end").is_none());
    assert_eq!(
        forecast["projected_exhaustion_date"],
        serde_json::json!("2025-01-20")
    );
    assert_eq!(forecast["threshold_kind"], serde_json::json!("forecast"));
    assert_eq!(forecast["spent_usd"], serde_json::json!(90.0));
    assert_eq!(forecast["limit_usd"], serde_json::json!(100.0));
    assert_eq!(forecast["projected_usd"], serde_json::json!(120.0));
}

/// Anomaly payload: exact field set and names (signal, scope, name,
/// observed_value, baseline_value, unit, evaluated_day, threshold),
/// `evaluated_day` rendered ISO `YYYY-MM-DD`. Byte-stable like every
/// other payload — any drift here is a receiver-visible wire break.
// Validates: performance-alerts design "Event payload"
#[test]
fn anomaly_payload_serializes_to_exact_json() {
    let data = PerformanceAnomalyData {
        signal: Signal::Spend,
        scope: "provider".to_string(),
        name: "openai".to_string(),
        observed_value: 180.0,
        baseline_value: 40.0,
        unit: Unit::Usd,
        evaluated_day: NaiveDate::from_ymd_opt(2025, 11, 22).expect("valid date"),
        threshold: ">=3x baseline and >=$50 over".to_string(),
    };
    let json = serde_json::to_string(&data).expect("serializes");
    let expected = concat!(
        r#"{"signal":"spend","scope":"provider","name":"openai","#,
        r#""observed_value":180.0,"baseline_value":40.0,"unit":"usd","#,
        r#""evaluated_day":"2025-11-22","threshold":">=3x baseline and >=$50 over"}"#
    );
    assert_eq!(json, expected);
}

/// Anomaly payload round-trips through serde (receiver side deserializes
/// the exact wire bytes back to the same value).
// Validates: performance-alerts design "Event payload"
#[test]
fn anomaly_payload_round_trips() {
    let data = PerformanceAnomalyData {
        signal: Signal::ErrorRate,
        scope: "model_group".to_string(),
        name: "gpt-4o-family".to_string(),
        observed_value: 0.06,
        baseline_value: 0.01,
        unit: Unit::Ratio,
        evaluated_day: NaiveDate::from_ymd_opt(2026, 1, 5).expect("valid date"),
        threshold: ">=3x baseline and >=2pp over".to_string(),
    };
    let json = serde_json::to_string(&data).expect("serializes");
    let back: PerformanceAnomalyData = serde_json::from_str(&json).expect("deserializes");
    assert_eq!(back, data);
}

    /// No-secret invariant: the payload's serialized JSON carries no
    /// key-secret or request-body material — the struct has no field
    /// capable of carrying them (aggregates and names only), so the
    /// invariant holds by construction and this asserts the bytes.
    /// Worst-case strings in the free-form fields (`name`, `threshold`)
    /// cannot smuggle secrets or add payload fields: the serialized
    /// key set stays exactly the fixed eight-key wire contract, so a
    /// future field addition (the smuggling route) fails here first.
    // Validates: performance-alerts design "Event payload" (NFR security)
    #[test]
    fn anomaly_payload_carries_no_secrets() {
        let data = PerformanceAnomalyData {
            signal: Signal::Traffic,
            scope: "gateway".to_string(),
            name: "gateway".to_string(),
            observed_value: 12000.0,
            baseline_value: 1000.0,
            unit: Unit::Requests,
            evaluated_day: NaiveDate::from_ymd_opt(2025, 11, 23).expect("valid date"),
            threshold: ">=10x baseline and >=1000 requests over".to_string(),
        };
        let json = serde_json::to_string(&data).expect("serializes");
        for marker in ["vk_", "sk-", "Bearer ", "api_key", "authorization"] {
            assert!(
                !json.to_ascii_lowercase().contains(
                    marker.to_ascii_lowercase().as_str()
                ),
                "payload JSON must not contain {marker:?}: {json}"
            );
        }

 let hostile = PerformanceAnomalyData {
 name: "vk_test123 sk-abc Bearer api_key authorization".to_string(),
 threshold: "vk_ prefix sk- Bearer api_key authorization".to_string(),
 ..data
 };
 let value = serde_json::to_value(&hostile).expect("serializes");
 let object = value.as_object().expect("object");
 let mut keys: Vec<&str> = object.keys().map(String::as_str).collect();
 keys.sort_unstable();
 assert_eq!(
 keys,
 vec![
 "baseline_value",
 "evaluated_day",
 "name",
 "observed_value",
 "scope",
 "signal",
 "threshold",
 "unit",
 ]
 );
 }

/// Usage-anomaly payload: exact field set and names (signal, scope,
/// key_id, name, observed_value, baseline_value, unit, evaluated_day,
/// detail, threshold), `evaluated_day` rendered ISO `YYYY-MM-DD`.
/// Byte-stable like every other payload — any drift here is a
/// receiver-visible wire break.
// Validates: usage-anomaly-alerts design "Event payload" (Req 3.3)
#[test]
    fn usage_anomaly_payload_serializes_to_exact_json() {
    let data = UsageAnomalyData::new(
    "spend".to_string(),
    "key-42".to_string(),
    Some("prod-key".to_string()),
    180.0,
    40.0,
    "usd".to_string(),
    "2025-11-22".to_string(),
    "observed $180.00 vs baseline $40.00".to_string(),
    ">=3x baseline and >=$50 over".to_string(),
    );
    let json = serde_json::to_string(&data).expect("serializes");
    let expected = concat!(
    r#"{"signal":"spend","scope":"virtual_key","key_id":"key-42","#,
    r#""name":"prod-key","observed_value":180.0,"baseline_value":40.0,"#,
    r#""unit":"usd","evaluated_day":"2025-11-22","#,
    r#""detail":"observed $180.00 vs baseline $40.00","#,
    r#""threshold":">=3x baseline and >=$50 over"}"#
    );
    assert_eq!(json, expected);
    }

/// A missing display name serializes without the `name` key — the
/// payload shape stays byte-stable for legacy keys with no name.
// Validates: usage-anomaly-alerts design "Event payload"
#[test]
    fn usage_anomaly_payload_without_name_omits_the_key() {
    let data = UsageAnomalyData::new(
    "new_activity".to_string(),
    "key-7".to_string(),
    None,
    25.0,
    0.0,
    "requests".to_string(),
    "2026-01-05".to_string(),
    "first traffic after 7 idle days".to_string(),
    ">=25 requests on the evaluated day".to_string(),
    );
    let json = serde_json::to_string(&data).expect("serializes");
    assert!(!json.contains(r#""name":"#));
    assert!(json.contains(r#""key_id":"key-7""#));
    }

/// Usage-anomaly payload round-trips through serde (receiver side
/// deserializes the exact wire bytes back to the same value).
// Validates: usage-anomaly-alerts design "Event payload"
#[test]
    fn usage_anomaly_payload_round_trips() {
    let data = UsageAnomalyData::new(
    "odd_hours".to_string(),
    "key-123".to_string(),
    Some("nightly-batch".to_string()),
    5200.0,
    1000.0,
    "requests".to_string(),
    "2026-01-05".to_string(),
    "62% of requests in low-share buckets".to_string(),
    ">=50% of requests in buckets with <2% baseline share".to_string(),
    );
    let json = serde_json::to_string(&data).expect("serializes");
    let back: UsageAnomalyData = serde_json::from_str(&json).expect("deserializes");
    assert_eq!(back, data);
    }

/// No-secret invariant (Req 3.3): the payload's serialized JSON
/// carries the key ID and display name but never any `vk_` key
/// material — the struct has no field capable of carrying it, so the
/// invariant holds by construction and this asserts the bytes. A
/// HOSTILE display name that looks like a secret (`sk-abc123`,
/// `Bearer ...`) stays a plain display-name string: it is not
/// redacted, not treated as a secret, and cannot smuggle extra
/// payload fields — the serialized key set stays exactly the fixed
/// ten-key wire contract, so a future field addition (the smuggling
/// route) fails here first.
// Validates: usage-anomaly-alerts design "Event payload" (NFR security, Req 3.3)
#[test]
    fn usage_anomaly_payload_carries_no_secrets() {
    let data = UsageAnomalyData::new(
    "probes".to_string(),
    "key-99".to_string(),
    Some("staging".to_string()),
    120.0,
    8.0,
    "requests".to_string(),
    "2025-11-23".to_string(),
    "validation-class failures: 120 (baseline 8)".to_string(),
    ">=50 failures and >=5x baseline".to_string(),
    );
    let json = serde_json::to_string(&data).expect("serializes");
    // The clean payload carries no secret markers.
    for marker in ["vk_", "sk-", "Bearer ", "api_key", "authorization"] {
    assert!(
    !json.to_ascii_lowercase().contains(
    marker.to_ascii_lowercase().as_str()
    ),
    "payload JSON must not contain {marker:?}: {json}"
    );
    }

    // A hostile DISPLAY NAME that looks like key material is carried
    // verbatim as the cosmetic name (a display name is not a secret and
    // must not be dropped), but it must never add or rename payload
    // fields — the wire key set stays the fixed contract.
    let hostile = UsageAnomalyData::new(
    "probes".to_string(),
    "key-99".to_string(),
    Some("sk-abc123 Bearer vk_test123".to_string()),
    120.0,
    8.0,
    "requests".to_string(),
    "2025-11-23".to_string(),
    "vk_test123 sk-abc Bearer api_key authorization".to_string(),
    ">=50 failures and >=5x baseline".to_string(),
    );
    let value = serde_json::to_value(&hostile).expect("serializes");
    let object = value.as_object().expect("object");
    let mut keys: Vec<&str> = object.keys().map(String::as_str).collect();
    keys.sort_unstable();
    assert_eq!(
    keys,
    vec![
    "baseline_value",
    "detail",
    "evaluated_day",
    "key_id",
    "name",
    "observed_value",
    "scope",
    "signal",
    "threshold",
    "unit",
    ]
    );
    // The key id stays the id, not the vk_ value.
    assert_eq!(object["key_id"], serde_json::json!("key-99"));
    assert_eq!(
    object["name"],
    serde_json::json!("sk-abc123 Bearer vk_test123")
    );
    }
}
