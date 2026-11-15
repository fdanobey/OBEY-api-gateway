//! Pure event-envelope types for model-deprecation webhooks
//! (model-deprecation-alerts spec).
//!
//! The envelope mirrors the budget-alert wire contract
//! ([`crate::webhooks::events`]): `POST` JSON body
//! `{"id","type","created_at","signed","data":{...}}`, byte-stable under
//! [`serde_json`]. The event name `model.deprecated` and the
//! [`crate::config::BudgetEventType`] enum are owned by the config module
//! (single config authority); this module owns the deprecation payload
//! and its typed `data` body.
//!
//! _Requirements: 3.3 (payload fields), 4.5 (event emission contract)_

use chrono::{DateTime, Utc};
use serde::{Deserialize, Serialize};

use crate::config::BudgetEventType;

/// Where a deprecation signal came from. Serialized verbatim as
/// `"provider_metadata" | "operator_override" | "runtime_detection"` in
/// the event `data` body (design "Event payload").
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum EvidenceSource {
    #[serde(rename = "provider_metadata")]
    ProviderMetadata,
    #[serde(rename = "operator_override")]
    OperatorOverride,
    #[serde(rename = "runtime_detection")]
    RuntimeDetection,
}

/// Typed `data` body of a [`ModelDeprecatedEvent`] (design "Event
/// payload", fields verbatim). Optional fields are `None` when unknown
/// (`retirement_date`/`days_remaining` on a date-less deprecation,
/// `replacement_model` when no hint exists, `last_used_at` when the
/// usage ledger has no recent hit).
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct ModelDeprecatedEventData {
    /// Provider name the deprecated model belongs to.
    pub provider: String,
    /// Deprecated provider model id (never a `vk_` secret value).
    pub model: String,
    /// Known retirement date; `None` on date-less deprecations
    /// (Req 3.1 `deprecated: true` with no date field).
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub retirement_date: Option<String>,
    /// Whole days until `retirement_date`; `None` when the date is
    /// unknown.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub days_remaining: Option<i64>,
    /// Replacement model id when known (scan `successor_model` or
    /// operator `replacement_model` override).
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub replacement_model: Option<String>,
    /// Which signal opened the episode (Req 3.3).
    pub evidence_source: EvidenceSource,
    /// Requests routed to this exact `(provider, model)` pair over the
    /// last 7 days (usage gate, Req 3.1/3.2).
    pub requests_7d: u64,
    /// Last request timestamp from the usage ledger; `None` when the
    /// ledger has no recent entry.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub last_used_at: Option<String>,
    /// `true` when runtime detection saw the model gone (404/410
    /// model-not-found) — the stronger escalation signal (Req 4.4).
    pub runtime_gone: bool,
}

/// The model-deprecation webhook event envelope (design "Event
/// payload"). Field names are verbatim on the wire, identical to the
/// budget-alert envelope:
/// `{"id","type","created_at","signed","data":{...}}`. `type` reuses the
/// config module's [`BudgetEventType`] — one event-name authority, no
/// duplicate enum.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct ModelDeprecatedEvent {
    /// Fresh UUID v4 per event; also the `webhook-id` header value when
    /// signed.
    pub id: uuid::Uuid,
    /// Event name: `model.deprecated`.
    #[serde(rename = "type")]
    pub event_type: BudgetEventType,
    /// UTC creation instant; RFC 3339 via chrono's serde impl.
    pub created_at: DateTime<Utc>,
    /// `false` when the receiving endpoint has no secret (unsigned
    /// delivery). Filled per-endpoint by the delivery task; the
    /// constructor defaults it to `false`.
    #[serde(default)]
    pub signed: bool,
    /// Typed event body.
    pub data: ModelDeprecatedEventData,
}

impl ModelDeprecatedEvent {
    /// Build a `model.deprecated` event for a newly detected
    /// deprecation episode (Reqs 3.3, 4.1).
    pub fn new(data: ModelDeprecatedEventData) -> Self {
        Self {
            id: uuid::Uuid::new_v4(),
            event_type: BudgetEventType::ModelDeprecated,
            created_at: Utc::now(),
            signed: false,
            data,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Envelope serialization is byte-stable: exact field set, exact
    /// names, `Option::None` omitted — mirroring
    /// `warning_envelope_serializes_to_exact_json`. Any drift here is a
    /// receiver-visible wire break.
    // Validates: Requirements 3.3, 4.5
    #[test]
    fn full_envelope_serializes_to_exact_json() {
        let data = ModelDeprecatedEventData {
            provider: "openai".to_string(),
            model: "gpt-4-turbo".to_string(),
            retirement_date: Some("2026-12-20".to_string()),
            days_remaining: Some(36),
            replacement_model: Some("gpt-4.1".to_string()),
            evidence_source: EvidenceSource::ProviderMetadata,
            requests_7d: 1523,
            last_used_at: Some("2026-11-14T18:22:09Z".to_string()),
            runtime_gone: false,
        };
        let event = ModelDeprecatedEvent {
            id: uuid::Uuid::parse_str("00000000-0000-4000-8000-00000000000a")
                .expect("valid uuid"),
            event_type: BudgetEventType::ModelDeprecated,
            created_at: DateTime::parse_from_rfc3339("2026-11-14T19:00:00Z")
                .expect("valid rfc3339")
                .with_timezone(&Utc),
            signed: false,
            data,
        };

        let json = serde_json::to_string(&event).expect("serializes");
        let expected = concat!(
            r#"{"id":"00000000-0000-4000-8000-00000000000a","type":"model.deprecated","#,
            r#""created_at":"2026-11-14T19:00:00Z","signed":false,"data":{"provider":"openai","#,
            r#""model":"gpt-4-turbo","retirement_date":"2026-12-20","days_remaining":36,"#,
            r#""replacement_model":"gpt-4.1","evidence_source":"provider_metadata","#,
            r#""requests_7d":1523,"last_used_at":"2026-11-14T18:22:09Z","runtime_gone":false}}"#
        );
        assert_eq!(json, expected);
    }

    /// Minimal envelope: date-less runtime-gone deprecation —
    /// `retirement_date`/`days_remaining`/`replacement_model`/
    /// `last_used_at` omitted (None), non-Option fields present.
    // Validates: Requirements 3.1 (date-less), 3.3, 4.5
    #[test]
    fn minimal_envelope_serializes_to_exact_json() {
        let data = ModelDeprecatedEventData {
            provider: "openrouter".to_string(),
            model: "meta-llm/llama-3-70b-instruct".to_string(),
            retirement_date: None,
            days_remaining: None,
            replacement_model: None,
            evidence_source: EvidenceSource::RuntimeDetection,
            requests_7d: 7,
            last_used_at: None,
            runtime_gone: true,
        };
        let event = ModelDeprecatedEvent {
            id: uuid::Uuid::parse_str("00000000-0000-4000-8000-00000000000b")
                .expect("valid uuid"),
            event_type: BudgetEventType::ModelDeprecated,
            created_at: DateTime::parse_from_rfc3339("2026-11-14T19:00:00Z")
                .expect("valid rfc3339")
                .with_timezone(&Utc),
            signed: false,
            data,
        };

        let json = serde_json::to_string(&event).expect("serializes");
        let expected = concat!(
            r#"{"id":"00000000-0000-4000-8000-00000000000b","type":"model.deprecated","#,
            r#""created_at":"2026-11-14T19:00:00Z","signed":false,"data":{"provider":"openrouter","#,
            r#""model":"meta-llm/llama-3-70b-instruct","evidence_source":"runtime_detection","#,
            r#""requests_7d":7,"runtime_gone":true}}"#
        );
        assert_eq!(json, expected);
    }

    /// Constructor stamps the right type and a fresh UUID v4; `signed`
    /// defaults to `false` (delivery flips it per-endpoint).
    // Validates: Requirements 4.1, 4.5
    #[test]
    fn constructor_sets_event_type_and_fresh_id() {
        let data = ModelDeprecatedEventData {
            provider: "openai".to_string(),
            model: "gpt-4-turbo".to_string(),
            retirement_date: None,
            days_remaining: None,
            replacement_model: None,
            evidence_source: EvidenceSource::OperatorOverride,
            requests_7d: 0,
            last_used_at: None,
            runtime_gone: false,
        };
        let event = ModelDeprecatedEvent::new(data);
        assert_eq!(event.event_type, BudgetEventType::ModelDeprecated);
        assert!(!event.signed);
        assert_eq!(event.id.get_version_num(), 4);
    }
}
