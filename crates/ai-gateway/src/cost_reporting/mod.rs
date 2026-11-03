//! Cost reporting core types, parameter validation, and micro-USD aggregation
//! helpers (spec: billing-grade-cost-reporting, Task 1).
//!
//! Pure types and logic only: no SQL and no HTTP coupling. Query-layer and
//! admin-route wiring live in later tasks (`query.rs`, `csv.rs`, admin routes).
//!
//! Timestamp contract: `CostReportParams::start`/`end` and
//! `ReconcileParams::start`/`end` are RFC 3339 strings (e.g.
//! `2026-01-01T00:00:00Z`), matching the public query API; validation parses
//! them once via `chrono` (an existing dependency) into epoch seconds, and all
//! downstream logic uses integer epoch seconds only.
//!
//! Money contract (Requirement 6): every cost is aggregated as integer
//! micro-USD (`cost_micro_usd: i64`). Conversion to USD happens exactly once,
//! at serialization, through the single helper `micro_usd_to_usd` (f64) /
//! [`micro_usd_to_usd_string`] (exactly 6 decimal places, integer math, no
//! float accumulation). Serialized JSON exposes the field as `cost_usd`.

use chrono::DateTime;
use serde::{Deserialize, Serialize, Serializer};

pub mod csv;
pub mod query;

/// Maximum allowed filter length in characters for `provider`/`key_id`
/// (defense in depth: filters are bound SQL parameters, never concatenated,
/// but an unbounded filter invites abuse of the query surface).
pub const MAX_FILTER_CHARS: usize = 256;

/// Maximum allowed report range in days (Requirement 1.7).
pub const MAX_RANGE_DAYS: i64 = 366;

const SECONDS_PER_DAY: i64 = 86_400;

/// Dimension to group cost report rows by (Requirement 1).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum GroupBy {
    Day,
    Provider,
    Model,
    VirtualKey,
}

impl GroupBy {
    /// Parse the `group_by` query parameter value.
    pub fn parse(value: &str) -> Option<GroupBy> {
        match value {
            "day" => Some(GroupBy::Day),
            "provider" => Some(GroupBy::Provider),
            "model" => Some(GroupBy::Model),
            "virtual_key" => Some(GroupBy::VirtualKey),
            _ => None,
        }
    }

    pub fn as_str(&self) -> &'static str {
        match self {
            GroupBy::Day => "day",
            GroupBy::Provider => "provider",
            GroupBy::Model => "model",
            GroupBy::VirtualKey => "virtual_key",
        }
    }

    /// Allowed values, for validation error messages.
    pub fn allowed_values() -> &'static str {
        "day, provider, model, virtual_key"
    }
}

/// Output format for a cost report (Requirement 4).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ReportFormat {
    Csv,
    Json,
}

impl ReportFormat {
    /// Parse the `format` query parameter value.
    pub fn parse(value: &str) -> Option<ReportFormat> {
        match value {
            "csv" => Some(ReportFormat::Csv),
            "json" => Some(ReportFormat::Json),
            _ => None,
        }
    }

    pub fn as_str(&self) -> &'static str {
        match self {
            ReportFormat::Csv => "csv",
            ReportFormat::Json => "json",
        }
    }

    /// Allowed values, for validation error messages.
    pub fn allowed_values() -> &'static str {
        "csv, json"
    }
}

/// Typed validation error for cost report and reconcile parameters, mappable
/// to HTTP 400 with a JSON body naming the invalid parameter.
#[derive(Debug, Clone, PartialEq, Eq, thiserror::Error)]
pub enum CostReportValidationError {
    #[error("missing required query parameter: start (RFC 3339 timestamp)")]
    MissingStart,
    #[error("invalid 'start' value '{value}': expected an RFC 3339 timestamp, e.g. 2026-01-01T00:00:00Z")]
    InvalidStart { value: String },
    #[error("missing required query parameter: end (RFC 3339 timestamp)")]
    MissingEnd,
    #[error("invalid 'end' value '{value}': expected an RFC 3339 timestamp, e.g. 2026-01-01T00:00:00Z")]
    InvalidEnd { value: String },
    #[error("invalid 'end': must be greater than 'start'")]
    EndNotAfterStart,
    #[error(
        "invalid range: start/end span exceeds the {MAX_RANGE_DAYS}-day limit"
    )]
    RangeTooLarge,
    #[error("invalid 'group_by' value '{value}': must be one of {}", GroupBy::allowed_values())]
    InvalidGroupBy { value: String },
    #[error("invalid 'format' value '{value}': must be one of {}", ReportFormat::allowed_values())]
    InvalidFormat { value: String },
    #[error(
        "invalid '{parameter}' value: filter exceeds the {MAX_FILTER_CHARS} character limit"
    )]
    FilterTooLong { parameter: &'static str, len: usize },
}

impl CostReportValidationError {
    /// Name of the offending query parameter (or "range" for range-shape
    /// errors), for HTTP 400 JSON error bodies.
    pub fn parameter(&self) -> &'static str {
        match self {
            CostReportValidationError::MissingStart
            | CostReportValidationError::InvalidStart { .. } => "start",
            CostReportValidationError::MissingEnd
            | CostReportValidationError::InvalidEnd { .. }
            | CostReportValidationError::EndNotAfterStart => "end",
            CostReportValidationError::RangeTooLarge => "range",
            CostReportValidationError::InvalidGroupBy { .. } => "group_by",
            CostReportValidationError::InvalidFormat { .. } => "format",
            CostReportValidationError::FilterTooLong { parameter, .. } => parameter,
        }
    }
}

/// Raw `GET /admin/cost-reports` query parameters. `start`/`end` are RFC 3339
/// strings; everything else arrives as raw strings and is defaulted/validated
/// by [`CostReportParams::validate`].
#[derive(Debug, Clone, Default, Deserialize)]
pub struct CostReportParams {
    pub start: Option<String>,
    pub end: Option<String>,
    pub group_by: Option<String>,
    pub provider: Option<String>,
    pub key_id: Option<String>,
    pub format: Option<String>,
}

/// Raw `GET /admin/cost-reports/reconcile` query parameters.
#[derive(Debug, Clone, Default, Deserialize)]
pub struct ReconcileParams {
    pub start: Option<String>,
    pub end: Option<String>,
}

/// A validated time range in epoch seconds.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct ValidatedRange {
    pub start_secs: i64,
    pub end_secs: i64,
}

/// Validated cost report parameters, ready for the query layer.
#[derive(Debug, Clone, PartialEq)]
pub struct ValidatedCostReportParams {
    pub range: ValidatedRange,
    pub group_by: GroupBy,
    pub provider: Option<String>,
    pub key_id: Option<String>,
    pub format: ReportFormat,
}

fn parse_rfc3339_to_epoch(raw: &str) -> Option<i64> {
    DateTime::parse_from_rfc3339(raw.trim())
        .ok()
        .map(|dt| dt.timestamp())
}

/// Validate a start/end RFC 3339 pair: present, parseable, `end > start`,
/// and at most [`MAX_RANGE_DAYS`] days apart.
pub fn validate_range(
    start: &Option<String>,
    end: &Option<String>,
) -> Result<ValidatedRange, CostReportValidationError> {
    let start_raw = start.as_deref().map(str::trim).unwrap_or("");
    if start_raw.is_empty() {
        return Err(CostReportValidationError::MissingStart);
    }
    let start_secs = parse_rfc3339_to_epoch(start_raw)
        .ok_or_else(|| CostReportValidationError::InvalidStart {
            value: start_raw.to_string(),
        })?;

    let end_raw = end.as_deref().map(str::trim).unwrap_or("");
    if end_raw.is_empty() {
        return Err(CostReportValidationError::MissingEnd);
    }
    let end_secs = parse_rfc3339_to_epoch(end_raw).ok_or_else(|| {
        CostReportValidationError::InvalidEnd {
            value: end_raw.to_string(),
        }
    })?;

    if end_secs <= start_secs {
        return Err(CostReportValidationError::EndNotAfterStart);
    }
    if end_secs - start_secs > MAX_RANGE_DAYS * SECONDS_PER_DAY {
        return Err(CostReportValidationError::RangeTooLarge);
    }
    Ok(ValidatedRange {
        start_secs,
        end_secs,
    })
}

impl CostReportParams {
    /// Validate all parameters, applying the defaults `group_by=day` and
    /// `format=json` for omitted values (Requirements 1.2 and 4.2).
    pub fn validate(
        &self,
    ) -> Result<ValidatedCostReportParams, CostReportValidationError> {
        let range = validate_range(&self.start, &self.end)?;

        let group_by = match self.group_by.as_deref().map(str::trim) {
            None | Some("") => GroupBy::Day,
            Some(raw) => GroupBy::parse(raw).ok_or_else(|| {
                CostReportValidationError::InvalidGroupBy {
                    value: raw.to_string(),
                }
            })?,
        };

    let format = match self.format.as_deref().map(str::trim) {
        None | Some("") => ReportFormat::Json,
        Some(raw) => ReportFormat::parse(raw).ok_or_else(|| {
            CostReportValidationError::InvalidFormat {
                value: raw.to_string(),
            }
        })?,
    };

        // Filters normalize exactly like group_by/format: trim, then
        // empty → None so " " cannot silently narrow a report to nothing.
        let provider = normalize_filter(&self.provider);
        let key_id = normalize_filter(&self.key_id);
        for (parameter, filter) in [("provider", &provider), ("key_id", &key_id)] {
            if let Some(filter) = filter.as_deref() {
                if filter.chars().count() > MAX_FILTER_CHARS {
                    return Err(CostReportValidationError::FilterTooLong {
                        parameter,
                        len: filter.chars().count(),
                    });
                }
            }
        }

        Ok(ValidatedCostReportParams {
            range,
            group_by,
            provider,
            key_id,
            format,
        })
    }
}

/// Trim a filter parameter and collapse whitespace-only values to `None`
/// (same normalization `group_by`/`format` get, so `" openai"` filters like
/// `"openai"` instead of returning a silently empty report).
fn normalize_filter(raw: &Option<String>) -> Option<String> {
    match raw.as_deref().map(str::trim) {
        Some("") | None => None,
        Some(trimmed) => Some(trimmed.to_string()),
    }
}

impl ReconcileParams {
    /// Validate the start/end pair through the same range rules the cost
    /// report uses (present, RFC 3339, `end > start`, ≤ 366 days).
    pub fn validate(&self) -> Result<ValidatedRange, CostReportValidationError> {
        validate_range(&self.start, &self.end)
    }
}

/// Single conversion helper: integer micro-USD to USD (f64). Applied exactly
/// once per value at serialization; never used for intermediate math.
fn micro_usd_to_usd(micro_usd: i64) -> f64 {
    micro_usd as f64 / 1_000_000.0
}

/// Single conversion helper: integer micro-USD to a display string with
/// exactly 6 decimal places, via integer math (deterministic, no float
/// accumulation). Handles negative values defensively with the sign in front.
pub fn micro_usd_to_usd_string(micro_usd: i64) -> String {
    let magnitude = (micro_usd as i128).unsigned_abs();
    let whole = magnitude / 1_000_000;
    let frac = magnitude % 1_000_000;
    let sign = if micro_usd < 0 { "-" } else { "" };
    format!("{sign}{whole}.{frac:06}")
}

fn ser_cost_usd<S>(micro_usd: &i64, serializer: S) -> Result<S::Ok, S::Error>
where
    S: Serializer,
{
    serializer.serialize_f64(micro_usd_to_usd(*micro_usd))
}

/// One aggregated report row (design.md Data Models). `cost_micro_usd` is the
/// authoritative integer aggregate; JSON serialization exposes it as
/// `cost_usd` (f64) via the single conversion helper. Use
/// [`ReportRow::cost_usd_string`] where exactly 6 decimal places are required
/// (e.g. CSV).
#[derive(Debug, Clone, Serialize)]
pub struct ReportRow {
    /// `day` "YYYY-MM-DD" | provider name | model name | key_id | "unattributed".
    pub dimension: String,
    /// Epoch seconds, inclusive bounds of the aggregation window.
    pub window_start: i64,
    /// Epoch seconds, exclusive bound of the aggregation window.
    pub window_end: i64,
    pub requests: i64,
    pub input_tokens: i64,
    pub output_tokens: i64,
    pub cache_read_tokens: i64,
    pub cache_creation_tokens: i64,
    pub reasoning_tokens: i64,
    #[serde(rename = "cost_usd", serialize_with = "ser_cost_usd")]
    pub cost_micro_usd: i64,
    /// Requests whose cost could not be computed (missing model pricing);
    /// counted in `requests` but not in `cost_usd` (Requirement 1.8).
    pub cost_unpriced_requests: i64,
}

impl ReportRow {
    /// Exactly-6-decimal USD rendering of this row's cost (CSV, display).
    pub fn cost_usd_string(&self) -> String {
        micro_usd_to_usd_string(self.cost_micro_usd)
    }
}

/// A complete cost report: ordered rows plus a totals row over the full
/// requested range (Requirement 5.2 total row).
#[derive(Debug, Clone, Serialize)]
pub struct CostReport {
    pub rows: Vec<ReportRow>,
    pub totals: ReportRow,
}

impl CostReport {
    /// Build a report: order rows deterministically for `group_by`, then
    /// aggregate totals in integer micro-USD (single conversion at
    /// serialization only).
    pub fn new(
        group_by: GroupBy,
        mut rows: Vec<ReportRow>,
        range_start_secs: i64,
        range_end_secs: i64,
    ) -> CostReport {
        order_rows(group_by, &mut rows);
        let totals = total_row(&rows, range_start_secs, range_end_secs);
        CostReport { rows, totals }
    }
}

/// Deterministic output ordering (Requirement 6.3): `day` grouping keeps UTC
/// calendar order (dimension ascending); every other grouping orders by cost
/// descending with ties broken by dimension ascending — a total order.
pub fn order_rows(group_by: GroupBy, rows: &mut [ReportRow]) {
    match group_by {
        GroupBy::Day => rows.sort_by(|a, b| a.dimension.cmp(&b.dimension)),
        _ => sort_rows_by_cost_desc(rows),
    }
}

/// Cost descending, tie-break dimension ascending (total, deterministic).
pub fn sort_rows_by_cost_desc(rows: &mut [ReportRow]) {
    rows.sort_by(|a, b| {
        b.cost_micro_usd
            .cmp(&a.cost_micro_usd)
            .then_with(|| a.dimension.cmp(&b.dimension))
    });
}

fn total_row(rows: &[ReportRow], window_start: i64, window_end: i64) -> ReportRow {
    let mut totals = ReportRow {
        dimension: "total".to_string(),
        window_start,
        window_end,
        requests: 0,
        input_tokens: 0,
        output_tokens: 0,
        cache_read_tokens: 0,
        cache_creation_tokens: 0,
        reasoning_tokens: 0,
        cost_micro_usd: 0,
        cost_unpriced_requests: 0,
    };
    for row in rows {
        totals.requests = totals.requests.saturating_add(row.requests);
        totals.input_tokens = totals.input_tokens.saturating_add(row.input_tokens);
        totals.output_tokens = totals.output_tokens.saturating_add(row.output_tokens);
        totals.cache_read_tokens = totals
            .cache_read_tokens
            .saturating_add(row.cache_read_tokens);
        totals.cache_creation_tokens = totals
            .cache_creation_tokens
            .saturating_add(row.cache_creation_tokens);
        totals.reasoning_tokens = totals
            .reasoning_tokens
            .saturating_add(row.reasoning_tokens);
        totals.cost_micro_usd = totals.cost_micro_usd.saturating_add(row.cost_micro_usd);
        totals.cost_unpriced_requests = totals
            .cost_unpriced_requests
            .saturating_add(row.cost_unpriced_requests);
    }
    totals
}

/// Reconciliation ledger totals (design.md Data Models). `cost_usd` is
/// serialized from integer micro-USD via the single conversion helper.
#[derive(Debug, Clone, Serialize)]
pub struct LedgerTotals {
    /// Billable (non-`#attempt`) requests in the window.
    pub requests: i64,
    /// Integer micro-USD ledger cost; serialized as `cost_usd` (f64).
    #[serde(rename = "cost_usd", serialize_with = "ser_cost_usd")]
    pub cost_micro_usd: i64,
}

/// Reconciliation provider-spend totals: current-window store spend with the
/// window bounds and a caveat flag when the requested window is not aligned
/// to the provider budget window (design.md Error Handling #5).
#[derive(Debug, Clone, Serialize)]
pub struct ProviderSpendTotals {
    /// Integer micro-USD summed `current_spend_usd`; serialized as `cost_usd`.
    #[serde(rename = "cost_usd", serialize_with = "ser_cost_usd")]
    pub cost_micro_usd: i64,
    /// Earliest stored `window_start` anchor across provider rows.
    pub window_start: i64,
    /// Latest stored `updated_at` accrual instant.
    pub window_end: i64,
    /// True when the requested window is not aligned to the stored window.
    pub caveat: bool,
}

/// Reconciliation virtual-key usage totals.
#[derive(Debug, Clone, Serialize)]
pub struct KeyUsageTotals {
    /// Integer micro-USD windowed `virtual_key_usage` cost; serialized as `cost_usd`.
    #[serde(rename = "cost_usd", serialize_with = "ser_cost_usd")]
    pub cost_micro_usd: i64,
}

/// Reconciliation report comparing the request log ledger against the
/// provider budget store and virtual-key usage store for a window
/// (Requirement 3). `difference_usd` is ledger minus provider spend,
/// computed once in integer micro-USD.
#[derive(Debug, Clone, Serialize)]
pub struct ReconciliationReport {
    pub ledger: LedgerTotals,
    pub provider_spend: ProviderSpendTotals,
    pub key_usage: KeyUsageTotals,
    /// Requests in the window whose cost could not be computed.
    pub unpriced_requests: i64,
    /// Ledger minus provider spend in integer micro-USD; serialized as `difference_usd`.
    #[serde(rename = "difference_usd", serialize_with = "ser_cost_usd")]
    pub difference_micro_usd: i64,
}

impl ReconciliationReport {
    pub fn new(
        ledger: LedgerTotals,
        provider_spend: ProviderSpendTotals,
        key_usage: KeyUsageTotals,
        unpriced_requests: i64,
    ) -> ReconciliationReport {
        let difference_micro_usd = ledger
            .cost_micro_usd
            .saturating_sub(provider_spend.cost_micro_usd);
        ReconciliationReport {
            ledger,
            provider_spend,
            key_usage,
            unpriced_requests,
            difference_micro_usd,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn params(start: Option<&str>, end: Option<&str>) -> CostReportParams {
        CostReportParams {
            start: start.map(str::to_string),
            end: end.map(str::to_string),
            ..CostReportParams::default()
        }
    }

    fn row(dimension: &str, cost_micro_usd: i64) -> ReportRow {
        ReportRow {
            dimension: dimension.to_string(),
            window_start: 0,
            window_end: 0,
            requests: 1,
            input_tokens: 10,
            output_tokens: 20,
            cache_read_tokens: 0,
            cache_creation_tokens: 0,
        reasoning_tokens: 0,
        cost_micro_usd,
        cost_unpriced_requests: 0,
    }
}

    #[test]
    fn validation_missing_start_is_rejected() {
        let err = params(None, Some("2026-02-01T00:00:00Z"))
            .validate()
            .unwrap_err();
        assert_eq!(err, CostReportValidationError::MissingStart);
        assert_eq!(err.parameter(), "start");
        assert!(err.to_string().contains("start"));
    }

    #[test]
    fn validation_unparseable_start_is_rejected() {
        let err = params(Some("yesterday"), Some("2026-02-01T00:00:00Z"))
            .validate()
            .unwrap_err();
        assert_eq!(
            err,
            CostReportValidationError::InvalidStart {
                value: "yesterday".to_string()
            }
        );
        assert_eq!(err.parameter(), "start");
        assert!(err.to_string().contains("RFC 3339"));
    }

    #[test]
    fn validation_missing_end_is_rejected() {
        let err = params(Some("2026-01-01T00:00:00Z"), None)
            .validate()
            .unwrap_err();
        assert_eq!(err, CostReportValidationError::MissingEnd);
        assert_eq!(err.parameter(), "end");
    }

    #[test]
    fn validation_unparseable_end_is_rejected() {
        let err = params(Some("2026-01-01T00:00:00Z"), Some("not-a-time"))
            .validate()
            .unwrap_err();
        assert_eq!(
            err,
            CostReportValidationError::InvalidEnd {
                value: "not-a-time".to_string()
            }
        );
        assert_eq!(err.parameter(), "end");
    }

    #[test]
    fn validation_end_equal_to_start_is_rejected() {
        let err = params(
            Some("2026-01-01T00:00:00Z"),
            Some("2026-01-01T00:00:00Z"),
        )
        .validate()
        .unwrap_err();
        assert_eq!(err, CostReportValidationError::EndNotAfterStart);
        assert_eq!(err.parameter(), "end");
    }

    #[test]
    fn validation_end_before_start_is_rejected() {
        let err = params(Some("2026-02-01T00:00:00Z"), Some("2026-01-01T00:00:00Z"))
            .validate()
            .unwrap_err();
        assert_eq!(err, CostReportValidationError::EndNotAfterStart);
    }

    #[test]
    fn validation_range_over_366_days_is_rejected_and_names_limit() {
        let start = DateTime::parse_from_rfc3339("2026-03-01T00:00:00Z").unwrap();
        let too_far = start + chrono::Duration::days(MAX_RANGE_DAYS + 1);
        let err = params(
            Some("2026-03-01T00:00:00Z"),
            Some(&too_far.to_rfc3339()),
        )
        .validate()
        .unwrap_err();
        assert_eq!(err, CostReportValidationError::RangeTooLarge);
        assert_eq!(err.parameter(), "range");
        assert!(err.to_string().contains("366-day"), "{err}");
    }

    #[test]
    fn validation_range_of_exactly_366_days_is_allowed() {
        let start = DateTime::parse_from_rfc3339("2026-03-01T00:00:00Z").unwrap();
        let boundary = start + chrono::Duration::days(MAX_RANGE_DAYS);
        let validated = params(
            Some("2026-03-01T00:00:00Z"),
            Some(&boundary.to_rfc3339()),
        )
        .validate()
        .unwrap();
        assert_eq!(
            validated.range.end_secs - validated.range.start_secs,
            MAX_RANGE_DAYS * SECONDS_PER_DAY
        );
    }

    #[test]
    fn validation_unknown_group_by_is_rejected_with_allowed_values() {
        let mut p = params(Some("2026-01-01T00:00:00Z"), Some("2026-01-02T00:00:00Z"));
        p.group_by = Some("user".to_string());
        let err = p.validate().unwrap_err();
        assert_eq!(
            err,
            CostReportValidationError::InvalidGroupBy {
                value: "user".to_string()
            }
        );
        assert_eq!(err.parameter(), "group_by");
        let msg = err.to_string();
        assert!(msg.contains("group_by"), "{msg}");
        assert!(msg.contains(GroupBy::allowed_values()), "{msg}");
    }

    #[test]
    fn validation_unknown_format_is_rejected_with_allowed_values() {
        let mut p = params(Some("2026-01-01T00:00:00Z"), Some("2026-01-02T00:00:00Z"));
        p.format = Some("pdf".to_string());
        let err = p.validate().unwrap_err();
        assert_eq!(
            err,
            CostReportValidationError::InvalidFormat {
                value: "pdf".to_string()
            }
        );
        assert_eq!(err.parameter(), "format");
        let msg = err.to_string();
        assert!(msg.contains("format"), "{msg}");
        assert!(msg.contains(ReportFormat::allowed_values()), "{msg}");
    }

    #[test]
    fn validation_defaults_group_by_day_and_format_json() {
        let validated =
            params(Some("2026-01-01T00:00:00Z"), Some("2026-01-02T00:00:00Z"))
                .validate()
                .unwrap();
        assert_eq!(validated.group_by, GroupBy::Day);
        assert_eq!(validated.format, ReportFormat::Json);
        assert_eq!(validated.provider, None);
        assert_eq!(validated.key_id, None);
    }

    #[test]
    fn validation_parses_every_group_by_and_format_value_and_filters() {
        for (raw, expected) in [
            ("day", GroupBy::Day),
            ("provider", GroupBy::Provider),
            ("model", GroupBy::Model),
            ("virtual_key", GroupBy::VirtualKey),
        ] {
            assert_eq!(GroupBy::parse(raw), Some(expected));
        }
        assert_eq!(ReportFormat::parse("csv"), Some(ReportFormat::Csv));
        assert_eq!(ReportFormat::parse("json"), Some(ReportFormat::Json));

        let mut p = params(Some("2026-01-01T00:00:00Z"), Some("2026-01-02T00:00:00Z"));
        p.group_by = Some("virtual_key".to_string());
        p.format = Some("csv".to_string());
        p.provider = Some("openai".to_string());
        p.key_id = Some("key_42".to_string());
        let validated = p.validate().unwrap();
    assert_eq!(validated.group_by, GroupBy::VirtualKey);
    assert_eq!(validated.format, ReportFormat::Csv);
    assert_eq!(validated.provider.as_deref(), Some("openai"));
    assert_eq!(validated.key_id.as_deref(), Some("key_42"));
}

#[test]
fn validation_rejects_provider_filter_over_256_chars() {
    let mut p = params(Some("2026-01-01T00:00:00Z"), Some("2026-01-02T00:00:00Z"));
    p.provider = Some("x".repeat(MAX_FILTER_CHARS + 1));
    let err = p.validate().unwrap_err();
    assert_eq!(
        err,
        CostReportValidationError::FilterTooLong {
            parameter: "provider",
            len: MAX_FILTER_CHARS + 1,
        }
    );
    assert_eq!(err.parameter(), "provider");
    assert!(err.to_string().contains("provider"), "{err}");
    assert!(err.to_string().contains("256"), "{err}");
}

#[test]
fn validation_rejects_key_id_filter_over_256_chars() {
    let mut p = params(Some("2026-01-01T00:00:00Z"), Some("2026-01-02T00:00:00Z"));
    p.key_id = Some("k".repeat(10_000));
    let err = p.validate().unwrap_err();
    assert_eq!(err.parameter(), "key_id");
    assert!(matches!(
        err,
        CostReportValidationError::FilterTooLong {
            parameter: "key_id",
            ..
        }
    ));
}

#[test]
fn validation_accepts_filters_at_exactly_256_chars() {
    let mut p = params(Some("2026-01-01T00:00:00Z"), Some("2026-01-02T00:00:00Z"));
    p.provider = Some("p".repeat(MAX_FILTER_CHARS));
    p.key_id = Some("k".repeat(MAX_FILTER_CHARS));
    let validated = p.validate().unwrap();
    assert_eq!(validated.provider.as_deref().map(str::len), Some(MAX_FILTER_CHARS));
    assert_eq!(validated.key_id.as_deref().map(str::len), Some(MAX_FILTER_CHARS));
}

#[test]
fn validation_trims_filters_and_whitespace_only_becomes_none() {
    // A padded filter behaves exactly like its explicit form — " openai"
    // must not silently produce an empty report.
    let mut padded = params(Some("2026-01-01T00:00:00Z"), Some("2026-01-02T00:00:00Z"));
    padded.provider = Some(" openai ".to_string());
    padded.key_id = Some("\tkey_42\n".to_string());
    let mut explicit = params(Some("2026-01-01T00:00:00Z"), Some("2026-01-02T00:00:00Z"));
    explicit.provider = Some("openai".to_string());
    explicit.key_id = Some("key_42".to_string());
    assert_eq!(padded.validate().unwrap(), explicit.validate().unwrap());

    // Whitespace-only filters collapse to None (no filtering).
    let mut blank = params(Some("2026-01-01T00:00:00Z"), Some("2026-01-02T00:00:00Z"));
    blank.provider = Some("   ".to_string());
    blank.key_id = Some("\t \n".to_string());
    let validated = blank.validate().unwrap();
    assert_eq!(validated.provider, None);
    assert_eq!(validated.key_id, None);
}

    #[test]
    fn validation_parses_rfc3339_to_epoch_seconds() {
        let validated =
            params(Some("2026-01-01T00:00:00Z"), Some("2026-01-02T00:00:00Z"))
                .validate()
                .unwrap();
        assert_eq!(validated.range.start_secs, 1_767_225_600);
        assert_eq!(validated.range.end_secs, 1_767_312_000);
    }

    #[test]
    fn reconcile_params_validate_through_the_same_rules() {
        assert_eq!(
            ReconcileParams::default().validate().unwrap_err(),
            CostReportValidationError::MissingStart
        );
        let ok = ReconcileParams {
            start: Some("2026-01-01T00:00:00Z".to_string()),
            end: Some("2026-01-02T00:00:00Z".to_string()),
        }
        .validate()
        .unwrap();
        assert_eq!(ok.end_secs - ok.start_secs, SECONDS_PER_DAY);
    }

    #[test]
    fn micro_usd_string_has_exactly_six_decimals() {
        assert_eq!(micro_usd_to_usd_string(0), "0.000000");
        assert_eq!(micro_usd_to_usd_string(1), "0.000001");
        assert_eq!(micro_usd_to_usd_string(999_999), "0.999999");
        assert_eq!(micro_usd_to_usd_string(1_000_000), "1.000000");
        assert_eq!(micro_usd_to_usd_string(1_234_567), "1.234567");
        assert_eq!(micro_usd_to_usd_string(12_345_678_912_345), "12345678.912345");
    }

    #[test]
    fn micro_usd_string_handles_max_i64_and_negatives() {
        assert_eq!(
            micro_usd_to_usd_string(i64::MAX),
            "9223372036854.775807"
        );
        assert_eq!(
            micro_usd_to_usd_string(i64::MIN),
            "-9223372036854.775808"
        );
        assert_eq!(micro_usd_to_usd_string(-1), "-0.000001");
        assert_eq!(micro_usd_to_usd_string(-1_500_000), "-1.500000");
    }

#[test]
fn micro_usd_to_usd_is_a_single_deterministic_conversion() {
    assert_eq!(micro_usd_to_usd(0), 0.0);
    assert_eq!(micro_usd_to_usd(1), 0.000001);
    assert_eq!(micro_usd_to_usd(1_000_000), 1.0);
    assert_eq!(micro_usd_to_usd(1_234_567), 1.234567);
    assert_eq!(micro_usd_to_usd(i64::MAX), 9_223_372_036_854.775_4);
    assert_eq!(micro_usd_to_usd(42), 42.0 / 1_000_000.0);
    assert_eq!(micro_usd_to_usd(7_777_777), 7.777777);
    assert_eq!(micro_usd_to_usd(123_456_789), 123.456789);
}

    #[test]
    fn cost_descending_order_breaks_ties_by_dimension_ascending() {
        let mut rows = vec![
            row("beta", 500),
            row("alpha", 500),
            row("gamma", 9_000),
            row("delta", 0),
            row("zeta", 500),
        ];
        sort_rows_by_cost_desc(&mut rows);
        let dims: Vec<&str> = rows.iter().map(|r| r.dimension.as_str()).collect();
        assert_eq!(dims, vec!["gamma", "alpha", "beta", "zeta", "delta"]);
    }

    #[test]
    fn equal_cost_rows_fully_ordered_by_dimension() {
        let mut rows = vec![row("c", 7), row("a", 7), row("b", 7)];
        sort_rows_by_cost_desc(&mut rows);
        let dims: Vec<&str> = rows.iter().map(|r| r.dimension.as_str()).collect();
        assert_eq!(dims, vec!["a", "b", "c"]);
    }

    #[test]
    fn day_grouping_keeps_calendar_order() {
        let mut rows = vec![
            row("2026-01-03", 3),
            row("2026-01-01", 1),
            row("2026-01-02", 2),
        ];
        order_rows(GroupBy::Day, &mut rows);
        let dims: Vec<&str> = rows.iter().map(|r| r.dimension.as_str()).collect();
        assert_eq!(dims, vec!["2026-01-01", "2026-01-02", "2026-01-03"]);
    }

    #[test]
    fn report_totals_sum_integer_micro_usd_across_rows() {
        let rows = vec![
            row("a", 1_500_000),
            row("b", 250_000),
            row("c", 1),
        ];
        let report = CostReport::new(GroupBy::Provider, rows, 100, 200);
        assert_eq!(report.totals.cost_micro_usd, 1_750_001);
        assert_eq!(report.totals.cost_usd_string(), "1.750001");
        assert_eq!(report.totals.requests, 3);
        assert_eq!(report.totals.input_tokens, 30);
        assert_eq!(report.totals.output_tokens, 60);
        assert_eq!(report.totals.window_start, 100);
        assert_eq!(report.totals.window_end, 200);
        let dims: Vec<&str> =
            report.rows.iter().map(|r| r.dimension.as_str()).collect();
        assert_eq!(dims, vec!["a", "b", "c"]);
    }

    #[test]
    fn report_rows_serialize_cost_usd_from_micro() {
        let rows = vec![row("a", 1_234_567)];
        let report = CostReport::new(GroupBy::Provider, rows, 0, 1);
        let json = serde_json::to_value(&report).unwrap();
        assert!(json.get("cost_micro_usd").is_none());
        assert_eq!(json["rows"][0]["cost_usd"], serde_json::json!(1.234567));
        assert_eq!(json["totals"]["cost_usd"], serde_json::json!(1.234567));
        assert!(json["rows"][0]["dimension"].is_string());
    }

#[test]
fn reconciliation_difference_is_ledger_minus_provider_spend() {
    let report = ReconciliationReport::new(
        LedgerTotals {
            requests: 10,
            cost_micro_usd: 2_000_000,
        },
        ProviderSpendTotals {
            cost_micro_usd: 1_750_000,
            window_start: 0,
            window_end: 86_400,
            caveat: true,
        },
        KeyUsageTotals {
            cost_micro_usd: 1_900_000,
        },
        2,
    );
        assert_eq!(report.difference_micro_usd, 250_000);
        assert_eq!(micro_usd_to_usd_string(report.difference_micro_usd), "0.250000");
        let json = serde_json::to_value(&report).unwrap();
        assert_eq!(json["ledger"]["cost_usd"], serde_json::json!(2.0));
        assert_eq!(json["provider_spend"]["cost_usd"], serde_json::json!(1.75));
        assert_eq!(json["key_usage"]["cost_usd"], serde_json::json!(1.9));
        assert_eq!(json["difference_usd"], serde_json::json!(0.25));
        assert_eq!(json["provider_spend"]["caveat"], serde_json::json!(true));
        assert_eq!(json["unpriced_requests"], serde_json::json!(2));
    }
}
