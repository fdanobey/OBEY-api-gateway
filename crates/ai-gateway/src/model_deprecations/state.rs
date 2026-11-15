//! Episode state machine for model-deprecation evaluation
//! (model-deprecation-alerts spec).
//!
//! This module owns the per-`(provider, model)` deprecation registry
//! and decides, from merged scan data, operator overrides, runtime
//! detections, and injected usage evidence, whether a
//! [`ModelDeprecatedEventData`] event opens. It is pure with respect
//! to I/O — no config, metrics, or async runtime dependencies — the
//! clock and the usage lookup are injected, so the entire episode
//! decision logic is unit-testable and property-testable in
//! isolation (the `webhooks/alerts.rs` discipline).
//!
//! Episode semantics (Reqs 3.1, 3.2, 4.1, 4.4):
//! - a model enters a deprecation episode when it qualifies (known
//! retirement within `warn_days` — inclusive at the boundary — or
//! already past, or date-less `deprecated: true`, or runtime-gone)
//! AND the 7-day usage gate passes (`requests > 0`); usage `0` or a
//! failed ledger lookup (`None` from the usage closure) suppresses
//! emission WITHOUT opening the episode, so a later gate pass still
//! alerts (a broken ledger must not become alert spam, nor may it
//! swallow the episode forever);
//! - exactly one event per episode; a warn→gone escalation ends the
//! warn episode and opens a gone episode (the stronger event emits
//! once);
//! - the episode ends (re-arm) when the condition clears (override
//! removed, model dropped from a successful scan, runtime-present
//! recorded, gone mark past its 24-hour bound) so a later
//! re-qualification emits a fresh event;
//! - operator overrides (`deprecation_date`/`replacement_model`)
//! take precedence ABOVE scan-sourced metadata for the same key
//! (Req 2.1) — including pushing a model back OUT of the warn
//! window.
//!
//! Merge discipline (Req 1.3): `record_scan` REPLACES the provider's
//! catalog view — models absent from a successful scan lose their
//! signal (episode re-arms) — while providers not mentioned in the
//! call keep their previous data (the scanner stale-keeps failed
//! providers by simply not calling). `remove_provider` /
//! `retain_providers` drop whole providers on config reload.
//!
//! Registry is in-memory only (design "Episode key"): DashMap with
//! episode entries allocated only when an event actually fires,
//! closed episodes removed on the next evaluation, and expired
//! runtime-gone marks evicted on touch or via [`Self::evict_expired`]
//! — the same bounded-registry shape as `webhooks/alerts.rs`.
//!
//! _Requirements: 1.4, 2.1, 3.1, 3.2, 4.1, 4.4, 5.3_

use std::collections::{BTreeSet, HashMap, HashSet};

use chrono::{DateTime, NaiveDate, SecondsFormat, Utc};
use dashmap::DashMap;
use dashmap::mapref::entry::Entry;

use crate::model_deprecations::events::{EvidenceSource, ModelDeprecatedEventData};

/// Registry key: the exact `(provider, model)` pair every signal,
/// episode, and usage query is anchored to.
type RegistryKey = (String, String);

/// Worst-case bound on a runtime-gone false positive: an unconfirmed
/// mark stops counting (and is evicted) 24 hours after detection
/// (Req 5.2 — reload, a re-arming scan, or this bound, whichever
/// comes first).
const RUNTIME_GONE_BOUND_HOURS: i64 = 24;

/// One model's deprecation signal from a successful provider scan
/// (design "Component 1" handoff). The scanner parses the retirement
/// date (precedence `shutdown_date` > `expiration_date` >
/// `deprecation_date`); `deprecated: true` with no date opens a
/// date-less episode.
#[derive(Debug, Clone, PartialEq)]
pub struct ScanModel {
    /// Deprecated provider model id.
    pub model: String,
    /// Parsed retirement date; `None` when the listing carries no
    /// date field.
    pub retirement_date: Option<NaiveDate>,
    /// Scan-sourced replacement hint (`successor_model`).
    pub replacement: Option<String>,
    /// Date-less `deprecated: true` flag from provider metadata.
    pub deprecated: bool,
}

/// Operator override for one configured `(provider, model)` pair,
/// projected from `model_groups[].models[]` config by the caller
/// (config types never leak into this module). `deprecation_date` /
/// `replacement_model` take precedence ABOVE scan data (Req 2.1);
/// `model_groups` names the configured groups referencing the pair
/// (status rows only, design "Component 5").
#[derive(Debug, Clone, PartialEq)]
pub struct OverrideSpec {
    pub provider: String,
    pub model: String,
    /// ISO 8601 date or datetime string (config-validated).
    pub deprecation_date: Option<String>,
    pub replacement_model: Option<String>,
    /// Configured model groups referencing this provider model.
    pub model_groups: Vec<String>,
}

/// One row of the admin deprecation status view (design
/// "DeprecationStatusRow"; `requests_7d` is `0` when the usage
/// ledger lookup failed — the row stays, the ledger-failure marker
/// is the caller's to render).
#[derive(Debug, Clone, PartialEq, serde::Serialize, serde::Deserialize)]
pub struct StatusRow {
    pub provider: String,
    pub model: String,
    pub retirement_date: Option<String>,
    pub days_remaining: Option<i64>,
    pub replacement_model: Option<String>,
    pub evidence_source: EvidenceSource,
    pub requests_7d: u64,
    pub last_used_at: Option<String>,
    pub runtime_gone: bool,
    /// Configured model groups referencing this provider model.
    pub model_groups: Vec<String>,
    /// Whether a deprecation episode is currently open (one event
    /// already emitted; re-armed when the signal clears).
    pub episode_open: bool,
}

/// Scan-sourced deprecation signal stored per `(provider, model)`.
#[derive(Debug, Clone, Default)]
struct ScanEntry {
    retirement_date: Option<NaiveDate>,
    replacement: Option<String>,
    deprecated: bool,
}

/// A runtime-detection mark: `marked_at` records the detection time
/// (Req 4.1) and `gone_until` enforces the 24-hour unconfirmed bound
/// (Req 5.2).
#[derive(Debug, Clone)]
struct GoneMark {
    marked_at: DateTime<Utc>,
    gone_until: DateTime<Utc>,
}

/// Which level an open deprecation episode was announced at. `Gone`
/// is the stronger signal: a warn→gone transition escalates (emits
/// once); a gone episode never downgrades while any signal persists.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum EpisodeLevel {
    /// Retiring within the warn window, already past, or date-less
    /// `deprecated: true`.
    Warn,
    /// Runtime-detected gone (404/410 model-not-found signature).
    Gone,
}

/// Effective deprecation signal for one key after merging overrides
/// (highest), scan data, and runtime detection.
#[derive(Debug, Clone)]
struct EffectiveSignal {
    retirement: Option<NaiveDate>,
    replacement: Option<String>,
    deprecated_flag: bool,
    gone_active: bool,
    evidence: EvidenceSource,
}

/// In-memory deprecation registry + episode evaluation (design
/// "Component 2").
///
/// All episode transitions go through the DashMap entry API — the
/// check-and-set happens atomically under the shard lock — so two
/// concurrent evaluations qualifying the same model produce exactly
/// one event (the `webhooks/alerts.rs` atomicity discipline).
///
/// Registry shape: episode entries are allocated only when an event
/// actually fires, removed when the signal clears (re-arm), and
/// runtime-gone marks expire after 24 hours — so the maps stay
/// bounded by live signals (plus explicit `remove_provider` /
/// `retain_providers` on config reload).
#[derive(Debug, Default)]
pub struct DeprecationState {
    scan: DashMap<RegistryKey, ScanEntry>,
    // Nested per-provider buckets so the router hot-path probes
    // (`is_gone` / `is_gone_at` / `gone_since`) run with borrowed
    // `&str` keys (std's `String: Borrow<str>`) — a flat
    // `(String, String)` key allocates two Strings per lookup.
    // Lock order is always outer→inner; the inner map is a
    // distinct DashMap instance, so no same-map re-entrancy.
    runtime_gone: DashMap<String, DashMap<String, GoneMark>>,
    episodes: DashMap<RegistryKey, EpisodeLevel>,
}

impl DeprecationState {
    /// Merge one provider's successful scan result (design
    /// "Component 2"). Replaces the provider's catalog view —
    /// models absent from a successful scan lose their signal —
    /// while providers not mentioned keep their previous data.
    /// Models observed in the fresh listing have any runtime-gone
    /// mark cleared (Req 5.2 re-arm).
pub fn record_scan(&self, provider: &str, models: Vec<ScanModel>) {
self.scan.retain(|(p, _), _| p != provider);
for model in models {
let key = (provider.to_string(), model.model.clone());
self.scan.insert(
key.clone(),
ScanEntry {
retirement_date: model.retirement_date,
replacement: model.replacement,
deprecated: model.deprecated,
},
);
// The fresh listing observes the model present: re-arm
// any runtime-gone mark (Req 5.2).
if let Some(marks) = self.runtime_gone.get(provider) {
marks.remove(model.model.as_str());
}
}
}

/// Drop every trace of a provider (scan data, runtime-gone
/// marks, open episodes) — config-reload path for a removed
/// provider (Req 1.4).
pub fn remove_provider(&self, provider: &str) {
self.scan.retain(|(p, _), _| p != provider);
self.runtime_gone.retain(|p, _| p.as_str() != provider);
self.episodes.retain(|(p, _), _| p != provider);
}

/// Retain only the named providers across all three registries
/// — config-reload sweep (Req 1.4).
pub fn retain_providers(&self, keep: &[String]) {
let keep: HashSet<&String> = keep.iter().collect();
self.scan.retain(|(p, _), _| keep.contains(p));
self.runtime_gone.retain(|p, _| keep.contains(p));
self.episodes.retain(|(p, _), _| keep.contains(p));
}

    /// Record a runtime model-not-found detection (404/410
    /// signature, Req 4.1): stamps the detection time and arms the
    /// 24-hour unconfirmed bound.
    pub fn record_runtime_gone(&self, provider: &str, model: &str, now: DateTime<Utc>) {
        self.runtime_gone
            .entry(provider.to_string())
            .or_default()
            .insert(
                model.to_string(),
                GoneMark {
                    marked_at: now,
                    gone_until: now + chrono::Duration::hours(RUNTIME_GONE_BOUND_HOURS),
                },
            );
    }

    /// Record the model serving again (successful request or
    /// re-arming scan): clears the runtime-gone mark so the episode
    /// can re-arm (Req 4.4). Cheap no-op when no mark exists.
    pub fn record_runtime_present(&self, provider: &str, model: &str) {
        if let Some(models) = self.runtime_gone.get(provider) {
            models.remove(model);
        }
    }

    /// Whether a runtime-gone mark is present (unbounded view for
    /// the router's candidate skip — reload, a re-arming scan, or
    /// [`Self::evict_expired`]/[`Self::is_gone_at`] enforce the
    /// 24-hour bound).
    pub fn is_gone(&self, provider: &str, model: &str) -> bool {
        self.runtime_gone
            .get(provider)
            .is_some_and(|models| models.contains_key(model))
    }

/// Whether a runtime-gone mark is present AND still inside the
/// 24-hour unconfirmed bound at `now` (Req 5.2).
pub fn is_gone_at(&self, provider: &str, model: &str, now: DateTime<Utc>) -> bool {
let models = match self.runtime_gone.get(provider) {
Some(models) => models,
None => return false,
};
let result = match models.get(model) {
Some(mark) => now < mark.gone_until,
None => false,
};
result
}

/// Detection time of the runtime-gone mark (Req 4.1 "record the
/// detection time"); `None` when no mark exists.
pub fn gone_since(&self, provider: &str, model: &str) -> Option<DateTime<Utc>> {
let models = self.runtime_gone.get(provider)?;
let mark = models.get(model)?;
let stamp = mark.marked_at;
Some(stamp)
}

    /// Reload sweep for still-configured providers (Req 5.2: the
    /// skip lasts until reload/re-arm/24h — whichever comes first):
    /// clears ONLY runtime-gone marks, leaving scan data and open
    /// episodes untouched (a reload must not re-arm alerting on
    /// models that still qualify).
    pub fn clear_runtime_gone(&self, provider: &str) {
        if let Some(models) = self.runtime_gone.get(provider) {
            models.clear();
        }
    }

    /// Sweep every runtime-gone mark past its 24-hour bound (Req
    /// 5.2); returns how many were dropped. `evaluate` also evicts
    /// expired marks it touches; this is the scanner-cadence full
    /// pass that keeps the registry bounded.
pub fn evict_expired(&self, now: DateTime<Utc>) -> usize {
let count_marks =
|map: &DashMap<String, DashMap<String, GoneMark>>| -> usize {
map.iter().map(|models| models.value().len()).sum()
};
let before = count_marks(&self.runtime_gone);
self.runtime_gone.retain(|_, models| {
models.retain(|_, mark| now < mark.gone_until);
!models.is_empty()
});
before - count_marks(&self.runtime_gone)
}

    /// Pure episode evaluation (design "Component 2"): returns the
    /// `model.deprecated` event payloads to emit, sorted by
    /// `(provider, model)`. Operator `overrides` apply ABOVE scan
    /// data; `usage_lookup(provider, model) -> None` (ledger
    /// failure) or `Some((0, _))` suppresses emission for that key
    /// WITHOUT opening the episode; `warn_days` is the resolved
    /// `warn_days_before`; `now` is the injected clock.
    pub fn evaluate<F>(
        &self,
        overrides: &[OverrideSpec],
        usage_lookup: F,
        warn_days: u32,
        now: DateTime<Utc>,
    ) -> Vec<ModelDeprecatedEventData>
    where
        F: Fn(&str, &str) -> Option<(u64, Option<DateTime<Utc>>)>,
    {
        let override_map: HashMap<RegistryKey, &OverrideSpec> = overrides
            .iter()
            .map(|entry| ((entry.provider.clone(), entry.model.clone()), entry))
            .collect();
        let keys = self.candidate_keys(&override_map);

        // Re-arm sweep: an episode whose key lost every signal
        // source (dropped from scan, gone mark cleared, override
        // removed) is not in the candidate set — close it here. Keys
        // still carrying a signal are re-examined per key below (the
        // gone-expiry and override-pullback paths clear through the
        // per-key branch). Episodes are bounded by fired events, so
        // this retain is cheap.
        self.episodes.retain(|key, _| keys.contains(key));

        let mut events = Vec::new();

for key in keys {
// Expired gone marks are evicted on touch (Req 5.2 bound
// holds even without the scanner-cadence sweep).
let gone_active = {
let bucket = self.runtime_gone.get(&key.0);
let mark = bucket.as_ref().and_then(|models| models.get(&key.1));
let expired = matches!(&mark, Some(mark) if now >= mark.gone_until);
let present = mark.is_some();
drop(mark);
drop(bucket);
if expired {
if let Some(models) = self.runtime_gone.get(&key.0) {
models.remove(key.1.as_str());
}
}
!expired && present
};
            let scan = self.scan.get(&key).map(|entry| entry.clone());
            let signal =
                resolve_signal(scan.as_ref(), gone_active, override_map.get(&key).copied());

            let level = match qualification(&signal, warn_days, now) {
                Some(level) => level,
                None => {
                    // Signal cleared: close (re-arm) the episode
                    // unconditionally — closure is not emission and
                    // must not depend on the usage gate.
                    self.episodes.remove(&key);
                    continue;
                }
            };

            // Skip the ledger lookup when no transition can emit:
            // the episode is already open at this level, or a gone
            // episode is terminal (no downgrade while any signal
            // persists). The entry match below re-checks under the
            // shard lock, so the pre-check is purely a lookup saver.
            let open = self.episodes.get(&key).map(|episode| *episode);
            if open == Some(EpisodeLevel::Gone)
                || (open == Some(EpisodeLevel::Warn) && level == EpisodeLevel::Warn)
            {
                continue;
            }

            // Usage gate (Reqs 3.1, 3.2): only in-use models alert;
            // a failed ledger lookup suppresses without opening the
            // episode so a recovered lookup still alerts.
            let (requests_7d, last_used) = match usage_lookup(&key.0, &key.1) {
                Some(usage) => usage,
                None => continue,
            };
            if requests_7d == 0 {
                continue;
            }

            match self.episodes.entry(key.clone()) {
                Entry::Vacant(vacant) => {
                    vacant.insert(level);
                    events.push(Self::event_data(
                        &key, &signal, level, requests_7d, last_used, now,
                    ));
                }
                Entry::Occupied(mut occupied) => {
                    // Race re-check under the shard lock: the only
                    // remaining transition is warn→gone escalation
                    // (emits once, Req 4.4).
                    if *occupied.get() == EpisodeLevel::Warn && level == EpisodeLevel::Gone {
                        occupied.insert(EpisodeLevel::Gone);
                        events.push(Self::event_data(
                            &key, &signal, level, requests_7d, last_used, now,
                        ));
                    }
                }
            }
        }
        events
    }

    /// Admin status rows (design "Component 5"): every key with a
    /// current deprecation signal — qualifying for an episode at
    /// `warn_days` (the admin view shows what alerting acts on) or
    /// carrying an active runtime-gone mark — sorted by `(provider,
    /// model)`. `requests_7d` is `0` when the usage lookup fails.
    /// Read-only: never opens or closes episodes.
    pub fn status_rows<F>(
        &self,
        overrides: &[OverrideSpec],
        usage_lookup: F,
        warn_days: u32,
        now: DateTime<Utc>,
    ) -> Vec<StatusRow>
    where
        F: Fn(&str, &str) -> Option<(u64, Option<DateTime<Utc>>)>,
    {
        let override_map: HashMap<RegistryKey, &OverrideSpec> = overrides
            .iter()
            .map(|entry| ((entry.provider.clone(), entry.model.clone()), entry))
            .collect();
        let keys = self.candidate_keys(&override_map);
        let mut rows = Vec::new();

        for key in keys {
            let gone_active = self.is_gone_at(&key.0, &key.1, now);
            let scan = self.scan.get(&key).map(|entry| entry.clone());
            let over = override_map.get(&key).copied();
            let signal = resolve_signal(scan.as_ref(), gone_active, over);

            if qualification(&signal, warn_days, now).is_none() {
                continue;
            }
            let (requests_7d, last_used) = usage_lookup(&key.0, &key.1).unwrap_or((0, None));
            rows.push(StatusRow {
                provider: key.0.clone(),
                model: key.1.clone(),
                retirement_date: signal.retirement.map(|date| date.to_string()),
                days_remaining: signal.retirement.map(|date| days_remaining(date, now)),
                replacement_model: signal.replacement.clone(),
                evidence_source: signal.evidence,
                requests_7d,
                last_used_at: last_used
                    .map(|stamp| stamp.to_rfc3339_opts(SecondsFormat::Secs, true)),
                runtime_gone: signal.gone_active,
                model_groups: over
                    .map(|entry| entry.model_groups.clone())
                    .unwrap_or_default(),
                episode_open: self.episodes.contains_key(&key),
            });
        }
        rows
    }

    /// Sorted union of every key any signal could come from: scan
    /// entries, runtime-gone marks, and configured overrides.
fn candidate_keys<'a>(
&self,
override_map: &HashMap<RegistryKey, &'a OverrideSpec>,
) -> BTreeSet<RegistryKey> {
let mut keys = BTreeSet::new();
for entry in self.scan.iter() {
keys.insert(entry.key().clone());
}
for provider_bucket in self.runtime_gone.iter() {
for model in provider_bucket.value().iter() {
keys.insert((provider_bucket.key().clone(), model.key().clone()));
}
}
for key in override_map.keys() {
keys.insert(key.clone());
}
keys
}

    /// Build the emitted payload. Escalation (`Gone`) always
    /// reports `runtime_detection` evidence (the stronger signal,
    /// Req 4.4) while still carrying the merged retirement/replacement
    /// hints from override or scan data.
    fn event_data(
        key: &RegistryKey,
        signal: &EffectiveSignal,
        level: EpisodeLevel,
        requests_7d: u64,
        last_used: Option<DateTime<Utc>>,
        now: DateTime<Utc>,
    ) -> ModelDeprecatedEventData {
        ModelDeprecatedEventData {
            provider: key.0.clone(),
            model: key.1.clone(),
            retirement_date: signal.retirement.map(|date| date.to_string()),
            days_remaining: signal.retirement.map(|date| days_remaining(date, now)),
            replacement_model: signal.replacement.clone(),
            evidence_source: if level == EpisodeLevel::Gone {
                EvidenceSource::RuntimeDetection
            } else {
                signal.evidence
            },
            requests_7d,
            last_used_at: last_used
                .map(|stamp| stamp.to_rfc3339_opts(SecondsFormat::Secs, true)),
            runtime_gone: level == EpisodeLevel::Gone,
        }
    }
}

/// Merge one key's signals: override date/replacement above scan
/// data (Req 2.1), runtime-gone strongest of all. An unparseable
/// override date (config validation prevents it) falls back to the
/// scan signal rather than panicking.
fn resolve_signal(
    scan: Option<&ScanEntry>,
    gone_active: bool,
    over: Option<&OverrideSpec>,
) -> EffectiveSignal {
    let override_date = over
        .and_then(|entry| entry.deprecation_date.as_deref())
        .and_then(parse_iso_date);
    let retirement = override_date.or_else(|| scan.and_then(|entry| entry.retirement_date));
    let replacement = over
        .and_then(|entry| entry.replacement_model.clone())
        .or_else(|| scan.and_then(|entry| entry.replacement.clone()));
    let deprecated_flag = scan.map(|entry| entry.deprecated).unwrap_or(false);
    let evidence = if gone_active {
        EvidenceSource::RuntimeDetection
    } else if override_date.is_some() {
        EvidenceSource::OperatorOverride
    } else {
        EvidenceSource::ProviderMetadata
    };
    EffectiveSignal {
        retirement,
        replacement,
        deprecated_flag,
        gone_active,
        evidence,
    }
}

/// Episode qualification at `now`: runtime-gone escalates to
/// `Gone`; a known retirement within `warn_days` (inclusive at the
/// boundary — already-past dates carry negative days and always
/// qualify); a date-less `deprecated: true` flag opens a date-less
/// warn episode (Req 3.1).
fn qualification(
    signal: &EffectiveSignal,
    warn_days: u32,
    now: DateTime<Utc>,
) -> Option<EpisodeLevel> {
    if signal.gone_active {
        return Some(EpisodeLevel::Gone);
    }
    if let Some(retirement) = signal.retirement {
        if days_remaining(retirement, now) <= i64::from(warn_days) {
            return Some(EpisodeLevel::Warn);
        }
        return None;
    }
    if signal.deprecated_flag {
        return Some(EpisodeLevel::Warn);
    }
    None
}

/// Whole days from `now` to `retirement` (negative once past) —
/// the `days_remaining` payload field.
fn days_remaining(retirement: NaiveDate, now: DateTime<Utc>) -> i64 {
    (retirement - now.date_naive()).num_days()
}

/// ISO 8601 calendar date (`YYYY-MM-DD`) or full RFC 3339 datetime,
/// mirroring the config validator's accepted shapes (defensive —
/// config validation guarantees parseable override dates).
fn parse_iso_date(value: &str) -> Option<NaiveDate> {
    if let Ok(date) = NaiveDate::parse_from_str(value, "%Y-%m-%d") {
        return Some(date);
    }
    chrono::DateTime::parse_from_rfc3339(value)
        .ok()
        .map(|datetime| datetime.date_naive())
}

#[cfg(test)]
mod tests {
    use super::*;
    use chrono::TimeZone;

    fn now() -> DateTime<Utc> {
        Utc.with_ymd_and_hms(2026, 11, 14, 12, 0, 0).unwrap()
    }

    fn later(hours: i64) -> DateTime<Utc> {
        now() + chrono::Duration::hours(hours)
    }

    fn retirement_in(days: i64) -> Option<NaiveDate> {
        Some(now().date_naive() + chrono::Duration::days(days))
    }

    fn scan_model(
        model: &str,
        retirement_date: Option<NaiveDate>,
        replacement: Option<&str>,
        deprecated: bool,
    ) -> ScanModel {
        ScanModel {
            model: model.to_string(),
            retirement_date,
            replacement: replacement.map(|r| r.to_string()),
            deprecated,
        }
    }

    fn retiring_soon(model: &str, days: i64) -> ScanModel {
        scan_model(model, retirement_in(days), None, false)
    }

    fn override_for(
        provider: &str,
        model: &str,
        deprecation_date: Option<&str>,
        replacement_model: Option<&str>,
    ) -> OverrideSpec {
        OverrideSpec {
            provider: provider.to_string(),
            model: model.to_string(),
            deprecation_date: deprecation_date.map(|d| d.to_string()),
            replacement_model: replacement_model.map(|r| r.to_string()),
            model_groups: Vec::new(),
        }
    }

    fn usage(count: u64) -> impl Fn(&str, &str) -> Option<(u64, Option<DateTime<Utc>>)> {
        move |_, _| Some((count, None))
    }

    fn no_usage() -> impl Fn(&str, &str) -> Option<(u64, Option<DateTime<Utc>>)> {
        move |_, _| None
    }

    // === Day-boundary matrix ==============================================
    //
    // Exactly warn_days days remaining fires (inclusive boundary);
    // one day beyond the window stays silent; today and already-past
    // dates always qualify (Req 4.1 "already past").

    // Validates: Requirement 4.1 (warn window boundary)
    #[test]
    fn boundary_exact_warn_days_remaining_emits_once() {
        let state = DeprecationState::default();
        state.record_scan("openai", vec![retiring_soon("gpt-4-turbo", 30)]);
        let events = state.evaluate(&[], usage(10), 30, now());
        assert_eq!(events.len(), 1);
        assert_eq!(events[0].days_remaining, Some(30));
        assert_eq!(
            events[0].retirement_date.as_deref(),
            retirement_in(30).map(|d| d.to_string()).as_deref()
        );
    }

    // Validates: Requirement 4.1 (warn window boundary)
    #[test]
    fn boundary_one_day_beyond_warn_window_silent() {
        let state = DeprecationState::default();
        state.record_scan("openai", vec![retiring_soon("gpt-4-turbo", 31)]);
        assert!(state.evaluate(&[], usage(10), 30, now()).is_empty());
        // No fire means no episode state allocated (alerts.rs
        // no-allocation-on-no-fire discipline).
        assert_eq!(state.episodes.len(), 0);
    }

    // Validates: Requirement 4.1 (already-past retirement)
    #[test]
    fn boundary_today_and_past_retirement_dates_emit() {
        let state = DeprecationState::default();
        state.record_scan(
            "openai",
            vec![retiring_soon("today", 0), retiring_soon("yesterday", -1)],
        );
        let events = state.evaluate(&[], usage(10), 30, now());
        assert_eq!(events.len(), 2);
        assert_eq!(events[0].model, "today");
        assert_eq!(events[0].days_remaining, Some(0));
        assert_eq!(events[1].model, "yesterday");
        assert_eq!(events[1].days_remaining, Some(-1));
    }

    // === Date-less deprecated flag ========================================

    // Validates: Requirement 3.1 (date-less deprecated episode)
    #[test]
    fn date_less_deprecated_flag_opens_episode_with_null_date() {
        let state = DeprecationState::default();
        state.record_scan(
            "cloudflare",
            vec![scan_model("@cf/old", None, Some("@cf/new"), true)],
        );
        let events = state.evaluate(&[], usage(3), 30, now());
        assert_eq!(events.len(), 1);
        assert_eq!(events[0].retirement_date, None);
        assert_eq!(events[0].days_remaining, None);
        assert_eq!(events[0].replacement_model.as_deref(), Some("@cf/new"));
        assert_eq!(events[0].evidence_source, EvidenceSource::ProviderMetadata);
        assert!(!events[0].runtime_gone);
    }

    // === Override precedence ==============================================

    // Validates: Requirement 2.1 (override above scan data)
    #[test]
    fn override_date_takes_precedence_over_scan_data() {
        // Scan says 60d out (silent at warn 30); override pulls it
        // to 10d out — fires with operator evidence and the
        // override's date.
        let state = DeprecationState::default();
        state.record_scan("openai", vec![retiring_soon("gpt-4-turbo", 60)]);
        let events = state.evaluate(
            &[override_for("openai", "gpt-4-turbo", Some("2026-11-24"), None)],
            usage(10),
            30,
            now(),
        );
        assert_eq!(events.len(), 1);
        assert_eq!(events[0].retirement_date.as_deref(), Some("2026-11-24"));
        assert_eq!(events[0].days_remaining, Some(10));
        assert_eq!(events[0].evidence_source, EvidenceSource::OperatorOverride);

        // Override pushes a scan-retiring model back OUT of the
        // window: precedence works both directions.
        let state2 = DeprecationState::default();
        state2.record_scan("openai", vec![retiring_soon("gpt-4-turbo", 5)]);
        assert!(state2
            .evaluate(
                &[override_for("openai", "gpt-4-turbo", Some("2027-01-15"), None)],
                usage(10),
                30,
                now(),
            )
            .is_empty());
    }

    // Validates: Requirement 2.1 (override replacement above scan)
    #[test]
    fn override_replacement_wins_over_scan_successor() {
        let state = DeprecationState::default();
        state.record_scan(
            "openai",
            vec![scan_model(
                "gpt-4-turbo",
                retirement_in(10),
                Some("scan-successor"),
                false,
            )],
        );
        let events = state.evaluate(
            &[override_for("openai", "gpt-4-turbo", None, Some("op-successor"))],
            usage(10),
            30,
            now(),
        );
        assert_eq!(events.len(), 1);
        assert_eq!(events[0].replacement_model.as_deref(), Some("op-successor"));
        // Replacement-only override does not change the evidence:
        // the deprecation signal still comes from the provider.
        assert_eq!(events[0].evidence_source, EvidenceSource::ProviderMetadata);

        // Scan had no successor: the override hint stands alone.
        let state2 = DeprecationState::default();
        state2.record_scan("openai", vec![retiring_soon("gpt-4-turbo", 10)]);
        let events2 = state2.evaluate(
            &[override_for("openai", "gpt-4-turbo", None, Some("op-successor"))],
            usage(10),
            30,
            now(),
        );
        assert_eq!(events2[0].replacement_model.as_deref(), Some("op-successor"));
    }

    // === Usage gate =======================================================

    // Validates: Requirements 3.1, 3.2 (usage gate)
    #[test]
    fn usage_gate_zero_suppresses_without_opening_episode() {
        let state = DeprecationState::default();
        state.record_scan("openai", vec![retiring_soon("gpt-4-turbo", 5)]);
        // Usage 0: no event, and the episode is NOT opened — the
        // unused model still shows in status rows (Req 3.2) but a
        // later gate pass must still alert.
        assert!(state.evaluate(&[], usage(0), 30, now()).is_empty());
        assert_eq!(state.episodes.len(), 0);
        let later_pass = state.evaluate(&[], usage(4), 30, now());
        assert_eq!(later_pass.len(), 1);
        assert_eq!(later_pass[0].requests_7d, 4);
    }

    // Validates: Requirements 3.1, 3.2 (ledger failure suppresses)
    #[test]
    fn usage_gate_ledger_failure_suppresses_without_opening_episode() {
        let state = DeprecationState::default();
        state.record_scan("openai", vec![retiring_soon("gpt-4-turbo", 5)]);
        // Lookup None (ledger failure): suppressed, episode not
        // opened — a recovered ledger read still emits.
        assert!(state.evaluate(&[], no_usage(), 30, now()).is_empty());
        assert_eq!(state.episodes.len(), 0);
        assert_eq!(state.evaluate(&[], usage(9), 30, now()).len(), 1);
    }

    // === Episode end + re-arm =============================================

    // Validates: Requirement 4.4 (episode end and fresh re-fire)
    #[test]
    fn episode_rearms_after_scan_drops_model() {
        let state = DeprecationState::default();
        state.record_scan("openai", vec![retiring_soon("gpt-4-turbo", 5)]);
        assert_eq!(state.evaluate(&[], usage(10), 30, now()).len(), 1);
        // Same signal again: exactly one event per episode.
        assert!(state.evaluate(&[], usage(10), 30, now()).is_empty());
        // Successful scan no longer lists the model: signal cleared.
        // Episode closure is lazy (evaluated on the next pass — the
        // alerts.rs discipline): this evaluate emits nothing and
        // closes the stale episode.
        state.record_scan("openai", vec![retiring_soon("gpt-4.1", 400)]);
        assert!(state.evaluate(&[], usage(10), 30, now()).is_empty());
        assert_eq!(state.episodes.len(), 0);
        // Re-qualification emits a fresh event.
        state.record_scan("openai", vec![retiring_soon("gpt-4-turbo", 3)]);
        assert_eq!(state.evaluate(&[], usage(10), 30, now()).len(), 1);
    }

    // Validates: Requirement 4.4 (override removal re-arms)
    #[test]
    fn episode_rearms_when_override_removed() {
        let state = DeprecationState::default();
        let over = override_for("openai", "gpt-4-turbo", Some("2026-11-20"), None);
        assert_eq!(state.evaluate(&[over.clone()], usage(10), 30, now()).len(), 1);
        assert!(state.evaluate(&[over.clone()], usage(10), 30, now()).is_empty());
        // Operator clears the override: episode closes.
        assert!(state.evaluate(&[], usage(10), 30, now()).is_empty());
        assert_eq!(state.episodes.len(), 0);
        // Override returns: fresh episode, fresh event.
        assert_eq!(state.evaluate(&[over], usage(10), 30, now()).len(), 1);
    }

    // === Warn→gone escalation =============================================

    // Validates: Requirement 4.4 (escalation emits the stronger
    // event exactly once)
    #[test]
    fn warn_to_gone_escalation_emits_exactly_once() {
        let state = DeprecationState::default();
        state.record_scan("openai", vec![retiring_soon("gpt-4-turbo", 10)]);
        let warn_events = state.evaluate(&[], usage(10), 30, now());
        assert_eq!(warn_events.len(), 1);
        assert!(!warn_events[0].runtime_gone);
        assert_eq!(warn_events[0].evidence_source, EvidenceSource::ProviderMetadata);

        // Runtime detection arrives while the warn episode is open:
        // escalation ends the warn episode and opens a gone episode.
        state.record_runtime_gone("openai", "gpt-4-turbo", now());
        let escalated = state.evaluate(&[], usage(10), 30, now());
        assert_eq!(escalated.len(), 1);
        assert!(escalated[0].runtime_gone);
        assert_eq!(escalated[0].evidence_source, EvidenceSource::RuntimeDetection);
        // The gone episode still carries the scan-sourced date hint.
        assert_eq!(escalated[0].days_remaining, Some(10));

        // Repeated gone signals never re-emit.
        state.record_runtime_gone("openai", "gpt-4-turbo", later(1));
        assert!(state.evaluate(&[], usage(10), 30, later(2)).is_empty());
    }

    // Validates: Requirement 4.4 (no duplicate when runtime fires
    // over an already-gone episode)
    #[test]
    fn gone_episode_stays_silent_on_repeated_detections() {
        let state = DeprecationState::default();
        state.record_runtime_gone("openai", "gpt-4-turbo", now());
        assert_eq!(state.evaluate(&[], usage(10), 30, now()).len(), 1);
        state.record_runtime_gone("openai", "gpt-4-turbo", later(1));
        assert!(state.evaluate(&[], usage(10), 30, later(2)).is_empty());
    }

    // === Runtime-gone set/clear + 24h bound ==============================

    // Validates: Requirements 4.4, 5.2 (runtime-present re-arm)
    #[test]
    fn runtime_present_clears_gone_and_rearms() {
        let state = DeprecationState::default();
        state.record_runtime_gone("openai", "gpt-4-turbo", now());
        assert!(state.is_gone("openai", "gpt-4-turbo"));
        assert_eq!(state.evaluate(&[], usage(10), 30, now()).len(), 1);

        // Model serves again: mark cleared, episode closed.
        state.record_runtime_present("openai", "gpt-4-turbo");
        assert!(!state.is_gone("openai", "gpt-4-turbo"));
        assert!(state.evaluate(&[], usage(10), 30, now()).is_empty());
        assert_eq!(state.episodes.len(), 0);

        // A later gone detection re-qualifies: fresh event.
        state.record_runtime_gone("openai", "gpt-4-turbo", later(1));
        assert_eq!(state.evaluate(&[], usage(10), 30, later(1)).len(), 1);
    }

    // Validates: Requirement 5.2 (re-arming scan clears gone)
    #[test]
    fn scan_seeing_model_present_clears_gone_mark() {
        let state = DeprecationState::default();
        state.record_runtime_gone("openai", "gpt-4-turbo", now());
        assert!(state.is_gone("openai", "gpt-4-turbo"));
        // Fresh scan lists the model (no deprecation signal): the
        // listing itself observes the model present.
        state.record_scan("openai", vec![retiring_soon("gpt-4-turbo", 400)]);
        assert!(!state.is_gone("openai", "gpt-4-turbo"));
        assert!(state.evaluate(&[], usage(10), 30, now()).is_empty());
    }

    // Validates: Requirement 5.2 (24h unconfirmed bound)
    #[test]
    fn gone_mark_expires_after_24_hours() {
        let state = DeprecationState::default();
        state.record_runtime_gone("openai", "gpt-4-turbo", now());
        assert_eq!(state.gone_since("openai", "gpt-4-turbo"), Some(now()));
        // Inside the bound: gone counts, event fires.
        assert!(state.is_gone_at("openai", "gpt-4-turbo", later(23)));
        assert_eq!(state.evaluate(&[], usage(10), 30, later(23)).len(), 1);
        // At/after the bound: the mark stops counting...
        assert!(!state.is_gone_at("openai", "gpt-4-turbo", later(24)));
        // ...and evaluate evicts it on touch, closing the episode.
        assert!(state.evaluate(&[], usage(10), 30, later(24)).is_empty());
        assert!(!state.is_gone("openai", "gpt-4-turbo"));
        assert_eq!(state.episodes.len(), 0);
    }

    // === Merge discipline =================================================

    // Validates: Requirement 1.3 (stale-keep: unmentioned providers
    // keep previous data)
    #[test]
    fn rescan_keeps_unmentioned_providers() {
        let state = DeprecationState::default();
        state.record_scan("openai", vec![retiring_soon("gpt-4-turbo", 5)]);
        state.record_scan("groq", vec![retiring_soon("llama-3", 5)]);
        // Only groq rescanned; openai scan data must survive.
        state.record_scan("groq", vec![retiring_soon("llama-4", 400)]);
        let events = state.evaluate(&[], usage(10), 30, now());
        assert_eq!(events.len(), 1);
        assert_eq!(events[0].provider, "openai");
        assert_eq!(events[0].model, "gpt-4-turbo");
    }

    // Validates: Requirement 1.4 (config reload drops removed
    // providers entirely)
    #[test]
    fn remove_and_retain_providers_drop_all_state() {
        let state = DeprecationState::default();
        state.record_scan("openai", vec![retiring_soon("gpt-4-turbo", 5)]);
        state.record_scan("groq", vec![retiring_soon("llama-3", 5)]);
        assert_eq!(state.evaluate(&[], usage(10), 30, now()).len(), 2);

        state.remove_provider("openai");
        assert_eq!(state.scan.len(), 1);
        assert_eq!(state.episodes.len(), 1);

state.retain_providers(&["anthropic".to_string()]);
assert_eq!(state.scan.len(), 0);
assert_eq!(state.runtime_gone.len(), 0);
assert_eq!(state.episodes.len(), 0);
}

// Validates: Requirement 5.2 (reload clears runtime-gone marks for
// still-configured providers while scan/episode state persists)
#[test]
fn clear_runtime_gone_clears_marks_but_keeps_scan_and_episodes() {
let state = DeprecationState::default();
state.record_scan("openai", vec![retiring_soon("gpt-4-turbo", 5)]);
assert_eq!(state.evaluate(&[], usage(10), 30, now()).len(), 1);
state.record_runtime_gone("openai", "gpt-4-turbo", now());
assert!(state.is_gone("openai", "gpt-4-turbo"));

// Reload sweep for a still-configured provider: the gone mark
// clears (candidate skip ends, Req 5.2) ...
state.clear_runtime_gone("openai");
assert!(!state.is_gone("openai", "gpt-4-turbo"));
assert!(!state.is_gone_at("openai", "gpt-4-turbo", now()));
// ... while scan data and the open episode survive.
assert_eq!(state.scan.len(), 1);
assert_eq!(state.episodes.len(), 1);
// The surviving scan signal still qualifies (no duplicate
// emission from the still-open episode).
assert!(state.evaluate(&[], usage(10), 30, now()).is_empty());
assert_eq!(state.episodes.len(), 1);
}

    // === Status rows ======================================================

    // Validates: Requirement 3.2 (unused deprecated models still
    // listed) + design "DeprecationStatusRow" shape
    #[test]
    fn status_rows_list_unused_and_ledger_failed_models() {
        let state = DeprecationState::default();
        state.record_scan(
            "openai",
            vec![
                retiring_soon("used-model", 5),
                retiring_soon("unused-model", 40),
                scan_model("date-less", None, None, true),
                retiring_soon("healthy-model", 400),
            ],
        );
        let mut over = override_for("openai", "unused-model", None, None);
        over.model_groups = vec!["main".to_string()];
        let rows = state.status_rows(&[over.clone()], no_usage(), 30, now());

        // Healthy model absent; both qualifying keys present, sorted
        // (the beyond-window model waits silently for its window);
        // ledger failure renders 0 usage.
        assert_eq!(rows.len(), 2);
        assert_eq!(rows[0].model, "date-less");
        assert_eq!(rows[0].retirement_date, None);
        assert_eq!(rows[0].days_remaining, None);
        assert_eq!(rows[1].model, "used-model");
        assert_eq!(rows[1].days_remaining, Some(5));
        assert_eq!(rows[1].requests_7d, 0);
        assert_eq!(rows[1].last_used_at, None);
        assert_eq!(rows[1].model_groups, Vec::<String>::new());

        // The beyond-window model appears once its date crosses the
        // window boundary — still no episode (nothing fired).
        let soon = later(24 * 35);
        let rows_soon = state.status_rows(&[over.clone()], usage(1), 30, soon);
        let approaching = rows_soon
            .iter()
            .find(|r| r.model == "unused-model")
            .expect("window crossed");
        assert_eq!(approaching.days_remaining, Some(5));
        assert_eq!(approaching.model_groups, vec!["main".to_string()]);
        assert!(!approaching.episode_open);

        // Episode opens for the used model (usage scoped to it —
        // the date-less model's ledger read fails, so it stays
        // unopened); the row reflects both.
        let used_only = |_: &str, m: &str| -> Option<(u64, Option<DateTime<Utc>>)> {
            (m == "used-model").then_some((7, None))
        };
        state.evaluate(&[], used_only, 30, now());
        let rows_after = state.status_rows(&[over], usage(7), 30, now());
        let used = rows_after.iter().find(|r| r.model == "used-model").unwrap();
        assert!(used.episode_open);
        assert_eq!(used.requests_7d, 7);
        let date_less = rows_after.iter().find(|r| r.model == "date-less").unwrap();
        assert!(!date_less.episode_open);
    }

    // Validates: design "Component 5" (runtime-gone rows show
    // runtime evidence)
    #[test]
    fn status_rows_reflect_runtime_gone() {
        let state = DeprecationState::default();
        state.record_runtime_gone("openai", "gpt-4-turbo", now());
        let rows = state.status_rows(&[], usage(3), 30, now());
        assert_eq!(rows.len(), 1);
        assert!(rows[0].runtime_gone);
        assert_eq!(rows[0].evidence_source, EvidenceSource::RuntimeDetection);
        assert_eq!(rows[0].retirement_date, None);
    }

    // === Concurrent evaluation: exactly one event per episode =============

    // Validates: Requirement 4.1 (design "Atomicity" discipline)
    #[test]
    fn concurrent_qualifications_open_exactly_one_episode() {
        const THREADS: usize = 16;
        let state = DeprecationState::default();
        state.record_scan("openai", vec![retiring_soon("gpt-4-turbo", 5)]);
        let state = std::sync::Arc::new(state);
        let mut handles = Vec::new();
        for _ in 0..THREADS {
            let state = std::sync::Arc::clone(&state);
            handles.push(std::thread::spawn(move || {
                state.evaluate(&[], usage(10), 30, now()).len()
            }));
        }
        let total: usize = handles
            .into_iter()
            .map(|h| h.join().expect("thread joins"))
            .sum();
        assert_eq!(total, 1);
    }

    // === Property-based tests =============================================

    use proptest::prelude::*;

    proptest! {
        #![proptest_config(ProptestConfig::with_cases(64))]

        // Feature: model-deprecation-alerts, Property 1: no double
        // fire — for any retirement offset and warn window, an
        // episode that has already fired never fires again while the
        // signal persists, and a fresh state fires at most once.
        // Validates: Requirements 4.1, 4.4
        #[test]
        fn prop_episode_fires_at_most_once(
            days_offset in -400i64..=400,
            warn_days in 1u32..=365,
        ) {
            let state = DeprecationState::default();
            state.record_scan(
                "openai",
                vec![scan_model("m", retirement_in(days_offset), None, false)],
            );
            let first = state.evaluate(&[], usage(10), warn_days, now());
            let second = state.evaluate(&[], usage(10), warn_days, now());
            let third = state.evaluate(&[], usage(50), warn_days, now());
            let total = first.len() + second.len() + third.len();
            prop_assert!(total <= 1);
            prop_assert!(second.is_empty());
            prop_assert!(third.is_empty());
        }

        // Feature: model-deprecation-alerts, Property 2: the usage
        // gate is a strict floor — zero usage and ledger failure
        // never emit; once usage exists the episode fires at most
        // once regardless of how usage grows.
        // Validates: Requirements 3.1, 3.2
        #[test]
        fn prop_usage_gate_bounds_emission(
            days_offset in 0i64..=30,
            low in 0u64..=10,
            extra in 0u64..=1000,
        ) {
            let state = DeprecationState::default();
            state.record_scan("openai", vec![retiring_soon("m", days_offset)]);
            let warn_days = 30u32;
            // The gate pass/fail floor: 0 usage or a failed ledger
            // lookup is always silent.
            let silent = state.evaluate(&[], usage(0), warn_days, now());
            let failed = state.evaluate(&[], no_usage(), warn_days, now());
            prop_assert!(silent.is_empty());
            prop_assert!(failed.is_empty());
            prop_assert_eq!(state.episodes.len(), 0);
            // With real usage the episode fires at most once, never
            // again as usage grows.
            let high = low + extra + 1;
            let loud = state.evaluate(&[], usage(high), warn_days, now());
            let repeat = state.evaluate(&[], usage(high * 2), warn_days, now());
            prop_assert!(loud.len() + repeat.len() <= 1);
            prop_assert!(repeat.is_empty());
        }

        // Feature: model-deprecation-alerts, Property 3: eviction
        // bounds — registries never exceed the number of distinct
        // signal-carrying keys: episodes only for fired events,
        // gone marks only inside the 24h bound, scan entries only
        // for currently-listed models.
        // Validates: Requirement 5.3 (bounded registry)
        #[test]
        fn prop_registries_stay_bounded(
            marks in proptest::collection::vec((1u32..=10, 1i64..=72), 0..=24),
        ) {
            let state = DeprecationState::default();
            let mut stamp = now();
            for (idx, (model, hours)) in marks.iter().enumerate() {
                stamp = later(*hours);
                state.record_runtime_gone("openai", &format!("m-{idx}-{model}"), stamp);
            }
// Evict at the latest stamp: only marks inside the 24h
// bound survive, and the sweep actually reclaims them.
let evicted = state.evict_expired(stamp);
prop_assert!(state.runtime_gone.len() <= 24);
let surviving: usize = state
.runtime_gone
.iter()
.map(|models| models.value().len())
.sum();
prop_assert_eq!(surviving + evicted, marks.len());
// Every surviving mark is still inside its bound.
for models in state.runtime_gone.iter() {
for mark in models.value().iter() {
prop_assert!(stamp < mark.value().gone_until);
}
}
}
    }
}
