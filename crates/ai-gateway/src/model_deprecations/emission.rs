//! Event-emission orchestration for model-deprecation alerts
//! (model-deprecation-alerts spec, Task 6).
//!
//! Bridges the pure state core ([`DeprecationState::evaluate`]) to the
//! generalized webhook delivery seam
//! ([`WebhookNotifier::emit_value`], design D9): projects operator
//! overrides from config, resolves usage evidence from the requests
//! ledger on the blocking pool, evaluates episodes, and emits every
//! resulting `model.deprecated` envelope — all inside a spawned task,
//! never blocking the caller (the scanner pass or the runtime-gone
//! reporting seam).
//!
//! Dormancy contract (mirrors `webhooks/provider_hook.rs`): when
//! `config.webhooks.endpoints` is empty the orchestrator skips
//! evaluation AND emission entirely — no ledger writes, no usage
//! queries. A failed ledger lookup returns `None` from the usage
//! closure and the state core suppresses that key's event without
//! opening the episode (a broken ledger must not become alert spam
//! nor swallow the episode forever).
//!
//! _Requirements: 3.1, 3.2, 4.5 (emission), 3.3-dormancy (webhooks)_

use std::sync::Arc;

use chrono::{DateTime, Utc};
use tokio::sync::RwLock;

use crate::config::Config;
use crate::logger::RequestLogger;
use crate::model_deprecations::events::ModelDeprecatedEvent;
use crate::model_deprecations::state::{DeprecationState, OverrideSpec};
use crate::webhooks::delivery::WebhookNotifier;

/// Usage-evidence lookback: events carry `requests_7d` (Reqs
/// 3.1/3.2 — "requests routed to this exact pair over the last 7
/// days").
const USAGE_LOOKBACK_DAYS: i64 = 7;

/// Snapshot of everything the orchestrator clones out of config under
/// the read lock (dropped before any I/O — the delivery.rs
/// config-read discipline).
#[derive(Debug, Default)]
struct ConfigSnapshot {
    /// Operator overrides projected from `model_groups[].models[]`
    /// (`deprecation_date` / `replacement_model`), carrying the group
    /// names referencing each pair.
    overrides: Vec<OverrideSpec>,
    /// `model_deprecations.warn_days_before`.
    warn_days: u32,
    /// Dormancy: `true` when no webhook endpoints are configured.
    webhooks_dormant: bool,
}

impl ConfigSnapshot {
    fn take(config: &Config) -> Self {
        let mut overrides = Vec::new();
        for group in &config.model_groups {
            for entry in &group.models {
                // Only pairs carrying an operator signal project an
                // override: a bare (provider, model) entry with no
                // deprecation fields must not shadow scan data.
                if entry.deprecation_date.is_some() || entry.replacement_model.is_some() {
                    overrides.push(OverrideSpec {
                        provider: entry.provider.clone(),
                        model: entry.model.clone(),
                        deprecation_date: entry.deprecation_date.clone(),
                        replacement_model: entry.replacement_model.clone(),
                        model_groups: vec![group.name.clone()],
                    });
                }
            }
        }
        Self {
            overrides,
            warn_days: config.model_deprecations.warn_days_before,
            webhooks_dormant: config.webhooks.endpoints.is_empty(),
        }
    }
}

/// Merge `model_groups` names for duplicate `(provider, model)` pairs
/// across groups into one `OverrideSpec` (a pair may appear in
/// several groups; the status row lists them all).
fn dedupe_overrides(overrides: Vec<OverrideSpec>) -> Vec<OverrideSpec> {
    use std::collections::BTreeMap;
    let mut merged: BTreeMap<(String, String), OverrideSpec> = BTreeMap::new();
    for entry in overrides {
        let key = (entry.provider.clone(), entry.model.clone());
        match merged.remove(&key) {
            Some(existing) => {
                let mut groups = existing.model_groups;
                for name in entry.model_groups {
                    if !groups.contains(&name) {
                        groups.push(name);
                    }
                }
        // First-seen override fields win: when config lists the same
        // pair in several groups with diverging values, the first
        // entry in config order determines the emitted date/replacement
        // (validation does not reject such drift).
                merged.insert(
                    key,
                    OverrideSpec {
                        provider: existing.provider,
                        model: existing.model,
                        deprecation_date: existing.deprecation_date,
                        replacement_model: existing.replacement_model,
                        model_groups: groups,
                    },
                );
            }
            None => {
                merged.insert(key, entry);
            }
        }
    }
    merged.into_values().collect()
}

/// Evaluate and emit `model.deprecated` events (Task 6 orchestration):
/// call after every scan pass completes and after a runtime-gone
/// detection. Spawns the whole pipeline — dormancy check, override
/// projection, blocking-pool usage lookups, pure evaluation, and one
/// generalized emit per resulting event — so the caller (scanner pass,
/// admin manual trigger, runtime hook) never waits.
pub fn spawn_deprecation_emission(
    config: Arc<RwLock<Config>>,
    state: Arc<DeprecationState>,
    logger: Arc<RequestLogger>,
    notifier: Arc<WebhookNotifier>,
) {
    tokio::spawn(async move {
        // Config snapshot under the read lock; the guard drops at the
        // end of this block, before any ledger I/O (delivery.rs
        // discipline — no lock across await).
        let snapshot = {
            let guard = config.read().await;
            ConfigSnapshot::take(&guard)
        };
        // Dormancy contract: zero endpoints → no evaluation, no
        // usage queries, no ledger writes (provider_hook.rs
        // discipline).
        if snapshot.webhooks_dormant {
            return;
        }
        let overrides = dedupe_overrides(snapshot.overrides);
        let warn_days = snapshot.warn_days;
        let now = Utc::now();

        // Pure evaluation with blocking-pool usage lookups: each
        // candidate key's evidence query opens its own WAL reader
        // connection against the requests ledger (cheap; candidate
        // sets stay small — bounded by live deprecation signals).
        let events = tokio::task::spawn_blocking(move || {
            let lookup = |provider: &str, model: &str| -> Option<(u64, Option<DateTime<Utc>>)> {
                let since = Utc::now() - chrono::Duration::days(USAGE_LOOKBACK_DAYS);
                logger.usage_since(provider, model, since)
            };
            state.evaluate(&overrides, lookup, warn_days, now)
        })
        .await
        .unwrap_or_default();

        for data in events {
            // Typed constructor keeps the envelope shape owned by the
            // events module (D9: typed constructors on both sides).
            let event = ModelDeprecatedEvent::new(data);
            let envelope = serde_json::to_value(&event).ok();
            if let Some(envelope) = envelope {
                notifier.emit_value("model.deprecated", envelope).await;
            }
        }
    });
}

#[cfg(test)]
mod tests {
    use super::*;

    fn override_entry(
        provider: &str,
        model: &str,
        date: Option<&str>,
        replacement: Option<&str>,
        groups: &[&str],
    ) -> OverrideSpec {
        OverrideSpec {
            provider: provider.to_string(),
            model: model.to_string(),
            deprecation_date: date.map(str::to_string),
            replacement_model: replacement.map(str::to_string),
            model_groups: groups.iter().map(|name| name.to_string()).collect(),
        }
    }

    /// Duplicate pairs across groups merge into one spec carrying all
    /// group names (status rows list every referencing group).
    #[test]
    fn duplicate_pairs_merge_group_names() {
        let merged = dedupe_overrides(vec![
            override_entry("p", "m", Some("2026-12-01"), None, &["a"]),
            override_entry("p", "m", None, Some("m2"), &["b"]),
            override_entry("p", "m", None, None, &["a"]),
            override_entry("p", "other", None, Some("x"), &["c"]),
        ]);
        assert_eq!(merged.len(), 2);
        assert_eq!(merged[0].model, "m");
        assert_eq!(merged[0].model_groups, vec!["a".to_string(), "b".to_string()]);
        // First-seen override fields win for the same pair.
        assert_eq!(merged[0].deprecation_date.as_deref(), Some("2026-12-01"));
        assert_eq!(merged[0].replacement_model.as_deref(), None);
        assert_eq!(merged[1].model, "other");
        assert_eq!(merged[1].replacement_model.as_deref(), Some("x"));
    }

    /// The snapshot projects ONLY signal-carrying config entries — a
    /// bare pair with no deprecation fields must not shadow scan data
    /// (an override with no date and no replacement is a no-op).
    #[test]
    fn snapshot_skips_signal_free_entries() {
        let config: Config = serde_yaml::from_str(
            "server:\n host: 127.0.0.1\n port: 0\nproviders: []\nmodel_groups:\n  - name: g1\n    models:\n      - provider: p\n        model: with-date\n        deprecation_date: \"2026-12-01\"\n      - provider: p\n        model: bare\n      - provider: p\n        model: with-replacement\n        replacement_model: next-gen\n",
        )
        .expect("test config must parse");
        let snapshot = ConfigSnapshot::take(&config);
        assert_eq!(snapshot.overrides.len(), 2);
        assert_eq!(snapshot.overrides[0].model, "with-date");
        assert_eq!(snapshot.overrides[1].model, "with-replacement");
        assert_eq!(snapshot.warn_days, 30);
        assert!(snapshot.webhooks_dormant);
    }
}
