//! Per-`provider:model` cooldown for Electron Hub Coding Plan (DevPass)
//! `model_slots_exceeded` refusals.
//!
//! Past the daily or weekly full-speed headroom a DevPass seat drops into
//! a slow lane with fewer concurrent slots. A Heavy model that needs more
//! slots than the lane allows (`glm-5.3:dev` needs 3, the Turbo slow lane
//! has 2) is refused with `403` and code `model_slots_exceeded`, with no
//! `Retry-After`. The refusal lasts until the headroom resets: daily at
//! 21:00 UTC, weekly on Sunday at 21:00 UTC
//! (<https://docs.electronhub.ai/billing/coding-plan.md#limits>).
//!
//! Other models on the same key keep working, so the cooldown is keyed
//! per `provider:model` and never touches the provider-wide rate-limit
//! cooldown.

use std::time::Duration;

use chrono::{DateTime, Datelike, Utc};
use dashmap::DashMap;

/// Ledger `error_class` for a `model_slots_exceeded` refusal.
pub(crate) const MODEL_SLOTS_EXCEEDED_CLASS: &str = "model_slots_exceeded";

/// Hour (UTC) at which Electron Hub resets DevPass headroom.
const RESET_HOUR_UTC: u32 = 21;

/// Whether an upstream failure is a `model_slots_exceeded` refusal: a 4xx
/// whose body carries the `model_slots_exceeded` code (case-insensitive,
/// so it matches whichever error envelope field holds it).
pub fn is_model_slots_exceeded(status: u16, body: &str) -> bool {
    (400..500).contains(&status) && body.to_ascii_lowercase().contains("model_slots_exceeded")
}

/// Next daily headroom reset strictly after `now`.
pub fn next_daily_reset(now: DateTime<Utc>) -> DateTime<Utc> {
    let today = now
        .date_naive()
        .and_hms_opt(RESET_HOUR_UTC, 0, 0)
        .expect("21:00:00 is a valid time")
        .and_utc();
    if now < today {
        today
    } else {
        today + chrono::Duration::days(1)
    }
}

/// Next weekly headroom reset (Sunday 21:00 UTC) strictly after `now`.
pub fn next_weekly_reset(now: DateTime<Utc>) -> DateTime<Utc> {
    let daily = next_daily_reset(now);
    let days_ahead = (7 - daily.weekday().num_days_from_sunday()) % 7;
    daily + chrono::Duration::days(i64::from(days_ahead))
}

/// When a refused `provider:model` becomes eligible again.
///
/// An upstream hint (Retry-After header or body reset field, already
/// parsed by the caller) wins. Otherwise a body that names the weekly
/// budget waits for the weekly reset, and anything else waits for the
/// next daily reset; a weekly-exhausted seat then costs one refused
/// probe per day. The result is clamped to `max_cooldown` from `now`.
pub fn cooldown_deadline(
    now: DateTime<Utc>,
    upstream_hint: Option<Duration>,
    body: &str,
    max_cooldown: Duration,
) -> DateTime<Utc> {
    let to_chrono =
        |d: Duration| chrono::Duration::from_std(d).unwrap_or_else(|_| chrono::Duration::days(7));
    let cap = now + to_chrono(max_cooldown);
    let deadline = match upstream_hint.map(|d| to_chrono(d.min(max_cooldown))) {
        Some(d) => now + d,
        None if body.to_ascii_lowercase().contains("weekly") => next_weekly_reset(now),
        None => next_daily_reset(now),
    };
    deadline.min(cap)
}

/// Cooldown deadlines per `(provider, model)`.
#[derive(Debug, Default)]
pub struct ModelSlotCooldowns {
    until: DashMap<(String, String), DateTime<Utc>>,
}

impl ModelSlotCooldowns {
    /// Keep `(provider, model)` out of candidate selection until `until`.
    pub fn mark(&self, provider: &str, model: &str, until: DateTime<Utc>) {
        self.until
            .insert((provider.to_string(), model.to_string()), until);
    }

    /// Whether `(provider, model)` is still cooling down at `now`. An
    /// expired entry is removed.
    pub fn is_cooling_at(&self, provider: &str, model: &str, now: DateTime<Utc>) -> bool {
        let key = (provider.to_string(), model.to_string());
        let expired = match self.until.get(&key) {
            Some(deadline) if now < *deadline => return true,
            Some(_) => true,
            None => false,
        };
        if expired {
            self.until.remove_if(&key, |_, deadline| now >= *deadline);
        }
        false
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use chrono::TimeZone;

    fn at(y: i32, mo: u32, d: u32, h: u32, mi: u32) -> DateTime<Utc> {
        Utc.with_ymd_and_hms(y, mo, d, h, mi, 0).unwrap()
    }

    const WEEK: Duration = Duration::from_secs(7 * 24 * 60 * 60);

    #[test]
    fn signature_matches_code_on_any_4xx_only() {
        let cases: &[(u16, &str, bool)] = &[
            (403, r#"{"error":{"message":"Heavy needs 3 slots","code":"model_slots_exceeded"}}"#, true),
            (403, r#"{"error":{"type":"model_slots_exceeded"}}"#, true),
            (403, "MODEL_SLOTS_EXCEEDED", true),
            (429, "model_slots_exceeded", true),
            (403, r#"{"error":{"message":"Concurrency pause"}}"#, false),
            (403, "", false),
            (500, "model_slots_exceeded", false),
            (200, "model_slots_exceeded", false),
            (403, "model slots exceeded", false),
        ];
        for (status, body, expected) in cases {
            assert_eq!(
                is_model_slots_exceeded(*status, body),
                *expected,
                "status {status} body {body:?}"
            );
        }
    }

    #[test]
    fn daily_reset_is_next_21_utc() {
        assert_eq!(next_daily_reset(at(2026, 11, 20, 15, 0)), at(2026, 11, 20, 21, 0));
        assert_eq!(next_daily_reset(at(2026, 11, 20, 21, 0)), at(2026, 11, 21, 21, 0));
        assert_eq!(next_daily_reset(at(2026, 11, 20, 23, 9)), at(2026, 11, 21, 21, 0));
    }

    #[test]
    fn weekly_reset_is_next_sunday_21_utc() {
        // 2026-11-22 is a Sunday.
        assert_eq!(next_weekly_reset(at(2026, 11, 20, 15, 0)), at(2026, 11, 22, 21, 0));
        assert_eq!(next_weekly_reset(at(2026, 11, 22, 20, 59)), at(2026, 11, 22, 21, 0));
        assert_eq!(next_weekly_reset(at(2026, 11, 22, 21, 0)), at(2026, 11, 29, 21, 0));
    }

    #[test]
    fn deadline_prefers_upstream_hint_then_body_window() {
        let now = at(2026, 11, 20, 15, 0);
        assert_eq!(
            cooldown_deadline(now, Some(Duration::from_secs(600)), "", WEEK),
            at(2026, 11, 20, 15, 10)
        );
        assert_eq!(
            cooldown_deadline(now, None, "weekly headroom used", WEEK),
            at(2026, 11, 22, 21, 0)
        );
        assert_eq!(cooldown_deadline(now, None, "", WEEK), at(2026, 11, 20, 21, 0));
        assert_eq!(
            cooldown_deadline(now, Some(Duration::from_secs(30 * 86_400)), "", WEEK),
            now + chrono::Duration::days(7)
        );
    }

    #[test]
    fn cooldown_is_per_provider_model_and_expires() {
        let store = ModelSlotCooldowns::default();
        let now = at(2026, 11, 20, 15, 0);
        store.mark("Electron Hub", "glm-5.3:dev", at(2026, 11, 20, 21, 0));
        assert!(store.is_cooling_at("Electron Hub", "glm-5.3:dev", now));
        assert!(!store.is_cooling_at("Electron Hub", "glm-5.3-flash:dev", now));
        assert!(!store.is_cooling_at("Other", "glm-5.3:dev", now));
        assert!(!store.is_cooling_at("Electron Hub", "glm-5.3:dev", at(2026, 11, 20, 21, 0)));
        assert!(store.until.is_empty(), "expired entry is removed");
    }
}
