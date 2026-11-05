#![allow(dead_code)]
//! Tracks OpenAI rate-limit usage for OAuth (browser-login) providers.
//!
//! API-key OpenAI responses expose count-based `x-ratelimit-*` headers, while
//! the ChatGPT Codex backend used by browser login exposes percentage-based
//! `x-codex-{primary,secondary}-*` headers. Both formats are normalized into
//! the same short/weekly snapshot consumed by the admin provider card.
//! GitHub Copilot responses additionally carry `x-copilot-premium-requests-*`
//! percentage headers, recorded as a separate premium window in the snapshot.

use serde::{Deserialize, Serialize};
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::Arc;
use std::time::{SystemTime, UNIX_EPOCH};
use tokio::sync::RwLock;

/// A single rate-limit window (e.g., "requests per 5h" or "tokens per week").
#[derive(Debug, Clone, Serialize, Deserialize, Default)]
pub struct RateLimitWindow {
    /// Maximum allowed in this window (from `x-ratelimit-limit-*`).
    pub limit: Option<u64>,
    /// Remaining in this window (from `x-ratelimit-remaining-*`).
    pub remaining: Option<u64>,
    /// Seconds until this window resets (from `x-ratelimit-reset-*`).
    /// Stored as absolute Unix epoch timestamp so it stays valid over time.
    pub resets_at: Option<u64>,
    /// When `true`, `limit` and `remaining` represent percentages (0–100)
    /// derived from `x-codex-*-used-percent` headers, not absolute counts.
    /// The frontend should render these as "37%" rather than "37 / 100".
    #[serde(default)]
    pub is_percentage: bool,
}

/// Threshold (in seconds) used to distinguish the short (5h) window from the
/// weekly window. If the parsed reset duration is >= this value, the headers
/// are attributed to the weekly window; otherwise to the short window.
const WEEKLY_THRESHOLD_SECS: u64 = 86_400; // 24h

impl RateLimitWindow {
    /// Percentage of the window used (0.0–100.0), or None if no limit data.
    pub fn usage_percent(&self) -> Option<f64> {
        match (self.limit, self.remaining) {
            (Some(limit), Some(remaining)) if limit > 0 => {
                Some(((limit - remaining.min(limit)) as f64 / limit as f64) * 100.0)
            }
            _ => None,
        }
    }

    /// Seconds remaining until the window resets, or None.
    pub fn reset_in_secs(&self) -> Option<u64> {
        let resets_at = self.resets_at?;
        let now = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .map(|d| d.as_secs())
            .unwrap_or(0);
        if resets_at > now {
            Some(resets_at - now)
        } else {
            None
        }
    }
}

/// A pair of request/token windows that belong to the same time horizon
/// (either the short ~5h window or the weekly ~7d window).
#[derive(Debug, Clone, Serialize, Deserialize, Default)]
pub struct WindowPair {
    pub requests: RateLimitWindow,
    pub tokens: RateLimitWindow,
}

/// Snapshot of all tracked rate-limit windows for the OAuth provider.
///
/// OpenAI browser-login accounts expose two sliding windows:
///   - `short`  — the ~5 hour window (requests + tokens).
///   - `weekly` — the 7-day weekly window (requests + tokens).
#[derive(Debug, Clone, Serialize, Deserialize, Default)]
pub struct UsageSnapshot {
    pub short: WindowPair,
    pub weekly: WindowPair,
    /// Premium-request allowance (GitHub Copilot `x-copilot-premium-requests-*`
    /// headers). Percentage-based: `used`/`limit` are percentages of the
    /// subscription's premium-request allowance, independent of the count-based
    /// short/weekly windows.
    #[serde(default)]
    pub premium: Option<RateLimitWindow>,
    /// Unix timestamp when this snapshot was last updated.
    pub updated_at: u64,
}

/// Thread-safe tracker for OpenAI OAuth usage headers.
#[derive(Debug, Clone)]
pub struct UsageTracker {
    state: Arc<RwLock<UsageSnapshot>>,
    /// Generation counter (G6): `clear()` bumps it; `update_from_headers`
    /// captures it before parsing and publishes only when unchanged, so a
    /// response that lands after logout never repopulates the previous
    /// account's windows.
    epoch: Arc<AtomicU64>,
}

impl UsageTracker {
    pub fn new() -> Self {
        Self {
            state: Arc::new(RwLock::new(UsageSnapshot::default())),
            epoch: Arc::new(AtomicU64::new(0)),
        }
    }

    /// Update tracked usage from either OpenAI API or ChatGPT Codex response headers.
    ///
    /// API-key responses use count-based `x-ratelimit-*` headers. The Codex
    /// backend used by OAuth browser login instead reports percentage-based
    /// primary (short) and secondary (weekly) windows through `x-codex-*`.
    pub async fn update_from_headers(&self, headers: &reqwest::header::HeaderMap) {
        let epoch = self.epoch.load(Ordering::Acquire);
        self.update_if_current(epoch, headers).await;
    }

    /// Publish a header-derived update only when `epoch` is still the
    /// current generation (G6): an update that straddled a `clear()` is
    /// dropped instead of repopulating the logged-out account's windows.
    async fn update_if_current(&self, epoch: u64, headers: &reqwest::header::HeaderMap) {
        let now = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .map(|d| d.as_secs())
            .unwrap_or(0);

        let mut state = self.state.write().await;
        if epoch != self.epoch.load(Ordering::Acquire) {
            // A clear() (logout) raced this update: drop the stale payload.
            return;
        }
        let mut updated = update_from_codex_headers(&mut state, headers, now);
        updated |= update_from_standard_headers(&mut state, headers, now);
        updated |= update_from_premium_headers(&mut state, headers, now);

        if updated {
            state.updated_at = now;
        }
    }

    /// Get the current usage snapshot.
    pub async fn snapshot(&self) -> UsageSnapshot {
    self.state.read().await.clone()
    }

    /// Clear all cached windows (short, weekly, premium) and reset
    /// `updated_at` — the logout path (Copilot admin Req 6.2 "clear cached
    /// usage") calls this so a stale pre-logout snapshot never lingers for
    /// the next account session. Bumps the epoch so in-flight updates that
    /// captured the pre-clear generation are discarded (G6).
    pub async fn clear(&self) {
        self.epoch.fetch_add(1, Ordering::AcqRel);
        *self.state.write().await = UsageSnapshot::default();
    }

    /// Test-only seam for G6: current generation counter.
    #[cfg(test)]
    fn current_epoch(&self) -> u64 {
        self.epoch.load(Ordering::Acquire)
    }

    /// Compute a fallback cooldown duration from the tracked headers when
    /// the provider returns 429 but no `Retry-After` header.
    ///
    /// When a window is known to be exhausted (`remaining == 0`), uses the
    /// longest reset among exhausted windows, so a spent 5h window does not
    /// cause a multi-day pause while the weekly window still has headroom.
    /// With no window known to be exhausted, uses the longest reset across
    /// all windows. Returns `None` if no reset data is available.
    pub async fn fallback_cooldown_secs(&self) -> Option<u64> {
        let state = self.state.read().await;
        let windows = [
            &state.short.requests,
            &state.short.tokens,
            &state.weekly.requests,
            &state.weekly.tokens,
        ];
        let exhausted = windows
            .iter()
            .filter(|window| window.remaining == Some(0))
            .filter_map(|window| window.reset_in_secs())
            .max();
        exhausted.or_else(|| {
            windows
                .iter()
                .filter_map(|window| window.reset_in_secs())
                .max()
        })
    }
}

fn update_from_codex_headers(
    state: &mut UsageSnapshot,
    headers: &reqwest::header::HeaderMap,
    now: u64,
) -> bool {
    let primary = codex_window(headers, "primary", now);
    let secondary = codex_window(headers, "secondary", now);

    let mut updated = false;
    if let Some(window) = primary {
        assign_codex_window(state, window, now);
        updated = true;
    }
    if let Some(window) = secondary {
        assign_codex_window(state, window, now);
        updated = true;
    }

    updated
}

fn assign_codex_window(state: &mut UsageSnapshot, window: RateLimitWindow, now: u64) {
    let is_weekly = window
        .resets_at
        .map(|reset| reset.saturating_sub(now) >= WEEKLY_THRESHOLD_SECS)
        .unwrap_or(false);

    if is_weekly {
        state.weekly.requests = window;
    } else {
        state.short.requests = window;
    }
}

fn codex_window(
    headers: &reqwest::header::HeaderMap,
    kind: &str,
    now: u64,
) -> Option<RateLimitWindow> {
    let used_percent = header_f64(headers, &format!("x-codex-{kind}-used-percent"));
    let reset_at =
        header_u64(headers, &format!("x-codex-{kind}-reset-at")).map(normalize_epoch_seconds);

    if used_percent.is_none() && reset_at.is_none() {
        return None;
    }

    let limit = used_percent.map(|_| 100);
    let remaining = used_percent.map(|used| {
        let clamped = used.clamp(0.0, 100.0);
        (100.0 - clamped).round() as u64
    });

    Some(RateLimitWindow {
        limit,
        remaining,
        resets_at: reset_at.filter(|reset| *reset > now),
        is_percentage: true,
    })
}

fn update_from_standard_headers(
    state: &mut UsageSnapshot,
    headers: &reqwest::header::HeaderMap,
    now: u64,
) -> bool {
    let req_limit = header_u64(headers, "x-ratelimit-limit-requests");
    let req_remaining = header_u64(headers, "x-ratelimit-remaining-requests");
    let req_reset_secs =
        header_str(headers, "x-ratelimit-reset-requests").and_then(parse_reset_duration);
    let tok_limit = header_u64(headers, "x-ratelimit-limit-tokens");
    let tok_remaining = header_u64(headers, "x-ratelimit-remaining-tokens");
    let tok_reset_secs =
        header_str(headers, "x-ratelimit-reset-tokens").and_then(parse_reset_duration);

    if req_limit.is_none()
        && req_remaining.is_none()
        && req_reset_secs.is_none()
        && tok_limit.is_none()
        && tok_remaining.is_none()
        && tok_reset_secs.is_none()
    {
        return false;
    }

    let classification_secs = req_reset_secs.or(tok_reset_secs).unwrap_or(0);
    let target = if classification_secs >= WEEKLY_THRESHOLD_SECS {
        &mut state.weekly
    } else {
        &mut state.short
    };

    if let Some(value) = req_limit {
        target.requests.limit = Some(value);
    }
    if let Some(value) = req_remaining {
        target.requests.remaining = Some(value);
    }
    if let Some(seconds) = req_reset_secs {
        target.requests.resets_at = Some(now + seconds);
    }
    if let Some(value) = tok_limit {
        target.tokens.limit = Some(value);
    }
    if let Some(value) = tok_remaining {
        target.tokens.remaining = Some(value);
    }
    if let Some(seconds) = tok_reset_secs {
        target.tokens.resets_at = Some(now + seconds);
    }

    true
}

/// Parse GitHub Copilot premium-request headers (`x-copilot-premium-requests-*`)
/// into a percentage window recorded alongside the count-based windows.
///
/// `used` and `limit` are percentages (e.g. used=37, limit=100 means 37% of
/// the premium-request allowance consumed), so the window is marked
/// `is_percentage = true`. The reset header follows the same duration
/// convention as the count-based `x-ratelimit-reset-*` headers (e.g. "1h").
/// Returns `false` when no premium headers are present, leaving the snapshot
/// untouched (old serialized payloads without the field keep parsing via
/// `serde(default)`).
fn update_from_premium_headers(
    state: &mut UsageSnapshot,
    headers: &reqwest::header::HeaderMap,
    now: u64,
) -> bool {
    let used = header_u64(headers, "x-copilot-premium-requests-used");
    let limit = header_u64(headers, "x-copilot-premium-requests-limit");
    let reset_secs = header_str(headers, "x-copilot-premium-requests-reset")
        .and_then(parse_reset_duration);

    if used.is_none() && limit.is_none() && reset_secs.is_none() {
        return false;
    }

    let used_pct = used.unwrap_or(0).min(100);
    let limit_pct = limit.unwrap_or(100).min(100).max(1);
    let remaining_pct = limit_pct.saturating_sub(used_pct);

    state.premium = Some(RateLimitWindow {
        limit: Some(limit_pct),
        remaining: Some(remaining_pct),
        resets_at: reset_secs.map(|secs| now + secs),
        is_percentage: true,
    });

    true
}

fn normalize_epoch_seconds(value: u64) -> u64 {
    if value > 10_000_000_000 {
        value / 1_000
    } else {
        value
    }
}

/// Extract a header value as a string slice.
fn header_str<'a>(headers: &'a reqwest::header::HeaderMap, name: &str) -> Option<&'a str> {
    headers.get(name)?.to_str().ok()
}

/// Extract a header value as u64.
fn header_u64(headers: &reqwest::header::HeaderMap, name: &str) -> Option<u64> {
    header_str(headers, name)?.trim().parse().ok()
}

/// Extract a header value as f64.
fn header_f64(headers: &reqwest::header::HeaderMap, name: &str) -> Option<f64> {
    header_str(headers, name)?.trim().parse().ok()
}

/// Parse OpenAI's reset duration format.
///
/// OpenAI returns reset times in formats like:
/// - `"6m0s"` (6 minutes, 0 seconds)
/// - `"1s"` (1 second)
/// - `"12ms"` (12 milliseconds, treated as 1 second minimum)
/// - `"4h32m10s"` (4 hours, 32 minutes, 10 seconds)
/// - `"1d2h3m4s"` (1 day, 2 hours, 3 minutes, 4 seconds)
/// - A plain integer (interpreted as seconds)
fn parse_reset_duration(s: &str) -> Option<u64> {
    let s = s.trim();

    // Plain integer (seconds)
    if let Ok(secs) = s.parse::<u64>() {
        return Some(secs);
    }

    // Duration notation: e.g. "4h32m10s", "6m0s", "12ms"
    let mut total_secs: u64 = 0;
    let mut current_num = String::new();

    let chars: Vec<char> = s.chars().collect();
    let mut i = 0;
    while i < chars.len() {
        let c = chars[i];
        if c.is_ascii_digit() || c == '.' {
            current_num.push(c);
            i += 1;
        } else if c == 'm' && i + 1 < chars.len() && chars[i + 1] == 's' {
            // milliseconds — round up to at least 1s
            if let Ok(ms) = current_num.parse::<f64>() {
                total_secs += (ms / 1000.0).ceil() as u64;
            }
            current_num.clear();
            i += 2;
        } else {
            if let Ok(num) = current_num.parse::<f64>() {
                match c {
                    'd' => total_secs += (num * 86400.0) as u64,
                    'h' => total_secs += (num * 3600.0) as u64,
                    'm' => total_secs += (num * 60.0) as u64,
                    's' => total_secs += num as u64,
                    _ => {}
                }
            }
            current_num.clear();
            i += 1;
        }
    }

    // If there's a trailing number without unit, treat as seconds
    if !current_num.is_empty() {
        if let Ok(num) = current_num.parse::<u64>() {
            total_secs += num;
        }
    }

    if total_secs > 0 {
        Some(total_secs)
    } else {
        Some(1)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_parse_reset_duration_simple_seconds() {
        assert_eq!(parse_reset_duration("1s"), Some(1));
        assert_eq!(parse_reset_duration("30s"), Some(30));
    }

    #[test]
    fn test_parse_reset_duration_minutes_seconds() {
        assert_eq!(parse_reset_duration("6m0s"), Some(360));
        assert_eq!(parse_reset_duration("1m30s"), Some(90));
    }

    #[test]
    fn test_parse_reset_duration_hours() {
        assert_eq!(parse_reset_duration("4h32m10s"), Some(16330));
        assert_eq!(parse_reset_duration("1h"), Some(3600));
    }

    #[test]
    fn test_parse_reset_duration_milliseconds() {
        assert_eq!(parse_reset_duration("12ms"), Some(1));
        assert_eq!(parse_reset_duration("1500ms"), Some(2));
    }

    #[test]
    fn test_parse_reset_duration_plain_integer() {
        assert_eq!(parse_reset_duration("60"), Some(60));
    }

    #[test]
    fn test_parse_reset_duration_days() {
        assert_eq!(parse_reset_duration("1d2h3m4s"), Some(93784));
    }

    #[test]
    fn test_usage_percent() {
        let w = RateLimitWindow {
            limit: Some(100),
            remaining: Some(25),
            resets_at: None,
            is_percentage: false,
        };
        assert_eq!(w.usage_percent(), Some(75.0));
    }

    #[test]
    fn test_usage_percent_full() {
        let w = RateLimitWindow {
            limit: Some(100),
            remaining: Some(0),
            resets_at: None,
            is_percentage: false,
        };
        assert_eq!(w.usage_percent(), Some(100.0));
    }

    #[test]
    fn test_usage_percent_none_when_no_data() {
        let w = RateLimitWindow::default();
        assert_eq!(w.usage_percent(), None);
    }

    #[tokio::test]
    async fn test_short_window_classification() {
    let tracker = UsageTracker::new();
    let mut headers = reqwest::header::HeaderMap::new();
    headers.insert("x-ratelimit-limit-requests", "100".parse().unwrap());
    headers.insert("x-ratelimit-remaining-requests", "40".parse().unwrap());
    headers.insert("x-ratelimit-reset-requests", "3h30m0s".parse().unwrap());
    tracker.update_from_headers(&headers).await;
    let snap = tracker.snapshot().await;
    assert_eq!(snap.short.requests.limit, Some(100));
    assert_eq!(snap.short.requests.remaining, Some(40));
    assert!(snap.short.requests.resets_at.is_some());
    // Weekly window should not have been populated
    assert_eq!(snap.weekly.requests.limit, None);
    }

    #[tokio::test]
    async fn test_clear_resets_all_windows() {
    let tracker = UsageTracker::new();
    // Weekly window first (5d reset >= weekly threshold)...
    let mut weekly = reqwest::header::HeaderMap::new();
    weekly.insert("x-ratelimit-limit-tokens", "500".parse().unwrap());
    weekly.insert("x-ratelimit-remaining-tokens", "100".parse().unwrap());
    weekly.insert("x-ratelimit-reset-tokens", "5d12h0m0s".parse().unwrap());
    tracker.update_from_headers(&weekly).await;
    // ...then short (3h reset) + premium in a second update.
    let mut headers = reqwest::header::HeaderMap::new();
    headers.insert("x-ratelimit-limit-requests", "100".parse().unwrap());
    headers.insert("x-ratelimit-remaining-requests", "40".parse().unwrap());
    headers.insert("x-ratelimit-reset-requests", "3h0m0s".parse().unwrap());
    headers.insert("x-copilot-premium-requests-used", "37".parse().unwrap());
    headers.insert("x-copilot-premium-requests-limit", "100".parse().unwrap());
    headers.insert("x-copilot-premium-requests-reset", "27d0h0m0s".parse().unwrap());
    tracker.update_from_headers(&headers).await;
    let snap = tracker.snapshot().await;
    assert!(snap.short.requests.limit.is_some());
    assert!(snap.weekly.tokens.limit.is_some());
    assert!(snap.premium.is_some());
    assert!(snap.updated_at > 0);

    tracker.clear().await;
    let cleared = tracker.snapshot().await;
    assert_eq!(cleared.short.requests.limit, None);
    assert_eq!(cleared.short.requests.remaining, None);
    assert_eq!(cleared.weekly.tokens.limit, None);
    assert_eq!(cleared.weekly.tokens.remaining, None);
    assert!(cleared.premium.is_none());
    assert_eq!(cleared.updated_at, 0);
}

/// G6: a stale in-flight update that captured the pre-clear epoch must
/// not repopulate windows after `clear()` (logout). Deterministic
/// interleaving: capture the epoch exactly like `update_from_headers`
/// does, clear() in between, then publish through the same guarded path.
#[tokio::test]
async fn test_stale_update_after_clear_does_not_repopulate() {
    let tracker = UsageTracker::new();
    let mut headers = reqwest::header::HeaderMap::new();
    headers.insert("x-ratelimit-limit-requests", "100".parse().unwrap());
    headers.insert("x-ratelimit-remaining-requests", "40".parse().unwrap());
    headers.insert("x-ratelimit-reset-requests", "3h0m0s".parse().unwrap());
    tracker.update_from_headers(&headers).await;
    assert!(tracker.snapshot().await.short.requests.limit.is_some());

    // In-flight response: epoch captured before logout's clear()...
    let stale_epoch = tracker.current_epoch();
    tracker.clear().await;
    // ...and lands after it. The guarded publish must drop it.
    tracker.update_if_current(stale_epoch, &headers).await;

    let snap = tracker.snapshot().await;
    assert_eq!(
        snap.short.requests.limit, None,
        "stale pre-clear update must not repopulate windows after logout clear"
    );
    assert_eq!(snap.updated_at, 0);
}

/// G6 companion: a FRESH update after clear() (new epoch) still applies.
#[tokio::test]
async fn test_fresh_update_after_clear_applies() {
    let tracker = UsageTracker::new();
    tracker.clear().await;

    let mut headers = reqwest::header::HeaderMap::new();
    headers.insert("x-ratelimit-limit-requests", "100".parse().unwrap());
    headers.insert("x-ratelimit-remaining-requests", "40".parse().unwrap());
    headers.insert("x-ratelimit-reset-requests", "3h0m0s".parse().unwrap());
    tracker.update_from_headers(&headers).await;

    let snap = tracker.snapshot().await;
    assert_eq!(snap.short.requests.limit, Some(100));
    assert!(snap.updated_at > 0);
}

    #[tokio::test]
    async fn test_weekly_window_classification() {
        let tracker = UsageTracker::new();
        let mut headers = reqwest::header::HeaderMap::new();
        headers.insert("x-ratelimit-limit-requests", "500".parse().unwrap());
        headers.insert("x-ratelimit-remaining-requests", "100".parse().unwrap());
        headers.insert("x-ratelimit-reset-requests", "5d12h0m0s".parse().unwrap());
        tracker.update_from_headers(&headers).await;
        let snap = tracker.snapshot().await;
        assert_eq!(snap.weekly.requests.limit, Some(500));
        assert_eq!(snap.weekly.requests.remaining, Some(100));
        assert!(snap.weekly.requests.resets_at.is_some());
        // Short window should not have been populated
        assert_eq!(snap.short.requests.limit, None);
    }

    #[tokio::test]
    async fn test_codex_primary_header_with_weekly_reset_is_routed_to_weekly() {
        let tracker = UsageTracker::new();
        let now = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .map(|d| d.as_secs())
            .unwrap_or(0);
        let mut headers = reqwest::header::HeaderMap::new();
        // 100% used in the long-term window, reset ~7 days out.
        headers.insert("x-codex-primary-used-percent", "100".parse().unwrap());
        headers.insert(
            "x-codex-primary-reset-at",
            (now + 7 * 86_400).to_string().parse().unwrap(),
        );

        tracker.update_from_headers(&headers).await;
        let snap = tracker.snapshot().await;

        // Should be routed to weekly based on reset duration, not short.
        assert_eq!(snap.weekly.requests.limit, Some(100));
        assert_eq!(snap.weekly.requests.remaining, Some(0));
        assert!(snap.weekly.requests.resets_at.is_some());
        assert_eq!(snap.short.requests.limit, None);
        assert_eq!(snap.short.requests.remaining, None);
    }

    #[tokio::test]
    async fn test_codex_primary_header_with_short_reset_is_routed_to_short() {
        let tracker = UsageTracker::new();
        let now = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .map(|d| d.as_secs())
            .unwrap_or(0);
        let mut headers = reqwest::header::HeaderMap::new();
        // Short-term window, reset 3h30m out.
        headers.insert("x-codex-primary-used-percent", "37.4".parse().unwrap());
        headers.insert(
            "x-codex-primary-reset-at",
            (now + 12_600).to_string().parse().unwrap(),
        );

        tracker.update_from_headers(&headers).await;
        let snap = tracker.snapshot().await;

        assert_eq!(snap.short.requests.limit, Some(100));
        assert_eq!(snap.short.requests.remaining, Some(63));
        assert!(snap.short.requests.resets_at.is_some());
        assert_eq!(snap.weekly.requests.limit, None);
    }

    #[tokio::test]
    async fn test_codex_windows_from_oauth_headers() {
        let tracker = UsageTracker::new();
        let now = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .map(|d| d.as_secs())
            .unwrap_or(0);
        let mut headers = reqwest::header::HeaderMap::new();
        headers.insert("x-codex-primary-used-percent", "37.4".parse().unwrap());
        headers.insert("x-codex-primary-window-minutes", "300".parse().unwrap());
        headers.insert(
            "x-codex-primary-reset-at",
            (now + 18_000).to_string().parse().unwrap(),
        );
        headers.insert("x-codex-secondary-used-percent", "82".parse().unwrap());
        headers.insert("x-codex-secondary-window-minutes", "10080".parse().unwrap());
        headers.insert(
            "x-codex-secondary-reset-at",
            (now + 7 * 86_400).to_string().parse().unwrap(),
        );

        tracker.update_from_headers(&headers).await;
        let snap = tracker.snapshot().await;

        assert_eq!(snap.short.requests.limit, Some(100));
        assert_eq!(snap.short.requests.remaining, Some(63));
        assert!(snap.short.requests.resets_at.is_some());
        assert_eq!(snap.weekly.requests.limit, Some(100));
        assert_eq!(snap.weekly.requests.remaining, Some(18));
        assert!(snap.weekly.requests.resets_at.is_some());
        assert!(snap.updated_at > 0);
    }

    #[tokio::test]
    async fn test_codex_reset_at_milliseconds_are_normalized() {
        let tracker = UsageTracker::new();
        let now = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .map(|d| d.as_secs())
            .unwrap_or(0);
        let mut headers = reqwest::header::HeaderMap::new();
        headers.insert("x-codex-primary-used-percent", "10".parse().unwrap());
        headers.insert(
            "x-codex-primary-reset-at",
            ((now + 3_600) * 1_000).to_string().parse().unwrap(),
        );

        tracker.update_from_headers(&headers).await;
        let snap = tracker.snapshot().await;

        let reset_at = snap.short.requests.resets_at.expect("reset should be set");
        assert!(reset_at > now && reset_at <= now + 3_600);
    }

    #[tokio::test]
    async fn test_unrelated_headers_do_not_mark_usage_updated() {
        let tracker = UsageTracker::new();
        let mut headers = reqwest::header::HeaderMap::new();
        headers.insert("content-type", "text/event-stream".parse().unwrap());

        tracker.update_from_headers(&headers).await;
        let snap = tracker.snapshot().await;

        assert_eq!(snap.updated_at, 0);
    }

    #[tokio::test]
    async fn test_fallback_cooldown_picks_longest_across_windows() {
        let tracker = UsageTracker::new();
        // Short window: resets in ~1h
        let mut h1 = reqwest::header::HeaderMap::new();
        h1.insert("x-ratelimit-limit-requests", "100".parse().unwrap());
        h1.insert("x-ratelimit-remaining-requests", "50".parse().unwrap());
        h1.insert("x-ratelimit-reset-requests", "1h".parse().unwrap());
        tracker.update_from_headers(&h1).await;
        // Weekly window: resets in ~3d
        let mut h2 = reqwest::header::HeaderMap::new();
        h2.insert("x-ratelimit-limit-requests", "500".parse().unwrap());
        h2.insert("x-ratelimit-remaining-requests", "200".parse().unwrap());
        h2.insert("x-ratelimit-reset-requests", "3d".parse().unwrap());
        tracker.update_from_headers(&h2).await;
        let cooldown = tracker.fallback_cooldown_secs().await;
        assert!(cooldown.is_some());
        // 3d = 259200s, should be > 1h = 3600s
        assert!(cooldown.unwrap() > 200_000);
    }

    #[tokio::test]
    async fn test_fallback_cooldown_prefers_exhausted_short_window_over_weekly_headroom() {
        // Observed Codex state: the 5h window is spent (100% used) while the
        // weekly window still has headroom. The pause must follow the 5h reset,
        // not the multi-day weekly reset.
        let tracker = UsageTracker::new();
        let now = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .unwrap()
            .as_secs();
        let mut headers = reqwest::header::HeaderMap::new();
        headers.insert("x-codex-primary-used-percent", "100".parse().unwrap());
        headers.insert(
            "x-codex-primary-reset-at",
            (now + 3_600).to_string().parse().unwrap(),
        );
        headers.insert("x-codex-secondary-used-percent", "16".parse().unwrap());
        headers.insert(
            "x-codex-secondary-reset-at",
            (now + 5 * 86_400).to_string().parse().unwrap(),
        );
        tracker.update_from_headers(&headers).await;

        let cooldown = tracker.fallback_cooldown_secs().await.expect("cooldown");
        assert!(
            cooldown <= 3_600,
            "expected the ~1h short-window reset, got {cooldown}s"
        );
    }

    #[tokio::test]
    async fn test_premium_requests_headers_populate_percentage_window() {
        let tracker = UsageTracker::new();
        let now = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .unwrap()
            .as_secs();
        let mut headers = reqwest::header::HeaderMap::new();
        headers.insert("x-copilot-premium-requests-used", "37".parse().unwrap());
        headers.insert("x-copilot-premium-requests-limit", "100".parse().unwrap());
        headers.insert("x-copilot-premium-requests-reset", "1d12h".parse().unwrap());

        tracker.update_from_headers(&headers).await;
        let snap = tracker.snapshot().await;

        let premium = snap.premium.expect("premium window should be set");
        assert!(premium.is_percentage);
        assert_eq!(premium.limit, Some(100));
        assert_eq!(premium.remaining, Some(63));
        let resets_at = premium.resets_at.expect("premium reset should be set");
        assert!(resets_at > now && resets_at <= now + 36 * 3_600);
        // Percentage semantics: usage_percent derives from used/limit.
        assert_eq!(premium.usage_percent(), Some(37.0));
        assert!(snap.updated_at > 0);
    }

    #[tokio::test]
    async fn test_premium_requests_absent_headers_leave_window_unset() {
        let tracker = UsageTracker::new();
        let mut headers = reqwest::header::HeaderMap::new();
        headers.insert("x-ratelimit-limit-requests", "100".parse().unwrap());
        headers.insert("x-ratelimit-remaining-requests", "40".parse().unwrap());
        headers.insert("x-ratelimit-reset-requests", "3h30m0s".parse().unwrap());

        tracker.update_from_headers(&headers).await;
        let snap = tracker.snapshot().await;

        assert!(snap.premium.is_none());
        // Count-based windows still populated exactly as before.
        assert_eq!(snap.short.requests.limit, Some(100));
        assert_eq!(snap.short.requests.remaining, Some(40));
        assert!(snap.short.requests.resets_at.is_some());
        assert!(!snap.short.requests.is_percentage);
    }

    #[tokio::test]
    async fn test_premium_requests_recorded_alongside_count_windows() {
        let tracker = UsageTracker::new();
        let mut headers = reqwest::header::HeaderMap::new();
        headers.insert("x-ratelimit-limit-requests", "100".parse().unwrap());
        headers.insert("x-ratelimit-remaining-requests", "40".parse().unwrap());
        headers.insert("x-ratelimit-reset-requests", "3h30m0s".parse().unwrap());
        headers.insert("x-copilot-premium-requests-used", "82".parse().unwrap());
        headers.insert("x-copilot-premium-requests-limit", "100".parse().unwrap());
        headers.insert("x-copilot-premium-requests-reset", "5d".parse().unwrap());

        tracker.update_from_headers(&headers).await;
        let snap = tracker.snapshot().await;

        // Count window semantics unchanged (Req 7.1).
        assert_eq!(snap.short.requests.limit, Some(100));
        assert_eq!(snap.short.requests.remaining, Some(40));
        assert!(!snap.short.requests.is_percentage);
        // Premium window recorded alongside (Req 7.2).
        let premium = snap.premium.expect("premium window should be set");
        assert!(premium.is_percentage);
        assert_eq!(premium.remaining, Some(18));
    }

    #[tokio::test]
    async fn test_premium_requests_partial_headers_use_defaults() {
        let tracker = UsageTracker::new();
        let mut headers = reqwest::header::HeaderMap::new();
        // Only the used percentage is present.
        headers.insert("x-copilot-premium-requests-used", "25".parse().unwrap());

        tracker.update_from_headers(&headers).await;
        let snap = tracker.snapshot().await;

        let premium = snap.premium.expect("premium window should be set");
        assert!(premium.is_percentage);
        assert_eq!(premium.limit, Some(100));
        assert_eq!(premium.remaining, Some(75));
        assert_eq!(premium.resets_at, None);
    }

    #[tokio::test]
    async fn test_premium_requests_snapshot_json_round_trip_without_field() {
        // Additive JSON shape: an old serialized payload (no `premium` field)
        // must still deserialize into the snapshot shape via serde(default).
        let json = serde_json::json!({
            "short": {"requests": {}, "tokens": {}},
            "weekly": {"requests": {}, "tokens": {}},
            "updated_at": 42
        });
        let snap: UsageSnapshot = serde_json::from_value(json).expect("old payload parses");
        assert!(snap.premium.is_none());
        assert_eq!(snap.updated_at, 42);
    }
}
