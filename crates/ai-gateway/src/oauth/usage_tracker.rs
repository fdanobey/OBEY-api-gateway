#![allow(dead_code)]
//! Tracks OpenAI rate-limit usage for OAuth (browser-login) providers.
//!
//! OpenAI returns `x-ratelimit-*` headers on every response. For ChatGPT Plus
//! / Pro browser-login accounts, these manifest as two sliding windows:
//!   1. A short window (typically 3-5 hours) — the "5h" limit.
//!   2. A weekly window (7 days) — the "weekly" limit.
//!
//! However, OpenAI only sends a single set of headers per response (no `*-5h`
//! or `*-weekly` suffixes). The router captures these on every response, so the
//! tracker updates the `short` (5h) window on the first response and the
//! `weekly` window on responses where the upstream indicates a long cooldown
//! (Weekly quota exhausted).
//!
//! To distinguish the two, the caller passes a `WindowKind` hint so the
//! tracker knows which window the headers belong to. The admin UI displays both
//! windows side by side.

use std::sync::Arc;
use std::time::{SystemTime, UNIX_EPOCH};
use tokio::sync::RwLock;
use serde::Serialize;

/// A single rate-limit window (e.g., "requests per 5h" or "tokens per week").
#[derive(Debug, Clone, Serialize, Default)]
pub struct RateLimitWindow {
    /// Maximum allowed in this window (from `x-ratelimit-limit-*`).
    pub limit: Option<u64>,
    /// Remaining in this window (from `x-ratelimit-remaining-*`).
    pub remaining: Option<u64>,
    /// Seconds until this window resets (from `x-ratelimit-reset-*`).
    /// Stored as absolute Unix epoch timestamp so it stays valid over time.
    pub resets_at: Option<u64>,
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
#[derive(Debug, Clone, Serialize, Default)]
pub struct WindowPair {
    pub requests: RateLimitWindow,
    pub tokens: RateLimitWindow,
}

/// Snapshot of all tracked rate-limit windows for the OAuth provider.
///
/// OpenAI browser-login accounts expose two sliding windows:
///   - `short`  — the ~5 hour window (requests + tokens).
///   - `weekly` — the 7-day weekly window (requests + tokens).
#[derive(Debug, Clone, Serialize, Default)]
pub struct UsageSnapshot {
    pub short: WindowPair,
    pub weekly: WindowPair,
    /// Unix timestamp when this snapshot was last updated.
    pub updated_at: u64,
}

/// Thread-safe tracker for OpenAI OAuth usage headers.
#[derive(Debug, Clone)]
pub struct UsageTracker {
    state: Arc<RwLock<UsageSnapshot>>,
}

impl UsageTracker {
    pub fn new() -> Self {
        Self {
            state: Arc::new(RwLock::new(UsageSnapshot::default())),
        }
    }

    /// Update tracked usage from response headers.
    ///
    /// Parses the standard OpenAI `x-ratelimit-*` headers:
    /// - `x-ratelimit-limit-requests` / `x-ratelimit-limit-tokens`
    /// - `x-ratelimit-remaining-requests` / `x-ratelimit-remaining-tokens`
    /// - `x-ratelimit-reset-requests` / `x-ratelimit-reset-tokens`
    ///
    /// OpenAI sends a single set of headers per response. The reset duration
    /// is used to classify the response into the short (~5h) or weekly (~7d)
    /// window: if the requests reset duration is >= 24h the headers are applied
    /// to the weekly window, otherwise to the short window. Both windows
    /// accumulate independently over time so the admin UI can display both.
    pub async fn update_from_headers(&self, headers: &reqwest::header::HeaderMap) {
        let now = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .map(|d| d.as_secs())
            .unwrap_or(0);

        let mut state = self.state.write().await;

        // Parse request-window values
        let req_limit = header_u64(headers, "x-ratelimit-limit-requests");
        let req_remaining = header_u64(headers, "x-ratelimit-remaining-requests");
        let req_reset_secs = header_str(headers, "x-ratelimit-reset-requests")
            .and_then(|s| parse_reset_duration(s));

        // Parse token-window values
        let tok_limit = header_u64(headers, "x-ratelimit-limit-tokens");
        let tok_remaining = header_u64(headers, "x-ratelimit-remaining-tokens");
        let tok_reset_secs = header_str(headers, "x-ratelimit-reset-tokens")
            .and_then(|s| parse_reset_duration(s));

        // Classify: use the requests reset duration as the primary signal,
        // falling back to the tokens reset duration.
        let classification_secs = req_reset_secs.or(tok_reset_secs).unwrap_or(0);
        let is_weekly = classification_secs >= WEEKLY_THRESHOLD_SECS;
        let target = if is_weekly { &mut state.weekly } else { &mut state.short };

        if let Some(v) = req_limit {
            target.requests.limit = Some(v);
        }
        if let Some(v) = req_remaining {
            target.requests.remaining = Some(v);
        }
        if let Some(secs) = req_reset_secs {
            target.requests.resets_at = Some(now + secs);
        }

        if let Some(v) = tok_limit {
            target.tokens.limit = Some(v);
        }
        if let Some(v) = tok_remaining {
            target.tokens.remaining = Some(v);
        }
        if let Some(secs) = tok_reset_secs {
            target.tokens.resets_at = Some(now + secs);
        }

        state.updated_at = now;
    }

    /// Get the current usage snapshot.
    pub async fn snapshot(&self) -> UsageSnapshot {
        self.state.read().await.clone()
    }

    /// Compute a fallback cooldown duration from the tracked headers when
    /// the provider returns 429 but no `Retry-After` header.
    ///
    /// Uses the longest reset across both the short and weekly windows
    /// (requests and tokens). Returns `None` if no reset data is available.
    pub async fn fallback_cooldown_secs(&self) -> Option<u64> {
        let state = self.state.read().await;
        let resets = [
            state.short.requests.reset_in_secs(),
            state.short.tokens.reset_in_secs(),
            state.weekly.requests.reset_in_secs(),
            state.weekly.tokens.reset_in_secs(),
        ];
        resets.into_iter().flatten().max()
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

    if total_secs > 0 { Some(total_secs) } else { Some(1) }
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
        };
        assert_eq!(w.usage_percent(), Some(75.0));
    }

    #[test]
    fn test_usage_percent_full() {
        let w = RateLimitWindow {
            limit: Some(100),
            remaining: Some(0),
            resets_at: None,
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
}
