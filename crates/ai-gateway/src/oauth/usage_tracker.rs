//! Tracks OpenAI rate-limit usage for OAuth (browser-login) providers.
//!
//! OpenAI returns `x-ratelimit-*` headers on every response. For ChatGPT Plus
//! / Pro browser-login accounts, these manifest as sliding-window limits
//! (typically a short window like 3-5h and a weekly window). This module
//! captures those headers and exposes a snapshot for the admin UI.

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

/// Snapshot of all tracked rate-limit windows for the OAuth provider.
#[derive(Debug, Clone, Serialize, Default)]
pub struct UsageSnapshot {
    pub requests: RateLimitWindow,
    pub tokens: RateLimitWindow,
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
    pub async fn update_from_headers(&self, headers: &reqwest::header::HeaderMap) {
        let now = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .map(|d| d.as_secs())
            .unwrap_or(0);

        let mut state = self.state.write().await;

        // Requests window
        if let Some(v) = header_u64(headers, "x-ratelimit-limit-requests") {
            state.requests.limit = Some(v);
        }
        if let Some(v) = header_u64(headers, "x-ratelimit-remaining-requests") {
            state.requests.remaining = Some(v);
        }
        if let Some(reset_str) = header_str(headers, "x-ratelimit-reset-requests") {
            if let Some(secs) = parse_reset_duration(reset_str) {
                state.requests.resets_at = Some(now + secs);
            }
        }

        // Tokens window
        if let Some(v) = header_u64(headers, "x-ratelimit-limit-tokens") {
            state.tokens.limit = Some(v);
        }
        if let Some(v) = header_u64(headers, "x-ratelimit-remaining-tokens") {
            state.tokens.remaining = Some(v);
        }
        if let Some(reset_str) = header_str(headers, "x-ratelimit-reset-tokens") {
            if let Some(secs) = parse_reset_duration(reset_str) {
                state.tokens.resets_at = Some(now + secs);
            }
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
    /// Uses the longer of the two reset windows (requests vs tokens).
    /// Returns `None` if no reset data is available.
    pub async fn fallback_cooldown_secs(&self) -> Option<u64> {
        let state = self.state.read().await;
        let req_reset = state.requests.reset_in_secs();
        let tok_reset = state.tokens.reset_in_secs();
        match (req_reset, tok_reset) {
            (Some(r), Some(t)) => Some(r.max(t)),
            (Some(r), None) => Some(r),
            (None, Some(t)) => Some(t),
            (None, None) => None,
        }
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
}
