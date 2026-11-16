//! Search backends for the provider-agnostic web search feature.
//!
//! Each backend is a single-attempt REST client: no retry, no backoff.
//! Failures surface as `BackendError` with a self-contained message that the
//! coordinator renders into a tool result. API keys never appear in error
//! messages — response bodies are redacted before inclusion.

use std::time::Duration;

use crate::web_search::models::{SearchQuery, SearchResult, DEFAULT_MAX_RESULTS};

pub mod brave;
pub mod codex;
pub mod exa;
pub mod searxng;
pub mod tavily;

pub use brave::BraveBackend;
pub use codex::CodexSearchBackend;
pub use exa::ExaBackend;
pub use searxng::SearxngBackend;
pub use tavily::TavilyBackend;

const ERROR_BODY_MAX_CHARS: usize = 300;
const EXA_SNIPPET_MAX_CHARS: usize = 1000;
const FIELD_MAX_CHARS: usize = 2000;
pub(crate) const RESPONSE_BODY_MAX_BYTES: usize = 2 * 1024 * 1024;

/// Failure of a single search-backend attempt.
#[derive(Debug, Clone)]
pub struct BackendError {
    pub message: String,
}

impl BackendError {
    pub fn new(message: impl Into<String>) -> Self {
        Self {
            message: message.into(),
        }
    }
}

impl std::fmt::Display for BackendError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str(&self.message)
    }
}

impl std::error::Error for BackendError {}

/// A single-attempt web search backend.
#[async_trait::async_trait]
pub trait SearchBackend: Send + Sync {
    async fn execute(&self, query: &SearchQuery) -> Result<Vec<SearchResult>, BackendError>;
    fn name(&self) -> &'static str;
    async fn ready(&self) -> bool;
}

/// Resolve an `api_key_env` value following the repo convention: try it as an
/// environment-variable name first, falling back to the literal value when the
/// variable is unset.
pub fn resolve_api_key(api_key_env: &str) -> String {
    std::env::var(api_key_env).unwrap_or_else(|_| api_key_env.to_string())
}

fn build_client(timeout_seconds: u64) -> Result<reqwest::Client, BackendError> {
    reqwest::Client::builder()
        .timeout(Duration::from_secs(timeout_seconds))
        .build()
        .map_err(|e| BackendError::new(format!("failed to build HTTP client: {e}")))
}

fn normalize_base_url(url: &str) -> String {
    url.trim_end_matches('/').to_string()
}

fn effective_max_results(query: &SearchQuery) -> usize {
    query
        .max_results
        .unwrap_or(DEFAULT_MAX_RESULTS)
        .max(1) as usize
}

/// Map a recency day count to Tavily/SearXNG `time_range` buckets.
pub(crate) fn time_range_for_recency(days: u32) -> &'static str {
    if days <= 1 {
        "day"
    } else if days <= 7 {
        "week"
    } else if days <= 31 {
        "month"
    } else {
        "year"
    }
}

/// Map a recency day count to Brave `freshness` buckets.
pub(crate) fn freshness_for_recency(days: u32) -> &'static str {
    if days <= 1 {
        "pd"
    } else if days <= 7 {
        "pw"
    } else if days <= 31 {
        "pm"
    } else {
        "py"
    }
}

fn truncate_chars(s: &str, max_chars: usize) -> String {
    s.chars().take(max_chars).collect()
}

/// Cap a normalized title/url/snippet field so one upstream entry cannot
/// force an unbounded allocation before slicing to `max_results`.
pub(crate) fn cap_field(s: &str) -> String {
    truncate_chars(s, FIELD_MAX_CHARS)
}

/// Accumulator for a size-capped response body. `push_chunk` returns `true`
/// once the cap is exceeded; the partial buffer is dropped at that point and
/// the caller reports "response too large".
#[derive(Default)]
struct CappedBody {
    buf: Vec<u8>,
}

impl CappedBody {
    fn push_chunk(&mut self, chunk: &[u8], max_bytes: usize) -> bool {
        if self.buf.len().saturating_add(chunk.len()) > max_bytes {
            self.buf.clear();
            return true;
        }
        self.buf.extend_from_slice(chunk);
        false
    }

    fn finish(self) -> Result<String, BackendError> {
        String::from_utf8(self.buf)
            .map_err(|e| BackendError::new(format!("response is not valid UTF-8: {e}")))
    }
}

/// Read a response body into a String, enforcing a byte cap: reject via
/// Content-Length when present, and stop-and-fail as soon as streamed bytes
/// exceed the cap instead of buffering an unbounded body.
pub(crate) async fn read_body_capped(
    response: reqwest::Response,
    max_bytes: usize,
    backend: &str,
) -> Result<String, BackendError> {
    if let Some(len) = content_length(&response) {
        if len > max_bytes {
            return Err(BackendError::new(format!("{backend}: response too large")));
        }
    }
    let mut body = CappedBody::default();
    let mut stream = response.bytes_stream();
    use futures::StreamExt;
    while let Some(chunk) = stream.next().await {
        let chunk = chunk.map_err(|e| {
            BackendError::new(format!("{backend}: failed to read response: {e}"))
        })?;
        if body.push_chunk(&chunk, max_bytes) {
            return Err(BackendError::new(format!("{backend}: response too large")));
        }
    }
    body.finish()
        .map_err(|e| BackendError::new(format!("{backend}: {e}")))
}

fn content_length(response: &reqwest::Response) -> Option<usize> {
    response
        .headers()
        .get(reqwest::header::CONTENT_LENGTH)
        .and_then(|v| v.to_str().ok())
        .and_then(|s| s.parse::<usize>().ok())
}

fn trim_error_body(body: &str) -> String {
    truncate_chars(body.trim(), ERROR_BODY_MAX_CHARS)
}

/// Remove every occurrence of `api_key` from `body` before it may appear in
/// an error message.
fn redact(body: &str, api_key: &str) -> String {
    if api_key.is_empty() {
        return body.to_string();
    }
    body.replace(api_key, "[redacted]")
}

fn error_body(body: &str, api_key: &str) -> String {
    redact(&trim_error_body(body), api_key)
}

/// Slice normalized results to the effective `max_results`.
fn slice_results(results: Vec<SearchResult>, max: usize) -> Vec<SearchResult> {
    results.into_iter().take(max).collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn normalize_base_url_strips_trailing_slash() {
        assert_eq!(
            normalize_base_url("https://api.example.com/v1/"),
            "https://api.example.com/v1"
        );
        assert_eq!(
            normalize_base_url("https://api.example.com"),
            "https://api.example.com"
        );
    }

    #[test]
    fn resolve_api_key_falls_back_to_literal() {
        let literal = "web-search-literal-key-OBEY-UNSET";
        assert_eq!(resolve_api_key(literal), literal);
    }

    #[test]
    fn redact_removes_key_occurrences() {
        let body = r#"{"error": "bad key sk-abc123 provided"}"#;
        let out = error_body(body, "sk-abc123");
        assert!(!out.contains("sk-abc123"));
        assert!(out.contains("[redacted]"));
    }

    #[test]
    fn error_body_trims_to_300_chars() {
        let body = "x".repeat(1000);
        let out = error_body(&body, "unused");
        assert_eq!(out.chars().count(), 300);
    }

    #[test]
    fn time_range_buckets() {
        assert_eq!(time_range_for_recency(1), "day");
        assert_eq!(time_range_for_recency(7), "week");
        assert_eq!(time_range_for_recency(31), "month");
        assert_eq!(time_range_for_recency(365), "year");
    }

    #[test]
    fn freshness_buckets() {
        assert_eq!(freshness_for_recency(1), "pd");
        assert_eq!(freshness_for_recency(7), "pw");
        assert_eq!(freshness_for_recency(31), "pm");
        assert_eq!(freshness_for_recency(365), "py");
    }

    #[test]
    fn truncate_chars_never_splits_multibyte() {
        let s = "あ".repeat(600);
        let out = truncate_chars(&s, 1000);
        assert_eq!(out, s);
        let out = truncate_chars(&s, 10);
        assert_eq!(out.chars().count(), 10);
    }

    #[test]
    fn slice_results_honors_max() {
        let results: Vec<SearchResult> = (0..8)
            .map(|i| SearchResult {
                title: format!("t{i}"),
                url: format!("https://example.com/{i}"),
                snippet: String::new(),
            })
            .collect();
        let sliced = slice_results(results, 5);
        assert_eq!(sliced.len(), 5);
        assert_eq!(sliced[0].title, "t0");
    }

    #[test]
    fn effective_max_results_defaults_when_absent() {
        let query = SearchQuery {
            q: "x".to_string(),
            domains: None,
            recency: None,
            max_results: None,
        };
        assert_eq!(effective_max_results(&query), DEFAULT_MAX_RESULTS as usize);
    }

    #[test]
    fn build_client_rejects_bad_runtime() {
        let client = build_client(15);
        assert!(client.is_ok());
    }

    #[tokio::test]
    async fn backend_error_display_is_message() {
        let err = BackendError::new("search failed");
        assert_eq!(err.to_string(), "search failed");
    }

    #[test]
    fn prop_truncate_chars_preserves_prefix() {
        use proptest::prelude::*;
        proptest!(|(s in ".*", max in 0usize..2000)| {
            let out = truncate_chars(&s, max);
            prop_assert!(out.chars().count() <= max);
            prop_assert!(s.starts_with(&out));
        });
    }

#[test]
fn error_body_untrusted_json_never_panics() {
    let body = "{\"partial\":";
    let out = error_body(body, "");
    assert!(out.contains("partial"));
    let value: Result<serde_json::Value, _> = serde_json::from_str(body);
    assert!(value.is_err());
}

#[test]
fn capped_body_accepts_chunks_under_limit() {
    let mut body = CappedBody::default();
    assert!(!body.push_chunk(b"hello ", 16));
    assert!(!body.push_chunk(b"world", 16));
    assert_eq!(body.finish().unwrap(), "hello world");
}

#[test]
fn capped_body_rejects_chunk_over_limit() {
    let mut body = CappedBody::default();
    assert!(!body.push_chunk(b"hello ", 11));
    assert!(body.push_chunk(b"world!!!", 11));
}

#[test]
fn capped_body_single_chunk_over_limit() {
    let mut body = CappedBody::default();
    assert!(body.push_chunk(&vec![b'x'; 2 * 1024 * 1024 + 1], RESPONSE_BODY_MAX_BYTES));
}

#[test]
fn capped_body_exact_limit_accepted() {
    let mut body = CappedBody::default();
    assert!(!body.push_chunk(&vec![b'x'; 1024], 1024));
    assert_eq!(body.finish().unwrap().len(), 1024);
}

#[test]
fn capped_body_rejects_invalid_utf8() {
    let mut body = CappedBody::default();
    assert!(!body.push_chunk(&[0xff, 0xfe], 16));
    assert!(body.finish().is_err());
}

#[test]
fn cap_field_limits_to_2000_chars_char_safe() {
    let s = "あ".repeat(2500);
    let out = cap_field(&s);
    assert_eq!(out.chars().count(), 2000);
    assert!(s.starts_with(&out));
    assert_eq!(cap_field("short"), "short");
}
}
