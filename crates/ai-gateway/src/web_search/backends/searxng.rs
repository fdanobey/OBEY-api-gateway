//! SearXNG backend.
//!
//! GET {base_url}/search with `format=json`. No API key. `domains` has no
//! single-request equivalent in the JSON format and is dropped. If the
//! instance rejects `format=json` (403 with a body mentioning the format
//! parameter), the error says so honestly.

use async_trait::async_trait;
use serde_json::Value;

use crate::web_search::backends::{
    build_client, cap_field, effective_max_results, error_body, normalize_base_url,
    read_body_capped, time_range_for_recency, BackendError, SearchBackend,
    RESPONSE_BODY_MAX_BYTES,
};
use crate::web_search::models::{SearchQuery, SearchResult};

pub struct SearxngBackend {
    http: reqwest::Client,
    base_url: String,
}

/// Request parts (pure, testable): (full URL, query params).
fn build_request_parts(base_url: &str, query: &SearchQuery) -> (String, Vec<(String, String)>) {
    let base = normalize_base_url(base_url);
    let url = format!("{base}/search");
    let mut params = vec![
        ("q".to_string(), query.q.clone()),
        ("format".to_string(), "json".to_string()),
        ("safesearch".to_string(), "1".to_string()),
    ];
    if let Some(days) = query.recency {
        params.push(("time_range".to_string(), time_range_for_recency(days).to_string()));
    }
    (url, params)
}

/// Normalize a SearXNG JSON response body into search results. String
/// fields are capped during normalization and parsing stops after
/// `max_results` entries, so oversized upstream arrays never allocate in
/// full.
fn normalize_response(body: &str, max_results: usize) -> Result<Vec<SearchResult>, BackendError> {
    let value: Value = serde_json::from_str(body)
        .map_err(|e| BackendError::new(format!("searxng: invalid JSON response: {e}")))?;
    let Some(results) = value.get("results").and_then(Value::as_array) else {
        return Ok(Vec::new());
    };
    let mut normalized: Vec<SearchResult> = Vec::with_capacity(max_results.min(results.len()));
    for item in results {
        if normalized.len() >= max_results {
            break;
        }
        let Some(url) = item.get("url").and_then(Value::as_str) else {
            continue;
        };
        normalized.push(SearchResult {
            title: cap_field(item.get("title").and_then(Value::as_str).unwrap_or_default()),
            url: cap_field(url),
            snippet: cap_field(
                item
                    .get("content")
                    .and_then(Value::as_str)
                    .unwrap_or_default(),
            ),
        });
    }
    Ok(normalized)
}

#[async_trait]
impl SearchBackend for SearxngBackend {
    async fn execute(&self, query: &SearchQuery) -> Result<Vec<SearchResult>, BackendError> {
        let (url, params) = build_request_parts(&self.base_url, query);
        let response = self
            .http
            .get(&url)
            .query(&params)
            .send()
            .await
            .map_err(|e| BackendError::new(format!("searxng: request failed: {e}")))?;

        let status = response.status();
        let text = read_body_capped(response, RESPONSE_BODY_MAX_BYTES, "searxng").await?;

        if !status.is_success() {
            if status.as_u16() == 403 && text.to_lowercase().contains("format") {
                return Err(BackendError::new(
                    "searxng: instance rejected format=json (JSON output not enabled on this instance)"
                ));
            }
            return Err(BackendError::new(format!(
                "searxng: HTTP {}: {}",
                status.as_u16(),
                error_body(&text, "")
            )));
        }

        let results = normalize_response(&text, effective_max_results(query))?;
        Ok(results)
    }

    fn name(&self) -> &'static str {
        "searxng"
    }

    async fn ready(&self) -> bool {
        true
    }
}

impl SearxngBackend {
    pub fn new(base_url: String, timeout_seconds: u64) -> Result<Self, BackendError> {
        Ok(Self {
            http: build_client(timeout_seconds)?,
            base_url: normalize_base_url(&base_url),
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    const MAX: usize = 5;

    fn query() -> SearchQuery {
        SearchQuery {
            q: "rust async".to_string(),
            domains: None,
            recency: None,
            max_results: Some(5),
        }
    }

    #[test]
    fn request_parts_basic() {
        let (url, params) = build_request_parts("http://localhost:8888/", &query());
        assert_eq!(url, "http://localhost:8888/search");
        assert!(params.contains(&("q".to_string(), "rust async".to_string())));
        assert!(params.contains(&("format".to_string(), "json".to_string())));
        assert!(params.contains(&("safesearch".to_string(), "1".to_string())));
        assert!(!params.iter().any(|(k, _)| k == "time_range"));
    }

    #[test]
    fn request_parts_maps_recency_to_time_range() {
        let mut q = query();
        q.recency = Some(30);
        let (_, params) = build_request_parts("http://localhost:8888", &q);
        assert!(params.contains(&("time_range".to_string(), "month".to_string())));
    }

    #[test]
    fn request_parts_drops_domains() {
        let mut q = query();
        q.domains = Some(vec!["example.com".to_string()]);
        let (_, params) = build_request_parts("http://localhost:8888", &q);
        assert!(!params.iter().any(|(k, _)| k == "sites" || k == "domains"));
    }

    #[test]
    fn constructor_normalizes_base_url() {
        let backend = SearxngBackend::new("http://localhost:8888///".to_string(), 15).unwrap();
        assert_eq!(backend.base_url, "http://localhost:8888");
    }

    #[test]
    fn normalize_fixture() {
        let fixture = r#"{
  "results": [
    {"title": "SearXNG hit", "url": "https://searx.example/a", "content": "Snippet"},
    {"title": "Another", "url": "https://searx.example/b", "content": "Text"}
  ],
  "answers": ["42"]
}"#;
        let results = normalize_response(fixture, MAX).unwrap();
        assert_eq!(results.len(), 2);
        assert_eq!(results[0].title, "SearXNG hit");
        assert_eq!(results[0].snippet, "Snippet");
    }

    #[test]
    fn normalize_skips_missing_url() {
        let fixture = r#"{"results": [{"title": "t", "content": "c"}]}"#;
        assert!(normalize_response(fixture, MAX).unwrap().is_empty());
    }

    #[test]
    fn normalize_stops_after_max_results_entries() {
        let entries: Vec<String> = (0..100)
            .map(|i| format!(r#"{{"title": "t{i}", "url": "https://e.example/{i}"}}"#))
            .collect();
        let fixture = format!(r#"{{"results": [{}]}}"#, entries.join(","));
        let results = normalize_response(&fixture, 5).unwrap();
        assert_eq!(results.len(), 5);
        assert_eq!(results[4].url, "https://e.example/4");
    }

    #[test]
    fn normalize_caps_long_fields_to_2000_chars() {
        let long = "x".repeat(5000);
        let fixture = serde_json::json!({
            "results": [{"title": long, "url": "https://a.example", "content": long}]
        })
        .to_string();
        let results = normalize_response(&fixture, MAX).unwrap();
        assert_eq!(results[0].title.chars().count(), 2000);
        assert_eq!(results[0].snippet.chars().count(), 2000);
    }

    #[test]
    fn normalize_malformed_json_is_error() {
        let err = normalize_response("<html>not json</html>", MAX).unwrap_err();
        assert!(err.message.contains("invalid JSON"));
    }

    #[test]
    fn normalize_missing_results_returns_empty() {
        assert!(normalize_response(r#"{"answers": []}"#, MAX)
            .unwrap()
            .is_empty());
    }

    #[tokio::test]
    async fn ready_is_always_true() {
        let backend = SearxngBackend::new("http://localhost:8888".to_string(), 15).unwrap();
        assert!(backend.ready().await);
    }

    #[test]
    fn name_is_searxng() {
        let backend = SearxngBackend::new("http://localhost:8888".to_string(), 15).unwrap();
        assert_eq!(backend.name(), "searxng");
    }
}
