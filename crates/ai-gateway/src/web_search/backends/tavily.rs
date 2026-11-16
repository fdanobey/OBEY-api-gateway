//! Tavily search backend.
//!
//! POST https://api.tavily.com/search with the API key in the JSON body
//! (`api_key` field, classic Tavily API). Recency has no native parameter —
//! it is dropped.

use async_trait::async_trait;
use serde_json::{json, Value};

use crate::web_search::backends::{
    build_client, cap_field, effective_max_results, error_body, normalize_base_url,
    read_body_capped, BackendError, SearchBackend, RESPONSE_BODY_MAX_BYTES,
};
use crate::web_search::models::{SearchQuery, SearchResult};

pub const DEFAULT_ENDPOINT: &str = "https://api.tavily.com/search";

pub struct TavilyBackend {
    http: reqwest::Client,
    api_key: String,
    endpoint: String,
}

/// Request parts (pure, testable): (endpoint, JSON body).
fn build_request_parts(api_key: &str, endpoint: &str, query: &SearchQuery) -> (String, Value) {
    let mut body = json!({
        "api_key": api_key,
        "query": query.q,
        "search_depth": "basic",
        "max_results": query.max_results.unwrap_or(crate::web_search::models::DEFAULT_MAX_RESULTS),
    });
    if let Some(domains) = &query.domains {
        if !domains.is_empty() {
            body["include_domains"] = json!(domains);
        }
    }
    (endpoint.to_string(), body)
}

/// Normalize a Tavily response body into search results. String fields are
/// capped during normalization and parsing stops after `max_results`
/// entries, so oversized upstream arrays never allocate in full.
fn normalize_response(body: &str, max_results: usize) -> Result<Vec<SearchResult>, BackendError> {
    let value: Value = serde_json::from_str(body)
        .map_err(|e| BackendError::new(format!("tavily: invalid JSON response: {e}")))?;
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
impl SearchBackend for TavilyBackend {
    async fn execute(&self, query: &SearchQuery) -> Result<Vec<SearchResult>, BackendError> {
        let (endpoint, body) = build_request_parts(&self.api_key, &self.endpoint, query);
        let response = self
            .http
            .post(&endpoint)
            .header(reqwest::header::CONTENT_TYPE, "application/json")
            .json(&body)
            .send()
            .await
            .map_err(|e| BackendError::new(format!("tavily: request failed: {e}")))?;

        let status = response.status();
        let text = read_body_capped(response, RESPONSE_BODY_MAX_BYTES, "tavily").await?;

        if !status.is_success() {
            return Err(BackendError::new(format!(
                "tavily: HTTP {}: {}",
                status.as_u16(),
                error_body(&text, &self.api_key)
            )));
        }

        let results = normalize_response(&text, effective_max_results(query))?;
        Ok(results)
    }

    fn name(&self) -> &'static str {
        "tavily"
    }

    async fn ready(&self) -> bool {
        !self.api_key.is_empty()
    }
}

impl TavilyBackend {
    pub fn new(
        api_key: String,
        base_url: Option<String>,
        timeout_seconds: u64,
    ) -> Result<Self, BackendError> {
        Ok(Self {
            http: build_client(timeout_seconds)?,
            endpoint: base_url
                .map(|url| normalize_base_url(&url))
                .unwrap_or_else(|| DEFAULT_ENDPOINT.to_string()),
            api_key,
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
        let (endpoint, body) = build_request_parts("tvly-key", DEFAULT_ENDPOINT, &query());
        assert_eq!(endpoint, "https://api.tavily.com/search");
        assert_eq!(body["query"], "rust async");
        assert_eq!(body["api_key"], "tvly-key");
        assert_eq!(body["search_depth"], "basic");
        assert_eq!(body["max_results"], 5);
        assert!(body.get("include_domains").is_none());
    }

    #[test]
    fn request_parts_maps_domains() {
        let mut q = query();
        q.domains = Some(vec!["docs.rs".to_string(), "crates.io".to_string()]);
        let (_, body) = build_request_parts("k", DEFAULT_ENDPOINT, &q);
        assert_eq!(body["include_domains"], json!(["docs.rs", "crates.io"]));
    }

    #[test]
    fn request_parts_drops_recency() {
        let mut q = query();
        q.recency = Some(7);
        let (_, body) = build_request_parts("k", DEFAULT_ENDPOINT, &q);
        assert!(body.get("topic").is_none());
        assert!(body.get("days").is_none());
    }

    #[test]
    fn request_parts_default_max_results_when_absent() {
        let mut q = query();
        q.max_results = None;
        let (_, body) = build_request_parts("k", DEFAULT_ENDPOINT, &q);
        assert_eq!(
            body["max_results"],
            json!(crate::web_search::models::DEFAULT_MAX_RESULTS)
        );
    }

    #[test]
    fn base_url_override_replaces_endpoint() {
        let backend = TavilyBackend::new(
            "k".to_string(),
            Some("https://proxy.example.com/tavily/".to_string()),
            15,
        )
        .unwrap();
        assert_eq!(backend.endpoint, "https://proxy.example.com/tavily");
    }

    #[test]
    fn normalize_fixture() {
        let fixture = r#"{
  "results": [
    {"title": "Async Rust", "url": "https://tokio.rs", "content": "Async runtime"},
    {"title": "Book", "url": "https://doc.rust-lang.org", "content": "The book"}
  ]
}"#;
        let results = normalize_response(fixture, MAX).unwrap();
        assert_eq!(results.len(), 2);
        assert_eq!(results[0].title, "Async Rust");
        assert_eq!(results[0].url, "https://tokio.rs");
        assert_eq!(results[0].snippet, "Async runtime");
        assert_eq!(results[1].url, "https://doc.rust-lang.org");
    }

    #[test]
    fn normalize_skips_missing_url_or_title() {
        let fixture = r#"{
  "results": [
    {"title": "No URL", "content": "c"},
    {"url": "https://a.example"},
    {"title": 42, "url": "https://b.example", "content": "c"},
    {"title": "OK", "url": "https://c.example"}
  ]
}"#;
        let results = normalize_response(fixture, MAX).unwrap();
        assert_eq!(results.len(), 3);
        assert_eq!(results[0].url, "https://a.example");
        assert_eq!(results[0].title, "");
        assert_eq!(results[1].url, "https://b.example");
        assert_eq!(results[1].title, "");
        assert_eq!(results[2].url, "https://c.example");
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
        let fixture = json!({
            "results": [{"title": long, "url": "https://a.example", "content": long}]
        })
        .to_string();
        let results = normalize_response(&fixture, MAX).unwrap();
        assert_eq!(results[0].title.chars().count(), 2000);
        assert_eq!(results[0].snippet.chars().count(), 2000);
    }

    #[test]
    fn normalize_malformed_json_is_error() {
        let err = normalize_response("{not json", MAX).unwrap_err();
        assert!(err.message.contains("invalid JSON"));
    }

    #[test]
    fn normalize_missing_results_array_returns_empty() {
        assert!(normalize_response(r#"{"answer": "hi"}"#, MAX)
            .unwrap()
            .is_empty());
        assert!(normalize_response(r#"{"results": "nope"}"#, MAX)
            .unwrap()
            .is_empty());
    }

    #[test]
    fn constructor_builds_client_with_timeout() {
        let backend = TavilyBackend::new("k".to_string(), None, 10).unwrap();
        assert_eq!(backend.endpoint, DEFAULT_ENDPOINT);
    }

    #[tokio::test]
    async fn ready_true_when_key_nonempty() {
        let backend = TavilyBackend::new("k".to_string(), None, 15).unwrap();
        assert!(backend.ready().await);
    }

    #[tokio::test]
    async fn ready_false_when_key_empty() {
        let backend = TavilyBackend::new(String::new(), None, 15).unwrap();
        assert!(!backend.ready().await);
    }

    #[test]
    fn name_is_tavily() {
        let backend = TavilyBackend::new("k".to_string(), None, 15).unwrap();
        assert_eq!(backend.name(), "tavily");
    }
}
