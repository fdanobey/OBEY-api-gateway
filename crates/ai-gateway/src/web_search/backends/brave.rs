//! Brave Search API backend.
//!
//! GET https://api.search.brave.com/res/v1/web/search with the
//! `X-Subscription-Token` header. Recency maps to `freshness`; `domains` has
//! no reliable single-request filter and is dropped.

use async_trait::async_trait;
use serde_json::Value;

use crate::web_search::backends::{
    build_client, cap_field, effective_max_results, error_body, freshness_for_recency,
    normalize_base_url, read_body_capped, BackendError, SearchBackend, RESPONSE_BODY_MAX_BYTES,
};
use crate::web_search::models::{SearchQuery, SearchResult};

pub const DEFAULT_ENDPOINT: &str = "https://api.search.brave.com/res/v1/web/search";

pub struct BraveBackend {
    http: reqwest::Client,
    api_key: String,
    endpoint: String,
}

/// Request parts (pure, testable): (endpoint, query params).
fn build_request_parts(
    endpoint: &str,
    query: &SearchQuery,
) -> (String, Vec<(String, String)>) {
    let mut params = vec![("q".to_string(), query.q.clone())];
    params.push((
        "count".to_string(),
        query
            .max_results
            .unwrap_or(crate::web_search::models::DEFAULT_MAX_RESULTS)
            .to_string(),
    ));
    if let Some(days) = query.recency {
        params.push(("freshness".to_string(), freshness_for_recency(days).to_string()));
    }
    (endpoint.to_string(), params)
}

/// Normalize a Brave response body into search results. String fields are
/// capped during normalization and parsing stops after `max_results`
/// entries, so oversized upstream arrays never allocate in full.
fn normalize_response(body: &str, max_results: usize) -> Result<Vec<SearchResult>, BackendError> {
    let value: Value = serde_json::from_str(body)
        .map_err(|e| BackendError::new(format!("brave: invalid JSON response: {e}")))?;
    let Some(results) = value
        .get("web")
        .and_then(|web| web.get("results"))
        .and_then(Value::as_array)
    else {
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
                    .get("description")
                    .and_then(Value::as_str)
                    .unwrap_or_default(),
            ),
        });
    }
    Ok(normalized)
}

#[async_trait]
impl SearchBackend for BraveBackend {
    async fn execute(&self, query: &SearchQuery) -> Result<Vec<SearchResult>, BackendError> {
        let (endpoint, params) = build_request_parts(&self.endpoint, query);
        let response = self
            .http
            .get(&endpoint)
            .header("X-Subscription-Token", &self.api_key)
            .header(reqwest::header::ACCEPT, "application/json")
            .query(&params)
            .send()
            .await
            .map_err(|e| BackendError::new(format!("brave: request failed: {e}")))?;

        let status = response.status();
        let text = read_body_capped(response, RESPONSE_BODY_MAX_BYTES, "brave").await?;

        if !status.is_success() {
            return Err(BackendError::new(format!(
                "brave: HTTP {}: {}",
                status.as_u16(),
                error_body(&text, &self.api_key)
            )));
        }

        let results = normalize_response(&text, effective_max_results(query))?;
        Ok(results)
    }

    fn name(&self) -> &'static str {
        "brave"
    }

    async fn ready(&self) -> bool {
        !self.api_key.is_empty()
    }
}

impl BraveBackend {
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
    use serde_json::json;

    const MAX: usize = 3;

    fn query() -> SearchQuery {
        SearchQuery {
            q: "rust async".to_string(),
            domains: None,
            recency: None,
            max_results: Some(3),
        }
    }

    #[test]
    fn request_parts_basic() {
        let (endpoint, params) = build_request_parts(DEFAULT_ENDPOINT, &query());
        assert_eq!(endpoint, DEFAULT_ENDPOINT);
        assert!(params.contains(&("q".to_string(), "rust async".to_string())));
        assert!(params.contains(&("count".to_string(), "3".to_string())));
        assert!(!params.iter().any(|(k, _)| k == "freshness"));
    }

    #[test]
    fn request_parts_maps_recency_to_freshness() {
        let mut q = query();
        q.recency = Some(7);
        let (_, params) = build_request_parts(DEFAULT_ENDPOINT, &q);
        assert!(params.contains(&("freshness".to_string(), "pw".to_string())));
    }

    #[test]
    fn request_parts_drops_domains() {
        let mut q = query();
        q.domains = Some(vec!["example.com".to_string()]);
        let (_, params) = build_request_parts(DEFAULT_ENDPOINT, &q);
        assert!(!params.iter().any(|(k, _)| k == "site" || k == "domains"));
    }

    #[test]
    fn request_parts_default_count_when_absent() {
        let mut q = query();
        q.max_results = None;
        let (_, params) = build_request_parts(DEFAULT_ENDPOINT, &q);
        assert!(params.contains(&(
            "count".to_string(),
            crate::web_search::models::DEFAULT_MAX_RESULTS.to_string()
        )));
    }

    #[test]
    fn base_url_override_replaces_endpoint() {
        let backend = BraveBackend::new(
            "k".to_string(),
            Some("https://proxy.example.com/brave//".to_string()),
            15,
        )
        .unwrap();
        assert_eq!(backend.endpoint, "https://proxy.example.com/brave");
    }

    #[test]
    fn normalize_fixture() {
        let fixture = r#"{
  "web": {
    "results": [
      {"title": "Tokio", "url": "https://tokio.rs", "description": "Runtime"},
      {"title": "Async book", "url": "https://rust-lang.github.io/async-book/", "description": "Book"}
    ]
  }
}"#;
        let results = normalize_response(fixture, MAX).unwrap();
        assert_eq!(results.len(), 2);
        assert_eq!(results[0].title, "Tokio");
        assert_eq!(results[0].snippet, "Runtime");
        assert_eq!(results[1].url, "https://rust-lang.github.io/async-book/");
    }

    #[test]
    fn normalize_skips_missing_url() {
        let fixture = r#"{
  "web": {
    "results": [
      {"title": "No URL", "description": "d"},
      {"title": "OK", "url": "https://ok.example"}
    ]
  }
}"#;
        let results = normalize_response(fixture, MAX).unwrap();
        assert_eq!(results.len(), 1);
        assert_eq!(results[0].url, "https://ok.example");
        assert_eq!(results[0].snippet, "");
    }

    #[test]
    fn normalize_stops_after_max_results_entries() {
        let entries: Vec<String> = (0..100)
            .map(|i| format!(r#"{{"title": "t{i}", "url": "https://e.example/{i}"}}"#))
            .collect();
        let fixture = format!(r#"{{"web": {{"results": [{}]}}}}"#, entries.join(","));
        let results = normalize_response(&fixture, 3).unwrap();
        assert_eq!(results.len(), 3);
        assert_eq!(results[2].url, "https://e.example/2");
    }

    #[test]
    fn normalize_caps_long_fields_to_2000_chars() {
        let long = "x".repeat(5000);
        let fixture = json!({
            "web": {"results": [{"title": long, "url": "https://a.example", "description": long}]}
        })
        .to_string();
        let results = normalize_response(&fixture, MAX).unwrap();
        assert_eq!(results[0].title.chars().count(), 2000);
        assert_eq!(results[0].snippet.chars().count(), 2000);
    }

    #[test]
    fn normalize_malformed_json_is_error() {
        let err = normalize_response("]]]", MAX).unwrap_err();
        assert!(err.message.contains("invalid JSON"));
    }

    #[test]
    fn normalize_missing_web_results_returns_empty() {
        assert!(normalize_response(r#"{"query": {"original": "x"}}"#, MAX)
            .unwrap()
            .is_empty());
    }

    #[tokio::test]
    async fn ready_true_when_key_nonempty() {
        let backend = BraveBackend::new("k".to_string(), None, 15).unwrap();
        assert!(backend.ready().await);
    }

    #[tokio::test]
    async fn ready_false_when_key_empty() {
        let backend = BraveBackend::new(String::new(), None, 15).unwrap();
        assert!(!backend.ready().await);
    }

    #[test]
    fn name_is_brave() {
        let backend = BraveBackend::new("k".to_string(), None, 15).unwrap();
        assert_eq!(backend.name(), "brave");
    }
}
