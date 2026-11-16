//! Exa search backend.
//!
//! POST https://api.exa.ai/search with the `x-api-key` header. Recency maps
//! to `startPublishedDate` (days-ago ISO date); domains map to
//! `includeDomains`.

use async_trait::async_trait;
use serde_json::{json, Value};

use crate::web_search::backends::{
    build_client, cap_field, effective_max_results, error_body, normalize_base_url,
    read_body_capped, truncate_chars, BackendError, SearchBackend, EXA_SNIPPET_MAX_CHARS,
    RESPONSE_BODY_MAX_BYTES,
};
use crate::web_search::models::{SearchQuery, SearchResult};

pub const DEFAULT_ENDPOINT: &str = "https://api.exa.ai/search";

pub struct ExaBackend {
    http: reqwest::Client,
    api_key: String,
    endpoint: String,
}

fn days_ago_iso_date(days: u32) -> String {
    let date = chrono::Utc::now() - chrono::Duration::days(days as i64);
    date.format("%Y-%m-%d").to_string()
}

/// Request parts (pure, testable): (endpoint, JSON body).
fn build_request_parts(endpoint: &str, query: &SearchQuery) -> (String, Value) {
    let mut body = json!({
        "query": query.q,
        "numResults": query.max_results.unwrap_or(crate::web_search::models::DEFAULT_MAX_RESULTS),
    });
    if let Some(domains) = &query.domains {
        if !domains.is_empty() {
            body["includeDomains"] = json!(domains);
        }
    }
    if let Some(days) = query.recency {
        body["startPublishedDate"] = json!(days_ago_iso_date(days));
    }
    (endpoint.to_string(), body)
}

/// Normalize an Exa response body into search results. Title/url/snippet
/// are capped during normalization and parsing stops after `max_results`
/// entries, so oversized upstream arrays never allocate in full.
fn normalize_response(body: &str, max_results: usize) -> Result<Vec<SearchResult>, BackendError> {
    let value: Value = serde_json::from_str(body)
        .map_err(|e| BackendError::new(format!("exa: invalid JSON response: {e}")))?;
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
            snippet: truncate_chars(
                item.get("text").and_then(Value::as_str).unwrap_or_default(),
                EXA_SNIPPET_MAX_CHARS,
            ),
        });
    }
    Ok(normalized)
}

#[async_trait]
impl SearchBackend for ExaBackend {
    async fn execute(&self, query: &SearchQuery) -> Result<Vec<SearchResult>, BackendError> {
        let (endpoint, body) = build_request_parts(&self.endpoint, query);
        let response = self
            .http
            .post(&endpoint)
            .header("x-api-key", &self.api_key)
            .header(reqwest::header::CONTENT_TYPE, "application/json")
            .json(&body)
            .send()
            .await
            .map_err(|e| BackendError::new(format!("exa: request failed: {e}")))?;

        let status = response.status();
        let text = read_body_capped(response, RESPONSE_BODY_MAX_BYTES, "exa").await?;

        if !status.is_success() {
            return Err(BackendError::new(format!(
                "exa: HTTP {}: {}",
                status.as_u16(),
                error_body(&text, &self.api_key)
            )));
        }

        let results = normalize_response(&text, effective_max_results(query))?;
        Ok(results)
    }

    fn name(&self) -> &'static str {
        "exa"
    }

    async fn ready(&self) -> bool {
        !self.api_key.is_empty()
    }
}

impl ExaBackend {
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

    const MAX: usize = 4;

    fn query() -> SearchQuery {
        SearchQuery {
            q: "rust async".to_string(),
            domains: None,
            recency: None,
            max_results: Some(4),
        }
    }

    #[test]
    fn request_parts_basic() {
        let (endpoint, body) = build_request_parts(DEFAULT_ENDPOINT, &query());
        assert_eq!(endpoint, DEFAULT_ENDPOINT);
        assert_eq!(body["query"], "rust async");
        assert_eq!(body["numResults"], 4);
        assert!(body.get("includeDomains").is_none());
        assert!(body.get("startPublishedDate").is_none());
    }

    #[test]
    fn request_parts_maps_domains() {
        let mut q = query();
        q.domains = Some(vec!["exa.example".to_string()]);
        let (_, body) = build_request_parts(DEFAULT_ENDPOINT, &q);
        assert_eq!(body["includeDomains"], json!(["exa.example"]));
    }

    #[test]
    fn request_parts_maps_recency_to_start_published_date() {
        let mut q = query();
        q.recency = Some(7);
        let (_, body) = build_request_parts(DEFAULT_ENDPOINT, &q);
        let date = body["startPublishedDate"].as_str().unwrap();
        let parsed = chrono::NaiveDate::parse_from_str(date, "%Y-%m-%d");
        assert!(parsed.is_ok());
        let expected = chrono::Utc::now() - chrono::Duration::days(7);
        let diff = (parsed.unwrap() - expected.date_naive()).num_days().abs();
        assert!(diff <= 1, "date {date} should be ~7 days ago");
    }

    #[test]
    fn request_parts_no_recency_leaves_date_absent() {
        let (_, body) = build_request_parts(DEFAULT_ENDPOINT, &query());
        assert!(body.get("startPublishedDate").is_none());
    }

    #[test]
    fn base_url_override_replaces_endpoint() {
        let backend = ExaBackend::new(
            "k".to_string(),
            Some("https://proxy.example.com/exa/".to_string()),
            15,
        )
        .unwrap();
        assert_eq!(backend.endpoint, "https://proxy.example.com/exa");
    }

    #[test]
    fn normalize_fixture() {
        let fixture = r#"{
  "results": [
    {"title": "Exa hit", "url": "https://exa.example/a", "text": "Body text"},
    {"title": "Second", "url": "https://exa.example/b", "text": "More"}
  ]
}"#;
        let results = normalize_response(fixture, MAX).unwrap();
        assert_eq!(results.len(), 2);
        assert_eq!(results[0].snippet, "Body text");
        assert_eq!(results[1].title, "Second");
    }

    #[test]
    fn normalize_truncates_long_text_to_1000_chars() {
        let long = "x".repeat(2500);
        let fixture = json!({
            "results": [{"title": "t", "url": "https://exa.example/long", "text": long}]
        })
        .to_string();
        let results = normalize_response(&fixture, MAX).unwrap();
        assert_eq!(results[0].snippet.chars().count(), 1000);
    }

    #[test]
    fn normalize_caps_long_title_and_url_to_2000_chars() {
        let long = "x".repeat(5000);
        let long_url = format!("https://exa.example/{}", "p".repeat(5000));
        let fixture = json!({
            "results": [{"title": long, "url": long_url, "text": "t"}]
        })
        .to_string();
        let results = normalize_response(&fixture, MAX).unwrap();
        assert_eq!(results[0].title.chars().count(), 2000);
        assert!(results[0].url.chars().count() <= 2000);
        assert!(results[0].url.starts_with("https://exa.example/"));
    }

    #[test]
    fn normalize_stops_after_max_results_entries() {
        let entries: Vec<String> = (0..100)
            .map(|i| format!(r#"{{"title": "t{i}", "url": "https://e.example/{i}"}}"#))
            .collect();
        let fixture = format!(r#"{{"results": [{}]}}"#, entries.join(","));
        let results = normalize_response(&fixture, 4).unwrap();
        assert_eq!(results.len(), 4);
        assert_eq!(results[3].url, "https://e.example/3");
    }

    #[test]
    fn normalize_missing_text_is_empty_snippet() {
        let fixture = r#"{"results": [{"title": "t", "url": "https://exa.example"}]}"#;
        let results = normalize_response(fixture, MAX).unwrap();
        assert_eq!(results[0].snippet, "");
    }

    #[test]
    fn normalize_skips_missing_url() {
        let fixture = r#"{"results": [{"title": "no url", "text": "t"}]}"#;
        let results = normalize_response(fixture, MAX).unwrap();
        assert!(results.is_empty());
    }

    #[test]
    fn normalize_malformed_json_is_error() {
        let err = normalize_response("nope", MAX).unwrap_err();
        assert!(err.message.contains("invalid JSON"));
    }

    #[test]
    fn normalize_missing_results_returns_empty() {
        assert!(normalize_response(r#"{"requestId": "x"}"#, MAX)
            .unwrap()
            .is_empty());
    }

    #[tokio::test]
    async fn ready_true_when_key_nonempty() {
        let backend = ExaBackend::new("k".to_string(), None, 15).unwrap();
        assert!(backend.ready().await);
    }

    #[tokio::test]
    async fn ready_false_when_key_empty() {
        let backend = ExaBackend::new(String::new(), None, 15).unwrap();
        assert!(!backend.ready().await);
    }

    #[test]
    fn name_is_exa() {
        let backend = ExaBackend::new("k".to_string(), None, 15).unwrap();
        assert_eq!(backend.name(), "exa");
    }
}
