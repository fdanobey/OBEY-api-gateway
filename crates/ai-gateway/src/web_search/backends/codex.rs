//! Codex search backend: adapter over the existing `codex::search::SearchExecutor`.
//!
//! Mapping notes (documented by code shape in [`to_codex_args`]):
//! * `q`, `domains`, `recency` map 1:1 onto `CodexSearchArgs`.
//! * `max_results` has no Codex equivalent and is applied locally by slicing
//!   the normalized results; `response_length` is left unset (executor
//!   defaults to `short`).
//!
//! The executor returns a `ToolResult` whose `content` is the deterministic
//! text rendering produced by `normalize_search_response` (lines shaped
//! `[n] title - url (ref_id: x)` followed by an indented snippet line).
//! [`normalize_content`] parses that rendering back into `SearchResult`s.
//! Executor error messages are self-curated (no tokens, no keys) and are
//! forwarded verbatim as `BackendError`.

use std::sync::Arc;

use async_trait::async_trait;

use crate::codex::search::executor::SearchExecutor;
use crate::codex::search::models::CodexSearchArgs;
use crate::oauth::OAuthManager;
use crate::web_search::backends::{effective_max_results, slice_results, BackendError, SearchBackend};
use crate::web_search::models::{SearchQuery, SearchResult};

/// Readiness probe for the Codex backend. Mirrors the router's
/// `codex_search_ready` signal: a usable OAuth session must exist. The
/// config `enabled` half of that check stays with the coordinator, which
/// owns the config lock.
#[async_trait]
pub trait CodexSearchReadiness: Send + Sync {
    async fn ready(&self) -> bool;
}

/// Default readiness: an OAuth access token is retrievable — the same
/// underlying check the router uses before constructing the executor.
pub struct OAuthSessionReadiness {
    oauth: Arc<OAuthManager>,
}

impl OAuthSessionReadiness {
    pub fn new(oauth: Arc<OAuthManager>) -> Self {
        Self { oauth }
    }
}

#[async_trait]
impl CodexSearchReadiness for OAuthSessionReadiness {
    async fn ready(&self) -> bool {
        self.oauth.get_access_token().await.is_some()
    }
}

pub struct CodexSearchBackend {
    executor: Arc<SearchExecutor>,
    readiness: Arc<dyn CodexSearchReadiness>,
}

impl CodexSearchBackend {
    pub fn new(executor: Arc<SearchExecutor>, readiness: Arc<dyn CodexSearchReadiness>) -> Self {
        Self {
            executor,
            readiness,
        }
    }
}

/// Pure mapping `SearchQuery` -> `CodexSearchArgs`. `max_results` is
/// intentionally absent from the target type (sliced locally after
/// normalization); `response_length` is left to the executor's default.
fn to_codex_args(query: &SearchQuery) -> CodexSearchArgs {
    CodexSearchArgs {
        q: query.q.clone(),
        domains: query.domains.clone(),
        recency: query.recency,
        response_length: None,
    }
}

/// Parse the executor's rendered tool content back into normalized results.
/// Lines shaped `[n] title - url (ref_id: x)` become entries; a following
/// line indented with a single space is that entry's snippet. The
/// `Search results:` header, the optional plaintext `output` preamble, and
/// `ref_id` markers are dropped.
fn normalize_content(content: &str) -> Vec<SearchResult> {
    let mut results: Vec<SearchResult> = Vec::new();
    for line in content.lines() {
        if line.starts_with('[') {
            let Some(rest) = line.strip_prefix('[').and_then(|r| r.split_once("] ")) else {
                continue;
            };
            let mut rest = rest.1.to_string();
            if let Some((without_ref, _)) = rest.rsplit_once(" (ref_id: ") {
                rest = without_ref.to_string();
            }
            let Some((title, url)) = rest.rsplit_once(" - ") else {
                continue;
            };
            results.push(SearchResult {
                title: title.to_string(),
                url: url.to_string(),
                snippet: String::new(),
            });
        } else if (line.starts_with(' ') || line.starts_with('\t')) && !line.trim().is_empty() {
            if let Some(last) = results.last_mut() {
                if last.snippet.is_empty() {
                    last.snippet = line.trim().to_string();
                } else {
                    last.snippet.push('\n');
                    last.snippet.push_str(line.trim());
                }
            }
        }
    }
    results
}

#[async_trait]
impl SearchBackend for CodexSearchBackend {
    async fn execute(&self, query: &SearchQuery) -> Result<Vec<SearchResult>, BackendError> {
        let args = to_codex_args(query);
        let max = effective_max_results(query);
        let result = self.executor.execute_search_once(args).await;
        if result.is_error {
            return Err(BackendError::new(result.content));
        }
        Ok(slice_results(normalize_content(&result.content), max))
    }

    fn name(&self) -> &'static str {
        "codex"
    }

    async fn ready(&self) -> bool {
        self.readiness.ready().await
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::codex::search::executor::Sleeper;
    use crate::codex::search::metrics::SearchMetrics;
    use crate::oauth::{OAuthManager, OAuthTokenStore, UsageTracker};
    use std::time::Duration;

    struct NoopSleeper;

    #[async_trait]
    impl Sleeper for NoopSleeper {
        async fn sleep(&self, _duration: Duration) {}
    }

    struct FixedReadiness(bool);

    #[async_trait]
    impl CodexSearchReadiness for FixedReadiness {
        async fn ready(&self) -> bool {
            self.0
        }
    }

    fn make_executor() -> Arc<SearchExecutor> {
        let store = OAuthTokenStore::new(
            std::env::temp_dir().join(format!("oauth-websearch-{}", uuid::Uuid::new_v4())),
        );
        let oauth = Arc::new(OAuthManager::new(store, reqwest::Client::new()));
        Arc::new(SearchExecutor::with_sleeper(
            reqwest::Client::new(),
            oauth,
            Arc::new(UsageTracker::new()),
            Arc::new(SearchMetrics::new()),
            "http://localhost/search".to_string(),
            Duration::from_secs(15),
            Arc::new(NoopSleeper),
        ))
    }

    fn make_backend(ready: bool) -> CodexSearchBackend {
        CodexSearchBackend::new(make_executor(), Arc::new(FixedReadiness(ready)))
    }

    fn query(q: &str) -> SearchQuery {
        SearchQuery {
            q: q.to_string(),
            domains: None,
            recency: None,
            max_results: None,
        }
    }

    #[test]
    fn to_codex_args_maps_q_domains_recency_and_drops_max_results() {
        let q = SearchQuery {
            q: "rust async".to_string(),
            domains: Some(vec!["docs.rs".to_string(), "crates.io".to_string()]),
            recency: Some(7),
            max_results: Some(3),
        };
        let args = to_codex_args(&q);
        assert_eq!(args.q, "rust async");
        assert_eq!(
            args.domains,
            Some(vec!["docs.rs".to_string(), "crates.io".to_string()])
        );
        assert_eq!(args.recency, Some(7));
        assert_eq!(args.response_length, None);
    }

    #[test]
    fn to_codex_args_all_optionals_absent() {
        let args = to_codex_args(&query("only-q"));
        assert_eq!(args.q, "only-q");
        assert_eq!(args.domains, None);
        assert_eq!(args.recency, None);
        assert_eq!(args.response_length, None);
    }

    #[test]
    fn normalize_content_parses_title_url_snippet_and_strips_ref_id() {
        let content = "Search results:\n[1] Async Rust book - https://rust-lang.github.io/async-book/ (ref_id: r1)\n intro paragraph\n[2] Tokio runtime docs - https://docs.rs/tokio\n runtime guide";
        let results = normalize_content(content);
        assert_eq!(results.len(), 2);
        assert_eq!(results[0].title, "Async Rust book");
        assert_eq!(results[0].url, "https://rust-lang.github.io/async-book/");
        assert_eq!(results[0].snippet, "intro paragraph");
        assert_eq!(results[1].title, "Tokio runtime docs");
        assert_eq!(results[1].url, "https://docs.rs/tokio");
        assert_eq!(results[1].snippet, "runtime guide");
    }

    #[test]
    fn normalize_content_ignores_output_preamble_and_header() {
        let content = "Plaintext answer line\nSearch results:\n[1] T - https://a.example\n s";
        let results = normalize_content(content);
        assert_eq!(results.len(), 1);
        assert_eq!(results[0].title, "T");
        assert_eq!(results[0].url, "https://a.example");
    }

    #[test]
    fn normalize_content_empty_and_non_matching_inputs_yield_empty() {
        assert!(normalize_content("").is_empty());
        assert!(normalize_content("Search authentication unavailable.").is_empty());
        assert!(normalize_content("Search results:\n").is_empty());
    }

    #[test]
    fn name_is_codex() {
        assert_eq!(make_backend(true).name(), "codex");
    }

    #[tokio::test]
    async fn ready_delegates_to_readiness_handle() {
        assert!(make_backend(true).ready().await);
        assert!(!make_backend(false).ready().await);
    }
}
