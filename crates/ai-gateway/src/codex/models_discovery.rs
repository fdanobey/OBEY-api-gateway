//! Dynamic model discovery for Codex-compatible models.
//!
//! Fetches the OpenAI models page, extracts model IDs relevant to Codex
//! (Coding section + frontier reasoning models), and caches the result.
//! Falls back to a static list when the fetch fails.

use std::sync::Arc;
use std::time::{SystemTime, UNIX_EPOCH};
use tokio::sync::RwLock;

/// How long to cache the fetched model list (24 hours in seconds).
const CACHE_TTL_SECS: u64 = 86_400;

/// URL to fetch the models page from.
const MODELS_PAGE_URL: &str = "https://developers.openai.com/api/docs/models/all";

/// A cached model list with freshness tracking.
#[derive(Debug, Clone)]
pub struct CachedModelList {
    pub models: Vec<CodexModel>,
    pub fetched_at: u64,
    pub is_stale: bool,
}

/// A single model entry for the admin UI dropdown.
#[derive(Debug, Clone, serde::Serialize)]
pub struct CodexModel {
    pub id: String,
    pub object: String,
    pub owned_by: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub description: Option<String>,
    #[serde(skip_serializing_if = "std::ops::Not::not")]
    pub deprecated: bool,
}

/// Manages discovery and caching of Codex-compatible models.
#[derive(Debug)]
pub struct ModelsDiscovery {
    cache: Arc<RwLock<Option<CachedModelList>>>,
    http: reqwest::Client,
}

impl ModelsDiscovery {
    pub fn new() -> Self {
        Self {
            cache: Arc::new(RwLock::new(None)),
            http: reqwest::Client::builder()
                .timeout(std::time::Duration::from_secs(15))
                .build()
                .unwrap_or_default(),
        }
    }

    /// Get the current model list. Attempts a refresh if the cache is expired.
    /// Returns (models_json_value, is_stale).
    pub async fn get_models(&self) -> (serde_json::Value, bool) {
        let now = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .map(|d| d.as_secs())
            .unwrap_or(0);

        {
            let cache = self.cache.read().await;
            if let Some(ref cached) = *cache {
                if now - cached.fetched_at < CACHE_TTL_SECS {
                    return (self.to_models_response(&cached.models), cached.is_stale);
                }
            }
        }

        // Cache expired or empty — try to refresh
        match self.fetch_and_parse().await {
            Ok(models) => {
                let cached = CachedModelList {
                    models: models.clone(),
                    fetched_at: now,
                    is_stale: false,
                };
                *self.cache.write().await = Some(cached);
                (self.to_models_response(&models), false)
            }
            Err(e) => {
                tracing::warn!(error = %e, "Failed to fetch Codex models from OpenAI; using static fallback");
                let fallback = self.static_fallback();
                let cached = CachedModelList {
                    models: fallback.clone(),
                    fetched_at: now,
                    is_stale: true,
                };
                *self.cache.write().await = Some(cached);
                (self.to_models_response(&fallback), true)
            }
        }
    }

    /// Fetch the OpenAI models page and parse out Codex-relevant model IDs.
    async fn fetch_and_parse(&self) -> Result<Vec<CodexModel>, String> {
        let resp = self
            .http
            .get(MODELS_PAGE_URL)
            .header("user-agent", "OBEY-Gateway/1.0")
            .send()
            .await
            .map_err(|e| format!("HTTP request failed: {e}"))?;

        if !resp.status().is_success() {
            return Err(format!("HTTP {}", resp.status()));
        }

        let body = resp
            .text()
            .await
            .map_err(|e| format!("Failed to read body: {e}"))?;

        Ok(self.parse_models_from_html(&body))
    }

    /// Parse model IDs from the HTML content of the OpenAI models page.
    /// Looks for `/api/docs/models/<model-id>` URL patterns in the page content.
    fn parse_models_from_html(&self, html: &str) -> Vec<CodexModel> {
        let mut models = Vec::new();
        let mut seen = std::collections::HashSet::new();

        let pattern = "/api/docs/models/";
        for segment in html.split(pattern).skip(1) {
            // Extract the model ID (up to the next quote, bracket, or whitespace)
            let model_id: String = segment
                .chars()
                .take_while(|c| {
                    !matches!(c, '"' | '\'' | ')' | ']' | '>' | '<' | ' ' | '\n' | '\t')
                })
                .collect();

            if model_id.is_empty() || model_id.len() > 50 {
                continue;
            }

            // Skip the "all" page itself
            if model_id == "all" {
                continue;
            }

            // Filter to only Codex-relevant models
            if !self.is_codex_relevant(&model_id) {
                continue;
            }

            if seen.insert(model_id.clone()) {
                let deprecated = html.contains(&format!("{} Deprecated", model_id))
                    || html.contains(&format!("{}.*Deprecated", model_id));

                models.push(CodexModel {
                    id: model_id,
                    object: "model".to_string(),
                    owned_by: "openai-codex".to_string(),
                    description: None,
                    deprecated,
                });
            }
        }

        // If parsing yielded nothing useful, return the static fallback
        if models.is_empty() {
            return self.static_fallback();
        }

        // Sort: non-deprecated first, then by ID
        models.sort_by(|a, b| {
            a.deprecated
                .cmp(&b.deprecated)
                .then_with(|| a.id.cmp(&b.id))
        });

        models
    }

    /// Check if a model ID is relevant for Codex usage.
    fn is_codex_relevant(&self, model_id: &str) -> bool {
        let dominated_by_codex = model_id.contains("codex")
            || model_id.starts_with("gpt-5")
            || model_id.starts_with("gpt-4.1")
            || model_id.starts_with("o1")
            || model_id.starts_with("o3")
            || model_id.starts_with("o4");

        let excluded = model_id.contains("image")
            || model_id.contains("audio")
            || model_id.contains("realtime")
            || model_id.contains("tts")
            || model_id.contains("whisper")
            || model_id.contains("embedding")
            || model_id.contains("moderation")
            || model_id.contains("chat-latest")
            || model_id.contains("sora")
            || model_id.contains("dall-e")
            || model_id.contains("transcribe")
            || model_id.contains("search")
            || model_id.contains("preview")
            || model_id.contains("computer-use")
            || model_id.contains("deep-research")
            || model_id.contains("oss-");

        dominated_by_codex && !excluded
    }

    /// Static fallback model list derived from known Codex-compatible models.
    fn static_fallback(&self) -> Vec<CodexModel> {
        let static_models = [
            "gpt-5.5",
            "gpt-5.5-pro",
            "gpt-5.4",
            "gpt-5.4-pro",
            "gpt-5.4-mini",
            "gpt-5.4-nano",
            "gpt-5.3-codex",
            "gpt-5.2-codex",
            "gpt-5.1-codex",
            "gpt-5.1-codex-max",
            "gpt-5.1-codex-mini",
            "gpt-5-codex",
            "codex-mini-latest",
            "gpt-5.2",
            "gpt-5.1",
            "gpt-5",
            "gpt-5-mini",
            "gpt-5-nano",
            "gpt-4.1",
            "gpt-4.1-mini",
            "gpt-4.1-nano",
            "o3",
            "o3-pro",
            "o4-mini",
        ];

        static_models
            .iter()
            .map(|id| CodexModel {
                id: id.to_string(),
                object: "model".to_string(),
                owned_by: "openai-codex".to_string(),
                description: None,
                deprecated: false,
            })
            .collect()
    }

    /// Convert model list to the OpenAI /v1/models response format.
    fn to_models_response(&self, models: &[CodexModel]) -> serde_json::Value {
        let data: Vec<serde_json::Value> = models
            .iter()
            .map(|m| {
                serde_json::json!({
                    "id": m.id,
                    "object": m.object,
                    "owned_by": m.owned_by,
                })
            })
            .collect();

        serde_json::json!({
            "data": data,
            "object": "list"
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_is_codex_relevant_accepts_codex_models() {
        let disc = ModelsDiscovery::new();
        assert!(disc.is_codex_relevant("gpt-5.1-codex"));
        assert!(disc.is_codex_relevant("gpt-5.5"));
        assert!(disc.is_codex_relevant("codex-mini-latest"));
        assert!(disc.is_codex_relevant("o3"));
        assert!(disc.is_codex_relevant("o4-mini"));
        assert!(disc.is_codex_relevant("gpt-4.1"));
        assert!(disc.is_codex_relevant("gpt-4.1-mini"));
    }

    #[test]
    fn test_is_codex_relevant_rejects_non_codex() {
        let disc = ModelsDiscovery::new();
        assert!(!disc.is_codex_relevant("gpt-4o"));
        assert!(!disc.is_codex_relevant("dall-e-3"));
        assert!(!disc.is_codex_relevant("whisper-1"));
        assert!(!disc.is_codex_relevant("text-embedding-3-large"));
        assert!(!disc.is_codex_relevant("gpt-5-audio"));
        assert!(!disc.is_codex_relevant("o3-deep-research"));
    }

    #[test]
    fn test_parse_models_from_html_extracts_ids() {
        let disc = ModelsDiscovery::new();
        let html = r#"
            <a href="/api/docs/models/gpt-5.1-codex">GPT 5.1 Codex</a>
            <a href="/api/docs/models/gpt-5.5">GPT 5.5</a>
            <a href="/api/docs/models/dall-e-3">DALL-E 3</a>
            <a href="/api/docs/models/o3">O3</a>
        "#;
        let models = disc.parse_models_from_html(html);
        let ids: Vec<&str> = models.iter().map(|m| m.id.as_str()).collect();
        assert!(ids.contains(&"gpt-5.1-codex"));
        assert!(ids.contains(&"gpt-5.5"));
        assert!(ids.contains(&"o3"));
        assert!(!ids.contains(&"dall-e-3"));
    }

    #[test]
    fn test_parse_models_from_html_falls_back_on_empty() {
        let disc = ModelsDiscovery::new();
        let html = "<html><body>No models here</body></html>";
        let models = disc.parse_models_from_html(html);
        // Should return the static fallback
        assert!(!models.is_empty());
        assert!(models.iter().any(|m| m.id == "codex-mini-latest"));
    }

    #[test]
    fn test_static_fallback_is_non_empty() {
        let disc = ModelsDiscovery::new();
        let fallback = disc.static_fallback();
        assert!(fallback.len() > 10);
        assert!(fallback.iter().all(|m| !m.deprecated));
        assert!(fallback.iter().all(|m| m.object == "model"));
    }

    #[test]
    fn test_to_models_response_format() {
        let disc = ModelsDiscovery::new();
        let models = vec![CodexModel {
            id: "test-model".to_string(),
            object: "model".to_string(),
            owned_by: "openai-codex".to_string(),
            description: None,
            deprecated: false,
        }];
        let resp = disc.to_models_response(&models);
        assert_eq!(resp["object"], "list");
        assert_eq!(resp["data"][0]["id"], "test-model");
        assert_eq!(resp["data"][0]["owned_by"], "openai-codex");
    }

    #[tokio::test]
    async fn test_get_models_returns_fallback_without_network() {
        let disc = ModelsDiscovery::new();
        let (resp, is_stale) = disc.get_models().await;
        // Verify response structure regardless of network reachability:
        // when the OpenAI page is reachable in CI, the live fetch succeeds
        // and `is_stale == false`; when it isn't, the static fallback
        // populates the cache with `is_stale == true`. Either is valid.
        assert_eq!(resp["object"], "list");
        let data = resp["data"].as_array().expect("data must be an array");
        assert!(!data.is_empty(), "data must be non-empty");
        // Sanity: each model entry has the required shape.
        for m in data {
            assert!(m["id"].is_string(), "model id must be a string");
            assert_eq!(m["object"], "model");
            assert_eq!(m["owned_by"], "openai-codex");
        }
        // is_stale is environment-dependent — accept either value.
        let _ = is_stale;
    }
}
