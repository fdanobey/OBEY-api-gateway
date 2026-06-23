#![allow(dead_code)]
//! Exact-match in-memory response cache (Tier 1).
//!
//! Complements the semantic [`super::SemanticCache`] (Tier 2) by catching
//! byte-for-byte identical retries before any embedding round-trip.
//!
//! Key design:
//!   - SHA-256 over a canonical JSON representation of the request that
//!     includes `model`, full `messages`, `tools`/`tool_choice`,
//!     `response_format`, `top_p`, `frequency_penalty`, `presence_penalty`,
//!     `stop`, `seed`, `n`, and `temperature` (rounded to threshold).
//!   - `stream` is deliberately excluded so streaming and non-streaming
//!     callers share cache entries.
//!
//! Eligibility:
//!   - `temperature <= temperature_threshold` (default 0.15)
//!   - `n == 1` (default if absent)
//!
//! Eviction: lazy TTL on read, oldest-first batch eviction when size exceeds
//! `max_entries` (no LRU bookkeeping; cheap and good enough for ~10^3 entries).

use std::sync::Arc;
use std::time::{Duration, Instant};

use dashmap::DashMap;
use ring::digest;

use crate::config::ExactCacheConfig;
use crate::models::openai::OpenAIRequest;

/// Fields in `OpenAIRequest::extra` that are part of the cache key.
///
/// Whitelist (rather than blacklist) so we never accidentally key on a
/// non-deterministic transport field like `user`, `request-id`, or trace
/// metadata.
const KEYED_EXTRA_FIELDS: &[&str] = &[
    "tools",
    "tool_choice",
    "response_format",
    "top_p",
    "frequency_penalty",
    "presence_penalty",
    "stop",
    "seed",
    "n",
    "logit_bias",
    "logprobs",
    "top_logprobs",
];

#[derive(Debug, Clone)]
struct ExactCacheEntry {
    response_json: String,
    inserted_at: Instant,
}

/// Exact-match in-memory cache backed by [`DashMap`].
pub struct ExactCache {
    entries: Arc<DashMap<[u8; 32], ExactCacheEntry>>,
    enabled: bool,
    max_entries: usize,
    ttl: Duration,
    temperature_threshold: f32,
}

impl ExactCache {
    /// Construct from configuration.
    pub fn new(config: &ExactCacheConfig) -> Self {
        Self {
            entries: Arc::new(DashMap::new()),
            enabled: config.enabled,
            max_entries: config.max_entries.max(1),
            ttl: Duration::from_secs(config.ttl_seconds),
            temperature_threshold: config.temperature_threshold,
        }
    }

    /// True if the request is eligible to participate in the cache.
    ///
    /// Both `get` and `set` are gated on this so writes and reads stay in
    /// sync.  Disabled caches always return `false`.
    pub fn is_eligible(&self, request: &OpenAIRequest) -> bool {
        self.enabled && is_cache_eligible(request, self.temperature_threshold)
    }

    /// Look up a cached response for the given request.
    ///
    /// Returns `None` if the request is ineligible, no entry exists, or
    /// the entry has expired (expired entries are removed lazily).
    pub fn get(&self, request: &OpenAIRequest) -> Option<String> {
        if !self.is_eligible(request) {
            return None;
        }
        let key = self.compute_key(request);

        let expired = match self.entries.get(&key) {
            Some(entry) => {
                if entry.inserted_at.elapsed() > self.ttl {
                    true
                } else {
                    return Some(entry.response_json.clone());
                }
            }
            None => return None,
        };

        if expired {
            self.entries.remove(&key);
        }
        None
    }

    /// Store a response.  No-op if the request is ineligible.
    pub fn set(&self, request: &OpenAIRequest, response_json: String) {
        if !self.is_eligible(request) {
            return;
        }
        let key = self.compute_key(request);
        self.entries.insert(
            key,
            ExactCacheEntry {
                response_json,
                inserted_at: Instant::now(),
            },
        );
        self.maybe_evict();
    }

    /// Current entry count (mostly for tests / metrics).
    pub fn len(&self) -> usize {
        self.entries.len()
    }

    fn compute_key(&self, request: &OpenAIRequest) -> [u8; 32] {
        let canonical = canonicalize_request(request, self.temperature_threshold);
        let d = digest::digest(&digest::SHA256, canonical.as_bytes());
        let mut out = [0u8; 32];
        out.copy_from_slice(d.as_ref());
        out
    }

    /// When the cache exceeds capacity, drop the oldest 10% (or at least
    /// the overflow) by `inserted_at` timestamp.
    fn maybe_evict(&self) {
        let len = self.entries.len();
        if len <= self.max_entries {
            return;
        }
        let overflow = len - self.max_entries;
        let target = overflow + (self.max_entries / 10).max(1);

        let mut by_age: Vec<([u8; 32], Instant)> = self
            .entries
            .iter()
            .map(|e| (*e.key(), e.value().inserted_at))
            .collect();
        by_age.sort_by_key(|(_, t)| *t);

        for (k, _) in by_age.into_iter().take(target) {
            self.entries.remove(&k);
        }
    }
}

/// Shared eligibility predicate so both tiers can use one rule.
pub fn is_cache_eligible(request: &OpenAIRequest, temperature_threshold: f32) -> bool {
    let temperature = request.temperature.unwrap_or(0.0);
    if temperature > temperature_threshold {
        return false;
    }
    let n = request
        .extra
        .get("n")
        .and_then(|v| v.as_u64())
        .unwrap_or(1);
    if n != 1 {
        return false;
    }
    true
}

/// Build a canonical, key-sorted JSON string for the cacheable parts of a
/// request.  Two byte-equivalent requests produce identical strings; two
/// requests differing only in transport fields (`stream`, `user`, etc.)
/// produce identical strings.
fn canonicalize_request(request: &OpenAIRequest, temperature_threshold: f32) -> String {
    let mut root = serde_json::Map::new();
    root.insert("model".to_string(), serde_json::Value::String(request.model.clone()));
    root.insert(
        "messages".to_string(),
        serde_json::to_value(&request.messages).unwrap_or(serde_json::Value::Null),
    );

    // Treat any temperature at or below the threshold as 0 so callers who
    // send 0.0 / 0.05 / 0.1 share a key.  Above threshold the request is
    // ineligible anyway.
    let canonical_temp = request
        .temperature
        .map(|t| if t <= temperature_threshold { 0.0 } else { t })
        .unwrap_or(0.0);
    root.insert(
        "temperature".to_string(),
        serde_json::Number::from_f64(canonical_temp as f64)
            .map(serde_json::Value::Number)
            .unwrap_or(serde_json::Value::Null),
    );

    if let Some(mt) = request.max_tokens {
        root.insert("max_tokens".to_string(), serde_json::json!(mt));
    }

    for field in KEYED_EXTRA_FIELDS {
        if let Some(v) = request.extra.get(*field) {
            root.insert((*field).to_string(), v.clone());
        }
    }

    canonicalize_value(&serde_json::Value::Object(root))
}

/// Recursively canonicalize a JSON value: object keys sorted lexically,
/// arrays preserve order, scalars unchanged.
fn canonicalize_value(v: &serde_json::Value) -> String {
    match v {
        serde_json::Value::Object(m) => {
            let mut keys: Vec<&String> = m.keys().collect();
            keys.sort();
            let mut s = String::from("{");
            for (i, k) in keys.iter().enumerate() {
                if i > 0 {
                    s.push(',');
                }
                s.push_str(&serde_json::to_string(k).unwrap_or_default());
                s.push(':');
                s.push_str(&canonicalize_value(&m[*k]));
            }
            s.push('}');
            s
        }
        serde_json::Value::Array(a) => {
            let mut s = String::from("[");
            for (i, v) in a.iter().enumerate() {
                if i > 0 {
                    s.push(',');
                }
                s.push_str(&canonicalize_value(v));
            }
            s.push(']');
            s
        }
        other => serde_json::to_string(other).unwrap_or_default(),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::models::openai::Message;

    fn req(temp: Option<f32>) -> OpenAIRequest {
        OpenAIRequest {
            model: "gpt-4".to_string(),
            messages: vec![Message {
                role: "user".to_string(),
                content: serde_json::Value::String("hi".to_string()),
                extra: Default::default(),
            }],
            stream: false,
            temperature: temp,
            max_tokens: None,
            extra: Default::default(),
        }
    }

    fn cfg() -> ExactCacheConfig {
        ExactCacheConfig {
            enabled: true,
            max_entries: 100,
            ttl_seconds: 3600,
            temperature_threshold: 0.15,
        }
    }

    #[test]
    fn eligibility_temperature_threshold() {
        let c = ExactCache::new(&cfg());
        assert!(c.is_eligible(&req(None)));
        assert!(c.is_eligible(&req(Some(0.0))));
        assert!(c.is_eligible(&req(Some(0.15))));
        assert!(!c.is_eligible(&req(Some(0.16))));
        assert!(!c.is_eligible(&req(Some(0.7))));
    }

    #[test]
    fn disabled_cache_is_inert() {
        let mut config = cfg();
        config.enabled = false;
        let c = ExactCache::new(&config);
        let r = req(Some(0.0));
        c.set(&r, "X".to_string());
        assert_eq!(c.len(), 0);
        assert!(c.get(&r).is_none());
        assert!(!c.is_eligible(&r));
    }

    #[test]
    fn eligibility_n_must_be_one() {
        let c = ExactCache::new(&cfg());
        let mut r = req(None);
        r.extra.insert("n".into(), serde_json::json!(1));
        assert!(c.is_eligible(&r));
        r.extra.insert("n".into(), serde_json::json!(3));
        assert!(!c.is_eligible(&r));
    }

    #[test]
    fn round_trip_get_set() {
        let c = ExactCache::new(&cfg());
        let r = req(Some(0.0));
        assert!(c.get(&r).is_none());
        c.set(&r, r#"{"id":"resp-1"}"#.to_string());
        assert_eq!(c.get(&r).as_deref(), Some(r#"{"id":"resp-1"}"#));
    }

    #[test]
    fn streaming_and_nonstreaming_share_key() {
        let c = ExactCache::new(&cfg());
        let mut r = req(Some(0.0));
        c.set(&r, "X".to_string());
        r.stream = true; // must not change the key
        assert_eq!(c.get(&r).as_deref(), Some("X"));
    }

    #[test]
    fn temperature_below_threshold_collapses_to_same_key() {
        let c = ExactCache::new(&cfg());
        c.set(&req(Some(0.0)), "A".to_string());
        // 0.05 still below threshold → same canonical key → same hit
        assert_eq!(c.get(&req(Some(0.05))).as_deref(), Some("A"));
        assert_eq!(c.get(&req(Some(0.15))).as_deref(), Some("A"));
    }

    #[test]
    fn different_tools_yield_different_keys() {
        let c = ExactCache::new(&cfg());
        let mut a = req(Some(0.0));
        a.extra.insert(
            "tools".into(),
            serde_json::json!([{"type":"function","function":{"name":"foo"}}]),
        );
        let mut b = req(Some(0.0));
        b.extra.insert(
            "tools".into(),
            serde_json::json!([{"type":"function","function":{"name":"bar"}}]),
        );
        c.set(&a, "A".to_string());
        assert_eq!(c.get(&a).as_deref(), Some("A"));
        assert!(c.get(&b).is_none());
    }

    #[test]
    fn unrelated_extra_fields_do_not_change_key() {
        // user / request_id should not be part of the key
        let c = ExactCache::new(&cfg());
        let mut a = req(Some(0.0));
        a.extra.insert("user".into(), serde_json::json!("alice"));
        c.set(&a, "X".to_string());
        let mut b = req(Some(0.0));
        b.extra.insert("user".into(), serde_json::json!("bob"));
        assert_eq!(c.get(&b).as_deref(), Some("X"));
    }

    #[test]
    fn ineligible_request_neither_reads_nor_writes() {
        let c = ExactCache::new(&cfg());
        let r = req(Some(0.7));
        c.set(&r, "X".to_string());
        assert_eq!(c.len(), 0);
        assert!(c.get(&r).is_none());
    }

    #[test]
    fn ttl_expiry_purges_entry_lazily() {
        let cfg = ExactCacheConfig {
            enabled: true,
            max_entries: 100,
            ttl_seconds: 0, // immediate expiry
            temperature_threshold: 0.15,
        };
        let c = ExactCache::new(&cfg);
        let r = req(Some(0.0));
        c.set(&r, "X".to_string());
        std::thread::sleep(Duration::from_millis(10));
        assert!(c.get(&r).is_none());
        assert_eq!(c.len(), 0);
    }

    #[test]
    fn eviction_drops_oldest_when_capped() {
        let cfg = ExactCacheConfig {
            enabled: true,
            max_entries: 3,
            ttl_seconds: 3600,
            temperature_threshold: 0.15,
        };
        let c = ExactCache::new(&cfg);
        for i in 0..10 {
            let mut r = req(Some(0.0));
            r.model = format!("model-{}", i);
            c.set(&r, format!("v-{}", i));
            // Force a different inserted_at on each insert
            std::thread::sleep(Duration::from_millis(2));
        }
        assert!(c.len() <= 3);
    }

    #[test]
    fn canonicalize_sorts_object_keys() {
        let v = serde_json::json!({"b":1,"a":2,"c":{"y":1,"x":2}});
        let s = canonicalize_value(&v);
        assert_eq!(s, r#"{"a":2,"b":1,"c":{"x":2,"y":1}}"#);
    }

    #[test]
    fn canonicalize_preserves_array_order() {
        let v = serde_json::json!([3, 1, 2]);
        assert_eq!(canonicalize_value(&v), "[3,1,2]");
    }
}
