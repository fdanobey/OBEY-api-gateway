//! Prompt-cache sticky routing: prefix-hash to provider affinity map with TTL.
//!
//! [`StickyCache`] maps a canonical hash of the cacheable conversation prefix
//! to the most recently successful provider for that prefix. Entries expire
//! lazily on [`StickyCache::get`] lookups and via an explicit
//! [`StickyCache::evict_expired`] sweep.
//!
//! Backing storage is a lock-free [`DashMap`] mirroring the existing circuit
//! breaker and rate-limiter patterns.

use std::time::{Duration, Instant};

use dashmap::DashMap;
use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};

use crate::models::openai::OpenAIRequest;

/// Cache token breakdown reported by a provider for a single turn.
///
/// Plain value type: it is copied into sticky entries and (de)serialized with
/// telemetry payloads.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub struct CacheUsage {
    /// Prompt tokens served from the upstream KV cache (cache read).
    pub cache_read_input_tokens: u64,
    /// Prompt tokens written into the cache this turn (cache creation).
    pub cache_creation_input_tokens: u64,
    /// Prompt tokens charged at the full uncached rate.
    pub uncached_input_tokens: u64,
}

/// Affinity entry: which provider + model last served a given prefix, and when
/// that affinity expires.
///
/// `expires_at` is a [`std::time::Instant`] (monotonic, process-local) so this
/// struct deliberately does **not** implement `Serialize`; only the
/// `last_success_usage` field is persisted to durable telemetry.
#[derive(Debug, Clone)]
pub struct ProviderStickyEntry {
    /// Identifier of the provider that last served this prefix successfully.
    pub provider_id: String,
    /// Concrete provider-side model id used for that turn.
    pub model_id: String,
    /// Cache token breakdown reported by the provider, if any.
    pub last_success_usage: Option<CacheUsage>,
    /// Monotonic deadline after which the entry is considered stale.
    pub expires_at: Instant,
}

/// Prefix-to-provider affinity map with TTL-based expiration.
pub struct StickyCache {
    map: DashMap<u64, ProviderStickyEntry>,
    ttl: Duration,
}

impl StickyCache {
    /// Creates an empty cache whose entries expire after `ttl`.
    pub fn new(ttl: Duration) -> Self {
        Self {
            map: DashMap::new(),
            ttl,
        }
    }

    /// Looks up the affinity entry for `hash`.
    ///
    /// Returns [`None`] if the entry is missing or expired. Expired entries
    /// are removed opportunistically during the lookup (lazy eviction) without
    /// holding a write lock across the read path.
 pub fn get(&self, hash: u64) -> Option<ProviderStickyEntry> {
 let now = Instant::now();
 self.map.remove_if(&hash, |_, entry| entry.expires_at <= now);
 self.map.get(&hash).map(|entry| entry.value().clone())
 }

 /// Returns the `(provider, model)` pair of the affinity entry for
 /// `hash`, if present and fresh.
 ///
 /// Same freshness semantics as [`StickyCache::get`] (expired entries
 /// are lazily evicted and indistinguishable from a miss). Lightweight
 /// projection for the reasoning-compat source-model attribution
 /// (reasoning-failover-compat spec, Task 6): the family is classified
 /// at the call site from the returned model id.
 pub fn get_model_affinity(&self, hash: u64) -> Option<(String, String)> {
 self.get(hash)
 .map(|entry| (entry.provider_id, entry.model_id))
 }

    /// Upserts an affinity entry for `hash`, refreshing its expiry deadline.
    pub fn insert(
        &self,
        hash: u64,
        provider_id: String,
        model_id: String,
        usage: Option<CacheUsage>,
    ) {
        let expires_at = Instant::now() + self.ttl;
        self.map.insert(
            hash,
            ProviderStickyEntry {
                provider_id,
                model_id,
                last_success_usage: usage,
                expires_at,
            },
        );
    }

    /// Sweeps the whole map, removing every entry whose TTL has elapsed.
    pub fn evict_expired(&self) {
        let now = Instant::now();
        self.map.retain(|_, entry| entry.expires_at > now);
    }

    /// Removes every entry, keeping the configured TTL. Used on config
    /// hot-reload to reset prefix affinity alongside the circuit-breaker
    /// and rate-limiter state.
    pub fn clear(&self) {
        self.map.clear();
    }

    /// Computes a stable affinity hash of the cacheable conversation prefix.
    ///
    /// # Rule (chosen per `design.md` "Canonicalization rule for prefix hashing")
    ///
    /// The prefix covers:
    /// 1. The requested `model`.
    /// 2. The full `tools` JSON array from `request.extra["tools"]` (if any).
    /// 3. The message prefix. In the OpenAI request shape there is no separate
    ///    `system` field, so the leading `system` message is part of the
    ///    message vector and is covered implicitly.
    ///
    /// The message prefix is **every message strictly before the LAST message
    /// whose `role == "user"`**. The newest user turn is the conversation tail
    /// and is excluded, along with any trailing tool-result messages that
    /// follow it. Consequences of this rule:
    /// - If the last message is a `user` turn, that turn is excluded.
    /// - If the last message is a tool result (or assistant reply) and a
    ///   `user` turn exists earlier, the tail from that `user` turn onward
    ///   (including the trailing non-user messages) is excluded.
    /// - If there is no `user` message at all, the prefix is all messages.
    /// - If there are no messages, only `model` + `tools` contribute.
    ///
    /// Because the tail is sliced at the last `user` turn, appending a single
    /// new non-user tail message (e.g. an assistant reply following the user's
    /// turn) does **not** change the prefix hash, whereas appending a new
    /// `user` turn does (the prefix grows to include the previous turn).
    ///
    /// # Hashing strategy
    ///
    /// Fields are fed incrementally into a SHA-256 hasher behind distinct byte
    /// tags (no request cloning). Each `Message` sub-part and the `tools`
    /// value are serialized with `serde_json::to_vec`; serde_json uses a
    /// `BTreeMap`-backed map by default (no `preserve_order` feature is
    /// enabled in this crate), so object key order is canonical regardless of
    /// input ordering. The first 8 big-endian bytes of the SHA-256 digest form
    /// the `u64` affinity key.
    ///
    /// This reuses the same stable-hash strategy the codebase already relies on
    /// for tool-definition and asset deduplication (`sha2` is an existing
    /// workspace dependency).
    pub fn compute_prefix_hash(request: &OpenAIRequest) -> u64 {
        let mut hasher = Sha256::new();

        hasher.update(b"m");
        hasher.update(request.model.as_bytes());

        hasher.update(b"t");
        if let Some(tools) = request.extra.get("tools") {
            let bytes = serde_json::to_vec(tools).expect("serde_json::Value is serializable");
            hasher.update(&bytes);
        }

        hasher.update(b"p");
        let prefix_len = request
            .messages
            .iter()
            .rposition(|message| message.role == "user")
            .unwrap_or(request.messages.len());
        for message in &request.messages[..prefix_len] {
            hasher.update(b"s");
            let bytes = serde_json::to_vec(message).expect("Message is serializable");
            hasher.update(&bytes);
        }

        let digest = hasher.finalize();
        let mut out = [0u8; 8];
        out.copy_from_slice(&digest[..8]);
        u64::from_be_bytes(out)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn request_from_json(json: &str) -> OpenAIRequest {
        serde_json::from_str(json).expect("test request JSON is valid")
    }

    #[test]
    fn identical_requests_hash_identically_regardless_of_field_order() {
        let ordered = request_from_json(
            r#"{"model":"gpt-4o","messages":[{"role":"system","content":"You are helpful."},{"role":"user","content":"Hi"}],"tools":[{"type":"function","function":{"name":"get_weather"}}]}"#,
        );
        let reordered = request_from_json(
            r#"{"messages":[{"role":"system","content":"You are helpful."},{"role":"user","content":"Hi"}],"tools":[{"function":{"name":"get_weather"},"type":"function"}],"model":"gpt-4o"}"#,
        );
        assert_eq!(
            StickyCache::compute_prefix_hash(&ordered),
            StickyCache::compute_prefix_hash(&reordered),
        );
    }

    #[test]
    fn prefix_changes_alter_the_hash() {
        let base = request_from_json(
            r#"{"model":"gpt-4o","messages":[{"role":"system","content":"You are helpful."},{"role":"user","content":"Hi"}]}"#,
        );

        let new_model = request_from_json(
            r#"{"model":"gpt-4o-mini","messages":[{"role":"system","content":"You are helpful."},{"role":"user","content":"Hi"}]}"#,
        );
        assert_ne!(
            StickyCache::compute_prefix_hash(&base),
            StickyCache::compute_prefix_hash(&new_model),
        );

        let new_tools = request_from_json(
            r#"{"model":"gpt-4o","messages":[{"role":"system","content":"You are helpful."},{"role":"user","content":"Hi"}],"tools":[{"type":"function","function":{"name":"get_time"}}]}"#,
        );
        assert_ne!(
            StickyCache::compute_prefix_hash(&base),
            StickyCache::compute_prefix_hash(&new_tools),
        );

        let new_system = request_from_json(
            r#"{"model":"gpt-4o","messages":[{"role":"system","content":"You are very helpful."},{"role":"user","content":"Hi"}]}"#,
        );
        assert_ne!(
            StickyCache::compute_prefix_hash(&base),
            StickyCache::compute_prefix_hash(&new_system),
        );
    }

    #[test]
    fn appending_a_non_user_tail_does_not_change_the_prefix_hash() {
        let before = request_from_json(
            r#"{"model":"gpt-4o","messages":[{"role":"system","content":"S"},{"role":"user","content":"Hi"}]}"#,
        );
        let after = request_from_json(
            r#"{"model":"gpt-4o","messages":[{"role":"system","content":"S"},{"role":"user","content":"Hi"},{"role":"assistant","content":"Hello!"}]}"#,
        );
        assert_eq!(
            StickyCache::compute_prefix_hash(&before),
            StickyCache::compute_prefix_hash(&after),
        );

        let with_new_user_turn = request_from_json(
            r#"{"model":"gpt-4o","messages":[{"role":"system","content":"S"},{"role":"user","content":"Hi"},{"role":"assistant","content":"Hello!"},{"role":"user","content":"Again"}]}"#,
        );
        assert_ne!(
            StickyCache::compute_prefix_hash(&after),
            StickyCache::compute_prefix_hash(&with_new_user_turn),
        );
    }

    #[test]
    fn trailing_tool_result_tail_is_excluded_with_the_user_turn() {
        let two_turns = request_from_json(
            r#"{"model":"gpt-4o","messages":[{"role":"system","content":"S"},{"role":"user","content":"Hi"},{"role":"assistant","content":"Hello!"},{"role":"user","content":"Use the tool"}]}"#,
        );
        let with_tool_tail = request_from_json(
            r#"{"model":"gpt-4o","messages":[{"role":"system","content":"S"},{"role":"user","content":"Hi"},{"role":"assistant","content":"Hello!"},{"role":"user","content":"Use the tool"},{"role":"tool","content":"42","tool_call_id":"call_1"}]}"#,
        );
        assert_eq!(
            StickyCache::compute_prefix_hash(&two_turns),
            StickyCache::compute_prefix_hash(&with_tool_tail),
        );
    }

    #[test]
    fn single_user_turn_yields_an_empty_message_prefix() {
        let a = request_from_json(r#"{"model":"gpt-4o","messages":[{"role":"user","content":"A"}]}"#);
        let b = request_from_json(r#"{"model":"gpt-4o","messages":[{"role":"user","content":"B"}]}"#);
        assert_eq!(
            StickyCache::compute_prefix_hash(&a),
            StickyCache::compute_prefix_hash(&b),
        );

        let with_tools = request_from_json(
            r#"{"model":"gpt-4o","messages":[{"role":"user","content":"A"}],"tools":[]}"#,
        );
        assert_ne!(
            StickyCache::compute_prefix_hash(&a),
            StickyCache::compute_prefix_hash(&with_tools),
        );
    }

    #[test]
    fn no_user_message_hashes_all_messages() {
        let only_assistant = request_from_json(
            r#"{"model":"gpt-4o","messages":[{"role":"assistant","content":"Hi"}]}"#,
        );
        let different_assistant = request_from_json(
            r#"{"model":"gpt-4o","messages":[{"role":"assistant","content":"Yo"}]}"#,
        );
        assert_ne!(
            StickyCache::compute_prefix_hash(&only_assistant),
            StickyCache::compute_prefix_hash(&different_assistant),
        );
    }

    #[test]
    fn cache_usage_serde_round_trip() {
        let usage = CacheUsage {
            cache_read_input_tokens: 10,
            cache_creation_input_tokens: 20,
            uncached_input_tokens: 30,
        };
        let json = serde_json::to_string(&usage).expect("serializable");
        assert_eq!(
            serde_json::from_str::<CacheUsage>(&json).expect("deserializable"),
            usage,
        );
    }

    #[test]
    fn insert_and_get_return_latest_values() {
        let cache = StickyCache::new(Duration::from_secs(30));
        cache.insert(1, "prov-a".into(), "model-a".into(), None);
        let first = cache.get(1).expect("first insert present");
        assert_eq!(first.provider_id, "prov-a");
        assert_eq!(first.model_id, "model-a");
        assert!(first.last_success_usage.is_none());

        cache.insert(
            1,
            "prov-b".into(),
            "model-b".into(),
            Some(CacheUsage {
                cache_read_input_tokens: 5,
                cache_creation_input_tokens: 0,
                uncached_input_tokens: 0,
            }),
        );
        let second = cache.get(1).expect("upsert present");
        assert_eq!(second.provider_id, "prov-b");
        assert_eq!(second.model_id, "model-b");
        assert_eq!(
            second.last_success_usage,
            Some(CacheUsage {
                cache_read_input_tokens: 5,
                cache_creation_input_tokens: 0,
                uncached_input_tokens: 0,
            }),
        );
    }

    #[test]
    fn get_returns_none_for_missing_keys() {
        let cache = StickyCache::new(Duration::from_secs(30));
        assert!(cache.get(0).is_none());
    }

    #[test]
    fn get_lazily_evicts_expired_entries() {
        let cache = StickyCache::new(Duration::from_millis(50));
        cache.insert(1, "prov".into(), "model".into(), None);
        assert!(cache.get(1).is_some());

        std::thread::sleep(Duration::from_millis(80));
        assert!(cache.get(1).is_none());
        assert_eq!(cache.map.len(), 0);
    }

    #[test]
    fn insert_refreshes_expiry_on_upsert() {
        let cache = StickyCache::new(Duration::from_millis(120));
        cache.insert(1, "prov".into(), "model".into(), None);
        std::thread::sleep(Duration::from_millis(80));
        cache.insert(1, "prov".into(), "model".into(), None);
        std::thread::sleep(Duration::from_millis(80));
        assert!(cache.get(1).is_some());
    }

    #[test]
    fn evict_expired_clears_only_expired_entries() {
        let short = StickyCache::new(Duration::from_millis(40));
        let long = StickyCache::new(Duration::from_secs(30));
        short.insert(1, "p".into(), "m".into(), None);
        long.insert(1, "p".into(), "m".into(), None);

        std::thread::sleep(Duration::from_millis(80));
        short.evict_expired();
        long.evict_expired();

        assert!(short.get(1).is_none());
        assert!(long.get(1).is_some());
        assert_eq!(short.map.len(), 0);
    }

    #[test]
    fn concurrent_inserts_and_lookups_are_consistent() {
        let cache = StickyCache::new(Duration::from_secs(30));
        std::thread::scope(|scope| {
            for thread in 0..8u64 {
                let cache = &cache;
                scope.spawn(move || {
                    for i in 0..400u64 {
                        let key = i % 16;
                        cache.insert(
                            key,
                            format!("p{thread}"),
                            format!("m{i}"),
                            Some(CacheUsage {
                                cache_read_input_tokens: i,
                                cache_creation_input_tokens: 0,
                                uncached_input_tokens: 0,
                            }),
                        );
                        if let Some(entry) = cache.get(key) {
                            assert!(entry.provider_id.starts_with('p'));
                            assert!(entry.model_id.starts_with('m'));
                        }
                    }
                });
            }
        });

for key in 0..16u64 {
let entry = cache.get(key).expect("every key remains present");
assert!(entry.provider_id.starts_with('p'));
assert!(entry.model_id.starts_with('m'));
}
}

#[test]
fn get_model_affinity_returns_provider_and_model_pair() {
let cache = StickyCache::new(Duration::from_secs(30));
cache.insert(
7,
"anthropic".to_string(),
"claude-sonnet-4-5".to_string(),
None,
);
assert_eq!(
cache.get_model_affinity(7),
Some(("anthropic".to_string(), "claude-sonnet-4-5".to_string()))
);
}

#[test]
fn get_model_affinity_expires_with_the_entry_ttl() {
let cache = StickyCache::new(Duration::from_millis(50));
cache.insert(
7,
"anthropic".to_string(),
"claude-sonnet-4-5".to_string(),
None,
);
assert!(cache.get_model_affinity(7).is_some());
std::thread::sleep(Duration::from_millis(120));
assert!(cache.get_model_affinity(7).is_none());
}

#[test]
fn get_model_affinity_misses_when_never_inserted() {
let cache = StickyCache::new(Duration::from_secs(30));
assert!(cache.get_model_affinity(42).is_none());
}
}
