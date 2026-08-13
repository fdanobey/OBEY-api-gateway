//! Core data types for the tool compression pipeline.
//!
//! Defines `ToolDefinition`, `CompressionContext`, `ProviderCaps`, and the
//! `ProviderCapabilityMap` with default entries for known providers.

use std::collections::HashMap;

use serde::{Deserialize, Serialize};

use super::config::CompressionLevel;

// ─── Basic ID types ───────────────────────────────────────────────────────────

/// Opaque session identifier for state tracking.
pub type SessionId = String;

/// Opaque API key identifier for per-key usage stats.
pub type ApiKeyId = String;

/// Per-session tool usage frequency: tool name → call count.
pub type ToolUsageMap = HashMap<String, u64>;

/// Set of tool names disclosed in a progressive disclosure session.
pub type DisclosureSet = std::collections::HashSet<String>;

// ─── ToolDefinition ───────────────────────────────────────────────────────────

/// A single tool definition flowing through the compression pipeline.
#[derive(Debug, Clone)]
pub struct ToolDefinition {
    /// Raw JSON value of the complete tool entry (type + function object).
    pub raw: serde_json::Value,

    /// Extracted function name for fast lookup.
    pub name: String,

    /// 64-bit content hash for dedup/cache-placement comparisons.
    pub content_hash: u64,
}

// ─── CompressionContext ───────────────────────────────────────────────────────

/// Mutable context threaded through all pipeline stages within a single request.
#[derive(Debug, Clone)]
pub struct CompressionContext {
    /// The effective compression level resolved for this request.
    pub level: CompressionLevel,

    /// Provider capabilities for the target provider.
    pub provider_caps: ProviderCaps,

    /// The provider name/type (e.g. "openai", "anthropic").
    pub provider_name: String,

    /// Model name for tier-detection and feedback-loop keying.
    pub model: String,

    /// Model group name (for per-group overrides and feedback).
    pub model_group: String,

    /// Original (uncompressed) tool definitions preserved for disclosure.
    pub original_tools: Vec<ToolDefinition>,

    /// Running total of estimated tokens saved.
    pub tokens_saved: u64,

    /// Strategies that have been applied so far.
    pub strategies_applied: Vec<String>,

    /// Tools deferred by semantic retrieval (below similarity threshold).
    pub deferred_tools: Vec<ToolDefinition>,

    /// Session ID for stateful stages (pruning, disclosure, cache placement).
    pub session_id: Option<SessionId>,

    /// API key ID for per-key aggregate tracking.
    pub api_key_id: Option<ApiKeyId>,

    /// Tool usage map for the current session.
    pub session_usage: ToolUsageMap,

    /// Number of requests processed in this session so far.
    /// Used by Tool_Pruner to check against `min_requests` threshold.
    pub session_request_count: u64,

    /// Previously disclosed tool names in this session.
    pub disclosed_tools: DisclosureSet,

    /// Concatenated content from the current request messages (system + user).
    /// Used by Tool_Pruner to detect references to pruned tools for restore.
    pub message_content: Option<String>,

    /// Previous tool content hashes from the last request in this session.
    /// Populated by the middleware from `ToolCompressionState.placement_state`.
    /// Used by `CachePlacementOptimizer` to identify stable vs new/modified tools.
    pub previous_hashes: Option<Vec<u64>>,

    /// Whether the request has `"stream": true`. Stages that inject synthetic
    /// drill-down tools (namespace grouper, progressive disclosure) must no-op
    /// because the resolution loop cannot intercept streaming responses.
    pub is_streaming: bool,
}

impl Default for CompressionContext {
    fn default() -> Self {
        Self {
            level: CompressionLevel::Medium,
            provider_caps: ProviderCaps::conservative(),
            provider_name: String::new(),
            model: String::new(),
            model_group: String::new(),
            original_tools: Vec::new(),
            tokens_saved: 0,
            strategies_applied: Vec::new(),
            deferred_tools: Vec::new(),
            session_id: None,
            api_key_id: None,
            session_usage: HashMap::new(),
            session_request_count: 0,
            disclosed_tools: std::collections::HashSet::new(),
            message_content: None,
            previous_hashes: None,
            is_streaming: false,
        }
    }
}

// ─── ProviderCaps ─────────────────────────────────────────────────────────────

/// Capability flags for a specific provider type.
///
/// Used by pipeline stages to conditionally enable/disable transformations
/// that depend on provider-specific JSON Schema support.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct ProviderCaps {
    /// Provider supports `$ref` / `$defs` in tool schemas.
    pub supports_ref: bool,

    /// Provider supports `"nullable": true` shorthand.
    pub supports_nullable: bool,

    /// Provider supports prompt caching (prefix-based).
    pub supports_prompt_caching: bool,

    /// Provider supports native `tool_search` / deferred loading.
    pub supports_tool_search: bool,

    /// Provider supports EasyTool-style canonical text format.
    pub supports_canonical_format: bool,

    /// Maximum number of tools the provider accepts (None = unlimited).
    pub max_tools: Option<u32>,

    /// Model capability tier override (None = auto-detect).
    pub model_tier: Option<u8>,
}

impl ProviderCaps {
    /// Conservative defaults: all features disabled, no limits.
    /// Used as fallback for unknown providers.
    pub fn conservative() -> Self {
        Self {
            supports_ref: false,
            supports_nullable: false,
            supports_prompt_caching: false,
            supports_tool_search: false,
            supports_canonical_format: false,
            max_tools: None,
            model_tier: None,
        }
    }

    /// Merge config-provided overrides on top of this instance.
    /// Only `Some` values in the overlay replace existing fields.
    pub fn merge(&mut self, overlay: &ProviderCapsOverlay) {
        if let Some(v) = overlay.supports_ref {
            self.supports_ref = v;
        }
        if let Some(v) = overlay.supports_nullable {
            self.supports_nullable = v;
        }
        if let Some(v) = overlay.supports_prompt_caching {
            self.supports_prompt_caching = v;
        }
        if let Some(v) = overlay.supports_tool_search {
            self.supports_tool_search = v;
        }
        if let Some(v) = overlay.supports_canonical_format {
            self.supports_canonical_format = v;
        }
        if let Some(v) = overlay.max_tools {
            self.max_tools = Some(v);
        }
        if let Some(v) = overlay.model_tier {
            self.model_tier = Some(v);
        }
    }
}

/// Partial overlay for config-driven capability overrides.
#[derive(Debug, Clone, Default, PartialEq, Serialize, Deserialize)]
pub struct ProviderCapsOverlay {
    pub supports_ref: Option<bool>,
    pub supports_nullable: Option<bool>,
    pub supports_prompt_caching: Option<bool>,
    pub supports_tool_search: Option<bool>,
    pub supports_canonical_format: Option<bool>,
    pub max_tools: Option<u32>,
    pub model_tier: Option<u8>,
}

// ─── ProviderCapabilityMap ────────────────────────────────────────────────────

/// Map of provider type name → capability flags.
///
/// Pre-populated with sensible defaults for known providers. Unknown providers
/// receive conservative (all-false) defaults.
#[derive(Debug, Clone)]
pub struct ProviderCapabilityMap {
    inner: HashMap<String, ProviderCaps>,
}

impl ProviderCapabilityMap {
    /// Look up capabilities for a provider. Returns conservative defaults for
    /// unknown providers.
    pub fn get(&self, provider: &str) -> ProviderCaps {
        self.inner
            .get(provider)
            .cloned()
            .unwrap_or_else(ProviderCaps::conservative)
    }

    /// Merge config-provided overrides on top of defaults.
    /// Creates entries for providers not already in the map.
    pub fn merge_overrides(&mut self, overrides: &HashMap<String, ProviderCapsOverlay>) {
        for (provider, overlay) in overrides {
            let caps = self
                .inner
                .entry(provider.clone())
                .or_insert_with(ProviderCaps::conservative);
            caps.merge(overlay);
        }
    }

    /// Returns a reference to the inner map (useful for iteration/testing).
    pub fn inner(&self) -> &HashMap<String, ProviderCaps> {
        &self.inner
    }
}

impl Default for ProviderCapabilityMap {
    fn default() -> Self {
        let mut map = HashMap::with_capacity(8);

        // OpenAI: full JSON Schema support, prompt caching, tool_search
        map.insert(
            "openai".to_string(),
            ProviderCaps {
                supports_ref: true,
                supports_nullable: true,
                supports_prompt_caching: true,
                supports_tool_search: true,
                supports_canonical_format: true,
                max_tools: Some(128),
                model_tier: None,
            },
        );

        // Azure: uses OpenAI API format, same capabilities
        map.insert(
            "azure".to_string(),
            ProviderCaps {
                supports_ref: true,
                supports_nullable: true,
                supports_prompt_caching: true,
                supports_tool_search: true,
                supports_canonical_format: true,
                max_tools: Some(128),
                model_tier: None,
            },
        );

        // Anthropic: no $ref, no nullable shorthand, has prompt caching
        map.insert(
            "anthropic".to_string(),
            ProviderCaps {
                supports_ref: false,
                supports_nullable: false,
                supports_prompt_caching: true,
                supports_tool_search: false,
                supports_canonical_format: true,
                max_tools: Some(200),
                model_tier: None,
            },
        );

        // Google (Gemini): supports $ref and nullable, prompt caching
        map.insert(
            "google".to_string(),
            ProviderCaps {
                supports_ref: true,
                supports_nullable: true,
                supports_prompt_caching: true,
                supports_tool_search: false,
                supports_canonical_format: false,
                max_tools: Some(128),
                model_tier: None,
            },
        );

        // Groq: limited schema support, no caching
        map.insert(
            "groq".to_string(),
            ProviderCaps {
                supports_ref: false,
                supports_nullable: true,
                supports_prompt_caching: false,
                supports_tool_search: false,
                supports_canonical_format: false,
                max_tools: Some(64),
                model_tier: None,
            },
        );

        // Mistral: similar to Groq
        map.insert(
            "mistral".to_string(),
            ProviderCaps {
                supports_ref: false,
                supports_nullable: true,
                supports_prompt_caching: false,
                supports_tool_search: false,
                supports_canonical_format: false,
                max_tools: Some(64),
                model_tier: None,
            },
        );

        // Cohere: minimal schema support
        map.insert(
            "cohere".to_string(),
            ProviderCaps {
                supports_ref: false,
                supports_nullable: false,
                supports_prompt_caching: false,
                supports_tool_search: false,
                supports_canonical_format: false,
                max_tools: Some(200),
                model_tier: None,
            },
        );

        // Bedrock: uses underlying model capabilities via OpenAI-compat layer
        // Conservative defaults since actual caps depend on the underlying model
        map.insert(
            "bedrock".to_string(),
            ProviderCaps {
                supports_ref: false,
                supports_nullable: false,
                supports_prompt_caching: true,
                supports_tool_search: false,
                supports_canonical_format: false,
                max_tools: Some(200),
                model_tier: None,
            },
        );

        // NVIDIA NIM: OpenAI-compatible but limited schema features
        map.insert(
            "nvidia_nim".to_string(),
            ProviderCaps {
                supports_ref: false,
                supports_nullable: true,
                supports_prompt_caching: false,
                supports_tool_search: false,
                supports_canonical_format: false,
                max_tools: Some(64),
                model_tier: None,
            },
        );

        // Ollama: local models, minimal schema support
        map.insert(
            "ollama".to_string(),
            ProviderCaps {
                supports_ref: false,
                supports_nullable: false,
                supports_prompt_caching: false,
                supports_tool_search: false,
                supports_canonical_format: false,
                max_tools: None,
                model_tier: None,
            },
        );

        Self { inner: map }
    }
}

// ─── Tests ────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn default_map_contains_known_providers() {
        let map = ProviderCapabilityMap::default();
        let known = [
            "openai",
            "azure",
            "anthropic",
            "google",
            "groq",
            "mistral",
            "cohere",
            "bedrock",
            "nvidia_nim",
            "ollama",
        ];
        for name in &known {
            let caps = map.get(name);
            // Just verify we get a non-conservative result for known providers
            // (at least one field differs from conservative for most)
            assert!(
                caps != ProviderCaps::conservative() || *name == "ollama",
                "expected non-conservative caps for {name}"
            );
        }
    }

    #[test]
    fn unknown_provider_returns_conservative() {
        let map = ProviderCapabilityMap::default();
        let caps = map.get("unknown_provider");
        assert_eq!(caps, ProviderCaps::conservative());
    }

    #[test]
    fn conservative_defaults_all_false() {
        let caps = ProviderCaps::conservative();
        assert!(!caps.supports_ref);
        assert!(!caps.supports_nullable);
        assert!(!caps.supports_prompt_caching);
        assert!(!caps.supports_tool_search);
        assert!(!caps.supports_canonical_format);
        assert_eq!(caps.max_tools, None);
        assert_eq!(caps.model_tier, None);
    }

    #[test]
    fn openai_has_full_support() {
        let map = ProviderCapabilityMap::default();
        let caps = map.get("openai");
        assert!(caps.supports_ref);
        assert!(caps.supports_nullable);
        assert!(caps.supports_prompt_caching);
        assert!(caps.supports_tool_search);
        assert!(caps.supports_canonical_format);
        assert_eq!(caps.max_tools, Some(128));
    }

    #[test]
    fn azure_matches_openai() {
        let map = ProviderCapabilityMap::default();
        assert_eq!(map.get("openai"), map.get("azure"));
    }

    #[test]
    fn anthropic_no_ref_no_nullable() {
        let map = ProviderCapabilityMap::default();
        let caps = map.get("anthropic");
        assert!(!caps.supports_ref);
        assert!(!caps.supports_nullable);
        assert!(caps.supports_prompt_caching);
        assert!(!caps.supports_tool_search);
        assert!(caps.supports_canonical_format);
        assert_eq!(caps.max_tools, Some(200));
    }

    #[test]
    fn google_supports_ref_and_nullable() {
        let map = ProviderCapabilityMap::default();
        let caps = map.get("google");
        assert!(caps.supports_ref);
        assert!(caps.supports_nullable);
        assert!(caps.supports_prompt_caching);
        assert!(!caps.supports_tool_search);
        assert!(!caps.supports_canonical_format);
        assert_eq!(caps.max_tools, Some(128));
    }

    #[test]
    fn merge_overlay_updates_fields() {
        let mut caps = ProviderCaps::conservative();
        let overlay = ProviderCapsOverlay {
            supports_ref: Some(true),
            max_tools: Some(50),
            ..Default::default()
        };
        caps.merge(&overlay);
        assert!(caps.supports_ref);
        assert_eq!(caps.max_tools, Some(50));
        // Unset fields remain unchanged
        assert!(!caps.supports_nullable);
    }

    #[test]
    fn merge_overrides_on_map() {
        let mut map = ProviderCapabilityMap::default();
        let mut overrides = HashMap::new();
        overrides.insert(
            "openai".to_string(),
            ProviderCapsOverlay {
                max_tools: Some(256),
                ..Default::default()
            },
        );
        overrides.insert(
            "custom_provider".to_string(),
            ProviderCapsOverlay {
                supports_ref: Some(true),
                supports_prompt_caching: Some(true),
                ..Default::default()
            },
        );
        map.merge_overrides(&overrides);

        // OpenAI max_tools updated
        assert_eq!(map.get("openai").max_tools, Some(256));
        // OpenAI other fields preserved
        assert!(map.get("openai").supports_ref);

        // Custom provider created from conservative + overlay
        let custom = map.get("custom_provider");
        assert!(custom.supports_ref);
        assert!(custom.supports_prompt_caching);
        assert!(!custom.supports_nullable);
    }
}
