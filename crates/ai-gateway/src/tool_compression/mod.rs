//! Tool Definition Compression middleware.
//!
//! This module implements a Tower middleware layer that intercepts the `tools`
//! array in OpenAI-format chat completion requests and applies configurable
//! compression strategies to reduce token waste from tool definitions.
//!
//! When disabled (the default), the middleware contributes zero overhead.

pub mod config;
pub mod middleware;
pub mod stage;
pub mod stages;
pub mod state;
pub mod tfidf;
pub mod types;
pub mod usage;
pub mod validation;

#[cfg(test)]
mod integration_tests;

pub use config::ToolCompressionConfig;
pub use middleware::{
    OriginalToolsExtension, ToolCompressionApplied, ToolCompressionLayer, ToolCompressionService,
};
pub use stage::CompressionStage;
pub use state::ToolCompressionState;
pub use types::{
    ApiKeyId, CompressionContext, DisclosureSet, ProviderCapabilityMap, ProviderCaps, SessionId,
    ToolDefinition, ToolUsageMap,
};
pub use tfidf::TfIdfScorer;
pub use usage::{KeyUsageState, UsageTracker};
