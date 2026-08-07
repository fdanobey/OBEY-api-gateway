//! Core compression pipeline trait.
//!
//! Each stage in the tool compression pipeline implements [`CompressionStage`],
//! receiving the mutable tools array and shared context to apply its
//! transformation in-place.

use super::config::{CompressionLevel, ToolCompressionConfig};
use super::types::{CompressionContext, ToolDefinition};

/// A single stage in the compression pipeline.
///
/// Each stage receives a mutable tools array and compression context,
/// applying its transformation in-place.
pub trait CompressionStage: Send + Sync {
    /// Apply this stage's compression to the tools array.
    /// Returns the number of tokens estimated saved by this stage.
    fn apply(&self, tools: &mut Vec<ToolDefinition>, ctx: &mut CompressionContext) -> u64;

    /// Whether this stage is enabled given the current config and level.
    fn is_enabled(&self, config: &ToolCompressionConfig, level: CompressionLevel) -> bool;
}
