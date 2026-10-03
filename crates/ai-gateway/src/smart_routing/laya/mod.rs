//! Laya (Convai Innovations) System One family: a thin façade over the
//! shared `systemone` core bound to [`SystemOneFamily::LAYA`].
//!
//! Laya differences captured by the descriptor: hosted default
//! `https://api.laya-ai.com`, 503 additionally retryable, confidence gated
//! on `answer_confidence` (falling back to `confidence`), `laya` model-id
//! token, and the `typed-decisions` default checkpoint pinned when the
//! endpoint exposes no model listing.

pub mod classifier;

pub use crate::smart_routing::systemone::SystemOneFamily;

/// The Laya family descriptor.
pub const FAMILY: SystemOneFamily = SystemOneFamily::LAYA;

// Re-export the classifier facade for ergonomic imports.
pub use classifier::LayaClassifier;
