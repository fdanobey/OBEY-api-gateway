//! Jev (TypeSafe) System One family: a thin façade over the shared
//! `systemone` core bound to [`SystemOneFamily::JEV`].
//!
//! All family-neutral behavior — wire types, rubric, composition, discovery,
//! transport, and the classifier facade — lives in
//! [`crate::smart_routing::systemone`]; this module preserves the historical
//! `jev::` import paths and adds the Jev-config adapters.

pub mod classifier;
pub mod client;
pub mod compose;
pub mod discovery;
pub mod models;
pub mod rubric;

pub use crate::smart_routing::systemone::SystemOneFamily;

/// The Jev family descriptor.
pub const FAMILY: SystemOneFamily = SystemOneFamily::JEV;

// Re-export the classifier facade for ergonomic imports.
pub use classifier::JevClassifier;
pub use rubric::visit_message_text;
