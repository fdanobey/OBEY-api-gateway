//! Jev (System One) complexity classifier: typed decision-model integration
//! for smart routing.
//!
//! Submodules:
//! - `models`: typed System One evaluation and model-listing payloads.
//! - `rubric`: the fixed question set and privacy-bounded state assembly.
//! - `compose`: pure answer composition with confidence gating.
//! - `discovery`: model resolution with TTL cache.
//! - `client`: HTTP client with retry/backoff.
//! - `classifier`: the `JevClassifier` facade implementing `OptionalClassifier`.

pub mod classifier;
pub mod client;
pub mod compose;
pub mod discovery;
pub mod models;
pub mod rubric;

// Re-export the classifier facade for ergonomic imports.
pub use classifier::JevClassifier;
pub use rubric::visit_message_text;
