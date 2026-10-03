//! Jev wire models: re-exports of the shared System One types plus the
//! family-bound helper preserving the historical `is_jev_model` contract.

pub use crate::smart_routing::systemone::models::{
    Answer, ModelEntry, ModelsListing, Question, SystemOneRequest, SystemOneResponse, Usage,
};

use super::FAMILY;

/// True when a model id is Jev-capable: a word-delimited, case-insensitive
/// `jev` token in the id (e.g. `jev-1.13.0`, `typesafe/jev-1.13`, `Jev`).
/// `jealous-model` and `notjev` do not qualify.
pub fn is_jev_model(id: &str) -> bool {
    FAMILY.is_family_model(id)
}
