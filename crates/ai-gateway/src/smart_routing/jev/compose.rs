//! Jev composition: re-exports of the shared System One composition with the
//! Jev confidence selection applied.

pub use crate::smart_routing::systemone::compose::{
    blend_scores, dimension_answer, parse_task_type, ComposeResult, JevTrustConfig,
};

use std::collections::HashMap;

use crate::smart_routing::config::DimensionWeights;
use crate::smart_routing::systemone::SystemOneFamily;

use super::models::Answer;

/// Compose answers into a complexity score and task type using Jev
/// confidence semantics (the `confidence` field).
pub fn compose(
    answers: &HashMap<String, Answer>,
    weights: &DimensionWeights,
    trust: &JevTrustConfig,
) -> ComposeResult {
    crate::smart_routing::systemone::compose::compose(
        answers,
        weights,
        trust,
        &SystemOneFamily::JEV,
    )
}
