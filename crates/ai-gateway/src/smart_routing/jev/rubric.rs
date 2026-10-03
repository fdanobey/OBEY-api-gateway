//! Jev rubric: re-export of the shared System One rubric plus the
//! Jev-config-bound `build_state` used by the classifier facade.

pub use crate::smart_routing::systemone::rubric::{
    build_questions, visit_message_text, DIMENSION_LEVELS, DIMENSION_QUESTION_IDS,
    TASK_TYPE_QUESTION_ID,
};

use crate::smart_routing::config::JevConfig;
use crate::models::openai::OpenAIRequest;
use serde_json::Value;

/// Assemble the state value sent to the endpoint from a request using the
/// Jev config's char budget.
pub fn build_state(request: &OpenAIRequest, config: &JevConfig) -> Value {
    crate::smart_routing::systemone::rubric::build_state(request, config.char_budget)
}
