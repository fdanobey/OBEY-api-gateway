use serde::{Deserialize, Deserializer, Serialize, Serializer};

use crate::models::openai::Message;

/// Model capability tier for smart routing.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum SmartRoutingTier {
    Fast,
    Balanced,
    Powerful,
}

impl SmartRoutingTier {
    /// Return the next more capable tier, if one exists.
    pub fn escalate(self) -> Option<Self> {
        match self {
            Self::Fast => Some(Self::Balanced),
            Self::Balanced => Some(Self::Powerful),
            Self::Powerful => None,
        }
    }
}

/// Complexity score guaranteed to be finite and within `0.0..=1.0`.
#[derive(Debug, Clone, Copy, PartialEq, PartialOrd)]
pub struct ComplexityScore(f64);

impl ComplexityScore {
    /// Create a normalized score.
    ///
    /// Finite values are clamped to `0.0..=1.0`, infinities map to the
    /// corresponding boundary, and NaN maps to the neutral fallback `0.5`.
    pub fn new(value: f64) -> Self {
        let normalized = if value.is_nan() {
            0.5
        } else if value == f64::INFINITY {
            1.0
        } else if value == f64::NEG_INFINITY {
            0.0
        } else {
            value.clamp(0.0, 1.0)
        };

        Self(normalized)
    }

    /// Return the normalized score.
    pub fn value(self) -> f64 {
        self.0
    }
}

impl Serialize for ComplexityScore {
    fn serialize<S>(&self, serializer: S) -> Result<S::Ok, S::Error>
    where
        S: Serializer,
    {
        serializer.serialize_f64(self.0)
    }
}

impl<'de> Deserialize<'de> for ComplexityScore {
    fn deserialize<D>(deserializer: D) -> Result<Self, D::Error>
    where
        D: Deserializer<'de>,
    {
        f64::deserialize(deserializer).map(Self::new)
    }
}

/// Complete routing decision produced by the smart router.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct RoutingDecision {
    pub score: ComplexityScore,
    pub adjusted_score: ComplexityScore,
    pub tier: SmartRoutingTier,
    pub task_type: TaskType,
    pub classifier: ClassifierUsed,
    pub escalated: bool,
    pub escalation_count: u8,
    pub cache_hit: bool,
    pub budget_downgraded: bool,
    pub context_filtered: bool,
}

/// Classifier backend that produced a routing decision.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum ClassifierUsed {
    Heuristic,
    Ml,
    Llm,
    Composite,
}

/// Detected task category used for specialist routing.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum TaskType {
    CodeGeneration,
    MathReasoning,
    CreativeWriting,
    #[serde(rename = "factual_qa")]
    FactualQA,
    ToolUse,
    Summarization,
    General,
}

impl TaskType {
    /// Conservatively detect a task category from existing request messages.
    ///
    /// Ambiguous input is resolved using this fixed precedence:
    /// tool use, code generation, math reasoning, summarization, creative
    /// writing, factual Q&A, then general.
    pub fn detect(messages: &[Message]) -> Self {
        crate::smart_routing::heuristic::HeuristicScorer::default()
            .score(messages)
            .task_type
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use proptest::prelude::*;
    use serde_json::{json, Map, Value};

    fn message(content: &str) -> Message {
        Message {
            role: "user".to_string(),
            content: Value::String(content.to_string()),
            extra: Map::new(),
        }
    }

    #[test]
    fn tiers_escalate_in_order() {
        assert_eq!(
            SmartRoutingTier::Fast.escalate(),
            Some(SmartRoutingTier::Balanced)
        );
        assert_eq!(
            SmartRoutingTier::Balanced.escalate(),
            Some(SmartRoutingTier::Powerful)
        );
        assert_eq!(SmartRoutingTier::Powerful.escalate(), None);
    }

    #[test]
    fn finite_scores_are_clamped() {
        assert_eq!(ComplexityScore::new(-1.0).value(), 0.0);
        assert_eq!(ComplexityScore::new(0.25).value(), 0.25);
        assert_eq!(ComplexityScore::new(2.0).value(), 1.0);
    }

    #[test]
    fn nan_uses_neutral_fallback() {
        assert_eq!(ComplexityScore::new(f64::NAN).value(), 0.5);
    }

    #[test]
    fn infinities_map_to_boundaries() {
        assert_eq!(ComplexityScore::new(f64::INFINITY).value(), 1.0);
        assert_eq!(ComplexityScore::new(f64::NEG_INFINITY).value(), 0.0);
    }

    proptest! {
    #![proptest_config(ProptestConfig::with_cases(64))]

    #[test]
    fn complexity_score_clamps_finite_f64(value in any::<f64>().prop_filter(
    "property covers finite f64 values only",
    |value| value.is_finite(),
    )) {
    let score = ComplexityScore::new(value).value();
    let expected = value.clamp(0.0, 1.0);

    prop_assert!(score.is_finite());
    prop_assert!((0.0..=1.0).contains(&score));
    prop_assert_eq!(score, expected);
    }
    }

    #[test]
    fn deserialization_preserves_score_invariant() {
        let score: ComplexityScore = serde_json::from_str("2.5").unwrap();
        assert_eq!(score.value(), 1.0);
    }

    #[test]
    fn serde_uses_compatible_enum_names() {
        assert_eq!(
            serde_json::to_value(SmartRoutingTier::Powerful).unwrap(),
            json!("powerful")
        );
        assert_eq!(
            serde_json::to_value(ClassifierUsed::Llm).unwrap(),
            json!("llm")
        );
        assert_eq!(
            serde_json::to_value(TaskType::CodeGeneration).unwrap(),
            json!("code_generation")
        );
        assert_eq!(
            serde_json::from_value::<TaskType>(json!("factual_qa")).unwrap(),
            TaskType::FactualQA
        );
    }

    #[test]
    fn task_detection_uses_required_precedence() {
        let mut tool_message = message(
            "Write code to solve the equation, summarize it, and turn it into a poem. What is it?",
        );
        tool_message.role = "tool".to_string();
        assert_eq!(TaskType::detect(&[tool_message]), TaskType::ToolUse);

        assert_eq!(
            TaskType::detect(&[message(
                "Write code to solve the equation, then summarize it as a poem."
            )]),
            TaskType::CodeGeneration
        );
        assert_eq!(
            TaskType::detect(&[message(
                "Solve the equation, summarize the answer, then write a poem."
            )]),
            TaskType::MathReasoning
        );
        assert_eq!(
            TaskType::detect(&[message("Summarize this and write a poem.")]),
            TaskType::Summarization
        );
        assert_eq!(
            TaskType::detect(&[message("Write a poem. What is the theme?")]),
            TaskType::CreativeWriting
        );
        assert_eq!(
            TaskType::detect(&[message("What is the capital of France?")]),
            TaskType::FactualQA
        );
        assert_eq!(
            TaskType::detect(&[message("Hello there")]),
            TaskType::General
        );
    }
}
