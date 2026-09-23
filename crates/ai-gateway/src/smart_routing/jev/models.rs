//! Typed request/response models for the TypeSafe System One evaluation
//! endpoint (`POST {base_url}/v1/systemone`) and the [OI]-compatible model
//! listing (`GET {base_url}/v1/models`).
//!
//! All `Debug` implementations redact payload content: classification state
//! carries user request text and must never appear in logs or diagnostics.

use std::collections::HashMap;
use std::fmt;

use serde::{Deserialize, Serialize};
use serde_json::Value;

/// Evaluation request: state plus a map of typed questions.
#[derive(Clone, Serialize)]
pub struct SystemOneRequest {
    /// Content to evaluate: role-tagged, char-bounded message excerpts plus
    /// content-free metadata.
    pub state: Value,
    /// Resolved model identifier (never `auto`).
    pub model: String,
    /// Question map; answers return under the same keys.
    pub questions: HashMap<String, Question>,
}

impl fmt::Debug for SystemOneRequest {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter
            .debug_struct("SystemOneRequest")
            .field("state", &"<redacted>")
            .field("model", &self.model)
            .field("questions", &self.question_ids())
            .finish()
    }
}

impl SystemOneRequest {
    fn question_ids(&self) -> Vec<&str> {
        let mut ids: Vec<&str> = self.questions.keys().map(String::as_str).collect();
        ids.sort_unstable();
        ids
    }
}

/// One typed question.
///
/// `instructions` and `criteria` are `serde_json::Value` because the API
/// accepts strings or structured objects at those positions; rubric
/// construction is the sole producer.
#[derive(Clone, Serialize)]
pub struct Question {
    /// `"score"` or `"choice"`.
    #[serde(rename = "type")]
    pub question_type: &'static str,
    pub instructions: Value,
    pub criteria: Value,
}

impl fmt::Debug for Question {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter
            .debug_struct("Question")
            .field("type", &self.question_type)
            .field("instructions", &"<redacted>")
            .field("criteria", &"<redacted>")
            .finish()
    }
}

/// Evaluation response: one answer per question.
#[derive(Debug, Clone, Deserialize)]
pub struct SystemOneResponse {
    pub model: String,
    pub answers: HashMap<String, Answer>,
    #[serde(default)]
    pub usage: Usage,
}

/// Token usage reported by the endpoint.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq, Deserialize)]
pub struct Usage {
    #[serde(default)]
    pub input_tokens: u64,
    #[serde(default)]
    pub output_tokens: u64,
}

/// One typed answer: score or choice, each with a probability distribution
/// and confidence.
#[derive(Debug, Clone, Deserialize)]
#[serde(tag = "type", rename_all = "lowercase")]
pub enum Answer {
    Score {
        /// Probability-weighted position across levels; can land between
        /// levels.
        score: f64,
        #[serde(default)]
        confidence: f64,
        #[serde(default)]
        probabilities: HashMap<String, f64>,
        #[serde(default)]
        legend: HashMap<String, Value>,
    },
    Choice {
        /// Highest-probability option.
        choice: String,
        #[serde(default)]
        probabilities: HashMap<String, f64>,
        #[serde(default)]
        confidence: f64,
    },
    /// Yes/no probability; the rubric does not ask noul questions today,
    /// but responses must tolerate them.
    Noul {
        #[serde(default)]
        noul: f64,
    },
}

impl Answer {
    /// Confidence for this answer. Score and Choice carry one; anything else
    /// has none.
    pub fn confidence(&self) -> Option<f64> {
        match self {
            Self::Score { confidence, .. } | Self::Choice { confidence, .. } => Some(*confidence),
            Self::Noul { .. } => None,
        }
    }
}

/// [OI]-compatible model listing entry.
#[derive(Debug, Clone, Deserialize)]
pub struct ModelEntry {
    pub id: String,
    #[serde(default)]
    #[allow(dead_code)]
    pub owned_by: Option<String>,
}

/// [OI]-compatible model listing response (`{ "data": [...] }`).
#[derive(Debug, Clone, Deserialize)]
pub struct ModelsListing {
    #[serde(default)]
    pub data: Vec<ModelEntry>,
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Golden response from the documented TypeSafe API example
    /// (docs.typesafe.ai/api) with a Score and a Choice answer.
    #[test]
    fn deserializes_documented_response_shape() {
        let payload = serde_json::json!({
            "model": "jev-1.13.0",
            "answers": {
                "is_urgent": { "type": "noul", "noul": 0.95 },
                "department": {
                    "type": "choice",
                    "choice": "billing",
                    "probabilities": { "billing": 0.88, "technical": 0.12, "sales": 0.0 },
                    "confidence": 0.81
                },
                "frustration": {
                    "type": "score",
                    "score": 1.05,
                    "legend": { "0": "Calm", "1": "Frustrated", "2": "Very angry" },
                    "probabilities": { "0": 0.0, "1": 0.95, "2": 0.05 },
                    "confidence": 0.92
                }
            },
            "usage": { "input_tokens": 307, "output_tokens": 20 }
        });

        let response: SystemOneResponse = serde_deserialize(payload);
        assert_eq!(response.model, "jev-1.13.0");
        assert_eq!(response.usage.input_tokens, 307);

        let frustration = response.answers.get("frustration").unwrap();
        match frustration {
            Answer::Score {
                score, confidence, ..
            } => {
                assert!((score - 1.05).abs() < f64::EPSILON);
                assert!((confidence - 0.92).abs() < f64::EPSILON);
            }
            other => panic!("expected score answer, got {other:?}"),
        }

        let department = response.answers.get("department").unwrap();
        match department {
            Answer::Choice {
                choice,
                confidence,
                probabilities,
                ..
            } => {
                assert_eq!(choice, "billing");
                assert!((confidence - 0.81).abs() < f64::EPSILON);
                assert_eq!(probabilities.len(), 3);
            }
            other => panic!("expected choice answer, got {other:?}"),
        }
    }

    #[test]
    fn tolerates_missing_usage_and_probabilities() {
        let payload = serde_json::json!({
            "model": "jev-1.13.0",
            "answers": {
                "dimension": { "type": "score", "score": 2.0, "confidence": 1.0 }
            }
        });
        let response: SystemOneResponse = serde_deserialize(payload);
        assert_eq!(response.usage, Usage::default());
        assert!(matches!(
            response.answers.get("dimension"),
            Some(Answer::Score { score, .. }) if *score == 2.0
        ));
    }

    #[test]
    fn deserializes_openai_compatible_models_listing() {
        let payload = serde_json::json!({
            "data": [
                { "id": "jev-1.13.0", "owned_by": "typesafe" },
                { "id": "typesafe/jev-1.12.2" },
                { "id": "gpt-4o-mini", "object": "model", "owned_by": "openai" }
            ]
        });
        let listing: ModelsListing = serde_deserialize(payload);
        assert_eq!(listing.data.len(), 3);
        assert_eq!(listing.data[0].id, "jev-1.13.0");
        assert!(listing.data[1].owned_by.is_none());
    }

    #[test]
    fn tolerates_empty_models_listing() {
        let listing: ModelsListing = serde_deserialize(serde_json::json!({}));
        assert!(listing.data.is_empty());
    }

    #[test]
    fn request_debug_redacts_state_and_questions() {
        let mut questions = HashMap::new();
        questions.insert(
            "reasoning_depth".to_string(),
            Question {
                question_type: "score",
                instructions: Value::String("How deep?".to_string()),
                criteria: Value::Array(vec![]),
            },
        );
        let request = SystemOneRequest {
            state: Value::String("secret user request content".to_string()),
            model: "jev-1.13.0".to_string(),
            questions,
        };

        let debug = format!("{request:?}");
        assert!(debug.contains("jev-1.13.0"));
        assert!(debug.contains("reasoning_depth"));
        assert!(!debug.contains("secret user request content"));
        assert!(!debug.contains("How deep?"));
    }

    #[test]
    fn request_serializes_question_type_as_type() {
        let mut questions = HashMap::new();
        questions.insert(
            "q".to_string(),
            Question {
                question_type: "score",
                instructions: Value::String("instructions".to_string()),
                criteria: Value::Array(vec![Value::String("level".to_string())]),
            },
        );
        let request = SystemOneRequest {
            state: Value::String("state".to_string()),
            model: "jev-latest".to_string(),
            questions,
        };
        let json = serde_json::to_value(&request).unwrap();
        assert_eq!(json["questions"]["q"]["type"], "score");
        assert_eq!(json["model"], "jev-latest");
        assert!(json.get("state").is_some());
    }

    fn serde_deserialize<T: serde::de::DeserializeOwned>(payload: serde_json::Value) -> T {
        serde_json::from_value(payload).unwrap()
    }
}
