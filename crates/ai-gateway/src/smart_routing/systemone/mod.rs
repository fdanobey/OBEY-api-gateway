//! Shared System One foundation: the family-neutral wire protocol, rubric,
//! composition, discovery, client, and classifier core used by both the Jev
//! and Laya decision-model families.
//!
//! Family differences are captured by the small immutable [`SystemOneFamily`]
//! descriptor — defaults, retryable statuses, confidence-field selection,
//! model-id token, and the missing-listing default — so the shared
//! implementation stays parameterized without dynamic dispatch on the hot
//! path.

pub mod classifier;
pub mod client;
pub mod compose;
pub mod discovery;
pub mod models;
pub mod rubric;

/// How a family selects the gating confidence for an answer.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ConfidenceSelection {
    /// Gate on the `confidence` field (Jev).
    Confidence,
    /// Gate on `answer_confidence` when present, falling back to
    /// `confidence` (Laya).
    AnswerConfidenceOrConfidence,
}

/// Immutable description of one System One provider family.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct SystemOneFamily {
    /// Family name used in logs, metrics labels, and config scopes.
    pub name: &'static str,
    /// Hosted endpoint default base URL.
    pub default_base_url: &'static str,
    /// Model used when the operator sets `model: auto` and the endpoint does
    /// not expose a model listing.
    pub default_model: &'static str,
    /// Transient HTTP statuses that merit a bounded retry.
    pub retryable_statuses: &'static [u16],
    /// Which confidence field composition gates on.
    pub confidence_selection: ConfidenceSelection,
    /// Word-delimited token identifying family model ids (e.g. `jev`,
    /// `laya`).
    pub model_token: &'static str,
    /// Model pinned when the endpoint exposes no model listing, if the
    /// family defines one.
    pub missing_listing_default: Option<&'static str>,
}

impl SystemOneFamily {
    /// The Jev (TypeSafe) family.
    pub const JEV: Self = Self {
        name: "jev",
        default_base_url: "https://api.typesafe.ai",
        default_model: "auto",
        retryable_statuses: &[429, 529],
        confidence_selection: ConfidenceSelection::Confidence,
        model_token: "jev",
        missing_listing_default: None,
    };

    /// The Laya (Convai Innovations) family.
    pub const LAYA: Self = Self {
        name: "laya",
        default_base_url: "https://api.laya-ai.com",
        default_model: "auto",
        retryable_statuses: &[429, 529, 503],
        confidence_selection: ConfidenceSelection::AnswerConfidenceOrConfidence,
        model_token: "laya",
        missing_listing_default: Some("typed-decisions"),
    };
}

impl SystemOneFamily {
    /// True when a model id belongs to this family: a word-delimited,
    /// case-insensitive family token in the id (`jev-1.13.0`,
    /// `typesafe/jev-1.13`, `laya-2.0.0`, `convai/laya-english`).
    /// `jealous-model`, `notjev`, and `layabout` do not qualify.
    pub fn is_family_model(&self, id: &str) -> bool {
        id.split(|character: char| !character.is_ascii_alphanumeric())
            .any(|token| token.eq_ignore_ascii_case(self.model_token))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn jev_token_detection_is_word_delimited() {
        assert!(SystemOneFamily::JEV.is_family_model("jev-1.13.0"));
        assert!(SystemOneFamily::JEV.is_family_model("typesafe/jev-1.13"));
        assert!(SystemOneFamily::JEV.is_family_model("Jev"));
        for id in ["jealous-model", "notjev", "gpt-4o-mini", "jevvia", "laya-2.0.0"] {
            assert!(!SystemOneFamily::JEV.is_family_model(id), "{id}");
        }
    }

    #[test]
    fn laya_token_detection_is_word_delimited() {
        assert!(SystemOneFamily::LAYA.is_family_model("laya-2.0.0"));
        assert!(SystemOneFamily::LAYA.is_family_model("convai/laya-english"));
        assert!(SystemOneFamily::LAYA.is_family_model("Laya"));
        for id in ["layabout", "notlaya", "gpt-4o-mini", "relay-a", "jev-1.13.0"] {
            assert!(!SystemOneFamily::LAYA.is_family_model(id), "{id}");
        }
    }

    #[test]
    fn family_descriptors_are_distinct() {
        assert_ne!(SystemOneFamily::JEV, SystemOneFamily::LAYA);
        assert_eq!(SystemOneFamily::JEV.retryable_statuses, &[429, 529]);
        assert_eq!(SystemOneFamily::LAYA.retryable_statuses, &[429, 529, 503]);
        assert_eq!(
            SystemOneFamily::JEV.confidence_selection,
            ConfidenceSelection::Confidence
        );
        assert_eq!(
            SystemOneFamily::LAYA.confidence_selection,
            ConfidenceSelection::AnswerConfidenceOrConfidence
        );
        assert_eq!(SystemOneFamily::JEV.missing_listing_default, None);
        assert_eq!(
            SystemOneFamily::LAYA.missing_listing_default,
            Some("typed-decisions")
        );
    }
}
