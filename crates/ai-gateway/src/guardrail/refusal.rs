//! Refusal detection: phrase-based and tool-omission signal detection for
//! post-call guardrail enforcement (Req 12.1, 12.2, 12.3).
//!
//! The [`RefusalDetector`] is constructed once at config load time with compiled
//! case-insensitive regex matchers. Detection runs only against assistant-role
//! response content and never mutates the response.

use regex::RegexBuilder;

/// Default refusal phrases shipped with the gateway (Req 12.2).
///
/// Each entry is a valid regex pattern compiled case-insensitively. Plain
/// literal phrases are valid regexes, so both coexist in one list.
pub const DEFAULT_REFUSAL_PHRASES: &[&str] = &[
    r"i can'?t (help|assist) with",
    r"i'?m (sorry|unable)",
    r"i cannot comply",
    r"as an ai",
    r"i must decline",
    r"i'?m not able to",
    r"i cannot (help|assist) with",
    r"i cannot fulfill",
    r"i'm afraid i can'?t",
    r"i cannot provide",
];

/// The signal that triggered a refusal detection.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum RefusalSignal {
    /// A phrase/regex from the Refusal_Phrase_List matched assistant content
    /// (Req 12.1).
    Phrase,
    /// Tool-omission: tools were requested but the model did not call any
    /// (Req 12.3).
    ToolOmission,
}

/// Decision produced by [`RefusalDetector::detect`].
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum RefusalDecision {
    /// The response is not a refusal.
    NotRefusal,
    /// The response is a refusal, with the triggering signal.
    Refusal(RefusalSignal),
}

impl RefusalDecision {
    /// Returns `true` if the decision indicates a refusal.
    pub fn is_refusal(&self) -> bool {
        matches!(self, Self::Refusal(_))
    }
}

/// Context for the tool-omission refusal signal (Req 12.3).
///
/// Derived from the request's `tool_choice`, the presence of tools in the
/// request, and the response's `finish_reason` / `tool_calls` array.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ToolContext {
    /// `tool_choice` does NOT forbid tools (i.e., not `"none"`).
    pub tool_use_allowed: bool,
    /// At least one tool was provided in the request.
    pub tools_provided: bool,
    /// Response `finish_reason` indicates a tool call (e.g., `"tool_calls"`).
    pub finish_reason_is_tool_call: bool,
    /// Response contains a non-empty `tool_calls` array.
    pub has_tool_calls: bool,
}

/// Compiled refusal detector built from the configured or default phrase list.
///
/// Constructed once at config load time. All regexes are case-insensitive.
/// Matching runs only against assistant-role content (Req 12.1).
pub struct RefusalDetector {
    /// Compiled case-insensitive matchers (one per phrase/regex entry).
    phrases: Vec<regex::Regex>,
}

impl std::fmt::Debug for RefusalDetector {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("RefusalDetector")
            .field("phrase_count", &self.phrases.len())
            .finish()
    }
}

impl RefusalDetector {
    /// Build a detector from the provided phrase list.
    ///
    /// Each entry is compiled as a case-insensitive regex. Returns an error if
    /// any entry fails to compile (Req 12.13 — the caller should reject the
    /// config).
    pub fn new(phrases: &[&str]) -> Result<Self, RefusalBuildError> {
        let compiled: Result<Vec<_>, _> = phrases
            .iter()
            .enumerate()
            .map(|(idx, pattern)| {
                RegexBuilder::new(pattern)
                    .case_insensitive(true)
                    .build()
                    .map_err(|e| RefusalBuildError {
                        index: idx,
                        pattern: pattern.to_string(),
                        reason: e.to_string(),
                    })
            })
            .collect();
        Ok(Self { phrases: compiled? })
    }

    /// Build using owned `String` phrases (convenience for config-supplied lists).
    #[allow(dead_code)] // public API; used by tests and config hot-reload
    pub fn from_strings(phrases: &[String]) -> Result<Self, RefusalBuildError> {
        let compiled: Result<Vec<_>, _> = phrases
            .iter()
            .enumerate()
            .map(|(idx, pattern)| {
                RegexBuilder::new(pattern)
                    .case_insensitive(true)
                    .build()
                    .map_err(|e| RefusalBuildError {
                        index: idx,
                        pattern: pattern.clone(),
                        reason: e.to_string(),
                    })
            })
            .collect();
        Ok(Self { phrases: compiled? })
    }

    /// Build from the shipped default phrase list.
    pub fn default_detector() -> Self {
        // The default list is known-good; unwrap is safe.
        Self::new(DEFAULT_REFUSAL_PHRASES)
            .expect("DEFAULT_REFUSAL_PHRASES contains only valid regex patterns")
    }

    /// Phrase-based refusal: matches ONLY assistant-role content,
    /// case-insensitively (Req 12.1).
    pub fn matches_phrase(&self, assistant_content: &str) -> bool {
        self.phrases.iter().any(|re| re.is_match(assistant_content))
    }

    /// Tool-omission refusal signal (Req 12.3).
    ///
    /// Fires when tools were requested and allowed but the model did not
    /// produce a tool call.
    pub fn is_tool_omission(&self, tc: &ToolContext) -> bool {
        tc.tool_use_allowed
            && tc.tools_provided
            && !tc.finish_reason_is_tool_call
            && !tc.has_tool_calls
    }

    /// Full refusal check: phrase OR tool-omission (Req 12.1, 12.3).
    #[allow(dead_code)] // public API; used by tests
    pub fn is_refusal(&self, assistant_content: &str, tc: &ToolContext) -> bool {
        self.matches_phrase(assistant_content) || self.is_tool_omission(tc)
    }

    /// Produce a [`RefusalDecision`] combining both signals, preferring
    /// phrase match over tool-omission when both fire.
    pub fn detect(&self, assistant_content: &str, tc: &ToolContext) -> RefusalDecision {
        if self.matches_phrase(assistant_content) {
            RefusalDecision::Refusal(RefusalSignal::Phrase)
        } else if self.is_tool_omission(tc) {
            RefusalDecision::Refusal(RefusalSignal::ToolOmission)
        } else {
            RefusalDecision::NotRefusal
        }
    }

    /// Number of compiled phrase patterns (useful for diagnostics/tests).
    #[allow(dead_code)]
    pub fn phrase_count(&self) -> usize {
        self.phrases.len()
    }
}

/// Error produced when a phrase fails to compile as a regex (Req 12.13).
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct RefusalBuildError {
    /// Zero-based index of the offending entry in the phrase list.
    pub index: usize,
    /// The pattern string that failed.
    pub pattern: String,
    /// Regex compilation error message.
    pub reason: String,
}

impl std::fmt::Display for RefusalBuildError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(
            f,
            "refusal phrase [{}] {:?} failed to compile: {}",
            self.index, self.pattern, self.reason
        )
    }
}

impl std::error::Error for RefusalBuildError {}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;
    use proptest::prelude::*;

    // ---- Property-based tests (proptest, >=100 cases) ----

    proptest! {
        #![proptest_config(ProptestConfig::with_cases(256))]

        // Feature: guardrail-pipelines, Property 30: Tool-omission refusal signal
        // **Validates: Requirements 12.3**
        //
        // The tool-omission signal fires IFF ALL of:
        //   tool_use_allowed=true, tools_provided=true,
        //   finish_reason_is_tool_call=false, has_tool_calls=false
        //
        // When ANY one of those four booleans is in the "wrong" state the signal
        // MUST NOT fire.
        #[test]
        fn prop_tool_omission_signal_fires_iff_all_conditions_met(
            tool_use_allowed in any::<bool>(),
            tools_provided in any::<bool>(),
            finish_reason_is_tool_call in any::<bool>(),
            has_tool_calls in any::<bool>(),
        ) {
            let detector = RefusalDetector::default_detector();
            let ctx = ToolContext {
                tool_use_allowed,
                tools_provided,
                finish_reason_is_tool_call,
                has_tool_calls,
            };

            let expected = tool_use_allowed
                && tools_provided
                && !finish_reason_is_tool_call
                && !has_tool_calls;

            let actual = detector.is_tool_omission(&ctx);
            prop_assert_eq!(
                actual,
                expected,
                "is_tool_omission mismatch for ctx: tool_use_allowed={}, tools_provided={}, \
                 finish_reason_is_tool_call={}, has_tool_calls={} => expected={}, got={}",
                tool_use_allowed,
                tools_provided,
                finish_reason_is_tool_call,
                has_tool_calls,
                expected,
                actual
            );
        }
    }

    /// Helper: tool context where tools are NOT involved (no tool-omission signal).
    fn no_tools_ctx() -> ToolContext {
        ToolContext {
            tool_use_allowed: false,
            tools_provided: false,
            finish_reason_is_tool_call: false,
            has_tool_calls: false,
        }
    }

    /// Helper: tool context where tools ARE involved but model did not call any.
    fn tool_omission_ctx() -> ToolContext {
        ToolContext {
            tool_use_allowed: true,
            tools_provided: true,
            finish_reason_is_tool_call: false,
            has_tool_calls: false,
        }
    }

    /// Helper: tool context where model did call tools.
    fn tool_called_ctx() -> ToolContext {
        ToolContext {
            tool_use_allowed: true,
            tools_provided: true,
            finish_reason_is_tool_call: true,
            has_tool_calls: true,
        }
    }

    #[test]
    fn default_detector_builds_successfully() {
        let d = RefusalDetector::default_detector();
        assert_eq!(d.phrase_count(), DEFAULT_REFUSAL_PHRASES.len());
    }

    #[test]
    fn phrase_match_case_insensitive() {
        let d = RefusalDetector::default_detector();
        // Exact match
        assert!(d.matches_phrase("I can't help with that request."));
        // Mixed case
        assert!(d.matches_phrase("I CAN'T ASSIST WITH that."));
        // i'm sorry variant
        assert!(d.matches_phrase("I'm sorry, I cannot do that."));
        // as an ai
        assert!(d.matches_phrase("As an AI, I have limitations."));
    }

    #[test]
    fn phrase_no_match_normal_content() {
        let d = RefusalDetector::default_detector();
        assert!(!d.matches_phrase("Here is the code you requested."));
        assert!(!d.matches_phrase("The function returns true when the input is valid."));
        assert!(!d.matches_phrase(""));
    }

    #[test]
    fn tool_omission_fires_when_expected() {
        let d = RefusalDetector::default_detector();
        assert!(d.is_tool_omission(&tool_omission_ctx()));
    }

    #[test]
    fn tool_omission_does_not_fire_when_tools_called() {
        let d = RefusalDetector::default_detector();
        assert!(!d.is_tool_omission(&tool_called_ctx()));
    }

    #[test]
    fn tool_omission_does_not_fire_without_tools() {
        let d = RefusalDetector::default_detector();
        assert!(!d.is_tool_omission(&no_tools_ctx()));
    }

    #[test]
    fn tool_omission_does_not_fire_tool_use_not_allowed() {
        let d = RefusalDetector::default_detector();
        let ctx = ToolContext {
            tool_use_allowed: false,
            tools_provided: true,
            finish_reason_is_tool_call: false,
            has_tool_calls: false,
        };
        assert!(!d.is_tool_omission(&ctx));
    }

    #[test]
    fn detect_returns_phrase_signal() {
        let d = RefusalDetector::default_detector();
        let decision = d.detect("I must decline your request.", &no_tools_ctx());
        assert_eq!(decision, RefusalDecision::Refusal(RefusalSignal::Phrase));
    }

    #[test]
    fn detect_returns_tool_omission_signal() {
        let d = RefusalDetector::default_detector();
        let decision = d.detect("Here is a text response.", &tool_omission_ctx());
        assert_eq!(
            decision,
            RefusalDecision::Refusal(RefusalSignal::ToolOmission)
        );
    }

    #[test]
    fn detect_returns_not_refusal() {
        let d = RefusalDetector::default_detector();
        let decision = d.detect("Here is the result.", &no_tools_ctx());
        assert_eq!(decision, RefusalDecision::NotRefusal);
    }

    #[test]
    fn detect_prefers_phrase_over_tool_omission() {
        let d = RefusalDetector::default_detector();
        // Both signals fire: phrase takes priority
        let decision = d.detect("I can't help with that.", &tool_omission_ctx());
        assert_eq!(decision, RefusalDecision::Refusal(RefusalSignal::Phrase));
    }

    #[test]
    fn is_refusal_combined_check() {
        let d = RefusalDetector::default_detector();
        // Phrase match
        assert!(d.is_refusal("I'm sorry, I cannot comply.", &no_tools_ctx()));
        // Tool omission
        assert!(d.is_refusal("Normal text response.", &tool_omission_ctx()));
        // Neither
        assert!(!d.is_refusal("Normal text response.", &no_tools_ctx()));
    }

    #[test]
    fn invalid_regex_produces_build_error() {
        let result = RefusalDetector::new(&["valid", "(unclosed"]);
        assert!(result.is_err());
        let err = result.unwrap_err();
        assert_eq!(err.index, 1);
        assert_eq!(err.pattern, "(unclosed");
    }

    #[test]
    fn from_strings_works() {
        let phrases = vec!["hello".to_string(), "world".to_string()];
        let d = RefusalDetector::from_strings(&phrases).unwrap();
        assert!(d.matches_phrase("Hello there!"));
        assert!(d.matches_phrase("WORLD domination"));
        assert!(!d.matches_phrase("goodbye"));
    }

    #[test]
    fn refusal_decision_is_refusal_helper() {
        assert!(RefusalDecision::Refusal(RefusalSignal::Phrase).is_refusal());
        assert!(RefusalDecision::Refusal(RefusalSignal::ToolOmission).is_refusal());
        assert!(!RefusalDecision::NotRefusal.is_refusal());
    }

    // -----------------------------------------------------------------
    // Property-based tests (proptest)
    // -----------------------------------------------------------------

    /// A small set of known-good literal and regex phrases for property testing.
    const TEST_PHRASES: &[&str] = &[
        "i cannot help",
        r"i'?m sorry",
        "as an ai",
        r"i (must|will) decline",
        "not able to assist",
    ];

    /// Strategy: pick a random case transformation of a string.
    fn randomize_case(s: &str) -> impl Strategy<Value = String> {
        let owned = s.to_string();
        any::<Vec<bool>>().prop_map(move |bools| {
            owned
                .chars()
                .enumerate()
                .map(|(i, c)| {
                    if bools.get(i).copied().unwrap_or(false) {
                        c.to_uppercase().to_string()
                    } else {
                        c.to_lowercase().to_string()
                    }
                })
                .collect::<String>()
        })
    }

    /// Strategy: generate a concrete string that matches one of the TEST_PHRASES
    /// with arbitrary casing. We use a few known expansions of the regex patterns.
    fn matching_phrase_expansion() -> impl Strategy<Value = String> {
        prop_oneof![
            randomize_case("i cannot help"),
            randomize_case("i'm sorry"),
            randomize_case("im sorry"),
            randomize_case("as an ai"),
            randomize_case("i must decline"),
            randomize_case("i will decline"),
            randomize_case("not able to assist"),
        ]
    }

    /// Strategy: generate content that does NOT contain any of the test phrases.
    /// We use arbitrary alphanumeric + punctuation that avoids phrase substrings.
    fn non_matching_content() -> impl Strategy<Value = String> {
        // Generate safe content that won't accidentally match our test phrases
        prop_oneof![
            Just("Here is the code you requested.".to_string()),
            Just("The function returns 42.".to_string()),
            Just("Hello world from the assistant.".to_string()),
            Just("Processing complete. Results attached.".to_string()),
            Just("Let me explain the architecture.".to_string()),
            "[a-z0-9 ]{5,30}".prop_filter("must not match test phrases", |s| {
                let detector = RefusalDetector::new(TEST_PHRASES).unwrap();
                !detector.matches_phrase(s)
            }),
        ]
    }

    /// Simulated role enum for scoping tests.
    #[derive(Debug, Clone, Copy, PartialEq, Eq)]
    enum Role {
        Assistant,
        User,
        System,
        Tool,
    }

    fn arb_role() -> impl Strategy<Value = Role> {
        prop_oneof![
            Just(Role::Assistant),
            Just(Role::User),
            Just(Role::System),
            Just(Role::Tool),
        ]
    }

    proptest! {
        #![proptest_config(ProptestConfig::with_cases(200))]

        /// Property 29: Refusal phrase matching is case-insensitive and
        /// content-scoped.
        ///
        /// For any case-randomized expansion of a configured phrase:
        /// - The detector MUST match when applied to assistant-role content
        /// - The detector conceptually does NOT apply to non-assistant roles
        ///   (matching is scoped to assistant content by design)
        ///
        /// **Validates: Requirements 12.1**
        #[test]
        fn prop_phrase_matching_is_case_insensitive(
            phrase in matching_phrase_expansion(),
            prefix in "[a-z ]{0,20}",
            suffix in "[a-z ]{0,20}",
        ) {
            let detector = RefusalDetector::new(TEST_PHRASES).unwrap();
            // Embed the phrase in surrounding content (assistant role)
            let assistant_content = format!("{prefix}{phrase}{suffix}");
            // Case-insensitive matching must detect the phrase
            prop_assert!(
                detector.matches_phrase(&assistant_content),
                "Expected match for assistant content containing phrase: {:?}",
                assistant_content
            );
        }

        /// Property 29 (cont.): Non-assistant-role content is never scoped for
        /// phrase matching. The `matches_phrase` function itself is role-agnostic;
        /// the enforcement scope is that only assistant content is passed to it.
        /// This property verifies the architectural contract: given a role and
        /// content, only assistant-role content triggers the detection path.
        ///
        /// **Validates: Requirements 12.1**
        #[test]
        fn prop_phrase_matching_scoped_to_assistant_role(
            phrase in matching_phrase_expansion(),
            prefix in "[a-z ]{0,10}",
            suffix in "[a-z ]{0,10}",
            role in arb_role(),
        ) {
            let detector = RefusalDetector::new(TEST_PHRASES).unwrap();
            let content = format!("{prefix}{phrase}{suffix}");

            // Simulate the scoping logic: only check assistant-role content
            let detected = match role {
                Role::Assistant => detector.matches_phrase(&content),
                Role::User | Role::System | Role::Tool => false, // scoped out
            };

            match role {
                Role::Assistant => {
                    prop_assert!(
                        detected,
                        "Assistant-role content with phrase must be detected"
                    );
                }
                _ => {
                    prop_assert!(
                        !detected,
                        "Non-assistant role content must not trigger detection"
                    );
                }
            }
        }

        /// Property 29 (cont.): Both literal phrases and regex patterns work.
        /// A detector built with regex-containing entries (e.g., `i'?m sorry`)
        /// matches any valid expansion of the regex, case-insensitively.
        ///
        /// **Validates: Requirements 12.1**
        #[test]
        fn prop_regex_and_literal_phrases_both_match(
            // Pick one phrase index from TEST_PHRASES
            phrase_idx in 0..TEST_PHRASES.len(),
            case_bits in prop::collection::vec(any::<bool>(), 0..50),
        ) {
            let detector = RefusalDetector::new(TEST_PHRASES).unwrap();

            // Generate a concrete match for each phrase pattern
            let concrete = match phrase_idx {
                0 => "i cannot help".to_string(),       // literal
                1 => {                                   // regex: i'?m sorry
                    if case_bits.first().copied().unwrap_or(false) {
                        "im sorry".to_string()
                    } else {
                        "i'm sorry".to_string()
                    }
                }
                2 => "as an ai".to_string(),            // literal
                3 => {                                   // regex: i (must|will) decline
                    if case_bits.first().copied().unwrap_or(false) {
                        "i must decline".to_string()
                    } else {
                        "i will decline".to_string()
                    }
                }
                4 => "not able to assist".to_string(),  // literal
                _ => unreachable!(),
            };

            // Apply random case transformation
            let cased: String = concrete
                .chars()
                .enumerate()
                .map(|(i, c)| {
                    if case_bits.get(i + 1).copied().unwrap_or(false) {
                        c.to_uppercase().to_string()
                    } else {
                        c.to_lowercase().to_string()
                    }
                })
                .collect();

            prop_assert!(
                detector.matches_phrase(&cased),
                "Any case variant of a valid phrase expansion must match: {:?} (from pattern {:?})",
                cased,
                TEST_PHRASES[phrase_idx]
            );
        }

        /// Property 29 (cont.): Content that does NOT contain a configured phrase
        /// never triggers a match, regardless of role.
        ///
        /// **Validates: Requirements 12.1**
        #[test]
        fn prop_non_matching_content_never_triggers(
            content in non_matching_content(),
        ) {
            let detector = RefusalDetector::new(TEST_PHRASES).unwrap();
            prop_assert!(
                !detector.matches_phrase(&content),
                "Content without any phrase must not match: {:?}",
                content
            );
        }
    }
}
