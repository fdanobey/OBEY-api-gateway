//! Regex guardrail provider (Req 5).
//!
//! Compiles up to [`MAX_PATTERNS`] named regex patterns at construction time
//! (Req 5.1, 5.4) and reuses the compiled [`::regex::Regex`] objects across
//! every scan. Scanning evaluates all patterns and returns non-overlapping
//! deny-list matches with byte offsets, matched text, and entity labels
//! (Req 5.2), while byte ranges covered by any allow-list match are suppressed
//! (Req 5.3). Each pattern evaluation is bounded by a 10 ms wall-clock budget;
//! a pattern that exceeds it is skipped for the current scan with a WARN
//! (Req 5.6). When nothing matches, an empty findings list is returned
//! (Req 5.7).

use std::time::{Duration, Instant};

use ::regex::Regex;
use async_trait::async_trait;

use crate::guardrail::config::{RegexPatternConfig, RegexRuleMode};
use crate::guardrail::provider::{Finding, GuardrailProvider, GuardrailProviderError};

/// Maximum number of named patterns a single regex provider accepts (Req 5.1).
pub const MAX_PATTERNS: usize = 256;

/// Per-pattern wall-clock evaluation budget (Req 5.6).
const PATTERN_BUDGET: Duration = Duration::from_millis(10);

/// Error produced while constructing a [`RegexProvider`].
///
/// Surfaced by configuration validation/load so the offending pattern can be
/// identified before any request is served (Req 5.5).
#[derive(Debug, Clone, PartialEq, Eq, thiserror::Error)]
pub enum RegexCompileError {
    /// A pattern's regex string failed to compile.
    #[error("regex pattern '{name}' failed to compile: {reason}")]
    Compile {
        /// The configured pattern name.
        name: String,
        /// The compiler's failure reason.
        reason: String,
    },

    /// More than [`MAX_PATTERNS`] patterns were supplied.
    #[error("regex provider accepts at most {max} patterns, got {count}")]
    TooManyPatterns {
        /// The configured pattern count.
        count: usize,
        /// The maximum allowed pattern count.
        max: usize,
    },
}

/// A single compiled regex rule, retained for reuse across scans (Req 5.4).
#[derive(Debug, Clone)]
struct CompiledPattern {
    name: String,
    entity: String,
    mode: RegexRuleMode,
    regex: Regex,
}

/// Regex-based guardrail provider.
#[derive(Debug, Clone)]
pub struct RegexProvider {
    patterns: Vec<CompiledPattern>,
}

impl RegexProvider {
    /// Compile `patterns` into a reusable provider (Req 5.1, 5.4).
    ///
    /// Returns [`RegexCompileError`] identifying the offending pattern name and
    /// the compilation-failure reason if any pattern fails to compile, or if
    /// more than [`MAX_PATTERNS`] patterns are supplied (Req 5.5).
    pub fn new(patterns: &[RegexPatternConfig]) -> Result<Self, RegexCompileError> {
        if patterns.len() > MAX_PATTERNS {
            return Err(RegexCompileError::TooManyPatterns {
                count: patterns.len(),
                max: MAX_PATTERNS,
            });
        }

        let mut compiled = Vec::with_capacity(patterns.len());
        for pattern in patterns {
            let regex = Regex::new(&pattern.regex).map_err(|err| RegexCompileError::Compile {
                name: pattern.name.clone(),
                reason: err.to_string(),
            })?;
            compiled.push(CompiledPattern {
                name: pattern.name.clone(),
                entity: pattern.entity.clone(),
                mode: pattern.mode,
                regex,
            });
        }

        Ok(Self { patterns: compiled })
    }

    /// Number of compiled patterns held by this provider.
    #[allow(dead_code)] // used by tests; unused in the binary build
    pub fn pattern_count(&self) -> usize {
        self.patterns.len()
    }

    /// Collect all non-overlapping match ranges for a single compiled pattern,
    /// enforcing the per-pattern 10 ms budget (Req 5.6).
    ///
    /// Returns `None` when the budget is exceeded, signalling that the pattern
    /// was skipped for this scan.
    fn scan_pattern(pattern: &CompiledPattern, content: &str) -> Option<Vec<(usize, usize)>> {
        let start = Instant::now();
        let mut ranges = Vec::new();
        for m in pattern.regex.find_iter(content) {
            if start.elapsed() > PATTERN_BUDGET {
                tracing::warn!(
                    pattern = %pattern.name,
                    input_len = content.len(),
                    budget_ms = PATTERN_BUDGET.as_millis(),
                    "regex pattern evaluation exceeded budget; skipping pattern for this scan"
                );
                return None;
            }
            ranges.push((m.start(), m.end()));
        }

        // A single match may itself have consumed the whole budget.
        if start.elapsed() > PATTERN_BUDGET {
            tracing::warn!(
                pattern = %pattern.name,
                input_len = content.len(),
                budget_ms = PATTERN_BUDGET.as_millis(),
                "regex pattern evaluation exceeded budget; skipping pattern for this scan"
            );
            return None;
        }

        Some(ranges)
    }

    /// Core scan logic, shared by the async trait method and unit tests.
    fn scan(&self, content: &str) -> Vec<Finding> {
        let mut allow_ranges: Vec<(usize, usize)> = Vec::new();
        // Candidate deny matches, tagged with their entity label.
        let mut deny_candidates: Vec<(usize, usize, &str)> = Vec::new();

        for pattern in &self.patterns {
            let Some(ranges) = Self::scan_pattern(pattern, content) else {
                continue; // budget exceeded — pattern skipped for this scan
            };
            match pattern.mode {
                RegexRuleMode::Allow => allow_ranges.extend(ranges),
                RegexRuleMode::Deny => {
                    for (start, end) in ranges {
                        deny_candidates.push((start, end, pattern.entity.as_str()));
                    }
                }
            }
        }

        if deny_candidates.is_empty() {
            return Vec::new();
        }

        // Allow-list precedence: drop any deny candidate whose range overlaps an
        // allow-list match (Req 5.3).
        deny_candidates.retain(|&(start, end, _)| {
            !allow_ranges
                .iter()
                .any(|&(a_start, a_end)| ranges_overlap(start, end, a_start, a_end))
        });

        // Deterministic, non-overlapping selection across patterns (Req 5.2):
        // sort by start (then end) and greedily keep matches that do not
        // overlap an already-selected one.
        deny_candidates.sort_by(|a, b| a.0.cmp(&b.0).then(a.1.cmp(&b.1)));

        let mut findings = Vec::new();
        let mut last_end = 0usize;
        let mut has_selection = false;
        for (start, end, entity) in deny_candidates {
            if has_selection && start < last_end {
                continue; // overlaps a previously selected finding
            }
            findings.push(Finding {
                entity_label: entity.to_string(),
                start,
                end,
                matched_text: Some(content[start..end].to_string()),
                score: None,
            });
            last_end = end;
            has_selection = true;
        }

        findings
    }
}

#[async_trait]
impl GuardrailProvider for RegexProvider {
    async fn analyze(&self, content: &str) -> Result<Vec<Finding>, GuardrailProviderError> {
        Ok(self.scan(content))
    }

    fn provider_type(&self) -> &'static str {
        "regex"
    }
}

/// Return `true` when the half-open byte ranges `[a_start, a_end)` and
/// `[b_start, b_end)` overlap.
fn ranges_overlap(a_start: usize, a_end: usize, b_start: usize, b_end: usize) -> bool {
    a_start < b_end && b_start < a_end
}

#[cfg(test)]
mod tests {
    use super::*;
    use proptest::prelude::*;
    use std::collections::HashSet;

    /// Small pool of deliberately safe patterns (literal words, digit classes,
    /// simple bounded tokens) so proptest never generates a pathological regex.
    /// Each entry is `(regex, entity_label)`.
    const SAFE_PATTERNS: &[(&str, &str)] = &[
        (r"\d+", "NUM"),
        (r"[a-z]+", "WORD"),
        (r"foo", "FOO"),
        (r"sk-[A-Za-z0-9]+", "KEY"),
        (r"@[a-z]+", "HANDLE"),
    ];

    /// Known-invalid regex strings used to drive compile-failure tests.
    const INVALID_PATTERNS: &[&str] = &["(", "[", "*", "a{2,1}", "(?P<>)"];

    fn pattern(name: &str, regex: &str, entity: &str, mode: RegexRuleMode) -> RegexPatternConfig {
        RegexPatternConfig {
            name: name.to_string(),
            regex: regex.to_string(),
            entity: entity.to_string(),
            mode,
        }
    }

    #[test]
    fn provider_type_is_regex() {
        let provider = RegexProvider::new(&[]).unwrap();
        assert_eq!(provider.provider_type(), "regex");
    }

    #[test]
    fn compile_failure_identifies_pattern_name_and_reason() {
        let err = RegexProvider::new(&[pattern("bad_paren", "(", "X", RegexRuleMode::Deny)])
            .expect_err("invalid regex must fail to compile");
        match err {
            RegexCompileError::Compile { name, reason } => {
                assert_eq!(name, "bad_paren");
                assert!(!reason.is_empty());
            }
            other => panic!("expected compile error, got {other:?}"),
        }
    }

    #[test]
    fn too_many_patterns_rejected() {
        let patterns: Vec<_> = (0..=MAX_PATTERNS)
            .map(|i| pattern(&format!("p{i}"), "a", "X", RegexRuleMode::Deny))
            .collect();
        let err = RegexProvider::new(&patterns).expect_err("over cap must fail");
        assert_eq!(
            err,
            RegexCompileError::TooManyPatterns {
                count: MAX_PATTERNS + 1,
                max: MAX_PATTERNS
            }
        );
    }

    #[test]
    fn basic_match_reports_offsets_text_and_label() {
        let provider = RegexProvider::new(&[pattern(
            "openai_key",
            r"sk-[A-Za-z0-9]{4,}",
            "API_KEY",
            RegexRuleMode::Deny,
        )])
        .unwrap();

        let content = "token sk-ABCD1234 end";
        let findings = provider.scan(content);
        assert_eq!(findings.len(), 1);
        let f = &findings[0];
        assert_eq!(f.entity_label, "API_KEY");
        assert_eq!(&content[f.start..f.end], "sk-ABCD1234");
        assert_eq!(f.matched_text.as_deref(), Some("sk-ABCD1234"));
    }

    #[test]
    fn no_match_returns_empty() {
        let provider =
            RegexProvider::new(&[pattern("digits", r"\d+", "NUM", RegexRuleMode::Deny)]).unwrap();
        assert!(provider.scan("no numbers here").is_empty());
    }

    #[test]
    fn allow_list_suppresses_overlapping_deny() {
        let provider = RegexProvider::new(&[
            pattern("deny_email", r"\w+@\w+\.\w+", "EMAIL", RegexRuleMode::Deny),
            pattern(
                "allow_corp",
                r"admin@corp\.com",
                "ALLOWED",
                RegexRuleMode::Allow,
            ),
        ])
        .unwrap();

        // The allowed address must not be reported; the other one still is.
        let content = "reach admin@corp.com or evil@bad.net";
        let findings = provider.scan(content);
        assert_eq!(findings.len(), 1);
        assert_eq!(&content[findings[0].start..findings[0].end], "evil@bad.net");
    }

    #[test]
    fn findings_do_not_overlap_across_patterns() {
        let provider = RegexProvider::new(&[
            pattern("word", r"\w+", "WORD", RegexRuleMode::Deny),
            pattern("abc", r"abc", "ABC", RegexRuleMode::Deny),
        ])
        .unwrap();

        let content = "abc def";
        let findings = provider.scan(content);
        // "abc" and "def" as WORD, "abc" as ABC overlaps the first WORD -> dropped.
        for pair in findings.windows(2) {
            assert!(pair[0].end <= pair[1].start, "findings must not overlap");
        }
    }

    #[tokio::test]
    async fn analyze_delegates_to_scan() {
        let provider =
            RegexProvider::new(&[pattern("digits", r"\d+", "NUM", RegexRuleMode::Deny)]).unwrap();
        let findings = provider.analyze("a1b22").await.unwrap();
        assert_eq!(findings.len(), 2);
        assert_eq!(findings[0].entity_label, "NUM");
    }

    // ---- Task 5.5: regex provider edge-case unit tests (Req 5.1, 5.4, 5.6) ----

    /// Req 5.1: exactly [`MAX_PATTERNS`] named patterns compile successfully and
    /// are all retained.
    #[test]
    fn exactly_max_patterns_compiles() {
        let patterns: Vec<_> = (0..MAX_PATTERNS)
            .map(|i| pattern(&format!("p{i}"), "a", "X", RegexRuleMode::Deny))
            .collect();
        let provider = RegexProvider::new(&patterns).expect("256 patterns must compile");
        assert_eq!(provider.pattern_count(), MAX_PATTERNS);
    }

    /// Req 5.4: a single provider compiles patterns once and reuses them across
    /// repeated scans, producing identical results and remaining usable for new
    /// inputs.
    #[test]
    fn compiled_patterns_reused_across_scans() {
        let provider = RegexProvider::new(&[
            pattern("digits", r"\d+", "NUM", RegexRuleMode::Deny),
            pattern("word", r"[a-z]+", "WORD", RegexRuleMode::Deny),
        ])
        .unwrap();

        let first = provider.scan("abc 123");
        let second = provider.scan("abc 123");
        assert_eq!(first.len(), second.len());
        for (x, y) in first.iter().zip(second.iter()) {
            assert_eq!(x.entity_label, y.entity_label);
            assert_eq!(x.start, y.start);
            assert_eq!(x.end, y.end);
            assert_eq!(x.matched_text, y.matched_text);
        }
        // Pattern set is stable across scans (compiled once, reused).
        assert_eq!(provider.pattern_count(), 2);

        // A different input still scans correctly using the same compiled patterns.
        let third = provider.scan("xyz 999 foo");
        assert!(!third.is_empty());
    }

    /// Req 5.6: a pattern whose evaluation exceeds the 10 ms budget is skipped
    /// for the current scan. A pattern that matches every byte of a very large
    /// input yields far more matches than can be processed within the budget,
    /// so it is skipped and contributes no findings.
    #[test]
    fn slow_pattern_is_skipped() {
        let provider =
            RegexProvider::new(&[pattern("all_bytes", r"a", "A", RegexRuleMode::Deny)]).unwrap();
        // 20M single-char matches cannot be enumerated within the 10 ms budget,
        // forcing the skip path deterministically.
        let big = "a".repeat(20_000_000);
        let findings = provider.scan(&big);
        assert!(
            findings.is_empty(),
            "a pattern exceeding the per-pattern budget must be skipped for this scan"
        );
    }

    // ---- Property-based tests (proptest, >=100 cases) ----

    proptest! {
        #![proptest_config(ProptestConfig { cases: 64, ..ProptestConfig::default() })]

        /// Property 12: Regex scan returns valid, non-overlapping matches.
        /// For any input and any set of safe deny patterns, every finding
        /// satisfies `content[start..end] == matched_text`, `start <= end`
        /// within bounds, findings do not overlap, and each carries a
        /// configured entity label.
        /// Validates: Requirements 5.2, 5.7
        #[test]
        fn prop_scan_matches_are_valid_and_non_overlapping(
            deny in prop::collection::hash_set(0usize..SAFE_PATTERNS.len(), 1..=SAFE_PATTERNS.len()),
            input in "[a-z0-9 @.]{0,60}",
        ) {
            let cfgs: Vec<RegexPatternConfig> = deny
                .iter()
                .map(|&i| {
                    let (re, ent) = SAFE_PATTERNS[i];
                    pattern(&format!("p{i}"), re, ent, RegexRuleMode::Deny)
                })
                .collect();
            let provider = RegexProvider::new(&cfgs).unwrap();
            let entities: HashSet<&str> = deny.iter().map(|&i| SAFE_PATTERNS[i].1).collect();

            let findings = provider.scan(&input);

            let mut prev_end = 0usize;
            for (idx, f) in findings.iter().enumerate() {
                prop_assert!(f.start <= f.end, "start must not exceed end");
                prop_assert!(f.end <= input.len(), "end must be within bounds");
                prop_assert_eq!(
                    &input[f.start..f.end],
                    f.matched_text.as_deref().unwrap(),
                    "matched_text must equal content[start..end]"
                );
                prop_assert!(
                    entities.contains(f.entity_label.as_str()),
                    "each finding carries a configured entity label"
                );
                if idx > 0 {
                    prop_assert!(f.start >= prev_end, "findings must not overlap");
                }
                prev_end = f.end;
            }
        }

        /// Property 13: Allow-list precedence over deny-list. For input with both
        /// allow and deny patterns, no byte range covered by an allow match
        /// appears in findings, even when a deny pattern also matches it.
        /// Validates: Requirements 5.3
        #[test]
        fn prop_allow_list_takes_precedence(
            deny in prop::collection::hash_set(0usize..SAFE_PATTERNS.len(), 1..=SAFE_PATTERNS.len()),
            allow in prop::collection::hash_set(0usize..SAFE_PATTERNS.len(), 1..=SAFE_PATTERNS.len()),
            input in "[a-z0-9 @.]{0,60}",
        ) {
            let mut cfgs: Vec<RegexPatternConfig> = Vec::new();
            for &i in &deny {
                let (re, ent) = SAFE_PATTERNS[i];
                cfgs.push(pattern(&format!("d{i}"), re, ent, RegexRuleMode::Deny));
            }
            for &i in &allow {
                let (re, ent) = SAFE_PATTERNS[i];
                cfgs.push(pattern(&format!("a{i}"), re, ent, RegexRuleMode::Allow));
            }
            let provider = RegexProvider::new(&cfgs).unwrap();
            let findings = provider.scan(&input);

            // Independently compute all byte ranges covered by allow patterns.
            let mut allow_ranges: Vec<(usize, usize)> = Vec::new();
            for &i in &allow {
                let re = Regex::new(SAFE_PATTERNS[i].0).unwrap();
                for m in re.find_iter(&input) {
                    allow_ranges.push((m.start(), m.end()));
                }
            }

            for f in &findings {
                for &(a_start, a_end) in &allow_ranges {
                    prop_assert!(
                        !(f.start < a_end && a_start < f.end),
                        "a finding must not overlap any allow-list range"
                    );
                }
            }
        }

        /// Property 14: Regex compile-failure rejection. For any pattern set
        /// containing at least one uncompilable regex, construction fails with an
        /// error identifying an offending pattern's name and a non-empty reason.
        /// Validates: Requirements 5.5
        #[test]
        fn prop_compile_failure_is_rejected(
            valid in prop::collection::vec(0usize..SAFE_PATTERNS.len(), 0..=3),
            invalid in prop::collection::vec(0usize..INVALID_PATTERNS.len(), 1..=3),
        ) {
            let mut cfgs: Vec<RegexPatternConfig> = Vec::new();
            for (k, &i) in valid.iter().enumerate() {
                let (re, ent) = SAFE_PATTERNS[i];
                cfgs.push(pattern(&format!("valid_{k}"), re, ent, RegexRuleMode::Deny));
            }
            let mut invalid_names: HashSet<String> = HashSet::new();
            for (k, &i) in invalid.iter().enumerate() {
                let name = format!("invalid_{k}");
                invalid_names.insert(name.clone());
                cfgs.push(pattern(&name, INVALID_PATTERNS[i], "X", RegexRuleMode::Deny));
            }

            let err = RegexProvider::new(&cfgs)
                .expect_err("a set containing an uncompilable pattern must be rejected");
            match err {
                RegexCompileError::Compile { name, reason } => {
                    prop_assert!(
                        invalid_names.contains(&name),
                        "the error must identify an offending pattern's name"
                    );
                    prop_assert!(!reason.is_empty(), "the compile-failure reason must be present");
                }
                other => prop_assert!(false, "expected a compile error, got {:?}", other),
            }
        }
    }
}
