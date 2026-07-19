//! PII placeholder generation, masking, and transparent re-injection.
//!
//! This module implements the pure PII round-trip logic for pre-call redaction
//! and post-call restoration:
//!
//! - [`GuardrailContext`] owns the request-scoped Re_Injection_Map
//!   (placeholder → original), a per-entity-type sequence counter, and a total
//!   counter capped at [`MAX_REINJECTION_ENTRIES`] entries per request
//!   (Req 4.3, 4.6). It lives only for the duration of the request and is
//!   dropped when the handler returns (Req 2.6, 4.6).
//! - [`GuardrailContext::redact`] replaces each detected span with a
//!   deterministic placeholder and records the mapping (Req 2.1, 2.6).
//! - [`mask`] performs byte-preserving `*` replacement without touching the map
//!   (Req 2.3).
//! - [`GuardrailContext::system_instruction`] yields a preserve-placeholders
//!   instruction only when the map is non-empty (Req 4.4).
//! - [`GuardrailContext::reinject`] restores original values by exact
//!   placeholder replacement, warning on missing (Req 4.5) or corrupted
//!   (Req 4.7) placeholders.

use std::collections::HashMap;
use std::sync::LazyLock;

use regex::Regex;
use serde_json::Value;

use crate::guardrail::config::InstructionInsertionMode;
use crate::guardrail::provider::Finding;
use crate::models::openai::Message;

/// Maximum number of distinct Re_Injection_Map entries per request (Req 4.3).
pub const MAX_REINJECTION_ENTRIES: usize = 256;

/// System instruction prepended to a redacted request so the LLM preserves
/// placeholder tokens verbatim in its response (Req 4.4).
#[allow(dead_code)] // public constant; used by tests and inject_redaction_notice
pub const PRESERVE_PLACEHOLDERS_INSTRUCTION: &str = "The user's message contains placeholder tokens of the form <<PII_TYPE_N>> that stand in for redacted sensitive values. Preserve every such placeholder token exactly as written, verbatim, in your response. Do not alter, translate, reformat, split, or omit any placeholder token.";

/// Documented default text for the Redaction_Notice_Instruction (Req 4.8).
///
/// Provider-agnostic; contains both the placeholder-preservation clause and the
/// redaction-notice clause. Applied verbatim whenever a pipeline does not
/// configure an override and the Re_Injection_Map is non-empty.
pub const DEFAULT_REDACTION_NOTICE_INSTRUCTION: &str = "\
Notice from the security layer: This request was processed by an authorized upstream \
data-protection layer. Sensitive values (secrets, API keys, credentials, and personally \
identifiable information) have been intentionally redacted and replaced with opaque \
placeholder tokens of the form <<PII_TYPE_N>> (for example <<PII_EMAIL_1>>). This \
redaction is expected and authorized; the placeholders stand in for real values that \
will be restored downstream before the response reaches the end user.\n\n\
Instructions:\n\
1. Treat every placeholder token as valid, well-formed input. Do not treat redacted, \
masked, or placeholder-shaped values as errors, corruption, or a policy problem, and do \
not refuse, warn, or ask for the original values solely because content was redacted.\n\
2. Proceed normally with the requested task, including any tool calls, shell commands, \
and computer-use actions. Use the placeholder tokens exactly where the corresponding \
real value would be used.\n\
3. Reproduce every placeholder token verbatim and unchanged in your response wherever it \
would naturally appear. Do not rename, reformat, translate, split, escape, or omit \
placeholder tokens, so that the downstream layer can restore the original values.";

/// Matches a fully-formed PII placeholder: `<<PII_{ENTITY_TYPE}_{N}>>`.
///
/// `{ENTITY_TYPE}` is one or more uppercase-normalized label characters
/// (letters, digits, underscore) and `{N}` is a decimal sequence number. Used
/// during re-injection to distinguish complete placeholders from corrupted
/// fragments (Req 4.7).
static PLACEHOLDER_RE: LazyLock<Regex> = LazyLock::new(|| {
    Regex::new(r"<<PII_[A-Za-z0-9_]+_[0-9]+>>").expect("placeholder regex is valid")
});

/// Marker that begins every placeholder; a `<<PII_` occurrence that is not part
/// of a complete placeholder match is treated as a corrupted fragment.
const PLACEHOLDER_PREFIX: &str = "<<PII_";

/// Normalize an entity label into the `{ENTITY_TYPE}` portion of a placeholder.
///
/// Uppercases ASCII letters and replaces any character that is not a letter or
/// digit with `_`, guaranteeing the generated token matches [`PLACEHOLDER_RE`].
fn normalize_entity_type(label: &str) -> String {
    let normalized: String = label
        .chars()
        .map(|c| {
            if c.is_ascii_alphanumeric() {
                c.to_ascii_uppercase()
            } else {
                '_'
            }
        })
        .collect();
    if normalized.is_empty() {
        "UNKNOWN".to_string()
    } else {
        normalized
    }
}

/// Byte-preserving mask: replace every byte within each finding span with `*`,
/// leaving all bytes outside the spans unchanged (Req 2.3).
///
/// The output has byte length identical to `content`. Overlapping or adjacent
/// spans are handled naturally since each covered byte is independently
/// replaced. Spans with out-of-bounds or inverted offsets are ignored. Masking
/// never populates a Re_Injection_Map.
pub fn mask(content: &str, findings: &[Finding]) -> String {
    let bytes = content.as_bytes();
    // Track which byte positions fall inside any masked span.
    let mut masked = vec![false; bytes.len()];
    for finding in findings {
        if finding.start >= finding.end || finding.end > bytes.len() {
            continue;
        }
        for flag in masked.iter_mut().take(finding.end).skip(finding.start) {
            *flag = true;
        }
    }

    // Rebuild the output byte-for-byte, substituting `*` for masked positions.
    // Masking a byte with `*` (ASCII) preserves total byte length and position.
    let mut out = Vec::with_capacity(bytes.len());
    for (i, &b) in bytes.iter().enumerate() {
        if masked[i] {
            out.push(b'*');
        } else {
            out.push(b);
        }
    }
    // Safe: `content` is valid UTF-8 and we only ever swapped whole bytes that
    // were flagged; masking replaces bytes indiscriminately within a span, so a
    // multi-byte char inside a span becomes one `*` per byte. Any masked byte is
    // ASCII `*`; unmasked bytes are copied verbatim, preserving the original
    // UTF-8 sequences. The result is therefore valid UTF-8.
    String::from_utf8(out).expect("masking preserves UTF-8 validity")
}

/// Inject the redaction-notice instruction into the request messages when the
/// Re_Injection_Map is non-empty (Req 4.11, 4.13).
///
/// When `ctx` is empty (no redaction occurred), no instruction is injected and
/// the request messages are left unchanged (Req 4.13).
///
/// # Parameters
///
/// * `messages` — mutable reference to the request's messages array.
/// * `ctx` — the request-scoped `GuardrailContext` (checked for non-emptiness).
/// * `override_instruction` — per-pipeline override text; when `Some`, replaces
///   the default instruction in full (Req 4.8, 4.9).
/// * `mode` — insertion mode controlling how the instruction is placed (Req 4.10).
pub fn inject_redaction_notice(
    messages: &mut Vec<Message>,
    ctx: &GuardrailContext,
    override_instruction: Option<&str>,
    mode: InstructionInsertionMode,
) {
    // Req 4.13: only inject when map is non-empty.
    if ctx.is_empty() {
        return;
    }

    // Req 4.8, 4.9: resolve instruction text.
    let instruction_text = override_instruction.unwrap_or(DEFAULT_REDACTION_NOTICE_INSTRUCTION);

    match mode {
        InstructionInsertionMode::Separate => {
            insert_separate(messages, instruction_text);
        }
        InstructionInsertionMode::Merged => {
            // Attempt to merge into existing leading system message.
            if !try_merge_into_leading(messages, instruction_text) {
                // Fallback to separate if no leading system message exists.
                insert_separate(messages, instruction_text);
            }
        }
    }
}

/// Insert a new system message at position 0, before all existing messages.
fn insert_separate(messages: &mut Vec<Message>, instruction_text: &str) {
    messages.insert(
        0,
        Message {
            role: "system".to_string(),
            content: Value::String(instruction_text.to_string()),
            extra: serde_json::Map::new(),
        },
    );
}

/// Try to prepend the instruction text into the existing leading system message.
///
/// Returns `true` if a leading system message was found and merged into, `false`
/// otherwise (caller should fall back to `insert_separate`).
fn try_merge_into_leading(messages: &mut Vec<Message>, instruction_text: &str) -> bool {
    if messages.is_empty() {
        return false;
    }

    let first = &messages[0];
    if first.role != "system" {
        return false;
    }

    // Prepend the instruction into the existing content, separated by "\n\n".
    let existing_text = messages[0].content_as_text();
    let merged = format!("{instruction_text}\n\n{existing_text}");
    messages[0].content = Value::String(merged);
    true
}

/// Request-scoped PII state: the Re_Injection_Map, per-entity-type sequence
/// counters, and a total-entry cap (Req 4.3, 4.6).
///
/// Owned by the request handler and dropped at the end of the request scope so
/// the map is held only in memory for the request-response cycle (Req 2.6, 4.6).
#[derive(Debug, Default)]
pub struct GuardrailContext {
    /// Ordered Re_Injection_Map: placeholder → original sensitive value.
    reinjection_map: Vec<(String, String)>,
    /// Next sequence number `N` per normalized entity type (Req 4.3).
    next_sequence: HashMap<String, u32>,
    /// De-duplication index: (entity_type, original_value) → placeholder, so an
    /// identical value of the same entity type reuses its placeholder (Req 4.3).
    dedup: HashMap<(String, String), String>,
}

/// Outcome of [`GuardrailContext::placeholder_for`].
#[derive(Debug, Clone, PartialEq)]
pub enum PlaceholderResult {
    /// A placeholder with a corresponding re-injection entry in the map.
    Mapped(String),
    /// A placeholder issued after the 256-entry cap was reached (Req 4.12):
    /// the value is still redacted but no re-injection entry exists for it.
    Overflow(String),
}

impl PlaceholderResult {
    /// Extract the placeholder string regardless of variant.
    #[allow(dead_code)] // public API; used by tests and future callers
    pub fn placeholder(&self) -> &str {
        match self {
            PlaceholderResult::Mapped(p) | PlaceholderResult::Overflow(p) => p,
        }
    }

    /// Returns `true` if a re-injection entry was created.
    #[allow(dead_code)] // public API; used by tests
    pub fn is_mapped(&self) -> bool {
        matches!(self, PlaceholderResult::Mapped(_))
    }

    /// Returns `true` if the value overflowed the cap (no re-injection entry).
    #[allow(dead_code)] // public API; used by tests
    pub fn is_overflow(&self) -> bool {
        matches!(self, PlaceholderResult::Overflow(_))
    }

    /// Unwrap the placeholder string, panicking if `Overflow` (for tests that
    /// expect a mapped entry).
    #[cfg(test)]
    fn unwrap(self) -> String {
        match self {
            PlaceholderResult::Mapped(p) => p,
            PlaceholderResult::Overflow(_) => panic!("expected Mapped, got Overflow"),
        }
    }
}

impl GuardrailContext {
    /// Create an empty context.
    pub fn new() -> Self {
        Self::default()
    }

    /// Number of Re_Injection_Map entries recorded so far.
    #[allow(dead_code)] // used by tests; unused in the binary build
    pub fn len(&self) -> usize {
        self.reinjection_map.len()
    }

    /// Return `true` if the Re_Injection_Map is empty.
    pub fn is_empty(&self) -> bool {
        self.reinjection_map.is_empty()
    }

    /// Borrow the Re_Injection_Map as `(placeholder, original)` pairs.
    #[allow(dead_code)] // public API / test-only; unused in the binary build
    pub fn reinjection_map(&self) -> &[(String, String)] {
        &self.reinjection_map
    }

    /// Result from [`GuardrailContext::placeholder_for`] indicating the
    /// placeholder to use and whether a re-injection entry was created.
    pub fn placeholder_for(&mut self, entity_label: &str, value: &str) -> PlaceholderResult {
        let entity_type = normalize_entity_type(entity_label);
        let key = (entity_type.clone(), value.to_string());

        // Reuse an existing placeholder for an identical value (dedup).
        if let Some(existing) = self.dedup.get(&key) {
            return PlaceholderResult::Mapped(existing.clone());
        }

        // Allocate the next sequence number for this entity type (starts at 1).
        let seq = self.next_sequence.entry(entity_type.clone()).or_insert(0);
        *seq += 1;
        let placeholder = format!("<<PII_{entity_type}_{seq}>>");

        // Record dedup entry so repeated occurrences of the same value reuse
        // this placeholder regardless of whether re-injection is possible.
        self.dedup.insert(key, placeholder.clone());

        // Enforce the per-request entry cap (Req 4.12): the first 256 distinct
        // values get re-injection entries; excess values are still redacted but
        // no re-injection entry is created for them.
        if self.reinjection_map.len() >= MAX_REINJECTION_ENTRIES {
            return PlaceholderResult::Overflow(placeholder);
        }

        self.reinjection_map
            .push((placeholder.clone(), value.to_string()));
        PlaceholderResult::Mapped(placeholder)
    }

    /// Redact detected spans by replacing each with its placeholder and
    /// recording `placeholder → original` in the Re_Injection_Map (Req 2.1, 2.6).
    ///
    /// Findings are applied right-to-left (by descending start offset) so byte
    /// offsets of not-yet-processed spans remain valid after each replacement.
    /// Overlapping spans are resolved by skipping any finding that intersects a
    /// span already redacted in this call. When the entry cap is hit, excess
    /// values are still redacted with a placeholder but no re-injection entry is
    /// created (Req 4.12); a WARN log records the excluded count.
    pub fn redact(&mut self, content: &str, findings: &[Finding]) -> String {
        // Filter to in-bounds, non-empty spans and sort by descending start so
        // replacements do not invalidate the offsets of earlier spans.
        let mut spans: Vec<&Finding> = findings
            .iter()
            .filter(|f| f.start < f.end && f.end <= content.len())
            .filter(|f| content.is_char_boundary(f.start) && content.is_char_boundary(f.end))
            .collect();
        spans.sort_by(|a, b| b.start.cmp(&a.start));

        let mut out = content.to_string();
        // Track the smallest start already consumed to drop overlapping spans.
        let mut last_consumed_start = usize::MAX;
        let mut overflow_count: usize = 0;
        for finding in spans {
            // Skip a span that overlaps one already redacted to its right.
            if finding.end > last_consumed_start {
                continue;
            }
            let value = &content[finding.start..finding.end];
            let result = self.placeholder_for(&finding.entity_label, value);
            match result {
                PlaceholderResult::Mapped(placeholder) => {
                    out.replace_range(finding.start..finding.end, &placeholder);
                    last_consumed_start = finding.start;
                }
                PlaceholderResult::Overflow(placeholder) => {
                    // Still redact (Req 4.12), but no re-injection entry exists.
                    out.replace_range(finding.start..finding.end, &placeholder);
                    last_consumed_start = finding.start;
                    overflow_count += 1;
                }
            }
        }

        if overflow_count > 0 {
            tracing::warn!(
                excluded_count = overflow_count,
                cap = MAX_REINJECTION_ENTRIES,
                "PII redaction: {overflow_count} detected value(s) exceeded the {MAX_REINJECTION_ENTRIES}-entry re-injection cap and were redacted without re-injection entries"
            );
        }

        out
    }

    /// Return the preserve-placeholders system instruction, but only when the
    /// Re_Injection_Map holds at least one entry (Req 4.4). Returns `None` when
    /// no redaction occurred, so callers prepend nothing.
    #[allow(dead_code)] // public API; used by tests and inject_redaction_notice
    pub fn system_instruction(&self) -> Option<&'static str> {
        if self.is_empty() {
            None
        } else {
            Some(PRESERVE_PLACEHOLDERS_INSTRUCTION)
        }
    }

    /// Restore original values in an LLM response by exact placeholder
    /// replacement (Req 4.2).
    ///
    /// Every occurrence of each mapped placeholder is replaced with its original
    /// value, including repeated occurrences. A placeholder present in the map
    /// but absent from `content` is logged at WARN and skipped (Req 4.5). After
    /// substitution, any residual `<<PII_` fragment that is not a complete,
    /// mapped placeholder is left unchanged and logged at WARN (Req 4.7).
    pub fn reinject(&self, content: &str) -> String {
        let mut out = content.to_string();
        for (placeholder, original) in &self.reinjection_map {
            if out.contains(placeholder.as_str()) {
                out = out.replace(placeholder.as_str(), original);
            } else {
                // Expected placeholder from the map never appeared (Req 4.5).
                tracing::warn!(
                    placeholder = %placeholder,
                    "PII re-injection: expected placeholder missing from response; skipping substitution"
                );
            }
        }

        self.warn_corrupted_fragments(&out);
        out
    }

    /// Scan for residual `<<PII_` fragments that are not complete placeholders
    /// and log each at WARN, leaving the response text unchanged (Req 4.7).
    fn warn_corrupted_fragments(&self, content: &str) {
        // Collect byte ranges of complete placeholders so a `<<PII_` prefix that
        // starts a complete placeholder is not misreported as corrupted.
        let complete: Vec<(usize, usize)> = PLACEHOLDER_RE
            .find_iter(content)
            .map(|m| (m.start(), m.end()))
            .collect();

        let mut search_from = 0usize;
        while let Some(rel) = content[search_from..].find(PLACEHOLDER_PREFIX) {
            let at = search_from + rel;
            let is_complete = complete.iter().any(|&(s, _)| s == at);
            if !is_complete {
                // Show a bounded fragment starting at the corrupted marker.
                let end = (at + 32).min(content.len());
                let fragment_end = content
                    .char_indices()
                    .map(|(i, _)| i)
                    .chain(std::iter::once(content.len()))
                    .find(|&i| i >= end)
                    .unwrap_or(content.len());
                tracing::warn!(
                    fragment = %&content[at..fragment_end],
                    "PII re-injection: corrupted or partial placeholder fragment left unchanged"
                );
            }
            search_from = at + PLACEHOLDER_PREFIX.len();
            if search_from >= content.len() {
                break;
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn finding(label: &str, start: usize, end: usize) -> Finding {
        Finding {
            entity_label: label.to_string(),
            start,
            end,
            matched_text: None,
            score: None,
        }
    }

    #[test]
    fn placeholder_format_and_sequence_increment() {
        let mut ctx = GuardrailContext::new();
        let p1 = ctx.placeholder_for("EMAIL", "a@x.com").unwrap();
        let p2 = ctx.placeholder_for("EMAIL", "b@x.com").unwrap();
        let p3 = ctx.placeholder_for("SSN", "111-22-3333").unwrap();

        assert_eq!(p1, "<<PII_EMAIL_1>>");
        assert_eq!(p2, "<<PII_EMAIL_2>>");
        // Sequence is per entity type, so SSN restarts at 1.
        assert_eq!(p3, "<<PII_SSN_1>>");
        assert_eq!(ctx.len(), 3);
    }

    #[test]
    fn identical_values_reuse_placeholder() {
        let mut ctx = GuardrailContext::new();
        let first = ctx.placeholder_for("EMAIL", "a@x.com").unwrap();
        let again = ctx.placeholder_for("EMAIL", "a@x.com").unwrap();
        assert_eq!(first, again);
        assert_eq!(ctx.len(), 1);
    }

    #[test]
    fn entity_label_is_normalized() {
        let mut ctx = GuardrailContext::new();
        let p = ctx.placeholder_for("email address", "a@x.com").unwrap();
        assert_eq!(p, "<<PII_EMAIL_ADDRESS_1>>");
        assert!(PLACEHOLDER_RE.is_match(&p));
    }

    #[test]
    fn cap_at_256_entries() {
        let mut ctx = GuardrailContext::new();
        for i in 0..MAX_REINJECTION_ENTRIES {
            assert!(ctx.placeholder_for("T", &format!("v{i}")).is_mapped());
        }
        assert_eq!(ctx.len(), MAX_REINJECTION_ENTRIES);
        // The 257th distinct value is still redacted but gets no re-injection entry.
        assert!(ctx.placeholder_for("T", "overflow").is_overflow());
        // A duplicate of an existing value still resolves as mapped after the cap.
        assert!(ctx.placeholder_for("T", "v0").is_mapped());
        assert_eq!(ctx.len(), MAX_REINJECTION_ENTRIES);
    }

    #[test]
    fn redact_replaces_spans_and_records_map() {
        let mut ctx = GuardrailContext::new();
        let content = "email a@x.com and ssn 111-22-3333 here";
        let findings = vec![
            finding("EMAIL", 6, 13), // "a@x.com"
            finding("SSN", 22, 33),  // "111-22-3333"
        ];
        let redacted = ctx.redact(content, &findings);
        assert_eq!(redacted, "email <<PII_EMAIL_1>> and ssn <<PII_SSN_1>> here");
        assert_eq!(ctx.len(), 2);
    }

    #[test]
    fn redact_dedup_same_value_across_spans() {
        let mut ctx = GuardrailContext::new();
        let content = "a@x.com and a@x.com";
        let findings = vec![finding("EMAIL", 0, 7), finding("EMAIL", 12, 19)];
        let redacted = ctx.redact(content, &findings);
        assert_eq!(redacted, "<<PII_EMAIL_1>> and <<PII_EMAIL_1>>");
        assert_eq!(ctx.len(), 1);
    }

    #[test]
    fn mask_preserves_length_and_position() {
        let content = "call 555-1234 now";
        let findings = vec![finding("PHONE", 5, 13)]; // "555-1234"
        let masked = mask(content, &findings);
        assert_eq!(masked, "call ******** now");
        assert_eq!(masked.len(), content.len());
    }

    #[test]
    fn mask_does_not_populate_map() {
        // `mask` is a free function with no context; verify redaction map stays
        // untouched when only masking is used.
        let ctx = GuardrailContext::new();
        let _ = mask("secret", &[finding("X", 0, 6)]);
        assert!(ctx.is_empty());
    }

    #[test]
    fn system_instruction_tracks_map_non_emptiness() {
        let mut ctx = GuardrailContext::new();
        assert!(ctx.system_instruction().is_none());
        ctx.placeholder_for("EMAIL", "a@x.com");
        assert_eq!(
            ctx.system_instruction(),
            Some(PRESERVE_PLACEHOLDERS_INSTRUCTION)
        );
    }

    #[test]
    fn reinject_restores_all_occurrences() {
        let mut ctx = GuardrailContext::new();
        let content = "user a@x.com wrote to a@x.com";
        let findings = vec![finding("EMAIL", 5, 12), finding("EMAIL", 22, 29)];
        let redacted = ctx.redact(content, &findings);
        assert_eq!(redacted, "user <<PII_EMAIL_1>> wrote to <<PII_EMAIL_1>>");

        // The LLM echoes the placeholder twice; both are restored.
        let response = format!("Reply to {} and {}", "<<PII_EMAIL_1>>", "<<PII_EMAIL_1>>");
        let restored = ctx.reinject(&response);
        assert_eq!(restored, "Reply to a@x.com and a@x.com");
    }

    #[test]
    fn reinject_round_trip_identity() {
        let mut ctx = GuardrailContext::new();
        let content = "ssn 111-22-3333 email a@x.com";
        let findings = vec![finding("SSN", 4, 15), finding("EMAIL", 22, 29)];
        let redacted = ctx.redact(content, &findings);
        // Re-injecting the redacted content restores the original exactly.
        assert_eq!(ctx.reinject(&redacted), content);
    }

    #[test]
    fn reinject_skips_missing_placeholder() {
        let mut ctx = GuardrailContext::new();
        ctx.placeholder_for("EMAIL", "a@x.com");
        // Response contains none of the mapped placeholders.
        let restored = ctx.reinject("no placeholders here");
        assert_eq!(restored, "no placeholders here");
    }

    #[test]
    fn reinject_leaves_corrupted_fragment_unchanged() {
        let mut ctx = GuardrailContext::new();
        ctx.placeholder_for("EMAIL", "a@x.com"); // <<PII_EMAIL_1>>
                                                 // A corrupted/partial fragment must be left as-is.
        let response = "here is <<PII_EMAIL_ and a stray <<PII_EMAIL_1>>";
        let restored = ctx.reinject(response);
        assert_eq!(restored, "here is <<PII_EMAIL_ and a stray a@x.com");
    }
}

#[cfg(test)]
mod property_tests {
    //! Property-based tests for the pure PII round-trip logic.
    //!
    //! Generators build content from a sequence of labelled segments so that
    //! detected spans are always in-bounds, non-overlapping, and aligned to
    //! char boundaries (segment concatenation only ever splits at UTF-8
    //! boundaries). This keeps the generated inputs meaningful while exercising
    //! the masking, redaction, sequencing, and system-instruction logic.

    use super::*;
    use proptest::prelude::*;

    /// A generated content segment.
    ///
    /// * `is_pii` — whether this segment becomes a detected span.
    /// * `label` — entity label (already normalization-safe: uppercase alnum,
    ///   non-empty) so `normalize_entity_type` is the identity, letting the test
    ///   model the entity type as the label directly.
    /// * `text` — segment bytes; restricted to characters excluding `<`/`>` so
    ///   generated content can never contain a literal placeholder token.
    type Segment = (bool, String, String);

    /// Strategy for a single segment. PII values are drawn from a small pool to
    /// force duplicate values (exercising dedup), while text excludes the angle
    /// brackets used by placeholder tokens.
    fn segment_strategy() -> impl Strategy<Value = Segment> {
        (
            any::<bool>(),
            "[A-Z][A-Z0-9]{0,6}",
            "[a-zA-Z0-9@._ -]{1,10}",
        )
    }

    /// Build content and its findings from a segment list. Returns the assembled
    /// content, the findings for PII segments (in left-to-right order), and a
    /// per-byte mask indicating which positions fall inside a PII span.
    fn build(segments: &[Segment]) -> (String, Vec<Finding>, Vec<bool>) {
        let mut content = String::new();
        let mut findings = Vec::new();
        let mut in_span: Vec<bool> = Vec::new();
        for (is_pii, label, text) in segments {
            let start = content.len();
            content.push_str(text);
            let end = content.len();
            let span_len = end - start;
            if *is_pii {
                findings.push(Finding {
                    entity_label: label.clone(),
                    start,
                    end,
                    matched_text: Some(text.clone()),
                    score: None,
                });
                in_span.extend(std::iter::repeat(true).take(span_len));
            } else {
                in_span.extend(std::iter::repeat(false).take(span_len));
            }
        }
        (content, findings, in_span)
    }

    proptest! {
        #![proptest_config(ProptestConfig::with_cases(200))]

        // Feature: guardrail-pipelines, Property 5: Mask preserves length and position
        // **Validates: Requirements 2.3**
        #[test]
        fn prop_mask_preserves_length_and_position(
            segments in prop::collection::vec(segment_strategy(), 0..14)
        ) {
            let (content, findings, in_span) = build(&segments);
            let masked = mask(&content, &findings);

            // Output byte length equals input byte length.
            prop_assert_eq!(masked.len(), content.len());

            // Every byte inside a span becomes `*`; every byte outside is
            // unchanged.
            let masked_bytes = masked.as_bytes();
            let orig_bytes = content.as_bytes();
            for i in 0..orig_bytes.len() {
                if in_span[i] {
                    prop_assert_eq!(masked_bytes[i], b'*');
                } else {
                    prop_assert_eq!(masked_bytes[i], orig_bytes[i]);
                }
            }
        }
    }

    proptest! {
        #![proptest_config(ProptestConfig::with_cases(200))]

        // Feature: guardrail-pipelines, Property 6: Redact-then-reinject round-trip
        // **Validates: Requirements 2.1, 4.2**
        #[test]
        fn prop_redact_then_reinject_round_trip(
            segments in prop::collection::vec(segment_strategy(), 0..14)
        ) {
            let (content, findings, _) = build(&segments);
            let mut ctx = GuardrailContext::new();
            let redacted = ctx.redact(&content, &findings);

            // Re-injecting the redacted content restores the original exactly,
            // including when the same placeholder appears multiple times (a value
            // that repeats across PII segments dedups to one placeholder and is
            // restored at every occurrence).
            let restored = ctx.reinject(&redacted);
            prop_assert_eq!(restored, content);
        }
    }

    proptest! {
        #![proptest_config(ProptestConfig::with_cases(200))]

        // Feature: guardrail-pipelines, Property 7: Placeholder format and sequencing
        // **Validates: Requirements 4.3**
        #[test]
        fn prop_placeholder_format_and_sequencing(
            // Enough pairs, with values from a small pool, to exercise both
            // duplicate reuse and the 256-entry cap.
            pairs in prop::collection::vec(("[A-Z][A-Z0-9]{0,6}", "[a-z0-9]{1,4}"), 0..320)
        ) {
            let mut ctx = GuardrailContext::new();

            // Independent model: (entity_type, value) -> placeholder, and the
            // next sequence number per entity type. Labels are normalization-safe
            // so the entity type equals the label.
            use std::collections::HashMap as Map;
            let mut seen: Map<(String, String), String> = Map::new();
            let mut per_type_seq: Map<String, u32> = Map::new();
            let mut distinct_total: usize = 0usize;
            let mut all_placeholders: Vec<String> = Vec::new();

            for (label, value) in &pairs {
                let key = (label.clone(), value.clone());
                let result = ctx.placeholder_for(label, value);

                if let Some(existing) = seen.get(&key) {
                    // Identical value of the same entity type reuses its token.
                    prop_assert_eq!(result.placeholder(), existing.as_str());
                } else if distinct_total >= MAX_REINJECTION_ENTRIES {
                    // Cap reached: a new distinct value still gets a placeholder
                    // but it is an overflow (no re-injection entry).
                    prop_assert!(result.is_overflow());
                    // The placeholder still follows the format.
                    let seq = per_type_seq.entry(label.clone()).or_insert(0);
                    *seq += 1;
                    let expected = format!("<<PII_{}_{}>>", label, seq);
                    prop_assert_eq!(result.placeholder(), expected.as_str());
                    prop_assert!(PLACEHOLDER_RE.is_match(&expected));
                    seen.insert(key, expected);
                    distinct_total += 1;
                } else {
                    // New distinct value: N starts at 1 and increments by 1 per
                    // distinct value of this entity type.
                    prop_assert!(result.is_mapped());
                    let seq = per_type_seq.entry(label.clone()).or_insert(0);
                    *seq += 1;
                    let expected = format!("<<PII_{}_{}>>", label, seq);
                    prop_assert_eq!(result.placeholder(), expected.as_str());

                    // Format matches the placeholder grammar.
                    prop_assert!(PLACEHOLDER_RE.is_match(&expected));

                    all_placeholders.push(expected.clone());
                    seen.insert(key, expected);
                    distinct_total += 1;
                }
            }

            // Never more than 256 re-injection entries created per request.
            prop_assert!(ctx.len() <= MAX_REINJECTION_ENTRIES);
            // ctx.len() reflects only mapped entries (not overflow).
            let mapped_count = distinct_total.min(MAX_REINJECTION_ENTRIES);
            prop_assert_eq!(ctx.len(), mapped_count);

            // Distinct values map to distinct placeholders.
            let mut sorted = all_placeholders.clone();
            sorted.sort();
            sorted.dedup();
            prop_assert_eq!(sorted.len(), all_placeholders.len());
        }
    }

    proptest! {
        #![proptest_config(ProptestConfig::with_cases(100))]

        // Feature: guardrail-pipelines, Property 28: Re-injection overflow beyond 256 distinct values
        // **Validates: Requirements 4.12**
        #[test]
        fn prop_reinjection_overflow(
            n in 257u32..300u32,
        ) {
            let mut ctx = GuardrailContext::new();
            let n = n as usize;

            // Generate N distinct values (using the index to guarantee uniqueness).
            // Use a prefix format that avoids substring collisions across indices.
            let values: Vec<String> = (0..n).map(|i| format!("value__{i:04}")).collect();

            // --- Part 1: placeholder_for produces Mapped/Overflow correctly ---
            let mut results: Vec<PlaceholderResult> = Vec::with_capacity(n);
            for (i, val) in values.iter().enumerate() {
                let result = ctx.placeholder_for("T", val);

                // All values get a valid placeholder assigned.
                prop_assert!(!result.placeholder().is_empty());
                prop_assert!(PLACEHOLDER_RE.is_match(result.placeholder()));

                // First 256 produce Mapped, rest produce Overflow.
                if i < MAX_REINJECTION_ENTRIES {
                    prop_assert!(
                        result.is_mapped(),
                        "Value {} at index {} should be Mapped but got Overflow", val, i
                    );
                } else {
                    prop_assert!(
                        result.is_overflow(),
                        "Value {} at index {} should be Overflow but got Mapped", val, i
                    );
                }
                results.push(result);
            }

            // The Re_Injection_Map never exceeds 256 entries.
            prop_assert_eq!(ctx.len(), MAX_REINJECTION_ENTRIES);

            // Dedup: a previously-seen value reuses its placeholder.
            let dedup_result = ctx.placeholder_for("T", &values[0]);
            prop_assert!(dedup_result.is_mapped());
            prop_assert_eq!(dedup_result.placeholder(), results[0].placeholder());

            // Dedup for an overflow value also returns its existing placeholder.
            let overflow_idx = MAX_REINJECTION_ENTRIES;
            let dedup_overflow = ctx.placeholder_for("T", &values[overflow_idx]);
            prop_assert_eq!(dedup_overflow.placeholder(), results[overflow_idx].placeholder());

            // Map size still capped after dedup calls.
            prop_assert_eq!(ctx.len(), MAX_REINJECTION_ENTRIES);

            // --- Part 2: redact() still replaces ALL spans (including overflow) ---
            // Build content with N distinct detected values as adjacent segments.
            // NOTE: redact() processes findings right-to-left by start offset, so
            // placeholder sequence numbers are allocated from the last finding
            // first. We verify behavior without assuming allocation order.
            let mut ctx2 = GuardrailContext::new();
            let sep = "|"; // separator between values
            let mut content = String::new();
            let mut findings = Vec::new();
            for (i, val) in values.iter().enumerate() {
                let start = content.len();
                content.push_str(val);
                let end = content.len();
                findings.push(Finding {
                    entity_label: "DATA".to_string(),
                    start,
                    end,
                    matched_text: Some(val.clone()),
                    score: None,
                });
                if i < n - 1 {
                    content.push_str(sep);
                }
            }

            let redacted = ctx2.redact(&content, &findings);

            // Every original value must be absent from the redacted text —
            // overflow values are still redacted with placeholders.
            for val in &values {
                prop_assert!(
                    !redacted.contains(val.as_str()),
                    "Value '{}' should have been redacted from the output", val
                );
            }

            // The redacted output contains ONLY placeholder tokens (and separators).
            // Every PII_DATA placeholder in the output matches the regex.
            for segment in redacted.split(sep) {
                if !segment.is_empty() {
                    prop_assert!(
                        PLACEHOLDER_RE.is_match(segment),
                        "Non-separator segment '{}' in redacted output is not a valid placeholder",
                        segment
                    );
                }
            }

            // The Re_Injection_Map must contain exactly 256 entries (not more).
            prop_assert_eq!(ctx2.len(), MAX_REINJECTION_ENTRIES);

            // --- Part 3: reinject restores only mapped values; overflow stays ---
            // Use the redacted text as the LLM "response" (simulates the LLM
            // echoing all placeholders verbatim). After re-injection, the first
            // 256 mapped entries are restored; overflow placeholders remain.
            let restored = ctx2.reinject(&redacted);

            // The map contains exactly 256 entries, each mapping placeholder→original.
            let map = ctx2.reinjection_map();
            prop_assert_eq!(map.len(), MAX_REINJECTION_ENTRIES);

            // Every mapped original value must appear in the restored output.
            for (_placeholder, original) in map {
                prop_assert!(
                    restored.contains(original.as_str()),
                    "Mapped value '{}' should have been restored by re-injection",
                    original
                );
            }

            // Overflow values (those NOT in the map) must NOT appear in the
            // restored output; their placeholder tokens remain instead.
            let mapped_values: std::collections::HashSet<&str> =
                map.iter().map(|(_, v)| v.as_str()).collect();
            let mut overflow_count = 0usize;
            for val in &values {
                if !mapped_values.contains(val.as_str()) {
                    // This value was NOT mapped — it was overflow-redacted.
                    prop_assert!(
                        !restored.contains(val.as_str()),
                        "Overflow value '{}' should NOT appear in the restored response",
                        val
                    );
                    overflow_count += 1;
                }
            }

            // There must be at least one overflow value (n > 256).
            prop_assert!(overflow_count > 0);
            prop_assert_eq!(overflow_count, n - MAX_REINJECTION_ENTRIES);

            // Verify overflow placeholders remain in the restored output.
            // Count placeholder tokens still present — should equal overflow count.
            let remaining_placeholders: Vec<&str> = PLACEHOLDER_RE
                .find_iter(&restored)
                .map(|m| m.as_str())
                .collect();
            prop_assert_eq!(
                remaining_placeholders.len(),
                overflow_count,
                "Expected {} unreplaced overflow placeholders, found {}",
                overflow_count,
                remaining_placeholders.len()
            );
        }
    }

    proptest! {
        #![proptest_config(ProptestConfig::with_cases(200))]

        // Feature: guardrail-pipelines, Property 8: System instruction presence tracks map non-emptiness
        // **Validates: Requirements 4.4**
        #[test]
        fn prop_system_instruction_tracks_map_non_emptiness(
            segments in prop::collection::vec(segment_strategy(), 0..14)
        ) {
            let (content, findings, _) = build(&segments);
            let mut ctx = GuardrailContext::new();
            let _ = ctx.redact(&content, &findings);

            // After redaction, the preserve-placeholders instruction is present
            // iff the Re_Injection_Map holds at least one entry.
            prop_assert_eq!(ctx.system_instruction().is_some(), !ctx.is_empty());
            if ctx.system_instruction().is_some() {
                prop_assert_eq!(
                    ctx.system_instruction(),
                    Some(PRESERVE_PLACEHOLDERS_INSTRUCTION)
                );
            }
        }
    }

    proptest! {
        #![proptest_config(ProptestConfig::with_cases(200))]

        // Feature: guardrail-pipelines, Property 25: Instruction text source selection
        // **Validates: Requirements 4.8, 4.9**
        #[test]
        fn prop_instruction_text_source_selection(
            override_text in "[a-zA-Z0-9 .,!?:;_-]{1,80}"
        ) {
            // Create a GuardrailContext with at least one re-injection entry so the
            // instruction IS injected.
            let mut ctx = GuardrailContext::new();
            ctx.placeholder_for("EMAIL", "test@example.com");
            prop_assert!(!ctx.is_empty());

            // Case 1: When override is Some(custom_text), the injected instruction
            // uses the custom text (not the default).
            let mut messages_override = vec![Message {
                role: "user".to_string(),
                content: serde_json::Value::String("hello".to_string()),
                extra: serde_json::Map::new(),
            }];
            inject_redaction_notice(
                &mut messages_override,
                &ctx,
                Some(&override_text),
                InstructionInsertionMode::Separate,
            );
            prop_assert_eq!(messages_override.len(), 2);
            prop_assert_eq!(&messages_override[0].role, "system");
            let injected_content = messages_override[0].content_as_text();
            prop_assert!(
                injected_content.contains(&override_text),
                "Expected injected instruction to contain override text '{}', got '{}'",
                override_text,
                injected_content
            );
            prop_assert!(
                !injected_content.contains(DEFAULT_REDACTION_NOTICE_INSTRUCTION),
                "Override instruction should NOT contain the default text"
            );

            // Case 2: When override is None, the injected instruction uses
            // DEFAULT_REDACTION_NOTICE_INSTRUCTION.
            let mut messages_default = vec![Message {
                role: "user".to_string(),
                content: serde_json::Value::String("hello".to_string()),
                extra: serde_json::Map::new(),
            }];
            inject_redaction_notice(
                &mut messages_default,
                &ctx,
                None,
                InstructionInsertionMode::Separate,
            );
            prop_assert_eq!(messages_default.len(), 2);
            let default_content = messages_default[0].content_as_text();
            prop_assert!(
                default_content.contains(DEFAULT_REDACTION_NOTICE_INSTRUCTION),
                "Expected default instruction text, got '{}'",
                default_content
            );
        }
    }

    proptest! {
        #![proptest_config(ProptestConfig::with_cases(200))]

        // Feature: guardrail-pipelines, Property 27: Entity-type normalization in placeholders
        // **Validates: Requirements 4.3**
        #[test]
        fn prop_entity_type_normalization(
            label in "\\PC{0,20}"
        ) {
            let mut ctx = GuardrailContext::new();
            let result = ctx.placeholder_for(&label, "dummy_value");
            let placeholder = result.placeholder().to_string();

            // The placeholder must start with <<PII_ and end with >>.
            prop_assert!(placeholder.starts_with("<<PII_"));
            prop_assert!(placeholder.ends_with(">>"));

            // Strip prefix "<<PII_" (6 bytes) and suffix ">>" (2 bytes) to get
            // the inner part: ENTITY_TYPE_N.
            let inner = &placeholder[6..placeholder.len() - 2];

            // The last segment after the final '_' is the sequence number N.
            // The entity-type is everything before that last '_'.
            let last_underscore = inner.rfind('_').expect("placeholder must contain underscore for sequence number");
            let entity_type_segment = &inner[..last_underscore];
            let seq_segment = &inner[last_underscore + 1..];

            // The sequence segment must be a non-empty decimal number.
            prop_assert!(!seq_segment.is_empty());
            prop_assert!(seq_segment.chars().all(|c| c.is_ascii_digit()));

            // The entity-type segment must match ^[A-Z0-9_]+$ (non-empty).
            let entity_type_re = regex::Regex::new(r"^[A-Z0-9_]+$").unwrap();
            prop_assert!(
                entity_type_re.is_match(entity_type_segment),
                "entity_type_segment {:?} did not match ^[A-Z0-9_]+$ for label {:?}",
                entity_type_segment,
                label
            );

            // Verify the entity-type is derived correctly from the label:
            // - Empty label normalizes to "UNKNOWN"
            // - ASCII alphanumeric chars are uppercased
            // - All other chars (including unicode) become '_'
            let expected = if label.is_empty() {
                "UNKNOWN".to_string()
            } else {
                label.chars().map(|c| {
                    if c.is_ascii_alphanumeric() {
                        c.to_ascii_uppercase()
                    } else {
                        '_'
                    }
                }).collect::<String>()
            };
            prop_assert_eq!(entity_type_segment, expected.as_str());
        }
    }

    proptest! {
        #![proptest_config(ProptestConfig::with_cases(200))]

        // Feature: guardrail-pipelines, Property 24: Redaction-notice instruction composition tracks map non-emptiness
        // **Validates: Requirements 4.11, 4.13**
        #[test]
        fn prop_redaction_notice_composition(
            segments in prop::collection::vec(segment_strategy(), 0..14),
            user_text in "[a-zA-Z0-9 ]{1,30}",
        ) {
            let (content, findings, _) = build(&segments);
            let mut ctx = GuardrailContext::new();
            let _ = ctx.redact(&content, &findings);

            // Build a messages array with a single user message.
            let mut messages = vec![Message {
                role: "user".to_string(),
                content: serde_json::Value::String(user_text.clone()),
                extra: serde_json::Map::new(),
            }];
            let original_len = messages.len();

            inject_redaction_notice(
                &mut messages,
                &ctx,
                None,
                InstructionInsertionMode::Separate,
            );

            if ctx.is_empty() {
                // Req 4.13: when Re_Injection_Map is empty, no instruction is
                // injected and messages remain unchanged.
                prop_assert_eq!(messages.len(), original_len);
                prop_assert_eq!(&messages[0].role, "user");
                prop_assert_eq!(messages[0].content_as_text(), user_text);
            } else {
                // Req 4.11: when Re_Injection_Map is non-empty, an instruction
                // IS injected (messages grow by one system message at position 0).
                prop_assert_eq!(messages.len(), original_len + 1);
                prop_assert_eq!(&messages[0].role, "system");

                let injected_text = messages[0].content_as_text();

                // The injected instruction contains the placeholder-preservation
                // clause (instructs LLM to preserve placeholder tokens verbatim).
                prop_assert!(
                    injected_text.contains("placeholder"),
                    "Injected instruction must contain the placeholder-preservation clause"
                );

                // The injected instruction contains the redaction-notice clause
                // (informs the LLM that redaction is authorized and expected).
                prop_assert!(
                    injected_text.contains("redacted"),
                    "Injected instruction must contain the redaction-notice clause"
                );

                // Original user message is preserved after the injected system message.
                prop_assert_eq!(&messages[1].role, "user");
                prop_assert_eq!(messages[1].content_as_text(), user_text);
            }
        }
    }
}

#[cfg(test)]
mod edge_case_tests {
    //! Task 4.6 — placeholder edge cases and request-scope lifetime.
    //! _Requirements: 2.6, 4.1, 4.5, 4.6, 4.7_

    use super::*;

    // Req 4.5 — every expected placeholder missing from the response is skipped,
    // and unrelated text is returned unmodified.
    #[test]
    fn reinject_skips_multiple_missing_placeholders() {
        let mut ctx = GuardrailContext::new();
        ctx.placeholder_for("EMAIL", "a@x.com"); // <<PII_EMAIL_1>>
        ctx.placeholder_for("SSN", "111-22-3333"); // <<PII_SSN_1>>
        let restored = ctx.reinject("the model returned no placeholders at all");
        assert_eq!(restored, "the model returned no placeholders at all");
    }

    // Req 4.2, 4.5 — a subset of placeholders present: present ones are restored,
    // absent ones are skipped without affecting the rest of the response.
    #[test]
    fn reinject_restores_present_and_skips_absent() {
        let mut ctx = GuardrailContext::new();
        let p_email = ctx.placeholder_for("EMAIL", "a@x.com").unwrap();
        let _p_ssn = ctx.placeholder_for("SSN", "111-22-3333").unwrap();
        let response = format!("contact {p_email} soon");
        let restored = ctx.reinject(&response);
        assert_eq!(restored, "contact a@x.com soon");
    }

    // Req 4.7 — a partial placeholder prefix at end-of-string is left unchanged.
    #[test]
    fn reinject_leaves_trailing_partial_fragment_unchanged() {
        let mut ctx = GuardrailContext::new();
        ctx.placeholder_for("EMAIL", "a@x.com");
        let response = "dangling fragment <<PII_";
        let restored = ctx.reinject(response);
        assert_eq!(restored, "dangling fragment <<PII_");
    }

    // Req 4.7 — a fragment missing the numeric sequence and closing `>>` is not a
    // complete placeholder and must be left unchanged.
    #[test]
    fn reinject_leaves_incomplete_placeholder_without_number_unchanged() {
        let mut ctx = GuardrailContext::new();
        ctx.placeholder_for("EMAIL", "a@x.com");
        let response = "broken <<PII_EMAIL_ token";
        let restored = ctx.reinject(response);
        assert_eq!(restored, "broken <<PII_EMAIL_ token");
    }

    // Req 4.7 — a complete, mapped placeholder adjacent to a corrupted fragment:
    // the complete one is restored, the fragment is untouched.
    #[test]
    fn reinject_mixes_complete_and_corrupted() {
        let mut ctx = GuardrailContext::new();
        let p = ctx.placeholder_for("CARD", "4111111111111111").unwrap();
        let response = format!("good {p} bad <<PII_CARD_ end");
        let restored = ctx.reinject(&response);
        assert_eq!(restored, "good 4111111111111111 bad <<PII_CARD_ end");
    }

    // Req 4.7 — a placeholder with a mismatched sequence number (present in the
    // text but never issued into the map) is treated as unmatched and left as-is.
    #[test]
    fn reinject_leaves_unmapped_complete_placeholder_unchanged() {
        let mut ctx = GuardrailContext::new();
        ctx.placeholder_for("EMAIL", "a@x.com"); // only <<PII_EMAIL_1>> is mapped
        let response = "unexpected <<PII_EMAIL_9>> here";
        let restored = ctx.reinject(response);
        // The unmapped placeholder is not in the map, so it is left unchanged.
        assert_eq!(restored, "unexpected <<PII_EMAIL_9>> here");
    }

    // Req 2.6, 4.1, 4.6 — the Re_Injection_Map is request-scoped: it lives only
    // inside a `GuardrailContext` value and there is no shared/global storage, so
    // dropping the context at request-scope end discards the map, and a freshly
    // constructed context for a subsequent request starts empty.
    #[test]
    fn map_is_request_scoped_and_drops_with_context() {
        // First "request": populate a context, then let it drop at scope end.
        let recorded_len = {
            let mut ctx = GuardrailContext::new();
            ctx.placeholder_for("EMAIL", "a@x.com");
            ctx.placeholder_for("SSN", "111-22-3333");
            assert_eq!(ctx.len(), 2);
            ctx.len()
            // `ctx` is dropped here, discarding its Re_Injection_Map.
        };
        assert_eq!(recorded_len, 2);

        // Second "request": a new context observes none of the prior entries,
        // proving the map is not held in any shared/global location.
        let next = GuardrailContext::new();
        assert!(next.is_empty());
        assert_eq!(next.len(), 0);
        assert!(next.reinjection_map().is_empty());
        assert!(next.system_instruction().is_none());
    }
}

#[cfg(test)]
mod redaction_notice_tests {
    //! Task 15.2 — redaction-notice instruction injection tests.
    //! _Requirements: 4.8, 4.9, 4.10, 4.11, 4.13_

    use super::*;

    fn system_msg(text: &str) -> Message {
        Message {
            role: "system".to_string(),
            content: Value::String(text.to_string()),
            extra: serde_json::Map::new(),
        }
    }

    fn user_msg(text: &str) -> Message {
        Message {
            role: "user".to_string(),
            content: Value::String(text.to_string()),
            extra: serde_json::Map::new(),
        }
    }

    // Req 4.13: when context is empty, no instruction is injected.
    #[test]
    fn no_injection_when_map_empty() {
        let ctx = GuardrailContext::new();
        let mut messages = vec![user_msg("hello")];
        inject_redaction_notice(
            &mut messages,
            &ctx,
            None,
            InstructionInsertionMode::Separate,
        );
        assert_eq!(messages.len(), 1);
        assert_eq!(messages[0].role, "user");
    }

    // Req 4.11: when map is non-empty, instruction is injected in separate mode.
    #[test]
    fn injects_default_instruction_separate_mode() {
        let mut ctx = GuardrailContext::new();
        ctx.placeholder_for("EMAIL", "a@x.com");
        let mut messages = vec![user_msg("hi")];
        inject_redaction_notice(
            &mut messages,
            &ctx,
            None,
            InstructionInsertionMode::Separate,
        );
        assert_eq!(messages.len(), 2);
        assert_eq!(messages[0].role, "system");
        assert_eq!(
            messages[0].content_as_text(),
            DEFAULT_REDACTION_NOTICE_INSTRUCTION
        );
        assert_eq!(messages[1].content_as_text(), "hi");
    }

    // Req 4.8, 4.9: per-pipeline override replaces default in full.
    #[test]
    fn override_instruction_replaces_default() {
        let mut ctx = GuardrailContext::new();
        ctx.placeholder_for("SSN", "111-22-3333");
        let mut messages = vec![user_msg("data")];
        let custom = "Custom redaction notice here.";
        inject_redaction_notice(
            &mut messages,
            &ctx,
            Some(custom),
            InstructionInsertionMode::Separate,
        );
        assert_eq!(messages.len(), 2);
        assert_eq!(messages[0].content_as_text(), custom);
    }

    // Req 4.10: merged mode prepends into existing leading system message.
    #[test]
    fn merged_mode_prepends_into_leading_system_message() {
        let mut ctx = GuardrailContext::new();
        ctx.placeholder_for("EMAIL", "a@x.com");
        let mut messages = vec![system_msg("You are helpful."), user_msg("hi")];
        inject_redaction_notice(&mut messages, &ctx, None, InstructionInsertionMode::Merged);
        // No new message added; content prepended to existing system message.
        assert_eq!(messages.len(), 2);
        assert_eq!(messages[0].role, "system");
        let content = messages[0].content_as_text();
        assert!(content.starts_with(DEFAULT_REDACTION_NOTICE_INSTRUCTION));
        assert!(content.contains("\n\nYou are helpful."));
    }

    // Req 4.10: merged mode falls back to separate when no leading system message.
    #[test]
    fn merged_mode_falls_back_to_separate_when_no_system() {
        let mut ctx = GuardrailContext::new();
        ctx.placeholder_for("EMAIL", "a@x.com");
        let mut messages = vec![user_msg("hi")];
        inject_redaction_notice(&mut messages, &ctx, None, InstructionInsertionMode::Merged);
        // Falls back: inserts a new system message at position 0.
        assert_eq!(messages.len(), 2);
        assert_eq!(messages[0].role, "system");
        assert_eq!(
            messages[0].content_as_text(),
            DEFAULT_REDACTION_NOTICE_INSTRUCTION
        );
    }

    // Req 4.10: merged mode with override text.
    #[test]
    fn merged_mode_with_override() {
        let mut ctx = GuardrailContext::new();
        ctx.placeholder_for("PHONE", "555-1234");
        let mut messages = vec![system_msg("Base prompt."), user_msg("call me")];
        let custom = "Override notice.";
        inject_redaction_notice(
            &mut messages,
            &ctx,
            Some(custom),
            InstructionInsertionMode::Merged,
        );
        assert_eq!(messages.len(), 2);
        let content = messages[0].content_as_text();
        assert!(content.starts_with("Override notice."));
        assert!(content.contains("\n\nBase prompt."));
    }

    // Req 4.13: no injection on empty messages array with empty context.
    #[test]
    fn no_injection_empty_messages_empty_context() {
        let ctx = GuardrailContext::new();
        let mut messages: Vec<Message> = vec![];
        inject_redaction_notice(
            &mut messages,
            &ctx,
            None,
            InstructionInsertionMode::Separate,
        );
        assert!(messages.is_empty());
    }

    // Edge case: merged mode with empty messages array but non-empty context.
    #[test]
    fn merged_mode_empty_messages_non_empty_context_falls_back_to_separate() {
        let mut ctx = GuardrailContext::new();
        ctx.placeholder_for("KEY", "secret123");
        let mut messages: Vec<Message> = vec![];
        inject_redaction_notice(&mut messages, &ctx, None, InstructionInsertionMode::Merged);
        // Falls back to separate: one new system message inserted.
        assert_eq!(messages.len(), 1);
        assert_eq!(messages[0].role, "system");
        assert_eq!(
            messages[0].content_as_text(),
            DEFAULT_REDACTION_NOTICE_INSTRUCTION
        );
    }
}

#[cfg(test)]
mod insertion_mode_property_tests {
    //! Property-based tests for insertion-mode message placement (Req 4.10).

    use super::*;
    use proptest::prelude::*;

    /// Strategy for a random message role.
    fn role_strategy() -> impl Strategy<Value = String> {
        prop_oneof![
            Just("system".to_string()),
            Just("user".to_string()),
            Just("assistant".to_string()),
            Just("tool".to_string()),
        ]
    }

    /// Strategy for a random message with a given role and content.
    fn message_strategy() -> impl Strategy<Value = Message> {
        (role_strategy(), "[a-zA-Z0-9 .,!?]{1,40}").prop_map(|(role, content)| Message {
            role,
            content: serde_json::Value::String(content),
            extra: serde_json::Map::new(),
        })
    }

    /// Strategy for messages that guarantees at least one message exists.
    fn non_empty_messages_strategy() -> impl Strategy<Value = Vec<Message>> {
        prop::collection::vec(message_strategy(), 1..8)
    }

    /// Strategy for a non-empty GuardrailContext (at least one entry so injection triggers).
    fn non_empty_ctx_strategy() -> impl Strategy<Value = GuardrailContext> {
        prop::collection::vec(("[A-Z]{1,5}", "[a-z0-9]{1,6}"), 1..4).prop_map(|pairs| {
            let mut ctx = GuardrailContext::new();
            for (label, value) in pairs {
                ctx.placeholder_for(&label, &value);
            }
            ctx
        })
    }

    proptest! {
        #![proptest_config(ProptestConfig::with_cases(200))]

        // Feature: guardrail-pipelines, Property 26: Insertion-mode placement preserves existing messages
        // **Validates: Requirements 4.10**
        #[test]
        fn prop_insertion_mode_placement(
            original_messages in non_empty_messages_strategy(),
            ctx in non_empty_ctx_strategy(),
        ) {
            let instruction_text = DEFAULT_REDACTION_NOTICE_INSTRUCTION;

            // --- Test Separate mode ---
            {
                let mut messages = original_messages.clone();
                inject_redaction_notice(
                    &mut messages,
                    &ctx,
                    None,
                    InstructionInsertionMode::Separate,
                );

                // A new system message is inserted at position 0.
                prop_assert_eq!(messages.len(), original_messages.len() + 1);
                prop_assert_eq!(&messages[0].role, "system");
                prop_assert_eq!(messages[0].content_as_text(), instruction_text);

                // All original messages are present at positions 1..N in their original order.
                for (i, orig_msg) in original_messages.iter().enumerate() {
                    prop_assert_eq!(&messages[i + 1].role, &orig_msg.role);
                    prop_assert_eq!(messages[i + 1].content_as_text(), orig_msg.content_as_text());
                }
            }

            // --- Test Merged mode ---
            {
                let has_leading_system = original_messages[0].role == "system";

                let mut messages = original_messages.clone();
                inject_redaction_notice(
                    &mut messages,
                    &ctx,
                    None,
                    InstructionInsertionMode::Merged,
                );

                if has_leading_system {
                    // Merged mode WITH a leading system message:
                    // - No new message added; array length unchanged.
                    prop_assert_eq!(messages.len(), original_messages.len());

                    // - The existing leading system message's content now starts
                    //   with the instruction text followed by "\n\n" then original content.
                    let merged_content = messages[0].content_as_text();
                    let original_content = original_messages[0].content_as_text();
                    let expected_prefix = format!("{}\n\n{}", instruction_text, original_content);
                    prop_assert_eq!(merged_content, expected_prefix);

                    // - All other messages remain unchanged.
                    for i in 1..original_messages.len() {
                        prop_assert_eq!(&messages[i].role, &original_messages[i].role);
                        prop_assert_eq!(
                            messages[i].content_as_text(),
                            original_messages[i].content_as_text()
                        );
                    }
                } else {
                    // Merged mode WITHOUT a leading system message: falls back to Separate behavior.
                    prop_assert_eq!(messages.len(), original_messages.len() + 1);
                    prop_assert_eq!(&messages[0].role, "system");
                    prop_assert_eq!(messages[0].content_as_text(), instruction_text);

                    // All original messages present at positions 1..N in original order.
                    for (i, orig_msg) in original_messages.iter().enumerate() {
                        prop_assert_eq!(&messages[i + 1].role, &orig_msg.role);
                        prop_assert_eq!(
                            messages[i + 1].content_as_text(),
                            orig_msg.content_as_text()
                        );
                    }
                }
            }
        }
    }
}
