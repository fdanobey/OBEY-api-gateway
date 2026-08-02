//! Shared description compression utilities.
//!
//! Extracted from `compression/engines/tool_def.rs` to provide a stable,
//! shared surface for both the existing `ToolDefinitionEngine` and the new
//! `DescriptionCompressor` in the tool compression middleware.

use serde_json::Value;
use std::collections::HashSet;

/// Maximum number of sentences retained in a compressed summary.
pub const MAX_SUMMARY_SENTENCES: usize = 2;

/// Maximum character length for a compressed summary.
pub const MAX_SUMMARY_CHARS: usize = 240;

/// Section markers whose trailing content is removed during truncation.
pub const REMOVABLE_SECTION_MARKERS: [&str; 10] = [
    " caveat:",
    " caveats:",
    " example:",
    " examples:",
    " for example,",
    " for example:",
    " important:",
    " note:",
    " notes:",
    " warning:",
];

/// Sentence prefixes that mark a sentence as removable.
pub const REMOVABLE_SENTENCE_PREFIXES: [&str; 13] = [
    "caveat:",
    "caveats:",
    "e.g.",
    "example:",
    "examples:",
    "for example",
    "important:",
    "keep in mind",
    "note:",
    "notes:",
    "please note",
    "warning:",
    "warnings:",
];

/// Verbose introductory prefixes that are stripped to produce concise sentences.
pub const VERBOSE_PREFIXES: [&str; 8] = [
    "the purpose of this function is to ",
    "the purpose of this tool is to ",
    "this function allows you to ",
    "this function can be used to ",
    "this function is used to ",
    "this tool allows you to ",
    "this tool can be used to ",
    "this tool is used to ",
];

/// Schema fields whose values are treated as literals and never recursed into.
const LITERAL_SCHEMA_FIELDS: [&str; 5] = ["const", "default", "enum", "example", "examples"];

/// Recursively walks a JSON value and compresses all string-valued `description`
/// fields in-place. Returns `true` if any field was modified.
///
/// Fields listed in [`LITERAL_SCHEMA_FIELDS`] are never descended into.
pub fn compress_description_fields(value: &mut Value) -> bool {
    match value {
        Value::Array(values) => values.iter_mut().fold(false, |changed, value| {
            compress_description_fields(value) || changed
        }),
        Value::Object(object) => {
            let mut changed = false;
            for (field, value) in object {
                if field == "description" {
                    if let Value::String(description) = value {
                        let compressed = compress_description(description);
                        if compressed.len() < description.len() {
                            *description = compressed;
                            changed = true;
                        }
                    }
                } else if !LITERAL_SCHEMA_FIELDS.contains(&field.as_str()) {
                    changed = compress_description_fields(value) || changed;
                }
            }
            changed
        }
        _ => false,
    }
}

/// Compresses a single description string using sentence extraction,
/// removable-section truncation, verbose-prefix stripping, and deduplication.
pub fn compress_description(description: &str) -> String {
    let normalized = normalize_whitespace(description);
    if normalized.is_empty() {
        return normalized;
    }

    let without_sections = truncate_removable_sections(&normalized);
    let mut summaries = Vec::new();
    let mut seen = HashSet::new();

    for sentence in split_sentences(without_sections) {
        let sentence = sentence.trim();
        if sentence.is_empty() || is_removable_sentence(sentence) {
            continue;
        }
        let concise = remove_verbose_prefix(sentence);
        let identity = concise.to_lowercase();
        if seen.insert(identity) {
            summaries.push(concise.to_owned());
        }
        if summaries.len() == MAX_SUMMARY_SENTENCES {
            break;
        }
    }

    if summaries.is_empty() {
        return normalized;
    }

    truncate_summary(&summaries.join(" "))
}

/// Collapses all whitespace runs into single spaces and trims the result.
pub fn normalize_whitespace(text: &str) -> String {
    let mut output = String::with_capacity(text.len());
    let mut pending_space = false;
    for character in text.chars() {
        if character.is_whitespace() {
            pending_space = !output.is_empty();
        } else {
            if pending_space {
                output.push(' ');
                pending_space = false;
            }
            output.push(character);
        }
    }
    output
}

/// Truncates text at the earliest removable-section marker.
pub fn truncate_removable_sections(text: &str) -> &str {
    let lowercase = text.to_ascii_lowercase();
    REMOVABLE_SECTION_MARKERS
        .iter()
        .filter_map(|marker| lowercase.find(marker))
        .min()
        .map_or(text, |index| text[..index].trim_end())
}

/// Splits text into sentences at `.` `!` `?` boundaries followed by whitespace
/// or end-of-string.
pub fn split_sentences(text: &str) -> Vec<&str> {
    let mut sentences = Vec::new();
    let mut start = 0;
    let mut characters = text.char_indices().peekable();

    while let Some((index, character)) = characters.next() {
        if matches!(character, '.' | '!' | '?')
            && characters
                .peek()
                .is_none_or(|(_, next)| next.is_whitespace())
        {
            let end = index + character.len_utf8();
            sentences.push(&text[start..end]);
            while let Some((next_index, next)) = characters.peek().copied() {
                if !next.is_whitespace() {
                    start = next_index;
                    break;
                }
                characters.next();
                start = text.len();
            }
        }
    }

    if start < text.len() {
        sentences.push(&text[start..]);
    }
    sentences
}

/// Returns `true` if a sentence begins with a removable prefix.
pub fn is_removable_sentence(sentence: &str) -> bool {
    let lowercase = sentence.to_ascii_lowercase();
    REMOVABLE_SENTENCE_PREFIXES
        .iter()
        .any(|prefix| lowercase.starts_with(prefix))
}

/// Strips verbose introductory prefixes from a sentence.
pub fn remove_verbose_prefix(sentence: &str) -> &str {
    let lowercase = sentence.to_ascii_lowercase();
    VERBOSE_PREFIXES
        .iter()
        .find_map(|prefix| {
            lowercase
                .strip_prefix(prefix)
                .map(|suffix| &sentence[sentence.len() - suffix.len()..])
                .map(str::trim_start)
        })
        .filter(|concise| !concise.is_empty())
        .unwrap_or(sentence)
}

/// Truncates a joined summary to [`MAX_SUMMARY_CHARS`], breaking at word boundaries.
pub fn truncate_summary(summary: &str) -> String {
    if summary.chars().count() <= MAX_SUMMARY_CHARS {
        return summary.to_owned();
    }

    let hard_boundary = summary
        .char_indices()
        .nth(MAX_SUMMARY_CHARS)
        .map_or(summary.len(), |(index, _)| index);
    let boundary = summary[..hard_boundary]
        .char_indices()
        .rev()
        .find_map(|(index, character)| character.is_whitespace().then_some(index))
        .unwrap_or(hard_boundary);
    let mut truncated = summary[..boundary]
        .trim_end_matches([' ', ',', ';', ':'])
        .to_owned();
    if !truncated.ends_with(['.', '!', '?']) {
        truncated.push('.');
    }
    truncated
}
