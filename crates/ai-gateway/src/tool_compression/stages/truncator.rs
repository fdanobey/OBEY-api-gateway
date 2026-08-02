//! Description Truncator stage — progressively reduces tool/parameter descriptions.
//!
//! Operates at four intensity levels:
//! - **Low**: Remove example patterns from descriptions
//! - **Medium**: First-sentence extraction + remove parameter descriptions
//! - **High**: Remove all description fields entirely
//! - **Max**: Remove all descriptions + replace large enum arrays with count annotation

use std::sync::LazyLock;

use regex::Regex;
use serde_json::Value;

use crate::tool_compression::config::{CompressionLevel, ToolCompressionConfig};
use crate::tool_compression::stage::CompressionStage;
use crate::tool_compression::state::ToolCompressionState;
use crate::tool_compression::types::{CompressionContext, ToolDefinition};

// ─── Compiled regex patterns ──────────────────────────────────────────────────

/// Matches inline example patterns like `e.g., ...` up to sentence end or closing paren.
static RE_EG: LazyLock<Regex> = LazyLock::new(|| {
    Regex::new(r"(?i)\s*e\.g\.,?\s*[^.)\n]*[.)]?").unwrap()
});

/// Matches `for example:` followed by content up to period or newline.
static RE_FOR_EXAMPLE: LazyLock<Regex> = LazyLock::new(|| {
    Regex::new(r"(?i)\s*for example:\s*[^.\n]*\.?").unwrap()
});

/// Matches `Example:` followed by content to end of line.
static RE_EXAMPLE_LABEL: LazyLock<Regex> = LazyLock::new(|| {
    Regex::new(r"(?i)\s*Example:\s*[^\n]*").unwrap()
});

/// Matches fenced code blocks (``` ... ```).
static RE_FENCED_CODE: LazyLock<Regex> = LazyLock::new(|| {
    Regex::new(r"(?s)```[^\n]*\n.*?```").unwrap()
});

// ─── DescriptionTruncator ─────────────────────────────────────────────────────

/// Description truncation stage.
///
/// Progressively reduces description verbosity based on the effective
/// compression level. Integrates with `DescriptionCompressorState` to
/// prefer pre-computed compressed descriptions when available.
pub struct DescriptionTruncator {
    /// Optional shared state for pre-computed descriptions.
    /// When `Some`, the truncator will check for pre-computed descriptions
    /// before applying runtime truncation.
    pub state: Option<std::sync::Arc<ToolCompressionState>>,
}

impl DescriptionTruncator {
    /// Create a new truncator without shared state.
    pub fn new() -> Self {
        Self { state: None }
    }

    /// Create a new truncator with shared state for pre-computed descriptions.
    pub fn with_state(state: std::sync::Arc<ToolCompressionState>) -> Self {
        Self { state: Some(state) }
    }
}

impl CompressionStage for DescriptionTruncator {
    fn apply(&self, tools: &mut Vec<ToolDefinition>, ctx: &mut CompressionContext) -> u64 {
        let level = ctx.level;
        let config = ToolCompressionConfig::default(); // config from ctx is not stored; use defaults
        let min_preserve = config.description_truncation.min_preserve_length as usize;
        let mut total_saved: u64 = 0;

        for tool in tools.iter_mut() {
            let before = estimate_tokens(&tool.raw);

            match level {
                CompressionLevel::Low => {
                    // Only remove example patterns from descriptions
                    remove_examples_recursive(&mut tool.raw);
                }
                CompressionLevel::Medium => {
                    // Check pre-computed descriptions first
                    if let Some(ref st) = self.state {
                        if let Some(precomputed) = st.description_compressor.get(&tool.name) {
                            set_tool_description(&mut tool.raw, &precomputed);
                        } else {
                            truncate_to_first_sentence(&mut tool.raw, min_preserve);
                        }
                    } else {
                        truncate_to_first_sentence(&mut tool.raw, min_preserve);
                    }
                    // Remove parameter-level descriptions
                    remove_param_descriptions(&mut tool.raw);
                }
                CompressionLevel::High => {
                    // Remove ALL descriptions (tool-level and parameter-level)
                    remove_all_descriptions(&mut tool.raw);
                }
                CompressionLevel::Max => {
                    // Remove all descriptions + replace large enums
                    remove_all_descriptions(&mut tool.raw);
                    replace_large_enums(&mut tool.raw);
                }
            }

            let after = estimate_tokens(&tool.raw);
            total_saved += before.saturating_sub(after);
        }

        if total_saved > 0 {
            ctx.strategies_applied.push("description_truncator".to_string());
        }
        ctx.tokens_saved += total_saved;
        total_saved
    }

    fn is_enabled(&self, config: &ToolCompressionConfig, level: CompressionLevel) -> bool {
        match level {
            CompressionLevel::Low => config.description_truncation.remove_examples,
            CompressionLevel::Medium | CompressionLevel::High | CompressionLevel::Max => true,
        }
    }
}

// ─── Internal helpers ─────────────────────────────────────────────────────────

/// Estimate token count as character_count / 4.
fn estimate_tokens(value: &Value) -> u64 {
    let s = value.to_string();
    (s.len() as u64) / 4
}

/// Set the top-level tool description (inside `function` object).
fn set_tool_description(raw: &mut Value, desc: &str) {
    if let Some(func) = raw.pointer_mut("/function") {
        if let Some(obj) = func.as_object_mut() {
            obj.insert("description".to_string(), Value::String(desc.to_string()));
        }
    }
}

/// Remove example patterns from all description fields recursively.
fn remove_examples_recursive(value: &mut Value) {
    match value {
        Value::Object(map) => {
            if let Some(Value::String(desc)) = map.get_mut("description") {
                *desc = remove_example_patterns(desc);
                if desc.trim().is_empty() {
                    map.remove("description");
                }
            }
            for (_k, v) in map.iter_mut() {
                remove_examples_recursive(v);
            }
        }
        Value::Array(arr) => {
            for item in arr.iter_mut() {
                remove_examples_recursive(item);
            }
        }
        _ => {}
    }
}

/// Apply regex-based example removal to a description string.
fn remove_example_patterns(desc: &str) -> String {
    let result = RE_FENCED_CODE.replace_all(desc, "");
    let result = RE_EG.replace_all(&result, "");
    let result = RE_FOR_EXAMPLE.replace_all(&result, "");
    let result = RE_EXAMPLE_LABEL.replace_all(&result, "");
    result.trim().to_string()
}

/// Truncate tool-level description to first sentence. Preserves short descriptions.
fn truncate_to_first_sentence(raw: &mut Value, min_preserve_length: usize) {
    if let Some(func) = raw.pointer_mut("/function") {
        if let Some(obj) = func.as_object_mut() {
            if let Some(Value::String(desc)) = obj.get("description") {
                if desc.len() <= min_preserve_length {
                    return; // Preserve short descriptions unchanged
                }
                let first = extract_first_sentence(desc);
                obj.insert("description".to_string(), Value::String(first));
            }
        }
    }
}

/// Extract the first sentence from a description.
///
/// Splits on sentence boundaries: `. ` (period-space), `? `, `! `, or end-of-string.
/// Takes the first sentence including its terminal punctuation.
fn extract_first_sentence(text: &str) -> String {
    // Look for sentence-ending patterns followed by space
    let boundaries = [". ", "? ", "! "];
    let mut earliest_end = text.len();

    for boundary in &boundaries {
        if let Some(pos) = text.find(boundary) {
            let end = pos + boundary.len() - 1; // Include the punctuation, exclude trailing space
            if end < earliest_end {
                earliest_end = end;
            }
        }
    }

    text[..earliest_end].to_string()
}

/// Remove all parameter-level descriptions (inside `function.parameters.properties`).
fn remove_param_descriptions(raw: &mut Value) {
    if let Some(params) = raw.pointer_mut("/function/parameters") {
        remove_descriptions_from_properties(params);
    }
}

/// Recursively remove `description` from property schemas.
fn remove_descriptions_from_properties(schema: &mut Value) {
    if let Some(obj) = schema.as_object_mut() {
        if let Some(props) = obj.get_mut("properties") {
            if let Some(props_obj) = props.as_object_mut() {
                for (_key, prop_schema) in props_obj.iter_mut() {
                    if let Some(prop_obj) = prop_schema.as_object_mut() {
                        prop_obj.remove("description");
                        // Recurse into nested objects
                        if prop_obj.get("properties").is_some() {
                            remove_descriptions_from_properties(prop_schema);
                        }
                    }
                }
            }
        }
    }
}

/// Remove ALL description fields recursively (tool-level and parameter-level).
fn remove_all_descriptions(value: &mut Value) {
    match value {
        Value::Object(map) => {
            map.remove("description");
            for (_k, v) in map.iter_mut() {
                remove_all_descriptions(v);
            }
        }
        Value::Array(arr) => {
            for item in arr.iter_mut() {
                remove_all_descriptions(item);
            }
        }
        _ => {}
    }
}

/// Replace enum arrays with >5 entries with a count annotation description.
fn replace_large_enums(value: &mut Value) {
    match value {
        Value::Object(map) => {
            if let Some(enum_val) = map.get("enum") {
                if let Some(arr) = enum_val.as_array() {
                    if arr.len() > 5 {
                        let count = arr.len();
                        map.remove("enum");
                        map.insert(
                            "description".to_string(),
                            Value::String(format!("One of {} allowed values", count)),
                        );
                        // Still recurse into remaining fields
                        for (_k, v) in map.iter_mut() {
                            replace_large_enums(v);
                        }
                        return;
                    }
                }
            }
            for (_k, v) in map.iter_mut() {
                replace_large_enums(v);
            }
        }
        Value::Array(arr) => {
            for item in arr.iter_mut() {
                replace_large_enums(item);
            }
        }
        _ => {}
    }
}

// ─── Tests ────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;
    use proptest::prelude::*;
    use serde_json::json;

    fn make_tool(name: &str, raw: Value) -> ToolDefinition {
        ToolDefinition {
            raw,
            name: name.to_string(),
            content_hash: 0,
        }
    }

    fn default_ctx(level: CompressionLevel) -> CompressionContext {
        CompressionContext {
            level,
            ..Default::default()
        }
    }

    // ─── Example removal (Low level) ─────────────────────────────────

    #[test]
    fn low_removes_eg_pattern() {
        let desc = "The user ID, e.g., 12345 or abc.";
        let result = remove_example_patterns(desc);
        assert!(!result.contains("e.g."));
        assert!(result.contains("The user ID"));
    }

    #[test]
    fn low_removes_for_example_pattern() {
        let desc = "A status string. for example: active or inactive.";
        let result = remove_example_patterns(desc);
        assert!(!result.contains("for example:"));
    }

    #[test]
    fn low_removes_example_label() {
        let desc = "The name field.\nExample: John Doe";
        let result = remove_example_patterns(desc);
        assert!(!result.contains("Example:"));
        assert!(result.contains("The name field."));
    }

    #[test]
    fn low_removes_fenced_code_blocks() {
        let desc = "A query string.\n```sql\nSELECT * FROM users\n```\nUsed for search.";
        let result = remove_example_patterns(desc);
        assert!(!result.contains("```"));
        assert!(!result.contains("SELECT"));
        assert!(result.contains("A query string."));
        assert!(result.contains("Used for search."));
    }

    // ─── First-sentence extraction ───────────────────────────────────

    #[test]
    fn extracts_first_sentence_period() {
        let result = extract_first_sentence("First sentence. Second sentence.");
        assert_eq!(result, "First sentence.");
    }

    #[test]
    fn extracts_first_sentence_question() {
        let result = extract_first_sentence("Is this a tool? More details here.");
        assert_eq!(result, "Is this a tool?");
    }

    #[test]
    fn extracts_first_sentence_exclamation() {
        let result = extract_first_sentence("Do not use! Unless necessary.");
        assert_eq!(result, "Do not use!");
    }

    #[test]
    fn preserves_single_sentence_no_trailing_space() {
        let result = extract_first_sentence("Just a single sentence");
        assert_eq!(result, "Just a single sentence");
    }

    // ─── Medium level truncation ─────────────────────────────────────

    #[test]
    fn medium_truncates_tool_description() {
        let raw = json!({
            "type": "function",
            "function": {
                "name": "get_user",
                "description": "Retrieves a user by ID. Supports filtering and pagination options.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "id": {
                            "type": "string",
                            "description": "The unique user identifier"
                        }
                    }
                }
            }
        });
        let mut tools = vec![make_tool("get_user", raw)];
        let mut ctx = default_ctx(CompressionLevel::Medium);
        let stage = DescriptionTruncator::new();
        stage.apply(&mut tools, &mut ctx);

        let desc = tools[0].raw["function"]["description"].as_str().unwrap();
        assert_eq!(desc, "Retrieves a user by ID.");
        // Parameter descriptions should be removed
        assert!(tools[0].raw["function"]["parameters"]["properties"]["id"]
            .get("description")
            .is_none());
    }

    #[test]
    fn medium_preserves_short_descriptions() {
        let raw = json!({
            "type": "function",
            "function": {
                "name": "ping",
                "description": "Health check",
                "parameters": { "type": "object", "properties": {} }
            }
        });
        let mut tools = vec![make_tool("ping", raw)];
        let mut ctx = default_ctx(CompressionLevel::Medium);
        let stage = DescriptionTruncator::new();
        stage.apply(&mut tools, &mut ctx);

        let desc = tools[0].raw["function"]["description"].as_str().unwrap();
        assert_eq!(desc, "Health check");
    }

    // ─── High level removal ──────────────────────────────────────────

    #[test]
    fn high_removes_all_descriptions() {
        let raw = json!({
            "type": "function",
            "function": {
                "name": "search",
                "description": "Search for items in the database.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "query": {
                            "type": "string",
                            "description": "The search query"
                        }
                    }
                }
            }
        });
        let mut tools = vec![make_tool("search", raw)];
        let mut ctx = default_ctx(CompressionLevel::High);
        let stage = DescriptionTruncator::new();
        stage.apply(&mut tools, &mut ctx);

        assert!(tools[0].raw["function"].get("description").is_none());
        assert!(tools[0].raw["function"]["parameters"]["properties"]["query"]
            .get("description")
            .is_none());
    }

    // ─── Max level enum replacement ──────────────────────────────────

    #[test]
    fn max_replaces_large_enums() {
        let raw = json!({
            "type": "function",
            "function": {
                "name": "set_status",
                "description": "Set entity status.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "status": {
                            "type": "string",
                            "enum": ["a", "b", "c", "d", "e", "f", "g"]
                        }
                    }
                }
            }
        });
        let mut tools = vec![make_tool("set_status", raw)];
        let mut ctx = default_ctx(CompressionLevel::Max);
        let stage = DescriptionTruncator::new();
        stage.apply(&mut tools, &mut ctx);

        let status = &tools[0].raw["function"]["parameters"]["properties"]["status"];
        assert!(status.get("enum").is_none());
        assert_eq!(
            status["description"].as_str().unwrap(),
            "One of 7 allowed values"
        );
    }

    #[test]
    fn max_preserves_small_enums() {
        let raw = json!({
            "type": "function",
            "function": {
                "name": "set_mode",
                "description": "Set mode.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "mode": {
                            "type": "string",
                            "enum": ["a", "b", "c"]
                        }
                    }
                }
            }
        });
        let mut tools = vec![make_tool("set_mode", raw)];
        let mut ctx = default_ctx(CompressionLevel::Max);
        let stage = DescriptionTruncator::new();
        stage.apply(&mut tools, &mut ctx);

        let mode = &tools[0].raw["function"]["parameters"]["properties"]["mode"];
        assert!(mode.get("enum").is_some());
        assert_eq!(mode["enum"].as_array().unwrap().len(), 3);
    }

    // ─── is_enabled tests ────────────────────────────────────────────

    #[test]
    fn is_enabled_low_depends_on_remove_examples() {
        let stage = DescriptionTruncator::new();
        let mut config = ToolCompressionConfig::default();

        assert!(stage.is_enabled(&config, CompressionLevel::Low));

        config.description_truncation.remove_examples = false;
        assert!(!stage.is_enabled(&config, CompressionLevel::Low));
    }

    #[test]
    fn is_enabled_always_true_for_medium_and_above() {
        let stage = DescriptionTruncator::new();
        let mut config = ToolCompressionConfig::default();
        config.description_truncation.remove_examples = false;

        assert!(stage.is_enabled(&config, CompressionLevel::Medium));
        assert!(stage.is_enabled(&config, CompressionLevel::High));
        assert!(stage.is_enabled(&config, CompressionLevel::Max));
    }

    // ─── Name preservation ───────────────────────────────────────────

    #[test]
    fn never_modifies_tool_name() {
        let raw = json!({
            "type": "function",
            "function": {
                "name": "my_tool_name",
                "description": "A very long description that should be truncated. With extra sentences.",
                "parameters": {
                    "type": "object",
                    "required": ["x"],
                    "properties": {
                        "x": { "type": "string", "description": "param desc" }
                    }
                }
            }
        });

        for level in [
            CompressionLevel::Low,
            CompressionLevel::Medium,
            CompressionLevel::High,
            CompressionLevel::Max,
        ] {
            let mut tools = vec![make_tool("my_tool_name", raw.clone())];
            let mut ctx = default_ctx(level);
            let stage = DescriptionTruncator::new();
            stage.apply(&mut tools, &mut ctx);

            assert_eq!(
                tools[0].raw["function"]["name"].as_str().unwrap(),
                "my_tool_name"
            );
        }
    }

    #[test]
    fn never_modifies_required_array() {
        let raw = json!({
            "type": "function",
            "function": {
                "name": "test",
                "description": "Description that is long enough to truncate. Second sentence here.",
                "parameters": {
                    "type": "object",
                    "required": ["a", "b"],
                    "properties": {
                        "a": { "type": "string", "description": "First param" },
                        "b": { "type": "integer", "description": "Second param" }
                    }
                }
            }
        });

        for level in [
            CompressionLevel::Low,
            CompressionLevel::Medium,
            CompressionLevel::High,
            CompressionLevel::Max,
        ] {
            let mut tools = vec![make_tool("test", raw.clone())];
            let mut ctx = default_ctx(level);
            let stage = DescriptionTruncator::new();
            stage.apply(&mut tools, &mut ctx);

            let required = tools[0].raw["function"]["parameters"]["required"]
                .as_array()
                .unwrap();
            assert_eq!(required.len(), 2);
            assert_eq!(required[0], "a");
            assert_eq!(required[1], "b");
        }
    }

    // ─── Token savings tracking ──────────────────────────────────────

    #[test]
    fn tracks_tokens_saved() {
        let raw = json!({
            "type": "function",
            "function": {
                "name": "verbose_tool",
                "description": "This is a very verbose description. It contains multiple sentences. And even more content that should be removed at higher levels.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "param": {
                            "type": "string",
                            "description": "A parameter with a long description that adds tokens"
                        }
                    }
                }
            }
        });
        let mut tools = vec![make_tool("verbose_tool", raw)];
        let mut ctx = default_ctx(CompressionLevel::High);
        let stage = DescriptionTruncator::new();
        let saved = stage.apply(&mut tools, &mut ctx);

        assert!(saved > 0);
        assert_eq!(ctx.tokens_saved, saved);
        assert!(ctx.strategies_applied.contains(&"description_truncator".to_string()));
    }

    // ─── Property Tests ──────────────────────────────────────────────

    // Feature: tool-definition-compression, Property 5: Description Truncation First-Sentence Extraction
    // **Validates: Requirements 2.2, 2.7**
    //
    // Strategy: Generate multi-sentence strings with various sentence-ending punctuation,
    // verify correct truncation boundary and min-preserve-length behavior.

    /// Strategy to generate a sentence fragment without sentence-ending punctuation followed by space.
    fn sentence_fragment() -> impl Strategy<Value = String> {
        "[A-Za-z ]{3,30}".prop_filter("no sentence boundary inside", |s| {
            !s.contains(". ") && !s.contains("? ") && !s.contains("! ")
        })
    }

    /// Strategy to generate a sentence-ending punctuation character.
    fn sentence_terminator() -> impl Strategy<Value = char> {
        prop_oneof![Just('.'), Just('?'), Just('!')]
    }

    proptest! {
        #![proptest_config(ProptestConfig::with_cases(200))]

        /// Property: extract_first_sentence always returns text ending at the first
        /// sentence boundary (`. `, `? `, `! `), and the result is a prefix of the original.
        #[test]
        fn prop_first_sentence_extraction_boundary(
            first in sentence_fragment(),
            terminator in sentence_terminator(),
            second in sentence_fragment(),
        ) {
            // Construct multi-sentence input: "First<term> Second"
            let input = format!("{}{} {}", first.trim(), terminator, second.trim());
            let result = extract_first_sentence(&input);

            // Result must end with the terminator character
            prop_assert!(
                result.ends_with(terminator),
                "Expected result to end with '{}', got: '{}'", terminator, result
            );

            // Result length must be <= original length
            prop_assert!(result.len() <= input.len(),
                "Result '{}' longer than input '{}'", result, input);

            // Result must be a prefix of the original (up to the boundary)
            prop_assert!(input.starts_with(&result),
                "Result '{}' is not a prefix of input '{}'", result, input);

            // The character after the result in the original must be a space
            // (confirming we stopped at "<punct> " boundary)
            if result.len() < input.len() {
                let next_char = input.as_bytes()[result.len()];
                prop_assert_eq!(next_char, b' ',
                    "Character after result should be space, got '{}'", next_char as char);
            }
        }

        /// Property: extract_first_sentence on a string with no sentence boundary
        /// returns the full string unchanged.
        #[test]
        fn prop_no_boundary_returns_full_string(
            text in "[A-Za-z]{5,50}"
        ) {
            // No ". ", "? ", or "! " in alpha-only string
            let result = extract_first_sentence(&text);
            prop_assert_eq!(&result, &text,
                "Without sentence boundary, should return full string");
        }

        /// Property: descriptions ≤ 20 characters are preserved unchanged at Medium level.
        #[test]
        fn prop_short_description_preserved_at_medium(
            desc in "[A-Za-z ]{1,20}"
        ) {
            let desc_trimmed = desc.trim().to_string();
            // Only test descriptions that are actually ≤ 20 chars after trimming
            prop_assume!(!desc_trimmed.is_empty() && desc_trimmed.len() <= 20);

            let raw = json!({
                "type": "function",
                "function": {
                    "name": "test_tool",
                    "description": desc_trimmed,
                    "parameters": { "type": "object", "properties": {} }
                }
            });
            let mut tools = vec![make_tool("test_tool", raw)];
            let mut ctx = default_ctx(CompressionLevel::Medium);
            let stage = DescriptionTruncator::new();
            stage.apply(&mut tools, &mut ctx);

            let result_desc = tools[0].raw["function"]["description"].as_str().unwrap();
            prop_assert_eq!(result_desc, desc_trimmed.as_str(),
                "Short description should be preserved unchanged at Medium level");
        }

        /// Property: extract_first_sentence result length is always ≤ original length
        /// for any arbitrary input with sentence boundaries mixed in.
        #[test]
        fn prop_first_sentence_never_longer_than_input(
            text in ".{1,200}"
        ) {
            let result = extract_first_sentence(&text);
            prop_assert!(result.len() <= text.len(),
                "Result length {} exceeds input length {}", result.len(), text.len());
        }

        /// Property: when the first boundary is a period, question mark, or exclamation mark,
        /// the result always contains at most one such boundary (the first one).
        #[test]
        fn prop_first_sentence_picks_earliest_boundary(
            first in sentence_fragment(),
            term1 in sentence_terminator(),
            middle in sentence_fragment(),
            term2 in sentence_terminator(),
            last in sentence_fragment(),
        ) {
            let input = format!("{}{} {}{} {}", first.trim(), term1, middle.trim(), term2, last.trim());
            let result = extract_first_sentence(&input);

            // The result should be the first sentence only
            let expected = format!("{}{}", first.trim(), term1);
            prop_assert_eq!(&result, &expected,
                "Should extract only first sentence from multi-sentence input");
        }
    }
}
