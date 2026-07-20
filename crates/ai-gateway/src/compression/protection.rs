//! Compression protection-region scanning.

use regex::Regex;
use serde_json::Value;
use std::ops::Range;

/// A half-open UTF-8 byte range containing content that compression must not alter.
pub type ProtectedRange = Range<usize>;

const JSON_VALIDATION_BUDGET_MULTIPLIER: usize = 8;

/// Finds byte ranges that compression engines must preserve verbatim.
///
/// All regular expressions are compiled when the scanner is constructed. The
/// parsers used for nested constructs are bounded or linear over the input and
/// never slice text without first establishing UTF-8 byte boundaries.
#[derive(Debug)]
pub struct ProtectionScanner {
    fenced_code_start: Regex,
    indented_code_start: Regex,
    url: Regex,
    unix_path: Regex,
    windows_path: Regex,
    identifier: Regex,
    definition_head: Regex,
    tool_marker: Regex,
}

impl ProtectionScanner {
    /// Compiles the built-in protection patterns.
    pub fn new() -> Result<Self, regex::Error> {
        Ok(Self {
            fenced_code_start: Regex::new(r"(?m)^ {0,3}(?P<fence>`{3,}|~{3,})[^\r\n]*\r?$")?,
            indented_code_start: Regex::new(r"(?m)^ {4}[^\r\n]*(?:\r?\n|$)")?,
            url: Regex::new(r#"(?i)\b(?:https?|ftp|file)://[^\s<>\"'`]+"#)?,
            unix_path: Regex::new(
                r#"(?m)(?:^|[\s(\"'`=])(?P<path>/(?:[^\s/<>:\"'`]+/)+[^\s/<>:\"'`]*)"#,
            )?,
            windows_path: Regex::new(
                r#"(?m)(?:^|[\s(\"'`=])(?P<path>[A-Za-z]:\\[^\s\r\n<>\"|?*]+(?:\\[^\s\r\n<>\"|?*]+)*)"#,
            )?,
            identifier: Regex::new(
                r"(?x)\b(?:
                    [a-z][a-z0-9]*[A-Z][A-Za-z0-9]*
                    |
                    [A-Z][a-z0-9]+(?:[A-Z][A-Za-z0-9]*)+
                    |
                    [a-z][a-z0-9]*(?:_[a-z0-9]+)+
                    |
                    [A-Z][A-Z0-9]*(?:_[A-Z0-9]+)+
                )\b",
            )?,
            definition_head: Regex::new(
                r"(?xm)(?:^|[^A-Za-z0-9_])(?P<head>
                    (?:(?:pub(?:\([^\r\n)]*\))?\s+)?(?:unsafe\s+)?(?:async\s+)?fn
                    |(?:async\s+)?def
                    |function
                    |call_tool
                    |call_function
                    |invoke_tool
                    |invoke_function
                )\s+[A-Za-z_][A-Za-z0-9_]*(?:\s*<[^()\r\n]*>)?\s*\()",
            )?,
            tool_marker: Regex::new(
                r#"(?ims)(?:<tool_call\b[^>]*>.*?</tool_call\s*>|<tool_use\b[^>]*>.*?</tool_use\s*>|<function_call\b[^>]*>.*?</function_call\s*>|\b"?(?:tool_use|tool_call|function_call|tool_definition|tools|functions)"?\s*[:=]\s*)"#,
            )?,
        })
    }

    /// Returns sorted, merged, non-overlapping protected byte ranges.
    pub fn scan(&self, text: &str) -> Vec<ProtectedRange> {
        let mut ranges = Vec::new();

        self.scan_fenced_code(text, &mut ranges);
        self.scan_indented_code(text, &mut ranges);
        self.scan_regex_matches(text, &self.url, None, true, &mut ranges);
        self.scan_regex_matches(text, &self.unix_path, Some("path"), true, &mut ranges);
        self.scan_regex_matches(text, &self.windows_path, Some("path"), true, &mut ranges);
        self.scan_regex_matches(text, &self.identifier, None, false, &mut ranges);
        self.scan_json(text, &mut ranges);
        self.scan_math(text, &mut ranges);
        self.scan_definitions(text, &mut ranges);
        for captures in self.tool_marker.captures_iter(text) {
            let Some(marker) = captures.get(0) else {
                continue;
            };
            let mut end = marker.end();
            let marker_text = marker.as_str();
            let marker_is_tag = marker_text.trim_start().starts_with('<');
            if !marker_is_tag {
                let mut search = end;
                while text
                    .as_bytes()
                    .get(search)
                    .is_some_and(u8::is_ascii_whitespace)
                {
                    search += 1;
                }
                if matches!(text.as_bytes().get(search), Some(b'{') | Some(b'[')) {
                    if let Some(json_range) = find_balanced_json_at(text, search) {
                        end = json_range.end;
                    }
                }
            }
            ranges.push(marker.start()..end);
        }

        merge_ranges(text, ranges)
    }

    /// Returns the complementary ranges that engines may transform.
    pub fn unprotected_ranges(&self, text: &str) -> Vec<Range<usize>> {
        let protected = self.scan(text);
        complement_ranges(text.len(), &protected)
    }

    /// Applies a transform only to unprotected spans and copies protected bytes
    /// directly from the input.
    pub fn transform_unprotected<F>(&self, text: &str, mut transform: F) -> String
    where
        F: FnMut(&str) -> String,
    {
        let protected = self.scan(text);
        let mut output = String::with_capacity(text.len());
        let mut cursor = 0;

        for protected_range in protected {
            if cursor < protected_range.start {
                output.push_str(&transform(&text[cursor..protected_range.start]));
            }
            output.push_str(&text[protected_range.clone()]);
            cursor = protected_range.end;
        }

        if cursor < text.len() {
            output.push_str(&transform(&text[cursor..]));
        }

        output
    }

    /// Returns true when any byte in `candidate` intersects a protected range.
    pub fn intersects_protected(protected: &[ProtectedRange], candidate: Range<usize>) -> bool {
        if candidate.start >= candidate.end {
            return false;
        }

        protected
            .iter()
            .take_while(|range| range.start < candidate.end)
            .any(|range| range.end > candidate.start)
    }

    /// Returns true when one protected range contains all of `candidate`.
    pub fn is_fully_protected(protected: &[ProtectedRange], candidate: Range<usize>) -> bool {
        candidate.start < candidate.end
            && protected
                .iter()
                .take_while(|range| range.start <= candidate.start)
                .any(|range| range.start <= candidate.start && range.end >= candidate.end)
    }

    fn scan_regex_matches(
        &self,
        text: &str,
        regex: &Regex,
        capture_name: Option<&str>,
        trim_trailing: bool,
        ranges: &mut Vec<ProtectedRange>,
    ) {
        for captures in regex.captures_iter(text) {
            let Some(matched) = capture_name
                .and_then(|name| captures.name(name))
                .or_else(|| captures.get(0))
            else {
                continue;
            };
            let mut end = matched.end();
            if trim_trailing {
                end = trim_token_end(text, matched.start(), end);
            }
            if matched.start() < end {
                ranges.push(matched.start()..end);
            }
        }
    }

    fn scan_fenced_code(&self, text: &str, ranges: &mut Vec<ProtectedRange>) {
        let mut cursor = 0;
        while let Some(captures) = self.fenced_code_start.captures_at(text, cursor) {
            let Some(opening) = captures.get(0) else {
                break;
            };
            let Some(marker) = captures.name("fence") else {
                cursor = opening.end();
                continue;
            };

            let marker_byte = marker.as_str().as_bytes()[0];
            let marker_len = marker.as_str().len();
            let opening_end = line_end_with_newline(text, opening.start());
            let closing_end = find_closing_fence(text, opening_end, marker_byte, marker_len)
                .unwrap_or(text.len());
            ranges.push(opening.start()..closing_end);
            cursor = closing_end.max(opening.end());
        }
    }

    fn scan_indented_code(&self, text: &str, ranges: &mut Vec<ProtectedRange>) {
        let mut cursor = 0;
        while let Some(opening) = self.indented_code_start.find_at(text, cursor) {
            let start = opening.start();
            let mut end = opening.end();
            let mut line_start = end;

            while line_start < text.len() {
                let line_end = line_end_without_newline(text, line_start);
                let line = &text[line_start..line_end];
                if line.starts_with("    ") || line.trim().is_empty() {
                    end = line_end_with_newline(text, line_start);
                    line_start = end;
                } else {
                    break;
                }
            }

            ranges.push(start..end);
            cursor = end.max(opening.end());
        }
    }

    fn scan_json(&self, text: &str, ranges: &mut Vec<ProtectedRange>) {
        let bytes = text.as_bytes();
        let mut stack: Vec<(u8, usize)> = Vec::new();
        let mut candidates = Vec::new();
        let mut in_string = false;
        let mut escaped = false;

        for (index, byte) in bytes.iter().copied().enumerate() {
            if stack.is_empty() {
                if byte == b'{' || byte == b'[' {
                    stack.push((byte, index));
                    in_string = false;
                    escaped = false;
                }
                continue;
            }

            if in_string {
                if escaped {
                    escaped = false;
                } else if byte == b'\\' {
                    escaped = true;
                } else if byte == b'"' {
                    in_string = false;
                }
                continue;
            }

            match byte {
                b'"' => in_string = true,
                b'{' | b'[' => stack.push((byte, index)),
                b'}' | b']' => {
                    let expected = if byte == b'}' { b'{' } else { b'[' };
                    if stack
                        .last()
                        .is_some_and(|(opening, _)| *opening == expected)
                    {
                        if let Some((_, start)) = stack.pop() {
                            candidates.push(start..index + 1);
                        }
                    } else {
                        stack.clear();
                        in_string = false;
                        escaped = false;
                    }
                }
                _ => {}
            }
        }

        candidates.sort_unstable_by(|left, right| {
            left.start
                .cmp(&right.start)
                .then_with(|| right.end.cmp(&left.end))
        });
        candidates.dedup();

        let mut validation_budget = text
            .len()
            .saturating_mul(JSON_VALIDATION_BUDGET_MULTIPLIER)
            .max(1024);
        let mut valid_ranges: Vec<ProtectedRange> = Vec::new();

        for candidate in candidates {
            if valid_ranges
                .iter()
                .any(|valid| valid.start <= candidate.start && valid.end >= candidate.end)
            {
                continue;
            }

            let candidate_len = candidate.end - candidate.start;
            if candidate_len > validation_budget {
                continue;
            }
            validation_budget -= candidate_len;

            let candidate_text = &text[candidate.clone()];
            if serde_json::from_str::<Value>(candidate_text).is_ok() {
                valid_ranges.push(candidate);
            }
        }

        ranges.extend(valid_ranges);
    }

    fn scan_math(&self, text: &str, ranges: &mut Vec<ProtectedRange>) {
        let bytes = text.as_bytes();
        let mut cursor = 0;

        while cursor < bytes.len() {
            if bytes[cursor] != b'$' || is_escaped(bytes, cursor) {
                cursor += 1;
                continue;
            }

            let delimiter_len = if bytes.get(cursor + 1) == Some(&b'$') {
                2
            } else {
                1
            };
            let content_start = cursor + delimiter_len;
            let mut search = content_start;
            let mut closing = None;

            while search < bytes.len() {
                if bytes[search] == b'$' && !is_escaped(bytes, search) {
                    let matches_delimiter =
                        delimiter_len == 1 || bytes.get(search + 1) == Some(&b'$');
                    if matches_delimiter {
                        closing = Some(search + delimiter_len);
                        break;
                    }
                }
                if delimiter_len == 1 && bytes[search] == b'\n' {
                    break;
                }
                search += 1;
            }

            if let Some(end) = closing.filter(|_| search > content_start) {
                ranges.push(cursor..end);
                cursor = end;
            } else {
                cursor += delimiter_len;
            }
        }
    }

    fn scan_definitions(&self, text: &str, ranges: &mut Vec<ProtectedRange>) {
        for captures in self.definition_head.captures_iter(text) {
            let Some(head) = captures.name("head") else {
                continue;
            };
            let opening_parenthesis = head.end().saturating_sub(1);
            let line_end = line_end_without_newline(text, opening_parenthesis);
            let end =
                find_balanced_parenthesis(text, opening_parenthesis, line_end).unwrap_or(line_end);
            ranges.push(head.start()..end);
        }
    }
}

impl Default for ProtectionScanner {
    fn default() -> Self {
        Self::new().expect("built-in protection regexes must compile")
    }
}

fn find_balanced_json_at(text: &str, start: usize) -> Option<Range<usize>> {
    let bytes = text.as_bytes();
    let first = *bytes.get(start)?;
    if first != b'{' && first != b'[' {
        return None;
    }

    let mut stack = vec![first];
    let mut in_string = false;
    let mut escaped = false;

    for (offset, byte) in bytes[start + 1..].iter().copied().enumerate() {
        if in_string {
            if escaped {
                escaped = false;
            } else if byte == b'\\' {
                escaped = true;
            } else if byte == b'"' {
                in_string = false;
            }
            continue;
        }

        match byte {
            b'"' => in_string = true,
            b'{' | b'[' => stack.push(byte),
            b'}' | b']' => {
                let expected = if byte == b'}' { b'{' } else { b'[' };
                if stack.pop() != Some(expected) {
                    return None;
                }
                if stack.is_empty() {
                    let end = start + offset + 2;
                    return serde_json::from_str::<Value>(&text[start..end])
                        .is_ok()
                        .then_some(start..end);
                }
            }
            _ => {}
        }
    }

    None
}

fn line_end_without_newline(text: &str, start: usize) -> usize {
    let end = text[start..]
        .find('\n')
        .map_or(text.len(), |offset| start + offset);
    if end > start && text.as_bytes().get(end - 1) == Some(&b'\r') {
        end - 1
    } else {
        end
    }
}

fn line_end_with_newline(text: &str, start: usize) -> usize {
    text[start..]
        .find('\n')
        .map_or(text.len(), |offset| start + offset + 1)
}

fn find_closing_fence(
    text: &str,
    mut line_start: usize,
    marker: u8,
    minimum_len: usize,
) -> Option<usize> {
    while line_start < text.len() {
        let line_end = line_end_without_newline(text, line_start);
        let line = &text.as_bytes()[line_start..line_end];
        let indentation = line.iter().take_while(|byte| **byte == b' ').count();

        if indentation <= 3 {
            let marker_count = line[indentation..]
                .iter()
                .take_while(|byte| **byte == marker)
                .count();
            if marker_count >= minimum_len
                && line[indentation + marker_count..]
                    .iter()
                    .all(|byte| byte.is_ascii_whitespace())
            {
                return Some(line_end_with_newline(text, line_start));
            }
        }

        let next = line_end_with_newline(text, line_start);
        if next <= line_start {
            break;
        }
        line_start = next;
    }
    None
}

fn find_balanced_parenthesis(text: &str, opening: usize, limit: usize) -> Option<usize> {
    let bytes = text.as_bytes();
    if bytes.get(opening) != Some(&b'(') {
        return None;
    }

    let mut depth = 0usize;
    let mut quote = None;
    let mut escaped = false;

    for (offset, byte) in bytes[opening..limit].iter().copied().enumerate() {
        let index = opening + offset;
        if let Some(active_quote) = quote {
            if escaped {
                escaped = false;
            } else if byte == b'\\' {
                escaped = true;
            } else if byte == active_quote {
                quote = None;
            }
            continue;
        }

        match byte {
            b'\'' | b'"' | b'`' => quote = Some(byte),
            b'(' => depth += 1,
            b')' => {
                depth = depth.saturating_sub(1);
                if depth == 0 {
                    return Some(index + 1);
                }
            }
            _ => {}
        }
    }

    None
}

fn trim_token_end(text: &str, start: usize, mut end: usize) -> usize {
    while end > start {
        let byte = text.as_bytes()[end - 1];
        let trim = matches!(byte, b'.' | b',' | b';' | b':' | b'!' | b'?')
            || (byte == b')' && delimiter_is_unbalanced(&text[start..end], b'(', b')'))
            || (byte == b']' && delimiter_is_unbalanced(&text[start..end], b'[', b']'))
            || (byte == b'}' && delimiter_is_unbalanced(&text[start..end], b'{', b'}'));
        if !trim {
            break;
        }
        end -= 1;
    }
    end
}

fn delimiter_is_unbalanced(text: &str, opening: u8, closing: u8) -> bool {
    text.bytes().filter(|byte| *byte == closing).count()
        > text.bytes().filter(|byte| *byte == opening).count()
}

fn is_escaped(bytes: &[u8], index: usize) -> bool {
    bytes[..index]
        .iter()
        .rev()
        .take_while(|byte| **byte == b'\\')
        .count()
        % 2
        == 1
}

fn merge_ranges(text: &str, mut ranges: Vec<ProtectedRange>) -> Vec<ProtectedRange> {
    ranges.retain(|range| {
        range.start < range.end
            && range.end <= text.len()
            && text.is_char_boundary(range.start)
            && text.is_char_boundary(range.end)
    });
    ranges.sort_unstable_by_key(|range| (range.start, range.end));

    let mut merged: Vec<ProtectedRange> = Vec::with_capacity(ranges.len());
    for range in ranges {
        if let Some(previous) = merged.last_mut() {
            if range.start <= previous.end {
                previous.end = previous.end.max(range.end);
                continue;
            }
        }
        merged.push(range);
    }
    merged
}

fn complement_ranges(text_len: usize, protected: &[ProtectedRange]) -> Vec<Range<usize>> {
    let mut ranges = Vec::new();
    let mut cursor = 0;

    for protected_range in protected {
        if cursor < protected_range.start {
            ranges.push(cursor..protected_range.start);
        }
        cursor = cursor.max(protected_range.end);
    }
    if cursor < text_len {
        ranges.push(cursor..text_len);
    }

    ranges
}

#[cfg(test)]
mod tests {
    use super::*;
    use proptest::prelude::*;

    fn scanner() -> ProtectionScanner {
        ProtectionScanner::new().unwrap()
    }

    fn protected_text(text: &str) -> Vec<&str> {
        let scanner = scanner();
        scanner
            .scan(text)
            .into_iter()
            .map(|range| &text[range])
            .collect()
    }

    #[test]
    fn protects_backtick_tilde_and_unclosed_fences() {
        let text = "before\n```rust\nlet value = 1;\n```\nmiddle\n~~~\nraw\n~~~\nafter";
        let protected = protected_text(text);
        assert!(protected.iter().any(|value| value.starts_with("```rust")));
        assert!(protected.iter().any(|value| value.starts_with("~~~")));

        let unclosed = "text\n```\nlet π = true;";
        let range = scanner().scan(unclosed).pop().unwrap();
        assert_eq!(&unclosed[range], "```\nlet π = true;");
    }

    #[test]
    fn protects_indented_code_blocks_with_blank_lines() {
        let text = "intro\n    let first = 1;\n\n    let second = 2;\noutro";
        assert!(protected_text(text)
            .iter()
            .any(|value| value.contains("let first") && value.contains("let second")));
    }

    #[test]
    fn protects_supported_urls_without_sentence_punctuation() {
        let text = "See https://example.com/a?q=1, ftp://host/pub/file and file:///tmp/data.json.";
        let protected = protected_text(text);
        assert!(protected.contains(&"https://example.com/a?q=1"));
        assert!(protected.contains(&"ftp://host/pub/file"));
        assert!(protected.contains(&"file:///tmp/data.json"));
    }

    #[test]
    fn protects_unix_and_windows_paths() {
        let text = r"Open /usr/local/bin/tool and C:\Users\alice\project\main.rs now.";
        let protected = protected_text(text);
        assert!(protected.contains(&"/usr/local/bin/tool"));
        assert!(protected.contains(&r"C:\Users\alice\project\main.rs"));
    }

    #[test]
    fn protects_only_json_that_actually_parses() {
        let text = r#"prose {not valid json}; valid {"nested":[1,{"quote":"brace } and escaped \"quote\""}]} tail [true,null,3]"#;
        let protected = protected_text(text);
        assert!(!protected.contains(&"{not valid json}"));
        assert!(protected
            .iter()
            .any(|value| value.starts_with(r#"{"nested""#)));
        assert!(protected.contains(&"[true,null,3]"));
    }

    #[test]
    fn finds_valid_inner_json_inside_invalid_outer_delimiters() {
        let text = r#"broken { nope: [1, {"ok":true}] trailing"#;
        assert!(protected_text(text).contains(&r#"[1, {"ok":true}]"#));
    }

    #[test]
    fn protects_programming_identifier_styles_without_plain_words() {
        let text = "plain camelCase snake_case PascalCase SCREAMING_SNAKE_CASE other";
        let protected = protected_text(text);
        for identifier in [
            "camelCase",
            "snake_case",
            "PascalCase",
            "SCREAMING_SNAKE_CASE",
        ] {
            assert!(protected.contains(&identifier));
        }
        assert!(!protected.contains(&"plain"));
    }

    #[test]
    fn protects_inline_and_block_math_but_not_unclosed_or_escaped_dollars() {
        let text = "before $x + y$ and $$\nE = mc^2\n$$ after \\$literal and $unclosed";
        let protected = protected_text(text);
        assert!(protected.contains(&"$x + y$"));
        assert!(protected.contains(&"$$\nE = mc^2\n$$"));
        assert!(!protected.iter().any(|value| value.contains("literal")));
        assert!(!protected.iter().any(|value| value.contains("unclosed")));
    }

    #[test]
    fn protects_recognizable_function_and_tool_definitions() {
        let text = "fn calculateTotal(value: i64, nested: Option<(i32, i32)>) -> i64 {\nfunction runTask(name, options) {\ntool_call: {\"name\":\"lookup\",\"arguments\":{\"q\":\"rust\"}}\n<tool_use>{\"name\":\"read\"}</tool_use>";
        let protected = protected_text(text);
        assert!(protected
            .iter()
            .any(|value| value.contains("fn calculateTotal(value: i64")));
        assert!(protected
            .iter()
            .any(|value| value.contains("function runTask(name, options)")));
        assert!(protected
            .iter()
            .any(|value| value.contains("tool_call:") && value.contains(r#""q":"rust""#)));
        assert!(protected.iter().any(|value| value.contains("<tool_use>")));
    }

    #[test]
    fn ranges_are_merged_sorted_and_helpers_preserve_protected_bytes() {
        let text = "change https://example.com/snake_case and $x_y$ please";
        let scanner = scanner();
        let protected = scanner.scan(text);
        assert!(protected.windows(2).all(|pair| pair[0].end < pair[1].start));

        let output = scanner.transform_unprotected(text, |segment| segment.to_uppercase());
        assert!(output.contains("https://example.com/snake_case"));
        assert!(output.contains("$x_y$"));
        assert!(output.starts_with("CHANGE "));

        let url_start = text.find("https://").unwrap();
        assert!(ProtectionScanner::intersects_protected(
            &protected,
            url_start..url_start + 5
        ));
        assert!(ProtectionScanner::is_fully_protected(
            &protected,
            url_start..url_start + "https://example.com/snake_case".len()
        ));
    }

    #[test]
    fn handles_arbitrary_unicode_and_unclosed_delimiters_without_invalid_ranges() {
        let text = "🙂 Ελληνικά [\"未完了\\\" { $x π C:\\路径\\文件 /路径/文件";
        for range in scanner().scan(text) {
            assert!(text.is_char_boundary(range.start));
            assert!(text.is_char_boundary(range.end));
            assert!(range.start < range.end);
            assert!(range.end <= text.len());
        }
    }

    proptest! {
        #![proptest_config(ProptestConfig::with_cases(128))]

        #[test]
        fn property_ranges_are_valid_ordered_and_extract_byte_for_byte(
            prefix in ".{0,80}",
            body in "[A-Za-z0-9 _+=:/.-]{0,80}",
            suffix in ".{0,80}",
        ) {
            let text = format!(
                "{prefix}\n```text\n{body}\n```\nhttps://example.test/{body}\n{{\"value\":\"{body}\"}}\n{suffix}"
            );
            let scanner = scanner();
            let ranges = scanner.scan(&text);

            for range in &ranges {
                prop_assert!(range.start < range.end);
                prop_assert!(range.end <= text.len());
                prop_assert!(text.is_char_boundary(range.start));
                prop_assert!(text.is_char_boundary(range.end));
            }
            for pair in ranges.windows(2) {
                prop_assert!(pair[0].end < pair[1].start);
            }

            let output = scanner.transform_unprotected(&text, |segment| ".".repeat(segment.len()));
            prop_assert_eq!(output.len(), text.len());
            for range in ranges {
                prop_assert_eq!(
                    output[range.clone()].as_bytes(),
                    text[range].as_bytes()
                );
            }
        }
    }
}
