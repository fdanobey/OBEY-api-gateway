//! Data models, argument parsing, and tool-result rendering for the
//! provider-agnostic web search feature.

use serde::{Deserialize, Serialize};
use serde_json::{json, Value};

/// Default result count applied when the model omits `max_results`.
pub const DEFAULT_MAX_RESULTS: u8 = 5;

/// Default byte budget for rendered tool results (24 KiB).
pub const DEFAULT_RESULT_BUDGET_BYTES: usize = 24 * 1024;

const MAX_QUERY_CHARS: usize = 2000;
const MAX_DOMAINS: usize = 10;
const MAX_DOMAIN_ENTRY_CHARS: usize = 253;
const MIN_RECENCY_DAYS: u64 = 1;
const MAX_RECENCY_DAYS: u64 = 365;
const MIN_MAX_RESULTS: u64 = 1;
const MAX_MAX_RESULTS: u64 = 10;
const SNIPPET_CAP_STEPS: [usize; 13] = [2048, 1024, 512, 256, 128, 64, 32, 16, 8, 4, 2, 1, 0];

/// Arguments for the `web_search` tool.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct SearchQuery {
    pub q: String,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub domains: Option<Vec<String>>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub recency: Option<u32>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub max_results: Option<u8>,
}

/// A single normalized search result.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct SearchResult {
    pub title: String,
    pub url: String,
    pub snippet: String,
}

fn is_plausible_hostname(domain: &str) -> bool {
    if domain.chars().count() > MAX_DOMAIN_ENTRY_CHARS || domain.contains(' ') {
        return false;
    }
    domain.split('.').all(|label| {
        !label.is_empty()
            && !label.starts_with('-')
            && !label.ends_with('-')
            && label.chars().all(|c| c.is_ascii_alphanumeric() || c == '-')
    })
}

/// Parses the JSON arguments object the model sends for a `web_search` tool
/// call. Unknown fields are ignored; wrong types and out-of-range values are
/// rejected with an error mentioning the offending field.
pub fn parse_web_search_args(args: &str) -> Result<SearchQuery, String> {
    let value: Value =
        serde_json::from_str(args).map_err(|e| format!("invalid JSON arguments: {e}"))?;
    let obj = value
        .as_object()
        .ok_or_else(|| "arguments must be a JSON object".to_string())?;

    let q = match obj.get("q") {
        Some(Value::String(s)) => s.clone(),
        Some(_) => return Err("field 'q' must be a string".to_string()),
        None => return Err("missing required field 'q'".to_string()),
    };
    if q.trim().is_empty() {
        return Err("field 'q' must not be empty".to_string());
    }
    if q.chars().count() > MAX_QUERY_CHARS {
        return Err("field 'q' must be at most 2000 characters".to_string());
    }

    let domains = match obj.get("domains") {
        None | Some(Value::Null) => None,
        Some(Value::Array(arr)) => {
            if arr.len() > MAX_DOMAINS {
                return Err("field 'domains' must have at most 10 entries".to_string());
            }
        let mut domains = Vec::with_capacity(arr.len());
        for item in arr {
            let domain = item
                .as_str()
                .ok_or_else(|| "field 'domains' must be an array of strings".to_string())?;
            if domain.trim().is_empty() {
                return Err("field 'domains' must not contain empty entries".to_string());
            }
            if !is_plausible_hostname(domain) {
                return Err(format!("web_search tool: invalid domain '{domain}'"));
            }
            domains.push(domain.to_string());
        }
            Some(domains)
        }
        Some(_) => return Err("field 'domains' must be an array of strings".to_string()),
    };

    let recency = match obj.get("recency") {
        None | Some(Value::Null) => None,
        Some(Value::Number(n)) => {
            let days = n
                .as_u64()
                .ok_or_else(|| "field 'recency' must be a non-negative integer".to_string())?;
            if !(MIN_RECENCY_DAYS..=MAX_RECENCY_DAYS).contains(&days) {
                return Err("field 'recency' must be between 1 and 365 days".to_string());
            }
            Some(days as u32)
        }
        Some(_) => return Err("field 'recency' must be a non-negative integer".to_string()),
    };

    let max_results = match obj.get("max_results") {
        None | Some(Value::Null) => Some(DEFAULT_MAX_RESULTS),
        Some(Value::Number(n)) => {
            let n = n
                .as_u64()
                .ok_or_else(|| "field 'max_results' must be a non-negative integer".to_string())?;
            if !(MIN_MAX_RESULTS..=MAX_MAX_RESULTS).contains(&n) {
                return Err("field 'max_results' must be between 1 and 10".to_string());
            }
            Some(n as u8)
        }
        Some(_) => return Err("field 'max_results' must be a non-negative integer".to_string()),
    };

    Ok(SearchQuery {
        q,
        domains,
        recency,
        max_results,
    })
}

/// Renders normalized results as a JSON object with a `results` array under a
/// hard byte budget: snippets are truncated first (never splitting a multibyte
/// character), then whole results are dropped from the tail with a
/// `"truncated": true` marker field, then the list is cut entirely. Output is
/// always valid JSON; budgets below the minimal marker fall back to `{}`.
pub fn render_tool_results(results: &[SearchResult], max_bytes: usize) -> String {
    if results.is_empty() {
        let rendered = render_entries(&[], false);
        if rendered.len() <= max_bytes {
            return rendered;
        }
        return "{}".to_string();
    }

    let full: Vec<Value> = results.iter().map(|r| result_entry(r, usize::MAX)).collect();
    let rendered = render_entries(&full, false);
    if rendered.len() <= max_bytes {
        return rendered;
    }

    for cap in SNIPPET_CAP_STEPS {
        let entries: Vec<Value> = results.iter().map(|r| result_entry(r, cap)).collect();
        let rendered = render_entries(&entries, false);
        if rendered.len() <= max_bytes {
            return rendered;
        }
    }

    let mut count = results.len();
    while count > 0 {
        let entries: Vec<Value> = results[..count].iter().map(|r| result_entry(r, 0)).collect();
        let rendered = render_entries(&entries, true);
        if rendered.len() <= max_bytes {
            return rendered;
        }
        count -= 1;
    }

    let cut = render_entries(&[], true);
    if cut.len() <= max_bytes {
        return cut;
    }
    let bare = json!({"truncated": true}).to_string();
    if bare.len() <= max_bytes {
        return bare;
    }
    "{}".to_string()
}

fn render_entries(entries: &[Value], truncated: bool) -> String {
    if truncated {
        json!({ "results": entries, "truncated": true }).to_string()
    } else {
        json!({ "results": entries }).to_string()
    }
}

fn result_entry(result: &SearchResult, snippet_cap: usize) -> Value {
    json!({
        "title": result.title,
        "url": result.url,
        "snippet": truncate_at_char_boundary(&result.snippet, snippet_cap),
    })
}

fn truncate_at_char_boundary(s: &str, max_bytes: usize) -> &str {
    if s.len() <= max_bytes {
        return s;
    }
    let mut end = max_bytes;
    while end > 0 && !s.is_char_boundary(end) {
        end -= 1;
    }
    &s[..end]
}

#[cfg(test)]
mod tests {
    use super::*;

    fn result(title: &str, url: &str, snippet: &str) -> SearchResult {
        SearchResult {
            title: title.to_string(),
            url: url.to_string(),
            snippet: snippet.to_string(),
        }
    }

    #[test]
    fn parse_minimal_query_only_applies_default_max_results() {
        let q = parse_web_search_args(r#"{"q":"rust serde"}"#).unwrap();
        assert_eq!(q.q, "rust serde");
        assert_eq!(q.domains, None);
        assert_eq!(q.recency, None);
        assert_eq!(q.max_results, Some(DEFAULT_MAX_RESULTS));
    }

    #[test]
    fn parse_full_arguments() {
        let args = json!({
            "q": "rust serde",
            "domains": ["docs.rs", "crates.io"],
            "recency": 30,
            "max_results": 3
        })
        .to_string();
        let q = parse_web_search_args(&args).unwrap();
        assert_eq!(q.q, "rust serde");
        assert_eq!(
            q.domains,
            Some(vec!["docs.rs".to_string(), "crates.io".to_string()])
        );
        assert_eq!(q.recency, Some(30));
        assert_eq!(q.max_results, Some(3));
    }

    #[test]
    fn parse_ignores_unknown_fields() {
        let args = json!({"q": "x", "future_field": 123}).to_string();
        assert!(parse_web_search_args(&args).is_ok());
    }

    #[test]
    fn parse_null_optional_fields_treated_as_absent() {
        let args = json!({"q": "x", "domains": null, "recency": null, "max_results": null}).to_string();
        let q = parse_web_search_args(&args).unwrap();
        assert_eq!(q.domains, None);
        assert_eq!(q.recency, None);
        assert_eq!(q.max_results, Some(DEFAULT_MAX_RESULTS));
    }

    #[test]
    fn parse_missing_q_rejected() {
        let err = parse_web_search_args("{}").unwrap_err();
        assert!(err.contains("missing required field 'q'"), "{err}");
    }

    #[test]
    fn parse_empty_q_rejected() {
        let args = json!({"q": ""}).to_string();
        let err = parse_web_search_args(&args).unwrap_err();
        assert!(err.contains("field 'q' must not be empty"), "{err}");
    }

    #[test]
    fn parse_whitespace_q_rejected() {
        let args = json!({"q": "   "}).to_string();
        assert!(parse_web_search_args(&args)
            .unwrap_err()
            .contains("field 'q' must not be empty"));
    }

    #[test]
    fn parse_q_exactly_2000_chars_accepted() {
        let args = json!({"q": "x".repeat(2000)}).to_string();
        assert!(parse_web_search_args(&args).is_ok());
    }

    #[test]
    fn parse_q_over_2000_chars_rejected() {
        let args = json!({"q": "x".repeat(2001)}).to_string();
        let err = parse_web_search_args(&args).unwrap_err();
        assert!(err.contains("at most 2000"), "{err}");
    }

    #[test]
    fn parse_domains_exactly_10_accepted() {
        let domains: Vec<String> = (0..10).map(|i| format!("d{i}.example")).collect();
        let args = json!({"q": "x", "domains": domains}).to_string();
        assert!(parse_web_search_args(&args).is_ok());
    }

    #[test]
    fn parse_domains_over_10_rejected() {
        let domains: Vec<String> = (0..11).map(|i| format!("d{i}.example")).collect();
        let args = json!({"q": "x", "domains": domains}).to_string();
        let err = parse_web_search_args(&args).unwrap_err();
        assert!(err.contains("field 'domains' must have at most 10 entries"), "{err}");
    }

    #[test]
    fn parse_empty_domain_entry_rejected() {
        let args = json!({"q": "x", "domains": ["a.example", ""]}).to_string();
        let err = parse_web_search_args(&args).unwrap_err();
        assert!(
            err.contains("field 'domains' must not contain empty entries"),
            "{err}"
        );
    }

    #[test]
    fn parse_valid_domain_entries_accepted() {
        for domain in [
            "example.com",
            "docs.rs",
            "sub-domain.example.co.uk",
            "xn--80ak6aa92e.example",
            "a.b",
        ] {
            let args = json!({"q": "x", "domains": [domain]}).to_string();
            let parsed = parse_web_search_args(&args)
                .unwrap_or_else(|e| panic!("{domain} should parse: {e}"));
            assert_eq!(parsed.domains, Some(vec![domain.to_string()]));
        }
    }

    #[test]
    fn parse_domain_with_scheme_rejected() {
        let args = json!({"q": "x", "domains": ["https://example.com"]}).to_string();
        let err = parse_web_search_args(&args).unwrap_err();
        assert!(
            err.contains("web_search tool: invalid domain 'https://example.com'"),
            "{err}"
        );
    }

    #[test]
    fn parse_domain_with_spaces_rejected() {
        let args = json!({"q": "x", "domains": ["exa mple.com"]}).to_string();
        let err = parse_web_search_args(&args).unwrap_err();
        assert!(err.contains("web_search tool: invalid domain"), "{err}");
    }

    #[test]
    fn parse_domain_with_userinfo_rejected() {
        let args = json!({"q": "x", "domains": ["user@example.com"]}).to_string();
        let err = parse_web_search_args(&args).unwrap_err();
        assert!(err.contains("web_search tool: invalid domain 'user@example.com'"), "{err}");
    }

    #[test]
    fn parse_domain_with_port_rejected() {
        let args = json!({"q": "x", "domains": ["example.com:8080"]}).to_string();
        assert!(parse_web_search_args(&args)
            .unwrap_err()
            .contains("web_search tool: invalid domain"));
    }

    #[test]
    fn parse_domain_over_253_chars_rejected() {
        let long = format!("{}.example", "a".repeat(250));
        assert!(long.chars().count() > 253);
        let args = json!({"q": "x", "domains": [long]}).to_string();
        assert!(parse_web_search_args(&args)
            .unwrap_err()
            .contains("web_search tool: invalid domain"));
    }

    #[test]
    fn parse_domain_exactly_253_chars_accepted() {
        let long = format!("{}.example", "a".repeat(245));
        assert_eq!(long.chars().count(), 253);
        let args = json!({"q": "x", "domains": [long]}).to_string();
        assert!(parse_web_search_args(&args).is_ok());
    }

    #[test]
    fn parse_domain_empty_labels_rejected() {
        for bad in ["example.com.", ".example.com", "exa..mple.com"] {
            let args = json!({"q": "x", "domains": [bad]}).to_string();
            let err = parse_web_search_args(&args).unwrap_err();
            assert!(err.contains("web_search tool: invalid domain"), "{bad}: {err}");
        }
    }

    #[test]
    fn parse_domain_leading_trailing_hyphen_label_rejected() {
        for bad in ["-example.com", "example-.com", "ex-ample.com"] {
            let args = json!({"q": "x", "domains": [bad]}).to_string();
            let parsed = parse_web_search_args(&args);
            if bad == "ex-ample.com" {
                assert!(parsed.is_ok(), "inner hyphen label is valid");
            } else {
                assert!(
                    parsed.unwrap_err().contains("web_search tool: invalid domain"),
                    "{bad} must be rejected"
                );
            }
        }
    }

    #[test]
    fn parse_recency_boundaries_accepted() {
        for recency in [1u32, 365] {
            let args = json!({"q": "x", "recency": recency}).to_string();
            assert_eq!(parse_web_search_args(&args).unwrap().recency, Some(recency));
        }
    }

    #[test]
    fn parse_recency_zero_rejected() {
        let args = json!({"q": "x", "recency": 0}).to_string();
        let err = parse_web_search_args(&args).unwrap_err();
        assert!(err.contains("field 'recency' must be between 1 and 365"), "{err}");
    }

    #[test]
    fn parse_recency_366_rejected() {
        let args = json!({"q": "x", "recency": 366}).to_string();
        assert!(parse_web_search_args(&args)
            .unwrap_err()
            .contains("field 'recency' must be between 1 and 365"));
    }

    #[test]
    fn parse_negative_recency_rejected() {
        let args = json!({"q": "x", "recency": -1}).to_string();
        let err = parse_web_search_args(&args).unwrap_err();
        assert!(err.contains("field 'recency' must be a non-negative integer"), "{err}");
    }

    #[test]
    fn parse_max_results_boundaries_accepted() {
        for max_results in [1u8, 10] {
            let args = json!({"q": "x", "max_results": max_results}).to_string();
            assert_eq!(
                parse_web_search_args(&args).unwrap().max_results,
                Some(max_results)
            );
        }
    }

    #[test]
    fn parse_max_results_zero_rejected() {
        let args = json!({"q": "x", "max_results": 0}).to_string();
        let err = parse_web_search_args(&args).unwrap_err();
        assert!(err.contains("field 'max_results' must be between 1 and 10"), "{err}");
    }

    #[test]
    fn parse_max_results_11_rejected() {
        let args = json!({"q": "x", "max_results": 11}).to_string();
        assert!(parse_web_search_args(&args)
            .unwrap_err()
            .contains("field 'max_results' must be between 1 and 10"));
    }

    #[test]
    fn parse_non_string_q_rejected() {
        let args = json!({"q": 42}).to_string();
        let err = parse_web_search_args(&args).unwrap_err();
        assert!(err.contains("field 'q' must be a string"), "{err}");
    }

    #[test]
    fn parse_non_array_domains_rejected() {
        let args = json!({"q": "x", "domains": "example.com"}).to_string();
        let err = parse_web_search_args(&args).unwrap_err();
        assert!(err.contains("field 'domains' must be an array of strings"), "{err}");
    }

    #[test]
    fn parse_non_string_domain_entry_rejected() {
        let args = json!({"q": "x", "domains": [1]}).to_string();
        assert!(parse_web_search_args(&args)
            .unwrap_err()
            .contains("field 'domains' must be an array of strings"));
    }

    #[test]
    fn parse_non_integer_recency_rejected() {
        for args in [
            json!({"q": "x", "recency": "week"}).to_string(),
            json!({"q": "x", "recency": 2.5}).to_string(),
        ] {
            assert!(parse_web_search_args(&args)
                .unwrap_err()
                .contains("field 'recency'"));
        }
    }

    #[test]
    fn parse_non_integer_max_results_rejected() {
        for args in [
            json!({"q": "x", "max_results": [5]}).to_string(),
            json!({"q": "x", "max_results": "5"}).to_string(),
        ] {
            assert!(parse_web_search_args(&args)
                .unwrap_err()
                .contains("field 'max_results'"));
        }
    }

    #[test]
    fn parse_non_object_arguments_rejected() {
        let err = parse_web_search_args("[1,2]").unwrap_err();
        assert!(err.contains("arguments must be a JSON object"), "{err}");
    }

    #[test]
    fn parse_invalid_json_rejected() {
        let err = parse_web_search_args("not json").unwrap_err();
        assert!(err.contains("invalid JSON arguments"), "{err}");
    }

    #[test]
    fn render_small_list_under_budget_passthrough() {
        let results = vec![
            result("t1", "https://a.example", "s1"),
            result("t2", "https://b.example", "s2"),
        ];
        let out = render_tool_results(&results, DEFAULT_RESULT_BUDGET_BYTES);
        let v: Value = serde_json::from_str(&out).unwrap();
        assert_eq!(v["results"].as_array().unwrap().len(), 2);
        assert_eq!(v["results"][0]["title"], "t1");
        assert_eq!(v["results"][0]["url"], "https://a.example");
        assert_eq!(v["results"][0]["snippet"], "s1");
        assert_eq!(v["results"][1]["snippet"], "s2");
        assert!(v.get("truncated").is_none());
    }

    #[test]
    fn render_empty_results() {
        let out = render_tool_results(&[], DEFAULT_RESULT_BUDGET_BYTES);
        assert_eq!(out, r#"{"results":[]}"#);
        assert!(serde_json::from_str::<Value>(&out).is_ok());
    }

    #[test]
    fn render_snippet_truncation_keeps_title_and_url() {
        let results = vec![result("t", "https://example.com/a", &"x".repeat(300))];
        let out = render_tool_results(&results, 100);
        assert!(out.len() <= 100, "len {} > 100", out.len());
        let v: Value = serde_json::from_str(&out).unwrap();
        assert_eq!(v["results"].as_array().unwrap().len(), 1);
        assert_eq!(v["results"][0]["title"], "t");
        assert_eq!(v["results"][0]["url"], "https://example.com/a");
        assert_eq!(v["results"][0]["snippet"].as_str().unwrap().len(), 16);
        assert!(v.get("truncated").is_none());
    }

    #[test]
    fn render_drop_results_marks_truncated() {
        let results: Vec<SearchResult> = (0..5)
            .map(|i| {
                result(
                    &format!("Result {i}"),
                    &format!("https://example.com/{i}"),
                    "body",
                )
            })
            .collect();
        let out = render_tool_results(&results, 150);
        assert!(out.len() <= 150, "len {} > 150", out.len());
        let v: Value = serde_json::from_str(&out).unwrap();
        assert_eq!(v["truncated"], json!(true));
        let arr = v["results"].as_array().unwrap();
        assert_eq!(arr.len(), 1);
        assert_eq!(arr[0]["title"], "Result 0");
    }

    #[test]
    fn render_respects_byte_budget_across_sizes() {
        let results: Vec<SearchResult> = (0..5)
            .map(|i| {
                result(
                    &format!("Result {i}"),
                    &format!("https://example.com/{i}"),
                    &"lorem ipsum ".repeat(20),
                )
            })
            .collect();
        for budget in [64usize, 100, 150, 300, 1024, 4096] {
            let out = render_tool_results(&results, budget);
            assert!(out.len() <= budget, "budget {budget} exceeded: {}", out.len());
            assert!(
                serde_json::from_str::<Value>(&out).is_ok(),
                "invalid JSON at budget {budget}: {out}"
            );
        }
    }

    #[test]
    fn render_multibyte_safety_cjk() {
        let snippet = "検".repeat(100);
        let results = vec![result("CJK", "https://example.jp", &snippet)];
        let out = render_tool_results(&results, 60);
        assert!(out.len() <= 60, "len {} > 60", out.len());
        let v: Value = serde_json::from_str(&out).unwrap();
        assert!(
            serde_json::from_str::<Value>(&out).is_ok(),
            "invalid JSON: {out}"
        );
        if let Some(arr) = v["results"].as_array() {
            if !arr.is_empty() {
                if let Some(snippet_out) = arr[0]["snippet"].as_str() {
                    assert!(snippet.starts_with(snippet_out));
                }
            }
        }
    }

    #[test]
    fn render_floor_below_marker_budget_stays_valid_json() {
        let results = vec![result("t", "u", "s")];
        let out = render_tool_results(&results, 1);
        assert!(serde_json::from_str::<Value>(&out).is_ok());
    }

    #[test]
    fn default_result_budget_is_24_kib() {
        assert_eq!(DEFAULT_RESULT_BUDGET_BYTES, 24 * 1024);
    }
}
