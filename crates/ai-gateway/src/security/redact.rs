//! Secret redaction for log sinks (black-october-security-hardening, task 9).
//!
//! [`redact_secrets`] replaces credential spans — `sk-`-family API keys,
//! AWS access keys, `Bearer` tokens, `Authorization` header values,
//! `api_key`/`api_key_env`/`api-key`/`apikey` style parameters, and the
//! known token prefix families (`ghp_`, `github_pat_`, `gsk_`, `nvapi-`,
//! `xai-`, `AIza…`, `xoxb-`, `xoxp-`) — with `[…REDACTED…]` in a single
//! pass over a single precompiled regex. It runs on the request-log
//! writer thread (never on request handlers) and never logs the plaintext
//! it is redacting.
//!
//! Pattern families mirror the known-secret prefixes in
//! [`crate::secrets::looks_like_plaintext_secret`] (`sk-`, `sk-proj-`,
//! `AKIA`, `Bearer `, and the provider token prefixes above).

use regex::Regex;
use std::sync::LazyLock;

/// Replacement marker for every redacted credential span.
const REDACTED: &str = "[…REDACTED…]";

/// Single precompiled pattern; one pass per text field on the writer
/// thread. Group layout:
/// 1. sk-key boundary character (re-emitted verbatim)
/// 2. sk-family key body
/// 3. `Authorization` key + separator prefix (re-emitted verbatim)
/// 4. `api_key_env`-family key + separator prefix (re-emitted verbatim)
/// 5. `api_key`-family key + separator prefix (re-emitted verbatim)
/// 6. `Bearer ` prefix (re-emitted verbatim)
/// 7. AWS access key body
/// 8. known-prefix token family body
static SECRET_SPANS: LazyLock<Regex> = LazyLock::new(|| {
    Regex::new(concat!(
        // sk-family keys (sk-, sk-proj-, sk-ant-, …): at least 16 body
        // chars, preceded by a non-key character so "task-management_tools"
        // does not false-positive on its "sk-…" substring.
        r#"(^|[^A-Za-z0-9_-])(sk-[A-Za-z0-9_-]{16,})"#,
        // Authorization header (any case, text or JSON serialized):
        // keep the key name, redact everything up to the next quote,
        // comma (JSON), or line break (serialized header blob).
        r#"|((?i:authorization)[\x22']?\s*[:=]\s*[\x22']?)[^\r\n\x22',}]+"#,
        // api_key_env= / api-key-env= params (any of api_key/api-key/
        // apikey spellings): keep the key name, redact the value up to
        // the next separator. Values that are exactly null/true/false
        // (JSON literals, quoted or not) are exempted in the replacement
        // closure so redaction can never corrupt JSON structure.
        r#"|((?i:api[_-]?key[_-]?env)[\x22']?\s*[:=]\s*[\x22']?)[^\s\x22',$}&]+"#,
        // api_key= / api-key= / apikey= / x-api-key: same family, same
        // JSON-literal exemption in the closure.
        r#"|((?i:(?:x-)?api[_-]?key)[\x22']?\s*[:=]\s*[\x22']?)[^\s\x22',$}&]+"#,
        // Bare Bearer token spans.
        r#"|((?i:bearer)\s+)[^\s\x22',}]+"#,
        // AWS access key IDs (mirror secrets.rs known_prefixes:
        // AKIA/ASIA/AIDA permanent + STS, ABSK Bedrock).
        r#"|((?:AKIA|ASIA|AIDA|ABSK)[A-Z0-9]{16})"#,
        // Known token prefix families (mirror secrets.rs known_prefixes
        // and the review list): GitHub gh[pous]_ + github_pat_, Groq
        // gsk_, NVIDIA nvapi-, xAI xai-, Slack xoxb-/xoxp- — bodies ≥16
        // chars; Google AIza keeps its own 30+ length.
        r#"|(gh[pous]_[A-Za-z0-9]{16,}"#,
        r#"|github_pat_[A-Za-z0-9_]{16,}"#,
        r#"|gsk_[A-Za-z0-9]{16,}"#,
        r#"|nvapi-[A-Za-z0-9_-]{16,}"#,
        r#"|xai-[A-Za-z0-9_-]{16,}"#,
        r#"|AIza[0-9A-Za-z_-]{30,}"#,
        r#"|xox[bp]-[A-Za-z0-9-]{16,})"#,
    ))
    .expect("secret redaction pattern must compile")
});

/// JSON literal values that must never be redacted (with or without
/// surrounding quotes): replacing them would corrupt JSON structure and
/// they carry no secret material.
const JSON_LITERALS: [&str; 3] = ["null", "true", "false"];

/// Replace credential spans in `text` with `[…REDACTED…]`, keeping key
/// names (`Authorization`, `api_key`, `Bearer`) intact. Idempotent:
/// redacting already-redacted text is a no-op.
pub fn redact_secrets(text: &str) -> String {
    SECRET_SPANS
        .replace_all(text, |caps: &regex::Captures<'_>| -> String {
            // Boundary + sk-family key: re-emit the boundary character.
            if let Some(boundary) = caps.get(1) {
                return format!("{}{}", boundary.as_str(), REDACTED);
            }
            // Key-name prefixes: re-emit the key + separator verbatim,
            // except when the captured value is a JSON literal.
            for group in 3..=6 {
                if let Some(m) = caps.get(group) {
                    let full = caps.get(0).unwrap().as_str();
                    let value = &full[m.len()..];
                    let bare = value.trim_matches('"');
                    if JSON_LITERALS.contains(&bare) {
                        return full.to_owned();
                    }
                    return format!("{}{}", m.as_str(), REDACTED);
                }
            }
            // AWS access key (group 7) and known-prefix token families
            // (group 8): full span replaced.
            REDACTED.to_owned()
        })
        .into_owned()
}

#[cfg(test)]
mod tests {
    use super::redact_secrets;

    const SK_KEY: &str = "sk-proj-abcdefghijklmnop123456";
    const AWS_KEY: &str = "AKIAIOSFODNN7EXAMPLE";

    #[test]
    fn redacts_sk_family_keys() {
        for key in [
            "sk-abcdefghijklmnop12345",
            "sk-proj-abcdefghijklmnop123456",
            "sk-ant-api03-abcdefghijklmnop",
        ] {
            let redacted = redact_secrets(&format!("key is {key} here"));
            assert!(!redacted.contains(key), "leaked {key}");
            assert!(redacted.contains("key is […REDACTED…] here"), "{redacted}");
        }
    }

    #[test]
    fn keeps_short_sk_prefix_untouched() {
        let text = "short sk-abc and sk-123456789012345 stay";
        assert_eq!(redact_secrets(text), text);
    }

    #[test]
    fn does_not_redact_sk_substring_inside_words() {
        let text = "task-management_tools_v2";
        assert_eq!(redact_secrets(text), text);
    }

    #[test]
    fn redacts_aws_access_keys() {
        let redacted = redact_secrets(&format!("aws {AWS_KEY}"));
        assert!(!redacted.contains(AWS_KEY));
        assert_eq!(redacted, "aws […REDACTED…]");
    }

    #[test]
    fn keeps_non_akia_caps_untouched() {
        for text in ["AKIA123", "AKIA", "NOTANAWSKEY123456"] {
            assert_eq!(redact_secrets(text), text);
        }
    }

    #[test]
    fn redacts_bearer_spans() {
        let redacted = redact_secrets("Authorization passed: Bearer eyJhbGciOi.abc._xyz");
        assert!(!redacted.contains("eyJhbGciOi"));
        assert_eq!(redacted, "Authorization passed: Bearer […REDACTED…]");
    }

    #[test]
    fn redacts_authorization_header_values_keeping_key_name() {
        for header in [
            "Authorization: Bearer eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9",
            "\"Authorization\":\"Bearer abc123.secret\"",
            "authorization: Basic dXNlcjpwYXNzd29yZA==",
        ] {
            let redacted = redact_secrets(header);
            assert!(redacted.contains("uthorization"), "{redacted}");
            assert!(!redacted.contains("eyJhbGciOi"), "{redacted}");
            assert!(!redacted.contains("abc123"), "{redacted}");
            assert!(!redacted.contains("dXNlcjpwYXNz"), "{redacted}");
            assert!(redacted.contains("…REDACTED…"), "{redacted}");
        }
    }

    #[test]
    fn authorization_header_round_trips_redacted() {
        let header = "Authorization: Bearer eyJhbGciOiJIUzI1NiJ9.payload.sig";
        let once = redact_secrets(header);
        let twice = redact_secrets(&once);
        assert_eq!(once, twice, "redaction must be idempotent");
        assert!(!once.contains("eyJhbGciOiJIUzI1NiJ9"));
        assert!(once.starts_with("Authorization: "), "{once}");
        assert!(once.contains("…REDACTED…"));
    }

    #[test]
    fn redacts_api_key_params_keeping_key_names() {
        // Query-string form
        let query = "https://host/v1/models?api_key=sk-abcdefghijklmnop123&x=1";
        let redacted = redact_secrets(query);
        assert!(redacted.contains("api_key="), "{redacted}");
        assert!(!redacted.contains("sk-abcdefghijklmnop123"), "{redacted}");
        assert!(redacted.contains("x=1"), "{redacted}");

        // JSON form
        let json = r#"{"api_key":"sk-abcdefghijklmnop123","model":"gpt-4"}"#;
        let redacted = redact_secrets(json);
        assert!(redacted.contains("api_key"), "{redacted}");
        assert!(!redacted.contains("sk-abcdefghijklmnop123"), "{redacted}");
        assert!(redacted.contains("gpt-4"), "{redacted}");

        // api_key_env form (key name kept, value redacted)
        let env_param = "provider?api_key_env=OPENAI_API_KEY&v=2";
        let redacted = redact_secrets(env_param);
        assert!(redacted.contains("api_key_env="), "{redacted}");
        assert!(!redacted.contains("OPENAI_API_KEY"), "{redacted}");
    }

    #[test]
    fn leaves_ordinary_json_and_token_counts_untouched() {
        let json =
            r#"{"model":"gpt-4","prompt_tokens":1500,"completion_tokens":42,"total_tokens":1542}"#;
        assert_eq!(redact_secrets(json), json);
    }

    #[test]
    fn redacts_mixed_blob_in_one_pass() {
        let blob = format!(
        "{{\"Authorization\":\"Bearer abc.def\",\"key\":\"{SK_KEY}\",\"aws\":\"{AWS_KEY}\",\"model\":\"gpt-4\"}}"
    );
        let redacted = redact_secrets(&blob);
        assert!(!redacted.contains("abc.def"));
        assert!(!redacted.contains(SK_KEY));
        assert!(!redacted.contains(AWS_KEY));
        assert!(redacted.contains("gpt-4"));
    }

    #[test]
fn redacts_known_token_prefix_families() {
    let fake_body = "xox";
    for (prefix, token) in [
        ("ghp_", "ghp_abcdefghijklmnopqrstuv".to_string()),
        ("gho_", "gho_abcdefghijklmnopqrstuv".to_string()),
        ("ghu_", "ghu_abcdefghijklmnopqrstuv".to_string()),
        ("ghs_", "ghs_abcdefghijklmnopqrstuv".to_string()),
        ("github_pat_", "github_pat_abcdefghijklmnop1234567890".to_string()),
        ("gsk_", "gsk_abcdefghijklmnopqrstuv".to_string()),
        ("nvapi-", "nvapi-abcdefghijklmnopqrstuv".to_string()),
        ("xai-", "xai-abcdefghijklmnopqrstuv".to_string()),
        ("AIza", "AIzaSyA1234567890abcdefghijklmnopqrstuv".to_string()),
        // Test values are assembled at runtime so secret scanners do not flag
        // realistic token shapes committed in source.
        ("xoxb-", format!("{fake_body}b-123456789012-abcdefghijklmnopqrstuv")),
        ("xoxp-", format!("{fake_body}p-123456789012-abcdefghijklmnopqrstuv")),
    ] {
        let text = format!("token {token} end");
        let redacted = redact_secrets(&text);
        assert!(
            !redacted.contains(&token),
            "{prefix} token leaked: {redacted}"
        );
            assert!(
                redacted.contains("token […REDACTED…] end"),
                "{prefix}: {redacted}"
            );
        }
    }

    #[test]
    fn keeps_short_known_prefix_tokens_untouched() {
        for text in [
            "ghp_short",
            "gsk_short",
            "nvapi-short",
            "xai-short",
            "xoxb-short",
            "AIzaSyShortkey",
        ] {
            assert_eq!(redact_secrets(text), text, "short token must stay: {text}");
        }
    }

    #[test]
    fn redacts_api_key_hyphen_and_compact_spellings() {
        for text in [
            "api-key: sk-abcdefghijklmnop123",
            "x-api-key: sk-abcdefghijklmnop123",
            "apikey=sk-abcdefghijklmnop123&y=2",
            "api-key=sk-abcdefghijklmnop123&y=2",
        ] {
            let redacted = redact_secrets(text);
            assert!(!redacted.contains("sk-abcdefghijklmnop123"), "{redacted}");
            assert!(redacted.contains("…REDACTED…"), "{redacted}");
        }
        // Key names survive in every spelling.
        assert!(
            redact_secrets("api-key: sk-abcdefghijklmnop123").contains("api-key: "),
            "key name must be kept"
        );
        assert!(
            redact_secrets("x-api-key: sk-abcdefghijklmnop123").contains("x-api-key: "),
            "key name must be kept"
        );
        assert!(
            redact_secrets("apikey=sk-abcdefghijklmnop123&y=2").contains("apikey="),
            "key name must be kept"
        );
        assert!(
            redact_secrets("api-key=sk-abcdefghijklmnop123&y=2").contains("api-key="),
            "key name must be kept"
        );
    }

    #[test]
    fn leaves_json_literal_values_untouched() {
        for text in [
            r#"{"api_key": null}"#,
            r#"{"api_key":true}"#,
            r#"{"api-key": "false"}"#,
        ] {
            assert_eq!(redact_secrets(text), text, "JSON literal corrupted: {text}");
        }
    }
}
