use axum::http::{header, HeaderMap};

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum ClientTool {
    ClaudeCode,
    Codex,
    Cursor,
    GeminiCli,
    QwenCode,
    RooCode,
    OpenCode,
    Unattributed,
}

impl ClientTool {
    pub fn as_str(&self) -> &'static str {
        match self {
            ClientTool::ClaudeCode => "claude-code",
            ClientTool::Codex => "codex",
            ClientTool::Cursor => "cursor",
            ClientTool::GeminiCli => "gemini-cli",
            ClientTool::QwenCode => "qwen-code",
            ClientTool::RooCode => "roo-code",
            ClientTool::OpenCode => "opencode",
            ClientTool::Unattributed => "unattributed",
        }
    }
}

const SIGNATURES: &[(&str, &[&str])] = &[
    ("claude-code", &["claude-cli", "claude-code"]),
    ("codex", &["codex/", "codex_cli", "codex-http"]),
    ("cursor", &["cursor/", "cursor-"]),
    ("gemini-cli", &["gemini-cli", "google-gemini"]),
    ("qwen-code", &["qwen-code", "qwen_cli"]),
    ("roo-code", &["roo-code", "roo_cline"]),
    ("opencode", &["opencode"]),
];

fn is_token_start(ua: &str, idx: usize) -> bool {
    idx == 0 || matches!(ua.as_bytes()[idx - 1], b' ' | b'/' | b'(')
}

fn signature_present(ua: &str, sig: &str) -> bool {
    let sig_lower = sig.to_ascii_lowercase();
    let hay = ua.to_ascii_lowercase();
    let mut from = 0;
    while let Some(pos) = hay[from..].find(&sig_lower) {
        let idx = from + pos;
        if is_token_start(&hay, idx) {
            return true;
        }
        from = idx + 1;
    }
    false
}

fn from_token(token: &str) -> ClientTool {
    match token {
        "claude-code" => ClientTool::ClaudeCode,
        "codex" => ClientTool::Codex,
        "cursor" => ClientTool::Cursor,
        "gemini-cli" => ClientTool::GeminiCli,
        "qwen-code" => ClientTool::QwenCode,
        "roo-code" => ClientTool::RooCode,
        "opencode" => ClientTool::OpenCode,
        _ => ClientTool::Unattributed,
    }
}

pub fn classify(headers: &HeaderMap) -> ClientTool {
    let ua = headers
        .get(header::USER_AGENT)
        .and_then(|v| v.to_str().ok())
        .map(str::trim)
        .filter(|s| !s.is_empty());
    let Some(ua) = ua else {
        return ClientTool::Unattributed;
    };
    for (token, sigs) in SIGNATURES {
        if sigs.iter().any(|sig| signature_present(ua, sig)) {
            return from_token(token);
        }
    }
    ClientTool::Unattributed
}

#[cfg(test)]
mod tests {
    use super::*;

    fn headers_with(ua: Option<&str>) -> HeaderMap {
        let mut h = HeaderMap::new();
        if let Some(ua) = ua {
            h.insert(
                header::USER_AGENT,
                axum::http::HeaderValue::from_str(ua).unwrap(),
            );
        }
        h
    }

    fn classify_ua(ua: Option<&str>) -> ClientTool {
        classify(&headers_with(ua))
    }

    #[test]
    fn every_vocabulary_row_positive() {
        for (ua, expected) in [
            ("claude-cli/2.0.1", ClientTool::ClaudeCode),
            ("claude-code/1.0.0 (Windows)", ClientTool::ClaudeCode),
            ("codex/0.42.0", ClientTool::Codex),
            ("codex_cli_rs/1.0", ClientTool::Codex),
            ("codex-http/1.2", ClientTool::Codex),
            ("cursor/0.48.0", ClientTool::Cursor),
            ("Gemini-CLI/0.8.1", ClientTool::GeminiCli),
            ("google-gemini/1.0", ClientTool::GeminiCli),
            ("Qwen-Code/0.9.0", ClientTool::QwenCode),
            ("Qwen_Cli/2.0", ClientTool::QwenCode),
            ("Roo-Code/3.18.0", ClientTool::RooCode),
            ("roo_cline/1.5", ClientTool::RooCode),
            ("OpenCode/0.6.0", ClientTool::OpenCode),
            ("agent (opencode/1.0)", ClientTool::OpenCode),
        ] {
            assert_eq!(classify_ua(Some(ua)), expected, "ua={ua}");
        }
    }

    #[test]
    fn every_vocabulary_row_negative() {
        for ua in [
            "vscode/1.95.0",
            "warp/2025.1",
            "my-cursor-tool/1.0",
            "my-codex-tool/1.0",
            "notcursor/1.0",
            "precursor/2.0",
            "bulk-cursor/1.0",
            "node-fetch/3.0",
            "python-requests/2.32",
            "my-codex_cli/1.0",
            "abcodex_cli/1.0",
        ] {
            assert_eq!(classify_ua(Some(ua)), ClientTool::Unattributed, "ua={ua}");
        }
    }

    #[test]
    fn token_boundary_guard_cursor_embedded() {
        assert_eq!(classify_ua(Some("my-cursor-tool/1.0")), ClientTool::Unattributed);
        assert_eq!(classify_ua(Some("x-cursor/1.0")), ClientTool::Unattributed);
        assert_eq!(classify_ua(Some("cursor/1.0")), ClientTool::Cursor);
        assert_eq!(classify_ua(Some("cursor-agent/0.5")), ClientTool::Cursor);
        assert_eq!(classify_ua(Some(" cursor/1.0")), ClientTool::Cursor);
    }

    #[test]
    fn disjoint_no_shadowing() {
        let mut tokens: Vec<&str> = SIGNATURES.iter().map(|(t, _)| *t).collect();
        tokens.push("unattributed");
        for (i, a) in tokens.iter().enumerate() {
            for b in tokens.iter().skip(i + 1) {
                assert_ne!(a, b, "duplicate canonical token");
                assert!(
                    !a.starts_with(*b) && !b.starts_with(*a),
                    "shadowing: {a} vs {b}"
                );
            }
        }
        for (i, (_, sigs_a)) in SIGNATURES.iter().enumerate() {
            for (_, sigs_b) in SIGNATURES.iter().skip(i + 1) {
                for sa in *sigs_a {
                    for sb in *sigs_b {
                        assert!(
                            !sa.starts_with(sb) && !sb.starts_with(sa),
                            "signature shadowing: {sa} vs {sb}"
                        );
                    }
                }
            }
        }
    }

    #[test]
    fn case_insensitive() {
        assert_eq!(classify_ua(Some("CLAUDE-CLI/2.0")), ClientTool::ClaudeCode);
        assert_eq!(classify_ua(Some("Claude-Code/1.0")), ClientTool::ClaudeCode);
        assert_eq!(classify_ua(Some("GEMINI-CLI/0.8")), ClientTool::GeminiCli);
        assert_eq!(classify_ua(Some("RoO-Code/3.0")), ClientTool::RooCode);
        assert_eq!(classify_ua(Some("QWEN-CODE/1.0")), ClientTool::QwenCode);
    }

    #[test]
    fn malformed_header_fallback() {
        let mut h = HeaderMap::new();
        h.insert(
            header::USER_AGENT,
            axum::http::HeaderValue::from_bytes(&[0xff, 0xfe]).unwrap(),
        );
        assert_eq!(classify(&h), ClientTool::Unattributed);
    }

    #[test]
    fn missing_or_empty_fallback() {
        assert_eq!(classify(&HeaderMap::new()), ClientTool::Unattributed);
        assert_eq!(classify_ua(None), ClientTool::Unattributed);
        assert_eq!(classify_ua(Some("")), ClientTool::Unattributed);
        assert_eq!(classify_ua(Some("   ")), ClientTool::Unattributed);
    }

#[test]
fn signature_table_order_wins_over_position() {
    let ua = "claude-cli/1.0 codex/1.0";
    assert_eq!(classify_ua(Some(ua)), ClientTool::ClaudeCode);
    let ua2 = "codex/1.0 claude-cli/2.0";
    assert_eq!(classify_ua(Some(ua2)), ClientTool::ClaudeCode);
}

    #[test]
    fn match_after_open_paren_boundary() {
        assert_eq!(
            classify_ua(Some("agent (codex_http/1.0)")),
            ClientTool::Unattributed
        );
        assert_eq!(
            classify_ua(Some("agent (codex/1.0)")),
            ClientTool::Codex
        );
        assert_eq!(
            classify_ua(Some("agent (cursor/1.0)")),
            ClientTool::Cursor
        );
    }

    #[test]
    fn slash_boundary_after_product_name() {
        assert_eq!(classify_ua(Some("Mozilla/5.0 cursor/1.0")), ClientTool::Cursor);
        assert_eq!(
            classify_ua(Some("libcodex/1.0")),
            ClientTool::Unattributed
        );
        assert_eq!(
            classify_ua(Some("somecodex/1.0")),
            ClientTool::Unattributed
        );
    }

#[test]
fn later_occurrence_can_match() {
    for (ua, expected) in [
        ("my-opencode/1.0 opencode/2.0", ClientTool::OpenCode),
        ("xcodex_cli/1.0 codex/2.0", ClientTool::Codex),
    ] {
        assert_eq!(classify_ua(Some(ua)), expected, "ua={ua}");
    }
}

    #[test]
    fn as_str_round_trip() {
        for (token, _) in SIGNATURES {
            let tool = from_token(token);
            assert_eq!(tool.as_str(), *token);
        }
        assert_eq!(ClientTool::Unattributed.as_str(), "unattributed");
    }
}
