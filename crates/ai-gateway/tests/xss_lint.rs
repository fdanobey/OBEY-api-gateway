//! XSS lint regression test for the embedded SPAs
//! (spec: black-october-security-hardening, task 12, Req 5).
//!
//! Scans the same static assets that rust-embed embeds into the binary and
//! fails when an `innerHTML`/`outerHTML`/`insertAdjacentHTML` sink references
//! a dynamic value (bare variable or concatenation with an identifier, or a
//! template-literal interpolation) unless every dynamic operand on the sink
//! line is wrapped in an `escapeHtml(`/`esc(` call, or the line carries a
//! reviewed `xss-audit:` marker (on the sink line or the line directly above
//! it).

static ADMIN_INDEX: &str = include_str!("../src/admin/static/index.html");
static DASHBOARD_INDEX: &str = include_str!("../src/dashboard/static/index.html");

/// Remove quoted string literals (including their contents) so remaining
/// identifier characters prove a dynamic reference on the RHS.
fn strip_string_literals(s: &str) -> String {
    let mut out = String::new();
    let mut chars = s.chars().peekable();
    while let Some(c) = chars.next() {
        if c == '\'' || c == '"' || c == '`' {
            let quote = c;
            while let Some(nc) = chars.next() {
                if nc == '\\' {
                    chars.next();
                    continue;
                }
                if nc == quote {
                    break;
                }
            }
        } else {
            out.push(c);
        }
    }
    out
}

/// Strip comments (`// ...` and `/* ... */`) so an escape call mentioned in
/// a trailing comment cannot satisfy the escape requirement, and so
/// commented-out sink lines are never flagged.
fn strip_comments(s: &str) -> String {
    let mut out = String::new();
    let mut chars = s.chars().peekable();
    let mut in_line_comment = false;
    let mut in_block_comment = false;
    while let Some(c) = chars.next() {
        match (c, in_line_comment, in_block_comment) {
            ('\n', true, _) => {
                in_line_comment = false;
                out.push(c);
            }
            (_, true, _) | (_, _, true) => {
                if in_block_comment && c == '*' && chars.peek() == Some(&'/') {
                    chars.next();
                    in_block_comment = false;
                }
            }
            ('/', false, false) if chars.peek() == Some(&'/') => {
                chars.next();
                in_line_comment = true;
            }
            ('/', false, false) if chars.peek() == Some(&'*') => {
                chars.next();
                in_block_comment = true;
            }
            (c, false, false) => out.push(c),
        }
    }
    out
}

/// True when every non-literal (dynamic) operand inside `expr` is wrapped in
/// an escape call (`escapeHtml(...)` / `escHtml(...)` / `esc(...)`).
///
/// S10 tightening: the old heuristic passed a line when `esc(`/`escapeHtml(`
/// appeared ANYWHERE on it, so `desc(x)` or another substring spoof could
/// silence a sink that also concatenated a raw variable. This operand-level
/// check requires each dynamic operand to be an escape call by itself:
/// escape calls are matched at identifier boundaries and their whole
/// balanced-paren argument list is removed; any surviving identifier
/// character is an unescaped dynamic operand.
fn dynamic_operands_all_escaped(expr: &str) -> bool {
    let stripped = strip_string_literals(expr);
    let chars: Vec<char> = stripped.chars().collect();
    let is_word = |c: char| c.is_ascii_alphanumeric() || c == '_' || c == '$';
    let escapes = ["escapeHtml", "escHtml", "esc"];
    let mut out = String::new();
    let mut i = 0;
    while i < chars.len() {
        let mut matched = None;
        for name in escapes {
            let n = name.chars().count();
            let starts = chars[i..].len() >= n + 1
                && chars[i..i + n].iter().collect::<String>() == name
                && chars[i + n] == '('
                && (i == 0 || !is_word(chars[i - 1]));
            if starts {
                matched = Some(n);
                break;
            }
        }
        if let Some(n) = matched {
            // Skip the escape call's balanced argument list wholesale.
            let mut depth = 0usize;
            let mut j = i + n;
            while j < chars.len() {
                if chars[j] == '(' {
                    depth += 1;
                } else if chars[j] == ')' {
                    depth -= 1;
                    if depth == 0 {
                        break;
                    }
                }
                j += 1;
            }
            i = (j + 1).min(chars.len());
            continue;
        }
        out.push(chars[i]);
        i += 1;
    }
    !out.chars()
        .any(|c| c.is_ascii_alphabetic() || c == '_' || c == '$')
}

fn lint_asset(name: &str, source: &str) -> Vec<String> {
    let mut violations = Vec::new();
    let lines: Vec<&str> = source.lines().collect();
    for (i, line) in lines.iter().enumerate() {
        let trimmed = line.trim_start();
        if trimmed.starts_with("//") || trimmed.starts_with("/*") || trimmed.starts_with('*') {
            continue;
        }
        let is_assignment_sink = line.contains(".innerHTML=")
            || line.contains(".innerHTML =")
            || line.contains(".outerHTML=")
            || line.contains(".outerHTML =");
        let is_call_sink = line.contains("insertAdjacentHTML(");
        if !is_assignment_sink && !is_call_sink {
            continue;
        }
        // Reviewed allowlist: marker comment on the sink line or right above.
        if line.contains("xss-audit:") || (i > 0 && lines[i - 1].contains("xss-audit:")) {
            continue;
        }
        // S10: escape heuristic is operand-level (see
        // `dynamic_operands_all_escaped`); a bare `esc(` substring anywhere
        // on the line no longer silences a sink.
        // RHS analysis: assignment RHS or insertAdjacentHTML(...) arguments.
        let code = strip_comments(line);
        let rhs: &str = if is_assignment_sink {
            let pos = code
                .find(".innerHTML")
                .or_else(|| code.find(".outerHTML"))
                .unwrap_or(0);
            match code[pos..].find('=') {
                Some(eq) => &code[pos + eq + 1..],
                None => "",
            }
        } else {
            match code.find("insertAdjacentHTML(") {
                Some(pos) => &code[pos + "insertAdjacentHTML(".len()..],
                None => "",
            }
        };
        if rhs.contains("${") {
            let literal_interpolation_only = {
                // Template literal with only literal/escaped content still
                // interpolates dynamically unless each `${...}` wraps an
                // escape call; check the template as a whole.
                let template = rhs.trim().strip_suffix(';').unwrap_or(rhs.trim());
                template.starts_with('`')
                    && template.ends_with('`')
                    && template.matches("${").count() > 0
                    && template_interpolations_all_escaped(template)
            };
            if !literal_interpolation_only {
                violations.push(format!(
                    "{name}:{}: template interpolation into HTML sink: {}",
                    i + 1,
                    line.trim()
                ));
                continue;
            }
        }
        if !dynamic_operands_all_escaped(rhs) {
            violations.push(format!(
                "{name}:{}: dynamic value into HTML sink without escaping or review marker: {}",
                i + 1,
                line.trim()
            ));
        }
    }
    violations
}

/// Every `${...}` interpolation inside a template literal must itself be an
/// escape call (`esc(...)`/`escapeHtml(...)`) for the sink to pass.
fn template_interpolations_all_escaped(template: &str) -> bool {
    let mut rest = template;
    while let Some(start) = rest.find("${") {
        let after = &rest[start + 2..];
        let Some(end) = after.find('}') else {
            return false;
        };
        let inner = &after[..end];
        let inner_trim = inner.trim();
        let escaped = (inner_trim.starts_with("esc(") || inner_trim.starts_with("escapeHtml("))
            && inner_trim.ends_with(')');
        if !escaped {
            return false;
        }
        rest = &after[end + 1..];
    }
    true
}

#[test]
fn xss_lint_embedded_uis_have_no_unescaped_dynamic_html_sinks() {
    let mut violations = lint_asset("admin/static/index.html", ADMIN_INDEX);
    violations.extend(lint_asset("dashboard/static/index.html", DASHBOARD_INDEX));
    assert!(
        violations.is_empty(),
        "unescaped dynamic HTML sinks found (escape the value, use DOM \
construction, or add a reviewed `xss-audit:` marker with \
justification):\n{}",
        violations.join("\n")
    );
}

// ---------------------------------------------------------------------------
// Fail-case coverage (task T1): the lint must RETURN violations for the
// canonical bypass shapes; presence of `esc(`/`escapeHtml(` text anywhere on
// the line must not silence a sink (S10).
// ---------------------------------------------------------------------------

fn count_violations(source: &str) -> Vec<String> {
    lint_asset("fixture.html", source)
}

#[test]
fn xss_lint_flags_bare_variable_inner_html_sink() {
    let violations = count_violations("el.innerHTML = userData;");
    assert_eq!(
        violations.len(),
        1,
        "bare variable sink must flag: {violations:?}"
    );
    assert!(
        violations[0].contains("fixture.html:1"),
        "{}",
        violations[0]
    );
}

#[test]
fn xss_lint_flags_template_interpolation_into_inner_html() {
    let violations = count_violations("el.innerHTML = `<b>${userName}</b>`;");
    assert_eq!(
        violations.len(),
        1,
        "template interpolation sink must flag: {violations:?}"
    );
    assert!(
        violations[0].contains("template interpolation"),
        "{}",
        violations[0]
    );
}

#[test]
fn xss_lint_flags_unescaped_operand_despite_escape_call_elsewhere() {
    // `desc(` spoofing (S10): substring `esc(` inside another identifier plus
    // a raw concatenated operand must NOT silence the sink.
    let spoof = "el.innerHTML = '<b>' + desc(row.name) + '</b>' + rawHtml;";
    let violations = count_violations(spoof);
    assert_eq!(
        violations.len(),
        1,
        "desc( substring spoof must not silence the sink: {violations:?}"
    );
    // Same shape but with a real trailing escape call and one raw operand:
    // still a violation (EVERY dynamic operand must be escaped).
    let mixed = "el.innerHTML = '<i>' + esc(safe) + '</i>' + rawHtml;";
    let violations = count_violations(mixed);
    assert_eq!(
        violations.len(),
        1,
        "partially escaped concatenation must flag: {violations:?}"
    );
}

#[test]
fn xss_lint_flags_insert_adjacent_html_variant() {
    let violations =
        count_violations("el.insertAdjacentHTML('beforeend', '<b>' + userInput + '</b>');");
    assert_eq!(
        violations.len(),
        1,
        "insertAdjacentHTML dynamic argument must flag: {violations:?}"
    );
    assert!(
        violations[0].contains("fixture.html:1"),
        "{}",
        violations[0]
    );
}

#[test]
fn xss_lint_accepts_fully_escaped_and_static_lines() {
    // Fully-escaped concatenation passes.
    assert!(count_violations("el.innerHTML = '<b>' + escapeHtml(userName) + '</b>';").is_empty());
    // Fully-escaped template interpolation passes.
    assert!(count_violations("el.innerHTML = `<b>${esc(userName)}</b>`;").is_empty());
    // Static literal only passes.
    assert!(count_violations("el.innerHTML = '<span>ok</span>';").is_empty());
    // Trailing comment is stripped: an escape call in a comment alone must
    // NOT silence a raw operand.
    assert_eq!(
        count_violations("el.innerHTML = raw; // esc(x)").len(),
        1,
        "escape call in a comment must not silence a sink"
    );
    // Reviewed marker still suppresses.
    assert!(count_violations("el.innerHTML = raw; /* xss-audit: reviewed */").is_empty());
}
