//! CSV rendering of cost reports (spec: billing-grade-cost-reporting,
//! Task 5; Requirement 4).
//!
//! Pure function over the report types: no I/O, no HTTP. Deterministic:
//! constant header row, rows in the order they appear in the report, CRLF
//! line endings, UTF-8. `cost_usd` is rendered through
//! [`crate::cost_reporting::ReportRow::cost_usd_string`] (integer math,
//! exactly 6 decimal places) — never float formatting. Dimension values are
//! arbitrary provider/model strings, so every field is escaped per RFC 4180
//! (quote on comma, quote, CR, LF; double embedded quotes) and dimension
//! cells starting with a spreadsheet-formula character (`=`, `+`, `-`, `@`,
//! TAB, CR) are neutralized with a leading `'` — the model dimension is
//! client-controlled, so an unguarded cell could execute as a formula when
//! the CSV is opened in Excel/Sheets. Numeric fields are rendered from
//! integers and stay untouched (a leading `-` there is a benign negative
//! number, and the 6-decimal cost contract must not gain a prefix).

use crate::cost_reporting::{CostReport, ReportRow};

const CSV_HEADER: &str = "dimension,requests,input_tokens,output_tokens,cache_read_tokens,cache_creation_tokens,reasoning_tokens,cost_usd,cost_unpriced_requests";

/// Characters that make spreadsheet applications interpret a cell as a
/// formula when they lead the value.
const FORMULA_PREFIXES: [char; 6] = ['=', '+', '-', '@', '\t', '\r'];

/// Quote a single CSV field per RFC 4180 and neutralize formula injection:
/// fields containing a comma, double quote, CR, or LF are wrapped in double
/// quotes with embedded quotes doubled, and fields starting with a formula
/// trigger character get a `'` prefix BEFORE quoting so spreadsheets read
/// them as text. Used for the dimension field only — numeric fields are
/// rendered from integers and never pass through here.
fn escape_field(field: &str) -> String {
    let guarded = match field.chars().next() {
        Some(first) if FORMULA_PREFIXES.contains(&first) => {
            format!("'{field}")
        }
        _ => field.to_string(),
    };
    if guarded.contains(',') || guarded.contains('"') || guarded.contains('\r')
        || guarded.contains('\n')
    {
        format!("\"{}\"", guarded.replace('"', "\"\""))
    } else {
        guarded
    }
}

/// Render one report row as a CSV line (without the trailing CRLF).
fn render_row(row: &ReportRow) -> String {
    [
        escape_field(&row.dimension),
        row.requests.to_string(),
        row.input_tokens.to_string(),
        row.output_tokens.to_string(),
        row.cache_read_tokens.to_string(),
        row.cache_creation_tokens.to_string(),
        row.reasoning_tokens.to_string(),
        row.cost_usd_string(),
        row.cost_unpriced_requests.to_string(),
    ]
    .join(",")
}

/// Render a cost report as CSV: a constant header row followed by one row
/// per result row. CRLF line endings throughout; the output always ends
/// with a CRLF. An empty report yields the header row only.
pub fn render_report_csv(report: &CostReport) -> String {
    let mut out = String::with_capacity(CSV_HEADER.len() + 3 + report.rows.len() * 96);
    out.push_str(CSV_HEADER);
    out.push_str("\r\n");
    for row in &report.rows {
        out.push_str(&render_row(row));
        out.push_str("\r\n");
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::cost_reporting::GroupBy;

    fn row(dimension: &str, cost_micro_usd: i64) -> ReportRow {
        ReportRow {
            dimension: dimension.to_string(),
            window_start: 0,
            window_end: 86_400,
            requests: 3,
            input_tokens: 1_500,
            output_tokens: 2_500,
            cache_read_tokens: 100,
            cache_creation_tokens: 40,
            reasoning_tokens: 700,
        cost_micro_usd,
        cost_unpriced_requests: 2,
    }
}

    fn report(rows: Vec<ReportRow>) -> CostReport {
        CostReport::new(GroupBy::Provider, rows, 0, 86_400)
    }

    #[test]
    fn empty_report_renders_header_row_only() {
        let csv = render_report_csv(&report(Vec::new()));
        assert_eq!(csv, format!("{CSV_HEADER}\r\n"));
        assert!(csv.ends_with("\r\n"));
    }

    #[test]
    fn basic_row_renders_all_columns_in_header_order() {
        let csv = render_report_csv(&report(vec![row("openai", 1_234_567)]));
        let lines: Vec<&str> = csv.split("\r\n").collect();
        assert_eq!(
            lines[0],
            "dimension,requests,input_tokens,output_tokens,cache_read_tokens,cache_creation_tokens,reasoning_tokens,cost_usd,cost_unpriced_requests"
        );
        assert_eq!(
            lines[1],
            "openai,3,1500,2500,100,40,700,1.234567,2"
        );
        assert_eq!(lines[2], "");
        assert!(csv.ends_with("\r\n"));
    }

    #[test]
    fn dimension_with_comma_is_quoted() {
        let csv = render_report_csv(&report(vec![row("acme, inc.", 0)]));
        assert!(csv.contains("\"acme, inc.\",3,1500"));
    }

    #[test]
    fn dimension_with_double_quotes_doubles_them() {
        let csv = render_report_csv(&report(vec![row("say \"hi\"", 0)]));
        assert!(csv.contains("\"say \"\"hi\"\"\",3,1500"));
    }

    #[test]
    fn dimension_with_newline_is_quoted_and_crlf_terminates_lines() {
        let csv = render_report_csv(&report(vec![row("line1\nline2", 0)]));
        // Embedded bare LF stays inside the quoted field; record separator
        // stays CRLF, so the rendered output still splits into exactly
        // header + one quoted field chunk + trailing terminator.
        assert!(csv.contains("\"line1\nline2\""));
        let records: Vec<&str> = csv.split("\r\n").collect();
        assert_eq!(records.len(), 3); // header, data record, trailing empty
    }

    #[test]
    fn dimension_with_crlf_is_quoted() {
        let csv = render_report_csv(&report(vec![row("a\r\nb", 0)]));
        assert!(csv.contains("\"a\r\nb\""));
    }

    #[test]
    fn unicode_dimension_passes_through_unquoted() {
        let csv = render_report_csv(&report(vec![row("prövider-日本", 0)]));
        assert!(csv.contains("prövider-日本,3,"));
    }

#[test]
fn cost_usd_renders_with_exactly_six_decimals() {
    // The full six-decimal case table lives in mod.rs
    // (`micro_usd_string_has_exactly_six_decimals`); here one canonical
    // non-trivial value proves the formatting reaches the rendered CSV
    // line through `cost_usd_string`.
    let csv = render_report_csv(&report(vec![row("d", 1_234_567)]));
    let data_line = csv.split("\r\n").nth(1).unwrap();
    let cost = data_line.rsplit(',').nth(1).unwrap();
    assert_eq!(cost, "1.234567", "line: {data_line}");
}

#[test]
fn multiple_rows_render_in_report_order() {
    let rows = vec![row("b", 500), row("a", 9_000)];
    let rep = CostReport::new(GroupBy::Provider, rows, 0, 86_400);
    // Provider grouping sorts cost-descending: a (0.009) then b (0.0005).
    let csv = render_report_csv(&rep);
    let a_pos = csv.find("\r\na,").unwrap();
    let b_pos = csv.find("\r\nb,").unwrap();
    assert!(a_pos < b_pos);
}

#[test]
fn formula_trigger_prefixes_are_neutralized_with_leading_quote() {
    // Each of the six trigger characters must gain a `'` prefix BEFORE
    // RFC 4180 quoting so spreadsheets read the cell as text. Values
    // without RFC 4180 specials stay unquoted; the `'` guard still leads.
    for dangerous in ["=1+1", "+cmd", "-2+3", "@SUM(A1)", "\tinjected", "\rinjected"] {
        let csv = render_report_csv(&report(vec![row(dangerous, 0)]));
        // The `'` guard must lead the field value — inside the quotes
        // when RFC 4180 quoting also applies (embedded comma/quote/CR/LF).
        let rendered = format!("'{dangerous}");
        assert!(
            csv.contains(&rendered) || csv.contains(&format!("\"'{dangerous}\"")),
            "dimension {dangerous:?} must render with a leading quote guard: {csv:?}"
        );
        // And the guard must come BEFORE any RFC 4180 opening quote.
        let guard_pos = csv.find('\'').unwrap();
        let first_field_after_header = csv.split("\r\n").nth(1).unwrap();
        assert!(
            first_field_after_header.starts_with('\'')
                || first_field_after_header.starts_with("\"'"),
            "the guarded value must lead the data record: {csv:?}"
        );
        let _ = guard_pos;
    }
}

#[test]
fn formula_prefix_guard_applies_before_quoting_not_after() {
    // A value starting with `=` AND containing a comma must render as
    // "'=...,<rest>" inside one quoted field — guard first, quote second.
    let csv = render_report_csv(&report(vec![row("=SUM(A1),2", 0)]));
    assert!(csv.contains("\"'=SUM(A1),2\""), "{csv:?}");
}

#[test]
fn safe_values_render_without_formula_guard() {
    // Numbers, plain provider/model names, and already-quoted text pass
    // through without the `'` prefix (the guard must not fire).
    for safe in ["openai", "gpt-4o-mini", "a=b", "x+y", "n@h", "tab\tinside", "key_42"] {
        let csv = render_report_csv(&report(vec![row(safe, 0)]));
        assert!(!csv.contains("'"), "safe value {safe:?} must stay unguarded: {csv:?}");
    }
}

#[test]
fn negative_cost_keeps_six_decimals_and_no_formula_prefix() {
    // The `-` guard applies to the dimension field only: a negative cost
    // renders as a plain negative number with exactly 6 decimals —
    // `-1.500000` — and must NOT gain a `'` prefix or lose precision.
    let rows = vec![row("d", -1_500_000)];
    let rep = CostReport::new(GroupBy::Provider, rows, 0, 86_400);
    let csv = render_report_csv(&rep);
    assert!(csv.contains(",-1.500000,"), "cost column stays exact: {csv:?}");
    assert!(!csv.contains("'-"), "numeric fields never get the formula prefix: {csv:?}");
}
}
