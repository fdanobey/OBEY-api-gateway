use uuid::Uuid;

pub type TraceId = [u8; 16];
pub type SpanId = [u8; 8];

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct W3cContext {
    pub version: u8,
    pub trace_id: TraceId,
    pub parent_span_id: SpanId,
    pub trace_flags: u8,
}

impl W3cContext {
    pub fn sampled(&self) -> bool {
        self.trace_flags & 0x01 == 0x01
    }

    pub fn trace_id_hex(&self) -> String {
        to_hex(&self.trace_id)
    }

    pub fn parent_span_id_hex(&self) -> String {
        to_hex(&self.parent_span_id)
    }
}

pub fn new_trace_id() -> TraceId {
    *Uuid::new_v4().as_bytes()
}

pub fn parse_traceparent(value: &str) -> Option<W3cContext> {
    let b = value.trim().as_bytes();
    // version(2) '-' trace-id(32) '-' parent-id(16) '-' flags(2) = 55
    if b.len() < 55 {
        return None;
    }
    let version = hex_byte(&b[..2])?;
    if version == 0xff {
        return None;
    }
    if b[2] != b'-' {
        return None;
    }
    let trace_id = hex_field::<16>(&b[3..35])?;
    if b[35] != b'-' {
        return None;
    }
    let parent_span_id = hex_field::<8>(&b[36..52])?;
    if b[52] != b'-' {
        return None;
    }
    let trace_flags = hex_byte(&b[53..55])?;
    let extra = &b[55..];
    if version == 0x00 {
        if !extra.is_empty() {
            return None;
        }
    } else if !extra.is_empty() && extra[0] != b'-' {
        return None;
    }
    Some(W3cContext {
        version,
        trace_id,
        parent_span_id,
        trace_flags,
    })
}

fn hex_value(c: u8) -> Option<u8> {
    match c {
        b'0'..=b'9' => Some(c - b'0'),
        b'a'..=b'f' => Some(c - b'a' + 10),
        b'A'..=b'F' => Some(c - b'A' + 10),
        _ => None,
    }
}

fn hex_byte(b: &[u8]) -> Option<u8> {
    if b.len() != 2 {
        return None;
    }
    Some(hex_value(b[0])? << 4 | hex_value(b[1])?)
}

fn hex_field<const N: usize>(b: &[u8]) -> Option<[u8; N]> {
    if b.len() != N * 2 {
        return None;
    }
    let mut out = [0u8; N];
    for (i, chunk) in b.chunks_exact(2).enumerate() {
        out[i] = hex_value(chunk[0])? << 4 | hex_value(chunk[1])?;
    }
    if out.iter().all(|&b| b == 0) {
        return None;
    }
    Some(out)
}

fn to_hex(bytes: &[u8]) -> String {
    let mut s = String::with_capacity(bytes.len() * 2);
    for b in bytes {
        s.push(char::from_digit((b >> 4) as u32, 16).unwrap());
        s.push(char::from_digit((b & 0x0f) as u32, 16).unwrap());
    }
    s
}

#[cfg(test)]
mod tests {
    use super::*;

    fn ctx(value: &str) -> W3cContext {
        parse_traceparent(value).expect("valid traceparent")
    }

    #[test]
    fn parses_valid_version_00_sampled() {
        let c = ctx("00-4bf92f3577b34da6a3ce929d0e0e4736-00f067aa0ba902b7-01");
        assert_eq!(c.version, 0x00);
        assert_eq!(
            c.trace_id_hex(),
            "4bf92f3577b34da6a3ce929d0e0e4736"
        );
        assert_eq!(c.parent_span_id_hex(), "00f067aa0ba902b7");
        assert_eq!(c.trace_flags, 0x01);
        assert!(c.sampled());
    }

    #[test]
    fn parses_valid_version_00_unsampled() {
        let c = ctx("00-4bf92f3577b34da6a3ce929d0e0e4736-00f067aa0ba902b7-00");
        assert_eq!(c.trace_flags, 0x00);
        assert!(!c.sampled());
    }

    #[test]
    fn parses_uppercase_hex() {
        let c = ctx("00-4BF92F3577B34DA6A3CE929D0E0E4736-00F067AA0BA902B7-01");
        assert_eq!(
            c.trace_id_hex(),
            "4bf92f3577b34da6a3ce929d0e0e4736"
        );
        assert_eq!(c.parent_span_id_hex(), "00f067aa0ba902b7");
    }

    #[test]
    fn mixed_case_hex_accepted() {
        let c = ctx("00-4bF92f3577B34dA6a3Ce929d0e0E4736-00f067Aa0Ba902B7-01");
        assert_eq!(c.trace_flags, 0x01);
    }

    #[test]
    fn parses_nonzero_version() {
        let c = ctx("01-4bf92f3577b34da6a3ce929d0e0e4736-00f067aa0ba902b7-01");
        assert_eq!(c.version, 0x01);
        assert!(c.sampled());
    }

    #[test]
    fn nonzero_version_allows_trailing_optional_fields() {
        let c = ctx("01-4bf92f3577b34da6a3ce929d0e0e4736-00f067aa0ba902b7-01-extra");
        assert_eq!(c.version, 0x01);
        assert_eq!(c.trace_flags, 0x01);
    }

    #[test]
    fn nonzero_version_allows_empty_optional_fields() {
        let c = ctx("cf-4bf92f3577b34da6a3ce929d0e0e4736-00f067aa0ba902b7-01-");
        assert_eq!(c.version, 0xcf);
    }

    #[test]
    fn trims_surrounding_whitespace() {
        let c = ctx("  00-4bf92f3577b34da6a3ce929d0e0e4736-00f067aa0ba902b7-01  ");
        assert!(c.sampled());
    }

    #[test]
    fn flags_only_least_significant_bit_governs_sampled() {
        let c = ctx("00-4bf92f3577b34da6a3ce929d0e0e4736-00f067aa0ba902b7-02");
        assert_eq!(c.trace_flags, 0x02);
        assert!(!c.sampled());
    }

    #[test]
    fn rejects_version_ff() {
        assert!(parse_traceparent("ff-4bf92f3577b34da6a3ce929d0e0e4736-00f067aa0ba902b7-01").is_none());
    }

    #[test]
    fn rejects_version_zero_with_trailing_junk() {
        assert!(
            parse_traceparent("00-4bf92f3577b34da6a3ce929d0e0e4736-00f067aa0ba902b7-01-extra").is_none()
        );
        assert!(
            parse_traceparent("00-4bf92f3577b34da6a3ce929d0e0e4736-00f067aa0ba902b7-01extra").is_none()
        );
    }

    #[test]
    fn rejects_too_short_trace_id() {
        assert!(
            parse_traceparent("00-4bf92f3577b34da6a3ce929d0e0e47-00f067aa0ba902b7-01").is_none()
        );
    }

    #[test]
    fn rejects_too_long_trace_id() {
        assert!(
            parse_traceparent(
                "00-4bf92f3577b34da6a3ce929d0e0e4736ff-00f067aa0ba902b7-01"
            )
            .is_none()
        );
    }

    #[test]
    fn rejects_wrong_parent_id_length() {
        assert!(
            parse_traceparent("00-4bf92f3577b34da6a3ce929d0e0e4736-00f067aa0ba902b-01").is_none()
        );
        assert!(
            parse_traceparent("00-4bf92f3577b34da6a3ce929d0e0e4736-0f067aa0ba902b7-01").is_none()
        );
    }

    #[test]
    fn rejects_non_hex_characters() {
        assert!(
            parse_traceparent("00-4bf92f3577b34da6a3ce929d0e0e473g-00f067aa0ba902b7-01").is_none()
        );
        assert!(
            parse_traceparent("0g-4bf92f3577b34da6a3ce929d0e0e4736-00f067aa0ba902b7-01").is_none()
        );
        assert!(
            parse_traceparent("00-4bf92f3577b34da6a3ce929d0e0e4736-00f067aa0ba902b7-0g").is_none()
        );
    }

    #[test]
    fn rejects_all_zero_trace_id() {
        assert!(parse_traceparent("00-00000000000000000000000000000000-00f067aa0ba902b7-01").is_none());
    }

    #[test]
    fn rejects_all_zero_parent_id() {
        assert!(parse_traceparent("00-4bf92f3577b34da6a3ce929d0e0e4736-0000000000000000-01").is_none());
    }

    #[test]
    fn rejects_short_and_garbage_input() {
        assert!(parse_traceparent("").is_none());
        assert!(parse_traceparent("-").is_none());
        assert!(parse_traceparent("garbage").is_none());
        assert!(parse_traceparent("004bf92f3577b34da6a3ce929d0e0e473600f067aa0ba902b701").is_none());
    }

    #[test]
    fn new_trace_id_is_128_bit_nonzero_and_unique() {
        let a = new_trace_id();
        let b = new_trace_id();
        assert_eq!(a.len(), 16);
        assert!(a.iter().any(|&x| x != 0));
        assert_ne!(a, b);
        assert_eq!(to_hex(&a).len(), 32);
        assert!(to_hex(&a).chars().all(|c| c.is_ascii_hexdigit() && !c.is_ascii_uppercase()));
    }
}
