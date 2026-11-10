use flate2::write::GzEncoder;
use flate2::Compression;
use serde_json::{json, Map, Value};
use std::io::Write as _;
use std::time::{SystemTime, UNIX_EPOCH};

/// Default per-value byte bound for string span attributes (Req 3.6:
/// content attributes are size-bounded by the existing logging body-size
/// limit, `logging.max_body_size_bytes`, whose default is 10000).
pub const DEFAULT_MAX_ATTR_VALUE_BYTES: usize = 10_000;

const SCOPE_NAME: &str = "ai-gateway";
const SCOPE_VERSION: &str = env!("CARGO_PKG_VERSION");

/// Span kind subset used by the gateway, encoded on the wire as the OTLP
/// protobuf enum integer (opentelemetry-proto `SpanKind`: INTERNAL=1,
/// SERVER=2, CLIENT=3) per the OTLP JSON protobuf mapping.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum SpanKind {
    Client,
    Internal,
    Server,
}

impl SpanKind {
    fn otlp_code(self) -> u8 {
        match self {
            SpanKind::Internal => 1,
            SpanKind::Server => 2,
            SpanKind::Client => 3,
        }
    }
}

/// Span status, encoded as the OTLP protobuf enum integer
/// (`StatusCode`: OK=1, ERROR=2). Absent status means proto-default
/// `UNSET` and omits the field entirely.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum SpanStatus {
    Ok,
    Error,
}

impl SpanStatus {
    fn otlp_code(self) -> u8 {
        match self {
            SpanStatus::Ok => 1,
            SpanStatus::Error => 2,
        }
    }
}

/// Typed attribute value mapping to the OTLP JSON `AnyValue` one-of
/// (`stringValue`/`intValue`/`doubleValue`/`boolValue`). Integers encode
/// as decimal strings per the protobuf JSON mapping for 64-bit ints.
#[derive(Debug, Clone, PartialEq)]
pub enum AttrValue {
    String(String),
    Int(i64),
    Double(f64),
    Bool(bool),
}

/// OTLP resource attributes (Req 10.2: `service.name` configurable,
/// `service.version`).
#[derive(Debug, Clone, PartialEq)]
pub struct Resource {
    pub attributes: Vec<(String, AttrValue)>,
}

impl Resource {
    pub fn new(service_name: &str, service_version: &str) -> Self {
        Resource {
            attributes: vec![
                (
                    "service.name".to_string(),
                    AttrValue::String(service_name.to_string()),
                ),
                (
                    "service.version".to_string(),
                    AttrValue::String(service_version.to_string()),
                ),
            ],
        }
    }
}

/// In-memory span model handed to the encoder by span assembly (design
/// "SpanData" data model; never persisted).
#[derive(Debug, Clone, PartialEq)]
pub struct SpanData {
    pub trace_id: [u8; 16],
    pub span_id: [u8; 8],
    pub parent_span_id: Option<[u8; 8]>,
    pub name: String,
    pub kind: SpanKind,
    pub status: Option<SpanStatus>,
    pub start_time: SystemTime,
    pub end_time: SystemTime,
    pub attributes: Vec<(String, AttrValue)>,
}

/// Encode a span batch as gzip-compressed OTLP/HTTP JSON
/// (`ExportTraceServiceRequest`), bounding string attribute values by
/// [`DEFAULT_MAX_ATTR_VALUE_BYTES`].
pub fn encode_batch(spans: Vec<SpanData>, resource: &Resource) -> Vec<u8> {
    encode_batch_with_limit(spans, resource, DEFAULT_MAX_ATTR_VALUE_BYTES)
}

/// [`encode_batch`] with an explicit string-attribute byte bound.
pub fn encode_batch_with_limit(
    spans: Vec<SpanData>,
    resource: &Resource,
    max_attr_value_bytes: usize,
) -> Vec<u8> {
    let payload = Value::Object(build_export_request(&spans, resource, max_attr_value_bytes));
    gzip(payload.to_string().as_bytes())
}

/// Bound a string attribute value to `max_bytes` (measured on the UTF-8
/// bytes), appending an explicit truncation marker naming the number of
/// dropped bytes (Req 3.6). Values within the bound pass through
/// unchanged.
pub fn truncate_attr_value(value: &str, max_bytes: usize) -> String {
    if value.len() <= max_bytes {
        return value.to_string();
    }
    let mut cut = max_bytes;
    while cut > 0 && !value.is_char_boundary(cut) {
        cut -= 1;
    }
    let removed = value.len() - cut;
    format!("{}... [truncated by {} bytes]", &value[..cut], removed)
}

fn build_export_request(
    spans: &[SpanData],
    resource: &Resource,
    max_attr_value_bytes: usize,
) -> Map<String, Value> {
    let mut scope = Map::new();
    scope.insert("name".to_string(), Value::String(SCOPE_NAME.to_string()));
    scope.insert("version".to_string(), Value::String(SCOPE_VERSION.to_string()));

    let mut scope_spans = Map::new();
    scope_spans.insert("scope".to_string(), Value::Object(scope));
    scope_spans.insert(
        "spans".to_string(),
        Value::Array(
            spans
                .iter()
                .map(|span| encode_span(span, max_attr_value_bytes))
                .collect(),
        ),
    );

    let mut resource_spans = Map::new();
    resource_spans.insert("resource".to_string(), encode_resource(resource, max_attr_value_bytes));
    resource_spans.insert(
        "scopeSpans".to_string(),
        Value::Array(vec![Value::Object(scope_spans)]),
    );

    let mut root = Map::new();
    root.insert(
        "resourceSpans".to_string(),
        Value::Array(vec![Value::Object(resource_spans)]),
    );
    root
}

fn encode_resource(resource: &Resource, max_attr_value_bytes: usize) -> Value {
    let mut object = Map::new();
    if !resource.attributes.is_empty() {
        object.insert(
            "attributes".to_string(),
            encode_attributes(&resource.attributes, max_attr_value_bytes),
        );
    }
    Value::Object(object)
}

fn encode_span(span: &SpanData, max_attr_value_bytes: usize) -> Value {
    let mut object = Map::new();
    object.insert("traceId".to_string(), Value::String(to_hex(&span.trace_id)));
    object.insert("spanId".to_string(), Value::String(to_hex(&span.span_id)));
    if let Some(parent) = span.parent_span_id {
        object.insert("parentSpanId".to_string(), Value::String(to_hex(&parent)));
    }
    object.insert("name".to_string(), Value::String(span.name.clone()));
    object.insert("kind".to_string(), Value::from(span.kind.otlp_code()));
    object.insert(
        "startTimeUnixNano".to_string(),
        Value::String(unix_nanos(span.start_time).to_string()),
    );
    object.insert(
        "endTimeUnixNano".to_string(),
        Value::String(unix_nanos(span.end_time).to_string()),
    );
    if let Some(status) = span.status {
        let mut status_object = Map::new();
        status_object.insert("code".to_string(), Value::from(status.otlp_code()));
        object.insert("status".to_string(), Value::Object(status_object));
    }
    if !span.attributes.is_empty() {
        object.insert(
            "attributes".to_string(),
            encode_attributes(&span.attributes, max_attr_value_bytes),
        );
    }
    Value::Object(object)
}

fn encode_attributes(attributes: &[(String, AttrValue)], max_attr_value_bytes: usize) -> Value {
    Value::Array(
        attributes
            .iter()
            .map(|(key, value)| {
                json!({
                    "key": key,
                    "value": encode_attr_value(value, max_attr_value_bytes),
                })
            })
            .collect(),
    )
}

fn encode_attr_value(value: &AttrValue, max_attr_value_bytes: usize) -> Value {
    match value {
        AttrValue::String(s) => typed_value(
            "stringValue",
            Value::String(truncate_attr_value(s, max_attr_value_bytes)),
        ),
        AttrValue::Int(i) => typed_value("intValue", Value::String(i.to_string())),
        AttrValue::Double(d) => typed_value("doubleValue", Value::from(*d)),
        AttrValue::Bool(b) => typed_value("boolValue", Value::Bool(*b)),
    }
}

fn typed_value(tag: &str, value: Value) -> Value {
    let mut object = Map::new();
    object.insert(tag.to_string(), value);
    Value::Object(object)
}

fn unix_nanos(time: SystemTime) -> u128 {
    time.duration_since(UNIX_EPOCH)
        .map(|d| d.as_nanos())
        .unwrap_or_default()
}

fn to_hex(bytes: &[u8]) -> String {
    let mut s = String::with_capacity(bytes.len() * 2);
    for b in bytes {
        s.push(char::from_digit((b >> 4) as u32, 16).unwrap());
        s.push(char::from_digit((b & 0x0f) as u32, 16).unwrap());
    }
    s
}

fn gzip(data: &[u8]) -> Vec<u8> {
    let mut encoder = GzEncoder::new(Vec::new(), Compression::default());
    encoder.write_all(data).expect("in-memory gzip write");
    encoder.finish().expect("in-memory gzip finish")
}

#[cfg(test)]
mod tests {
    use super::*;
    use flate2::read::GzDecoder;
    use std::io::Read as _;
    use std::time::Duration;

    fn gunzip_json(bytes: &[u8]) -> Value {
        let mut decoder = GzDecoder::new(bytes);
        let mut out = String::new();
        decoder.read_to_string(&mut out).expect("valid gzip stream");
        serde_json::from_str(&out).expect("valid OTLP JSON")
    }

    fn from_hex<const N: usize>(value: &str) -> [u8; N] {
        let bytes = value.as_bytes();
        assert_eq!(bytes.len(), N * 2);
        let mut out = [0u8; N];
        for i in 0..N {
            let hi = (bytes[2 * i] as char).to_digit(16).unwrap() as u8;
            let lo = (bytes[2 * i + 1] as char).to_digit(16).unwrap() as u8;
            out[i] = hi << 4 | lo;
        }
        out
    }

    fn spans_of_batch() -> Vec<SpanData> {
        let start = UNIX_EPOCH + Duration::from_nanos(1_544_712_660_000_000_000);
        let end = start + Duration::from_millis(250);
        vec![
            SpanData {
                trace_id: from_hex("5B8EFFF798038103D269B633813FC60C"),
                span_id: from_hex("EEE19B7EC3C1B174"),
                parent_span_id: None,
                name: "POST /v1/chat/completions".to_string(),
                kind: SpanKind::Server,
                status: Some(SpanStatus::Ok),
                start_time: start,
                end_time: end,
                attributes: vec![
                    ("obey.request_id".to_string(), AttrValue::String("req-1".to_string())),
                    ("http.response.status_code".to_string(), AttrValue::Int(200)),
                    ("gen_ai.usage.cost".to_string(), AttrValue::Double(0.0125)),
                    ("streamed".to_string(), AttrValue::Bool(true)),
                ],
            },
            SpanData {
                trace_id: from_hex("5B8EFFF798038103D269B633813FC60C"),
                span_id: from_hex("EEE19B7EC3C1B175"),
                parent_span_id: Some(from_hex("EEE19B7EC3C1B174")),
                name: "chat gpt-4o".to_string(),
                kind: SpanKind::Client,
                status: None,
                start_time: start,
                end_time: end,
                attributes: vec![(
                    "gen_ai.usage.input_tokens".to_string(),
                    AttrValue::Int(42),
                )],
            },
        ]
    }

    fn attribute<'a>(value: &'a Value, key: &str) -> &'a Value {
        value["attributes"]
            .as_array()
            .expect("attributes array")
            .iter()
            .find(|attr| attr["key"] == key)
            .map(|attr| &attr["value"])
            .unwrap_or_else(|| panic!("attribute {key} present"))
    }

    fn is_decimal_string(value: &Value) -> bool {
        value
            .as_str()
            .map(|s| !s.is_empty() && s.bytes().all(|b| b.is_ascii_digit()))
            .unwrap_or(false)
    }

    #[test]
    fn round_trip_batch_structure() {
        let encoded = encode_batch(spans_of_batch(), &Resource::new("ai-gateway", env!("CARGO_PKG_VERSION")));
        let root = gunzip_json(&encoded);

        let resource_spans = root["resourceSpans"].as_array().expect("resourceSpans");
        assert_eq!(resource_spans.len(), 1);
        let rs = &resource_spans[0];

        let service_name = attribute(&rs["resource"], "service.name");
        assert_eq!(service_name, &json!({"stringValue": "ai-gateway"}));
        let service_version = attribute(&rs["resource"], "service.version");
        assert_eq!(
            service_version,
            &json!({"stringValue": env!("CARGO_PKG_VERSION")})
        );

        let scope_spans = rs["scopeSpans"].as_array().expect("scopeSpans");
        assert_eq!(scope_spans.len(), 1);
        assert_eq!(scope_spans[0]["scope"]["name"], "ai-gateway");

        let spans = scope_spans[0]["spans"].as_array().expect("spans");
        assert_eq!(spans.len(), 2);

        let (root_span, child_span) = (&spans[0], &spans[1]);
        for span in [root_span, child_span] {
            let trace_id = span["traceId"].as_str().expect("traceId string");
            assert_eq!(trace_id.len(), 32);
            assert!(trace_id.bytes().all(|b| b.is_ascii_hexdigit()));
            assert_eq!(span["spanId"].as_str().unwrap().len(), 16);
            assert!(is_decimal_string(&span["startTimeUnixNano"]));
            assert!(is_decimal_string(&span["endTimeUnixNano"]));
        }
        assert_eq!(root_span["traceId"], child_span["traceId"]);
        assert!(root_span.get("parentSpanId").is_none());
        assert_eq!(child_span["parentSpanId"], root_span["spanId"]);

        let start: u128 = root_span["startTimeUnixNano"]
            .as_str()
            .unwrap()
            .parse()
            .unwrap();
        let end: u128 = root_span["endTimeUnixNano"].as_str().unwrap().parse().unwrap();
        assert_eq!(end - start, 250_000_000);

        assert_eq!(
            attribute(root_span, "http.response.status_code"),
            &json!({"intValue": "200"})
        );
        assert_eq!(
            attribute(root_span, "gen_ai.usage.cost"),
            &json!({"doubleValue": 0.0125})
        );
        assert_eq!(
            attribute(root_span, "streamed"),
            &json!({"boolValue": true})
        );
        assert_eq!(
            attribute(child_span, "gen_ai.usage.input_tokens"),
            &json!({"intValue": "42"})
        );

        for attr in root_span["attributes"]
            .as_array()
            .unwrap()
            .iter()
            .chain(child_span["attributes"].as_array().unwrap())
        {
            let tags = attr["value"].as_object().expect("value object").len();
            assert_eq!(tags, 1, "exactly one value type tag");
        }
    }

    #[test]
    fn fixture_matches_otlp_spec_example() {
        let span = SpanData {
            trace_id: from_hex("5B8EFFF798038103D269B633813FC60C"),
            span_id: from_hex("EEE19B7EC3C1B174"),
            parent_span_id: Some(from_hex("EEE19B7EC3C1B173")),
            name: "I'm a server span".to_string(),
            kind: SpanKind::Server,
            status: None,
            start_time: UNIX_EPOCH + Duration::from_nanos(1_544_712_660_000_000_000),
            end_time: UNIX_EPOCH + Duration::from_nanos(1_544_712_661_000_000_000),
            attributes: vec![(
                "my.span.attr".to_string(),
                AttrValue::String("some value".to_string()),
            )],
        };
        let root = gunzip_json(&encode_batch(
            vec![span],
            &Resource::new("my.service", "1.0.0"),
        ));

        let rs = &root["resourceSpans"][0];
        assert_eq!(
            rs.as_object().unwrap().keys().collect::<Vec<_>>(),
            ["resource", "scopeSpans"]
        );
        assert_eq!(
            attribute(&rs["resource"], "service.name"),
            &json!({"stringValue": "my.service"})
        );

        let scope_spans = &rs["scopeSpans"][0];
        let span_json = &scope_spans["spans"][0];
        for key in [
            "traceId",
            "spanId",
            "parentSpanId",
            "name",
            "kind",
            "startTimeUnixNano",
            "endTimeUnixNano",
            "attributes",
        ] {
            assert!(span_json.get(key).is_some(), "key {key} present");
        }
        assert_eq!(
            span_json["traceId"].as_str().unwrap().to_ascii_lowercase(),
            "5b8efff798038103d269b633813fc60c"
        );
        assert_eq!(
            span_json["spanId"].as_str().unwrap().to_ascii_lowercase(),
            "eee19b7ec3c1b174"
        );
        assert_eq!(
            span_json["parentSpanId"].as_str().unwrap().to_ascii_lowercase(),
            "eee19b7ec3c1b173"
        );
        assert_eq!(span_json["name"], "I'm a server span");
        assert_eq!(span_json["kind"], 2);
        assert_eq!(span_json["startTimeUnixNano"], "1544712660000000000");
        assert_eq!(span_json["endTimeUnixNano"], "1544712661000000000");
        assert_eq!(
            span_json["attributes"][0],
            json!({"key": "my.span.attr", "value": {"stringValue": "some value"}})
        );
    }

    #[test]
    fn truncation_marker_bounds_string_values() {
        let oversized = "x".repeat(12_345);
        let bounded = truncate_attr_value(&oversized, 10_000);
        assert_eq!(
            bounded,
            format!("{}... [truncated by 2345 bytes]", "x".repeat(10_000))
        );
        assert!(bounded.len() <= 10_000 + "... [truncated by 2345 bytes]".len());

        let small = "short value";
        assert_eq!(truncate_attr_value(small, 10_000), small);

        let unicode = "é".repeat(4);
        let bounded = truncate_attr_value(&unicode, 5);
        assert_eq!(bounded, "éé... [truncated by 4 bytes]");

        let mut span = spans_of_batch();
        span[0].attributes.push((
            "gen_ai.prompt".to_string(),
            AttrValue::String("y".repeat(20_000)),
        ));
        let root = gunzip_json(&encode_batch(span, &Resource::new("ai-gateway", "0.7.10")));
        let prompt = &root["resourceSpans"][0]["scopeSpans"][0]["spans"][0];
        let value = attribute(prompt, "gen_ai.prompt")["stringValue"].as_str().unwrap();
        assert!(value.contains("[truncated by "));
        assert!(value.len() <= DEFAULT_MAX_ATTR_VALUE_BYTES + 40);
    }

    #[test]
    fn kind_and_status_encode_as_otlp_integers() {
        let make = |kind, status| SpanData {
            trace_id: from_hex("5B8EFFF798038103D269B633813FC60C"),
            span_id: from_hex("EEE19B7EC3C1B175"),
            parent_span_id: None,
            name: "s".to_string(),
            kind,
            status,
            start_time: UNIX_EPOCH,
            end_time: UNIX_EPOCH,
            attributes: Vec::new(),
        };
        let root = gunzip_json(&encode_batch(
            vec![
                make(SpanKind::Internal, None),
                make(SpanKind::Server, Some(SpanStatus::Ok)),
                make(SpanKind::Client, Some(SpanStatus::Error)),
            ],
            &Resource::new("ai-gateway", "0.7.10"),
        ));
        let spans = &root["resourceSpans"][0]["scopeSpans"][0]["spans"];
        assert_eq!(spans[0]["kind"], 1);
        assert!(spans[0].get("status").is_none());
        assert_eq!(spans[1]["kind"], 2);
        assert_eq!(spans[1]["status"]["code"], 1);
        assert_eq!(spans[2]["kind"], 3);
        assert_eq!(spans[2]["status"]["code"], 2);
    }

    #[test]
    fn output_is_gzip() {
        let encoded = encode_batch(spans_of_batch(), &Resource::new("ai-gateway", "0.7.10"));
        assert_eq!(&encoded[..2], &[0x1f, 0x8b]);
    }
}
