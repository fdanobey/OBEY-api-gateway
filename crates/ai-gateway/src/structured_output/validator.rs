use crate::models::openai::OpenAIRequest;
use jsonschema::{Draft, Validator};
use serde_json::Value;
use std::fmt;
use std::panic::{catch_unwind, AssertUnwindSafe};
use std::sync::Arc;
use std::time::Duration;

const VALIDATION_TIMEOUT: Duration = Duration::from_millis(100);
const MAX_SCHEMA_VIOLATIONS: usize = 50;
const MAX_ACTUAL_CHARS: usize = 200;
const MAX_EXPECTED_CHARS: usize = 200;
const MAX_COMPILE_MESSAGE_CHARS: usize = 500;

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum ChoiceValidationResult {
    Pass,
    JsonParseError {
        byte_offset: usize,
        expected: String,
    },
    SchemaViolations(Vec<SchemaViolation>),
    Skipped,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct SchemaViolation {
    pub path: String,
    pub expected: String,
    pub actual: String,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ChoiceValidationOutcome {
    pub result: ChoiceValidationResult,
    pub internal_skip: Option<ValidationInternalSkip>,
}

impl ChoiceValidationOutcome {
    fn completed(result: ChoiceValidationResult) -> Self {
        Self {
            result,
            internal_skip: None,
        }
    }

    fn internal_skip(reason: ValidationInternalSkip) -> Self {
        Self {
            result: ChoiceValidationResult::Skipped,
            internal_skip: Some(reason),
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ValidationInternalSkip {
    Timeout,
    WorkerPanicked,
    WorkerCancelled,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum SchemaCompileError {
    SchemaMustBeObject,
    SchemaMustNotBeEmpty,
    InvalidSchema { path: String, message: String },
    CompilerPanicked,
}

impl fmt::Display for SchemaCompileError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::SchemaMustBeObject => formatter.write_str("JSON Schema must be an object"),
            Self::SchemaMustNotBeEmpty => {
                formatter.write_str("JSON Schema object must not be empty")
            }
            Self::InvalidSchema { path, message } if path.is_empty() => {
                write!(formatter, "invalid JSON Schema: {message}")
            }
            Self::InvalidSchema { path, message } => {
                write!(formatter, "invalid JSON Schema at {path}: {message}")
            }
            Self::CompilerPanicked => formatter.write_str("JSON Schema compiler panicked"),
        }
    }
}

impl std::error::Error for SchemaCompileError {}

#[derive(Clone)]
pub struct SchemaContext {
    pub compiled: Arc<Validator>,
    pub raw_schema: Value,
    pub schema_char_len: usize,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum MalformedResponseFormat {
    ResponseFormatMustBeObject,
    ResponseFormatTypeMissing,
    ResponseFormatTypeInvalid,
    JsonSchemaMissing,
    JsonSchemaMustBeObject,
    SchemaMissing,
}

impl fmt::Display for MalformedResponseFormat {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::ResponseFormatMustBeObject => {
                formatter.write_str("response_format must be an object")
            }
            Self::ResponseFormatTypeMissing => {
                formatter.write_str("response_format.type is missing")
            }
            Self::ResponseFormatTypeInvalid => {
                formatter.write_str("response_format.type must be a string")
            }
            Self::JsonSchemaMissing => {
                formatter.write_str("response_format.json_schema is missing")
            }
            Self::JsonSchemaMustBeObject => {
                formatter.write_str("response_format.json_schema must be an object")
            }
            Self::SchemaMissing => {
                formatter.write_str("response_format.json_schema.schema is missing")
            }
        }
    }
}

#[derive(Clone)]
pub enum SchemaContextExtraction {
    NotApplicable,
    Malformed(MalformedResponseFormat),
    CompileFailed(SchemaCompileError),
    Ready(SchemaContext),
}

pub fn extract_schema_context(request: &OpenAIRequest) -> SchemaContextExtraction {
    let Some(response_format) = request.extra.get("response_format") else {
        return SchemaContextExtraction::NotApplicable;
    };
    let Some(response_format) = response_format.as_object() else {
        return SchemaContextExtraction::Malformed(
            MalformedResponseFormat::ResponseFormatMustBeObject,
        );
    };

    let Some(response_format_type) = response_format.get("type") else {
        return SchemaContextExtraction::Malformed(
            MalformedResponseFormat::ResponseFormatTypeMissing,
        );
    };
    let Some(response_format_type) = response_format_type.as_str() else {
        return SchemaContextExtraction::Malformed(
            MalformedResponseFormat::ResponseFormatTypeInvalid,
        );
    };
    if response_format_type != "json_schema" {
        return SchemaContextExtraction::NotApplicable;
    }

    let Some(json_schema) = response_format.get("json_schema") else {
        return SchemaContextExtraction::Malformed(MalformedResponseFormat::JsonSchemaMissing);
    };
    let Some(json_schema) = json_schema.as_object() else {
        return SchemaContextExtraction::Malformed(MalformedResponseFormat::JsonSchemaMustBeObject);
    };
    let Some(schema) = json_schema.get("schema") else {
        return SchemaContextExtraction::Malformed(MalformedResponseFormat::SchemaMissing);
    };

    let raw_schema = schema.clone();
    let schema_char_len = raw_schema.to_string().chars().count();
    match compile_schema(&raw_schema) {
        Ok(compiled) => SchemaContextExtraction::Ready(SchemaContext {
            compiled,
            raw_schema,
            schema_char_len,
        }),
        Err(error) => SchemaContextExtraction::CompileFailed(error),
    }
}

pub fn warn_schema_context_skipped(trace_id: &str) {
    tracing::warn!(trace_id, "structured output validation skipped");
}

pub fn compile_schema(schema: &Value) -> Result<Arc<Validator>, SchemaCompileError> {
    let object = schema
        .as_object()
        .ok_or(SchemaCompileError::SchemaMustBeObject)?;
    if object.is_empty() {
        return Err(SchemaCompileError::SchemaMustNotBeEmpty);
    }

    let compiled = catch_unwind(AssertUnwindSafe(|| {
        jsonschema::options()
            .with_draft(Draft::Draft202012)
            .build(schema)
    }))
    .map_err(|_| SchemaCompileError::CompilerPanicked)?;

    compiled.map(Arc::new).map_err(|error| {
        let path = error.schema_path().as_str().to_owned();
        let message = sanitize_and_truncate(&error.to_string(), MAX_COMPILE_MESSAGE_CHARS);
        SchemaCompileError::InvalidSchema { path, message }
    })
}

pub fn validate_response(validator: &Validator, content: &str) -> ChoiceValidationResult {
    if content.trim().is_empty() {
        return ChoiceValidationResult::Skipped;
    }

    let instance = match serde_json::from_str::<Value>(content) {
        Ok(instance) => instance,
        Err(error) => {
            return ChoiceValidationResult::JsonParseError {
                byte_offset: parse_byte_offset(content, &error),
                expected: parse_error_expected(&error),
            };
        }
    };

    let violations = validator
        .iter_errors(&instance)
        .take(MAX_SCHEMA_VIOLATIONS)
        .map(|error| SchemaViolation {
            path: error.instance_path().as_str().to_owned(),
            expected: expected_constraint(&error),
            actual: truncate_chars(&error.instance().to_string(), MAX_ACTUAL_CHARS),
        })
        .collect::<Vec<_>>();

    if violations.is_empty() {
        ChoiceValidationResult::Pass
    } else {
        ChoiceValidationResult::SchemaViolations(violations)
    }
}

pub async fn validate_response_async(
    validator: Arc<Validator>,
    content: String,
) -> ChoiceValidationOutcome {
    run_blocking_validation(
        move || validate_response(validator.as_ref(), &content),
        VALIDATION_TIMEOUT,
    )
    .await
}

async fn run_blocking_validation<F>(work: F, timeout: Duration) -> ChoiceValidationOutcome
where
    F: FnOnce() -> ChoiceValidationResult + Send + 'static,
{
    let mut task = tokio::task::spawn_blocking(work);

    match tokio::time::timeout(timeout, &mut task).await {
        Ok(Ok(result)) => ChoiceValidationOutcome::completed(result),
        Ok(Err(error)) if error.is_panic() => {
            ChoiceValidationOutcome::internal_skip(ValidationInternalSkip::WorkerPanicked)
        }
        Ok(Err(_)) => {
            ChoiceValidationOutcome::internal_skip(ValidationInternalSkip::WorkerCancelled)
        }
        Err(_) => ChoiceValidationOutcome::internal_skip(ValidationInternalSkip::Timeout),
    }
}

fn expected_constraint(error: &jsonschema::ValidationError<'_>) -> String {
    let keyword = error.kind().keyword();
    let description = error.masked().to_string();
    let constraint = description
        .strip_prefix("value ")
        .unwrap_or(description.as_str());
    sanitize_and_truncate(&format!("{keyword}: {constraint}"), MAX_EXPECTED_CHARS)
}

fn parse_byte_offset(content: &str, error: &serde_json::Error) -> usize {
    if error.line() == 0 {
        return 0;
    }

    let line_start = content
        .split_inclusive('\n')
        .take(error.line().saturating_sub(1))
        .map(str::len)
        .sum::<usize>();

    let byte_in_line = content
        .split('\n')
        .nth(error.line().saturating_sub(1))
        .map_or(0, |line| {
            let character_index = error.column().saturating_sub(1);
            line.char_indices()
                .nth(character_index)
                .map_or(line.len(), |(byte_index, _)| byte_index)
        });

    line_start.saturating_add(byte_in_line).min(content.len())
}

fn parse_error_expected(error: &serde_json::Error) -> String {
    let message = error.to_string();
    let without_location = message
        .split_once(" at line ")
        .map_or(message.as_str(), |(expected, _)| expected);
    sanitize_and_truncate(without_location, MAX_EXPECTED_CHARS)
}

fn sanitize_and_truncate(value: &str, max_chars: usize) -> String {
    let sanitized = value
        .chars()
        .map(|character| {
            if character.is_control() {
                ' '
            } else {
                character
            }
        })
        .collect::<String>();
    truncate_chars(&sanitized, max_chars)
}

fn truncate_chars(value: &str, max_chars: usize) -> String {
    let mut characters = value.chars();
    let prefix_length = max_chars.saturating_sub(1);
    let prefix = characters.by_ref().take(prefix_length).collect::<String>();

    if characters.next().is_none() {
        return prefix;
    }
    if max_chars == 0 {
        return String::new();
    }

    let mut truncated = prefix;
    truncated.push('…');
    truncated
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::models::openai::OpenAIRequest;
    use proptest::prelude::*;
    use serde_json::{json, Map};

    fn request_with_response_format(response_format: Option<Value>) -> OpenAIRequest {
        let mut extra = Map::new();
        if let Some(response_format) = response_format {
            extra.insert("response_format".to_owned(), response_format);
        }
        OpenAIRequest {
            model: "test-model".to_owned(),
            messages: Vec::new(),
            stream: false,
            temperature: None,
            max_tokens: None,
            extra,
        }
    }

    fn assert_not_applicable(response_format: Option<Value>) {
        assert!(matches!(
            extract_schema_context(&request_with_response_format(response_format)),
            SchemaContextExtraction::NotApplicable
        ));
    }

    fn assert_malformed(response_format: Value, expected: MalformedResponseFormat) {
        match extract_schema_context(&request_with_response_format(Some(response_format))) {
            SchemaContextExtraction::Malformed(actual) => assert_eq!(actual, expected),
            _ => panic!("expected malformed schema request"),
        }
    }

    fn object_validator(properties: Value) -> Arc<Validator> {
        compile_schema(&json!({
        "type": "object",
        "properties": properties,
        "additionalProperties": false
        }))
        .unwrap()
    }

    #[derive(Clone, Copy, Debug)]
    enum ActivationCase {
        MissingResponseFormat,
        NonObjectResponseFormat,
        MissingType,
        NonStringType,
        OtherType,
        MissingJsonSchema,
        NonObjectJsonSchema,
        MissingSchema,
        NonObjectSchema,
        EmptySchema,
        InvalidSchema,
        ValidSchema,
    }

    fn activation_case_strategy() -> impl Strategy<Value = ActivationCase> {
        prop_oneof![
            Just(ActivationCase::MissingResponseFormat),
            Just(ActivationCase::NonObjectResponseFormat),
            Just(ActivationCase::MissingType),
            Just(ActivationCase::NonStringType),
            Just(ActivationCase::OtherType),
            Just(ActivationCase::MissingJsonSchema),
            Just(ActivationCase::NonObjectJsonSchema),
            Just(ActivationCase::MissingSchema),
            Just(ActivationCase::NonObjectSchema),
            Just(ActivationCase::EmptySchema),
            Just(ActivationCase::InvalidSchema),
            Just(ActivationCase::ValidSchema),
        ]
    }

    fn activation_request(case: ActivationCase, noise: Map<String, Value>) -> OpenAIRequest {
        let response_format = match case {
            ActivationCase::MissingResponseFormat => None,
            ActivationCase::NonObjectResponseFormat => Some(json!(["json_schema"])),
            ActivationCase::MissingType => Some(Value::Object(noise)),
            ActivationCase::NonStringType => Some(json!({"type": true, "noise": noise})),
            ActivationCase::OtherType => Some(json!({
            "type": "json_object",
            "json_schema": {"schema": {"type": "object"}},
            "noise": noise
            })),
            ActivationCase::MissingJsonSchema => {
                Some(json!({"type": "json_schema", "noise": noise}))
            }
            ActivationCase::NonObjectJsonSchema => Some(json!({
            "type": "json_schema",
            "json_schema": ["schema"],
            "noise": noise
            })),
            ActivationCase::MissingSchema => Some(json!({
            "type": "json_schema",
            "json_schema": {"name": "generated"},
            "noise": noise
            })),
            ActivationCase::NonObjectSchema => Some(json!({
            "type": "json_schema",
            "json_schema": {"schema": true},
            "noise": noise
            })),
            ActivationCase::EmptySchema => Some(json!({
            "type": "json_schema",
            "json_schema": {"schema": {}},
            "noise": noise
            })),
            ActivationCase::InvalidSchema => Some(json!({
            "type": "json_schema",
            "json_schema": {"schema": {"type": 42}},
            "noise": noise
            })),
            ActivationCase::ValidSchema => Some(json!({
            "type": "json_schema",
            "json_schema": {
            "name": "generated",
            "schema": {
            "type": "object",
            "properties": {"value": {"type": "integer"}}
            }
            },
            "noise": noise
            })),
        };
        request_with_response_format(response_format)
    }

    fn json_value_strategy() -> impl Strategy<Value = Value> {
        let leaf = prop_oneof![
            Just(Value::Null),
            any::<bool>().prop_map(Value::Bool),
            (-1_000_000_i64..=1_000_000).prop_map(|number| json!(number)),
            "[ -~]{0,48}".prop_map(Value::String),
        ];
        leaf.prop_recursive(3, 48, 8, |inner| {
            prop_oneof![
                prop::collection::vec(inner.clone(), 0..6).prop_map(Value::Array),
                prop::collection::hash_map("[a-z][a-z0-9_]{0,12}", inner, 0..6)
                    .prop_map(|entries| Value::Object(entries.into_iter().collect()),),
            ]
        })
    }

    fn conforming_schema_and_value_strategy() -> impl Strategy<Value = (Value, Value)> {
        prop_oneof![
            (-1_000_000_i64..=1_000_000)
                .prop_map(|value| { (json!({"type": "integer"}), json!(value)) }),
            "[ -~]{0,96}"
                .prop_map(|value| { (json!({"type": "string", "maxLength": 96}), json!(value)) }),
            prop::collection::vec(any::<bool>(), 0..16).prop_map(|value| {
                (
                    json!({"type": "array", "items": {"type": "boolean"}}),
                    json!(value),
                )
            }),
            ("[a-z][a-z0-9_]{0,12}", -10_000_i64..=10_000).prop_map(|(key, value)| {
                (
                    json!({
                    "type": "object",
                    "properties": {key.clone(): {"type": "integer"}},
                    "required": [key.clone()],
                    "additionalProperties": false
                    }),
                    json!({key: value}),
                )
            }),
        ]
    }

    fn invalid_json_strategy() -> impl Strategy<Value = (String, usize)> {
        (
            "[a-zA-Z_][a-zA-Z0-9_]{0,20}",
            r#"[^"\\\p{C}]{0,24}"#,
            r#"[^"\\\p{C}]{0,24}"#,
            r#"[^"\\\p{C}]{0,24}"#,
        )
            .prop_map(|(key, first, second, suffix)| {
                let prefix = format!("{{\n  \"{key}\": [\"{first}\", \"{second}\", ");
                let byte_offset = prefix.len();
                (format!("{prefix}]{suffix}"), byte_offset)
            })
    }

    fn pointer_segment(value: &str) -> String {
        value.replace('~', "~0").replace('/', "~1")
    }

    // Feature: structured-output-validation, Property 1: Activation Predicate Correctness
    proptest! {
        #![proptest_config(ProptestConfig::with_cases(128))]

        #[test]
        fn prop_activation_predicate_correctness(
        case in activation_case_strategy(),
        noise in prop::collection::hash_map(
        "noise_[a-z][a-z0-9_]{0,8}",
        json_value_strategy(),
        0..6,
        ),
        ) {
        let request = activation_request(case, noise.into_iter().collect());
        let extraction = extract_schema_context(&request);

        prop_assert_eq!(
        matches!(extraction, SchemaContextExtraction::Ready(_)),
        matches!(case, ActivationCase::ValidSchema),
        "unexpected activation result for {:?}",
    case,
        );
        }
        }

    // Feature: structured-output-validation, Property 2: Valid Content Passes Through Unmodified
    proptest! {
    #![proptest_config(ProptestConfig::with_cases(128))]

    #[test]
    fn prop_valid_content_passes_through_unmodified(
    (schema, value) in conforming_schema_and_value_strategy(),
    ) {
    let validator = compile_schema(&schema).expect("generated schema must compile");
    let content = serde_json::to_string_pretty(&value).expect("generated value must serialize");
    let original = content.clone();

    prop_assert_eq!(
    validate_response(validator.as_ref(), &content),
    ChoiceValidationResult::Pass,
    );
    prop_assert_eq!(content.as_bytes(), original.as_bytes());
    }
    }

    // Feature: structured-output-validation, Property 3: Invalid JSON Produces Parse Error with Offset
    proptest! {
    #![proptest_config(ProptestConfig::with_cases(128))]

    #[test]
    fn prop_invalid_json_produces_parse_error_with_offset(
    (content, expected_offset) in invalid_json_strategy(),
    ) {
    let validator = compile_schema(&json!({"type": "object"}))
    .expect("static schema must compile");
    let result = validate_response(validator.as_ref(), &content);
    let ChoiceValidationResult::JsonParseError { byte_offset, expected } = result else {
    return Err(TestCaseError::fail("generated invalid JSON did not produce a parse error"));
    };

    prop_assert!(byte_offset <= content.len());
    prop_assert!(expected_offset <= content.len());
    prop_assert!(!expected.is_empty());
    prop_assert!(expected.chars().count() <= MAX_EXPECTED_CHARS);
    prop_assert!(!expected.chars().any(char::is_control));
    }
    }

    // Feature: structured-output-validation, Property 4: Schema Violations Are Bounded and Structured
    proptest! {
    #![proptest_config(ProptestConfig::with_cases(128))]

    #[test]
    fn prop_schema_violations_are_bounded_and_structured(
    field_count in 1_usize..=75,
    key_prefix in "[a-z][a-z0-9_]{0,8}",
    actual_character in prop_oneof![Just('界'), Just('é'), Just('x')],
    actual_length in 201_usize..260,
    ) {
    let properties = (0..field_count)
    .map(|index| {
    (
    format!("{key_prefix}_{index}"),
    json!({"type": "integer"}),
    )
    })
    .collect::<Map<_, _>>();
    let instance = properties
    .keys()
    .cloned()
    .map(|key| (key, Value::String(actual_character.to_string().repeat(actual_length))))
    .collect::<Map<_, _>>();
    let schema = json!({
    "type": "object",
    "properties": properties,
    "required": properties.keys().collect::<Vec<_>>(),
    "additionalProperties": false,
    });
    let validator = compile_schema(&schema).expect("generated schema must compile");
    let content = serde_json::to_string(&instance).expect("generated instance must serialize");
    let result = validate_response(validator.as_ref(), &content);
    let ChoiceValidationResult::SchemaViolations(violations) = result else {
    return Err(TestCaseError::fail("generated invalid instance did not violate its schema"));
    };

    prop_assert_eq!(violations.len(), field_count.min(MAX_SCHEMA_VIOLATIONS));
    prop_assert!((1..=MAX_SCHEMA_VIOLATIONS).contains(&violations.len()));
    for violation in violations {
    prop_assert!(violation.path.starts_with('/'));
    let field = violation.path.strip_prefix('/').expect("path prefix checked");
    prop_assert!(instance.contains_key(field));
    prop_assert_eq!(violation.path.as_str(), format!("/{}", pointer_segment(field)));
    prop_assert!(!violation.expected.is_empty());
    prop_assert!(violation.expected.chars().count() <= MAX_EXPECTED_CHARS);
    prop_assert!(violation.actual.chars().count() <= MAX_ACTUAL_CHARS);
    prop_assert!(violation.actual.ends_with('…'));
    prop_assert!(std::str::from_utf8(violation.actual.as_bytes()).is_ok());
    }
    }
    }

    #[test]
    fn extraction_is_not_applicable_without_response_format_or_for_other_types() {
        assert_not_applicable(None);
        assert_not_applicable(Some(json!({"type": "text"})));
        assert_not_applicable(Some(json!({"type": "json_object"})));
        assert_not_applicable(Some(json!({"type": "unknown"})));
    }

    #[test]
    fn extraction_classifies_malformed_response_format_shapes() {
        for response_format in [Value::Null, json!("json_schema"), json!([]), json!(1)] {
            assert_malformed(
                response_format,
                MalformedResponseFormat::ResponseFormatMustBeObject,
            );
        }
        assert_malformed(
            json!({}),
            MalformedResponseFormat::ResponseFormatTypeMissing,
        );
        assert_malformed(
            json!({"type": null}),
            MalformedResponseFormat::ResponseFormatTypeInvalid,
        );
        assert_malformed(
            json!({"type": 1}),
            MalformedResponseFormat::ResponseFormatTypeInvalid,
        );
    }

    #[test]
    fn extraction_classifies_malformed_json_schema_shapes() {
        assert_malformed(
            json!({"type": "json_schema"}),
            MalformedResponseFormat::JsonSchemaMissing,
        );
        for json_schema in [Value::Null, json!("schema"), json!([]), json!(1)] {
            assert_malformed(
                json!({"type": "json_schema", "json_schema": json_schema}),
                MalformedResponseFormat::JsonSchemaMustBeObject,
            );
        }
        assert_malformed(
            json!({"type": "json_schema", "json_schema": {}}),
            MalformedResponseFormat::SchemaMissing,
        );
    }

    #[test]
    fn extraction_classifies_non_object_empty_and_invalid_schemas_as_compile_failures() {
        let cases = [
            (json!(true), SchemaCompileError::SchemaMustBeObject),
            (json!({}), SchemaCompileError::SchemaMustNotBeEmpty),
        ];
        for (schema, expected) in cases {
            match extract_schema_context(&request_with_response_format(Some(json!({
                "type": "json_schema",
                "json_schema": {"schema": schema}
            })))) {
                SchemaContextExtraction::CompileFailed(actual) => assert_eq!(actual, expected),
                _ => panic!("expected schema compilation failure"),
            }
        }

        assert!(matches!(
            extract_schema_context(&request_with_response_format(Some(json!({
                "type": "json_schema",
                "json_schema": {"schema": {"type": 42}}
            })))),
            SchemaContextExtraction::CompileFailed(SchemaCompileError::InvalidSchema { .. })
        ));
    }

    #[test]
    fn extraction_builds_context_with_raw_schema_length_and_compiled_validator() {
        let schema = json!({
            "type": "object",
            "properties": {"name": {"type": "string"}},
            "required": ["name"]
        });
        let expected_len = schema.to_string().chars().count();
        let request = request_with_response_format(Some(json!({
            "type": "json_schema",
            "json_schema": {
                "name": "person",
                "strict": true,
                "schema": schema
            }
        })));

        let SchemaContextExtraction::Ready(context) = extract_schema_context(&request) else {
            panic!("expected ready schema context");
        };

        assert_eq!(context.raw_schema, schema);
        assert_eq!(context.schema_char_len, expected_len);
        assert_eq!(
            validate_response(context.compiled.as_ref(), r#"{"name":"Ada"}"#),
            ChoiceValidationResult::Pass
        );
        assert!(matches!(
            validate_response(context.compiled.as_ref(), r#"{"name":1}"#),
            ChoiceValidationResult::SchemaViolations(_)
        ));
    }

    #[test]
    fn skip_warning_reasons_do_not_contain_schema_or_content() {
        let reasons = [
            MalformedResponseFormat::ResponseFormatMustBeObject.to_string(),
            MalformedResponseFormat::ResponseFormatTypeMissing.to_string(),
            MalformedResponseFormat::ResponseFormatTypeInvalid.to_string(),
            MalformedResponseFormat::JsonSchemaMissing.to_string(),
            MalformedResponseFormat::JsonSchemaMustBeObject.to_string(),
            MalformedResponseFormat::SchemaMissing.to_string(),
        ];

        for reason in reasons {
            assert!(!reason.contains("secret-schema-value"));
            assert!(!reason.contains("secret-response-content"));
        }
        warn_schema_context_skipped("trace-test");
    }

    #[test]
    fn compile_rejects_non_object_and_empty_schemas() {
        assert_eq!(
            compile_schema(&json!(true)).unwrap_err(),
            SchemaCompileError::SchemaMustBeObject
        );
        assert_eq!(
            compile_schema(&json!({})).unwrap_err(),
            SchemaCompileError::SchemaMustNotBeEmpty
        );
    }

    #[test]
    fn compile_classifies_invalid_draft_2020_12_schema() {
        let error = compile_schema(&json!({"type": 42})).unwrap_err();

        match error {
            SchemaCompileError::InvalidSchema { path, message } => {
                assert!(!path.is_empty());
                assert!(!message.is_empty());
                assert!(message.chars().count() <= MAX_COMPILE_MESSAGE_CHARS + 1);
                assert!(!message.chars().any(char::is_control));
            }
            other => panic!("unexpected compile result: {other:?}"),
        }
    }

    #[test]
    fn draft_2020_12_keywords_are_enforced() {
        let validator = compile_schema(&json!({
            "type": "array",
            "prefixItems": [{"type": "string"}],
            "items": false
        }))
        .unwrap();

        assert_eq!(
            validate_response(&validator, r#"["first"]"#),
            ChoiceValidationResult::Pass
        );
        let result = validate_response(&validator, r#"["first", "extra"]"#);
        let ChoiceValidationResult::SchemaViolations(violations) = result else {
            panic!("expected schema violations");
        };
        assert!(violations.iter().any(|violation| {
            violation.expected.contains("items") || violation.expected.contains("falseSchema")
        }));
    }

    #[test]
    fn valid_response_passes_without_transformation() {
        let validator = object_validator(json!({"name": {"type": "string"}}));

        assert_eq!(
            validate_response(&validator, r#"{"name":"Ada"}"#),
            ChoiceValidationResult::Pass
        );
    }

    #[test]
    fn whitespace_only_content_is_skipped() {
        let validator = object_validator(json!({}));

        assert_eq!(
            validate_response(&validator, " \n\t\r "),
            ChoiceValidationResult::Skipped
        );
    }

    #[test]
    fn parse_error_reports_bounded_sanitized_byte_offset() {
        let validator = object_validator(json!({}));
        let content = "{\n  \"é\": true,\n  \"broken\": ]\n}";
        let result = validate_response(&validator, content);

        let ChoiceValidationResult::JsonParseError {
            byte_offset,
            expected,
        } = result
        else {
            panic!("expected JSON parse error");
        };
        assert_eq!(byte_offset, content.find(']').unwrap());
        assert!(!expected.is_empty());
        assert!(expected.chars().count() <= MAX_EXPECTED_CHARS + 1);
        assert!(!expected.chars().any(char::is_control));
        assert!(!expected.contains(" at line "));
    }

    #[test]
    fn violations_include_pointer_constraint_and_actual_value() {
        let validator = object_validator(json!({
            "profile": {
                "type": "object",
                "properties": {"age": {"type": "integer", "minimum": 18}},
                "required": ["age"]
            }
        }));
        let result = validate_response(&validator, r#"{"profile":{"age":12}}"#);

        let ChoiceValidationResult::SchemaViolations(violations) = result else {
            panic!("expected schema violations");
        };
        assert_eq!(violations[0].path, "/profile/age");
        assert!(violations[0].expected.contains("minimum"));
        assert_eq!(violations[0].actual, "12");
    }

    #[test]
    fn violations_are_capped_at_fifty() {
        let required = (0..75)
            .map(|index| format!("field_{index}"))
            .collect::<Vec<_>>();
        let validator = compile_schema(&json!({
            "type": "object",
            "required": required
        }))
        .unwrap();
        let result = validate_response(&validator, "{}");

        let ChoiceValidationResult::SchemaViolations(violations) = result else {
            panic!("expected schema violations");
        };
        assert_eq!(violations.len(), MAX_SCHEMA_VIOLATIONS);
    }

    #[test]
    fn actual_value_truncation_is_utf8_character_safe() {
        let validator = compile_schema(&json!({"type": "integer"})).unwrap();
        let content = serde_json::to_string(&"界".repeat(250)).unwrap();
        let result = validate_response(&validator, &content);

        let ChoiceValidationResult::SchemaViolations(violations) = result else {
            panic!("expected schema violations");
        };
        assert_eq!(violations[0].actual.chars().count(), MAX_ACTUAL_CHARS);
        assert!(violations[0].actual.ends_with('…'));
        assert!(std::str::from_utf8(violations[0].actual.as_bytes()).is_ok());
    }

    #[tokio::test]
    async fn async_validation_returns_completed_result() {
        let validator = object_validator(json!({"id": {"type": "integer"}}));
        let outcome = validate_response_async(validator, r#"{"id":7}"#.to_owned()).await;

        assert_eq!(outcome.result, ChoiceValidationResult::Pass);
        assert_eq!(outcome.internal_skip, None);
    }

    #[tokio::test]
    async fn async_timeout_is_an_internal_skip() {
        let outcome = run_blocking_validation(
            || {
                std::thread::sleep(Duration::from_millis(25));
                ChoiceValidationResult::Pass
            },
            Duration::from_millis(1),
        )
        .await;

        assert_eq!(outcome.result, ChoiceValidationResult::Skipped);
        assert_eq!(outcome.internal_skip, Some(ValidationInternalSkip::Timeout));
    }

    #[tokio::test]
    async fn async_worker_panic_is_an_internal_skip() {
        let outcome =
            run_blocking_validation(|| panic!("intentional test panic"), Duration::from_secs(1))
                .await;

        assert_eq!(outcome.result, ChoiceValidationResult::Skipped);
        assert_eq!(
            outcome.internal_skip,
            Some(ValidationInternalSkip::WorkerPanicked)
        );
    }
}
