use super::engines::{tool_def::ToolDefinitionEngine, CompressiblePayload, CompressionContext};
use crate::models::openai::OpenAIRequest;
use proptest::prelude::*;
use proptest::test_runner::TestCaseResult;
use serde_json::{json, Map, Number, Value};
use std::collections::BTreeSet;

const PRESERVED_FIELDS: [&str; 4] = ["name", "type", "required", "enum"];

type PathBytes = Vec<(String, Vec<u8>)>;
type PathValues = Vec<(String, Value)>;

fn identifier() -> impl Strategy<Value = String> {
    proptest::string::string_regex("[a-z][a-z0-9_]{0,15}")
        .expect("tool-schema identifier regex must compile")
}

fn verbose_description(subject: &str, seed: u16) -> String {
    format!(
        "This tool can be used to carefully process the generated {subject} value for case {seed}. \
         It provides a detailed and reliable operation while preserving every structured schema \
         constraint supplied by the caller. For example, callers may provide generated case \
         {seed} values during normal operation. Note: this intentionally verbose explanatory \
         prose is safe to summarize without changing schema data."
    )
}

fn scalar_schema(kind: u8, seed: u16, subject: &str) -> Value {
    let description = verbose_description(subject, seed);
    match kind % 4 {
        0 => {
            let first = format!("value_{seed}_alpha");
            let second = format!("value_{seed}_beta");
            json!({
                "type": "string",
                "description": description,
                "enum": [first.clone(), second.clone()],
                "default": first,
                "const": second
            })
        }
        1 => {
            let first = i64::from(seed);
            let second = first + 1;
            json!({
                "type": "integer",
                "description": description,
                "enum": [first, second],
                "default": first,
                "const": second
            })
        }
        2 => {
            let first = Number::from_f64(f64::from(seed) + 0.25)
                .expect("finite generated number must be valid JSON");
            let second = Number::from_f64(f64::from(seed) + 0.75)
                .expect("finite generated number must be valid JSON");
            json!({
                "type": "number",
                "description": description,
                "enum": [Value::Number(first.clone()), Value::Number(second.clone())],
                "default": Value::Number(first),
                "const": Value::Number(second)
            })
        }
        _ => {
            let selected = seed % 2 == 0;
            json!({
                "type": "boolean",
                "description": description,
                "enum": [false, true],
                "default": !selected,
                "const": selected
            })
        }
    }
}

fn tool_definitions_strategy() -> impl Strategy<Value = Value> {
    (
        proptest::collection::btree_set(identifier(), 1..4),
        proptest::collection::btree_set(identifier(), 1..5),
        proptest::collection::btree_set(identifier(), 1..5),
        proptest::collection::vec(any::<u8>(), 1..16),
        proptest::collection::vec(any::<bool>(), 1..16),
        any::<u16>(),
    )
        .prop_map(
            |(function_names, property_names, nested_names, kinds, required_flags, seed)| {
                generated_tools(
                    function_names,
                    property_names,
                    nested_names,
                    &kinds,
                    &required_flags,
                    seed,
                )
            },
        )
}

fn generated_tools(
    function_names: BTreeSet<String>,
    property_names: BTreeSet<String>,
    nested_names: BTreeSet<String>,
    kinds: &[u8],
    required_flags: &[bool],
    seed: u16,
) -> Value {
    let functions = function_names.into_iter().collect::<Vec<_>>();
    let properties = property_names.into_iter().collect::<Vec<_>>();
    let nested = nested_names.into_iter().collect::<Vec<_>>();
    let tools = functions
        .iter()
        .enumerate()
        .map(|(function_index, function_name)| {
            let mut outer_properties = Map::new();
            let mut outer_required = Vec::new();

            for (property_index, property_name) in properties.iter().enumerate() {
                let mut nested_properties = Map::new();
                let mut nested_required = Vec::new();
                for (nested_index, nested_name) in nested.iter().enumerate() {
                    let discriminator = function_index * properties.len() * nested.len()
                        + property_index * nested.len()
                        + nested_index;
                    let nested_seed = seed.wrapping_add(discriminator as u16);
                    nested_properties.insert(
                        nested_name.clone(),
                        scalar_schema(kinds[discriminator % kinds.len()], nested_seed, nested_name),
                    );
                    if required_flags[discriminator % required_flags.len()] {
                        nested_required.push(nested_name.clone());
                    }
                }

                outer_properties.insert(
                    property_name.clone(),
                    json!({
                        "type": "object",
                        "description": verbose_description(property_name, seed),
                        "properties": nested_properties,
                        "required": nested_required,
                        "additionalProperties": false,
                        "default": {"description": "literal default description", "case": seed},
                        "const": {"description": "literal const description", "case": seed},
                    }),
                );
                if required_flags[(function_index + property_index) % required_flags.len()] {
                    outer_required.push(property_name.clone());
                }
            }

            json!({
                "type": "function",
                "function": {
                    "name": function_name,
                    "description": verbose_description(function_name, seed),
                    "parameters": {
                        "type": "object",
                        "description": verbose_description("parameter object", seed),
                        "properties": outer_properties,
                        "required": outer_required,
                        "additionalProperties": false
                    }
                }
            })
        })
        .collect::<Vec<_>>();
    Value::Array(tools)
}

fn payload(tools: Value) -> CompressiblePayload {
    let mut extra = Map::new();
    extra.insert("tools".to_owned(), tools);
    CompressiblePayload::from(OpenAIRequest {
        model: "gpt-4o".to_owned(),
        messages: Vec::new(),
        stream: false,
        temperature: None,
        max_tokens: None,
        extra,
    })
}

fn path_child(path: &str, child: &str) -> String {
    if path.is_empty() {
        format!("/{child}")
    } else {
        format!("{path}/{child}")
    }
}

fn non_description_projection(value: &Value) -> PathBytes {
    fn visit(value: &Value, path: &str, projection: &mut PathBytes) {
        match value {
            Value::Object(object) => {
                let keys = object.keys().collect::<Vec<_>>();
                projection.push((
                    format!("{path}#object_keys"),
                    serde_json::to_vec(&keys).expect("object keys must serialize"),
                ));
                for (key, child) in object {
                    if key == "description" && child.is_string() {
                        continue;
                    }
                    visit(child, &path_child(path, key), projection);
                }
            }
            Value::Array(values) => {
                projection.push((
                    format!("{path}#array_len"),
                    values.len().to_le_bytes().to_vec(),
                ));
                for (index, child) in values.iter().enumerate() {
                    visit(child, &path_child(path, &index.to_string()), projection);
                }
            }
            _ => projection.push((
                path.to_owned(),
                serde_json::to_vec(value).expect("JSON leaf must serialize"),
            )),
        }
    }

    let mut projection = Vec::new();
    visit(value, "", &mut projection);
    projection
}

fn selected_field_values(value: &Value) -> PathValues {
    fn visit(value: &Value, path: &str, selected: &mut PathValues) {
        match value {
            Value::Object(object) => {
                for (key, child) in object {
                    let child_path = path_child(path, key);
                    if PRESERVED_FIELDS.contains(&key.as_str()) {
                        selected.push((child_path.clone(), child.clone()));
                    }
                    visit(child, &child_path, selected);
                }
            }
            Value::Array(values) => {
                for (index, child) in values.iter().enumerate() {
                    visit(child, &path_child(path, &index.to_string()), selected);
                }
            }
            _ => {}
        }
    }

    let mut selected = Vec::new();
    visit(value, "", &mut selected);
    selected
}

fn property_name_paths(value: &Value) -> Vec<(String, Vec<String>)> {
    fn visit(value: &Value, path: &str, properties: &mut Vec<(String, Vec<String>)>) {
        match value {
            Value::Object(object) => {
                for (key, child) in object {
                    let child_path = path_child(path, key);
                    if key == "properties" {
                        if let Some(property_object) = child.as_object() {
                            properties.push((
                                child_path.clone(),
                                property_object.keys().cloned().collect(),
                            ));
                        }
                    }
                    visit(child, &child_path, properties);
                }
            }
            Value::Array(values) => {
                for (index, child) in values.iter().enumerate() {
                    visit(child, &path_child(path, &index.to_string()), properties);
                }
            }
            _ => {}
        }
    }

    let mut properties = Vec::new();
    visit(value, "", &mut properties);
    properties
}

fn description_values(value: &Value) -> PathValues {
    fn visit(value: &Value, path: &str, descriptions: &mut PathValues) {
        match value {
            Value::Object(object) => {
                for (key, child) in object {
                    let child_path = path_child(path, key);
                    if key == "description" && child.is_string() {
                        descriptions.push((child_path.clone(), child.clone()));
                    }
                    visit(child, &child_path, descriptions);
                }
            }
            Value::Array(values) => {
                for (index, child) in values.iter().enumerate() {
                    visit(child, &path_child(path, &index.to_string()), descriptions);
                }
            }
            _ => {}
        }
    }

    let mut descriptions = Vec::new();
    visit(value, "", &mut descriptions);
    descriptions
}

fn assert_tool_schema_preservation(tools: Value) -> TestCaseResult {
    let original_tools = tools.clone();
    let context = CompressionContext::new("gpt-4o", "property-test-non-strict");
    let mut payload = payload(tools);
    let input_tokens = context
        .token_counter
        .count_request(&payload.clone().into_openai_request());
    let runtime = tokio::runtime::Builder::new_current_thread()
        .build()
        .expect("tool-schema property-test runtime must build");
    let engine = ToolDefinitionEngine::new(["strict-provider"]);

    let (result, report) = runtime.block_on(engine.compress_with_report(&mut payload, &context));
    let output_tools = payload
        .tool_definitions
        .as_ref()
        .expect("generated tool definitions must remain present");
    let output_tokens = context
        .token_counter
        .count_request(&payload.clone().into_openai_request());

    prop_assert_eq!(
        non_description_projection(output_tools),
        non_description_projection(&original_tools),
        "every non-description path and serialized value must remain byte-identical"
    );
    prop_assert_eq!(
        selected_field_values(output_tools),
        selected_field_values(&original_tools),
        "function names, schema types, required arrays, and enums must be preserved"
    );
    prop_assert_eq!(
        property_name_paths(output_tools),
        property_name_paths(&original_tools),
        "nested property names and paths must be preserved"
    );

    let original_descriptions = description_values(&original_tools);
    let output_descriptions = description_values(output_tools);
    prop_assert_eq!(
        original_descriptions
            .iter()
            .map(|(path, _)| path)
            .collect::<Vec<_>>(),
        output_descriptions
            .iter()
            .map(|(path, _)| path)
            .collect::<Vec<_>>(),
        "description paths must not be added or removed"
    );
    prop_assert!(
        original_descriptions
            .iter()
            .zip(&output_descriptions)
            .any(|((_, original), (_, output))| original != output),
        "generated verbose descriptions should exercise compression"
    );

    prop_assert_eq!(result.tokens_before, input_tokens);
    prop_assert_eq!(result.tokens_after, output_tokens);
    prop_assert!(output_tokens <= input_tokens);
    prop_assert_eq!(
        report.tool_definitions_tokens_saved,
        input_tokens.saturating_sub(output_tokens)
    );
    prop_assert!(result.applied);
    Ok(())
}

proptest! {
    #![proptest_config(ProptestConfig::with_cases(64))]

    #[test]
    fn property_13_tool_schema_preservation(tools in tool_definitions_strategy()) {
        assert_tool_schema_preservation(tools)?;
    }
}
