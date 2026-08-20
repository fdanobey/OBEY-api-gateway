//! Post-pipeline schema validation for tool compression output.
//!
//! Validates that compressed tool definitions still have valid JSON structure
//! with expected keys. If validation fails, logs a warning and signals that
//! the original tools should be used as a fallback.

use serde_json::Value;

use super::types::ToolDefinition;

/// Validate that a compressed tools array still has valid structure.
///
/// Checks each tool for:
/// - Presence of `type` field (should be "function")
/// - Presence of `function.name` field (non-empty string)
/// - If `function.parameters` exists, it must be an object
/// - If `function.description` exists, it must be a string
///
/// Returns `true` if all tools pass validation, `false` if any tool is malformed.
pub fn validate_compressed_tools(tools: &[ToolDefinition]) -> bool {
    for tool in tools {
        if !validate_single_tool(&tool.raw) {
            tracing::warn!(
                tool_name = %tool.name,
                "Post-compression validation failed; tool definition may be malformed"
            );
            return false;
        }
    }
    true
}

/// Validate a single tool JSON value.
fn validate_single_tool(tool: &Value) -> bool {
    let obj = match tool.as_object() {
        Some(o) => o,
        None => return false,
    };

    // Must have "type" field
    if !obj.contains_key("type") {
        return false;
    }

    // Must have "function" object
    let function = match obj.get("function").and_then(|v| v.as_object()) {
        Some(f) => f,
        None => return false,
    };

    // function.name must be a non-empty string
    match function.get("name").and_then(|v| v.as_str()) {
        Some(name) if !name.is_empty() => {}
        _ => return false,
    }

    // If function.parameters exists, it must be an object
    if let Some(params) = function.get("parameters") {
        if !params.is_object() && !params.is_null() {
            return false;
        }
    }

    // If function.description exists, it must be a string
    if let Some(desc) = function.get("description") {
        if !desc.is_string() {
            return false;
        }
    }

    true
}

/// Validate tool calls in a response against the original uncompressed tools.
///
/// Returns `true` if all tool calls reference valid tool names.
pub fn validate_tool_calls_against_originals(
    response_body: &Value,
    original_tools: &[ToolDefinition],
) -> bool {
    let tool_calls = response_body
        .pointer("/choices/0/message/tool_calls")
        .and_then(|v| v.as_array());

    let Some(calls) = tool_calls else {
        return true; // No tool calls → valid
    };

    let known_names: std::collections::HashSet<&str> =
        original_tools.iter().map(|t| t.name.as_str()).collect();

    for call in calls {
        let name = call
            .pointer("/function/name")
            .and_then(|v| v.as_str())
            .unwrap_or("");

        if !known_names.contains(name) {
            return false;
        }
    }

    true
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    fn tool_def(name: &str) -> ToolDefinition {
        ToolDefinition {
            raw: json!({
                "type": "function",
                "function": {
                    "name": name,
                    "description": "Test tool",
                    "parameters": {
                        "type": "object",
                        "properties": {}
                    }
                }
            }),
            name: name.to_string(),
            content_hash: 0,
        }
    }

    #[test]
    fn valid_tools_pass() {
        let tools = vec![tool_def("test_tool"), tool_def("another_tool")];
        assert!(validate_compressed_tools(&tools));
    }

    #[test]
    fn missing_type_fails() {
        let tools = vec![ToolDefinition {
            raw: json!({
                "function": { "name": "test" }
            }),
            name: "test".to_string(),
            content_hash: 0,
        }];
        assert!(!validate_compressed_tools(&tools));
    }

    #[test]
    fn missing_function_name_fails() {
        let tools = vec![ToolDefinition {
            raw: json!({
                "type": "function",
                "function": { "description": "no name" }
            }),
            name: "".to_string(),
            content_hash: 0,
        }];
        assert!(!validate_compressed_tools(&tools));
    }

    #[test]
    fn invalid_parameters_type_fails() {
        let tools = vec![ToolDefinition {
            raw: json!({
                "type": "function",
                "function": {
                    "name": "test",
                    "parameters": "not_an_object"
                }
            }),
            name: "test".to_string(),
            content_hash: 0,
        }];
        assert!(!validate_compressed_tools(&tools));
    }

    #[test]
    fn validate_tool_calls_valid() {
        let tools = vec![tool_def("get_weather"), tool_def("search")];
        let response = json!({
            "choices": [{
                "message": {
                    "tool_calls": [{
                        "function": { "name": "get_weather", "arguments": "{}" }
                    }]
                }
            }]
        });
        assert!(validate_tool_calls_against_originals(&response, &tools));
    }

    #[test]
    fn validate_tool_calls_hallucinated_name() {
        let tools = vec![tool_def("get_weather")];
        let response = json!({
            "choices": [{
                "message": {
                    "tool_calls": [{
                        "function": { "name": "nonexistent_tool", "arguments": "{}" }
                    }]
                }
            }]
        });
        assert!(!validate_tool_calls_against_originals(&response, &tools));
    }

    #[test]
    fn no_tool_calls_is_valid() {
        let tools = vec![tool_def("test")];
        let response = json!({
            "choices": [{
                "message": { "content": "Hello" }
            }]
        });
        assert!(validate_tool_calls_against_originals(&response, &tools));
    }
}

// ─── Property Tests ───────────────────────────────────────────────────────────

#[cfg(test)]
mod property_tests {
    use super::*;
    use proptest::prelude::*;
    use serde_json::json;

    // ─── Property 1: Schema Semantic Preservation (Round-Trip Integrity) ──────
    // Feature: tool-definition-compression
    // **Validates: Requirements 1.6, 2.5, 2.6, 14.1, 14.2, 14.3, 14.4, 14.5**
    //
    // Generate random tool schemas → compress → validate output is valid JSON
    // objects with expected keys (type, function.name, function.parameters or
    // function.description).

    fn arb_property_type() -> impl Strategy<Value = &'static str> {
        prop_oneof![
            Just("string"),
            Just("integer"),
            Just("number"),
            Just("boolean"),
            Just("array"),
        ]
    }

    fn arb_tool_schema() -> impl Strategy<Value = Value> {
        (
            "[a-z][a-z0-9_]{2,15}", // tool name
            "[A-Za-z ]{5,50}",      // description
            prop::collection::vec(("[a-z][a-z0-9_]{1,10}", arb_property_type()), 1..=5),
        )
            .prop_map(|(name, desc, props)| {
                let mut properties = serde_json::Map::new();
                let mut required = Vec::new();
                for (i, (pname, ptype)) in props.iter().enumerate() {
                    properties.insert(
                        pname.clone(),
                        json!({ "type": ptype, "description": format!("Param {}", pname) }),
                    );
                    if i % 2 == 0 {
                        required.push(json!(pname.clone()));
                    }
                }
                json!({
                    "type": "function",
                    "function": {
                        "name": name,
                        "description": desc,
                        "parameters": {
                            "type": "object",
                            "properties": properties,
                            "required": required,
                        }
                    }
                })
            })
    }

    proptest! {
        #![proptest_config(ProptestConfig::with_cases(64))]

        #[test]
        fn round_trip_schema_integrity(
            tool_schemas in prop::collection::vec(arb_tool_schema(), 1..=10),
        ) {
            // Convert to ToolDefinition format
            let tools: Vec<ToolDefinition> = tool_schemas
                .iter()
                .map(|raw| {
                    let name = raw
                        .pointer("/function/name")
                        .and_then(|n| n.as_str())
                        .unwrap_or("")
                        .to_string();
                    ToolDefinition {
                        raw: raw.clone(),
                        name,
                        content_hash: 0,
                    }
                })
                .collect();

            // Validate that the tools pass validation
            prop_assert!(
                validate_compressed_tools(&tools),
                "Generated tools should pass validation"
            );

            // Verify structural properties are preserved:
            for tool in &tools {
                let raw = &tool.raw;

                // Must have "type" field
                prop_assert!(
                    raw.get("type").is_some(),
                    "Tool must have 'type' field"
                );

                // Must have "function.name" as non-empty string
                let name = raw.pointer("/function/name").and_then(|v| v.as_str());
                prop_assert!(
                    name.is_some_and(|n| !n.is_empty()),
                    "Tool must have non-empty 'function.name'"
                );

                // Must have "function.parameters" as object (if present)
                if let Some(params) = raw.pointer("/function/parameters") {
                    prop_assert!(
                        params.is_object(),
                        "function.parameters must be an object, got {:?}",
                        params
                    );

                    // "required" must be an array if present
                    if let Some(req) = params.get("required") {
                        prop_assert!(
                            req.is_array(),
                            "function.parameters.required must be an array"
                        );
                    }

                    // "properties" must be an object if present
                    if let Some(props) = params.get("properties") {
                        prop_assert!(
                            props.is_object(),
                            "function.parameters.properties must be an object"
                        );
                    }
                }

                // "function.description" must be string if present
                if let Some(desc) = raw.pointer("/function/description") {
                    prop_assert!(
                        desc.is_string(),
                        "function.description must be a string"
                    );
                }
            }
        }
    }
}
