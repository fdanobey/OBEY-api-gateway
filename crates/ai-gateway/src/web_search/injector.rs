//! Tool injector — appends the `web_search` tool definition.
//!
//! When the provider-agnostic web search feature is enabled, the injector
//! transparently augments outgoing chat-completion requests with an
//! OpenAI-compatible function tool so downstream models can invoke web search.

use std::sync::LazyLock;

use serde_json::{json, Value};

use crate::models::openai::OpenAIRequest;

use crate::web_search::models::DEFAULT_MAX_RESULTS;

pub const WEB_SEARCH_TOOL_NAME: &str = "web_search";

static WEB_SEARCH_DEFINITION: LazyLock<Value> = LazyLock::new(|| {
    json!({
        "type": "function",
        "function": {
            "name": WEB_SEARCH_TOOL_NAME,
            "description": "Search the web for current information using a single query. Cite the source URLs of any results you use directly in your answer text.",
            "parameters": {
                "type": "object",
                "properties": {
                    "q": {
                        "type": "string",
                        "minLength": 1,
                        "maxLength": 2000,
                        "description": "Search query text"
                    },
                    "domains": {
                        "type": "array",
                        "items": { "type": "string" },
                        "maxItems": 10,
                        "description": "Restrict results to these domains"
                    },
                    "recency": {
                        "type": "integer",
                        "minimum": 1,
                        "maximum": 365,
                        "description": "Filter results by age in days"
                    },
                    "max_results": {
                        "type": "integer",
                        "minimum": 1,
                        "maximum": 10,
                        "default": DEFAULT_MAX_RESULTS,
                        "description": "Number of results to return (default: 5)"
                    }
                },
                "required": ["q"]
            }
        }
    })
});

/// Returns the `web_search` tool definition.
pub fn web_search_definition() -> &'static Value {
    &WEB_SEARCH_DEFINITION
}

fn tool_array_contains(tools: &[Value], name: &str) -> bool {
    tools.iter().any(|tool| {
        tool.get("function")
            .and_then(|f| f.get("name"))
            .and_then(|n| n.as_str())
            == Some(name)
    })
}

/// Appends the `web_search` tool definition to `tools` unless a
/// client-declared tool with the same name already exists. Returns `true`
/// when the definition was injected.
pub fn inject_web_search_tool(tools: &mut Vec<Value>) -> bool {
    if tool_array_contains(tools, WEB_SEARCH_TOOL_NAME) {
        return false;
    }
    tools.push(WEB_SEARCH_DEFINITION.clone());
    true
}

fn is_web_search_tool_call(tc: &Value) -> bool {
    tc.get("function")
        .and_then(|f| f.get("name"))
        .and_then(|n| n.as_str())
        == Some(WEB_SEARCH_TOOL_NAME)
}

/// Removes the injected `web_search` tool definition from the request's
/// tools array. Client tools are preserved untouched; an emptied tools key is
/// dropped entirely.
pub fn strip_web_search_tool_definition(request: &mut OpenAIRequest) {
    if let Some(Value::Array(arr)) = request.extra.get_mut("tools") {
        arr.retain(|tool| !is_web_search_tool_call(tool));
        if arr.is_empty() {
            request.extra.remove("tools");
        }
    }
}

/// Removes gateway-executed `web_search` tool calls from the response's
/// tool_calls array. The gateway has already executed them server-side;
/// leaking them back to the client causes harnesses to reject the response.
/// Client tool calls are preserved untouched; an emptied tool_calls key is
/// dropped entirely.
pub fn strip_web_search_tool_calls(response: &mut crate::models::openai::OpenAIResponse) {
    for choice in &mut response.choices {
        let should_remove = {
            let calls = choice
                .message
                .extra
                .get_mut("tool_calls")
                .and_then(|v| v.as_array_mut());
            if let Some(calls) = calls {
                calls.retain(|tc| !is_web_search_tool_call(tc));
                calls.is_empty()
            } else {
                false
            }
        };
        if should_remove {
            choice.message.extra.remove("tool_calls");
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::models::openai::{Message, OpenAIRequest, OpenAIResponse};
    use serde_json::{json, Map};

    fn make_request() -> OpenAIRequest {
        OpenAIRequest {
            model: "gpt-4o".to_string(),
            messages: vec![],
            stream: false,
            temperature: None,
            max_tokens: None,
            extra: Map::new(),
        }
    }

    fn tools_names(req: &OpenAIRequest) -> Vec<String> {
        req.extra
            .get("tools")
            .and_then(|t| t.as_array())
.map(|arr| {
                arr.iter()
                    .filter_map(|t| {
                        t.get("function")
                            .and_then(|f| f.get("name"))
                            .and_then(|n| n.as_str())
                            .map(|s| s.to_string())
                    })
                    .collect::<Vec<_>>()
            })
            .unwrap_or_default()
    }

    #[test]
    fn injects_when_absent() {
        let mut tools: Vec<Value> = vec![];
        assert!(inject_web_search_tool(&mut tools));
        assert_eq!(tools.len(), 1);
        assert_eq!(
            tools[0]
                .get("function")
                .and_then(|f| f.get("name"))
                .and_then(|n| n.as_str()),
            Some(WEB_SEARCH_TOOL_NAME)
        );
    }

    #[test]
    fn injects_into_empty_array() {
        let mut tools: Vec<Value> = vec![json!({
            "type": "function",
            "function": {"name": "custom_tool", "parameters": {"type": "object"}}
        })];
        assert!(inject_web_search_tool(&mut tools));
        let names: Vec<&str> = tools
            .iter()
            .filter_map(|t| {
                t.get("function")
                    .and_then(|f| f.get("name"))
                    .and_then(|n| n.as_str())
            })
            .collect();
        assert_eq!(names, vec!["custom_tool", "web_search"]);
    }

    #[test]
    fn no_duplicate_when_client_declares_web_search() {
        let mut tools: Vec<Value> = vec![json!({
            "type": "function",
            "function": {"name": "web_search", "parameters": {"type": "object"}}
        })];
        assert!(!inject_web_search_tool(&mut tools));
        assert_eq!(tools.len(), 1);
        assert_eq!(
            tools[0]
                .get("function")
                .and_then(|f| f.get("parameters"))
                .and_then(|p| p.get("type"))
                .and_then(|t| t.as_str()),
            Some("object")
        );
    }

    #[test]
    fn injection_is_idempotent() {
        let mut tools: Vec<Value> = vec![];
        assert!(inject_web_search_tool(&mut tools));
        assert!(!inject_web_search_tool(&mut tools));
        assert_eq!(tools.len(), 1);
    }

    #[test]
    fn definition_has_correct_name_and_schema() {
        let def = web_search_definition();
        assert_eq!(
            def.get("type").and_then(|t| t.as_str()),
            Some("function")
        );
        let function = def.get("function").unwrap();
        assert_eq!(
            function.get("name").and_then(|n| n.as_str()),
            Some("web_search")
        );
        let params = function.get("parameters").unwrap();
        assert_eq!(params.get("type").and_then(|t| t.as_str()), Some("object"));
        let props = params.get("properties").unwrap();
        assert_eq!(props.get("q").unwrap().get("type"), Some(&json!("string")));
        assert_eq!(
            params.get("required").and_then(|r| r.as_array()).map(|a| a.len()),
            Some(1)
        );
        assert_eq!(
            params
                .get("required")
                .unwrap()
                .get(0)
                .and_then(|r| r.as_str()),
            Some("q")
        );
        assert_eq!(
            props.get("domains").unwrap().get("type"),
            Some(&json!("array"))
        );
        assert_eq!(
            props.get("recency").unwrap().get("type"),
            Some(&json!("integer"))
        );
        assert_eq!(
            props.get("recency").unwrap().get("minimum"),
            Some(&json!(1))
        );
        assert_eq!(
            props.get("recency").unwrap().get("maximum"),
            Some(&json!(365))
        );
        let max_results = props.get("max_results").unwrap();
        assert_eq!(max_results.get("type"), Some(&json!("integer")));
        assert_eq!(max_results.get("minimum"), Some(&json!(1)));
        assert_eq!(max_results.get("maximum"), Some(&json!(10)));
        assert_eq!(max_results.get("default"), Some(&json!(5)));
    }

    #[test]
    fn strip_definition_removes_only_web_search() {
        let mut request = make_request();
        request.extra.insert(
            "tools".to_string(),
            json!([
                {"type": "function", "function": {"name": "client_tool"}},
                {"type": "function", "function": {"name": "web_search"}}
            ]),
        );
        strip_web_search_tool_definition(&mut request);
        assert_eq!(tools_names(&request), vec!["client_tool"]);
    }

    #[test]
    fn strip_definition_drops_emptied_tools_key() {
        let mut request = make_request();
        request.extra.insert(
            "tools".to_string(),
            json!([{"type": "function", "function": {"name": "web_search"}}]),
        );
        strip_web_search_tool_definition(&mut request);
        assert!(!request.extra.contains_key("tools"));
    }

    #[test]
    fn strip_definition_noop_without_tools_key() {
        let mut request = make_request();
        strip_web_search_tool_definition(&mut request);
        assert!(!request.extra.contains_key("tools"));
    }

    #[test]
    fn strip_definition_noop_with_non_array_tools() {
        let mut request = make_request();
        request.extra.insert("tools".to_string(), json!({"not": "array"}));
        strip_web_search_tool_definition(&mut request);
        assert_eq!(request.extra.get("tools"), Some(&json!({"not": "array"})));
    }

    fn make_response_with_tool_calls(tool_calls: Vec<Value>) -> OpenAIResponse {
        let message = Message {
            role: "assistant".to_string(),
            content: serde_json::Value::Null,
            extra: {
                let mut extra = Map::new();
                extra.insert("tool_calls".to_string(), Value::Array(tool_calls));
                extra
            },
        };
        OpenAIResponse {
            id: String::new(),
            object: String::new(),
            created: 0,
            model: String::new(),
            choices: vec![crate::models::openai::Choice {
                index: 0,
                message,
                finish_reason: Some("tool_calls".to_string()),
                extra: Map::new(),
            }],
            usage: Default::default(),
            extra: Map::new(),
        }
    }

    fn make_tool_call(id: &str, name: &str, args: &str) -> Value {
        json!({
            "id": id,
            "type": "function",
            "function": {"name": name, "arguments": args}
        })
    }

    #[test]
    fn strip_calls_removes_only_web_search_calls() {
        let mut response = make_response_with_tool_calls(vec![
            make_tool_call("call_1", "web_search", r#"{"q":"x"}"#),
            make_tool_call("call_2", "client_tool", "{}"),
        ]);
        strip_web_search_tool_calls(&mut response);
        let calls = response.choices[0]
            .message
            .extra
            .get("tool_calls")
            .and_then(|t| t.as_array())
            .unwrap();
        assert_eq!(calls.len(), 1);
        assert_eq!(
            calls[0]
                .get("function")
                .and_then(|f| f.get("name"))
                .and_then(|n| n.as_str()),
            Some("client_tool")
        );
    }

    #[test]
    fn strip_calls_drops_emptied_tool_calls_key() {
        let mut response = make_response_with_tool_calls(vec![make_tool_call(
            "call_1",
            "web_search",
            r#"{"q":"x"}"#,
        )]);
        strip_web_search_tool_calls(&mut response);
        assert!(response.choices[0]
            .message
            .extra
            .get("tool_calls")
            .is_none());
    }

    #[test]
    fn strip_calls_noop_without_tool_calls() {
        let mut response = make_response_with_tool_calls(vec![]);
        response.choices[0].message.extra.remove("tool_calls");
        strip_web_search_tool_calls(&mut response);
        assert!(response.choices[0]
            .message
            .extra
            .get("tool_calls")
            .is_none());
    }

    #[test]
    fn strip_calls_noop_with_non_array_tool_calls() {
        let mut response = make_response_with_tool_calls(vec![]);
        response.choices[0]
            .message
            .extra
            .insert("tool_calls".to_string(), json!({"not": "array"}));
        strip_web_search_tool_calls(&mut response);
        assert_eq!(
            response.choices[0].message.extra.get("tool_calls"),
            Some(&json!({"not": "array"}))
        );
    }

    fn make_multi_choice_response_with_tool_calls(
        tool_calls_per_choice: Vec<Vec<Value>>,
    ) -> OpenAIResponse {
        let choices = tool_calls_per_choice
            .into_iter()
            .enumerate()
            .map(|(index, tool_calls)| crate::models::openai::Choice {
                index: index as u32,
                message: Message {
                    role: "assistant".to_string(),
                    content: serde_json::Value::Null,
                    extra: {
                        let mut extra = Map::new();
                        extra.insert("tool_calls".to_string(), Value::Array(tool_calls));
                        extra
                    },
                },
                finish_reason: Some("tool_calls".to_string()),
                extra: Map::new(),
            })
            .collect();
        OpenAIResponse {
            id: String::new(),
            object: String::new(),
            created: 0,
            model: String::new(),
            choices,
            usage: Default::default(),
            extra: Map::new(),
        }
    }

    #[test]
    fn strip_calls_removes_web_search_from_every_choice() {
        let mut response = make_multi_choice_response_with_tool_calls(vec![
            vec![
                make_tool_call("call_1", "web_search", r#"{"q":"a"}"#),
                make_tool_call("call_2", "client_tool", "{}"),
            ],
            vec![make_tool_call("call_3", "web_search", r#"{"q":"b"}"#)],
            vec![make_tool_call("call_4", "another_client_tool", "{}")],
        ]);
        strip_web_search_tool_calls(&mut response);
        let calls = response.choices[0]
            .message
            .extra
            .get("tool_calls")
            .and_then(|t| t.as_array())
            .unwrap();
        assert_eq!(calls.len(), 1);
        assert_eq!(
            calls[0]
                .get("function")
                .and_then(|f| f.get("name"))
                .and_then(|n| n.as_str()),
            Some("client_tool")
        );
        assert!(response.choices[1]
            .message
            .extra
            .get("tool_calls")
            .is_none());
        let calls2 = response.choices[2]
            .message
            .extra
            .get("tool_calls")
            .and_then(|t| t.as_array())
            .unwrap();
        assert_eq!(calls2.len(), 1);
    }

    #[test]
    fn strip_calls_drops_emptied_key_in_later_choice() {
        let mut response = make_multi_choice_response_with_tool_calls(vec![
            vec![make_tool_call("call_1", "client_tool", "{}")],
            vec![make_tool_call("call_2", "web_search", r#"{"q":"x"}"#)],
        ]);
        strip_web_search_tool_calls(&mut response);
        assert!(response.choices[0]
            .message
            .extra
            .get("tool_calls")
            .is_some());
        assert!(response.choices[1]
            .message
            .extra
            .get("tool_calls")
            .is_none());
    }

    #[test]
    fn strip_calls_noop_with_empty_choices() {
        let mut response = make_multi_choice_response_with_tool_calls(vec![]);
        strip_web_search_tool_calls(&mut response);
        assert!(response.choices.is_empty());
    }
}
