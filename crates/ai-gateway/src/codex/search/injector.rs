//! Tool injector — appends `codex_search` and `codex_web` tool definitions.
//!
//! When the Codex Search feature is enabled and OAuth is active, the injector
//! transparently augments outgoing chat-completion requests with two
//! OpenAI-compatible function tools so downstream models can invoke web search.

use std::sync::LazyLock;

use serde_json::{json, Value};

use crate::models::openai::OpenAIRequest;

static CODEX_SEARCH_DEFINITION: LazyLock<Value> = LazyLock::new(|| {
    json!({
        "type": "function",
        "function": {
            "name": "codex_search",
            "description": "Search the web for current information using a single query.",
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
                    "response_length": {
                        "type": "string",
                        "enum": ["short", "medium", "long"],
                        "description": "Hint for result verbosity (default: short)"
                    }
                },
                "required": ["q"]
            }
        }
    })
});

static CODEX_WEB_DEFINITION: LazyLock<Value> = LazyLock::new(|| {
    json!({
        "type": "function",
        "function": {
            "name": "codex_web",
            "description": "Multi-step web research tool supporting search, open, find, and click commands within a session.",
            "parameters": {
                "type": "object",
                "properties": {
                    "session_id": {
                        "type": "string",
                        "maxLength": 128,
                        "description": "Existing session ID to continue a research session"
                    },
                    "commands": {
                        "type": "object",
                        "properties": {
                            "search_query": {
                                "type": "array",
                                "maxItems": 10,
                                "items": {
                                    "type": "object",
                                    "properties": {
                                        "q": { "type": "string", "description": "Search query text" },
                                        "domains": { "type": "array", "items": { "type": "string" }, "description": "Restrict to these domains" },
                                        "recency": { "type": "integer", "description": "Filter by age in days" }
                                    },
                                    "required": ["q"]
                                }
                            },
                            "open": {
                                "type": "array",
                                "maxItems": 10,
                                "items": {
                                    "type": "object",
                                    "properties": {
                                        "ref_id": { "type": "string", "description": "Reference ID from search results" },
                                        "lineno": { "type": "integer", "description": "Optional line number to open at" }
                                    },
                                    "required": ["ref_id"]
                                }
                            },
                            "find": {
                                "type": "array",
                                "maxItems": 10,
                                "items": {
                                    "type": "object",
                                    "properties": {
                                        "ref_id": { "type": "string", "description": "Reference ID of the page to search" },
                                        "pattern": { "type": "string", "description": "Text pattern to find" }
                                    },
                                    "required": ["ref_id", "pattern"]
                                }
                            },
                            "click": {
                                "type": "array",
                                "maxItems": 10,
                                "items": {
                                    "type": "object",
                                    "properties": {
                                        "ref_id": { "type": "string", "description": "Reference ID of the result to click" },
                                        "id": { "type": "integer", "description": "Numeric ID of the link to click" }
                                    },
                                    "required": ["ref_id"]
                                }
                            }
                        }
                    },
                    "response_length": {
                        "type": "string",
                        "enum": ["short", "medium", "long"],
                        "description": "Hint for result verbosity"
                    }
                }
            }
        }
    })
});

/// Injects `codex_search` and `codex_web` tool definitions into outgoing
/// chat completion requests when OAuth is active and the feature is enabled.
pub struct ToolInjector;

impl ToolInjector {
    /// Returns the codex_search tool definition.
    #[allow(dead_code)]
    pub fn codex_search_definition() -> &'static Value {
        &CODEX_SEARCH_DEFINITION
    }

    /// Returns the codex_web tool definition.
    #[allow(dead_code)]
    pub fn codex_web_definition() -> &'static Value {
        &CODEX_WEB_DEFINITION
    }

    /// Conditionally injects tool definitions into the request's tools array.
    pub fn inject(request: &mut OpenAIRequest, oauth_active: bool, enabled: bool) {
        if !enabled || !oauth_active {
            return;
        }

        let has_codex_search;
        let has_codex_web;
        let mut tools_array: Vec<Value>;

        match request.extra.get("tools") {
            Some(Value::Array(arr)) => {
                tools_array = arr.clone();
                has_codex_search = tool_array_contains(&tools_array, "codex_search");
                has_codex_web = tool_array_contains(&tools_array, "codex_web");
            }
            Some(Value::Null) | None => {
                tools_array = Vec::new();
                has_codex_search = false;
                has_codex_web = false;
            }
            Some(_other) => {
                return;
            }
        }

        if !has_codex_search {
            tools_array.push(CODEX_SEARCH_DEFINITION.clone());
        }
        if !has_codex_web {
            tools_array.push(CODEX_WEB_DEFINITION.clone());
        }

        request
            .extra
            .insert("tools".to_string(), Value::Array(tools_array));
    }
}

fn tool_array_contains(tools: &[Value], name: &str) -> bool {
    tools.iter().any(|tool| {
        tool.get("function")
            .and_then(|f| f.get("name"))
            .and_then(|n| n.as_str())
            == Some(name)
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::models::openai::OpenAIRequest;
    use serde_json::{json, Map, Value};

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
                    .collect()
            })
            .unwrap_or_default()
    }

    #[test]
    fn inject_when_enabled_and_oauth_active() {
        let mut req = make_request();
        ToolInjector::inject(&mut req, true, true);
        let names = tools_names(&req);
        assert!(names.contains(&"codex_search".to_string()));
        assert!(names.contains(&"codex_web".to_string()));
    }

    #[test]
    fn inject_noop_when_disabled() {
        let mut req = make_request();
        ToolInjector::inject(&mut req, true, false);
        assert!(!req.extra.contains_key("tools"));
    }

    #[test]
    fn inject_noop_when_oauth_inactive() {
        let mut req = make_request();
        ToolInjector::inject(&mut req, false, true);
        assert!(!req.extra.contains_key("tools"));
    }

    #[test]
    fn inject_no_duplicate_when_client_provides_codex_search() {
        let mut req = make_request();
        req.extra.insert(
            "tools".to_string(),
            json!([
                {"type": "function", "function": {"name": "codex_search", "parameters": {"type": "object"}}}
            ]),
        );
        ToolInjector::inject(&mut req, true, true);
        let names = tools_names(&req);
        let codex_search_count = names.iter().filter(|n| *n == "codex_search").count();
        assert_eq!(codex_search_count, 1);
        assert!(names.contains(&"codex_web".to_string()));
    }

    #[test]
    fn inject_no_duplicate_when_client_provides_both() {
        let mut req = make_request();
        req.extra.insert(
            "tools".to_string(),
            json!([
                {"type": "function", "function": {"name": "codex_search", "parameters": {"type": "object"}}},
                {"type": "function", "function": {"name": "codex_web", "parameters": {"type": "object"}}}
            ]),
        );
        ToolInjector::inject(&mut req, true, true);
        let names = tools_names(&req);
        let codex_search_count = names.iter().filter(|n| *n == "codex_search").count();
        let codex_web_count = names.iter().filter(|n| *n == "codex_web").count();
        assert_eq!(codex_search_count, 1);
        assert_eq!(codex_web_count, 1);
    }

    #[test]
    fn inject_creates_array_when_tools_null() {
        let mut req = make_request();
        req.extra.insert("tools".to_string(), Value::Null);
        ToolInjector::inject(&mut req, true, true);
        let names = tools_names(&req);
        assert!(names.contains(&"codex_search".to_string()));
        assert!(names.contains(&"codex_web".to_string()));
    }

    #[test]
    fn inject_creates_array_when_tools_empty() {
        let mut req = make_request();
        req.extra.insert("tools".to_string(), json!([]));
        ToolInjector::inject(&mut req, true, true);
        let names = tools_names(&req);
        assert!(names.contains(&"codex_search".to_string()));
        assert!(names.contains(&"codex_web".to_string()));
    }

    #[test]
    fn inject_preserves_other_client_tools() {
        let mut req = make_request();
        req.extra.insert(
            "tools".to_string(),
            json!([
                {"type": "function", "function": {"name": "custom_tool", "parameters": {"type": "object"}}}
            ]),
        );
        ToolInjector::inject(&mut req, true, true);
        let names = tools_names(&req);
        assert!(names.contains(&"custom_tool".to_string()));
        assert!(names.contains(&"codex_search".to_string()));
        assert!(names.contains(&"codex_web".to_string()));
    }

    #[test]
    fn inject_does_not_touch_non_array_tools() {
        let mut req = make_request();
        req.extra.insert("tools".to_string(), json!("something"));
        ToolInjector::inject(&mut req, true, true);
        assert_eq!(req.extra.get("tools"), Some(&json!("something")));
    }

    #[test]
    fn definitions_have_correct_names() {
        let search_name = ToolInjector::codex_search_definition()
            .get("function")
            .and_then(|f| f.get("name"))
            .and_then(|n| n.as_str());
        assert_eq!(search_name, Some("codex_search"));
        let web_name = ToolInjector::codex_web_definition()
            .get("function")
            .and_then(|f| f.get("name"))
            .and_then(|n| n.as_str());
        assert_eq!(web_name, Some("codex_web"));
    }

    mod property_tests {
        use super::*;
        use proptest::prelude::*;

        fn tool_strategy() -> impl Strategy<Value = Value> {
            prop_oneof![
                Just(json!({
                    "type": "function",
                    "function": {
                        "name": "codex_search",
                        "parameters": {"type": "object"}
                    }
                })),
                Just(json!({
                    "type": "function",
                    "function": {
                        "name": "codex_web",
                        "parameters": {"type": "object"}
                    }
                })),
                Just(json!({
                    "type": "function",
                    "function": {
                        "name": "codex_search"
                    }
                })),
                Just(json!({
                    "type": "function",
                    "function": {
                        "name": "other_tool",
                        "parameters": {"type": "object"}
                    }
                })),
            ]
        }

        proptest! {
            #![proptest_config(ProptestConfig::with_cases(100))]

            // Feature: codex-search, Property 3: Injection idempotence
            #[test]
            fn prop_injection_idempotence(
                existing in proptest::collection::vec(tool_strategy(), 0..=5)
            ) {
                let mut req = make_request();
                req.extra.insert("tools".to_string(), Value::Array(existing.clone()));
                let initial_search = existing
                    .iter()
                    .filter(|t| {
                        t.get("function")
                            .and_then(|f| f.get("name"))
                            .and_then(|n| n.as_str())
                            == Some("codex_search")
                    })
                    .count();
                let initial_web = existing
                    .iter()
                    .filter(|t| {
                        t.get("function")
                            .and_then(|f| f.get("name"))
                            .and_then(|n| n.as_str())
                            == Some("codex_web")
                    })
                    .count();
                ToolInjector::inject(&mut req, true, true);
                ToolInjector::inject(&mut req, true, true);

                let names = tools_names(&req);
                let codex_search_count = names.iter().filter(|n| *n == "codex_search").count();
                let codex_web_count = names.iter().filter(|n| *n == "codex_web").count();
                let expected_search = initial_search.max(1);
                let expected_web = initial_web.max(1);
                prop_assert_eq!(
                    codex_search_count,
                    expected_search,
                    "codex_search duplicated after double inject"
                );
                prop_assert_eq!(
                    codex_web_count,
                    expected_web,
                    "codex_web duplicated after double inject"
                );
            }

            // Feature: codex-search, Property 4: Injection inactive when OAuth unavailable
            #[test]
            fn prop_injection_noop_when_oauth_inactive(
                existing in proptest::collection::vec(tool_strategy(), 0..=5)
            ) {
                let mut req = make_request();
                req.extra.insert("tools".to_string(), Value::Array(existing.clone()));
                let original = req.extra.get("tools").cloned().unwrap();
                ToolInjector::inject(&mut req, false, true);
                prop_assert_eq!(
                    req.extra.get("tools"),
                    Some(&original),
                    "tools changed when OAuth inactive"
                );
            }
        }
    }
}
