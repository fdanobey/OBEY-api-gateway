//! Serde request/response types for the Codex Search feature.

use serde::{Deserialize, Serialize};

/// Arguments for the `codex_search` tool.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct CodexSearchArgs {
    pub q: String,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub domains: Option<Vec<String>>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub recency: Option<u32>,
}

/// Arguments for the `codex_web` tool.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct CodexWebArgs {
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub session_id: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub commands: Option<CodexWebCommands>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub response_length: Option<ResponseLength>,
}

/// Command object for `codex_web`.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct CodexWebCommands {
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub search_query: Option<Vec<SearchQueryCommand>>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub open: Option<Vec<OpenCommand>>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub find: Option<Vec<FindCommand>>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub click: Option<Vec<ClickCommand>>,
}

/// A single search-query command.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct SearchQueryCommand {
    pub q: String,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub domains: Option<Vec<String>>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub recency: Option<u32>,
}

/// An open command referencing a result by `ref_id`.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct OpenCommand {
    pub ref_id: String,
}

/// A find-in-page command.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct FindCommand {
    pub ref_id: String,
    pub pattern: String,
}

/// A click command referencing a result by `ref_id`.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct ClickCommand {
    pub ref_id: String,
}

/// Response length hint for `codex_web`. Known values are `short`, `medium`,
/// `long`. Unknown string values are preserved verbatim for forward
/// compatibility.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ResponseLength(String);

impl ResponseLength {
    pub fn short() -> Self {
        Self("short".to_string())
    }
    pub fn medium() -> Self {
        Self("medium".to_string())
    }
    pub fn long() -> Self {
        Self("long".to_string())
    }
    pub fn as_str(&self) -> &str {
        &self.0
    }
    pub fn from_string(s: String) -> Self {
        Self(s)
    }
}

impl Serialize for ResponseLength {
    fn serialize<S>(&self, serializer: S) -> Result<S::Ok, S::Error>
    where
        S: serde::Serializer,
    {
        serializer.serialize_str(&self.0)
    }
}

impl<'de> Deserialize<'de> for ResponseLength {
    fn deserialize<D>(deserializer: D) -> Result<Self, D::Error>
    where
        D: serde::Deserializer<'de>,
    {
        let s = String::deserialize(deserializer)?;
        Ok(Self(s))
    }
}

/// Upstream search API request body.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct CodexSearchRequest {
    pub model: String,
    pub session_id: String,
    pub commands: CodexSearchRequestCommands,
    #[serde(flatten)]
    pub extra: serde_json::Map<String, serde_json::Value>,
}

/// Commands for the upstream search request.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct CodexSearchRequestCommands {
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub search_query: Option<Vec<SearchQueryCommand>>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub open: Option<Vec<OpenCommand>>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub find: Option<Vec<FindCommand>>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub click: Option<Vec<ClickCommand>>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub response_length: Option<ResponseLength>,
    #[serde(flatten)]
    pub extra: serde_json::Map<String, serde_json::Value>,
}

/// Upstream search response (opaque, untrusted).
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct CodexSearchResponse {
    #[serde(flatten)]
    pub extra: serde_json::Map<String, serde_json::Value>,
}

/// Result returned by the search executor.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct ToolResult {
    pub content: String,
    pub is_error: bool,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub session_id: Option<String>,
}

#[cfg(test)]
mod property_tests {
    use super::*;
    use proptest::collection::vec;
    use proptest::option;
    use proptest::prelude::*;

    fn nonempty_id() -> impl Strategy<Value = String> {
        "[a-zA-Z0-9_\\-]{1,64}"
    }

    fn query_text() -> impl Strategy<Value = String> {
        "[a-z0-9 .,!?]{1,200}"
    }

    fn domains_strategy() -> impl Strategy<Value = Vec<String>> {
        vec(nonempty_id(), 0..5)
    }

    fn search_query_command_strategy() -> impl Strategy<Value = SearchQueryCommand> {
        (
            query_text(),
            option::of(domains_strategy()),
            option::of(0u32..10000),
        )
            .prop_map(|(q, domains, recency)| SearchQueryCommand {
                q,
                domains,
                recency,
            })
    }

    fn open_command_strategy() -> impl Strategy<Value = OpenCommand> {
        nonempty_id().prop_map(|ref_id| OpenCommand { ref_id })
    }

    fn find_command_strategy() -> impl Strategy<Value = FindCommand> {
        (nonempty_id(), query_text()).prop_map(|(ref_id, pattern)| FindCommand { ref_id, pattern })
    }

    fn click_command_strategy() -> impl Strategy<Value = ClickCommand> {
        nonempty_id().prop_map(|ref_id| ClickCommand { ref_id })
    }

    fn response_length_strategy() -> impl Strategy<Value = ResponseLength> {
        any::<String>().prop_map(ResponseLength::from_string)
    }

    fn search_request_commands_strategy() -> impl Strategy<Value = CodexSearchRequestCommands> {
        (
            option::of(vec(search_query_command_strategy(), 0..4)),
            option::of(vec(open_command_strategy(), 0..4)),
            option::of(vec(find_command_strategy(), 0..4)),
            option::of(vec(click_command_strategy(), 0..4)),
            option::of(response_length_strategy()),
        )
            .prop_map(|(search_query, open, find, click, response_length)| {
                CodexSearchRequestCommands {
                    search_query,
                    open,
                    find,
                    click,
                    response_length,
                    extra: serde_json::Map::new(),
                }
            })
    }

    fn search_request_strategy() -> impl Strategy<Value = CodexSearchRequest> {
        (
            nonempty_id(),
            nonempty_id(),
            search_request_commands_strategy(),
        )
            .prop_map(|(model, session_id, commands)| CodexSearchRequest {
                model,
                session_id,
                commands,
                extra: serde_json::Map::new(),
            })
    }

    fn json_value_strategy() -> impl Strategy<Value = serde_json::Value> {
        prop_oneof![
            any::<String>().prop_map(serde_json::Value::String),
            any::<bool>().prop_map(serde_json::Value::Bool),
            any::<i64>().prop_map(|n| serde_json::Value::Number(serde_json::Number::from(n))),
            any::<f64>().prop_map(|n| {
                serde_json::Number::from_f64(n)
                    .map(serde_json::Value::Number)
                    .unwrap_or(serde_json::Value::Null)
            }),
        ]
    }

    fn json_object_strategy(
        max_depth: u32,
    ) -> impl Strategy<Value = serde_json::Map<String, serde_json::Value>> {
        prop_oneof![
            json_value_strategy().prop_map(|v| {
                let mut m = serde_json::Map::new();
                m.insert("k".to_string(), v);
                m
            }),
            if max_depth == 0 {
                json_value_strategy()
                    .prop_map(|v| {
                        let mut m = serde_json::Map::new();
                        m.insert("k".to_string(), v);
                        m
                    })
                    .boxed()
            } else {
                json_object_strategy(max_depth - 1)
                    .prop_map(|inner| {
                        let mut m = serde_json::Map::new();
                        m.insert("nested".to_string(), serde_json::Value::Object(inner));
                        m
                    })
                    .boxed()
            },
        ]
    }

    proptest! {
        #![proptest_config(ProptestConfig::with_cases(100))]

        // Feature: codex-search, Property 1: Request serialization round-trip
        #[test]
        fn prop_request_round_trip(req in search_request_strategy()) {
            let json = serde_json::to_value(&req).unwrap();
            let back: CodexSearchRequest = serde_json::from_value(json).unwrap();
            prop_assert_eq!(req, back);
        }

        // Feature: codex-search, Property 2: Response JSON round-trip
        #[test]
        fn prop_response_round_trip(obj in json_object_strategy(3)) {
            let original = serde_json::Value::Object(obj.clone());
            let resp: CodexSearchResponse = serde_json::from_value(original).unwrap();
            let serialized = serde_json::to_value(&resp).unwrap();
            if let serde_json::Value::Object(serialized_obj) = serialized {
                for (key, value) in &obj {
                    prop_assert!(
                        serialized_obj.contains_key(key),
                        "missing key: {}",
                        key
                    );
                    prop_assert_eq!(serialized_obj.get(key), Some(value));
                }
            } else {
                prop_assert!(false, "serialized response is not an object");
            }
        }

        // Feature: codex-search, Property 8: Absent optional fields remain absent
        #[test]
        fn prop_absent_optional_fields(q in query_text()) {
            let args = CodexSearchArgs {
                q,
                domains: None,
                recency: None,
            };
            let json = serde_json::to_value(&args).unwrap();
            if let serde_json::Value::Object(map) = &json {
                prop_assert!(
                    !map.contains_key("domains"),
                    "domains should be absent when None"
                );
                prop_assert!(
                    !map.contains_key("recency"),
                    "recency should be absent when None"
                );
            } else {
                prop_assert!(false, "serialized args is not an object");
            }
        }

        // Feature: codex-search, Property 9: Unknown enum values survive round-trip
        #[test]
        fn prop_unknown_enum_values(s in "[a-zA-Z0-9_\\-]{1,40}") {
            let rl = ResponseLength::from_string(s.clone());
            let json = serde_json::to_value(&rl).unwrap();
            let back: ResponseLength = serde_json::from_value(json).unwrap();
            prop_assert_eq!(back.as_str(), s);
        }
    }
}
