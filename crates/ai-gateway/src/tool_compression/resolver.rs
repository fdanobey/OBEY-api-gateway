//! Resolution of synthetic drill-down tool calls emitted by the compression stages.
//!
//! When `progressive_disclosure` or `namespace_grouping` is active, the middleware
//! replaces tool definitions with minimal listings plus a synthetic drill-down tool
//! (`get_tool_schema` / `get_tools_in_namespace`). When the model calls one of these,
//! the middleware must resolve it against the original (full) tool definitions and feed
//! the result back so the model can actually invoke the underlying tool.
//!
//! This module is pure and unit-testable: it takes the provider's response JSON, the
//! (mutable) outbound request JSON, and the original tools, and returns the set of
//! tool names that were disclosed (so the caller can persist them for multi-turn use).

use serde_json::{json, Value};

use crate::tool_compression::stages::disclosure::resolve_get_tool_schema;
use crate::tool_compression::types::ToolDefinition;

/// Synthetic tool that returns the full parameter schema for a single tool.
pub const GET_TOOL_SCHEMA: &str = "get_tool_schema";
/// Synthetic tool that returns the full schemas for every tool in a namespace.
pub const GET_TOOLS_IN_NAMESPACE: &str = "get_tools_in_namespace";
/// Prefix for namespace summary pseudo-tools (`ns_github`, `ns_other`, …).
/// Models sometimes call these directly; the resolver treats them as an implicit
/// `get_tools_in_namespace` with the suffix as the namespace argument.
pub const NS_PREFIX: &str = "ns_";

/// Returns the namespace prefix of a tool name (first segment before `_` or `.`).
/// Tools without a separator belong to the implicit `"other"` namespace.
/// Mirrors `NamespaceGrouper::extract_prefix`.
pub fn namespace_of(name: &str) -> Option<String> {
    name.find(|c: char| c == '_' || c == '.')
        .map(|pos| name[..pos].to_string())
}

/// All original tool definitions belonging to `ns` (or the `"other"` bucket when
/// `ns == "other"`). Used to re-inject full schemas into the outgoing `tools` array
/// so the model can call them after disclosure.
pub fn tools_in_namespace(ns: &str, original_tools: &[ToolDefinition]) -> Vec<Value> {
    original_tools
        .iter()
        .filter(|t| match namespace_of(&t.name) {
            Some(p) => p == ns,
            None => ns == "other",
        })
        .map(|t| t.raw.clone())
        .collect()
}

/// Resolve a single synthetic drill-down tool call.
///
/// `arguments` is the JSON string passed by the model. Returns the content string
/// to place in a `tool` result message, or `None` if `name` is not a synthetic tool
/// or the arguments are malformed.
pub fn resolve_synthetic_tool_call(
    name: &str,
    arguments: &str,
    original_tools: &[ToolDefinition],
) -> Option<String> {
    match name {
        GET_TOOL_SCHEMA => {
            let parsed: Value = serde_json::from_str(arguments).ok()?;
            let tool_name = parsed.get("tool_name").and_then(|v| v.as_str())?;
            let schema = resolve_get_tool_schema(tool_name, original_tools).ok()?;
            Some(serde_json::to_string_pretty(&schema).unwrap_or_default())
        }
        GET_TOOLS_IN_NAMESPACE => {
            let parsed: Value = serde_json::from_str(arguments).ok()?;
            let ns = parsed.get("namespace").and_then(|v| v.as_str())?;
            let schemas = tools_in_namespace(ns, original_tools);
            Some(serde_json::to_string_pretty(&schemas).unwrap_or_default())
        }
        _ if name.starts_with(NS_PREFIX) => {
            // Model called a namespace summary pseudo-tool directly (e.g. `ns_other`).
            // Treat it as get_tools_in_namespace for the suffix namespace.
            let ns = &name[NS_PREFIX.len()..];
            let schemas = tools_in_namespace(ns, original_tools);
            Some(serde_json::to_string_pretty(&schemas).unwrap_or_default())
        }
        _ => None,
    }
}

/// Inspect the provider's response and, if the assistant message contains synthetic
/// drill-down tool calls, resolve them against `original_tools`, feed the results
/// back (as `tool` messages) into `req_json["messages"]`, and re-inject the disclosed
/// full tool schemas into `req_json["tools"]` so the model can call them on the next
/// model turn.
///
/// `max_reinject` bounds how many tool schemas are re-injected into `req_json["tools"]`
/// for a single `get_tools_in_namespace` drill-down. This keeps the provider below its
/// tool-count limit (e.g. "maximum of 100 tools allowed"): when a namespace holds more
/// tools than `max_reinject` (and `max_reinject > 0`), only the first `max_reinject` are
/// made callable this turn and the tool result notes the remainder. `0` disables the cap.
///
/// Returns `Some(disclosed_tool_names)` when at least one synthetic call was resolved
/// (the caller should persist these for multi-turn disclosure), or `None` otherwise.
pub fn resolve_synthetic_in_response(
    resp_json: &Value,
    req_json: &mut Value,
    original_tools: &[ToolDefinition],
    max_reinject: u32,
) -> Option<Vec<String>> {
    let choices = resp_json.get("choices").and_then(|c| c.as_array())?;
    let first = choices.first()?;
    let message = first.get("message")?;
    let tool_calls = message.get("tool_calls").and_then(|t| t.as_array())?;

    let mut disclosed: Vec<String> = Vec::new();
    let mut tool_results: Vec<Value> = Vec::new();
    let mut injected: Vec<Value> = Vec::new();

    for tc in tool_calls {
        let Some(fn_obj) = tc.get("function") else {
            continue;
        };
        let Some(name) = fn_obj.get("name").and_then(|n| n.as_str()) else {
            continue;
        };
        if name != GET_TOOL_SCHEMA && name != GET_TOOLS_IN_NAMESPACE && !name.starts_with(NS_PREFIX) {
            continue;
        }
        let args = fn_obj.get("arguments").and_then(|a| a.as_str()).unwrap_or("{}");
        let call_id = tc.get("id").and_then(|i| i.as_str()).unwrap_or("");

        // Resolve the synthetic call. If resolution fails (e.g. invalid namespace or
        // malformed arguments), produce an error result instead of aborting the entire
        // resolution loop — the model needs a tool response for its tool_call_id.
        let mut content = match resolve_synthetic_tool_call(name, args, original_tools) {
            Some(c) => c,
            None => {
                // Build a helpful error so the model can recover.
                let available: Vec<&str> = original_tools.iter().map(|t| t.name.as_str()).collect();
                format!(
                    "Error resolving {}: invalid arguments or unknown target. Available tools: {}",
                    name,
                    available.join(", ")
                )
            }
        };

        // Record which original tools are now disclosed.
        match name {
            GET_TOOL_SCHEMA => {
                if let Ok(parsed) = serde_json::from_str::<Value>(args) {
                    if let Some(tn) = parsed.get("tool_name").and_then(|v| v.as_str()) {
                        disclosed.push(tn.to_string());
                    }
                }
            }
            GET_TOOLS_IN_NAMESPACE => {
                if let Ok(parsed) = serde_json::from_str::<Value>(args) {
                    if let Some(ns) = parsed.get("namespace").and_then(|v| v.as_str()) {
                        for t in tools_in_namespace(ns, original_tools) {
                            if let Some(n) = t.pointer("/function/name").and_then(|v| v.as_str()) {
                                disclosed.push(n.to_string());
                            }
                            injected.push(t);
                        }
                        // Cap re-injection so the provider stays under its tool-count limit
                        // (e.g. "maximum of 100 tools allowed"). The model still sees every
                        // schema in `content`; only the callable set is bounded this turn.
                        if max_reinject > 0 && injected.len() > max_reinject as usize {
                            let excess = injected.len() - max_reinject as usize;
                            injected.truncate(max_reinject as usize);
                            disclosed.truncate(max_reinject as usize);
                            content.push_str(&format!(
                                "\n\nNote: {} additional tool(s) in namespace '{}' were omitted from the callable tool list to stay within the provider's tool-count limit. Request a specific tool with get_tool_schema to make it callable this turn.",
                                excess, ns
                            ));
                        }
                    }
                }
            }
            _ if name.starts_with(NS_PREFIX) => {
                // Model called ns_<prefix> directly — disclose the same tools as
                // get_tools_in_namespace for the suffix.
                let ns = &name[NS_PREFIX.len()..];
                for t in tools_in_namespace(ns, original_tools) {
                    if let Some(n) = t.pointer("/function/name").and_then(|v| v.as_str()) {
                        disclosed.push(n.to_string());
                    }
                    injected.push(t);
                }
                if max_reinject > 0 && injected.len() > max_reinject as usize {
                    let excess = injected.len() - max_reinject as usize;
                    injected.truncate(max_reinject as usize);
                    disclosed.truncate(max_reinject as usize);
                    content.push_str(&format!(
                        "\n\nNote: {} additional tool(s) in namespace '{}' were omitted from the callable tool list to stay within the provider's tool-count limit. Request a specific tool with get_tool_schema to make it callable this turn.",
                        excess, ns
                    ));
                }
            }
            _ => {}
        }

        tool_results.push(json!({
            "role": "tool",
            "tool_call_id": call_id,
            "content": content,
        }));
    }

    if tool_results.is_empty() {
        return None;
    }

    // Mirror the assistant message that issued the synthetic call, then append results.
    if let Some(messages) = req_json.get_mut("messages").and_then(|m| m.as_array_mut()) {
        messages.push(message.clone());
        for m in tool_results {
            messages.push(m);
        }
    }

    // Re-inject disclosed full tool schemas into the tools array so they are callable.
    if !injected.is_empty() {
        if let Some(tools) = req_json.get_mut("tools").and_then(|t| t.as_array_mut()) {
            for t in injected {
                tools.push(t);
            }
        }
    }

    Some(disclosed)
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    fn make_tool(name: &str) -> ToolDefinition {
        ToolDefinition {
            raw: json!({
                "type": "function",
                "function": {
                    "name": name,
                    "description": "d",
                    "parameters": {"type": "object", "properties": {}}
                }
            }),
            name: name.to_string(),
            content_hash: 0,
        }
    }

    #[test]
    fn namespace_of_splits_on_sep() {
        assert_eq!(namespace_of("fs_read"), Some("fs".to_string()));
        assert_eq!(namespace_of("git.commit"), Some("git".to_string()));
        assert_eq!(namespace_of("tool"), None);
    }

    #[test]
    fn tools_in_namespace_filters() {
        let tools = vec![make_tool("fs_read"), make_tool("fs_write"), make_tool("git_log")];
        let fs = tools_in_namespace("fs", &tools);
        assert_eq!(fs.len(), 2);
        let other = tools_in_namespace("other", &tools);
        assert_eq!(other.len(), 0);
    }

    #[test]
    fn resolves_single_schema() {
        let tools = vec![make_tool("fs_read")];
        let out = resolve_synthetic_tool_call(
            "get_tool_schema",
            r#"{"tool_name":"fs_read"}"#,
            &tools,
        )
        .expect("should resolve");
        assert!(out.contains("fs_read"));
    }

    #[test]
    fn resolves_namespace() {
        let tools = vec![make_tool("fs_read"), make_tool("fs_write"), make_tool("git_log")];
        let out = resolve_synthetic_tool_call(
            "get_tools_in_namespace",
            r#"{"namespace":"fs"}"#,
            &tools,
        )
        .expect("should resolve");
        assert!(out.contains("fs_read"));
        assert!(out.contains("fs_write"));
        assert!(!out.contains("git_log"));
    }

    #[test]
    fn resolve_in_response_injects_messages_and_tools() {
        let tools = vec![make_tool("fs_read"), make_tool("fs_write")];
        let resp = json!({
            "choices": [{
                "message": {
                    "role": "assistant",
                    "content": null,
                    "tool_calls": [{
                        "id": "call_1",
                        "type": "function",
                        "function": {"name": "get_tools_in_namespace", "arguments": "{\"namespace\":\"fs\"}"}
                    }]
                }
            }]
        });
        let mut req = json!({
            "tools": [{"type":"function","function":{"name":"get_tools_in_namespace"}}],
            "messages": [{"role":"user","content":"list files"}]
        });
        let disclosed = resolve_synthetic_in_response(&resp, &mut req, &tools, 0).expect("handled");
        assert_eq!(disclosed.len(), 2);

        let msgs = req["messages"].as_array().unwrap();
        assert_eq!(msgs.len(), 3); // user + assistant + tool result
        assert_eq!(msgs[2]["role"], "tool");
        assert_eq!(msgs[2]["tool_call_id"], "call_1");

        let tools_arr = req["tools"].as_array().unwrap();
        assert!(tools_arr.iter().any(|t| t.pointer("/function/name") == Some(&json!("fs_read"))));
    }

    #[test]
    fn resolve_in_response_caps_reinjection_to_max() {
        // 10 tools in the "fs" namespace; cap re-injection at 3 so a provider with a
        // small tool-count limit is never exceeded on drill-down.
        let tools: Vec<ToolDefinition> = (1..=10)
            .map(|i| make_tool(&format!("fs_t{}", i)))
            .collect();
        let resp = json!({
            "choices": [{
                "message": {
                    "role": "assistant",
                    "content": null,
                    "tool_calls": [{
                        "id": "call_1",
                        "type": "function",
                        "function": {"name": "get_tools_in_namespace", "arguments": "{\"namespace\":\"fs\"}"}
                    }]
                }
            }]
        });
        let mut req = json!({
            "tools": [{"type":"function","function":{"name":"get_tools_in_namespace"}}],
            "messages": [{"role":"user","content":"list files"}]
        });
        let disclosed = resolve_synthetic_in_response(&resp, &mut req, &tools, 3).expect("handled");
        assert_eq!(disclosed.len(), 3, "only capped number should be disclosed");

        let tools_arr = req["tools"].as_array().unwrap();
        let reinjected = tools_arr
            .iter()
            .filter(|t| t.pointer("/function/name").map(|n| n.as_str().unwrap_or("")).map(|n| n.starts_with("fs_")).unwrap_or(false))
            .count();
        assert_eq!(reinjected, 3, "provider must see at most the cap of callable tools");

        // Full content still lists every tool, with a truncation note.
        let tool_msg = &req["messages"].as_array().unwrap()[2]["content"];
        assert!(tool_msg.as_str().unwrap().contains("7 additional tool(s)"));
    }

    #[test]
    fn resolve_in_response_none_when_no_synthetic() {
        let tools = vec![make_tool("fs_read")];
        let resp = json!({
            "choices": [{
                "message": {
                    "role": "assistant",
                    "content": "hello",
                    "tool_calls": [{
                        "id": "call_1",
                        "type": "function",
                        "function": {"name": "fs_read", "arguments": "{}"}
                    }]
                }
            }]
        });
        let mut req = json!({"tools":[], "messages":[]});
        assert!(resolve_synthetic_in_response(&resp, &mut req, &tools, 0).is_none());
    }
}
