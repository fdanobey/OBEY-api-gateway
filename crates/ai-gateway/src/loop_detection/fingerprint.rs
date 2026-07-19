use serde_json::Value;
use std::{
    collections::hash_map::DefaultHasher,
    hash::{Hash, Hasher},
};

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ToolCall {
    pub function_name: String,
    pub argument_keys: Vec<String>,
}

pub struct ToolCallFingerprint;

impl ToolCallFingerprint {
    pub fn compute(tool_calls: &[ToolCall]) -> Option<u64> {
        if tool_calls.is_empty() {
            return None;
        }

        let mut hasher = DefaultHasher::new();
        tool_calls.len().hash(&mut hasher);
        for tool_call in tool_calls {
            tool_call.function_name.hash(&mut hasher);
            let mut keys = tool_call.argument_keys.clone();
            keys.sort_unstable();
            keys.dedup();
            keys.hash(&mut hasher);
        }
        Some(hasher.finish())
    }

    pub fn from_json(tool_calls: &Value) -> Option<u64> {
        let tool_calls = tool_calls.as_array()?;
        let normalized = tool_calls
            .iter()
            .filter_map(|tool_call| {
                let function = tool_call.get("function").unwrap_or(tool_call);
                let function_name = function.get("name").and_then(Value::as_str)?.to_string();
                let argument_keys = argument_keys(function.get("arguments"));
                Some(ToolCall {
                    function_name,
                    argument_keys,
                })
            })
            .collect::<Vec<_>>();
        Self::compute(&normalized)
    }
}

#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub struct FingerprintTracker {
    last_fingerprint: Option<u64>,
    consecutive_count: u32,
}

impl FingerprintTracker {
    pub fn observe(&mut self, fingerprint: Option<u64>) -> f32 {
        let Some(fingerprint) = fingerprint else {
            return 0.0;
        };

        if self.last_fingerprint == Some(fingerprint) {
            self.consecutive_count = self.consecutive_count.saturating_add(1);
        } else {
            self.last_fingerprint = Some(fingerprint);
            self.consecutive_count = 1;
        }

        repetition_score(self.consecutive_count)
    }

    pub fn consecutive_count(&self) -> u32 {
        self.consecutive_count
    }

    pub fn last_fingerprint(&self) -> Option<u64> {
        self.last_fingerprint
    }
}

pub fn repetition_score(consecutive_count: u32) -> f32 {
    match consecutive_count {
        0 | 1 => 0.0,
        2 => 0.4,
        3 => 0.7,
        _ => 1.0,
    }
}

fn argument_keys(arguments: Option<&Value>) -> Vec<String> {
    match arguments {
        Some(Value::Object(arguments)) => arguments.keys().cloned().collect(),
        Some(Value::String(arguments)) => serde_json::from_str::<Value>(arguments)
            .ok()
            .and_then(|value| {
                value
                    .as_object()
                    .map(|object| object.keys().cloned().collect())
            })
            .unwrap_or_default(),
        _ => Vec::new(),
    }
}
