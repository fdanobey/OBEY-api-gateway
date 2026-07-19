use serde_json::Value;
use std::{
    collections::hash_map::DefaultHasher,
    hash::{Hash, Hasher},
};

const STOP_WORDS: &[&str] = &[
    "a", "an", "and", "are", "as", "at", "be", "but", "by", "for", "from", "has", "he", "her",
    "hers", "him", "his", "i", "if", "in", "into", "is", "it", "its", "me", "my", "no", "not",
    "of", "on", "or", "our", "ours", "she", "so", "that", "the", "their", "theirs", "them", "they",
    "this", "to", "up", "was", "we", "were", "what", "when", "where", "which", "who", "will",
    "with", "you", "your", "yours",
];

pub fn compute(content: &str) -> u64 {
    let tokens = tokenize(content);
    if tokens.is_empty() {
        return 0;
    }

    let features = shingle(&tokens);
    let mut accumulators = [0i32; 64];
    for feature in features {
        let hash = stable_hash(&feature);
        for (bit, accumulator) in accumulators.iter_mut().enumerate() {
            if hash & (1u64 << bit) == 0 {
                *accumulator -= 1;
            } else {
                *accumulator += 1;
            }
        }
    }

    accumulators
        .iter()
        .enumerate()
        .fold(0u64, |hash, (bit, accumulator)| {
            if *accumulator >= 0 {
                hash | (1u64 << bit)
            } else {
                hash
            }
        })
}

pub fn compute_messages(messages: &Value) -> u64 {
    let Some(messages) = messages.as_array() else {
        return 0;
    };
    let content = messages
        .iter()
        .filter(|message| message.get("role").and_then(Value::as_str) != Some("tool"))
        .filter_map(|message| message.get("content"))
        .filter_map(extract_text_content)
        .collect::<Vec<_>>()
        .join(" ");
    compute(&content)
}

pub fn hamming_distance(left: u64, right: u64) -> u32 {
    (left ^ right).count_ones()
}

pub fn similarity(left: u64, right: u64) -> f32 {
    1.0 - hamming_distance(left, right) as f32 / 64.0
}

fn tokenize(content: &str) -> Vec<String> {
    content
        .split_whitespace()
        .map(|token| token.to_lowercase())
        .map(|token| {
            token
                .trim_matches(|character: char| !character.is_alphanumeric() && character != '_')
                .to_string()
        })
        .filter(|token| !token.is_empty())
        .filter(|token| STOP_WORDS.binary_search(&token.as_str()).is_err())
        .collect()
}

fn shingle(tokens: &[String]) -> Vec<String> {
    if tokens.len() < 3 {
        return tokens.to_vec();
    }
    tokens
        .windows(3)
        .map(|window| window.join("\u{1f}"))
        .collect()
}

fn stable_hash<T: Hash>(value: &T) -> u64 {
    let mut hasher = DefaultHasher::new();
    value.hash(&mut hasher);
    hasher.finish()
}

fn extract_text_content(content: &Value) -> Option<String> {
    if let Some(text) = content.as_str() {
        return Some(text.to_string());
    }
    let blocks = content.as_array()?;
    let text = blocks
        .iter()
        .filter(|block| block.get("type").and_then(Value::as_str) != Some("tool_result"))
        .filter_map(|block| block.get("text").and_then(Value::as_str))
        .collect::<Vec<_>>()
        .join(" ");
    (!text.is_empty()).then_some(text)
}
