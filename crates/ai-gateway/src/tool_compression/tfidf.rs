//! Shared TF-IDF / BM25 scoring utility for tool compression.
//!
//! Used by:
//! - `SemanticRetriever` (fallback scoring when embedding latency budget is exceeded)
//! - `DescriptionCompressor` (token importance scoring for redundancy removal)

use std::collections::HashMap;

/// Splits text into lowercase tokens, filtering out tokens shorter than 2 chars.
/// Splits on whitespace and common punctuation: .,;:!?()[]{}/"'
fn tokenize(text: &str) -> Vec<String> {
    text.split(|c: char| c.is_whitespace() || ".,;:!?()[]{}/\"'".contains(c))
        .map(|s| s.to_lowercase())
        .filter(|s| s.len() >= 2)
        .collect()
}

/// TF-IDF and BM25 scorer built from a corpus of documents.
///
/// Vocabulary maps each unique token to the number of documents containing it.
pub struct TfIdfScorer {
    /// token → document frequency (number of documents containing the token)
    vocabulary: HashMap<String, u32>,
    /// Total number of documents in the corpus
    doc_count: u32,
    /// Average document length (in tokens) for BM25 normalization
    avg_doc_len: f32,
}

impl TfIdfScorer {
    /// Build vocabulary from a corpus of documents.
    ///
    /// Each document is tokenized by whitespace + punctuation splitting, lowercased,
    /// and tokens < 2 chars are filtered out.
    pub fn new(documents: &[&str]) -> Self {
        let mut vocabulary: HashMap<String, u32> = HashMap::new();
        let mut total_tokens: usize = 0;

        for doc in documents {
            let tokens = tokenize(doc);
            total_tokens += tokens.len();

            // Count each unique token once per document
            let mut seen: HashMap<&str, bool> = HashMap::new();
            for token in &tokens {
                if seen.insert(token.as_str(), true).is_none() {
                    *vocabulary.entry(token.clone()).or_insert(0) += 1;
                }
            }
        }

        let doc_count = documents.len() as u32;
        let avg_doc_len = if doc_count > 0 {
            total_tokens as f32 / doc_count as f32
        } else {
            1.0
        };

        Self {
            vocabulary,
            doc_count,
            avg_doc_len,
        }
    }

    /// Score each document against a query using BM25.
    ///
    /// Returns normalized scores in 0.0..=1.0 range (max normalization).
    /// Standard BM25 parameters: k1 = 1.2, b = 0.75.
    pub fn score_query(&self, query: &str, documents: &[&str]) -> Vec<f32> {
        const K1: f32 = 1.2;
        const B: f32 = 0.75;

        let query_terms = tokenize(query);

        if documents.is_empty() || query_terms.is_empty() {
            return vec![0.0; documents.len()];
        }

        let mut scores: Vec<f32> = Vec::with_capacity(documents.len());

        for doc in documents {
            let doc_tokens = tokenize(doc);
            let doc_len = doc_tokens.len() as f32;

            // Build term frequency map for this document
            let mut tf_map: HashMap<&str, u32> = HashMap::new();
            for token in &doc_tokens {
                *tf_map.entry(token.as_str()).or_insert(0) += 1;
            }

            let mut score: f32 = 0.0;
            for term in &query_terms {
                let tf = *tf_map.get(term.as_str()).unwrap_or(&0) as f32;
                if tf == 0.0 {
                    continue;
                }

                // IDF: log((N - df + 0.5) / (df + 0.5) + 1.0)
                let df = *self.vocabulary.get(term.as_str()).unwrap_or(&0) as f32;
                let n = self.doc_count as f32;
                let idf = ((n - df + 0.5) / (df + 0.5) + 1.0).ln();

                // BM25 TF component
                let tf_norm = (tf * (K1 + 1.0))
                    / (tf + K1 * (1.0 - B + B * doc_len / self.avg_doc_len));

                score += idf * tf_norm;
            }

            scores.push(score);
        }

        // Max normalization to [0.0, 1.0]
        let max_score = scores.iter().copied().fold(0.0_f32, f32::max);
        if max_score > 0.0 {
            for s in &mut scores {
                *s /= max_score;
            }
        }

        scores
    }

    /// Score each token in a description for importance relative to parameter vocabulary.
    ///
    /// Tokens that also appear in `parameter_names` or `parameter_types` are penalized
    /// (redundant with schema). Returns token → importance pairs preserving order.
    pub fn score_token_importance(
        &self,
        description: &str,
        parameter_names: &[&str],
        parameter_types: &[&str],
    ) -> Vec<(String, f32)> {
        let tokens = tokenize(description);
        if tokens.is_empty() {
            return Vec::new();
        }

        // Build a set of lowercased parameter vocabulary for redundancy check
        let param_vocab: std::collections::HashSet<String> = parameter_names
            .iter()
            .chain(parameter_types.iter())
            .flat_map(|s| tokenize(s))
            .collect();

        // Term frequency within this description
        let mut tf_map: HashMap<&str, u32> = HashMap::new();
        for token in &tokens {
            *tf_map.entry(token.as_str()).or_insert(0) += 1;
        }

        let doc_len = tokens.len() as f32;
        let mut results: Vec<(String, f32)> = Vec::with_capacity(tokens.len());

        for token in &tokens {
            let tf = *tf_map.get(token.as_str()).unwrap_or(&1) as f32;

            // IDF from corpus vocabulary
            let df = *self.vocabulary.get(token.as_str()).unwrap_or(&0) as f32;
            let n = self.doc_count as f32;
            let idf = if df > 0.0 && n > 0.0 {
                (n / df).ln() + 1.0
            } else {
                // Token not in corpus — treat as novel (high importance)
                (n + 1.0).ln() + 1.0
            };

            // Base TF-IDF score (normalized by doc length to keep scores comparable)
            let base_score = (tf / doc_len) * idf;

            // Penalize tokens that are redundant with parameter schema
            let importance = if param_vocab.contains(token.as_str()) {
                base_score * 0.3 // 70% reduction for redundant tokens
            } else {
                base_score
            };

            results.push((token.clone(), importance));
        }

        results
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_tokenize_basic() {
        let tokens = tokenize("Hello, world! This is a test.");
        assert_eq!(tokens, vec!["hello", "world", "this", "is", "test"]);
    }

    #[test]
    fn test_tokenize_filters_short() {
        let tokens = tokenize("a b cd ef");
        assert_eq!(tokens, vec!["cd", "ef"]);
    }

    #[test]
    fn test_new_empty_corpus() {
        let scorer = TfIdfScorer::new(&[]);
        assert_eq!(scorer.doc_count, 0);
        assert!(scorer.vocabulary.is_empty());
    }

    #[test]
    fn test_new_builds_vocabulary() {
        let docs = vec!["hello world", "world test", "hello test foo"];
        let scorer = TfIdfScorer::new(&docs);
        assert_eq!(scorer.doc_count, 3);
        assert_eq!(scorer.vocabulary.get("hello"), Some(&2));
        assert_eq!(scorer.vocabulary.get("world"), Some(&2));
        assert_eq!(scorer.vocabulary.get("test"), Some(&2));
        assert_eq!(scorer.vocabulary.get("foo"), Some(&1));
    }

    #[test]
    fn test_score_query_empty() {
        let scorer = TfIdfScorer::new(&["hello world"]);
        assert_eq!(scorer.score_query("test", &[]), Vec::<f32>::new());
        assert_eq!(scorer.score_query("", &["hello"]), vec![0.0]);
    }

    #[test]
    fn test_score_query_ranking() {
        let docs = vec![
            "search github repositories by name",
            "send slack message to channel",
            "get weather forecast for location",
        ];
        let scorer = TfIdfScorer::new(&docs);
        let scores = scorer.score_query("search repositories", &docs);

        // First document should score highest
        assert_eq!(scores.len(), 3);
        assert!(scores[0] > scores[1]);
        assert!(scores[0] > scores[2]);
        // Normalized: top score should be 1.0
        assert!((scores[0] - 1.0).abs() < f32::EPSILON);
    }

    #[test]
    fn test_score_query_no_match() {
        let docs = vec!["hello world", "foo bar"];
        let scorer = TfIdfScorer::new(&docs);
        let scores = scorer.score_query("xyz", &docs);
        assert_eq!(scores, vec![0.0, 0.0]);
    }

    #[test]
    fn test_score_token_importance_basic() {
        let docs = vec![
            "Search for GitHub repositories by name, language, and star count",
            "Send a message to a Slack channel",
            "Get current weather for a location",
        ];
        let scorer = TfIdfScorer::new(&docs);

        let results = scorer.score_token_importance(
            "Search for GitHub repositories by name, language, and star count",
            &["query", "language", "min_stars"],
            &["string", "integer"],
        );

        assert!(!results.is_empty());

        // "language" appears in parameter_names → should be penalized
        let language_score = results
            .iter()
            .find(|(t, _)| t == "language")
            .map(|(_, s)| *s)
            .unwrap();
        // "github" does NOT appear in params → should not be penalized
        let github_score = results
            .iter()
            .find(|(t, _)| t == "github")
            .map(|(_, s)| *s)
            .unwrap();

        // GitHub should score higher than language (language is redundant with params)
        assert!(github_score > language_score);
    }

    #[test]
    fn test_score_token_importance_empty() {
        let scorer = TfIdfScorer::new(&["hello world"]);
        let results = scorer.score_token_importance("", &[], &[]);
        assert!(results.is_empty());
    }
}
