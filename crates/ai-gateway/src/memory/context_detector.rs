//! Automatic project and agent context detection for persistent memory.

use std::sync::LazyLock;

use regex::Regex;
use sha2::{Digest, Sha256};

use crate::models::openai::{Message, OpenAIRequest};

use super::ContextType;

const MIN_SYSTEM_PROMPT_CHARS: usize = 200;
const MAX_FINGERPRINT_CHARS: usize = 500;
pub const MAX_CONTEXT_LABEL_CHARS: usize = 64;

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct DetectedContext {
    pub context: ContextType,
    pub display_name: Option<String>,
}

static WINDOWS_PATH_RE: LazyLock<Regex> = LazyLock::new(|| {
    Regex::new(r#"(?i)(?:^|[^A-Za-z0-9_])([A-Z]:\\[^\s\x00-\x1f<>|?*"]+)"#)
        .expect("Windows path regex must compile")
});

static UNIX_PATH_RE: LazyLock<Regex> = LazyLock::new(|| {
    Regex::new(r#"(?:^|[^A-Za-z0-9_])(/[^\s/\x00-\x1f<>|?*"]+/[^\s\x00-\x1f<>|?*"]+)"#)
        .expect("Unix path regex must compile")
});

static RELATIVE_PATH_RE: LazyLock<Regex> = LazyLock::new(|| {
    Regex::new(r"(?:^|[^A-Za-z0-9_])(?:src|crates|packages)[/\\][^\s\x00-\x1f]+")
        .expect("relative path regex must compile")
});

/// Detects project, agent, or user context from OpenAI chat messages.
#[derive(Debug, Clone)]
pub struct ContextDetector {
    default_prompts: Vec<String>,
}

impl ContextDetector {
    /// Creates a detector and normalizes configured default prompts once.
    pub fn new(default_prompts: Vec<String>) -> Self {
        Self {
            default_prompts: default_prompts
                .into_iter()
                .map(|prompt| normalize_whitespace(&prompt))
                .collect(),
        }
    }

    /// Detects context from an OpenAI request.
    pub fn detect(&self, request: &OpenAIRequest) -> ContextType {
        self.detect_with_label(request).context
    }

    pub fn detect_with_label(&self, request: &OpenAIRequest) -> DetectedContext {
        self.detect_messages_with_label(&request.messages)
    }

    /// Detects context from a message slice.
    pub fn detect_messages(&self, messages: &[Message]) -> ContextType {
        self.detect_messages_with_label(messages).context
    }

    pub fn detect_messages_with_label(&self, messages: &[Message]) -> DetectedContext {
        let message_texts: Vec<String> = messages.iter().map(Message::content_as_text).collect();

        if let Some((project_hash, display_name)) = detect_project(&message_texts) {
            return DetectedContext {
                context: ContextType::Project(project_hash),
                display_name,
            };
        }

        DetectedContext {
            context: self.detect_agent(messages).unwrap_or(ContextType::User),
            display_name: None,
        }
    }

    fn detect_agent(&self, messages: &[Message]) -> Option<ContextType> {
        let system_prompt = messages
            .iter()
            .filter(|message| message.role.eq_ignore_ascii_case("system"))
            .map(Message::content_as_text)
            .collect::<Vec<_>>()
            .join("\n");
        let normalized = normalize_whitespace(&system_prompt);

        if normalized.chars().count() <= MIN_SYSTEM_PROMPT_CHARS
            || self
                .default_prompts
                .iter()
                .any(|prompt| prompt == &normalized)
        {
            return None;
        }

        let fingerprint_input: String = normalized.chars().take(MAX_FINGERPRINT_CHARS).collect();
        Some(ContextType::Agent(short_hash(&fingerprint_input)))
    }
}

impl Default for ContextDetector {
    fn default() -> Self {
        Self::new(Vec::new())
    }
}

fn detect_project(message_texts: &[String]) -> Option<(String, Option<String>)> {
    let mut windows_paths = Vec::new();
    let mut unix_paths = Vec::new();

    for text in message_texts {
        windows_paths.extend(extract_paths(text, &WINDOWS_PATH_RE, PathStyle::Windows));
        unix_paths.extend(extract_paths(text, &UNIX_PATH_RE, PathStyle::Unix));
        let _has_relative_marker = RELATIVE_PATH_RE.is_match(text);
    }

    project_prefix(&windows_paths, PathStyle::Windows)
        .or_else(|| project_prefix(&unix_paths, PathStyle::Unix))
        .map(|prefix| {
            let display_name = project_basename(&prefix);
            (short_hash(&prefix), display_name)
        })
}

fn project_basename(prefix: &str) -> Option<String> {
    let candidate = prefix
        .trim_end_matches(['/', '\\'])
        .rsplit(['/', '\\'])
        .next()?;
    sanitize_label(candidate)
}

pub fn sanitize_label(value: &str) -> Option<String> {
    let sanitized: String = value
        .chars()
        .filter(|character| {
            character.is_ascii_alphanumeric() || matches!(character, ' ' | '_' | '-')
        })
        .take(MAX_CONTEXT_LABEL_CHARS)
        .collect();
    let sanitized = sanitized.trim().to_owned();
    (!sanitized.is_empty()).then_some(sanitized)
}

#[derive(Clone, Copy)]
enum PathStyle {
    Windows,
    Unix,
}

fn extract_paths(text: &str, regex: &Regex, style: PathStyle) -> Vec<Vec<String>> {
    regex
        .captures_iter(text)
        .filter_map(|captures| captures.get(1))
        .filter_map(|matched| path_directory_components(matched.as_str(), style))
        .collect()
}

fn path_directory_components(path: &str, style: PathStyle) -> Option<Vec<String>> {
    let trimmed = path.trim_end_matches(|character: char| {
        matches!(character, '.' | ',' | ';' | ':' | '!' | ')' | ']' | '}')
    });
    let separator = match style {
        PathStyle::Windows => '\\',
        PathStyle::Unix => '/',
    };
    let mut components: Vec<String> = trimmed
        .split(separator)
        .filter(|component| !component.is_empty())
        .map(str::to_owned)
        .collect();

    match style {
        PathStyle::Windows => {
            let drive = components.first_mut()?;
            if drive.len() != 2 || !drive.ends_with(':') {
                return None;
            }
            drive.make_ascii_uppercase();
        }
        PathStyle::Unix => {}
    }

    if components.len() < 2 {
        return None;
    }

    components.pop();
    Some(components)
}

fn project_prefix(paths: &[Vec<String>], style: PathStyle) -> Option<String> {
    if paths.len() < 2 {
        return None;
    }

    let mut common_length = paths[0].len();
    for path in &paths[1..] {
        common_length = paths[0]
            .iter()
            .zip(path)
            .take(common_length)
            .take_while(|(left, right)| components_equal(left, right, style))
            .count();
    }

    let minimum_components = match style {
        PathStyle::Windows => 2,
        PathStyle::Unix => 1,
    };
    if common_length < minimum_components {
        return None;
    }

    let common = &paths[0][..common_length];
    Some(match style {
        PathStyle::Windows => {
            let normalized = common
                .iter()
                .map(|component| component.to_lowercase())
                .collect::<Vec<_>>()
                .join("/");
            format!("{normalized}/")
        }
        PathStyle::Unix => format!("/{}/", common.join("/")),
    })
}

fn components_equal(left: &str, right: &str, style: PathStyle) -> bool {
    match style {
        PathStyle::Windows => left.eq_ignore_ascii_case(right),
        PathStyle::Unix => left == right,
    }
}

fn normalize_whitespace(value: &str) -> String {
    value.split_whitespace().collect::<Vec<_>>().join(" ")
}

fn short_hash(value: &str) -> String {
    let digest = Sha256::digest(value.as_bytes());
    let mut hash = String::with_capacity(16);
    for byte in &digest[..8] {
        use std::fmt::Write;
        write!(&mut hash, "{byte:02x}").expect("writing to a String cannot fail");
    }
    hash
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::{json, Map, Value};

    fn message(role: &str, content: impl Into<Value>) -> Message {
        Message {
            role: role.to_owned(),
            content: content.into(),
            extra: Map::new(),
        }
    }

    fn request(messages: Vec<Message>) -> OpenAIRequest {
        OpenAIRequest {
            model: "test-model".to_owned(),
            messages,
            stream: false,
            temperature: None,
            max_tokens: None,
            extra: Map::new(),
        }
    }

    fn project_hash(context: ContextType) -> String {
        match context {
            ContextType::Project(hash) => hash,
            other => panic!("expected project context, got {other:?}"),
        }
    }

    #[test]
    fn labels_use_only_bounded_project_basename() {
        let detector = ContextDetector::default();
        let detected = detector.detect_messages_with_label(&[
            message("user", r"C:\\Users\\secret\\Safe Project!\\src\\main.rs"),
            message("user", r"C:\\Users\\secret\\Safe Project!\\tests\\api.rs"),
        ]);
        assert!(matches!(detected.context, ContextType::Project(_)));
        assert_eq!(detected.display_name.as_deref(), Some("Safe Project"));
        assert!(!detected.display_name.unwrap().contains("secret"));
        assert!(sanitize_label(&"x".repeat(100)).unwrap().len() <= MAX_CONTEXT_LABEL_CHARS);
    }

    #[test]
    fn detects_windows_project_and_normalizes_case() {
        let detector = ContextDetector::default();
        let upper = request(vec![message(
            "user",
            r"Open C:\Work\Gateway\src\main.rs and C:\Work\Gateway\tests\api.rs",
        )]);
        let lower = request(vec![message(
            "user",
            r"Open c:\work\gateway\src\lib.rs and c:\work\gateway\benches\api.rs",
        )]);

        assert_eq!(detector.detect(&upper), detector.detect(&lower));
        assert_eq!(
            project_hash(detector.detect(&upper)),
            short_hash("c:/work/gateway/")
        );
    }

    #[test]
    fn detects_unix_project_across_all_messages_and_array_parts() {
        let detector = ContextDetector::default();
        let messages = vec![
            message("assistant", "Inspect /srv/gateway/src/main.rs"),
            message(
                "user",
                json!([
                    {"type": "image_url", "image_url": {"url": "unused"}},
                    {"type": "text", "text": "and /srv/gateway/tests/api.rs"}
                ]),
            ),
        ];

        assert_eq!(
            detector.detect_messages(&messages),
            ContextType::Project(short_hash("/srv/gateway/"))
        );
    }

    #[test]
    fn rejects_root_only_common_prefixes() {
        let detector = ContextDetector::default();
        let unix = vec![message("user", "/alpha/a.rs and /beta/b.rs")];
        let windows = vec![message("user", r"C:\alpha\a.rs and D:\beta\b.rs")];

        assert_eq!(detector.detect_messages(&unix), ContextType::User);
        assert_eq!(detector.detect_messages(&windows), ContextType::User);
    }

    #[test]
    fn keeps_mixed_path_formats_separate() {
        let detector = ContextDetector::default();
        let messages = vec![message(
            "user",
            r"C:\repo\src\a.rs, /unrelated/one/a.rs, and C:\repo\tests\b.rs",
        )];

        assert_eq!(
            detector.detect_messages(&messages),
            ContextType::Project(short_hash("c:/repo/"))
        );
    }

    #[test]
    fn relative_markers_never_count_toward_threshold() {
        let detector = ContextDetector::default();
        let relative_only = vec![message(
            "user",
            "src/main.rs crates/core/lib.rs packages/ui/app.ts",
        )];
        let one_absolute = vec![message("user", "/srv/repo/src/main.rs and src/lib.rs")];

        assert_eq!(detector.detect_messages(&relative_only), ContextType::User);
        assert_eq!(detector.detect_messages(&one_absolute), ContextType::User);
    }

    #[test]
    fn project_detection_has_priority_over_long_system_prompt() {
        let detector = ContextDetector::default();
        let messages = vec![
            message("system", "agent ".repeat(60)),
            message("user", "/opt/project/src/a.rs /opt/project/tests/b.rs"),
        ];

        assert!(matches!(
            detector.detect_messages(&messages),
            ContextType::Project(_)
        ));
    }

    #[test]
    fn normalizes_and_combines_system_prompts_for_default_matching() {
        let first = "A".repeat(120);
        let second = "B".repeat(120);
        let configured = format!("  {first}\t\n {second}  ");
        let detector = ContextDetector::new(vec![configured]);
        let messages = vec![message("system", first), message("system", second)];

        assert_eq!(detector.detect_messages(&messages), ContextType::User);
    }

    #[test]
    fn hashes_first_500_unicode_characters_safely() {
        let shared = "🦀".repeat(500);
        let detector = ContextDetector::default();
        let first = vec![message("system", format!("{shared}alpha"))];
        let second = vec![message("system", format!("{shared}beta"))];
        let expected = ContextType::Agent(short_hash(&shared));

        assert_eq!(detector.detect_messages(&first), expected);
        assert_eq!(detector.detect_messages(&second), expected);
    }

    #[test]
    fn hashes_are_deterministic_lowercase_hex() {
        let detector = ContextDetector::default();
        let messages = vec![message("user", "/home/me/repo/a.rs /home/me/repo/b.rs")];
        let first = project_hash(detector.detect_messages(&messages));
        let second = project_hash(detector.detect_messages(&messages));

        assert_eq!(first, second);
        assert_eq!(first.len(), 16);
        assert!(first
            .bytes()
            .all(|byte| byte.is_ascii_digit() || (b'a'..=b'f').contains(&byte)));
    }

    #[test]
    fn odd_content_values_do_not_panic() {
        let detector = ContextDetector::default();
        let messages = vec![
            message("system", Value::Null),
            message("user", json!({"x": 1})),
        ];

        assert_eq!(detector.detect_messages(&messages), ContextType::User);
    }

    mod property_tests {
        use super::*;
        use proptest::prelude::*;

        #[derive(Debug)]
        enum ProjectPathPair {
            Windows {
                drive: char,
                workspace: String,
                project: String,
                first_file: String,
                second_file: String,
            },
            Unix {
                workspace: String,
                project: String,
                first_file: String,
                second_file: String,
            },
        }

        fn path_component() -> impl Strategy<Value = String> {
            "[a-z][a-z0-9_-]{0,15}"
        }

        fn project_path_pair() -> impl Strategy<Value = ProjectPathPair> {
            prop_oneof![
                (
                    proptest::char::range('A', 'Z'),
                    path_component(),
                    path_component(),
                    path_component(),
                    path_component(),
                )
                    .prop_map(
                        |(drive, workspace, project, first_file, second_file)| {
                            ProjectPathPair::Windows {
                                drive,
                                workspace,
                                project,
                                first_file,
                                second_file,
                            }
                        },
                    ),
                (
                    path_component(),
                    path_component(),
                    path_component(),
                    path_component(),
                )
                    .prop_map(|(workspace, project, first_file, second_file)| {
                        ProjectPathPair::Unix {
                            workspace,
                            project,
                            first_file,
                            second_file,
                        }
                    },),
            ]
        }

        impl ProjectPathPair {
            fn paths_and_prefix(&self) -> (String, String, String) {
                match self {
                    Self::Windows {
                        drive,
                        workspace,
                        project,
                        first_file,
                        second_file,
                    } => (
                        format!(r"{drive}:\{workspace}\{project}\src\{first_file}.rs"),
                        format!(r"{drive}:\{workspace}\{project}\tests\{second_file}.rs"),
                        format!("{}:/{workspace}/{project}/", drive.to_ascii_lowercase()),
                    ),
                    Self::Unix {
                        workspace,
                        project,
                        first_file,
                        second_file,
                    } => (
                        format!("/{workspace}/{project}/src/{first_file}.rs"),
                        format!("/{workspace}/{project}/tests/{second_file}.rs"),
                        format!("/{workspace}/{project}/"),
                    ),
                }
            }
        }

        proptest! {
        #![proptest_config(ProptestConfig::with_cases(128))]

        #[test]
        fn project_paths_have_priority_over_long_system_prompts(
        pair in project_path_pair(),
        system_prompt in "[A-Za-z0-9]{201,900}",
        split_paths in any::<bool>(),
        ) {
        let (first_path, second_path, prefix) = pair.paths_and_prefix();
        let mut messages = vec![message("system", system_prompt)];
        if split_paths {
        messages.push(message("user", format!("Inspect {first_path}")));
        messages.push(message("assistant", format!("Compare with {second_path}")));
        } else {
        messages.push(message(
        "user",
        format!("Inspect {first_path} and compare with {second_path}"),
        ));
        }

        prop_assert_eq!(
        ContextDetector::default().detect_messages(&messages),
        ContextType::Project(short_hash(&prefix)),
        );
        }

        #[test]
        fn long_system_prompts_without_paths_detect_agent(
        system_prompt in "[A-Za-z0-9]{201,900}",
        uppercase_role in any::<bool>(),
        ) {
        let role = if uppercase_role { "SYSTEM" } else { "system" };
        let expected_input: String = system_prompt.chars().take(MAX_FINGERPRINT_CHARS).collect();
        let messages = vec![message(role, system_prompt)];

        prop_assert_eq!(
        ContextDetector::default().detect_messages(&messages),
        ContextType::Agent(short_hash(&expected_input)),
        );
        }
        }
    }
}
