//! Pure helpers for the versioned prompt library: parsing the `prompt`
//! request-body field into a store reference, rendering `{{variable}}`
//! placeholders, and extracting variable names — no runtime state of
//! their own, unit-testable in isolation. Store-level reference
//! resolution ([`resolve_reference`]) is a pure function over a
//! `&PromptStore`, dispatched from the chat handler at request time.

use serde_json::{Map, Value};

use super::store::{ListOrder, ListParams, PromptStore, StoreError};
use crate::models::openai::Message;

/// A parsed `prompt` body field: bare name, `name@label`, or uuid id with an
/// optional pinned version and template variables.
#[derive(Debug, Clone, PartialEq)]
pub struct PromptReference {
    pub id: String,
    pub version: Option<i64>,
    pub variables: Option<Map<String, Value>>,
}

/// A 400-class validation error naming the offending `prompt` sub-field.
#[derive(Debug, thiserror::Error)]
#[error("{message}")]
pub struct PromptFieldError {
    field: &'static str,
    message: String,
}

impl PromptFieldError {
    fn new(field: &'static str, message: impl Into<String>) -> Self {
        Self {
            field,
            message: message.into(),
        }
    }

    /// The offending field name (`prompt`, `id`, `version`, or `variables`).
    pub fn field(&self) -> &'static str {
        self.field
    }
}

/// Parse the `prompt` body field into a [`PromptReference`].
///
/// `id` accepts a bare name, a `name@label` reference, or a uuid. An
/// explicit `version` is valid only with a bare name/id reference —
/// combined with a label-bearing id the reference is ambiguous and
/// rejected, mirroring the label+version mutual-exclusivity rule.
pub fn parse_prompt_field(value: &Value) -> Result<PromptReference, PromptFieldError> {
    let Some(object) = value.as_object() else {
        return Err(PromptFieldError::new("prompt", "prompt must be an object"));
    };

    let id = match object.get("id") {
        None => return Err(PromptFieldError::new("id", "id is required")),
        Some(id) => id,
    };
    let Some(id) = id.as_str() else {
        return Err(PromptFieldError::new("id", "id must be a string"));
    };
    if id.is_empty() {
        return Err(PromptFieldError::new("id", "id must not be empty"));
    }

    let version = match object.get("version") {
        None | Some(Value::Null) => None,
        Some(value) => {
            let version = value
                .as_i64()
                .ok_or_else(|| PromptFieldError::new("version", "version must be a positive integer"))?;
            if version < 1 {
                return Err(PromptFieldError::new(
                    "version",
                    "version must be a positive integer",
                ));
            }
            Some(version)
        }
    };

    let variables = match object.get("variables") {
        None | Some(Value::Null) => None,
        Some(value) => {
            let map = value
                .as_object()
                .ok_or_else(|| PromptFieldError::new("variables", "variables must be an object"))?;
            Some(map.clone())
        }
    };

    if version.is_some() && id.contains('@') {
        return Err(PromptFieldError::new(
            "version",
            format!(
                "version cannot be combined with the label reference '{id}'; the reference is ambiguous"
            ),
        ));
    }

    // A label reference carries exactly one '@' with non-empty parts:
    // '@label', 'name@', and 'a@b@c' are malformed, not store misses.
    if id.contains('@') {
        let (name, label) = id.split_once('@').expect("checked above");
        let well_formed =
            !name.is_empty() && !label.is_empty() && !label.contains('@');
        if !well_formed {
            return Err(PromptFieldError::new(
                "id",
                format!(
                    "id must be a bare name, 'name@label', or uuid; '{id}' is not a valid reference"
                ),
            ));
        }
    }

    Ok(PromptReference {
        id: id.to_owned(),
        version,
        variables,
    })
}

/// Render `{{variable}}` placeholders in template content with the provided
/// values. Missing variables and unknown tokens become the empty string,
/// matching the gateway's substitution semantics; non-string values are
/// serialized compactly via serde_json. Braces that do not form a
/// `{{name}}` pattern are left untouched.
///
/// Scanning is literal, not brace-nesting-aware: after each `{{` the NEXT
/// `}}` closes the token and everything between (trimmed) is the variable
/// name. So `{{{name}}}` looks up the name `{name` (a miss for normal
/// names), renders it as the empty string, and leaves one trailing `}`;
/// there is no attempt to match brace runs. An empty name (e.g. `{{ }}`)
/// keeps the whole token literal.
pub fn render(template_content: &str, variables: &Map<String, Value>) -> String {
    let mut result = String::with_capacity(template_content.len());
    let mut rest = template_content;
    while let Some(start) = rest.find("{{") {
        result.push_str(&rest[..start]);
        let after = &rest[start + 2..];
        let Some(end) = after.find("}}") else {
            result.push_str(&rest[start..]);
            return result;
        };
        let name = after[..end].trim();
        if name.is_empty() {
            result.push_str(&rest[start..start + end + 4]);
        } else {
            match variables.get(name) {
                Some(Value::String(text)) => result.push_str(text),
                Some(other) => result.push_str(&other.to_string()),
                None => {}
            }
        }
        rest = &after[end + 2..];
    }
    result.push_str(rest);
    result
}

/// Extract the sorted, de-duplicated `{{variable}}` placeholder names.
pub fn extract_variables(content: &str) -> Vec<String> {
    let mut names = Vec::new();
    let mut rest = content;
    while let Some(start) = rest.find("{{") {
        rest = &rest[start + 2..];
        let Some(end) = rest.find("}}") else {
            break;
        };
        let name = rest[..end].trim();
        if !name.is_empty() {
            names.push(name.to_owned());
        }
        rest = &rest[end + 2..];
    }
    names.sort();
    names.dedup();
    names
}

/// A resolved prompt reference: the version selected (by pinned number,
/// by movable label, or `latest`/`production` default) plus that
/// version's template messages, ready for `{{variable}}` rendering.
#[derive(Debug, Clone)]
pub struct ResolvedPrompt {
    pub prompt_id: String,
    pub name: String,
    pub version: i64,
    pub label: Option<String>,
    pub messages: Vec<Message>,
}

/// Resolution failure for a `prompt` reference. `NotFound` carries the
/// display form of the reference (`name` or `name@label`) plus the
/// label that failed to resolve, so the chat handler can return a
/// 404-class `invalid_request_error` naming the prompt and label —
/// never a silent fallback to another version.
#[derive(Debug, thiserror::Error)]
pub enum ResolveError {
    #[error("prompt '{reference}' with label '{label}' was not found")]
    NotFound { reference: String, label: String },
    #[error(transparent)]
    Store(#[from] StoreError),
}

/// The default label applied to a bare name reference (label omitted).
const DEFAULT_LABEL: &str = "production";

/// Resolve a parsed `prompt` reference against a store snapshot.
///
/// The reference id is interpreted as a uuid, `name@label`, or bare
/// `name` (label defaults to `production`). An explicit pinned
/// `version` overrides label resolution — a label-bearing id combined
/// with a version is already rejected at parse time. Version content
/// is parsed as a JSON array of chat messages; when a version stores
/// raw template text instead, it degrades to a single user message
/// carrying the rendered text. Unresolvable references fail with
/// [`ResolveError::NotFound`] naming the prompt and label.
pub fn resolve_reference(
    store: &PromptStore,
    reference: &PromptReference,
) -> Result<ResolvedPrompt, ResolveError> {
    // Split `name@label` (first '@' wins; names cannot contain '@').
    let (name, label) = match reference.id.split_once('@') {
        Some((name, label)) => (name, label.to_owned()),
        None => (reference.id.as_str(), DEFAULT_LABEL.to_owned()),
    };

    // Resolve the prompt row by uuid first, then by name. Name lookup
    // walks the store listing (names are unique and bounded by the
    // store's prompt cap), matching by exact name. The requested label
    // is threaded through so a miss names what the client asked for.
    let prompt = match store.get_prompt(&reference.id) {
        Ok(prompt) => prompt,
        Err(StoreError::NotFound { .. }) => {
            find_prompt_by_name(store, name, &label)?
        }
        Err(error) => return Err(error.into()),
    };

    // Pin wins over label resolution; otherwise the label's current
    // version (virtual `latest` = MAX(version)) serves the call. A
    // label no version carries is an unresolvable reference — never a
    // silent fallback to another version.
    let (version, resolved_label) = match reference.version {
        Some(version) => (version, None),
        None => {
            let version = prompt
                .labels
                .get(&label)
                .copied()
                .or_else(|| (label == "latest").then_some(prompt.latest_version))
                .ok_or_else(|| ResolveError::NotFound {
                    reference: prompt.name.clone(),
                    label: label.clone(),
                })?;
            (version, Some(label))
        }
    };

    let content = version_content(store, &prompt.id, &prompt.name, version)?;
    let messages = template_messages(&content);

    Ok(ResolvedPrompt {
        prompt_id: prompt.id,
        name: prompt.name,
        version,
        label: resolved_label,
        messages,
    })
}

/// Find a prompt row by exact name via the store listing; names are
/// unique (enforced at create), so the first match is the match. The
/// requested label rides along so an exhausted scan reports the label
/// the client actually asked for, not the default.
fn find_prompt_by_name(
    store: &PromptStore,
    name: &str,
    requested_label: &str,
) -> Result<super::store::Prompt, ResolveError> {
    // Ascending creation order + max cap makes this a bounded scan; the
    // cursor advances until the name is found or the store ends.
    let mut params = ListParams {
        limit: 100,
        order: ListOrder::Asc,
        after: None,
        before: None,
    };
    loop {
        let page = store.list_prompts(&params)?;
        let found = page
            .data
            .iter()
            .find(|item| item.get("name").and_then(Value::as_str) == Some(name));
        if let Some(item) = found {
            let id = item.get("id").and_then(Value::as_str).unwrap_or_default();
            return store.get_prompt(id).map_err(ResolveError::from);
        }
        if !page.has_more {
            return Err(ResolveError::NotFound {
                reference: name.to_owned(),
                label: requested_label.to_owned(),
            });
        }
        let next_cursor = page
            .data
            .last()
            .and_then(|item| item.get("id"))
            .and_then(Value::as_str)
            .map(str::to_owned);
        match next_cursor {
            Some(cursor) => params.after = Some(cursor),
            // The page claims more data but carries no cursor id: the
            // name cannot be found by continuing the scan.
            None => {
                return Err(StoreError::InvalidRequest(format!(
                    "prompt '{name}' not found: listing cursor ended before a match"
                ))
                .into())
            }
        }
    }
}

/// Load one version's stored content or fail with a version NotFound.
/// A single targeted fetch; the store never loads sibling versions.
fn version_content(
    store: &PromptStore,
    prompt_id: &str,
    prompt_name: &str,
    version: i64,
) -> Result<String, ResolveError> {
    store
        .get_version(prompt_id, version)
        .map(|stored| stored.content)
        .map_err(|error| match error {
            StoreError::NotFound { .. } => ResolveError::NotFound {
                reference: prompt_name.to_owned(),
                label: format!("version {version}"),
            },
            error => ResolveError::Store(error),
        })
}

/// Parse stored version content into template messages. The canonical
/// shape is a JSON array of chat messages; raw template text degrades
/// to a single user message carrying the text (rendered later), noted
/// as a deviation from the messages-JSON assumption.
fn template_messages(content: &str) -> Vec<Message> {
    if let Ok(messages) = serde_json::from_str::<Vec<Message>>(content) {
        if !messages.is_empty() {
            return messages;
        }
    }
    vec![Message {
        role: "user".to_owned(),
        content: Value::String(content.to_owned()),
        extra: Map::new(),
    }]
}

/// Render `{{variable}}` placeholders per template message. String
/// content renders in place; non-string content (arrays, objects) is
/// left untouched for the provider to interpret. Missing variables
/// become the empty string via [`render`].
pub fn render_messages(messages: &[Message], variables: &Map<String, Value>) -> Vec<Message> {
    messages
        .iter()
        .map(|message| {
            let mut rendered = message.clone();
            if let Value::String(text) = &message.content {
                rendered.content = Value::String(render(text, variables));
            }
            rendered
        })
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    fn reference(id: &str, version: Option<i64>) -> PromptReference {
        PromptReference {
            id: id.to_owned(),
            version,
            variables: None,
        }
    }

    /// Greet prompt: v1 stores canonical messages JSON, v2 stores raw
    /// template text; `staging` is promoted to v2.
    fn populated_store() -> (tempfile::TempDir, PromptStore, String) {
        let temp = tempfile::tempdir().unwrap();
        let store = PromptStore::open(&temp.path().join("prompts.db")).unwrap();
        let prompt = store
            .create_prompt(
                "greet",
                r#"[{"role":"system","content":"You are {{tone}}."},{"role":"user","content":"Hello {{name}}!"}]"#,
                "tester",
            )
            .unwrap();
        store
            .create_version(&prompt.id, None, "Hi {{name}}, welcome back!", "raw variant", "tester")
            .unwrap();
        store.promote_label(&prompt.id, "staging", 2, "tester").unwrap();
        (temp, store, prompt.id)
    }

    #[test]
    fn resolve_reference_by_uuid_serves_production_v1() {
        let (_temp, store, id) = populated_store();
        let resolved = resolve_reference(&store, &reference(&id, None)).unwrap();
        assert_eq!(resolved.prompt_id, id);
        assert_eq!(resolved.name, "greet");
        assert_eq!(resolved.version, 1);
        assert_eq!(resolved.label.as_deref(), Some("production"));
        assert_eq!(resolved.messages.len(), 2);
        assert_eq!(resolved.messages[0].role, "system");
        assert_eq!(resolved.messages[1].role, "user");
    }

    #[test]
    fn resolve_reference_bare_name_defaults_to_production() {
        let (_temp, store, _id) = populated_store();
        let resolved = resolve_reference(&store, &reference("greet", None)).unwrap();
        assert_eq!(resolved.version, 1);
        assert_eq!(resolved.label.as_deref(), Some("production"));
    }

    #[test]
    fn resolve_reference_name_label_serves_moved_version() {
        let (_temp, store, _id) = populated_store();
        let resolved = resolve_reference(&store, &reference("greet@staging", None)).unwrap();
        assert_eq!(resolved.version, 2);
        assert_eq!(resolved.label.as_deref(), Some("staging"));
    }

    #[test]
    fn resolve_reference_latest_serves_newest_version() {
        let (_temp, store, _id) = populated_store();
        let resolved = resolve_reference(&store, &reference("greet@latest", None)).unwrap();
        assert_eq!(resolved.version, 2);
        assert_eq!(resolved.label.as_deref(), Some("latest"));
    }

    #[test]
    fn resolve_reference_pinned_version_overrides_label_resolution() {
        let (_temp, store, _id) = populated_store();
        let resolved = resolve_reference(&store, &reference("greet", Some(2))).unwrap();
        assert_eq!(resolved.version, 2);
        assert_eq!(resolved.label, None);
    }

    #[test]
    fn resolve_reference_unknown_name_names_prompt_and_label() {
        let (_temp, store, _id) = populated_store();
        let error = resolve_reference(&store, &reference("nope", None)).unwrap_err();
        match error {
            ResolveError::NotFound { reference, label } => {
                assert_eq!(reference, "nope");
                assert_eq!(label, "production");
            }
            other => panic!("expected NotFound, got {other:?}"),
        }
    }

    #[test]
    fn resolve_reference_unknown_label_names_prompt_and_label() {
        let (_temp, store, _id) = populated_store();
        let error = resolve_reference(&store, &reference("greet@development", None)).unwrap_err();
        match error {
            ResolveError::NotFound { reference, label } => {
                assert_eq!(reference, "greet");
                assert_eq!(label, "development");
            }
            other => panic!("expected NotFound, got {other:?}"),
        }
    }

    #[test]
    fn resolve_reference_deleted_prompt_is_not_found() {
        let temp = tempfile::tempdir().unwrap();
        let store = PromptStore::open(&temp.path().join("prompts.db")).unwrap();
        let prompt = store.create_prompt("gone", "farewell", "tester").unwrap();
        store.delete_prompt(&prompt.id).unwrap();
        let error = resolve_reference(&store, &reference("gone", None)).unwrap_err();
        match error {
            ResolveError::NotFound { reference, label } => {
                assert_eq!(reference, "gone");
                assert_eq!(label, "production");
            }
            other => panic!("expected NotFound, got {other:?}"),
        }
    }

    #[test]
    fn resolve_reference_unknown_pinned_version_is_not_found() {
        let (_temp, store, _id) = populated_store();
        let error = resolve_reference(&store, &reference("greet", Some(99))).unwrap_err();
        match error {
            ResolveError::NotFound { reference, label } => {
                assert_eq!(reference, "greet");
                assert_eq!(label, "version 99");
            }
            other => panic!("expected NotFound, got {other:?}"),
        }
    }

    #[test]
    fn resolve_reference_raw_text_version_falls_back_to_single_user_message() {
        let (_temp, store, _id) = populated_store();
        let resolved = resolve_reference(&store, &reference("greet@staging", None)).unwrap();
        assert_eq!(resolved.messages.len(), 1);
        assert_eq!(resolved.messages[0].role, "user");
        assert_eq!(resolved.messages[0].content_as_text(), "Hi {{name}}, welcome back!");
    }

    #[test]
    fn render_messages_renders_string_content_per_message() {
        let (_temp, store, _id) = populated_store();
        let resolved = resolve_reference(&store, &reference("greet", None)).unwrap();
        let variables = Map::from_iter([
            ("tone".to_owned(), json!("concise")),
            ("name".to_owned(), json!("World")),
        ]);
        let rendered = render_messages(&resolved.messages, &variables);
        assert_eq!(rendered[0].content_as_text(), "You are concise.");
        assert_eq!(rendered[1].content_as_text(), "Hello World!");
    }

    #[test]
    fn render_messages_missing_variables_become_empty_and_non_string_content_untouched() {
        let messages = vec![
            Message {
                role: "user".to_owned(),
                content: json!("a={{known}} b={{missing}}"),
                extra: Map::new(),
            },
            Message {
                role: "user".to_owned(),
                content: json!([{"type": "text", "text": "kept {{name}}"}]),
                extra: Map::new(),
            },
        ];
        let variables = Map::from_iter([("known".to_owned(), json!("value"))]);
        let rendered = render_messages(&messages, &variables);
        assert_eq!(rendered[0].content_as_text(), "a=value b=");
        // Array content is not string content: left untouched for the
        // provider to interpret.
        assert_eq!(
            rendered[1].content,
            json!([{"type": "text", "text": "kept {{name}}"}])
        );
    }

    #[test]
    fn parse_accepts_bare_name_label_uuid_and_version_pin() {
        let bare = parse_prompt_field(&json!({"id": "greeting"})).unwrap();
        assert_eq!(
            bare,
            PromptReference {
                id: "greeting".to_owned(),
                version: None,
                variables: None
            }
        );

        let labeled = parse_prompt_field(&json!({"id": "greeting@staging"})).unwrap();
        assert_eq!(labeled.id, "greeting@staging");
        assert_eq!(labeled.version, None);

        let uuid = parse_prompt_field(&json!({"id": "00000000-0000-0000-0000-000000000001"}))
            .unwrap();
        assert_eq!(uuid.id, "00000000-0000-0000-0000-000000000001");

        let full = parse_prompt_field(&json!({
            "id": "greeting",
            "version": 3,
            "variables": {"name": "World", "count": 2}
        }))
        .unwrap();
        assert_eq!(full.version, Some(3));
        assert_eq!(
            full.variables.as_ref().unwrap().get("name"),
            Some(&json!("World"))
        );

        // Explicit nulls are treated like absent fields.
        let nulled = parse_prompt_field(&json!({"id": "greeting", "version": null, "variables": null}))
            .unwrap();
        assert_eq!(nulled.version, None);
        assert_eq!(nulled.variables, None);
    }

    #[test]
    fn parse_rejects_invalid_prompt_fields() {
        for value in [json!("greeting"), json!(42), json!([]), json!(null)] {
            let error = parse_prompt_field(&value).unwrap_err();
            assert_eq!(error.field(), "prompt", "{value:?}");
        }
        for value in [json!({}), json!({"id": ""}), json!({"id": 7}), json!({"id": ["a"]})] {
            let error = parse_prompt_field(&value).unwrap_err();
            assert_eq!(error.field(), "id", "{value:?}");
        }
        for value in [
            json!({"id": "greeting", "version": 0}),
            json!({"id": "greeting", "version": -1}),
            json!({"id": "greeting", "version": 1.5}),
            json!({"id": "greeting", "version": "2"}),
        ] {
            let error = parse_prompt_field(&value).unwrap_err();
            assert_eq!(error.field(), "version", "{value:?}");
        }
        for value in [
            json!({"id": "greeting", "variables": [1]}),
            json!({"id": "greeting", "variables": "map"}),
        ] {
            let error = parse_prompt_field(&value).unwrap_err();
            assert_eq!(error.field(), "variables", "{value:?}");
        }
    }

#[test]
fn parse_rejects_malformed_label_references() {
    // '@label', 'name@', and 'a@b@c' are malformed id references: a
    // 400-class parse error naming `id`, never a store miss.
    for value in [
        json!({"id": "@staging"}),
        json!({"id": "greeting@"}),
        json!({"id": "a@b@c"}),
        json!({"id": "@"}),
    ] {
        let error = parse_prompt_field(&value).unwrap_err();
        assert_eq!(error.field(), "id", "{value:?}");
        let rendered = error.to_string();
        assert!(
            rendered.contains("name@label"),
            "expected malformed-reference message for {value:?}, got {rendered}"
        );
    }
    // One '@' with both parts non-empty stays valid.
    assert_eq!(
        parse_prompt_field(&json!({"id": "greeting@staging"})).unwrap().id,
        "greeting@staging"
    );
}

#[test]
fn resolve_reference_unknown_name_with_label_names_staging() {
    let (_temp, store, _id) = populated_store();
    let error = resolve_reference(&store, &reference("ghost@staging", None)).unwrap_err();
    match error {
        ResolveError::NotFound { reference, label } => {
            assert_eq!(reference, "ghost");
            assert_eq!(label, "staging");
        }
        other => panic!("expected NotFound, got {other:?}"),
    }
}

#[test]
fn get_version_fetches_one_version_without_loading_siblings() {
    let (_temp, store, id) = populated_store();
    let fetched = store.get_version(&id, 1).unwrap();
    assert_eq!(fetched.prompt_id, id);
    assert_eq!(fetched.version, 1);
    assert_eq!(fetched.content, r#"[{"role":"system","content":"You are {{tone}}."},{"role":"user","content":"Hello {{name}}!"}]"#);
    assert_eq!(fetched.change_note, "initial version");

    let second = store.get_version(&id, 2).unwrap();
    assert_eq!(second.version, 2);
    assert_eq!(second.content, "Hi {{name}}, welcome back!");

    let missing = store.get_version(&id, 99).unwrap_err();
    assert!(matches!(missing, StoreError::NotFound { .. }));

    let unknown_prompt = store
        .get_version("00000000-0000-0000-0000-0000000000ff", 1)
        .unwrap_err();
    assert!(matches!(unknown_prompt, StoreError::NotFound { .. }));
}

#[test]
fn parse_rejects_label_reference_combined_with_version() {
        let error =
            parse_prompt_field(&json!({"id": "greeting@production", "version": 2})).unwrap_err();
        assert_eq!(error.field(), "version");
        assert!(
            error.to_string().contains("ambiguous"),
            "expected ambiguity message, got {error}"
        );
        // A bare name with a pinned version stays valid.
        assert_eq!(
            parse_prompt_field(&json!({"id": "greeting", "version": 2}))
                .unwrap()
                .version,
            Some(2)
        );
    }

    #[test]
    fn render_substitutes_variables_and_serializes_non_strings() {
        let variables = Map::from_iter([
            ("name".to_owned(), json!("World")),
            ("count".to_owned(), json!(3)),
            ("flag".to_owned(), json!(true)),
            ("list".to_owned(), json!([1, 2])),
            ("nested".to_owned(), json!({"a": 1})),
        ]);
        let rendered = render(
            "Hello {{name}}, count={{count}}, flag={{flag}}, list={{list}}, nested={{nested}}",
            &variables,
        );
        assert_eq!(
            rendered,
            "Hello World, count=3, flag=true, list=[1,2], nested={\"a\":1}"
        );
        // Whitespace inside the braces is trimmed, matching extract_variables.
        assert_eq!(render("{{ name }}", &variables), "World");
    }

    #[test]
    fn render_missing_and_unknown_variables_become_empty() {
        let variables = Map::from_iter([("known".to_owned(), json!("value"))]);
        assert_eq!(
            render("A={{known}} B={{missing}} C={{other}}", &variables),
            "A=value B= C="
        );
        assert_eq!(render("no placeholders here", &variables), "no placeholders here");
    }

#[test]
fn render_leaves_unmatched_braces_untouched() {
 let variables = Map::new();
 assert_eq!(render("plain {braces} stay", &variables), "plain {braces} stay");
 assert_eq!(render("a } b { c", &variables), "a } b { c");
 assert_eq!(render("{{ unclosed", &variables), "{{ unclosed");
 assert_eq!(render("{{ }}", &variables), "{{ }}");
 assert_eq!(render("}}", &variables), "}}");
}

/// Documented literal-scan semantics (see `render`): a `{{` token runs to
/// the NEXT `}}`; brace runs are not nested-matched. `{{{name}}}` looks up
/// `{name`, misses, renders empty, and leaves the leftover `}`.
#[test]
fn render_brace_runs_scan_literally_to_the_next_closing_pair() {
 let variables = Map::from_iter([("name".to_owned(), json!("World"))]);
 assert_eq!(render("{{{name}}}", &variables), "}");
 assert_eq!(render("{{{ name }}}", &variables), "}");
 assert_eq!(render("a {{{name}}} b", &variables), "a } b");
 // A variable literally named `{name` would match the inner text.
 let braced = Map::from_iter([("{name".to_owned(), json!("hit"))]);
 assert_eq!(render("{{{name}}}", &braced), "hit}");
}

    #[test]
    fn extract_variables_returns_sorted_deduped_names() {
        let content = "Dear {{zeta}} and {{alpha}}, again {{ zeta }} and {{}}.";
        assert_eq!(
            extract_variables(content),
            vec!["alpha".to_owned(), "zeta".to_owned()]
        );
        assert!(extract_variables("no placeholders").is_empty());
    }
}
