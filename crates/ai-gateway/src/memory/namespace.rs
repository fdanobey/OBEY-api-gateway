//! Safe namespace construction and validation for persistent memories.

use super::{ContextType, ResolvedNamespace};

/// Maximum number of characters accepted in a virtual-key namespace segment.
pub const MAX_VK_ID_CHARS: usize = 128;

/// Maximum number of characters accepted in a namespace.
pub const MAX_NAMESPACE_CHARS: usize = 256;

const MAX_CONTEXT_ID_CHARS: usize = 16;
const DEFAULT_SEGMENT: &str = "default";

impl ResolvedNamespace {
    /// Resolve isolated user and context scopes from caller and detection data.
    pub fn resolve(vk_id: Option<&str>, context_type: &ContextType) -> Self {
        let sanitized_vk = vk_id
            .map(sanitize_vk_id)
            .unwrap_or_else(|| DEFAULT_SEGMENT.to_owned());
        let user_scope = format!("user::{sanitized_vk}");

        let context_scope = match context_type {
            ContextType::Project(context_id) => {
                Some(resolve_context_scope(&user_scope, "project", context_id))
            }
            ContextType::Agent(context_id) => {
                Some(resolve_context_scope(&user_scope, "agent", context_id))
            }
            ContextType::User => None,
        };

        debug_assert!(validate_namespace(&user_scope));
        debug_assert!(context_scope.as_deref().map_or(true, validate_namespace));

        Self {
            user_scope,
            context_scope,
        }
    }
}

/// Restrict a virtual-key ID to one safe namespace segment.
pub fn sanitize_vk_id(vk_id: &str) -> String {
    sanitize_segment(vk_id, MAX_VK_ID_CHARS)
}

/// Validate a namespace accepted from storage or administrative boundaries.
///
/// A valid namespace is ASCII, at most 256 characters, and consists of one or
/// more nonempty `[A-Za-z0-9_-]+` segments separated only by `::`.
pub fn validate_namespace(namespace: &str) -> bool {
    if namespace.is_empty() || namespace.len() > MAX_NAMESPACE_CHARS {
        return false;
    }

    namespace.split("::").all(|segment| {
        !segment.is_empty()
            && segment
                .bytes()
                .all(|byte| byte.is_ascii_alphanumeric() || matches!(byte, b'_' | b'-'))
    })
}

fn resolve_context_scope(user_scope: &str, context_kind: &str, context_id: &str) -> String {
    let context_id = validate_context_id(context_id)
        .then(|| context_id.to_owned())
        .unwrap_or_else(|| DEFAULT_SEGMENT.to_owned());
    let namespace = format!("{user_scope}::{context_kind}::{context_id}");

    if validate_namespace(&namespace) {
        namespace
    } else {
        format!("user::{DEFAULT_SEGMENT}::{context_kind}::{DEFAULT_SEGMENT}")
    }
}

fn validate_context_id(context_id: &str) -> bool {
    !context_id.is_empty()
        && context_id.len() <= MAX_CONTEXT_ID_CHARS
        && context_id
            .bytes()
            .all(|byte| byte.is_ascii_alphanumeric() || matches!(byte, b'_' | b'-'))
}

fn sanitize_segment(value: &str, max_chars: usize) -> String {
    let sanitized: String = value
        .bytes()
        .filter(|byte| byte.is_ascii_alphanumeric() || matches!(byte, b'_' | b'-'))
        .take(max_chars)
        .map(char::from)
        .collect();

    if sanitized.is_empty() {
        DEFAULT_SEGMENT.to_owned()
    } else {
        sanitized
    }
}

#[cfg(test)]
mod property_tests {
    use super::*;
    use proptest::prelude::*;

    fn expected_context_id(context_id: &str) -> &str {
        if validate_context_id(context_id) {
            context_id
        } else {
            DEFAULT_SEGMENT
        }
    }

    proptest! {
        #![proptest_config(ProptestConfig::with_cases(128))]

        #[test]
        fn prop_sanitized_vk_id_is_safe_ascii_and_bounded(vk_id in any::<String>()) {
        let sanitized = sanitize_vk_id(&vk_id);

        prop_assert!(!sanitized.is_empty());
        prop_assert!(sanitized.len() <= MAX_VK_ID_CHARS);
        prop_assert!(sanitized.is_ascii());
    let safe_ascii = sanitized.bytes().all(|byte| {
    byte.is_ascii_alphanumeric() || matches!(byte, b'_' | b'-')
    });
    prop_assert!(safe_ascii);

        if !vk_id.bytes().any(|byte| {
        byte.is_ascii_alphanumeric() || matches!(byte, b'_' | b'-')
        }) {
        prop_assert_eq!(sanitized, DEFAULT_SEGMENT);
        }
        }

        #[test]
        fn prop_resolution_formats_every_context_variant(
        vk_id in proptest::option::of(any::<String>()),
        project_id in any::<String>(),
        agent_id in any::<String>(),
        ) {
        let sanitized_vk = vk_id
        .as_deref()
        .map(sanitize_vk_id)
        .unwrap_or_else(|| DEFAULT_SEGMENT.to_owned());
        let expected_user_scope = format!("user::{sanitized_vk}");

        let user = ResolvedNamespace::resolve(vk_id.as_deref(), &ContextType::User);
        prop_assert_eq!(&user.user_scope, &expected_user_scope);
        prop_assert_eq!(user.context_scope, None);

        let project = ResolvedNamespace::resolve(
        vk_id.as_deref(),
        &ContextType::Project(project_id.clone()),
        );
        prop_assert_eq!(&project.user_scope, &expected_user_scope);
    let expected_project_scope = format!(
    "{expected_user_scope}::project::{}",
    expected_context_id(&project_id)
    );
    prop_assert_eq!(
    project.context_scope.as_deref(),
    Some(expected_project_scope.as_str())
    );

        let agent = ResolvedNamespace::resolve(
        vk_id.as_deref(),
        &ContextType::Agent(agent_id.clone()),
        );
        prop_assert_eq!(&agent.user_scope, &expected_user_scope);
    let expected_agent_scope = format!(
    "{expected_user_scope}::agent::{}",
    expected_context_id(&agent_id)
    );
    prop_assert_eq!(
    agent.context_scope.as_deref(),
    Some(expected_agent_scope.as_str())
    );
        }

        #[test]
        fn prop_resolved_namespaces_are_valid_and_within_limit(
        vk_id in proptest::option::of(any::<String>()),
        context_id in any::<String>(),
        ) {
        let contexts = [
        ContextType::User,
        ContextType::Project(context_id.clone()),
        ContextType::Agent(context_id),
        ];

        for context_type in contexts {
        let resolved = ResolvedNamespace::resolve(vk_id.as_deref(), &context_type);
        prop_assert!(resolved.user_scope.len() <= MAX_NAMESPACE_CHARS);
        prop_assert!(validate_namespace(&resolved.user_scope));

        if let Some(context_scope) = resolved.context_scope {
        prop_assert!(context_scope.len() <= MAX_NAMESPACE_CHARS);
        prop_assert!(validate_namespace(&context_scope));
        }
        }
        }
        }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn sanitizes_separator_injection_and_unsafe_characters() {
        assert_eq!(sanitize_vk_id("vk::admin/../*?\n-name"), "vkadmin-name");
        assert_eq!(sanitize_vk_id("::::"), "default");
    }

    #[test]
    fn unicode_is_removed_without_transliteration() {
        assert_eq!(sanitize_vk_id("café_東京-user"), "caf_-user");
        assert_eq!(sanitize_vk_id("東京"), "default");
    }

    #[test]
    fn vk_id_is_capped_at_128_safe_characters() {
        let sanitized = sanitize_vk_id(&format!("{}ignored", "a".repeat(128)));
        assert_eq!(sanitized.len(), 128);
        assert_eq!(sanitized, "a".repeat(128));
    }

    #[test]
    fn resolves_exact_user_project_and_agent_formats() {
        let user = ResolvedNamespace::resolve(Some("vk_123-test"), &ContextType::User);
        assert_eq!(user.user_scope, "user::vk_123-test");
        assert_eq!(user.context_scope, None);

        let project = ResolvedNamespace::resolve(
            Some("vk_123-test"),
            &ContextType::Project("0123456789abcdef".to_owned()),
        );
        assert_eq!(project.user_scope, "user::vk_123-test");
        assert_eq!(
            project.context_scope.as_deref(),
            Some("user::vk_123-test::project::0123456789abcdef")
        );

        let agent = ResolvedNamespace::resolve(
            Some("vk_123-test"),
            &ContextType::Agent("fedcba9876543210".to_owned()),
        );
        assert_eq!(agent.user_scope, "user::vk_123-test");
        assert_eq!(
            agent.context_scope.as_deref(),
            Some("user::vk_123-test::agent::fedcba9876543210")
        );
    }

    #[test]
    fn absent_or_empty_vk_id_uses_default_scope() {
        for vk_id in [None, Some(""), Some("::/東京/*")] {
            let resolved = ResolvedNamespace::resolve(
                vk_id,
                &ContextType::Project("0123456789abcdef".to_owned()),
            );
            assert_eq!(resolved.user_scope, "user::default");
            assert_eq!(
                resolved.context_scope.as_deref(),
                Some("user::default::project::0123456789abcdef")
            );
        }
    }

    #[test]
    fn context_identifiers_are_sanitized_capped_and_defaulted() {
        let malicious = ResolvedNamespace::resolve(
            Some("caller"),
            &ContextType::Project("ab::cd/../*EFghijklmnopqrstuvwxyz".to_owned()),
        );
        assert_eq!(
            malicious.context_scope.as_deref(),
            Some("user::caller::project::default")
        );

        let empty =
            ResolvedNamespace::resolve(Some("caller"), &ContextType::Agent("::/*東京".to_owned()));
        assert_eq!(
            empty.context_scope.as_deref(),
            Some("user::caller::agent::default")
        );
    }

    #[test]
    fn resolved_namespaces_stay_within_the_storage_limit() {
        let resolved = ResolvedNamespace::resolve(
            Some(&"v".repeat(1_000)),
            &ContextType::Project("h".repeat(1_000)),
        );

        assert_eq!(resolved.user_scope.len(), 134);
        assert_eq!(resolved.context_scope.as_ref().unwrap().len(), 152);
        assert!(resolved.user_scope.len() <= MAX_NAMESPACE_CHARS);
        assert!(resolved.context_scope.as_ref().unwrap().len() <= MAX_NAMESPACE_CHARS);
    }

    #[test]
    fn validates_strict_safe_namespace_grammar() {
        for namespace in [
            "user",
            "user::default",
            "user::vk_123-test::project::0123456789abcdef",
        ] {
            assert!(
                validate_namespace(namespace),
                "{namespace:?} should be valid"
            );
        }

        for namespace in [
            "",
            "::user",
            "user::",
            "user::::project",
            "user::..::project",
            "user::default/../admin",
            "user::*",
            "user::?",
            "user::default\n",
            "user::défaut",
            "user:default",
            "user:::default",
        ] {
            assert!(
                !validate_namespace(namespace),
                "{namespace:?} should be invalid"
            );
        }
    }

    #[test]
    fn validator_enforces_256_character_boundary() {
        assert!(validate_namespace(&"a".repeat(256)));
        assert!(!validate_namespace(&"a".repeat(257)));
    }
}
