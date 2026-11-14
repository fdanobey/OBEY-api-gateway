//! Prompt-library configuration and field-level validation.

use std::fmt;

use serde::{Deserialize, Serialize};

const MAX_DATABASE_PATH_CHARS: usize = 4_096;

/// Versioned prompt-library settings from the top-level `prompt_library`
/// section. Storage caps (prompts, versions per prompt) are fixed store
/// constants, not operator configuration.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(default, deny_unknown_fields)]
pub struct PromptLibraryConfig {
    pub database_path: String,
}

impl Default for PromptLibraryConfig {
    fn default() -> Self {
        Self {
            database_path: "./prompts.db".to_owned(),
        }
    }
}

impl PromptLibraryConfig {
    /// Validates all prompt-library fields.
    pub fn validate(&self) -> PromptLibraryValidationResult<()> {
        let mut errors = Vec::new();

        validate_database_path(&self.database_path, &mut errors);

        validation_result(errors)
    }
}

/// One structured, field-addressable prompt-library configuration failure.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct PromptLibraryConfigError {
    pub field: String,
    pub message: String,
}

impl PromptLibraryConfigError {
    pub fn new(field: impl Into<String>, message: impl Into<String>) -> Self {
        Self {
            field: field.into(),
            message: message.into(),
        }
    }
}

impl fmt::Display for PromptLibraryConfigError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            formatter,
            "invalid prompt_library.{}: {}",
            self.field, self.message
        )
    }
}

impl std::error::Error for PromptLibraryConfigError {}

/// Aggregated result returned by prompt-library configuration validation.
pub type PromptLibraryValidationResult<T> = Result<T, Vec<PromptLibraryConfigError>>;

fn validate_database_path(value: &str, errors: &mut Vec<PromptLibraryConfigError>) {
    validate_nonempty("database_path", value, "must not be empty", errors);
    if value.contains('\0') {
        errors.push(PromptLibraryConfigError::new(
            "database_path",
            format!(
                "contains a NUL character in {value:?}; expected a path without NUL characters"
            ),
        ));
    }
    validate_max_chars("database_path", value, MAX_DATABASE_PATH_CHARS, errors);
}

fn validate_nonempty(
    field: &str,
    value: &str,
    constraint: &str,
    errors: &mut Vec<PromptLibraryConfigError>,
) {
    if value.trim().is_empty() {
        errors.push(PromptLibraryConfigError::new(
            field,
            format!("has value {value:?}; {constraint}"),
        ));
    }
}

fn validate_max_chars(
    field: &str,
    value: &str,
    max: usize,
    errors: &mut Vec<PromptLibraryConfigError>,
) {
    let length = value.chars().count();
    if length > max {
        errors.push(PromptLibraryConfigError::new(
            field,
            format!("has {length} characters; expected at most {max} characters"),
        ));
    }
}

fn validation_result(errors: Vec<PromptLibraryConfigError>) -> PromptLibraryValidationResult<()> {
    if errors.is_empty() {
        Ok(())
    } else {
        Err(errors)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn has_error(errors: &[PromptLibraryConfigError], field: &str, message: &str) -> bool {
        errors
            .iter()
            .any(|error| error.field == field && error.message.contains(message))
    }

    #[test]
    fn serde_defaults_match_documented_values() {
        let config: PromptLibraryConfig = serde_yaml::from_str("").unwrap();
        assert_eq!(config, PromptLibraryConfig::default());
        assert_eq!(config.database_path, "./prompts.db");
        assert!(config.validate().is_ok());
    }

    #[test]
    fn valid_full_configuration_deserializes_and_validates() {
        let config: PromptLibraryConfig =
            serde_yaml::from_str("database_path: /var/lib/gateway/prompts.db\n").unwrap();
        assert_eq!(config.database_path, "/var/lib/gateway/prompts.db");
        assert!(config.validate().is_ok());
    }

    #[test]
    fn unknown_fields_are_rejected() {
        let yaml = "database_path: ./prompts.db\nunknown_key: true\n";
        assert!(serde_yaml::from_str::<PromptLibraryConfig>(yaml).is_err());
    }

    #[test]
    fn database_path_wrong_type_is_rejected() {
        assert!(serde_yaml::from_str::<PromptLibraryConfig>("database_path:\n  - a\n  - b\n")
            .is_err());
        assert!(
            serde_yaml::from_str::<PromptLibraryConfig>("database_path:\n  nested: true\n")
                .is_err()
        );
    }

    #[test]
    fn database_path_validation_errors_name_the_field() {
        let config = PromptLibraryConfig {
            database_path: String::new(),
        };
        let errors = config.validate().unwrap_err();
        assert!(has_error(&errors, "database_path", "must not be empty"));
        assert!(errors.iter().any(|error| error
            .to_string()
            .starts_with("invalid prompt_library.database_path:")));

        let config = PromptLibraryConfig {
            database_path: "path\u{0}with\u{0}nul".to_owned(),
        };
        let errors = config.validate().unwrap_err();
        assert!(has_error(&errors, "database_path", "NUL character"));

        let config = PromptLibraryConfig {
            database_path: "p".repeat(MAX_DATABASE_PATH_CHARS + 1),
        };
        let errors = config.validate().unwrap_err();
        assert!(has_error(
            &errors,
            "database_path",
            &format!("expected at most {MAX_DATABASE_PATH_CHARS} characters")
        ));
    }
}
