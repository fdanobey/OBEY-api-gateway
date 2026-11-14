//! Core types for the versioned prompt library.
//!
//! Feature implementations live in sibling modules. Only modules whose files
//! exist in the current implementation wave are declared here. Re-exports
//! are limited to the names consumed outside this module (gateway handlers,
//! admin routes, config); everything else stays reachable via the
//! `pub mod` paths below.

pub mod admin;
pub mod config;
pub mod resolver;
pub mod store;

use std::path::Path;
use std::sync::Arc;

pub use config::PromptLibraryConfig;
pub use resolver::{parse_prompt_field, render_messages, resolve_reference, ResolveError};
pub use store::{ListPage, ListParams, PromptLabel, PromptStore, StoreError};

/// Gateway-facing prompt-library facade: the opened store plus a snapshot
/// of the resolved config it was built from. Held via `Arc` inside
/// `AppState.prompt_library`; hot-reload swaps in a freshly built library
/// while in-flight requests keep the `Arc` they cloned at request start.
pub struct PromptLibrary {
    store: PromptStore,
    database_path: String,
}

impl std::fmt::Debug for PromptLibrary {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        formatter
            .debug_struct("PromptLibrary")
            .field("database_path", &self.database_path)
            .finish_non_exhaustive()
    }
}

impl PromptLibrary {
    /// The opened prompt store.
    pub fn store(&self) -> &PromptStore {
        &self.store
    }

    /// The `prompt_library.database_path` this library was built from
    /// (hot-reload compares it against a newly requested path).
    pub fn database_path(&self) -> &str {
        &self.database_path
    }
}

/// Open the prompt store described by `config` and wrap it in a
/// [`PromptLibrary`]. A descriptive error is returned when the database
/// cannot be opened; callers degrade the feature, never the gateway.
pub fn build_prompt_library(config: &PromptLibraryConfig) -> Result<Arc<PromptLibrary>, String> {
    match PromptStore::open(Path::new(&config.database_path)) {
        Ok(store) => Ok(Arc::new(PromptLibrary {
            store,
            database_path: config.database_path.clone(),
        })),
        Err(error) => Err(format!(
            "failed to open prompt library database at {}: {error}",
            config.database_path
        )),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn build_prompt_library_opens_temp_database() {
        let temp = tempfile::tempdir().unwrap();
        let config = PromptLibraryConfig {
            database_path: temp.path().join("prompts.db").to_string_lossy().into_owned(),
        };
        let library = build_prompt_library(&config).unwrap();
        assert_eq!(library.database_path(), config.database_path);
        let page = library
            .store()
            .list_prompts(&ListParams::default())
            .unwrap();
        assert!(page.data.is_empty());
    }

    #[test]
    fn build_prompt_library_fails_descriptively_when_parent_is_a_file() {
        let temp = tempfile::tempdir().unwrap();
        let parent = temp.path().join("not-a-directory");
        std::fs::write(&parent, b"occupied").unwrap();
        let config = PromptLibraryConfig {
            database_path: parent.join("prompts.db").to_string_lossy().into_owned(),
        };
        let error = build_prompt_library(&config).unwrap_err();
        assert!(error.contains("prompt library"), "{error}");
        assert!(error.contains(&config.database_path), "{error}");
    }
}
