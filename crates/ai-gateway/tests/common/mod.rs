//! Shared test utilities for integration tests.
//!
//! Each integration-test file in `tests/` is compiled as its own crate, so
//! they cannot share code through `src/`. A file at `tests/common/mod.rs` is
//! the Cargo-blessed way to share helpers across test binaries.
//!
//! ## Purpose
//!
//! Every `GatewayServer::new` opens three SQLite databases:
//! - `logging.database_path` (default `./logs.db`)
//! - a sibling `assistants.db`
//! - `virtual_keys.database_path` (default `./keys.db`)
//!
//! When multiple tests run concurrently inside the *same* test binary, their
//! `GatewayServer` instances share the same default files, producing
//! SQL-level lock contention and file cleanup races. This module's
//! [`isolate_databases`] redirects every database into a unique temporary
//! directory so no two servers ever touch the same file.

use ai_gateway::config::Config;

/// Redirect the logging / assistants / virtual-key SQLite databases into a
/// unique temporary directory so parallel tests never share `./logs.db`.
///
/// The directory is **intentionally leaked**: the gateway holds open SQLite
/// handles for the lifetime of the test process, and removing the directory
/// before those handles close causes Windows delete-failures. The OS temp
/// cleaner reclaims the (tiny) files eventually.
pub fn isolate_databases(config: &mut Config) {
    let dir = tempfile::tempdir().expect("create temp dir for isolated test databases").into_path();
    config.logging.database_path = dir.join("logs.db").to_string_lossy().into_owned();
    config.virtual_keys.database_path = dir.join("keys.db").to_string_lossy().into_owned();
}
