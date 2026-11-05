#![allow(dead_code)]
//! Encrypted at-rest persistence for GitHub Copilot credentials.
//!
//! Mirrors [`super::super::store::OAuthTokenStore`]: the payload is
//! serialized to JSON, encrypted with the router's master key via
//! [`crate::secrets::encrypt_provider_secret`] (AES-256-GCM, `enc-v1:`
//! format), and persisted to a single file (`copilot_tokens.enc`) colocated
//! with `oauth_tokens.enc` and `master.key` under the router's secure
//! storage directory. On Unix the file is created with `0600`
//! permissions via [`crate::secrets::set_owner_only_permissions`],
//! matching the hardening applied to `master.key`.
//!
//! GitHub's device flow returns no refresh token, so the payload keeps the
//! long-lived `github_access_token` (`gho_…`), an optional short-lived
//! Copilot bearer with its expiry, and the metadata needed to rebuild the
//! session on restart (`api_base`, `machine_id`, `sku`, `scopes`,
//! `updated_at`).
//!
//! # Security invariants
//!
//! - The plaintext JSON never lands on disk; only the `enc-v1:` blob does
//!   (Req 3.5: tokens persisted encrypted, never plaintext).
//! - All token fields are `<redacted>` in `Debug` output (Req 8.4).

use std::fmt;
use std::fs;
use std::path::{Path, PathBuf};

use serde::{Deserialize, Serialize};

use super::super::error::OAuthError;
use crate::secrets;

/// Filename used for the encrypted Copilot credential blob, colocated with
/// `oauth_tokens.enc` and `master.key` under the router's secure storage
/// directory.
const COPILOT_TOKENS_FILENAME: &str = "copilot_tokens.enc";

/// Credential payload persisted by [`CopilotTokenStore`].
///
/// Serialized as JSON before being handed to
/// [`crate::secrets::encrypt_provider_secret`]; the resulting `enc-v1:`
/// blob is what actually lands on disk.
///
/// # Debug redaction
///
/// `github_access_token` and `copilot_token` are formatted as
/// `<redacted>` by the manual [`fmt::Debug`] implementation so accidental
/// `{:?}` logging, `tracing` field capture, or panic messages cannot leak
/// Bearer material (Req 8.4: tokens never appear in any log output).
/// Non-secret fields (`copilot_expires_at`, `api_base`, `machine_id`,
/// `sku`, `scopes`, `updated_at`) are emitted verbatim so expiry and
/// routing diagnostics remain useful.
#[derive(Clone, Serialize, Deserialize)]
pub struct CopilotStoredCredentials {
    /// Long-lived GitHub OAuth access token (`gho_…`) used to exchange for
    /// short-lived Copilot API tokens. The device flow returns no refresh
    /// token. Never logged; see the `Debug` impl on this struct.
    pub github_access_token: String,
    /// Short-lived Copilot API bearer token, when one is currently held.
    /// Never logged; see the `Debug` impl on this struct.
    pub copilot_token: Option<String>,
    /// Expiry of [`Self::copilot_token`] as a Unix timestamp in seconds.
    pub copilot_expires_at: Option<u64>,
    /// Copilot API base URL (`https://api.githubcopilot.com` or the
    /// `endpoints.api` override from the token exchange response).
    pub api_base: String,
    /// Stable machine identifier generated at login and reused across
    /// restarts; an opaque random UUID, not a credential.
    pub machine_id: String,
    /// Copilot SKU reported by the token exchange, when present.
    pub sku: Option<String>,
    /// Space-separated OAuth scope string granted by GitHub.
    pub scopes: String,
    /// Unix timestamp (seconds) of the last successful store write.
    pub updated_at: u64,
}

impl fmt::Debug for CopilotStoredCredentials {
    /// Redacts both token fields to preserve the "tokens never logged"
    /// invariant (Req 8.4). Non-secret fields are emitted verbatim so
    /// expiry and routing diagnostics remain useful.
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("CopilotStoredCredentials")
            .field("github_access_token", &"<redacted>")
            .field("copilot_token", &"<redacted>")
            .field("copilot_expires_at", &self.copilot_expires_at)
            .field("api_base", &self.api_base)
            .field("machine_id", &self.machine_id)
            .field("sku", &self.sku)
            .field("scopes", &self.scopes)
            .field("updated_at", &self.updated_at)
            .finish()
    }
}

/// Encrypted Copilot credential store backed by a single file on disk.
///
/// Holds only the destination path; all cryptographic material is sourced
/// from [`crate::secrets`] on demand so this type is cheap to clone and
/// share across async tasks.
#[derive(Debug, Clone)]
pub struct CopilotTokenStore {
    file_path: PathBuf,
}

impl CopilotTokenStore {
    /// Create a new store that reads and writes credentials at `file_path`.
    ///
    /// The file is not created or touched until the first `save` call. Use
    /// [`CopilotTokenStore::default_path`] to derive the conventional
    /// location alongside the master key and `oauth_tokens.enc`.
    pub fn new(file_path: impl Into<PathBuf>) -> Self {
        Self {
            file_path: file_path.into(),
        }
    }

    /// Default on-disk location: `{storage_dir}/copilot_tokens.enc`.
    ///
    /// Mirrors [`super::super::store::OAuthTokenStore::default_path`]:
    /// callers typically pass the parent of
    /// [`crate::secrets::master_key_path`] so the Copilot blob lives next
    /// to the master key it is encrypted with and to `oauth_tokens.enc`.
    pub fn default_path(storage_dir: &Path) -> PathBuf {
        storage_dir.join(COPILOT_TOKENS_FILENAME)
    }

    /// Path the store reads from and writes to.
    pub fn file_path(&self) -> &Path {
        &self.file_path
    }

    /// Encrypt and persist `credentials` to [`Self::file_path`].
    ///
    /// The payload is serialized to JSON, encrypted with the router's
    /// master key via [`crate::secrets::encrypt_provider_secret`]
    /// (AES-256-GCM, `enc-v1:` prefix), and written atomically-ish via
    /// [`fs::write`]. The parent directory is created on demand. Only the
    /// ciphertext is written — the plaintext JSON never touches the disk
    /// (Req 3.5).
    ///
    /// On Unix the file's mode is tightened to `0600` immediately after
    /// write via [`secrets::set_owner_only_permissions`], matching the
    /// hardening already applied to `master.key` and `oauth_tokens.enc`.
    /// On non-Unix platforms that helper is a no-op.
    ///
    /// # Logging
    ///
    /// Arguments are skipped from the tracing span so neither the plaintext
    /// JSON nor the resulting ciphertext are ever captured — only the
    /// target path is recorded (Req 8.4).
    #[tracing::instrument(skip(self, credentials), fields(path = %self.file_path.display()))]
    pub fn save(&self, credentials: &CopilotStoredCredentials) -> Result<(), OAuthError> {
        let plaintext = serde_json::to_string(credentials)?;
        let ciphertext = secrets::encrypt_provider_secret(&plaintext)?;

        if let Some(parent) = self.file_path.parent() {
            fs::create_dir_all(parent)?;
        }
        fs::write(&self.file_path, ciphertext)?;
        secrets::set_owner_only_permissions(&self.file_path)?;
        Ok(())
    }

    /// Load, decrypt, and deserialize credentials from
    /// [`Self::file_path`].
    ///
    /// A missing file maps to `None` (fresh install or logged-out state).
    /// A present-but-corrupt blob surfaces as the existing
    /// [`OAuthError::Crypto`] / [`OAuthError::Serde`] variants; there is no
    /// plaintext fallback.
    #[tracing::instrument(skip(self), fields(path = %self.file_path.display()))]
    pub fn load(&self) -> Result<Option<CopilotStoredCredentials>, OAuthError> {
        let ciphertext = match fs::read_to_string(&self.file_path) {
            Ok(s) => s,
            Err(e) if e.kind() == std::io::ErrorKind::NotFound => return Ok(None),
            Err(e) => return Err(OAuthError::Io(e)),
        };
        let plaintext = secrets::decrypt_provider_secret(&ciphertext)?;
        Ok(Some(serde_json::from_str(&plaintext)?))
    }

    /// Remove the credential file at [`Self::file_path`] (logout).
    ///
    /// A missing file is treated as success so logout is idempotent.
    #[tracing::instrument(skip(self), fields(path = %self.file_path.display()))]
    pub fn delete(&self) -> Result<(), OAuthError> {
        match fs::remove_file(&self.file_path) {
            Ok(()) => Ok(()),
            Err(e) if e.kind() == std::io::ErrorKind::NotFound => Ok(()),
            Err(e) => Err(OAuthError::Io(e)),
        }
    }
}

#[cfg(test)]
mod tests {
use super::*;
use crate::oauth::github_copilot::test_support::secure_dir;
use crate::secrets::{
decrypt_provider_secret_with_key, encrypt_provider_secret_with_key,
};
use proptest::prelude::*;

fn unique_store() -> CopilotTokenStore {
crate::oauth::github_copilot::test_support::unique_store("copilot_tokens")
}

fn sample_credentials() -> CopilotStoredCredentials {
CopilotStoredCredentials {
github_access_token: "gho_16C7e42F292c6912E7710c838347Ae178B4a".into(),
copilot_token: Some("tid=abcd;exp=1770000000;sku=githubcopilot".into()),
copilot_expires_at: Some(1770000000),
api_base: "https://api.githubcopilot.com".into(),
machine_id: "6f9619ff-8b86-d011-b42d-00cf4fc964ff".into(),
sku: Some("githubcopilot".into()),
scopes: "read:user copilot".into(),
updated_at: 1762262400,
}
}

#[test]
    fn round_trip_save_load_delete() {
        let store = unique_store();
        assert!(
            store.load().expect("load empty").is_none(),
            "fresh store must be empty"
        );

        let creds = sample_credentials();
        store.save(&creds).expect("save");

        assert!(
            store.file_path().exists(),
            "encrypted credential file must exist after save"
        );
        let loaded = store.load().expect("load").expect("credentials");
        assert_eq!(loaded.github_access_token, creds.github_access_token);
        assert_eq!(loaded.copilot_token, creds.copilot_token);
        assert_eq!(loaded.copilot_expires_at, creds.copilot_expires_at);
        assert_eq!(loaded.api_base, creds.api_base);
        assert_eq!(loaded.machine_id, creds.machine_id);
        assert_eq!(loaded.sku, creds.sku);
        assert_eq!(loaded.scopes, creds.scopes);
        assert_eq!(loaded.updated_at, creds.updated_at);

        store.delete().expect("delete");
        assert!(
            store.load().expect("load after delete").is_none(),
            "credentials must be gone after delete"
        );
        store.delete().expect("delete again is idempotent");
    }

    #[test]
    fn round_trip_without_optional_fields() {
        let store = unique_store();
        let creds = CopilotStoredCredentials {
            github_access_token: "gho_minimal".into(),
            copilot_token: None,
            copilot_expires_at: None,
            api_base: "https://api.githubcopilot.com".into(),
            machine_id: "machine".into(),
            sku: None,
            scopes: "copilot".into(),
            updated_at: 42,
        };
        store.save(&creds).expect("save");

        let loaded = store.load().expect("load").expect("credentials");
        assert_eq!(loaded.github_access_token, "gho_minimal");
        assert_eq!(loaded.copilot_token, None);
        assert_eq!(loaded.copilot_expires_at, None);
        assert_eq!(loaded.sku, None);
        assert_eq!(loaded.updated_at, 42);
    }

    #[test]
    fn save_creates_missing_parent_directories() {
        let store = CopilotTokenStore::new(secure_dir().join("nested").join("deep").join("copilot_tokens.enc"));
        assert!(store.load().expect("load").is_none());
        store.save(&sample_credentials()).expect("save into nested path");
        assert!(store.load().expect("load").is_some());
    }

    #[test]
    fn saved_file_is_encrypted_not_plaintext_json() {
        let store = unique_store();
        store.save(&sample_credentials()).expect("save");

        let raw = fs::read_to_string(store.file_path()).expect("read raw file");
        assert!(raw.starts_with("enc-v1:"), "file must be an enc-v1 blob");
        assert!(
            !raw.contains("gho_"),
            "file must not contain the GitHub access token in plaintext"
        );
        assert!(
            !raw.contains("api.githubcopilot.com"),
            "file must not contain the API base in plaintext"
        );
        assert!(
            !raw.contains("copilot"),
            "file must not contain field names or values in plaintext"
        );
        assert!(
            serde_json::from_str::<serde_json::Value>(&raw).is_err(),
            "saved file must not be parseable as the plaintext JSON of tokens"
        );
    }

    #[test]
    fn debug_output_contains_no_token_material() {
        let creds = sample_credentials();
        let debug = format!("{:?}", creds);
        assert!(!debug.contains("gho_"), "GitHub token must be redacted");
        assert!(!debug.contains("tid="), "Copilot token must be redacted");
        assert!(debug.contains("<redacted>"));
        assert!(debug.contains("api.githubcopilot.com"));
        assert!(debug.contains("6f9619ff"));
        assert!(debug.contains("githubcopilot"));
    }

    #[test]
    fn corrupt_blob_fails_without_plaintext_fallback() {
        let store = unique_store();
        fs::write(store.file_path(), "enc-v1:AAAA:BBBB").expect("write garbage");
        assert!(store.load().is_err(), "tampered blob must not load");
    }

    #[test]
    fn default_path_lives_next_to_oauth_tokens_enc() {
        let dir = Path::new("/secure/storage");
        let copilot = CopilotTokenStore::default_path(dir);
        assert_eq!(
            copilot.file_name().and_then(|n| n.to_str()),
            Some("copilot_tokens.enc")
        );
        assert_eq!(copilot.parent(), Some(dir));
    }

    proptest! {
        #![proptest_config(ProptestConfig::with_cases(8))]
        /// Pure crypto round trip without touching the shared env.
        #[test]
        fn prop_round_trip_with_key(
            access_token in "[a-zA-Z0-9_]{8,64}",
            token in proptest::option::of("[a-zA-Z0-9_]{8,64}"),
            expires_at in proptest::option::of(0u64..4_102_444_800),
            updated_at in 0u64..4_102_444_800,
            key_bytes in proptest::collection::vec(0u8..=255, 32)
        ) {
            let mut key = [0u8; 32];
            key.copy_from_slice(&key_bytes);
            let creds = CopilotStoredCredentials {
                github_access_token: access_token.clone(),
                copilot_token: token.clone(),
                copilot_expires_at: expires_at,
                api_base: "https://api.githubcopilot.com".into(),
                machine_id: "m".into(),
                sku: None,
                scopes: "copilot".into(),
                updated_at,
            };
            let plaintext = serde_json::to_string(&creds).unwrap();
            let ciphertext = encrypt_provider_secret_with_key(&key, &plaintext).unwrap();
            let decrypted = decrypt_provider_secret_with_key(&key, &ciphertext).unwrap();
            let parsed: CopilotStoredCredentials = serde_json::from_str(&decrypted).unwrap();
            prop_assert_eq!(parsed.github_access_token, access_token);
            prop_assert_eq!(parsed.copilot_token, token);
            prop_assert_eq!(parsed.copilot_expires_at, expires_at);
            prop_assert_eq!(parsed.updated_at, updated_at);
        }
    }
}
