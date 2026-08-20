#![allow(dead_code)]
//! Encrypted at-rest persistence for OpenAI OAuth tokens.
//!
//! Tokens are persisted to a single file on disk, encrypted with the router's
//! existing master key via [`crate::secrets::encrypt_provider_secret`] and
//! [`crate::secrets::decrypt_provider_secret`] (AES-256-GCM, `enc-v1:` format).
//! On Unix platforms the file is created with `0600` permissions mirroring the
//! pattern used for `master.key` in [`crate::secrets`].
//!
//! Only the struct scaffold and constructor are declared here. The
//! [`StoredTokens`] payload, `save` / `load` / `delete` methods, permission
//! hardening, and error mapping are filled in by subsequent subtasks
//! (2.2 – 2.7) of the `openai-oauth-login` spec.

use std::fmt;
use std::fs;
use std::path::{Path, PathBuf};

use serde::{Deserialize, Serialize};

use super::error::OAuthError;
use crate::secrets;

/// Filename used for the encrypted OAuth token blob, colocated with
/// `master.key` under the router's secure storage directory.
const OAUTH_TOKENS_FILENAME: &str = "oauth_tokens.enc";

/// Plaintext token payload persisted by [`OAuthTokenStore`].
///
/// Serialized as JSON before being handed to
/// [`crate::secrets::encrypt_provider_secret`]; the resulting `enc-v1:` blob
/// is what actually lands on disk. Kept `pub(crate)` because callers outside
/// the `oauth` module should only ever observe tokens via store methods that
/// enforce encryption and expiry semantics.
///
/// # Debug redaction
///
/// `access_token` and `refresh_token` are formatted as `<redacted>` by the
/// manual [`fmt::Debug`] implementation so that accidental `{:?}` logging,
/// `tracing` field capture, or panic messages cannot leak Bearer material
/// (Req 8.4: tokens never appear in any log output).
#[derive(Clone, Serialize, Deserialize)]
pub(crate) struct StoredTokens {
    /// OAuth access token used as the upstream Bearer credential.
    /// Never logged; see the `Debug` impl on this struct.
    pub access_token: String,
    /// OAuth refresh token used by the background refresh task.
    /// Never logged; see the `Debug` impl on this struct.
    pub refresh_token: String,
    /// Access-token expiry as a Unix timestamp in seconds.
    pub expires_at: u64,
    /// Space-separated scope string echoed back by the authorization server.
    pub scopes: String,
}

impl StoredTokens {
    /// Returns `true` once `now` has reached or passed `expires_at`.
    ///
    /// Both inputs are Unix timestamps in seconds. Intended to be called with
    /// `SystemTime::now()` converted to seconds since `UNIX_EPOCH`.
    pub(crate) fn is_expired(&self, now: u64) -> bool {
        self.expires_at <= now
    }

    /// Returns `true` when the access token is within `window_secs` of
    /// expiring (Req 5.2 calls this with `window_secs = 300`).
    ///
    /// Uses saturating subtraction so an already-expired token trivially
    /// needs a refresh regardless of window size.
    pub(crate) fn needs_refresh(&self, now: u64, window_secs: u64) -> bool {
        self.expires_at.saturating_sub(now) <= window_secs
    }
}

impl fmt::Debug for StoredTokens {
    /// Redacts both token fields to preserve the "tokens never logged"
    /// invariant (Req 8.4). Non-secret fields (`expires_at`, `scopes`) are
    /// emitted verbatim so expiry and scope diagnostics remain useful.
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("StoredTokens")
            .field("access_token", &"<redacted>")
            .field("refresh_token", &"<redacted>")
            .field("expires_at", &self.expires_at)
            .field("scopes", &self.scopes)
            .finish()
    }
}

/// Encrypted token store backed by a single file on disk.
///
/// Holds only the destination path; all cryptographic material is sourced
/// from [`crate::secrets`] on demand so this type is cheap to clone and
/// share across async tasks.
#[derive(Debug, Clone)]
pub struct OAuthTokenStore {
    file_path: PathBuf,
}

impl OAuthTokenStore {
    /// Create a new store that reads and writes tokens at `file_path`.
    ///
    /// The file is not created or touched until the first `save` call
    /// (added in task 2.3). Use [`OAuthTokenStore::default_path`] to derive
    /// the conventional location alongside the master key.
    pub fn new(file_path: impl Into<PathBuf>) -> Self {
        Self {
            file_path: file_path.into(),
        }
    }

    /// Default on-disk location: `{storage_dir}/oauth_tokens.enc`.
    ///
    /// Callers typically pass the parent of [`crate::secrets::master_key_path`]
    /// so the OAuth blob lives next to the master key it is encrypted with.
    pub fn default_path(storage_dir: &Path) -> PathBuf {
        storage_dir.join(OAUTH_TOKENS_FILENAME)
    }

    /// Path the store reads from and writes to.
    pub fn file_path(&self) -> &Path {
        &self.file_path
    }

    /// Encrypt and persist `tokens` to [`Self::file_path`].
    ///
    /// The payload is serialized to JSON, encrypted with the router's master
    /// key via [`crate::secrets::encrypt_provider_secret`] (AES-256-GCM,
    /// `enc-v1:` prefix), and written atomically-ish via [`fs::write`]. The
    /// parent directory is created on demand.
    ///
    /// On Unix the file's mode is tightened to `0600` immediately after
    /// write via [`secrets::set_owner_only_permissions`], matching the
    /// hardening already applied to `master.key` (Req 4.5). On non-Unix
    /// platforms that helper is a no-op.
    ///
    /// # Logging
    ///
    /// Arguments are skipped from the tracing span so neither the plaintext
    /// JSON nor the resulting ciphertext are ever captured — only the target
    /// path is recorded (Req 8.4).
    #[tracing::instrument(skip(self, tokens), fields(path = %self.file_path.display()))]
    pub(crate) fn save(&self, tokens: &StoredTokens) -> Result<(), OAuthError> {
        let json = serde_json::to_string(tokens)?;
        let ciphertext = secrets::encrypt_provider_secret(&json)?;

        if let Some(parent) = self.file_path.parent() {
            fs::create_dir_all(parent)?;
        }

        fs::write(&self.file_path, ciphertext)?;
        secrets::set_owner_only_permissions(&self.file_path)?;
        Ok(())
    }

    /// Load and decrypt tokens from [`Self::file_path`], returning `None` when
    /// the file does not yet exist (first-run / post-logout state) or when
    /// the on-disk blob is corrupted / undecryptable (Req 4.4).
    ///
    /// The on-disk blob is read as UTF-8, handed to
    /// [`crate::secrets::decrypt_provider_secret`] (AES-256-GCM, `enc-v1:`
    /// format), and the resulting JSON is parsed back into [`StoredTokens`].
    ///
    /// # Corruption handling (Req 4.4)
    ///
    /// * Missing file → `Ok(None)` (first-run / post-logout).
    /// * I/O error reading the file → propagated as [`OAuthError::Io`] since
    ///   this indicates an environmental problem, not corruption.
    /// * Decrypt failure ([`OAuthError::Crypto`]) → a warning is emitted via
    ///   `tracing::warn!` and `Ok(None)` is returned so the caller treats the
    ///   provider as unauthenticated and can re-initiate the login flow.
    /// * JSON parse failure ([`OAuthError::Serde`]) → same treatment as a
    ///   decrypt failure: warn and return `Ok(None)`.
    ///
    /// Only the underlying error `Display` is logged; neither the ciphertext,
    /// the plaintext, nor any token field is ever captured (Req 8.4).
    ///
    /// # Logging
    ///
    /// `self` is skipped from the tracing span so neither the ciphertext nor
    /// the decrypted token material is ever captured — only the target path
    /// is recorded (Req 8.4).
    #[tracing::instrument(skip(self), fields(path = %self.file_path.display()))]
    pub(crate) fn load(&self) -> Result<Option<StoredTokens>, OAuthError> {
        if !self.file_path.exists() {
            return Ok(None);
        }

        let ciphertext = fs::read_to_string(&self.file_path)?;
        let plaintext = match secrets::decrypt_provider_secret(&ciphertext) {
            Ok(p) => p,
            Err(e) => {
                tracing::warn!(
                    error = %e,
                    "OAuth token file failed to decrypt; treating as unauthenticated"
                );
                return Ok(None);
            }
        };
        let tokens = match serde_json::from_str::<StoredTokens>(&plaintext) {
            Ok(t) => t,
            Err(e) => {
                tracing::warn!(
                    error = %e,
                    "OAuth token file failed to parse; treating as unauthenticated"
                );
                return Ok(None);
            }
        };
        Ok(Some(tokens))
    }

    /// Remove the on-disk token file, if present.
    ///
    /// Idempotent: returns `Ok(())` when [`Self::file_path`] does not exist so
    /// logout flows can be invoked repeatedly (Req 6.x) without surfacing
    /// `NotFound` errors to the caller. When the file is present it is
    /// deleted via [`fs::remove_file`]; any I/O error propagates through the
    /// `#[from] std::io::Error` conversion on [`OAuthError`].
    ///
    /// # Logging
    ///
    /// `self` is skipped from the tracing span so only the target path is
    /// recorded; no token material is ever in scope here, but the skip keeps
    /// the instrumentation shape consistent with `save` / `load` (Req 8.4).
    #[tracing::instrument(skip(self), fields(path = %self.file_path.display()))]
    pub(crate) fn delete(&self) -> Result<(), OAuthError> {
        if !self.file_path.exists() {
            return Ok(());
        }

        fs::remove_file(&self.file_path)?;
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::secrets::{decrypt_provider_secret_with_key, encrypt_provider_secret_with_key};
    use proptest::prelude::*;

    // Feature: openai-oauth-login, Property 4: Token Store Round-Trip
    // **Validates: Requirements 4.1, 4.2**
    //
    // For all valid `StoredTokens` values,
    // `decrypt(encrypt(serialize(tokens))) == serialize(tokens)` — i.e. the
    // full save→load cycle returns the original bytes. This exercises the
    // same primitives `OAuthTokenStore::save` / `load` use (serde_json →
    // AES-256-GCM `enc-v1:` blob) via the `_with_key` variants from
    // `crate::secrets`, keeping the test hermetic: no `master.key` file is
    // read or created, no filesystem state leaks between proptest cases or
    // across parallel test threads, and the encryption key itself is drawn
    // from proptest so the property holds for arbitrary 256-bit keys.
    //
    // Field strategies:
    // * `access_token` / `refresh_token` — printable ASCII up to 256 chars
    //   covers real OpenAI Bearer material (base64url, JWT-shaped, etc.)
    //   plus empty strings for robustness; excludes control bytes that
    //   `serde_json` would escape in ways irrelevant to the round-trip.
    // * `expires_at` — full `u64` range; the store must survive any Unix
    //   timestamp including 0 and `u64::MAX`.
    // * `scopes` — space-separated scope alphabet (RFC 6749 §3.3 allows
    //   `%x21 / %x23-5B / %x5D-7E`; we keep it to the common subset
    //   actually emitted by `auth.openai.com`).
    // * `key_bytes` — arbitrary 32-byte AES-256 key.
    proptest! {
        #![proptest_config(ProptestConfig {
            cases: 64,
            .. ProptestConfig::default()
        })]

        #[test]
        fn prop_token_store_round_trip(
            access_token in "[ -~]{0,256}",
            refresh_token in "[ -~]{0,256}",
            expires_at in any::<u64>(),
            scopes in "[a-zA-Z0-9_\\- ]{0,128}",
            key_bytes in any::<[u8; 32]>(),
        ) {
            let tokens = StoredTokens {
                access_token,
                refresh_token,
                expires_at,
                scopes,
            };

            // 1. Serialize tokens to JSON (matches `save()`).
            let plaintext = serde_json::to_string(&tokens)
                .expect("StoredTokens must serialize as JSON");

            // 2. Encrypt via the same AES-256-GCM path used on disk.
            let ciphertext = encrypt_provider_secret_with_key(&key_bytes, &plaintext)
                .expect("encryption must succeed for valid plaintext");

            // 3. Decrypt with the same key (matches `load()`).
            let decrypted = decrypt_provider_secret_with_key(&key_bytes, &ciphertext)
                .expect("decryption must succeed for ciphertext produced by the same key");

            // Property 4 core invariant: bytes survive the round-trip.
            prop_assert_eq!(
                &decrypted,
                &plaintext,
                "encrypt→decrypt round-trip must yield the original serialized tokens"
            );

            // And the parsed struct recovers every field verbatim.
            // Compare field-by-field because `StoredTokens`' manual `Debug`
            // impl redacts token values (Req 8.4), which would defeat
            // `prop_assert_eq!`'s counterexample output for the whole struct.
            let restored: StoredTokens = serde_json::from_str(&decrypted)
                .expect("decrypted JSON must parse back into StoredTokens");
            prop_assert_eq!(&restored.access_token, &tokens.access_token);
            prop_assert_eq!(&restored.refresh_token, &tokens.refresh_token);
            prop_assert_eq!(restored.expires_at, tokens.expires_at);
            prop_assert_eq!(&restored.scopes, &tokens.scopes);
        }
    }

    // Feature: openai-oauth-login, Property 5: Refresh Trigger Timing
    // **Validates: Requirement 5.2**
    //
    // For all `(expires_at, now)` pairs where `expires_at > 0` and `now > 0`,
    // `StoredTokens::needs_refresh(now, 300)` SHALL return `true` if and
    // only if `expires_at - now <= 300` (saturating subtraction, so an
    // already-expired token always triggers refresh regardless of the
    // window). This pins the 5-minute pre-expiry refresh window used by
    // the background Token_Manager task (Req 5.2).
    //
    // The token string fields are irrelevant to the timing predicate, so
    // they are held at empty strings to keep the generator small and the
    // failure output readable. `expires_at` and `now` span `1..=u64::MAX`
    // to exercise boundary arithmetic around `u64::MAX - 300`, zero, and
    // all interior values.
    proptest! {
        #![proptest_config(ProptestConfig {
            cases: 64,
            .. ProptestConfig::default()
        })]

        #[test]
        fn prop_refresh_trigger_timing(
            expires_at in 1u64..=u64::MAX,
            now in 1u64..=u64::MAX,
        ) {
            const WINDOW: u64 = 300;
            let tokens = StoredTokens {
                access_token: String::new(),
                refresh_token: String::new(),
                expires_at,
                scopes: String::new(),
            };
            let triggered = tokens.needs_refresh(now, WINDOW);
            let expected = expires_at.saturating_sub(now) <= WINDOW;
            prop_assert_eq!(triggered, expected);
        }
    }
}
