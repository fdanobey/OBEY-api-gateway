use base64::{engine::general_purpose::STANDARD, Engine};
use rand::RngCore;
use ring::aead::{Aad, LessSafeKey, Nonce, UnboundKey, AES_256_GCM, NONCE_LEN};
use std::fs;
use std::path::PathBuf;

const APP_DIR_NAME: &str = "ai-gateway";
/// Explicit override for the secrets/master-key directory. Recommended for
/// containers (Docker) where `HOME`/`XDG_CONFIG_HOME`/`APPDATA` are unset;
/// point it at a mounted volume so the master key (and encrypted secrets)
/// persist across restarts.
const DATA_DIR_ENV: &str = "AI_GATEWAY_DATA_DIR";
const MASTER_KEY_FILE: &str = "master.key";
const ENCRYPTED_PREFIX: &str = "enc-v1:";
const NONCE_SIZE: usize = NONCE_LEN;
const KEY_SIZE: usize = 32;

#[derive(Debug, thiserror::Error)]
pub enum SecretError {
    #[error("Secure local storage directory is unavailable")]
    StorageDirectoryUnavailable,
    #[error("Failed to create secure local storage directory: {0}")]
    CreateStorageDirectory(String),
    #[error("Failed to read master key: {0}")]
    ReadMasterKey(String),
    #[error("Failed to write master key: {0}")]
    WriteMasterKey(String),
    #[error("Master key data is invalid")]
    InvalidMasterKey,
    #[error("Encrypted secret format is invalid")]
    InvalidEncryptedFormat,
    #[error("Failed to encrypt provider secret")]
    EncryptFailed,
    #[error("Failed to decrypt provider secret")]
    DecryptFailed,
}

pub fn load_or_create_master_key() -> Result<[u8; KEY_SIZE], SecretError> {
    let key_path = master_key_path()?;

    if key_path.exists() {
        return read_master_key(&key_path);
    }

    if let Some(parent) = key_path.parent() {
        fs::create_dir_all(parent)
            .map_err(|e| SecretError::CreateStorageDirectory(e.to_string()))?;
    }

    let mut key = [0u8; KEY_SIZE];
    rand::rngs::OsRng.fill_bytes(&mut key);

    fs::write(&key_path, STANDARD.encode(key))
        .map_err(|e| SecretError::WriteMasterKey(e.to_string()))?;

    set_owner_only_permissions(&key_path)
        .map_err(|e| SecretError::WriteMasterKey(e.to_string()))?;

    Ok(key)
}

pub fn encrypt_provider_secret(plaintext: &str) -> Result<String, SecretError> {
    let key = load_or_create_master_key()?;
    encrypt_provider_secret_with_key(&key, plaintext)
}

pub fn decrypt_provider_secret(ciphertext: &str) -> Result<String, SecretError> {
    let key = load_or_create_master_key()?;
    decrypt_provider_secret_with_key(&key, ciphertext)
}

pub fn encrypt_provider_secret_with_key(
    key: &[u8; KEY_SIZE],
    plaintext: &str,
) -> Result<String, SecretError> {
    let mut nonce_bytes = [0u8; NONCE_SIZE];
    rand::rngs::OsRng.fill_bytes(&mut nonce_bytes);
    let nonce = Nonce::assume_unique_for_key(nonce_bytes);
    let unbound = UnboundKey::new(&AES_256_GCM, key).map_err(|_| SecretError::EncryptFailed)?;
    let cipher = LessSafeKey::new(unbound);
    let mut in_out = plaintext.as_bytes().to_vec();

    cipher
        .seal_in_place_append_tag(nonce, Aad::empty(), &mut in_out)
        .map_err(|_| SecretError::EncryptFailed)?;

    Ok(format!(
        "{}{}:{}",
        ENCRYPTED_PREFIX,
        STANDARD.encode(nonce_bytes),
        STANDARD.encode(in_out)
    ))
}

pub fn decrypt_provider_secret_with_key(
    key: &[u8; KEY_SIZE],
    ciphertext: &str,
) -> Result<String, SecretError> {
    let payload = ciphertext
        .strip_prefix(ENCRYPTED_PREFIX)
        .ok_or(SecretError::InvalidEncryptedFormat)?;
    let (nonce_b64, data_b64) = payload
        .split_once(':')
        .ok_or(SecretError::InvalidEncryptedFormat)?;

    let nonce_vec = STANDARD
        .decode(nonce_b64)
        .map_err(|_| SecretError::InvalidEncryptedFormat)?;
    if nonce_vec.len() != NONCE_SIZE {
        return Err(SecretError::InvalidEncryptedFormat);
    }

    let encrypted = STANDARD
        .decode(data_b64)
        .map_err(|_| SecretError::InvalidEncryptedFormat)?;

    let mut nonce_bytes = [0u8; NONCE_SIZE];
    nonce_bytes.copy_from_slice(&nonce_vec);
    let nonce = Nonce::assume_unique_for_key(nonce_bytes);
    let unbound = UnboundKey::new(&AES_256_GCM, key).map_err(|_| SecretError::DecryptFailed)?;
    let cipher = LessSafeKey::new(unbound);
    let mut in_out = encrypted;
    let decrypted = cipher
        .open_in_place(nonce, Aad::empty(), &mut in_out)
        .map_err(|_| SecretError::DecryptFailed)?;

    String::from_utf8(decrypted.to_vec()).map_err(|_| SecretError::DecryptFailed)
}

pub fn is_encrypted_secret(value: &str) -> bool {
    value.starts_with(ENCRYPTED_PREFIX)
}

pub fn looks_like_plaintext_secret(value: &str) -> bool {
    let trimmed = value.trim();
    if trimmed.is_empty() || is_encrypted_secret(trimmed) {
        return false;
    }

    // Check known secret prefixes FIRST (before env var check)
    // This ensures AWS keys like AKIAIOSFODNN7EXAMPLE are recognized as secrets
    // even though they look like env var references (all uppercase + digits)
    let known_prefixes = [
        "sk-", "sk-proj-", "nvapi-", "ghp_", "gho_", "gsk_", "xai-", "AIza", "Bearer ",
        "AKIA", "ASIA", "AIDA", "ABSK",  // AWS access key IDs (permanent, temporary STS, IAM user, Bedrock)
    ];

    if known_prefixes.iter().any(|prefix| trimmed.starts_with(prefix)) {
        return true;
    }

    // After checking known prefixes, reject env var references
    if is_env_var_reference(trimmed) {
        return false;
    }

    let has_lowercase = trimmed.chars().any(|c| c.is_ascii_lowercase());
    let has_digit = trimmed.chars().any(|c| c.is_ascii_digit());
    let has_separator = trimmed.contains('-') || trimmed.contains('_');
    let long_enough = trimmed.len() >= 24;

    long_enough && has_lowercase && has_digit && has_separator
}

pub fn is_env_var_reference(value: &str) -> bool {
    let trimmed = value.trim();
    !trimmed.is_empty()
        && trimmed
            .chars()
            .all(|c| c.is_ascii_uppercase() || c.is_ascii_digit() || c == '_')
        && trimmed.chars().next().is_some_and(|c| c.is_ascii_uppercase() || c == '_')
}

pub fn master_key_path() -> Result<PathBuf, SecretError> {
    Ok(storage_dir()?.join(MASTER_KEY_FILE))
}

/// Crate-visible shim for resolving the secrets directory path so other
/// modules (e.g. the Codex `InstructionsStore`) can colocate their on-disk
/// caches alongside the master key and OAuth token blob without exposing
/// the private `storage_dir()` helper.
pub(crate) fn storage_dir_path() -> Result<PathBuf, SecretError> {
    storage_dir()
}

/// Resolve the per-user secrets directory (colocated with the master key and
/// OAuth token store). Crate-visible so the Codex `InstructionsStore` can
/// colocate its disk cache with the OAuth blob (design §5).
///
/// Resolution order:
/// 1. `AI_GATEWAY_DATA_DIR` (explicit override; recommended for containers)
/// 2. `%APPDATA%/ai-gateway` (Windows)
/// 3. `$XDG_CONFIG_HOME/ai-gateway`
/// 4. `$HOME/.config/ai-gateway`
/// 5. `./.ai-gateway` relative to the current working directory (fallback so
///    headless/container environments without HOME never hard-fail)
pub(crate) fn storage_dir() -> Result<PathBuf, SecretError> {
    let appdata = if cfg!(target_os = "windows") {
        std::env::var("APPDATA").ok()
    } else {
        None
    };

    resolve_storage_dir(
        std::env::var(DATA_DIR_ENV).ok(),
        appdata,
        std::env::var("XDG_CONFIG_HOME").ok(),
        std::env::var("HOME").ok(),
        std::env::current_dir().ok(),
    )
}

/// Pure resolution of the secrets directory from candidate inputs. Separated
/// from environment lookups so the precedence rules are unit-testable.
fn resolve_storage_dir(
    data_dir_override: Option<String>,
    appdata: Option<String>,
    xdg_config_home: Option<String>,
    home: Option<String>,
    current_dir: Option<PathBuf>,
) -> Result<PathBuf, SecretError> {
    let is_set = |value: &str| !value.trim().is_empty();

    if let Some(dir) = data_dir_override.filter(|v| is_set(v)) {
        return Ok(PathBuf::from(dir));
    }

    if let Some(appdata) = appdata.filter(|v| is_set(v)) {
        return Ok(PathBuf::from(appdata).join(APP_DIR_NAME));
    }

    if let Some(xdg) = xdg_config_home.filter(|v| is_set(v)) {
        return Ok(PathBuf::from(xdg).join(APP_DIR_NAME));
    }

    if let Some(home) = home.filter(|v| is_set(v)) {
        return Ok(PathBuf::from(home).join(".config").join(APP_DIR_NAME));
    }

    // Final fallback: keep secrets next to the working directory so headless
    // and container deployments (no HOME/XDG/APPDATA) still function. Operators
    // should mount AI_GATEWAY_DATA_DIR for durable, owner-scoped storage.
    if let Some(cwd) = current_dir {
        return Ok(cwd.join(".ai-gateway"));
    }

    Err(SecretError::StorageDirectoryUnavailable)
}

fn read_master_key(path: &PathBuf) -> Result<[u8; KEY_SIZE], SecretError> {
    let encoded = fs::read_to_string(path).map_err(|e| SecretError::ReadMasterKey(e.to_string()))?;
    let decoded = STANDARD
        .decode(encoded.trim())
        .map_err(|_| SecretError::InvalidMasterKey)?;

    if decoded.len() != KEY_SIZE {
        return Err(SecretError::InvalidMasterKey);
    }

    let mut key = [0u8; KEY_SIZE];
    key.copy_from_slice(&decoded);
    Ok(key)
}

/// Restrict `path` to owner-only read/write (`0600`) on Unix; no-op elsewhere.
///
/// Shared across `secrets` (master key) and `oauth::store` (encrypted token
/// blob) so both at-rest artifacts land with the same hardened permissions.
#[cfg(unix)]
pub(crate) fn set_owner_only_permissions(path: &PathBuf) -> std::io::Result<()> {
    use std::os::unix::fs::PermissionsExt;

    let permissions = PermissionsExt::from_mode(0o600);
    fs::set_permissions(path, permissions)
}

#[cfg(not(unix))]
pub(crate) fn set_owner_only_permissions(_path: &PathBuf) -> std::io::Result<()> {
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_encrypt_decrypt_round_trip() {
        let key = [7u8; KEY_SIZE];
        let plaintext = "sk-test-secret-1234567890";

        let encrypted = encrypt_provider_secret_with_key(&key, plaintext).unwrap();
        assert!(is_encrypted_secret(&encrypted));

        let decrypted = decrypt_provider_secret_with_key(&key, &encrypted).unwrap();
        assert_eq!(decrypted, plaintext);
    }

    #[test]
    fn test_detect_env_var_reference() {
        assert!(is_env_var_reference("OPENAI_API_KEY"));
        assert!(is_env_var_reference("_LOCAL_SECRET_1"));
        assert!(!is_env_var_reference("sk-test-123"));
        assert!(!is_env_var_reference("OpenAiKey"));
    }

    #[test]
    fn test_detect_plaintext_secret() {
        assert!(looks_like_plaintext_secret("sk-test-12345678901234567890"));
        assert!(looks_like_plaintext_secret("nvapi-1234567890abcdefghijklmnop"));
        assert!(!looks_like_plaintext_secret("OPENAI_API_KEY"));
        assert!(!looks_like_plaintext_secret("enc-v1:abc:def"));
    }

    #[test]
    fn test_detect_aws_access_key_as_plaintext() {
        // AWS permanent access key (starts with AKIA)
        assert!(looks_like_plaintext_secret("AKIAIOSFODNN7EXAMPLE"));
        // AWS temporary STS credentials (starts with ASIA)
        assert!(looks_like_plaintext_secret("ASIAIOSFODNN7EXAMPLE"));
        // AWS IAM user key (starts with AIDA)
        assert!(looks_like_plaintext_secret("AIDAIOSFODNN7EXAMPLE"));
        // AWS Bedrock API key (starts with ABSK)
        assert!(looks_like_plaintext_secret("ABSKIOSFODNN7EXAMPLE"));
    }

    #[test]
    fn test_reject_invalid_encrypted_format() {
        let key = [9u8; KEY_SIZE];
        let err = decrypt_provider_secret_with_key(&key, "bad-value").unwrap_err();
        assert!(matches!(err, SecretError::InvalidEncryptedFormat));
    }

    #[test]
    fn test_resolve_storage_dir_prefers_explicit_override() {
        let dir = resolve_storage_dir(
            Some("/data/secrets".to_string()),
            Some("C:/AppData".to_string()),
            Some("/xdg".to_string()),
            Some("/home/user".to_string()),
            Some(PathBuf::from("/cwd")),
        )
        .unwrap();
        assert_eq!(dir, PathBuf::from("/data/secrets"));
    }

    #[test]
    fn test_resolve_storage_dir_falls_back_to_cwd_when_env_unset() {
        // Reproduces the Docker case: no APPDATA/XDG/HOME set.
        let dir = resolve_storage_dir(None, None, None, None, Some(PathBuf::from("/app")))
            .unwrap();
        assert_eq!(dir, PathBuf::from("/app").join(".ai-gateway"));
    }

    #[test]
    fn test_resolve_storage_dir_ignores_blank_values() {
        let dir = resolve_storage_dir(
            Some("   ".to_string()),
            None,
            Some(String::new()),
            Some("/home/user".to_string()),
            Some(PathBuf::from("/cwd")),
        )
        .unwrap();
        assert_eq!(dir, PathBuf::from("/home/user").join(".config").join(APP_DIR_NAME));
    }

    #[test]
    fn test_resolve_storage_dir_unavailable_without_any_source() {
        let err = resolve_storage_dir(None, None, None, None, None).unwrap_err();
        assert!(matches!(err, SecretError::StorageDirectoryUnavailable));
    }
}
