#![allow(dead_code)]
//! GitHub Copilot OAuth module family (device-flow login).
//!
//! Submodules implement the credential lifecycle for the `github-copilot`
//! provider type per the `github-copilot-oauth-provider` spec:
//! - [`store`]: encrypted at-rest persistence for Copilot credentials
//!   (`copilot_tokens.enc`, `enc-v1` AES-256-GCM, colocated with
//!   `oauth_tokens.enc` and `master.key`)
//!
//! Sibling submodules: `catalog` (task 8: entitlement-filtered model
//! catalog synced from `{api_base}/models` on the existing catalog
//! cadence) and `copilot_token` implements the GitHub-token →
//! Copilot-token exchange (Req 3.1) and `manager` owns the whole
//! credential lifecycle (task 5: session states, login dedupe,
//! background refresh, inline refresh-once, logout).
//!
//! The module root also hosts small helpers shared across the family,
//! starting with stable machine-id generation for the persisted credential
//! payload (design "Machine/session ids": generated ids live inside the
//! encrypted store, never in a plaintext file).

pub mod catalog;
pub mod copilot_token;
pub mod device_flow;
pub mod manager;
pub mod store;

use rand::rngs::SysRng;
use rand::TryRng;

/// Generate a fresh, random, UUID-v4-shaped machine identifier.
///
/// Persisted inside the encrypted [`store::CopilotStoredCredentials`]
/// payload (never a plaintext file) and reused across restarts so the
/// gateway presents a stable machine id to the Copilot API, mirroring the
/// machine-id behavior of first-party clients without fabricating device
/// telemetry (design Cut List).
///
/// Not a credential: the value is an opaque random identifier and is safe
/// to surface in non-secret diagnostics, unlike the token fields.
pub fn generate_machine_id() -> String {
    let mut bytes = [0u8; 16];
    SysRng.try_fill_bytes(&mut bytes).expect("OS RNG failed while generating machine id");
    // Force UUID version 4 (random) and RFC 4122 variant bits so the id is
    // shape-compatible with the machine ids emitted by first-party clients.
    bytes[6] = (bytes[6] & 0x0f) | 0x40;
    bytes[8] = (bytes[8] & 0x3f) | 0x80;

    let hex: String = bytes.iter().map(|b| format!("{:02x}", b)).collect();
    format!(
        "{}-{}-{}-{}-{}",
        &hex[0..8],
        &hex[8..12],
        &hex[12..16],
        &hex[16..20],
        &hex[20..32]
    )
}

#[cfg(test)]
pub(crate) mod test_support {
//! Shared test scaffolding for the Copilot family's lib test modules
//! (store, manager, catalog, and the providers adapter). Integration
//! tests in `tests/` keep their own copies — separate crates can't
//! import `#[cfg(test)]` items cleanly.

use super::store::CopilotTokenStore;

/// Hermetic secure-storage dir shared by the Copilot lib tests.
///
/// The shared unit-test secrets dir (`secrets::test_support`), isolated
/// from the developer machine with its master key already created. Each
/// test uses its own unique file name inside it, so parallel tests never
/// share a credential file.
pub(crate) fn secure_dir() -> &'static std::path::Path {
crate::secrets::test_support::shared_data_dir()
}

/// Unique credential store inside the shared secure dir for one test,
/// named after `prefix` so sibling test modules never collide.
pub(crate) fn unique_store(prefix: &str) -> CopilotTokenStore {
static COUNTER: std::sync::atomic::AtomicU64 = std::sync::atomic::AtomicU64::new(0);
let n = COUNTER.fetch_add(1, std::sync::atomic::Ordering::SeqCst);
CopilotTokenStore::new(secure_dir().join(format!("{prefix}_{n}.enc")))
}

/// Plain reqwest client for wiremock-driven tests.
pub(crate) fn test_client() -> reqwest::Client {
reqwest::Client::builder().build().unwrap()
}

/// Current wall-clock time plus `secs`, clamped at 0 (same helper as
/// the OpenAI manager).
pub(crate) fn now_plus(secs: u64) -> u64 {
now_unix_secs().saturating_add(secs)
}

/// Current wall-clock time as seconds since the Unix epoch, clamped
/// at 0.
pub(crate) fn now_unix_secs() -> u64 {
std::time::SystemTime::now()
.duration_since(std::time::UNIX_EPOCH)
.map(|d| d.as_secs())
.unwrap_or(0)
}
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn generated_machine_ids_are_uuid_shaped_and_unique() {
        let a = generate_machine_id();
        let b = generate_machine_id();
        assert_ne!(a, b, "machine ids must not collide across calls");
        assert_eq!(a.len(), 36);
        let groups: Vec<&str> = a.split('-').collect();
        assert_eq!(
            groups.iter().map(|g| g.len()).collect::<Vec<_>>(),
            vec![8, 4, 4, 4, 12],
            "machine id must be UUID-shaped"
        );
        assert!(groups[2].starts_with('4'), "version nibble must be 4");
        assert!(
            groups[3].starts_with('8')
                || groups[3].starts_with('9')
                || groups[3].starts_with('a')
                || groups[3].starts_with('b'),
            "variant nibble must be RFC 4122"
        );
        assert!(a
            .chars()
            .all(|c| c.is_ascii_hexdigit() || c == '-'));
    }
}
