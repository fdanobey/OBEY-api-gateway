//! PKCE code_verifier/code_challenge and CSRF state generation.
//!
//! Implements RFC 7636 (PKCE) with the `S256` challenge method and a
//! base64url-no-padding state parameter, per
//! [`design.md` §1 "PKCE Module"](../../../../.kiro/specs/openai-oauth-login/design.md).

use base64::{engine::general_purpose::URL_SAFE_NO_PAD, Engine};
use rand::{rngs::OsRng, RngCore};
use ring::digest::{digest, SHA256};

/// A PKCE verifier/challenge pair produced by [`generate_pkce`].
///
/// `code_verifier` is the secret held by the client and submitted during
/// token exchange; `code_challenge` is the derived public value sent on the
/// authorization request (`code_challenge_method=S256`).
pub struct PkceChallenge {
    /// The 43-character base64url-no-pad verifier (32 random bytes).
    pub code_verifier: String,
    /// `base64url_no_pad(SHA256(code_verifier))`.
    pub code_challenge: String,
}

/// Generate a fresh PKCE verifier/challenge pair using the OS CSPRNG.
///
/// - 32 random bytes → base64url-no-pad verifier (43 chars)
/// - SHA-256 of the verifier ASCII bytes → base64url-no-pad challenge (43 chars)
pub fn generate_pkce() -> PkceChallenge {
    let mut verifier_bytes = [0u8; 32];
    OsRng.fill_bytes(&mut verifier_bytes);
    let code_verifier = URL_SAFE_NO_PAD.encode(verifier_bytes);

    let hash = digest(&SHA256, code_verifier.as_bytes());
    let code_challenge = URL_SAFE_NO_PAD.encode(hash.as_ref());

    PkceChallenge {
        code_verifier,
        code_challenge,
    }
}

/// Generate a CSRF `state` parameter: 16 random bytes encoded as
/// base64url-no-pad (22 characters).
pub fn generate_state() -> String {
    let mut state_bytes = [0u8; 16];
    OsRng.fill_bytes(&mut state_bytes);
    URL_SAFE_NO_PAD.encode(state_bytes)
}

#[cfg(test)]
mod tests {
    use super::*;
    use proptest::prelude::*;

    // Feature: openai-oauth-login, Property 1: PKCE Verifier Format
    // **Validates: Requirements 1.2, 1.3**
    //
    // For all generated PKCE pairs, the `code_verifier` is exactly 43 characters
    // long and contains only base64url characters `[A-Za-z0-9_-]`.
    //
    // `generate_pkce()` draws entropy from `OsRng`, so we drive many iterations
    // by threading a throwaway proptest-generated seed through the test body;
    // each invocation samples a fresh verifier from the RNG.
    proptest! {
        #![proptest_config(ProptestConfig {
            cases: 64,
            .. ProptestConfig::default()
        })]

        #[test]
        fn prop_pkce_verifier_format(_seed in any::<u64>()) {
            let pkce = generate_pkce();

            // 32 random bytes → base64url-no-pad → 43 characters.
            prop_assert_eq!(
                pkce.code_verifier.len(),
                43,
                "code_verifier must be exactly 43 characters, got {}: {:?}",
                pkce.code_verifier.len(),
                pkce.code_verifier
            );

            // base64url alphabet: A-Z, a-z, 0-9, '-', '_' (no padding).
            for c in pkce.code_verifier.chars() {
                prop_assert!(
                    c.is_ascii_alphanumeric() || c == '-' || c == '_',
                    "code_verifier contains non-base64url character {:?} in {:?}",
                    c,
                    pkce.code_verifier
                );
            }
        }
    }

    // Feature: openai-oauth-login, Property 2: PKCE Challenge Derivation
    // **Validates: Requirements 1.3, 8.1**
    //
    // For all generated PKCE pairs, the `code_challenge` equals the
    // base64url-no-pad encoding of `SHA-256(code_verifier_ascii_bytes)`.
    // This pins the implementation to RFC 7636 S256 so any drift (wrong
    // hash, padding, alphabet, or hashing the raw 32 bytes instead of the
    // encoded verifier string) is detected.
    //
    // The `_seed` input only drives proptest iteration count; entropy for
    // the PKCE pair comes from `OsRng` inside `generate_pkce()`.
    proptest! {
        #![proptest_config(ProptestConfig {
            cases: 64,
            .. ProptestConfig::default()
        })]

        #[test]
        fn prop_pkce_challenge_derivation(_seed in any::<u64>()) {
            let pkce = generate_pkce();

            // Independently recompute the expected challenge from the
            // verifier using the same primitives the spec mandates
            // (SHA-256 over the ASCII verifier, then base64url-no-pad).
            let expected_hash = digest(&SHA256, pkce.code_verifier.as_bytes());
            let expected_challenge = URL_SAFE_NO_PAD.encode(expected_hash.as_ref());

            prop_assert_eq!(
                &pkce.code_challenge,
                &expected_challenge,
                "code_challenge must equal base64url_no_pad(SHA256(code_verifier)); \
                 verifier={:?}, got challenge={:?}, expected={:?}",
                pkce.code_verifier,
                pkce.code_challenge,
                expected_challenge
            );

            // SHA-256 → 32 bytes → base64url-no-pad → 43 chars.
            prop_assert_eq!(
                pkce.code_challenge.len(),
                43,
                "code_challenge must be 43 chars (SHA-256 → base64url-no-pad), got {}: {:?}",
                pkce.code_challenge.len(),
                pkce.code_challenge
            );
        }
    }

    // Feature: openai-oauth-login, Property 3: State Parameter Minimum Length
    // **Validates: Requirement 1.4**
    //
    // For all generated state parameters, the decoded byte length is at least
    // 16 bytes and the string uses only base64url-no-pad alphabet characters.
    // This guarantees sufficient CSRF entropy (≥128 bits) regardless of the
    // specific encoded string length.
    //
    // The `_seed` input only drives proptest iteration count; entropy for the
    // state value comes from `OsRng` inside `generate_state()`.
    proptest! {
        #![proptest_config(ProptestConfig {
            cases: 64,
            .. ProptestConfig::default()
        })]

        #[test]
        fn prop_state_parameter_min_length(_seed in any::<u64>()) {
            let state = generate_state();

            // Non-empty sanity check.
            prop_assert!(
                !state.is_empty(),
                "generate_state() returned an empty string"
            );

            // All characters must be in the base64url (no-pad) alphabet.
            for c in state.chars() {
                prop_assert!(
                    c.is_ascii_alphanumeric() || c == '-' || c == '_',
                    "state contains non-base64url character {:?} in {:?}",
                    c,
                    state
                );
            }

            // Decoded byte length must be at least 16 (≥128 bits of entropy).
            let decoded = URL_SAFE_NO_PAD
                .decode(&state)
                .expect("state must be valid base64url-no-pad");
            prop_assert!(
                decoded.len() >= 16,
                "decoded state must be >= 16 bytes, got {} bytes from {:?}",
                decoded.len(),
                state
            );
        }
    }
}
