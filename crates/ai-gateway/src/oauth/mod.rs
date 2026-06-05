//! OpenAI OAuth 2.0 + PKCE login module.
//!
//! Submodules implement the full Authorization Code with PKCE flow:
//! - [`pkce`]: code_verifier / code_challenge / state generation
//! - [`error`]: [`OAuthError`] covering all failure modes
//! - [`store`]: encrypted at-rest token persistence
//! - [`token`]: authorization-code exchange and refresh
//! - [`callback`]: loopback HTTP server that receives the authorization code
//! - [`flow`]: flow orchestration, session state machine, background refresh
//!
//! Public API is re-exported from this module root. Implementations are added
//! across tasks 1.x – 5.x of the `openai-oauth-login` spec.

pub mod callback;
pub mod error;
pub mod flow;
pub mod manager;
pub mod pkce;
pub mod store;
pub mod token;
pub mod usage_tracker;

pub use callback::{start_callback_server, AuthorizationCode};
pub use error::OAuthError;
pub use flow::{FlowState, InitiationOutcome, OAuthFlow, OAuthSessionState};
// Re-exported for internal consumers in later subtasks (token exchange step).
#[allow(unused_imports)]
pub(crate) use flow::{build_authorization_url, AUTHORIZATION_ENDPOINT};
pub use manager::OAuthManager;
// Re-exported for the Property 8 test in task 5.9.
#[allow(unused_imports)]
pub(crate) use manager::backoff_delay_secs;
pub use pkce::{generate_pkce, generate_state, PkceChallenge};
pub use store::OAuthTokenStore;
pub use token::{
    exchange_code, refresh_access_token, TokenResponse, OPENAI_CLIENT_ID, TOKEN_ENDPOINT,
};
pub use usage_tracker::UsageTracker;

// Additional re-exports are added as each submodule lands in subsequent tasks
// (5.x flow).
