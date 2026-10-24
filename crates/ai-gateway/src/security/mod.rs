//! Security hardening primitives (black-october-security-hardening spec).
//!
//! Currently home to the outbound-fetch SSRF guard ([`outbound_fetch`])
//! and the secret-redaction layer for log sinks ([`redact`]); later spec
//! components (CORS tightening) land here too.

pub mod outbound_fetch;
pub mod redact;

pub use outbound_fetch::{
    guarded_redirect_policy, std_resolve_host, validate_outbound_url, OutboundFetchConfig,
    OutboundFetchError,
};
pub use redact::redact_secrets;
