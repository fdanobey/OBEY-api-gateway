//! Response caching for the OBEY API gateway.
//!
//! Two cooperating tiers:
//!
//! * [`ExactCache`] — in-memory SHA-256 keyed cache that catches byte-for-byte
//!   identical retries with no embedding round-trip.  Default-on, zero
//!   external dependencies.
//! * [`SemanticCache`] — Qdrant-backed embedding similarity cache that catches
//!   paraphrased requests.  Optional, requires Qdrant and an embedding
//!   provider.
//!
//! Both tiers gate eligibility on [`is_cache_eligible`] so writes and reads
//! agree.

pub mod exact;
pub mod semantic;

pub use exact::ExactCache;
pub use semantic::SemanticCache;
