//! Daily performance-drift detection against a 7-day rolling
//! baseline (spec: performance-alerts).
//!
//! Current surface: the pure evaluation core ([`state`]). The ledger
//! aggregates ([`query`]) and the emission orchestrator ([`emission`])
//! arrive with later spec tasks.
//!
//! _Requirements: 2, 3_

pub mod emission;
pub mod query;
pub mod state;
