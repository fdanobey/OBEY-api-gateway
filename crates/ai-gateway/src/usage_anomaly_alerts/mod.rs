//! Per-key usage-anomaly detection against a 7-day rolling
//! baseline (spec: usage-anomaly-alerts).
//!
//! Current surface: the pure evaluation core ([`state`]), the
//! ledger aggregates ([`query`]), and the emission orchestrator
//! ([`emission`]).
//!
//! _Requirements: 1, 2_

pub mod emission;
pub mod query;
pub mod state;
