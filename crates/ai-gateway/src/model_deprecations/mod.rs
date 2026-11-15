//! Model deprecation detection and alerting (model-deprecation-alerts
//! spec).
//!
//! Current surface: the pure event envelope ([`events`]) plus the
//! emission orchestrator ([`emission`]) bridging the state core to
//! the generalized webhook delivery seam. The scanner's background
//! loop and admin wiring arrive with later spec tasks.
//!
//! _Requirements: 3.3, 4.5 (event surface owned here)_

pub mod emission;
pub mod events;
pub mod scanner;
pub mod state;
pub mod hook;
