//! Budget-alert webhook subsystem (budget-alerts-unified-webhooks spec).
//!
//! Current surface: the pure event envelope ([`events`]), the threshold
//! state machine ([`alerts`]), the durable event/delivery store
//! ([`ledger`]), the delivery engine ([`delivery`]), and the provider
//! evaluation hook ([`provider_hook`]) the router calls after durable
//! spend accrual. `events`/`alerts` are I/O-free and fully
//! unit-testable; delivery owns all network/signing/retry concerns
//! behind [`WebhookNotifier`]. Admin wiring arrives with later spec
//! tasks.
//!
//! _Requirements: 1.1-1.9, 2.1-2.3 (evaluation surface owned here)_

pub mod admin;
pub mod alerts;
pub mod delivery;
pub mod events;
pub mod ledger;
pub mod provider_hook;

pub use delivery::WebhookNotifier;
pub use provider_hook::ProviderBudgetAlerts;
