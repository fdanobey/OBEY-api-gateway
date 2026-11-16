pub mod backends;
pub mod config;
pub mod coordinator;
pub mod injector;
pub mod metrics;
pub mod models;

pub use backends::{BackendError, SearchBackend};
pub use coordinator::{ActiveSearchConfig, WebSearchCoordinator};
