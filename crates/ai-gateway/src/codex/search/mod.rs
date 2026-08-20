//! Codex search submodule.
//!
//! Provides transparent web-search tool injection, interception, and execution
//! for the Codex translation pipeline. The search functionality is composed of
//! six cooperating modules:
//!
//! - [`config`]: configuration for the search feature
//! - [`models`]: request/response and tool-definition data models
//! - [`metrics`]: collection and reporting of search usage metrics
//! - [`injector`]: injects search tool definitions into outbound requests
//! - [`executor`]: executes search tool calls and returns results
//! - [`interceptor`]: intercepts search tool calls from streamed responses

pub mod config;
pub mod executor;
pub mod injector;
pub mod interceptor;
pub mod metrics;
pub mod models;

pub use config::CodexSearchConfig;
pub use executor::SearchExecutor;
pub use injector::ToolInjector;
pub use interceptor::ToolInterceptor;
pub use metrics::SearchMetrics;
