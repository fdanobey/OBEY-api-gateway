//! Token-compression pipeline scaffolding and public API.

pub mod caveman;
pub mod config;
pub mod engines;
pub mod pipeline;
pub mod precompressed;
pub mod protection;
pub mod stats;
pub mod token_counter;

#[cfg(test)]
mod aging_property_tests;
#[cfg(test)]
mod critical_property_tests;
#[cfg(test)]
mod no_increase_property_tests;
#[cfg(test)]
mod pipeline_property_tests;
#[cfg(test)]
mod property_tests;
#[cfg(test)]
mod protection_property_tests;
#[cfg(test)]
mod secret_property_tests;
#[cfg(test)]
mod stats_property_tests;
#[cfg(test)]
mod tool_pair_property_tests;
#[cfg(test)]
mod tool_schema_property_tests;

pub use engines::{
    CompressiblePayload, CompressionContext, CompressionEngine, CompressionLevel, EngineResult,
};
