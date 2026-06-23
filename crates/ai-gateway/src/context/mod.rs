//! Context management for automatic context window handling.
//!
//! This module provides automatic context truncation when requests exceed
//! a model's context window limits. It supports multiple truncation strategies
//! and can be configured per model group.

mod manager;
mod strategies;

pub use manager::ContextManager;
