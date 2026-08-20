//! Codex backend translation pipeline.
//!
//! This module implements the Chat Completions ↔ Responses API translation
//! layer that allows OAuth-authenticated OpenAI providers to dispatch through
//! `https://chatgpt.com/backend-api/codex/responses`.
//!
//! Submodules are populated incrementally by subsequent tasks in the
//! `codex-backend-translation` spec. Public re-exports are added to this file
//! as each submodule's contents become available.

pub mod client;
pub mod effort_map;
pub mod errors;
pub mod instructions;
pub mod jwt;
pub mod model_map;
pub mod models_discovery;
pub mod search;
pub mod sse;
pub mod translate_request;
pub mod translate_response;

pub use crate::codex::instructions::InstructionsStore;
