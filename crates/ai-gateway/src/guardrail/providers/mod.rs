//! Guardrail provider backend implementations.
//!
//! Each submodule implements the `GuardrailProvider` trait for one backend
//! type. Submodules are declared as they are implemented in later tasks so the
//! crate keeps compiling:
//!
//! - `regex`       — RegexProvider (task 5).
//! - `presidio`    — PresidioProvider (task 7).
//! - `custom_http` — CustomHttpProvider (task 7).
//! - `moderation`  — OpenAiModerationProvider (task 7).
//! - `lakera`      — LakeraProvider (task 7).
//! - `semantic`    — SemanticProvider (task 8).

pub mod custom_http;
pub mod lakera;
pub mod moderation;
pub mod presidio;
pub mod regex;
pub mod semantic;
