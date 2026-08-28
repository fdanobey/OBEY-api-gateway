//! Guardrail provider backend implementations.
//!
//! Each submodule implements the `GuardrailProvider` trait for one backend
//! type. Submodules are declared as they are implemented in later tasks so the
//! crate keeps compiling:
//!
//! - `regex` â€” RegexProvider (task 5).
//! - `presidio` â€” PresidioProvider (task 7).
//! - `custom_http` â€” CustomHttpProvider (task 7).
//! - `moderation` â€” OpenAiModerationProvider (task 7).
//! - `lakera` â€” LakeraProvider (task 7).
//! - `semantic` â€” SemanticProvider (task 8).
//! - `unicode_stego` â€” UnicodeStegoProvider (indirect-injection defense).

pub mod custom_http;
pub mod lakera;
pub mod moderation;
pub mod presidio;
pub mod regex;
pub mod semantic;
pub mod unicode_stego;
