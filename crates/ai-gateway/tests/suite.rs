//! Single integration-test binary for `ai-gateway`.
//!
//! Each file under `tests/` used to be its own test crate, so `cargo test`
//! linked ~80 binaries (~90 MB each with debuginfo) against the full
//! `ai_gateway` library. That dominated build time and ran the linker out
//! of memory. `Cargo.toml` sets `autotests = false`; every test file is a
//! module of this crate instead.
//!
//! Exceptions (own `[[test]]` targets in `Cargo.toml`): the
//! `github_copilot_*` files point the process-global `AI_GATEWAY_DATA_DIR`
//! at their own secure-storage directories and would race inside a
//! shared process.
//!
//! Adding a test file: declare it below (or register it as a `[[test]]`
//! target). `every_test_file_is_compiled` fails if a file is left out.
//!
//! Filter by file: `cargo test -p ai-gateway --test suite <file_stem>::`.

mod common;

mod admin_budget_reset;
mod admin_codex_test_connection;
mod anthropic_front_door_roundtrip;
mod anthropic_front_door_streaming;
mod anthropic_translation_property;
mod assistants_crud;
mod budget_persistence;
mod budget_rollover;
mod config_budget_policy;
mod context_overflow_routing;
mod cost_reporting;
mod cost_reporting_admin;
mod cost_reporting_query;
mod decisions_front_door;
mod failover_effectiveness;
mod guardrail_custom_http;
mod guardrail_e2e;
mod guardrail_indirect_injection;
mod guardrail_presidio;
mod guardrail_refusal_failover;
mod guardrail_semantic;
mod guardrail_streaming;
mod guardrail_streaming_refusal;
mod guardrail_timing;
mod integration;
mod jev_classifier_integration;
mod jev_performance;
mod laya_classifier_integration;
mod loop_detection_admin;
mod loop_detection_config;
mod loop_detection_enforcement;
mod loop_detection_events;
mod loop_detection_eviction;
mod loop_detection_fingerprint;
mod loop_detection_injection;
mod loop_detection_integration;
mod loop_detection_metrics;
mod loop_detection_middleware;
mod loop_detection_reload;
mod loop_detection_scorer;
mod loop_detection_session;
mod loop_detection_signals;
mod loop_detection_simhash;
mod loop_detection_vk_overrides;
mod memory_enhancement_integration;
mod mock_providers;
mod models_context_metadata;
mod pass_through_routes;
mod performance;
mod prompt_cache_routing;
mod rate_history;
mod reasoning_compat_property;
mod reasoning_failover_compat;
mod request_outcome_logging;
mod responses_endpoints;
mod responses_property_tests;
mod responses_translation;
mod structured_output_integration;
mod structured_output_reload;
mod supply_chain;
mod telemetry_admin;
mod telemetry_export;
mod telemetry_observability;
mod telemetry_propagation;
mod telemetry_sampling;
mod token_compression_cache_property;
mod token_compression_counter_property;
mod token_compression_e2e;
mod token_compression_observability;
mod token_compression_short_circuit_property;
mod tray_integration;
mod virtual_keys_integration;
mod webhooks_admin;
mod webhooks_provider_alerts;
mod webhooks_reload;
mod webhooks_vk_alerts;
mod xss_lint;

/// Guard for `autotests = false`: a new `tests/*.rs` file that is neither
/// declared above nor registered as a `[[test]]` target would otherwise be
/// silently skipped.
#[test]
fn every_test_file_is_compiled() {
    let manifest_dir = std::path::Path::new(env!("CARGO_MANIFEST_DIR"));
    let suite = include_str!("suite.rs");
    let manifest = std::fs::read_to_string(manifest_dir.join("Cargo.toml")).unwrap();
    let mut missing = Vec::new();
    for entry in std::fs::read_dir(manifest_dir.join("tests")).unwrap() {
        let path = entry.unwrap().path();
        if path.extension().and_then(|ext| ext.to_str()) != Some("rs") {
            continue;
        }
        let stem = path.file_stem().unwrap().to_str().unwrap();
        if stem == "suite" {
            continue;
        }
        let declared = suite.contains(&format!("\nmod {stem};"));
        let own_target = manifest.contains(&format!("path = \"tests/{stem}.rs\""));
        if !declared && !own_target {
            missing.push(stem.to_owned());
        }
    }
    assert!(
        missing.is_empty(),
        "test files not compiled by any target (add `mod <name>;` to tests/suite.rs \
         or a [[test]] entry to Cargo.toml): {missing:?}"
    );
}
