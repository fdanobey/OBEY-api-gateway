# Verification log: GLM / Electron Hub timeout remediation

## Iteration 1: integration fixer (cross-FEAT seams)

Environment: Windows 11, PowerShell, main working tree. Nothing committed or pushed. The live instance was not contacted.

### Seam fixes (crates/ai-gateway/src/router/router.rs)

1. Pass-through non-success tries were not all recorded. FEAT-002 left the streaming path unrecorded, and FEAT-003 (DD11) recorded only TTFB, send-error and 5xx. A pass-through 429, a context-length 4xx, an image-rejection 4xx or any other 4xx left no `#attempt` row, even though AGENTS.md says "every failed try is logged". Fix: `route_request_streaming_excluding` now records every non-success pass-through response once, as `upstream_http_<code>`, right after the body is drained and before any fallback. The 5xx arm reuses that error instead of recording a second time. Routing, fallback and client responses are unchanged. New test: `streaming_passthrough_rate_limit_is_recorded_in_ledger`.
2. DD11 combined with DD12 double-recorded a failure. After a pass-through TTFB, send or 5xx failure opened the breaker, `record_breaker_skips` (which runs on the post-exclusion list) also logged a `circuit_open_skip` row for the provider that the caller had excluded, on top of its `ttfb_timeout` row. Fix: `record_breaker_skips` now takes `exclude` and skips excluded keys, which matches DD12 ("dropped because the breaker is open"). `streaming_ttfb_timeout_fallback_excludes_failed_provider` now asserts exactly one ledger row for provider `a`.
3. ADR 0001: one sentence each added to the DD11 and DD12 entries to describe the two fixes above.

### Commands and results

| Command | Result |
|---|---|
| `cargo check -p ai-gateway --all-targets` (before fixes) | exit 0, 0 warnings, 0 errors |
| `cargo test -p ai-gateway --no-run` (before fixes) | exit 0, 0 warnings |
| `cargo test -p ai-gateway --lib` (baseline, before fixes) | 2393 passed, 0 failed (1203.6 s) |
| `cargo check -p ai-gateway --all-targets` (after fixes) | exit 0, 0 warnings, 0 errors |
| `cargo test -p ai-gateway --lib streaming_` | 48 passed, 0 failed |
| `cargo test -p ai-gateway --lib` (after fixes) | 2394 passed, 0 failed (1202.0 s) |
| `cargo test -p ai-gateway --test <stem>` for all 50 binaries (3 batches) | all exit 0, 0 failed (counts below) |
| `cargo test -p ai-gateway -- --list` | exit 0. All four AGENTS.md guard tests are listed, plus the new test |
| `cargo check -p ai-gateway --all-targets` (final) | exit 0, 0 warnings, 0 errors |

Integration binaries (passed / ignored):
- admin_codex_test_connection 4
- assistants_crud 9
- guardrail_custom_http 4
- guardrail_e2e 8
- guardrail_indirect_injection 13 / 1
- guardrail_presidio 7
- guardrail_refusal_failover 5
- guardrail_semantic 3 / 1
- guardrail_streaming 5
- guardrail_streaming_refusal 4
- guardrail_timing 0 / 3
- integration 35
- jev_classifier_integration 8
- jev_performance 3 / 1
- laya_classifier_integration 11
- loop_detection_admin 1
- loop_detection_config 5
- loop_detection_enforcement 6
- loop_detection_events 2
- loop_detection_eviction 3
- loop_detection_fingerprint 5
- loop_detection_injection 4
- loop_detection_integration 6
- loop_detection_metrics 2
- loop_detection_middleware 3
- loop_detection_reload 2
- loop_detection_scorer 4
- loop_detection_session 5
- loop_detection_signals 6
- loop_detection_simhash 6
- loop_detection_vk_overrides 3
- mock_providers 4
- pass_through_routes 1
- performance 7 / 8
- prompt_cache_routing 7
- reasoning_compat_property 12
- reasoning_failover_compat 12
- request_outcome_logging 4
- responses_endpoints 7
- responses_property_tests 8
- responses_translation 7
- structured_output_integration 9
- structured_output_reload 2
- token_compression_cache_property 1
- token_compression_counter_property 1
- token_compression_e2e 4
- token_compression_observability 4
- token_compression_short_circuit_property 1
- tray_integration 0
- virtual_keys_integration 13

The wall-clock `--ignored` tests were not run.

### Static checks

- `gateway_failed_attempts` in crates/: the channel is gone. Only two negative assertions remain, in router.rs tests and tests/request_outcome_logging.rs, and both assert that the key is absent. The ADR also mentions it historically.
- AllProvidersFailed client JSON: the handlers.rs IntoResponse mapping is unchanged (the only addition is the ChunkTimeout arm). `ProviderAttempt.duration_ms` and `error_class` are `skip_serializing`. `error::tests::aggregated_error_json_has_no_internal_attempt_fields` pins the attempt keys to exactly error/model/provider/status_code/timestamp, and `all_providers_failed_logs_attempt_rows` asserts the client body has no `duration_ms`/`error_class`. The failover skip pushes keep the same messages and status codes as before, so only the builder metadata was added.
- Guard tests listed by `--list`: buffered_sse_reader_allows_generation_longer_than_ttfb_when_bytes_flow, buffered_dispatch_requests_upstream_stream_and_reassembles_sse and timeout_class_errors_fail_over_without_same_provider_retry (in the lib and bin targets), and abandoned_stream_writes_client_disconnect_outcome_row (request_outcome_logging).
- `git status`: only intended files are modified. The untracked files are .agents/, tests/request_outcome_logging.rs, docs/adr/, and wiki-local/, which already existed (dated 2026-08/09) and was not created by this work. Temp test logs under target\ were deleted.
- D1-D4 were not implemented.
