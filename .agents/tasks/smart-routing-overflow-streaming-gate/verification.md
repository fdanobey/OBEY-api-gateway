# Verification — iteration 1

All commands run from `c:\Fbn_Dev_Apps\OBEY API Router` in PowerShell (Windows 11).

| Command | Result |
|---|---|
| `cargo check -p ai-gateway --all-targets` | `Finished dev profile` — 0 errors, 0 warnings (no `^warning`/`^error` lines) |
| `$env:PROPTEST_CASES="64"; cargo test -p ai-gateway -- smart_routing_ streaming_requires_buffering plan_overflow streaming_reliability_enabled` (lib) | 33 passed, 0 failed |
| `$env:PROPTEST_CASES="64"; cargo test -p ai-gateway -j 4 smart_routing` | 51 test binaries, 515 passed, 0 failed (lib 257 + bin 257 + 1 integration) |
| `$env:PROPTEST_CASES="64"; cargo test -p ai-gateway -j 4 --no-fail-fast` (full suite, includes all `router::` and `gateway::handlers` tests) | 52 result lines, 4950 passed, 0 failed, 15 ignored (existing `#[ignore]` latency tests + 1 doc-test); lib 2333 passed, bin 2336 passed |

Note: the first unthrottled `cargo test` attempt failed to link with OS error 1455 ("paging file is too small"), which is host memory pressure, not a code failure; rerunning with `-j 2`/`-j 4` succeeded. No pre-existing failures were observed.

New tests (all pass):
- `smart_routing::tests::plan_overflow_lists_context_safe_non_candidates`
- `smart_routing::tests::plan_overflow_is_empty_when_bypassed`
- `config::streaming_config_tests::streaming_reliability_enabled_reflects_early_event_and_keepalive`
- `router::router::tests::streaming_requires_buffering_gate_table`
- `router::router::tests::smart_routing_untiered_models_only_overflow_after_tier_candidates` (c)
- `router::router::tests::smart_routing_overflow_excludes_context_unsafe_models` (b)
- `router::router::tests::smart_routing_overflow_serves_request_when_tier_candidates_unavailable` (a: cooldown + open CB → overflow served in priority order, one upstream request)
- `router::router::tests::smart_routing_streaming_passthrough_when_reliability_enabled`
- `router::router::tests::smart_routing_streaming_buffers_when_reliability_disabled`
- `router::router::tests::smart_routing_streaming_passthrough_reaches_overflow_candidate`

Temp logs (`target\sr-test.log`, `target\full-test.log`) deleted. Nothing committed.
