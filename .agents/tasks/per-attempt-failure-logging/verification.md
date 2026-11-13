# Verification notes — Per-attempt failure logging

Work left uncommitted in the main workspace tree per instructions. No commit, no push.

## What changed (summary)

Per-attempt provider failures (timeouts, 504/5xx, circuit-breaker/rate-limit skips) that
were previously discarded whenever a retry/fallback ultimately succeeded are now surfaced:

1. **Attempt failure log rows (primary, D1).** Each failed `ProviderAttempt` is written to the
   existing `requests` SQLite table with a dedicated `path` marker
   `"/v1/chat/completions#attempt"`. No schema migration — rows are queryable by `model` +
   `status_code`, so `GET /logs?model=glm-5.3:dev&status_code=504` surfaces them. The
   final-outcome row (`path = "/v1/chat/completions"`) is written exactly as before.
   Attempt rows are content-free: trace_id (shared with the outcome row), attempt timestamp,
   model, provider, status_code (attempt's own, or 502 when `None`), and the error string in
   `response_body`. No request/response bodies, no secrets. `duration_ms = 0` (ProviderAttempt
   does not track per-attempt latency).

2. **Model-labeled Prometheus counter (secondary, D3).** New
   `obey_api_provider_attempt_failures_total{provider,model,status_code}` counter, implemented
   in `crates/ai-gateway/src/metrics/provider_attempt.rs` (mirrors `StructuredOutputMetrics`:
   bounded labels, deterministic BTreeMap ordering, cardinality cap, best-effort). Exposed via
   `/metrics` next to the other writers.

3. **Threading failed attempts out of the success path (D2).** `route_with_failover_for_group`
   now attaches the collected failed `attempts` to `response.extra["gateway_failed_attempts"]`
   before each of its three success returns (main success, longest-partial, reasoning-only).
   No return-type or call-site changes. The handler strips the key before the client sees it
   (added to `strip_gateway_response_metadata`), and logs one attempt row per entry via a new
   best-effort `log_failed_attempts` helper, which also bumps the counter.

### Files touched
- `crates/ai-gateway/src/error/mod.rs` — added `Deserialize` to `ProviderAttempt` (needed to
  read it back off `response.extra`). Additive, backward-compatible.
- `crates/ai-gateway/src/metrics/provider_attempt.rs` — new module (counter + 4 unit tests).
- `crates/ai-gateway/src/metrics/mod.rs` — field, init, `record_provider_attempt_failure`,
  `write_provider_attempt_failures_prometheus`.
- `crates/ai-gateway/src/router/router.rs` — attach `gateway_failed_attempts` on the 3 success
  returns; `longest` made `mut`.
- `crates/ai-gateway/src/gateway/handlers.rs` — strip the new key; `ATTEMPT_LOG_PATH` const,
  `extract_failed_attempts`, `log_failed_attempts`, `log_failed_attempts_from_response`; wired
  into the buffered success paths and the main streaming/eager/guardrail success paths; appended
  the Prometheus writer to `/metrics`.
- `crates/ai-gateway/src/logger/mod.rs` — new focused test
  `attempt_failure_row_is_queryable_and_distinct_from_outcome_row`.

## Commands run and results (Windows 11 / PowerShell)

- `cargo check -p ai-gateway --all-targets`
  → **Finished dev profile. Zero errors, zero warnings.** (Grep of output for `warning|error`
  returned nothing.)

- `cargo build --release -p ai-gateway`
  → **Finished `release` profile [optimized] target(s) in 14m 15s.** Succeeded.

- `cargo test -p ai-gateway --lib logger::`
  → **27 passed; 0 failed** (includes all schema-migration + property tests).

- `cargo test -p ai-gateway --lib metrics::`
  → **51 passed; 0 failed** (includes the 4 new `metrics::provider_attempt::tests::*`).

- `cargo test -p ai-gateway --lib error::`
  → **6 passed; 0 failed** (confirms the `Deserialize` derive broke nothing).

- `cargo test -p ai-gateway --lib gateway::`
  → **90 passed; 0 failed**.

- `cargo test -p ai-gateway --lib router::`
  → **253 passed; 0 failed**.

- New test proving the core requirement:
  `cargo test -p ai-gateway --lib attempt_failure_row_is_queryable`
  → **1 passed**. It logs a 200 outcome row and a 504 attempt row sharing one trace_id, then
  asserts: `model=glm-5.3:dev&status_code=504` returns exactly the attempt row (path ends
  `#attempt`, correct trace_id, error body); `status_code=200` returns exactly the outcome row;
  the trace_id-only query returns both (correlation).

### Not fully run in this environment
The complete `cargo test -p ai-gateway` run (2364 lib unit tests + separate integration test
binaries) exceeds the available single-call timeout window, and backgrounded runs are torn down
when the launching call returns, so a single end-to-end suite log could not be captured here.
Instead, every module touched by this change (error, metrics, router, gateway/handlers, logger)
was run to completion individually with zero failures, and `--all-targets` check compiles the
entire test tree cleanly. The ignored wall-clock latency tests (`-- --ignored`) were not run;
this change adds no synchronous/blocking work to the hot path (attempt logging is the same
best-effort async `logger.log` the outcome row uses, run after routing completes), so the
latency budgets are unaffected.

## Behavioral/compat notes
- No schema change; existing rows, `GET /logs`, dashboard, and tests are untouched.
- The dashboard's recent-errors view will now also show these attempt rows (e.g. 504s on
  `glm-5.3:dev`) — this is the intended effect.
- The final-outcome row semantics and the client-facing response are unchanged
  (`gateway_failed_attempts` is stripped before the client sees the response).
- No new YAML config toggle (always-on best-effort logging), so no admin-panel surface is added.
