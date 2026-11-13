# Implementation Report — Per-attempt failure logging (ai-gateway)

Status: implemented, verified, approved by review. Changes left uncommitted in the working tree (no commit, no push).

## What changed

Failed upstream provider attempts (timeouts, 504/5xx, circuit-breaker/rate-limit skips) that
were previously discarded whenever a retry or fallback ultimately succeeded are now surfaced in
the SQLite request log and in a model-labeled Prometheus counter. A model such as `glm-5.3:dev`
returning 504s is now visible even when the end-user response came back `200` via failover.

### Files changed (one line each)

- `crates/ai-gateway/src/error/mod.rs` — added `Deserialize` to `ProviderAttempt` so the handler can read the attempts back off `response.extra`; additive, backward-compatible.
- `crates/ai-gateway/src/metrics/provider_attempt.rs` — new module: `obey_api_provider_attempt_failures_total{provider,model,status_code}` counter with bounded labels, deterministic ordering, cardinality cap, and 4 unit tests.
- `crates/ai-gateway/src/metrics/mod.rs` — added the counter field + init, `record_provider_attempt_failure(provider, model, status_code)`, and `write_provider_attempt_failures_prometheus()`.
- `crates/ai-gateway/src/router/router.rs` — in `route_with_failover_for_group`, attaches the collected failed `attempts` to `response.extra["gateway_failed_attempts"]` before each of the three success returns (main success, longest-partial, reasoning-only); `longest` binding made `mut`. No return-type or call-site changes.
- `crates/ai-gateway/src/gateway/handlers.rs` — added `ATTEMPT_LOG_PATH` const, `extract_failed_attempts`, `log_failed_attempts`/`log_failed_attempts_from_response`; wired into buffered + streaming success paths; strips `gateway_failed_attempts` in `strip_gateway_response_metadata` so it never reaches the client; appended the Prometheus writer to the `/metrics` exposition.
- `crates/ai-gateway/src/logger/mod.rs` — new focused test `attempt_failure_row_is_queryable_and_distinct_from_outcome_row` proving attempt rows are queryable by `model` + `status_code` and distinct from (yet correlated by `trace_id` with) the outcome row.

## How to query the new attempt rows

Each failed attempt is written to the existing `requests` table as its own row, sharing the
`trace_id` of the final-outcome row. Query via the existing `/logs` endpoint by model +
status code — no new filter is required:

```
GET http://192.168.1.110:8080/logs?model=glm-5.3:dev&status_code=504
```

This returns the per-attempt 504 rows for `glm-5.3:dev`, including attempts that were masked by
a subsequent successful retry/fallback. The row's `response_body` holds the provider error
string; `provider`, `model`, `status_code`, and `timestamp` identify the attempt. The matching
final-outcome row for the same request is retrievable by its shared `trace_id`.

Attempt rows are content-free: only `provider`, `model`, `status_code`, the error string (in
`response_body`), `timestamp`, and the shared `trace_id`. No request/response bodies, no
secrets or API keys. `duration_ms = 0` because `ProviderAttempt` does not track per-attempt
latency.

## Schema / migration note

**No schema migration was introduced** (design decision D1). Attempt rows reuse the existing
`requests` table and are distinguished from the final-outcome row by a dedicated `path` marker:

- Attempt row: `path = "/v1/chat/completions#attempt"`
- Final-outcome row: `path = "/v1/chat/completions"` (unchanged)

The existing `LogEntry` / `LogFilter` SELECT already supports filtering by `model` and
`status_code`, so the required query works with zero query-layer changes. Because no column was
added and no existing row/semantic was altered, this is strictly additive and fully
backward-compatible: existing rows, the `/logs` endpoint, the dashboard, and all prior tests are
untouched. (The marker is a `path` value rather than a new column; a reviewer wanting a
dedicated column could add one as an additive follow-up, but it is not required.)

## Model-labeled Prometheus counter

**Implemented** (not deferred). The counter
`obey_api_provider_attempt_failures_total{provider,model,status_code}` lives in the new
`crates/ai-gateway/src/metrics/provider_attempt.rs` module (bounded labels, cardinality cap,
best-effort), is recorded per attempt via `Metrics::record_provider_attempt_failure`, and is
exposed through the `/metrics` endpoint alongside the other Prometheus writers.

## Verification commands and results (Windows 11 / PowerShell)

- `cargo check -p ai-gateway --all-targets` → **EXIT=0, 0 warnings, 0 errors** (clean-build gate satisfied).
- `cargo build --release -p ai-gateway` → **EXIT=0, 0 warnings** (release binary builds).
- `cargo test -p ai-gateway`:
  - lib unit tests (`--lib`): **2365 passed; 0 failed; 8 ignored** (wall-clock `#[ignore]` budget tests).
  - integration test binaries (all 45 under `crates/ai-gateway/tests/`): **all passed; 0 failed**, run in batches to fit the per-call timeout. Highlights: `integration` 35 passed, `virtual_keys_integration` 13 passed, `structured_output_integration` 9 passed, `assistants_crud` 9 passed, the full `loop_detection_*`, `guardrail_*`, `responses_*`, and `token_compression_*` suites all green.
  - The core requirement is proven by `logger::tests::attempt_failure_row_is_queryable_and_distinct_from_outcome_row` and the 4 `metrics::provider_attempt::tests::*` counter tests, all passing.

The full `cargo test -p ai-gateway` exceeds a single terminal-call timeout window (compiling
~2365 lib tests plus 45 integration binaries), so the run was split: all test binaries were
compiled once (`cargo test --no-run`, EXIT=0) and then every lib and integration binary was
executed to completion with zero failures.
