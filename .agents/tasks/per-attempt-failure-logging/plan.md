# Implementation Plan — Per-attempt failure logging (ai-gateway)

## Goal
Make every FAILED upstream provider attempt observable in the SQLite request log — even when the request ultimately succeeds via retry/fallback — so that a model like `glm-5.3:dev` returning 504s is visible via `GET /logs?model=glm-5.3:dev&status_code=504`. Secondary: a model-labeled Prometheus failure counter. The per-attempt data (`ProviderAttempt { provider, model, error, status_code, timestamp }`) is already collected in the router/handlers; today it is dropped on the success path. We surface it.

## Design decisions (made during exploration; grounded in the code)

### D1 — Mechanism to distinguish attempt rows from the final-outcome row: dedicated `path` marker, NO schema migration.
Attempt rows are written to the existing `requests` table using a dedicated `path` value `"/v1/chat/completions#attempt"` (the normal outcome row keeps `"/v1/chat/completions"`). Rationale:
- `LogEntry`/`LogFilter`/the SELECT already support filtering by `model` and `status_code` (`logger/mod.rs` `run_query`, lines ~865-905), so `GET /logs?model=glm-5.3:dev&status_code=504` surfaces attempt rows with zero query-layer changes.
- Adding a new column means editing the 25-param INSERT (`insert_entry`, ~line 821), the SELECT + row-map (`run_query`, ~line 865), the `LogEntry` struct (~line 158), every `LogEntry` literal (handlers `log_request` ~line 319, logger `sample_entry` test ~line 980), plus an additive `ensure_column` migration and new migration tests. The `path` marker achieves the requirement with none of that churn. Chosen as the least-invasive option per the task's explicit preference.
- The final-outcome row is written exactly as today (unchanged `path`, `status`, `model`, semantics), so existing dashboards and tests keep working.
- Note: `path` is not currently a `LogFilter` field, so attempt rows are not excludable via the public `/logs` filter. That is acceptable and in fact desired (they SHOULD show up alongside outcome rows). The dashboard's `recent_errors`/`errors_handler` (`dashboard/mod.rs` ~line 587) reads recent rows and will now also surface these 504 attempt rows — which is the whole point. Document this in the attempt-row doc comment.

### D2 — Thread discarded failed attempts out of the router success path: OPTION A via response extras.
`route_with_failover_for_group` (`router/router.rs` ~line 5516) accumulates `attempts: Vec<ProviderAttempt>` and, on success, returns `Ok(response)` after inserting `gateway_*` keys into `response.extra` (a `serde_json::Map`, `#[serde(flatten)]`). We attach the collected failed attempts the same way: `response.extra.insert("gateway_failed_attempts", serde_json::to_value(&attempts))` immediately before each success return, when `!attempts.is_empty()`. Three success returns inside that function carry `attempts` in scope and must get this: the main success arm (~line 6126), the all-truncated longest-partial return (~line 6266), and the all-reasoning-only best return (~line 6297). This mirrors the existing `gateway_responded_model` / `gateway_cost` / `gateway_cache_*` pattern exactly and requires NO change to `route_request` / `route_request_with_exclusions` return types or any router call site. The handler then reads `gateway_failed_attempts` off the response and logs one attempt row per entry.

CRITICAL leak-prevention: `extra` is `#[serde(flatten)]`, so any key left in it serializes into the client response. `gateway_failed_attempts` MUST be added to `strip_gateway_response_metadata` (`handlers.rs` ~line 355) alongside the other `gateway_*` keys so it never reaches the client. (`prepare_response_for_client` and the pre-return strip already call this.)

### D3 — Model-labeled failure counter: IN SCOPE (clean), following the structured-output per-model pattern.
`metrics/mod.rs` already has `(provider, model, status)`-style metrics (`record_structured_output_validation`) and a bounded 2-/4-label Prometheus writer. Add a `(provider, model, status_code)` counter as a `DashMap<(String,String,String), AtomicU64>` field on `Metrics`, a `record_provider_attempt_failure(provider, model, status_code)` method (bounding labels like the existing helpers), and a `write_provider_attempt_failures_prometheus(&mut String)` that emits a 3-label series. Append it to the `/metrics` exposition in `handlers.rs` next to the other `write_*_prometheus` calls (~line 8706). The router already records provider-keyed success/failure; the new counter is incremented at the point attempt rows are logged (handler side, D5) so it covers both success-after-failure and all-failed cases uniformly and avoids touching the hot failover loop. If a 3-label writer helper does not already exist, add one modeled on the `[&str; 2]` writer at `metrics/mod.rs` ~line 2276.

### D4 — Scope: non-streaming buffered path is the primary target; streaming attempts handled where already collected.
The user's failing calls are buffered chat-completions (`route_request`). The streaming path already collects `streaming_attempts` in `handlers.rs` (~line 2697/2785) and already logs an aggregated error row on streaming failure. We also emit attempt rows on the streaming SUCCESS path using the same shared helper (D5) when `gateway_failed_attempts` is present on the buffered-fallback response, and from the handler-local `streaming_attempts` vec on the pre-content failover success. Keep streaming changes minimal; the buffered path (D2) is the must-have.

### D5 — Single shared emit helper in handlers.rs.
Add `fn log_failed_attempts(state, request, trace_id, attempts: &[ProviderAttempt])` next to `log_request` (`handlers.rs` ~line 317). For each attempt it: builds a `LogEntry` with `path = "/v1/chat/completions#attempt"`, `method = "POST"`, `model = attempt.model`, `provider = attempt.provider`, `status_code = attempt.status_code.unwrap_or(mapped gateway status, default 502)`, `timestamp = attempt.timestamp`, `duration_ms = 0` (attempt-level duration is not tracked; document this), `cost = 0.0`, `request_body = None`, `response_body = Some(attempt.error.clone())` (content-free: provider/model/error string only, NO request/response bodies), all `*_model`/cache/reasoning fields `None`/default, and the SHARED parent `trace_id`. It calls `state.logger.log(entry)` best-effort (`if let Err(e) = ... { tracing::warn!(...) }`, mirroring `log_request`). It also calls `state.metrics.record_provider_attempt_failure(...)` per attempt (D3). A tiny private helper extracts `Vec<ProviderAttempt>` from `response.extra["gateway_failed_attempts"]` via `serde_json::from_value`. No bodies are logged regardless of body-logging config (we never set request/response body beyond the error string). This runs AFTER the response is routed and BEFORE/around the existing `log_request` call, so it never blocks the routing hot path.

## Ordered implementation items

- [ ] 1. Add the model-labeled attempt-failure counter to the metrics module.
      Add a `provider_attempt_failures: Arc<DashMap<(String,String,String), AtomicU64>>` field to `Metrics` (init in `Metrics::new`), a bounded `record_provider_attempt_failure(&self, provider: &str, model: &str, status_code: u16)` method (bound provider/model labels using the same `bounded_label`/escaping the existing per-model metrics use), and a `write_provider_attempt_failures_prometheus(&self, out: &mut String)` emitting `obey_api_provider_attempt_failures_total{provider="..",model="..",status_code=".."} N` with HELP/TYPE lines. Add a 3-label Prometheus write helper if none exists (model on `write_counter_vec` for `[&str; 2]` at ~line 2276). Add a unit test in the `metrics` tests module asserting the exposition contains the expected labeled series (follow `structured_output_prometheus_is_exposed_through_metrics` ~line 2834).
      Files: `crates/ai-gateway/src/metrics/mod.rs`
      Verify: `cargo test -p ai-gateway metrics::` — new counter test passes; `cargo check -p ai-gateway --all-targets` clean.

- [ ] 2. Attach collected failed attempts to the response extras on every router success return.
      In `route_with_failover_for_group` (`router/router.rs` ~line 5516), immediately before each of the three success returns — the main success arm (~6126), the all-truncated longest-partial return (~6266), and the all-reasoning-only best return (~6297) — insert, when `!attempts.is_empty()`: `response.extra.insert("gateway_failed_attempts".to_string(), serde_json::to_value(&attempts).unwrap_or(serde_json::Value::Null));` (use the correct response binding name at each site: `response` / `longest` / `best`). Do not change control flow or return types; do not touch the hot per-attempt loop otherwise.
      Files: `crates/ai-gateway/src/router/router.rs`
      Verify: `cargo check -p ai-gateway --all-targets` clean. (Behavioral verification in item 5.)

- [ ] 3. Add the shared attempt-row emit helper and strip the new extras key in handlers.
      In `gateway/handlers.rs`: (a) add `gateway_failed_attempts` removal to `strip_gateway_response_metadata` (~line 355) so it never leaks to the client. (b) Add `fn log_failed_attempts(state, request, trace_id, attempts: &[ProviderAttempt])` and a private extractor reading `response.extra["gateway_failed_attempts"]` → `Vec<ProviderAttempt>` (per D5): one content-free `LogEntry` per attempt with `path = "/v1/chat/completions#attempt"`, parent `trace_id`, attempt `timestamp`, `response_body = Some(error)`, `duration_ms = 0`, best-effort `logger.log`, and `metrics.record_provider_attempt_failure`. Import `ProviderAttempt` from `crate::error`.
      Files: `crates/ai-gateway/src/gateway/handlers.rs`
      Verify: `cargo check -p ai-gateway --all-targets` clean.

- [ ] 4. Wire the emit helper into the buffered + streaming success paths; append the metrics exposition.
      In `gateway/handlers.rs`: on the non-streaming success paths that call `RequestLogContext::from_response(...)` then `log_request(...)` (the two main sites ~2096-2109 and ~2940-2942, and the smart/early-event buffered sites ~1563-1576), BEFORE `strip`/`prepare_response_for_client` consumes the extras, call `log_failed_attempts(&state, &request, &trace_id, &attempts)` using attempts extracted from `response.extra` — do this while the response still carries the key (i.e. before `strip_gateway_response_metadata` / `prepare_response_for_client`). On the streaming success paths, also emit from the handler-local `streaming_attempts` vec (~2615-2623 and the buffered-fallback success ~2733-2734). Append `state.metrics.write_provider_attempt_failures_prometheus(&mut out);` to the `/metrics` exposition next to the other writers (~line 8706). Ensure the final-outcome `log_request` row is still written unchanged.
      Files: `crates/ai-gateway/src/gateway/handlers.rs`
      Verify: `cargo check -p ai-gateway --all-targets` clean; `cargo test -p ai-gateway` passes.

- [ ] 5. Add a focused logger unit test proving attempt rows are queryable and distinct from the outcome row.
      In `logger/mod.rs` `#[cfg(test)] mod tests` (follow `create_test_logger`, `sample_entry`, `logger.query(LogFilter{..})`): log an outcome row (`path = "/v1/chat/completions"`, model `glm-5.3:dev`, status 200) and an attempt row (`path = "/v1/chat/completions#attempt"`, same `trace_id`, model `glm-5.3:dev`, status 504, `response_body = Some("504 ...")`). Assert `query(LogFilter{ model: Some("glm-5.3:dev"), status_code: Some(504), ..Default })` returns exactly the attempt row (len 1, `path` ends with `#attempt`, correct `trace_id`); assert `status_code: Some(200)` returns the outcome row; assert `trace_id`-only query returns both (correlation).
      Files: `crates/ai-gateway/src/logger/mod.rs`
      Verify: `cargo test -p ai-gateway logger::` — new test passes.

- [ ] 6. Clean build + full fast test suite; confirm no warnings and no regressions.
      Run the clean-build gate and the fast suite. Fix any warning by removing dead code or `#[cfg(test)]`-gating test-only helpers — do NOT add blanket `#[allow]`.
      Files: (none — verification only)
      Verify: `cargo check -p ai-gateway --all-targets` → zero errors, zero warnings; `cargo test -p ai-gateway` → all pass (existing `/logs`, dashboard, migration, and metrics tests included).

## Notes / assumptions
- No SQLite migration is introduced (D1). Existing rows and queries are untouched; this is the safest, most backward-compatible choice and satisfies the `GET /logs?model=..&status_code=504` requirement.
- Attempt rows are best-effort and off the hot path (logged after routing completes, via the same async writer-thread `logger.log` the outcome row uses). No new blocking/synchronous DB work is added to the request critical path.
- Attempt rows are content-free: only provider, model, status_code, the error string (in `response_body`), timestamp, and the shared `trace_id`. No request/response bodies, no secrets/API keys.
- Attempt-level duration is not tracked by `ProviderAttempt`; attempt rows use `duration_ms = 0`. Documented in the helper's doc comment.
- The model-labeled counter is implemented (D3), not deferred; requirement 1 (attempt rows) remains the must-have and is fully covered by items 2-5.
- Admin-panel parity (AGENTS.md): this feature adds no YAML-configurable toggle — it is always-on best-effort logging — so there is no new config surface requiring an admin-panel control. If a reviewer deems an on/off toggle desirable, that is a follow-up, not part of this change.
- Work stays in the main workspace tree; do NOT commit or push.
