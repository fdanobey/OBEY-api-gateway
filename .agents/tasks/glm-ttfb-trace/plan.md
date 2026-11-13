# Implementation Plan — GLM / Electron Hub timeout remediation (ai-gateway)

Source of truth for *why*: `investigation.md` (same folder). This plan implements fixes F1-F9.
D1-D4 (Codex Search streaming bypass, loop-detection behavior, client timeout, codex_search_ready
doc mismatch) are **pending user decisions and must NOT be implemented.**

Repo: `c:\Fbn_Dev_Apps\OBEY API Router` (main working tree, no worktree). Windows 11 / PowerShell.
Always pass absolute paths. Do **not** commit or push: the user approves before anything is
committed or deployed. Do not touch the live instance.

Build/test commands (from AGENTS.md):
- Clean-build gate: `cargo check -p ai-gateway --all-targets` → zero errors, **zero warnings**
  (no blanket `#[allow]`; gate test-only helpers with `#[cfg(test)]`).
- Unit tests: `cargo test -p ai-gateway --lib <filter>`.
- Integration tests: `cargo test -p ai-gateway --test <file_stem>`.
- Full suite: `cargo test -p ai-gateway`. It exceeds one terminal timeout, so run `--lib`, then
  the integration binaries in batches (`cargo test -p ai-gateway --no-run` first). Use 600-1800 s
  timeouts for cargo commands.

## Design decisions (made here; do not re-decide)

- **DD1 Upstream transport.** The buffered HTTP dispatch (`dispatch_attempts_under_permit`,
  `router.rs` ~4029-5160) sends `stream: true` and reassembles internally. It is controlled by a
  new per-provider `buffered_upstream_streaming: Option<bool>` (serde default `None`).
  `Provider::effective_buffered_upstream_streaming()` = `unwrap_or(provider_type != "bedrock")`.
  Per-provider (not global) so a single misbehaving provider can opt out. Default on so the live
  fix needs no config edit. Bedrock and Codex keep their own dispatch and are untouched.
- **DD2 Stay on stream:false** for a request with `n > 1`, `logprobs`/`top_logprobs`, or `audio`/
  `modalities` containing audio. Reassembly only rebuilds `choices[0]` text/tool calls/reasoning.
- **DD3 Timeouts when streaming upstream.**
  - The first-byte deadline (`ttfb_timeout`) covers headers **and** the first body chunk, of any
    content (SSE comments count as liveness).
  - The idle gap between chunks is `streaming.chunk_timeout_seconds` (`StreamingConfig` default
    when the section is absent) → new `GatewayError::ChunkTimeout(secs)` → HTTP 504.
  - The overall cap is `total_timeout`.
  - With `stream:false` (opt-out or DD2), today's semantics are kept **exactly**: TTFB on headers,
    then `response.text()` bounded by the remaining total.
- **DD4 `stream_options`.** Insert `stream_options: {"include_usage": true}` into `outgoing.extra`
  only when it is absent. If the provider answers 400 and the body mentions `stream_options` or
  `stream`, retry the same provider **once** with `stream:false` and no gateway-inserted
  `stream_options` (one-shot flag, skip backoff, mirror `image_strip_retry_done`). Log a warn that
  recommends `buffered_upstream_streaming: false`.
- **DD5 Parse chain unchanged in order.** JSON first (a provider may ignore `stream:true`), then
  SSE.
  - SSE detection is broadened from `starts_with("data: ")` to "after skipping blank, `:`-comment
    and `event:`/`id:`/`retry:` lines, the first line starts with `data:`".
  - A mid-stream `error` frame follows the existing error-in-200 rules: rate-limit-shaped
    (`is_rate_limited(200, body)`) → 429 cooldown path and fail over; otherwise retryable provider
    error.
  - A stream with neither `[DONE]` nor any `finish_reason` → retryable provider error
    "Upstream stream ended before completion".
- **DD6 Reassembly keeps the full `usage`.** Deserialize the last `usage` object into `Usage`
  (keeps `extra` such as `prompt_tokens_details`, `completion_tokens_details`) and keep the
  token-estimate fallback only when no usage frame arrived.
- **DD7 Attempt ledger replaces the `gateway_failed_attempts` extras channel.**
  - `ActiveRequestHandle` gains a ledger (`record_failed_attempt`, `failed_attempt_count`,
    `take_failed_attempts`) and an `outcome_logged` flag. Neither is serialized to the dashboard
    (`#[serde(skip)]`).
  - `ProviderAttempt` gains `duration_ms: Option<u64>` and `error_class: Option<String>`, both
    `#[serde(default, skip_serializing)]`, so the client-visible `AllProvidersFailed.attempts`
    JSON (`handlers.rs` ~1197-1206) is unchanged.
  - Remove the router's three `gateway_failed_attempts` insertions (`router.rs` ~6127-6137, ~6280,
    ~6317) and the handler's `extract_failed_attempts`/`log_failed_attempts_from_response`, plus
    its strip line. Update the prior feature's tests to assert through the ledger.
  - Dedupe rule: `dispatch_attempts_under_permit` records **every** failed HTTP try.
    `route_with_failover_for_group` records skips and content failovers. Its `Err` arm records
    only if `failed_attempt_count()` did not grow during that provider's `attempt_with_retry`
    call (covers slot-saturation 503, OAuth 401, Codex/Bedrock errors).
- **DD8 Content-free `error_class`** (≤ 32 chars, stable snake_case). Allowed values:
  - `upstream_http_<code>`, `ttfb_timeout`, `total_timeout`, `chunk_timeout`, `network_error`
  - `body_read_error`, `stream_truncated`, `error_in_200`, `rate_limit_skip`, `cooldown_skip`
  - `circuit_open_skip`, `budget_skip`, `slot_saturated`, `empty_response`, `reasoning_only`
  - `truncated_response`, `client_disconnect`, `gateway_deadline`, `other`

  Stored in a new additive `requests.error_class TEXT` column (via the logger's existing
  `ensure_column` migration pattern), always written regardless of `response_body_logging`.
  Attempt-row `status_code` is the upstream HTTP status when one was received. Router timeouts
  (`ttfb/total/chunk`) use 504, network/body errors use 502, and skips keep their current codes
  (402/429/503).
- **DD9 Abandoned requests.** If the request future drops while the handle's `outcome_logged` is
  false and the guard was not `complete()`d, write one content-free outcome row:
  - path `/v1/chat/completions`, `request_body: None`
  - provider/model from the handle snapshot, `duration_ms` = elapsed
  - status 499 + `error_class: client_disconnect`; or 504 + `gateway_deadline` when elapsed ≥
    `runtime_limits.request_timeout_seconds() - 1`
  - `response_body` = a short gateway message ("request abandoned after Ns while on
    <provider>/<model>, attempt N, phase P"), still subject to `response_body_logging`

  Then drain the ledger into `#attempt` rows. Every existing terminal `log_request` call in the
  chat paths must mark `outcome_logged` (introduce `log_outcome(state, request, ctx, handle)`).
- **DD10 Timeout-class errors fail over without a same-provider retry.** Timeout class:
  `TtfbTimeout`, `TotalTimeout`, `ChunkTimeout`, `Provider{status_code: Some(504|524)}`. In the
  retry loop these `return Err(..)` immediately instead of `continue`. 500/502/network/body-read
  errors keep `max_retries_per_provider`. No config knob, rationale in the ADR.
- **DD11 Pass-through fallbacks exclude the failed entry.** Applies to TTFB timeout, send error
  and 5xx in `route_request_streaming_excluding` (`router.rs` ~9000-9018, ~9156-9162):
  - record a ledger attempt (`ttfb_timeout` / `network_error` / `upstream_http_<code>`)
  - `cb.record_failure()`
  - fall back with `route_request_with_exclusions(request, active, &excluded)`, where `excluded`
    is `exclude` + the failed `provider:model`, mirroring the 429 path at ~9050-9073

  4xx (non-429, after the existing context-length/image handling) keeps today's fallback.
- **DD12 Breaker pre-filter visibility.** In `route_request_with_exclusions` (once per request,
  guarded by a handle flag so the streaming→buffered hop does not double-record), record a
  `circuit_open_skip` attempt (status 503) for each group member whose breaker is currently
  Open and that `select_provider_order` therefore removed.
- **DD13 Admin parity (AGENTS.md).** `buffered_upstream_streaming` gets a checkbox in the provider
  card of `crates/ai-gateway/src/admin/static/index.html`:
  - load beside `.prov-ttfb-timeout` (~line 2977): checked when the value is `null`/`true`
  - save beside `ttfb_timeout_seconds` in the providers save literal (~line 4083), writing
    `true`/`false`

---

## FEAT-001 — Buffered dispatch streams upstream (F1, F2, F3)

- [ ] 1. Add the provider option and the ChunkTimeout error.
      Add `buffered_upstream_streaming: Option<bool>` (`#[serde(default)]`, doc comment per DD1) to
      `Provider` in `config/mod.rs` (~400-500, next to `ttfb_timeout_seconds`), plus
      `effective_buffered_upstream_streaming()` and unit tests:
      - default true for `openai`/`nvidia_nim`/`ollama`
      - false for `bedrock`
      - an explicit `false` wins
      - YAML without the field parses

      Add `GatewayError::ChunkTimeout(u64)` ("Provider stopped sending data for {0}s") mapped to
      `GATEWAY_TIMEOUT` in `error/mod.rs`. Handle it in every exhaustive match, including
      `IntoResponse` in `handlers.rs` ~1163 and `classify_stream_error` ~4289, using the same SSE
      error type the relay uses for its inter-chunk timeout. Add `buffered_upstream_streaming: None`
      to every `Provider { .. }` literal; there are ~34 sites (search `ttfb_timeout_seconds:`
      across `src/` and `tests/`).
      Files: `crates/ai-gateway/src/config/mod.rs`, `crates/ai-gateway/src/error/mod.rs`, `crates/ai-gateway/src/gateway/handlers.rs`, all `Provider` literal sites
      Verify: `cargo check -p ai-gateway --all-targets` (0 warnings); `cargo test -p ai-gateway --lib config::` and `--lib error::` pass.

- [ ] 2. Preserve full usage and harden SSE reassembly (F2).
      In `Router::reassemble_sse_response` (`router.rs` ~5169-5415):
      - keep the last `usage` object as a `Usage` with its `extra` (DD6)
      - return `Err("Upstream stream ended before completion")` when the body has no `[DONE]` and
        no chunk carried a `finish_reason` (DD5)

      Add a helper `fn body_looks_like_sse(body: &str) -> bool` per DD5 and use it in place of
      `body_text.starts_with("data: ")` (~4818). Tests in the router test module:
      - `reassemble_sse_response_preserves_usage_details`
      - `buffered_sse_without_done_or_finish_reason_is_truncation_error`
      - `buffered_sse_with_leading_comment_lines_is_reassembled`

      Existing `reassemble_sse_response_*` tests must stay green.
      Files: `crates/ai-gateway/src/router/router.rs`
      Verify: `cargo test -p ai-gateway --lib reassemble` and `--lib buffered_sse` pass; the clean-build gate passes.

- [ ] 3. Incremental upstream body reader.
      Add a private async helper in `router.rs`, for example
      `read_upstream_body(response: reqwest::Response, first_byte_deadline: tokio::time::Instant, idle: Option<Duration>, total_deadline: tokio::time::Instant) -> Result<String, GatewayError>`.
      It reads `response.bytes_stream()` (reqwest `stream` feature is already used by the relay).
      Behavior:
      - no first chunk by `first_byte_deadline` → `TtfbTimeout`
      - a gap longer than `idle` after the first chunk → `ChunkTimeout`
      - past `total_deadline` → `TotalTimeout`
      - a stream error → `Provider{ "Failed to read response body: …", Some(status) }` (caller supplies the status)

      Unit tests build `reqwest::Response` from a delayed stream, following
      `stalling_streaming_response` in `handlers.rs` tests ~6931:
      - `buffered_sse_reader_ttfb_covers_first_body_chunk` (headers immediate, first chunk after the deadline → TtfbTimeout)
      - `buffered_sse_reader_allows_generation_longer_than_ttfb_when_bytes_flow` (chunks every 200 ms for 3 s with first-byte 1 s, idle 1 s → Ok with the full body). **This is the key regression guard.**
      - `buffered_sse_reader_idle_gap_is_chunk_timeout`
      - `buffered_sse_reader_total_timeout`
      Files: `crates/ai-gateway/src/router/router.rs`
      Verify: `cargo test -p ai-gateway --lib buffered_sse_reader` passes.

- [ ] 4. Stream upstream in `dispatch_attempts_under_permit`.
      Replace the unconditional `outgoing.stream = false` (~4262-4270) with:
      `let stream_upstream = provider_cfg.effective_buffered_upstream_streaming() && !requires_non_streaming(&outgoing)` (DD2).
      Set `outgoing.stream = stream_upstream`. When it is true, insert `stream_options.include_usage`
      if absent (DD4) **before** `sanitize_request_for_provider`.
      In the retry loop (~4560-4650):
      - keep `tokio::time::timeout(ttfb_timeout, send())`
      - when `stream_upstream` and the status is success, read the body with the step 3 reader
        (first-byte deadline = request_start + ttfb, idle = streaming chunk timeout, total =
        request_start + total)
      - otherwise keep the current `response.text()` path unchanged
      - non-success statuses still use `response.text()` (error bodies are small JSON)

      Add the DD4 one-shot `stream_options` 400 fallback in the 4xx branch. Make SSE mid-stream
      error frames follow DD5. The `has_tools`/`gateway_tool_hint_injected`/
      `gateway_reasoning_compat_actions` annotations on the SSE success branch (~4826-4843) already
      exist; keep them.
      Router tests (wiremock; follow existing `MockServer` tests, `test_provider`, `create_test_config`):
      - `buffered_dispatch_requests_upstream_stream_and_reassembles_sse`: matcher `body_partial_json({"stream":true,"stream_options":{"include_usage":true}})`. The `text/event-stream` body carries reasoning_content + content + `finish_reason:"stop"` + usage with `prompt_tokens_details.cached_tokens` and `completion_tokens_details.reasoning_tokens`. Client request `stream:false`. Assert content, reasoning carrier, usage extras, and that `gateway_cache_read_tokens`/`gateway_reasoning_tokens` extras are populated.
      - `buffered_dispatch_accepts_json_body_when_provider_ignores_stream_true`
      - `buffered_upstream_streaming_disabled_sends_stream_false` (matcher `{"stream":false}`, plus a check that no `stream_options` was added)
      - `buffered_dispatch_keeps_stream_false_for_multi_choice_or_logprobs`
      - `buffered_dispatch_falls_back_to_stream_false_when_stream_options_rejected` (first response 400 `{"error":{"message":"Unrecognized request argument: stream_options"}}` `up_to_n_times(1)`, then 200 JSON matched on `{"stream":false}`)
      Files: `crates/ai-gateway/src/router/router.rs`
      Verify: `cargo test -p ai-gateway --lib buffered_` passes. Then run the integration tests that mock JSON upstreams and confirm they still pass: `cargo test -p ai-gateway --test integration`, `--test prompt_cache_routing`, `--test reasoning_failover_compat`, `--test structured_output_integration`, `--test guardrail_streaming`, `--test token_compression_e2e`.

- [ ] 5. Admin-panel parity for `buffered_upstream_streaming` (DD13).
      Add the checkbox to the provider card markup, load it (~2977) and save it (~4083) in
      `crates/ai-gateway/src/admin/static/index.html`. Add the field to the `Provider` literals in
      `admin/mod.rs` (~2150, ~2718). Add an admin round-trip test in the `admin/mod.rs` test module
      (follow the existing config GET/POST tests): saving a provider with
      `buffered_upstream_streaming: false` persists, and the field comes back from `GET /admin/config`.
      Files: `crates/ai-gateway/src/admin/static/index.html`, `crates/ai-gateway/src/admin/mod.rs`
      Verify: `cargo test -p ai-gateway --lib admin::` passes. Optional: run the gateway locally with a throwaway config and use chrome-devtools to confirm the checkbox renders checked by default and that saving shows `buffered_upstream_streaming` in `GET /admin/config`. Delete the throwaway config afterwards.

- [ ] 6. FEAT-001 gate.
      Verify: `cargo check -p ai-gateway --all-targets` (0 warnings); `cargo test -p ai-gateway --lib` passes.

## FEAT-002 — Attempt ledger, abandoned-request rows, error_class (F4, F5, F8)

- [ ] 7. Ledger on `ActiveRequestHandle` and extended `ProviderAttempt` (DD7).
      In `active_requests.rs`, add `#[serde(skip)]` fields `failed_attempts: Vec<ProviderAttempt>`,
      `outcome_logged: bool` and `breaker_skips_recorded: bool` to `ActiveRequestInfo`. Initialize
      them in `build_active_request_info` (`handlers.rs` ~1326) and in test samples. Add handle
      methods: `record_failed_attempt`, `failed_attempt_count`, `take_failed_attempts`,
      `mark_outcome_logged`, `outcome_logged`, `claim_breaker_skip_recording`.
      In `error/mod.rs`:
      - add `duration_ms: Option<u64>` and `error_class: Option<String>` to `ProviderAttempt`
        (`#[serde(default, skip_serializing)]`)
      - add builder helpers `with_duration(Duration)` and `with_error_class(&str)`
      - add `fn error_class_for(err: &GatewayError) -> String` implementing DD8
      - add a test asserting `AggregatedError` JSON is unchanged (no new keys)

      Unit test `failed_attempt_ledger_records_and_drains`.
      Files: `crates/ai-gateway/src/active_requests.rs`, `crates/ai-gateway/src/error/mod.rs`, `crates/ai-gateway/src/gateway/handlers.rs`
      Verify: `cargo test -p ai-gateway --lib active_requests::` and `--lib error::` pass.

- [ ] 8. `error_class` column in the request log.
      Add `error_class: Option<String>` to `LogEntry` (`logger/mod.rs` ~158). Add the additive
      column via the existing `ensure_column` migration and wire it into the INSERT (~821) and the
      SELECT/row-map (~865-935). Update every `LogEntry { .. }` literal (search `reasoning_compat_actions:`
      in `src/` and `tests/`). `error_class` is always stored (not gated by body logging).
      Test `error_class_column_migrates_and_round_trips`: a legacy table without the column
      migrates, and the value round-trips through `query`.
      Files: `crates/ai-gateway/src/logger/mod.rs`, `LogEntry` literal sites (handlers.rs, dashboard/mod.rs tests, logger tests)
      Verify: `cargo test -p ai-gateway --lib logger::` passes.

- [ ] 9. Record attempts in the router (DD7 dedupe, DD8, DD12).
      In `dispatch_attempts_under_permit`, record every failed HTTP try into `active` with
      `duration_ms = request_start.elapsed()`, status and `error_class`:
      - TTFB/total/chunk timeout
      - body read error
      - error-in-200
      - retryable status, non-retryable 4xx return, 503 return, 429 return
      - network error
      - stream-truncated
      - the DD4 400 that triggers the stream:false retry (class `upstream_http_400`)

      In `route_with_failover_for_group` (~5516-6250):
      - record budget/CB-open/cooldown/rate-limit skips and reasoning-only/empty/truncated
        failovers into the ledger
      - in the `Err` arm, record only when the ledger did not grow during the call
      - remove the three `gateway_failed_attempts` insertions

      In `route_request_with_exclusions` (~8368), record DD12 breaker skips once per request.
      Router test `ttfb_timeout_attempt_records_duration_and_error_class`: wiremock delay > ttfb,
      handle passed, ledger holds one attempt with class `ttfb_timeout` and `duration_ms` ≥ ttfb.
      Files: `crates/ai-gateway/src/router/router.rs`
      Verify: `cargo test -p ai-gateway --lib router::` passes (update the prior `gateway_failed_attempts` tests to the ledger).

- [ ] 10. Handler: drain the ledger at every terminal point, write abandoned-request rows (DD9), add pre-dispatch timing (F8).
      In `handlers.rs`:
      - refactor `log_failed_attempts` to take `&[ProviderAttempt]` from `active_handle.take_failed_attempts()` and write `duration_ms`/`error_class`
      - delete `extract_failed_attempts`, `log_failed_attempts_from_response` and the `gateway_failed_attempts` strip
      - add `log_outcome(state, request, ctx, handle)` = `log_request` + `mark_outcome_logged`, and use it at every terminal `log_request` in `chat_completions_non_stream`, `chat_completions_stream` (early-event path, failover loop, early-event-disabled path), `stream_buffered_with_post_call` and `stream_eager_structured_output`
      - drain attempts on **error** paths too
      - set `error_class` on error outcome rows via `error_class_for`

      Extend `RequestCompleteGuard` (~1268-1323) with an optional abandon logger holding:
      `Arc` logger, metrics, handle, requested model, kind, and the global deadline seconds from
      `state.runtime_limits`. Its `Drop` writes the DD9 row and drains the ledger when the guard
      is not completed and `outcome_logged` is false.

      In `router.rs`, add `pre_dispatch_ms` (now − `started_at_ms` from the handle) to the
      "Calling provider" info logs (~4285 and the streaming one ~8990). In
      `preprocess_memory_request`, log elapsed at info when > 1 s.
      Integration tests in a new `crates/ai-gateway/tests/request_outcome_logging.rs`, following
      `tests/token_compression_observability.rs` (`server.build_router()`, `/dashboard/logs`) and
      `tests/common::isolate_databases`:
      - `abandoned_stream_writes_client_disconnect_outcome_row`: wiremock upstream delayed 20 s; send `stream:true`; read the first SSE frame; drop the body; poll `/dashboard/logs` ≤ 5 s for status 499 / `error_class` `client_disconnect` with the mock provider name
      - `completed_stream_does_not_write_abandon_row`
      - `same_provider_retry_failure_is_logged_as_attempt_row`: first 500 `up_to_n_times(1)` then 200; one 200 outcome + one `#attempt` row with status 500 and `duration_ms > 0`
      - `all_providers_failed_logs_attempt_rows`
      Files: `crates/ai-gateway/src/gateway/handlers.rs`, `crates/ai-gateway/src/router/router.rs`, `crates/ai-gateway/tests/request_outcome_logging.rs`
      Verify: `cargo test -p ai-gateway --test request_outcome_logging` and `cargo test -p ai-gateway --lib gateway::` pass; the clean-build gate passes.

## FEAT-003 — Failover policy for timeouts and pass-through fallbacks + docs (F6, F7, F9)

- [ ] 11. Timeout-class errors fail over without a same-provider retry (DD10).
      Add `fn is_timeout_class(err: &GatewayError) -> bool` in `router.rs`. In the
      `dispatch_attempts_under_permit` retry loop, return immediately (after recording the ledger
      attempt) on TTFB, total and chunk timeouts and on HTTP 504/524, instead of `continue`.
      Router tests (two wiremock providers in one group, `max_retries_per_provider: 1`):
      - `timeout_class_errors_fail_over_without_same_provider_retry` (A delays beyond ttfb, `.expect(1)`; B answers)
      - `upstream_504_fails_over_without_same_provider_retry` (A 504 `.expect(1)`)
      - `upstream_500_still_retries_same_provider` (A 500 `.expect(2)`)
      Files: `crates/ai-gateway/src/router/router.rs`
      Verify: `cargo test -p ai-gateway --lib fail_over_without_same_provider_retry` and `--lib still_retries_same_provider` pass.

- [ ] 12. Pass-through fallbacks exclude and record the failed provider (DD11).
      Edit `route_request_streaming_excluding` at the TTFB timeout and send error arms (~9002-9018)
      and the final non-success fallback (~9156-9162, 5xx only).
      Test `streaming_ttfb_timeout_fallback_excludes_failed_provider`:
      - pass-through enabled, Codex search off
      - provider A delays headers beyond ttfb, `.expect(1)`
      - provider B answers
      - assert A was hit once, the ledger has a `ttfb_timeout` attempt, and A's breaker recorded a failure

      Follow the existing `smart_routing_streaming_passthrough_*` tests for setup.
      Files: `crates/ai-gateway/src/router/router.rs`
      Verify: `cargo test -p ai-gateway --lib streaming_ttfb_timeout_fallback` passes.

- [ ] 13. ADR, AGENTS.md and wiki (F9).
      Create `docs/adr/0001-buffered-dispatch-streams-upstream.md`:
      - Context: the 2026-10-05/06 incident numbers from `investigation.md` §2
      - Decision: DD1-DD12
      - Consequences, and alternatives rejected: raising TTFB; leaving `stream:false`; a config knob for the retry policy
      - Pending D1-D3

      Add a bullet group to `AGENTS.md` "Non-Obvious Patterns":
      - buffered dispatch streams upstream; never force `stream:false` by default
      - TTFB = first body byte
      - timeout-class errors don't retry the same provider
      - every failed try and every abandoned request is logged via the ledger (499/504 rows, `error_class`)

      Name the guard tests `buffered_sse_reader_allows_generation_longer_than_ttfb_when_bytes_flow`,
      `buffered_dispatch_requests_upstream_stream_and_reassembles_sse`,
      `timeout_class_errors_fail_over_without_same_provider_retry` and
      `abandoned_stream_writes_client_disconnect_outcome_row`.
      Update the `wiki/Streaming.md` Buffer-and-Replay row to list "Codex Search enabled (all
      streaming requests)" and note that buffered responses are fetched from the provider as a
      stream. Add the `buffered_upstream_streaming` row to the `wiki/Providers.md` timeouts/fields
      table.
      Files: `docs/adr/0001-buffered-dispatch-streams-upstream.md`, `AGENTS.md`, `wiki/Streaming.md`, `wiki/Providers.md`
      Verify: the files exist and every referenced test name exists (`cargo test -p ai-gateway -- --list` contains each name).

- [ ] 14. Final gate.
      Verify: `cargo check -p ai-gateway --all-targets` → 0 errors, 0 warnings. `cargo test -p ai-gateway --lib` passes. Every integration binary passes (`cargo test -p ai-gateway --no-run`, then `--test <stem>` in batches). `git status` shows only intended files, nothing committed.

## Post-deploy verification (user-run; not part of the loop)
After the user approves and redeploys, check that:
- `GET /dashboard/logs?model=glm-5.3:dev&status_code=504` stops growing
- EH glm-5.3:dev rows longer than 100 s appear without `#attempt` siblings
- abandoned requests now appear as 499 rows with `error_class`
- the "Calling provider" log lines show `pre_dispatch_ms`, which attributes the ~5 s unexplained overhead

If EH still 504s at ~100 s on streams, the ledger shows `upstream_http_504` with
`duration_ms ≈ 100000`; send that to EH.
