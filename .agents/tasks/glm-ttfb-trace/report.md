# GLM / Electron Hub reliability: report

All times UTC (`Z`) unless marked MST. Electron Hub's dashboard shows completion times in MST (UTC-7).

## 1. Verdict

The failures have three separate sources, and they stack.

- **Provider (Electron Hub).** For non-streaming `glm-5.3:dev` requests with ~135-155K-token prompts, EH returns HTTP 504 with the plain-text body `error code: 504` 99-103 s after the request is sent. EH's backend keeps generating, so its dashboard logs the same request as Success with full output tokens. Between 11:59:59 AM and 1:33:30 PM MST on Oct 5, EH logged 33 dev Successes and only 2 of them reached the caller. Through 2026-10-04 ~18:40Z, EH completed requests of this kind that ran 125-261 s. Since ~21:00Z that day, nothing has succeeded past ~120 s. The cause on EH's side is unverified.
- **Router.** We found no holds, lock contention or runtime blocking. Slots, the rate limiter, config locks, the breaker and SQLite were all checked and ruled out. What the router did get wrong:
  1. While Codex Search is enabled, every streaming request is buffered and sent upstream with `stream:false`. That walks straight into EH's ~100 s non-streaming limit.
  2. TTFB was measured at response headers. On `stream:false` that caps the whole generation at 120 s, which is why Nano-GPT and NIM failed the same way.
  3. After a timeout or 504, the router retried the same provider. That added ~100 s and a second EH generation, which explains the duplicate entries on EH's dashboard.
  4. Abandoned requests (~19 of them), hidden retries (~19) and breaker skips left no log rows.
  5. Smaller delays: ~7 s spent before the upstream send, and a 2 s loop-detection throttle caused by the client's re-sends.
- **Client.** The client gives up at about 300 s and re-sends the same turn 4-6 s later. The router has no 300 s timer on this path.

## 2. Electron Hub support ticket (ready to paste)

> **Subject:** glm-5.3:dev non-streaming requests return HTTP 504 after ~100 s, but your dashboard shows them as Success
>
> Since about 2026-10-04 21:00 UTC (Oct 4, 2 PM MST), our application's non-streaming `POST /v1/chat/completions` calls to `glm-5.3:dev` (prompts ~135K-155K tokens, mostly cached) often fail. They return **HTTP 504 with the plain-text body `error code: 504`, consistently 99-103 seconds after the request is sent**. Measured examples (UTC, Oct 6):
> - sent 05:15:39, 504 at 05:17:20
> - sent 05:17:22, 504 at 05:19:05
> - sent 05:07:13, 504 at 05:08:55
>
> Between 06:37 and 08:10 UTC on Oct 6 we logged nine more 504s from your endpoint.
>
> Your usage dashboard lists these same requests as **Success** with full output tokens. On Oct 5 between 11:59:59 AM and 1:33:30 PM MST, it shows 33 successful `glm-5.3:dev` completions on our key. Our application received only 2 of them: the ones completing at 1:19:59 PM and 1:30:37 PM. Every other request ended for us in a 504, or we stopped waiting. Several prompts appear repeatedly on your side, for example input 136,202 tokens ten times between 12:37 and 1:14 PM. Those are our retries after receiving your 504s, and your side generated each one to completion.
>
> Before this, on Oct 3-4, non-streaming requests of the same kind that took 125-261 seconds completed normally.
>
> Questions:
> 1. Is there a ~100 s timeout on non-streaming responses (edge or load balancer)? Was it introduced or changed around Oct 4?
> 2. Are streaming (`stream: true`) requests exempt from it?
> 3. When your edge returns 504, or the caller disconnects, is the backend generation cancelled? Are those requests billed or counted against limits?
> 4. Can the dashboard mark requests whose response was never delivered (504 or disconnect) instead of showing Success?
>
> Account: Master Key (Chandler/Mesa, AZ). Your dashboard uses MST (UTC-7) completion times; our times are UTC.

## 3. Router changes (UNCOMMITTED, NOT deployed)

All changes are in the working tree only. Nothing is committed, pushed or deployed. They are waiting for your approval to commit and redeploy.

| File | Change |
|---|---|
| `src/router/router.rs` | Buffered dispatch now sends `stream:true` + `stream_options.include_usage` and reassembles the SSE internally. A new incremental reader enforces: TTFB = headers plus the first body byte, idle gap = `chunk_timeout_seconds`, `total_timeout` per try. Other changes: one-shot `stream:false` fallback when a provider rejects `stream_options`; reassembly keeps the full usage object; a stream that ends without `[DONE]` or `finish_reason` is a retryable truncation; timeout-class errors (TTFB/total/chunk, upstream 504/524) fail over with no same-provider retry; pass-through fallbacks exclude the failed `provider:model`; attempt and breaker-skip recording; `pre_dispatch_ms` in the "Calling provider" log |
| `src/config/mod.rs` | New per-provider `buffered_upstream_streaming` (default on, off for Bedrock) |
| `src/error/mod.rs` | `GatewayError::ChunkTimeout` (504); internal `duration_ms`/`error_class` on attempts; client JSON unchanged |
| `src/active_requests.rs` | Per-request attempt ledger, `outcome_logged` flag, breaker-skip claim |
| `src/gateway/handlers.rs` | `log_outcome` at every terminal point; drains the ledger; abandoned requests write a 499 `client_disconnect` or 504 `gateway_deadline` row; memory-timing log |
| `src/logger/mod.rs` | Additive `error_class` column (migration) |
| `src/admin/static/index.html`, `src/admin/mod.rs` | Admin checkbox for `buffered_upstream_streaming`, with a round-trip test |
| `tests/request_outcome_logging.rs` (new) + literal-field updates in other tests | Outcome and attempt row tests |
| `docs/adr/0001-buffered-dispatch-streams-upstream.md` (new) | ADR: context, decisions, rejected alternatives |
| `AGENTS.md` | Non-Obvious-Patterns note: never force `stream:false` by default, plus the guard test names |
| `wiki/Streaming.md`, `wiki/Providers.md` | Docs for Codex Search buffer-and-replay and the new option |

Regression guards named in AGENTS.md:
- `buffered_sse_reader_allows_generation_longer_than_ttfb_when_bytes_flow`
- `buffered_dispatch_requests_upstream_stream_and_reassembles_sse`
- `timeout_class_errors_fail_over_without_same_provider_retry`
- `abandoned_stream_writes_client_disconnect_outcome_row`

Also covered by tests: `upstream_504_fails_over_without_same_provider_retry`, `upstream_500_still_retries_same_provider`, `streaming_ttfb_timeout_fallback_excludes_failed_provider`, `same_provider_retry_failure_is_logged_as_attempt_row`, `all_providers_failed_logs_attempt_rows`, `error_class_column_migrates_and_round_trips`.

Diff size: 28 tracked files, +2561/−323, plus the untracked new test, the ADR and `.agents/`.

## 4. Verification

- `cargo check -p ai-gateway --all-targets`: exit 0, 0 warnings, 0 errors (final run).
- `cargo test -p ai-gateway --lib`: 2394 passed, 0 failed.
- All 50 integration binaries pass, including `request_outcome_logging` 4, `integration` 35 and `structured_output_integration` 9. The wall-clock `--ignored` tests were not run.
- `cargo test -- --list` shows all four guard tests.
- No `#[allow]` added, no secrets in the diff, and no internal `gateway_*` keys reach clients.
- Reviewer verdict: **APPROVED**, with 5 minor, non-blocking findings:
  1. A client that sends `stream_options` without `include_usage` gets estimated rather than real usage.
  2. The `stream:false` fallback needs a free retry slot (it never fires with `max_retries_per_provider: 0`).
  3. The "stream" rejection matcher is too broad.
  4. A doc comment in handlers.rs is attached to the wrong function.
  5. No test streams a slow body end to end through the dispatch path.

## 5. Decisions needed from you

- **D1 Codex Search vs live streaming.** Even with the fix, the Codex Search gate means the client gets tokens only once the full completion is assembled. Turns longer than ~290 s will still hit the client's 300 s abort. Options:
  1. A per-model-group opt-out to pass-through, which means no web search for those requests.
  2. Raise the client timeout.
  3. Build search interception at stream time (a large feature).
- **D2 Loop detection on re-sent turns.** The client's identical re-sends after failures escalate the session (currently `throttle`, 2 s per request). The next level, `inject`, would edit the system prompt and break the prompt cache. Should we exempt re-sends whose previous request never completed?
- **D3 Client timeout.** The ~300 s abort is a client setting. Consider 600 s or more for GLM.
- **D4 `codex_search_ready` doc/code mismatch.** It returns ready when `enabled: true` even without an OAuth token. No live impact today; flagged only.

## 6. Post-deploy checks (after approval and redeploy)

- `GET /dashboard/logs?model=glm-5.3:dev&status_code=504` should stop growing.
- EH glm-5.3:dev rows longer than 100 s should appear with no `#attempt` siblings.
- Abandoned requests should show up as 499 rows with an `error_class`.
- "Calling provider" log lines should include `pre_dispatch_ms`, which will attribute the ~5 s of unexplained overhead.
- Reassembled rows should still carry `cache_read_tokens`/`reasoning_tokens`.
- If EH still returns 504 at ~100 s on streams, the ledger will show `upstream_http_504` with `duration_ms` ≈ 100000. Add that to the ticket.
- Watch for `stream_truncated` rows with a near-constant `duration_ms`, which would point to an edge cutting streams at a fixed age.

## 7. Synthetic probe to ignore on EH's dashboard

One test request was sent at **2026-10-06 13:00:29Z (06:00:29 AM MST)** to `glm-5.3:dev`: a short "Count from 1 to 60" prompt with `stream:true`, trace `dd4b6315`. It appears on EH's dashboard as a small request completing around 06:00:35 AM MST. You can ignore it.
