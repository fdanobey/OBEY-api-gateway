# Electron Hub rate-limit skips: report

Not committed and not pushed. Live config was not changed.

## Verdict

The router never over-consumed tokens. The skips came from the live Electron Hub limit (`rate_limit_per_minute: 2`, i.e. 1 token per 30 s) and from a policy that failed over at once, even when the next token was seconds away. Electron Hub's 100% success is accurate: skipped requests were refused by the local bucket (`rate_limit_skip`, 429, 0 ms) and were never sent to Electron Hub.

Evidence (`investigation.md` §1-§5):

- **Over-consumption sites: none.** Every chat path charged once per dispatched candidate at the single admission site (`route_with_failover_for_group`). The streaming pass-through `consume()` can't be reached while Codex Search is on, because the buffered path returns before it. `check_available()` is used only in tests.
- **Log replay** of the day (549 Electron Hub dispatches, 141 skips) fits a 2/min bucket for 85-97% of events. An 8/min bucket predicts 0 of the 141 skips, even when charging 2-3 tokens per dispatch.
- **The real defect was under-count.** Several real upstream requests took no token: same-provider retries, the `stream:false` retry, repair retries, the reasoning nudge, Codex Search rounds, and Codex/Bedrock-internal retries.

Tokens per client request on the primary (Codex Search on, as live):

| Request | Before | After |
|---|---|---|
| `stream:false`, plain success | 1 token / 1 request | 1 / 1 |
| `stream:true` (forced buffered), plain success | 1 / 1 | 1 / 1 |
| either, 500 then same-provider retry | 1 / 2 (measured 0.999 vs 2 with the fix reverted) | 2 / 2 |
| either, one Codex Search round | 1 / 2 | 2 / 2 |
| skipped candidate (bucket, cooldown or breaker) | 0 | 0 |

## Files changed

- `src/router/rate_limiter.rs`: adds `RateLimitShortfall` (renders the bucket state), `try_acquire`, `acquire_within` (bounded wait) and `debit`, with paused-clock unit tests.
- `src/router/router.rs`: adds `FirstSendToken` and `MeteredClient`. Admission uses the bounded wait (primary only). A debit runs right before the dispatch-loop `send()`, and on Codex Search, nudge and Codex/Bedrock sends. Also the bucket-state skip message and the guard tests.
- `src/config/mod.rs`, `src/config/validation.rs`: `rate_limit_max_wait_ms` (default 10000, range 0-60000).
- `src/logger/mod.rs`: `LogFilter.min_status_code`.
- `src/dashboard/mod.rs`: `recent_errors` filters `>= 400` in SQL before `LIMIT`; the REST and WebSocket paths share it.
- `src/dashboard/static/index.html`: Errors tab gets Kind and Trace columns and labels for `error_class`.
- `src/admin/static/index.html`, `src/admin/mod.rs`: "Rate-limit max wait (ms)" control with load, save and range check, plus a round-trip test.
- `src/gateway/mod.rs`, `src/request_body_limit.rs`, `src/tray/mod.rs`, `tests/*.rs` (11 files): `rate_limit_max_wait_ms: None` added to `Provider` literals.
- `AGENTS.md`: one pointer line to ADR 0002 and its guard tests.
- `docs/adr/0002-provider-rate-limit-accounting.md`: new.

## Tests added

| Test | Guards |
|---|---|
| `single_request_consumes_exactly_one_provider_rate_token` (router.rs) | Tokens consumed == upstream requests, stream and non-stream with Codex Search on: plain, 500→retry, and a search round |
| `sequential_requests_under_limit_never_rate_limit_skip` (router.rs) | 8 sequential requests at 8/min: never skipped, never failed over. Catches any future double charge |
| `rate_limit_skip_waits_briefly_for_primary_token` (router.rs) | Primary waits for a token within `rate_limit_max_wait_ms`; over budget it still fails over |
| `rate_limit_skip_message_states_local_bucket_state` (router.rs) | Skip text says "local rate limit N/min exhausted … request not sent to provider" |
| `errors_tab_includes_failed_attempts_on_successful_request` (dashboard/mod.rs) | An attempt row stays on the Errors tab behind 101 newer successes |
| `acquire_within_*` (3), `debit_charges_even_when_empty_and_delays_next_admission` (rate_limiter.rs) | Limiter primitives: wait inside budget, refuse beyond it, never wait through a cooldown, debit never blocks |
| `admin_config_round_trips_rate_limit_max_wait_ms` (admin/mod.rs) | Admin parity and the 0-60000 range check |

## ADR

`docs/adr/0002-provider-rate-limit-accounting.md`

## Recommended live-config changes (not applied)

- **Electron Hub `rate_limit_per_minute`**: the live value is 2. The Oct 6 snapshot showed 8. If 8 is the deliberate value, restore it under Admin → Providers → Electron Hub. This fix alone does not stop skips at 2/min: the 30 s refill is longer than the 10 s default wait.
- **If 2/min is intended**, set Electron Hub `rate_limit_max_wait_ms` to 30000 so the primary waits about one refill instead of failing over.
- **8/min is still right only if** retries and Codex Search rounds fit inside it. Each one now takes a token. A turn with one search round uses 2, so a sequential client running searches may wait up to 7.5 s at admission. Watch `rate_limit_exhaustions_by_provider` for a day before changing the limit.
- **NIM `ttfb_timeout`**: NIM z-ai/glm-5.3 and z-ai/glm-5.3-flash often hit the 120 s TTFB limit, and z-ai/glm-5.3 returned 404s. Lowering NIM TTFB to about 30-45 s, or moving those entries down the GLM group, makes any remaining failover cost seconds instead of minutes.

## What to check after deploy

- `GET /dashboard/logs?model=glm-5.3:dev&status_code=429` should stop growing during sequential use at the restored limit. At 2/min with a 30000 ms wait, expect rare skips only after search-heavy turns.
- `GET /dashboard/metrics`: `rate_limit_exhaustions_by_provider["Electron Hub"]` should stay flat.
- On the Errors tab, failovers should show as "Attempt (failover/retry)" rows with a reason, e.g. "Local rate limiter: not sent to provider" or "No first byte before TTFB timeout".
- The skip text should read `local rate limit N/min exhausted (x tokens available); next token in Ns; request not sent to provider`.
- When a request waited for a token, the only record is an `info` log line ("Waited for the local rate-limit token…"). That wait is included in the outcome row's `duration_ms`, so don't count it as Electron Hub latency in the ticket.

## Verification (2026-10-08, Windows 11, PowerShell)

| Step | Command | Result |
|---|---|---|
| 1 | `cargo check -p ai-gateway --all-targets` | exit 0, 0 errors, 0 warnings. Re-run after restoring the mutation: same |
| 2 | `cargo build --release -p ai-gateway` | exit 0, 0 warnings, 11m 02s (LTO). It overran two 300 s calls, so it was finished in the background with a log |
| 3a | `cargo test -p ai-gateway --no-run` | exit 0, 0 warnings, 52 test executables |
| 3b | lib unit tests | 2404 passed, 0 failed, 0 ignored (1205 s; the semantic-cache proptests are slow) |
| 3c | bin unit tests | 2407 passed, 0 failed |
| 3d | 50 integration binaries, run one at a time | 286 passed, 0 failed, 14 ignored (`#[ignore]` wall-clock budget tests) |
| 4 | Core accounting reverted (`debit()` decrement removed), then `cargo test -p ai-gateway --lib -- single_request_consumes_exactly_one_provider_rate_token sequential_requests_under_limit_never_rate_limit_skip` | exit 101: `single_request_…` **FAILED**, `non-stream retry: tokens consumed (0.999) must equal upstream requests (2)`. `sequential_…` passed, as expected: the old code never over-charged |
| 4 | Decrement restored; the 10 ADR 0002 guard and primitive tests | 10 passed, 0 failed. `rate_limiter.rs` diff stat matches the pre-mutation stat; no `TEMP-MUTATION` left |

The ADR verification record covers the other pre-fix mutations: wait disabled, old `limit*4` error scan, old "Rate limit exhausted" text, and an injected double charge that makes `sequential_…` fail. Temp logs under `target\` were deleted.

### Open review findings (non-blocking, not fixed here)

1. The ADR, config doc and admin hint promise the wait never exceeds one refill. The code bounds it only by `rate_limit_max_wait_ms`, so a negative balance after debits can exceed one refill.
2. Wait time has no metric or ledger marker (see the last "What to check" item).
3. A prepaid token is lost when no send follows (`slot_saturated`, pre-send config error). The ordering predates this change.
4. The wait test's 400 ms lower bound uses the real clock and may flake under load.
5. No test covers the nudge debit, the charge on the `stream:false` retry, or `MeteredClient` on the Codex and Bedrock paths.
