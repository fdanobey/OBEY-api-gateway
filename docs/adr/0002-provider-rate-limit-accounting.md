# ADR 0002: Provider rate-limit accounting and bounded wait

Status: accepted (pending deploy approval)
Date: 2026-10-08

## Context

After ADR 0001 was deployed, Electron Hub showed 100% success while the gateway kept failing over from `glm-5.3:dev` to NIM. The Errors tab showed nothing useful.

- The skipped requests never reached Electron Hub. The local token bucket refused them (`rate_limit_skip`, 429, 0 ms), and the gateway failed over to a chain that often cost 2-4 minutes (NIM TTFB timeouts, then Electron Hub flash).
- The live `rate_limit_per_minute` for Electron Hub was **2**, not 8. A replay of the day's logs fits 2/min and explains none of the 141 skips at 8/min. The code never charged two tokens for one request (`.agents/tasks/eh-rate-limit-overconsumption/investigation.md` §3-§5).
- The code did **under**-count. Same-provider retries, the one-shot `stream:false` retry, the image/reasoning repair retries, the reasoning nudge, Codex Search rounds and Codex/Bedrock repair retries all sent real upstream requests without a token.
- The gateway skipped to the fallback even when the next token was seconds away (7.5 s at 8/min).
- The Errors tab scanned the newest `limit * 4` rows and then filtered `status >= 400`, so 100 later successes hid an error. Attempt rows carry no body (content-free policy), so the reason column was blank. The skip message "Rate limit exhausted" read like an upstream 429.

## Decision

1. **Exactly one token per real upstream HTTP request.** Nothing else is charged: skipped candidates, cooldown/breaker/budget skips, compression, search execution and other local work cost nothing.
   - **Admission gate:** `route_with_failover_for_group` takes the token for a candidate's first send (`RateLimiter::acquire_within`) and passes `FirstSendToken::Prepaid` to `attempt_with_retry`. If no token is available, the candidate is skipped and recorded as a `rate_limit_skip` attempt.
   - **Continuations are debited:** every later send of an admitted request calls `RateLimiter::debit`. This covers the HTTP dispatch loop (one charge per loop iteration, right before `send()`), `MeteredClient` around the Codex and Bedrock clients (repair retries, Codex-internal Codex Search rounds), `SearchResubmitter` rounds and the reasoning nudge (`FirstSendToken::Debit`). A debit never blocks and never refuses. The balance may go negative, which delays the next admission. In-flight work is never aborted for a token, and the configured limit still holds over the window.
   - The streaming pass-through gate (`route_request_streaming_excluding`), fine-tuning and pass-through endpoints keep one `consume()` per send. Each of their fallbacks is a separate request and goes through the admission gate.
2. **Bounded wait at admission.** Only the primary candidate (index 0 in the failover order) waits, and only when the next token is at most `min(rate_limit_max_wait_ms, one refill)` away. The default is 10000 ms, the validated range is 0-60000, and `0` disables the wait. The wait never covers an upstream cooldown. It runs with nothing held: the config guard is dropped first and the provider permit is acquired later inside `attempt_with_retry`. Sends inside `dispatch_attempts_under_permit` hold the permit, so they never wait. The admin panel exposes the setting as "Rate-limit max wait (ms)" next to the rate limit.
3. **Skip text states the bucket.** `RateLimitShortfall` renders `local rate limit 8/min exhausted (0.12 tokens available); next token in 6.6s; request not sent to provider`, or `local cooldown after an upstream rate limit (Ns remaining); request not sent to provider`. Status 429 and `error_class: rate_limit_skip` are unchanged (DD8).
4. **The Errors tab shows failed attempts.** `recent_errors` filters `status_code >= 400` in SQL (`LogFilter.min_status_code`) before `LIMIT`, so the newest 25 errors always appear, including `#attempt` rows of requests that later succeeded. The JSON shape is unchanged. The UI adds a Kind column ("Attempt (failover/retry)" or "Final") and a short trace id. When there is no body, the reason falls back to a label for `error_class`. No content is added.

## Consequences

- At 8/min, a sequential client that drains the bucket waits up to 7.5 s for the primary. Before, it failed over for minutes.
- Retries and Codex Search rounds now count against the limit. A request with one search round uses 2 tokens, so after it the next admission may wait or skip. That is the account limit the setting exists to protect. At the live 2/min (30 s per token), the 10 s default wait does not cover a refill and skips still happen. Raise the limit or set `rate_limit_max_wait_ms` up to 30000 if that is intended.
- Under contention, another request can take the refilled token first. The waiter then re-checks within the same budget and skips if it runs out, as before.
- SDK-internal retries (AWS SDK, the Codex client's 401 refresh) are invisible to the gateway and are not charged.

## Invariants and guard tests

Do not regress these.

| Invariant | Test |
|---|---|
| Tokens consumed == upstream requests received (stream and non-stream with Codex Search on; plain, 500→retry, search round) | `single_request_consumes_exactly_one_provider_rate_token` (router.rs) |
| Sequential requests at the limit are never skipped (no over-charge) | `sequential_requests_under_limit_never_rate_limit_skip` (router.rs) |
| Primary waits briefly for its token; over budget it still fails over | `rate_limit_skip_waits_briefly_for_primary_token` (router.rs) |
| Skip text states the local bucket state and "not sent" | `rate_limit_skip_message_states_local_bucket_state` (router.rs) |
| Errors tab keeps attempt rows behind 100+ newer successes | `errors_tab_includes_failed_attempts_on_successful_request` (dashboard/mod.rs) |
| Limiter primitives | `acquire_within_*`, `debit_charges_even_when_empty_and_delays_next_admission` (rate_limiter.rs) |
| Admin round-trip and range check | `admin_config_round_trips_rate_limit_max_wait_ms` (admin/mod.rs) |

## Verification record (2026-10-08, Windows 11)

- `cargo check -p ai-gateway --all-targets`: 0 errors, 0 warnings.
- `PROPTEST_CASES=64 cargo test -p ai-gateway`: all 53 test binaries passed with 0 failures (lib: 2404 passed).
- Pre-fix behavior, shown by temporary mutations that were all reverted. The new tests can't compile against the old code because the API changed.
  - `debit()` as a no-op (old continuation accounting): `single_request_consumes_exactly_one_provider_rate_token` failed with `non-stream retry: tokens consumed (0.998) must equal upstream requests (2)`.
  - Bounded wait disabled (old skip-at-once policy): `rate_limit_skip_waits_briefly_for_primary_token` failed because `backup` served the request.
  - Old `recent_errors` (`limit * 4` scan, then filter): `errors_tab_includes_failed_attempts_on_successful_request` failed with `rate_limit_skip attempt missing from Errors tab: []`.
  - Old message "Rate limit exhausted": `rate_limit_skip_message_states_local_bucket_state` failed.
  - `sequential_requests_under_limit_never_rate_limit_skip` passed under all of the above. This matches the investigation: the old code never over-charged. With an injected double charge at admission, it failed, and `single_request_...` failed with `tokens consumed (1.999) must equal upstream requests (1)`.
- UI checked with chrome-devtools against a local build on a temp config:
  - The Errors tab rendered a `rate_limit_skip` attempt as "Attempt (failover/retry) | 429 | Local rate limiter: not sent to provider", with no console errors.
  - The admin panel loaded `rate_limit_max_wait_ms` (0 and blank), saved 5000, and `GET /admin/config` returned it.
