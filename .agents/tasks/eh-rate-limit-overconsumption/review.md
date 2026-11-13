# Exact-once provider rate-limit accounting, bounded primary wait, and an honest Errors tab

The investigation found no token over-consumption. Electron Hub skips came from a live `rate_limit_per_minute: 2`, a policy that skipped to a minutes-slower fallback when the next token was seconds away, and an Errors tab that hid attempt rows. The diff makes the invariant "one token per real upstream HTTP request" structural. Admission (`acquire_within`) prepays the first send. Every later send is `debit`ed right before the single `send()` in the HTTP dispatch loop, or by the `MeteredClient` wrapper around the Codex and Bedrock clients. Codex Search rounds and the reasoning nudge are debited too. The primary candidate can wait up to `rate_limit_max_wait_ms` (default 10 s, cap 60 s, matching admin control). The Errors tab now filters in SQL and labels content-free attempt rows. The skip message states the bucket state and says "not sent to provider".

Watch for: the "never more than one refill" wait cap in the docs and admin hint is not enforced in code (confirmed). Waits are folded into request latency without a separate signal, which matters while the user builds an Electron Hub latency ticket (likely). The live Electron Hub limit is still 2/min, and only a config change fixes that.

**Verdict**: APPROVED

## High-level view

Accounting is now enforced at the point of sending, not at the point of deciding. The admission gate in `route_with_failover_for_group` takes one token per dispatched candidate and passes `FirstSendToken::Prepaid`. `dispatch_attempts_under_permit` has exactly one HTTP `send()`, and a debit sits immediately before it unless the send is prepaid. Same-provider retries, the `stream:false` retry and the repair retries all loop through that point, so they can't escape charging. Codex and Bedrock perform their own HTTP, so `MeteredClient` charges per `chat_completion`. Skipped candidates still cost nothing. Debits never block and may drive the balance negative, so nothing waits while the provider permit is held. Electron Hub's limit is not touched by the diff; the live 2/min remains an operator setting.

The bounded wait applies only to candidate index 0, runs after the config guard is dropped and before the permit is acquired, and never waits through an upstream cooldown. It is bounded by `rate_limit_max_wait_ms` (validated 0-60000, default 10000). Admin-panel parity is complete: input, load, save, client-side range check, and a round-trip test. The ADR, config doc and admin hint all promise the wait never exceeds one token's refill, but the code bounds it only by `max_wait`. A negative balance after debits can make the wait longer than one refill.

On the Errors tab, `LogFilter.min_status_code` moves the `status_code >= 400` filter into SQL ahead of `LIMIT` (`status_code` is already indexed). The REST handler and the WebSocket push both use it. The JSON shape is unchanged. The UI adds Kind ("Attempt (failover/retry)" or "Final") and Trace columns and maps `error_class` to escaped labels. No content is added.

Guard tests cover the invariant (plain, 500→retry and search round, with stream and non-stream, Codex Search on), sequential traffic at 8/min, the bounded wait and its over-budget fallback, the skip message, and the Errors tab under 101 newer successes. The new APIs mean these tests can't compile against old code. Instead, the ADR's verification record shows each one failing under a targeted reverting mutation. The record also shows a clean `cargo check --all-targets` and a full passing suite. The ADR exists and AGENTS.md has a single pointer line. ADR 0001 surfaces are untouched apart from the debit inserted before the send.

<details>
<summary>Issues (6)</summary>

1. **Unenforced one-refill wait cap** (confirmed, non-blocking): `acquire_within` and the router bound the wait only by `rate_limit_max_wait_ms`. The config doc, ADR §2 and the admin hint all say it never exceeds one refill. Either clamp `max_wait` to `60/rpm` seconds at the admission site, or drop the claim from all three places.
2. **Wait time invisible in logs** (likely, non-blocking): a successful primary that waited is visible only through a `tracing::info!`. Its outcome row's duration includes up to 10-60 s of local wait. Record the wait (a metric such as `rate_limit_wait_ms_by_provider`, or a ledger/active-phase marker) so local waiting isn't mistaken for Electron Hub latency.
3. **Admission token lost when no send follows** (likely, pre-existing ordering, non-blocking): the prepaid token is taken before compression and `acquire_provider_slot_or_reject`. A `slot_saturated` rejection or a pre-send configuration error spends a token with no upstream request, which contradicts ADR 0002's "nothing else is charged". Refund on pre-send failure, or narrow the ADR wording.
4. **Wall-clock lower bound in wait test** (possible, non-blocking): `rate_limit_skip_waits_briefly_for_primary_token` asserts `waited >= 400ms` against an expected ~500 ms on the real clock. Under heavy parallel load that leaves about 100 ms of slack. Consider a paused clock, or assert only "primary served, no skip".
5. **Untested continuation paths** (non-blocking): no test exercises the reasoning-nudge `Debit`, the `stream:false` retry, or the `MeteredClient` charging on the Codex/Bedrock paths. The single-`send()` structure covers the `stream:false` retry by construction, but the nudge and the wrapper have no guard.
6. **Live Electron Hub limit still 2/min** (confirmed, operational): with 10 s default wait and a 30 s refill, skips continue at 2/min. Restore the deliberate 8/min in the admin panel (Providers → Electron Hub). Or, if 2/min is intended, set `rate_limit_max_wait_ms` to 30000.

</details>

<details>
<summary>Details</summary>

### Where the token is charged now

```
route_with_failover_for_group (per candidate, no locks/permit held)
  └─ RateLimiter::acquire_within(max_wait if index 0 else 0)   ← admission, may wait
       ok → attempt_with_retry(.., Prepaid)
              └─ acquire_provider_slot_or_reject                 ← permit held from here
                  └─ dispatch_attempts_under_permit
                       ├─ Codex:   MeteredClient(prepaid) → each chat_completion charges (repairs, interceptor rounds)
                       ├─ Bedrock: MeteredClient(prepaid) → each chat_completion charges
                       └─ HTTP:    for attempt { … debit unless prepaid; send() }  ← the only send()
       err → rate_limit_skip attempt row (429, bucket-state message), continue
SearchResubmitter / reasoning nudge → attempt_with_retry(.., Debit)
```

Every chat-completions upstream request in the router goes through the dispatch-loop `send()` or through a metered client, so the invariant is structural rather than a convention. Charging at the send site closes the under-count sites U1-U5 from the investigation. The streaming pass-through, fine-tuning and pass-through endpoints keep their single `consume()`, and each of their fallbacks passes admission again.

### Bounded wait semantics

`acquire_within` loops on `try_acquire`. It refuses at once when `elapsed + retry_in > max_wait` or when the refusal is a cooldown, and otherwise sleeps for `retry_in + 1ms` with the bucket mutex released. Contention is handled by re-checking against the same overall budget. The router snapshots per-provider waits before `drop(config)`, so the sleep never runs under the config guard. The permit is acquired later, inside `attempt_with_retry_with_permit`, and the streaming pass-through fallback explicitly drops its permit before calling `route_request`. No path waits while holding a permit.

The gap is the cap. `retry_in = (1 - tokens) / rate`, and debits can leave `tokens` negative. At 8/min with the 10 s default, a balance of −0.3 after a search round gives `retry_in` ≈ 9.75 s. That is inside the budget, so the primary waits longer than the 7.5 s one-refill time the docs promise. At `max_wait` 60000 and 2/min, a balance of −1 produces a 60 s wait. This stays within the configured cap, so requirement 3 holds, but three user-facing texts describe a bound that doesn't exist.

The wait only reaches `tracing::info!`. The outcome row of a waited request carries the full duration with no marker. The user is assembling latency evidence for an Electron Hub ticket, so up to 10 s of local wait per request could be read as provider slowness.

### Admission token before pre-send work

The admission token is taken before compression and before `acquire_provider_slot_or_reject`. If the slot is saturated, or the provider config or OAuth wiring fails before the dispatch loop, the token is spent without any upstream request. This ordering predates the diff. However, ADR 0002 now states as an invariant that nothing besides real upstream requests is charged. At Electron Hub's `max_connections: 2` with sequential traffic this is rare. Under concurrency, each `slot_saturated` rejection leaks a token.

### Test coverage

The router wait test uses the real clock with a 120/min bucket. Its `>= 400ms` lower bound sits against an expected ~500 ms, which could flake on a loaded machine. The limiter unit tests use `start_paused`.

Not tested: the nudge's `Debit`, the `stream:false` retry's charge, `MeteredClient` on the Codex and Bedrock paths, and token loss on `slot_saturated`.

</details>

<details>
<summary>File map</summary>

- `router/rate_limiter.rs`: `RateLimitShortfall` (Display with bucket state), `try_acquire`, `acquire_within`, `debit`; paused-clock unit tests.
- `router/router.rs`: `FirstSendToken`, `MeteredClient`, admission with bounded wait, debit before the dispatch-loop `send()`, Codex/Bedrock metering, `SearchResubmitter` and nudge debits, new guard tests.
- `config/mod.rs`, `config/validation.rs`: `rate_limit_max_wait_ms`, default 10000, range 0-60000.
- `logger/mod.rs`: `LogFilter.min_status_code`.
- `dashboard/mod.rs`, `dashboard/static/index.html`: SQL-filtered `recent_errors`, Kind and Trace columns, `error_class` labels, guard test.
- `admin/static/index.html`, `admin/mod.rs`: "Rate-limit max wait (ms)" control with load/save/validation; round-trip test.
- `docs/adr/0002-provider-rate-limit-accounting.md`, `AGENTS.md`: ADR with verification record; one pointer line.
- Remaining `src/**` and `tests/**` files: `rate_limit_max_wait_ms: None` added to `Provider` literals.

Full diff: `git diff` in the repository root.

</details>
