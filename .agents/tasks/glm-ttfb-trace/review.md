# Buffered dispatch streams upstream, timeout-class failover, and per-try request logging

The buffered HTTP dispatch now asks providers for an SSE stream (`stream: true` + `stream_options.include_usage`) and reassembles it, instead of always sending `stream: false`. That removes the ~100 s Electron Hub non-streaming 504 and the 120 s header-TTFB cap that killed long GLM generations. TTFB now covers the first body byte, idle gaps use `streaming.chunk_timeout_seconds`, and `total_timeout` caps each try. Timeout-class failures (TTFB/total/chunk, upstream 504/524) fail over without a same-provider retry, and pass-through TTFB/send/5xx fallbacks exclude the failed `provider:model`. A per-request attempt ledger on `ActiveRequestHandle` replaces the `gateway_failed_attempts` response-extras channel, so every failed try, skip and abandoned request is logged with `duration_ms` and a content-free `error_class`. D1-D4 were not implemented: the Codex Search gate, loop detection and `codex_search_ready` are untouched.

Watch for: a client-supplied `stream_options` without `include_usage` now drops real usage on the buffered path (likely). The DD4 `stream:false` fallback never fires when `max_retries_per_provider: 0` (confirmed). `is_stream_param_rejection` matches any 400 body that contains "stream" (confirmed). A doc comment was split onto the wrong function in handlers.rs (confirmed). None of these block shipping.

**Verdict**: APPROVED

## High-level view

The transport switch is per provider (`buffered_upstream_streaming`, default on except Bedrock). Requests with `n > 1`, `logprobs`/`top_logprobs` or audio, and providers that opt out, keep `stream: false` with the old header-TTFB plus `response.text()` timing. A streamed success body goes through `read_upstream_body`, which enforces first-byte, idle and total deadlines measured from each try's own `request_start`. Error statuses still read small JSON bodies with `response.text()`. Reassembly keeps the last `usage` object whole, so `prompt_tokens_details`/`completion_tokens_details` survive and the gateway cache/reasoning extras populate. A stream with neither `[DONE]` nor a `finish_reason` is a retryable truncation, while the shared `reassemble_sse_response` callers still accept partial bodies. One gap: the gateway only adds `stream_options` when the client did not send one.

DD10 is applied at every timeout site in the dispatch loop: send TTFB, the reader's timeout arms, the `stream:false` total timeout, and HTTP 504/524. A 500 still uses `max_retries_per_provider`. The pass-through path records every non-success try once. On TTFB, send error or 5xx it counts a breaker failure and falls back with the failed entry excluded. When nothing is left, it returns the original failure instead of a misleading 400.

Ledger dedupe works through the failover `Err` arm: it writes to the ledger only when the dispatch did not grow it during that provider's call. Slot saturation, OAuth, Codex and Bedrock errors are still covered. The client-visible `AllProvidersFailed.attempts` JSON is unchanged, because `duration_ms` and `error_class` are `skip_serializing` and breaker pre-filter skips go only to the ledger. Every terminal chat log goes through `log_outcome`, which drains the ledger, writes the row and marks `outcome_logged`. `RequestCompleteGuard::drop` writes the 499/504 abandon row only when the guard was not completed and no outcome was logged, and it marks the handle before writing, so the row appears at most once.

The evidence holds up. `cargo check -p ai-gateway --all-targets` finished with 0 warnings and 0 errors on this pass. 40 targeted lib tests passed (buffered_*, timeout-class, pass-through fallback, ledger, error_class, abandon classification, admin round-trip). `request_outcome_logging` (4), `integration` (35) and `structured_output_integration` (9) passed. verification.md records the full lib run (2394) and all 50 integration binaries passing. No `#[allow]` was added, no secrets appear in the diff, and no `gateway_*` key leaks to clients.

<details>
<summary>Issues (5)</summary>

1. **Client `stream_options` suppresses gateway usage** (likely, minor). When the client already sends `stream_options` without `include_usage: true`, the upstream stream has no usage frame, so cost, budget and VK accounting fall back to token estimates. Before this change, `stream:false` returned real usage. Fix: when `stream_upstream` is true, merge `include_usage: true` into an existing `stream_options` object, and restore the client's original on the DD4 fallback.
2. **DD4 fallback needs a retry slot** (confirmed, minor). The `stream:false` fallback is a `continue` inside `0..=max_retries`, so with `max_retries_per_provider: 0` a provider that rejects `stream_options` fails over on every buffered request and never gets the retry. Fix: let the one-shot fallback run without consuming the retry budget (for example, extend the loop bound by one when it fires), or document that the opt-out is required in that configuration.
3. **`is_stream_param_rejection` is broad** (confirmed, minor). Any 400 body that contains "stream" (after removing "upstream") turns off upstream streaming and retries once, which wastes one try on unrelated 400s that echo the word. Fix: match parameter-shaped phrases such as `stream_options`, `'stream'`, `"stream"` or `stream is not supported`.
4. **Doc comment attached to the wrong function** (confirmed, minor). In `gateway/handlers.rs` (~4731-4752), `relay_failure_class` was inserted between `is_transient_stream_truncation`'s doc comment and its `fn`. The truncation docs now describe `relay_failure_class`, and `is_transient_stream_truncation` has none. Fix: move `relay_failure_class`/`relay_failure_attempt` above that doc block.
5. **Dispatch-level reader wiring not exercised end to end** (confirmed, minor). The key guard tests `read_upstream_body` directly. No dispatch test streams a slow body through `dispatch_attempts_under_permit`, so mis-wired deadlines (for example `first_byte_deadline` computed from the wrong instant, or the reader bypassed) would not fail any test. Fix: add a dispatch test against a small local streaming server that sends chunks spaced below the idle window for longer than `ttfb_timeout_seconds`.

</details>

<details>
<summary>Details</summary>

### Upstream transport selection and `stream_options`

`stream_upstream = effective_buffered_upstream_streaming() && !requires_non_streaming(&outgoing)` replaces the old unconditional `outgoing.stream = false` (router.rs ~4467). `inserted_stream_options` is set only when the key is absent, which follows DD4 to the letter. The consequence is item 1. The live clients (Roo/Kilo-style) usually send `{include_usage: true}` and are unaffected. A client that sends `stream_options` for other reasons (`include_usage: false`, or only obfuscation flags) gets an upstream stream with no usage frame, so `reassemble_sse_stream` estimates `completion_tokens` from character length and leaves `prompt_tokens` at 0. Before the change, `stream:false` always returned the provider's real usage on this path. The client never sees the upstream stream options (the gateway replays the response itself), so overriding `include_usage` for the upstream call has no client-visible effect.

NIM's allowlist strips `stream_options`, so NIM streams without usage and the "Sanitized request" info log now fires on every NIM buffered request. The ADR documents this. It is log noise only.

### One-shot `stream:false` fallback

```
for attempt in 0..=max_retries {
    ...
    if stream_upstream && !stream_fallback_done && status_code == 400
        && Self::is_stream_param_rejection(&body_text) {
        stream_fallback_done = true; stream_upstream = false; outgoing.stream = false;
        if inserted_stream_options { outgoing.extra.remove("stream_options"); }
        skip_next_backoff = true; record_try(...); last_error = Some(err); continue;
    }
```

The fallback uses one iteration of the retry budget (item 2), the same as the image-strip retry it mirrors. With `max_retries_per_provider: 1` (the live setting) a 400 on the first try is rescued. A 400 after an earlier 500 retry is not, and neither is any 400 under `max_retries_per_provider: 0`. The rejection matcher (item 3) runs only on a 400 and is bounded by the one-shot flag, so the worst case is one wasted try per request. It is still a false positive whenever a validation message or an echoed parameter list contains "stream". The fallback is not persisted across requests, so a provider that rejects streaming pays the extra round trip on every request until `buffered_upstream_streaming: false` is set. The warn log says so.

### Incremental reader and deadlines

The deadlines come from the per-try `request_start`, so a same-provider retry after a 500 gets fresh windows. As item 5 says, nothing above the reader proves the dispatch actually routes streamed success bodies through it with those limits. `ttfb_timeout_attempt_records_duration_and_error_class` and the DD10 TTFB test use a header delay, which trips the `send()` timeout, not the body reader.

A clean server close mid-generation is classified as `stream_truncated` and retried on the same provider (DD5). If Electron Hub's edge turns out to cut streaming responses at a fixed age instead of returning 504, this would reproduce the duplicate-generation pattern DD10 removed for 504s. The post-deploy ledger check (`stream_truncated` rows with `duration_ms` near a constant) shows this directly.

### Ledger dedupe and the abandoned-request row

```
dispatch_attempts_under_permit ── record_try() on every failed try ──► ledger
route_with_failover_for_group  ── skips / content failovers ─────────► ledger + aggregate
        Err arm ── ledger grew during this provider? ── yes ─► aggregate only
                                                        └─ no ──► ledger + aggregate
route_request_with_exclusions  ── circuit_open_skip (once, excluded keys skipped) ─► ledger
handler log_outcome / guard Drop ── take_failed_attempts() ─► #attempt rows
```

Every `continue`/`return Err` in the dispatch loop records first (checked line by line), and cache-hit and guardrail exits complete the guard or log through `log_outcome`, so neither a hidden retry nor a false 499 row is possible on the chat paths.

### Documentation corruption in handlers.rs

`relay_failure_class` sits between the long `/// Classify a pre-content RelayOutcome::FailedBeforeContent reason…` block and `fn is_transient_stream_truncation` (item 4). rustdoc now joins both texts onto `relay_failure_class`, and the truncation classifier has no documentation.

</details>

<details>
<summary>File map</summary>

- `crates/ai-gateway/src/router/router.rs`: streamed buffered dispatch, `read_upstream_body`, SSE detection and reassembly (`reassemble_sse_stream`, `sse_data_payloads`), DD4 fallback, DD10 `is_timeout_class`, ledger recording (`record_try_failure`, `push_attempt`, `record_breaker_skips`), DD11 `passthrough_fallback_excluding`, `pre_dispatch_ms`, tests.
- `crates/ai-gateway/src/gateway/handlers.rs`: `log_outcome`, ledger-based `log_failed_attempts`, `AbandonLogger` in `RequestCompleteGuard`, `ChunkTimeout` response/SSE classification, relay attempt classes, memory timing log.
- `crates/ai-gateway/src/active_requests.rs`: ledger, `outcome_logged`, breaker-skip claim, `elapsed_ms`.
- `crates/ai-gateway/src/error/mod.rs`: `ChunkTimeout`, internal `ProviderAttempt` fields, `error_class_for`.
- `crates/ai-gateway/src/logger/mod.rs`: additive `error_class` column.
- `crates/ai-gateway/src/config/mod.rs`: `buffered_upstream_streaming` and its effective default.
- `crates/ai-gateway/src/admin/static/index.html`, `admin/mod.rs`: checkbox load/save and round-trip test.
- `crates/ai-gateway/tests/request_outcome_logging.rs` (new): abandon/complete/retry/all-failed log rows.
- `crates/ai-gateway/tests/structured_output_integration.rs`: wire assertions updated to `stream:true`.
- Other `src/` and `tests/` files: `buffered_upstream_streaming: None` / `error_class: None` literal fields.
- `docs/adr/0001-buffered-dispatch-streams-upstream.md` (new), `AGENTS.md`, `wiki/Streaming.md`, `wiki/Providers.md`: ADR, guard-test note, docs.

Full diff: `git -C "c:\Fbn_Dev_Apps\OBEY API Router" diff HEAD` plus the untracked files above.

</details>
