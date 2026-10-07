# ADR 0001: Buffered dispatch streams upstream; timeouts fail over

Status: accepted (pending deploy approval)
Date: 2026-10-06

## Context

On 2026-10-05/06 the `GLM` model group (primary Electron Hub `glm-5.3:dev`, ~135-155K-token prompts) mostly failed.

- Codex Search was enabled, so every streaming client request took the buffered path, and the buffered dispatch always sent `stream: false` upstream. A probe confirmed this: headers arrived at 112 ms, and all content plus `[DONE]` arrived together at 5.8 s.
- Electron Hub answered those non-streaming requests with a plain-text `HTTP 504` (`error code: 504`) 99-103 s after the send. Four traces were measured, and nine 504 `#attempt` rows were logged between 06:37Z and 08:10Z on Oct 6. The Electron Hub backend kept generating and recorded each request as Success. In one 94-minute window on Oct 5, Electron Hub listed 33 `glm-5.3:dev` successes and the gateway received only 2 of them. On Oct 3-4, requests of the same kind that took 125-261 s still completed. After ~Oct 4 21:00Z, no single try lasted longer than ~120 s.
- The gateway retried the same provider after each 504 or TTFB timeout. That cost another ~100-120 s and produced a duplicate generation. One prompt (136,202 input tokens) was generated 10 times in 37 minutes.
- TTFB was measured at the response headers. With `stream: false`, headers arrive only after the whole completion, so the 120 s TTFB capped every Nano-GPT/NIM generation. Two traces show TTFB, retry, TTFB, failover.
- The client aborts each request at ~300 s. About 19 abandoned requests left no log row. Another ~19 same-provider retry failures and the breaker skips were invisible, because only per-provider final errors were logged.
- The streaming pass-through TTFB, send-error and non-success fallbacks re-sent the request to the provider that had just failed. Only the 429 path excluded it.

Full evidence: `.agents/tasks/glm-ttfb-trace/investigation.md` §2-§4.

## Decision

1. **DD1** The buffered HTTP dispatch sends `stream: true` and reassembles the SSE internally. This is controlled per provider by `buffered_upstream_streaming` (default on, off for `bedrock`). Bedrock and Codex keep their own dispatch.
2. **DD2** `n > 1`, `logprobs`/`top_logprobs` and audio output stay on `stream: false`, because reassembly only rebuilds `choices[0]`.
3. **DD3** When streaming upstream, TTFB covers the headers plus the first body chunk. Idle gaps use `streaming.chunk_timeout_seconds` (`ChunkTimeout`, 504), and `total_timeout` caps the try. With `stream: false`, the previous semantics apply unchanged.
4. **DD4** The gateway inserts `stream_options.include_usage` only when it is absent. A 400 that rejects `stream`/`stream_options` is retried once with `stream: false`, and a warning suggests the opt-out.
5. **DD5** The parse order stays JSON first, then SSE. SSE detection skips comment, `event:`, `id:` and `retry:` lines. Mid-stream error frames follow the error-in-200 rules. A stream with neither `[DONE]` nor a `finish_reason` is a retryable truncation.
6. **DD6** Reassembly keeps the full `usage` object, including cached and reasoning token details.
7. **DD7** A per-request attempt ledger on `ActiveRequestHandle` replaces the `gateway_failed_attempts` response-extras channel. Every failed try is recorded with `duration_ms` and `error_class`. Neither field appears in client JSON.
8. **DD8** `error_class` is a content-free, stable snake_case value, always stored in `requests.error_class`.
9. **DD9** A request dropped before its outcome was logged writes one row: 499 `client_disconnect`, or 504 `gateway_deadline` near the global deadline. Its ledger is drained into `#attempt` rows.
10. **DD10** Timeout-class errors fail over without a same-provider retry: `TtfbTimeout`, `TotalTimeout`, `ChunkTimeout`, and upstream 504/524 (`is_timeout_class`). 500, 502, network and body-read errors keep `max_retries_per_provider`. There is no config knob.
11. **DD11** A pass-through TTFB timeout, send error or 5xx records a ledger attempt and a breaker failure, then falls back to the buffered path with that `provider:model` excluded. If no other entry is available, the failure is returned to the client instead of re-sending to the same provider. A 4xx keeps the previous fallback. Every non-success pass-through response (429 and 4xx included) is recorded once as an `upstream_http_<code>` ledger attempt.
12. **DD12** Group members that selection drops because their breaker is open are recorded as `circuit_open_skip` once per request. Entries the caller excluded (their failed try is already recorded) are not.

The admin panel exposes `buffered_upstream_streaming` ("Stream upstream for buffered requests"), as AGENTS.md requires.

## Consequences

- Long generations survive upstream non-streaming edge limits, because bytes flow from the first token. TTFB now measures the real first byte, so a generation that keeps streaming runs up to `total_timeout`.
- A timeout costs one try per provider instead of `1 + max_retries_per_provider`, which roughly halves time-to-failover and stops duplicate provider-side generations. A provider that only timed out transiently is not retried within the same request. The next request can still use it unless its breaker opened.
- In a single-provider group, a pass-through TTFB/send/5xx failure now reaches the client instead of being re-sent to the same provider.
- Every failed try, skip and abandoned request is visible in the request log, whatever the `response_body_logging` setting.
- While Codex Search is enabled, the client still receives buffered content only at the end of the completion. That is pending decision D1.
- NIM's request allowlist strips `stream_options`, so NIM streams without `include_usage` and the gateway estimates its usage.

Guard tests (named in AGENTS.md): `buffered_sse_reader_allows_generation_longer_than_ttfb_when_bytes_flow`, `buffered_dispatch_requests_upstream_stream_and_reassembles_sse`, `timeout_class_errors_fail_over_without_same_provider_retry`, `abandoned_stream_writes_client_disconnect_outcome_row`. Related tests: `upstream_504_fails_over_without_same_provider_retry`, `upstream_500_still_retries_same_provider`, `streaming_ttfb_timeout_fallback_excludes_failed_provider`.

## Alternatives rejected

- **Raise `ttfb_timeout_seconds`.** It does not help against Electron Hub's own ~100 s 504, which arrived before the gateway's 120 s TTFB. It also delays detection of providers that are really dead.
- **Keep `stream: false` for buffered dispatch.** Any provider edge with a non-streaming timeout caps generation length, and header-based TTFB keeps capping whole generations.
- **A config knob for the timeout retry policy.** A timeout on a long deterministic generation repeats almost every time, and retrying duplicates provider-side work and billing. No valid use case justified the extra option.

## Pending (not decided here)

- **D1** Codex Search vs live streaming: a per-group opt-out to pass-through without gateway search, a longer client timeout, or stream-time search interception.
- **D2** Loop detection escalates on client re-sends of turns that failed in the gateway. Exempting them is a behavior change.
- **D3** The client's ~300 s abort is a client setting. 600 s or more is suggested for GLM.
