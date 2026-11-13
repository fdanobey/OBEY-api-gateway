# GLM / Electron Hub reliability: root-cause investigation (read-only)

Scope: model group `GLM` (primary Electron Hub `glm-5.3:dev`) on the live instance
`http://192.168.1.110:8080`, plus the code path in `crates/ai-gateway` at HEAD `b2ed659`
(the per-attempt logging from the previous task is already committed in `0168024`; the tree
was clean apart from untracked `.agents/` and `wiki-local/`). No code or live config was
changed. One synthetic probe was sent (details in §2.1).

All times are UTC (`Z`) unless marked MST. Electron Hub's dashboard shows **MST (UTC-7)
completion times** (proven in §2.4).

---

## 1. Verdict

There are two independent failures stacked on each other, plus a client-side ceiling.

**Provider side (Electron Hub):** for non-streaming `chat.completions` on `glm-5.3:dev`
with ~135-155K-token prompts, EH returns **HTTP 504 with body `error code: 504`
about 100-103 s after the request is sent** (four traced instances, nine logged attempt
rows). EH's backend keeps generating, so its dashboard records the same requests as
`Success` with full output-token counts. Through 2026-10-04 ~18:40Z, EH completed
non-streaming requests of this kind that took 125-261 s. Since ~2026-10-04 21:00Z no
single-attempt success has gone past ~120 s. That looks like a provider-side change, but the
exact cause on EH's side is UNVERIFIED.

**Router side:** with `codex_search.enabled: true` and a valid OpenAI OAuth session,
**every streaming client request is forced onto the buffered path and sent upstream with
`stream: false`** (`router.rs:8738-8748` → `route_request` → `router.rs:4270`). This
gate has been in place since Aug 2026 and is not new today. It is what turns EH's ~100 s
non-streaming limit into user-visible failures. The router then makes it worse in four ways:
- It retries the same provider after a 504 or TTFB timeout. That costs another ~100 s and
  produces a second EH generation.
- Its 120 s TTFB is measured at response headers. On a `stream:false` request that is a cap on
  the whole generation, which is why Nano-GPT and NIM fail the same way.
- It writes **no log row** for requests the client abandons.
- It hides same-provider retry failures and breaker skips.

**Client side:** the client aborts each request at about **300 s** and re-sends the same
turn 4-6 s later. Long turns never fit in 300 s once the router has spent ~200 s on two
doomed EH attempts and then 120 s+ on the next provider.

Causal chain, observed live (traces `3fe7ca6a`, `9589b2ea`, `07f921e3`):
client `stream:true` → buffered, `stream:false` to EH → EH 504 at ~100 s (EH keeps
generating and logs Success) → router retries EH ~1 s later → second 504 at ~100 s
(second EH generation) → breaker failure → failover to Nano-GPT/NIM `stream:false` → router TTFB
120 s → client aborts at ~300 s → **nothing logged** → client re-sends the same turn →
loop detection escalates to `throttle`, adding 2 s per request.

---

## 2. Empirical evidence (live instance, read-only GETs + 1 probe)

### 2.1 Probe: the client sees buffered replay, not streaming (CONFIRMED)
At 13:00:29.563Z (06:00:29 AM MST) I sent one probe: `model: glm-5.3:dev`, `stream: true`,
prompt "Count from 1 to 60…", trace `dd4b6315-801f-473c-94ec-6210b7c219f6`.
It will show up on EH's dashboard as a small glm-5.3:dev request around 06:00:35 AM MST.
- Headers arrived at 112 ms. The synthetic role "early event" arrived at 122 ms.
- Reasoning, content and `[DONE]` all arrived at **5,843 ms in the same millisecond**, as
  5 SSE events in total. A pass-through relay would have spread dozens of chunks across the
  generation. Router row: 13:00:36Z, EH glm-5.3:dev, 5,828 ms.

### 2.2 EH 504 timing (CONFIRMED, measured by passively polling `/dashboard/metrics` `active_requests_list` at 1-1.5 s resolution)
| trace | EH try | upstream send ≈ | 504 seen ≈ | elapsed |
|---|---|---|---|---|
| 3fe7ca6a | 1 | 04:25:07-08Z | 04:26:46-47Z | ~99-100 s |
| 9589b2ea | 2 (retry) | 05:07:13Z | 05:08:55Z | ~102 s |
| 07f921e3 | 1 | 05:15:39Z | 05:17:20Z | ~101 s |
| 07f921e3 | 2 (retry) | 05:17:22Z | 05:19:05Z | ~103 s |
| 999a185d | 1+2 | 04:15:02Z | both by 04:18:45Z | ~110 s average (coarse) |

`last_error` for each was `Provider error: Electron Hub - HTTP 504: error code: 504`.
That body is the plain-text 504 format of an edge proxy, not a JSON API error.
The router's own TTFB of 120 s never fired for EH; EH answered first.
Logged `#attempt` 504 rows for EH glm-5.3:dev (status 504 = real upstream HTTP 504): 06:37:01, 07:03:02,
07:07:26, 07:44:03, 07:48:26, 07:54:16, 08:00:34, 08:05:29, 08:09:33Z. Each of those requests
was then served by Nano-GPT after 216-296 s in total.

### 2.3 Nano-GPT/NIM fail through the router's own TTFB on stream:false (CONFIRMED)
`f4b993da`: Nano-GPT try 1 at 04:20:05.0Z → `Request timeout after 120s waiting for first
byte` at 04:22:04.4Z → retry → second TTFB timeout → NIM at 04:24:08.8Z. `46f956f7` shows the
same pattern from 05:10:36.7Z to 05:12:36.2Z to 05:14:40.7Z.

### 2.4 EH dashboard timestamps are completion times in MST (CONFIRMED)
Exact matches between the router's completion-time rows and the EH dashboard pasted by the user:
- EH 01:19:59 PM ↔ router 20:20:00Z (59.5 s, 138,368 cache-read ↔ "138.4K cached", 39 out)
- EH 01:30:37 PM ↔ router 20:30:37Z (90.5 s, 143,872 ↔ "143.9K cached", 1,423 out)
- EH flash 01:13:27 PM ↔ router 20:13:28Z; EH flash 12:11:25 PM ↔ router 19:11:26Z

In the EH window 11:59:59 AM-01:33:30 PM MST (18:59:59-20:33:30Z), EH lists 37 Success
entries (33 `glm-5.3:dev` + 4 flash). **The router received only 2 of the 33 dev
completions**, at 20:20:00Z and 20:30:37Z. The other ~31 were generated by EH after the router
had received a 504 for them, or after the client had given up.

### 2.5 Duplicate generations (CONSISTENT; per-pair attribution UNVERIFIED)
The same prompt was generated repeatedly. Input 136,202 tokens appears 10× between 12:37:14
and 01:14:47 PM MST, 136,200 another 3×. 144,356 appears 4× (01:25:36, 01:27:04, 01:30:37,
01:30:52 PM). 141,118 4×, 142,205 4×. Each client request produces up to two EH
generations: the try plus the same-provider retry after the 504. Each client re-send after a
300 s abort produces another. Because EH logs completion time, a pair like 12:04:37/12:04:48
is two generations that started ~101 s apart and finished ~11 s apart. That fits EH
throughput of roughly 35-65 tok/s (the router saw ~60-67 tok/s on its own successes, for
example 6,718 reasoning tokens in 105 s).

### 2.6 Client aborts at ~300 s and nothing is logged (CONFIRMED)
| trace | handler start | disappeared from registry | lifetime | next request |
|---|---|---|---|---|
| 999a185d | 04:14:53.9Z | ≤04:19:52.9Z | ~298 s | +4.2 s |
| f4b993da | 04:19:57.1Z | ≤04:24:55.6Z | ~298 s | +4.8 s |
| 9589b2ea | 05:05:26.4Z | ≤05:10:24.5Z | ~298 s | +5.1 s |
| 46f956f7 | 05:10:29.6Z | ≤05:15:27.3Z | ~298 s | +5.5 s |

None of them has any row in `/dashboard/logs`. Add the 2 s loop-detection throttle that runs
before the handler starts and the client-side lifetime is ≈300 s.
The router has no 300 s timer on this path:
- `request_timeout_middleware` only bounds time-to-headers, and SSE returns immediately. The
  effective deadline is `max(300, 800+60) = 860 s` (`runtime_limits.rs:104-118, 142-170`).
- `chunk_timeout_seconds` (300) is only used by the pass-through relay (`handlers.rs:4759`).

So the 300 s abort is a client setting (UNVERIFIED which client). Counters since the
~01:10Z restart: `request_count` 530 against ≈510 outcome rows, so **≈19 requests left no trace**.

### 2.7 Hidden same-provider retries (CONFIRMED by counters)
Since the restart, EH has `retry_count` = 49 same-provider retries and `failed_requests` = 30
provider-level failures. Only 10 EH failure `#attempt` rows exist (9×504, 1×502).
- ≈19 first tries failed and were rescued by the retry. They are invisible: those requests show
  as single 200 rows of 129-194 s (05:28:30Z 171 s, 08:13:15Z 194 s, …).
- ≈20 provider failures belong to abandoned requests (§2.6).

### 2.8 Weekend vs today (CONFIRMED pattern; cause UNVERIFIED)
Max single-row EH glm-5.3:dev success duration per 3 h bucket, all on the same buffered
`stream:false` path. `reasoning_compat_actions` is set on 684/739 rows, and only the buffered
success path sets it.
- 2026-10-03 18h → 2026-10-04 18h: max 165-261 s per bucket, 49 rows >100 s.
- 2026-10-04 21h → 2026-10-05 21h: max ≤ 120 s, **0 rows > 125 s**.

There was no router change in that window that affects timeouts or buffering. `b496311`
(2026-10-04 21:21Z) only relaxes the Smart Routing buffering gate. The Codex Search gate
dates from `b2d69e1`/`60d84c3` (Aug 2026). I have no history of the live config, so a
config change on the user's side is UNVERIFIED but unlikely.

### 2.9 Other live facts
- Rate limiter: 2 EH exhaustions since restart. Both came from a burst at 06:26-06:27Z, when
  11 EH requests completed within one minute, several of them in overlapping pairs. The 8/min
  limiter worked as designed; the client was not strictly sequential at that moment.
- Loop detection: the user's session `ses_ef3c…` is at enforcement `throttle`
  (dominant signal `content_similarity`, 20 requests). Every request sleeps 2 s
  (`loop_detection/middleware.rs:313-317`) because the client keeps re-sending the same turn.
- `logging.response_body_logging: false`, so `#attempt` rows lose their error string
  (`response_body` null) and `duration_ms` is 0 by design. The rows cannot say *why* an
  attempt failed or *how long* it took.
- Pre-dispatch latency from handler start to the first upstream send was 6.7-7.9 s for
  ~150K-token requests (f4b993da 7.9, 3fe7ca6a 7.7, 46f956f7 7.1, 07f921e3 6.7). Logged
  compression accounts for ~1.5-1.7 s. The rest is unattributed (§3d).
- Prometheus is `null`, so `/metrics` returns 404 and the previous task's attempt-failure
  counter is not scrapeable.

---

## 3. Hypotheses (a-h)

### a. Is the upstream request `stream:true` or `stream:false`?
- **Client `stream:true` → upstream `stream:false`: CONFIRMED** by code, live config and the probe.
  `chat_completions_stream` (`handlers.rs:2237`) has early events enabled and calls
  `route_request_streaming` (`handlers.rs:2492`). `route_request_streaming_excluding` hits
  `if self.codex_search_ready().await.is_some()` → `StreamingResponse::Buffered(self.route_request(..))`
  (`router.rs:8738-8748`). `codex_search_ready` (`router.rs:3861-3878`) returns `Some` because
  `codex_search.enabled: Some(true)` and `effective_enabled` is `self.enabled.unwrap_or(has_codex_provider)`
  (`codex/search/config.rs:47-50`). The OAuth status is `authenticated` anyway.
  `dispatch_attempts_under_permit` then sets `outgoing.stream = false` (`router.rs:4262-4270`).
- **Client `stream:false` → upstream `stream:false`: CONFIRMED** (same `router.rs:4270`).
- Other features that would also force buffering, checked:
  - Guardrail post-call: GLM has no binding; only Deepseek/KIMI do (`handlers.rs:2397-2420`). Not active.
  - Structured output: only `json_schema` requests (`handlers.rs:669-672`). Not active for normal turns.
  - Nano-GPT/Kimi transformation (`router.rs:8262-8275`): always buffered for Nano-GPT.
  - Smart Routing: does not force buffering because reliability is enabled (`router.rs:485-490`).
  - Compression, memory, reasoning_compat: never force buffering.
- **The hypothesis that 120 s TTFB on a buffered upstream kills any completion over 120 s:
  CONFIRMED for Nano-GPT/NIM (§2.3).** For EH the cap is EH's own ~100 s 504 (§2.2), which
  arrives before the router's TTFB.
- Side effect, CONFIRMED by code: `maybe_inject_search_tools` (`router.rs:5668`, `3884-3887`)
  adds the `codex_search`/`codex_web` tools to every GLM request.

### b. What TTFB measures
- Buffered (all live GLM traffic): `tokio::time::timeout(ttfb, req.json(&outgoing).send())`
  (`router.rs:4577-4591`). `send()` resolves at **response headers**. For `stream:false`,
  headers usually come only after the whole completion, so "TTFB" really caps the whole
  generation. The body is then read with the remaining total (`router.rs:4614-4642`).
  **CONFIRMED.**
- Pass-through: TTFB also applies to headers only (`router.rs:9000-9018`), and the relay then
  applies `chunk_timeout_seconds` between chunks (`handlers.rs:4759-4875`). Not exercised live.
- Whether EH sends SSE headers and first chunks immediately with `stream:true`:
  **UNVERIFIED**. It cannot be tested through the router while the Codex gate is active, and I
  did not call EH directly (no key handling). Verify after deploy (§7).

### c. Await points that can hold a request before the upstream send
| Point | Finding | Verdict |
|---|---|---|
| Provider slot (`max_connections` 8, `provider_slot_wait` = 800 s, `router.rs:210-213, 2948-3015`) | RAII permit; the std `Mutex` is never held across await (`router.rs:120-196`). Permits drop with the future on disconnect or timeout. Live: the first send happened within ~7 s of start, 1 in-flight request | **REFUTED** as a cause |
| Rate limiter 8/min (`router.rs:5650-5662`; streaming `8858-8869`) | 1 token per provider entry per request; same-provider retries do not consume. Smart Routing (OpenRouter) and memory extraction (OpenRouter/Ollama) do not use EH tokens. 2 exhaustions, from a real 11-requests-per-minute burst | **REFUTED** (works as designed) |
| Config `RwLock` | Guards dropped before network awaits (`router.rs:4262`, `5560`). Minor: in `dispatch_attempts_under_permit` the OAuth `get_access_token().await` (~`4172-4180`) and Bedrock client setup run while the read guard is still held. Not on the EH path | **REFUTED** for EH (minor latent issue noted) |
| Circuit breaker | No waiting; open entries are skipped (`router.rs:1638-1693`, `5598-5608`). Backoff [5,10,150,300] means EH is skipped for up to 5 min after a failed half-open probe | **REFUTED** as a hold; it does shape routing |
| Retry backoff | 1 s ±20% before the same-provider retry (`router.rs:4502-4555`) | CONFIRMED, small |
| Loop-detection throttle | 2 s `sleep` before the handler (`loop_detection/middleware.rs:313-317`); session live at `throttle` | **CONFIRMED** router-side delay, triggered by client re-sends |
| Memory pre-processing | Full-request token count `TokenCounter::new().count_request` (`handlers.rs:892`) plus Qdrant/Ollama retrieval bounded at 5 s | Bounded; magnitude **UNVERIFIED** |
| Smart Routing (jev via OpenRouter, 1 s timeout × 2 attempts, cached by fingerprint) | Runs in the streaming selection and again inside the buffered `route_request` (second call should hit the cache) | Bounded ≤ ~2 s; **UNVERIFIED** split |

### d. Blocking-the-runtime audit (multi-thread `#[tokio::main]`, `main.rs:109`)
| Work | Where | At single-request volume |
|---|---|---|
| Lite compression of ~260K tokens inline, per provider tried | `prepare_compressed_request_with_stats` (`router.rs:5655`; streaming `8752`) | ~1.5-1.7 s per row (logged). Adds latency, does not block other tasks. **CONFIRMED** |
| Repeated full-request tokenization (post-truncation prep, memory `count_request`, compression) | `handlers.rs:2262`, `892`; compression engine | Probably most of the unattributed ~5 s. **UNVERIFIED**, needs stage timing |
| serde of ~1 MB bodies (loop-detection re-serialization `middleware.rs:320`, handler deserialization, `log_request` `serde_json::to_string(request)` `handlers.rs:331`, many `request.clone()`) | various | ms-level. Negligible |
| SQLite writes | `logger.log` → `try_send` to a dedicated thread (`logger/mod.rs:360-399`) | Non-blocking. **REFUTED** |
| SQLite queries from the dashboard | `logger.query` blocks on `recv()` on a tokio worker (`logger/mod.rs:518-528`, `dashboard/mod.rs:522, 587`) | Not on the request path. Low impact |
| System One discovery `block_in_place` + `futures::executor::block_on` | `smart_routing/systemone/classifier.rs:74-77, 359` | Bounded by 1 s, refreshes every 600 s. Low |
| `std::sync::Mutex` across await, `std::thread::sleep` | none on the request path (sleep only in tests) | **REFUTED** |
| Regex guardrails | not bound for GLM | n/a |

### e. Client disconnect and router-side abandonment
- The upstream future is dropped, so reqwest closes the connection. This happens on client
  disconnect (the `async_stream` is dropped) and also on the router's own `tokio::time::timeout`.
  Permits are released (RAII). The active-request entry is removed by
  `RequestCompleteGuard::drop` (`handlers.rs:1306-1322`), which only emits a debug trace.
- **No SQLite row is written** for an abandoned request: no outcome row and no attempt rows.
  **CONFIRMED** live (§2.6) and in code: the only DB write is `log_request` (`handlers.rs:317`)
  or `log_failed_attempts` (`391`) at terminal points.
- EH does not cancel generation when the caller goes away or when its own edge returns 504.
  It completes and logs Success. **CONFIRMED** by correlation (§2.4). This is provider-side.

### f. Retry/failover semantics on TTFB timeout and 504
- TTFB timeout → `continue`, which retries the **same provider** after backoff
  (`router.rs:4577-4591`). HTTP 504 is a "Retryable error", also retried on the same provider
  (`router.rs:5118-5126`); only 503 fails over immediately (`5109-5117`).
  `max_retries_per_provider: 1` means 2 EH generations per request. **CONFIRMED** live (attempt 2, phase `retry`).
- After the retries, the failover `Err` arm records **one** breaker failure per request
  (`router.rs:6141-6151`). Three requests open the breaker. **CONFIRMED.**
- Is this desirable for slow-start reasoning models? **No.** A timeout on a long deterministic
  generation repeats with near certainty. The retry doubles latency (2×100 s) and EH's
  load/billing, and it creates the duplicate pairs on EH's dashboard.
- Latent bug: the pass-through TTFB timeout, send error and non-success fallbacks call
  `route_request` **without excluding** the provider that just failed and without recording an
  attempt or breaker failure (`router.rs:9000-9018`, `9156-9162`). That re-sends to the same
  provider with `stream:false`. Only the 429 path excludes (`9064-9073`). **CONFIRMED** by code;
  not active live because the Codex gate returns earlier.

### g. Why the per-attempt logging produced zero (now 12) `#attempt` rows
**CONFIRMED** causes:
1. All Oct 5 rows predate the deploy (~01:10Z Oct 6).
2. Abandoned requests never reach a terminal logging point (§2.6, ≈19 requests).
3. Same-provider retry failures inside `dispatch_attempts_under_permit` are never turned into
   `ProviderAttempt`s. Only the per-provider final error is (`router.rs:6243-6249`). ≈19 hidden (§2.7).
4. Breaker-open providers are silently pre-filtered by `select_provider_order`
   (`router.rs:1638-1693`) before `route_with_failover_for_group` could record its
   "Circuit breaker open" attempt.
5. The all-providers-failed path logs only the aggregated outcome row, not per-attempt rows
   (`handlers.rs:2226-2232`, `2984-2994`).
6. Pass-through fallbacks record no attempt (f).
7. Where rows do exist, they carry `duration_ms = 0` and `response_body` null (body logging
   off), and a router TTFB timeout gets fallback status 502 (`handlers.rs:363, 401`). A
   "504 from EH" and a "router gave up at 120 s" cannot be told apart.

### h. Empirical confirmation summary
- Probe (§2.1): buffered replay. CONFIRMED.
- EH 504 at ~100 s (§2.2). CONFIRMED.
- Router TTFB at 120 s on Nano-GPT (§2.3). CONFIRMED.
- Client abort at ~300 s with no row (§2.6). CONFIRMED.
- Time-to-first-SSE-chunk *from EH* under `stream:true`: UNVERIFIED (see b).

---

## 4. Router-side vs provider-side

| # | Finding | Side |
|---|---|---|
| P1 | EH returns HTTP 504 (`error code: 504`) ~100 s into non-streaming glm-5.3:dev requests while the generation completes and is logged Success | **Provider (EH)** |
| P2 | Long non-streaming requests (125-261 s) succeeded through 2026-10-04 ~18:40Z and stopped succeeding from ~21:00Z | **Provider (EH)**, cause UNVERIFIED |
| P3 | EH does not cancel generations when the caller disconnects or its edge 504s | **Provider (EH)** |
| R1 | Codex Search forces every streaming request to buffered with `stream:false` upstream; the client gets no tokens until completion | **Router** (design gate) |
| R2 | TTFB measured at headers on `stream:false` effectively caps the whole generation at 120 s (Nano-GPT/NIM) | **Router** |
| R3 | Same-provider retry on 504/TTFB timeout duplicates generation and adds ~100-120 s | **Router** |
| R4 | Abandoned requests, hidden retries, breaker skips, durations and reasons are not logged | **Router** (observability) |
| R5 | Pass-through fallbacks re-send to the provider that just failed, without exclusion | **Router** (latent) |
| R6 | ~7-8 s pre-dispatch plus a 2 s loop-detection throttle | **Router** (delay, not failure) |
| C1 | Client aborts at ~300 s and re-sends the same turn | **Client** |

---

## 5. Provider-facing summary (for the Electron Hub ticket — does not mention our gateway)

> **Subject:** glm-5.3:dev non-streaming requests return HTTP 504 after ~100 s, but your dashboard shows them as Success
>
> Since about 2026-10-04 21:00 UTC (Oct 4, 2 PM MST), our non-streaming `POST /v1/chat/completions` calls to `glm-5.3:dev` (prompts ~135K-155K tokens, mostly cached) frequently fail with **HTTP 504 and the plain-text body `error code: 504`, consistently 99-103 seconds after the request is sent**. Measured examples (UTC): sent 05:15:39, 504 at 05:17:20; sent 05:17:22, 504 at 05:19:05; sent 05:07:13, 504 at 05:08:55 (Oct 6). Between 06:37 and 08:10 UTC on Oct 6 we logged nine more 504s from your endpoint.
>
> Your usage dashboard lists these same requests as **Success** with full output tokens. On Oct 5 between 11:59:59 AM and 1:33:30 PM MST, your dashboard shows 33 successful `glm-5.3:dev` completions on our key. Our application received only 2 of them (the ones completing at 1:19:59 PM and 1:30:37 PM). Every other request ended for us in a 504, or we stopped waiting. Several prompts show up repeatedly on your side (for example input 136,202 tokens ten times between 12:37 and 1:14 PM). Those are our retries after receiving your 504s; each one was generated to completion on your side.
>
> Before this, on Oct 3-4, non-streaming requests of the same kind that took 125-261 seconds completed normally.
>
> Questions:
> 1. Is there a ~100 s timeout on non-streaming responses (edge or load balancer)? Was it introduced or changed around Oct 4?
> 2. Are streaming (`stream: true`) requests exempt from it?
> 3. When your edge returns 504, or the caller disconnects, is the backend generation cancelled? Are those requests billed or counted against limits?
> 4. Can the dashboard mark requests whose response was never delivered (504/disconnect) instead of Success?
>
> Account: Master Key (Chandler/Mesa, AZ). Times: your dashboard uses MST (UTC-7) completion times; ours are UTC.

---

## 6. Proposed fix list (router-side; implementation plan in `plan.md`)

| # | File(s) | Change | Why | Regression test |
|---|---|---|---|---|
| F1 | `router/router.rs` (`dispatch_attempts_under_permit` ~4255-4860), `config/mod.rs` (`Provider`), admin UI | **Buffered dispatch streams upstream.** Send `stream:true` (+ `stream_options.include_usage`) for HTTP providers whenever `Provider::effective_buffered_upstream_streaming()` is true (new per-provider option, default true). Read the body incrementally: TTFB covers headers **and** the first body chunk, an idle gap uses `streaming.chunk_timeout_seconds`, and the remaining total still applies. Then reuse the existing JSON→SSE parse chain (`reassemble_sse_response`). Keep `stream:false` for `n>1`, `logprobs`, audio, Bedrock/Codex, and the opt-out. On a 400 that mentions `stream_options`/`stream`, retry once with `stream:false` | Bytes flow from the first token, so the ~100 s non-streaming edge limit and the 120 s header-TTFB cap no longer kill long generations. TTFB then measures the real first byte. Client-visible output is unchanged | `buffered_dispatch_requests_upstream_stream_and_reassembles_sse`, `buffered_sse_reader_allows_generation_longer_than_ttfb_when_bytes_flow`, `buffered_sse_reader_ttfb_covers_first_body_chunk`, `buffered_sse_reader_idle_gap_is_chunk_timeout`, `buffered_upstream_streaming_disabled_sends_stream_false`, `buffered_dispatch_keeps_stream_false_for_multi_choice_or_logprobs`, `buffered_dispatch_falls_back_to_stream_false_when_stream_options_rejected` |
| F2 | `router/router.rs` `reassemble_sse_response` (5169-5415) | Keep the full `usage` object (prompt/completion token details such as `cached_tokens`, `reasoning_tokens`). Treat an SSE body with neither `[DONE]` nor any `finish_reason` as a truncated stream (retryable error). Detect SSE bodies that start with `:` comments or `event:` lines | Prevents F1 from losing cache/reasoning telemetry and from accepting a silently truncated stream | `reassemble_sse_response_preserves_usage_details`, `buffered_sse_without_done_or_finish_reason_is_truncation_error`, `buffered_sse_with_leading_comment_lines_is_reassembled` |
| F3 | `error/mod.rs`, `gateway/handlers.rs` | New `GatewayError::ChunkTimeout(u64)` → 504, classified like the relay's inter-chunk timeout | Distinct, typed idle-timeout error for F1 | covered by the F1 reader test |
| F4 | `active_requests.rs`, `router/router.rs`, `gateway/handlers.rs`, `logger/mod.rs`, `error/mod.rs` | **Per-request attempt ledger on `ActiveRequestHandle`.** Record every failed upstream try (including same-provider retries) with `duration_ms` and a content-free `error_class`, plus skips (breaker/cooldown/rate-limit/budget) and content failovers. The handler drains the ledger at every terminal point, including errors. Replaces the `gateway_failed_attempts` response-extras channel. New additive `error_class` column (`ensure_column` migration) | Makes hidden retries, breaker skips and failure reasons visible regardless of `response_body_logging` | `failed_attempt_ledger_records_and_drains`, `same_provider_retry_failure_is_logged_as_attempt_row`, `all_providers_failed_logs_attempt_rows`, `error_class_column_migrates_and_round_trips` |
| F5 | `gateway/handlers.rs` (`RequestCompleteGuard` 1268-1323) | **Abandoned-request outcome row.** If the request future is dropped before an outcome was logged, write a content-free row: status 499 with `error_class: client_disconnect`, or 504 `gateway_deadline` if elapsed ≥ the global deadline. Include provider/model/attempt/phase from the handle and drain the ledger | ≈19 requests vanished today with no trace | `abandoned_stream_writes_client_disconnect_outcome_row`, `completed_stream_does_not_write_abandon_row` |
| F6 | `router/router.rs` dispatch retry loop | **Timeout-class errors fail over without a same-provider retry**: `TtfbTimeout`, `TotalTimeout`, `ChunkTimeout`, upstream HTTP 504/524. 500/502/network keep `max_retries_per_provider` | Removes the duplicate ~100 s EH generation and halves time-to-failover | `timeout_class_errors_fail_over_without_same_provider_retry`, `upstream_504_fails_over_without_same_provider_retry`, `upstream_500_still_retries_same_provider` |
| F7 | `router/router.rs` `route_request_streaming_excluding` (9000-9018, 9156-9162) | On pass-through TTFB timeout, send error or 5xx: record the attempt in the ledger, record a breaker failure, and fall back via `route_request_with_exclusions` excluding the failed `provider:model`. 4xx keeps the current fallback | Stops a re-send to the provider that just failed, in `stream:false` mode | `streaming_ttfb_timeout_fallback_excludes_failed_provider` |
| F8 | `router/router.rs` ("Calling provider" log ~4285), `gateway/handlers.rs` memory preprocessing | Add `pre_dispatch_ms` (now − `started_at_ms`) to the existing info log, and log memory preprocessing time when > 1 s | Lets the ~5 s unattributed pre-dispatch delay be attributed after deploy | log-only; no behavior test |
| F9 | `docs/adr/0001-buffered-dispatch-streams-upstream.md`, `AGENTS.md`, `wiki/Streaming.md`, `wiki/Providers.md` | ADR plus a Non-Obvious-Patterns note naming the guard tests. Document that Codex Search forces buffer-and-replay, and document the new provider option | Prevents reintroduction | n/a (docs) |

## 7. Pending user decisions (NOT in the implementation plan)
- **D1 Codex Search vs live streaming.** Even after F1, the client receives tokens only when
  the full completion is assembled, because the Codex Search gate stays in place. Turns longer
  than ~290 s will still hit the client's 300 s abort. Options:
  1. A per-model-group opt-out that lets streaming requests use pass-through. Those requests
     would get no gateway web search.
  2. Raise the client timeout.
  3. Stream-time search interception (large feature).
  This changes behavior for the model, so it needs the user's call.
- **D2 Loop detection on re-sent failed turns.** Today the client's identical re-sends after
  router failures escalate the session (currently `throttle`; next levels `inject` and
  `hardstop`). `inject` would edit the system prompt and break the provider's prompt cache.
  Exempting re-sends whose previous request never completed is a behavior change.
- **D3 Client timeout.** The ~300 s abort is a client setting. Consider 600 s or more for GLM.
- **D4** `codex_search_ready` returns `Some` when `enabled: true` even without an OAuth token
  (doc/code mismatch, `router.rs:3854-3878` vs `codex/search/config.rs:47-50`). No live impact
  because the token is valid; flagged only.

## 8. Items to verify during implementation / after deploy
1. EH (and Nano-GPT/NIM) accept `stream:true` with `stream_options.include_usage` and send
   the first SSE bytes promptly. Verify after deploy: `#attempt` rows with `error_class` like
   `upstream_http_504` should disappear for EH, and EH successes longer than 100 s should appear
   **without** an attempt row. If a provider rejects `stream_options`, F1's one-shot fallback
   and the per-provider opt-out cover it.
2. EH's 504 is specific to non-streaming. If EH also 504s on streams at ~100 s, F1 will not
   help; the ledger rows will show it as `upstream_http_504` with `duration_ms` ≈ 100000.
3. Attribution of the ~5 s unexplained pre-dispatch delay, using F8 logs.
4. That reassembled responses keep `cache_read_tokens`/`reasoning_tokens` populated in
   dashboard rows. Covered by the F2 test locally; confirm live.

Temporary files used for this investigation (under `%TEMP%`, `glm_*`) were deleted.
