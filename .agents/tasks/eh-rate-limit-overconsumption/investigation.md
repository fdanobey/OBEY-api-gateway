# Electron Hub rate-limit skips: investigation

Read-only. Code at HEAD (clean tree). The live instance is http://192.168.1.110:8080, queried with GET `/admin/config`, `/dashboard/logs`, `/dashboard/metrics` and `/dashboard/errors`. No key values were printed. All times are UTC.

## 1. Verdict

**Root cause: router-side configuration plus router design. There is no token over-consumption bug, and Electron Hub is not at fault.**

- **The live limit is 2 per minute, not 8.** `/admin/config` reports Electron Hub with `rate_limit_per_minute: 2` and `max_connections: 2`. The Oct 6 investigation recorded 8 and 8 (`.agents/tasks/glm-ttfb-trace/investigation.md` §2.9 and table row "Rate limiter 8/min"), so the value was changed at some point after that. I can't see config history and can't tell who changed it or when.
- **The bucket refills too slowly for the user's traffic.** A 2/min bucket holds 2 tokens and gets 1 back every 30 s. Today the median gap between Electron Hub dispatch starts was 31 s and the p10 gap was 12.9 s; EH request duration p50 was 17.3 s and p10 was 9.2 s. Sequential short requests therefore drain the bucket within 2-3 requests, and every request after that is skipped locally and fails over.
- **The code charges one token per Electron Hub candidate it actually dispatches.** No path charges twice for the same upstream request (§3).
- **A replay of today's logs fits 2/min and rules out 8/min.** I replayed every EH dispatch and every local skip from 00:00Z through the latest log row (18:25Z), charging 1 token per dispatch. Results:
  - A 2/min bucket predicts 85-97% of events, depending on the time window.
  - An 8/min bucket predicts **0 of 141** skips, even when charging 2 or 3 tokens per dispatch.

  The leftover misfit at 2/min fits timing error in the replay. Fallback flash dispatches really take their token about 2 minutes after the request starts, but the replay charges them at the start. A config save also refills the bucket (`clear_rate_limiters`).
- **Electron Hub's dashboard is accurate.** Its 100% success matches the router's metrics (`provider_health`: Electron Hub 142/142). Requests skipped by the local limiter are never sent to Electron Hub, so they can't appear there.

So the user's 1-3 sequential requests per minute can't exhaust an **8**/min bucket under this code. They do exhaust the **2**/min bucket that is live now, and that is the limiter working as configured.

The behavior that really hurts is router-side:

- **Skipping instead of waiting.** One token is at most 30 s away at 2/min, or 7.5 s at 8/min. Instead of waiting for it, the router fails over to NIM, which often hits its 120 s TTFB timeout and then lands back on EH flash. That costs 2-4 minutes per skip.
- **The Errors tab hides the reason.** It shows these rows with an empty "failure reason" (§6).

**Question for the user, needed before any change:** is 2/min the intended Electron Hub limit, or should it be 8? If 8 is right, restoring it in the admin panel (Providers → Electron Hub → Rate limit) removes almost all skips. That needs no code change.

## 2. Live facts

| Item | Value |
|---|---|
| Electron Hub | rpm **2**, max_connections **2**, ttfb 120 s, total 800 s, `buffered_upstream_streaming` null (on) |
| Other providers (rpm / max_connections) | Nano-GPT 10/100, Nvidia NIM 40/10, OpenAI 10/100, Openrouter 2/100, Ollama 2/2, AWS Bedrock 60/100 |
| GLM group order | EH glm-5.3:dev (p1), Nano-GPT z-ai/glm-latest (4), NIM z-ai/glm-5.3 (7), Nano-GPT z-ai/glm-5.2:thinking (10), EH glm-5.3-flash:dev (13), Nano-GPT/NIM z-ai/glm-5.3-flash (16/19), Nano-GPT gpt-oss-120b (22), OpenAI gpt-6.1-sol (25), NIM gpt-oss-20b (28), Bedrock zai.glm-5 (31), Ollama gemma4:e4b (34) |
| Codex Search | enabled, max_iterations 5 |
| Streaming | passthrough_enabled true; Codex Search forces the buffered path anyway |
| Smart Routing | enabled. Classifier is `jev` at openrouter.ai; cascade disabled; no GLM override. Does not call Electron Hub |
| Memory | auto-extract via provider `Openrouter`; embeddings via Ollama. Does not touch EH's bucket |
| Semantic cache | Ollama embeddings and Qdrant; not EH |
| Cache-aware routing | disabled |
| Retry | max_retries_per_provider 1 |
| `/dashboard/metrics` | `rate_limit_exhaustions_by_provider`: Electron Hub 37, Nano-GPT 1 (since restart); `retry_count_by_provider` empty |
| Logs since 00:00Z | 690 EH rows: 549 real dispatches and 141 local `rate_limit_skip` rows (429, 0 ms) |

Example trace from 18:19-18:25Z, all sequential:

- Six EH successes started at 18:19:49, 18:20:25, 18:21:16, 18:21:56, 18:22:10 and 18:22:27.
- **18:22:43** EH glm-5.3:dev `rate_limit_skip`. At 2/min the bucket held about 0.5 tokens at that moment, so the skip is exactly what the model predicts.
- NIM z-ai/glm-5.3 returned `upstream_http_404`.
- **18:22:48** EH glm-5.3-flash:dev `rate_limit_skip`. Same bucket, still empty.
- NIM z-ai/glm-5.3-flash hit `ttfb_timeout` (120 s), then OpenAI gpt-6.1-sol returned `upstream_http_400`.
- NIM openai/gpt-oss-20b returned 200 at **18:25:07**. The user waited 145 s, on a different model, to save a ≤30 s wait.

## 3. Every token site and every upstream HTTP call

The provider limiter is `router/rate_limiter.rs`, a token bucket:

- `RateLimiter::new` at `:49`: capacity = rpm, refills at rpm/60 per second, starts full.
- `check_available` at `:112` checks without consuming and is `#[allow(dead_code)]`.
- `consume` at `:136` checks and decrements; it returns false without decrementing when fewer than 1 token is available.
- There is one limiter per provider **name**, created in `Router::get_rate_limiter` (`router/router.rs:3031-3048`, `RateLimiter::new(rate_limit)` at `:3045`). Every Electron Hub model (glm-5.3:dev, glm-5.3-flash:dev, in any group) shares one bucket.
- `clear_rate_limiters` (`:3084`, called from `gateway/mod.rs:896` on config reload) resets buckets to full.

### 3a. Sites that consume a token (`consume()`)

| # | Site | Path | Correct? |
|---|---|---|---|
| C1 | `router.rs:6303-6321` in `route_with_failover_for_group` | Once per candidate entry, before compression and before `attempt_with_retry`. On failure it pushes a `rate_limit_skip` attempt (429) and continues. A failed consume does not decrement | Yes: 1 per dispatched entry. Skipped entries take 0 |
| C2 | `router.rs:9537-9547` in `route_request_streaming_excluding` | Pass-through streaming only, after all the buffered gates (Codex Search gate `:9415-9425`, transform/Codex/Bedrock `:9488-9498`, OAuth `:9517-9527`). On failure it does not decrement and falls into `route_request` → C1 | Yes. **Unreachable live** because Codex Search is enabled |
| C3 | `router.rs:10004-10013`, `route_fine_tuning_pass_through` | Fine-tuning proxy | Yes; not chat |
| C4 | `router.rs:10150-10158`, `send_provider_pass_through` | Embeddings, audio, images | Yes; not chat |

### 3b. Sites that check only (no consume)

- `router.rs:1857-1876` in `select_provider_order` checks only `cooldown_remaining()` and the metrics cooldown, never the bucket. Bucket exhaustion is deliberately left to C1.
- `rate_limiter.rs:112` `check_available` is used only in tests (`router.rs:16408`).
- `cooldown_remaining`, `apply_cooldown` and `clear_cooldown` (`router.rs:2326, 5438, 5728, 6664, 6850, 9764`) handle upstream 429 cooldowns, not the bucket.

### 3c. Real upstream HTTP requests that consume no token (under-count)

| # | Site | What it is |
|---|---|---|
| U1 | `router.rs:52-82` `SearchResubmitter::chat_completion` → `attempt_with_retry` | Codex Search resubmission. Each search round sends the conversation, plus the tool result, back to **the same provider**. Up to `max_iterations` (5) extra EH requests per client turn, none charged |
| U2 | `router.rs:4737` retry loop in `dispatch_attempts_under_permit` | Same-provider retries for 500/502/network errors (`max_retries_per_provider` 1) |
| U3 | `router.rs:5294-5309` | One-shot `stream:false` retry after a stream-parameter rejection |
| U4 | `router.rs:6411` → `:7311` | Reasoning-continuation nudge: one same-provider re-dispatch per request |
| U5 | `dispatch_buffered_with_context_retry` (`:4303`, `:4365`) | Context and image repair retries inside Codex/Bedrock dispatch (not EH) |
| U6 | `admin/mod.rs:1987` | Admin-initiated provider request (operator action) |

### 3d. Re-entries that consume again, legitimately

Each of these sends a new upstream request:

- Handler-level single-target re-dispatch via `route_with_failover` → C1:
  - refusal failover, `gateway/handlers.rs:1879`, `:3399`, `:4243`; needs guardrails, which are not active live
  - structured-output retries, `:2123`, `:3574`; only when the client sends `response_format`
- Pass-through fallbacks after a real upstream error, via `route_request`/`route_request_with_exclusions` → C1: `router.rs:9790`, `:9817`, `:9874`, `:9888`, plus `passthrough_fallback_excluding`. The token spent at C2 paid for the failed pass-through request; the fallback is a second real request. Unreachable live (C2 is unreachable).

### 3e. Other outbound HTTP (not EH, no EH token)

- `cache/semantic.rs:405` (Ollama embeddings)
- `gateway/handlers.rs:9887`, `:10140` (Codex/ChatGPT backend with OAuth)
- the Codex Search `SearchExecutor` (OpenAI search endpoint)
- the Smart Routing `jev` classifier (openrouter.ai)
- memory auto-extract (provider `Openrouter`)

None of these is keyed to "Electron Hub", so none can drain its bucket.

## 4. Tokens per GLM chat request on Electron Hub (live config)

| Client request | Path | EH tokens | Real EH upstream requests |
|---|---|---|---|
| `stream:true` | handler `route_request_streaming` → Codex Search gate `router.rs:9415` → `route_request` → C1 once for glm-5.3:dev → `attempt_with_retry` (upstream `stream:true`, ADR 0001) | **1** | 1 + U1 search rounds (0-5) + U2/U3/U4 |
| `stream:false` | handler `route_request` (`handlers.rs:1734`) → C1 once | **1** | Same as above |

If the chain also reaches EH glm-5.3-flash:dev and its consume succeeds, that entry takes 1 more token for 1 more real request. A skipped entry takes 0. So tokens ≤ real requests, always: **the code under-counts and never over-counts.**

## 5. Suspects from the brief

| Suspect | Finding | Verdict |
|---|---|---|
| Streaming consume (`:9537`) plus buffered consume (C1) double-charging since ADR 0001 | With Codex Search on, the streaming path returns `Buffered(route_request)` at `:9415`, **before** `:9537`. Without Codex Search, C2 then a C1 fallback happens only after a real upstream failure (two real requests), or after a failed C2 consume (no decrement) | **Refuted** |
| `check_available()` paired with `consume()` | `check_available` is dead code outside tests; no pairs | **Refuted** |
| Retries and the `stream:false` fallback | No token charged (U2, U3) | **Refuted**; under-count |
| Codex Search loop | Inner rounds call `attempt_with_retry` directly and are real EH requests, uncharged (U1) | **Refuted** as over-count; under-count |
| Smart Routing, memory, guardrails, summarization, truncation, capability probes, model lists, health checks | None targets Electron Hub in the live config, and none calls `consume` on EH's limiter | **Refuted** |
| Responses/Messages/Assistants translation (`handlers.rs:3187`, `3735`, `4026`, `8259`, `9780`, `10266`) | Each is a separate client request → one C1 per entry | **Refuted** (each logs its own outcome row) |
| Outer loop charging a token per candidate on fallback re-entry | C1 runs per dispatched candidate. Failed consumes don't decrement. Smart Routing cascade is disabled | **Refuted** |
| Other callers sharing EH's bucket | Only other EH traffic does (any model group, both EH models). That traffic appears in `/dashboard/logs?provider=Electron%20Hub` and is included in the replay | **Refuted** as hidden consumption |
| Live limit isn't what the brief assumed | `rate_limit_per_minute: 2` live; the replay fits 2/min and not 8/min | **Confirmed root cause** |

## 6. Why the Errors tab looked empty

- **The rows are there.** `GET /dashboard/errors` returns them: 25 rows, all `#attempt` rows with `error_class` set (`rate_limit_skip`, `ttfb_timeout`, `upstream_http_404/400`, `circuit_open_skip`).
- **The UI renders an empty reason for them.** `dashboard/static/index.html:804-817` `renderErrors` shows `e.error || e.response_body`. Both are null on attempt rows (content-free policy), so the reason column is blank. It also has no column for `error_class` or for attempt-vs-final. A recovered 429 local skip looks like an unexplained, possibly stale row.
- **The window is fragile.** `dashboard/mod.rs:586-599` `recent_errors` takes the newest `limit*4` (100) rows and then filters `status_code >= 400`. Once more than 100 successful rows land after an error, it disappears from the tab. This also feeds the WebSocket push at `:390`.
- I can't see what the user's browser showed. The API side is verified; the rendering explanation comes from reading the code.

## 7. Provider-side notes (separate issues; no change proposed)

- NIM z-ai/glm-5.3 and z-ai/glm-5.3-flash often hit `ttfb_timeout` at exactly 120 s (`provider_health` NIM 8/34). NIM z-ai/glm-5.3 also returned 404 today (18:15, 18:22).
- OpenAI gpt-6.1-sol returned `upstream_http_400` on this traffic, and Nano-GPT returned `upstream_http_429` and is in cooldown.
- These make the fallback chain slow. That is why one local EH skip costs minutes.
- There is nothing to raise with Electron Hub about these skips: those requests were never sent to them.

## 8. Fix plan (needs user approval; no code changed)

### 0. User decision first

Confirm whether Electron Hub should be 2/min or 8/min. If 8, set it in the admin panel; this alone removes almost all skips. Items 1-4 are worth doing either way.

### 1. Exactly one token per real upstream HTTP request

- Add `RateLimiter::acquire(max_wait) -> Result<(), RateLimitShortfall>` in `router/rate_limiter.rs`. It covers consume, optional bounded wait (item 2) and bucket state for messages (item 4).
- Keep C1 as the gate for the **first** request to a candidate. Charge the extra real requests through the same limiter:
  - U2 retry iterations (`attempt > 0` in the `:4737` loop)
  - U3 `stream:false` retry
  - U4 nudge
  - U1 `SearchResubmitter::chat_completion`

  If no token arrives within the bounded wait, return a `GatewayError::Provider { status_code: Some(429) }` tagged `rate_limit_skip`, so the existing failover handles it.
- Keep C2 as is. It is correct: each pass-through fallback is a separate request.
- Charge nothing for skipped candidates, cooldown or breaker skips, compression, or other local work. The current code already gets this right; the guard tests lock it in.
- **Behavior change to flag:** at 2/min, charging Codex Search rounds and retries means more local throttling of those inner calls. That matches the account limit the setting exists to protect.

### 2. Bounded wait: recommended, for

- Add `rate_limit_max_wait_ms: Option<u64>` to `Provider` (`config/mod.rs`, next to `rate_limit_per_minute` at `:439`).
- Validation: 0-60000. Default **10000 ms** globally (covers 8/min's 7.5 s refill). Set 30000 for Electron Hub if it stays at 2/min.
- At C1, if the shortfall refills within `min(max_wait, remaining request budget)`, `tokio::time::sleep` and then consume. If another request takes the token in the meantime, skip as today.
- C1 holds no config guard (dropped at `router.rs:~6208`), no concurrency permit (acquired later inside `attempt_with_retry_with_permit`) and no breaker or DashMap guard. Sleeping there is safe. Never sleep inside `dispatch_attempts_under_permit` while holding the provider permit: release it, wait, re-acquire.
- Record the wait in the attempt ledger as a non-failure, or in a metric `rate_limit_wait_ms_by_provider`. Update `active` to a "waiting for rate limit" phase so the dashboard shows it.
- **Admin panel parity is required** (AGENTS.md): add a "Rate-limit max wait (ms)" input next to `.prov-rate` in `admin/static/index.html` (`:2985` load, `:4089` save).
- Rationale: at 8/min the worst wait is 7.5 s, against 2-4 minutes on the fallback chain today. At 2/min it is ≤30 s against the same minutes.

### 3. Errors tab shows failed attempts and failovers (content-free, backward-compatible)

- `recent_errors` (`dashboard/mod.rs:586`): query errors at the SQL level, using a new optional `LogFilter.min_status_code` handled in `logger/mod.rs` `query`. The newest 25 errors then always appear regardless of later successes. The JSON shape (`Vec<LogEntry>`) stays the same.
- `renderErrors` (`index.html:804`):
  - Reason column shows `e.error || e.response_body || errorClassLabel(e.error_class)`.
  - New "Kind" column: `Attempt (recovered)` when `path` ends with `#attempt`, otherwise `Final`.
  - Short trace id for correlation.
  - Labels: `rate_limit_skip` → "Local rate limiter: not sent to provider"; `ttfb_timeout` → "No first byte before TTFB timeout"; `upstream_http_NNN` → "Provider returned HTTP NNN"; `cooldown_skip`, `circuit_open_skip`, `client_disconnect`, `gateway_deadline` likewise.
- No request or response content is added.

### 4. Make `rate_limit_skip` unmistakable

- The `ProviderAttempt` message at `router.rs:6315` changes from "Rate limit exhausted" to: `"Local rate limiter (2/min): 0.54 tokens available, next token in 14s; request not sent to provider"`. The numbers come from `RateLimitShortfall`.
- Keep `status_code` 429 and `error_class` `rate_limit_skip` (stable DD8 vocabulary, API-compatible). Item 3's UI label shows the "not sent" meaning.
- Mirror the wording in the `warn!` at `:6306`.

## 9. Guard tests (fail before the fix, pass after)

Patterns to follow:

- router unit tests in `router/router.rs` with `wiremock` and the `test_provider`/`test_group` helpers (around `:11957`, `:13239-13256`)
- the tight-bucket tests (around `:16320-16470`)
- `#[tokio::test(start_paused = true)]` for deterministic clocks (as in `rate_limiter.rs` tests)
- `tower::ServiceExt::oneshot` plus `common::isolate_databases` for dashboard and integration tests

Tests:

1. **`single_request_consumes_exactly_one_provider_rate_token`** (router.rs)
   - Invariant: tokens consumed == HTTP requests received by the mock, for client `stream:true` and `stream:false`.
   - Cases:
     - (a) plain success: 1 == 1, passes on both builds
     - (b) mock returns 500 then 200 with `max_retries_per_provider: 1`: 2 requests and 1 token pre-fix, so **fails**
     - (c) Codex Search resubmission: drive `SearchResubmitter::chat_completion` directly against the mock (no router-level OAuth test helper exists), or add an `OAuthManager` test token helper: 2 requests and 1 token pre-fix, so **fails**
2. **`sequential_requests_under_limit_never_rate_limit_skip`**
   - rpm 8, paused clock, 8 sequential requests spaced 8 s apart. Assert no `rate_limit_skip` ledger rows and exactly 8 mock requests.
   - Regression guard against future over-consumption. It **passes on pre-fix code too**, because no over-count exists today. That is the honest result.
   - Stricter variant: include one Codex Search round per request, and assert ≤ rpm tokens per rolling 60 s.
3. **`rate_limit_skip_waits_briefly_for_primary_token`**
   - rpm 2, `rate_limit_max_wait_ms` 30000, paused clock. Three sequential requests at t = 0, 1 s and 2 s. The third must be served by the primary after a ≤30 s wait, with no fallback mock hit.
   - Pre-fix: the third skips to the fallback, so **fails**.
   - Also assert a request over the wait budget still skips.
4. **`errors_tab_includes_failed_attempts_on_successful_request`** (`dashboard/mod.rs` tests, pattern at `:1426`)
   - Log one `#attempt` 429 `rate_limit_skip` row, then 101 newer 200 rows. `GET /dashboard/errors` must contain the attempt row with `error_class`.
   - Pre-fix: dropped by the `limit*4` window, so **fails**.
5. **`rate_limit_skip_message_states_local_bucket_state`**
   - Drain rpm 2, then route. The aggregated attempt message must contain "Local rate limiter" and "not sent".
   - Pre-fix: "Rate limit exhausted", so **fails**.

UI verification for item 3 (chrome-devtools): open the dashboard's Recent Errors tab. A `rate_limit_skip` row must show the "Local rate limiter" label and "Attempt (recovered)", with no console errors.
