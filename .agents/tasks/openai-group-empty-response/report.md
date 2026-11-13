# "Open AI" group: turn ends without a usable response

## Summary

The "Open AI" group is currently served by **AWS Bedrock `openai.gpt-5.6-sol`**, not the OpenAI (Codex OAuth) provider. The OpenAI provider is in an upstream rate-limit cooldown: its weekly Codex quota is at 0%. Nano-GPT is also cooling down.

Bedrock's `openai.gpt-5.*` models go through the Mantle **Responses** adapter. That adapter keeps only the answer text and **throws away every `function_call` output item**. It also hard-codes `finish_reason: "stop"`. So when the model makes a tool call, which is most Kilo Code turns:

- **Tool call with a short preamble** ("I'll read README.md…"): the router accepts the turn as a success. Kilo receives a one-line message, `finish_reason: stop` and no `tool_calls`, so the agent turn ends. This matches the user's report and the deployed logs.
- **Tool call with no text**: the turn counts as empty. The router fails over (Bedrock 5.6 → Bedrock 5.5, both dropped the same way) to NIM `gpt-oss-20b`, which is degraded and slow on 30k-token prompts. If that also fails, the client gets an error instead of an answer.

Root cause: `crates/ai-gateway/src/providers/bedrock.rs`, `chat_completion_responses_api`, lines 2447–2467 (text-only extraction) and 2555–2561 (`openai_response_from_text(..., "stop")` with no `tool_calls`).

## (a) Reproduction (deployed instance, 2026-10-05 ~16:17–16:24Z)

All probes were small. Secrets were not printed. The admin and dashboard endpoints answered without auth (`admin.auth.enabled: false`).

### Deployed state

- `GET /v1/models` lists the group as **`Open AI`** (with a space).
- Group members and priorities, from `GET /admin/config`:

| Priority | Model | Provider | Tier |
|---|---|---|---|
| 1 | gpt-6.1-sol | OpenAI | powerful |
| 4 | gpt-6-sol | OpenAI | powerful |
| 7 | gpt-5.5 | OpenAI | balanced |
| 10 | gpt-5.6-luna | OpenAI | fast |
| 13 | openai/gpt-oss-120b | Nano-GPT | fast |
| 16 | openai.gpt-5.6-sol | AWS Bedrock | (none) |
| 19 | openai.gpt-5.5-2026-04-23 | AWS Bedrock | (none) |
| 22 | openai/gpt-oss-20b | Nvidia NIM | fast |

- OpenAI provider: `auth_method: oauth` (Codex). `GET /admin/oauth/openai/usage` shows `weekly.requests.remaining: 0` (resets 2026-10-07 16:23Z).
- `/dashboard/metrics` shows these cooldowns:
  - OpenAI: until 2026-10-05 23:39Z.
  - Nano-GPT: until 2026-10-06 00:00Z.
  - NIM: 60% success rate, average ~63 s.
- AWS Bedrock: `type: bedrock`, base `https://bedrock-mantle.us-east-2.api.aws/v1` (API-key/Mantle mode).
- `codex_search.enabled: true` with OAuth authenticated. Smart routing is enabled (`classifier: jev`).

### Request log history

From `GET /dashboard/logs`, filtered to `requested_model == "Open AI"`:

- Up to 2026-10-02, requests were served by `OpenAI/gpt-6.1-sol` and `gpt-5.6-sol`.
- From 2026-10-04 on, they were served by Nano-GPT, NIM `gpt-oss-20b` and **Bedrock `openai.gpt-5.6-sol`**.
- The user's session:
  - 16:04:46 and 16:05:25: NIM `gpt-oss-20b`, prompt growing 29197 → 30088 tokens. The agent loop was progressing.
  - 16:08:44, 16:08:54 and 16:09:11: **Bedrock `openai.gpt-5.6-sol`**, 200, ~2.5 s each, **identical 30942-token prompts**. Identical prompts look like a re-sent or retried turn, not an agent loop moving forward.
  - Cost per call: 0.168300 / 0.168399 / 0.168564. At $5.5/M input and $33/M output, that is only about 10–40 output tokens: a one-line reply.
- Request and response bodies are not logged (`request_body_logging: false`), so the payload could not be read directly.

### Probe 1: Kilo-shaped request to the group

Streaming, system prompt, one `tools` function, `stream_options.include_usage=true`, "Say hi". It routed to the fast tier on NIM `gpt-oss-20b`. Both text and the finish/usage chunk arrived correctly:

```
data: {"choices":[{"delta":{"role":"assistant"},...}],"model":"Open AI",...}
data: {"choices":[{"delta":{"reasoning":"User just says ..."}...}],"model":"openai/gpt-oss-20b"}
data: {"choices":[{"delta":{"content":"Hi!"}...}]}
data: {"choices":[{"delta":{},"finish_reason":"stop"...}],"usage":{"completion_tokens":35,"prompt_tokens":936,...}}
data: {"choices":[{"delta":{"content":"\n\n---\n📝 10 memories injected | 0 memories stored"}...}],"id":"chatcmpl-memory-..."}
data: [DONE]
```

### Probe 2: same request pinned to `openai.gpt-5.6-sol` (Bedrock), plain "Say hi"

It returned `Hi!`, `finish_reason: stop`, usage 901/6. Plain text works.

### Probe 3: pinned to `openai.gpt-5.6-sol`, prompt forcing a tool call, streaming

The response was served by **NIM `gpt-oss-20b`**, not Bedrock (log trace `8fb9183d`: requested `openai.gpt-5.6-sol`, provider Nvidia NIM, 4.5 s total). Both Bedrock attempts were judged empty and failed over within seconds. That fits a function_call-only reply whose call was dropped. The NIM tool_call deltas themselves were well formed (`index`, `id`, `type`, `function.name`, then argument deltas, then `finish_reason: tool_calls`).

### Probe 4 (decisive): pinned to `openai.gpt-5.6-sol`, "say what you'll do, then call the tool", non-streaming

```
HTTP/1.1 200 OK   x-trace-id: 2b45facb-...
{"model":"openai.gpt-5.6-sol","choices":[{"index":0,"message":{"role":"assistant",
 "content":"I’ll read `README.md` and return its contents.\n\n---\n📝 8 memories injected | 0 memories stored"},
 "finish_reason":"stop"}],"usage":{"prompt_tokens":954,"completion_tokens":38,...}}
```

There are 38 completion tokens, but the visible text is about 12. The rest was the `read_file` function_call that the gateway dropped. No `tool_calls` came back, and `finish_reason` is `stop`. Kilo Code shows this one-liner and ends the turn. This matches the user's ~2.5 s, ~10–40-output-token Bedrock 200s.

## (b) Root cause, with code references

1. **Why traffic reaches Bedrock.**
   - The OpenAI provider is in a 429 cooldown. The streaming router skips cooled-down candidates (`router/router.rs` ~8640, `provider_cooldown_remaining_secs`), and so does the buffered failover loop.
   - Nano-GPT is also cooling down. The next candidates are Bedrock (16, 19) and then NIM (22).
2. **Bedrock always takes the buffered path.**
   - `router/router.rs:8754` sets `is_bedrock`. `router/router.rs:8699` sends codex-search-active requests to the buffered path.
   - So the request goes through `route_request` → `BedrockProvider::chat_completion_stream` → `dispatch_mantle_stream` (`providers/bedrock.rs` ~2950–3010). For `MantleApi::Responses` models, this buffers the result and emits one OpenAIResponse event (`bedrock.rs:3005`).
   - `mantle_api_for_model` (`bedrock.rs:30–41`) maps `openai.gpt-5.6*` and `openai.gpt-5.5*` to `MantleApi::Responses`.
3. **The Responses adapter drops tool calls.** In `chat_completion_responses_api` (`bedrock.rs:2324–2567`):
   - Tools are forwarded upstream (`bedrock.rs:2416`, `responses_tools_from_chat`), so the model can and does emit `function_call` items.
   - The reply parser reads only `output_text`, or else the first `output[].content[].text` (`bedrock.rs:2447–2467`). `output[]` items with `type: "function_call"` (`call_id`, `name`, `arguments`) are never read.
   - The response is built with `openai_response_from_text(&model, content, …, "stop")` (`bedrock.rs:2555–2561`, constructor at `bedrock.rs:2781`). `message.extra` is empty, so there are no `tool_calls`, and `finish_reason` is always `"stop"`. `status: "incomplete"` / `max_output_tokens` is never mapped to `length`.
   - Chat history going *into* Mantle is already converted to `function_call` / `function_call_output` items (commit 89f2dff, `bedrock.rs` ~700). Only the output direction is missing.
4. **How the router judges the result** (`router/router.rs`):
   - Preamble plus dropped call: content is non-empty, so `response_has_content` (`6669`) passes. The turn is returned as a successful "stop". This is the user-visible stall.
   - Call only: content is empty. The check at `5851` fails over to the next candidate. Bedrock 5.5 repeats the same drop. Then it goes to NIM `gpt-oss-20b` (degraded, slow), or ends in `AllProvidersFailed` (`6300`). Earlier logs show 22 Bedrock `Open AI` 502s.

## (c) Alternative and contributing hypotheses (ranked)

1. **Confirmed contributor: OpenAI provider exhausted.** The weekly Codex quota is at 0%, so the group silently degrades to Bedrock and NIM. It is not the bug itself, but it is why the bug surfaced now. It resolves after the weekly reset (2026-10-07 16:23Z); the next cooldown expiry is 23:39Z today.
2. **Likely contributor: NIM `gpt-oss-20b` on 30k-token Kilo prompts.** It has a 60% success rate and is slow (9–30 s, failures recorded). When Bedrock fails over on a call-only turn, NIM is the last stop, and a failure there gives an error or a stalled turn. Confirm via `/dashboard/metrics` NIM `last_failure_reason` at the time of a stall.
3. **Secondary defect: reasoning copied into content on tool-call turns.** `promote_reasoning_to_content` runs unconditionally at `router/router.rs:5840`, even though the comment says "no tools in play". When a provider returns reasoning plus `tool_calls` with empty content, the chain of thought is promoted into `content` (Probe 3 showed reasoning duplicated as content). It is not the cause, but it leaks CoT into Kilo's history.
4. **Secondary hazard: memory footer sent after the finish chunk.** The memory feedback footer (`memory/mod.rs:827`) arrives as a separate content delta *after* the `finish_reason` / usage chunk, with a different `id` (`chatcmpl-memory-…`). On tool-call turns it adds stray text after `finish_reason: tool_calls`. Kilo tolerated it in the probes, but stricter clients may drop it or treat the turn as done. It is not the cause, because text was delivered.
5. **Refuted: SSE framing or usage chunk dropped.** Every streamed probe had a role chunk, deltas, finish+usage, and `[DONE]`. Tool-call deltas had `index`, `id` and `type`.
6. **Unlikely: exact or semantic cache replay.** The three user calls each incurred Bedrock cost and ~2.5 s latency, so they were not cache hits. `should_cache_response` also refuses tool-call turns.

## (d) Recommended fix outline (not implemented)

1. In `chat_completion_responses_api` (`bedrock.rs`), parse `output[]` by item type:
   - `message` → join every `output_text` part, not just the first `text`.
   - `function_call` → `message.extra["tool_calls"] = [{id: call_id, type: "function", function: {name, arguments}}]`.
   - Ignore `reasoning` items, or map them to `reasoning` / `reasoning_content`.
2. Derive `finish_reason`:
   - `tool_calls` if any function_call is present.
   - `length` if `status == "incomplete"` with `incomplete_details.reason == "max_output_tokens"`.
   - Otherwise `stop`.
3. Treat a function_call-only reply as content. `response_has_content` then passes, and failover stops wasting Bedrock 5.5 and NIM attempts.
4. Add `wiremock` tests next to the existing Responses tests in `bedrock.rs` (e.g. `test_responses_compaction_only_surfaces_summary_block`). Cover function_call-only, text + function_call, multiple calls and incomplete/length. Also add a streaming-handler test that the replayed chunks carry the `tool_calls` deltas.
5. Optional follow-ups:
   - Gate `promote_reasoning_to_content` at `router.rs:5840` on "no `tool_calls`" (and ideally on `!tools_present`).
   - Emit the memory footer before the terminal finish chunk, or skip it on `tool_calls` turns.
   - Turn admin auth on: `/admin/config` (including PUT and reload) is open to the LAN.
6. Immediate workaround until it is fixed: lower Bedrock `openai.gpt-5.*` below a tool-capable provider in the "Open AI" group, or remove those models, until the Codex quota resets.

## (e) Not verified

- **Exact Bedrock payload for the user's three requests.** Body logging is off. The diagnosis comes from Probe 4 (same model and adapter, reproduced) plus the matching cost, latency and output-token signature. I did not read server tracing logs (stdout on the host), where the `"2xx but no extractable content"` warn lines from call-only turns would appear.
- **Deployed build vs repo HEAD (c200d5f).** There is no version endpoint. The observed behavior matches HEAD's code, and the `responses-codex-native` worktree branch is already merged into master.
- **Which Kilo Code version, and how it reacts to a text-only reply** (ends the turn vs "no tool used" retry). That depends on the client. The repeated identical 30942-token prompts suggest the user re-sent the turn.
- **Which smart-routing tier the user's requests picked.** No routing-decision log was readable over HTTP. Either way, the OpenAI candidates were skipped by cooldown.
