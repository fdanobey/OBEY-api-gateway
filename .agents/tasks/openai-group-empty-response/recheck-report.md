# Recheck — "Open AI" group still ends turns without a tool call (Bedrock openai.gpt-5.6-sol)

## Summary / Verdict

**VERDICT: The fix IS live in the deployed binary, but a second defect remains — a compaction‑vs‑tool_call precedence bug in the Mantle Responses adapter.**

The deployed server now correctly translates Mantle Responses `function_call` output items into OpenAI `tool_calls` (confirmed live below). But `chat_completion_responses_api` has an earlier **compaction‑only short‑circuit** that fires whenever the upstream reply contains any `output[]` item with an `encrypted_content` field AND the extracted answer text is empty. That branch returns a response carrying only the summary block, with `content: ""`, `finish_reason: "stop"`, and **no `tool_calls`** — it never looks at the `tool_calls` the parser already extracted. 

On the user's real Kilo/Codex turns (~31k tokens, `task_type=tool_use`), the upstream reply contains **both** a server‑side compaction/summary block **and** a `function_call` item. A function_call‑only turn has empty message text, so `content.trim().is_empty()` is true and a summary block is present → the compaction branch wins and the tool call is dropped. The turn comes back as content‑only text with `finish_reason stop`, so Kilo ends the turn with no action. This is exactly the "compaction‑vs‑tool_call precedence" non‑blocking finding the earlier review raised, now promoted to the live root cause.

Decisive corroboration from the user's host logs (deployed binary): on every failing turn this line fires —
`INFO Bedrock Mantle Responses: compaction-only response, surfacing summary block(s) provider=AWS Bedrock model=openai.gpt-5.6-sol summary_blocks=1` — while smart routing shows those same traces as `group='Open AI' tier=powerful task_type=tool_use model=openai.gpt-5.6-sol original_tokens≈31417 stream=true`.

## (a) Live decisive probe (non‑streaming, pinned to Bedrock)

Re‑established state (2026 recheck, read‑only):
- `GET /v1/models` — 41 models; group listed as **`Open AI`** (with a space); Bedrock ids `openai.gpt-5.6-sol` and `openai.gpt-5.5-2026-04-23` present.
- `GET /admin/oauth/openai/usage` — **weekly requests remaining = 0** (resets `1791390197`). OpenAI (Codex OAuth) is still forced out, so the group still degrades to Bedrock. Admin endpoints still answered without auth.
- `GET /admin/config`, `/dashboard/metrics`, `/dashboard/logs` — all answered (read‑only).

Decisive non‑streaming probe pinned to `openai.gpt-5.6-sol`, system prompt + one `read_file` tool + a prompt that forces a short preamble then a tool call (trace `346cdf1c-323d-4058-841b-ef87e34b04bc`):

```jsonc
model_served : "openai.gpt-5.6-sol"      // genuine Bedrock, not a NIM failover
finish_reason : "tool_calls"
message: {
  "role": "assistant",
  "content": "I'll read `README.md` and use its contents for the next step.",
  "tool_calls": [{
    "id": "call_eda1bc189c3d596dbaa5024534a08d8d",
    "type": "function",
    "function": { "name": "read_file", "arguments": "{\"path\":\"README.md\"}" }
  }]
}
usage: prompt 956 / completion 42   // no compaction block (has_compaction=False)
```

Interpretation: with a **small** prompt (956 tokens) there is **no server‑side compaction**, so no `encrypted_content` item is returned, the compaction short‑circuit does NOT fire, and the fix works — `tool_calls` present, `finish_reason tool_calls`. This both (1) proves the fix is deployed, and (2) explains why this cheap probe passes while the user's large turns fail: only the large Codex turns carry a `compaction_trigger` and get a summary block back, which triggers the buggy branch. The probe cannot reproduce the failing path precisely because it is too small to provoke compaction; the host logs supply that half of the evidence.

## (b) Root cause — exact code

File: `crates/ai-gateway/src/providers/bedrock.rs`, fn `chat_completion_responses_api`.

1. The parser is correct. `responses_output_to_chat` (defined at **bedrock.rs:768**) iterates `output[]` by `type` and already builds `tool_calls` from `function_call` items (`{id: call_id, type: "function", function: {name, arguments}}`), concatenates `message` text, and collects `reasoning`. Its result is destructured at **bedrock.rs ~2552–2558** (`ResponsesOutput { text, tool_calls, reasoning, incomplete_max_tokens }`).

2. The precedence bug. Immediately after, the adapter collects every `output[]` item that has an `encrypted_content` field into `compaction_summaries` (**bedrock.rs ~2585–2595**), then short‑circuits:

   ```rust
   // bedrock.rs:2598
   if content.trim().is_empty() && !compaction_summaries.is_empty() {
       tracing::info!(..., "Bedrock Mantle Responses: compaction-only response, surfacing summary block(s)"); // bedrock.rs:2606
       let mut response = self.openai_response_from_text(&model, String::new(), prompt_tokens, completion_tokens, "stop");
       // inserts message.extra["compaction"] + response.extra["compaction_output"]
       return Ok(ProviderResponse { response, ... });   // returns here — tool_calls NEVER attached
   }
   ```

   The condition `content.trim().is_empty() && !compaction_summaries.is_empty()` is satisfied by a turn that has (a) a `function_call` item (no answer text → empty `content`) and (b) a compaction/summary block in the same reply. The branch builds the response from empty text with hard‑coded `finish_reason "stop"` and `return`s **before** the tool_calls/finish_reason logic at **bedrock.rs ~2640‑onward** (where `finish_reason = "tool_calls"` and `message.extra["tool_calls"]` would be set). The already‑extracted `tool_calls` variable is simply dropped.

3. Why it wins over tool_calls: it is an **early `return`** that is positioned after the parser but ahead of the tool_calls attachment, and its guard only tests `content` emptiness + presence of a summary block. It never consults `tool_calls`. So when both are present, it short‑circuits and the function_call is lost.

4. Can both legitimately coexist? Yes — the host logs confirm it: `task_type=tool_use` turns (so the model is emitting a `function_call`) simultaneously log `summary_blocks=1`. Server‑side compaction fires on these large (~31k‑token) Codex turns because the request carries a `compaction_trigger`; Bedrock returns the new compaction/summary block in the same Responses `output[]` as the model's `function_call`. That is the exact overlap the current guard mishandles.

## (c) Deployment freshness — resolved, not the issue

The fix is deployed. Evidence: (1) the live probe above returns `tool_calls` + `finish_reason tool_calls` on a non‑compaction Bedrock turn — behavior only the post‑fix parser produces; (2) the host logs emit the `compaction-only response, surfacing summary block(s)` line, which only exists in the post‑fix code path. No redeploy gap. The user's "same behavior after update" is caused by the remaining precedence bug on compaction turns, not by a stale binary.

## (d) Fix outline (not implemented)

In `chat_completion_responses_api` (`crates/ai-gateway/src/providers/bedrock.rs`), the function_call extraction must take precedence over — or be merged with — the compaction‑only surfacing. The adapter must NEVER drop `tool_calls`.

Concretely, change the compaction short‑circuit guard and body so that when `tool_calls` is non‑empty it does NOT short‑circuit as a dead "stop" turn:

- Guard change at **bedrock.rs:2598**: only take the compaction‑only early return when `tool_calls.is_empty()` as well, e.g.
  `if content.trim().is_empty() && tool_calls.is_empty() && !compaction_summaries.is_empty() { ... }`.
- When tool_calls ARE present alongside a summary block, fall through to the normal path (bedrock.rs ~2640+) so that:
  - `finish_reason = "tool_calls"`,
  - `message.extra["tool_calls"]` is populated,
  - AND the compaction block is still surfaced — attach `message.extra["compaction"]` / `response.extra["compaction_output"]` from `compaction_summaries` in the fall‑through path too, so the client can still replay the compaction state. (Surface the summary as compaction metadata/reasoning, never as the sole content, and never at the cost of the tool_calls.)
- Keep the existing compaction‑only behavior unchanged for the genuine case: empty content AND no tool_calls AND a summary block present.

Suggested tests (mirror the existing wiremock Responses tests next to `test_responses_compaction_only_surfaces_summary_block`):
- `test_responses_function_call_with_compaction_block_keeps_tool_calls`: output = one `function_call` + one `{encrypted_content}` summary item, empty message text → assert `message.extra["tool_calls"]` has the call, `finish_reason == "tool_calls"`, and the compaction block is still surfaced (`message.extra["compaction"]` present).
- `test_responses_text_plus_compaction_plus_function_call`: message text + summary + function_call → tool_calls present, finish_reason tool_calls, content preserved, compaction surfaced.
- Keep `test_responses_compaction_only_surfaces_summary_block` green (compaction‑only, no tool_calls → still `stop` with the summary).

Verify: `cargo test -p ai-gateway responses_` (worktree/crate manifest) all green; `cargo check -p ai-gateway --all-targets` clean.

Immediate operator workaround until patched: lower the Bedrock `openai.gpt-5.*` members below a tool‑capable provider in the "Open AI" group, or remove them, until the fix ships / the Codex weekly quota resets (`1791390197`).

## Secondary issues observed (lower priority, kept separate)

1. **Memory‑extraction JSON parse warnings.** The host logs show `WARN ... memory extraction provider failed to parse extraction response as JSON`. This is in the memory subsystem, unrelated to the tool_call drop; it degrades memory capture but does not end the turn. Track separately.
2. **Reasoning promoted into content on tool‑call turns** and **memory footer appended after the finish chunk** — both were flagged in the original report as secondary and are addressed by plan items 4–7; not the live root cause here.
3. **Admin endpoints remain unauthenticated** on the LAN (`admin.auth.enabled: false`); `/admin/config` answered without auth during this recheck. The deployed fix added only a startup warning per the user's decision; posture is unchanged.

## (e) Not verified

- **Exact upstream Mantle Responses JSON for the user's failing turns.** Request/response body logging is off (`request_body_logging: false`), so the raw `output[]` (summary block + function_call together) was not captured directly over HTTP. The diagnosis rests on: the host‑log `summary_blocks=1` + `task_type=tool_use` on the same traces, the code path at bedrock.rs:2598, and the live probe showing the non‑compaction path works. These three agree.
- **Live dashboard‑log row for probe trace `346cdf1c`.** The `/dashboard/logs` query returned (endpoint reachable) but console rendering was truncated in this shell; the `model_served: openai.gpt-5.6-sol` field in the probe response already confirms Bedrock served it (not a NIM failover).
- **Streaming variant.** Not separately re‑captured this round; the bug is in the buffered Responses adapter that both the non‑stream path and `dispatch_mantle_stream` replay feed from, so fixing the adapter fixes both. (`build_streaming_chunks` already synthesizes tool_call deltas from `message.extra["tool_calls"]`, so once the adapter stops dropping them, the stream carries them.)
