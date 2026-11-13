# Bedrock Responses adapter: surface tool_calls, reasoning, and finish_reason

The "Open AI" group was silently degrading to AWS Bedrock `openai.gpt-5.*` (Codex quota exhausted), and the Mantle Responses adapter only extracted answer text — it dropped every `function_call` output item and hard-coded `finish_reason: "stop"`. A tool-call-only turn became an empty turn (failing the group over) and a preamble-plus-call turn returned a one-liner with no `tool_calls`, ending the agent turn. This branch rewrites the Responses `output[]` parser to translate `message`/`function_call`/`reasoning` items into the gateway's OpenAI chat shape, derives `finish_reason` correctly, stops promoting reasoning into content on tool-call turns, skips the trailing memory footer on tool-call turns, and adds a startup WARN when the admin panel runs without auth. Four focused commits, one per concern, all centered on the four files the plan named.

Watch for: the compaction-only branch still runs before the new tool_calls logic and wins on an empty-text response, so a hypothetical response carrying both an `encrypted_content` summary and a `function_call` would surface the summary and drop the call (possible, not a real Responses shape — see details). Everything else maps cleanly to the task.

**Verdict**: APPROVED

## High-level view

The core fix is a new `responses_output_to_chat` helper in `bedrock.rs` that walks `output[]` by item type: `message` parts are concatenated across all `output_text` parts (not just the first), `function_call` items become OpenAI `tool_calls` on `message.extra` using the exact `{id, type:"function", function:{name, arguments}}` shape the rest of the codebase requires, and `reasoning` items land in `message.extra["reasoning_content"]`, never in content. `finish_reason` is `tool_calls` when any call is present, `length` on incomplete+`max_output_tokens`, else `stop`. A function_call-only reply is now a valid non-empty turn.

The streaming requirement is satisfied without new streaming code: the adapter buffers and the downstream `build_streaming_chunks` already reads `message.extra["tool_calls"]`. The test exercises `streaming_chunks_from_response` against real adapter output and asserts the header delta (index/id/type/name), the arguments delta, and the terminal `tool_calls` finish — so the "no code change needed" claim is verified by behavior, not just asserted.

`promote_reasoning_to_content` now early-returns `false` when the choice carries a non-empty `tool_calls` array, so reasoning no longer leaks into content on agent turns; the plain-text promotion path is untouched. The memory footer is gated by a new `is_tool_call_turn` helper at all four buffered emit sites, so no stray content trails a `tool_calls` finish; plain-text turns still get the footer and `[DONE]` ordering is unchanged.

Admin auth default is deliberately left as-is per the recorded user directive — back-compat is preserved and the sole safeguard is a startup WARN naming the unauthenticated-admin risk, factored into a pure `admin_auth_warning` helper. The admin-panel toggle and env wiring already existed in `index.html`; parity is a verify-only item and the controls are present.

Tests were added for every case the task enumerates, and the coder recorded a clean build (0 warnings) with 4989 tests passing.

<details>
<summary>Issues (2)</summary>

1. **Compaction-vs-tool_call precedence** — the compaction-only branch runs before the tool_calls logic and matches on empty text alone, so a response with both an `encrypted_content` summary and a `function_call` would surface the summary and drop the call. Not a real Responses payload shape and per the plan's "compaction branch unchanged, ordered before"; leave as-is or add a one-line guard if defensiveness is wanted. (possible)
2. **Admin-auth default unchanged** — task item 4 asked that the default-change risk be "reasoned about"; the branch keeps the open-admin default and warns instead, matching the recorded user directive. Verify this directive is the intended final decision, since the task wording also mentions "defaults to true". (confirmed, resolved by directive)

</details>

<details>
<summary>Details</summary>

### Responses output translation (`responses_output_to_chat`)

`message` items concatenate every part whose `type` is `output_text` (with a fallback to any part `text`), in order — `test_responses_message_concatenates_all_output_text_parts` proves both parts of a two-part message are joined, which is exactly the bug the old `.find_map(text)` caused. `function_call` items build `{id: call_id, type: "function", function: {name, arguments}}`, skip empty-name items, and apply the input-side arguments-string rule (String as-is, Null/absent → `"{}"`, other → `.to_string()`), mirroring `chat_messages_to_responses_input` so the output and input directions agree. `reasoning` items pull text from both `summary[]` and `content[]` into the reasoning string and never touch `text`; `test_responses_reasoning_not_in_content` confirms content stays empty while `reasoning_content` is set and the tool_call still comes through.

`finish_reason` derivation is correct and ordered: `tool_calls` first, then `length` on `status == "incomplete"` with `incomplete_details.reason == "max_output_tokens"`, else `stop`. `test_responses_incomplete_max_output_tokens_maps_length` covers the length path with text present. The empty-content WARN now fires only when text is empty AND tool_calls empty AND no compaction, so a legitimate call-only turn no longer logs a false "empty response".

One ordering subtlety: the compaction-only branch (`content.trim().is_empty() && !compaction_summaries.is_empty()`) still runs before the tool_calls/finish_reason block and returns early. If a single response ever carried both a summary block (`encrypted_content`) and a `function_call`, the summary wins and the call is dropped. This is per the plan ("compaction branch unchanged, ordered before the new logic"), and a mixed summary+function_call payload is not a shape the Responses API emits, so this is a possible-only concern — flagged for completeness, not blocking.

### Streaming replay carries tool_call deltas

The design relies on the buffered→SSE path: the Responses adapter buffers one `OpenAIResponse`, and `streaming_chunks_from_response` → `build_streaming_chunks` synthesizes the client deltas, reading `message.extra["tool_calls"]` (confirmed present at `handlers.rs:4992`). Because `tool_calls` live on the `#[serde(flatten)]` extra map they survive serialization. Rather than asserting this by inspection, `test_responses_streaming_replay_carries_tool_call_deltas` runs a function_call-only adapter response through `streaming_chunks_from_response` and asserts a header delta with `index:0`, `id`, `type:"function"`, `function.name`, a non-empty `function.arguments` delta equal to the serialized args, and a terminal chunk with `finish_reason: "tool_calls"`. That is the full required delta sequence, verified end-to-end.

### Reasoning no longer promoted on tool-call turns

`promote_reasoning_to_content` gains a `has_tool_calls` guard (same pattern as the existing `reasoning_only_turn` check) that early-returns `false` before the reasoning-promotion logic. `promote_reasoning_to_content_skips_when_tool_calls_present` asserts a response with empty content, `reasoning_content`, and a valid `tool_calls` entry is not promoted and content stays empty; `promote_reasoning_to_content_still_promotes_plain_text` confirms the plain reasoning-only turn is unchanged. This closes the CoT-leak hazard from the report without regressing plain-text behavior.

### Memory footer skipped on tool-call turns

`is_tool_call_turn` returns true when the first choice finishes with `tool_calls` OR carries a non-empty `tool_calls` array, and all four buffered footer emit sites now gate `memory_feedback_chunk` on `!is_tool_call_turn(&response)`. The true-streaming relay path (no buffered response) is left untouched — it is not the Bedrock buffered path. `memory_footer_skipped_on_tool_call_turn` verifies classification via the `tool_calls` array alone (finish_reason is "stop" in the fixture) and that the synthesized stream ends on a `tool_calls` finish chunk; `memory_footer_emitted_on_plain_text_turn` confirms plain turns are not misclassified. This matches "no stray content after a tool_calls finish" while preserving the footer for plain text.

### Admin-auth startup warning and config back-compat

`AdminAuthConfig` default and `#[serde(default)]` are left untouched, so omitted-field and explicit-`false` deployments still boot with open admin — back-compat preserved exactly as the recorded user directive requires. The safeguard is `admin_auth_warning(admin_enabled, admin_auth_enabled)`, a pure helper returning `Some(msg)` only when admin is enabled and auth disabled, logged via `tracing::warn!` right after the admin mount. The message names the UNAUTHENTICATED risk and the remediation (`admin.auth.enabled` + username/password env). Three unit tests cover the enabled-disabled, enabled-enabled, and admin-disabled cases. The task line "defaults to true when omitted" is superseded by the plan's recorded directive to warn rather than flip the default; this is a deliberate, documented trade-off, so it is not a blocking finding — noted so the directive stays visible.

### Admin panel parity

The `admin-auth-enabled` checkbox (index.html:345), `admin-user-env`/`admin-pass-env` inputs, config-load wiring (:3902), and save wiring (:4072) are all present. This was a verify-only item; no HTML change was needed and none was made.

### Scope, build, and tests

The diff touches only the four files the plan named (678 insertions / 37 deletions), is organized as four single-concern commits, adds no blanket `#[allow]`, no markdown change-log files, and preserves public API shapes (the one new `pub(crate)` symbol, `admin_auth_warning`, is a test seam, not an API-surface change). Per the task I did not re-run the suites; the coder's commit records `cargo check --all-targets` with 0 warnings and `cargo test` with 4989 passed, 0 failed. The 12 planned tests are all present across the four files.

</details>

<details>
<summary>File map</summary>

- `crates/ai-gateway/src/providers/bedrock.rs` — new `responses_output_to_chat` parser + `ResponsesOutput`; wired into `chat_completion_responses_api` for tool_calls/reasoning/finish_reason; 7 new wiremock/streaming tests.
- `crates/ai-gateway/src/router/router.rs` — `promote_reasoning_to_content` gated off on tool_calls; 2 new unit tests.
- `crates/ai-gateway/src/gateway/handlers.rs` — `is_tool_call_turn` helper; memory footer gated at 4 buffered emit sites; 2 new tests.
- `crates/ai-gateway/src/gateway/mod.rs` — `admin_auth_warning` pure helper + startup WARN after admin mount; 3 new tests.
- `crates/ai-gateway/src/admin/static/index.html` — unchanged (parity verify-only; toggle already present).

Full diff: `git -C "c:\Fbn_Dev_Apps\OBEY API Router\.worktrees\bedrock-responses-toolcalls" diff master...fix/bedrock-responses-toolcalls`

</details>
