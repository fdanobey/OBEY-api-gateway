# Implementation Plan — OBEY API Router: "Open AI" group empty-response fixes

All work happens in the worktree:
`c:\Fbn_Dev_Apps\OBEY API Router\.worktrees\bedrock-responses-toolcalls`

Step agents' cwd is the MAIN workspace. Always use absolute paths rooted at the worktree, and run cargo with the worktree manifest:

```
cargo check -p ai-gateway --all-targets --manifest-path "c:\Fbn_Dev_Apps\OBEY API Router\.worktrees\bedrock-responses-toolcalls\Cargo.toml"
cargo test  -p ai-gateway            --manifest-path "c:\Fbn_Dev_Apps\OBEY API Router\.worktrees\bedrock-responses-toolcalls\Cargo.toml"
```

Clean-build gate: zero errors AND zero warnings. No blanket `#[allow]` — remove dead code or gate test-only helpers with `#[cfg(test)]`. Surgical diffs, match existing style, no refactors, no markdown change-log files, preserve public API and config back-compat. Windows/PowerShell, reasonable timeouts on every command.

---

## Design decisions (grounded in the code read)

- **Output tool_calls JSON shape.** The INPUT side (`bedrock.rs` `chat_messages_to_responses_input`, ~675-705) reads assistant `tool_calls` as `{ "id", "function": { "name", "arguments" (string) } }` and emits Responses `function_call` items `{ type, call_id, name, arguments }`. The gateway's own non-Responses production and `response_has_content` (`router.rs` ~6689-6716) require each OpenAI tool_call to have non-empty `id`, a `type`, and non-empty `function.name`. So the Responses OUTPUT translation must build exactly:
  ```json
  { "id": <call_id>, "type": "function", "function": { "name": <name>, "arguments": <arguments string> } }
  ```
  placed on `message.extra["tool_calls"]` as a JSON array. `arguments` is passed through as a string (Responses `function_call.arguments` is already a JSON string); if upstream sends a non-string, stringify it (mirror the input-side rule: String→as-is, Null/absent→`"{}"`, other→`.to_string()`).

- **Streaming replay needs NO new code.** The Responses adapter buffers and `dispatch_mantle_stream` (~2995-3010) emits ONE SSE event = the full serialized `OpenAIResponse`. The client-facing SSE deltas are synthesized downstream by `build_streaming_chunks` (`gateway/handlers.rs` ~4989-5210), which already reads `message.extra["tool_calls"]`, emits the header delta (`index/id/type/function.name`), the arguments delta, repeats for multiple calls, replays reasoning first, and sets the terminal `finish_reason: "tool_calls"` whenever tool_calls are present. Therefore populating `message.extra["tool_calls"]` + the choice `finish_reason` is sufficient; the streaming requirement is covered by a test, not a code change. `tool_calls` live in `message.extra` (a `#[serde(flatten)]` map) so they survive `serde_json::to_string`.

- **Reasoning items → `reasoning_content`.** The gateway's reasoning carriers are `reasoning` / `reasoning_content` on `message.extra` (`router.rs::reasoning_text` ~6312-6320; `handlers.rs::reasoning_delta` ~5247). Map Responses `reasoning` output items to `message.extra["reasoning_content"]` (never into `content`). This is consumed by the streaming replay for reasoning-first ordering and is stripped from history on the way back in.

- **finish_reason derivation.** `tool_calls` if any `function_call` item present; else `length` if top-level `status == "incomplete"` and `incomplete_details.reason == "max_output_tokens"`; else `stop`. (`build_streaming_chunks` forces `tool_calls` when tool_calls exist, so the choice value mainly matters for the non-stream path and `length`.)

- **Memory footer (task 3).** `memory_feedback_chunk` (`handlers.rs` ~999-1011) is emitted as a separate `content` delta AFTER `streaming_chunks_from_response(&response)` (which already includes the terminal finish/usage chunk) at four buffered-emit sites (~2404, ~2730, ~2814, ~2951) and the pass-through relay. Decision: **skip the footer when the turn is a tool_calls turn** (simplest, matches "no stray content after a tool_calls finish" and preserves plain-text behavior). Gate each emit site on "response is not a tool_calls turn" using the `response` in scope (finish_reason == "tool_calls" OR non-empty `message.extra["tool_calls"]`). A tiny local helper keeps the four sites consistent.

- **Admin auth default (task 4) — DECISION per user directive (workflow notification):** Do NOT change the default of `admin.auth.enabled`; leave `AdminAuthConfig::default()` = `false` and the `#[serde(default)]` behavior untouched. Back-compat is fully preserved (deployments that omit the field keep open access). The SOLE safeguard is a startup WARN when admin is enabled but admin auth is disabled, stating the admin endpoints are unauthenticated and reachable by anyone who can reach the port. Rationale recorded here: flipping the default would silently break existing LAN deployments that rely on open admin; the user explicitly chose the warning-only approach.

- **Admin-panel parity (task 4).** The `admin-auth-enabled` checkbox + username/password env inputs ALREADY exist in `admin/static/index.html` (~345-356) and are wired in both config load (~3902) and save (~4072). Parity is satisfied; this is a verify-only item (no HTML change expected).

- **Decomposition decision.** Not decomposed into FEAT artifacts: the fixes are interdependent, centered on `bedrock.rs`, share one crate build/test gate, and the existing implement+review loop (gated on `review.json` `verdict == APPROVED`) is the right vehicle.

---

## Ordered implementation items

- [ ] 1. **Add Responses `output[]` parser helper in `bedrock.rs`.** Add a private free function (near `chat_messages_to_responses_input`, e.g. `fn responses_output_to_chat(value: &serde_json::Value) -> (String, Vec<serde_json::Value>, Option<String>, bool)` returning `(text, tool_calls, reasoning_content, incomplete_max_tokens)`), that iterates the top-level `output` array by item `type`:
      - `message`: concatenate ALL parts' text — join every `content[]` part where `type` is `output_text` (field `text`), falling back to any part `text`. Preserve order.
      - `function_call` (fields `call_id`, `name`, `arguments`): push `{ "id": call_id, "type": "function", "function": { "name": name, "arguments": <arguments as string> } }`. Skip items with empty `name`. Arguments-string rule mirrors the input side (String→as-is, Null/absent→`"{}"`, other→`.to_string()`).
      - `reasoning`: collect reasoning text (e.g. `summary[].text` / `content[].text`) into the reasoning string; never into `text`.
      Also compute `incomplete_max_tokens` from top-level `status == "incomplete"` && `incomplete_details.reason == "max_output_tokens"`. Keep `output_text` top-level string as a fallback for `text` when present (back-compat with existing `test_versioned_gpt_55_uses_responses_api`).
      Files: `crates\ai-gateway\src\providers\bedrock.rs`
      Verify: `cargo check -p ai-gateway --all-targets --manifest-path "<worktree>\Cargo.toml"` compiles clean (helper may be `#[cfg(test)]`-exercised only until wired in item 2; if left unused it must be removed or used — prefer wiring in item 2 in the same change).

- [ ] 2. **Wire the parser into `chat_completion_responses_api` and derive finish_reason/tool_calls.** In `chat_completion_responses_api` (~2447-2561), replace the text-only extraction (`value.get("output_text")...find_map(text)`) with a call to the item-type parser from item 1. Preserve the existing compaction-only branch (summary blocks with `encrypted_content`) unchanged and ordered BEFORE the new logic. Then:
      - Build the response via `openai_response_from_text(&model, text, prompt_tokens, completion_tokens, finish_reason)` where `finish_reason` = `"tool_calls"` if tool_calls non-empty, else `"length"` if `incomplete_max_tokens`, else `"stop"`.
      - After construction, if tool_calls non-empty insert `message.extra["tool_calls"] = Array(tool_calls)`; if reasoning present insert `message.extra["reasoning_content"] = String(reasoning)`.
      - A function_call-only reply (empty text) must still produce a non-empty, valid turn: content may be empty string, tool_calls populated, finish_reason `"tool_calls"`. Keep the existing "2xx but no extractable content" WARN but only when text is empty AND tool_calls empty AND no compaction.
      Files: `crates\ai-gateway\src\providers\bedrock.rs`
      Verify: `cargo check -p ai-gateway --all-targets --manifest-path "<worktree>\Cargo.toml"` clean; existing `test_responses_compaction_only_surfaces_summary_block` and `test_versioned_gpt_55_uses_responses_api` still pass via `cargo test -p ai-gateway --manifest-path "<worktree>\Cargo.toml" test_responses test_versioned_gpt_55`.

- [ ] 3. **Add Bedrock Responses output-translation tests (wiremock).** Next to `test_responses_compaction_only_surfaces_summary_block` (~4291) in `bedrock.rs` tests, using the same `MockServer` + `create_api_key_mode_provider_for_base_url` + `create_test_chat_request(false)` pattern, model `openai.gpt-5.6-sol`:
      - `test_responses_function_call_only_maps_tool_calls`: output = single `function_call`, no text → assert `message.extra["tool_calls"]` has one entry with `id`/`type:"function"`/`function.name`/`function.arguments`, content empty, `finish_reason == "tool_calls"`.
      - `test_responses_text_and_function_call`: output = a `message` (output_text) + a `function_call` → assert both text content and one tool_call, finish_reason `tool_calls`.
      - `test_responses_multiple_function_calls`: two `function_call` items → assert two tool_calls preserving order and distinct ids.
      - `test_responses_message_concatenates_all_output_text_parts`: a `message` with multiple `output_text` parts → assert full concatenation (not just the first).
      - `test_responses_incomplete_max_output_tokens_maps_length`: `status:"incomplete"`, `incomplete_details.reason:"max_output_tokens"`, text present → assert `finish_reason == "length"`.
      - `test_responses_reasoning_not_in_content`: output = `reasoning` item + `function_call` → assert `content` empty, `message.extra["reasoning_content"]` set, tool_calls present.
      - `test_responses_streaming_replay_carries_tool_call_deltas`: call `provider.chat_completion` (function_call-only), then pass the resulting `response` through `crate::gateway::handlers::streaming_chunks_from_response(&response)` and assert the chunk sequence contains a tool_call header delta (`index:0`, `id`, `type:"function"`, `function.name`), an arguments delta, and a terminal chunk with `finish_reason == "tool_calls"`. (Confirm `streaming_chunks_from_response` is reachable from the bedrock test module; if visibility blocks it, place this one test in `gateway/handlers.rs` tests instead and construct the `OpenAIResponse` directly.)
      Files: `crates\ai-gateway\src\providers\bedrock.rs` (and possibly `crates\ai-gateway\src\gateway\handlers.rs` for the streaming-replay test)
      Verify: `cargo test -p ai-gateway --manifest-path "<worktree>\Cargo.toml" responses_` — all new tests pass.

- [ ] 4. **Gate `promote_reasoning_to_content` on tool_calls (task 2).** In `router.rs` `promote_reasoning_to_content` (~6491-6502), after the content-empty check, early-return `false` when the choice has a non-empty `message.extra["tool_calls"]` array (reuse the same `has_tool_calls` pattern used in `reasoning_only_turn` ~6479-6486). Prefer also gating on "tools were present in the request" IF the request/tools context is cleanly in scope at this function; if not cleanly available, the tool_calls guard is sufficient and the plain-text path is unchanged. Keep current behavior for plain-text turns (no tool_calls, empty content → promote).
      Files: `crates\ai-gateway\src\router\router.rs`
      Verify: `cargo check -p ai-gateway --all-targets --manifest-path "<worktree>\Cargo.toml"` clean.

- [ ] 5. **Add reasoning-not-promoted-on-tool-calls test.** In `router.rs` tests, add `promote_reasoning_to_content_skips_when_tool_calls_present`: build an `OpenAIResponse` with empty content, `message.extra["reasoning_content"]` set, and a valid `message.extra["tool_calls"]` entry; assert `Router::promote_reasoning_to_content(&mut r)` returns `false` and `content` stays empty. Add a companion `promote_reasoning_to_content_still_promotes_plain_text` (empty content, reasoning present, no tool_calls) asserting it returns `true` and content becomes the reasoning text. (Match the visibility used by existing `router.rs` unit tests for these associated fns.)
      Files: `crates\ai-gateway\src\router\router.rs`
      Verify: `cargo test -p ai-gateway --manifest-path "<worktree>\Cargo.toml" promote_reasoning_to_content` — both tests pass.

- [ ] 6. **Skip memory footer on tool_calls turns (task 3).** In `gateway/handlers.rs`, add a small private helper, e.g. `fn is_tool_call_turn(response: &OpenAIResponse) -> bool` (true when first choice `finish_reason == "tool_calls"` OR `message.extra["tool_calls"]` is a non-empty array). At each buffered memory-footer emit site (~2404, ~2730, ~2814, ~2951), change `if let Some(suffix) = memory_suffix.as_deref()` to also require `!is_tool_call_turn(&response)` (the relevant `response` is in scope at every buffered site). For the pass-through relay site, apply the same guard if a buffered `response` is available there; if the relay path has no buffered response (true streaming), leave it unchanged (that path is not the Bedrock buffered path and does not regress). Preserve footer behavior for plain-text turns and the terminal/`[DONE]` ordering otherwise.
      Files: `crates\ai-gateway\src\gateway\handlers.rs`
      Verify: `cargo check -p ai-gateway --all-targets --manifest-path "<worktree>\Cargo.toml"` clean.

- [ ] 7. **Add memory-footer-ordering test.** In `gateway/handlers.rs` tests, add `memory_footer_skipped_on_tool_call_turn`: build an `OpenAIResponse` whose first choice has `finish_reason == "tool_calls"` and a valid `tool_calls` entry; assert `is_tool_call_turn(&response)` is `true` (and, if a chunk-assembly helper is unit-testable, assert the emitted chunk list for such a response ends at the terminal `tool_calls` finish chunk with no trailing `chatcmpl-memory-` content chunk). Add `memory_footer_emitted_on_plain_text_turn`: a plain `finish_reason == "stop"` response → `is_tool_call_turn` is `false`. Keep this test at the helper level if the full SSE stream is not unit-testable with the existing setup.
      Files: `crates\ai-gateway\src\gateway\handlers.rs`
      Verify: `cargo test -p ai-gateway --manifest-path "<worktree>\Cargo.toml" memory_footer` — tests pass.

- [ ] 8. **Add startup WARN when admin auth is disabled (task 4).** In `gateway/mod.rs` where admin routes are mounted (~437-466): extend the destructured snapshot to also read `cfg.admin.auth.enabled` (e.g. `admin_auth_enabled`). Inside the `if admin_enabled { ... }` block, after mounting, emit a clear `tracing::warn!` when `!admin_auth_enabled`, stating admin endpoints are UNAUTHENTICATED and reachable by anyone who can reach the port, and that enabling `admin.auth.enabled` with username/password env vars secures them. Do NOT change any config default or `#[serde(default)]`. To make it testable, factor the single-line decision into a tiny pure helper, e.g. `pub(crate) fn admin_auth_warning(admin_enabled: bool, admin_auth_enabled: bool) -> Option<&'static str>` (returns `Some(msg)` only when `admin_enabled && !admin_auth_enabled`), and have the mount block log the returned message.
      Files: `crates\ai-gateway\src\gateway\mod.rs`
      Verify: `cargo check -p ai-gateway --all-targets --manifest-path "<worktree>\Cargo.toml"` clean.

- [ ] 9. **Add admin-auth-disabled WARN test.** In `gateway/mod.rs` tests, add unit tests on the `admin_auth_warning` helper: `admin_auth_warning_present_when_disabled` (`admin_enabled=true, admin_auth_enabled=false` → `Some`), `admin_auth_warning_absent_when_enabled` (`true,true` → `None`), and `admin_auth_warning_absent_when_admin_disabled` (`false,false` → `None`). If feasible with the existing tracing-capture pattern (`CapturedWriter` + `tracing::subscriber::with_default` as used in `config/validation.rs` ~1069-1075, 1748-1775), also assert the WARN text is actually emitted by driving the helper + log line under a capturing subscriber; otherwise the pure-helper assertions are sufficient.
      Files: `crates\ai-gateway\src\gateway\mod.rs`
      Verify: `cargo test -p ai-gateway --manifest-path "<worktree>\Cargo.toml" admin_auth_warning` — tests pass.

- [ ] 10. **Verify admin-panel parity (task 4, verify-only).** Confirm `crates\ai-gateway\src\admin\static\index.html` still contains the `admin-auth-enabled` checkbox (~345), `admin-user-env`/`admin-pass-env` inputs (~351/355), the config-load wiring (~3902 reads `cfg.admin.auth.enabled/username_env/password_env`), and the save wiring (~4072 writes `cfg.admin.auth`). No HTML change expected; if any of these are missing, add the control/wiring to match. (This is static HTML — no cargo verification; a `grep` confirmation of the IDs is acceptable here because the toggle is pre-existing UI, not program behavior.)
      Files: `crates\ai-gateway\src\admin\static\index.html` (only if a gap is found)
      Verify: confirm the four IDs/wirings present; no build impact.

- [ ] 11. **Final clean-build and full test gate.** Run the whole crate check and test suite against the worktree manifest and fix any remaining warnings/failures (no blanket `#[allow]`; remove dead code or `#[cfg(test)]`-gate helpers). Clean up any temp artifacts created during verification.
      Files: (none — verification only)
      Verify:
      ```
      cargo check -p ai-gateway --all-targets --manifest-path "c:\Fbn_Dev_Apps\OBEY API Router\.worktrees\bedrock-responses-toolcalls\Cargo.toml"
      cargo test  -p ai-gateway            --manifest-path "c:\Fbn_Dev_Apps\OBEY API Router\.worktrees\bedrock-responses-toolcalls\Cargo.toml"
      ```
      Expected: zero errors, zero warnings; all tests (including the new ones) pass.

---

## Full list of tests to add

1. `test_responses_function_call_only_maps_tool_calls` (bedrock.rs)
2. `test_responses_text_and_function_call` (bedrock.rs)
3. `test_responses_multiple_function_calls` (bedrock.rs)
4. `test_responses_message_concatenates_all_output_text_parts` (bedrock.rs)
5. `test_responses_incomplete_max_output_tokens_maps_length` (bedrock.rs)
6. `test_responses_reasoning_not_in_content` (bedrock.rs)
7. `test_responses_streaming_replay_carries_tool_call_deltas` (bedrock.rs or handlers.rs)
8. `promote_reasoning_to_content_skips_when_tool_calls_present` (router.rs)
9. `promote_reasoning_to_content_still_promotes_plain_text` (router.rs)
10. `memory_footer_skipped_on_tool_call_turn` (handlers.rs)
11. `memory_footer_emitted_on_plain_text_turn` (handlers.rs)
12. `admin_auth_warning_present_when_disabled` / `..._absent_when_enabled` / `..._absent_when_admin_disabled` (gateway/mod.rs); optional capturing-subscriber WARN assertion.

## Exact cargo commands (worktree manifest)

```
cargo check -p ai-gateway --all-targets --manifest-path "c:\Fbn_Dev_Apps\OBEY API Router\.worktrees\bedrock-responses-toolcalls\Cargo.toml"
cargo test  -p ai-gateway            --manifest-path "c:\Fbn_Dev_Apps\OBEY API Router\.worktrees\bedrock-responses-toolcalls\Cargo.toml"
```

(Alternatively run with `cwd` set to the worktree root and omit `--manifest-path`.)
