# Recheck Verification — fix/bedrock-compaction-toolcalls

Worktree: `.worktrees/bedrock-compaction-toolcalls`
Branch HEAD: `c9f2eb3` on top of `69b7d22` (base `62507cb` = master/0.6.8)
Branch touches only: `crates/ai-gateway/src/providers/bedrock.rs` (FIX 1) and `crates/ai-gateway/src/memory/extraction_adapter.rs` (FIX 2).

## Commands run

### 1. cargo check (zero errors AND zero warnings required)
```
cargo check -p ai-gateway --all-targets --manifest-path "c:\Fbn_Dev_Apps\OBEY API Router\.worktrees\bedrock-compaction-toolcalls\Cargo.toml"
```
Result: PASS. 0 errors, 0 warnings. (Filtering compiler output for `warning|error` returned no lines; the only emitted line was `Finished dev profile`.)

### 2. cargo test — targeted at both fixes
```
cargo test -p ai-gateway --manifest-path "c:\Fbn_Dev_Apps\OBEY API Router\.worktrees\bedrock-compaction-toolcalls\Cargo.toml" --lib -- providers::bedrock memory::extraction_adapter
```
Result: PASS. `test result: ok. 121 passed; 0 failed; 0 ignored; 2239 filtered out`. Includes every FIX 1 and FIX 2 test (listed below).

### 3. cargo test — full suite
```
cargo test -p ai-gateway --manifest-path "c:\Fbn_Dev_Apps\OBEY API Router\.worktrees\bedrock-compaction-toolcalls\Cargo.toml"
```
Result: 2360 tests compiled and ran. 4 failures, all in `providers::nvidia_nim` fallback catalog / models-endpoint code that this branch does NOT touch:
- `providers::nvidia_nim::tests::fallback_catalog_has_expected_models_and_metadata`
- `providers::nvidia_nim::tests::list_models_falls_back_on_live_catalog_error`
- `providers::nvidia_nim::tests::list_models_falls_back_on_empty_live_catalog`
- `gateway::tests::test_models_endpoint_surfaces_nvidia_nim_fallback_without_manual_models`

These are pre-existing and unrelated: the failure is a hardcoded fallback-model-list data mismatch in `providers/nvidia_nim.rs` (left/right model-id sets differ), introduced by master commit `6b1aa48 chore(nvidia-nim): sync built-in fallback models`. The branch diff (`62507cb..HEAD`) modifies only `bedrock.rs` and `extraction_adapter.rs`, so these cannot be a regression from either fix. (The main-workspace master checkout currently fails to even compile its lib tests — a `Deserialize` bound error on `Vec<ProviderAttempt>` — so the worktree is in a cleaner state than master.)

Cargo.lock copy step was not needed — no aws-config/rustc version mismatch occurred.

No source code was modified in this verification step.

## New tests — FIX 1 (bedrock: keep tool_calls on compaction+function_call turns)
`crates/ai-gateway/src/providers/bedrock.rs`:
- `test_responses_function_call_with_compaction_block_keeps_tool_calls`
- `test_responses_text_plus_compaction_plus_function_call`
- `test_responses_compaction_only_still_stops`
- `test_responses_streaming_replay_compaction_plus_function_call_carries_tool_call_deltas`

## New tests — FIX 2 (memory: lenient extraction parse, degrade garbage to empty)
`crates/ai-gateway/src/memory/extraction_adapter.rs`:
- `parse_response_prose_wrapped_array_extracts_candidates`
- `parse_response_single_object_is_wrapped`
- `parse_response_wrapper_key_is_unwrapped`
- `parse_response_unparseable_prose_returns_empty`

## Pure compaction-only behavior
Unchanged — confirmed green by `test_responses_compaction_only_still_stops` and `test_responses_compaction_only_surfaces_summary_block`: a compaction-only turn (no function_call) still stops and surfaces the summary block exactly as before.
