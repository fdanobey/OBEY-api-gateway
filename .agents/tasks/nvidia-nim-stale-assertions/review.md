# Structural rewrite of NVIDIA NIM fallback catalog test assertions

The branch `chore/nvidia-nim-fallback-sync` rewrites four value-locked test assertion sites so an automated sync that rewrites the curated NVIDIA NIM model list (changing both the entries and the list size) no longer breaks the suite. Value-locked checks — `len() == 4`, hard-coded ids like `nvidia/nemotron-3-nano`, per-model `context_window`/`supports_vision` literals — are replaced with invariants tied to the production source const `NVIDIA_NIM_FALLBACK_MODELS`: non-empty, count equals the const length, per-index id/owned_by equality against the const, well-formed `owner/model` ids, non-empty `owned_by`, `object == "model"`, and a `source_url` domain check. The change is confined to test code in exactly the two files named in the task; the sync script, CI workflow, bedrock/router fixtures, and all production code are untouched.

Watch for: the mapping test dropped its `context_window`/`supports_vision`/`max_completion_tokens` coverage and does not re-assert those fields are mapped from the const into the `Model` (likely — partial coverage reduction, non-blocking); the three dropped attribute assertions were genuinely value-locked and had to go (confirmed).

**Verdict**: APPROVED

## High-level view

All four sites now derive their expectations from the production `pub const NVIDIA_NIM_FALLBACK_MODELS` rather than literals. No fixed count and no hard-coded model id survives anywhere in the rewritten assertions — counts use `NVIDIA_NIM_FALLBACK_MODELS.len()` and ids compare against `NVIDIA_NIM_FALLBACK_MODELS[i].id`. This is exactly the invariant the task requires: resilient to an arbitrary catalog rewrite.

Each test's original intent survives in structural form. Ordering tests assert `models[i].id == NVIDIA_NIM_FALLBACK_MODELS[i].id` across the full list; the mapping test checks per-index id and owned_by plus a const-level `source_url` domain assertion; a shared `#[cfg(test)] assert_model_well_formed` helper enforces id shape, non-empty owned_by, and object tag. None of the tests was gutted into a trivially-true existence check — the weakest new assertion (`!models.is_empty()`) is always paired with the per-index/const-length checks.

The one coverage give-up is in `fallback_catalog_has_expected_models_and_metadata`: the old test asserted that `context_window` and `supports_vision` round-trip from the const through `fallback_models()` into the mapped `Model`. Those specific-value asserts were correctly removed as value-locked, but the mapping test no longer verifies these fields pass through at all. That is a real but minor reduction in mapping coverage, not a blocker.

Scope is clean. The diff is two files, both test-only edits; the `openai/gpt-oss-120b` literal no longer appears in any assertion (it survives only inside the const definition, which was not edited), and bedrock/router fixtures are absent from the diff entirely.

<details>
<summary>Issues (1)</summary>

1. **Mapping test no longer verifies attribute round-trip** (likely, non-blocking) — `fallback_catalog_has_expected_models_and_metadata` checks `id`/`owned_by` per index but no longer asserts `context_window`, `supports_vision`, or `max_completion_tokens` are mapped from the const into the `Model`. Consider adding per-index equality for these fields against `NVIDIA_NIM_FALLBACK_MODELS[i]` to restore the mapping-correctness coverage without reintroducing value-locking.

</details>

<details>
<summary>Details</summary>

## No fixed count and no hard-coded id survives

Every count assertion is now relative to the const. In `gateway/mod.rs` the `assert_eq!(ids.len(), 4)` became `assert_eq!(ids.len(), NVIDIA_NIM_FALLBACK_MODELS.len())`, and the four `ids.contains("...")` literals became a loop over the const asserting each `model.id` is surfaced. The two `nvidia_nim.rs` ordering tests already used `NVIDIA_NIM_FALLBACK_MODELS.len()` for the count and now also read index expectations from the const — `models[0].id == NVIDIA_NIM_FALLBACK_MODELS[0].id` replaces the `openai/gpt-oss-120b` literal, and the `models[2].id == "nvidia/nemotron-3-nano"` literal is replaced by a full-list `for (i, model) ... assert_eq!(model.id, NVIDIA_NIM_FALLBACK_MODELS[i].id)`. The mapping test's exact-id `vec![...]` comparison is gone, replaced by per-index id/owned_by equality against the const. A grep confirms no model-id string literal remains in any assertion; the only occurrences of names like `moonshotai/kimi-k3` are inside the const definition itself (lines 25-47), which the diff does not touch.

## Original intent preserved per test

The ordering tests (`list_models_falls_back_on_empty_live_catalog`, `list_models_falls_back_on_live_catalog_error`) existed to prove the fallback is returned in source order on empty/errored live catalogs. That intent is intact and in fact strengthened: the error-path test previously only checked a single element (`models[2]`), and now verifies the entire ordering via the indexed loop. The endpoint test's intent — the `/models` endpoint surfaces the fallback catalog when no manual models and no live upstream exist — is preserved via the "every const id is surfaced" loop plus an `owner/model` shape check on each surfaced id.

The shared `assert_model_well_formed` helper is correctly gated as test-only, so it does not trip the repo's clean-build / no-dead-code requirement.

## The dropped attribute assertions were value-locked

The removed lines — `models[0].context_window == Some(128_000)`, `kimi.supports_vision`, `kimi.context_window == Some(1_000_000)`, and "every non-kimi model is not vision-capable" — are precisely the kind of value-locked assertions the task requires removing. They encode per-model attribute values and a vision-capability distribution that the sync overwrites. Against the current const all three entries are `supports_vision: true`, so the old "only kimi is vision-capable" invariant is already false; keeping it would re-break on the next sync. Removal is correct.

The residual gap is that the mapping test now proves id/owned_by map through from the const but is silent on `context_window`/`supports_vision`/`max_completion_tokens`. `fallback_models()` does map all of these (lines 52-64), so a structural per-index assert — e.g. `assert_eq!(model.context_window, NVIDIA_NIM_FALLBACK_MODELS[i].context_window)` — would restore mapping-correctness coverage for those fields without hard-coding any value and without breaking on a sync. This is the single improvement worth making; it does not block the merge because the original intent (catalog shape, ordering, mapping, well-formedness) otherwise survives and nothing was reduced to a trivially-true check.

## Scope and evidence

The diff is `--name-only` exactly `crates/ai-gateway/src/gateway/mod.rs` and `crates/ai-gateway/src/providers/nvidia_nim.rs`, both test-only edits (`mod tests` and the `#[cfg(test)]`-reachable endpoint test). No sync script, CI workflow, or non-test production code appears. The bedrock.rs/router.rs `openai.gpt-oss-120b` fixtures are not in the diff and were left alone. The commit message records the verification run on Windows PowerShell: `RUSTFLAGS='-D warnings' cargo check -p ai-gateway --all-targets` clean with zero warnings, `cargo test -p ai-gateway --lib providers::nvidia_nim` 6 passed, and the endpoint test 1 passed. No articulable doubt remains that would justify a spot-check; the recorded evidence matches the diff (the asserted model names in the commit message — moonshotai/kimi-k3, meta/muse-glimmer-30b, google/diffusiongemma-26b-a4b-it — match the current const).

</details>

<details>
<summary>File map</summary>

- `crates/ai-gateway/src/providers/nvidia_nim.rs` — added test-only `assert_model_well_formed` helper; rewrote the mapping test and two ordering tests to assert against `NVIDIA_NIM_FALLBACK_MODELS` structurally; dropped value-locked context_window/vision asserts.
- `crates/ai-gateway/src/gateway/mod.rs` — rewrote `test_models_endpoint_surfaces_nvidia_nim_fallback_without_manual_models` to drop `len() == 4` and literal id `contains` checks in favor of const-length + per-const-id surfaced + owner/model shape checks.

Full diff: `git diff master...chore/nvidia-nim-fallback-sync`

</details>
