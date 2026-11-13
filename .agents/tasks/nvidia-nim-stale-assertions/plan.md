# Implementation Plan: Refactor stale NVIDIA NIM catalog assertions

## Context (verified against the checked-out branch)

- The sync has **already rewritten** `NVIDIA_NIM_FALLBACK_MODELS` in
  `crates/ai-gateway/src/providers/nvidia_nim.rs` to 3 entries:
  `moonshotai/kimi-k3`, `meta/muse-glimmer-30b`, `google/diffusiongemma-26b-a4b-it`.
  This is why the 4 value-locked tests fail: they hard-code the old IDs (`openai/gpt-oss-120b`,
  `meta/llama-3.1-70b-instruct`, `nvidia/nemotron-3-nano`) and `len() == 4`.
- `NimFallbackModel` fields (all `&'static str` / primitives):
  `id`, `owned_by`, `supports_vision`, `context_window`, `max_completion_tokens`, `source_url`.
- `fallback_models()` maps the const 1:1 into `Vec<Model>`, preserving order; `Model` has
  `id`, `object`, `owned_by`, `created`, `context_window`, `max_completion_tokens`, `supports_vision`.
- The provider's `list_models()` returns the live catalog when non-empty, else `fallback_models()`.

Decisions already fixed by the user and applied here: **drop the fixed-count contract and all
specific-ID / specific-field-value expectations**; replace with structural invariants that still
provide real coverage. Only test code changes. No sync script, CI, or production changes. Leave the
`bedrock.rs` / `router.rs` `gpt-oss-120b` fixtures untouched.

A small `#[cfg(test)]` helper that validates one `Model` against the shared invariants removes
duplication across the three provider tests and keeps the clean-build rule satisfied (no dead code,
no blanket `#[allow]`).

---

## Plan

- [ ] 1. Add a `#[cfg(test)]` invariant helper inside the existing `mod tests` of
      `crates/ai-gateway/src/providers/nvidia_nim.rs`. It asserts, for a single `Model`:
      id is `owner/model` (exactly one `/`, non-empty owner and model segments); `owned_by`
      non-empty; `object == "model"`; `source_url` is **not available on `Model`**, so the helper
      covers only `Model` fields — `source_url` is validated separately against the const in step 2.
      Signature: `fn assert_model_well_formed(model: &Model)`.
      Files: `crates/ai-gateway/src/providers/nvidia_nim.rs`
      Verify: `cargo check -p ai-gateway --all-targets` — compiles with zero warnings.

- [ ] 2. Rewrite `fallback_catalog_has_expected_models_and_metadata` (currently ~lines 196-224) into
      a structural test. Current code asserts `ids == vec![4 specific IDs]`,
      `models[0].context_window == Some(128_000)`, and a kimi-k3-specific vision block.
      Replacement invariants:
      (a) `let models = fallback_models(); assert!(!models.is_empty());`
      (b) `assert_eq!(models.len(), NVIDIA_NIM_FALLBACK_MODELS.len());` (ties mapping to the source
      const without a hard number);
      (c) for every `i`, `assert_eq!(models[i].id, NVIDIA_NIM_FALLBACK_MODELS[i].id);` and
      `assert_eq!(models[i].owned_by, NVIDIA_NIM_FALLBACK_MODELS[i].owned_by);`
      (mapping is order-preserving and faithful);
      (d) call `assert_model_well_formed(&model)` for each model;
      (e) validate the const directly: for each `NimFallbackModel`, `owned_by` non-empty, id is
      `owner/model`, and `source_url` is non-empty and starts with `https://build.nvidia.com`.
      Remove the `context_window == Some(128_000)` and the kimi-k3-only vision assertions entirely
      (both are value-locked to a specific curated entry).
      Why it survives arbitrary catalog changes: no specific ID, count, or field value is named; all
      checks are over whatever the const contains. Why it still has teeth: it fails if the mapping
      drops/reorders entries, if any id is malformed, if `owned_by` is empty, or if a `source_url`
      stops pointing at the NVIDIA build domain — i.e. the catalog *shape* and *mapping correctness*,
      which was the test's original intent.
      Files: `crates/ai-gateway/src/providers/nvidia_nim.rs`
      Verify: `cargo test -p ai-gateway fallback_catalog_has_expected_models_and_metadata` — passes.

- [ ] 3. Rewrite the two fixed-ID assertions in `list_models_falls_back_on_empty_live_catalog`
      (~lines 272-277: `assert_eq!(models.len(), NVIDIA_NIM_FALLBACK_MODELS.len())` then
      `assert_eq!(models[0].id, "openai/gpt-oss-120b")`). Keep the length-vs-const assertion (it is
      already structural). Replace the `models[0].id == "openai/gpt-oss-120b"` line with an
      order/mapping-consistency check against the source const:
      `assert_eq!(models[0].id, NVIDIA_NIM_FALLBACK_MODELS[0].id);` and
      `assert_model_well_formed(&models[0]);` (guard with `if !models.is_empty()` or assert
      non-empty first). Preserves the original intent: on empty live catalog the provider returns the
      fallback catalog in source order. Survives catalog changes because index 0's id is read from the
      const, not hard-coded; still meaningful because it fails if fallback isn't used or ordering breaks.
      Files: `crates/ai-gateway/src/providers/nvidia_nim.rs`
      Verify: `cargo test -p ai-gateway list_models_falls_back_on_empty_live_catalog` — passes.

- [ ] 4. Rewrite the two fixed-ID assertions in `list_models_falls_back_on_live_catalog_error`
      (~lines 296-301: `assert_eq!(models.len(), NVIDIA_NIM_FALLBACK_MODELS.len())` then
      `assert_eq!(models[2].id, "nvidia/nemotron-3-nano")`). Keep the length-vs-const assertion.
      The `models[2]` reference assumes >= 3 entries, which an arbitrary catalog may not have — so
      first assert non-empty, then assert full-order consistency generically:
      `for i in 0..models.len() { assert_eq!(models[i].id, NVIDIA_NIM_FALLBACK_MODELS[i].id); }`
      and `assert_model_well_formed` on each. This drops the brittle `[2]`/specific-ID expectation
      while preserving intent (on live-catalog error, provider falls back in source order). Survives
      any catalog size/content; still fails if fallback isn't used, mapping is wrong, or order breaks.
      Files: `crates/ai-gateway/src/providers/nvidia_nim.rs`
      Verify: `cargo test -p ai-gateway list_models_falls_back_on_live_catalog_error` — passes.

- [ ] 5. Rewrite `test_models_endpoint_surfaces_nvidia_nim_fallback_without_manual_models` in
      `crates/ai-gateway/src/gateway/mod.rs` (~lines 1585-1591:
      `assert_eq!(ids.len(), 4)` plus four `ids.contains(<specific id>)`). The `/v1/models`
      response has no live upstream here, so it surfaces the fallback catalog. Replacement:
      import the const (`use crate::providers::nvidia_nim::NVIDIA_NIM_FALLBACK_MODELS;` inside the
      test or module test scope) and assert:
      (a) `assert!(!ids.is_empty());`
      (b) `assert_eq!(ids.len(), NVIDIA_NIM_FALLBACK_MODELS.len());` (endpoint exposes exactly the
      fallback catalog, tied to the const, no magic number);
      (c) every expected const id is surfaced:
      `for m in NVIDIA_NIM_FALLBACK_MODELS { assert!(ids.contains(m.id)); }`
      (d) every surfaced id is well-formed `owner/model` (reuse an inline check or assert each
      contains exactly one `/` with non-empty segments).
      Confirm `NVIDIA_NIM_FALLBACK_MODELS` is `pub` and reachable from `gateway::mod` (it is `pub`
      in `providers::nvidia_nim`); add the `use` where the surrounding test module places imports.
      Why it survives: no fixed count, no literal IDs — all derived from the const. Why meaningful:
      fails if the endpoint stops surfacing the fallback catalog, surfaces extra/fewer models, or
      emits malformed IDs. Preserves original intent (endpoint exposes the NVIDIA NIM fallback
      catalog when no manual models are configured).
      Files: `crates/ai-gateway/src/gateway/mod.rs`
      Verify: `cargo test -p ai-gateway test_models_endpoint_surfaces_nvidia_nim_fallback_without_manual_models` — passes.

- [ ] 6. Full clean-build and full-suite verification.
      Files: none (verification only).
      Verify: `cargo check -p ai-gateway --all-targets` reports zero warnings AND
      `cargo test -p ai-gateway` passes with no failures (the 4 previously-failing assertions now
      pass and no `#[allow]` was added).

## Notes / assumptions

- `source_url` lives on `NimFallbackModel`, not on `Model`, so the NVIDIA-domain check is applied to
  the const (step 2e), not inside `assert_model_well_formed`. This is why step 1's helper is scoped
  to `Model` fields only.
- Step 1's helper is `#[cfg(test)]` by virtue of living in `mod tests`, satisfying the
  no-dead-code / no-blanket-`#[allow]` requirement.
- Left untouched per instructions: the sync script, CI workflow, and the `bedrock.rs` / `router.rs`
  `gpt-oss-120b` request-translation fixtures.
