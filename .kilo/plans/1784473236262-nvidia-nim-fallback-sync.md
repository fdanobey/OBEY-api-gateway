# NVIDIA NIM fallback catalog sync

## Goal

Add a small built-in fallback catalog for the NVIDIA hosted API (`https://integrate.api.nvidia.com/v1`) and automate updates when current hosted chat models differ. This tracks hosted API availability, not the self-hosted Certified NIM Support Matrix.

## Verified current state

- `crates/ai-gateway/src/providers/nvidia_nim.rs` delegates `list_models()` directly to the generic OpenAI-compatible `/v1/models` client and has no fallback.
- Gateway `GET /v1/models` in `crates/ai-gateway/src/gateway/handlers.rs` does **not** query providers. It only aggregates model groups and `Provider.manual_models`.
- Admin model discovery in `crates/ai-gateway/src/admin/mod.rs` proxies upstream `/v1/models` verbatim. It has no provider-type parameter and no NVIDIA fallback.
- `manual_models` is the only existing fallback mechanism. It defaults to empty and is merged by the gateway and admin UI.
- The intended initial curated set is:
  - `openai/gpt-oss-120b`
  - `meta/llama-3.1-70b-instruct`
  - `nvidia/nemotron-3-nano`
- `NVIDIA_API_KEY` is not available locally, so current hosted reachability cannot be verified in this planning session. The first workflow dry run must be treated as the authoritative check.
- The worktree already has unrelated edits, including the new `Model.supports_vision` field. Preserve them and only stage files from this implementation.

## Decisions

- Catalog: NVIDIA hosted API only.
- Breadth: three-model curated subset, not a full catalog mirror.
- Runtime: built-in fallback applies when NVIDIA live discovery errors or returns an empty list; configured `manual_models` remain additive overrides.
- Ranking: use live `/v1/models`, chat probes, and best-effort `build.nvidia.com` metadata. Ranking is deterministic.
- Safety: replace a missing curated model only after two consecutive scheduled runs confirm an actionable absence. HTTP 429 means available; 5xx/timeouts are transient and reset confirmation.
- Automation: weekly Windows GitHub Actions workflow opens/updates one PR.
- Tooling: PowerShell script plus workflow; no new Cargo dependencies.

## Implementation tasks

### 1. Add the central fallback catalog

Edit `crates/ai-gateway/src/providers/nvidia_nim.rs`:

1. Add `NimFallbackModel` with `id`, `owned_by`, `supports_vision`, `context_window`, `max_completion_tokens`, and `source_url`.
2. Add a clearly delimited generated block:

   `// BEGIN NVIDIA NIM FALLBACK MODELS`

   `pub const NVIDIA_NIM_FALLBACK_MODELS: &[NimFallbackModel] = &[...];`

   `// END NVIDIA NIM FALLBACK MODELS`

3. Seed it with the three IDs above. Use publisher-qualified IDs. Only populate capability values verified from NVIDIA model cards; use `None` rather than guessing.
4. Add `pub fn fallback_models() -> Vec<Model>` to map entries into the existing `Model` shape, including `supports_vision`.
5. Change `NvidiaNIMProvider::list_models()`:
   - return live models when non-empty;
   - log a warning and return `fallback_models()` on upstream error;
   - log a warning and return `fallback_models()` on an empty live response.
6. Add wiremock-backed tests for non-empty passthrough, empty fallback, and error fallback. Assert exact fallback IDs and metadata.

### 2. Surface fallback through both discovery paths

Edit `crates/ai-gateway/src/gateway/handlers.rs`:

1. After configured `manual_models` are inserted, detect every configured provider with `provider_type == "nvidia_nim"`.
2. Insert `fallback_models()` for each NVIDIA provider, using the configured provider name as `owned_by` so response ownership remains consistent with manual entries.
3. Preserve the existing `seen_ids` ordering: model groups and explicit manual models are inserted first, then built-ins fill missing IDs.
4. This endpoint has no live provider query, so built-ins are always the final safety net here rather than conditional on a live call.
5. Ensure virtual-key model filtering continues to run after insertion.
6. Add a handler test covering NVIDIA fallback visibility, manual override deduplication, and virtual-key filtering.

Edit `crates/ai-gateway/src/admin/mod.rs` and `crates/ai-gateway/src/admin/static/index.html`:

1. Extend `ProxyModelsParams` with optional `provider_type`.
2. Have the UI include `provider_type` in every `/admin/providers/models` request.
3. In `proxy_provider_models`, when `provider_type == "nvidia_nim"`:
   - parse successful upstream JSON and return it unchanged when `.data` is non-empty;
   - return `{ "object": "list", "data": fallback_models() }` when upstream succeeds with empty data;
   - return the same fallback response when upstream request/status/body parsing fails, while logging the failure.
4. Keep existing verbatim proxy behavior for all other provider types, including Bedrock.
5. Add admin tests for NVIDIA empty/error fallback and non-NVIDIA status passthrough.

### 3. Add the deterministic sync script

Create `scripts/sync-nvidia-nim-fallback.ps1` with parameters:

- `-TopN` (default `3`)
- `-DryRun`
- `-StatePath` (workflow supplies a persisted path)
- `-FromCache` (optional offline fixture/catalog JSON)

Behavior:

1. Require `NVIDIA_API_KEY` unless `-FromCache` is supplied. Exit `2` when missing.
2. Fetch every `/v1/models` page. Support both `has_more` and token-based pagination; abort without mutation if a response indicates another page but provides no usable continuation token.
3. Extract and deduplicate `.data[].id`.
4. Pre-filter obvious non-chat endpoint families (`embed`, `rerank`, `reward`, `ocr`, `parse`, `asr`, `tts`, `stt`, `retriever`). Do not exclude generic `vision`/`vl` models because some accept `/chat/completions`.
5. Probe candidates with a minimal non-streaming chat request (`max_tokens: 8`, 30-second timeout):
   - 200 with `choices` or `model`: available;
   - 429: available;
   - 401/403: hard auth failure, exit `3` with no edit;
   - 404: actionable absent;
   - 400/422: non-chat/skipped;
   - 5xx/network/timeout: transient, never actionable.
6. Fetch `https://build.nvidia.com/{id}` metadata for available IDs, best effort. Parse YAML front matter and capability/specification headings only. A failed card fetch leaves metadata unknown and does not remove the candidate.
7. Rank deterministically by:
   - model-card updated/release timestamp descending;
   - capability count descending (`function calling`, `structured output`, `reasoning`, `vision`);
   - active or total parameter count descending when parseable;
   - ID ascending as final tie-breaker.
8. Current fallback IDs remain pinned while available. Ranking selects replacements only for confirmed-absent slots; this avoids replacing all three merely because newer models appear.
9. Load the persisted state file. For each current ID:
   - actionable absence after prior actionable absence: confirmation count 2, eligible for replacement;
   - available or transient: reset count to 0;
   - first actionable absence: record count 1, no source edit.
10. Replace confirmed-absent slots with highest-ranked available IDs not already in the curated set. Never shrink below `TopN`; if insufficient viable replacements exist, report failure without editing.
11. Rewrite only the delimited generated block in `nvidia_nim.rs`, update a generated provenance timestamp/comment, and preserve all surrounding code.
12. `-DryRun` never edits source but still emits the proposed diff and updated state to a separate output path supplied by the workflow.
13. Emit one compact JSON summary to stdout.
14. Exit codes: `0` no source change; `1` source rewritten/proposed; `2` missing key; `3` auth or incomplete-catalog hard failure; `4` script error.
15. Add Pester-free offline self-tests inside the script behind `-FromCache`, or provide committed JSON fixtures under `scripts/fixtures/` and a `-ValidateFixtures` switch. Cover ranking, two-run confirmation, transient reset, generated-block replacement, and idempotence without network access.

### 4. Persist two-run confirmation correctly

GitHub-hosted runners are ephemeral, so `scripts/.cache` alone cannot implement consecutive-run confirmation.

Create `.github/workflows/sync-nvidia-nim-fallback.yml`:

1. Trigger weekly Monday at `06:17 UTC` and via `workflow_dispatch` with `dry_run` input.
2. Use `windows-latest`, `actions/checkout@v4`, and full history.
3. Permissions: `contents: write`, `pull-requests: write`.
4. Restore confirmation state with `actions/cache@v4` from a workflow-owned directory such as `$RUNNER_TEMP/nim-sync-state` using:
   - a new unique primary key containing `github.run_id` so the updated state is saved every run;
   - restore prefix `nim-fallback-state-` so the newest prior state is loaded.
5. Pass the restored state path to the script. Never commit this operational state.
6. Run the script and capture its exit code/JSON summary without letting PowerShell terminate before workflow branching.
7. Exit 0: publish step summary; no PR.
8. Exit 1 and `dry_run == true`: publish proposed changes; do not commit/push.
9. Exit 1 and normal scheduled/manual run:
   - create/reset deterministic branch `chore/nvidia-nim-fallback-sync`;
   - commit only `crates/ai-gateway/src/providers/nvidia_nim.rs`;
   - force-with-lease update that automation-owned branch if it already exists;
   - create a PR if none exists, otherwise update the existing PR body;
   - title `chore(nvidia-nim): sync built-in fallback models`.
10. Exit 2/3/4: fail the workflow after writing a warning and summary; never open/update a PR.
11. Use repository secret `NVIDIA_API_KEY`. Document the required secret.
12. Add `scripts/.cache/` to `.gitignore` only if the local script uses it; workflow confirmation state remains under runner temp.

### 5. Update operator-facing documentation

Edit only existing docs/UI:

- `crates/ai-gateway/src/admin/static/index.html`: explain that `manual_models` are optional overrides and NVIDIA has a maintained built-in fallback.
- `crates/ai-gateway/config.example.yaml`: add a commented NVIDIA provider example; do not prefill `manual_models`.
- `wiki/Providers.md`: document fallback activation, manual override behavior, sync workflow, and required `NVIDIA_API_KEY` repository secret.
- Do not create a changelog or new general documentation file.

### 6. Validate

1. Preserve unrelated working-tree edits; inspect `git diff` before and after implementation.
2. Run offline script fixture validation twice to prove first absence only records state and second absence proposes replacement; run a transient fixture between them to prove reset.
3. Run `cargo fmt --check -p ai-gateway`.
4. Run focused tests:
   - `cargo test -p ai-gateway --lib providers::nvidia_nim`
   - targeted admin/gateway model-list tests added above.
5. Run `cargo test -p ai-gateway` with `CARGO_BUILD_JOBS=1`.
6. Run `cargo build --release -p ai-gateway` with `CARGO_BUILD_JOBS=1`.
7. If a key is available, run the script with `-DryRun` and verify no source mutation. If unavailable, report live verification as pending and rely on the first workflow dry run.
8. After merge, configure `NVIDIA_API_KEY` and run `workflow_dispatch` with `dry_run=true` before enabling reliance on the weekly update.

## Failure behavior

- 429 confirms availability.
- 5xx, timeout, DNS, and metadata-page failures are transient and cannot evict a model.
- 401/403 or incomplete pagination aborts the run without editing.
- Missing web metadata degrades ranking to API/probe data and lexical tie-breakers.
- A current model is never replaced solely because a newer model ranks higher.
- Existing open automation PR is updated rather than duplicated.

## Out of scope

- Certified/self-hosted NIM support matrix mirroring.
- Rewriting users' saved `manual_models`.
- Runtime chat health probes in request handling.
- New Rust dependencies.
- Release automation changes.
