# AWS Bedrock manual fallback catalog refresh

## Goal

Replace the single stale Bedrock fallback list with two compatibility-correct catalogs, route Bedrock SDK chat through Converse, add Mantle Responses and Anthropic Messages adapters, and automate drift detection against AWS model documentation so the fallback stays current.

## Verified current state

- Runtime backup list: `crates/ai-gateway/src/providers/bedrock.rs:659-695` returns `backup_models()`. The constant at lines 660-675 mixes:
  - OpenAI Mantle Responses-only IDs (`openai.gpt-5.5`, `openai.gpt-5.4`) the gateway cannot invoke, because API-key chat posts to `/v1/chat/completions` (bedrock.rs:802, 883).
  - Runtime-only IDs ending `-1:0` for gpt-oss, plus legacy/not-currently-listed Claude 3.x IDs, Nova Pro/Lite, and `amazon.titan-text-express-v1`.
- It is merged into live results unconditionally at bedrock.rs:1134-1146, so stale IDs always surface.
- SDK chat uses InvokeModel with four hand-written translators (Claude/Titan/Jurassic/Command) at bedrock.rs:285-430, and `translate_model_id` only maps legacy OpenAI aliases at bedrock.rs:255-282. Current Claude/Nova/Llama/DeepSeek models are not routable through SDK mode.
- Admin UI list: `crates/ai-gateway/src/admin/static/index.html:1164-1211` defines `BEDROCK_BUILTIN_MODELS`, a flat array mixing Mantle IDs, runtime IDs, Responses-only IDs, Messages-only Claude IDs, unverified Claude 4.x IDs, and legacy entries. The UI adds these unconditionally at lines 1957, 2025 before live results, and the `fetchModelsForBedrock` path at line 2007 queries both `/v1/models` and `/openai/v1/models`.
- `/v1/models` aggregation in `crates/ai-gateway/src/gateway/handlers.rs:3222-3320` does not query providers; it only reads `model_groups` and `Provider.manual_models`. Bedrock has no `nvidia_nim`-style unconditional built-in injection here, so the runtime/admin fallbacks are the only Bedrock safety net.
- Admin proxy in `crates/ai-gateway/src/admin/mod.rs:1167-1309` passes upstream responses through verbatim for non-NVIDIA provider types; for Bedrock it relies on the UI-side merge of `BEDROCK_BUILTIN_MODELS`.
- Tests pinning stale content: `test_list_models_falls_back_to_backup_on_failure` (bedrock.rs:1692) and `test_backup_models_includes_new_openai_models` (bedrock.rs:1719) assert `openai.gpt-5.5`/`openai.gpt-5.4` are present.
- Existing precedent for the verifier: `scripts/sync-nvidia-nim-fallback.ps1` and `.github/workflows/sync-nvidia-nim-fallback.yml` already implement a docs-derived, weekly, two-run-confirmation, force-with-lease PR workflow with a JSON summary and exit-code semantics. The Bedrock verifier should mirror this shape.
- `model_supports_reasoning` at bedrock.rs:84-114 already pattern-matches Claude 3.5 Sonnet v2+, Claude 3 Opus, and `claude-4`/`claude-5`; the proptest known-reasoning set at bedrock.rs:1992-2006 only lists synthetic Claude IDs and needs current documented ones once Responses/Messages are wired.

## AWS documentation snapshot (2026-07-19)

Verified against the Bedrock User Guide endpoint availability, API compatibility, model cards, and model lifecycle pages. Key findings:

- OpenAI `openai.gpt-5.5`, `openai.gpt-5.4`, and the new `openai.gpt-5.6-sol/terra/luna` are Mantle **Responses-only** (`https://bedrock-mantle.{region}.api.aws/openai/v1`). They do not support Chat Completions, Converse, or Invoke.
- `openai.gpt-oss-120b`/`openai.gpt-oss-20b` have distinct IDs per endpoint: Mantle Chat uses `openai.gpt-oss-120b`/`openai.gpt-oss-20b`, runtime uses `openai.gpt-oss-120b-1:0`/`openai.gpt-oss-20b-1:0`.
- AWS lists no Claude model as OpenAI Chat Completions-compatible. Claude models that are on Mantle use **Messages**; runtime-capable Claude models use Invoke/Converse. Mantle-present Claude: Sonnet 5, Fable 5, Mythos 5 (preview), Haiku 4.5, Opus 4.7, Opus 4.8. Runtime-present: Opus 4.1/4.5/4.6, Sonnet 4/4.5/4.6, Haiku 4.5, Claude 3 Haiku, Claude 3.5 Haiku.
- Amazon: Nova 2 Lite is current and Active; Nova Premier is Legacy; Nova Pro/Lite/Micro are Active; Titan Text Express/Lite are absent from the current catalog and endpoint pages.
- AI21 Jamba 1.5 Large/Mini and Cohere Command R/R+ are Legacy and runtime-only.
- Meta Llama 3.x/4 models are runtime-only and Invoke/Converse compatible; `meta.llama3-1-405b-instruct-v1:0` is Legacy.
- DeepSeek V3.2 and V3.1 support Chat Completions on Mantle with ID `deepseek.v3.2`/`deepseek.v3.1`; DeepSeek-R1 is runtime-only.
- Mistral Large 3 supports Chat; Mistral 7B/Large/Small/Mixtral legacy are runtime-only.
- Qwen3 32B has endpoint-specific IDs: `qwen.qwen3-32b-v1:0` runtime, `qwen.qwen3-32b` Mantle.
- MiniMax M2.x, Google Gemma 3, NVIDIA Nemotron, Moonshot Kimi K2.5, Z.AI GLM, Writer Palmyra Vision, and xAI Grok 4.3 support Chat on Mantle per the compatibility table.
- `ListFoundationModels` returns `modelSummaries[].modelLifecycle.status` ACTIVE|LEGACY; inference profiles are not included and require a separate `list-inference-profiles` call. The verifier uses documentation first, and `modelLifecycle` only as a pass-through quality check when AWS credentials are available.

## Decisions

- Split the fallback into two compatibility-correct catalogs keyed to current invocation paths. The Mantle Chat catalog uses Mantle IDs and only Chat-compatible models; the runtime catalog uses Converse-capable runtime IDs once migration is done.
- Move SDK chat from InvokeModel to Converse and ConverseStream so the runtime catalog can truthfully list current Claude, Nova, Llama, DeepSeek, and Cohere models.
- Add a Bedrock Mantle Responses adapter and an Anthropic Messages adapter, both behind the existing API-key path, so GPT-5.6/5.5/5.4 and Claude families become invocable and listable.
- Make live provider `/models` discovery authoritative at runtime; keep the built-in catalogs as the safety net only, mirroring the NVIDIA NIM pattern.
- Add a weekly docs-driven verifier that opens one PR against the delimited generated blocks. Documentation is authoritative; authenticated Mantle/control-plane listings are advisory only and cannot evict a model.
- Two-run confirmation for any actionable removal, mirroring `sync-nvidia-nim-fallback`. HTTP 429 counts as available; 5xx/timeout or documentation fetch failure is transient and resets confirmation; documentation parse failures that leave the compatibility matrix incomplete fail the run without editing.
- Exclude legacy, preview/gated, safeguard/moderation, and embedding/image/video/speech-only models by default.
- No new Cargo dependencies. Reuse `aws-sdk-bedrockruntime` Converse surface and existing `reqwest`+`async-stream` machinery.

## Phased implementation

Each phase is independently shippable. Phases 2 and 3 may be reordered; phase 4 depends only on the marker conventions introduced in phase 1.

### Phase 1 — Corrected, invocable fallback catalogs

Edit `crates/ai-gateway/src/providers/bedrock.rs`:

1. Add a delimited generated block per catalog:
   `// BEGIN BEDROCK MANTLE CHAT FALLBACK MODELS`
   `pub const BEDROCK_MANTLE_CHAT_FALLBACK: &[BedrockFallbackModel] = &[...];`
   `// END BEDROCK MANTLE CHAT FALLBACK MODELS`
   `// BEGIN BEDROCK RUNTIME FALLBACK MODELS`
   `pub const BEDROCK_RUNTIME_FALLBACK: &[BedrockFallbackModel] = &[...];`
   `// END BEDROCK RUNTIME FALLBACK MODELS`
   `BedrockFallbackModel` carries `id`, `owned_by`, `endpoint` (enum `Mantle`|`Runtime`), `supports_vision`, `supports_reasoning`, `context_window`, `max_completion_tokens`, `source_url`.
2. Seed the Mantle Chat catalog with verified current Chat-capable Mantle IDs (e.g. `openai.gpt-oss-120b`, `openai.gpt-oss-20b`, `deepseek.v3.2`, `deepseek.v3.1`, `mistral.mistral-large-3-675b-instruct`, `qwen.qwen3-32b`, and current MiniMax/Gemma/Kimi/GLM/Grok entries chosen by the verifier). Use only IDs confirmed by the Programmatic Access section of each model card. Leave `None` for metadata not verified from the card.
3. Seed the runtime catalog with verified Converse-capable runtime IDs (Claude Sonnet 5, Claude Opus 4.8/4.7, Claude Haiku 4.5, Nova 2 Lite, Nova Pro/Lite/Micro, `openai.gpt-oss-120b-1:0`, `openai.gpt-oss-20b-1:0`, Meta Llama 3.1/3.2/3.3/4, DeepSeek V3.2/V3.1/R1, Mistral Large 3, Cohere Command R/R+ until flagged legacy). Exclude `meta.llama3-1-405b-instruct-v1:0` and Nova Premier (Legacy).
4. Replace `backup_models()` with `pub fn mantle_chat_fallback_models() -> Vec<Model>` and `pub fn runtime_fallback_models() -> Vec<Model>`, mapping the new structs to `Model` with `supports_vision` and, when added in phase 3, `supports_reasoning`.
5. In `list_models`, merge only the catalog matching the active `BedrockAuthMode`:
   - `BedrockAuthMode::ApiKey` → merge `mantle_chat_fallback_models()` after live `list_models_api_key` results.
   - `BedrockAuthMode::AwsSdk` → merge `runtime_fallback_models()` after `ListFoundationModels` results.
   Each merge keeps the existing `seen_ids` dedup so live IDs survive first.
6. Add a `BedrockEndpoint` enum tag to each `Model` via the existing `Model` shape if it gains an endpoint field later; for now leave the runtime `Model` shape unchanged and choose the catalog by `auth_mode`.
7. Update the three fallback tests:
   - `test_list_models_falls_back_to_backup_on_failure` should assert only the Mantle Chat fallback appears in API-key mode.
   - `test_backup_models_includes_new_openai_models` becomes `test_mantle_chat_fallback_includes_only_chat_compatible_models` and asserts `openai.gpt-5.5`/`openai.gpt-5.4` are **absent**.
   - Add `test_runtime_fallback_uses_converse_ids` asserting runtime-only IDs (`openai.gpt-oss-120b-1:0`, `anthropic.claude-opus-4-8`) are present in the runtime catalog and `openai.gpt-5.5`/`openai.gpt-5.4` absent.
8. Update `model_supports_reasoning` knowledge base and its proptest known-set at bedrock.rs:1992-2032 to use verified current IDs only (`anthropic.claude-sonnet-5`, `anthropic.claude-opus-4-8`, `anthropic.claude-opus-4-7` for true; Nova Pro, Llama, DeepSeek, Cohere for false). Do not introduce new patterns yet; the Responses/Messages work in phase 3 will revisit.

Edit `crates/ai-gateway/src/admin/static/index.html`:

1. Split `BEDROCK_BUILTIN_MODELS` at lines 1164-1211 into `BEDROCK_MANTLE_CHAT_MODELS` and `BEDROCK_RUNTIME_MODELS`.
2. In `fetchModelsForBedrock` (line 2007) and the model-row `fetchModelsForProvider` bedrock branch (line 1946), select which built-in list to merge based on whether the configured provider is API-key or SDK mode. The provider card already tracks `prov-key-env`; add a `data-auth-mode` attribute set on save so the UI knows the active mode. Default to the Mantle Chat list when the mode is ambiguous.
3. Update the manual-models hint near line 429 to state that Bedrock ships two maintained built-in fallbacks and that `manual_models` are optional overrides.
4. Keep the orphan-model preservation logic at lines 1917 and 2056 intact so saved routes keep working after the catalog split.

Edit `crates/ai-gateway/config.example.yaml` and `wiki/Providers.md`:

1. State the two-fallback behavior, when each activates, and that `manual_models` are additive overrides.
2. Do not prefill `manual_models`.

### Phase 2 — SDK chat via Converse

Edit `crates/ai-gateway/src/providers/bedrock.rs`:

1. Replace the `BedrockAuthMode::AwsSdk` chat path at lines 975-1001 with `client.converse()` using a normalized `ConverseRequest` built from `OpenAIRequest`. Implement `translate_to_converse_messages`, `translate_to_converse_config` (always set `max_tokens` explicitly), and `translate_converse_response`.
2. Replace the streaming path at lines 1017-1064 with `client.converse_stream()` and re-chunk into OpenAI SSE. Reuse the buffer-and-replay behavior already used elsewhere for translated streams.
3. Add `translate_model_id` entries for current model families (Claude Sonnet 5, Opus 4.x/5, Haiku 4.5, Nova 2 Lite, Nova Pro/Lite/Micro, DeepSeek, Mistral Large 3, Llama 4) so callers using short aliases still reach the right runtime IDs.
4. Delete the four InvokeModel legacy translators (`translate_claude_request`, `translate_titan_request`, `translate_jurassic_request`, `translate_command_request`) and the matching response translators, plus the property test that depends on them (`prop_bedrock_translation_round_trip`, bedrock.rs:1834). Replace with a Converse round-trip proptest over current runtime model IDs.
5. Keep the `BedrockAuthMode::ApiKey` path unchanged; it already speaks OpenAI Chat Completions to Mantle.
6. Confirm `CliƧ`/prompt caching and `reasoning` injection in `crates/ai-gateway/src/router/router.rs:1161-1179` still apply only when the model is known reasoning-capable. Update `model_supports_reasoning` to recognize Nova 2 Lite and current Claude families once their cards confirm reasoning support.

### Phase 3 — Responses and Messages adapters

Edit `crates/ai-gateway/src/providers/bedrock.rs`:

1. Add a Mantle Responses path invoked when `BedrockAuthMode::ApiKey` and the requested model is in the Mantle Responses set (GPT-5.6 sol/terra/luna, GPT-5.5, GPT-5.4). POST to `https://bedrock-mantle.{region}.api.aws/openai/v1/responses` with OpenAI Responses request shape; translate Responses output back to OpenAI chat completion `choices` so the rest of the gateway is unchanged.
2. Add an Anthropic Messages translation path invoked when the requested model is Mantle-present Claude (Sonnet 5, Fable 5, Opus 4.7/4.8, Haiku 4.5). Translate OpenAI messages to Anthropic `messages` + `system`, POST to `https://bedrock-mantle.{region}.api.aws/v1/messages`, and convert the Anthropic response back to OpenAI chat completion. Handle `thinking` consistently with the existing `reasoning` flag.
3. Add a routing helper `select_mantle_api(model_id, auth_mode) -> MantleApi` returning `Chat`|`Responses`|`Messages`. Use it in both chat and stream paths. Document the dispatch in `wiki/Streaming.md` (Bedrock buffer-and-replay now covers Responses and Messages).
4. Extend `list_models_api_key` (bedrock.rs:703-751) to also query `/openai/v1/responses` discoverable model cards if AWS exposes them; otherwise seed Responses-only IDs from the Mantle Chat fallback's sibling Responses catalog (a new `BEDROCK_MANTLE_RESPONSES_FALLBACK` constant). Add the Responses-only IDs to the admin UI's Mantle Responses list with a clear `(Responses)` suffix so users know which API they invoke.
5. Add `BEDROCK_MANTLE_MESSAGES_FALLBACK` for Claude IDs and surface them in the UI with a `(Messages)` suffix.
6. Update `model_supports_reasoning` and proptest known sets to include the Sonnet 5 / Opus 4.8 / Fable 5 IDs reachable through Responses/Messages.
7. Update `wiki/Providers.md` Bedrock section and `crates/ai-gateway/README.md` Bedrock authentication/models section with the three Mantle APIs and which models use which.

### Phase 4 — Weekly docs-driven verifier

Create `scripts/sync-bedrock-fallback.ps1` mirroring `sync-nvidia-nim-fallback.ps1`:

- Parameters: `-DryRun`, `-StatePath`, `-FromCache`, `-ValidateFixtures`, optional `-Regions`.
- Authoritative source: scrape the Bedrock User Guide pages listed below and parse the endpoint availability, API compatibility, Model Lifecycle, and each model card's Programmatic Access section. Build a structured `compatibility` map keyed by model card name with fields `runtime_id`, `mantle_chat_id`, `mantle_responses_id`, `mantle_messages_id`, `apis`, `endpoints`, `lifecycle`, `source_url`.
- Advisory cross-check: if AWS credentials are supplied via `AWS_*` env or profile, call `aws bedrock list-foundation-models --region <region>` for each `-Regions` region and `aws bedrock list-inference-profiles --region <region>`; merge `modelLifecycle.status` into the compatibility map. Region-specific removals never evict; they only annotate.
- Destination: rewrite only the four delimited generated blocks in `bedrock.rs` and the corresponding UI arrays in `index.html`. Preserve all surrounding code.
- Replacement policy: a current catalog entry is replaced only after two consecutive scheduled runs confirm the model is absent from documentation compatibility tables for its endpoint AND any advisory live listing also marks it absent or LEGACY. Documentation fetch failure fails the run.
- Additions: any model newly present in docs with a verified Programmatic Access ID is added on the first run that confirms it.
- Conservative defaults: never include Legacy, Preview/Gated, Safeguard/moderation, embedding/image/video/speech-only, or IDs the compatibility table does not verify. Leave `context_window`/`max_completion_tokens` `None` unless the model card prints them.
- `-DryRun` never edits source; emit proposed diff and updated state to a workflow-supplied output path.
- Exit codes: 0 no drift; 1 drift (rewritten unless `-DryRun`); 2 missing AWS credential when advisory check is requested but unavailable; 3 hard documentation scrape/parse failure; 4 script error.
- Add committed JSON fixtures under `scripts/fixtures/bedrock/` covering endpoint availability, API compatibility, and a model card, plus a `-ValidateFixtures` offline exercise of parsing and generated-block replacement.

Create `.github/workflows/sync-bedrock-fallback.yml`:

- Trigger weekly Tuesday 06:17 UTC and via `workflow_dispatch` with `dry_run`.
- `windows-latest`, `actions/checkout@v4` full history, `actions/cache@v4` with key `bedrock-fallback-state-${{ github.run_id }}` and restore prefix `bedrock-fallback-state-`.
- Configure `aws-region` and an IAM role/secret with `bedrock:ListFoundationModels` and `bedrock:ListInferenceProfiles` only; document the required secret and minimal IAM policy. The workflow still runs docs-first without AWS credentials.
- Branch `chore/bedrock-fallback-sync`, force-with-lease, single PR titled `chore(bedrock): sync built-in fallback models`. Commit only `crates/ai-gateway/src/providers/bedrock.rs` and `crates/ai-gateway/src/admin/static/index.html`.
- Publish summary JSON via `$env:GITHUB_STEP_SUMMARY`; warn (do not fail) on exit 2 (advisory creds missing) when docs alone produced no drift.
- Mirror the NVIDIA workflow's PR-update vs create logic so existing PRs are updated rather than duplicated.

Documentation sources for the verifier:
- `https://docs.aws.amazon.com/bedrock/latest/userguide/models-endpoint-availability.html`
- `https://docs.aws.amazon.com/bedrock/latest/userguide/models-api-compatibility.html`
- `https://docs.aws.amazon.com/bedrock/latest/userguide/model-lifecycle.html`
- `https://docs.aws.amazon.com/bedrock/latest/userguide/model-cards.html` and each provider/model card
- `https://docs.aws.amazon.com/bedrock/latest/userguide/models-supported.html`
- Inference profiles: `https://docs.aws.amazon.com/bedrock/latest/userguide/inference-profiles-support.html`

## Validation

1. `cargo fmt --check -p ai-gateway`.
2. Focused: `cargo test -p ai-gateway --lib providers::bedrock` and admin/gateway model-list tests.
3. Full: `cargo test -p ai-gateway` with `CARGO_BUILD_JOBS=1` (low-memory build constraint per environment.md).
4. Release: `cargo build --release -p ai-gateway` with `CARGO_BUILD_JOBS=1`.
5. New wiremock tests:
   - API-key mode returns only Mantle Chat fallback when both Mantle paths fail.
   - API-key mode surfaces Responses-only and Messages-only IDs only after phase 3 adapters land (assert absence before, presence after).
   - SDK mode returns only runtime fallback and invokes Converse (assert request shape).
   - Streaming over ConverseStream and over Mantle Responses/Messages both produce OpenAI-shaped SSE.
6. Fixture-driven verifier runs twice with a missing-model fixture to prove two-run confirmation; one transient fixture in between resets the count.
7. Manual: after merge, run `scripts/sync-bedrock-fallback.ps1 -DryRun` and `workflow_dispatch` with `dry_run=true` before relying on the weekly schedule.

## Failure behavior

- Transient docs HTTP failure: fail the verifier run, no edit, no PR.
- Region-specific live absence: never evicts; only annotates.
- Live-listing `LEGACY` matches docs Legacy: eligible for replacement after two confirmations.
- Two-run confirmation required for any removal; one run only records state.
- Manual `manual_models` entries are additive overrides and are never rewritten by the verifier.
- Verifier never adds a model ID the compatibility tables do not verify, even if a live listing suggests it.

## Out of scope

- Rewriting users' saved `manual_models`.
- Self-hosted/cross-region inference profile ARN catalog mirroring beyond the advisory cross-check.
- New provider types or non-Bedrock fallbacks.
- Replacing `model_supports_reasoning`'s pattern approach with a per-model metadata table (defer to a future capability-metadata refactor).
- Non-AWS fallback sources (e.g. third-party status pages).
