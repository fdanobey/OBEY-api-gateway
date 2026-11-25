# AGENTS.md

This file provides guidance to agents when working with code in this repository.

## Build & Run

```bash
cargo build --release -p ai-gateway    # Release binary at target/release/ai-gateway
cargo run -p ai-gateway -- --config ./config.yaml
```

### Clean-Build Requirement

Builds must be clean — zero errors, zero warnings. After any major change, run `cargo check -p ai-gateway --all-targets` and fix every warning (unused imports, dead code, etc.) before considering the work done. Do not silence warnings with blanket `#[allow]` attributes; remove the dead code or gate genuinely test-only helpers with `#[cfg(test)]` instead.

## Test

```bash
cargo test -p ai-gateway               # All tests
cargo test -p ai-gateway <test_name>   # Single test
cargo test -p ai-gateway -- --nocapture  # With output
```

### Test Profiles

- **Fast (default)**: `cargo test -p ai-gateway` — unit and integration tests with isolated temp databases.
- **Full coverage**: `cargo test -p ai-gateway -- --ignored` — includes wall-clock latency assertions.
- **Property tests budget**: `PROPTEST_CASES=64 cargo test -p ai-gateway` — lower case count for faster runs.

### Performance / Latency Budget Tests

Wall-clock budget tests are marked `#[ignore]` and run with `--ignored`:
- `performance.rs`: startup < 2s, forwarding overhead < 10ms, concurrent requests
- `guardrail_timing.rs`: pre-call < 100ms/500ms, streaming assembly < 500ms

### Test Database Isolation

Every `GatewayServer::new` opens SQLite databases. Tests use `common::isolate_databases` to redirect these into unique temp directories, avoiding lock contention across parallel tests.

### Test Binaries (keep the build small)

- Integration tests compile into one binary, `tests/suite.rs` (`autotests = false`). A new `tests/foo.rs` must be declared there as `mod foo;` (it reaches helpers with `use crate::common;`); `every_test_file_is_compiled` fails otherwise. Run one file with `cargo test -p ai-gateway --test suite foo::`.
- Only files that mutate process-global state another file depends on get their own `[[test]]` target (the `github_copilot_*` files set `AI_GATEWAY_DATA_DIR`).
- Library unit tests never set `AI_GATEWAY_DATA_DIR`: in `cfg(test)` builds `secrets::storage_dir()` resolves to `secrets::test_support::shared_data_dir()` (one temp dir per process, master key pre-created). Use unique credential file names in it; a test that persists to a fixed default path (`oauth_tokens.enc`, `copilot_tokens.enc`) takes `test_support::scoped_data_dir()` for its body.
- `src/main.rs` uses the `ai_gateway` library; never re-declare library modules with `mod` there (that compiled the crate twice and ran every unit test twice).
- The test profile builds `ai-gateway` at `opt-level = 1` and dependencies at `opt-level = 2` (root `Cargo.toml`). Do not raise it back to 2 (≈8 GB per rustc, the OOM cause) or drop it to 0 (stack overflows in `anthropic_front_door_streaming`, `performance.rs` gate fails).
- `Qdrant::from_url(..).build()` runs a blocking health check; on Windows a refused loopback connect costs ~2 s. Test clients use `.skip_compatibility_check()`.

## Non-Obvious Patterns

- **Provider API key resolution** (`Provider::resolve_api_key` in [`config/mod.rs`](crates/ai-gateway/src/config/mod.rs)): the decrypted `api_key_encrypted` / `resolved_api_key` wins; otherwise `api_key_env` is read as an env var only when it looks like one (`secrets::is_env_var_reference`, UPPER_SNAKE). A plaintext key typed into `api_key_env` is encrypted into `api_key_encrypted` on save (`normalize_config_for_storage` in `admin/mod.rs`, helpers in `secrets.rs`). Guardrail and web-search backends still use the older rule: env var first, literal fallback (`resolve_api_key(api_key_env)`).
- **Base URL normalization**: Provider URLs are stripped of trailing `/` and `/v1` is appended if missing (`dispatch_attempts_under_permit` in [`router.rs`](crates/ai-gateway/src/router/router.rs)). Bedrock in API-key mode without `custom_vpc_endpoint` ignores `base_url` and uses `https://bedrock-mantle.<region>.api.aws/v1`.
- **Config path resolution**: CLI `--config` → `CONFIG_PATH` env → `./config.yaml` (`resolve_config_path` in [`validation.rs`](crates/ai-gateway/src/config/validation.rs))
- **Hot-reload resets runtime state**: `/admin/config/reload` clears circuit breakers, rate limiters, retry budgets, the sticky cache and HTTP clients (`apply_runtime_config_update` in `gateway/mod.rs`). `/admin/circuit-breaker/reset` clears breakers alone.
- **Tests use `tower::ServiceExt::oneshot()`**: Integration tests don't bind ports; they call router directly
- **Property tests with proptest**: Many tests use `proptest!` macro for randomized input validation
- **Admin panel parity (standing user requirement)**: every YAML-configurable feature must also expose matching controls in the embedded admin panel (`crates/ai-gateway/src/admin/static/index.html`). Config-only features without UI are considered incomplete; specs must include admin-UI tasks.
- **Upstream timeouts and failover** (see [ADR 0001](docs/adr/0001-buffered-dispatch-streams-upstream.md); do not regress):
  - Buffered dispatch (`dispatch_attempts_under_permit`) streams upstream (`stream: true` + `stream_options.include_usage`) and reassembles the SSE. Never force `stream: false` by default: provider edges cut long non-streaming requests (Electron Hub returned 504 at ~100 s). Only `buffered_upstream_streaming: false`, `n > 1`, `logprobs` and audio use `stream: false`. While Codex Search is enabled, every streaming client request takes this buffered path.
  - TTFB = headers plus the first body byte; idle gaps use `streaming.chunk_timeout_seconds` (`ChunkTimeout`); `total_timeout` caps the whole try.
  - Timeout-class errors (TTFB/total/chunk timeout, upstream 504/524) fail over without a same-provider retry (`is_timeout_class`). 500/502/network errors keep `max_retries_per_provider`. Pass-through TTFB/send/5xx fallbacks exclude the failed `provider:model`, as the 429 path does.
  - Every failed try and every abandoned request is logged through the `ActiveRequestHandle` attempt ledger: `#attempt` rows with `duration_ms`/`error_class`, and 499 `client_disconnect` / 504 `gateway_deadline` outcome rows written by `RequestCompleteGuard`. Do not reintroduce response-extras channels for attempts.
  - Guard tests: `buffered_sse_reader_allows_generation_longer_than_ttfb_when_bytes_flow`, `buffered_dispatch_requests_upstream_stream_and_reassembles_sse`, `timeout_class_errors_fail_over_without_same_provider_retry` (router.rs) and `abandoned_stream_writes_client_disconnect_outcome_row` (tests/request_outcome_logging.rs).
- **Buffered reasoning progress relay** (see [ADR 0003](docs/adr/0003-buffered-reasoning-progress-relay.md)): while a streaming request is buffered, `read_upstream_body` copies reasoning-only deltas (`classify_reasoning_frame`, fail-closed) into a bounded fail-open channel on `ActiveRequestHandle`, and the handler forwards them as `chat.completion.chunk` frames between keep-alives so client first-token timeouts don't fire. The final replay omits already-relayed reasoning. Toggle: `streaming.relay_reasoning_progress` (default `true`). The relay is transport only; it never consumes rate tokens or delays dispatch.
- **Provider rate-limit accounting** (see [ADR 0002](docs/adr/0002-provider-rate-limit-accounting.md); do not regress): exactly one token per real upstream request. Admission (`FirstSendToken::Prepaid`) is the only gate and may wait up to `rate_limit_max_wait_ms` for the primary with nothing held. Every retry, Codex Search round or nudge is `debit`ed and never blocks. The Errors tab filters errors in SQL. Guard tests: `single_request_consumes_exactly_one_provider_rate_token`, `sequential_requests_under_limit_never_rate_limit_skip`, `rate_limit_skip_waits_briefly_for_primary_token`, `errors_tab_includes_failed_attempts_on_successful_request`.

## Branches, CI & Release

- `development` is the integration branch: feature PRs and Dependabot target it. Never push or open feature PRs to `master`; only the Release workflow advances it.
- CI (`.github/workflows/ci.yml`) runs on push/PR to `development`: `Build & validate fallback catalogs` (`cargo check -p ai-gateway --all-targets` with `RUSTFLAGS=-D warnings`, then NVIDIA NIM fixture validation) is the required check; `Supply-chain gate (cargo-deny)` also runs.
- Releases: see `.kiro/steering/bump-version-release.md`.

## Agent skills

### Issue tracker

Issues are tracked as local markdown files under `.scratch/<feature>/`. See `docs/agents/issue-tracker.md`.

### Domain docs

Single-context: root `CONTEXT.md` + `docs/adr/` (created lazily by `/domain-modeling`). See `docs/agents/domain.md`.
