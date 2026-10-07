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

## Non-Obvious Patterns

- **API key resolution**: `api_key_env` in config is tried as env var name first, falls back to literal value if env var not found ([`router.rs:286-291`](crates/ai-gateway/src/router/router.rs:286))
- **Base URL normalization**: Provider URLs are stripped of trailing `/` and `/v1` is appended if missing ([`router.rs:278-283`](crates/ai-gateway/src/router/router.rs:278))
- **Config path resolution**: CLI `--config` → `CONFIG_PATH` env → `./config.yaml` ([`validation.rs`](crates/ai-gateway/src/config/validation.rs))
- **Circuit breaker reset**: All circuit breakers clear on config hot-reload via `/admin/config/reload`
- **Tests use `tower::ServiceExt::oneshot()`**: Integration tests don't bind ports; they call router directly
- **Property tests with proptest**: Many tests use `proptest!` macro for randomized input validation
- **Admin panel parity (standing user requirement)**: every YAML-configurable feature must also expose matching controls in the embedded admin panel (`crates/ai-gateway/src/admin/static/index.html`). Config-only features without UI are considered incomplete; specs must include admin-UI tasks.
- **Upstream timeouts and failover** (see [ADR 0001](docs/adr/0001-buffered-dispatch-streams-upstream.md); do not regress):
  - Buffered dispatch (`dispatch_attempts_under_permit`) streams upstream (`stream: true` + `stream_options.include_usage`) and reassembles the SSE. Never force `stream: false` by default: provider edges cut long non-streaming requests (Electron Hub returned 504 at ~100 s). Only `buffered_upstream_streaming: false`, `n > 1`, `logprobs` and audio use `stream: false`. While Codex Search is enabled, every streaming client request takes this buffered path.
  - TTFB = headers plus the first body byte; idle gaps use `streaming.chunk_timeout_seconds` (`ChunkTimeout`); `total_timeout` caps the whole try.
  - Timeout-class errors (TTFB/total/chunk timeout, upstream 504/524) fail over without a same-provider retry (`is_timeout_class`). 500/502/network errors keep `max_retries_per_provider`. Pass-through TTFB/send/5xx fallbacks exclude the failed `provider:model`, as the 429 path does.
  - Every failed try and every abandoned request is logged through the `ActiveRequestHandle` attempt ledger: `#attempt` rows with `duration_ms`/`error_class`, and 499 `client_disconnect` / 504 `gateway_deadline` outcome rows written by `RequestCompleteGuard`. Do not reintroduce response-extras channels for attempts.
  - Guard tests: `buffered_sse_reader_allows_generation_longer_than_ttfb_when_bytes_flow`, `buffered_dispatch_requests_upstream_stream_and_reassembles_sse`, `timeout_class_errors_fail_over_without_same_provider_retry` (router.rs) and `abandoned_stream_writes_client_disconnect_outcome_row` (tests/request_outcome_logging.rs).

## Agent skills

### Issue tracker

Issues are tracked as local markdown files under `.scratch/<feature>/`. See `docs/agents/issue-tracker.md`.

### Domain docs

Single-context: root `CONTEXT.md` + `docs/adr/` (created lazily by `/domain-modeling`). See `docs/agents/domain.md`.
