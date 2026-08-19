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

## Agent skills

### Issue tracker

Issues are tracked as local markdown files under `.scratch/<feature>/`. See `docs/agents/issue-tracker.md`.

### Domain docs

Single-context: root `CONTEXT.md` + `docs/adr/` (created lazily by `/domain-modeling`). See `docs/agents/domain.md`.
