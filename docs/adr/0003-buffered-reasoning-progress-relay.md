# ADR 0003: Buffered reasoning progress relay

Status: accepted (pending deploy approval)
Date: 2026-10-15

## Context

Buffered dispatch (ADR 0001) streams upstream with `stream: true` and reassembles the SSE before emitting the final response to the client. Some streaming clients abort those requests even though upstream bytes are flowing. Kilo Code, running on the Bun runtime, hardcodes a 300s first-token timeout: while the model "thinks" for minutes before emitting content, the gateway holds the connection open with only keep-alive comments — no `chat.completion.chunk` frame ever reaches the client, so the client times out and disconnects (Kilo-Org/kilocode#1681; a client-side fix is infeasible).

Upstream reasoning-only deltas (non-empty `reasoning` / `reasoning_content`, no content) already flow during buffering. Forwarding them to the client as progress frames keeps first-token timeouts alive without changing dispatch semantics.

## Decision

1. **Pure frame classifier.** `classify_reasoning_frame` treats a frame as reasoning iff it carries a non-empty `reasoning`/`reasoning_content` field, no non-null `content`, and no `tool_calls`. Anything ambiguous is **not** classified as reasoning (fail-closed): content, tool calls, usage, and finish frames are never relayed.
2. **Tap in `read_upstream_body`.** This is the single seam through which every openai-compatible buffered dispatch reads the upstream SSE; ADR 0001 semantics are preserved unchanged (`stream: true` + `include_usage`, reassembly untouched). Reasoning deltas are duplicated into a bounded `mpsc(128)` **fail-open** relay: if the channel is full or the receiver is gone, frames are dropped and dispatch proceeds.
3. **`ActiveRequestHandle` carries the relay sender.** The handle already spans the routing task and the handler, so multi-attempt failover gets relayed reasoning for free; each attempt's frames share the response id the handler renders.
4. **Handler-side `select!` loops forward frames to the client** between keep-alives, rendered as `chat.completion.chunk` frames with the shared response id and no `role` during buffering. The final response is emitted minus the already-relayed reasoning deltas — the assembled response object still carries the full reasoning text, but the final re-chunk suppresses reasoning when any frame was relayed, so nothing is double-emitted.
5. **ADR 0002 preserved.** The relay is transport, not dispatch: still exactly one rate token per real upstream request. Retries and failover debits as before.
6. **Config and admin parity.** `streaming.relay_reasoning_progress` (default `true`, `#[serde(default = "default_true")]`), editable as a checkbox in the admin panel's Streaming Reliability section. `reliability_enabled()` is unchanged — the relay is independent of the early-event/keep-alive gate.
7. **Telemetry.** `gateway_reasoning_relayed_frames` rides the existing response-extras path into the request log; no schema change.

## Alternatives rejected

- **Client-side fix (raise the first-token timeout in Kilo Code)** — infeasible; the timeout is hardcoded in the Bun runtime (Kilo-Org/kilocode#1681).
- **Empty-content-delta heartbeats** — indistinguishable from real empty deltas; unverifiable by clients, rejected.
- **Per-model exemption from Codex Search buffered dispatch** — rejected by product decision: interception of tool calls requires buffering.

## Non-goals

- Codex OAuth provider dispatch (native pass-through already forwards reasoning live).
- The structured-output eager path (schema-validated streams must remain byte-exact).
- Non-streaming requests (no client connection to keep alive).

## Consequences

- Streaming clients see `chat.completion.chunk` frames with the shared response id and no `role` during buffering; clients that ignore unknown deltas are unaffected, clients that render reasoning show live progress.
- A full or disconnected relay channel drops frames (fail-open): dispatch, retries, and the final response are never delayed or aborted by the relay.
- The final replay omits reasoning deltas that were already relayed, so clients see each reasoning token exactly once.
- Disabling `relay_reasoning_progress` restores the pre-ADR behavior (keep-alive comments only) for clients that reject mid-stream reasoning frames.

## Invariants and guard tests

Do not regress these.

| Invariant | Test |
|---|---|
| Reasoning frames relayed during buffering; content/tool_calls/usage held back | router.rs / handlers.rs relay tests (see spec `buffered-reasoning-progress-relay`) |
| Final response emits reasoning minus already-relayed deltas (no double-emit) | re-chunk suppression tests (handlers.rs) |
| Relay failure never blocks dispatch; ADR 0002 token accounting unchanged | `single_request_consumes_exactly_one_provider_rate_token` (router.rs) |
| Config + admin round-trip | manual admin-panel verification (file:// + stubbed fetch) |
