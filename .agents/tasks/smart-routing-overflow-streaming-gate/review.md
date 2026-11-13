# Smart Routing soft tier overflow and Streaming Reliability buffering gate

Smart Routing used to hard-filter a group down to the selected tier. When those models were all in cooldown or had an open breaker, the request failed even though other healthy, context-safe models in the same group could have served it. This change adds a soft overflow. The planner now returns the context-safe models outside the tier set (`CandidatePlan::overflow`). A new shared router helper, `resolve_routed_providers`, puts those models after the health-gated, sticky-promoted tier list, so the buffered and streaming paths cannot drift apart. The second change narrows the streaming rule. Smart Routing used to force buffer-and-replay unconditionally. Now it forces buffering only when Streaming Reliability is off (`emit_early_event == false && keepalive_interval_seconds == 0`). The pass-through variant also carries the routing decision, so the structured-output eager path still emits Smart Routing headers.

Watch for: in the main chat-completions handler, keep-alive-only Streaming Reliability still buffers because the handler only reaches the streaming router when `emit_early_event` is on (confirmed, pre-existing, not SR-forced). SR pass-through pre-content failover now re-runs the full Smart Routing plan on every retry (likely). Sticky affinity is not applied inside the overflow list (confirmed, minor).

**Verdict**: APPROVED

## High-level view

Overflow is computed once, inside `SmartRouter::plan`, as `context_safe_group − candidates`, keyed by `(provider, model)`. Overflow never starts from the raw group, so context-unsafe models and unknown-context models (when `allow_unknown_context_window` is false) can never enter it. Overflow is empty whenever tier filtering is bypassed: pinned requests, or groups with no tier tags. Pinned behavior is therefore unchanged.

The router sorts the tier list and the overflow list separately with `select_provider_order`, applies sticky promotion only to the tier list, and appends overflow with a `provider:model` de-dup. This guarantees the ordering contract: every healthy tier candidate is attempted before any overflow model. The failover group passed down stays the tier-filtered group. That group is only consulted for compression settings, so overflow models route normally. Overflow is a flat priority/cost/latency sort across other tiers and untiered models, not an adjacent-tier walk. Operator `priority` values decide whether a Balanced model is tried before a Fast one.

The streaming gate is a pure function, `streaming_requires_buffering`, built on the new `StreamingConfig::reliability_enabled()`. It exactly matches the requested condition, and `passthrough_enabled: false` still always buffers. The cascade stays confined to the buffered path. The SR decision metric is recorded once on the primary pass-through and skipped on exclusion re-plans. Public surface grows by one field on `CandidatePlan`, one field on `StreamingResponse::PassThrough`, and one method on `StreamingConfig`. No new config, labels, or UI.

Evidence in `verification.md` records `cargo check -p ai-gateway --all-targets` with zero errors and zero warnings. The full `cargo test -p ai-gateway` run had 4950 passed and 0 failed. The ten new tests cover all four required scenarios.

<details>
<summary>Issues (4)</summary>

1. **Keep-alive-only reliability never reaches pass-through in chat completions** (confirmed, non-blocking): `chat_completions_stream` calls the buffered `route_request` whenever `emit_early_event` is false, regardless of keep-alive. The new gate's `keepalive > 0` branch is therefore only effective on the Responses API and post-call guardrail paths. Either document that turning on only keep-alive does not enable SR pass-through for `/v1/chat/completions`, or route that handler branch through `route_request_streaming` when `keepalive_interval_seconds > 0`.
2. **Full SR re-plan on every pre-content streaming failover** (likely, non-blocking): every `route_request_streaming_excluding` retry from the handler re-runs `SmartRouter::plan`. That repeats the semantic-cache lookup, the classifier (an extra LLM call when the LLM classifier is used), the budget check and its metric, and `observe_plan`. Accept it explicitly, or pass the already-resolved order and decision into the excluding call.
3. **Sticky affinity skipped inside overflow** (confirmed, minor): `promote_sticky_provider` runs only on the tier list. When every tier candidate is unhealthy, the overflow model that served the prefix last is not preferred, and prompt-cache reuse is lost. Optionally apply sticky promotion to the overflow sub-list too; it would still stay after the tier list.
4. **No test for other-tier / non-specialist overflow** (confirmed, non-blocking): the plan and router tests only put untiered models in overflow. Add one `plan` test with a Balanced-tagged model and a same-tier non-specialist when the selected tier filters on specialization, and assert both appear in `overflow`.

</details>

<details>
<summary>Details</summary>

### Where overflow is computed and why context safety holds

The planner pipeline already ran `filter_by_context_capacity`, then `filter_by_tier`. The new block reuses `context_safe_group`, the context filter's output, so overflow inherits the same unknown-window policy (`pinned_context.allow_unknown_context_window || policy.allow_unknown_context_window`). It does not re-implement that policy. When nothing is context-safe, the planner returns `ContextCapacity` before overflow is computed, so overflow cannot become a back door for unsafe models. A `HashSet` seeded with the tier candidates handles de-dup and also collapses duplicates inside the group itself. The first occurrence wins, in group order.

```
group.models ──context filter──► context_safe_group ──filter_by_tier──► candidates
                                          │                                  │
                                          └──────── minus (provider,model) ──┴──► overflow
```

### Ordering contract in `resolve_routed_providers`

```
select_provider_order(tier group) ─► promote_sticky ─► [T1, T2, ...]
select_provider_order(overflow)   ─────────────────► [O1, O2, ...]  (dedup by provider:model)
providers = [T..., O...]  ─► exclusion filter ─► route_with_failover_for_group / streaming loop
```

Each sub-list is sorted on its own, so a low-priority-number overflow model never jumps ahead of a tier candidate. `smart_routing_untiered_models_only_overflow_after_tier_candidates` asserts exactly this: priority 1 and 10 untiered models land after a priority 50 Powerful model. One side effect is that overflow ordering is a single priority/cost/latency sort. For a Powerful request with tier candidates down, a Fast model with a lower `priority` number is tried before a Balanced one. That matches the requested "same rules", but it puts the burden of capability ordering on operator priorities. That matters for the user's goal of weighting toward the most capable coding models.

Sticky promotion applies only to the tier list (issue 3). The intent was that an overflow model is never promoted ahead of a tier candidate. That still holds if promotion is also applied *within* the overflow sub-list. As written, a conversation that failed over to an overflow model loses its cache affinity on the next turn whenever the tier is still down.

### Keep-alive-only Streaming Reliability in chat completions

The gate itself is correct, but there is a gap one layer up (issue 1). In `chat_completions_stream`, the `if streaming_config.emit_early_event` branch is the only place that calls `route_request_streaming`. The else branch buffers through `route_request` for every request. With `emit_early_event: false, keepalive_interval_seconds: 5`, Streaming Reliability counts as enabled by the new definition, but chat-completions SR requests still buffer. Smart Routing is not the cause, so the letter of Change 2 is met. Still, an operator who enables only keep-alive in the Streaming Reliability tab will not see SR pass-through on the main endpoint. The Responses API (`responses_create_stream`) and `stream_buffered_with_post_call` call the streaming router unconditionally, so the keep-alive branch of the gate does matter there.

### Re-planning on SR pass-through failover

Before this change, SR streaming always buffered, so the handler's pre-content failover loops never ran for SR traffic. Now they do. Each `route_request_streaming_excluding(&request, &tried_providers, …)` call re-enters `resolve_routed_providers` and therefore runs the full `SmartRouter::plan` again. That includes the semantic-cache lookup, classification, the budget check, the `cache_lookup`/`budget_decision` metric increments, and `quality_evaluator.observe_plan` (issue 2). With the heuristic classifier this costs only CPU. With an LLM or composite classifier it adds a classifier round-trip to every failover hop, and that hop happens while the client is waiting on an already-open SSE stream. Ordering stays correct, because overflow guarantees the excluded set leaves the remaining context-safe models available.

### Test coverage

All four required scenarios have tests. Not tested: other-tier tagged models or same-tier non-specialists appearing in `overflow` (issue 4; the code is a generic set difference, but the spec names those categories explicitly), `allow_unknown_context_window: true` admitting 0-context models into overflow, and overflow interaction with `version_fallback_enabled` sorting.

</details>

<details>
<summary>File map</summary>

- `crates/ai-gateway/src/smart_routing/mod.rs`: `CandidatePlan::overflow` field, overflow computation in `plan`, two plan tests.
- `crates/ai-gateway/src/config/mod.rs`: `StreamingConfig::reliability_enabled()` plus a table test.
- `crates/ai-gateway/src/router/router.rs`: `RoutedProviders`, `resolve_routed_providers`, `streaming_requires_buffering`, extracted `record_smart_routing_decision_metric`, `PassThrough.smart_routing`, both routing paths switched to the helper, seven router tests.
- `crates/ai-gateway/src/gateway/handlers.rs`: `PassThrough` destructure updates; eager structured path inserts `gateway_smart_routing`.
- `wiki/Routing-and-Failover.md`: new "Smart Routing Tier Overflow" subsection.
- `wiki/Streaming.md`: buffer-and-replay table row and pass-through sentence for SR.

Full diff: `git diff` in the workspace root (uncommitted).

</details>
