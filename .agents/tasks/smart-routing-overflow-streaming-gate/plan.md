# Implementation Plan — Smart Routing tier overflow + streaming buffering gate

Work in the main tree `c:\Fbn_Dev_Apps\OBEY API Router` (no worktree). Do NOT commit. PowerShell only.
Binding: `cargo check -p ai-gateway --all-targets` must report zero errors and zero warnings; no blanket `#[allow]`.

## Verified facts (from exploration; line numbers as of planning)

- `crates/ai-gateway/src/smart_routing/mod.rs`
  - `pub struct CandidatePlan` (l.253): `decision`, `candidates`, `excluded_for_context`, `estimated_context_tokens`, `bypassed`. Constructed in exactly ONE place: end of `SmartRouter::plan` (l.585). No test builds it literally.
  - `SmartRouter::plan` (l.441): `filter_by_context_capacity` → `context_safe_group` (l.549-550) → `filter_by_tier(&context_safe_group, …)` (l.552) → `candidates = tier_filter.model_group.models`. `context_safe_group` is still in scope after tier filtering.
  - `filter_by_tier` (l.820) stays unchanged (hard filter semantics; cascade uses it).
  - Test helpers in `mod tests`: `request(content)`, `candidate(name, context_window, tier)` (provider is always `"test"`), `generated_candidate(index, tier, specializations)`, `group(models)`, `input(&req,&group,&pinned)`, `config(ClassifierMode)`.
- `crates/ai-gateway/src/config/mod.rs` `StreamingConfig` (l.1162): `emit_early_event: bool` (default true), `keepalive_interval_seconds: u64` (default 5; doc: "0 = disabled, use axum default"), `passthrough_enabled: bool` (default true), `chunk_timeout_seconds`, `retry_on_truncation`, `retry_on_stream_truncation`. Test module `streaming_config_tests` (l.1268).
- `crates/ai-gateway/src/router/router.rs`
  - `pub enum StreamingResponse` (l.376) with `PassThrough { byte_stream, provider, model, compression, concurrency_permit }`; only constructed at l.8947.
  - `smart_routing_plan` (l.1375, private) → `Option<CandidatePlan>`.
  - `select_provider_order` (l.1441): CB + cooldown gating, priority/cost/latency sort, version fallback. `promote_sticky_provider(&self, request, &mut Vec<ProviderModel>)` (~l.1590).
  - `route_request_with_exclusions` (l.8135): replaces group models with `plan.candidates` (l.8152-8161), selects order, sticky, exclusion filter, `route_with_failover_for_group`, cascade loop (l.8202-8261, leave alone), metric + `gateway_smart_routing` extra (l.8262-8285).
  - `route_request_streaming_excluding` (l.8373): same plan replacement (l.8396-8405), select + sticky, then gate at l.8418-8425 `if !streaming_config.passthrough_enabled || (routing_decision.is_some() && !routing_bypassed)` → Buffered. Candidate loop iterates `providers` skipping `exclude`; every fallback calls `route_request`/`route_request_with_exclusions` (which re-plan, so they pick up overflow automatically). Returns `PassThrough` at l.8947.
  - `route_with_failover_for_group` only uses `model_group` for compression lookup, not `models` — passing the tier-filtered group as today is fine.
  - Test helpers in `mod tests`: `create_test_config()` (l.10864), `test_metrics()`, `test_provider(name, base_url)`, `test_model(provider, priority)` / `test_model_named(provider, model, priority)` (tier None, context_window 0), `test_group(models)` (name `"test-group"`), `completion_response()` (l.11004), wiremock `MockServer`/`Mock`/`ResponseTemplate` already imported (see `streaming_provider_receives_compressed_body_before_response` l.11469). Cooldown pattern: `metrics.set_provider_cooldown(provider, reason_string, deadline_unix_secs)` (see ~l.14273). Existing SR router test: `disabled_smart_routing_preserves_provider_order` (l.12144) — new tests go right after it.
- `crates/ai-gateway/src/gateway/handlers.rs` PassThrough destructure sites: l.2407 (early-event path; SSE headers already committed before routing), l.2691 and l.2776 (pre-content failover loops), l.3435 (`stream_eager_structured_output` — assembles a response and later calls `smart_routing_headers(&response)` at l.3633, the only streaming path that attaches Smart Routing headers), l.3716 (already uses `..`), l.9906 (Responses API, explicit `field: _` style).
- Handler only calls `route_request_streaming` when `emit_early_event` is true (l.2341/2365) or in the structured-output eager path (l.3431); the early-event-disabled path calls buffered `route_request` directly (l.2884). No handler gating change is needed.
- No existing router test asserts hard-tier-only routing or SR-forced streaming buffering, so nothing existing must be rewritten; `smart_routing` module tests on `filter_by_tier` stay valid.
- Wiki: no page describes hard tier filtering or SR-forced buffering. `wiki/Streaming.md` "Streaming Modes" table (l.12-16) lists when buffer-and-replay is used; `wiki/Routing-and-Failover.md` has "Priority & Cost-Aware Routing" (l.145). Brief additions only.

## Design decisions

- D1 Overflow exposure: add `pub overflow: Vec<ProviderModel>` to `CandidatePlan`, computed inside `SmartRouter::plan` from `context_safe_group.models` minus the tier-selected `candidates` (keyed `provider:model`, first occurrence wins), empty when `tier_filter.bypassed`. Rationale: context safety was already computed there; one constructor; no re-filtering in the router; single additive public field.
- D2 Router ordering: new private helper `resolve_routed_providers` used by both buffered and streaming paths, so ordering cannot drift between them. Tier list = `select_provider_order(tier group)` + sticky promotion (sticky only within tier list, so an overflow model is never promoted ahead of tier candidates); overflow list = `select_provider_order(group with models = overflow)` (no sticky), appended with provider:model de-dup.
- D3 Gate: `StreamingConfig::reliability_enabled()` = `emit_early_event || keepalive_interval_seconds > 0` (0 = axum default = treated as disabled). Pure fn `streaming_requires_buffering(&StreamingConfig, smart_routing_applied) -> bool` in router.rs = `!passthrough_enabled || (smart_routing_applied && !reliability_enabled())`, unit-testable.
- D4 Pass-through SR context: add `smart_routing: Option<RoutingDecision>` to `StreamingResponse::PassThrough` (Some only when SR applied and not bypassed). The structured eager path inserts it into the assembled response's `gateway_smart_routing` extra so its existing `smart_routing_headers` call still emits headers; early-event paths ignore it (headers already committed — existing limitation, unchanged). Record the existing `record_smart_routing_decision` metric for pass-through only when `exclude.is_empty()` (primary attempt) so pre-content failover re-plans don't double count; extract the metric call into a helper shared with `route_request_with_exclusions`. No new labels. Cascade never runs for pass-through (it lives only in `route_request_with_exclusions`).

## Steps

- [x] 1. Add `overflow` to `CandidatePlan` and compute it in `SmartRouter::plan`.
      In `crates/ai-gateway/src/smart_routing/mod.rs`: add field with doc comment
      `/// Context-safe group models not selected by tier filtering, in group order. Empty when bypassed.`
      `pub overflow: Vec<ProviderModel>,`. In `plan`, after `let mut candidates = tier_filter.model_group.models;` (and the `prefer_candidates` block), compute:
      ```rust
      let overflow = if tier_filter.bypassed {
          Vec::new()
      } else {
          let mut seen: std::collections::HashSet<(String, String)> = candidates
              .iter()
              .map(|c| (c.provider.clone(), c.model.clone()))
              .collect();
          context_safe_group
              .models
              .iter()
              .filter(|m| seen.insert((m.provider.clone(), m.model.clone())))
              .cloned()
              .collect()
      };
      ```
      and pass `overflow` into the `CandidatePlan { … }` literal. Add tests in the same file's `mod tests`:
      - `plan_overflow_lists_context_safe_non_candidates`: only Powerful is tagged so adjacent-tier fallback always lands on it. Group = [`candidate("tier-pick", 200_000, Powerful)`, `candidate("untiered", 200_000, Powerful)` with `tier = None`, `candidate("too-small", 100, Powerful)` with `tier = None`]. Heuristic classifier (`config(ClassifierMode::Heuristic)`), `request("hello")`, default pinned. Assert `plan.candidates` models == ["tier-pick"], `plan.overflow` models == ["untiered"] ("too-small" absent because it fails context), `!plan.bypassed`.
      - `plan_overflow_is_empty_when_bypassed`: same group, `PinnedRoutingContext { model: Some("tier-pick".into()), ..Default::default() }` → `plan.bypassed` and `plan.overflow.is_empty()`.
      Files: crates/ai-gateway/src/smart_routing/mod.rs
      Verify: `cargo test -p ai-gateway smart_routing::tests::plan_overflow` passes; `cargo test -p ai-gateway smart_routing` all pass.

- [x] 2. Add `StreamingConfig::reliability_enabled`.
      In `crates/ai-gateway/src/config/mod.rs` add after `impl Default for StreamingConfig`:
      ```rust
      impl StreamingConfig {
          /// Streaming Reliability is active when the early synthetic SSE event or a
          /// custom keep-alive interval is enabled (`0` = axum default = disabled).
          pub fn reliability_enabled(&self) -> bool {
              self.emit_early_event || self.keepalive_interval_seconds > 0
          }
      }
      ```
      Add test `streaming_reliability_enabled_reflects_early_event_and_keepalive` in `mod streaming_config_tests`: default → true; (false, 0) → false; (false, 10) → true; (true, 0) → true (use `StreamingConfig { emit_early_event, keepalive_interval_seconds, ..StreamingConfig::default() }`).
      Note: method is unused in non-test code until step 4; do steps 2-4 before running `cargo check --all-targets` for the zero-warning gate (pub method on a pub type will not warn anyway).
      Files: crates/ai-gateway/src/config/mod.rs
      Verify: `cargo test -p ai-gateway streaming_config_tests` passes.

- [x] 3. Router: shared provider-order helper with overflow (Change 1) in both paths.
      In `crates/ai-gateway/src/router/router.rs`:
      a. Add a private struct near `StreamingResponse`:
      ```rust
      /// Attempt order plus Smart Routing context resolved for one request.
      struct RoutedProviders {
          /// Group used for compression/failover (models = tier candidates when SR applied).
          model_group: ModelGroup,
          /// Health-gated attempt order: tier candidates (sticky-promoted) then overflow.
          providers: Vec<ProviderModel>,
          decision: Option<crate::smart_routing::tier::RoutingDecision>,
          bypassed: bool,
      }
      ```
      b. Add `async fn resolve_routed_providers(&self, request: &OpenAIRequest, model_group: ModelGroup) -> Result<RoutedProviders, GatewayError>` (place right after `smart_routing_plan`): call `smart_routing_plan`; on `Some(plan)` build `tier_group = model_group.clone()` with `models = plan.candidates`, keep `overflow = plan.overflow` (empty if `plan.bypassed`); on `None` use the original group, empty overflow, `decision None`, `bypassed true`. Then `let mut providers = self.select_provider_order(&tier_group).await; self.promote_sticky_provider(request, &mut providers).await;`. If overflow non-empty: `let mut overflow_group = model_group.clone(); overflow_group.models = overflow; let extra = self.select_provider_order(&overflow_group).await;` build a `HashSet<String>` of `provider:model` from `providers` and push each `extra` entry whose key inserts successfully. Doc comment states tier candidates first, then context-safe overflow (untiered / other tiers / non-specialists), both health-gated and sorted by `select_provider_order`, sticky only within tier list. Return `RoutedProviders { model_group: tier_group, providers, decision, bypassed }`.
      c. Extract the metric block from `route_request_with_exclusions` (the `self.metrics.record_smart_routing_decision(SmartRoutingDecisionMetric { … })` call, l.8264-8277) into `fn record_smart_routing_decision_metric(&self, group: &str, decision: &crate::smart_routing::tier::RoutingDecision)`; call it from the original place (same values).
      d. `route_request_with_exclusions`: replace l.8152-8167 (plan → filtered group → `select_provider_order` → sticky) with
      `let RoutedProviders { model_group, providers, decision: mut routing_decision, bypassed: routing_bypassed } = self.resolve_routed_providers(&prepared_request, model_group).await?;` keeping the exclusion filter, empty check, failover call, cascade loop and response metadata unchanged.
      e. `route_request_streaming_excluding`: replace l.8396-8411 likewise (`let RoutedProviders { model_group, providers, decision: routing_decision, bypassed: routing_bypassed } = self.resolve_routed_providers(&prepared_request, model_group).await?;` — no `mut`, sticky is applied in the helper; keep the empty-list error). Leave the gate for step 4.
      Files: crates/ai-gateway/src/router/router.rs
      Verify: `cargo check -p ai-gateway --all-targets` — zero errors/warnings; `cargo test -p ai-gateway router::router::tests` passes.

- [x] 4. Streaming gate (Change 2) and pass-through SR context.
      In `crates/ai-gateway/src/router/router.rs`:
      a. Add free fn near the `smart_routing_*_name` helpers (l.271-300):
      ```rust
      /// Smart Routing forces buffer-and-replay only when Streaming Reliability
      /// (early SSE event / keep-alive) is disabled; pass-through disabled globally
      /// always buffers.
      fn streaming_requires_buffering(streaming: &crate::config::StreamingConfig, smart_routing_applied: bool) -> bool {
          !streaming.passthrough_enabled || (smart_routing_applied && !streaming.reliability_enabled())
      }
      ```
      (use whatever path `StreamingConfig` is already imported under in router.rs).
      b. Replace the gate at l.8418-8425 with `let smart_routing_applied = routing_decision.is_some() && !routing_bypassed; if streaming_requires_buffering(&streaming_config, smart_routing_applied) { debug!(smart_routing_applied, "Streaming pass-through disabled or Smart Routing without Streaming Reliability, using buffered path"); return Ok(StreamingResponse::Buffered(self.route_request(request, active.clone()).await?)); }`. Update the doc comment on `route_request_streaming` (l.8331-8347) with one bullet for the SR rule.
      c. Add field to `StreamingResponse::PassThrough`: `/// Smart Routing decision when SR applied (not bypassed); None otherwise. smart_routing: Option<crate::smart_routing::tier::RoutingDecision>,`. At the `Ok(StreamingResponse::PassThrough { … })` return (l.8947): before it, `let smart_routing = routing_decision.filter(|_| !routing_bypassed); if exclude.is_empty() { if let Some(decision) = smart_routing.as_ref() { self.record_smart_routing_decision_metric(&model_group.name, decision); } }` and add `smart_routing,` to the literal. Ensure `routing_decision`/`model_group` are not moved earlier (borrow-check).
      d. In `crates/ai-gateway/src/gateway/handlers.rs` update destructures: l.2407, l.2691, l.2776 append `..` to the field list; l.9906 add `smart_routing: _,`; l.3435 (`stream_eager_structured_output`) bind `smart_routing` and, next to the existing `gateway_provider`/`gateway_responded_model`/`gateway_compression` inserts (~l.3537-3549), add `if let Some(decision) = smart_routing { response.extra.insert("gateway_smart_routing".to_owned(), serde_json::to_value(decision).unwrap_or(serde_json::Value::Null)); }` so the existing `smart_routing_headers(&response)` at ~l.3633 emits headers for pass-through too. l.3716 already uses `..`.
      Files: crates/ai-gateway/src/router/router.rs, crates/ai-gateway/src/gateway/handlers.rs
      Verify: `cargo check -p ai-gateway --all-targets` — zero errors/warnings.

- [x] 5. Router tests for overflow and the gate.
      In `crates/ai-gateway/src/router/router.rs` `mod tests`, after `disabled_smart_routing_preserves_provider_order`, add a helper `fn sr_model(provider: &str, model: &str, priority: u32, tier: Option<SmartRoutingTier>, context_window: u32) -> ProviderModel` (built from `test_model_named` then setting `tier`/`context_window`) and `fn smart_routing_config() -> Config` (= `create_test_config()` with `smart_routing.enabled = true`, `retry.max_retries_per_provider = 0`). Make tier selection deterministic by tagging only `Powerful` in each group (adjacent-tier fallback always lands on Powerful). Use request model `"test-group"` (not pinned) with a short user message. Tests:
      - `streaming_requires_buffering_gate_table` (sync `#[test]`): passthrough false → true regardless; SR applied + reliability enabled (default) → false; SR applied + `emit_early_event:false, keepalive_interval_seconds:0` → true; SR not applied + reliability disabled → false.
      - `smart_routing_untiered_models_only_overflow_after_tier_candidates` (c): group = [untiered `"cheap"` priority 1, Powerful `"tier"` priority 50, untiered `"backup"` priority 10], all ctx 200_000. `resolve_routed_providers` → providers in order `tier`, `cheap`, `backup`; `decision.is_some()`, `!bypassed`.
      - `smart_routing_overflow_excludes_context_unsafe_models` (b): group = [Powerful `"tier"` ctx 200_000, untiered `"tiny"` ctx 100, untiered `"unknown"` ctx 0, untiered `"safe"` ctx 200_000]; default `allow_unknown_context_window=false` → providers == [`tier`, `safe`].
      - `smart_routing_overflow_serves_request_when_tier_candidates_unavailable` (a): one `MockServer`; providers `tier-cooled`, `tier-open`, `overflow-first`, `overflow-second` all `test_provider(name, server.uri())`, each with a distinct upstream model name; group = [Powerful `tier-cooled` p1, Powerful `tier-open` p2, untiered `overflow-second` p20, untiered `overflow-first` p10]. Put `tier-cooled` in cooldown via `router.metrics.set_provider_cooldown("tier-cooled", "rate limited".to_string(), now+300)`; open `tier-open:<model>` CB by setting `config.circuit_breaker.failure_threshold = 1` and calling `router.get_circuit_breaker("tier-open:<model>").await.record_failure().await`. Mount `Mock` POST `/v1/chat/completions` with `body_string_contains("\"model\":\"<overflow-first model>\"")` → `completion_response()` `.expect(1)`, and a catch-all for other models `.expect(0)` (or assert via `received_requests`). `route_request` → `Ok`; `response.extra["gateway_provider"] == "overflow-first"`; `response.extra` contains `gateway_smart_routing` with tier `powerful`.
      - `smart_routing_streaming_passthrough_when_reliability_enabled`: config `streaming: None` (defaults), one Powerful model on a mock returning SSE (`content-type: text/event-stream`, body `data: [DONE]\n\n`, matcher `body_string_contains("\"stream\":true")`, `.expect(1)`); `route_request_streaming` → `PassThrough { smart_routing: Some(d), .. }` with `d.tier == SmartRoutingTier::Powerful`.
      - `smart_routing_streaming_buffers_when_reliability_disabled`: same but `config.streaming = Some(StreamingConfig { emit_early_event: false, keepalive_interval_seconds: 0, ..Default::default() })` and mock returns `completion_response()` JSON; result is `Buffered(resp)` with `resp.extra` containing `gateway_smart_routing`.
      - `smart_routing_streaming_passthrough_reaches_overflow_candidate`: reliability default; Powerful `tier-cooled` in cooldown + untiered `overflow` on an SSE mock; `PassThrough { provider, .. }` has `provider == "overflow"`.
      Files: crates/ai-gateway/src/router/router.rs
      Verify: `cargo test -p ai-gateway router::router::tests::smart_routing_` and `cargo test -p ai-gateway streaming_requires_buffering_gate_table` pass.

- [x] 6. Brief wiki updates (no new files).
      `wiki/Streaming.md`: in the "Streaming Modes" table's Buffer-and-Replay "When Used" cell add "Smart Routing when Streaming Reliability (early event and keep-alive) is disabled"; one sentence under "True Streaming Pass-Through" that Smart Routing requests stream via pass-through over the smart-routing ordered provider list when Streaming Reliability is enabled (cascade escalation applies only to buffered responses).
      `wiki/Routing-and-Failover.md`: add a short "### Smart Routing Tier Overflow" under "Priority & Cost-Aware Routing": tier-selected models are tried first; if all are unavailable, the rest of the group's context-safe models (untiered, other tiers, non-specialists) are tried in normal priority/cost/latency order.
      Files: wiki/Streaming.md, wiki/Routing-and-Failover.md
      Verify: render check by reading the edited sections; no code impact.

- [x] 7. Final verification.
      Verify (PowerShell, long timeouts):
      - `cargo check -p ai-gateway --all-targets` — zero errors, zero warnings.
      - `cargo test -p ai-gateway` — all pass (timeout ≥ 1800s; `$env:PROPTEST_CASES = "64"` acceptable for speed).
      - Confirm no temp files remain and nothing was committed (`git status --short` lists only the intended source/wiki edits plus this task folder).

## Notes / accepted edge cases
- Pre-content streaming failover re-plans per attempt (existing behavior); metric is recorded only on the primary pass-through, but a pass-through that later falls back to buffered `route_request` records again (rare double count, accepted).
- Early-event SSE paths cannot carry Smart Routing headers (headers committed before routing) — unchanged from today's buffered early-event behavior.
- Cascade loop still escalates using `filter_by_tier` on the full group; unchanged by design.
