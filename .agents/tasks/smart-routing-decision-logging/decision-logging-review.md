# Per-request Smart Routing decision log and classifier fallback reasons (pass 2)

The change adds one content-free `Smart routing decision` INFO line per routed request, fed by a `RoutingDiagnostics` struct (configured classifier, fallback reason, blended flag, classifier latency, candidate/overflow counts) carried on `CandidatePlan`. `classify_optional` now returns an annotated `Classification` instead of `Option`, and a shared `fall_back` helper records the existing fallback metrics plus a WARN limited to one per 60 s per family and reason. Since pass 1, the duplicate-line bug on streaming re-plans is fixed with a once-per-request claim flag on `ActiveRequestHandle`. The `classify()` arms are restored to their original shape. The low-confidence miss is now logged at DEBUG, and the wiki covers the double WARN for System One.

Watch for: the non-test diff is still about 378 changed lines against the ~200 budget. The excess comes from what the task asks for (rate limiter, dedupe flag, field threading), not from a refactor (confirmed). The wiki paragraph has a few small inaccuracies: streaming timing, the `backend`/`unavailable` mapping, and the "second WARN" claim for timeouts (confirmed).

**Verdict**: APPROVED

## High-level view

Pass 1's blocking issue is closed. `SmartRoutingDecisionLog::emit` claims a `#[serde(skip)]` flag under the handle's existing mutex before logging. Every production streaming call and re-plan in `handlers.rs` passes a clone of the same registered handle. So the primary pass-through, the 429 → `route_request_with_exclusions` fallback, and every other `route_request` fallback inside the streaming function share one claim. No privacy leak was found. The line, both WARNs, and the new DEBUG carry only enum labels, numbers, bools, HTTP status, and the gateway trace id. The System One WARN deliberately skips the `Unavailable { reason }` string. No config fields or metric labels were added.

The remaining gaps are non-blocking. Size is still about 1.9× the budget, but nothing obvious is left to trim without dropping a requirement. Dedupe is tested at the flag level and on the primary pass-through, not end to end through a re-plan. The wiki overstates two things. Streaming logs when the pass-through is established, not when the request completes. Transport failures surface as `unavailable`, not `backend`.

<details>
<summary>Issues (3)</summary>

1. **Diff still over the ~200-line budget** (non-blocking, confirmed): about 327 inserted and 51 deleted non-test lines. Pass 1's suggested trim was applied, and what remains is the rate limiter, the dedupe flag, field threading, and the log builder. Accept it as is, or drop `ClassifierMode::as_str` (14 lines) if an existing label helper can be reused.
2. **Wiki wording on streaming timing, failure mapping, and the second WARN** (non-blocking, confirmed): "Each completed request" and "Requests that fail on every provider log no line" only hold for the buffered path. A streaming pass-through logs when the upstream stream is established and stays logged if the relay later fails on every re-plan. `backend (HTTP/auth/transport)` is also wrong: System One transport and unexpected-status errors map to `unavailable`, and 422 maps to `invalid_output`. "A System One HTTP failure logs a second rate-limited WARN" doesn't cover gateway-side timeouts, which return before the call-failure arm and log only the router WARN. Reword these three phrases.
3. **No end-to-end dedupe test through a streaming re-plan** (non-blocking, confirmed): the claim flag is unit-tested, and the pass-through test checks that the claim was consumed. Nothing drives pass-through → re-plan → 429 → buffered and checks that the second exit skips the line. Add one if a mock-provider harness for that path is cheap. Otherwise it is acceptable, since tracing output can't be asserted here anyway.

</details>

<details>
<summary>Details</summary>

### Once-per-request claim

```
handler registers ActiveRequestHandle (one Arc per request)
  route_request_streaming(h)              -> PassThrough, exclude=[]  -> emit -> claim() = Some(trace) -> LOG
    relay fails before content
  route_request_streaming_excluding(h, tried)
    429 -> route_request_with_exclusions(h, excluded) -> emit -> claim() = None -> skip
    or PassThrough (exclude non-empty)                 -> guard + claim -> skip
```

The claim sits inside `emit`, so both success exits share one gate. The streaming exit keeps its `exclude.is_empty()` guard only for the existing metric. When the primary streaming attempt hits a 429, the buffered path runs with a non-empty `excluded` before anything has logged. The claim still logs that line, where an `exclude` gate would have dropped it. The logged line describes the first successful plan. A later re-plan may re-run the classifier and land elsewhere, and that outcome is not logged. If the handle mutex is poisoned, `claim` returns `None` and the line is silently dropped (fail-silent, like the other handle setters). The assistants-run path routes with `None`, so it logs once and has no `trace_id`.

### Classifier fallback WARNs

The router WARN skips `LowConfidence` (expected) and `NoJevAtEndpoint` (discovery already warns). Composite failures warn under the `composite` family rather than `ml`. That matches the config, but an operator chasing an ML backend failure has to know the mapping. When the heuristic assessment itself errors in heuristic-only mode, the decision line carries a `fallback_reason` with `classifier_configured=heuristic` and no WARN. The wiki's "configured classifier fell back" description doesn't cover that case.

### Test coverage

Covered:

- fallback-reason propagation for Timeout, Backend, LowConfidence, and Unavailable (no classifier attached), with latency present and absent
- the blended flag and confidence on the plan
- the blended flag surviving a Laya score-cache hit against a mock endpoint
- composite success and ML-failure annotation
- the rate-limiter windows per key
- log-field rounding and labels
- overflow counts
- the once-only claim, including through a cloned handle

Not tested: the end-to-end streaming re-plan dedupe (issue 3), the System One call-failure WARN branch (no tracing helper), and the heuristic-only `fallback_reason`.

</details>

<details>
<summary>File map</summary>

- `crates/ai-gateway/src/active_requests.rs`: `smart_routing_logged` flag (serde-skipped), `claim_smart_routing_log()`, unit test.
- `crates/ai-gateway/src/gateway/handlers.rs`: initialize the new flag.
- `crates/ai-gateway/src/router/router.rs`: `SmartRoutingDecisionLog` builder/emitter, `diagnostics` on `RoutedProviders`, emits at the buffered and pass-through exits, field-format test, extended streaming and overflow tests.
- `crates/ai-gateway/src/smart_routing/mod.rs`: `fallback_reason`/`blended`/`classifier_latency_ms` on `Classification`, `blended` on `ClassifierOutput`, `ClassifierFailure::as_str`, `WarnRateLimiter`, `RoutingDiagnostics` on `CandidatePlan`, `fall_back` helper, tests.
- `crates/ai-gateway/src/smart_routing/systemone/classifier.rs`: blended flag in output and score cache, rate-limited call-failure WARN (class and status), DEBUG on low-confidence fallback.
- `crates/ai-gateway/src/smart_routing/config.rs`: `ClassifierMode::as_str`.
- `crates/ai-gateway/src/smart_routing/budget_controller.rs`: test fixture fields.
- `crates/ai-gateway/tests/laya_classifier_integration.rs`: blend-policy integration test with cache hit.
- `wiki/Routing-and-Failover.md`: "Smart Routing Decision Log" section.

The untracked `wiki-local/` directory predates this task (last written 2026-10-03, no decision-log content) and is not part of the change.

Full diff: `git -C "c:\Fbn_Dev_Apps\OBEY API Router" diff`.

</details>
