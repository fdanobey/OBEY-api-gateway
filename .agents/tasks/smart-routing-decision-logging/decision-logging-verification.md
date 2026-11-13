# Smart Routing decision logging: verification (iteration 2)

All commands run from `c:\Fbn_Dev_Apps\OBEY API Router` in PowerShell on Windows 11, on `master`, working tree uncommitted. This iteration resumed after a machine reboot; the review fixes were already in the tree and were re-checked against `git diff` before re-running everything below.

## Review findings

| # | Finding | Status |
|---|---|---|
| 1 (blocking) | Duplicate decision line on streaming re-plan → buffered | Fixed. `ActiveRequestInfo.smart_routing_logged` (`#[serde(skip)]`) plus `ActiveRequestHandle::claim_smart_routing_log()` returns the trace id once per request and `None` after. `SmartRoutingDecisionLog::emit` claims before logging, so the buffered exit (`route_request_with_exclusions`, including the primary-streaming 429 path with a non-empty `excluded`) and the streaming pass-through exit log at most once per request. All production re-plans in `handlers.rs` pass the same `active_handle`. The streaming exit keeps its `exclude.is_empty()` guard for the pre-existing metric. |
| 2 (non-blocking) | Diff over budget | `classify()` match arms restored; each arm now only drops `.clone()` / `.unwrap_or(...)`. Composite keeps its original shape (`if ml.fallback_reason.is_some() { return ml; }` replaces the `let … else`). Remaining size is the feature itself; see the line count below. |
| 3 (non-blocking) | Double WARN per System One HTTP failure | Kept both on purpose (the classifier WARN has error class + HTTP status, the router WARN has family + reason). The wiki now says a System One HTTP failure logs a second rate-limited WARN. |
| 4 (non-blocking) | Low-confidence fallback drops the confidence | `ClassifierFailure` carries no payload, and setting `Classification.confidence` on the fallback path would change the serialized `gateway_smart_routing.classifier_confidence`. So the System One `fallback_policy: fallback` branch logs the missed confidence and `min_confidence` at DEBUG (`System One low confidence; using heuristic`). Documented in the wiki. The INFO decision line is unchanged. |

## Commands and results

| Command | Result |
|---|---|
| `cargo check -p ai-gateway --all-targets --message-format short` | Exit 0. Zero errors, zero warnings (output contains only `Finished`). |
| `$env:PROPTEST_CASES=64; cargo test -p ai-gateway smart_routing` | Exit 0. lib: 262 passed, 0 failed. bin: 262 passed, 0 failed. `tests\performance.rs`: 1 passed. All other test binaries: 0 matched. |
| `$env:PROPTEST_CASES=64; cargo test -p ai-gateway router` | Exit 0. lib: 260 passed, 0 failed. bin: 260 passed, 0 failed. One integration binary: 1 passed. No `FAILED` in output. |
| `cargo test -p ai-gateway --test laya_classifier_integration --test jev_classifier_integration --test jev_performance` | Exit 0. laya: 11 passed. jev: 8 passed. jev_performance: 3 passed, 1 ignored (wall-clock test, `#[ignore]` by design). |
| `cargo check -p ai-gateway --all-targets --features ml-router --message-format short` | Exit 0. One warning: `ml_classifier.rs:47 constant ONNX_WEIGHTS_SUFFIX is never used`. Pre-existing: `ml_classifier.rs` is not in `git status`/`git diff`. |

No rustc ICE or corrupt incremental artifact this run, so no cache directories were deleted. (Iteration 1 deleted only `target\debug\incremental\token_compression_e2e-*` and `loop_detection_config-*` after an LTO ICE.)

## Changed lines

Counted with `git diff -U0` per file, attributing each `+`/`-` line to its new-file position and excluding lines at or after the file's `#[cfg(test)] mod tests` start.

| File | Non-test + | Non-test − |
|---|---|---|
| `src/smart_routing/mod.rs` | 157 | 44 |
| `src/router/router.rs` | 101 | 4 |
| `src/smart_routing/systemone/classifier.rs` | 41 | 3 |
| `src/smart_routing/config.rs` | 14 | 0 |
| `src/active_requests.rs` | 13 | 0 |
| `src/gateway/handlers.rs` | 1 | 0 |
| `src/smart_routing/budget_controller.rs` | 0 | 0 |
| Total | 327 | 51 |

Non-test total: 378 changed lines, still over the ~200 budget. The bulk is required by the task: `WarnRateLimiter` + `ClassifierFailure::as_str` (~50), the `fall_back` helper that replaces three duplicated fallback blocks (~40), `RoutingDiagnostics` threading (~30), `SmartRoutingDecisionLog` builder/emitter (~80), the System One call-failure WARN (~20), and `ClassifierMode::as_str` (14). Tests and fixtures: 226 inserted, 1 deleted. Wiki: 10 inserted.

## Tests covering the change

- `active_requests::tests::smart_routing_log_is_claimed_once_per_request` (new): the first claim returns the trace id. Later claims, including through a cloned handle, return `None`.
- `router::router::tests::smart_routing_streaming_passthrough_when_reliability_enabled` (extended): after the primary pass-through, `handle.claim_smart_routing_log()` is `None`, so any later re-plan on the request logs nothing.
- `smart_routing::tests::plan_carries_configured_classifier_fallback_reason`: Timeout, Backend, LowConfidence, and Unavailable (no classifier attached) reach `CandidatePlan.diagnostics`, with latency and candidate/overflow counts.
- `smart_routing::tests::plan_surfaces_blended_classifier_output`: the blended flag and confidence reach the plan.
- Composite classify test (extended): success has `fallback_reason=None` and latency. ML failure returns the heuristic with `fallback_reason=Backend`.
- `smart_routing::tests::fallback_warnings_are_rate_limited_per_family_and_reason`: one WARN per 60 s per `(family, reason)`.
- `router::router::tests::smart_routing_decision_log_fields_are_rounded_and_labeled`: score and confidence rounded to 3 decimals, string labels, latency rounded to whole ms.
- `smart_routing_untiered_models_only_overflow_after_tier_candidates` (extended): `diagnostics.candidates/overflow` = (1, 2).
- `laya_classifier_integration::laya_classifier_low_confidence_blend_policy_flags_blended_output`: a real classifier with `fallback_policy: blend` returns `blended=true`, and the flag survives a score-cache hit.

Not verified: no live run against the gateway or the router at 192.168.1.110 (out of scope by instruction). Tracing output is not asserted because the repo has no tracing-test helper. Once-per-request emission is tested at the flag level and on the streaming pass-through; there is no end-to-end streaming-truncation → re-plan → buffered test.
