# Smart Routing with Jev

Jev is TypeSafe AI's System One decision model. Instead of generating a text label and requiring the gateway to parse it, Jev returns typed Score and Choice answers with probability distributions and confidence. OBEY uses those answers to select a Fast, Balanced, or Powerful model tier before generating a response.

## Why use it

The existing heuristic classifier is local, deterministic, and free. Jev is useful when request complexity cannot be captured reliably by token counts and keywords alone:

- reasoning depth;
- tool-call coupling;
- synthesis across context;
- strict output-precision requirements;
- specialist code/math/domain load;
- ambiguity and underspecification.

All six dimensions and a task-type choice are evaluated in one System One call. The gateway normalizes and weights the dimensions in code, then applies the same existing decision engine, budgets, context filtering, tier specialization, and cascade behavior.

## Provider-neutral setup

TypeSafe is the default endpoint:

```yaml
smart_routing:
  enabled: true
  classifier: jev
  jev:
    api_key_env: JEV_API_KEY
    base_url: https://api.typesafe.ai
    model: auto
```

The base URL is configurable. A compatible router such as OpenRouter can be selected without changing the routing policy:

```yaml
smart_routing:
  enabled: true
  classifier: jev
  jev:
    api_key_env: OPENROUTER_API_KEY
    base_url: https://openrouter.ai/api
    model: auto
```

The endpoint must use HTTPS and expose compatible `/v1/models` and `/v1/systemone` paths.

## Model auto-selection

`model: auto` queries the configured endpoint's model listing, keeps ids containing a word-delimited `jev` token, and selects the newest version. Semantic version components compare numerically (`1.13` is newer than `1.9`). Discovery is cached for `discovery_ttl_secs` (default 600 seconds).

Pin a concrete model id when reproducibility is more important than automatic upgrades:

```yaml
jev:
  model: typesafe/jev-1.13.0
```

If no Jev model is available, discovery fails, a pinned model is unavailable, or the Jev endpoint times out, smart routing falls back to the existing classifier chain. The generation request itself does not fail. Missing-Jev warnings are rate-limited to once per discovery TTL window.

## Confidence gating

The default policy trusts Jev only when its least-confident complexity dimension meets `min_confidence`:

```yaml
jev:
  min_confidence: 0.60
  min_task_confidence: 0.50
  fallback_policy: fallback
```

- `fallback`: uncertain Jev classifications use the existing heuristic/ML/LLM fallback behavior.
- `blend`: uncertain Jev scores blend with the heuristic score in proportion to confidence.
- Low task-type confidence keeps heuristic task detection even when the complexity score is accepted.

Thresholds are policy controls, not universal constants. Validate them on your own traffic before reducing cost through more aggressive Fast-tier selection.

## Dimension weights

Weights are finite, non-negative, and cannot all be zero. They do not need to sum to one; the gateway normalizes them.

```yaml
jev:
  dimension_weights:
    reasoning_depth: 0.30
    tool_coupling: 0.20
    context_synthesis: 0.15
    output_precision: 0.15
    domain_load: 0.15
    ambiguity: 0.05
```

## Resilience

- Timeout: 250–2000 ms; default 1000 ms.
- Retries: only HTTP 429 and 529, at most two retries, exponential backoff, `Retry-After` honored within the time budget.
- No retry: 401, 403, or 422 (operator action is required).
- Response bodies are capped and payloads are never logged.
- A 1000-entry, 5-minute SimHash cache avoids classifying identical requests repeatedly.

## Privacy and security

The gateway sends only the latest user message (or latest non-tool message), bounded by `char_budget`. Tool-result content is excluded. The API key is resolved from `api_key_env` and is redacted from debug output, logs, metrics, dashboard responses, and request records. Configure credentials through environment variables where possible.

## Metrics

When smart routing is enabled, `/metrics` exposes:

- `obey_api_smart_routing_jev_consults_total`
- `obey_api_smart_routing_jev_fallbacks_total`
- `obey_api_smart_routing_jev_confidence`
- `obey_api_smart_routing_jev_latency_ms`
- `obey_api_smart_routing_jev_discovery_refreshes_total`

Labels are bounded; prompts, model-group names, API keys, and response content are not used as raw metric labels.

## Admin and dashboard

The Admin Panel's **Smart Routing → Jev Classifier** section exposes all Jev configuration fields. The Dashboard's **Smart Routing** tab shows the configured endpoint, model mode, and confidence gate. `gateway_smart_routing` response metadata includes optional `classifier_confidence` and `resolved_model` fields for Jev decisions; the fields are omitted from non-Jev decisions.

### Credential storage

`jev.api_key` is a startup-only input and is never written to disk in plaintext. When a key is saved through the Admin Panel, it is encrypted at rest (the same master-key encryption used for provider keys) and stored in `jev.api_key_encrypted`; the admin API never returns the stored key, and saving other settings without retyping it preserves the stored value. `jev.api_key_env` remains the recommended setup: either an environment variable name or a literal that the admin save path will encrypt on first write.

### Per-group trust overrides

Model groups can require stricter confidence than the global policy via `model_group_overrides.<group>.jev_trust` (`min_confidence`, `min_task_confidence`, `fallback_policy`). Overrides apply at classification time; credentials and endpoint settings stay global.
