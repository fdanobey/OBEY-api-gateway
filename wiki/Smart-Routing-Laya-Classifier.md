# Smart Routing with Laya

Laya is Convai Innovations' System One decision model family. It is wire-compatible with Jev: a single `/v1/systemone` call evaluates six complexity dimensions and a task-type choice, returning typed Score and Choice answers. OBEY applies those answers to the same tier selection, budgets, context filtering, specialization, and cascade behavior used for Jev.

This page covers Laya-specific behavior. For shared System One concepts (tier selection, dimension weights, fallback chain), see [Smart Routing with Jev](Smart-Routing-Jev-Classifier.md).

## Why Laya

Jev and Laya are interchangeable from the routing engine's perspective. Choose Laya when:

- your System One provider hosts Laya checkpoints (e.g. `convai/laya-*` on OpenRouter);
- you run a self-hosted `laya-serve` endpoint inside your network;
- you want Convai's published decision checkpoints (`english`, `multilingual`, `typed-decisions`).

## Setup

Hosted default:

```yaml
smart_routing:
  enabled: true
  classifier: laya
  laya:
    api_key_env: LAYA_API_KEY
    base_url: https://api.laya-ai.com
    model: auto
```

Self-hosted `laya-serve`:

```yaml
smart_routing:
  enabled: true
  classifier: laya
  laya:
    base_url: http://laya-serve.internal:8080
    model: typed-decisions
```

HTTPS is required for public endpoints. Plain HTTP is permitted only for localhost and RFC 1918/private-range hosts, so self-hosted `laya-serve` instances behind a trusted network can be configured without TLS termination.

## Checkpoints and model selection

- `model: auto` queries the endpoint's model listing, keeps ids containing a word-delimited `laya` token, and selects the newest version (semantic components compare numerically). Discovery is cached for `discovery_ttl_secs` (default 600 seconds).
- When the endpoint exposes no listing (self-hosted `laya-serve`), `auto` pins the `typed-decisions` checkpoint — the default for typed routing decisions.
- Pin a checkpoint explicitly for reproducibility: `english`, `multilingual`, or `typed-decisions`.
- If a pinned model is absent from the endpoint, each request falls back to the heuristic classifier chain; the generation request never fails. Missing-Laya warnings are rate-limited to once per discovery TTL window.

## Confidence gating

Laya responses carry two confidence fields:

- `answer_confidence` — the checkpoint's own calibration of the answer;
- `confidence` — the raw probability-based confidence.

The gate reads `answer_confidence` and falls back to `confidence` when the field is absent. Defaults:

```yaml
laya:
  min_confidence: 0.55      # applied to answer_confidence
  min_task_confidence: 0.50
  fallback_policy: fallback  # fallback | blend
```

- `fallback`: uncertain classifications use the existing heuristic/ML/LLM fallback behavior.
- `blend`: uncertain Laya scores blend with the heuristic score in proportion to confidence.
- Low task-type confidence keeps heuristic task detection even when the complexity score is accepted.

Thresholds are policy controls. Validate on your own traffic before tightening.

## Resilience

- Timeout: 250–2000 ms; default 800 ms.
- Retries: HTTP 429, 529, and 503, at most two retries, exponential backoff, `Retry-After` honored within the time budget.
- No retry: 401, 403, or 422 (operator action required).
- 413 payload-limit responses halve the `char_budget` (floor 256) instead of failing, so oversized states are retried compacted rather than dropped.
- Response bodies are capped and payloads are never logged.
- A 1000-entry, 5-minute SimHash cache avoids classifying identical requests repeatedly.

## Privacy and security

The gateway sends only the latest user message (or latest non-tool message), bounded by `char_budget`. Tool-result content is excluded. The API key is resolved from `api_key_env` and is redacted from debug output, logs, metrics, dashboard responses, and request records. No prompt material is exported into metric labels.

## Metrics

System One metrics are labeled by family (`jev` or `laya`):

- `obey_api_smart_routing_jev_consults_total{family="laya"}`
- `obey_api_smart_routing_jev_fallbacks_total{family="laya"}`
- `obey_api_smart_routing_jev_confidence{family="laya"}`
- `obey_api_smart_routing_jev_latency_ms{family="laya"}`
- `obey_api_smart_routing_jev_discovery_refreshes_total{family="laya"}`

Labels are bounded; prompts, model-group names, API keys, and response content are not used as raw metric labels.

## Admin and dashboard

The Admin Panel's **Smart Routing → Laya Classifier** section exposes every `laya` field with the same masked-key handling as Jev. The Dashboard's **Smart Routing** tab shows the Laya endpoint, model, and confidence gate. `gateway_smart_routing` response metadata includes optional `classifier_confidence` and `resolved_model` fields for Laya decisions; the fields are omitted from non-System One decisions.

### Credential storage

`laya.api_key` is a startup-only input and is never written to disk in plaintext. When a key is saved through the Admin Panel, it is encrypted at rest (the same master-key encryption used for provider keys) and stored in `laya.api_key_encrypted`; the admin API never returns the stored key, and saving other settings without retyping it preserves the stored value. `laya.api_key_env` remains the recommended setup.

### Per-group trust overrides

Model groups can require stricter confidence than the global policy via `model_group_overrides.<group>.jev_trust` (`min_confidence`, `min_task_confidence`, `fallback_policy`). The override block is shared between Jev and Laya; when the active family is Laya the override thresholds apply to the Laya confidence gate.

## Migrating from Jev

1. Set `classifier: laya` and add the `laya:` block (base URL, key, model).
2. Re-point per-group trust overrides only if you also tuned `jev_trust` values; the same block applies.
3. Metrics queries filtering `family="jev"` need a new series for `family="laya"`; alert thresholds carry over unchanged because the confidence semantics are the same.
4. Jev and Laya may both be configured simultaneously; only the `classifier:` value determines which is active at runtime.
