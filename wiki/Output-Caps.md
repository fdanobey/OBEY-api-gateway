# Output Caps

Output caps let you set a hard ceiling on the number of output tokens a request may produce, either per virtual key or per model group. The ceiling applies to every request that flows through the gateway, regardless of which provider serves it.

Caps are **ceiling-only** — they lower a request's output limit when the limit is missing or exceeds the cap, and never raise it. A client that asks for fewer tokens than the cap always gets what it asked for.

When no caps are configured anywhere, behavior is byte-identical to a gateway without the feature.

---

## Where Caps Live

| Scope | Field | Applies To |
|-------|-------|------------|
| Virtual key | `max_output_tokens` | Every request authenticated with that key |
| Model group | `max_output_tokens` | Every request routed to that group |

When both are set, the **lower** of the two wins.

---

## How Caps Are Applied

For each incoming request the gateway:

1. Resolves the effective cap: `min(virtual_key_cap, model_group_cap)` — ignoring whichever side is unset.
2. Reads the request's existing output-token field for its dialect (see table below).
3. If the field is **absent**, injects the cap as a typed value.
4. If the field is **present and above the cap**, lowers it to the cap.
5. If the field is **present and at or below the cap**, leaves it untouched.

The client's original value is never raised, and a cap is never applied above what the client asked for.

---

## Dialect Coverage

| Endpoint | Field(s) Clamped |
|----------|-----------------|
| `/v1/chat/completions` | `max_tokens`, `max_completion_tokens`, `max_output_tokens` |
| `/v1/completions` | `max_tokens` |
| `/v1/messages` (Anthropic) | `max_tokens` |
| `/v1/responses` | `max_output_tokens` |

> **Codex dispatch path:** Requests routed through the Codex OAuth backend are **not** clamped. This is a known limitation of the current dispatch path.

---

## Configuration

### Per Model Group (YAML)

```yaml
model_groups:
  - name: expensive-models
    max_output_tokens: 4096
    models:
      - provider: openai
        model: gpt-4
```

Omit `max_output_tokens` (or set it to `null`) to leave the group uncapped.

### Per Virtual Key

Set `max_output_tokens` when creating or updating a key through the admin API or the **Virtual Keys** tab in the admin panel.

```bash
curl -X POST http://localhost:8080/admin/keys \
  -H 'Content-Type: application/json' \
  -d '{"name":"team-a","max_output_tokens":2048}'
```

---

## Admin UI

- **Virtual Keys tab** — the create/edit form includes a **Max output tokens** field. Leave it blank for no cap.
- **Model Groups editor** — each group card exposes a **Max output tokens** field alongside the existing priority and cost settings.

Changes to a virtual key's cap take effect on the **next request** — no restart or reload needed. Changes to a model group's cap apply via **config hot-reload** (`POST /admin/config/reload` or the admin panel Save button).

---

## Observability

| Signal | Detail |
|--------|--------|
| Prometheus counter | `gateway_output_caps_applied_total` — incremented each time a cap lowers or injects a value |
| Request log metadata | `output_cap_metadata` — records `applied_value` (the cap that was enforced) and `originals` (each declared carrier with the client's pre-clamp value; empty when the cap was injected) |

---

## Default Behavior

With no `max_output_tokens` set on any virtual key or model group, the gateway makes no modifications to output-token fields. Existing deployments are unaffected.
