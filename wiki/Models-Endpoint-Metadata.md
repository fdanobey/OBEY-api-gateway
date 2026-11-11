# Models Endpoint Metadata

The OpenAI-compatible `/v1/models` endpoint lists every model the gateway can serve — one entry per configured model group plus one entry per individual backing model. Entries now carry optional **capacity metadata** so clients can size their requests before sending them.

---

## Capacity fields

Each entry may expose:

| Field | Type | Meaning |
|---|---|---|
| `context_window` | integer | Total context capacity in tokens (input + output) |
| `max_completion_tokens` | integer | Maximum generatable output tokens |

Both fields are **optional**: when capacity is unknown for an entry, the keys are simply omitted rather than zero-filled.

### Group entries: MAX aggregation

A group entry (e.g. `gpt-4o`) fans out to multiple enabled provider/model candidates, each with its own declared `context_window`. The group entry advertises the **maximum** over the enabled candidates' *positive* declared windows — the largest context any candidate in the group can accept.

### Per-model entries: first declaration wins

An individual model (e.g. `gpt-4o-2024-11-20`) may back several groups. Its entry uses the model's declared `context_window`, and when the same model appears in multiple groups, the **first declaration wins**: the `context_window` from the first group that declares it is the one advertised. Later declarations do not override it.

---

## Retrieving a single model: `GET /v1/models/{model_id}`

Retrieve one model entry without paging the full list:

```
GET /v1/models/gpt-4o
```

Returns the same entry shape as the list endpoint. Semantics:

- **200** — the entry exists and is visible to the caller.
- **404** — the `model_id` is unknown, or it is hidden from this caller (e.g. a virtual key with a model-access allowlist that does not include it).

Virtual-key model-access restrictions apply identically to the list and retrieve endpoints.

---

## Upstream ingestion aliases

When the gateway refreshes its catalog from an upstream OpenAI-compatible `/v1/models` response, providers name capacity fields differently. The gateway accepts all of the following aliases:

| Upstream field | Source flavor | Maps to |
|---|---|---|
| `context_length` | OpenRouter | `context_window` |
| `max_model_len` | vLLM | `context_window` |
| `max_output_tokens` | LiteLLM | `max_completion_tokens` |

`max_input_tokens` is **deliberately ignored**: input-only limits would understate total capacity, and the two supported capacity notions (total window, max output) are sufficient for sizing requests.

---

## Admin UI

No new admin controls are required. Context-window sizing for individual models is already configurable via the existing `mdl-context-window` control in the admin panel's model-group editor, and the capacity fields described here are read from that same configuration plus upstream ingestion.
