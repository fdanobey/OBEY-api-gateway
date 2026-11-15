# Model Deprecation Alerts

Providers retire models. When a model you route to is scheduled for shutdown — or has already been removed — requests start failing with 404s that look like routing bugs. Model deprecation alerts watch for this: the gateway scans provider model listings for deprecation metadata, combines it with operator overrides and runtime 404/410 evidence, sends a `model.deprecated` webhook when a model you actually use is on its way out, and fails over around models it knows are gone.

The feature needs no configuration to run its scanners, but it stays quiet by default: with no webhook endpoints configured, nothing is sent. See [Dormancy and dedup](#dormancy-and-dedup) for the full contract.

---

## How It Works

The pipeline has four stages:

1. **Scan** — every 24 hours (plus one pass shortly after startup), the gateway fetches `GET {base}/models` from each configured OpenAI-compatible provider and parses per-model deprecation fields. A manual scan can be triggered from the admin panel at any time.
2. **Evaluate** — scan results are merged with operator `deprecation_date`/`replacement_model` overrides from config (overrides win) and checked against the warn window. A model inside the window opens an alert episode.
3. **Alert** — if webhook endpoints exist and the model received traffic in the last 7 days, a `model.deprecated` event is delivered once per episode.
4. **Failover** — independently of scanning, responses with HTTP 404/410 and a model-not-found body mark the model gone: the failed candidate is skipped, failover kicks in without a same-provider retry, and later requests skip that `provider:model` pair until the gone mark expires.

Runtime detection means the gateway reacts to a removal the moment a request hits it — no need to wait for the next scan.

---

## Provider Scanning

The scanner only queries provider types that serve an OpenAI-compatible `/models` listing behind plain Bearer auth:

| Scanned | | | |
|---|---|---|---|
| `openai` (API-key auth) | `groq` | `together` | `nvidia_nim` |
| `vllm` | `lmstudio` | `ollama` | `custom` |

| Excluded | Reason |
|---|---|
| `anthropic` | Different auth shape (`x-api-key` + `anthropic-version` headers) — not reachable via the Bearer fetch |
| `azure` | Different auth shape (`api-key` header + `api-version` query parameter) |
| `google` | Different auth shape (key as query parameter) |
| Codex (`openai` + `auth_method: oauth`) | OAuth token is scoped away from `/models` |
| `github-copilot` (device OAuth) | OAuth token is scoped away from `/models` |
| `bedrock` | No OpenAI-compatible models endpoint |
| `jev` / `laya` / `systemone` | Decision-family wire protocol, no models listing |

Each scan pass has a 10-second per-provider timeout and runs through the outbound-fetch SSRF guard. A failed provider scan never fails the loop: the provider's previous scan data is kept untouched and exactly one `warn` is logged with the provider name and an error class. No URLs, credentials, or response bodies appear in logs.

While a scan is in flight, a manual trigger returns `already_running` instead of starting a second concurrent pass.

---

## Parsed Metadata Fields

Model listings from different providers name their deprecation fields differently. The scanner reads all of them:

| Listing field | Meaning |
|---|---|
| `shutdown_date` | Planned retirement date (OpenAI shape) |
| `expiration_date` | Planned retirement date (OpenRouter shape) |
| `deprecation_date` | Planned retirement date (generic shape) |
| `deprecated: true` | Date-less flag: model is deprecated, no date known |
| `successor_model` | Suggested replacement model id |

When more than one date field is present, precedence is `shutdown_date` > `expiration_date` > `deprecation_date`. Both plain dates (`2026-12-20`) and full timestamps (RFC 3339) are accepted.

---

## Runtime Detection and Failover

When an upstream response has status **404 or 410** and its body mentions `model_not_found`, "does not exist", or "deprecat" (case-insensitive substring — so "deprecated" and "deprecation" both match), the gateway treats it as a removed model rather than a retryable error:

- The error is classified failover-class: the failed `provider:model` is excluded from this request's candidates and the next candidate is tried — no same-provider retry.
- The model is marked gone, and subsequent requests skip that `provider:model` pair in candidate selection.
- If no candidate remains, the request terminates with the aggregated error; attempt-ledger rows record the outcome with a `model_not_found` error class.

A 404 whose body does not match the signature stays terminal — the gateway does not guess.

Gone marks are bounded: they expire 24 hours after detection unless re-confirmed, so a provider outage that mimics a 404 cannot blackhole a model forever. A config hot-reload drops deprecation state for providers removed from the config and keeps everything else (scan data, episodes, gone marks of still-configured providers persist — a reload must not re-open routing to a dead model).

---

## Configuration

Two knobs control evaluation, both optional.

### Warn window

```yaml
model_deprecations:
  warn_days_before: 30 # 1..=365; default 30
```

A model alerts once its retirement date is this many days away (or past). Raise it if you want more lead time to migrate.

### Per-model overrides

Operator-declared facts win over provider metadata — useful when a provider's listing is missing or wrong, or when you know a retirement date before the provider publishes it:

```yaml
model_groups:
  - name: "gpt-4-group"
    models:
      - provider: "openai"
        model: "gpt-4-turbo"
        # Optional deprecation overrides:
        deprecation_date: "2026-12-20" # ISO 8601 date
        replacement_model: "gpt-4.1" # reported in the alert payload
```

`deprecation_date` must parse as an ISO 8601 date; validation errors name the model group, provider, and model. `replacement_model` is carried into the `model.deprecated` event payload as the suggested successor. Both fields are also editable per-model in the admin panel.

---

## Webhook Events

With webhook endpoints configured, a qualifying deprecation emits a `model.deprecated` event. Endpoints can select it via the `events` filter (see the webhooks section of `config.example.yaml`); delivery, signing, and retries work exactly like the budget-alert events.

Payload example (field set and names as they appear on the wire):

```json
{"id":"00000000-0000-4000-8000-00000000000a","type":"model.deprecated","created_at":"2026-11-14T19:00:00Z","signed":false,"data":{"provider":"openai","model":"gpt-4-turbo","retirement_date":"2026-12-20","days_remaining":36,"replacement_model":"gpt-4.1","evidence_source":"provider_metadata","requests_7d":1523,"last_used_at":"2026-11-14T18:22:09Z","runtime_gone":false}}
```

| Field | Meaning |
|---|---|
| `data.provider` | Provider the deprecated model belongs to |
| `data.model` | Deprecated provider model id |
| `data.retirement_date` | Known retirement date; omitted on date-less deprecations |
| `data.days_remaining` | Whole days until retirement; omitted when the date is unknown |
| `data.replacement_model` | Suggested successor (scan `successor_model` or operator override); omitted when unknown |
| `data.evidence_source` | `provider_metadata`, `operator_override`, or `runtime_detection` — which signal opened the episode |
| `data.requests_7d` | Requests routed to this exact `provider:model` pair over the last 7 days |
| `data.last_used_at` | Last request timestamp from the usage ledger; omitted when no recent entry exists |
| `data.runtime_gone` | `true` when runtime detection saw the model gone (404/410) — the stronger signal |

A date-less runtime detection (`deprecated: true` with no date, or a 404 on a model the scanner never saw dated) omits the date fields:

```json
{"id":"00000000-0000-4000-8000-00000000000b","type":"model.deprecated","created_at":"2026-11-14T19:00:00Z","signed":false,"data":{"provider":"openrouter","model":"meta-llm/llama-3-70b-instruct","evidence_source":"runtime_detection","requests_7d":7,"runtime_gone":true}}
```

### Dormancy and dedup

The emission side is deliberately quiet:

- **No endpoints, no events.** With no webhook endpoints configured, evaluation is skipped entirely — nothing is sent, nothing is written.
- **Usage gate.** Only models with at least one routed request in the last 7 days alert. A deprecation on a model you never call is not your alert.
- **Episode dedup.** One event per episode. A model sitting inside the warn window does not re-alert on every scan; escalation from warn to runtime-gone sends exactly one more event. Restarting the gateway re-arms episodes, so at most one duplicate per episode can arrive after a restart.

---

## Admin Panel

The admin panel exposes the feature under **Model Deprecations**:

- Status table: per `provider:model` row — retirement date, days remaining, evidence source, recent request count, and gone state.
- **Scan now** button (backed by `POST /admin/model-deprecations/scan`): starts a scan pass immediately, or reports `already_running` when one is in flight.
- Per-model override fields (Deprecation Date, Replacement Model) in the model edit dialog, editing the same config fields hot-reload applies.
- The warn-days control (`warn_days_before`) and the `model.deprecated` option in the webhook event filter.

Plain HTTP equivalents for scripts:

```text
GET  /admin/model-deprecations
POST /admin/model-deprecations/scan
```

---

## Dashboard

`model.deprecated` rows appear in the dashboard's recent-alerts feed alongside budget alerts, with the model and provider named. The threshold column is blank for deprecation rows (they carry no percentage).
