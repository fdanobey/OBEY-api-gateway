# Structured Output Validation

Structured Output Validation ensures that model responses conform to the JSON Schema declared in the caller's `response_format` field. When a response fails validation, the gateway automatically retries with corrective prompts — without the caller needing any retry logic.

---

## How It Works

```
┌──────────────────────────────────────────────────────────────────┐
│                  Structured Output Flow                           │
│                                                                  │
│  Request with response_format ──► Provider ──► Response          │
│                                                      │           │
│                                              Validate against    │
│                                              JSON Schema         │
│                                                      │           │
│                                    ┌─────────────────┼──────┐    │
│                                    │ Valid           │Invalid│    │
│                                    ▼                 ▼       │    │
│                              Return to         Build retry   │    │
│                              caller            with errors   │    │
│                                                      │       │    │
│                                              Re-send to      │    │
│                                              provider        │    │
│                                              (up to N times) │    │
│                                    └─────────────────────────┘    │
└──────────────────────────────────────────────────────────────────┘
```

1. Caller sends a request with `response_format: { type: "json_schema", json_schema: {...} }`
2. Gateway routes to provider, gets response
3. Gateway validates the response content against the declared schema
4. If valid → return immediately
5. If invalid → build a corrective retry prompt including the schema violations, re-send to the provider
6. Repeat up to `max_retries` times
7. Return the last response (valid or not) to the caller

---

## Admin Panel

![Structured Output Admin](images/admin-structured-output.png)

The Structured Output tab provides:
- **Enable/disable** toggle for the validation system
- **Max Retries** — corrective retry attempts (0–5)
- **Retry Temperature** — temperature for corrective requests (0.0–2.0, lower = more deterministic)
- **Passthrough Providers** — providers that skip validation (they handle it natively)

---

## Configuration

```yaml
structured_output:
  enabled: true
  max_retries: 1
  retry_temperature: 0
  passthrough_providers:
    - openai
    - anthropic
```

### Fields

| Field | Default | Description |
|-------|---------|-------------|
| `enabled` | `true` | Enable structured output validation globally |
| `max_retries` | `1` | Corrective retry attempts before returning the failed response (0–5) |
| `retry_temperature` | `0` | Temperature for corrective retry requests (0.0–2.0) |
| `passthrough_providers` | `[]` | Provider names that skip validation (native constrained decoding) |

---

## Passthrough Providers

Some providers (OpenAI, Anthropic) support native constrained decoding — they enforce JSON Schema compliance at the token generation level. For these providers, gateway-side validation is redundant.

List them in `passthrough_providers` to skip post-response validation:

```yaml
structured_output:
  passthrough_providers:
    - openai
    - anthropic
```

The gateway still processes the request normally; it just skips the validate-and-retry loop for these providers.

---

## Per-Model-Group Overrides

Model groups can override the global structured output settings:

```yaml
model_groups:
  - name: coding-group
    structured_output:
      enabled: true
      max_retries: 3    # More retries for code generation
    models:
      - provider: openai
        model: gpt-4.1
```

---

## Per-Provider-Model Passthrough

Individual provider-model entries can explicitly enable or disable native structured-output passthrough:

```yaml
model_groups:
  - name: mixed-group
    models:
      - provider: openai
        model: gpt-4.1
        structured_output_passthrough: true   # Skip validation
      - provider: ollama
        model: llama3
        structured_output_passthrough: false  # Always validate
```

---

## How Corrective Retries Work

When validation fails, the gateway builds a corrective prompt:

1. Includes the original schema
2. Lists the specific violations found (missing fields, wrong types, extra properties)
3. Asks the model to fix the response
4. Lowers the temperature to `retry_temperature` for more deterministic output

Example corrective message:
```
The previous response did not conform to the required JSON Schema.
Violations found:
- $.name: missing required field
- $.age: expected integer, got string

Please provide a corrected response that matches the schema exactly.
```

---

## Prometheus Metrics

```
obey_structured_output_validations_total{provider, model, status}
```

Where `status` is one of:
- `valid` — passed validation on first attempt
- `corrected` — passed after corrective retry
- `failed` — failed after all retries exhausted
- `skipped` — passthrough provider, validation not attempted

---

## Next Steps

- [Configuration](Configuration) — full config reference
- [Routing & Failover](Routing-and-Failover) — how requests reach providers
- [Admin Panel & Dashboard](Admin-Panel-and-Dashboard) — web UIs
