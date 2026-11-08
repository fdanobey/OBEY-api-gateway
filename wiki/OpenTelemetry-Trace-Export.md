# OpenTelemetry Trace Export

OBEY API Gateway can emit one distributed trace per client request — a root span for the gateway hop plus one child span per provider attempt — to any OTLP/HTTP-compatible backend (OpenTelemetry Collector, Jaeger, Grafana Tempo, SigNoz, Langfuse, Honeycomb, Datadog). Attempt spans follow the OpenTelemetry GenAI semantic conventions, so token and cost data render natively in LLM-aware backends without custom parsing.

The feature is disabled by default, fully self-hosted, and fail-open: trace export can never add latency to, or fail, a client request.

---

## Enablement

Add an `opentelemetry` section to `config.yaml` (or use the admin panel's Tracing page):

```yaml
opentelemetry:
  enabled: true
  endpoint: "http://localhost:4318/v1/traces"
  headers:                      # optional; values support ${ENV_VAR} substitution
    Authorization: "Bearer ${OTEL_AUTH_TOKEN}"
  sampling_ratio: 1.0           # 0.0–1.0
  capture_content: false        # privacy: see below
  service_name: "ai-gateway"    # resource service.name
```

| Field | Default | Notes |
|-------|---------|-------|
| `enabled` | `false` | When `true`, a non-empty http/https `endpoint` is required |
| `endpoint` | — | Full OTLP/HTTP traces URL, e.g. `http://localhost:4318/v1/traces` |
| `headers` | `{}` | Added to every export POST; values resolve `${ENV_VAR}` at load/reload |
| `sampling_ratio` | `1.0` | Deterministic head-sampling ratio in `[0.0, 1.0]` |
| `capture_content` | `false` | When `true`, bounded prompt/completion content is attached to spans |
| `service_name` | `"ai-gateway"` | Resource `service.name`; distinguish multi-instance estates |

An absent section is fully disabled with zero overhead (a single enabled check per request). A dormant-but-broken section is still reported by config validation at load and hot-reload.

Settings apply via hot-reload (`/admin/config/reload` or an admin-panel save) without a restart: enable, disable, retarget the endpoint, change headers, ratio, or capture — in-flight requests finish against the settings in effect at their start.

## Endpoint Examples

**Local OpenTelemetry Collector** (recommended starting point):

```yaml
opentelemetry:
  enabled: true
  endpoint: "http://localhost:4318/v1/traces"
```

**Jaeger** (native OTLP HTTP port):

```yaml
opentelemetry:
  enabled: true
  endpoint: "http://localhost:4318/v1/traces"   # Jaeger all-in-one OTLP/HTTP
```

**Langfuse** (OTLP ingestion):

```yaml
opentelemetry:
  enabled: true
  endpoint: "https://cloud.langfuse.com/api/public/otel/v1/traces"
  headers:
    Authorization: "Basic ${LANGFUSE_BASIC_AUTH}"   # base64(pk:sk)
```

Any OTLP/HTTP JSON + gzip endpoint works the same way — point `endpoint` at your backend's `/v1/traces` path and add its auth header.

## Trace Structure

- **One trace per request.** The root span covers the gateway hop (HTTP method, route, status code, `obey.request_id`, model group, virtual-key id when present) with request-aggregated `gen_ai.usage.*` and `gen_ai.usage.cost` attributes.
- **One child span per provider attempt** (kind `CLIENT`, named `{operation} {model}`, e.g. `chat gpt-4`): provider, request/response model, per-attempt token usage, outcome, `error.type` on failures, and attempt index. Failover chains render as sibling attempts under one root.
- **Abandoned requests still emit** a root span with the terminal outcome (`client_disconnect`, `gateway_deadline`).
- **Trace-id continuity:** an inbound W3C `traceparent` header joins the gateway hop into your application's trace. Otherwise the gateway reuses its request trace id when it is a UUID (correlating the SQLite log, `X-Trace-Id` response header, and active-requests view); non-UUID request ids get a fresh 128-bit trace id.

## Sampling

`sampling_ratio` is a **deterministic** head sampler: the decision is a fixed-threshold hash of the trace id, so the same trace id always maps to the same decision (no double-counting or split sessions across restarts).

- `1.0` (default): every request without an inbound sampling decision is sampled.
- `0.0`: no spans exported (exporter stays configured for hot-reload).
- **Parent-based override:** an inbound `traceparent` with `sampled=1` is always kept and `sampled=0` is always respected, regardless of the ratio — an application that already paid the sampling decision keeps its gateway hop in the trace.

For high-volume estates, prefer running a local OpenTelemetry Collector with **tail sampling** and keeping the gateway ratio at `1.0`: tail sampling decides on complete traces (root + all attempts), while head sampling at the gateway can drop a trace before its failover chain is assembled.

## Fail-Open Delivery

Export is asynchronous and bounded; it can never affect client traffic:

- Spans queue in a bounded queue (1024 spans); whole request batches are accepted or dropped whole — backends never render half a request.
- Delivery POSTs gzip OTLP JSON with your headers; HTTP 2xx is success, anything else retries 3 times (5 s/30 s/2 m backoff) and is then dropped and counted.
- An unreachable or misconfigured endpoint never blocks startup or requests; the exporter stays dormant and reports why.
- Graceful shutdown performs a bounded final flush (5 s budget); leftovers are dropped and counted.

## Monitoring Export Health

- Prometheus counters on `/metrics`: `otlp_spans_exported_total`, `otlp_spans_dropped_total`, `otlp_export_failures_total`.
- Admin panel → **Tracing** page → status card: enabled state, endpoint host, counters, last error class and timestamp, health. The same data is available at `GET /admin/telemetry/status`.
- Drop/failure warnings are rate-limited (every 30 s) and never contain span content or header values.

## Privacy: Content Capture

By default **no message content is ever captured** — spans contain metadata, token counts, and costs only. Enabling `capture_content: true` attaches prompt/completion content to attempt spans, size-bounded by `logging.max_body_size_bytes` with an explicit truncation marker. Authorization material (API keys, tokens, auth headers) is never captured in any mode. Header values configured for the exporter are masked (`[configured]`) in admin responses, and mask-preserved saves keep the stored secret.

## Admin Panel Parity

Every YAML knob has a matching control in the admin panel's **Tracing** page: enable toggle, endpoint, masked headers editor, sampling ratio, capture-content toggle (with privacy warning), service name, plus the live status card. Saves run through the same validation as YAML and apply via hot-reload.
