# Web Search Injection

The gateway can inject a server-side `web_search` tool into chat requests. The client never sees the tool: the model calls it, the gateway executes the search against a configured backend, and results flow back as tool output inside the same client request.

---

## How It Works

1. A single `web_search` tool is injected into the request sent to the routed provider.
2. The model decides when to search. It can search iteratively, up to `max_iterations` times per request.
3. Each call is executed server-side against the configured backend.
4. Results are normalized to title/url/snippet and returned as the tool result.
5. With `output_to_chat` (default on), results are also appended to the assistant message content, so they survive context and tool compression.

Tool arguments:

| Argument | Constraint |
|----------|------------|
| `q` | Required, up to 2000 characters |
| `domains` | Up to 10 domains |
| `recency` | Days, 1-365 |
| `max_results` | 1-10, default 5 |

---

## Configuration

```yaml
web_search:
 backend: "tavily" # tavily | brave | exa | searxng | codex
 api_key_env: "TAVILY_API_KEY" # env var name, resolved at runtime; literal fallback
 # enabled: true # implicit when a backend is set; explicit false disables
 # base_url: "https://api.tavily.com/search" # optional override on REST backends
 # timeout_seconds: 15 # 1..120 per search call
 # max_iterations: 5 # 1..20 searches per request
 # output_to_chat: true # append results to assistant content
```

- Setting `backend` enables the feature. `enabled: false` disables it even when a backend is configured.
- `api_key_env` is required for `tavily`, `brave`, and `exa`. It holds the env var name, not the key; a literal value is accepted as fallback.
- `base_url` is required for `searxng` and optional on `tavily`/`brave`/`exa` (useful for proxies and relays).
- `codex` needs no key or URL; it reuses the active OpenAI OAuth session.

---

## Backend Matrix

Not every backend supports every tool argument. Unsupported values are dropped silently.

| Backend | domains | recency | max_results | Notes |
|---------|---------|---------|-------------|-------|
| tavily | include_domains | dropped | yes | API key required |
| brave | dropped | freshness buckets (pd/pw/pm/py) | count | API key required |
| exa | includeDomains | startPublishedDate | numResults | API key required |
| searxng | dropped | time_range buckets (day/week/month/year) | sliced | `base_url` required; JSON format must be enabled on the instance |
| codex | yes | yes | sliced locally | requires an active OAuth session |

### SearXNG Setup

SearXNG is self-hosted and needs no API key. Enable the JSON format in the instance's `settings.yml` and point `base_url` at the search endpoint:

```yaml
web_search:
 backend: "searxng"
 base_url: "http://localhost:8888/search"
```

Aggregating DuckDuckGo (which has no official API) through a SearXNG instance is a legitimate use of SearXNG's own engine configuration.

### Codex

```yaml
web_search:
 backend: "codex"
```

Zero additional configuration. The search runs through the ChatGPT Codex search endpoint using the OAuth token; see [OAuth & Codex](OAuth-and-Codex).

---

## Precedence

- `web_search` enabled: the generic path is used only. The older `codex_search`/`codex_web` injection is suppressed.
- Only `codex_search` configured: legacy behavior is unchanged.
- Neither configured: nothing is injected.

---

## Failure Behavior

- Backend error: the tool returns an honest "no search ran this turn" result and the request completes normally.
- Backend not ready (missing key, no OAuth session): nothing is injected, a warning is logged, and the request proceeds without search.
- Readiness is visible in the admin panel's Web Search tab and via `GET /admin/web-search/status`.

---

## Observability

- Prometheus: `obey_web_search_tool_executions_total{backend}` and `obey_web_search_latency_ms{backend}`.
- Request log attempt rows with classes `web_search_success`, `web_search_error`, `web_search_validation_error`, and `web_search_budget_exhausted`, each carrying a truncated query.

---

## Next Steps

- [Configuration](Configuration) - full config reference
- [OAuth & Codex](OAuth-and-Codex) - OAuth sessions used by the codex backend
- [Admin Panel & Dashboard](Admin-Panel-and-Dashboard) - Web Search tab and status endpoint
