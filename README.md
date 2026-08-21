<p align="center">
  <img src="Assets/logo.jpg" alt="OBEY API Gateway" width="200" />
</p>

<h1 align="center">OBEY API Gateway</h1>

<p align="center">
  OpenAI-compatible AI gateway with intelligent routing, automatic failover, and multi-provider support.<br/>
  Single Rust binary. No runtime dependencies. Just download and run.
</p>

<p align="center">
  <a href="https://github.com/fdanobey/OBEY-api-gateway/releases/latest"><img src="https://img.shields.io/github/v/release/fdanobey/OBEY-api-gateway?style=flat-square" alt="Release" /></a>
  <a href="https://github.com/fdanobey/OBEY-api-gateway/actions"><img src="https://img.shields.io/github/actions/workflow/status/fdanobey/OBEY-api-gateway/release.yml?style=flat-square&label=build" alt="Build" /></a>
  <a href="https://opensource.org/licenses/MIT"><img src="https://img.shields.io/badge/license-MIT-blue?style=flat-square" alt="License" /></a>
</p>

<p align="center">
  <strong><a href="https://github.com/fdanobey/OBEY-api-gateway/releases/latest">Download Latest Release</a></strong> · Windows installer and portable zip
</p>

<p align="center">
  <a href="https://github.com/fdanobey/OBEY-api-gateway/wiki">📖 Documentation Wiki</a>
</p>

<p align="center">
  <a href="https://railway.com/deploy?template=https%3A%2F%2Fgithub.com%2Ffdanobey%2FOBEY-api-gateway"><img src="https://railway.com/button.svg" alt="Deploy on Railway" /></a>
</p>

---

## Why?

AI providers go down. Rate limits hit. Models get deprecated. When that happens, you're stuck manually switching providers, changing model names, and hoping the next one works.

OBEY API Gateway sits between your application and your AI providers. Point your existing OpenAI SDK at it instead of `api.openai.com`, and you get automatic failover, circuit breakers, and multi-provider routing without changing your application code.

## Key Features

- **Drop-in OpenAI replacement** — full `/v1/*` API compatibility (chat, completions, embeddings, images, audio, assistants)
- **Multi-provider routing** — OpenAI, Ollama, AWS Bedrock, Groq, Together AI, NVIDIA NIM, vLLM, LM Studio
- **Automatic failover** — circuit breakers + retry with exponential backoff across providers
- **Smart rate-limit failover** — instantly skips providers that return 429 (or rate-limit-shaped 200 envelopes), honors `Retry-After` / `X-RateLimit-Reset` / Anthropic ISO reset headers, and supports weekly-quota providers like Nano-GPT through per-provider cooldown overrides
- **Priority & cost-aware routing** — configure model groups with priority, cost, and latency-based selection
- **Context window management** — automatic truncation when requests exceed model limits
- **Streaming reliability** — true SSE pass-through for capable providers with early synthetic events (sub-500ms TTFB), configurable keep-alive, graceful in-stream error frames, mid-stream failover, and inter-chunk/total timeouts (see [Streaming Reliability](#streaming-reliability))
- **Response caching** — built-in two-tier cache: in-memory exact-match (default-on, no setup) plus optional semantic Qdrant tier; works for both streaming and non-streaming requests, including tool-using clients
- **OpenAI OAuth login** — browser-based sign-in with your ChatGPT Plus/Pro subscription (PKCE flow, automatic token refresh)
- **Codex backend translation** — transparently routes OAuth-authenticated requests through the ChatGPT Codex backend, translating Chat Completions ↔ Responses API on the fly
- **Codex web search** — automatic web-search tool injection available to every model group while a valid OpenAI OAuth token is active; the routed provider serves the completion while search calls execute server-side against the Codex search API (configurable timeout, max iterations, base URL, and chat output so results survive context compression)
- **Guardrail pipelines** — configurable pre-call and post-call policy enforcement with PII redaction/re-injection, regex scanning, Presidio NLP, OpenAI Moderation, Lakera, semantic prompt guard, and custom HTTP providers; includes refusal detection with automatic failover (see [Guardrail Pipelines](#guardrail-pipelines))
- **Agent loop detection** — multi-signal confidence scorer detects repetitive agent behavior (tool-call repetition, content similarity, error cycling, response stagnation, token/cost velocity, context growth) and escalates through Warn → Throttle → Inject → Hard-Stop enforcement levels; per-virtual-key overrides, session admin API, and Prometheus histograms (see [Agent Loop Detection](#agent-loop-detection))
- **Virtual key management** — issue per-caller API keys (`vk_…`) with independent USD/token budgets, rate limits, model-access restrictions, and expiry; authenticate callers without sharing real provider keys (see [Virtual Key Management](#virtual-key-management))
- **Encrypted API key storage** — provider keys encrypted at rest with a machine-local master key
- **Assistants API** — full local OpenAI-compatible Assistants implementation (assistants, threads, messages, runs, run steps, files) backed by SQLite; multi-tenant with per-virtual-key isolation, resource quotas, and run execution through gateway routing (see [Assistants API](#assistants-api))
- **Active request tracking** — live in-flight request registry surfaces per-request phase (primary / retry / failover / cascade), target provider, elapsed time, and virtual key to the dashboard in real time
- **Admin panel & dashboard** — embedded web UIs for configuration, metrics, in-flight request view, and log viewing
- **Prometheus metrics** — `/metrics` endpoint for existing monitoring infrastructure
- **Request logging** — SQLite-based structured logging with configurable retention
- **TLS support** — optional HTTPS with certificate configuration
- **Windows system tray** — double-click desktop app with splash screen and tray menu
- **Structured output validation** — JSON Schema validation of model responses with automatic retry on schema violations, per-model-group policies, and Prometheus metrics (see [Structured Output Validation](#structured-output-validation))
- **Persistent memory store** — cross-session memory extraction, namespace-scoped storage with decay scheduling, context-aware injection, sensitive data filtering, and Qdrant-backed retrieval
- **Token compression** — 9 multi-engine compression strategies (lite, standard, aggressive, ultra, RTK, stacked, tool_def, language_pack, perplexity) with hierarchical configuration, protection rules, cache-aware downgrades, and Prometheus observability (see [Token Compression](#token-compression))
- **Tool definition compression** — 12-stage pipeline for reducing token waste from large `tools` arrays: schema minification, description truncation, deduplication with `$ref`, frequency-based pruning, progressive disclosure with namespace grouping, semantic retrieval (TF-IDF + embeddings), canonical text rewriting, cache-aware placement, and adaptive feedback loop with auto-tuning; provider-aware, per-model-group overrides, zero overhead when disabled (see [Tool Definition Compression](#tool-definition-compression))
- **Hot config reload** — change settings through the admin UI without restarting
- **Dynamic request body limit** — configurable `max_request_size_mb` (default 10 MB) enforced per-request; adjustable via admin UI or hot-reload without restart, rejects oversized payloads with HTTP 413 before forwarding
- **Smart timeouts** — split TTFB / total timeouts with auto-detection of thinking models (o1, o3, DeepSeek-R1, Claude)
- **Smart model routing** — complexity-aware tier selection (Fast / Balanced / Powerful) with heuristic, ML (ONNX), LLM, or composite classifiers; cascade escalation, online optimization, A/B testing, budget limits, semantic routing cache, and per-model-group overrides (see [Smart Model Routing](#smart-model-routing))

## Quick Start

### Option 1: Download (Windows)

Grab the [latest release](https://github.com/fdanobey/OBEY-api-gateway/releases/latest) — either the installer (`.exe`) or portable zip. Double-click to run. The gateway starts on `http://localhost:8080` and opens the dashboard automatically on first launch.

### Option 2: Deploy to Railway

Click the button above to deploy directly from this repo. Railway picks up the included [`Dockerfile`](Dockerfile) and [`railway.toml`](railway.toml) automatically. Set your provider API keys (`OPENAI_API_KEY`, etc.) as environment variables in the Railway dashboard and you're live in under a minute.

> **Persist your keys on Railway:** attach a Railway Volume mounted at `/data` (the image's `AI_GATEWAY_DATA_DIR`). Railway's container filesystem is ephemeral, so without a volume the encryption master key is regenerated on every redeploy and previously saved `api_key_encrypted` values can no longer be decrypted. Alternatively, supply keys via plain environment variables (`OPENAI_API_KEY`, etc.), which never touch the encrypted store.

### Option 3: Docker

```bash
# Build the image
docker build -t obey-api-gateway .

# Run with config and env vars
docker run -d \
  -p 8080:8080 \
  -e OPENAI_API_KEY=sk-... \
  -v $(pwd)/config.yaml:/app/config.yaml \
  -v ai-gateway-data:/data \
  -v ai-gateway-models:/app/models \
  obey-api-gateway --config /app/config.yaml
```

> **Persist keys and ONNX assets:** `/data` stores the encryption master key; `/app/models` stores the optional Perplexity model/runtime downloaded from Admin → Compression. Mount both named volumes as shown so keys and the roughly 350 MB ONNX bundle survive container replacement. The gateway process must be able to write `/app/models`; a read-only bind mount produces an actionable install error.

#### Updating (Docker)

To update to the latest version:

```bash
# Pull latest source and rebuild
git pull origin master
docker build -t obey-api-gateway .

# Stop and remove the old container (data volume is preserved)
docker stop obey-api-gateway && docker rm obey-api-gateway

# Start with the new image
docker run -d --name obey-api-gateway \
  -p 8080:8080 \
  -e OPENAI_API_KEY=sk-... \
  -v $(pwd)/config.yaml:/app/config.yaml \
  -v ai-gateway-data:/data \
  -v ai-gateway-models:/app/models \
  obey-api-gateway --config /app/config.yaml
```

If you're using Docker Compose:

```yaml
# docker-compose.yml
services:
  obey-api-gateway:
    build: .
    ports:
      - "8080:8080"
    volumes:
      - ./config.yaml:/app/config.yaml
      - ai-gateway-data:/data
      - ai-gateway-models:/app/models
    environment:
      - OPENAI_API_KEY=sk-...

volumes:
  ai-gateway-data:
  ai-gateway-models:
```

```bash
# Update with Compose
git pull origin master
docker compose up -d --build
```

> **Note:** Encrypted keys persist in `ai-gateway-data`; downloaded ONNX assets persist in `ai-gateway-models`. Never remove either volume unless you intend to reset that data.

### Option 4: Build from Source

```bash
# Clone
git clone https://github.com/fdanobey/OBEY-api-gateway.git
cd OBEY-api-gateway

# Build (headless)
cargo build --release -p ai-gateway

# Build with Windows tray support
cargo build --release -p ai-gateway --features tray

# Run
./target/release/ai-gateway --config ./config.yaml
```

### Point Your App at the Gateway

```bash
# Any OpenAI-compatible SDK or tool
export OPENAI_API_BASE=http://localhost:8080/v1
```

```python
# Python example
from openai import OpenAI
client = OpenAI(base_url="http://localhost:8080/v1", api_key="unused")
response = client.chat.completions.create(
    model="gpt-4-group",  # Use your model group name
    messages=[{"role": "user", "content": "Hello!"}]
)
```

## Configuration

Config file is resolved in this order:

1. `--config` CLI flag
2. `CONFIG_PATH` environment variable
3. `./config.yaml` in working directory

If no config exists on first run, a default is created automatically. See [`config.example.yaml`](crates/ai-gateway/config.example.yaml) for the full reference.

### Minimal Example

```yaml
server:
  host: "0.0.0.0"
  port: 8080
  request_timeout_seconds: 30
  max_request_size_mb: 10             # Dynamic body limit (hot-reloadable)

providers:
  - name: "openai"
    type: "openai"
    base_url: "https://api.openai.com/v1"
    api_key_env: "OPENAI_API_KEY"       # Env var name, not the key itself
    timeout_seconds: 30                 # Legacy: used as total_timeout if split fields omitted
    # ttfb_timeout_seconds: 30          # Time-to-first-byte (default: 30s, thinking models: 120s)
    # total_timeout_seconds: 300        # Total round-trip  (default: 300s, thinking models: 600s)

  - name: "ollama"
    type: "ollama"
    base_url: "http://localhost:11434"
    timeout_seconds: 120

model_groups:
  - name: "gpt-4-group"
    models:
      - provider: "openai"
        model: "gpt-4"
        priority: 1                     # Lower = higher priority
      - provider: "ollama"
        model: "llama3"
        priority: 2                     # Fallback
```

### Provider Types

| Type | Provider | Notes |
|------|----------|-------|
| `openai` | OpenAI, Nano-GPT, any OpenAI-compatible API | Generic OpenAI protocol |
| `ollama` | Ollama | Local models, no API key needed |
| `bedrock` | AWS Bedrock | API key or AWS SDK auth ([details below](#bedrock-authentication)) |
| `groq` | Groq | |
| `together` | Together AI | |
| `nvidia_nim` | NVIDIA NIM | |
| `vllm` | vLLM | Self-hosted inference |
| `lmstudio` | LM Studio | Local models |

### API Key Management

Provider keys can be configured four ways:

1. **Environment variable reference** — set `api_key_env: "OPENAI_API_KEY"` and export the env var
2. **Admin UI** — enter keys through the web interface; they're encrypted automatically
3. **Encrypted in config** — stored as `api_key_encrypted: "enc-v1:<nonce>:<ciphertext>"`
4. **OAuth login** — for OpenAI providers, authenticate via browser sign-in (see below)

The master encryption key is stored outside the config file in your platform's secure directory (e.g. `%APPDATA%\ai-gateway\master.key` on Windows).

### OpenAI OAuth Login

Instead of manually creating and managing OpenAI API keys, you can authenticate with your ChatGPT Plus/Pro subscription via browser-based OAuth:

```yaml
providers:
  - name: "openai-oauth"
    type: "openai"
    base_url: "https://api.openai.com/v1"
    auth_method: "oauth"              # Use OAuth instead of api_key_env
```

Trigger the login flow via the admin API:

```bash
# Initiate browser-based login
curl -X POST http://localhost:8080/admin/oauth/openai/login

# Check session status
curl http://localhost:8080/admin/oauth/openai/status

# Logout (clear stored tokens)
curl -X POST http://localhost:8080/admin/oauth/openai/logout
```

The gateway handles the full token lifecycle automatically:
- Opens your default browser to OpenAI's authorization page
- Receives the callback on a local loopback server
- Exchanges the authorization code for tokens (PKCE + S256)
- Encrypts and persists tokens to disk (survives restarts)
- Refreshes the access token in the background before expiry
- Falls back to the next provider if the OAuth session expires

**Security:** Tokens are encrypted at rest with AES-256-GCM, the callback server binds exclusively to `127.0.0.1`, and token values are never logged at any level.

### Bedrock Authentication

AWS Bedrock supports two modes:

```yaml
# Mode 1: API key (Bedrock Mantle endpoint)
- name: "bedrock-api-key"
  type: "bedrock"
  region: "us-east-1"
  api_key_env: "AWS_BEARER_TOKEN_BEDROCK"

# Mode 2: AWS SDK credentials (env vars, shared credentials, IAM role)
- name: "bedrock-sdk"
  type: "bedrock"
  region: "us-east-1"
```

Bedrock-specific options:

| Field | Default | Description |
|-------|---------|-------------|
| `cross_region_inference` | `false` | Prefix model IDs with region group (e.g. `us.`) for cross-region routing |
| `global_inference_profile` | `false` | Let AWS auto-select the optimal region |
| `prompt_caching` | `false` | Enable prompt caching (Claude 3.5+ models) |
| `custom_vpc_endpoint` | `false` | Use `base_url` as-is instead of auto-generating the Mantle endpoint |
| `reasoning` | `true` | Enable extended thinking for supported models |

> **Claude Fable 5 prerequisite:** Before you can invoke Claude Fable 5 through Bedrock, you must opt into data sharing via the AWS Data Retention API by enabling `provider_data_share` on your account. There is no console UI for this at launch — it must be done programmatically (e.g. via the AWS CLI or SDK). This is a one-time, account-level setting; once enabled, the model works through the gateway like any other Bedrock model with no router-side changes needed.

### Response Caching

The gateway runs a two-tier response cache for chat completions. Both tiers cache the same key space, so a single entry serves streaming and non-streaming callers.

| Tier | Backend | Default | Catches |
|------|---------|---------|---------|
| 1 — Exact | In-memory `DashMap`, SHA-256 keyed | **Enabled** | Byte-identical retries, agent loops, dedup |
| 2 — Semantic | Qdrant + embedding provider | Disabled | Paraphrased / near-identical prompts |

**Eligibility (both tiers):** `temperature ≤ temperature_threshold` (default `0.15`) **and** `n == 1`. Higher temperatures imply non-determinism and are skipped to avoid replaying randomized output.

**Key fields:** `model`, full `messages`, `tools`, `tool_choice`, `response_format`, `top_p`, `frequency_penalty`, `presence_penalty`, `stop`, `seed`, `n`, `max_tokens`. The `stream` flag and per-request transport metadata (`user`, request-id, trace-id) are intentionally excluded.

**Write-side filter:** responses with `tool_calls`, `finish_reason: length`, or `finish_reason: content_filter` are never stored, regardless of eligibility.

```yaml
# Tier 1 — exact-match in-memory cache (defaults shown; section optional)
exact_cache:
  enabled: true
  max_entries: 5000           # oldest-first eviction above this
  ttl_seconds: 3600
  temperature_threshold: 0.15

# Tier 2 — semantic cache (optional, requires Qdrant)
# semantic_cache:
#   enabled: true
#   qdrant_url: "http://localhost:6334"     # gRPC port, not 6333
#   collection_name: "ai_gateway_cache"
#   similarity_threshold: 0.95
#   embedding_provider: "openai"            # must match a provider name above
#   embedding_model: "text-embedding-3-small"
#   ttl_seconds: 3600
#   max_cache_size: 10000
```

The dashboard's **Cache Hit Rate** card stays at `N/A` until the first eligible request is observed, then switches to a percentage. To force traffic through the cache, send the same request twice with `temperature: 0` (or omit it).

### Streaming Reliability

When a client requests `stream: true`, the gateway improves perceived reliability for slow/thinking models and flaky upstreams:

- **Early synthetic event** — emits a `role: assistant` SSE chunk within ~500ms so clients don't idle-timeout while the model "thinks" (skipped on cache hits).
- **Configurable keep-alive** — periodic SSE comments keep client connections from timing out during long generations.
- **True streaming pass-through** — for OpenAI-compatible providers, upstream SSE chunks are relayed in real time; providers needing response transformation (Bedrock, XML-tool rewrite, Kimi/Nano-GPT token sanitization) and Codex OAuth providers automatically fall back to buffer-and-replay.
- **Graceful error frames** — TTFB / total / inter-chunk timeouts and mid-stream failures are surfaced as `{"error":{...}}` SSE events followed by `[DONE]`, never a silent disconnect.
- **Mid-stream failover** — if a provider fails *before* any content reaches the client, the gateway transparently retries the next provider (no duplicate role event); after content has been sent it emits an error and closes.
- **Truncation retry** — a `finish_reason: "length"` response that stops well short of the requested `max_tokens` is treated as a truncation and retried on the next provider; if every provider truncates, the longest partial is returned.

All fields are optional with safe defaults, so existing configs keep working unchanged:

```yaml
streaming:
  emit_early_event: true            # synthetic role:assistant chunk before upstream responds
  keepalive_interval_seconds: 5     # 0–60; 0 disables (axum default)
  passthrough_enabled: true         # true SSE relay for capable providers
  chunk_timeout_seconds: 60         # max gap between SSE chunks (min 5)
  retry_on_truncation: true         # failover on suspicious finish_reason=length
```

### Timeout Configuration

Timeouts are split into two phases for clarity:

| Field | Default (standard) | Default (thinking models) | Description |
|-------|-------------------|--------------------------|-------------|
| `ttfb_timeout_seconds` | 30s | 120s | Time-to-first-byte — how long to wait for the provider to start responding |
| `total_timeout_seconds` | 300s | 600s | Total round-trip — ceiling for the entire request including body transfer |
| `timeout_seconds` | 30s | 30s | Legacy field — used as `total_timeout_seconds` when split fields are omitted |

Thinking models (o1, o3, DeepSeek-R1, QwQ, Claude 3.5 Sonnet v2+, Claude Opus/4+) are detected automatically and get higher defaults. You can override per-provider:

```yaml
providers:
  - name: "openai"
    type: "openai"
    base_url: "https://api.openai.com/v1"
    api_key_env: "OPENAI_API_KEY"
    ttfb_timeout_seconds: 60        # Wait up to 60s for first byte
    total_timeout_seconds: 600      # Allow up to 10 min total
```

When a timeout fires, the error response tells the user exactly which timeout was hit and which config field to adjust.

### Environment Variables

| Variable | Purpose |
|----------|---------|
| `CONFIG_PATH` | Override config file location |
| `AI_GATEWAY_DATA_DIR` | Override the secrets/master-key directory (recommended for Docker; mount a volume so encrypted keys persist) |
| `OPENAI_API_KEY` | Provider API key (name matches `api_key_env` in config) |
| `ADMIN_USERNAME` | Admin panel username |
| `ADMIN_PASSWORD` | Admin panel password |
| `RUST_LOG` | Tracing filter (`info`, `debug`, `ai_gateway=trace`) |

## Token Compression

Token compression is opt-in and defaults to disabled. When enabled, the gateway can transparently compress request payloads before forwarding to providers, reducing token consumption and costs—especially for applications with large conversation histories or repetitive content.

The system ships with 9 compression engines, each tuned for different content types and use cases. Named levels (lite, standard, aggressive, ultra, rtk, stacked) resolve to ordered engine chains; tool_def, language_pack, and perplexity are standalone engines that can be composed via custom pipelines:

| Strategy | Use Case | Trade-off |
|----------|----------|-----------|
| lite | Minimal compression; best for single-turn requests | Lowest compression ratio; preserves readability |
| standard | Balanced compression; default for multi-turn conversations | Good compression + retention of meaning |
| aggressive | High compression; removes structure and whitespace | Medium compression ratio; may lose formatting context |
| ultra | Maximum compression; removes all non-essential tokens | High compression; risky for complex logic/code |
| rtk | Round-trip-knowledge; preserves semantic meaning for cached responses | High compression with semantic safety |
| stacked | Layered compression; applies RTK + standard sequentially | Highest compression; slowest |
| tool_def | Optimized for tool/function definitions | Compresses JSON schemas and callable signatures |
| language_pack | Language-aware compression; detects dominant language | Respects linguistic boundaries |
| perplexity | Perplexity-model optimized; high compression for long-context requests | Tailored for perplexity models |

### Configuration

```yaml
compression:
  enabled: false                         # Enable compression globally (default: false)
  default_level: lite                    # Default compression level (default: lite)
  auto_threshold_tokens: 0              # Auto-trigger threshold; 0 = disabled (default: 0)
  caveman_output: false                  # Collapse output to extreme brevity (default: false)
  compress_tool_definitions: false       # Compress tool/function definitions (default: false)
  language: en                           # Language for language_pack engine (default: en)
  language_packs_dir: ./language_packs   # Directory for language pack files
  time_budget_ms:                        # Per-level time budgets in milliseconds
    lite: 500
    standard: 500
    aggressive: 2000
    ultra: 2000
    rtk: 2000
    stacked: 2000
  protection_rules:                      # Content patterns never compressed
    - code_blocks
    - urls
    - file_paths
    - json_structures
    - identifiers
    - math_expressions
    - tool_definitions
    - structured_tool_output
  precompressed_contexts: []            # Pre-compressed file mappings
  rtk:
    grouping_strategy: balanced         # aggressive | balanced | conservative
  perplexity:
    enabled: false                       # Requires an ONNX perplexity scorer model
    redundancy_threshold: 0.5
    compression_ratio_target: 5
    model_path: ./models/perplexity_scorer.onnx
  custom_pipelines: {}                  # Named custom engine chains
```

Provider and model-group compression blocks override these defaults field-by-field. Setting level: none on a provider or model group explicitly disables compression for that scope.

### Custom Pipelines

Administrators can define named custom pipelines that chain any combination of the 9 engines in a specified order:

```yaml
compression:
  custom_pipelines:
    terminal_then_prose:
      engines: [rtk, standard, lite]
```

Callers select a custom pipeline by name via the x-compression-pipeline request header.

### Cache-Aware Downgrade

When a provider supports prompt caching (e.g., Anthropic Claude) and a request contains cache_control markers, the gateway automatically preserves the cached prefix byte-for-byte. Aggressive, ultra, RTK, and stacked levels are downgraded to none for the protected prefix while still compressing the suffix, ensuring cache hits are not invalidated.

### Observability

Compression execution is fully observable via Prometheus metrics:

| Metric | Type | Description |
|--------|------|-------------|
| obey_compression_tokens_saved_total | counter | Total tokens saved by compression operations (by level and provider) |
| obey_compression_ratio | histogram | Compressed token ratio (compressed / original; 1.0 when original is zero) |
| obey_compression_duration_seconds | histogram | Compression operation duration in seconds |

For detailed compression implementation, configuration examples, and performance tuning, see the compression source directory.

## Tool Definition Compression

When MCP servers or agentic frameworks provide 50–200+ tool definitions per request, the `tools` array alone can consume 10,000–50,000 tokens. Tool Definition Compression is a dedicated Tower middleware that intercepts the `tools` field and applies a 12-stage pipeline to reduce its token footprint without degrading model tool-calling accuracy.

The middleware is **opt-in** (disabled by default, zero per-request overhead when off) and sits after guardrails but before provider dispatch.

### Compression Stages (fixed order)

| # | Stage | Level | Description |
|---|-------|-------|-------------|
| 1 | Schema Minifier | All | Remove `title`, `additionalProperties: false`, empty descriptions; collapse single-element `anyOf`/`oneOf`; nullable union simplification |
| 2 | Description Truncator | Low+ | Example removal (Low), first-sentence extraction (Medium), full removal (High), large enum replacement (Max) |
| 3 | Schema Deduplicator | Medium+ | Replace duplicate parameter schemas across tools with `$ref` / `$defs` references |
| 4 | Tool Pruner | Max | Remove tools with zero calls after `min_requests` threshold; respects `always_include` globs |
| 5 | Progressive Disclosure | High+ | Replace full schemas with minimal name+description listing; synthetic `get_tool_schema` tool for on-demand retrieval |
| 6 | Namespace Grouper | High+ | Cluster tools by prefix, emit namespace summaries with `get_tools_in_namespace` drill-down |
| 7 | Semantic Retriever | Config | TF-IDF/BM25 (or embedding) hybrid scoring; keep top-K tools relevant to user message; defer the rest |
| 8 | Description Compressor | Config | TF-IDF token-importance scoring; remove parameter-redundant tokens from descriptions |
| 9 | Canonical Rewriter | Max | Convert JSON Schema to compact `tool: / desc: / params:` text format for supported models |
| 10 | Cache Placement | High+ | Reorder stable tools before new/modified tools for prefix cache hits |
| 11 | Feedback Loop | Always | Rolling-window error detection; auto-reduces compression level on quality regression |
| 12 | Auto-Tuner | Always | Model tier detection (glob patterns → Low/Medium/High); prompt cache skip when all hashes match |

### Configuration

```yaml
tool_compression:
  enabled: false                      # Zero overhead when disabled (default)
  level: medium                       # low | medium | high | max
  progressive_disclosure: false       # Enable two-tier listing with get_tool_schema
  cache_placement: true               # Reorder for prefix cache hits
  deduplication: true                 # $ref dedup for identical schemas

  pruning:
    enabled: false
    min_requests: 5                   # Session requests before pruning activates
    always_include: ["github_*"]      # Glob patterns never pruned

  minification:
    remove_titles: true
    collapse_single_unions: true
    remove_additional_properties: true
    remove_empty_descriptions: true

  description_truncation:
    tool_level: first_sentence        # none | first_sentence | remove
    parameter_level: remove           # none | remove
    remove_examples: true
    min_preserve_length: 20

  semantic_retrieval:
    enabled: false
    top_k: 20
    similarity_threshold: 0.3
    frequency_weight: 0.3

  canonical_rewriting:
    enabled: false
    allowed_models: ["gpt-4*", "claude-3*"]

  feedback_loop:
    enabled: true
    error_threshold: 0.10
    recovery_window: 50
    rolling_window: 100

  auto_tuning:
    enabled: true
    model_tiers:
      "gpt-4o*": 1                    # Tier 1 → Low compression
      "gpt-4*": 2                     # Tier 2 → Medium
      "gpt-3.5*": 3                   # Tier 3 → High

  namespace_grouping:
    enabled: false
    min_tools_for_grouping: 10

  precomputed_descriptions:
    enabled: false
    method: tfidf                      # tfidf | manual | model

  # Per-model-group overrides
  model_group_overrides:
    coding-group:
      level: high
      progressive_disclosure: true
```

### Request Headers

| Header | Description |
|--------|-------------|
| `X-Tool-Compression-Disable: true` | Bypass compression for this request |
| `X-Tool-Compression-Level: <level>` | Override compression level (none/low/medium/high/max) |

### Response Headers

| Header | Description |
|--------|-------------|
| `X-Tool-Compression-Level` | Effective compression level applied |
| `X-Tool-Compression-Ratio` | Reduction ratio (e.g., `0.65` = 65% reduction) |
| `X-Tool-Compression-Tokens-Saved` | Estimated tokens saved |

### Admin API

| Method | Path | Description |
|--------|------|-------------|
| `GET` | `/admin/tool-compression/feedback` | List feedback loop states per model group |
| `POST` | `/admin/tool-compression/feedback/:group/lock` | Lock a group's compression level |
| `POST` | `/admin/tool-compression/feedback/:group/unlock` | Unlock a group |
| `POST` | `/admin/tool-compression/feedback/:group/reset` | Reset feedback state |
| `POST` | `/admin/tool-compression/feedback/reset-all` | Reset all feedback state |
| `GET` | `/admin/tool-compression/descriptions` | List pre-computed compressed descriptions |
| `POST` | `/admin/tool-compression/descriptions/recompute` | Trigger description recomputation |

### Provider Awareness

The pipeline respects per-provider capabilities automatically:

| Provider | `$ref` support | Nullable shorthand | Prompt caching | Canonical format | Max tools |
|----------|:-:|:-:|:-:|:-:|:-:|
| OpenAI | ✓ | ✓ | ✓ | ✓ | 128 |
| Anthropic | ✗ | ✗ | ✓ | ✓ | 200 |
| Google | ✓ | ✓ | ✓ | ✗ | 128 |
| Groq | ✗ | ✓ | ✗ | ✗ | 64 |
| Bedrock | ✗ | ✗ | ✓ | ✗ | 200 |

Stages that require unsupported features (e.g., `$ref` deduplication on Anthropic) are automatically skipped.

## Virtual Key Management

Instead of every caller sharing your real provider keys, administrators can issue **virtual keys** (`vk_…`) that authenticate individual callers to the gateway. Each key carries its own budgets, rate limits, model-access rules, and expiry, all enforced at the proxy layer before requests reach upstream providers. This enables multi-tenant usage tracking, cost control, and access governance without exposing provider credentials.

Virtual keys are stored encrypted in a dedicated SQLite database (`keys.db`), separate from request logs.

### Enforcement Modes

Enforcement is opt-in and defaults to `disabled`, so existing deployments are unaffected:

```yaml
virtual_keys:
  enforcement: disabled       # disabled | optional | required
  database_path: "./keys.db"  # dedicated key/usage store
```

| Mode | Behavior |
|------|----------|
| `disabled` (default) | Virtual keys are ignored; requests route with provider keys directly |
| `optional` | Requests with a `vk_` bearer token are validated and tracked; requests without one pass through |
| `required` | Every proxied request must present a valid virtual key (else `401`) |

The enforcement pipeline runs in order: **authenticate → model access → budget → rate limit → forward**, then usage (spend + tokens) is recorded from the provider response.

### Per-Key Constraints

| Constraint | Description |
|------------|-------------|
| `budget_limit_usd` | Cumulative USD spend cap (`0.01`–`999,999,999.99`) → `429` when reached |
| `token_budget` | Cumulative token cap (input + output) → `429` when reached |
| `budget_window` | `daily` / `weekly` / `monthly` reset window (omit for a lifetime limit) |
| `requests_per_minute` | Per-key RPM token-bucket → `429` + `Retry-After` |
| `tokens_per_minute` | Per-key TPM rolling 60s window → `429` + `Retry-After` |
| `model_access` | Whitelist of model group names (omit to allow all) → `403` on denial |
| `expires_in` | `never`, `1_year`, `6_months`, `3_months`, `1_month`, `2_weeks`, `1_week`, `3_days`, `1_day` |

### Admin API

Virtual keys are managed through the admin API (protected by the existing admin auth) or the **Virtual Keys** tab in the admin panel:

```bash
# Create a key (returns the full vk_ value exactly once)
curl -X POST http://localhost:8080/admin/keys \
  -H 'Content-Type: application/json' \
  -d '{"name":"team-a","budget_limit_usd":50,"budget_window":"monthly","requests_per_minute":60}'

# List / inspect / update / revoke / delete
curl http://localhost:8080/admin/keys
curl http://localhost:8080/admin/keys/{id}
curl -X PATCH  http://localhost:8080/admin/keys/{id} -H 'Content-Type: application/json' -d '{"budget_limit_usd":100}'
curl -X POST   http://localhost:8080/admin/keys/{id}/revoke
curl -X DELETE http://localhost:8080/admin/keys/{id}

# Per-key usage aggregate over a time range
curl "http://localhost:8080/admin/keys/{id}/usage?start=2024-01-01T00:00:00Z&end=2024-01-31T23:59:59Z"
```

Callers then authenticate with the issued key:

```bash
curl http://localhost:8080/v1/chat/completions \
  -H "Authorization: Bearer vk_your_key_here" \
  -H "Content-Type: application/json" \
  -d '{"model":"gpt-4-group","messages":[{"role":"user","content":"Hello!"}]}'
```

The admin panel's **Virtual Keys** section provides a searchable/sortable key table (with budget and expiry warnings), create/edit forms, one-time key reveal, revoke/delete confirmations, and a 30-day per-key usage chart.

## Guardrail Pipelines

Guardrail Pipelines add a configurable policy-enforcement layer that intercepts requests before provider routing (pre-call) and responses before returning to the caller (post-call). Use them for PII/DLP filtering, content moderation, semantic prompt guarding, and sensitive data redaction with transparent re-injection.

Guardrails are opt-in per virtual key, model group, or route and execute through a pluggable provider interface.

### Guardrail Providers

| Type | Description |
|------|-------------|
| `regex` | Up to 256 named patterns with allow/deny modes; compiled at load time, per-pattern 10ms budget |
| `presidio` | Presidio-compatible NLP PII detection via HTTP; configurable entity types and confidence threshold |
| `openai_moderation` | OpenAI Moderation API integration |
| `lakera` | Lakera Guard prompt injection detection |
| `semantic` | Embedding-based similarity matching against allow/deny example collections in Qdrant |
| `custom_http` | POST content to any HTTP endpoint implementing the documented findings JSON schema |

### Policy Actions

| Action | Pre-Call | Post-Call | Behavior |
|--------|:--------:|:---------:|----------|
| `allow` | ✓ | ✓ | Pass through unmodified |
| `block` | ✓ | ✓ | Reject with HTTP 403 (pre-call stops forwarding; post-call discards response) |
| `mask` | ✓ | | Replace each character with `*`, preserving byte length |
| `redact` | ✓ | ✓ | Replace with placeholder tokens (pre-call) or `[REDACTED]` (post-call) |
| `replace_with_policy_message` | | ✓ | Replace assistant content with a configured message |

### PII Redaction & Re-Injection

When a pre-call stage uses the `redact` action, detected PII values are replaced with deterministic placeholder tokens (`<<PII_EMAIL_1>>`, `<<PII_SSN_2>>`, etc.) before the request reaches the LLM. A system instruction is automatically prepended telling the model to preserve placeholders verbatim. After the LLM responds, placeholders are transparently restored to original values before the response reaches the caller.

- Up to 256 distinct values per request receive re-injection entries
- Identical values reuse the same placeholder (deduplication)
- Configurable redaction-notice instruction text and insertion mode (`separate` or `merged`)
- The Re_Injection_Map is held only in memory for the request duration and discarded immediately after

### Refusal Detection & Failover

The gateway can detect model refusals (via phrase matching or tool-call omission) and optionally fail over to the next provider in the fallback ordering:

- **Phrase matching** — case-insensitive regex patterns against assistant-role content (ships with a default list, overridable per-pipeline)
- **Tool-omission signal** — fires when tools were provided but the model didn't call any
- **Bounded failover** — re-dispatches the already-redacted request to the next eligible target (skipping open circuit breakers), attempting each at most once
- **Toggle** — `failover_on_refusal` per-pipeline or per-binding, disabled by default

### Pipeline Ordering

When multiple pipelines apply (global default + virtual-key + model-group + route), stages are concatenated in a fixed order:

1. Global default pipeline stages
2. Virtual-key pipeline stages
3. Model-group pipeline stages
4. Route pipeline stages

Halting actions (`block`, `replace_with_policy_message`) short-circuit immediately; non-halting actions continue to the next stage.

### Streaming Support

For SSE responses with a post-call pipeline, the gateway buffers the streamed response (up to 10 MB), sends keep-alive comments during buffering, applies guardrail analysis on the assembled content, and re-chunks the result into SSE events matching the original chunk boundaries.

### Configuration Example

```yaml
guardrails:
  providers:
    - name: secret-scanner
      type: regex
      failure_policy: fail_close
      patterns:
        - { name: openai_key, regex: "sk-[A-Za-z0-9]{20,}", entity: API_KEY, mode: deny }

    - name: pii-detector
      type: presidio
      failure_policy: fail_open
      endpoint: "http://presidio:3000/analyze"
      language: en
      entities: [EMAIL_ADDRESS, US_SSN, CREDIT_CARD]
      confidence_threshold: 0.6

    - name: prompt-guard
      type: semantic
      failure_policy: fail_open
      allow_collection: "guardrail_allow"
      deny_collection: "guardrail_deny"
      allow_threshold: 0.90
      deny_threshold: 0.85

  pipelines:
    - name: standard
      failover_on_refusal: true
      stages:
        - { name: pii-redact, provider: pii-detector, phase: pre_call, action: redact }
        - { name: secret-block, provider: secret-scanner, phase: pre_call, action: block }
        - { name: injection-guard, provider: prompt-guard, phase: pre_call, action: block }
        - { name: out-redact, provider: secret-scanner, phase: post_call, action: redact }

  global_default_pipeline: standard

  bindings:
    virtual_keys:
      vk_team_a: standard
    model_groups:
      gpt-4-group: standard
    routes:
      "/v1/chat/completions": standard
```

### Failure Policies

Each provider must declare a `failure_policy`:

- **`fail_open`** — on timeout or error, skip the stage and continue the pipeline
- **`fail_close`** — on timeout or error, halt the pipeline and return HTTP 503

### Observability

Guardrail execution is fully observable:

- Counter: `obey_api_guardrail_stage_executions_total{pipeline, stage, provider, action}`
- Histogram: `obey_api_guardrail_stage_latency_ms{pipeline, stage, provider}` (buckets: 5–5000ms)
- Counter: `obey_api_guardrail_refusal_detected_total{pipeline, signal}`
- Counter: `obey_api_guardrail_refusal_failover_total{pipeline, outcome}`
- INFO logs for non-pass actions (never includes triggering content)
- WARN logs for provider errors
- Per-request guardrail summary in the request log entry

## Agent Loop Detection

AI coding agents and automation pipelines can get stuck in repetitive loops — retrying the same tool call, cycling through identical errors, or regenerating near-identical content without progress. The loop detection system monitors request patterns per-session and applies graduated enforcement to break these loops before they burn tokens and cost.

Loop detection is **opt-in** (disabled by default) and operates as Tower middleware on `/v1/chat/completions`.

### How It Works

Each caller session is tracked independently (resolved by virtual key ID, `x-session-id` header, or IP). On every request, seven signals are computed from the session's recent history and combined into a weighted confidence score (EMA-smoothed). When confidence exceeds thresholds for consecutive requests, the enforcement level escalates.

### Detection Signals

| Signal | What it measures |
|--------|-----------------|
| `content_similarity` | SimHash similarity between the current request and recent requests |
| `tool_call_repetition` | Consecutive identical tool-call fingerprints |
| `response_stagnation` | Responses with matching structure and token count |
| `token_velocity` | Tokens consumed per minute exceeding threshold |
| `error_cycling` | Repeated requests after provider errors with high content similarity |
| `context_growth` | Context token growth disproportionate to new information |
| `cost_velocity` | USD spend rate per minute exceeding threshold |

Weights must sum to 1.0 and are fully configurable.

### Enforcement Levels

| Level | Confidence | Consecutive | Behavior |
|-------|-----------|-------------|----------|
| **None** | — | — | Normal operation |
| **Warn** | ≥ 0.30 | 3 | `x-loop-warning` response header with confidence and dominant signal |
| **Throttle** | ≥ 0.50 | 5 | Artificial delay (default 2s) before forwarding |
| **Inject** | ≥ 0.70 | 7 | System prompt instruction appended telling the model to change strategy |
| **Hard-Stop** | ≥ 0.90 | 10 | Request rejected with HTTP 429 and `Retry-After: 60` |

Enforcement de-escalates automatically after 5 consecutive low-confidence requests.

### Injection Strategies

When the `inject` level is reached, the gateway appends a break instruction to the system prompt:

| Strategy | Behavior |
|----------|----------|
| `system_prompt_append` (default) | Appends a generic "loop detected — change approach" instruction |
| `context_aware` | Tailors the instruction based on the dominant signal (e.g., names the repeated tool, or tells the model to stop retrying a failing operation) |

A custom `break_instruction_template` (up to 2000 chars) can be configured globally or per virtual key.

### Configuration

```yaml
loop_detection:
  enabled: true
  session_timeout_minutes: 30       # Session expires after inactivity
  max_sessions: 10000               # LRU eviction above this
  history_depth: 5                  # Requests retained per session

  thresholds:
    warn_confidence: 0.30
    throttle_confidence: 0.50
    inject_confidence: 0.70
    hardstop_confidence: 0.90

  consecutive_counts:
    warn: 3
    throttle: 5
    inject: 7
    hardstop: 10

  weights:
    content_similarity: 0.25
    tool_call_repetition: 0.20
    response_stagnation: 0.15
    token_velocity: 0.10
    error_cycling: 0.15
    context_growth: 0.10
    cost_velocity: 0.05

  throttle_delay_seconds: 2
  injection_strategy: system_prompt_append   # or context_aware
  ema_alpha: 0.3                             # EMA smoothing factor
  eviction_interval_seconds: 60
  token_velocity_threshold: 10000.0          # tokens/min before signal fires
  cost_velocity_threshold: 0.5               # USD/min before signal fires
  # break_instruction_template: "Custom instruction text..."
```

### Per-Virtual-Key Overrides

Virtual keys can carry their own loop detection settings that merge with (and override) the global config:

```yaml
# In virtual key creation/update via admin API
{
  "name": "aggressive-agent",
  "loop_detection": {
    "thresholds": { "warn_confidence": 0.20, "throttle_confidence": 0.40, "inject_confidence": 0.60, "hardstop_confidence": 0.80 },
    "consecutive_counts": { "warn": 2, "throttle": 3, "inject": 5, "hardstop": 7 },
    "injection_strategy": "context_aware",
    "break_instruction_template": "You are stuck in a loop. Stop and ask the user for guidance."
  }
}
```

### Admin API

| Method | Path | Description |
|--------|------|-------------|
| `GET` | `/admin/loop-detection/sessions` | List active sessions (paginated, `?limit=50&offset=0`) |
| `GET` | `/admin/loop-detection/sessions/{id}` | Full session detail with signal history and escalation timeline |
| `POST` | `/admin/loop-detection/sessions/{id}/reset` | Reset a session's enforcement state |
| `GET` | `/admin/loop-detection/stats` | Aggregate stats: enforcement counts, signal distribution, top sessions, memory estimate |

### Observability

| Metric | Type | Description |
|--------|------|-------------|
| `obey_loop_confidence_score` | histogram | Per-virtual-key confidence distribution (buckets: 0.1–1.0) |
| `obey_loop_enforcement_total` | counter | Enforcement transitions by level and virtual key |
| `obey_loop_sessions_active` | gauge | Current active session count |
| `obey_loop_sessions_evicted_total` | counter | Total sessions evicted by LRU |

All enforcement actions are logged at INFO level with session ID, virtual key, confidence, dominant signal, and all signal values. Hard-stops log at ERROR with full session state for forensic analysis.

### Hard-Stop Response

When a session reaches the hard-stop level, the gateway returns:

```json
{
  "error": {
    "reason": "loop_detected",
    "session_id": "...",
    "confidence": 0.95,
    "dominant_signal": "tool_call_repetition",
    "enforcement_level": "hard_stop"
  }
}
```

HTTP status: `429 Too Many Requests` with `Retry-After: 60`.

## Structured Output Validation

When the caller specifies `response_format: { type: "json_schema", json_schema: {...} }`, the gateway validates model responses against the provided schema before returning them. Invalid responses are automatically retried (up to a configurable limit) by re-prompting the model with the validation errors.

This is opt-in and defaults to disabled:

```yaml
structured_output:
  enabled: true
  max_retries: 1                      # Corrective retry attempts (0–5)
  retry_temperature: 0                # Temperature for corrective retries (0.0–2.0)
  passthrough_providers: [openai]     # Providers with native constrained decoding (skip validation)
```

Per-model-group overrides are supported via `model_groups[].structured_output`. Prometheus metrics track validation attempts, failures, retry counts, and latency.

## Smart Model Routing

Smart Model Routing automatically selects the best model within a group based on request complexity. Instead of always routing to the highest-priority provider, the system classifies each request and dispatches it to the cheapest tier that can handle it well.

### Model Tiers

Each model in a group can be assigned a capability tier:

```yaml
model_groups:
  - name: "smart-group"
    models:
      - provider: "openai"
        model: "gpt-4.1"
        tier: powerful
        context_window: 128000
        specializations: [code_generation, factual_qa]
      - provider: "openai"
        model: "gpt-4o"
        tier: balanced
        context_window: 128000
      - provider: "groq"
        model: "llama3-8b-8192"
        tier: fast
        context_window: 8192
```

### Classifier Modes

| Mode | Description |
|------|-------------|
| `heuristic` (default) | Weighted signal analysis (message count, tokens, code blocks, tool calls, math, reasoning keywords) |
| `ml` | ONNX model inference (requires `ml-router` build feature) |
| `llm` | Delegates classification to a configured LLM |
| `composite` | Weighted blend of heuristic + ML |

### Configuration

```yaml
smart_routing:
  enabled: true
  classifier: heuristic
  cost_quality_threshold: 0.5       # 0 = favor cost, 1 = favor quality
  tier_boundaries:
    fast_max: 0.33                  # Complexity 0–0.33 → Fast
    balanced_max: 0.66              # Complexity 0.33–0.66 → Balanced
  cascade:
    enabled: true
    max_escalations: 2
  reserved_output_tokens: 1024
  provider_overhead_tokens: 64
  context_safety_margin_tokens: 256
```

### Key Capabilities

- **Context capacity filtering** — excludes models that can't fit the request (returns HTTP 413 when no model can)
- **Cascade escalation** — monitors response quality and re-dispatches to a higher tier if insufficient
- **Online optimizer** — adjusts tier boundaries based on observed quality over time
- **Budget limits** — per-model-group hourly/daily/monthly USD caps that downgrade tiers when reached
- **A/B testing** — compare routing policies with traffic splitting
- **Semantic routing cache** — caches classification decisions for semantically similar requests
- **Safe simulation** — test routing decisions without sending requests (Admin Panel → Smart Routing → Simulate Without Generation)

Per-model-group overrides allow different classifier modes, thresholds, and cascade settings per group. The Admin Panel provides a dedicated **Smart Routing** tab with full configuration, ONNX asset management, and simulation tooling.

For full details, see the [Smart Routing wiki page](https://github.com/fdanobey/OBEY-api-gateway/wiki/Smart-Routing).

## API Endpoints

All `/v1/*` endpoints are OpenAI-compatible. Requests include an `x-trace-id` response header for correlation.

| Method | Path | Description |
|--------|------|-------------|
| `POST` | `/admin/oauth/openai/login` | Initiate OpenAI OAuth login |
| `GET` | `/admin/oauth/openai/status` | OAuth session status |
| `POST` | `/admin/oauth/openai/logout` | Clear OAuth tokens |
| `POST` | `/admin/keys` | Create a virtual key (returns full `vk_` value once) |
| `GET` | `/admin/keys` | List virtual keys (paginated) |
| `GET` | `/admin/keys/{id}` | Inspect a virtual key |
| `PATCH` | `/admin/keys/{id}` | Update virtual key constraints |
| `DELETE` | `/admin/keys/{id}` | Delete a virtual key + its usage history |
| `POST` | `/admin/keys/{id}/revoke` | Revoke a virtual key |
| `GET` | `/admin/keys/{id}/usage` | Per-key usage aggregate for a time range |
| `GET` | `/admin/loop-detection/sessions` | List active loop detection sessions |
| `GET` | `/admin/loop-detection/sessions/{id}` | Session detail with signal history |
| `POST` | `/admin/loop-detection/sessions/{id}/reset` | Reset session enforcement state |
| `GET` | `/admin/loop-detection/stats` | Aggregate loop detection stats |
| `GET` | `/admin/memory/entries?namespace=...` | List memory entries for a namespace |
| `POST` | `/admin/memory/entries` | Create a memory entry |
| `DELETE` | `/admin/memory/entries/{id}` | Delete a memory entry |
| `DELETE` | `/admin/memory/namespaces/{namespace}` | Clear all entries in a namespace |
| `GET` | `/admin/memory/stats` | Memory store statistics |
| `GET` | `/admin/memory/projects` | List detected project namespaces |
| `GET` | `/health` | Health check |
| `GET` | `/metrics` | Prometheus metrics |
| `POST` | `/v1/chat/completions` | Chat completions (streaming + non-streaming) |
| `POST` | `/v1/completions` | Legacy completions |
| `POST` | `/v1/embeddings` | Embeddings |
| `GET` | `/v1/models` | List available models |
| `POST` | `/v1/images/generations` | Image generation |
| `POST` | `/v1/audio/transcriptions` | Audio transcription |
| `POST` | `/v1/audio/translations` | Audio translation |
| `*` | `/v1/assistants/**` | Assistants API passthrough |
| `*` | `/v1/threads/**` | Threads & messages passthrough |
| `*` | `/v1/files/**` | Files API passthrough |
| `*` | `/v1/fine_tuning/**` | Fine-tuning API passthrough |

## How Routing Works

1. Your app requests model `"gpt-4-group"`
2. The gateway finds the matching model group
3. Providers are sorted by priority → cost → latency
4. Providers with open circuit breakers or exhausted rate limits are skipped
5. The request goes to the highest-priority available provider
6. On failure, the gateway retries with the next provider in the list
7. If a context-length error is detected, the gateway truncates and retries automatically

### Circuit Breaker

Each provider has an independent circuit breaker. After `failure_threshold` consecutive failures, the circuit opens and the provider is temporarily removed from rotation. Backoff follows a configurable sequence (e.g. 5s → 10s → 20s → 40s → 300s). Circuit breakers reset on config hot-reload.

### Rate Limit Handling

When a provider returns a 429 (or a rate-limit-shaped HTTP 200 envelope from providers like Nano-GPT and OpenRouter), the gateway:

1. Fails over to the next provider immediately — no retry against the rate-limited one
2. Parses the upstream's reset signal and applies a per-provider cooldown
3. Skips the cooled-down provider in `select_provider_order` until the window expires

Signals consulted, in order of preference:

| Source | Examples |
|--------|----------|
| `Retry-After` header | seconds (`Retry-After: 60`) or RFC 2822 date |
| `retry-after-ms` header | millisecond precision (Anthropic) |
| `X-RateLimit-Reset` / `X-RateLimit-Reset-After` | epoch seconds or relative seconds (OpenAI, GitHub-style) |
| `anthropic-ratelimit-*-reset` | RFC 3339 ISO timestamps |
| `error.retry_after` / `retry_after_ms` body fields | numeric seconds / ms |
| `error.reset_at` / `reset` body fields | epoch seconds or RFC 3339 |
| Period markers in error message | "weekly limit" → 7d, "daily limit" → 24h, "hourly limit" → 1h |

The chosen cooldown is bounded by, in order: per-provider `max_rate_limit_cooldown_seconds`, the global `retry.max_rate_limit_cooldown_seconds` cap, and a hard 7-day safety backstop in the limiter.

```yaml
retry:
  # Global policy cap for rate-limit cooldowns.
  # 24h covers daily quotas without burning weekly ones.
  max_rate_limit_cooldown_seconds: 86400
  # Cooldown applied when no upstream signal is parseable.
  default_rate_limit_cooldown_seconds: 30

providers:
  - name: "nano-gpt"
    type: "openai"
    base_url: "https://nano-gpt.com/api/v1"
    api_key_env: "NANO_GPT_API_KEY"
    # Opt this provider into a 7-day cooldown for weekly quota windows.
    # Without this, "weekly limit reached" gets clamped to the 24h global cap.
    max_rate_limit_cooldown_seconds: 604800
```

Per-provider rate limiting is also enforced internally via a token bucket (`rate_limit_per_minute`), independent of upstream signals.

### Context Management

When a provider returns a context-length error, the gateway can automatically truncate the conversation and retry:

- **`remove_oldest`** — removes oldest messages, preserving system messages
- **`sliding_window`** — keeps only the N most recent messages

```yaml
context:
  enabled: true
  truncation_strategy: "remove_oldest"
  sliding_window_size: 10
  max_truncation_retries: 3
```

## Admin Panel & Dashboard

Both are embedded SPAs compiled into the binary — no external dependencies.

- **Admin** (`/admin`) — provider configuration, API key management, circuit breaker status, smart model routing (classifier, tiers, cascade, simulation), token compression & tool compression settings, persistent memory configuration & entry browser, config hot-reload
- **Dashboard** (`/dashboard`) — real-time metrics via WebSocket, provider health, in-flight request tracking (per-request phase, target provider, elapsed time), compression statistics (token & tool), persistent memory event visualizations (injection/extraction/eviction timeline, namespace activity), error logs, request log viewer

```yaml
admin:
  enabled: true
  path: "/admin"
  auth:
    enabled: true
    username_env: "ADMIN_USERNAME"
    password_env: "ADMIN_PASSWORD"

dashboard:
  enabled: true
  path: "/dashboard"
```

## Assistants API

The gateway includes a full local implementation of the OpenAI Assistants API — no upstream proxy required. All data is stored in a dedicated SQLite database (`assistants.db`, created automatically alongside your logging database).

### Supported Endpoints

| Method | Endpoint | Description |
|--------|----------|-------------|
| `POST` | `/v1/assistants` | Create an assistant |
| `GET` | `/v1/assistants` | List assistants |
| `GET` | `/v1/assistants/{id}` | Retrieve an assistant |
| `POST` | `/v1/assistants/{id}` | Modify an assistant |
| `DELETE` | `/v1/assistants/{id}` | Delete an assistant |
| `POST` | `/v1/threads` | Create a thread (optional initial messages) |
| `GET` | `/v1/threads` | List threads |
| `GET` | `/v1/threads/{id}` | Retrieve a thread |
| `POST` | `/v1/threads/{id}` | Modify a thread |
| `DELETE` | `/v1/threads/{id}` | Delete a thread (cascades) |
| `POST` | `/v1/threads/{id}/messages` | Create a message |
| `GET` | `/v1/threads/{id}/messages` | List messages |
| `GET` | `/v1/threads/{id}/messages/{msg_id}` | Retrieve a message |
| `POST` | `/v1/threads/{id}/messages/{msg_id}` | Modify a message |
| `DELETE` | `/v1/threads/{id}/messages/{msg_id}` | Delete a message |
| `POST` | `/v1/threads/{id}/runs` | Start a run |
| `GET` | `/v1/threads/{id}/runs` | List runs |
| `GET` | `/v1/threads/{id}/runs/{run_id}` | Get run status |
| `POST` | `/v1/threads/{id}/runs/{run_id}/cancel` | Cancel a run |
| `GET` | `/v1/threads/{id}/runs/{run_id}/steps` | List run steps |
| `POST` | `/v1/files` | Upload a file |
| `GET` | `/v1/files` | List files |
| `GET` | `/v1/files/{file_id}` | Get file metadata |
| `GET` | `/v1/files/{file_id}/content` | Download file content |
| `DELETE` | `/v1/files/{file_id}` | Delete a file |

### Multi-Tenant Isolation

When virtual keys are enabled, each `vk_` key gets its own isolated namespace. Assistants, threads, messages, and files created by one key are invisible to another. Resource quotas are enforced per-owner:

| Resource | Limit |
|----------|-------|
| Threads per owner | 1,000 |
| Messages per thread | 10,000 |
| Files per owner | 1,000 |
| File storage per owner | 256 MB |
| Single file size | 4 MB |

### Run Execution

When a run is started, the gateway builds a chat completion request from the thread's messages plus the assistant's instructions, then routes it through the normal model group routing (with failover, circuit breakers, etc.). The assistant's `model` field maps to a configured model group name.

> **Note:** Tool action execution (model calling tools mid-run and returning results iteratively) is not yet supported. Runs that require tool outputs will complete with `requires_action` status.

## Desktop / System Tray Mode

When built with `--features tray` on Windows, the binary runs as a desktop application:

- First launch shows a splash screen, starts the gateway, and opens the dashboard
- Subsequent launches start silently in the system tray
- Tray menu provides quick access to Dashboard, Admin, server status, and Quit
- Single-instance guard prevents duplicate processes
- Optional Windows login startup entry

## Project Structure

```
.
├── Cargo.toml                        # Workspace manifest
├── crates/
│   └── ai-gateway/
│       ├── Cargo.toml                # Crate manifest & dependencies
│       ├── build.rs                  # Windows resource embedding
│       ├── config.example.yaml       # Reference configuration
│       └── src/
│           ├── main.rs               # Entry point, CLI, tray bootstrap
│           ├── lib.rs                # Public module exports
│           ├── config/               # Config structs & validation
│           ├── gateway/              # HTTP server, middleware, route handlers
│           ├── router/               # Provider selection, circuit breaker, rate limiter
│           ├── providers/            # Provider implementations (8 providers)
│           ├── context/              # Context window management & truncation
│           ├── guardrail/            # Guardrail pipelines: PII redaction/re-injection, regex, Presidio, semantic, refusal detection & failover
│           ├── loop_detection/      # Agent loop detection: multi-signal scoring, graduated enforcement, session management, admin API
│           ├── cache/                # Response caching: in-memory exact-match (tier 1) + optional Qdrant semantic (tier 2)
│           ├── admin/                # Admin panel routes & embedded UI
│           ├── dashboard/            # Dashboard routes & WebSocket metrics
│           ├── logger/               # SQLite request logging
│           ├── metrics/              # Prometheus metrics
│           ├── oauth/                # OpenAI OAuth 2.0 login (PKCE flow)
│           ├── codex/               # Codex backend: Chat Completions → Responses API translation, model discovery, instructions store, web search
│           ├── virtual_keys/         # Virtual key management (auth, budgets, rate limits, usage, admin API)
│           ├── compression/          # Token compression engines & pipelines
│           ├── tool_compression/     # Tool definition compression: 12-stage pipeline, provider-aware middleware
│           ├── structured_output/    # JSON Schema response validation with retry
│           ├── smart_routing/       # Smart model routing: complexity classification, tier selection, cascade, A/B testing
│           ├── assistants/          # OpenAI Assistants API: local SQLite-backed CRUD for assistants, threads, messages, runs, files
│           ├── active_requests.rs   # Live in-flight request registry for dashboard phase tracking
│           ├── request_body_limit.rs # Dynamic per-request body size enforcement (hot-reloadable)
│           ├── memory/               # Persistent memory store: extraction, injection, decay, namespaces, Qdrant
│           ├── secrets.rs            # API key encryption/decryption
│           ├── error/                # Error types & HTTP status mapping
│           ├── models/               # OpenAI-compatible data models
│           └── tray/                 # Windows system tray (feature-gated)
├── scripts/
│   ├── build-release.ps1             # Release packaging script
│   ├── build-installer.ps1           # Inno Setup installer build
│   └── installer.iss                 # Inno Setup configuration
├── Assets/                           # Icons and logos
└── .github/workflows/release.yml     # CI/CD: build + GitHub Release on tag
```

## Technologies

| Category | Technology |
|----------|-----------|
| Language | Rust (2021 edition) |
| Async runtime | Tokio |
| Web framework | Axum + Tower middleware |
| HTTP client | Reqwest |
| Database | SQLite (rusqlite, bundled) |
| Vector DB | Qdrant (optional, for semantic cache) |
| Crypto | ring + base64 |
| TLS | rustls via axum-server |
| AWS | aws-sdk-bedrockruntime + aws-sdk-bedrock + aws-config |
| CLI | clap |
| Logging | tracing + tracing-subscriber |
| Asset embedding | rust-embed |
| Testing | proptest, wiremock, tempfile |
| CI/CD | GitHub Actions |
| Installer | Inno Setup 6 |

## Building for Release

```powershell
# Full release package (binary + assets + zip)
powershell -ExecutionPolicy Bypass -File ./scripts/build-release.ps1

# Release package + Windows installer
powershell -ExecutionPolicy Bypass -File ./scripts/build-installer.ps1
```

The release profile is optimized for size and performance:

```toml
[profile.release]
opt-level = 3
lto = true
strip = true
codegen-units = 1
panic = "abort"
```

## Testing

```bash
cargo test -p ai-gateway               # All tests
cargo test -p ai-gateway <test_name>   # Single test
cargo test -p ai-gateway -- --nocapture # With output
```

Tests use `tower::ServiceExt::oneshot()` for integration testing (no port binding) and `proptest` for property-based validation of config parsing and input handling.

## Contributing

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/my-feature`)
3. Make your changes — match the existing code style and patterns
4. Run `cargo test -p ai-gateway` and ensure all tests pass
5. Run `cargo clippy -p ai-gateway` for lint checks
6. Submit a pull request

### Guidelines

- Keep patches focused and minimal
- Pin dependency versions; justify new dependencies
- Use environment variables for secrets — never hardcode API keys
- Add tests for new routing logic or provider implementations
- Property-based tests (`proptest`) are preferred for input validation

## License

This project is licensed under the [MIT License](LICENSE).

---

## Troubleshooting

| Symptom | Likely Cause | Fix |
|---------|-------------|-----|
| `Connection refused` on startup | Port already in use | Change `server.port` in config or stop the conflicting process |
| Provider returns 401 | API key not resolved | Check that `api_key_env` matches an exported env var, or use the Admin UI to set it |
| All providers circuit-broken | Sustained upstream failures | Use `/admin/config/reload` to reset breakers, or check provider status pages |
| Provider hammered with 429s every few minutes | Weekly-quota provider (Nano-GPT etc.) capped at the 24h global default | Set `max_rate_limit_cooldown_seconds: 604800` on that provider so its cooldown can extend to a full week when the upstream signals "weekly limit reached" |
| Context-length errors loop | Truncation disabled or max retries hit | Enable `context.enabled: true` and increase `max_truncation_retries` |
| Dashboard shows no data | WebSocket blocked by proxy | Ensure your reverse proxy passes `Upgrade: websocket` headers |
| Cache Hit Rate stuck on `N/A` | Zero eligible requests observed yet | `N/A` means the cache has never been consulted. Send two identical requests with `temperature ≤ 0.15` and `n: 1`. Tool-using requests are eligible too. |
| Agent gets 429 with `loop_detected` | Loop detection hard-stop triggered | The agent is stuck in a repetitive loop. Reset the session via `POST /admin/loop-detection/sessions/{id}/reset`, or adjust thresholds/consecutive counts. Check the `dominant_signal` field for the root cause. |
| `x-loop-warning` header appearing | Loop confidence is elevated | Not blocking yet — the agent is showing repetitive patterns. Monitor the dominant signal. If false-positive, raise thresholds or lower the relevant signal weight. |
| Timeout on large prompts | `total_timeout_seconds` too low for model | Increase the provider's `total_timeout_seconds`. For thinking models (o1, o3, DeepSeek-R1), also check `ttfb_timeout_seconds` — these models need longer to start responding |
