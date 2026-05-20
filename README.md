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
- **Response caching** — built-in two-tier cache: in-memory exact-match (default-on, no setup) plus optional semantic Qdrant tier; works for both streaming and non-streaming requests, including tool-using clients
- **OpenAI OAuth login** — browser-based sign-in with your ChatGPT Plus/Pro subscription (PKCE flow, automatic token refresh)
- **Codex backend translation** — transparently routes OAuth-authenticated requests through the ChatGPT Codex backend, translating Chat Completions ↔ Responses API on the fly
- **Encrypted API key storage** — provider keys encrypted at rest with a machine-local master key
- **Admin panel & dashboard** — embedded web UIs for configuration, metrics, and log viewing
- **Prometheus metrics** — `/metrics` endpoint for existing monitoring infrastructure
- **Request logging** — SQLite-based structured logging with configurable retention
- **TLS support** — optional HTTPS with certificate configuration
- **Windows system tray** — double-click desktop app with splash screen and tray menu
- **Hot config reload** — change settings through the admin UI without restarting
- **Smart timeouts** — split TTFB / total timeouts with auto-detection of thinking models (o1, o3, DeepSeek-R1, Claude)

## Quick Start

### Option 1: Download (Windows)

Grab the [latest release](https://github.com/fdanobey/OBEY-api-gateway/releases/latest) — either the installer (`.exe`) or portable zip. Double-click to run. The gateway starts on `http://localhost:8080` and opens the dashboard automatically on first launch.

### Option 2: Deploy to Railway

Click the button above to deploy directly from this repo. Railway picks up the included [`Dockerfile`](Dockerfile) and [`railway.toml`](railway.toml) automatically. Set your provider API keys (`OPENAI_API_KEY`, etc.) as environment variables in the Railway dashboard and you're live in under a minute.

### Option 3: Docker

```bash
# Build the image
docker build -t obey-api-gateway .

# Run with config and env vars
docker run -d \
  -p 8080:8080 \
  -e OPENAI_API_KEY=sk-... \
  -v $(pwd)/config.yaml:/app/config.yaml \
  obey-api-gateway --config /app/config.yaml
```

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
| `OPENAI_API_KEY` | Provider API key (name matches `api_key_env` in config) |
| `ADMIN_USERNAME` | Admin panel username |
| `ADMIN_PASSWORD` | Admin panel password |
| `RUST_LOG` | Tracing filter (`info`, `debug`, `ai_gateway=trace`) |

## API Endpoints

All `/v1/*` endpoints are OpenAI-compatible. Requests include an `x-trace-id` response header for correlation.

| Method | Path | Description |
|--------|------|-------------|
| `POST` | `/admin/oauth/openai/login` | Initiate OpenAI OAuth login |
| `GET` | `/admin/oauth/openai/status` | OAuth session status |
| `POST` | `/admin/oauth/openai/logout` | Clear OAuth tokens |
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

- **Admin** (`/admin`) — provider configuration, API key management, circuit breaker status, config hot-reload
- **Dashboard** (`/dashboard`) — real-time metrics via WebSocket, provider health, error logs, request log viewer

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
│           ├── cache/                # Response caching: in-memory exact-match (tier 1) + optional Qdrant semantic (tier 2)
│           ├── admin/                # Admin panel routes & embedded UI
│           ├── dashboard/            # Dashboard routes & WebSocket metrics
│           ├── logger/               # SQLite request logging
│           ├── metrics/              # Prometheus metrics
│           ├── oauth/                # OpenAI OAuth 2.0 login (PKCE flow)
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
| AWS | aws-sdk-bedrockruntime + aws-config |
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
| Timeout on large prompts | `total_timeout_seconds` too low for model | Increase the provider's `total_timeout_seconds`. For thinking models (o1, o3, DeepSeek-R1), also check `ttfb_timeout_seconds` — these models need longer to start responding |
