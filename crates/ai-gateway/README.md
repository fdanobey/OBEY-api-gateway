# AI Gateway

OpenAI-compatible API gateway with intelligent routing, automatic failover, and multi-provider support. Single Rust binary with embedded admin/dashboard assets plus a Chart.js CDN dependency for the dashboard UI.

## Quick Start

```bash
# Build
cargo build --release -p ai-gateway

# Run (looks for ./config.yaml by default)
./target/release/ai-gateway

# Or specify a config file
./target/release/ai-gateway --config /path/to/config.yaml
```

### Desktop / System Tray Mode

When built with the tray feature on Windows, the executable can be launched by double-clicking `ai-gateway.exe` without command-line arguments.

- first launch shows the logo splash flow, starts the gateway in the background, opens the dashboard automatically, and marks the launch as completed in `config.yaml`
- subsequent launches start in the background with tray-mode behavior and do not auto-open the browser
- the tray menu exposes dashboard launch, server status, and quit actions

Build the desktop-enabled binary with:

```bash
cargo build --release -p ai-gateway --features tray
```

## Configuration

Config file resolution order:

1. `--config` CLI flag
2. `CONFIG_PATH` environment variable
3. `./config.yaml` in working directory

### Example `config.yaml`

```yaml
server:
  host: "0.0.0.0"
  port: 8080
  request_timeout_seconds: 30
  max_request_size_mb: 10

# Optional TLS
tls:
  enabled: false
  cert_path: "./cert.pem"
  key_path: "./key.pem"

providers:
  - name: "openai"
    type: "openai"
    base_url: "https://api.openai.com/v1"
    api_key_env: "OPENAI_API_KEY"
    timeout_seconds: 30
    max_connections: 100
    rate_limit_per_minute: 60

  - name: "ollama"
    type: "ollama"
    base_url: "http://localhost:11434"
    timeout_seconds: 60

  - name: "bedrock-api-key"
    type: "bedrock"
    region: "us-east-1"
    api_key_env: "AWS_BEARER_TOKEN_BEDROCK"
    timeout_seconds: 60

  - name: "bedrock-sdk"
    type: "bedrock"
    region: "us-east-1"
    timeout_seconds: 60

model_groups:
  - name: "gpt-4-group"
    version_fallback_enabled: false
    models:
      - provider: "openai"
        model: "gpt-4"
        cost_per_million_input_tokens: 10.0
        cost_per_million_output_tokens: 30.0
        priority: 100

# Optional sections
cors:
  enabled: true
  allowed_origins: ["*"]
  allowed_methods: ["GET", "POST", "OPTIONS"]
  allowed_headers: ["Content-Type", "Authorization"]

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
  metrics_update_interval_seconds: 1

first_launch_completed: false

tray:
  show_notifications: true
  auto_open_browser: true
  splash_duration_ms: 3000

circuit_breaker:
  failure_threshold: 3
  backoff_sequence_seconds: [5, 10, 20, 40, 300]
  success_threshold: 1

retry:
  max_retries_per_provider: 1
  backoff_sequence_seconds: [1, 2, 4]

logging:
  level: "info"
  database_path: "./logs.db"
  request_body_logging: false
  response_body_logging: false
  retention_days: 30

semantic_cache:
  enabled: false
  qdrant_url: "http://localhost:6334"
  collection_name: "ai_gateway_cache"
  similarity_threshold: 0.95
  embedding_provider: "openai"
  embedding_model: "text-embedding-3-small"
  ttl_seconds: 3600
  max_cache_size: 10000

prometheus:
  enabled: true
  path: "/metrics"

# Context management - automatic context window handling
context:
  enabled: true
  truncation_strategy: "remove_oldest"  # "remove_oldest" or "sliding_window"
  sliding_window_size: 10  # Number of messages to keep (for sliding_window strategy)
  capabilities_cache_ttl_seconds: 3600  # Cache TTL for model capabilities
  max_truncation_retries: 3  # Max retries when context exceeds limits
```

## Environment Variables

| Variable | Purpose |
|---|---|
| `CONFIG_PATH` | Override config file location |
| `AI_GATEWAY_DATA_DIR` | Override the secrets/master-key directory (recommended for Docker; mount a volume so encrypted keys persist) |
| `OPENAI_API_KEY` | OpenAI provider API key (name matches `api_key_env` in config) |
| `ADMIN_USERNAME` | Admin panel username (when `admin.auth.enabled: true`) |
| `ADMIN_PASSWORD` | Admin panel password (when `admin.auth.enabled: true`) |
| `RUST_LOG` | Tracing filter (e.g. `info`, `debug`, `ai_gateway=trace`) |

Provider keys can now be configured in three ways:

- environment variable reference in [`api_key_env`](crates/ai-gateway/config.example.yaml)
- plaintext key entered through the admin UI or YAML as migration/setup input
- persisted encrypted value in `api_key_encrypted`

### Bedrock authentication modes

Bedrock supports both of the following provider configurations:

- `api_key_env` set: the gateway uses the Bedrock Mantle endpoint at `https://bedrock-mantle.<region>.api.aws/v1` with `Authorization: Bearer <token>`
- no `api_key_env`: the gateway falls back to AWS SDK credential resolution using environment variables, shared credentials, or IAM role credentials

Example Bedrock API key provider:

```yaml
providers:
  - name: "bedrock-api-key"
    type: "bedrock"
    region: "us-east-1"
    api_key_env: "AWS_BEARER_TOKEN_BEDROCK"
    timeout_seconds: 60
```

Example Bedrock AWS SDK provider:

```yaml
providers:
  - name: "bedrock-sdk"
    type: "bedrock"
    region: "us-east-1"
    timeout_seconds: 60
```

When plaintext is submitted through the admin UI or found in YAML during a save flow, the gateway encrypts it automatically and writes only the encrypted payload back to `config.yaml`. The machine-local master key is stored outside YAML in the platform config directory (for example `%APPDATA%\ai-gateway\master.key` on Windows).

Example encrypted provider entry:

```yaml
providers:
  - name: "openai"
    type: "openai"
    base_url: "https://api.openai.com/v1"
    api_key_encrypted: "enc-v1:<nonce>:<ciphertext>"
```

The admin UI shows whether a provider key is missing, environment-backed, encrypted, or plaintext pending encryption. Custom headers also support `${ENV_VAR}` substitution.

## API Endpoints

| Method | Path | Description |
|---|---|---|
| `GET` | `/health` | Health check (200 OK / 503 shutting down) |
| `GET` | `/metrics` | Prometheus metrics (when enabled) |
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

All `/v1/*` endpoints are OpenAI-compatible.

The chat completions endpoint echoes request correlation through the `x-trace-id` response header. If the client provides `x-request-id` or `x-trace-id`, that value is reused; otherwise the gateway generates one.

## Admin Panel & Dashboard

- **Admin panel**: configurable via `admin.path`, disabled when `admin.enabled: false`
- **Dashboard**: configurable via `dashboard.path`, disabled when `dashboard.enabled: false`

Both are embedded SPAs served from the binary. The admin UI now round-trips the live Rust config schema, including retry backoff arrays, circuit-breaker backoff arrays, semantic cache settings, and context controls. YAML imports are loaded into the form for review and are only applied after an explicit save.

The dashboard supports:
- Real-time metrics over WebSocket
- Provider and per-model circuit-breaker visibility
- Recent error retrieval from logged failed requests
- Log viewer filtering by timestamp, provider, model, status code, and trace id

For custom paths, the embedded UIs automatically use the configured admin/dashboard route prefixes for links and API calls.

## Building for Release

```bash
cargo build --release -p ai-gateway --features tray
# Binary at: target/release/ai-gateway
```

For a packaged Windows desktop release with the splash assets and README copied into `./release`, run:

```powershell
powershell -ExecutionPolicy Bypass -File ./scripts/build-release.ps1
```

The release script builds the tray-enabled executable, copies [`Assets/icon.ico`](../../Assets/icon.ico) and [`Assets/logo.jpg`](../../Assets/logo.jpg) into the release bundle, and produces a zip archive for distribution.

The packaged release also includes a ready-to-edit [`config.yaml`](config.example.yaml) alongside [`config.example.yaml`](config.example.yaml), so a double-click launch does not fail when no configuration file exists yet.

## Context Management

The gateway automatically handles context window limits to prevent `context_length_exceeded` errors from providers. When a request exceeds a model's context window, the gateway can automatically truncate the context using configurable strategies.

### How It Works

1. **Model Capabilities Discovery**: When providers expose context window limits via their `/v1/models` endpoint, the gateway caches this information for intelligent routing decisions.

2. **Automatic Truncation**: When a context length error is detected, the gateway can automatically truncate the request and retry:
   - **`remove_oldest`**: Removes the oldest messages while preserving system messages
   - **`sliding_window`**: Keeps only the most recent N messages

3. **Configuration**: Enable and configure via the `context` section in `config.yaml`:
   ```yaml
   context:
     enabled: true
     truncation_strategy: "remove_oldest"
     sliding_window_size: 10
     capabilities_cache_ttl_seconds: 3600
     max_truncation_retries: 3
   ```

4. **Admin UI**: Configure context management settings from the "Context" tab in the admin panel at `/admin`.

### Supported Error Messages

The gateway detects context length errors from various provider error messages:
- `context_length_exceeded`
- `maximum context length`
- `token limit`
- `context window`
- `too many tokens`
- `input is too long`
