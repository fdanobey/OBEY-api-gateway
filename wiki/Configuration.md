# Configuration

OBEY API Gateway uses a single YAML configuration file. On first run, a default config is created automatically if none exists.

---

## Config File Resolution

The config file is resolved in this order:

1. `--config` CLI flag (highest priority)
2. `CONFIG_PATH` environment variable
3. `./config.yaml` in working directory

```powershell
# CLI flag
ai-gateway --config C:\path\to\config.yaml

# Environment variable
$env:CONFIG_PATH = "C:\path\to\config.yaml"
ai-gateway
```

---

## Admin Panel Configuration UI

The entire configuration can be managed visually through the Admin Panel:

![Admin Panel - Server Settings](images/admin-panel.png)

---

## Minimal Configuration

```yaml
server:
  host: "0.0.0.0"
  port: 8080
  request_timeout_seconds: 30

providers:
  - name: "openai"
    type: "openai"
    base_url: "https://api.openai.com/v1"
    api_key_env: "OPENAI_API_KEY"
    timeout_seconds: 30

model_groups:
  - name: "gpt-4-group"
    models:
      - provider: "openai"
        model: "gpt-4"
        priority: 1
```

---

## Full Configuration Reference

### Server

```yaml
server:
  host: "0.0.0.0"            # Bind address
  port: 8080                  # Listen port
  request_timeout_seconds: 30 # Global request timeout
  max_request_size_mb: 10     # Max request body size
```

### TLS

```yaml
tls:
  enabled: true
  cert_path: "./cert.pem"
  key_path: "./key.pem"
```

### Providers

See [Providers](Providers) for full details on each provider type.

```yaml
providers:
  - name: "openai"
    type: "openai"                    # openai | ollama | bedrock | groq | together | vllm | lmstudio | nvidia_nim
    base_url: "https://api.openai.com/v1"
    api_key_env: "OPENAI_API_KEY"     # Env var name, resolved at runtime
    timeout_seconds: 30               # Legacy total timeout
    ttfb_timeout_seconds: 30          # Time-to-first-byte timeout
    total_timeout_seconds: 300        # Total request timeout
    max_connections: 100              # Connection pool limit
    rate_limit_per_minute: 60         # 0 = unlimited
    custom_headers:
      X-Custom: "${MY_ENV_VAR}"       # Supports env var substitution
```

### Model Groups

```yaml
model_groups:
  - name: "gpt-4-group"
    version_fallback_enabled: false   # Try older versions on failure
    models:
      - provider: "openai"
        model: "gpt-4"
        cost_per_million_input_tokens: 10.0
        cost_per_million_output_tokens: 30.0
        priority: 100                 # Lower = higher priority
```

### Circuit Breaker

```yaml
circuit_breaker:
  failure_threshold: 3                # Failures before opening
  backoff_sequence_seconds: [5, 10, 20, 40, 300]
  success_threshold: 1                # Successes to close
```

![Circuit Breaker Settings](images/admin-circuit-breaker.png)

### Retry

```yaml
retry:
  max_retries_per_provider: 1
  backoff_sequence_seconds: [1, 2, 4]
```

### Logging

```yaml
logging:
  level: "info"                       # trace | debug | info | warn | error
  database_path: "./logs.db"          # SQLite log file
  request_body_logging: false
  response_body_logging: false
  max_body_size_bytes: 10000
  excluded_fields: ["api_key", "authorization"]
  retention_days: 30
  cleanup_schedule_hours: 24
```

### Admin Panel

```yaml
admin:
  enabled: true
  path: "/admin"
  auth:
    enabled: false
    username_env: "ADMIN_USERNAME"    # Env var for username
    password_env: "ADMIN_PASSWORD"    # Env var for password
```

### Dashboard

```yaml
dashboard:
  enabled: true
  path: "/dashboard"
  metrics_update_interval_seconds: 1
```

### CORS

```yaml
cors:
  enabled: false
  allowed_origins: ["*"]
  allowed_methods: ["GET", "POST", "OPTIONS"]
  allowed_headers: ["Content-Type", "Authorization"]
```

### Exact Cache (Tier 1)

```yaml
exact_cache:
  enabled: true
  max_entries: 5000
  ttl_seconds: 3600
  temperature_threshold: 0.15
```

### Semantic Cache (Tier 2)

```yaml
semantic_cache:
  enabled: true
  qdrant_url: "http://localhost:6334"       # gRPC port
  collection_name: "ai_gateway_cache"
  similarity_threshold: 0.95
  embedding_provider: "openai"
  embedding_model: "text-embedding-3-small"
  ttl_seconds: 3600
  max_cache_size: 10000
```

### Streaming

```yaml
streaming:
  emit_early_event: true              # Synthetic role event before upstream responds
  keepalive_interval_seconds: 5       # SSE keep-alive (0 = disabled)
  passthrough_enabled: true           # True SSE relay for capable providers
  chunk_timeout_seconds: 60           # Max gap between SSE chunks
  retry_on_truncation: true           # Failover on finish_reason=length
```

### Prometheus Metrics

```yaml
prometheus:
  enabled: true
  path: "/metrics"
```

### Virtual Keys

```yaml
virtual_keys:
  enforcement: disabled               # disabled | optional | required
  database_path: "./keys.db"
```

### System Tray (Windows)

```yaml
first_launch_completed: false

tray:
  show_notifications: true
  auto_open_browser: true
  splash_duration_ms: 3000
```

---

## Environment Variables

| Variable | Purpose |
|----------|---------|
| `CONFIG_PATH` | Override config file location |
| `AI_GATEWAY_DATA_DIR` | Override secrets/master-key directory (recommended for Docker) |
| `OPENAI_API_KEY` | Provider API key (name matches `api_key_env` in config) |
| `ADMIN_USERNAME` | Admin panel username |
| `ADMIN_PASSWORD` | Admin panel password |
| `RUST_LOG` | Tracing filter (`info`, `debug`, `ai_gateway=trace`) |
| `OAUTH_CALLBACK_BIND_HOST` | Bind address for OAuth callback (Docker: `0.0.0.0`) |

---

## API Key Resolution

The `api_key_env` field is resolved at runtime:

1. First tried as an **environment variable name** (e.g., `OPENAI_API_KEY` → looks up `$env:OPENAI_API_KEY`)
2. If the env var doesn't exist, the literal string value is used as the key

### Custom Header Environment Substitution

Custom headers support `${ENV_VAR}` syntax:

```yaml
providers:
  - name: "custom"
    type: "openai"
    custom_headers:
      X-API-Token: "${MY_SECRET_TOKEN}"
```

---

## Base URL Normalization

Provider URLs are automatically normalized:
- Trailing `/` is stripped
- `/v1` is appended if not already present (for OpenAI-compatible providers)

---

## Hot Reload

Configuration can be reloaded without restarting:

```bash
curl -X POST http://localhost:8080/admin/config/reload
```

Or use the Admin Panel UI. Circuit breakers are reset on config reload.

---

## Next Steps

- [Providers](Providers) — detailed provider configuration
- [Routing & Failover](Routing-and-Failover) — how requests are routed
- [Security](Security) — TLS, encryption, authentication
