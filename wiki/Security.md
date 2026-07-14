# Security

OBEY API Gateway is designed with security-first principles: encrypted key storage, optional TLS, admin authentication, and careful secrets handling.

---

## Encrypted API Key Storage

Provider API keys are encrypted at rest using AES-256-GCM.

### How It Works

1. A **master key** is generated on first run and stored outside the config file
2. Provider keys entered via the Admin UI are encrypted before being written to `config.yaml`
3. Encrypted keys are stored as `api_key_encrypted: "enc-v1:<nonce>:<ciphertext>"`
4. At runtime, keys are decrypted in memory only when needed for a provider request

### Master Key Location

| Platform | Location |
|----------|----------|
| Windows | `%APPDATA%\ai-gateway\master.key` |
| Linux | `~/.config/ai-gateway/master.key` |
| Docker | `/data/master.key` (via `AI_GATEWAY_DATA_DIR`) |

> **Critical:** If the master key is lost, all encrypted API keys become undecryptable. Back up the master key or use environment variable references instead.

### Overriding the Data Directory

```yaml
# Environment variable
AI_GATEWAY_DATA_DIR=/custom/path
```

For Docker, mount a persistent volume at `/data`:
```bash
docker run -v ai-gateway-data:/data obey-api-gateway
```

---

## TLS Configuration

Enable HTTPS for production deployments:

```yaml
tls:
  enabled: true
  cert_path: "./cert.pem"
  key_path: "./key.pem"
```

- Uses `rustls` (pure-Rust TLS implementation, no OpenSSL dependency)
- Supports PEM-encoded certificates and keys
- Certificate chain should include intermediates

### Generating Self-Signed Certificates (Development)

```powershell
# Using OpenSSL (if available)
openssl req -x509 -newkey rsa:4096 -keyout key.pem -out cert.pem -days 365 -nodes

# Or use mkcert for trusted local certs
mkcert -install
mkcert localhost 127.0.0.1 ::1
```

---

## Admin Authentication

Protect the admin panel and API with HTTP Basic Auth:

```yaml
admin:
  enabled: true
  path: "/admin"
  auth:
    enabled: true
    username_env: "ADMIN_USERNAME"
    password_env: "ADMIN_PASSWORD"
```

Credentials are resolved from environment variables at runtime — never stored in the config file.

```powershell
$env:ADMIN_USERNAME = "admin"
$env:ADMIN_PASSWORD = "your-secure-password"
ai-gateway --config ./config.yaml
```

When auth is enabled:
- All `/admin/*` endpoints require Basic Auth
- The dashboard remains publicly accessible (for monitoring displays)
- Invalid credentials return `401 Unauthorized`

---

## API Key Resolution Safety

The `api_key_env` field is designed to avoid key exposure:

1. Value is treated as an **environment variable name** first
2. The actual key lives in the environment, not in config YAML
3. Config files can be committed to version control without exposing secrets

```yaml
# Safe: references an env var
api_key_env: "OPENAI_API_KEY"

# Also works: literal key (but not recommended for shared configs)
api_key_env: "sk-actual-key-here"
```

---

## Virtual Key Security

[Virtual keys](Virtual-Keys) add an additional security layer:

- Callers never see real provider API keys
- Each virtual key has independent budgets and rate limits
- Keys are stored encrypted in a separate SQLite database (`keys.db`)
- Revocation is instant — no provider key rotation needed
- Full `vk_` token shown exactly once at creation; cannot be recovered

---

## OAuth Token Security

For [OAuth-based authentication](OAuth-and-Codex):

| Measure | Implementation |
|---------|---------------|
| Token storage | AES-256-GCM encryption at rest |
| Callback server | Binds to `127.0.0.1` only (not exposed to network) |
| PKCE flow | S256 challenge prevents authorization code interception |
| Token logging | Values never logged at any level |
| Background refresh | Tokens refreshed before expiry without user interaction |

---

## Logging Security

The logging system protects sensitive data:

```yaml
logging:
  excluded_fields: ["api_key", "authorization"]
  request_body_logging: false
  response_body_logging: false
  max_body_size_bytes: 10000
```

- **Excluded fields** are stripped from log entries
- Body logging is disabled by default
- When enabled, bodies are truncated to `max_body_size_bytes`
- Log retention is configurable with automatic cleanup

---

## Network Security

### CORS

Restrict which origins can access the gateway:

```yaml
cors:
  enabled: true
  allowed_origins: ["https://your-app.com"]
  allowed_methods: ["GET", "POST", "OPTIONS"]
  allowed_headers: ["Content-Type", "Authorization"]
```

### Rate Limiting

Per-provider rate limits prevent abuse:

```yaml
providers:
  - name: "openai"
    rate_limit_per_minute: 60     # 0 = unlimited
```

Per-key rate limits via [Virtual Keys](Virtual-Keys):
- `requests_per_minute` — RPM token bucket
- `tokens_per_minute` — TPM rolling window

---

## Secrets Handling Summary

| Secret Type | Storage | Access |
|-------------|---------|--------|
| Provider API keys | Encrypted in config (or env vars) | Decrypted in memory at runtime |
| Master encryption key | Platform-specific secure directory | Read once at startup |
| Admin credentials | Environment variables | Resolved at request time |
| OAuth tokens | Encrypted on disk | Decrypted for provider requests |
| Virtual keys | Encrypted in SQLite | Shown once at creation |

### What's Never Logged

- API keys and bearer tokens
- Authorization headers
- OAuth access/refresh tokens
- Master key material
- Request/response bodies (unless explicitly enabled)

---

## Docker Security Considerations

1. **Mount a persistent volume at `/data`** — preserves the master key across container rebuilds
2. **Use environment variables for secrets** — don't bake keys into images
3. **Map port 1455 only if OAuth is needed** — minimizes attack surface
4. **Run as non-root** — the image doesn't require root privileges

```bash
docker run -d \
  -p 8080:8080 \
  -e OPENAI_API_KEY=sk-... \
  -e ADMIN_USERNAME=admin \
  -e ADMIN_PASSWORD=secure-password \
  -v ai-gateway-data:/data \
  obey-api-gateway
```

---

## Next Steps

- [Configuration](Configuration) — full config reference
- [Virtual Keys](Virtual-Keys) — per-caller access control
- [OAuth & Codex](OAuth-and-Codex) — OAuth token details
