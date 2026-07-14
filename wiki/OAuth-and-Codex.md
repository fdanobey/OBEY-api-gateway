# OAuth & Codex

OBEY API Gateway supports browser-based OpenAI authentication and Codex backend translation, enabling use of your ChatGPT Plus/Pro subscription through the gateway.

---

## OpenAI OAuth Login

Instead of manually creating and managing OpenAI API keys, authenticate with your ChatGPT subscription via browser-based OAuth.

### Configuration

```yaml
providers:
  - name: "openai-oauth"
    type: "openai"
    base_url: "https://api.openai.com/v1"
    auth_method: "oauth"              # Use OAuth instead of api_key_env
```

### Login Flow

```bash
# 1. Initiate browser-based login
curl -X POST http://localhost:8080/admin/oauth/openai/login

# 2. Browser opens to OpenAI's authorization page
#    (user signs in with ChatGPT credentials)

# 3. Check session status
curl http://localhost:8080/admin/oauth/openai/status

# 4. Logout when needed
curl -X POST http://localhost:8080/admin/oauth/openai/logout
```

### Token Lifecycle

The gateway handles the full token lifecycle automatically:

1. **Initiate** — opens your default browser to OpenAI's authorization page
2. **Callback** — receives the redirect on a local loopback server (port 1455)
3. **Exchange** — trades the authorization code for tokens (PKCE + S256)
4. **Persist** — encrypts and saves tokens to disk (survives restarts)
5. **Refresh** — renews the access token in the background before expiry
6. **Failover** — falls back to the next provider if the OAuth session expires

### Security

| Measure | Detail |
|---------|--------|
| Token encryption | AES-256-GCM at rest |
| Callback binding | `127.0.0.1` only (localhost) |
| Token logging | Values never logged at any level |
| PKCE flow | S256 challenge for authorization code exchange |

---

## Codex Backend Translation

When using OAuth authentication, the gateway can transparently route requests through the ChatGPT Codex backend, translating between the Chat Completions API and the Responses API on the fly.

### How It Works

```
Client (Chat Completions API)
         │
         ▼
┌─────────────────────┐
│  OBEY API Gateway   │
│                     │
│  Translate request: │
│  Chat Completions   │
│  → Responses API    │
└────────┬────────────┘
         │
         ▼
┌─────────────────────┐
│  ChatGPT Codex      │
│  Backend            │
│  (Responses API)    │
└────────┬────────────┘
         │
         ▼
┌─────────────────────┐
│  OBEY API Gateway   │
│                     │
│  Translate response:│
│  Responses API      │
│  → Chat Completions │
└────────┬────────────┘
         │
         ▼
Client (standard response)
```

### What Gets Translated

- **Request**: Chat Completions format → Responses API format
- **Response**: Responses API format → Chat Completions format
- **Streaming**: SSE events are translated on the fly

### Codex Instructions Store

The gateway maintains an instructions store for Codex-capable providers. System instructions are managed separately and injected into Codex requests as needed.

---

## Docker Considerations

When running in Docker, the OAuth callback server needs to be reachable from the host browser:

```dockerfile
# Already set in the official Dockerfile
ENV OAUTH_CALLBACK_BIND_HOST=0.0.0.0
EXPOSE 1455
```

Map port 1455 when running the container:

```bash
docker run -d \
  -p 8080:8080 \
  -p 1455:1455 \
  -v ai-gateway-data:/data \
  obey-api-gateway
```

---

## OAuth Usage Tracking

The gateway tracks OpenAI rate-limit headers from OAuth provider responses:

- Displayed in the admin UI for usage visibility
- Used as fallback cooldown when no `Retry-After` header is present on 429 responses
- Helps inform when you're approaching subscription limits

---

## Failover Behavior

OAuth providers participate in the normal failover chain:

1. If the OAuth token is valid → request routed through OAuth provider
2. If the token expires or refresh fails → circuit breaker trips
3. Gateway fails over to the next provider in the model group

This means you can configure OAuth as your primary provider with an API-key provider as fallback:

```yaml
providers:
  - name: "openai-oauth"
    type: "openai"
    base_url: "https://api.openai.com/v1"
    auth_method: "oauth"

  - name: "openai-api"
    type: "openai"
    base_url: "https://api.openai.com/v1"
    api_key_env: "OPENAI_API_KEY"

model_groups:
  - name: "gpt-4-group"
    models:
      - provider: "openai-oauth"
        model: "gpt-4"
        priority: 1               # Try OAuth first (free with subscription)
      - provider: "openai-api"
        model: "gpt-4"
        priority: 2               # Fall back to API key
```

---

## Next Steps

- [Security](Security) — token encryption and storage
- [Providers](Providers) — provider configuration
- [Admin Panel & Dashboard](Admin-Panel-and-Dashboard) — OAuth management UI
