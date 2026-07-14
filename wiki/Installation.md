# Installation

OBEY API Gateway offers multiple deployment methods. Pick the one that matches your environment.

---

## Option 1: Download (Windows)

The easiest way to get started on Windows.

1. Download the [latest release](https://github.com/fdanobey/OBEY-api-gateway/releases/latest) — choose the **installer** (`.exe`) or the **portable zip**
2. Run the installer or extract the zip to your preferred location
3. Double-click `ai-gateway.exe`

The gateway starts on `http://localhost:8080` and opens the dashboard automatically on first launch.

### Windows System Tray Mode

The release build includes system tray integration:

- Splash screen on first launch
- System tray icon with context menu
- Single-instance enforcement (second launch brings existing instance to front)
- Notification when already running

### Running as a Windows Service

For always-on operation, register as a Windows service:

**Using NSSM (recommended):**
```powershell
# Install NSSM from https://nssm.cc
nssm install ai-gateway "C:\path\to\ai-gateway.exe"
nssm set ai-gateway AppDirectory "C:\path\to"
nssm set ai-gateway AppEnvironmentExtra "OPENAI_API_KEY=sk-..."
nssm start ai-gateway
```

**Using sc.exe:**
```powershell
sc create ai-gateway binPath="C:\path\to\ai-gateway.exe"
```

---

## Option 2: Deploy to Railway

One-click cloud deployment:

[![Deploy on Railway](https://railway.com/button.svg)](https://railway.com/deploy?template=https%3A%2F%2Fgithub.com%2Ffdanobey%2FOBEY-api-gateway)

Railway picks up the included `Dockerfile` and `railway.toml` automatically.

**Steps:**
1. Click the deploy button above
2. Set your provider API keys (`OPENAI_API_KEY`, etc.) as environment variables in the Railway dashboard
3. You're live in under a minute

> **Persist your keys on Railway:** Attach a Railway Volume mounted at `/data` (the image's `AI_GATEWAY_DATA_DIR`). Railway's container filesystem is ephemeral — without a volume the encryption master key regenerates on every redeploy and previously saved `api_key_encrypted` values become undecryptable. Alternatively, supply keys via plain environment variables which never touch the encrypted store.

---

## Option 3: Docker

### Build and Run

```bash
# Build the image
docker build -t obey-api-gateway .

# Run with config and env vars
docker run -d --name obey-api-gateway \
  -p 8080:8080 \
  -e OPENAI_API_KEY=sk-... \
  -v $(pwd)/config.yaml:/app/config.yaml \
  -v ai-gateway-data:/data \
  obey-api-gateway --config /app/config.yaml
```

> **Persist your keys:** The image sets `AI_GATEWAY_DATA_DIR=/data` and declares it as a volume. Mount a named volume (or host path) at `/data` so the encryption master key survives container restarts and rebuilds.

### Docker Compose

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
    environment:
      - OPENAI_API_KEY=sk-...

volumes:
  ai-gateway-data:
```

```bash
docker compose up -d
```

### Updating (Docker)

```bash
# Pull latest source and rebuild
git pull origin master
docker build -t obey-api-gateway .

# Stop and remove old container (data volume is preserved)
docker stop obey-api-gateway && docker rm obey-api-gateway

# Start with new image
docker run -d --name obey-api-gateway \
  -p 8080:8080 \
  -e OPENAI_API_KEY=sk-... \
  -v $(pwd)/config.yaml:/app/config.yaml \
  -v ai-gateway-data:/data \
  obey-api-gateway --config /app/config.yaml
```

With Compose:
```bash
git pull origin master
docker compose up -d --build
```

> Never `docker volume rm ai-gateway-data` unless you intend to reset all stored secrets.

### Exposed Ports

| Port | Purpose |
|------|---------|
| `8080` | Main gateway + admin + dashboard |
| `1455` | OAuth callback server (for browser-based OpenAI login) |

---

## Option 4: Build from Source

### Prerequisites

- [Rust](https://rustup.rs/) (stable toolchain)
- Windows, Linux, or macOS

### Build

```bash
# Clone
git clone https://github.com/fdanobey/OBEY-api-gateway.git
cd OBEY-api-gateway

# Build (headless — no tray icon)
cargo build --release -p ai-gateway

# Build with Windows tray support
cargo build --release -p ai-gateway --features tray
```

### Run

```bash
./target/release/ai-gateway --config ./config.yaml
```

On first run without a config file, a default `config.yaml` is created automatically.

---

## Verifying the Installation

After starting the gateway, verify it's running:

```bash
# Health check
curl http://localhost:8080/health

# Check available models
curl http://localhost:8080/v1/models
```

Open the dashboard in your browser:
```
http://localhost:8080/dashboard
```

![Dashboard Overview](images/dashboard-overview.png)

---

## Pointing Your App at the Gateway

Any OpenAI-compatible SDK or tool works:

```bash
# Environment variable
export OPENAI_API_BASE=http://localhost:8080/v1
```

```python
# Python
from openai import OpenAI
client = OpenAI(base_url="http://localhost:8080/v1", api_key="unused")
```

```typescript
// TypeScript / Node.js
import OpenAI from 'openai';
const client = new OpenAI({
  baseURL: 'http://localhost:8080/v1',
  apiKey: 'unused'
});
```

```bash
# curl
curl http://localhost:8080/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{"model":"gpt-4-group","messages":[{"role":"user","content":"Hello!"}]}'
```

---

## Next Steps

- [Configuration](Configuration) — customize the config file
- [Providers](Providers) — add your AI providers
- [Admin Panel & Dashboard](Admin-Panel-and-Dashboard) — explore the web UI
