# GitHub Copilot Provider

OBEY API Gateway can route requests through an existing GitHub Copilot subscription (Free, Pro, Pro+, Business, or Enterprise with a Copilot Chat seat). A `github-copilot` provider signs in with GitHub via a device-flow login started from the admin panel, exchanges the GitHub token for a short-lived Copilot API token, refreshes it in the background, and dispatches OpenAI-compatible chat-completions traffic (streaming included) through the same model-group, failover, budget, and observability machinery as every other provider.

Models used through this provider are subscription-included: token usage is recorded at zero cost, with model name and provider attribution preserved.

---

## Support Status

The endpoint set this provider uses (`api.githubcopilot.com` and the Copilot token exchange) is unofficial. It is not part of GitHub's documented public API, it may change or break without notice, and it is used by major third-party clients: opencode (with GitHub's announced partnership support), LiteLLM, and copilot-api.

If the surface changes, Copilot candidates fail over to the next provider in the model group rather than producing outages.

---

## Configuration

```yaml
providers:
- name: "github-copilot"
type: "github-copilot"
auth_method: "device-oauth" # default; sign in via the admin panel
timeout_seconds: 60
```

`base_url` is optional and defaults to `https://api.githubcopilot.com`.

### PAT Variant (Headless)

For containers and headless deployments, authenticate with a GitHub PAT instead of device flow. The PAT is resolved through the standard `api_key`/`api_key_env` mechanism:

```yaml
providers:
- name: "github-copilot-pat"
type: "github-copilot"
auth_method: "api-key"
api_key_env: "GITHUB_COPILOT_PAT" # PAT from an account with a Copilot Chat seat
timeout_seconds: 60
```

`auth_method` is authoritative. If the selected method's credential is missing or invalid, the gateway fails with an actionable error rather than silently falling back to the other method.

---

## Signing In (Device Flow)

1. Add a `github-copilot` provider (device-oauth) and save.
2. On the provider card in the admin panel, click Sign in with GitHub.
3. The gateway displays a verification URL and a device code, and copies the code to the clipboard.
4. Open `https://github.com/login/device`, enter the code, and authorize.
5. The card switches to authenticated with token expiry and a usage snapshot.

Tokens are encrypted at rest with the same master-key mechanism as OpenAI OAuth (`enc-v1`) and are never logged. Logout (on the provider card or via `POST /admin/oauth/github-copilot/logout`) deletes the persisted tokens, cancels refresh tasks, and clears cached usage.

---

## Admin Endpoints

```text
POST /admin/oauth/github-copilot/login    # start device-flow login (returns URL + user code)
POST /admin/oauth/github-copilot/cancel   # cancel a pending login
POST /admin/oauth/github-copilot/logout   # delete tokens, stop refresh, clear usage
GET  /admin/oauth/github-copilot/status   # session state and token expiry
GET  /admin/oauth/github-copilot/usage    # usage snapshot from rate-limit headers
```

Calling `login` while a login is already in flight returns the in-flight verification URL and code instead of starting a second flow.

---

## Usage Display

Copilot responses carry `x-ratelimit-*` headers, which the gateway parses into two count-based windows (short and weekly). When `x-copilot-premium-requests-*` headers are present, the usage card also shows the premium-request percentage and its reset time.

Until a Copilot response has been observed, the card reads "no usage data yet" rather than showing zero values.

---

## Model Catalog

While the Copilot session is authenticated, the gateway fetches the live model catalog from the API base and exposes the models enabled for your account in `/v1/models`. Models are configured per group rather than per provider; when a model-group entry naming this provider lists a model the catalog does not contain, saving the config produces a warning (not a block). If the catalog fetch fails, the last known catalog keeps serving and a warning appears on the provider's admin status card.

---

## Routing and Failover

- Copilot candidates in model groups are ordered, retried, circuit-broken, and failed over with the same semantics as other providers, with attempts logged in the attempt ledger.
- When the session is unauthenticated or expired, the candidate is marked ineligible (skipped, not errored) so requests fail over without spurious attempt errors; the skip reason is visible through the existing skip observability.
- Upstream 401/403 triggers one token refresh and one re-dispatch attempt before failing over.

---

## Troubleshooting

| Symptom | Cause and fix |
|---------|---------------|
| Entitlement missing / Copilot Chat seat required | The token exchange failed because the account has no Copilot Chat access. Sign in with an account holding an active Copilot subscription with Chat, or use a PAT from such an account. |
| Token expired | The Copilot token can no longer be refreshed. The GitHub token is kept, so signing in again from the provider card restores the session. |
| Model not supported / absent from catalog | The live catalog lists only models enabled for your account. Pick model IDs that appear in `/v1/models` while the session is authenticated; configured models missing from the catalog trigger a save-time warning. |

---

## Next Steps

- [OAuth & Codex](OAuth-and-Codex) — the OpenAI OAuth flow this provider mirrors
- [Providers](Providers) — provider configuration reference
- [Routing & Failover](Routing-and-Failover) — how Copilot candidates participate in failover
