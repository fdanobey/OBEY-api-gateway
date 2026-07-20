# Agent Loop Detection

AI coding agents and automation pipelines can get stuck in repetitive loops — retrying the same tool call, cycling through identical errors, or regenerating near-identical content without making progress. The loop detection system monitors request patterns per-session and applies graduated enforcement to break these loops before they burn tokens and cost.

Loop detection is **opt-in** (`enabled: false` by default) and operates as Tower middleware on `/v1/chat/completions`.

---

## How It Works

Each caller session is tracked independently, resolved by:
1. Virtual key ID (if present)
2. `x-session-id` request header
3. Client IP address

On every request, seven signals are computed from the session's recent history and combined into a single **confidence score** using configurable weights. The score is smoothed with an exponential moving average (EMA) to prevent single-request spikes from triggering enforcement. When confidence exceeds the configured threshold for a required number of consecutive requests, the enforcement level escalates.

Sessions automatically expire after `session_timeout_minutes` of inactivity. An LRU eviction loop runs periodically to cap memory usage at `max_sessions`.

---

## Detection Signals

| Signal | Weight (default) | What it measures |
|--------|:---:|-----------------|
| `content_similarity` | 0.25 | SimHash similarity between the current request and recent requests in the session window |
| `tool_call_repetition` | 0.20 | Consecutive identical tool-call fingerprints (function name + argument hash) |
| `response_stagnation` | 0.15 | Provider responses with matching block-structure hash and near-identical token counts |
| `error_cycling` | 0.15 | Repeated requests after provider errors with high content similarity (same failing request retried) |
| `token_velocity` | 0.10 | Tokens consumed per minute exceeding `token_velocity_threshold` |
| `context_growth` | 0.10 | Context token growth disproportionate to new unique information tokens |
| `cost_velocity` | 0.05 | USD spend rate per minute exceeding `cost_velocity_threshold` |

**Weights must sum to 1.0** (within a tolerance of 0.001). Adjust weights to prioritize signals relevant to your workload — for example, increase `tool_call_repetition` for tool-heavy agents, or `error_cycling` for retry-prone pipelines.

---

## Enforcement Levels

Enforcement escalates through four levels, each requiring the confidence to exceed its threshold for a minimum number of consecutive scored requests:

| Level | Default Threshold | Default Consecutive | Behavior |
|-------|:---------:|:---:|----------|
| **None** | — | — | Normal operation, no intervention |
| **Warn** | 0.30 | 3 | Adds `x-loop-warning` response header with confidence value and dominant signal name |
| **Throttle** | 0.50 | 5 | Introduces an artificial delay (default 2s) before forwarding the request |
| **Inject** | 0.70 | 7 | Appends a break instruction to the system prompt telling the model to change strategy |
| **Hard-Stop** | 0.90 | 10 | Rejects the request with HTTP `429 Too Many Requests` and `Retry-After: 60` |

### De-escalation

After **5 consecutive low-confidence requests** (below `warn_confidence`), the enforcement level steps down one level. This allows sessions to recover naturally when the agent breaks out of a loop.

---

## Injection Strategies

When the enforcement level reaches **Inject**, the gateway modifies the outgoing request by appending an instruction to the system prompt (or inserting one if no system message exists):

| Strategy | Behavior |
|----------|----------|
| `system_prompt_append` (default) | Appends a generic instruction: *"Loop detected. You are repeating the same actions without progress. Stop the current approach, summarize what you have tried, and propose a fundamentally different strategy."* |
| `context_aware` | Tailors the instruction based on the dominant signal. For `tool_call_repetition` it names the specific tool. For `error_cycling` it tells the model to stop retrying the failing operation. |

### Custom Break Instruction

You can provide a `break_instruction_template` (1–2000 characters) to override the default text:

```yaml
loop_detection:
  break_instruction_template: "STOP. You are in a loop. Ask the user for help instead of continuing."
```

This can also be set per-virtual-key for multi-tenant deployments.

---

## Configuration Reference

```yaml
loop_detection:
  enabled: true                          # Master switch (default: false)
  session_timeout_minutes: 30            # Session TTL after last activity (1–1440)
  max_sessions: 10000                    # LRU eviction cap (100–1,000,000)
  history_depth: 5                       # Requests retained per session (2–50)

  thresholds:
    warn_confidence: 0.30                # Must be strictly ascending
    throttle_confidence: 0.50
    inject_confidence: 0.70
    hardstop_confidence: 0.90

  consecutive_counts:
    warn: 3                              # Must be non-decreasing
    throttle: 5
    inject: 7
    hardstop: 10

  weights:                               # Must sum to 1.0 (±0.001)
    content_similarity: 0.25
    tool_call_repetition: 0.20
    response_stagnation: 0.15
    token_velocity: 0.10
    error_cycling: 0.15
    context_growth: 0.10
    cost_velocity: 0.05

  throttle_delay_seconds: 2              # Delay at Throttle level (1–30)
  injection_strategy: system_prompt_append  # or context_aware
  ema_alpha: 0.3                         # EMA smoothing factor (0.01–1.0)
  eviction_interval_seconds: 60          # How often the eviction loop runs (10–3600)
  token_velocity_threshold: 10000.0      # Tokens/min before signal fires
  cost_velocity_threshold: 0.5           # USD/min before signal fires
  # break_instruction_template: "..."    # Custom injection text (optional)
```

### Validation Rules

- `thresholds` must be strictly ascending: warn < throttle < inject < hardstop
- `consecutive_counts` must be non-decreasing: warn ≤ throttle ≤ inject ≤ hardstop
- `weights` must each be in `[0.0, 1.0]` and sum to `1.0`
- All numeric fields have documented min/max ranges

If validation fails at startup or hot-reload, the gateway logs the specific error(s) and rejects the config change.

---

## Per-Virtual-Key Overrides

Each virtual key can carry a `loop_detection` override block that merges with the global config. Only specified fields are overridden; the rest fall through to global defaults.

```json
{
  "name": "aggressive-agent",
  "loop_detection": {
    "thresholds": {
      "warn_confidence": 0.20,
      "throttle_confidence": 0.40,
      "inject_confidence": 0.60,
      "hardstop_confidence": 0.80
    },
    "consecutive_counts": {
      "warn": 2,
      "throttle": 3,
      "inject": 5,
      "hardstop": 7
    },
    "injection_strategy": "context_aware",
    "break_instruction_template": "You are stuck in a loop. Stop and ask the user for guidance."
  }
}
```

This allows different sensitivity levels for different callers — tighter for expensive or untrusted agents, looser for batch workloads that legitimately repeat similar requests.

---

## Admin API

### List Sessions

```
GET /admin/loop-detection/sessions?limit=50&offset=0
```

Returns paginated session summaries including session ID, virtual key, request count, current confidence, enforcement level, dominant signal, and last activity timestamp.

### Session Detail

```
GET /admin/loop-detection/sessions/{session_id}
```

Returns full session state:
- Current and peak confidence
- Enforcement level and escalation timeline (with timestamps)
- All signal values from recent history
- Recent request hashes and tool-call fingerprints
- Response descriptors (token count, structure hash, error flag)
- Total tokens consumed and estimated cost

### Reset Session

```
POST /admin/loop-detection/sessions/{session_id}/reset
```

Resets enforcement state, confidence, and history for the session. The session continues to exist (not deleted) but starts fresh. Useful for false-positive recovery or after manual intervention.

### Aggregate Stats

```
GET /admin/loop-detection/stats
```

Returns:
- Total active sessions
- Enforcement level distribution (how many sessions are at each level)
- Signal distribution (which signals are dominant across all sessions)
- Average confidence across all sessions
- Top 10 highest-confidence sessions
- Estimated memory usage
- Eviction count and rate

---

## Observability

### Prometheus Metrics

| Metric | Type | Labels | Description |
|--------|------|--------|-------------|
| `obey_loop_confidence_score` | histogram | `virtual_key` | Per-virtual-key confidence distribution (buckets: 0.1–1.0) |
| `obey_loop_enforcement_total` | counter | `level`, `virtual_key` | Enforcement level transitions |
| `obey_loop_sessions_active` | gauge | — | Current number of tracked sessions |
| `obey_loop_sessions_evicted_total` | counter | — | Total sessions removed by LRU eviction |

### Logging

| Event | Level | Contents |
|-------|-------|----------|
| Enforcement transition | INFO | Session ID, virtual key, from/to levels, confidence, all 7 signal values, consecutive count, request ID, timestamp |
| Hard-stop | ERROR | Full session state dump including recent hashes, tool fingerprints, response descriptors, escalation history, total tokens, and cost |

No request content or provider responses are logged — only structural metadata and signal values.

---

## Response Headers

| Header | When | Format |
|--------|------|--------|
| `x-loop-warning` | Enforcement ≥ Warn | `{confidence:.3}; dominant_signal={signal_name}` |

Example: `x-loop-warning: 0.672; dominant_signal=tool_call_repetition`

Clients can use this header to implement their own loop-breaking logic upstream.

---

## Hard-Stop Response

When enforcement reaches Hard-Stop, the request is rejected immediately:

**Status:** `429 Too Many Requests`  
**Headers:** `Retry-After: 60`, `Content-Type: application/json`

```json
{
  "error": {
    "reason": "loop_detected",
    "session_id": "abc123...",
    "confidence": 0.95,
    "dominant_signal": "tool_call_repetition",
    "enforcement_level": "hard_stop"
  }
}
```

The client should wait at least 60 seconds or change its approach before retrying. The session can also be manually reset via the admin API.

---

## Architecture

The loop detection system is implemented as a Tower `Layer`/`Service` middleware:

```
Request → LoopDetectorService
            ├── Session resolution (VK ID / header / IP)
            ├── Request fingerprinting (SimHash, tool-call hash)
            ├── Signal computation (7 signals from session history)
            ├── Confidence scoring (weighted sum + EMA)
            ├── Enforcement evaluation (threshold + consecutive check)
            ├── Optional injection (system prompt modification)
            ├── Optional throttle (sleep)
            ├── Hard-stop check (reject or forward)
            └── Forward to inner service
Response ← Record response descriptor for future stagnation detection
```

Key implementation details:
- **Session state** is stored in a concurrent `DashMap` with per-session mutex locks for consistency
- **SimHash** provides O(1) approximate content similarity without storing full request text
- **Tool-call fingerprinting** hashes function names + argument structure (not values) for repetition detection
- **Eviction** runs on a background Tokio task at configurable intervals
- **Hot-reload** clears all sessions when config changes (circuit breaker reset semantics)

---

## Troubleshooting

| Symptom | Cause | Fix |
|---------|-------|-----|
| False-positive hard-stops on legitimate batch work | Repetitive-but-valid requests look like loops | Raise thresholds, increase consecutive counts, or use per-VK overrides with looser settings |
| Loop not detected fast enough | Default consecutive counts are too high for your use case | Lower `consecutive_counts` (e.g., warn: 2, throttle: 3, inject: 4, hardstop: 6) |
| `x-loop-warning` but no escalation | Confidence hovers near threshold without staying above for enough consecutive requests | Lower the relevant threshold or increase `ema_alpha` for faster response |
| High memory usage from sessions | Too many concurrent callers tracked | Lower `max_sessions` or reduce `session_timeout_minutes` |
| Injection not changing agent behavior | Generic instruction insufficient | Switch to `context_aware` strategy or provide a custom `break_instruction_template` |
