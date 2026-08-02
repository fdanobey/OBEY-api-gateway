# Tool Definition Compression

Tool Definition Compression reduces token usage from tool/function definitions that are sent with every API request. When applications register dozens or hundreds of tools (common with MCP servers and agentic frameworks), the definitions themselves can consume thousands of tokens per request.

The compression pipeline applies multiple strategies in sequence to minimize token overhead while preserving model comprehension of available tools.

---

## How It Works

```
┌─────────────────────────────────────────────────────────────────────┐
│                    Tool Compression Pipeline                         │
│                                                                     │
│  Input tools ──► Minifier ──► Truncator ──► Semantic Retriever     │
│                                                    │                │
│  Output tools ◄── Canonical Rewriter ◄── Cache Placement           │
│                         ◄── Namespace Grouper ◄── Pruner           │
│                                   ◄── Deduplicator ◄───────────────┘
└─────────────────────────────────────────────────────────────────────┘
```

Each stage runs in fixed order, and the feedback loop monitors downstream error rates to automatically adjust compression aggressiveness.

---

## Admin Panel

The Compression tab in the admin panel includes a Tool Compression section at the bottom:

![Tool Compression Admin](images/admin-tool-compression.png)

---

## Configuration

```yaml
tool_compression:
  enabled: true
  level: medium              # low | medium | high | max
  progressive_disclosure: false

  pruning:
    enabled: false
    min_requests: 5          # Session requests before pruning activates
    always_include:          # Never-prune patterns (exact or glob)
      - "read_file"
      - "mcp_*"

  cache_placement: true      # Reorder tools for prompt-cache hits
  deduplication: true        # Merge identical parameter schemas

  minification:
    remove_titles: true
    collapse_single_unions: true
    remove_additional_properties: true
    remove_empty_descriptions: true

  description_truncation:
    tool_level: first_sentence   # none | first_sentence | remove
    parameter_level: remove      # none | remove
    remove_examples: true
    min_preserve_length: 20

  semantic_retrieval:
    enabled: false
    embedding_model: builtin-minilm
    top_k: 20
    similarity_threshold: 0.3
    frequency_weight: 0.3

  canonical_rewriting:
    enabled: false
    allowed_models: []

  feedback_loop:
    enabled: true
    error_threshold: 0.10    # Error rate that triggers level reduction
    recovery_window: 50      # Low-error requests before level increase
    rolling_window: 100

  auto_tuning:
    enabled: true
    model_tiers: {}          # Model glob → tier level (1-3)

  namespace_grouping:
    enabled: false
    min_tools_for_grouping: 10
    namespace_mappings: {}

  precomputed_descriptions:
    enabled: false
    method: tfidf            # tfidf | manual | model
    descriptions: {}         # tool_name → compressed text

  model_group_overrides: {}  # Per-group partial overrides
  provider_overrides: {}     # Per-provider capability settings
  debug_validation: false
```

---

## Compression Levels

| Level | Behavior |
|-------|----------|
| **Low** | Minification + deduplication only |
| **Medium** | Adds description truncation and cache placement |
| **High** | Adds namespace grouping, semantic retrieval, pruning |
| **Max** | All stages at maximum aggressiveness |

The feedback loop can downgrade levels automatically when error rates increase.

---

## Pipeline Stages

### 1. Schema Minifier

Removes structural noise from JSON Schema definitions:
- Strips `title` fields
- Collapses single-item `anyOf`/`oneOf` unions
- Removes `additionalProperties: false` (model default)
- Removes empty `description: ""` fields

### 2. Description Truncator

Shortens verbose tool and parameter descriptions:
- **Tool-level:** Keep first sentence, remove entirely, or leave unchanged
- **Parameter-level:** Remove descriptions or leave unchanged
- **Examples:** Strip "Example: ..." and "e.g. ..." suffixes

### 3. Semantic Retriever

Selects only the most relevant tools based on the current message:
- Uses TF-IDF (default) or embedding similarity
- Hybrid scoring: `(1 - frequency_weight) * semantic + frequency_weight * usage_frequency`
- Only tools scoring above `similarity_threshold` are included

### 4. Schema Deduplicator

Identifies tools with identical parameter schemas and consolidates references to save repeated JSON.

### 5. Tool Pruner

Removes tools that have never been called in the current session after `min_requests` interactions. Protected tools matching `always_include` patterns are never pruned.

### 6. Namespace Grouper

Groups tools by prefix (e.g., `mcp_github_*`, `mcp_jira_*`) into namespace summaries, reducing the total tool count presented to the model.

### 7. Cache Placement Optimizer

Reorders tool definitions to maximize prompt-cache hit rates by placing stable (unchanged between requests) tools at the beginning.

### 8. Canonical Rewriter

For supported models, rewrites tool definitions into a compact canonical text format instead of verbose JSON Schema.

---

## Feedback Loop

The feedback loop monitors tool-call error rates per model group:

1. **Error detection:** Tracks recent requests in a rolling window
2. **Downgrade:** When errors exceed `error_threshold`, reduces compression level
3. **Recovery:** After `recovery_window` consecutive low-error requests, attempts level increase
4. **Floor:** Never downgrades below `Low`

This ensures aggressive compression doesn't break tool calling for specific models.

---

## Auto-Tuning

When enabled, auto-tuning assigns compression levels based on model capability tiers:

```yaml
auto_tuning:
  enabled: true
  model_tiers:
    "gpt-4*": 3        # Tier 3 → can handle Max compression
    "claude-*": 3
    "gpt-3.5*": 1      # Tier 1 → only Low compression
```

Models not matching any pattern use the global `level` setting.

---

## Per-Model-Group Overrides

```yaml
tool_compression:
  enabled: true
  level: medium

  model_group_overrides:
    coding-group:
      level: high
      pruning:
        enabled: true
        min_requests: 3
    simple-group:
      enabled: false
```

---

## Dashboard

The Compression tab in the dashboard shows live tool compression metrics:

![Dashboard Compression](images/dashboard-compression.png)

Metrics include:
- **Avg Reduction Ratio** — percentage of tokens saved
- **Total Tokens Saved** — cumulative savings
- **Requests Compressed** — how many requests passed through the pipeline
- **Tools Pruned** — tools removed by the pruner
- **Token Savings (last 24h)** — timeline chart
- **Feedback Level per Model Group** — current auto-adjusted levels
- **Pruning Activity** — tracked keys, sessions with pruned tools
- **Progressive Disclosure Activity** — active sessions, disclosed tools

### Test Compression

The dashboard includes a "Test Compression" section where you can paste a JSON tools array and preview the compressed output with the current configuration.

---

## Prometheus Metrics

```
obey_tool_compression_tokens_saved_total{model_group}
obey_tool_compression_requests_total{model_group}
obey_tool_compression_ratio{model_group}
obey_tool_compression_tools_pruned{model_group}
obey_tool_compression_feedback_level{model_group}
obey_tool_compression_feedback_error_rate{model_group}
obey_tool_compression_feedback_adjustments_total{model_group}
```

---

## Next Steps

- [Token Compression](Token-Compression) — message-level compression (different from tool compression)
- [Configuration](Configuration) — full config reference
- [Admin Panel & Dashboard](Admin-Panel-and-Dashboard) — web UIs
