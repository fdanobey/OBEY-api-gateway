# Token Compression

OBEY API Gateway can transparently compress request payloads before forwarding them to providers, reducing token consumption and cost — especially for applications with large conversation histories or repetitive content.

Compression is **opt-in and disabled by default**. Enable it only after selecting and testing a level that suits your traffic.

---

## Compression Engines

The gateway ships with 9 compression engines. Named levels resolve to ordered engine chains; `tool_def`, `language_pack`, and `perplexity` are standalone engines that you compose through [custom pipelines](#custom-pipelines).

| Level / Engine | Use Case | Trade-off |
|----------------|----------|-----------|
| `lite` | Minimal compression; best for single-turn requests | Lowest ratio; preserves readability |
| `standard` | Balanced compression; default for multi-turn chats | Good ratio + meaning retention |
| `aggressive` | High compression; removes structure and whitespace | Medium ratio; may lose formatting context |
| `ultra` | Maximum compression; removes all non-essential tokens | High ratio; risky for complex logic/code |
| `rtk` | Round-trip-knowledge; preserves semantics for cached responses | High ratio with semantic safety |
| `stacked` | Layered; applies RTK + standard sequentially | Highest ratio; slowest |
| `tool_def` | Optimized for tool/function definitions | Compresses JSON schemas and callable signatures |
| `language_pack` | Language-aware; detects dominant language | Respects linguistic boundaries |
| `perplexity` | Perplexity-model optimized for long-context requests | Requires an external ONNX scorer model |

---

## Admin UI

Open the admin panel and select the **Compression** tab. The page opens with compression disabled and every field showing its default value.

![Token Compression settings](images/admin-compression.png)

The page is organized into sections:

| Section | Purpose |
|---------|---------|
| **Global Behavior** | Master enable switch, default level, auto-threshold, language, and output flags |
| **Time Budgets** | Per-engine processing budget in milliseconds |
| **Protection Rules** | Content structures that are never compressed |
| **RTK** | Grouping strategy for the round-trip-knowledge engine |
| **Perplexity / ONNX** | Optional perplexity engine and its external model settings |
| **Precompressed Contexts** | Map source contexts to precompressed artifacts |
| **Custom Pipelines** | Named, ordered engine chains selectable per request |

---

## UI Setup

Follow these steps to enable compression from the admin panel.

1. **Enable globally** — tick **Enable token compression globally** under *Global Behavior*.
2. **Pick a default level** — choose from `None`, `Lite`, `Standard`, `Aggressive`, `Ultra`, `RTK`, or `Stacked`. `Standard` is a good balanced starting point for multi-turn conversations.
3. **Set an automatic threshold (optional)** — leave `0` to compress only when a request explicitly opts in, or enter a positive token count to auto-compress requests above that size.
4. **Adjust time budgets (optional)** — each enabled engine needs a positive millisecond budget. Heavier levels (`aggressive`, `ultra`, `rtk`, `stacked`) default to `2000` ms; the light levels default to `500` ms.
5. **Review protection rules** — keep structures such as code blocks, URLs, file paths, JSON, identifiers, math expressions, and tool definitions checked so they are preserved verbatim.
6. **Add a custom pipeline (optional)** — click **+ Add Pipeline**, give it a name, and enter ordered, comma-separated engines (for example `rtk, standard, lite`).
7. **Save** — click **Save**. Changes apply via hot reload; no restart is required.

The screenshot below shows compression enabled with the `Standard` default level and a custom `terminal_then_prose` pipeline defined:

![Token Compression configured](images/admin-compression-configured.png)

> **Perplexity / ONNX:** Enabling the perplexity engine also requires an external ONNX model at the configured path and a compatible ONNX runtime in the gateway environment. The admin panel does not install either dependency.

---

## Configuration

The admin UI writes to the `compression` block in your config file. You can also edit it directly:

```yaml
compression:
  enabled: false                         # Enable compression globally (default: false)
  default_level: lite                    # Default level (default: lite)
  auto_threshold_tokens: 0               # Auto-trigger threshold; 0 = disabled
  caveman_output: false                  # Collapse output to extreme brevity
  compress_tool_definitions: false       # Compress tool/function definitions
  language: en                           # Language for the language_pack engine
  language_packs_dir: ./language_packs   # Directory for language pack files
  time_budget_ms:                        # Per-level time budgets (milliseconds)
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
  precompressed_contexts: []             # Pre-compressed file mappings
  rtk:
    grouping_strategy: balanced          # aggressive | balanced | conservative
  perplexity:
    enabled: false                       # Requires an ONNX perplexity scorer model
    redundancy_threshold: 0.5            # 0.0–1.0 inclusive
    compression_ratio_target: 5          # 1–20 inclusive
    model_path: ./models/perplexity_scorer.onnx
  custom_pipelines: {}                   # Named custom engine chains
```

Provider and model-group compression blocks override these defaults field-by-field. Setting `level: none` on a provider or model group explicitly disables compression for that scope.

---

## Custom Pipelines

Custom pipelines chain any combination of the 9 engines in a specified order:

```yaml
compression:
  custom_pipelines:
    terminal_then_prose:
      engines: [rtk, standard, lite]
```

Callers select a custom pipeline by name via the `x-compression-pipeline` request header.

---

## Cache-Aware Downgrade

When a provider supports prompt caching (for example Anthropic Claude) and a request contains `cache_control` markers, the gateway preserves the cached prefix byte-for-byte. The `aggressive`, `ultra`, `rtk`, and `stacked` levels are downgraded to `none` for the protected prefix while still compressing the suffix, so cache hits are not invalidated.

---

## Observability

Compression execution is exported via Prometheus metrics on the `/metrics` endpoint:

| Metric | Type | Description |
|--------|------|-------------|
| `obey_compression_tokens_saved_total` | counter | Total tokens saved (by level and provider) |
| `obey_compression_ratio` | histogram | Compressed token ratio (compressed / original) |
| `obey_compression_duration_seconds` | histogram | Compression operation duration in seconds |

---

## Next Steps

- [Caching](Caching) — how compression interacts with prompt caching
- [Configuration](Configuration) — full config reference
- [Admin Panel & Dashboard](Admin-Panel-and-Dashboard) — hot reload and metrics
