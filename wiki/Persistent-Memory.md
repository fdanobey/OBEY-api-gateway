# Persistent Memory

Persistent Memory is a cross-session memory system that extracts facts, preferences, and decisions from conversations and injects them into future requests. It enables models to recall context from previous interactions without the caller managing memory explicitly.

---

## How It Works

```
┌──────────────────────────────────────────────────────────────────┐
│                    Memory Lifecycle                               │
│                                                                  │
│  ┌─────────┐    ┌───────────┐    ┌──────────┐                   │
│  │ Inject  │───▶│  Request  │───▶│ Provider │                   │
│  │memories │    │           │    │          │                   │
│  └─────────┘    └───────────┘    └──────────┘                   │
│       ▲                                │                         │
│       │                                ▼                         │
│  ┌─────────┐                     ┌──────────┐                   │
│  │  Store  │◀────────────────────│ Extract  │                   │
│  │(SQLite) │                     │ memories │                   │
│  └─────────┘                     └──────────┘                   │
│       │                                                          │
│       ▼                                                          │
│  ┌─────────┐    ┌───────────┐                                   │
│  │  Decay  │───▶│  Evict    │                                   │
│  │schedule │    │ low-score │                                   │
│  └─────────┘    └───────────┘                                   │
└──────────────────────────────────────────────────────────────────┘
```

1. **Injection:** Before each request, relevant memories are retrieved and injected
2. **Processing:** The request is sent to the provider as normal
3. **Extraction:** After the response, facts/preferences/decisions are extracted
4. **Storage:** New memories are persisted to the SQLite database
5. **Decay:** Periodically, a decay schedule reduces relevance scores
6. **Eviction:** Low-scoring entries are evicted when namespace limits are reached

---

## Admin Panel

![Persistent Memory Admin](images/admin-memory.png)

The Persistent Memory tab provides:
- **General settings** — enable/disable, database path, injection strategy, token limits
- **Decay and limits** — schedule frequency, max entries per namespace
- **Sensitive content** — allow/block PII-like content
- **Automatic Extraction** — provider/model for extracting memories from conversations
- **Vector Search (Qdrant)** — optional semantic retrieval alongside BM25 lexical search
- **Memory Store — Live Stats** — entry count, namespaces, avg relevance, storage size
- **Memory Browser** — list entries by namespace, clear namespaces
- **Create Memory Entry** — manually add facts, preferences, or corrections
- **Detected Projects** — auto-discovered project namespace scopes

---

## Configuration

```yaml
memory:
  enabled: true
  database_path: ./memory.db
  injection_strategy: system_prompt_prefix   # or synthetic_message
  max_injection_tokens: 500
  auto_extract_enabled: false
  auto_extract_provider: openai
  auto_extract_model: gpt-4.1-mini
  auto_extract_min_turns: 4
  decay_schedule_hours: 24
  max_memories_per_namespace: 1000
  allow_sensitive_storage: false
  show_feedback: true
  default_prompts: []
  custom_sensitive_patterns: []
  qdrant:
    qdrant_url: https://qdrant.example.com:6333
    qdrant_collection: obey_memories
    similarity_threshold: 0.7
    embedding_provider: openai
    embedding_model: text-embedding-3-small
    fts_weight: 0.4
    vector_weight: 0.6
```

### Fields

| Field | Default | Description |
|-------|---------|-------------|
| `enabled` | `false` | Enable persistent memory globally |
| `database_path` | `./memory.db` | SQLite file for memory persistence |
| `injection_strategy` | `system_prompt_prefix` | How memories are placed: `system_prompt_prefix` or `synthetic_message` |
| `max_injection_tokens` | `500` | Token budget for injected memories (0–10000) |
| `auto_extract_enabled` | `false` | Automatically extract memories from conversations |
| `auto_extract_provider` | — | Provider to use for extraction (must be configured) |
| `auto_extract_model` | — | Model to use for extraction |
| `auto_extract_min_turns` | `4` | Minimum conversation turns before triggering extraction (1–100) |
| `decay_schedule_hours` | `24` | How often relevance decay runs (1–8760) |
| `max_memories_per_namespace` | `1000` | Maximum entries per namespace before eviction (1–100000) |
| `allow_sensitive_storage` | `false` | When disabled, PII-like content is rejected |
| `show_feedback` | `true` | Include memory metadata in response headers |
| `default_prompts` | `[]` | System prompts that identify the default assistant context |
| `custom_sensitive_patterns` | `[]` | Additional regex patterns for sensitive content detection |

---

## Injection Strategies

### System Prompt Prefix

Memories are prepended to the system message:

```
[Recalled context from previous conversations]
- User prefers TypeScript over JavaScript
- Project uses Next.js 15 with App Router
- Database is PostgreSQL on Supabase

[Original system prompt follows...]
You are a helpful coding assistant...
```

### Synthetic Message

A separate `system` message containing memories is inserted before the conversation:

```json
{
  "role": "system",
  "content": "[Memory context]\n- User prefers TypeScript..."
}
```

---

## Namespaces

Memories are scoped by namespace, which is derived from:
- **Virtual Key ID** — isolates memories per caller
- **Context type** — detected from the conversation (project, agent, user)

Namespace format: `{vk_scope}::{context_kind}::{context_id}`

Example: `user_abc123::project::my-web-app`

---

## Memory Types

| Type | Description |
|------|-------------|
| **Fact** | Objective information ("uses PostgreSQL 16") |
| **Preference** | User preferences ("prefers functional style") |
| **Decision** | Decisions made ("chose Tailwind over styled-components") |
| **Correction** | Corrections to model behavior ("don't suggest jQuery") |

---

## Retrieval and Scoring

When injecting memories, the system retrieves and ranks them:

1. **Lexical (BM25):** Full-text search against the current message
2. **Vector (Qdrant):** Semantic similarity when Qdrant is configured
3. **Hybrid scoring:** `fts_weight * bm25_score + vector_weight * cosine_similarity`
4. **Token budget:** Top-scoring memories are included up to `max_injection_tokens`

---

## Relevance Decay

Memories that are not accessed decay over time:
- Every `decay_schedule_hours`, unused memory scores are reduced
- High-frequency accessed memories retain their scores
- When entries exceed `max_memories_per_namespace`, lowest-scoring entries are evicted

---

## Vector Search (Qdrant)

Optional Qdrant integration provides semantic retrieval:

```yaml
memory:
  qdrant:
    qdrant_url: http://localhost:6333
    qdrant_collection: obey_memories
    similarity_threshold: 0.7
    embedding_provider: openai
    embedding_model: text-embedding-3-small
    fts_weight: 0.4
    vector_weight: 0.6
```

When configured, memories are embedded and stored in Qdrant for semantic similarity search. The hybrid scorer combines BM25 lexical matches with vector similarity.

---

## Per-Provider and Per-Model-Group Overrides

```yaml
providers:
  - name: openai
    memory:
      enabled: true
      max_injection_tokens: 750

model_groups:
  - name: coding-group
    memory:
      enabled: true
      injection_strategy: synthetic_message
      max_injection_tokens: 1000
      show_feedback: false
    models:
      - provider: openai
        model: gpt-4.1
```

Precedence: model-group override > provider override > global config.

---

## Admin API Endpoints

```bash
# List entries in a namespace
curl "http://localhost:8080/admin/memory/entries?namespace=user::project::myapp"

# Create a memory entry
curl -X POST http://localhost:8080/admin/memory/entries \
  -H 'Content-Type: application/json' \
  -d '{"namespace": "user::project::myapp", "memory_type": "fact", "content": "Uses React 19"}'

# Delete a specific entry
curl -X DELETE http://localhost:8080/admin/memory/entries/{id}

# Clear all entries in a namespace
curl -X DELETE http://localhost:8080/admin/memory/namespaces/{namespace}

# Memory store statistics
curl http://localhost:8080/admin/memory/stats

# List detected project namespaces
curl http://localhost:8080/admin/memory/projects
```

---

## Dashboard

The Memory tab in the dashboard shows real-time memory system activity:

![Dashboard Memory](images/dashboard-memory.png)

Metrics include:
- **Total Events** — all memory operations
- **Injections** — memories recalled and injected into requests
- **Extractions** — new memories extracted from conversations
- **Evictions** — entries removed by decay/limits
- **Memory Events Timeline** — injection/extraction/eviction activity over time
- **Event Type Distribution** — pie chart of operation types
- **Namespace Activity** — per-namespace operation breakdown

---

## Sensitive Content Protection

When `allow_sensitive_storage: false` (default), the system rejects content matching:
- Email addresses
- Phone numbers
- Social security numbers
- Credit card numbers
- API keys and tokens
- Custom patterns from `custom_sensitive_patterns`

---

## Docker Considerations

The SQLite database at `database_path` must persist across container restarts:

```yaml
volumes:
  - ai-gateway-data:/data

# Set database_path to /data/memory.db
```

---

## Next Steps

- [Virtual Keys](Virtual-Keys) — namespace isolation via virtual keys
- [Configuration](Configuration) — full config reference
- [Admin Panel & Dashboard](Admin-Panel-and-Dashboard) — web UIs
