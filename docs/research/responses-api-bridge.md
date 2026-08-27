# Research: Implementing the OpenAI Responses API (`/v1/responses`) on top of Chat Completions

**Goal:** Bridge an incoming OpenAI Responses API request to backend `/chat/completions`
endpoints (and synthesize Responses-shaped output back), so clients that hardcode the
Responses wire format (Codex CLI, OpenAI SDK `responses.create`) work against providers that
only expose Chat Completions (or against our own router, which is Chat-Completions-native).

**Primary sources examined**
- LiteLLM — `litellm/responses/litellm_completion_transformation/transformation.py` (Responses→Chat bridge), `streaming_iterator.py`, `session_handler.py`.
- OpenAI Agents SDK (Python) — `src/agents/models/chatcmpl_converter.py` (`Converter.items_to_messages` / `message_to_output_items`), `openai_chatcompletions.py`.
- OpenAI Node SDK — `src/resources/responses/responses.ts`.
- Our own codebase — `crates/ai-gateway/src/codex/translate_request.rs` (`ChatToResponsesTranslator`, `responses_to_chat`) and `crates/ai-gateway/src/providers/bedrock.rs` (`mantle_responses_input`, `chat_completion_responses_api`).

> Note on direction: our existing helpers (`ChatToResponsesTranslator`, `mantle_responses_input`)
> translate **Chat Completions → Responses** (because Bedrock Mantle and the Codex backend
> *consume* the Responses shape). The gap this research targets is the **inverse bridge**:
> a `/v1/responses` entrypoint that accepts the Responses shape and fans out to Chat
> Completions backends (or to our router). The patterns below are the inverse of what we
> already have, and reuse the same `Message` / `OpenAIRequest` types.

---

## 1. Translating a Responses request → Chat Completions request

### Core shape (LiteLLM)
`LiteLLMCompletionResponsesConfig.transform_responses_api_request_to_chat_completion_request`
builds a dict with these direct field maps:

| Responses field | Chat Completions field |
|---|---|
| `input` (+ `instructions`) | `messages` |
| `max_output_tokens` | `max_tokens` |
| `tools` | `tools` (with reshaping, see §5) |
| `tool_choice` | `tool_choice` |
| `text.format` | `response_format` (see §7) |
| `temperature`, `top_p`, `user`, `parallel_tool_calls` | same |
| `reasoning.effort` | `reasoning_effort` (string) or full dict when `summary` set |

It then drops all `None` values: `litellm_completion_request = {k: v for k, v in ... if v is not None}`.

For streaming it injects `stream_options: { include_usage: true }` because Responses
`response.completed` always carries usage. **Adaptation:** when `stream: true` we must send
`stream_options.include_usage=true` to the backend so we can synthesize usage in the
`response.completed` event.

### The `input` → `messages` transform
`transform_responses_api_input_to_messages`:
1. If `instructions` present, prepend a system message (see §4).
2. Run every input item through `_transform_response_input_param_to_chat_completion_message`.

A string `input` becomes a single `user` message. An array is mapped item-by-item (§5).

**Rust sketch (mirrors our `Message { role, content, extra }`):**

```rust
fn responses_input_to_messages(
    input: &Value,
    instructions: Option<&str>,
) -> Vec<Message> {
    let mut messages = Vec::new();
    if let Some(sys) = instructions {
        // §4 — instructions become the system message
        messages.push(Message { role: "system".into(), content: json!(sys), extra: Map::new() });
    }
    match input {
        Value::String(s) => messages.push(Message {
            role: "user".into(), content: json!(s), extra: Map::new(),
        }),
        Value::Array(items) => {
            // §5 — walk items, attaching tool_calls to the active assistant message
            // and emitting `role: "tool"` messages for function_call_output.
            for item in items { push_item_as_message(item, &mut messages); }
        }
        _ => {}
    }
    messages
}
```

Our existing `responses_to_chat` (test-only, `codex/translate_request.rs:1305`) already does the
simple `type: "message"` branch (concatenates `input_text`/`output_text`). It needs to be
promoted to a real translator covering `function_call` / `function_call_output` / `reasoning`.

---

## 2. Synthesizing a Responses response from a Chat Completions response

### LiteLLM (`transform_chat_completion_response_to_responses_api_response`)
- Wraps the chat `ModelResponse` into a `ResponsesAPIResponse` with `object: "response"`.
- Derives overall `status` from `choices[0].finish_reason` (e.g. `length` → `incomplete`).
- `output` is built from `choices[0].message`: a `message` item (role `assistant`,
  `content: [{type:"output_text", text}]`) plus one `function_call` item per `tool_call`.
- `usage` is copied through (`prompt_tokens`, `completion_tokens`, `total_tokens`).

### OpenAI Agents SDK (`Converter.message_to_output_items`)
The cleanest reference for the inverse mapping:

```python
# ChatCompletionMessage -> list[ResponseOutputItem]
if message.content:
    message_item.content.append(ResponseOutputText(
        text=message.content, type="output_text", annotations=..., logprobs=[]))
if message.refusal:
    message_item.content.append(ResponseOutputRefusal(...))
if message.tool_calls:
    for tc in message.tool_calls:
        if tc.type == "function":
            items.append(ResponseFunctionToolCall(
                id=FAKE_RESPONSES_ID, call_id=tc.id,
                arguments=tc.function.arguments, name=tc.function.name,
                type="function_call"))
```

**Rust target shape (what we must emit to the client):**
```jsonc
{
  "id": "resp_...", "object": "response", "created_at": 1741290958,
  "model": "<model>", "status": "completed",
  "output": [
    { "type": "message", "role": "assistant", "status": "completed",
      "content": [ { "type": "output_text", "text": "...", "annotations": [] } ] }
    // + { "type": "function_call", "call_id": "...", "name": "...", "arguments": "{}" }
  ],
  "usage": { "input_tokens": N, "output_tokens": M, "total_tokens": T },
  "text": { "format": { "type": "text" } }
}
```

To match `response.output_text` semantics (the SDK aggregates all `output_text`), we should
also expose `output_text` (concatenation of all `output[].content[].text`) on the response
object for convenience, mirroring `openai-node` `addOutputText`.

---

## 3. `previous_response_id` conversation continuity

### Two viable strategies
**A. Stateless re-send (LiteLLM proxy, our likely fit):** On each request, fetch the prior
response's stored input+output and prepend them as chat messages, then append the new input.

LiteLLM `ResponsesSessionHandler.get_chat_completion_message_history_for_previous_response_id`:
1. `get_all_spend_logs_for_previous_response_id` — SQL over spend logs:
   `SELECT session_id FROM LiteLLM_SpendLogs WHERE request_id = $1` then
   `SELECT * FROM LiteLLM_SpendLogs WHERE session_id IN (...) ORDER BY endTime ASC`.
2. For each log: re-run `transform_responses_api_input_to_messages(input)` to rebuild input
   messages, and append `choice.message` (the assistant output) from the stored response.
3. Concatenate `session_messages + new_messages`, then fix up tool-call pairing
   (`_ensure_tool_results_have_corresponding_tool_calls`) so every `tool` message has a
   preceding assistant `tool_calls` entry.
4. `instructions` is **NOT** carried over — the caller must resend `instructions` each turn
   (OpenAI's own rule). This matches our router: `instructions` is a top-level request field.

**B. Server-managed (OpenAI native):** OpenAI stores the chain and bills all prior input
tokens again. `previous_response_id` is *not* an instruction-carrier.

**Adaptation for our gateway:** we already persist request/response pairs (logs.db). Reuse
that store: decode `previous_response_id` → look up the prior request's `input` + stored
`output`, translate both to chat `messages`, and prepend. Provide a `get_responses` endpoint
(`GET /v1/responses/{id}`) backed by the same store (LiteLLM implements `aget_responses`).

> OpenAI migration note: "Even when using `previous_response_id`, all previous input tokens
> for responses in the chain are still billed as input tokens." So continuity is a
> convenience, not a cost optimization.

---

## 4. Mapping `instructions` → system message

- LiteLLM: `if responses_api_request.get("instructions"): messages.append(transform_instructions_to_system_message(instructions))`. A string `instructions` → one `system` message.
- OpenAI Agents SDK: `system_instructions` is inserted as the first chat message
  (`converted_messages.insert(0, {"content": system_instructions, "role": "system"})`).
- Edge case (LiteLLM reverse path): if `input` ends up empty but `instructions` exists,
  carry `instructions` as a `system`-role *input item* instead, because the Responses API
  rejects an empty `input`.

**Rule:** `instructions` (when present) ⇒ leading `system` message, placed before all
`input`-derived messages. If both a `system` input item and `instructions` exist, LiteLLM
concatenates them with a space (reverse direction). For the forward (Responses→Chat) bridge,
treat `instructions` as the single authoritative system message; if a `system` input item is
also present, concatenate to avoid clobbering.

---

## 5. `function_call` and `function_call_output` input items

### Forward: Responses input items → Chat messages
Mirrors OpenAI Agents `Converter.items_to_messages` (the authoritative mapping):

- `type: "function_call"` (role-less input item): ensure an *assistant* message exists
  (`ensure_assistant_message`), then append a `tool_calls` entry:
  ```python
  ChatCompletionMessageFunctionToolCallParam(
      id=func_call["call_id"],   # NOTE: uses call_id as the chat tool_call id
      type="function",
      function={"name": func_call["name"], "arguments": func_call["arguments"] or "{}"})
  ```
- `type: "function_call_output"` (input item with `call_id` + `output`): emit a
  `role: "tool"` message keyed by `tool_call_id = func_output["call_id"]`, content = the
  output text. OpenAI Agents extracts only `text` parts (Chat Completions can't carry
  non-text tool results) and substitutes a `"[tool output omitted]"` placeholder when empty.
- Ordering matters: tool outputs must follow the assistant message that issued the call. The
  converter uses a `current_assistant_msg` accumulator and flushes it before a `tool`
  message (`flush_assistant_message()` before the `function_call_output` branch).
- `reasoning` items: typically **omitted** from chat history (most providers reject reasoning
  content in input); OpenAI Agents only replays reasoning when explicitly enabled.

### Reverse: Chat `tool_calls`/`tool` messages → Responses output/input
- `message.tool_calls` → `function_call` output items (`call_id = tool_call.id`,
  `name`, `arguments`). This is what `message_to_output_items` does and what we must emit so
  a client can loop: it sends back a `function_call_output` with the same `call_id`.
- `role: "tool"` messages → `function_call_output` input items (`call_id = tool_call_id`).

**Rust sketch for the forward direction:**
```rust
fn push_item_as_message(item: &Value, messages: &mut Vec<Message>, current_asst: &mut Option<usize>) {
    match item.get("type").and_then(|t| t.as_str()) {
        Some("message") => { /* role + content -> user/assistant/system message */ }
        Some("function_call") => {
            // ensure assistant message exists; append tool_call
            let id = item["call_id"].as_str().unwrap_or("").to_string();
            let args = item["arguments"].as_str().unwrap_or("{}").to_string();
            // attach as tool_calls on *current_asst (or create one)
        }
        Some("function_call_output") => {
            // flush assistant first
            messages.push(Message {
                role: "tool".into(),
                content: item["output"].clone(),
                extra: tool_call_id(item["call_id"].clone()),
            });
        }
        Some("reasoning") => { /* skip or replay per config */ }
        _ => {}
    }
}
```

LiteLLM's bridge additionally guards `_ensure_tool_results_have_corresponding_tool_calls`
when `previous_response_id` is used: if a `tool` message has a `tool_call_id` with no matching
assistant `tool_calls` (e.g. recovered from logs), it reconstructs the assistant call from
`tools` so the provider doesn't 400.

---

## 6. `max_output_tokens` ↔ `max_tokens`

- Forward: `max_output_tokens` → chat `max_tokens` (LiteLLM: `"max_tokens": responses_api_request.get("max_output_tokens")`).
- Reverse (LiteLLM `chat→responses`): `max_tokens` **and** `max_completion_tokens` both map to
  `max_output_tokens`:
  ```python
  if key in ("max_tokens", "max_completion_tokens"):
      responses_api_request["max_output_tokens"] = value
  ```
- Our `bedrock.rs` already does the forward side: `max_output_tokens: max_tokens.unwrap_or(2048)`.

**Rule:** treat `max_output_tokens` as the single source of truth; map 1:1 to `max_tokens`
when calling Chat Completions backends. Default to a sane cap (e.g. 2048) when absent, matching
bedrock's behavior, but prefer honoring the client's value exactly.

---

## 7. `text.format` → `response_format`

### Forward (Responses → Chat): LiteLLM `_transform_text_format_to_response_format`
- `text.format.type == "json_schema"` → chat
  `response_format = { "type": "json_schema", "json_schema": { "name", "strict", "schema" } }`.
- `text.format.type == "json_object"` → `{ "type": "json_object" }`.
- Otherwise `None` (plain text).

```python
def _transform_text_format_to_response_format(text_param):
    fmt = (text_param or {}).get("format") or {}
    t = fmt.get("type")
    if t == "json_schema":
        return {"type": "json_schema",
                "json_schema": {"name": fmt.get("name"),
                                "strict": fmt.get("strict", False),
                                "schema": fmt.get("schema")}}
    if t == "json_object":
        return {"type": "json_object"}
    return None
```

### Reverse (Chat → Responses): LiteLLM `_transform_response_format_to_text_format`
- chat `response_format.type == "json_schema"` → `text = { "format": { "type": "json_schema", ... } }`.
- chat `response_format.type == "json_object"` → `text = { "format": { "type": "json_object" } }`.

Our `codex/translate_request.rs::translate_response_format` (line 182) already implements the
forward chat→responses direction and is the model to mirror for the inverse. Structured
Outputs note: `strict: true` JSON schema is the Responses equivalent of Chat Completions
`response_format` with `strict`.

**Rule:** preserve `name`/`strict`/`schema` verbatim when translating between the two shapes;
default to no `response_format` (plain text) when `text.format.type == "text"` or absent.

---

## 8. Synthesizing Responses streaming events from Chat Completions SSE

This is the most involved piece. Two sub-problems: (a) emit the lifecycle events, (b) translate
each chat `delta` chunk into the typed Responses events.

### Lifecycle / synthetic events (LiteLLM `streaming_iterator.py`)
Given a completed `ResponsesAPIResponse`, build:
1. `response.created` — response with `status: "in_progress"`, `output: []`.
2. `response.in_progress`.
3. Per output item:
   - `response.output_item.added` (item with partial content)
   - `response.content_part.added`
   - `response.output_text.delta` (slice text into ~N-char chunks; LiteLLM `CHUNK_SIZE`)
   - `response.output_text.annotation.added` (if citations)
   - `response.output_text.done`
   - `response.content_part.done`
   - `response.output_item.done`
4. `response.completed` (full response incl. usage).

Non-streaming chat responses are "faked" into this same event sequence (LiteLLM
`MockResponsesAPIStreamingIterator` slices full text into 5-char deltas). We should do the
same for non-streaming backends so a client requesting `stream: true` gets a uniform SSE feed.

### Per-chunk delta translation (LiteLLM `_transform_chat_completion_chunk_to_response_api_chunk`)
Priority order per chunk:
1. **Annotations** (URL citations) → queue `output_text.annotation.added` events.
2. **Reasoning** (`delta.reasoning_content`) → `response.reasoning_summary_text.delta`.
3. **Text delta** (`delta.content`) → `response.output_text.delta` (with `item_id`,
   `output_index`, `content_index`, `sequence_number`).
4. **Tool-call deltas** (`delta.tool_calls`) → queue `response.function_call_arguments.delta`
   events, emitted one at a time (so each tool call streams its argument fragments in order).
5. **Pending annotation / tool events** drained when a chunk has no text.

```python
def _transform_chat_completion_chunk_to_response_api_chunk(self, chunk):
    if self._cached_item_id is None and chunk.id:
        self._cached_item_id = chunk.id
    item_id = self._cached_item_id or chunk.id
    # 1 annotations, 2 reasoning, 3 text, 4 tool_calls (queued), 5 drain pending
    if delta_content := self._get_delta_string_from_streaming_choices(chunk.choices):
        return OutputTextDeltaEvent(type="response.output_text.delta",
                                    item_id=item_id, output_index=0,
                                    content_index=0, delta=delta_content)
    ...
```

Final chunk (chat `finish_reason` / `[DONE]`): synthesize `response.completed` by running the
full chat response through `transform_chat_completion_response_to_responses_api_response`
(§2) and wrapping it.

### Adapting to our router
- Our SSE framing already exists (`providers/mod.rs` SSE helpers). Reuse it to emit
  `event: <type>\ndata: <json>\n\n`.
- The `item_id` should be stable across the stream (cache the first chunk's `id` or generate
  one like `msg_<uuid>`; LiteLLM uses `f"msg_{uuid.uuid4()}"`).
- Always emit `stream_options: { include_usage: true }` to the backend (§1) so the terminal
  `response.completed` carries usage.
- Vercel/OpenAEON event catalogs confirm the canonical event set:
  `response.created → in_progress → output_item.added → content_part.added →
  output_text.delta* → output_text.done → content_part.done → output_item.done →
  response.completed` (+ `response.failed` / `response.incomplete` on errors).

---

## Recommended module layout for our Rust implementation

Reuse the existing `Message`/`OpenAIRequest`/`OpenAIResponse` types and add a sibling bridge
module (e.g. `crates/ai-gateway/src/responses_bridge/`):

- `request.rs` — `responses_request_to_chat(request: Value) -> OpenAIRequest`:
  - `instructions` → leading system message (§4)
  - `input` → messages, handling `function_call`/`function_call_output`/`reasoning` (§5)
  - `max_output_tokens` → `max_tokens` (§6)
  - `text.format` → `response_format` (§7)
  - inject `stream_options.include_usage = true` when streaming (§1, §8)
- `response.rs` — `chat_response_to_responses(chat: OpenAIResponse) -> Value` (§2) and
  `synthesize_stream_events(chat_stream) -> impl Stream<ResponsesEvent>` (§8), plus
  `fake_stream_from_completed` for non-streaming backends.
- `session.rs` — `previous_response_id` resolver backed by our logs.db store (§3), reusing the
  spend-log replay approach (input→messages + stored output→assistant messages).
- `handlers.rs` — `POST /v1/responses` and `GET /v1/responses/{id}` entrypoints.

This is essentially the inverse of `ChatToResponsesTranslator` (codex) and
`mantle_responses_input` (bedrock), so we can share item-walking helpers between the two
directions.

**Key correctness rules to encode (from sources):**
1. `instructions` is never carried by `previous_response_id`; resend it every turn.
2. Every `function_call_output` must have a matching preceding assistant `function_call`
   (`call_id` parity); repair from `tools` when recovering from logs.
3. Empty `input` is rejected by the Responses API — fall back to a `system` input item when
   only `instructions` is present.
4. Tool outputs cannot be empty/non-text in Chat Completions — substitute a placeholder or
   restrict to `text`.
5. Streaming must always end with a `response.completed` carrying `usage`; emit
   `include_usage: true`.
6. `max_output_tokens` and `max_tokens`/`max_completion_tokens` are the same knob named
   differently per API.
