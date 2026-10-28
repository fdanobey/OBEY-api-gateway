//! Streaming + failover integration tests for the Anthropic Messages front
//! door (spec `anthropic-messages-front-door`, Task 9).
//!
//! Covers Requirement 2 (SSE event order, interleaving, ping keep-alive,
//! mid-stream error frames, pre-content failover transparency + attempt
//! ledger) and Requirement 4.3 (beta header passthrough).
//!
//! Harness (wiremock upstream, `common::isolate_databases`, `oneshot`);
//! SSE capture follows `request_outcome_logging.rs` / `pass_through_routes.rs`
//! (collect the response body to a string, parse `event:`/`data:` lines).
//!
//! DEVIATIONS (documented, Task 9):
//! 1. RESOLVED: `chat_deltas_from_chunk` (src/gateway/handlers.rs) now reads
//!    the delta from `choices/0/delta` (top-level `delta` kept as fallback),
//!    so the event-order tests below run un-ignored.
//! 2. Case 6 (Mantle beta-header passthrough) cannot be intercepted
//!    end-to-end: the router hardcodes the Mantle base URL
//!    (`build_mantle_base_url` → `https://bedrock-mantle.{region}.api.aws/v1`)
//!    for bedrock API-key providers, and `BedrockProvider`'s fields are
//!    private, so an integration test cannot point it at wiremock. The
//!    verbatim forwarding contract is already covered by the provider-level
//!    wiremock tests in `src/providers/bedrock.rs` (`mod
//!    anthropic_passthrough`). This suite pins the handler side of the
//!    chain: full beta capture (unknown values, comma-joined multi-header
//!    form) into the passthrough scope the Mantle adapter reads.

mod common;

use std::time::Duration;

use ai_gateway::anthropic::passthrough::{
    beta_header_value, sse_relay_frames, AnthropicPassthroughScope,
};
use ai_gateway::config::{Config, EnforcementMode, StreamingConfig};
use ai_gateway::gateway::GatewayServer;
use ai_gateway::logger::LogEntry;
use axum::body::Body;
use axum::http::{Request, StatusCode};
use serde_json::{json, Value};
use tower::ServiceExt;
use wiremock::matchers::{method, path};
use wiremock::{Mock, MockServer, ResponseTemplate};

const VERSION: &str = "2023-06-01";

// ---------------------------------------------------------------------------
// Harness (no cross-crate imports)
// ---------------------------------------------------------------------------

fn base_config() -> Config {
    let mut config: Config = serde_json::from_value(json!({
        "server": { "host": "127.0.0.1", "port": 0 },
        "providers": [
            { "name": "primary", "type": "openai", "base_url": "http://127.0.0.1:9/v1", "api_key": "test-key" }
        ],
        "model_groups": [{
            "name": "g",
            "models": [{ "provider": "primary", "model": "gpt-4", "priority": 1 }]
        }]
    }))
    .expect("base streaming config should deserialize");
    config.virtual_keys.enforcement = EnforcementMode::Disabled;
    config
}

fn two_provider_config(primary_url: String, fallback_url: String) -> Config {
    let mut config: Config = serde_json::from_value(json!({
        "server": { "host": "127.0.0.1", "port": 0 },
        "providers": [
            { "name": "primary", "type": "openai", "base_url": primary_url, "api_key": "test-key" },
            { "name": "fallback", "type": "openai", "base_url": fallback_url, "api_key": "test-key" }
        ],
        "model_groups": [{
            "name": "g",
            "models": [
                { "provider": "primary", "model": "gpt-4", "priority": 1 },
                { "provider": "fallback", "model": "gpt-4", "priority": 2 }
            ]
        }]
    }))
    .expect("two-provider config should deserialize");
    config.virtual_keys.enforcement = EnforcementMode::Disabled;
    config.retry.max_retries_per_provider = 0;
    config.retry.backoff_sequence_seconds = vec![0];
    config.retry.jitter_enabled = false;
    config
}

/// Mount a wiremock upstream returning an SSE body on /v1/chat/completions.
async fn sse_upstream(body: String) -> MockServer {
    let server = MockServer::start().await;
    Mock::given(method("POST"))
        .and(path("/v1/chat/completions"))
        .respond_with(
            ResponseTemplate::new(200)
                .insert_header("content-type", "text/event-stream")
                .set_body_string(body),
        )
        .mount(&server)
        .await;
    server
}

/// Build one upstream chat SSE chunk: `data: {json}\n\n` with the delta
/// nested under `choices[0].delta` (the wire shape every OpenAI-compatible
/// provider emits).
fn chat_chunk(delta: serde_json::Value, finish_reason: Option<&str>) -> String {
    let chunk = json!({
        "id": "chatcmpl-mock",
        "object": "chat.completion.chunk",
        "created": 1700000000_i64,
        "model": "gpt-4",
        "choices": [{
            "index": 0,
            "delta": delta,
            "finish_reason": finish_reason.map(|f| Value::String(f.to_string()))
                .unwrap_or(Value::Null)
        }]
    });
    format!("data: {chunk}\n\n")
}

/// POST /v1/messages with `stream: true`.
async fn post_messages_stream(app: axum::Router, body: Value) -> (StatusCode, String) {
    let request = Request::post("/v1/messages")
        .header("content-type", "application/json")
        .header("anthropic-version", VERSION)
        .body(Body::from(body.to_string()))
        .unwrap();
    let response = app.oneshot(request).await.unwrap();
    let status = response.status();
    let bytes = axum::body::to_bytes(response.into_body(), 4 * 1024 * 1024)
        .await
        .unwrap();
    (status, String::from_utf8_lossy(&bytes).to_string())
}

/// One parsed SSE event: `(event name, data JSON)`.
fn sse_events(raw: &str) -> Vec<(String, Value)> {
    let mut events = Vec::new();
    let mut name: Option<String> = None;
    for line in raw.lines() {
        let line = line.trim_end_matches('\r');
        if let Some(event) = line.strip_prefix("event: ") {
            name = Some(event.trim().to_string());
        } else if let Some(data) = line.strip_prefix("data: ") {
            let value =
                serde_json::from_str::<Value>(data).unwrap_or(Value::String(data.to_string()));
            events.push((name.clone().unwrap_or_default(), value));
            name = None;
        }
    }
    events
}

fn event_names(raw: &str) -> Vec<String> {
    sse_events(raw).into_iter().map(|(name, _)| name).collect()
}

fn anthropic_stream_request() -> Value {
    json!({
        "model": "g",
        "max_tokens": 128,
        "stream": true,
        "messages": [{ "role": "user", "content": "hello" }]
    })
}

async fn poll_logs(
    server: &GatewayServer,
    timeout: Duration,
    done: impl Fn(&[LogEntry]) -> bool,
) -> Vec<LogEntry> {
    let deadline = tokio::time::Instant::now() + timeout;
    loop {
        let response = server
            .build_router()
            .oneshot(Request::get("/dashboard/logs").body(Body::empty()).unwrap())
            .await
            .unwrap();
        let bytes = axum::body::to_bytes(response.into_body(), 4 * 1024 * 1024)
            .await
            .unwrap();
        let rows: Vec<LogEntry> =
            serde_json::from_slice(&bytes).expect("dashboard logs should be LogEntry JSON");
        if done(&rows) || tokio::time::Instant::now() >= deadline {
            return rows;
        }
        tokio::time::sleep(Duration::from_millis(50)).await;
    }
}

// ---------------------------------------------------------------------------
// Raw-TCP SSE mocks (paced / truncated bodies; mirrors integration.rs's
// hand-rolled upstreams for behaviors wiremock cannot express)
// ---------------------------------------------------------------------------

/// Serve SSE chunks with real inter-chunk delays so the client-visible
/// stream has genuine idle gaps (wiremock cannot pace individual chunks).
async fn start_paced_sse_mock(writes: Vec<Vec<u8>>, delay_ms: u64) -> String {
    use tokio::io::{AsyncReadExt, AsyncWriteExt};
    let listener = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
    let addr = listener.local_addr().unwrap();
    tokio::spawn(async move {
        let (mut socket, _) = match tokio::time::timeout(Duration::from_secs(10), listener.accept())
            .await
        {
            Ok(accepted) => accepted.expect("paced SSE mock: accept() failed"),
            Err(_) => panic!("paced SSE mock: the gateway never connected within 10s"),
        };
        let mut scratch = [0u8; 8192];
        let _ = tokio::time::timeout(Duration::from_secs(5), socket.read(&mut scratch)).await;
        let head = "HTTP/1.1 200 OK\r\ncontent-type: text/event-stream\r\nconnection: close\r\n\r\n";
        socket.write_all(head.as_bytes()).await.unwrap();
        for chunk in writes {
            tokio::time::sleep(Duration::from_millis(delay_ms)).await;
            socket.write_all(&chunk).await.unwrap();
        }
        socket.shutdown().await.unwrap();
    });
    format!("http://{addr}")
}

/// Serve SSE chunks, then truncate the body: the head advertises a
/// `content-length` larger than what is written, so the client sees a
/// mid-body transport error (premature EOF) — the mid-stream failure class.
async fn start_truncated_sse_mock(writes: Vec<Vec<u8>>) -> String {
    use tokio::io::{AsyncReadExt, AsyncWriteExt};
    let listener = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
    let addr = listener.local_addr().unwrap();
    tokio::spawn(async move {
        let (mut socket, _) = match tokio::time::timeout(Duration::from_secs(10), listener.accept())
            .await
        {
            Ok(accepted) => accepted.expect("truncated SSE mock: accept() failed"),
            Err(_) => panic!("truncated SSE mock: the gateway never connected within 10s"),
        };
        let mut scratch = [0u8; 8192];
        let _ = tokio::time::timeout(Duration::from_secs(5), socket.read(&mut scratch)).await;
        let advertised: usize = 64 * 1024;
        let head = format!(
            "HTTP/1.1 200 OK\r\ncontent-type: text/event-stream\r\ncontent-length: {advertised}\r\n\r\n"
        );
        socket.write_all(head.as_bytes()).await.unwrap();
        for chunk in writes {
            socket.write_all(&chunk).await.unwrap();
        }
        // Premature FIN: far fewer bytes than the advertised content-length.
        socket.shutdown().await.unwrap();
    });
    format!("http://{addr}")
}

// ---------------------------------------------------------------------------
// 1. Text-only stream: full event order (Req 2.1)
// ---------------------------------------------------------------------------

#[tokio::test]
async fn text_stream_full_event_order() {
    let body = [
        chat_chunk(json!({ "role": "assistant", "content": "" }), None),
        chat_chunk(json!({ "content": "Hello" }), None),
        chat_chunk(json!({ "content": " world" }), None),
        chat_chunk(json!({ "content": "" }), Some("stop")),
    ]
    .concat();
    let upstream = sse_upstream(body).await;

    let mut config = base_config();
    config.providers[0].base_url = Some(upstream.uri());
    common::isolate_databases(&mut config);
    let server = GatewayServer::new(config, None).await.unwrap();

    let (status, raw) =
        post_messages_stream(server.build_router(), anthropic_stream_request()).await;
    assert_eq!(status, StatusCode::OK, "body: {raw}");

    let events = sse_events(&raw);
    let names: Vec<&str> = events.iter().map(|(name, _)| name.as_str()).collect();
    assert_eq!(
        names,
        vec![
            "message_start",
            "content_block_start",
            "content_block_delta",
            "content_block_delta",
            "content_block_stop",
            "message_delta",
            "message_stop",
        ],
        "raw stream:\n{raw}"
    );

    // Text block is index 0.
    assert_eq!(events[1].1["index"], 0);
    assert_eq!(events[1].1["content_block"]["type"], "text");

    // Concatenated text deltas form the full text.
    let text: String = events
        .iter()
        .filter_map(|(_, data)| data["delta"]["text"].as_str().map(str::to_string))
        .collect();
    assert_eq!(text, "Hello world");

    // message_delta carries stop_reason + cumulative usage.
    let message_delta = &events[events.len() - 2].1;
    assert_eq!(message_delta["delta"]["stop_reason"], "end_turn");
    assert!(message_delta["usage"]["output_tokens"].as_u64().unwrap_or(0) >= 1);

    // message_start scaffold (Req 2.1): role assistant, empty content.
    assert_eq!(events[0].1["type"], "message_start");
    assert_eq!(events[0].1["message"]["role"], "assistant");
    assert_eq!(events[0].1["message"]["content"], json!([]));
}

// ---------------------------------------------------------------------------
// 2. Interleaved text + tool_use + reasoning (Req 2.2, 2.3, 2.6)
// ---------------------------------------------------------------------------

#[tokio::test]
async fn interleaved_text_tool_reasoning_blocks() {
    let body = [
        chat_chunk(json!({ "role": "assistant", "content": "" }), None),
        chat_chunk(json!({ "reasoning": "pondering" }), None),
        chat_chunk(json!({ "content": "answer" }), None),
        chat_chunk(
            json!({ "tool_calls": [{
                "index": 0,
                "id": "call_weather",
                "type": "function",
                "function": { "name": "get_weather", "arguments": "" }
            }]}),
            None,
        ),
        chat_chunk(
            json!({ "tool_calls": [{
                "index": 0,
                "function": { "arguments": "{\"city\":" }
            }]}),
            None,
        ),
        chat_chunk(
            json!({ "tool_calls": [{
                "index": 0,
                "function": { "arguments": "\"Paris\"}" }
            }]}),
            None,
        ),
        chat_chunk(json!({ "content": "" }), Some("tool_calls")),
    ]
    .concat();
    let upstream = sse_upstream(body).await;

    let mut config = base_config();
    config.providers[0].base_url = Some(upstream.uri());
    common::isolate_databases(&mut config);
    let server = GatewayServer::new(config, None).await.unwrap();

    let (status, raw) =
        post_messages_stream(server.build_router(), anthropic_stream_request()).await;
    assert_eq!(status, StatusCode::OK, "body: {raw}");

    let events = sse_events(&raw);

    // Every block closed before message_delta (Req 2.6).
    let message_delta_pos = events
        .iter()
        .position(|(name, _)| name == "message_delta")
        .expect("message_delta present");
    let starts: Vec<(usize, usize)> = events
        .iter()
        .enumerate()
        .filter(|(_, (name, _))| name == "content_block_start")
        .map(|(pos, (_, data))| (pos, data["index"].as_u64().unwrap_or(999) as usize))
        .collect();
    let stops: Vec<usize> = events
        .iter()
        .enumerate()
        .filter(|(_, (name, _))| name == "content_block_stop")
        .map(|(_, (_, data))| data["index"].as_u64().unwrap_or(999) as usize)
        .collect();
    // Sequential block indices: thinking(0), text(1), tool_use(2).
    assert_eq!(
        starts.iter().map(|(_, i)| *i).collect::<Vec<_>>(),
        vec![0, 1, 2]
    );
    assert_eq!(stops, vec![0, 1, 2]);
    for (pos, _) in &starts {
        assert!(pos < &message_delta_pos, "block start after message_delta");
    }

    // Thinking block: thinking_delta inside block 0 (Req 2.3).
    assert_eq!(events[1].1["index"], 0);
    assert_eq!(events[1].1["content_block"]["type"], "thinking");
    let thinking: String = events
        .iter()
        .filter_map(|(_, data)| data["delta"]["thinking"].as_str().map(str::to_string))
        .collect();
    assert_eq!(thinking, "pondering");

    // Text block.
    let text: String = events
        .iter()
        .filter_map(|(_, data)| data["delta"]["text"].as_str().map(str::to_string))
        .collect();
    assert_eq!(text, "answer");

    // Tool block: content_block_start with empty input {} then
    // input_json_delta fragments whose concatenation is valid JSON (Req 2.2).
    let (_, tool_start) = events
        .iter()
        .find(|(name, data)| name == "content_block_start" && data["content_block"]["type"] == "tool_use")
        .expect("tool_use block start");
    assert_eq!(tool_start["content_block"]["input"], json!({}));

    let partial: String = events
        .iter()
        .filter_map(|(_, data)| data["delta"]["partial_json"].as_str().map(str::to_string))
        .collect();
    assert_eq!(partial, "{\"city\":\"Paris\"}");
    let parsed: Value =
        serde_json::from_str(&partial).expect("concatenated partial_json is valid JSON");
    assert_eq!(parsed, json!({ "city": "Paris" }));

    // stop_reason reflects the tool_call finish.
    let (_, message_delta) = events.iter().rev().find(|(name, _)| name == "message_delta").unwrap();
    assert_eq!(message_delta["delta"]["stop_reason"], "tool_use");
}

// ---------------------------------------------------------------------------
// 3. Ping during stall (Req 2.4 — liveness presence, generous bounds)
// ---------------------------------------------------------------------------

#[tokio::test]
async fn stalled_stream_emits_ping_keepalive() {
    // Inter-chunk gaps (1.5s) exceed the keep-alive interval (1s), so the SSE
    // layer must emit liveness during the stall. Presence assertion only —
    // no timing assertions (Task 9 constraint). 1s is the smallest interval
    // the config supports (whole seconds, floored at 1 by
    // keep_alive_ping_interval).
    let writes = vec![
        chat_chunk(json!({ "content": "part-one" }), None).into_bytes(),
        chat_chunk(json!({ "content": "part-two" }), None).into_bytes(),
        chat_chunk(json!({ "content": "" }), Some("stop")).into_bytes(),
    ];
    let uri = start_paced_sse_mock(writes, 1_500).await;

    let mut config = base_config();
    config.providers[0].base_url = Some(uri);
    config.streaming = Some(StreamingConfig {
        keepalive_interval_seconds: 1,
        ..StreamingConfig::default()
    });
    common::isolate_databases(&mut config);
    let server = GatewayServer::new(config, None).await.unwrap();

    let (status, raw) =
        post_messages_stream(server.build_router(), anthropic_stream_request()).await;
    assert_eq!(status, StatusCode::OK, "body: {raw}");

    // Liveness presence: the relay loop must emit an Anthropic wire-format
    // `event: ping` frame (Req 2.4) during the stall. Generous bounds —
    // presence only, no timing assertions (Task 9 constraint).
    let has_ping_event = event_names(&raw).iter().any(|name| name == "ping");
    assert!(
        has_ping_event,
        "no `event: ping` frame during a 1.5s stall (keepalive_interval=1s); raw:\n{raw}"
    );
}

// ---------------------------------------------------------------------------
// 3b. Multi-byte UTF-8 split across TCP chunk boundaries (relay regression)
// ---------------------------------------------------------------------------

#[tokio::test]
async fn utf8_multi_byte_char_split_across_chunk_boundaries_round_trips() {
    // The relay once corrupted a multi-byte UTF-8 char landing on a chunk
    // boundary (per-chunk lossy decode). Line-buffered decoding must wait
    // for the complete char: an emoji (U+1F60A "😊", 4 UTF-8 bytes) inside
    // a text_delta payload is split mid-sequence across two TCP writes; the
    // concatenated text_delta content must round-trip exactly, with no
    // U+FFFD replacement chars.
    let full_text = "hi你😊bye";
    let chunk1 = chat_chunk(json!({ "content": "hi" }), None);
    let chunk2 = chat_chunk(json!({ "content": "你😊" }), None);
    let chunk3 = chat_chunk(json!({ "content": "bye" }), None);
    let chunk4 = chat_chunk(json!({ "content": "" }), Some("stop"));

    // serde_json serializes non-ASCII as literal UTF-8 (no \u escapes), so
    // chunk2's raw bytes contain the emoji's 4 bytes directly; split them
    // across two TCP writes mid-sequence (first byte, then remaining three).
    let raw2 = chunk2.as_bytes();
    let emoji_pos = raw2
        .windows(4)
        .position(|w| w == [0xF0, 0x9F, 0x98, 0x8A])
        .expect("chunk2 must contain the raw emoji bytes");
    let (w2a, w2b) = raw2.split_at(emoji_pos + 1);

    let writes = vec![
        chunk1.into_bytes(),
        w2a.to_vec(),
        w2b.to_vec(),
        chunk3.into_bytes(),
        chunk4.into_bytes(),
    ];
    let uri = start_paced_sse_mock(writes, 50).await;

    let mut config = base_config();
    config.providers[0].base_url = Some(uri);
    common::isolate_databases(&mut config);
    let server = GatewayServer::new(config, None).await.unwrap();

    let (status, raw) =
        post_messages_stream(server.build_router(), anthropic_stream_request()).await;
    assert_eq!(status, StatusCode::OK, "body: {raw}");

    let text: String = sse_events(&raw)
        .into_iter()
        .filter_map(|(_, data)| data["delta"]["text"].as_str().map(str::to_string))
        .collect();
    assert_eq!(text, full_text, "raw stream:\n{raw}");
    assert!(
        !text.contains('\u{FFFD}'),
        "multi-byte char split across chunk boundaries must never decode lossy; raw:\n{raw}"
    );
}

// ---------------------------------------------------------------------------
// 3c. stop_sequence streaming mapping (handlers.rs suffix-match rule)
// ---------------------------------------------------------------------------

#[tokio::test]
async fn stop_sequence_hit_maps_to_stop_sequence_stop_reason() {
    // Upstream streams text ending with the request's stop sequence and
    // finish_reason "stop". handlers.rs (streaming + replay paths) applies
    // the buffered synthesizer's suffix-match rule: when finish_reason is
    // "stop" and the generated text ends with a requested stop sequence,
    // message_delta.delta.stop_reason == "stop_sequence" AND
    // message_delta.delta.stop_sequence names the matched sequence.
    let body = [
        chat_chunk(json!({ "role": "assistant", "content": "" }), None),
        chat_chunk(json!({ "content": "one two three END" }), None),
        chat_chunk(json!({ "content": "" }), Some("stop")),
    ]
    .concat();
    let upstream = sse_upstream(body).await;

    let mut config = base_config();
    config.providers[0].base_url = Some(upstream.uri());
    common::isolate_databases(&mut config);
    let server = GatewayServer::new(config, None).await.unwrap();

    let mut request = anthropic_stream_request();
    request["stop_sequences"] = json!(["STOP", "END"]);

    let (status, raw) = post_messages_stream(server.build_router(), request).await;
    assert_eq!(status, StatusCode::OK, "body: {raw}");

    let events = sse_events(&raw);
    let (_, message_delta) = events
        .iter()
        .rev()
        .find(|(name, _)| name == "message_delta")
        .expect("message_delta present");
    assert_eq!(
        message_delta["delta"]["stop_reason"], "stop_sequence",
        "raw:\n{raw}"
    );
    assert_eq!(
        message_delta["delta"]["stop_sequence"], "END",
        "the suffix-matched sequence must be named; raw:\n{raw}"
    );

    // Text is relayed uncut (the gateway does not trim the sequence).
    let text: String = events
        .iter()
        .filter_map(|(_, data)| data["delta"]["text"].as_str().map(str::to_string))
        .collect();
    assert_eq!(text, "one two three END", "raw:\n{raw}");
}

// ---------------------------------------------------------------------------
// 3d. sse_relay_frames: upstream error frame AFTER message_stop (unit-style)
// ---------------------------------------------------------------------------

#[test]
fn sse_relay_frames_relays_error_frame_after_message_stop_in_place() {
    // Anthropic native streams may carry an `event: error` frame after
    // `message_stop` (e.g. late upstream failure during relay). The verbatim
    // relay must preserve the frame in place: same event name, same data,
    // same ordering. `sse_relay_frames` is `pub` in
    // src/anthropic/passthrough.rs, so this calls the library fn directly.
    let raw = concat!(
        "event: message_stop\n",
        "data: {\"type\":\"message_stop\"}\n",
        "\n",
        "event: error\n",
        "data: {\"type\":\"error\",\"error\":{\"type\":\"overloaded_error\",\"message\":\"late failure\"}}\n",
        "\n",
    );
    let frames = sse_relay_frames(raw);
    assert_eq!(frames.len(), 2, "both frames relay");
    assert_eq!(frames[0].0, "message_stop");
    assert_eq!(frames[0].1, "{\"type\":\"message_stop\"}");
    assert_eq!(frames[1].0, "error", "event name preserved");
    assert_eq!(
        frames[1].1,
        "{\"type\":\"error\",\"error\":{\"type\":\"overloaded_error\",\"message\":\"late failure\"}}"
    );
    // Ordering intact: the error frame stays after message_stop.
    let names: Vec<&str> = frames.iter().map(|(name, _)| name.as_str()).collect();
    assert_eq!(names, vec!["message_stop", "error"]);
}

// ---------------------------------------------------------------------------
// 4. Mid-stream failure → event: error frame (Req 2.5)
// ---------------------------------------------------------------------------

#[tokio::test]
async fn mid_stream_failure_emits_error_frame() {
    // Single provider, max_retries 0 → failover impossible. The upstream
    // delivers a good first chunk, then the body is truncated (premature
    // EOF against the advertised content-length) — a mid-stream failure
    // after the 200/SSE headers.
    let first_chunk = chat_chunk(json!({ "content": "partial" }), None);
    let uri = start_truncated_sse_mock(vec![first_chunk.into_bytes()]).await;

    let mut config = base_config();
    config.providers[0].base_url = Some(uri);
    config.retry.max_retries_per_provider = 0;
    config.retry.jitter_enabled = false;
    common::isolate_databases(&mut config);
    let server = GatewayServer::new(config, None).await.unwrap();

    let (status, raw) =
        post_messages_stream(server.build_router(), anthropic_stream_request()).await;
    assert_eq!(status, StatusCode::OK, "body: {raw}");

    let events = sse_events(&raw);
    let error_events: Vec<&(String, Value)> =
        events.iter().filter(|(name, _)| name == "error").collect();
    assert!(
        !error_events.is_empty(),
        "stream must end with an event: error frame, never a silent disconnect; raw:\n{raw}"
    );
    // The error frame carries an Anthropic error object (Req 3.1 shape).
    let (_, data) = error_events[0];
    assert_eq!(data["type"], "error");
    assert!(data["error"]["type"].as_str().is_some());
    // The error frame is terminal: nothing follows it.
    let last = events.last().expect("stream has events");
    assert_eq!(last.0, "error", "raw:\n{raw}");
}

// ---------------------------------------------------------------------------
// 5. Pre-content failover transparent + attempt ledger rows (Req 6.4)
// ---------------------------------------------------------------------------

#[tokio::test]
async fn pre_content_failover_transparent_with_attempt_rows() {
    // Primary always 500s at TTFB; fallback serves the full stream.
    let primary = MockServer::start().await;
    Mock::given(method("POST"))
        .and(path("/v1/chat/completions"))
        .respond_with(ResponseTemplate::new(500).set_body_string("primary down"))
        .mount(&primary)
        .await;
    let fallback_body = [
        chat_chunk(json!({ "role": "assistant", "content": "" }), None),
        chat_chunk(json!({ "content": "from fallback" }), None),
        chat_chunk(json!({ "content": "" }), Some("stop")),
    ]
    .concat();
    let fallback = sse_upstream(fallback_body).await;

    let mut config = two_provider_config(primary.uri(), fallback.uri());
    common::isolate_databases(&mut config);
    let server = GatewayServer::new(config, None).await.unwrap();

    let (status, raw) =
        post_messages_stream(server.build_router(), anthropic_stream_request()).await;
    assert_eq!(status, StatusCode::OK, "body: {raw}");

    let events = sse_events(&raw);
    let names: Vec<&str> = events.iter().map(|(name, _)| name.as_str()).collect();
    // COMPLETE valid sequence: single message_start, no error frame (Req 6.4:
    // the client-visible event stream is unaffected by the pre-content move).
    assert_eq!(
        names.first(),
        Some(&"message_start"),
        "client stream must start with message_start; raw:\n{raw}"
    );
    assert_eq!(
        names.iter().filter(|name| **name == "message_start").count(),
        1,
        "failover must be transparent: exactly one message_start; raw:\n{raw}"
    );
    assert_eq!(names.last(), Some(&"message_stop"));
    assert!(!names.contains(&"error"), "no error frame after transparent failover");

    // Attempt ledger: #attempt row for the failed primary try with
    // error_class (mirrors request_outcome_logging.rs assertions).
    let rows = poll_logs(&server, Duration::from_secs(5), |rows| {
        rows.iter().any(|row| row.path.contains("#attempt"))
            && rows.iter().any(|row| row.path == "/v1/messages")
    })
    .await;
    let attempts: Vec<&LogEntry> = rows.iter().filter(|row| row.path.contains("#attempt")).collect();
    assert!(!attempts.is_empty(), "expected #attempt rows; rows: {rows:?}");
    let primary_attempts: Vec<&&LogEntry> =
        attempts.iter().filter(|row| row.provider == "primary").collect();
    assert!(
        !primary_attempts.is_empty(),
        "no #attempt row for the failed primary try; rows: {rows:?}"
    );
    for row in primary_attempts {
        assert_eq!(row.status_code, 500);
        assert_eq!(row.error_class.as_deref(), Some("upstream_http_500"));
    }

    // The anthropic-protocol outcome row exists (Req 6.3: /v1/messages
    // requests distinguishable in the log views).
    let outcomes: Vec<&LogEntry> = rows.iter().filter(|row| row.path == "/v1/messages").collect();
    assert!(!outcomes.is_empty(), "expected /v1/messages outcome row; rows: {rows:?}");
}

// ---------------------------------------------------------------------------
// 6. Mantle passthrough forwards beta headers verbatim (Req 4.3)
// ---------------------------------------------------------------------------

// See file-level deviation 2: end-to-end interception is impossible (Mantle
// base URL hardcoded; BedrockProvider fields private). The provider side is
// covered by src/providers/bedrock.rs `mod anthropic_passthrough` wiremock
// tests (capturing at {root}/anthropic/v1/messages). This test pins the
// handler-side chain: full beta capture and scope installation.
#[tokio::test]
async fn mantle_passthrough_forwards_beta_headers_verbatim() {
    // Handler-side capture: unusual beta values (known + unknown, comma
    // list) survive verbatim — no rejection, no rewriting (Req 2.2/4.3).
    let mut headers = axum::http::HeaderMap::new();
    headers.insert("anthropic-version", VERSION.parse().unwrap());
    headers.append(
        "anthropic-beta",
        "context-1h-2025-08-07,unknown-beta-x".parse().unwrap(),
    );
    let beta = beta_header_value(&headers);
    assert_eq!(beta.as_deref(), Some("context-1h-2025-08-07,unknown-beta-x"));

    // Scope installation: the context the Mantle Messages adapter reads
    // carries the verbatim values (Req 4.3).
    let body = json!({
        "model": "anthropic.claude-sonnet-4-6",
        "max_tokens": 64,
        "stream": true,
        "messages": [{ "role": "user", "content": "hi" }]
    });
    let scope = AnthropicPassthroughScope::new(body.clone(), VERSION.to_string(), beta.clone());
    let context = scope
        .scope(async {
            ai_gateway::anthropic::passthrough::current().expect("scope must install context")
        })
        .await;
    assert_eq!(context.anthropic_version, VERSION);
    assert_eq!(
        context.anthropic_beta.as_deref(),
        Some("context-1h-2025-08-07,unknown-beta-x")
    );
    assert_eq!(context.body, body);

    // Multiple header lines join with ',' in arrival order (documented
    // open-list form, Req 4.3).
    let mut multi = axum::http::HeaderMap::new();
    multi.append("anthropic-beta", "context-1h-2025-08-07".parse().unwrap());
    multi.append("anthropic-beta", "unknown-beta-x".parse().unwrap());
    assert_eq!(
        beta_header_value(&multi).as_deref(),
        Some("context-1h-2025-08-07,unknown-beta-x")
    );
}
