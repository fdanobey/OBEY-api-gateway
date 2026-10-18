//! Reasoning progress relay: pure classification of reasoning-only SSE frame
//! payloads plus a bounded, fail-open tap handle used to forward reasoning
//! deltas to a waiting client while buffered dispatch completes upstream.

use serde_json::Value;
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::Arc;
use tokio::sync::mpsc;

/// A single frame forwarded through the relay channel.
#[derive(Debug, Clone, PartialEq)]
pub enum RelayFrame {
    Reasoning { text: String },
}

/// Cheap-clone handle to a bounded relay channel.
///
/// `try_relay` never blocks and never fails: a full or disconnected channel
/// drops the frame so the buffered pipeline is never perturbed by the relay.
#[derive(Debug, Clone)]
pub struct RelayHandle {
    tx: mpsc::Sender<RelayFrame>,
    frames_sent: Arc<AtomicU64>,
    frames_forwarded: Arc<AtomicU64>,
    dropped: Arc<AtomicU64>,
}

impl RelayHandle {
    /// Creates a handle/receiver pair over a bounded channel.
    pub fn new() -> (Self, mpsc::Receiver<RelayFrame>) {
        let (tx, rx) = mpsc::channel(128);
        (
            Self {
                tx,
                frames_sent: Arc::new(AtomicU64::new(0)),
                frames_forwarded: Arc::new(AtomicU64::new(0)),
                dropped: Arc::new(AtomicU64::new(0)),
            },
            rx,
        )
    }

    /// Attempts to enqueue a frame, counting full/disconnected drops.
    pub fn try_relay(&self, frame: RelayFrame) {
        if self.tx.try_send(frame).is_ok() {
            self.frames_sent.fetch_add(1, Ordering::Relaxed);
        } else {
            self.dropped.fetch_add(1, Ordering::Relaxed);
        }
    }

    /// Number of frames dropped because the channel was full or disconnected.
    pub fn dropped(&self) -> u64 {
        self.dropped.load(Ordering::Relaxed)
    }

    /// Number of frames forwarded to the client.
    // Bin target: consumed by relay telemetry logging (spec Tasks 6-8).
    pub fn forwarded(&self) -> u64 {
        self.frames_forwarded.load(Ordering::Relaxed)
    }

    pub fn record_forwarded(&self) -> u64 {
        self.frames_forwarded.fetch_add(1, Ordering::Relaxed) + 1
    }

    #[cfg(test)]
    pub fn sent(&self) -> u64 {
        self.frames_sent.load(Ordering::Relaxed)
    }
}

/// Classifies a `data:` payload as relayable reasoning.
///
/// Returns the reasoning text iff `choices[0].delta` carries a non-empty
/// `reasoning` or `reasoning_content` string and carries no non-null
/// `content` and no `tool_calls` field. Anything else (content or tool-call
/// deltas, usage-only frames, `[DONE]`, non-JSON) fails closed to `None`.
///
/// Note: the text is returned owned because `serde_json` owns parsed strings
/// (escaped strings cannot be borrowed zero-copy) and [`RelayFrame::Reasoning`]
/// takes an owned `String` anyway.
pub fn classify_reasoning_frame(payload: &str) -> Option<String> {
    let value: Value = serde_json::from_str(payload).ok()?;
    let delta = value.pointer("/choices/0/delta")?.as_object()?;

    if delta.contains_key("tool_calls") {
        return None;
    }
    if delta
        .get("content")
        .is_some_and(|content| !content.is_null())
    {
        return None;
    }

    for field in ["reasoning", "reasoning_content"] {
        if let Some(text) = delta.get(field).and_then(Value::as_str) {
            if !text.trim().is_empty() {
                return Some(text.to_owned());
            }
        }
    }

    None
}

/// Returns the payload after the `data:` prefix (one optional leading space
/// stripped), or `None` for non-data SSE lines (comments, `event:`, `id:`,
/// `retry:`).
///
/// Line splitting upstream of this helper assumes LF/CRLF terminators
/// (what providers emit); bare-CR streams are unsupported and fail closed.
pub fn sse_data_payload(line: &str) -> Option<&str> {
    let payload = line.strip_prefix("data:")?;
    Some(payload.strip_prefix(' ').unwrap_or(payload))
}

#[cfg(test)]
mod tests {
    use super::{classify_reasoning_frame, sse_data_payload, RelayFrame, RelayHandle};

    fn reasoning_payload(delta: &str) -> String {
        format!(r#"{{"choices":[{{"index":0,"delta":{delta}}}]}}"#)
    }

    #[test]
    fn classifies_pure_reasoning_delta() {
        let payload = reasoning_payload(r#"{"reasoning":"thinking hard"}"#);
        assert_eq!(
            classify_reasoning_frame(&payload).as_deref(),
            Some("thinking hard")
        );
    }

    #[test]
    fn classifies_reasoning_content_delta() {
        let payload = reasoning_payload(r#"{"reasoning_content":"deep thought"}"#);
        assert_eq!(
            classify_reasoning_frame(&payload).as_deref(),
            Some("deep thought")
        );
    }

    #[test]
    fn prefers_reasoning_over_reasoning_content() {
        let payload = reasoning_payload(r#"{"reasoning":"first","reasoning_content":"second"}"#);
        assert_eq!(classify_reasoning_frame(&payload).as_deref(), Some("first"));
    }

    #[test]
    fn rejects_reasoning_with_content() {
        let payload = reasoning_payload(r#"{"reasoning":"thinking","content":"answer"}"#);
        assert_eq!(classify_reasoning_frame(&payload), None);
    }

    #[test]
    fn rejects_reasoning_with_tool_calls() {
        let payload = reasoning_payload(r#"{"reasoning":"thinking","tool_calls":[{"id":"1"}]}"#);
        assert_eq!(classify_reasoning_frame(&payload), None);
    }

    #[test]
    fn rejects_empty_reasoning_string() {
        let payload = reasoning_payload(r#"{"reasoning":""}"#);
        assert_eq!(classify_reasoning_frame(&payload), None);
    }

    #[test]
    fn rejects_whitespace_only_reasoning() {
        let payload = reasoning_payload(r#"{"reasoning":"   \n  "}"#);
        assert_eq!(classify_reasoning_frame(&payload), None);
    }

    #[test]
    fn allows_null_content_alongside_reasoning() {
        let payload = reasoning_payload(r#"{"reasoning":"thinking","content":null}"#);
        assert_eq!(
            classify_reasoning_frame(&payload).as_deref(),
            Some("thinking")
        );
    }

    #[test]
    fn rejects_done_sentinel() {
        assert_eq!(classify_reasoning_frame("[DONE]"), None);
    }

    #[test]
    fn rejects_non_json_payload() {
        assert_eq!(classify_reasoning_frame("not json"), None);
        assert_eq!(classify_reasoning_frame(""), None);
    }

    #[test]
    fn rejects_usage_only_frame() {
        let payload =
            r#"{"usage":{"prompt_tokens":1,"completion_tokens":2,"total_tokens":3},"choices":[]}"#;
        assert_eq!(classify_reasoning_frame(payload), None);
    }

    #[test]
    fn rejects_frame_without_choices() {
        let payload = r#"{"id":"x","usage":{"total_tokens":3}}"#;
        assert_eq!(classify_reasoning_frame(payload), None);
    }

    #[test]
    fn unescapes_reasoning_text() {
        let payload = reasoning_payload(r#"{"reasoning":"line one\nline two"}"#);
        assert_eq!(
            classify_reasoning_frame(&payload).as_deref(),
            Some("line one\nline two")
        );
    }

    #[test]
    fn sse_data_payload_variants() {
        assert_eq!(sse_data_payload(r#"data: {"x":1}"#), Some(r#"{"x":1}"#));
        assert_eq!(sse_data_payload(r#"data:{"x":1}"#), Some(r#"{"x":1}"#));
        assert_eq!(sse_data_payload(": comment"), None);
        assert_eq!(sse_data_payload("event: foo"), None);
        assert_eq!(sse_data_payload("id: 1"), None);
        assert_eq!(sse_data_payload("retry: 100"), None);
    }

    #[test]
    fn try_relay_increments_counter_on_success() {
        let (handle, mut receiver) = RelayHandle::new();
        handle.try_relay(RelayFrame::Reasoning {
            text: "thinking".to_owned(),
        });
        assert_eq!(handle.sent(), 1);

        let frame = receiver.try_recv().expect("frame enqueued");
        assert_eq!(
            frame,
            RelayFrame::Reasoning {
                text: "thinking".to_owned()
            }
        );
    }

    #[test]
    fn try_relay_ignores_disconnected_receiver() {
        let (handle, receiver) = RelayHandle::new();
        drop(receiver);
        handle.try_relay(RelayFrame::Reasoning {
            text: "thinking".to_owned(),
        });
        assert_eq!(handle.sent(), 0);
        assert_eq!(handle.dropped(), 1);
    }

    #[test]
    fn try_relay_counts_dropped_frames_when_channel_is_full() {
        let (handle, _receiver) = RelayHandle::new();
        // Capacity is bounded (128): fill it, then verify the overflow drop
        // counter increments alongside the sent counter staying put.
        for i in 0..128 {
            handle.try_relay(RelayFrame::Reasoning {
                text: format!("thinking {i}"),
            });
        }
        assert_eq!(handle.sent(), 128);
        assert_eq!(handle.dropped(), 0);

        handle.try_relay(RelayFrame::Reasoning {
            text: "overflow".to_owned(),
        });
        assert_eq!(handle.sent(), 128);
        assert_eq!(handle.dropped(), 1);
    }

    #[test]
    fn dropped_counter_survives_handle_clones() {
        let (handle, receiver) = RelayHandle::new();
        let cloned = handle.clone();
        drop(receiver);
        cloned.try_relay(RelayFrame::Reasoning {
            text: "thinking".to_owned(),
        });
        assert_eq!(handle.dropped(), 1);
    }
}
