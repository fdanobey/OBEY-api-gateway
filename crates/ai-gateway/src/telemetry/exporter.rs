//! Background OTLP trace exporter (opentelemetry-trace-export spec,
//! design "Component 6", Req 6).
//!
//! [`TelemetryExporter`] owns a bounded queue (1024 spans; unit =
//! individual spans) and a single background delivery task. Span batches
//! are accepted whole or dropped whole — never partially — so backends
//! never render half a request. Delivery POSTs gzip-compressed
//! OTLP/HTTP JSON (`otlp::encode_batch`) with the configured headers;
//! HTTP 2xx is delivered, anything else or a transport error retries
//! with 5 s/30 s/2 m backoff, then the batch is dropped and counted.
//!
//! Fail-open discipline (Req 6.2/6.3): `try_enqueue` is non-blocking
//! try-send only, never panics, and export failures never surface to
//! client requests. The exporter starts dormant (`new` spawns nothing);
//! the single delivery task exists only while the config is enabled.
//!
//! Task lifecycle: the task owns the receiving end for its lifetime and
//! returns it to the shared slot on every clean exit (disable), so a
//! later re-enable or shutdown flush reuses the same channel. A task
//! that panics drops its receiver; the supervisor (Error Scenario 5)
//! detects the closed channel on the next enqueue or `apply_config`
//! and re-spawns on a rebuilt channel. `task_alive` is owned by the
//! task and its supervisor alone — `apply_config` never writes it, so
//! no second delivery task can be spawned under a live one.
//!
//! Shutdown deadlines: `shutdown_flush` publishes a deadline epoch the
//! delivery task honours inside its retry chain — each attempt and
//! each backoff sleep is deadline-aware, so an active 5 s/30 s/2 m
//! chain aborts within the budget instead of tailing ~155 s past
//! shutdown. Aborted batches are counted dropped exactly once.
//!
//! Config-read discipline (mirrors `webhooks/delivery.rs`): the
//! endpoint/header/service-name snapshot is cloned under the state lock
//! and the lock is dropped before any network I/O — never held across an
//! await.
//!
//! Logging discipline (Req 7.4): rate-limited warns carry the endpoint
//! HOST and error class only — never span content, never header values,
//! never raw transport error text (which can embed URLs, query strings
//! or userinfo).
//!
//! _Requirements: 6.1, 6.2, 6.3, 6.4, 6.5, 6.6, 7.1, 7.3, 7.4_

use std::sync::atomic::{AtomicBool, AtomicI64, AtomicU64, Ordering};
use std::sync::Arc;
use std::time::Duration;

use futures::future::BoxFuture;
use tokio::sync::{mpsc, Mutex, Notify, RwLock, watch};

use crate::metrics::TelemetryMetrics;
use crate::telemetry::config::TelemetryConfig;
use crate::telemetry::otlp::{encode_batch, Resource, SpanData};

/// Queue capacity in individual spans (design D10).
pub const QUEUE_CAPACITY_SPANS: usize = 1024;

/// Maximum spans per delivery batch.
pub const BATCH_MAX_SPANS: usize = 64;

/// Idle interval before the delivery task flushes a partial batch.
pub const BATCH_INTERVAL: Duration = Duration::from_secs(5);

/// Retry schedule (design Component 6): attempt 1 fires immediately,
/// attempts 2-4 follow after 5 s / 30 s / 2 m. The injected sleeper lets
/// tests compress the schedule to zero.
const RETRY_DELAYS: [Duration; 3] = [
    Duration::from_secs(5),
    Duration::from_secs(30),
    Duration::from_secs(120),
];

/// Total per-attempt HTTP timeout (connect + headers + body).
const ATTEMPT_TIMEOUT: Duration = Duration::from_secs(30);

/// Graceful-shutdown flush budget (Error Scenario 6).
const SHUTDOWN_FLUSH_BUDGET: Duration = Duration::from_secs(5);

/// How long `shutdown_flush` waits for the task's flush completion
/// signal before giving up (budget plus scheduling slack).
const SHUTDOWN_JOIN_BUDGET: Duration = Duration::from_secs(6);

/// Minimum spacing between rate-limited drop/failure warnings (Req 7.4).
const WARN_INTERVAL: Duration = Duration::from_secs(30);

/// Marker string the default sender returns when the guarded reqwest
/// client could not be constructed; `classify_transport` maps it to the
/// dedicated `client_build_failed` class (never a fallback client).
const CLIENT_BUILD_FAILED: &str = "otlp client build failed";

/// No-deadline sentinel for the shared shutdown epoch.
const NO_DEADLINE: i64 = i64::MIN;

/// Injectable delay seam: production sleeps via `tokio::time::sleep`;
/// tests inject an instant sleeper to compress the 5 s/30 s/2 m backoff
/// and the shutdown budget (mirrors `webhooks/delivery.rs`).
pub type ExporterSleeper = Arc<dyn Fn(Duration) -> BoxFuture<'static, ()> + Send + Sync>;

/// Injectable clock seam returning UTC epoch seconds (last-error
/// timestamps and shutdown-deadline arithmetic); tests can pin it.
pub type ExporterClock = Arc<dyn Fn() -> i64 + Send + Sync>;

/// Injectable HTTP sender seam: production uses a reqwest client; tests
/// substitute a fake to assert requests without a network.
pub type HttpSender =
Arc<dyn Fn(HttpRequest) -> BoxFuture<'static, Result<u16, String>> + Send + Sync>;

/// A fully-resolved export request snapshot: endpoint URL, header pairs,
/// and the gzip-encoded OTLP body. Header values are resolved once and
/// never logged or exposed.
#[derive(Debug, Clone)]
pub struct HttpRequest {
    pub url: String,
    pub headers: Vec<(String, String)>,
    pub body: Vec<u8>,
}

impl HttpRequest {
    /// Endpoint host for status/logging lines only (Req 7.3: host only).
    pub fn host(&self) -> String {
        host_of(&self.url)
    }
}

/// Classification of the most recent export failure, for the admin
/// status card (Req 7.3). Contains no span content, no header values
/// and no raw transport text — transport failures map onto a fixed
/// allowlisted category set so URL/query/userinfo material from reqwest
/// error strings can never leak into status or logs.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum ExportErrorClass {
    Http(u16),
    Timeout,
    Connect,
    Dns,
    TransportBody,
    TransportOther,
    InvalidEndpoint,
    TaskCrashed,
    ClientBuildFailed,
}

impl ExportErrorClass {
    /// Stable short label for status lines and the admin card.
    pub fn label(&self) -> String {
        match self {
            ExportErrorClass::Http(status) => format!("http_{status}"),
            ExportErrorClass::Timeout => "timeout".to_string(),
            ExportErrorClass::Connect => "connect".to_string(),
            ExportErrorClass::Dns => "dns".to_string(),
            ExportErrorClass::TransportBody => "transport_body".to_string(),
            ExportErrorClass::TransportOther => "transport_other".to_string(),
            ExportErrorClass::InvalidEndpoint => "invalid_endpoint".to_string(),
            ExportErrorClass::TaskCrashed => "task_crashed".to_string(),
            ExportErrorClass::ClientBuildFailed => "client_build_failed".to_string(),
        }
    }
}

/// Immutable config snapshot handed to the delivery task.
#[derive(Debug, Clone)]
struct ExportTarget {
    endpoint: String,
    headers: Vec<(String, String)>,
    service_name: String,
}

impl ExportTarget {
    /// Resolve a target from config; `None` when disabled or unusable
    /// (missing/empty/invalid-scheme endpoint), which keeps the exporter
    /// dormant (Error Scenario 3).
    fn from_config(config: &TelemetryConfig) -> Option<Self> {
        if !config.enabled {
            return None;
        }
        let endpoint = config.endpoint.as_deref()?.trim().to_string();
        if endpoint.is_empty() {
            return None;
        }
        let scheme_ok = reqwest::Url::parse(&endpoint)
            .map(|url| matches!(url.scheme(), "http" | "https"))
            .unwrap_or(false);
        if !scheme_ok {
            return None;
        }
        let mut headers: Vec<(String, String)> = config.resolve_headers().into_iter().collect();
        headers.sort_by(|left, right| left.0.cmp(&right.0));
        Some(ExportTarget {
            endpoint,
            headers,
            service_name: config.service_name.clone(),
        })
    }

    fn build_request(&self, spans: Vec<SpanData>) -> HttpRequest {
        let resource = Resource::new(&self.service_name, env!("CARGO_PKG_VERSION"));
        HttpRequest {
            url: self.endpoint.clone(),
            headers: self.headers.clone(),
            body: encode_batch(spans, &resource),
        }
    }
}

fn host_of(endpoint: &str) -> String {
    reqwest::Url::parse(endpoint)
        .ok()
        .and_then(|url| url.host_str().map(str::to_string))
        .unwrap_or_else(|| "unparseable-host".to_string())
}

/// Map a transport error message onto the fixed allowlisted class set.
/// The raw message is NEVER persisted or exposed — reqwest strings can
/// embed the full request URL, query strings and userinfo credentials.
fn classify_transport(message: &str) -> ExportErrorClass {
    let lowered = message.to_ascii_lowercase();
    if lowered.contains(CLIENT_BUILD_FAILED) {
        ExportErrorClass::ClientBuildFailed
    } else if lowered.contains("timed out") || lowered.contains("timeout") {
        ExportErrorClass::Timeout
    } else if lowered.contains("dns") || lowered.contains("resolve") {
        ExportErrorClass::Dns
    } else if lowered.contains("connect") || lowered.contains("refused") {
        ExportErrorClass::Connect
    } else if lowered.contains("body") || lowered.contains("decode") {
        ExportErrorClass::TransportBody
    } else {
        ExportErrorClass::TransportOther
    }
}

fn default_sleeper() -> ExporterSleeper {
    Arc::new(|delay: Duration| Box::pin(tokio::time::sleep(delay)))
}

fn default_clock() -> ExporterClock {
    Arc::new(|| chrono::Utc::now().timestamp())
}

fn default_sender() -> HttpSender {
    // Built once with redirects disabled and a 30 s total timeout:
    // collectors must answer directly with 2xx (a redirect fails the
    // attempt) and no single attempt may hang past the per-attempt
    // budget. Construction failure NEVER falls back to a default
    // client — a default client follows redirects cross-host with
    // credential headers attached and has no timeout, so every request
    // instead fails fast with the classified `client_build_failed`
    // marker (mirrors the no-fallback client discipline in
    // `webhooks/delivery.rs`).
    static CLIENT: std::sync::OnceLock<Result<reqwest::Client, ()>> = std::sync::OnceLock::new();
    Arc::new(move |request: HttpRequest| {
        Box::pin(async move {
            let client = CLIENT.get_or_init(|| {
                reqwest::Client::builder()
                    .redirect(reqwest::redirect::Policy::none())
                    .timeout(ATTEMPT_TIMEOUT)
                    .build()
                    .map_err(|_| ())
            });
            let client = match client {
                Ok(client) => client,
                Err(()) => {
                    // Host-only warn (Req 7.4): the raw builder error is
                    // never logged or surfaced.
                    tracing::warn!(
                        host = %request.host(),
                        "otlp exporter: http client construction failed"
                    );
                    return Err(CLIENT_BUILD_FAILED.to_string());
                }
            };
            let mut builder = client.post(&request.url).body(request.body);
            for (name, value) in &request.headers {
                builder = builder.header(name, value);
            }
            builder = builder
                .header("Content-Type", "application/json")
                .header("Content-Encoding", "gzip");
            let response = builder.send().await.map_err(|e| e.to_string())?;
            let status = response.status().as_u16();
            // Drain the response so the pooled connection is released;
            // the status code is all the protocol needs.
            let _ = response.bytes().await;
            Ok(status)
        })
    })
}

/// Snapshot of a span batch that a shutdown budget aborted mid-delivery.
/// The batch itself was already counted dropped where the abort was
/// detected — the marker just tells callers to stop retrying without
/// counting anything a second time.
struct ExportAborted;

/// Captured shutdown deadline for one delivery chain. The epoch value is
/// read ONCE per chain so a later `shutdown_flush` returning (and
/// resetting the shared epoch) can never un-abort an in-flight retry
/// loop.
#[derive(Clone, Copy)]
struct ExporterDeadline {
    epoch_seconds: i64,
}

impl ExporterDeadline {
    fn from_epoch(epoch_seconds: i64) -> Self {
        Self { epoch_seconds }
    }

    fn exceeded(&self, now_epoch_seconds: i64) -> bool {
        self.epoch_seconds != NO_DEADLINE && now_epoch_seconds >= self.epoch_seconds
    }
}

/// Snapshot of exporter health for the admin status card (Req 7.3):
/// counters, last error class + timestamp, endpoint HOST only. Never
/// header values, never span content.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ExporterStatus {
    pub enabled: bool,
    pub endpoint_host: Option<String>,
    pub spans_exported: u64,
    pub spans_dropped: u64,
    pub export_failures: u64,
    pub last_error_class: Option<String>,
    pub last_error_at_epoch: Option<i64>,
    pub healthy: bool,
}

/// Last-error state shared between the handle and the delivery task.
#[derive(Default)]
struct LastError {
    class: Mutex<Option<ExportErrorClass>>,
    at_epoch: AtomicU64,
}

impl LastError {
    async fn record(&self, class: ExportErrorClass, clock: &ExporterClock) {
        {
            let mut guard = self.class.lock().await;
            *guard = Some(class);
        }
        self.at_epoch.store(clock().max(0) as u64, Ordering::Relaxed);
    }

    async fn snapshot(&self) -> (Option<String>, Option<i64>) {
        let guard = self.class.lock().await;
        let seconds = self.at_epoch.load(Ordering::Relaxed);
        (
            guard.as_ref().map(ExportErrorClass::label),
            match seconds {
                0 => None,
                s => Some(s as i64),
            },
        )
    }
}

/// State shared between the handle and the delivery task.
struct ExporterTaskShared {
    target: Arc<RwLock<Option<ExportTarget>>>,
    running: Arc<AtomicBool>,
    task_alive: Arc<AtomicBool>,
    clean_exit: Arc<AtomicBool>,
    sender: HttpSender,
    sleeper: ExporterSleeper,
    clock: ExporterClock,
    metrics: Arc<TelemetryMetrics>,
    last_error: Arc<LastError>,
    /// Slot the task returns its receiver to on clean exit.
    receiver_slot: Arc<Mutex<Option<mpsc::Receiver<SpanData>>>>,
    /// Shutdown signal: interrupts retry backoffs and wakes the task
    /// for its bounded final flush.
    wake: Arc<Notify>,
    /// Current queue sender, for sender-alive checks during the final
    /// flush (a live sender means more spans may still arrive).
    queue_tx: Arc<std::sync::RwLock<mpsc::Sender<SpanData>>>,
    /// Active shutdown deadline in epoch seconds (`NO_DEADLINE` when
    /// no shutdown is in progress); captured per delivery chain.
    deadline_epoch: Arc<AtomicI64>,
    /// Completion generation the task bumps as its FINAL observable
    /// action on exit.
    done_tx: Arc<watch::Sender<u64>>,
}

/// Shared exporter core: state, queue, counters, seams.
struct ExporterInner {
    target: Arc<RwLock<Option<ExportTarget>>>,
    running: Arc<AtomicBool>,
    task_alive: Arc<AtomicBool>,
    clean_exit: Arc<AtomicBool>,
    /// Sending half; rebuilt only by crash repair (the old receiver
    /// died with the panicked task).
    queue: std::sync::RwLock<mpsc::Sender<SpanData>>,
    /// Slot the delivery task parks its receiver in on clean exit.
    receiver_slot: Arc<Mutex<Option<mpsc::Receiver<SpanData>>>>,
    metrics: Arc<TelemetryMetrics>,
    sleeper: ExporterSleeper,
    clock: ExporterClock,
    sender: HttpSender,
    last_error: Arc<LastError>,
    /// Rate-limit state for handle-side warns (Req 7.4). Tests pin the
    /// clock, so cross-test interference on this shared epoch only ever
    /// suppresses a warn, never fails an assertion.
    last_warn_at_epoch: AtomicU64,
    /// Wakes the delivery task for the shutdown flush and interrupts
    /// retry backoff sleeps.
    wake: Arc<Notify>,
    /// Completion generation: bumped (as the FINAL observable action)
    /// whenever a delivery task — or its supervisor — finishes, so
    /// `shutdown_flush` waits on a generation captured BEFORE the
    /// liveness check and never extends the wait on a dead task.
    done_tx: Arc<watch::Sender<u64>>,
    done_rx: watch::Receiver<u64>,
    /// Active shutdown deadline (epoch seconds, `NO_DEADLINE` off).
    deadline_epoch: Arc<AtomicI64>,
}

impl ExporterInner {
    fn task_shared(self: &Arc<Self>) -> ExporterTaskShared {
        ExporterTaskShared {
            target: self.target.clone(),
            running: self.running.clone(),
            task_alive: self.task_alive.clone(),
            clean_exit: self.clean_exit.clone(),
            sender: self.sender.clone(),
            sleeper: self.sleeper.clone(),
            clock: self.clock.clone(),
            metrics: self.metrics.clone(),
            last_error: self.last_error.clone(),
            receiver_slot: self.receiver_slot.clone(),
            wake: self.wake.clone(),
            queue_tx: Arc::new(std::sync::RwLock::new(
                self.queue
                    .read()
                    .unwrap_or_else(|poisoned| poisoned.into_inner())
                    .clone(),
            )),
            deadline_epoch: self.deadline_epoch.clone(),
            done_tx: self.done_tx.clone(),
        }
    }
}

/// Cloneable OTLP trace exporter handle (design "Component 6"). Clones
/// share one queue, one delivery task and one set of counters; `AppState`
/// (task 6) holds it next to the other shared managers.
#[derive(Clone)]
pub struct TelemetryExporter {
    inner: Arc<ExporterInner>,
}

impl std::fmt::Debug for TelemetryExporter {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("TelemetryExporter")
            .field("enabled", &self.inner.running.load(Ordering::Relaxed))
            .finish_non_exhaustive()
    }
}

impl TelemetryExporter {
    /// Build a dormant exporter: no background task, no client spin-up,
    /// nothing network-facing until `apply_config` enables the config
    /// (Req 6.6).
    pub fn new(metrics: Arc<TelemetryMetrics>) -> Self {
        let (queue_tx, queue_rx) = mpsc::channel(QUEUE_CAPACITY_SPANS);
        let (done_tx, done_rx) = watch::channel(0u64);
        Self {
            inner: Arc::new(ExporterInner {
                target: Arc::new(RwLock::new(None)),
                running: Arc::new(AtomicBool::new(false)),
                task_alive: Arc::new(AtomicBool::new(false)),
                clean_exit: Arc::new(AtomicBool::new(false)),
                queue: std::sync::RwLock::new(queue_tx),
                receiver_slot: Arc::new(Mutex::new(Some(queue_rx))),
                metrics,
                sleeper: default_sleeper(),
                clock: default_clock(),
                sender: default_sender(),
                last_error: Arc::new(LastError::default()),
                last_warn_at_epoch: AtomicU64::new(0),
                wake: Arc::new(Notify::new()),
                done_tx: Arc::new(done_tx),
                done_rx,
                deadline_epoch: Arc::new(AtomicI64::new(NO_DEADLINE)),
            }),
        }
    }

    /// Injectable sleeper (builder-style, before first clone).
    pub fn with_sleeper(mut self, sleeper: ExporterSleeper) -> Self {
        let inner = Arc::get_mut(&mut self.inner)
            .expect("seams must be injected before cloning the handle");
        inner.sleeper = sleeper;
        self
    }

    /// Injectable clock (builder-style, before first clone).
    pub fn with_clock(mut self, clock: ExporterClock) -> Self {
        let inner = Arc::get_mut(&mut self.inner)
            .expect("seams must be injected before cloning the handle");
        inner.clock = clock;
        self
    }

    /// Injectable HTTP sender (builder-style, before first clone).
    pub fn with_sender(mut self, sender: HttpSender) -> Self {
        let inner = Arc::get_mut(&mut self.inner)
            .expect("seams must be injected before cloning the handle");
        inner.sender = sender;
        self
    }

    /// Apply a telemetry config: enable spawns the single delivery task,
    /// disable stops it, endpoint/header/service-name changes retarget
    /// delivery at the next batch boundary. Idempotent: an already-live
    /// task is never spawned twice (Req 9 hot-reload discipline).
    /// Task-ownership flags (`task_alive`, `clean_exit`) are written
    /// ONLY by the delivery task and its supervisor — this method never
    /// touches them, so an invalid endpoint applied while a task is
    /// live simply parks the target and lets the task exit itself at
    /// its next tick (no second task can ever be spawned under the
    /// live one).
    pub async fn apply_config(&self, config: &TelemetryConfig) {
        let target = ExportTarget::from_config(config);

        if config.enabled && target.is_none() {
            // Enabled but unusable endpoint (Error Scenario 3): stay
            // dormant, record + warn. A live task observes
            // `running == false` at its next timer/send event and
            // exits on its own; task flags stay untouched here.
            self.inner.running.store(false, Ordering::Relaxed);
            let host = host_of(config.endpoint.as_deref().unwrap_or(""));
            self.inner
                .last_error
                .record(ExportErrorClass::InvalidEndpoint, &self.inner.clock)
                .await;
            self.warn_limited(&host, "opentelemetry endpoint invalid; exporter dormant");
            let mut guard = self.inner.target.write().await;
            *guard = None;
            return;
        }

        {
            let mut guard = self.inner.target.write().await;
            *guard = target;
        }

        if config.enabled {
            self.inner.running.store(true, Ordering::Relaxed);
            ensure_task(&self.inner).await;
        } else {
            self.inner.running.store(false, Ordering::Relaxed);
            // The live task observes the flag at its next timer/send
            // event and exits without further POSTs; task_alive is
            // cleared by the task itself on exit.
        }
    }

    /// Accept a span batch whole or drop it whole (design "Component 6"):
    /// `true` when every span was queued; `false` when the batch was
    /// dropped entirely (disabled, queue full, or the channel closed
    /// after a task crash — a partial trace is never enqueued).
    /// Every nonempty rejected batch increments the dropped counter
    /// exactly once here. Non-blocking, panic-free, and never surfaces
    /// export problems to request handling (Req 6.2).
    pub fn try_enqueue(&self, spans: Vec<SpanData>) -> bool {
        if spans.is_empty() {
            return true;
        }
        if !self.inner.running.load(Ordering::Relaxed) {
            // Disabled rejection counts as dropped: the caller offered
            // real spans that will never export (counted here ONLY —
            // callers must not count rejections again).
            self.inner.metrics.record_dropped(spans.len() as u64);
            return false;
        }
        // Supervisor hook (Error Scenario 5): a dead task is re-spawned
        // lazily; the repair itself runs on the async runtime so this
        // path stays sync and non-blocking.
        self.schedule_repair_if_dead();
        let queue = self
            .inner
            .queue
            .read()
            .unwrap_or_else(|poisoned| poisoned.into_inner());
        let outcome = match queue.try_reserve_many(spans.len()) {
            Ok(permits) => {
                // Re-check enabled under the reserved capacity: a
                // concurrent apply_config(disable) may have flipped the
                // flag between the first check and now.
                if !self.inner.running.load(Ordering::Relaxed) {
                    drop(permits);
                    self.inner.metrics.record_dropped(spans.len() as u64);
                    return false;
                }
                for (permit, span) in permits.zip(spans) {
                    permit.send(span);
                }
                true
            }
            Err(error) => {
                // Whole-batch drop + counter (Error Scenario 1); a
                // closed channel means the task crashed holding the
                // receiver — the scheduled repair rebuilds it.
                self.inner.metrics.record_dropped(spans.len() as u64);
                if matches!(error, mpsc::error::TrySendError::Closed(())) {
                    self.schedule_repair_if_dead();
                } else {
                    self.warn_limited_sync("queue full; span batch dropped");
                }
                false
            }
        };
        outcome
    }

    /// Graceful shutdown flush (Req 6.4, Error Scenario 6): signal the
    /// delivery task to drain queued spans within a bounded 5 s budget;
    /// leftovers are dropped and counted. The published deadline epoch
    /// also aborts any in-flight retry chain at the budget, so shutdown
    /// never tails ~155 s behind a 5 s/30 s/2 m backoff. If no task is
    /// running, an inline bounded flush drains what the (parked)
    /// channel holds. Safe to call repeatedly and on a dormant exporter.
    pub async fn shutdown_flush(&self) {
        self.inner.running.store(false, Ordering::Relaxed);
        // Capture the completion generation BEFORE the liveness check:
        // a task finishing in between still bumps this generation (the
        // bump is the task's final observable action), so the wait
        // below can never block on an already-dead generation.
        let observed = *self.inner.done_rx.borrow();
        let task_was_alive = self.inner.task_alive.load(Ordering::Relaxed);
        let budget_end = (self.inner.clock)() + SHUTDOWN_FLUSH_BUDGET.as_secs() as i64;
        self.inner.deadline_epoch.store(budget_end, Ordering::SeqCst);
        if task_was_alive {
            let mut done = self.inner.done_rx.clone();
            self.inner.wake.notify_one();
            // Wait for this task's generation to complete (clean exit
            // after the bounded flush, or the supervisor's crash
            // housekeeping), bounded by the join budget.
            let _ = tokio::time::timeout(SHUTDOWN_JOIN_BUDGET, async move {
                loop {
                    if *done.borrow() > observed {
                        return;
                    }
                    if done.changed().await.is_err() {
                        return;
                    }
                }
            })
            .await;
            self.inner.deadline_epoch.store(NO_DEADLINE, Ordering::SeqCst);
            return;
        }
        // No live task: drain the parked channel inline, bounded.
        let receiver = {
            let mut guard = self.inner.receiver_slot.lock().await;
            guard.take()
        };
        let Some(mut receiver) = receiver else {
            self.inner.deadline_epoch.store(NO_DEADLINE, Ordering::SeqCst);
            return;
        };
        let shared = self.inner.task_shared();
        let mut pending: Vec<SpanData> = Vec::with_capacity(BATCH_MAX_SPANS);
        bounded_final_flush(&shared, &mut receiver, &mut pending).await;
        self.inner.deadline_epoch.store(NO_DEADLINE, Ordering::SeqCst);
        // Return the receiver so a later re-enable can reuse the channel.
        let mut guard = self.inner.receiver_slot.lock().await;
        if guard.is_none() {
            *guard = Some(receiver);
        }
    }

    /// Point-in-time health/status snapshot (Req 7.3): counters, last
    /// error class + timestamp, endpoint HOST only. Header values are
    /// never part of the snapshot.
    pub async fn status(&self) -> ExporterStatus {
        let target = self.inner.target.read().await.clone();
        let (last_error_class, last_error_at_epoch) = self.inner.last_error.snapshot().await;
        let enabled = self.inner.running.load(Ordering::Relaxed);
        ExporterStatus {
            enabled,
            endpoint_host: target.as_ref().map(|t| host_of(&t.endpoint)),
            spans_exported: self.inner.metrics.exported(),
            spans_dropped: self.inner.metrics.dropped(),
            export_failures: self.inner.metrics.failures(),
            last_error_class,
            last_error_at_epoch,
            healthy: !enabled || self.inner.task_alive.load(Ordering::Relaxed),
        }
    }

    /// Shared counter handle (Prometheus rendering reads the same
    /// atomics the exporter writes).
    pub fn metrics(&self) -> Arc<TelemetryMetrics> {
        self.inner.metrics.clone()
    }

    /// Supervisor (Error Scenario 5): if the delivery task died while
    /// enabled, spawn the repair on the current runtime (fire-and-forget
    /// so `try_enqueue` stays sync). Off-runtime calls skip the repair;
    /// the next `apply_config` heals instead.
    fn schedule_repair_if_dead(&self) {
        if !self.inner.running.load(Ordering::Relaxed)
            || self.inner.task_alive.load(Ordering::Relaxed)
        {
            return;
        }
        if let Ok(handle) = tokio::runtime::Handle::try_current() {
            let inner = self.inner.clone();
            handle.spawn(async move {
                ensure_task(&inner).await;
            });
        }
    }

    fn warn_limited(&self, host: &str, message: &str) {
        let now = (self.inner.clock)().max(0) as u64;
        let last = self.inner.last_warn_at_epoch.load(Ordering::Relaxed);
        if now.saturating_sub(last) < WARN_INTERVAL.as_secs() {
            return;
        }
        self.inner.last_warn_at_epoch.store(now, Ordering::Relaxed);
        tracing::warn!(host = %host, "otlp exporter: {message}");
    }

    fn warn_limited_sync(&self, message: &str) {
        let now = (self.inner.clock)().max(0) as u64;
        let last = self.inner.last_warn_at_epoch.load(Ordering::Relaxed);
        if now.saturating_sub(last) < WARN_INTERVAL.as_secs() {
            return;
        }
        self.inner.last_warn_at_epoch.store(now, Ordering::Relaxed);
        tracing::warn!("otlp exporter: {message}");
    }
}

/// Spawn the single delivery task if enabled and not alive — the ONLY
/// spawner, idempotent under the receiver-slot lock. A missing receiver
/// with a dead task means the previous task panicked holding the
/// receiver: rebuild the channel and re-spawn (Error Scenario 5). The
/// crash itself is counted ONCE, by the supervisor that joined the
/// panicked task — the rebuild here stays silent so `export_failures`
/// is never double-counted.
async fn ensure_task(inner: &Arc<ExporterInner>) {
    let mut slot = inner.receiver_slot.lock().await;
    if inner.task_alive.load(Ordering::Relaxed) {
        return;
    }
    let receiver = match slot.take() {
        Some(receiver) => receiver,
        None => {
            // Crash recovery: the old receiver died with the task, so
            // the channel (and any spans parked in it) is unrecoverable.
            // The failure was already counted by the supervisor; this
            // rebuild only restores the channel.
            let (tx, rx) = mpsc::channel(QUEUE_CAPACITY_SPANS);
            {
                let mut queue = inner
                    .queue
                    .write()
                    .unwrap_or_else(|poisoned| poisoned.into_inner());
                *queue = tx;
            }
            rx
        }
    };
    inner.clean_exit.store(false, Ordering::Relaxed);
    inner.task_alive.store(true, Ordering::Relaxed);
    let shared = inner.task_shared();
    let handle = tokio::spawn(delivery_task(shared, receiver));
    // Supervisor (Error Scenario 5): join the delivery task so a panic
    // is still observed and housekeeping runs. The supervisor is the
    // single crash counter.
    let supervisor_inner = inner.clone();
    tokio::spawn(async move {
        let join = handle.await;
        let clean = matches!(join, Ok(())) && supervisor_inner.clean_exit.load(Ordering::Relaxed);
        supervisor_inner.task_alive.store(false, Ordering::Relaxed);
        if !clean {
            // Crashed (Error Scenario 5): mark unhealthy, count the
            // failure (single owner); the receiver is gone with the
            // task, so the next ensure_task call rebuilds the channel
            // without counting again.
            supervisor_inner.metrics.record_failure();
            supervisor_inner
                .last_error
                .record(ExportErrorClass::TaskCrashed, &supervisor_inner.clock)
                .await;
        }
        // Bump the completion generation LAST so a pending
        // shutdown_flush observes the task's exit only after every
        // other observable effect (receiver handoff, counters) has
        // settled.
        supervisor_inner.done_tx.send_modify(|gen| *gen += 1);
    });
}

/// The single background delivery task: batches up to
/// [`BATCH_MAX_SPANS`] spans, flushes a partial batch after
/// [`BATCH_INTERVAL`] of idleness, performs the bounded final flush on
/// shutdown, and exits without further POSTs once disabled.
async fn delivery_task(shared: ExporterTaskShared, mut receiver: mpsc::Receiver<SpanData>) {
    let mut pending: Vec<SpanData> = Vec::with_capacity(BATCH_MAX_SPANS);
    'main: loop {
        tokio::select! {
            biased;
            _ = shared.wake.notified() => {
                // Shutdown (Req 6.4): bounded final flush, then exit.
                bounded_final_flush(&shared, &mut receiver, &mut pending).await;
                break 'main;
            }
            _ = tokio::time::sleep(BATCH_INTERVAL) => {
                if !shared.running.load(Ordering::Relaxed) {
                    break 'main;
                }
                if !pending.is_empty() {
                    let deadline = ExporterDeadline::from_epoch(
                        shared.deadline_epoch.load(Ordering::SeqCst),
                    );
                    if deliver_batch(&shared, &mut pending, &deadline).await.is_err() {
                        break 'main;
                    }
                }
                if !shared.running.load(Ordering::Relaxed) {
                    break 'main;
                }
            }
            maybe = receiver.recv() => {
                match maybe {
                    Some(span) => {
                        pending.push(span);
                        if pending.len() >= BATCH_MAX_SPANS {
                            if shared.running.load(Ordering::Relaxed) {
                                let deadline = ExporterDeadline::from_epoch(
                                    shared.deadline_epoch.load(Ordering::SeqCst),
                                );
                                if deliver_batch(&shared, &mut pending, &deadline).await.is_err() {
                                    break 'main;
                                }
                            }
                        }
                    }
                    None => break 'main,
                }
            }
        }
    }
    // Exit housekeeping: spans parked in `pending` on a no-flush exit
    // were accepted but will never deliver — count them, never silently.
    if !pending.is_empty() {
        shared.metrics.record_dropped(pending.len() as u64);
        warn_limited_static(&shared.clock, "unknown-host", "exporter stopped with pending spans; dropped");
    }
    shared.clean_exit.store(true, Ordering::Relaxed);
    // Publish the receiver FIRST: once the slot is filled, a concurrent
    // re-enable's ensure_task reuses the channel. Only then clear
    // task_alive (so the liveness flag never points at a task whose
    // receiver is still unpublished), and bump the completion
    // generation as the FINAL observable action.
    let mut slot = shared.receiver_slot.lock().await;
    if slot.is_none() {
        *slot = Some(receiver);
    } else {
        let mut orphaned: u64 = 0;
        while receiver.try_recv().is_ok() {
            orphaned += 1;
        }
        if orphaned > 0 {
            shared.metrics.record_dropped(orphaned);
        }
    }
    drop(slot);
    shared.task_alive.store(false, Ordering::Relaxed);
    shared.done_tx.send_modify(|gen| *gen += 1);
}

/// Bounded final flush (Error Scenario 6): drain the channel and deliver
/// what fits within [`SHUTDOWN_FLUSH_BUDGET`]; leftover spans are
/// dropped and counted. Runs inside the delivery task on shutdown, or
/// inline from `shutdown_flush` when no task is alive.
///
/// The already-queued partial batch is delivered IMMEDIATELY (drained
/// via `try_recv` before any waiting) so a deliverable batch never
/// burns the budget waiting for spans that never arrive; the remaining
/// budget is only spent waiting when the sender half is still alive
/// (more spans may still be in flight).
async fn bounded_final_flush(
    shared: &ExporterTaskShared,
    receiver: &mut mpsc::Receiver<SpanData>,
    pending: &mut Vec<SpanData>,
) {
    let deadline = ExporterDeadline::from_epoch(shared.deadline_epoch.load(Ordering::SeqCst));
    // Drain whatever is already queued before touching the clock.
    loop {
        match receiver.try_recv() {
            Ok(span) => {
                pending.push(span);
                if pending.len() >= BATCH_MAX_SPANS {
                    break;
                }
            }
            Err(_) => break,
        }
    }
    if !pending.is_empty() {
        // Deliver the partial batch at once; an abort counted the
        // drained spans dropped inside deliver_batch.
        let _ = deliver_batch(shared, pending, &deadline).await;
    }
    // Only wait for MORE spans when the sender is still alive; a dead
    // sender means the drained batch was everything.
    let sender_alive = !shared
        .queue_tx
        .read()
        .unwrap_or_else(|poisoned| poisoned.into_inner())
        .is_closed();
    if sender_alive {
        loop {
            let remaining = match deadline_remaining(&deadline, shared).checked_sub(Duration::ZERO) {
                Some(remaining) => remaining,
                None => break,
            };
            if remaining.is_zero() {
                break;
            }
            match tokio::time::timeout(remaining, receiver.recv()).await {
                Ok(Some(span)) => {
                    pending.push(span);
                    if pending.len() >= BATCH_MAX_SPANS {
                        if deliver_batch(shared, pending, &deadline).await.is_err() {
                            break;
                        }
                    }
                }
                Ok(None) => break,
                Err(_elapsed) => break,
            }
        }
        if !pending.is_empty() {
            let _ = deliver_batch(shared, pending, &deadline).await;
        }
    }
    // Anything still unreachable in the channel counts as dropped.
    let mut leftover: u64 = 0;
    while receiver.try_recv().is_ok() {
        leftover += 1;
    }
    if leftover > 0 {
        shared.metrics.record_dropped(leftover);
        warn_limited_static(&shared.clock, "unknown-host", "shutdown flush overflow; spans dropped");
    }
}

/// Remaining wall budget for a final-flush wait. The shared deadline is
/// second-granularity (clock seam); without an active deadline the
/// flush never waits at all.
fn deadline_remaining(deadline: &ExporterDeadline, shared: &ExporterTaskShared) -> Duration {
    if deadline.epoch_seconds == NO_DEADLINE {
        return Duration::ZERO;
    }
    let now_epoch = (shared.clock)();
    if now_epoch >= deadline.epoch_seconds {
        return Duration::ZERO;
    }
    let seconds_left = (deadline.epoch_seconds - now_epoch) as u64;
    // Keep a floor of one millisecond so the timeout still parks the
    // task (a zero timeout would spin); the epoch check above bounds
    // the real wait.
    Duration::from_millis((seconds_left * 1000).max(1))
}

/// Deliver `pending` with the full retry schedule (5 s/30 s/2 m, then
/// drop + failure counter). Splits at [`BATCH_MAX_SPANS`] so an
/// oversized batch still posts in bounded chunks.
///
/// Deadline-aware (Error Scenario 6): when a shutdown budget is active,
/// every attempt and every backoff sleep checks/aborts at the deadline
/// (the sleep races the shutdown wake signal), so an in-flight chain
/// never tails past the budget. A batch aborted mid-chain — including
/// spans already drained from `pending` — is counted dropped EXACTLY
/// once, right here; the caller must not count again.
async fn deliver_batch(
    shared: &ExporterTaskShared,
    pending: &mut Vec<SpanData>,
    deadline: &ExporterDeadline,
) -> Result<(), ExportAborted> {
    while !pending.is_empty() {
        let target = shared.target.read().await.clone();
        let Some(target) = target else {
            // Disabled/retargeted mid-flight: drop + count without
            // retries (never surface to requests).
            shared.metrics.record_dropped(pending.len() as u64);
            shared.metrics.record_failure();
            shared
                .last_error
                .record(ExportErrorClass::InvalidEndpoint, &shared.clock)
                .await;
            pending.clear();
            return Ok(());
        };

        let take = pending.len().min(BATCH_MAX_SPANS);
        let batch: Vec<SpanData> = pending.drain(..take).collect();
        let span_count = batch.len() as u64;
        let request = target.build_request(batch);
        let host = request.host();

        let mut attempt = 0usize;
        loop {
            if deadline.exceeded((shared.clock)()) {
                // Shutdown budget reached: the drained batch counts
                // dropped once, here; the caller sees the abort marker.
                shared.metrics.record_dropped(span_count);
                warn_limited_static(
                    &shared.clock,
                    &host,
                    "export aborted at shutdown budget; batch dropped",
                );
                return Err(ExportAborted);
            }
            let result = (shared.sender)(request.clone()).await;
            let failed = match result {
                Ok(status) if (200..300).contains(&status) => {
                    shared.metrics.record_exported(span_count);
                    false
                }
                Ok(status) => {
                    shared
                        .last_error
                        .record(ExportErrorClass::Http(status), &shared.clock)
                        .await;
                    true
                }
                Err(message) => {
                    shared
                        .last_error
                        .record(classify_transport(&message), &shared.clock)
                        .await;
                    true
                }
            };
            if !failed {
                break;
            }
            if attempt >= RETRY_DELAYS.len() {
                // Retry exhaustion (Error Scenario 2): drop the
                // batch, count one failure.
                shared.metrics.record_failure();
                shared.metrics.record_dropped(span_count);
                warn_limited_static(
                    &shared.clock,
                    &host,
                    "export failed after 4 attempts; batch dropped",
                );
                break;
            }
            if deadline.exceeded((shared.clock)()) {
                shared.metrics.record_dropped(span_count);
                warn_limited_static(
                    &shared.clock,
                    &host,
                    "export aborted at shutdown budget; batch dropped",
                );
                return Err(ExportAborted);
            }
            // Deadline-aware backoff: a shutdown wake interrupts the
            // sleep (biased so a pending wake always wins the race).
            let notified = shared.wake.notified();
            tokio::pin!(notified);
            tokio::select! {
                biased;
                _ = &mut notified => {
                    shared.metrics.record_dropped(span_count);
                    warn_limited_static(
                        &shared.clock,
                        &host,
                        "export aborted at shutdown budget; batch dropped",
                    );
                    return Err(ExportAborted);
                }
                _ = (shared.sleeper)(RETRY_DELAYS[attempt]) => {}
            }
            attempt += 1;
        }
    }
    Ok(())
}

/// Rate-limited warn shared with the delivery task (Req 7.4): endpoint
/// HOST and outcome only — no span content, no header values, no raw
/// transport error text.
fn warn_limited_static(clock: &ExporterClock, host: &str, message: &str) {
    static LAST_WARN: AtomicU64 = AtomicU64::new(0);
    let now = (clock)().max(0) as u64;
    let last = LAST_WARN.load(Ordering::Relaxed);
    if now.saturating_sub(last) < WARN_INTERVAL.as_secs() {
        return;
    }
    if LAST_WARN
        .compare_exchange(last, now, Ordering::Relaxed, Ordering::Relaxed)
        .is_err()
    {
        return;
    }
    tracing::warn!(host = %host, "otlp exporter: {message}");
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::sync::atomic::AtomicUsize;

    // Tests share the process-wide LAST_WARN rate-limit epoch. Every
    // test pins its clock to a constant, so cross-test interference can
    // only ever SUPPRESS a warn (never re-fires one unexpectedly) and
    // no assertion below depends on warn emission — the suite stays
    // deterministic under RUST_TEST_THREADS=2.

    fn instant_sleeper() -> ExporterSleeper {
        Arc::new(|_delay: Duration| Box::pin(async {}))
    }

    fn pinned_clock() -> ExporterClock {
        Arc::new(|| 1_700_000_000)
    }

    fn span(name: &str) -> SpanData {
        SpanData {
            trace_id: [1u8; 16],
            span_id: [2u8; 8],
            parent_span_id: None,
            name: name.to_string(),
            kind: crate::telemetry::otlp::SpanKind::Internal,
            status: None,
            start_time: std::time::SystemTime::UNIX_EPOCH,
            end_time: std::time::SystemTime::UNIX_EPOCH,
            attributes: Vec::new(),
        }
    }

    fn spans(count: usize) -> Vec<SpanData> {
        (0..count).map(|i| span(&format!("span-{i}"))).collect()
    }

    fn telemetry_config(endpoint: &str) -> TelemetryConfig {
        serde_yaml::from_str(&format!(
            "enabled: true\nendpoint: {endpoint}\nservice_name: test-svc\n"
        ))
        .unwrap()
    }

    fn invalid_endpoint_config() -> TelemetryConfig {
        serde_yaml::from_str("enabled: true\nendpoint: \"\"\n").unwrap()
    }

    /// Fake sender recording every call and answering a scripted status
    /// sequence: entries are consumed in order, the LAST entry repeats
    /// forever, and an empty script always answers 200.
    #[derive(Clone)]
    struct FakeSender {
        calls: Arc<AtomicUsize>,
        statuses: Arc<std::sync::Mutex<Vec<Result<u16, String>>>>,
        urls_seen: Arc<std::sync::Mutex<Vec<String>>>,
    }

    impl FakeSender {
        fn new(statuses: Vec<Result<u16, String>>) -> Self {
            Self {
                calls: Arc::new(AtomicUsize::new(0)),
                statuses: Arc::new(std::sync::Mutex::new(statuses)),
                urls_seen: Arc::new(std::sync::Mutex::new(Vec::new())),
            }
        }

        fn to_sender(&self) -> HttpSender {
            let calls = self.calls.clone();
            let statuses = self.statuses.clone();
            let urls_seen = self.urls_seen.clone();
            Arc::new(move |request: HttpRequest| {
                calls.fetch_add(1, Ordering::SeqCst);
                urls_seen.lock().unwrap().push(request.url.clone());
                let next = {
                    let mut guard = statuses.lock().unwrap();
                    if guard.len() > 1 {
                        guard.remove(0)
                    } else if guard.len() == 1 {
                        guard[0].clone()
                    } else {
                        Ok(200)
                    }
                };
                Box::pin(async move { next })
            })
        }
    }

    async fn wait_until(predicate: impl Fn() -> bool) {
        for _ in 0..1000 {
            if predicate() {
                return;
            }
            tokio::time::sleep(Duration::from_millis(10)).await;
        }
        panic!("condition not reached within wait window");
    }

    #[tokio::test]
    async fn queue_full_drops_whole_batch_and_counts() {
        let metrics = Arc::new(TelemetryMetrics::new());
        let exporter = TelemetryExporter::new(metrics.clone())
            .with_sleeper(instant_sleeper())
            .with_clock(pinned_clock())
            .with_sender(FakeSender::new(Vec::new()).to_sender());

        // Dormant: nothing accepted, and the rejected spans count as
        // dropped exactly once (F10 discipline).
        assert!(!exporter.try_enqueue(spans(4)));
        assert_eq!(metrics.dropped(), 4);

        exporter
            .apply_config(&telemetry_config("http://collector.internal/v1/traces"))
            .await;

        // Fill the queue to capacity (unit = individual spans).
        assert!(exporter.try_enqueue(spans(QUEUE_CAPACITY_SPANS)));
        // Whole-batch semantics: an overflowing batch is fully dropped
        // and counted.
        assert!(!exporter.try_enqueue(spans(8)));
        assert_eq!(metrics.dropped(), 12);
        assert!(!exporter.try_enqueue(spans(1)));
        assert_eq!(metrics.dropped(), 13);
        assert_eq!(metrics.exported(), 0);
    }

    #[tokio::test(start_paused = true)]
    async fn retry_exhaustion_counts_one_failure_per_batch() {
        let metrics = Arc::new(TelemetryMetrics::new());
        let fake = FakeSender::new(vec![Ok(500), Ok(500), Ok(500), Ok(500)]);
        let exporter = TelemetryExporter::new(metrics.clone())
            .with_sleeper(instant_sleeper())
            .with_clock(pinned_clock())
            .with_sender(fake.to_sender());

        exporter
            .apply_config(&telemetry_config("http://collector.internal/v1/traces"))
            .await;
        assert!(exporter.try_enqueue(spans(3)));

        wait_until(|| metrics.failures() == 1).await;
        // Initial attempt + 3 retries = 4 calls for one batch.
        assert_eq!(fake.calls.load(Ordering::SeqCst), 4);
        assert_eq!(metrics.exported(), 0);
        assert_eq!(metrics.dropped(), 3);
        let status = exporter.status().await;
        assert_eq!(status.last_error_class.as_deref(), Some("http_500"));
        assert!(status.last_error_at_epoch.is_some());
    }

    #[tokio::test(start_paused = true)]
    async fn two_hundred_counts_exported_span_count() {
        let metrics = Arc::new(TelemetryMetrics::new());
        let fake = FakeSender::new(Vec::new());
        let exporter = TelemetryExporter::new(metrics.clone())
            .with_sleeper(instant_sleeper())
            .with_clock(pinned_clock())
            .with_sender(fake.to_sender());

        exporter
            .apply_config(&telemetry_config("http://collector.internal/v1/traces"))
            .await;
        assert!(exporter.try_enqueue(spans(5)));

        wait_until(|| metrics.exported() == 5).await;
        assert_eq!(fake.calls.load(Ordering::SeqCst), 1);
        assert_eq!(metrics.dropped(), 0);
        assert_eq!(metrics.failures(), 0);
        let status = exporter.status().await;
        assert!(status.healthy);
        assert_eq!(status.endpoint_host.as_deref(), Some("collector.internal"));
    }

    #[tokio::test(start_paused = true)]
    async fn transport_errors_classify_and_exhaust() {
        let metrics = Arc::new(TelemetryMetrics::new());
        let fake = FakeSender::new(vec![
            Err("connection refused".to_string()),
            Err("connection refused".to_string()),
            Err("connection refused".to_string()),
            Err("operation timed out".to_string()),
        ]);
        let exporter = TelemetryExporter::new(metrics.clone())
            .with_sleeper(instant_sleeper())
            .with_clock(pinned_clock())
            .with_sender(fake.to_sender());

        exporter
            .apply_config(&telemetry_config("http://collector.internal/v1/traces"))
            .await;
        assert!(exporter.try_enqueue(spans(2)));

        wait_until(|| metrics.failures() == 1).await;
        assert_eq!(fake.calls.load(Ordering::SeqCst), 4);
        // The last observed error class wins the status slot.
        let status = exporter.status().await;
        assert_eq!(status.last_error_class.as_deref(), Some("timeout"));
    }

    #[tokio::test(start_paused = true)]
    async fn shutdown_flush_delivers_queued_spans_within_budget() {
        let metrics = Arc::new(TelemetryMetrics::new());
        let fake = FakeSender::new(Vec::new());
        let exporter = TelemetryExporter::new(metrics.clone())
            .with_sleeper(instant_sleeper())
            .with_clock(pinned_clock())
            .with_sender(fake.to_sender());

        exporter
            .apply_config(&telemetry_config("http://collector.internal/v1/traces"))
            .await;
        exporter.try_enqueue(spans(10));
        exporter.shutdown_flush().await;

        // The queued partial batch is delivered immediately by the
        // try_recv drain (F4) — not dropped after burning the budget.
        assert_eq!(metrics.exported(), 10);
        assert_eq!(metrics.dropped(), 0);
        assert!(fake.calls.load(Ordering::SeqCst) >= 1);
    }

    #[tokio::test(start_paused = true)]
    async fn shutdown_flush_overflow_drops_remainder() {
        let metrics = Arc::new(TelemetryMetrics::new());
        // Never-succeeding sender: the shutdown budget aborts the
        // chain, and span conservation is what the bounded budget
        // guarantees.
        let fake = FakeSender::new(vec![Err("operation timed out".to_string())]);
        let exporter = TelemetryExporter::new(metrics.clone())
            .with_sleeper(instant_sleeper())
            .with_clock(pinned_clock())
            .with_sender(fake.to_sender());

        exporter
            .apply_config(&telemetry_config("http://collector.internal/v1/traces"))
            .await;
        exporter.try_enqueue(spans(6));
        exporter.shutdown_flush().await;

        let status = exporter.status().await;
        assert!(!status.enabled);
        // Conservation: every queued span is either exported or
        // dropped, exactly once.
        assert_eq!(metrics.exported() + metrics.dropped(), 6);
        assert_eq!(metrics.dropped(), 6);
    }

    #[tokio::test(start_paused = true)]
    async fn apply_config_enable_disable_and_retarget() {
        let metrics = Arc::new(TelemetryMetrics::new());
        let fake = FakeSender::new(Vec::new());
        let exporter = TelemetryExporter::new(metrics.clone())
            .with_sleeper(instant_sleeper())
            .with_clock(pinned_clock())
            .with_sender(fake.to_sender());

        // Dormant by default: no task, no calls, no host.
        let status = exporter.status().await;
        assert!(!status.enabled);
        assert_eq!(status.endpoint_host, None);
        assert_eq!(fake.calls.load(Ordering::SeqCst), 0);

        // Enable: task spawns, batch delivers.
        exporter
            .apply_config(&telemetry_config("http://first.internal/v1/traces"))
            .await;
        assert!(exporter.try_enqueue(spans(1)));
        wait_until(|| fake.calls.load(Ordering::SeqCst) >= 1).await;
        assert_eq!(
            exporter.status().await.endpoint_host.as_deref(),
            Some("first.internal")
        );

        // Disable: enqueue refuses (and counts the rejection), no
        // further POSTs.
        exporter.apply_config(&TelemetryConfig::default()).await;
        let calls_after_disable = fake.calls.load(Ordering::SeqCst);
        let dropped_after_disable = metrics.dropped();
        assert!(!exporter.try_enqueue(spans(1)));
        assert_eq!(metrics.dropped(), dropped_after_disable + 1);
        tokio::time::sleep(Duration::from_millis(50)).await;
        assert_eq!(fake.calls.load(Ordering::SeqCst), calls_after_disable);

        // Retarget: new endpoint host shows in status and next request.
        exporter
            .apply_config(&telemetry_config("http://second.internal/v1/traces"))
            .await;
        assert!(exporter.try_enqueue(spans(1)));
        wait_until(|| fake.calls.load(Ordering::SeqCst) > calls_after_disable).await;
        let status = exporter.status().await;
        assert_eq!(status.endpoint_host.as_deref(), Some("second.internal"));
        let urls = fake.urls_seen.lock().unwrap();
        assert!(urls
            .iter()
            .any(|url| url == "http://second.internal/v1/traces"));
    }

    #[tokio::test]
    async fn prometheus_render_contains_all_three_counters() {
        let metrics = Arc::new(TelemetryMetrics::new());
        metrics.record_exported(3);
        metrics.record_dropped(1);
        metrics.record_failure();

        let mut out = String::new();
        metrics.write_prometheus(&mut out);

        assert!(out.contains("otlp_spans_exported_total 3\n"));
        assert!(out.contains("otlp_spans_dropped_total 1\n"));
        assert!(out.contains("otlp_export_failures_total 1\n"));
        assert!(out.contains("# TYPE otlp_spans_exported_total counter"));
    }

    #[tokio::test]
    async fn disabled_config_keeps_exporter_dormant() {
        let metrics = Arc::new(TelemetryMetrics::new());
        let fake = FakeSender::new(Vec::new());
        let exporter = TelemetryExporter::new(metrics.clone())
            .with_sleeper(instant_sleeper())
            .with_clock(pinned_clock())
            .with_sender(fake.to_sender());

        exporter.apply_config(&TelemetryConfig::default()).await;
        assert!(!exporter.try_enqueue(spans(2)));
        tokio::time::sleep(Duration::from_millis(30)).await;
        assert_eq!(fake.calls.load(Ordering::SeqCst), 0);
        // The rejected batch counts dropped exactly once.
        assert_eq!(metrics.dropped(), 2);
        assert_eq!(metrics.exported(), 0);
    }

    #[tokio::test]
    async fn invalid_endpoint_stays_dormant_and_records_error() {
        let metrics = Arc::new(TelemetryMetrics::new());
        let fake = FakeSender::new(Vec::new());
        let exporter = TelemetryExporter::new(metrics.clone())
            .with_sleeper(instant_sleeper())
            .with_clock(pinned_clock())
            .with_sender(fake.to_sender());

        let config = invalid_endpoint_config();
        exporter.apply_config(&config).await;

        assert!(!exporter.try_enqueue(spans(1)));
        assert_eq!(metrics.dropped(), 1);
        let status = exporter.status().await;
        assert!(!status.enabled);
        assert_eq!(
            status.last_error_class.as_deref(),
            Some("invalid_endpoint")
        );
        assert_eq!(fake.calls.load(Ordering::SeqCst), 0);
    }

    #[tokio::test(start_paused = true)]
    async fn invalid_endpoint_while_live_never_double_spawns() {
        let metrics = Arc::new(TelemetryMetrics::new());
        let fake = FakeSender::new(Vec::new());
        let exporter = TelemetryExporter::new(metrics.clone())
            .with_sleeper(instant_sleeper())
            .with_clock(pinned_clock())
            .with_sender(fake.to_sender());

        exporter
            .apply_config(&telemetry_config("http://collector.internal/v1/traces"))
            .await;
        assert!(exporter.try_enqueue(spans(2)));
        wait_until(|| metrics.exported() == 2).await;

        // Invalid endpoint applied while the task is live (F1): the
        // liveness flag is NOT forcibly cleared — the task exits on
        // its own at the next tick.
        exporter.apply_config(&invalid_endpoint_config()).await;
        assert!(exporter.inner.task_alive.load(Ordering::Relaxed));
        let status = exporter.status().await;
        assert!(!status.enabled);
        assert_eq!(
            status.last_error_class.as_deref(),
            Some("invalid_endpoint")
        );
        wait_until(|| !exporter.inner.task_alive.load(Ordering::Relaxed)).await;
        // The clean exit returned the receiver to the slot.
        assert!(exporter.inner.receiver_slot.lock().await.is_some());

        // Re-enable: the parked receiver is reused — no crash
        // misclassification, no failure count, no second channel.
        exporter
            .apply_config(&telemetry_config("http://second.internal/v1/traces"))
            .await;
        assert!(exporter.try_enqueue(spans(3)));
        wait_until(|| metrics.exported() == 5).await;
        assert_eq!(metrics.failures(), 0);
        assert_eq!(metrics.dropped(), 0);
    }

    #[tokio::test(start_paused = true)]
    async fn clean_exit_returns_receiver_and_reenable_reuses_it() {
        let metrics = Arc::new(TelemetryMetrics::new());
        let fake = FakeSender::new(Vec::new());
        let exporter = TelemetryExporter::new(metrics.clone())
            .with_sleeper(instant_sleeper())
            .with_clock(pinned_clock())
            .with_sender(fake.to_sender());

        exporter
            .apply_config(&telemetry_config("http://collector.internal/v1/traces"))
            .await;
        assert!(exporter.try_enqueue(spans(1)));
        wait_until(|| metrics.exported() == 1).await;

        // Disable: the task performs its exit housekeeping — receiver
        // published to the slot BEFORE task_alive clears and BEFORE
        // the done-generation bump (F2 ordering).
        exporter.apply_config(&TelemetryConfig::default()).await;
        wait_until(|| !exporter.inner.task_alive.load(Ordering::Relaxed)).await;
        assert!(exporter.inner.receiver_slot.lock().await.is_some());
        // The clean exit must not be misclassified as a crash.
        assert_eq!(metrics.failures(), 0);

        // Inline shutdown path borrows and returns the same receiver.
        exporter.shutdown_flush().await;
        assert!(exporter.inner.receiver_slot.lock().await.is_some());

        // Re-enable reuses the channel and keeps delivering.
        exporter
            .apply_config(&telemetry_config("http://collector.internal/v1/traces"))
            .await;
        assert!(exporter.try_enqueue(spans(2)));
        wait_until(|| metrics.exported() == 3).await;
        assert_eq!(metrics.failures(), 0);
    }

    #[tokio::test(start_paused = true)]
    async fn shutdown_aborts_long_retry_chain_within_budget() {
        let metrics = Arc::new(TelemetryMetrics::new());
        // Real (time-compressed-by-pause) sleeper: the chain must be
        // sitting in a 5 s backoff when shutdown fires, and the wake
        // signal — not the elapsed 30 s/2 m delays — ends it.
        let fake = FakeSender::new(vec![Err("operation timed out".to_string())]);
        let exporter = TelemetryExporter::new(metrics.clone())
            .with_clock(pinned_clock())
            .with_sender(fake.to_sender());

        exporter
            .apply_config(&telemetry_config("http://collector.internal/v1/traces"))
            .await;
        assert!(exporter.try_enqueue(spans(2)));
        wait_until(|| fake.calls.load(Ordering::SeqCst) >= 1).await;
        exporter.shutdown_flush().await;

        // The retry chain was interrupted after its first attempt —
        // not 4 — and the spans count dropped exactly once.
        assert_eq!(fake.calls.load(Ordering::SeqCst), 1);
        assert_eq!(metrics.exported(), 0);
        assert_eq!(metrics.dropped(), 2);
        assert_eq!(metrics.failures(), 0);
        assert!(!exporter.inner.task_alive.load(Ordering::Relaxed));
    }

    #[tokio::test(start_paused = true)]
    async fn aborted_delivery_counts_spans_dropped_once() {
        let metrics = Arc::new(TelemetryMetrics::new());
        let fake = FakeSender::new(vec![Err("operation timed out".to_string())]);
        let exporter = TelemetryExporter::new(metrics.clone())
            .with_clock(pinned_clock())
            .with_sender(fake.to_sender());

        exporter
            .apply_config(&telemetry_config("http://collector.internal/v1/traces"))
            .await;
        assert!(exporter.try_enqueue(spans(4)));
        wait_until(|| fake.calls.load(Ordering::SeqCst) >= 1).await;
        exporter.shutdown_flush().await;

        // Exactly one dropped count per span despite the abort passing
        // through both the retry loop and the flush caller.
        assert_eq!(metrics.dropped(), 4);
        assert_eq!(metrics.exported(), 0);
        assert_eq!(metrics.exported() + metrics.dropped(), 4);
    }

    #[tokio::test]
    async fn disabled_rejections_are_counted_exactly_once() {
        let metrics = Arc::new(TelemetryMetrics::new());
        let fake = FakeSender::new(Vec::new());
        let exporter = TelemetryExporter::new(metrics.clone())
            .with_sleeper(instant_sleeper())
            .with_clock(pinned_clock())
            .with_sender(fake.to_sender());

        // Dormant rejection: counted once, never again.
        assert!(!exporter.try_enqueue(spans(2)));
        assert_eq!(metrics.dropped(), 2);
        // Explicitly disabled rejection: counted once more.
        exporter.apply_config(&TelemetryConfig::default()).await;
        assert!(!exporter.try_enqueue(spans(3)));
        assert_eq!(metrics.dropped(), 5);
        assert_eq!(fake.calls.load(Ordering::SeqCst), 0);
    }

    #[test]
    fn transport_classification_is_allowlisted_and_leak_free() {
        // Raw reqwest strings can embed URLs, queries and userinfo —
        // the classified label must never carry any of it.
        let sneaky = "error sending request for url (https://key:sekret@collector.internal/v1/traces?x=1) connection failed";
        assert_eq!(classify_transport(sneaky), ExportErrorClass::Connect);
        assert!(!classify_transport(sneaky).label().contains("sekret"));
        assert!(!classify_transport(sneaky).label().contains("collector.internal"));

        assert_eq!(
            classify_transport("dns resolution error"),
            ExportErrorClass::Dns
        );
        assert_eq!(
            classify_transport("request body decode failure"),
            ExportErrorClass::TransportBody
        );
        let other = classify_transport("something entirely novel");
        assert_eq!(other, ExportErrorClass::TransportOther);
        assert_eq!(other.label(), "transport_other");

        let leaked = classify_transport("GET http://a:b@c.example/x went wrong");
        assert_eq!(leaked, ExportErrorClass::TransportOther);
        assert!(!leaked.label().contains("b@c.example"));
    }

    #[test]
    fn client_build_failure_maps_to_dedicated_class() {
        // The default sender's no-fallback failure marker classifies
        // into its own stable class (F8) — label matches the admin-UI
        // vocabulary.
        assert_eq!(
            classify_transport(CLIENT_BUILD_FAILED),
            ExportErrorClass::ClientBuildFailed
        );
        assert_eq!(
            ExportErrorClass::ClientBuildFailed.label(),
            "client_build_failed"
        );
    }

    #[tokio::test(start_paused = true)]
    async fn supervisor_recovers_after_task_panic() {
        let metrics = Arc::new(TelemetryMetrics::new());
        // First sender call panics inside the delivery task; every later
        // call succeeds.
        let calls = Arc::new(AtomicUsize::new(0));
        let panic_calls = calls.clone();
        let sender: HttpSender = Arc::new(move |_request: HttpRequest| {
            let index = panic_calls.fetch_add(1, Ordering::SeqCst);
            Box::pin(async move {
                if index == 0 {
                    panic!("collector exploded");
                }
                Ok(200)
            })
        });
        let exporter = TelemetryExporter::new(metrics.clone())
            .with_sleeper(instant_sleeper())
            .with_clock(pinned_clock())
            .with_sender(sender);

        exporter
            .apply_config(&telemetry_config("http://collector.internal/v1/traces"))
            .await;
        assert!(exporter.try_enqueue(spans(2)));

        // The panicking task dies (Error Scenario 5): enabled but not
        // healthy, and the crash is visible once repair notices.
        loop {
            if !exporter.status().await.healthy {
                break;
            }
            tokio::time::sleep(Duration::from_millis(10)).await;
        }
        // Repair triggers on the next enqueue (closed channel): the
        // first post-crash batch is dropped, and the task re-spawns on
        // a fresh channel. The crash itself was counted ONCE by the
        // supervisor; the channel rebuild must not count again (F3).
        let _ = exporter.try_enqueue(spans(2));
        wait_until(|| metrics.failures() >= 1).await;
        let mut accepted = false;
        for _ in 0..1000 {
            if exporter.try_enqueue(spans(2)) {
                accepted = true;
                break;
            }
            tokio::time::sleep(Duration::from_millis(10)).await;
        }
        assert!(accepted, "re-spawned task never accepted a batch");
        wait_until(|| metrics.exported() == 2).await;
        assert_eq!(metrics.failures(), 1);
        let status = exporter.status().await;
        assert_eq!(
            status.last_error_class.as_deref(),
            Some("task_crashed")
        );
        assert!(status.healthy);
    }

    #[tokio::test]
    async fn request_shape_carries_headers_and_gzip_body() {
        let mut config = telemetry_config("http://collector.internal/v1/traces");
        config.headers.insert(
            "Authorization".to_string(),
            "Bearer test-token".to_string(),
        );
        let target = ExportTarget::from_config(&config).expect("valid target");
        let request = target.build_request(spans(2));
        assert_eq!(request.url, "http://collector.internal/v1/traces");
        assert!(request.headers.iter().any(|(name, value)| {
            name == "Authorization" && value == "Bearer test-token"
        }));
        // gzip magic bytes 0x1f 0x8b (encode_batch always gzips).
        assert_eq!(&request.body[..2], &[0x1f, 0x8b]);
        assert_eq!(request.host(), "collector.internal");
    }
}
