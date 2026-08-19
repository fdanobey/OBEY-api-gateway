#![allow(dead_code)]
use chrono::{DateTime, Utc};
use rusqlite::{params, Connection};
use serde::{Deserialize, Serialize};
use std::path::Path;
use std::sync::mpsc::{self, Receiver, SyncSender, TryRecvError, TrySendError};
use std::sync::{LazyLock, Mutex};
use thiserror::Error;

use crate::{
    compression::{
        stats::{sanitize_operational_metadata, CompressionStats, MAX_ENGINE_LABEL_LEN},
        CompressionLevel,
    },
    config::LoggingConfig,
};

#[derive(Debug, Error)]
pub enum LoggerError {
    #[error("Database error: {0}")]
    Database(#[from] rusqlite::Error),

    #[error("Serialization error: {0}")]
    Serialization(#[from] serde_json::Error),

    #[error("IO error: {0}")]
    Io(#[from] std::io::Error),

    #[error("Failed to enqueue log command: {0}")]
    Enqueue(String),
}

pub type Result<T> = std::result::Result<T, LoggerError>;

const MAX_COMPRESSION_ENGINES: usize = 32;

/// Content-free compression metadata persisted with a request log.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct CompressionLogMetadata {
    #[serde(alias = "level")]
    pub compression_level: String,
    pub original_tokens: u32,
    pub compressed_tokens: u32,
    pub savings_percent: f64,
    pub engines_applied: Vec<String>,
    pub duration_ms: u64,
    pub auto_triggered: bool,
    #[serde(default)]
    pub cache_downgrade_applied: bool,
    #[serde(default)]
    pub tool_definitions_tokens_saved: u32,
    #[serde(default)]
    pub caveman_applied: bool,
    #[serde(default)]
    pub timed_out: bool,
    #[serde(default)]
    pub error: bool,
}

impl From<&CompressionStats> for CompressionLogMetadata {
    fn from(stats: &CompressionStats) -> Self {
        Self {
            compression_level: compression_level_label(stats.level).to_owned(),
            original_tokens: stats.original_tokens,
            compressed_tokens: stats.compressed_tokens,
            savings_percent: if stats.savings_percent.is_finite() {
                stats.savings_percent.clamp(0.0, 100.0)
            } else {
                0.0
            },
            engines_applied: stats
                .engines_applied
                .iter()
                .take(MAX_COMPRESSION_ENGINES)
                .map(|engine| sanitize_operational_metadata(engine, MAX_ENGINE_LABEL_LEN))
                .collect(),
            duration_ms: stats.compression_time_ms,
            auto_triggered: stats.auto_triggered,
            cache_downgrade_applied: stats.cache_downgrade_applied,
            tool_definitions_tokens_saved: stats.tool_definitions_tokens_saved,
            caveman_applied: stats.caveman_applied,
            timed_out: stats.timed_out,
            error: stats.error,
        }
    }
}

impl CompressionLogMetadata {
    fn sanitized(&self) -> Self {
        let savings_percent = if self.original_tokens == 0 {
            0.0
        } else {
            f64::from(self.original_tokens.saturating_sub(self.compressed_tokens)) * 100.0
                / f64::from(self.original_tokens)
        };

        Self {
            compression_level: sanitize_compression_level(&self.compression_level),
            original_tokens: self.original_tokens,
            compressed_tokens: self.compressed_tokens,
            savings_percent,
            engines_applied: self
                .engines_applied
                .iter()
                .take(MAX_COMPRESSION_ENGINES)
                .map(|engine| sanitize_operational_metadata(engine, MAX_ENGINE_LABEL_LEN))
                .collect(),
            duration_ms: self.duration_ms,
            auto_triggered: self.auto_triggered,
            cache_downgrade_applied: self.cache_downgrade_applied,
            tool_definitions_tokens_saved: self.tool_definitions_tokens_saved,
            caveman_applied: self.caveman_applied,
            timed_out: self.timed_out,
            error: self.error,
        }
    }
}

fn compression_level_label(level: CompressionLevel) -> &'static str {
    match level {
        CompressionLevel::None => "none",
        CompressionLevel::Lite => "lite",
        CompressionLevel::Standard => "standard",
        CompressionLevel::Aggressive => "aggressive",
        CompressionLevel::Ultra => "ultra",
        CompressionLevel::Rtk => "rtk",
        CompressionLevel::Stacked => "stacked",
    }
}

fn sanitize_compression_level(level: &str) -> String {
    let sanitized = sanitize_operational_metadata(level, 32).to_ascii_lowercase();
    if sanitized
        .chars()
        .all(|character| character.is_ascii_alphanumeric() || matches!(character, '_' | '-'))
    {
        sanitized
    } else {
        String::new()
    }
}

fn compression_log_metadata_from_json(json: &str) -> Option<CompressionLogMetadata> {
    match serde_json::from_str::<CompressionLogMetadata>(json) {
        Ok(metadata) => Some(metadata.sanitized()),
        Err(error) => {
            tracing::warn!(%error, "Ignoring malformed compression log metadata");
            None
        }
    }
}

/// Log entry for a single request
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct LogEntry {
    pub trace_id: String,
    pub timestamp: DateTime<Utc>,
    pub method: String,
    pub path: String,
    pub model: String,
    pub provider: String,
    pub status_code: u16,
    pub duration_ms: u64,
    pub cost: f64,
    pub request_body: Option<String>,
    pub response_body: Option<String>,
    /// The model version originally requested by the client
    pub requested_model: Option<String>,
    /// The model version that actually responded (may differ if version fallback occurred)
    pub responded_model: Option<String>,
    /// Content-free metadata describing request compression, when applied or attempted.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub compression: Option<CompressionLogMetadata>,
    #[serde(default)]
    pub memories_injected: u32,
    #[serde(default)]
    pub memories_stored: u32,
    #[serde(default)]
    pub injection_tokens: u32,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub detected_project: Option<String>,
}

/// Filter for querying log entries
#[derive(Debug, Clone, Default)]
pub struct LogFilter {
    pub trace_id: Option<String>,
    pub start_time: Option<DateTime<Utc>>,
    pub end_time: Option<DateTime<Utc>>,
    pub model: Option<String>,
    pub provider: Option<String>,
    pub status_code: Option<u16>,
    pub compression_level: Option<String>,
    pub limit: Option<usize>,
}

/// Depth of the bounded queue feeding the writer thread. Request handlers
/// never block on log writes: when the queue is full because the disk
/// cannot keep up, commands are dropped with a warning instead of
/// stalling the hot path.
const LOG_QUEUE_CAPACITY: usize = 4096;

/// Maximum number of entries committed per transaction on the writer
/// thread. Batched transactions amortize fsync cost across concurrent
/// requests instead of paying one autocommit fsync per entry.
const MAX_WRITE_BATCH: usize = 64;

/// Command processed by the dedicated writer thread. The command channel
/// is FIFO, so a `Query` always observes every `Write` enqueued before it.
enum LoggerCommand {
    Write(LogEntry),
    Query {
        filter: Box<LogFilter>,
        responder: mpsc::Sender<Result<Vec<LogEntry>>>,
    },
    Cleanup {
        responder: mpsc::Sender<Result<usize>>,
    },
    Checkpoint {
        responder: mpsc::Sender<()>,
    },
}

/// Request logger with a SQLite backend driven by a dedicated writer
/// thread.
///
/// `log` enqueues onto a bounded channel and returns immediately, so
/// request handlers never block on disk I/O and never contend on a
/// connection mutex. The writer thread owns the only `Connection`, batches
/// consecutive writes into a single transaction, and services queries and
/// maintenance commands in FIFO order.
pub struct RequestLogger {
    sender: Option<SyncSender<LoggerCommand>>,
    /// `JoinHandle` is `Send` but not `Sync`; the `Mutex` keeps the logger
    /// shareable via `Arc<AppState>` across axum handlers.
    writer: Mutex<Option<std::thread::JoinHandle<()>>>,
}

impl RequestLogger {
    /// Create a new RequestLogger with the given configuration
    pub fn new(config: LoggingConfig) -> Result<Self> {
        let db_path = Path::new(&config.database_path);

        // Create parent directory if it doesn't exist
        if let Some(parent) = db_path.parent() {
            std::fs::create_dir_all(parent)?;
        }

        let conn = Connection::open(db_path)?;

        // Create schema
        Self::create_schema(&conn)?;

        let (sender, receiver) = mpsc::sync_channel(LOG_QUEUE_CAPACITY);
        let writer = std::thread::Builder::new()
            .name("request-log-writer".to_owned())
            .spawn(move || writer_loop(conn, config, receiver))
            .map_err(LoggerError::from)?;

        Ok(Self {
            sender: Some(sender),
            writer: Mutex::new(Some(writer)),
        })
    }

    /// Enqueue a command without ever blocking the caller.
    fn send_command(&self, command: LoggerCommand) -> Result<()> {
        let Some(sender) = self.sender.as_ref() else {
            return Err(LoggerError::Enqueue(
                "request log writer is shut down".to_owned(),
            ));
        };
        match sender.try_send(command) {
            Ok(()) => Ok(()),
            Err(TrySendError::Full(_)) => {
                tracing::warn!("Request log queue full; dropping command");
                Err(LoggerError::Enqueue("request log queue full".to_owned()))
            }
            Err(TrySendError::Disconnected(_)) => Err(LoggerError::Enqueue(
                "request log writer stopped".to_owned(),
            )),
        }
    }

    /// Log a request entry.
    ///
    /// The entry is persisted asynchronously by the writer thread; this
    /// call only fails when the bounded queue is full or the writer is
    /// gone, so callers can warn without stalling the response path.
    pub fn log(&self, entry: LogEntry) -> Result<()> {
        self.send_command(LoggerCommand::Write(entry))
    }

    /// Create database schema with indexes
    fn create_schema(conn: &Connection) -> Result<()> {
        conn.execute(
            "CREATE TABLE IF NOT EXISTS requests (
                id INTEGER PRIMARY KEY,
                trace_id TEXT NOT NULL,
                timestamp INTEGER NOT NULL,
                method TEXT NOT NULL,
                path TEXT NOT NULL,
                model TEXT NOT NULL,
                provider TEXT NOT NULL,
                status_code INTEGER NOT NULL,
                duration_ms INTEGER NOT NULL,
                cost REAL NOT NULL,
                request_body TEXT,
                response_body TEXT,
                requested_model TEXT,
                responded_model TEXT,
                compression_metadata TEXT,
                compression_level TEXT,
                memories_injected INTEGER NOT NULL DEFAULT 0,
                memories_stored INTEGER NOT NULL DEFAULT 0,
                injection_tokens INTEGER NOT NULL DEFAULT 0,
                detected_project TEXT
            )",
            [],
        )?;

        Self::ensure_column(conn, "compression_metadata", "TEXT")?;
        Self::ensure_column(conn, "compression_level", "TEXT")?;
        Self::ensure_column(conn, "memories_injected", "INTEGER NOT NULL DEFAULT 0")?;
        Self::ensure_column(conn, "memories_stored", "INTEGER NOT NULL DEFAULT 0")?;
        Self::ensure_column(conn, "injection_tokens", "INTEGER NOT NULL DEFAULT 0")?;
        Self::ensure_column(conn, "detected_project", "TEXT")?;

        // Create indexes for common query patterns
        conn.execute(
            "CREATE INDEX IF NOT EXISTS idx_requests_timestamp ON requests(timestamp)",
            [],
        )?;

        conn.execute(
            "CREATE INDEX IF NOT EXISTS idx_requests_model ON requests(model)",
            [],
        )?;

        conn.execute(
            "CREATE INDEX IF NOT EXISTS idx_requests_provider ON requests(provider)",
            [],
        )?;

        conn.execute(
            "CREATE INDEX IF NOT EXISTS idx_requests_status_code ON requests(status_code)",
            [],
        )?;

        conn.execute(
            "CREATE INDEX IF NOT EXISTS idx_requests_trace_id ON requests(trace_id)",
            [],
        )?;

        conn.execute(
            "CREATE INDEX IF NOT EXISTS idx_requests_compression_level ON requests(compression_level)",
            [],
        )?;

        Ok(())
    }

    fn ensure_column(conn: &Connection, column_name: &str, column_type: &str) -> Result<()> {
        let mut stmt = conn.prepare("PRAGMA table_info(requests)")?;
        let columns = stmt
            .query_map([], |row| row.get::<_, String>(1))?
            .collect::<std::result::Result<Vec<_>, _>>()?;

        if !columns.iter().any(|column| column == column_name) {
            conn.execute(
                &format!("ALTER TABLE requests ADD COLUMN {column_name} {column_type}"),
                [],
            )?;
        }

        Ok(())
    }

    /// Query log entries with optional filtering.
    ///
    /// Blocks on a round-trip to the writer thread; the FIFO command
    /// channel guarantees every entry logged before this call is visible
    /// to the query.
    pub fn query(&self, filter: LogFilter) -> Result<Vec<LogEntry>> {
        let (responder, response) = mpsc::channel();
        self.send_command(LoggerCommand::Query {
            filter: Box::new(filter),
            responder,
        })?;
        response.recv().map_err(|_| {
            LoggerError::Enqueue("request log writer stopped during query".to_owned())
        })?
    }

    /// Clean up old log entries based on retention policy
    pub fn cleanup_old_logs(&self) -> Result<usize> {
        let (responder, response) = mpsc::channel();
        self.send_command(LoggerCommand::Cleanup { responder })?;
        response.recv().map_err(|_| {
            LoggerError::Enqueue("request log writer stopped during cleanup".to_owned())
        })?
    }

    /// Flush pending writes and checkpoint the WAL (Req 18.3).
    /// Called during graceful shutdown to ensure all data is persisted.
    pub fn flush(&self) {
        let (responder, response) = mpsc::channel();
        if self
            .send_command(LoggerCommand::Checkpoint { responder })
            .is_ok()
        {
            let _ = response.recv();
            tracing::info!("RequestLogger flushed and WAL checkpointed");
        }
    }
}

impl Drop for RequestLogger {
    fn drop(&mut self) {
        // Close the command channel first so the writer drains every
        // queued entry, then join it so pending writes are durable before
        // the logger goes away (callers may re-open the same database).
        self.sender = None;
        let writer = self
            .writer
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner)
            .take();
        if let Some(writer) = writer {
            let _ = writer.join();
        }
    }
}

/// Applies the logging config's redaction, exclusion, and size-limit rules
/// to request/response bodies. Owned by the writer thread so regex and
/// JSON processing never runs on request handlers.
struct BodySanitizer {
    config: LoggingConfig,
}

impl BodySanitizer {
    fn new(config: LoggingConfig) -> Self {
        Self { config }
    }

    /// Process body for logging: apply size limits, field exclusion, and API key redaction
    fn process_body(&self, body: &str) -> String {
        // First, redact API keys
        let redacted = self.redact_api_keys(body);

        // Then, exclude fields if configured
        let excluded = self.exclude_fields(&redacted);

        // Finally, apply size limit
        self.apply_size_limit(&excluded)
    }

    /// Redact API keys and authorization tokens from text
    fn redact_api_keys(&self, text: &str) -> String {
        let mut result = text.to_string();
        for (pattern, replacement) in REDACTION_RULES.iter() {
            result = pattern.replace_all(&result, *replacement).to_string();
        }
        result
    }

    /// Exclude configured fields from JSON body
    fn exclude_fields(&self, body: &str) -> String {
        if self.config.excluded_fields.is_empty() {
            return body.to_string();
        }

        // Try to parse as JSON
        if let Ok(mut json) = serde_json::from_str::<serde_json::Value>(body) {
            // Recursively redact excluded fields
            self.redact_json_fields(&mut json);
            serde_json::to_string(&json).unwrap_or_else(|_| body.to_string())
        } else {
            body.to_string()
        }
    }

    /// Recursively redact excluded fields in JSON
    fn redact_json_fields(&self, value: &mut serde_json::Value) {
        match value {
            serde_json::Value::Object(map) => {
                for (key, val) in map.iter_mut() {
                    // Check if this field should be excluded
                    if self
                        .config
                        .excluded_fields
                        .iter()
                        .any(|f| f.eq_ignore_ascii_case(key))
                    {
                        *val = serde_json::Value::String("[REDACTED]".to_string());
                    } else {
                        // Recursively process nested objects/arrays
                        self.redact_json_fields(val);
                    }
                }
            }
            serde_json::Value::Array(arr) => {
                for item in arr.iter_mut() {
                    self.redact_json_fields(item);
                }
            }
            _ => {}
        }
    }

    /// Apply size limit to body, truncating if necessary
    fn apply_size_limit(&self, body: &str) -> String {
        if body.len() <= self.config.max_body_size_bytes {
            body.to_string()
        } else {
            let truncated = &body[..self.config.max_body_size_bytes];
            format!(
                "{}... [TRUNCATED: body exceeds {} bytes]",
                truncated, self.config.max_body_size_bytes
            )
        }
    }

}

/// Precompiled redaction rules; compiling these per call dominated the old
/// synchronous write path.
static REDACTION_RULES: LazyLock<Vec<(regex::Regex, &'static str)>> = LazyLock::new(|| {
    use regex::Regex;
    vec![
        // OpenAI keys: sk-... with 20+ chars (covers sk-proj-..., sk-svcacct-..., etc.)
        (
            Regex::new(r"sk-[a-zA-Z0-9_\-]{20,}").unwrap(),
            "[REDACTED]",
        ),
        (
            Regex::new(r"Bearer\s+[a-zA-Z0-9\-_.]+").unwrap(),
            "Bearer [REDACTED]",
        ),
        (
            Regex::new(
                r#"(?i)(api[_-]?key|authorization)["']?\s*[:=]\s*["']?[a-zA-Z0-9\-_.]+"#,
            )
            .unwrap(),
            "$1: [REDACTED]",
        ),
        // AWS access keys: AKIA...
        (
            Regex::new(r"AKIA[A-Z0-9]{16}").unwrap(),
            "[REDACTED]",
        ),
    ]
});

/// Writer thread body: owns the only SQLite connection and processes
/// commands in FIFO order, batching consecutive writes into a single
/// transaction.
fn writer_loop(mut conn: Connection, config: LoggingConfig, receiver: Receiver<LoggerCommand>) {
    let sanitizer = BodySanitizer::new(config.clone());
    // WAL + NORMAL synchronous keeps batched commits cheap; flush()
    // checkpoints the WAL for durability on graceful shutdown.
    let _ = conn.execute_batch(
        "PRAGMA journal_mode=WAL; PRAGMA synchronous=NORMAL; PRAGMA busy_timeout=5000;",
    );

    while let Ok(command) = receiver.recv() {
        match command {
            LoggerCommand::Write(entry) => {
                let mut batch = Vec::with_capacity(MAX_WRITE_BATCH);
                batch.push(entry);
                loop {
                    match receiver.try_recv() {
                        Ok(LoggerCommand::Write(entry)) => {
                            batch.push(entry);
                            if batch.len() >= MAX_WRITE_BATCH {
                                break;
                            }
                        }
                        Ok(other) => {
                            write_batch(&mut conn, &sanitizer, &config, &batch);
                            handle_command(&mut conn, &sanitizer, &config, other);
                            batch.clear();
                            break;
                        }
                        Err(TryRecvError::Empty) => break,
                        Err(TryRecvError::Disconnected) => {
                            write_batch(&mut conn, &sanitizer, &config, &batch);
                            return;
                        }
                    }
                }
                write_batch(&mut conn, &sanitizer, &config, &batch);
            }
            other => handle_command(&mut conn, &sanitizer, &config, other),
        }
    }
}

fn handle_command(
    conn: &mut Connection,
    sanitizer: &BodySanitizer,
    config: &LoggingConfig,
    command: LoggerCommand,
) {
    match command {
        LoggerCommand::Write(entry) => {
            write_batch(conn, sanitizer, config, std::slice::from_ref(&entry))
        }
        LoggerCommand::Query { filter, responder } => {
            let _ = responder.send(run_query(conn, &filter));
        }
        LoggerCommand::Cleanup { responder } => {
            let _ = responder.send(run_cleanup(conn, config));
        }
        LoggerCommand::Checkpoint { responder } => {
            // Checkpoint WAL to ensure all data is written to the main database file
            let _ = conn.execute_batch("PRAGMA wal_checkpoint(TRUNCATE);");
            let _ = responder.send(());
        }
    }
}

/// Commit a batch of entries in a single transaction. On failure the
/// entries are retried individually so one malformed entry does not
/// discard the rest of the batch.
fn write_batch(
    conn: &mut Connection,
    sanitizer: &BodySanitizer,
    config: &LoggingConfig,
    entries: &[LogEntry],
) {
    if entries.is_empty() {
        return;
    }

    let batch = (|| -> Result<()> {
        let tx = conn.transaction()?;
        for entry in entries {
            insert_entry(&tx, sanitizer, config, entry)?;
        }
        tx.commit()?;
        Ok(())
    })();

    if let Err(error) = batch {
        tracing::error!(%error, count = entries.len(), "Failed to commit request log batch");
        for entry in entries {
            if let Err(error) = insert_entry(conn, sanitizer, config, entry) {
                tracing::error!(%error, trace_id = %entry.trace_id, "Failed to write request log entry");
            }
        }
    }
}

/// Persist a single entry. Runs on the writer thread only.
fn insert_entry(
    conn: &Connection,
    sanitizer: &BodySanitizer,
    config: &LoggingConfig,
    entry: &LogEntry,
) -> Result<()> {
    // Process request body if logging is enabled
    let request_body = if config.request_body_logging {
        entry
            .request_body
            .as_deref()
            .map(|body| sanitizer.process_body(body))
    } else {
        None
    };

    // Process response body if logging is enabled
    let response_body = if config.response_body_logging {
        entry
            .response_body
            .as_deref()
            .map(|body| sanitizer.process_body(body))
    } else {
        None
    };

    let compression = entry
        .compression
        .as_ref()
        .map(CompressionLogMetadata::sanitized);
    let compression_level = compression
        .as_ref()
        .map(|metadata| metadata.compression_level.clone());
    let compression_metadata = compression
        .as_ref()
        .map(serde_json::to_string)
        .transpose()?;

    conn.execute(
        "INSERT INTO requests (
            trace_id, timestamp, method, path, model, provider,
            status_code, duration_ms, cost, request_body, response_body,
            requested_model, responded_model, compression_metadata, compression_level,
            memories_injected, memories_stored, injection_tokens, detected_project
        ) VALUES (
            ?1, ?2, ?3, ?4, ?5, ?6, ?7, ?8, ?9, ?10, ?11, ?12, ?13, ?14, ?15,
            ?16, ?17, ?18, ?19
        )",
        params![
            entry.trace_id,
            entry.timestamp.timestamp(),
            entry.method,
            entry.path,
            entry.model,
            entry.provider,
            entry.status_code,
            entry.duration_ms,
            entry.cost,
            request_body,
            response_body,
            entry.requested_model,
            entry.responded_model,
            compression_metadata,
            compression_level,
            entry.memories_injected,
            entry.memories_stored,
            entry.injection_tokens,
            entry.detected_project,
        ],
    )?;

    Ok(())
}

/// Query log entries with optional filtering. Runs on the writer thread.
fn run_query(conn: &Connection, filter: &LogFilter) -> Result<Vec<LogEntry>> {
    let mut query = String::from("SELECT trace_id, timestamp, method, path, model, provider, status_code, duration_ms, cost, request_body, response_body, requested_model, responded_model, compression_metadata, memories_injected, memories_stored, injection_tokens, detected_project FROM requests WHERE 1=1");
    let mut params: Vec<Box<dyn rusqlite::ToSql>> = Vec::new();

    if let Some(ref trace_id) = filter.trace_id {
        query.push_str(" AND trace_id = ?");
        params.push(Box::new(trace_id.clone()));
    }

    if let Some(start_time) = filter.start_time {
        query.push_str(" AND timestamp >= ?");
        params.push(Box::new(start_time.timestamp()));
    }

    if let Some(end_time) = filter.end_time {
        query.push_str(" AND timestamp <= ?");
        params.push(Box::new(end_time.timestamp()));
    }

    if let Some(ref model) = filter.model {
        query.push_str(" AND model = ?");
        params.push(Box::new(model.clone()));
    }

    if let Some(ref provider) = filter.provider {
        query.push_str(" AND provider = ?");
        params.push(Box::new(provider.clone()));
    }

    if let Some(status_code) = filter.status_code {
        query.push_str(" AND status_code = ?");
        params.push(Box::new(status_code));
    }

    if let Some(ref compression_level) = filter.compression_level {
        query.push_str(" AND compression_level = ? COLLATE NOCASE");
        params.push(Box::new(compression_level.clone()));
    }

    query.push_str(" ORDER BY timestamp DESC");

    if let Some(limit) = filter.limit {
        query.push_str(" LIMIT ?");
        params.push(Box::new(limit as i64));
    }

    let param_refs: Vec<&dyn rusqlite::ToSql> = params.iter().map(|p| p.as_ref()).collect();

    let mut stmt = conn.prepare(&query)?;
    let entries = stmt
        .query_map(param_refs.as_slice(), |row| {
            let compression_json: Option<String> = row.get(13)?;
            let compression = compression_json
                .as_deref()
                .and_then(compression_log_metadata_from_json);

            Ok(LogEntry {
                trace_id: row.get(0)?,
                timestamp: DateTime::from_timestamp(row.get(1)?, 0).unwrap(),
                method: row.get(2)?,
                path: row.get(3)?,
                model: row.get(4)?,
                provider: row.get(5)?,
                status_code: row.get(6)?,
                duration_ms: row.get(7)?,
                cost: row.get(8)?,
                request_body: row.get(9)?,
                response_body: row.get(10)?,
                requested_model: row.get(11)?,
                responded_model: row.get(12)?,
                compression,
                memories_injected: row.get(14)?,
                memories_stored: row.get(15)?,
                injection_tokens: row.get(16)?,
                detected_project: row.get(17)?,
            })
        })?
        .collect::<std::result::Result<Vec<_>, _>>()?;

    Ok(entries)
}

/// Clean up old log entries based on retention policy. Runs on the writer
/// thread.
fn run_cleanup(conn: &Connection, config: &LoggingConfig) -> Result<usize> {
    if config.retention_days == 0 {
        // Retention disabled
        return Ok(0);
    }

    let cutoff_timestamp =
        Utc::now().timestamp() - (config.retention_days as i64 * 24 * 60 * 60);

    let deleted = conn.execute(
        "DELETE FROM requests WHERE timestamp < ?1",
        params![cutoff_timestamp],
    )?;

    Ok(deleted)
}

#[cfg(test)]
mod tests {
    use super::*;
    use tempfile::NamedTempFile;

    fn create_test_logger() -> (RequestLogger, NamedTempFile) {
        let temp_file = NamedTempFile::new().unwrap();
        let config = LoggingConfig {
            level: "info".to_string(),
            database_path: temp_file.path().to_str().unwrap().to_string(),
            request_body_logging: true,
            response_body_logging: true,
            max_body_size_bytes: 1000,
            excluded_fields: vec!["api_key".to_string(), "password".to_string()],
            retention_days: 30,
            cleanup_schedule_hours: 24,
        };

    let logger = RequestLogger::new(config).unwrap();
    (logger, temp_file)
}

fn test_sanitizer() -> BodySanitizer {
    BodySanitizer::new(LoggingConfig {
        level: "info".to_string(),
        database_path: String::new(),
        request_body_logging: true,
        response_body_logging: true,
        max_body_size_bytes: 1000,
        excluded_fields: vec!["api_key".to_string(), "password".to_string()],
        retention_days: 30,
        cleanup_schedule_hours: 24,
    })
}

fn sample_entry(trace_id: &str, compression: Option<CompressionLogMetadata>) -> LogEntry {
        LogEntry {
            trace_id: trace_id.to_owned(),
            timestamp: Utc::now(),
            method: "POST".to_owned(),
            path: "/v1/chat/completions".to_owned(),
            model: "gpt-4".to_owned(),
            provider: "openai".to_owned(),
            status_code: 200,
            duration_ms: 1500,
            cost: 0.05,
            request_body: None,
            response_body: None,
            requested_model: None,
            responded_model: None,
            compression,
            memories_injected: 0,
            memories_stored: 0,
            injection_tokens: 0,
            detected_project: None,
        }
    }

    fn sample_compression(level: CompressionLevel) -> CompressionLogMetadata {
        CompressionLogMetadata {
            compression_level: compression_level_label(level).to_owned(),
            original_tokens: 1_000,
            compressed_tokens: 600,
            savings_percent: 40.0,
            engines_applied: vec!["semantic_dedup".to_owned(), "tool_schema".to_owned()],
            duration_ms: 42,
            auto_triggered: true,
            cache_downgrade_applied: false,
            tool_definitions_tokens_saved: 120,
            caveman_applied: true,
            timed_out: false,
            error: false,
        }
    }

    #[test]
    fn migrates_old_schema_and_preserves_old_rows() {
        let temp_file = NamedTempFile::new().unwrap();
        let conn = Connection::open(temp_file.path()).unwrap();
        conn.execute_batch(
            "CREATE TABLE requests (
                id INTEGER PRIMARY KEY,
                trace_id TEXT NOT NULL,
                timestamp INTEGER NOT NULL,
                method TEXT NOT NULL,
                path TEXT NOT NULL,
                model TEXT NOT NULL,
                provider TEXT NOT NULL,
                status_code INTEGER NOT NULL,
                duration_ms INTEGER NOT NULL,
                cost REAL NOT NULL,
                request_body TEXT,
                response_body TEXT,
                requested_model TEXT,
                responded_model TEXT
            );
            INSERT INTO requests (
                trace_id, timestamp, method, path, model, provider, status_code,
                duration_ms, cost, request_body, response_body, requested_model, responded_model
            ) VALUES (
                'legacy', 1700000000, 'POST', '/v1/chat/completions', 'gpt-4',
                'openai', 200, 10, 0.0, NULL, NULL, NULL, NULL
            );",
        )
        .unwrap();
        drop(conn);

        let config = LoggingConfig {
            database_path: temp_file.path().to_string_lossy().into_owned(),
            ..Default::default()
        };
        let logger = RequestLogger::new(config.clone()).unwrap();
        let legacy = logger
            .query(LogFilter {
                trace_id: Some("legacy".to_owned()),
                ..Default::default()
            })
            .unwrap();
        assert_eq!(legacy.len(), 1);
        assert!(legacy[0].compression.is_none());
        drop(logger);

        RequestLogger::new(config).unwrap();
        let conn = Connection::open(temp_file.path()).unwrap();
        let columns = conn
            .prepare("PRAGMA table_info(requests)")
            .unwrap()
            .query_map([], |row| row.get::<_, String>(1))
            .unwrap()
            .collect::<std::result::Result<Vec<_>, _>>()
            .unwrap();
        assert!(columns.contains(&"compression_metadata".to_owned()));
        assert!(columns.contains(&"compression_level".to_owned()));
    }

    #[test]
    fn compression_metadata_round_trips() {
        let (logger, _temp) = create_test_logger();
        let metadata = sample_compression(CompressionLevel::Aggressive);
        logger
            .log(sample_entry("compressed", Some(metadata.clone())))
            .unwrap();

        let results = logger
            .query(LogFilter {
                trace_id: Some("compressed".to_owned()),
                ..Default::default()
            })
            .unwrap();
        assert_eq!(results[0].compression.as_ref(), Some(&metadata));
    }

    #[test]
    fn filters_compression_level_with_existing_filters() {
        let (logger, _temp) = create_test_logger();
        let mut matching = sample_entry(
            "matching",
            Some(sample_compression(CompressionLevel::Standard)),
        );
        matching.provider = "anthropic".to_owned();
        matching.status_code = 201;
        logger.log(matching).unwrap();

        let mut wrong_level = sample_entry(
            "wrong-level",
            Some(sample_compression(CompressionLevel::Lite)),
        );
        wrong_level.provider = "anthropic".to_owned();
        wrong_level.status_code = 201;
        logger.log(wrong_level).unwrap();
        logger
            .log(sample_entry(
                "wrong-provider",
                Some(sample_compression(CompressionLevel::Standard)),
            ))
            .unwrap();

        let results = logger
            .query(LogFilter {
                model: Some("gpt-4".to_owned()),
                provider: Some("anthropic".to_owned()),
                status_code: Some(201),
                compression_level: Some("standard".to_owned()),
                ..Default::default()
            })
            .unwrap();
        assert_eq!(results.len(), 1);
        assert_eq!(results[0].trace_id, "matching");
    }

    #[test]
    fn malformed_compression_metadata_is_ignored() {
    let (logger, temp) = create_test_logger();
    logger.log(sample_entry("malformed", None)).unwrap();
    // Round-trip the writer thread so the INSERT has committed before the
    // direct SQL corruption below executes.
    logger.flush();
    Connection::open(temp.path())
        .unwrap()
        .execute(
            "UPDATE requests SET compression_metadata = ?1 WHERE trace_id = ?2",
            params!["{not valid json", "malformed"],
        )
        .unwrap();

        let results = logger
            .query(LogFilter {
                trace_id: Some("malformed".to_owned()),
                ..Default::default()
            })
            .unwrap();
        assert_eq!(results.len(), 1);
        assert!(results[0].compression.is_none());
    }

    #[test]
    fn compression_serialization_is_bounded_and_secret_free() {
    let (logger, temp) = create_test_logger();
    let mut metadata = sample_compression(CompressionLevel::Ultra);
        metadata.engines_applied = vec![
            "semantic sk-super-secret-token-1234567890".repeat(10),
            "Bearer another-secret".to_owned(),
        ];
        metadata.savings_percent = f64::NAN;
    logger
        .log(sample_entry("safe-metadata", Some(metadata)))
        .unwrap();
    logger.flush();

    let conn = Connection::open(temp.path()).unwrap();
        let json: String = conn
            .query_row(
                "SELECT compression_metadata FROM requests WHERE trace_id = ?1",
                params!["safe-metadata"],
                |row| row.get(0),
            )
            .unwrap();
        assert!(!json.contains("super-secret-token"));
        assert!(!json.contains("another-secret"));
        let stored: CompressionLogMetadata = serde_json::from_str(&json).unwrap();
        assert!(stored
            .engines_applied
            .iter()
            .all(|engine| engine.len() <= MAX_ENGINE_LABEL_LEN));
        assert_eq!(stored.savings_percent, 40.0);
    }

    #[test]
    fn compression_metadata_from_stats_is_content_free() {
        let stats = CompressionStats {
            request_id: "request sk-request-secret".to_owned(),
            level: CompressionLevel::Standard,
            engines_applied: vec!["engine Bearer engine-secret".to_owned()],
            original_tokens: 100,
            compressed_tokens: 50,
            savings_percent: 50.0,
            compression_time_ms: 7,
            auto_triggered: true,
            cache_downgrade_applied: true,
            tool_definitions_tokens_saved: 5,
            caveman_applied: false,
            timed_out: false,
            error: false,
            provider: "provider sk-provider-secret".to_owned(),
            model: "model sk-model-secret".to_owned(),
            engine_results: Vec::new(),
        };

        let json = serde_json::to_string(&CompressionLogMetadata::from(&stats)).unwrap();
        assert_eq!(
            serde_json::from_str::<serde_json::Value>(&json).unwrap()["compression_level"],
            "standard"
        );
        assert!(!json.contains("request-secret"));
        assert!(!json.contains("provider-secret"));
        assert!(!json.contains("model-secret"));
        assert!(!json.contains("engine-secret"));
    }

    #[test]
    fn test_log_and_query() {
        let (logger, _temp) = create_test_logger();

        let entry = LogEntry {
            trace_id: "test-123".to_string(),
            timestamp: Utc::now(),
            method: "POST".to_string(),
            path: "/v1/chat/completions".to_string(),
            model: "gpt-4".to_string(),
            provider: "openai".to_string(),
            status_code: 200,
            duration_ms: 1500,
            cost: 0.05,
            request_body: Some(r#"{"model":"gpt-4"}"#.to_string()),
            response_body: Some(r#"{"choices":[]}"#.to_string()),
            requested_model: None,
            responded_model: None,
            compression: None,
            memories_injected: 0,
            memories_stored: 0,
            injection_tokens: 0,
            detected_project: None,
        };

        logger.log(entry.clone()).unwrap();

        let results = logger
            .query(LogFilter {
                trace_id: Some("test-123".to_string()),
                ..Default::default()
            })
            .unwrap();

        assert_eq!(results.len(), 1);
        assert_eq!(results[0].trace_id, "test-123");
        assert_eq!(results[0].model, "gpt-4");
    }

    #[test]
    fn memory_observability_fields_serialize_default_and_round_trip() {
        let (logger, _temp) = create_test_logger();
        let default_json = serde_json::to_value(sample_entry("defaults", None)).unwrap();
        assert_eq!(default_json["memories_injected"], 0);
        assert_eq!(default_json["memories_stored"], 0);
        assert_eq!(default_json["injection_tokens"], 0);
        assert!(default_json.get("detected_project").is_none());

        let mut entry = sample_entry("memory-counts", None);
        entry.memories_injected = 3;
        entry.memories_stored = 2;
        entry.injection_tokens = 240;
        entry.detected_project = Some("0123456789abcdef".to_owned());
        logger.log(entry).unwrap();

        let result = logger
            .query(LogFilter {
                trace_id: Some("memory-counts".to_owned()),
                ..Default::default()
            })
            .unwrap()
            .pop()
            .unwrap();
        assert_eq!(result.memories_injected, 3);
        assert_eq!(result.memories_stored, 2);
        assert_eq!(result.injection_tokens, 240);
        assert_eq!(result.detected_project.as_deref(), Some("0123456789abcdef"));
    }

    #[test]
fn test_api_key_redaction() {
    let sanitizer = test_sanitizer();

        // Standard 48-char key
        let body =
            r#"{"api_key":"sk-1234567890abcdefghijklmnopqrstuvwxyz1234567890ab","message":"test"}"#;
        let redacted = sanitizer.redact_api_keys(body);
        assert!(!redacted.contains("sk-1234567890"));
        assert!(redacted.contains("[REDACTED]"));

        // Longer project-scoped key (sk-proj-...)
        let body2 =
            r#"{"key":"sk-proj-abcdefghijklmnopqrstuvwxyz1234567890abcdefghijklmnopqrstuvwxyz"}"#;
        let redacted2 = sanitizer.redact_api_keys(body2);
        assert!(!redacted2.contains("sk-proj-"));
        assert!(redacted2.contains("[REDACTED]"));

        // AWS access key
        let body3 = r#"{"aws_key":"AKIAIOSFODNN7EXAMPLE"}"#;
        let redacted3 = sanitizer.redact_api_keys(body3);
        assert!(!redacted3.contains("AKIAIOSFODNN7EXAMPLE"));
        assert!(redacted3.contains("[REDACTED]"));
    }

    #[test]
fn test_field_exclusion() {
    let sanitizer = test_sanitizer();

        let body = r#"{"api_key":"secret","password":"pass123","message":"test"}"#;
        let excluded = sanitizer.exclude_fields(body);

        let json: serde_json::Value = serde_json::from_str(&excluded).unwrap();
        assert_eq!(json["api_key"], "[REDACTED]");
        assert_eq!(json["password"], "[REDACTED]");
        assert_eq!(json["message"], "test");
    }

    #[test]
fn test_size_limit() {
    let sanitizer = test_sanitizer();

        let large_body = "x".repeat(2000);
        let limited = sanitizer.apply_size_limit(&large_body);

        assert!(limited.len() < 2000);
        assert!(limited.contains("[TRUNCATED"));
    }

    #[test]
    fn test_cleanup_old_logs() {
        let (logger, _temp) = create_test_logger();

        // Insert an old entry
        let old_entry = LogEntry {
            trace_id: "old-123".to_string(),
            timestamp: Utc::now() - chrono::Duration::days(60),
            method: "POST".to_string(),
            path: "/v1/chat/completions".to_string(),
            model: "gpt-4".to_string(),
            provider: "openai".to_string(),
            status_code: 200,
            duration_ms: 1500,
            cost: 0.05,
            request_body: None,
            response_body: None,
            requested_model: None,
            responded_model: None,
            compression: None,
            memories_injected: 0,
            memories_stored: 0,
            injection_tokens: 0,
            detected_project: None,
        };

        logger.log(old_entry).unwrap();

        // Insert a recent entry
        let recent_entry = LogEntry {
            trace_id: "recent-123".to_string(),
            timestamp: Utc::now(),
            method: "POST".to_string(),
            path: "/v1/chat/completions".to_string(),
            model: "gpt-4".to_string(),
            provider: "openai".to_string(),
            status_code: 200,
            duration_ms: 1500,
            cost: 0.05,
            request_body: None,
            response_body: None,
            requested_model: None,
            responded_model: None,
            compression: None,
            memories_injected: 0,
            memories_stored: 0,
            injection_tokens: 0,
            detected_project: None,
        };

        logger.log(recent_entry).unwrap();

        // Cleanup should remove the old entry
        let deleted = logger.cleanup_old_logs().unwrap();
        assert_eq!(deleted, 1);

        // Verify only recent entry remains
        let results = logger.query(LogFilter::default()).unwrap();
        assert_eq!(results.len(), 1);
        assert_eq!(results[0].trace_id, "recent-123");
    }
}

#[cfg(test)]
mod property_tests {
    use super::*;
    use proptest::prelude::*;
    use tempfile::NamedTempFile;

    fn arb_log_entry() -> impl Strategy<Value = LogEntry> {
        (
            "[a-z0-9-]{8,36}",
            any::<i64>().prop_map(|ts| {
                DateTime::from_timestamp(ts.abs() % 2_000_000_000, 0).unwrap_or_else(|| Utc::now())
            }),
            prop::sample::select(vec!["GET", "POST", "PUT", "DELETE"]),
            prop::sample::select(vec![
                "/v1/chat/completions",
                "/v1/completions",
                "/v1/embeddings",
            ]),
            "[a-z0-9-]{3,20}",
            "[a-z0-9-]{3,20}",
            100u16..600u16,
            0u64..10000u64,
            0.0f64..10.0f64,
            prop::option::of(prop::string::string_regex("[a-zA-Z0-9 {}\":,]{0,500}").unwrap()),
            prop::option::of(prop::string::string_regex("[a-zA-Z0-9 {}\":,]{0,500}").unwrap()),
        )
            .prop_map(
                |(
                    trace_id,
                    timestamp,
                    method,
                    path,
                    model,
                    provider,
                    status_code,
                    duration_ms,
                    cost,
                    request_body,
                    response_body,
                )| {
                    LogEntry {
                        trace_id,
                        timestamp,
                        method: method.to_string(),
                        path: path.to_string(),
                        model,
                        provider,
                        status_code,
                        duration_ms,
                        cost,
                        request_body,
                        response_body,
                        requested_model: None,
                        responded_model: None,
                        compression: None,
                        memories_injected: 0,
                        memories_stored: 0,
                        injection_tokens: 0,
                        detected_project: None,
                    }
                },
            )
    }

    fn create_test_logger_with_config(config: LoggingConfig) -> (RequestLogger, NamedTempFile) {
        let temp_file = NamedTempFile::new().unwrap();
        let mut config = config;
        config.database_path = temp_file.path().to_str().unwrap().to_string();
        let logger = RequestLogger::new(config).unwrap();
        (logger, temp_file)
    }

    // Property 12: Request Logging Completeness
    // Validates: Requirements 14.1, 14.2, 33.2, 33.5
    proptest! {
        #[test]
        fn prop_request_logging_completeness(entry in arb_log_entry()) {
            let config = LoggingConfig::default();
            let (logger, _temp) = create_test_logger_with_config(config);

            // Log the entry
            logger.log(entry.clone()).unwrap();

            // Query by trace_id
            let results = logger.query(LogFilter {
                trace_id: Some(entry.trace_id.clone()),
                ..Default::default()
            }).unwrap();

            // Should find exactly one entry
            prop_assert_eq!(results.len(), 1);

            let logged = &results[0];

            // Verify all required fields are present
            prop_assert_eq!(&logged.trace_id, &entry.trace_id);
            prop_assert_eq!(&logged.method, &entry.method);
            prop_assert_eq!(&logged.path, &entry.path);
            prop_assert_eq!(&logged.model, &entry.model);
            prop_assert_eq!(&logged.provider, &entry.provider);
            prop_assert_eq!(logged.status_code, entry.status_code);
            prop_assert_eq!(logged.duration_ms, entry.duration_ms);
            prop_assert!((logged.cost - entry.cost).abs() < 0.001);
        }
    }

    // Property 14: Log Retention Cleanup
    // Validates: Requirements 14.6, 38.2
    proptest! {
        #[test]
        fn prop_log_retention_cleanup(
            retention_days in 1u32..90u32,
            days_old in 1i64..180i64,
        ) {
            // Skip the exact boundary (days_old == retention_days) because
            // sub-second clock drift between entry creation and cleanup makes
            // the outcome non-deterministic at that point.
            prop_assume!(days_old != retention_days as i64);

            let config = LoggingConfig {
                retention_days,
                ..Default::default()
            };
            let (logger, _temp) = create_test_logger_with_config(config);

            // Create entry with specific age
            let timestamp = Utc::now() - chrono::Duration::days(days_old);
            let entry = LogEntry {
                trace_id: format!("test-{}", days_old),
                timestamp,
                method: "POST".to_string(),
                path: "/v1/chat/completions".to_string(),
                model: "gpt-4".to_string(),
                provider: "openai".to_string(),
                status_code: 200,
                duration_ms: 1000,
                cost: 0.01,
                request_body: None,
                response_body: None,
                requested_model: None,
                responded_model: None,
                compression: None,
                memories_injected: 0,
                memories_stored: 0,
                injection_tokens: 0,
                detected_project: None,
            };

            logger.log(entry.clone()).unwrap();

            // Run cleanup
            let deleted = logger.cleanup_old_logs().unwrap();

            // Verify cleanup behavior
            if days_old > retention_days as i64 {
                // Entry should be deleted
                prop_assert_eq!(deleted, 1);

                let results = logger.query(LogFilter {
                    trace_id: Some(entry.trace_id.clone()),
                    ..Default::default()
                }).unwrap();
                prop_assert_eq!(results.len(), 0);
            } else {
                // Entry should remain
                prop_assert_eq!(deleted, 0);

                let results = logger.query(LogFilter {
                    trace_id: Some(entry.trace_id.clone()),
                    ..Default::default()
                }).unwrap();
                prop_assert_eq!(results.len(), 1);
            }
        }
    }

    // Property 31: Body Logging Size Limit
    // Validates: Requirements 27.4
    proptest! {
        #[test]
        fn prop_body_logging_size_limit(
            body_size in 1usize..5000usize,
            max_size in 100usize..2000usize,
        ) {
            let config = LoggingConfig {
                request_body_logging: true,
                max_body_size_bytes: max_size,
                ..Default::default()
            };
            let (logger, _temp) = create_test_logger_with_config(config);

            // Create body of specific size
            let body = "x".repeat(body_size);

            let entry = LogEntry {
                trace_id: "test-size".to_string(),
                timestamp: Utc::now(),
                method: "POST".to_string(),
                path: "/v1/chat/completions".to_string(),
                model: "gpt-4".to_string(),
                provider: "openai".to_string(),
                status_code: 200,
                duration_ms: 1000,
                cost: 0.01,
                request_body: Some(body.clone()),
                response_body: None,
                requested_model: None,
                responded_model: None,
                compression: None,
                memories_injected: 0,
                memories_stored: 0,
                injection_tokens: 0,
                detected_project: None,
            };

            logger.log(entry).unwrap();

            let results = logger.query(LogFilter {
                trace_id: Some("test-size".to_string()),
                ..Default::default()
            }).unwrap();

            prop_assert_eq!(results.len(), 1);

            if let Some(logged_body) = &results[0].request_body {
                if body_size > max_size {
                    // Body should be truncated
                    prop_assert!(logged_body.contains("[TRUNCATED"));
                    prop_assert!(logged_body.len() > max_size); // Includes truncation message
                } else {
                    // Body should be intact
                    prop_assert_eq!(logged_body, &body);
                }
            }
        }
    }

    // Property 32: Body Logging Field Exclusion
    // Validates: Requirements 27.6
    proptest! {
        #[test]
        fn prop_body_logging_field_exclusion(
            secret_value in "[a-zA-Z0-9]{10,50}",
            public_value in "[a-zA-Z0-9]{10,50}",
        ) {
            let config = LoggingConfig {
                request_body_logging: true,
                excluded_fields: vec!["api_key".to_string(), "password".to_string()],
                ..Default::default()
            };
            let (logger, _temp) = create_test_logger_with_config(config);

            // Create JSON body with excluded fields
            let body = format!(
                r#"{{"api_key":"{}","password":"secret","message":"{}"}}"#,
                secret_value, public_value
            );

            let entry = LogEntry {
                trace_id: "test-exclusion".to_string(),
                timestamp: Utc::now(),
                method: "POST".to_string(),
                path: "/v1/chat/completions".to_string(),
                model: "gpt-4".to_string(),
                provider: "openai".to_string(),
                status_code: 200,
                duration_ms: 1000,
                cost: 0.01,
                request_body: Some(body),
                response_body: None,
                requested_model: None,
                responded_model: None,
                compression: None,
                memories_injected: 0,
                memories_stored: 0,
                injection_tokens: 0,
                detected_project: None,
            };

            logger.log(entry).unwrap();

            let results = logger.query(LogFilter {
                trace_id: Some("test-exclusion".to_string()),
                ..Default::default()
            }).unwrap();

            prop_assert_eq!(results.len(), 1);

            if let Some(logged_body) = &results[0].request_body {
                // Excluded fields should be redacted
                prop_assert!(!logged_body.contains(&secret_value));
                prop_assert!(logged_body.contains("[REDACTED]"));

                // Public fields should remain
                prop_assert!(logged_body.contains(&public_value));
            }
        }
    }

    // Property 10: API Key Redaction
    // Validates: Requirements 19.3, 19.4, 19.9, 19.10
    proptest! {
        #[test]
        fn prop_api_key_redaction(
            message in "[a-zA-Z0-9 ]{10,50}",
        ) {
            let config = LoggingConfig {
                request_body_logging: true,
                ..Default::default()
            };
            let (logger, _temp) = create_test_logger_with_config(config);

            // Create body with API key patterns
            let api_key = "sk-1234567890abcdefghijklmnopqrstuvwxyz1234567890ab";
            let bearer_token = "Bearer eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9";
            let body = format!(
                r#"{{"api_key":"{}","authorization":"{}","message":"{}"}}"#,
                api_key, bearer_token, message
            );

            let entry = LogEntry {
                trace_id: "test-redaction".to_string(),
                timestamp: Utc::now(),
                method: "POST".to_string(),
                path: "/v1/chat/completions".to_string(),
                model: "gpt-4".to_string(),
                provider: "openai".to_string(),
                status_code: 200,
                duration_ms: 1000,
                cost: 0.01,
                request_body: Some(body),
                response_body: None,
                requested_model: None,
                responded_model: None,
                compression: None,
                memories_injected: 0,
                memories_stored: 0,
                injection_tokens: 0,
                detected_project: None,
            };

            logger.log(entry).unwrap();

            let results = logger.query(LogFilter {
                trace_id: Some("test-redaction".to_string()),
                ..Default::default()
            }).unwrap();

            prop_assert_eq!(results.len(), 1);

            if let Some(logged_body) = &results[0].request_body {
                // API keys should be redacted
                prop_assert!(!logged_body.contains(api_key));
                prop_assert!(!logged_body.contains(bearer_token));
                prop_assert!(logged_body.contains("[REDACTED]"));

                // Message should remain
                prop_assert!(logged_body.contains(&message));
            }
        }
    }

    // Property 22: Version Fallback Logging
    // **Validates: Requirements 5.9**
    //
    // For any request where version fallback occurs, the log entry shall contain
    // both the requested model version and the version that successfully responded.
    proptest! {
        #![proptest_config(ProptestConfig {
            cases: 50,
            .. ProptestConfig::default()
        })]

        #[test]
        fn prop_version_fallback_logging(
            requested in prop::sample::select(vec![
                "gpt-4-turbo-2024-04-09",
                "gpt-4-turbo-2024-01-25",
                "claude-3-opus-2024-02-29",
                "llama-3-70b-2024-03-15",
            ]),
            responded in prop::sample::select(vec![
                "gpt-4-turbo-2024-01-25",
                "gpt-4-turbo-2023-11-06",
                "claude-3-opus-2024-01-01",
                "llama-3-70b-2024-01-10",
            ]),
            trace_id in "[a-z0-9]{8,20}",
        ) {
            let config = LoggingConfig::default();
            let (logger, _temp) = create_test_logger_with_config(config);

            // Simulate a version fallback: requested != responded
            let entry = LogEntry {
                trace_id: trace_id.clone(),
                timestamp: Utc::now(),
                method: "POST".to_string(),
                path: "/v1/chat/completions".to_string(),
                model: responded.to_string(),
                provider: "openai".to_string(),
                status_code: 200,
                duration_ms: 1500,
                cost: 0.05,
                request_body: None,
                response_body: None,
                requested_model: Some(requested.to_string()),
                responded_model: Some(responded.to_string()),
                compression: None,
                memories_injected: 0,
                memories_stored: 0,
                injection_tokens: 0,
                detected_project: None,
            };

            logger.log(entry).unwrap();

            let results = logger.query(LogFilter {
                trace_id: Some(trace_id.clone()),
                ..Default::default()
            }).unwrap();

            prop_assert_eq!(results.len(), 1);
            let logged = &results[0];

            // Both requested and responded versions must be recorded
            prop_assert!(
                logged.requested_model.is_some(),
                "requested_model must be recorded when version fallback occurs"
            );
            prop_assert!(
                logged.responded_model.is_some(),
                "responded_model must be recorded when version fallback occurs"
            );
            prop_assert_eq!(
                logged.requested_model.as_deref(),
                Some(requested),
                "requested_model must match the originally requested version"
            );
            prop_assert_eq!(
                logged.responded_model.as_deref(),
                Some(responded),
                "responded_model must match the version that actually responded"
            );
        }
    }
}
