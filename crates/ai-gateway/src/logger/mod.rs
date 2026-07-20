#![allow(dead_code)]
use chrono::{DateTime, Utc};
use rusqlite::{params, Connection};
use serde::{Deserialize, Serialize};
use std::path::Path;
use std::sync::{Arc, Mutex};
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

/// Request logger with SQLite backend
pub struct RequestLogger {
    conn: Arc<Mutex<Connection>>,
    config: LoggingConfig,
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

        Ok(Self {
            conn: Arc::new(Mutex::new(conn)),
            config,
        })
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
                compression_level TEXT
            )",
            [],
        )?;

        Self::ensure_column(conn, "compression_metadata", "TEXT")?;
        Self::ensure_column(conn, "compression_level", "TEXT")?;

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

    /// Log a request entry
    pub fn log(&self, entry: LogEntry) -> Result<()> {
        let conn = self.conn.lock().unwrap();

        // Process request body if logging is enabled
        let request_body = if self.config.request_body_logging {
            entry.request_body.map(|body| self.process_body(&body))
        } else {
            None
        };

        // Process response body if logging is enabled
        let response_body = if self.config.response_body_logging {
            entry.response_body.map(|body| self.process_body(&body))
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
                requested_model, responded_model, compression_metadata, compression_level
            ) VALUES (?1, ?2, ?3, ?4, ?5, ?6, ?7, ?8, ?9, ?10, ?11, ?12, ?13, ?14, ?15)",
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
            ],
        )?;

        Ok(())
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
        let patterns = [
            // OpenAI keys: sk-... with 20+ chars (covers sk-proj-..., sk-svcacct-..., etc.)
            (
                regex::Regex::new(r"sk-[a-zA-Z0-9_\-]{20,}").unwrap(),
                "[REDACTED]",
            ),
            (
                regex::Regex::new(r"Bearer\s+[a-zA-Z0-9\-_.]+").unwrap(),
                "Bearer [REDACTED]",
            ),
            (
                regex::Regex::new(
                    r#"(?i)(api[_-]?key|authorization)["']?\s*[:=]\s*["']?[a-zA-Z0-9\-_.]+"#,
                )
                .unwrap(),
                "$1: [REDACTED]",
            ),
            // AWS access keys: AKIA...
            (
                regex::Regex::new(r"AKIA[A-Z0-9]{16}").unwrap(),
                "[REDACTED]",
            ),
        ];

        let mut result = text.to_string();
        for (re, replacement) in &patterns {
            result = re.replace_all(&result, *replacement).to_string();
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

    /// Query log entries with optional filtering
    pub fn query(&self, filter: LogFilter) -> Result<Vec<LogEntry>> {
        let conn = self.conn.lock().unwrap();

        let mut query = String::from("SELECT trace_id, timestamp, method, path, model, provider, status_code, duration_ms, cost, request_body, response_body, requested_model, responded_model, compression_metadata FROM requests WHERE 1=1");
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
                })
            })?
            .collect::<std::result::Result<Vec<_>, _>>()?;

        Ok(entries)
    }

    /// Flush pending writes and checkpoint the WAL (Req 18.3).
    /// Called during graceful shutdown to ensure all data is persisted.
    pub fn flush(&self) {
        if let Ok(conn) = self.conn.lock() {
            // Checkpoint WAL to ensure all data is written to the main database file
            let _ = conn.execute_batch("PRAGMA wal_checkpoint(TRUNCATE);");
            tracing::info!("RequestLogger flushed and WAL checkpointed");
        }
    }

    /// Clean up old log entries based on retention policy
    pub fn cleanup_old_logs(&self) -> Result<usize> {
        if self.config.retention_days == 0 {
            // Retention disabled
            return Ok(0);
        }

        let conn = self.conn.lock().unwrap();
        let cutoff_timestamp =
            Utc::now().timestamp() - (self.config.retention_days as i64 * 24 * 60 * 60);

        let deleted = conn.execute(
            "DELETE FROM requests WHERE timestamp < ?1",
            params![cutoff_timestamp],
        )?;

        Ok(deleted)
    }
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
        let (logger, _temp) = create_test_logger();
        logger.log(sample_entry("malformed", None)).unwrap();
        logger
            .conn
            .lock()
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
        let (logger, _temp) = create_test_logger();
        let mut metadata = sample_compression(CompressionLevel::Ultra);
        metadata.engines_applied = vec![
            "semantic sk-super-secret-token-1234567890".repeat(10),
            "Bearer another-secret".to_owned(),
        ];
        metadata.savings_percent = f64::NAN;
        logger
            .log(sample_entry("safe-metadata", Some(metadata)))
            .unwrap();

        let conn = logger.conn.lock().unwrap();
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
    fn test_api_key_redaction() {
        let (logger, _temp) = create_test_logger();

        // Standard 48-char key
        let body =
            r#"{"api_key":"sk-1234567890abcdefghijklmnopqrstuvwxyz1234567890ab","message":"test"}"#;
        let redacted = logger.redact_api_keys(body);
        assert!(!redacted.contains("sk-1234567890"));
        assert!(redacted.contains("[REDACTED]"));

        // Longer project-scoped key (sk-proj-...)
        let body2 =
            r#"{"key":"sk-proj-abcdefghijklmnopqrstuvwxyz1234567890abcdefghijklmnopqrstuvwxyz"}"#;
        let redacted2 = logger.redact_api_keys(body2);
        assert!(!redacted2.contains("sk-proj-"));
        assert!(redacted2.contains("[REDACTED]"));

        // AWS access key
        let body3 = r#"{"aws_key":"AKIAIOSFODNN7EXAMPLE"}"#;
        let redacted3 = logger.redact_api_keys(body3);
        assert!(!redacted3.contains("AKIAIOSFODNN7EXAMPLE"));
        assert!(redacted3.contains("[REDACTED]"));
    }

    #[test]
    fn test_field_exclusion() {
        let (logger, _temp) = create_test_logger();

        let body = r#"{"api_key":"secret","password":"pass123","message":"test"}"#;
        let excluded = logger.exclude_fields(body);

        let json: serde_json::Value = serde_json::from_str(&excluded).unwrap();
        assert_eq!(json["api_key"], "[REDACTED]");
        assert_eq!(json["password"], "[REDACTED]");
        assert_eq!(json["message"], "test");
    }

    #[test]
    fn test_size_limit() {
        let (logger, _temp) = create_test_logger();

        let large_body = "x".repeat(2000);
        let limited = logger.apply_size_limit(&large_body);

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
