//! SQLite-backed persistence for virtual keys (`keys.db`).
//!
//! This module owns the schema for the `virtual_keys` and `virtual_key_usage`
//! tables, and provides the CRUD operations used by the higher-level
//! [`super::VirtualKeyManager`]. The connection is wrapped in an
//! `Arc<Mutex<Connection>>` to match the pattern used by the request logger.
//!
//! Timestamps are stored as UTC epoch seconds (`INTEGER`) and converted to and
//! from [`chrono::DateTime<Utc>`] at the boundary.

use std::path::Path;
use std::sync::{Arc, Mutex};

use chrono::{DateTime, Utc};
use rusqlite::types::Type as SqlType;
use rusqlite::{params, Connection, Row};

use super::models::{BudgetWindow, KeyStatus, UsageAggregate, UsageRecord};

/// Errors produced by the [`KeyStore`].
#[derive(Debug, thiserror::Error)]
pub enum KeyStoreError {
    #[error("Database error: {0}")]
    Database(#[from] rusqlite::Error),

    #[error("Serialization error: {0}")]
    Serialization(#[from] serde_json::Error),

    #[error("IO error: {0}")]
    Io(#[from] std::io::Error),

    #[error("Invalid stored data: {0}")]
    InvalidData(String),
}

/// A full row of the `virtual_keys` table.
///
/// This is the persistence-layer representation. Higher layers convert this
/// into `AuthenticatedKey` / `VirtualKeyInfo` as needed (later tasks).
#[derive(Debug, Clone, PartialEq)]
pub struct StoredVirtualKey {
    /// UUID v4 primary key.
    pub id: String,
    /// SHA-256 hash of the full key value, used for authentication lookups.
    pub key_hash: String,
    /// First 8 characters of the key value, for masked display.
    pub key_prefix: String,
    /// AES-256-GCM encrypted full key value.
    pub encrypted_key: String,
    pub name: Option<String>,
    pub status: KeyStatus,
    pub budget_limit_usd: Option<f64>,
    pub token_budget: Option<u64>,
    pub budget_window: Option<BudgetWindow>,
    pub current_spend_usd: f64,
    pub current_tokens_used: u64,
    pub window_start: Option<DateTime<Utc>>,
    pub requests_per_minute: Option<u32>,
    pub tokens_per_minute: Option<u64>,
    pub model_access_list: Option<Vec<String>>,
    pub expires_at: Option<DateTime<Utc>>,
    pub created_at: DateTime<Utc>,
    pub last_used_at: Option<DateTime<Utc>>,
    pub request_count: u64,
}

/// Partial update descriptor for [`KeyStore::update_key`].
///
/// Each field uses `Option` to indicate "leave unchanged" (`None`). Nullable
/// columns use `Option<Option<T>>` so callers can distinguish "unchanged"
/// (`None`) from "set to NULL" (`Some(None)`) and "set to a value"
/// (`Some(Some(v))`).
#[derive(Debug, Clone, Default)]
pub struct KeyUpdates {
    pub name: Option<Option<String>>,
    pub status: Option<KeyStatus>,
    pub budget_limit_usd: Option<Option<f64>>,
    pub token_budget: Option<Option<u64>>,
    pub budget_window: Option<Option<BudgetWindow>>,
    pub requests_per_minute: Option<Option<u32>>,
    pub tokens_per_minute: Option<Option<u64>>,
    pub model_access_list: Option<Option<Vec<String>>>,
    pub expires_at: Option<Option<DateTime<Utc>>>,
    pub window_start: Option<Option<DateTime<Utc>>>,
    pub current_spend_usd: Option<f64>,
    pub current_tokens_used: Option<u64>,
    pub last_used_at: Option<Option<DateTime<Utc>>>,
    pub request_count: Option<u64>,
}

impl KeyUpdates {
    /// Returns `true` when no fields are set (the update would be a no-op).
    fn is_empty(&self) -> bool {
        self.name.is_none()
            && self.status.is_none()
            && self.budget_limit_usd.is_none()
            && self.token_budget.is_none()
            && self.budget_window.is_none()
            && self.requests_per_minute.is_none()
            && self.tokens_per_minute.is_none()
            && self.model_access_list.is_none()
            && self.expires_at.is_none()
            && self.window_start.is_none()
            && self.current_spend_usd.is_none()
            && self.current_tokens_used.is_none()
            && self.last_used_at.is_none()
            && self.request_count.is_none()
    }
}

/// SQLite-backed key store. Cheaply cloneable via the shared connection handle.
pub struct KeyStore {
    conn: Arc<Mutex<Connection>>,
}

impl KeyStore {
    /// Open (or create) the database at `db_path`, run schema migration, and
    /// enable foreign-key enforcement so `ON DELETE CASCADE` works.
    pub fn new(db_path: &Path) -> Result<Self, KeyStoreError> {
        if let Some(parent) = db_path.parent() {
            if !parent.as_os_str().is_empty() {
                std::fs::create_dir_all(parent)?;
            }
        }

        let conn = Connection::open(db_path)?;
        // CASCADE deletes require foreign keys to be enabled per-connection.
        conn.pragma_update(None, "foreign_keys", "ON")?;
        Self::create_schema(&conn)?;

        Ok(Self {
            conn: Arc::new(Mutex::new(conn)),
        })
    }

    /// Create the schema and indexes if they do not already exist.
    fn create_schema(conn: &Connection) -> Result<(), KeyStoreError> {
        conn.execute_batch(
            "CREATE TABLE IF NOT EXISTS virtual_keys (
                id TEXT PRIMARY KEY,
                key_hash TEXT NOT NULL UNIQUE,
                key_prefix TEXT NOT NULL,
                encrypted_key TEXT NOT NULL,
                name TEXT,
                status TEXT NOT NULL DEFAULT 'active',
                budget_limit_usd REAL,
                token_budget INTEGER,
                budget_window TEXT,
                current_spend_usd REAL NOT NULL DEFAULT 0.0,
                current_tokens_used INTEGER NOT NULL DEFAULT 0,
                window_start_timestamp INTEGER,
                requests_per_minute INTEGER,
                tokens_per_minute INTEGER,
                model_access_list TEXT,
                expires_at INTEGER,
                created_at INTEGER NOT NULL,
                last_used_at INTEGER,
                request_count INTEGER NOT NULL DEFAULT 0
            );

            CREATE INDEX IF NOT EXISTS idx_virtual_keys_key_hash ON virtual_keys(key_hash);
            CREATE INDEX IF NOT EXISTS idx_virtual_keys_status ON virtual_keys(status);
            CREATE INDEX IF NOT EXISTS idx_virtual_keys_created_at ON virtual_keys(created_at);
            CREATE INDEX IF NOT EXISTS idx_virtual_keys_name ON virtual_keys(name);

            CREATE TABLE IF NOT EXISTS virtual_key_usage (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                key_id TEXT NOT NULL REFERENCES virtual_keys(id) ON DELETE CASCADE,
                model_group TEXT NOT NULL,
                model TEXT NOT NULL,
                input_tokens INTEGER NOT NULL,
                output_tokens INTEGER NOT NULL,
                cost_usd REAL NOT NULL,
                timestamp INTEGER NOT NULL
            );

            CREATE INDEX IF NOT EXISTS idx_usage_key_id ON virtual_key_usage(key_id);
            CREATE INDEX IF NOT EXISTS idx_usage_timestamp ON virtual_key_usage(timestamp);
            CREATE INDEX IF NOT EXISTS idx_usage_key_timestamp ON virtual_key_usage(key_id, timestamp);",
        )?;

        Ok(())
    }

    /// Insert a new key row.
    pub fn create_key(&self, key: &StoredVirtualKey) -> Result<(), KeyStoreError> {
        let model_access_json = serialize_model_access(&key.model_access_list)?;
        let conn = self.conn.lock().expect("keys.db mutex poisoned");

        conn.execute(
            "INSERT INTO virtual_keys (
                id, key_hash, key_prefix, encrypted_key, name, status,
                budget_limit_usd, token_budget, budget_window,
                current_spend_usd, current_tokens_used, window_start_timestamp,
                requests_per_minute, tokens_per_minute, model_access_list,
                expires_at, created_at, last_used_at, request_count
            ) VALUES (
                ?1, ?2, ?3, ?4, ?5, ?6,
                ?7, ?8, ?9,
                ?10, ?11, ?12,
                ?13, ?14, ?15,
                ?16, ?17, ?18, ?19
            )",
            params![
                key.id,
                key.key_hash,
                key.key_prefix,
                key.encrypted_key,
                key.name,
                key_status_to_str(&key.status),
                key.budget_limit_usd,
                key.token_budget.map(|v| v as i64),
                key.budget_window.as_ref().map(budget_window_to_str),
                key.current_spend_usd,
                key.current_tokens_used as i64,
                key.window_start.map(|t| t.timestamp()),
                key.requests_per_minute.map(|v| v as i64),
                key.tokens_per_minute.map(|v| v as i64),
                model_access_json,
                key.expires_at.map(|t| t.timestamp()),
                key.created_at.timestamp(),
                key.last_used_at.map(|t| t.timestamp()),
                key.request_count as i64,
            ],
        )?;

        Ok(())
    }

    /// Look up a key by its SHA-256 hash (authentication path).
    pub fn get_key_by_hash(
        &self,
        key_hash: &str,
    ) -> Result<Option<StoredVirtualKey>, KeyStoreError> {
        let conn = self.conn.lock().expect("keys.db mutex poisoned");
        let mut stmt = conn.prepare(&format!(
            "SELECT {SELECT_COLUMNS} FROM virtual_keys WHERE key_hash = ?1"
        ))?;
        let mut rows = stmt.query_map(params![key_hash], row_to_stored_key)?;
        match rows.next() {
            Some(row) => Ok(Some(row?)),
            None => Ok(None),
        }
    }

    /// Look up a key by its UUID.
    pub fn get_key_by_id(&self, id: &str) -> Result<Option<StoredVirtualKey>, KeyStoreError> {
        let conn = self.conn.lock().expect("keys.db mutex poisoned");
        let mut stmt = conn.prepare(&format!(
            "SELECT {SELECT_COLUMNS} FROM virtual_keys WHERE id = ?1"
        ))?;
        let mut rows = stmt.query_map(params![id], row_to_stored_key)?;
        match rows.next() {
            Some(row) => Ok(Some(row?)),
            None => Ok(None),
        }
    }

    /// Apply a partial update to the key identified by `id`. A no-op when
    /// `updates` contains no set fields.
    pub fn update_key(&self, id: &str, updates: &KeyUpdates) -> Result<(), KeyStoreError> {
        if updates.is_empty() {
            return Ok(());
        }

        let mut assignments: Vec<String> = Vec::new();
        let mut values: Vec<Box<dyn rusqlite::ToSql>> = Vec::new();

        if let Some(name) = &updates.name {
            assignments.push(format!("name = ?{}", values.len() + 1));
            values.push(Box::new(name.clone()));
        }
        if let Some(status) = &updates.status {
            assignments.push(format!("status = ?{}", values.len() + 1));
            values.push(Box::new(key_status_to_str(status).to_string()));
        }
        if let Some(budget) = &updates.budget_limit_usd {
            assignments.push(format!("budget_limit_usd = ?{}", values.len() + 1));
            values.push(Box::new(*budget));
        }
        if let Some(token_budget) = &updates.token_budget {
            assignments.push(format!("token_budget = ?{}", values.len() + 1));
            values.push(Box::new(token_budget.map(|v| v as i64)));
        }
        if let Some(window) = &updates.budget_window {
            assignments.push(format!("budget_window = ?{}", values.len() + 1));
            values.push(Box::new(window.as_ref().map(budget_window_to_str)));
        }
        if let Some(rpm) = &updates.requests_per_minute {
            assignments.push(format!("requests_per_minute = ?{}", values.len() + 1));
            values.push(Box::new(rpm.map(|v| v as i64)));
        }
        if let Some(tpm) = &updates.tokens_per_minute {
            assignments.push(format!("tokens_per_minute = ?{}", values.len() + 1));
            values.push(Box::new(tpm.map(|v| v as i64)));
        }
        if let Some(model_access) = &updates.model_access_list {
            assignments.push(format!("model_access_list = ?{}", values.len() + 1));
            values.push(Box::new(serialize_model_access(model_access)?));
        }
        if let Some(expires_at) = &updates.expires_at {
            assignments.push(format!("expires_at = ?{}", values.len() + 1));
            values.push(Box::new(expires_at.map(|t| t.timestamp())));
        }
        if let Some(window_start) = &updates.window_start {
            assignments.push(format!("window_start_timestamp = ?{}", values.len() + 1));
            values.push(Box::new(window_start.map(|t| t.timestamp())));
        }
        if let Some(spend) = &updates.current_spend_usd {
            assignments.push(format!("current_spend_usd = ?{}", values.len() + 1));
            values.push(Box::new(*spend));
        }
        if let Some(tokens) = &updates.current_tokens_used {
            assignments.push(format!("current_tokens_used = ?{}", values.len() + 1));
            values.push(Box::new(*tokens as i64));
        }
        if let Some(last_used) = &updates.last_used_at {
            assignments.push(format!("last_used_at = ?{}", values.len() + 1));
            values.push(Box::new(last_used.map(|t| t.timestamp())));
        }
        if let Some(count) = &updates.request_count {
            assignments.push(format!("request_count = ?{}", values.len() + 1));
            values.push(Box::new(*count as i64));
        }

        let sql = format!(
            "UPDATE virtual_keys SET {} WHERE id = ?{}",
            assignments.join(", "),
            values.len() + 1
        );
        values.push(Box::new(id.to_string()));

        let conn = self.conn.lock().expect("keys.db mutex poisoned");
        let param_refs: Vec<&dyn rusqlite::ToSql> = values.iter().map(|v| v.as_ref()).collect();
        conn.execute(&sql, param_refs.as_slice())?;

        Ok(())
    }

    /// Delete a key. Associated usage rows are removed via `ON DELETE CASCADE`.
    pub fn delete_key(&self, id: &str) -> Result<(), KeyStoreError> {
        let conn = self.conn.lock().expect("keys.db mutex poisoned");
        conn.execute("DELETE FROM virtual_keys WHERE id = ?1", params![id])?;
        Ok(())
    }

    /// Return a page of keys ordered by `created_at` descending, along with the
    /// total number of keys in the store.
    pub fn list_keys(
        &self,
        limit: u32,
        offset: u32,
    ) -> Result<(Vec<StoredVirtualKey>, u64), KeyStoreError> {
        let conn = self.conn.lock().expect("keys.db mutex poisoned");

        let total: i64 =
            conn.query_row("SELECT COUNT(*) FROM virtual_keys", [], |row| row.get(0))?;

        let mut stmt = conn.prepare(&format!(
            "SELECT {SELECT_COLUMNS} FROM virtual_keys \
             ORDER BY created_at DESC, id DESC LIMIT ?1 OFFSET ?2"
        ))?;
        let keys = stmt
            .query_map(params![limit as i64, offset as i64], row_to_stored_key)?
            .collect::<Result<Vec<_>, _>>()?;

        Ok((keys, total as u64))
    }

    /// Atomically add to the cumulative spend and token counters, bump the
    /// request count, and record the last-used timestamp.
    pub fn update_usage_counters(
        &self,
        id: &str,
        spend_delta: f64,
        tokens_delta: i64,
    ) -> Result<(), KeyStoreError> {
        let conn = self.conn.lock().expect("keys.db mutex poisoned");
        conn.execute(
            "UPDATE virtual_keys SET
                current_spend_usd = current_spend_usd + ?1,
                current_tokens_used = current_tokens_used + ?2,
                request_count = request_count + 1,
                last_used_at = ?3
             WHERE id = ?4",
            params![spend_delta, tokens_delta, Utc::now().timestamp(), id],
        )?;
        Ok(())
    }

    /// Reset the budget-window counters to zero and set the window start to now.
    pub fn reset_window_counters(&self, id: &str) -> Result<(), KeyStoreError> {
        let conn = self.conn.lock().expect("keys.db mutex poisoned");
        conn.execute(
            "UPDATE virtual_keys SET
                current_spend_usd = 0.0,
                current_tokens_used = 0,
                window_start_timestamp = ?1
             WHERE id = ?2",
            params![Utc::now().timestamp(), id],
        )?;
        Ok(())
    }

    /// Insert a single usage row into `virtual_key_usage`.
    ///
    /// Records the per-request breakdown (model group/model, token counts,
    /// computed cost, UTC timestamp) used for later aggregation. The caller is
    /// responsible for advancing the key's cumulative counters (see
    /// [`KeyStore::update_usage_counters`]).
    ///
    /// _Requirements: 9.1, 9.6_
    pub fn insert_usage_record(&self, record: &UsageRecord) -> Result<(), KeyStoreError> {
        let conn = self.conn.lock().expect("keys.db mutex poisoned");
        conn.execute(
            "INSERT INTO virtual_key_usage (
                key_id, model_group, model, input_tokens, output_tokens,
                cost_usd, timestamp
            ) VALUES (?1, ?2, ?3, ?4, ?5, ?6, ?7)",
            params![
                record.key_id,
                record.model_group,
                record.model,
                record.input_tokens as i64,
                record.output_tokens as i64,
                record.cost_usd,
                record.timestamp.timestamp(),
            ],
        )?;
        Ok(())
    }

    /// Aggregate usage rows for `key_id` whose timestamp falls within
    /// `[start, end]` inclusive.
    ///
    /// Sums `cost_usd`, `input_tokens`, and `output_tokens`, and counts the
    /// matching rows. Returns zero values when no rows fall in the range.
    ///
    /// _Requirements: 9.2, 9.4_
    pub fn query_aggregate(
        &self,
        key_id: &str,
        start: DateTime<Utc>,
        end: DateTime<Utc>,
    ) -> Result<UsageAggregate, KeyStoreError> {
        let conn = self.conn.lock().expect("keys.db mutex poisoned");
        // COALESCE guards against NULL SUMs when no rows match; COUNT(*) is
        // already 0 in that case. Bounds are inclusive on both ends.
        let aggregate = conn.query_row(
            "SELECT
                COALESCE(SUM(cost_usd), 0.0),
                COALESCE(SUM(input_tokens), 0),
                COALESCE(SUM(output_tokens), 0),
                COUNT(*)
             FROM virtual_key_usage
             WHERE key_id = ?1 AND timestamp >= ?2 AND timestamp <= ?3",
            params![key_id, start.timestamp(), end.timestamp()],
            |row| {
                let total_spend_usd: f64 = row.get(0)?;
                let total_input_tokens: i64 = row.get(1)?;
                let total_output_tokens: i64 = row.get(2)?;
                let total_requests: i64 = row.get(3)?;
                Ok(UsageAggregate {
                    total_spend_usd,
                    total_input_tokens: total_input_tokens as u64,
                    total_output_tokens: total_output_tokens as u64,
                    total_requests: total_requests as u64,
                })
            },
        )?;
        Ok(aggregate)
    }
}

/// Column list shared by all `SELECT` statements, kept in sync with
/// [`row_to_stored_key`].
const SELECT_COLUMNS: &str = "id, key_hash, key_prefix, encrypted_key, name, status, \
     budget_limit_usd, token_budget, budget_window, current_spend_usd, \
     current_tokens_used, window_start_timestamp, requests_per_minute, \
     tokens_per_minute, model_access_list, expires_at, created_at, \
     last_used_at, request_count";

/// Map a row (selected via [`SELECT_COLUMNS`]) into a [`StoredVirtualKey`].
fn row_to_stored_key(row: &Row) -> rusqlite::Result<StoredVirtualKey> {
    let status_str: String = row.get(5)?;
    let status = key_status_from_str(&status_str)
        .map_err(|e| rusqlite::Error::FromSqlConversionFailure(5, SqlType::Text, Box::new(e)))?;

    let budget_window_str: Option<String> = row.get(8)?;
    let budget_window = match budget_window_str {
        Some(s) => Some(budget_window_from_str(&s).map_err(|e| {
            rusqlite::Error::FromSqlConversionFailure(8, SqlType::Text, Box::new(e))
        })?),
        None => None,
    };

    let model_access_json: Option<String> = row.get(14)?;
    let model_access_list = match model_access_json {
        Some(json) => Some(serde_json::from_str::<Vec<String>>(&json).map_err(|e| {
            rusqlite::Error::FromSqlConversionFailure(14, SqlType::Text, Box::new(e))
        })?),
        None => None,
    };

    let token_budget: Option<i64> = row.get(7)?;
    let requests_per_minute: Option<i64> = row.get(12)?;
    let tokens_per_minute: Option<i64> = row.get(13)?;
    let window_start_ts: Option<i64> = row.get(11)?;
    let expires_at_ts: Option<i64> = row.get(15)?;
    let created_at_ts: i64 = row.get(16)?;
    let last_used_at_ts: Option<i64> = row.get(17)?;
    let current_tokens_used: i64 = row.get(10)?;
    let request_count: i64 = row.get(18)?;

    Ok(StoredVirtualKey {
        id: row.get(0)?,
        key_hash: row.get(1)?,
        key_prefix: row.get(2)?,
        encrypted_key: row.get(3)?,
        name: row.get(4)?,
        status,
        budget_limit_usd: row.get(6)?,
        token_budget: token_budget.map(|v| v as u64),
        budget_window,
        current_spend_usd: row.get(9)?,
        current_tokens_used: current_tokens_used as u64,
        window_start: epoch_to_datetime_opt(window_start_ts, 11)?,
        requests_per_minute: requests_per_minute.map(|v| v as u32),
        tokens_per_minute: tokens_per_minute.map(|v| v as u64),
        model_access_list,
        expires_at: epoch_to_datetime_opt(expires_at_ts, 15)?,
        created_at: epoch_to_datetime(created_at_ts, 16)?,
        last_used_at: epoch_to_datetime_opt(last_used_at_ts, 17)?,
        request_count: request_count as u64,
    })
}

fn epoch_to_datetime(secs: i64, col: usize) -> rusqlite::Result<DateTime<Utc>> {
    DateTime::from_timestamp(secs, 0).ok_or_else(|| {
        rusqlite::Error::FromSqlConversionFailure(
            col,
            SqlType::Integer,
            Box::new(KeyStoreError::InvalidData(format!(
                "invalid epoch timestamp: {secs}"
            ))),
        )
    })
}

fn epoch_to_datetime_opt(
    secs: Option<i64>,
    col: usize,
) -> rusqlite::Result<Option<DateTime<Utc>>> {
    match secs {
        Some(s) => Ok(Some(epoch_to_datetime(s, col)?)),
        None => Ok(None),
    }
}

fn serialize_model_access(list: &Option<Vec<String>>) -> Result<Option<String>, KeyStoreError> {
    match list {
        Some(items) => Ok(Some(serde_json::to_string(items)?)),
        None => Ok(None),
    }
}

fn key_status_to_str(status: &KeyStatus) -> &'static str {
    match status {
        KeyStatus::Active => "active",
        KeyStatus::Expired => "expired",
        KeyStatus::Revoked => "revoked",
    }
}

fn key_status_from_str(s: &str) -> Result<KeyStatus, KeyStoreError> {
    match s {
        "active" => Ok(KeyStatus::Active),
        "expired" => Ok(KeyStatus::Expired),
        "revoked" => Ok(KeyStatus::Revoked),
        other => Err(KeyStoreError::InvalidData(format!(
            "unknown key status: {other}"
        ))),
    }
}

fn budget_window_to_str(window: &BudgetWindow) -> &'static str {
    match window {
        BudgetWindow::Daily => "daily",
        BudgetWindow::Weekly => "weekly",
        BudgetWindow::Monthly => "monthly",
    }
}

fn budget_window_from_str(s: &str) -> Result<BudgetWindow, KeyStoreError> {
    match s {
        "daily" => Ok(BudgetWindow::Daily),
        "weekly" => Ok(BudgetWindow::Weekly),
        "monthly" => Ok(BudgetWindow::Monthly),
        other => Err(KeyStoreError::InvalidData(format!(
            "unknown budget window: {other}"
        ))),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use chrono::TimeZone;
    use tempfile::NamedTempFile;

    fn temp_store() -> (KeyStore, NamedTempFile) {
        let temp = NamedTempFile::new().unwrap();
        let store = KeyStore::new(temp.path()).unwrap();
        (store, temp)
    }

    fn ts(secs: i64) -> DateTime<Utc> {
        Utc.timestamp_opt(secs, 0).unwrap()
    }

    fn sample_key(id: &str, created_at: i64) -> StoredVirtualKey {
        StoredVirtualKey {
            id: id.to_string(),
            key_hash: format!("hash-{id}"),
            key_prefix: "vk_12345".to_string(),
            encrypted_key: format!("enc-{id}"),
            name: Some(format!("key {id}")),
            status: KeyStatus::Active,
            budget_limit_usd: Some(100.0),
            token_budget: Some(50_000),
            budget_window: Some(BudgetWindow::Daily),
            current_spend_usd: 0.0,
            current_tokens_used: 0,
            window_start: Some(ts(created_at)),
            requests_per_minute: Some(60),
            tokens_per_minute: Some(10_000),
            model_access_list: Some(vec!["gpt-4".to_string(), "claude".to_string()]),
            expires_at: Some(ts(created_at + 86_400)),
            created_at: ts(created_at),
            last_used_at: None,
            request_count: 0,
        }
    }

    #[test]
    fn create_and_get_round_trip() {
        let (store, _tmp) = temp_store();
        let key = sample_key("id-1", 1_700_000_000);
        store.create_key(&key).unwrap();

        let by_id = store.get_key_by_id("id-1").unwrap().unwrap();
        assert_eq!(by_id, key);

        let by_hash = store.get_key_by_hash("hash-id-1").unwrap().unwrap();
        assert_eq!(by_hash, key);
    }

    #[test]
    fn get_missing_returns_none() {
        let (store, _tmp) = temp_store();
        assert!(store.get_key_by_id("nope").unwrap().is_none());
        assert!(store.get_key_by_hash("nope").unwrap().is_none());
    }

    #[test]
    fn null_optionals_round_trip() {
        let (store, _tmp) = temp_store();
        let mut key = sample_key("id-null", 1_700_000_100);
        key.name = None;
        key.budget_limit_usd = None;
        key.token_budget = None;
        key.budget_window = None;
        key.window_start = None;
        key.requests_per_minute = None;
        key.tokens_per_minute = None;
        key.model_access_list = None;
        key.expires_at = None;
        key.last_used_at = None;
        store.create_key(&key).unwrap();

        let loaded = store.get_key_by_id("id-null").unwrap().unwrap();
        assert_eq!(loaded, key);
    }

    #[test]
    fn update_key_partial_preserves_omitted() {
        let (store, _tmp) = temp_store();
        let key = sample_key("id-upd", 1_700_000_200);
        store.create_key(&key).unwrap();

        // Change only the name and clear the budget window.
        let updates = KeyUpdates {
            name: Some(Some("renamed".to_string())),
            budget_window: Some(None),
            ..Default::default()
        };
        store.update_key("id-upd", &updates).unwrap();

        let loaded = store.get_key_by_id("id-upd").unwrap().unwrap();
        assert_eq!(loaded.name.as_deref(), Some("renamed"));
        assert_eq!(loaded.budget_window, None);
        // Untouched fields retain their values.
        assert_eq!(loaded.budget_limit_usd, Some(100.0));
        assert_eq!(loaded.requests_per_minute, Some(60));
        assert_eq!(loaded.model_access_list, key.model_access_list);
    }

    #[test]
    fn update_key_empty_is_noop() {
        let (store, _tmp) = temp_store();
        let key = sample_key("id-noop", 1_700_000_250);
        store.create_key(&key).unwrap();

        store.update_key("id-noop", &KeyUpdates::default()).unwrap();
        assert_eq!(store.get_key_by_id("id-noop").unwrap().unwrap(), key);
    }

    #[test]
    fn delete_key_removes_row() {
        let (store, _tmp) = temp_store();
        let key = sample_key("id-del", 1_700_000_300);
        store.create_key(&key).unwrap();
        store.delete_key("id-del").unwrap();
        assert!(store.get_key_by_id("id-del").unwrap().is_none());
    }

    #[test]
    fn list_keys_orders_by_created_desc_with_total() {
        let (store, _tmp) = temp_store();
        store.create_key(&sample_key("a", 100)).unwrap();
        store.create_key(&sample_key("b", 300)).unwrap();
        store.create_key(&sample_key("c", 200)).unwrap();

        let (page, total) = store.list_keys(10, 0).unwrap();
        assert_eq!(total, 3);
        let ids: Vec<&str> = page.iter().map(|k| k.id.as_str()).collect();
        assert_eq!(ids, vec!["b", "c", "a"]);

        // Pagination: limit 2, offset 1 -> next two by descending created_at.
        let (page2, total2) = store.list_keys(2, 1).unwrap();
        assert_eq!(total2, 3);
        let ids2: Vec<&str> = page2.iter().map(|k| k.id.as_str()).collect();
        assert_eq!(ids2, vec!["c", "a"]);
    }

    #[test]
    fn update_usage_counters_accumulates() {
        let (store, _tmp) = temp_store();
        store.create_key(&sample_key("id-usage", 1_700_000_400)).unwrap();

        store.update_usage_counters("id-usage", 1.5, 100).unwrap();
        store.update_usage_counters("id-usage", 2.25, 50).unwrap();

        let loaded = store.get_key_by_id("id-usage").unwrap().unwrap();
        assert!((loaded.current_spend_usd - 3.75).abs() < 1e-9);
        assert_eq!(loaded.current_tokens_used, 150);
        assert_eq!(loaded.request_count, 2);
        assert!(loaded.last_used_at.is_some());
    }

    #[test]
    fn reset_window_counters_zeroes_spend_and_tokens() {
        let (store, _tmp) = temp_store();
        store.create_key(&sample_key("id-reset", 1_700_000_500)).unwrap();
        store.update_usage_counters("id-reset", 10.0, 500).unwrap();

        store.reset_window_counters("id-reset").unwrap();

        let loaded = store.get_key_by_id("id-reset").unwrap().unwrap();
        assert_eq!(loaded.current_spend_usd, 0.0);
        assert_eq!(loaded.current_tokens_used, 0);
        assert!(loaded.window_start.is_some());
        // request_count is unaffected by a window reset.
        assert_eq!(loaded.request_count, 1);
    }

    #[test]
    fn delete_cascades_usage_rows() {
        let (store, _tmp) = temp_store();
        store.create_key(&sample_key("id-cascade", 1_700_000_600)).unwrap();
        {
            let conn = store.conn.lock().unwrap();
            conn.execute(
                "INSERT INTO virtual_key_usage
                    (key_id, model_group, model, input_tokens, output_tokens, cost_usd, timestamp)
                 VALUES ('id-cascade', 'grp', 'gpt-4', 10, 20, 0.01, 1700000600)",
                [],
            )
            .unwrap();
        }
        store.delete_key("id-cascade").unwrap();

        let conn = store.conn.lock().unwrap();
        let remaining: i64 = conn
            .query_row(
                "SELECT COUNT(*) FROM virtual_key_usage WHERE key_id = 'id-cascade'",
                [],
                |row| row.get(0),
            )
            .unwrap();
        assert_eq!(remaining, 0);
    }

    #[test]
    fn duplicate_hash_rejected() {
        let (store, _tmp) = temp_store();
        let key = sample_key("id-dup1", 1_700_000_700);
        store.create_key(&key).unwrap();
        let mut dup = sample_key("id-dup2", 1_700_000_701);
        dup.key_hash = key.key_hash.clone();
        assert!(store.create_key(&dup).is_err());
    }
}
