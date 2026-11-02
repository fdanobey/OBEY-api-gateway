//! SQLite-backed ledger for budget-alert webhook events and delivery
//! attempts (`webhooks.db`, a sibling of the logs database — same
//! parent rule as `budgets.db`).
//!
//! This module owns the `alert_events` and `deliveries` tables: the
//! durable record of every emitted budget alert and each per-attempt
//! delivery outcome, backing the admin ledger view and the dashboard
//! recent-alerts snapshot (Reqs 4.3, 4.7, 4.8, 5.4). The connection
//! discipline mirrors [`crate::metrics::provider_budget_store::ProviderBudgetStore`]
//! verbatim: WAL + `synchronous=NORMAL` + `busy_timeout`, a single
//! `Arc<Mutex<Connection>>`, idempotent schema init inside one
//! transaction with `ensure_column` evolution, and no Tokio APIs —
//! callers bridge to blocking threads via `spawn_blocking`.
//!
//! The ledger is observability, not control (design, Error Handling 3):
//! read failures log a `warn` and degrade to empty results; write
//! failures log a `warn` and surface the error without panicking, so
//! webhook delivery never depends on this store succeeding.
//!
//! Timestamps are stored as UTC epoch seconds (`INTEGER`).

use std::path::Path;
use std::sync::{Arc, Mutex};
use std::time::{Duration, Instant};

use chrono::Utc;
use rusqlite::{params, Connection, Row};

/// Retention bound: the newest N events (and their deliveries) kept by
/// [`WebhookLedgerStore::insert_event`] — bounded growth, zero ops
/// burden (design, ledger schema).
const EVENT_RETENTION_LIMIT: i64 = 500;

/// TTL of the cached recent-events rows served to dashboard snapshots;
/// mirror of `BUDGET_STORE_ROWS_TTL` in the metrics module.
const RECENT_EVENTS_CACHE_TTL: Duration = Duration::from_secs(1);

/// Outcome value marking an attempt IN PROGRESS: the delivery engine
/// writes the row before network I/O and overwrites it with the
/// final outcome when the attempt completes. A chain whose latest row
/// is still `inflight` when the gateway restarts is the crash
/// signature [`WebhookLedgerStore::park_unfinished_chains`] looks
/// for — including a crash during the very FIRST attempt (previously
/// invisible: no row, no chain).
///
/// Outcome taxonomy for the other values: `success` / `parked` /
/// `skipped_no_endpoints` are terminal; `failed` rows are written
/// only for attempts the chain will retry past (or that end it
/// without retries, like a first-attempt SSRF rejection) — an
/// all-`failed` chain is terminal by construction (only the final
/// attempt transitions to `parked`), so it must NOT be parked as
/// "interrupted".
pub(crate) const OUTCOME_INFLIGHT: &str = "inflight";

/// Errors produced by the [`WebhookLedgerStore`].
#[derive(Debug, thiserror::Error)]
pub enum WebhookLedgerError {
    #[error("Database error: {0}")]
    Database(#[from] rusqlite::Error),

    #[error("IO error: {0}")]
    Io(#[from] std::io::Error),
}

/// A row of the `alert_events` table.
#[derive(Debug, Clone, PartialEq)]
pub struct AlertEventRow {
    /// Event id (uuid, primary key).
    pub id: String,
    /// Event type, e.g. `budget.warning`.
    pub event_type: String,
    /// `provider` | `virtual_key`.
    pub scope: String,
    /// Provider name or `key:<id>:<name>`.
    pub scope_label: String,
    /// Serialized event payload at emit time.
    pub payload_json: String,
    /// UTC epoch seconds.
    pub created_at: i64,
}

/// A row of the `deliveries` table (one per attempt).
#[derive(Debug, Clone, PartialEq)]
pub struct DeliveryRow {
pub id: i64,
pub event_id: String,
/// URL at attempt time (endpoint identity).
pub endpoint_url: String,
/// 1..=4.
pub attempt: i64,
/// `inflight` (attempt started, not yet finished) | `success` |
/// `failed` (attempt finished non-terminal — chain continues or
/// ended without retries) | `parked` | `skipped_no_endpoints`.
pub outcome: String,
/// HTTP status when a response arrived.
pub status_code: Option<i64>,
/// Error class when no response arrived.
pub error: Option<String>,
/// UTC epoch seconds.
pub attempted_at: i64,
}

/// Entry of the 1-second TTL recent-events cache (see
/// [`WebhookLedgerStore::recent_events_cached`]).
struct RecentEventsCacheEntry {
    at: Instant,
    limit: u32,
    rows: Vec<AlertEventRow>,
}

/// SQLite-backed webhook event/delivery ledger. Shared by cloning the
/// store's `Arc` (mirror of the virtual-key and provider-budget
/// stores); contains no Tokio APIs.
pub struct WebhookLedgerStore {
    conn: Arc<Mutex<Connection>>,
    /// 1-second TTL cache of the `recent_events` result for the
    /// dashboard snapshot, so snapshot polling does not hit SQLite on
    /// every read. Guarded by a short-lived Mutex; see
    /// `cached_budget_store_rows` in the metrics module.
    events_cache: Mutex<Option<RecentEventsCacheEntry>>,
}

// Manual impl: the connection has no meaningful debug representation
// (same as the provider-budget store).
impl std::fmt::Debug for WebhookLedgerStore {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("WebhookLedgerStore")
            .finish_non_exhaustive()
    }
}

impl WebhookLedgerStore {
    /// Open (or create) the database at `db_path`, run the idempotent
    /// schema migration, and apply the same WAL/busy-timeout discipline
    /// as the provider-budget store so bursts of off-path ledger writes
    /// cannot stall the process on fsyncs.
    pub fn new(db_path: &Path) -> Result<Self, WebhookLedgerError> {
        if let Some(parent) = db_path.parent() {
            if !parent.as_os_str().is_empty() {
                std::fs::create_dir_all(parent)?;
            }
        }

        let conn = Connection::open(db_path)?;
        // See `ProviderBudgetStore::new`: WAL + synchronous=NORMAL
        // removes the per-commit fsync and `busy_timeout` makes lock
        // contention wait rather than fail. Foreign keys are enforced
        // so delivery rows can never outlive their event.
        conn.busy_timeout(Duration::from_secs(5))?;
        conn.pragma_update(None, "journal_mode", "WAL")?;
        conn.pragma_update(None, "synchronous", "NORMAL")?;
        conn.pragma_update(None, "foreign_keys", "ON")?;
        let store = Self {
            conn: Arc::new(Mutex::new(conn)),
            events_cache: Mutex::new(None),
        };
        store.init()?;
        Ok(store)
    }

    /// Open an in-memory (`:memory:`) ledger: same schema, no disk
    /// file. Degraded-mode sentinel for gateway startup when the
    /// on-disk `webhooks.db` cannot be opened (budget-alerts-unified-
    /// webhooks, Task 5) — alert delivery keeps working, event
    /// history is lost on restart.
    pub fn new_in_memory() -> Self {
        let conn = Connection::open_in_memory().expect("in-memory webhooks ledger");
        let _ = conn.busy_timeout(Duration::from_secs(5));
        let _ = conn.pragma_update(None, "foreign_keys", "ON");
        let store = Self {
            conn: Arc::new(Mutex::new(conn)),
            events_cache: Mutex::new(None),
        };
        store.init().expect("in-memory webhooks ledger schema");
        store
    }

    /// Idempotent schema creation and migration, run inside a single
    /// short transaction so a crash mid-migration never leaves a
    /// half-evolved schema. Safe to call repeatedly (e.g. again after
    /// `new`).
    pub fn init(&self) -> Result<(), WebhookLedgerError> {
        let mut conn = self.conn.lock().expect("webhooks.db mutex poisoned");
        let tx = conn.transaction()?;
        tx.execute_batch(
            "CREATE TABLE IF NOT EXISTS alert_events (
                id TEXT PRIMARY KEY,
                type TEXT NOT NULL,
                scope TEXT NOT NULL,
                scope_label TEXT NOT NULL,
                payload_json TEXT NOT NULL,
                created_at INTEGER NOT NULL
            );
            CREATE TABLE IF NOT EXISTS deliveries (
                id INTEGER PRIMARY KEY,
                event_id TEXT NOT NULL REFERENCES alert_events(id),
                endpoint_url TEXT NOT NULL,
                attempt INTEGER NOT NULL,
                outcome TEXT NOT NULL,
                status_code INTEGER,
                error TEXT,
                attempted_at INTEGER NOT NULL
            );
            CREATE INDEX IF NOT EXISTS idx_deliveries_event ON deliveries(event_id);
            CREATE INDEX IF NOT EXISTS idx_deliveries_endpoint_time
                ON deliveries(endpoint_url, attempted_at);
            CREATE INDEX IF NOT EXISTS idx_alert_events_time ON alert_events(created_at);",
        )?;
        // ensure_column-style evolution, mirroring the provider-budget
        // store: no-ops when the columns already exist, adds them for
        // any earlier partial schema.
        Self::ensure_column(&tx, "alert_events", "scope_label", "TEXT NOT NULL DEFAULT ''")?;
        Self::ensure_column(&tx, "alert_events", "payload_json", "TEXT NOT NULL DEFAULT '{}'")?;
        Self::ensure_column(&tx, "deliveries", "status_code", "INTEGER")?;
        Self::ensure_column(&tx, "deliveries", "error", "TEXT")?;
        tx.commit()?;
        Ok(())
    }

    /// Add `column` (with SQL declaration `decl`) to `table` unless it
    /// already exists.
    fn ensure_column(
        conn: &Connection,
        table: &str,
        column: &str,
        decl: &str,
    ) -> Result<(), WebhookLedgerError> {
        let exists = conn
            .prepare(&format!("PRAGMA table_info({table})"))?
            .query_map([], |row| row.get::<_, String>(1))?
            .collect::<Result<Vec<_>, _>>()?
            .iter()
            .any(|name| name == column);
        if !exists {
            conn.execute(
                &format!("ALTER TABLE {table} ADD COLUMN {column} {decl}"),
                [],
            )?;
        }
        Ok(())
    }

    /// Record an emitted event, then run the count-bound retention pass
    /// inside the same transaction: keep the newest
    /// [`EVENT_RETENTION_LIMIT`] events and delete the rest together
    /// with their deliveries (deliveries first — the FK on
    /// `event_id` requires children to go before parents).
    ///
    /// Write path: failures log a `warn` and return the error; the
    /// caller (delivery code) proceeds regardless — the ledger is
    /// observability, not control.
    pub fn insert_event(
        &self,
        id: &str,
        event_type: &str,
        scope: &str,
        scope_label: &str,
        payload_json: &str,
        created_at: i64,
    ) -> Result<(), WebhookLedgerError> {
        let mut conn = self.conn.lock().expect("webhooks.db mutex poisoned");
        let result = insert_event_with_retention(
            &mut conn,
            id,
            event_type,
            scope,
            scope_label,
            payload_json,
            created_at,
        );
        if let Err(error) = &result {
            tracing::warn!(error = %error, "webhooks ledger: insert_event failed");
        }
        result?;
        Ok(())
    }

/// Record one delivery attempt for `event_id` against
/// `endpoint_url`. `outcome` is one of `inflight | success | failed |
/// parked | skipped_no_endpoints` (validated by the delivery layer,
/// not the store). Write path: failures log a `warn` and return the
/// error.
pub fn record_attempt(
&self,
event_id: &str,
endpoint_url: &str,
attempt: i64,
outcome: &str,
status_code: Option<i64>,
error: Option<&str>,
attempted_at: i64,
) -> Result<(), WebhookLedgerError> {
let conn = self.conn.lock().expect("webhooks.db mutex poisoned");
let result = conn.execute(
"INSERT INTO deliveries
(event_id, endpoint_url, attempt, outcome, status_code, error, attempted_at)
VALUES (?1, ?2, ?3, ?4, ?5, ?6, ?7)",
params![event_id, endpoint_url, attempt, outcome, status_code, error, attempted_at],
);
match result {
Ok(_) => Ok(()),
Err(db_error) => {
tracing::warn!(error = %db_error, "webhooks ledger: record_attempt failed");
Err(WebhookLedgerError::Database(db_error))
}
}
}

/// Mark one delivery attempt as IN PROGRESS (design "Error Handling 5"
/// crash-visibility): the delivery engine writes this row BEFORE any
/// network I/O for the attempt and calls
/// [`Self::finish_inflight_attempt`] with the terminal outcome when
/// the attempt completes. A crash between the two leaves the row
/// `inflight`, which is exactly the signature
/// [`Self::park_unfinished_chains`] parks at startup — including a
/// crash during the FIRST attempt of a chain (no completed row
/// otherwise exists). Write path: failures log a `warn` and return
/// the error.
pub fn record_inflight_attempt(
&self,
event_id: &str,
endpoint_url: &str,
attempt: i64,
attempted_at: i64,
) -> Result<(), WebhookLedgerError> {
self.record_attempt(
event_id,
endpoint_url,
attempt,
OUTCOME_INFLIGHT,
None,
None,
attempted_at,
)
}

/// Overwrite the attempt row written by
/// [`Self::record_inflight_attempt`] with its final outcome. The
/// (event, endpoint, attempt) triple identifies the row — the inflight
/// marker is unique per attempt because only one chain runs per
/// (event, endpoint) at a time. Write path: failures log a `warn`
/// and return the error.
pub fn finish_inflight_attempt(
&self,
event_id: &str,
endpoint_url: &str,
attempt: i64,
outcome: &str,
status_code: Option<i64>,
error: Option<&str>,
attempted_at: i64,
) -> Result<(), WebhookLedgerError> {
let conn = self.conn.lock().expect("webhooks.db mutex poisoned");
let result = conn.execute(
"UPDATE deliveries
SET outcome = ?4, status_code = ?5, error = ?6, attempted_at = ?7
WHERE event_id = ?1 AND endpoint_url = ?2 AND attempt = ?3
AND outcome = 'inflight'",
params![event_id, endpoint_url, attempt, outcome, status_code, error, attempted_at],
);
match result {
Ok(_) => Ok(()),
Err(db_error) => {
tracing::warn!(error = %db_error, "webhooks ledger: finish_inflight_attempt failed");
Err(WebhookLedgerError::Database(db_error))
}
}
}

    /// Record an event that was emitted while the subsystem was active
    /// but had no subscribing endpoint: a single `deliveries` row with
    /// outcome `skipped_no_endpoints` and an empty `endpoint_url`
    /// (there was no endpoint to name). Write path: failures log a
    /// `warn` and return the error.
    pub fn record_undelivered(&self, event_id: &str) -> Result<(), WebhookLedgerError> {
        let conn = self.conn.lock().expect("webhooks.db mutex poisoned");
        let result = conn.execute(
            "INSERT INTO deliveries
                (event_id, endpoint_url, attempt, outcome, status_code, error, attempted_at)
            VALUES (?1, '', 1, 'skipped_no_endpoints', NULL, NULL, ?2)",
            params![event_id, now_epoch()],
        );
        match result {
            Ok(_) => Ok(()),
            Err(db_error) => {
                tracing::warn!(error = %db_error, "webhooks ledger: record_undelivered failed");
                Err(WebhookLedgerError::Database(db_error))
            }
        }
    }

/// Startup pass: any delivery chain (per event and endpoint) whose
/// LATEST attempt row is a non-terminal `inflight` marker (the
/// attempt row written before network I/O and never finished) is from
/// a chain interrupted by a restart — append one `parked` row so the
/// panel shows it as a failed delivery (never silently resumed, never
/// silently dropped; design, Error Handling 5). Chains whose attempts
/// all finished `failed` are TERMINAL (an all-failed chain exhausted
/// its attempts or ended without retries, e.g. a first-attempt SSRF
/// rejection — only the final attempt writes the terminal `parked`
/// row) and are never touched. Returns the number of chains parked; a
/// second run parks nothing.
///
/// Write path: failures log a `warn` and return the error.
pub fn park_unfinished_chains(&self) -> Result<usize, WebhookLedgerError> {
let mut conn = self.conn.lock().expect("webhooks.db mutex poisoned");
let result = park_unfinished_chains_locked(&mut conn);
if let Err(error) = &result {
tracing::warn!(error = %error, "webhooks ledger: park_unfinished_chains failed");
}
result.map_err(WebhookLedgerError::Database)
}

    /// The newest `limit` events, newest first. Read path: failures
    /// log a `warn` and degrade to an empty list (design, Error
    /// Handling 3).
    pub fn recent_events(&self, limit: u32) -> Vec<AlertEventRow> {
        let conn = self.conn.lock().expect("webhooks.db mutex poisoned");
        let mut stmt = match conn.prepare(
            "SELECT id, type, scope, scope_label, payload_json, created_at
            FROM alert_events
            ORDER BY created_at DESC, rowid DESC
            LIMIT ?1",
        ) {
            Ok(stmt) => stmt,
            Err(error) => {
                tracing::warn!(error = %error, "webhooks ledger: recent_events query failed");
                return Vec::new();
            }
        };
        let rows = match stmt.query_map(params![limit as i64], row_to_alert_event) {
            Ok(rows) => rows
                .collect::<Result<Vec<_>, _>>()
                .unwrap_or_else(|error| {
                    tracing::warn!(error = %error, "webhooks ledger: recent_events row read failed");
                    Vec::new()
                }),
            Err(error) => {
                tracing::warn!(error = %error, "webhooks ledger: recent_events query failed");
                Vec::new()
            }
        };
    rows
}

/// [`Self::recent_events`] with an optional SQL-level `webhook.test`
/// exclusion applied BEFORE the limit — the dashboard feed (design
/// Component 5) shows budget events only, so admin test sends can
/// never crowd out real alert history within the newest-20 window.
fn recent_events_filtered(&self, limit: u32, exclude_tests: bool) -> Vec<AlertEventRow> {
    let conn = self.conn.lock().expect("webhooks.db mutex poisoned");
    let mut stmt = match conn.prepare(
        "SELECT id, type, scope, scope_label, payload_json, created_at
        FROM alert_events
        WHERE CASE WHEN ?2 THEN type != 'webhook.test' ELSE 1 END
        ORDER BY created_at DESC, rowid DESC
        LIMIT ?1",
    ) {
        Ok(stmt) => stmt,
        Err(error) => {
            tracing::warn!(error = %error, "webhooks ledger: recent_events query failed");
            return Vec::new();
        }
    };
    let rows = match stmt.query_map(
        params![limit as i64, exclude_tests],
        row_to_alert_event,
    ) {
        Ok(rows) => rows
            .collect::<Result<Vec<_>, _>>()
            .unwrap_or_else(|error| {
                tracing::warn!(error = %error, "webhooks ledger: recent_events row read failed");
                Vec::new()
            }),
        Err(error) => {
            tracing::warn!(error = %error, "webhooks ledger: recent_events query failed");
            Vec::new()
        }
    };
    rows
}

/// The newest `limit` delivery attempts for one endpoint URL (or,
/// with `None`, across ALL endpoints), newest first. Read path:
/// failures log a `warn` and degrade to an empty list.
pub fn recent_deliveries(&self, endpoint_url: Option<&str>, limit: u32) -> Vec<DeliveryRow> {
    let conn = self.conn.lock().expect("webhooks.db mutex poisoned");
    let mut stmt = match conn.prepare(&format!(
        "SELECT id, event_id, endpoint_url, attempt, outcome, status_code, error, attempted_at
        FROM deliveries
        {where_clause}
        ORDER BY attempted_at DESC, id DESC
        LIMIT ?{limit_param}",
        where_clause = match endpoint_url {
            Some(_) => "WHERE endpoint_url = ?1",
            None => "",
        },
        limit_param = if endpoint_url.is_some() { 2 } else { 1 },
    )) {
        Ok(stmt) => stmt,
        Err(error) => {
            tracing::warn!(
                error = %error,
                "webhooks ledger: recent_deliveries query failed"
            );
            return Vec::new();
        }
    };
    let rows = match endpoint_url {
        Some(url) => match stmt.query_map(params![url, limit as i64], row_to_delivery) {
            Ok(rows) => rows
                .collect::<Result<Vec<_>, _>>()
                .unwrap_or_else(|error| {
                    tracing::warn!(
                        error = %error,
                        "webhooks ledger: recent_deliveries row read failed"
                    );
                    Vec::new()
                }),
            Err(error) => {
                tracing::warn!(
                    error = %error,
                    "webhooks ledger: recent_deliveries query failed"
                );
                Vec::new()
            }
        },
        None => match stmt.query_map(params![limit as i64], row_to_delivery) {
            Ok(rows) => rows
                .collect::<Result<Vec<_>, _>>()
                .unwrap_or_else(|error| {
                    tracing::warn!(
                        error = %error,
                        "webhooks ledger: recent_deliveries row read failed"
                    );
                    Vec::new()
                }),
            Err(error) => {
                tracing::warn!(
                    error = %error,
                    "webhooks ledger: recent_deliveries query failed"
                );
                Vec::new()
            }
        },
    };
    rows
}

/// Batched failing-rollup read: for EACH of `endpoint_urls`, the
/// newest `per_endpoint_limit` attempts, in ONE SQL statement (a
/// `ROW_NUMBER() OVER (PARTITION BY endpoint_url ...)` window query
/// restricted to the requested URLs). Returns `(url, rows)` pairs
/// grouped per URL; a URL with no rows is omitted. Read path:
/// failures log a `warn` and degrade to an empty list.
pub fn recent_deliveries_per_endpoint(
    &self,
    endpoint_urls: &[String],
    per_endpoint_limit: u32,
) -> Vec<(String, Vec<DeliveryRow>)> {
    if endpoint_urls.is_empty() {
        return Vec::new();
    }
    let conn = self.conn.lock().expect("webhooks.db mutex poisoned");
    // Window function over per-endpoint attempt recency, newest
    // first. `?1` is the per-endpoint limit; `?2..` are the bound
    // endpoint URLs of the IN-list.
    let placeholders = endpoint_urls
        .iter()
        .enumerate()
        .map(|(index, _)| format!("?{}", index + 2))
        .collect::<Vec<_>>()
        .join(", ");
    let sql = format!(
        "SELECT endpoint_url, id, event_id, attempt, outcome, status_code, error, attempted_at
        FROM (
            SELECT id, event_id, endpoint_url, attempt, outcome, status_code, error, attempted_at,
                   ROW_NUMBER() OVER (
                       PARTITION BY endpoint_url
                       ORDER BY attempted_at DESC, id DESC
                   ) AS rn
            FROM deliveries
            WHERE endpoint_url IN ({placeholders})
        )
        WHERE rn <= ?1
        ORDER BY endpoint_url, attempted_at DESC, id DESC"
    );
    let Ok(mut stmt) = conn.prepare(&sql) else {
        tracing::warn!("webhooks ledger: recent_deliveries_per_endpoint query failed to prepare");
        return Vec::new();
    };
    let mut bound_params: Vec<&dyn rusqlite::ToSql> =
        vec![&per_endpoint_limit];
    for url in endpoint_urls {
        bound_params.push(url);
    }
    let rows_iter = match stmt.query_map(bound_params.as_slice(), |row| {
        Ok((
            row.get::<_, String>(0)?,
            DeliveryRow {
                id: row.get(1)?,
                event_id: row.get(2)?,
                endpoint_url: row.get(0)?,
                attempt: row.get(3)?,
                outcome: row.get(4)?,
                status_code: row.get(5)?,
                error: row.get(6)?,
                attempted_at: row.get(7)?,
            },
        ))
    }) {
        Ok(rows) => rows,
        Err(_) => {
            tracing::warn!("webhooks ledger: recent_deliveries_per_endpoint query failed");
            return Vec::new();
        }
    };
    let mut grouped: Vec<(String, Vec<DeliveryRow>)> = Vec::new();
    for row in rows_iter {
        let Ok((url, delivery)) = row else {
            tracing::warn!("webhooks ledger: recent_deliveries_per_endpoint row read failed");
            return Vec::new();
        };
        match grouped.last_mut() {
            Some((last_url, bucket)) if *last_url == url => bucket.push(delivery),
            _ => grouped.push((url, vec![delivery])),
        }
    }
    grouped
}

    /// `recent_events` served from a 1-second TTL cache, mirroring
    /// `cached_budget_store_rows` in the metrics module: on a cold or
    /// expired cache the SQLite read runs inside the same short lock;
    /// concurrent snapshot reads inside the TTL window clone the cached
    /// rows without touching the database. The cache is keyed by the
    /// requested limit (the dashboard always asks for the same one).
    /// `webhook.test` rows are excluded BEFORE the limit (design
    /// Component 5: the dashboard feed carries budget events only, so
    /// admin test sends can never crowd out real alert history).
    pub fn recent_events_cached(&self, limit: u32) -> Vec<AlertEventRow> {
        let mut cache = self
            .events_cache
            .lock()
            .expect("webhooks ledger events cache poisoned");
        let fresh = cache
            .as_ref()
            .map(|entry| entry.limit == limit && entry.at.elapsed() < RECENT_EVENTS_CACHE_TTL)
            .unwrap_or(false);
        if !fresh {
            let rows = self.recent_events_filtered(limit, /* exclude_tests: */ true);
            *cache = Some(RecentEventsCacheEntry {
                at: Instant::now(),
                limit,
                rows,
            });
        }
        cache
            .as_ref()
            .map(|entry| entry.rows.clone())
            .unwrap_or_default()
    }

    /// Test-only seam: age the recent-events cache past its TTL so the
    /// next `recent_events_cached` call refreshes without sleeping.
    /// (The TTL itself uses real time, exactly like the metrics
    /// budget-rows cache; that module has no injection point either.)
    #[cfg(test)]
    pub(crate) fn expire_recent_events_cache_for_test(&self) {
        let mut cache = self
            .events_cache
            .lock()
            .expect("webhooks ledger events cache poisoned");
        if let Some(entry) = cache.as_mut() {
            entry.at -= RECENT_EVENTS_CACHE_TTL + Duration::from_millis(10);
        }
    }
}

/// `insert_event` body: insert the event and run the retention pass in
/// one transaction (children first, respecting the `deliveries` FK).
/// Events of UNFINISHED chains (an `inflight` delivery row — the
/// chain is mid-retry) are EXCLUDED from the prune tail: deleting
/// them would make the chain's later attempt updates fail the
/// `event_id` foreign key. Only events with no non-terminal delivery
/// can fall out of the 500-newest window.
fn insert_event_with_retention(
conn: &mut Connection,
id: &str,
event_type: &str,
scope: &str,
scope_label: &str,
payload_json: &str,
created_at: i64,
) -> Result<(), rusqlite::Error> {
let tx = conn.transaction()?;
tx.execute(
"INSERT INTO alert_events (id, type, scope, scope_label, payload_json, created_at)
VALUES (?1, ?2, ?3, ?4, ?5, ?6)",
params![id, event_type, scope, scope_label, payload_json, created_at],
)?;
// Newest EVENT_RETENTION_LIMIT events stay (unfinished chains'
// events excepted); deliveries of the pruned tail go first so the
// FK never sees an orphan. The LEFT JOIN keeps events that have ANY
// inflight delivery row out of the pruned sub-select entirely.
tx.execute(
"DELETE FROM deliveries WHERE event_id IN (
SELECT pruned.id FROM (
SELECT id FROM alert_events
ORDER BY created_at DESC, rowid DESC
LIMIT -1 OFFSET ?1
) AS pruned
LEFT JOIN deliveries AS d
ON d.event_id = pruned.id AND d.outcome = 'inflight'
WHERE d.event_id IS NULL)",
params![EVENT_RETENTION_LIMIT],
)?;
tx.execute(
"DELETE FROM alert_events WHERE id IN (
SELECT pruned.id FROM (
SELECT id FROM alert_events
ORDER BY created_at DESC, rowid DESC
LIMIT -1 OFFSET ?1
) AS pruned
LEFT JOIN deliveries AS d
ON d.event_id = pruned.id AND d.outcome = 'inflight'
WHERE d.event_id IS NULL)",
params![EVENT_RETENTION_LIMIT],
)?;
tx.commit()?;
Ok(())
}

/// `park_unfinished_chains` body: find every (event, endpoint) chain
/// whose LATEST attempt row is still the non-terminal `inflight`
/// marker (written before network I/O, never finished — the crash
/// signature) and append one `parked` row (attempt = max + 1, capped
/// at the 4-attempt policy bound). All-`failed` chains are terminal
/// and excluded by construction.
fn park_unfinished_chains_locked(conn: &mut Connection) -> Result<usize, rusqlite::Error> {
let tx = conn.transaction()?;
let mut stmt = tx.prepare(
"SELECT event_id, endpoint_url, MAX(attempt) AS max_attempt
FROM deliveries
GROUP BY event_id, endpoint_url
HAVING MAX(attempt) = MAX(
CASE WHEN outcome = 'inflight' THEN attempt ELSE -1 END
)",
)?;
let chains = stmt
.query_map([], |row| {
Ok((
row.get::<_, String>(0)?,
row.get::<_, String>(1)?,
row.get::<_, i64>(2)?,
))
})?
.collect::<Result<Vec<_>, _>>()?;
drop(stmt);
let now = now_epoch();
let mut parked = 0usize;
for (event_id, endpoint_url, max_attempt) in chains {
let attempt = max_attempt.saturating_add(1).min(4);
tx.execute(
"INSERT INTO deliveries
(event_id, endpoint_url, attempt, outcome, status_code, error, attempted_at)
VALUES (?1, ?2, ?3, 'parked', NULL, NULL, ?4)",
params![event_id, endpoint_url, attempt, now],
)?;
parked += 1;
}
tx.commit()?;
Ok(parked)
}

fn row_to_alert_event(row: &Row) -> Result<AlertEventRow, rusqlite::Error> {
    Ok(AlertEventRow {
        id: row.get(0)?,
        event_type: row.get(1)?,
        scope: row.get(2)?,
        scope_label: row.get(3)?,
        payload_json: row.get(4)?,
        created_at: row.get(5)?,
    })
}

fn row_to_delivery(row: &Row) -> Result<DeliveryRow, rusqlite::Error> {
    Ok(DeliveryRow {
        id: row.get(0)?,
        event_id: row.get(1)?,
        endpoint_url: row.get(2)?,
        attempt: row.get(3)?,
        outcome: row.get(4)?,
        status_code: row.get(5)?,
        error: row.get(6)?,
        attempted_at: row.get(7)?,
    })
}

fn now_epoch() -> i64 {
    Utc::now().timestamp()
}

#[cfg(test)]
mod tests {
    use super::*;
    use tempfile::NamedTempFile;

    fn temp_store() -> (WebhookLedgerStore, NamedTempFile) {
        let temp = NamedTempFile::new().unwrap();
        let store = WebhookLedgerStore::new(temp.path()).unwrap();
        (store, temp)
    }

    #[test]
    fn init_is_idempotent() {
        let (store, temp) = temp_store();
        // `new` already ran init once; a second run must succeed and
        // leave the tables usable and intact.
        store.init().unwrap();
        store
            .insert_event("evt-init", "budget.warning", "provider", "openai", "{}", 1000)
            .unwrap();
        store.init().unwrap();
        assert_eq!(store.recent_events(10).len(), 1);
        // Double-init on the same path (second store, same file): the
        // migration is a no-op and existing rows survive.
        let second = WebhookLedgerStore::new(temp.path()).unwrap();
        assert_eq!(second.recent_events(10).len(), 1);
    }

    #[test]
    fn migration_adds_missing_columns_to_legacy_tables() {
        let temp = NamedTempFile::new().unwrap();
        {
            let conn = Connection::open(temp.path()).unwrap();
            conn.execute(
                "CREATE TABLE alert_events (
                    id TEXT PRIMARY KEY,
                    type TEXT NOT NULL,
                    scope TEXT NOT NULL,
                    created_at INTEGER NOT NULL
                )",
                [],
            )
            .unwrap();
            conn.execute(
                "CREATE TABLE deliveries (
                    id INTEGER PRIMARY KEY,
                    event_id TEXT NOT NULL REFERENCES alert_events(id),
                    endpoint_url TEXT NOT NULL,
                    attempt INTEGER NOT NULL,
                    outcome TEXT NOT NULL,
                    attempted_at INTEGER NOT NULL
                )",
                [],
            )
            .unwrap();
            conn.execute(
                "INSERT INTO alert_events (id, type, scope, created_at)
                VALUES ('legacy-1', 'budget.warning', 'provider', 42)",
                [],
            )
            .unwrap();
        }
        let store = WebhookLedgerStore::new(temp.path()).unwrap();
        let events = store.recent_events(10);
        assert_eq!(events.len(), 1);
        assert_eq!(events[0].id, "legacy-1");
        assert_eq!(events[0].scope_label, "");
        assert_eq!(events[0].payload_json, "{}");
        // The evolved tables remain writable.
        store
            .record_attempt(
                "legacy-1",
                "https://hooks.example.com/a",
                1,
                "success",
                Some(200),
                None,
                43,
            )
            .unwrap();
        assert_eq!(
            store
                .recent_deliveries(Some("https://hooks.example.com/a"), 10)
                .len(),
            1
        );
    }

    #[test]
    fn insert_event_and_reads_round_trip() {
        let (store, _tmp) = temp_store();
        store
            .insert_event("evt-1", "budget.warning", "provider", "openai", r#"{"a":1}"#, 1000)
            .unwrap();
        store
            .insert_event(
                "evt-2",
                "budget.exhausted",
                "virtual_key",
                "key:7:team",
                r#"{"b":2}"#,
                2000,
            )
            .unwrap();

        let events = store.recent_events(10);
        assert_eq!(events.len(), 2);
        // Newest first.
        assert_eq!(events[0].id, "evt-2");
        assert_eq!(events[0].event_type, "budget.exhausted");
        assert_eq!(events[0].scope, "virtual_key");
        assert_eq!(events[0].scope_label, "key:7:team");
        assert_eq!(events[0].payload_json, r#"{"b":2}"#);
        assert_eq!(events[0].created_at, 2000);
        assert_eq!(events[1].id, "evt-1");

        let limited = store.recent_events(1);
        assert_eq!(limited.len(), 1);
        assert_eq!(limited[0].id, "evt-2");

        store
            .record_attempt(
                "evt-1",
                "https://hooks.example.com/a",
                1,
                "failed",
                Some(500),
                Some("http_500"),
                3000,
            )
            .unwrap();
        store
            .record_attempt(
                "evt-1",
                "https://hooks.example.com/a",
                2,
                "success",
                Some(200),
                None,
                4000,
            )
            .unwrap();
        store
            .record_attempt(
                "evt-2",
                "https://hooks.example.com/b",
                1,
                "failed",
                None,
                Some("connect_timeout"),
                5000,
            )
            .unwrap();

        let deliveries = store.recent_deliveries(Some("https://hooks.example.com/a"), 10);
        assert_eq!(deliveries.len(), 2);
        assert_eq!(deliveries[0].attempt, 2);
        assert_eq!(deliveries[0].outcome, "success");
        assert_eq!(deliveries[0].status_code, Some(200));
        assert_eq!(deliveries[0].error, None);
        assert_eq!(deliveries[0].event_id, "evt-1");
        assert_eq!(deliveries[1].attempt, 1);
        assert_eq!(deliveries[1].status_code, Some(500));
        assert_eq!(deliveries[1].error, Some("http_500".to_string()));

        let other = store.recent_deliveries(Some("https://hooks.example.com/b"), 10);
        assert_eq!(other.len(), 1);
        assert_eq!(other[0].error, Some("connect_timeout".to_string()));

        // Undelivered events: one skipped row with an empty endpoint URL.
        store.record_undelivered("evt-2").unwrap();
        let skipped = store.recent_deliveries(Some(""), 10);
        assert_eq!(skipped.len(), 1);
        assert_eq!(skipped[0].event_id, "evt-2");
        assert_eq!(skipped[0].outcome, "skipped_no_endpoints");
        assert_eq!(skipped[0].attempt, 1);
        assert_eq!(skipped[0].status_code, None);
        assert_eq!(skipped[0].error, None);
    }

    #[test]
    fn retention_prunes_to_newest_500_events_and_their_deliveries() {
        let (store, _tmp) = temp_store();
        for i in 0..500 {
            store
                .insert_event(
                    &format!("evt-{i}"),
                    "budget.warning",
                    "provider",
                    "openai",
                    "{}",
                    1000 + i,
                )
                .unwrap();
        }
        // A delivery on the soon-to-be-pruned oldest event must
        // disappear with it (children first, FK respected).
        store
            .record_attempt(
                "evt-0",
                "https://hooks.example.com/a",
                1,
                "failed",
                Some(500),
                None,
                999_999,
            )
            .unwrap();
        for i in 500..505 {
            store
                .insert_event(
                    &format!("evt-{i}"),
                    "budget.warning",
                    "provider",
                    "openai",
                    "{}",
                    1000 + i,
                )
                .unwrap();
        }

        let events = store.recent_events(1000);
        assert_eq!(events.len(), 500);
        let ids: std::collections::HashSet<&str> =
            events.iter().map(|event| event.id.as_str()).collect();
        for pruned in ["evt-0", "evt-1", "evt-2", "evt-3", "evt-4"] {
            assert!(!ids.contains(pruned), "{pruned} should have been pruned");
        }
        assert!(ids.contains("evt-5"));
        assert!(ids.contains("evt-504"));
        assert_eq!(events[0].id, "evt-504");
        assert_eq!(events[499].id, "evt-5");
        assert!(store
            .recent_deliveries(Some("https://hooks.example.com/a"), 100)
            .is_empty());
    }

#[test]
fn park_unfinished_chains_marks_only_inflight_chains() {
let (store, _tmp) = temp_store();
let url = "https://hooks.example.com/a";
// Interrupted chain: attempt 2 was marked inflight (crash between
// the pre-I/O write and the outcome write), attempt 1 finished
// failed (the chain was retrying when the process died).
store
.insert_event("evt-interrupted", "budget.warning", "provider", "openai", "{}", 1000)
.unwrap();
store
.record_attempt("evt-interrupted", url, 1, "failed", Some(500), None, 1100)
.unwrap();
store
.record_inflight_attempt("evt-interrupted", url, 2, 1200)
.unwrap();
// Crash BEFORE the first attempt completed: the only row is the
// inflight marker written before network I/O — previously invisible
// (no completed row, no chain).
store
.insert_event("evt-crash-first", "budget.warning", "provider", "openai", "{}", 1300)
.unwrap();
store
.record_inflight_attempt("evt-crash-first", url, 1, 1400)
.unwrap();
// All-failed chain (all attempts finished): TERMINAL — a
// single-shot test send or a chain whose attempts all completed
// `failed` must NOT get a bogus parked row appended.
store
.insert_event("evt-all-failed", "budget.exhausted", "provider", "openai", "{}", 1500)
.unwrap();
for attempt in 1..=4 {
store
.record_attempt("evt-all-failed", url, attempt, "failed", None, Some("timeout"), 1500 + attempt)
.unwrap();
}
// First-attempt SSRF rejection: terminal `failed` (not parked).
store
.insert_event("evt-ssrf", "budget.warning", "provider", "openai", "{}", 1600)
.unwrap();
store
.record_attempt("evt-ssrf", url, 1, "failed", None, Some("ssrf: loopback"), 1700)
.unwrap();
// Finished chain: success.
store
.insert_event("evt-success", "budget.warning", "provider", "openai", "{}", 2000)
.unwrap();
store
.record_attempt("evt-success", url, 1, "success", Some(200), None, 2100)
.unwrap();
// Finished by skip.
store
.insert_event("evt-skipped", "budget.warning", "provider", "openai", "{}", 3000)
.unwrap();
store.record_undelivered("evt-skipped").unwrap();
// No deliveries at all: not a chain, nothing to park.
store
.insert_event("evt-bare", "budget.warning", "provider", "openai", "{}", 4000)
.unwrap();

let parked = store.park_unfinished_chains().unwrap();
assert_eq!(parked, 2);

let deliveries = store.recent_deliveries(Some(url), 100);
let parked_rows: Vec<&DeliveryRow> = deliveries
.iter()
.filter(|delivery| delivery.outcome == "parked")
.collect();
assert_eq!(parked_rows.len(), 2);
let interrupted: Vec<_> = parked_rows
.iter()
.filter(|row| row.event_id == "evt-interrupted")
.collect();
assert_eq!(interrupted.len(), 1);
assert_eq!(interrupted[0].attempt, 3); // max attempt + 1
assert_eq!(interrupted[0].status_code, None);
assert_eq!(interrupted[0].error, None);
let crash_first: Vec<_> = parked_rows
.iter()
.filter(|row| row.event_id == "evt-crash-first")
.collect();
assert_eq!(crash_first.len(), 1);
assert_eq!(crash_first[0].attempt, 2);

// All-failed chains are terminal: no parked row, attempts intact.
let all_failed: Vec<_> = deliveries
.iter()
.filter(|row| row.event_id == "evt-all-failed")
.collect();
assert_eq!(all_failed.len(), 4, "all-failed chain untouched");
// SSRF first-attempt rejection is terminal: single failed row.
let ssrf: Vec<_> = deliveries
.iter()
.filter(|row| row.event_id == "evt-ssrf")
.collect();
assert_eq!(ssrf.len(), 1, "ssrf rejection is terminal, not parked");

// Finished chains are untouched.
assert_eq!(store.recent_deliveries(Some(""), 100).len(), 1);
let success_rows: Vec<_> = deliveries
.iter()
.filter(|row| row.event_id == "evt-success")
.collect();
assert_eq!(success_rows.len(), 1);

// Idempotent: a second pass finds no inflight chains.
assert_eq!(store.park_unfinished_chains().unwrap(), 0);
}

#[test]
fn finish_inflight_attempt_overwrites_marker_and_ignores_foreign_rows() {
let (store, _tmp) = temp_store();
let url = "https://hooks.example.com/a";
store
.insert_event("evt-flight", "budget.warning", "provider", "openai", "{}", 1000)
.unwrap();
store.record_inflight_attempt("evt-flight", url, 1, 1100).unwrap();
// Finishing writes the terminal outcome over the marker row.
store
.finish_inflight_attempt("evt-flight", url, 1, "success", Some(200), None, 1200)
.unwrap();
let rows = store.recent_deliveries(Some(url), 10);
assert_eq!(rows.len(), 1);
assert_eq!(rows[0].outcome, "success");
assert_eq!(rows[0].status_code, Some(200));
assert_eq!(rows[0].attempted_at, 1200);
// A finish for a row that is not inflight (already terminal) is a
// no-op — the WHERE clause keeps it from overwriting history.
store
.finish_inflight_attempt("evt-flight", url, 1, "failed", None, Some("late"), 1300)
.unwrap();
let rows = store.recent_deliveries(Some(url), 10);
assert_eq!(rows.len(), 1);
assert_eq!(rows[0].outcome, "success");
}

#[test]
fn retention_keeps_events_of_inflight_chains() {
let (store, _tmp) = temp_store();
let url = "https://hooks.example.com/a";
// 500 events fill the retention window to its boundary; the
// inflight-marked event is the OLDEST (the pruned tail).
store
.insert_event("evt-inflight-old", "budget.warning", "provider", "openai", "{}", 1)
.unwrap();
store.record_inflight_attempt("evt-inflight-old", url, 1, 2).unwrap();
for n in 1..=EVENT_RETENTION_LIMIT {
store
.insert_event(&format!("evt-{n:03}"), "budget.warning", "provider", "openai", "{}", 1000 + n as i64)
.unwrap();
}
// The newest 500 (incl. evt-000..) survive plus the inflight one.
let events = store.recent_events(1000);
assert_eq!(events.len(), usize::try_from(EVENT_RETENTION_LIMIT + 1).unwrap());
assert!(events.iter().any(|row| row.id == "evt-inflight-old"));
// A later finish for the retained inflight row still succeeds —
// the FK target survived pruning — and the SECOND pruning pass (a
// fresh batch of newer events) then sees the chain terminal and
// prunes it with the rest of the tail.
store
.finish_inflight_attempt("evt-inflight-old", url, 1, "success", Some(200), None, 10_000)
.unwrap();
// A terminal sibling event outside the window is pruned with its
// deliveries.
store
.insert_event("evt-terminal-old", "budget.warning", "provider", "openai", "{}", 500_000)
.unwrap();
store
.record_attempt("evt-terminal-old", url, 1, "failed", None, Some("timeout"), 500_001)
.unwrap();
// Insert enough NEWER events to push evt-terminal-old out of the
// 500-newest window.
for n in 1..=EVENT_RETENTION_LIMIT {
store
.insert_event(&format!("evt-new-{n:03}"), "budget.warning", "provider", "openai", "{}", 1_000_000 + n as i64)
.unwrap();
}
let events = store.recent_events(2000);
assert!(events.iter().all(|row| row.id != "evt-terminal-old"));
assert!(
events.iter().all(|row| row.id != "evt-inflight-old"),
"finished chain left the retention exception and pruned normally"
);
}

    #[test]
    fn recent_events_cached_serves_rows_within_ttl_and_refreshes_after() {
        let (store, _tmp) = temp_store();
        store
            .insert_event("evt-1", "budget.warning", "provider", "openai", "{}", 1000)
            .unwrap();

        let first = store.recent_events_cached(20);
        assert_eq!(first.len(), 1);
        assert_eq!(first[0].id, "evt-1");

        // Within the TTL the cache serves the snapshot it holds: a new
        // event stays invisible until the cache expires.
        store
            .insert_event("evt-2", "budget.exhausted", "provider", "openai", "{}", 2000)
            .unwrap();
        let stale = store.recent_events_cached(20);
        assert_eq!(stale.len(), 1);
        assert_eq!(stale[0].id, "evt-1");

        // Past the TTL (aged via the test seam; real time otherwise,
        // like the metrics budget-rows cache) the next call refreshes.
        store.expire_recent_events_cache_for_test();
        let fresh = store.recent_events_cached(20);
        assert_eq!(fresh.len(), 2);
        assert_eq!(fresh[0].id, "evt-2");

        // A different limit is a different cache entry, not a stale hit.
        store
            .insert_event("evt-3", "budget.reset", "provider", "openai", "{}", 3000)
            .unwrap();
        let other_limit = store.recent_events_cached(5);
        assert_eq!(other_limit.len(), 3);
    }
}
