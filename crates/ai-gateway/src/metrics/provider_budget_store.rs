//! SQLite-backed persistence for per-provider budget windows (`budgets.db`).
//!
//! This module owns the `provider_budgets` table, the durable owner of
//! per-provider budget-window spend (the metrics atomics act as a write-through
//! cache of this store). The connection is wrapped in an
//! `Arc<Mutex<Connection>>` to match the pattern used by the virtual-key
//! store, and — like that store — contains no Tokio APIs because it is
//! called from plain `std::thread`-spawned tasks.
//!
//! Timestamps are stored as UTC epoch seconds (`INTEGER`).

use std::path::Path;
use std::sync::{Arc, Mutex};
use std::time::Duration;

use chrono::Utc;
use rusqlite::types::Type as SqlType;
use rusqlite::{params, Connection, Row};

use crate::config::BudgetResetPolicy;

/// Errors produced by the [`ProviderBudgetStore`].
#[derive(Debug, thiserror::Error)]
pub enum ProviderBudgetStoreError {
    #[error("Database error: {0}")]
    Database(#[from] rusqlite::Error),

    #[error("IO error: {0}")]
    Io(#[from] std::io::Error),

    #[error("Invalid stored data: {0}")]
    InvalidData(String),
}

/// A full row of the `provider_budgets` table.
#[derive(Debug, Clone, PartialEq)]
pub struct ProviderBudgetRow {
    /// Provider name (primary key).
    pub provider: String,
    /// Configured budget limit in USD; `NULL` = spend-only row (no limit
    /// configured; config is the authority for limits).
    pub limit_usd: Option<f64>,
    /// Reset policy recorded at insert time (always the default `manual`;
    /// config-seeded policy is the authoritative source — this column is
    /// row metadata).
    pub window: BudgetResetPolicy,
    /// UTC epoch seconds the current window opened at (insert-time anchor
    /// or the last `reset_window` instant).
    pub window_start: i64,
    /// Spend accrued in the current window (USD).
    pub current_spend_usd: f64,
    /// UTC epoch seconds of the last write to this row.
    pub updated_at: i64,
}

/// SQLite-backed per-provider budget store. Cheaply cloneable in practice via
/// the shared connection handle (mirror of [`crate::virtual_keys::KeyStore`]).
pub struct ProviderBudgetStore {
    conn: Arc<Mutex<Connection>>,
}

// Manual impl: `Metrics` derives `Debug` and holds this store; the connection
// itself has no meaningful debug representation.
impl std::fmt::Debug for ProviderBudgetStore {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("ProviderBudgetStore")
            .finish_non_exhaustive()
    }
}

/// Result of a [`ProviderBudgetStore::reset_window`] call: which
/// reset won and the facts the callers need to act on it exactly once.
///
/// `did_reset` is true only for the call whose CAS hit — the single
/// winner that performed the zeroing. `previous_spend` is the spend
/// that was zeroed (winner) or the fresh post-rollover spend the
/// re-read observed (CAS-miss loser). `new_anchor` is the window
/// anchor the reset stamped (winner) or the anchor the concurrent
/// winning reset already stamped (loser) — callers emit follow-up
/// events (budget.reset) with THIS anchor, never with a post-hoc
/// read-back that may already belong to an even later reset.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct ResetWindowOutcome {
/// True when THIS call performed the reset (CAS hit).
pub did_reset: bool,
/// Spend before the reset (winner) / current spend after the
/// other writer's reset (loser).
pub previous_spend: f64,
/// The window anchor now in force (the one this reset stamped, or
/// the concurrent reset's stamp).
pub new_anchor: i64,
}

impl ProviderBudgetStore {
    /// Open (or create) the database at `db_path`, run the idempotent schema
    /// migration, and apply the same WAL/busy-timeout discipline as the
    /// virtual-key store so bursts of off-path spend writes cannot stall the
    /// process on fsyncs.
    pub fn new(db_path: &Path) -> Result<Self, ProviderBudgetStoreError> {
        if let Some(parent) = db_path.parent() {
            if !parent.as_os_str().is_empty() {
                std::fs::create_dir_all(parent)?;
            }
        }

        let conn = Connection::open(db_path)?;
        // See `KeyStore::new`: WAL + synchronous=NORMAL removes the per-commit
        // fsync and `busy_timeout` makes lock contention wait rather than fail.
        conn.busy_timeout(Duration::from_secs(5))?;
        conn.pragma_update(None, "journal_mode", "WAL")?;
        conn.pragma_update(None, "synchronous", "NORMAL")?;
        let store = Self {
            conn: Arc::new(Mutex::new(conn)),
        };
        store.init()?;
        Ok(store)
    }

    /// Idempotent schema creation and migration, run inside a single short
    /// transaction so a crash mid-migration never leaves a half-evolved
    /// table. Safe to call repeatedly (e.g. again after `new`).
    pub fn init(&self) -> Result<(), ProviderBudgetStoreError> {
        let mut conn = self.conn.lock().expect("budgets.db mutex poisoned");
        let tx = conn.transaction()?;
        tx.execute_batch(
            "CREATE TABLE IF NOT EXISTS provider_budgets (
                provider TEXT PRIMARY KEY,
                limit_usd REAL,
                window TEXT CHECK(window IN ('daily','weekly','monthly','manual')),
                window_start INTEGER,
                current_spend_usd REAL NOT NULL DEFAULT 0.0,
                updated_at INTEGER
            );",
        )?;
        // ensure_column-style evolution, mirroring `KeyStore::migrate_schema`:
        // no-ops when the columns already exist, adds them for any earlier
        // partial schema.
        Self::ensure_column(&tx, "provider_budgets", "window", "TEXT")?;
        Self::ensure_column(&tx, "provider_budgets", "window_start", "INTEGER")?;
        tx.commit()?;
        Ok(())
    }

    /// Add `column` (with SQL declaration `decl`) to `table` unless it already
    /// exists.
    fn ensure_column(
        conn: &Connection,
        table: &str,
        column: &str,
        decl: &str,
    ) -> Result<(), ProviderBudgetStoreError> {
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

    /// Accrue `usd` of spend for `provider` in the window that opened at
    /// `window_start_epoch`, as a single conditional upsert (lost-update
    /// guard). The row is created on first spend with the default (manual)
    /// window and no limit; config seeding happens at the server layer.
    /// Called off the response path by the router's spend-accrual wiring
    /// (persistent budgets, Task 4).
    ///
    /// Because accrual writes are fire-and-forget (spawned off the
    /// response path), a write can race a concurrent `reset_window` and
    /// otherwise land in the NEW window, re-adding pre-rollover spend.
    /// The upsert is therefore window-conditional:
    ///
    /// - fresh row (`INSERT` branch): the row is created anchored at
    ///   `window_start_epoch` (first accrual of a window);
    /// - existing row (`DO UPDATE` branch): the spend is added only when
    ///   the row's `window_start` still equals `window_start_epoch` —
    ///   a write belonging to an older window is silently DROPPED, never
    ///   an error.
    ///
    /// Non-finite (NaN/±inf) or negative `usd` values are skipped as
    /// invalid input (quiet `Ok`; the in-memory atomics never see them
    /// either because cost attribution clamps upstream).
    pub fn add_spend(
        &self,
        provider: &str,
        usd: f64,
        window_start_epoch: i64,
    ) -> Result<(), ProviderBudgetStoreError> {
        if !usd.is_finite() || usd < 0.0 {
            return Ok(());
        }
        let conn = self.conn.lock().expect("budgets.db mutex poisoned");
        let now = now_epoch();
        conn.execute(
            "INSERT INTO provider_budgets
                (provider, limit_usd, window, window_start, current_spend_usd, updated_at)
             VALUES (?1, NULL, ?5, ?2, ?3, ?4)
             ON CONFLICT(provider) DO UPDATE SET
                current_spend_usd = current_spend_usd + ?3,
                updated_at = ?4
             WHERE window_start = ?2",
            params![
                provider,
                window_start_epoch,
                usd,
                now,
                window_to_str(BudgetResetPolicy::default())
            ],
        )?;
        Ok(())
    }

    /// All budget rows, ordered by provider name.
    pub fn get_all(&self) -> Result<Vec<ProviderBudgetRow>, ProviderBudgetStoreError> {
        let conn = self.conn.lock().expect("budgets.db mutex poisoned");
        let mut stmt = conn.prepare(
            "SELECT provider, limit_usd, window, window_start, current_spend_usd, updated_at
             FROM provider_budgets
             ORDER BY provider",
        )?;
        let rows = stmt.query_map([], row_to_budget_row)?;
        Ok(rows.collect::<Result<Vec<_>, _>>()?)
    }

    /// The durable `(current_spend_usd, window_start)` for `provider`,
    /// or `None` when the provider has no row. Read by the post-persist
    /// alert hook (budget-alerts-unified-webhooks, Task 5) inside the
    /// off-path spend-write task, so the evaluation sees the exact
    /// durable total the write produced (including when the conditional
    /// upsert dropped the write after a concurrent window reset).
    pub fn get_spend_and_window(
        &self,
        provider: &str,
    ) -> Result<Option<(f64, i64)>, ProviderBudgetStoreError> {
        let conn = self.conn.lock().expect("budgets.db mutex poisoned");
        let row = conn.query_row(
            "SELECT current_spend_usd, window_start FROM provider_budgets WHERE provider = ?1",
            params![provider],
            |row| Ok((row.get::<_, f64>(0)?, row.get::<_, i64>(1)?)),
        );
        match row {
            Ok(value) => Ok(Some(value)),
            Err(rusqlite::Error::QueryReturnedNoRows) => Ok(None),
            Err(e) => Err(e.into()),
        }
    }

    /// The durable `window_start` (UTC epoch seconds) for `provider`, or
    /// `None` when the provider has no row. Read by the router's lazy
    /// window-rollover check (persistent budgets, Task 4): the store owns
    /// the window boundary, including across restarts.
    pub fn get_window_start(
        &self,
        provider: &str,
    ) -> Result<Option<i64>, ProviderBudgetStoreError> {
        let conn = self.conn.lock().expect("budgets.db mutex poisoned");
        let start = conn.query_row(
            "SELECT window_start FROM provider_budgets WHERE provider = ?1",
            params![provider],
            |row| row.get::<_, Option<i64>>(0),
        );
        match start {
            Ok(value) => Ok(value),
            Err(rusqlite::Error::QueryReturnedNoRows) => Ok(None),
            Err(e) => Err(e.into()),
        }
    }

/// Reset one provider's budget window via compare-and-swap on the
/// durable `window_start`, zeroing the spend only when the row is
/// still anchored at `expected_window_start_epoch`. On a CAS hit,
/// [`ResetWindowOutcome::did_reset`] is true and `new_anchor` is the
/// instant this reset stamped; on a CAS miss (another writer already
/// advanced the window) the row is re-read once and the outcome
/// reports the concurrent reset's fresh spend and anchor — the window
/// is treated as already reset, and only the CAS winner should emit
/// the follow-up reset event. `previous_spend` is `None` when the
/// provider has no row.
///
/// Consumed by the router's lazy window-rollover check and the admin
/// manual reset (persistent-budgets Task 4/8). The single
/// `Arc<Mutex<Connection>>` serializes store accesses in-process; the
/// CAS clause additionally makes the UPDATE safe against any external
/// writer (WAL allows a second connection, e.g. tests) and keeps the
/// reset idempotent under concurrent callers.
pub fn reset_window(
&self,
provider: &str,
expected_window_start_epoch: i64,
) -> Result<Option<ResetWindowOutcome>, ProviderBudgetStoreError> {
let mut conn = self.conn.lock().expect("budgets.db mutex poisoned");
let tx = conn.transaction()?;
let current = tx.query_row(
"SELECT current_spend_usd FROM provider_budgets WHERE provider = ?1",
params![provider],
|row| row.get::<_, f64>(0),
);
let previous = match current {
Ok(spend) => spend,
Err(rusqlite::Error::QueryReturnedNoRows) => {
tx.commit()?;
return Ok(None);
}
Err(e) => return Err(e.into()),
};
let now = now_epoch();
let updated = tx.execute(
"UPDATE provider_budgets
SET current_spend_usd = 0.0, window_start = ?3, updated_at = ?3
WHERE provider = ?1 AND window_start = ?2",
params![provider, expected_window_start_epoch, now],
)?;
if updated > 0 {
// CAS hit: this call performed the reset; report the spend
// that was just zeroed and the anchor this reset stamped.
tx.commit()?;
Ok(Some(ResetWindowOutcome {
did_reset: true,
previous_spend: previous,
new_anchor: now,
}))
} else {
// CAS miss: another writer already advanced the window. The
// UPDATE changed nothing; commit the no-op transaction and
// report the fresh row state (the other reset's anchor and
// the ~0 spend it left behind) so the loser can stay silent.
tx.commit()?;
let fresh = conn.query_row(
"SELECT current_spend_usd, window_start FROM provider_budgets WHERE provider = ?1",
params![provider],
|row| Ok((row.get::<_, f64>(0)?, row.get::<_, i64>(1)?)),
);
match fresh {
Ok((spend, anchor)) => Ok(Some(ResetWindowOutcome {
did_reset: false,
previous_spend: spend,
new_anchor: anchor,
})),
Err(rusqlite::Error::QueryReturnedNoRows) => Ok(None),
Err(e) => Err(e.into()),
}
}
}
}

fn row_to_budget_row(row: &Row) -> Result<ProviderBudgetRow, rusqlite::Error> {
    let window_str: Option<String> = row.get(2)?;
    let window = window_str
        .as_deref()
        .map(window_from_str)
        .transpose()
        .map_err(|e| rusqlite::Error::FromSqlConversionFailure(2, SqlType::Text, Box::new(e)))?
        .unwrap_or(BudgetResetPolicy::Manual);
    Ok(ProviderBudgetRow {
        provider: row.get(0)?,
        limit_usd: row.get(1)?,
        window,
        window_start: row.get::<_, Option<i64>>(3)?.unwrap_or(0),
        current_spend_usd: row.get(4)?,
        updated_at: row.get::<_, Option<i64>>(5)?.unwrap_or(0),
    })
}

fn window_to_str(window: BudgetResetPolicy) -> &'static str {
    match window {
        BudgetResetPolicy::Daily => "daily",
        BudgetResetPolicy::Weekly => "weekly",
        BudgetResetPolicy::Monthly => "monthly",
        BudgetResetPolicy::Manual => "manual",
    }
}

fn window_from_str(s: &str) -> Result<BudgetResetPolicy, ProviderBudgetStoreError> {
    match s {
        "daily" => Ok(BudgetResetPolicy::Daily),
        "weekly" => Ok(BudgetResetPolicy::Weekly),
        "monthly" => Ok(BudgetResetPolicy::Monthly),
        "manual" => Ok(BudgetResetPolicy::Manual),
        other => Err(ProviderBudgetStoreError::InvalidData(format!(
            "unknown budget reset policy: {other}"
        ))),
    }
}

fn now_epoch() -> i64 {
    Utc::now().timestamp()
}

#[cfg(test)]
mod tests {
    use super::*;
    use tempfile::NamedTempFile;

    fn temp_store() -> (ProviderBudgetStore, NamedTempFile) {
        let temp = NamedTempFile::new().unwrap();
        let store = ProviderBudgetStore::new(temp.path()).unwrap();
        (store, temp)
    }

    #[test]
    fn init_is_idempotent() {
        let (store, _tmp) = temp_store();
        // `new` already ran init once; a second run must succeed and leave a
        // usable table.
        store.init().unwrap();
        store.add_spend("prov-init", 1.0, now_epoch()).unwrap();
        let rows = store.get_all().unwrap();
        assert_eq!(rows.len(), 1);
        assert_eq!(rows[0].current_spend_usd, 1.0);
    }

    #[test]
    fn migration_adds_missing_columns_to_legacy_table() {
        let temp = NamedTempFile::new().unwrap();
        {
            let conn = Connection::open(temp.path()).unwrap();
            conn.execute(
                "CREATE TABLE provider_budgets (
                    provider TEXT PRIMARY KEY,
                    limit_usd REAL,
                    current_spend_usd REAL NOT NULL DEFAULT 0.0,
                    updated_at INTEGER
                )",
                [],
            )
            .unwrap();
            conn.execute(
                "INSERT INTO provider_budgets (provider, current_spend_usd)
                 VALUES ('legacy', 5.0)",
                [],
            )
            .unwrap();
        }
        let store = ProviderBudgetStore::new(temp.path()).unwrap();
        let rows = store.get_all().unwrap();
        assert_eq!(rows.len(), 1);
        assert_eq!(rows[0].provider, "legacy");
        assert_eq!(rows[0].window, BudgetResetPolicy::Manual);
        assert_eq!(rows[0].current_spend_usd, 5.0);
    }

    #[test]
    fn add_spend_accrues_and_upserts() {
        let (store, _tmp) = temp_store();
        let anchor = now_epoch();
        store.add_spend("openai", 0.25, anchor).unwrap();
        store.add_spend("openai", 0.75, anchor).unwrap();
        store.add_spend("anthropic", 2.0, anchor).unwrap();

        let rows = store.get_all().unwrap();
        assert_eq!(rows.len(), 2);

        let openai = rows
            .iter()
            .find(|row| row.provider == "openai")
            .expect("openai row");
        assert_eq!(openai.current_spend_usd, 1.0);
        assert_eq!(openai.window, BudgetResetPolicy::Manual);
        assert_eq!(openai.limit_usd, None);
        assert!(openai.window_start > 0);
        assert!(openai.updated_at >= openai.window_start);

        let anthropic = rows
            .iter()
            .find(|row| row.provider == "anthropic")
            .expect("anthropic row");
        assert_eq!(anthropic.current_spend_usd, 2.0);
    }

    #[test]
    fn add_spend_drops_stale_window_writes() {
        let (store, _tmp) = temp_store();
        // Backdate the anchor so the later reset_window (which stamps
        // "now") can never collide with it at second granularity.
        let anchor = now_epoch() - 100;
        store.add_spend("openai", 1.0, anchor).unwrap();
        // A concurrent reset advanced the window...
        store.reset_window("openai", anchor).unwrap();
        // ...so this pre-rollover write (stale anchor) must be DROPPED,
        // not applied to the fresh window.
        store.add_spend("openai", 2.0, anchor).unwrap();
        let rows = store.get_all().unwrap();
        assert_eq!(rows[0].current_spend_usd, 0.0, "stale-window write dropped");
        // A write carrying the NEW anchor accrues normally.
        let new_anchor = rows[0].window_start;
        store.add_spend("openai", 2.0, new_anchor).unwrap();
        assert_eq!(store.get_all().unwrap()[0].current_spend_usd, 2.0);
    }

    #[test]
    fn add_spend_skips_non_finite_and_negative() {
        let (store, _tmp) = temp_store();
        let anchor = now_epoch();
        store.add_spend("openai", f64::NAN, anchor).unwrap();
        store.add_spend("openai", f64::INFINITY, anchor).unwrap();
        store.add_spend("openai", -1.0, anchor).unwrap();
        // Zero is a valid no-op accrual (row creation only).
        store.add_spend("openai", 0.0, anchor).unwrap();
        let rows = store.get_all().unwrap();
        assert_eq!(rows.len(), 1, "invalid inputs must not create rows");
        assert_eq!(rows[0].current_spend_usd, 0.0);
    }

#[test]
fn reset_window_returns_previous_spend_and_zeroes() {
let (store, _tmp) = temp_store();
let anchor = now_epoch();
store.add_spend("openai", 3.5, anchor).unwrap();
let stored_anchor = store.get_window_start("openai").unwrap().unwrap();

let outcome = store.reset_window("openai", stored_anchor).unwrap().unwrap();
assert!(outcome.did_reset);
assert_eq!(outcome.previous_spend, 3.5);
// The reset stamps "now" as the new anchor — same-second resets
// legitimately tie, so the assertion is monotonic (>=), not strict.
assert!(outcome.new_anchor >= stored_anchor);

let rows = store.get_all().unwrap();
assert_eq!(rows.len(), 1);
assert_eq!(rows[0].current_spend_usd, 0.0);
assert_eq!(rows[0].window_start, outcome.new_anchor);

// CAS miss is only observable when the winner stamped a DIFFERENT
// anchor; same-second ties make the stale CAS hit again. Pin the
// row's anchor forward (simulating a later winning reset) and re-CAS
// with the original: did_reset is false and the outcome reports the
// concurrent winner's anchor and the ~0 spend it left.
{
let conn = store.conn.lock().unwrap();
conn.execute(
"UPDATE provider_budgets SET window_start = ?1 WHERE provider = 'openai'",
params![stored_anchor + 12345],
)
.unwrap();
}
let missed = store.reset_window("openai", stored_anchor).unwrap().unwrap();
assert!(!missed.did_reset);
assert_eq!(missed.previous_spend, 0.0);
assert_eq!(missed.new_anchor, stored_anchor + 12345);
// A second CAS with the CURRENT anchor hits and reports the
// (now zero) spend.
let new_anchor = store.get_window_start("openai").unwrap().unwrap();
let again = store.reset_window("openai", new_anchor).unwrap().unwrap();
assert!(again.did_reset);
assert_eq!(again.previous_spend, 0.0);
}

#[test]
fn reset_window_unknown_provider_returns_none() {
let (store, _tmp) = temp_store();
assert_eq!(store.reset_window("missing", now_epoch()).unwrap(), None);
}

    #[test]
    fn get_all_returns_rows_for_metrics_seeding() {
        let (store, _tmp) = temp_store();
        let anchor = now_epoch();
        store.add_spend("openai", 1.25, anchor).unwrap();
        store.add_spend("anthropic", 0.5, anchor).unwrap();

        let rows = store.get_all().unwrap();
        assert_eq!(rows.len(), 2);
        // Ordered by provider name.
        assert_eq!(rows[0].provider, "anthropic");
        assert_eq!(rows[0].current_spend_usd, 0.5);
        assert_eq!(rows[1].provider, "openai");
        assert_eq!(rows[1].current_spend_usd, 1.25);
        assert_eq!(rows[1].window, BudgetResetPolicy::Manual);
    }
}
