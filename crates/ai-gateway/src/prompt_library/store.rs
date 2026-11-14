//! SQLite-backed persistence for the versioned prompt library.
//!
//! The store is dormant until the first prompt is created: `open` only brings
//! the schema up to date inside one transaction guarded by the
//! `user_version` pragma. Migrations are additive (`CREATE ... IF NOT
//! EXISTS`, `ALTER TABLE ... ADD COLUMN`) so a pre-existing older file never
//! loses data. Timestamps are stored as UTC epoch seconds (`INTEGER`),
//! matching the assistants and memory stores.

use std::collections::BTreeMap;
use std::path::Path;
use std::sync::{Arc, Mutex, MutexGuard};
use std::time::Duration;

use chrono::Utc;
use rusqlite::{params, Connection, OptionalExtension, Transaction};
use serde::{Deserialize, Serialize};
use serde_json::Value;
use uuid::Uuid;

use super::resolver::extract_variables;

const SCHEMA_VERSION: i64 = 2;
pub(crate) const MAX_PROMPTS: usize = 1_000;
pub(crate) const MAX_VERSIONS_PER_PROMPT: usize = 100;
const MAX_NAME_CHARS: usize = 100;
const MAX_CONTENT_BYTES: usize = 100_000;
const MAX_CHANGE_NOTE_CHARS: usize = 500;
const DEFAULT_LIST_LIMIT: usize = 20;
const MAX_LIST_LIMIT: usize = 100;
const INITIAL_CHANGE_NOTE: &str = "initial version";
/// Movable reserved labels; `latest` is virtual and never stored.
const RESERVED_LABELS: [&str; 3] = ["production", "staging", "development"];
const MAX_DIFF_LINES: usize = 5_000;

const SCHEMA_DDL: &str = "CREATE TABLE IF NOT EXISTS prompts (
    id TEXT PRIMARY KEY,
    name TEXT UNIQUE NOT NULL,
    created_at INTEGER NOT NULL,
    updated_at INTEGER NOT NULL
);
CREATE TABLE IF NOT EXISTS prompt_versions (
    prompt_id TEXT NOT NULL REFERENCES prompts(id) ON DELETE CASCADE,
    version INTEGER NOT NULL,
    content TEXT NOT NULL,
    variables TEXT NOT NULL,
    change_note TEXT NOT NULL,
    created_by TEXT NOT NULL,
    created_at INTEGER NOT NULL,
    PRIMARY KEY (prompt_id, version)
);
CREATE INDEX IF NOT EXISTS idx_prompt_versions_prompt
    ON prompt_versions(prompt_id, version);
CREATE TABLE IF NOT EXISTS prompt_labels (
    prompt_id TEXT NOT NULL REFERENCES prompts(id) ON DELETE CASCADE,
    label TEXT NOT NULL,
    version INTEGER NOT NULL,
    moved_at INTEGER NOT NULL,
    PRIMARY KEY (prompt_id, label),
    FOREIGN KEY (prompt_id, version) REFERENCES prompt_versions(prompt_id, version)
);
CREATE TABLE IF NOT EXISTS prompt_promotions (
    id TEXT PRIMARY KEY,
    prompt_id TEXT NOT NULL REFERENCES prompts(id) ON DELETE CASCADE,
    label TEXT NOT NULL,
    from_version INTEGER,
    to_version INTEGER NOT NULL,
    actor TEXT NOT NULL,
    created_at INTEGER NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_prompt_promotions_prompt
    ON prompt_promotions(prompt_id, created_at);";

#[derive(Debug, thiserror::Error)]
pub enum StoreError {
    #[error("database operation failed")]
    Sqlite(#[from] rusqlite::Error),
    #[error("failed to prepare storage directory")]
    Io(#[from] std::io::Error),
    #[error("stored record is invalid")]
    Serialization(#[from] serde_json::Error),
    #[error("prompt store lock is unavailable")]
    Lock,
    #[error("{object} '{id}' was not found")]
    NotFound { object: &'static str, id: String },
    #[error("{0}")]
    InvalidRequest(String),
    #[error("{object} exceeds the {max_bytes} byte storage limit")]
    TooLarge {
        object: &'static str,
        max_bytes: usize,
    },
    #[error("{object} exceeds the {max_count} item limit")]
    LimitExceeded {
        object: &'static str,
        max_count: usize,
    },
}

#[derive(Debug, Clone, Deserialize)]
pub struct ListParams {
    #[serde(default = "default_list_limit")]
    pub limit: usize,
    #[serde(default)]
    pub order: ListOrder,
    #[serde(default)]
    pub after: Option<String>,
    #[serde(default)]
    pub before: Option<String>,
}

impl Default for ListParams {
    fn default() -> Self {
        Self {
            limit: DEFAULT_LIST_LIMIT,
            order: ListOrder::Desc,
            after: None,
            before: None,
        }
    }
}

fn default_list_limit() -> usize {
    DEFAULT_LIST_LIMIT
}

#[derive(Debug, Clone, Copy, Default, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum ListOrder {
    Asc,
    #[default]
    Desc,
}

#[derive(Debug, Clone)]
pub struct ListPage {
    pub data: Vec<Value>,
    pub has_more: bool,
}

impl ListPage {
    pub fn into_openai_response(self) -> Value {
        let first_id = self.data.first().and_then(|item| item.get("id")).cloned();
        let last_id = self.data.last().and_then(|item| item.get("id")).cloned();
        serde_json::json!({
            "object": "list",
            "data": self.data,
            "first_id": first_id,
            "last_id": last_id,
            "has_more": self.has_more
        })
    }
}

/// A prompt listing item: identity, per-label version numbers, the virtual
/// `latest` version, and timestamps.
#[derive(Debug, Clone, Serialize)]
pub struct Prompt {
    pub id: String,
    pub object: &'static str,
    pub name: String,
    pub labels: BTreeMap<String, i64>,
    pub latest_version: i64,
    pub version_count: i64,
    pub created_at: i64,
    pub updated_at: i64,
}

/// One immutable version of a prompt.
#[derive(Debug, Clone, Serialize)]
pub struct PromptVersion {
    pub prompt_id: String,
    pub object: &'static str,
    pub version: i64,
    pub content: String,
    pub variables: Vec<String>,
    pub change_note: String,
    pub created_by: String,
    pub created_at: i64,
}

/// One promotion-audit row: a label moved from one version to another.
#[derive(Debug, Clone, Serialize)]
pub struct PromptPromotion {
    pub id: String,
    pub prompt_id: String,
    pub object: &'static str,
    pub label: String,
    pub from_version: Option<i64>,
    pub to_version: i64,
    pub actor: String,
    pub created_at: i64,
}

/// One label entry for [`PromptStore::list_labels`]: movable labels carry
/// their current version (or none); the virtual `latest` label is computed
/// as `MAX(version)` and never stored.
#[derive(Debug, Clone, Serialize)]
pub struct PromptLabel {
    pub prompt_id: String,
    pub object: &'static str,
    pub label: String,
    pub version: Option<i64>,
    #[serde(rename = "virtual")]
    pub is_virtual: bool,
}

/// One per-line diff entry between two versions of a prompt.
#[derive(Debug, Clone, Serialize, PartialEq)]
#[serde(rename_all = "lowercase", tag = "kind")]
pub enum DiffLine {
    Added { text: String },
    Removed { text: String },
    Unchanged { text: String },
}

impl DiffLine {
    pub fn text(&self) -> &str {
        match self {
            DiffLine::Added { text }
            | DiffLine::Removed { text }
            | DiffLine::Unchanged { text } => text,
        }
    }
}

/// SQLite-backed prompt store. Clones share one synchronous connection.
#[derive(Clone)]
pub struct PromptStore {
    conn: Arc<Mutex<Connection>>,
}

impl PromptStore {
    /// Open or create the prompt database and bring its schema up to date.
    pub fn open(db_path: &Path) -> Result<Self, StoreError> {
        if let Some(parent) = db_path.parent() {
            if !parent.as_os_str().is_empty() {
                std::fs::create_dir_all(parent)?;
            }
        }

        let mut conn = Connection::open(db_path)?;
        conn.busy_timeout(Duration::from_secs(5))?;
        conn.pragma_update(None, "foreign_keys", "ON")?;
        if db_path != Path::new(":memory:") {
            conn.pragma_update(None, "journal_mode", "WAL")?;
        }
        conn.pragma_update(None, "synchronous", "NORMAL")?;
        Self::initialize_schema(&mut conn)?;

        Ok(Self {
            conn: Arc::new(Mutex::new(conn)),
        })
    }

    fn initialize_schema(conn: &mut Connection) -> Result<(), StoreError> {
        let current: i64 = conn.query_row("PRAGMA user_version", [], |row| row.get(0))?;
        if current > SCHEMA_VERSION {
            return Err(StoreError::InvalidRequest(format!(
                "prompt store schema version {current} is newer than supported version {SCHEMA_VERSION}"
            )));
        }
        if current == SCHEMA_VERSION {
            return Ok(());
        }

        // Additive by construction: every statement is IF NOT EXISTS, so a
        // pre-existing older file keeps all rows and only gains new objects.
        let transaction = conn.transaction()?;
        transaction.execute_batch(SCHEMA_DDL)?;
        transaction.pragma_update(None, "user_version", SCHEMA_VERSION)?;
        transaction.commit()?;
        Ok(())
    }

    /// Create a prompt as version 1 carrying the `production` label, with an
    /// initial promotion audit row, all inside one transaction.
    pub fn create_prompt(
        &self,
        name: &str,
        content: &str,
        created_by: &str,
    ) -> Result<Prompt, StoreError> {
        validate_name(name)?;
        validate_content(content)?;

        let now = Utc::now().timestamp();
        let id = Uuid::new_v4().to_string();

        let mut conn = self.lock()?;
        let transaction = conn.transaction()?;
        let prompt_count: i64 =
            transaction.query_row("SELECT COUNT(*) FROM prompts", [], |row| row.get(0))?;
        if prompt_count >= MAX_PROMPTS as i64 {
            return Err(StoreError::LimitExceeded {
                object: "prompts",
                max_count: MAX_PROMPTS,
            });
        }
        let duplicate: Option<String> = transaction
            .query_row(
                "SELECT id FROM prompts WHERE name = ?1",
                params![name],
                |row| row.get(0),
            )
            .optional()?;
        if duplicate.is_some() {
            return Err(StoreError::InvalidRequest(format!(
                "name '{name}' already exists"
            )));
        }

        transaction.execute(
            "INSERT INTO prompts (id, name, created_at, updated_at)
             VALUES (?1, ?2, ?3, ?4)",
            params![id, name, now, now],
        )?;
        insert_version(&transaction, &id, 1, content, INITIAL_CHANGE_NOTE, created_by, now)?;
        transaction.execute(
            "INSERT INTO prompt_labels (prompt_id, label, version, moved_at)
             VALUES (?1, 'production', 1, ?2)",
            params![id, now],
        )?;
        transaction.execute(
            "INSERT INTO prompt_promotions (id, prompt_id, label, from_version, to_version, actor, created_at)
             VALUES (?1, ?2, 'production', NULL, 1, ?3, ?4)",
            params![Uuid::new_v4().to_string(), id, created_by, now],
        )?;
        transaction.commit()?;
        drop(conn);

        Ok(Prompt {
            id,
            object: "prompt",
            name: name.to_owned(),
            labels: BTreeMap::from([("production".to_owned(), 1)]),
            latest_version: 1,
            version_count: 1,
            created_at: now,
            updated_at: now,
        })
    }

/// Append a new immutable version to an existing prompt inside one
/// transaction. The version number is `MAX(version) + 1`; there is no
/// update path for stored versions anywhere in the store. A `new_name`
/// renames the prompt in the SAME transaction (Req 1.3): same kebab
/// rules and name uniqueness as create, `None` keeps the current name.
pub fn create_version(
    &self,
    prompt_id: &str,
    new_name: Option<&str>,
    content: &str,
    change_note: &str,
    created_by: &str,
) -> Result<PromptVersion, StoreError> {
    validate_content(content)?;
    validate_change_note(change_note)?;
    if let Some(name) = new_name {
        validate_name(name)?;
    }

    let now = Utc::now().timestamp();

    let mut conn = self.lock()?;
    let transaction = conn.transaction()?;
    ensure_prompt_exists(&transaction, prompt_id)?;
    if let Some(name) = new_name {
        let duplicate: Option<i64> = transaction
            .query_row(
                "SELECT 1 FROM prompts WHERE name = ?1 AND id != ?2",
                params![name, prompt_id],
                |row| row.get(0),
            )
            .optional()?;
        if duplicate.is_some() {
            return Err(StoreError::InvalidRequest(format!(
                "name '{name}' already exists"
            )));
        }
    }
    let (version_count, latest): (i64, i64) = transaction.query_row(
        "SELECT COUNT(*), COALESCE(MAX(version), 0)
FROM prompt_versions WHERE prompt_id = ?1",
        params![prompt_id],
        |row| Ok((row.get(0)?, row.get(1)?)),
    )?;
    if version_count >= MAX_VERSIONS_PER_PROMPT as i64 {
        return Err(StoreError::LimitExceeded {
            object: "versions",
            max_count: MAX_VERSIONS_PER_PROMPT,
        });
    }

    let version = latest + 1;
    insert_version(&transaction, prompt_id, version, content, change_note, created_by, now)?;
    transaction.execute(
        "UPDATE prompts SET updated_at = ?1 WHERE id = ?2",
        params![now, prompt_id],
    )?;
    if let Some(name) = new_name {
        transaction.execute(
            "UPDATE prompts SET name = ?1 WHERE id = ?2",
            params![name, prompt_id],
        )?;
    }
    transaction.commit()?;
    drop(conn);

        Ok(PromptVersion {
            prompt_id: prompt_id.to_owned(),
            object: "prompt.version",
            version,
            content: content.to_owned(),
            variables: extract_variables(content),
            change_note: change_note.to_owned(),
            created_by: created_by.to_owned(),
            created_at: now,
        })
    }

    /// List prompts with an OpenAI-style cursor envelope.
    pub fn list_prompts(&self, list: &ListParams) -> Result<ListPage, StoreError> {
        if list.limit == 0 || list.limit > MAX_LIST_LIMIT {
            return Err(StoreError::InvalidRequest(format!(
                "limit must be between 1 and {MAX_LIST_LIMIT}"
            )));
        }
        if list.after.is_some() && list.before.is_some() {
            return Err(StoreError::InvalidRequest(
                "after and before cannot be used together".into(),
            ));
        }

        let conn = self.lock()?;
        let mut prompts: Vec<(String, String, i64, i64)> = Vec::new();
        {
            let mut statement = conn.prepare(
                "SELECT id, name, created_at, updated_at
                 FROM prompts ORDER BY created_at ASC, id ASC",
            )?;
            let rows = statement.query_map([], prompt_summary_row)?;
            for row in rows {
                prompts.push(row?);
            }
        }
        let mut label_rows: BTreeMap<String, BTreeMap<String, i64>> = BTreeMap::new();
        {
            let mut statement =
                conn.prepare("SELECT prompt_id, label, version FROM prompt_labels")?;
            let rows = statement.query_map([], |row| {
                Ok((
                    row.get::<_, String>(0)?,
                    row.get::<_, String>(1)?,
                    row.get::<_, i64>(2)?,
                ))
            })?;
            for row in rows {
                let (prompt_id, label, version) = row?;
                label_rows.entry(prompt_id).or_default().insert(label, version);
            }
        }
        let mut version_rows: BTreeMap<String, (i64, i64)> = BTreeMap::new();
        {
            let mut statement = conn.prepare(
                "SELECT prompt_id, COUNT(*), COALESCE(MAX(version), 0)
                 FROM prompt_versions GROUP BY prompt_id",
            )?;
            let rows = statement.query_map([], |row| {
                Ok((
                    row.get::<_, String>(0)?,
                    row.get::<_, i64>(1)?,
                    row.get::<_, i64>(2)?,
                ))
            })?;
            for row in rows {
                let (prompt_id, count, latest) = row?;
                version_rows.insert(prompt_id, (count, latest));
            }
        }
        drop(conn);

        let mut records: Vec<Value> = prompts
            .into_iter()
            .map(|(id, name, created_at, updated_at)| {
                let labels = label_rows.remove(&id).unwrap_or_default();
                let (version_count, latest_version) =
                    version_rows.get(&id).copied().unwrap_or((0, 0));
                serde_json::to_value(Prompt {
                    id,
                    object: "prompt",
                    name,
                    labels,
                    latest_version,
                    version_count,
                    created_at,
                    updated_at,
                })
            })
            .collect::<Result<Vec<_>, _>>()?;

        if matches!(list.order, ListOrder::Desc) {
            records.reverse();
        }

        if let Some(after) = &list.after {
            let index = cursor_index(&records, after)?;
            records = records.into_iter().skip(index + 1).collect();
        } else if let Some(before) = &list.before {
            let index = cursor_index(&records, before)?;
            records.truncate(index);
        }

        let has_more = records.len() > list.limit;
        records.truncate(list.limit);
        Ok(ListPage {
            data: records,
            has_more,
        })
    }

    /// Fetch one prompt by id.
    pub fn get_prompt(&self, id: &str) -> Result<Prompt, StoreError> {
        let conn = self.lock()?;
        let (name, created_at, updated_at): (String, i64, i64) = conn
            .query_row(
                "SELECT name, created_at, updated_at FROM prompts WHERE id = ?1",
                params![id],
                |row| Ok((row.get(0)?, row.get(1)?, row.get(2)?)),
            )
            .optional()?
            .ok_or_else(|| StoreError::NotFound {
                object: "prompt",
                id: id.to_owned(),
            })?;
        let mut labels = BTreeMap::new();
        {
            let mut statement =
                conn.prepare("SELECT label, version FROM prompt_labels WHERE prompt_id = ?1")?;
            let rows = statement.query_map(params![id], |row| {
                Ok((row.get::<_, String>(0)?, row.get::<_, i64>(1)?))
            })?;
            for row in rows {
                let (label, version) = row?;
                labels.insert(label, version);
            }
        }
        let (version_count, latest_version): (i64, i64) = conn.query_row(
            "SELECT COUNT(*), COALESCE(MAX(version), 0)
             FROM prompt_versions WHERE prompt_id = ?1",
            params![id],
            |row| Ok((row.get(0)?, row.get(1)?)),
        )?;

        Ok(Prompt {
            id: id.to_owned(),
            object: "prompt",
            name,
            labels,
            latest_version,
            version_count,
            created_at,
            updated_at,
        })
    }

    /// Delete a prompt and all of its versions, labels, and promotion rows.
    pub fn delete_prompt(&self, id: &str) -> Result<(), StoreError> {
        let mut conn = self.lock()?;
        let transaction = conn.transaction()?;
        ensure_prompt_exists(&transaction, id)?;
        transaction.execute(
            "DELETE FROM prompt_labels WHERE prompt_id = ?1",
            params![id],
        )?;
        transaction.execute(
            "DELETE FROM prompt_promotions WHERE prompt_id = ?1",
            params![id],
        )?;
        transaction.execute(
            "DELETE FROM prompt_versions WHERE prompt_id = ?1",
            params![id],
        )?;
        transaction.execute("DELETE FROM prompts WHERE id = ?1", params![id])?;
        transaction.commit()?;
        Ok(())
    }

    /// List all versions of a prompt, newest first.
    pub fn list_versions(&self, prompt_id: &str) -> Result<Vec<PromptVersion>, StoreError> {
        let conn = self.lock()?;
        ensure_prompt_exists(&conn, prompt_id)?;
        let mut statement = conn.prepare(
            "SELECT version, content, variables, change_note, created_by, created_at
            FROM prompt_versions WHERE prompt_id = ?1 ORDER BY version DESC",
        )?;
        let rows = statement.query_map(params![prompt_id], |row| {
            let variables: Vec<String> =
                serde_json::from_str(&row.get::<_, String>(2)?).unwrap_or_default();
            Ok(PromptVersion {
                prompt_id: prompt_id.to_owned(),
                object: "prompt.version",
                version: row.get(0)?,
                content: row.get(1)?,
                variables,
                change_note: row.get(3)?,
                created_by: row.get(4)?,
                created_at: row.get(5)?,
            })
        })?;
        let mut versions = Vec::new();
        for row in rows {
            versions.push(row?);
        }
        Ok(versions)
    }

    /// Fetch one version by number with a single SQL select instead of
    /// loading and cloning every stored version of the prompt.
    pub fn get_version(&self, prompt_id: &str, version: i64) -> Result<PromptVersion, StoreError> {
        let conn = self.lock()?;
        ensure_prompt_exists(&conn, prompt_id)?;
        drop(conn);
        self.get_version_unchecked(prompt_id, version)
    }

    /// Single-version fetch; the caller has already verified the prompt
    /// exists (same connection discipline as the other read paths).
    fn get_version_unchecked(&self, prompt_id: &str, version: i64) -> Result<PromptVersion, StoreError> {
        let conn = self.lock()?;
        conn.query_row(
            "SELECT version, content, variables, change_note, created_by, created_at
            FROM prompt_versions WHERE prompt_id = ?1 AND version = ?2",
            params![prompt_id, version],
            |row| {
                let variables: Vec<String> =
                    serde_json::from_str(&row.get::<_, String>(2)?).unwrap_or_default();
                Ok(PromptVersion {
                    prompt_id: prompt_id.to_owned(),
                    object: "prompt.version",
                    version: row.get(0)?,
                    content: row.get(1)?,
                    variables,
                    change_note: row.get(3)?,
                    created_by: row.get(4)?,
                    created_at: row.get(5)?,
                })
            },
        )
        .optional()?
        .ok_or_else(|| StoreError::NotFound {
            object: "version",
            id: format!("{prompt_id}/{version}"),
        })
    }

    /// Atomically move a reserved movable label to a version and record a
    /// promotion audit row. The virtual `latest` label (always the newest
    /// version) is rejected with a field-naming error, as is any label
    /// outside the reserved set.
    pub fn promote_label(
        &self,
        prompt_id: &str,
        label: &str,
        to_version: i64,
        actor: &str,
    ) -> Result<(), StoreError> {
        if label == "latest" {
            return Err(StoreError::InvalidRequest(
                "label 'latest' is virtual and always points to the newest version; it cannot be moved"
                    .into(),
            ));
        }
        if !RESERVED_LABELS.contains(&label) {
            return Err(StoreError::InvalidRequest(format!(
                "label must be one of production, staging, development"
            )));
        }

        let now = Utc::now().timestamp();
        let mut conn = self.lock()?;
        let transaction = conn.transaction()?;
        ensure_prompt_exists(&transaction, prompt_id)?;
        let target: Option<i64> = transaction
            .query_row(
                "SELECT version FROM prompt_versions WHERE prompt_id = ?1 AND version = ?2",
                params![prompt_id, to_version],
                |row| row.get(0),
            )
            .optional()?;
        if target.is_none() {
            return Err(StoreError::NotFound {
                object: "version",
                id: format!("{prompt_id}/{to_version}"),
            });
        }
        let from_version: Option<i64> = transaction
            .query_row(
                "SELECT version FROM prompt_labels WHERE prompt_id = ?1 AND label = ?2",
                params![prompt_id, label],
                |row| row.get(0),
            )
            .optional()?;
        transaction.execute(
            "INSERT INTO prompt_labels (prompt_id, label, version, moved_at)
            VALUES (?1, ?2, ?3, ?4)
            ON CONFLICT(prompt_id, label) DO UPDATE SET version = excluded.version, moved_at = excluded.moved_at",
            params![prompt_id, label, to_version, now],
        )?;
        transaction.execute(
            "INSERT INTO prompt_promotions (id, prompt_id, label, from_version, to_version, actor, created_at)
            VALUES (?1, ?2, ?3, ?4, ?5, ?6, ?7)",
            params![
                Uuid::new_v4().to_string(),
                prompt_id,
                label,
                from_version,
                to_version,
                actor,
                now
            ],
        )?;
        transaction.commit()?;
        Ok(())
    }

    /// List promotion-audit rows for a prompt, newest first.
    pub fn list_promotions(&self, prompt_id: &str) -> Result<ListPage, StoreError> {
        let conn = self.lock()?;
        ensure_prompt_exists(&conn, prompt_id)?;
        let mut promotions = Vec::new();
        {
            let mut statement = conn.prepare(
                "SELECT id, label, from_version, to_version, actor, created_at
                FROM prompt_promotions WHERE prompt_id = ?1
                ORDER BY created_at DESC, id DESC",
            )?;
            let rows = statement.query_map(params![prompt_id], |row| {
                Ok(PromptPromotion {
                    id: row.get(0)?,
                    prompt_id: prompt_id.to_owned(),
                    object: "prompt.promotion",
                    label: row.get(1)?,
                    from_version: row.get(2)?,
                    to_version: row.get(3)?,
                    actor: row.get(4)?,
                    created_at: row.get(5)?,
                })
            })?;
            for row in rows {
                promotions.push(row?);
            }
        }
        drop(conn);
        let data = promotions
            .into_iter()
            .map(serde_json::to_value)
            .collect::<Result<Vec<_>, _>>()?;
        Ok(ListPage { data, has_more: false })
    }

    /// List the reserved labels for a prompt with their current version
    /// (none when the label has never been moved). The `latest` label is
    /// virtual: computed as `MAX(version)` and included marked virtual.
    pub fn list_labels(&self, prompt_id: &str) -> Result<Vec<PromptLabel>, StoreError> {
        let conn = self.lock()?;
        ensure_prompt_exists(&conn, prompt_id)?;
        let mut stored: BTreeMap<String, i64> = BTreeMap::new();
        {
            let mut statement =
                conn.prepare("SELECT label, version FROM prompt_labels WHERE prompt_id = ?1")?;
            let rows = statement.query_map(params![prompt_id], |row| {
                Ok((row.get::<_, String>(0)?, row.get::<_, i64>(1)?))
            })?;
            for row in rows {
                let (label, version) = row?;
                stored.insert(label, version);
            }
        }
        let latest: i64 = conn.query_row(
            "SELECT COALESCE(MAX(version), 0) FROM prompt_versions WHERE prompt_id = ?1",
            params![prompt_id],
            |row| row.get(0),
        )?;

        let mut labels = Vec::new();
        for label in RESERVED_LABELS {
            labels.push(PromptLabel {
                prompt_id: prompt_id.to_owned(),
                object: "prompt.label",
                label: label.to_owned(),
                version: stored.get(label).copied(),
                is_virtual: false,
            });
        }
        labels.push(PromptLabel {
            prompt_id: prompt_id.to_owned(),
            object: "prompt.label",
            label: "latest".to_owned(),
            version: if latest > 0 { Some(latest) } else { None },
            is_virtual: true,
        });
        Ok(labels)
    }

    /// Per-line diff between two stored versions of a prompt. Sides beyond
    /// [`MAX_DIFF_LINES`] lines fall back to whole-block added/removed so
    /// the LCS matrix stays bounded.
    pub fn diff_versions(
        &self,
        prompt_id: &str,
        from_version: i64,
        to_version: i64,
    ) -> Result<Vec<DiffLine>, StoreError> {
        if from_version == to_version {
            return Err(StoreError::InvalidRequest(
                "from_version and to_version must differ".into(),
            ));
        }
        let conn = self.lock()?;
        ensure_prompt_exists(&conn, prompt_id)?;
        let from = fetch_version_content(&conn, prompt_id, from_version)?;
        let to = fetch_version_content(&conn, prompt_id, to_version)?;
        drop(conn);

        // A single trailing newline is a file terminator, not a line:
        // strip one from each side so it yields no phantom empty line.
        let from_trimmed = from.strip_suffix('\n').unwrap_or(&from[..]);
        let to_trimmed = to.strip_suffix('\n').unwrap_or(&to[..]);
        let from_lines: Vec<&str> = from_trimmed.split('\n').collect();
        let to_lines: Vec<&str> = to_trimmed.split('\n').collect();
        if from_lines.len() > MAX_DIFF_LINES || to_lines.len() > MAX_DIFF_LINES {
            return Ok(whole_block_diff(&from_lines, &to_lines));
        }
        Ok(diff_lines(&from_lines, &to_lines))
    }

    fn lock(&self) -> Result<MutexGuard<'_, Connection>, StoreError> {
        self.conn.lock().map_err(|_| StoreError::Lock)
    }
}

fn prompt_summary_row(row: &rusqlite::Row<'_>) -> rusqlite::Result<(String, String, i64, i64)> {
    Ok((
        row.get(0)?,
        row.get(1)?,
        row.get(2)?,
        row.get(3)?,
    ))
}

fn insert_version(
    transaction: &Transaction<'_>,
    prompt_id: &str,
    version: i64,
    content: &str,
    change_note: &str,
    created_by: &str,
    created_at: i64,
) -> Result<(), StoreError> {
    let variables = serde_json::to_string(&extract_variables(content))?;
    transaction.execute(
        "INSERT INTO prompt_versions
             (prompt_id, version, content, variables, change_note, created_by, created_at)
         VALUES (?1, ?2, ?3, ?4, ?5, ?6, ?7)",
        params![prompt_id, version, content, variables, change_note, created_by, created_at],
    )?;
    Ok(())
}

fn ensure_prompt_exists(conn: &Connection, prompt_id: &str) -> Result<(), StoreError> {
    let exists: Option<i64> = conn
        .query_row(
            "SELECT 1 FROM prompts WHERE id = ?1",
            params![prompt_id],
            |row| row.get(0),
        )
        .optional()?;
    if exists.is_none() {
        return Err(StoreError::NotFound {
            object: "prompt",
            id: prompt_id.to_owned(),
        });
    }
    Ok(())
}

/// Load one version's content or fail with a version NotFound.
fn fetch_version_content(
    conn: &Connection,
    prompt_id: &str,
    version: i64,
) -> Result<String, StoreError> {
    conn.query_row(
        "SELECT content FROM prompt_versions WHERE prompt_id = ?1 AND version = ?2",
        params![prompt_id, version],
        |row| row.get(0),
    )
    .optional()?
    .ok_or_else(|| StoreError::NotFound {
        object: "version",
        id: format!("{prompt_id}/{version}"),
    })
}

/// Whole-block fallback: every `from` line removed, every `to` line
/// added; used when the LCS matrix would exceed its bounded size.
fn whole_block_diff(from: &[&str], to: &[&str]) -> Vec<DiffLine> {
    let mut lines = Vec::new();
    for text in from {
        lines.push(DiffLine::Removed {
            text: (*text).to_owned(),
        });
    }
    for text in to {
        lines.push(DiffLine::Added {
            text: (*text).to_owned(),
        });
    }
    lines
}

/// Per-line LCS diff between two line slices, capped by
/// [`MAX_DIFF_LINES`] on both sides before this is called.
fn diff_lines(from: &[&str], to: &[&str]) -> Vec<DiffLine> {
    let rows = from.len() + 1;
    let columns = to.len() + 1;
    // u32 cells halve the matrix footprint at the line cap; the checked
    // lengths guarantee a future cap raise cannot silently wrap.
    let (cell_rows, cell_columns) = match (u32::try_from(rows), u32::try_from(columns)) {
        (Ok(rows), Ok(columns))
            if rows <= MAX_DIFF_LINES as u32 + 1 && columns <= MAX_DIFF_LINES as u32 + 1 =>
        {
            (rows, columns)
        }
        _ => return whole_block_diff(from, to),
    };
    let mut table: Vec<Vec<u32>> = vec![vec![0_u32; cell_columns as usize]; cell_rows as usize];
    for (i, from_line) in from.iter().enumerate().rev() {
        for (j, to_line) in to.iter().enumerate().rev() {
            table[i][j] = if from_line == to_line {
                table[i + 1][j + 1] + 1
            } else {
                table[i + 1][j].max(table[i][j + 1])
            };
        }
    }

    let mut lines = Vec::new();
    let mut i = 0;
    let mut j = 0;
    while i < from.len() && j < to.len() {
        if from[i] == to[j] {
            lines.push(DiffLine::Unchanged {
                text: from[i].to_owned(),
            });
            i += 1;
            j += 1;
        } else if table[i + 1][j] >= table[i][j + 1] {
            lines.push(DiffLine::Removed {
                text: from[i].to_owned(),
            });
            i += 1;
        } else {
            lines.push(DiffLine::Added {
                text: to[j].to_owned(),
            });
            j += 1;
        }
    }
    for text in &from[i..] {
        lines.push(DiffLine::Removed {
            text: (*text).to_owned(),
        });
    }
    for text in &to[j..] {
        lines.push(DiffLine::Added {
            text: (*text).to_owned(),
        });
    }
    lines
}

fn cursor_index(records: &[Value], cursor: &str) -> Result<usize, StoreError> {
    records
        .iter()
        .position(|record| record.get("id").and_then(Value::as_str) == Some(cursor))
        .ok_or_else(|| StoreError::InvalidRequest(format!("cursor '{cursor}' was not found")))
}

/// Lowercase-kebab check without regex: lowercase ASCII letters and digits
/// separated by single hyphens, no leading or trailing hyphen.
fn validate_name(name: &str) -> Result<(), StoreError> {
    if name.is_empty() {
        return Err(StoreError::InvalidRequest("name must not be empty".into()));
    }
    let length = name.chars().count();
    if length > MAX_NAME_CHARS {
        return Err(StoreError::InvalidRequest(format!(
            "name has {length} characters; expected at most {MAX_NAME_CHARS} characters"
        )));
    }
    let mut previous_hyphen = false;
    for (index, character) in name.chars().enumerate() {
        if character == '-' {
            if index == 0 {
                return Err(StoreError::InvalidRequest(
                    "name must not start with a hyphen".into(),
                ));
            }
            if previous_hyphen {
                return Err(StoreError::InvalidRequest(
                    "name must not contain consecutive hyphens".into(),
                ));
            }
            previous_hyphen = true;
        } else {
            if !character.is_ascii_lowercase() && !character.is_ascii_digit() {
                return Err(StoreError::InvalidRequest(
                    "name must be lowercase ASCII letters, digits, and single hyphens".into(),
                ));
            }
            previous_hyphen = false;
        }
    }
    if previous_hyphen {
        return Err(StoreError::InvalidRequest(
            "name must not end with a hyphen".into(),
        ));
    }
    Ok(())
}

fn validate_content(content: &str) -> Result<(), StoreError> {
    if content.is_empty() {
        return Err(StoreError::InvalidRequest(
            "content must not be empty".into(),
        ));
    }
    if content.len() > MAX_CONTENT_BYTES {
        return Err(StoreError::TooLarge {
            object: "content",
            max_bytes: MAX_CONTENT_BYTES,
        });
    }
    Ok(())
}

fn validate_change_note(change_note: &str) -> Result<(), StoreError> {
    let length = change_note.chars().count();
    if length == 0 || length > MAX_CHANGE_NOTE_CHARS {
        return Err(StoreError::InvalidRequest(format!(
            "change_note must be between 1 and {MAX_CHANGE_NOTE_CHARS} characters"
        )));
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    fn temp_store() -> (tempfile::TempDir, PromptStore) {
        let temp = tempfile::tempdir().unwrap();
        let store = PromptStore::open(&temp.path().join("prompts.db")).unwrap();
        (temp, store)
    }

    fn user_version(store: &PromptStore) -> i64 {
        store
            .lock()
            .unwrap()
            .query_row("PRAGMA user_version", [], |row| row.get(0))
            .unwrap()
    }

    fn table_exists(store: &PromptStore, table: &str) -> bool {
        store
            .lock()
            .unwrap()
            .query_row(
                "SELECT 1 FROM sqlite_master WHERE type = 'table' AND name = ?1",
                params![table],
                |row| row.get::<_, i64>(0),
            )
            .optional()
            .unwrap()
            .is_some()
    }

    #[test]
    fn fresh_open_is_dormant_and_lists_empty() {
        let (temp, store) = temp_store();
        let page = store.list_prompts(&ListParams::default()).unwrap();
        let envelope = page.into_openai_response();
        assert_eq!(envelope["object"], "list");
        assert_eq!(envelope["data"], json!([]));
        assert_eq!(envelope["first_id"], Value::Null);
        assert_eq!(envelope["last_id"], Value::Null);
        assert_eq!(envelope["has_more"], false);
        assert_eq!(user_version(&store), SCHEMA_VERSION);
        for table in [
            "prompts",
            "prompt_versions",
            "prompt_labels",
            "prompt_promotions",
        ] {
            assert!(table_exists(&store, table));
        }
        drop(store);
        // Reopening an up-to-date file is a no-op fast path.
        let reopened = PromptStore::open(&temp.path().join("prompts.db")).unwrap();
        assert_eq!(user_version(&reopened), SCHEMA_VERSION);
    }

    #[test]
    fn create_prompt_creates_v1_production_label_and_promotion_row() {
        let (_temp, store) = temp_store();
        let prompt = store
            .create_prompt("greeting", "Hello {{name}}!", "admin")
            .unwrap();

        assert_eq!(prompt.name, "greeting");
        assert_eq!(prompt.labels.get("production"), Some(&1));
        assert_eq!(prompt.latest_version, 1);
        assert_eq!(prompt.version_count, 1);
        assert!(prompt.created_at > 0);
        assert_eq!(prompt.created_at, prompt.updated_at);

        let conn = store.lock().unwrap();
        let (content, variables, change_note, created_by): (String, String, String, String) = conn
            .query_row(
                "SELECT content, variables, change_note, created_by
                 FROM prompt_versions WHERE prompt_id = ?1 AND version = 1",
                params![prompt.id],
                |row| Ok((row.get(0)?, row.get(1)?, row.get(2)?, row.get(3)?)),
            )
            .unwrap();
        assert_eq!(content, "Hello {{name}}!");
        assert_eq!(variables, json!(["name"]).to_string());
        assert_eq!(change_note, "initial version");
        assert_eq!(created_by, "admin");

        let label_version: i64 = conn
            .query_row(
                "SELECT version FROM prompt_labels WHERE prompt_id = ?1 AND label = 'production'",
                params![prompt.id],
                |row| row.get(0),
            )
            .unwrap();
        assert_eq!(label_version, 1);

        let (from_version, to_version, label, actor): (Option<i64>, i64, String, String) = conn
            .query_row(
                "SELECT from_version, to_version, label, actor
                 FROM prompt_promotions WHERE prompt_id = ?1",
                params![prompt.id],
                |row| Ok((row.get(0)?, row.get(1)?, row.get(2)?, row.get(3)?)),
            )
            .unwrap();
        assert_eq!(from_version, None);
        assert_eq!(to_version, 1);
        assert_eq!(label, "production");
        assert_eq!(actor, "admin");
    }

    #[test]
    fn create_prompt_rejects_invalid_names() {
        let (_temp, store) = temp_store();
        for name in [
            "",
            &"a".repeat(MAX_NAME_CHARS + 1),
            "Greeting",
            "double--hyphen",
            "-leading",
            "trailing-",
        ] {
            let error = store.create_prompt(name, "content", "admin").unwrap_err();
            assert!(
                matches!(error, StoreError::InvalidRequest(ref message) if message.contains("name")),
                "expected name-naming error for {name:?}, got {error:?}"
            );
        }
    // Nothing was stored by any failed attempt.
    assert_eq!(
        store.list_prompts(&ListParams::default()).unwrap().data.len(),
        0
    );
}

/// The MAX_PROMPTS cap is enforced inside create_prompt's transactional
/// COUNT check: seeding 999 rows via direct SQL (fast) plus one real
/// create fills the store, and the next create is rejected with
/// `LimitExceeded` naming the cap, storing nothing.
#[test]
fn create_prompt_enforces_thousand_prompt_cap() {
    let (_temp, store) = temp_store();
    {
        let mut conn = store.lock().unwrap();
        let transaction = conn.transaction().unwrap();
        for i in 0..(MAX_PROMPTS - 1) {
            transaction
                .execute(
                    "INSERT INTO prompts (id, name, created_at, updated_at)
                     VALUES (?1, ?2, 1700000000, 1700000000)",
                    params![format!("seed-{i}"), format!("seed-{i}")],
                )
                .unwrap();
        }
        transaction.commit().unwrap();
    }
    store.create_prompt("filler", "content", "admin").unwrap();
    assert_eq!(prompt_count(&store), MAX_PROMPTS as i64);

    let error = store
        .create_prompt("one-too-many", "content", "admin")
        .unwrap_err();
    assert!(
        matches!(error, StoreError::LimitExceeded { object: "prompts", max_count: MAX_PROMPTS }),
        "expected LimitExceeded naming the prompts cap, got {error:?}"
    );
    assert_eq!(
        prompt_count(&store),
        MAX_PROMPTS as i64,
        "the rejected create stored nothing"
    );
}

fn prompt_count(store: &PromptStore) -> i64 {
    store
        .lock()
        .unwrap()
        .query_row("SELECT COUNT(*) FROM prompts", [], |row| row.get(0))
        .unwrap()
}

    #[test]
    fn create_prompt_rejects_out_of_bounds_content() {
        let (_temp, store) = temp_store();
        let empty = store.create_prompt("empty", "", "admin").unwrap_err();
        assert!(
            matches!(empty, StoreError::InvalidRequest(ref message) if message.contains("content"))
        );
        let oversized = store
            .create_prompt("oversized", &"x".repeat(MAX_CONTENT_BYTES + 1), "admin")
            .unwrap_err();
        assert!(matches!(
            oversized,
            StoreError::TooLarge {
                object: "content",
                max_bytes: MAX_CONTENT_BYTES
            }
        ));
    }

    #[test]
    fn create_prompt_rejects_duplicate_name() {
        let (_temp, store) = temp_store();
        store.create_prompt("unique", "one", "admin").unwrap();
        let error = store
            .create_prompt("unique", "two", "admin")
            .unwrap_err();
        assert!(
            matches!(error, StoreError::InvalidRequest(ref message) if message.contains("name"))
        );
        // The original prompt is unchanged.
        let original = store.list_prompts(&ListParams::default()).unwrap();
        assert_eq!(original.data.len(), 1);
        assert_eq!(original.data[0]["version_count"], 1);
    }

    #[test]
    fn create_version_increments_extracts_variables_and_enforces_note_bounds() {
        let (_temp, store) = temp_store();
        let prompt = store
            .create_prompt(
                "story",
                "Dear {{zeta}} and {{alpha}}, again {{zeta}}.",
                "admin",
            )
            .unwrap();

        let first = store.list_versions(&prompt.id).unwrap();
        assert_eq!(first.len(), 1);
        assert_eq!(first[0].version, 1);
        assert_eq!(first[0].variables, vec!["alpha".to_owned(), "zeta".to_owned()]);

        let before = store.get_prompt(&prompt.id).unwrap().updated_at;
        let second = store
            .create_version(&prompt.id, None, "Now {{gamma}} only.", "swap variables", "editor")
            .unwrap();
        assert_eq!(second.version, 2);
        assert_eq!(second.variables, vec!["gamma".to_owned()]);
        assert_eq!(second.change_note, "swap variables");

        let after = store.get_prompt(&prompt.id).unwrap();
        assert!(after.updated_at >= before);
        assert_eq!(after.latest_version, 2);
        assert_eq!(after.version_count, 2);

        // Newest-first ordering.
        let versions = store.list_versions(&prompt.id).unwrap();
        assert_eq!(
            versions.iter().map(|version| version.version).collect::<Vec<_>>(),
            vec![2, 1]
        );

        for note in ["", &"n".repeat(MAX_CHANGE_NOTE_CHARS + 1)] {
            let error = store
                .create_version(&prompt.id, None, "content", note, "admin")
                .unwrap_err();
            assert!(
                matches!(error, StoreError::InvalidRequest(ref message) if message.contains("change_note")),
                "expected change_note error for {note:?}, got {error:?}"
            );
        }
        // Failed writes stored nothing.
        assert_eq!(store.get_prompt(&prompt.id).unwrap().version_count, 2);
}

    #[test]
    fn create_version_renames_atomically_and_frees_the_old_name() {
        let (_temp, store) = temp_store();
        let other = store.create_prompt("taken", "v1", "admin").unwrap();
        let prompt = store.create_prompt("greeting", "Hello!", "admin").unwrap();

        // Rename to a taken name is rejected and stores nothing.
        let error = store
         .create_version(&prompt.id, Some("taken"), "Hello again!", "rename try", "admin")
         .unwrap_err();
        assert!(
            matches!(error, StoreError::InvalidRequest(ref message) if message.contains("name"))
        );
        assert_eq!(store.get_prompt(&prompt.id).unwrap().name, "greeting");
        assert_eq!(store.get_prompt(&prompt.id).unwrap().version_count, 1);

        // Invalid names are rejected with the same kebab rules as create.
        for bad in ["", "UPPER", "-lead", "double--hyphen", &"x".repeat(MAX_NAME_CHARS + 1)] {
            let error = store
             .create_version(&prompt.id, Some(bad), "content", "note", "admin")
             .unwrap_err();
            assert!(
                matches!(error, StoreError::InvalidRequest(ref message) if message.contains("name")),
                "expected name error for {bad:?}, got {error:?}"
            );
        }

        // Rename plus new version lands in one transaction: new name, same id,
        // version count up, and the old name is free again.
        let version = store
         .create_version(&prompt.id, Some("welcome"), "Hello again!", "rename", "admin")
         .unwrap();
        assert_eq!(version.version, 2);
        let renamed = store.get_prompt(&prompt.id).unwrap();
        assert_eq!(renamed.name, "welcome");
        assert_eq!(renamed.version_count, 2);
        let recreated = store.create_prompt("greeting", "fresh", "admin").unwrap();
        assert_ne!(recreated.id, prompt.id);
        assert_ne!(recreated.id, other.id);

        // Same-name rename keeps the row intact.
        store
         .create_version(&prompt.id, Some("welcome"), "Hello thrice!", "same name", "admin")
         .unwrap();
        assert_eq!(store.get_prompt(&prompt.id).unwrap().name, "welcome");
        assert_eq!(store.get_prompt(&prompt.id).unwrap().version_count, 3);

        // Name-only edit (same content) still creates a new immutable version.
        store
         .create_version(&prompt.id, Some("welcome-v2"), "Hello thrice!", "name only", "admin")
         .unwrap();
        let versions = store.list_versions(&prompt.id).unwrap();
        assert_eq!(
            versions.iter().map(|version| version.version).collect::<Vec<_>>(),
            vec![4, 3, 2, 1]
        );
        assert_eq!(store.get_prompt(&prompt.id).unwrap().name, "welcome-v2");
    }

    #[test]
    fn create_version_enforces_hundred_version_cap() {
        let (_temp, store) = temp_store();
        let prompt = store.create_prompt("capped", "v1", "admin").unwrap();
        for version in 2..=MAX_VERSIONS_PER_PROMPT {
            store
                .create_version(&prompt.id, None, "next", &format!("note {version}"), "admin")
                .unwrap();
        }
        assert_eq!(
            store.get_prompt(&prompt.id).unwrap().version_count,
            MAX_VERSIONS_PER_PROMPT as i64
        );
        let error = store
            .create_version(&prompt.id, None, "one too many", "note", "admin")
            .unwrap_err();
        assert!(matches!(
            error,
            StoreError::LimitExceeded {
                object: "versions",
                max_count: MAX_VERSIONS_PER_PROMPT
            }
        ));
    }

    #[test]
    fn list_prompts_envelope_limits_and_cursors() {
        let (_temp, store) = temp_store();
        let mut ids = Vec::new();
        for name in ["prompt-a", "prompt-b", "prompt-c"] {
            ids.push(store.create_prompt(name, "content", "admin").unwrap().id);
        }
        // Force distinct ordering timestamps (epoch seconds can tie in tests).
        {
            let conn = store.lock().unwrap();
            for (index, id) in ids.iter().enumerate() {
                conn.execute(
                    "UPDATE prompts SET created_at = ?1 WHERE id = ?2",
                    params![100 * (index as i64 + 1), id],
                )
                .unwrap();
            }
        }

        let first_page = store
            .list_prompts(&ListParams {
                limit: 2,
                ..ListParams::default()
            })
            .unwrap();
        assert_eq!(first_page.data.len(), 2);
        assert!(first_page.has_more);
        // Default order is descending: newest (prompt-c) first.
        assert_eq!(first_page.data[0]["name"], "prompt-c");
        assert_eq!(first_page.data[0]["labels"]["production"], 1);
        assert_eq!(first_page.data[0]["version_count"], 1);
        assert_eq!(first_page.data[1]["name"], "prompt-b");

        let envelope = first_page.clone().into_openai_response();
        assert_eq!(envelope["object"], "list");
        let first_id = envelope["first_id"].as_str().unwrap().to_owned();
        let last_id = envelope["last_id"].as_str().unwrap().to_owned();
        assert_eq!(first_id, ids[2]);
        assert_eq!(last_id, ids[1]);

        let second_page = store
            .list_prompts(&ListParams {
                limit: 2,
                after: Some(last_id),
                ..ListParams::default()
            })
            .unwrap();
        assert_eq!(second_page.data.len(), 1);
        assert!(!second_page.has_more);
        assert_eq!(second_page.data[0]["name"], "prompt-a");

        let ascending = store
            .list_prompts(&ListParams {
                limit: 2,
                order: ListOrder::Asc,
                ..ListParams::default()
            })
            .unwrap();
        assert_eq!(ascending.data[0]["name"], "prompt-a");

        for limit in [0, MAX_LIST_LIMIT + 1] {
            let error = store
                .list_prompts(&ListParams {
                    limit,
                    ..ListParams::default()
                })
                .unwrap_err();
            assert!(
                matches!(error, StoreError::InvalidRequest(ref message) if message.contains("limit"))
            );
        }
    }

    #[test]
    fn delete_prompt_cascades_and_get_returns_not_found() {
        let (_temp, store) = temp_store();
        let prompt = store.create_prompt("doomed", "content", "admin").unwrap();
        store
            .create_version(&prompt.id, None, "content two", "second", "admin")
            .unwrap();

        store.delete_prompt(&prompt.id).unwrap();

        let error = store.get_prompt(&prompt.id).unwrap_err();
        assert!(matches!(
            error,
            StoreError::NotFound { object: "prompt", ref id } if id == &prompt.id
        ));
        assert!(matches!(
            store.list_versions(&prompt.id),
            Err(StoreError::NotFound { .. })
        ));
        {
            let conn = store.lock().unwrap();
            for table in [
                "prompts",
                "prompt_versions",
                "prompt_labels",
                "prompt_promotions",
            ] {
                let count: i64 = conn
                    .query_row(
                        &format!("SELECT COUNT(*) FROM {table}"),
                        [],
                        |row| row.get(0),
                    )
                    .unwrap();
                assert_eq!(count, 0, "{table} should be empty after delete");
            }
        }
        assert!(matches!(
            store.delete_prompt(&prompt.id),
            Err(StoreError::NotFound { .. })
        ));
    }

    #[test]
    fn opening_schema_v1_file_migrates_additively() {
        let temp = tempfile::tempdir().unwrap();
        let path = temp.path().join("prompts.db");
        {
            let conn = Connection::open(&path).unwrap();
        conn.execute_batch(
            "CREATE TABLE prompts (
                id TEXT PRIMARY KEY,
                name TEXT UNIQUE NOT NULL,
                created_at INTEGER NOT NULL,
                updated_at INTEGER NOT NULL
            );
            CREATE TABLE prompt_versions (
                prompt_id TEXT NOT NULL,
                version INTEGER NOT NULL,
                content TEXT NOT NULL,
                variables TEXT NOT NULL,
                change_note TEXT NOT NULL,
                created_by TEXT NOT NULL,
                created_at INTEGER NOT NULL,
                PRIMARY KEY (prompt_id, version)
            );
            INSERT INTO prompts VALUES
                ('00000000-0000-0000-0000-000000000001', 'legacy-prompt', 1, 1);
            INSERT INTO prompt_versions VALUES
                ('00000000-0000-0000-0000-000000000001', 1,
                'legacy content', '[]', 'legacy note', 'admin', 1);
            PRAGMA user_version = 1;",
        )
        .unwrap();
            }
    
            let store = PromptStore::open(&path).unwrap();
            assert_eq!(user_version(&store), SCHEMA_VERSION);
            assert!(table_exists(&store, "prompt_labels"));
            assert!(table_exists(&store, "prompt_promotions"));
    
            // Pre-existing rows survive the additive migration.
            let migrated = store
                .get_prompt("00000000-0000-0000-0000-000000000001")
                .unwrap();
            assert_eq!(migrated.name, "legacy-prompt");
            assert_eq!(migrated.version_count, 1);
            assert_eq!(migrated.latest_version, 1);
            let versions = store
                .list_versions("00000000-0000-0000-0000-000000000001")
                .unwrap();
            assert_eq!(versions[0].content, "legacy content");
            assert_eq!(versions[0].change_note, "legacy note");
    
    // The migrated store accepts new writes alongside legacy rows.
    let fresh = store.create_prompt("post-migration", "new", "admin").unwrap();
    assert_eq!(fresh.version_count, 1);
    }
    
    #[test]
    fn promote_label_moves_label_and_records_audit_row() {
    let (_temp, store) = temp_store();
    let prompt = store.create_prompt("greeting", "v1", "admin").unwrap();
    store
    .create_version(&prompt.id, None, "v2", "second", "admin")
    .unwrap();
    
    store.promote_label(&prompt.id, "staging", 2, "op").unwrap();
    
    let conn = store.lock().unwrap();
    let label_version: i64 = conn
    .query_row(
    "SELECT version FROM prompt_labels WHERE prompt_id = ?1 AND label = 'staging'",
    params![prompt.id],
    |row| row.get(0),
    )
    .unwrap();
    assert_eq!(label_version, 2);
    let (from_version, to_version, actor): (Option<i64>, i64, String) = conn
    .query_row(
    "SELECT from_version, to_version, actor FROM prompt_promotions
    WHERE prompt_id = ?1 AND label = 'staging'",
    params![prompt.id],
    |row| Ok((row.get(0)?, row.get(1)?, row.get(2)?)),
    )
    .unwrap();
    assert_eq!(from_version, None);
    assert_eq!(to_version, 2);
    assert_eq!(actor, "op");
    drop(conn);
    
        // Moving an existing label records the previous version.
        store.promote_label(&prompt.id, "production", 2, "op").unwrap();
        let conn = store.lock().unwrap();
        let (from_version, to_version): (Option<i64>, i64) = conn
            .query_row(
                "SELECT from_version, to_version FROM prompt_promotions
                WHERE prompt_id = ?1 AND label = 'production' AND to_version = 2",
                params![prompt.id],
                |row| Ok((row.get(0)?, row.get(1)?)),
            )
            .unwrap();
    assert_eq!(from_version, Some(1));
    assert_eq!(to_version, 2);
    drop(conn);
    let labels = store.list_labels(&prompt.id).unwrap();
    assert_eq!(
    labels
    .iter()
    .find(|entry| entry.label == "production")
    .unwrap()
    .version,
    Some(2)
    );
    }
    
    #[test]
    fn promote_label_rejects_latest_unknown_label_version_and_prompt() {
    let (_temp, store) = temp_store();
    let prompt = store.create_prompt("greeting", "v1", "admin").unwrap();
    
    let latest = store
    .promote_label(&prompt.id, "latest", 1, "admin")
    .unwrap_err();
    assert!(
    matches!(latest, StoreError::InvalidRequest(ref message) if message.contains("label"))
    );
    
    let unknown_label = store
    .promote_label(&prompt.id, "canary", 1, "admin")
    .unwrap_err();
    assert!(
    matches!(unknown_label, StoreError::InvalidRequest(ref message) if message.contains("label"))
    );
    
    let unknown_version = store
    .promote_label(&prompt.id, "staging", 99, "admin")
    .unwrap_err();
    assert!(matches!(unknown_version, StoreError::NotFound { .. }));
    
    let unknown_prompt = store
    .promote_label("00000000-0000-0000-0000-0000000000ff", "staging", 1, "admin")
    .unwrap_err();
    assert!(matches!(unknown_prompt, StoreError::NotFound { .. }));
    
    // Failed promotions stored nothing.
    let conn = store.lock().unwrap();
    let staging: i64 = conn
    .query_row(
    "SELECT COUNT(*) FROM prompt_labels WHERE prompt_id = ?1 AND label = 'staging'",
    params![prompt.id],
    |row| row.get(0),
    )
    .unwrap();
    assert_eq!(staging, 0);
    }
    
    #[test]
    fn list_promotions_returns_newest_first() {
    let (_temp, store) = temp_store();
    let prompt = store.create_prompt("greeting", "v1", "admin").unwrap();
    store
    .create_version(&prompt.id, None, "v2", "second", "admin")
    .unwrap();
    
        store.promote_label(&prompt.id, "staging", 2, "a").unwrap();
        store.promote_label(&prompt.id, "production", 2, "b").unwrap();
        // Force distinct ordering timestamps (epoch seconds can tie in tests).
        {
            let conn = store.lock().unwrap();
            conn.execute(
                "UPDATE prompt_promotions SET created_at = 300
                WHERE prompt_id = ?1 AND label = 'production' AND to_version = 2",
                params![prompt.id],
            )
            .unwrap();
            conn.execute(
                "UPDATE prompt_promotions SET created_at = 200
                WHERE prompt_id = ?1 AND label = 'staging'",
                params![prompt.id],
            )
            .unwrap();
            conn.execute(
                "UPDATE prompt_promotions SET created_at = 100
                WHERE prompt_id = ?1 AND from_version IS NULL AND to_version = 1",
                params![prompt.id],
            )
            .unwrap();
        }
    
    let page = store.list_promotions(&prompt.id).unwrap();
    let envelope = page.into_openai_response();
    assert_eq!(envelope["object"], "list");
    let data = envelope["data"].as_array().unwrap();
    assert_eq!(data.len(), 3);
    // Newest first: production -> 2, staging -> 2, initial production -> 1.
    assert_eq!(data[0]["label"], "production");
    assert_eq!(data[0]["from_version"], 1);
    assert_eq!(data[0]["to_version"], 2);
    assert_eq!(data[1]["label"], "staging");
    assert_eq!(data[1]["from_version"], Value::Null);
    assert_eq!(data[2]["label"], "production");
    assert_eq!(data[2]["from_version"], Value::Null);
    assert_eq!(data[2]["to_version"], 1);
    // Envelope ids point at the newest and oldest rows.
    assert_eq!(envelope["first_id"], data[0]["id"]);
    assert_eq!(envelope["last_id"], data[2]["id"]);
    assert_eq!(envelope["has_more"], false);
    
    assert!(matches!(
    store.list_promotions("00000000-0000-0000-0000-0000000000ff"),
    Err(StoreError::NotFound { .. })
    ));
    }
    
    #[test]
    fn list_labels_marks_virtual_latest_without_storing_it() {
    let (_temp, store) = temp_store();
    let prompt = store.create_prompt("greeting", "v1", "admin").unwrap();
    store
    .create_version(&prompt.id, None, "v2", "second", "admin")
    .unwrap();
    
    let labels = store.list_labels(&prompt.id).unwrap();
    assert_eq!(labels.len(), 4);
    let by_label = |label: &str| {
    labels
    .iter()
    .find(|entry| entry.label == label)
    .unwrap_or_else(|| panic!("missing label {label}"))
    };
    assert_eq!(by_label("production").version, Some(1));
    assert_eq!(by_label("staging").version, None);
    assert_eq!(by_label("development").version, None);
    assert!(!by_label("production").is_virtual);
    let latest = by_label("latest");
    assert_eq!(latest.version, Some(2));
    assert!(latest.is_virtual);
    
    // The virtual label is never stored.
    let conn = store.lock().unwrap();
    let latest_rows: i64 = conn
    .query_row(
    "SELECT COUNT(*) FROM prompt_labels WHERE prompt_id = ?1 AND label = 'latest'",
    params![prompt.id],
    |row| row.get(0),
    )
    .unwrap();
    assert_eq!(latest_rows, 0);
    drop(conn);
    
    assert!(matches!(
    store.list_labels("00000000-0000-0000-0000-0000000000ff"),
    Err(StoreError::NotFound { .. })
    ));
    }
    
    #[test]
    fn diff_versions_reports_added_removed_and_unchanged_lines() {
    let (_temp, store) = temp_store();
    let prompt = store
    .create_prompt(
    "greeting",
    "line one\nline two\nline three\nline four",
    "admin",
    )
    .unwrap();
    store
.create_version(
&prompt.id,
None,
"line one\nline two changed\nline three\nline four\nline five",
"second",
"admin",
)
    .unwrap();
    
    let diff = store.diff_versions(&prompt.id, 1, 2).unwrap();
    let summary: Vec<(&str, &str)> =
    diff.iter().map(|entry| (kind_of(entry), entry.text())).collect();
    assert_eq!(
    summary,
    vec![
    ("unchanged", "line one"),
    ("removed", "line two"),
    ("added", "line two changed"),
    ("unchanged", "line three"),
    ("unchanged", "line four"),
    ("added", "line five"),
    ]
    );
    }
    
    #[test]
    fn diff_versions_falls_back_to_whole_block_beyond_line_cap() {
        let (_temp, store) = temp_store();
        // No trailing newline: exactly MAX_DIFF_LINES + 1 lines per side.
        let prompt = store
            .create_prompt("big", &format!("{}a", "a\n".repeat(MAX_DIFF_LINES)), "admin")
            .unwrap();
        store
            .create_version(&prompt.id, None, &format!("{}b", "b\n".repeat(MAX_DIFF_LINES)), "second", "admin")
            .unwrap();

        let diff = store.diff_versions(&prompt.id, 1, 2).unwrap();
        assert_eq!(diff.len(), 2 * (MAX_DIFF_LINES + 1));
        assert!(diff.iter().all(|entry| {
            matches!(entry, DiffLine::Removed { .. }) || matches!(entry, DiffLine::Added { .. })
        }));
    }
    
    #[test]
    fn diff_versions_rejects_equal_and_unknown_versions() {
    let (_temp, store) = temp_store();
    let prompt = store.create_prompt("greeting", "v1", "admin").unwrap();
    
    let equal = store.diff_versions(&prompt.id, 1, 1).unwrap_err();
    assert!(
    matches!(equal, StoreError::InvalidRequest(ref message) if message.contains("version"))
    );
    
    let unknown = store.diff_versions(&prompt.id, 1, 9).unwrap_err();
    assert!(matches!(unknown, StoreError::NotFound { .. }));
    
    let unknown_prompt =
    store.diff_versions("00000000-0000-0000-0000-0000000000ff", 1, 2).unwrap_err();
    assert!(matches!(unknown_prompt, StoreError::NotFound { .. }));
    }
    
    #[test]
fn diff_versions_strips_one_trailing_newline_before_splitting() {
    let (_temp, store) = temp_store();
    let prompt = store
        .create_prompt("trailed", "line one\nline two\n", "admin")
        .unwrap();
    store
        .create_version(&prompt.id, None, "line one\nline two changed\n", "second", "admin")
        .unwrap();

    let diff = store.diff_versions(&prompt.id, 1, 2).unwrap();
    let summary: Vec<(&str, &str)> =
        diff.iter().map(|entry| (kind_of(entry), entry.text())).collect();
    // One trailing '\n' is a terminator, not a line: no phantom empty
    // removed/added entries appear after the last real line.
    assert_eq!(
        summary,
        vec![
            ("unchanged", "line one"),
            ("removed", "line two"),
            ("added", "line two changed"),
        ]
    );
}

#[test]
fn diff_lines_lcs_behavior_unchanged_with_u32_cells() {
    // Exercises the u32-cell LCS table end to end: mixed inserts,
    // deletes, and common lines produce the same sequence as before.
    let (_temp, store) = temp_store();
    let prompt = store
        .create_prompt(
            "mixed",
            "alpha\nbeta\ngamma\ndelta\nepsilon",
            "admin",
        )
        .unwrap();
    store
.create_version(
&prompt.id,
None,
"alpha\ngamma\ndelta\nzeta\nepsilon",
"second",
"admin",
)
        .unwrap();

    let diff = store.diff_versions(&prompt.id, 1, 2).unwrap();
    let summary: Vec<(&str, &str)> =
        diff.iter().map(|entry| (kind_of(entry), entry.text())).collect();
    assert_eq!(
        summary,
        vec![
            ("unchanged", "alpha"),
            ("removed", "beta"),
            ("unchanged", "gamma"),
            ("unchanged", "delta"),
            ("added", "zeta"),
            ("unchanged", "epsilon"),
        ]
    );
}

fn kind_of(entry: &DiffLine) -> &'static str {
    match entry {
    DiffLine::Added { .. } => "added",
    DiffLine::Removed { .. } => "removed",
    DiffLine::Unchanged { .. } => "unchanged",
    }
    }
    }
