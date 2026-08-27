//! SQLite-backed persistence for stored Responses API objects.
//!
//! Mirrors the ownership and pagination discipline of [`crate::assistants`]:
//! every row is scoped by `owner_id` with a composite primary key, listing
//! cursors ride the `(created_at, id)` tuple, and inserts prune the oldest
//! rows beyond a per-owner retention cap.

use std::path::{Path, PathBuf};
use std::sync::{Arc, Mutex, MutexGuard};
use std::time::Duration;

use crate::assistants::{ListOrder, StoreError};
use crate::responses::models::{InputItem, OutputItem, ResponseError, ResponsesUsage};
use rusqlite::{params, Connection, OptionalExtension};
use serde::Deserialize;

const DEFAULT_LIST_LIMIT: usize = 20;
const MAX_LIST_LIMIT: usize = 100;
const MAX_RESPONSES_PER_OWNER: usize = 1_000;

/// A response persisted in the store.
#[derive(Debug, Clone)]
pub struct StoredResponse {
    pub id: String,
    pub owner_id: String,
    pub created_at: i64,
    pub model: String,
    pub instructions: Option<String>,
    pub store: bool,
    pub input_items: Vec<InputItem>,
    pub output_items: Vec<OutputItem>,
    pub usage: Option<ResponsesUsage>,
    pub metadata: Option<serde_json::Map<String, serde_json::Value>>,
    pub previous_response_id: Option<String>,
    pub error: Option<ResponseError>,
}

/// Pagination parameters for listing endpoints.
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

/// Raw row shape fetched from the `responses` table; JSON columns are
/// decoded after the query so decode failures map to `StoreError`.
type ResponseRow = (
    String,
    String,
    i64,
    String,
    String,
    Option<String>,
    Option<String>,
    Option<String>,
    Option<String>,
    String,
    Option<String>,
    bool,
);

const RESPONSE_COLUMNS: &str = "owner_id, id, created_at, input_items_json, output_items_json, \
     usage_json, metadata_json, previous_response_id, error_json, model, instructions, store";

#[derive(Clone)]
pub struct ResponsesStore {
    conn: Arc<Mutex<Connection>>,
}

impl ResponsesStore {
    pub fn new(db_path: &Path) -> Result<Self, StoreError> {
        if let Some(parent) = db_path.parent() {
            if !parent.as_os_str().is_empty() {
                std::fs::create_dir_all(parent)?;
            }
        }

        let conn = Connection::open(db_path)?;
        conn.busy_timeout(Duration::from_secs(5))?;
        conn.pragma_update(None, "foreign_keys", "ON")?;
        conn.pragma_update(None, "journal_mode", "WAL")?;
        conn.pragma_update(None, "synchronous", "NORMAL")?;
        Self::create_schema(&conn)?;

        Ok(Self {
            conn: Arc::new(Mutex::new(conn)),
        })
    }

    pub fn sibling_database_path(logging_database_path: &str) -> PathBuf {
        let logging_path = Path::new(logging_database_path);
        logging_path
            .parent()
            .filter(|parent| !parent.as_os_str().is_empty())
            .unwrap_or_else(|| Path::new("."))
            .join("responses.db")
    }

    fn create_schema(conn: &Connection) -> Result<(), StoreError> {
        conn.execute_batch(
            "CREATE TABLE IF NOT EXISTS responses (
                owner_id TEXT NOT NULL,
                id TEXT NOT NULL,
                created_at INTEGER NOT NULL,
                input_items_json TEXT NOT NULL,
                output_items_json TEXT NOT NULL,
                usage_json TEXT,
                metadata_json TEXT,
                previous_response_id TEXT,
                error_json TEXT,
                model TEXT NOT NULL,
                instructions TEXT,
                store BOOLEAN NOT NULL DEFAULT 0,
                PRIMARY KEY (owner_id, id)
            );
            CREATE INDEX IF NOT EXISTS idx_responses_owner_time
                ON responses(owner_id, created_at, id);",
        )?;
        Ok(())
    }

    pub fn store_response(&self, owner: &str, response: &StoredResponse) -> Result<(), StoreError> {
        let input_items_json = serde_json::to_string(&response.input_items)?;
        let output_items_json = serde_json::to_string(&response.output_items)?;
        let usage_json = match &response.usage {
            Some(usage) => Some(serde_json::to_string(usage)?),
            None => None,
        };
        let metadata_json = match &response.metadata {
            Some(metadata) => Some(serde_json::to_string(metadata)?),
            None => None,
        };
        let error_json = match &response.error {
            Some(error) => Some(serde_json::to_string(error)?),
            None => None,
        };

        let mut conn = self.lock()?;
        let transaction = conn.transaction()?;
        transaction.execute(
            "INSERT INTO responses (
                owner_id, id, created_at, input_items_json, output_items_json,
                usage_json, metadata_json, previous_response_id, error_json,
                model, instructions, store
            ) VALUES (?1, ?2, ?3, ?4, ?5, ?6, ?7, ?8, ?9, ?10, ?11, ?12)
            ON CONFLICT(owner_id, id) DO UPDATE SET
                created_at = excluded.created_at,
                input_items_json = excluded.input_items_json,
                output_items_json = excluded.output_items_json,
                usage_json = excluded.usage_json,
                metadata_json = excluded.metadata_json,
                previous_response_id = excluded.previous_response_id,
                error_json = excluded.error_json,
                model = excluded.model,
                instructions = excluded.instructions,
                store = excluded.store",
            params![
                owner,
                response.id,
                response.created_at,
                input_items_json,
                output_items_json,
                usage_json,
                metadata_json,
                response.previous_response_id,
                error_json,
                response.model,
                response.instructions,
                response.store
            ],
        )?;
transaction.execute(
    "DELETE FROM responses WHERE owner_id = ?1 AND id IN (
        SELECT id FROM responses WHERE owner_id = ?1
        ORDER BY created_at DESC, id DESC
        LIMIT -1 OFFSET ?2
    )",
    params![owner, MAX_RESPONSES_PER_OWNER as i64],
)?;
        transaction.commit()?;
        Ok(())
    }

    pub fn get_response(&self, owner: &str, id: &str) -> Result<Option<StoredResponse>, StoreError> {
        let sql = format!("SELECT {RESPONSE_COLUMNS} FROM responses WHERE owner_id = ?1 AND id = ?2");
        let row = self
            .lock()?
            .query_row(&sql, params![owner, id], map_response_row)
            .optional()?;
        row.map(decode_response_row).transpose()
    }

    pub fn delete_response(&self, owner: &str, id: &str) -> Result<bool, StoreError> {
        let changed = self.lock()?.execute(
            "DELETE FROM responses WHERE owner_id = ?1 AND id = ?2",
            params![owner, id],
        )?;
        Ok(changed > 0)
    }

    pub fn list_responses(
        &self,
        owner: &str,
        params: &ListParams,
    ) -> Result<Vec<StoredResponse>, StoreError> {
        validate_list_params(params)?;
        let desc = matches!(params.order, ListOrder::Desc);

        let conn = self.lock()?;
        let mut sql = format!("SELECT {RESPONSE_COLUMNS} FROM responses WHERE owner_id = ?1");
        let mut cursor_values: Option<(String, i64)> = None;
        if let Some(cursor) = params.after.as_deref().or(params.before.as_deref()) {
            let cursor_created_at: i64 = conn
                .query_row(
                    "SELECT created_at FROM responses WHERE owner_id = ?1 AND id = ?2",
                    params![owner, cursor],
                    |row| row.get(0),
                )
                .optional()?
                .ok_or_else(|| {
                    StoreError::InvalidRequest(format!("cursor '{cursor}' was not found"))
                })?;
            // `after` moves beyond the cursor in listing order; `before` stays
            // behind it. Descending listings therefore flip the comparison.
            let goes_after = params.after.is_some();
            let comparison = if goes_after != desc { ">" } else { "<" };
            sql.push_str(&format!(" AND (created_at, id) {comparison} (?2, ?3)"));
            cursor_values = Some((cursor.to_string(), cursor_created_at));
        }
        sql.push_str(if desc {
            " ORDER BY created_at DESC, id DESC"
        } else {
            " ORDER BY created_at ASC, id ASC"
        });
        sql.push_str(&format!(" LIMIT ?{}", if cursor_values.is_some() { 4 } else { 2 }));

let rows: Vec<ResponseRow> = match &cursor_values {
Some((cursor, cursor_created_at)) => {
query_response_rows(&conn, &sql, params![owner, cursor_created_at, cursor, params.limit])?
}
None => query_response_rows(&conn, &sql, params![owner, params.limit])?,
};
rows.into_iter().map(decode_response_row).collect()
}

    pub fn list_input_items(
        &self,
        owner: &str,
        response_id: &str,
        params: &ListParams,
    ) -> Result<Vec<InputItem>, StoreError> {
        validate_list_params(params)?;
        let response = self
            .get_response(owner, response_id)?
            .ok_or_else(|| StoreError::NotFound {
                object: "response",
                id: response_id.to_string(),
            })?;
        let mut items = response.input_items;
        if matches!(params.order, ListOrder::Desc) {
            items.reverse();
        }
        if let Some(after) = &params.after {
            let index = input_item_index(&items, after)?;
            items = items.into_iter().skip(index + 1).collect();
        } else if let Some(before) = &params.before {
            let index = input_item_index(&items, before)?;
            items.truncate(index);
        }
        items.truncate(params.limit);
        Ok(items)
    }

    fn lock(&self) -> Result<MutexGuard<'_, Connection>, StoreError> {
        self.conn.lock().map_err(|_| StoreError::Lock)
    }
}

fn validate_list_params(params: &ListParams) -> Result<(), StoreError> {
    if params.limit == 0 || params.limit > MAX_LIST_LIMIT {
        return Err(StoreError::InvalidRequest(format!(
            "limit must be between 1 and {MAX_LIST_LIMIT}"
        )));
    }
    if params.after.is_some() && params.before.is_some() {
        return Err(StoreError::InvalidRequest(
            "after and before cannot be used together".into(),
        ));
    }
    Ok(())
}

fn query_response_rows(
    conn: &Connection,
    sql: &str,
    bind: impl rusqlite::Params,
) -> Result<Vec<ResponseRow>, StoreError> {
    let mut statement = conn.prepare(sql)?;
    let rows = statement
        .query_map(bind, map_response_row)?
        .collect::<Result<Vec<_>, _>>()?;
    Ok(rows)
}

fn map_response_row(row: &rusqlite::Row<'_>) -> rusqlite::Result<ResponseRow> {
    Ok((
        row.get(0)?,
        row.get(1)?,
        row.get(2)?,
        row.get(3)?,
        row.get(4)?,
        row.get(5)?,
        row.get(6)?,
        row.get(7)?,
        row.get(8)?,
        row.get(9)?,
        row.get(10)?,
        row.get(11)?,
    ))
}

fn decode_response_row(row: ResponseRow) -> Result<StoredResponse, StoreError> {
    let (
        owner_id,
        id,
        created_at,
        input_items_json,
        output_items_json,
        usage_json,
        metadata_json,
        previous_response_id,
        error_json,
        model,
        instructions,
        store,
    ) = row;
    Ok(StoredResponse {
        id,
        owner_id,
        created_at,
        input_items: serde_json::from_str(&input_items_json)?,
        output_items: serde_json::from_str(&output_items_json)?,
        usage: usage_json
            .map(|encoded| serde_json::from_str(&encoded))
            .transpose()?,
        metadata: metadata_json
            .map(|encoded| serde_json::from_str(&encoded))
            .transpose()?,
        previous_response_id,
        error: error_json
            .map(|encoded| serde_json::from_str(&encoded))
            .transpose()?,
        model,
        instructions,
        store,
    })
}

fn input_item_index(items: &[InputItem], cursor: &str) -> Result<usize, StoreError> {
    items
        .iter()
        .position(|item| input_item_id(item) == Some(cursor))
        .ok_or_else(|| StoreError::InvalidRequest(format!("cursor '{cursor}' was not found")))
}

fn input_item_id(item: &InputItem) -> Option<&str> {
    match item {
        InputItem::Easy(message) => message.extra.get("id").and_then(|value| value.as_str()),
        InputItem::Typed(typed) => match typed {
            crate::responses::models::TypedInputItem::Message(message) => {
                message.extra.get("id").and_then(|value| value.as_str())
            }
            crate::responses::models::TypedInputItem::FunctionCall(call) => {
                call.id.as_deref()
            }
            crate::responses::models::TypedInputItem::FunctionCallOutput(output) => {
                output.id.as_deref()
            }
            crate::responses::models::TypedInputItem::Reasoning(reasoning) => {
                reasoning.id.as_deref()
            }
            crate::responses::models::TypedInputItem::ItemReference(reference) => {
                Some(reference.id.as_str())
            }
        },
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    fn make_response(owner: &str, id: &str, created_at: i64) -> StoredResponse {
        StoredResponse {
            id: id.to_string(),
            owner_id: owner.to_string(),
            created_at,
            model: "gpt-4o".to_string(),
            instructions: Some("be terse".to_string()),
            store: true,
            input_items: vec![InputItem::Easy(
                crate::responses::models::EasyInputMessage {
                    content: crate::responses::models::EasyInputContent::Text(format!(
                        "input for {id}"
                    )),
                    role: "user".to_string(),
                    phase: None,
                    extra: serde_json::Map::new(),
                },
            )],
            output_items: Vec::new(),
            usage: Some(ResponsesUsage {
                input_tokens: 10,
                output_tokens: 5,
                total_tokens: 15,
                ..Default::default()
            }),
            metadata: Some(
                [("key".to_string(), json!("value"))]
                    .into_iter()
                    .collect(),
            ),
            previous_response_id: None,
            error: None,
        }
    }

    #[test]
    fn stores_and_retrieves_owner_scoped() {
        let temp = tempfile::tempdir().unwrap();
        let path = temp.path().join("responses.db");
        let store = ResponsesStore::new(&path).unwrap();
        let response = make_response("owner-a", "resp_1", 1_000);
        store.store_response("owner-a", &response).unwrap();

        let fetched = store.get_response("owner-a", "resp_1").unwrap().unwrap();
        assert_eq!(fetched.id, "resp_1");
        assert_eq!(fetched.model, "gpt-4o");
        assert_eq!(fetched.created_at, 1_000);
        assert_eq!(fetched.usage.as_ref().unwrap().total_tokens, 15);
        assert_eq!(fetched.metadata.as_ref().unwrap()["key"], "value");
        assert!(matches!(
            fetched.input_items.first(),
            Some(InputItem::Easy(_))
        ));

        assert!(store.get_response("owner-b", "resp_1").unwrap().is_none());
        assert!(store.get_response("owner-a", "resp_missing").unwrap().is_none());
        drop(store);

        let reopened = ResponsesStore::new(&path).unwrap();
        let persisted = reopened.get_response("owner-a", "resp_1").unwrap().unwrap();
        assert_eq!(persisted.instructions.as_deref(), Some("be terse"));
    }

    #[test]
    fn delete_response_reports_existence() {
        let temp = tempfile::tempdir().unwrap();
        let store = ResponsesStore::new(&temp.path().join("responses.db")).unwrap();
        store
            .store_response("owner-a", &make_response("owner-a", "resp_1", 1_000))
            .unwrap();

        assert!(store.delete_response("owner-b", "resp_1").unwrap() == false);
        assert!(store.delete_response("owner-a", "resp_1").unwrap());
        assert!(store.get_response("owner-a", "resp_1").unwrap().is_none());
        assert!(!store.delete_response("owner-a", "resp_1").unwrap());
    }

    #[test]
    fn list_paginates_with_cursors_and_limit() {
        let temp = tempfile::tempdir().unwrap();
        let store = ResponsesStore::new(&temp.path().join("responses.db")).unwrap();
        for index in 0..5 {
            store
                .store_response(
                    "owner-a",
                    &make_response("owner-a", &format!("resp_{index}"), index as i64),
                )
                .unwrap();
        }

        let page = store
            .list_responses(
                "owner-a",
                &ListParams {
                    limit: 3,
                    order: ListOrder::Asc,
                    after: None,
                    before: None,
                },
            )
            .unwrap();
        assert_eq!(
            page.iter().map(|r| r.id.as_str()).collect::<Vec<_>>(),
            vec!["resp_0", "resp_1", "resp_2"]
        );

        let after = store
            .list_responses(
                "owner-a",
                &ListParams {
                    limit: 10,
                    order: ListOrder::Asc,
                    after: Some("resp_1".to_string()),
                    before: None,
                },
            )
            .unwrap();
        assert_eq!(
            after.iter().map(|r| r.id.as_str()).collect::<Vec<_>>(),
            vec!["resp_2", "resp_3", "resp_4"]
        );

        let before = store
            .list_responses(
                "owner-a",
        &ListParams {
            limit: 10,
            order: ListOrder::Desc,
            after: None,
            before: Some("resp_2".to_string()),
        },
            )
            .unwrap();
        assert_eq!(
            before.iter().map(|r| r.id.as_str()).collect::<Vec<_>>(),
            vec!["resp_4", "resp_3"]
        );

        let desc_after = store
            .list_responses(
                "owner-a",
                &ListParams {
                    limit: 2,
                    order: ListOrder::Desc,
                    after: Some("resp_3".to_string()),
                    before: None,
                },
            )
            .unwrap();
        assert_eq!(
            desc_after.iter().map(|r| r.id.as_str()).collect::<Vec<_>>(),
            vec!["resp_2", "resp_1"]
        );

        assert!(store
            .list_responses(
                "owner-a",
                &ListParams {
                    after: Some("nope".to_string()),
                    ..ListParams::default()
                },
            )
            .is_err());
        assert!(store
            .list_responses(
                "owner-a",
                &ListParams {
                    limit: 0,
                    ..ListParams::default()
                },
            )
            .is_err());
    }

    #[test]
    fn list_enforces_ownership() {
        let temp = tempfile::tempdir().unwrap();
        let store = ResponsesStore::new(&temp.path().join("responses.db")).unwrap();
        store
            .store_response("owner-a", &make_response("owner-a", "resp_1", 1_000))
            .unwrap();
        assert!(store
            .list_responses("owner-b", &ListParams::default())
            .unwrap()
            .is_empty());
        assert!(store
            .list_input_items("owner-b", "resp_1", &ListParams::default())
            .is_err());
    }

    #[test]
    fn retention_cap_prunes_oldest_responses() {
        let temp = tempfile::tempdir().unwrap();
        let store = ResponsesStore::new(&temp.path().join("responses.db")).unwrap();
        for index in 0..(MAX_RESPONSES_PER_OWNER + 5) {
            store
                .store_response(
                    "owner-a",
                    &make_response("owner-a", &format!("resp_{index}"), index as i64),
                )
                .unwrap();
        }

        let all = store
            .list_responses(
                "owner-a",
                &ListParams {
                    limit: MAX_LIST_LIMIT,
                    order: ListOrder::Asc,
                    ..ListParams::default()
                },
            )
            .unwrap();
        let oldest_page = all.first().unwrap();
        assert_eq!(oldest_page.id, "resp_5");
        assert!(store.get_response("owner-a", "resp_0").unwrap().is_none());
        assert!(store.get_response("owner-a", "resp_4").unwrap().is_none());
        assert!(store
            .get_response("owner-a", &format!("resp_{}", MAX_RESPONSES_PER_OWNER + 4))
            .unwrap()
            .is_some());
    }

    #[test]
    fn schema_creation_is_idempotent() {
        let temp = tempfile::tempdir().unwrap();
        let path = temp.path().join("responses.db");
        let first = ResponsesStore::new(&path).unwrap();
        drop(first);
        let second = ResponsesStore::new(&path).unwrap();
        second
            .store_response("owner", &make_response("owner", "resp_1", 1_000))
            .unwrap();
        assert!(second.get_response("owner", "resp_1").unwrap().is_some());
    }

    #[test]
    fn sibling_database_path_derives_responses_db() {
        let with_dir = ResponsesStore::sibling_database_path("data/logs.db");
        assert_eq!(with_dir.file_name().unwrap(), "responses.db");
        assert_eq!(with_dir.parent().unwrap(), std::path::Path::new("data"));

        let bare = ResponsesStore::sibling_database_path("logs.db");
        assert_eq!(bare.file_name().unwrap(), "responses.db");
    }

    #[test]
    fn list_input_items_paginates_by_item_cursor() {
        let temp = tempfile::tempdir().unwrap();
        let store = ResponsesStore::new(&temp.path().join("responses.db")).unwrap();
        let mut response = make_response("owner-a", "resp_1", 1_000);
        response.input_items = vec![
            InputItem::Typed(crate::responses::models::TypedInputItem::FunctionCall(
                crate::responses::models::FunctionCall {
                    id: Some("fc_0".to_string()),
                    call_id: "call_0".to_string(),
                    name: "lookup".to_string(),
                    arguments: "{}".to_string(),
                    status: None,
                    extra: serde_json::Map::new(),
                },
            )),
            InputItem::Easy(crate::responses::models::EasyInputMessage {
                content: crate::responses::models::EasyInputContent::Text("second".to_string()),
                role: "user".to_string(),
                phase: None,
                extra: serde_json::Map::new(),
            }),
            InputItem::Typed(crate::responses::models::TypedInputItem::ItemReference(
                crate::responses::models::ItemReference {
                    id: "msg_9".to_string(),
                    extra: serde_json::Map::new(),
                },
            )),
        ];
        store.store_response("owner-a", &response).unwrap();

        let after = store
            .list_input_items(
                "owner-a",
                "resp_1",
                &ListParams {
                    limit: 10,
                    order: ListOrder::Asc,
                    after: Some("fc_0".to_string()),
                    before: None,
                },
            )
            .unwrap();
        assert_eq!(after.len(), 2);

let before = store
.list_input_items(
"owner-a",
"resp_1",
&ListParams {
limit: 10,
order: ListOrder::Asc,
after: None,
before: Some("msg_9".to_string()),
},
)
.unwrap();
assert_eq!(before.len(), 2);

let desc_after = store
.list_input_items(
"owner-a",
"resp_1",
&ListParams {
limit: 10,
order: ListOrder::Desc,
after: Some("msg_9".to_string()),
before: None,
},
)
.unwrap();
assert_eq!(desc_after.len(), 2);

assert!(store
.list_input_items(
"owner-a",
"resp_1",
&ListParams {
after: Some("ghost".to_string()),
..ListParams::default()
},
)
.is_err());

        let limited = store
            .list_input_items(
                "owner-a",
                "resp_1",
                &ListParams {
                    limit: 1,
                    ..ListParams::default()
                },
            )
            .unwrap();
        assert_eq!(limited.len(), 1);
    }
}
