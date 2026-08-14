//! SQLite-backed persistence for OpenAI-compatible assistants, threads, and messages.

use std::path::{Path, PathBuf};
use std::sync::{Arc, Mutex, MutexGuard};
use std::time::Duration;

use crate::models::openai::{Message, OpenAIRequest, OpenAIResponse};
use chrono::Utc;
use rusqlite::{params, Connection, OptionalExtension, Transaction};
use serde::Deserialize;
use serde_json::{json, Map, Value};
use uuid::Uuid;

const MAX_ASSISTANT_BYTES: usize = 256 * 1024;
const MAX_THREAD_BYTES: usize = 256 * 1024;
const MAX_MESSAGE_BYTES: usize = 1024 * 1024;
const MAX_RUN_BYTES: usize = 256 * 1024;
const MAX_RUN_STEP_BYTES: usize = 256 * 1024;
pub const MAX_FILE_BYTES: usize = 4 * 1024 * 1024;
const DEFAULT_LIST_LIMIT: usize = 20;
const MAX_LIST_LIMIT: usize = 100;
const MAX_RUN_CONTEXT_MESSAGES: usize = 100;
const MAX_THREADS_PER_OWNER: usize = 1_000;
const MAX_MESSAGES_PER_THREAD: usize = 10_000;
const MAX_FILES_PER_OWNER: usize = 1_000;
const MAX_FILE_STORAGE_PER_OWNER: usize = 256 * 1024 * 1024;

#[derive(Debug, thiserror::Error)]
pub enum StoreError {
    #[error("database operation failed")]
    Database(#[from] rusqlite::Error),
    #[error("stored record is invalid")]
    Serialization(#[from] serde_json::Error),
    #[error("failed to prepare storage directory")]
    Io(#[from] std::io::Error),
    #[error("assistants store lock is unavailable")]
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

#[derive(Debug, Clone)]
pub struct RunExecution {
    pub run: Value,
    pub request: OpenAIRequest,
}

#[derive(Debug, Clone)]
pub struct StoredFileContent {
    pub filename: String,
    pub content: Vec<u8>,
}

impl ListPage {
    pub fn into_openai_response(self) -> Value {
        let first_id = self.data.first().and_then(|item| item.get("id")).cloned();
        let last_id = self.data.last().and_then(|item| item.get("id")).cloned();
        json!({
            "object": "list",
            "data": self.data,
            "first_id": first_id,
            "last_id": last_id,
            "has_more": self.has_more
        })
    }
}

#[derive(Clone)]
pub struct AssistantsStore {
    conn: Arc<Mutex<Connection>>,
}

impl AssistantsStore {
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
            .join("assistants.db")
    }

    fn create_schema(conn: &Connection) -> Result<(), StoreError> {
        conn.execute_batch(
            "CREATE TABLE IF NOT EXISTS assistants (
                owner_id TEXT NOT NULL,
                id TEXT NOT NULL,
                created_at INTEGER NOT NULL,
                payload TEXT NOT NULL,
                PRIMARY KEY (owner_id, id)
            );
            CREATE INDEX IF NOT EXISTS idx_assistants_owner_created
                ON assistants(owner_id, created_at, id);

            CREATE TABLE IF NOT EXISTS assistant_threads (
                owner_id TEXT NOT NULL,
                id TEXT NOT NULL,
                created_at INTEGER NOT NULL,
                payload TEXT NOT NULL,
                PRIMARY KEY (owner_id, id)
            );
            CREATE INDEX IF NOT EXISTS idx_threads_owner_created
                ON assistant_threads(owner_id, created_at, id);

            CREATE TABLE IF NOT EXISTS assistant_messages (
                owner_id TEXT NOT NULL,
                id TEXT NOT NULL,
                thread_id TEXT NOT NULL,
                created_at INTEGER NOT NULL,
                payload TEXT NOT NULL,
                PRIMARY KEY (owner_id, id),
                FOREIGN KEY (owner_id, thread_id)
                    REFERENCES assistant_threads(owner_id, id) ON DELETE CASCADE
            );
CREATE INDEX IF NOT EXISTS idx_messages_owner_thread_created
ON assistant_messages(owner_id, thread_id, created_at, id);

CREATE TABLE IF NOT EXISTS assistant_runs (
owner_id TEXT NOT NULL,
id TEXT NOT NULL,
thread_id TEXT NOT NULL,
created_at INTEGER NOT NULL,
payload TEXT NOT NULL,
PRIMARY KEY (owner_id, id),
FOREIGN KEY (owner_id, thread_id)
REFERENCES assistant_threads(owner_id, id) ON DELETE CASCADE
);
CREATE INDEX IF NOT EXISTS idx_runs_owner_thread_created
ON assistant_runs(owner_id, thread_id, created_at, id);

CREATE TABLE IF NOT EXISTS assistant_run_steps (
owner_id TEXT NOT NULL,
id TEXT NOT NULL,
thread_id TEXT NOT NULL,
run_id TEXT NOT NULL,
created_at INTEGER NOT NULL,
payload TEXT NOT NULL,
PRIMARY KEY (owner_id, id),
FOREIGN KEY (owner_id, thread_id)
REFERENCES assistant_threads(owner_id, id) ON DELETE CASCADE,
FOREIGN KEY (owner_id, run_id)
REFERENCES assistant_runs(owner_id, id) ON DELETE CASCADE
);
CREATE INDEX IF NOT EXISTS idx_run_steps_owner_run_created
ON assistant_run_steps(owner_id, run_id, created_at, id);

CREATE TABLE IF NOT EXISTS assistant_files (
owner_id TEXT NOT NULL,
id TEXT NOT NULL,
created_at INTEGER NOT NULL,
filename TEXT NOT NULL,
content BLOB NOT NULL,
payload TEXT NOT NULL,
PRIMARY KEY (owner_id, id)
);
CREATE INDEX IF NOT EXISTS idx_files_owner_created
ON assistant_files(owner_id, created_at, id);",
        )?;
        Ok(())
    }

    pub fn create_assistant(&self, owner: &str, input: Value) -> Result<Value, StoreError> {
        let mut fields = object_input(input, "assistant")?;
        strip_protected(&mut fields);
        validate_assistant_model(&fields)?;

        let id = prefixed_id("asst");
        let created_at = Utc::now().timestamp();
        fields.insert("id".into(), Value::String(id.clone()));
        fields.insert("object".into(), Value::String("assistant".into()));
        fields.insert("created_at".into(), Value::from(created_at));
        fields.entry("name").or_insert(Value::Null);
        fields.entry("description").or_insert(Value::Null);
        fields.entry("instructions").or_insert(Value::Null);
        fields.entry("tools").or_insert_with(|| json!([]));
        fields.entry("metadata").or_insert_with(|| json!({}));
        let payload = Value::Object(fields);
        let encoded = encode_bounded(&payload, "assistant", MAX_ASSISTANT_BYTES)?;

        self.lock()?.execute(
            "INSERT INTO assistants (owner_id, id, created_at, payload) VALUES (?1, ?2, ?3, ?4)",
            params![owner, id, created_at, encoded],
        )?;
        Ok(payload)
    }

    pub fn list_assistants(&self, owner: &str, list: &ListParams) -> Result<ListPage, StoreError> {
        self.list_records("assistants", owner, None, list)
    }

    pub fn get_assistant(&self, owner: &str, id: &str) -> Result<Value, StoreError> {
        self.get_record("assistants", "assistant", owner, id)
    }

    pub fn modify_assistant(
        &self,
        owner: &str,
        id: &str,
        input: Value,
    ) -> Result<Value, StoreError> {
        let updates = object_input(input, "assistant")?;
        self.modify_record(
            "assistants",
            "assistant",
            owner,
            id,
            updates,
            MAX_ASSISTANT_BYTES,
            validate_assistant_model,
        )
    }

    pub fn delete_assistant(&self, owner: &str, id: &str) -> Result<Value, StoreError> {
        self.delete_record("assistants", "assistant", owner, id)
    }

    pub fn create_thread(&self, owner: &str, input: Value) -> Result<Value, StoreError> {
        let mut fields = object_input(input, "thread")?;
        strip_protected(&mut fields);
        let initial_messages = match fields.remove("messages") {
            None => Vec::new(),
            Some(Value::Array(messages)) => messages,
            Some(_) => {
                return Err(StoreError::InvalidRequest(
                    "messages must be an array".into(),
                ))
            }
        };

        let id = prefixed_id("thread");
        let created_at = Utc::now().timestamp();
        fields.insert("id".into(), Value::String(id.clone()));
        fields.insert("object".into(), Value::String("thread".into()));
        fields.insert("created_at".into(), Value::from(created_at));
        fields.entry("metadata").or_insert_with(|| json!({}));
        fields.entry("tool_resources").or_insert_with(|| json!({}));
        let payload = Value::Object(fields);
        let encoded = encode_bounded(&payload, "thread", MAX_THREAD_BYTES)?;

        let mut conn = self.lock()?;
        let thread_count: i64 = conn.query_row(
            "SELECT COUNT(*) FROM assistant_threads WHERE owner_id = ?1",
            params![owner],
            |row| row.get(0),
        )?;
        if thread_count >= MAX_THREADS_PER_OWNER as i64 {
            return Err(StoreError::InvalidRequest(format!(
                "thread quota exceeded: maximum {} threads per owner",
                MAX_THREADS_PER_OWNER
            )));
        }
        let transaction = conn.transaction()?;
        transaction.execute(
            "INSERT INTO assistant_threads (owner_id, id, created_at, payload) VALUES (?1, ?2, ?3, ?4)",
            params![owner, id, created_at, encoded],
        )?;
        for message in initial_messages {
            Self::insert_message(&transaction, owner, &id, message)?;
        }
        transaction.commit()?;
        Ok(payload)
    }

    pub fn list_threads(&self, owner: &str, list: &ListParams) -> Result<ListPage, StoreError> {
        self.list_records("assistant_threads", owner, None, list)
    }

    pub fn get_thread(&self, owner: &str, id: &str) -> Result<Value, StoreError> {
        self.get_record("assistant_threads", "thread", owner, id)
    }

    pub fn modify_thread(&self, owner: &str, id: &str, input: Value) -> Result<Value, StoreError> {
        let mut updates = object_input(input, "thread")?;
        updates.remove("messages");
        self.modify_record(
            "assistant_threads",
            "thread",
            owner,
            id,
            updates,
            MAX_THREAD_BYTES,
            |_| Ok(()),
        )
    }

    pub fn delete_thread(&self, owner: &str, id: &str) -> Result<Value, StoreError> {
        self.delete_record("assistant_threads", "thread", owner, id)
    }

    pub fn create_message(
        &self,
        owner: &str,
        thread_id: &str,
        input: Value,
    ) -> Result<Value, StoreError> {
        let mut conn = self.lock()?;
        let transaction = conn.transaction()?;
        ensure_thread_exists(&transaction, owner, thread_id)?;
        let msg_count: i64 = transaction.query_row(
            "SELECT COUNT(*) FROM assistant_messages WHERE owner_id = ?1 AND thread_id = ?2",
            params![owner, thread_id],
            |row| row.get(0),
        )?;
        if msg_count >= MAX_MESSAGES_PER_THREAD as i64 {
            return Err(StoreError::InvalidRequest(format!(
                "message quota exceeded: maximum {} messages per thread",
                MAX_MESSAGES_PER_THREAD
            )));
        }
        let message = Self::insert_message(&transaction, owner, thread_id, input)?;
        transaction.commit()?;
        Ok(message)
    }

    pub fn list_messages(
        &self,
        owner: &str,
        thread_id: &str,
        list: &ListParams,
    ) -> Result<ListPage, StoreError> {
        let conn = self.lock()?;
        ensure_thread_exists(&conn, owner, thread_id)?;
        drop(conn);
        self.list_records("assistant_messages", owner, Some(thread_id), list)
    }

    fn messages_for_run(&self, owner: &str, thread_id: &str) -> Result<Vec<Value>, StoreError> {
        let conn = self.lock()?;
        ensure_thread_exists(&conn, owner, thread_id)?;
        let mut statement = conn.prepare(
            "SELECT payload FROM assistant_messages
             WHERE owner_id = ?1 AND thread_id = ?2
             ORDER BY created_at ASC, rowid ASC
             LIMIT ?3",
        )?;
        let records = statement
            .query_map(params![owner, thread_id, MAX_RUN_CONTEXT_MESSAGES as i64], |row| row.get::<_, String>(0))?
            .collect::<Result<Vec<_>, _>>()?;
        records
            .into_iter()
            .map(|encoded| serde_json::from_str(&encoded).map_err(StoreError::from))
            .collect()
    }

    pub fn get_message(
        &self,
        owner: &str,
        thread_id: &str,
        message_id: &str,
    ) -> Result<Value, StoreError> {
        let conn = self.lock()?;
        let payload = conn
            .query_row(
                "SELECT payload FROM assistant_messages
                 WHERE owner_id = ?1 AND thread_id = ?2 AND id = ?3",
                params![owner, thread_id, message_id],
                |row| row.get::<_, String>(0),
            )
            .optional()?
            .ok_or_else(|| StoreError::NotFound {
                object: "message",
                id: message_id.to_string(),
            })?;
        Ok(serde_json::from_str(&payload)?)
    }

    pub fn modify_message(
        &self,
        owner: &str,
        thread_id: &str,
        message_id: &str,
        input: Value,
    ) -> Result<Value, StoreError> {
        let updates = object_input(input, "message")?;
        let mut current = self.get_message(owner, thread_id, message_id)?;
        merge_updates(&mut current, updates);
        let encoded = encode_bounded(&current, "message", MAX_MESSAGE_BYTES)?;
        let changed = self.lock()?.execute(
            "UPDATE assistant_messages SET payload = ?1
             WHERE owner_id = ?2 AND thread_id = ?3 AND id = ?4",
            params![encoded, owner, thread_id, message_id],
        )?;
        if changed == 0 {
            return Err(StoreError::NotFound {
                object: "message",
                id: message_id.to_string(),
            });
        }
        Ok(current)
    }

    pub fn delete_message(
        &self,
        owner: &str,
        thread_id: &str,
        message_id: &str,
    ) -> Result<Value, StoreError> {
        let changed = self.lock()?.execute(
            "DELETE FROM assistant_messages
WHERE owner_id = ?1 AND thread_id = ?2 AND id = ?3",
            params![owner, thread_id, message_id],
        )?;
        if changed == 0 {
            return Err(StoreError::NotFound {
                object: "message",
                id: message_id.to_string(),
            });
        }
        Ok(json!({
        "id": message_id,
        "object": "thread.message.deleted",
        "deleted": true
        }))
    }

    pub fn start_run(
        &self,
        owner: &str,
        thread_id: &str,
        input: Value,
    ) -> Result<RunExecution, StoreError> {
        let mut fields = object_input(input, "run")?;
        strip_protected(&mut fields);
        let assistant_id = fields
            .get("assistant_id")
            .and_then(Value::as_str)
            .filter(|id| !id.is_empty())
            .ok_or_else(|| StoreError::InvalidRequest("assistant_id is required".into()))?
            .to_string();
        let assistant = self.get_assistant(owner, &assistant_id)?;
        let model = fields
            .get("model")
            .and_then(Value::as_str)
            .or_else(|| assistant.get("model").and_then(Value::as_str))
            .filter(|model| !model.is_empty())
            .ok_or_else(|| StoreError::InvalidRequest("assistant model is required".into()))?
            .to_string();
        let messages_page = self.messages_for_run(owner, thread_id)?;
        let mut messages = Vec::new();
        let instructions = fields
            .get("instructions")
            .and_then(Value::as_str)
            .or_else(|| assistant.get("instructions").and_then(Value::as_str))
            .filter(|instructions| !instructions.is_empty());
        let additional_instructions = fields
            .get("additional_instructions")
            .and_then(Value::as_str)
            .filter(|instructions| !instructions.is_empty());
        if instructions.is_some() || additional_instructions.is_some() {
            messages.push(Message {
                role: "system".into(),
                content: Value::String(
                    [instructions, additional_instructions]
                        .into_iter()
                        .flatten()
                        .collect::<Vec<_>>()
                        .join("\n\n"),
                ),
                extra: Map::new(),
            });
        }
        for message in messages_page {
            messages.push(Message {
                role: message
                    .get("role")
                    .and_then(Value::as_str)
                    .unwrap_or("user")
                    .to_string(),
                content: message_content_as_chat(message.get("content"))?,
                extra: Map::new(),
            });
        }
        if messages.is_empty() {
            return Err(StoreError::InvalidRequest(
                "thread must contain at least one message".into(),
            ));
        }

        let tools = fields
            .get("tools")
            .or_else(|| assistant.get("tools"))
            .cloned()
            .unwrap_or_else(|| json!([]));
        let tools = tools
            .as_array()
            .ok_or_else(|| StoreError::InvalidRequest("tools must be an array".into()))?;
        if tools
            .iter()
            .any(|tool| tool.get("type").and_then(Value::as_str) != Some("function"))
        {
            return Err(StoreError::InvalidRequest(
                "only function tools are supported for runs".into(),
            ));
        }
        if fields.contains_key("additional_messages") {
            return Err(StoreError::InvalidRequest(
                "additional_messages are not supported; add messages to the thread before starting a run"
                    .into(),
            ));
        }
        if fields.contains_key("truncation_strategy") {
            return Err(StoreError::InvalidRequest(
                "truncation_strategy is not supported".into(),
            ));
        }

        let mut extra = Map::new();
        if !tools.is_empty() {
            extra.insert("tools".into(), Value::Array(tools.clone()));
        }
        for option in [
            "tool_choice",
            "parallel_tool_calls",
            "response_format",
            "top_p",
            "frequency_penalty",
            "presence_penalty",
            "stop",
            "seed",
            "reasoning_effort",
        ] {
            if let Some(value) = fields.get(option) {
                extra.insert(option.into(), value.clone());
            }
        }
        let temperature = optional_f32(&fields, "temperature")?;
        let max_tokens = optional_u32(&fields, "max_completion_tokens")?;

        let id = prefixed_id("run");
        let created_at = Utc::now().timestamp();
        fields.insert("id".into(), Value::String(id.clone()));
        fields.insert("object".into(), Value::String("thread.run".into()));
        fields.insert("created_at".into(), Value::from(created_at));
        fields.insert("thread_id".into(), Value::String(thread_id.to_string()));
        fields.insert("assistant_id".into(), Value::String(assistant_id));
        fields.insert("model".into(), Value::String(model.clone()));
        fields.insert("status".into(), Value::String("in_progress".into()));
        fields.insert("started_at".into(), Value::from(created_at));
        fields.insert("completed_at".into(), Value::Null);
        fields.insert("cancelled_at".into(), Value::Null);
        fields.insert("failed_at".into(), Value::Null);
        fields.insert("last_error".into(), Value::Null);
        fields.entry("metadata").or_insert_with(|| json!({}));
        let run = Value::Object(fields);
        let encoded = encode_bounded(&run, "run", MAX_RUN_BYTES)?;
        self.lock()?.execute(
            "INSERT INTO assistant_runs (owner_id, id, thread_id, created_at, payload)
VALUES (?1, ?2, ?3, ?4, ?5)",
            params![owner, id, thread_id, created_at, encoded],
        )?;
        Ok(RunExecution {
            run,
            request: OpenAIRequest {
                model,
                messages,
                stream: false,
                temperature,
                max_tokens,
                extra,
            },
        })
    }

    pub fn complete_run(
        &self,
        owner: &str,
        thread_id: &str,
        run_id: &str,
        response: &OpenAIResponse,
    ) -> Result<Value, StoreError> {
        let mut conn = self.lock()?;
        let transaction = conn.transaction()?;
        let mut run = get_run_record(&transaction, owner, thread_id, run_id)?;
        if run.get("status").and_then(Value::as_str) == Some("cancelled") {
            return Ok(run);
        }
        let choice = response.choices.first().ok_or_else(|| {
            StoreError::InvalidRequest("provider returned no completion choice".into())
        })?;
        if choice
            .message
            .extra
            .get("tool_calls")
            .and_then(Value::as_array)
            .is_some_and(|tool_calls| !tool_calls.is_empty())
        {
            return Err(StoreError::InvalidRequest(
                "tool action execution is not supported for assistant runs".into(),
            ));
        }
        let assistant_id = run.get("assistant_id").cloned().unwrap_or(Value::Null);
        let message = Self::insert_message(
            &transaction,
            owner,
            thread_id,
            json!({
            "role": "assistant",
            "content": choice.message.content,
            "assistant_id": assistant_id,
            "run_id": run_id
            }),
        )?;
        let step_id = prefixed_id("step");
        let completed_at = Utc::now().timestamp();
        let step = json!({
        "id": step_id,
        "object": "thread.run.step",
        "created_at": completed_at,
        "assistant_id": run.get("assistant_id").cloned().unwrap_or(Value::Null),
        "thread_id": thread_id,
        "run_id": run_id,
        "type": "message_creation",
        "status": "completed",
        "step_details": {
        "type": "message_creation",
        "message_creation": { "message_id": message["id"] }
        },
        "last_error": Value::Null,
        "expired_at": Value::Null,
        "cancelled_at": Value::Null,
        "failed_at": Value::Null,
        "completed_at": completed_at,
        "metadata": {}
        });
        let encoded_step = encode_bounded(&step, "run step", MAX_RUN_STEP_BYTES)?;
        transaction.execute(
            "INSERT INTO assistant_run_steps
(owner_id, id, thread_id, run_id, created_at, payload)
VALUES (?1, ?2, ?3, ?4, ?5, ?6)",
            params![
                owner,
                step_id,
                thread_id,
                run_id,
                completed_at,
                encoded_step
            ],
        )?;
        if let Some(fields) = run.as_object_mut() {
            fields.insert("status".into(), Value::String("completed".into()));
            fields.insert("completed_at".into(), Value::from(completed_at));
        }
        update_run_record(&transaction, owner, thread_id, run_id, &run)?;
        transaction.commit()?;
        Ok(run)
    }

    pub fn fail_run(
        &self,
        owner: &str,
        thread_id: &str,
        run_id: &str,
        message: &str,
    ) -> Result<Value, StoreError> {
        let mut run = self.get_run(owner, thread_id, run_id)?;
        if let Some(fields) = run.as_object_mut() {
            fields.insert("status".into(), Value::String("failed".into()));
            fields.insert("failed_at".into(), Value::from(Utc::now().timestamp()));
            fields.insert(
                "last_error".into(),
                json!({"code": "server_error", "message": message}),
            );
        }
        let conn = self.lock()?;
        update_run_record(&conn, owner, thread_id, run_id, &run)?;
        Ok(run)
    }

    pub fn list_runs(
        &self,
        owner: &str,
        thread_id: &str,
        list: &ListParams,
    ) -> Result<ListPage, StoreError> {
        let conn = self.lock()?;
        ensure_thread_exists(&conn, owner, thread_id)?;
        drop(conn);
        self.list_records("assistant_runs", owner, Some(thread_id), list)
    }

    pub fn get_run(&self, owner: &str, thread_id: &str, run_id: &str) -> Result<Value, StoreError> {
        let conn = self.lock()?;
        get_run_record(&conn, owner, thread_id, run_id)
    }

    pub fn cancel_run(
        &self,
        owner: &str,
        thread_id: &str,
        run_id: &str,
    ) -> Result<Value, StoreError> {
        let mut run = self.get_run(owner, thread_id, run_id)?;
        if let Some(fields) = run.as_object_mut() {
            if matches!(
                fields.get("status").and_then(Value::as_str),
                Some("queued" | "in_progress")
            ) {
                fields.insert("status".into(), Value::String("cancelled".into()));
                fields.insert("cancelled_at".into(), Value::from(Utc::now().timestamp()));
            }
        }
        let conn = self.lock()?;
        update_run_record(&conn, owner, thread_id, run_id, &run)?;
        Ok(run)
    }

    pub fn list_run_steps(
        &self,
        owner: &str,
        thread_id: &str,
        run_id: &str,
        list: &ListParams,
    ) -> Result<ListPage, StoreError> {
        self.get_run(owner, thread_id, run_id)?;
        self.list_records("assistant_run_steps", owner, Some(run_id), list)
    }

    pub fn create_file(
        &self,
        owner: &str,
        filename: String,
        purpose: String,
        content: Vec<u8>,
    ) -> Result<Value, StoreError> {
        if content.len() > MAX_FILE_BYTES {
            return Err(StoreError::TooLarge {
                object: "file",
                max_bytes: MAX_FILE_BYTES,
            });
        }
        if filename.trim().is_empty() {
            return Err(StoreError::InvalidRequest("file name is required".into()));
        }
        let conn = self.lock()?;
        let file_count: i64 = conn.query_row(
            "SELECT COUNT(*) FROM assistant_files WHERE owner_id = ?1",
            params![owner],
            |row| row.get(0),
        )?;
        if file_count >= MAX_FILES_PER_OWNER as i64 {
            return Err(StoreError::InvalidRequest(format!(
                "file quota exceeded: maximum {} files per owner",
                MAX_FILES_PER_OWNER
            )));
        }
        let total_bytes: i64 = conn.query_row(
            "SELECT COALESCE(SUM(LENGTH(content)), 0) FROM assistant_files WHERE owner_id = ?1",
            params![owner],
            |row| row.get(0),
        )?;
        if total_bytes + content.len() as i64 > MAX_FILE_STORAGE_PER_OWNER as i64 {
            return Err(StoreError::InvalidRequest(format!(
                "file storage quota exceeded: maximum {} bytes per owner",
                MAX_FILE_STORAGE_PER_OWNER
            )));
        }
        drop(conn);
        let id = prefixed_id("file");
        let created_at = Utc::now().timestamp();
        let payload = json!({
        "id": id,
        "object": "file",
        "bytes": content.len(),
        "created_at": created_at,
        "filename": filename,
        "purpose": purpose,
        "status": "processed",
        "status_details": Value::Null
        });
        let encoded = encode_bounded(&payload, "file metadata", MAX_RUN_BYTES)?;
        self.lock()?.execute(
            "INSERT INTO assistant_files (owner_id, id, created_at, filename, content, payload)
VALUES (?1, ?2, ?3, ?4, ?5, ?6)",
            params![owner, id, created_at, filename, content, encoded],
        )?;
        Ok(payload)
    }

    pub fn list_files(&self, owner: &str, list: &ListParams) -> Result<ListPage, StoreError> {
        self.list_records("assistant_files", owner, None, list)
    }

    pub fn get_file(&self, owner: &str, file_id: &str) -> Result<Value, StoreError> {
        self.get_record("assistant_files", "file", owner, file_id)
    }

    pub fn get_file_content(
        &self,
        owner: &str,
        file_id: &str,
    ) -> Result<StoredFileContent, StoreError> {
        self.lock()?
            .query_row(
                "SELECT filename, content FROM assistant_files WHERE owner_id = ?1 AND id = ?2",
                params![owner, file_id],
                |row| {
                    Ok(StoredFileContent {
                        filename: row.get(0)?,
                        content: row.get(1)?,
                    })
                },
            )
            .optional()?
            .ok_or_else(|| StoreError::NotFound {
                object: "file",
                id: file_id.to_string(),
            })
    }

    pub fn delete_file(&self, owner: &str, file_id: &str) -> Result<Value, StoreError> {
        self.delete_record("assistant_files", "file", owner, file_id)
    }

    fn insert_message(
        transaction: &Transaction<'_>,
        owner: &str,
        thread_id: &str,
        input: Value,
    ) -> Result<Value, StoreError> {
        let mut fields = object_input(input, "message")?;
        strip_protected(&mut fields);
        let role = fields
            .get("role")
            .and_then(Value::as_str)
            .filter(|role| matches!(*role, "user" | "assistant"))
            .ok_or_else(|| {
                StoreError::InvalidRequest(
                    "message role must be either 'user' or 'assistant'".into(),
                )
            })?;
        let _ = role;
        let content = fields
            .remove("content")
            .ok_or_else(|| StoreError::InvalidRequest("message content is required".into()))?;
        fields.insert("content".into(), normalize_message_content(content)?);

        let id = prefixed_id("msg");
        let created_at = Utc::now().timestamp();
        fields.insert("id".into(), Value::String(id.clone()));
        fields.insert("object".into(), Value::String("thread.message".into()));
        fields.insert("created_at".into(), Value::from(created_at));
        fields.insert("thread_id".into(), Value::String(thread_id.to_string()));
        fields.entry("status").or_insert_with(|| json!("completed"));
        fields.entry("incomplete_details").or_insert(Value::Null);
        fields.entry("completed_at").or_insert(Value::Null);
        fields.entry("incomplete_at").or_insert(Value::Null);
        fields.entry("assistant_id").or_insert(Value::Null);
        fields.entry("run_id").or_insert(Value::Null);
        fields.entry("attachments").or_insert_with(|| json!([]));
        fields.entry("metadata").or_insert_with(|| json!({}));
        let payload = Value::Object(fields);
        let encoded = encode_bounded(&payload, "message", MAX_MESSAGE_BYTES)?;
        transaction.execute(
            "INSERT INTO assistant_messages
             (owner_id, id, thread_id, created_at, payload)
             VALUES (?1, ?2, ?3, ?4, ?5)",
            params![owner, id, thread_id, created_at, encoded],
        )?;
        Ok(payload)
    }

    fn get_record(
        &self,
        table: &'static str,
        object: &'static str,
        owner: &str,
        id: &str,
    ) -> Result<Value, StoreError> {
        let sql = format!("SELECT payload FROM {table} WHERE owner_id = ?1 AND id = ?2");
        let payload = self
            .lock()?
            .query_row(&sql, params![owner, id], |row| row.get::<_, String>(0))
            .optional()?
            .ok_or_else(|| StoreError::NotFound {
                object,
                id: id.to_string(),
            })?;
        Ok(serde_json::from_str(&payload)?)
    }

    #[allow(clippy::too_many_arguments)]
    fn modify_record(
        &self,
        table: &'static str,
        object: &'static str,
        owner: &str,
        id: &str,
        mut updates: Map<String, Value>,
        max_bytes: usize,
        validate: fn(&Map<String, Value>) -> Result<(), StoreError>,
    ) -> Result<Value, StoreError> {
        strip_protected(&mut updates);
        let mut current = self.get_record(table, object, owner, id)?;
        merge_updates(&mut current, updates);
        let fields = current
            .as_object()
            .ok_or_else(|| StoreError::InvalidRequest(format!("{object} is invalid")))?;
        validate(fields)?;
        let encoded = encode_bounded(&current, object, max_bytes)?;
        let sql = format!("UPDATE {table} SET payload = ?1 WHERE owner_id = ?2 AND id = ?3");
        let changed = self.lock()?.execute(&sql, params![encoded, owner, id])?;
        if changed == 0 {
            return Err(StoreError::NotFound {
                object,
                id: id.to_string(),
            });
        }
        Ok(current)
    }

    fn delete_record(
        &self,
        table: &'static str,
        object: &'static str,
        owner: &str,
        id: &str,
    ) -> Result<Value, StoreError> {
        let sql = format!("DELETE FROM {table} WHERE owner_id = ?1 AND id = ?2");
        let changed = self.lock()?.execute(&sql, params![owner, id])?;
        if changed == 0 {
            return Err(StoreError::NotFound {
                object,
                id: id.to_string(),
            });
        }
        Ok(json!({
            "id": id,
            "object": format!("{object}.deleted"),
            "deleted": true
        }))
    }

    fn list_records(
        &self,
        table: &'static str,
        owner: &str,
        thread_id: Option<&str>,
        list: &ListParams,
    ) -> Result<ListPage, StoreError> {
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
        let mut records = if let Some(thread_id) = thread_id {
            let scope_column = if table == "assistant_run_steps" {
                "run_id"
            } else {
                "thread_id"
            };
            let sql = format!(
                "SELECT payload FROM {table}
WHERE owner_id = ?1 AND {scope_column} = ?2
ORDER BY created_at ASC, id ASC"
            );
            let mut statement = conn.prepare(&sql)?;
            let rows = statement
                .query_map(params![owner, thread_id], |row| row.get::<_, String>(0))?
                .collect::<Result<Vec<_>, _>>()?;
            rows
        } else {
            let sql = format!(
                "SELECT payload FROM {table}
                 WHERE owner_id = ?1 ORDER BY created_at ASC, id ASC"
            );
            let mut statement = conn.prepare(&sql)?;
            let rows = statement
                .query_map(params![owner], |row| row.get::<_, String>(0))?
                .collect::<Result<Vec<_>, _>>()?;
            rows
        };
        drop(conn);

        if matches!(list.order, ListOrder::Desc) {
            records.reverse();
        }
        let mut records = records
            .into_iter()
            .map(|encoded| serde_json::from_str::<Value>(&encoded))
            .collect::<Result<Vec<_>, _>>()?;

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

    fn lock(&self) -> Result<MutexGuard<'_, Connection>, StoreError> {
        self.conn.lock().map_err(|_| StoreError::Lock)
    }
}

fn object_input(input: Value, object: &str) -> Result<Map<String, Value>, StoreError> {
    input
        .as_object()
        .cloned()
        .ok_or_else(|| StoreError::InvalidRequest(format!("{object} body must be a JSON object")))
}

const IMMUTABLE_MESSAGE_FIELDS: &[&str] = &[
    "id",
    "object",
    "created_at",
    "thread_id",
    "role",
    "content",
    "assistant_id",
    "run_id",
    "status",
    "completed_at",
    "incomplete_at",
    "incomplete_details",
];

fn strip_protected(fields: &mut Map<String, Value>) {
    for key in ["id", "object", "created_at", "thread_id"] {
        fields.remove(key);
    }
}

fn merge_updates(current: &mut Value, mut updates: Map<String, Value>) {
    strip_protected(&mut updates);
    for key in IMMUTABLE_MESSAGE_FIELDS {
        updates.remove(*key);
    }
    if let Some(fields) = current.as_object_mut() {
        fields.extend(updates);
    }
}

fn validate_assistant_model(fields: &Map<String, Value>) -> Result<(), StoreError> {
    fields
        .get("model")
        .and_then(Value::as_str)
        .filter(|model| !model.trim().is_empty())
        .map(|_| ())
        .ok_or_else(|| StoreError::InvalidRequest("assistant model is required".into()))
}

fn optional_f32(fields: &Map<String, Value>, key: &str) -> Result<Option<f32>, StoreError> {
    fields
        .get(key)
        .map(|value| {
            value
                .as_f64()
                .map(|number| number as f32)
                .ok_or_else(|| StoreError::InvalidRequest(format!("{key} must be a number")))
        })
        .transpose()
}

fn optional_u32(fields: &Map<String, Value>, key: &str) -> Result<Option<u32>, StoreError> {
    fields
        .get(key)
        .map(|value| {
            value
                .as_u64()
                .and_then(|number| u32::try_from(number).ok())
                .ok_or_else(|| {
                    StoreError::InvalidRequest(format!(
                        "{key} must be a non-negative 32-bit integer"
                    ))
                })
        })
        .transpose()
}

fn encode_bounded(
    payload: &Value,
    object: &'static str,
    max_bytes: usize,
) -> Result<String, StoreError> {
    let encoded = serde_json::to_string(payload)?;
    if encoded.len() > max_bytes {
        return Err(StoreError::TooLarge { object, max_bytes });
    }
    Ok(encoded)
}

fn prefixed_id(prefix: &str) -> String {
    format!("{prefix}_{}", Uuid::new_v4().simple())
}

fn ensure_thread_exists(conn: &Connection, owner: &str, thread_id: &str) -> Result<(), StoreError> {
    let exists = conn.query_row(
        "SELECT EXISTS(
            SELECT 1 FROM assistant_threads WHERE owner_id = ?1 AND id = ?2
        )",
        params![owner, thread_id],
        |row| row.get::<_, bool>(0),
    )?;
    if exists {
        Ok(())
    } else {
        Err(StoreError::NotFound {
            object: "thread",
            id: thread_id.to_string(),
        })
    }
}

fn normalize_message_content(content: Value) -> Result<Value, StoreError> {
    match content {
        Value::String(value) => Ok(json!([{
            "type": "text",
            "text": { "value": value, "annotations": [] }
        }])),
        Value::Array(parts) => parts
            .into_iter()
            .map(|part| match part {
                Value::Object(mut part) => {
                    if part.get("type").and_then(Value::as_str) == Some("text") {
                        if let Some(Value::String(value)) = part.remove("text") {
                            part.insert(
                                "text".into(),
                                json!({ "value": value, "annotations": [] }),
                            );
                        }
                    }
                    Ok(Value::Object(part))
                }
                _ => Err(StoreError::InvalidRequest(
                    "message content parts must be JSON objects".into(),
                )),
            })
            .collect::<Result<Vec<_>, _>>()
            .map(Value::Array),
        _ => Err(StoreError::InvalidRequest(
            "message content must be a string or array".into(),
        )),
    }
}

fn message_content_as_chat(content: Option<&Value>) -> Result<Value, StoreError> {
    let Some(Value::Array(parts)) = content else {
        return Ok(content.cloned().unwrap_or(Value::Null));
    };
    let mut chat_parts: Vec<Value> = Vec::with_capacity(parts.len());
    for part in parts {
        let part_type = part.get("type").and_then(Value::as_str).unwrap_or("");
        match part_type {
            "text" => {
                let text = match part.get("text") {
                    Some(Value::String(text)) => text.as_str(),
                    Some(Value::Object(text)) => {
                        text.get("value").and_then(Value::as_str).unwrap_or("")
                    }
                    _ => "",
                };
                if !text.is_empty() {
                    chat_parts.push(json!({"type": "text", "text": text}));
                }
            }
            "image_file" => {
                let file_id = part
                    .get("image_file")
                    .and_then(|f| f.get("file_id"))
                    .and_then(Value::as_str);
                if let Some(file_id) = file_id {
                    chat_parts.push(json!({
                        "type": "image_url",
                        "image_url": {
                            "url": format!("assistants://{}", file_id)
                        }
                    }));
                }
            }
            "image_url" => {
                let url = part
                    .get("image_url")
                    .and_then(|u| u.get("url"))
                    .and_then(Value::as_str);
                if let Some(url) = url {
                    chat_parts.push(json!({
                        "type": "image_url",
                        "image_url": { "url": url }
                    }));
                }
            }
            "" => {
                return Err(StoreError::InvalidRequest(
                    "message content part is missing 'type'".into(),
                ));
            }
            other => {
                return Err(StoreError::InvalidRequest(format!(
                    "unsupported message content part type '{other}' for runs"
                )));
            }
        }
    }
    if chat_parts.is_empty() {
        return Err(StoreError::InvalidRequest(
            "message has no translatable content parts".into(),
        ));
    }
    if chat_parts.len() == 1 && chat_parts[0].get("type").and_then(Value::as_str) == Some("text") {
        let text = chat_parts
            .remove(0)
            .get("text")
            .and_then(Value::as_str)
            .unwrap_or("")
            .to_string();
        return Ok(Value::String(text));
    }
    Ok(Value::Array(chat_parts))
}

fn get_run_record(
    conn: &Connection,
    owner: &str,
    thread_id: &str,
    run_id: &str,
) -> Result<Value, StoreError> {
    let payload = conn
        .query_row(
            "SELECT payload FROM assistant_runs
WHERE owner_id = ?1 AND thread_id = ?2 AND id = ?3",
            params![owner, thread_id, run_id],
            |row| row.get::<_, String>(0),
        )
        .optional()?
        .ok_or_else(|| StoreError::NotFound {
            object: "run",
            id: run_id.to_string(),
        })?;
    Ok(serde_json::from_str(&payload)?)
}

fn update_run_record(
    conn: &Connection,
    owner: &str,
    thread_id: &str,
    run_id: &str,
    run: &Value,
) -> Result<(), StoreError> {
    let encoded = encode_bounded(run, "run", MAX_RUN_BYTES)?;
    let changed = conn.execute(
        "UPDATE assistant_runs SET payload = ?1
WHERE owner_id = ?2 AND thread_id = ?3 AND id = ?4",
        params![encoded, owner, thread_id, run_id],
    )?;
    if changed == 0 {
        return Err(StoreError::NotFound {
            object: "run",
            id: run_id.to_string(),
        });
    }
    Ok(())
}

fn cursor_index(records: &[Value], cursor: &str) -> Result<usize, StoreError> {
    records
        .iter()
        .position(|record| record.get("id").and_then(Value::as_str) == Some(cursor))
        .ok_or_else(|| StoreError::InvalidRequest(format!("cursor '{cursor}' was not found")))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn persists_records_and_scopes_every_lookup_by_owner() {
        let temp = tempfile::tempdir().unwrap();
        let path = temp.path().join("assistants.db");
        let store = AssistantsStore::new(&path).unwrap();
        let assistant = store
            .create_assistant("owner-a", json!({"model": "gpt-4o", "name": "helper"}))
            .unwrap();
        let assistant_id = assistant["id"].as_str().unwrap().to_string();
        let thread = store
            .create_thread(
                "owner-a",
                json!({"messages": [{"role": "user", "content": "hello"}]}),
            )
            .unwrap();
        let thread_id = thread["id"].as_str().unwrap().to_string();

        assert!(matches!(
            store.get_assistant("owner-b", &assistant_id),
            Err(StoreError::NotFound { .. })
        ));
        assert!(matches!(
            store.list_messages("owner-b", &thread_id, &ListParams::default()),
            Err(StoreError::NotFound { .. })
        ));
        drop(store);

        let reopened = AssistantsStore::new(&path).unwrap();
        assert_eq!(
            reopened.get_assistant("owner-a", &assistant_id).unwrap()["name"],
            "helper"
        );
        assert_eq!(
            reopened
                .list_messages("owner-a", &thread_id, &ListParams::default())
                .unwrap()
                .data
                .len(),
            1
        );
    }

    #[test]
    fn thread_delete_cascades_messages_and_limits_record_size() {
        let temp = tempfile::tempdir().unwrap();
        let store = AssistantsStore::new(&temp.path().join("assistants.db")).unwrap();
        let thread = store.create_thread("owner", json!({})).unwrap();
        let thread_id = thread["id"].as_str().unwrap();
        store
            .create_message(
                "owner",
                thread_id,
                json!({"role": "user", "content": "hello"}),
            )
            .unwrap();
        store.delete_thread("owner", thread_id).unwrap();
        assert!(matches!(
            store.list_messages("owner", thread_id, &ListParams::default()),
            Err(StoreError::NotFound { .. })
        ));

        let oversized = "x".repeat(MAX_ASSISTANT_BYTES);
        assert!(matches!(
            store.create_assistant(
                "owner",
                json!({"model": "gpt-4o", "instructions": oversized})
            ),
            Err(StoreError::TooLarge { .. })
        ));
    }

    #[test]
    fn run_request_forwards_tools_options_and_all_messages() {
        let temp = tempfile::tempdir().unwrap();
        let store = AssistantsStore::new(&temp.path().join("assistants.db")).unwrap();
        let assistant = store
            .create_assistant(
                "owner",
                json!({
                    "model": "gpt-4o",
                    "instructions": "base",
                    "tools": [{
                        "type": "function",
                        "function": {"name": "lookup", "parameters": {"type": "object"}}
                    }]
                }),
            )
            .unwrap();
        let thread = store.create_thread("owner", json!({})).unwrap();
        let thread_id = thread["id"].as_str().unwrap();
        for index in 0..105 {
            store
                .create_message(
                    "owner",
                    thread_id,
                    json!({"role": "user", "content": format!("message-{index}")}),
                )
                .unwrap();
        }

        let execution = store
            .start_run(
                "owner",
                thread_id,
                json!({
                    "assistant_id": assistant["id"],
                    "additional_instructions": "extra",
                    "tool_choice": "auto",
                    "parallel_tool_calls": false,
                    "response_format": {"type": "json_object"},
                    "temperature": 0.25,
                    "max_completion_tokens": 321
                }),
            )
            .unwrap();

        assert_eq!(execution.request.messages.len(), 106);
        assert_eq!(execution.request.messages[0].content, "base\n\nextra");
        assert_eq!(
            execution.request.messages.last().unwrap().content,
            "message-104"
        );
        assert_eq!(execution.request.extra["tools"], assistant["tools"]);
        assert_eq!(execution.request.extra["tool_choice"], "auto");
        assert_eq!(execution.request.extra["parallel_tool_calls"], false);
        assert_eq!(
            execution.request.extra["response_format"]["type"],
            "json_object"
        );
        assert_eq!(execution.request.temperature, Some(0.25));
        assert_eq!(execution.request.max_tokens, Some(321));
    }

    #[test]
    fn list_pagination_uses_openai_cursor_shape() {
        let temp = tempfile::tempdir().unwrap();
        let store = AssistantsStore::new(&temp.path().join("assistants.db")).unwrap();
        for index in 0..3 {
            store
                .create_assistant("owner", json!({"model": "gpt-4o", "name": index}))
                .unwrap();
        }
        let first = store
            .list_assistants(
                "owner",
                &ListParams {
                    limit: 2,
                    order: ListOrder::Asc,
                    after: None,
                    before: None,
                },
            )
            .unwrap();
        assert!(first.has_more);
        let cursor = first.data[1]["id"].as_str().unwrap().to_string();
        let second = store
            .list_assistants(
                "owner",
                &ListParams {
                    limit: 2,
                    order: ListOrder::Asc,
                    after: Some(cursor),
                    before: None,
                },
            )
            .unwrap();
        assert_eq!(second.data.len(), 1);
        assert!(!second.has_more);
    }
}
