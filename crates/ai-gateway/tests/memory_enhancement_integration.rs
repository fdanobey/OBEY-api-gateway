//! Integration tests for the `llm-memory-module-enhancement` spec (task 10).
//!
//! Coverage map (spec `llm-memory-module-enhancement`, requirements.md):
//! * Req 1 (edit): admin PATCH `entries/{id}` updates content/type, records
//!   `updated_at`, re-indexes in the fake vector tier, 404s on missing ids,
//!   and 400s on invalid/sensitive content.
//! * Req 2 (supersession): supersede excludes an entry from retrieval (the
//! injector's candidate source) while keeping it visible with
//! `include_superseded`; deleting a superseded entry still works and
//! removes its vector point; the admin DELETE endpoint routes through
//! `admin_delete` and issues the vector point delete; a v3-shaped
//! database migrates to v4 with rows and FTS content preserved.
//! * Req 3 (consolidation): end-to-end merge through a fake extraction
//!   provider (merged entry created, both sources superseded, counts and
//!   last-pass metrics correct, vector deletes/index observed); malformed
//!   provider output skips the batch while serving continues; config
//!   validation names `consolidation_provider` on an unresolvable pair, the
//!   `auto_extract` fallback resolves, and hot-reload applies consolidation
//!   settings.
//! * Req 4/5 (search/export): cross-namespace search hit, empty-`q` falls
//!   back to the filtered listing, export includes superseded entries plus
//!   a total.
//! * Req 8 (per-request memory mode): `X-Obey-Memory-Mode: off` skips
//! injection AND extraction (non-stream and buffered streaming paths);
//! `read-only` injects but skips extraction; an absent or unrecognized
//! header value leaves configured behavior unchanged.
//!
//! Conventions follow `guardrail_e2e.rs`: the full Axum router is driven
//! through `tower::ServiceExt::oneshot()` (no port binding), wiremock plays
//! the single upstream provider, and `common::isolate_databases` keeps
//! parallel tests on separate SQLite files. Memory-system-level tests build
//! `MemorySystem` directly with fake vector tiers and extraction providers,
//! mirroring the seams used by the unit tests in `src/memory/mod.rs`.

use std::sync::{Arc, Mutex};

use axum::body::Body;
use axum::http::{Request, StatusCode};
use tower::ServiceExt;

use ai_gateway::config::Config;
use ai_gateway::gateway::GatewayServer;
use ai_gateway::memory::{
    ConsolidationCounts, MemoryConfig, MemoryEntry, MemoryError, MemoryExtractionProvider,
    MemoryExtractionProviderError, MemoryExtractionProviderRequest, MemorySystem, MemoryType,
    MemoryVectorTier, StructuredMemoryCandidate, VectorMatch,
};

use async_trait::async_trait;
use uuid::Uuid;
use wiremock::matchers::{method as wm_method, path as wm_path};
use wiremock::{Mock, MockServer, ResponseTemplate};

mod common;

// ---------------------------------------------------------------------------
// Fakes
// ---------------------------------------------------------------------------

/// Records `index`/`delete_entry`/`delete_namespace` calls, mirroring the
/// unit-test `RecordingVectorTier` from `src/memory/mod.rs`.
struct RecordingVectorTier {
    events: Mutex<Vec<String>>,
}

impl RecordingVectorTier {
    fn new() -> Self {
        Self {
            events: Mutex::new(Vec::new()),
        }
    }

    fn events(&self) -> Vec<String> {
        self.events.lock().unwrap().clone()
    }

    fn count(&self, event: &str) -> usize {
        self.events()
            .iter()
            .filter(|recorded| recorded.as_str() == event)
            .count()
    }
}

#[async_trait]
impl MemoryVectorTier for RecordingVectorTier {
    async fn index(&self, entry: &MemoryEntry) -> Result<(), MemoryError> {
        self.events
            .lock()
            .unwrap()
            .push(format!("index:{}", entry.id));
        Ok(())
    }

    async fn search(&self, _query: &str, _limit: usize) -> Result<Vec<VectorMatch>, MemoryError> {
        Ok(Vec::new())
    }

    async fn delete_entry(&self, id: Uuid) -> Result<(), MemoryError> {
        self.events.lock().unwrap().push(format!("delete:{id}"));
        Ok(())
    }

    async fn delete_namespace(&self, namespace: &str) -> Result<(), MemoryError> {
        self.events
            .lock()
            .unwrap()
            .push(format!("delete_namespace:{namespace}"));
        Ok(())
    }
}

/// Fake extraction provider whose raw output (joined candidate contents) is
/// settable after fixture entries exist, so consolidation decisions can
/// reference the real entry ids.
struct FakeExtractionProvider {
    response: Mutex<Option<String>>,
}

impl FakeExtractionProvider {
    fn new() -> Self {
        Self {
            response: Mutex::new(None),
        }
    }

    fn set_raw_output(&self, output: String) {
        *self.response.lock().unwrap() = Some(output);
    }
}

#[async_trait]
impl MemoryExtractionProvider for FakeExtractionProvider {
    async fn extract(
        &self,
        _request: MemoryExtractionProviderRequest,
    ) -> Result<Vec<StructuredMemoryCandidate>, MemoryExtractionProviderError> {
        let output = self.response.lock().unwrap().clone().unwrap_or_default();
        Ok(vec![StructuredMemoryCandidate {
            content: output,
            memory_type: MemoryType::Fact,
        }])
    }
}

// ---------------------------------------------------------------------------
// Memory-system fixtures
// ---------------------------------------------------------------------------

/// Near-duplicate contents sharing most tokens so `group_candidates`
/// (similarity >= 0.6) places them in one batch.
const DUP_A: &str = "The deployment target for staging is fly.io in the eu-west region";
const DUP_B: &str = "The deployment target for staging is fly.io in the eu-central region";

fn memory_config(db_path: &str) -> MemoryConfig {
    MemoryConfig {
        enabled: true,
        database_path: db_path.to_owned(),
        consolidation_enabled: true,
        consolidation_provider: "consolidation-provider".to_owned(),
        consolidation_model: "test-model".to_owned(),
        ..MemoryConfig::default()
    }
}

/// Memory system plus recording tier plus the temp directory backing the
/// database. Fields drop in declaration order, so the system's SQLite
/// handle closes before the directory cleanup runs (Windows cannot delete
/// open files) — no temp dir leaks per run.
struct MemoryFixture {
    system: Arc<MemorySystem>,
    tier: Arc<RecordingVectorTier>,
    _dir: tempfile::TempDir,
}

/// Build a `MemoryFixture`: a `MemorySystem` with a recording vector tier
/// and a fake extraction provider, backed by a fresh temp-database path.
async fn memory_system_with(provider: Arc<FakeExtractionProvider>) -> MemoryFixture {
    let dir = tempfile::tempdir().expect("create temp dir for memory database");
    let db_path = dir.path().join("memory.db");
    let system = Arc::new(
        MemorySystem::new_with_vector(
            memory_config(db_path.to_string_lossy().as_ref()),
            None,
            Some(provider),
            None,
        )
        .await
        .expect("memory system should initialize"),
    );
    let tier = Arc::new(RecordingVectorTier::new());
    system
        .set_vector_tier(Some(tier.clone() as Arc<dyn MemoryVectorTier>))
        .await;
    MemoryFixture {
        system,
        tier,
        _dir: dir,
    }
}

async fn create_entry(system: &MemorySystem, namespace: &str, content: &str) -> MemoryEntry {
    system
        .admin_create(namespace.to_owned(), content.to_owned(), MemoryType::Fact)
        .await
        .expect("entry creation should succeed")
}

/// Mirror the src-test convention: poll with `yield_now` inside a timeout
/// until spawned vector tasks have recorded `expected` matching events.
async fn wait_for_events(tier: &RecordingVectorTier, event: &str, expected: usize) {
    tokio::time::timeout(std::time::Duration::from_secs(2), async {
        loop {
            if tier.count(event) >= expected {
                return;
            }
            tokio::task::yield_now().await;
        }
    })
    .await
    .unwrap_or_else(|_| panic!("expected {expected} '{event}' events"));
}

// ---------------------------------------------------------------------------
// Gateway-level fixtures (Req 8)
// ---------------------------------------------------------------------------

/// Start a mock OpenAI-compatible upstream returning a static non-streaming
/// completion (same shape as `guardrail_e2e.rs`).
async fn start_mock_provider() -> MockServer {
    let server = MockServer::start().await;
    let body = serde_json::json!({
        "id": "chatcmpl-memory-mode",
        "object": "chat.completion",
        "created": 1_700_000_000_i64,
        "model": "gpt-4",
        "choices": [{
            "index": 0,
            "message": { "role": "assistant", "content": "Mock assistant reply." },
            "finish_reason": "stop"
        }],
        "usage": { "prompt_tokens": 10, "completion_tokens": 5, "total_tokens": 15 }
    });
    Mock::given(wm_method("POST"))
        .and(wm_path("/v1/chat/completions"))
        .respond_with(ResponseTemplate::new(200).set_body_json(body))
        .mount(&server)
        .await;
    server
}

/// Build a config (via YAML, resilient to unrelated field churn) with one
/// provider at `mock_uri` and a memory section; `memory_yaml` carries the
/// database path (forward slashes keep the YAML double-quoted string valid
/// on Windows).
fn config_with_memory(mock_uri: &str, memory_yaml: &str) -> Config {
    let yaml = format!(
        r#"
server:
  host: "127.0.0.1"
  port: 8080
  request_timeout_seconds: 30
  max_request_size_mb: 10
providers:
- name: "test-provider"
  type: "openai"
  base_url: "{mock_uri}"
  timeout_seconds: 30
model_groups:
- name: "test-group"
  version_fallback_enabled: false
  models:
  - provider: "test-provider"
    model: "gpt-4"
    priority: 100
retry:
  max_retries_per_provider: 0
  backoff_sequence_seconds: [1, 2, 4]
streaming:
  passthrough_enabled: false
memory:
  enabled: true
  auto_extract_enabled: true
  auto_extract_provider: "test-provider"
  auto_extract_model: "gpt-4"
{memory_yaml}
"#
    );
    serde_yaml::from_str::<Config>(&yaml).expect("test config YAML should deserialize")
}

async fn send(app: axum::Router, req: Request<Body>) -> (StatusCode, Vec<u8>) {
    let resp = app.oneshot(req).await.unwrap();
    let status = resp.status();
    let body = axum::body::to_bytes(resp.into_body(), 4 * 1024 * 1024)
        .await
        .unwrap();
    (status, body.to_vec())
}

/// Gateway plus its app, live memory system, config, and the temp dir
/// backing the memory database (cleaned up on drop, after the server's
/// SQLite handle closes — no temp dir leaks per run).
struct GatewayFixture {
    server: GatewayServer,
    app: axum::Router,
    system: Arc<MemorySystem>,
    config: Config,
    _dir: tempfile::TempDir,
}

/// Seed a gateway with one injectable memory fact; returns the server (for
/// hot-reload), its app, and the live memory system for post-request
/// assertions.
async fn gateway_with_memory(mock_uri: &str) -> GatewayFixture {
    let dir = tempfile::tempdir().expect("create temp dir for gateway memory database");
    let db_path = dir.path().join("memory.db");
    let memory_yaml = format!(
        "  database_path: \"{}\"",
        db_path.to_string_lossy().replace('\\', "/")
    );
    let mut config = config_with_memory(mock_uri, &memory_yaml);
    common::isolate_databases(&mut config);
    let server = GatewayServer::new(config.clone(), None).await.unwrap();
    let system = server
        .state
        .memory_system
        .read()
        .await
        .clone()
        .expect("memory system should initialize");
    create_entry(&system, "user::default", "The user prefers dark mode in editors").await;
    let app = server.build_router();
    GatewayFixture {
        server,
        app,
        system,
        config,
        _dir: dir,
    }
}

fn chat_request(headers: &[(&str, &str)], user_content: &str) -> Request<Body> {
    chat_request_with_stream(headers, user_content, false)
}

/// Same as `chat_request` but with control over `stream` for the buffered
/// streaming path (mirrors `post_stream` in `guardrail_streaming.rs`).
fn chat_request_with_stream(
    headers: &[(&str, &str)],
    user_content: &str,
    stream: bool,
) -> Request<Body> {
    let body = serde_json::json!({
        "model": "gpt-4",
        "messages": [{ "role": "user", "content": user_content }],
        "stream": stream
    });
    let mut builder = Request::post("/v1/chat/completions")
        .header("content-type", "application/json");
    for (name, value) in headers {
        builder = builder.header(*name, *value);
    }
    builder
        .body(Body::from(serde_json::to_string(&body).unwrap()))
        .unwrap()
}

/// Every JSON body the mock upstream received, for injection assertions.
/// Wiremock records a request before answering it, so every body is already
/// captured by the time the client response returns.
async fn upstream_bodies(server: &MockServer) -> Vec<serde_json::Value> {
    server
        .received_requests()
        .await
        .unwrap()
        .iter()
        .map(|request| serde_json::from_slice(&request.body).unwrap())
        .collect()
}

fn any_message_contains(body: &serde_json::Value, needle: &str) -> bool {
    body["messages"]
        .as_array()
        .map(|messages| {
            messages
                .iter()
                .any(|message| message["content"].as_str().is_some_and(|c| c.contains(needle)))
        })
        .unwrap_or(false)
}

// ---------------------------------------------------------------------------
// Req 1: edit stored memories
// ---------------------------------------------------------------------------

#[tokio::test]
async fn admin_edit_updates_content_type_and_reindexes() {
    let provider = Arc::new(FakeExtractionProvider::new());
    let MemoryFixture { system, tier, .. } = memory_system_with(provider).await;
    let entry = create_entry(&system, "user::edit", "Original safe stored content.").await;

    let app = ai_gateway::memory::admin::routes(system.clone());
    let request = Request::patch(format!("/entries/{}", entry.id))
        .header("content-type", "application/json")
        .body(Body::from(
            serde_json::to_string(&serde_json::json!({
                "content": "Edited useful content for the entry.",
                "memory_type": "context"
            }))
            .unwrap(),
        ))
        .unwrap();
    let (status, body) = send(app, request).await;
    assert_eq!(status, StatusCode::OK);
    let updated: serde_json::Value = serde_json::from_slice(&body).unwrap();
    assert_eq!(updated["content"], "Edited useful content for the entry.");
    assert_eq!(updated["memory_type"], "context");
    assert!(updated["updated_at"].is_i64(), "edit must record updated_at");

    // Reindex path: the stale point is deleted (awaited inline by
    // admin_edit) and the edited entry is re-indexed by a spawned task.
    assert_eq!(tier.count(&format!("delete:{}", entry.id)), 1);
    wait_for_events(&tier, &format!("index:{}", entry.id), 2).await;
    let stored = system.store.get_entry_by_id(entry.id).unwrap().unwrap();
    assert!(stored.updated_at.is_some());
    assert_eq!(stored.memory_type, MemoryType::Context);
}

#[tokio::test]
async fn admin_edit_returns_404_for_missing_id() {
    let provider = Arc::new(FakeExtractionProvider::new());
    let MemoryFixture { system, .. } = memory_system_with(provider).await;
    let app = ai_gateway::memory::admin::routes(system);
    let missing = Uuid::new_v4();
    let request = Request::patch(format!("/entries/{missing}"))
        .header("content-type", "application/json")
        .body(Body::from(
            serde_json::to_string(&serde_json::json!({
                "content": "Content for a missing entry.",
                "memory_type": "fact"
            }))
            .unwrap(),
        ))
        .unwrap();
    let (status, _body) = send(app, request).await;
    assert_eq!(status, StatusCode::NOT_FOUND);
}

#[tokio::test]
async fn admin_edit_rejects_invalid_and_sensitive_content() {
    let provider = Arc::new(FakeExtractionProvider::new());
    let MemoryFixture { system, .. } = memory_system_with(provider).await;
    let entry = create_entry(&system, "user::edit", "Original safe stored content.").await;
    let app = ai_gateway::memory::admin::routes(system.clone());

    // Too-short content and a synthetic OpenAI-style key both answer 400.
    for content in ["tiny", "key sk-ABCDEFGHIJKLMNOPQRSTUVWX"] {
        let request = Request::patch(format!("/entries/{}", entry.id))
            .header("content-type", "application/json")
            .body(Body::from(
                serde_json::to_string(&serde_json::json!({
                    "content": content,
                    "memory_type": "fact"
                }))
                .unwrap(),
            ))
            .unwrap();
        let (status, _body) = send(app.clone(), request).await;
        assert_eq!(status, StatusCode::BAD_REQUEST, "content: {content}");
    }

    let stored = system.store.get_entry_by_id(entry.id).unwrap().unwrap();
    assert_eq!(stored.content, "Original safe stored content.");
    assert!(stored.updated_at.is_none(), "rejected edits must not write");
}

// ---------------------------------------------------------------------------
// Req 2: supersession lifecycle
// ---------------------------------------------------------------------------

#[tokio::test]
async fn superseded_entry_is_excluded_from_retrieval_but_listed() {
    let provider = Arc::new(FakeExtractionProvider::new());
    let MemoryFixture { system, .. } = memory_system_with(provider).await;
    let old =
        create_entry(&system, "user::default", "The API endpoint is example.com/api/v1").await;
    let new =
        create_entry(&system, "user::default", "The API endpoint is example.com/api/v2").await;

    let app = ai_gateway::memory::admin::routes(system.clone());
    let request = Request::post(format!("/entries/{}/supersede", old.id))
        .header("content-type", "application/json")
        .body(Body::from(
            serde_json::to_string(&serde_json::json!({
                "superseded_by": new.id.to_string()
            }))
            .unwrap(),
        ))
        .unwrap();
    let (status, body) = send(app, request).await;
    assert_eq!(status, StatusCode::OK);
    let marked: serde_json::Value = serde_json::from_slice(&body).unwrap();
    assert_eq!(marked["marked_count"], 1);
    assert_eq!(marked["entry"]["superseded_by"], new.id.to_string());

    // Excluded from retrieval — the injector's candidate source (Req 2.1).
    let retrieved = system
        .store
        .retrieve("user::default", None, "API endpoint example.com", None)
        .unwrap();
    assert!(
        retrieved.iter().all(|memory| memory.entry.id != old.id),
        "superseded entry must not be retrieved"
    );
    assert!(
        retrieved.iter().any(|memory| memory.entry.id == new.id),
        "active entry must still be retrieved"
    );

    // Default list hides it; include_superseded lists it annotated (Req 2.3).
    let hidden = system
        .store
        .list_entries_with_superseded("user::default", 50, 0, false)
        .unwrap();
    assert!(hidden.entries.iter().all(|e| e.id != old.id));
    let shown = system
        .store
        .list_entries_with_superseded("user::default", 50, 0, true)
        .unwrap();
    let annotated = shown
        .entries
        .iter()
        .find(|e| e.id == old.id)
        .expect("superseded entry must be listed with include_superseded");
    assert_eq!(annotated.superseded_by, Some(new.id));
}

#[tokio::test]
async fn deleting_superseded_entry_still_removes_vector_point() {
    let provider = Arc::new(FakeExtractionProvider::new());
    let MemoryFixture { system, tier, .. } = memory_system_with(provider).await;
    let old = create_entry(&system, "user::delete", DUP_A).await;
    let new = create_entry(&system, "user::delete", DUP_B).await;
    system
        .store
        .mark_superseded(&[old.id], new.id)
        .expect("supersede should succeed");

    system
        .admin_delete(old.id)
        .await
        .expect("deleting a superseded entry must succeed (Req 2.5)");

    assert!(system.store.get_entry_by_id(old.id).unwrap().is_none());
    assert_eq!(tier.count(&format!("delete:{}", old.id)), 1);
    assert!(system.store.get_entry_by_id(new.id).unwrap().is_some());
}

/// The admin HTTP DELETE endpoint routes through `admin_delete`, so the
/// fake tier must observe the vector point delete for the deleted id —
/// the same cleanup the store-level test above asserts on the direct
/// `MemorySystem::admin_delete` path.
#[tokio::test]
async fn admin_delete_endpoint_issues_vector_point_delete() {
    let provider = Arc::new(FakeExtractionProvider::new());
    let MemoryFixture { system, tier, .. } = memory_system_with(provider).await;
    let entry = create_entry(&system, "user::delete-http", DUP_A).await;

    let app = ai_gateway::memory::admin::routes(system.clone());
    let request = Request::delete(format!("/entries/{}", entry.id)).body(Body::empty()).unwrap();
    let (status, _body) = send(app, request).await;
    assert_eq!(status, StatusCode::NO_CONTENT);

    assert!(system.store.get_entry_by_id(entry.id).unwrap().is_none());
    assert_eq!(
        tier.count(&format!("delete:{}", entry.id)),
        1,
        "HTTP DELETE must issue the vector point delete via admin_delete"
    );

    // A repeated DELETE answers 404 without a second tier delete.
    let app = ai_gateway::memory::admin::routes(system);
    let request = Request::delete(format!("/entries/{}", entry.id)).body(Body::empty()).unwrap();
    let (status, _body) = send(app, request).await;
    assert_eq!(status, StatusCode::NOT_FOUND);
    assert_eq!(tier.count(&format!("delete:{}", entry.id)), 1);
}

#[tokio::test]
async fn v3_database_migrates_to_v4_preserving_rows() {
    use rusqlite::Connection;

    let dir = tempfile::tempdir().expect("create temp dir for v3 database");
    let db_path = dir.path().join("memory.db");
    {
        let conn = Connection::open(&db_path).unwrap();
        conn.execute_batch(
            "CREATE TABLE memories (
id TEXT PRIMARY KEY, namespace TEXT NOT NULL, content TEXT NOT NULL,
memory_type TEXT NOT NULL, relevance_score REAL NOT NULL,
created_at INTEGER NOT NULL, last_accessed_at INTEGER NOT NULL,
access_count INTEGER NOT NULL, source_request_id TEXT,
vector_index_status TEXT NOT NULL DEFAULT 'not_configured'
CHECK(vector_index_status IN ('not_configured', 'pending', 'indexed', 'retry')),
vector_retry_count INTEGER NOT NULL DEFAULT 0 CHECK(vector_retry_count >= 0),
vector_last_attempt_at INTEGER, vector_next_retry_at INTEGER, vector_last_error TEXT
);
INSERT INTO memories (
id, namespace, content, memory_type, relevance_score,
created_at, last_accessed_at, access_count, source_request_id
) VALUES (
'00000000-0000-0000-0000-000000000001', 'user::v3',
'legacy searchable content', 'fact', 1.0, 1700000000, 1700000000, 0, NULL
);
PRAGMA user_version = 3;",
        )
        .unwrap();
    }

    // Opening through MemorySystem applies the migration (Req 2.4). Only
    // user-visible outcomes are asserted here — entry readability and FTS
    // hit after migration, plus the resulting user_version; schema and
    // column introspection stays with the store unit test.
    let system = MemorySystem::new(
        memory_config(db_path.to_string_lossy().as_ref()),
        None,
        None,
    )
    .await
    .unwrap();

    let migrated = system
        .store
        .get_entry_by_id(Uuid::parse_str("00000000-0000-0000-0000-000000000001").unwrap())
        .unwrap()
        .expect("v3 row must survive migration and stay readable");
    assert_eq!(migrated.content, "legacy searchable content");
    assert_eq!(migrated.updated_at, None);
    assert_eq!(migrated.superseded_by, None);

    // Schema reached v4 (WAL mode permits a second reader connection while
    // the store keeps its handle open).
    let reader = Connection::open(&db_path).unwrap();
    let version: i64 = reader
        .query_row("PRAGMA user_version", [], |row| row.get(0))
        .unwrap();
    assert_eq!(version, 4);

    // The migrated row is searchable through the rebuilt FTS index. The
    // 2023 fixture timestamps decay the recency boost below the retrieval
    // scoring threshold, so FTS membership is asserted directly rather
    // than through `retrieve`.
    let fts_hits: i64 = reader
        .query_row(
            "SELECT count(*) FROM memories_fts WHERE memories_fts MATCH 'legacy'",
            [],
            |row| row.get(0),
        )
        .unwrap();
    assert_eq!(fts_hits, 1, "migrated content must be FTS-searchable");
}

// ---------------------------------------------------------------------------
// Req 3: consolidation end-to-end
// ---------------------------------------------------------------------------

#[tokio::test]
async fn consolidation_merges_near_duplicates_and_supersedes_sources() {
    let provider = Arc::new(FakeExtractionProvider::new());
    let MemoryFixture { system, tier, .. } = memory_system_with(provider.clone()).await;
    let a = create_entry(&system, "user::consolidate", DUP_A).await;
    let b = create_entry(&system, "user::consolidate", DUP_B).await;

    // The provider decision references the real entry ids and returns
    // strict-JSON decisions through the candidate-content channel.
    provider.set_raw_output(
        serde_json::json!([{
            "action": "merge",
            "source_ids": [a.id.to_string(), b.id.to_string()],
            "merged_content": "The deployment target for staging is fly.io in the eu-west region"
        }])
        .to_string(),
    );

    let counts = system.run_consolidation_now().await.unwrap();
    assert_eq!(
        counts,
        ConsolidationCounts {
            merged: 1,
            superseded: 2,
            kept: 0,
            skipped_batches: 0,
            provider_errors: 0,
        }
    );

    // Both sources are superseded; exactly one active merged entry exists.
    let listed = system
        .store
        .list_entries_with_superseded("user::consolidate", 50, 0, true)
        .unwrap();
    assert_eq!(listed.entries.len(), 3);
    let merged = listed
        .entries
        .iter()
        .find(|entry| entry.superseded_by.is_none())
        .expect("merged entry must exist and stay active");
    assert!(merged.content.contains("fly.io"));
    let merged_id = merged.id;
    assert_ne!(merged_id, a.id);
    assert_ne!(merged_id, b.id);
    let sources: Vec<_> = listed
        .entries
        .iter()
        .filter(|entry| entry.superseded_by.is_some())
        .collect();
    assert_eq!(sources.len(), 2);
    for source in &sources {
        assert_eq!(source.superseded_by, Some(merged_id));
    }

    // Last-pass metric summary (Req 3.6).
    let last_pass = system.metrics.last_consolidation_pass().unwrap();
    assert_eq!(last_pass.counts, counts);

    // Vector tier: both source points deleted, merged entry indexed.
    assert_eq!(tier.count(&format!("delete:{}", a.id)), 1);
    assert_eq!(tier.count(&format!("delete:{}", b.id)), 1);
    wait_for_events(&tier, &format!("index:{merged_id}"), 1).await;
}

#[tokio::test]
async fn malformed_provider_output_skips_batch_and_serving_continues() {
    let provider = Arc::new(FakeExtractionProvider::new());
    let MemoryFixture { system, .. } = memory_system_with(provider.clone()).await;
    let a = create_entry(&system, "user::malformed", DUP_A).await;
    let _b = create_entry(&system, "user::malformed", DUP_B).await;

    provider.set_raw_output("this is not valid JSON".to_owned());
    let counts = system.run_consolidation_now().await.unwrap();
    assert_eq!(counts.skipped_batches, 1);
    assert_eq!(counts.merged, 0);
    assert_eq!(counts.superseded, 0);

    // Serving is unaffected: both entries stay active and retrievable.
    let listed = system
        .store
        .list_entries_with_superseded("user::malformed", 50, 0, false)
        .unwrap();
    assert_eq!(listed.entries.len(), 2);
    assert!(
        listed
            .entries
            .iter()
            .all(|entry| entry.superseded_by.is_none()),
        "no entry may be superseded by a skipped batch"
    );
    let retrieved = system
        .store
        .retrieve("user::malformed", None, "deployment staging fly", None)
        .unwrap();
    assert!(
        retrieved.iter().any(|memory| memory.entry.id == a.id),
        "retrieval must keep serving the untouched entries"
    );
}

// ---------------------------------------------------------------------------
// Req 4/5: admin search and export
// ---------------------------------------------------------------------------

#[tokio::test]
async fn admin_search_finds_entries_across_namespaces() {
    let provider = Arc::new(FakeExtractionProvider::new());
    let MemoryFixture { system, .. } = memory_system_with(provider).await;
    create_entry(&system, "user::alpha", "Kubernetes rollout strategy notes").await;
    create_entry(&system, "user::beta", "Kubernetes ingress configuration details").await;
    create_entry(&system, "user::gamma", "Unrelated lunch preferences listed here").await;

    let app = ai_gateway::memory::admin::routes(system);
    let request = Request::get("/search?q=kubernetes")
        .body(Body::empty())
        .unwrap();
    let (status, body) = send(app, request).await;
    assert_eq!(status, StatusCode::OK);
    let page: serde_json::Value = serde_json::from_slice(&body).unwrap();
    assert_eq!(page["total_count"], 2);
    let contents: Vec<&str> = page["entries"]
        .as_array()
        .unwrap()
        .iter()
        .map(|entry| entry["content"].as_str().unwrap())
        .collect();
    assert!(contents.iter().any(|content| content.contains("rollout")));
    assert!(contents.iter().any(|content| content.contains("ingress")));
}

#[tokio::test]
async fn admin_search_empty_query_falls_back_to_listing() {
    let provider = Arc::new(FakeExtractionProvider::new());
    let MemoryFixture { system, .. } = memory_system_with(provider).await;
    create_entry(&system, "user::alpha", "First searchable stored fact").await;
    create_entry(&system, "user::beta", "Second searchable stored fact").await;

    let app = ai_gateway::memory::admin::routes(system);
    let request = Request::get("/search?q=").body(Body::empty()).unwrap();
    let (status, body) = send(app, request).await;
    assert_eq!(status, StatusCode::OK);
    let page: serde_json::Value = serde_json::from_slice(&body).unwrap();
    assert_eq!(page["total_count"], 2);
    assert_eq!(page["entries"].as_array().unwrap().len(), 2);
}

#[tokio::test]
async fn admin_export_includes_superseded_entries_and_total() {
    let provider = Arc::new(FakeExtractionProvider::new());
    let MemoryFixture { system, .. } = memory_system_with(provider).await;
    let old = create_entry(&system, "user::export", "Exported superseded memory entry").await;
    let new = create_entry(&system, "user::export", "Exported active memory entry").await;
    system
        .store
        .mark_superseded(&[old.id], new.id)
        .expect("supersede should succeed");

    let app = ai_gateway::memory::admin::routes(system);
    let request = Request::get("/export").body(Body::empty()).unwrap();
    let (status, body) = send(app, request).await;
    assert_eq!(status, StatusCode::OK);
    let export: serde_json::Value = serde_json::from_slice(&body).unwrap();
    assert_eq!(export["total"], 2);
    let entries = export["entries"].as_array().unwrap();
    let superseded = entries
        .iter()
        .find(|entry| entry["content"]
            .as_str()
            .is_some_and(|content| content.contains("superseded")))
        .expect("export must include the superseded entry");
    assert_eq!(
        superseded["superseded_by"].as_str().unwrap(),
        new.id.to_string()
    );
    assert!(
        entries.iter().any(|entry| entry["superseded_by"].is_null()),
        "active entries export without a superseded_by value"
    );
}

// ---------------------------------------------------------------------------
// Req 8: per-request memory mode
// ---------------------------------------------------------------------------

#[tokio::test]
async fn memory_mode_off_skips_injection_and_extraction() {
    let mock = start_mock_provider().await;
    let server_uri = mock.uri();
    let GatewayFixture { app, system, .. } = gateway_with_memory(&server_uri).await;

    // The content carries an extraction trigger ("I prefer") and retrieval
    // tokens ("dark mode"), so both memory paths would fire if not gated.
    let request = chat_request(
        &[("x-obey-memory-mode", "off")],
        "Remember that I prefer dark mode everywhere.",
    );
    let (status, _body) = send(app, request).await;
    assert_eq!(status, StatusCode::OK);

    let bodies = upstream_bodies(&mock).await;
    assert!(
        bodies
            .iter()
            .all(|body| !any_message_contains(body, "[Recalled memories]")),
        "mode off must skip injection"
    );
    assert_eq!(
        system.store.stats().unwrap().total_count,
        1,
        "mode off must skip extraction (only the seeded entry remains)"
    );
}

#[tokio::test]
async fn memory_mode_read_only_injects_but_skips_extraction() {
    let mock = start_mock_provider().await;
    let server_uri = mock.uri();
    let GatewayFixture { app, system, .. } = gateway_with_memory(&server_uri).await;

    let request = chat_request(
        &[("x-obey-memory-mode", "read-only")],
        "Remember this: what is my dark mode preference?",
    );
    let (status, _body) = send(app, request).await;
    assert_eq!(status, StatusCode::OK);

    let bodies = upstream_bodies(&mock).await;
    assert!(
        bodies
            .iter()
            .any(|body| any_message_contains(body, "[Recalled memories]")),
        "read-only mode must still inject"
    );
    assert_eq!(
        system.store.stats().unwrap().total_count,
        1,
        "read-only mode must skip extraction despite the trigger"
    );
}

#[tokio::test]
async fn absent_memory_mode_header_applies_configured_behavior() {
    let mock = start_mock_provider().await;
    let server_uri = mock.uri();
    let GatewayFixture { app, system, .. } = gateway_with_memory(&server_uri).await;

    // The content both shares retrieval tokens with the seeded memory and
    // carries an explicit "Remember this" trigger, so the configured path
    // injects AND stores (auto-extraction needs `auto_extract_min_turns`
    // caller turns, default 4, so the explicit trigger is the deterministic
    // write path here).
    let request = chat_request(&[], "Remember this: I like dark mode themes.");
    let (status, _body) = send(app, request).await;
    assert_eq!(status, StatusCode::OK);

    let bodies = upstream_bodies(&mock).await;
    assert!(
        bodies
            .iter()
            .any(|body| any_message_contains(body, "[Recalled memories]")),
        "absent header must apply configured injection"
    );
    assert!(
        system.store.stats().unwrap().total_count >= 2,
        "absent header must let configured extraction store the new memory"
    );
}

#[tokio::test]
async fn memory_mode_off_skips_injection_and_extraction_on_streaming_path() {
    let mock = start_mock_provider().await;
    let server_uri = mock.uri();
    let GatewayFixture { app, system, .. } = gateway_with_memory(&server_uri).await;

    // Buffered streaming path (`stream: true`, pass-through disabled in the
    // fixture config): dispatch streams upstream and reassembles the SSE.
    let request = chat_request_with_stream(
        &[("x-obey-memory-mode", "off")],
        "Remember that I prefer dark mode everywhere.",
        true,
    );
    let (status, body) = send(app, request).await;
    assert_eq!(status, StatusCode::OK);
    let text = std::str::from_utf8(&body).unwrap();
    assert!(text.contains("data: "), "streaming response must be SSE");
    assert!(text.contains("[DONE]"), "stream must terminate with [DONE]");

    let bodies = upstream_bodies(&mock).await;
    assert!(
        bodies
            .iter()
            .all(|body| !any_message_contains(body, "[Recalled memories]")),
        "mode off must skip injection on the streaming path"
    );
    assert_eq!(
        system.store.stats().unwrap().total_count,
        1,
        "mode off must skip extraction on the streaming path (only the seeded entry remains)"
    );
}

#[tokio::test]
async fn unrecognized_memory_mode_value_applies_configured_behavior() {
    let mock = start_mock_provider().await;
    let server_uri = mock.uri();
    let GatewayFixture { app, system, .. } = gateway_with_memory(&server_uri).await;

    // "full" is not a recognized mode (only `off` and `read-only` parse), so
    // the header must fall back to `Configured`: inject + extract.
    let request = chat_request(
        &[("x-obey-memory-mode", "full")],
        "Remember this: I like dark mode themes.",
    );
    let (status, _body) = send(app, request).await;
    assert_eq!(status, StatusCode::OK);

    let bodies = upstream_bodies(&mock).await;
    assert!(
        bodies
            .iter()
            .any(|body| any_message_contains(body, "[Recalled memories]")),
        "unrecognized mode value must fall back to configured injection"
    );
    assert!(
        system.store.stats().unwrap().total_count >= 2,
        "unrecognized mode value must fall back to configured extraction"
    );
}

// ---------------------------------------------------------------------------
// Req 3.1/3.7: config validation, fallback, hot-reload
// ---------------------------------------------------------------------------

#[test]
fn default_memory_config_validates_unchanged() {
    assert!(MemoryConfig::default().validate().is_ok());
}

#[test]
fn consolidation_enabled_with_unknown_provider_fails_validation_naming_field() {
    let config = MemoryConfig {
        consolidation_enabled: true,
        consolidation_provider: "missing-provider".to_owned(),
        consolidation_model: "some-model".to_owned(),
        ..MemoryConfig::default()
    };
    let errors = config
        .validate_with_provider_names(["test-provider"])
        .unwrap_err();
    assert!(
        errors
            .iter()
            .any(|error| error.field == "consolidation_provider"),
        "error must name consolidation_provider, got {errors:?}"
    );
}

#[test]
fn consolidation_falls_back_to_auto_extract_pair() {
    let config = MemoryConfig {
        consolidation_enabled: true,
        auto_extract_provider: "test-provider".to_owned(),
        auto_extract_model: "test-model".to_owned(),
        ..MemoryConfig::default()
    };
    assert!(config.validate_with_provider_names(["test-provider"]).is_ok());
    assert_eq!(
        config.resolve_consolidation_pair().unwrap(),
        ("test-provider".to_owned(), "test-model".to_owned())
    );
}

#[tokio::test]
async fn gateway_hot_reload_applies_consolidation_settings() {
    let mock = start_mock_provider().await;
    let server_uri = mock.uri();
    let GatewayFixture { server, system, mut config, .. } = gateway_with_memory(&server_uri).await;
    assert!(
        !system.config.read().await.consolidation_enabled,
        "consolidation must start disabled"
    );

    let memory = config.memory.as_mut().unwrap();
    memory.consolidation_enabled = true;
    memory.consolidation_provider = "test-provider".to_owned();
    memory.consolidation_model = "gpt-4".to_owned();
    server
        .reload_config(config)
        .await
        .expect("hot-reload should apply consolidation settings");

    let snapshot = system.config.read().await;
    assert!(snapshot.consolidation_enabled);
    assert_eq!(
        snapshot.resolve_consolidation_pair().unwrap(),
        ("test-provider".to_owned(), "gpt-4".to_owned())
    );
}
