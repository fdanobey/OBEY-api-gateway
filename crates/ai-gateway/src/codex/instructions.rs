//! Codex system-prompt instructions store.
//!
//! Implements the bundled-fallback → on-disk cache → GitHub refresh ladder
//! described in design §5 of `codex-backend-translation`. This file covers
//! the **task 7.1**, **task 7.2**, and **task 7.3** slices:
//!
//! * `CachedPrompt` / `InstructionsStore` data model.
//! * Compile-time bundled fallback via `include_str!`.
//! * `new()` resolves the on-disk cache paths beside the OAuth token store
//!   and attempts to seed the in-memory cache from disk; missing or
//!   malformed cache files fall back to the bundled prompt with
//!   `fetched_at = 0`.
//! * `get()` short-circuits on the per-provider override and otherwise
//!   clones the cached body under the read lock.
//! * `save_cache()` writes the body and the etag/timestamp atomically and
//!   re-applies owner-only (`0600`) permissions on Unix (Req 12.5).
//! * `maybe_refresh()` performs a conditional GET against
//!   `self.github_url` with a 24-hour gate (Req 7.2, 7.3, 7.4, 7.5, 7.8).

use std::path::{Path, PathBuf};

use anyhow::Context;
use tokio::sync::RwLock;

use crate::config::Config;
use crate::secrets;

/// Default upstream source for the Codex system prompt. Used when
/// [`Config::codex_instructions_url`] is `None`.
pub const DEFAULT_CODEX_INSTRUCTIONS_URL: &str =
    "https://raw.githubusercontent.com/openai/codex/main/codex-rs/core/gpt_5_1_prompt.md";

/// Refresh gate: skip the conditional GET when the in-memory cache is
/// younger than 24 hours (Req 7.8). `fetched_at == 0` is a sentinel
/// meaning "force refresh on next opportunity" and bypasses the gate.
const REFRESH_INTERVAL_SECS: u64 = 86_400;

/// Filename of the cached prompt body, colocated with `master.key` and
/// `oauth_tokens.enc` inside the per-user secrets directory.
const CODEX_INSTRUCTIONS_FILENAME: &str = "codex_instructions.md";
/// Filename of the companion etag/timestamp file.
const CODEX_INSTRUCTIONS_ETAG_FILENAME: &str = "codex_instructions.etag";

/// Bundled fallback prompt compiled into the binary. Vendored verbatim from
/// `openai/codex` at a pinned commit (see `NOTICE` for attribution and
/// upstream path). Satisfies Req 7.1 — the gateway is always able to serve
/// *some* Codex-shaped instructions, even before the first disk-cache load
/// or GitHub fetch.
const BUNDLED_PROMPT: &str = include_str!("gpt_5_1_prompt.md");

/// In-memory snapshot of the Codex system prompt plus the metadata needed
/// to drive the 24-hour conditional-GET refresh loop (task 7.3).
///
/// `fetched_at == 0` is a sentinel: "the cache body is the bundled fallback,
/// we have never successfully loaded from disk or network" (Req 7.5, 7.7).
/// The refresher treats it as stale and triggers an unconditional fetch on
/// startup.
#[derive(Debug, Clone)]
pub(crate) struct CachedPrompt {
    pub(crate) body: String,
    pub(crate) etag: Option<String>,
    pub(crate) fetched_at: u64,
}

/// Codex system-prompt store. One instance per gateway; shared across the
/// router via `Arc<InstructionsStore>`.
#[derive(Debug)]
pub struct InstructionsStore {
    cache: RwLock<CachedPrompt>,
    /// Persistent copy of the prompt body
    /// (`<secrets_dir>/codex_instructions.md`).
    disk_path: PathBuf,
    /// Companion ETag + timestamp file
    /// (`<secrets_dir>/codex_instructions.etag`).
    etag_path: PathBuf,
    /// Upstream URL used by the conditional-GET refresh in
    /// [`Self::maybe_refresh`]. Resolved from
    /// [`Config::codex_instructions_url`] at construction, falling back to
    /// [`DEFAULT_CODEX_INSTRUCTIONS_URL`] when absent.
    github_url: String,
    /// HTTP client reused across refreshes.
    http: reqwest::Client,
}

impl InstructionsStore {
    /// Build a new store seeded from the on-disk cache when present, falling
    /// back to the bundled prompt otherwise.
    ///
    /// Resolves `<secrets_dir>/codex_instructions.md` and
    /// `<secrets_dir>/codex_instructions.etag` via
    /// [`secrets::storage_dir_path`] so the cache lives alongside the OAuth
    /// token blob (Req 7.4). When either file is missing or the etag side-
    /// car cannot be parsed, the in-memory cache is seeded with
    /// [`BUNDLED_PROMPT`] and `fetched_at = 0` so [`Self::maybe_refresh`]
    /// treats it as stale.
    ///
    /// Honours [`Config::codex_instructions_url`] when set; otherwise
    /// falls back to [`DEFAULT_CODEX_INSTRUCTIONS_URL`].
    pub async fn new(cfg: &Config) -> anyhow::Result<Self> {
        let storage_dir = secrets::storage_dir_path()
            .context("resolve secrets storage dir for codex instructions cache")?;
        let disk_path = storage_dir.join(CODEX_INSTRUCTIONS_FILENAME);
        let etag_path = storage_dir.join(CODEX_INSTRUCTIONS_ETAG_FILENAME);

        let github_url = cfg
            .codex_instructions_url
            .clone()
            .unwrap_or_else(|| DEFAULT_CODEX_INSTRUCTIONS_URL.to_string());

        let cached = match load_disk_cache(&disk_path, &etag_path).await {
            Some(p) => p,
            None => CachedPrompt {
                body: BUNDLED_PROMPT.to_string(),
                etag: None,
                fetched_at: 0,
            },
        };

        Ok(Self {
            cache: RwLock::new(cached),
            disk_path,
            etag_path,
            github_url,
            http: reqwest::Client::new(),
        })
    }

    /// Resolve the instructions string for a single request.
    ///
    /// When `override_` is `Some`, it short-circuits any cache lookup so
    /// operators can pin a provider to a custom prompt (Req 7.7). Otherwise
    /// the cached body is cloned under the read lock (Req 7.6).
    pub async fn get(&self, override_: Option<&str>) -> String {
        if let Some(value) = override_ {
            return value.to_string();
        }
        self.cache.read().await.body.clone()
    }

    /// Persist `body` + `etag`/`fetched_at` to disk atomically and refresh
    /// the in-memory cache under the write lock.
    ///
    /// Writes go to `<path>.tmp` first and are then renamed into place so
    /// readers never observe a half-written file. After the rename
    /// succeeds, [`secrets::set_owner_only_permissions`] is called on the
    /// final path to enforce mode `0600` on Unix (Req 12.5); on non-Unix
    /// platforms the call is a no-op.
    ///
    /// The etag side-car is encoded as `<etag>\n<unix_ts>`; an absent etag
    /// is serialized as the empty string, preserving the trailing
    /// timestamp line so [`parse_etag_file`] still succeeds on read.
    pub(crate) async fn save_cache(
        &self,
        body: &str,
        etag: Option<&str>,
        fetched_at: u64,
    ) -> anyhow::Result<()> {
        if let Some(parent) = self.disk_path.parent() {
            tokio::fs::create_dir_all(parent).await.with_context(|| {
                format!(
                    "create codex instructions cache dir at {}",
                    parent.display()
                )
            })?;
        }

        atomic_write_with_owner_only(&self.disk_path, body.as_bytes()).await?;

        let etag_blob = format!("{}\n{}", etag.unwrap_or(""), fetched_at);
        atomic_write_with_owner_only(&self.etag_path, etag_blob.as_bytes()).await?;

        let mut guard = self.cache.write().await;
        *guard = CachedPrompt {
            body: body.to_string(),
            etag: etag.map(|s| s.to_string()),
            fetched_at,
        };
        Ok(())
    }

    /// Rewrite only the etag side-car file (preserving the existing etag)
    /// and bump `cache.fetched_at` under the write lock.
    ///
    /// Used by [`Self::maybe_refresh`] on a `304 Not Modified` response so
    /// the 24-hour gate (Req 7.8) restarts without rewriting the body
    /// (Req 7.3).
    pub(crate) async fn touch_etag(&self, fetched_at: u64) -> anyhow::Result<()> {
        if let Some(parent) = self.etag_path.parent() {
            tokio::fs::create_dir_all(parent).await.with_context(|| {
                format!(
                    "create codex instructions cache dir at {}",
                    parent.display()
                )
            })?;
        }

        let etag = self.cache.read().await.etag.clone();
        let etag_blob = format!("{}\n{}", etag.as_deref().unwrap_or(""), fetched_at);
        atomic_write_with_owner_only(&self.etag_path, etag_blob.as_bytes()).await?;

        self.cache.write().await.fetched_at = fetched_at;
        Ok(())
    }

    /// Conditional GET against [`Self::github_url`] with a 24-hour gate.
    ///
    /// * Skips when `now - fetched_at < REFRESH_INTERVAL_SECS`. The
    ///   `fetched_at == 0` sentinel always triggers a fetch — this is the
    ///   "first attempt at startup" carve-out from Req 7.8.
    /// * Sends `If-None-Match: <etag>` when an etag is cached.
    /// * `200 OK` → persist body + new etag + timestamp via
    ///   [`Self::save_cache`] (Req 7.4).
    /// * `304 Not Modified` → retain body, bump `fetched_at`, rewrite the
    ///   `.etag` side-car via [`Self::touch_etag`] (Req 7.3).
    /// * Network error or any other status → keep the current cache, emit
    ///   `warn!` with the URL host and error kind only (Req 7.5).
    ///
    /// Logs never include the response body, the request body, or any
    /// header value.
    pub async fn maybe_refresh(&self) {
        // Snapshot the gate inputs under a short read lock so we never hold
        // the lock across network I/O.
        let (fetched_at, etag) = {
            let guard = self.cache.read().await;
            (guard.fetched_at, guard.etag.clone())
        };

        let now = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .map(|d| d.as_secs())
            .unwrap_or_default();

        if fetched_at != 0 && now.saturating_sub(fetched_at) < REFRESH_INTERVAL_SECS {
            return;
        }

        let host = reqwest::Url::parse(&self.github_url)
            .ok()
            .and_then(|u| u.host_str().map(|s| s.to_string()))
            .unwrap_or_else(|| "<unparsable-url>".to_string());

        let mut req = self.http.get(&self.github_url);
        if let Some(tag) = etag.as_deref() {
            req = req.header(reqwest::header::IF_NONE_MATCH, tag);
        }

        let resp = match req.send().await {
            Ok(r) => r,
            Err(e) => {
                tracing::warn!(
                    host = %host,
                    error_kind = ?e.without_url(),
                    "codex instructions refresh: network error; keeping current cache",
                );
                return;
            }
        };

        let status = resp.status();
        if status == reqwest::StatusCode::NOT_MODIFIED {
            if let Err(e) = self.touch_etag(now).await {
                tracing::warn!(
                    host = %host,
                    error = %e,
                    "codex instructions refresh: 304 received but failed to rewrite etag side-car",
                );
            }
            return;
        }

        if !status.is_success() {
            tracing::warn!(
                host = %host,
                status = %status.as_u16(),
                "codex instructions refresh: non-2xx response; keeping current cache",
            );
            return;
        }

        // 200: extract the new etag header *before* consuming the body.
        let new_etag = resp
            .headers()
            .get(reqwest::header::ETAG)
            .and_then(|v| v.to_str().ok())
            .map(|s| s.to_string());

        let body = match resp.text().await {
            Ok(b) => b,
            Err(e) => {
                tracing::warn!(
                    host = %host,
                    error = %e,
                    "codex instructions refresh: failed to read response body; keeping current cache",
                );
                return;
            }
        };

        if let Err(e) = self.save_cache(&body, new_etag.as_deref(), now).await {
            tracing::warn!(
                host = %host,
                error = %e,
                "codex instructions refresh: failed to persist cache; keeping current cache",
            );
        }
    }

    /// Test-only constructor that pins both cache file paths and the
    /// upstream GitHub URL so [`Self::maybe_refresh`] can be driven against
    /// a `wiremock::MockServer` without touching the real OAuth-secrets
    /// directory.
    #[cfg(test)]
    pub(crate) fn new_for_test_with_paths_and_url(
        disk_path: PathBuf,
        etag_path: PathBuf,
        github_url: String,
    ) -> Self {
        Self {
            cache: RwLock::new(CachedPrompt {
                body: BUNDLED_PROMPT.to_string(),
                etag: None,
                fetched_at: 0,
            }),
            disk_path,
            etag_path,
            github_url,
            http: reqwest::Client::new(),
        }
    }

    /// Test-only constructor that pins both cache file paths so the disk
    /// cache round-trip can be exercised against a `tempfile::tempdir()`
    /// without going through `secrets::storage_dir_path`. Thin wrapper
    /// around [`Self::new_for_test_with_paths_and_url`] preserved for
    /// back-compat with the task 7.1 / 7.2 tests.
    #[cfg(test)]
    pub(crate) fn new_for_test_with_paths(disk_path: PathBuf, etag_path: PathBuf) -> Self {
        Self::new_for_test_with_paths_and_url(
            disk_path,
            etag_path,
            DEFAULT_CODEX_INSTRUCTIONS_URL.to_string(),
        )
    }
}

/// Attempt to load the prompt body and its etag/timestamp side-car from
/// disk.
///
/// Returns `None` when either file is missing or when the etag side-car is
/// malformed. The body file's content is returned verbatim (UTF-8). Any
/// I/O error other than `NotFound` is logged at `warn!` (path only, no
/// body bytes) and treated as a missing-cache fallthrough so the caller
/// reseeds with the bundled prompt.
pub(crate) async fn load_disk_cache(disk_path: &Path, etag_path: &Path) -> Option<CachedPrompt> {
    let body = match tokio::fs::read_to_string(disk_path).await {
        Ok(b) => b,
        Err(e) if e.kind() == std::io::ErrorKind::NotFound => return None,
        Err(e) => {
            tracing::warn!(
                path = %disk_path.display(),
                error = %e,
                "failed to read codex instructions cache; falling back to bundled prompt"
            );
            return None;
        }
    };

    let etag_raw = match tokio::fs::read_to_string(etag_path).await {
        Ok(s) => s,
        Err(e) if e.kind() == std::io::ErrorKind::NotFound => return None,
        Err(e) => {
            tracing::warn!(
                path = %etag_path.display(),
                error = %e,
                "failed to read codex instructions etag side-car; falling back to bundled prompt"
            );
            return None;
        }
    };

    let (etag, fetched_at) = match parse_etag_file(&etag_raw) {
        Some(parsed) => parsed,
        None => {
            tracing::warn!(
                path = %etag_path.display(),
                "codex instructions etag side-car is malformed; falling back to bundled prompt"
            );
            return None;
        }
    };

    Some(CachedPrompt {
        body,
        etag,
        fetched_at,
    })
}

/// Parse the `<etag>\n<unix_ts>` etag side-car format.
///
/// Returns:
/// * `Some((Some(etag), ts))` when the first line is a non-empty etag and
///   the second line parses as `u64`.
/// * `Some((None, ts))` when the first line is empty (i.e. the file
///   begins with `\n<ts>`) and the second line parses as `u64`. This is
///   how [`InstructionsStore::save_cache`] serializes a missing etag.
/// * `None` when the input does not contain at least two lines or the
///   timestamp line cannot be parsed as `u64`.
pub(crate) fn parse_etag_file(s: &str) -> Option<(Option<String>, u64)> {
    let mut lines = s.split('\n');
    let etag_line = lines.next()?;
    let ts_line = lines.next()?;
    let ts: u64 = ts_line.trim().parse().ok()?;
    let etag = if etag_line.is_empty() {
        None
    } else {
        Some(etag_line.to_string())
    };
    Some((etag, ts))
}

/// Atomic write helper: dump `bytes` to `<path>.tmp`, rename into `path`,
/// then re-apply owner-only permissions on the final file.
async fn atomic_write_with_owner_only(path: &Path, bytes: &[u8]) -> anyhow::Result<()> {
    let tmp_path = {
        let mut p = path.as_os_str().to_owned();
        p.push(".tmp");
        PathBuf::from(p)
    };

    tokio::fs::write(&tmp_path, bytes).await.with_context(|| {
        format!(
            "write codex instructions tmp file at {}",
            tmp_path.display()
        )
    })?;

    tokio::fs::rename(&tmp_path, path).await.with_context(|| {
        format!(
            "rename codex instructions tmp file {} -> {}",
            tmp_path.display(),
            path.display()
        )
    })?;

    let final_path = path.to_path_buf();
    secrets::set_owner_only_permissions(&final_path)
        .with_context(|| format!("apply owner-only perms to {}", path.display()))?;
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use tempfile::tempdir;

    #[test]
    fn bundled_prompt_constant_is_non_empty() {
        assert!(
            !BUNDLED_PROMPT.is_empty(),
            "include_str!(\"gpt_5_1_prompt.md\") must embed the vendored prompt"
        );
    }

    #[tokio::test]
    async fn get_with_override_short_circuits() {
        let dir = tempdir().unwrap();
        let store = InstructionsStore::new_for_test_with_paths(
            dir.path().join("codex_instructions.md"),
            dir.path().join("codex_instructions.etag"),
        );
        let out = store.get(Some("custom")).await;
        assert_eq!(out, "custom");
    }

    #[tokio::test]
    async fn fresh_store_serves_bundled_prompt_when_no_disk_cache() {
        let dir = tempdir().unwrap();
        let store = InstructionsStore::new_for_test_with_paths(
            dir.path().join("codex_instructions.md"),
            dir.path().join("codex_instructions.etag"),
        );
        assert_eq!(store.get(None).await, BUNDLED_PROMPT);
    }

    #[test]
    fn parse_etag_file_round_trip() {
        let s = "W/\"abc123\"\n1700000000";
        let (etag, ts) = parse_etag_file(s).unwrap();
        assert_eq!(etag.as_deref(), Some("W/\"abc123\""));
        assert_eq!(ts, 1_700_000_000);
    }

    #[test]
    fn parse_etag_file_empty_etag() {
        let s = "\n42";
        let (etag, ts) = parse_etag_file(s).unwrap();
        assert!(etag.is_none(), "leading newline must yield None etag");
        assert_eq!(ts, 42);
    }

    #[test]
    fn parse_etag_file_malformed() {
        // Single line — no timestamp.
        assert!(parse_etag_file("only-one-line").is_none());
        // Non-numeric timestamp.
        assert!(parse_etag_file("etag\nnot-a-number").is_none());
        // Empty input.
        assert!(parse_etag_file("").is_none());
    }

    #[tokio::test]
    async fn disk_cache_round_trip() {
        let dir = tempdir().unwrap();
        let disk_path = dir.path().join("codex_instructions.md");
        let etag_path = dir.path().join("codex_instructions.etag");

        let store =
            InstructionsStore::new_for_test_with_paths(disk_path.clone(), etag_path.clone());

        let body = "you are codex\n";
        let etag = Some("W/\"v1\"");
        let ts: u64 = 1_700_000_001;
        store.save_cache(body, etag, ts).await.unwrap();

        // In-memory cache reflects the write.
        assert_eq!(store.get(None).await, body);

        // Files exist on disk with the documented format.
        let on_disk_body = tokio::fs::read_to_string(&disk_path).await.unwrap();
        assert_eq!(on_disk_body, body);
        let on_disk_etag = tokio::fs::read_to_string(&etag_path).await.unwrap();
        assert_eq!(on_disk_etag, format!("W/\"v1\"\n{ts}"));

        // load_disk_cache replays the same triple.
        let reloaded = load_disk_cache(&disk_path, &etag_path).await.unwrap();
        assert_eq!(reloaded.body, body);
        assert_eq!(reloaded.etag.as_deref(), Some("W/\"v1\""));
        assert_eq!(reloaded.fetched_at, ts);
    }

    #[tokio::test]
    async fn missing_disk_cache_falls_back_to_bundled() {
        let dir = tempdir().unwrap();
        let disk_path = dir.path().join("codex_instructions.md");
        let etag_path = dir.path().join("codex_instructions.etag");

        // Neither file exists yet.
        assert!(load_disk_cache(&disk_path, &etag_path).await.is_none());

        // Body present, etag missing -> still fall back.
        tokio::fs::write(&disk_path, b"partial").await.unwrap();
        assert!(load_disk_cache(&disk_path, &etag_path).await.is_none());

        // Both present but etag malformed -> still fall back.
        tokio::fs::write(&etag_path, b"only-one-line")
            .await
            .unwrap();
        assert!(load_disk_cache(&disk_path, &etag_path).await.is_none());
    }

    #[cfg(unix)]
    #[tokio::test]
    async fn saved_files_are_mode_0600() {
        use std::os::unix::fs::PermissionsExt;

        let dir = tempdir().unwrap();
        let disk_path = dir.path().join("codex_instructions.md");
        let etag_path = dir.path().join("codex_instructions.etag");

        let store =
            InstructionsStore::new_for_test_with_paths(disk_path.clone(), etag_path.clone());
        store
            .save_cache("body", Some("etag"), 1_700_000_002)
            .await
            .unwrap();

        for p in [&disk_path, &etag_path] {
            let mode = tokio::fs::metadata(p).await.unwrap().permissions().mode() & 0o777;
            assert_eq!(mode, 0o600, "{} must be mode 0600", p.display());
        }
    }

    // -----------------------------------------------------------------
    // Task 7.3 — `maybe_refresh` GitHub-fetch tests (wiremock-driven).
    // -----------------------------------------------------------------

    /// Helper: stamp the in-memory cache + side-car files with a known
    /// `(body, etag, fetched_at)` triple so refresh-gate behaviour can be
    /// exercised without going through the network on the seed path.
    async fn seed_cache(
        store: &InstructionsStore,
        body: &str,
        etag: Option<&str>,
        fetched_at: u64,
    ) {
        store.save_cache(body, etag, fetched_at).await.unwrap();
    }

    fn now_secs() -> u64 {
        std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .map(|d| d.as_secs())
            .unwrap_or_default()
    }

    #[tokio::test]
    async fn refresh_200_persists_body_and_etag() {
        use wiremock::matchers::{method, path};
        use wiremock::{Mock, MockServer, ResponseTemplate};

        let dir = tempdir().unwrap();
        let disk_path = dir.path().join("codex_instructions.md");
        let etag_path = dir.path().join("codex_instructions.etag");

        let server = MockServer::start().await;
        Mock::given(method("GET"))
            .and(path("/prompt.md"))
            .respond_with(
                ResponseTemplate::new(200)
                    .insert_header("ETag", "W/\"v2\"")
                    .set_body_string("fresh prompt body"),
            )
            .expect(1)
            .mount(&server)
            .await;

        let store = InstructionsStore::new_for_test_with_paths_and_url(
            disk_path.clone(),
            etag_path.clone(),
            format!("{}/prompt.md", server.uri()),
        );

        let before = now_secs();
        store.maybe_refresh().await;
        let after = now_secs();

        assert_eq!(store.get(None).await, "fresh prompt body");

        let on_disk_body = tokio::fs::read_to_string(&disk_path).await.unwrap();
        assert_eq!(on_disk_body, "fresh prompt body");

        let on_disk_etag = tokio::fs::read_to_string(&etag_path).await.unwrap();
        let (etag, ts) = parse_etag_file(&on_disk_etag).unwrap();
        assert_eq!(etag.as_deref(), Some("W/\"v2\""));
        assert!(
            ts >= before && ts <= after,
            "fetched_at {ts} must fall within [{before}, {after}]"
        );
    }

    #[tokio::test]
    async fn refresh_304_bumps_timestamp_only() {
        use wiremock::matchers::{header, method, path};
        use wiremock::{Mock, MockServer, ResponseTemplate};

        let dir = tempdir().unwrap();
        let disk_path = dir.path().join("codex_instructions.md");
        let etag_path = dir.path().join("codex_instructions.etag");

        let server = MockServer::start().await;
        Mock::given(method("GET"))
            .and(path("/prompt.md"))
            .and(header("If-None-Match", "W/v1"))
            .respond_with(ResponseTemplate::new(304))
            .expect(1)
            .mount(&server)
            .await;

        let store = InstructionsStore::new_for_test_with_paths_and_url(
            disk_path.clone(),
            etag_path.clone(),
            format!("{}/prompt.md", server.uri()),
        );

        // Pre-seed: body="old", etag="W/v1", fetched_at=0 (sentinel forces a
        // fetch and we expect the conditional GET to come back 304).
        seed_cache(&store, "old", Some("W/v1"), 0).await;

        let before = now_secs();
        store.maybe_refresh().await;
        let after = now_secs();

        // Body unchanged on disk and in cache.
        assert_eq!(store.get(None).await, "old");
        let on_disk_body = tokio::fs::read_to_string(&disk_path).await.unwrap();
        assert_eq!(on_disk_body, "old");

        // Side-car rewritten: same etag, advanced timestamp.
        let on_disk_etag = tokio::fs::read_to_string(&etag_path).await.unwrap();
        let (etag, ts) = parse_etag_file(&on_disk_etag).unwrap();
        assert_eq!(etag.as_deref(), Some("W/v1"));
        assert!(
            ts >= before && ts <= after,
            "fetched_at {ts} must fall within [{before}, {after}] after 304"
        );
        assert!(ts > 0, "304 must advance fetched_at past the 0 sentinel");
    }

    #[tokio::test]
    async fn refresh_network_error_keeps_cache() {
        let dir = tempdir().unwrap();
        let disk_path = dir.path().join("codex_instructions.md");
        let etag_path = dir.path().join("codex_instructions.etag");

        // Port 1 is reserved (tcpmux) and not bound in dev environments —
        // this guarantees a network-level connect failure without taking a
        // real listener offline.
        let store = InstructionsStore::new_for_test_with_paths_and_url(
            disk_path.clone(),
            etag_path.clone(),
            "http://127.0.0.1:1/prompt.md".to_string(),
        );

        seed_cache(&store, "stable", Some("W/locked"), 12_345).await;

        // Must not panic; cache must survive the error path.
        store.maybe_refresh().await;

        assert_eq!(store.get(None).await, "stable");
        let on_disk_body = tokio::fs::read_to_string(&disk_path).await.unwrap();
        assert_eq!(on_disk_body, "stable");

        let on_disk_etag = tokio::fs::read_to_string(&etag_path).await.unwrap();
        let (etag, ts) = parse_etag_file(&on_disk_etag).unwrap();
        assert_eq!(etag.as_deref(), Some("W/locked"));
        assert_eq!(ts, 12_345, "network error must not bump fetched_at");
    }

    #[tokio::test]
    async fn refresh_skips_when_recent() {
        use wiremock::matchers::{method, path};
        use wiremock::{Mock, MockServer, ResponseTemplate};

        let dir = tempdir().unwrap();
        let disk_path = dir.path().join("codex_instructions.md");
        let etag_path = dir.path().join("codex_instructions.etag");

        let server = MockServer::start().await;
        // Any hit at all is a regression — would force a 500 if invoked.
        Mock::given(method("GET"))
            .and(path("/prompt.md"))
            .respond_with(ResponseTemplate::new(500))
            .expect(0)
            .mount(&server)
            .await;

        let store = InstructionsStore::new_for_test_with_paths_and_url(
            disk_path.clone(),
            etag_path.clone(),
            format!("{}/prompt.md", server.uri()),
        );

        // Pre-seed with `fetched_at = now` so the 24-hour gate skips the
        // request entirely.
        seed_cache(&store, "cached", Some("W/recent"), now_secs()).await;

        store.maybe_refresh().await;

        // Body must be unchanged; mock's `.expect(0)` is asserted on drop
        // when the server is dropped at end of test.
        assert_eq!(store.get(None).await, "cached");
    }

    #[tokio::test]
    async fn refresh_forces_when_fetched_at_zero() {
        use wiremock::matchers::{method, path};
        use wiremock::{Mock, MockServer, ResponseTemplate};

        let dir = tempdir().unwrap();
        let disk_path = dir.path().join("codex_instructions.md");
        let etag_path = dir.path().join("codex_instructions.etag");

        let server = MockServer::start().await;
        Mock::given(method("GET"))
            .and(path("/prompt.md"))
            .respond_with(
                ResponseTemplate::new(200)
                    .insert_header("ETag", "W/\"v3\"")
                    .set_body_string("startup body"),
            )
            .expect(1)
            .mount(&server)
            .await;

        let store = InstructionsStore::new_for_test_with_paths_and_url(
            disk_path.clone(),
            etag_path.clone(),
            format!("{}/prompt.md", server.uri()),
        );

        // The default constructor seeds `fetched_at = 0`, so even though
        // wall-clock arithmetic `now - 0` is technically much greater than
        // the 24h gate, we want to confirm the sentinel branch in
        // `maybe_refresh` actually issues the request rather than relying
        // on the gate happening to be exceeded.
        assert_eq!(store.get(None).await, BUNDLED_PROMPT);

        store.maybe_refresh().await;

        assert_eq!(store.get(None).await, "startup body");
        let (etag, _ts) =
            parse_etag_file(&tokio::fs::read_to_string(&etag_path).await.unwrap()).unwrap();
        assert_eq!(etag.as_deref(), Some("W/\"v3\""));
    }
}
