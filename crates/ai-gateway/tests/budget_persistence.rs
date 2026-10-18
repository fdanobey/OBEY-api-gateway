//! Persistent provider-budget store wiring (persistent-budgets, Task 3).
//!
//! `budgets.db` is a sibling of the logs database; on startup the
//! `GatewayServer` seeds configured limits into metrics and restores
//! persisted per-provider spend from the store into the cost atomics
//! (store value wins — durable truth). These tests prove the
//! rebuild-from-same-DB path end to end through `GatewayServer::new`.
//!
//! Unlike `common::isolate_databases` (a fresh temp dir per config), these
//! tests need *stable* database paths across two `GatewayServer::new`
//! calls, so they manage a shared temp directory directly — the same
//! pattern as `virtual_keys_integration` / `token_compression_observability`.

mod common;

use ai_gateway::config::*;
use ai_gateway::gateway::GatewayServer;

use common::budget_test_config;

const PROVIDER_NAME: &str = "budget-provider";

/// `budgets.db` must land next to logs.db/keys.db and the configured limit
/// must be seeded into the metrics snapshot.
#[tokio::test]
async fn gateway_creates_budgets_db_sibling_of_logs_db_and_seeds_limit() {
    let dir = tempfile::tempdir().expect("temp dir");
    let config = budget_test_config(
        Some(dir.path()),
        Some(ProviderBudgetConfig {
            limit_usd: 50.0,
            reset_policy: None,
        }),
    );
    let server = GatewayServer::new(config, None)
        .await
        .expect("server builds");
    let budgets_db = dir.path().join("budgets.db");
    assert!(
        budgets_db.is_file(),
        "budgets.db must exist at {}",
        budgets_db.display()
    );
    let snap = server.state.metrics.snapshot();
    let (_, limit) = snap
        .budget_limit_by_provider
        .iter()
        .find(|(provider, _)| provider == PROVIDER_NAME)
        .expect("configured limit seeded");
    assert_eq!(*limit, 50.0);
}

/// Guard test (Task 10): rebuilding a server from the same `budgets.db`
/// restores persisted spend into the metrics atomics — durable store wins
/// over fresh atomics. Providers without a configured limit are included
/// (spend-only rows).
#[tokio::test]
async fn budget_persistence_survives_reboot() {
    let dir = tempfile::tempdir().expect("temp dir");

    // First boot: no limit configured (spend-only row), accrue $12.34 of
    // durable spend through the store the server attached to metrics.
    {
        let config = budget_test_config(Some(dir.path()), None);
        let server = GatewayServer::new(config, None)
            .await
            .expect("first server builds");
        let store = server
            .state
            .metrics
            .provider_budget_store()
            .expect("store attached to metrics");
        store
            .add_spend(PROVIDER_NAME, 12.34, chrono::Utc::now().timestamp())
            .expect("spend persisted");
    }

    // Second boot from the same directory: the persisted spend must be
    // restored into the fresh metrics atomics (write-through cache).
    {
        let config = budget_test_config(Some(dir.path()), None);
        let server = GatewayServer::new(config, None)
            .await
            .expect("second server builds");
        let restored = server
            .state
            .metrics
            .current_provider_cost_usd(PROVIDER_NAME);
        assert!(
            (restored - 12.34).abs() < 1e-9,
            "expected restored spend 12.34, got {restored}"
        );
        let snap = server.state.metrics.snapshot();
        let (_, cost) = snap
            .cost_by_provider
            .iter()
            .find(|(provider, _)| provider == PROVIDER_NAME)
            .expect("restored spend present in cost_by_provider");
        assert!((cost - 12.34).abs() < 1e-9);
    }
}

/// Config reload re-seeds limits from the reloaded config without
/// clearing spend: neither the durable store row nor the in-memory spend
/// atomics may be zeroed by a reload.
#[tokio::test]
async fn config_reload_reseeds_limits_without_clearing_store_spend() {
    let dir = tempfile::tempdir().expect("temp dir");
    let config = budget_test_config(
        Some(dir.path()),
        Some(ProviderBudgetConfig {
            limit_usd: 50.0,
            reset_policy: None,
        }),
    );
    let server = GatewayServer::new(config, None)
        .await
        .expect("server builds");
    let store = server
        .state
        .metrics
        .provider_budget_store()
        .expect("store attached");
    store
        .add_spend(PROVIDER_NAME, 5.0, chrono::Utc::now().timestamp())
        .expect("spend persisted");
    // Accrue the same spend into the in-memory atomics, exactly like a
    // cost-attributed request would — reload must preserve it too.
    server.state.metrics.add_cost(PROVIDER_NAME, 5.0);

    // Reload with a raised limit.
    let reloaded = budget_test_config(
        Some(dir.path()),
        Some(ProviderBudgetConfig {
            limit_usd: 75.0,
            reset_policy: None,
        }),
    );
    server
        .reload_config(reloaded)
        .await
        .expect("reload applies");

    let snap = server.state.metrics.snapshot();
    let (_, limit) = snap
        .budget_limit_by_provider
        .iter()
        .find(|(provider, _)| provider == PROVIDER_NAME)
        .expect("limit re-seeded after reload");
    assert_eq!(*limit, 75.0, "reload must re-seed limits from new config");

    // Store spend untouched by reload (durable truth preserved).
    let rows = store.get_all().expect("store rows readable");
    let row = rows
        .iter()
        .find(|row| row.provider == PROVIDER_NAME)
        .expect("provider row present");
    assert!((row.current_spend_usd - 5.0).abs() < 1e-9);
    // In-memory spend atomics survive the reload as well: reload re-seeds
    // limits only and must never zero accrued spend.
    let in_memory = server
        .state
        .metrics
        .current_provider_cost_usd(PROVIDER_NAME);
    assert!(
        (in_memory - 5.0).abs() < 1e-9,
        "reload must not clear in-memory spend atomics, got {in_memory}"
    );
}

/// A store that cannot be opened/migrated must abort startup (budgets are
/// a safety control) with an error naming the exact path and SQLite error.
/// A directory at the budgets.db location makes the SQLite open fail.
#[tokio::test]
async fn budget_store_open_failure_aborts_startup_with_actionable_error() {
    let dir = tempfile::tempdir().expect("temp dir");
    let budgets_db = dir.path().join("budgets.db");
    std::fs::create_dir(&budgets_db).expect("create blocking directory at budgets.db path");

    let config = budget_test_config(
        Some(dir.path()),
        Some(ProviderBudgetConfig {
            limit_usd: 50.0,
            reset_policy: None,
        }),
    );
    let err = match GatewayServer::new(config, None).await {
        Ok(_) => panic!("startup must abort on budgets.db open failure"),
        Err(err) => err.to_string().to_lowercase(),
    };
    assert!(
        err.contains("budget"),
        "error must name the budget store: {err}"
    );
    assert!(
        err.contains(&budgets_db.to_string_lossy().to_lowercase()),
        "error must name the exact path {}: {err}",
        budgets_db.display()
    );
}

/// Task 7: `MetricsSnapshot::budget_status_by_provider` derives
/// server-authoritative budget status for the dashboard. Daily policy gets
/// a next_reset_epoch at the next 00:00 UTC boundary; manual gets `None`;
/// providers without a configured limit appear spend-only (no fake limit or
/// remaining); exhaustion reflects spent >= limit.
#[tokio::test]
async fn snapshot_carries_budget_status_by_provider() {
    let dir = tempfile::tempdir().expect("temp dir");

    // daily: limit 50 with 20 accrued spend (in-window).
    let daily_dir = dir.path().join("daily");
    std::fs::create_dir_all(&daily_dir).expect("daily dir");
    let daily_config = budget_test_config(
        Some(&daily_dir),
        Some(ProviderBudgetConfig {
            limit_usd: 50.0,
            reset_policy: Some(BudgetResetPolicy::Daily),
        }),
    );
    let daily_server = GatewayServer::new(daily_config, None)
        .await
        .expect("daily server builds");
    let daily_store = daily_server
        .state
        .metrics
        .provider_budget_store()
        .expect("store attached");
    daily_store
        .add_spend(PROVIDER_NAME, 20.0, chrono::Utc::now().timestamp())
        .expect("daily spend persisted");
    daily_server.state.metrics.add_cost(PROVIDER_NAME, 20.0);

    let snap = daily_server.state.metrics.snapshot();
    let daily = snap
        .budget_status_by_provider
        .get(PROVIDER_NAME)
        .expect("daily provider present in budget_status_by_provider");
    assert_eq!(daily.limit_usd, Some(50.0));
    assert!((daily.spent_usd - 20.0).abs() < 1e-9);
    assert_eq!(daily.remaining_usd, Some(30.0));
    assert_eq!(daily.window, BudgetResetPolicy::Daily);
    assert!(!daily.exhausted, "20 of 50 must not be exhausted");
    let next_reset = daily
        .next_reset_epoch
        .expect("daily policy carries a next_reset_epoch");
    let now = std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .unwrap()
        .as_secs() as i64;
    assert!(
        next_reset > now && next_reset <= now + 24 * 3600,
        "daily next_reset_epoch {next_reset} must fall within the next 24h from {now}"
    );

    // manual + no limit: spend-only row, next_reset_epoch None.
    let manual_dir = dir.path().join("manual");
    std::fs::create_dir_all(&manual_dir).expect("manual dir");
    let manual_config = budget_test_config(Some(&manual_dir), None);
    let manual_server = GatewayServer::new(manual_config, None)
        .await
        .expect("manual server builds");
    let manual_store = manual_server
        .state
        .metrics
        .provider_budget_store()
        .expect("store attached");
    manual_store
        .add_spend(PROVIDER_NAME, 12.34, chrono::Utc::now().timestamp())
        .expect("manual spend persisted");
    manual_server.state.metrics.add_cost(PROVIDER_NAME, 12.34);

    let snap = manual_server.state.metrics.snapshot();
    let manual = snap
        .budget_status_by_provider
        .get(PROVIDER_NAME)
        .expect("manual provider present in budget_status_by_provider");
    assert_eq!(
        manual.limit_usd, None,
        "no configured limit must not invent one"
    );
    assert!((manual.spent_usd - 12.34).abs() < 1e-9);
    assert_eq!(manual.remaining_usd, None);
    assert_eq!(manual.window, BudgetResetPolicy::Manual);
    assert_eq!(manual.next_reset_epoch, None, "manual policy never resets");
    assert!(!manual.exhausted);

    // Exhaustion: spend past a configured limit flips the flag.
    manual_server
        .state
        .metrics
        .set_provider_budget_limit(PROVIDER_NAME, 10.0);
    let s = manual_server.state.metrics.snapshot();
    let status = s
        .budget_status_by_provider
        .get(PROVIDER_NAME)
        .expect("status present with seeded limit");
    assert!(
        status.exhausted,
        "12.34 spent against a 10.0 limit must read exhausted"
    );
    assert_eq!(status.limit_usd, Some(10.0));
}
