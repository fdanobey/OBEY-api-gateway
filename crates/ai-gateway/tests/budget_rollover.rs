//! Off-path spend accrual + lazy window rollover (persistent budgets,
//! Task 4).
//!
//! Exercises the full gateway HTTP surface via `tower::ServiceExt::oneshot()`
//! with wiremock providers — no real ports bound, isolated SQLite databases.
//! `budgets.db` lives next to logs.db (Task 3 placement), so
//! `common::isolate_databases` isolates it too.
//!
//! Covers:
//! - (a) accrual: a cost-attributed request persists spend to the store,
//!   including for providers WITHOUT a configured limit (Req 1.3);
//! - (b) lazy rollover: an expired Daily window zeroes both the metrics
//!   spend atomics and the store row BEFORE the budget_skip decision, so
//!   an exhausted provider becomes eligible again;
//! - (c) Manual policy never rolls over.

use axum::body::Body;
use axum::http::{Request, StatusCode};
use proptest::prelude::*;
use tower::ServiceExt;
use wiremock::matchers::{method as wm_method, path as wm_path};
use wiremock::{Mock, MockServer, ResponseTemplate};

use ai_gateway::config::*;
use ai_gateway::gateway::GatewayServer;

const PROVIDER_NAME: &str = "budget-provider";

/// Build the base config with databases redirected into `db_dir`
/// (mirrors `common::isolate_databases`, but keeps the directory handle so
/// the rollover tests can address `budgets.db` directly for backdating).
fn isolated_config(db_dir: &std::path::Path) -> Config {
    let mut config = base_config();
    config.logging.database_path = db_dir.join("logs.db").to_string_lossy().into_owned();
    config.virtual_keys.database_path = db_dir.join("keys.db").to_string_lossy().into_owned();
    config
}

fn base_config() -> Config {
    Config {
        security: Default::default(),
        cache_aware_routing: Default::default(),
        server: ServerConfig {
            host: "127.0.0.1".to_string(),
            port: 0,
            request_timeout_seconds: 30,
            max_request_size_mb: 10,
        },
        tls: None,
        admin: AdminConfig::default(),
        dashboard: DashboardConfig::default(),
        cors: CorsConfig::default(),
        providers: vec![],
        model_groups: vec![],
        circuit_breaker: CircuitBreakerConfig::default(),
        retry: RetryConfig::default(),
        logging: LoggingConfig::default(),
        semantic_cache: None,
        exact_cache: ExactCacheConfig::default(),
        prometheus: None,
        context: ContextConfig::default(),
        compression: Default::default(),
        memory: None,
        first_launch_completed: false,
        tray: TrayConfig::default(),
        codex_instructions_url: None,
        streaming: None,
        virtual_keys: Default::default(),
        loop_detection: Default::default(),
        guardrails: None,
        tool_compression: Default::default(),
        smart_routing: Default::default(),
        xhigh_models_allowlist: Default::default(),
        reasoning_models_allowlist: Default::default(),
        codex_search: None,
        structured_output: None,
        reasoning_compat: Default::default(),
    }
}

fn provider_with_budget(uri: &str, budget: Option<ProviderBudgetConfig>) -> Provider {
    Provider {
        name: PROVIDER_NAME.to_string(),
        provider_type: "openai".to_string(),
        base_url: Some(uri.to_string()),
        api_key_env: None,
        api_key_encrypted: None,
        api_secret_env: None,
        api_secret_encrypted: None,
        auth_method: None,
        resolved_api_key: None,
        resolved_api_secret: None,
        region: None,
        timeout_seconds: 30,
        ttfb_timeout_seconds: None,
        total_timeout_seconds: None,
        buffered_upstream_streaming: None,
        rate_limit_max_wait_ms: None,
        max_connections: 10,
        rate_limit_per_minute: 0,
        custom_headers: Default::default(),
        user_agent: None,
        connection_pool: ProviderConnectionPoolConfig::default(),
        budget,
        manual_models: vec![],
        global_inference_profile: false,
        cross_region_inference: false,
        custom_vpc_endpoint: false,
        prompt_caching: false,
        compression: None,
        reasoning: true,
        codex_base_url_override: None,
        codex_model_override: None,
        instructions_override: None,
        max_rate_limit_cooldown_seconds: None,
        memory: None,
    }
}

/// $1/M input, $2/M output → 1M prompt + 1M completion = $3.00 per call.
fn expensive_model() -> ProviderModel {
    ProviderModel {
        provider: PROVIDER_NAME.to_string(),
        model: "gpt-test".to_string(),
        cost_per_million_input_tokens: 1.0,
        cost_per_million_output_tokens: 2.0,
        cost_per_million_cache_read_input_tokens: None,
        cost_per_million_cache_creation_input_tokens: None,
        cache_min_tokens: None,
        cache_support: None,
        structured_output_passthrough: None,
        priority: 100,
        tier: None,
        context_window: 0,
        specializations: vec![],
        cost_per_million_reasoning_tokens: None,
        reasoning_family: None,
        reasoning_parameter: None,
    }
}

fn chat_body(turn: &str) -> Body {
    Body::from(
        serde_json::to_string(&serde_json::json!({
            "model": "budget-group",
            "messages": [{"role": "user", "content": format!("hello {turn}")}]
        }))
        .unwrap(),
    )
}

/// Mount a mock chat-completions endpoint returning $3.00 of usage
/// (1M prompt + 1M completion at $1/M + $2/M). `expect` counts expected
/// upstream hits (wiremock asserts on drop).
async fn mount_ok_mock(upstream: &MockServer, expect: u64) {
    let body = serde_json::json!({
        "id": "chatcmpl-budget-test",
        "object": "chat.completion",
        "created": 1700000000_i64,
        "model": "gpt-test",
        "choices": [{
            "index": 0,
            "message": {"role": "assistant", "content": "ok"},
            "finish_reason": "stop"
        }],
        "usage": {
            "prompt_tokens": 1_000_000,
            "completion_tokens": 1_000_000,
            "total_tokens": 2_000_000
        }
    });
    Mock::given(wm_method("POST"))
        .and(wm_path("/v1/chat/completions"))
        .respond_with(ResponseTemplate::new(200).set_body_json(body))
        .expect(expect)
        .mount(upstream)
        .await;
}

/// Distinct turns per call so the exact-match response cache can never
/// short-circuit the dispatch (each body hashes differently).
async fn post_chat(app: axum::Router, turn: &str) -> StatusCode {
    let req = Request::post("/v1/chat/completions")
        .header("content-type", "application/json")
        .body(chat_body(turn))
        .unwrap();
    let resp = app.oneshot(req).await.unwrap();
    resp.status()
}

/// Poll the store until `provider`'s durable spend reaches at least
/// `min_usd` (the accrual write is spawned off the response path).
async fn wait_for_store_spend(server: &GatewayServer, provider: &str, min_usd: f64) -> f64 {
    let store = server
        .state
        .metrics
        .provider_budget_store()
        .expect("store attached");
    for _ in 0..100 {
        if let Ok(rows) = store.get_all() {
            if let Some(row) = rows.iter().find(|row| row.provider == provider) {
                if row.current_spend_usd >= min_usd {
                    return row.current_spend_usd;
                }
            }
        }
        tokio::time::sleep(std::time::Duration::from_millis(50)).await;
    }
    panic!("durable spend for {provider} never reached {min_usd} USD");
}

/// Backdate `provider`'s durable window_start to a fixed instant well
/// before every rollover boundary (task-4 test hook: simulates an expired
/// window without waiting on wall-clock time). Opens its own short-lived
/// SQLite connection to the isolated budgets.db (WAL mode allows the
/// second writer); the row is created (zero spend) when absent.
fn backdate_window_start(
    server: &GatewayServer,
    budgets_db: &std::path::Path,
    provider: &str,
    epoch: i64,
) {
    let store = server
        .state
        .metrics
        .provider_budget_store()
        .expect("store attached");
    // Ensure the row exists, then backdate the window anchor.
    // Anchor the insert at the (soon-backdated) epoch so the row is born
    // with the stale anchor the test then rolls from.
    store.add_spend(provider, 0.0, epoch).expect("row created");
    let conn = rusqlite::Connection::open(budgets_db).expect("open budgets.db");
    conn.execute(
        "UPDATE provider_budgets SET window_start = ?2 WHERE provider = ?1",
        rusqlite::params![provider, epoch],
    )
    .expect("backdate window_start");
}

// ---------------------------------------------------------------------------
// (a) Accrual persists to the durable store after a cost-attributed request
// ---------------------------------------------------------------------------

/// A successful request accrues spend into the store — including for a
/// provider WITHOUT a configured limit (Req 1.3: persist accrual for all
/// providers; limits are metadata).
#[tokio::test]
async fn cost_attributed_request_persists_spend_to_store() {
    let upstream = MockServer::start().await;
    let db_dir = tempfile::tempdir().expect("temp dir");

    let mut config = isolated_config(db_dir.path());
    // No budget configured — spend-only row must still be persisted.
    config.providers = vec![provider_with_budget(&upstream.uri(), None)];
    config.model_groups = vec![ModelGroup {
        name: "budget-group".to_string(),
        version_fallback_enabled: false,
        compression: None,
        structured_output: None,
        memory: None,
        models: vec![expensive_model()],
    }];
    let server = GatewayServer::new(config, None).await.unwrap();
    let app = server.build_router();

    mount_ok_mock(&upstream, 1).await;
    let status = post_chat(app, "one").await;
    assert_eq!(status, StatusCode::OK);

    // 1M prompt @ $1/M + 1M completion @ $2/M = $3.00; the write is
    // off-path, so poll until it lands.
    let spend = wait_for_store_spend(&server, PROVIDER_NAME, 3.0).await;
    assert!(
        (spend - 3.0).abs() < 1e-6,
        "expected durable spend of $3.00, got {spend}"
    );
    // In-memory truth accrued synchronously on the response path.
    let in_memory = server
        .state
        .metrics
        .current_provider_cost_usd(PROVIDER_NAME);
    assert!(
        (in_memory - 3.0).abs() < 1e-6,
        "expected in-memory spend of $3.00, got {in_memory}"
    );
}

// ---------------------------------------------------------------------------
// (b) Lazy rollover: expired Daily window zeroes spend before the check
// ---------------------------------------------------------------------------

/// Daily policy + expired window: the budget check lazily rolls the window
/// (metrics atomics AND store row) before deciding, so an over-limit
/// provider is NOT skipped and the request succeeds.
#[tokio::test]
async fn provider_budget_window_resets_lazily() {
    let upstream = MockServer::start().await;
    let db_dir = tempfile::tempdir().expect("temp dir");
    let budgets_db = db_dir.path().join("budgets.db");

    let mut config = isolated_config(db_dir.path());
    config.providers = vec![provider_with_budget(
        &upstream.uri(),
        Some(ProviderBudgetConfig {
            // Two $3.00 requests exhaust a $5.00 budget within a window.
            limit_usd: 5.0,
            reset_policy: Some(BudgetResetPolicy::Daily),
        }),
    )];
    config.model_groups = vec![ModelGroup {
        name: "budget-group".to_string(),
        version_fallback_enabled: false,
        compression: None,
        structured_output: None,
        memory: None,
        models: vec![expensive_model()],
    }];
    let server = GatewayServer::new(config, None).await.unwrap();
    let app = server.build_router();

    // Exhaust the window: $6.00 of in-memory + durable spend.
    mount_ok_mock(&upstream, 2).await;
    let status = post_chat(app.clone(), "one").await;
    assert_eq!(status, StatusCode::OK);
    let spend = wait_for_store_spend(&server, PROVIDER_NAME, 3.0).await;
    assert!((spend - 3.0).abs() < 1e-6);
    let store = server
        .state
        .metrics
        .provider_budget_store()
        .expect("store attached");
    let anchor = store
        .get_window_start(PROVIDER_NAME)
        .expect("window start readable")
        .expect("row exists after first request");
    store
        .add_spend(PROVIDER_NAME, 3.0, anchor)
        .expect("durable $6.00");
    server.state.metrics.add_cost(PROVIDER_NAME, 3.0); // in-memory $6.00 ≥ $5.00

    // Backdate the durable window anchor to yesterday → window expired.
    let yesterday = chrono::Utc::now().timestamp() - 86_400;
    backdate_window_start(&server, &budgets_db, PROVIDER_NAME, yesterday);

    // With an expired window the lazy rollover must run BEFORE the skip
    // decision: spend resets to $0, the $5.00 limit is no longer met, and
    // the request succeeds instead of budget_skip 502.
    let status = post_chat(app, "two").await;
    assert_eq!(
        status,
        StatusCode::OK,
        "expired daily window must roll over before the budget check"
    );

    // Store row was rolled: spend reflects only the post-rollover request.
    let spend = wait_for_store_spend(&server, PROVIDER_NAME, 3.0).await;
    assert!(
        (spend - 3.0).abs() < 1e-6,
        "post-rollover durable spend must be $3.00 (fresh window), got {spend}"
    );
    let in_memory = server
        .state
        .metrics
        .current_provider_cost_usd(PROVIDER_NAME);
    assert!(
        (in_memory - 3.0).abs() < 1e-6,
        "post-rollover in-memory spend must be $3.00, got {in_memory}"
    );
    // No exhaustion was recorded for the rolled request.
    let snap = server.state.metrics.snapshot();
    let exhausted = snap
        .budget_exhaustions_by_provider
        .iter()
        .find(|(provider, _)| provider == PROVIDER_NAME)
        .map(|(_, count)| *count)
        .unwrap_or(0);
    assert_eq!(exhausted, 0, "rolled provider must not be budget_skipped");
}

// ---------------------------------------------------------------------------
// (c) Manual policy never rolls over
// ---------------------------------------------------------------------------

/// Manual policy + expired-by-wall-clock window anchor: no rollover, the
/// exhausted provider is skipped exactly as before (budget_skip
/// semantics unchanged).
#[tokio::test]
async fn manual_policy_never_rolls_over() {
    let upstream = MockServer::start().await;
    let db_dir = tempfile::tempdir().expect("temp dir");
    let budgets_db = db_dir.path().join("budgets.db");

    let mut config = isolated_config(db_dir.path());
    config.providers = vec![provider_with_budget(
        &upstream.uri(),
        Some(ProviderBudgetConfig {
            limit_usd: 5.0,
            reset_policy: Some(BudgetResetPolicy::Manual),
        }),
    )];
    config.model_groups = vec![ModelGroup {
        name: "budget-group".to_string(),
        version_fallback_enabled: false,
        compression: None,
        structured_output: None,
        memory: None,
        models: vec![expensive_model()],
    }];
    let server = GatewayServer::new(config, None).await.unwrap();
    let app = server.build_router();

    // Accrue durable spend and backdate the anchor — even a week-old
    // Manual window must not roll.
    server.state.metrics.add_cost(PROVIDER_NAME, 6.0); // ≥ limit in-memory
    backdate_window_start(
        &server,
        &budgets_db,
        PROVIDER_NAME,
        chrono::Utc::now().timestamp() - 604_800,
    );

    let status = post_chat(app, "manual").await;
    assert_eq!(
        status,
        StatusCode::BAD_GATEWAY,
        "manual policy with exhausted budget must still budget_skip (502)"
    );
    let in_memory = server
        .state
        .metrics
        .current_provider_cost_usd(PROVIDER_NAME);
    assert!(
        (in_memory - 6.0).abs() < 1e-6,
        "manual policy must never roll in-memory spend, got {in_memory}"
    );
    let store = server
        .state
        .metrics
        .provider_budget_store()
        .expect("store attached");
    let rows = store.get_all().expect("store rows");
    let row = rows
        .iter()
        .find(|row| row.provider == PROVIDER_NAME)
        .expect("row present");
    assert!(
        row.current_spend_usd.abs() < 1e-9,
        "manual policy must never roll durable spend, got ${}",
        row.current_spend_usd
    );
    let snap = server.state.metrics.snapshot();
    let exhausted = snap
        .budget_exhaustions_by_provider
        .iter()
        .find(|(provider, _)| provider == PROVIDER_NAME)
        .map(|(_, count)| *count)
        .unwrap_or(0);
    assert_eq!(
        exhausted, 1,
        "exhausted manual budget must record budget_skip"
    );
}

// ---------------------------------------------------------------------------
// (d) Guard (Task 10): provider window boundaries == VK budget.rs boundaries
// ---------------------------------------------------------------------------

// Guard companion to `provider_budget_window_resets_lazily`: the provider
// next-reset boundary must EQUAL the virtual-key budget semantics for
// daily/weekly/monthly across varied timestamps — daily 00:00 UTC, weekly
// Monday 00:00 UTC, monthly first-of-month 00:00 UTC. The provider path
// (`Metrics::next_reset_epoch`) and the VK path
// (`virtual_keys::budget::window_expired` anchored one second earlier)
// must never disagree, or the two budget systems would roll at different
// instants.
proptest! {
#![proptest_config(proptest::prelude::ProptestConfig::with_cases(64))]

#[test]
fn provider_window_boundaries_equal_virtual_key_boundaries(
year in 2024i32..=2030u32 as i32,
day_of_year in 1u32..=365,
hour in 0u32..=23,
minute in 0u32..=59,
) {
        use ai_gateway::config::BudgetResetPolicy;
        use ai_gateway::virtual_keys::budget::window_expired;
        use ai_gateway::virtual_keys::models::BudgetWindow;
        use chrono::Datelike as _;

// Any instant inside the year (leap-day overflow falls back to Jan 1).
let start = chrono::NaiveDate::from_yo_opt(year, day_of_year)
.unwrap_or_else(|| chrono::NaiveDate::from_ymd_opt(year, 1, 1).unwrap())
.and_hms_opt(hour, minute, 0)
.unwrap()
.and_utc();

for (policy, window) in [
(BudgetResetPolicy::Daily, BudgetWindow::Daily),
(BudgetResetPolicy::Weekly, BudgetWindow::Weekly),
(BudgetResetPolicy::Monthly, BudgetWindow::Monthly),
] {
let boundary_epoch = ai_gateway::metrics::next_reset_epoch(
policy,
start.timestamp(),
)
.expect("windowed policy always yields a boundary");
let boundary =
chrono::DateTime::from_timestamp(boundary_epoch, 0).unwrap();

// Boundary sanity: strictly after the anchor, within one period.
prop_assert!(boundary > start);
prop_assert!(boundary - start <= chrono::Duration::days(32));

// Exactly VK equivalence: one second before the boundary the VK
// window is still active; at the boundary it has expired. The
// provider boundary is thus the SAME instant the VK logic rolls.
prop_assert!(!window_expired(
&window,
start,
boundary - chrono::Duration::seconds(1)
));
prop_assert!(window_expired(&window, start, boundary));

// The boundary itself lands on the canonical UTC edge.
let boundary_naive = boundary.naive_utc();
prop_assert_eq!(boundary_naive.time(), chrono::NaiveTime::MIN);
            match window {
                BudgetWindow::Daily => {}
                BudgetWindow::Weekly => {
                    prop_assert_eq!(boundary.weekday(), chrono::Weekday::Mon);
                }
                BudgetWindow::Monthly => {
                    prop_assert_eq!(boundary.day(), 1);
                }
            }
}
}
}
