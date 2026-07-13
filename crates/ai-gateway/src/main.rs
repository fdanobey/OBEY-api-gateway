#![cfg_attr(all(target_os = "windows", feature = "tray"), windows_subsystem = "windows")]

use clap::Parser;
use std::path::PathBuf;
use tracing_subscriber::{layer::SubscriberExt, util::SubscriberInitExt};

mod config;
mod context;
mod providers;
mod router;
mod cache;
mod codex;
mod logger;
mod admin;
mod dashboard;
mod models;
mod error;
mod gateway;
mod guardrail;
mod metrics;
mod oauth;
mod secrets;
mod virtual_keys;
#[cfg(feature = "tray")]
mod tray;

#[derive(Parser, Debug)]
#[command(name = "ai-gateway")]
#[command(about = "OBEY-API: OpenAI-compatible AI gateway with intelligent routing", long_about = None)]
struct Cli {
    /// Path to configuration file
    #[arg(short, long, value_name = "FILE")]
    config: Option<PathBuf>,
}

#[tokio::main]
async fn main() -> anyhow::Result<()> {
    // Initialize tracing
    tracing_subscriber::registry()
        .with(
            tracing_subscriber::EnvFilter::try_from_default_env()
                .unwrap_or_else(|_| "debug".into()),
        )
        .with(tracing_subscriber::fmt::layer().with_target(false))
        .init();

    tracing::info!("OBEY-API Gateway starting...");

    // In tray mode the working directory may not be the exe's directory
    // (e.g. when launched from a Start Menu shortcut). Normalise it so
    // that relative paths in config.yaml (logs.db, certs, etc.) resolve
    // correctly.
    #[cfg(all(target_os = "windows", feature = "tray"))]
    {
        if let Ok(exe) = std::env::current_exe() {
            if let Some(dir) = exe.parent() {
                let _ = std::env::set_current_dir(dir);
            }
        }
    }
    
    // Parse CLI arguments
    let cli = Cli::parse();
    
    // Resolve config path
    let config_path = config::resolve_config_path(cli.config);
    tracing::info!("Loading configuration from: {}", config_path.display());

    if config::bootstrap_config_if_missing(&config_path)
        .map_err(|error| anyhow::anyhow!(error))?
    {
        tracing::info!("Created default configuration at {}", config_path.display());
    }
    
    // Load and validate configuration
    let config = match config::load_and_validate_config(&config_path) {
        Ok(cfg) => cfg,
        Err(e) => {
            tracing::error!("{}", e);
            std::process::exit(1);
        }
    };
    
    tracing::info!("Configuration loaded successfully");

    // Each configuration compiles exactly one of the following blocks as the
    // function's tail expression, so there is no unreachable trailing
    // expression under the `tray` feature.
    #[cfg(feature = "tray")]
    {
        run_tray_mode(config, config_path).await
    }

    #[cfg(not(feature = "tray"))]
    {
        // Create and start the gateway server
        tracing::info!("Server will listen on {}:{}", config.server.host, config.server.port);
        let server = gateway::GatewayServer::new(config, Some(config_path)).await
            .map_err(|e| {
                tracing::error!("Failed to initialize gateway: {}", e);
                anyhow::anyhow!("{}", e)
            })?;

        tracing::info!("Gateway initialized, starting HTTP server...");
        server.start().await.map_err(|e| {
            tracing::error!("Gateway server error: {}", e);
            anyhow::anyhow!("{}", e)
        })?;

        Ok(())
    }
}

#[cfg(feature = "tray")]
async fn run_tray_mode(config: config::Config, config_path: PathBuf) -> anyhow::Result<()> {
    let instance_guard = tray::SingleInstanceGuard::acquire("obey-api-gateway")?;
    if instance_guard.is_already_running() {
        tray::NotificationManager::notify_already_running()?;
        instance_guard.bring_to_front();
        return Ok(());
    }

    let config = tray::prepare_startup_config(config, &config_path)
        .map_err(|error| anyhow::anyhow!(error.to_string()))?;

    tracing::info!("Server will listen on {}:{}", config.server.host, config.server.port);

    let mut tray_manager = tray::TrayManager::new(config.clone()).await?;
    tray_manager.attach_instance_guard(instance_guard);

    let initial_status = tray_manager.server_status().await;
    tracing::info!(
        config_path = %config_path.display(),
        bind_host = %config.server.host,
        bind_port = config.server.port,
        admin_enabled = config.admin.enabled,
        admin_path = %config.admin.path,
        dashboard_enabled = config.dashboard.enabled,
        dashboard_path = %config.dashboard.path,
        admin_url = %initial_status.admin_url(),
        dashboard_url = %initial_status.dashboard_url(),
        auto_open_browser = config.tray.auto_open_browser,
        first_launch_completed = config.first_launch_completed,
        "Tray mode resolved configuration and browser URLs"
    );
    if tray::client_host_for_bind_host(&config.server.host) != config.server.host {
        tracing::info!(
            bind_host = %config.server.host,
            admin_url = %initial_status.admin_url(),
            dashboard_url = %initial_status.dashboard_url(),
            "Tray browser URLs are using a loopback host because the configured bind host is a wildcard address"
        );
    }

    let shutdown = std::sync::Arc::new(tokio::sync::Notify::new());
    let shutdown_for_server = shutdown.clone();

    let server = gateway::GatewayServer::new(config.clone(), Some(config_path.clone()))
        .await
        .map_err(|e| {
            tracing::error!("Failed to initialize tray-mode gateway: {}", e);
            anyhow::anyhow!("{}", e)
        })?;

    let server_task = tokio::spawn(async move {
        let result = server
            .start_with_shutdown(async move {
                shutdown_for_server.notified().await;
            })
            .await;

        match &result {
            Ok(()) => tracing::info!("Tray-mode gateway server task exited cleanly"),
            Err(error) => tracing::error!(%error, "Tray-mode gateway server task exited with an error"),
        }

        result
    });

    tray_manager.set_server_running(true).await;
    tracing::info!("Tray manager marked the server as running while the HTTP server task is still starting");

    if tray_manager.is_first_launch().await {
        tray_manager.show_first_launch_experience().await?;
        if config.tray.auto_open_browser {
            tracing::info!("First launch detected; attempting automatic dashboard browser launch");
            if let Err(error) = tray_manager.open_dashboard().await {
                tracing::warn!(%error, "Failed to open dashboard automatically on first launch");
            }
        }

        let mut persisted = config.clone();
        persisted.first_launch_completed = true;
        config::save_config(&config_path, &persisted)
            .map_err(|error| anyhow::anyhow!(error))?;
        tray_manager.mark_first_launch_complete().await;
    }

    tokio::select! {
        result = tray_manager.run() => {
            result?;
        }
        signal = tokio::signal::ctrl_c() => {
            if let Err(error) = signal {
                tracing::warn!(%error, "Failed to listen for Ctrl+C in tray mode");
            }
            tray_manager.request_shutdown();
        }
    }

    shutdown.notify_waiters();

    match server_task.await {
        Ok(Ok(())) => {}
        Ok(Err(error)) => return Err(anyhow::anyhow!("{}", error)),
        Err(error) => return Err(anyhow::anyhow!("Tray-mode server task failed: {}", error)),
    }

    Ok(())
}
