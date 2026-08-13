#![cfg_attr(
    all(target_os = "windows", feature = "tray"),
    windows_subsystem = "windows"
)]

use clap::{Parser, Subcommand, ValueEnum};
use std::{
    fs::{self, File},
    io::Read,
    path::{Path, PathBuf},
};
use tracing_subscriber::{layer::SubscriberExt, util::SubscriberInitExt};

mod active_requests;
mod admin;
mod cache;
mod codex;
#[allow(dead_code, unused_imports)]
mod compression;
mod config;
mod context;
mod dashboard;
mod description_utils;
mod error;
mod gateway;
mod guardrail;
mod logger;
#[allow(dead_code, unused_imports)]
mod loop_detection;
#[allow(dead_code, unused_imports)]
mod memory;
mod metrics;
mod models;
mod oauth;
mod providers;
mod router;
mod secrets;
#[allow(dead_code, unused_imports)]
mod smart_routing;
#[allow(dead_code, unused_imports)]
mod structured_output;
#[allow(dead_code, unused_imports)]
mod tool_compression;
#[cfg(feature = "tray")]
mod tray;
mod virtual_keys;

#[derive(Parser, Debug)]
#[command(name = "ai-gateway")]
#[command(about = "OBEY-API: OpenAI-compatible AI gateway with intelligent routing", long_about = None)]
struct Cli {
    /// Path to configuration file
    #[arg(short, long, value_name = "FILE", global = true)]
    config: Option<PathBuf>,

    #[command(subcommand)]
    command: Option<Command>,
}

#[derive(Subcommand, Debug)]
enum Command {
    /// Pre-compress a context file for later gateway use
    CompressContext {
        /// Natural-language context file to compress
        input: PathBuf,
        /// Destination for the compressed artifact
        output: PathBuf,
        /// Compression level (defaults to compression.default_level)
        #[arg(long, value_enum)]
        level: Option<CliCompressionLevel>,
        /// Replace an existing output artifact and metadata sidecar
        #[arg(long)]
        force: bool,
    },
}

#[derive(Clone, Copy, Debug, ValueEnum)]
enum CliCompressionLevel {
    Lite,
    Standard,
    Aggressive,
    Ultra,
    Rtk,
    Stacked,
}

impl From<CliCompressionLevel> for compression::CompressionLevel {
    fn from(level: CliCompressionLevel) -> Self {
        match level {
            CliCompressionLevel::Lite => Self::Lite,
            CliCompressionLevel::Standard => Self::Standard,
            CliCompressionLevel::Aggressive => Self::Aggressive,
            CliCompressionLevel::Ultra => Self::Ultra,
            CliCompressionLevel::Rtk => Self::Rtk,
            CliCompressionLevel::Stacked => Self::Stacked,
        }
    }
}

const MAX_CONTEXT_BYTES: u64 = compression::precompressed::DEFAULT_MAX_CONTEXT_FILE_BYTES;
const OFFLINE_TIME_BUDGET_MS: u64 = 60_000;
const OFFLINE_MODEL: &str = "gpt-4o";

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

    if config::bootstrap_config_if_missing(&config_path).map_err(|error| anyhow::anyhow!(error))? {
        tracing::info!("Created default configuration at {}", config_path.display());
    }

    // Load and validate configuration
    let config = config::load_and_validate_config(&config_path).map_err(anyhow::Error::msg)?;

    tracing::info!("Configuration loaded successfully");

    if let Some(Command::CompressContext {
        input,
        output,
        level,
        force,
    }) = cli.command
    {
        return compress_context_file(&config, &input, &output, level, force).await;
    }

    tracing::info!("OBEY-API Gateway starting...");

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
        tracing::info!(
            "Server will listen on {}:{}",
            config.server.host,
            config.server.port
        );
        let server = gateway::GatewayServer::new(config, Some(config_path))
            .await
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

async fn compress_context_file(
    config: &config::Config,
    input: &Path,
    output: &Path,
    requested_level: Option<CliCompressionLevel>,
    force: bool,
) -> anyhow::Result<()> {
    use chrono::Utc;
    use compression::{
        config::EffectiveCompressionConfig,
        pipeline::{CompressionPipeline, CompressionRequestMetadata},
        precompressed::{metadata_path_for, write_precompressed_atomic, PrecompressedMetadata},
        CompressiblePayload, CompressionContext, CompressionLevel,
    };
    use models::openai::{Message, OpenAIRequest};

    let input = canonical_regular_input(input)?;
    let output = absolute_output_path(output)?;
    let sidecar = metadata_path_for(&output);
    if paths_refer_to_same_file(&input, &output)? {
        anyhow::bail!(
            "input and output must be different files: `{}`",
            input.display()
        );
    }
    refuse_existing_output(&output, &sidecar, force)?;

    let source_bytes = read_context_file(&input)?;
    let source_text = std::str::from_utf8(&source_bytes)
        .map_err(|_| anyhow::anyhow!("input is not valid UTF-8: `{}`", input.display()))?;
    let level = requested_level
        .map(CompressionLevel::from)
        .unwrap_or(config.compression.default_level);
    if level == CompressionLevel::None {
        anyhow::bail!("compression level `none` cannot be used for compress-context");
    }

    let mut compression_config = config.compression.clone();
    compression_config.enabled = true;
    set_offline_time_budget(&mut compression_config.time_budget_ms);
    let language = compression_config.language.clone();
    let caveman_output = compression_config.caveman_output;
    let pipeline = CompressionPipeline::from_config(compression_config);
    let payload = CompressiblePayload::from_openai_request(OpenAIRequest {
        model: OFFLINE_MODEL.to_owned(),
        messages: vec![Message {
            role: "document".to_owned(),
            content: serde_json::Value::String(source_text.to_owned()),
            extra: serde_json::Map::new(),
        }],
        stream: false,
        temperature: None,
        max_tokens: None,
        extra: serde_json::Map::new(),
    });
    let context = CompressionContext {
        language,
        ..CompressionContext::new(OFFLINE_MODEL, "offline")
    };
    let result = pipeline
        .compress_explicit(
            payload,
            context,
            EffectiveCompressionConfig {
                enabled: true,
                level,
                auto_threshold_tokens: 0,
                caveman_output,
            },
            CompressionRequestMetadata {
                request_id: "offline-compress-context".to_owned(),
                ..CompressionRequestMetadata::default()
            },
        )
        .await;

    if result.timed_out {
        anyhow::bail!(
            "offline compression exceeded the {} ms time budget; no output was written",
            OFFLINE_TIME_BUDGET_MS
        );
    }
    if result.error {
        let summary = result
            .errors
            .iter()
            .map(ToString::to_string)
            .collect::<Vec<_>>()
            .join("; ");
        anyhow::bail!("offline compression failed; no output was written: {summary}");
    }

    let compressed = result
        .payload
        .messages
        .first()
        .and_then(|message| message.content.as_text())
        .ok_or_else(|| anyhow::anyhow!("compression pipeline returned no text artifact"))?;
    let (artifact, compressed_tokens) = if result.final_tokens < result.original_tokens {
        (compressed.as_bytes(), result.final_tokens)
    } else {
        (source_bytes.as_slice(), result.original_tokens)
    };
    let metadata = PrecompressedMetadata::for_source(
        u64::from(result.original_tokens),
        u64::from(compressed_tokens),
        level,
        Utc::now(),
        &source_bytes,
    )?;
    let root = common_canonical_parent(&input, &output)?;
    let sidecar = write_precompressed_atomic(&root, &input, &output, artifact, &metadata).map_err(
        |error| {
            anyhow::anyhow!(
                "failed to atomically write output under common root `{}`: {error}",
                root.display()
            )
        },
    )?;

    let savings_percent = if metadata.original_tokens == 0 {
        0.0
    } else {
        metadata
            .original_tokens
            .saturating_sub(metadata.compressed_tokens) as f64
            * 100.0
            / metadata.original_tokens as f64
    };
    println!("original_tokens: {}", metadata.original_tokens);
    println!("compressed_tokens: {}", metadata.compressed_tokens);
    println!("savings_percent: {savings_percent:.2}");
    println!("target_met: {}", metadata.meets_target_reduction());
    println!("output: {}", output.display());
    println!("sidecar: {}", sidecar.display());
    Ok(())
}

fn set_offline_time_budget(budgets: &mut compression::config::TimeBudgetConfig) {
    budgets.lite = budgets.lite.max(OFFLINE_TIME_BUDGET_MS);
    budgets.standard = budgets.standard.max(OFFLINE_TIME_BUDGET_MS);
    budgets.aggressive = budgets.aggressive.max(OFFLINE_TIME_BUDGET_MS);
    budgets.ultra = budgets.ultra.max(OFFLINE_TIME_BUDGET_MS);
    budgets.rtk = budgets.rtk.max(OFFLINE_TIME_BUDGET_MS);
    budgets.stacked = budgets.stacked.max(OFFLINE_TIME_BUDGET_MS);
}

fn canonical_regular_input(path: &Path) -> anyhow::Result<PathBuf> {
    let metadata = fs::symlink_metadata(path)
        .map_err(|error| anyhow::anyhow!("failed to access input `{}`: {error}", path.display()))?;
    if !metadata.file_type().is_file() {
        anyhow::bail!("input is not a regular file: `{}`", path.display());
    }
    path.canonicalize()
        .map_err(|error| anyhow::anyhow!("failed to resolve input `{}`: {error}", path.display()))
}

fn absolute_output_path(path: &Path) -> anyhow::Result<PathBuf> {
    let file_name = path
        .file_name()
        .ok_or_else(|| anyhow::anyhow!("output must name a file: `{}`", path.display()))?;
    let parent = path
        .parent()
        .filter(|parent| !parent.as_os_str().is_empty())
        .unwrap_or_else(|| Path::new("."));
    let canonical_parent = parent.canonicalize().map_err(|error| {
        anyhow::anyhow!(
            "failed to resolve output parent `{}`: {error}",
            parent.display()
        )
    })?;
    if !canonical_parent.is_dir() {
        anyhow::bail!(
            "output parent is not a directory: `{}`",
            canonical_parent.display()
        );
    }
    Ok(canonical_parent.join(file_name))
}

fn paths_refer_to_same_file(input: &Path, output: &Path) -> anyhow::Result<bool> {
    if input == output {
        return Ok(true);
    }
    match fs::canonicalize(output) {
        Ok(existing) => Ok(existing == input),
        Err(error) if error.kind() == std::io::ErrorKind::NotFound => Ok(false),
        Err(error) => Err(anyhow::anyhow!(
            "failed to resolve output `{}`: {error}",
            output.display()
        )),
    }
}

fn refuse_existing_output(output: &Path, sidecar: &Path, force: bool) -> anyhow::Result<()> {
    for path in [output, sidecar] {
        match fs::symlink_metadata(path) {
            Ok(metadata) if !metadata.file_type().is_file() => {
                anyhow::bail!("output target is not a regular file: `{}`", path.display())
            }
            Ok(_) if !force => anyhow::bail!(
                "output already exists: `{}`; pass --force to replace it",
                path.display()
            ),
            Ok(_) => {}
            Err(error) if error.kind() == std::io::ErrorKind::NotFound => {}
            Err(error) => anyhow::bail!("failed to access output `{}`: {error}", path.display()),
        }
    }
    Ok(())
}

fn read_context_file(path: &Path) -> anyhow::Result<Vec<u8>> {
    let metadata = fs::metadata(path).map_err(|error| {
        anyhow::anyhow!("failed to inspect input `{}`: {error}", path.display())
    })?;
    if metadata.len() > MAX_CONTEXT_BYTES {
        anyhow::bail!(
            "input `{}` is too large ({} bytes; maximum {} bytes)",
            path.display(),
            metadata.len(),
            MAX_CONTEXT_BYTES
        );
    }
    let mut bytes = Vec::with_capacity(metadata.len() as usize);
    File::open(path)
        .and_then(|file| file.take(MAX_CONTEXT_BYTES + 1).read_to_end(&mut bytes))
        .map_err(|error| anyhow::anyhow!("failed to read input `{}`: {error}", path.display()))?;
    if bytes.len() as u64 > MAX_CONTEXT_BYTES {
        anyhow::bail!(
            "input `{}` exceeded the {} byte limit",
            path.display(),
            MAX_CONTEXT_BYTES
        );
    }
    Ok(bytes)
}

fn common_canonical_parent(input: &Path, output: &Path) -> anyhow::Result<PathBuf> {
    let input_parent = input
        .parent()
        .ok_or_else(|| anyhow::anyhow!("input has no parent: `{}`", input.display()))?;
    let output_parent = output
        .parent()
        .ok_or_else(|| anyhow::anyhow!("output has no parent: `{}`", output.display()))?;
    let mut common = PathBuf::new();
    for (left, right) in input_parent.components().zip(output_parent.components()) {
        if left != right {
            break;
        }
        common.push(left.as_os_str());
    }
    if common.as_os_str().is_empty() || !common.is_dir() {
        anyhow::bail!(
            "input `{}` and output `{}` do not share a directory root required by the atomic writer",
            input.display(),
            output.display()
        );
    }
    common.canonicalize().map_err(|error| {
        anyhow::anyhow!(
            "failed to resolve common root `{}`: {error}",
            common.display()
        )
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use compression::precompressed::{metadata_path_for, PrecompressedMetadata};
    use tempfile::TempDir;

    #[test]
    fn cli_preserves_legacy_config_flag() {
        let cli = Cli::try_parse_from(["ai-gateway", "--config", "custom.yaml"]).unwrap();
        assert_eq!(cli.config, Some(PathBuf::from("custom.yaml")));
        assert!(cli.command.is_none());
    }

    #[test]
    fn cli_parses_compress_context_options() {
        let cli = Cli::try_parse_from([
            "ai-gateway",
            "--config",
            "custom.yaml",
            "compress-context",
            "input.md",
            "output.md",
            "--level",
            "standard",
            "--force",
        ])
        .unwrap();
        assert_eq!(cli.config, Some(PathBuf::from("custom.yaml")));
        assert!(matches!(
            cli.command,
            Some(Command::CompressContext {
                input,
                output,
                level: Some(CliCompressionLevel::Standard),
                force: true,
            }) if input == PathBuf::from("input.md") && output == PathBuf::from("output.md")
        ));
    }

    #[tokio::test]
    async fn compress_context_writes_safe_artifact_and_refuses_overwrite() {
        let directory = TempDir::new().unwrap();
        let input = directory.path().join("context.md");
        let output = directory.path().join("context.compressed.md");
        let protected = concat!(
            "```rust\nlet token = \"KEEP_CODE\";\n```\n",
            "https://example.com/KEEP_URL?q=1\n",
            "Path /opt/obey/KEEP_PATH/config.json and C:\\obey\\KEEP_WINDOWS\\config.json.\n",
            "Structured {\"KEEP_JSON\":[1,2,{\"exact\":true}]} data.\n"
        );
        let prose = "Please, in order to actually make use of this context, keep in mind that it is important to note that the following information is very useful. ";
        let source = format!("{protected}{}", prose.repeat(40));
        fs::write(&input, source.as_bytes()).unwrap();

        let mut config = test_config();
        config.compression.default_level = compression::CompressionLevel::Standard;
        compress_context_file(&config, &input, &output, None, false)
            .await
            .unwrap();

        let artifact = fs::read_to_string(&output).unwrap();
        for exact in [
            "```rust\nlet token = \"KEEP_CODE\";\n```",
            "https://example.com/KEEP_URL?q=1",
            "/opt/obey/KEEP_PATH/config.json",
            "C:\\obey\\KEEP_WINDOWS\\config.json",
            "{\"KEEP_JSON\":[1,2,{\"exact\":true}]}",
        ] {
            assert!(artifact.contains(exact), "missing protected bytes: {exact}");
        }
        let sidecar = metadata_path_for(&output);
        let metadata: PrecompressedMetadata =
            serde_json::from_slice(&fs::read(&sidecar).unwrap()).unwrap();
        assert_eq!(metadata.level, compression::CompressionLevel::Standard);
        assert!(metadata.compressed_tokens <= metadata.original_tokens);

        let original_artifact = fs::read(&output).unwrap();
        assert!(compress_context_file(&config, &input, &output, None, false)
            .await
            .is_err());
        assert_eq!(fs::read(&output).unwrap(), original_artifact);
        assert!(compress_context_file(&config, &input, &input, None, true)
            .await
            .is_err());
    }

    fn test_config() -> config::Config {
        serde_yaml::from_str(include_str!("../config.example.yaml")).unwrap()
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

    tracing::info!(
        "Server will listen on {}:{}",
        config.server.host,
        config.server.port
    );

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
            Err(error) => {
                tracing::error!(%error, "Tray-mode gateway server task exited with an error")
            }
        }

        result
    });

    tray_manager.set_server_running(true).await;
    tracing::info!(
        "Tray manager marked the server as running while the HTTP server task is still starting"
    );

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
        config::save_config(&config_path, &persisted).map_err(|error| anyhow::anyhow!(error))?;
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
